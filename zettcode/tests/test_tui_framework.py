import builtins
import importlib
import sys
from io import StringIO

import pytest

from zettcode.tui_framework import (
    Canvas,
    DifferentialRenderer,
    EventType,
    InputEvent,
    MouseAction,
    Rect,
    ScrollableText,
    Span,
    Style,
    TextInput,
    TextLine,
)
from zettcode.tui_framework.components import _layout_input
from zettcode.tui_framework.input import InputDecoder


def test_tui_framework_imports_where_termios_is_unavailable(monkeypatch) -> None:
    """ZettCode must load on platforms without raw-mode terminal modules."""
    real_import = builtins.__import__
    saved_modules = {
        name: sys.modules.get(name) for name in ("zettcode.tui_framework.terminal", "zettcode.tui_framework")
    }

    def blocked(name, *args, **kwargs):
        if name in {"termios", "tty"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        for module_name in saved_modules:
            sys.modules.pop(module_name, None)
        module = importlib.import_module("zettcode.tui_framework")
        assert hasattr(module, "Terminal")
    finally:
        for module_name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = saved


def test_terminal_refuses_platforms_without_posix_raw_mode(monkeypatch) -> None:
    from zettcode.tui_framework import terminal as terminal_module

    monkeypatch.setattr(terminal_module, "POSIX", False)

    with pytest.raises(RuntimeError, match="POSIX terminal"):
        with terminal_module.Terminal(input_fd=0, output=StringIO()):
            pass


class Actions:
    def __init__(self):
        self.focused = None
        self.copied = ""
        self.invalidated = False
        self.exited = False
        self.refreshed = False

    def focus(self, component):
        self.focused = component

    def invalidate(self):
        self.invalidated = True

    def copy(self, text):
        self.copied = text

    def exit(self):
        self.exited = True

    def refresh(self):
        self.refreshed = True


def test_input_decoder_handles_fragmented_utf8_keys_mouse_and_paste():
    decoder = InputDecoder()

    assert decoder.feed("你".encode()[:2]) == []
    text_event = decoder.feed("你".encode()[2:])[0]
    key_event = decoder.feed(b"\x1b[5~")[0]
    mouse_event = decoder.feed(b"\x1b[<64;10;8M")[0]
    paste_event = decoder.feed(b"\x1b[200~one\ntwo\x1b[201~")[0]

    assert (text_event.type, text_event.text) == (EventType.TEXT, "你")
    assert key_event.key == "page_up"
    assert (mouse_event.action, mouse_event.x, mouse_event.y) == (MouseAction.SCROLL_UP, 9, 7)
    assert paste_event.text == "one\ntwo"


def test_input_decoder_resolves_standalone_escape_explicitly():
    decoder = InputDecoder()

    assert decoder.feed(b"\x1b") == []
    assert decoder.flush_escape() == InputEvent(EventType.KEY, key="escape")


def test_input_decoder_maps_readline_and_modified_navigation_keys():
    decoder = InputDecoder()

    events = decoder.feed(b"\x01\x05\x0b\x17\x1f\x1b[1;5D\x1b[1;3C\x1b\x7f\x1b[Z")

    assert [event.key for event in events] == [
        "ctrl_a",
        "ctrl_e",
        "ctrl_k",
        "ctrl_w",
        "ctrl_underscore",
        "ctrl_left",
        "alt_right",
        "alt_backspace",
        "backtab",
    ]


def test_differential_renderer_skips_unchanged_rows():
    output = StringIO()
    renderer = DifferentialRenderer(output)
    first = Canvas(8, 2)
    first.draw_text(0, 0, "hello", Style(foreground="#79b88b"))
    renderer.render(first)
    first_frame = output.getvalue()
    output.seek(0)
    output.truncate()

    second = Canvas(8, 2)
    second.draw_text(0, 0, "hello", Style(foreground="#79b88b"))
    second.draw_text(0, 1, "world")
    renderer.render(second)
    diff = output.getvalue()

    assert "\x1b[2J" in first_frame
    assert "\x1b[1;1H" not in diff
    assert "\x1b[2;1H" in diff
    assert "world" in diff


def test_canvas_never_emits_untrusted_terminal_control_characters():
    output = StringIO()
    canvas = Canvas(20, 1)
    canvas.draw_text(0, 0, "safe\x1b[31mred")

    DifferentialRenderer(output).render(canvas)

    rendered = output.getvalue()
    assert "safe\x1b[31mred" not in rendered
    assert "safe�[31mred" in rendered


def test_canvas_and_scrollable_text_expand_tabs_without_replacement_glyphs():
    output = StringIO()
    direct = Canvas(20, 1)
    direct.draw_text(0, 0, "a\tb")
    DifferentialRenderer(output).render(direct)
    assert "a   b" in output.getvalue()
    assert "�" not in output.getvalue()

    view = ScrollableText(lambda width: [TextLine((Span("\tindented\tvalue"),))])
    view.layout(Rect(0, 0, 24, 1))
    canvas = Canvas(24, 1)
    view.render(canvas)
    rendered = "".join(cell.character for cell in canvas.cells[0] if not cell.continuation)
    assert rendered.startswith("    indented    value")
    assert "�" not in rendered


def test_scrollable_text_wheel_moves_viewport_and_drag_selects():
    lines = [TextLine((Span(f"line {index}"),)) for index in range(20)]
    view = ScrollableText(lambda width: lines)
    view.layout(Rect(0, 0, 20, 5))
    view.render(Canvas(20, 5))
    actions = Actions()
    assert view.scroll_top == 15

    view.handle(InputEvent(EventType.MOUSE, x=2, y=2, action=MouseAction.SCROLL_UP), actions)
    assert view.scroll_top == 12
    view.handle(InputEvent(EventType.MOUSE, x=0, y=1, action=MouseAction.DOWN), actions)
    view.handle(InputEvent(EventType.MOUSE, x=6, y=1, action=MouseAction.MOVE), actions)
    view.handle(InputEvent(EventType.MOUSE, x=6, y=1, action=MouseAction.UP), actions)

    assert actions.focused is view
    assert view.selected_text() == "line 1"
    assert actions.copied == "line 1"


def test_scrollable_text_double_click_selects_the_visual_line():
    clicks = iter((10.0, 10.2))
    lines = [TextLine((Span("first"),)), TextLine((Span("entire second line"),))]
    view = ScrollableText(lambda width: lines, clock=lambda: next(clicks))
    view.layout(Rect(0, 0, 30, 4))
    view.render(Canvas(30, 4))
    actions = Actions()

    for _ in range(2):
        view.handle(InputEvent(EventType.MOUSE, x=4, y=1, action=MouseAction.DOWN), actions)
        view.handle(InputEvent(EventType.MOUSE, x=4, y=1, action=MouseAction.UP), actions)

    assert actions.focused is view
    assert view.selected_text() == "entire second line"
    assert actions.copied == "entire second line"


def test_scrollable_text_keyboard_navigation_and_drag_autoscroll():
    lines = [TextLine((Span(f"line {index}"),)) for index in range(20)]
    view = ScrollableText(lambda width: lines)
    view.layout(Rect(0, 0, 20, 5))
    view.render(Canvas(20, 5))
    actions = Actions()
    actions.focus(view)

    view.handle(InputEvent(EventType.KEY, key="home"), actions)
    assert view.scroll_top == 0
    view.handle(InputEvent(EventType.KEY, key="down"), actions)
    assert view.scroll_top == 1

    view.handle(InputEvent(EventType.MOUSE, x=0, y=2, action=MouseAction.DOWN), actions)
    view.handle(InputEvent(EventType.MOUSE, x=4, y=6, action=MouseAction.MOVE), actions)
    assert view.scroll_top == 2
    assert view.selected_text()

    view.handle(InputEvent(EventType.KEY, key="end"), actions)
    assert view.scroll_top == 15


def test_text_input_edits_and_submits_without_external_framework():
    submitted = []
    editor = TextInput(submitted.append)
    actions = Actions()

    editor.handle(InputEvent(EventType.TEXT, text="hello"), actions)
    editor.handle(InputEvent(EventType.KEY, key="alt_enter"), actions)
    editor.handle(InputEvent(EventType.PASTE, text="world"), actions)
    editor.handle(InputEvent(EventType.KEY, key="enter"), actions)

    assert submitted == ["hello\nworld"]
    assert editor.text == ""


def test_rejected_submission_preserves_editor_text():
    editor = TextInput(lambda value: False)
    actions = Actions()
    editor.handle(InputEvent(EventType.TEXT, text="keep me"), actions)

    editor.handle(InputEvent(EventType.KEY, key="enter"), actions)

    assert editor.text == "keep me"


def test_editor_supports_line_word_kill_yank_and_undo_shortcuts():
    editor = TextInput(lambda value: True)
    actions = Actions()
    editor.handle(InputEvent(EventType.TEXT, text="alpha beta\ngamma delta"), actions)

    editor.handle(InputEvent(EventType.KEY, key="ctrl_a"), actions)
    assert editor.position == len("alpha beta\n")
    editor.handle(InputEvent(EventType.KEY, key="ctrl_k"), actions)
    assert editor.text == "alpha beta\n"
    editor.handle(InputEvent(EventType.KEY, key="ctrl_y"), actions)
    assert editor.text == "alpha beta\ngamma delta"
    editor.handle(InputEvent(EventType.KEY, key="ctrl_z"), actions)
    assert editor.text == "alpha beta\n"

    editor.handle(InputEvent(EventType.KEY, key="ctrl_home"), actions)
    editor.handle(InputEvent(EventType.KEY, key="ctrl_right"), actions)
    assert editor.position == len("alpha")
    editor.handle(InputEvent(EventType.KEY, key="d", alt=True), actions)
    assert editor.text == "alpha\n"


def test_editor_navigates_multiline_input_and_submission_history():
    submitted = []
    editor = TextInput(submitted.append)
    actions = Actions()
    editor.handle(InputEvent(EventType.TEXT, text="one\ntwo"), actions)

    editor.handle(InputEvent(EventType.KEY, key="up"), actions)
    assert editor.position == 3
    editor.handle(InputEvent(EventType.KEY, key="down"), actions)
    assert editor.position == len(editor.text)
    editor.handle(InputEvent(EventType.KEY, key="enter"), actions)
    editor.handle(InputEvent(EventType.TEXT, text="second command"), actions)
    editor.handle(InputEvent(EventType.KEY, key="enter"), actions)

    editor.handle(InputEvent(EventType.KEY, key="up"), actions)
    assert editor.text == "second command"
    editor.handle(InputEvent(EventType.KEY, key="up"), actions)
    assert editor.text == "one\ntwo"
    editor.handle(InputEvent(EventType.KEY, key="down"), actions)
    assert editor.text == "second command"
    editor.handle(InputEvent(EventType.KEY, key="down"), actions)
    assert editor.text == ""


def test_editor_completes_commands_in_both_directions():
    editor = TextInput(lambda value: True, completions=("/clear", "/close", "/quit"))
    actions = Actions()
    editor.handle(InputEvent(EventType.TEXT, text="/cl"), actions)

    editor.handle(InputEvent(EventType.KEY, key="tab"), actions)
    assert editor.text == "/clear"
    editor.handle(InputEvent(EventType.KEY, key="tab"), actions)
    assert editor.text == "/close"
    editor.handle(InputEvent(EventType.KEY, key="backtab"), actions)
    assert editor.text == "/clear"


def test_editor_wraps_cursor_at_the_terminal_edge():
    lines, cursor = _layout_input("123", 3, 5, 2)

    assert lines == ["123", ""]
    assert cursor == (1, 0)


def test_editor_expands_pasted_tabs_before_cursor_layout():
    editor = TextInput(lambda value: True)
    actions = Actions()

    editor.handle(InputEvent(EventType.PASTE, text="\tgo\n\treturn"), actions)

    assert editor.text == "    go\n    return"
    assert editor.position == len(editor.text)
