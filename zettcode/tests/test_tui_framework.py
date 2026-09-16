from io import StringIO

from zettcode.tui_framework import Canvas, DifferentialRenderer, EventType, InputEvent, MouseAction, Style
from zettcode.tui_framework.input import InputDecoder


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
