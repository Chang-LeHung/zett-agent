from io import StringIO

from zettcode.tui_framework import Canvas, DifferentialRenderer, Style


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
