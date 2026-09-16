from zett_agent import ToolMessage

from zettcode.tui import _tool_preview


def test_tool_preview_is_single_line_and_bounded():
    message = ToolMessage(tool_call_id="1", name="read_file", content="line one\n" + "x" * 300)

    preview = _tool_preview(message, limit=40)

    assert "\n" not in preview
    assert len(preview) == 40
    assert preview.endswith("…")
