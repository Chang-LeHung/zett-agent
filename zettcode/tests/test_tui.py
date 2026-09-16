from contextlib import aclosing

from wcwidth import wcwidth
from zett_agent import (
    AgentEvent,
    AgentEventType,
    AssistantMessage,
    ModelEvent,
    ModelResponse,
    ServerToolCall,
    ToolCall,
    ToolMessage,
    create_agent,
)

from zettcode.tui import LineMetadata, Transcript, TUIEventDispatcher, _tool_preview


async def test_dispatcher_preserves_stream_order_and_parallel_tool_names():
    transcript = Transcript()
    dispatcher = TUIEventDispatcher(transcript)
    dispatcher.begin_turn("inspect the project")
    await dispatcher.dispatch(AgentEvent(AgentEventType.REASONING_DELTA, "session", delta="checking"))
    await dispatcher.dispatch(
        AgentEvent(
            AgentEventType.TOOL_STARTED,
            "session",
            tool_calls=[ToolCall("1", "glob"), ToolCall("2", "grep")],
        )
    )
    await dispatcher.dispatch(AgentEvent(AgentEventType.TEXT_DELTA, "session", delta="done"))

    assert "checking" not in transcript.text
    assert "glob" in transcript.text
    assert "grep" in transcript.text
    assert transcript.toggle_latest_thinking()
    assert transcript.text.index("inspect the project") < transcript.text.index("checking")
    assert transcript.text.index("checking") < transcript.text.index("done")


async def test_agent_client_dispatches_events_into_tui_transcript():
    class Model:
        async def stream(self, request):
            yield ModelEvent.reasoning("checking")
            yield ModelEvent.text("done")
            yield ModelEvent.completed(ModelResponse(AssistantMessage(content="done")))

    transcript = Transcript()
    dispatcher = TUIEventDispatcher(transcript)
    dispatcher.begin_turn("inspect the project")
    client = await create_agent(Model(), extensions=[], event_dispatcher=dispatcher)

    async with aclosing(client.stream("inspect the project")) as events:
        emitted = [event async for event in events]

    assert emitted[-1].type == AgentEventType.RUN_COMPLETED
    assert "checking" not in transcript.text
    assert transcript.toggle_latest_thinking()
    assert transcript.text.index("inspect the project") < transcript.text.index("checking")
    assert transcript.text.index("checking") < transcript.text.index("done")


async def test_tool_output_keeps_its_line_structure_visible():
    transcript = Transcript()
    dispatcher = TUIEventDispatcher(transcript)
    call = ToolCall("call-1", "read_file", {"path": "README.md"})

    await dispatcher.dispatch(AgentEvent(AgentEventType.TOOL_STARTED, "session", tool_calls=[call]))
    await dispatcher.dispatch(
        AgentEvent(
            AgentEventType.TOOL_COMPLETED,
            "session",
            tool_calls=[call],
            message=ToolMessage(tool_call_id=call.id, name=call.name, content="first line\nsecond line"),
        )
    )

    assert "✓ read_file" in transcript.text
    assert "└ first line" in transcript.text
    assert "second line" in transcript.text


async def test_skipped_tool_does_not_remain_running():
    transcript = Transcript()
    dispatcher = TUIEventDispatcher(transcript)
    call = ToolCall("call-1", "write_file")

    await dispatcher.dispatch(AgentEvent(AgentEventType.TOOL_STARTED, "session", tool_calls=[call]))
    await dispatcher.dispatch(
        AgentEvent(
            AgentEventType.TOOL_SKIPPED,
            "session",
            tool_calls=[call],
            message=ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content="Skipped because a newer user message took priority",
                success=False,
            ),
        )
    )

    assert "– write_file" in transcript.text
    assert "Running…" not in transcript.text


def test_tool_preview_is_single_line_and_bounded():
    message = ToolMessage(tool_call_id="1", name="read_file", content="line one\n" + "x" * 300)

    preview = _tool_preview(message, limit=40)

    assert "\n" not in preview
    assert len(preview) == 40
    assert preview.endswith("…")


def test_long_tool_output_is_bounded_and_can_expand_safely():
    transcript = Transcript()
    transcript.start_tool("call", "run_shell", {})
    transcript.complete_tool("call", "\n".join(f"line {index}" for index in range(100)))

    collapsed = transcript.lines(80)
    collapsed_text = "\n".join(line.text for line in collapsed)
    header = next(line for line in collapsed if "run_shell" in line.text)

    assert "line 4" in collapsed_text
    assert "line 5" not in collapsed_text
    assert "95 more rows" in collapsed_text
    assert isinstance(header.metadata, LineMetadata)
    assert header.metadata.tool_id is not None

    assert transcript.toggle_tool(header.metadata.tool_id)
    expanded_text = "\n".join(line.text for line in transcript.lines(80))
    assert "line 39" in expanded_text
    assert "line 40" not in expanded_text
    assert "60 more rows" in expanded_text


def test_tool_output_strips_ansi_and_pulses_only_while_running():
    transcript = Transcript()
    transcript.start_tool("call", "run_shell", {})
    transcript.animation_frame = 0
    first = next(line for line in transcript.lines(80) if "run_shell" in line.text)
    transcript.animation_frame = 6
    second = next(line for line in transcript.lines(80) if "run_shell" in line.text)

    assert first.text != second.text or first.spans[0].style != second.spans[0].style

    transcript.complete_tool("call", "\x1b[31mdanger\x1b[0m")
    rendered = "\n".join(line.text for line in transcript.lines(80))
    assert "danger" in rendered
    assert "\x1b" not in rendered


def test_running_tool_uses_a_moving_character_highlight_wave():
    transcript = Transcript()
    transcript.start_tool("call", "run_shell", {})
    transcript.animation_frame = 0
    first = next(line for line in transcript.lines(80) if "run_shell" in line.text)
    transcript.animation_frame = 4
    second = next(line for line in transcript.lines(80) if "run_shell" in line.text)

    first_colors = [span.style.foreground for span in first.spans]
    second_colors = [span.style.foreground for span in second.spans]
    assert len(set(first_colors)) > 3
    assert first_colors != second_colors
    assert first.text != second.text


def test_repeated_tool_completion_keeps_updating_the_same_entry():
    transcript = Transcript()
    transcript.start_tool("call-1", "read_file", {})
    transcript.complete_tool("call-1", "first")

    assert transcript.entries[0].call_id == "call-1"

    transcript.complete_tool("call-1", "second")

    assert len(transcript.entries) == 1
    assert transcript.entries[0].text == "second"


async def test_dispatcher_uses_server_tool_name():
    transcript = Transcript()
    dispatcher = TUIEventDispatcher(transcript)

    await dispatcher.dispatch(
        AgentEvent(
            AgentEventType.SERVER_TOOL_STARTED,
            "session",
            server_tool_call=ServerToolCall("call", "web_search"),
        )
    )

    assert "web_search" in transcript.text


def test_thinking_click_expands_and_pointer_leave_collapses():
    transcript = Transcript()
    transcript.append_thinking("private reasoning")
    header = next(line for line in transcript.lines(80) if "Thinking" in line.text)

    assert isinstance(header.metadata, LineMetadata)
    assert header.metadata.thinking_header
    assert header.metadata.thinking_id is not None
    transcript.toggle_thinking(header.metadata.thinking_id)
    assert "private reasoning" in transcript.text

    transcript.collapse_thinking_except(None)
    assert "private reasoning" not in transcript.text


def test_assistant_markdown_renders_semantics_without_control_markers():
    transcript = Transcript()
    transcript.append_answer("# Result\n**fixed**\n- first\n- `second`")

    lines = transcript.lines(80)
    rendered = "\n".join(line.text for line in lines)

    assert "# Result" not in rendered
    assert "**fixed**" not in rendered
    assert "• first" in rendered
    assert "• second" in rendered
    assert any(span.style.bold for line in lines for span in line.spans if span.text == "fixed")


def test_assistant_markdown_renders_a_fitted_unicode_table():
    transcript = Transcript()
    transcript.append_answer(
        "| Name | Count | 说明 |\n| :--- | ---: | :---: |\n| **Alpha** | 12 | 中文内容 |\n| a\\|b | 3 | `x|y` |"
    )

    lines = transcript.lines(42)
    rendered = "\n".join(line.text for line in lines)

    assert "╭" in rendered and "┬" in rendered and "╯" in rendered
    assert "├" in rendered and "┼" in rendered and "┤" in rendered
    assert "╰" in rendered and "┴" in rendered and "╯" in rendered
    assert "a|b" in rendered
    assert "x|y" in rendered
    assert "---:" not in rendered
    assert any(span.style.bold for line in lines for span in line.spans if span.text == "Alpha")
    assert all(sum(max(0, wcwidth(character)) for character in line.text) <= 42 for line in lines)


def test_markdown_table_truncates_cells_and_excess_columns_on_narrow_terminals():
    transcript = Transcript()
    transcript.append_answer(
        "| first | second | third | fourth |\n"
        "| --- | --- | --- | --- |\n"
        "| a very long value that cannot fit | 二号内容非常长 | three | four |"
    )

    lines = transcript.lines(20)
    rendered = "\n".join(line.text for line in lines)

    assert "…" in rendered
    assert all(sum(max(0, wcwidth(character)) for character in line.text) <= 20 for line in lines)
