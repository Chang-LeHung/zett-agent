import runpy
from pathlib import Path

from zett_agent.events import (
    AgentEvent,
    AgentEventType,
)


def test_each_compaction_cycle_prints_its_own_reasoning_and_summary_labels(capsys):
    example = runpy.run_path(Path(__file__).parents[1] / "examples" / "coding_agent.py")
    print_event = example["print_event"]
    output_state = {
        "compaction_reasoning": False,
        "compaction_summary": False,
        "reasoning": False,
        "answer": False,
    }

    for _ in range(2):
        print_event(AgentEvent(AgentEventType.COMPACTION_STARTED, "session"), output_state)
        print_event(
            AgentEvent(AgentEventType.COMPACTION_REASONING_DELTA, "session", delta="thinking"),
            output_state,
        )
        print_event(
            AgentEvent(AgentEventType.COMPACTION_TEXT_DELTA, "session", delta="summary"),
            output_state,
        )

    output = capsys.readouterr().out
    assert output.count("[compaction] started") == 2
    assert output.count("[compaction reasoning]") == 2
    assert output.count("[compaction summary]") == 2
