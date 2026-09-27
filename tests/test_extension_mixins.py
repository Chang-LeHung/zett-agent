"""Public extension hook groups remain small, disjoint, and composable."""

from zett_agent.extensions.base import (
    AgentEventHooksMixin,
    AgentExtension,
    AgentModelHooksMixin,
    AgentRunHooksMixin,
    AgentSetupHooksMixin,
    AgentToolHooksMixin,
    AgentTurnHooksMixin,
    MiddlewareHook,
)


def test_agent_extension_composes_each_hook_group_once() -> None:
    assert AgentExtension.__bases__ == (
        MiddlewareHook,
        AgentSetupHooksMixin,
        AgentRunHooksMixin,
        AgentTurnHooksMixin,
        AgentModelHooksMixin,
        AgentToolHooksMixin,
        AgentEventHooksMixin,
    )


def test_agent_extension_default_priority_is_one_hundred() -> None:
    extension = AgentExtension()

    assert extension.priority == 100


def test_hook_groups_own_disjoint_lifecycle_methods() -> None:
    groups = {
        MiddlewareHook: {"on_model_request", "on_tool_call"},
        AgentSetupHooksMixin: {"on_tool", "on_state", "on_message"},
        AgentRunHooksMixin: {"before_run", "after_run", "on_success", "on_error"},
        AgentTurnHooksMixin: {"before_turn", "after_turn"},
        AgentModelHooksMixin: {"before_model", "after_model"},
        AgentToolHooksMixin: {"before_tool", "after_tool"},
        AgentEventHooksMixin: {
            "accept",
            "on_event",
        },
    }

    observed: set[str] = set()
    for mixin, expected in groups.items():
        methods = {name for name, value in vars(mixin).items() if callable(value) and not name.startswith("__")}
        assert methods == expected
        assert observed.isdisjoint(methods)
        observed.update(methods)

    assert observed == {
        "on_tool",
        "on_state",
        "on_message",
        "before_run",
        "after_run",
        "on_success",
        "on_error",
        "before_turn",
        "after_turn",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "on_event",
        "accept",
        "on_model_request",
        "on_tool_call",
    }
