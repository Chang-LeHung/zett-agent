"""Public extension hook groups remain small, disjoint, and composable."""

from zett_agent import (
    AgentEventHooksMixin,
    AgentExtension,
    AgentModelHooksMixin,
    AgentRunHooksMixin,
    AgentSetupHooksMixin,
    AgentToolHooksMixin,
    MiddlewareHook,
)


def test_agent_extension_composes_each_hook_group_once() -> None:
    assert AgentExtension.__bases__ == (
        MiddlewareHook,
        AgentSetupHooksMixin,
        AgentRunHooksMixin,
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
        AgentModelHooksMixin: {"before_model", "after_model"},
        AgentToolHooksMixin: {"before_tool", "after_tool"},
        AgentEventHooksMixin: {
            "accept",
            "before_model_events",
            "before_tool_events",
            "after_model_events",
            "after_tool_events",
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
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "before_model_events",
        "before_tool_events",
        "after_model_events",
        "after_tool_events",
        "on_event",
        "accept",
        "on_model_request",
        "on_tool_call",
    }
