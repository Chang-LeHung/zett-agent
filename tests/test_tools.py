import json
from typing import Annotated

import pytest
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, Field, ValidationError

from zett_agent.tools.base import (
    AgentTool,
    get_tool_guidelines,
    get_tool_snippet,
    render_tool_guidance,
    render_tool_search_text,
    tool,
)


class Item(BaseModel):
    amount: int = Field(gt=0)


class Order(BaseModel):
    items: list[Item]


class TreeNode(BaseModel):
    label: str = "node"
    children: list["TreeNode"] = []


@tool(guidelines="Sum a validated order.")
def total(order: Order) -> int:
    """Sum the item amounts."""
    return sum(item.amount for item in order.items)


async def test_nested_models_are_validated_before_execution():
    assert await total({"order": {"items": [{"amount": 2}, {"amount": 3}]}}) == 5
    with pytest.raises(ValidationError):
        await total({"order": {"items": [{"amount": -1}]}})
    with pytest.raises(ValidationError):
        await total({"order": {"items": []}, "unexpected": True})


def test_nested_models_export_a_self_contained_schema():
    schema = total.parameters
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    order = schema["properties"]["order"]
    # pydantic titles every model and property with its own name, which only repeats
    # the argument key the model already sees.
    assert "title" not in schema
    assert "title" not in order
    assert set(order["properties"]) == {"items"}
    assert set(order["properties"]["items"]["items"]["properties"]) == {"amount"}
    assert order["properties"]["items"]["items"]["properties"]["amount"]["exclusiveMinimum"] == 0


def test_recursive_argument_models_are_rejected_because_refs_cannot_be_expanded():
    with pytest.raises(ValueError, match="recursive model"):

        @tool(guidelines="Walk a recursive tree.")
        def walk(tree: TreeNode) -> int:
            """Count nodes in a tree."""
            return 0


async def test_async_tool_defaults_and_field_descriptions():
    @tool
    async def greet(name: Annotated[str, Field(description="Person to greet")], suffix: str = "!") -> str:
        """Greet a person.

        Guidelines:
            - Use when the user asks for a greeting.
        """
        return name + suffix

    assert await greet({"name": "Ada"}) == "Ada!"
    assert greet.parameters["properties"]["name"]["description"] == "Person to greet"
    assert greet.description == "Greet a person."


def test_tool_definition_rejects_missing_annotations():
    with pytest.raises(ValueError, match="annotation"):

        @tool
        def invalid(value):
            """Invalid tool."""
            return value


def test_tool_definition_rejects_unsupported_parameters_and_missing_description():
    with pytest.raises(ValueError, match="named parameters"):

        @tool
        def variadic(*values: int) -> int:
            """Add values."""
            return sum(values)

    with pytest.raises(ValueError, match="docstring description"):

        @tool
        def undocumented(value: int) -> int:
            return value


def test_tool_definition_requires_guidelines():
    with pytest.raises(ValueError, match="at least one non-empty guideline"):

        @tool
        def undocumented_guidelines(value: int) -> int:
            """Return one value."""
            return value


def test_agent_tool_may_share_guidance_elsewhere_but_not_hold_blank_guidelines():
    from zett_agent.tools.base import AgentTool, render_tool_guidance

    async def handler(**_: object) -> str:
        return ""

    bare = AgentTool(name="bare", description="No per-tool rules.", parameters={}, handler=handler, guidelines=())
    assert render_tool_guidance([bare]) == ""
    with pytest.raises(ValueError, match="cannot be blank"):
        AgentTool(name="blank", description="Blank rule.", parameters={}, handler=handler, guidelines=(" ",))


def test_result_serialization_supports_models():
    assert total.serialize_result(Item(amount=2)) == '{"amount": 2}'
    assert total.serialize_result("plain text") == "plain text"


def test_deferred_tool_propagates_to_definition():
    @tool(deferred=True, guidelines="Use only when the optional capability is needed.")
    def later(value: int) -> int:
        """Process one optional value."""
        return value

    assert later.deferred is True
    assert later.definition.deferred is True


def test_local_tool_search_tool_propagates_to_definition():
    @tool(local_tool_search=True, guidelines="Use to find optional tools.")
    def find_tools(query: str) -> list[FunctionToolParam]:
        """Return the tools that match one query.

        Args:
            query: What the model wants to do.
        """
        return []

    assert find_tools.local_tool_search is True
    assert find_tools.definition.local_tool_search is True


def test_local_tool_search_tool_cannot_defer_its_own_schema():
    with pytest.raises(ValueError, match="cannot also defer"):

        @tool(local_tool_search=True, deferred=True, guidelines="Use to find optional tools.")
        def find_tools(query: str) -> list[FunctionToolParam]:
            """Return the tools that match one query.

            Args:
                query: What the model wants to do.
            """
            return []


@pytest.mark.parametrize("keyword", ["local_tool_search"])
def test_tool_flags_must_be_boolean(keyword):
    with pytest.raises(ValueError, match=f"{keyword} must be a boolean"):
        AgentTool(
            name="flag",
            description="Flagged tool",
            parameters={"type": "object"},
            handler=lambda: None,
            guidelines=("Use for flag tests.",),
            **{keyword: "yes"},
        )


def test_a_return_schema_must_be_an_object_or_none():
    with pytest.raises(ValueError, match="return_schema must be a JSON schema object"):
        AgentTool(
            name="flag",
            description="Flagged tool",
            parameters={"type": "object"},
            handler=lambda: None,
            guidelines=("Use for flag tests.",),
            return_schema="integer",  # type: ignore[arg-type]  # the value a caller might pass
        )


def test_tool_docstring_supplies_description_args_snippet_and_guidelines():
    def search_knowledge(query: str, limit: int = 10) -> str:
        """Search stored knowledge.

        Args:
            query: Text to search for across
                titles and content.
            limit: Maximum number of results.

        Snippet:
            documented(query="agents", limit=5)

        Guidelines:
            - Use a narrow query first.
            - Increase limit only when needed.
        """
        return query

    assert get_tool_snippet(search_knowledge) == 'documented(query="agents", limit=5)'
    assert get_tool_guidelines(search_knowledge) == (
        "Use a narrow query first.",
        "Increase limit only when needed.",
    )

    documented = tool(search_knowledge)
    assert documented.description == "Search stored knowledge."
    assert (
        documented.parameters["properties"]["query"]["description"] == "Text to search for across titles and content."
    )
    assert documented.parameters["properties"]["limit"]["description"] == "Maximum number of results."
    assert documented.snippet == 'documented(query="agents", limit=5)'
    assert documented.guidelines == ("Use a narrow query first.", "Increase limit only when needed.")

    prompt = render_tool_guidance([documented])
    assert prompt.index("# Tool snippets") < prompt.index("# Tool guidelines")
    assert documented.snippet in prompt
    assert "# Tool guidelines\n## search_knowledge" in prompt
    assert "- Use a narrow query first." in prompt
    assert "- Increase limit only when needed." in prompt
    assert prompt.count("## search_knowledge") == 1


def test_tool_prompt_metadata_helpers_read_an_undecorated_function():
    def inspect_value(value: int) -> int:
        """Inspect one value.

        Snippet:
            inspect_value(value=1)

        Guidelines:
            - Inspect first.
            - Then report.
        """
        return value

    assert get_tool_snippet(inspect_value) == "inspect_value(value=1)"
    assert get_tool_guidelines(inspect_value) == ("Inspect first.", "Then report.")

    registered = tool(inspect_value)
    assert render_tool_guidance([]) == ""
    assert render_tool_guidance([registered]) == (
        "# Tool snippets\n"
        "- inspect_value: inspect_value(value=1)\n\n"
        "# Tool guidelines\n"
        "## inspect_value\n"
        "- Inspect first.\n"
        "- Then report."
    )


def test_tool_prompt_metadata_helpers_return_empty_values_for_missing_sections():
    def plain(value: int) -> int:
        """Return one value."""
        return value

    assert get_tool_snippet(plain) == ""
    assert get_tool_guidelines(plain) == ()


def test_render_tool_guidance_indents_multiline_snippets():
    def create_artifact(content: str) -> str:
        """Create one artifact.

        Args:
            content: Serialized artifact content.

        Snippet:
            create_artifact(content={"artifact_type": "card", "title": "One"})
            create_artifact(content={"artifact_type": "slides", "title": "Two", "content": "# Topic"})

        Guidelines:
            - Create only when useful.
        """
        return content

    prompt = render_tool_guidance([tool(create_artifact)])

    assert (
        "- create_artifact:\n"
        '  create_artifact(content={"artifact_type": "card", "title": "One"})\n'
        '  create_artifact(content={"artifact_type": "slides", "title": "Two", "content": "# Topic"})'
    ) in prompt


def test_render_tool_search_text_collects_every_model_facing_field():
    @tool(guidelines=("Inspect the depot inventory.", "Report exact pallet counts."))
    def depot_report(site: str) -> str:
        """Return a report for one depot.

        Args:
            site: Two-letter depot code.

        Snippet:
            depot_report(site="sh")
        """
        return site

    text = render_tool_search_text(depot_report)

    assert "depot report" in text
    assert "Return a report for one depot." in text
    assert "Two-letter depot code." in text
    assert 'depot_report(site="sh")' in text
    assert "Inspect the depot inventory." in text
    assert "Report exact pallet counts." in text


def test_tool_docstring_rejects_unknown_argument_documentation():
    with pytest.raises(ValueError, match="unknown parameters: missing"):

        @tool
        def invalid(value: str) -> str:
            """Return a value.

            Args:
                missing: This name is not in the function signature.
            """
            return value


def test_decorator_metadata_can_explicitly_override_docstring_metadata():
    @tool(name="lookup", snippet='lookup(query="zett")', guidelines=["First rule.", "Second rule."])
    def search(query: str) -> str:
        """Search content."""
        return query

    assert search.name == "lookup"
    assert search.snippet == 'lookup(query="zett")'
    assert search.guidelines == ("First rule.", "Second rule.")


def test_a_return_annotation_becomes_a_schema():
    @tool(guidelines="Return numbers.")
    def numbers() -> list[int]:
        """Return a list."""
        return []

    @tool(guidelines="Return an order.")
    def order() -> Order:
        """Return an order."""
        return Order(items=[])

    assert numbers.return_schema == {"items": {"type": "integer"}, "type": "array"}
    assert order.return_schema is not None
    # A nested model is inlined, because the schema is pasted into a prompt rather than
    # sent to a provider that could resolve ``$defs``.
    rendered = json.dumps(order.return_schema)
    assert "$defs" not in rendered and "$ref" not in rendered
    assert order.return_schema["properties"]["items"]["type"] == "array"
    # Inlining must keep constraints, not only replace references with objects.
    amount = order.return_schema["properties"]["items"]["items"]["properties"]["amount"]
    assert amount["exclusiveMinimum"] == 0
    # pydantic adds a ``title`` for every class; it costs tokens and says nothing.
    assert "title" not in order.return_schema
    assert "title" not in order.return_schema["properties"]["items"]["items"]


def test_a_returned_model_keeps_a_field_named_title():
    class Card(BaseModel):
        title: str
        body: str

    @tool(guidelines="Return a card.")
    def card() -> Card:
        """Return a card."""
        return Card(title="One", body="Two")

    assert card.return_schema is not None
    assert set(card.return_schema["properties"]) == {"title", "body"}
    assert card.return_schema["properties"]["title"] == {"type": "string"}
    # The ``title`` annotation that named the model itself is still gone.
    assert "title" not in card.return_schema


def test_a_tool_without_a_return_annotation_says_nothing_about_its_result():
    from typing import Any

    @tool(guidelines="Return an unannotated value.")
    def unannotated():
        """Return without a hint."""
        return 1

    @tool(guidelines="Return nothing.")
    def nothing() -> None:
        """Return nothing at all."""

    @tool(guidelines="Return anything.")
    def anything() -> Any:
        """Return any value."""
        return 1

    assert unannotated.return_schema is None
    assert nothing.return_schema == {"type": "null"}
    # An empty schema describes nothing, and a line saying so only costs tokens.
    assert anything.return_schema is None


def test_a_return_annotation_pydantic_cannot_describe_is_just_no_hint():
    @tool(guidelines="Return a class.")
    def kind() -> type:
        """Return a class object."""
        return int

    # ``type`` has no JSON Schema; the tool must still exist, with no schema to show.
    assert kind.return_schema is None
