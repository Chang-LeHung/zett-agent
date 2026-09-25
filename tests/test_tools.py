from typing import Annotated

import pytest
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, Field, ValidationError

from zett_agent import (
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


def test_pydantic_exports_resolvable_nested_schemas():
    schema = total.parameters
    assert schema["properties"]["order"]["$ref"] == "#/$defs/Order"
    assert schema["$defs"]["Order"]["properties"]["items"]["items"]["$ref"] == "#/$defs/Item"
    assert schema["$defs"]["Item"]["properties"]["amount"]["exclusiveMinimum"] == 0


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
