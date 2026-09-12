from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from zett_agent import get_tool_guidelines, get_tool_snippet, render_tool_guidance, tool


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
