from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from kcs_agent import tool


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
        """Greet a person."""
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


def test_result_serialization_supports_models():
    assert total.serialize_result(Item(amount=2)) == '{"amount": 2}'
