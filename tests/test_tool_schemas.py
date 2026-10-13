"""Tool argument schemas ship self-contained: no ``$defs`` and no ``$ref``."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, create_model

from zett_agent.tools.base import _inline_schema_references, tool


class Leaf(BaseModel):
    amount: int = Field(gt=0)


class Middle(BaseModel):
    leaves: list[Leaf]
    matrix: list[list[Leaf]] = []
    note: str | None = None
    by_name: dict[str, Leaf] = {}


class Root(BaseModel):
    middles: list[Middle]
    lookup: dict[str, Leaf]
    pair: tuple[Leaf, Leaf]


class Cat(BaseModel):
    kind: Literal["cat"]
    name: str


class Dog(BaseModel):
    kind: Literal["dog"]


Pet = Annotated[Cat | Dog, Field(discriminator="kind")]


class Shelter(BaseModel):
    pet: Pet
    backup: Pet


class Node(BaseModel):
    label: str = "node"
    children: list[Node] = []


class Ping(BaseModel):
    pong: Pong | None = None


class Pong(BaseModel):
    ping: Ping | None = None


@tool(guidelines="Accept deeply nested arguments.")
def deep(root: Root) -> int:
    """Walk nested arguments."""
    return 0


@tool(guidelines="Accept a tagged union.")
def shelter(pet: Shelter) -> int:
    """Accept one tagged pet."""
    return 0


@tool(guidelines="Accept flat arguments.")
def flat(name: str, count: int = 1) -> int:
    """Accept scalars only."""
    return count


def assert_self_contained(schema: dict) -> None:
    text = json.dumps(schema)
    assert "$defs" not in schema
    assert "$ref" not in text
    assert "#/$defs/" not in text


def test_flat_schemas_are_untouched():
    schema = flat.parameters
    assert_self_contained(schema)
    assert schema["properties"]["name"] == {"title": "Name", "type": "string"}
    assert schema["required"] == ["name"]


def test_deeply_nested_models_are_expanded_at_every_position():
    schema = deep.parameters
    assert_self_contained(schema)

    root = schema["properties"]["root"]
    middle = root["properties"]["middles"]["items"]
    leaf = middle["properties"]["leaves"]["items"]
    assert leaf["properties"]["amount"]["exclusiveMinimum"] == 0
    # Nested lists, optional fields and free-form mappings expand too.
    assert middle["properties"]["matrix"]["items"]["items"]["properties"]["amount"]["type"] == "integer"
    assert middle["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert middle["properties"]["by_name"]["additionalProperties"]["properties"]["amount"]["type"] == "integer"
    assert root["properties"]["lookup"]["additionalProperties"]["properties"]["amount"]["type"] == "integer"
    # Tuples become prefixItems and both members expand independently.
    pair = root["properties"]["pair"]
    assert pair["minItems"] == pair["maxItems"] == 2
    assert [member["properties"]["amount"]["type"] for member in pair["prefixItems"]] == ["integer", "integer"]


def test_repeated_definitions_expand_at_each_use_site():
    schema = shelter.parameters
    assert_self_contained(schema)

    shelter_schema = schema["properties"]["pet"]
    pet = shelter_schema["properties"]["pet"]
    assert {choice["properties"]["kind"]["const"] for choice in pet["oneOf"]} == {"cat", "dog"}
    # The same union appears twice and each site carries its own expanded copy.
    assert len(shelter_schema["properties"]["backup"]["oneOf"]) == len(pet["oneOf"])


def test_openapi_discriminator_is_dropped_because_its_mapping_holds_refs():
    assert "discriminator" not in json.dumps(shelter.parameters)


def test_definitions_used_once_are_inlined_without_losing_constraints():
    class Tagged(BaseModel):
        value: int = Field(gt=0)

    model = create_model("ToolInput", first=(Tagged, ...))
    schema = model.model_json_schema()
    assert "$defs" in schema

    inlined = _inline_schema_references(schema)
    assert_self_contained(inlined)
    assert inlined["properties"]["first"]["properties"]["value"]["exclusiveMinimum"] == 0


def test_schema_with_unused_definitions_drops_them():
    schema = {"properties": {"value": {"type": "string"}}, "$defs": {"Unused": {"type": "object"}}}
    assert _inline_schema_references(schema) == {"properties": {"value": {"type": "string"}}}


def test_unknown_reference_forms_are_rejected():
    with pytest.raises(ValueError, match="unsupported schema reference"):
        _inline_schema_references({"properties": {"value": {"$ref": "#/properties/other"}}})


def test_self_recursive_models_are_rejected():
    with pytest.raises(ValueError, match="recursive model 'Node'"):

        @tool(guidelines="Walk a recursive tree.")
        def walk(node: Node) -> int:
            """Count nodes."""
            return 0


def test_mutually_recursive_models_are_rejected():
    with pytest.raises(ValueError, match="recursive model"):

        @tool(guidelines="Follow a cycle.")
        def cycle(start: Ping) -> int:
            """Follow a cycle."""
            return 0
