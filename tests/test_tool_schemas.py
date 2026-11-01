"""Tool argument schemas ship self-contained: no ``$defs`` and no ``$ref``."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, create_model

from zett_agent.tools.base import _inline_schema_references, _strip_schema_keywords, tool


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


def test_a_flat_schema_keeps_its_properties_and_loses_pydantic_titles():
    schema = flat.parameters
    assert_self_contained(schema)
    # The only thing removed is pydantic's auto title, which repeats the key.
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["required"] == ["name"]
    # The model's own ``title`` goes too, while the object keywords stay.
    assert "title" not in schema
    assert schema["additionalProperties"] is False


class Chapter(BaseModel):
    title: str
    pages: int
    summary: str | None = None


@tool(guidelines="Accept a title argument.")
def titled(title: str, body: str) -> int:
    """Store a title."""
    return 0


@tool(guidelines="Accept a chapter.")
def chapter(chapter: Chapter) -> int:
    """Accept one chapter."""
    return 0


@tool(guidelines="Accept a documented title.")
def documented_title(title: str) -> int:
    """Store a title.

    Args:
        title: The name to show.
    """
    return 0


def test_a_property_named_title_is_kept_as_a_name():
    schema = titled.parameters
    assert set(schema["properties"]) == {"title", "body"}
    assert schema["properties"]["title"] == {"type": "string"}
    assert schema["required"] == ["title", "body"]


def test_a_documented_property_named_title_keeps_its_description():
    # The docstring merge addresses one property by name; a name-aware strip is what
    # keeps it from raising ``KeyError`` here.
    assert documented_title.parameters["properties"]["title"] == {
        "type": "string",
        "description": "The name to show.",
    }


def test_a_nested_model_keeps_a_field_named_title():
    chapter_schema = chapter.parameters["properties"]["chapter"]
    assert set(chapter_schema["properties"]) == {"title", "pages", "summary"}
    assert chapter_schema["required"] == ["title", "pages"]
    assert chapter_schema["properties"]["title"] == {"type": "string"}
    assert "title" not in chapter_schema


def test_an_explicit_field_title_is_dropped_and_the_description_keeps_the_meaning():
    @tool(guidelines="Accept a titled argument.")
    def explicit(name: str = Field(title="Display name", description="The name to show.")) -> int:
        """Store a name."""
        return 0

    # After ``model_json_schema`` an author's ``title`` and pydantic's automatic one
    # look alike, so the runtime drops both and the description carries the meaning.
    assert explicit.parameters["properties"]["name"] == {"type": "string", "description": "The name to show."}


def test_union_and_enum_branches_lose_only_the_title_annotation():
    @tool(guidelines="Accept a choice.")
    def chosen(kind: Literal["a", "b"], extra: str | None = None) -> int:
        """Pick one."""
        return 0

    kind = chosen.parameters["properties"]["kind"]
    assert kind == {"enum": ["a", "b"], "type": "string"}
    # A branch of ``anyOf`` is a schema, so the strip reaches inside it too.
    extra = chosen.parameters["properties"]["extra"]
    assert extra == {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}


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


def test_the_strip_walks_applicator_schemas_and_leaves_names_and_data_alone():
    schema = {
        "title": "ToolInput",
        "properties": {"title": {"title": "Title", "type": "string"}},
        "$defs": {"Title": {"title": "Title", "type": "object"}},
        "dependentSchemas": {"title": {"title": "Title", "required": ["title"]}},
        "dependencies": {"title": {"title": "Title"}, "other": ["a", "b"]},
        "items": {"title": "Item", "type": "integer"},
        # ``additionalItems`` is a single schema: only ``items`` may hold a list in
        # draft-07, and the case at the end of this test covers that list form.
        "additionalItems": {"title": "Extra", "type": "boolean"},
        "prefixItems": [{"title": "First", "type": "string"}, {"title": "Second", "type": "integer"}],
        "anyOf": [{"title": "A", "type": "string"}],
        "allOf": [{"title": "B", "type": "integer"}],
        "propertyNames": {"title": "Name", "pattern": "^[a-z]+$"},
        "if": {"title": "If", "required": ["a"]},
        "then": {"title": "Then", "required": ["b"]},
        "additionalProperties": False,
        "examples": [{"title": "data, not an annotation"}],
    }

    before = deepcopy(schema)
    assert _strip_schema_keywords(schema, {"title"}) == {
        "properties": {"title": {"type": "string"}},
        "$defs": {"Title": {"type": "object"}},
        "dependentSchemas": {"title": {"required": ["title"]}},
        "dependencies": {"title": {}, "other": ["a", "b"]},
        "items": {"type": "integer"},
        "additionalItems": {"type": "boolean"},
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "anyOf": [{"type": "string"}],
        "allOf": [{"type": "integer"}],
        "propertyNames": {"pattern": "^[a-z]+$"},
        "if": {"required": ["a"]},
        "then": {"required": ["b"]},
        "additionalProperties": False,
        "examples": [{"title": "data, not an annotation"}],
    }
    # The walk copies schema nodes and leaves the input untouched, while data values
    # stay the caller's own objects.
    assert schema == before
    stripped = _strip_schema_keywords(schema, {"title"})
    assert stripped["examples"] is schema["examples"]

    # Draft-07 also allows a list of schemas directly under ``items``.
    assert _strip_schema_keywords({"items": [{"title": "Left", "type": "string"}]}, {"title"}) == {
        "items": [{"type": "string"}]
    }
