from __future__ import annotations

from copy import deepcopy

import pytest

from agent_libos.utils.openai_schema import (
    compact_model_json_schema,
    normalize_openai_chat_tool_schema,
    normalize_openai_strict_schema,
    normalize_openai_structured_output_schema,
    openai_chat_tool_schema,
    openai_responses_tool_schema,
)


def test_openai_schema_normalization_rejects_cycles_before_copying() -> None:
    schema: dict[str, object] = {"type": "object"}
    schema["properties"] = {"self": schema}

    with pytest.raises(ValueError, match="cyclic containers"):
        normalize_openai_strict_schema(schema)


def test_openai_schema_normalization_rejects_excessive_depth() -> None:
    schema: dict[str, object] = {"type": "string"}
    for _ in range(70):
        schema = {"type": "array", "items": schema}

    with pytest.raises(ValueError, match="maximum depth"):
        normalize_openai_strict_schema(schema)


def test_openai_schema_normalization_counts_repeated_alias_occurrences() -> None:
    shared = {"type": "string"}
    schema = {"anyOf": [shared] * 4_096}

    with pytest.raises(ValueError, match="maximum node count"):
        normalize_openai_strict_schema(schema)


def test_openai_schema_normalization_rejects_excessive_encoded_size() -> None:
    schema = {
        "type": "string",
        "description": "x" * 1_048_576,
    }

    with pytest.raises(ValueError, match="maximum encoded bytes"):
        normalize_openai_strict_schema(schema)


def test_openai_schema_normalization_preserves_bounded_behavior() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }

    normalized, strict = normalize_openai_strict_schema(schema)

    assert strict is True
    assert normalized["additionalProperties"] is False
    assert normalized["required"] == ["value"]
    assert schema == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }


def test_openai_schema_normalization_returns_detached_nonstrict_schema() -> None:
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    normalized, strict = normalize_openai_strict_schema(schema)
    normalized["additionalProperties"]["type"] = "number"

    assert strict is False
    assert schema["additionalProperties"] == {"type": "string"}


@pytest.mark.parametrize("required", [None, [], ["mode"]])
@pytest.mark.parametrize("dynamic_schema", [
    {"type": "object", "additionalProperties": True},
    {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}},
    {"type": "object", "patternProperties": {"^x": {"type": "integer"}}},
    {"allOf": [{"type": "object", "properties": {"value": {"type": "number"}}}]},
    {"type": "object", "unevaluatedProperties": True},
    {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": True}]},
    {"type": "array", "items": {"type": "object", "additionalProperties": True}},
    {"$defs": {"Labels": {"type": "object", "additionalProperties": {"type": "string"}}}, "$ref": "#/$defs/Labels"},
])
def test_nonstrict_fallback_restores_original_required_types_and_constraints(
    required: list[str] | None, dynamic_schema: dict[str, object],
) -> None:
    # These earlier objects are normalized before the unsupported payload is
    # reached. None of their strict-mode rewrites may escape on failure.
    expected = {
        "type": "object",
        "description": "Keep optional fields optional.",
        "minProperties": 0,
        "maxProperties": 5,
        "$defs": {
            "Choice": {
                "type": "object", "properties": {"value": {"type": "integer", "minimum": 1}},
                "required": [],
            },
        },
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "safe"], "default": "safe"},
            "result": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}], "default": None},
            "choice": {"$ref": "#/$defs/Choice"},
            "options": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "payload": dynamic_schema,
        },
    }
    if required is not None:
        expected["required"] = required
    schema = deepcopy(expected)
    schema["title"] = "Generated root title"
    schema["properties"]["mode"]["title"] = "Generated mode title"
    schema["$defs"]["Choice"]["title"] = "Generated definition title"
    original = deepcopy(schema)

    normalized, strict = normalize_openai_strict_schema(schema)

    assert strict is False
    assert normalized == expected
    assert schema == original
    normalized["properties"]["mode"]["enum"].append("changed")
    normalized["$defs"]["Choice"]["required"].append("value")
    assert schema == original


def test_strict_success_still_normalizes_nested_objects_without_mutating_input() -> None:
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "result": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}], "default": None},
            "options": {"title": "Options", "type": "object", "properties": {"count": {"type": "integer", "minimum": 0}}},
        },
    }
    original = deepcopy(schema)

    normalized, strict = normalize_openai_strict_schema(schema)

    assert strict is True
    assert normalized == {
        "type": "object", "additionalProperties": False, "required": ["result", "options"],
        "properties": {
            "result": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}], "default": None},
            "options": {
                "type": "object", "additionalProperties": False, "required": ["count"],
                "properties": {"count": {"type": "integer", "minimum": 0}},
            },
        },
    }
    assert schema == original
    normalized["properties"]["options"]["properties"]["count"]["minimum"] = -1
    assert schema == original


def test_nonstrict_fallback_preserves_literal_titles_and_named_title_properties() -> None:
    schema = {
        "title": "Generated annotation",
        "type": "object",
        "properties": {
            "title": {"title": "Generated field annotation", "type": "string", "default": "Untitled"},
            "payload": {
                "type": "object", "additionalProperties": True,
                "default": {"title": "Business data"},
                "examples": [{"title": "Example data"}],
            },
        },
    }

    normalized, strict = normalize_openai_strict_schema(schema)

    assert strict is False
    assert "title" not in normalized
    assert normalized["properties"]["title"] == {"type": "string", "default": "Untitled"}
    assert normalized["properties"]["payload"]["default"] == {"title": "Business data"}
    assert normalized["properties"]["payload"]["examples"] == [{"title": "Example data"}]
    assert "required" not in normalized


def test_nonstrict_tool_conversion_stays_optional_through_chat_and_responses() -> None:
    schema = {
        "title": "StoreRecord",
        "type": "object",
        "properties": {
            "result": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}], "default": None},
            "payload": {"type": "object", "additionalProperties": True},
        },
    }
    original = deepcopy(schema)
    expected = {key: value for key, value in schema.items() if key != "title"}

    chat = openai_chat_tool_schema("store_record", "Store an optional result.", schema)
    normalized_chat = normalize_openai_chat_tool_schema(chat)
    responses = openai_responses_tool_schema(normalized_chat)

    assert chat == normalized_chat
    assert responses is not None
    for tool in (chat["function"], normalized_chat["function"], responses):
        assert tool["strict"] is False
        assert tool["parameters"] == expected
        assert "required" not in tool["parameters"]
        assert "additionalProperties" not in tool["parameters"]
    assert schema == original
    with pytest.raises(ValueError, match="compatible with OpenAI strict"):
        normalize_openai_structured_output_schema(schema)


def test_openai_schema_normalization_removes_only_generated_titles_recursively() -> None:
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "mode": {
                "title": "Mode",
                "description": "Keep this guidance.",
                "enum": ["one", "two"],
                "default": "one",
            },
            "nested": {
                "title": "Nested",
                "type": "array",
                "items": {"title": "Item", "type": "integer", "minimum": 1},
            },
        },
    }

    normalized, strict = normalize_openai_strict_schema(schema)

    assert strict is True
    assert "title" not in str(normalized)
    assert normalized["properties"]["mode"] == {
        "description": "Keep this guidance.",
        "enum": ["one", "two"],
        "default": "one",
    }
    assert normalized["properties"]["nested"]["items"]["minimum"] == 1
    assert schema["title"] == "Root"


def test_openai_schema_title_compaction_preserves_title_named_schema_entries() -> None:
    schema = {
        "title": "Root annotation",
        "type": "object",
        "$defs": {
            "title": {
                "title": "Definition annotation",
                "type": "string",
            }
        },
        "properties": {
            "title": {
                "title": "Property annotation",
                "$ref": "#/$defs/title",
            }
        },
    }

    strict_schema, strict = normalize_openai_strict_schema(schema)
    compact_schema = compact_model_json_schema(schema)

    assert strict is True
    for selected in (strict_schema, compact_schema):
        assert "title" not in selected
        assert selected["properties"]["title"] == {"$ref": "#/$defs/title"}
        assert selected["$defs"]["title"] == {"type": "string"}
    assert schema["title"] == "Root annotation"
