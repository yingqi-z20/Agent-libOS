from __future__ import annotations

import pytest

from agent_libos.utils.openai_schema import (
    compact_model_json_schema,
    normalize_openai_strict_schema,
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
