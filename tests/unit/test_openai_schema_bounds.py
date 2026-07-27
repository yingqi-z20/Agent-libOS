from __future__ import annotations

import pytest

from agent_libos.utils.openai_schema import normalize_openai_strict_schema


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
