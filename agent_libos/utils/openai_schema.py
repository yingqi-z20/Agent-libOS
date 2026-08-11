from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


_OPENAI_SCHEMA_MAX_DEPTH = 64
_OPENAI_SCHEMA_MAX_NODES = 4_096
_OPENAI_SCHEMA_MAX_BYTES = 1_048_576


def openai_chat_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized, strict = normalize_openai_strict_schema(
        parameters or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": normalized,
            "strict": strict,
        },
    }


def normalize_openai_chat_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        return deepcopy(tool)
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    if not isinstance(function, dict):
        return deepcopy(tool)
    name = function.get("name")
    if not name:
        return deepcopy(tool)
    normalized, strict = normalize_openai_strict_schema(
        function.get("parameters") or {"type": "object", "properties": {}}
    )
    selected = deepcopy(tool)
    selected["function"] = dict(function)
    selected["function"]["parameters"] = normalized
    selected["function"]["strict"] = strict
    return selected


def openai_responses_tool_schema(chat_tool: dict[str, Any]) -> dict[str, Any] | None:
    if chat_tool.get("type") != "function":
        return None
    function = (
        chat_tool.get("function")
        if isinstance(chat_tool.get("function"), dict)
        else chat_tool
    )
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    normalized, strict = normalize_openai_strict_schema(
        function.get("parameters") or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "name": name,
        "description": function.get("description", ""),
        "parameters": normalized,
        "strict": strict,
    }


def normalize_openai_strict_schema(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return an OpenAI strict-mode schema when this can be done safely.

    OpenAI strict function/schema mode requires closed objects and all declared
    properties in ``required``. Schemas that intentionally accept arbitrary keys
    are left unchanged and marked non-strict so runtime validation semantics stay
    compatible.
    """

    _validate_schema_bounds(schema)
    candidate = deepcopy(schema)
    _strip_model_annotation_titles(candidate)
    if _normalize_schema(candidate):
        return candidate, True
    return candidate, False


def compact_model_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove non-semantic generated annotations from a model-facing schema."""

    _validate_schema_bounds(schema)
    selected = deepcopy(schema)
    _strip_model_annotation_titles(selected)
    return selected


def _strip_model_annotation_titles(value: Any) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            current.pop("title", None)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def normalize_openai_structured_output_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    normalized, strict = normalize_openai_strict_schema(schema)
    if not strict:
        raise ValueError(
            "structured output schema must be compatible with OpenAI strict JSON schema"
        )
    return normalized


def _validate_schema_bounds(schema: dict[str, Any]) -> None:
    """Reject cyclic or oversized schemas before copying or recursive traversal."""

    pending: list[tuple[Any, int, bool]] = [(schema, 0, False)]
    active_containers: set[int] = set()
    node_count = 0
    encoded_bytes = 0

    while pending:
        value, depth, leaving = pending.pop()
        if leaving:
            active_containers.remove(id(value))
            continue

        node_count += 1
        if node_count > _OPENAI_SCHEMA_MAX_NODES:
            raise ValueError(
                "OpenAI schema exceeds maximum node count="
                f"{_OPENAI_SCHEMA_MAX_NODES}"
            )
        if depth > _OPENAI_SCHEMA_MAX_DEPTH:
            raise ValueError(
                "OpenAI schema exceeds maximum depth="
                f"{_OPENAI_SCHEMA_MAX_DEPTH}"
            )

        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("OpenAI schema object keys must be strings")
            container_id = id(value)
            if container_id in active_containers:
                raise ValueError("OpenAI schema must not contain cyclic containers")
            active_containers.add(container_id)
            pending.append((value, depth, True))
            encoded_bytes += 2 + max(0, len(value) - 1) + len(value)
            for key, child in reversed(tuple(value.items())):
                encoded_bytes += len(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                pending.append((child, depth + 1, False))
        elif isinstance(value, list):
            container_id = id(value)
            if container_id in active_containers:
                raise ValueError("OpenAI schema must not contain cyclic containers")
            active_containers.add(container_id)
            pending.append((value, depth, True))
            encoded_bytes += 2 + max(0, len(value) - 1)
            pending.extend(
                (child, depth + 1, False) for child in reversed(value)
            )
        else:
            try:
                encoded = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("OpenAI schema must contain only JSON values") from exc
            encoded_bytes += len(encoded.encode("utf-8"))

        if encoded_bytes > _OPENAI_SCHEMA_MAX_BYTES:
            raise ValueError(
                "OpenAI schema exceeds maximum encoded bytes="
                f"{_OPENAI_SCHEMA_MAX_BYTES}"
            )


def _normalize_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False

    strict = True
    for dynamic_key in (
        "patternProperties",
        "propertyNames",
        "allOf",
        "oneOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
    ):
        if dynamic_key in schema:
            strict = False
    if schema.get("unevaluatedProperties") not in (None, False):
        strict = False

    for defs_key in ("$defs", "definitions"):
        defs = schema.get(defs_key)
        if isinstance(defs, dict):
            for definition in defs.values():
                strict = _normalize_schema(definition) and strict

    variants = schema.get("anyOf")
    if isinstance(variants, list):
        for variant in variants:
            strict = _normalize_schema(variant) and strict

    items = schema.get("items")
    if isinstance(items, dict):
        strict = _normalize_schema(items) and strict
    prefix_items = schema.get("prefixItems")
    if isinstance(prefix_items, list):
        for item in prefix_items:
            strict = _normalize_schema(item) and strict

    schema_type = schema.get("type")
    is_object = (
        schema_type == "object"
        or "properties" in schema
        or "additionalProperties" in schema
    )
    if not is_object:
        return strict

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict) or additional is True:
        return False
    schema["additionalProperties"] = False

    properties = schema.get("properties")
    if properties is None:
        properties = {}
        schema["properties"] = properties
    if not isinstance(properties, dict):
        return False
    schema["required"] = [str(name) for name in properties]
    for prop in properties.values():
        strict = _normalize_schema(prop) and strict
    return strict


__all__ = [
    "compact_model_json_schema",
    "normalize_openai_chat_tool_schema",
    "normalize_openai_strict_schema",
    "normalize_openai_structured_output_schema",
    "openai_chat_tool_schema",
    "openai_responses_tool_schema",
]
