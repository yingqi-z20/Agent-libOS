from __future__ import annotations

from copy import deepcopy
from typing import Any


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

    original = deepcopy(schema)
    candidate = deepcopy(schema)
    if _normalize_schema(candidate):
        return candidate, True
    return original, False


def normalize_openai_structured_output_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    normalized, strict = normalize_openai_strict_schema(schema)
    if not strict:
        raise ValueError(
            "structured output schema must be compatible with OpenAI strict JSON schema"
        )
    return normalized


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
    "normalize_openai_chat_tool_schema",
    "normalize_openai_strict_schema",
    "normalize_openai_structured_output_schema",
    "openai_chat_tool_schema",
    "openai_responses_tool_schema",
]
