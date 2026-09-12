"""Narrow normalization for provider extensions absent from the OpenAI SDK."""
from __future__ import annotations

from typing import Any


def omit_unset_sdk_defaults(
    value: Any, fields: dict[str, Any], allowed: set[str],
) -> dict[str, Any]:
    """Remove only synthetic message fields on an Aliyun extractor item.

    The SDK constructs an unknown Responses output as ResponseOutputMessage,
    materializing absent message fields as None. Explicit fields, including
    explicit nulls, must survive so protocol allowlists still reject them.
    Provider extensions in __pydantic_extra__ are checked independently.
    """
    if fields.get("type") != "web_extractor_call":
        return fields
    # Most Runtime use never imports the SDK. Only this provider extension
    # requires inspecting its concrete fallback model.
    from openai.types.responses import ResponseOutputMessage

    if not isinstance(value, ResponseOutputMessage):
        return fields
    supplied = value.__pydantic_fields_set__
    omitted = {name for name in ("content", "role", "phase")
               if name not in allowed and name not in supplied and fields.get(name) is None}
    return {name: child for name, child in fields.items() if name not in omitted}
