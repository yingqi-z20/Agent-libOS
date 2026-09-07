"""Bounded, private wire state for stateless OpenAI Responses continuations.

These values contain opaque provider state. They are never observability data
and must be kept out of messages, events, public call records and images.
"""
from __future__ import annotations

import json
import math
from typing import Any


RESPONSE_ITEMS_MAX_BYTES = 4 * 1024 * 1024
RESPONSE_ITEMS_MAX_ITEMS = 2_048
RESPONSE_ITEMS_MAX_STRING_BYTES = 1024 * 1024
_MAX_DEPTH = 16
_MAX_NODES = 32_768
_ITEM_FIELDS = {
    "reasoning": {"type", "id", "summary", "content", "encrypted_content", "status"},
    "message": {"type", "id", "role", "content", "phase", "status"},
    "function_call": {"type", "id", "call_id", "name", "arguments", "status"},
    "function_call_output": {"type", "id", "call_id", "output", "status"},
}


class ResponseItemsError(ValueError):
    """A continuation cannot be represented completely and safely."""


class _BoundedWireCopy:
    def __init__(self) -> None:
        self.nodes = 0
        self.string_bytes = 0

    def copy(self, value: Any, depth: int = 0) -> Any:
        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ResponseItemsError("Responses replay structure exceeds the supported bound")
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, (int, float)):
            return self._number(value)
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_NODES:
                raise ResponseItemsError("Responses replay structure exceeds the supported bound")
            return [self.copy(child, depth + 1) for child in value]
        return self._object(value, depth)

    def _string(self, value: str) -> str:
        if len(value) > RESPONSE_ITEMS_MAX_STRING_BYTES:
            raise ResponseItemsError("Responses replay text exceeds the supported bound")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise ResponseItemsError("Responses replay text is not valid UTF-8") from None
        self.string_bytes += size
        if size > RESPONSE_ITEMS_MAX_STRING_BYTES or self.string_bytes > RESPONSE_ITEMS_MAX_BYTES:
            raise ResponseItemsError("Responses replay text exceeds the supported bound")
        return value

    @staticmethod
    def _number(value: int | float) -> int | float:
        if isinstance(value, float) and not math.isfinite(value):
            raise ResponseItemsError("Responses replay contains a non-finite number")
        if isinstance(value, int) and value.bit_length() > 64:
            raise ResponseItemsError("Responses replay integer exceeds the supported bound")
        return value

    def _object(self, value: Any, depth: int) -> dict[str, Any]:
        if not isinstance(value, dict):
            value = self._sdk_fields(value)
        if len(value) > _MAX_NODES:
            raise ResponseItemsError("Responses replay structure exceeds the supported bound")
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResponseItemsError("Responses replay object keys must be strings")
            copied[self.copy(key, depth + 1)] = self.copy(child, depth + 1)
        return copied

    @staticmethod
    def _sdk_fields(value: Any) -> dict[str, Any]:
        # SDK BaseModels and deterministic stand-ins expose fields directly;
        # do not invoke arbitrary serialization hooks on provider objects.
        fields = getattr(value, "__dict__", None)
        if not isinstance(fields, dict):
            raise ResponseItemsError("Responses replay contains an unsupported value")
        extras = getattr(value, "__pydantic_extra__", None)
        if len(fields) > _MAX_NODES or (isinstance(extras, dict) and len(extras) > _MAX_NODES):
            raise ResponseItemsError("Responses replay structure exceeds the supported bound")
        copied = {key: child for key, child in fields.items() if isinstance(key, str) and not key.startswith("_")}
        if isinstance(extras, dict):
            copied.update(extras)
        return copied


def validate_response_items(items: Any, *, output: bool = False) -> list[dict[str, Any]]:
    """Validate and copy complete ordered wire items without lossy projection.

    Input requires every tool output to match an earlier call exactly once,
    and every call to have its result. Errors never quote provider values.
    """
    if not isinstance(items, (list, tuple)) or len(items) > RESPONSE_ITEMS_MAX_ITEMS:
        raise ResponseItemsError("Responses replay item count exceeds the supported bound")
    selected = _BoundedWireCopy().copy(items)
    calls: set[str] = set()
    completed: set[str] = set()
    for item in selected:
        kind = _validate_item_shape(item, output=output)
        if kind == "reasoning":
            _validate_reasoning(item)
        elif kind == "message":
            _validate_message(item, output=output)
        else:
            _validate_tool_item(item, kind=kind, calls=calls, completed=completed)
    if not output and calls != completed:
        raise ResponseItemsError("Responses replay contains an unfinished tool group")
    if len(json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > RESPONSE_ITEMS_MAX_BYTES:
        raise ResponseItemsError("Responses replay exceeds the supported byte bound")
    return selected


def _validate_item_shape(item: Any, *, output: bool) -> str:
    if not isinstance(item, dict):
        raise ResponseItemsError("Responses replay items must be objects")
    kind = item.get("type", "message" if "role" in item else None)
    if not isinstance(kind, str):
        raise ResponseItemsError("Responses replay contains an invalid item type")
    allowed = _ITEM_FIELDS.get(kind)
    if allowed is None or (output and kind == "function_call_output"):
        raise ResponseItemsError("Responses replay contains an unsupported item type")
    if kind == "function_call":
        # SDK models materialize absent routing metadata as None. Normalize
        # those defaults only; non-null routing and unknown fields fail closed.
        for key in ("caller", "namespace"):
            if item.get(key) is None:
                item.pop(key, None)
    if set(item) - allowed:
        raise ResponseItemsError("Responses replay contains unsupported item fields")
    if item.get("status") is not None and item["status"] != "completed":
        raise ResponseItemsError("Responses replay contains an incomplete item")
    if item.get("id") is not None and not isinstance(item["id"], str):
        raise ResponseItemsError("Responses replay contains an invalid item identifier")
    return kind


def _validate_reasoning(item: dict[str, Any]) -> None:
    if item.get("encrypted_content") is not None and not isinstance(item["encrypted_content"], str):
        raise ResponseItemsError("Responses replay contains invalid encrypted reasoning")
    for key in ("summary", "content"):
        if item.get(key) is not None and not isinstance(item[key], list):
            raise ResponseItemsError("Responses replay contains invalid reasoning blocks")
        for block in item.get(key) or []:
            expected = "summary_text" if key == "summary" else "reasoning_text"
            if not isinstance(block, dict) or set(block) != {"type", "text"} or block.get("type") != expected or not isinstance(block.get("text"), str):
                raise ResponseItemsError("Responses replay contains an unsupported reasoning block")


def _validate_message(item: dict[str, Any], *, output: bool) -> None:
    roles = {"assistant"} if output else {"system", "developer", "user", "assistant"}
    if not isinstance(item.get("role"), str) or item["role"] not in roles or not isinstance(item.get("content"), (str, list)):
        raise ResponseItemsError("Responses replay contains an invalid message")
    if item.get("phase") is not None and item["phase"] not in ("commentary", "final_answer"):
        raise ResponseItemsError("Responses replay contains an unsupported assistant phase")
    if isinstance(item["content"], list):
        for block in item["content"]:
            _validate_message_block(block, output=output)


def _validate_tool_item(item: dict[str, Any], *, kind: str, calls: set[str], completed: set[str]) -> None:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ResponseItemsError("Responses replay contains an invalid tool call identifier")
    if kind == "function_call":
        if not isinstance(item.get("name"), str) or not item["name"] or not isinstance(item.get("arguments"), str):
            raise ResponseItemsError("Responses replay contains an invalid function call")
        if call_id in calls:
            raise ResponseItemsError("Responses replay contains duplicate tool calls")
        calls.add(call_id)
    else:
        if call_id not in calls or call_id in completed:
            raise ResponseItemsError("Responses replay contains an unpaired tool result")
        if not isinstance(item.get("output"), str):
            raise ResponseItemsError("Responses replay tool results must be safe text")
        completed.add(call_id)


def _validate_message_block(block: Any, *, output: bool) -> None:
    if not isinstance(block, dict):
        raise ResponseItemsError("Responses replay contains an invalid message block")
    kind = block.get("type")
    allowed = {
        "output_text": {"type", "text", "annotations", "logprobs"},
        "refusal": {"type", "refusal"},
        "input_text": {"type", "text", "prompt_cache_breakpoint"},
    }
    if not isinstance(kind, str) or kind not in allowed or (output and kind == "input_text"):
        raise ResponseItemsError("Responses replay contains an unsupported message block")
    if set(block) - allowed[kind] or not isinstance(block.get("refusal" if kind == "refusal" else "text"), str):
        raise ResponseItemsError("Responses replay contains invalid message block fields")
    for key in ("annotations", "logprobs"):
        if block.get(key) is not None and not isinstance(block[key], list):
            raise ResponseItemsError("Responses replay contains invalid message block metadata")
    if "prompt_cache_breakpoint" in block and block["prompt_cache_breakpoint"] != {"mode": "explicit"}:
        raise ResponseItemsError("Responses replay contains an unsupported cache breakpoint")
