from __future__ import annotations

import json
import math
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.utils.serde import bounded_json_loads


_DEFAULT_ARGUMENT_LIMIT_BYTES = (
    DEFAULT_CONFIG.tools.tool_call_args_hard_limit_bytes
)


def _validate_direct_json_value(value: Any) -> None:
    """Reject Python-only values before canonical JSON serialization."""

    pending = [(value, False)]
    active_container_ids: set[int] = set()
    while pending:
        current, exiting = pending.pop()
        if exiting:
            active_container_ids.remove(id(current))
            continue
        current_type = type(current)
        if current is None or current_type in {str, bool, int}:
            continue
        if current_type is float:
            if not math.isfinite(current):
                raise ValueError("JSON numbers must be finite")
            continue
        if current_type not in {dict, list}:
            raise ValueError(
                "tool arguments must contain only JSON-compatible values"
            )
        container_id = id(current)
        if container_id in active_container_ids:
            raise ValueError("tool arguments must not contain circular values")
        active_container_ids.add(container_id)
        pending.append((current, True))
        if current_type is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise ValueError("tool argument object keys must be strings")
                pending.append((item, False))
        else:
            pending.extend((item, False) for item in current)


def _canonicalize_argument_object(
    value: dict[str, Any],
    *,
    max_argument_bytes: int,
) -> dict[str, Any]:
    _validate_direct_json_value(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("tool arguments must be a valid JSON object") from exc
    decoded = bounded_json_loads(
        serialized,
        max_bytes=max_argument_bytes,
    )
    if not isinstance(decoded, dict):
        raise ValueError("tool arguments must decode to an object")
    return decoded


def tool_call_to_action(
    tool_call: dict[str, Any],
    *,
    max_argument_bytes: int = _DEFAULT_ARGUMENT_LIMIT_BYTES,
) -> dict[str, Any]:
    name = str(tool_call.get("name") or "").strip()
    raw_args = tool_call.get("arguments")
    # Empty arguments are common for no-arg tool calls, but other false-y
    # values such as [] or 0 are malformed protocol frames and should surface
    # as repairable LLM output errors.
    if raw_args is None or raw_args == "":
        raw_args = "{}"
    if isinstance(raw_args, str):
        args = bounded_json_loads(
            raw_args,
            max_bytes=max_argument_bytes,
        )
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        raise ValueError(f"invalid tool arguments for {name}: {type(raw_args).__name__}")
    if not isinstance(args, dict):
        raise ValueError(f"tool arguments for {name} must decode to an object")
    args = _canonicalize_argument_object(
        args,
        max_argument_bytes=max_argument_bytes,
    )
    if not name:
        fallback_name = str(args.get("action") or "").strip()
        if not fallback_name:
            raise ValueError("tool call is missing a function name")
        name = fallback_name
    args = {key: value for key, value in args.items() if key != "action"}
    return {**args, "action": name}
