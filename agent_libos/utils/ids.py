from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(value: object) -> int:
    """Return a conservative, Provider-neutral token estimate.

    ASCII text is budgeted at roughly three characters per token while each
    non-ASCII code point is budgeted as one token. Structured values are first
    converted to deterministic canonical JSON so estimates never depend on a
    Python ``repr`` or an object's memory address.
    """

    text = value if isinstance(value, str) else _canonical_estimation_json(value)
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 3) + non_ascii_chars)


def _canonical_estimation_json(value: object) -> str:
    try:
        normalized = _normalize_estimation_value(value, active=set(), depth=0)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except Exception:
        # User-defined serialization hooks and legacy cyclic values must not
        # make resource accounting fail. A type-only marker is deterministic
        # and, unlike repr(value), never embeds a process-specific address.
        return json.dumps(
            {"_non_json_type": _stable_type_name(value)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _normalize_estimation_value(
    value: Any,
    *,
    active: set[int],
    depth: int,
) -> Any:
    if depth >= 64:
        return {"_max_depth_type": _stable_type_name(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {"_non_finite_number": label}
    if isinstance(value, Enum):
        return _normalize_estimation_value(
            value.value,
            active=active,
            depth=depth + 1,
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "_binary_type": type(value).__name__,
            "base64": base64.b64encode(raw).decode("ascii"),
        }

    identity = id(value)
    if identity in active:
        return {"_cycle_type": _stable_type_name(value)}

    active.add(identity)
    try:
        return _normalize_estimation_container(value, active=active, depth=depth)
    finally:
        active.remove(identity)


def _normalize_estimation_container(value: Any, *, active: set[int], depth: int) -> Any:
    next_depth = depth + 1
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_estimation_value(
                getattr(value, field.name), active=active, depth=next_depth
            )
            for field in fields(value)
        }

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _normalize_estimation_value(dumped, active=active, depth=next_depth)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_estimation_value(to_dict(), active=active, depth=next_depth)

    if isinstance(value, Mapping):
        return _normalize_estimation_mapping(value, active=active, depth=next_depth)
    if isinstance(value, (list, tuple)):
        return [
            _normalize_estimation_value(item, active=active, depth=next_depth)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _normalize_estimation_value(item, active=active, depth=next_depth)
            for item in value
        ]
        items.sort(key=_normalized_sort_key)
        return {"_set_items": items}
    return {"_non_json_type": _stable_type_name(value)}


def _normalize_estimation_mapping(
    value: Mapping[Any, Any],
    *,
    active: set[int],
    depth: int,
) -> Any:
    if all(isinstance(key, str) for key in value):
        return {
            key: _normalize_estimation_value(item, active=active, depth=depth)
            for key, item in value.items()
        }
    entries = [
        [
            _normalize_estimation_value(key, active=active, depth=depth),
            _normalize_estimation_value(item, active=active, depth=depth),
        ]
        for key, item in value.items()
    ]
    entries.sort(key=_normalized_sort_key)
    return {"_mapping_entries": entries}


def _normalized_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_type_name(value: object) -> str:
    value_type = type(value)
    module = getattr(value_type, "__module__", "") or ""
    qualname = getattr(value_type, "__qualname__", "") or value_type.__name__
    return f"{module}.{qualname}" if module else qualname
