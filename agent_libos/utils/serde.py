from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


# Match CPython's default integer conversion ceiling explicitly so manifest
# parsing stays bounded even when the interpreter-wide limit is disabled.
_MAX_JSON_INTEGER_DIGITS = 4_300
# Keep externally supplied manifests well below interpreter recursion limits.
# CPython versions differ in whether deeply nested JSON raises RecursionError,
# so enforce one stable container-depth ceiling after decoding.
_MAX_JSON_NESTING_DEPTH = 256


def _parse_bounded_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ValueError(
            "JSON integer exceeds maximum digits="
            f"{_MAX_JSON_INTEGER_DIGITS}"
        )
    return int(value)


def _validate_json_nesting_depth(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    while pending:
        current, parent_depth = pending.pop()
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue

        depth = parent_depth + 1
        if depth > _MAX_JSON_NESTING_DEPTH:
            raise ValueError(
                "JSON nesting exceeds maximum depth="
                f"{_MAX_JSON_NESTING_DEPTH}"
            )
        pending.extend(
            (child, depth)
            for child in children
            if isinstance(child, (dict, list))
        )


def bounded_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode externally supplied JSON with fixed integer and depth ceilings.

    Callers retain responsibility for their byte limit and for translating
    parser ``ValueError``/``RecursionError`` failures into the appropriate
    domain error.
    """

    decoded = json.loads(value, parse_int=_parse_bounded_json_integer)
    _validate_json_nesting_depth(decoded)
    return decoded


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(mode="json"))
        except TypeError:
            return to_jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(item)
            for key, item in vars(value).items()
            if not callable(item)
        }
    return value


def dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=True, sort_keys=True, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)
