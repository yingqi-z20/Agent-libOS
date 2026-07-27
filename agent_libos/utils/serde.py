from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


# Match CPython's default integer conversion ceiling explicitly so manifest
# parsing stays bounded even when the interpreter-wide limit is disabled.
_MAX_JSON_INTEGER_DIGITS = 4_300
# Keep externally supplied manifests well below interpreter recursion limits.
# CPython versions differ in whether deeply nested JSON raises RecursionError,
# so enforce one stable container-depth ceiling before and after decoding.
_MAX_JSON_NESTING_DEPTH = 256
# Bound the decoded object graph independently of the byte ceiling owned by
# each caller.  A fixed shared limit keeps parser behavior deterministic for
# manifests and provider text even when a compact document contains a very
# large number of scalar/container nodes.
_MAX_JSON_NODES = 100_000


def _parse_bounded_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ValueError(
            "JSON integer exceeds maximum digits="
            f"{_MAX_JSON_INTEGER_DIGITS}"
        )
    return int(value)


def _parse_finite_json_float(value: str) -> float:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError("JSON numbers must be finite")
    return selected


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError("JSON numbers must be finite")


def _decode_json_text(value: str | bytes | bytearray) -> str:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("JSON input must be valid UTF-8") from exc
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("JSON input must be valid UTF-8") from exc
    raise ValueError("JSON input must be str, bytes, or bytearray")


def _validate_json_source_nesting_depth(value: str) -> None:
    """Reject excessive nesting before CPython's recursive decoder sees it."""

    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING_DEPTH:
                raise ValueError(
                    "JSON nesting exceeds maximum depth="
                    f"{_MAX_JSON_NESTING_DEPTH}"
                )
        elif character in "]}":
            depth = max(0, depth - 1)


def _validate_json_nesting_depth(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, parent_depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError(
                f"JSON document exceeds maximum nodes={_MAX_JSON_NODES}"
            )
        if isinstance(current, dict):
            # Object keys are JSON scalar nodes too, but never add container
            # depth.  Account for them without needlessly pushing strings.
            nodes += len(current)
            if nodes > _MAX_JSON_NODES:
                raise ValueError(
                    f"JSON document exceeds maximum nodes={_MAX_JSON_NODES}"
                )
            children = current.values()
        elif isinstance(current, list):
            children = iter(current)
        else:
            continue

        depth = parent_depth + 1
        if depth > _MAX_JSON_NESTING_DEPTH:
            raise ValueError(
                "JSON nesting exceeds maximum depth="
                f"{_MAX_JSON_NESTING_DEPTH}"
            )
        pending.extend((child, depth) for child in children)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate names."""

    selected: dict[str, Any] = {}
    for key, item in pairs:
        if key in selected:
            # Keep the diagnostic independent of attacker-controlled key text.
            raise ValueError("JSON objects must not contain duplicate keys")
        selected[key] = item
    return selected


def bounded_json_loads(
    value: str | bytes | bytearray,
    *,
    max_bytes: int | None = None,
    reject_duplicate_keys: bool = True,
) -> Any:
    """Decode strict UTF-8 JSON with fixed numeric and depth ceilings.

    Every encoding, syntax, numeric, duplicate-key, or nesting failure is
    surfaced as a ``ValueError``. Callers can provide their domain byte limit
    and retain responsibility for translating failures into domain errors.

    ``reject_duplicate_keys=False`` is an explicit compatibility escape hatch
    for callers that intentionally consume legacy last-key-wins JSON. New
    authority or persistence boundaries should retain the strict default.
    """

    text = _decode_json_text(value)
    if not isinstance(reject_duplicate_keys, bool):
        raise ValueError("JSON reject_duplicate_keys must be a boolean")
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("JSON max_bytes must be an integer")
        if max_bytes < 1:
            raise ValueError("JSON max_bytes must be >= 1")
        if len(text.encode("utf-8")) > max_bytes:
            raise ValueError(f"JSON input exceeds max_bytes={max_bytes}")
    _validate_json_source_nesting_depth(text)
    try:
        decoded = json.loads(
            text,
            parse_int=_parse_bounded_json_integer,
            parse_float=_parse_finite_json_float,
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=(
                _unique_json_object
                if reject_duplicate_keys
                else None
            ),
        )
    except RecursionError as exc:
        raise ValueError(
            "JSON nesting exceeds maximum depth="
            f"{_MAX_JSON_NESTING_DEPTH}"
        ) from exc
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
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )


def loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)
