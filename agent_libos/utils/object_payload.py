"""Canonical payload digests shared by trusted Object lineage consumers."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from agent_libos.utils.serde import dumps, to_jsonable


def object_payload_sha256(payload: Any) -> str:
    """Hash the runtime's finite JSON projection, including legacy NaN rows."""

    return hashlib.sha256(dumps(_finite_json_projection(to_jsonable(payload))).encode("utf-8")).hexdigest()


def _finite_json_projection(value: Any) -> Any:
    if type(value) is float and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return {"_non_finite_number": label}
    if isinstance(value, dict):
        return {key: _finite_json_projection(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json_projection(item) for item in value]
    return value
