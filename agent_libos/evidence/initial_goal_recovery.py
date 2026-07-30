from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


INITIAL_GOAL_RECOVERY_KEY = "initial_goal_recovery"
INITIAL_GOAL_RECOVERY_SCHEMA_VERSION = 1
INITIAL_GOAL_RECOVERY_FULL = "full"
INITIAL_GOAL_RECOVERY_HASH_ONLY = "hash_only"


@dataclass(frozen=True, slots=True)
class InitialGoalRecoveryEnvelope:
    pid: str
    process_created_at: str
    image_id: str
    goal_oid: str
    object_version: int
    object_identity_sha256: str
    payload_sha256: str
    payload_bytes: int
    retention: str
    payload: Any | None = None

    @property
    def recoverable(self) -> bool:
        return self.retention == INITIAL_GOAL_RECOVERY_FULL


def canonical_initial_goal_json_bytes(value: Any) -> bytes:
    """Encode an exact JSON value without coercive ``default=str`` fallbacks."""

    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def initial_goal_object_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_initial_goal_json_bytes(dict(identity))).hexdigest()


def initial_goal_object_identity(
    *,
    oid: str,
    namespace: str,
    name: str,
    object_type: str,
    schema_version: str,
    metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
    version: int,
    immutable: bool,
    created_by: str,
    owner_kind: str,
    owner_id: str,
    created_at: str,
) -> dict[str, Any]:
    for field_name, value in (
        ("oid", oid),
        ("namespace", namespace),
        ("name", name),
        ("object_type", object_type),
        ("schema_version", schema_version),
        ("created_by", created_by),
        ("owner_kind", owner_kind),
        ("owner_id", owner_id),
        ("created_at", created_at),
    ):
        _require_nonempty_text(value, field_name)
    _require_positive_int(version, "object identity version")
    if type(immutable) is not bool:
        raise ValueError("initial goal recovery object identity immutable must be boolean")
    if not isinstance(metadata, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError(
            "initial goal recovery object metadata and provenance must be objects"
        )
    return {
        "oid": oid,
        "namespace": namespace,
        "name": name,
        "type": object_type,
        "schema_version": schema_version,
        "metadata": dict(metadata),
        "provenance": dict(provenance),
        "version": version,
        "immutable": immutable,
        "created_by": created_by,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "created_at": created_at,
    }


def build_initial_goal_recovery_envelope(
    *,
    pid: str,
    process_created_at: str,
    image_id: str,
    goal_oid: str,
    object_version: int,
    object_identity_sha256: str,
    payload: Any,
    persist_full_io: bool,
    payload_hard_limit_bytes: int,
) -> dict[str, Any]:
    _require_nonempty_text(pid, "pid")
    _require_nonempty_text(process_created_at, "process_created_at")
    _require_nonempty_text(image_id, "image_id")
    _require_nonempty_text(goal_oid, "goal_oid")
    _require_positive_int(object_version, "object_version")
    _require_sha256(object_identity_sha256, "object_identity_sha256")
    _require_positive_int(payload_hard_limit_bytes, "payload_hard_limit_bytes")
    if type(persist_full_io) is not bool:
        raise ValueError("persist_full_io must be a boolean")

    encoded = canonical_initial_goal_json_bytes(payload)
    if len(encoded) > payload_hard_limit_bytes:
        raise ValueError(
            "initial goal recovery payload exceeds configured hard limit: "
            f"{len(encoded)} > {payload_hard_limit_bytes}"
        )
    envelope: dict[str, Any] = {
        "schema_version": INITIAL_GOAL_RECOVERY_SCHEMA_VERSION,
        "pid": pid,
        "process_created_at": process_created_at,
        "image_id": image_id,
        "goal_oid": goal_oid,
        "object_version": object_version,
        "object_identity_sha256": object_identity_sha256,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload_bytes": len(encoded),
        "retention": (
            INITIAL_GOAL_RECOVERY_FULL
            if persist_full_io
            else INITIAL_GOAL_RECOVERY_HASH_ONLY
        ),
    }
    if persist_full_io:
        envelope["payload"] = deepcopy(payload)
    return envelope


def decode_initial_goal_recovery_envelope(
    value: Any,
    *,
    payload_hard_limit_bytes: int,
) -> InitialGoalRecoveryEnvelope:
    _require_positive_int(payload_hard_limit_bytes, "payload_hard_limit_bytes")
    if not isinstance(value, Mapping):
        raise ValueError("initial goal recovery envelope must be an object")
    selected = dict(value)
    retention = selected.get("retention")
    common_keys = {
        "schema_version",
        "pid",
        "process_created_at",
        "image_id",
        "goal_oid",
        "object_version",
        "object_identity_sha256",
        "payload_sha256",
        "payload_bytes",
        "retention",
    }
    expected_keys = (
        common_keys | {"payload"}
        if retention == INITIAL_GOAL_RECOVERY_FULL
        else common_keys
    )
    if set(selected) != expected_keys:
        raise ValueError("initial goal recovery envelope has an invalid shape")
    if selected.get("schema_version") != INITIAL_GOAL_RECOVERY_SCHEMA_VERSION:
        raise ValueError("initial goal recovery envelope has an unsupported schema")
    if retention not in {
        INITIAL_GOAL_RECOVERY_FULL,
        INITIAL_GOAL_RECOVERY_HASH_ONLY,
    }:
        raise ValueError("initial goal recovery envelope has an invalid retention tier")

    pid = _require_nonempty_text(selected.get("pid"), "pid")
    process_created_at = _require_nonempty_text(
        selected.get("process_created_at"), "process_created_at"
    )
    image_id = _require_nonempty_text(selected.get("image_id"), "image_id")
    goal_oid = _require_nonempty_text(selected.get("goal_oid"), "goal_oid")
    object_version = _require_positive_int(
        selected.get("object_version"), "object_version"
    )
    object_identity_sha256 = _require_sha256(
        selected.get("object_identity_sha256"), "object_identity_sha256"
    )
    payload_sha256 = _require_sha256(
        selected.get("payload_sha256"), "payload_sha256"
    )
    payload_bytes = _require_nonnegative_int(
        selected.get("payload_bytes"), "payload_bytes"
    )
    if payload_bytes > payload_hard_limit_bytes:
        raise ValueError(
            "initial goal recovery payload exceeds configured hard limit: "
            f"{payload_bytes} > {payload_hard_limit_bytes}"
        )

    payload = None
    if retention == INITIAL_GOAL_RECOVERY_FULL:
        payload = deepcopy(selected["payload"])
        encoded = canonical_initial_goal_json_bytes(payload)
        if len(encoded) != payload_bytes:
            raise ValueError("initial goal recovery payload byte count changed")
        if hashlib.sha256(encoded).hexdigest() != payload_sha256:
            raise ValueError("initial goal recovery payload digest changed")

    return InitialGoalRecoveryEnvelope(
        pid=pid,
        process_created_at=process_created_at,
        image_id=image_id,
        goal_oid=goal_oid,
        object_version=object_version,
        object_identity_sha256=object_identity_sha256,
        payload_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        retention=str(retention),
        payload=payload,
    )


def redact_initial_goal_recovery_envelope(
    value: Any,
    *,
    payload_hard_limit_bytes: int,
) -> dict[str, Any]:
    decoded = decode_initial_goal_recovery_envelope(
        value,
        payload_hard_limit_bytes=payload_hard_limit_bytes,
    )
    return {
        "schema_version": INITIAL_GOAL_RECOVERY_SCHEMA_VERSION,
        "pid": decoded.pid,
        "process_created_at": decoded.process_created_at,
        "image_id": decoded.image_id,
        "goal_oid": decoded.goal_oid,
        "object_version": decoded.object_version,
        "object_identity_sha256": decoded.object_identity_sha256,
        "payload_sha256": decoded.payload_sha256,
        "payload_bytes": decoded.payload_bytes,
        "retention": INITIAL_GOAL_RECOVERY_HASH_ONLY,
    }


def redact_initial_goal_recovery_receipt_projection(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove reversible goal content from every ordinary publication view."""

    projected = deepcopy(dict(receipt))
    phases = projected.get("phases")
    if not isinstance(phases, list):
        return projected
    safe_keys = {
        "schema_version",
        "pid",
        "process_created_at",
        "image_id",
        "goal_oid",
        "object_version",
        "object_identity_sha256",
        "payload_sha256",
        "payload_bytes",
    }
    for phase in phases:
        if not isinstance(phase, dict) or INITIAL_GOAL_RECOVERY_KEY not in phase:
            continue
        raw = phase[INITIAL_GOAL_RECOVERY_KEY]
        if not isinstance(raw, Mapping):
            phase[INITIAL_GOAL_RECOVERY_KEY] = {
                "schema_version": INITIAL_GOAL_RECOVERY_SCHEMA_VERSION,
                "retention": "unavailable",
            }
            continue
        redacted = {
            key: deepcopy(raw[key])
            for key in safe_keys
            if key in raw
        }
        redacted["retention"] = INITIAL_GOAL_RECOVERY_HASH_ONLY
        phase[INITIAL_GOAL_RECOVERY_KEY] = redacted
    return projected


def _validate_json_value(value: Any) -> None:
    active: set[int] = set()
    leaving_marker = object()
    stack: list[tuple[Any, object | None]] = [(value, None)]
    while stack:
        current, marker = stack.pop()
        if marker is leaving_marker:
            active.remove(id(current))
            continue
        current_type = type(current)
        if current is None or current_type in {str, bool, int}:
            continue
        if current_type is float:
            if not math.isfinite(current):
                raise ValueError("initial goal recovery payload contains a non-finite number")
            continue
        if current_type not in {dict, list}:
            raise ValueError(
                "initial goal recovery payload contains a non-JSON value: "
                f"{current_type.__name__}"
            )
        identity = id(current)
        if identity in active:
            raise ValueError("initial goal recovery payload contains a container cycle")
        active.add(identity)
        stack.append((current, leaving_marker))
        if current_type is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise ValueError("initial goal recovery payload object keys must be strings")
                stack.append((item, None))
        else:
            for item in current:
                stack.append((item, None))


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"initial goal recovery {field_name} must be non-empty text")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"initial goal recovery {field_name} must be positive")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"initial goal recovery {field_name} must be non-negative")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"initial goal recovery {field_name} must be canonical sha256")
    return value
