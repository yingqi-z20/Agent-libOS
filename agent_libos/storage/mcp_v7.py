"""Payload-free durable MCP client state introduced by store schema v7.

The records in this module are deliberately *not* wire-protocol models.  They
form the narrow persistence boundary between the modern MCP client managers and
RuntimeStore.  Bearer-like remote identifiers may be represented only by an
opaque credential-broker reference plus a digest.  Provider content, OAuth
tokens, client secrets, authorization codes, PKCE material, and OAuth state
have no column or metadata escape hatch in this contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from agent_libos.models.exceptions import ValidationError


MCP_V7_QUERY_HARD_LIMIT = 500
MCP_V7_METADATA_MAX_BYTES = 8 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTINUATION_STATUSES = frozenset(
    {
        "input_required",
        "dispatching",
        "complete",
        "cancelled",
        "expired",
        "needs_attention",
    }
)
_REMOTE_TASK_STATUSES = frozenset(
    {
        "working",
        "input_required",
        "completed",
        "failed",
        "cancelled",
        "cancel_requested",
        "update_dispatching",
        "cancel_dispatching",
        "needs_attention",
    }
)
_SUBSCRIPTION_STATUSES = frozenset(
    {"starting", "active", "stopping", "stopped", "lost", "needs_attention"}
)
_AUTH_STATUSES = frozenset(
    {
        "unconfigured",
        "authorization_required",
        "authorized",
        "expired",
        "revoked",
        "needs_attention",
    }
)
_SIDE_EFFECT_PREPARATION_STATUSES = frozenset({"prepared", "cleaning"})
_SIDE_EFFECT_OPERATION_KINDS = frozenset({"continuation", "remote_task"})
_SIDE_EFFECT_CLEANUP_MODES = frozenset({"abort", "retire"})

# Metadata is a small, closed diagnostics carrier.  It must never become a
# second payload column merely because a field was not anticipated in the DDL.
_METADATA_KEYS = frozenset(
    {
        "automatic_retry_disabled",
        "dispatch_state",
        "input_schema_sha256",
        "last_error_code",
        "reason_code",
        "request_method",
        "retry_class",
    }
)

# These are codes, not text fields.  A merely shape-checked string would be a
# covert persistence channel for a reflected token, authorization code, remote
# task id, or provider error body.  Keep the vocabulary closed and expand it
# only when a runtime-owned diagnostic is introduced deliberately.
_DISPATCH_STATES = frozenset({"not_started", "started", "unknown"})
_RETRY_CLASSES = frozenset(
    {"not_applicable", "not_retryable", "reobserve_required", "unsafe_or_unknown"}
)
_REQUEST_METHODS = frozenset(
    {
        "completion/complete",
        "elicitation/create",
        "initialize",
        "prompts/get",
        "prompts/list",
        "resources/list",
        "resources/read",
        "resources/templates/list",
        "subscriptions/listen",
        "tasks/cancel",
        "tasks/get",
        "tasks/update",
        "tools/call",
        "tools/list",
    }
)
_DIAGNOSTIC_CODES = frozenset(
    {
        # OAuth/auth projection diagnostics.
        "audience_mismatch",
        "authentication_failed",
        "authorization_required",
        "credential_expired",
        "credential_missing",
        "credential_revoked",
        "credential_rotation_unknown",
        "issuer_mismatch",
        "logout_unknown",
        "metadata_unavailable",
        "needs_attention",
        "refresh_failed",
        "refresh_unknown",
        "resource_mismatch",
        "scope_escalation",
        "secure_backend_unavailable",
        # Continuation and remote-task recovery diagnostics.
        "broker_integrity",
        "broker_missing",
        "broker_unavailable",
        "cancel_unknown",
        "dispatch_unknown",
        "expired",
        "human_cancel_unknown",
        "invalid_result",
        "invalid_result_type",
        "remote_id_integrity",
        "round_integrity",
        "round_limit",
        "settlement_conflict",
        "settlement_invalid",
        "state_binding",
        "state_integrity",
        "state_missing",
        "state_unavailable",
        "task_handler_missing",
        "unsupported_input_request",
        "update_unknown",
        # Subscription lifecycle diagnostics.  Provider strings and exception
        # class names are intentionally collapsed into these Host-owned codes.
        "runtime_restart",
        "subscription_connection_lost",
        "subscription_failure",
        "subscription_receive_failed",
        "subscription_start_failed",
    }
)
_METADATA_CODE_VALUES = MappingProxyType(
    {
        "dispatch_state": _DISPATCH_STATES,
        "last_error_code": _DIAGNOSTIC_CODES,
        "reason_code": _DIAGNOSTIC_CODES,
        "request_method": _REQUEST_METHODS,
        "retry_class": _RETRY_CLASSES,
    }
)


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or value != value.strip()
        or "\x00" in value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValidationError(f"{label} must be bounded canonical text")
    return value


def _optional_text(
    value: object,
    label: str,
    *,
    maximum: int = 512,
) -> str | None:
    return None if value is None else _text(value, label, maximum=maximum)


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _counter(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValidationError(f"{label} must be a {qualifier} exact integer")
    return value


def _timestamp(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _optional_timestamp(value: object, label: str) -> str | None:
    return None if value is None else _timestamp(value, label)


def _metadata(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValidationError(f"{label} must be a JSON object with string keys")
    unknown = sorted(set(value) - _METADATA_KEYS)
    if unknown:
        raise ValidationError(f"{label} contains unsupported fields: {unknown}")
    selected: dict[str, Any] = {}
    for key, item in value.items():
        if key == "automatic_retry_disabled":
            if type(item) is not bool or item is not True:
                raise ValidationError(
                    f"{label}.automatic_retry_disabled must be exactly true"
                )
            selected[key] = item
            continue
        if key == "input_schema_sha256":
            selected[key] = _sha256(item, f"{label}.{key}")
            continue
        allowed = _METADATA_CODE_VALUES.get(key)
        if allowed is None or type(item) is not str or item not in allowed:
            raise ValidationError(f"{label}.{key} is not an allowlisted diagnostic code")
        selected[key] = item
    encoded = json.dumps(
        selected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > MCP_V7_METADATA_MAX_BYTES:
        raise ValidationError(f"{label} exceeds the metadata byte limit")
    return MappingProxyType(selected)


def _side_effect_metadata(
    value: object,
    label: str,
    *,
    status: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValidationError(f"{label} must be a JSON object with string keys")
    cleanup_mode = value.get("cleanup_mode")
    if cleanup_mode not in _SIDE_EFFECT_CLEANUP_MODES:
        raise ValidationError(f"{label}.cleanup_mode is invalid")
    raw_retire_refs = value.get("retire_refs")
    if not isinstance(raw_retire_refs, (list, tuple)) or len(raw_retire_refs) > 2:
        raise ValidationError(f"{label}.retire_refs must contain at most two refs")
    retire_refs = tuple(
        sorted(
            _text(item, f"{label}.retire_refs", maximum=512)
            for item in raw_retire_refs
        )
    )
    if len(set(retire_refs)) != len(retire_refs):
        raise ValidationError(f"{label}.retire_refs must be unique")
    if status == "prepared" and cleanup_mode != "abort":
        raise ValidationError(f"{label} prepared rows require abort cleanup")
    retire_human_request_id = value.get("retire_human_request_id")
    retire_human_preview_sha256 = value.get("retire_human_preview_sha256")
    if (retire_human_request_id is None) != (
        retire_human_preview_sha256 is None
    ):
        raise ValidationError(f"{label} retire Human binding is incomplete")
    if retire_human_request_id is not None:
        retire_human_request_id = _text(
            retire_human_request_id,
            f"{label}.retire_human_request_id",
            maximum=512,
        )
        retire_human_preview_sha256 = _sha256(
            retire_human_preview_sha256,
            f"{label}.retire_human_preview_sha256",
        )
    base = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "cleanup_mode",
            "retire_refs",
            "retire_human_request_id",
            "retire_human_preview_sha256",
        }
    }
    selected = dict(_metadata(base, label))
    selected["cleanup_mode"] = cleanup_mode
    selected["retire_refs"] = retire_refs
    if retire_human_request_id is not None:
        selected["retire_human_request_id"] = retire_human_request_id
        selected["retire_human_preview_sha256"] = retire_human_preview_sha256
    encoded = json.dumps(
        selected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > MCP_V7_METADATA_MAX_BYTES:
        raise ValidationError(f"{label} exceeds the metadata byte limit")
    return MappingProxyType(selected)


def canonical_mcp_v7_metadata_json(value: Mapping[str, Any]) -> str:
    selected = _metadata(value, "MCP v7 metadata")
    return json.dumps(
        dict(selected),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_mcp_v7_metadata_json(value: object) -> Mapping[str, Any]:
    if type(value) is not str or len(value.encode("utf-8")) > MCP_V7_METADATA_MAX_BYTES:
        raise ValidationError("persisted MCP v7 metadata is invalid")
    try:
        decoded = json.loads(value)
    except (RecursionError, ValueError) as exc:
        raise ValidationError("persisted MCP v7 metadata is invalid") from exc
    selected = _metadata(decoded, "persisted MCP v7 metadata")
    if value != canonical_mcp_v7_metadata_json(selected):
        raise ValidationError("persisted MCP v7 metadata is not canonical")
    return selected


def canonical_mcp_v7_side_effect_metadata_json(value: Mapping[str, Any]) -> str:
    selected = _side_effect_metadata(value, "MCP v7 side-effect metadata")
    return json.dumps(
        dict(selected),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_mcp_v7_side_effect_metadata_json(value: object) -> Mapping[str, Any]:
    if type(value) is not str or len(value.encode("utf-8")) > MCP_V7_METADATA_MAX_BYTES:
        raise ValidationError("persisted MCP v7 side-effect metadata is invalid")
    try:
        decoded = json.loads(value)
    except (RecursionError, ValueError) as exc:
        raise ValidationError("persisted MCP v7 side-effect metadata is invalid") from exc
    selected = _side_effect_metadata(
        decoded,
        "persisted MCP v7 side-effect metadata",
    )
    if value != canonical_mcp_v7_side_effect_metadata_json(selected):
        raise ValidationError(
            "persisted MCP v7 side-effect metadata is not canonical"
        )
    return selected


class _McpV7Record:
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = dict(value) if isinstance(value, Mapping) else value
        return result


def _common_record(
    *,
    server_id: str,
    server_spec_sha256: str,
    server_generation: int,
    owner_id: str,
    auth_principal_sha256: str,
    auth_scope_sha256: str,
    revision: int,
    created_at: str,
    updated_at: str,
) -> tuple[str, str]:
    _text(server_id, "MCP server id", maximum=128)
    _sha256(server_spec_sha256, "MCP server spec")
    _counter(server_generation, "MCP server generation")
    _text(owner_id, "MCP owner id")
    _sha256(auth_principal_sha256, "MCP auth principal")
    _sha256(auth_scope_sha256, "MCP auth scope")
    _counter(revision, "MCP record revision")
    created = _timestamp(created_at, "MCP record created_at")
    updated = _timestamp(updated_at, "MCP record updated_at")
    if updated < created:
        raise ValidationError("MCP record updated_at precedes created_at")
    return created, updated


@dataclass(frozen=True, slots=True)
class McpContinuationRecord(_McpV7Record):
    continuation_id: str
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    request_sha256: str
    effect_id: str
    capability_sha256: str
    data_flow_sha256: str
    human_request_id: str
    broker_ref: str | None
    broker_value_sha256: str | None
    status: str
    revision: int
    expires_at: str
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.continuation_id, "MCP continuation id")
        created, updated = _common_record(
            server_id=self.server_id,
            server_spec_sha256=self.server_spec_sha256,
            server_generation=self.server_generation,
            owner_id=self.owner_id,
            auth_principal_sha256=self.auth_principal_sha256,
            auth_scope_sha256=self.auth_scope_sha256,
            revision=self.revision,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
        _sha256(self.request_sha256, "MCP continuation request")
        _text(self.effect_id, "MCP continuation effect id")
        _sha256(self.capability_sha256, "MCP continuation capability")
        _sha256(self.data_flow_sha256, "MCP continuation data-flow binding")
        _text(self.human_request_id, "MCP continuation Human request id")
        _optional_text(self.broker_ref, "MCP continuation broker ref")
        _optional_sha256(self.broker_value_sha256, "MCP continuation broker value")
        if (self.broker_ref is None) != (self.broker_value_sha256 is None):
            raise ValidationError("MCP continuation broker binding is incomplete")
        if self.status not in _CONTINUATION_STATUSES:
            raise ValidationError("MCP continuation status is invalid")
        expires = _timestamp(self.expires_at, "MCP continuation expires_at")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "MCP continuation metadata"))
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "expires_at", expires)


@dataclass(frozen=True, slots=True)
class McpRemoteTaskRecord(_McpV7Record):
    task_ref: str
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    origin_request_sha256: str
    origin_effect_id: str
    human_request_id: str | None
    broker_ref: str | None
    remote_id_sha256: str
    status: str
    revision: int
    expires_at: str | None
    poll_interval_ms: int | None
    status_message_sha256: str | None
    result_ref: str | None
    result_sha256: str | None
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.task_ref, "MCP remote task ref")
        created, updated = _common_record(
            server_id=self.server_id,
            server_spec_sha256=self.server_spec_sha256,
            server_generation=self.server_generation,
            owner_id=self.owner_id,
            auth_principal_sha256=self.auth_principal_sha256,
            auth_scope_sha256=self.auth_scope_sha256,
            revision=self.revision,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
        _sha256(self.origin_request_sha256, "MCP remote task origin request")
        _text(self.origin_effect_id, "MCP remote task origin effect id")
        _optional_text(self.human_request_id, "MCP remote task Human request id")
        _optional_text(self.broker_ref, "MCP remote task broker ref")
        _sha256(self.remote_id_sha256, "MCP remote task remote id")
        if self.status not in _REMOTE_TASK_STATUSES:
            raise ValidationError("MCP remote task status is invalid")
        if self.status == "input_required" and self.human_request_id is None:
            raise ValidationError(
                "MCP input-required remote task requires a Human request id"
            )
        expires = _optional_timestamp(self.expires_at, "MCP remote task expires_at")
        if self.poll_interval_ms is not None:
            _counter(self.poll_interval_ms, "MCP remote task poll interval")
        _optional_sha256(self.status_message_sha256, "MCP remote task status message")
        _optional_text(self.result_ref, "MCP remote task result ref")
        _optional_sha256(self.result_sha256, "MCP remote task result")
        if (self.result_ref is None) != (self.result_sha256 is None):
            raise ValidationError("MCP remote task result binding is incomplete")
        if self.broker_ref is not None and self.broker_ref == self.result_ref:
            raise ValidationError("MCP remote task broker slots must be distinct")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "MCP remote task metadata"))
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "expires_at", expires)


@dataclass(frozen=True, slots=True)
class McpSubscriptionRecord(_McpV7Record):
    subscription_id: str
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    requested_filter_sha256: str
    acknowledged_filter_sha256: str | None
    status: str
    queue_limit: int
    event_max_bytes: int
    received_count: int
    dropped_count: int
    revision: int
    last_event_at: str | None
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.subscription_id, "MCP subscription id")
        created, updated = _common_record(
            server_id=self.server_id,
            server_spec_sha256=self.server_spec_sha256,
            server_generation=self.server_generation,
            owner_id=self.owner_id,
            auth_principal_sha256=self.auth_principal_sha256,
            auth_scope_sha256=self.auth_scope_sha256,
            revision=self.revision,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
        _sha256(self.requested_filter_sha256, "MCP requested subscription filter")
        _optional_sha256(
            self.acknowledged_filter_sha256,
            "MCP acknowledged subscription filter",
        )
        if self.status not in _SUBSCRIPTION_STATUSES:
            raise ValidationError("MCP subscription status is invalid")
        _counter(self.queue_limit, "MCP subscription queue limit", positive=True)
        _counter(self.event_max_bytes, "MCP subscription event byte limit", positive=True)
        _counter(self.received_count, "MCP subscription received count")
        _counter(self.dropped_count, "MCP subscription dropped count")
        if self.dropped_count > self.received_count:
            raise ValidationError("MCP subscription dropped count exceeds received count")
        last_event = _optional_timestamp(self.last_event_at, "MCP subscription last event")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "MCP subscription metadata"))
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "last_event_at", last_event)


@dataclass(frozen=True, slots=True)
class McpAuthMetadataRecord(_McpV7Record):
    profile_id: str
    server_id: str
    server_spec_sha256: str
    server_generation: int
    status: str
    issuer_sha256: str | None
    resource_sha256: str | None
    audience_sha256: str | None
    scopes_sha256: str
    principal_sha256: str | None
    expires_at: str | None
    credential_generation: int
    revision: int
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.profile_id, "MCP auth profile id")
        _text(self.server_id, "MCP auth server id", maximum=128)
        _sha256(self.server_spec_sha256, "MCP auth server spec")
        _counter(self.server_generation, "MCP auth server generation")
        if self.status not in _AUTH_STATUSES:
            raise ValidationError("MCP auth status is invalid")
        _optional_sha256(self.issuer_sha256, "MCP auth issuer")
        _optional_sha256(self.resource_sha256, "MCP auth resource")
        _optional_sha256(self.audience_sha256, "MCP auth audience")
        _sha256(self.scopes_sha256, "MCP auth scopes")
        _optional_sha256(self.principal_sha256, "MCP auth principal")
        expires = _optional_timestamp(self.expires_at, "MCP auth expires_at")
        _counter(self.credential_generation, "MCP credential generation")
        _counter(self.revision, "MCP auth revision")
        created = _timestamp(self.created_at, "MCP auth created_at")
        updated = _timestamp(self.updated_at, "MCP auth updated_at")
        if updated < created:
            raise ValidationError("MCP auth updated_at precedes created_at")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "MCP auth metadata"))
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "expires_at", expires)


@dataclass(frozen=True, slots=True)
class McpSideEffectPreparationRecord(_McpV7Record):
    """Durable ownership of preallocated Human and broker side-effect slots.

    The referenced Human row or broker value may not exist yet: this record is
    intentionally committed *before* either side effect.  It therefore stores
    only preallocated local identifiers and SHA-256 commitments, never the
    Human payload or broker value.
    """

    preparation_id: str
    operation_kind: str
    operation_id: str
    operation_revision: int | None
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    human_request_id: str | None
    human_preview_sha256: str | None
    broker_ref: str | None
    broker_value_sha256: str | None
    result_ref: str | None
    result_sha256: str | None
    status: str
    revision: int
    expires_at: str
    metadata: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.preparation_id, "MCP side-effect preparation id")
        if self.operation_kind not in _SIDE_EFFECT_OPERATION_KINDS:
            raise ValidationError("MCP side-effect preparation kind is invalid")
        _text(self.operation_id, "MCP side-effect operation id")
        if self.operation_revision is not None:
            _counter(
                self.operation_revision,
                "MCP side-effect operation revision",
            )
        created, updated = _common_record(
            server_id=self.server_id,
            server_spec_sha256=self.server_spec_sha256,
            server_generation=self.server_generation,
            owner_id=self.owner_id,
            auth_principal_sha256=self.auth_principal_sha256,
            auth_scope_sha256=self.auth_scope_sha256,
            revision=self.revision,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
        _optional_text(
            self.human_request_id,
            "MCP side-effect planned Human request id",
        )
        _optional_sha256(
            self.human_preview_sha256,
            "MCP side-effect Human preview",
        )
        if (self.human_request_id is None) != (self.human_preview_sha256 is None):
            raise ValidationError(
                "MCP side-effect planned Human binding is incomplete"
            )
        _optional_text(self.broker_ref, "MCP side-effect broker ref")
        _optional_sha256(
            self.broker_value_sha256,
            "MCP side-effect broker value",
        )
        if (self.broker_ref is None) != (self.broker_value_sha256 is None):
            raise ValidationError("MCP side-effect broker binding is incomplete")
        _optional_text(self.result_ref, "MCP side-effect result ref")
        _optional_sha256(self.result_sha256, "MCP side-effect result value")
        if (self.result_ref is None) != (self.result_sha256 is None):
            raise ValidationError("MCP side-effect result binding is incomplete")
        if self.status not in _SIDE_EFFECT_PREPARATION_STATUSES:
            raise ValidationError("MCP side-effect preparation status is invalid")
        metadata = _side_effect_metadata(
            self.metadata,
            "MCP side-effect preparation metadata",
            status=self.status,
        )
        if (
            self.human_request_id is None
            and self.broker_ref is None
            and self.result_ref is None
            and not metadata.get("retire_refs")
            and metadata.get("retire_human_request_id") is None
        ):
            raise ValidationError(
                "MCP side-effect preparation must own at least one side-effect slot"
            )
        if self.broker_ref is not None and self.broker_ref == self.result_ref:
            raise ValidationError("MCP side-effect broker slots must be distinct")
        expires = _timestamp(
            self.expires_at,
            "MCP side-effect preparation expires_at",
        )
        object.__setattr__(
            self,
            "metadata",
            metadata,
        )
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "expires_at", expires)


__all__ = [
    "MCP_V7_METADATA_MAX_BYTES",
    "MCP_V7_QUERY_HARD_LIMIT",
    "McpAuthMetadataRecord",
    "McpContinuationRecord",
    "McpRemoteTaskRecord",
    "McpSideEffectPreparationRecord",
    "McpSubscriptionRecord",
    "canonical_mcp_v7_metadata_json",
    "canonical_mcp_v7_side_effect_metadata_json",
    "parse_mcp_v7_metadata_json",
    "parse_mcp_v7_side_effect_metadata_json",
]
