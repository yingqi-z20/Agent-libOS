from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import (
    MachinePolicySettlementV1,
    SemanticControlStateV1,
    SemanticPolicyEpochV1,
    SemanticReasonCode,
)
from agent_libos.utils.serde import bounded_json_loads


SEMANTIC_V6_QUERY_HARD_LIMIT = 500
SEMANTIC_FLOW_LINEAGE_HARD_LIMIT = 1_000
SEMANTIC_V6_RECORD_MAX_BYTES = 256 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FLOW_ENTITY_KINDS = frozenset(
    {
        "root_goal",
        "object_version",
        "file_binding_version",
        "provider_result",
        "tool_result",
        "materialization",
        "model_output",
    }
)
_FLOW_ACTIVITY_KINDS = frozenset(
    {
        "process_spawn",
        "provider_call",
        "tool_call",
        "llm_call",
        "object_create",
        "object_update",
        "object_append",
        "object_materialize",
        "object_read",
        "file_read",
        "file_write",
        "transformation",
        "aggregation",
        "conditional",
        "tool_selection",
        "memory_retrieval",
    }
)
_FLOW_NODE_TYPES = frozenset({"entity", "activity"})
_FLOW_EDGE_RELATIONS = frozenset({"direct", "indirect", "control"})
_FLOW_LABEL_SOURCES = frozenset({"host", "model", "deterministic"})
_FLOW_LOCATOR_KINDS = frozenset({"json_field", "text_chunk"})
_FLOW_COVERAGE = frozenset(
    {"complete", "partial", "unknown", "conflict", "stale"}
)
_FLOW_SENSITIVITY = frozenset(
    {"public", "normal", "confidential", "restricted", "secret"}
)
_FLOW_TRUST = frozenset(
    {"untrusted", "unknown", "user_asserted", "verified", "trusted"}
)
_FLOW_INTEGRITY = frozenset({"untrusted", "unknown", "checked", "verified"})
_CONTROL_MODES = frozenset({"off", "shadow", "enforce_deny", "canary_auto"})
_SETTLEMENT_OUTCOMES = frozenset(
    {
        "issued",
        "denied",
        "require_human",
        "race_lost",
        "stale",
        "budget_exhausted",
        "revoked",
        "expired",
        "failed",
    }
)
_REVIEW_OUTCOMES = frozenset({"safe", "unsafe", "inconclusive"})
_HUMAN_OUTCOMES = frozenset({"approved", "rejected", "cancelled"})
_HUMAN_OUTCOME_SOURCES = frozenset({"human", "machine_policy", "cancel"})
_MACHINE_LIFECYCLE_OUTCOMES = frozenset(
    {
        "issued",
        "consumed",
        "succeeded",
        "failed",
        "outcome_unknown",
        "expired",
        "revoked",
        "race_lost",
    }
)
_ROLLOUT_SCOPE_KEYS = frozenset(
    {
        "schema_version",
        "tenant_bucket_sha256s",
        "auto_approval_rules",
        "hard_deny_rules",
        "allow_parameters",
    }
)
_ROLLOUT_SCOPE_V1_KEYS = frozenset(
    {"schema_version", "tenant_bucket_sha256s", "auto_approval_rules"}
)
_ROLLOUT_RULE_KEYS = frozenset(
    {
        "rule_id_sha256",
        "action_id",
        "rights",
        "resource_kind",
        "match_sha256",
        "covering_prefix_sha256s",
    }
)
_ROLLOUT_ACTION_RIGHTS = {
    "filesystem.read": ("read",),
    "git.read": ("read",),
    "git.diff": ("diff",),
}
_ROLLOUT_ALLOW_PARAMETER_KEYS = frozenset(
    {
        "catalog_version",
        "classifier_profile_id_sha256",
        "classifier_profile_sha256",
        "classifier_model_sha256",
        "minimum_confidence_bps",
        "required_calibration_bucket",
        "capability_ttl_s",
        "per_rule_per_minute_limit",
        "per_rule_per_day_limit",
        "max_inflight",
    }
)
_ROLLOUT_DENY_RIGHTS = frozenset(
    {"read", "write", "execute", "link", "diff", "materialize", "delete"}
)
_ROLLOUT_ACTION_ID = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z"
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "argv",
        "body",
        "command",
        "content",
        "credential",
        "file_content",
        "message",
        "path",
        "prompt",
        "raw_response",
        "reasoning",
        "response",
        "secret",
        "text",
    }
)
_SENSITIVE_STRING_MARKER = re.compile(
    r"(?:credential|password|passwd|private[_ -]?key|secret|bearer|api[_ -]?key)",
    re.IGNORECASE,
)


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
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
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


def _flow_path_sha256s(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("flow assertion path digests must be a sequence")
    selected = tuple(value)
    if len(selected) > 32:
        raise ValidationError("flow assertion path digest depth is invalid")
    for digest in selected:
        _sha256(digest, "flow assertion path segment")
    return selected


def _absent_flow_locator_carrier(
    *,
    locator_kind: str | None,
    path_sha256s: tuple[str, ...],
    value_sha256: str | None,
    coordinates: tuple[int | None, int | None, int | None],
) -> None:
    if (
        locator_kind is not None
        or path_sha256s
        or value_sha256 is not None
        or any(value is not None for value in coordinates)
    ):
        raise ValidationError("flow assertion locator carrier requires locator_sha256")


def _json_flow_locator_carrier(
    *,
    path_sha256s: tuple[str, ...],
    value_sha256: str | None,
    coordinates: tuple[int | None, int | None, int | None],
) -> None:
    if (
        not path_sha256s
        or value_sha256 is None
        or any(value is not None for value in coordinates)
    ):
        raise ValidationError("JSON flow assertion locator carrier is incomplete")


def _text_flow_locator_carrier(
    *,
    path_sha256s: tuple[str, ...],
    value_sha256: str | None,
    coordinates: tuple[int | None, int | None, int | None],
) -> None:
    if path_sha256s or value_sha256 is None or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in coordinates
    ):
        raise ValidationError("text flow assertion locator carrier is incomplete")
    _ordinal, offset_start, offset_end = coordinates
    assert offset_start is not None and offset_end is not None
    if offset_end <= offset_start:
        raise ValidationError("text flow assertion locator end must exceed start")


def _flow_locator_carrier(
    *,
    locator_sha256: str | None,
    locator_kind: str | None,
    path_sha256s: object,
    value_sha256: str | None,
    coordinates: tuple[int | None, int | None, int | None],
) -> tuple[str, ...]:
    _optional_sha256(locator_sha256, "flow assertion locator")
    if locator_kind is not None and locator_kind not in _FLOW_LOCATOR_KINDS:
        raise ValidationError("flow assertion locator kind is invalid")
    selected_path = _flow_path_sha256s(path_sha256s)
    _optional_sha256(value_sha256, "flow assertion locator value")
    if locator_sha256 is None:
        _absent_flow_locator_carrier(
            locator_kind=locator_kind,
            path_sha256s=selected_path,
            value_sha256=value_sha256,
            coordinates=coordinates,
        )
    elif locator_kind == "json_field":
        _json_flow_locator_carrier(
            path_sha256s=selected_path,
            value_sha256=value_sha256,
            coordinates=coordinates,
        )
    elif locator_kind == "text_chunk":
        _text_flow_locator_carrier(
            path_sha256s=selected_path,
            value_sha256=value_sha256,
            coordinates=coordinates,
        )
    else:
        raise ValidationError("flow assertion locator carrier is incomplete")
    return selected_path


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _counter(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative exact integer")
    return value


def _strict_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = bounded_json_loads(encoded, max_bytes=SEMANTIC_V6_RECORD_MAX_BYTES)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be bounded strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ValidationError(f"{label} must be a JSON object")
    pending: list[Any] = [decoded]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if str(key).strip().casefold() in _FORBIDDEN_PAYLOAD_KEYS:
                    raise ValidationError(f"{label} contains a forbidden payload field")
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str):
            # Evidence JSON is never a payload escape hatch.  Canonical storage
            # columns carry identifiers and digests; these small auxiliary
            # objects may contain only payload-free metadata.  Obvious secret
            # material and paths are rejected regardless of their key.
            if (
                len(current) > 512
                or _SENSITIVE_STRING_MARKER.search(current) is not None
                or "/" in current
                or "\\" in current
                or any(ord(character) < 0x20 for character in current)
            ):
                raise ValidationError(f"{label} contains unsafe free-form text")
    return MappingProxyType(decoded)


def _flow_lattice_labels(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Canonicalize the closed label lattice without treating `secret` as payload."""

    if not isinstance(value, Mapping) or set(value) != {
        "sensitivity",
        "trust_level",
        "integrity",
    }:
        raise ValidationError(
            "flow entity labels must contain only the canonical lattice vector"
        )
    selected = dict(value)
    if selected.get("sensitivity") not in _FLOW_SENSITIVITY:
        raise ValidationError("flow entity sensitivity is invalid")
    if selected.get("trust_level") not in _FLOW_TRUST:
        raise ValidationError("flow entity trust level is invalid")
    if selected.get("integrity") not in _FLOW_INTEGRITY:
        raise ValidationError("flow entity integrity is invalid")
    try:
        encoded = json.dumps(
            selected,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = bounded_json_loads(encoded, max_bytes=1_024)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError("flow entity labels must be bounded strict JSON") from exc
    return MappingProxyType(decoded)


def canonical_record_json(value: Mapping[str, Any]) -> str:
    selected = _strict_mapping(
        _mutable_json_value(value),
        "semantic v6 record",
    )
    return json.dumps(
        dict(selected),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_flow_labels_json(value: Mapping[str, Any]) -> str:
    selected = _flow_lattice_labels(value)
    return json.dumps(
        dict(selected),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_flow_path_sha256s_json(value: object) -> str:
    selected = _flow_path_sha256s(value)
    return json.dumps(
        list(selected),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def decode_flow_path_sha256s_json(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValidationError("persisted flow assertion path digests are invalid")
    try:
        decoded = bounded_json_loads(value, max_bytes=4_096)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "persisted flow assertion path digests are invalid"
        ) from exc
    if not isinstance(decoded, list):
        raise ValidationError("persisted flow assertion path digests are invalid")
    selected = _flow_path_sha256s(decoded)
    if value != canonical_flow_path_sha256s_json(selected):
        raise ValidationError(
            "persisted flow assertion path digests are not canonical"
        )
    return selected


def semantic_v6_record_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_record_json(value).encode("utf-8")).hexdigest()


class _SemanticV6Record:
    def to_dict(self) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            selected[item.name] = _mutable_json_value(value)
        return selected


def _mutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _mutable_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return value


def _immutable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _immutable_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_json_value(item) for item in value)
    return value


def _rollout_scope(value: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(_strict_mapping(value, "semantic rollout scope"))
    scope_version = selected.get("schema_version")
    legacy_v1 = scope_version == 1 and set(selected) == _ROLLOUT_SCOPE_V1_KEYS
    current_v2 = scope_version == 2 and set(selected) == _ROLLOUT_SCOPE_KEYS
    if not legacy_v1 and not current_v2:
        raise ValidationError("semantic rollout scope contract is invalid")
    _validate_rollout_tenants(selected.get("tenant_bucket_sha256s"))
    auto_rules = _validated_rollout_rules(
        selected.get("auto_approval_rules"),
        rule_kind="auto-approval",
        catalog_rule=True,
    )
    if legacy_v1:
        # v1 records predate hard-deny and allow-parameter commitments.  They
        # remain readable as immutable history, while the control comparator
        # rejects them as insufficient evidence for any later rotation.
        return _immutable_json_value(selected)
    deny_rules = _validated_rollout_rules(
        selected.get("hard_deny_rules"),
        rule_kind="hard-deny",
        catalog_rule=False,
    )
    all_rules = [*auto_rules, *deny_rules]
    if len({rule["rule_id_sha256"] for rule in all_rules}) != len(all_rules):
        raise ValidationError("semantic rollout rule ids must be unique")
    _validate_rollout_allow_parameters(selected.get("allow_parameters"))
    return _immutable_json_value(selected)


def _validate_rollout_tenants(value: object) -> None:
    if (
        not isinstance(value, list)
        or len(value) > 1_024
        or value != sorted(set(value))
    ):
        raise ValidationError("semantic rollout tenant scope is invalid")
    for tenant in value:
        _sha256(tenant, "semantic rollout tenant bucket")


def _validated_rollout_rules(
    value: object,
    *,
    rule_kind: str,
    catalog_rule: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 1_024:
        raise ValidationError(
            f"semantic rollout {rule_kind} rule scope is invalid"
        )
    for rule in value:
        _validate_rollout_rule(rule, catalog_rule=catalog_rule)
    if value != _canonical_rollout_rules(value):
        raise ValidationError(
            f"semantic rollout {rule_kind} rules must be canonically ordered"
        )
    if len({rule["rule_id_sha256"] for rule in value}) != len(value):
        raise ValidationError("semantic rollout rule ids must be unique")
    return value


def _canonical_rollout_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rules,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _validate_rollout_rule(value: object, *, catalog_rule: bool) -> None:
    if not isinstance(value, dict) or set(value) != _ROLLOUT_RULE_KEYS:
        raise ValidationError("semantic rollout rule contract is invalid")
    _sha256(value.get("rule_id_sha256"), "semantic rollout rule id")
    action_id = _text(
        value.get("action_id"), "semantic rollout action", maximum=128
    )
    if _ROLLOUT_ACTION_ID.fullmatch(action_id) is None:
        raise ValidationError("semantic rollout action is not exact")
    if catalog_rule and action_id not in _ROLLOUT_ACTION_RIGHTS:
        raise ValidationError("semantic rollout action is outside catalog v1")
    if value.get("resource_kind") not in {"exact", "prefix"}:
        raise ValidationError("semantic rollout resource kind is invalid")
    _sha256(value.get("match_sha256"), "semantic rollout resource match")
    for key, label in (
        ("rights", "semantic rollout rights"),
        ("covering_prefix_sha256s", "semantic rollout covering prefixes"),
    ):
        items = value.get(key)
        if (
            not isinstance(items, list)
            or len(items) > 256
            or items != sorted(set(items))
        ):
            raise ValidationError(f"{label} must be a canonical bounded set")
        if key == "rights" and (
            not items
            or (
                catalog_rule
                and tuple(items) != _ROLLOUT_ACTION_RIGHTS[action_id]
            )
            or (
                not catalog_rule
                and not set(items).issubset(_ROLLOUT_DENY_RIGHTS)
            )
        ):
            raise ValidationError(f"{label} are invalid for the rule kind")
        for item in items:
            if key == "rights":
                _text(item, label, maximum=64)
            else:
                _sha256(item, label)


def _validate_rollout_allow_parameters(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _ROLLOUT_ALLOW_PARAMETER_KEYS:
        raise ValidationError("semantic rollout allow parameters are invalid")
    if value.get("catalog_version") != 1 or type(value.get("catalog_version")) is not int:
        raise ValidationError("semantic rollout catalog version is invalid")
    for field in (
        "classifier_profile_id_sha256",
        "classifier_profile_sha256",
        "classifier_model_sha256",
    ):
        _optional_sha256(value.get(field), f"semantic rollout {field}")
    identity = (
        value.get("classifier_profile_id_sha256"),
        value.get("classifier_profile_sha256"),
        value.get("classifier_model_sha256"),
    )
    if any(item is None for item in identity) and any(
        item is not None for item in identity
    ):
        raise ValidationError("semantic rollout classifier identity is incomplete")
    if value.get("required_calibration_bucket") != "very_high":
        raise ValidationError("semantic rollout calibration bucket is invalid")
    bounded_parameters = {
        "minimum_confidence_bps": (9_900, 10_000),
        "capability_ttl_s": (1, 300),
        "per_rule_per_minute_limit": (1, 10),
        "per_rule_per_day_limit": (1, 100),
        "max_inflight": (1, 2),
    }
    for field, (minimum, maximum) in bounded_parameters.items():
        item = value.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            raise ValidationError(f"semantic rollout {field} is invalid")


@dataclass(frozen=True, slots=True)
class SemanticV6Cursor(_SemanticV6Record):
    created_at: str
    record_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "cursor timestamp"))
        _text(self.record_id, "cursor record id")


@dataclass(frozen=True, slots=True)
class SemanticFlowEntityRecord(_SemanticV6Record):
    entity_id: str
    kind: str
    pid: str | None
    tenant_bucket_sha256: str
    content_sha256: str
    version_sha256: str
    provenance_sha256: str
    baseline_labels: Mapping[str, Any]
    identity_present: bool
    identity_mixed: bool
    coverage: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.entity_id, "flow entity id")
        if self.kind not in _FLOW_ENTITY_KINDS:
            raise ValidationError("flow entity kind is invalid")
        _optional_text(self.pid, "flow entity pid")
        for value, label in (
            (self.tenant_bucket_sha256, "flow entity tenant bucket"),
            (self.content_sha256, "flow entity content"),
            (self.version_sha256, "flow entity version"),
            (self.provenance_sha256, "flow entity provenance"),
        ):
            _sha256(value, label)
        if type(self.identity_present) is not bool or type(self.identity_mixed) is not bool:
            raise ValidationError("flow entity identity flags must be booleans")
        if self.identity_mixed and not self.identity_present:
            raise ValidationError("mixed flow identity requires identity presence")
        if self.coverage not in _FLOW_COVERAGE:
            raise ValidationError("flow entity coverage is invalid")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("flow entity schema version must be 1")
        labels = _flow_lattice_labels(self.baseline_labels)
        object.__setattr__(self, "baseline_labels", labels)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "flow entity created_at"))


@dataclass(frozen=True, slots=True)
class SemanticFlowActivityRecord(_SemanticV6Record):
    activity_id: str
    kind: str
    pid: str
    action_id: str | None
    effect_id: str | None
    state_sha256: str
    provider_spec_sha256: str | None
    tool_schema_sha256: str | None
    model_artifact_sha256: str | None
    tenant_bucket_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.activity_id, "flow activity id")
        if self.kind not in _FLOW_ACTIVITY_KINDS:
            raise ValidationError("flow activity kind is invalid")
        _text(self.pid, "flow activity pid")
        _optional_text(self.action_id, "flow activity action")
        _optional_text(self.effect_id, "flow activity effect")
        _sha256(self.state_sha256, "flow activity state")
        _sha256(self.tenant_bucket_sha256, "flow activity tenant bucket")
        for value, label in (
            (self.provider_spec_sha256, "flow activity provider"),
            (self.tool_schema_sha256, "flow activity tool"),
            (self.model_artifact_sha256, "flow activity model"),
        ):
            _optional_sha256(value, label)
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("flow activity schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "flow activity created_at"))


@dataclass(frozen=True, slots=True)
class SemanticFlowEdgeRecord(_SemanticV6Record):
    edge_id: str
    relation: str
    source_node_id: str
    source_node_type: str
    target_node_id: str
    target_node_type: str
    pid: str
    provenance_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.edge_id, "flow edge id")
        if self.relation not in _FLOW_EDGE_RELATIONS:
            raise ValidationError("flow edge relation is invalid")
        _text(self.source_node_id, "flow edge source")
        _text(self.target_node_id, "flow edge target")
        if self.source_node_type not in _FLOW_NODE_TYPES or self.target_node_type not in _FLOW_NODE_TYPES:
            raise ValidationError("flow edge node type is invalid")
        if self.source_node_id == self.target_node_id and self.source_node_type == self.target_node_type:
            raise ValidationError("flow edge cannot be a self-loop")
        _text(self.pid, "flow edge pid")
        _sha256(self.provenance_sha256, "flow edge provenance")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("flow edge schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "flow edge created_at"))


@dataclass(frozen=True, slots=True)
class SemanticFlowLabelAssertionRecord(_SemanticV6Record):
    assertion_id: str
    entity_id: str
    source: str
    sensitivity_floor: str
    integrity_ceiling: str
    trust_ceiling: str
    evidence_sha256: str
    assessment_id: str | None
    locator_sha256: str | None
    category: str | None
    coverage: str
    created_at: str
    locator_kind: str | None = None
    path_sha256s: tuple[str, ...] = ()
    value_sha256: str | None = None
    ordinal: int | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.assertion_id, "flow assertion id")
        _text(self.entity_id, "flow assertion entity")
        if self.source not in _FLOW_LABEL_SOURCES:
            raise ValidationError("flow assertion source is invalid")
        for value, label in (
            (self.sensitivity_floor, "flow assertion sensitivity"),
            (self.integrity_ceiling, "flow assertion integrity"),
            (self.trust_ceiling, "flow assertion trust"),
        ):
            _text(value, label, maximum=64)
        _sha256(self.evidence_sha256, "flow assertion evidence")
        _optional_text(self.assessment_id, "flow assertion assessment")
        selected_path = _flow_locator_carrier(
            locator_sha256=self.locator_sha256,
            locator_kind=self.locator_kind,
            path_sha256s=self.path_sha256s,
            value_sha256=self.value_sha256,
            coordinates=(self.ordinal, self.offset_start, self.offset_end),
        )
        object.__setattr__(self, "path_sha256s", selected_path)
        _optional_text(self.category, "flow assertion category", maximum=128)
        if self.coverage not in _FLOW_COVERAGE:
            raise ValidationError("flow assertion coverage is invalid")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("flow assertion schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "flow assertion created_at"))


@dataclass(frozen=True, slots=True)
class SemanticLegacyCoverageRecord(_SemanticV6Record):
    singleton: int
    source_schema_version: int
    assessment_count: int
    coverage: str
    evidence_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.singleton) is not int or self.singleton != 1:
            raise ValidationError("semantic legacy coverage singleton must be 1")
        if (
            type(self.source_schema_version) is not int
            or self.source_schema_version != 5
        ):
            raise ValidationError(
                "semantic legacy coverage source schema version must be 5"
            )
        _counter(self.assessment_count, "semantic legacy assessment count")
        if self.coverage != "unknown":
            raise ValidationError("semantic legacy coverage must be unknown")
        _sha256(self.evidence_sha256, "semantic legacy coverage evidence")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic legacy coverage schema version must be 1")
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, "semantic legacy coverage created_at"),
        )


@dataclass(frozen=True, slots=True)
class SemanticHumanOutcomeLinkRecord(_SemanticV6Record):
    link_id: str
    request_id: str
    request_revision: int
    pid: str
    assessment_id: str | None
    job_id: str | None
    settlement_id: str | None
    outcome: str
    source: str
    decision_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.link_id, "semantic Human outcome link id")
        _text(self.request_id, "semantic Human outcome request id")
        _counter(self.request_revision, "semantic Human outcome request revision")
        _text(self.pid, "semantic Human outcome pid")
        _optional_text(self.assessment_id, "semantic Human outcome assessment id")
        _optional_text(self.job_id, "semantic Human outcome job id")
        _optional_text(self.settlement_id, "semantic Human outcome settlement id")
        if self.outcome not in _HUMAN_OUTCOMES:
            raise ValidationError("semantic Human outcome is invalid")
        if self.source not in _HUMAN_OUTCOME_SOURCES:
            raise ValidationError("semantic Human outcome source is invalid")
        if (self.source == "cancel") != (self.outcome == "cancelled"):
            raise ValidationError(
                "semantic Human cancellation outcome/source is inconsistent"
            )
        _sha256(self.decision_sha256, "semantic Human outcome decision")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic Human outcome schema version must be 1")
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, "semantic Human outcome created_at"),
        )


@dataclass(frozen=True, slots=True)
class SemanticFlowBundle:
    entities: tuple[SemanticFlowEntityRecord, ...] = ()
    activities: tuple[SemanticFlowActivityRecord, ...] = ()
    edges: tuple[SemanticFlowEdgeRecord, ...] = ()
    assertions: tuple[SemanticFlowLabelAssertionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [value.to_dict() for value in self.entities],
            "activities": [value.to_dict() for value in self.activities],
            "edges": [value.to_dict() for value in self.edges],
            "assertions": [value.to_dict() for value in self.assertions],
        }


@dataclass(frozen=True, slots=True)
class SemanticFlowPage:
    records: tuple[Any, ...]
    next_cursor: SemanticV6Cursor | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [value.to_dict() for value in self.records],
            "next_cursor": (
                self.next_cursor.to_dict() if self.next_cursor is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SemanticPolicyEpochRecord(_SemanticV6Record):
    epoch_id: str
    generation: int
    catalog_version: int
    policy_sha256: str
    expected_previous_sha256: str | None
    rollout_scope: Mapping[str, Any]
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.epoch_id, "semantic policy epoch id")
        _counter(self.generation, "semantic policy generation")
        if self.generation == 0:
            raise ValidationError("semantic policy generation must be positive")
        if type(self.catalog_version) is not int or self.catalog_version != 1:
            raise ValidationError("semantic action catalog version must be 1")
        _sha256(self.policy_sha256, "semantic policy")
        _optional_sha256(self.expected_previous_sha256, "semantic previous policy")
        object.__setattr__(self, "rollout_scope", _rollout_scope(self.rollout_scope))
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic policy epoch schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "semantic policy created_at"))


@dataclass(frozen=True, slots=True)
class SemanticControlStateRecord(_SemanticV6Record):
    revision: int
    generation: int
    mode: str
    active_epoch_id: str | None
    active_policy_sha256: str | None
    tripped: bool
    trip_code: str | None
    updated_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _counter(self.revision, "semantic control revision")
        _counter(self.generation, "semantic control generation")
        if self.mode not in _CONTROL_MODES:
            raise ValidationError("semantic control mode is invalid")
        _optional_text(self.active_epoch_id, "semantic active epoch")
        _optional_sha256(self.active_policy_sha256, "semantic active policy")
        if (self.active_epoch_id is None) != (self.active_policy_sha256 is None):
            raise ValidationError("semantic control epoch identity must be complete")
        if type(self.tripped) is not bool:
            raise ValidationError("semantic control tripped must be a boolean")
        _optional_text(self.trip_code, "semantic trip code", maximum=128)
        if self.tripped != (self.trip_code is not None):
            raise ValidationError("semantic trip state and code must agree")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic control schema version must be 1")
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "semantic control updated_at"))


@dataclass(frozen=True, slots=True)
class SemanticControlTransitionRecord(_SemanticV6Record):
    transition_id: str
    revision: int
    generation: int
    mode: str
    active_epoch_id: str | None
    active_policy_sha256: str | None
    tripped: bool
    trip_code: str | None
    evidence_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.transition_id, "semantic control transition id")
        state = SemanticControlStateRecord(
            revision=self.revision,
            generation=self.generation,
            mode=self.mode,
            active_epoch_id=self.active_epoch_id,
            active_policy_sha256=self.active_policy_sha256,
            tripped=self.tripped,
            trip_code=self.trip_code,
            updated_at=self.created_at,
            schema_version=self.schema_version,
        )
        _sha256(self.evidence_sha256, "semantic control transition evidence")
        object.__setattr__(self, "created_at", state.updated_at)


@dataclass(frozen=True, slots=True)
class SemanticMachineSettlementRecord(_SemanticV6Record):
    settlement_id: str
    assessment_id: str | None
    job_id: str | None
    request_id: str
    request_revision: int
    pid: str
    operation_id: str | None
    effect_id: str
    epoch_id: str
    policy_sha256: str
    tenant_bucket_sha256: str
    action_id: str
    outcome: str
    capability_id: str | None
    binding_sha256: str
    decision_sha256: str
    matched_rule_id: str | None
    reason_codes: tuple[str, ...]
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, label in (
            (self.settlement_id, "machine settlement id"),
            (self.request_id, "machine settlement request"),
            (self.pid, "machine settlement pid"),
            (self.effect_id, "machine settlement effect"),
            (self.epoch_id, "machine settlement epoch"),
            (self.action_id, "machine settlement action"),
        ):
            _text(value, label)
        _optional_text(self.assessment_id, "machine settlement assessment")
        _optional_text(self.job_id, "machine settlement job")
        _optional_text(self.operation_id, "machine settlement operation")
        _optional_text(self.capability_id, "machine settlement capability")
        _counter(self.request_revision, "machine settlement request revision")
        for value, label in (
            (self.policy_sha256, "machine settlement policy"),
            (self.tenant_bucket_sha256, "machine settlement tenant"),
            (self.binding_sha256, "machine settlement binding"),
            (self.decision_sha256, "machine settlement decision"),
        ):
            _sha256(value, label)
        _optional_text(self.matched_rule_id, "machine settlement matched rule")
        if self.outcome not in _SETTLEMENT_OUTCOMES:
            raise ValidationError("machine settlement outcome is invalid")
        if (self.outcome == "issued") != (self.capability_id is not None):
            raise ValidationError("only an issued settlement may bind a capability")
        if not isinstance(self.reason_codes, tuple) or len(self.reason_codes) > 32:
            raise ValidationError("machine settlement reason codes must be a bounded tuple")
        try:
            selected_reasons = tuple(
                SemanticReasonCode(value).value for value in self.reason_codes
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("machine settlement reason code is invalid") from exc
        if len(set(selected_reasons)) != len(selected_reasons):
            raise ValidationError("machine settlement reason codes must be unique")
        object.__setattr__(self, "reason_codes", selected_reasons)
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("machine settlement schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "machine settlement created_at"))


@dataclass(frozen=True, slots=True)
class SemanticReviewLabelRecord(_SemanticV6Record):
    review_id: str
    settlement_id: str
    outcome: str
    reviewer_sha256: str
    evidence_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.review_id, "semantic review id")
        _text(self.settlement_id, "semantic review settlement")
        if self.outcome not in _REVIEW_OUTCOMES:
            raise ValidationError("semantic review outcome is invalid")
        _sha256(self.reviewer_sha256, "semantic review reviewer")
        _sha256(self.evidence_sha256, "semantic review evidence")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic review schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "semantic review created_at"))


@dataclass(frozen=True, slots=True)
class SemanticHealthEventRecord(_SemanticV6Record):
    event_id: str
    event_kind: str
    severity: str
    epoch_id: str | None
    tenant_bucket_sha256: str | None
    evidence_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.event_id, "semantic health event id")
        _text(self.event_kind, "semantic health event kind", maximum=128)
        if self.severity not in {"info", "warning", "critical"}:
            raise ValidationError("semantic health severity is invalid")
        _optional_text(self.epoch_id, "semantic health epoch")
        _optional_sha256(self.tenant_bucket_sha256, "semantic health tenant")
        _sha256(self.evidence_sha256, "semantic health evidence")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic health schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "semantic health created_at"))


@dataclass(frozen=True, slots=True)
class SemanticMachineOutcomeRecord(_SemanticV6Record):
    outcome_id: str
    settlement_id: str
    effect_id: str
    outcome: str
    evidence_sha256: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.outcome_id, "semantic machine outcome id")
        _text(self.settlement_id, "semantic machine outcome settlement")
        _text(self.effect_id, "semantic machine outcome effect")
        if self.outcome not in _MACHINE_LIFECYCLE_OUTCOMES:
            raise ValidationError("semantic machine lifecycle outcome is invalid")
        _sha256(self.evidence_sha256, "semantic machine outcome evidence")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValidationError("semantic machine outcome schema version must be 1")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "semantic machine outcome created_at"))


@dataclass(frozen=True, slots=True)
class SemanticRateBudgetRecord(_SemanticV6Record):
    bucket_id: str
    epoch_id: str
    tenant_bucket_sha256: str
    rule_id: str
    minute_window_started_at: str
    day_window_started_at: str
    minute_count: int
    day_count: int
    inflight_count: int
    revision: int
    updated_at: str

    def __post_init__(self) -> None:
        _text(self.bucket_id, "semantic rate bucket id")
        _text(self.epoch_id, "semantic rate bucket epoch")
        _sha256(self.tenant_bucket_sha256, "semantic rate bucket tenant")
        _text(self.rule_id, "semantic rate bucket rule")
        if self.bucket_id != semantic_rate_budget_bucket_id(
            tenant_bucket_sha256=self.tenant_bucket_sha256,
            rule_id=self.rule_id,
        ):
            raise ValidationError(
                "semantic rate bucket identity must bind the stable tenant/rule scope"
            )
        for value, label in (
            (self.minute_count, "semantic minute count"),
            (self.day_count, "semantic day count"),
            (self.inflight_count, "semantic inflight count"),
            (self.revision, "semantic rate bucket revision"),
        ):
            _counter(value, label)
        object.__setattr__(self, "minute_window_started_at", _timestamp(self.minute_window_started_at, "semantic minute budget window"))
        object.__setattr__(self, "day_window_started_at", _timestamp(self.day_window_started_at, "semantic day budget window"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "semantic budget updated_at"))


def semantic_rate_budget_bucket_id(
    *,
    tenant_bucket_sha256: str,
    rule_id: str,
) -> str:
    """Return the Store-canonical cross-epoch logical-rule budget identity."""

    _sha256(tenant_bucket_sha256, "semantic rate bucket tenant")
    _text(rule_id, "semantic rate bucket rule")
    encoded = json.dumps(
        {
            "schema_version": 2,
            "catalog_version": 1,
            "tenant_bucket_sha256": tenant_bucket_sha256,
            "logical_rule_id": rule_id,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"semantic-budget-v2:{hashlib.sha256(encoded).hexdigest()}"


def require_query_limit(limit: int, *, hard_limit: int = SEMANTIC_V6_QUERY_HARD_LIMIT) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= hard_limit:
        raise ValidationError(f"semantic v6 query limit must be between 1 and {hard_limit}")
    return limit


def records_page(records: Sequence[Any], *, limit: int, id_field: str) -> SemanticFlowPage:
    selected = tuple(records[:limit])
    next_cursor = None
    if len(records) > limit and selected:
        last = selected[-1]
        next_cursor = SemanticV6Cursor(last.created_at, getattr(last, id_field))
    return SemanticFlowPage(selected, next_cursor)


def policy_epoch_storage_record(
    value: SemanticPolicyEpochRecord | SemanticPolicyEpochV1,
) -> SemanticPolicyEpochRecord:
    if isinstance(value, SemanticPolicyEpochRecord):
        return value
    if not isinstance(value, SemanticPolicyEpochV1):
        raise ValidationError("semantic policy epoch must be a typed record")
    return SemanticPolicyEpochRecord(
        epoch_id=value.epoch_id,
        generation=value.generation,
        catalog_version=value.catalog_version,
        policy_sha256=value.canonical_sha256(),
        expected_previous_sha256=value.expected_previous_sha256,
        rollout_scope=_policy_rollout_scope(value),
        created_at=value.created_at,
        schema_version=value.schema_version,
    )


def _policy_rollout_scope(value: SemanticPolicyEpochV1) -> Mapping[str, Any]:
    auto_rules = [_policy_rollout_rule(rule) for rule in value.auto_approval_rules]
    deny_rules = [_policy_rollout_rule(rule) for rule in value.hard_deny_rules]
    auto_rules.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    deny_rules.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return {
        "schema_version": 2,
        "tenant_bucket_sha256s": sorted(value.tenant_bucket_sha256s),
        "auto_approval_rules": auto_rules,
        "hard_deny_rules": deny_rules,
        "allow_parameters": {
            "catalog_version": value.catalog_version,
            "classifier_profile_id_sha256": (
                _sha256_text(value.classifier_profile_id)
                if value.classifier_profile_id is not None
                else None
            ),
            "classifier_profile_sha256": value.classifier_profile_sha256,
            "classifier_model_sha256": value.classifier_model_sha256,
            "minimum_confidence_bps": value.minimum_confidence_bps,
            "required_calibration_bucket": (
                value.required_calibration_bucket.value
            ),
            "capability_ttl_s": value.capability_ttl_s,
            "per_rule_per_minute_limit": value.per_rule_per_minute_limit,
            "per_rule_per_day_limit": value.per_rule_per_day_limit,
            "max_inflight": value.max_inflight,
        },
    }


def _policy_rollout_rule(rule: Any) -> dict[str, Any]:
    is_prefix = rule.resource.endswith("*")
    resource_match = rule.resource[:-1] if is_prefix else rule.resource
    prefix_values = {
        resource_match[: index + 1]
        for index, character in enumerate(resource_match)
        if character in {":", "/"}
    }
    if is_prefix:
        prefix_values.add(resource_match)
    return {
        "rule_id_sha256": _sha256_text(rule.rule_id),
        "action_id": rule.authority_operation,
        "rights": sorted(rule.rights),
        "resource_kind": "prefix" if is_prefix else "exact",
        "match_sha256": _sha256_text(resource_match),
        "covering_prefix_sha256s": sorted(
            _sha256_text(prefix) for prefix in prefix_values
        ),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def control_state_storage_record(
    value: SemanticControlStateRecord | SemanticControlStateV1,
) -> SemanticControlStateRecord:
    if isinstance(value, SemanticControlStateRecord):
        return value
    if not isinstance(value, SemanticControlStateV1):
        raise ValidationError("semantic control state must be a typed record")
    return SemanticControlStateRecord(
        revision=value.revision,
        generation=value.generation,
        mode=value.mode.value,
        active_epoch_id=value.active_epoch_id,
        active_policy_sha256=value.active_policy_sha256,
        tripped=value.tripped,
        trip_code=value.trip_code.value if value.trip_code is not None else None,
        updated_at=value.updated_at,
        schema_version=value.schema_version,
    )


def machine_settlement_storage_record(
    value: SemanticMachineSettlementRecord | MachinePolicySettlementV1,
) -> SemanticMachineSettlementRecord:
    if isinstance(value, SemanticMachineSettlementRecord):
        return value
    if not isinstance(value, MachinePolicySettlementV1):
        raise ValidationError("semantic machine settlement must be a typed record")
    return SemanticMachineSettlementRecord(
        settlement_id=value.settlement_id,
        assessment_id=value.assessment_id,
        job_id=value.job_id,
        request_id=value.request_id,
        request_revision=value.request_revision,
        pid=value.pid,
        operation_id=value.operation_id,
        effect_id=value.effect_id,
        epoch_id=value.epoch_id,
        policy_sha256=value.policy_sha256,
        tenant_bucket_sha256=value.tenant_bucket_sha256,
        action_id=value.action_id,
        outcome=value.outcome.value,
        capability_id=value.capability_id,
        binding_sha256=value.binding_sha256,
        decision_sha256=value.decision_sha256,
        matched_rule_id=value.matched_rule_id,
        reason_codes=tuple(item.value for item in value.reason_codes),
        created_at=value.created_at,
        schema_version=value.schema_version,
    )
