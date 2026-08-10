"""Strict, payload-free projections for semantic read-only surfaces.

The semantic services are trusted Runtime components, but their persistence
records deliberately contain more provenance than the GUI and CLI may expose.
Every function in this module therefore constructs a fresh allow-listed
mapping and validates bounds before returning it.  Unknown source fields are
ignored so future private evidence cannot become public by accident.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from agent_libos.models.data_flow import (
    DataIntegrity,
    DataSensitivity,
    DataTrustLevel,
)
from agent_libos.models.semantic import (
    MachinePolicySettlementV1,
    SemanticAssessmentKind,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticControlStateV1,
    SemanticDataCategory,
    SemanticDataFinding,
    SemanticDataLocator,
    SemanticDomain,
    SemanticFinding,
    SemanticFindingSeverity,
    SemanticFindingSource,
    SemanticFlowStatusV1,
    SemanticPredicate,
    SemanticReasonCode,
    SemanticReviewLabelV1,
    SemanticStatusV3,
    SemanticTripCode,
    ShadowPolicyOutcome,
)
from agent_libos.utils.serde import to_jsonable


PAGE_DEFAULT = 50
PAGE_MAX = 100
CURSOR_MAX_CHARS = 2_048
ID_MAX_CHARS = 512
FILTER_MAX_CHARS = 512
MAX_SAFE_COUNTER = 2**53 - 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTION_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$")

ASSESSMENT_STATUSES = (
    "success",
    "skipped_policy",
    "egress_blocked",
    "timeout",
    "provider_error",
    "provider_outcome_unknown",
    "invalid_schema",
    "ood",
    "abstained",
    "stale_input",
)
ASSESSMENT_DOMAINS = (
    "filesystem",
    "shell",
    "git",
    "jsonrpc",
    "mcp",
    "runtime",
    "unknown",
)
SEMANTIC_MODES = frozenset({"off", "shadow", "enforce_deny", "canary_auto"})
ADAPTERS = frozenset({"deterministic", "external", "scripted"})
FLOW_COVERAGE = ("complete", "partial", "unknown", "conflict", "stale")
FLOW_DIRECTIONS = frozenset({"upstream", "downstream"})
FLOW_NODE_TYPES = frozenset({"entity", "activity"})
FLOW_ENTITY_KINDS = frozenset(
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
FLOW_ACTIVITY_KINDS = frozenset(
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
FLOW_RELATIONS = frozenset({"direct", "indirect", "control"})
SENSITIVITY = frozenset(
    {"public", "normal", "confidential", "restricted", "secret"}
)
TRUST_LEVEL = frozenset(
    {"untrusted", "unknown", "user_asserted", "verified", "trusted"}
)
INTEGRITY = frozenset({"untrusted", "unknown", "checked", "verified"})
SETTLEMENT_OUTCOMES = frozenset(
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
REVIEW_OUTCOMES = frozenset({"safe", "unsafe", "inconclusive"})
CONTROL_MODES = SEMANTIC_MODES
CONTROL_STATES = frozenset({"inactive", "active", "tripped", "revoked"})
HEALTH_SEVERITIES = frozenset({"info", "warning", "critical"})
ASSESSMENT_KINDS = frozenset(item.value for item in SemanticAssessmentKind)
ASSESSMENT_STATUS_VALUES = frozenset(
    item.value for item in SemanticAssessmentStatus
)
ASSESSMENT_DOMAIN_VALUES = frozenset(item.value for item in SemanticDomain)
SHADOW_OUTCOMES = frozenset(item.value for item in ShadowPolicyOutcome)
REASON_CODES = frozenset(item.value for item in SemanticReasonCode)
CALIBRATION_BUCKETS = frozenset(
    item.value for item in SemanticCalibrationBucket
)
FINDING_SEVERITIES = frozenset(item.value for item in SemanticFindingSeverity)
FINDING_SOURCES = frozenset(item.value for item in SemanticFindingSource)
DATA_CATEGORIES = frozenset(item.value for item in SemanticDataCategory)
DATA_LOCATORS = frozenset(item.value for item in SemanticDataLocator)
SENSITIVITY_VALUES = frozenset(item.value for item in DataSensitivity)
INTEGRITY_VALUES = frozenset(item.value for item in DataIntegrity)
TRUST_VALUES = frozenset(item.value for item in DataTrustLevel)
PREDICATES = frozenset(item.value for item in SemanticPredicate)
HUMAN_OUTCOMES = frozenset(
    {"pending", "approved", "rejected", "edited", "cancelled", "delivered"}
)
HUMAN_TERMINAL_OUTCOMES = frozenset({"approved", "rejected", "cancelled"})
HUMAN_OUTCOME_SOURCES = frozenset({"human", "machine_policy", "cancel"})
TRIP_CODES = frozenset(item.value for item in SemanticTripCode)
HEALTH_EVENT_KINDS = frozenset(
    {
        "semantic_control_disable_conflict",
        "semantic_control_authority_cleared",
        "semantic_policy_activated",
        "semantic_policy_rotated",
        "semantic_control_startup_conflict",
        "capture_failed",
        "semantic_unsafe_review_control_unsettled",
        "semantic_unsafe_review_fallback_trip",
        *(f"semantic_safety_trip:{item.value}" for item in SemanticTripCode),
    }
)


def mapping(value: Any, *, label: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise TypeError(f"{label} must be a mapping")
    return converted


def counter(value: Any, *, label: str) -> int:
    if type(value) is int and 0 <= value <= MAX_SAFE_COUNTER:
        return value
    raise TypeError(f"{label} is invalid")


def positive_counter(value: Any, *, label: str) -> int:
    selected = counter(value, label=label)
    if selected == 0:
        raise TypeError(f"{label} must be positive")
    return selected


def bounded_text(
    value: Any,
    *,
    label: str,
    maximum: int = FILTER_MAX_CHARS,
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(not character.isprintable() for character in value)
    ):
        raise TypeError(f"{label} is invalid")
    return value


def identifier(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value
    raise TypeError(f"{label} is invalid")


def digest(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return value
    raise TypeError(f"{label} is not a lowercase sha256 digest")


def enum(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, str) and value in allowed:
        return value
    raise TypeError(f"{label} is invalid")


def boolean(value: Any, *, label: str) -> bool:
    if type(value) is bool:
        return value
    raise TypeError(f"{label} is invalid")


def string_list(
    value: Any,
    *,
    label: str,
    maximum_items: int = 128,
    item_maximum: int = 4_096,
) -> list[str]:
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) > maximum_items
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > item_maximum
            or any(not character.isprintable() for character in item)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise TypeError(f"{label} is invalid")
    return list(value)


def enum_list(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
    maximum_items: int = 128,
) -> list[str]:
    selected = string_list(
        value,
        label=label,
        maximum_items=maximum_items,
        item_maximum=128,
    )
    if any(item not in allowed for item in selected):
        raise TypeError(f"{label} contains an unknown enum value")
    return selected


def confidence(value: Any, *, label: str) -> int:
    selected = counter(value, label=label)
    if selected > 10_000:
        raise TypeError(f"{label} exceeds 10000 basis points")
    return selected


def timestamp(value: Any, *, label: str) -> str:
    selected = bounded_text(value, label=label, maximum=64)
    assert selected is not None
    try:
        parsed = datetime.fromisoformat(selected.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TypeError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TypeError(f"{label} must include a timezone")
    parsed.astimezone(timezone.utc)
    return selected


def _counter_map(
    value: Any,
    *,
    keys: tuple[str, ...],
    label: str,
    allow_missing: bool = False,
) -> dict[str, int]:
    raw = mapping(value, label=label)
    if (set(raw) - set(keys)) or (not allow_missing and set(raw) != set(keys)):
        raise TypeError(f"{label} must contain every canonical key exactly once")
    return {
        key: counter(raw.get(key, 0), label=f"{label}.{key}") for key in keys
    }


def _rate(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is invalid")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise TypeError(f"{label} is invalid")
    return selected


def _ratio(
    value: Any,
    *,
    numerator_name: str,
    denominator_name: str,
    rate_name: str,
    label: str,
) -> dict[str, int | float | None]:
    raw = mapping(value, label=label)
    numerator = counter(raw.get(numerator_name, 0), label=f"{label}.{numerator_name}")
    denominator = counter(
        raw.get(denominator_name, 0),
        label=f"{label}.{denominator_name}",
    )
    rate = _rate(raw.get(rate_name), label=f"{label}.{rate_name}")
    if numerator > denominator:
        raise TypeError(f"{label} numerator exceeds denominator")
    if denominator == 0:
        if numerator != 0 or rate is not None:
            raise TypeError(f"{label} empty denominator requires 0/null")
    elif rate is None or not math.isclose(
        rate,
        numerator / denominator,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TypeError(f"{label} rate is inconsistent")
    return {
        numerator_name: numerator,
        denominator_name: denominator,
        rate_name: rate,
    }


def project_flow_status(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic flow status")
    try:
        raw = SemanticFlowStatusV1.from_dict(raw).to_dict()
    except (TypeError, ValueError) as exc:
        raise TypeError("semantic flow status has an invalid v1 contract") from exc
    legacy = mapping(
        raw.get("legacy_history"),
        label="semantic legacy flow history",
    )
    present = boolean(
        legacy.get("present"),
        label="semantic legacy flow present",
    )
    source_schema_version = legacy.get("source_schema_version")
    legacy_coverage = legacy.get("coverage")
    evidence_sha256 = legacy.get("evidence_sha256")
    created_at = legacy.get("created_at")
    if present:
        if source_schema_version != 5 or legacy_coverage != "unknown":
            raise TypeError("semantic legacy flow history boundary is invalid")
        evidence_sha256 = digest(
            evidence_sha256,
            label="semantic legacy flow evidence",
        )
        created_at = timestamp(
            created_at,
            label="semantic legacy flow created_at",
        )
    elif any(
        selected is not None
        for selected in (
            source_schema_version,
            legacy_coverage,
            evidence_sha256,
            created_at,
        )
    ):
        raise TypeError("absent semantic legacy flow history must be empty")
    return {
        "schema_version": 1,
        "available": boolean(raw.get("available"), label="semantic flow available"),
        "counts": _counter_map(
            raw.get("counts"),
            keys=("entities", "activities", "edges", "label_assertions"),
            label="semantic flow counts",
        ),
        "coverage": _counter_map(
            raw.get("coverage"),
            keys=FLOW_COVERAGE,
            label="semantic flow coverage",
        ),
        "capture_failures": counter(
            raw.get("capture_failures", 0),
            label="semantic flow capture failures",
        ),
        "legacy_history": {
            "present": present,
            "source_schema_version": source_schema_version,
            "assessment_count": counter(
                legacy.get("assessment_count"),
                label="semantic legacy flow assessment count",
            ),
            "coverage": legacy_coverage,
            "evidence_sha256": evidence_sha256,
            "created_at": created_at,
        },
    }


def _semantic_status_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    queue_source = mapping(raw.get("queue", {}), label="semantic queue")
    assessment_source = mapping(
        raw.get("assessments", {}), label="semantic assessments"
    )
    by_status_source = mapping(
        assessment_source.get("by_status"), label="semantic assessment statuses"
    )
    by_domain_source = mapping(
        assessment_source.get("by_domain"), label="semantic assessment domains"
    )
    control_source = mapping(raw.get("control"), label="semantic status control")
    machine_source = mapping(raw.get("machine"), label="semantic machine counters")
    actual_source = mapping(
        raw.get("actual_auto_approval"), label="semantic actual auto approval"
    )
    review_source = mapping(
        raw.get("review_metrics"), label="semantic review metrics"
    )
    return {
        "schema_version": raw.get("schema_version"),
        "mode": raw.get("mode"),
        "adapter": raw.get("adapter"),
        "profile_id": raw.get("profile_id"),
        "control": {
            key: control_source.get(key)
            for key in (
                "catalog_version",
                "active_epoch_id",
                "active_epoch_sha256",
                "generation",
                "state",
                "trip_reason_code",
            )
        },
        "queue": {
            key: queue_source.get(key, 0)
            for key in (
                "queued",
                "leased",
                "succeeded",
                "failed",
                "cancelled",
                "capture_failures",
            )
        },
        "assessments": {
            **{
                key: assessment_source.get(key, 0)
                for key in (
                    "total",
                    "success",
                    "error",
                    "ood",
                    "would_issue_exact_once",
                    "would_deny",
                    "require_human",
                )
            },
            "by_status": {
                key: by_status_source.get(key, 0) for key in ASSESSMENT_STATUSES
            },
            "by_domain": {
                key: by_domain_source.get(key, 0) for key in ASSESSMENT_DOMAINS
            },
        },
        "flow": project_flow_status(raw.get("flow")),
        "machine": {
            key: machine_source.get(key, 0)
            for key in (
                "eligible",
                "issued",
                "consumed",
                "succeeded",
                "failed",
                "unknown",
                "expired",
                "revoked",
                "race_lost",
                "denied",
            )
        },
        "actual_auto_approval": {
            "numerator": actual_source.get("numerator", 0),
            "denominator": actual_source.get("denominator", 0),
            "rate": actual_source.get("rate"),
        },
        "review_metrics": {
            key: review_source.get(key)
            for key in (
                "reviewed",
                "safe",
                "unsafe",
                "unsafe_rate",
                "issued_reviewed",
                "issued_review_rate",
            )
        },
    }


def _validated_semantic_status(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        status = SemanticStatusV3.from_dict(candidate)
    except ValueError as exc:
        if any(
            marker in str(exc).casefold()
            for marker in ("match", "exceed", "equal", "inconsistent")
        ):
            raise TypeError("semantic status counters are inconsistent") from exc
        raise TypeError("semantic status has an invalid v3 contract") from exc
    except TypeError as exc:
        raise TypeError("semantic status has an invalid v3 contract") from exc
    return status.to_dict()


def _project_status_assessments(value: Any) -> dict[str, Any]:
    assessment_raw = mapping(value, label="semantic assessments")
    by_status = _counter_map(
        assessment_raw.get("by_status"),
        keys=ASSESSMENT_STATUSES,
        label="semantic assessment statuses",
    )
    by_domain = _counter_map(
        assessment_raw.get("by_domain"),
        keys=ASSESSMENT_DOMAINS,
        label="semantic assessment domains",
    )
    assessment_fields = (
        "total",
        "success",
        "error",
        "ood",
        "would_issue_exact_once",
        "would_deny",
        "require_human",
    )
    assessments = {
        key: counter(
            assessment_raw.get(key, 0), label=f"semantic assessments.{key}"
        )
        for key in assessment_fields
    }
    assessments.update({"by_status": by_status, "by_domain": by_domain})
    if (
        sum(by_status.values()) != assessments["total"]
        or sum(by_domain.values()) != assessments["total"]
        or assessments["success"] + assessments["error"] != assessments["total"]
        or assessments["would_issue_exact_once"]
        + assessments["would_deny"]
        + assessments["require_human"]
        != assessments["total"]
        or assessments["ood"] != by_status["ood"]
    ):
        raise TypeError("semantic assessment aggregate counters are inconsistent")
    return assessments


def _project_status_control(value: Any) -> dict[str, Any]:
    control_raw = mapping(value, label="semantic status control")
    return {
        "catalog_version": (
            None
            if control_raw.get("catalog_version") is None
            else positive_counter(
                control_raw.get("catalog_version"),
                label="semantic control catalog version",
            )
        ),
        "active_epoch_id": bounded_text(
            control_raw.get("active_epoch_id"),
            label="semantic active epoch id",
            nullable=True,
        ),
        "active_epoch_sha256": digest(
            control_raw.get("active_epoch_sha256"),
            label="semantic active epoch digest",
            nullable=True,
        ),
        "generation": counter(
            control_raw.get("generation"), label="semantic control generation"
        ),
        "state": enum(
            control_raw.get("state"),
            label="semantic public control state",
            allowed=CONTROL_STATES,
        ),
        "trip_reason_code": bounded_text(
            control_raw.get("trip_reason_code"),
            label="semantic trip reason code",
            maximum=256,
            nullable=True,
        ),
    }


def _project_status_review_metrics(value: Any) -> dict[str, Any]:
    review_raw = mapping(value, label="semantic review metrics")
    return {
        "reviewed": counter(
            review_raw.get("reviewed"), label="semantic review reviewed"
        ),
        "safe": counter(review_raw.get("safe"), label="semantic review safe"),
        "unsafe": counter(
            review_raw.get("unsafe"), label="semantic review unsafe"
        ),
        "unsafe_rate": _rate(
            review_raw.get("unsafe_rate"), label="semantic review unsafe rate"
        ),
        "issued_reviewed": counter(
            review_raw.get("issued_reviewed"),
            label="semantic review issued reviewed",
        ),
        "issued_review_rate": _rate(
            review_raw.get("issued_review_rate"),
            label="semantic review issued review rate",
        ),
    }


def project_semantic_status(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic status")
    raw = _validated_semantic_status(_semantic_status_candidate(raw))
    return {
        "schema_version": 3,
        "mode": enum(
            raw.get("mode"), label="semantic mode", allowed=SEMANTIC_MODES
        ),
        "adapter": enum(
            raw.get("adapter"), label="semantic adapter", allowed=ADAPTERS
        ),
        "profile_id": _profile_id(raw.get("profile_id")),
        "control": _project_status_control(raw.get("control")),
        "queue": _counter_map(
            raw.get("queue", {}),
            keys=(
                "queued",
                "leased",
                "succeeded",
                "failed",
                "cancelled",
                "capture_failures",
            ),
            label="semantic queue",
        ),
        "assessments": _project_status_assessments(raw.get("assessments", {})),
        "flow": project_flow_status(raw.get("flow")),
        "machine": _counter_map(
            raw.get("machine"),
            keys=(
                "eligible",
                "issued",
                "consumed",
                "succeeded",
                "failed",
                "unknown",
                "expired",
                "revoked",
                "race_lost",
                "denied",
            ),
            label="semantic machine counters",
        ),
        "actual_auto_approval": _ratio(
            raw.get("actual_auto_approval", {}),
            numerator_name="numerator",
            denominator_name="denominator",
            rate_name="rate",
            label="semantic actual auto approval",
        ),
        "review_metrics": _project_status_review_metrics(
            raw.get("review_metrics")
        ),
    }


def _profile_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and _PROFILE_ID.fullmatch(value):
        return value
    raise TypeError("semantic profile id is invalid")


def _project_labels(value: Any, *, nullable: bool = False) -> dict[str, str] | None:
    if value is None and nullable:
        return None
    raw = mapping(value, label="semantic flow labels")
    return {
        "sensitivity": str(
            enum(
                raw.get("sensitivity"),
                label="semantic flow label sensitivity",
                allowed=SENSITIVITY,
            )
        ),
        "trust_level": str(
            enum(
                raw.get("trust_level"),
                label="semantic flow label trust level",
                allowed=TRUST_LEVEL,
            )
        ),
        "integrity": str(
            enum(
                raw.get("integrity"),
                label="semantic flow label integrity",
                allowed=INTEGRITY,
            )
        ),
    }


def project_flow_entity(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic flow entity")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic flow entity has an unsupported schema version")
    return {
        "schema_version": 1,
        "entity_id": identifier(raw.get("entity_id"), label="flow entity id"),
        "kind": enum(
            raw.get("kind"), label="flow entity kind", allowed=FLOW_ENTITY_KINDS
        ),
        "pid": identifier(raw.get("pid"), label="flow entity pid", nullable=True),
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="flow entity tenant bucket",
        ),
        "content_sha256": digest(
            raw.get("content_sha256"), label="flow entity content"
        ),
        "version_sha256": digest(
            raw.get("version_sha256"), label="flow entity version"
        ),
        "provenance_sha256": digest(
            raw.get("provenance_sha256"),
            label="flow entity provenance",
        ),
        "baseline_labels": _project_labels(raw.get("baseline_labels")),
        "identity_present": boolean(
            raw.get("identity_present"), label="flow entity identity_present"
        ),
        "identity_mixed": boolean(
            raw.get("identity_mixed"), label="flow entity identity_mixed"
        ),
        "coverage": enum(
            raw.get("coverage"),
            label="flow entity coverage",
            allowed=frozenset(FLOW_COVERAGE),
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="flow entity created_at", maximum=64
        ),
    }


def project_flow_activity(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic flow activity")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic flow activity has an unsupported schema version")
    action_id = raw.get("action_id")
    if action_id is not None and (
        not isinstance(action_id, str) or _ACTION_ID.fullmatch(action_id) is None
    ):
        raise TypeError("flow activity action_id is invalid")
    return {
        "schema_version": 1,
        "activity_id": identifier(
            raw.get("activity_id"), label="flow activity id"
        ),
        "kind": enum(
            raw.get("kind"),
            label="flow activity kind",
            allowed=FLOW_ACTIVITY_KINDS,
        ),
        "pid": identifier(raw.get("pid"), label="flow activity pid"),
        "action_id": action_id,
        "effect_id": identifier(
            raw.get("effect_id"), label="flow activity effect id", nullable=True
        ),
        "state_sha256": digest(
            raw.get("state_sha256"), label="flow activity state"
        ),
        "provider_spec_sha256": digest(
            raw.get("provider_spec_sha256"),
            label="flow activity provider",
            nullable=True,
        ),
        "tool_schema_sha256": digest(
            raw.get("tool_schema_sha256"),
            label="flow activity tool",
            nullable=True,
        ),
        "model_artifact_sha256": digest(
            raw.get("model_artifact_sha256"),
            label="flow activity model",
            nullable=True,
        ),
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="flow activity tenant bucket",
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="flow activity created_at", maximum=64
        ),
    }


def project_flow_edge(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic flow edge")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic flow edge has an unsupported schema version")
    return {
        "schema_version": 1,
        "edge_id": identifier(raw.get("edge_id"), label="flow edge id"),
        "relation": enum(
            raw.get("relation"),
            label="flow edge relation",
            allowed=FLOW_RELATIONS,
        ),
        "source_node_id": identifier(
            raw.get("source_node_id"), label="flow edge source node id"
        ),
        "source_node_type": enum(
            raw.get("source_node_type"),
            label="flow edge source node type",
            allowed=FLOW_NODE_TYPES,
        ),
        "target_node_id": identifier(
            raw.get("target_node_id"), label="flow edge target node id"
        ),
        "target_node_type": enum(
            raw.get("target_node_type"),
            label="flow edge target node type",
            allowed=FLOW_NODE_TYPES,
        ),
        "pid": identifier(raw.get("pid"), label="flow edge pid"),
        "provenance_sha256": digest(
            raw.get("provenance_sha256"), label="flow edge provenance"
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="flow edge created_at", maximum=64
        ),
    }


def project_page(
    value: Any,
    *,
    item_projector: Callable[[Any], dict[str, Any]],
    maximum_items: int,
    label: str,
) -> dict[str, Any]:
    raw = mapping(value, label=label)
    if raw.get("schema_version") != 1:
        raise TypeError(f"{label} has an unsupported schema version")
    items = raw.get("items")
    cursor = raw.get("next_cursor")
    if not isinstance(items, list) or len(items) > maximum_items:
        raise TypeError(f"{label} items exceed the requested bound")
    if cursor is not None:
        cursor = bounded_text(
            cursor,
            label=f"{label} next cursor",
            maximum=CURSOR_MAX_CHARS,
        )
    return {
        "schema_version": 1,
        "items": [item_projector(item) for item in items],
        "next_cursor": cursor,
    }


def project_assessment_summary(value: Any) -> dict[str, Any]:
    """Project one assessment through a closed, payload-free type contract."""

    raw = mapping(value, label="semantic assessment")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic assessment has an unsupported schema version")
    action_id = raw.get("action_id")
    if not isinstance(action_id, str) or _ACTION_ID.fullmatch(action_id) is None:
        raise TypeError("semantic assessment action id is invalid")
    status = enum(
        raw.get("status"),
        label="semantic assessment status",
        allowed=ASSESSMENT_STATUS_VALUES,
    )
    ood = boolean(raw.get("ood"), label="semantic assessment ood")
    abstain = boolean(raw.get("abstain"), label="semantic assessment abstain")
    if ood != (status == SemanticAssessmentStatus.OOD.value):
        raise TypeError("semantic assessment ood status is inconsistent")
    if abstain != (status == SemanticAssessmentStatus.ABSTAINED.value):
        raise TypeError("semantic assessment abstain status is inconsistent")
    created_at = timestamp(
        raw.get("created_at"), label="semantic assessment created_at"
    )
    completed_at = timestamp(
        raw.get("completed_at"), label="semantic assessment completed_at"
    )
    if _parsed_timestamp(completed_at) < _parsed_timestamp(created_at):
        raise TypeError("semantic assessment completion precedes creation")
    return {
        "assessment_id": bounded_text(
            raw.get("assessment_id"), label="semantic assessment id"
        ),
        "job_id": bounded_text(raw.get("job_id"), label="semantic assessment job id"),
        "kind": enum(
            raw.get("kind"),
            label="semantic assessment kind",
            allowed=ASSESSMENT_KINDS,
        ),
        "status": status,
        "domain": enum(
            raw.get("domain"),
            label="semantic assessment domain",
            allowed=ASSESSMENT_DOMAIN_VALUES,
        ),
        "action_id": action_id,
        "pid": bounded_text(
            raw.get("pid"), label="semantic assessment pid", nullable=True
        ),
        "request_id": bounded_text(
            raw.get("request_id"),
            label="semantic assessment request id",
            nullable=True,
        ),
        "operation_id": bounded_text(
            raw.get("operation_id"),
            label="semantic assessment operation id",
            nullable=True,
        ),
        "effect_id": bounded_text(
            raw.get("effect_id"),
            label="semantic assessment effect id",
            nullable=True,
        ),
        "shadow_outcome": enum(
            raw.get("shadow_outcome"),
            label="semantic assessment shadow outcome",
            allowed=SHADOW_OUTCOMES,
        ),
        "reason_codes": enum_list(
            raw.get("reason_codes"),
            label="semantic assessment reason codes",
            allowed=REASON_CODES,
            maximum_items=64,
        ),
        "ood": ood,
        "abstain": abstain,
        "confidence_bps": confidence(
            raw.get("confidence_bps"),
            label="semantic assessment confidence",
        ),
        "calibration_bucket": enum(
            raw.get("calibration_bucket"),
            label="semantic assessment calibration bucket",
            allowed=CALIBRATION_BUCKETS,
        ),
        "input_tokens": _nullable_counter(
            raw.get("input_tokens"), label="semantic assessment input_tokens"
        ),
        "output_tokens": _nullable_counter(
            raw.get("output_tokens"), label="semantic assessment output_tokens"
        ),
        "cost_microunits": _nullable_counter(
            raw.get("cost_microunits"),
            label="semantic assessment cost_microunits",
        ),
        "classifier_id": bounded_text(
            raw.get("classifier_id"), label="semantic assessment classifier id"
        ),
        "classifier_version": bounded_text(
            raw.get("classifier_version"),
            label="semantic assessment classifier version",
        ),
        "artifact_sha256": digest(
            raw.get("artifact_sha256"), label="semantic assessment artifact"
        ),
        "input_sha256": digest(
            raw.get("input_sha256"), label="semantic assessment input"
        ),
        "feature_snapshot_sha256": digest(
            raw.get("feature_snapshot_sha256"),
            label="semantic assessment feature snapshot",
        ),
        "policy_sha256": digest(
            raw.get("policy_sha256"), label="semantic assessment policy"
        ),
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="semantic assessment tenant bucket",
            nullable=True,
        ),
        "created_at": created_at,
        "completed_at": completed_at,
        "latency_ms": counter(
            raw.get("latency_ms"), label="semantic assessment latency_ms"
        ),
        "human_outcome": enum(
            raw.get("human_outcome"),
            label="semantic assessment invalid human outcome",
            allowed=HUMAN_OUTCOMES,
            nullable=True,
        ),
    }


def project_assessment_detail(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic assessment")
    summary = project_assessment_summary(raw)
    return {
        **summary,
        "findings": _project_assessment_findings(raw.get("findings")),
        "data_findings": _project_assessment_data_findings(
            raw.get("data_findings"), kind=summary["kind"]
        ),
        "matched_rule_ids": string_list(
            raw.get("matched_rule_ids"),
            label="semantic assessment matched rule ids",
            maximum_items=64,
            item_maximum=128,
        ),
        "proven_predicates": enum_list(
            raw.get("proven_predicates"),
            label="semantic assessment proven predicates",
            allowed=PREDICATES,
            maximum_items=len(PREDICATES),
        ),
        "missing_predicates": enum_list(
            raw.get("missing_predicates"),
            label="semantic assessment missing predicates",
            allowed=PREDICATES,
            maximum_items=len(PREDICATES),
        ),
        **{
            field: digest(
                raw.get(field),
                label=f"semantic assessment {field}",
                nullable=True,
            )
            for field in (
                "source_refs_sha256",
                "data_labels_sha256",
                "sink_identity_sha256",
                "tool_schema_sha256",
                "provider_spec_sha256",
                "manifest_sha256",
                "resource_sha256",
                "args_sha256",
                "state_sha256",
            )
        },
        "action_sha256": digest(
            raw.get("action_sha256"), label="semantic assessment action digest"
        ),
        "projection_sha256": digest(
            raw.get("projection_sha256"),
            label="semantic assessment projection digest",
        ),
    }


def _project_assessment_findings(value: Any) -> list[dict[str, Any]]:
    items = _bounded_mapping_list(value, label="semantic assessment findings")
    selected: list[dict[str, Any]] = []
    fields = ("code", "severity", "confidence_bps", "evidence_sha256", "source")
    for item in items:
        candidate = {field: item.get(field) for field in fields}
        try:
            finding = SemanticFinding.from_dict(candidate)
        except (TypeError, ValueError) as exc:
            raise TypeError("semantic assessment finding is invalid") from exc
        selected.append(finding.to_dict())
    return selected


def _project_assessment_data_findings(
    value: Any,
    *,
    kind: Any,
) -> list[dict[str, Any]]:
    items = _bounded_mapping_list(value, label="semantic assessment data findings")
    coarse_by_kind = {
        SemanticAssessmentKind.APPROVAL.value: SemanticDataLocator.APPROVAL_REQUEST.value,
        SemanticAssessmentKind.ROOT_GOAL.value: SemanticDataLocator.ROOT_GOAL.value,
        SemanticAssessmentKind.PROVIDER_INGRESS.value: SemanticDataLocator.PROVIDER_RESULT.value,
    }
    coarse_locator = coarse_by_kind.get(kind)
    if coarse_locator is None:
        raise TypeError("semantic assessment data finding kind is invalid")
    fields = (
        "category",
        "field",
        "span_start",
        "span_end",
        "sensitivity_floor",
        "integrity_ceiling",
        "trust_ceiling",
        "confidence_bps",
        "evidence_sha256",
    )
    selected: list[dict[str, Any]] = []
    for item in items:
        candidate = {field: item.get(field) for field in fields}
        try:
            finding = SemanticDataFinding.from_dict(candidate)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "semantic assessment contains an invalid data finding"
            ) from exc
        projected = finding.to_dict()
        if (
            projected["field"] != SemanticDataLocator.REDACTED_INTENT.value
            and projected["field"] != coarse_locator
        ):
            raise TypeError(
                "semantic assessment contains an invalid data finding locator"
            )
        selected.append(projected)
    return selected


def _bounded_mapping_list(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 64:
        raise TypeError(f"{label} is invalid")
    return [mapping(item, label=label) for item in value]


def _nullable_counter(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    return counter(value, label=label)


def _parsed_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def project_flow_lineage(value: Any, *, maximum_items: int) -> dict[str, Any]:
    raw = mapping(value, label="semantic flow lineage")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic flow lineage has an unsupported schema version")
    items = raw.get("items")
    if not isinstance(items, list) or len(items) > maximum_items:
        raise TypeError("semantic flow lineage items exceed the requested bound")
    projected: list[dict[str, Any]] = []
    for item in items:
        selected = mapping(item, label="semantic flow lineage item")
        node_type = enum(
            selected.get("node_type"),
            label="semantic lineage node type",
            allowed=FLOW_NODE_TYPES,
        )
        depth = counter(selected.get("depth"), label="semantic lineage depth")
        if depth > 16:
            raise TypeError("semantic lineage depth exceeds hard limit")
        projected.append(
            {
                "depth": depth,
                "edge": project_flow_edge(selected.get("edge")),
                "node_type": node_type,
                "node": (
                    project_flow_entity(selected.get("node"))
                    if node_type == "entity"
                    else project_flow_activity(selected.get("node"))
                ),
            }
        )
    cursor = raw.get("next_cursor")
    if cursor is not None:
        cursor = bounded_text(
            cursor,
            label="semantic flow lineage next cursor",
            maximum=CURSOR_MAX_CHARS,
        )
    return {
        "schema_version": 1,
        "root_node_id": identifier(
            raw.get("root_node_id"), label="semantic lineage root node id"
        ),
        "direction": enum(
            raw.get("direction"),
            label="semantic lineage direction",
            allowed=FLOW_DIRECTIONS,
        ),
        "items": projected,
        "effective_labels": _project_labels(
            raw.get("effective_labels"), nullable=True
        ),
        "coverage": enum(
            raw.get("coverage"),
            label="semantic lineage coverage",
            allowed=frozenset(FLOW_COVERAGE),
        ),
        "next_cursor": cursor,
        "truncated": boolean(
            raw.get("truncated"), label="semantic lineage truncated"
        ),
    }


def project_machine_settlement(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic machine settlement")
    core_fields = (
        "schema_version",
        "settlement_id",
        "assessment_id",
        "job_id",
        "request_id",
        "request_revision",
        "pid",
        "operation_id",
        "effect_id",
        "epoch_id",
        "policy_sha256",
        "tenant_bucket_sha256",
        "action_id",
        "outcome",
        "capability_id",
        "binding_sha256",
        "decision_sha256",
        "matched_rule_id",
        "reason_codes",
        "created_at",
    )
    try:
        core = MachinePolicySettlementV1.from_dict(
            {field: raw.get(field) for field in core_fields}
        ).to_dict()
    except (TypeError, ValueError) as exc:
        raise TypeError("semantic settlement has an invalid v1 contract") from exc
    human = _project_settlement_human_outcome(raw)
    raw = core
    action_id = raw.get("action_id")
    if not isinstance(action_id, str) or _ACTION_ID.fullmatch(action_id) is None:
        raise TypeError("semantic settlement action id is invalid")
    request_revision = counter(
        raw.get("request_revision"), label="semantic settlement request revision"
    )
    return {
        "schema_version": 1,
        "settlement_id": bounded_text(
            raw.get("settlement_id"), label="semantic settlement id"
        ),
        "assessment_id": bounded_text(
            raw.get("assessment_id"),
            label="semantic settlement assessment id",
            nullable=True,
        ),
        "job_id": bounded_text(
            raw.get("job_id"), label="semantic settlement job id", nullable=True
        ),
        "request_id": bounded_text(
            raw.get("request_id"), label="semantic settlement request id"
        ),
        "request_revision": request_revision,
        "pid": bounded_text(raw.get("pid"), label="semantic settlement pid"),
        "operation_id": bounded_text(
            raw.get("operation_id"),
            label="semantic settlement operation id",
            nullable=True,
        ),
        "effect_id": bounded_text(
            raw.get("effect_id"), label="semantic settlement effect id"
        ),
        "epoch_id": bounded_text(
            raw.get("epoch_id"), label="semantic settlement epoch id"
        ),
        "policy_sha256": digest(
            raw.get("policy_sha256"), label="semantic settlement policy"
        ),
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="semantic settlement tenant bucket",
        ),
        "action_id": action_id,
        "outcome": enum(
            raw.get("outcome"),
            label="semantic settlement outcome",
            allowed=SETTLEMENT_OUTCOMES,
        ),
        "capability_id": bounded_text(
            raw.get("capability_id"),
            label="semantic settlement capability id",
            nullable=True,
        ),
        "binding_sha256": digest(
            raw.get("binding_sha256"), label="semantic settlement binding"
        ),
        "decision_sha256": digest(
            raw.get("decision_sha256"), label="semantic settlement decision"
        ),
        "matched_rule_id": bounded_text(
            raw.get("matched_rule_id"),
            label="semantic settlement matched rule id",
            maximum=128,
            nullable=True,
        ),
        "reason_codes": string_list(
            raw.get("reason_codes"), label="semantic settlement reason codes"
        ),
        "created_at": bounded_text(
            raw.get("created_at"),
            label="semantic settlement created_at",
            maximum=64,
        ),
        **human,
    }


def _project_settlement_human_outcome(raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "human_outcome",
        "human_outcome_source",
        "human_outcome_request_revision",
        "human_outcome_decision_sha256",
        "human_outcome_created_at",
    )
    values = tuple(raw.get(field) for field in fields)
    if all(value is None for value in values):
        return {field: None for field in fields}
    if any(value is None for value in values):
        raise TypeError("semantic settlement Human outcome link is incomplete")
    outcome = enum(
        values[0],
        label="semantic settlement Human outcome",
        allowed=HUMAN_TERMINAL_OUTCOMES,
    )
    source = enum(
        values[1],
        label="semantic settlement Human outcome source",
        allowed=HUMAN_OUTCOME_SOURCES,
    )
    if (source == "cancel") != (outcome == "cancelled"):
        raise TypeError("semantic settlement Human outcome source is inconsistent")
    return {
        "human_outcome": outcome,
        "human_outcome_source": source,
        "human_outcome_request_revision": counter(
            values[2], label="semantic settlement Human outcome request revision"
        ),
        "human_outcome_decision_sha256": digest(
            values[3], label="semantic settlement Human outcome decision"
        ),
        "human_outcome_created_at": timestamp(
            values[4], label="semantic settlement Human outcome created_at"
        ),
    }


def project_policy_epoch(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic policy epoch")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic policy epoch has an unsupported schema version")
    catalog_version = positive_counter(
        raw.get("catalog_version"), label="semantic policy catalog version"
    )
    if catalog_version != 1:
        raise TypeError("semantic policy catalog version is unsupported")
    return {
        "schema_version": 1,
        "epoch_id": bounded_text(raw.get("epoch_id"), label="semantic policy epoch id"),
        "generation": positive_counter(
            raw.get("generation"), label="semantic policy generation"
        ),
        "catalog_version": catalog_version,
        "policy_sha256": digest(
            raw.get("policy_sha256"), label="semantic policy digest"
        ),
        "expected_previous_sha256": digest(
            raw.get("expected_previous_sha256"),
            label="semantic previous policy digest",
            nullable=True,
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="semantic policy created_at", maximum=64
        ),
    }


def project_control(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic control")
    try:
        raw = SemanticControlStateV1.from_dict(raw).to_dict()
    except (TypeError, ValueError) as exc:
        raise TypeError("semantic control has an invalid v1 contract") from exc
    mode = enum(raw.get("mode"), label="semantic control mode", allowed=CONTROL_MODES)
    active_epoch_id = bounded_text(
        raw.get("active_epoch_id"),
        label="semantic control active epoch",
        nullable=True,
    )
    active_policy_sha256 = digest(
        raw.get("active_policy_sha256"),
        label="semantic control active policy",
        nullable=True,
    )
    tripped = boolean(raw.get("tripped"), label="semantic control tripped")
    trip_code = bounded_text(
        raw.get("trip_code"),
        label="semantic control trip code",
        maximum=256,
        nullable=True,
    )
    if (active_epoch_id is None) != (active_policy_sha256 is None):
        raise TypeError("semantic control active epoch binding is incomplete")
    if tripped != (trip_code is not None):
        raise TypeError("semantic control trip fields are inconsistent")
    return {
        "schema_version": 1,
        "revision": counter(raw.get("revision"), label="semantic control revision"),
        "generation": counter(
            raw.get("generation"), label="semantic control generation"
        ),
        "mode": mode,
        "active_epoch_id": active_epoch_id,
        "active_policy_sha256": active_policy_sha256,
        "tripped": tripped,
        "trip_code": trip_code,
        "updated_at": bounded_text(
            raw.get("updated_at"), label="semantic control updated_at", maximum=64
        ),
    }


def project_health_event(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic health event")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic health event has an unsupported schema version")
    return {
        "schema_version": 1,
        "event_id": bounded_text(raw.get("event_id"), label="semantic health event id"),
        "event_kind": enum(
            raw.get("event_kind"),
            label="semantic health event kind",
            allowed=HEALTH_EVENT_KINDS,
        ),
        "severity": enum(
            raw.get("severity"),
            label="semantic health severity",
            allowed=HEALTH_SEVERITIES,
        ),
        "epoch_id": bounded_text(
            raw.get("epoch_id"), label="semantic health epoch id", nullable=True
        ),
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="semantic health tenant bucket",
            nullable=True,
        ),
        "evidence_sha256": digest(
            raw.get("evidence_sha256"), label="semantic health evidence"
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="semantic health created_at", maximum=64
        ),
    }


def project_review_label(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic review label")
    try:
        raw = SemanticReviewLabelV1.from_dict(raw).to_dict()
    except (TypeError, ValueError) as exc:
        raise TypeError("semantic review label has an invalid v1 contract") from exc
    return {
        "schema_version": 1,
        "review_id": bounded_text(raw.get("review_id"), label="semantic review id"),
        "settlement_id": bounded_text(
            raw.get("settlement_id"), label="semantic review settlement id"
        ),
        "outcome": enum(
            raw.get("outcome"),
            label="semantic review outcome",
            allowed=REVIEW_OUTCOMES,
        ),
        "reviewer_sha256": digest(
            raw.get("reviewer_sha256"), label="semantic reviewer"
        ),
        "evidence_sha256": digest(
            raw.get("evidence_sha256"), label="semantic review evidence"
        ),
        "created_at": bounded_text(
            raw.get("created_at"), label="semantic review created_at", maximum=64
        ),
    }


def project_metrics(value: Any) -> dict[str, Any]:
    raw = mapping(value, label="semantic metrics")
    if raw.get("schema_version") != 1:
        raise TypeError("semantic metrics has an unsupported schema version")
    action_id = raw.get("action_id")
    if action_id is not None and (
        not isinstance(action_id, str) or _ACTION_ID.fullmatch(action_id) is None
    ):
        raise TypeError("semantic metrics action id is invalid")
    machine = _counter_map(
        raw.get("machine"),
        keys=(
            "eligible",
            "issued",
            "consumed",
            "succeeded",
            "failed",
            "unknown",
            "expired",
            "revoked",
            "race_lost",
            "denied",
        ),
        label="semantic metric machine counters",
    )
    actual = _ratio(
        raw.get("actual_auto_approval"),
        numerator_name="numerator",
        denominator_name="denominator",
        rate_name="rate",
        label="semantic metric actual auto approval",
    )
    if (
        actual["numerator"] != machine["issued"]
        or actual["denominator"] != machine["eligible"]
    ):
        raise TypeError(
            "semantic metric auto-approval ratio must use issued / eligible"
        )
    review_raw = mapping(raw.get("review_metrics"), label="semantic metric reviews")
    reviewed = counter(
        review_raw.get("reviewed"), label="semantic metric reviewed"
    )
    safe = counter(review_raw.get("safe"), label="semantic metric safe")
    unsafe = counter(review_raw.get("unsafe"), label="semantic metric unsafe")
    unsafe_rate = _rate(
        review_raw.get("unsafe_rate"), label="semantic metric unsafe rate"
    )
    issued_reviewed = counter(
        review_raw.get("issued_reviewed"),
        label="semantic metric issued reviewed",
    )
    issued_review_rate = _rate(
        review_raw.get("issued_review_rate"),
        label="semantic metric issued review rate",
    )
    if safe + unsafe > reviewed:
        raise TypeError("semantic metric review counters are inconsistent")
    if reviewed == 0:
        if unsafe != 0 or unsafe_rate is not None:
            raise TypeError("empty semantic metric reviews require a null rate")
    elif unsafe_rate is None or not math.isclose(
        unsafe_rate,
        unsafe / reviewed,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TypeError("semantic metric unsafe rate is inconsistent")
    issued = machine["issued"]
    if issued_reviewed > issued:
        raise TypeError("semantic metric issued review coverage exceeds issued grants")
    if issued == 0:
        if issued_reviewed != 0 or issued_review_rate is not None:
            raise TypeError("empty issued grants require a null review coverage rate")
    elif issued_review_rate is None or not math.isclose(
        issued_review_rate,
        issued_reviewed / issued,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TypeError("semantic metric issued review coverage is inconsistent")
    return {
        "schema_version": 1,
        "window": bounded_text(
            raw.get("window"), label="semantic metric window", nullable=True
        ),
        "action_id": action_id,
        "tenant_bucket_sha256": digest(
            raw.get("tenant_bucket_sha256"),
            label="semantic metric tenant bucket",
            nullable=True,
        ),
        "epoch_id": bounded_text(
            raw.get("epoch_id"), label="semantic metric epoch id", nullable=True
        ),
        "risk": enum(
            raw.get("risk"),
            label="semantic metric risk",
            allowed=frozenset({"low", "medium", "high", "critical"}),
            nullable=True,
        ),
        "machine": machine,
        "actual_auto_approval": actual,
        "review_metrics": {
            "reviewed": reviewed,
            "safe": safe,
            "unsafe": unsafe,
            "unsafe_rate": unsafe_rate,
            "issued_reviewed": issued_reviewed,
            "issued_review_rate": issued_review_rate,
        },
    }
