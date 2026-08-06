from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from agent_libos.models.base import StrEnum
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import (
    SemanticAssessmentKind,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticDataFinding,
    SemanticDomain,
    SemanticFinding,
    SemanticPredicate,
    SemanticReasonCode,
)
from agent_libos.utils.serde import bounded_json_loads


SEMANTIC_PROJECTION_MAX_BYTES = 16 * 1024
SEMANTIC_ASSESSMENT_RECORD_MAX_BYTES = 256 * 1024
SEMANTIC_QUERY_HARD_LIMIT = 500
# Assessment rows cross JSON/JavaScript read surfaces.  Keep every durable
# counter exactly representable by those consumers as well as by the store.
SEMANTIC_RECORD_INTEGER_MAX = (1 << 53) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEMANTIC_ACTION = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_TERMINAL_JOB_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "egress_blocked",
        "provider_outcome_unknown",
        "cancelled",
        "expired",
    }
)
_PROJECTION_FORBIDDEN_KEYS = frozenset(
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
_BINDING_DIGEST_KEYS = frozenset(
    {
        "artifact_sha256",
        "input_sha256",
        "feature_snapshot_sha256",
        "policy_sha256",
        "manifest_sha256",
        "action_sha256",
        "resource_sha256",
        "args_sha256",
        "state_sha256",
        "source_refs_sha256",
        "data_labels_sha256",
        "sink_identity_sha256",
        "tool_schema_sha256",
        "provider_spec_sha256",
        "tenant_bucket_sha256",
    }
)
_REQUIRED_BINDING_DIGEST_KEYS = frozenset(
    {
        "artifact_sha256",
        "input_sha256",
        "feature_snapshot_sha256",
        "policy_sha256",
        "action_sha256",
    }
)
_SEMANTIC_JOB_ERROR_CODES = frozenset(
    {
        "abstained",
        "capture_failed",
        "disabled",
        "egress_blocked",
        "invalid_schema",
        "out_of_distribution",
        "projection_expired",
        "provider_error",
        "provider_outcome_unknown",
        "stale_input",
        "timeout",
        "worker_error",
    }
)
_SEMANTIC_HUMAN_OUTCOMES = frozenset(
    {
        "pending",
        "approved",
        "rejected",
        "edited",
        "cancelled",
        "delivered",
    }
)


class SemanticAssessmentJobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EGRESS_BLOCKED = "egress_blocked"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SemanticProjectionRetention(StrEnum):
    REDACTED = "redacted"
    HASH_ONLY = "hash_only"


_SEMANTIC_SHADOW_OUTCOMES = frozenset(
    {"would_issue_exact_once", "require_human", "would_deny"}
)


def _semantic_count_mapping(
    value: Mapping[str, int],
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValidationError(
            f"{label} must contain every canonical outcome exactly once"
        )
    selected: dict[str, int] = {}
    for key in sorted(expected):
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError(
                f"{label} values must be non-negative exact integers"
            )
        selected[key] = count
    return MappingProxyType(selected)


@dataclass(frozen=True, slots=True)
class SemanticStatusAggregate:
    """One database-snapshot view of queue and assessment health."""

    job_total: int
    job_counts: Mapping[str, int]
    assessment_total: int
    assessment_status_counts: Mapping[str, int]
    assessment_domain_counts: Mapping[str, int]
    assessment_ood_count: int
    shadow_outcome_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        for label, count in (
            ("semantic job total", self.job_total),
            ("semantic assessment total", self.assessment_total),
            ("semantic assessment OOD count", self.assessment_ood_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValidationError(
                    f"{label} must be a non-negative exact integer"
                )
        jobs = _semantic_count_mapping(
            self.job_counts,
            expected=frozenset(item.value for item in SemanticAssessmentJobStatus),
            label="semantic job status counts",
        )
        assessments = _semantic_count_mapping(
            self.assessment_status_counts,
            expected=frozenset(item.value for item in SemanticAssessmentStatus),
            label="semantic assessment status counts",
        )
        domains = _semantic_count_mapping(
            self.assessment_domain_counts,
            expected=frozenset(item.value for item in SemanticDomain),
            label="semantic assessment domain counts",
        )
        outcomes = _semantic_count_mapping(
            self.shadow_outcome_counts,
            expected=_SEMANTIC_SHADOW_OUTCOMES,
            label="semantic shadow outcome counts",
        )
        if sum(jobs.values()) != self.job_total:
            raise ValidationError("semantic job status counts do not match total")
        if sum(assessments.values()) != self.assessment_total:
            raise ValidationError("semantic assessment status counts do not match total")
        if sum(domains.values()) != self.assessment_total:
            raise ValidationError(
                "semantic assessment domain counts do not match total"
            )
        if sum(outcomes.values()) != self.assessment_total:
            raise ValidationError(
                "semantic shadow outcome counts do not match total"
            )
        if self.assessment_ood_count != assessments[SemanticAssessmentStatus.OOD.value]:
            raise ValidationError(
                "semantic assessment OOD count does not match OOD status count"
            )
        object.__setattr__(self, "job_counts", jobs)
        object.__setattr__(self, "assessment_status_counts", assessments)
        object.__setattr__(self, "assessment_domain_counts", domains)
        object.__setattr__(self, "shadow_outcome_counts", outcomes)


def _require_text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValidationError(f"{label} must be bounded non-empty text")
    return value


def _optional_text(
    value: object,
    label: str,
    *,
    maximum: int = 512,
) -> str | None:
    if value is None:
        return None
    return _require_text(value, label, maximum=maximum)


def _parse_semantic_timestamp(value: object, label: str) -> datetime:
    selected = _require_text(value, label, maximum=128)
    try:
        parsed = datetime.fromisoformat(selected.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_semantic_timestamp(value: object, label: str) -> str:
    return _parse_semantic_timestamp(value, label).isoformat(
        timespec="microseconds"
    )


def _optional_semantic_timestamp(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _canonical_semantic_timestamp(value, label)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label)


def _strict_json_mapping(
    value: Mapping[str, Any],
    *,
    label: str,
    maximum_bytes: int,
) -> dict[str, Any]:
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
        decoded = bounded_json_loads(encoded, max_bytes=maximum_bytes)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be bounded strict JSON") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - mapping round-trip defense
        raise ValidationError(f"{label} must be a JSON object")
    return decoded


def _reject_projection_content(value: Mapping[str, Any]) -> None:
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                normalized = str(key).strip().casefold()
                if normalized in _PROJECTION_FORBIDDEN_KEYS:
                    raise ValidationError(
                        "semantic job projection contains a forbidden content field"
                    )
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)


def _validated_bindings(value: Mapping[str, Any]) -> dict[str, str | None]:
    selected = _strict_json_mapping(
        value,
        label="semantic job bindings",
        maximum_bytes=8 * 1024,
    )
    unknown = sorted(set(selected) - _BINDING_DIGEST_KEYS)
    missing = sorted(_REQUIRED_BINDING_DIGEST_KEYS - set(selected))
    if unknown or missing:
        raise ValidationError(
            "semantic job binding digest keys are invalid: "
            f"unknown={unknown}, missing={missing}"
        )
    result: dict[str, str | None] = {}
    for key, item in selected.items():
        result[key] = _optional_sha256(item, f"semantic job {key}")
    for key in _REQUIRED_BINDING_DIGEST_KEYS:
        if result[key] is None:
            raise ValidationError(f"semantic job {key} cannot be null")
    return result


@dataclass(frozen=True, slots=True)
class SemanticAssessmentCursor:
    created_at: str
    assessment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "created_at",
            _canonical_semantic_timestamp(
                self.created_at,
                "semantic assessment cursor timestamp",
            ),
        )
        _require_text(self.assessment_id, "semantic assessment cursor id")


def _validate_semantic_job_identity(record: Any) -> None:
    _require_text(record.job_id, "semantic job id")
    _require_text(record.kind, "semantic job kind", maximum=128)
    _require_text(record.domain, "semantic job domain", maximum=128)
    try:
        SemanticAssessmentKind(record.kind)
        SemanticDomain(record.domain)
    except ValueError as exc:
        raise ValidationError("semantic job kind/domain is invalid") from exc
    if not isinstance(record.status, SemanticAssessmentJobStatus):
        raise ValidationError("semantic job status is invalid")
    for label, value in (
        ("assessment id", record.assessment_id),
        ("pid", record.pid),
        ("request id", record.request_id),
        ("operation id", record.operation_id),
        ("effect id", record.effect_id),
        ("lease owner id", record.lease_owner_id),
        ("lease id", record.lease_id),
    ):
        _optional_text(value, f"semantic job {label}", maximum=1024)
    if (
        record.error_code is not None
        and record.error_code not in _SEMANTIC_JOB_ERROR_CODES
    ):
        raise ValidationError("semantic job error code is invalid")
    object.__setattr__(
        record,
        "created_at",
        _canonical_semantic_timestamp(record.created_at, "semantic job created_at"),
    )
    object.__setattr__(
        record,
        "updated_at",
        _canonical_semantic_timestamp(record.updated_at, "semantic job updated_at"),
    )
    for field_name, label in (
        ("lease_expires_at", "semantic job lease expiry"),
        ("projection_expires_at", "semantic job projection expiry"),
        ("completed_at", "semantic job completion timestamp"),
    ):
        object.__setattr__(
            record,
            field_name,
            _optional_semantic_timestamp(getattr(record, field_name), label),
        )


def _validate_semantic_job_counters(record: Any) -> None:
    if (
        isinstance(record.revision, bool)
        or not isinstance(record.revision, int)
        or record.revision < 0
    ):
        raise ValidationError("semantic job revision must be non-negative")
    if (
        isinstance(record.attempt_count, bool)
        or not isinstance(record.attempt_count, int)
        or record.attempt_count not in {0, 1}
    ):
        raise ValidationError("semantic job attempt count must be zero or one")
    _require_sha256(record.projection_sha256, "semantic job projection digest")
    if not isinstance(record.projection_retention, SemanticProjectionRetention):
        raise ValidationError("semantic job projection retention is invalid")


def _validate_semantic_job_lease(record: Any) -> None:
    leased = record.status is SemanticAssessmentJobStatus.CLAIMED
    lease_fields_present = (
        record.lease_owner_id is not None,
        record.lease_id is not None,
        record.lease_expires_at is not None,
    )
    if leased != all(lease_fields_present) or any(lease_fields_present) != all(
        lease_fields_present
    ):
        raise ValidationError("semantic job lease fields are incoherent")
    if leased and _parse_semantic_timestamp(
        record.lease_expires_at,
        "semantic job lease expiry",
    ) <= _parse_semantic_timestamp(record.updated_at, "semantic job updated_at"):
        raise ValidationError("semantic job lease expiry must follow its claim time")


def _validate_semantic_job_terminal(record: Any, projection: Mapping[str, Any]) -> None:
    terminal = record.status.value in _TERMINAL_JOB_STATUSES
    if not terminal:
        action_id = projection.get("action_id")
        if (
            not isinstance(action_id, str)
            or _SEMANTIC_ACTION.fullmatch(action_id) is None
        ):
            raise ValidationError(
                "active semantic job projection requires a valid action_id"
            )
        if record.error_code is not None:
            raise ValidationError("active semantic job cannot have an error code")
        if record.completed_at is not None or record.assessment_id is not None:
            raise ValidationError(
                "active semantic job cannot have terminal assessment fields"
            )
        return
    if record.completed_at is None or record.assessment_id is None:
        raise ValidationError(
            "terminal semantic job requires completion and assessment ids"
        )
    if (
        record.projection_retention is not SemanticProjectionRetention.HASH_ONLY
        or projection != {}
        or record.projection_expires_at is not None
    ):
        raise ValidationError(
            "terminal semantic job projection must be scrubbed to hash-only"
        )


def _validate_semantic_job_projection_digest(
    record: Any,
    projection: Mapping[str, Any],
) -> None:
    if record.status.value in _TERMINAL_JOB_STATUSES:
        return
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(encoded).hexdigest(),
        record.projection_sha256,
    ):
        raise ValidationError("semantic job projection digest does not match payload")


@dataclass(frozen=True, slots=True)
class SemanticAssessmentJobRecord:
    job_id: str
    kind: str
    status: SemanticAssessmentJobStatus
    domain: str
    bindings: Mapping[str, str | None]
    projection: Mapping[str, Any]
    projection_sha256: str
    projection_retention: SemanticProjectionRetention
    created_at: str
    updated_at: str
    assessment_id: str | None = None
    pid: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    effect_id: str | None = None
    revision: int = 0
    attempt_count: int = 0
    lease_owner_id: str | None = None
    lease_id: str | None = None
    lease_expires_at: str | None = None
    projection_expires_at: str | None = None
    error_code: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        _validate_semantic_job_identity(self)
        _validate_semantic_job_counters(self)
        bindings = _validated_bindings(self.bindings)
        projection = _strict_json_mapping(
            self.projection,
            label="semantic job projection",
            maximum_bytes=SEMANTIC_PROJECTION_MAX_BYTES,
        )
        _reject_projection_content(projection)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "projection", projection)
        _validate_semantic_job_projection_digest(self, projection)
        _validate_semantic_job_lease(self)
        _validate_semantic_job_terminal(self, projection)


_ASSESSMENT_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "assessment_id",
        "job_id",
        "kind",
        "status",
        "domain",
        "action_id",
        "pid",
        "request_id",
        "operation_id",
        "effect_id",
        "shadow_outcome",
        "reason_codes",
        "ood",
        "abstain",
        "confidence_bps",
        "calibration_bucket",
        "input_tokens",
        "output_tokens",
        "cost_microunits",
        "classifier_id",
        "classifier_version",
        "artifact_sha256",
        "input_sha256",
        "feature_snapshot_sha256",
        "policy_sha256",
        "manifest_sha256",
        "action_sha256",
        "resource_sha256",
        "args_sha256",
        "state_sha256",
        "projection_sha256",
        "created_at",
        "completed_at",
        "latency_ms",
        "human_outcome",
        "findings",
        "data_findings",
        "matched_rule_ids",
        "proven_predicates",
        "missing_predicates",
        "source_refs_sha256",
        "data_labels_sha256",
        "sink_identity_sha256",
        "tool_schema_sha256",
        "provider_spec_sha256",
        "tenant_bucket_sha256",
    }
)


def _string_tuple(value: object, label: str, *, maximum: int = 256) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise ValidationError(f"{label} must be a bounded list")
    return tuple(
        _require_text(item, f"{label} item", maximum=4096)
        for item in value
    )


def _finding_tuple(
    value: object,
    *,
    label: str,
    keys: frozenset[str],
    model: type[SemanticFinding] | type[SemanticDataFinding],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise ValidationError(f"{label} must be a bounded list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != keys:
            raise ValidationError(f"{label} item has invalid fields")
        selected = _strict_json_mapping(
            item,
            label=f"{label} item",
            maximum_bytes=16 * 1024,
        )
        try:
            result.append(model.from_dict(selected).to_dict())
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label} item is invalid") from exc
    return tuple(result)


_FINDING_KEYS = frozenset(
    {"code", "severity", "confidence_bps", "evidence_sha256", "source"}
)
_DATA_FINDING_KEYS = frozenset(
    {
        "category",
        "field",
        "span_start",
        "span_end",
        "sensitivity_floor",
        "integrity_ceiling",
        "trust_ceiling",
        "confidence_bps",
        "evidence_sha256",
    }
)


def _validate_semantic_assessment_identity(record: Any) -> None:
    if record.schema_version != 1 or isinstance(record.schema_version, bool):
        raise ValidationError("semantic assessment record schema version must be 1")
    for label, value, maximum in (
        ("assessment id", record.assessment_id, 512),
        ("job id", record.job_id, 512),
        ("kind", record.kind, 128),
        ("status", record.status, 128),
        ("domain", record.domain, 128),
        ("action id", record.action_id, 128),
        ("shadow outcome", record.shadow_outcome, 128),
        ("classifier id", record.classifier_id, 512),
        ("classifier version", record.classifier_version, 512),
        ("created_at", record.created_at, 128),
        ("completed_at", record.completed_at, 128),
    ):
        _require_text(value, f"semantic assessment {label}", maximum=maximum)
    if _SEMANTIC_ACTION.fullmatch(record.action_id) is None:
        raise ValidationError(
            "semantic assessment action_id must be a dotted lower-case identifier"
        )
    created_at = _canonical_semantic_timestamp(
        record.created_at,
        "semantic assessment created_at",
    )
    completed_at = _canonical_semantic_timestamp(
        record.completed_at,
        "semantic assessment completed_at",
    )
    if _parse_semantic_timestamp(
        completed_at,
        "semantic assessment completed_at",
    ) < _parse_semantic_timestamp(
        created_at,
        "semantic assessment created_at",
    ):
        raise ValidationError(
            "semantic assessment completion cannot precede creation"
        )
    object.__setattr__(record, "created_at", created_at)
    object.__setattr__(record, "completed_at", completed_at)
    if record.shadow_outcome not in {
        "would_issue_exact_once",
        "require_human",
        "would_deny",
    }:
        raise ValidationError("semantic assessment shadow outcome is invalid")
    try:
        SemanticAssessmentKind(record.kind)
        SemanticDomain(record.domain)
        SemanticAssessmentStatus(record.status)
        SemanticCalibrationBucket(record.calibration_bucket)
    except ValueError as exc:
        raise ValidationError(
            "semantic assessment kind/domain/status/calibration is invalid"
        ) from exc
    for label, value in (
        ("pid", record.pid),
        ("request id", record.request_id),
        ("operation id", record.operation_id),
        ("effect id", record.effect_id),
    ):
        _optional_text(value, f"semantic assessment {label}", maximum=1024)
    if (
        record.human_outcome is not None
        and record.human_outcome not in _SEMANTIC_HUMAN_OUTCOMES
    ):
        raise ValidationError("semantic assessment human outcome is invalid")


def _validate_semantic_assessment_numbers(record: Any) -> None:
    if not isinstance(record.ood, bool) or not isinstance(record.abstain, bool):
        raise ValidationError("semantic assessment OOD/abstain flags must be boolean")
    status = SemanticAssessmentStatus(record.status)
    if record.ood != (status is SemanticAssessmentStatus.OOD):
        raise ValidationError("semantic assessment OOD status and flag must match")
    if record.abstain != (status is SemanticAssessmentStatus.ABSTAINED):
        raise ValidationError("semantic assessment abstained status and flag must match")
    if (
        isinstance(record.confidence_bps, bool)
        or not isinstance(record.confidence_bps, int)
        or not 0 <= record.confidence_bps <= 10_000
    ):
        raise ValidationError("semantic assessment confidence_bps is invalid")
    if (
        isinstance(record.latency_ms, bool)
        or not isinstance(record.latency_ms, int)
        or not 0 <= record.latency_ms <= SEMANTIC_RECORD_INTEGER_MAX
    ):
        raise ValidationError("semantic assessment latency_ms is invalid")
    for label, value in (
        ("input_tokens", record.input_tokens),
        ("output_tokens", record.output_tokens),
        ("cost_microunits", record.cost_microunits),
    ):
        if value is not None and (
            type(value) is not int
            or not 0 <= value <= SEMANTIC_RECORD_INTEGER_MAX
        ):
            raise ValidationError(
                f"semantic assessment {label} must be an exact integer from 0 through 2^53-1 or null"
            )


def _validate_semantic_assessment_digests(record: Any) -> None:
    for label, value in (
        ("artifact", record.artifact_sha256),
        ("input", record.input_sha256),
        ("feature snapshot", record.feature_snapshot_sha256),
        ("policy", record.policy_sha256),
        ("action", record.action_sha256),
        ("projection", record.projection_sha256),
    ):
        _require_sha256(value, f"semantic assessment {label} digest")
    for label, value in (
        ("manifest", record.manifest_sha256),
        ("resource", record.resource_sha256),
        ("args", record.args_sha256),
        ("state", record.state_sha256),
        ("source refs", record.source_refs_sha256),
        ("data labels", record.data_labels_sha256),
        ("sink identity", record.sink_identity_sha256),
        ("tool schema", record.tool_schema_sha256),
        ("provider spec", record.provider_spec_sha256),
        ("tenant bucket", record.tenant_bucket_sha256),
    ):
        _optional_sha256(value, f"semantic assessment {label} digest")


def _normalize_semantic_assessment_lists(record: Any) -> None:
    reasons = _string_tuple(record.reason_codes, "reason codes")
    try:
        tuple(SemanticReasonCode(value) for value in reasons)
    except ValueError as exc:
        raise ValidationError("semantic assessment reason code is invalid") from exc
    proven = _string_tuple(record.proven_predicates, "proven predicates")
    missing = _string_tuple(record.missing_predicates, "missing predicates")
    try:
        tuple(SemanticPredicate(value) for value in (*proven, *missing))
    except ValueError as exc:
        raise ValidationError("semantic assessment predicate is invalid") from exc
    object.__setattr__(record, "reason_codes", reasons)
    object.__setattr__(
        record,
        "matched_rule_ids",
        _string_tuple(record.matched_rule_ids, "matched rule ids"),
    )
    object.__setattr__(record, "proven_predicates", proven)
    object.__setattr__(record, "missing_predicates", missing)
    object.__setattr__(
        record,
        "findings",
        _finding_tuple(
            record.findings,
            label="semantic findings",
            keys=_FINDING_KEYS,
            model=SemanticFinding,
        ),
    )
    object.__setattr__(
        record,
        "data_findings",
        _finding_tuple(
            record.data_findings,
            label="semantic data findings",
            keys=_DATA_FINDING_KEYS,
            model=SemanticDataFinding,
        ),
    )


def _validate_semantic_assessment_size(record: Any) -> None:
    _strict_json_mapping(
        record.to_dict(),
        label="semantic assessment record",
        maximum_bytes=SEMANTIC_ASSESSMENT_RECORD_MAX_BYTES,
    )


@dataclass(frozen=True, slots=True)
class SemanticAssessmentRecord:
    assessment_id: str
    job_id: str
    kind: str
    status: str
    domain: str
    action_id: str
    shadow_outcome: str
    reason_codes: tuple[str, ...]
    ood: bool
    abstain: bool
    confidence_bps: int
    calibration_bucket: str
    classifier_id: str
    classifier_version: str
    artifact_sha256: str
    input_sha256: str
    feature_snapshot_sha256: str
    policy_sha256: str
    action_sha256: str
    projection_sha256: str
    created_at: str
    completed_at: str
    latency_ms: int
    findings: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    data_findings: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    matched_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    proven_predicates: tuple[str, ...] = field(default_factory=tuple)
    missing_predicates: tuple[str, ...] = field(default_factory=tuple)
    pid: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    effect_id: str | None = None
    human_outcome: str | None = None
    manifest_sha256: str | None = None
    resource_sha256: str | None = None
    args_sha256: str | None = None
    state_sha256: str | None = None
    source_refs_sha256: str | None = None
    data_labels_sha256: str | None = None
    sink_identity_sha256: str | None = None
    tool_schema_sha256: str | None = None
    provider_spec_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_microunits: int | None = None
    tenant_bucket_sha256: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_semantic_assessment_identity(self)
        _validate_semantic_assessment_numbers(self)
        _validate_semantic_assessment_digests(self)
        _normalize_semantic_assessment_lists(self)
        _validate_semantic_assessment_size(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "domain": self.domain,
            "action_id": self.action_id,
            "pid": self.pid,
            "request_id": self.request_id,
            "operation_id": self.operation_id,
            "effect_id": self.effect_id,
            "shadow_outcome": self.shadow_outcome,
            "reason_codes": list(self.reason_codes),
            "ood": self.ood,
            "abstain": self.abstain,
            "confidence_bps": self.confidence_bps,
            "calibration_bucket": self.calibration_bucket,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_microunits": self.cost_microunits,
            "classifier_id": self.classifier_id,
            "classifier_version": self.classifier_version,
            "artifact_sha256": self.artifact_sha256,
            "input_sha256": self.input_sha256,
            "feature_snapshot_sha256": self.feature_snapshot_sha256,
            "policy_sha256": self.policy_sha256,
            "manifest_sha256": self.manifest_sha256,
            "action_sha256": self.action_sha256,
            "resource_sha256": self.resource_sha256,
            "args_sha256": self.args_sha256,
            "state_sha256": self.state_sha256,
            "projection_sha256": self.projection_sha256,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "human_outcome": self.human_outcome,
            "findings": [dict(item) for item in self.findings],
            "data_findings": [dict(item) for item in self.data_findings],
            "matched_rule_ids": list(self.matched_rule_ids),
            "proven_predicates": list(self.proven_predicates),
            "missing_predicates": list(self.missing_predicates),
            "source_refs_sha256": self.source_refs_sha256,
            "data_labels_sha256": self.data_labels_sha256,
            "sink_identity_sha256": self.sink_identity_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "provider_spec_sha256": self.provider_spec_sha256,
            "tenant_bucket_sha256": self.tenant_bucket_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticAssessmentRecord:
        selected = _strict_json_mapping(
            value,
            label="semantic assessment record",
            maximum_bytes=SEMANTIC_ASSESSMENT_RECORD_MAX_BYTES,
        )
        if set(selected) != _ASSESSMENT_RECORD_KEYS:
            raise ValidationError("semantic assessment record fields are invalid")
        return cls(**selected)


@dataclass(frozen=True, slots=True)
class SemanticAssessmentPage:
    records: tuple[SemanticAssessmentRecord, ...]
    next_cursor: SemanticAssessmentCursor | None = None

    def __post_init__(self) -> None:
        if len(self.records) > SEMANTIC_QUERY_HARD_LIMIT:
            raise ValidationError("semantic assessment page exceeds hard limit")
        if self.next_cursor is not None and not self.records:
            raise ValidationError("empty semantic assessment page cannot have a cursor")
