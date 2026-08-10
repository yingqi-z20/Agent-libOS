from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from agent_libos.models import (
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
)
from agent_libos.models.base import StrEnum
from agent_libos.models.data_flow import integrity_rank, sensitivity_rank
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import (
    SemanticApprovalBindingV2,
    SemanticDataFinding,
    SemanticLegacyFlowHistoryV1,
    SemanticTripCode,
)
from agent_libos.storage.semantic_v6 import (
    SEMANTIC_FLOW_LINEAGE_HARD_LIMIT,
    SEMANTIC_V6_QUERY_HARD_LIMIT,
    SemanticFlowActivityRecord,
    SemanticFlowBundle,
    SemanticFlowEdgeRecord,
    SemanticFlowEntityRecord,
    SemanticFlowLabelAssertionRecord,
    SemanticFlowPage,
    SemanticV6Cursor,
    require_query_limit,
)
from agent_libos.utils.ids import utc_now


FLOW_SCHEMA_VERSION = 1
FLOW_LINEAGE_DEPTH_HARD_LIMIT = 16
FLOW_LABEL_ASSERTION_HARD_LIMIT = 256
FLOW_CURSOR_MAX_BYTES = 2_048
FLOW_JSON_LOCATOR_MAX_BYTES = 4_096
FLOW_JSON_LOCATOR_MAX_DEPTH = 32
FLOW_JSON_LOCATOR_SEGMENT_MAX_CHARS = 256
FLOW_NO_TENANT_BUCKET_SHA256 = hashlib.sha256(
    b"agent-libos:semantic-flow:no-tenant:v1"
).hexdigest()
FLOW_UNBUCKETED_IDENTITY_SHA256 = hashlib.sha256(
    b"agent-libos:semantic-flow:unbucketed-identity:v1"
).hexdigest()


class FlowEntityKind(StrEnum):
    ROOT_GOAL = "root_goal"
    OBJECT_VERSION = "object_version"
    FILE_BINDING_VERSION = "file_binding_version"
    PROVIDER_RESULT = "provider_result"
    TOOL_RESULT = "tool_result"
    MATERIALIZATION = "materialization"
    MODEL_OUTPUT = "model_output"


class FlowActivityKind(StrEnum):
    PROCESS_SPAWN = "process_spawn"
    PROVIDER_CALL = "provider_call"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    OBJECT_CREATE = "object_create"
    OBJECT_UPDATE = "object_update"
    OBJECT_APPEND = "object_append"
    OBJECT_MATERIALIZE = "object_materialize"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    CONDITIONAL = "conditional"
    TOOL_SELECTION = "tool_selection"
    MEMORY_RETRIEVAL = "memory_retrieval"
    OBJECT_READ = "object_read"


class FlowNodeType(StrEnum):
    ENTITY = "entity"
    ACTIVITY = "activity"


class FlowEdgeRelation(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONTROL = "control"


class FlowLabelSource(StrEnum):
    HOST = "host"
    MODEL = "model"
    DETERMINISTIC = "deterministic"


class FlowCoverageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    STALE = "stale"


class FlowLocatorKind(StrEnum):
    JSON_FIELD = "json_field"
    TEXT_CHUNK = "text_chunk"


class FlowLineageDirection(StrEnum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


class FlowMemoryGateReason(StrEnum):
    ALLOW = "allow"
    ENTITY_MISSING = "entity_missing"
    UNKNOWN_COVERAGE = "unknown_coverage"
    CROSS_TENANT = "cross_tenant"
    MIXED_IDENTITY = "mixed_identity"
    LOW_INTEGRITY_CONTROL = "low_integrity_control"
    LOW_TRUST_CONTROL = "low_trust_control"


class FlowEligibilityReason(StrEnum):
    ELIGIBLE = "eligible"
    UNSUPPORTED_ACTION = "unsupported_action"
    ENTITY_MISSING = "entity_missing"
    WRONG_ENTITY_KIND = "wrong_entity_kind"
    CAPTURE_FAILURE = "capture_failure"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    MIXED_IDENTITY = "mixed_identity"
    TENANT_MISMATCH = "tenant_mismatch"
    LABEL_TOO_SENSITIVE = "label_too_sensitive"
    LOW_INTEGRITY = "low_integrity"
    LOW_TRUST = "low_trust"
    CONTENT_DRIFT = "content_drift"
    VERSION_DRIFT = "version_drift"
    STATE_DRIFT = "state_drift"
    ACTION_DRIFT = "action_drift"


_TRUST_ORDER = (
    DataTrustLevel.UNTRUSTED,
    DataTrustLevel.UNKNOWN,
    DataTrustLevel.USER_ASSERTED,
    DataTrustLevel.VERIFIED,
    DataTrustLevel.TRUSTED,
)
_COVERAGE_PRECEDENCE = {
    FlowCoverageStatus.COMPLETE: 0,
    FlowCoverageStatus.PARTIAL: 1,
    FlowCoverageStatus.UNKNOWN: 2,
    FlowCoverageStatus.STALE: 3,
    FlowCoverageStatus.CONFLICT: 4,
}


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValidationError("semantic flow value must be canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    return None if value is None else _require_sha256(value, label)


def _require_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValidationError(f"{label} must be a bounded timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValidationError(f"{label} must be bounded canonical text")
    return value


def _enum(enum_type: type[StrEnum], value: object, label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ValidationError(f"{label} is invalid")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValidationError(f"{label} is invalid") from exc


@dataclass(frozen=True, slots=True)
class FlowLabelVector:
    """Payload-free semantic label coordinates.

    Identity and origin are intentionally absent.  Persisted flow records carry
    only Host-keyed tenant buckets plus boolean identity facts.
    """

    sensitivity: DataSensitivity = DataSensitivity.NORMAL
    trust_level: DataTrustLevel = DataTrustLevel.UNKNOWN
    integrity: DataIntegrity = DataIntegrity.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensitivity",
            _enum(DataSensitivity, self.sensitivity, "flow sensitivity"),
        )
        object.__setattr__(
            self,
            "trust_level",
            _enum(DataTrustLevel, self.trust_level, "flow trust level"),
        )
        object.__setattr__(
            self,
            "integrity",
            _enum(DataIntegrity, self.integrity, "flow integrity"),
        )

    @classmethod
    def from_data_labels(cls, labels: DataLabels) -> FlowLabelVector:
        if not isinstance(labels, DataLabels):
            raise TypeError("flow labels must be DataLabels")
        return cls(
            sensitivity=labels.sensitivity,
            trust_level=labels.trust_level,
            integrity=labels.integrity,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FlowLabelVector:
        if not isinstance(value, Mapping) or set(value) != {
            "sensitivity",
            "trust_level",
            "integrity",
        }:
            raise ValidationError("flow label vector fields are invalid")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, str]:
        return {
            "sensitivity": self.sensitivity.value,
            "trust_level": self.trust_level.value,
            "integrity": self.integrity.value,
        }

    def tighten(self, other: FlowLabelVector) -> FlowLabelVector:
        if not isinstance(other, FlowLabelVector):
            raise TypeError("flow label tightening requires FlowLabelVector")
        return FlowLabelVector(
            sensitivity=(
                other.sensitivity
                if sensitivity_rank(other.sensitivity)
                > sensitivity_rank(self.sensitivity)
                else self.sensitivity
            ),
            trust_level=(
                other.trust_level
                if _TRUST_ORDER.index(other.trust_level)
                < _TRUST_ORDER.index(self.trust_level)
                else self.trust_level
            ),
            integrity=(
                other.integrity
                if integrity_rank(other.integrity) < integrity_rank(self.integrity)
                else self.integrity
            ),
        )

    def is_tightening_of(self, baseline: FlowLabelVector) -> bool:
        if not isinstance(baseline, FlowLabelVector):
            raise TypeError("flow label baseline must be FlowLabelVector")
        return (
            sensitivity_rank(self.sensitivity)
            >= sensitivity_rank(baseline.sensitivity)
            and integrity_rank(self.integrity) <= integrity_rank(baseline.integrity)
            and _TRUST_ORDER.index(self.trust_level)
            <= _TRUST_ORDER.index(baseline.trust_level)
        )


@dataclass(frozen=True, slots=True)
class FlowDataLocator:
    """A field/chunk location containing no path or content."""

    kind: FlowLocatorKind
    locator_sha256: str
    value_sha256: str
    path_sha256s: tuple[str, ...] = ()
    ordinal: int | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    schema_version: int = FLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(FlowLocatorKind, self.kind, "flow locator kind"),
        )
        _require_sha256(self.locator_sha256, "flow locator")
        _require_sha256(self.value_sha256, "flow locator value")
        if not isinstance(self.path_sha256s, tuple) or any(
            not isinstance(value, str) for value in self.path_sha256s
        ):
            raise ValidationError("flow locator path digests must be a frozen tuple")
        for value in self.path_sha256s:
            _require_sha256(value, "flow locator path segment")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValidationError("flow locator schema_version must be 1")
        values = (self.ordinal, self.offset_start, self.offset_end)
        if self.kind is FlowLocatorKind.JSON_FIELD:
            if not 1 <= len(self.path_sha256s) <= FLOW_JSON_LOCATOR_MAX_DEPTH:
                raise ValidationError("JSON flow locator path digest depth is invalid")
            if any(value is not None for value in values):
                raise ValidationError("JSON flow locator must not expose offsets")
            return
        if self.path_sha256s:
            raise ValidationError("text flow locator must not contain JSON path digests")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValidationError("text flow locator requires non-negative offsets")
        assert self.offset_start is not None and self.offset_end is not None
        if self.offset_end <= self.offset_start:
            raise ValidationError("text flow locator end must exceed start")

    @classmethod
    def json_field(
        cls,
        path: Sequence[str | int],
        *,
        value_sha256: str,
    ) -> FlowDataLocator:
        if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
            raise ValidationError("JSON flow path must be a sequence")
        selected = tuple(path)
        if not selected or len(selected) > FLOW_JSON_LOCATOR_MAX_DEPTH:
            raise ValidationError("JSON flow path depth is invalid")
        normalized: list[dict[str, Any]] = []
        for segment in selected:
            if isinstance(segment, bool) or not isinstance(segment, (str, int)):
                raise ValidationError("JSON flow path segment is invalid")
            if isinstance(segment, str):
                if (
                    not segment
                    or len(segment) > FLOW_JSON_LOCATOR_SEGMENT_MAX_CHARS
                    or any(ord(character) < 0x20 for character in segment)
                ):
                    raise ValidationError("JSON flow path segment is invalid")
                normalized.append({"type": "key", "sha256": _digest(segment)})
            else:
                if segment < 0:
                    raise ValidationError("JSON flow array index must be non-negative")
                normalized.append({"type": "index", "value": segment})
        encoded = _canonical_bytes(normalized)
        if len(encoded) > FLOW_JSON_LOCATOR_MAX_BYTES:
            raise ValidationError("JSON flow locator exceeds byte limit")
        selected_value_sha256 = _require_sha256(
            value_sha256,
            "JSON flow value",
        )
        return cls(
            kind=FlowLocatorKind.JSON_FIELD,
            locator_sha256=hashlib.sha256(encoded).hexdigest(),
            value_sha256=selected_value_sha256,
            path_sha256s=tuple(_digest(item) for item in normalized),
        )

    @classmethod
    def text_chunk(
        cls,
        *,
        ordinal: int,
        offset_start: int,
        offset_end: int,
        content_sha256: str,
    ) -> FlowDataLocator:
        selected_content = _require_sha256(
            content_sha256,
            "text flow chunk content",
        )
        descriptor = {
            "kind": FlowLocatorKind.TEXT_CHUNK.value,
            "ordinal": ordinal,
            "offset_start": offset_start,
            "offset_end": offset_end,
            "content_sha256": selected_content,
        }
        return cls(
            kind=FlowLocatorKind.TEXT_CHUNK,
            locator_sha256=_digest(descriptor),
            value_sha256=selected_content,
            ordinal=ordinal,
            offset_start=offset_start,
            offset_end=offset_end,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "locator_sha256": self.locator_sha256,
            "value_sha256": self.value_sha256,
            "path_sha256s": list(self.path_sha256s),
            "ordinal": self.ordinal,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
        }


@dataclass(frozen=True, slots=True)
class FlowNodeRef:
    node_id: str
    node_type: FlowNodeType

    def __post_init__(self) -> None:
        _require_identifier(self.node_id, "flow node reference")
        object.__setattr__(
            self,
            "node_type",
            _enum(FlowNodeType, self.node_type, "flow node reference type"),
        )


@dataclass(frozen=True, slots=True)
class FlowInputEdge:
    source: FlowNodeRef
    relation: FlowEdgeRelation = FlowEdgeRelation.DIRECT

    def __post_init__(self) -> None:
        if not isinstance(self.source, FlowNodeRef):
            raise TypeError("flow input edge source must be FlowNodeRef")
        object.__setattr__(
            self,
            "relation",
            _enum(FlowEdgeRelation, self.relation, "flow input edge relation"),
        )


@dataclass(frozen=True, slots=True)
class FlowOutputEdge:
    target: FlowNodeRef
    relation: FlowEdgeRelation = FlowEdgeRelation.DIRECT

    def __post_init__(self) -> None:
        if not isinstance(self.target, FlowNodeRef):
            raise TypeError("flow output edge target must be FlowNodeRef")
        object.__setattr__(
            self,
            "relation",
            _enum(FlowEdgeRelation, self.relation, "flow output edge relation"),
        )


@dataclass(frozen=True, slots=True)
class FlowMemoryGateDecision:
    allowed: bool
    reason: FlowMemoryGateReason
    coverage: FlowCoverageStatus
    effective_labels: FlowLabelVector | None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("flow memory gate allowed must be boolean")
        object.__setattr__(
            self,
            "reason",
            _enum(FlowMemoryGateReason, self.reason, "flow memory gate reason"),
        )
        object.__setattr__(
            self,
            "coverage",
            _enum(FlowCoverageStatus, self.coverage, "flow memory gate coverage"),
        )
        if self.effective_labels is not None and not isinstance(
            self.effective_labels,
            FlowLabelVector,
        ):
            raise TypeError("flow memory gate labels must be FlowLabelVector")
        if self.allowed != (self.reason is FlowMemoryGateReason.ALLOW):
            raise ValidationError("flow memory gate outcome is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value,
            "coverage": self.coverage.value,
            "effective_labels": (
                self.effective_labels.to_dict()
                if self.effective_labels is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class FlowApprovalEligibility:
    eligible: bool
    action_id: str
    entity_id: str
    coverage: FlowCoverageStatus
    reason_codes: tuple[FlowEligibilityReason, ...]
    effective_labels: FlowLabelVector | None
    content_sha256: str | None
    version_sha256: str | None
    provenance_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("flow approval eligibility must be boolean")
        _require_identifier(self.action_id, "flow eligibility action")
        _require_identifier(self.entity_id, "flow eligibility entity")
        object.__setattr__(
            self,
            "coverage",
            _enum(FlowCoverageStatus, self.coverage, "flow eligibility coverage"),
        )
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise ValidationError("flow eligibility requires reason codes")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _enum(FlowEligibilityReason, value, "flow eligibility reason")
                for value in self.reason_codes
            ),
        )
        if self.effective_labels is not None and not isinstance(
            self.effective_labels,
            FlowLabelVector,
        ):
            raise TypeError("flow eligibility labels must be FlowLabelVector")
        for name in ("content_sha256", "version_sha256", "provenance_sha256"):
            _optional_sha256(getattr(self, name), f"flow eligibility {name}")
        if self.eligible != (
            self.reason_codes == (FlowEligibilityReason.ELIGIBLE,)
            and self.coverage is FlowCoverageStatus.COMPLETE
        ):
            raise ValidationError("flow approval eligibility is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "action_id": self.action_id,
            "entity_id": self.entity_id,
            "coverage": self.coverage.value,
            "reason_codes": [item.value for item in self.reason_codes],
            "effective_labels": (
                self.effective_labels.to_dict()
                if self.effective_labels is not None
                else None
            ),
            "content_sha256": self.content_sha256,
            "version_sha256": self.version_sha256,
            "provenance_sha256": self.provenance_sha256,
        }

    def canonical_sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class FlowEffectiveLabels:
    labels: FlowLabelVector
    coverage: FlowCoverageStatus
    assertion_count: int
    conflict_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.labels, FlowLabelVector):
            raise TypeError("effective flow labels must be FlowLabelVector")
        object.__setattr__(
            self,
            "coverage",
            _enum(FlowCoverageStatus, self.coverage, "flow coverage"),
        )
        for name in ("assertion_count", "conflict_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels.to_dict(),
            "coverage": self.coverage.value,
            "assertion_count": self.assertion_count,
            "conflict_count": self.conflict_count,
        }


@dataclass(frozen=True, slots=True)
class FlowCoverageReport:
    entity_id: str
    status: FlowCoverageStatus
    effective_labels: FlowLabelVector
    entity_count: int
    activity_count: int
    edge_count: int
    assertion_count: int
    max_depth: int

    def __post_init__(self) -> None:
        _require_identifier(self.entity_id, "flow coverage entity")
        object.__setattr__(
            self,
            "status",
            _enum(FlowCoverageStatus, self.status, "flow coverage status"),
        )
        if not isinstance(self.effective_labels, FlowLabelVector):
            raise TypeError("coverage effective_labels must be FlowLabelVector")
        for name in (
            "entity_count",
            "activity_count",
            "edge_count",
            "assertion_count",
            "max_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"flow coverage {name} is invalid")

    @property
    def complete(self) -> bool:
        return self.status is FlowCoverageStatus.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "status": self.status.value,
            "effective_labels": self.effective_labels.to_dict(),
            "entity_count": self.entity_count,
            "activity_count": self.activity_count,
            "edge_count": self.edge_count,
            "assertion_count": self.assertion_count,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True, slots=True)
class _DerivedCaptureFacts:
    entity_kind: FlowEntityKind
    activity_kind: FlowActivityKind
    pid: str
    action_id: str
    content_sha256: str
    version_sha256: str
    state_sha256: str
    provenance_sha256: str
    labels: FlowLabelVector
    tenant_bucket_sha256: str
    coverage: FlowCoverageStatus
    effect_id: str | None
    provider_spec_sha256: str | None
    tool_schema_sha256: str | None
    model_artifact_sha256: str | None
    created_at: str
    activity_id: str
    entity_id: str


@dataclass(frozen=True, slots=True)
class _RootGoalObjectBindingFacts:
    pid: str
    root_entity_id: str
    object_entity_id: str
    root_state_sha256: str
    object_content_sha256: str
    object_version_sha256: str
    object_provenance_sha256: str
    tenant_bucket_sha256: str


@dataclass(frozen=True, slots=True)
class _LineageTraversalResult:
    discovered: tuple[tuple[SemanticFlowEdgeRecord, dict[str, Any]], ...]
    traversal_truncated: bool
    scope_conflict: bool
    cycle_detected: bool
    missing_node: bool


class SemanticFlowRepositoryPort(Protocol):
    """Append-only persistence needed by the semantic flow service."""

    def append_semantic_flow_bundle(
        self,
        *,
        entities: tuple[SemanticFlowEntityRecord, ...] = (),
        activities: tuple[SemanticFlowActivityRecord, ...] = (),
        edges: tuple[SemanticFlowEdgeRecord, ...] = (),
        assertions: tuple[SemanticFlowLabelAssertionRecord, ...] = (),
    ) -> SemanticFlowBundle: ...

    def get_semantic_flow_entity(
        self,
        entity_id: str,
    ) -> SemanticFlowEntityRecord | None: ...

    def get_semantic_flow_activity(
        self,
        activity_id: str,
    ) -> SemanticFlowActivityRecord | None: ...

    def query_semantic_flow_entities(
        self,
        *,
        after: SemanticV6Cursor | None,
        limit: int,
        pid: str | None = None,
        kind: str | None = None,
        tenant_bucket_sha256: str | None = None,
    ) -> SemanticFlowPage: ...

    def query_semantic_flow_activities(
        self,
        *,
        after: SemanticV6Cursor | None,
        limit: int,
        pid: str | None = None,
        kind: str | None = None,
    ) -> SemanticFlowPage: ...

    def query_semantic_flow_edges(
        self,
        *,
        after: SemanticV6Cursor | None,
        limit: int,
        pid: str | None = None,
        relation: str | None = None,
        node_id: str | None = None,
    ) -> SemanticFlowPage: ...

    def query_semantic_flow_label_assertions(
        self,
        *,
        entity_id: str,
        after: SemanticV6Cursor | None,
        limit: int,
    ) -> SemanticFlowPage: ...


def flow_record_to_dict(record: Any) -> dict[str, Any]:
    """Serialize a strict v6 record without dataclass ``asdict`` surprises."""

    if isinstance(record, SemanticFlowEntityRecord):
        labels = FlowLabelVector.from_dict(record.baseline_labels)
        return {
            "schema_version": record.schema_version,
            "entity_id": record.entity_id,
            "kind": record.kind,
            "pid": record.pid,
            "tenant_bucket_sha256": record.tenant_bucket_sha256,
            "content_sha256": record.content_sha256,
            "version_sha256": record.version_sha256,
            "provenance_sha256": record.provenance_sha256,
            "baseline_labels": labels.to_dict(),
            "identity_present": record.identity_present,
            "identity_mixed": record.identity_mixed,
            "coverage": record.coverage,
            "created_at": record.created_at,
        }
    if isinstance(record, SemanticFlowActivityRecord):
        return {
            "schema_version": record.schema_version,
            "activity_id": record.activity_id,
            "kind": record.kind,
            "pid": record.pid,
            "action_id": record.action_id,
            "effect_id": record.effect_id,
            "state_sha256": record.state_sha256,
            "provider_spec_sha256": record.provider_spec_sha256,
            "tool_schema_sha256": record.tool_schema_sha256,
            "model_artifact_sha256": record.model_artifact_sha256,
            "tenant_bucket_sha256": record.tenant_bucket_sha256,
            "created_at": record.created_at,
        }
    if isinstance(record, SemanticFlowEdgeRecord):
        return {
            "schema_version": record.schema_version,
            "edge_id": record.edge_id,
            "relation": record.relation,
            "source_node_id": record.source_node_id,
            "source_node_type": record.source_node_type,
            "target_node_id": record.target_node_id,
            "target_node_type": record.target_node_type,
            "pid": record.pid,
            "provenance_sha256": record.provenance_sha256,
            "created_at": record.created_at,
        }
    if isinstance(record, SemanticFlowLabelAssertionRecord):
        return {
            "schema_version": record.schema_version,
            "assertion_id": record.assertion_id,
            "entity_id": record.entity_id,
            "source": record.source,
            "sensitivity_floor": record.sensitivity_floor,
            "integrity_ceiling": record.integrity_ceiling,
            "trust_ceiling": record.trust_ceiling,
            "evidence_sha256": record.evidence_sha256,
            "assessment_id": record.assessment_id,
            "locator_sha256": record.locator_sha256,
            "locator_kind": record.locator_kind,
            "path_sha256s": list(record.path_sha256s),
            "value_sha256": record.value_sha256,
            "ordinal": record.ordinal,
            "offset_start": record.offset_start,
            "offset_end": record.offset_end,
            "category": record.category,
            "coverage": record.coverage,
            "created_at": record.created_at,
        }
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        selected = to_dict()
        if isinstance(selected, Mapping):
            return dict(selected)
    raise TypeError("unsupported semantic flow record")


def compute_effective_labels(
    baseline: FlowLabelVector | DataLabels | Mapping[str, Any],
    assertions: Iterable[SemanticFlowLabelAssertionRecord],
    *,
    initial_coverage: FlowCoverageStatus | str = FlowCoverageStatus.COMPLETE,
) -> FlowEffectiveLabels:
    """Apply append-only findings conservatively and flag weakening evidence."""

    if isinstance(baseline, DataLabels):
        selected_baseline = FlowLabelVector.from_data_labels(baseline)
    elif isinstance(baseline, FlowLabelVector):
        selected_baseline = baseline
    elif isinstance(baseline, Mapping):
        selected_baseline = FlowLabelVector.from_dict(baseline)
    else:
        raise TypeError("flow label baseline is invalid")
    selected_coverage = _enum(
        FlowCoverageStatus,
        initial_coverage,
        "flow initial coverage",
    )
    effective = selected_baseline
    count = 0
    conflicts = 0
    for assertion in assertions:
        if not isinstance(assertion, SemanticFlowLabelAssertionRecord):
            raise TypeError("flow assertions must be SemanticFlowLabelAssertionRecord")
        count += 1
        suggestion = FlowLabelVector(
            sensitivity=assertion.sensitivity_floor,
            integrity=assertion.integrity_ceiling,
            trust_level=assertion.trust_ceiling,
        )
        if not suggestion.is_tightening_of(selected_baseline):
            conflicts += 1
        else:
            effective = effective.tighten(suggestion)
        assertion_coverage = _enum(
            FlowCoverageStatus,
            assertion.coverage,
            "flow assertion coverage",
        )
        if _COVERAGE_PRECEDENCE[assertion_coverage] > _COVERAGE_PRECEDENCE[
            selected_coverage
        ]:
            selected_coverage = assertion_coverage
    if conflicts:
        selected_coverage = FlowCoverageStatus.CONFLICT
    return FlowEffectiveLabels(
        labels=effective,
        coverage=selected_coverage,
        assertion_count=count,
        conflict_count=conflicts,
    )


def root_goal_entity_id(
    *,
    pid: str,
    content_sha256: str,
    state_sha256: str,
    goal_oid: str | None = None,
    goal_version: int | None = None,
) -> str:
    """Return the stable entity id used for one root-goal version."""

    # ``state_sha256`` already binds the goal version and process revision.
    # Optional raw identifiers are accepted by the capture API only so they
    # can strengthen the separately persisted version/provenance digests; the
    # stable public id remains reconstructable from the assessment ledger.
    _root_goal_binding_sha256(goal_oid=goal_oid, goal_version=goal_version)
    version_sha256 = _digest(
        {
            "kind": FlowEntityKind.ROOT_GOAL.value,
            "pid": _require_identifier(pid, "flow root pid"),
            "content_sha256": _require_sha256(
                content_sha256,
                "flow root content",
            ),
            "state_sha256": _require_sha256(state_sha256, "flow root state"),
        }
    )
    return _stable_id("flowent", version_sha256)


def provider_result_entity_id(
    *,
    pid: str,
    effect_id: str,
    result_sha256: str,
    state_sha256: str,
) -> str:
    """Return the stable entity id used for one provider result version."""

    version_sha256 = _digest(
        {
            "kind": FlowEntityKind.PROVIDER_RESULT.value,
            "pid": _require_identifier(pid, "flow provider pid"),
            "effect_id": _require_identifier(effect_id, "flow provider effect"),
            "content_sha256": _require_sha256(
                result_sha256,
                "flow provider result",
            ),
            "state_sha256": _require_sha256(
                state_sha256,
                "flow provider state",
            ),
        }
    )
    return _stable_id("flowent", version_sha256)


class SemanticFlowService:
    """Payload-free append-only semantic FlowGraph facade."""

    def __init__(
        self,
        repository: SemanticFlowRepositoryPort,
        *,
        capture_enabled: Callable[[], bool] | None = None,
        capture_failure_observer: Callable[..., None] | None = None,
    ) -> None:
        self._repository = repository
        self._capture_enabled = capture_enabled or (lambda: True)
        self._capture_failure_observer = capture_failure_observer
        self._capture_failures = 0

    def capture_root_goal(
        self,
        *,
        pid: str,
        goal_oid: str | None = None,
        goal_version: int | None = None,
        content_sha256: str,
        state_sha256: str,
        provenance_sha256: str,
        labels: DataLabels,
        tenant_bucket_sha256: str | None,
        created_at: str | None = None,
    ) -> SemanticFlowBundle | None:
        if not self._capture_enabled():
            return None
        try:
            selected_pid = _require_identifier(pid, "flow root pid")
            selected_content = _require_sha256(
                content_sha256,
                "flow root content",
            )
            selected_state = _require_sha256(state_sha256, "flow root state")
            selected_source_provenance = _require_sha256(
                provenance_sha256,
                "flow root provenance",
            )
            goal_binding_sha256 = _root_goal_binding_sha256(
                goal_oid=goal_oid,
                goal_version=goal_version,
            )
            selected_provenance = _digest(
                {
                    "source_provenance_sha256": selected_source_provenance,
                    "goal_binding_sha256": goal_binding_sha256,
                }
            )
            vector = FlowLabelVector.from_data_labels(labels)
            bucket, coverage = _tenant_bucket_and_coverage(
                labels,
                tenant_bucket_sha256,
            )
            if goal_oid is None or goal_version is None:
                coverage = _worse_coverage(
                    coverage,
                    FlowCoverageStatus.PARTIAL,
                )
            timestamp = _require_timestamp(
                created_at or utc_now(),
                "flow root created_at",
            )
            version_sha256 = _digest(
                {
                    "kind": FlowEntityKind.ROOT_GOAL.value,
                    "pid": selected_pid,
                    "content_sha256": selected_content,
                    "state_sha256": selected_state,
                    "goal_binding_sha256": goal_binding_sha256,
                }
            )
            entity_id = root_goal_entity_id(
                pid=selected_pid,
                content_sha256=selected_content,
                state_sha256=selected_state,
                goal_oid=goal_oid,
                goal_version=goal_version,
            )
            activity_id = _stable_id(
                "flowact",
                _digest(
                    {
                        "kind": FlowActivityKind.PROCESS_SPAWN.value,
                        "pid": selected_pid,
                        "state_sha256": selected_state,
                    }
                ),
            )
            edge_id = _edge_id(
                FlowEdgeRelation.CONTROL,
                entity_id,
                activity_id,
                selected_provenance,
            )
            entity = SemanticFlowEntityRecord(
                entity_id=entity_id,
                kind=FlowEntityKind.ROOT_GOAL.value,
                pid=selected_pid,
                tenant_bucket_sha256=bucket,
                content_sha256=selected_content,
                version_sha256=version_sha256,
                provenance_sha256=selected_provenance,
                baseline_labels=vector.to_dict(),
                identity_present=_identity_present(labels),
                identity_mixed=labels.is_mixed_identity,
                coverage=coverage.value,
                created_at=timestamp,
            )
            activity = SemanticFlowActivityRecord(
                activity_id=activity_id,
                kind=FlowActivityKind.PROCESS_SPAWN.value,
                pid=selected_pid,
                action_id="runtime.root_goal",
                effect_id=None,
                state_sha256=selected_state,
                provider_spec_sha256=None,
                tool_schema_sha256=None,
                model_artifact_sha256=None,
                tenant_bucket_sha256=bucket,
                created_at=timestamp,
            )
            edge = SemanticFlowEdgeRecord(
                edge_id=edge_id,
                relation=FlowEdgeRelation.CONTROL.value,
                source_node_id=entity_id,
                source_node_type=FlowNodeType.ENTITY.value,
                target_node_id=activity_id,
                target_node_type=FlowNodeType.ACTIVITY.value,
                pid=selected_pid,
                provenance_sha256=selected_provenance,
                created_at=timestamp,
            )
            return self._append_bundle(
                entities=(entity,),
                activities=(activity,),
                edges=(edge,),
            )
        except Exception:
            self._record_capture_failure("root_goal")
            raise

    def capture_provider_ingress(
        self,
        *,
        pid: str,
        effect_id: str,
        action_id: str,
        result_sha256: str,
        state_sha256: str,
        provider_spec_sha256: str | None,
        tool_schema_sha256: str | None,
        labels: DataLabels,
        tenant_bucket_sha256: str | None,
        provenance_sha256: str | None = None,
        model_artifact_sha256: str | None = None,
        created_at: str | None = None,
    ) -> SemanticFlowBundle | None:
        if not self._capture_enabled():
            return None
        try:
            selected_pid = _require_identifier(pid, "flow provider pid")
            selected_effect = _require_identifier(
                effect_id,
                "flow provider effect",
            )
            selected_action = _require_identifier(
                action_id,
                "flow provider action",
            )
            selected_result = _require_sha256(result_sha256, "flow provider result")
            selected_state = _require_sha256(state_sha256, "flow provider state")
            selected_provider = _optional_sha256(
                provider_spec_sha256,
                "flow provider specification",
            )
            selected_tool = _optional_sha256(
                tool_schema_sha256,
                "flow provider tool schema",
            )
            selected_model = _optional_sha256(
                model_artifact_sha256,
                "flow provider model artifact",
            )
            selected_provenance = (
                _require_sha256(provenance_sha256, "flow provider provenance")
                if provenance_sha256 is not None
                else _digest(
                    {
                        "effect_id": selected_effect,
                        "result_sha256": selected_result,
                        "state_sha256": selected_state,
                        "provider_spec_sha256": selected_provider,
                        "tool_schema_sha256": selected_tool,
                    }
                )
            )
            vector = FlowLabelVector.from_data_labels(labels)
            bucket, tenant_coverage = _tenant_bucket_and_coverage(
                labels,
                tenant_bucket_sha256,
            )
            coverage = tenant_coverage
            if selected_provider is None:
                coverage = _worse_coverage(
                    coverage,
                    FlowCoverageStatus.PARTIAL,
                )
            timestamp = _require_timestamp(
                created_at or utc_now(),
                "flow provider created_at",
            )
            version_sha256 = _digest(
                {
                    "kind": FlowEntityKind.PROVIDER_RESULT.value,
                    "pid": selected_pid,
                    "effect_id": selected_effect,
                    "content_sha256": selected_result,
                    "state_sha256": selected_state,
                }
            )
            entity_id = provider_result_entity_id(
                pid=selected_pid,
                effect_id=selected_effect,
                result_sha256=selected_result,
                state_sha256=selected_state,
            )
            activity_id = _stable_id(
                "flowact",
                _digest(
                    {
                        "kind": FlowActivityKind.PROVIDER_CALL.value,
                        "pid": selected_pid,
                        "effect_id": selected_effect,
                        "state_sha256": selected_state,
                    }
                ),
            )
            edge_id = _edge_id(
                FlowEdgeRelation.DIRECT,
                activity_id,
                entity_id,
                selected_provenance,
            )
            entity = SemanticFlowEntityRecord(
                entity_id=entity_id,
                kind=FlowEntityKind.PROVIDER_RESULT.value,
                pid=selected_pid,
                tenant_bucket_sha256=bucket,
                content_sha256=selected_result,
                version_sha256=version_sha256,
                provenance_sha256=selected_provenance,
                baseline_labels=vector.to_dict(),
                identity_present=_identity_present(labels),
                identity_mixed=labels.is_mixed_identity,
                coverage=coverage.value,
                created_at=timestamp,
            )
            activity = SemanticFlowActivityRecord(
                activity_id=activity_id,
                kind=FlowActivityKind.PROVIDER_CALL.value,
                pid=selected_pid,
                action_id=selected_action,
                effect_id=selected_effect,
                state_sha256=selected_state,
                provider_spec_sha256=selected_provider,
                tool_schema_sha256=selected_tool,
                model_artifact_sha256=selected_model,
                tenant_bucket_sha256=bucket,
                created_at=timestamp,
            )
            edge = SemanticFlowEdgeRecord(
                edge_id=edge_id,
                relation=FlowEdgeRelation.DIRECT.value,
                source_node_id=activity_id,
                source_node_type=FlowNodeType.ACTIVITY.value,
                target_node_id=entity_id,
                target_node_type=FlowNodeType.ENTITY.value,
                pid=selected_pid,
                provenance_sha256=selected_provenance,
                created_at=timestamp,
            )
            return self._append_bundle(
                entities=(entity,),
                activities=(activity,),
                edges=(edge,),
            )
        except Exception:
            self._record_capture_failure("provider_ingress")
            raise

    def bind_root_goal_object(
        self,
        *,
        pid: str,
        root_entity_id: str,
        object_entity_id: str,
        root_state_sha256: str,
        object_content_sha256: str,
        object_version_sha256: str,
        object_provenance_sha256: str,
        tenant_bucket_sha256: str,
    ) -> SemanticFlowBundle | None:
        """Bind the Host root-goal view to its exact initial Object version.

        The relation carries no Object id or payload.  Exact repeats resolve to
        the same activity and edges, while a second target for the same root is
        rejected as ambiguous rather than widening lineage.
        """

        if not self._capture_enabled():
            return None
        failures_before = self._capture_failures
        try:
            facts = self._normalize_root_goal_object_binding(
                pid=pid,
                root_entity_id=root_entity_id,
                object_entity_id=object_entity_id,
                root_state_sha256=root_state_sha256,
                object_content_sha256=object_content_sha256,
                object_version_sha256=object_version_sha256,
                object_provenance_sha256=object_provenance_sha256,
                tenant_bucket_sha256=tenant_bucket_sha256,
            )
            root, object_version = self._require_root_goal_object_records(facts)
            existing_targets = self._root_goal_object_targets(facts.root_entity_id)
            if existing_targets and existing_targets != {facts.object_entity_id}:
                raise ValidationError(
                    "flow root goal is already bound to another Object version"
                )
            binding_provenance = _digest(
                {
                    "schema_version": FLOW_SCHEMA_VERSION,
                    "root_entity_id": facts.root_entity_id,
                    "root_state_sha256": facts.root_state_sha256,
                    "object_entity_id": facts.object_entity_id,
                    "object_content_sha256": facts.object_content_sha256,
                    "object_version_sha256": facts.object_version_sha256,
                    "object_provenance_sha256": facts.object_provenance_sha256,
                    "tenant_bucket_sha256": facts.tenant_bucket_sha256,
                }
            )
            return self.capture_activity(
                activity_kind=FlowActivityKind.TRANSFORMATION,
                pid=facts.pid,
                action_id="runtime.root_goal_object_binding",
                state_sha256=facts.root_state_sha256,
                provenance_sha256=binding_provenance,
                tenant_bucket_sha256=facts.tenant_bucket_sha256,
                inputs=(
                    FlowInputEdge(
                        FlowNodeRef(facts.root_entity_id, FlowNodeType.ENTITY),
                        FlowEdgeRelation.DIRECT,
                    ),
                ),
                outputs=(
                    FlowOutputEdge(
                        FlowNodeRef(facts.object_entity_id, FlowNodeType.ENTITY),
                        FlowEdgeRelation.DIRECT,
                    ),
                ),
                # The target version already existed when this relation was
                # appended.  Its immutable capture timestamp preserves causal
                # ordering and makes an exact repeat byte-for-byte idempotent.
                created_at=object_version.created_at,
            )
        except Exception:
            if self._capture_failures == failures_before:
                self._record_capture_failure("root_goal_object_binding")
            raise

    @staticmethod
    def _normalize_root_goal_object_binding(
        *,
        pid: str,
        root_entity_id: str,
        object_entity_id: str,
        root_state_sha256: str,
        object_content_sha256: str,
        object_version_sha256: str,
        object_provenance_sha256: str,
        tenant_bucket_sha256: str,
    ) -> _RootGoalObjectBindingFacts:
        return _RootGoalObjectBindingFacts(
            pid=_require_identifier(pid, "flow root binding pid"),
            root_entity_id=_require_identifier(
                root_entity_id,
                "flow root binding root entity",
            ),
            object_entity_id=_require_identifier(
                object_entity_id,
                "flow root binding object entity",
            ),
            root_state_sha256=_require_sha256(
                root_state_sha256,
                "flow root binding state",
            ),
            object_content_sha256=_require_sha256(
                object_content_sha256,
                "flow root binding content",
            ),
            object_version_sha256=_require_sha256(
                object_version_sha256,
                "flow root binding object version",
            ),
            object_provenance_sha256=_require_sha256(
                object_provenance_sha256,
                "flow root binding object provenance",
            ),
            tenant_bucket_sha256=_require_sha256(
                tenant_bucket_sha256,
                "flow root binding tenant",
            ),
        )

    def _require_root_goal_object_records(
        self,
        facts: _RootGoalObjectBindingFacts,
    ) -> tuple[SemanticFlowEntityRecord, SemanticFlowEntityRecord]:
        root = self._repository.get_semantic_flow_entity(facts.root_entity_id)
        object_version = self._repository.get_semantic_flow_entity(
            facts.object_entity_id
        )
        if root is None or object_version is None:
            raise ValidationError("flow root binding references an unknown entity")
        self._require_root_goal_object_scope(root, object_version, facts)
        self._require_root_goal_object_digests(root, object_version, facts)
        return root, object_version

    @staticmethod
    def _require_root_goal_object_scope(
        root: SemanticFlowEntityRecord,
        object_version: SemanticFlowEntityRecord,
        facts: _RootGoalObjectBindingFacts,
    ) -> None:
        if root.kind != FlowEntityKind.ROOT_GOAL.value:
            raise ValidationError("flow root binding source is not a root goal")
        if object_version.kind != FlowEntityKind.OBJECT_VERSION.value:
            raise ValidationError("flow root binding target is not an Object version")
        if root.pid != facts.pid or object_version.pid != facts.pid:
            raise ValidationError("flow root binding cannot cross PID scope")
        if (
            root.tenant_bucket_sha256 != facts.tenant_bucket_sha256
            or object_version.tenant_bucket_sha256 != facts.tenant_bucket_sha256
        ):
            raise ValidationError("flow root binding cannot cross tenant scope")

    def _require_root_goal_object_digests(
        self,
        root: SemanticFlowEntityRecord,
        object_version: SemanticFlowEntityRecord,
        facts: _RootGoalObjectBindingFacts,
    ) -> None:
        object_digests = (
            object_version.content_sha256,
            object_version.version_sha256,
            object_version.provenance_sha256,
        )
        expected_digests = (
            facts.object_content_sha256,
            facts.object_version_sha256,
            facts.object_provenance_sha256,
        )
        if root.content_sha256 != facts.object_content_sha256:
            raise ValidationError("flow root binding content digest changed")
        if object_digests != expected_digests:
            raise ValidationError("flow root binding version digest changed")
        identity = (
            root.baseline_labels,
            root.identity_present,
            root.identity_mixed,
        )
        object_identity = (
            object_version.baseline_labels,
            object_version.identity_present,
            object_version.identity_mixed,
        )
        if identity != object_identity:
            raise ValidationError("flow root binding label identity changed")
        if self._root_goal_state(facts.root_entity_id) != facts.root_state_sha256:
            raise ValidationError("flow root binding process state changed")

    def _root_goal_state(self, root_entity_id: str) -> str | None:
        states: set[str] = set()
        after: SemanticV6Cursor | None = None
        while True:
            page = self._repository.query_semantic_flow_edges(
                after=after,
                limit=SEMANTIC_V6_QUERY_HARD_LIMIT,
                pid=None,
                relation=FlowEdgeRelation.CONTROL.value,
                node_id=root_entity_id,
            )
            for edge in page.records:
                if (
                    edge.source_node_type != FlowNodeType.ENTITY.value
                    or edge.source_node_id != root_entity_id
                    or edge.target_node_type != FlowNodeType.ACTIVITY.value
                ):
                    continue
                activity = self._repository.get_semantic_flow_activity(
                    edge.target_node_id
                )
                if (
                    activity is not None
                    and activity.kind == FlowActivityKind.PROCESS_SPAWN.value
                    and activity.action_id == "runtime.root_goal"
                ):
                    states.add(activity.state_sha256)
            if page.next_cursor is None or len(states) > 1:
                break
            after = page.next_cursor
        return next(iter(states)) if len(states) == 1 else None

    def _root_goal_object_targets(self, root_entity_id: str) -> set[str]:
        activities: set[str] = set()
        after: SemanticV6Cursor | None = None
        while True:
            page = self._repository.query_semantic_flow_edges(
                after=after,
                limit=SEMANTIC_V6_QUERY_HARD_LIMIT,
                pid=None,
                relation=FlowEdgeRelation.DIRECT.value,
                node_id=root_entity_id,
            )
            for edge in page.records:
                if (
                    edge.source_node_type != FlowNodeType.ENTITY.value
                    or edge.source_node_id != root_entity_id
                    or edge.target_node_type != FlowNodeType.ACTIVITY.value
                ):
                    continue
                activity = self._repository.get_semantic_flow_activity(
                    edge.target_node_id
                )
                if (
                    activity is not None
                    and activity.kind == FlowActivityKind.TRANSFORMATION.value
                    and activity.action_id == "runtime.root_goal_object_binding"
                ):
                    activities.add(activity.activity_id)
            if page.next_cursor is None:
                break
            after = page.next_cursor
        if len(activities) > 1:
            raise ValidationError("flow root goal has duplicate Object bindings")
        targets: set[str] = set()
        for activity_id in activities:
            after = None
            while True:
                page = self._repository.query_semantic_flow_edges(
                    after=after,
                    limit=SEMANTIC_V6_QUERY_HARD_LIMIT,
                    pid=None,
                    relation=FlowEdgeRelation.DIRECT.value,
                    node_id=activity_id,
                )
                for edge in page.records:
                    if (
                        edge.source_node_type == FlowNodeType.ACTIVITY.value
                        and edge.source_node_id == activity_id
                        and edge.target_node_type == FlowNodeType.ENTITY.value
                    ):
                        targets.add(edge.target_node_id)
                if page.next_cursor is None:
                    break
                after = page.next_cursor
        if activities and len(targets) != 1:
            raise ValidationError("flow root goal Object binding is ambiguous")
        return targets

    def capture_derived_entity(
        self,
        *,
        entity_kind: FlowEntityKind | str,
        activity_kind: FlowActivityKind | str,
        pid: str,
        action_id: str,
        content_sha256: str,
        version_sha256: str,
        state_sha256: str,
        provenance_sha256: str,
        labels: DataLabels,
        tenant_bucket_sha256: str | None,
        inputs: tuple[FlowInputEdge, ...] = (),
        effect_id: str | None = None,
        provider_spec_sha256: str | None = None,
        tool_schema_sha256: str | None = None,
        model_artifact_sha256: str | None = None,
        coverage: FlowCoverageStatus | str = FlowCoverageStatus.COMPLETE,
        created_at: str | None = None,
    ) -> SemanticFlowBundle | None:
        """Capture a Host-bound activity and its immutable output version.

        The interface deliberately accepts only identifiers, typed labels and
        digests.  Callers must hash Object/File/Tool/LLM payloads before crossing
        this boundary.  Input edges are checked against the target PID and
        Host-keyed tenant bucket before the append transaction is attempted.
        """

        if not self._capture_enabled():
            return None
        try:
            facts = self._normalize_derived_capture(
                entity_kind=entity_kind,
                activity_kind=activity_kind,
                pid=pid,
                action_id=action_id,
                content_sha256=content_sha256,
                version_sha256=version_sha256,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                labels=labels,
                tenant_bucket_sha256=tenant_bucket_sha256,
                inputs=inputs,
                effect_id=effect_id,
                provider_spec_sha256=provider_spec_sha256,
                tool_schema_sha256=tool_schema_sha256,
                model_artifact_sha256=model_artifact_sha256,
                coverage=coverage,
                created_at=created_at,
            )
            activity, entity = self._derived_records(facts, labels=labels)
            edges = self._derived_edges(facts, inputs=inputs)
            return self._append_bundle(
                entities=(entity,),
                activities=(activity,),
                edges=edges,
            )
        except Exception:
            self._record_capture_failure("derived_entity")
            raise

    def _normalize_derived_capture(
        self,
        *,
        entity_kind: FlowEntityKind | str,
        activity_kind: FlowActivityKind | str,
        pid: str,
        action_id: str,
        content_sha256: str,
        version_sha256: str,
        state_sha256: str,
        provenance_sha256: str,
        labels: DataLabels,
        tenant_bucket_sha256: str | None,
        inputs: tuple[FlowInputEdge, ...],
        effect_id: str | None,
        provider_spec_sha256: str | None,
        tool_schema_sha256: str | None,
        model_artifact_sha256: str | None,
        coverage: FlowCoverageStatus | str,
        created_at: str | None,
    ) -> _DerivedCaptureFacts:
        selected_entity_kind = _enum(
            FlowEntityKind,
            entity_kind,
            "flow derived entity kind",
        )
        if selected_entity_kind in {
            FlowEntityKind.ROOT_GOAL,
            FlowEntityKind.PROVIDER_RESULT,
        }:
            raise ValidationError(
                "root and provider entities require their dedicated capture"
            )
        selected_activity_kind = _enum(
            FlowActivityKind,
            activity_kind,
            "flow derived activity kind",
        )
        selected_pid = _require_identifier(pid, "flow derived pid")
        selected_action = _require_identifier(action_id, "flow derived action")
        selected_content = _require_sha256(content_sha256, "flow derived content")
        selected_version = _require_sha256(version_sha256, "flow derived version")
        selected_state = _require_sha256(state_sha256, "flow derived state")
        selected_provenance = _require_sha256(
            provenance_sha256,
            "flow derived provenance",
        )
        selected_effect = (
            _require_identifier(effect_id, "flow derived effect")
            if effect_id is not None
            else None
        )
        selected_provider = _optional_sha256(
            provider_spec_sha256,
            "flow derived provider",
        )
        selected_tool = _optional_sha256(tool_schema_sha256, "flow derived tool")
        selected_model = _optional_sha256(
            model_artifact_sha256,
            "flow derived model",
        )
        self._validate_input_edges(inputs)
        vector = FlowLabelVector.from_data_labels(labels)
        bucket, identity_coverage = _tenant_bucket_and_coverage(
            labels,
            tenant_bucket_sha256,
        )
        selected_coverage = self._derived_coverage(
            entity_kind=selected_entity_kind,
            activity_kind=selected_activity_kind,
            inputs=inputs,
            identity_coverage=identity_coverage,
            requested=coverage,
        )
        timestamp = _require_timestamp(
            created_at or utc_now(),
            "flow derived created_at",
        )
        activity_id = _derived_activity_id(
            activity_kind=selected_activity_kind,
            pid=selected_pid,
            action_id=selected_action,
            effect_id=selected_effect,
            state_sha256=selected_state,
            provider_spec_sha256=selected_provider,
            tool_schema_sha256=selected_tool,
            model_artifact_sha256=selected_model,
            content_sha256=selected_content,
            version_sha256=selected_version,
            provenance_sha256=selected_provenance,
        )
        entity_id = _derived_entity_id(
            entity_kind=selected_entity_kind,
            pid=selected_pid,
            version_sha256=selected_version,
            content_sha256=selected_content,
            activity_id=activity_id,
        )
        return _DerivedCaptureFacts(
            entity_kind=selected_entity_kind,
            activity_kind=selected_activity_kind,
            pid=selected_pid,
            action_id=selected_action,
            content_sha256=selected_content,
            version_sha256=selected_version,
            state_sha256=selected_state,
            provenance_sha256=selected_provenance,
            labels=vector,
            tenant_bucket_sha256=bucket,
            coverage=selected_coverage,
            effect_id=selected_effect,
            provider_spec_sha256=selected_provider,
            tool_schema_sha256=selected_tool,
            model_artifact_sha256=selected_model,
            created_at=timestamp,
            activity_id=activity_id,
            entity_id=entity_id,
        )

    @staticmethod
    def _validate_input_edges(inputs: tuple[FlowInputEdge, ...]) -> None:
        if not isinstance(inputs, tuple) or any(
            not isinstance(item, FlowInputEdge) for item in inputs
        ):
            raise TypeError("flow inputs must be a frozen FlowInputEdge tuple")
        if len(inputs) > 256:
            raise ValidationError("flow input edge bound is exceeded")

    @staticmethod
    def _derived_coverage(
        *,
        entity_kind: FlowEntityKind,
        activity_kind: FlowActivityKind,
        inputs: tuple[FlowInputEdge, ...],
        identity_coverage: FlowCoverageStatus,
        requested: FlowCoverageStatus | str,
    ) -> FlowCoverageStatus:
        selected = _worse_coverage(
            identity_coverage,
            _enum(FlowCoverageStatus, requested, "flow derived coverage"),
        )
        input_required = entity_kind in {
            FlowEntityKind.OBJECT_VERSION,
            FlowEntityKind.TOOL_RESULT,
            FlowEntityKind.MATERIALIZATION,
            FlowEntityKind.MODEL_OUTPUT,
        }
        if (
            entity_kind is FlowEntityKind.OBJECT_VERSION
            and activity_kind is FlowActivityKind.OBJECT_CREATE
        ):
            input_required = False
        if input_required and not inputs:
            return _worse_coverage(selected, FlowCoverageStatus.PARTIAL)
        return selected

    @staticmethod
    def _derived_records(
        facts: _DerivedCaptureFacts,
        *,
        labels: DataLabels,
    ) -> tuple[SemanticFlowActivityRecord, SemanticFlowEntityRecord]:
        activity = SemanticFlowActivityRecord(
            activity_id=facts.activity_id,
            kind=facts.activity_kind.value,
            pid=facts.pid,
            action_id=facts.action_id,
            effect_id=facts.effect_id,
            state_sha256=facts.state_sha256,
            provider_spec_sha256=facts.provider_spec_sha256,
            tool_schema_sha256=facts.tool_schema_sha256,
            model_artifact_sha256=facts.model_artifact_sha256,
            tenant_bucket_sha256=facts.tenant_bucket_sha256,
            created_at=facts.created_at,
        )
        entity = SemanticFlowEntityRecord(
            entity_id=facts.entity_id,
            kind=facts.entity_kind.value,
            pid=facts.pid,
            tenant_bucket_sha256=facts.tenant_bucket_sha256,
            content_sha256=facts.content_sha256,
            version_sha256=facts.version_sha256,
            provenance_sha256=facts.provenance_sha256,
            baseline_labels=facts.labels.to_dict(),
            identity_present=_identity_present(labels),
            identity_mixed=labels.is_mixed_identity,
            coverage=facts.coverage.value,
            created_at=facts.created_at,
        )
        return activity, entity

    def _derived_edges(
        self,
        facts: _DerivedCaptureFacts,
        *,
        inputs: tuple[FlowInputEdge, ...],
    ) -> tuple[SemanticFlowEdgeRecord, ...]:
        edges = tuple(self._derived_input_edge(facts, item) for item in inputs)
        return (*edges, _derived_output_edge(facts))

    def _derived_input_edge(
        self,
        facts: _DerivedCaptureFacts,
        input_edge: FlowInputEdge,
    ) -> SemanticFlowEdgeRecord:
        source = self._require_node_scope(
            input_edge.source,
            pid=facts.pid,
            tenant_bucket_sha256=facts.tenant_bucket_sha256,
        )
        source_provenance = getattr(
            source,
            "provenance_sha256",
            getattr(source, "state_sha256", None),
        )
        edge_provenance = _digest(
            {
                "capture_provenance_sha256": facts.provenance_sha256,
                "source_provenance_sha256": source_provenance,
                "activity_state_sha256": facts.state_sha256,
            }
        )
        return SemanticFlowEdgeRecord(
            edge_id=_edge_id(
                input_edge.relation,
                input_edge.source.node_id,
                facts.activity_id,
                edge_provenance,
            ),
            relation=input_edge.relation.value,
            source_node_id=input_edge.source.node_id,
            source_node_type=input_edge.source.node_type.value,
            target_node_id=facts.activity_id,
            target_node_type=FlowNodeType.ACTIVITY.value,
            pid=facts.pid,
            provenance_sha256=edge_provenance,
            created_at=facts.created_at,
        )

    def capture_activity(
        self,
        *,
        activity_kind: FlowActivityKind | str,
        pid: str,
        action_id: str,
        state_sha256: str,
        provenance_sha256: str,
        tenant_bucket_sha256: str,
        inputs: tuple[FlowInputEdge, ...],
        outputs: tuple[FlowOutputEdge, ...],
        effect_id: str | None = None,
        provider_spec_sha256: str | None = None,
        tool_schema_sha256: str | None = None,
        model_artifact_sha256: str | None = None,
        created_at: str | None = None,
    ) -> SemanticFlowBundle | None:
        """Append a payload-free activity between already captured versions."""

        if not self._capture_enabled():
            return None
        try:
            self._validate_activity_edges(inputs, outputs)
            selected_kind = _enum(
                FlowActivityKind,
                activity_kind,
                "flow relation activity kind",
            )
            selected_pid = _require_identifier(pid, "flow relation activity pid")
            selected_action = _require_identifier(
                action_id,
                "flow relation activity action",
            )
            selected_state = _require_sha256(
                state_sha256,
                "flow relation activity state",
            )
            selected_provenance = _require_sha256(
                provenance_sha256,
                "flow relation activity provenance",
            )
            selected_bucket = _require_sha256(
                tenant_bucket_sha256,
                "flow relation activity tenant",
            )
            selected_effect = (
                _require_identifier(effect_id, "flow relation activity effect")
                if effect_id is not None
                else None
            )
            provider = _optional_sha256(
                provider_spec_sha256,
                "flow relation activity provider",
            )
            tool = _optional_sha256(
                tool_schema_sha256,
                "flow relation activity tool",
            )
            model = _optional_sha256(
                model_artifact_sha256,
                "flow relation activity model",
            )
            timestamp = _require_timestamp(
                created_at or utc_now(),
                "flow relation activity created_at",
            )
            activity_id = _relation_activity_id(
                activity_kind=selected_kind,
                pid=selected_pid,
                action_id=selected_action,
                effect_id=selected_effect,
                state_sha256=selected_state,
                provider_spec_sha256=provider,
                tool_schema_sha256=tool,
                model_artifact_sha256=model,
                provenance_sha256=selected_provenance,
                inputs=inputs,
                outputs=outputs,
            )
            activity = SemanticFlowActivityRecord(
                activity_id=activity_id,
                kind=selected_kind.value,
                pid=selected_pid,
                action_id=selected_action,
                effect_id=selected_effect,
                state_sha256=selected_state,
                provider_spec_sha256=provider,
                tool_schema_sha256=tool,
                model_artifact_sha256=model,
                tenant_bucket_sha256=selected_bucket,
                created_at=timestamp,
            )
            edges = self._relation_activity_edges(
                activity,
                provenance_sha256=selected_provenance,
                inputs=inputs,
                outputs=outputs,
            )
            return self._append_bundle(activities=(activity,), edges=edges)
        except Exception:
            self._record_capture_failure("relation_activity")
            raise

    @staticmethod
    def _validate_activity_edges(
        inputs: tuple[FlowInputEdge, ...],
        outputs: tuple[FlowOutputEdge, ...],
    ) -> None:
        if not isinstance(inputs, tuple) or any(
            not isinstance(item, FlowInputEdge) for item in inputs
        ):
            raise TypeError("flow activity inputs must be a frozen tuple")
        if not isinstance(outputs, tuple) or any(
            not isinstance(item, FlowOutputEdge) for item in outputs
        ):
            raise TypeError("flow activity outputs must be a frozen tuple")
        if not inputs or not outputs or len(inputs) + len(outputs) > 256:
            raise ValidationError("flow activity edge set is empty or exceeds its bound")

    def _relation_activity_edges(
        self,
        activity: SemanticFlowActivityRecord,
        *,
        provenance_sha256: str,
        inputs: tuple[FlowInputEdge, ...],
        outputs: tuple[FlowOutputEdge, ...],
    ) -> tuple[SemanticFlowEdgeRecord, ...]:
        incoming = tuple(
            self._relation_input_edge(activity, item, provenance_sha256)
            for item in inputs
        )
        outgoing = tuple(
            self._relation_output_edge(activity, item, provenance_sha256)
            for item in outputs
        )
        return (*incoming, *outgoing)

    def _relation_input_edge(
        self,
        activity: SemanticFlowActivityRecord,
        selected: FlowInputEdge,
        provenance_sha256: str,
    ) -> SemanticFlowEdgeRecord:
        self._require_node_scope(
            selected.source,
            pid=activity.pid,
            tenant_bucket_sha256=activity.tenant_bucket_sha256,
        )
        return _relation_edge(
            relation=selected.relation,
            source=selected.source,
            target=FlowNodeRef(activity.activity_id, FlowNodeType.ACTIVITY),
            pid=activity.pid,
            provenance_sha256=provenance_sha256,
            created_at=activity.created_at,
        )

    def _relation_output_edge(
        self,
        activity: SemanticFlowActivityRecord,
        selected: FlowOutputEdge,
        provenance_sha256: str,
    ) -> SemanticFlowEdgeRecord:
        self._require_node_scope(
            selected.target,
            pid=activity.pid,
            tenant_bucket_sha256=activity.tenant_bucket_sha256,
        )
        return _relation_edge(
            relation=selected.relation,
            source=FlowNodeRef(activity.activity_id, FlowNodeType.ACTIVITY),
            target=selected.target,
            pid=activity.pid,
            provenance_sha256=provenance_sha256,
            created_at=activity.created_at,
        )

    def capture_object_version(
        self,
        *,
        operation: str,
        **facts: Any,
    ) -> SemanticFlowBundle | None:
        kinds = {
            "read": FlowActivityKind.OBJECT_READ,
            "create": FlowActivityKind.OBJECT_CREATE,
            "update": FlowActivityKind.OBJECT_UPDATE,
            "append": FlowActivityKind.OBJECT_APPEND,
        }
        selected_kind = kinds.get(operation)
        if selected_kind is None:
            raise ValidationError("flow object operation is invalid")
        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.OBJECT_VERSION,
            activity_kind=selected_kind,
            **facts,
        )

    def capture_file_version(
        self,
        *,
        operation: str,
        **facts: Any,
    ) -> SemanticFlowBundle | None:
        kinds = {
            "read": FlowActivityKind.FILE_READ,
            "write": FlowActivityKind.FILE_WRITE,
        }
        selected_kind = kinds.get(operation)
        if selected_kind is None:
            raise ValidationError("flow file operation is invalid")
        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.FILE_BINDING_VERSION,
            activity_kind=selected_kind,
            **facts,
        )

    def capture_tool_result(self, **facts: Any) -> SemanticFlowBundle | None:
        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.TOOL_RESULT,
            activity_kind=FlowActivityKind.TOOL_CALL,
            **facts,
        )

    def capture_materialization(self, **facts: Any) -> SemanticFlowBundle | None:
        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.MATERIALIZATION,
            activity_kind=FlowActivityKind.OBJECT_MATERIALIZE,
            **facts,
        )

    def capture_model_output(self, **facts: Any) -> SemanticFlowBundle | None:
        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.MODEL_OUTPUT,
            activity_kind=FlowActivityKind.LLM_CALL,
            **facts,
        )

    def capture_git_snapshot(self, **facts: Any) -> SemanticFlowBundle | None:
        """Capture a frozen local repo/ref/state binding as a source version."""

        return self.capture_derived_entity(
            entity_kind=FlowEntityKind.FILE_BINDING_VERSION,
            activity_kind=FlowActivityKind.PROVIDER_CALL,
            **facts,
        )

    def memory_gate(
        self,
        entity_id: str,
        *,
        target_tenant_bucket_sha256: str,
        relation: FlowEdgeRelation | str,
    ) -> FlowMemoryGateDecision:
        """Fail closed before memory data influences a high-authority action."""

        selected_entity_id = _require_identifier(entity_id, "flow memory entity")
        target_bucket = _require_sha256(
            target_tenant_bucket_sha256,
            "flow memory target tenant",
        )
        selected_relation = _enum(
            FlowEdgeRelation,
            relation,
            "flow memory relation",
        )
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            return FlowMemoryGateDecision(
                allowed=False,
                reason=FlowMemoryGateReason.ENTITY_MISSING,
                coverage=FlowCoverageStatus.UNKNOWN,
                effective_labels=None,
            )
        effective = self.effective_labels(selected_entity_id)
        if entity.tenant_bucket_sha256 != target_bucket:
            return FlowMemoryGateDecision(
                allowed=False,
                reason=FlowMemoryGateReason.CROSS_TENANT,
                coverage=effective.coverage,
                effective_labels=effective.labels,
            )
        if entity.identity_mixed:
            return FlowMemoryGateDecision(
                allowed=False,
                reason=FlowMemoryGateReason.MIXED_IDENTITY,
                coverage=FlowCoverageStatus.CONFLICT,
                effective_labels=effective.labels,
            )
        if effective.coverage is not FlowCoverageStatus.COMPLETE:
            return FlowMemoryGateDecision(
                allowed=False,
                reason=FlowMemoryGateReason.UNKNOWN_COVERAGE,
                coverage=effective.coverage,
                effective_labels=effective.labels,
            )
        if selected_relation is FlowEdgeRelation.CONTROL:
            if integrity_rank(effective.labels.integrity) < integrity_rank(
                DataIntegrity.CHECKED
            ):
                return FlowMemoryGateDecision(
                    allowed=False,
                    reason=FlowMemoryGateReason.LOW_INTEGRITY_CONTROL,
                    coverage=effective.coverage,
                    effective_labels=effective.labels,
                )
            if _TRUST_ORDER.index(effective.labels.trust_level) < _TRUST_ORDER.index(
                DataTrustLevel.USER_ASSERTED
            ):
                return FlowMemoryGateDecision(
                    allowed=False,
                    reason=FlowMemoryGateReason.LOW_TRUST_CONTROL,
                    coverage=effective.coverage,
                    effective_labels=effective.labels,
                )
        return FlowMemoryGateDecision(
            allowed=True,
            reason=FlowMemoryGateReason.ALLOW,
            coverage=effective.coverage,
            effective_labels=effective.labels,
        )

    def approval_eligibility(
        self,
        *,
        action_id: str,
        entity_id: str,
        tenant_bucket_sha256: str,
        current_content_sha256: str,
        current_version_sha256: str,
        current_state_sha256: str,
        max_depth: int = 8,
    ) -> FlowApprovalEligibility:
        """Return authoritative FlowGraph facts for the Phase-4 broker.

        Callers must supply digests freshly resolved from the filesystem or Git
        primitive.  Stored evidence alone can never prove that a binding is
        still current.  Any unknown, partial, conflicting or stale fact fails
        closed without converting the result into an allow predicate.
        """

        selected_action = _require_identifier(action_id, "flow eligibility action")
        selected_entity_id = _require_identifier(
            entity_id,
            "flow eligibility entity",
        )
        selected_tenant = _require_sha256(
            tenant_bucket_sha256,
            "flow eligibility tenant",
        )
        selected_content = _require_sha256(
            current_content_sha256,
            "flow eligibility current content",
        )
        selected_version = _require_sha256(
            current_version_sha256,
            "flow eligibility current version",
        )
        selected_state = _require_sha256(
            current_state_sha256,
            "flow eligibility current state",
        )
        reasons: list[FlowEligibilityReason] = []
        if selected_action not in {"filesystem.read", "git.read", "git.diff"}:
            reasons.append(FlowEligibilityReason.UNSUPPORTED_ACTION)
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            reasons.append(FlowEligibilityReason.ENTITY_MISSING)
            return FlowApprovalEligibility(
                eligible=False,
                action_id=selected_action,
                entity_id=selected_entity_id,
                coverage=FlowCoverageStatus.UNKNOWN,
                reason_codes=tuple(dict.fromkeys(reasons)),
                effective_labels=None,
                content_sha256=None,
                version_sha256=None,
                provenance_sha256=None,
            )
        if entity.kind != FlowEntityKind.FILE_BINDING_VERSION.value:
            reasons.append(FlowEligibilityReason.WRONG_ENTITY_KIND)
        if self.status()["capture_failures"]:
            reasons.append(FlowEligibilityReason.CAPTURE_FAILURE)
        coverage = self.coverage(selected_entity_id, max_depth=max_depth)
        if coverage.status is not FlowCoverageStatus.COMPLETE:
            reasons.append(FlowEligibilityReason.COVERAGE_INCOMPLETE)
        if entity.identity_mixed:
            reasons.append(FlowEligibilityReason.MIXED_IDENTITY)
        if entity.tenant_bucket_sha256 != selected_tenant:
            reasons.append(FlowEligibilityReason.TENANT_MISMATCH)
        if sensitivity_rank(coverage.effective_labels.sensitivity) > sensitivity_rank(
            DataSensitivity.NORMAL
        ):
            reasons.append(FlowEligibilityReason.LABEL_TOO_SENSITIVE)
        if integrity_rank(coverage.effective_labels.integrity) < integrity_rank(
            DataIntegrity.UNKNOWN
        ):
            reasons.append(FlowEligibilityReason.LOW_INTEGRITY)
        if _TRUST_ORDER.index(coverage.effective_labels.trust_level) < _TRUST_ORDER.index(
            DataTrustLevel.UNKNOWN
        ):
            reasons.append(FlowEligibilityReason.LOW_TRUST)
        if entity.content_sha256 != selected_content:
            reasons.append(FlowEligibilityReason.CONTENT_DRIFT)
        if entity.version_sha256 != selected_version:
            reasons.append(FlowEligibilityReason.VERSION_DRIFT)
        activity = self._output_activity_for_entity(selected_entity_id)
        if activity is None or activity.state_sha256 != selected_state:
            reasons.append(FlowEligibilityReason.STATE_DRIFT)
        if activity is None or activity.action_id != selected_action:
            reasons.append(FlowEligibilityReason.ACTION_DRIFT)
        unique_reasons = tuple(dict.fromkeys(reasons))
        if not unique_reasons:
            unique_reasons = (FlowEligibilityReason.ELIGIBLE,)
        return FlowApprovalEligibility(
            eligible=unique_reasons == (FlowEligibilityReason.ELIGIBLE,),
            action_id=selected_action,
            entity_id=selected_entity_id,
            coverage=coverage.status,
            reason_codes=unique_reasons,
            effective_labels=coverage.effective_labels,
            content_sha256=entity.content_sha256,
            version_sha256=entity.version_sha256,
            provenance_sha256=entity.provenance_sha256,
        )

    def append_label_assertion(
        self,
        *,
        entity_id: str,
        finding: SemanticDataFinding,
        source: FlowLabelSource | str,
        assessment_id: str | None,
        locator: FlowDataLocator | None = None,
        coverage: FlowCoverageStatus | str = FlowCoverageStatus.COMPLETE,
        created_at: str | None = None,
    ) -> SemanticFlowLabelAssertionRecord:
        selected_entity_id = _require_identifier(entity_id, "flow assertion entity")
        if not isinstance(finding, SemanticDataFinding):
            raise TypeError("flow label assertion requires SemanticDataFinding")
        selected_source = _enum(FlowLabelSource, source, "flow label source")
        selected_coverage = _enum(
            FlowCoverageStatus,
            coverage,
            "flow assertion coverage",
        )
        if locator is not None and not isinstance(locator, FlowDataLocator):
            raise TypeError("flow assertion locator must be FlowDataLocator")
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            raise ValidationError("flow assertion entity does not exist")
        existing = tuple(
            self._repository.query_semantic_flow_label_assertions(
                entity_id=selected_entity_id,
                after=None,
                limit=FLOW_LABEL_ASSERTION_HARD_LIMIT,
            ).records
        )
        if len(existing) >= FLOW_LABEL_ASSERTION_HARD_LIMIT:
            raise ValidationError("flow assertion bound is exhausted")
        suggestion = FlowLabelVector(
            sensitivity=finding.sensitivity_floor,
            integrity=finding.integrity_ceiling,
            trust_level=finding.trust_ceiling,
        )
        baseline = FlowLabelVector.from_dict(entity.baseline_labels)
        if not suggestion.is_tightening_of(baseline):
            raise ValidationError(
                "flow semantic assertion cannot declassify or endorse"
            )
        timestamp = _require_timestamp(
            created_at or utc_now(),
            "flow assertion created_at",
        )
        locator_sha256 = locator.locator_sha256 if locator is not None else None
        evidence_sha256 = _digest(
            {
                "finding_evidence_sha256": finding.evidence_sha256,
                "locator": locator.to_dict() if locator is not None else None,
                "entity_id": selected_entity_id,
            }
        )
        assertion_id = _stable_id("flowassert", evidence_sha256)
        assertion = SemanticFlowLabelAssertionRecord(
            assertion_id=assertion_id,
            entity_id=selected_entity_id,
            source=selected_source.value,
            sensitivity_floor=finding.sensitivity_floor.value,
            integrity_ceiling=finding.integrity_ceiling.value,
            trust_ceiling=finding.trust_ceiling.value,
            evidence_sha256=evidence_sha256,
            assessment_id=assessment_id,
            locator_sha256=locator_sha256,
            locator_kind=(locator.kind.value if locator is not None else None),
            path_sha256s=(locator.path_sha256s if locator is not None else ()),
            value_sha256=(locator.value_sha256 if locator is not None else None),
            ordinal=(locator.ordinal if locator is not None else None),
            offset_start=(locator.offset_start if locator is not None else None),
            offset_end=(locator.offset_end if locator is not None else None),
            category=finding.category.value,
            coverage=selected_coverage.value,
            created_at=timestamp,
        )
        persisted = self._append_bundle(
            assertions=(assertion,),
        )
        return persisted.assertions[0] if persisted.assertions else assertion

    def append_assessment_findings(
        self,
        *,
        entity_id: str,
        assessment_id: str,
        findings: tuple[SemanticDataFinding, ...],
        source: FlowLabelSource | str = FlowLabelSource.MODEL,
        coverage: FlowCoverageStatus | str = FlowCoverageStatus.COMPLETE,
        created_at: str | None = None,
    ) -> tuple[SemanticFlowLabelAssertionRecord, ...]:
        try:
            return self._append_assessment_findings(
                entity_id=entity_id,
                assessment_id=assessment_id,
                findings=findings,
                source=source,
                coverage=coverage,
                created_at=created_at,
            )
        except Exception:
            self._record_capture_failure("assessment_findings")
            raise

    def _append_assessment_findings(
        self,
        *,
        entity_id: str,
        assessment_id: str,
        findings: tuple[SemanticDataFinding, ...],
        source: FlowLabelSource | str,
        coverage: FlowCoverageStatus | str,
        created_at: str | None,
    ) -> tuple[SemanticFlowLabelAssertionRecord, ...]:
        """Append one assessment's coarse findings without retaining content.

        Existing Phase-1 findings contain only an allowlisted coarse locator or
        a bounded redacted-intent span.  They therefore remain coarse graph
        assertions; JSON field and text chunk locators are added only by Host
        capture points that can prove their exact input version.
        """

        selected_entity_id = _require_identifier(entity_id, "flow assertion entity")
        selected_assessment_id = _require_identifier(
            assessment_id,
            "flow assertion assessment",
        )
        if not isinstance(findings, tuple) or any(
            not isinstance(item, SemanticDataFinding) for item in findings
        ):
            raise TypeError("flow assessment findings must be a frozen tuple")
        if len(findings) > 64:
            raise ValidationError("flow assessment findings exceed bound")
        if not findings:
            return ()
        selected_source = _enum(FlowLabelSource, source, "flow label source")
        selected_coverage = _enum(
            FlowCoverageStatus,
            coverage,
            "flow assertion coverage",
        )
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            raise ValidationError("flow assertion entity does not exist")
        existing = tuple(
            self._repository.query_semantic_flow_label_assertions(
                entity_id=selected_entity_id,
                after=None,
                limit=FLOW_LABEL_ASSERTION_HARD_LIMIT,
            ).records
        )
        if len(existing) + len(findings) > FLOW_LABEL_ASSERTION_HARD_LIMIT:
            raise ValidationError("flow assertion bound is exhausted")
        baseline = FlowLabelVector.from_dict(entity.baseline_labels)
        timestamp = _require_timestamp(
            created_at or utc_now(),
            "flow assertion created_at",
        )
        assertions: list[SemanticFlowLabelAssertionRecord] = []
        for index, finding in enumerate(findings):
            suggestion = FlowLabelVector(
                sensitivity=finding.sensitivity_floor,
                integrity=finding.integrity_ceiling,
                trust_level=finding.trust_ceiling,
            )
            if not suggestion.is_tightening_of(baseline):
                raise ValidationError(
                    "flow semantic assertion cannot declassify or endorse"
                )
            evidence_sha256 = _digest(
                {
                    "assessment_id": selected_assessment_id,
                    "entity_id": selected_entity_id,
                    "finding_index": index,
                    "finding_evidence_sha256": finding.evidence_sha256,
                    "field": finding.field.value,
                    "span_start": finding.span_start,
                    "span_end": finding.span_end,
                }
            )
            assertions.append(
                SemanticFlowLabelAssertionRecord(
                    assertion_id=_stable_id("flowassert", evidence_sha256),
                    entity_id=selected_entity_id,
                    source=selected_source.value,
                    sensitivity_floor=finding.sensitivity_floor.value,
                    integrity_ceiling=finding.integrity_ceiling.value,
                    trust_ceiling=finding.trust_ceiling.value,
                    evidence_sha256=evidence_sha256,
                    assessment_id=selected_assessment_id,
                    locator_sha256=None,
                    category=finding.category.value,
                    coverage=selected_coverage.value,
                    created_at=timestamp,
                )
            )
        persisted = self._append_bundle(
            assertions=tuple(assertions),
        )
        return persisted.assertions or tuple(assertions)

    def effective_labels(self, entity_id: str) -> FlowEffectiveLabels:
        selected_entity_id = _require_identifier(entity_id, "flow entity id")
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            return FlowEffectiveLabels(
                labels=FlowLabelVector(),
                coverage=FlowCoverageStatus.UNKNOWN,
                assertion_count=0,
                conflict_count=0,
            )
        page = self._repository.query_semantic_flow_label_assertions(
            entity_id=selected_entity_id,
            after=None,
            limit=FLOW_LABEL_ASSERTION_HARD_LIMIT,
        )
        effective = compute_effective_labels(
            entity.baseline_labels,
            page.records,
            initial_coverage=entity.coverage,
        )
        if page.next_cursor is None:
            return effective
        # Appends are normally capped before this point, but a concurrent
        # append race or an alternate repository must never make a bounded
        # label read look complete while evidence was omitted.  Preserve the
        # monotonic labels already observed and fail Phase-4 eligibility closed.
        return FlowEffectiveLabels(
            labels=effective.labels,
            coverage=_worse_coverage(
                effective.coverage,
                FlowCoverageStatus.PARTIAL,
            ),
            assertion_count=effective.assertion_count,
            conflict_count=effective.conflict_count,
        )

    def status(self) -> dict[str, Any]:
        aggregate = getattr(self._repository, "semantic_flow_status_aggregate", None)
        if callable(aggregate):
            selected = aggregate()
            if isinstance(selected, Mapping):
                return _normalize_status(selected, self._capture_failures)
            to_dict = getattr(selected, "to_dict", None)
            if callable(to_dict):
                value = to_dict()
                if isinstance(value, Mapping):
                    return _normalize_status(value, self._capture_failures)
        return {
            "schema_version": 1,
            "available": True,
            "counts": {
                "entities": 0,
                "activities": 0,
                "edges": 0,
                "label_assertions": 0,
            },
            "coverage": {status.value: 0 for status in FlowCoverageStatus},
            "capture_failures": self._capture_failures,
            "legacy_history": SemanticLegacyFlowHistoryV1().to_dict(),
        }

    # Runtime/API-facing aliases keep SemanticManager independent from the
    # persistence vocabulary while satisfying ``SemanticFlowPort``.
    def flow_status(self) -> dict[str, Any]:
        return self.status()

    @property
    def capture_failure_count(self) -> int:
        return self._capture_failures

    def get_entity(self, entity_id: str) -> SemanticFlowEntityRecord | None:
        """Read one payload-free entity through the narrow FlowGraph facade."""

        selected = _require_identifier(entity_id, "flow entity")
        return self._repository.get_semantic_flow_entity(selected)

    def list_entities(
        self,
        *,
        after: str | SemanticV6Cursor | None = None,
        limit: int = 50,
        pid: str | None = None,
        kind: FlowEntityKind | str | None = None,
        tenant_bucket_sha256: str | None = None,
    ) -> dict[str, Any]:
        selected_limit = require_query_limit(limit)
        selected_kind = (
            None
            if kind is None
            else _enum(FlowEntityKind, kind, "flow entity kind").value
        )
        page = self._repository.query_semantic_flow_entities(
            after=_decode_cursor(after),
            limit=selected_limit,
            pid=pid,
            kind=selected_kind,
            tenant_bucket_sha256=(
                _optional_sha256(
                    tenant_bucket_sha256,
                    "flow tenant bucket",
                )
            ),
        )
        return _page_dict(page)

    def query_flow_entities(self, **filters: Any) -> dict[str, Any]:
        return self.list_entities(**filters)

    def list_activities(
        self,
        *,
        after: str | SemanticV6Cursor | None = None,
        limit: int = 50,
        pid: str | None = None,
        kind: FlowActivityKind | str | None = None,
    ) -> dict[str, Any]:
        selected_limit = require_query_limit(limit)
        selected_kind = (
            None
            if kind is None
            else _enum(FlowActivityKind, kind, "flow activity kind").value
        )
        page = self._repository.query_semantic_flow_activities(
            after=_decode_cursor(after),
            limit=selected_limit,
            pid=pid,
            kind=selected_kind,
        )
        return _page_dict(page)

    def query_flow_activities(self, **filters: Any) -> dict[str, Any]:
        return self.list_activities(**filters)

    def list_edges(
        self,
        *,
        after: str | SemanticV6Cursor | None = None,
        limit: int = 50,
        pid: str | None = None,
        relation: FlowEdgeRelation | str | None = None,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        selected_limit = require_query_limit(limit)
        selected_relation = (
            None
            if relation is None
            else _enum(FlowEdgeRelation, relation, "flow edge relation").value
        )
        page = self._repository.query_semantic_flow_edges(
            after=_decode_cursor(after),
            limit=selected_limit,
            pid=pid,
            relation=selected_relation,
            node_id=node_id,
        )
        return _page_dict(page)

    def query_flow_edges(self, **filters: Any) -> dict[str, Any]:
        return self.list_edges(**filters)

    def lineage(
        self,
        node_id: str,
        *,
        direction: FlowLineageDirection | str = FlowLineageDirection.UPSTREAM,
        after: str | SemanticV6Cursor | None = None,
        limit: int = 100,
        max_depth: int = 8,
    ) -> dict[str, Any]:
        selected_node_id = _require_identifier(node_id, "flow lineage node id")
        selected_direction = _enum(
            FlowLineageDirection,
            direction,
            "flow lineage direction",
        )
        selected_limit = require_query_limit(
            limit,
            hard_limit=SEMANTIC_FLOW_LINEAGE_HARD_LIMIT,
        )
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 1 <= max_depth <= FLOW_LINEAGE_DEPTH_HARD_LIMIT
        ):
            raise ValidationError(
                f"flow lineage depth must be between 1 and {FLOW_LINEAGE_DEPTH_HARD_LIMIT}"
            )
        query = getattr(self._repository, "query_semantic_flow_lineage", None)
        if not callable(query):
            return self._fallback_lineage(
                selected_node_id,
                direction=selected_direction,
                after=_decode_cursor(after),
                limit=selected_limit,
                max_depth=max_depth,
            )
        page = query(
            node_id=selected_node_id,
            direction=selected_direction.value,
            after=_decode_cursor(after),
            limit=selected_limit,
            max_depth=max_depth,
        )
        return _lineage_dict(
            selected_node_id,
            selected_direction,
            page,
            self.effective_labels(selected_node_id),
        )

    def query_flow_lineage(
        self,
        node_id: str,
        **filters: Any,
    ) -> dict[str, Any]:
        return self.lineage(node_id, **filters)

    def coverage(
        self,
        entity_id: str,
        *,
        max_depth: int = 8,
    ) -> FlowCoverageReport:
        selected_entity_id = _require_identifier(entity_id, "flow coverage entity")
        entity = self._repository.get_semantic_flow_entity(selected_entity_id)
        if entity is None:
            return FlowCoverageReport(
                entity_id=selected_entity_id,
                status=FlowCoverageStatus.UNKNOWN,
                effective_labels=FlowLabelVector(),
                entity_count=0,
                activity_count=0,
                edge_count=0,
                assertion_count=0,
                max_depth=0,
            )
        lineage = self.lineage(
            selected_entity_id,
            direction=FlowLineageDirection.UPSTREAM,
            limit=SEMANTIC_FLOW_LINEAGE_HARD_LIMIT,
            max_depth=max_depth,
        )
        effective = self.effective_labels(selected_entity_id)
        aggregate_labels = effective.labels
        assertion_count = effective.assertion_count
        seen_entities = {selected_entity_id}
        entities = 1
        activities = 0
        statuses = [effective.coverage]
        try:
            statuses.append(FlowCoverageStatus(lineage["coverage"]))
        except (KeyError, TypeError, ValueError):
            statuses.append(FlowCoverageStatus.UNKNOWN)
        deepest = 0
        for item in lineage["items"]:
            deepest = max(deepest, int(item.get("depth", 0)))
            node = item.get("node")
            if not isinstance(node, Mapping):
                statuses.append(FlowCoverageStatus.UNKNOWN)
                continue
            if item.get("node_type") == FlowNodeType.ENTITY.value:
                node_id = node.get("entity_id")
                if not isinstance(node_id, str) or node_id in seen_entities:
                    continue
                seen_entities.add(node_id)
                entities += 1
                try:
                    statuses.append(FlowCoverageStatus(node.get("coverage")))
                except (TypeError, ValueError):
                    statuses.append(FlowCoverageStatus.UNKNOWN)
                upstream = self.effective_labels(node_id)
                aggregate_labels = aggregate_labels.tighten(upstream.labels)
                assertion_count += upstream.assertion_count
                statuses.append(upstream.coverage)
            else:
                activities += 1
        if lineage["truncated"]:
            statuses.append(FlowCoverageStatus.PARTIAL)
        selected_status = max(statuses, key=_COVERAGE_PRECEDENCE.__getitem__)
        return FlowCoverageReport(
            entity_id=selected_entity_id,
            status=selected_status,
            effective_labels=aggregate_labels,
            entity_count=entities,
            activity_count=activities,
            edge_count=len(lineage["items"]),
            assertion_count=assertion_count,
            max_depth=deepest,
        )

    def _fallback_lineage(
        self,
        node_id: str,
        *,
        direction: FlowLineageDirection,
        after: SemanticV6Cursor | None,
        limit: int,
        max_depth: int,
    ) -> dict[str, Any]:
        """Use bounded breadth-first edge queries when the backend has no CTE."""
        root_type, root = self._resolve_lineage_root(node_id)
        traversal = self._scan_lineage(
            node_id=node_id,
            root_type=root_type,
            root=root,
            direction=direction,
            max_depth=max_depth,
        )
        selected_items, next_cursor, page_truncated = _paginate_lineage(
            traversal.discovered,
            after=after,
            limit=limit,
        )
        effective = self.effective_labels(node_id)
        selected_coverage = _lineage_coverage(effective.coverage, traversal)
        return {
            "schema_version": 1,
            "root_node_id": node_id,
            "direction": direction.value,
            "items": [item for _edge, item in selected_items],
            "effective_labels": effective.labels.to_dict(),
            "coverage": selected_coverage.value,
            "next_cursor": _encode_cursor(next_cursor),
            "truncated": traversal.traversal_truncated or page_truncated,
        }

    def _resolve_lineage_root(self, node_id: str) -> tuple[str, Any | None]:
        root_type = FlowNodeType.ENTITY.value
        root = self._get_node(root_type, node_id)
        if root is not None:
            return root_type, root
        root_type = FlowNodeType.ACTIVITY.value
        return root_type, self._get_node(root_type, node_id)

    def _scan_lineage(
        self,
        *,
        node_id: str,
        root_type: str,
        root: Any | None,
        direction: FlowLineageDirection,
        max_depth: int,
    ) -> _LineageTraversalResult:
        root_pid = getattr(root, "pid", None)
        root_tenant = getattr(root, "tenant_bucket_sha256", None)
        root_key = (root_type, node_id)
        frontier = [(root_type, node_id, 0, frozenset((root_key,)))]
        seen_nodes = {root_key}
        seen_edges: set[str] = set()
        discovered: list[tuple[SemanticFlowEdgeRecord, dict[str, Any]]] = []
        truncated = False
        conflict = False
        cycle = False
        missing = root is None
        while frontier:
            _node_type, selected_node, depth, ancestry = frontier.pop(0)
            if depth >= max_depth:
                truncated = True
                continue
            for edge in self._iter_lineage_edges(selected_node, pid=root_pid):
                adjacent_ref = _lineage_adjacent(edge, selected_node, direction)
                if edge.edge_id in seen_edges or adjacent_ref is None:
                    continue
                seen_edges.add(edge.edge_id)
                adjacent_type, adjacent_id = adjacent_ref
                adjacent, edge_missing, edge_conflict = self._scoped_lineage_node(
                    adjacent_type,
                    adjacent_id,
                    pid=root_pid,
                    tenant_bucket_sha256=root_tenant,
                )
                missing = missing or edge_missing
                conflict = conflict or edge_conflict
                discovered.append(
                    (
                        edge,
                        _lineage_item(
                            edge,
                            adjacent_type=adjacent_type,
                            adjacent=adjacent,
                            depth=depth + 1,
                        ),
                    )
                )
                if len(discovered) >= SEMANTIC_FLOW_LINEAGE_HARD_LIMIT:
                    return _LineageTraversalResult(
                        tuple(discovered), True, conflict, cycle, missing
                    )
                key = (adjacent_type, adjacent_id)
                if adjacent is not None and key in ancestry:
                    cycle = True
                elif adjacent is not None and key not in seen_nodes:
                    seen_nodes.add(key)
                    frontier.append((adjacent_type, adjacent_id, depth + 1, ancestry | {key}))
        return _LineageTraversalResult(
            tuple(discovered), truncated, conflict, cycle, missing
        )

    def _iter_lineage_edges(
        self,
        node_id: str,
        *,
        pid: str | None,
    ) -> Iterable[SemanticFlowEdgeRecord]:
        after: SemanticV6Cursor | None = None
        while True:
            page = self._repository.query_semantic_flow_edges(
                after=after,
                limit=SEMANTIC_V6_QUERY_HARD_LIMIT,
                pid=pid,
                relation=None,
                node_id=node_id,
            )
            yield from page.records
            if page.next_cursor is None:
                return
            after = page.next_cursor

    def _scoped_lineage_node(
        self,
        node_type: str,
        node_id: str,
        *,
        pid: str | None,
        tenant_bucket_sha256: str | None,
    ) -> tuple[Any | None, bool, bool]:
        selected = self._get_node(node_type, node_id)
        if selected is None:
            return None, True, False
        scope_changed = (
            getattr(selected, "pid", None) != pid
            or getattr(selected, "tenant_bucket_sha256", None)
            != tenant_bucket_sha256
        )
        # The SQL append boundary prevents this, but alternate repositories are
        # still interpreted fail-closed rather than exposing cross-scope data.
        return (None, False, True) if scope_changed else (selected, False, False)

    def _get_node(self, node_type: str, node_id: str) -> Any | None:
        if node_type == FlowNodeType.ENTITY.value:
            return self._repository.get_semantic_flow_entity(node_id)
        getter = getattr(self._repository, "get_semantic_flow_activity", None)
        return getter(node_id) if callable(getter) else None

    def _append_bundle(
        self,
        *,
        entities: tuple[SemanticFlowEntityRecord, ...] = (),
        activities: tuple[SemanticFlowActivityRecord, ...] = (),
        edges: tuple[SemanticFlowEdgeRecord, ...] = (),
        assertions: tuple[SemanticFlowLabelAssertionRecord, ...] = (),
    ) -> SemanticFlowBundle:
        append = self._repository.append_semantic_flow_bundle
        transaction = getattr(self._repository, "transaction", None)
        if not callable(transaction):
            return append(
                entities=entities,
                activities=activities,
                edges=edges,
                assertions=assertions,
            )
        # A capture may run under a business transaction.  The nested
        # savepoint ensures an injected SQL/storage failure cannot poison that
        # outer transaction on PostgreSQL or alter the business result.
        with transaction():
            return append(
                entities=entities,
                activities=activities,
                edges=edges,
                assertions=assertions,
            )

    def _require_node_scope(
        self,
        node: FlowNodeRef,
        *,
        pid: str,
        tenant_bucket_sha256: str,
    ) -> Any:
        selected = self._get_node(node.node_type.value, node.node_id)
        if selected is None:
            raise ValidationError("flow input references an unknown node")
        if getattr(selected, "pid", None) != pid:
            raise ValidationError("flow input cannot cross PID scope")
        if getattr(selected, "tenant_bucket_sha256", None) != tenant_bucket_sha256:
            raise ValidationError("flow input cannot cross tenant scope")
        return selected

    def _output_activity_for_entity(
        self,
        entity_id: str,
    ) -> SemanticFlowActivityRecord | None:
        matches: dict[str, SemanticFlowActivityRecord] = {}
        after: SemanticV6Cursor | None = None
        while True:
            page = self._repository.query_semantic_flow_edges(
                after=after,
                limit=SEMANTIC_V6_QUERY_HARD_LIMIT,
                pid=None,
                relation=FlowEdgeRelation.DIRECT.value,
                node_id=entity_id,
            )
            for edge in page.records:
                if (
                    edge.target_node_type != FlowNodeType.ENTITY.value
                    or edge.target_node_id != entity_id
                    or edge.source_node_type != FlowNodeType.ACTIVITY.value
                ):
                    continue
                activity = self._get_node(
                    FlowNodeType.ACTIVITY.value,
                    edge.source_node_id,
                )
                if isinstance(activity, SemanticFlowActivityRecord):
                    matches[activity.activity_id] = activity
            if page.next_cursor is None or len(matches) > 1:
                break
            after = page.next_cursor
        return next(iter(matches.values())) if len(matches) == 1 else None

    def _record_capture_failure(self, source: str) -> None:
        self._capture_failures += 1
        observer = self._capture_failure_observer
        if observer is not None:
            try:
                observer(source=source)
            except Exception:
                pass


class SemanticFlowProvenanceValidator:
    """Adapter injected into the exact-once authority validator.

    The resolver is Host-owned and must freshly resolve the current file or Git
    binding.  Models and Runtime modules never receive this adapter or its
    resolver.  A snapshot digest is both produced at issuance and rechecked at
    authorize/prepare/reserve/dispatch.
    """

    _LIVE_KEYS = frozenset(
        {
            "entity_id",
            "current_content_sha256",
            "current_version_sha256",
            "current_state_sha256",
            "source_labels_sha256",
            "source_refs_sha256",
        }
    )

    def __init__(
        self,
        flow: SemanticFlowService,
        *,
        live_binding_resolver: Callable[..., Mapping[str, Any]],
    ) -> None:
        if not isinstance(flow, SemanticFlowService):
            raise TypeError("semantic flow provenance validator requires its service")
        if not callable(live_binding_resolver):
            raise TypeError("semantic flow live binding resolver must be callable")
        self._flow = flow
        self._live_binding_resolver = live_binding_resolver

    def snapshot(
        self,
        *,
        action_id: str,
        tenant_bucket_sha256: str,
        entity_id: str,
        current_content_sha256: str,
        current_version_sha256: str,
        current_state_sha256: str,
    ) -> FlowApprovalEligibility:
        return self._flow.approval_eligibility(
            action_id=action_id,
            entity_id=entity_id,
            tenant_bucket_sha256=tenant_bucket_sha256,
            current_content_sha256=current_content_sha256,
            current_version_sha256=current_version_sha256,
            current_state_sha256=current_state_sha256,
        )

    def __call__(
        self,
        *,
        binding: SemanticApprovalBindingV2,
        phase: str,
        capability: Any,
        context: Mapping[str, Any],
        effect_id: str | None,
        control: Any,
        epoch: Any,
    ) -> None:
        # Local import avoids making the enforcement module a construction-time
        # dependency of the payload-free Flow service.
        from agent_libos.semantic.enforcement import SemanticSafetyTripRequired

        if not isinstance(binding, SemanticApprovalBindingV2):
            raise CapabilityDenied("semantic flow binding has an invalid type")
        try:
            live = self._live_binding_resolver(
                binding=binding,
                phase=phase,
                capability=capability,
                context=dict(context),
                effect_id=effect_id,
                control=control,
                epoch=epoch,
            )
        except SemanticSafetyTripRequired:
            raise
        except Exception as exc:
            raise CapabilityDenied(
                "semantic flow live binding could not be resolved"
            ) from exc
        if not isinstance(live, Mapping) or set(live) != self._LIVE_KEYS:
            raise CapabilityDenied(
                "semantic flow live binding resolver returned invalid facts"
            )
        if (
            live["source_labels_sha256"] != binding.source_labels_sha256
            or live["source_refs_sha256"] != binding.source_refs_sha256
        ):
            raise SemanticSafetyTripRequired(
                SemanticTripCode.BINDING_MISMATCH,
                "semantic source labels or references changed",
            )
        try:
            decision = self.snapshot(
                action_id=binding.authority_operation,
                tenant_bucket_sha256=binding.tenant_bucket_sha256,
                entity_id=live["entity_id"],
                current_content_sha256=live["current_content_sha256"],
                current_version_sha256=live["current_version_sha256"],
                current_state_sha256=live["current_state_sha256"],
            )
        except Exception as exc:
            raise CapabilityDenied("semantic flow evidence is invalid") from exc
        if not decision.eligible:
            _raise_eligibility_trip(decision)
            raise CapabilityDenied(
                "semantic flow evidence is incomplete, stale, or ineligible"
            )
        if decision.canonical_sha256() != binding.flow_snapshot_sha256:
            raise SemanticSafetyTripRequired(
                SemanticTripCode.BINDING_MISMATCH,
                "semantic FlowGraph snapshot binding changed",
            )


def _raise_eligibility_trip(decision: FlowApprovalEligibility) -> None:
    from agent_libos.semantic.enforcement import SemanticSafetyTripRequired

    reasons = frozenset(decision.reason_codes)
    if reasons & {
        FlowEligibilityReason.TENANT_MISMATCH,
        FlowEligibilityReason.MIXED_IDENTITY,
    }:
        raise SemanticSafetyTripRequired(
            SemanticTripCode.CROSS_TENANT,
            "semantic FlowGraph crossed its exact tenant boundary",
        )
    if FlowEligibilityReason.LABEL_TOO_SENSITIVE in reasons:
        raise SemanticSafetyTripRequired(
            SemanticTripCode.CRITICAL_HIGH_GRANT,
            "semantic FlowGraph became too sensitive for catalog v1",
        )
    if FlowEligibilityReason.ACTION_DRIFT in reasons:
        raise SemanticSafetyTripRequired(
            SemanticTripCode.UNAUTHORIZED_EFFECT,
            "semantic FlowGraph action binding changed",
        )


def _derived_activity_id(
    *,
    activity_kind: FlowActivityKind,
    pid: str,
    action_id: str,
    effect_id: str | None,
    state_sha256: str,
    provider_spec_sha256: str | None,
    tool_schema_sha256: str | None,
    model_artifact_sha256: str | None,
    content_sha256: str,
    version_sha256: str,
    provenance_sha256: str,
) -> str:
    return _stable_id(
        "flowact",
        _digest(
            {
                "kind": activity_kind.value,
                "pid": pid,
                "action_id": action_id,
                "effect_id": effect_id,
                "state_sha256": state_sha256,
                "provider_spec_sha256": provider_spec_sha256,
                "tool_schema_sha256": tool_schema_sha256,
                "model_artifact_sha256": model_artifact_sha256,
                "content_sha256": content_sha256,
                "version_sha256": version_sha256,
                "provenance_sha256": provenance_sha256,
            }
        ),
    )


def _relation_activity_id(
    *,
    activity_kind: FlowActivityKind,
    pid: str,
    action_id: str,
    effect_id: str | None,
    state_sha256: str,
    provider_spec_sha256: str | None,
    tool_schema_sha256: str | None,
    model_artifact_sha256: str | None,
    provenance_sha256: str,
    inputs: tuple[FlowInputEdge, ...],
    outputs: tuple[FlowOutputEdge, ...],
) -> str:
    input_refs = sorted(
        (
            item.source.node_type.value,
            item.source.node_id,
            item.relation.value,
        )
        for item in inputs
    )
    output_refs = sorted(
        (
            item.target.node_type.value,
            item.target.node_id,
            item.relation.value,
        )
        for item in outputs
    )
    return _stable_id(
        "flowact",
        _digest(
            {
                "kind": activity_kind.value,
                "pid": pid,
                "action_id": action_id,
                "effect_id": effect_id,
                "state_sha256": state_sha256,
                "provider_spec_sha256": provider_spec_sha256,
                "tool_schema_sha256": tool_schema_sha256,
                "model_artifact_sha256": model_artifact_sha256,
                "provenance_sha256": provenance_sha256,
                "inputs": input_refs,
                "outputs": output_refs,
            }
        ),
    )


def _derived_entity_id(
    *,
    entity_kind: FlowEntityKind,
    pid: str,
    version_sha256: str,
    content_sha256: str,
    activity_id: str,
) -> str:
    return _stable_id(
        "flowent",
        _digest(
            {
                "kind": entity_kind.value,
                "pid": pid,
                "version_sha256": version_sha256,
                "content_sha256": content_sha256,
                "activity_id": activity_id,
            }
        ),
    )


def _derived_output_edge(facts: _DerivedCaptureFacts) -> SemanticFlowEdgeRecord:
    return SemanticFlowEdgeRecord(
        edge_id=_edge_id(
            FlowEdgeRelation.DIRECT,
            facts.activity_id,
            facts.entity_id,
            facts.provenance_sha256,
        ),
        relation=FlowEdgeRelation.DIRECT.value,
        source_node_id=facts.activity_id,
        source_node_type=FlowNodeType.ACTIVITY.value,
        target_node_id=facts.entity_id,
        target_node_type=FlowNodeType.ENTITY.value,
        pid=facts.pid,
        provenance_sha256=facts.provenance_sha256,
        created_at=facts.created_at,
    )


def _relation_edge(
    *,
    relation: FlowEdgeRelation,
    source: FlowNodeRef,
    target: FlowNodeRef,
    pid: str,
    provenance_sha256: str,
    created_at: str,
) -> SemanticFlowEdgeRecord:
    return SemanticFlowEdgeRecord(
        edge_id=_edge_id(
            relation,
            source.node_id,
            target.node_id,
            provenance_sha256,
        ),
        relation=relation.value,
        source_node_id=source.node_id,
        source_node_type=source.node_type.value,
        target_node_id=target.node_id,
        target_node_type=target.node_type.value,
        pid=pid,
        provenance_sha256=provenance_sha256,
        created_at=created_at,
    )


def _lineage_adjacent(
    edge: SemanticFlowEdgeRecord,
    selected_node_id: str,
    direction: FlowLineageDirection,
) -> tuple[str, str] | None:
    if direction is FlowLineageDirection.UPSTREAM:
        if edge.target_node_id != selected_node_id:
            return None
        return edge.source_node_type, edge.source_node_id
    if edge.source_node_id != selected_node_id:
        return None
    return edge.target_node_type, edge.target_node_id


def _lineage_item(
    edge: SemanticFlowEdgeRecord,
    *,
    adjacent_type: str,
    adjacent: Any | None,
    depth: int,
) -> dict[str, Any]:
    return {
        "depth": depth,
        "edge": flow_record_to_dict(edge),
        "node_type": adjacent_type,
        "node": flow_record_to_dict(adjacent) if adjacent is not None else None,
    }


def _paginate_lineage(
    discovered: tuple[tuple[SemanticFlowEdgeRecord, dict[str, Any]], ...],
    *,
    after: SemanticV6Cursor | None,
    limit: int,
) -> tuple[
    list[tuple[SemanticFlowEdgeRecord, dict[str, Any]]],
    SemanticV6Cursor | None,
    bool,
]:
    ordered = sorted(
        discovered,
        key=lambda selected: (selected[0].created_at, selected[0].edge_id),
    )
    if after is not None:
        ordered = [
            selected
            for selected in ordered
            if (selected[0].created_at, selected[0].edge_id)
            > (after.created_at, after.record_id)
        ]
    selected_items = ordered[:limit]
    page_truncated = len(ordered) > limit
    next_cursor = None
    if page_truncated and selected_items:
        edge = selected_items[-1][0]
        next_cursor = SemanticV6Cursor(edge.created_at, edge.edge_id)
    return selected_items, next_cursor, page_truncated


def _lineage_coverage(
    baseline: FlowCoverageStatus,
    traversal: _LineageTraversalResult,
) -> FlowCoverageStatus:
    if traversal.scope_conflict or traversal.cycle_detected:
        return FlowCoverageStatus.CONFLICT
    if traversal.missing_node:
        return _worse_coverage(baseline, FlowCoverageStatus.UNKNOWN)
    if traversal.traversal_truncated:
        return _worse_coverage(baseline, FlowCoverageStatus.PARTIAL)
    return baseline


def _identity_present(labels: DataLabels) -> bool:
    return labels.tenant is not None or labels.principal is not None


def _root_goal_binding_sha256(
    *,
    goal_oid: str | None,
    goal_version: int | None,
) -> str:
    if goal_oid is not None and (
        not isinstance(goal_oid, str) or not goal_oid or len(goal_oid) > 4_096
    ):
        raise ValidationError("flow root goal oid is invalid")
    if goal_version is not None and (
        isinstance(goal_version, bool)
        or not isinstance(goal_version, int)
        or goal_version < 1
    ):
        raise ValidationError("flow root goal version is invalid")
    return _digest(
        {
            # Only this digest is retained.  The potentially sensitive object
            # identifier is never copied into a flow record or API response.
            "goal_oid_sha256": (
                hashlib.sha256(goal_oid.encode("utf-8")).hexdigest()
                if goal_oid is not None
                else None
            ),
            "goal_version": goal_version,
        }
    )


def _tenant_bucket_and_coverage(
    labels: DataLabels,
    tenant_bucket_sha256: str | None,
) -> tuple[str, FlowCoverageStatus]:
    if not isinstance(labels, DataLabels):
        raise TypeError("flow labels must be DataLabels")
    if tenant_bucket_sha256 is not None:
        return (
            _require_sha256(tenant_bucket_sha256, "flow tenant bucket"),
            (
                FlowCoverageStatus.CONFLICT
                if labels.is_mixed_identity
                else FlowCoverageStatus.COMPLETE
            ),
        )
    if _identity_present(labels):
        return FLOW_UNBUCKETED_IDENTITY_SHA256, FlowCoverageStatus.UNKNOWN
    return FLOW_NO_TENANT_BUCKET_SHA256, FlowCoverageStatus.COMPLETE


def _stable_id(prefix: str, digest: str) -> str:
    return f"{prefix}_{_require_sha256(digest, 'flow stable id digest')[:32]}"


def _edge_id(
    relation: FlowEdgeRelation,
    source_node_id: str,
    target_node_id: str,
    provenance_sha256: str,
) -> str:
    return _stable_id(
        "flowedge",
        _digest(
            {
                "relation": relation.value,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "provenance_sha256": provenance_sha256,
            }
        ),
    )


def _worse_coverage(
    left: FlowCoverageStatus,
    right: FlowCoverageStatus,
) -> FlowCoverageStatus:
    return max((left, right), key=_COVERAGE_PRECEDENCE.__getitem__)


def _encode_cursor(cursor: SemanticV6Cursor | None) -> str | None:
    if cursor is None:
        return None
    payload = _canonical_bytes(
        {"created_at": cursor.created_at, "record_id": cursor.record_id}
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(
    value: str | SemanticV6Cursor | None,
) -> SemanticV6Cursor | None:
    if value is None or isinstance(value, SemanticV6Cursor):
        return value
    if not isinstance(value, str) or not value or len(value) > FLOW_CURSOR_MAX_BYTES:
        raise ValidationError("semantic flow cursor is invalid")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        if len(raw) > FLOW_CURSOR_MAX_BYTES:
            raise ValueError
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError("semantic flow cursor is invalid") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"created_at", "record_id"}:
        raise ValidationError("semantic flow cursor is invalid")
    return SemanticV6Cursor(
        created_at=decoded["created_at"],
        record_id=decoded["record_id"],
    )


def _page_dict(page: SemanticFlowPage) -> dict[str, Any]:
    if not isinstance(page, SemanticFlowPage):
        raise TypeError("semantic flow repository returned an invalid page")
    return {
        "schema_version": 1,
        "items": [flow_record_to_dict(record) for record in page.records],
        "next_cursor": _encode_cursor(page.next_cursor),
    }


def _lineage_dict(
    root_node_id: str,
    direction: FlowLineageDirection,
    page: Any,
    effective: FlowEffectiveLabels,
) -> dict[str, Any]:
    if isinstance(page, Mapping):
        selected = dict(page)
        items = selected.get("items", ())
        next_cursor = selected.get("next_cursor")
        truncated = selected.get("truncated", False)
    else:
        items = getattr(page, "items", getattr(page, "records", ()))
        next_cursor = getattr(page, "next_cursor", None)
        truncated = getattr(page, "truncated", next_cursor is not None)
    normalized_items: list[dict[str, Any]] = []
    for item in tuple(items):
        if isinstance(item, Mapping):
            selected_item = dict(item)
        else:
            selected_item = {
                "depth": getattr(item, "depth"),
                "edge": flow_record_to_dict(getattr(item, "edge")),
                "node_type": getattr(item, "node_type"),
                "node": (
                    flow_record_to_dict(getattr(item, "node"))
                    if getattr(item, "node", None) is not None
                    else None
                ),
            }
        edge = selected_item.get("edge")
        node = selected_item.get("node")
        if edge is not None and not isinstance(edge, Mapping):
            selected_item["edge"] = flow_record_to_dict(edge)
        if node is not None and not isinstance(node, Mapping):
            selected_item["node"] = flow_record_to_dict(node)
        normalized_items.append(selected_item)
    return {
        "schema_version": 1,
        "root_node_id": root_node_id,
        "direction": direction.value,
        "items": normalized_items,
        "effective_labels": effective.labels.to_dict(),
        "coverage": effective.coverage.value,
        "next_cursor": (
            next_cursor
            if isinstance(next_cursor, str) or next_cursor is None
            else _encode_cursor(next_cursor)
        ),
        "truncated": bool(truncated),
    }


def _normalize_status(
    value: Mapping[str, Any],
    capture_failures: int,
) -> dict[str, Any]:
    counts_source = value.get("counts")
    coverage_source = value.get("coverage")
    counts = dict(counts_source) if isinstance(counts_source, Mapping) else {}
    coverage = (
        dict(coverage_source) if isinstance(coverage_source, Mapping) else {}
    )
    legacy_source = value.get("legacy_history")
    legacy = (
        SemanticLegacyFlowHistoryV1.from_dict(legacy_source)
        if isinstance(legacy_source, Mapping)
        else SemanticLegacyFlowHistoryV1()
    )
    return {
        "schema_version": 1,
        "available": True,
        "counts": {
            "entities": int(counts.get("entities", 0)),
            "activities": int(counts.get("activities", 0)),
            "edges": int(counts.get("edges", 0)),
            "label_assertions": int(counts.get("label_assertions", 0)),
        },
        "coverage": {
            status.value: int(coverage.get(status.value, 0))
            for status in FlowCoverageStatus
        },
        "capture_failures": max(
            capture_failures,
            int(value.get("capture_failures", 0)),
        ),
        "legacy_history": legacy.to_dict(),
    }


# Public semantic names intentionally alias the strict storage records.  This
# keeps persistence validation at the Host boundary and avoids parallel model
# definitions with subtly different fields.
FlowEntityRecord = SemanticFlowEntityRecord
FlowActivityRecord = SemanticFlowActivityRecord
FlowEdgeRecord = SemanticFlowEdgeRecord
FlowLabelAssertionRecord = SemanticFlowLabelAssertionRecord
FlowCursor = SemanticV6Cursor
FlowPage = SemanticFlowPage


__all__ = [
    "FLOW_JSON_LOCATOR_MAX_DEPTH",
    "FLOW_LABEL_ASSERTION_HARD_LIMIT",
    "FLOW_LINEAGE_DEPTH_HARD_LIMIT",
    "FLOW_NO_TENANT_BUCKET_SHA256",
    "FLOW_SCHEMA_VERSION",
    "FLOW_UNBUCKETED_IDENTITY_SHA256",
    "FlowActivityKind",
    "FlowActivityRecord",
    "FlowApprovalEligibility",
    "FlowCoverageReport",
    "FlowCoverageStatus",
    "FlowCursor",
    "FlowDataLocator",
    "FlowEdgeRecord",
    "FlowEdgeRelation",
    "FlowEffectiveLabels",
    "FlowEligibilityReason",
    "FlowEntityKind",
    "FlowEntityRecord",
    "FlowInputEdge",
    "FlowLabelAssertionRecord",
    "FlowLabelSource",
    "FlowLabelVector",
    "FlowLineageDirection",
    "FlowLocatorKind",
    "FlowMemoryGateDecision",
    "FlowMemoryGateReason",
    "FlowNodeRef",
    "FlowNodeType",
    "FlowOutputEdge",
    "FlowPage",
    "SemanticFlowRepositoryPort",
    "SemanticFlowProvenanceValidator",
    "SemanticFlowService",
    "compute_effective_labels",
    "flow_record_to_dict",
    "provider_result_entity_id",
    "root_goal_entity_id",
]
