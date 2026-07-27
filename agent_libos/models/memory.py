from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_libos.models.base import CapabilityID, MemoryViewID, NamespaceID, OID, PID, SnapshotID, StrEnum
from agent_libos.models.data_flow import DataIntegrity, DataSensitivity, DataTrustLevel


class _UnsetPayload:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetPayload()


class ObjectType(StrEnum):
    TASK = "task"
    GOAL = "goal"
    PLAN = "plan"
    STEP = "step"
    CONSTRAINT = "constraint"
    MESSAGE = "message"
    HUMAN_DECISION = "human_decision"
    HUMAN_REQUEST = "human_request"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    ERROR_TRACE = "error_trace"
    CODE_PATCH = "code_patch"
    TEST_RESULT = "test_result"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    SUMMARY = "summary"
    SKILL = "skill"
    TOOL_SPEC = "tool_spec"
    TOOL_CANDIDATE = "tool_candidate"
    TOOL_ARTIFACT = "tool_artifact"
    CHECKPOINT = "checkpoint"
    PROCESS_STATE = "process_state"
    EXTERNAL_REF = "external_ref"
    ARTIFACT = "artifact"


class ObjectRight(StrEnum):
    READ = "read"
    WRITE = "write"
    LINK = "link"
    DIFF = "diff"
    MATERIALIZE = "materialize"
    DELETE = "delete"
    GRANT = "grant"


class RelationType(StrEnum):
    HAS_PLAN = "has_plan"
    HAS_STEP = "has_step"
    CONSTRAINED_BY = "constrained_by"
    SUPPORTED_BY = "supported_by"
    PRODUCED = "produced"
    EVALUATED_BY = "evaluated_by"
    DERIVED_FROM = "derived_from"
    SUMMARIZES = "summarizes"
    REFERENCES = "references"
    APPROVED_BY = "approved_by"
    REJECTED_BY = "rejected_by"
    SUPERSEDES = "supersedes"
    BLOCKED_BY = "blocked_by"
    ASSIGNED_TO = "assigned_to"


class ViewMode(StrEnum):
    READ_ONLY = "read_only"
    COPY_ON_WRITE = "copy_on_write"
    MUTABLE = "mutable"
    EPHEMERAL = "ephemeral"


class ObjectOwnerKind(StrEnum):
    PROCESS = "process"
    PROCESS_RESULT = "process_result"
    OBJECT_TASK = "object_task"
    RUNTIME = "runtime"


class ObjectLifecycleState(StrEnum):
    LIVE = "live"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class PersistedObjectState:
    """Payload-free durable Object state used by security revalidation."""

    oid: OID
    lifecycle_state: ObjectLifecycleState
    version: int
    payload_present: bool
    recovered_after_reopen: bool

    def __post_init__(self) -> None:
        if not isinstance(self.oid, str) or not self.oid:
            raise ValueError("persisted Object state oid must be non-empty")
        if not isinstance(self.lifecycle_state, ObjectLifecycleState):
            raise ValueError("persisted Object lifecycle state is invalid")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version <= 0
        ):
            raise ValueError("persisted Object version must be positive")
        if type(self.payload_present) is not bool:
            raise ValueError("persisted Object payload_present must be boolean")
        if type(self.recovered_after_reopen) is not bool:
            raise ValueError(
                "persisted Object recovered_after_reopen must be boolean"
            )
        if self.payload_present and self.recovered_after_reopen:
            raise ValueError(
                "a present Object payload cannot be marked recovered after reopen"
            )
        if (
            self.recovered_after_reopen
            and self.lifecycle_state is not ObjectLifecycleState.RELEASED
        ):
            raise ValueError(
                "only a released Object can carry the reopen recovery marker"
            )


@dataclass(frozen=True, slots=True)
class ObjectPayloadRecoverySummary:
    """Bounded diagnostics for runtime-only payload rows released on reopen."""

    total_count: int = 0
    sample_oids: tuple[OID, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("object payload recovery total_count must be non-negative")
        if not isinstance(self.sample_oids, tuple):
            raise ValueError("object payload recovery sample must be a tuple")
        if len(self.sample_oids) > self.total_count:
            raise ValueError("object payload recovery sample exceeds total")
        if any(not isinstance(oid, str) or not oid for oid in self.sample_oids):
            raise ValueError("object payload recovery sample OIDs must not be empty")

    @property
    def truncated(self) -> bool:
        return self.total_count > len(self.sample_oids)

    def __len__(self) -> int:
        return self.total_count


@dataclass
class ObjectMetadata:
    title: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    mime_type: str | None = None
    token_estimate: int | None = None
    embedding_refs: list[str] = field(default_factory=list)
    indexes: list[str] = field(default_factory=list)
    sensitivity: str = DataSensitivity.NORMAL.value
    retention_policy: str = "default"
    trust_level: str = "unknown"
    integrity: str = "unknown"
    origin: str | None = "local"
    tenant: str | None = None
    principal: str | None = None
    declassification_authority: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the mutable metadata envelope before it crosses a boundary."""

        for field_name in ("title", "summary", "mime_type"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"object metadata {field_name} must be a string or null")
        for field_name in ("tags", "embedding_refs", "indexes"):
            values = getattr(self, field_name)
            if not isinstance(values, list):
                raise ValueError(f"object metadata {field_name} must be a list")
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f"object metadata {field_name} items must be strings")
        if (
            self.token_estimate is not None
            and (
                isinstance(self.token_estimate, bool)
                or not isinstance(self.token_estimate, int)
                or self.token_estimate < 0
            )
        ):
            raise ValueError("object metadata token_estimate must be a non-negative integer or null")
        if not isinstance(self.retention_policy, str) or not self.retention_policy.strip():
            raise ValueError("object metadata retention_policy must be a non-empty string")
        _validate_object_label(
            "sensitivity",
            self.sensitivity,
            {item.value for item in DataSensitivity},
        )
        _validate_object_label(
            "trust_level",
            self.trust_level,
            {item.value for item in DataTrustLevel},
        )
        _validate_object_label(
            "integrity",
            self.integrity,
            {item.value for item in DataIntegrity},
        )
        for field_name in ("origin", "tenant", "principal", "declassification_authority"):
            _validate_optional_identity(field_name, getattr(self, field_name))

    @classmethod
    def from_persisted(cls, value: Mapping[str, Any]) -> ObjectMetadata:
        """Decode metadata written before data-label enums were enforced.

        New writes remain strict through ``__post_init__``. Historical stores,
        however, accepted arbitrary strings. Unknown confidentiality labels are
        therefore raised to the most restrictive value, while unknown
        integrity/trust and identities are reduced conservatively.
        """

        if not isinstance(value, Mapping):
            raise ValueError("persisted object metadata must be an object")
        selected = dict(value)
        _normalize_persisted_ordered_label(
            selected,
            "sensitivity",
            {item.value for item in DataSensitivity},
            fallback=DataSensitivity.SECRET.value,
        )
        _normalize_persisted_ordered_label(
            selected,
            "trust_level",
            {item.value for item in DataTrustLevel},
            fallback=DataTrustLevel.UNTRUSTED.value,
        )
        _normalize_persisted_ordered_label(
            selected,
            "integrity",
            {item.value for item in DataIntegrity},
            fallback=DataIntegrity.UNTRUSTED.value,
        )
        for field_name, fallback in (
            ("origin", "derived"),
            ("tenant", "mixed"),
            ("principal", "mixed"),
            ("declassification_authority", None),
        ):
            if field_name not in selected:
                continue
            try:
                _validate_optional_identity(field_name, selected[field_name])
            except ValueError:
                selected[field_name] = fallback
        return cls(**selected)


def _validate_object_label(name: str, value: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"invalid object data label {name}: {value!r}")


def _validate_optional_identity(name: str, value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid object data label {name}: {value!r}")
    if value == "mixed" and name not in {"tenant", "principal"}:
        raise ValueError(f"invalid object data label {name}: {value!r}")
    if len(value) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"invalid object data label {name}: {value!r}")


def _normalize_persisted_ordered_label(
    selected: dict[str, Any],
    name: str,
    allowed: set[str],
    *,
    fallback: str,
) -> None:
    if name in selected and (
        not isinstance(selected[name], str) or selected[name] not in allowed
    ):
        selected[name] = fallback


@dataclass
class Provenance:
    source_refs: list[str] = field(default_factory=list)
    created_from_action: str | None = None
    parent_oids: list[OID] = field(default_factory=list)
    source_operation_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectNamespace:
    namespace: NamespaceID
    parent_namespace: NamespaceID | None
    metadata: dict[str, Any]
    created_by: PID | str
    created_at: str
    updated_at: str


@dataclass(frozen=True, order=True, slots=True)
class ObjectNamespaceCursor:
    """Stable keyset cursor for one namespace's direct children."""

    namespace: NamespaceID

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace
            or "\x00" in self.namespace
        ):
            raise ValueError("object namespace cursor must not be empty")


@dataclass(frozen=True, slots=True)
class ObjectNamespacePage:
    """One hard-bounded page of direct child namespaces."""

    records: tuple[ObjectNamespace, ...]
    next_cursor: ObjectNamespaceCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty object namespace page cannot have a cursor")

    @property
    def exhausted(self) -> bool:
        return self.next_cursor is None


@dataclass(frozen=True)
class AgentObject:
    oid: OID
    namespace: NamespaceID
    name: str
    type: ObjectType
    schema_version: str
    payload: Any
    metadata: ObjectMetadata
    provenance: Provenance
    version: int
    immutable: bool
    created_by: PID | str
    created_at: str
    updated_at: str
    owner_kind: ObjectOwnerKind = ObjectOwnerKind.PROCESS
    owner_id: PID | str | None = None
    lifecycle_state: ObjectLifecycleState = ObjectLifecycleState.LIVE
    deleted_at: str | None = None


@dataclass(frozen=True, order=True, slots=True)
class ObjectRefCursor:
    """Stable keyset cursor for a namespace's live Object directory."""

    updated_at: str
    created_at: str
    oid: OID

    def __post_init__(self) -> None:
        for field_name in ("updated_at", "created_at", "oid"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(
                    f"object reference cursor {field_name} must not be empty"
                )


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Payload-free live Object projection for bounded authorization scans."""

    oid: OID
    namespace: NamespaceID
    name: str
    type: ObjectType
    schema_version: str
    metadata: ObjectMetadata
    version: int
    immutable: bool
    created_by: PID | str
    owner_kind: ObjectOwnerKind
    owner_id: PID | str | None
    created_at: str
    updated_at: str

    @property
    def cursor(self) -> ObjectRefCursor:
        return ObjectRefCursor(
            updated_at=self.updated_at,
            created_at=self.created_at,
            oid=self.oid,
        )

    @property
    def search_text(self) -> str:
        """Metadata-only search projection; Object payloads are never rendered."""

        return " ".join(
            (
                self.namespace,
                self.name,
                self.metadata.title or "",
                self.metadata.summary or "",
                " ".join(self.metadata.tags),
            )
        )


@dataclass(frozen=True, slots=True)
class ObjectRefPage:
    """One hard-bounded page of payload-free live Object projections."""

    records: tuple[ObjectRef, ...]
    next_cursor: ObjectRefCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty object reference page cannot have a cursor")

    @property
    def exhausted(self) -> bool:
        return self.next_cursor is None


@dataclass(frozen=True)
class ObjectHandle:
    oid: OID
    rights: set[str]
    capability_id: CapabilityID
    expires_at: str | None = None


@dataclass
class ObjectFilter:
    type: ObjectType | None = None
    tags: list[str] = field(default_factory=list)
    text: str | None = None


@dataclass
class ObjectQuery:
    type: ObjectType | str | None = None
    tags: list[str] = field(default_factory=list)
    text: str | None = None
    limit: int | None = None
    name: str | None = None
    namespace: NamespaceID | None = None


@dataclass
class ObjectPatch:
    name: str | None = None
    namespace: NamespaceID | None = None
    payload: Any = UNSET
    metadata: ObjectMetadata | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class ObjectLink:
    link_id: str
    src: OID
    relation: RelationType
    dst: OID
    metadata: dict[str, Any]
    created_by: PID | str
    created_at: str


@dataclass
class MemoryView:
    view_id: MemoryViewID
    owner_pid: PID
    roots: list[ObjectHandle]
    filters: list[ObjectFilter]
    rights_policy: str
    created_from: MemoryViewID | SnapshotID | None
    mode: ViewMode


@dataclass
class MemoryViewSpec:
    roots: list[ObjectHandle] | None = None
    mode: ViewMode = ViewMode.READ_ONLY
    include_parent_roots: bool = True
    rights: set[str] | None = None


@dataclass
class MergePolicy:
    include_child_created: bool = True
    include_updated: bool = True
    grant_rights: set[str] = field(default_factory=lambda: {"read", "materialize", "link"})


@dataclass
class MergeResult:
    merged_oids: list[OID]
    skipped_oids: list[OID]
    merged_handles: list[ObjectHandle] = field(default_factory=list)
    adopted_oids: list[OID] = field(default_factory=list)
    released_oids: list[OID] = field(default_factory=list)
    retained_child_owned_oids: list[OID] = field(default_factory=list)
    pinned_child_owned_oids: list[OID] = field(default_factory=list)


@dataclass
class MaterializedContext:
    text: str
    object_refs: list[OID]
    token_count: int
    omitted_objects: list[OID]
    policy_used: str
    materialization_id: str | None = None
    view_id: str | None = None
    budget_tokens: int | None = None
    object_manifest: list[dict[str, Any]] = field(default_factory=list)
