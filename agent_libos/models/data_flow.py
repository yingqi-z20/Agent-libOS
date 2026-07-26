from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from agent_libos.models.base import OID, PID, StrEnum

if TYPE_CHECKING:
    from agent_libos.models.memory import ObjectMetadata


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVITY_ORDER = ("public", "normal", "confidential", "restricted", "secret")
_TRUST_ORDER = ("untrusted", "unknown", "user_asserted", "verified", "trusted")
_INTEGRITY_ORDER = ("untrusted", "unknown", "checked", "verified")
_PROVIDER_BACKED_PREFIXES = ("llm:", "jsonrpc:", "mcp:", "mcp_stdio:", "shell:", "pty:")


class DataSensitivity(StrEnum):
    PUBLIC = "public"
    NORMAL = "normal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class DataTrustLevel(StrEnum):
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"
    USER_ASSERTED = "user_asserted"
    VERIFIED = "verified"
    TRUSTED = "trusted"


class DataIntegrity(StrEnum):
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"
    CHECKED = "checked"
    VERIFIED = "verified"


class SinkTrustLevel(StrEnum):
    UNTRUSTED = "untrusted"
    CONDITIONAL = "conditional"
    TRUSTED = "trusted"


class DataFlowDirection(StrEnum):
    NONE = "none"
    INGRESS = "ingress"
    EGRESS = "egress"
    BIDIRECTIONAL = "bidirectional"


class DataFlowOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    RELEASE_REQUIRED = "release_required"


def sensitivity_rank(value: DataSensitivity | str) -> int:
    selected = _coerce_enum(DataSensitivity, value, "sensitivity")
    return _SENSITIVITY_ORDER.index(selected.value)


def integrity_rank(value: DataIntegrity | str) -> int:
    selected = _coerce_enum(DataIntegrity, value, "integrity")
    return _INTEGRITY_ORDER.index(selected.value)


@dataclass(frozen=True)
class DataLabels:
    """Trusted, payload-free labels used by the data-flow gate.

    ``tenant`` and ``principal`` are deliberately singular. Aggregating
    different non-empty identities produces the reserved value ``mixed``,
    which is never a valid sink clearance.
    """

    sensitivity: DataSensitivity = DataSensitivity.NORMAL
    trust_level: DataTrustLevel = DataTrustLevel.UNKNOWN
    integrity: DataIntegrity = DataIntegrity.UNKNOWN
    origin: str | None = "local"
    tenant: str | None = None
    principal: str | None = None
    declassification_authority: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitivity", _coerce_enum(DataSensitivity, self.sensitivity, "sensitivity"))
        object.__setattr__(self, "trust_level", _coerce_enum(DataTrustLevel, self.trust_level, "trust_level"))
        object.__setattr__(self, "integrity", _coerce_enum(DataIntegrity, self.integrity, "integrity"))
        for name in ("origin", "tenant", "principal", "declassification_authority"):
            _validate_optional_text(name, getattr(self, name), allow_mixed=name in {"tenant", "principal"})

    @property
    def is_mixed_identity(self) -> bool:
        return self.tenant == "mixed" or self.principal == "mixed"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "sensitivity": self.sensitivity.value,
            "trust_level": self.trust_level.value,
            "integrity": self.integrity.value,
            "origin": self.origin,
            "tenant": self.tenant,
            "principal": self.principal,
            "declassification_authority": self.declassification_authority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataLabels:
        if not isinstance(value, Mapping):
            raise ValueError("data labels must be an object")
        unknown = set(value) - {
            "sensitivity",
            "trust_level",
            "integrity",
            "origin",
            "tenant",
            "principal",
            "declassification_authority",
        }
        if unknown:
            raise ValueError(f"data labels contain unknown fields: {sorted(unknown)}")
        return cls(
            sensitivity=value.get("sensitivity", DataSensitivity.NORMAL),
            trust_level=value.get("trust_level", DataTrustLevel.UNKNOWN),
            integrity=value.get("integrity", DataIntegrity.UNKNOWN),
            origin=value.get("origin", "local"),
            tenant=value.get("tenant"),
            principal=value.get("principal"),
            declassification_authority=value.get("declassification_authority"),
        )

    @classmethod
    def from_object_metadata(cls, metadata: ObjectMetadata | Any) -> DataLabels:
        return cls(
            sensitivity=getattr(metadata, "sensitivity", DataSensitivity.NORMAL),
            trust_level=getattr(metadata, "trust_level", DataTrustLevel.UNKNOWN),
            integrity=getattr(metadata, "integrity", DataIntegrity.UNKNOWN),
            origin=getattr(metadata, "origin", "local"),
            tenant=getattr(metadata, "tenant", None),
            principal=getattr(metadata, "principal", None),
            declassification_authority=getattr(metadata, "declassification_authority", None),
        )

    @classmethod
    def aggregate(cls, labels: Iterable[DataLabels]) -> DataLabels:
        selected = tuple(labels)
        if not selected:
            return cls()
        return cls(
            sensitivity=max(selected, key=lambda item: sensitivity_rank(item.sensitivity)).sensitivity,
            trust_level=min(
                selected,
                key=lambda item: _TRUST_ORDER.index(item.trust_level.value),
            ).trust_level,
            integrity=min(
                selected,
                key=lambda item: _INTEGRITY_ORDER.index(item.integrity.value),
            ).integrity,
            origin=_merge_identity((item.origin for item in selected), mixed="derived"),
            tenant=_merge_identity((item.tenant for item in selected), mixed="mixed"),
            principal=_merge_identity((item.principal for item in selected), mixed="mixed"),
            declassification_authority=_merge_declassification_authority(
                item.declassification_authority for item in selected
            ),
        )

    def labels_hash(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class DataSourceRef:
    oid: OID
    version: int
    content_sha256: str

    def __post_init__(self) -> None:
        _require_text("source oid", self.oid)
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("source version must be a positive integer")
        _require_sha256("source content_sha256", self.content_sha256)

    def to_dict(self) -> dict[str, str | int]:
        return {"oid": self.oid, "version": self.version, "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataSourceRef:
        if not isinstance(value, Mapping):
            raise ValueError("data source ref must be an object")
        if set(value) != {"oid", "version", "content_sha256"}:
            raise ValueError("data source ref must contain only oid, version, and content_sha256")
        return cls(
            oid=value["oid"],
            version=value["version"],
            content_sha256=value["content_sha256"],
        )


@dataclass(frozen=True)
class DataFlowContext:
    labels: DataLabels = field(default_factory=DataLabels)
    source_refs: tuple[DataSourceRef, ...] = ()
    materialization_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.labels, DataLabels):
            object.__setattr__(self, "labels", DataLabels.from_dict(self.labels))
        object.__setattr__(self, "source_refs", _normalize_source_refs(self.source_refs))
        _validate_optional_text("materialization_id", self.materialization_id)

    @classmethod
    def aggregate(cls, contexts: Iterable[DataFlowContext]) -> DataFlowContext:
        selected = tuple(contexts)
        refs: dict[tuple[str, int, str], DataSourceRef] = {}
        for context in selected:
            for ref in context.source_refs:
                refs[(ref.oid, ref.version, ref.content_sha256)] = ref
        return cls(
            labels=DataLabels.aggregate(item.labels for item in selected),
            source_refs=tuple(refs[key] for key in sorted(refs)),
        )

    def source_refs_hash(self) -> str:
        return _canonical_sha256([item.to_dict() for item in self.source_refs])

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "materialization_id": self.materialization_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataFlowContext:
        if not isinstance(value, Mapping):
            raise ValueError("data-flow context must be an object")
        unknown = set(value) - {"labels", "source_refs", "materialization_id"}
        if unknown:
            raise ValueError(f"data-flow context contains unknown fields: {sorted(unknown)}")
        return cls(
            labels=DataLabels.from_dict(value.get("labels") or {}),
            source_refs=tuple(
                DataSourceRef.from_dict(item)
                for item in tuple(value.get("source_refs") or ())
            ),
            materialization_id=value.get("materialization_id"),
        )


@dataclass(frozen=True)
class DataSink:
    identity: str
    identity_sha256: str | None = None
    trust_identity: str | None = None
    trust_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text("sink identity", self.identity)
        if "*" in self.identity:
            raise ValueError("concrete sink identity must not contain wildcards")
        if self.identity != self.identity.strip() or ":" not in self.identity:
            raise ValueError("sink identity must be a canonical namespaced value")
        namespace, target = self.identity.split(":", 1)
        if not namespace or not target:
            raise ValueError("sink identity must include a non-empty namespace and target")
        _optional_sha256("sink identity_sha256", self.identity_sha256)
        if self.trust_identity is not None:
            _require_text("sink trust_identity", self.trust_identity)
            if "*" in self.trust_identity or ":" not in self.trust_identity:
                raise ValueError("sink trust_identity must be a concrete canonical namespaced value")
        _optional_sha256("sink trust_identity_sha256", self.trust_identity_sha256)

    @property
    def registry_identity(self) -> str:
        return self.trust_identity or self.identity

    @property
    def registry_identity_sha256(self) -> str | None:
        return self.trust_identity_sha256 or self.identity_sha256


@dataclass(frozen=True)
class SinkTrustRule:
    pattern: str
    trust_level: SinkTrustLevel = SinkTrustLevel.UNTRUSTED
    max_sensitivity: DataSensitivity = DataSensitivity.NORMAL
    tenants: tuple[str, ...] = ()
    principals: tuple[str, ...] = ()
    identity_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_sink_pattern(self.pattern)
        object.__setattr__(self, "trust_level", _coerce_enum(SinkTrustLevel, self.trust_level, "trust_level"))
        object.__setattr__(
            self,
            "max_sensitivity",
            _coerce_enum(DataSensitivity, self.max_sensitivity, "max_sensitivity"),
        )
        object.__setattr__(self, "tenants", _normalize_clearance_identities("tenants", self.tenants))
        object.__setattr__(self, "principals", _normalize_clearance_identities("principals", self.principals))
        _optional_sha256("identity_sha256", self.identity_sha256)
        if self.trust_level is SinkTrustLevel.UNTRUSTED and sensitivity_rank(self.max_sensitivity) > sensitivity_rank(
            DataSensitivity.NORMAL
        ):
            raise ValueError("untrusted sink max_sensitivity must not exceed normal")
        if (
            self.pattern.startswith(_PROVIDER_BACKED_PREFIXES)
            and sensitivity_rank(self.max_sensitivity) > sensitivity_rank(DataSensitivity.NORMAL)
            and self.identity_sha256 is None
        ):
            raise ValueError("provider-backed sink clearance above normal requires identity_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "trust_level": self.trust_level.value,
            "max_sensitivity": self.max_sensitivity.value,
            "tenants": list(self.tenants),
            "principals": list(self.principals),
            "identity_sha256": self.identity_sha256,
        }

    def spec_hash(self) -> str:
        return _canonical_sha256({"schema_version": 1, **self.to_dict()})


@dataclass(frozen=True)
class SinkTrustSpec:
    trust_id: str
    pattern: str
    trust_level: SinkTrustLevel
    max_sensitivity: DataSensitivity
    generation: int
    created_by: str
    created_at: str
    tenants: tuple[str, ...] = ()
    principals: tuple[str, ...] = ()
    identity_sha256: str | None = None
    spec_hash: str = ""
    active: bool = True
    deactivated_at: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text("trust_id", self.trust_id)
        _require_text("created_by", self.created_by)
        _require_text("created_at", self.created_at)
        if self.schema_version != 1:
            raise ValueError(f"unsupported sink trust schema_version: {self.schema_version}")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise ValueError("sink trust generation must be a positive integer")
        if not isinstance(self.active, bool):
            raise ValueError("sink trust active must be boolean")
        _validate_optional_text("deactivated_at", self.deactivated_at)
        if self.active and self.deactivated_at is not None:
            raise ValueError("active sink trust record cannot have deactivated_at")
        if not self.active and self.deactivated_at is None:
            raise ValueError("inactive sink trust record requires deactivated_at")
        rule = SinkTrustRule(
            pattern=self.pattern,
            trust_level=self.trust_level,
            max_sensitivity=self.max_sensitivity,
            tenants=self.tenants,
            principals=self.principals,
            identity_sha256=self.identity_sha256,
        )
        object.__setattr__(self, "trust_level", rule.trust_level)
        object.__setattr__(self, "max_sensitivity", rule.max_sensitivity)
        object.__setattr__(self, "tenants", rule.tenants)
        object.__setattr__(self, "principals", rule.principals)
        actual_hash = rule.spec_hash()
        if self.spec_hash and self.spec_hash != actual_hash:
            raise ValueError("sink trust spec_hash does not match the canonical spec")
        object.__setattr__(self, "spec_hash", actual_hash)

    @property
    def rule(self) -> SinkTrustRule:
        return SinkTrustRule(
            pattern=self.pattern,
            trust_level=self.trust_level,
            max_sensitivity=self.max_sensitivity,
            tenants=self.tenants,
            principals=self.principals,
            identity_sha256=self.identity_sha256,
        )


@dataclass(frozen=True)
class DataFlowDecision:
    decision_id: str
    pid: PID
    sink: str
    direction: DataFlowDirection
    outcome: DataFlowOutcome
    reason: str
    labels: DataLabels
    source_refs: tuple[DataSourceRef, ...]
    payload_hash: str
    registry_generation: int
    created_at: str
    trust_id: str | None = None
    trust_hash: str | None = None
    release_capability_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "pid", "sink", "reason", "created_at"):
            _require_text(name, getattr(self, name))
        if len(self.reason) > 1_024 or "\x00" in self.reason:
            raise ValueError("data-flow decision reason must be at most 1024 characters without NUL")
        DataSink(self.sink)
        object.__setattr__(self, "direction", _coerce_enum(DataFlowDirection, self.direction, "direction"))
        object.__setattr__(self, "outcome", _coerce_enum(DataFlowOutcome, self.outcome, "outcome"))
        if not isinstance(self.labels, DataLabels):
            object.__setattr__(self, "labels", DataLabels.from_dict(self.labels))
        object.__setattr__(self, "source_refs", _normalize_source_refs(self.source_refs))
        _require_sha256("payload_hash", self.payload_hash)
        if (
            isinstance(self.registry_generation, bool)
            or not isinstance(self.registry_generation, int)
            or self.registry_generation < 0
        ):
            raise ValueError("registry_generation must be a non-negative integer")
        _validate_optional_text("trust_id", self.trust_id)
        _optional_sha256("trust_hash", self.trust_hash)
        _validate_optional_text("release_capability_id", self.release_capability_id)
        if (self.trust_id is None) != (self.trust_hash is None):
            raise ValueError("trust_id and trust_hash must either both be set or both be absent")


@dataclass(frozen=True)
class DataReleaseBinding:
    sink: str
    sink_identity_sha256: str | None
    trust_id: str
    trust_hash: str
    registry_generation: int
    manifest_hash: str
    labels_hash: str
    source_refs_hash: str
    payload_hash: str
    operation: str
    target_state_version: str | int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported data release binding schema_version: {self.schema_version}")
        DataSink(self.sink, self.sink_identity_sha256)
        for name in ("trust_id", "operation"):
            _require_text(name, getattr(self, name))
        for name in ("trust_hash", "manifest_hash", "labels_hash", "source_refs_hash", "payload_hash"):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.registry_generation, bool)
            or not isinstance(self.registry_generation, int)
            or self.registry_generation < 0
        ):
            raise ValueError("registry_generation must be a non-negative integer")
        if self.target_state_version is not None and (
            isinstance(self.target_state_version, bool)
            or not isinstance(self.target_state_version, (str, int))
        ):
            raise ValueError("target_state_version must be a string, integer, or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sink": self.sink,
            "sink_identity_sha256": self.sink_identity_sha256,
            "trust_id": self.trust_id,
            "trust_hash": self.trust_hash,
            "registry_generation": self.registry_generation,
            "manifest_hash": self.manifest_hash,
            "labels_hash": self.labels_hash,
            "source_refs_hash": self.source_refs_hash,
            "payload_hash": self.payload_hash,
            "operation": self.operation,
            "target_state_version": self.target_state_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataReleaseBinding:
        if not isinstance(value, Mapping):
            raise ValueError("data release binding must be an object")
        expected = {
            "schema_version",
            "sink",
            "sink_identity_sha256",
            "trust_id",
            "trust_hash",
            "registry_generation",
            "manifest_hash",
            "labels_hash",
            "source_refs_hash",
            "payload_hash",
            "operation",
            "target_state_version",
        }
        unknown = set(value) - expected
        if unknown:
            raise ValueError(f"data release binding contains unknown fields: {sorted(unknown)}")
        return cls(
            schema_version=value.get("schema_version", 1),
            sink=value.get("sink"),
            sink_identity_sha256=value.get("sink_identity_sha256"),
            trust_id=value.get("trust_id"),
            trust_hash=value.get("trust_hash"),
            registry_generation=value.get("registry_generation"),
            manifest_hash=value.get("manifest_hash"),
            labels_hash=value.get("labels_hash"),
            source_refs_hash=value.get("source_refs_hash"),
            payload_hash=value.get("payload_hash"),
            operation=value.get("operation"),
            target_state_version=value.get("target_state_version"),
        )

    @classmethod
    def normalize(cls, value: DataReleaseBinding | Mapping[str, Any]) -> dict[str, Any]:
        selected = value if isinstance(value, cls) else cls.from_dict(value)
        return selected.to_dict()


@dataclass(frozen=True)
class FileLabelBinding:
    binding_id: str
    normalized_path: str
    content_sha256: str | None
    labels: DataLabels
    source_refs: tuple[DataSourceRef, ...]
    generation: int
    tombstoned: bool
    active: bool
    created_by: str
    created_at: str
    superseded_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("binding_id", "normalized_path", "created_by", "created_at"):
            _require_text(name, getattr(self, name))
        if "\x00" in self.normalized_path:
            raise ValueError("normalized_path must not contain NUL")
        _optional_sha256("content_sha256", self.content_sha256)
        if not isinstance(self.labels, DataLabels):
            object.__setattr__(self, "labels", DataLabels.from_dict(self.labels))
        object.__setattr__(self, "source_refs", _normalize_source_refs(self.source_refs))
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 1:
            raise ValueError("file label generation must be a positive integer")
        if not isinstance(self.tombstoned, bool) or not isinstance(self.active, bool):
            raise ValueError("file label tombstoned and active fields must be boolean")
        if self.tombstoned and self.content_sha256 is not None:
            raise ValueError("tombstoned file label binding must not carry a content hash")
        if not self.tombstoned and self.content_sha256 is None:
            raise ValueError("live file label binding requires a content hash")
        _validate_optional_text("superseded_at", self.superseded_at)
        if self.active and self.superseded_at is not None:
            raise ValueError("active file label binding cannot have superseded_at")
        if not self.active and self.superseded_at is None:
            raise ValueError("inactive file label binding requires superseded_at")


def sink_pattern_matches(pattern: str, sink: str) -> bool:
    _validate_sink_pattern(pattern)
    DataSink(sink)
    return sink.startswith(pattern[:-1]) if pattern.endswith("*") else sink == pattern


def _coerce_enum(enum_type: type[StrEnum], value: Any, label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{label} must be one of: {allowed}") from exc


def _validate_sink_pattern(pattern: str) -> None:
    _require_text("sink pattern", pattern)
    if pattern != pattern.strip() or ":" not in pattern or "\x00" in pattern:
        raise ValueError("sink pattern must be a canonical namespaced value")
    if pattern.count("*") > 1 or ("*" in pattern and not pattern.endswith("*")):
        raise ValueError("sink pattern only supports one trailing wildcard")
    namespace, target = pattern.split(":", 1)
    if not namespace or not target:
        raise ValueError("sink pattern must include a non-empty namespace and target")


def _normalize_clearance_identities(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence, not a string")
    selected: list[str] = []
    for value in values:
        _require_text(name, value)
        if value in {"*", "mixed"}:
            raise ValueError(f"{name} must explicitly enumerate identities and cannot contain {value!r}")
        if value not in selected:
            selected.append(value)
    return tuple(sorted(selected))


def _normalize_source_refs(values: Iterable[DataSourceRef | Mapping[str, Any]]) -> tuple[DataSourceRef, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("source_refs must be a sequence")
    selected: dict[tuple[str, int, str], DataSourceRef] = {}
    for value in values:
        ref = value if isinstance(value, DataSourceRef) else DataSourceRef.from_dict(value)
        selected[(ref.oid, ref.version, ref.content_sha256)] = ref
    return tuple(selected[key] for key in sorted(selected))


def _merge_identity(values: Iterable[str | None], mixed: str | None) -> str | None:
    selected = sorted({value for value in values if value not in {None, ""}})
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return mixed


def _merge_declassification_authority(values: Iterable[str | None]) -> str | None:
    selected = tuple(values)
    if not selected or selected[0] is None:
        return None
    return selected[0] if all(value == selected[0] for value in selected) else None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _optional_sha256(name: str, value: Any) -> None:
    if value is not None:
        _require_sha256(name, value)


def _require_text(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{name} must be a non-empty string without control characters")


def _validate_optional_text(name: str, value: Any, *, allow_mixed: bool = False) -> None:
    if value is None:
        return
    _require_text(name, value)
    if value != value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a canonical string of at most 256 characters")
    if value == "mixed" and not allow_mixed:
        raise ValueError(f"{name} must not use the reserved mixed identity")
