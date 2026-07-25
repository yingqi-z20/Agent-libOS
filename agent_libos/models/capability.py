from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_libos.models.base import CapabilityID, StrEnum


class CapabilityRight(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    LINK = "link"
    DIFF = "diff"
    MATERIALIZE = "materialize"
    DELETE = "delete"
    GRANT = "grant"
    REVOKE = "revoke"
    APPROVE = "approve"
    ADMIN = "admin"


class CapabilityEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class _ImmutableDict(dict):
    """JSON-compatible mapping that rejects ordinary in-place mutation."""

    @staticmethod
    def _reject_mutation(*args: object, **kwargs: object) -> None:
        raise TypeError("authority rule conditions are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __copy__(self) -> "_ImmutableDict":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableDict":
        return self


class _ImmutableList(list):
    """JSON-compatible sequence that rejects ordinary in-place mutation."""

    @staticmethod
    def _reject_mutation(*args: object, **kwargs: object) -> None:
        raise TypeError("authority rule conditions are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation

    def __copy__(self) -> "_ImmutableList":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableList":
        return self


def _freeze_authority_condition(value: Any) -> Any:
    if isinstance(value, (_ImmutableDict, _ImmutableList)):
        return value
    if isinstance(value, Mapping):
        return _ImmutableDict(
            (key, _freeze_authority_condition(item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return _ImmutableList(_freeze_authority_condition(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_authority_condition(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_authority_condition(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


class CapabilityStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    DISABLED = "disabled"
    EXEC_REVOKED = "exec_revoked"


class AuthorityRisk(StrEnum):
    HARMLESS = "harmless"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"


class ResourceScope(StrEnum):
    EXACT = "exact"
    SUBTREE = "subtree"
    PREFIX = "prefix"
    GLOBAL = "global"


@dataclass(frozen=True)
class ResourcePattern:
    """Canonical typed resource pattern used for capability matching."""

    raw: str
    kind: str
    body: str
    scope: ResourceScope


@dataclass(frozen=True)
class OperationContext:
    """Primitive-specific authorization context recorded with decisions."""

    primitive: str | None = None
    operation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorityRule:
    """Deterministic rule attached to an authority grant or runtime profile."""

    rule_id: str
    operation: str
    effect: CapabilityEffect
    risk: AuthorityRisk
    conditions: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, Mapping):
            raise TypeError("authority rule conditions must be a mapping")
        object.__setattr__(
            self,
            "conditions",
            _freeze_authority_condition(self.conditions),
        )


@dataclass(frozen=True)
class CapabilityLease:
    expires_at: str | None = None
    uses_remaining: int | None = None


@dataclass(frozen=True)
class DelegationPolicy:
    delegable: bool = False
    revocable: bool = True
    max_delegation_depth: int | None = None


@dataclass(frozen=True)
class SandboxProfile:
    operation: str
    resource: str
    effect: CapabilityEffect
    risk: AuthorityRisk
    rule_id: str | None = None
    restrictions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilitySpec:
    resource: str
    rights: set[str]
    effect: CapabilityEffect = CapabilityEffect.ALLOW
    rules: list[AuthorityRule | dict[str, Any]] = field(default_factory=list)
    lease: CapabilityLease | dict[str, Any] | None = None
    delegation: DelegationPolicy | dict[str, Any] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    uses_remaining: int | None = None
    delegable: bool = False
    revocable: bool = True
    max_delegation_depth: int | None = None


@dataclass(frozen=True)
class CapabilityDecision:
    subject: str
    resource: str
    right: str
    allowed: bool
    effect: CapabilityEffect | None
    reason: str
    matched_capability_ids: list[CapabilityID] = field(default_factory=list)
    selected_capability_id: CapabilityID | None = None
    consume_capability_id: CapabilityID | None = None
    human_request_id: str | None = None
    issuer_chain: list[CapabilityID] = field(default_factory=list)
    constraint_results: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def policy(self) -> str:
        if self.effect == CapabilityEffect.ALLOW:
            return "allow_once" if self.consume_capability_id else "always_allow"
        if self.effect == CapabilityEffect.DENY:
            return "always_deny"
        if self.effect == CapabilityEffect.ASK:
            return "ask_each_time"
        return "missing"


@dataclass(frozen=True)
class Capability:
    cap_id: CapabilityID
    subject: str
    resource: str
    rights: set[str]
    constraints: dict[str, Any]
    issued_by: str
    issued_at: str
    expires_at: str | None = None
    delegable: bool = False
    revocable: bool = True
    effect: CapabilityEffect = CapabilityEffect.ALLOW
    issuer_cap_id: CapabilityID | None = None
    parent_cap_id: CapabilityID | None = None
    delegation_depth: int = 0
    max_delegation_depth: int | None = None
    uses_remaining: int | None = None
    status: CapabilityStatus = CapabilityStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def revoked(self) -> bool:
        return self.status in {
            CapabilityStatus.REVOKED,
            CapabilityStatus.EXEC_REVOKED,
        }

    @property
    def active(self) -> bool:
        return self.status == CapabilityStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class CapabilityUseReservationRecoverySummary:
    """Bounded diagnostics for stale capability-use reservations."""

    total_count: int
    sample_reservation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("capability reservation recovery total_count must be non-negative")
        if len(self.sample_reservation_ids) > self.total_count:
            raise ValueError("capability reservation recovery sample exceeds total")
        if any(
            not isinstance(reservation_id, str) or not reservation_id
            for reservation_id in self.sample_reservation_ids
        ):
            raise ValueError("capability reservation recovery sample IDs must not be empty")

    @property
    def truncated(self) -> bool:
        return len(self.sample_reservation_ids) < self.total_count

    def __len__(self) -> int:
        return self.total_count
