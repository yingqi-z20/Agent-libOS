from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_libos.models.base import AuditID, EventID, PID, StrEnum


_EXTERNAL_EFFECT_STATES = frozenset({"pending", "finalized"})
_EXTERNAL_EFFECT_TRANSACTION_STATES = frozenset(
    {
        "prepared",
        "authorized",
        "approved",
        "dispatched",
        "committed",
        "failed",
        "unknown",
        "compensated",
    }
)
_PAYLOAD_RETENTION_SCHEMA_VERSION = 1
_PAYLOAD_RETENTION_TIERS = frozenset({"full", "summary", "hash_only"})
_SHA256_HEX_LENGTH = 64


class ExternalEffectRollbackClass(StrEnum):
    IRREVERSIBLE = "irreversible"
    ROLLBACKABLE = "rollbackable"
    NO_ROLLBACK_REQUIRED = "no_rollback_required"
    UNKNOWN = "unknown"


class ExternalEffectRollbackStatus(StrEnum):
    NOT_SUPPORTED = "not_supported"
    NOT_APPLIED = "not_applied"
    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"


def default_external_effect_rollback_status(
    rollback_class: ExternalEffectRollbackClass,
) -> ExternalEffectRollbackStatus:
    """Return the conservative omitted status for a rollback class."""

    return {
        ExternalEffectRollbackClass.IRREVERSIBLE: ExternalEffectRollbackStatus.NOT_SUPPORTED,
        ExternalEffectRollbackClass.ROLLBACKABLE: ExternalEffectRollbackStatus.NOT_APPLIED,
        ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED: ExternalEffectRollbackStatus.NOT_REQUIRED,
        ExternalEffectRollbackClass.UNKNOWN: ExternalEffectRollbackStatus.UNKNOWN,
    }[rollback_class]


@dataclass(frozen=True)
class ExternalEffectClassification:
    rollback_class: ExternalEffectRollbackClass
    rollback_status: ExternalEffectRollbackStatus
    state_mutation: bool
    information_flow: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalEffectRecord:
    effect_id: str
    record_id: AuditID | None
    event_id: EventID | None
    pid: PID
    provider: str
    operation: str
    target: str | None
    rollback_class: ExternalEffectRollbackClass
    rollback_status: ExternalEffectRollbackStatus
    state_mutation: bool
    information_flow: bool
    provider_metadata: dict[str, Any]
    created_at: str
    effect_state: str = "finalized"
    transaction_state: str = "committed"
    canonical_args_hash: str | None = None
    idempotency_key: str | None = None
    provider_receipt: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None
    payload_retention_schema_version: int = _PAYLOAD_RETENTION_SCHEMA_VERSION
    payload_retention_tier: str = "full"
    payload_retention_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.effect_state not in _EXTERNAL_EFFECT_STATES:
            raise ValueError(f"invalid external effect state: {self.effect_state!r}")
        if self.transaction_state not in _EXTERNAL_EFFECT_TRANSACTION_STATES:
            raise ValueError(f"invalid external effect transaction state: {self.transaction_state!r}")
        if (
            isinstance(self.payload_retention_schema_version, bool)
            or not isinstance(self.payload_retention_schema_version, int)
            or self.payload_retention_schema_version
            != _PAYLOAD_RETENTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "invalid external effect payload-retention schema version: "
                f"{self.payload_retention_schema_version!r}"
            )
        if self.payload_retention_tier not in _PAYLOAD_RETENTION_TIERS:
            raise ValueError(
                "invalid external effect payload-retention tier: "
                f"{self.payload_retention_tier!r}"
            )
        if self.payload_retention_tier == "full":
            if self.payload_retention_sha256 is not None:
                raise ValueError(
                    "full external effect payload cannot have a retention digest"
                )
        elif not _valid_sha256(self.payload_retention_sha256):
            raise ValueError(
                "retained external effect payload requires a SHA-256 digest"
            )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, order=True)
class ExternalEffectCursor:
    """Stable keyset cursor for bounded external-effect scans."""

    created_at: str
    effect_id: str

    def __post_init__(self) -> None:
        if not self.created_at:
            raise ValueError("external effect cursor created_at must not be empty")
        if not self.effect_id:
            raise ValueError("external effect cursor effect_id must not be empty")


@dataclass(frozen=True)
class ExternalEffectRecoveryQuery:
    """One bounded page of rows eligible for startup recovery."""

    effect_state: str = "pending"
    transaction_states: tuple[str, ...] = ()
    after: ExternalEffectCursor | None = None
    limit: int = 500

    def __post_init__(self) -> None:
        if self.effect_state not in _EXTERNAL_EFFECT_STATES:
            raise ValueError(
                f"invalid external effect recovery state: {self.effect_state!r}"
            )
        selected_states = tuple(dict.fromkeys(self.transaction_states))
        invalid = sorted(
            state
            for state in selected_states
            if state not in _EXTERNAL_EFFECT_TRANSACTION_STATES
        )
        if invalid:
            raise ValueError(
                "invalid external effect recovery transaction states: "
                f"{invalid}"
            )
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
            raise ValueError("external effect recovery page limit must be positive")
        object.__setattr__(self, "transaction_states", selected_states)


@dataclass(frozen=True)
class ExternalEffectPage:
    """A bounded recovery result and the cursor for the next page."""

    records: tuple[ExternalEffectRecord, ...]
    next_cursor: ExternalEffectCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty external effect page cannot have a next cursor")


@dataclass(frozen=True, slots=True)
class ExternalEffectRecoverySummary:
    """Bounded diagnostics for a fully processed external-effect backlog."""

    total_count: int
    sample_effect_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("external effect recovery total_count must be non-negative")
        if len(self.sample_effect_ids) > self.total_count:
            raise ValueError("external effect recovery sample exceeds total")
        if any(not isinstance(effect_id, str) or not effect_id for effect_id in self.sample_effect_ids):
            raise ValueError("external effect recovery sample IDs must not be empty")

    @property
    def truncated(self) -> bool:
        return len(self.sample_effect_ids) < self.total_count

    def __len__(self) -> int:
        return self.total_count


@dataclass(frozen=True, slots=True)
class ExternalEffectRecoverySettlement:
    """Atomic result of one authoritative external-effect reconciliation.

    ``provider_state`` preserves the provider's exact conclusion.  In
    particular, ``not_started`` is stored as a finalized ``failed``
    transaction while the provider conclusion remains explicit in both this
    result and the effect metadata.  That keeps the existing durable state
    vocabulary stable while proving that replay is safe.
    """

    effect: ExternalEffectRecord
    previous_transaction_state: str
    provider_state: str
    transition_seq: int
    audit_record_id: AuditID
    restored_capability_reservation_ids: tuple[str, ...] = ()
    committed_capability_reservation_ids: tuple[str, ...] = ()
    released_resource_reservation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.effect.effect_state != "finalized":
            raise ValueError("external effect recovery settlement must be finalized")
        if self.previous_transaction_state not in {
            "prepared",
            "authorized",
            "approved",
            "dispatched",
            "unknown",
        }:
            raise ValueError(
                "invalid external effect recovery previous transaction state: "
                f"{self.previous_transaction_state!r}"
            )
        if self.provider_state not in {
            "committed",
            "failed",
            "compensated",
            "not_started",
        }:
            raise ValueError(
                "invalid external effect recovery provider state: "
                f"{self.provider_state!r}"
            )
        if isinstance(self.transition_seq, bool) or not isinstance(
            self.transition_seq, int
        ) or self.transition_seq <= 0:
            raise ValueError(
                "external effect recovery transition sequence must be positive"
            )
        if not isinstance(self.audit_record_id, str) or not self.audit_record_id:
            raise ValueError(
                "external effect recovery settlement audit record must not be empty"
            )
        reservation_ids = (
            *self.restored_capability_reservation_ids,
            *self.committed_capability_reservation_ids,
            *self.released_resource_reservation_ids,
        )
        if any(not isinstance(item, str) or not item for item in reservation_ids):
            raise ValueError(
                "external effect recovery reservation IDs must not be empty"
            )
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError(
                "external effect recovery reservation IDs must not overlap"
            )
