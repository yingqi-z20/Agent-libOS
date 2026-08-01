from __future__ import annotations

from contextlib import AbstractContextManager
from collections.abc import Iterable
from typing import Any, Protocol

from agent_libos.models import (
    AuditRecord,
    Capability,
    ExternalEffectPage,
    ExternalEffectRecord,
    ExternalEffectRecoveryQuery,
    ExternalEffectRecoverySettlement,
    OperationEvidenceLink,
    OperationRecord,
)


class ProtectedEffectPort(Protocol):
    """Atomic persistence surface for protected provider-effect settlement."""

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[Any]:
        ...

    def insert_external_effect(self, record: ExternalEffectRecord) -> None:
        ...

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None:
        ...

    def list_external_effects(
        self,
        *,
        created_after: str | None = None,
        pid: str | None = None,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]:
        ...

    def query_external_effect_recovery(
        self,
        query: ExternalEffectRecoveryQuery,
    ) -> ExternalEffectPage:
        ...

    def get_external_effect_by_idempotency(
        self,
        pid: str,
        idempotency_key: str,
    ) -> ExternalEffectRecord | None:
        ...

    def finalize_external_effect(
        self,
        intent_effect_id: str,
        record: ExternalEffectRecord,
    ) -> bool:
        ...

    def transition_external_effect(
        self,
        effect_id: str,
        *,
        expected_states: Iterable[str],
        transaction_state: str,
        provider_metadata: dict[str, Any] | None = None,
        provider_receipt: dict[str, Any] | None = None,
        updated_at: str,
    ) -> bool:
        ...

    def settle_external_effect_recovery(
        self,
        effect_id: str,
        *,
        expected_transaction_state: str,
        provider_state: str,
        provider_metadata: dict[str, Any],
        provider_receipt: dict[str, Any],
        audit_record: AuditRecord,
        updated_at: str,
        run_id: str | None = None,
        runtime_epoch: int | None = None,
    ) -> ExternalEffectRecoverySettlement | None:
        """Atomically settle one provider-verified recovery conclusion.

        Supplying ``run_id`` and ``runtime_epoch`` adds an exact TaskRun
        membership/epoch fence.  The provider verification itself deliberately
        lives above this persistence port.
        """

        ...

    def abandon_external_effect_intent(self, effect_id: str) -> bool:
        ...

    def get_capability_use_reservation(
        self,
        reservation_id: str,
    ) -> dict[str, Any] | None:
        ...

    def list_operation_evidence(
        self,
        *,
        operation_ids: Iterable[str] | None = None,
        evidence_types: Iterable[str] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationEvidenceLink]:
        ...

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        ...

    def get_capability(self, cap_id: str) -> Capability | None:
        ...


class EffectAuthorityPort(Protocol):
    """Task-authority checks required at a provider-effect boundary."""

    def assert_effect(self, pid: str, effect_class: str) -> None:
        ...

    def get_for_process(self, pid: str) -> Any | None:
        ...
