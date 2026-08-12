from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    approval_binding_sha256,
    normalize_approval_binding,
)
from agent_libos.models import Capability, CapabilityEffect, CapabilityStatus
from agent_libos.models.external_effect import ExternalEffectRollbackStatus
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import SemanticRuntimeMode, SemanticTripCode
from agent_libos.storage import SemanticMachineOutcomeRecord


_TERMINAL_OUTCOMES = frozenset(
    {"succeeded", "failed", "outcome_unknown", "expired", "revoked", "race_lost"}
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticAuthorityRecoverySummary:
    scanned: int = 0
    retained_not_started: int = 0
    succeeded: int = 0
    failed: int = 0
    outcome_unknown: int = 0
    expired: int = 0
    revoked: int = 0
    already_terminal: int = 0


class SemanticMachineAuthorityRecovery:
    """Bounded startup reconciliation for issued exact-once grants.

    It never dispatches or replays a provider call.  Prepared/not-started
    authority keeps the same capability and inflight reservation; every known
    terminal conclusion wins one append-only slot and releases that reservation
    in the same transaction.  Ambiguity exhausts authority and trips control
    before Runtime admission opens.
    """

    def __init__(
        self,
        repository: Any,
        *,
        capability_reader: Callable[[str], Capability | None],
        capability_revoker: Callable[[str], Any],
        capability_expired: Callable[[Capability], bool],
        effect_reader: Callable[[str], Any | None],
        rate_budget: Any,
        control: Any,
        local_trip: Callable[[], None],
    ) -> None:
        callbacks = (
            capability_reader,
            capability_revoker,
            capability_expired,
            effect_reader,
            local_trip,
        )
        if repository is None or any(not callable(item) for item in callbacks):
            raise TypeError("semantic authority recovery dependencies are invalid")
        self._repository = repository
        self._capability_reader = capability_reader
        self._capability_revoker = capability_revoker
        self._capability_expired = capability_expired
        self._effect_reader = effect_reader
        self._rate_budget = rate_budget
        self._control = control
        self._local_trip = local_trip

    def recover(
        self,
        *,
        page_size: int = 100,
        max_records: int = 10_000,
    ) -> SemanticAuthorityRecoverySummary:
        if type(page_size) is not int or not 1 <= page_size <= 500:
            raise ValueError("semantic recovery page_size must be in [1, 500]")
        if type(max_records) is not int or not 1 <= max_records <= 100_000:
            raise ValueError("semantic recovery max_records must be in [1, 100000]")
        counts = {field: 0 for field in SemanticAuthorityRecoverySummary.__dataclass_fields__}
        after = None
        while counts["scanned"] < max_records:
            page = self._repository.query_unresolved_semantic_machine_settlements(
                after=after,
                limit=min(page_size, max_records - counts["scanned"]),
            )
            for settlement in page.records:
                result = self._recover_one(settlement)
                counts["scanned"] += 1
                counts[result] += 1
            if page.next_cursor is None:
                return SemanticAuthorityRecoverySummary(**counts)
            after = page.next_cursor
        # A hard cap is not a best-effort success: unexamined issued grants
        # would otherwise become reachable when admission opens.
        probe = self._repository.query_unresolved_semantic_machine_settlements(
            after=after,
            limit=1,
        )
        if probe.records:
            self._trip_recovery_overflow(max_records)
            raise CapabilityDenied("semantic authority recovery exceeded its bound")
        return SemanticAuthorityRecoverySummary(**counts)

    def _trip_recovery_overflow(self, max_records: int) -> None:
        self._local_trip()
        state = self._control.current()
        if (
            state.mode is SemanticRuntimeMode.CANARY_AUTO
            and not state.tripped
            and state.active_epoch_id is not None
        ):
            self._control.trip(
                SemanticTripCode.BINDING_MISMATCH,
                evidence_sha256=_digest(
                    {
                        "schema_version": 1,
                        "recovery": "bounded_scan_overflow",
                        "active_epoch_id": state.active_epoch_id,
                        "max_records": max_records,
                    }
                ),
            )

    def _recover_one(self, settlement: Any) -> str:
        if self._terminal_outcome(settlement.settlement_id) is not None:
            return "already_terminal"
        capability = (
            self._capability_reader(settlement.capability_id)
            if isinstance(settlement.capability_id, str)
            else None
        )
        effect = self._effect_reader(settlement.effect_id)
        outcome = self._classify(settlement, capability, effect)
        if outcome == "retained_not_started":
            return outcome
        if outcome == "outcome_unknown":
            self._trip_unknown(settlement)
        self._terminalize(settlement, capability, outcome)
        return outcome

    def _classify(
        self,
        settlement: Any,
        capability: Capability | None,
        effect: Any | None,
    ) -> str:
        if capability is None or not self._capability_matches(settlement, capability):
            return "outcome_unknown"
        if capability.uses_remaining == 0:
            if effect is None:
                return "outcome_unknown"
            if effect.effect_state == "finalized" and effect.transaction_state == "committed":
                return "succeeded"
            if self._known_failed(effect):
                return "failed"
            # Ambiguity outranks a prior durable trip/revocation.  This lets a
            # crash after committing the trip but before appending the terminal
            # lifecycle finish as outcome_unknown on reopen.
            return "outcome_unknown"
        if capability.uses_remaining != 1:
            return "outcome_unknown"
        if effect is not None and not self._certified_not_started(effect):
            # Ambiguous/effectful state outranks a trip committed by an
            # earlier failed recovery attempt.  Reopen must finish the same
            # outcome_unknown slot, never relabel it as a mere revocation.
            return "outcome_unknown"
        if self._capability_expired(capability):
            return "expired"
        if not self._control_matches(settlement) or capability.status is not CapabilityStatus.ACTIVE:
            return "revoked"
        return "retained_not_started"

    def _capability_matches(self, settlement: Any, capability: Capability) -> bool:
        try:
            binding = normalize_approval_binding(
                capability.constraints.get(APPROVAL_BINDING_KEY)
            )
            metadata = capability.metadata.get("semantic_auto_approval")
            rule_id = settlement.matched_rule_id
            if not isinstance(rule_id, str) or not rule_id:
                return False
            budget_id = self._rate_budget.bucket_id_for(
                epoch_id=settlement.epoch_id,
                tenant_bucket_sha256=settlement.tenant_bucket_sha256,
                rule_id=rule_id,
            )
            expected_metadata = {
                "schema_version": 1,
                "binding_sha256": settlement.binding_sha256,
                "request_id": settlement.request_id,
                "assessment_id": settlement.assessment_id,
                "policy_epoch_id": settlement.epoch_id,
                "matched_rule_id": rule_id,
                "settlement_id": settlement.settlement_id,
                "budget_bucket_id": budget_id,
            }
            return bool(
                isinstance(metadata, Mapping)
                and dict(metadata) == expected_metadata
                and self._capability_shape_matches(
                    settlement,
                    capability,
                    binding,
                )
                and self._binding_settlement_matches(settlement, binding)
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            return False

    @staticmethod
    def _capability_shape_matches(
        settlement: Any,
        capability: Capability,
        binding: Mapping[str, Any],
    ) -> bool:
        return bool(
            settlement.outcome == "issued"
            and capability.cap_id == settlement.capability_id
            and capability.subject == settlement.pid == binding["pid"]
            and capability.resource == binding["resource"]
            and capability.rights == {binding["right"]}
            and capability.effect is CapabilityEffect.ALLOW
            and capability.issued_by
            == f"policy:semantic:{settlement.epoch_id}"
            and capability.issued_at == binding["issued_at"]
            and capability.expires_at == binding["expires_at"]
            and capability.delegable is False
            and capability.revocable is True
            and capability.issuer_cap_id is None
            and capability.parent_cap_id is None
            and capability.delegation_depth == 0
            and capability.max_delegation_depth is None
        )

    @staticmethod
    def _binding_settlement_matches(
        settlement: Any,
        binding: Mapping[str, Any],
    ) -> bool:
        return bool(
            binding["effect_id"] == settlement.effect_id
            and binding["request_id"] == settlement.request_id
            and binding["request_revision"] == settlement.request_revision
            and binding["assessment_id"] == settlement.assessment_id
            and binding["authority_operation"] == settlement.action_id
            and binding["policy_epoch_id"] == settlement.epoch_id
            and binding["policy_epoch_sha256"] == settlement.policy_sha256
            and binding["tenant_bucket_sha256"]
            == settlement.tenant_bucket_sha256
            and approval_binding_sha256(binding) == settlement.binding_sha256
        )

    def _control_matches(self, settlement: Any) -> bool:
        state = self._control.current()
        return bool(
            state.mode is SemanticRuntimeMode.CANARY_AUTO
            and not state.tripped
            and state.active_epoch_id == settlement.epoch_id
            and state.active_policy_sha256 == settlement.policy_sha256
        )

    @staticmethod
    def _certified_not_started(effect: Any) -> bool:
        effect_state = getattr(effect, "effect_state", None)
        transaction_state = getattr(effect, "transaction_state", None)
        if transaction_state == "prepared":
            return effect_state != "finalized"
        # Provider flags are supporting evidence only.  A committed,
        # dispatched, or otherwise completed effect must never retain a
        # replayable one-shot grant because it carries a contradictory flag.
        if effect_state != "finalized" or transaction_state != "failed":
            return False
        receipt = getattr(effect, "provider_receipt", None)
        metadata = getattr(effect, "provider_metadata", None)
        if isinstance(metadata, Mapping) and (
            metadata.get("provider_reconciliation_state") == "not_started"
            and metadata.get("certified_not_started") is True
        ):
            return True
        if not isinstance(receipt, Mapping):
            return False
        return bool(
            (
                receipt.get("dispatch_status") == "not_started"
                and receipt.get("certified") is True
            )
            or receipt.get("certified_not_started") is True
        )

    @staticmethod
    def _known_failed(effect: Any) -> bool:
        return bool(
            getattr(effect, "effect_state", None) == "finalized"
            and getattr(effect, "transaction_state", None) == "failed"
            and getattr(effect, "rollback_status", None)
            is not ExternalEffectRollbackStatus.UNKNOWN
        )

    def _terminal_outcome(self, settlement_id: str) -> str | None:
        page = self._repository.query_semantic_machine_outcomes(
            after=None,
            limit=16,
            settlement_id=settlement_id,
        )
        if page.next_cursor is not None:
            raise ValidationError("semantic settlement lifecycle exceeds its bound")
        terminal = [item.outcome for item in page.records if item.outcome in _TERMINAL_OUTCOMES]
        if len(set(terminal)) > 1:
            raise ValidationError("semantic settlement has contradictory terminal outcomes")
        return terminal[0] if terminal else None

    def _trip_unknown(self, settlement: Any) -> None:
        self._local_trip()
        state = self._control.current()
        if (
            state.mode is SemanticRuntimeMode.CANARY_AUTO
            and not state.tripped
        ):
            self._control.trip(
                SemanticTripCode.PROVIDER_OUTCOME_UNKNOWN,
                evidence_sha256=_digest(
                    {
                        "schema_version": 1,
                        "recovery": "provider_outcome_unknown",
                        "settlement_id": settlement.settlement_id,
                        "effect_id": settlement.effect_id,
                        "settlement_epoch_id": settlement.epoch_id,
                        "active_epoch_id": state.active_epoch_id,
                    }
                ),
                tenant_bucket_sha256=settlement.tenant_bucket_sha256,
            )

    def _terminalize(
        self,
        settlement: Any,
        capability: Capability | None,
        outcome: str,
    ) -> None:
        if outcome not in _TERMINAL_OUTCOMES:
            raise ValidationError("semantic recovery outcome is not terminal")
        budget_id = self._rate_budget.bucket_id_for(
            epoch_id=settlement.epoch_id,
            tenant_bucket_sha256=settlement.tenant_bucket_sha256,
            rule_id=settlement.matched_rule_id,
        )
        with self._repository.transaction():
            if self._terminal_outcome(settlement.settlement_id) is not None:
                return
            if capability is not None and capability.uses_remaining == 0:
                self._append_outcome(settlement, "consumed", slot="consumed")
            if (
                capability is not None
                and capability.status is CapabilityStatus.ACTIVE
                and outcome in {"outcome_unknown", "expired", "revoked"}
            ):
                self._capability_revoker(capability.cap_id)
            appended = self._append_outcome(settlement, outcome, slot="terminal")
            if appended:
                self._rate_budget.release(budget_id)

    def _append_outcome(
        self,
        settlement: Any,
        outcome: str,
        *,
        slot: str,
    ) -> bool:
        identity = {
            "schema_version": 1,
            "lifecycle_slot": slot,
            "effect_id": settlement.effect_id,
            "settlement_id": settlement.settlement_id,
            "capability_id": settlement.capability_id,
            "binding_sha256": settlement.binding_sha256,
        }
        evidence = {
            "schema_version": 1,
            "source": "startup_recovery",
            "outcome": outcome,
            **identity,
        }
        return self._repository.append_semantic_machine_outcome_if_absent(
            SemanticMachineOutcomeRecord(
                outcome_id="semantic-outcome:" + _digest(identity),
                settlement_id=settlement.settlement_id,
                effect_id=settlement.effect_id,
                outcome=outcome,
                evidence_sha256=_digest(evidence),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )


__all__ = [
    "SemanticAuthorityRecoverySummary",
    "SemanticMachineAuthorityRecovery",
]
