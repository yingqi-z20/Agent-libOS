from __future__ import annotations

from agent_libos.capability.admission import (
    CapabilityAdmissionPort,
    install_instance_admission_guards,
    revalidate_admission,
)
from agent_libos.models import Capability, CapabilityDecision, EventType
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.ports import AuditPort, CapabilityStorePort, EventPort, OperationPort
from agent_libos.utils.ids import new_id, utc_now


CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS = frozenset(
    {
        "commit",
        "consume",
        "reserve",
        "reserve_decision",
        "restore",
    }
)


def _positive_count(value: object, *, operation: str) -> int:
    if type(value) is not int or value < 1:
        raise ValidationError(
            f"capability {operation} count must be a positive integer"
        )
    return value


class CapabilityLeaseService:
    """Atomic finite-use consumption, reservation, and settlement."""

    def __init__(
        self,
        store: CapabilityStorePort,
        audit: AuditPort,
        events: EventPort,
        operations: OperationPort,
        *,
        admission: CapabilityAdmissionPort | None = None,
    ) -> None:
        self.store = store
        self.audit = audit
        self.events = events
        self.operations = operations
        self._admission = admission
        install_instance_admission_guards(
            self,
            admission=admission,
            mutation_methods=CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS,
        )

    def consume(
        self,
        cap_id: str,
        *,
        used_by: str,
        reason: str = "capability use consumed",
        count: int = 1,
    ) -> Capability:
        count = _positive_count(count, operation="consume")
        with self.store.transaction():
            revalidate_admission(self._admission)
            cap = self._require_capability(cap_id)
            if cap.uses_remaining is None:
                revalidate_admission(self._admission)
                return cap
            updated = self.store.consume_capability_uses(cap_id, count)
            if updated is None:
                raise CapabilityDenied(f"capability use exhausted: {cap_id}")
            self.audit.record(
                actor=used_by,
                action="capability.consume",
                target=cap.resource,
                capability_refs=[cap_id],
                decision={"uses_remaining": updated.uses_remaining, "count": count, "reason": reason},
            )
            if updated.revoked:
                self._emit_revoked(updated, source=used_by, reason=reason)
            revalidate_admission(self._admission)
            return updated

    def reserve(
        self,
        cap_id: str,
        *,
        reserved_by: str,
        reason: str = "capability use reserved",
        count: int = 1,
    ) -> str:
        count = _positive_count(count, operation="reservation")
        with self.store.transaction():
            revalidate_admission(self._admission)
            cap = self._require_capability(cap_id)
            if cap.uses_remaining is None:
                raise ValidationError(f"capability is not finite-use: {cap_id}")
            reservation_id = new_id("capres")
            updated = self.store.reserve_capability_uses(
                cap_id,
                reservation_id,
                count=count,
                reserved_by=reserved_by,
                reason=reason,
                created_at=utc_now(),
            )
            if updated is None:
                raise CapabilityDenied(f"capability use exhausted: {cap_id}")
            self.audit.record(
                actor=reserved_by,
                action="capability.reserve_use",
                target=cap.resource,
                capability_refs=[cap_id],
                decision={
                    "reservation_id": reservation_id,
                    "uses_remaining": updated.uses_remaining,
                    "count": count,
                    "reason": reason,
                },
            )
            self._link_reservation(
                reservation_id,
                "reservation",
                {"capability_id": cap_id, "status": "reserved", "count": count},
                expect=True,
            )
            if updated.revoked:
                self._emit_revoked(
                    updated,
                    source=reserved_by,
                    reason=reason,
                    reservation_id=reservation_id,
                )
            revalidate_admission(self._admission)
            return reservation_id

    def reserve_decision(
        self,
        decision: CapabilityDecision | None,
        *,
        used_by: str,
        reason: str,
    ) -> str | None:
        if decision is None or decision.consume_capability_id is None:
            return None
        return self.reserve(str(decision.consume_capability_id), reserved_by=used_by, reason=reason)

    def commit(self, reservation_id: str | None, *, committed_by: str, reason: str) -> bool:
        if reservation_id is None:
            return False
        with self.store.transaction():
            revalidate_admission(self._admission)
            committed = self.store.commit_capability_use_reservation(
                reservation_id,
                updated_at=utc_now(),
            )
            self.audit.record(
                actor=committed_by,
                action="capability.commit_reserved_use",
                target=f"capability_reservation:{reservation_id}",
                decision={"committed": committed, "reason": reason},
            )
            self._link_reservation(
                reservation_id,
                "result",
                {"status": "committed" if committed else "commit_skipped"},
            )
            revalidate_admission(self._admission)
            return committed

    def restore(
        self,
        reservation_id: str | None,
        *,
        restored_by: str,
        reason: str = "reserved capability use restored",
    ) -> Capability | None:
        if reservation_id is None:
            return None
        with self.store.transaction():
            revalidate_admission(self._admission)
            updated = self.store.restore_capability_use_reservation(
                reservation_id,
                updated_at=utc_now(),
            )
            if updated is None:
                self.audit.record(
                    actor=restored_by,
                    action="capability.restore_reserved_use_skipped",
                    target=f"capability_reservation:{reservation_id}",
                    decision={"restored": False, "reason": reason},
                )
                self._link_reservation(reservation_id, "result", {"status": "restore_skipped"})
                revalidate_admission(self._admission)
                return None
            self.audit.record(
                actor=restored_by,
                action="capability.restore_reserved_use",
                target=updated.resource,
                capability_refs=[updated.cap_id],
                decision={
                    "reservation_id": reservation_id,
                    "uses_remaining": updated.uses_remaining,
                    "reason": reason,
                },
            )
            self.events.emit(
                EventType.CAPABILITY_GRANTED,
                source=restored_by,
                target=updated.subject,
                payload={
                    "capability_id": updated.cap_id,
                    "reason": reason,
                    "reservation_id": reservation_id,
                    "uses_remaining": updated.uses_remaining,
                },
            )
            self._link_reservation(
                reservation_id,
                "result",
                {"status": "restored", "capability_id": updated.cap_id},
            )
            revalidate_admission(self._admission)
            return updated

    def _require_capability(self, cap_id: str) -> Capability:
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        return cap

    def _link_reservation(
        self,
        reservation_id: str,
        role: str,
        metadata: dict[str, object],
        *,
        expect: bool = False,
    ) -> None:
        if expect:
            self.operations.expect("reservation")
        self.operations.link_evidence(
            "capability_reservation",
            reservation_id,
            role,
            metadata=metadata,
        )

    def _emit_revoked(
        self,
        cap: Capability,
        *,
        source: str,
        reason: str,
        reservation_id: str | None = None,
    ) -> None:
        payload = {"capability_id": cap.cap_id, "reason": reason}
        if reservation_id is not None:
            payload["reservation_id"] = reservation_id
        self.events.emit(
            EventType.CAPABILITY_REVOKED,
            source=source,
            target=cap.subject,
            payload=payload,
        )


__all__ = [
    "CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS",
    "CapabilityLeaseService",
]
