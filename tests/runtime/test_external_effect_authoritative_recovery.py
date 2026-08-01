from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.models import (
    AgentProcess,
    AuditRecord,
    Capability,
    CapabilityRight,
    CapabilityStatus,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    ResourceUsageReservationStatus,
    TaskRunRecord,
    TaskRunSpecV1,
    TaskRunStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import PostgresStore, SQLiteStore


BACKENDS = ("sqlite", pytest.param("postgres", marks=pytest.mark.postgres))
NOW = "2030-01-01T00:00:00+00:00"
LATER = "2030-01-01T00:00:01+00:00"


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_effect_recovery_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    scoped = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
    try:
        yield scoped
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


@contextlib.contextmanager
def _store(
    backend: str,
    tmp_path: Path,
) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        store = SQLiteStore(tmp_path / f"effect-recovery-{uuid4().hex}.sqlite")
        try:
            yield store
        finally:
            store.close()
        return
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            yield store
        finally:
            store.close()


def _seed_run_and_process(
    store: SQLiteStore | PostgresStore,
    *,
    run_id: str,
    pid: str,
) -> int:
    epoch = store.claim_runtime_epoch(f"runtime-{run_id}")
    spec = TaskRunSpecV1(
        goal={"goal": "recover provider effect"},
        display_title="Recover provider effect",
        image_id="base-agent:v0",
    )
    store.insert_task_run(
        TaskRunRecord.from_spec(
            run_id,
            spec,
            status=TaskRunStatus.NEEDS_ATTENTION,
            runtime_epoch=epoch,
            root_pid=pid,
            active_pid=pid,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    store.insert_process(
        AgentProcess(
            pid=pid,
            parent_pid=None,
            image_id="base-agent:v0",
            status=ProcessStatus.RUNNABLE,
            goal_oid=None,
            memory_view=None,
            capabilities=[],
            loaded_skills={},
            tool_table={},
            event_cursor=None,
            checkpoint_head=None,
            resource_budget=ResourceBudget(),
            resource_usage=ResourceUsage(),
            created_at=NOW,
            updated_at=NOW,
            task_run_id=run_id,
            task_run_epoch=epoch,
            task_run_role="root",
        )
    )
    return epoch


def _seed_effect_and_reservations(
    store: SQLiteStore | PostgresStore,
    *,
    effect_id: str,
    pid: str,
    transaction_state: str = "unknown",
) -> tuple[ExternalEffectRecord, str, str, str]:
    actor = pid
    contract_name = "primitive.test.write"
    capability_id = f"cap-{effect_id}"
    capability_reservation_id = f"cap-reservation-{effect_id}"
    resource_reservation_id = f"resource-reservation-{effect_id}"
    store.insert_capability(
        Capability(
            cap_id=capability_id,
            subject=actor,
            resource="provider:test",
            rights={CapabilityRight.WRITE.value},
            constraints={},
            issued_by="test",
            issued_at=NOW,
            uses_remaining=1,
        )
    )
    reserved = store.reserve_capability_uses(
        capability_id,
        capability_reservation_id,
        reserved_by=actor,
        reason=f"protected operation reserved authority for {contract_name}",
        created_at=NOW,
    )
    assert reserved is not None
    effect = ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider="provider",
        operation="write",
        target="target",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=True,
        information_flow=False,
        provider_metadata={
            "protected_operation": {
                "actor": actor,
                "contract_name": contract_name,
                "reservation_ids": [capability_reservation_id],
            },
            "completed_provider_phases": [],
        },
        created_at=NOW,
        effect_state="pending",
        transaction_state=transaction_state,
        idempotency_key=f"key-{effect_id}",
        updated_at=NOW,
    )
    store.insert_external_effect(effect)
    store.insert_resource_usage_reservation(
        reservation_id=resource_reservation_id,
        pid=pid,
        usage=ResourceUsage(external_write_bytes=7),
        reserved_by=effect_id,
        reason="provider write maximum",
        created_at=NOW,
    )
    return (
        effect,
        capability_id,
        capability_reservation_id,
        resource_reservation_id,
    )


def _audit(effect_id: str) -> AuditRecord:
    return AuditRecord(
        record_id=f"audit-{effect_id}",
        timestamp=LATER,
        actor="runtime.recovery",
        action="external_effect.recovery_settled",
        target=f"external_effect:{effect_id}",
        input_refs=[],
        output_refs=[],
        capability_refs=[],
        decision={"provider_state": "not_started"},
        correlation_id=effect_id,
    )


def _settlement_metadata(effect: ExternalEffectRecord) -> dict[str, object]:
    return {
        **effect.provider_metadata,
        "reconciled": True,
        "reconciliation_source": "test_provider_receipt",
        "provider_reconciliation_state": "not_started",
        "transaction_state": "failed",
        "outcome": "not_started",
        "certified_not_started": True,
        "dispatch_status": "not_started",
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("previous_state", ("unknown", "prepared"))
def test_not_started_receipt_atomically_restores_reserved_authority_and_evidence(
    backend: str,
    previous_state: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        run_id = "run-not-started"
        pid = "pid-not-started"
        effect_id = "effect-not-started"
        epoch = _seed_run_and_process(store, run_id=run_id, pid=pid)
        effect, capability_id, capability_reservation_id, resource_reservation_id = (
            _seed_effect_and_reservations(
                store,
                effect_id=effect_id,
                pid=pid,
                transaction_state=previous_state,
            )
        )

        if previous_state == "prepared":
            with pytest.raises(
                ValidationError,
                match="requires a certified not-started provider conclusion",
            ):
                store.settle_external_effect_recovery(
                    effect_id,
                    expected_transaction_state=previous_state,
                    provider_state="failed",
                    provider_metadata=_settlement_metadata(effect),
                    provider_receipt={"lookup_id": "uncertified-failure"},
                    audit_record=_audit(effect_id),
                    updated_at=LATER,
                    run_id=run_id,
                    runtime_epoch=epoch,
                )
            assert store.get_external_effect(effect_id) == effect

        result = store.settle_external_effect_recovery(
            effect_id,
            expected_transaction_state=previous_state,
            provider_state="not_started",
            provider_metadata=_settlement_metadata(effect),
            provider_receipt={"lookup_id": "provider-proof-1"},
            audit_record=_audit(effect_id),
            updated_at=LATER,
            run_id=run_id,
            runtime_epoch=epoch,
        )

        assert result is not None
        assert result.provider_state == "not_started"
        assert result.effect.effect_state == "finalized"
        assert result.effect.transaction_state == "failed"
        assert result.effect.idempotency_key is None
        assert result.restored_capability_reservation_ids == (
            capability_reservation_id,
        )
        assert result.released_resource_reservation_ids == (
            resource_reservation_id,
        )
        capability = store.get_capability(capability_id)
        assert capability is not None
        assert capability.status is CapabilityStatus.ACTIVE
        assert capability.uses_remaining == 1
        capability_reservation = store.get_capability_use_reservation(
            capability_reservation_id
        )
        assert capability_reservation is not None
        assert capability_reservation["status"] == "restored"
        resource_reservation = store.get_resource_usage_reservation(
            resource_reservation_id
        )
        assert resource_reservation is not None
        assert (
            resource_reservation["status"]
            == ResourceUsageReservationStatus.RELEASED.value
        )
        assert resource_reservation["settled_usage"] == ResourceUsage()
        audit = store.get_audit(result.audit_record_id)
        assert audit is not None
        assert audit.decision is not None
        assert audit.decision["transition_seq"] == result.transition_seq
        assert audit.decision["restored_capability_reservation_ids"] == [
            capability_reservation_id
        ]
        assert store.settle_external_effect_recovery(
            effect_id,
            expected_transaction_state=previous_state,
            provider_state="not_started",
            provider_metadata=_settlement_metadata(effect),
            provider_receipt={"lookup_id": "provider-proof-1"},
            audit_record=_audit(effect_id),
            updated_at=LATER,
            run_id=run_id,
            runtime_epoch=epoch,
        ) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_receipt_settlement_repeats_exact_run_epoch_and_effect_state_fences(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        run_id = "run-fenced-effect"
        pid = "pid-fenced-effect"
        effect_id = "effect-fenced"
        epoch = _seed_run_and_process(store, run_id=run_id, pid=pid)
        effect, capability_id, capability_reservation_id, resource_reservation_id = (
            _seed_effect_and_reservations(store, effect_id=effect_id, pid=pid)
        )

        assert store.settle_external_effect_recovery(
            effect_id,
            expected_transaction_state="dispatched",
            provider_state="not_started",
            provider_metadata=_settlement_metadata(effect),
            provider_receipt={"lookup_id": "provider-proof-fenced"},
            audit_record=_audit(effect_id),
            updated_at=LATER,
            run_id=run_id,
            runtime_epoch=epoch,
        ) is None
        assert store.settle_external_effect_recovery(
            effect_id,
            expected_transaction_state="unknown",
            provider_state="not_started",
            provider_metadata=_settlement_metadata(effect),
            provider_receipt={"lookup_id": "provider-proof-fenced"},
            audit_record=_audit(effect_id),
            updated_at=LATER,
            run_id=run_id,
            runtime_epoch=epoch + 1,
        ) is None

        assert store.get_external_effect(effect_id) == effect
        capability = store.get_capability(capability_id)
        assert capability is not None
        assert capability.status is CapabilityStatus.REVOKED
        assert capability.uses_remaining == 0
        assert store.get_capability_use_reservation(capability_reservation_id)[
            "status"
        ] == "reserved"
        assert store.get_resource_usage_reservation(resource_reservation_id)[
            "status"
        ] == ResourceUsageReservationStatus.ACTIVE.value
        assert store.get_audit(f"audit-{effect_id}") is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_receipt_settlement_rolls_back_effect_and_reservations_if_audit_fails(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _store(backend, tmp_path) as store:
        run_id = "run-atomic-effect"
        pid = "pid-atomic-effect"
        effect_id = "effect-atomic"
        epoch = _seed_run_and_process(store, run_id=run_id, pid=pid)
        effect, capability_id, capability_reservation_id, resource_reservation_id = (
            _seed_effect_and_reservations(store, effect_id=effect_id, pid=pid)
        )
        ledger_seq = store.current_effect_ledger_seq()

        def fail_audit(_record: AuditRecord) -> None:
            raise RuntimeError("injected audit failure")

        monkeypatch.setattr(store, "insert_audit", fail_audit)
        with pytest.raises(RuntimeError, match="injected audit failure"):
            store.settle_external_effect_recovery(
                effect_id,
                expected_transaction_state="unknown",
                provider_state="not_started",
                provider_metadata=_settlement_metadata(effect),
                provider_receipt={"lookup_id": "provider-proof-atomic"},
                audit_record=_audit(effect_id),
                updated_at=LATER,
                run_id=run_id,
                runtime_epoch=epoch,
            )

        assert store.get_external_effect(effect_id) == effect
        assert store.current_effect_ledger_seq() == ledger_seq
        capability = store.get_capability(capability_id)
        assert capability is not None
        assert capability.status is CapabilityStatus.REVOKED
        assert capability.uses_remaining == 0
        assert store.get_capability_use_reservation(capability_reservation_id)[
            "status"
        ] == "reserved"
        assert store.get_resource_usage_reservation(resource_reservation_id)[
            "status"
        ] == ResourceUsageReservationStatus.ACTIVE.value
