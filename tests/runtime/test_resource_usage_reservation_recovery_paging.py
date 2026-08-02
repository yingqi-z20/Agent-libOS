from __future__ import annotations

import contextlib
import os
import tracemalloc
from collections.abc import Iterator
from itertools import repeat
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults, load_config_file
from agent_libos.models import (
    AgentProcess,
    EventType,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    ResourceUsageReservation,
    ResourceUsageReservationCursor,
    ResourceUsageReservationPage,
    ResourceUsageReservationRecoverySummary,
    ResourceUsageReservationStatus,
)
from agent_libos.models.exceptions import ResourceLimitExceeded, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.resource_manager import ResourceManager
from agent_libos.storage import PostgresStore, SQLiteStore, UnitOfWork


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_usage_reservation_recovery_query_is_typed_stable_and_hard_bounded(
    backend: str,
) -> None:
    with _resource_store(backend, page_size=3, hard_limit=3) as store:
        unit = UnitOfWork(store)
        created_at = "2026-01-01T00:00:00Z"
        for index in range(7):
            _insert_reservation(
                unit,
                f"reservation-{index}",
                created_at=created_at,
            )
        _insert_reservation(
            unit,
            "reservation-terminal",
            created_at="2025-12-31T00:00:00Z",
        )
        assert unit.resources.settle_resource_usage_reservation(
            "reservation-terminal",
            status=ResourceUsageReservationStatus.RELEASED,
            settled_usage=ResourceUsage(),
            updated_at=created_at,
        )

        first = unit.resources.query_resource_usage_reservation_recovery(
            after=None,
            limit=3,
        )
        assert isinstance(first, ResourceUsageReservationPage)
        assert all(isinstance(record, ResourceUsageReservation) for record in first.records)
        assert [record.reservation_id for record in first.records] == [
            "reservation-0",
            "reservation-1",
            "reservation-2",
        ]
        assert first.next_cursor == ResourceUsageReservationCursor(
            created_at,
            "reservation-2",
        )

        # Removing the first page from the ACTIVE set must not offset-skip the
        # next page because the cursor is tied to the durable ordering key.
        for record in first.records:
            assert unit.resources.settle_resource_usage_reservation(
                record.reservation_id,
                status=ResourceUsageReservationStatus.RELEASED,
                settled_usage=ResourceUsage(),
                updated_at="2026-01-02T00:00:00Z",
            )
        second = unit.resources.query_resource_usage_reservation_recovery(
            after=first.next_cursor,
            limit=3,
        )
        third = unit.resources.query_resource_usage_reservation_recovery(
            after=second.next_cursor,
            limit=3,
        )
        assert [record.reservation_id for record in second.records] == [
            "reservation-3",
            "reservation-4",
            "reservation-5",
        ]
        assert [record.reservation_id for record in third.records] == [
            "reservation-6"
        ]
        assert third.next_cursor is None

        for invalid_limit in (0, -1, True, 4):
            with pytest.raises(ValidationError):
                unit.resources.query_resource_usage_reservation_recovery(
                    after=None,
                    limit=invalid_limit,
                )
        with pytest.raises(ValidationError, match="cursor"):
            unit.resources.query_resource_usage_reservation_recovery(
                after=object(),  # type: ignore[arg-type]
                limit=1,
            )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_charge_maximum_recovery_is_page_linear_and_converges_after_overage(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_size = 17
    reservation_count = 257
    with _resource_store(backend, page_size=page_size) as store:
        unit = UnitOfWork(store)
        pid = "pid-resource-recovery"
        unit.processes.insert_process(
            _process(
                pid,
                max_external_write_bytes=5,
            )
        )
        with unit.transaction():
            for index in range(reservation_count):
                effect_id = f"effect-{index:04d}"
                unit.evidence.insert_external_effect(
                    _dispatched_effect(effect_id, pid=pid)
                )
                _insert_reservation(
                    unit,
                    f"reservation-{index:04d}",
                    pid=pid,
                    reserved_by=effect_id,
                    usage=ResourceUsage(external_write_bytes=2),
                    created_at="2026-01-01T00:00:00Z",
                )

        manager = _resource_manager(unit, store.config)
        original_query = (
            manager.resource_repository.query_resource_usage_reservation_recovery
        )
        query_cursors: list[ResourceUsageReservationCursor | None] = []

        def observed_query(
            *,
            after: ResourceUsageReservationCursor | None,
            limit: int,
        ) -> ResourceUsageReservationPage:
            query_cursors.append(after)
            return original_query(after=after, limit=limit)

        monkeypatch.setattr(
            manager.resource_repository,
            "query_resource_usage_reservation_recovery",
            observed_query,
        )
        monkeypatch.setattr(
            manager.resource_repository,
            "list_resource_usage_reservations",
            lambda **_kwargs: pytest.fail("recovery used the unbounded list API"),
        )

        summary = manager.recover_usage_reservations()

        assert isinstance(summary, ResourceUsageReservationRecoverySummary)
        assert summary.total_count == reservation_count
        assert len(summary.sample_reservation_ids) == page_size
        assert summary.truncated
        assert len(summary) == reservation_count
        assert len(query_cursors) == (
            reservation_count + page_size - 1
        ) // page_size
        assert query_cursors[0] is None
        recovered = store.list_resource_usage_reservations(pid=pid)
        assert len(recovered) == reservation_count
        assert all(
            reservation["status"]
            == ResourceUsageReservationStatus.CHARGED_MAXIMUM.value
            for reservation in recovered
        )
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.external_write_bytes == 2 * reservation_count
        assert process.status is ProcessStatus.KILLED


def test_recovery_settlement_and_charge_roll_back_together_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            resource_usage_reservation_recovery_page_size=1,
        )
    )
    path = tmp_path / "resource-recovery-atomicity.sqlite"
    store = SQLiteStore(path, config=config)
    try:
        unit = UnitOfWork(store)
        pid = "pid-crash-atomicity"
        unit.processes.insert_process(_process(pid, max_external_write_bytes=10))
        unit.evidence.insert_external_effect(_dispatched_effect("effect-crash", pid=pid))
        _insert_reservation(
            unit,
            "reservation-crash",
            pid=pid,
            reserved_by="effect-crash",
            usage=ResourceUsage(external_write_bytes=3),
        )
        manager = _resource_manager(unit, config)
        class InjectedCrash(BaseException):
            pass

        def crash_before_charge(*_args: Any, **_kwargs: Any) -> None:
            pending = unit.resources.get_resource_usage_reservation(
                "reservation-crash"
            )
            assert pending is not None
            assert pending.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
            raise InjectedCrash()

        monkeypatch.setattr(manager, "_charge", crash_before_charge)
        with pytest.raises(InjectedCrash):
            manager.recover_usage_reservations()

        rolled_back = unit.resources.get_resource_usage_reservation(
            "reservation-crash"
        )
        assert rolled_back is not None
        assert rolled_back.status is ResourceUsageReservationStatus.ACTIVE
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.external_write_bytes == 0
    finally:
        store.close()

    reopened = SQLiteStore(path, config=config)
    try:
        unit = UnitOfWork(reopened)
        manager = _resource_manager(unit, config)
        durable = unit.resources.get_resource_usage_reservation(
            "reservation-crash"
        )
        assert durable is not None
        assert durable.status is ResourceUsageReservationStatus.ACTIVE

        summary = manager.recover_usage_reservations()
        assert summary.total_count == 1
        settled = unit.resources.get_resource_usage_reservation(
            "reservation-crash"
        )
        assert settled is not None
        assert settled.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.external_write_bytes == 3
    finally:
        reopened.close()


def test_recovery_charge_rolls_back_if_settlement_fails_after_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        pid = "pid-crash-after-charge"
        unit.processes.insert_process(_process(pid, max_external_write_bytes=10))
        unit.evidence.insert_external_effect(
            _dispatched_effect("effect-crash-after-charge", pid=pid)
        )
        _insert_reservation(
            unit,
            "reservation-crash-after-charge",
            pid=pid,
            reserved_by="effect-crash-after-charge",
            usage=ResourceUsage(external_write_bytes=3),
        )
        manager = _resource_manager(unit, store.config)
        original_charge = manager._charge

        class InjectedCrash(BaseException):
            pass

        def charge_then_crash(*args: Any, **kwargs: Any) -> None:
            original_charge(*args, **kwargs)
            pending = unit.resources.get_resource_usage_reservation(
                "reservation-crash-after-charge"
            )
            assert pending is not None
            assert pending.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
            process = unit.processes.get_process(pid)
            assert process is not None
            assert process.resource_usage.external_write_bytes == 3
            raise InjectedCrash()

        monkeypatch.setattr(manager, "_charge", charge_then_crash)
        with pytest.raises(InjectedCrash):
            manager.recover_usage_reservations()

        rolled_back = unit.resources.get_resource_usage_reservation(
            "reservation-crash-after-charge"
        )
        assert rolled_back is not None
        assert rolled_back.status is ResourceUsageReservationStatus.ACTIVE
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.external_write_bytes == 0
    finally:
        store.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_llm_unknown_recovery_charges_aggregate_without_inventing_components(
    backend: str,
) -> None:
    with _resource_store(backend, page_size=2, hard_limit=2) as store:
        unit = UnitOfWork(store)
        pid = f"pid-llm-maximum-{backend}"
        unit.processes.insert_process(
            _process(
                pid,
                max_external_write_bytes=10,
                max_llm_calls=1,
                max_llm_total_tokens=100,
            )
        )
        effect_id = f"effect-llm-maximum-{backend}"
        reservation_id = f"reservation-llm-maximum-{backend}"
        unit.evidence.insert_external_effect(
            _dispatched_effect(effect_id, pid=pid)
        )
        _insert_reservation(
            unit,
            reservation_id,
            pid=pid,
            reserved_by=effect_id,
            reason="llm.request",
            usage=ResourceUsage(
                llm_calls=1,
                llm_prompt_tokens=80,
                llm_completion_tokens=20,
                llm_total_tokens=100,
            ),
        )
        manager = _resource_manager(unit, store.config)

        assert manager.recover_usage_reservations().total_count == 1
        reservation = unit.resources.get_resource_usage_reservation(
            reservation_id
        )
        assert reservation is not None
        assert reservation.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
        assert reservation.settled_usage == ResourceUsage(
            llm_calls=1,
            llm_total_tokens=100,
        )
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert process.resource_usage.llm_total_tokens == 100
        assert manager.recover_usage_reservations().total_count == 0


@pytest.mark.parametrize("backend", STORE_BACKENDS)
@pytest.mark.parametrize(
    "failure_type",
    [RuntimeError, KeyboardInterrupt],
    ids=["exception", "base_exception"],
)
def test_recovery_overage_runs_terminal_hooks_only_after_outer_commit(
    backend: str,
    failure_type: type[BaseException],
) -> None:
    with _resource_store(backend) as store:
        unit = UnitOfWork(store)
        pid = "pid-resource-settlement-post-commit"
        reservation_id = "reservation-resource-settlement-post-commit"
        effect_id = "effect-resource-settlement-post-commit"
        unit.processes.insert_process(_process(pid, max_external_write_bytes=1))
        unit.evidence.insert_external_effect(_dispatched_effect(effect_id, pid=pid))
        _insert_reservation(
            unit,
            reservation_id,
            pid=pid,
            reserved_by=effect_id,
            usage=ResourceUsage(external_write_bytes=2),
        )
        manager = _resource_manager(unit, store.config)
        parent_wakes: list[str] = []
        terminal_notifications: list[str] = []
        process_finalizations: list[tuple[tuple[str, ...], str]] = []
        callback_transaction_depths: list[int] = []

        def record_depth() -> None:
            callback_transaction_depths.append(int(store._transaction_depth))

        def wake_parent(killed_pid: str) -> None:
            record_depth()
            parent_wakes.append(killed_pid)

        def notify_terminal(killed_pid: str) -> None:
            record_depth()
            terminal_notifications.append(killed_pid)

        def finalize_processes(killed: list[str], *, reason: str) -> None:
            record_depth()
            process_finalizations.append((tuple(killed), reason))

        manager._wake_parent_waiting_on_child = wake_parent  # type: ignore[method-assign]
        manager.bind_object_task_terminal_notifier(notify_terminal)
        manager.bind_process_kill_finalizer(finalize_processes)
        before_event_ids = {event.event_id for event in manager.events.list()}
        before_audit_ids = {record.record_id for record in manager.audit.trace()}
        interruption = failure_type("injected resource settlement outer commit failure")
        original_guard = store._admission_commit_guard

        @contextlib.contextmanager
        def reject_commit() -> Iterator[None]:
            raise interruption
            yield

        store._admission_commit_guard = reject_commit
        try:
            with pytest.raises(failure_type) as caught:
                manager.recover_usage_reservations()
        finally:
            store._admission_commit_guard = original_guard

        assert caught.value is interruption
        assert parent_wakes == []
        assert terminal_notifications == []
        assert process_finalizations == []
        assert callback_transaction_depths == []
        reservation = unit.resources.get_resource_usage_reservation(reservation_id)
        assert reservation is not None
        assert reservation.status is ResourceUsageReservationStatus.ACTIVE
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.status is ProcessStatus.RUNNABLE
        assert process.resource_usage.external_write_bytes == 0
        assert not [
            event
            for event in manager.events.list()
            if event.event_id not in before_event_ids
            and event.type
            in {EventType.RESOURCE_CHARGED, EventType.RESOURCE_LIMIT_EXCEEDED}
        ]
        assert not [
            record
            for record in manager.audit.trace()
            if record.record_id not in before_audit_ids
            and record.action in {"resource.charge", "resource.limit_exceeded"}
        ]

        recovered = manager.recover_usage_reservations()
        assert recovered.total_count == 1
        assert parent_wakes == [pid]
        assert terminal_notifications == [pid]
        assert len(process_finalizations) == 1
        assert process_finalizations[0][0] == (pid,)
        assert callback_transaction_depths == [0, 0, 0]
        reservation = unit.resources.get_resource_usage_reservation(reservation_id)
        assert reservation is not None
        assert reservation.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
        assert reservation.settled_usage == ResourceUsage(external_write_bytes=2)
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.status is ProcessStatus.KILLED
        assert process.resource_usage.external_write_bytes == 2
        assert len(
            [
                event
                for event in manager.events.list()
                if event.event_id not in before_event_ids
                and event.type == EventType.RESOURCE_CHARGED
            ]
        ) == 1
        assert len(
            [
                event
                for event in manager.events.list()
                if event.event_id not in before_event_ids
                and event.type == EventType.RESOURCE_LIMIT_EXCEEDED
            ]
        ) == 1

        assert manager.recover_usage_reservations().total_count == 0
        assert parent_wakes == [pid]
        assert terminal_notifications == [pid]
        assert len(process_finalizations) == 1
        assert callback_transaction_depths == [0, 0, 0]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_ordinary_over_budget_settlement_rolls_back_the_reservation(
    backend: str,
) -> None:
    with _resource_store(backend) as store:
        unit = UnitOfWork(store)
        pid = "pid-ordinary-settlement-overage"
        unit.processes.insert_process(_process(pid, max_external_write_bytes=1))
        _insert_reservation(
            unit,
            "reservation-ordinary-overage",
            pid=pid,
            usage=ResourceUsage(external_write_bytes=2),
        )
        manager = _resource_manager(unit, store.config)

        with pytest.raises(ResourceLimitExceeded):
            manager.settle_usage_reservation(
                "reservation-ordinary-overage",
                actual_usage=ResourceUsage(external_write_bytes=2),
                source="test.ordinary-settlement",
            )

        reservation = unit.resources.get_resource_usage_reservation(
            "reservation-ordinary-overage"
        )
        assert reservation is not None
        assert reservation.status is ResourceUsageReservationStatus.ACTIVE
        process = unit.processes.get_process(pid)
        assert process is not None
        assert process.resource_usage.external_write_bytes == 0


def test_recovery_lease_is_required_before_the_first_repository_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        reads = 0

        def deny_recovery() -> None:
            raise RuntimeError("active startup recovery lease required")

        manager = ResourceManager(
            unit,
            AuditManager(unit.evidence),
            EventBus(unit.evidence),
            require_recovery_lease=deny_recovery,
        )

        def record_read(**_kwargs: Any) -> ResourceUsageReservationPage:
            nonlocal reads
            reads += 1
            return ResourceUsageReservationPage(records=())

        monkeypatch.setattr(
            manager.resource_repository,
            "query_resource_usage_reservation_recovery",
            record_read,
        )
        with pytest.raises(RuntimeError, match="active startup recovery lease"):
            manager.recover_usage_reservations()
        assert reads == 0
    finally:
        store.close()


def test_runtime_open_recovers_reservations_and_exposes_bounded_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resource-recovery-open.sqlite"
    store = SQLiteStore(path)
    try:
        _insert_reservation(UnitOfWork(store), "reservation-reopen")
    finally:
        store.close()

    runtime = Runtime.open(path)
    try:
        summary = runtime.recovered_resource_usage_reservations
        assert summary.total_count == 1
        assert summary.sample_reservation_ids == ("reservation-reopen",)
        reservation = runtime.uow.resources.get_resource_usage_reservation(
            "reservation-reopen"
        )
        assert reservation is not None
        assert reservation.status is ResourceUsageReservationStatus.RELEASED
    finally:
        runtime.close()


def test_million_row_recovery_keeps_pages_and_diagnostics_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_size = 31
    total = 1_000_000
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            resource_usage_reservation_recovery_page_size=page_size,
        )
    )
    store = SQLiteStore(":memory:", config=config)
    try:
        unit = UnitOfWork(store)
        manager = _resource_manager(unit, config)
        page = ResourceUsageReservationPage(
            records=tuple(
                ResourceUsageReservation(
                    reservation_id=f"reservation-{index}",
                    pid="pid-scale",
                    usage=ResourceUsage(external_write_bytes=1),
                    status=ResourceUsageReservationStatus.ACTIVE,
                    reserved_by=f"missing-effect-{index}",
                    reason="scale",
                    settled_usage=None,
                    created_at="2026-01-01T00:00:00Z",
                    updated_at="2026-01-01T00:00:00Z",
                )
                for index in range(page_size)
            )
        )
        monkeypatch.setattr(
            manager,
            "_iter_active_usage_reservations",
            lambda: repeat(page.records[0], total),
        )
        monkeypatch.setattr(
            manager.effects,
            "get_external_effect",
            lambda _effect_id: None,
        )
        monkeypatch.setattr(
            manager,
            "_settle_usage_reservation",
            lambda *_args, **_kwargs: ResourceUsage(),
        )

        tracemalloc.start()
        try:
            summary = manager.recover_usage_reservations()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert summary.total_count == total
        assert len(summary.sample_reservation_ids) == page_size
        assert summary.truncated
        assert peak < 5_000_000
    finally:
        store.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_usage_reservation_recovery_query_uses_status_first_index(
    backend: str,
) -> None:
    with _resource_store(backend) as store:
        sql = (
            "SELECT * FROM resource_usage_reservations "
            "WHERE status = ? AND (created_at, reservation_id) > (?, ?) "
            "ORDER BY created_at, reservation_id LIMIT ?"
        )
        params = (
            ResourceUsageReservationStatus.ACTIVE.value,
            "2026-01-01T00:00:00Z",
            "reservation-deep-cursor",
            10,
        )
        if backend == "sqlite":
            rows = store._query(
                f"EXPLAIN QUERY PLAN {sql}",
                params,
            )
            details = "\n".join(str(row["detail"]) for row in rows)
        else:
            store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
            rows = store.conn.execute(  # type: ignore[attr-defined]
                f"EXPLAIN {sql}",
                params,
            )
            details = "\n".join(str(row["QUERY PLAN"]) for row in rows)
        assert "idx_usage_reservations_recovery" in details
        normalized = details.lower().replace('"', "").replace(" ", "")
        assert "(created_at,reservation_id)>" in normalized


@pytest.mark.parametrize(
    "field_name",
    (
        "resource_usage_reservation_recovery_page_size",
        "resource_usage_reservation_recovery_page_hard_limit",
    ),
)
@pytest.mark.parametrize("value", (True, False))
def test_resource_recovery_config_rejects_bool(field_name: str, value: bool) -> None:
    with pytest.raises(PydanticValidationError):
        RuntimeDefaults(**{field_name: value})


def test_resource_recovery_config_requires_page_size_within_hard_limit() -> None:
    with pytest.raises(ValueError, match="resource_usage_reservation_recovery"):
        RuntimeDefaults(
            resource_usage_reservation_recovery_page_size=3,
            resource_usage_reservation_recovery_page_hard_limit=2,
        )


def test_resource_recovery_config_loads_from_yaml_and_models_are_exported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "runtime:",
                "  resource_usage_reservation_recovery_page_size: 17",
                "  resource_usage_reservation_recovery_page_hard_limit: 19",
            )
        ),
        encoding="utf-8",
    )

    config = load_config_file(path)

    assert config.runtime.resource_usage_reservation_recovery_page_size == 17
    assert config.runtime.resource_usage_reservation_recovery_page_hard_limit == 19
    assert ResourceUsageReservationCursor.__module__ == "agent_libos.models.process"
    assert ResourceUsageReservationRecoverySummary.__module__ == (
        "agent_libos.models.process"
    )


def _resource_manager(
    unit: UnitOfWork,
    config: AgentLibOSConfig,
) -> ResourceManager:
    return ResourceManager(
        unit,
        AuditManager(unit.evidence),
        EventBus(unit.evidence),
        require_recovery_lease=lambda: None,
        config=config,
    )


def _insert_reservation(
    unit: UnitOfWork,
    reservation_id: str,
    *,
    pid: str = "pid-reservation",
    reserved_by: str | None = None,
    usage: ResourceUsage | None = None,
    reason: str = "test",
    created_at: str = "2026-01-01T00:00:00Z",
) -> None:
    unit.resources.insert_resource_usage_reservation(
        reservation_id=reservation_id,
        pid=pid,
        usage=usage or ResourceUsage(external_write_bytes=1),
        reserved_by=reserved_by or f"missing-effect-{reservation_id}",
        reason=reason,
        created_at=created_at,
    )


def _process(
    pid: str,
    *,
    max_external_write_bytes: int,
    max_llm_calls: int | None = None,
    max_llm_total_tokens: int | None = None,
) -> AgentProcess:
    created_at = "2026-01-01T00:00:00Z"
    return AgentProcess(
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
        resource_budget=ResourceBudget(
            max_external_write_bytes=max_external_write_bytes,
            max_llm_calls=max_llm_calls,
            max_llm_total_tokens=max_llm_total_tokens,
        ),
        resource_usage=ResourceUsage(),
        created_at=created_at,
        updated_at=created_at,
    )


def _dispatched_effect(effect_id: str, *, pid: str) -> ExternalEffectRecord:
    created_at = "2026-01-01T00:00:00Z"
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider="test",
        operation="write",
        target=None,
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=True,
        information_flow=True,
        provider_metadata={},
        created_at=created_at,
        effect_state="pending",
        transaction_state="dispatched",
        provider_receipt={},
        updated_at=created_at,
    )


@contextlib.contextmanager
def _resource_store(
    backend: str,
    *,
    page_size: int = 3,
    hard_limit: int = 5_000,
) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                resource_usage_reservation_recovery_page_size=page_size,
                resource_usage_reservation_recovery_page_hard_limit=hard_limit,
            )
        )
        store = SQLiteStore(":memory:", config=config)
        try:
            yield store
        finally:
            store.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown store backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend="postgres",
                store_dsn=dsn,
                resource_usage_reservation_recovery_page_size=page_size,
                resource_usage_reservation_recovery_page_hard_limit=hard_limit,
            )
        )
        store = PostgresStore(dsn, config=config)
        try:
            yield store
        finally:
            store.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_resource_recovery_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
