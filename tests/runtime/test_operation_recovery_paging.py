from __future__ import annotations

import contextlib
import math
import os
import tracemalloc
from collections.abc import Iterator
from dataclasses import replace
from itertools import repeat
from threading import Event, Thread
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    OperationCursor,
    OperationEvidenceLink,
    OperationEvidenceRole,
    OperationKind,
    OperationOutcome,
    OperationPage,
    OperationRecord,
    OperationState,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.operation_manager import OperationManager
from agent_libos.storage import PostgresStore, SQLiteStore


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_stale_operation_recovery_is_bounded_stable_and_exact(backend: str) -> None:
    with _operation_store(backend, hard_limit=3) as store:
        records = [
            _operation(f"operation-{index}", started_at="2026-01-01T00:00:00Z")
            for index in range(7)
        ]
        for record in records:
            store.insert_operation(record)
        store.insert_operation(
            _operation(
                "terminal-newer",
                state=OperationState.TERMINAL,
                outcome=OperationOutcome.SUCCEEDED,
                started_at="2026-01-02T00:00:00Z",
            )
        )

        first = store.scan_stale_running_operations(after=None, limit=3)
        assert [record.operation_id for record in first.records] == [
            "operation-6",
            "operation-5",
            "operation-4",
        ]
        assert first.next_cursor == OperationCursor(
            "2026-01-01T00:00:00Z",
            "operation-4",
        )

        # Updating the already-returned rows cannot invalidate a typed keyset
        # cursor or cause the following page to repeat/skip an older row.
        for record in first.records:
            assert store.update_operation(
                replace(
                    record,
                    state=OperationState.TERMINAL,
                    outcome=OperationOutcome.INTERRUPTED,
                    completed_at="2026-01-03T00:00:00Z",
                ),
                expected_states=(OperationState.RUNNING.value,),
            )
        second = store.scan_stale_running_operations(
            after=first.next_cursor,
            limit=3,
        )
        assert [record.operation_id for record in second.records] == [
            "operation-3",
            "operation-2",
            "operation-1",
        ]
        assert second.next_cursor == OperationCursor(
            "2026-01-01T00:00:00Z",
            "operation-1",
        )
        third = store.scan_stale_running_operations(
            after=second.next_cursor,
            limit=3,
        )
        assert [record.operation_id for record in third.records] == ["operation-0"]
        assert third.next_cursor is None

        for invalid in (0, -1, True, 4):
            with pytest.raises(ValidationError):
                store.scan_stale_running_operations(after=None, limit=invalid)
        with pytest.raises(ValidationError, match="cursor"):
            store.scan_stale_running_operations(after=object(), limit=1)  # type: ignore[arg-type]
        with store.stale_operation_recovery_index():
            with pytest.raises(ValidationError, match="hard cap"):
                store.operation_ids_with_unknown_external_effects(
                    [f"operation-{index}" for index in range(4)]
                )
            with pytest.raises(ValidationError, match="hard cap"):
                store.operation_ids_with_unknown_external_effects(
                    repeat("operation-0")
                )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_unknown_external_effect_query_preserves_descendant_semantics(
    backend: str,
) -> None:
    with _operation_store(backend) as store:
        root = _operation("root")
        child = _operation("child", root="root", parent="root")
        grandchild = _operation("grandchild", root="root", parent="child")
        sibling = _operation("sibling", root="root", parent="root")
        other_root = _operation("other-root")
        finalized = _operation("finalized", root="other-root", parent="other-root")
        for record in (root, child, grandchild, sibling, other_root, finalized):
            store.insert_operation(record)

        _link_effect(
            store,
            grandchild.operation_id,
            _effect("pending-effect", effect_state="pending", transaction_state="prepared"),
        )
        _link_effect(
            store,
            other_root.operation_id,
            _effect(
                "unknown-effect",
                effect_state="finalized",
                transaction_state="unknown",
            ),
        )
        _link_effect(
            store,
            finalized.operation_id,
            _effect(
                "committed-effect",
                effect_state="finalized",
                transaction_state="committed",
            ),
        )

        assert store.operation_has_unknown_external_effect(root.operation_id)
        assert store.operation_has_unknown_external_effect(child.operation_id)
        assert store.operation_has_unknown_external_effect(grandchild.operation_id)
        assert not store.operation_has_unknown_external_effect(sibling.operation_id)
        assert store.operation_has_unknown_external_effect(other_root.operation_id)
        assert not store.operation_has_unknown_external_effect(finalized.operation_id)
        assert not store.operation_has_unknown_external_effect("missing-operation")
        with pytest.raises(ValidationError, match="non-empty text"):
            store.operation_has_unknown_external_effect("")

        with store.stale_operation_recovery_index():
            assert store.operation_ids_with_unknown_external_effects(
                (root.operation_id, child.operation_id, grandchild.operation_id)
            ) == {root.operation_id, child.operation_id, grandchild.operation_id}
            assert store.operation_ids_with_unknown_external_effects(
                (sibling.operation_id, other_root.operation_id, finalized.operation_id)
            ) == {other_root.operation_id}
            assert store.operation_ids_with_unknown_external_effects(
                ("missing-operation",)
            ) == set()
            with pytest.raises(ValidationError, match="non-empty text"):
                store.operation_ids_with_unknown_external_effects(("",))


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_stale_operation_recovery_temp_index_is_reentrant_and_commit_stable(
    backend: str,
) -> None:
    with _operation_store(backend) as store:
        operation = _operation("temp-reentrant")
        store.insert_operation(operation)
        _link_effect(
            store,
            operation.operation_id,
            _effect(
                "temp-reentrant-effect",
                effect_state="pending",
                transaction_state="prepared",
            ),
        )

        with pytest.raises(ValidationError, match="not active"):
            store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            )
        with store.stale_operation_recovery_index():
            assert store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            ) == {operation.operation_id}
            with store.stale_operation_recovery_index():
                assert store.operation_ids_with_unknown_external_effects(
                    (operation.operation_id,)
                ) == {operation.operation_id}
            assert store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            ) == {operation.operation_id}

            # Per-row lifecycle commits do not drop PostgreSQL's temp relation
            # or SQLite's TEMP table while the outer recovery session is live.
            assert store.update_operation(
                replace(
                    operation,
                    state=OperationState.TERMINAL,
                    outcome=OperationOutcome.UNKNOWN,
                    completed_at="2026-01-02T00:00:00Z",
                ),
                expected_states=(OperationState.RUNNING.value,),
            )
            assert store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            ) == {operation.operation_id}

        with store.stale_operation_recovery_index():
            assert store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            ) == set()


def test_stale_operation_recovery_temp_index_cleanup_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        operation = _operation("temp-cleanup")
        store.insert_operation(operation)
        _link_effect(
            store,
            operation.operation_id,
            _effect(
                "temp-cleanup-effect",
                effect_state="pending",
                transaction_state="prepared",
            ),
        )
        original_execute = store._execute
        fail_setup = True

        def injected_setup_failure(sql: str, params: Any = ()) -> Any:
            nonlocal fail_setup
            if fail_setup and "uncertain_effects(effect_id) AS MATERIALIZED" in sql:
                fail_setup = False
                raise RuntimeError("injected temp setup failure")
            return original_execute(sql, params)

        monkeypatch.setattr(store, "_execute", injected_setup_failure)
        with pytest.raises(RuntimeError, match="setup failure"):
            with store.stale_operation_recovery_index():
                pytest.fail("failed setup must not yield")

        # The failed setup dropped its partial table, so a clean reentry can
        # recreate and populate the exact snapshot.
        with store.stale_operation_recovery_index():
            assert store.operation_ids_with_unknown_external_effects(
                (operation.operation_id,)
            ) == {operation.operation_id}

        manager = OperationManager(
            store,
            recovery_page_size=1,
            require_recovery_lease=lambda: None,
        )
        original_finish = manager.finish
        fail_finish = True

        def injected_finish(*args: Any, **kwargs: Any) -> Any:
            nonlocal fail_finish
            if fail_finish:
                fail_finish = False
                raise RuntimeError("injected finish failure")
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(manager, "finish", injected_finish)
        with pytest.raises(RuntimeError, match="finish failure"):
            manager.interrupt_stale_running()
        summary = manager.interrupt_stale_running()
        assert summary.total_count == 1
        assert summary.sample_operation_ids == (operation.operation_id,)
    finally:
        store.close()


def test_stale_operation_recovery_cleanup_failure_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        original_execute = store._execute

        def fail_cleanup(sql: str, params: Any = ()) -> Any:
            if sql.startswith("DROP TABLE IF EXISTS"):
                raise RuntimeError("injected cleanup failure")
            return original_execute(sql, params)

        monkeypatch.setattr(store, "_execute", fail_cleanup)
        with pytest.raises(LookupError, match="primary body failure") as exc_info:
            with store.stale_operation_recovery_index():
                raise LookupError("primary body failure")
        assert any("cleanup also failed" in note for note in exc_info.value.__notes__)
        with pytest.raises(ValidationError, match="unusable"):
            store.get_operation("anything")
    finally:
        store.close()


def test_stale_operation_recovery_setup_cleanup_failure_preserves_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        original_execute = store._execute

        def fail_setup_and_cleanup(sql: str, params: Any = ()) -> Any:
            if "uncertain_effects(effect_id) AS MATERIALIZED" in sql:
                raise LookupError("primary setup failure")
            if sql.startswith("DROP TABLE IF EXISTS"):
                raise RuntimeError("secondary cleanup failure")
            return original_execute(sql, params)

        monkeypatch.setattr(store, "_execute", fail_setup_and_cleanup)
        with pytest.raises(LookupError, match="primary setup failure") as exc_info:
            with store.stale_operation_recovery_index():
                pytest.fail("failed setup must not yield")
        assert any("cleanup also failed" in note for note in exc_info.value.__notes__)
        with pytest.raises(ValidationError, match="unusable"):
            store.get_operation("anything")
    finally:
        store.close()


def test_stale_operation_recovery_index_holds_store_lock_across_pages() -> None:
    store = SQLiteStore(":memory:")
    inserted = Event()

    def concurrent_insert() -> None:
        store.insert_operation(_operation("concurrent-operation"))
        inserted.set()

    thread = Thread(target=concurrent_insert)
    try:
        with store.stale_operation_recovery_index():
            thread.start()
            assert not inserted.wait(0.05)
        assert inserted.wait(1)
        thread.join(timeout=1)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            thread.join(timeout=1)
        store.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_operation_recovery_queries_have_backend_indexes(backend: str) -> None:
    with _operation_store(backend) as store:
        root = _operation("explain-root")
        child = _operation("explain-child", root=root.operation_id, parent=root.operation_id)
        for record in (root, child):
            store.insert_operation(record)
        _link_effect(
            store,
            child.operation_id,
            _effect("explain-effect", effect_state="pending", transaction_state="prepared"),
        )
        if backend == "postgres":
            _seed_postgres_operation_plan_history(store)

        captured: dict[str, tuple[str, tuple[Any, ...]]] = {}
        original_query = store._query
        original_execute = store._execute

        def capture(sql: str, params: Any = ()) -> list[Any]:
            if "SELECT * FROM operations" in sql:
                captured["stale_page"] = (sql, tuple(params))
            elif "FROM agent_libos_stale_operation_recovery_unknown" in sql:
                captured["membership"] = (sql, tuple(params))
            elif "selected(root_operation_id, operation_id)" in sql:
                captured["online_exists"] = (sql, tuple(params))
            return original_query(sql, params)

        def capture_execute(sql: str, params: Any = ()) -> Any:
            if "uncertain_effects(effect_id) AS MATERIALIZED" in sql:
                captured["temp_build"] = (sql, tuple(params))
            return original_execute(sql, params)

        store._query = capture  # type: ignore[method-assign]
        store._execute = capture_execute  # type: ignore[method-assign]
        try:
            with store.stale_operation_recovery_index():
                store.scan_stale_running_operations(
                    after=OperationCursor(
                        "9999-12-31T23:59:59Z",
                        "operation-deep-cursor",
                    ),
                    limit=2,
                )
                store.operation_ids_with_unknown_external_effects(
                    (root.operation_id,)
                )
                store.operation_has_unknown_external_effect(root.operation_id)
                store._query = original_query  # type: ignore[method-assign]
                store._execute = original_execute  # type: ignore[method-assign]
                plans = {
                    label: _explain(store, backend, sql, params)
                    for label, (sql, params) in captured.items()
                }
                forced_temp_plan = (
                    _explain(
                        store,
                        backend,
                        *captured["temp_build"],
                        force_parameterized=True,
                    )
                    if backend == "postgres"
                    else ""
                )
        finally:
            store._query = original_query  # type: ignore[method-assign]
            store._execute = original_execute  # type: ignore[method-assign]

        assert set(plans) == {
            "stale_page",
            "temp_build",
            "membership",
            "online_exists",
        }
        assert "idx_operations_state_started" in plans["stale_page"]
        normalized_stale_plan = (
            plans["stale_page"].lower().replace('"', "").replace(" ", "")
        )
        assert "(started_at,operation_id)<" in normalized_stale_plan
        if backend == "postgres":
            assert "idx_external_effects_recovery_transaction" in plans["temp_build"]
        else:
            # Recent SQLite planners may split the two predicates across the
            # narrower state indexes instead of choosing the composite index.
            assert any(
                index_name in plans["temp_build"]
                for index_name in (
                    "idx_external_effects_recovery_transaction",
                    "idx_external_effects_recovery_state",
                )
            )
        assert "idx_external_effects_transaction_state" in plans["temp_build"]
        assert "idx_operation_evidence_lookup" in plans["temp_build"]
        assert "idx_operations_parent_root" in plans["online_exists"]
        assert "idx_operation_evidence_operation_type" in plans["online_exists"]
        assert "stale_operation_recovery_unknown" in plans["membership"]
        if backend == "postgres":
            temp_conditions = _postgres_index_conditions(plans["temp_build"])
            assert "evidence_id" in temp_conditions, plans["temp_build"]
            assert "operation_id" in temp_conditions
            forced_temp_conditions = _postgres_index_conditions(forced_temp_plan)
            assert "evidence_id" in forced_temp_conditions
            assert "operation_id" in forced_temp_conditions
            assert "parent_operation_id" in forced_temp_conditions
            online_conditions = _postgres_index_conditions(plans["online_exists"])
            assert "evidence_id" in online_conditions
            assert "operation_id" in online_conditions
            assert "parent_operation_id" in online_conditions
            assert "root_operation_id" in online_conditions

        index_names = _index_names(store, backend)
        assert {
            "idx_operations_state_started",
            "idx_operations_parent_root",
            "idx_operation_evidence_operation_type",
            "idx_external_effects_transaction_state",
        } <= index_names


def test_operation_manager_recovery_queries_are_page_linear_at_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_size = 64
    total = 2_049
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            operation_recovery_page_size=page_size,
            operation_recovery_page_hard_limit=page_size,
        )
    )
    store = SQLiteStore(":memory:", config=config)
    try:
        root_id = "scale-00000"
        for index in range(total):
            operation = _operation(
                f"scale-{index:05d}",
                root=root_id,
                parent=f"scale-{index - 1:05d}" if index else None,
            )
            store.insert_operation(operation)
        _link_effect(
            store,
            f"scale-{total - 1:05d}",
            _effect(
                "scale-effect-deepest",
                effect_state="pending",
                transaction_state="prepared",
            ),
        )

        batch_sizes: list[int] = []
        scan_calls = 0
        materialization_calls = 0
        original_batch_query = store.operation_ids_with_unknown_external_effects
        original_scan = store.scan_stale_running_operations
        original_execute = store._execute

        def count_batch(operation_ids: Any) -> set[str]:
            selected = tuple(operation_ids)
            batch_sizes.append(len(selected))
            return original_batch_query(selected)

        def count_scan(**kwargs: Any) -> Any:
            nonlocal scan_calls
            scan_calls += 1
            return original_scan(**kwargs)

        def count_materialization(sql: str, params: Any = ()) -> Any:
            nonlocal materialization_calls
            if "uncertain_effects(effect_id) AS MATERIALIZED" in sql:
                materialization_calls += 1
            return original_execute(sql, params)

        monkeypatch.setattr(
            store,
            "operation_ids_with_unknown_external_effects",
            count_batch,
        )
        monkeypatch.setattr(
            store,
            "scan_stale_running_operations",
            count_scan,
        )
        monkeypatch.setattr(
            store,
            "_execute",
            count_materialization,
        )

        recovered = OperationManager(
            store,
            recovery_page_size=page_size,
            require_recovery_lease=lambda: None,
        ).interrupt_stale_running()

        assert len(recovered) == total
        assert recovered.total_count == total
        assert len(recovered.sample_operation_ids) == page_size
        assert recovered.truncated
        assert materialization_calls == 1
        assert scan_calls == math.ceil(total / page_size)
        assert len(batch_sizes) == math.ceil(total / page_size)
        assert max(batch_sizes) == page_size
        assert sum(batch_sizes) == total
        counts = store._query(
            "SELECT outcome, COUNT(*) AS count FROM operations GROUP BY outcome"
        )
        assert {str(row["outcome"]): int(row["count"]) for row in counts} == {
            OperationOutcome.UNKNOWN.value: total
        }
    finally:
        store.close()


def test_million_row_operation_recovery_keeps_python_buffers_page_bounded() -> None:
    total = 1_000_000
    page_size = 4_096

    class RecoveryRecord:
        __slots__ = ("operation_id", "started_at")

        def __init__(self, operation_id: str) -> None:
            self.operation_id = operation_id
            self.started_at = "2026-01-01T00:00:00Z"

    class FinishedRecord:
        __slots__ = ("operation_id", "outcome")

        def __init__(self) -> None:
            self.operation_id = ""
            self.outcome = OperationOutcome.UNKNOWN

    class MillionRowRecoveryStore:
        def __init__(self) -> None:
            self.page_calls = 0
            self.membership_calls = 0
            self.max_page_records = 0
            self.max_membership_ids = 0
            self.materializations = 0

        @contextlib.contextmanager
        def stale_operation_recovery_index(self) -> Iterator[None]:
            self.materializations += 1
            yield

        def scan_stale_running_operations(
            self,
            *,
            after: OperationCursor | None,
            limit: int,
        ) -> OperationPage:
            self.page_calls += 1
            high = (
                total - 1
                if after is None
                else int(after.operation_id.removeprefix("million-")) - 1
            )
            low = max(-1, high - limit)
            records = tuple(
                RecoveryRecord(f"million-{index:07d}")
                for index in range(high, low, -1)
            )
            self.max_page_records = max(self.max_page_records, len(records))
            next_cursor = None
            if low >= 0:
                last = records[-1]
                next_cursor = OperationCursor(last.started_at, last.operation_id)
            return OperationPage(records=records, next_cursor=next_cursor)  # type: ignore[arg-type]

        def operation_ids_with_unknown_external_effects(
            self,
            operation_ids: Any,
        ) -> set[str]:
            self.membership_calls += 1
            selected = tuple(operation_ids)
            self.max_membership_ids = max(
                self.max_membership_ids,
                len(selected),
            )
            return set(selected)

    store = MillionRowRecoveryStore()
    manager = OperationManager(  # type: ignore[arg-type]
        store,
        recovery_page_size=page_size,
        require_recovery_lease=lambda: None,
    )
    finished = FinishedRecord()

    def finish(
        outcome: OperationOutcome | str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FinishedRecord:
        del metadata
        assert operation_id is not None
        finished.operation_id = operation_id
        finished.outcome = OperationOutcome(outcome)
        return finished

    manager.finish = finish  # type: ignore[method-assign]
    tracemalloc.start()
    try:
        recovered = manager.interrupt_stale_running()
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    expected_pages = math.ceil(total / page_size)
    assert recovered.total_count == total
    assert len(recovered.sample_operation_ids) == page_size
    assert recovered.truncated
    assert store.materializations == 1
    assert store.page_calls == expected_pages
    assert store.membership_calls == expected_pages
    assert store.max_page_records <= page_size
    assert store.max_membership_ids <= page_size
    assert peak_bytes < 16 * 1024 * 1024


def _operation(
    operation_id: str,
    *,
    root: str | None = None,
    parent: str | None = None,
    state: OperationState = OperationState.RUNNING,
    outcome: OperationOutcome = OperationOutcome.PENDING,
    started_at: str = "2026-01-01T00:00:00Z",
) -> OperationRecord:
    selected_root = root or operation_id
    return OperationRecord(
        operation_id=operation_id,
        root_operation_id=selected_root,
        parent_operation_id=parent,
        kind=OperationKind.RUNTIME,
        name="test.operation",
        actor="test",
        pid="pid-operation-recovery",
        state=state,
        outcome=outcome,
        started_at=started_at,
        updated_at=started_at,
        completed_at=started_at if state == OperationState.TERMINAL else None,
    )


def _effect(
    effect_id: str,
    *,
    effect_state: str,
    transaction_state: str,
) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid="pid-operation-recovery",
        provider="test-provider",
        operation="write",
        target="resource:test",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=True,
        information_flow=False,
        provider_metadata={},
        provider_receipt={},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        effect_state=effect_state,
        transaction_state=transaction_state,
        canonical_args_hash="a" * 64,
        idempotency_key=f"idempotency-{effect_id}",
    )


def _link_effect(
    store: SQLiteStore | PostgresStore,
    operation_id: str,
    effect: ExternalEffectRecord,
) -> None:
    store.insert_external_effect(effect)
    assert store.insert_operation_evidence(
        OperationEvidenceLink(
            link_id=f"link-{effect.effect_id}",
            operation_id=operation_id,
            evidence_type="external_effect",
            evidence_id=effect.effect_id,
            role=OperationEvidenceRole.EFFECT.value,
            created_at=effect.created_at,
        )
    )


def _explain(
    store: SQLiteStore | PostgresStore,
    backend: str,
    sql: str,
    params: tuple[Any, ...],
    *,
    force_parameterized: bool = False,
) -> str:
    if backend == "sqlite":
        rows = store._query(f"EXPLAIN QUERY PLAN {sql}", params)
    else:
        with store.transaction() as cursor:
            cursor.execute("SET LOCAL enable_seqscan = off")
            if force_parameterized:
                cursor.execute("SET LOCAL enable_hashjoin = off")
                cursor.execute("SET LOCAL enable_mergejoin = off")
                cursor.execute("SET LOCAL enable_material = off")
            rows = store._query(f"EXPLAIN {sql}", params)
    return "\n".join(
        str(value)
        for row in rows
        for value in (row.values() if hasattr(row, "values") else tuple(row))
    )


def _postgres_index_conditions(plan: str) -> str:
    return "\n".join(
        line.casefold() for line in plan.splitlines() if "index cond:" in line.casefold()
    )


def _seed_postgres_operation_plan_history(
    store: SQLiteStore | PostgresStore,
    *,
    count: int = 512,
) -> None:
    """Give PostgreSQL realistic selectivity for parameterized plan gates."""

    now = "2025-01-01T00:00:00Z"
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO operations (
                operation_id, root_operation_id, parent_operation_id, kind,
                name, actor, pid, state, outcome, expected_roles_json,
                metadata_json, runtime_publication_id, started_at, updated_at,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"plan-history-operation-{index:04d}",
                    f"plan-history-operation-{index:04d}",
                    None,
                    "runtime",
                    "test.plan_history",
                    "test",
                    None,
                    "terminal",
                    "succeeded",
                    "[]",
                    "{}",
                    None,
                    now,
                    now,
                    now,
                )
                for index in range(count)
            ],
        )
        cursor.executemany(
            """
            INSERT INTO external_effects (
                effect_id, record_id, event_id, pid, provider, operation,
                target, rollback_class, rollback_status, state_mutation,
                information_flow, provider_metadata_json, created_at,
                effect_state, transaction_state, canonical_args_hash,
                idempotency_key, provider_receipt_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"plan-history-effect-{index:04d}",
                    None,
                    None,
                    "pid-plan-history",
                    "test-provider",
                    "read",
                    None,
                    "unknown",
                    "unknown",
                    0,
                    0,
                    "{}",
                    now,
                    "finalized",
                    "committed",
                    None,
                    None,
                    "{}",
                    now,
                )
                for index in range(count)
            ],
        )
        cursor.executemany(
            """
            INSERT INTO operation_evidence (
                link_id, operation_id, evidence_type, evidence_id, role,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"plan-history-link-{index:04d}",
                    f"plan-history-operation-{index:04d}",
                    "external_effect",
                    f"plan-history-effect-{index:04d}",
                    "effect",
                    now,
                    "{}",
                )
                for index in range(count)
            ],
        )
        cursor.execute("ANALYZE operations")
        cursor.execute("ANALYZE operation_evidence")
        cursor.execute("ANALYZE external_effects")


def _index_names(
    store: SQLiteStore | PostgresStore,
    backend: str,
) -> set[str]:
    if backend == "sqlite":
        rows = store._query("PRAGMA index_list(operations)") + store._query(
            "PRAGMA index_list(operation_evidence)"
        ) + store._query(
            "PRAGMA index_list(external_effects)"
        )
    else:
        rows = store._query(
            "SELECT indexname AS name FROM pg_indexes "
            "WHERE schemaname = current_schema() "
            "AND tablename IN ("
            "'operations', 'operation_evidence', 'external_effects')"
        )
    return {str(row["name"]) for row in rows}


@contextlib.contextmanager
def _operation_store(
    backend: str,
    *,
    hard_limit: int = 5_000,
) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                operation_recovery_page_size=min(500, hard_limit),
                operation_recovery_page_hard_limit=hard_limit,
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
                operation_recovery_page_size=min(500, hard_limit),
                operation_recovery_page_hard_limit=hard_limit,
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
    schema = f"agent_libos_operation_recovery_{uuid4().hex}"
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
