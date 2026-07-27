from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from agent_libos.models import (
    Checkpoint,
    ObjectTaskNotification,
    ObjectTaskOwnerWatch,
    ObjectTaskStatus,
)
from agent_libos.runtime.checkpoint_manager import CheckpointManager
from agent_libos.storage import SQLRuntimeStore, SQLiteStore, UnitOfWork
from agent_libos.storage.postgres import _PostgresDialect
from agent_libos.utils.serde import dumps


def _seed_changed_effects(
    store: SQLiteStore,
    *,
    count: int,
    pid: str,
    sequence_start: int = 1,
    effect_prefix: str = "effect",
) -> int:
    effect_rows: list[tuple[Any, ...]] = []
    transition_rows: list[tuple[Any, ...]] = []
    for index in range(count):
        effect_id = f"{effect_prefix}-{index:05d}"
        created_at = f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z"
        effect_rows.append(
            (
                effect_id,
                None,
                None,
                pid,
                "test-provider",
                "read",
                None,
                "no_rollback_required",
                "not_required",
                0,
                0,
                "{}",
                created_at,
                "finalized",
                "committed",
                None,
                f"idempotency-{effect_prefix}-{index:05d}",
                "{}",
                created_at,
                1,
                "full",
                None,
            )
        )
        transition_rows.append(
            (
                sequence_start + index,
                effect_id,
                "finalized",
                "committed",
                created_at,
            )
        )
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO external_effects (
                effect_id, record_id, event_id, pid, provider, operation, target,
                rollback_class, rollback_status, state_mutation, information_flow,
                provider_metadata_json, created_at, effect_state, transaction_state,
                canonical_args_hash, idempotency_key, provider_receipt_json, updated_at,
                payload_retention_schema_version, payload_retention_tier,
                payload_retention_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            effect_rows,
        )
        cursor.executemany(
            """
            INSERT INTO external_effect_transitions (
                seq, effect_id, effect_state, transaction_state, occurred_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            transition_rows,
        )
        last_sequence = sequence_start + count - 1
        cursor.execute(
            "UPDATE runtime_counters SET value = ? "
            "WHERE counter_name = 'external_effect_ledger'",
            (last_sequence,),
        )
    return last_sequence


def test_changed_effect_read_pages_more_than_historical_sqlite_bind_limit() -> None:
    store = SQLiteStore(":memory:")
    try:
        last_sequence = _seed_changed_effects(
            store,
            count=1_005,
            pid="pid-target",
        )
        with store.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO external_effect_transitions (
                    seq, effect_id, effect_state, transaction_state, occurred_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    last_sequence + 1,
                    "effect-00000",
                    "finalized",
                    "committed",
                    "2026-01-01T01:00:00Z",
                ),
            )
            cursor.execute(
                "UPDATE runtime_counters SET value = ? "
                "WHERE counter_name = 'external_effect_ledger'",
                (last_sequence + 1,),
            )

        traced: list[str] = []
        store.conn.set_trace_callback(traced.append)
        records = store.list_external_effects_changed_after(0)
        store.conn.set_trace_callback(None)

        assert [record.effect_id for record in records] == [
            f"effect-{index:05d}" for index in range(1_005)
        ]
        assert [
            record.effect_id
            for record in store.list_external_effects_changed_after(last_sequence)
        ] == ["effect-00000"]
        assert store.list_external_effects_changed_after(last_sequence + 1) == []
        changed_queries = [
            statement
            for statement in traced
            if "FROM external_effects AS effect" in statement
        ]
        assert len(changed_queries) >= 3
        assert all("EXISTS" in statement for statement in changed_queries)
        assert all("effect_id IN" not in statement for statement in changed_queries)
    finally:
        store.close()


def test_changed_effect_pid_filter_batches_and_checkpoint_read_ignore_non_targets() -> None:
    store = SQLiteStore(":memory:")
    try:
        last_non_target = _seed_changed_effects(
            store,
            count=1_100,
            pid="pid-non-target",
            effect_prefix="unrelated",
        )
        _seed_changed_effects(
            store,
            count=3,
            pid="pid-target",
            sequence_start=last_non_target + 1,
            effect_prefix="target",
        )
        pid_filters = [f"pid-missing-{index:05d}" for index in range(1_100)]
        pid_filters.extend(("pid-target", "pid-target"))

        records = store.list_external_effects_changed_after(0, pids=pid_filters)
        assert [record.effect_id for record in records] == [
            "target-00000",
            "target-00001",
            "target-00002",
        ]

        manager = object.__new__(CheckpointManager)
        manager._evidence = UnitOfWork(store).evidence
        manager._external_effect_pids = lambda *_args, **_kwargs: pid_filters
        checkpoint = Checkpoint(
            checkpoint_id="checkpoint-old",
            pid="pid-target",
            reason="test",
            created_at="2026-01-01T00:00:00Z",
            effect_ledger_seq=0,
        )
        checkpoint_records = manager._external_effect_records_since(checkpoint)
        assert [record.effect_id for record in checkpoint_records] == [
            "target-00000",
            "target-00001",
            "target-00002",
        ]
    finally:
        store.close()


class _EmptyRecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: Any = ()) -> list[Any]:
        prepared = _PostgresDialect().prepare(sql, with_params=True)
        self.calls.append((prepared, tuple(params)))
        return []


def test_changed_effect_query_is_postgres_portable_and_never_exceeds_bind_batch() -> None:
    cursor = _EmptyRecordingCursor()
    store = object.__new__(SQLRuntimeStore)

    @contextmanager
    def transaction() -> Iterator[_EmptyRecordingCursor]:
        yield cursor

    store.transaction = transaction
    store._row_to_external_effect = lambda row: row

    assert store.list_external_effects_changed_after(
        0,
        pids=(f"pid-{index:05d}" for index in range(1_001)),
    ) == []
    assert len(cursor.calls) == 3
    for prepared, params in cursor.calls:
        assert "EXISTS" in prepared
        assert "effect_change.effect_id = effect.effect_id" in prepared
        assert "?" not in prepared
        assert "%s" in prepared
        assert len(params) <= 502


def _seed_object_tasks(
    store: SQLiteStore,
    *,
    count: int,
    task_prefix: str,
    creator_pid: str,
    owner_oid: str,
    status: ObjectTaskStatus,
    created_at: str,
    completed_at: str,
    wait: dict[str, Any] | None = None,
) -> None:
    rows = [
        (
            f"{task_prefix}-{index:05d}",
            owner_oid,
            creator_pid,
            None,
            "test.tool",
            None,
            status.value,
            "none",
            None,
            dumps(ObjectTaskNotification()),
            dumps(ObjectTaskOwnerWatch()),
            None,
            None,
            dumps(wait or {}),
            f"{created_at}-{index:05d}",
            completed_at,
            None,
            completed_at,
        )
        for index in range(count)
    ]
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO object_tasks (
                task_id, owner_oid, creator_pid, runner_pid, tool, tool_id, status,
                notification_status, notification_recipient_pid, notification_json,
                owner_watch_json, result_oid, error, wait_json, created_at,
                updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_checkpoint_task_supersede_filters_large_terminal_history_in_sql() -> None:
    store = SQLiteStore(":memory:")
    try:
        _seed_object_tasks(
            store,
            count=1_005,
            task_prefix="unrelated",
            creator_pid="pid-unrelated",
            owner_oid="oid-unrelated",
            status=ObjectTaskStatus.SUCCEEDED,
            created_at="2025-01-01T00:00:00Z",
            completed_at="2026-01-02T00:00:00Z",
        )
        _seed_object_tasks(
            store,
            count=1,
            task_prefix="target",
            creator_pid="pid-target",
            owner_oid="oid-target",
            status=ObjectTaskStatus.SUCCEEDED,
            created_at="2025-01-01T00:00:00Z",
            completed_at="2026-01-02T00:00:00Z",
        )
        decoded: list[str] = []
        decode = store._row_to_object_task
        store._row_to_object_task = lambda row: (
            decoded.append(str(row["task_id"])),
            decode(row),
        )[1]
        traced: list[str] = []
        store.conn.set_trace_callback(traced.append)

        checkpoint = Checkpoint(
            checkpoint_id="checkpoint-task-scope",
            pid="pid-target",
            reason="test",
            created_at="2026-01-01T00:00:00Z",
        )
        superseded = UnitOfWork(store).snapshots.supersede_object_tasks_after_checkpoint(
            ("pid-target",),
            ("oid-target",),
            checkpoint,
        )
        store.conn.set_trace_callback(None)

        assert superseded == ["target-00000"]
        assert decoded == ["target-00000"]
        assert store.get_object_task("target-00000").status == (
            ObjectTaskStatus.SUPERSEDED_BY_RESTORE
        )
        scoped_queries = [
            statement
            for statement in traced
            if statement.startswith("SELECT * FROM object_tasks WHERE")
        ]
        assert scoped_queries
        assert all("COALESCE(completed_at, updated_at)" in query for query in scoped_queries)
        assert all("LIMIT" in query for query in scoped_queries)
        assert all(
            "creator_pid IN" in query or "owner_oid IN" in query
            for query in scoped_queries
        )
    finally:
        store.close()


def test_checkpoint_task_result_reconcile_filters_scope_and_boundary_in_sql() -> None:
    store = SQLiteStore(":memory:")
    try:
        wait = {
            "previous_status": ObjectTaskStatus.SUCCEEDED.value,
            "previous_result_oid": "oid-result",
            "previous_error": None,
        }
        _seed_object_tasks(
            store,
            count=1_005,
            task_prefix="unrelated-result",
            creator_pid="pid-unrelated",
            owner_oid="oid-unrelated",
            status=ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN,
            created_at="2025-01-01T00:00:00Z",
            completed_at="2025-12-31T00:00:00Z",
            wait=wait,
        )
        _seed_object_tasks(
            store,
            count=1,
            task_prefix="target-result",
            creator_pid="pid-target",
            owner_oid="oid-result",
            status=ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN,
            created_at="2025-01-01T00:00:00Z",
            completed_at="2025-12-31T00:00:00Z",
            wait=wait,
        )
        store.get_object = lambda oid: object() if oid == "oid-result" else None
        checkpoint = Checkpoint(
            checkpoint_id="checkpoint-result-scope",
            pid="pid-target",
            reason="test",
            created_at="2026-01-01T00:00:00Z",
        )
        snapshot = SimpleNamespace(
            subtree_pids=("pid-target",),
            owned_object_oids=("oid-result",),
        )

        restored = UnitOfWork(store).snapshots.reconcile_restored_object_task_results(
            snapshot,
            checkpoint,
        )

        assert restored == ["target-result-00000"]
        task = store.get_object_task("target-result-00000")
        assert task.status == ObjectTaskStatus.SUCCEEDED
        assert task.result_oid == "oid-result"
        assert store.get_object_task("unrelated-result-00000").status == (
            ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
        )
    finally:
        store.close()
