from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    AgentProcess,
    OperationOutcome,
    OperationState,
    ProcessCursor,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    RuntimePublicationCursor,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import PostgresStore, SQLiteStore, UnitOfWork, open_store
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps
from tests.support.runtime import close_runtime


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_runtime_publication_kind_is_validated_at_repository_and_backend_boundaries(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store_target(backend, tmp_path) as (target, config):
        store = open_store(target, config=config)
        try:
            unit = UnitOfWork(store)
            with pytest.raises(ValidationError, match="invalid runtime publication kind"):
                unit.publications.insert_runtime_publication(
                    publication_id="publication-invalid-repository",
                    kind="typo",  # type: ignore[arg-type]
                    pid="pid-invalid",
                    owner_instance_id="test-owner",
                    plan={},
                )
            with pytest.raises(ValidationError, match="invalid runtime publication kind"):
                store.insert_runtime_publication(
                    publication_id="publication-invalid-backend",
                    kind="typo",
                    pid="pid-invalid",
                    owner_instance_id="test-owner",
                    plan={},
                )
            assert store.list_runtime_publications() == []
        finally:
            store.close()


def test_sqlite_reopen_fails_closed_on_invalid_durable_publication_kind(
    tmp_path: Path,
) -> None:
    database = tmp_path / "invalid-publication-kind.sqlite"
    store = SQLiteStore(database)
    store.insert_runtime_publication(
        publication_id="publication-corrupt-kind",
        kind="process_launch",
        pid="pid-corrupt-kind",
        owner_instance_id="test-owner",
        plan={},
    )
    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT publication_id FROM runtime_publications "
        "INDEXED BY idx_runtime_publications_invalid_domain "
        "WHERE kind NOT IN ('process_launch', 'process_exec', 'checkpoint_restore') "
        "OR state NOT IN ('planning', 'applying', 'reconciliation_pending', "
        "'committed', 'rollback_pending', 'rolled_back', 'failed', 'manual') "
        "OR operation_reconciled NOT IN (0, 1) LIMIT 1"
    ).fetchall()
    assert "idx_runtime_publications_invalid_domain" in "\n".join(
        str(row["detail"]) for row in plan
    )
    store.close()

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE runtime_publications SET kind = ? WHERE publication_id = ?",
            ("typo", "publication-corrupt-kind"),
        )

    with pytest.raises(
        ValidationError,
        match="invalid durable runtime publication domain",
    ):
        SQLiteStore(database)


def test_binding_writes_atomically_invalidate_reconciliation_marker(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "operation-marker-invalidation.sqlite")
    try:
        pid = runtime.process.spawn(goal="operation marker invalidation")
        publications = [
            publication
            for publication in runtime.store.list_runtime_publications(pid=pid)
            if publication["kind"] == "process_launch"
        ]
        assert len(publications) == 1
        old_publication = publications[0]
        old_publication_id = str(old_publication["publication_id"])
        operation_id = str(old_publication["plan"]["operation_id"])
        operation = runtime.store.get_operation(operation_id)
        assert operation is not None
        assert old_publication["operation_reconciled"] is True

        plan_noop = {
            "boot_kind": old_publication["plan"]["boot_kind"],
            "materialized_workspace_root": old_publication["plan"][
                "materialized_workspace_root"
            ],
        }
        assert runtime.store.update_runtime_publication_plan(
            old_publication_id,
            plan_noop,
            expected_states={"committed"},
        )
        assert runtime.store.get_runtime_publication(old_publication_id)[
            "operation_reconciled"
        ] is False
        assert runtime.store.mark_runtime_publication_operation_reconciled(
            old_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=operation_id,
        )
        plan_snapshot = runtime.store.get_runtime_publication(old_publication_id)
        assert plan_snapshot is not None
        assert not runtime.store.update_runtime_publication_plan(
            old_publication_id,
            {**plan_noop, "reconciliation_contract_revision": 2},
            expected_states={"committed"},
        )
        assert (
            runtime.store.get_runtime_publication(old_publication_id)
            == plan_snapshot
        )

        inserted_operation_id = f"operation-marker-insert-{uuid4().hex}"
        inserted_publication_id = f"publication-marker-insert-{uuid4().hex}"
        inserted_plan = dict(plan_snapshot["plan"])
        inserted_plan.pop("operation_id", None)
        inserted_plan.pop("operation_binding_version", None)
        inserted_plan.update(
            {
                "artifact_owner": f"publication:{inserted_publication_id}",
            }
        )
        runtime.store.insert_runtime_publication(
            publication_id=inserted_publication_id,
            kind="process_launch",
            pid=pid,
            owner_instance_id="test-owner",
            plan=inserted_plan,
        )
        assert runtime.store.advance_runtime_publication(
            inserted_publication_id,
            state="committed",
            phase="committed",
            expected_states={"planning"},
        )
        assert runtime.store.mark_runtime_publication_operation_reconciled(
            inserted_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=None,
        )
        inserted_operation = replace(
            operation,
            operation_id=inserted_operation_id,
            root_operation_id=inserted_operation_id,
            parent_operation_id=None,
            metadata={
                **operation.metadata,
                "runtime_publication_id": inserted_publication_id,
            },
            started_at=utc_now(),
            updated_at=utc_now(),
        )
        runtime.store.insert_operation(inserted_operation)
        assert runtime.store.get_runtime_publication(inserted_publication_id)[
            "operation_reconciled"
        ] is False
        assert runtime.store.list_operation_ids_by_runtime_publication_id(
            inserted_publication_id
        ) == [inserted_operation_id]
        assert not runtime.store.mark_runtime_publication_operation_reconciled(
            inserted_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=None,
        )

        new_publication_id = f"publication-marker-move-{uuid4().hex}"
        new_plan = dict(plan_snapshot["plan"])
        new_plan.pop("operation_id", None)
        new_plan.pop("operation_binding_version", None)
        new_plan["artifact_owner"] = f"publication:{new_publication_id}"
        runtime.store.insert_runtime_publication(
            publication_id=new_publication_id,
            kind="process_launch",
            pid=pid,
            owner_instance_id="test-owner",
            plan=new_plan,
        )
        assert runtime.store.advance_runtime_publication(
            new_publication_id,
            state="committed",
            phase="committed",
            expected_states={"planning"},
        )
        assert runtime.store.mark_runtime_publication_operation_reconciled(
            new_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=None,
        )

        moved = replace(
            operation,
            metadata={
                **operation.metadata,
                "runtime_publication_id": new_publication_id,
            },
            updated_at=utc_now(),
        )
        assert not runtime.store.update_operation(
            moved,
            expected_states={OperationState.RUNNING.value},
        )
        assert runtime.store.get_runtime_publication(old_publication_id)[
            "operation_reconciled"
        ] is True
        assert runtime.store.get_runtime_publication(new_publication_id)[
            "operation_reconciled"
        ] is True

        assert runtime.store.update_operation(
            moved,
            expected_states={operation.state.value},
        )
        assert runtime.store.get_runtime_publication(old_publication_id)[
            "operation_reconciled"
        ] is False
        assert runtime.store.get_runtime_publication(new_publication_id)[
            "operation_reconciled"
        ] is False
        assert not runtime.store.mark_runtime_publication_operation_reconciled(
            old_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=operation_id,
        )
        assert not runtime.store.mark_runtime_publication_operation_reconciled(
            new_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=None,
        )

        rebound = replace(
            moved,
            metadata={
                **moved.metadata,
                "runtime_publication_id": old_publication_id,
            },
            updated_at=utc_now(),
        )
        assert runtime.store.update_operation(
            rebound,
            expected_states={moved.state.value},
        )
        assert runtime.store.mark_runtime_publication_operation_reconciled(
            old_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=operation_id,
        )
        assert runtime.store.mark_runtime_publication_operation_reconciled(
            new_publication_id,
            expected_kind="process_launch",
            expected_state="committed",
            expected_phase="committed",
            expected_operation_id=None,
        )
        rebound_snapshot = runtime.store.get_operation(operation_id)
        old_snapshot = runtime.store.get_runtime_publication(old_publication_id)
        new_snapshot = runtime.store.get_runtime_publication(new_publication_id)
        assert rebound_snapshot is not None
        assert old_snapshot is not None
        assert new_snapshot is not None

        nested_move = replace(
            rebound_snapshot,
            metadata={
                **rebound_snapshot.metadata,
                "runtime_publication_id": new_publication_id,
            },
            updated_at=utc_now(),
        )
        with pytest.raises(RuntimeError, match="roll back outer transaction"):
            with runtime.store.transaction():
                assert runtime.store.update_operation(
                    nested_move,
                    expected_states={rebound_snapshot.state.value},
                )
                assert runtime.store.get_runtime_publication(old_publication_id)[
                    "operation_reconciled"
                ] is False
                assert runtime.store.get_runtime_publication(new_publication_id)[
                    "operation_reconciled"
                ] is False
                raise RuntimeError("roll back outer transaction")

        assert runtime.store.get_operation(operation_id) == rebound_snapshot
        assert runtime.store.get_runtime_publication(old_publication_id) == old_snapshot
        assert runtime.store.get_runtime_publication(new_publication_id) == new_snapshot
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_publication_reconciliation_repository_is_paged_cas_and_indexed(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store_target(backend, tmp_path) as (target, config):
        store = open_store(target, config=config)
        try:
            for index in range(3):
                publication_id = f"publication-page-{index}"
                store.insert_runtime_publication(
                    publication_id=publication_id,
                    kind="process_launch",
                    pid=f"pid-page-{index}",
                    owner_instance_id="test-owner",
                    plan={},
                )
                assert store.advance_runtime_publication(
                    publication_id,
                    state="committed",
                    phase="committed",
                    expected_states={"planning"},
                )

            for index in range(2):
                publication_id = f"publication-recovery-{index}"
                store.insert_runtime_publication(
                    publication_id=publication_id,
                    kind="process_exec",
                    pid=f"pid-recovery-{index}",
                    owner_instance_id="test-owner",
                    plan={},
                )
                assert store.advance_runtime_publication(
                    publication_id,
                    state="failed",
                    phase="failed",
                    expected_states={"planning"},
                )

            created_at = utc_now()
            for pid in ("pid-orphan-0", "pid-orphan-1", "pid-covered"):
                store.insert_process(_created_process(pid, created_at=created_at))
            store.insert_runtime_publication(
                publication_id="publication-covered",
                kind="process_launch",
                pid="pid-covered",
                owner_instance_id="test-owner",
                plan={},
            )

            first = store.query_runtime_publication_operation_reconciliation(
                kind="process_launch",
                state="committed",
                after=None,
                limit=1,
            )
            assert [item["publication_id"] for item in first.records] == [
                "publication-page-0"
            ]
            assert first.next_cursor == RuntimePublicationCursor(
                first.records[0]["created_at"],
                "publication-page-0",
            )
            assert not store.mark_runtime_publication_operation_reconciled(
                "publication-page-0",
                expected_kind="process_launch",
                expected_state="committed",
                expected_phase="wrong-phase",
                expected_operation_id=None,
            )
            assert store.mark_runtime_publication_operation_reconciled(
                "publication-page-0",
                expected_kind="process_launch",
                expected_state="committed",
                expected_phase="committed",
                expected_operation_id=None,
            )
            assert store.mark_runtime_publication_operation_reconciled(
                "publication-page-0",
                expected_kind="process_launch",
                expected_state="committed",
                expected_phase="committed",
                expected_operation_id=None,
            )
            second = store.query_runtime_publication_operation_reconciliation(
                kind="process_launch",
                state="committed",
                after=first.next_cursor,
                limit=1,
            )
            assert [item["publication_id"] for item in second.records] == [
                "publication-page-1"
            ]
            recovery_first = store.query_runtime_publication_recovery(
                kind="process_exec",
                state="failed",
                operation_reconciled=False,
                after=None,
                limit=1,
            )
            assert [item["publication_id"] for item in recovery_first.records] == [
                "publication-recovery-0"
            ]
            recovery_second = store.query_runtime_publication_recovery(
                kind="process_exec",
                state="failed",
                operation_reconciled=False,
                after=recovery_first.next_cursor,
                limit=1,
            )
            assert [item["publication_id"] for item in recovery_second.records] == [
                "publication-recovery-1"
            ]
            orphan_first = store.query_orphaned_created_processes(
                after=None,
                limit=1,
            )
            assert [process.pid for process in orphan_first.records] == ["pid-orphan-0"]
            assert orphan_first.next_cursor == ProcessCursor(
                created_at,
                "pid-orphan-0",
            )
            orphan_second = store.query_orphaned_created_processes(
                after=orphan_first.next_cursor,
                limit=1,
            )
            assert [process.pid for process in orphan_second.records] == ["pid-orphan-1"]
            assert store.runtime_publication_exists_for_pid(
                "pid-page-0",
                kind="process_launch",
            )
            assert not store.runtime_publication_exists_for_pid(
                "missing-pid",
                kind="process_launch",
            )
            with pytest.raises(ValidationError, match="hard cap"):
                store.query_runtime_publication_operation_reconciliation(
                    kind="process_launch",
                    state="committed",
                    after=None,
                    limit=config.runtime.publication_reconciliation_page_hard_limit
                    + 1,
                )
            _assert_reconciliation_query_uses_index(store, backend)
            _assert_recovery_query_uses_index(store, backend)
            _assert_pid_existence_query_uses_index(store, backend)
            _assert_orphan_query_uses_indexes(store, backend)
        finally:
            store.close()

        reopened = open_store(target, config=config)
        try:
            persisted = reopened.get_runtime_publication("publication-page-0")
            assert persisted is not None
            assert persisted["operation_reconciled"] is True
            remaining = reopened.query_runtime_publication_operation_reconciliation(
                kind="process_launch",
                state="committed",
                after=None,
                limit=10,
            )
            assert [item["publication_id"] for item in remaining.records] == [
                "publication-page-1",
                "publication-page-2",
            ]
        finally:
            reopened.close()


def test_online_terminalization_marks_all_publication_kinds_before_reopen(
    tmp_path: Path,
) -> None:
    target = tmp_path / "online-marker.sqlite"
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_reconciliation_page_size=2)
    )
    runtime = Runtime.open(target, config=config)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="online publication marker",
        )
        runtime.exec_process(
            pid,
            "base-agent:v0",
            goal="online exec publication marker",
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "online checkpoint publication marker",
            actor=pid,
            require_capability=False,
        )
        runtime.checkpoint.restore(
            "test",
            checkpoint_id,
            require_capability=False,
        )
        publications = runtime.store.list_runtime_publications(pid=pid)
        by_kind = {publication["kind"]: publication for publication in publications}
        assert set(by_kind) == {
            "process_launch",
            "process_exec",
            "checkpoint_restore",
        }
        assert all(publication["state"] == "committed" for publication in by_kind.values())
        assert all(
            publication["operation_reconciled"] is True
            for publication in by_kind.values()
        )
    finally:
        runtime.close()

    reopened = Runtime.open(target, config=config)
    try:
        assert reopened.process.reconcile_terminal_publications() == []
        page = reopened.uow.publications.query_runtime_publication_operation_reconciliation(
            kind="process_launch",
            state="committed",
            after=None,
            limit=2,
        )
        assert page.records == ()
        assert reopened.image_boot.reconcile_terminal_publications() == []
        checkpoint_page = (
            reopened.uow.publications.query_runtime_publication_operation_reconciliation(
                kind="checkpoint_restore",
                state="committed",
                after=None,
                limit=2,
            )
        )
        assert checkpoint_page.records == ()
    finally:
        reopened.close()


def test_ten_thousand_terminal_publications_reconcile_in_bounded_restart_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "publication-history.sqlite"
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_reconciliation_page_size=17)
    )
    store = SQLiteStore(target, config=config)
    now = utc_now()
    unreconciled_count = 2 * config.runtime.publication_reconciliation_page_size + 5
    with store.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO runtime_publications ("
            "publication_id, kind, pid, owner_instance_id, state, phase, "
            "plan_json, receipt_json, error_json, operation_reconciled, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    f"publication-history-{index:05d}",
                    "process_launch",
                    f"pid-history-{index:05d}",
                    "history-owner",
                    "committed",
                    "committed",
                    dumps({}),
                    dumps({"phases": [], "artifacts": []}),
                    None,
                    int(index >= unreconciled_count),
                    now,
                    now,
                )
                for index in range(10_000)
            ),
        )
    orphan_count = 2 * config.runtime.publication_reconciliation_page_size + 5
    for index in range(orphan_count):
        store.insert_process(
            _created_process(
                f"pid-orphan-history-{index:05d}",
                created_at=now,
            )
        )
    store.close()

    original_query = SQLiteStore._query
    publication_queries: list[tuple[str, int]] = []
    terminal_reconciliation_queries: list[tuple[str, str]] = []

    def tracked_query(
        selected_store: SQLiteStore,
        sql: str,
        params: object = (),
    ) -> list[object]:
        rows = original_query(selected_store, sql, params)  # type: ignore[arg-type]
        if "FROM runtime_publications" in sql:
            publication_queries.append((sql, len(rows)))
            if (
                "operation_reconciled = 0" in sql
                and isinstance(params, (list, tuple))
                and len(params) >= 2
            ):
                terminal_reconciliation_queries.append(
                    (str(params[0]), str(params[1]))
                )
        return rows

    monkeypatch.setattr(SQLiteStore, "_query", tracked_query)
    reopened = Runtime.open(target, config=config)
    try:
        assert publication_queries
        assert all("WHERE" in sql for sql, _row_count in publication_queries)
        assert max(row_count for _sql, row_count in publication_queries) <= 18
        reconciliation_rows = [
            row_count
            for sql, row_count in publication_queries
            if "operation_reconciled = 0" in sql
        ]
        assert sum(reconciliation_rows) == unreconciled_count + 2
        assert all(
            reopened.process.get(f"pid-orphan-history-{index:05d}").status
            == ProcessStatus.FAILED
            for index in range(orphan_count)
        )
        assert reopened.process.reconcile_terminal_publications() == []
    finally:
        close_runtime(reopened)

    publication_queries.clear()
    terminal_reconciliation_queries.clear()
    reopened_again = Runtime.open(target, config=config)
    try:
        reconciliation_rows = [
            row_count
            for sql, row_count in publication_queries
            if "operation_reconciled = 0" in sql
        ]
        expected_terminal_queries = {
            (kind, state)
            for kind in ("process_launch", "process_exec")
            for state in ("committed", "rolled_back", "failed", "manual")
        } | {("checkpoint_restore", "committed")}
        # Checkpoint delivery performs one recovery-time repair before generic
        # stale-operation handling and a second repair after startup hooks,
        # immediately before OPEN. Other terminal publication kinds need only
        # the generic recovery pass.
        assert len(terminal_reconciliation_queries) == len(
            expected_terminal_queries
        ) + 1
        assert set(terminal_reconciliation_queries) == expected_terminal_queries
        assert terminal_reconciliation_queries.count(
            ("checkpoint_restore", "committed")
        ) == 2
        assert sum(reconciliation_rows) == 0
        assert max(
            (row_count for _sql, row_count in publication_queries),
            default=0,
        ) <= 18
    finally:
        close_runtime(reopened_again)


def test_multi_page_reconciliation_and_recovery_collectors_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "bounded-collectors.sqlite"
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_reconciliation_page_size=17)
    )
    runtime = Runtime.open(target, config=config)
    try:
        now = utc_now()
        total = 2 * config.runtime.publication_reconciliation_page_size + 5
        with runtime.store.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO runtime_publications ("
                "publication_id, kind, pid, owner_instance_id, state, phase, "
                "plan_json, receipt_json, error_json, operation_reconciled, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"publication-collector-{index:05d}",
                        "process_launch",
                        f"pid-collector-{index:05d}",
                        "collector-owner",
                        "committed",
                        "committed",
                        dumps({}),
                        dumps({"phases": [], "artifacts": []}),
                        None,
                        0,
                        now,
                        now,
                    )
                    for index in range(total)
                ),
            )
        reconciled = runtime.process.reconcile_terminal_publications()
        assert len(reconciled) == config.runtime.publication_reconciliation_page_size
        assert runtime.store._query(
            "SELECT COUNT(*) AS count FROM runtime_publications "
            "WHERE operation_reconciled = 0"
        )[0]["count"] == 0

        recovered_ids: list[str] = []
        with runtime.store.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO runtime_publications ("
                "publication_id, kind, pid, owner_instance_id, state, phase, "
                "plan_json, receipt_json, error_json, operation_reconciled, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"publication-recovery-collector-{index:05d}",
                        "process_launch",
                        f"pid-recovery-collector-{index:05d}",
                        "collector-owner",
                        "planning",
                        "planned",
                        dumps({}),
                        dumps({"phases": [], "artifacts": []}),
                        None,
                        0,
                        now,
                        now,
                    )
                    for index in range(total)
                ),
            )

        close_runtime(runtime)

        def record_recovery(
            _manager: object,
            publication: dict[str, object],
        ) -> str:
            publication_id = str(publication["publication_id"])
            recovered_ids.append(publication_id)
            return publication_id

        monkeypatch.setattr(
            type(runtime.process),
            "_recover_launch_publication",
            record_recovery,
        )
        runtime = Runtime.open(target, config=config)
        recovered = runtime.recovered_runtime_publications
        assert len(recovered_ids) == total
        assert len(recovered) == config.runtime.publication_reconciliation_page_size
    finally:
        close_runtime(runtime)


def test_checkpoint_terminal_operation_reconciliation_is_paged_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_size = 2
    total = 2 * page_size + 1
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_reconciliation_page_size=page_size)
    )
    runtime = Runtime.open(
        tmp_path / "checkpoint-operation-reconciliation.sqlite",
        config=config,
    )
    try:
        pid = runtime.process.spawn(goal="paged checkpoint operation reconciliation")
        publication_ids: list[str] = []
        operation_ids: list[str] = []
        for index in range(total):
            checkpoint_id = runtime.checkpoint.create(
                pid,
                f"checkpoint operation page {index}",
                actor=pid,
                require_capability=False,
            )
            result = runtime.checkpoint.restore(
                "test",
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result["publication_id"])
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            operation_id = str(publication["plan"]["operation_id"])
            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            assert runtime.store.update_operation(
                replace(
                    operation,
                    state=OperationState.RUNNING,
                    outcome=OperationOutcome.PENDING,
                    completed_at=None,
                ),
                expected_states=[OperationState.TERMINAL.value],
            )
            publication_ids.append(publication_id)
            operation_ids.append(operation_id)

        original_query = runtime.store._query  # type: ignore[attr-defined]
        page_row_counts: list[int] = []

        def tracked_query(sql: str, params: object = ()) -> list[object]:
            rows = original_query(sql, params)
            if (
                "/* operation-reconciliation */" in sql
                and "kind = ?" in sql
                and params
                and params[0] == "checkpoint_restore"  # type: ignore[index]
            ):
                page_row_counts.append(len(rows))
            return rows

        monkeypatch.setattr(runtime.store, "_query", tracked_query)
        reconciled = runtime.checkpoint.reconcile_terminal_restore_publications()

        assert len(reconciled) == page_size
        assert len(page_row_counts) == 3
        assert max(page_row_counts) <= page_size + 1
        assert runtime.checkpoint.reconcile_terminal_restore_publications() == []
        for publication_id, operation_id in zip(
            publication_ids,
            operation_ids,
            strict=True,
        ):
            publication = runtime.store.get_runtime_publication(publication_id)
            operation = runtime.store.get_operation(operation_id)
            assert publication is not None
            assert publication["operation_reconciled"] is True
            assert operation is not None
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.SUCCEEDED
    finally:
        runtime.close()


def _assert_reconciliation_query_uses_index(store: object, backend: str) -> None:
    sql = (
        "SELECT * FROM runtime_publications /* operation-reconciliation */ "
        "WHERE kind = ? AND state = ? "
        "AND operation_reconciled = 0 "
        "AND (created_at, publication_id) > (?, ?) "
        "ORDER BY created_at, publication_id LIMIT ?"
    )
    params = (
        "process_launch",
        "committed",
        "2026-01-01T00:00:00Z",
        "publication-deep-cursor",
        10,
    )
    if backend == "sqlite":
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN QUERY PLAN {sql}",
            params,
        )
        details = "\n".join(str(row[3]) for row in rows)
    else:
        store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN {sql}",
            params,
        )
        details = "\n".join(str(row["QUERY PLAN"]) for row in rows)
    assert "idx_runtime_publications_operation_reconciliation" in details
    assert "(created_at,publication_id)>" in _normalized_plan(details)


def _assert_recovery_query_uses_index(store: object, backend: str) -> None:
    sql = (
        "SELECT * FROM runtime_publications /* recovery */ "
        "WHERE kind = ? AND state = ? AND operation_reconciled = ? "
        "AND (created_at, publication_id) > (?, ?) "
        "ORDER BY created_at, publication_id LIMIT ?"
    )
    params = (
        "process_exec",
        "failed",
        0,
        "2026-01-01T00:00:00Z",
        "publication-deep-cursor",
        10,
    )
    if backend == "sqlite":
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN QUERY PLAN {sql}",
            params,
        )
        details = "\n".join(str(row[3]) for row in rows)
    else:
        store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN {sql}",
            params,
        )
        details = "\n".join(str(row["QUERY PLAN"]) for row in rows)
    assert "idx_runtime_publications_operation_reconciliation" in details
    assert "(created_at,publication_id)>" in _normalized_plan(details)


def _assert_pid_existence_query_uses_index(store: object, backend: str) -> None:
    sql = (
        "SELECT 1 AS present FROM runtime_publications "
        "WHERE pid = ? AND kind = ? LIMIT 1"
    )
    if backend == "sqlite":
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN QUERY PLAN {sql}",
            ("pid-page-0", "process_launch"),
        )
        details = "\n".join(str(row[3]) for row in rows)
    else:
        store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN {sql}",
            ("pid-page-0", "process_launch"),
        )
        details = "\n".join(str(row["QUERY PLAN"]) for row in rows)
    assert "idx_runtime_publications_pid_kind" in details


def _assert_orphan_query_uses_indexes(store: object, backend: str) -> None:
    sql = (
        "SELECT processes.* FROM processes WHERE processes.status = ? "
        "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
        "WHERE runtime_publications.pid = processes.pid "
        "AND runtime_publications.kind = ? LIMIT 1 OFFSET 0) "
        "AND (processes.created_at, processes.pid) > (?, ?) "
        "ORDER BY processes.created_at, processes.pid LIMIT ?"
    )
    params = (
        ProcessStatus.CREATED.value,
        "process_launch",
        "2026-01-01T00:00:00Z",
        "pid-deep-cursor",
        10,
    )
    if backend == "sqlite":
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN QUERY PLAN {sql}",
            params,
        )
        details = "\n".join(str(row[3]) for row in rows)
    else:
        store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
        rows = store.conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN {sql}",
            params,
        )
        details = "\n".join(str(row["QUERY PLAN"]) for row in rows)
    assert "idx_processes_status_created" in details
    assert "idx_runtime_publications_pid_kind" in details
    assert "(created_at,pid)>" in _normalized_plan(details)
    if backend == "postgres":
        index_conditions = "\n".join(
            line.casefold()
            for line in details.splitlines()
            if "index cond:" in line.casefold()
        )
        assert "pid = processes.pid" in index_conditions, details
        assert "kind" in index_conditions


def _normalized_plan(details: str) -> str:
    return details.lower().replace('"', "").replace(" ", "")


def _created_process(pid: str, *, created_at: str) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        parent_pid=None,
        image_id="base-agent:v0",
        status=ProcessStatus.CREATED,
        goal_oid=None,
        memory_view=None,
        capabilities=[],
        loaded_skills={},
        tool_table={},
        event_cursor=None,
        checkpoint_head=None,
        resource_budget=ResourceBudget(),
        resource_usage=ResourceUsage(),
        created_at=created_at,
        updated_at=created_at,
    )


@contextlib.contextmanager
def _store_target(
    backend: str,
    tmp_path: Path,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if backend == "sqlite":
        yield tmp_path / "publication-reconciliation.sqlite", AgentLibOSConfig()
        return
    if backend == "postgres":
        with _postgres_schema_dsn() as dsn:
            yield dsn, AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
        return
    raise AssertionError(f"unknown backend: {backend}")


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_publication_reconcile_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
