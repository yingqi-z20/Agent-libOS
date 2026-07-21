from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from benchmarks.runtime_publication_recovery import runner as publication_runner
from benchmarks.runtime_publication_recovery import (
    PUBLICATION_SCALE_PROFILES,
    TERMINAL_RECONCILIATION_STATES,
    run_publication_scale_benchmark,
)
from experiments.run_publication_reconciliation_scale import main
from agent_libos.runtime.checkpoint_reconciliation import (
    CheckpointRestoreReconciler,
)
from agent_libos.runtime.operation_manager import OperationManager


def test_publication_scale_profile_and_workflows_are_stable() -> None:
    assert PUBLICATION_SCALE_PROFILES["ci"].total_records == 10_000
    assert PUBLICATION_SCALE_PROFILES["ci"].unreconciled_records == 1_001
    root = Path(__file__).resolve().parents[2]
    release_workflow = (root / ".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )
    scheduled_workflow = (
        root / ".github/workflows/external-effect-recovery-scale.yml"
    ).read_text(encoding="utf-8")
    for workflow in (release_workflow, scheduled_workflow):
        assert "run_publication_reconciliation_scale.py" in workflow
        assert "--profile ci" in workflow


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM runtime_publications",
        "SELECT COUNT(*) FROM runtime_publications",
        (
            "SELECT * FROM runtime_publications WHERE operation_reconciled = 0 "
            "ORDER BY created_at, publication_id LIMIT ?"
        ),
        (
            "SELECT * FROM runtime_publications "
            "WHERE publication_id = 'one' OR 1 = 1"
        ),
        (
            "SELECT * FROM runtime_publications /* operation-reconciliation */ "
            "WHERE kind = 'process_launch' AND state = 'committed' "
            "AND operation_reconciled = 0 OR 1 = 1 "
            "ORDER BY created_at, publication_id LIMIT 2"
        ),
        (
            "SELECT * FROM runtime_publications /* recovery */ "
            "WHERE kind = 'checkpoint_restore' AND state = 'planning' "
            "AND operation_reconciled = 0 OR 1 = 1 "
            "ORDER BY created_at, publication_id LIMIT 2"
        ),
        'SELECT * FROM "runtime_publications"',
        "SELECT * FROM `runtime_publications`",
        "SELECT * FROM [runtime_publications]",
        'SELECT * FROM "main"."runtime_publications"',
        "SELECT * FROM [main].[runtime_publications]",
        'SELECT * FROM-- comment separator\n"runtime_publications"',
        'SELECT * FROM "runtime_publications"rp',
        "SELECT * FROM `runtime_publications`rp",
        "SELECT * FROM [runtime_publications]rp",
        "SELECT * FROM 'runtime_publications'rp",
        (
            "SELECT * FROM runtime_publications /* operation-reconciliation */ "
            "WHERE kind = ? AND state = ? AND operation_reconciled = 0 "
            "AND (created_at > ? OR (created_at = ? AND publication_id > ?)) "
            "ORDER BY created_at, publication_id LIMIT ?"
        ),
    ],
)
def test_publication_scale_default_denies_unreviewed_selects(sql: str) -> None:
    assert publication_runner._publication_select_shape(sql) == "unreviewed"
    assert not publication_runner._is_reviewed_publication_trace(sql)


def test_publication_scale_reviews_only_the_exact_domain_validation_query() -> None:
    sql = publication_runner._PUBLICATION_DOMAIN_SELECT
    assert publication_runner._publication_select_shape(sql) == "domain_validation"
    assert publication_runner._is_reviewed_publication_trace(sql)
    assert (
        publication_runner._publication_select_shape(
            sql.replace("LIMIT 1", "LIMIT 2")
        )
        == "unreviewed"
    )


def test_publication_scale_rejects_direct_schema_initialization_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_finish_schema = (
        publication_runner.SQLiteStore._finish_schema_initialization
    )

    def finish_schema_with_unreviewed_scan(
        store: publication_runner.SQLiteStore,
    ) -> None:
        store.conn.execute("SELECT * FROM runtime_publications").fetchone()
        original_finish_schema(store)

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_finish_schema_initialization",
        finish_schema_with_unreviewed_scan,
    )

    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_traces_connect_wrapper_prefix_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = publication_runner.sqlite_storage.sqlite3.connect
    connection_calls = 0

    def connect_with_prefix_select(*args: Any, **kwargs: Any) -> Any:
        nonlocal connection_calls
        connection_calls += 1
        connection = real_connect(*args, **kwargs)
        # Seed creation uses the first connection.  The measured reopen starts
        # with the second, read-only preflight connection.
        if connection_calls == 2:
            connection.execute(
                "SELECT * FROM runtime_publications LIMIT 1"
            ).fetchone()
        return connection

    monkeypatch.setattr(
        publication_runner.sqlite_storage.sqlite3,
        "connect",
        connect_with_prefix_select,
    )
    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


@pytest.mark.parametrize("execution_api", ["cursor", "connection"])
def test_publication_scale_rejects_marker_shaped_or_injection_from_direct_sql(
    monkeypatch: pytest.MonkeyPatch,
    execution_api: str,
) -> None:
    original_finish_schema = (
        publication_runner.SQLiteStore._finish_schema_initialization
    )

    def finish_schema_with_marker_shaped_injection(
        store: publication_runner.SQLiteStore,
    ) -> None:
        sql = (
            "SELECT * FROM runtime_publications /* recovery */ "
            "WHERE kind = 'checkpoint_restore' AND state = 'planning' "
            "AND operation_reconciled = 0 OR 1 = 1 "
            "ORDER BY created_at, publication_id LIMIT 2"
        )
        if execution_api == "cursor":
            cursor = store.conn.cursor()
            try:
                cursor.execute(sql).fetchone()
            finally:
                cursor.close()
        else:
            store.conn.execute(sql).fetchone()
        original_finish_schema(store)

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_finish_schema_initialization",
        finish_schema_with_marker_shaped_injection,
    )

    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


@pytest.mark.parametrize(
    ("execution_api", "table_reference"),
    [
        ("cursor", '"runtime_publications"'),
        ("connection", "`runtime_publications`"),
        ("cursor", "[runtime_publications]"),
        ("connection", '"main"."runtime_publications"'),
        ("cursor", "[main].[runtime_publications]"),
        ("cursor", '"runtime_publications"rp'),
        ("connection", "`runtime_publications`rp"),
        ("cursor", "[runtime_publications]rp"),
        ("connection", "'runtime_publications'rp"),
    ],
)
def test_publication_scale_rejects_quoted_direct_publication_reads(
    monkeypatch: pytest.MonkeyPatch,
    execution_api: str,
    table_reference: str,
) -> None:
    original_finish_schema = (
        publication_runner.SQLiteStore._finish_schema_initialization
    )

    def finish_schema_with_quoted_scan(
        store: publication_runner.SQLiteStore,
    ) -> None:
        sql = f"SELECT * FROM {table_reference}"
        if execution_api == "cursor":
            cursor = store.conn.cursor()
            try:
                cursor.execute(sql).fetchone()
            finally:
                cursor.close()
        else:
            store.conn.execute(sql).fetchone()
        original_finish_schema(store)

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_finish_schema_initialization",
        finish_schema_with_quoted_scan,
    )

    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_pre_finish_direct_cursor_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_runner.SQLiteStore._create_runtime_publication_indexes

    def create_indexes_with_scan(store: publication_runner.SQLiteStore) -> None:
        original(store)
        cursor = store.conn.cursor()
        try:
            cursor.execute("SELECT * FROM runtime_publications").fetchone()
        finally:
            cursor.close()

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_create_runtime_publication_indexes",
        create_indexes_with_scan,
    )
    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_preflight_history_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_runner.SQLiteStore._preflight_existing_store

    def preflight_with_scan(
        store: publication_runner.SQLiteStore,
        database_path: Path,
    ) -> None:
        original(store, database_path)
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro",
            uri=True,
        )
        try:
            connection.execute("SELECT * FROM runtime_publications").fetchone()
        finally:
            connection.close()

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_preflight_existing_store",
        preflight_with_scan,
    )
    with pytest.raises(
        AssertionError,
        match="Runtime reopen executed an unreviewed publication",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_extra_reviewed_primary_key_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_runner.ProcessManager.reconcile_terminal_publications

    def reconcile_with_extra_reads(
        manager: publication_runner.ProcessManager,
    ) -> list[str]:
        reconciled = original(manager)
        backend = manager.publications._publication_backend
        for _ in range(100):
            backend.conn.execute(
                "SELECT * FROM runtime_publications WHERE publication_id = ?",
                ("publication-history-00000000",),
            ).fetchone()
        return reconciled

    monkeypatch.setattr(
        publication_runner.ProcessManager,
        "reconcile_terminal_publications",
        reconcile_with_extra_reads,
    )
    with pytest.raises(
        AssertionError,
        match="publication traced SELECT/helper multiset changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_substituted_terminal_query_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CheckpointRestoreReconciler.reconcile_terminal_publications

    def reconcile_with_duplicate_exec_key(
        reconciler: CheckpointRestoreReconciler,
        *,
        recover_payload_delivery: bool = False,
    ) -> list[str]:
        reconciler._store.query_runtime_publication_operation_reconciliation(
            kind="process_exec",
            state="committed",
            after=None,
            limit=reconciler._reconciliation_page_size,
        )
        return original(
            reconciler,
            recover_payload_delivery=recover_payload_delivery,
        )

    monkeypatch.setattr(
        CheckpointRestoreReconciler,
        "reconcile_terminal_publications",
        reconcile_with_duplicate_exec_key,
    )
    with pytest.raises(
        AssertionError,
        match="publication repository query-key multiset changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_direct_selects_substituted_for_helper_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_query = publication_runner.SQLiteStore._query
    real_reconcile = (
        publication_runner.ProcessManager.reconcile_terminal_publications
    )
    cached_rows: dict[str, list[dict[str, Any]]] = {}

    def query_with_substituted_exact_select(
        store: publication_runner.SQLiteStore,
        sql: str,
        params: Any = (),
    ) -> list[Any]:
        selected_params = tuple(params)
        if (
            publication_runner._publication_select_shape(sql)
            == "exact_publication"
            and selected_params
            and str(selected_params[0]) in cached_rows
        ):
            # Preserve helper row metrics while omitting its SQL execution.
            return list(cached_rows[str(selected_params[0])])
        return real_query(store, sql, selected_params)

    def reconcile_with_direct_replacements(
        manager: publication_runner.ProcessManager,
    ) -> list[str]:
        connection = manager.publications._publication_backend.conn
        for index in range(39):
            publication_id = f"publication-history-{index:08d}"
            rows = list(
                connection.execute(
                    "SELECT * FROM runtime_publications "
                    "WHERE publication_id = ?",
                    (publication_id,),
                )
            )
            cached_rows[publication_id] = [dict(row) for row in rows]
            # Each publication has two exact repository reads.  Match the old
            # shape totals exactly so only execution/helper correspondence can
            # distinguish these direct reads from measured helper work.
            list(
                connection.execute(
                    "SELECT * FROM runtime_publications "
                    "WHERE publication_id = ?",
                    (publication_id,),
                )
            )
        return real_reconcile(manager)

    monkeypatch.setattr(
        publication_runner.SQLiteStore,
        "_query",
        query_with_substituted_exact_select,
    )
    monkeypatch.setattr(
        publication_runner.ProcessManager,
        "reconcile_terminal_publications",
        reconcile_with_direct_replacements,
    )
    with pytest.raises(
        AssertionError,
        match="did not execute exactly one matching traced SELECT",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_traces_handler_connections_opened_during_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("Windows SQLite runtime ownership uses an exclusive connection lease")
    original = publication_runner.ProcessManager.reconcile_terminal_publications

    def reconcile_with_second_connection(
        manager: publication_runner.ProcessManager,
    ) -> list[str]:
        reconciled = original(manager)
        backend = manager.publications._publication_backend
        database_path = str(
            backend.conn.execute("PRAGMA database_list").fetchone()[2]
        )
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("SELECT * FROM runtime_publications").fetchone()
        finally:
            connection.close()
        return reconciled

    monkeypatch.setattr(
        publication_runner.ProcessManager,
        "reconcile_terminal_publications",
        reconcile_with_second_connection,
    )
    with pytest.raises(
        AssertionError,
        match="unreviewed publication statement",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE runtime_publications SET updated_at = updated_at RETURNING *",
        "DELETE FROM runtime_publications WHERE publication_id = "
        "'publication-history-00000000'",
        "UPDATE runtime_publications SET kind = 'process_exec', "
        "owner_instance_id = 'wrong-marker' WHERE publication_id = "
        "'publication-history-00000000'",
        "UPDATE runtime_publications SET updated_at = updated_at /* SELECT */",
    ],
)
def test_publication_scale_rejects_handler_dml_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tamper_sql: str,
) -> None:
    original = publication_runner.ProcessManager.reconcile_terminal_publications

    def reconcile_with_tamper(
        manager: publication_runner.ProcessManager,
    ) -> list[str]:
        reconciled = original(manager)
        connection = manager.publications._publication_backend.conn
        list(connection.execute(tamper_sql))
        connection.commit()
        return reconciled

    monkeypatch.setattr(
        publication_runner.ProcessManager,
        "reconcile_terminal_publications",
        reconcile_with_tamper,
    )
    with pytest.raises(
        AssertionError,
        match="unreviewed publication statement",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_same_name_weak_reconciliation_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_runner._seed_terminal_publication_history

    def seed_with_weak_index(
        store: publication_runner.SQLiteStore,
        **kwargs: int,
    ) -> None:
        original(store, **kwargs)
        with store.transaction() as cursor:
            cursor.execute(
                "DROP INDEX idx_runtime_publications_operation_reconciliation"
            )
            cursor.execute(
                "CREATE INDEX idx_runtime_publications_operation_reconciliation "
                "ON runtime_publications(state, created_at, publication_id)"
            )

    monkeypatch.setattr(
        publication_runner,
        "_seed_terminal_publication_history",
        seed_with_weak_index,
    )
    with pytest.raises(
        AssertionError,
        match="reconciliation index columns changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_exercises_bound_operation_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_reconciliation(*args: object, **kwargs: object) -> object:
        raise AssertionError("bound operation reconciliation bypassed")

    monkeypatch.setattr(
        OperationManager,
        "reconcile_runtime_publication",
        reject_reconciliation,
    )
    with pytest.raises(
        AssertionError,
        match="bound operation reconciliation bypassed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE operations SET pid = NULL "
        "WHERE operation_id = 'scale-operation-00000000'",
        "UPDATE operations SET metadata_json = "
        "json_remove(metadata_json, '$.runtime_publication_state') "
        "WHERE operation_id = 'scale-operation-00000000'",
        "UPDATE operations SET expected_roles_json = '[\"unexpected\"]' "
        "WHERE operation_id = 'scale-operation-00000000'",
        "UPDATE operations SET completed_at = NULL "
        "WHERE operation_id = 'scale-operation-00000000'",
        "UPDATE runtime_publications SET plan_json = "
        "json_remove(plan_json, '$.operation_id') "
        "WHERE publication_id = 'publication-history-00000000'",
    ],
)
def test_publication_scale_convergence_is_null_safe(
    monkeypatch: pytest.MonkeyPatch,
    tamper_sql: str,
) -> None:
    original = publication_runner._publication_convergence

    def convergence_after_null_tamper(
        store: publication_runner.SQLiteStore,
        **kwargs: int,
    ) -> dict[str, int]:
        store.conn.execute(tamper_sql)
        store.conn.commit()
        return original(store, **kwargs)

    monkeypatch.setattr(
        publication_runner,
        "_publication_convergence",
        convergence_after_null_tamper,
    )
    with pytest.raises(
        AssertionError,
        match="seeded publication/operation convergence changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "DELETE FROM runtime_publications WHERE publication_id = "
        "'publication-history-00000000'",
        "UPDATE runtime_publications SET state = 'rolled_back' "
        "WHERE publication_id = 'publication-history-00000000'",
    ],
)
def test_publication_scale_terminal_multiset_rejects_postcheck_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tamper_sql: str,
) -> None:
    original = publication_runner._publication_convergence

    def convergence_then_tamper(
        store: publication_runner.SQLiteStore,
        **kwargs: int,
    ) -> dict[str, int]:
        clean_result = original(store, **kwargs)
        store.conn.execute(tamper_sql)
        store.conn.commit()
        return clean_result

    monkeypatch.setattr(
        publication_runner,
        "_publication_convergence",
        convergence_then_tamper,
    )
    with pytest.raises(
        AssertionError,
        match="seeded publication terminal multiset changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_rejects_an_extra_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_runner._publication_convergence

    def convergence_after_extra_operation(
        store: publication_runner.SQLiteStore,
        **kwargs: int,
    ) -> dict[str, int]:
        store.conn.execute(
            """
            INSERT INTO operations (
                operation_id, root_operation_id, parent_operation_id, kind,
                name, actor, pid, state, outcome, expected_roles_json,
                metadata_json, runtime_publication_id, started_at, updated_at,
                completed_at
            )
            SELECT 'unexpected-operation', 'unexpected-operation', NULL, kind,
                   name, actor, pid, state, outcome, expected_roles_json,
                   '{}', NULL, started_at, updated_at, completed_at
              FROM operations
             LIMIT 1
            """
        )
        store.conn.commit()
        return original(store, **kwargs)

    monkeypatch.setattr(
        publication_runner,
        "_publication_convergence",
        convergence_after_extra_operation,
    )
    with pytest.raises(
        AssertionError,
        match="seeded publication/operation convergence changed",
    ):
        run_publication_scale_benchmark(
            total_records=40,
            unreconciled_records=39,
            page_size=17,
        )


def test_publication_scale_handler_is_bounded_and_independent_of_history() -> None:
    small = run_publication_scale_benchmark(
        total_records=100,
        unreconciled_records=39,
        page_size=17,
    )
    large = run_publication_scale_benchmark(
        total_records=1_000,
        unreconciled_records=39,
        page_size=17,
    )

    assert small.handler_reconciled_records == 39
    assert large.handler_reconciled_records == 39
    assert small.handler_sample_records == large.handler_sample_records == 17
    assert small.total_operations_after == large.total_operations_after == 39
    assert small.bound_operations_after == large.bound_operations_after == 39
    assert small.handler_query_calls == large.handler_query_calls == 3
    assert small.handler_raw_rows_fetched == 41
    assert large.handler_raw_rows_fetched == 41
    assert small.publication_select_calls == small.publication_query_calls
    assert large.publication_select_calls == large.publication_query_calls
    assert small.seeded_terminal_missing_rows == 0
    assert large.seeded_terminal_missing_rows == 0
    assert small.seeded_terminal_unexpected_rows == 0
    assert large.seeded_terminal_unexpected_rows == 0
    expected_reconciliation_queries = (
        sum(len(states) for states in TERMINAL_RECONCILIATION_STATES.values())
        - 1
        + 3
    )
    assert (
        small.reconciliation_query_calls
        == large.reconciliation_query_calls
        == expected_reconciliation_queries
    )
    assert small.publication_query_calls == large.publication_query_calls
    assert small.domain_validation_query_calls == 1
    assert large.domain_validation_query_calls == 1
    assert small.domain_validation_rows_fetched == 0
    assert large.domain_validation_rows_fetched == 0
    assert small.domain_validation_index in "\n".join(
        small.domain_validation_query_plan
    )
    assert small.reconciliation_index in "\n".join(
        small.reconciliation_query_plan_first
    )
    assert small.reconciliation_index in "\n".join(
        small.reconciliation_query_plan_resumed
    )
    assert "/* operation-reconciliation */" in small.reconciliation_sql_first
    assert small.reconciliation_params_first[:2] == (
        "process_launch",
        "committed",
    )
    assert "/* operation-reconciliation */" in small.reconciliation_sql_resumed
    assert small.reconciliation_params_resumed[:2] == (
        "process_launch",
        "committed",
    )
    assert len(small.reconciliation_params_resumed) > len(
        small.reconciliation_params_first
    )


def test_publication_scale_cli_writes_structural_metrics(tmp_path: Path) -> None:
    output = tmp_path / "publication-scale.json"
    assert main(
        [
            "--profile",
            "ci",
            "--total-records",
            "101",
            "--unreconciled-records",
            "39",
            "--page-size",
            "17",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total_records"] == 101
    assert payload["unreconciled_records"] == 39
    assert payload["handler_reconciled_records"] == 39
    assert payload["handler_sample_records"] == 17
    assert payload["total_operations_after"] == 39
    assert payload["handler_query_calls"] == 3
    assert payload["handler_raw_rows_fetched"] == 41
    assert payload["timing_is_informational_only"] is True


@pytest.mark.parametrize(
    ("total_records", "unreconciled_records", "page_size"),
    [
        (0, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (6_000, 5_002, 5_001),
    ],
)
def test_publication_scale_rejects_invalid_shapes(
    total_records: int,
    unreconciled_records: int,
    page_size: int,
) -> None:
    with pytest.raises(ValueError):
        run_publication_scale_benchmark(
            total_records=total_records,
            unreconciled_records=unreconciled_records,
            page_size=page_size,
        )
