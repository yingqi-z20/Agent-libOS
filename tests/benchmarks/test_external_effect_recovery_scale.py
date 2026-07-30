from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import ExternalEffectRecoveryQuery
from agent_libos.runtime import RuntimeBuilder
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.sql import SQLRuntimeStore, _V3_REQUIRED_COLUMNS
from benchmarks.external_effect_recovery import (
    BENCHMARK_PROFILES,
    run_recovery_scale_benchmark,
)
from benchmarks.external_effect_recovery import runner as recovery_runner
from experiments import recovery_artifact_metadata as recovery_metadata_module
from experiments.run_external_effect_recovery_scale import main
from experiments.recovery_artifact_metadata import (
    build_recovery_artifact_metadata,
    new_recovery_run_identity,
)


def test_ci_and_million_profiles_are_stable() -> None:
    assert BENCHMARK_PROFILES["ci"].total_records == 100_000
    assert BENCHMARK_PROFILES["million"].total_records == 1_000_000


def test_recovery_benchmark_serializes_complete_global_patch_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def fake_locked(**kwargs: object) -> int:
        selected = int(kwargs["total_records"])  # type: ignore[arg-type]
        if selected == 1:
            first_entered.set()
            assert release_first.wait(2)
        else:
            second_entered.set()
        return selected

    monkeypatch.setattr(
        recovery_runner,
        "_run_recovery_scale_benchmark_locked",
        fake_locked,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            run_recovery_scale_benchmark,
            total_records=1,
            pending_records=0,
            page_size=1,
        )
        assert first_entered.wait(2)
        second = executor.submit(
            run_recovery_scale_benchmark,
            total_records=2,
            pending_records=0,
            page_size=1,
        )
        assert not second_entered.wait(0.1)
        release_first.set()
        assert first.result(timeout=2) == 1
        assert second.result(timeout=2) == 2


def test_recovery_connect_trace_filter_matches_only_benchmark_database(
    tmp_path: Path,
) -> None:
    database = (tmp_path / "benchmark.sqlite").resolve()

    assert recovery_runner._connect_targets_database((database,), {}, database)
    assert recovery_runner._connect_targets_database(
        (f"{database.as_uri()}?mode=rw",),
        {"uri": True},
        database,
    )
    assert not recovery_runner._connect_targets_database(
        (tmp_path / "unrelated.sqlite",),
        {},
        database,
    )


def test_scale_profiles_are_wired_into_ci_and_nightly_workflows() -> None:
    root = Path(__file__).resolve().parents[2]
    release_workflow = (root / ".github/workflows/test.yml").read_text(
        encoding="utf-8"
    )
    scale_workflow = (
        root / ".github/workflows/external-effect-recovery-scale.yml"
    ).read_text(encoding="utf-8")

    assert "run_external_effect_recovery_scale.py" in release_workflow
    assert "--profile ci" in release_workflow
    assert "workflow_dispatch:" in scale_workflow
    assert "schedule:" in scale_workflow
    assert "--profile million" in scale_workflow


@pytest.mark.parametrize(
    ("transaction_states", "expected_index"),
    [
        ((), "idx_external_effects_recovery_state"),
        (("prepared",), "idx_external_effects_recovery_transaction"),
    ],
)
def test_sqlite_recovery_plans_use_the_matching_composite_index(
    transaction_states: tuple[str, ...],
    expected_index: str,
) -> None:
    result = run_recovery_scale_benchmark(
        total_records=2_000,
        pending_records=513,
        page_size=128,
        transaction_states=transaction_states,
    )

    assert result.recovery_index == expected_index
    assert expected_index in "\n".join(result.query_plan_first)
    assert expected_index in "\n".join(result.query_plan_resumed)


def test_recovery_work_depends_on_pending_pages_not_total_history() -> None:
    small_history = run_recovery_scale_benchmark(
        total_records=500,
        pending_records=73,
        page_size=32,
    )
    large_history = run_recovery_scale_benchmark(
        total_records=10_000,
        pending_records=73,
        page_size=32,
    )

    assert small_history.query_calls == large_history.query_calls == 3
    assert small_history.raw_rows_fetched == large_history.raw_rows_fetched == 75
    assert (
        small_history.recovery_work_units
        == large_history.recovery_work_units
        == 78
    )
    assert small_history.handler_recovered_records == 73
    assert large_history.handler_recovered_records == 73
    assert small_history.handler_sample_records == 32
    assert large_history.handler_sample_records == 32
    assert small_history.handler_query_calls == 4
    assert large_history.handler_query_calls == 4
    assert small_history.handler_observed_selects == 5
    assert large_history.handler_observed_selects == 5
    assert small_history.handler_observed_statements == 78
    assert large_history.handler_observed_statements == 78
    assert small_history.handler_observed_page_selects == 4
    assert large_history.handler_observed_page_selects == 4
    assert small_history.handler_observed_effect_id_selects == 0
    assert large_history.handler_observed_effect_id_selects == 0
    assert small_history.handler_observed_stale_operation_selects == 1
    assert large_history.handler_observed_stale_operation_selects == 1
    assert small_history.handler_rejected_selects == 0
    assert large_history.handler_rejected_selects == 0
    assert small_history.handler_pending_after == 0
    assert large_history.handler_pending_after == 0
    assert small_history.handler_delete_calls == 73
    assert large_history.handler_delete_calls == 73
    assert small_history.seeded_rows_after == 427
    assert large_history.seeded_rows_after == 9_927
    assert small_history.seeded_identity_mismatches == 0
    assert large_history.seeded_identity_mismatches == 0
    # Timings are intentionally recorded but never used as pass/fail criteria.
    assert small_history.recovery_seconds >= 0
    assert large_history.recovery_seconds >= 0
    assert small_history.reopen_seconds >= 0
    assert large_history.reopen_seconds >= 0


def test_scale_benchmark_rejects_store_initialization_history_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SQLRuntimeStore._finish_schema_initialization

    def injected_scan(store: SQLRuntimeStore) -> None:
        original(store)
        list(store.conn.execute("SELECT effect_id FROM external_effects"))

    monkeypatch.setattr(
        SQLRuntimeStore,
        "_finish_schema_initialization",
        injected_scan,
    )
    with pytest.raises(
        AssertionError,
        match="initialization external-effect statement default-deny rejected",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_benchmark_traces_connect_wrapper_prefix_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = recovery_runner.sqlite_storage.sqlite3.connect
    connection_calls = 0

    def connect_with_prefix_select(*args: Any, **kwargs: Any) -> Any:
        nonlocal connection_calls
        connection_calls += 1
        connection = real_connect(*args, **kwargs)
        # The first connection creates/seeds the benchmark database.  The
        # second is the read-only preflight connection on the measured reopen.
        if connection_calls == 2:
            connection.execute(
                "SELECT effect_id FROM external_effects LIMIT 1"
            ).fetchone()
        return connection

    monkeypatch.setattr(
        recovery_runner.sqlite_storage.sqlite3,
        "connect",
        connect_with_prefix_select,
    )
    with pytest.raises(
        AssertionError,
        match="initialization external-effect statement default-deny rejected",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


@pytest.mark.parametrize(
    "execution_path",
    ("_execute", "connection_execute", "cursor_execute"),
)
def test_scale_benchmark_rejects_runtime_assembly_query_bypasses(
    execution_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeBuilder._recover_runtime_state

    def injected_scan(host: Runtime) -> None:
        assert isinstance(host.store, SQLRuntimeStore)
        sql = "SELECT effect_id FROM external_effects"
        if execution_path == "_execute":
            list(host.store._execute(sql))
        elif execution_path == "connection_execute":
            list(host.store.conn.execute(sql))
        else:
            cursor = host.store.conn.cursor()
            try:
                list(cursor.execute(sql))
            finally:
                cursor.close()
        original(host)

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(injected_scan),
    )
    with pytest.raises(AssertionError, match="default-deny rejected"):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_runtime_assembly_select_allowlist_is_exact() -> None:
    assert recovery_runner._is_allowed_handler_select(
        """
        SELECT * FROM external_effects
         WHERE effect_state = ? AND transaction_state IN (?)
         ORDER BY created_at, effect_id LIMIT ?
        """
    )
    assert recovery_runner._is_allowed_handler_select(
        "SELECT * FROM external_effects WHERE effect_id = ?"
    )
    assert not recovery_runner._is_allowed_handler_select(
        "SELECT * FROM external_effects ORDER BY created_at, effect_id LIMIT ?"
    )
    assert not recovery_runner._is_allowed_handler_select(
        "SELECT effect_id FROM external_effects WHERE effect_id = ?"
    )
    assert not recovery_runner._is_allowed_handler_select(
        "SELECT * FROM external_effects WHERE effect_state = ? "
        "AND (created_at > ? OR (created_at = ? AND effect_id > ?)) "
        "ORDER BY created_at, effect_id LIMIT ?"
    )


def test_startup_v3_manifest_schema_probe_allowlist_is_exact() -> None:
    manifest_tables = sorted(_V3_REQUIRED_COLUMNS)

    def manifest_probe(tables: list[str]) -> str:
        names = ", ".join(f"'{name}'" for name in tables)
        return (
            "SELECT name, type FROM sqlite_master "
            f"WHERE name IN ({names})"
        )

    exact_probe = manifest_probe(manifest_tables)
    assert (
        recovery_runner._startup_statement_kind(exact_probe)
        == "v3_manifest_schema_probe"
    )

    near_or_arbitrary_probes = (
        manifest_probe(
            [name for name in manifest_tables if name != "external_effects"]
        ),
        manifest_probe(["external_effects"]),
        manifest_probe([*manifest_tables, "not_a_manifest_table"]),
        manifest_probe(list(reversed(manifest_tables))),
        manifest_probe([name.upper() for name in manifest_tables]),
        exact_probe.replace("SELECT name, type", "SELECT name, type, sql"),
        "SELECT name, type FROM sqlite_master WHERE name = 'external_effects'",
        "SELECT * FROM sqlite_master WHERE sql LIKE '%external_effects%'",
        "SELECT * FROM external_effects",
    )
    assert all(
        recovery_runner._startup_statement_kind(probe) is None
        for probe in near_or_arbitrary_probes
    )


def test_scale_benchmark_rejects_pre_finish_direct_cursor_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SQLRuntimeStore._create_external_effect_indexes

    def create_indexes_with_scan(store: SQLRuntimeStore) -> None:
        original(store)
        cursor = store.conn.cursor()
        try:
            cursor.execute("SELECT effect_id FROM external_effects").fetchone()
        finally:
            cursor.close()

    monkeypatch.setattr(
        SQLRuntimeStore,
        "_create_external_effect_indexes",
        create_indexes_with_scan,
    )
    with pytest.raises(
        AssertionError,
        match="initialization external-effect statement default-deny rejected",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_benchmark_rejects_preflight_external_effect_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = recovery_runner.SQLiteStore._preflight_existing_store

    def preflight_with_scan(
        store: recovery_runner.SQLiteStore,
        database_path: Path,
    ) -> None:
        original(store, database_path)
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro",
            uri=True,
        )
        try:
            connection.execute("SELECT effect_id FROM external_effects").fetchone()
        finally:
            connection.close()

    monkeypatch.setattr(
        recovery_runner.SQLiteStore,
        "_preflight_existing_store",
        preflight_with_scan,
    )
    with pytest.raises(
        AssertionError,
        match="initialization external-effect statement default-deny rejected",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_benchmark_rejects_extra_reviewed_effect_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeBuilder._recover_runtime_state

    def recover_with_extra_reads(host: Runtime) -> None:
        original(host)
        for _ in range(100):
            host.store.conn.execute(
                "SELECT * FROM external_effects WHERE effect_id = ?",
                ("scale-effect-000000000001",),
            ).fetchone()

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(recover_with_extra_reads),
    )
    with pytest.raises(
        AssertionError,
        match="external-effect statement ledger changed",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_benchmark_second_handler_connection_is_locked_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_connect = sqlite3.connect
    original = RuntimeBuilder._recover_runtime_state
    observed_lock_codes: list[int] = []

    def recover_with_second_connection(host: Runtime) -> None:
        original(host)
        database_path = str(
            host.store.conn.execute("PRAGMA database_list").fetchone()[2]
        )
        connection = direct_connect(database_path, timeout=0.0)
        try:
            with pytest.raises(sqlite3.OperationalError) as error:
                connection.execute("SELECT * FROM external_effects").fetchone()
            error_code = getattr(error.value, "sqlite_errorcode", None)
            assert error_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
            observed_lock_codes.append(error_code)
        finally:
            connection.close()

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(recover_with_second_connection),
    )
    result = run_recovery_scale_benchmark(
        total_records=10,
        pending_records=1,
        page_size=1,
    )

    assert observed_lock_codes
    assert result.handler_recovered_records == 1


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "DELETE FROM external_effects",
        "UPDATE external_effects SET effect_state = 'pending' "
        "WHERE effect_id = 'scale-effect-000000000001'",
    ],
)
def test_scale_benchmark_rejects_handler_external_effect_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tamper_sql: str,
) -> None:
    original = RuntimeBuilder._recover_runtime_state

    def recover_with_tamper(host: Runtime) -> None:
        original(host)
        host.store.conn.execute(tamper_sql)
        host.store.conn.commit()

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(recover_with_tamper),
    )
    with pytest.raises(
        AssertionError,
        match="statement default-deny rejected",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_benchmark_external_effect_convergence_is_null_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = recovery_runner._external_effect_convergence

    def convergence_after_null_tamper(
        store: recovery_runner.SQLiteStore,
        **kwargs: int,
    ) -> dict[str, int]:
        store.conn.execute(
            "UPDATE external_effects SET updated_at = NULL "
            "WHERE effect_id = 'scale-effect-000000000001'"
        )
        store.conn.commit()
        return original(store, **kwargs)

    monkeypatch.setattr(
        recovery_runner,
        "_external_effect_convergence",
        convergence_after_null_tamper,
    )
    with pytest.raises(
        AssertionError,
        match="seeded external-effect convergence changed",
    ):
        run_recovery_scale_benchmark(
            total_records=10,
            pending_records=1,
            page_size=1,
        )


def test_scale_cli_writes_machine_readable_structural_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery-scale.json"

    assert main(
        [
            "--profile",
            "ci",
            "--total-records",
            "1001",
            "--pending-records",
            "129",
            "--page-size",
            "64",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["total_records"] == 1_001
    assert payload["recovered_records"] == 129
    assert payload["query_calls"] == 3
    assert payload["handler_observed_page_selects"] == 4
    assert payload["handler_observed_stale_operation_selects"] == 1
    assert payload["handler_rejected_selects"] == 0
    assert payload["timing_is_informational_only"] is True
    metadata = payload["artifact_metadata"]
    assert metadata["schema_version"] == 1
    assert metadata["benchmark_id"] == "external-effect-recovery-scale"
    assert metadata["run_id"].startswith("benchmark_run_")
    assert metadata["started_at"] <= metadata["completed_at"]
    invocation = metadata["invocation"]
    assert invocation["selected_profile"] == "ci"
    assert invocation["classification"] == "custom-overrides"
    assert invocation["named_profile_evidence"] is False
    assert invocation["profile_matches_parameters"] is False
    assert invocation["explicit_overrides"] == {
        "total_records": 1_001,
        "pending_records": 129,
        "page_size": 64,
    }
    assert invocation["effective_parameters"] == {
        "total_records": 1_001,
        "pending_records": 129,
        "page_size": 64,
        "transaction_states": ["prepared"],
    }
    provenance = metadata["provenance"]
    assert provenance["schema_version"] == 1
    assert len(provenance["benchmark_sources"]) == 3
    assert len(provenance["benchmark_source_sha256"]) == 64
    assert "git" in provenance


@pytest.mark.parametrize(
    ("explicit_overrides", "classification", "named_profile_evidence"),
    [
        ({}, "named-profile", True),
        ({"page_size": 500}, "custom-overrides", False),
    ],
)
def test_recovery_artifact_does_not_conflate_overrides_with_named_profile(
    explicit_overrides: dict[str, int],
    classification: str,
    named_profile_evidence: bool,
) -> None:
    identity = new_recovery_run_identity(source_paths=(Path(__file__),))
    defaults = {"total_records": 100, "pending_records": 10, "page_size": 500}

    metadata = build_recovery_artifact_metadata(
        benchmark_id="test-recovery",
        identity=identity,
        selected_profile="ci",
        profile_defaults=defaults,
        explicit_overrides=explicit_overrides,
        effective_parameters={**defaults, "transaction_states": ["prepared"]},
    )

    assert metadata["invocation"]["classification"] == classification
    assert metadata["invocation"]["profile_matches_parameters"] is True
    assert (
        metadata["invocation"]["named_profile_evidence"]
        is named_profile_evidence
    )


def test_recovery_source_snapshot_is_captured_before_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "benchmark.py"
    source.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(recovery_metadata_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(recovery_metadata_module, "__file__", str(source))
    monkeypatch.setattr(
        recovery_metadata_module,
        "_git_provenance",
        lambda: {"available": False},
    )

    identity = recovery_metadata_module.new_recovery_run_identity(
        source_paths=(source,)
    )
    captured_sha256 = identity.source_entries[0].sha256
    source.write_text("version = 2\n", encoding="utf-8")
    metadata = recovery_metadata_module.build_recovery_artifact_metadata(
        benchmark_id="snapshot-timing",
        identity=identity,
        selected_profile="ci",
        profile_defaults={"total_records": 1},
        explicit_overrides={},
        effective_parameters={"total_records": 1},
    )

    assert metadata["provenance"]["benchmark_sources"] == [
        {"path": "benchmark.py", "sha256": captured_sha256}
    ]
    assert captured_sha256 != recovery_metadata_module._sha256_file(source)


@pytest.mark.parametrize(
    ("total_records", "pending_records", "page_size"),
    [
        (0, 0, 1),
        (1, 2, 1),
        (1, 0, 0),
    ],
)
def test_scale_benchmark_rejects_invalid_shapes(
    total_records: int,
    pending_records: int,
    page_size: int,
) -> None:
    with pytest.raises(ValueError):
        run_recovery_scale_benchmark(
            total_records=total_records,
            pending_records=pending_records,
            page_size=page_size,
        )


@pytest.mark.postgres
def test_postgres_recovery_plans_use_matching_composite_indexes() -> None:
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        store = PostgresStore(dsn, config=config)
        try:
            store.conn.execute("SET enable_seqscan = off")
            for transaction_states, expected_index in (
                ((), "idx_external_effects_recovery_state"),
                (("prepared",), "idx_external_effects_recovery_transaction"),
            ):
                captured: list[tuple[str, tuple[object, ...]]] = []
                original_query = store._query

                def tracked_query(
                    sql: str,
                    params: object = (),
                ) -> list[object]:
                    selected_params = tuple(params)  # type: ignore[arg-type]
                    if "FROM external_effects" in sql and "LIMIT" in sql:
                        captured.append((sql, selected_params))
                    return original_query(sql, selected_params)

                store._query = tracked_query  # type: ignore[method-assign]
                store.query_external_effect_recovery(
                    ExternalEffectRecoveryQuery(
                        transaction_states=transaction_states,
                        limit=500,
                    )
                )
                store._query = original_query  # type: ignore[method-assign]
                assert len(captured) == 1
                sql, params = captured[0]
                plan_rows = list(store.conn.execute(f"EXPLAIN {sql}", params))
                details = "\n".join(str(row["QUERY PLAN"]) for row in plan_rows)
                assert expected_index in details
        finally:
            store.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_recovery_scale_{uuid4().hex}"
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
