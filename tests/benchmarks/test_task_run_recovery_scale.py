from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_libos.runtime import RuntimeBuilder
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage.sql import SQLRuntimeStore
from benchmarks.durable_task_runs import (
    BENCHMARK_PROFILES,
    run_task_run_recovery_scale_benchmark,
)
from experiments.run_task_run_recovery_scale import run


def test_ci_profile_is_the_release_recovery_gate() -> None:
    profile = BENCHMARK_PROFILES["ci"]
    assert profile.total_runs == 100_000
    assert profile.recoverable_runs == 1_000
    assert profile.page_size == 500


def test_recovery_work_is_bounded_by_recoverable_pages_not_history() -> None:
    small = run_task_run_recovery_scale_benchmark(
        total_runs=100,
        recoverable_runs=73,
        page_size=32,
    )
    large = run_task_run_recovery_scale_benchmark(
        total_runs=10_000,
        recoverable_runs=73,
        page_size=32,
    )

    assert small.query_calls == large.query_calls == 6
    assert small.raw_rows_fetched == large.raw_rows_fetched == 150
    assert small.recovered_runs == large.recovered_runs == 73
    assert small.recovery_index == large.recovery_index == "idx_task_runs_recovery"
    assert small.recovery_seconds >= 0
    assert large.recovery_seconds >= 0


def test_recovery_gate_default_denies_an_unaccounted_full_table_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_query = SQLRuntimeStore._query

    def query_with_full_scan(
        store: SQLRuntimeStore,
        sql: str,
        params: Any = (),
    ) -> list[Any]:
        if "FROM task_runs INDEXED BY idx_task_runs_recovery" in sql:
            list(store.conn.execute("SELECT * FROM task_runs"))
        return original_query(store, sql, params)

    monkeypatch.setattr(SQLRuntimeStore, "_query", query_with_full_scan)

    with pytest.raises(AssertionError, match="default-deny rejected extra"):
        run_task_run_recovery_scale_benchmark(
            total_runs=100,
            recoverable_runs=10,
            page_size=5,
        )


def test_recovery_gate_default_denies_a_direct_startup_full_table_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_recovery = RuntimeBuilder._recover_runtime_state

    def recover_with_full_scan(host: Runtime) -> None:
        list(host.store.conn.execute("SELECT * FROM task_runs"))
        original_recovery(host)

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(recover_with_full_scan),
    )

    with pytest.raises(AssertionError, match="direct or full-table"):
        run_task_run_recovery_scale_benchmark(
            total_runs=100,
            recoverable_runs=10,
            page_size=5,
        )


def test_recovery_gate_rejects_executed_offset_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_query = SQLRuntimeStore._query

    def query_with_offset(
        store: SQLRuntimeStore,
        sql: str,
        params: Any = (),
    ) -> list[Any]:
        if "FROM task_runs INDEXED BY idx_task_runs_recovery" in sql:
            sql = f"{sql.rstrip()} OFFSET 0"
        return original_query(store, sql, params)

    monkeypatch.setattr(SQLRuntimeStore, "_query", query_with_offset)

    with pytest.raises(AssertionError, match="executed SQL shape"):
        run_task_run_recovery_scale_benchmark(
            total_runs=100,
            recoverable_runs=10,
            page_size=5,
        )


def test_release_profile_converges_in_two_pages_per_startup_phase() -> None:
    profile = BENCHMARK_PROFILES["ci"]
    result = run_task_run_recovery_scale_benchmark(
        total_runs=profile.total_runs,
        recoverable_runs=profile.recoverable_runs,
        page_size=profile.page_size,
    )

    assert result.recovered_runs == 1_000
    assert result.query_calls == result.expected_query_calls == 4
    assert result.raw_rows_fetched == result.expected_raw_rows_fetched == 2_002
    assert "idx_task_runs_recovery" in "\n".join(result.first_query_plan)
    assert "idx_task_runs_recovery" in "\n".join(result.resumed_query_plan)


def test_scale_artifact_is_atomic_machine_readable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.durable_task_runs.recovery_scale import (
        TaskRunRecoveryScaleProfile,
    )

    monkeypatch.setitem(
        BENCHMARK_PROFILES,
        "ci",
        TaskRunRecoveryScaleProfile(
            total_runs=100,
            recoverable_runs=10,
            page_size=5,
        ),
    )
    output = tmp_path / "task-run-recovery.json"
    payload = run("ci", output)

    assert payload["total_runs"] == 100
    assert payload["recoverable_runs"] == 10
    assert payload["timing_is_informational_only"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("total", "recoverable", "page_size"),
    [(-1, 0, 1), (1, 2, 1), (1, 0, 0), (1, 0, 501)],
)
def test_scale_gate_rejects_invalid_or_unbounded_profiles(
    total: int,
    recoverable: int,
    page_size: int,
) -> None:
    with pytest.raises(ValueError):
        run_task_run_recovery_scale_benchmark(
            total_runs=total,
            recoverable_runs=recoverable,
            page_size=page_size,
        )


def test_release_workflow_runs_crash_and_recovery_scale_gates() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/test.yml"
    ).read_text(encoding="utf-8")

    assert "run_task_run_crash_matrix.py" in workflow
    assert "run_task_run_recovery_scale.py" in workflow
    assert "task-run-recovery-scale-ci.json" in workflow
