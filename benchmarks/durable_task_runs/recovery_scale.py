from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import TaskRunStatus
from agent_libos.storage.sql import SQLRuntimeStore
from agent_libos.storage.sqlite import SQLiteStore


_TASK_RUN_INSERT_SQL = """
    INSERT INTO task_runs (
        run_id, spec_schema_version, display_title, image_id,
        launch_options_json, authority_manifest_id, status, revision,
        runtime_epoch, root_pid, active_pid, pause_generation,
        cancel_generation, binding_hash, deadline_at, retention,
        blockers_json, requirement_count, satisfied_requirement_count,
        step_count, completed_step_count, result_ref, created_at, updated_at,
        started_at, completed_at, finalized_at, payloads_purged_at
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?
    )
"""

_RECOVERY_FIRST_PAGE_SQL = " ".join(
    """
    SELECT * FROM task_runs INDEXED BY idx_task_runs_recovery
    WHERE status NOT IN ('succeeded', 'failed', 'cancelled')
    ORDER BY created_at COLLATE BINARY, run_id COLLATE BINARY LIMIT ?
    """.split()
)
_RECOVERY_RESUMED_PAGE_SQL = " ".join(
    """
    SELECT * FROM task_runs INDEXED BY idx_task_runs_recovery
    WHERE status NOT IN ('succeeded', 'failed', 'cancelled')
      AND (created_at, run_id) > (?, ?)
    ORDER BY created_at COLLATE BINARY, run_id COLLATE BINARY LIMIT ?
    """.split()
)
_RECOVERY_POINT_LOOKUP_SQL = "SELECT * FROM task_runs WHERE run_id = ?"
_POINT_LOOKUPS_PER_RECOVERABLE_RUN = 3
_TASK_RUN_TABLE_REFERENCE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:MAIN\.)?"
    r"(?:TASK_RUNS\b|\"TASK_RUNS\"|`TASK_RUNS`|\[TASK_RUNS\])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TaskRunRecoveryScaleProfile:
    total_runs: int
    recoverable_runs: int
    page_size: int


BENCHMARK_PROFILES = {
    "ci": TaskRunRecoveryScaleProfile(
        total_runs=100_000,
        recoverable_runs=1_000,
        page_size=500,
    ),
}


_BENCHMARK_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class TaskRunRecoveryScaleResult:
    schema_version: int
    total_runs: int
    recoverable_runs: int
    recovered_runs: int
    page_size: int
    query_calls: int
    expected_query_calls: int
    raw_rows_fetched: int
    expected_raw_rows_fetched: int
    recovery_index: str
    first_query_plan: tuple[str, ...]
    resumed_query_plan: tuple[str, ...]
    seed_seconds: float
    reopen_seconds: float
    recovery_seconds: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["first_query_plan"] = list(self.first_query_plan)
        payload["resumed_query_plan"] = list(self.resumed_query_plan)
        payload["timing_is_informational_only"] = True
        return payload


@dataclass(frozen=True, slots=True)
class _ObservedTaskRunQuery:
    sql: str
    params: tuple[Any, ...]
    row_count: int
    traced_statements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedRecoveryPage:
    sql: str
    params: tuple[Any, ...]
    row_count: int


def run_task_run_recovery_scale_benchmark(
    *,
    total_runs: int,
    recoverable_runs: int,
    page_size: int,
) -> TaskRunRecoveryScaleResult:
    """Hard-check bounded TaskRun recovery against a large terminal history."""

    _validate_profile(total_runs, recoverable_runs, page_size)
    with _BENCHMARK_LOCK:
        return _run_task_run_recovery_scale_benchmark_locked(
            total_runs=total_runs,
            recoverable_runs=recoverable_runs,
            page_size=page_size,
        )


def _run_task_run_recovery_scale_benchmark_locked(
    *,
    total_runs: int,
    recoverable_runs: int,
    page_size: int,
) -> TaskRunRecoveryScaleResult:
    with TemporaryDirectory(prefix="task-run-recovery-scale-") as directory:
        database = Path(directory) / "runtime.sqlite"
        config = replace(
            DEFAULT_CONFIG,
            task_runs=replace(
                DEFAULT_CONFIG.task_runs,
                plaintext_payloads_enabled=True,
                recovery_page_size=page_size,
                recovery_page_hard_limit=max(recoverable_runs, page_size, 500),
                list_hard_limit=max(recoverable_runs, 1_000),
            ),
        )
        seed_started = time.perf_counter()
        runtime = Runtime.open(database, config=config)
        try:
            created_keys: list[tuple[str, str]] = []
            # Seed one coherent pre-restart snapshot. The benchmark measures
            # recovery query shape, not the cost of 1,000 independent fsyncs.
            with runtime.store.transaction(include_object_payloads=True):
                for index in range(recoverable_runs):
                    summary = runtime.task_runs.create(
                        TaskRunSpecV1(
                            goal={"recovery_index": index},
                            display_title=f"Recoverable Run {index}",
                            image_id="base-agent:v0",
                        ),
                        client_request_id=f"scale-create-{index:06d}",
                    )
                    created_keys.append((summary.created_at, summary.run_id))
            expected_keys = sorted(created_keys)
            expected_ids = [run_id for _, run_id in expected_keys]
        finally:
            runtime.close()

        connection = sqlite3.connect(database)
        try:
            connection.executemany(
                _TASK_RUN_INSERT_SQL,
                _terminal_history_rows(total_runs - recoverable_runs),
            )
            connection.commit()
        finally:
            connection.close()
        seed_seconds = time.perf_counter() - seed_started

        original_query = SQLRuntimeStore._query
        original_sqlite_init = SQLiteStore.__init__
        observations: dict[int, list[_ObservedTaskRunQuery]] = {}
        traced_task_run_queries: dict[int, list[str]] = {}

        def observed_sqlite_init(
            store: SQLiteStore,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            original_sqlite_init(store, *args, **kwargs)
            statements: list[str] = []
            traced_task_run_queries[id(store)] = statements

            def trace(statement: str) -> None:
                normalized = _normalize_sql(statement)
                if _is_task_run_select(normalized):
                    statements.append(normalized)

            store.conn.set_trace_callback(trace)

        def observed_query(
            store: SQLRuntimeStore,
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            selected_params = tuple(params)
            traced = traced_task_run_queries.setdefault(id(store), [])
            trace_offset = len(traced)
            rows = original_query(store, sql, selected_params)
            normalized = _normalize_sql(sql)
            new_traces = tuple(traced[trace_offset:])
            if _is_task_run_select(normalized) or new_traces:
                observations.setdefault(id(store), []).append(
                    _ObservedTaskRunQuery(
                        sql=normalized,
                        params=selected_params,
                        row_count=len(rows),
                        traced_statements=new_traces,
                    )
                )
            return rows

        reopen_started = time.perf_counter()
        runtime: Runtime | None = None
        SQLiteStore.__init__ = observed_sqlite_init
        SQLRuntimeStore._query = observed_query
        try:
            runtime = Runtime.open(database, config=config)
        finally:
            SQLRuntimeStore._query = original_query
            SQLiteStore.__init__ = original_sqlite_init
            if runtime is not None:
                runtime.store.conn.set_trace_callback(None)
        reopen_seconds = time.perf_counter() - reopen_started
        recovery_seconds = reopen_seconds
        if runtime is None:  # pragma: no cover - Runtime.open raised above
            raise AssertionError("TaskRun recovery Runtime did not open")
        try:
            selected_observations = tuple(observations.get(id(runtime.store), ()))
            selected_traces = tuple(
                traced_task_run_queries.get(id(runtime.store), ())
            )
            (
                page_observations,
                first_plan,
                resumed_plan,
            ) = _assert_startup_query_contract(
                runtime.store,
                observations=selected_observations,
                traced_statements=selected_traces,
                expected_keys=expected_keys,
                page_size=page_size,
            )
            query_calls = len(page_observations)
            raw_rows_fetched = sum(
                observation.row_count for observation in page_observations
            )
            recovery_statements = [
                f"{observation.sql} PARAMS={observation.params!r}"
                for observation in page_observations
            ]
            recovery_index = "idx_task_runs_recovery"
            recovered_summaries = tuple(runtime.recovered_task_runs)
            recovered_total = int(runtime.recovered_task_run_count)
            if recoverable_runs:
                all_recovered = tuple(
                    runtime.task_runs.list(
                        statuses=(TaskRunStatus.QUEUED,),
                        limit=recoverable_runs,
                    ).records
                )
            else:
                all_recovered = ()
            recovered = [
                summary.run_id
                for summary in sorted(
                    all_recovered,
                    key=lambda item: (item.created_at, item.run_id),
                )
            ]
            expected_sample = expected_ids[: config.task_runs.recovery_sample_limit]
            if [summary.run_id for summary in recovered_summaries] != expected_sample:
                raise AssertionError(
                    "TaskRun startup recovery sample is not the exact ordered prefix"
                )
            if recovered_total != recoverable_runs:
                raise AssertionError(
                    "TaskRun startup recovery total did not converge"
                )
            if any(
                summary.status is not TaskRunStatus.QUEUED
                or summary.blockers
                or summary.root_pid is None
                for summary in all_recovered
            ):
                raise AssertionError(
                    "TaskRun startup recovery did not preserve safe queued projections"
                )
            if any(
                runtime.process.get(summary.root_pid or "").resource_usage.llm_calls
                != 0
                for summary in all_recovered
            ):
                raise AssertionError("TaskRun startup recovery dispatched model work")
        finally:
            runtime.close()

    pages_per_phase = max(1, math.ceil(recoverable_runs / page_size))
    # Startup intentionally performs two bounded passes in a fixed order:
    # zero-write payload validation, then epoch claim/projection recovery.
    expected_query_calls = pages_per_phase * 2
    raw_rows_per_phase = (
        0
        if recoverable_runs == 0
        else recoverable_runs + pages_per_phase - 1
    )
    expected_raw_rows = raw_rows_per_phase * 2
    if recovered != expected_ids:
        raise AssertionError("TaskRun recovery did not return the exact ordered set")
    if query_calls != expected_query_calls:
        raise AssertionError(
            f"TaskRun recovery used {query_calls} pages; expected {expected_query_calls}; "
            f"statements={recovery_statements!r}"
        )
    if raw_rows_fetched != expected_raw_rows:
        raise AssertionError(
            "TaskRun recovery fetched an unbounded or unexpected number of rows: "
            f"{raw_rows_fetched} != {expected_raw_rows}"
        )
    if len(recovery_statements) != expected_query_calls:
        raise AssertionError("TaskRun recovery query ledger is incomplete")

    return TaskRunRecoveryScaleResult(
        schema_version=1,
        total_runs=total_runs,
        recoverable_runs=recoverable_runs,
        recovered_runs=recovered_total,
        page_size=page_size,
        query_calls=query_calls,
        expected_query_calls=expected_query_calls,
        raw_rows_fetched=raw_rows_fetched,
        expected_raw_rows_fetched=expected_raw_rows,
        recovery_index=recovery_index,
        first_query_plan=first_plan,
        resumed_query_plan=resumed_plan,
        seed_seconds=seed_seconds,
        reopen_seconds=reopen_seconds,
        recovery_seconds=recovery_seconds,
    )


def _assert_startup_query_contract(
    store: SQLiteStore,
    *,
    observations: tuple[_ObservedTaskRunQuery, ...],
    traced_statements: tuple[str, ...],
    expected_keys: list[tuple[str, str]],
    page_size: int,
) -> tuple[
    tuple[_ObservedTaskRunQuery, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    allowed_sql = {
        _RECOVERY_FIRST_PAGE_SQL,
        _RECOVERY_RESUMED_PAGE_SQL,
        _RECOVERY_POINT_LOOKUP_SQL,
    }
    unexpected = [
        observation.sql
        for observation in observations
        if observation.sql not in allowed_sql
    ]
    if unexpected:
        raise AssertionError(
            "TaskRun startup query default-deny rejected an unreviewed statement: "
            f"{unexpected!r}"
        )

    accounted_traces = Counter(
        statement
        for observation in observations
        for statement in observation.traced_statements
    )
    direct_traces: list[str] = []
    for statement in traced_statements:
        if accounted_traces[statement] > 0:
            accounted_traces[statement] -= 1
        else:
            direct_traces.append(statement)
    if any(accounted_traces.values()):  # pragma: no cover - trace ordering contract
        raise AssertionError(
            "TaskRun startup query accounting recorded a statement absent from "
            "the SQLite execution trace"
        )
    for observation in observations:
        if len(observation.traced_statements) != 1:
            raise AssertionError(
                "TaskRun startup query default-deny rejected extra TaskRun SELECTs "
                "inside one repository query"
            )
        expected_trace = _expand_sql(observation.sql, observation.params)
        if observation.traced_statements[0] != expected_trace:
            raise AssertionError(
                "TaskRun startup query default-deny rejected an executed SQL shape "
                "that differs from the repository statement"
            )

    expected_pages = _expected_recovery_pages(expected_keys, page_size)
    page_observations = tuple(
        observation
        for observation in observations
        if observation.sql
        in {_RECOVERY_FIRST_PAGE_SQL, _RECOVERY_RESUMED_PAGE_SQL}
    )
    if len(page_observations) != len(expected_pages):
        raise AssertionError(
            "TaskRun recovery page count changed: "
            f"{len(page_observations)} != {len(expected_pages)}"
        )
    for page_number, (observed, expected) in enumerate(
        zip(page_observations, expected_pages, strict=True),
        start=1,
    ):
        if (
            observed.sql != expected.sql
            or observed.params != expected.params
            or observed.row_count != expected.row_count
        ):
            raise AssertionError(
                "TaskRun recovery exact keyset page contract changed at page "
                f"{page_number}: observed={(observed.sql, observed.params, observed.row_count)!r} "
                f"expected={(expected.sql, expected.params, expected.row_count)!r}"
            )

    point_observations = tuple(
        observation
        for observation in observations
        if observation.sql == _RECOVERY_POINT_LOOKUP_SQL
    )
    expected_point_counts: Counter[str] = Counter(
        {
            run_id: _POINT_LOOKUPS_PER_RECOVERABLE_RUN
            for _, run_id in expected_keys
        }
    )
    observed_point_counts: Counter[str] = Counter()
    for observation in point_observations:
        if (
            len(observation.params) != 1
            or not isinstance(observation.params[0], str)
            or observation.row_count != 1
        ):
            raise AssertionError(
                "TaskRun startup point lookup shape or row work changed"
            )
        observed_point_counts[observation.params[0]] += 1
    direct_point_sql = {
        _expand_sql(_RECOVERY_POINT_LOOKUP_SQL, (run_id,)): run_id
        for _, run_id in expected_keys
    }
    direct_point_counts: Counter[str] = Counter()
    unexpected_direct = []
    for statement in direct_traces:
        run_id = direct_point_sql.get(statement)
        if run_id is None:
            unexpected_direct.append(statement)
        else:
            direct_point_counts[run_id] += 1
    if unexpected_direct:
        raise AssertionError(
            "TaskRun startup query default-deny rejected a direct or full-table "
            f"TaskRun SELECT: {unexpected_direct!r}"
        )
    if any(count != 1 for count in direct_point_counts.values()) or set(
        direct_point_counts
    ) != set(expected_point_counts):
        raise AssertionError(
            "TaskRun startup direct epoch-claim point lookup contract changed: "
            f"actual={dict(direct_point_counts)!r}"
        )
    combined_point_counts = observed_point_counts + direct_point_counts
    if combined_point_counts != expected_point_counts:
        raise AssertionError(
            "TaskRun startup point lookup query work changed: "
            f"actual={dict(combined_point_counts)!r} "
            f"expected={dict(expected_point_counts)!r}"
        )

    expected_query_count = len(expected_pages) + sum(expected_point_counts.values())
    expected_row_count = sum(page.row_count for page in expected_pages) + sum(
        expected_point_counts.values()
    )
    if (
        len(observations) + len(direct_traces) != expected_query_count
        or len(traced_statements) != expected_query_count
        or sum(observation.row_count for observation in observations)
        + len(direct_traces)
        != expected_row_count
    ):
        raise AssertionError(
            "TaskRun startup total query/row work changed: "
            f"queries={len(observations) + len(direct_traces)}/{expected_query_count}, "
            f"traces={len(traced_statements)}/{expected_query_count}, "
            f"rows={sum(item.row_count for item in observations) + len(direct_traces)}"
            f"/{expected_row_count}"
        )

    _require_partial_recovery_index(store)
    page_plans = tuple(
        _query_plan_for_observed_statement(store, observation.traced_statements[0])
        for observation in page_observations
    )
    for plan in page_plans:
        if not any(
            "USING INDEX idx_task_runs_recovery" in detail for detail in plan
        ):
            raise AssertionError(
                "an actually observed TaskRun recovery page did not use the "
                "partial idx_task_runs_recovery index"
            )
    first_plan = next(
        (
            plan
            for observation, plan in zip(
                page_observations,
                page_plans,
                strict=True,
            )
            if observation.sql == _RECOVERY_FIRST_PAGE_SQL
        ),
        (),
    )
    resumed_plan = next(
        (
            plan
            for observation, plan in zip(
                page_observations,
                page_plans,
                strict=True,
            )
            if observation.sql == _RECOVERY_RESUMED_PAGE_SQL
        ),
        (),
    )
    return page_observations, first_plan, resumed_plan


def _expected_recovery_pages(
    expected_keys: list[tuple[str, str]],
    page_size: int,
) -> tuple[_ExpectedRecoveryPage, ...]:
    pages_per_phase = max(1, math.ceil(len(expected_keys) / page_size))
    expected: list[_ExpectedRecoveryPage] = []
    for _phase in range(2):
        for page_index in range(pages_per_phase):
            start = page_index * page_size
            remaining = max(0, len(expected_keys) - start)
            row_count = min(page_size + 1, remaining)
            if page_index == 0:
                sql = _RECOVERY_FIRST_PAGE_SQL
                params: tuple[Any, ...] = (page_size + 1,)
            else:
                cursor = expected_keys[start - 1]
                sql = _RECOVERY_RESUMED_PAGE_SQL
                params = (cursor[0], cursor[1], page_size + 1)
            expected.append(
                _ExpectedRecoveryPage(
                    sql=sql,
                    params=params,
                    row_count=row_count,
                )
            )
    return tuple(expected)


def _is_task_run_select(sql: str) -> bool:
    return (
        re.search(r"\bSELECT\b", sql, re.IGNORECASE) is not None
        and _TASK_RUN_TABLE_REFERENCE_RE.search(sql) is not None
    )


def _normalize_sql(sql: str) -> str:
    return " ".join(str(sql).split()).rstrip(";")


def _expand_sql(sql: str, params: tuple[Any, ...]) -> str:
    fragments = sql.split("?")
    if len(fragments) != len(params) + 1:
        raise AssertionError("TaskRun startup query placeholder contract changed")
    expanded = fragments[0]
    for value, fragment in zip(params, fragments[1:], strict=True):
        expanded += _sqlite_literal(value) + fragment
    return _normalize_sql(expanded)


def _sqlite_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if type(value) is int:
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise AssertionError(
        f"TaskRun startup query parameter type changed: {type(value).__name__}"
    )


def _require_partial_recovery_index(store: SQLiteStore) -> None:
    indexes = {
        str(row["name"]): row
        for row in store.conn.execute("PRAGMA index_list(task_runs)")
    }
    recovery = indexes.get("idx_task_runs_recovery")
    if recovery is None or int(recovery["partial"]) != 1:
        raise AssertionError(
            "idx_task_runs_recovery is missing or is no longer a partial index"
        )


def _query_plan_for_observed_statement(
    store: SQLiteStore,
    statement: str,
) -> tuple[str, ...]:
    rows = store.conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
    return tuple(str(row["detail"]) for row in rows)


def _validate_profile(total_runs: int, recoverable_runs: int, page_size: int) -> None:
    if type(total_runs) is not int or total_runs < 0:
        raise ValueError("total_runs must be a non-negative integer")
    if (
        type(recoverable_runs) is not int
        or recoverable_runs < 0
        or recoverable_runs > total_runs
    ):
        raise ValueError("recoverable_runs must be within total_runs")
    if type(page_size) is not int or page_size <= 0 or page_size > 500:
        raise ValueError("page_size must be between 1 and the recovery gate maximum 500")


def _terminal_history_rows(total_runs: int) -> list[tuple[Any, ...]]:
    created_at = "2035-01-01T00:00:00.000000+00:00"
    terminal_at = "2035-01-02T00:00:00.000000+00:00"
    rows: list[tuple[Any, ...]] = []
    for index in range(total_runs):
        run_id = f"run-history-{index:06d}"
        rows.append(
            _task_run_row(
                run_id=run_id,
                status="succeeded",
                root_pid=f"pid-history-{index:06d}",
                created_at=created_at,
                completed_at=terminal_at,
            )
        )
    return rows


def _task_run_row(
    *,
    run_id: str,
    status: str,
    root_pid: str | None,
    created_at: str,
    completed_at: str | None,
) -> tuple[Any, ...]:
    return (
        run_id,
        1,
        run_id,
        "base-agent:v0",
        "{}",
        None,
        status,
        0,
        1,
        root_pid,
        root_pid,
        0,
        0,
        "0" * 64,
        None,
        "permanent",
        "[]",
        0,
        0,
        0,
        0,
        None,
        created_at,
        completed_at or created_at,
        None,
        completed_at,
        completed_at,
        None,
    )
