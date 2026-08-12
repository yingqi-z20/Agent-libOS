from __future__ import annotations

import contextlib
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlsplit

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import ExternalEffectRecoveryQuery
from agent_libos.runtime import RuntimeBuilder
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import sqlite as sqlite_storage
from agent_libos.storage.sqlite import SQLiteStore
from agent_libos.utils.serde import dumps


_RECOVERY_INSERT_SQL = """
    INSERT INTO external_effects (
        effect_id, record_id, event_id, pid, provider, operation, target,
        rollback_class, rollback_status, state_mutation, information_flow,
        provider_metadata_json, created_at, effect_state, transaction_state,
        canonical_args_hash, idempotency_key, provider_receipt_json, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_RECOVERY_OPERATION_INSERT_SQL = """
    INSERT INTO operations (
        operation_id, root_operation_id, parent_operation_id, kind, name,
        actor, pid, state, outcome, expected_roles_json, metadata_json,
        runtime_publication_id, started_at, updated_at, completed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_RECOVERY_EVIDENCE_INSERT_SQL = """
    INSERT INTO operation_evidence (
        link_id, operation_id, evidence_type, evidence_id, role, created_at,
        metadata_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_SQL_TEXT_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_INTEGER_LITERAL_RE = re.compile(r"(?<![A-Z0-9_?])-?\d+(?![A-Z0-9_?])")
_HANDLER_RECOVERY_SELECT_RE = re.compile(
    r"SELECT \* FROM EXTERNAL_EFFECTS WHERE EFFECT_STATE = \?"
    r"(?: AND TRANSACTION_STATE IN \(\?(?:, \?)*\))?"
    r"(?: AND \(CREATED_AT, EFFECT_ID\) > \(\?, \?\))?"
    r" ORDER BY CREATED_AT, EFFECT_ID LIMIT \?"
)
_HANDLER_EFFECT_ID_SELECT = (
    "SELECT * FROM EXTERNAL_EFFECTS WHERE EFFECT_ID = ?"
)
_HANDLER_STALE_OPERATION_INDEX_SELECT = """
    WITH RECURSIVE
    uncertain_effects(effect_id) AS MATERIALIZED (
        SELECT effect_id
          FROM external_effects
         WHERE effect_state = ?
        UNION
        SELECT effect_id
          FROM external_effects
         WHERE transaction_state = ?
    ),
    unknown_nodes(
        root_operation_id,
        operation_id,
        parent_operation_id
    ) AS MATERIALIZED (
        SELECT DISTINCT operation.root_operation_id,
               operation.operation_id,
               operation.parent_operation_id
          FROM uncertain_effects
          CROSS JOIN operation_evidence AS evidence
          CROSS JOIN operations AS operation
         WHERE evidence.evidence_type = 'external_effect'
           AND evidence.evidence_id = uncertain_effects.effect_id
           AND operation.operation_id = evidence.operation_id
    ),
    ancestors(
        root_operation_id,
        operation_id,
        parent_operation_id
    ) AS (
        SELECT root_operation_id,
               operation_id,
               parent_operation_id
          FROM unknown_nodes
        UNION
        SELECT ancestors.root_operation_id,
               parent.operation_id,
               parent.parent_operation_id
          FROM ancestors
          JOIN operations AS parent
            ON parent.operation_id = ancestors.parent_operation_id
           AND parent.root_operation_id = ancestors.root_operation_id
    )
    INSERT INTO agent_libos_stale_operation_recovery_unknown (operation_id)
    SELECT DISTINCT operation.operation_id
      FROM ancestors
      JOIN operations AS operation
        ON operation.operation_id = ancestors.operation_id
     WHERE operation.state = ?
"""
_HANDLER_RECOVERY_DELETE_RE = re.compile(
    r"DELETE FROM EXTERNAL_EFFECTS WHERE EFFECT_ID = \?"
    r" AND RECORD_ID IS NULL AND EVENT_ID IS NULL"
    r" AND ROLLBACK_CLASS = \? AND ROLLBACK_STATUS = \?"
    r" AND EFFECT_STATE = \?"
)
_EXTERNAL_EFFECT_SCHEMA_INDEXES = frozenset(
    {
        "idx_external_effects_created",
        "idx_external_effects_pid_created",
        "idx_external_effects_recovery_state",
        "idx_external_effects_recovery_transaction",
        "idx_external_effects_transaction_state",
        "idx_external_effects_retention_eligible",
        "idx_external_effects_pid_idempotency",
    }
)
_EXTERNAL_EFFECT_SCHEMA_CATALOG_INDEXES = (
    _EXTERNAL_EFFECT_SCHEMA_INDEXES
    | {"sqlite_autoindex_external_effects_1"}
)
_SQLITE_SCHEMA_CATALOG_PROBE_RE = re.compile(
    r"SELECT NAME, TYPE FROM SQLITE_MASTER WHERE NAME IN "
    r"\(\?(?:, \?)+\)"
)
_SQLITE_SCHEMA_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]*")
_GLOBAL_PATCH_SCOPE_LOCK = threading.RLock()


def _connect_with_trace_installed(
    connect: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    register_connection: Callable[[Any], None],
) -> Any:
    """Install tracing in the connection factory, before ``connect`` returns.

    A wrapper around ``sqlite3.connect`` is allowed to run initialization SQL
    before it returns the connection.  Attaching the callback only after that
    wrapper returns leaves an unmeasured prefix.  Chaining the requested
    factory closes that gap while preserving custom connection subclasses.
    """

    selected_kwargs = dict(kwargs)
    selected_factory = selected_kwargs.pop(
        "factory",
        sqlite_storage.sqlite3.Connection,
    )

    def traced_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
        connection = selected_factory(*factory_args, **factory_kwargs)
        register_connection(connection)
        return connection

    selected_kwargs["factory"] = traced_factory
    connection = connect(*args, **selected_kwargs)
    # A non-conforming wrapper may replace the factory-produced connection.
    # Register the returned object as a fallback without double-registering it.
    register_connection(connection)
    return connection


def _connect_targets_database(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    database_path: Path,
) -> bool:
    raw_database = args[0] if args else kwargs.get("database")
    if isinstance(raw_database, os.PathLike):
        raw_database = os.fspath(raw_database)
    if not isinstance(raw_database, str) or raw_database == ":memory:":
        return False
    if raw_database.startswith("file:"):
        parsed = urlsplit(raw_database)
        raw_database = unquote(parsed.path or parsed.netloc)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", raw_database):
            raw_database = raw_database[1:]
    try:
        selected = Path(raw_database).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return os.path.normcase(str(selected)) == os.path.normcase(
        str(database_path.resolve(strict=False))
    )


class _ObservedSQLiteStore(SQLiteStore):
    """Benchmark-specific SQLite store type."""


@dataclass(frozen=True)
class RecoveryScaleProfile:
    """A repeatable population and pending-work shape."""

    total_records: int
    pending_records: int
    page_size: int


BENCHMARK_PROFILES = {
    "ci": RecoveryScaleProfile(
        total_records=100_000,
        pending_records=1_000,
        page_size=500,
    ),
    "million": RecoveryScaleProfile(
        total_records=1_000_000,
        pending_records=10_000,
        page_size=500,
    ),
}


@dataclass(frozen=True)
class RecoveryScaleResult:
    schema_version: int
    total_records: int
    pending_records: int
    recovered_records: int
    page_size: int
    query_calls: int
    expected_query_calls: int
    raw_rows_fetched: int
    expected_raw_rows_fetched: int
    recovery_work_units: int
    expected_recovery_work_units: int
    handler_recovered_records: int
    handler_sample_records: int
    handler_query_calls: int
    expected_handler_query_calls: int
    handler_raw_rows_fetched: int
    expected_handler_raw_rows_fetched: int
    startup_external_effect_statements: int
    startup_schema_probe_calls: int
    startup_ddl_calls: int
    handler_observed_statements: int
    handler_observed_selects: int
    handler_observed_page_selects: int
    handler_observed_effect_id_selects: int
    handler_observed_stale_operation_selects: int
    handler_rejected_selects: int
    handler_delete_calls: int
    handler_pending_after: int
    seeded_rows_after: int
    seeded_identity_mismatches: int
    seeded_convergence_mismatches: int
    recovery_index: str
    query_plan_first: tuple[str, ...]
    query_plan_resumed: tuple[str, ...]
    seed_seconds: float
    reopen_seconds: float
    recovery_seconds: float

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["query_plan_first"] = list(self.query_plan_first)
        result["query_plan_resumed"] = list(self.query_plan_resumed)
        result["timing_is_informational_only"] = True
        return result


def run_recovery_scale_benchmark(
    *,
    total_records: int,
    pending_records: int,
    page_size: int,
    transaction_states: tuple[str, ...] = ("prepared",),
) -> RecoveryScaleResult:
    """Run one identity-filtered benchmark in the serialized patch scope."""

    with _GLOBAL_PATCH_SCOPE_LOCK:
        return _run_recovery_scale_benchmark_locked(
            total_records=total_records,
            pending_records=pending_records,
            page_size=page_size,
            transaction_states=transaction_states,
        )


def _run_recovery_scale_benchmark_locked(
    *,
    total_records: int,
    pending_records: int,
    page_size: int,
    transaction_states: tuple[str, ...] = ("prepared",),
) -> RecoveryScaleResult:
    """Exercise the real keyset API and validate structural, not timing, bounds.

    Rows are inserted directly because fixture construction is not part of the
    recovery path being measured.  The scan itself goes exclusively through
    ``ExternalEffectRecoveryQuery`` and the store's public paging method.
    """

    _validate_shape(
        total_records=total_records,
        pending_records=pending_records,
        page_size=page_size,
    )
    temporary_directory = TemporaryDirectory(
        prefix="agent-libos-external-effect-recovery-"
    )
    database_path = Path(temporary_directory.name) / "runtime.sqlite"
    store: _ObservedSQLiteStore | None = None
    runtime: Runtime | None = None
    original_connect = sqlite_storage.sqlite3.connect
    try:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                external_effect_recovery_page_size=page_size,
                external_effect_recovery_page_hard_limit=max(page_size, 5_000),
            )
        )
        store = _ObservedSQLiteStore(database_path, config=config)
        seed_started = time.perf_counter()
        _seed_external_effect_history(
            store,
            total_records=total_records,
            pending_records=pending_records,
        )
        seed_seconds = time.perf_counter() - seed_started
        store.close()
        store = None

        startup_statements: list[str] = []
        startup_connections: list[Any] = []
        startup_connection_ids: set[int] = set()

        def register_startup_connection(connection: Any) -> None:
            if id(connection) in startup_connection_ids:
                return
            startup_connection_ids.add(id(connection))
            startup_connections.append(connection)
            connection.set_trace_callback(
                lambda sql: startup_statements.append(sql)
                if _is_external_effect_statement(sql)
                else None
            )

        def tracked_connect(*args: Any, **kwargs: Any) -> Any:
            if not _connect_targets_database(args, kwargs, database_path):
                return original_connect(*args, **kwargs)
            return _connect_with_trace_installed(
                original_connect,
                args,
                kwargs,
                register_connection=register_startup_connection,
            )

        sqlite_storage.sqlite3.connect = tracked_connect
        try:
            reopen_started = time.perf_counter()
            store = _ObservedSQLiteStore(database_path, config=config)
            reopen_seconds = time.perf_counter() - reopen_started
        finally:
            sqlite_storage.sqlite3.connect = original_connect
            for startup_connection in startup_connections:
                with contextlib.suppress(Exception):
                    startup_connection.set_trace_callback(None)

        startup_shapes = tuple(
            _startup_statement_kind(sql) for sql in startup_statements
        )
        rejected_startup_statements = tuple(
            sql
            for sql, shape in zip(startup_statements, startup_shapes, strict=True)
            if shape is None
        )
        if rejected_startup_statements:
            raise AssertionError(
                "store initialization external-effect statement default-deny rejected: "
                f"{rejected_startup_statements[0]}"
            )
        startup_counts = Counter(
            shape for shape in startup_shapes if shape is not None
        )
        _assert_startup_statement_contract(startup_counts)

        query_calls = 0
        raw_rows_fetched = 0
        captured_queries: list[tuple[str, tuple[Any, ...]]] = []
        original_query = store._query

        def tracked_query(
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            nonlocal query_calls, raw_rows_fetched
            selected_params = tuple(params)
            rows = original_query(sql, selected_params)
            if (
                "FROM external_effects" in sql
                and "ORDER BY created_at, effect_id" in sql
                and "LIMIT" in sql
            ):
                query_calls += 1
                raw_rows_fetched += len(rows)
                if len(captured_queries) < 2:
                    captured_queries.append((sql, selected_params))
            return rows

        store._query = tracked_query  # type: ignore[method-assign]
        recovery_started = time.perf_counter()
        recovered_records = _consume_recovery_pages(
            store,
            page_size=page_size,
            transaction_states=transaction_states,
        )
        recovery_seconds = time.perf_counter() - recovery_started
        store._query = original_query  # type: ignore[method-assign]

        expected_query_calls = max(1, math.ceil(pending_records / page_size))
        expected_raw_rows_fetched = pending_records + max(
            0,
            expected_query_calls - 1,
        )
        expected_recovery_work_units = (
            expected_raw_rows_fetched + expected_query_calls
        )
        expected_index = (
            "idx_external_effects_recovery_transaction"
            if transaction_states
            else "idx_external_effects_recovery_state"
        )
        _require_external_effect_index_contract(
            store,
            expected_index=expected_index,
        )
        plans = tuple(
            _sqlite_query_plan(store, sql, params)
            for sql, params in captured_queries
        )
        first_plan = plans[0] if plans else ()
        resumed_plan = plans[1] if len(plans) > 1 else ()

        handler_query_calls = 0
        handler_raw_rows_fetched = 0
        original_query = store._query

        def tracked_handler_query(
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            nonlocal handler_query_calls, handler_raw_rows_fetched
            if _is_external_effect_select(sql) and not _is_allowed_handler_select(sql):
                raise AssertionError(
                    "Runtime assembly external-effect SELECT default-deny rejected: "
                    f"{sql}"
                )
            rows = original_query(sql, tuple(params))
            if _handler_select_kind(sql) == "recovery_page":
                handler_query_calls += 1
                handler_raw_rows_fetched += len(rows)
            return rows

        def reject_full_history_scan(*_args: Any, **_kwargs: Any) -> list[Any]:
            raise AssertionError("Runtime recovery loaded full external-effect history")

        store._query = tracked_handler_query  # type: ignore[method-assign]
        store.list_external_effects = reject_full_history_scan  # type: ignore[method-assign]
        handler_observed_statements: list[str] = []
        handler_connections: list[Any] = []
        handler_connection_ids: set[int] = set()

        def register_handler_connection(connection: Any) -> None:
            if id(connection) in handler_connection_ids:
                return
            handler_connection_ids.add(id(connection))
            handler_connections.append(connection)
            connection.set_trace_callback(
                lambda sql: handler_observed_statements.append(sql)
                if _is_external_effect_statement(sql)
                else None
            )

        def tracked_handler_connect(*args: Any, **kwargs: Any) -> Any:
            if not _connect_targets_database(args, kwargs, database_path):
                return original_connect(*args, **kwargs)
            return _connect_with_trace_installed(
                original_connect,
                args,
                kwargs,
                register_connection=register_handler_connection,
            )

        store.conn.set_trace_callback(
            lambda sql: handler_observed_statements.append(sql)
            if _is_external_effect_statement(sql)
            else None
        )
        sqlite_storage.sqlite3.connect = tracked_handler_connect
        try:
            runtime = RuntimeBuilder.configured(Runtime, config=config).from_store(store)
        finally:
            sqlite_storage.sqlite3.connect = original_connect
            store.conn.set_trace_callback(None)
            for handler_connection in handler_connections:
                with contextlib.suppress(Exception):
                    handler_connection.set_trace_callback(None)

        handler_statement_kinds = tuple(
            _handler_statement_kind(sql) for sql in handler_observed_statements
        )
        rejected_handler_statements = tuple(
            sql
            for sql, kind in zip(
                handler_observed_statements,
                handler_statement_kinds,
                strict=True,
            )
            if kind is None
        )
        if rejected_handler_statements:
            raise AssertionError(
                "Runtime assembly external-effect statement default-deny rejected: "
                f"{rejected_handler_statements[0]}"
            )
        handler_summary = runtime.recovered_prepared_operations
        provider_summary = runtime.reconciled_external_effects
        handler_pending_after = int(
            original_query(
                "SELECT COUNT(*) AS count FROM external_effects WHERE effect_state = ?",
                ("pending",),
            )[0]["count"]
        )
        expected_handler_query_calls = expected_query_calls + 1
        handler_counts = Counter(
            kind for kind in handler_statement_kinds if kind is not None
        )
        _assert_handler_statement_contract(
            handler_counts,
            pending_records=pending_records,
            expected_handler_query_calls=expected_handler_query_calls,
        )
        convergence = _external_effect_convergence(
            store,
            total_records=total_records,
            pending_records=pending_records,
        )
        if (
            convergence["rows_after"] != total_records - pending_records
            or convergence["identity_mismatches"] != 0
            or convergence["convergence_mismatches"] != 0
        ):
            raise AssertionError(
                "seeded external-effect convergence changed: "
                f"{convergence}"
            )

        result = RecoveryScaleResult(
            schema_version=3,
            total_records=total_records,
            pending_records=pending_records,
            recovered_records=recovered_records,
            page_size=page_size,
            query_calls=query_calls,
            expected_query_calls=expected_query_calls,
            raw_rows_fetched=raw_rows_fetched,
            expected_raw_rows_fetched=expected_raw_rows_fetched,
            recovery_work_units=raw_rows_fetched + query_calls,
            expected_recovery_work_units=expected_recovery_work_units,
            handler_recovered_records=handler_summary.total_count,
            handler_sample_records=len(handler_summary.sample_effect_ids),
            handler_query_calls=handler_query_calls,
            expected_handler_query_calls=expected_handler_query_calls,
            handler_raw_rows_fetched=handler_raw_rows_fetched,
            expected_handler_raw_rows_fetched=expected_raw_rows_fetched,
            startup_external_effect_statements=len(startup_statements),
            startup_schema_probe_calls=startup_counts["schema_probe"],
            startup_ddl_calls=sum(
                shape.startswith("schema_") and shape != "schema_probe"
                for shape in startup_shapes
                if shape is not None
            ),
            handler_observed_statements=len(handler_observed_statements),
            handler_observed_selects=sum(
                kind in {"recovery_page", "effect_id", "stale_operation_index"}
                for kind in handler_statement_kinds
            ),
            handler_observed_page_selects=handler_statement_kinds.count(
                "recovery_page"
            ),
            handler_observed_effect_id_selects=handler_statement_kinds.count(
                "effect_id"
            ),
            handler_observed_stale_operation_selects=handler_statement_kinds.count(
                "stale_operation_index"
            ),
            handler_rejected_selects=len(rejected_handler_statements),
            handler_delete_calls=handler_counts["recovery_delete"],
            handler_pending_after=handler_pending_after,
            seeded_rows_after=convergence["rows_after"],
            seeded_identity_mismatches=convergence["identity_mismatches"],
            seeded_convergence_mismatches=convergence[
                "convergence_mismatches"
            ],
            recovery_index=expected_index,
            query_plan_first=first_plan,
            query_plan_resumed=resumed_plan,
            seed_seconds=seed_seconds,
            reopen_seconds=reopen_seconds,
            recovery_seconds=recovery_seconds,
        )
        _assert_structural_contract(result)
        if provider_summary.total_count != 0:
            raise AssertionError("prepared recovery leaked work into provider reconciliation")
        return result
    finally:
        sqlite_storage.sqlite3.connect = original_connect
        if runtime is not None:
            runtime.close()
            store = None
        elif store is not None:
            store.close()
        temporary_directory.cleanup()


def _validate_shape(
    *,
    total_records: int,
    pending_records: int,
    page_size: int,
) -> None:
    if (
        isinstance(total_records, bool)
        or not isinstance(total_records, int)
        or total_records <= 0
    ):
        raise ValueError("total_records must be a positive integer")
    if (
        isinstance(pending_records, bool)
        or not isinstance(pending_records, int)
        or not 0 <= pending_records <= total_records
    ):
        raise ValueError("pending_records must be between zero and total_records")
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size <= 0
    ):
        raise ValueError("page_size must be a positive integer")


def _seed_external_effect_history(
    store: SQLiteStore,
    *,
    total_records: int,
    pending_records: int,
    batch_size: int = 10_000,
) -> None:
    for batch_start in range(0, total_records, batch_size):
        batch_stop = min(total_records, batch_start + batch_size)
        with store.transaction() as cursor:
            cursor.executemany(
                _RECOVERY_INSERT_SQL,
                (
                    _external_effect_row(index, pending_records=pending_records)
                    for index in range(batch_start, batch_stop)
                ),
            )
            pending_stop = min(batch_stop, pending_records)
            if batch_start < pending_stop:
                cursor.executemany(
                    _RECOVERY_OPERATION_INSERT_SQL,
                    (
                        _prepared_operation_row(index)
                        for index in range(batch_start, pending_stop)
                    ),
                )
                cursor.executemany(
                    _RECOVERY_EVIDENCE_INSERT_SQL,
                    (
                        _prepared_effect_evidence_row(index)
                        for index in range(batch_start, pending_stop)
                    ),
                )


def _prepared_operation_row(index: int) -> tuple[Any, ...]:
    operation_id = f"scale-operation-{index:012d}"
    created_at = f"2026-01-01T00:00:00.{index:012d}Z"
    return (
        operation_id,
        operation_id,
        None,
        "primitive",
        "benchmark.external_effect_recovery",
        "scale-benchmark-pid",
        "scale-benchmark-pid",
        "terminal",
        "failed",
        dumps(["effect"]),
        dumps({}),
        None,
        created_at,
        created_at,
        created_at,
    )


def _prepared_effect_evidence_row(index: int) -> tuple[Any, ...]:
    created_at = f"2026-01-01T00:00:00.{index:012d}Z"
    return (
        f"scale-effect-link-{index:012d}",
        f"scale-operation-{index:012d}",
        "external_effect",
        f"scale-effect-{index:012d}",
        "effect",
        created_at,
        dumps(
            {
                "effect_state": "pending",
                "provider": "scale_benchmark",
                "operation": "recover",
            }
        ),
    )


def _external_effect_row(
    index: int,
    *,
    pending_records: int,
) -> tuple[Any, ...]:
    pending = index < pending_records
    created_at = f"2026-01-01T00:00:00.{index:012d}Z"
    return (
        f"scale-effect-{index:012d}",
        None,
        None,
        "scale-benchmark-pid",
        "scale_benchmark",
        "recover",
        None,
        "unknown",
        "unknown",
        0,
        0,
        dumps(
            {
                "protected_operation": {
                    "contract_name": "benchmark.external_effect_recovery",
                    "actor": "scale-benchmark-pid",
                    "reservation_ids": [],
                    "prepared_recovery": None,
                }
            }
            if pending
            else {}
        ),
        created_at,
        "pending" if pending else "finalized",
        "prepared" if pending else "committed",
        None,
        None,
        "{}",
        created_at,
    )


def _consume_recovery_pages(
    store: SQLiteStore,
    *,
    page_size: int,
    transaction_states: tuple[str, ...],
) -> int:
    query = ExternalEffectRecoveryQuery(
        effect_state="pending",
        transaction_states=transaction_states,
        limit=page_size,
    )
    recovered = 0
    previous_cursor = None
    while True:
        page = store.query_external_effect_recovery(query)
        recovered += len(page.records)
        if page.next_cursor is None:
            return recovered
        if previous_cursor is not None and page.next_cursor <= previous_cursor:
            raise AssertionError("external-effect recovery cursor did not advance")
        previous_cursor = page.next_cursor
        query = replace(query, after=page.next_cursor)


def _sqlite_query_plan(
    store: SQLiteStore,
    sql: str,
    params: tuple[Any, ...],
) -> tuple[str, ...]:
    rows = store.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)
    return tuple(str(row[3]) for row in rows)


def _is_external_effect_select(sql: str) -> bool:
    normalized = _strip_sql_comments(str(sql)).casefold()
    return (
        re.search(r"\bselect\b", normalized) is not None
        and "external_effects" in normalized
    )


def _is_external_effect_statement(sql: str) -> bool:
    normalized = _strip_sql_comments(str(sql)).casefold()
    if "external_effects" not in normalized:
        return False
    outside_literals = _SQL_TEXT_LITERAL_RE.sub("?", normalized)
    if "external_effects" in outside_literals:
        return True
    if re.match(
        r"\s*(?:insert(?:\s+or\s+[a-z]+)?|replace)\s+into\s+"
        r"(?:external_effects|[\"`']external_effects[\"`']|"
        r"\[external_effects\])",
        normalized,
    ) is not None:
        return True
    leading = re.match(r"\s*([a-z]+)", normalized)
    return bool(
        leading is not None
        and leading.group(1)
        in {"select", "with", "update", "delete", "create", "drop", "alter", "pragma"}
    )


def _is_allowed_handler_select(sql: str) -> bool:
    return _handler_select_kind(sql) is not None


def _handler_select_kind(sql: str) -> str | None:
    normalized = _normalize_handler_select(sql)
    if _HANDLER_RECOVERY_SELECT_RE.fullmatch(normalized):
        return "recovery_page"
    if normalized == _HANDLER_EFFECT_ID_SELECT:
        return "effect_id"
    if normalized == _normalize_handler_select(
        _HANDLER_STALE_OPERATION_INDEX_SELECT
    ):
        return "stale_operation_index"
    return None


def _handler_statement_kind(sql: str) -> str | None:
    if _is_canonical_schema_catalog_probe(sql):
        return "manifest_schema_probe"
    select_kind = _handler_select_kind(sql)
    if select_kind is not None:
        return select_kind
    if _HANDLER_RECOVERY_DELETE_RE.fullmatch(_normalize_handler_select(sql)):
        return "recovery_delete"
    return None


def _startup_statement_kind(sql: str) -> str | None:
    normalized = " ".join(str(sql).upper().split()).rstrip(";")
    normalized_literals = _SQL_TEXT_LITERAL_RE.sub("?", normalized)
    if _is_canonical_schema_catalog_probe(sql):
        return "manifest_schema_probe"
    if re.fullmatch(
        r"SELECT NAME, SQL FROM SQLITE_MASTER WHERE TYPE = \?"
        r" AND NAME IN \(\?(?:, \?)*\)",
        normalized_literals,
    ):
        return "keyset_collation_schema_probe"
    if normalized == 'PRAGMA TABLE_XINFO("EXTERNAL_EFFECTS")':
        return "catalog_table_xinfo"
    if normalized == 'PRAGMA FOREIGN_KEY_LIST("EXTERNAL_EFFECTS")':
        return "catalog_foreign_key_list"
    if normalized == 'PRAGMA INDEX_LIST("EXTERNAL_EFFECTS")':
        return "catalog_index_list"
    catalog_index_match = re.fullmatch(
        r'PRAGMA INDEX_XINFO\("([A-Z0-9_]+)"\)',
        normalized,
    )
    if catalog_index_match is not None:
        index_name = catalog_index_match.group(1).lower()
        if index_name in _EXTERNAL_EFFECT_SCHEMA_CATALOG_INDEXES:
            return f"catalog_index_xinfo:{index_name}"
    if normalized == "PRAGMA TABLE_INFO(EXTERNAL_EFFECTS)":
        return "schema_probe"
    if (
        normalized.startswith("CREATE TABLE IF NOT EXISTS EXTERNAL_EFFECTS (")
        and " SELECT " not in normalized
    ):
        return "schema_table"
    index_match = re.fullmatch(
        r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS ([A-Z0-9_]+)"
        r" ON EXTERNAL_EFFECTS(?:\s|\().+",
        normalized,
    )
    if index_match is not None:
        index_name = index_match.group(1).lower()
        if index_name in _EXTERNAL_EFFECT_SCHEMA_INDEXES:
            return f"schema_index:{index_name}"
    return None


def _is_canonical_schema_catalog_probe(sql: str) -> bool:
    """Recognize only the store's sorted, full-catalog introspection shape.

    The benchmark is interested in domain-table work, not the canonical
    schema validator reading ``sqlite_master``.  Matching a versioned table
    tuple made every additive schema release look like an unreviewed domain
    query.  The validator's SQL shape is stable: it selects only ``name`` and
    ``type``, uses a literal ``IN`` list, and emits sorted unique canonical
    identifiers including the schema marker.  No statement that reads rows
    from ``external_effects`` can satisfy this full match.
    """

    raw_sql = str(sql)
    normalized = " ".join(raw_sql.upper().split()).rstrip(";")
    normalized_literals = _SQL_TEXT_LITERAL_RE.sub("?", normalized)
    if _SQLITE_SCHEMA_CATALOG_PROBE_RE.fullmatch(normalized_literals) is None:
        return False
    table_names = tuple(
        literal[1:-1].replace("''", "'")
        for literal in _SQL_TEXT_LITERAL_RE.findall(raw_sql)
    )
    return (
        "runtime_schema" in table_names
        and table_names == tuple(sorted(set(table_names)))
        and all(
            _SQLITE_SCHEMA_IDENTIFIER_RE.fullmatch(name) is not None
            for name in table_names
        )
    )


def _normalize_handler_select(sql: str) -> str:
    normalized = " ".join(str(sql).upper().split()).rstrip(";")
    normalized = _SQL_TEXT_LITERAL_RE.sub("?", normalized)
    return _SQL_INTEGER_LITERAL_RE.sub("?", normalized)


def _strip_sql_comments(sql: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(sql):
        current = sql[index]
        if current in {"'", '"', "`"}:
            delimiter = current
            output.append(current)
            index += 1
            while index < len(sql):
                current = sql[index]
                output.append(current)
                index += 1
                if current != delimiter:
                    continue
                if index < len(sql) and sql[index] == delimiter:
                    output.append(sql[index])
                    index += 1
                    continue
                break
            continue
        if current == "[":
            closing = sql.find("]", index + 1)
            closing = len(sql) - 1 if closing < 0 else closing
            output.append(sql[index : closing + 1])
            index = closing + 1
            continue
        if sql.startswith("--", index):
            line_ends = tuple(
                position
                for position in (
                    sql.find("\n", index + 2),
                    sql.find("\r", index + 2),
                )
                if position >= 0
            )
            if not line_ends:
                output.append(" ")
                break
            index = min(line_ends) + 1
            output.append("\n")
            continue
        if sql.startswith("/*", index):
            closing = sql.find("*/", index + 2)
            output.append(" ")
            index = len(sql) if closing < 0 else closing + 2
            continue
        output.append(current)
        index += 1
    return "".join(output)


def _assert_startup_statement_contract(actual: Counter[str]) -> None:
    # Existing-store validation first runs against an isolated safety snapshot,
    # then repeats on the measured main connection to close the snapshot/open
    # race.  The latter includes one canonical full-catalog probe below.
    expected = Counter(
        {
            "schema_probe": 1,
            "keyset_collation_schema_probe": 1,
            "manifest_schema_probe": 1,
            "catalog_table_xinfo": 1,
            "catalog_foreign_key_list": 1,
            "catalog_index_list": 1,
            "schema_table": 1,
        }
    )
    expected.update(
        {f"schema_index:{name}": 1 for name in _EXTERNAL_EFFECT_SCHEMA_INDEXES}
    )
    expected.update(
        {
            f"catalog_index_xinfo:{name}": 1
            for name in _EXTERNAL_EFFECT_SCHEMA_CATALOG_INDEXES
        }
    )
    if actual != expected:
        raise AssertionError(
            "store initialization external-effect statement ledger changed: "
            f"actual={dict(actual)} expected={dict(expected)}"
        )


def _assert_handler_statement_contract(
    actual: Counter[str],
    *,
    pending_records: int,
    expected_handler_query_calls: int,
) -> None:
    expected = Counter(
        {
            "recovery_page": expected_handler_query_calls,
            "recovery_delete": pending_records,
            "stale_operation_index": 1,
        }
    )
    if actual != expected:
        raise AssertionError(
            "Runtime assembly external-effect statement ledger changed: "
            f"actual={dict(actual)} expected={dict(expected)}"
        )


def _external_effect_convergence(
    store: SQLiteStore,
    *,
    total_records: int,
    pending_records: int,
) -> dict[str, int]:
    row = store.conn.execute(
        """
        WITH seeded AS (
            SELECT external_effects.*,
                   CAST(SUBSTR(effect_id, 14) AS INTEGER) AS seed_index
              FROM external_effects
             WHERE effect_id LIKE 'scale-effect-%'
        )
        SELECT (SELECT COUNT(*) FROM external_effects) AS total_rows,
               COUNT(*) AS rows_after,
               COALESCE(SUM(CASE WHEN
                   seed_index < ? OR seed_index >= ? OR
                   effect_id IS NOT printf('scale-effect-%012d', seed_index) OR
                   record_id IS NOT NULL OR event_id IS NOT NULL OR
                   pid IS NOT 'scale-benchmark-pid' OR provider IS NOT 'scale_benchmark' OR
                   operation IS NOT 'recover' OR target IS NOT NULL
                   THEN 1 ELSE 0 END), 0) AS identity_mismatches,
               COALESCE(SUM(CASE WHEN
                   rollback_class IS NOT 'unknown' OR rollback_status IS NOT 'unknown' OR
                   state_mutation IS NOT 0 OR information_flow IS NOT 0 OR
                   provider_metadata_json IS NOT ? OR
                   created_at IS NOT printf('2026-01-01T00:00:00.%012dZ', seed_index) OR
                   effect_state IS NOT 'finalized' OR transaction_state IS NOT 'committed' OR
                   canonical_args_hash IS NOT NULL OR idempotency_key IS NOT NULL OR
                   provider_receipt_json IS NOT '{}' OR updated_at IS NOT created_at OR
                   payload_retention_schema_version IS NOT 1 OR
                   payload_retention_tier IS NOT 'full' OR
                   payload_retention_sha256 IS NOT NULL
                   THEN 1 ELSE 0 END), 0) AS convergence_mismatches
          FROM seeded
        """,
        (pending_records, total_records, dumps({})),
    ).fetchone()
    rows_after = int(row["rows_after"])
    identity_mismatches = int(row["identity_mismatches"])
    if int(row["total_rows"]) != rows_after:
        identity_mismatches += 1
    return {
        "rows_after": rows_after,
        "identity_mismatches": identity_mismatches,
        "convergence_mismatches": int(row["convergence_mismatches"]),
    }


def _require_external_effect_index_contract(
    store: SQLiteStore,
    *,
    expected_index: str,
) -> None:
    index_rows = {
        str(row["name"]): row
        for row in store.conn.execute("PRAGMA index_list(external_effects)")
    }
    index = index_rows.get(expected_index)
    if index is None or int(index["partial"]) != 0:
        raise AssertionError("external-effect recovery index contract changed")
    columns = tuple(
        str(row["name"])
        for row in store.conn.execute(f"PRAGMA index_info({expected_index})")
    )
    expected_columns = (
        (
            "effect_state",
            "transaction_state",
            "created_at",
            "effect_id",
        )
        if expected_index == "idx_external_effects_recovery_transaction"
        else ("effect_state", "created_at", "effect_id")
    )
    if columns != expected_columns:
        raise AssertionError("external-effect recovery index columns changed")


def _assert_structural_contract(result: RecoveryScaleResult) -> None:
    if result.startup_schema_probe_calls != 1:
        raise AssertionError("external-effect measured-main schema probe changed")
    if result.startup_ddl_calls != 8:
        raise AssertionError("external-effect initialization DDL ledger changed")
    if result.recovered_records != result.pending_records:
        raise AssertionError(
            "recovery scan did not return exactly the pending population"
        )
    if result.query_calls != result.expected_query_calls:
        raise AssertionError(
            "recovery query count is not proportional to pending pages: "
            f"{result.query_calls} != {result.expected_query_calls}"
        )
    if result.raw_rows_fetched != result.expected_raw_rows_fetched:
        raise AssertionError(
            "recovery fetched-row work exceeded pending rows plus page lookahead: "
            f"{result.raw_rows_fetched} != {result.expected_raw_rows_fetched}"
        )
    if result.recovery_work_units != result.expected_recovery_work_units:
        raise AssertionError("recovery work units do not match the structural bound")
    if result.handler_recovered_records != result.pending_records:
        raise AssertionError("Runtime handler did not recover the complete pending backlog")
    if result.handler_sample_records != min(result.pending_records, result.page_size):
        raise AssertionError("Runtime handler diagnostics are not page bounded")
    if result.handler_query_calls != result.expected_handler_query_calls:
        raise AssertionError("Runtime handler query work is not page proportional")
    if result.handler_raw_rows_fetched != result.expected_handler_raw_rows_fetched:
        raise AssertionError("Runtime handler fetched more than bounded page lookahead")
    if result.handler_rejected_selects != 0:
        raise AssertionError("Runtime handler trace contains a default-denied statement")
    if result.handler_delete_calls != result.pending_records:
        raise AssertionError("Runtime handler recovery delete ledger changed")
    if (
        result.seeded_rows_after != result.total_records - result.pending_records
        or result.seeded_identity_mismatches != 0
        or result.seeded_convergence_mismatches != 0
    ):
        raise AssertionError("seeded external-effect history did not converge exactly")
    if result.handler_observed_page_selects != result.handler_query_calls:
        raise AssertionError(
            "Runtime handler bypassed the reviewed recovery-page query path"
        )
    if result.handler_observed_effect_id_selects != 0:
        raise AssertionError(
            "prepared-only Runtime recovery unexpectedly read effects by primary key"
        )
    if result.handler_observed_stale_operation_selects != 1:
        raise AssertionError(
            "Runtime handler did not use the exact stale-operation index query once"
        )
    if result.handler_observed_selects != result.handler_query_calls + 1:
        raise AssertionError(
            "Runtime handler external-effect trace exceeded reviewed query work"
        )
    if (
        result.handler_observed_statements
        != result.handler_observed_selects + result.handler_delete_calls
    ):
        raise AssertionError(
            "Runtime handler external-effect statement ledger is incomplete"
        )
    if result.handler_pending_after != 0:
        raise AssertionError("Runtime handler left pending prepared effects")
    plans = (result.query_plan_first, result.query_plan_resumed)
    selected_plans = tuple(plan for plan in plans if plan)
    if not selected_plans:
        raise AssertionError("recovery query plan was not captured")
    for plan in selected_plans:
        details = "\n".join(plan)
        if result.recovery_index not in details:
            raise AssertionError(
                f"recovery query did not use {result.recovery_index}: {details}"
            )
        if any(
            detail.strip().startswith("SCAN external_effects")
            for detail in plan
        ):
            raise AssertionError(f"recovery query performed a table scan: {details}")
        normalized_details = details.upper()
        expected_constraints = (
            "(EFFECT_STATE=? AND TRANSACTION_STATE=?"
            if result.recovery_index
            == "idx_external_effects_recovery_transaction"
            else "(EFFECT_STATE=?"
        )
        if expected_constraints not in normalized_details:
            raise AssertionError(
                "external-effect recovery plan lost its exact search constraints"
            )
        if "USE TEMP B-TREE" in normalized_details:
            raise AssertionError(
                "external-effect recovery plan requires a temporary sort"
            )
