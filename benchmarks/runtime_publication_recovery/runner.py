from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlsplit

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import CheckpointPayloadDeliveryAttempt
from agent_libos.runtime.process_manager import ProcessManager
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import sqlite as sqlite_storage
from agent_libos.storage.repositories import CheckpointRestorePublicationWriter
from agent_libos.storage.sql import _V4_REQUIRED_COLUMNS
from agent_libos.storage.sqlite import SQLiteStore
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps


_MISSING_CLASS_ATTRIBUTE = object()
_GLOBAL_PATCH_SCOPE_LOCK = threading.RLock()
_PUBLICATION_DOMAIN_SELECT = " ".join(
    """
    SELECT publication_id, kind, state, operation_reconciled
    FROM runtime_publications INDEXED BY idx_runtime_publications_invalid_domain
    WHERE kind NOT IN ('process_launch', 'process_exec', 'checkpoint_restore')
    OR state NOT IN ('planning', 'applying', 'reconciliation_pending',
                     'committed', 'rollback_pending', 'rolled_back',
                     'failed', 'manual')
    OR operation_reconciled NOT IN (0, 1) LIMIT 1
    """.upper().split()
)
_SQL_TEXT_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")
_SQL_INTEGER_LITERAL_RE = re.compile(r"(?<![A-Z0-9_?])-?\d+(?![A-Z0-9_?])")
_SQLITE_V4_MANIFEST_TABLES = tuple(sorted(_V4_REQUIRED_COLUMNS))
_SQLITE_V4_MANIFEST_SCHEMA_PROBE_SHAPE = (
    "SELECT NAME, TYPE FROM SQLITE_MASTER WHERE NAME IN ("
    + ", ".join("?" for _ in _SQLITE_V4_MANIFEST_TABLES)
    + ")"
)
_OPERATION_RECONCILIATION_SELECT_RE = re.compile(
    r"SELECT \* FROM RUNTIME_PUBLICATIONS /\* OPERATION-RECONCILIATION \*/"
    r" WHERE KIND = \? AND STATE = \? AND OPERATION_RECONCILED = \?"
    r"(?: AND \(CREATED_AT, PUBLICATION_ID\) > \(\?, \?\))?"
    r" ORDER BY CREATED_AT, PUBLICATION_ID LIMIT \?"
)
_RECOVERY_SELECT_RE = re.compile(
    r"SELECT \* FROM RUNTIME_PUBLICATIONS /\* RECOVERY \*/"
    r" WHERE KIND = \? AND STATE = \? AND OPERATION_RECONCILED = \?"
    r"(?: AND \(CREATED_AT, PUBLICATION_ID\) > \(\?, \?\))?"
    r" ORDER BY CREATED_AT, PUBLICATION_ID LIMIT \?"
)
_EXACT_PUBLICATION_SELECT = (
    "SELECT * FROM RUNTIME_PUBLICATIONS WHERE PUBLICATION_ID = ?"
)
_ORPHAN_ANTIJOIN_SELECT_RE = re.compile(
    r"SELECT PROCESSES\.\* FROM PROCESSES WHERE PROCESSES\.STATUS = \?"
    r" AND NOT EXISTS \(SELECT \? FROM RUNTIME_PUBLICATIONS"
    r" WHERE RUNTIME_PUBLICATIONS\.PID = PROCESSES\.PID"
    r" AND RUNTIME_PUBLICATIONS\.KIND = \? LIMIT \? OFFSET \?\)"
    r"(?: AND \(PROCESSES\.CREATED_AT, PROCESSES\.PID\) > \(\?, \?\))?"
    r" ORDER BY PROCESSES\.CREATED_AT, PROCESSES\.PID LIMIT \?"
)
_PAYLOAD_DELIVERY_PAGE_SELECT_RE = re.compile(
    r"SELECT \* FROM RUNTIME_PUBLICATIONS INDEXED BY "
    r"IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_PAGE"
    r" /\* CHECKPOINT-PAYLOAD-DELIVERY \*/"
    r" WHERE KIND = \? AND STATE = \? AND PHASE = \?"
    r" AND PAYLOAD_DELIVERY_STATE = \?"
    r"(?: AND \(CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY\)"
    r" > \(\?, \?\))?"
    r" ORDER BY CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY"
    r" LIMIT \?"
)
_PAYLOAD_DELIVERY_ATTEMPT_SELECT_RE = re.compile(
    r"SELECT \* FROM RUNTIME_PUBLICATIONS INDEXED BY "
    r"IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_ATTEMPT"
    r" /\* CHECKPOINT-PAYLOAD-DELIVERY \*/"
    r" WHERE KIND = \? AND STATE = \? AND PHASE = \?"
    r" AND PAYLOAD_DELIVERY_STATE = \?"
    r" AND PAYLOAD_DELIVERY_ATTEMPT_ID = \?"
    r"(?: AND \(CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY\)"
    r" > \(\?, \?\))?"
    r" ORDER BY CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY"
    r" LIMIT \?"
)
_PAYLOAD_DELIVERY_ATTEMPTS_SELECT_RE = re.compile(
    r"SELECT ATTEMPT_ID, OWNER_INSTANCE_ID, STARTED_AT"
    r" FROM CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS INDEXED BY"
    r" IDX_CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS_STATE"
    r" WHERE STATE = \?"
    r"(?: AND \(STARTED_AT COLLATE BINARY, ATTEMPT_ID COLLATE BINARY\)"
    r" > \(\?, \?\))?"
    r" ORDER BY STARTED_AT COLLATE BINARY, ATTEMPT_ID COLLATE BINARY"
    r" LIMIT \?"
)
_PAYLOAD_DELIVERY_ATTEMPT_READBACK_RE = re.compile(
    r"SELECT ATTEMPT_ID, OWNER_INSTANCE_ID, STATE, STARTED_AT, ACKED_AT"
    r" FROM CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS WHERE ATTEMPT_ID = \?"
)
_PUBLICATION_RECONCILIATION_UPDATE_RE = re.compile(
    r"UPDATE RUNTIME_PUBLICATIONS SET OPERATION_RECONCILED = \?, UPDATED_AT = \?"
    r" WHERE PUBLICATION_ID = \? AND KIND = \? AND STATE = \? AND PHASE = \?"
    r" AND OPERATION_RECONCILED = \?"
)
_PUBLICATION_RECONCILIATION_INVALIDATION_RE = re.compile(
    r"UPDATE RUNTIME_PUBLICATIONS SET OPERATION_RECONCILED = \?, UPDATED_AT = \?"
    r" WHERE PUBLICATION_ID IN \(\?(?:, \?)*\)"
)
_PAYLOAD_DELIVERY_TRANSITION_UPDATE = " ".join(
    """
    UPDATE RUNTIME_PUBLICATIONS SET RECEIPT_JSON = ?,
    PAYLOAD_DELIVERY_STATE = ?, PAYLOAD_DELIVERY_ATTEMPT_ID = ?,
    PAYLOAD_DELIVERY_STARTED_AT = ?, OPERATION_RECONCILED = ?,
    OWNER_INSTANCE_ID = ?, UPDATED_AT = ?
    WHERE PUBLICATION_ID = ? AND KIND = ? AND STATE = ?
    AND PHASE = ? AND OWNER_INSTANCE_ID = ?
    AND OPERATION_RECONCILED = ? AND RECEIPT_JSON = ?
    AND PAYLOAD_DELIVERY_STATE = ?
    AND PAYLOAD_DELIVERY_ATTEMPT_ID = ?
    AND PAYLOAD_DELIVERY_STARTED_AT = ?
    AND (? = ? OR EXISTS (
        SELECT ? FROM CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS
        WHERE ATTEMPT_ID = ? AND OWNER_INSTANCE_ID = ?
        AND STARTED_AT = ? AND STATE = ?
    ))
    """.split()
)
_PAYLOAD_DELIVERY_PENDING_TRANSITION_UPDATE = (
    _PAYLOAD_DELIVERY_TRANSITION_UPDATE.replace(
        "AND PAYLOAD_DELIVERY_ATTEMPT_ID = ? "
        "AND PAYLOAD_DELIVERY_STARTED_AT = ? AND (? = ?",
        "AND PAYLOAD_DELIVERY_ATTEMPT_ID IS NULL "
        "AND PAYLOAD_DELIVERY_STARTED_AT IS NULL AND (? = ?",
    )
)
_PAYLOAD_DELIVERY_EMPTY_TRANSITION_UPDATE = (
    _PAYLOAD_DELIVERY_PENDING_TRANSITION_UPDATE.replace(
        "AND PAYLOAD_DELIVERY_STATE = ? "
        "AND PAYLOAD_DELIVERY_ATTEMPT_ID IS NULL",
        "AND PAYLOAD_DELIVERY_STATE IS NULL "
        "AND PAYLOAD_DELIVERY_ATTEMPT_ID IS NULL",
    )
)
_PAYLOAD_DELIVERY_ATTEMPT_BEGIN_RE = re.compile(
    r"INSERT INTO CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS"
    r" \(ATTEMPT_ID, OWNER_INSTANCE_ID, STATE, STARTED_AT, ACKED_AT, UPDATED_AT\)"
    r" VALUES \(\?, \?, \?, \?, NULL, \?\) ON CONFLICT DO NOTHING"
)
_PUBLICATION_SCHEMA_INDEXES = frozenset(
    {
        "idx_runtime_publications_state",
        "idx_runtime_publications_pid",
        "idx_runtime_publications_operation_reconciliation",
        "idx_runtime_publications_pid_kind",
        "idx_runtime_publications_invalid_domain",
        "idx_runtime_publications_payload_delivery_page",
        "idx_runtime_publications_payload_delivery_attempt",
        "idx_runtime_publications_payload_delivery_guard",
        "idx_checkpoint_payload_delivery_attempts_state",
        "idx_checkpoint_payload_delivery_attempts_preparing",
    }
)
_EXPECTED_DOMAIN_INDEX_SQL = " ".join(
    """
    CREATE INDEX IDX_RUNTIME_PUBLICATIONS_INVALID_DOMAIN
    ON RUNTIME_PUBLICATIONS(PUBLICATION_ID)
    WHERE KIND NOT IN ('PROCESS_LAUNCH', 'PROCESS_EXEC', 'CHECKPOINT_RESTORE')
    OR STATE NOT IN ('PLANNING', 'APPLYING', 'RECONCILIATION_PENDING',
                     'COMMITTED', 'ROLLBACK_PENDING', 'ROLLED_BACK',
                     'FAILED', 'MANUAL')
    OR OPERATION_RECONCILED NOT IN (0, 1)
    """.split()
)
_EXPECTED_PAYLOAD_PAGE_INDEX_SQL = " ".join(
    """
    CREATE INDEX IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_PAGE
    ON RUNTIME_PUBLICATIONS(
        PAYLOAD_DELIVERY_STATE,
        CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY
    )
    WHERE KIND = 'CHECKPOINT_RESTORE'
      AND STATE = 'COMMITTED'
      AND PHASE = 'RECONCILED'
      AND PAYLOAD_DELIVERY_STATE IS NOT NULL
    """.split()
)
_EXPECTED_PAYLOAD_ATTEMPT_INDEX_SQL = " ".join(
    """
    CREATE INDEX IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_ATTEMPT
    ON RUNTIME_PUBLICATIONS(
        PAYLOAD_DELIVERY_ATTEMPT_ID, PAYLOAD_DELIVERY_STATE,
        CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY
    )
    WHERE KIND = 'CHECKPOINT_RESTORE'
      AND STATE = 'COMMITTED'
      AND PHASE = 'RECONCILED'
      AND PAYLOAD_DELIVERY_ATTEMPT_ID IS NOT NULL
    """.split()
)
_EXPECTED_PAYLOAD_GUARD_INDEX_SQL = " ".join(
    """
    CREATE INDEX IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_GUARD
    ON RUNTIME_PUBLICATIONS(
        PAYLOAD_DELIVERY_ATTEMPT_ID, PAYLOAD_DELIVERY_STATE,
        OWNER_INSTANCE_ID, OPERATION_RECONCILED,
        CREATED_AT COLLATE BINARY, PUBLICATION_ID COLLATE BINARY
    )
    WHERE KIND = 'CHECKPOINT_RESTORE'
      AND STATE = 'COMMITTED'
      AND PHASE = 'RECONCILED'
      AND PAYLOAD_DELIVERY_ATTEMPT_ID IS NOT NULL
    """.split()
)
_EXPECTED_ATTEMPT_STATE_INDEX_SQL = " ".join(
    """
    CREATE INDEX IDX_CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS_STATE
    ON CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS(
        STATE, STARTED_AT COLLATE BINARY, ATTEMPT_ID COLLATE BINARY
    )
    """.split()
)
_EXPECTED_PREPARING_ATTEMPT_INDEX_SQL = " ".join(
    """
    CREATE UNIQUE INDEX IDX_CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS_PREPARING
    ON CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS(STATE)
    WHERE STATE = 'PREPARING'
    """.split()
)
_PAYLOAD_DELIVERY_ATTEMPT_ACK_SQL = (
    "UPDATE checkpoint_payload_delivery_attempts "
    "SET state = 'acked', acked_at = ?, updated_at = ? "
    "WHERE attempt_id = ? AND owner_instance_id = ? "
    "AND state = 'preparing' AND started_at = ? "
    "AND EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? "
    "AND payload_delivery_state = 'completed' "
    "AND owner_instance_id = ? "
    "AND operation_reconciled = 1 LIMIT 1) "
    "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? "
    "AND payload_delivery_state = 'confirmed' LIMIT 1) "
    "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? "
    "AND payload_delivery_state = 'completed' "
    "AND owner_instance_id = ? "
    "AND operation_reconciled = 0 LIMIT 1) "
    "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? "
    "AND payload_delivery_state = 'completed' "
    "AND owner_instance_id < ? LIMIT 1) "
    "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? "
    "AND payload_delivery_state = 'completed' "
    "AND owner_instance_id > ? LIMIT 1)"
)
_PAYLOAD_DELIVERY_ATTEMPT_ABORT_SQL = (
    "UPDATE checkpoint_payload_delivery_attempts "
    "SET state = 'aborted', updated_at = ? "
    "WHERE attempt_id = ? AND owner_instance_id = ? "
    "AND state = 'preparing' AND started_at = ? "
    "AND NOT EXISTS (SELECT 1 FROM runtime_publications "
    "INDEXED BY idx_runtime_publications_payload_delivery_guard "
    "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
    "AND phase = 'reconciled' "
    "AND payload_delivery_attempt_id IS NOT NULL "
    "AND payload_delivery_attempt_id = ? LIMIT 1)"
)
_PAYLOAD_DELIVERY_ATTEMPT_READBACK_SQL = (
    "SELECT attempt_id, owner_instance_id, state, started_at, acked_at "
    "FROM checkpoint_payload_delivery_attempts WHERE attempt_id = ?"
)
_PUBLICATION_REPOSITORY_SELECT_SHAPES = frozenset(
    {
        "domain_validation",
        "operation_reconciliation",
        "recovery",
        "exact_publication",
        "orphan_antijoin",
        "payload_delivery_page",
        "payload_delivery_attempt",
        "payload_delivery_attempts",
    }
)


def _connect_with_trace_installed(
    connect: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    register_connection: Callable[[Any], None],
) -> Any:
    """Install tracing in the factory before a connect wrapper can run SQL."""

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
    register_connection(connection)
    return connection


@dataclass(frozen=True)
class PublicationScaleProfile:
    total_records: int
    unreconciled_records: int
    page_size: int


@dataclass(frozen=True)
class _PayloadDeliveryTransitionEvidence:
    final_attempt: CheckpointPayloadDeliveryAttempt
    records_per_phase: int
    transition_rows: int
    transaction_pages: int
    max_page_records: int
    phase_page_counts: tuple[int, ...]


PUBLICATION_SCALE_PROFILES = {
    "ci": PublicationScaleProfile(
        total_records=10_000,
        unreconciled_records=1_001,
        page_size=500,
    ),
}

TERMINAL_RECONCILIATION_STATES = {
    "process_launch": ("committed", "rolled_back", "failed", "manual"),
    "process_exec": ("committed", "rolled_back", "failed", "manual"),
    # Failed/manual checkpoint restores are forward-recovery inputs rather
    # than terminal-operation-only repairs. Committed restore operations are
    # checked before startup hooks and once more before OPEN because hooks may
    # independently dirty their operation marker.
    "checkpoint_restore": ("committed", "committed"),
}

RECOVERY_QUERY_STATES = {
    "process_launch": (
        "planning",
        "applying",
        "rollback_pending",
        "failed",
        "manual",
    ),
    "process_exec": (
        "planning",
        "applying",
        "rollback_pending",
        "failed",
        "manual",
    ),
    "checkpoint_restore": (
        "planning",
        "applying",
        "reconciliation_pending",
        "failed",
        "manual",
    ),
}


@dataclass(frozen=True)
class PublicationScaleResult:
    schema_version: int
    total_records: int
    unreconciled_records: int
    handler_reconciled_records: int
    handler_sample_records: int
    page_size: int
    publication_statement_calls: int
    publication_select_calls: int
    publication_query_calls: int
    publication_schema_probe_calls: int
    publication_ddl_calls: int
    publication_update_calls: int
    publication_invalidation_calls: int
    seeded_rows_after: int
    seeded_identity_mismatches: int
    seeded_convergence_mismatches: int
    seeded_terminal_missing_rows: int
    seeded_terminal_unexpected_rows: int
    total_operations_after: int
    bound_operations_after: int
    bound_operation_mismatches: int
    domain_validation_query_calls: int
    domain_validation_rows_fetched: int
    domain_validation_index: str
    domain_validation_query_plan: tuple[str, ...]
    reconciliation_query_calls: int
    handler_query_calls: int
    expected_handler_query_calls: int
    handler_raw_rows_fetched: int
    expected_handler_raw_rows_fetched: int
    max_rows_fetched: int
    payload_delivery_records: int
    payload_delivery_query_calls: int
    expected_payload_delivery_query_calls: int
    payload_delivery_raw_rows_fetched: int
    expected_payload_delivery_raw_rows_fetched: int
    payload_delivery_max_rows_fetched: int
    payload_delivery_transition_phases: int
    payload_delivery_transition_rows: int
    payload_delivery_transition_transactions: int
    expected_payload_delivery_transition_transactions: int
    payload_delivery_transition_max_records: int
    payload_delivery_transition_page_counts: tuple[int, ...]
    payload_delivery_transition_update_calls: int
    payload_delivery_pending_query_calls: int
    payload_delivery_pending_rows_fetched: int
    payload_delivery_page_index: str
    payload_delivery_attempt_index: str
    payload_delivery_guard_index: str
    payload_delivery_query_plan_first: tuple[str, ...]
    payload_delivery_query_plan_resumed: tuple[str, ...]
    payload_delivery_pending_query_plan: tuple[str, ...]
    payload_delivery_fixture_mismatches: int
    payload_attempt_history_records: int
    payload_attempt_query_calls: int
    payload_attempt_rows_fetched: int
    payload_attempt_max_rows_fetched: int
    payload_attempt_index: str
    payload_attempt_query_plan: tuple[str, ...]
    payload_attempt_begin_calls: int
    payload_attempt_ack_calls: int
    payload_attempt_abort_calls: int
    payload_attempt_readback_calls: int
    payload_attempt_ack_query_plan: tuple[str, ...]
    payload_attempt_abort_query_plan: tuple[str, ...]
    payload_attempt_readback_query_plan: tuple[str, ...]
    payload_attempt_total_rows_after: int
    payload_attempt_preparing_rows_after: int
    payload_attempt_acked_rows_after: int
    payload_attempt_aborted_rows_after: int
    reconciliation_index: str
    reconciliation_sql_first: str
    reconciliation_params_first: tuple[Any, ...]
    reconciliation_query_plan_first: tuple[str, ...]
    reconciliation_sql_resumed: str
    reconciliation_params_resumed: tuple[Any, ...]
    reconciliation_query_plan_resumed: tuple[str, ...]
    seed_seconds: float
    reopen_seconds: float

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reconciliation_query_plan_first"] = list(
            self.reconciliation_query_plan_first
        )
        result["domain_validation_query_plan"] = list(
            self.domain_validation_query_plan
        )
        result["reconciliation_params_first"] = list(
            self.reconciliation_params_first
        )
        result["reconciliation_query_plan_resumed"] = list(
            self.reconciliation_query_plan_resumed
        )
        result["reconciliation_params_resumed"] = list(
            self.reconciliation_params_resumed
        )
        result["payload_delivery_transition_page_counts"] = list(
            self.payload_delivery_transition_page_counts
        )
        for field_name in (
            "payload_delivery_query_plan_first",
            "payload_delivery_query_plan_resumed",
            "payload_delivery_pending_query_plan",
            "payload_attempt_query_plan",
            "payload_attempt_ack_query_plan",
            "payload_attempt_abort_query_plan",
            "payload_attempt_readback_query_plan",
        ):
            result[field_name] = list(getattr(self, field_name))
        result["timing_is_informational_only"] = True
        return result


def run_publication_scale_benchmark(
    *,
    total_records: int,
    unreconciled_records: int,
    page_size: int,
) -> PublicationScaleResult:
    """Run one benchmark inside the serialized global-patch scope."""

    with _GLOBAL_PATCH_SCOPE_LOCK:
        return _run_publication_scale_benchmark_locked(
            total_records=total_records,
            unreconciled_records=unreconciled_records,
            page_size=page_size,
        )


def _run_publication_scale_benchmark_locked(
    *,
    total_records: int,
    unreconciled_records: int,
    page_size: int,
) -> PublicationScaleResult:
    _validate_shape(
        total_records=total_records,
        unreconciled_records=unreconciled_records,
        page_size=page_size,
    )
    temporary_directory = TemporaryDirectory(
        prefix="agent-libos-publication-recovery-"
    )
    database_path = Path(temporary_directory.name) / "runtime.sqlite"
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_reconciliation_page_size=page_size)
    )
    store: SQLiteStore | None = None
    runtime: Runtime | None = None
    original_query = SQLiteStore._query
    original_query_local = SQLiteStore.__dict__.get(
        "_query",
        _MISSING_CLASS_ATTRIBUTE,
    )
    original_reconcile = ProcessManager.reconcile_terminal_publications
    original_connect = sqlite_storage.sqlite3.connect
    try:
        store = SQLiteStore(database_path, config=config)
        seed_started = time.perf_counter()
        _seed_terminal_publication_history(
            store,
            total_records=total_records,
            unreconciled_records=unreconciled_records,
        )
        payload_delivery_records = (2 * page_size) + 5
        payload_delivery_attempt = _seed_payload_delivery_scale_fixture(
            store,
            payload_delivery_records=payload_delivery_records,
            attempt_history_records=total_records,
        )
        seed_seconds = time.perf_counter() - seed_started
        store.close()
        store = None

        queries: list[tuple[str, tuple[Any, ...], int, str]] = []
        traced_publication_statements: list[str] = []
        traced_connections: list[Any] = []
        traced_connection_ids: set[int] = set()
        handler_samples: list[tuple[str, ...]] = []
        query_trace_context = threading.local()

        def tracked_query(
            selected_store: SQLiteStore,
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            if not _store_targets_database(selected_store, database_path):
                return original_query(selected_store, sql, params)
            shape = _publication_select_shape(sql)
            if shape == "unreviewed":
                raise AssertionError(
                    f"Runtime reopen used an unreviewed publication SELECT: {sql}"
                )
            selected_params = tuple(params)
            observed_shapes: list[str] = []
            stack = getattr(query_trace_context, "stack", None)
            if stack is None:
                stack = []
                query_trace_context.stack = stack
            if shape is not None:
                stack.append(observed_shapes)
            try:
                rows = original_query(selected_store, sql, selected_params)
            finally:
                if shape is not None:
                    stack.pop()
            if shape is not None and observed_shapes != [shape]:
                raise AssertionError(
                    "publication repository query did not execute exactly one "
                    "matching traced SELECT: "
                    f"expected={shape!r} observed={observed_shapes!r} sql={sql}"
                )
            if shape is not None:
                queries.append((sql, selected_params, len(rows), shape))
            return rows

        def tracked_reconcile(manager: ProcessManager) -> list[str]:
            if not _manager_targets_database(manager, database_path):
                return original_reconcile(manager)
            reconciled = original_reconcile(manager)
            handler_samples.append(tuple(reconciled))
            return reconciled

        def register_traced_connection(connection: Any) -> None:
            if id(connection) in traced_connection_ids:
                return
            traced_connection_ids.add(id(connection))
            traced_connections.append(connection)

            def trace_publication_statement(sql: str) -> None:
                if not _is_publication_statement(sql):
                    return
                traced_publication_statements.append(sql)
                shape = _publication_statement_shape(sql)
                if shape not in _PUBLICATION_REPOSITORY_SELECT_SHAPES:
                    return
                stack = getattr(query_trace_context, "stack", ())
                if stack:
                    stack[-1].append(shape)

            connection.set_trace_callback(trace_publication_statement)

        def tracked_connect(*args: Any, **kwargs: Any) -> Any:
            if not _connect_targets_database(args, kwargs, database_path):
                return original_connect(*args, **kwargs)
            return _connect_with_trace_installed(
                original_connect,
                args,
                kwargs,
                register_connection=register_traced_connection,
            )

        SQLiteStore._query = tracked_query
        ProcessManager.reconcile_terminal_publications = tracked_reconcile
        sqlite_storage.sqlite3.connect = tracked_connect
        payload_transition_evidence: _PayloadDeliveryTransitionEvidence | None = None
        try:
            reopen_started = time.perf_counter()
            runtime = Runtime.open(database_path, config=config)
            reopen_seconds = time.perf_counter() - reopen_started
            if not isinstance(runtime.store, SQLiteStore):
                raise AssertionError(
                    "publication scale benchmark did not open SQLite"
                )
            payload_transition_evidence = _exercise_payload_delivery_state_machine(
                runtime.store,
                initial_attempt=payload_delivery_attempt,
                page_size=page_size,
                expected_records=payload_delivery_records,
            )
        finally:
            _run_cleanup_steps(
                [
                    (
                        "restore sqlite connect",
                        lambda: setattr(
                            sqlite_storage.sqlite3,
                            "connect",
                            original_connect,
                        ),
                    ),
                    (
                        "restore SQLiteStore._query",
                        lambda: _restore_class_attribute(
                            SQLiteStore,
                            "_query",
                            original_query_local,
                        ),
                    ),
                    (
                        "restore ProcessManager reconciliation",
                        lambda: setattr(
                            ProcessManager,
                            "reconcile_terminal_publications",
                            original_reconcile,
                        ),
                    ),
                    *[
                        (
                            "detach publication trace callback",
                            lambda connection=connection: _detach_trace_callback(
                                connection
                            ),
                        )
                        for connection in traced_connections
                    ],
                ]
            )

        selected_store = runtime.store
        if not isinstance(selected_store, SQLiteStore):
            raise AssertionError("publication scale benchmark did not open SQLite")
        trace_shapes = tuple(
            _publication_statement_shape(sql)
            for sql in traced_publication_statements
        )
        unreviewed_traces = [
            sql
            for sql, shape in zip(
                traced_publication_statements,
                trace_shapes,
                strict=True,
            )
            if shape in {None, "unreviewed"}
        ]
        if unreviewed_traces:
            raise AssertionError(
                "Runtime reopen executed an unreviewed publication statement: "
                f"{unreviewed_traces[0]}"
            )
        trace_shape_counts = Counter(
            shape for shape in trace_shapes if shape is not None
        )
        traced_repository_selects = Counter(
            shape
            for shape in trace_shapes
            if shape in _PUBLICATION_REPOSITORY_SELECT_SHAPES
        )
        measured_repository_selects = Counter(
            shape for _sql, _params, _row_count, shape in queries
        )
        if traced_repository_selects != measured_repository_selects:
            raise AssertionError(
                "publication traced SELECT/helper multiset changed: "
                f"traced={dict(traced_repository_selects)} "
                f"measured={dict(measured_repository_selects)}"
            )
        if payload_transition_evidence is None:
            raise AssertionError("payload delivery transition evidence was not recorded")
        if payload_transition_evidence.records_per_phase != payload_delivery_records:
            raise AssertionError(
                "payload delivery attempt page traversal lost or duplicated rows"
            )
        _assert_publication_query_key_contract(
            queries,
            unreconciled_records=unreconciled_records,
            payload_delivery_records=payload_delivery_records,
            expected_handler_query_calls=math.ceil(
                unreconciled_records / page_size
            ),
            expected_payload_delivery_page_calls=math.ceil(
                payload_delivery_records / page_size
            ),
        )
        payload_delivery_fixture = _payload_delivery_fixture_contract(
            selected_store,
            attempt=payload_transition_evidence.final_attempt,
            expected_records=payload_delivery_records,
        )
        payload_attempt_contract = _payload_attempt_fixture_contract(
            selected_store,
            attempt_history_records=total_records,
        )
        _discard_payload_delivery_fixture(
            selected_store,
            expected_records=payload_delivery_records,
        )
        convergence = _publication_convergence(
            selected_store,
            total_records=total_records,
            unreconciled_records=unreconciled_records,
        )
        if (
            convergence["total_rows"] != total_records
            or convergence["seeded_rows"] != total_records
            or convergence["identity_mismatches"] != 0
            or convergence["convergence_mismatches"] != 0
            or convergence["total_operations"] != unreconciled_records
            or convergence["bound_operations"] != unreconciled_records
            or convergence["operation_mismatches"] != 0
        ):
            raise AssertionError(
                "seeded publication/operation convergence changed: "
                f"{convergence}"
            )
        terminal_contract = _publication_terminal_contract(
            selected_store,
            total_records=total_records,
            unreconciled_records=unreconciled_records,
        )
        if (
            terminal_contract["missing_rows"] != 0
            or terminal_contract["unexpected_rows"] != 0
            or terminal_contract["reconciled_backlog_rows"]
            != unreconciled_records
        ):
            raise AssertionError(
                "seeded publication terminal multiset changed: "
                f"{terminal_contract}"
            )
        _require_publication_index_contract(selected_store)
        active_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape == "operation_reconciliation"
            and params[:2] == ("process_launch", "committed")
        ]
        domain_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape == "domain_validation"
        ]
        payload_delivery_attempt_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape == "payload_delivery_attempt"
        ]
        payload_delivery_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape in {"payload_delivery_page", "payload_delivery_attempt"}
            and row_count > 0
        ]
        payload_delivery_pending_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape == "payload_delivery_page" and row_count == 0
        ]
        payload_attempt_queries = [
            (sql, params, row_count)
            for sql, params, row_count, shape in queries
            if shape == "payload_delivery_attempts"
        ]
        plans = tuple(
            _query_plan(selected_store, sql, params)
            for sql, params, _row_count in active_queries[:2]
        )
        payload_delivery_plans = tuple(
            _query_plan(selected_store, sql, params)
            for sql, params, _row_count in payload_delivery_attempt_queries[:2]
        )
        payload_delivery_pending_plan = (
            _query_plan(
                selected_store,
                payload_delivery_pending_queries[0][0],
                payload_delivery_pending_queries[0][1],
            )
            if payload_delivery_pending_queries
            else ()
        )
        payload_attempt_plan = (
            _query_plan(
                selected_store,
                payload_attempt_queries[0][0],
                payload_attempt_queries[0][1],
            )
            if payload_attempt_queries
            else ()
        )
        payload_attempt_ack_plan = _query_plan(
            selected_store,
            _PAYLOAD_DELIVERY_ATTEMPT_ACK_SQL,
            (
                "now",
                "now",
                "attempt",
                "owner",
                "started",
                "attempt",
                "owner",
                "attempt",
                "attempt",
                "owner",
                "attempt",
                "owner",
                "attempt",
                "owner",
            ),
        )
        payload_attempt_abort_plan = _query_plan(
            selected_store,
            _PAYLOAD_DELIVERY_ATTEMPT_ABORT_SQL,
            ("now", "attempt", "owner", "started", "attempt"),
        )
        payload_attempt_readback_plan = _query_plan(
            selected_store,
            _PAYLOAD_DELIVERY_ATTEMPT_READBACK_SQL,
            ("attempt",),
        )
        expected_handler_query_calls = math.ceil(
            unreconciled_records / page_size
        )
        expected_handler_raw_rows_fetched = (
            unreconciled_records + expected_handler_query_calls - 1
        )
        expected_payload_delivery_page_calls = math.ceil(
            payload_delivery_records / page_size
        )
        expected_payload_delivery_query_calls = (
            5 * expected_payload_delivery_page_calls
        )
        expected_payload_delivery_raw_rows_fetched = (
            5
            * (
                payload_delivery_records
                + expected_payload_delivery_page_calls
                - 1
            )
        )
        _assert_publication_trace_contract(
            trace_shape_counts,
            unreconciled_records=unreconciled_records,
            payload_delivery_records=payload_delivery_records,
            expected_handler_query_calls=expected_handler_query_calls,
            expected_payload_delivery_page_calls=(
                expected_payload_delivery_page_calls
            ),
        )

        result = PublicationScaleResult(
            schema_version=3,
            total_records=total_records,
            unreconciled_records=unreconciled_records,
            handler_reconciled_records=terminal_contract[
                "reconciled_backlog_rows"
            ],
            handler_sample_records=(
                len(handler_samples[0]) if handler_samples else 0
            ),
            page_size=page_size,
            publication_statement_calls=len(traced_publication_statements),
            publication_select_calls=sum(
                shape in _PUBLICATION_REPOSITORY_SELECT_SHAPES
                for shape in trace_shapes
            ),
            publication_query_calls=len(queries),
            publication_schema_probe_calls=trace_shape_counts["schema_probe"],
            publication_ddl_calls=sum(
                shape.startswith("schema_") and shape != "schema_probe"
                for shape in trace_shapes
                if shape is not None
            ),
            publication_update_calls=trace_shape_counts[
                "operation_reconciliation_update"
            ],
            publication_invalidation_calls=trace_shape_counts[
                "operation_reconciliation_invalidation"
            ],
            seeded_rows_after=convergence["seeded_rows"],
            seeded_identity_mismatches=convergence["identity_mismatches"],
            seeded_convergence_mismatches=convergence[
                "convergence_mismatches"
            ],
            seeded_terminal_missing_rows=terminal_contract["missing_rows"],
            seeded_terminal_unexpected_rows=terminal_contract[
                "unexpected_rows"
            ],
            total_operations_after=convergence["total_operations"],
            bound_operations_after=convergence["bound_operations"],
            bound_operation_mismatches=convergence["operation_mismatches"],
            domain_validation_query_calls=len(domain_queries),
            domain_validation_rows_fetched=sum(
                row_count for _sql, _params, row_count in domain_queries
            ),
            domain_validation_index="idx_runtime_publications_invalid_domain",
            domain_validation_query_plan=(
                _query_plan(
                    selected_store,
                    domain_queries[0][0],
                    domain_queries[0][1],
                )
                if domain_queries
                else ()
            ),
            reconciliation_query_calls=sum(
                shape == "operation_reconciliation"
                for _sql, _params, _count, shape in queries
            ),
            handler_query_calls=len(active_queries),
            expected_handler_query_calls=expected_handler_query_calls,
            handler_raw_rows_fetched=sum(
                row_count for _sql, _params, row_count in active_queries
            ),
            expected_handler_raw_rows_fetched=(
                expected_handler_raw_rows_fetched
            ),
            max_rows_fetched=max(
                (count for _sql, _params, count, _shape in queries),
                default=0,
            ),
            payload_delivery_records=payload_delivery_records,
            payload_delivery_query_calls=len(payload_delivery_queries),
            expected_payload_delivery_query_calls=(
                expected_payload_delivery_query_calls
            ),
            payload_delivery_raw_rows_fetched=sum(
                row_count
                for _sql, _params, row_count in payload_delivery_queries
            ),
            expected_payload_delivery_raw_rows_fetched=(
                expected_payload_delivery_raw_rows_fetched
            ),
            payload_delivery_max_rows_fetched=max(
                (
                    row_count
                    for _sql, _params, row_count in payload_delivery_queries
                ),
                default=0,
            ),
            payload_delivery_transition_phases=5,
            payload_delivery_transition_rows=(
                payload_transition_evidence.transition_rows
            ),
            payload_delivery_transition_transactions=(
                payload_transition_evidence.transaction_pages
            ),
            expected_payload_delivery_transition_transactions=(
                expected_payload_delivery_query_calls
            ),
            payload_delivery_transition_max_records=(
                payload_transition_evidence.max_page_records
            ),
            payload_delivery_transition_page_counts=(
                payload_transition_evidence.phase_page_counts
            ),
            payload_delivery_transition_update_calls=trace_shape_counts[
                "payload_delivery_transition_update"
            ],
            payload_delivery_pending_query_calls=len(
                payload_delivery_pending_queries
            ),
            payload_delivery_pending_rows_fetched=sum(
                row_count
                for _sql, _params, row_count in payload_delivery_pending_queries
            ),
            payload_delivery_page_index=(
                "idx_runtime_publications_payload_delivery_page"
            ),
            payload_delivery_attempt_index=(
                "idx_runtime_publications_payload_delivery_attempt"
            ),
            payload_delivery_guard_index=(
                "idx_runtime_publications_payload_delivery_guard"
            ),
            payload_delivery_query_plan_first=(
                payload_delivery_plans[0] if payload_delivery_plans else ()
            ),
            payload_delivery_query_plan_resumed=(
                payload_delivery_plans[1]
                if len(payload_delivery_plans) > 1
                else ()
            ),
            payload_delivery_pending_query_plan=(
                payload_delivery_pending_plan
            ),
            payload_delivery_fixture_mismatches=payload_delivery_fixture[
                "mismatches"
            ],
            payload_attempt_history_records=total_records,
            payload_attempt_query_calls=len(payload_attempt_queries),
            payload_attempt_rows_fetched=sum(
                row_count for _sql, _params, row_count in payload_attempt_queries
            ),
            payload_attempt_max_rows_fetched=max(
                (
                    row_count
                    for _sql, _params, row_count in payload_attempt_queries
                ),
                default=0,
            ),
            payload_attempt_index=(
                "idx_checkpoint_payload_delivery_attempts_state"
            ),
            payload_attempt_query_plan=payload_attempt_plan,
            payload_attempt_begin_calls=trace_shape_counts[
                "payload_attempt_begin"
            ],
            payload_attempt_ack_calls=trace_shape_counts["payload_attempt_ack"],
            payload_attempt_abort_calls=trace_shape_counts[
                "payload_attempt_abort"
            ],
            payload_attempt_readback_calls=trace_shape_counts[
                "payload_attempt_readback"
            ],
            payload_attempt_ack_query_plan=payload_attempt_ack_plan,
            payload_attempt_abort_query_plan=payload_attempt_abort_plan,
            payload_attempt_readback_query_plan=(
                payload_attempt_readback_plan
            ),
            payload_attempt_total_rows_after=payload_attempt_contract[
                "total_rows"
            ],
            payload_attempt_preparing_rows_after=payload_attempt_contract[
                "preparing_rows"
            ],
            payload_attempt_acked_rows_after=payload_attempt_contract[
                "acked_rows"
            ],
            payload_attempt_aborted_rows_after=payload_attempt_contract[
                "aborted_rows"
            ],
            reconciliation_index=(
                "idx_runtime_publications_operation_reconciliation"
            ),
            reconciliation_sql_first=(active_queries[0][0] if active_queries else ""),
            reconciliation_params_first=(
                active_queries[0][1] if active_queries else ()
            ),
            reconciliation_query_plan_first=plans[0] if plans else (),
            reconciliation_sql_resumed=(
                active_queries[1][0] if len(active_queries) > 1 else ""
            ),
            reconciliation_params_resumed=(
                active_queries[1][1] if len(active_queries) > 1 else ()
            ),
            reconciliation_query_plan_resumed=(
                plans[1] if len(plans) > 1 else ()
            ),
            seed_seconds=seed_seconds,
            reopen_seconds=reopen_seconds,
        )
        _assert_structural_contract(result)
        return result
    finally:
        cleanup_steps: list[tuple[str, Callable[[], Any]]] = [
            (
                "restore SQLiteStore._query",
                lambda: _restore_class_attribute(
                    SQLiteStore,
                    "_query",
                    original_query_local,
                ),
            ),
            (
                "restore sqlite connect",
                lambda: setattr(
                    sqlite_storage.sqlite3,
                    "connect",
                    original_connect,
                ),
            ),
            (
                "restore ProcessManager reconciliation",
                lambda: setattr(
                    ProcessManager,
                    "reconcile_terminal_publications",
                    original_reconcile,
                ),
            ),
        ]
        if runtime is not None:
            cleanup_steps.append(("close Runtime", runtime.close))
        if store is not None:
            cleanup_steps.append(("close SQLiteStore", store.close))
        cleanup_steps.append(
            ("remove temporary benchmark directory", temporary_directory.cleanup)
        )
        _run_cleanup_steps(cleanup_steps)


def _connect_targets_database(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    database_path: Path,
) -> bool:
    raw_database = args[0] if args else kwargs.get("database")
    return _database_path_matches(raw_database, database_path)


def _store_targets_database(store: SQLiteStore, database_path: Path) -> bool:
    return _database_path_matches(getattr(store, "path", None), database_path)


def _manager_targets_database(
    manager: ProcessManager,
    database_path: Path,
) -> bool:
    publications = getattr(manager, "publications", None)
    backend = getattr(publications, "_publication_backend", None)
    return isinstance(backend, SQLiteStore) and _store_targets_database(
        backend,
        database_path,
    )


def _database_path_matches(raw_database: Any, database_path: Path) -> bool:
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


def _run_cleanup_steps(steps: list[tuple[str, Callable[[], Any]]]) -> None:
    primary = sys.exception()
    failures: list[BaseException] = []
    for label, callback in steps:
        try:
            callback()
        except BaseException as exc:
            exc.add_note(f"cleanup step failed: {label}")
            failures.append(exc)
    if not failures:
        return
    if primary is not None:
        for failure in failures:
            primary.add_note(
                f"secondary cleanup failure: {type(failure).__name__}: {failure}"
            )
        return
    if all(isinstance(failure, Exception) for failure in failures):
        raise ExceptionGroup(
            "publication benchmark cleanup failed",
            [failure for failure in failures if isinstance(failure, Exception)],
        )
    raise BaseExceptionGroup("publication benchmark cleanup failed", failures)


def _detach_trace_callback(connection: Any) -> None:
    try:
        connection.set_trace_callback(None)
    except sqlite_storage.sqlite3.ProgrammingError as exc:
        if "closed" in str(exc).casefold():
            return
        raise


def _validate_shape(
    *,
    total_records: int,
    unreconciled_records: int,
    page_size: int,
) -> None:
    for name, value in (
        ("total_records", total_records),
        ("unreconciled_records", unreconciled_records),
        ("page_size", page_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if unreconciled_records > total_records:
        raise ValueError("unreconciled_records must not exceed total_records")
    if unreconciled_records <= page_size:
        raise ValueError("unreconciled_records must span more than one page")
    if page_size > 5_000:
        raise ValueError("page_size exceeds the runtime publication hard limit")


def _seed_terminal_publication_history(
    store: SQLiteStore,
    *,
    total_records: int,
    unreconciled_records: int,
    batch_size: int = 10_000,
) -> None:
    now = utc_now()
    sql = (
        "INSERT INTO runtime_publications ("
        "publication_id, kind, pid, owner_instance_id, state, phase, plan_json, "
        "receipt_json, error_json, operation_reconciled, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    operation_sql = (
        "INSERT INTO operations ("
        "operation_id, root_operation_id, parent_operation_id, kind, name, "
        "actor, pid, state, outcome, expected_roles_json, metadata_json, "
        "runtime_publication_id, started_at, updated_at, completed_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for batch_start in range(0, total_records, batch_size):
        batch_stop = min(total_records, batch_start + batch_size)
        with store.transaction() as cursor:
            cursor.executemany(
                sql,
                (
                    (
                        f"publication-history-{index:08d}",
                        _seeded_publication_kind(
                            index,
                            unreconciled_records=unreconciled_records,
                        ),
                        f"pid-history-{index:08d}",
                        "scale-owner",
                        "committed",
                        "committed",
                        _seeded_publication_plan(
                            index,
                            unreconciled_records=unreconciled_records,
                        ),
                        dumps({"phases": [], "artifacts": []}),
                        None,
                        int(index >= unreconciled_records),
                        now,
                        now,
                    )
                    for index in range(batch_start, batch_stop)
                ),
            )
            cursor.executemany(
                operation_sql,
                (
                    (
                        f"scale-operation-{index:08d}",
                        f"scale-operation-{index:08d}",
                        None,
                        "runtime",
                        "process.spawn",
                        "runtime",
                        None,
                        "running",
                        "pending",
                        dumps([]),
                        dumps(
                            {
                                "runtime_publication_id": (
                                    f"publication-history-{index:08d}"
                                ),
                                "runtime_publication_kind": "process_launch",
                                "runtime_publication_bound": True,
                                "runtime_publication_binding_version": 1,
                            }
                        ),
                        f"publication-history-{index:08d}",
                        now,
                        now,
                        None,
                    )
                    for index in range(
                        batch_start,
                        min(batch_stop, unreconciled_records),
                    )
                ),
            )


def _seed_payload_delivery_scale_fixture(
    store: SQLiteStore,
    *,
    payload_delivery_records: int,
    attempt_history_records: int,
) -> CheckpointPayloadDeliveryAttempt:
    """Seed completed payload pages plus irrelevant historical attempts.

    The completed rows are ignored by startup recovery, then read through the
    production attempt-keyset helper while tracing is active. Historical
    acked/aborted control rows make the preparing-attempt lookup prove that its
    cost is independent of the table's total cardinality.
    """

    now = utc_now()
    delivery_attempt = CheckpointPayloadDeliveryAttempt(
        started_at=now,
        attempt_id="payload-attempt-transition-initial",
        owner_instance_id="payload-scale-owner",
    )
    attempt_sql = (
        "INSERT INTO checkpoint_payload_delivery_attempts ("
        "attempt_id, owner_instance_id, state, started_at, acked_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?)"
    )
    publication_sql = (
        "INSERT INTO runtime_publications ("
        "publication_id, kind, pid, owner_instance_id, state, phase, plan_json, "
        "receipt_json, error_json, operation_reconciled, "
        "payload_delivery_state, payload_delivery_attempt_id, "
        "payload_delivery_started_at, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with store.transaction() as cursor:
        cursor.executemany(
            attempt_sql,
            (
                (
                    f"payload-attempt-history-{index:08d}",
                    "payload-scale-owner",
                    "acked" if index % 2 == 0 else "aborted",
                    now,
                    now if index % 2 == 0 else None,
                    now,
                )
                for index in range(attempt_history_records)
            ),
        )
        receipt = dumps(
            {
                "phases": [],
                "artifacts": [],
                "payload_delivery": {"state": "completed"},
                "payload_delivery_attempt": {
                    "attempt_id": delivery_attempt.attempt_id,
                    "started_at": delivery_attempt.started_at,
                },
            }
        )
        cursor.executemany(
            publication_sql,
            (
                (
                    f"payload-delivery-history-{index:08d}",
                    "checkpoint_restore",
                    f"payload-delivery-pid-{index:08d}",
                    delivery_attempt.owner_instance_id,
                    "committed",
                    "reconciled",
                    dumps({}),
                    receipt,
                    None,
                    1,
                    "completed",
                    delivery_attempt.attempt_id,
                    delivery_attempt.started_at,
                    now,
                    now,
                )
                for index in range(payload_delivery_records)
            ),
        )
    return delivery_attempt


def _exercise_payload_delivery_state_machine(
    store: SQLiteStore,
    *,
    initial_attempt: CheckpointPayloadDeliveryAttempt,
    page_size: int,
    expected_records: int,
) -> _PayloadDeliveryTransitionEvidence:
    """Drive five delivery transitions with one page-sized transaction each."""

    writer = CheckpointRestorePublicationWriter(store)
    final_attempt = CheckpointPayloadDeliveryAttempt(
        started_at=utc_now(),
        attempt_id="payload-attempt-transition-final",
        owner_instance_id=initial_attempt.owner_instance_id,
    )

    phase_evidence: list[tuple[int, int, int]] = []
    if not writer.begin_checkpoint_payload_delivery_attempt(initial_attempt):
        raise AssertionError("payload delivery initial attempt insert lost")
    if writer.abort_checkpoint_payload_delivery_attempt(initial_attempt):
        raise AssertionError("payload delivery abort accepted bound rows")
    phase_evidence.append(
        _transition_payload_delivery_fixture_pages(
            store,
            writer=writer,
            expected_delivery_state="completed",
            delivery_state="pending",
            expected_attempt=initial_attempt,
            delivery_attempt=None,
            owner_instance_id=initial_attempt.owner_instance_id,
            page_size=page_size,
        )
    )

    phase_evidence.append(
        _transition_payload_delivery_fixture_pages(
            store,
            writer=writer,
            expected_delivery_state="pending",
            delivery_state="confirmed",
            expected_attempt=None,
            delivery_attempt=initial_attempt,
            owner_instance_id=initial_attempt.owner_instance_id,
            page_size=page_size,
        )
    )
    phase_evidence.append(
        _transition_payload_delivery_fixture_pages(
            store,
            writer=writer,
            expected_delivery_state="confirmed",
            delivery_state="pending",
            expected_attempt=initial_attempt,
            delivery_attempt=None,
            owner_instance_id=initial_attempt.owner_instance_id,
            page_size=page_size,
        )
    )
    if not writer.abort_checkpoint_payload_delivery_attempt(initial_attempt):
        raise AssertionError("payload delivery compensated abort exact CAS lost")
    if writer.abort_checkpoint_payload_delivery_attempt(initial_attempt):
        raise AssertionError("stale payload delivery abort CAS was accepted")

    if not writer.begin_checkpoint_payload_delivery_attempt(final_attempt):
        raise AssertionError("payload delivery final attempt insert lost")
    if writer.ack_checkpoint_payload_delivery_attempt(final_attempt):
        raise AssertionError("payload delivery ACK accepted an empty attempt")
    phase_evidence.append(
        _transition_payload_delivery_fixture_pages(
            store,
            writer=writer,
            expected_delivery_state="pending",
            delivery_state="confirmed",
            expected_attempt=None,
            delivery_attempt=final_attempt,
            owner_instance_id=final_attempt.owner_instance_id,
            page_size=page_size,
        )
    )
    phase_evidence.append(
        _transition_payload_delivery_fixture_pages(
            store,
            writer=writer,
            expected_delivery_state="confirmed",
            delivery_state="completed",
            expected_attempt=final_attempt,
            delivery_attempt=final_attempt,
            owner_instance_id=final_attempt.owner_instance_id,
            page_size=page_size,
        )
    )
    if not writer.ack_checkpoint_payload_delivery_attempt(final_attempt):
        raise AssertionError("payload delivery completed ACK exact CAS lost")
    if writer.ack_checkpoint_payload_delivery_attempt(final_attempt):
        raise AssertionError("stale payload delivery ACK CAS was accepted")

    records_per_phase = phase_evidence[0][0]
    if records_per_phase != expected_records or any(
        records != records_per_phase
        for records, _pages, _maximum in phase_evidence
    ):
        raise AssertionError("payload delivery transition phases diverged")
    return _PayloadDeliveryTransitionEvidence(
        final_attempt=final_attempt,
        records_per_phase=records_per_phase,
        transition_rows=sum(records for records, _pages, _max in phase_evidence),
        transaction_pages=sum(pages for _records, pages, _max in phase_evidence),
        max_page_records=max(maximum for _records, _pages, maximum in phase_evidence),
        phase_page_counts=tuple(pages for _records, pages, _max in phase_evidence),
    )


def _transition_payload_delivery_fixture_pages(
    store: SQLiteStore,
    *,
    writer: CheckpointRestorePublicationWriter,
    expected_delivery_state: str,
    delivery_state: str,
    expected_attempt: CheckpointPayloadDeliveryAttempt | None,
    delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
    owner_instance_id: str,
    page_size: int,
) -> tuple[int, int, int]:
    """Transition one complete state using only a scalar cursor and one page."""

    after = None
    records_seen = 0
    transaction_pages = 0
    max_page_records = 0
    while True:
        with store.transaction():
            page = store.query_checkpoint_restore_payload_deliveries(
                delivery_state=expected_delivery_state,
                attempt_id=(
                    expected_attempt.attempt_id
                    if expected_attempt is not None
                    else None
                ),
                after=after,
                limit=page_size,
            )
            max_page_records = max(max_page_records, len(page.records))
            for publication in page.records:
                if not writer.transition_payload_delivery(
                    publication["publication_id"],
                    expected_delivery_state=expected_delivery_state,
                    delivery_state=delivery_state,
                    expected_attempt=expected_attempt,
                    delivery_attempt=delivery_attempt,
                    owner_instance_id=owner_instance_id,
                    recovery_lease_id=None,
                ):
                    raise AssertionError(
                        "payload delivery fixture transition exact CAS lost"
                    )
            records_seen += len(page.records)
            transaction_pages += 1
        if page.next_cursor is None:
            return records_seen, transaction_pages, max_page_records
        if not page.records or page.next_cursor == after:
            raise AssertionError("payload delivery page cursor did not advance")
        after = page.next_cursor


def _payload_delivery_fixture_contract(
    store: SQLiteStore,
    *,
    attempt: CheckpointPayloadDeliveryAttempt,
    expected_records: int,
) -> dict[str, int]:
    """Verify the synthetic completed-page multiset with SQL aggregation."""

    receipt = dumps(
        {
            "phases": [],
            "artifacts": [],
            "payload_delivery": {"state": "completed"},
            "payload_delivery_attempt": {
                "attempt_id": attempt.attempt_id,
                "started_at": attempt.started_at,
            },
        }
    )
    row = store.conn.execute(
        """
        WITH fixture AS (
            SELECT runtime_publications.*,
                   CAST(SUBSTR(publication_id, 26) AS INTEGER) AS seed_index
              FROM runtime_publications
             WHERE publication_id LIKE 'payload-delivery-history-%'
        )
        SELECT COUNT(*) AS total_rows,
               COALESCE(SUM(CASE WHEN
                   seed_index < 0 OR seed_index >= ? OR
                   publication_id IS NOT printf(
                       'payload-delivery-history-%08d', seed_index
                   ) OR
                   pid IS NOT printf('payload-delivery-pid-%08d', seed_index) OR
                   owner_instance_id IS NOT ? OR kind IS NOT 'checkpoint_restore' OR
                   state IS NOT 'committed' OR phase IS NOT 'reconciled' OR
                   plan_json IS NOT ? OR receipt_json IS NOT ? OR
                   error_json IS NOT NULL OR operation_reconciled IS NOT 1 OR
                   payload_delivery_state IS NOT 'completed' OR
                   payload_delivery_attempt_id IS NOT ? OR
                   payload_delivery_started_at IS NOT ?
                   THEN 1 ELSE 0 END), 0) AS mismatches
          FROM fixture
        """,
        (
            expected_records,
            attempt.owner_instance_id,
            dumps({}),
            receipt,
            attempt.attempt_id,
            attempt.started_at,
        ),
    ).fetchone()
    result = {
        "total_rows": int(row["total_rows"]),
        "mismatches": int(row["mismatches"]),
    }
    if result != {"total_rows": expected_records, "mismatches": 0}:
        raise AssertionError(
            f"payload delivery fixture multiset changed: {result}"
        )
    return result


def _payload_attempt_fixture_contract(
    store: SQLiteStore,
    *,
    attempt_history_records: int,
) -> dict[str, int]:
    """Aggregate control-row state without materializing historical attempts."""

    row = store.conn.execute(
        """
        SELECT COUNT(*) AS total_rows,
               COALESCE(SUM(state = 'preparing'), 0) AS preparing_rows,
               COALESCE(SUM(state = 'acked'), 0) AS acked_rows,
               COALESCE(SUM(state = 'aborted'), 0) AS aborted_rows
          FROM checkpoint_payload_delivery_attempts
        """
    ).fetchone()
    result = {
        "total_rows": int(row["total_rows"]),
        "preparing_rows": int(row["preparing_rows"]),
        "acked_rows": int(row["acked_rows"]),
        "aborted_rows": int(row["aborted_rows"]),
    }
    expected = {
        "total_rows": attempt_history_records + 2,
        "preparing_rows": 0,
        "acked_rows": ((attempt_history_records + 1) // 2) + 1,
        "aborted_rows": (attempt_history_records // 2) + 1,
    }
    if result != expected:
        raise AssertionError(
            f"payload delivery attempt control-row multiset changed: {result}"
        )
    return result


def _discard_payload_delivery_fixture(
    store: SQLiteStore,
    *,
    expected_records: int,
) -> None:
    """Remove only the synthetic publication range before legacy checks."""

    with store.transaction() as cursor:
        deleted = cursor.execute(
            "DELETE FROM runtime_publications "
            "WHERE publication_id LIKE 'payload-delivery-history-%'"
        )
        if deleted.rowcount != expected_records:
            raise AssertionError(
                "payload delivery fixture cleanup cardinality changed"
            )


def _seeded_publication_kind(
    index: int,
    *,
    unreconciled_records: int,
) -> str:
    if index < unreconciled_records:
        return "process_launch"
    return ("process_launch", "process_exec", "checkpoint_restore")[
        (index - unreconciled_records) % 3
    ]


def _seeded_publication_plan(
    index: int,
    *,
    unreconciled_records: int,
) -> str:
    if index >= unreconciled_records:
        return dumps({})
    return dumps(
        {
            "operation_id": f"scale-operation-{index:08d}",
            "operation_binding_version": 1,
            "pid": f"pid-history-{index:08d}",
            "launch_kind": "spawn",
            "parent_pid": None,
        }
    )


def _query_plan(
    store: SQLiteStore,
    sql: str,
    params: tuple[Any, ...],
) -> tuple[str, ...]:
    rows = store.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)
    return tuple(str(row[3]) for row in rows)


def _restore_class_attribute(
    target: type[Any],
    name: str,
    original_local: Any,
) -> None:
    if original_local is _MISSING_CLASS_ATTRIBUTE:
        if name in target.__dict__:
            delattr(target, name)
        return
    setattr(target, name, original_local)


def _publication_select_shape(sql: str) -> str | None:
    if not _is_publication_select(sql):
        return None
    exact_normalized = " ".join(str(sql).upper().split()).rstrip(";")
    if exact_normalized == _PUBLICATION_DOMAIN_SELECT:
        return "domain_validation"
    normalized = _normalize_publication_select(exact_normalized)
    if _OPERATION_RECONCILIATION_SELECT_RE.fullmatch(normalized):
        return "operation_reconciliation"
    if _RECOVERY_SELECT_RE.fullmatch(normalized):
        return "recovery"
    if _PAYLOAD_DELIVERY_PAGE_SELECT_RE.fullmatch(normalized):
        return "payload_delivery_page"
    if _PAYLOAD_DELIVERY_ATTEMPT_SELECT_RE.fullmatch(normalized):
        return "payload_delivery_attempt"
    if _PAYLOAD_DELIVERY_ATTEMPTS_SELECT_RE.fullmatch(normalized):
        return "payload_delivery_attempts"
    if _PAYLOAD_DELIVERY_ATTEMPT_READBACK_RE.fullmatch(normalized):
        return "payload_attempt_readback"
    if normalized == _EXACT_PUBLICATION_SELECT:
        return "exact_publication"
    if _ORPHAN_ANTIJOIN_SELECT_RE.fullmatch(normalized):
        return "orphan_antijoin"
    return "unreviewed"


def _is_publication_select(sql: str) -> bool:
    normalized = _strip_sql_comments(str(sql)).casefold()
    return (
        re.search(r"\bselect\b", normalized) is not None
        and (
            "runtime_publications" in normalized
            or "checkpoint_payload_delivery_attempts" in normalized
        )
    )


def _is_publication_statement(sql: str) -> bool:
    normalized = _strip_sql_comments(str(sql)).casefold()
    return (
        "runtime_publications" in normalized
        or "checkpoint_payload_delivery_attempts" in normalized
    )


def _publication_statement_shape(sql: str) -> str | None:
    if not _is_publication_statement(sql):
        return None
    raw_sql = str(sql)
    raw_normalized = " ".join(raw_sql.upper().split()).rstrip(";")
    raw_normalized_literals = _normalize_publication_select(raw_normalized)
    if raw_normalized_literals == _SQLITE_V4_MANIFEST_SCHEMA_PROBE_SHAPE:
        manifest_tables = tuple(
            literal[1:-1].replace("''", "'")
            for literal in _SQL_TEXT_LITERAL_RE.findall(raw_sql)
        )
        if manifest_tables == _SQLITE_V4_MANIFEST_TABLES:
            return "v4_manifest_schema_probe"
    uncommented = _strip_sql_comments(raw_sql)
    normalized = " ".join(uncommented.upper().split()).rstrip(";")
    normalized_literals = _normalize_publication_select(normalized)
    if re.fullmatch(
        r"SELECT NAME, SQL FROM SQLITE_MASTER WHERE TYPE = \?"
        r" AND NAME IN \(\?(?:, \?)*\)",
        normalized_literals,
    ):
        return "keyset_collation_schema_probe"
    if normalized == "PRAGMA TABLE_INFO(RUNTIME_PUBLICATIONS)":
        return "schema_probe"
    if normalized == "PRAGMA TABLE_INFO(CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS)":
        return "payload_attempt_schema_probe"
    if _PUBLICATION_RECONCILIATION_UPDATE_RE.fullmatch(normalized_literals):
        return "operation_reconciliation_update"
    if _PUBLICATION_RECONCILIATION_INVALIDATION_RE.fullmatch(normalized_literals):
        return "operation_reconciliation_invalidation"
    payload_transition = re.sub(
        r"(PAYLOAD_DELIVERY_ATTEMPT_ID|PAYLOAD_DELIVERY_STARTED_AT) = NULL",
        r"\1 = ?",
        normalized_literals,
    ).replace("AND NULL IS NULL", "AND ? IS NULL")
    if _normalize_schema_sql(payload_transition) in {
        _normalize_schema_sql(_PAYLOAD_DELIVERY_TRANSITION_UPDATE),
        _normalize_schema_sql(_PAYLOAD_DELIVERY_PENDING_TRANSITION_UPDATE),
        _normalize_schema_sql(_PAYLOAD_DELIVERY_EMPTY_TRANSITION_UPDATE),
    }:
        return "payload_delivery_transition_update"
    if _PAYLOAD_DELIVERY_ATTEMPT_BEGIN_RE.fullmatch(normalized_literals):
        return "payload_attempt_begin"
    if normalized_literals == _normalize_publication_select(
        _PAYLOAD_DELIVERY_ATTEMPT_ACK_SQL
    ):
        return "payload_attempt_ack"
    if normalized_literals == _normalize_publication_select(
        _PAYLOAD_DELIVERY_ATTEMPT_ABORT_SQL
    ):
        return "payload_attempt_abort"
    if re.search(r"\bSELECT\b", normalized) is not None:
        return _publication_select_shape(sql) or "unreviewed"
    if (
        normalized.startswith(
            "CREATE TABLE IF NOT EXISTS RUNTIME_PUBLICATIONS ("
        )
        and " SELECT " not in normalized
    ):
        return "schema_table"
    if (
        normalized.startswith(
            "CREATE TABLE IF NOT EXISTS CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS ("
        )
        and " SELECT " not in normalized
    ):
        return "schema_payload_attempt_table"
    index_match = re.fullmatch(
        r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS ([A-Z0-9_]+)"
        r" ON (?:RUNTIME_PUBLICATIONS|CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS)"
        r"(?:\s|\().+",
        normalized,
    )
    if index_match is not None:
        index_name = index_match.group(1).lower()
        if index_name in _PUBLICATION_SCHEMA_INDEXES:
            return f"schema_index:{index_name}"
    return "unreviewed"


def _is_reviewed_publication_trace(sql: str) -> bool:
    shape = _publication_select_shape(sql)
    if shape is None:
        return True
    return shape != "unreviewed"


def _normalize_publication_select(sql: str) -> str:
    normalized = " ".join(str(sql).upper().split()).rstrip(";")
    normalized = _SQL_TEXT_LITERAL_RE.sub("?", normalized)
    return _SQL_INTEGER_LITERAL_RE.sub("?", normalized)


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments without treating comment markers in quotes as syntax."""

    output: list[str] = []
    index = 0
    length = len(sql)
    while index < length:
        current = sql[index]
        if current in {"'", '"', "`"}:
            delimiter = current
            output.append(current)
            index += 1
            while index < length:
                current = sql[index]
                output.append(current)
                index += 1
                if current != delimiter:
                    continue
                if index < length and sql[index] == delimiter:
                    output.append(sql[index])
                    index += 1
                    continue
                break
            continue
        if current == "[":
            output.append(current)
            index += 1
            while index < length:
                current = sql[index]
                output.append(current)
                index += 1
                if current == "]":
                    break
            continue
        if sql.startswith("--", index):
            line_end_candidates = (
                position
                for position in (
                    sql.find("\n", index + 2),
                    sql.find("\r", index + 2),
                )
                if position >= 0
            )
            line_end = min(line_end_candidates, default=-1)
            if line_end < 0:
                output.append(" ")
                break
            output.append("\n")
            index = line_end + 1
            continue
        if sql.startswith("/*", index):
            comment_end = sql.find("*/", index + 2)
            output.append(" ")
            index = length if comment_end < 0 else comment_end + 2
            continue
        output.append(current)
        index += 1
    return "".join(output)


def _publication_convergence(
    store: SQLiteStore,
    *,
    total_records: int,
    unreconciled_records: int,
) -> dict[str, int]:
    publication = store.conn.execute(
        """
        WITH seeded AS (
            SELECT runtime_publications.*,
                   CAST(SUBSTR(publication_id, 21) AS INTEGER) AS seed_index
              FROM runtime_publications
             WHERE publication_id LIKE 'publication-history-%'
        )
        SELECT (SELECT COUNT(*) FROM runtime_publications) AS total_rows,
               COUNT(*) AS seeded_rows,
               COALESCE(SUM(CASE WHEN
                   seed_index < 0 OR seed_index >= ? OR
                   publication_id IS NOT printf('publication-history-%08d', seed_index) OR
                   pid IS NOT printf('pid-history-%08d', seed_index) OR
                   owner_instance_id IS NOT 'scale-owner'
                   THEN 1 ELSE 0 END), 0) AS identity_mismatches,
               COALESCE(SUM(CASE WHEN
                   state IS NOT 'committed' OR phase IS NOT 'committed' OR
                   operation_reconciled IS NOT 1 OR error_json IS NOT NULL OR
                   receipt_json IS NOT ? OR
                   kind IS NOT CASE
                       WHEN seed_index < ? THEN 'process_launch'
                       WHEN ((seed_index - ?) % 3) = 0 THEN 'process_launch'
                       WHEN ((seed_index - ?) % 3) = 1 THEN 'process_exec'
                       ELSE 'checkpoint_restore'
                   END OR
                   CASE WHEN seed_index < ? THEN
                       json_extract(plan_json, '$.operation_id')
                           IS NOT printf('scale-operation-%08d', seed_index) OR
                       json_extract(plan_json, '$.operation_binding_version') IS NOT 1 OR
                       json_extract(plan_json, '$.pid')
                           IS NOT printf('pid-history-%08d', seed_index) OR
                       json_extract(plan_json, '$.launch_kind') IS NOT 'spawn' OR
                       json_type(plan_json, '$.parent_pid') IS NOT 'null'
                   ELSE plan_json IS NOT ? END
                   THEN 1 ELSE 0 END), 0) AS convergence_mismatches
          FROM seeded
        """,
        (
            total_records,
            dumps({"phases": [], "artifacts": []}),
            unreconciled_records,
            unreconciled_records,
            unreconciled_records,
            unreconciled_records,
            dumps({}),
        ),
    ).fetchone()
    operations = store.conn.execute(
        """
        WITH seeded AS (
            SELECT operations.*,
                   CAST(SUBSTR(operation_id, 17) AS INTEGER) AS seed_index
              FROM operations
             WHERE operation_id LIKE 'scale-operation-%'
        )
        SELECT (SELECT COUNT(*) FROM operations) AS total_operations,
               COUNT(*) AS bound_operations,
               COALESCE(SUM(CASE WHEN
                   seed_index < 0 OR seed_index >= ? OR
                   operation_id IS NOT printf('scale-operation-%08d', seed_index) OR
                   root_operation_id IS NOT operation_id OR
                   parent_operation_id IS NOT NULL OR kind IS NOT 'runtime' OR
                   name IS NOT 'process.spawn' OR actor IS NOT 'runtime' OR
                   pid IS NOT printf('pid-history-%08d', seed_index) OR
                   state IS NOT 'terminal' OR outcome IS NOT 'succeeded' OR
                   expected_roles_json IS NOT ? OR
                   completed_at IS NULL OR updated_at IS NOT completed_at OR
                   runtime_publication_id
                       IS NOT printf('publication-history-%08d', seed_index) OR
                   json_extract(metadata_json, '$.runtime_publication_id')
                       IS NOT printf('publication-history-%08d', seed_index) OR
                   json_extract(metadata_json, '$.runtime_publication_kind')
                       IS NOT 'process_launch' OR
                   json_extract(metadata_json, '$.runtime_publication_bound') IS NOT 1 OR
                   json_extract(metadata_json, '$.runtime_publication_binding_version') IS NOT 1 OR
                   json_extract(metadata_json, '$.runtime_publication_state')
                       IS NOT 'committed' OR
                   json_extract(metadata_json, '$.runtime_publication_phase')
                       IS NOT 'committed' OR
                   json_extract(metadata_json, '$.runtime_publication_reconciled') IS NOT 1 OR
                   json_extract(metadata_json, '$.runtime_publication_original_operation_state')
                       IS NOT 'running' OR
                   json_extract(metadata_json, '$.runtime_publication_original_operation_outcome')
                       IS NOT 'pending'
                   THEN 1 ELSE 0 END), 0) AS operation_mismatches
          FROM seeded
        """,
        (unreconciled_records, dumps([])),
    ).fetchone()
    return {
        "total_rows": int(publication["total_rows"]),
        "seeded_rows": int(publication["seeded_rows"]),
        "identity_mismatches": int(publication["identity_mismatches"]),
        "convergence_mismatches": int(publication["convergence_mismatches"]),
        "total_operations": int(operations["total_operations"]),
        "bound_operations": int(operations["bound_operations"]),
        "operation_mismatches": int(operations["operation_mismatches"]),
    }


def _publication_terminal_contract(
    store: SQLiteStore,
    *,
    total_records: int,
    unreconciled_records: int,
) -> dict[str, int]:
    """Compare every seeded publication to its exact terminal row shape.

    The aggregate implements multiset subtraction without materializing
    history in Python: every expected row is unique by publication id, an
    exact row contributes once, and total/matched cardinalities expose missing
    and unexpected rows independently.
    """

    receipt = dumps({"phases": [], "artifacts": []})
    plan_pattern = _seeded_publication_plan(
        0,
        unreconciled_records=unreconciled_records,
    ).replace(
        "scale-operation-00000000",
        "scale-operation-%08d",
    ).replace(
        "pid-history-00000000",
        "pid-history-%08d",
    )
    row = store.conn.execute(
        """
        WITH seeded AS (
            SELECT runtime_publications.*,
                   CAST(SUBSTR(publication_id, 21) AS INTEGER) AS seed_index
              FROM runtime_publications
             WHERE publication_id LIKE 'publication-history-%'
        ),
        classified AS (
            SELECT seeded.*,
                   CASE WHEN
                       seed_index >= 0 AND seed_index < ? AND
                       publication_id IS printf('publication-history-%08d', seed_index) AND
                       pid IS printf('pid-history-%08d', seed_index) AND
                       owner_instance_id IS 'scale-owner' AND
                       state IS 'committed' AND phase IS 'committed' AND
                       operation_reconciled IS 1 AND error_json IS NULL AND
                       receipt_json IS ? AND
                       kind IS CASE
                           WHEN seed_index < ? THEN 'process_launch'
                           WHEN ((seed_index - ?) % 3) = 0 THEN 'process_launch'
                           WHEN ((seed_index - ?) % 3) = 1 THEN 'process_exec'
                           ELSE 'checkpoint_restore'
                       END AND
                       plan_json IS CASE WHEN seed_index < ?
                           THEN printf(
                               ?,
                               seed_index,
                               seed_index
                           )
                           ELSE ?
                       END
                       THEN 1 ELSE 0
                   END AS exact_row
              FROM seeded
        )
        SELECT (SELECT COUNT(*) FROM runtime_publications) AS total_rows,
               COALESCE(SUM(exact_row), 0) AS exact_rows,
               COALESCE(SUM(
                   CASE WHEN seed_index < ? AND exact_row = 1 THEN 1 ELSE 0 END
               ), 0) AS reconciled_backlog_rows
          FROM classified
        """,
        (
            total_records,
            receipt,
            unreconciled_records,
            unreconciled_records,
            unreconciled_records,
            unreconciled_records,
            plan_pattern,
            dumps({}),
            unreconciled_records,
        ),
    ).fetchone()
    total_rows = int(row["total_rows"])
    exact_rows = int(row["exact_rows"])
    return {
        "missing_rows": total_records - exact_rows,
        "unexpected_rows": total_rows - exact_rows,
        "reconciled_backlog_rows": int(row["reconciled_backlog_rows"]),
    }


def _require_publication_index_contract(store: SQLiteStore) -> None:
    index_rows = {
        str(row["name"]): row
        for row in store.conn.execute("PRAGMA index_list(runtime_publications)")
    }
    reconciliation_name = "idx_runtime_publications_operation_reconciliation"
    reconciliation = index_rows.get(reconciliation_name)
    if reconciliation is None or int(reconciliation["partial"]) != 0:
        raise AssertionError("publication reconciliation index contract changed")
    reconciliation_columns = tuple(
        str(row["name"])
        for row in store.conn.execute(
            f"PRAGMA index_info({reconciliation_name})"
        )
    )
    if reconciliation_columns != (
        "state",
        "kind",
        "operation_reconciled",
        "created_at",
        "publication_id",
    ):
        raise AssertionError("publication reconciliation index columns changed")

    domain_name = "idx_runtime_publications_invalid_domain"
    domain = index_rows.get(domain_name)
    if domain is None or int(domain["partial"]) != 1:
        raise AssertionError("publication domain index predicate changed")
    domain_columns = tuple(
        str(row["name"])
        for row in store.conn.execute(f"PRAGMA index_info({domain_name})")
    )
    domain_sql_row = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (domain_name,),
    ).fetchone()
    domain_sql = (
        _normalize_schema_sql(str(domain_sql_row["sql"]))
        if domain_sql_row is not None
        else ""
    )
    if (
        domain_columns != ("publication_id",)
        or domain_sql != _normalize_schema_sql(_EXPECTED_DOMAIN_INDEX_SQL)
    ):
        raise AssertionError("publication domain index definition changed")

    _require_exact_index(
        store,
        table="runtime_publications",
        name="idx_runtime_publications_payload_delivery_page",
        columns=("payload_delivery_state", "created_at", "publication_id"),
        partial=True,
        unique=False,
        expected_sql=_EXPECTED_PAYLOAD_PAGE_INDEX_SQL,
    )
    _require_exact_index(
        store,
        table="runtime_publications",
        name="idx_runtime_publications_payload_delivery_attempt",
        columns=(
            "payload_delivery_attempt_id",
            "payload_delivery_state",
            "created_at",
            "publication_id",
        ),
        partial=True,
        unique=False,
        expected_sql=_EXPECTED_PAYLOAD_ATTEMPT_INDEX_SQL,
    )
    _require_exact_index(
        store,
        table="runtime_publications",
        name="idx_runtime_publications_payload_delivery_guard",
        columns=(
            "payload_delivery_attempt_id",
            "payload_delivery_state",
            "owner_instance_id",
            "operation_reconciled",
            "created_at",
            "publication_id",
        ),
        partial=True,
        unique=False,
        expected_sql=_EXPECTED_PAYLOAD_GUARD_INDEX_SQL,
    )
    _require_exact_index(
        store,
        table="runtime_publications",
        name="idx_runtime_publications_payload_delivery_guard",
        columns=(
            "payload_delivery_attempt_id",
            "payload_delivery_state",
            "owner_instance_id",
            "operation_reconciled",
            "created_at",
            "publication_id",
        ),
        partial=True,
        unique=False,
        expected_sql=_EXPECTED_PAYLOAD_GUARD_INDEX_SQL,
    )
    _require_exact_index(
        store,
        table="checkpoint_payload_delivery_attempts",
        name="idx_checkpoint_payload_delivery_attempts_state",
        columns=("state", "started_at", "attempt_id"),
        partial=False,
        unique=False,
        expected_sql=_EXPECTED_ATTEMPT_STATE_INDEX_SQL,
    )
    _require_exact_index(
        store,
        table="checkpoint_payload_delivery_attempts",
        name="idx_checkpoint_payload_delivery_attempts_preparing",
        columns=("state",),
        partial=True,
        unique=True,
        expected_sql=_EXPECTED_PREPARING_ATTEMPT_INDEX_SQL,
    )


def _require_exact_index(
    store: SQLiteStore,
    *,
    table: str,
    name: str,
    columns: tuple[str, ...],
    partial: bool,
    unique: bool,
    expected_sql: str,
) -> None:
    index_rows = {
        str(row["name"]): row
        for row in store.conn.execute(f"PRAGMA index_list({table})")
    }
    selected = index_rows.get(name)
    if selected is None or (
        bool(selected["partial"]) is not partial
        or bool(selected["unique"]) is not unique
    ):
        raise AssertionError(f"{name} inventory contract changed")
    actual_columns = tuple(
        str(row["name"])
        for row in store.conn.execute(f"PRAGMA index_info({name})")
    )
    sql_row = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()
    actual_sql = (
        _normalize_schema_sql(str(sql_row["sql"]))
        if sql_row is not None
        else ""
    )
    if actual_columns != columns or actual_sql != _normalize_schema_sql(
        expected_sql
    ):
        raise AssertionError(f"{name} definition changed")


def _normalize_schema_sql(sql: str) -> str:
    normalized = " ".join(str(sql).upper().split())
    normalized = re.sub(r"\(\s+", "(", normalized)
    return re.sub(r"\s+\)", ")", normalized)


def _assert_publication_trace_contract(
    actual: Counter[str],
    *,
    unreconciled_records: int,
    payload_delivery_records: int,
    expected_handler_query_calls: int,
    expected_payload_delivery_page_calls: int,
) -> None:
    # Existing-store validation runs against an isolated safety snapshot. This
    # ledger covers only the subsequently opened measured main connection.
    expected = Counter(
        {
            "schema_probe": 1,
            "payload_attempt_schema_probe": 1,
            "keyset_collation_schema_probe": 1,
            "v4_manifest_schema_probe": 1,
            "schema_table": 1,
            "schema_payload_attempt_table": 1,
            "domain_validation": 1,
            "recovery": 30,
            "operation_reconciliation": expected_handler_query_calls + 9,
            "exact_publication": (
                (unreconciled_records * 2) + (payload_delivery_records * 10)
            ),
            "orphan_antijoin": 1,
            "operation_reconciliation_update": unreconciled_records,
            "operation_reconciliation_invalidation": unreconciled_records * 2,
            "payload_delivery_attempts": 1,
            "payload_delivery_page": (
                2 + (2 * expected_payload_delivery_page_calls)
            ),
            "payload_delivery_attempt": (
                3 * expected_payload_delivery_page_calls
            ),
            "payload_delivery_transition_update": (
                5 * payload_delivery_records
            ),
            "payload_attempt_begin": 2,
            "payload_attempt_ack": 3,
            "payload_attempt_abort": 3,
            "payload_attempt_readback": 4,
        }
    )
    expected.update(
        {f"schema_index:{name}": 1 for name in _PUBLICATION_SCHEMA_INDEXES}
    )
    if actual != expected:
        raise AssertionError(
            "Runtime reopen publication statement ledger changed: "
            f"actual={dict(actual)} expected={dict(expected)}"
        )


def _assert_publication_query_key_contract(
    queries: list[tuple[str, tuple[Any, ...], int, str]],
    *,
    unreconciled_records: int,
    payload_delivery_records: int,
    expected_handler_query_calls: int,
    expected_payload_delivery_page_calls: int,
) -> None:
    actual = Counter(
        _publication_query_key(shape, params)
        for _sql, params, _row_count, shape in queries
    )
    expected: Counter[tuple[Any, ...]] = Counter({("domain_validation",): 1})
    for kind, states in RECOVERY_QUERY_STATES.items():
        for state in states:
            for operation_reconciled in (0, 1):
                expected[("recovery", kind, state, operation_reconciled)] = 1
    for kind, states in TERMINAL_RECONCILIATION_STATES.items():
        for state in states:
            expected[("operation_reconciliation", kind, state)] += (
                expected_handler_query_calls
                if (kind, state) == ("process_launch", "committed")
                else 1
            )
    for index in range(unreconciled_records):
        expected[("exact_publication", f"publication-history-{index:08d}")] = 2
    for index in range(payload_delivery_records):
        expected[("exact_publication", f"payload-delivery-history-{index:08d}")] = 10
    expected[("orphan_antijoin", "created", "process_launch")] = 1
    expected[("payload_delivery_attempts", "preparing")] = 1
    expected[("payload_delivery_page", "pending", None)] = (
        2 + (2 * expected_payload_delivery_page_calls)
    )
    expected[
        (
            "payload_delivery_attempt",
            "completed",
            "payload-attempt-transition-initial",
        )
    ] = expected_payload_delivery_page_calls
    expected[
        (
            "payload_delivery_attempt",
            "confirmed",
            "payload-attempt-transition-initial",
        )
    ] = expected_payload_delivery_page_calls
    expected[
        (
            "payload_delivery_attempt",
            "confirmed",
            "payload-attempt-transition-final",
        )
    ] = expected_payload_delivery_page_calls
    if actual != expected:
        raise AssertionError(
            "publication repository query-key multiset changed: "
            f"actual={dict(actual)} expected={dict(expected)}"
        )


def _publication_query_key(
    shape: str,
    params: tuple[Any, ...],
) -> tuple[Any, ...]:
    if shape == "domain_validation":
        return (shape,)
    if shape == "recovery" and len(params) >= 3:
        return (shape, str(params[0]), str(params[1]), int(params[2]))
    if shape == "operation_reconciliation" and len(params) >= 2:
        return (shape, str(params[0]), str(params[1]))
    if shape == "exact_publication" and params:
        return (shape, str(params[0]))
    if shape == "orphan_antijoin" and len(params) >= 2:
        return (shape, str(params[0]), str(params[1]))
    if shape == "payload_delivery_attempts":
        return (shape, "preparing")
    if shape == "payload_delivery_page" and params:
        return (shape, str(params[0]), None)
    if shape == "payload_delivery_attempt" and len(params) >= 2:
        return (shape, str(params[0]), str(params[1]))
    raise AssertionError(
        f"publication repository query parameters changed: shape={shape!r} "
        f"params={params!r}"
    )


def _assert_structural_contract(result: PublicationScaleResult) -> None:
    if result.publication_select_calls != result.publication_query_calls:
        raise AssertionError(
            "direct publication SELECTs bypassed repository row accounting"
        )
    if result.publication_schema_probe_calls != 1:
        raise AssertionError("publication measured-main schema probe changed")
    if result.publication_ddl_calls != 12:
        raise AssertionError("publication initialization DDL ledger changed")
    if result.publication_update_calls != result.unreconciled_records:
        raise AssertionError("publication reconciliation update ledger changed")
    if result.publication_invalidation_calls != result.unreconciled_records * 2:
        raise AssertionError("publication operation invalidation ledger changed")
    if (
        result.seeded_rows_after != result.total_records
        or result.seeded_identity_mismatches != 0
        or result.seeded_convergence_mismatches != 0
        or result.seeded_terminal_missing_rows != 0
        or result.seeded_terminal_unexpected_rows != 0
    ):
        raise AssertionError("seeded publication history did not converge exactly")
    if (
        result.total_operations_after != result.unreconciled_records
        or result.bound_operations_after != result.unreconciled_records
        or result.bound_operation_mismatches != 0
    ):
        raise AssertionError("bound publication operations did not converge exactly")
    if result.handler_reconciled_records != result.unreconciled_records:
        raise AssertionError("Runtime handler did not reconcile the exact backlog")
    if result.handler_sample_records != min(
        result.unreconciled_records,
        result.page_size,
    ):
        raise AssertionError("Runtime handler diagnostics are not page bounded")
    if result.handler_query_calls != result.expected_handler_query_calls:
        raise AssertionError("Runtime handler query count is not page proportional")
    if (
        result.handler_raw_rows_fetched
        != result.expected_handler_raw_rows_fetched
    ):
        raise AssertionError("Runtime handler fetched more than page lookahead")
    if result.max_rows_fetched > result.page_size + 1:
        raise AssertionError("publication recovery query exceeded its hard page bound")
    if result.payload_delivery_records != (2 * result.page_size) + 5:
        raise AssertionError("payload delivery fixture did not traverse exactly")
    if (
        result.payload_delivery_query_calls
        != result.expected_payload_delivery_query_calls
    ):
        raise AssertionError("payload delivery query count is not page proportional")
    if (
        result.payload_delivery_raw_rows_fetched
        != result.expected_payload_delivery_raw_rows_fetched
    ):
        raise AssertionError("payload delivery fetched more than page lookahead")
    if result.payload_delivery_max_rows_fetched > result.page_size + 1:
        raise AssertionError("payload delivery query exceeded its hard page bound")
    if (
        result.payload_delivery_transition_phases != 5
        or result.payload_delivery_transition_rows
        != result.payload_delivery_records * 5
        or result.payload_delivery_transition_transactions
        != result.expected_payload_delivery_transition_transactions
        or result.payload_delivery_transition_transactions
        != result.expected_payload_delivery_query_calls
        or result.payload_delivery_transition_max_records > result.page_size
        or len(result.payload_delivery_transition_page_counts) != 5
        or any(
            count
            != result.expected_payload_delivery_transition_transactions // 5
            for count in result.payload_delivery_transition_page_counts
        )
        or result.payload_delivery_transition_update_calls
        != result.payload_delivery_transition_rows
    ):
        raise AssertionError(
            "payload delivery transitions were not one bounded transaction per page"
        )
    if (
        result.payload_delivery_pending_query_calls != 2
        or result.payload_delivery_pending_rows_fetched != 0
    ):
        raise AssertionError(
            "empty global pending delivery checks changed cardinality"
        )
    if result.payload_delivery_fixture_mismatches != 0:
        raise AssertionError("payload delivery fixture changed during traversal")
    _assert_indexed_keyset_plan(
        result.payload_delivery_query_plan_first,
        index=result.payload_delivery_attempt_index,
        exact_constraint="PAYLOAD_DELIVERY_ATTEMPT_ID=? AND PAYLOAD_DELIVERY_STATE=?",
        label="payload delivery initial attempt page",
    )
    _assert_indexed_keyset_plan(
        result.payload_delivery_query_plan_resumed,
        index=result.payload_delivery_attempt_index,
        exact_constraint="PAYLOAD_DELIVERY_ATTEMPT_ID=? AND PAYLOAD_DELIVERY_STATE=?",
        label="payload delivery resumed attempt page",
    )
    _assert_indexed_keyset_plan(
        result.payload_delivery_pending_query_plan,
        index=result.payload_delivery_page_index,
        exact_constraint="PAYLOAD_DELIVERY_STATE=?",
        label="payload delivery global pending page",
    )
    if (
        result.payload_attempt_query_calls != 1
        or result.payload_attempt_rows_fetched != 0
        or result.payload_attempt_max_rows_fetched != 0
    ):
        raise AssertionError(
            "preparing-attempt recovery materialized historical control rows"
        )
    _assert_indexed_keyset_plan(
        result.payload_attempt_query_plan,
        index=result.payload_attempt_index,
        exact_constraint="STATE=?",
        label="payload delivery preparing-attempt page",
    )
    if (
        result.payload_attempt_total_rows_after
        != result.payload_attempt_history_records + 2
        or result.payload_attempt_preparing_rows_after != 0
        or result.payload_attempt_acked_rows_after
        != ((result.payload_attempt_history_records + 1) // 2) + 1
        or result.payload_attempt_aborted_rows_after
        != (result.payload_attempt_history_records // 2) + 1
    ):
        raise AssertionError("payload delivery control rows did not converge")
    if (
        result.payload_attempt_begin_calls != 2
        or result.payload_attempt_ack_calls != 3
        or result.payload_attempt_abort_calls != 3
        or result.payload_attempt_readback_calls != 4
    ):
        raise AssertionError("payload delivery exact CAS statement ledger changed")
    _assert_payload_attempt_guard_plan(
        result.payload_attempt_ack_query_plan,
        label="ACK",
        minimum_publication_searches=5,
    )
    _assert_payload_attempt_guard_plan(
        result.payload_attempt_abort_query_plan,
        label="abort",
        minimum_publication_searches=1,
    )
    _assert_payload_attempt_guard_plan(
        result.payload_attempt_readback_query_plan,
        label="readback",
        minimum_publication_searches=0,
    )
    if (
        result.domain_validation_query_calls != 1
        or result.domain_validation_rows_fetched != 0
    ):
        raise AssertionError(
            "runtime publication domain validation was not exact and bounded"
        )
    domain_plan = "\n".join(result.domain_validation_query_plan)
    if result.domain_validation_index not in domain_plan:
        raise AssertionError(
            "runtime publication domain validation scanned publication history"
        )
    expected_reconciliation_queries = (
        sum(len(states) for states in TERMINAL_RECONCILIATION_STATES.values())
        - 1
        + result.expected_handler_query_calls
    )
    if result.reconciliation_query_calls != expected_reconciliation_queries:
        raise AssertionError(
            "runtime reopen did not issue the expected bounded terminal pages"
        )
    for plan in (
        result.reconciliation_query_plan_first,
        result.reconciliation_query_plan_resumed,
    ):
        details = "\n".join(plan)
        if result.reconciliation_index not in details:
            raise AssertionError("actual publication reconciliation query missed its index")
        if any(
            detail.strip().startswith("SCAN runtime_publications")
            for detail in plan
        ):
            raise AssertionError(
                f"actual publication reconciliation query scanned history: {details}"
            )
        normalized_details = details.upper()
        if "(STATE=? AND KIND=? AND OPERATION_RECONCILED=?" not in normalized_details:
            raise AssertionError(
                "publication reconciliation plan lost its exact search constraints"
            )
        if "USE TEMP B-TREE" in normalized_details:
            raise AssertionError(
                "publication reconciliation plan requires a temporary sort"
            )


def _assert_indexed_keyset_plan(
    plan: tuple[str, ...],
    *,
    index: str,
    exact_constraint: str,
    label: str,
) -> None:
    details = "\n".join(plan)
    normalized = details.upper()
    if index not in details:
        raise AssertionError(f"{label} missed its exact index: {details}")
    if exact_constraint not in normalized:
        raise AssertionError(f"{label} lost its search constraints: {details}")
    if "SCAN " in normalized or "USE TEMP B-TREE" in normalized:
        raise AssertionError(f"{label} scanned or sorted durable history: {details}")


def _assert_payload_attempt_guard_plan(
    plan: tuple[str, ...],
    *,
    label: str,
    minimum_publication_searches: int,
) -> None:
    details = "\n".join(plan)
    normalized = details.upper()
    if (
        "CHECKPOINT_PAYLOAD_DELIVERY_ATTEMPTS" not in normalized
        or "ATTEMPT_ID=?" not in normalized
        or "SCAN " in normalized
        or "USE TEMP B-TREE" in normalized
    ):
        raise AssertionError(
            f"payload delivery {label} control CAS lost its PK lookup: {details}"
        )
    publication_index = "IDX_RUNTIME_PUBLICATIONS_PAYLOAD_DELIVERY_GUARD"
    if normalized.count(publication_index) < minimum_publication_searches:
        raise AssertionError(
            f"payload delivery {label} guard missed its attempt index: {details}"
        )
