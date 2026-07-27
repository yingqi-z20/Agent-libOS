from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    ObjectPayloadRecoverySummary,
    ObjectTask,
    ObjectTaskRecoveryCursor,
    ObjectTaskRecoveryPage,
    ObjectTaskRecoverySummary,
    ObjectTaskStatus,
)
from agent_libos.storage import SQLiteStore
from agent_libos.utils.serde import dumps


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
# Windows hosted runners can spend several minutes in the SQLite recovery
# stress fixture when four xdist workers contend for the same filesystem. This
# contract does not assert latency; the lane-level deadline remains the hard
# upper bound for a genuinely stuck recovery.
@pytest.mark.timeout(600 if os.name == "nt" else 120)
def test_runtime_object_and_task_recovery_is_keyset_paged_and_bounded(
    backend: str,
    tmp_path: Path,
) -> None:
    page_size = 37
    object_count = 1_205
    active_task_count = 1_201
    missing_result_count = 3
    notification_count = 3
    with _runtime_target(backend, tmp_path, page_size=page_size) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            assert runtime.object_tasks.shutdown()
            _seed_recovery_rows(
                runtime.store,
                object_count=object_count,
                active_task_count=active_task_count,
                missing_result_count=missing_result_count,
                notification_count=notification_count,
            )
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            payload_summary = reopened.recovered_missing_object_payloads
            task_summary = reopened.recovered_object_tasks
            assert isinstance(payload_summary, ObjectPayloadRecoverySummary)
            assert payload_summary.total_count == object_count
            assert len(payload_summary.sample_oids) == page_size
            assert payload_summary.truncated
            assert isinstance(task_summary, ObjectTaskRecoverySummary)
            assert task_summary.abandoned_total == active_task_count
            assert task_summary.result_unavailable_total == missing_result_count
            assert task_summary.notification_retried_total == notification_count
            assert len(task_summary.abandoned_sample) == page_size
            assert task_summary.truncated

            released = reopened.store.select_table_rows(
                "objects",
                "lifecycle_state = ?",
                ("released",),
            )
            assert len(released) == object_count
            assert reopened.store.select_table_rows("object_links") == []
            capabilities = reopened.store.select_table_rows("capabilities")
            assert len(capabilities) == object_count
            assert {row["status"] for row in capabilities} == {"revoked"}
            tasks = reopened.store.select_table_rows("object_tasks")
            statuses = [str(row["status"]) for row in tasks]
            assert statuses.count(ObjectTaskStatus.ABANDONED.value) == active_task_count
            assert (
                statuses.count(
                    ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN.value
                )
                == missing_result_count
            )
        finally:
            reopened.close()


def test_sqlite_object_recovery_queries_use_recovery_indexes() -> None:
    config = _config(page_size=2)
    store = SQLiteStore(":memory:", config=config)
    try:
        payload_plan = store.conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT oid, created_at FROM objects
             WHERE lifecycle_state = ? AND payload_json IN (
               '{"present": true, "storage": "runtime_memory"}',
               '{"storage": "runtime_memory", "present": true}',
               '{"present":true,"storage":"runtime_memory"}',
               '{"storage":"runtime_memory","present":true}'
             )
             AND (created_at, oid) > (?, ?)
             ORDER BY created_at, oid LIMIT ?
            """,
            ("live", "2026-01-01T00:00:00Z", "oid-deep-cursor", 2),
        )
        payload_details = "\n".join(
            str(row["detail"]) for row in payload_plan
        )
        assert "idx_objects_payload_recovery" in payload_details
        assert "(created_at,oid)>" in payload_details.replace(" ", "")

        recovery_queries = {
            "idx_object_tasks_recovery_active_eligible": (
                "status IN ('queued', 'running', 'waiting_human', "
                "'waiting_process', 'waiting_message')"
            ),
            "idx_object_tasks_recovery_result_eligible": (
                "status = 'succeeded' AND result_oid IS NOT NULL"
            ),
            "idx_object_tasks_recovery_notification_eligible": (
                "status IN ('succeeded', 'failed', 'cancelled') "
                "AND notification_status IN ('none', 'failed') "
                "AND notification_recipient_pid IS NOT NULL"
            ),
        }
        for index_name, predicate in recovery_queries.items():
            task_plan = store.conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM object_tasks "
                f"INDEXED BY {index_name} WHERE {predicate} "
                "AND (created_at, task_id) > (?, ?) "
                "ORDER BY created_at, task_id LIMIT ?",
                ("2026-01-01T00:00:00Z", "task-deep-cursor", 2),
            )
            details = "\n".join(str(row["detail"]) for row in task_plan)
            assert index_name in details
            assert "USE TEMP B-TREE" not in details
            assert "(created_at,task_id)>" in details.replace(" ", "")

        index_sql = {
            str(row["name"]): "".join(str(row["sql"]).lower().split())
            for row in store.conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE name LIKE 'idx_object_tasks_recovery_%_eligible'"
            )
        }
        assert "wherestatusin('queued','running','waiting_human'" in index_sql[
            "idx_object_tasks_recovery_active_eligible"
        ]
        assert "wherestatus='succeeded'andresult_oidisnotnull" in index_sql[
            "idx_object_tasks_recovery_result_eligible"
        ]
        assert "notification_recipient_pidisnotnull" in index_sql[
            "idx_object_tasks_recovery_notification_eligible"
        ]
    finally:
        store.close()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ObjectPayloadRecoverySummary(-1, ()), "non-negative"),
        (lambda: ObjectPayloadRecoverySummary(0, ("oid",)), "exceeds total"),
        (lambda: ObjectTaskRecoverySummary(abandoned_total=True), "non-negative"),
        (
            lambda: ObjectTaskRecoverySummary(
                notification_retried_total=0,
                notification_retried_sample=("task",),
            ),
            "exceeds total",
        ),
        (lambda: ObjectTaskRecoveryCursor("", "task"), "created_at"),
    ],
)
def test_object_recovery_models_reject_invalid_state(
    factory: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_object_task_recovery_page_rejects_invalid_cursor_and_order() -> None:
    first = _task("task-b", created_at="2026-01-01T00:00:00Z")
    second = _task("task-a", created_at="2026-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="strictly ordered"):
        ObjectTaskRecoveryPage(records=(first, second))
    with pytest.raises(ValueError, match="last record"):
        ObjectTaskRecoveryPage(
            records=(second, first),
            next_cursor=ObjectTaskRecoveryCursor(first.created_at, second.task_id),
        )


def _seed_recovery_rows(
    store: Any,
    *,
    object_count: int,
    active_task_count: int,
    missing_result_count: int,
    notification_count: int,
) -> None:
    created_at = "2026-01-01T00:00:00Z"
    oids = [f"oid-recovery-{index:05d}" for index in range(object_count)]
    present_marker = dumps(store.payload_marker(present=True))
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO objects (
                oid, namespace, name, type, schema_version, payload_json,
                metadata_json, provenance_json, version, immutable, created_by,
                owner_kind, owner_id, lifecycle_state, deleted_at, created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    oid,
                    "system",
                    f"recovery-{index:05d}",
                    "artifact",
                    "1",
                    present_marker,
                    "{}",
                    "{}",
                    1,
                    0,
                    "recovery-fixture",
                    "process",
                    "missing-creator",
                    "live",
                    None,
                    created_at,
                    created_at,
                )
                for index, oid in enumerate(oids)
            ],
        )
        cursor.executemany(
            """
            INSERT INTO object_links (
                id, src_oid, relation, dst_oid, metadata_json, created_by,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"link-recovery-{index:05d}",
                    oid,
                    "references",
                    oids[(index + 1) % object_count],
                    "{}",
                    "recovery-fixture",
                    created_at,
                )
                for index, oid in enumerate(oids)
            ],
        )
        cursor.executemany(
            """
            INSERT INTO capabilities (
                cap_id, subject, resource, rights_json, constraints_json,
                issued_by, issued_at, expires_at, delegable, revocable, effect,
                issuer_cap_id, parent_cap_id, delegation_depth,
                max_delegation_depth, uses_remaining, status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"cap-recovery-{index:05d}",
                    "missing-creator",
                    f"object:{oid}",
                    '["read"]',
                    "{}",
                    "recovery-fixture",
                    created_at,
                    None,
                    0,
                    1,
                    "allow",
                    None,
                    None,
                    0,
                    None,
                    None,
                    "active",
                    "{}",
                )
                for index, oid in enumerate(oids)
            ],
        )
        tasks = [
            _task_row(
                f"task-active-{index:05d}",
                owner_oid=oids[index % object_count],
                status="queued",
                created_at=created_at,
            )
            for index in range(active_task_count)
        ]
        tasks.extend(
            _task_row(
                f"task-result-{index:05d}",
                owner_oid=oids[index],
                status="succeeded",
                result_oid=oids[index],
                created_at=created_at,
            )
            for index in range(missing_result_count)
        )
        tasks.extend(
            _task_row(
                f"task-notify-{index:05d}",
                owner_oid=oids[index],
                status="failed",
                recipient_pid="missing-recipient",
                notification_status="failed",
                created_at=created_at,
            )
            for index in range(notification_count)
        )
        cursor.executemany(
            """
            INSERT INTO object_tasks (
                task_id, owner_oid, creator_pid, runner_pid, tool, tool_id,
                status, notification_status, notification_recipient_pid,
                notification_json, owner_watch_json, result_oid, error,
                wait_json, created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tasks,
        )


def _task_row(
    task_id: str,
    *,
    owner_oid: str,
    status: str,
    created_at: str,
    result_oid: str | None = None,
    recipient_pid: str | None = None,
    notification_status: str = "none",
) -> tuple[Any, ...]:
    notification = {
        "recipient_pid": recipient_pid,
        "kind": "normal",
        "channel": "object-task",
        "subject": None,
        "message_id": None,
        "status": notification_status,
        "error": "transient" if notification_status == "failed" else None,
    }
    return (
        task_id,
        owner_oid,
        "missing-creator",
        None,
        "fixture.noop",
        None,
        status,
        notification_status,
        recipient_pid,
        json.dumps(notification, sort_keys=True),
        "{}",
        result_oid,
        None,
        "{}",
        created_at,
        created_at,
        None,
        created_at if status in {"succeeded", "failed"} else None,
    )


def _task(task_id: str, *, created_at: str) -> ObjectTask:
    return ObjectTask(
        task_id=task_id,
        owner_oid="owner",
        creator_pid="creator",
        runner_pid=None,
        tool="fixture",
        tool_id=None,
        status=ObjectTaskStatus.QUEUED,
        created_at=created_at,
        updated_at=created_at,
    )


def _config(*, page_size: int, backend: str = "sqlite", dsn: str | None = None) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        runtime=RuntimeDefaults(
            store_backend=backend,
            store_dsn=dsn,
            object_payload_recovery_page_size=page_size,
            object_payload_recovery_page_hard_limit=page_size,
            object_task_recovery_page_size=page_size,
            object_task_recovery_page_hard_limit=page_size,
        )
    )


@contextlib.contextmanager
def _runtime_target(
    backend: str,
    tmp_path: Path,
    *,
    page_size: int,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if backend == "sqlite":
        yield tmp_path / "object-recovery-paging.sqlite", _config(
            page_size=page_size
        )
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        yield dsn, _config(page_size=page_size, backend="postgres", dsn=dsn)


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_object_recovery_{uuid4().hex}"
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
