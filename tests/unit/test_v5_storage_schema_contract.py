from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage.semantic_v5_migration import plan_store_v5_migration
from agent_libos.storage.sqlite import SQLiteStore
from agent_libos.storage.v6_schema_contract import V6_TABLES


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def _rewrite_table_sql(
    path: Path,
    table: str,
    mutation: Callable[[str], str],
) -> None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        assert row is not None and row[0]
        changed = mutation(str(row[0]))
        assert changed != row[0]
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = ?",
            (changed, table),
        )
        schema_version = int(
            connection.execute("PRAGMA schema_version").fetchone()[0]
        )
        connection.execute(f"PRAGMA schema_version = {schema_version + 1}")
        connection.execute("PRAGMA writable_schema = OFF")
        connection.commit()
    finally:
        connection.close()


def _make_v4_store(path: Path) -> None:
    SQLiteStore(path).close()
    connection = sqlite3.connect(path)
    try:
        for table in sorted(V6_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DROP TABLE semantic_assessments")
        connection.execute("DROP TABLE semantic_assessment_jobs")
        connection.execute("ALTER TABLE human_requests DROP COLUMN revision")
        connection.execute(
            "UPDATE runtime_schema SET schema_version = 4 WHERE singleton = 1"
        )
        connection.commit()
    finally:
        connection.close()


def _rebuild_human_requests_without_primary_key(
    path: Path,
    *,
    include_revision: bool,
) -> None:
    columns = (
        "request_id, pid, human, payload_json, status, decision_json, "
        "blocking, created_at, updated_at"
    )
    revision_definition = ""
    if include_revision:
        columns += ", revision"
        revision_definition = (
            ", revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)"
        )
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "ALTER TABLE human_requests RENAME TO human_requests_old;"
            "CREATE TABLE human_requests ("
            "request_id TEXT COLLATE BINARY,"
            "pid TEXT NOT NULL,"
            "human TEXT NOT NULL,"
            "payload_json TEXT NOT NULL,"
            "status TEXT NOT NULL,"
            "decision_json TEXT,"
            "blocking INTEGER NOT NULL,"
            "created_at TEXT COLLATE BINARY NOT NULL,"
            f"updated_at TEXT NOT NULL{revision_definition}"
            ");"
            f"INSERT INTO human_requests ({columns}) "
            f"SELECT {columns} FROM human_requests_old;"
            "DROP TABLE human_requests_old;"
            "CREATE INDEX idx_human_requests_pid_created "
            "ON human_requests(pid, created_at, request_id);"
            "CREATE INDEX idx_human_requests_human_status_created "
            "ON human_requests(human, status, created_at, request_id);"
            "CREATE INDEX idx_human_requests_status_created "
            "ON human_requests(status, created_at, request_id);"
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("case", "table", "old", "new", "error"),
    [
        (
            "human-request-status-nullable",
            "human_requests",
            "status TEXT NOT NULL",
            "status TEXT",
            "storage column contract",
        ),
        (
            "human-revision-type",
            "human_requests",
            "revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "revision TEXT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "storage column contract",
        ),
        (
            "human-revision-nullable",
            "human_requests",
            "revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "revision BIGINT DEFAULT 0 CHECK (revision >= 0)",
            "storage column contract",
        ),
        (
            "human-revision-default",
            "human_requests",
            "revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "revision BIGINT NOT NULL DEFAULT 1 CHECK (revision >= 0)",
            "storage column contract",
        ),
        (
            "human-revision-check",
            "human_requests",
            "revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "revision BIGINT NOT NULL DEFAULT 0",
            "storage CHECK contract",
        ),
        (
            "semantic-job-type",
            "semantic_assessment_jobs",
            "status TEXT NOT NULL CHECK",
            "status BLOB NOT NULL CHECK",
            "storage column contract",
        ),
        (
            "semantic-job-nullable",
            "semantic_assessment_jobs",
            "bindings_json TEXT NOT NULL",
            "bindings_json TEXT",
            "storage column contract",
        ),
        (
            "semantic-job-default",
            "semantic_assessment_jobs",
            "attempt_count INTEGER NOT NULL DEFAULT 0 CHECK",
            "attempt_count INTEGER NOT NULL CHECK",
            "storage column contract",
        ),
        (
            "semantic-job-check",
            "semantic_assessment_jobs",
            "revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)",
            "revision BIGINT NOT NULL DEFAULT 0",
            "storage CHECK contract",
        ),
        (
            "semantic-assessment-nullable",
            "semantic_assessments",
            "record_json TEXT NOT NULL",
            "record_json TEXT",
            "storage column contract",
        ),
        (
            "semantic-assessment-extra-check",
            "semantic_assessments",
            "record_json TEXT NOT NULL",
            "record_json TEXT NOT NULL CHECK (length(record_json) >= 0)",
            "storage CHECK contract",
        ),
        (
            "semantic-assessment-action-nullable",
            "semantic_assessments",
            "action_id TEXT NOT NULL",
            "action_id TEXT",
            "storage column contract",
        ),
        (
            "semantic-assessment-tenant-type",
            "semantic_assessments",
            "tenant_bucket_sha256 TEXT",
            "tenant_bucket_sha256 BLOB",
            "storage column contract",
        ),
        (
            "semantic-assessment-hidden-column",
            "semantic_assessments",
            "tenant_bucket_sha256 TEXT,",
            "tenant_bucket_sha256 TEXT, "
            "generated_probe TEXT GENERATED ALWAYS AS (action_id) VIRTUAL,",
            "storage column contract",
        ),
        (
            "semantic-assessment-foreign-key",
            "semantic_assessments",
            "pid TEXT,",
            "pid TEXT REFERENCES processes(pid) ON DELETE CASCADE,",
            "semantic relation boundary",
        ),
        (
            "human-request-foreign-key",
            "human_requests",
            "pid TEXT NOT NULL,",
            "pid TEXT NOT NULL REFERENCES processes(pid) ON DELETE CASCADE,",
            "semantic relation boundary",
        ),
        (
            "human-request-conflict-replace",
            "human_requests",
            "request_id TEXT COLLATE BINARY PRIMARY KEY,",
            "request_id TEXT COLLATE BINARY PRIMARY KEY ON CONFLICT REPLACE,",
            "semantic relation boundary",
        ),
    ],
)
def test_tampered_v5_sqlite_column_or_check_is_rejected_on_reopen(
    case: str,
    table: str,
    old: str,
    new: str,
    error: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{case}.sqlite"
    SQLiteStore(path).close()
    _rewrite_table_sql(
        path,
        table,
        lambda ddl: _replace_once(ddl, old, new),
    )

    with pytest.raises(UnsupportedStoreVersion, match=error):
        SQLiteStore(path)


def test_tampered_v5_sqlite_human_primary_key_is_rejected_on_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human-primary-key.sqlite"
    SQLiteStore(path).close()
    _rebuild_human_requests_without_primary_key(path, include_revision=True)

    with pytest.raises(
        UnsupportedStoreVersion,
        match="storage column contract",
    ):
        SQLiteStore(path)


def test_tampered_v5_sqlite_action_tenant_index_is_rejected_on_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "action-tenant-index.sqlite"
    SQLiteStore(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "DROP INDEX idx_semantic_assessments_action_tenant_created"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStoreVersion, match="semantic index manifest"):
        SQLiteStore(path)


@pytest.mark.parametrize(
    "index_name",
    [
        "idx_human_requests_pid_created",
        "idx_human_requests_human_status_created",
        "idx_human_requests_status_created",
    ],
)
def test_tampered_v5_sqlite_human_index_is_rejected_on_reopen(
    index_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{index_name}.sqlite"
    SQLiteStore(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"DROP INDEX {index_name}")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="human request index contract",
    ):
        SQLiteStore(path)


@pytest.mark.parametrize(
    ("statement", "params"),
    [
        (
            "INSERT INTO semantic_assessment_jobs ("
            "job_id, kind, status, domain, bindings_json, projection_json, "
            "projection_sha256, projection_retention, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                None,
                "approval",
                "queued",
                "filesystem",
                "{}",
                '{"action_id":"filesystem.read"}',
                "1" * 64,
                "redacted",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        ),
        (
            "INSERT INTO semantic_assessments ("
            "assessment_id, job_id, kind, status, domain, action_id, "
            "record_json, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                None,
                "job-null-id",
                "approval",
                "success",
                "filesystem",
                "filesystem.read",
                "{}",
                "2026-01-01T00:00:00Z",
            ),
        ),
    ],
)
def test_sqlite_semantic_primary_keys_reject_null_direct_writes(
    statement: str,
    params: tuple[object, ...],
) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            store.conn.execute(statement, params)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("table", "trigger_name"),
    [
        ("human_requests", "tamper_human_request"),
        ("semantic_assessment_jobs", "tamper_semantic_job"),
        ("semantic_assessments", "tamper_semantic_assessment"),
    ],
)
def test_tampered_v5_sqlite_user_trigger_is_rejected_on_reopen(
    table: str,
    trigger_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{trigger_name}.sqlite"
    SQLiteStore(path).close()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"CREATE TRIGGER {trigger_name} AFTER INSERT ON {table} "
            "BEGIN SELECT 1; END"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="semantic relation boundary",
    ):
        SQLiteStore(path)


@pytest.mark.parametrize(
    ("case", "old", "new"),
    [
        (
            "nullable-status",
            "status TEXT NOT NULL",
            "status TEXT",
        ),
        (
            "unexpected-foreign-key",
            "pid TEXT NOT NULL,",
            "pid TEXT NOT NULL REFERENCES processes(pid) ON DELETE CASCADE,",
        ),
    ],
)
def test_v4_migration_plan_rejects_noncanonical_human_table(
    case: str,
    old: str,
    new: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"v4-human-{case}.sqlite"
    _make_v4_store(path)
    _rewrite_table_sql(
        path,
        "human_requests",
        lambda ddl: _replace_once(ddl, old, new),
    )

    with pytest.raises(
        UnsupportedStoreVersion,
        match="schema v4 human request contract",
    ):
        plan_store_v5_migration(path)


def test_v4_migration_plan_rejects_missing_human_primary_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4-human-primary-key.sqlite"
    _make_v4_store(path)
    _rebuild_human_requests_without_primary_key(path, include_revision=False)

    with pytest.raises(
        UnsupportedStoreVersion,
        match="schema v4 human request contract",
    ):
        plan_store_v5_migration(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "DROP INDEX idx_human_requests_status_created",
        "CREATE TRIGGER tamper_v4_human AFTER INSERT ON human_requests "
        "BEGIN SELECT 1; END",
    ],
)
def test_v4_migration_plan_rejects_noncanonical_human_mutation_hook(
    mutation: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4-human-hook.sqlite"
    _make_v4_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(mutation)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="schema v4 human request contract",
    ):
        plan_store_v5_migration(path)
