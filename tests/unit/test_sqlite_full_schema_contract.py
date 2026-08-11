from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage import SQLiteStore
from agent_libos.storage.semantic_v5_migration import (
    apply_store_v5_migration,
    plan_store_v5_migration,
)
from agent_libos.storage.v6_schema_contract import V6_TABLES
from agent_libos.storage.v7_schema_contract import V7_TABLES


def _make_v4_store(path: Path) -> None:
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        for table in sorted(V7_TABLES | V6_TABLES):
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


def _rewrite_table_sql(path: Path, table: str, old: str, new: str) -> None:
    def rewrite(original: str) -> str:
        assert old in original
        return original.replace(old, new, 1)

    _transform_table_sql(path, table, rewrite)


def _transform_table_sql(
    path: Path,
    table: str,
    transform: Callable[[str], str],
) -> None:
    connection = sqlite3.connect(path)
    try:
        sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        assert sql_row is not None and sql_row[0] is not None
        original = str(sql_row[0])
        rewritten = transform(original)
        assert rewritten != original
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = ? "
            "WHERE type = 'table' AND name = ?",
            (rewritten, table),
        )
        connection.execute("PRAGMA writable_schema = OFF")
        connection.commit()
    finally:
        connection.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("table", "old", "new"),
    [
        (
            "capabilities",
            "constraints_json TEXT NOT NULL",
            "constraints_json BLOB NOT NULL",
        ),
        (
            "capabilities",
            "constraints_json TEXT NOT NULL",
            "constraints_json TEXT",
        ),
        (
            "capabilities",
            "metadata_json TEXT NOT NULL",
            "metadata_json TEXT NOT NULL CHECK (length(metadata_json) > 0)",
        ),
        (
            "capabilities",
            "subject TEXT NOT NULL",
            "subject TEXT COLLATE NOCASE NOT NULL",
        ),
        (
            "capabilities",
            "metadata_json TEXT NOT NULL",
            "metadata_json TEXT GENERATED ALWAYS AS ('{}') STORED",
        ),
        (
            "objects",
            "payload_json TEXT NOT NULL",
            "payload_json BLOB NOT NULL",
        ),
        (
            "task_run_requirements",
            "FOREIGN KEY(run_id) REFERENCES task_runs(run_id)",
            "FOREIGN KEY(run_id) REFERENCES task_runs(run_id) ON DELETE CASCADE",
        ),
    ],
)
def test_v5_reopen_rejects_full_table_contract_drift(
    tmp_path: Path,
    table: str,
    old: str,
    new: str,
) -> None:
    path = tmp_path / "runtime.sqlite"
    store = SQLiteStore(path)
    store.close()
    _rewrite_table_sql(path, table, old, new)

    with pytest.raises(UnsupportedStoreVersion):
        SQLiteStore(path)


@pytest.mark.parametrize("table_option", ["STRICT", "WITHOUT ROWID"])
def test_v5_reopen_rejects_noncanonical_table_options(
    tmp_path: Path,
    table_option: str,
) -> None:
    path = tmp_path / f"option-{table_option.casefold().replace(' ', '-')}.sqlite"
    store = SQLiteStore(path)
    store.close()
    _transform_table_sql(
        path,
        "capabilities",
        lambda original: f"{original.rstrip()} {table_option}",
    )

    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        SQLiteStore(path)


@pytest.mark.parametrize(
    "index_sql",
    [
        "CREATE INDEX idx_capabilities_subject_status "
        "ON capabilities(subject DESC, status)",
        "CREATE INDEX idx_capabilities_subject_status "
        "ON capabilities(subject, status) WHERE status = 'active'",
    ],
)
def test_v5_reopen_rejects_descending_or_partial_index_drift(
    tmp_path: Path,
    index_sql: str,
) -> None:
    path = tmp_path / "runtime.sqlite"
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX idx_capabilities_subject_status")
        connection.execute(index_sql)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStoreVersion):
        SQLiteStore(path)


def test_v5_reopen_rejects_index_trigger_and_conflict_policy_drift(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.sqlite"
    store = SQLiteStore(index_path)
    store.close()
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            "CREATE INDEX idx_capabilities_unexpected "
            "ON capabilities(subject, status, cap_id)"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        SQLiteStore(index_path)

    trigger_path = tmp_path / "trigger.sqlite"
    store = SQLiteStore(trigger_path)
    store.close()
    connection = sqlite3.connect(trigger_path)
    try:
        connection.execute(
            "CREATE TRIGGER capabilities_noop AFTER INSERT ON capabilities "
            "BEGIN SELECT 1; END"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        SQLiteStore(trigger_path)

    conflict_path = tmp_path / "conflict.sqlite"
    store = SQLiteStore(conflict_path)
    store.close()
    _rewrite_table_sql(
        conflict_path,
        "human_requests",
        "request_id TEXT COLLATE BINARY PRIMARY KEY",
        "request_id TEXT COLLATE BINARY PRIMARY KEY ON CONFLICT REPLACE",
    )
    with pytest.raises(UnsupportedStoreVersion):
        SQLiteStore(conflict_path)


def test_v5_reopen_rejects_unrelated_tables_and_views(tmp_path: Path) -> None:
    table_path = tmp_path / "table.sqlite"
    store = SQLiteStore(table_path)
    store.close()
    connection = sqlite3.connect(table_path)
    try:
        connection.execute("CREATE TABLE unexpected_extension (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnsupportedStoreVersion):
        SQLiteStore(table_path)

    view_path = tmp_path / "view.sqlite"
    store = SQLiteStore(view_path)
    store.close()
    connection = sqlite3.connect(view_path)
    try:
        connection.execute(
            "CREATE VIEW unexpected_processes AS SELECT pid FROM processes"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        SQLiteStore(view_path)


def test_v4_plan_rejects_type_drift_without_writes(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite"
    _make_v4_store(path)
    _rewrite_table_sql(
        path,
        "capabilities",
        "constraints_json TEXT NOT NULL",
        "constraints_json BLOB NOT NULL",
    )
    before = _sha256(path)

    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        plan_store_v5_migration(path)

    assert _sha256(path) == before
    assert _schema_version(path) == 4


def test_v4_apply_revalidates_full_catalog_and_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-backup.sqlite"
    _make_v4_store(path)
    plan = plan_store_v5_migration(path)
    _rewrite_table_sql(
        path,
        "capabilities",
        "constraints_json TEXT NOT NULL",
        "constraints_json BLOB NOT NULL",
    )
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    before = _sha256(path)

    with pytest.raises(UnsupportedStoreVersion, match="full catalog"):
        apply_store_v5_migration(
            path,
            sqlite_backup=backup,
            expected_plan_sha256=plan.plan_sha256,
        )

    assert _sha256(path) == before
    assert _schema_version(path) == 4
    connection = sqlite3.connect(path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        human_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(human_requests)")
        }
    finally:
        connection.close()
    assert "semantic_assessments" not in tables
    assert "revision" not in human_columns


def test_concurrent_first_open_never_bypasses_catalog_validation(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    paths: list[Path] = []
    for index in range(4):
        path = tmp_path / f"corrupt-{index}.sqlite"
        store = SQLiteStore(path)
        store.close()
        _rewrite_table_sql(
            path,
            "objects",
            "payload_json TEXT NOT NULL",
            "payload_json BLOB NOT NULL",
        )
        paths.append(path)

    def rejected(path: Path) -> bool:
        try:
            SQLiteStore(path)
        except UnsupportedStoreVersion:
            return True
        return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(rejected, paths)) == [True] * len(paths)
