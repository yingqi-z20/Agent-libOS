from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage import SQLiteStore
import agent_libos.storage.semantic_v5_migration as migration_module
from agent_libos.storage.semantic_v5_migration import (
    StoreV5MigrationError,
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
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table in sorted((V7_TABLES | V6_TABLES) & tables):
            connection.execute(f'DROP TABLE "{table}"')
        if "semantic_assessments" in tables:
            connection.execute("DROP TABLE semantic_assessments")
        if "semantic_assessment_jobs" in tables:
            connection.execute("DROP TABLE semantic_assessment_jobs")
        human_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(human_requests)")
        }
        if "revision" in human_columns:
            connection.execute(
                "ALTER TABLE human_requests DROP COLUMN revision"
            )
        connection.execute(
            "UPDATE runtime_schema SET schema_version = 4 WHERE singleton = 1"
        )
        connection.commit()
    finally:
        connection.close()


def _backup(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    os.chmod(destination, 0o600)


def _file_sha256(path: Path) -> str:
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


def _user_tables(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("sqlite:///C:/agent/runtime.sqlite", "C:/agent/runtime.sqlite"),
        ("sqlite:////C:/agent/runtime.sqlite", "C:/agent/runtime.sqlite"),
        (r"C:\agent\runtime.sqlite", r"C:\agent\runtime.sqlite"),
        ("C:/agent/runtime.sqlite", "C:/agent/runtime.sqlite"),
    ],
)
def test_sqlite_target_text_preserves_windows_drive_paths(
    target: str,
    expected: str,
) -> None:
    assert migration_module._sqlite_target_text(target, windows=True) == expected


def test_v5_plan_is_deterministic_and_dry_run_is_zero_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)
    before_source = _file_sha256(source)
    before_backup = _file_sha256(backup)
    before_names = {path.name for path in tmp_path.iterdir()}

    first = plan_store_v5_migration(source, sqlite_backup=backup)
    second = plan_store_v5_migration(source, sqlite_backup=backup)

    assert first == second
    assert first.backend == "sqlite"
    assert first.from_schema_version == 4
    assert first.to_schema_version == 5
    assert len(first.plan_sha256) == 64
    assert len(first.ddl_sha256) == 64
    assert first.to_dict()["schema_version"] == 1
    assert _file_sha256(source) == before_source
    assert _file_sha256(backup) == before_backup
    assert {path.name for path in tmp_path.iterdir()} == before_names
    assert _schema_version(source) == 4

    uri_plan = plan_store_v5_migration(f"sqlite:///{source.as_posix()}")
    assert uri_plan == first


def test_v5_apply_requires_expected_plan_and_verified_backup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            "INSERT INTO human_requests "
            "(request_id, pid, human, payload_json, status, decision_json, "
            "blocking, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-request",
                "legacy-pid",
                "operator",
                "{}",
                "pending",
                None,
                1,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    _backup(source, backup)
    plan = plan_store_v5_migration(source, sqlite_backup=backup)

    result = apply_store_v5_migration(
        source,
        sqlite_backup=backup,
        expected_plan_sha256=plan.plan_sha256,
    )

    assert result.applied is True
    assert result.already_applied is False
    assert result.plan == plan
    assert _schema_version(source) == 5
    assert {
        "semantic_assessment_jobs",
        "semantic_assessments",
    }.issubset(_user_tables(source))
    connection = sqlite3.connect(source)
    try:
        human_columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(human_requests)")
        }
        assert human_columns["revision"][2].upper() == "BIGINT"
        assert human_columns["revision"][3] == 1
        assert str(human_columns["revision"][4]) in {"0", "(0)"}
        assert connection.execute(
            "SELECT revision FROM human_requests WHERE request_id = ?",
            ("legacy-request",),
        ).fetchone() == (0,)
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert set(migration_module._SEMANTIC_INDEXES).issubset(indexes)
    finally:
        connection.close()


def test_v5_apply_rejects_plan_digest_mismatch_without_schema_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)

    with pytest.raises(StoreV5MigrationError, match="plan digest"):
        apply_store_v5_migration(
            source,
            sqlite_backup=backup,
            expected_plan_sha256="0" * 64,
        )

    assert _schema_version(source) == 4
    assert "revision" not in _human_columns(source)
    assert "semantic_assessments" not in _user_tables(source)


def test_v5_apply_rejects_stale_backup_under_locked_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            "INSERT INTO runtime_counters(counter_name, value) VALUES (?, ?)",
            ("after-backup", 1),
        )
        connection.commit()
    finally:
        connection.close()
    plan = plan_store_v5_migration(source)

    with pytest.raises(StoreV5MigrationError, match="backup does not match"):
        apply_store_v5_migration(
            source,
            sqlite_backup=backup,
            expected_plan_sha256=plan.plan_sha256,
        )

    assert _schema_version(source) == 4
    assert "revision" not in _human_columns(source)
    assert "semantic_assessments" not in _user_tables(source)


def test_v5_apply_rolls_back_ddl_and_marker_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)
    original = migration_module._MIGRATION_STATEMENTS
    monkeypatch.setattr(
        migration_module,
        "_MIGRATION_STATEMENTS",
        (
            original[0],
            "CREATE TABLE semantic_migration_interruption (value TEXT)",
            "THIS IS NOT SQL",
        ),
    )
    plan = plan_store_v5_migration(source, sqlite_backup=backup)

    with pytest.raises(StoreV5MigrationError, match="migration failed"):
        apply_store_v5_migration(
            source,
            sqlite_backup=backup,
            expected_plan_sha256=plan.plan_sha256,
        )

    assert _schema_version(source) == 4
    assert "revision" not in _human_columns(source)
    assert "semantic_migration_interruption" not in _user_tables(source)


def test_v5_plan_rejects_noncanonical_v4_without_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    _make_v4_store(source)
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE unexpected_extension (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    before = _file_sha256(source)

    with pytest.raises(UnsupportedStoreVersion, match="schema v4"):
        plan_store_v5_migration(source)

    assert _file_sha256(source) == before
    assert _schema_version(source) == 4


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable")
def test_v5_apply_requires_owner_only_self_contained_backup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)
    os.chmod(backup, 0o644)
    plan = plan_store_v5_migration(source)

    with pytest.raises(StoreV5MigrationError, match="mode 0600"):
        apply_store_v5_migration(
            source,
            sqlite_backup=backup,
            expected_plan_sha256=plan.plan_sha256,
        )

    assert _schema_version(source) == 4


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable")
def test_v5_apply_rejects_world_readable_source_but_dry_run_is_zero_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4-backup.sqlite"
    _make_v4_store(source)
    _backup(source, backup)
    os.chmod(source, 0o644)
    before_digest = _file_sha256(source)
    before_mode = source.stat().st_mode & 0o777

    plan = plan_store_v5_migration(source, sqlite_backup=backup)

    assert _file_sha256(source) == before_digest
    assert source.stat().st_mode & 0o777 == before_mode == 0o644
    with pytest.raises(StoreV5MigrationError, match="source must have mode 0600"):
        apply_store_v5_migration(
            source,
            sqlite_backup=backup,
            expected_plan_sha256=plan.plan_sha256,
        )
    assert _schema_version(source) == 4
    assert source.stat().st_mode & 0o777 == 0o644


def _human_columns(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(human_requests)")
        }
    finally:
        connection.close()
