from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

import agent_libos.storage.mcp_v7_migration as mcp_v7_migration
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage import SQLiteStore
from agent_libos.storage.mcp_v7_migration import (
    StoreV7MigrationError,
    apply_store_v7_migration,
    plan_store_v7_migration,
)
from agent_libos.storage.v7_schema_contract import V7_TABLES


def _v6_store(path: Path) -> None:
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        for table in sorted(V7_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 6 "
            "WHERE singleton = 1 AND schema_version = 7"
        )
        assert changed.rowcount == 1
        connection.execute(
            "INSERT INTO runtime_counters (counter_name, value) VALUES (?, ?)",
            ("mcp-v7-migration-sentinel", 7),
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(path, 0o600)


def _backup(source: Path, backup: Path) -> None:
    shutil.copyfile(source, backup)
    os.chmod(backup, 0o600)


def _schema_marker(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def test_v6_to_v7_plan_is_zero_write_and_apply_reopens_canonical_v7(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v6_store(source)
    _backup(source, backup)
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    backup_before = hashlib.sha256(backup.read_bytes()).hexdigest()

    plan = plan_store_v7_migration(source, sqlite_backup=backup)
    repeated = plan_store_v7_migration(source, sqlite_backup=backup)

    assert plan == repeated
    assert plan.from_schema_version == 6
    assert plan.to_schema_version == 7
    assert plan.backend == "sqlite"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_before
    with pytest.raises(StoreV7MigrationError, match="plan digest"):
        apply_store_v7_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before

    result = apply_store_v7_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )

    assert result.applied
    assert not result.already_applied
    assert _schema_marker(source) == 7
    assert _schema_marker(backup) == 6
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_before
    reopened = SQLiteStore(source)
    try:
        present = {
            str(row[0])
            for row in reopened.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert V7_TABLES <= present
        assert reopened.conn.execute(
            "SELECT value FROM runtime_counters WHERE counter_name = ?",
            ("mcp-v7-migration-sentinel",),
        ).fetchone()[0] == 7
    finally:
        reopened.close()


def test_v7_plan_requires_an_independent_exact_v6_backup(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v6_store(source)
    with pytest.raises(StoreV7MigrationError, match="verified backup"):
        plan_store_v7_migration(source)
    with pytest.raises(StoreV7MigrationError, match="independent"):
        plan_store_v7_migration(source, sqlite_backup=source)

    _backup(source, backup)
    connection = sqlite3.connect(backup)
    try:
        connection.execute(
            "UPDATE runtime_counters SET value = 8 WHERE counter_name = ?",
            ("mcp-v7-migration-sentinel",),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(StoreV7MigrationError, match="does not match"):
        plan_store_v7_migration(source, sqlite_backup=backup)


def test_runtime_refuses_v6_without_automatic_v7_migration(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _v6_store(source)
    before = source.read_bytes()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="explicit offline v6-to-v7 migration",
    ):
        SQLiteStore(source)

    assert source.read_bytes() == before
    assert _schema_marker(source) == 6


def test_v6_to_v7_failure_rolls_back_ddl_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v6_store(source)
    _backup(source, backup)
    plan = plan_store_v7_migration(source, sqlite_backup=backup)
    before = source.read_bytes()
    real_execute = mcp_v7_migration._execute_v7_ddl

    def fail_after_ddl(connection: object) -> None:
        real_execute(connection)
        raise StoreV7MigrationError("injected schema-v7 migration failure")

    monkeypatch.setattr(mcp_v7_migration, "_execute_v7_ddl", fail_after_ddl)
    with pytest.raises(StoreV7MigrationError, match="injected schema-v7"):
        apply_store_v7_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )

    assert source.read_bytes() == before
    assert _schema_marker(source) == 6
    connection = sqlite3.connect(source)
    try:
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert present.isdisjoint(V7_TABLES)
    finally:
        connection.close()
