from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from contextlib import contextmanager
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
    assert plan.schema_version == 2
    assert plan.migration_implementation_version == "v6-to-v7/3"
    assert len(plan.receipt_contract_sha256) == 64
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
    repeated = apply_store_v7_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert repeated.applied is False
    assert repeated.already_applied is True
    assert repeated.plan == plan
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


def test_v7_plan_is_bound_to_the_selected_sqlite_database(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.sqlite"
    first_backup = tmp_path / "first-backup.sqlite"
    second_source = tmp_path / "second.sqlite"
    second_backup = tmp_path / "second-backup.sqlite"
    _v6_store(first_source)
    _backup(first_source, first_backup)
    _backup(first_source, second_source)
    _backup(second_source, second_backup)

    first_plan = plan_store_v7_migration(
        first_source,
        sqlite_backup=first_backup,
    )
    second_plan = plan_store_v7_migration(
        second_source,
        sqlite_backup=second_backup,
    )

    assert first_plan.source_digest_sha256 == second_plan.source_digest_sha256
    assert first_plan.database_identity_sha256 != second_plan.database_identity_sha256
    assert first_plan.plan_sha256 != second_plan.plan_sha256
    with pytest.raises(StoreV7MigrationError, match="plan digest"):
        apply_store_v7_migration(
            second_source,
            sqlite_backup=second_backup,
            expected_plan_sha256=first_plan.plan_sha256,
        )


@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_v7_apply_reconciles_exact_target_after_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v6_store(source)
    _backup(source, backup)
    plan = plan_store_v7_migration(source, sqlite_backup=backup)

    if fault == "commit_ack":
        real_open = SQLiteStore._migration_apply_connection

        @contextmanager
        def lost_commit_ack(
            cls: type[SQLiteStore],
            path: Path,
            *,
            error_type: type[Exception],
            migration_label: str,
        ):
            del cls
            with real_open(
                path,
                error_type=error_type,
                migration_label=migration_label,
            ) as connection:
                class ConnectionProxy:
                    def __getattr__(self, name: str) -> object:
                        return getattr(connection, name)

                    def commit(self) -> None:
                        connection.commit()
                        raise RuntimeError("injected lost commit ACK")

                yield ConnectionProxy()

        monkeypatch.setattr(
            SQLiteStore,
            "_migration_apply_connection",
            classmethod(lost_commit_ack),
        )
        expected_error = "lost commit ACK"
    else:
        real_require = mcp_v7_migration._require_canonical_v7
        calls = 0

        def fail_post_commit_readback(backend: object, connection: object) -> None:
            nonlocal calls
            real_require(backend, connection)
            calls += 1
            if calls == 2:
                raise RuntimeError("injected post-commit readback failure")

        monkeypatch.setattr(
            mcp_v7_migration,
            "_require_canonical_v7",
            fail_post_commit_readback,
        )
        expected_error = "post-commit readback"

    with pytest.raises(RuntimeError, match=expected_error):
        apply_store_v7_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )
    assert _schema_marker(source) == 7

    with pytest.raises(StoreV7MigrationError, match="plan digest"):
        apply_store_v7_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )

    result = apply_store_v7_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert result.applied is False
    assert result.already_applied is True
    assert result.plan == plan


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
