from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import agent_libos.storage.llm_v8_migration as llm_v8_migration
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage import SQLiteStore
from agent_libos.storage.llm_v8_migration import (
    StoreV8MigrationError,
    apply_store_v8_migration,
    plan_store_v8_migration,
)
from agent_libos.storage.v8_schema_contract import V8_TABLES


def _v7_store(path: Path) -> None:
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX IF EXISTS idx_llm_pending_replay_recovery")
        for table in sorted(V8_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 7 "
            "WHERE singleton = 1 AND schema_version = 8"
        )
        assert changed.rowcount == 1
        connection.execute(
            "INSERT INTO runtime_counters (counter_name, value) VALUES (?, ?)",
            ("llm-v8-migration-sentinel", 8),
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


def test_v7_to_v8_plan_is_zero_write_and_apply_reopens_canonical_v8(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    backup_before = hashlib.sha256(backup.read_bytes()).hexdigest()

    plan = plan_store_v8_migration(source, sqlite_backup=backup)
    repeated = plan_store_v8_migration(source, sqlite_backup=backup)

    assert plan == repeated
    assert plan.from_schema_version == 7
    assert plan.to_schema_version == 8
    assert plan.backend == "sqlite"
    assert plan.schema_version == 2
    assert plan.migration_implementation_version == "v7-to-v8/1"
    assert len(plan.receipt_contract_sha256) == 64
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_before
    with pytest.raises(StoreV8MigrationError, match="plan digest"):
        apply_store_v8_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before

    result = apply_store_v8_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )

    assert result.applied
    assert not result.already_applied
    repeated = apply_store_v8_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert repeated.applied is False
    assert repeated.already_applied is True
    assert repeated.plan == plan
    assert _schema_marker(source) == 8
    assert _schema_marker(backup) == 7
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_before
    reopened = SQLiteStore(source)
    try:
        present = {
            str(row[0])
            for row in reopened.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert V8_TABLES <= present
        assert reopened.conn.execute(
            "SELECT value FROM runtime_counters WHERE counter_name = ?",
            ("llm-v8-migration-sentinel",),
        ).fetchone()[0] == 8
    finally:
        reopened.close()


def test_v8_plan_is_bound_to_the_selected_sqlite_database(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.sqlite"
    first_backup = tmp_path / "first-backup.sqlite"
    second_source = tmp_path / "second.sqlite"
    second_backup = tmp_path / "second-backup.sqlite"
    _v7_store(first_source)
    _backup(first_source, first_backup)
    _backup(first_source, second_source)
    _backup(second_source, second_backup)

    first_plan = plan_store_v8_migration(
        first_source,
        sqlite_backup=first_backup,
    )
    second_plan = plan_store_v8_migration(
        second_source,
        sqlite_backup=second_backup,
    )

    assert first_plan.source_digest_sha256 == second_plan.source_digest_sha256
    assert first_plan.database_identity_sha256 != second_plan.database_identity_sha256
    assert first_plan.plan_sha256 != second_plan.plan_sha256
    with pytest.raises(StoreV8MigrationError, match="plan digest"):
        apply_store_v8_migration(
            second_source,
            sqlite_backup=second_backup,
            expected_plan_sha256=first_plan.plan_sha256,
        )


@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_v8_apply_reconciles_exact_target_after_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)
    plan = plan_store_v8_migration(source, sqlite_backup=backup)

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
        real_require = llm_v8_migration._require_canonical_v8
        calls = 0

        def fail_post_commit_readback(backend: object, connection: object) -> None:
            nonlocal calls
            real_require(backend, connection)
            calls += 1
            if calls == 2:
                raise RuntimeError("injected post-commit readback failure")

        monkeypatch.setattr(
            llm_v8_migration,
            "_require_canonical_v8",
            fail_post_commit_readback,
        )
        expected_error = "post-commit readback"

    with pytest.raises(RuntimeError, match=expected_error):
        apply_store_v8_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )
    assert _schema_marker(source) == 8

    with pytest.raises(StoreV8MigrationError, match="plan digest"):
        apply_store_v8_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )

    result = apply_store_v8_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert result.applied is False
    assert result.already_applied is True
    assert result.plan == plan


def test_v8_plan_requires_an_independent_exact_v7_backup(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    with pytest.raises(StoreV8MigrationError, match="verified backup"):
        plan_store_v8_migration(source)
    with pytest.raises(StoreV8MigrationError, match="independent"):
        plan_store_v8_migration(source, sqlite_backup=source)

    _backup(source, backup)
    connection = sqlite3.connect(backup)
    try:
        connection.execute(
            "UPDATE runtime_counters SET value = 9 WHERE counter_name = ?",
            ("llm-v8-migration-sentinel",),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(StoreV8MigrationError, match="does not match"):
        plan_store_v8_migration(source, sqlite_backup=backup)


def test_runtime_refuses_v7_without_automatic_v8_migration(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _v7_store(source)
    before = source.read_bytes()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="explicit offline v7-to-v8 migration",
    ):
        SQLiteStore(source)

    assert source.read_bytes() == before
    assert _schema_marker(source) == 7


def test_v7_to_v8_failure_rolls_back_ddl_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)
    plan = plan_store_v8_migration(source, sqlite_backup=backup)
    before = source.read_bytes()
    real_execute = llm_v8_migration._execute_v8_ddl

    def fail_after_ddl(connection: object) -> None:
        real_execute(connection)
        raise StoreV8MigrationError("injected schema-v8 migration failure")

    monkeypatch.setattr(llm_v8_migration, "_execute_v8_ddl", fail_after_ddl)
    with pytest.raises(StoreV8MigrationError, match="injected schema-v8"):
        apply_store_v8_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )

    assert source.read_bytes() == before
    assert _schema_marker(source) == 7
    connection = sqlite3.connect(source)
    try:
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert present.isdisjoint(V8_TABLES)
    finally:
        connection.close()


@pytest.mark.parametrize("mutation", ("insert", "update", "delete"))
def test_v8_apply_rejects_source_drift_after_reviewed_plan(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)
    plan = plan_store_v8_migration(source, sqlite_backup=backup)
    with sqlite3.connect(source) as connection:
        if mutation == "insert":
            connection.execute(
                "INSERT INTO runtime_counters VALUES (?, ?)", ("new-counter", 1)
            )
        elif mutation == "update":
            connection.execute(
                "UPDATE runtime_counters SET value = value + 1 "
                "WHERE counter_name = 'llm-v8-migration-sentinel'"
            )
        else:
            connection.execute(
                "DELETE FROM runtime_counters "
                "WHERE counter_name = 'llm-v8-migration-sentinel'"
            )
    before = source.read_bytes()
    with pytest.raises(StoreV8MigrationError, match="backup does not match"):
        apply_store_v8_migration(
            source, expected_plan_sha256=plan.plan_sha256, sqlite_backup=backup
        )
    assert source.read_bytes() == before
    assert _schema_marker(source) == 7


def test_v8_ddl_cannot_corrupt_existing_rows_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)
    plan = plan_store_v8_migration(source, sqlite_backup=backup)
    before = source.read_bytes()
    real_execute = llm_v8_migration._execute_v8_ddl

    def corrupt_source(connection: sqlite3.Connection) -> None:
        real_execute(connection)
        connection.execute(
            "UPDATE runtime_counters SET value = 99 "
            "WHERE counter_name = 'llm-v8-migration-sentinel'"
        )

    monkeypatch.setattr(llm_v8_migration, "_execute_v8_ddl", corrupt_source)
    with pytest.raises(StoreV8MigrationError, match="changed the locked source state"):
        apply_store_v8_migration(
            source, expected_plan_sha256=plan.plan_sha256, sqlite_backup=backup
        )
    assert source.read_bytes() == before
    assert _schema_marker(source) == 7


def test_v8_plan_rejects_noncanonical_source_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE hidden_replay_state (ciphertext TEXT)")
    _backup(source, backup)
    before = source.read_bytes()
    with pytest.raises(UnsupportedStoreVersion):
        plan_store_v8_migration(source, sqlite_backup=backup)
    assert source.read_bytes() == before
    assert _schema_marker(source) == 7


def test_v8_migration_cli_runs_before_runtime_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    import agent_libos.api.cli as cli

    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v7_store(source)
    _backup(source, backup)

    def forbidden_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline migration must not open a Runtime")

    monkeypatch.setattr(cli.Runtime, "open", forbidden_open)
    args = ["--db", str(source), "store", "migrate", "--to", "8"]
    assert cli.main([*args, "--dry-run", "--sqlite-backup", str(backup)]) is None
    plan = json.loads(capsys.readouterr().out)
    assert plan["from_schema_version"] == 7
    assert plan["to_schema_version"] == 8
    assert cli.main([
        *args, "--apply", "--sqlite-backup", str(backup),
        "--expected-plan-sha256", plan["plan_sha256"],
    ]) is None
    assert json.loads(capsys.readouterr().out)["applied"] is True
    assert _schema_marker(source) == 8
