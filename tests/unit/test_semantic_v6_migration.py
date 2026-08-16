from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

import agent_libos.storage.semantic_v6_migration as semantic_v6_migration
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage import SQLiteStore
from agent_libos.storage.semantic_v6_migration import (
    StoreV6MigrationError,
    apply_store_v6_migration,
    plan_store_v6_migration,
)
from agent_libos.storage.mcp_v7_migration import (
    apply_store_v7_migration,
    plan_store_v7_migration,
)
from agent_libos.storage.v6_schema_contract import V6_TABLES
from agent_libos.storage.v7_schema_contract import V7_TABLES


def _v5_store(path: Path) -> None:
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        for table in sorted(V7_TABLES | V6_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 5 "
            "WHERE singleton = 1 AND schema_version = 7"
        )
        assert changed.rowcount == 1
        connection.commit()
    finally:
        connection.close()
    os.chmod(path, 0o600)


def _insert_v5_assessments(path: Path, count: int) -> None:
    connection = sqlite3.connect(path)
    try:
        for index in range(count):
            connection.execute(
                "INSERT INTO semantic_assessments "
                "(assessment_id, job_id, kind, status, domain, action_id, "
                "tenant_bucket_sha256, pid, request_id, operation_id, effect_id, "
                "shadow_outcome, ood, record_json, created_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"legacy-assessment-{index}",
                    f"legacy-job-{index}",
                    "approval",
                    "succeeded",
                    "filesystem",
                    "filesystem.read",
                    "a" * 64,
                    f"legacy-pid-{index}",
                    f"legacy-request-{index}",
                    None,
                    None,
                    "require_human",
                    0,
                    "{}",
                    "2026-08-07T00:00:00+00:00",
                    "2026-08-07T00:00:00+00:00",
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_v5_to_v6_dry_run_is_zero_write_and_apply_requires_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v5_store(source)
    _insert_v5_assessments(source, 2)
    shutil.copyfile(source, backup)
    os.chmod(backup, 0o600)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    plan = plan_store_v6_migration(source, sqlite_backup=backup)
    repeated_plan = plan_store_v6_migration(source, sqlite_backup=backup)

    assert plan.from_schema_version == 5
    assert plan.to_schema_version == 6
    assert plan.schema_version == 2
    assert plan.migration_implementation_version == "v5-to-v6/3"
    assert len(plan.receipt_contract_sha256) == 64
    assert repeated_plan == plan
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(StoreV6MigrationError, match="plan digest"):
        apply_store_v6_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before

    result = apply_store_v6_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )

    assert result.applied
    repeated = apply_store_v6_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert repeated.applied is False
    assert repeated.already_applied is True
    assert repeated.plan == plan
    with sqlite3.connect(source) as connection:
        assert connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone() == (6,)
    v6_backup = tmp_path / "backup-v6.sqlite"
    shutil.copyfile(source, v6_backup)
    os.chmod(v6_backup, 0o600)
    v7_plan = plan_store_v7_migration(source, sqlite_backup=v6_backup)
    apply_store_v7_migration(
        source,
        expected_plan_sha256=v7_plan.plan_sha256,
        sqlite_backup=v6_backup,
    )
    reopened = SQLiteStore(source)
    assert reopened.conn.execute(
        "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
    ).fetchone()[0] == 7
    legacy = reopened.get_semantic_legacy_coverage()
    assert legacy is not None
    assert legacy.source_schema_version == 5
    assert legacy.assessment_count == 2
    assert legacy.coverage == "unknown"
    assert reopened.semantic_flow_status_aggregate()["legacy_history"] == {
        "present": True,
        "source_schema_version": 5,
        "assessment_count": 2,
        "coverage": "unknown",
        "evidence_sha256": legacy.evidence_sha256,
        "created_at": legacy.created_at,
    }
    reopened.close()


def test_v6_plan_is_bound_to_the_selected_sqlite_database(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.sqlite"
    first_backup = tmp_path / "first-backup.sqlite"
    second_source = tmp_path / "second.sqlite"
    second_backup = tmp_path / "second-backup.sqlite"
    _v5_store(first_source)
    shutil.copyfile(first_source, first_backup)
    shutil.copyfile(first_source, second_source)
    shutil.copyfile(second_source, second_backup)
    for path in (first_backup, second_source, second_backup):
        os.chmod(path, 0o600)

    first_plan = plan_store_v6_migration(
        first_source,
        sqlite_backup=first_backup,
    )
    second_plan = plan_store_v6_migration(
        second_source,
        sqlite_backup=second_backup,
    )

    assert first_plan.source_digest_sha256 == second_plan.source_digest_sha256
    assert first_plan.database_identity_sha256 != second_plan.database_identity_sha256
    assert first_plan.plan_sha256 != second_plan.plan_sha256
    with pytest.raises(StoreV6MigrationError, match="plan digest"):
        apply_store_v6_migration(
            second_source,
            sqlite_backup=second_backup,
            expected_plan_sha256=first_plan.plan_sha256,
        )


@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_v6_apply_reconciles_exact_target_after_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v5_store(source)
    _insert_v5_assessments(source, 2)
    shutil.copyfile(source, backup)
    os.chmod(backup, 0o600)
    plan = plan_store_v6_migration(source, sqlite_backup=backup)

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
        real_require = semantic_v6_migration._require_canonical_v6
        calls = 0

        def fail_post_commit_readback(backend: object, connection: object) -> None:
            nonlocal calls
            real_require(backend, connection)
            calls += 1
            if calls == 2:
                raise RuntimeError("injected post-commit readback failure")

        monkeypatch.setattr(
            semantic_v6_migration,
            "_require_canonical_v6",
            fail_post_commit_readback,
        )
        expected_error = "post-commit readback"

    with pytest.raises(RuntimeError, match=expected_error):
        apply_store_v6_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )
    with sqlite3.connect(source) as connection:
        assert connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone() == (6,)

    with pytest.raises(StoreV6MigrationError, match="plan digest"):
        apply_store_v6_migration(
            source,
            expected_plan_sha256="0" * 64,
            sqlite_backup=backup,
        )

    result = apply_store_v6_migration(
        source,
        expected_plan_sha256=plan.plan_sha256,
        sqlite_backup=backup,
    )
    assert result.applied is False
    assert result.already_applied is True
    assert result.plan == plan


def test_v6_plan_requires_and_validates_independent_backup(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v5_store(source)
    with pytest.raises(StoreV6MigrationError, match="verified backup"):
        plan_store_v6_migration(source)
    with pytest.raises(StoreV6MigrationError, match="independent"):
        plan_store_v6_migration(source, sqlite_backup=source)
    _v5_store(backup)
    connection = sqlite3.connect(backup)
    try:
        connection.execute(
            "INSERT INTO runtime_counters (counter_name, value) VALUES (?, ?)",
            ("migration-test", 1),
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(backup, 0o600)
    with pytest.raises(StoreV6MigrationError, match="does not match"):
        plan_store_v6_migration(source, sqlite_backup=backup)


def test_runtime_refuses_v5_without_automatic_migration(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _v5_store(source)
    before = source.read_bytes()

    with pytest.raises(
        UnsupportedStoreVersion,
        match="explicit offline v5-to-v6 migration",
    ):
        SQLiteStore(source)

    assert source.read_bytes() == before
    connection = sqlite3.connect(source)
    try:
        assert connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone() == (5,)
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert present.isdisjoint(V6_TABLES)
    finally:
        connection.close()


def test_v5_to_v6_failure_rolls_back_ddl_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    _v5_store(source)
    shutil.copyfile(source, backup)
    os.chmod(backup, 0o600)
    plan = plan_store_v6_migration(source, sqlite_backup=backup)
    before = source.read_bytes()
    real_execute = semantic_v6_migration._execute_v6_ddl

    def fail_after_ddl(connection: object) -> None:
        real_execute(connection)
        raise StoreV6MigrationError("injected migration failure")

    monkeypatch.setattr(
        semantic_v6_migration,
        "_execute_v6_ddl",
        fail_after_ddl,
    )
    with pytest.raises(StoreV6MigrationError, match="injected"):
        apply_store_v6_migration(
            source,
            expected_plan_sha256=plan.plan_sha256,
            sqlite_backup=backup,
        )

    assert source.read_bytes() == before
    connection = sqlite3.connect(source)
    try:
        assert connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone() == (5,)
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert present.isdisjoint(V6_TABLES)
    finally:
        connection.close()
