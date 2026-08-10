from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
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
from agent_libos.storage.v6_schema_contract import V6_TABLES


def _v5_store(path: Path) -> None:
    store = SQLiteStore(path)
    store.close()
    connection = sqlite3.connect(path)
    try:
        for table in sorted(V6_TABLES):
            connection.execute(f'DROP TABLE "{table}"')
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 5 "
            "WHERE singleton = 1 AND schema_version = 6"
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
    reopened = SQLiteStore(source)
    assert reopened.conn.execute(
        "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
    ).fetchone()[0] == 6
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
