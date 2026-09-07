from __future__ import annotations

import pytest

import agent_libos.storage.llm_v8_migration as llm_v8_migration
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage.llm_v8_migration import (
    StoreV8MigrationError,
    apply_store_v8_migration,
    plan_store_v8_migration,
)
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.v8_schema_contract import V8_TABLES
from tests.runtime.test_semantic_v5_postgres_migration import _postgres_schema_dsn


pytestmark = pytest.mark.postgres


def _downgrade_to_v7(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    PostgresStore(dsn).close()
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("DROP INDEX IF EXISTS idx_llm_pending_replay_recovery")
        for table in sorted(V8_TABLES):
            connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 7 "
            "WHERE singleton = 1 AND schema_version = 8"
        )
        assert changed.rowcount == 1
        connection.execute(
            "INSERT INTO runtime_counters (counter_name, value) VALUES (%s, %s)",
            ("llm-v8-postgres-sentinel", 8),
        )


def test_postgres_v7_to_v8_migration_round_trip() -> None:
    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v7(dsn)
        plan = plan_store_v8_migration(dsn)
        assert plan.from_schema_version == 7
        assert plan.to_schema_version == 8
        with pytest.raises(UnsupportedStoreVersion, match="v7-to-v8"):
            PostgresStore(dsn)
        result = apply_store_v8_migration(
            dsn, expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )
        assert result.applied is True
        assert apply_store_v8_migration(
            dsn, expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        ).already_applied is True
        reopened = PostgresStore(dsn)
        try:
            assert reopened.conn.execute(
                "SELECT value FROM runtime_counters WHERE counter_name = ?",
                ("llm-v8-postgres-sentinel",),
            ).fetchone()["value"] == 8
            for table in V8_TABLES:
                assert reopened.conn.execute(
                    f'SELECT COUNT(*) AS count FROM "{table}"'
                ).fetchone()["count"] == 0
        finally:
            reopened.close()


def test_postgres_v8_ddl_failure_rolls_back_source_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v7(dsn)
        plan = plan_store_v8_migration(dsn)
        real_ddl = llm_v8_migration._execute_v8_ddl

        def corrupt_source(connection: object) -> None:
            real_ddl(connection)
            connection.execute(  # type: ignore[attr-defined]
                "UPDATE runtime_counters SET value = 99 "
                "WHERE counter_name = 'llm-v8-postgres-sentinel'"
            )

        monkeypatch.setattr(llm_v8_migration, "_execute_v8_ddl", corrupt_source)
        with pytest.raises(StoreV8MigrationError, match="changed the locked source state"):
            apply_store_v8_migration(
                dsn, expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (7,)
            assert connection.execute(
                "SELECT value FROM runtime_counters "
                "WHERE counter_name = 'llm-v8-postgres-sentinel'"
            ).fetchone() == (8,)
