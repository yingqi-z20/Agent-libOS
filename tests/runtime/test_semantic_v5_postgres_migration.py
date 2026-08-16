from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

import agent_libos.storage.semantic_v5_migration as semantic_v5_migration
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.semantic_v5_migration import (
    StoreV5MigrationError,
    apply_store_v5_migration,
    plan_store_v5_migration,
)
from agent_libos.storage.semantic_v6_migration import plan_store_v6_migration
from agent_libos.storage.v6_schema_contract import V6_TABLES
from agent_libos.storage.v7_schema_contract import V7_TABLES


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    import psycopg
    from psycopg import sql

    base_dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_v5_migration_{uuid4().hex}"
    parsed = urlsplit(base_dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    selected_dsn = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )
    with psycopg.connect(base_dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    try:
        yield selected_dsn
    finally:
        with psycopg.connect(base_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _downgrade_to_v4(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    PostgresStore(dsn).close()
    with psycopg.connect(dsn, autocommit=True) as connection:
        for table in sorted(V7_TABLES | V6_TABLES):
            connection.execute(
                sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
            )
        connection.execute("DROP TABLE semantic_assessments")
        connection.execute("DROP TABLE semantic_assessment_jobs")
        connection.execute("ALTER TABLE human_requests DROP COLUMN revision")
        connection.execute(
            "UPDATE runtime_schema SET schema_version = 4 WHERE singleton = 1"
        )


@pytest.mark.postgres
def test_postgres_v4_to_v5_migration_round_trip() -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for table in sorted(V7_TABLES | V6_TABLES):
                connection.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
                )
            connection.execute("DROP TABLE semantic_assessments")
            connection.execute("DROP TABLE semantic_assessment_jobs")
            connection.execute(
                "ALTER TABLE human_requests DROP COLUMN revision"
            )
            connection.execute(
                "UPDATE runtime_schema SET schema_version = 4 WHERE singleton = 1"
            )

        plan = plan_store_v5_migration(dsn)
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (4,)

        result = apply_store_v5_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )

        assert result.applied is True
        assert result.plan == plan
        next_plan = plan_store_v6_migration(dsn)
        assert next_plan.from_schema_version == 5
        assert next_plan.to_schema_version == 6
        with pytest.raises(
            UnsupportedStoreVersion,
            match="explicit offline v5-to-v6 migration",
        ):
            PostgresStore(dsn)


@pytest.mark.postgres
def test_postgres_v5_plan_is_bound_to_database_and_schema_identity() -> None:
    with _postgres_schema_dsn() as first_dsn, _postgres_schema_dsn() as second_dsn:
        _downgrade_to_v4(first_dsn)
        _downgrade_to_v4(second_dsn)
        first_plan = plan_store_v5_migration(first_dsn)
        second_plan = plan_store_v5_migration(second_dsn)

        assert first_plan.source_catalog_sha256 == second_plan.source_catalog_sha256
        assert (
            first_plan.database_identity_sha256
            != second_plan.database_identity_sha256
        )
        assert first_plan.plan_sha256 != second_plan.plan_sha256
        with pytest.raises(StoreV5MigrationError, match="plan digest"):
            apply_store_v5_migration(
                second_dsn,
                expected_plan_sha256=first_plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )


@pytest.mark.postgres
@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_postgres_v5_reconciles_exact_target_after_uncertain_commit(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v4(dsn)
        plan = plan_store_v5_migration(dsn)
        if fault == "commit_ack":
            real_commit = semantic_v5_migration._PostgresConnection.commit

            def lost_commit_ack(connection: object) -> None:
                real_commit(connection)
                raise RuntimeError("injected lost commit ACK")

            monkeypatch.setattr(
                semantic_v5_migration._PostgresConnection,
                "commit",
                lost_commit_ack,
            )
            expected_error = "lost commit ACK"
        else:
            real_require = semantic_v5_migration._require_canonical_v5
            calls = 0

            def fail_post_commit_readback(
                backend: object,
                connection: object,
            ) -> None:
                nonlocal calls
                real_require(backend, connection)
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected post-commit readback failure")

            monkeypatch.setattr(
                semantic_v5_migration,
                "_require_canonical_v5",
                fail_post_commit_readback,
            )
            expected_error = "post-commit readback"

        with pytest.raises(RuntimeError, match=expected_error):
            apply_store_v5_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (5,)

        result = apply_store_v5_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )
        assert result.applied is False
        assert result.already_applied is True
        assert result.plan == plan
