from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.semantic_v5_migration import (
    apply_store_v5_migration,
    plan_store_v5_migration,
)


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


@pytest.mark.postgres
def test_postgres_v4_to_v5_migration_round_trip() -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
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
        reopened = PostgresStore(dsn)
        try:
            assert reopened.conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == {"schema_version": 5}
        finally:
            reopened.close()
