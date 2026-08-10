from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.postgres_schema_contract import (
    POSTGRES_V4_BASELINE_4B43CB7_CATALOG_SHA256,
    expected_postgres_catalog,
    load_postgres_v6_manifest,
)
from agent_libos.storage.semantic_v5_migration import plan_store_v5_migration
from agent_libos.storage.v6_schema_contract import V6_TABLES


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    import psycopg
    from psycopg import sql

    base_dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_v5_contract_{uuid4().hex}"
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
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            "ALTER TABLE human_requests "
            "DROP CONSTRAINT human_requests_pkey",
            "storage (?:column|key constraint) contract|human request index contract",
        ),
        (
            "ALTER TABLE human_requests ALTER COLUMN status DROP NOT NULL",
            "storage column contract",
        ),
        (
            "DROP INDEX idx_human_requests_status_created",
            "human request index contract",
        ),
        (
            "ALTER TABLE human_requests "
            "ALTER COLUMN revision TYPE INTEGER",
            "storage column contract",
        ),
        (
            "ALTER TABLE human_requests "
            "ALTER COLUMN revision DROP NOT NULL",
            "storage column contract",
        ),
        (
            "ALTER TABLE semantic_assessment_jobs "
            "ALTER COLUMN attempt_count DROP DEFAULT",
            "storage column contract",
        ),
        (
            "ALTER TABLE semantic_assessment_jobs ADD CONSTRAINT "
            "duplicate_assessment_id UNIQUE (assessment_id)",
            "storage key constraint contract",
        ),
        (
            "ALTER TABLE human_requests "
            "DROP CONSTRAINT human_requests_revision_check",
            "storage CHECK contract",
        ),
        (
            "ALTER TABLE semantic_assessment_jobs "
            "DROP CONSTRAINT semantic_assessment_jobs_status_check",
            "storage CHECK contract",
        ),
        (
            "ALTER TABLE semantic_assessments "
            "ADD CHECK (length(record_json) >= 0)",
            "storage CHECK contract",
        ),
        (
            "ALTER TABLE semantic_assessments "
            "ALTER COLUMN action_id DROP NOT NULL",
            "storage column contract",
        ),
        (
            "DROP INDEX idx_semantic_assessments_action_tenant_created",
            "semantic index manifest",
        ),
        (
            "ALTER TABLE semantic_assessments ADD CONSTRAINT "
            "tamper_semantic_fk FOREIGN KEY (pid) REFERENCES processes(pid) "
            "ON DELETE CASCADE",
            "semantic relation boundary",
        ),
        (
            "ALTER TABLE human_requests ADD CONSTRAINT "
            "tamper_human_fk FOREIGN KEY (pid) REFERENCES processes(pid) "
            "ON DELETE CASCADE",
            "semantic relation boundary",
        ),
        (
            "CREATE TRIGGER tamper_semantic_trigger "
            "BEFORE UPDATE ON semantic_assessments FOR EACH ROW "
            "EXECUTE FUNCTION pg_catalog.suppress_redundant_updates_trigger()",
            "semantic relation boundary",
        ),
        (
            "CREATE TRIGGER tamper_human_trigger "
            "BEFORE UPDATE ON human_requests FOR EACH ROW "
            "EXECUTE FUNCTION pg_catalog.suppress_redundant_updates_trigger()",
            "semantic relation boundary",
        ),
        (
            "ALTER TABLE semantic_assessments ENABLE ROW LEVEL SECURITY",
            "semantic relation boundary",
        ),
        (
            "CREATE RULE tamper_semantic_rule AS ON DELETE "
            "TO semantic_assessments DO INSTEAD NOTHING",
            "semantic relation boundary",
        ),
    ],
)
def test_tampered_v5_postgres_contract_is_rejected_on_reopen(
    mutation: str,
    error: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(mutation)

        with pytest.raises(UnsupportedStoreVersion, match=error):
            PostgresStore(dsn)


@pytest.mark.postgres
def test_postgres_v4_plan_rejects_noncanonical_human_contract() -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for table in sorted(V6_TABLES):
                connection.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
                )
            connection.execute("DROP TABLE semantic_assessments")
            connection.execute("DROP TABLE semantic_assessment_jobs")
            connection.execute(
                "ALTER TABLE human_requests DROP COLUMN revision"
            )
            connection.execute(
                "ALTER TABLE human_requests ALTER COLUMN status DROP NOT NULL"
            )
            connection.execute(
                "UPDATE runtime_schema SET schema_version = 4 WHERE singleton = 1"
            )

        with pytest.raises(
            UnsupportedStoreVersion,
            match="schema v4 human request contract",
        ):
            plan_store_v5_migration(dsn)


@pytest.mark.postgres
def test_fresh_v6_postgres_catalog_manifest_reopens() -> None:
    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        PostgresStore(dsn).close()


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("statements", "case"),
    [
        (
            (
                "ALTER TABLE capabilities ALTER COLUMN constraints_json "
                "TYPE BIGINT USING 0",
            ),
            "non-semantic column type",
        ),
        (
            (
                "ALTER TABLE capabilities ALTER COLUMN constraints_json "
                "DROP NOT NULL",
            ),
            "non-semantic column nullability",
        ),
        (
            (
                "ALTER TABLE llm_calls ALTER COLUMN payload_retention_tier "
                "SET DEFAULT 'summary'",
            ),
            "non-semantic column default",
        ),
        (
            (
                "ALTER TABLE capabilities ALTER COLUMN constraints_json "
                "TYPE TEXT COLLATE \"C\"",
            ),
            "non-semantic column collation",
        ),
        (
            (
                "ALTER TABLE runtime_counters ALTER COLUMN value "
                "ADD GENERATED BY DEFAULT AS IDENTITY",
            ),
            "identity drift",
        ),
        (
            (
                "ALTER TABLE capabilities ADD COLUMN injected_generated BIGINT "
                "GENERATED ALWAYS AS (0) STORED",
            ),
            "generated column drift",
        ),
        (
            ("ALTER TABLE capabilities DROP CONSTRAINT capabilities_pkey",),
            "primary key drift",
        ),
        (
            (
                "ALTER TABLE task_runs DROP CONSTRAINT "
                "task_runs_spec_schema_version_check",
            ),
            "check drift",
        ),
        (
            (
                "ALTER TABLE process_terminal_cleanups DROP CONSTRAINT "
                "process_terminal_cleanups_pid_fkey",
                "ALTER TABLE process_terminal_cleanups ADD CONSTRAINT "
                "process_terminal_cleanups_pid_fkey FOREIGN KEY (pid) "
                "REFERENCES processes(pid) ON DELETE SET NULL",
            ),
            "foreign key action drift",
        ),
        (
            (
                "DROP INDEX idx_operations_runtime_publication",
                "CREATE UNIQUE INDEX idx_operations_runtime_publication "
                "ON operations(runtime_publication_id) "
                "WHERE runtime_publication_id IS NULL",
            ),
            "partial index predicate drift",
        ),
        (
            (
                "DROP INDEX idx_operations_state_started",
                "CREATE INDEX idx_operations_state_started "
                "ON operations(state, started_at, operation_id)",
            ),
            "descending index drift",
        ),
        (
            (
                "DROP INDEX idx_capabilities_subject_status",
                "CREATE INDEX idx_capabilities_subject_status "
                "ON capabilities(subject text_pattern_ops, status)",
            ),
            "operator class drift",
        ),
        (
            (
                "DROP INDEX idx_capabilities_subject_status",
                "CREATE INDEX idx_capabilities_subject_status "
                "ON capabilities(subject, status) INCLUDE (resource)",
            ),
            "included column drift",
        ),
        (
            (
                "CREATE TRIGGER injected_capabilities_trigger "
                "BEFORE UPDATE ON capabilities FOR EACH ROW EXECUTE FUNCTION "
                "pg_catalog.suppress_redundant_updates_trigger()",
            ),
            "user trigger",
        ),
        (
            ("ALTER TABLE capabilities ENABLE ROW LEVEL SECURITY",),
            "row level security",
        ),
        (
            (
                "CREATE POLICY injected_capabilities_policy ON capabilities "
                "USING (true)",
            ),
            "row policy",
        ),
        (
            (
                "CREATE RULE injected_capabilities_rule AS ON DELETE TO "
                "capabilities DO INSTEAD NOTHING",
            ),
            "rewrite rule",
        ),
        (
            ("ALTER TABLE capabilities SET UNLOGGED",),
            "relation persistence",
        ),
        (
            ("ALTER TABLE capabilities REPLICA IDENTITY FULL",),
            "replica identity",
        ),
        (
            ("ALTER TABLE capabilities SET (fillfactor = 70)",),
            "relation options",
        ),
        (("CREATE VIEW injected_view AS SELECT 1 AS value",), "extra view"),
        (("CREATE SEQUENCE injected_sequence",), "extra sequence"),
        (
            (
                "CREATE TABLE injected_parent ()",
                "ALTER TABLE capabilities INHERIT injected_parent",
            ),
            "table inheritance",
        ),
        (
            (
                "CREATE TABLE injected_partitioned (id INTEGER) "
                "PARTITION BY RANGE (id)",
            ),
            "partitioned relation",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_full_postgres_catalog_tamper_is_rejected(
    statements: tuple[str, ...],
    case: str,
) -> None:
    del case
    import psycopg

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for statement in statements:
                connection.execute(statement)

        with pytest.raises(
            UnsupportedStoreVersion,
            match="schema v6",
        ):
            PostgresStore(dsn)


def test_v4_manifest_is_pinned_to_actual_baseline_4b43cb7_catalog() -> None:
    import hashlib
    import json

    payload = json.dumps(
        expected_postgres_catalog(4),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == (
        POSTGRES_V4_BASELINE_4B43CB7_CATALOG_SHA256
    )


def test_committed_postgres_manifest_records_pinned_pg17_10_generator() -> None:
    manifest = load_postgres_v6_manifest()
    assert manifest["generated_postgres_version_num"] == 170010
    assert manifest["catalog"]["postgres_major"] == 17
