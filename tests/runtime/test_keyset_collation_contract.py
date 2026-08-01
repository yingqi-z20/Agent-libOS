from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.models import RuntimePublicationCursor
from agent_libos.models.exceptions import UnsupportedStoreVersion
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.sql import _V4_KEYSET_TEXT_COLUMNS
from agent_libos.storage.sqlite import SQLiteStore


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]

_SHARED_TIMESTAMP = "2026-01-01T00:00:00Z"
_MIXED_IDS = (
    "publication_a",
    "publication_A",
    "publication_ÿ",
    "publication_Ā",
)
_MIXED_TIMESTAMPS = (
    "cursor_a",
    "cursor_A",
    "cursor_ÿ",
    "cursor_Ā",
)


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_publication_keyset_matches_python_for_mixed_unicode_on_both_tuple_axes(
    backend: str,
) -> None:
    with _store_for_backend(backend) as store:
        _insert_publications(
            store,
            kind="process_launch",
            rows=[(publication_id, _SHARED_TIMESTAMP) for publication_id in _MIXED_IDS],
        )
        id_records, id_cursors = _collect_publication_pages(
            store,
            kind="process_launch",
        )
        assert [record["publication_id"] for record in id_records] == sorted(
            _MIXED_IDS
        )
        assert len({record["publication_id"] for record in id_records}) == len(
            _MIXED_IDS
        )
        assert id_cursors == sorted(id_cursors)

        primary_rows = [
            (f"timestamp-axis-{index}", created_at)
            for index, created_at in enumerate(_MIXED_TIMESTAMPS)
        ]
        _insert_publications(store, kind="process_exec", rows=primary_rows)
        timestamp_records, timestamp_cursors = _collect_publication_pages(
            store,
            kind="process_exec",
        )
        assert [record["created_at"] for record in timestamp_records] == sorted(
            _MIXED_TIMESTAMPS
        )
        assert len(timestamp_records) == len(_MIXED_TIMESTAMPS)
        assert timestamp_cursors == sorted(timestamp_cursors)


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_keyset_schema_probe_is_single_query_and_all_columns_are_canonical(
    backend: str,
) -> None:
    with _store_for_backend(backend) as store:
        counted = _CountingConnection(store.conn)
        collations = type(store)._probe_text_column_collations(counted)
        expected_name = "BINARY" if backend == "sqlite" else "C"
        expected_columns = {
            (table, column)
            for table, columns in _V4_KEYSET_TEXT_COLUMNS.items()
            for column in columns
        }
        assert counted.execute_calls == 1
        assert set(collations) == expected_columns
        assert set(collations.values()) == {expected_name}
        if backend == "postgres":
            row = store.conn.execute("SHOW server_encoding").fetchone()
            assert row is not None
            assert str(row["server_encoding"]).upper() == "UTF8"


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_publication_resumed_keyset_plan_is_composite_and_sort_free(
    backend: str,
) -> None:
    with _store_for_backend(backend) as store:
        query = """
            SELECT * FROM runtime_publications
             WHERE kind = ? AND state = ? AND operation_reconciled = ?
               AND (created_at, publication_id) > (?, ?)
             ORDER BY created_at, publication_id LIMIT ?
        """
        params = (
            "process_launch",
            "planning",
            0,
            _SHARED_TIMESTAMP,
            "publication_A",
            2,
        )
        if backend == "sqlite":
            rows = store.conn.execute(f"EXPLAIN QUERY PLAN {query}", params)
            plan = "\n".join(str(row["detail"]) for row in rows)
            normalized = plan.replace(" ", "")
            assert "idx_runtime_publications_operation_reconciliation" in plan
            assert "(created_at,publication_id)>" in normalized
            assert "USE TEMP B-TREE" not in plan
        else:
            with store.transaction() as cursor:
                cursor.execute("SET LOCAL enable_seqscan = off")
                rows = list(cursor.execute(f"EXPLAIN (COSTS OFF) {query}", params))
            plan = "\n".join(str(row["QUERY PLAN"]) for row in rows)
            assert "idx_runtime_publications_operation_reconciliation" in plan
            assert "Sort" not in plan


def test_postgres_keyset_probe_rejects_non_utf8_in_its_single_catalog_query() -> None:
    connection = _StaticRowsConnection(
        [
            {
                "table_name": "runtime_publications",
                "column_name": "publication_id",
                "collation_name": "C",
                "server_encoding": "LATIN1",
            }
        ]
    )
    with pytest.raises(UnsupportedStoreVersion, match="requires UTF8"):
        PostgresStore._probe_text_column_collations(connection)
    assert connection.execute_calls == 1


@pytest.mark.parametrize(
    "spoofed_declaration",
    [
        "created_at TEXT /* COLLATE BINARY */ COLLATE NOCASE NOT NULL",
        "created_at TEXT -- COLLATE BINARY\n COLLATE NOCASE NOT NULL",
        "created_at TEXT DEFAULT 'COLLATE BINARY' COLLATE NOCASE NOT NULL",
    ],
    ids=("block-comment", "line-comment", "string-literal"),
)
def test_sqlite_keyset_probe_rejects_collation_spoofed_by_non_code_sql(
    tmp_path: Path,
    spoofed_declaration: str,
) -> None:
    db_path = tmp_path / "collation-comment-spoof.sqlite"
    SQLiteStore(db_path).close()

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'runtime_publications'"
        ).fetchone()
        assert row is not None
        original = str(row[0])
        canonical = "created_at TEXT COLLATE BINARY NOT NULL"
        assert original.count(canonical) == 1
        spoofed = original.replace(canonical, spoofed_declaration, 1)
        schema_version = int(
            connection.execute("PRAGMA schema_version").fetchone()[0]
        )
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = ? "
            "WHERE type = 'table' AND name = 'runtime_publications'",
            (spoofed,),
        )
        connection.execute(f"PRAGMA schema_version = {schema_version + 1}")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedStoreVersion, match="keyset collation"):
        SQLiteStore(db_path)


def _insert_publications(
    store: SQLiteStore | PostgresStore,
    *,
    kind: str,
    rows: list[tuple[str, str]],
) -> None:
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO runtime_publications (
                publication_id, kind, pid, owner_instance_id, state, phase,
                plan_json, receipt_json, error_json, operation_reconciled,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    publication_id,
                    kind,
                    f"pid-{kind}",
                    "keyset-contract-owner",
                    "planning",
                    "planned",
                    "{}",
                    '{"artifacts": [], "phases": []}',
                    None,
                    0,
                    created_at,
                    created_at,
                )
                for publication_id, created_at in reversed(rows)
            ],
        )


def _collect_publication_pages(
    store: SQLiteStore | PostgresStore,
    *,
    kind: str,
) -> tuple[list[dict[str, Any]], list[RuntimePublicationCursor]]:
    records: list[dict[str, Any]] = []
    cursors: list[RuntimePublicationCursor] = []
    after: RuntimePublicationCursor | None = None
    while True:
        page = store.query_runtime_publication_recovery(
            kind=kind,
            state="planning",
            operation_reconciled=False,
            after=after,
            limit=2,
        )
        records.extend(page.records)
        if page.next_cursor is None:
            return records, cursors
        cursors.append(page.next_cursor)
        after = page.next_cursor


class _CountingConnection:
    def __init__(self, connection: Any):
        self.connection = connection
        self.execute_calls = 0

    def execute(self, sql: str, params: Any = ()) -> Any:
        self.execute_calls += 1
        return self.connection.execute(sql, params)


class _StaticRowsConnection:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self.execute_calls = 0

    def execute(self, _sql: str, _params: Any = ()) -> list[dict[str, str]]:
        self.execute_calls += 1
        return self.rows


@contextlib.contextmanager
def _store_for_backend(
    backend: str,
) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        store = SQLiteStore(":memory:")
        try:
            yield store
        finally:
            store.close()
        return
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            yield store
        finally:
            store.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_keyset_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
