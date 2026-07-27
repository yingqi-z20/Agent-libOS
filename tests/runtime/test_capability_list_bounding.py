from __future__ import annotations

from typing import Any

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import SQLiteStore, UnitOfWork


def _seed_capabilities(store: SQLiteStore, count: int) -> None:
    rows: list[tuple[Any, ...]] = [
        (
            f"cap-{index:05d}",
            "pid-subject",
            f"object:oid-{index:05d}",
            '["read"]',
            "{}",
            "test",
            f"2026-01-01T00:00:{index:05d}Z",
            None,
            0,
            1,
            "allow",
            None,
            None,
            0,
            None,
            None,
            "active",
            "{}",
        )
        for index in range(count)
    ]
    with store.transaction() as cursor:
        cursor.executemany(
            """
            INSERT INTO capabilities (
                cap_id, subject, resource, rights_json, constraints_json,
                issued_by, issued_at, expires_at, delegable, revocable, effect,
                issuer_cap_id, parent_cap_id, delegation_depth,
                max_delegation_depth, uses_remaining, status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_global_capability_list_applies_sql_limit_before_decoding() -> None:
    store = SQLiteStore(":memory:")
    try:
        _seed_capabilities(store, 250)
        decoded: list[str] = []
        decode = store._row_to_capability
        store._row_to_capability = lambda row: (
            decoded.append(str(row["cap_id"])),
            decode(row),
        )[1]
        traced: list[str] = []
        store.conn.set_trace_callback(traced.append)

        records = UnitOfWork(store).authority.list_capabilities(limit=100)
        store.conn.set_trace_callback(None)

        assert len(records) == 100
        assert len(decoded) == 100
        assert [record.cap_id for record in records] == [
            f"cap-{index:05d}" for index in range(100)
        ]
        queries = [
            statement
            for statement in traced
            if statement.startswith("SELECT * FROM capabilities")
        ]
        assert len(queries) == 1
        assert queries[0].endswith("LIMIT 100")
    finally:
        store.close()


def test_subject_capability_list_default_remains_complete_for_authority_checks() -> None:
    store = SQLiteStore(":memory:")
    try:
        _seed_capabilities(store, 250)

        records = UnitOfWork(store).authority.list_capabilities("pid-subject")

        assert len(records) == 250
    finally:
        store.close()


@pytest.mark.parametrize("limit", [True, 0, -1, 101, "1", 1.0])
def test_capability_list_limit_rejects_coercion_and_config_overflow(
    limit: object,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="capability list limit"):
            store.list_capabilities(limit=limit)
    finally:
        store.close()


def test_capability_page_downpushes_subject_active_cursor_and_limit() -> None:
    store = SQLiteStore(":memory:")
    try:
        _seed_capabilities(store, 12)
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE capabilities SET status = 'revoked' WHERE cap_id = ?",
                ("cap-00001",),
            )
            cursor.execute(
                "UPDATE capabilities SET subject = ? WHERE cap_id = ?",
                ("pid-other", "cap-00003"),
            )

        decoded: list[str] = []
        decode = store._row_to_capability
        store._row_to_capability = lambda row: (
            decoded.append(str(row["cap_id"])),
            decode(row),
        )[1]
        traced: list[str] = []
        store.conn.set_trace_callback(traced.append)

        records = UnitOfWork(store).authority.query_capabilities(
            "pid-subject",
            active_only=True,
            after_cap_id="cap-00000",
            limit=3,
        )
        store.conn.set_trace_callback(None)

        assert [record.cap_id for record in records] == [
            "cap-00002",
            "cap-00004",
            "cap-00005",
        ]
        assert decoded == ["cap-00002", "cap-00004", "cap-00005"]
        queries = [
            statement
            for statement in traced
            if statement.startswith("SELECT * FROM capabilities WHERE")
        ]
        assert len(queries) == 1
        query = queries[0]
        assert "subject = 'pid-subject'" in query
        assert "status = 'active'" in query
        assert "cap_id > 'cap-00000'" in query
        assert query.endswith("ORDER BY cap_id ASC LIMIT 3")
    finally:
        store.close()


def test_global_capability_pages_use_exclusive_id_cursor_without_gaps() -> None:
    store = SQLiteStore(":memory:")
    try:
        _seed_capabilities(store, 9)
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE capabilities SET status = 'disabled' WHERE cap_id = ?",
                ("cap-00004",),
            )

        first = store.query_capabilities(
            active_only=False,
            after_cap_id=None,
            limit=4,
        )
        second = store.query_capabilities(
            active_only=False,
            after_cap_id=str(first[-1].cap_id),
            limit=4,
        )
        third = store.query_capabilities(
            active_only=False,
            after_cap_id=str(second[-1].cap_id),
            limit=4,
        )

        assert [record.cap_id for record in first + second + third] == [
            f"cap-{index:05d}" for index in range(9)
        ]
    finally:
        store.close()


@pytest.mark.parametrize("active_only", [None, 0, 1, "true"])
def test_capability_page_rejects_coerced_active_filter(active_only: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="active_only"):
            store.query_capabilities(
                active_only=active_only,
                after_cap_id=None,
                limit=1,
            )
    finally:
        store.close()


@pytest.mark.parametrize("after_cap_id", ["", True, 1, 1.0])
def test_capability_page_rejects_invalid_cursor(after_cap_id: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="cursor"):
            store.query_capabilities(
                active_only=True,
                after_cap_id=after_cap_id,
                limit=1,
            )
    finally:
        store.close()


@pytest.mark.parametrize("limit", [True, 0, -1, 101, "1", 1.0])
def test_capability_page_rejects_coerced_or_oversized_limit(limit: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="capability page limit"):
            store.query_capabilities(
                active_only=True,
                after_cap_id=None,
                limit=limit,
            )
    finally:
        store.close()


@pytest.mark.parametrize("subject", ["", True, 1, 1.0])
def test_capability_page_rejects_invalid_subject(subject: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="subject"):
            store.query_capabilities(
                subject,
                active_only=True,
                after_cap_id=None,
                limit=1,
            )
    finally:
        store.close()
