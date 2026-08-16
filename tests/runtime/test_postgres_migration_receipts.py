from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

import pytest

import agent_libos.storage.mcp_v7_migration as mcp_v7_migration
import agent_libos.storage.semantic_v5_migration as semantic_v5_migration
import agent_libos.storage.semantic_v6_migration as semantic_v6_migration
from agent_libos.storage.mcp_v7_migration import (
    StoreV7MigrationError,
    apply_store_v7_migration,
    plan_store_v7_migration,
)
from agent_libos.storage.postgres import (
    _PostgresConnection,
    _postgres_runtime_lock_key,
)
from agent_libos.storage.semantic_v5_migration import (
    StoreV5MigrationError,
    apply_store_v5_migration,
    plan_store_v5_migration,
)
from agent_libos.storage.semantic_v6_migration import (
    StoreV6MigrationError,
    apply_store_v6_migration,
    plan_store_v6_migration,
)
from tests.runtime.test_mcp_v7_postgres import _downgrade_to_v6
from tests.runtime.test_semantic_v5_postgres_migration import (
    _downgrade_to_v4,
    _postgres_schema_dsn,
)
from tests.runtime.test_semantic_v6_postgres_migration import _downgrade_to_v5


pytestmark = pytest.mark.postgres


@dataclass(frozen=True)
class _MigrationCase:
    label: str
    module: Any
    setup_source: Callable[[str], None]
    plan: Callable[[str], Any]
    apply: Callable[..., Any]
    error: type[Exception]
    from_version: int
    to_version: int


_CASES = (
    _MigrationCase(
        label="v4-to-v5",
        module=semantic_v5_migration,
        setup_source=_downgrade_to_v4,
        plan=plan_store_v5_migration,
        apply=apply_store_v5_migration,
        error=StoreV5MigrationError,
        from_version=4,
        to_version=5,
    ),
    _MigrationCase(
        label="v5-to-v6",
        module=semantic_v6_migration,
        setup_source=_downgrade_to_v5,
        plan=plan_store_v6_migration,
        apply=apply_store_v6_migration,
        error=StoreV6MigrationError,
        from_version=5,
        to_version=6,
    ),
    _MigrationCase(
        label="v6-to-v7",
        module=mcp_v7_migration,
        setup_source=_downgrade_to_v6,
        plan=plan_store_v7_migration,
        apply=apply_store_v7_migration,
        error=StoreV7MigrationError,
        from_version=6,
        to_version=7,
    ),
)


def _case_id(case: _MigrationCase) -> str:
    return case.label


def _apply(case: _MigrationCase, dsn: str, plan: Any) -> Any:
    return case.apply(
        dsn,
        expected_plan_sha256=plan.plan_sha256,
        postgres_snapshot_confirmed=True,
    )


def _receipt_id(plan: Any) -> str:
    return (
        f"store-migration-v{plan.from_schema_version}-to-"
        f"v{plan.to_schema_version}:{plan.plan_sha256}"
    )


def _schema_version(dsn: str) -> int:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
        ).fetchone()
    assert row is not None
    return int(row[0])


def _receipt_count(dsn: str, *, record_id: str | None = None) -> int:
    import psycopg

    statement = (
        "SELECT COUNT(*) FROM audit_records "
        "WHERE action = 'store.schema_migration_committed'"
    )
    params: tuple[object, ...] = ()
    if record_id is not None:
        statement += " AND record_id = %s"
        params = (record_id,)
    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(statement, params).fetchone()
    assert row is not None
    return int(row[0])


def _receipt_decision(dsn: str, *, record_id: str) -> dict[str, Any]:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        row = connection.execute(
            "SELECT decision_json FROM audit_records WHERE record_id = %s",
            (record_id,),
        ).fetchone()
    assert row is not None
    selected = json.loads(str(row[0]))
    assert isinstance(selected, dict)
    return selected


class _SingleRowCursor:
    def __init__(self, row: dict[str, Any]):
        self._row = row

    def fetchone(self) -> dict[str, Any]:
        return dict(self._row)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.parametrize("mutation", ("insert", "update", "delete"))
def test_postgres_plan_rejects_source_tuple_drift(
    case: _MigrationCase,
    mutation: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        counter_name = f"migration-drift-{mutation}"
        if mutation in {"update", "delete"}:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(
                    "INSERT INTO runtime_counters (counter_name, value) "
                    "VALUES (%s, %s)",
                    (counter_name, 1),
                )
        plan = case.plan(dsn)

        with psycopg.connect(dsn, autocommit=True) as connection:
            if mutation == "insert":
                changed = connection.execute(
                    "INSERT INTO runtime_counters (counter_name, value) "
                    "VALUES (%s, %s)",
                    (counter_name, 1),
                )
            elif mutation == "update":
                changed = connection.execute(
                    "UPDATE runtime_counters SET value = value + 1 "
                    "WHERE counter_name = %s",
                    (counter_name,),
                )
            else:
                changed = connection.execute(
                    "DELETE FROM runtime_counters WHERE counter_name = %s",
                    (counter_name,),
                )
            assert changed.rowcount == 1

        with pytest.raises(case.error, match="plan digest"):
            _apply(case, dsn, plan)

        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn, record_id=_receipt_id(plan)) == 0


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.parametrize("receipt_fault", ("missing", "tampered"))
def test_postgres_same_version_retry_requires_exact_plan_receipt(
    case: _MigrationCase,
    receipt_fault: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        plan = case.plan(dsn)
        result = _apply(case, dsn, plan)
        assert result.applied is True
        receipt_id = _receipt_id(plan)
        assert _receipt_count(dsn, record_id=receipt_id) == 1
        decision = _receipt_decision(dsn, record_id=receipt_id)
        assert set(decision) == {
            "schema_version",
            "kind",
            "outcome",
            "plan",
            "binding_sha256",
        }
        binding = {
            key: value
            for key, value in decision.items()
            if key != "binding_sha256"
        }
        encoded_binding = json.dumps(
            binding,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert decision["schema_version"] == 1
        assert decision["kind"] == "agent_libos_store_migration_receipt"
        assert decision["outcome"] == "committed"
        assert decision["plan"] == plan.to_dict()
        assert decision["binding_sha256"] == hashlib.sha256(
            encoded_binding
        ).hexdigest()

        with psycopg.connect(dsn, autocommit=True) as connection:
            if receipt_fault == "missing":
                changed = connection.execute(
                    "DELETE FROM audit_records WHERE record_id = %s",
                    (receipt_id,),
                )
            else:
                changed = connection.execute(
                    "UPDATE audit_records SET decision_json = %s "
                    "WHERE record_id = %s",
                    ("{}", receipt_id),
                )
            assert changed.rowcount == 1

        with pytest.raises(case.error, match="receipt"):
            _apply(case, dsn, plan)

        assert _schema_version(dsn) == case.to_version


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_postgres_apply_rejects_search_path_identity_swap_after_lease(
    case: _MigrationCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        plan = case.plan(dsn)
        decoy_schema = f"agent_libos_migration_decoy_{uuid4().hex}"
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(decoy_schema))
            )
            connection.execute(
                sql.SQL(
                    "CREATE TABLE {}.runtime_schema ("
                    "singleton INTEGER PRIMARY KEY, "
                    "schema_version INTEGER NOT NULL)"
                ).format(sql.Identifier(decoy_schema))
            )
            connection.execute(
                sql.SQL(
                    "INSERT INTO {}.runtime_schema "
                    "(singleton, schema_version) VALUES (1, %s)"
                ).format(sql.Identifier(decoy_schema)),
                (case.from_version,),
            )

        real_lock = case.module._postgres_lock_source_relations

        def lock_then_swap_search_path(
            connection: Any,
            *,
            schema: str,
            tables: tuple[str, ...],
        ) -> None:
            real_lock(connection, schema=schema, tables=tables)
            connection._conn.execute(
                sql.SQL("SET LOCAL search_path TO {}").format(
                    sql.Identifier(decoy_schema)
                )
            )

        monkeypatch.setattr(
            case.module,
            "_postgres_lock_source_relations",
            lock_then_swap_search_path,
        )
        try:
            with pytest.raises(case.error, match="identity or marker changed"):
                _apply(case, dsn, plan)
        finally:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(decoy_schema)
                    )
                )

        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn, record_id=_receipt_id(plan)) == 0


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_postgres_plan_is_bound_to_cluster_system_identity(
    case: _MigrationCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        real_execute = semantic_v5_migration._PostgresConnection.execute
        selected_system_identifier = {"value": "10000000000000000001"}

        def execute_with_system_identifier(
            connection: Any,
            statement: str,
            params: tuple[object, ...] = (),
        ) -> Any:
            cursor = real_execute(connection, statement, params)
            if "pg_control_system" not in statement:
                return cursor
            row = cursor.fetchone()
            assert row is not None
            selected = dict(row)
            selected["control_system_allowed"] = True
            selected["system_identifier"] = selected_system_identifier["value"]
            return _SingleRowCursor(selected)

        monkeypatch.setattr(
            semantic_v5_migration._PostgresConnection,
            "execute",
            execute_with_system_identifier,
        )
        first_plan = case.plan(dsn)
        selected_system_identifier["value"] = "10000000000000000002"
        second_plan = case.plan(dsn)

        assert (
            first_plan.database_identity_sha256
            != second_plan.database_identity_sha256
        )
        assert first_plan.plan_sha256 != second_plan.plan_sha256
        with pytest.raises(case.error, match="plan digest"):
            _apply(case, dsn, first_plan)
        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn, record_id=_receipt_id(first_plan)) == 0


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_postgres_plan_fails_closed_without_cluster_system_identity(
    case: _MigrationCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        real_execute = semantic_v5_migration._PostgresConnection.execute

        def deny_control_system(
            connection: Any,
            statement: str,
            params: tuple[object, ...] = (),
        ) -> Any:
            if "pg_control_system" in statement:
                raise PermissionError("injected pg_control_system denial")
            return real_execute(connection, statement, params)

        monkeypatch.setattr(
            semantic_v5_migration._PostgresConnection,
            "execute",
            deny_control_system,
        )
        with pytest.raises(case.error, match="stable PostgreSQL cluster"):
            case.plan(dsn)
        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn) == 0


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_postgres_plan_requires_the_runtime_advisory_lease(
    case: _MigrationCase,
) -> None:
    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        holder = _PostgresConnection(dsn)
        try:
            database, schema, _endpoint = semantic_v5_migration._postgres_identity(
                holder
            )
            acquired = holder.execute(
                "SELECT pg_try_advisory_lock(?) AS acquired",
                (_postgres_runtime_lock_key(database, schema),),
            ).fetchone()
            assert acquired == {"acquired": True}
            with pytest.raises(case.error) as lease_failure:
                case.plan(dsn)
            assert str(lease_failure.value) == (
                "PostgreSQL runtime store is already open"
            )
            assert database not in str(lease_failure.value)
            assert schema not in str(lease_failure.value)
        finally:
            holder.close()

        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn) == 0


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_postgres_failed_migration_rolls_back_plan_receipt(
    case: _MigrationCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _postgres_schema_dsn() as dsn:
        case.setup_source(dsn)
        plan = case.plan(dsn)
        real_insert_receipt = case.module._insert_postgres_migration_receipt

        def fail_after_receipt(*args: Any, **kwargs: Any) -> None:
            real_insert_receipt(*args, **kwargs)
            raise case.error("injected failure after migration receipt")

        monkeypatch.setattr(
            case.module,
            "_insert_postgres_migration_receipt",
            fail_after_receipt,
        )

        with pytest.raises(case.error, match="injected failure"):
            _apply(case, dsn, plan)

        assert _schema_version(dsn) == case.from_version
        assert _receipt_count(dsn, record_id=_receipt_id(plan)) == 0
