"""Explicit, digest-bound RuntimeStore schema-v6 to schema-v7 migration."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.postgres import (
    PostgresStore,
    _PostgresConnection,
    _postgres_runtime_lock_key,
)
from agent_libos.storage.semantic_v5_migration import (
    StoreV5MigrationError,
    _backend_for_target,
    _canonical_sqlite_value,
    _framed_bytes,
    _require_exact_bool,
    _require_secure_regular_file,
    _sqlite_path,
    _sqlite_snapshot,
    _validated_expected_plan_sha256,
    _validated_sqlite_backup_path,
)
from agent_libos.storage.sql import SQLRuntimeStore, _V6_REQUIRED_COLUMNS
from agent_libos.storage.sqlite import SQLiteStore
from agent_libos.storage.v7_schema_contract import (
    V7_INDEX_CONTRACTS,
    V7_STORAGE_COLUMN_CONTRACTS,
    V7_STORAGE_FOREIGN_KEYS,
    V7_STORAGE_KEY_CONSTRAINTS,
    V7_STORAGE_SQLITE_CHECKS,
)


MIGRATION_PLAN_SCHEMA_VERSION = 1
MIGRATION_FROM_SCHEMA_VERSION = 6
MIGRATION_TO_SCHEMA_VERSION = 7


class StoreV7MigrationError(ValidationError):
    """The explicit schema-v7 migration could not be safely applied."""


_MIGRATION_STEPS = (
    "validate_canonical_v6",
    "acquire_offline_backend_lease",
    "create_payload_free_mcp_continuation_state",
    "create_payload_free_mcp_remote_task_state",
    "create_payload_free_mcp_subscription_state",
    "create_non_secret_mcp_auth_metadata",
    "compare_and_swap_schema_marker_6_to_7",
    "validate_canonical_v7",
    "commit",
)


@dataclass(frozen=True, slots=True)
class StoreV7MigrationPlan:
    backend: str
    ddl_sha256: str
    plan_sha256: str

    @property
    def schema_version(self) -> int:
        return MIGRATION_PLAN_SCHEMA_VERSION

    @property
    def from_schema_version(self) -> int:
        return MIGRATION_FROM_SCHEMA_VERSION

    @property
    def to_schema_version(self) -> int:
        return MIGRATION_TO_SCHEMA_VERSION

    @property
    def steps(self) -> tuple[str, ...]:
        return _MIGRATION_STEPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "from_schema_version": self.from_schema_version,
            "to_schema_version": self.to_schema_version,
            "steps": list(self.steps),
            "ddl_sha256": self.ddl_sha256,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class StoreV7MigrationResult:
    plan: StoreV7MigrationPlan
    applied: bool
    already_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "applied": self.applied,
            "already_applied": self.already_applied,
        }


def plan_store_v7_migration(
    target: str | Path,
    *,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV7MigrationPlan:
    """Perform a zero-write validation and return a deterministic v7 plan."""

    try:
        return _plan_store_v7_migration(
            target,
            sqlite_backup=sqlite_backup,
            postgres_snapshot_confirmed=postgres_snapshot_confirmed,
        )
    except StoreV5MigrationError as exc:
        raise StoreV7MigrationError(str(exc)) from exc


def _plan_store_v7_migration(
    target: str | Path,
    *,
    sqlite_backup: str | Path | None,
    postgres_snapshot_confirmed: bool,
) -> StoreV7MigrationPlan:
    _require_exact_bool(
        postgres_snapshot_confirmed,
        label="postgres_snapshot_confirmed",
    )
    backend = _backend_for_target(target)
    if backend == "sqlite":
        if sqlite_backup is None:
            raise StoreV7MigrationError(
                "SQLite schema-v7 planning requires an independent verified backup"
            )
        source_path = _sqlite_path(target, migration_label="schema-v7")
        with _sqlite_snapshot(
            source_path, label="SQLite source", migration_label="schema-v7"
        ) as source:
            _require_canonical_v6(SQLiteStore, source)
            source_digest = _sqlite_logical_v6_sha256(source)
        backup_path = _validated_sqlite_backup_path(
            sqlite_backup,
            source_path=source_path,
        )
        with _sqlite_snapshot(
            backup_path, label="SQLite backup", migration_label="schema-v7"
        ) as backup:
            _require_canonical_v6(SQLiteStore, backup)
            backup_digest = _sqlite_logical_v6_sha256(backup)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV7MigrationError(
                "SQLite backup does not match the canonical v6 source store"
            )
    else:
        if sqlite_backup is not None:
            raise StoreV7MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        connection = _PostgresConnection(str(target))
        try:
            _require_canonical_v6(PostgresStore, connection)
        finally:
            connection.close()
    return _build_plan(backend)


def apply_store_v7_migration(
    target: str | Path,
    *,
    expected_plan_sha256: str,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV7MigrationResult:
    """Apply v6-to-v7 under the same offline lease used by runtime startup."""

    try:
        expected = _validated_expected_plan_sha256(expected_plan_sha256)
        _require_exact_bool(
            postgres_snapshot_confirmed,
            label="postgres_snapshot_confirmed",
        )
        backend = _backend_for_target(target)
        if backend == "sqlite":
            if sqlite_backup is None:
                raise StoreV7MigrationError(
                    "SQLite schema-v7 apply requires a verified sqlite_backup"
                )
            source_path = _sqlite_path(target, migration_label="schema-v7")
            return _apply_sqlite(
                source_path,
                backup_path=_validated_sqlite_backup_path(
                    sqlite_backup,
                    source_path=source_path,
                ),
                expected_plan_sha256=expected,
            )
        if sqlite_backup is not None:
            raise StoreV7MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        if not postgres_snapshot_confirmed:
            raise StoreV7MigrationError(
                "PostgreSQL schema-v7 apply requires explicit operator snapshot confirmation"
            )
        return _apply_postgres(str(target), expected_plan_sha256=expected)
    except StoreV5MigrationError as exc:
        raise StoreV7MigrationError(str(exc)) from exc


def _contract_projection() -> dict[str, Any]:
    return {
        "tables": {
            table: [
                {
                    "name": name,
                    "type": contract.sql_type,
                    "nullable": contract.nullable,
                    "default": contract.default,
                    "primary_key_position": contract.primary_key_position,
                    "keyset_collation": contract.keyset_collation,
                }
                for name, contract in columns
            ]
            for table, columns in sorted(V7_STORAGE_COLUMN_CONTRACTS.items())
        },
        "checks": {
            table: list(checks)
            for table, checks in sorted(V7_STORAGE_SQLITE_CHECKS.items())
        },
        "keys": {
            table: [[kind, list(columns)] for kind, columns in keys]
            for table, keys in sorted(V7_STORAGE_KEY_CONSTRAINTS.items())
        },
        "foreign_keys": {
            table: [list(binding) for binding in bindings]
            for table, bindings in sorted(V7_STORAGE_FOREIGN_KEYS.items())
        },
        "indexes": {
            name: [table, list(columns), unique, partial]
            for name, (table, columns, unique, partial) in sorted(
                V7_INDEX_CONTRACTS.items()
            )
        },
        "marker_cas": "6->7",
        "payload_columns": [],
    }


def _build_plan(backend: str) -> StoreV7MigrationPlan:
    encoded_contract = json.dumps(
        {"backend": backend, "contract": _contract_projection()},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ddl_sha256 = hashlib.sha256(encoded_contract).hexdigest()
    body = {
        "schema_version": MIGRATION_PLAN_SCHEMA_VERSION,
        "backend": backend,
        "from_schema_version": MIGRATION_FROM_SCHEMA_VERSION,
        "to_schema_version": MIGRATION_TO_SCHEMA_VERSION,
        "steps": list(_MIGRATION_STEPS),
        "ddl_sha256": ddl_sha256,
    }
    encoded_plan = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return StoreV7MigrationPlan(
        backend=backend,
        ddl_sha256=ddl_sha256,
        plan_sha256=hashlib.sha256(encoded_plan).hexdigest(),
    )


def _require_schema_marker(connection: Any, *, expected: int) -> None:
    try:
        rows = list(
            connection.execute(
                "SELECT singleton, schema_version FROM runtime_schema ORDER BY singleton"
            )
        )
    except Exception as exc:
        raise UnsupportedStoreVersion(
            "Agent libOS schema marker is missing or unreadable"
        ) from exc
    selected = [dict(row) for row in rows]
    if selected != [{"singleton": 1, "schema_version": expected}]:
        raise UnsupportedStoreVersion(
            "unsupported Agent libOS store schema marker: "
            f"expected exactly singleton=1/version={expected}, found {selected!r}"
        )


def _require_canonical_v6(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=6)
    backend._require_v6_schema_shape(connection)
    _require_sqlite_integrity(backend, connection, version=6)


def _require_canonical_v7(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=7)
    backend._require_v7_schema_shape(connection)
    _require_sqlite_integrity(backend, connection, version=7)


def _require_sqlite_integrity(
    backend: type[Any],
    connection: Any,
    *,
    version: int,
) -> None:
    if backend is not SQLiteStore:
        return
    row = connection.execute("PRAGMA quick_check").fetchone()
    if row is None or row[0] != "ok":
        raise UnsupportedStoreVersion(
            f"SQLite schema-v{version} integrity check failed: "
            f"{row[0] if row else None!r}"
        )


def _sqlite_logical_v6_sha256(connection: Any) -> str:
    digest = hashlib.sha256()
    for table in sorted(_V6_REQUIRED_COLUMNS):
        columns = [
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        digest.update(_framed_bytes(b"table", table.encode("utf-8")))
        for column in columns:
            digest.update(_framed_bytes(b"column", column.encode("utf-8")))
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        order = ", ".join(f'"{column}" COLLATE BINARY' for column in columns)
        row_count = 0
        for row in connection.execute(
            f'SELECT {quoted_columns} FROM "{table}" ORDER BY {order}'
        ):
            row_count += 1
            digest.update(b"row")
            for value in row:
                digest.update(_canonical_sqlite_value(value))
        digest.update(_framed_bytes(b"rows", str(row_count).encode("ascii")))
    return digest.hexdigest()


def _execute_v7_ddl(connection: Any) -> None:
    helper = object.__new__(SQLRuntimeStore)
    helper.conn = connection
    SQLRuntimeStore._create_v7_mcp_schema(helper)


def _apply_sqlite(
    path: Path,
    *,
    backup_path: Path,
    expected_plan_sha256: str,
) -> StoreV7MigrationResult:
    _require_secure_regular_file(path, label="SQLite source", mode_600=True)
    with _sqlite_snapshot(
        backup_path, label="SQLite backup", migration_label="schema-v7"
    ) as backup:
        _require_canonical_v6(SQLiteStore, backup)
        backup_digest = _sqlite_logical_v6_sha256(backup)
    plan = _build_plan("sqlite")
    _require_expected_plan(plan, expected_plan_sha256)
    with SQLiteStore._migration_apply_connection(
        path,
        error_type=StoreV7MigrationError,
        migration_label="schema-v7",
    ) as connection:
        _require_canonical_v6(SQLiteStore, connection)
        source_digest = _sqlite_logical_v6_sha256(connection)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV7MigrationError(
                "SQLite backup does not match the locked canonical v6 source store"
            )
        _execute_v7_ddl(connection)
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = 7 "
            "WHERE singleton = 1 AND schema_version = 6"
        )
        if marker.rowcount != 1:
            raise StoreV7MigrationError(
                "schema marker compare-and-swap from v6 to v7 lost its race"
            )
        _require_canonical_v7(SQLiteStore, connection)
        connection.commit()
        _require_canonical_v7(SQLiteStore, connection)
    return StoreV7MigrationResult(plan=plan, applied=True)


def _apply_postgres(
    dsn: str,
    *,
    expected_plan_sha256: str,
) -> StoreV7MigrationResult:
    plan = _build_plan("postgres")
    _require_expected_plan(plan, expected_plan_sha256)
    connection = _PostgresConnection(dsn)
    transaction_started = False
    try:
        identity = connection.execute(
            "SELECT current_database() AS database_name, current_schema() AS schema_name"
        ).fetchone()
        database = str(identity.get("database_name") or "") if identity else ""
        schema = str(identity.get("schema_name") or "") if identity else ""
        if not database or not schema:
            raise StoreV7MigrationError(
                "unable to resolve PostgreSQL database/schema for migration lease"
            )
        lease_key = _postgres_runtime_lock_key(database, schema)
        lease = connection.execute(
            "SELECT pg_try_advisory_lock(?) AS acquired", (lease_key,)
        ).fetchone()
        if not lease or not lease.get("acquired"):
            raise StoreV7MigrationError(
                f"runtime store is already open: postgres:{database}/{schema}"
            )
        connection.execute("BEGIN")
        transaction_started = True
        _require_canonical_v6(PostgresStore, connection)
        _execute_v7_ddl(connection)
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = 7 "
            "WHERE singleton = 1 AND schema_version = 6"
        )
        if marker.rowcount != 1:
            raise StoreV7MigrationError(
                "schema marker compare-and-swap from v6 to v7 lost its race"
            )
        _require_canonical_v7(PostgresStore, connection)
        connection.commit()
        transaction_started = False
        _require_canonical_v7(PostgresStore, connection)
    except BaseException:
        if transaction_started:
            try:
                connection.rollback()
            except Exception:
                pass
        raise
    finally:
        connection.close()
    return StoreV7MigrationResult(plan=plan, applied=True)


def _require_expected_plan(
    plan: StoreV7MigrationPlan,
    expected_plan_sha256: str,
) -> None:
    if not hmac.compare_digest(plan.plan_sha256, expected_plan_sha256):
        raise StoreV7MigrationError(
            "schema-v7 migration plan digest does not match expected_plan_sha256"
        )


__all__ = [
    "MIGRATION_FROM_SCHEMA_VERSION",
    "MIGRATION_PLAN_SCHEMA_VERSION",
    "MIGRATION_TO_SCHEMA_VERSION",
    "StoreV7MigrationError",
    "StoreV7MigrationPlan",
    "StoreV7MigrationResult",
    "apply_store_v7_migration",
    "plan_store_v7_migration",
]
