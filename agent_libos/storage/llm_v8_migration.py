"""Explicit, digest-bound RuntimeStore schema-v7 to schema-v8 migration."""

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
    _expected_postgres_catalog_sha256,
    _insert_postgres_migration_receipt,
    _migration_receipt_contract_sha256,
    _migration_receipt_record_id,
    _open_postgres_migration_connection,
    _postgres_catalog_sha256,
    _postgres_database_identity_sha256_from_parts,
    _postgres_identity,
    _postgres_lock_source_relations,
    _postgres_source_state_sha256,
    _product_version,
    _read_schema_marker_version,
    _require_exact_bool,
    _require_postgres_migration_receipt,
    _require_secure_regular_file,
    _snapshot_receipt_sha256,
    _sqlite_database_identity_sha256,
    _sqlite_logical_projection_sha256,
    _sqlite_source_catalog_sha256,
    _sqlite_path,
    _sqlite_snapshot,
    _validated_expected_plan_sha256,
    _validated_sqlite_backup_path,
)
from agent_libos.storage.sql import (
    SQLRuntimeStore,
    _V7_REQUIRED_COLUMNS,
    _V8_REQUIRED_COLUMNS,
)
from agent_libos.storage.sqlite import SQLiteStore
from agent_libos.storage.v8_schema_contract import (
    V8_INDEX_CONTRACTS,
    V8_STORAGE_COLUMN_CONTRACTS,
    V8_STORAGE_FOREIGN_KEYS,
    V8_STORAGE_KEY_CONSTRAINTS,
    V8_STORAGE_SQLITE_CHECKS,
    V8_TABLES,
)


MIGRATION_PLAN_SCHEMA_VERSION = 2
MIGRATION_FROM_SCHEMA_VERSION = 7
MIGRATION_TO_SCHEMA_VERSION = 8
MIGRATION_IMPLEMENTATION_VERSION = "v7-to-v8/1"


class StoreV8MigrationError(ValidationError):
    """The explicit schema-v8 migration could not be safely applied."""


_MIGRATION_STEPS = (
    "validate_canonical_v7",
    "acquire_offline_backend_lease",
    "create_host_private_llm_replay_turns",
    "create_host_private_llm_replay_heads",
    "verify_unchanged_v7_source_projection",
    "compare_and_swap_schema_marker_7_to_8",
    "validate_canonical_v8",
    "commit",
)


@dataclass(frozen=True, slots=True)
class StoreV8MigrationPlan:
    backend: str
    ddl_sha256: str
    database_identity_sha256: str
    source_catalog_sha256: str
    source_digest_kind: str
    source_digest_sha256: str
    snapshot_receipt_sha256: str
    receipt_contract_sha256: str
    migration_implementation_version: str
    product_version: str
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
            "database_identity_sha256": self.database_identity_sha256,
            "source_catalog_sha256": self.source_catalog_sha256,
            "source_digest_kind": self.source_digest_kind,
            "source_digest_sha256": self.source_digest_sha256,
            "snapshot_receipt_sha256": self.snapshot_receipt_sha256,
            "receipt_contract_sha256": self.receipt_contract_sha256,
            "migration_implementation_version": (
                self.migration_implementation_version
            ),
            "product_version": self.product_version,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True, slots=True)
class StoreV8MigrationResult:
    plan: StoreV8MigrationPlan
    applied: bool
    already_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "applied": self.applied,
            "already_applied": self.already_applied,
        }


def plan_store_v8_migration(
    target: str | Path,
    *,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV8MigrationPlan:
    """Perform a zero-write validation and return a deterministic v8 plan."""

    try:
        return _plan_store_v8_migration(
            target,
            sqlite_backup=sqlite_backup,
            postgres_snapshot_confirmed=postgres_snapshot_confirmed,
        )
    except StoreV5MigrationError as exc:
        raise StoreV8MigrationError(str(exc)) from exc


def _plan_store_v8_migration(
    target: str | Path,
    *,
    sqlite_backup: str | Path | None,
    postgres_snapshot_confirmed: bool,
) -> StoreV8MigrationPlan:
    _require_exact_bool(
        postgres_snapshot_confirmed,
        label="postgres_snapshot_confirmed",
    )
    backend = _backend_for_target(target)
    if backend == "sqlite":
        if sqlite_backup is None:
            raise StoreV8MigrationError(
                "SQLite schema-v8 planning requires an independent verified backup"
            )
        source_path = _sqlite_path(target, migration_label="schema-v8")
        database_identity_sha256 = _sqlite_database_identity_sha256(source_path)
        with _sqlite_snapshot(
            source_path, label="SQLite source", migration_label="schema-v8"
        ) as source:
            _require_canonical_v7(SQLiteStore, source)
            source_digest = _sqlite_logical_v7_sha256(source)
        backup_path = _validated_sqlite_backup_path(
            sqlite_backup,
            source_path=source_path,
        )
        with _sqlite_snapshot(
            backup_path, label="SQLite backup", migration_label="schema-v8"
        ) as backup:
            _require_canonical_v7(SQLiteStore, backup)
            backup_digest = _sqlite_logical_v7_sha256(backup)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV8MigrationError(
                "SQLite backup does not match the canonical v7 source store"
            )
        plan = _build_plan(
            backend,
            database_identity_sha256=database_identity_sha256,
            source_catalog_sha256=_sqlite_source_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            ),
            source_digest_kind="sqlite-logical-v7",
            source_digest_sha256=source_digest,
        )
    else:
        if sqlite_backup is not None:
            raise StoreV8MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        connection = _open_postgres_migration_connection(
            str(target),
            connection_factory=_PostgresConnection,
            error_type=StoreV8MigrationError,
            to_schema_version=MIGRATION_TO_SCHEMA_VERSION,
        )
        transaction_started = False
        try:
            database, schema, endpoint_sha256 = _postgres_identity(connection)
            pre_identity_sha256 = _postgres_database_identity_sha256_from_parts(
                database,
                schema,
                endpoint_sha256,
            )
            lease_key = _postgres_runtime_lock_key(database, schema)
            lease = connection.execute(
                "SELECT pg_try_advisory_lock(?) AS acquired",
                (lease_key,),
            ).fetchone()
            if not lease or not lease.get("acquired"):
                raise StoreV8MigrationError(
                    "PostgreSQL runtime store is already open"
                )
            connection.execute(
                "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            transaction_started = True
            source_tables = tuple(sorted(_V7_REQUIRED_COLUMNS))
            _postgres_lock_source_relations(
                connection,
                schema=schema,
                tables=source_tables,
            )
            tx_database, tx_schema, tx_endpoint_sha256 = _postgres_identity(
                connection
            )
            database_identity_sha256 = (
                _postgres_database_identity_sha256_from_parts(
                    tx_database,
                    tx_schema,
                    tx_endpoint_sha256,
                )
            )
            if (
                tx_database != database
                or tx_schema != schema
                or not hmac.compare_digest(
                    database_identity_sha256,
                    pre_identity_sha256,
                )
            ):
                raise StoreV8MigrationError(
                    "PostgreSQL migration identity changed before source capture"
                )
            _require_canonical_v7(PostgresStore, connection)
            source_catalog_sha256 = _postgres_catalog_sha256(connection)
            source_digest_sha256 = _postgres_source_state_sha256(
                connection,
                schema=schema,
                tables=source_tables,
                source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
                observed_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
            )
            plan = _build_plan(
                backend,
                database_identity_sha256=database_identity_sha256,
                source_catalog_sha256=source_catalog_sha256,
                source_digest_kind="postgres-relation-state-v7",
                source_digest_sha256=source_digest_sha256,
            )
            connection.rollback()
            transaction_started = False
        except BaseException:
            if transaction_started:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise
        finally:
            connection.close()
    return plan


def apply_store_v8_migration(
    target: str | Path,
    *,
    expected_plan_sha256: str,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV8MigrationResult:
    """Apply v7-to-v8 under the same offline lease used by runtime startup."""

    try:
        expected = _validated_expected_plan_sha256(expected_plan_sha256)
        _require_exact_bool(
            postgres_snapshot_confirmed,
            label="postgres_snapshot_confirmed",
        )
        backend = _backend_for_target(target)
        if backend == "sqlite":
            if sqlite_backup is None:
                raise StoreV8MigrationError(
                    "SQLite schema-v8 apply requires a verified sqlite_backup"
                )
            source_path = _sqlite_path(target, migration_label="schema-v8")
            return _apply_sqlite(
                source_path,
                backup_path=_validated_sqlite_backup_path(
                    sqlite_backup,
                    source_path=source_path,
                ),
                expected_plan_sha256=expected,
            )
        if sqlite_backup is not None:
            raise StoreV8MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        if not postgres_snapshot_confirmed:
            raise StoreV8MigrationError(
                "PostgreSQL schema-v8 apply requires explicit operator snapshot confirmation"
            )
        return _apply_postgres(str(target), expected_plan_sha256=expected)
    except StoreV5MigrationError as exc:
        raise StoreV8MigrationError(str(exc)) from exc


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
            for table, columns in sorted(V8_STORAGE_COLUMN_CONTRACTS.items())
        },
        "checks": {
            table: list(checks)
            for table, checks in sorted(V8_STORAGE_SQLITE_CHECKS.items())
        },
        "keys": {
            table: [[kind, list(columns)] for kind, columns in keys]
            for table, keys in sorted(V8_STORAGE_KEY_CONSTRAINTS.items())
        },
        "foreign_keys": {
            table: [list(binding) for binding in bindings]
            for table, bindings in sorted(V8_STORAGE_FOREIGN_KEYS.items())
        },
        "indexes": {
            name: [table, list(columns), unique, partial]
            for name, (table, columns, unique, partial) in sorted(
                V8_INDEX_CONTRACTS.items()
            )
        },
        "marker_cas": "7->8",
        "payload_columns": ["llm_replay_turns.payload_json"],
    }


def _build_plan(
    backend: str,
    *,
    database_identity_sha256: str,
    source_catalog_sha256: str,
    source_digest_kind: str,
    source_digest_sha256: str,
) -> StoreV8MigrationPlan:
    encoded_contract = json.dumps(
        {"backend": backend, "contract": _contract_projection()},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ddl_sha256 = hashlib.sha256(encoded_contract).hexdigest()
    product_version = _product_version()
    snapshot_receipt_sha256 = _snapshot_receipt_sha256(
        backend=backend,
        database_identity_sha256=database_identity_sha256,
        source_catalog_sha256=source_catalog_sha256,
        source_digest_kind=source_digest_kind,
        source_digest_sha256=source_digest_sha256,
        from_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
        to_schema_version=MIGRATION_TO_SCHEMA_VERSION,
    )
    receipt_contract_sha256 = _migration_receipt_contract_sha256(
        backend=backend,
        from_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
        to_schema_version=MIGRATION_TO_SCHEMA_VERSION,
    )
    body = {
        "schema_version": MIGRATION_PLAN_SCHEMA_VERSION,
        "backend": backend,
        "from_schema_version": MIGRATION_FROM_SCHEMA_VERSION,
        "to_schema_version": MIGRATION_TO_SCHEMA_VERSION,
        "steps": list(_MIGRATION_STEPS),
        "ddl_sha256": ddl_sha256,
        "database_identity_sha256": database_identity_sha256,
        "source_catalog_sha256": source_catalog_sha256,
        "source_digest_kind": source_digest_kind,
        "source_digest_sha256": source_digest_sha256,
        "snapshot_receipt_sha256": snapshot_receipt_sha256,
        "receipt_contract_sha256": receipt_contract_sha256,
        "migration_implementation_version": MIGRATION_IMPLEMENTATION_VERSION,
        "product_version": product_version,
    }
    encoded_plan = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return StoreV8MigrationPlan(
        backend=backend,
        ddl_sha256=ddl_sha256,
        database_identity_sha256=database_identity_sha256,
        source_catalog_sha256=source_catalog_sha256,
        source_digest_kind=source_digest_kind,
        source_digest_sha256=source_digest_sha256,
        snapshot_receipt_sha256=snapshot_receipt_sha256,
        receipt_contract_sha256=receipt_contract_sha256,
        migration_implementation_version=MIGRATION_IMPLEMENTATION_VERSION,
        product_version=product_version,
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


def _require_canonical_v7(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=7)
    backend._require_v7_schema_shape(connection)
    _require_sqlite_integrity(backend, connection, version=7)


def _require_canonical_v8(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=8)
    backend._require_v8_schema_shape(connection)
    _require_sqlite_integrity(backend, connection, version=8)


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


def _require_v8_migration_postcondition(connection: Any) -> None:
    for table in sorted(V8_TABLES):
        row = connection.execute(
            f'SELECT COUNT(*) AS count FROM "{table}"'
        ).fetchone()
        if row is None or int(row["count"]) != 0:
            raise StoreV8MigrationError(
                "canonical v8 store is not the exact result of the planned v7 migration"
            )


def _sqlite_logical_v7_sha256(connection: Any) -> str:
    return _sqlite_logical_projection_sha256(
        connection,
        required_columns=_V7_REQUIRED_COLUMNS,
        source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
    )


def _execute_v8_ddl(connection: Any) -> None:
    helper = object.__new__(SQLRuntimeStore)
    helper.conn = connection
    SQLRuntimeStore._create_v8_llm_replay_schema(helper)


def _apply_sqlite(
    path: Path,
    *,
    backup_path: Path,
    expected_plan_sha256: str,
) -> StoreV8MigrationResult:
    _require_secure_regular_file(path, label="SQLite source", mode_600=True)
    with _sqlite_snapshot(
        backup_path, label="SQLite backup", migration_label="schema-v8"
    ) as backup:
        _require_canonical_v7(SQLiteStore, backup)
        backup_digest = _sqlite_logical_v7_sha256(backup)
    with SQLiteStore._migration_apply_connection(
        path,
        error_type=StoreV8MigrationError,
        migration_label="schema-v8",
    ) as connection:
        database_identity_sha256 = _sqlite_database_identity_sha256(path)
        marker_version = _read_schema_marker_version(connection)
        if marker_version == MIGRATION_TO_SCHEMA_VERSION:
            plan = _build_plan(
                "sqlite",
                database_identity_sha256=database_identity_sha256,
                source_catalog_sha256=_sqlite_source_catalog_sha256(
                    MIGRATION_FROM_SCHEMA_VERSION
                ),
                source_digest_kind="sqlite-logical-v7",
                source_digest_sha256=backup_digest,
            )
            _require_expected_plan(plan, expected_plan_sha256)
            _require_canonical_v8(SQLiteStore, connection)
            _require_v8_migration_postcondition(connection)
            migrated_source_digest = _sqlite_logical_v7_sha256(connection)
            if not hmac.compare_digest(migrated_source_digest, backup_digest):
                raise StoreV8MigrationError(
                    "canonical v8 store does not match the planned v7 source snapshot"
                )
            return StoreV8MigrationResult(
                plan=plan,
                applied=False,
                already_applied=True,
            )
        _require_canonical_v7(SQLiteStore, connection)
        source_digest = _sqlite_logical_v7_sha256(connection)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV8MigrationError(
                "SQLite backup does not match the locked canonical v7 source store"
            )
        plan = _build_plan(
            "sqlite",
            database_identity_sha256=database_identity_sha256,
            source_catalog_sha256=_sqlite_source_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            ),
            source_digest_kind="sqlite-logical-v7",
            source_digest_sha256=source_digest,
        )
        _require_expected_plan(plan, expected_plan_sha256)
        _execute_v8_ddl(connection)
        migrated_source_digest = _sqlite_logical_v7_sha256(connection)
        if not hmac.compare_digest(migrated_source_digest, source_digest):
            raise StoreV8MigrationError(
                "SQLite schema-v8 DDL changed the locked source state"
            )
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = 8 "
            "WHERE singleton = 1 AND schema_version = 7"
        )
        if marker.rowcount != 1:
            raise StoreV8MigrationError(
                "schema marker compare-and-swap from v7 to v8 lost its race"
            )
        _require_canonical_v8(SQLiteStore, connection)
        _require_v8_migration_postcondition(connection)
        connection.commit()
        _require_canonical_v8(SQLiteStore, connection)
        _require_v8_migration_postcondition(connection)
    return StoreV8MigrationResult(plan=plan, applied=True)


def _apply_postgres(
    dsn: str,
    *,
    expected_plan_sha256: str,
) -> StoreV8MigrationResult:
    connection = _open_postgres_migration_connection(
        dsn,
        connection_factory=_PostgresConnection,
        error_type=StoreV8MigrationError,
        to_schema_version=MIGRATION_TO_SCHEMA_VERSION,
    )
    transaction_started = False
    try:
        database, schema, endpoint_sha256 = _postgres_identity(connection)
        pre_identity_sha256 = _postgres_database_identity_sha256_from_parts(
            database,
            schema,
            endpoint_sha256,
        )
        lease_key = _postgres_runtime_lock_key(database, schema)
        lease = connection.execute(
            "SELECT pg_try_advisory_lock(?) AS acquired", (lease_key,)
        ).fetchone()
        if not lease or not lease.get("acquired"):
            raise StoreV8MigrationError(
                "PostgreSQL runtime store is already open"
            )
        marker_version = _read_schema_marker_version(connection)
        if marker_version == MIGRATION_FROM_SCHEMA_VERSION:
            locked_tables = tuple(sorted(_V7_REQUIRED_COLUMNS))
        elif marker_version == MIGRATION_TO_SCHEMA_VERSION:
            locked_tables = tuple(sorted(_V8_REQUIRED_COLUMNS))
        else:
            _require_schema_marker(
                connection,
                expected=MIGRATION_FROM_SCHEMA_VERSION,
            )
            raise AssertionError("unreachable unsupported migration marker")
        connection.execute(
            "BEGIN ISOLATION LEVEL REPEATABLE READ READ WRITE"
        )
        transaction_started = True
        _postgres_lock_source_relations(
            connection,
            schema=schema,
            tables=locked_tables,
        )
        tx_database, tx_schema, tx_endpoint_sha256 = _postgres_identity(connection)
        database_identity_sha256 = _postgres_database_identity_sha256_from_parts(
            tx_database,
            tx_schema,
            tx_endpoint_sha256,
        )
        tx_marker_version = _read_schema_marker_version(connection)
        if (
            tx_database != database
            or tx_schema != schema
            or tx_marker_version != marker_version
            or not hmac.compare_digest(
                database_identity_sha256,
                pre_identity_sha256,
            )
        ):
            raise StoreV8MigrationError(
                "PostgreSQL migration identity or marker changed before source lock"
            )
        source_tables = tuple(sorted(_V7_REQUIRED_COLUMNS))
        if marker_version == MIGRATION_TO_SCHEMA_VERSION:
            _require_canonical_v8(PostgresStore, connection)
            source_digest_sha256 = _postgres_source_state_sha256(
                connection,
                schema=schema,
                tables=source_tables,
                source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
                observed_schema_version=MIGRATION_TO_SCHEMA_VERSION,
                excluded_audit_record_id=(
                    f"store-migration-v{MIGRATION_FROM_SCHEMA_VERSION}-to-"
                    f"v{MIGRATION_TO_SCHEMA_VERSION}:{expected_plan_sha256}"
                ),
            )
            source_catalog_sha256 = _expected_postgres_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            )
            plan = _build_plan(
                "postgres",
                database_identity_sha256=database_identity_sha256,
                source_catalog_sha256=source_catalog_sha256,
                source_digest_kind="postgres-relation-state-v7",
                source_digest_sha256=source_digest_sha256,
            )
            _require_expected_plan(plan, expected_plan_sha256)
            _require_postgres_migration_receipt(connection, plan)
            _require_v8_migration_postcondition(connection)
            connection.rollback()
            transaction_started = False
            return StoreV8MigrationResult(
                plan=plan,
                applied=False,
                already_applied=True,
            )
        _require_canonical_v7(PostgresStore, connection)
        source_catalog_sha256 = _postgres_catalog_sha256(connection)
        source_digest_sha256 = _postgres_source_state_sha256(
            connection,
            schema=schema,
            tables=source_tables,
            source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
            observed_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
        )
        plan = _build_plan(
            "postgres",
            database_identity_sha256=database_identity_sha256,
            source_catalog_sha256=source_catalog_sha256,
            source_digest_kind="postgres-relation-state-v7",
            source_digest_sha256=source_digest_sha256,
        )
        _require_expected_plan(plan, expected_plan_sha256)
        _execute_v8_ddl(connection)
        _insert_postgres_migration_receipt(connection, plan)
        migrated_source_digest_sha256 = _postgres_source_state_sha256(
            connection,
            schema=schema,
            tables=source_tables,
            source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
            observed_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
            excluded_audit_record_id=_migration_receipt_record_id(plan),
        )
        if not hmac.compare_digest(
            migrated_source_digest_sha256,
            source_digest_sha256,
        ):
            raise StoreV8MigrationError(
                "PostgreSQL schema-v8 DDL changed the locked source state"
            )
        _require_postgres_migration_receipt(connection, plan)
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = 8 "
            "WHERE singleton = 1 AND schema_version = 7"
        )
        if marker.rowcount != 1:
            raise StoreV8MigrationError(
                "schema marker compare-and-swap from v7 to v8 lost its race"
            )
        _require_canonical_v8(PostgresStore, connection)
        _require_v8_migration_postcondition(connection)
        _require_postgres_migration_receipt(connection, plan)
        connection.commit()
        transaction_started = False
        _require_canonical_v8(PostgresStore, connection)
        _require_v8_migration_postcondition(connection)
        _require_postgres_migration_receipt(connection, plan)
    except BaseException:
        if transaction_started:
            try:
                connection.rollback()
            except Exception:
                pass
        raise
    finally:
        connection.close()
    return StoreV8MigrationResult(plan=plan, applied=True)


def _require_expected_plan(
    plan: StoreV8MigrationPlan,
    expected_plan_sha256: str,
) -> None:
    if not hmac.compare_digest(plan.plan_sha256, expected_plan_sha256):
        raise StoreV8MigrationError(
            "schema-v8 migration plan digest does not match expected_plan_sha256"
        )


__all__ = [
    "MIGRATION_FROM_SCHEMA_VERSION",
    "MIGRATION_PLAN_SCHEMA_VERSION",
    "MIGRATION_TO_SCHEMA_VERSION",
    "StoreV8MigrationError",
    "StoreV8MigrationPlan",
    "StoreV8MigrationResult",
    "apply_store_v8_migration",
    "plan_store_v8_migration",
]
