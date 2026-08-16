from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import unquote, urlsplit

from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.engine import split_sql_script
from agent_libos.storage.postgres import (
    PostgresStore,
    _PostgresConnection,
    _postgres_runtime_lock_key,
)
from agent_libos.storage.postgres_schema_contract import (
    capture_postgres_catalog,
    expected_postgres_catalog,
)
from agent_libos.storage.sql import _V4_REQUIRED_COLUMNS, _V5_REQUIRED_COLUMNS
from agent_libos.storage.sqlite import SQLiteStore, _sqlite_catalog_sha256
from agent_libos.utils.ids import utc_now


MIGRATION_PLAN_SCHEMA_VERSION = 2
MIGRATION_FROM_SCHEMA_VERSION = 4
MIGRATION_TO_SCHEMA_VERSION = 5
MIGRATION_IMPLEMENTATION_VERSION = "v4-to-v5/3"


class StoreV5MigrationError(ValidationError):
    """The explicit schema-v5 migration could not be safely planned or applied."""


# This intentionally has no import path from runtime startup. Keep its table
# constraints and named indexes aligned with SQLRuntimeStore's fresh-v5 DDL;
# the post-migration runtime validator rejects drift before commit.
_MIGRATION_DDL = """
ALTER TABLE human_requests
  ADD COLUMN revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0);

CREATE TABLE semantic_assessment_jobs (
  job_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
  assessment_id TEXT COLLATE BINARY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN (
    'queued', 'claimed', 'succeeded', 'failed', 'egress_blocked',
    'provider_outcome_unknown', 'cancelled', 'expired'
  )),
  domain TEXT NOT NULL,
  pid TEXT,
  request_id TEXT,
  operation_id TEXT,
  effect_id TEXT,
  revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (
    attempt_count >= 0 AND attempt_count <= 1
  ),
  lease_owner_id TEXT,
  lease_id TEXT,
  lease_expires_at TEXT,
  bindings_json TEXT NOT NULL,
  projection_json TEXT NOT NULL,
  projection_sha256 TEXT NOT NULL,
  projection_retention TEXT NOT NULL CHECK (
    projection_retention IN ('redacted', 'hash_only')
  ),
  projection_expires_at TEXT,
  error_code TEXT,
  created_at TEXT COLLATE BINARY NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE(assessment_id),
  CHECK (
    (lease_owner_id IS NULL AND lease_id IS NULL AND lease_expires_at IS NULL)
    OR
    (lease_owner_id IS NOT NULL AND lease_id IS NOT NULL
     AND lease_expires_at IS NOT NULL)
  ),
  CHECK ((status = 'claimed') = (lease_id IS NOT NULL)),
  CHECK (
    (status IN ('queued', 'claimed') AND completed_at IS NULL)
    OR
    (status NOT IN ('queued', 'claimed') AND completed_at IS NOT NULL)
  ),
  CHECK (
    status IN ('queued', 'claimed')
    OR (
      projection_retention = 'hash_only'
      AND projection_json = '{}'
      AND projection_expires_at IS NULL
    )
  )
);

CREATE INDEX idx_semantic_jobs_status_created
  ON semantic_assessment_jobs(
    status, created_at COLLATE BINARY, job_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_jobs_pid_created
  ON semantic_assessment_jobs(
    pid, created_at COLLATE BINARY, job_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_jobs_request_created
  ON semantic_assessment_jobs(
    request_id, created_at COLLATE BINARY, job_id COLLATE BINARY
  );

CREATE TABLE semantic_assessments (
  assessment_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
  job_id TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  domain TEXT NOT NULL,
  action_id TEXT NOT NULL,
  tenant_bucket_sha256 TEXT,
  pid TEXT,
  request_id TEXT,
  operation_id TEXT,
  effect_id TEXT,
  shadow_outcome TEXT,
  ood INTEGER NOT NULL CHECK (ood IN (0, 1)),
  record_json TEXT NOT NULL,
  created_at TEXT COLLATE BINARY NOT NULL,
  completed_at TEXT
);

CREATE INDEX idx_semantic_assessments_created
  ON semantic_assessments(
    created_at COLLATE BINARY, assessment_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_assessments_pid_created
  ON semantic_assessments(
    pid, created_at COLLATE BINARY, assessment_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_assessments_request_created
  ON semantic_assessments(
    request_id, created_at COLLATE BINARY, assessment_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_assessments_operation_created
  ON semantic_assessments(
    operation_id, created_at COLLATE BINARY, assessment_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_assessments_filter_created
  ON semantic_assessments(
    kind, status, domain, created_at COLLATE BINARY,
    assessment_id COLLATE BINARY
  );
CREATE INDEX idx_semantic_assessments_action_tenant_created
  ON semantic_assessments(
    action_id, tenant_bucket_sha256, created_at COLLATE BINARY,
    assessment_id COLLATE BINARY
  );
"""


_MIGRATION_STATEMENTS = tuple(split_sql_script(_MIGRATION_DDL))
_MARKER_CAS_PLAN_STATEMENT = (
    "UPDATE runtime_schema SET schema_version = 5 "
    "WHERE singleton = 1 AND schema_version = 4"
)

_MIGRATION_STEPS = (
    "validate_canonical_v4",
    "acquire_offline_backend_lease",
    "add_human_request_revision",
    "create_semantic_assessment_jobs",
    "create_semantic_assessments",
    "compare_and_swap_schema_marker_4_to_5",
    "validate_canonical_v5",
    "commit",
)

_SEMANTIC_INDEXES: dict[str, tuple[str, tuple[str, ...], bool, bool]] = {
    "idx_semantic_jobs_status_created": (
        "semantic_assessment_jobs",
        ("status", "created_at", "job_id"),
        False,
        False,
    ),
    "idx_semantic_jobs_pid_created": (
        "semantic_assessment_jobs",
        ("pid", "created_at", "job_id"),
        False,
        False,
    ),
    "idx_semantic_jobs_request_created": (
        "semantic_assessment_jobs",
        ("request_id", "created_at", "job_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_created": (
        "semantic_assessments",
        ("created_at", "assessment_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_pid_created": (
        "semantic_assessments",
        ("pid", "created_at", "assessment_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_request_created": (
        "semantic_assessments",
        ("request_id", "created_at", "assessment_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_operation_created": (
        "semantic_assessments",
        ("operation_id", "created_at", "assessment_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_filter_created": (
        "semantic_assessments",
        ("kind", "status", "domain", "created_at", "assessment_id"),
        False,
        False,
    ),
    "idx_semantic_assessments_action_tenant_created": (
        "semantic_assessments",
        ("action_id", "tenant_bucket_sha256", "created_at", "assessment_id"),
        False,
        False,
    ),
}

_SEMANTIC_UNIQUE_CONSTRAINTS: dict[str, frozenset[tuple[str, ...]]] = {
    "semantic_assessment_jobs": frozenset(
        {("job_id",), ("assessment_id",)}
    ),
    "semantic_assessments": frozenset({("assessment_id",), ("job_id",)}),
}

_POSTGRES_URI_SCHEMES = frozenset({"postgres", "postgresql"})
_LIBPQ_DSN = re.compile(
    r"(?:^|\s)(?:dbname|host|hostaddr|options|password|port|service|sslmode|user)\s*=",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MIGRATION_RECEIPT_SCHEMA_VERSION = 1
_MIGRATION_RECEIPT_ACTOR = "host:agent-libos-offline-migrator"
_MIGRATION_RECEIPT_ACTION = "store.schema_migration_committed"


@dataclass(frozen=True)
class StoreV5MigrationPlan:
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


@dataclass(frozen=True)
class StoreV5MigrationResult:
    plan: StoreV5MigrationPlan
    applied: bool
    already_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "applied": self.applied,
            "already_applied": self.already_applied,
        }


def plan_store_v5_migration(
    target: str | Path,
    *,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV5MigrationPlan:
    """Validate a canonical v4 store and return its deterministic v5 plan.

    SQLite planning copies the database family to a private temporary path and
    permits recovery only on that copy.  It never opens the selected database,
    creates a lease sidecar, or otherwise writes beside the target.
    """

    _require_exact_bool(
        postgres_snapshot_confirmed,
        label="postgres_snapshot_confirmed",
    )
    backend = _backend_for_target(target)
    if backend == "sqlite":
        source_path = _sqlite_path(target)
        database_identity_sha256 = _sqlite_database_identity_sha256(source_path)
        with _sqlite_snapshot(source_path, label="SQLite source") as source:
            _require_canonical_v4(SQLiteStore, source)
            source_digest = _sqlite_logical_v4_sha256(source)
        if sqlite_backup is not None:
            backup_path = _validated_sqlite_backup_path(
                sqlite_backup,
                source_path=source_path,
            )
            with _sqlite_snapshot(backup_path, label="SQLite backup") as backup:
                _require_canonical_v4(SQLiteStore, backup)
                backup_digest = _sqlite_logical_v4_sha256(backup)
            if not hmac.compare_digest(source_digest, backup_digest):
                raise StoreV5MigrationError(
                    "SQLite backup does not match the canonical v4 source store"
                )
        plan = _build_plan(
            backend,
            database_identity_sha256=database_identity_sha256,
            source_catalog_sha256=_sqlite_source_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            ),
            source_digest_kind="sqlite-logical-v4",
            source_digest_sha256=source_digest,
        )
    else:
        if sqlite_backup is not None:
            raise StoreV5MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        connection = _open_postgres_migration_connection(
            str(target),
            connection_factory=_PostgresConnection,
            error_type=StoreV5MigrationError,
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
                raise StoreV5MigrationError(
                    "PostgreSQL runtime store is already open"
                )
            connection.execute(
                "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            transaction_started = True
            source_tables = tuple(sorted(_V4_REQUIRED_COLUMNS))
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
                raise StoreV5MigrationError(
                    "PostgreSQL migration identity changed before source capture"
                )
            _require_canonical_v4(PostgresStore, connection)
            source_catalog_sha256 = _postgres_catalog_sha256(connection)
            source_digest = _postgres_source_state_sha256(
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
                source_digest_kind="postgres-relation-state-v4",
                source_digest_sha256=source_digest,
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


def apply_store_v5_migration(
    target: str | Path,
    *,
    expected_plan_sha256: str,
    sqlite_backup: str | Path | None = None,
    postgres_snapshot_confirmed: bool = False,
) -> StoreV5MigrationResult:
    """Apply the explicit schema-v5 migration under an offline backend lease."""

    expected = _validated_expected_plan_sha256(expected_plan_sha256)
    _require_exact_bool(
        postgres_snapshot_confirmed,
        label="postgres_snapshot_confirmed",
    )
    backend = _backend_for_target(target)
    if backend == "sqlite":
        if sqlite_backup is None:
            raise StoreV5MigrationError(
                "SQLite schema-v5 apply requires a verified sqlite_backup"
            )
        return _apply_sqlite(
            _sqlite_path(target),
            backup_path=_validated_sqlite_backup_path(
                sqlite_backup,
                source_path=_sqlite_path(target),
            ),
            expected_plan_sha256=expected,
        )
    if sqlite_backup is not None:
        raise StoreV5MigrationError(
            "sqlite_backup is valid only for a SQLite migration"
        )
    if not postgres_snapshot_confirmed:
        raise StoreV5MigrationError(
            "PostgreSQL schema-v5 apply requires explicit operator snapshot confirmation"
        )
    return _apply_postgres(
        str(target),
        expected_plan_sha256=expected,
    )


def _build_plan(
    backend: str,
    *,
    database_identity_sha256: str,
    source_catalog_sha256: str,
    source_digest_kind: str,
    source_digest_sha256: str,
) -> StoreV5MigrationPlan:
    canonical_ddl = "\n;\n".join(
        [
            *(
                _statement_for_backend(statement, backend=backend).strip()
                for statement in _MIGRATION_STATEMENTS
            ),
            _MARKER_CAS_PLAN_STATEMENT,
        ]
    )
    ddl_sha256 = hashlib.sha256(canonical_ddl.encode("utf-8")).hexdigest()
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
    plan_body = {
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
    encoded = json.dumps(
        plan_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return StoreV5MigrationPlan(
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
        plan_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _product_version() -> str:
    # Resolve lazily: importing agent_libos while its package __init__ imports
    # storage would otherwise create a cycle.  A migration plan is generated
    # only after package initialization has completed.
    import agent_libos

    selected = getattr(agent_libos, "__version__", None)
    if not isinstance(selected, str) or not selected:
        raise StoreV5MigrationError(
            "Agent libOS product version is unavailable for migration planning"
        )
    return selected


def _open_postgres_migration_connection(
    dsn: str,
    *,
    connection_factory: Callable[[str], Any],
    error_type: type[ValidationError],
    to_schema_version: int,
) -> Any:
    """Open a migration session without retaining provider error details."""

    try:
        return connection_factory(dsn)
    except Exception:
        pass
    error = error_type(
        "unable to open PostgreSQL "
        f"schema-v{to_schema_version} migration target"
    )
    error.__cause__ = None
    error.__context__ = None
    raise error from None


def _sqlite_database_identity_sha256(path: Path) -> str:
    _require_secure_regular_file(
        path,
        label="SQLite source",
        mode_600=False,
    )
    selected = os.stat(path, follow_symlinks=False)
    return _canonical_json_sha256(
        {
            "backend": "sqlite",
            "resolved_path": os.path.normcase(str(path.resolve())),
            "device": int(selected.st_dev),
            "inode": int(selected.st_ino),
        }
    )


def _postgres_identity(connection: Any) -> tuple[str, str, str]:
    try:
        row = connection.execute(
            "SELECT current_database() AS database_name, "
            "current_schema() AS schema_name, "
            "(SELECT oid FROM pg_catalog.pg_database "
            "WHERE datname = current_database()) AS database_oid, "
            "(SELECT oid FROM pg_catalog.pg_namespace "
            "WHERE nspname = current_schema()) AS schema_oid, "
            "COALESCE(inet_server_addr()::text, '') AS server_address, "
            "current_setting('port') AS server_port, "
            "has_function_privilege(current_user, "
            "'pg_catalog.pg_control_system()', 'EXECUTE') "
            "AS control_system_allowed, "
            "CASE WHEN has_function_privilege(current_user, "
            "'pg_catalog.pg_control_system()', 'EXECUTE') THEN "
            "(SELECT system_identifier::text "
            "FROM pg_catalog.pg_control_system()) ELSE NULL END "
            "AS system_identifier"
        ).fetchone()
    except Exception as exc:
        raise StoreV5MigrationError(
            "unable to resolve stable PostgreSQL cluster/database/schema identity"
        ) from exc
    (
        database,
        schema,
        system_identifier,
        database_oid,
        schema_oid,
        server_address,
        server_port,
    ) = _validated_postgres_identity_row(row)
    driver_connection = getattr(connection, "_conn", None)
    driver_info = getattr(driver_connection, "info", None)
    driver_host = str(getattr(driver_info, "host", "") or "")
    driver_hostaddr = str(getattr(driver_info, "hostaddr", "") or "")
    driver_port = str(getattr(driver_info, "port", "") or "")
    endpoint_sha256 = _canonical_json_sha256(
        {
            "driver_host_sha256": hashlib.sha256(
                driver_host.encode("utf-8")
            ).hexdigest(),
            "driver_hostaddr_sha256": hashlib.sha256(
                driver_hostaddr.encode("utf-8")
            ).hexdigest(),
            "driver_port": driver_port,
            "server_address_sha256": hashlib.sha256(
                server_address.encode("utf-8")
            ).hexdigest(),
            "server_port": server_port,
            "system_identifier_sha256": hashlib.sha256(
                system_identifier.encode("ascii")
            ).hexdigest(),
            "database_oid": database_oid,
            "schema_oid": schema_oid,
        }
    )
    return database, schema, endpoint_sha256


def _validated_postgres_identity_row(
    row: Mapping[str, Any] | None,
) -> tuple[str, str, str, int, int, str, str]:
    if row is None:
        raise StoreV5MigrationError(
            "unable to resolve stable PostgreSQL cluster/database/schema identity"
        )
    database = str(row.get("database_name") or "")
    schema = str(row.get("schema_name") or "")
    system_identifier = str(row.get("system_identifier") or "")
    database_oid = int(row.get("database_oid") or 0)
    schema_oid = int(row.get("schema_oid") or 0)
    valid = all(
        (
            database,
            schema,
            row.get("control_system_allowed") is True,
            system_identifier,
            database_oid > 0,
            schema_oid > 0,
        )
    )
    if not valid:
        raise StoreV5MigrationError(
            "unable to resolve stable PostgreSQL cluster/database/schema identity"
        )
    return (
        database,
        schema,
        system_identifier,
        database_oid,
        schema_oid,
        str(row.get("server_address") or ""),
        str(row.get("server_port") or ""),
    )


def _postgres_database_identity_sha256(connection: Any) -> str:
    database, schema, endpoint_sha256 = _postgres_identity(connection)
    return _postgres_database_identity_sha256_from_parts(
        database,
        schema,
        endpoint_sha256,
    )


def _postgres_database_identity_sha256_from_parts(
    database: str,
    schema: str,
    endpoint_sha256: str,
) -> str:
    return _canonical_json_sha256(
        {
            "backend": "postgres",
            "database": database,
            "schema": schema,
            "endpoint_sha256": endpoint_sha256,
        }
    )


def _postgres_catalog_sha256(connection: Any) -> str:
    return _canonical_json_sha256(capture_postgres_catalog(connection))


def _expected_postgres_catalog_sha256(store_version: int) -> str:
    return _canonical_json_sha256(expected_postgres_catalog(store_version))


def _sqlite_source_catalog_sha256(store_version: int) -> str:
    return _sqlite_catalog_sha256(
        SQLiteStore._canonical_full_schema_catalog(store_version)
    )


def _postgres_lock_source_relations(
    connection: Any,
    *,
    schema: str,
    tables: tuple[str, ...],
) -> None:
    if not tables or len(set(tables)) != len(tables):
        raise StoreV5MigrationError(
            "PostgreSQL migration source relation set is invalid"
        )
    driver_connection = getattr(connection, "_conn", None)
    if driver_connection is None:
        raise StoreV5MigrationError(
            "PostgreSQL migration cannot access its native driver session"
        )
    try:
        from psycopg import sql as pg_sql

        qualified = pg_sql.SQL(", ").join(
            pg_sql.Identifier(schema, table) for table in sorted(tables)
        )
        lock_cursor = driver_connection.execute(
            pg_sql.SQL(
                "LOCK TABLE {} IN ACCESS EXCLUSIVE MODE NOWAIT"
            ).format(qualified)
        )
        lock_cursor.close()
    except Exception as exc:
        raise StoreV5MigrationError(
            "unable to lock the complete PostgreSQL migration source catalog"
        ) from exc


def _postgres_source_state_sha256(
    connection: Any,
    *,
    schema: str,
    tables: tuple[str, ...],
    source_schema_version: int,
    observed_schema_version: int,
    excluded_audit_record_id: str | None = None,
) -> str:
    """Hash relation identity and visible MVCC tuple identity without payloads."""

    if not tables or len(set(tables)) != len(tables):
        raise StoreV5MigrationError(
            "PostgreSQL migration source relation set is invalid"
        )
    if observed_schema_version not in {
        source_schema_version,
        source_schema_version + 1,
    }:
        raise StoreV5MigrationError(
            "PostgreSQL migration schema marker normalization is invalid"
        )
    digest = hashlib.sha256()
    digest.update(_framed_bytes(b"algorithm", b"postgres-relation-state-v1"))
    digest.update(
        _framed_bytes(
            b"source-schema-version",
            str(source_schema_version).encode("ascii"),
        )
    )
    driver_connection = getattr(connection, "_conn", None)
    if driver_connection is None:
        raise StoreV5MigrationError(
            "PostgreSQL migration cannot access its native driver session"
        )
    for table in sorted(tables):
        relation_oid = _postgres_source_relation_oid(
            connection,
            schema=schema,
            table=table,
        )
        digest.update(_framed_bytes(b"table", table.encode("utf-8")))
        digest.update(
            _framed_bytes(b"relation-oid", str(relation_oid).encode("ascii"))
        )
        if table == "runtime_schema":
            _postgres_hash_normalized_schema_marker(
                digest,
                driver_connection=driver_connection,
                schema=schema,
                table=table,
                source_schema_version=source_schema_version,
                observed_schema_version=observed_schema_version,
            )
            continue
        _postgres_hash_visible_tuple_state(
            digest,
            connection,
            schema=schema,
            table=table,
            excluded_audit_record_id=(
                excluded_audit_record_id
                if table == "audit_records"
                else None
            ),
        )
    return digest.hexdigest()


def _postgres_source_relation_oid(
    connection: Any,
    *,
    schema: str,
    table: str,
) -> int:
    try:
        relation_cursor = connection.execute(
            "SELECT relation.oid AS relation_oid "
            "FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace "
            "ON namespace.oid = relation.relnamespace "
            "WHERE namespace.nspname = ? AND relation.relname = ? "
            "AND relation.relkind = 'r'",
            (schema, table),
        )
        try:
            relation = relation_cursor.fetchone()
        finally:
            relation_cursor.close()
    except Exception as exc:
        raise StoreV5MigrationError(
            "unable to read the locked PostgreSQL source relation identity"
        ) from exc
    relation_oid = int(relation.get("relation_oid") or 0) if relation else 0
    if relation_oid <= 0:
        raise StoreV5MigrationError(
            "PostgreSQL migration source relation identity is unavailable"
        )
    return relation_oid


def _postgres_hash_normalized_schema_marker(
    digest: Any,
    *,
    driver_connection: Any,
    schema: str,
    table: str,
    source_schema_version: int,
    observed_schema_version: int,
) -> None:
    from psycopg import sql as pg_sql

    marker_cursor = None
    try:
        marker_cursor = driver_connection.execute(
            pg_sql.SQL(
                "SELECT singleton, schema_version FROM {} ORDER BY singleton"
            ).format(pg_sql.Identifier(schema, table))
        )
        rows = [dict(row) for row in marker_cursor]
    except Exception as exc:
        raise StoreV5MigrationError(
            "unable to read the locked PostgreSQL schema marker"
        ) from exc
    finally:
        if marker_cursor is not None:
            marker_cursor.close()
    if rows != [{"singleton": 1, "schema_version": observed_schema_version}]:
        raise StoreV5MigrationError(
            "PostgreSQL migration schema marker cannot be normalized"
        )
    normalized = (
        f"singleton=1;schema_version={source_schema_version}"
    ).encode("ascii")
    digest.update(_framed_bytes(b"normalized-marker", normalized))
    digest.update(_framed_bytes(b"rows", b"1"))


def _postgres_hash_visible_tuple_state(
    digest: Any,
    connection: Any,
    *,
    schema: str,
    table: str,
    excluded_audit_record_id: str | None,
) -> None:
    row_count = 0
    for row in _postgres_stream_rows(
        connection,
        schema=schema,
        table=table,
        excluded_audit_record_id=excluded_audit_record_id,
    ):
        tuple_id = str(row.get("tuple_id") or "")
        xmin = str(row.get("xmin") or "")
        if not tuple_id or not xmin:
            raise StoreV5MigrationError(
                "PostgreSQL migration source tuple identity is unavailable"
            )
        row_count += 1
        digest.update(_framed_bytes(b"ctid", tuple_id.encode("ascii")))
        digest.update(_framed_bytes(b"xmin", xmin.encode("ascii")))
    digest.update(_framed_bytes(b"rows", str(row_count).encode("ascii")))


def _postgres_stream_rows(
    connection: Any,
    *,
    schema: str,
    table: str,
    excluded_audit_record_id: str | None,
) -> Iterator[Mapping[str, Any]]:
    """Yield a bounded-memory result from a psycopg server-side cursor."""

    driver_connection = getattr(connection, "_conn", None)
    if driver_connection is None:
        raise StoreV5MigrationError(
            "PostgreSQL migration cannot create a streaming source cursor"
        )
    cursor_name = "agent_libos_migration_source_rows_" + hashlib.sha256(
        f"{schema}\0{table}".encode("utf-8")
    ).hexdigest()[:16]
    cursor = driver_connection.cursor(name=cursor_name)
    try:
        from psycopg import sql as pg_sql

        cursor.itersize = 1024
        statement = pg_sql.SQL(
            "SELECT ctid::text AS tuple_id, xmin::text AS xmin "
            "FROM ONLY {}"
        ).format(pg_sql.Identifier(schema, table))
        params: tuple[object, ...] = ()
        if excluded_audit_record_id is not None:
            statement += pg_sql.SQL(" WHERE record_id <> %s")
            params = (excluded_audit_record_id,)
        statement += pg_sql.SQL(" ORDER BY ctid")
        cursor.execute(
            statement,
            params,
        )
        for row in cursor:
            yield row
    except Exception as exc:
        raise StoreV5MigrationError(
            "unable to stream the locked PostgreSQL migration source state"
        ) from exc
    finally:
        cursor.close()


def _migration_receipt_record_id(plan: Any) -> str:
    return (
        f"store-migration-v{plan.from_schema_version}-to-v{plan.to_schema_version}:"
        f"{plan.plan_sha256}"
    )


def _migration_receipt_decision(plan: Any) -> dict[str, Any]:
    binding = {
        "schema_version": _MIGRATION_RECEIPT_SCHEMA_VERSION,
        "kind": "agent_libos_store_migration_receipt",
        "outcome": "committed",
        "plan": plan.to_dict(),
    }
    return {
        **binding,
        "binding_sha256": _canonical_json_sha256(binding),
    }


def _migration_receipt_contract_sha256(
    *,
    backend: str,
    from_schema_version: int,
    to_schema_version: int,
) -> str:
    if backend == "sqlite":
        contract: dict[str, Any] = {
            "schema_version": 1,
            "backend": "sqlite",
            "mode": "no-database-receipt",
        }
    elif backend == "postgres":
        contract = {
            "schema_version": _MIGRATION_RECEIPT_SCHEMA_VERSION,
            "backend": "postgres",
            "mode": "exact-plan-bound-audit-record",
            "from_schema_version": from_schema_version,
            "to_schema_version": to_schema_version,
            "record_id": "store-migration-vFROM-to-vTO:PLAN_SHA256",
            "actor": _MIGRATION_RECEIPT_ACTOR,
            "action": _MIGRATION_RECEIPT_ACTION,
            "target": "runtime-store-schema-vTO",
            "reference_arrays": "empty",
            "correlation_id": "PLAN_SHA256",
            "parent_record_id": None,
            "gui_snapshot_visible": 1,
            "timestamp": "utc-iso8601-offset-zero",
            "decision": {
                "schema_version": _MIGRATION_RECEIPT_SCHEMA_VERSION,
                "kind": "agent_libos_store_migration_receipt",
                "outcome": "committed",
                "plan": "FULL_SCHEMA_V2_PLAN",
                "binding_sha256": "SHA256_OF_DECISION_WITHOUT_BINDING",
            },
            "source_reconstruction": "exclude-only-exact-current-receipt-id",
        }
    else:
        raise StoreV5MigrationError(
            f"unsupported migration backend: {backend}"
        )
    return _canonical_json_sha256(contract)


def _insert_postgres_migration_receipt(connection: Any, plan: Any) -> None:
    inserted = connection.execute(
        "INSERT INTO audit_records "
        "(record_id, timestamp, actor, action, target, input_refs_json, "
        "output_refs_json, capability_refs_json, decision_json, correlation_id, "
        "parent_record_id, gui_snapshot_visible) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _migration_receipt_record_id(plan),
            utc_now(),
            _MIGRATION_RECEIPT_ACTOR,
            _MIGRATION_RECEIPT_ACTION,
            f"runtime-store-schema-v{plan.to_schema_version}",
            "[]",
            "[]",
            "[]",
            json.dumps(
                _migration_receipt_decision(plan),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            plan.plan_sha256,
            None,
            1,
        ),
    )
    if inserted.rowcount != 1:
        raise StoreV5MigrationError(
            "PostgreSQL migration receipt was not appended"
        )


def _require_postgres_migration_receipt(connection: Any, plan: Any) -> None:
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT record_id, timestamp, actor, action, target, input_refs_json, "
            "output_refs_json, capability_refs_json, decision_json, correlation_id, "
            "parent_record_id, gui_snapshot_visible FROM audit_records "
            "WHERE record_id = ?",
            (_migration_receipt_record_id(plan),),
        )
    ]
    if len(rows) != 1:
        raise StoreV5MigrationError(
            "exact PostgreSQL migration receipt is missing"
        )
    row = rows[0]
    timestamp = str(row.get("timestamp") or "")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise StoreV5MigrationError(
            "exact PostgreSQL migration receipt is malformed"
        ) from exc
    expected_decision = json.dumps(
        _migration_receipt_decision(plan),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = {
        "record_id": _migration_receipt_record_id(plan),
        "actor": _MIGRATION_RECEIPT_ACTOR,
        "action": _MIGRATION_RECEIPT_ACTION,
        "target": f"runtime-store-schema-v{plan.to_schema_version}",
        "input_refs_json": "[]",
        "output_refs_json": "[]",
        "capability_refs_json": "[]",
        "decision_json": expected_decision,
        "correlation_id": plan.plan_sha256,
        "parent_record_id": None,
        "gui_snapshot_visible": 1,
    }
    observed = {key: row.get(key) for key in expected}
    if (
        observed != expected
        or parsed_timestamp.tzinfo is None
        or parsed_timestamp.utcoffset() != timedelta(0)
    ):
        raise StoreV5MigrationError(
            "exact PostgreSQL migration receipt is malformed or tampered"
        )


def _snapshot_receipt_sha256(
    *,
    backend: str,
    database_identity_sha256: str,
    source_catalog_sha256: str,
    source_digest_kind: str,
    source_digest_sha256: str,
    from_schema_version: int,
    to_schema_version: int,
) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": 1,
            "backend": backend,
            "database_identity_sha256": database_identity_sha256,
            "source_catalog_sha256": source_catalog_sha256,
            "source_digest_kind": source_digest_kind,
            "source_digest_sha256": source_digest_sha256,
            "from_schema_version": from_schema_version,
            "to_schema_version": to_schema_version,
            "capture": (
                "private-recovered-copy"
                if backend == "sqlite"
                else "repeatable-read-locked-relation-state"
            ),
        }
    )


def _backend_for_target(target: str | Path) -> str:
    text = str(target)
    parsed = urlsplit(text)
    if parsed.scheme.lower() in _POSTGRES_URI_SCHEMES or _LIBPQ_DSN.search(text):
        return "postgres"
    return "sqlite"


def _sqlite_path(
    target: str | Path,
    *,
    migration_label: str = "schema-v5",
) -> Path:
    text = _sqlite_target_text(target, migration_label=migration_label)
    path = Path(text).resolve()
    _require_secure_regular_file(path, label="SQLite source", mode_600=False)
    return path


def _sqlite_target_text(
    target: str | Path,
    *,
    windows: bool | None = None,
    migration_label: str = "schema-v5",
) -> str:
    """Decode a SQLite target without constructing a platform-specific Path."""

    text = str(target)
    if text in {"", ":memory:"}:
        raise StoreV5MigrationError(
            f"{migration_label} migration requires a file-backed SQLite store"
        )
    selected_windows = os.name == "nt" if windows is None else windows
    if not isinstance(selected_windows, bool):
        raise StoreV5MigrationError("SQLite target platform selector is invalid")
    parsed = urlsplit(text)
    if parsed.scheme.lower() == "sqlite":
        if parsed.netloc and parsed.path:
            text = unquote(f"//{parsed.netloc}{parsed.path}")
        elif parsed.path:
            text = unquote(parsed.path)
            if not parsed.netloc and text.startswith("//"):
                text = f"/{text.lstrip('/')}"
            if text.startswith("/") and len(text) > 2 and text[2] == ":":
                text = text[1:]
        else:
            raise StoreV5MigrationError(
                f"{migration_label} migration requires a file-backed SQLite store"
            )
    elif parsed.scheme and not (
        selected_windows
        and len(parsed.scheme) == 1
        and text[1:3] in {":\\", ":/"}
    ):
        raise StoreV5MigrationError(
            f"unsupported {migration_label} migration target scheme: {parsed.scheme}"
        )
    return text


def _validated_sqlite_backup_path(
    backup: str | Path,
    *,
    source_path: Path,
) -> Path:
    backup_path = Path(backup).resolve()
    _require_secure_regular_file(
        backup_path,
        label="SQLite backup",
        mode_600=True,
    )
    source_stat = os.stat(source_path, follow_symlinks=False)
    backup_stat = os.stat(backup_path, follow_symlinks=False)
    if (
        source_stat.st_dev == backup_stat.st_dev
        and source_stat.st_ino == backup_stat.st_ino
    ):
        raise StoreV5MigrationError(
            "SQLite backup must be an independent file, not the source store"
        )
    for suffix in ("-journal", "-wal", "-shm"):
        if Path(f"{backup_path}{suffix}").exists():
            raise StoreV5MigrationError(
                "SQLite backup must be a self-contained quiesced backup without sidecars"
            )
    return backup_path


def _require_secure_regular_file(
    path: Path,
    *,
    label: str,
    mode_600: bool,
) -> None:
    try:
        file_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise StoreV5MigrationError(f"{label} is not readable: {path}") from exc
    reparse_attribute = int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
    )
    path_attributes = int(getattr(file_stat, "st_file_attributes", 0))
    if not stat.S_ISREG(file_stat.st_mode) or path_attributes & reparse_attribute:
        raise StoreV5MigrationError(f"{label} must be a regular file: {path}")
    if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
        raise StoreV5MigrationError(
            f"{label} must be owned by the current user: {path}"
        )
    if file_stat.st_nlink != 1:
        raise StoreV5MigrationError(
            f"{label} must have exactly one hard link: {path}"
        )
    if os.name != "nt" and mode_600 and stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise StoreV5MigrationError(
            f"{label} must have mode 0600 before migration apply: {path}; "
            f"run chmod 600 {path} and retain the verified independent backup"
        )


def _sqlite_snapshot(
    path: Path,
    *,
    label: str,
    migration_label: str = "schema-v5",
) -> Any:
    _require_secure_regular_file(path, label=label, mode_600=False)
    return SQLiteStore._migration_snapshot_connection(
        path,
        label=label,
        error_type=StoreV5MigrationError,
        migration_label=migration_label,
    )


def _require_canonical_v4(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=MIGRATION_FROM_SCHEMA_VERSION)
    try:
        backend._require_v4_schema_shape(connection)
    except UnsupportedStoreVersion:
        raise
    except Exception as exc:
        raise UnsupportedStoreVersion(
            "unable to validate canonical Agent libOS schema v4"
        ) from exc
    if backend is SQLiteStore:
        row = connection.execute("PRAGMA quick_check").fetchone()
        value = row[0] if row is not None else None
        if value != "ok":
            raise UnsupportedStoreVersion(
                f"SQLite schema-v4 integrity check failed: {value!r}"
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


def _read_schema_marker_version(connection: Any) -> int:
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
    if (
        len(selected) != 1
        or selected[0].get("singleton") != 1
        or type(selected[0].get("schema_version")) is not int
    ):
        raise UnsupportedStoreVersion(
            "unsupported Agent libOS store schema marker: "
            f"expected exactly one singleton=1 row, found {selected!r}"
        )
    return int(selected[0]["schema_version"])


def _require_canonical_v5(backend: type[Any], connection: Any) -> None:
    _require_schema_marker(connection, expected=MIGRATION_TO_SCHEMA_VERSION)
    # Reuse the runtime's complete v5 contract for relation types, every old
    # index/counter invariant, keyset collations, and the new named indexes.
    # The checks below add migration-specific exactness for the two new UNIQUE
    # constraints and the ALTERed revision column.
    backend._require_v5_schema_shape(connection)
    shapes = backend._probe_index_shapes(
        connection,
        {"semantic_assessment_jobs", "semantic_assessments"},
    )
    _require_no_extra_semantic_indexes(shapes)
    _require_semantic_unique_constraints(shapes)
    if backend is SQLiteStore:
        _require_sqlite_revision_column(connection)


def _require_no_extra_semantic_indexes(
    shapes: Mapping[str, Mapping[str, Any]],
) -> None:
    declared = {
        name
        for name, shape in shapes.items()
        if shape.get("origin") == "declared"
    }
    extra_declared = sorted(declared - set(_SEMANTIC_INDEXES))
    if extra_declared:
        raise UnsupportedStoreVersion(
            "unsupported Agent libOS store schema v5 semantic indexes: "
            f"{{'<indexes>': {{'extra': {extra_declared!r}}}}}"
        )


def _require_semantic_unique_constraints(
    shapes: Mapping[str, Mapping[str, Any]],
) -> None:
    problems: dict[str, Any] = {}
    for table, required in _SEMANTIC_UNIQUE_CONSTRAINTS.items():
        observed = {
            tuple(shape.get("columns", ()))
            for shape in shapes.values()
            if shape.get("table") == table
            and bool(shape.get("unique"))
            and shape.get("origin") == "constraint"
        }
        missing = sorted(required - observed)
        if missing:
            problems[f"{table}.unique"] = {"missing": missing}
    if problems:
        raise UnsupportedStoreVersion(
            f"unsupported Agent libOS store schema v5 semantic indexes: {problems}"
        )


def _require_sqlite_revision_column(connection: Any) -> None:
    revision_rows = {
        str(row["name"]): dict(row)
        for row in connection.execute("PRAGMA table_info(human_requests)")
    }
    revision = revision_rows.get("revision")
    if (
        revision is None
        or str(revision["type"]).upper() != "BIGINT"
        or int(revision["notnull"]) != 1
        or str(revision["dflt_value"]) not in {"0", "(0)"}
    ):
        raise UnsupportedStoreVersion(
            "unsupported Agent libOS schema v5 human request revision column"
        )


def _require_v5_migration_postcondition(connection: Any) -> None:
    invalid_revision = connection.execute(
        "SELECT COUNT(*) AS count FROM human_requests WHERE revision != 0"
    ).fetchone()
    if invalid_revision is None or int(invalid_revision["count"]) != 0:
        raise StoreV5MigrationError(
            "canonical v5 store is not the exact result of the planned v4 migration"
        )
    for table in ("semantic_assessment_jobs", "semantic_assessments"):
        row = connection.execute(
            f'SELECT COUNT(*) AS count FROM "{table}"'
        ).fetchone()
        if row is None or int(row["count"]) != 0:
            raise StoreV5MigrationError(
                "canonical v5 store is not the exact result of the planned v4 migration"
            )


def _sqlite_logical_v4_sha256(connection: Any) -> str:
    return _sqlite_logical_projection_sha256(
        connection,
        required_columns=_V4_REQUIRED_COLUMNS,
        source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
    )


def _sqlite_logical_projection_sha256(
    connection: Any,
    *,
    required_columns: Mapping[str, frozenset[str]],
    source_schema_version: int,
) -> str:
    """Hash a source-version projection, including when read after migration."""

    digest = hashlib.sha256()
    for table in sorted(required_columns):
        actual_columns = [
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        required = required_columns[table]
        columns = [column for column in actual_columns if column in required]
        if set(columns) != set(required):
            raise StoreV5MigrationError(
                f"SQLite logical source projection is missing columns for {table}"
            )
        digest.update(_framed_bytes(b"table", table.encode("utf-8")))
        for column in columns:
            digest.update(_framed_bytes(b"column", column.encode("utf-8")))
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        order = ", ".join(
            f'"{column}" COLLATE BINARY' for column in columns
        )
        rows = connection.execute(
            f'SELECT {quoted_columns} FROM "{table}" ORDER BY {order}'
        )
        row_count = 0
        for row in rows:
            row_count += 1
            digest.update(b"row")
            for column, value in zip(columns, row, strict=True):
                if table == "runtime_schema" and column == "schema_version":
                    value = source_schema_version
                digest.update(_canonical_sqlite_value(value))
        digest.update(_framed_bytes(b"rows", str(row_count).encode("ascii")))
    return digest.hexdigest()


def _canonical_sqlite_value(value: Any) -> bytes:
    if value is None:
        return b"n\0\0\0\0\0\0\0\0\0"
    if isinstance(value, int):
        return _framed_bytes(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        return _framed_bytes(b"f", value.hex().encode("ascii"))
    if isinstance(value, str):
        return _framed_bytes(b"s", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _framed_bytes(b"b", value)
    raise StoreV5MigrationError(
        f"unsupported SQLite value type in backup verification: {type(value).__name__}"
    )


def _framed_bytes(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, byteorder="big") + payload


def _sqlite_apply_connection(path: Path) -> Any:
    return SQLiteStore._migration_apply_connection(
        path,
        error_type=StoreV5MigrationError,
    )


def _apply_sqlite(
    path: Path,
    *,
    backup_path: Path,
    expected_plan_sha256: str,
) -> StoreV5MigrationResult:
    _require_secure_regular_file(
        path,
        label="SQLite source",
        mode_600=True,
    )
    with _sqlite_snapshot(backup_path, label="SQLite backup") as backup:
        _require_canonical_v4(SQLiteStore, backup)
        backup_digest = _sqlite_logical_v4_sha256(backup)

    with _sqlite_apply_connection(path) as connection:
        database_identity_sha256 = _sqlite_database_identity_sha256(path)
        marker_version = _read_schema_marker_version(connection)
        if marker_version == MIGRATION_TO_SCHEMA_VERSION:
            plan = _build_plan(
                "sqlite",
                database_identity_sha256=database_identity_sha256,
                source_catalog_sha256=_sqlite_source_catalog_sha256(
                    MIGRATION_FROM_SCHEMA_VERSION
                ),
                source_digest_kind="sqlite-logical-v4",
                source_digest_sha256=backup_digest,
            )
            _require_expected_plan(plan, expected_plan_sha256)
            _require_canonical_v5(SQLiteStore, connection)
            _require_v5_migration_postcondition(connection)
            migrated_source_digest = _sqlite_logical_v4_sha256(connection)
            if not hmac.compare_digest(migrated_source_digest, backup_digest):
                raise StoreV5MigrationError(
                    "canonical v5 store does not match the planned v4 source snapshot"
                )
            return StoreV5MigrationResult(
                plan=plan,
                applied=False,
                already_applied=True,
            )
        _require_canonical_v4(SQLiteStore, connection)
        source_digest = _sqlite_logical_v4_sha256(connection)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV5MigrationError(
                "SQLite backup does not match the locked canonical v4 source store"
            )
        plan = _build_plan(
            "sqlite",
            database_identity_sha256=database_identity_sha256,
            source_catalog_sha256=_sqlite_source_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            ),
            source_digest_kind="sqlite-logical-v4",
            source_digest_sha256=source_digest,
        )
        _require_expected_plan(plan, expected_plan_sha256)
        _execute_migration_statements(connection, backend="sqlite")
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = ? "
            "WHERE singleton = 1 AND schema_version = ?",
            (MIGRATION_TO_SCHEMA_VERSION, MIGRATION_FROM_SCHEMA_VERSION),
        )
        if marker.rowcount != 1:
            raise StoreV5MigrationError(
                "schema marker compare-and-swap from v4 to v5 lost its race"
            )
        _require_canonical_v5(SQLiteStore, connection)
        _require_v5_migration_postcondition(connection)
        connection.commit()
        _require_canonical_v5(SQLiteStore, connection)
        _require_v5_migration_postcondition(connection)
    return StoreV5MigrationResult(plan=plan, applied=True)


def _apply_postgres(
    dsn: str,
    *,
    expected_plan_sha256: str,
) -> StoreV5MigrationResult:
    connection = _open_postgres_migration_connection(
        dsn,
        connection_factory=_PostgresConnection,
        error_type=StoreV5MigrationError,
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
            raise StoreV5MigrationError(
                "PostgreSQL runtime store is already open"
            )
        marker_version = _read_schema_marker_version(connection)
        if marker_version == MIGRATION_FROM_SCHEMA_VERSION:
            locked_tables = tuple(sorted(_V4_REQUIRED_COLUMNS))
        elif marker_version == MIGRATION_TO_SCHEMA_VERSION:
            locked_tables = tuple(sorted(_V5_REQUIRED_COLUMNS))
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
            raise StoreV5MigrationError(
                "PostgreSQL migration identity or marker changed before source lock"
            )
        source_tables = tuple(sorted(_V4_REQUIRED_COLUMNS))
        if marker_version == MIGRATION_TO_SCHEMA_VERSION:
            _require_canonical_v5(PostgresStore, connection)
            receipt_record_id = (
                f"store-migration-v{MIGRATION_FROM_SCHEMA_VERSION}-to-"
                f"v{MIGRATION_TO_SCHEMA_VERSION}:{expected_plan_sha256}"
            )
            source_digest_sha256 = _postgres_source_state_sha256(
                connection,
                schema=schema,
                tables=source_tables,
                source_schema_version=MIGRATION_FROM_SCHEMA_VERSION,
                observed_schema_version=MIGRATION_TO_SCHEMA_VERSION,
                excluded_audit_record_id=receipt_record_id,
            )
            source_catalog_sha256 = _expected_postgres_catalog_sha256(
                MIGRATION_FROM_SCHEMA_VERSION
            )
            plan = _build_plan(
                "postgres",
                database_identity_sha256=database_identity_sha256,
                source_catalog_sha256=source_catalog_sha256,
                source_digest_kind="postgres-relation-state-v4",
                source_digest_sha256=source_digest_sha256,
            )
            _require_expected_plan(plan, expected_plan_sha256)
            _require_postgres_migration_receipt(connection, plan)
            _require_v5_migration_postcondition(connection)
            connection.rollback()
            transaction_started = False
            return StoreV5MigrationResult(
                plan=plan,
                applied=False,
                already_applied=True,
            )
        _require_canonical_v4(PostgresStore, connection)
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
            source_digest_kind="postgres-relation-state-v4",
            source_digest_sha256=source_digest_sha256,
        )
        _require_expected_plan(plan, expected_plan_sha256)
        _execute_migration_statements(connection, backend="postgres")
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
            raise StoreV5MigrationError(
                "PostgreSQL schema-v5 DDL changed the locked source state"
            )
        _require_postgres_migration_receipt(connection, plan)
        marker = connection.execute(
            "UPDATE runtime_schema SET schema_version = ? "
            "WHERE singleton = 1 AND schema_version = ?",
            (MIGRATION_TO_SCHEMA_VERSION, MIGRATION_FROM_SCHEMA_VERSION),
        )
        if marker.rowcount != 1:
            raise StoreV5MigrationError(
                "schema marker compare-and-swap from v4 to v5 lost its race"
            )
        _require_canonical_v5(PostgresStore, connection)
        _require_v5_migration_postcondition(connection)
        _require_postgres_migration_receipt(connection, plan)
        connection.commit()
        transaction_started = False
        _require_canonical_v5(PostgresStore, connection)
        _require_v5_migration_postcondition(connection)
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
    return StoreV5MigrationResult(plan=plan, applied=True)


def _execute_migration_statements(connection: Any, *, backend: str) -> None:
    for statement in _MIGRATION_STATEMENTS:
        connection.execute(_statement_for_backend(statement, backend=backend))


def _statement_for_backend(statement: str, *, backend: str) -> str:
    if backend == "sqlite":
        return statement
    if backend == "postgres":
        return statement.replace("COLLATE BINARY", 'COLLATE "C"')
    raise StoreV5MigrationError(f"unsupported migration backend: {backend}")


def _validated_expected_plan_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StoreV5MigrationError(
            "expected_plan_sha256 must be a lowercase SHA-256 digest"
        )
    return value


def _require_expected_plan(
    plan: StoreV5MigrationPlan,
    expected_plan_sha256: str,
) -> None:
    if not hmac.compare_digest(plan.plan_sha256, expected_plan_sha256):
        raise StoreV5MigrationError(
            "schema-v5 migration plan digest does not match expected_plan_sha256"
        )


def _require_exact_bool(value: bool, *, label: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")


__all__ = [
    "MIGRATION_FROM_SCHEMA_VERSION",
    "MIGRATION_PLAN_SCHEMA_VERSION",
    "MIGRATION_TO_SCHEMA_VERSION",
    "StoreV5MigrationError",
    "StoreV5MigrationPlan",
    "StoreV5MigrationResult",
    "apply_store_v5_migration",
    "plan_store_v5_migration",
]
