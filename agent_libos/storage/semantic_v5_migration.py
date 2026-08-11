from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.engine import split_sql_script
from agent_libos.storage.postgres import (
    PostgresStore,
    _PostgresConnection,
    _postgres_runtime_lock_key,
)
from agent_libos.storage.sql import _V4_REQUIRED_COLUMNS
from agent_libos.storage.sqlite import SQLiteStore


MIGRATION_PLAN_SCHEMA_VERSION = 1
MIGRATION_FROM_SCHEMA_VERSION = 4
MIGRATION_TO_SCHEMA_VERSION = 5


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


@dataclass(frozen=True)
class StoreV5MigrationPlan:
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
    else:
        if sqlite_backup is not None:
            raise StoreV5MigrationError(
                "sqlite_backup is valid only for a SQLite migration"
            )
        connection = _PostgresConnection(str(target))
        try:
            _require_canonical_v4(PostgresStore, connection)
        finally:
            connection.close()
    return _build_plan(backend)


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


def _build_plan(backend: str) -> StoreV5MigrationPlan:
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
    plan_body = {
        "schema_version": MIGRATION_PLAN_SCHEMA_VERSION,
        "backend": backend,
        "from_schema_version": MIGRATION_FROM_SCHEMA_VERSION,
        "to_schema_version": MIGRATION_TO_SCHEMA_VERSION,
        "steps": list(_MIGRATION_STEPS),
        "ddl_sha256": ddl_sha256,
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
        plan_sha256=hashlib.sha256(encoded).hexdigest(),
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


def _sqlite_logical_v4_sha256(connection: Any) -> str:
    digest = hashlib.sha256()
    for table in sorted(_V4_REQUIRED_COLUMNS):
        columns = [
            str(row["name"])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
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
            for value in row:
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

    plan = _build_plan("sqlite")
    _require_expected_plan(plan, expected_plan_sha256)
    with _sqlite_apply_connection(path) as connection:
        _require_canonical_v4(SQLiteStore, connection)
        source_digest = _sqlite_logical_v4_sha256(connection)
        if not hmac.compare_digest(source_digest, backup_digest):
            raise StoreV5MigrationError(
                "SQLite backup does not match the locked canonical v4 source store"
            )
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
        connection.commit()
        _require_canonical_v5(SQLiteStore, connection)
    return StoreV5MigrationResult(plan=plan, applied=True)


def _apply_postgres(
    dsn: str,
    *,
    expected_plan_sha256: str,
) -> StoreV5MigrationResult:
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
            raise StoreV5MigrationError(
                "unable to resolve PostgreSQL database/schema for migration lease"
            )
        lease_key = _postgres_runtime_lock_key(database, schema)
        lease = connection.execute(
            "SELECT pg_try_advisory_lock(?) AS acquired",
            (lease_key,),
        ).fetchone()
        if not lease or not lease.get("acquired"):
            raise StoreV5MigrationError(
                f"runtime store is already open: postgres:{database}/{schema}"
            )
        connection.execute("BEGIN")
        transaction_started = True
        _require_canonical_v4(PostgresStore, connection)
        _execute_migration_statements(connection, backend="postgres")
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
        connection.commit()
        transaction_started = False
        _require_canonical_v5(PostgresStore, connection)
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
