from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from typing import Any, Mapping

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.engine import split_sql_script
from agent_libos.storage.sql import SQLRuntimeStore, _V3_KEYSET_TEXT_COLUMNS


def _postgres_runtime_lock_key(database: str, schema: str) -> int:
    """Return a stable signed bigint key scoped to one database/schema pair."""

    digest = hashlib.blake2b(digest_size=8, person=b"AgentLibOS")
    digest.update(database.encode("utf-8"))
    digest.update(b"\0")
    digest.update(schema.encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big", signed=True)


class _PostgresDialect:
    def prepare(self, sql: str, *, with_params: bool = False) -> str:
        text = sql.strip()
        table_match = _PRAGMA_TABLE_INFO.match(text)
        if table_match:
            table = table_match.group("table")
            return (
                "SELECT column_name AS name "
                "FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = "
                f"'{table}'"
            )
        if _PRAGMA_INDEX_LIST.match(text) or _PRAGMA_INDEX_INFO.match(text):
            return "SELECT NULL::text AS name WHERE false"

        was_insert_or_ignore = bool(
            re.search(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", sql, re.IGNORECASE)
        )
        was_insert_or_replace_skill_trust = bool(
            re.search(r"\bINSERT\s+OR\s+REPLACE\s+INTO\s+skill_trust\b", sql, re.IGNORECASE)
        )
        transformed = sql
        # SQLite's default/BINARY path ordering and PostgreSQL's database
        # locale can disagree. Shared prefix-range queries and their indexes use
        # an explicit bytewise collation on both backends.
        transformed = transformed.replace("COLLATE BINARY", 'COLLATE "C"')
        transformed = transformed.replace("INSERT OR IGNORE INTO", "INSERT INTO")
        transformed = transformed.replace(
            "INSERT OR REPLACE INTO skill_trust",
            "INSERT INTO skill_trust",
        )
        # Retain compatibility for migrations and downstream repositories that
        # still issue the former reviewed description-only SQLite expression.
        transformed = _SQLITE_SKILL_DESCRIPTION_JSON_EXTRACT.sub(
            "(package_json::jsonb ->> 'description')",
            transformed,
        )
        transformed = re.sub(
            r"\s+INDEXED\s+BY\s+[A-Za-z_][A-Za-z0-9_]*",
            "",
            transformed,
            flags=re.IGNORECASE,
        )
        transformed = _prepare_parameterized_sql(transformed) if with_params else transformed.replace("?", "%s")
        if was_insert_or_ignore and "ON CONFLICT" not in transformed:
            transformed = f"{transformed.rstrip()} ON CONFLICT DO NOTHING"
        if was_insert_or_replace_skill_trust and "ON CONFLICT" not in transformed:
            transformed = (
                f"{transformed.rstrip()} "
                "ON CONFLICT (source_type, source, package_sha256) DO UPDATE SET "
                "trust_id = EXCLUDED.trust_id, "
                "trusted_by = EXCLUDED.trusted_by, "
                "created_at = EXCLUDED.created_at, "
                "metadata_json = EXCLUDED.metadata_json"
            )
        return transformed


class _PostgresCursor:
    def __init__(self, cursor: Any, dialect: _PostgresDialect):
        self._cursor = cursor
        self._dialect = dialect

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> "_PostgresCursor":
        selected_params = tuple(params)
        if selected_params:
            prepared = self._dialect.prepare(sql, with_params=True)
            self._cursor.execute(prepared, selected_params)
        else:
            prepared = self._dialect.prepare(sql)
            self._cursor.execute(prepared)
        return self

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> None:
        self._cursor.executemany(self._dialect.prepare(sql, with_params=True), [tuple(params) for params in seq_of_params])

    def fetchone(self) -> dict[str, Any] | None:
        return self._cursor.fetchone()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._cursor)


class _PostgresConnection:
    def __init__(self, dsn: str):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised without optional dependency
            raise ValidationError(
                "PostgreSQL runtime store requires the optional dependency; "
                "install with `uv sync --extra postgres --all-groups`"
            ) from exc
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        self._dialect = _PostgresDialect()

    def close(self) -> None:
        self._conn.close()

    @property
    def closed(self) -> bool:
        """Driver-reported session state after a possibly partial close."""

        return bool(self._conn.closed)

    @property
    def in_transaction(self) -> bool | None:
        """Report a definite transaction state after a commit diagnostic."""

        status = self._conn.info.transaction_status
        name = str(getattr(status, "name", "")).upper()
        if name == "IDLE":
            return False
        if name in {"INTRANS", "INERROR"}:
            return True
        # ACTIVE should not remain observable after a synchronous commit call
        # returns. Treat it as indeterminate rather than claiming rollback can
        # definitely recover an outcome that may already have crossed commit.
        return None

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def cursor(self) -> _PostgresCursor:
        return _PostgresCursor(self._conn.cursor(), self._dialect)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _PostgresCursor:
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> None:
        self.cursor().executemany(sql, seq_of_params)

    def executescript(self, script: str) -> None:
        for statement in split_sql_script(script):
            if statement:
                self.execute(statement)


class PostgresStore(SQLRuntimeStore):
    """PostgreSQL runtime store backend."""

    KEYSET_TEXT_COLLATION = "C"

    def __init__(
        self,
        dsn: str,
        *,
        config: AgentLibOSConfig | None = None,
        initialize_schema: bool = True,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.path = dsn
        self.dsn = dsn
        self._runtime_lease_acquired = False
        self._runtime_lease_key: int | None = None
        conn = _PostgresConnection(dsn)
        try:
            self._acquire_runtime_lease(conn)
            self._init_store(
                dsn,
                config=config,
                conn=conn,
                initialize_schema=initialize_schema,
            )
        except BaseException as primary_error:
            cleanup_errors = self._close_connection_best_effort(conn)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "PostgreSQL store initialization and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        cleanup_errors = self._close_connection_best_effort(conn)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "PostgreSQL store cleanup failed",
                list(cleanup_errors),
            ) from None

    def _close_connection_best_effort(
        self,
        conn: _PostgresConnection,
    ) -> tuple[BaseException, ...]:
        """Close the owning session without a separate unlock commit point.

        PostgreSQL advisory locks are session-scoped. An explicit unlock before
        close creates an unobservable acknowledgement window: the server may
        release the lock while the client sees both unlock and close errors.
        Session close is therefore the sole irreversible ownership transition
        used by runtime handoff.
        """

        errors: list[BaseException] = []
        try:
            conn.close()
        except BaseException as exc:
            errors.append(exc)
        if self._postgres_connection_reports_closed(conn):
            # A successfully closed PostgreSQL session cannot retain a session
            # advisory lock, even if close itself reported a diagnostic.
            self._runtime_lease_acquired = False
            self._runtime_lease_key = None
            self._backend_ownership_release_observed = True
        return tuple(errors)

    @staticmethod
    def _postgres_connection_reports_closed(conn: Any) -> bool:
        return getattr(conn, "closed", None) is True

    def _runtime_ownership_released(self) -> bool:
        if not getattr(self, "_runtime_lease_acquired", False):
            return True
        conn = getattr(self, "conn", None)
        if conn is not None and self._postgres_connection_reports_closed(conn):
            self._runtime_lease_acquired = False
            self._runtime_lease_key = None
            return True
        return False

    @classmethod
    def _probe_user_schema_objects(cls, conn: Any) -> set[str]:
        rows = conn.execute(
            """
            SELECT relation.relname AS name
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
            """
        )
        return {str(row["name"]) for row in rows}

    @classmethod
    def _probe_text_column_collations(
        cls,
        conn: Any,
    ) -> Mapping[tuple[str, str], str]:
        tables = sorted(_V3_KEYSET_TEXT_COLUMNS)
        placeholders = ", ".join("?" for _ in tables)
        rows = conn.execute(
            f"""
            SELECT collation_row.collname AS collation_name
                 , relation.relname AS table_name
                 , attribute.attname AS column_name
                 , current_setting('server_encoding') AS server_encoding
              FROM pg_catalog.pg_attribute AS attribute
              JOIN pg_catalog.pg_class AS relation
                ON relation.oid = attribute.attrelid
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
              JOIN pg_catalog.pg_collation AS collation_row
                ON collation_row.oid = attribute.attcollation
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN ({placeholders})
               AND attribute.attnum > 0
               AND NOT attribute.attisdropped
            """,
            tables,
        )
        selected_rows = list(rows)
        server_encodings = {
            str(row["server_encoding"]).upper() for row in selected_rows
        }
        if server_encodings != {"UTF8"}:
            raise UnsupportedStoreVersion(
                "Agent libOS PostgreSQL keyset ordering requires UTF8 server_encoding; "
                f"found {sorted(server_encodings) or ['missing']}"
            )
        required = {
            (table, column)
            for table, columns in _V3_KEYSET_TEXT_COLUMNS.items()
            for column in columns
        }
        return {
            (str(row["table_name"]), str(row["column_name"])): str(
                row["collation_name"]
            )
            for row in selected_rows
            if (str(row["table_name"]), str(row["column_name"])) in required
        }

    def _acquire_runtime_lease(self, conn: _PostgresConnection) -> None:
        identity = conn.execute(
            "SELECT current_database() AS database_name, current_schema() AS schema_name"
        ).fetchone()
        database = str(identity.get("database_name") or "") if identity else ""
        schema = str(identity.get("schema_name") or "") if identity else ""
        if not database or not schema:
            raise ValidationError("unable to resolve PostgreSQL database/schema for runtime lease")
        lease_key = _postgres_runtime_lock_key(database, schema)
        row = conn.execute(
            "SELECT pg_try_advisory_lock(?) AS acquired",
            (lease_key,),
        ).fetchone()
        if not row or not row.get("acquired"):
            raise ValidationError(f"runtime store is already open: postgres:{database}/{schema}")
        self._runtime_lease_key = lease_key
        self._runtime_lease_acquired = True

_PRAGMA_TABLE_INFO = re.compile(r"^\s*PRAGMA\s+table_info\((?P<table>[A-Za-z_][A-Za-z0-9_]*)\)\s*$", re.IGNORECASE)
_PRAGMA_INDEX_LIST = re.compile(r"^\s*PRAGMA\s+index_list\((?P<table>[A-Za-z_][A-Za-z0-9_]*)\)\s*$", re.IGNORECASE)
_PRAGMA_INDEX_INFO = re.compile(r"^\s*PRAGMA\s+index_info\((?P<index>[A-Za-z_][A-Za-z0-9_]*)\)\s*$", re.IGNORECASE)
_SQLITE_SKILL_DESCRIPTION_JSON_EXTRACT = re.compile(
    r"json_extract\(\s*package_json\s*,\s*'\$\.description'\s*\)",
    re.IGNORECASE,
)


def _prepare_parameterized_sql(sql: str) -> str:
    """Convert SQLite placeholders and escape literal percents for psycopg."""

    parts: list[str] = []
    in_quote = False
    quote_char = ""
    index = 0
    while index < len(sql):
        char = sql[index]
        if char in {"'", '"'}:
            parts.append(char)
            if in_quote and char == quote_char:
                if index + 1 < len(sql) and sql[index + 1] == quote_char:
                    parts.append(sql[index + 1])
                    index += 2
                    continue
                in_quote = False
                quote_char = ""
            elif not in_quote:
                in_quote = True
                quote_char = char
        elif char == "?" and not in_quote:
            parts.append("%s")
        elif char == "%":
            parts.append("%%")
        else:
            parts.append(char)
        index += 1
    return "".join(parts)
