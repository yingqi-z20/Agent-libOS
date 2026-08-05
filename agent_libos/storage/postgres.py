from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any, Mapping

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.engine import split_sql_script
from agent_libos.storage.postgres_schema_contract import (
    require_postgres_catalog_contract,
)
from agent_libos.storage.sql import (
    SQLRuntimeStore,
    _V4_KEYSET_TEXT_COLUMNS,
    _V4_REQUIRED_COLUMNS,
    _V5_REQUIRED_COLUMNS,
)
from agent_libos.storage.v5_schema_contract import (
    HUMAN_REQUEST_INDEX_CONTRACTS,
    V4_HUMAN_REQUEST_CHECKS,
    V4_HUMAN_REQUEST_COLUMN_CONTRACTS,
    V4_HUMAN_REQUEST_KEY_CONSTRAINTS,
    V5_STORAGE_COLUMN_CONTRACTS,
    V5_STORAGE_KEY_CONSTRAINTS,
    V5_STORAGE_POSTGRES_CHECKS,
)


def _normalized_postgres_default(value: Any) -> str | None:
    if value is None:
        return None
    selected = str(value).strip().casefold()
    selected = re.sub(r"::(?:bigint|integer)\Z", "", selected).strip()
    while selected.startswith("(") and selected.endswith(")"):
        selected = selected[1:-1].strip()
    return selected


def _normalized_postgres_constraint(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _postgres_runtime_lock_key(database: str, schema: str) -> int:
    """Return a stable signed bigint key scoped to one database/schema pair."""

    digest = hashlib.blake2b(digest_size=8, person=b"AgentLibOS")
    digest.update(database.encode("utf-8"))
    digest.update(b"\0")
    digest.update(schema.encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big", signed=True)


# Closing a psycopg session is the only authoritative lease-release action.
# PostgreSQL may need a very small interval to observe the closed socket and
# release that backend's session lock.  A new session may therefore retry the
# non-blocking acquisition, but it never unlocks or changes sessions while
# doing so.  The attempt limit remains a second hard bound if a patched clock
# or scheduler does not advance as expected.
_POSTGRES_RUNTIME_LEASE_RETRY_LIMIT = 10
_POSTGRES_RUNTIME_LEASE_RETRY_INTERVAL_SECONDS = 0.01
_POSTGRES_RUNTIME_LEASE_RETRY_WINDOW_SECONDS = 0.1


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

    def close(self) -> None:
        self._cursor.close()

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
                "install with `uv sync --frozen --extra postgres`"
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
    REQUIRE_V4_INDEX_OPERABILITY = True

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

        errors = (
            self._close_transaction_cursors()
            if conn is getattr(self, "conn", None)
            else []
        )
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
    def _probe_user_tables(cls, conn: Any) -> set[str]:
        rows = conn.execute(
            """
            SELECT relation.relname AS name
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relkind IN ('r', 'p')
            """
        )
        return {str(row["name"]) for row in rows}

    @classmethod
    def _require_v4_schema_shape(cls, conn: Any) -> None:
        """Require every manifest relation to be an ordinary PostgreSQL table."""

        required_tables = sorted(_V4_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            f"""
            SELECT relation.relname AS name
                 , relation.relkind AS relation_kind
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN ({placeholders})
            """,
            required_tables,
        )
        relation_kinds = {
            str(row["name"]): str(row["relation_kind"])
            for row in rows
        }
        invalid_relations = {
            table: relation_kinds.get(table, "missing")
            for table in required_tables
            if relation_kinds.get(table) != "r"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v4 "
                "manifest relation types: "
                f"{invalid_relations}; expected PostgreSQL relkind 'r'"
            )
        super()._require_v4_schema_shape(conn)
        cls._require_v4_human_request_contract(conn)
        cls._require_canonical_catalog_contract(conn, store_version=4)

    @classmethod
    def _require_v4_human_request_contract(cls, conn: Any) -> None:
        problems = {
            **cls._storage_column_problems(
                conn,
                V4_HUMAN_REQUEST_COLUMN_CONTRACTS,
            ),
            **cls._storage_check_problems(
                conn,
                V4_HUMAN_REQUEST_CHECKS,
            ),
            **cls._storage_key_constraint_problems(
                conn,
                V4_HUMAN_REQUEST_KEY_CONSTRAINTS,
            ),
            **cls._human_request_index_problems(conn),
            **cls._v5_semantic_boundary_problems(conn),
        }
        if problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v4 human request contract: "
                f"{problems}"
            )

    @classmethod
    def _require_v5_schema_shape(cls, conn: Any) -> None:
        """Require every schema-v5 manifest relation to be an ordinary table."""

        required_tables = sorted(_V5_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            f"""
            SELECT relation.relname AS name
                 , relation.relkind AS relation_kind
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN ({placeholders})
            """,
            required_tables,
        )
        relation_kinds = {
            str(row["name"]): str(row["relation_kind"])
            for row in rows
        }
        invalid_relations = {
            table: relation_kinds.get(table, "missing")
            for table in required_tables
            if relation_kinds.get(table) != "r"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v5 "
                "manifest relation types: "
                f"{invalid_relations}; expected PostgreSQL relkind 'r'"
            )
        super()._require_v5_schema_shape(conn)
        cls._require_v5_storage_contract(conn)
        cls._require_canonical_catalog_contract(conn, store_version=5)

    @classmethod
    def _require_canonical_catalog_contract(
        cls,
        conn: Any,
        *,
        store_version: int,
    ) -> None:
        require_postgres_catalog_contract(conn, store_version=store_version)

    @classmethod
    def _require_v5_storage_contract(cls, conn: Any) -> None:
        column_problems = cls._storage_column_problems(
            conn,
            V5_STORAGE_COLUMN_CONTRACTS,
        )
        if column_problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v5 storage column contract: "
                f"{column_problems}"
            )
        check_problems = cls._storage_check_problems(
            conn,
            V5_STORAGE_POSTGRES_CHECKS,
        )
        if check_problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v5 storage CHECK contract: "
                f"{check_problems}"
            )
        key_problems = cls._storage_key_constraint_problems(
            conn,
            V5_STORAGE_KEY_CONSTRAINTS,
        )
        if key_problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v5 storage key constraint "
                f"contract: {key_problems}"
            )
        index_problems = cls._human_request_index_problems(conn)
        if index_problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v5 human request index contract: "
                f"{index_problems}"
            )
        boundary_problems = cls._v5_semantic_boundary_problems(conn)
        if boundary_problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v5 semantic relation boundary: "
                f"{boundary_problems}"
            )

    @classmethod
    def _storage_column_problems(
        cls,
        conn: Any,
        contracts: Mapping[str, tuple[tuple[str, Any], ...]],
    ) -> dict[str, Any]:
        tables = sorted(contracts)
        placeholders = ", ".join("?" for _ in tables)
        rows = list(
            conn.execute(
                f"""
                SELECT table_name,
                       column_name,
                       ordinal_position,
                       data_type,
                       is_nullable,
                       column_default,
                       collation_name,
                       is_identity,
                       is_generated
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name IN ({placeholders})
                 ORDER BY table_name, ordinal_position
                """,
                tables,
            )
        )
        by_table: dict[str, list[dict[str, Any]]] = {table: [] for table in tables}
        for row in rows:
            by_table[str(row["table_name"])].append(dict(row))
        problems: dict[str, Any] = {}
        for table, contract_items in contracts.items():
            selected_rows = by_table[table]
            expected_names = tuple(name for name, _ in contract_items)
            actual_names = tuple(str(row["column_name"]) for row in selected_rows)
            if actual_names != expected_names:
                problems[f"{table}.columns"] = {
                    "expected": expected_names,
                    "actual": actual_names,
                }
                continue
            for row, (name, contract) in zip(selected_rows, contract_items):
                actual = {
                    "type": str(row["data_type"]).casefold(),
                    "nullable": str(row["is_nullable"]).upper() == "YES",
                    "default": _normalized_postgres_default(
                        row["column_default"]
                    ),
                    "collation": (
                        str(row["collation_name"])
                        if row["collation_name"] is not None
                        else None
                    ),
                    "identity": str(row["is_identity"]).upper(),
                    "generated": str(row["is_generated"]).upper(),
                }
                expected = {
                    "type": contract.sql_type,
                    "nullable": contract.nullable,
                    "default": contract.default,
                    "collation": (
                        cls.KEYSET_TEXT_COLLATION
                        if contract.keyset_collation
                        else None
                    ),
                    "identity": "NO",
                    "generated": "NEVER",
                }
                if table != "human_requests":
                    actual["ordinal"] = int(row["ordinal_position"])
                    expected["ordinal"] = expected_names.index(name) + 1
                if actual != expected:
                    problems[f"{table}.{name}"] = {
                        "expected": expected,
                        "actual": actual,
                    }
        return problems

    @staticmethod
    def _storage_check_problems(
        conn: Any,
        contracts: Mapping[str, tuple[str, ...]],
    ) -> dict[str, Any]:
        tables = sorted(contracts)
        placeholders = ", ".join("?" for _ in tables)
        rows = list(
            conn.execute(
                f"""
                SELECT relation.relname AS table_name,
                       pg_catalog.pg_get_constraintdef(
                         constraint_row.oid, true
                       ) AS definition,
                       constraint_row.convalidated AS is_validated,
                       constraint_row.connoinherit AS no_inherit,
                       constraint_row.condeferrable AS is_deferrable,
                       constraint_row.condeferred AS is_deferred
                  FROM pg_catalog.pg_constraint AS constraint_row
                  JOIN pg_catalog.pg_class AS relation
                    ON relation.oid = constraint_row.conrelid
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = relation.relnamespace
                 WHERE namespace.nspname = current_schema()
                   AND relation.relname IN ({placeholders})
                   AND constraint_row.contype = 'c'
                """,
                tables,
            )
        )
        actual_by_table: dict[str, Counter[str]] = {
            table: Counter() for table in tables
        }
        invalid_flags: dict[str, int] = {}
        for row in rows:
            table = str(row["table_name"])
            actual_by_table[table][
                _normalized_postgres_constraint(row["definition"])
            ] += 1
            if (
                not bool(row["is_validated"])
                or bool(row["no_inherit"])
                or bool(row["is_deferrable"])
                or bool(row["is_deferred"])
            ):
                invalid_flags[table] = invalid_flags.get(table, 0) + 1
        problems: dict[str, Any] = {}
        for table, expected_definitions in contracts.items():
            actual = actual_by_table[table]
            expected = Counter(
                _normalized_postgres_constraint(definition)
                for definition in expected_definitions
            )
            if actual != expected or invalid_flags.get(table, 0):
                problems[table] = {
                    "expected_count": sum(expected.values()),
                    "actual_count": sum(actual.values()),
                    "missing": sum((expected - actual).values()),
                    "extra": sum((actual - expected).values()),
                    "invalid_flags": invalid_flags.get(table, 0),
                }
        return problems

    @staticmethod
    def _storage_key_constraint_problems(
        conn: Any,
        contracts: Mapping[
            str,
            tuple[tuple[str, tuple[str, ...]], ...],
        ],
    ) -> dict[str, Any]:
        tables = sorted(contracts)
        placeholders = ", ".join("?" for _ in tables)
        rows = list(
            conn.execute(
                f"""
                SELECT relation.relname AS table_name,
                       constraint_row.conname AS constraint_name,
                       constraint_row.contype AS constraint_type,
                       constraint_row.condeferrable AS is_deferrable,
                       constraint_row.condeferred AS is_deferred,
                       constraint_row.convalidated AS is_validated,
                       key_row.ordinality AS key_ordinal,
                       attribute.attname AS column_name
                  FROM pg_catalog.pg_constraint AS constraint_row
                  JOIN pg_catalog.pg_class AS relation
                    ON relation.oid = constraint_row.conrelid
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = relation.relnamespace
                  JOIN LATERAL unnest(constraint_row.conkey)
                       WITH ORDINALITY AS key_row(attnum, ordinality) ON true
                  JOIN pg_catalog.pg_attribute AS attribute
                    ON attribute.attrelid = relation.oid
                   AND attribute.attnum = key_row.attnum
                 WHERE namespace.nspname = current_schema()
                   AND relation.relname IN ({placeholders})
                   AND constraint_row.contype IN ('p', 'u')
                 ORDER BY relation.relname,
                          constraint_row.conname,
                          key_row.ordinality
                """,
                tables,
            )
        )
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["table_name"]), str(row["constraint_name"]))
            item = grouped.setdefault(
                key,
                {
                    "type": str(row["constraint_type"]),
                    "columns": [],
                    "valid": bool(row["is_validated"]),
                    "deferrable": bool(row["is_deferrable"]),
                    "deferred": bool(row["is_deferred"]),
                },
            )
            item["columns"].append(str(row["column_name"]))
        actual_by_table: dict[
            str,
            Counter[tuple[str, tuple[str, ...]]],
        ] = {table: Counter() for table in tables}
        for (table, _), item in grouped.items():
            kind = {
                "p": "primary_key",
                "u": "unique",
            }.get(str(item["type"]), f"unknown:{item['type']}")
            if item["deferrable"] or item["deferred"] or not item["valid"]:
                kind = f"invalid:{kind}"
            actual_by_table[table][(kind, tuple(item["columns"]))] += 1
        problems: dict[str, Any] = {}
        for table, expected_constraints in contracts.items():
            actual = actual_by_table[table]
            expected = Counter(expected_constraints)
            if actual != expected:
                problems[f"{table}.keys"] = {
                    "expected": sorted(expected.elements()),
                    "actual": sorted(actual.elements()),
                }
        return problems

    @classmethod
    def _human_request_index_problems(cls, conn: Any) -> dict[str, Any]:
        shapes = cls._probe_index_shapes(conn, {"human_requests"})
        problems: dict[str, Any] = {}
        for name, expected in HUMAN_REQUEST_INDEX_CONTRACTS.items():
            problem = cls._v4_index_shape_problem(
                name,
                expected,
                shapes.get(name),
            )
            if problem is not None:
                problems[name] = problem
        declared = {
            name
            for name, shape in shapes.items()
            if shape.get("origin") == "declared"
        }
        extra = sorted(declared - set(HUMAN_REQUEST_INDEX_CONTRACTS))
        if extra:
            problems["<extra human indexes>"] = extra
        unique_problem = cls._v4_unique_shape_problem(
            "human_requests",
            frozenset({("request_id",)}),
            shapes,
        )
        if unique_problem is not None:
            problems["human_requests.unique"] = unique_problem
        return problems

    @staticmethod
    def _v5_semantic_boundary_problems(conn: Any) -> dict[str, Any]:
        guarded_tables = (
            "human_requests",
            "semantic_assessment_jobs",
            "semantic_assessments",
        )
        placeholders = ", ".join("?" for _ in guarded_tables)
        foreign_keys = list(
            conn.execute(
                f"""
                SELECT relation.relname AS table_name,
                       constraint_row.conname AS constraint_name
                  FROM pg_catalog.pg_constraint AS constraint_row
                  JOIN pg_catalog.pg_class AS relation
                    ON relation.oid = constraint_row.conrelid
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = relation.relnamespace
                 WHERE namespace.nspname = current_schema()
                   AND relation.relname IN ({placeholders})
                   AND constraint_row.contype = 'f'
                """,
                guarded_tables,
            )
        )
        user_triggers = list(
            conn.execute(
                f"""
                SELECT relation.relname AS table_name,
                       trigger_row.tgname AS trigger_name
                  FROM pg_catalog.pg_trigger AS trigger_row
                  JOIN pg_catalog.pg_class AS relation
                    ON relation.oid = trigger_row.tgrelid
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = relation.relnamespace
                 WHERE namespace.nspname = current_schema()
                   AND relation.relname IN ({placeholders})
                   AND NOT trigger_row.tgisinternal
                """,
                guarded_tables,
            )
        )
        mutation_hooks = list(
            conn.execute(
                f"""
                SELECT relation.relname AS table_name,
                       relation.relrowsecurity AS row_security,
                       relation.relforcerowsecurity AS force_row_security,
                       policy_row.polname AS policy_name,
                       rewrite_row.rulename AS rule_name
                  FROM pg_catalog.pg_class AS relation
                  JOIN pg_catalog.pg_namespace AS namespace
                    ON namespace.oid = relation.relnamespace
             LEFT JOIN pg_catalog.pg_policy AS policy_row
                    ON policy_row.polrelid = relation.oid
             LEFT JOIN pg_catalog.pg_rewrite AS rewrite_row
                    ON rewrite_row.ev_class = relation.oid
                 WHERE namespace.nspname = current_schema()
                   AND relation.relname IN ({placeholders})
                   AND (
                     relation.relrowsecurity
                     OR relation.relforcerowsecurity
                     OR policy_row.oid IS NOT NULL
                     OR rewrite_row.oid IS NOT NULL
                   )
                """,
                guarded_tables,
            )
        )
        problems: dict[str, Any] = {}
        if foreign_keys:
            problems["foreign_keys"] = sorted(
                f"{row['table_name']}.{row['constraint_name']}"
                for row in foreign_keys
            )
        if user_triggers:
            problems["triggers"] = sorted(
                f"{row['table_name']}.{row['trigger_name']}"
                for row in user_triggers
            )
        if mutation_hooks:
            problems["policies_or_rules"] = sorted(
                (
                    str(row["table_name"]),
                    bool(row["row_security"]),
                    bool(row["force_row_security"]),
                    str(row["policy_name"] or ""),
                    str(row["rule_name"] or ""),
                )
                for row in mutation_hooks
            )
        return problems

    @classmethod
    def _probe_text_column_collations(
        cls,
        conn: Any,
        *,
        keyset_columns: Mapping[str, frozenset[str]] = _V4_KEYSET_TEXT_COLUMNS,
    ) -> Mapping[tuple[str, str], str]:
        tables = sorted(keyset_columns)
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
            for table, columns in keyset_columns.items()
            for column in columns
        }
        return {
            (str(row["table_name"]), str(row["column_name"])): str(
                row["collation_name"]
            )
            for row in selected_rows
            if (str(row["table_name"]), str(row["column_name"])) in required
        }

    @classmethod
    def _probe_index_shapes(
        cls,
        conn: Any,
        tables: set[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        selected_tables = sorted(tables)
        if not selected_tables:
            return {}
        placeholders = ", ".join("?" for _ in selected_tables)
        rows = conn.execute(
            f"""
            SELECT index_relation.relname AS index_name,
                   table_relation.relname AS table_name,
                   index_row.indisunique AS is_unique,
                   index_row.indisvalid AS is_valid,
                   index_row.indisready AS is_ready,
                   index_row.indislive AS is_live,
                   (index_row.indpred IS NOT NULL) AS is_partial,
                   pg_catalog.pg_get_expr(
                     index_row.indpred, index_row.indrelid
                   ) AS predicate_sql,
                   key_row.ordinality AS key_ordinal,
                   attribute.attname AS column_name,
                   collation_row.collname AS collation_name,
                   ((index_row.indoption[key_row.ordinality - 1] & 1) = 1)
                     AS is_descending,
                   (constraint_row.oid IS NOT NULL) AS is_constraint
              FROM pg_catalog.pg_index AS index_row
              JOIN pg_catalog.pg_class AS index_relation
                ON index_relation.oid = index_row.indexrelid
              JOIN pg_catalog.pg_class AS table_relation
                ON table_relation.oid = index_row.indrelid
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = table_relation.relnamespace
              JOIN LATERAL unnest(index_row.indkey)
                   WITH ORDINALITY AS key_row(attnum, ordinality) ON true
              LEFT JOIN pg_catalog.pg_attribute AS attribute
                ON attribute.attrelid = table_relation.oid
               AND attribute.attnum = key_row.attnum
              LEFT JOIN pg_catalog.pg_collation AS collation_row
                ON collation_row.oid =
                   index_row.indcollation[key_row.ordinality - 1]
              LEFT JOIN pg_catalog.pg_constraint AS constraint_row
                ON constraint_row.conindid = index_row.indexrelid
               AND constraint_row.contype IN ('p', 'u')
             WHERE namespace.nspname = current_schema()
               AND table_relation.relname IN ({placeholders})
               AND key_row.ordinality <= index_row.indnkeyatts
             ORDER BY index_relation.relname, key_row.ordinality
            """,
            selected_tables,
        )
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = str(row["index_name"])
            shape = grouped.setdefault(
                name,
                {
                    "table": str(row["table_name"]),
                    "columns": [],
                    "unique": bool(row["is_unique"]),
                    "valid": bool(row["is_valid"]),
                    "ready": bool(row["is_ready"]),
                    "live": bool(row["is_live"]),
                    "partial": bool(row["is_partial"]),
                    "descending": [],
                    "collations": [],
                    "origin": (
                        "constraint" if bool(row["is_constraint"]) else "declared"
                    ),
                    "predicate": cls._canonical_index_predicate(
                        row["predicate_sql"]
                    ),
                },
            )
            shape["columns"].append(
                str(row["column_name"])
                if row["column_name"] is not None
                else "<expression>"
            )
            shape["descending"].append(bool(row["is_descending"]))
            shape["collations"].append(
                str(row["collation_name"])
                if row["collation_name"] is not None
                else None
            )
        return {
            name: {
                **shape,
                "columns": tuple(shape["columns"]),
                "descending": tuple(shape["descending"]),
                "collations": tuple(shape["collations"]),
            }
            for name, shape in grouped.items()
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
        deadline = time.monotonic() + _POSTGRES_RUNTIME_LEASE_RETRY_WINDOW_SECONDS
        for retry_count in range(_POSTGRES_RUNTIME_LEASE_RETRY_LIMIT + 1):
            row = conn.execute(
                "SELECT pg_try_advisory_lock(?) AS acquired",
                (lease_key,),
            ).fetchone()
            acquired = row.get("acquired") if row is not None else None
            if type(acquired) is not bool:
                raise ValidationError(
                    "PostgreSQL returned an invalid runtime lease result"
                )
            if acquired:
                self._runtime_lease_key = lease_key
                self._runtime_lease_acquired = True
                return
            if retry_count == _POSTGRES_RUNTIME_LEASE_RETRY_LIMIT:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(
                min(_POSTGRES_RUNTIME_LEASE_RETRY_INTERVAL_SECONDS, remaining)
            )
        raise ValidationError(
            f"runtime store is already open: postgres:{database}/{schema}"
        )

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
