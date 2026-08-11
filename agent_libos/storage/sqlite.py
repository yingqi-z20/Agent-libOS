from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from threading import RLock, local
from typing import Any, ClassVar, Iterator, Mapping

from agent_libos.config import AgentLibOSConfig
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.sql import (
    SQLRuntimeStore,
    _V4_KEYSET_TEXT_COLUMNS,
    _V4_REQUIRED_COLUMNS,
    _V5_KEYSET_TEXT_COLUMNS,
    _V5_REQUIRED_COLUMNS,
    _V6_KEYSET_TEXT_COLUMNS,
    _V6_REQUIRED_COLUMNS,
    _V7_KEYSET_TEXT_COLUMNS,
    _V7_REQUIRED_COLUMNS,
)
from agent_libos.storage.v5_schema_contract import (
    HUMAN_REQUEST_INDEX_CONTRACTS,
    V4_HUMAN_REQUEST_CHECKS,
    V4_HUMAN_REQUEST_COLUMN_CONTRACTS,
    V4_HUMAN_REQUEST_KEY_CONSTRAINTS,
    V5_STORAGE_COLUMN_CONTRACTS,
    V5_STORAGE_KEY_CONSTRAINTS,
    V5_STORAGE_SQLITE_CHECKS,
)
from agent_libos.storage.v6_schema_contract import (
    V6_STORAGE_COLUMN_CONTRACTS,
    V6_STORAGE_KEY_CONSTRAINTS,
    V6_STORAGE_SQLITE_CHECKS,
    V6_TABLES,
)
from agent_libos.storage.v7_schema_contract import (
    V7_STORAGE_COLUMN_CONTRACTS,
    V7_STORAGE_KEY_CONSTRAINTS,
    V7_STORAGE_SQLITE_CHECKS,
    V7_TABLES,
)
from agent_libos.utils.ids import utc_now

try:  # pragma: no cover - Windows fallback is exercised only on non-POSIX hosts.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _sqlite_ignored_sql_end(sql: str, index: int) -> int | None:
    """Return the end of whitespace or a comment beginning at ``index``."""

    if sql[index].isspace():
        end = index + 1
        while end < len(sql) and sql[end].isspace():
            end += 1
        return end
    if sql.startswith("--", index):
        newline = sql.find("\n", index + 2)
        return len(sql) if newline < 0 else newline + 1
    if sql.startswith("/*", index):
        closing = sql.find("*/", index + 2)
        return len(sql) if closing < 0 else closing + 2
    return None


def _sqlite_delimited_value(
    sql: str,
    index: int,
    closing: str,
) -> tuple[str, int]:
    """Consume a SQLite string or quoted identifier with doubled escapes."""

    value: list[str] = []
    index += 1
    while index < len(sql):
        if sql[index] != closing:
            value.append(sql[index])
            index += 1
            continue
        if index + 1 < len(sql) and sql[index + 1] == closing:
            value.append(closing)
            index += 2
            continue
        return "".join(value), index + 1
    return "".join(value), index


def _sqlite_word_end(sql: str, index: int) -> int:
    end = index + 1
    while end < len(sql) and (sql[end].isalnum() or sql[end] in {"_", "$"}):
        end += 1
    return end


def _sqlite_schema_token(sql: str, index: int) -> tuple[tuple[str, str], int]:
    char = sql[index]
    if char == "'":
        value, end = _sqlite_delimited_value(sql, index, char)
        return ("literal", value), end
    if char in {'"', "`", "["}:
        closing = "]" if char == "[" else char
        value, end = _sqlite_delimited_value(sql, index, closing)
        return ("quoted", value), end
    if char.isalnum() or char in {"_", "$"}:
        end = _sqlite_word_end(sql, index)
        return ("word", sql[index:end]), end
    return ("symbol", char), index + 1


def _sqlite_schema_tokens(sql: str) -> list[tuple[str, str]]:
    """Tokenize SQLite DDL without treating comments or literals as code."""

    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(sql):
        ignored_end = _sqlite_ignored_sql_end(sql, index)
        if ignored_end is not None:
            index = ignored_end
            continue
        token, index = _sqlite_schema_token(sql, index)
        tokens.append(token)
    return tokens


def _sqlite_column_definitions(
    tokens: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    try:
        opening = tokens.index(("symbol", "("))
    except ValueError:
        return []
    definitions: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    depth = 1
    for token in tokens[opening + 1 :]:
        if token == ("symbol", "("):
            depth += 1
        elif token == ("symbol", ")"):
            depth -= 1
            if depth == 0:
                if current:
                    definitions.append(current)
                break
        if token == ("symbol", ",") and depth == 1:
            definitions.append(current)
            current = []
        else:
            current.append(token)
    return definitions


def _sqlite_top_level_collations(
    definition: list[tuple[str, str]],
) -> list[str] | None:
    collations: list[str] = []
    nested_depth = 0
    position = 2
    while position < len(definition):
        kind, value = definition[position]
        if (kind, value) == ("symbol", "("):
            nested_depth += 1
        elif (kind, value) == ("symbol", ")"):
            nested_depth = max(0, nested_depth - 1)
        elif (
            kind == "word"
            and value.upper() == "COLLATE"
            and nested_depth == 0
        ):
            if position + 1 >= len(definition):
                return None
            collation_kind, collation_name = definition[position + 1]
            if collation_kind not in {"word", "quoted"}:
                return None
            collations.append(collation_name.upper())
            position += 1
        position += 1
    return collations


def _sqlite_text_column_collation(
    definition: list[tuple[str, str]],
) -> tuple[str, str] | None:
    if len(definition) < 2:
        return None
    name_kind, name = definition[0]
    type_kind, declared_type = definition[1]
    if name_kind not in {"word", "quoted"}:
        return None
    if type_kind != "word" or declared_type.upper() != "TEXT":
        return None
    collations = _sqlite_top_level_collations(definition)
    if collations is None:
        return None
    return name, collations[-1] if collations else "BINARY"


def _sqlite_column_collations(sql: str) -> dict[str, str]:
    """Return effective declared collations for TEXT columns in table DDL.

    ``sqlite_master.sql`` preserves comments and literals.  Parsing tokens keeps
    non-code ``COLLATE BINARY`` text from disguising a later NOCASE constraint.
    """

    result: dict[str, str] = {}
    definitions = _sqlite_column_definitions(_sqlite_schema_tokens(sql))
    for definition in definitions:
        declared = _sqlite_text_column_collation(definition)
        if declared is not None:
            result[declared[0]] = declared[1]
    return result


def _sqlite_contract_token(token: tuple[str, str]) -> tuple[str, str]:
    kind, value = token
    if kind in {"word", "quoted"}:
        return "word", value.casefold()
    return kind, value


def _strip_redundant_sqlite_parentheses(
    tokens: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    selected = tokens
    while len(selected) >= 2 and selected[0] == ("symbol", "("):
        depth = 0
        closes_at_end = False
        for index, token in enumerate(selected):
            if token == ("symbol", "("):
                depth += 1
            elif token == ("symbol", ")"):
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(selected) - 1
                    break
        if not closes_at_end:
            break
        selected = selected[1:-1]
    return selected


def _sqlite_contract_expression(sql: str) -> tuple[tuple[str, str], ...]:
    tokens = tuple(
        _sqlite_contract_token(token) for token in _sqlite_schema_tokens(sql)
    )
    return _strip_redundant_sqlite_parentheses(tokens)


def _sqlite_check_expressions(sql: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    tokens = _sqlite_schema_tokens(sql)
    expressions: list[tuple[tuple[str, str], ...]] = []
    position = 0
    while position < len(tokens):
        kind, value = tokens[position]
        if kind != "word" or value.casefold() != "check":
            position += 1
            continue
        if position + 1 >= len(tokens) or tokens[position + 1] != (
            "symbol",
            "(",
        ):
            return ()
        depth = 1
        end = position + 2
        while end < len(tokens) and depth:
            if tokens[end] == ("symbol", "("):
                depth += 1
            elif tokens[end] == ("symbol", ")"):
                depth -= 1
            end += 1
        if depth:
            return ()
        expression = tuple(
            _sqlite_contract_token(token)
            for token in tokens[position + 2 : end - 1]
        )
        expressions.append(_strip_redundant_sqlite_parentheses(expression))
        position = end
    return tuple(expressions)


def _normalized_sqlite_default(value: Any) -> str | None:
    if value is None:
        return None
    selected = str(value).strip()
    while selected.startswith("(") and selected.endswith(")"):
        selected = selected[1:-1].strip()
    return selected.casefold()


_SQLITE_CANONICAL_CATALOG_LOCK = RLock()
_SQLITE_CANONICAL_CATALOGS: dict[int, dict[str, Any]] = {}
_SQLITE_CANONICAL_CATALOG_BUILD = local()
# Full catalog produced by the clean schema-v4 baseline commit ``4b43cb7``.
# The synthesized reference is required to match this golden digest so a
# future v5 DDL edit cannot silently redefine which production v4 stores are
# eligible for the only supported offline migration.
_SQLITE_CANONICAL_V4_CATALOG_SHA256 = (
    "0bfa8d224a417aff3d672684f52638cb913ba0f6e17beac8e794bba467e62015"
)
# Schema v5 is also a versioned disk contract.  Changing runtime DDL while
# leaving STORE_SCHEMA_VERSION at 5 must fail this ratchet instead of silently
# redefining v5 and making previously created stores unreadable.
_SQLITE_CANONICAL_V5_CATALOG_SHA256 = (
    "d92abab9c668bb44f348de0f78e1be1198854bbfec64f61e17af93bb5a0902e6"
)
_SQLITE_CANONICAL_V6_CATALOG_SHA256 = (
    "ac1735257279e943a9eaa4ad75ecb078c58b279e9ad4ba37aad2e2417d35c50d"
)
_SQLITE_CANONICAL_V7_CATALOG_SHA256 = (
    "e488c584f494028648354dda0be1d9fcfa8061560ed092d868298bd699af5565"
)
_SQLITE_CANONICAL_CATALOG_SHA256 = {
    4: _SQLITE_CANONICAL_V4_CATALOG_SHA256,
    5: _SQLITE_CANONICAL_V5_CATALOG_SHA256,
    6: _SQLITE_CANONICAL_V6_CATALOG_SHA256,
    7: _SQLITE_CANONICAL_V7_CATALOG_SHA256,
}


def _sqlite_quoted_identifier(value: str) -> str:
    """Quote one catalog-provided SQLite identifier without trusting its shape."""

    return '"' + value.replace('"', '""') + '"'


def _sqlite_normalized_schema_sql(value: Any) -> list[list[str]] | None:
    """Return a whitespace/comment/identifier-quote independent DDL token stream.

    The complete stream is deliberately retained.  PRAGMA metadata does not
    expose column/table conflict policies, DEFERRABLE clauses, or every table
    option.  Comparing normalized ``sqlite_master.sql`` therefore closes gaps
    such as ``PRIMARY KEY ON CONFLICT REPLACE`` while the structured catalog
    probes below independently cover effective types, generated columns,
    collations, foreign keys, and index keys.
    """

    if value is None:
        return None
    return [
        list(_sqlite_contract_token(token))
        for token in _sqlite_schema_tokens(str(value))
    ]


def _sqlite_table_catalog(
    conn: Any,
    table: str,
    options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    quoted_table = _sqlite_quoted_identifier(table)
    columns = []
    for row in conn.execute(f"PRAGMA table_xinfo({quoted_table})"):
        selected = dict(row)
        columns.append(
            {
                "ordinal": int(selected["cid"]),
                "name": str(selected["name"]),
                "type": str(selected["type"]).casefold(),
                "not_null": bool(selected["notnull"]),
                "default": _sqlite_normalized_schema_sql(
                    selected["dflt_value"]
                ),
                "primary_key_position": int(selected["pk"]),
                "hidden": int(selected["hidden"]),
            }
        )
    foreign_keys = []
    for row in conn.execute(f"PRAGMA foreign_key_list({quoted_table})"):
        selected = dict(row)
        foreign_keys.append(
            {
                "id": int(selected["id"]),
                "sequence": int(selected["seq"]),
                "table": str(selected["table"]),
                "from": (
                    None if selected["from"] is None else str(selected["from"])
                ),
                "to": None if selected["to"] is None else str(selected["to"]),
                "on_update": str(selected["on_update"]).casefold(),
                "on_delete": str(selected["on_delete"]).casefold(),
                "match": str(selected["match"]).casefold(),
            }
        )
    foreign_keys.sort(key=lambda item: (item["id"], item["sequence"]))
    index_rows = []
    index_names: set[str] = set()
    for row in conn.execute(f"PRAGMA index_list({quoted_table})"):
        selected = dict(row)
        index_name = str(selected["name"])
        index_names.add(index_name)
        index_rows.append(
            {
                "name": index_name,
                "unique": bool(selected["unique"]),
                "origin": str(selected["origin"]),
                "partial": bool(selected["partial"]),
            }
        )
    index_rows.sort(key=lambda item: item["name"])
    return (
        {
            "options": dict(options) if options is not None else None,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "indexes": index_rows,
        },
        index_names,
    )


def _sqlite_index_catalog(conn: Any, index_name: str) -> dict[str, Any]:
    quoted_index = _sqlite_quoted_identifier(index_name)
    key_rows = []
    auxiliary_rows = []
    for row in conn.execute(f"PRAGMA index_xinfo({quoted_index})"):
        selected = dict(row)
        item = {
            "sequence": int(selected["seqno"]),
            "column_id": int(selected["cid"]),
            "name": None if selected["name"] is None else str(selected["name"]),
            "descending": bool(selected["desc"]),
            "collation": (
                None
                if selected["coll"] is None
                else str(selected["coll"]).upper()
            ),
            "key": bool(selected["key"]),
        }
        (key_rows if item["key"] else auxiliary_rows).append(item)
    return {
        "keys": sorted(key_rows, key=lambda item: item["sequence"]),
        "auxiliary": sorted(auxiliary_rows, key=lambda item: item["sequence"]),
    }


def _sqlite_full_catalog_snapshot(conn: Any) -> dict[str, Any]:
    """Read the complete durable user-schema catalog without mutating ``conn``."""

    encoding_row = conn.execute("PRAGMA encoding").fetchone()
    encoding = (
        "missing"
        if encoding_row is None
        else str(
            encoding_row[0]
            if not isinstance(encoding_row, dict)
            else encoding_row["encoding"]
        )
    )
    master_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT type, name, tbl_name, sql
              FROM sqlite_master
             WHERE (
                    type = 'table' AND name NOT LIKE 'sqlite_%'
                   )
                OR (
                    type IN ('index', 'trigger')
                    AND tbl_name NOT LIKE 'sqlite_%'
                   )
                OR (
                    type = 'view' AND name NOT LIKE 'sqlite_%'
                   )
             ORDER BY type, name
            """
        )
    ]
    objects = [
        {
            "type": str(row["type"]),
            "name": str(row["name"]),
            "table": str(row["tbl_name"]),
            "sql": _sqlite_normalized_schema_sql(row["sql"]),
        }
        for row in master_rows
    ]
    table_names = sorted(
        str(row["name"])
        for row in master_rows
        if str(row["type"]) == "table"
    )

    table_list_by_name: dict[str, dict[str, Any]] = {}
    try:
        table_list_rows = list(conn.execute("PRAGMA main.table_list"))
    except sqlite3.DatabaseError as exc:  # pragma: no cover - supported runtime
        raise UnsupportedStoreVersion(
            "SQLite runtime cannot inspect canonical table options"
        ) from exc
    for row in table_list_rows:
        selected = dict(row)
        name = str(selected.get("name", ""))
        if name not in table_names:
            continue
        table_list_by_name[name] = {
            "type": str(selected.get("type", "")),
            "columns": int(selected.get("ncol", -1)),
            "without_rowid": bool(selected.get("wr", 0)),
            "strict": bool(selected.get("strict", 0)),
        }

    tables: dict[str, Any] = {}
    all_index_names: set[str] = set()
    for table in table_names:
        tables[table], index_names = _sqlite_table_catalog(
            conn,
            table,
            table_list_by_name.get(table),
        )
        all_index_names.update(index_names)

    indexes = {
        name: _sqlite_index_catalog(conn, name)
        for name in sorted(all_index_names)
    }

    return {
        "database": {"encoding": encoding.upper()},
        "objects": objects,
        "tables": tables,
        "indexes": indexes,
    }


def _sqlite_catalog_sha256(catalog: Mapping[str, Any]) -> str:
    payload = json.dumps(
        catalog,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sqlite_catalog_difference(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> str:
    expected_objects = expected.get("objects")
    actual_objects = actual.get("objects")
    if isinstance(expected_objects, list) and isinstance(actual_objects, list):
        expected_by_name = {
            str(item.get("name")): item
            for item in expected_objects
            if isinstance(item, dict)
        }
        actual_by_name = {
            str(item.get("name")): item
            for item in actual_objects
            if isinstance(item, dict)
        }
        binding_index = "idx_operations_runtime_publication"
        if expected_by_name.get(binding_index) != actual_by_name.get(binding_index):
            return "exact durable runtime-publication binding index differs"
    for section in ("database", "objects", "tables", "indexes"):
        expected_section = expected.get(section)
        actual_section = actual.get(section)
        if expected_section != actual_section:
            if isinstance(expected_section, dict) and isinstance(actual_section, dict):
                missing = sorted(set(expected_section) - set(actual_section))
                extra = sorted(set(actual_section) - set(expected_section))
                changed = sorted(
                    key
                    for key in set(expected_section) & set(actual_section)
                    if expected_section[key] != actual_section[key]
                )
                return (
                    f"{section}: missing={missing[:3]!r}, extra={extra[:3]!r}, "
                    f"changed={changed[:3]!r}"
                )
            return f"{section} differs"
    return "catalog differs"


class _SQLiteRuntimeLease:
    def __init__(
        self,
        handle: Any,
        path: Path,
        *,
        identity_handle: Any,
        identity_path: Path,
        database_identity: tuple[int, int],
    ) -> None:
        self.handle = handle
        self.path = path
        self.identity_handle = identity_handle
        self.identity_path = identity_path
        self.database_identity = database_identity


class SQLiteStore(SQLRuntimeStore):
    """SQLite runtime store backend.

    Connection setup, file hardening, and lease behavior remain SQLite-only;
    backend-neutral repositories live in :class:`SQLRuntimeStore`.
    """

    KEYSET_TEXT_COLLATION = "BINARY"
    _failed_owner_lock: ClassVar[RLock] = RLock()
    _failed_owners: ClassVar[dict[tuple[int, int], "SQLiteStore"]] = {}

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        config: AgentLibOSConfig | None = None,
        initialize_schema: bool = True,
    ) -> None:
        selected_path = str(path)
        connection_path = selected_path
        connection_target = selected_path
        connection_uri = False
        self._lease_handle: Any | None = None
        self._sqlite_connection_closed = False
        self._database_identity: tuple[int, int] | None = None
        # Every persistent connection also holds SQLite's own database lock.
        # The POSIX path/inode sidecars bind the selected pathname, while this
        # lock independently preserves the single-owner invariant for the file
        # SQLite actually opened if a same-UID Host administrator retargets the
        # pathname during sqlite3_connect().
        use_exclusive_database_lease = selected_path != ":memory:"
        if selected_path == ":memory:" and not initialize_schema:
            raise ValidationError(
                "offline migration requires an existing initialized Agent libOS store"
            )
        if selected_path != ":memory:":
            db_path = Path(selected_path)
            # Resolve existing symlinks and relative aliases before deriving
            # the lock path. Otherwise the same SQLite file can be opened by
            # two runtimes through distinct path spellings and receive two
            # independent lease files.
            canonical_path = db_path.resolve()
            existing_store = canonical_path.exists()
            if not existing_store:
                if not initialize_schema:
                    raise ValidationError(
                        "offline migration requires an existing initialized "
                        "Agent libOS store"
                    )
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
            if existing_store:
                # Validate without mutation before even the read-only schema
                # probe can ask SQLite to open a pre-existing sidecar. An
                # unsupported store must remain byte- and mode-identical.
                self._secure_database_files(
                    canonical_path,
                    tighten=False,
                    create_if_missing=False,
                )
                existing_stat = os.stat(
                    canonical_path,
                    follow_symlinks=False,
                )
                self._database_identity = (
                    existing_stat.st_dev,
                    existing_stat.st_ino,
                )
                self._retry_failed_owner(self._database_identity)
                fresh_store = self._preflight_existing_store(canonical_path)
                if not initialize_schema and fresh_store:
                    raise ValidationError(
                        "offline migration requires an existing initialized "
                        "Agent libOS store"
                    )
            # A supported existing store is tightened only after its version
            # gate. A fresh database is created owner-only here.
            self._secure_database_files(
                canonical_path,
                create_if_missing=not existing_store,
            )
            if self._database_identity is None:
                database_stat = os.stat(canonical_path, follow_symlinks=False)
                self._database_identity = (
                    database_stat.st_dev,
                    database_stat.st_ino,
                )
            connection_path = str(canonical_path)
            if existing_store:
                # Re-open an existing store in explicit rw mode so a final
                # disappearance after preflight cannot make sqlite3 create an
                # unrelated empty database at the same pathname.
                connection_target = f"{canonical_path.as_uri()}?mode=rw"
                connection_uri = True
            else:
                connection_target = connection_path
            if fcntl is not None and hasattr(os, "O_NOFOLLOW"):
                self._lease_handle = self._acquire_runtime_lease(canonical_path)
            # SQLite's kernel-managed EXCLUSIVE lock is crash-recoverable and
            # complements the POSIX path/inode leases. Where those sidecars are
            # unavailable, including Windows, it is the sole runtime lease.
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                connection_target,
                check_same_thread=False,
                timeout=0.0 if use_exclusive_database_lease else 5.0,
                uri=connection_uri,
            )
            # Make the live handle visible to the state-aware cleanup path even
            # when setup fails before SQLRuntimeStore._init_store assigns it.
            self.conn = conn
            if self._lease_handle is not None:
                self._require_database_lease_identity(Path(connection_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if use_exclusive_database_lease:
                self._acquire_exclusive_sqlite_lease(conn, Path(connection_path))
            self._init_store(
                selected_path,
                config=config,
                conn=conn,
                initialize_schema=initialize_schema,
            )
        except BaseException as primary_error:
            cleanup_errors = self._cleanup_failed_initialization(conn)
            if not self._runtime_ownership_released():
                self._retain_failed_owner()
            if not cleanup_errors:
                raise
            raise BaseExceptionGroup(
                "SQLite store initialization and cleanup failed",
                [primary_error, *cleanup_errors],
            ) from None

    @classmethod
    def _retry_failed_owner(cls, identity: tuple[int, int]) -> None:
        """Retry a quarantined failed constructor before admitting a successor."""

        with cls._failed_owner_lock:
            owner = cls._failed_owners.get(identity)
            if owner is None:
                return
            errors = owner._cleanup_failed_initialization(
                getattr(owner, "conn", None)
            )
            if owner._runtime_ownership_released():
                if cls._failed_owners.get(identity) is owner:
                    cls._failed_owners.pop(identity, None)
            if errors:
                if len(errors) == 1:
                    raise errors[0]
                raise BaseExceptionGroup(
                    "previous failed SQLite owner cleanup failed",
                    errors,
                ) from None
            if not owner._runtime_ownership_released():
                raise ValidationError(
                    "previous failed SQLite owner still holds the database lease"
                )

    def _retain_failed_owner(self) -> None:
        identity = self._database_identity
        if identity is None:
            return
        with type(self)._failed_owner_lock:
            existing = type(self)._failed_owners.get(identity)
            if existing is not None and existing is not self:
                raise ValidationError(
                    "multiple failed SQLite owners claim the same database identity"
                )
            type(self)._failed_owners[identity] = self

    def _cleanup_failed_initialization(
        self,
        conn: Any | None,
    ) -> list[BaseException]:
        """Close a partial connection and release its lease only after closure."""

        errors: list[BaseException] = []
        if conn is not None and not self._sqlite_connection_reports_closed():
            try:
                conn.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                self._sqlite_connection_closed = True
            if self._sqlite_connection_reports_closed():
                self._sqlite_connection_closed = True
        if self._sqlite_connection_reports_closed():
            try:
                self._release_runtime_lease()
            except BaseException as exc:
                errors.append(exc)
        return errors

    def _preflight_existing_store(self, db_path: Path) -> bool:
        """Reject an incompatible store without opening the original in SQLite.

        A SQLite ``mode=ro`` connection is not physically read-only for a WAL
        database: SQLite may create or update ``-shm`` read marks while opening
        it.  ``immutable=1`` avoids that write, but also ignores an uncheckpointed
        WAL and can therefore inspect a stale schema marker.  Copy the validated
        database family into a private temporary directory and let SQLite apply
        any WAL or hot-journal recovery only to that disposable snapshot.
        """

        conn: sqlite3.Connection | None = None
        with tempfile.TemporaryDirectory(
            prefix="agent-libos-sqlite-preflight-"
        ) as snapshot_directory:
            snapshot_path = Path(snapshot_directory) / db_path.name
            self._copy_preflight_database_family(db_path, snapshot_path)
            try:
                conn = sqlite3.connect(
                    f"{snapshot_path.as_uri()}?mode=rw",
                    timeout=0.0,
                    uri=True,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA schema_version").fetchone()
                return self._require_supported_store_version_for(conn)
            except sqlite3.Error as exc:
                busy_codes = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                if getattr(exc, "sqlite_errorcode", None) in busy_codes:
                    raise ValidationError(
                        f"runtime store is already open: {db_path}"
                    ) from exc
                raise ValidationError(
                    f"unable to read SQLite store schema: {db_path}"
                ) from exc
            finally:
                if conn is not None:
                    conn.close()

    @classmethod
    @contextmanager
    def _migration_snapshot_connection(
        cls,
        path: Path,
        *,
        label: str,
        error_type: type[ValidationError],
        migration_label: str = "schema-v5",
    ) -> Iterator[Any]:
        """Open a disposable recovered snapshot for an offline migration probe."""

        helper = cls.__new__(cls)
        connection: sqlite3.Connection | None = None
        with tempfile.TemporaryDirectory(
            prefix="agent-libos-migration-preflight-"
        ) as directory:
            snapshot_path = Path(directory) / "store.sqlite"
            try:
                helper._copy_preflight_database_family(path, snapshot_path)
                connection = sqlite3.connect(
                    f"{snapshot_path.as_uri()}?mode=rw",
                    timeout=0.0,
                    uri=True,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA schema_version").fetchone()
                yield connection
            except sqlite3.Error as exc:
                raise error_type(
                    f"unable to inspect {label} for {migration_label} migration"
                ) from exc
            finally:
                if connection is not None:
                    connection.close()

    @classmethod
    @contextmanager
    def _migration_apply_connection(
        cls,
        path: Path,
        *,
        error_type: type[ValidationError],
        migration_label: str = "schema-v5",
    ) -> Iterator[Any]:
        """Open the source under the backend's exclusive offline migration lease."""

        helper = cls.__new__(cls)
        helper._lease_handle = None
        lease_acquired = False
        connection: sqlite3.Connection | None = None
        try:
            helper._secure_database_files(
                path,
                tighten=False,
                create_if_missing=False,
            )
            # On POSIX use the same pathname and inode leases as Runtime.open().
            # Other platforms retain SQLite's kernel-managed exclusive lock below.
            if fcntl is not None and hasattr(os, "O_NOFOLLOW"):
                helper._lease_handle = helper._acquire_runtime_lease(path)
                lease_acquired = True
            connection = sqlite3.connect(
                f"{path.as_uri()}?mode=rw",
                timeout=0.0,
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            if lease_acquired:
                helper._require_database_lease_identity(path)
            connection.execute("PRAGMA foreign_keys = ON")
            locking = connection.execute("PRAGMA locking_mode=EXCLUSIVE").fetchone()
            if locking is None or str(locking[0]).lower() != "exclusive":
                raise error_type(
                    f"SQLite refused exclusive migration lease mode: {path}"
                )
            connection.execute("BEGIN EXCLUSIVE")
            yield connection
        except sqlite3.Error as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            busy_codes = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
            if getattr(exc, "sqlite_errorcode", None) in busy_codes:
                raise error_type(f"runtime store is already open: {path}") from exc
            raise error_type(
                f"SQLite {migration_label} migration failed: {path}"
            ) from exc
        except BaseException:
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise
        finally:
            if connection is not None:
                connection.close()
            if lease_acquired:
                helper._release_runtime_lease()

    def _copy_preflight_database_family(
        self,
        db_path: Path,
        snapshot_path: Path,
    ) -> None:
        """Copy a stable, validated database and its recovery sidecars."""

        for suffix in ("", "-journal", "-wal", "-shm"):
            source = Path(f"{db_path}{suffix}")
            destination = Path(f"{snapshot_path}{suffix}")
            try:
                self._copy_preflight_file(source, destination)
            except FileNotFoundError as exc:
                if not suffix:
                    raise ValidationError(
                        "SQLite database path changed or disappeared while "
                        f"opening: {db_path}"
                    ) from exc

    def _copy_preflight_file(self, source: Path, destination: Path) -> None:
        """Copy one SQLite file and reject identity/content races."""

        source_flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(str(source), source_flags)
        try:
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValidationError(
                    f"SQLite preflight source must be a regular file: {source}"
                )
            self._require_owned_file(before, source, label="SQLite preflight source")
            self._require_single_link(before, source, label="SQLite preflight source")
            self._require_open_path_identity(source_fd, source)

            destination_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            destination_fd = os.open(str(destination), destination_flags, 0o600)
            try:
                while True:
                    block = os.read(source_fd, 1024 * 1024)
                    if not block:
                        break
                    remaining = memoryview(block)
                    while remaining:
                        written = os.write(destination_fd, remaining)
                        if written <= 0:  # pragma: no cover - defensive OS guard.
                            raise OSError("short write while copying SQLite preflight snapshot")
                        remaining = remaining[written:]
            finally:
                os.close(destination_fd)

            after = os.fstat(source_fd)
            self._require_open_path_identity(source_fd, source)
            before_fingerprint = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_fingerprint = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_fingerprint != before_fingerprint:
                raise ValidationError(
                    f"SQLite preflight source changed while copying: {source}"
                )
        finally:
            os.close(source_fd)

    def _require_open_path_identity(self, fd: int, path: Path) -> None:
        """Require ``path`` still names the regular file held by ``fd``."""

        opened_stat = os.fstat(fd)
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValidationError(
                f"unsafe SQLite preflight source changed while opening: {path}"
            ) from exc
        reparse_attribute = int(
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
        )
        path_attributes = int(getattr(path_stat, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_attributes & reparse_attribute
            or path_stat.st_dev != opened_stat.st_dev
            or path_stat.st_ino != opened_stat.st_ino
        ):
            raise ValidationError(
                f"unsafe SQLite preflight source changed while opening: {path}"
            )
        self._require_single_link(
            path_stat,
            path,
            label="SQLite preflight source",
        )

    @classmethod
    def _require_supported_store_version_for(cls, conn: Any) -> bool:
        row = conn.execute("PRAGMA encoding").fetchone()
        encoding = str(row["encoding"]) if row is not None else "missing"
        if encoding.upper() != "UTF-8":
            raise UnsupportedStoreVersion(
                "Agent libOS SQLite keyset ordering requires UTF-8 database "
                f"encoding; found {encoding}"
            )
        return super()._require_supported_store_version_for(conn)

    @classmethod
    def _probe_user_schema_objects(cls, conn: Any) -> set[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
        return {str(row["name"]) for row in rows}

    @classmethod
    def _probe_user_tables(cls, conn: Any) -> set[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        return {str(row["name"]) for row in rows}

    @classmethod
    def _require_v4_schema_shape(cls, conn: Any) -> None:
        """Require every manifest relation to have SQLite type ``table``."""

        required_tables = sorted(_V4_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master "
            f"WHERE name IN ({placeholders})",
            required_tables,
        )
        relation_types = {
            str(row["name"]): str(row["type"]).lower()
            for row in rows
        }
        invalid_relations = {
            table: relation_types.get(table, "missing")
            for table in required_tables
            if relation_types.get(table) != "table"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v4 "
                "manifest relation types: "
                f"{invalid_relations}; expected type 'table'"
            )
        super()._require_v4_schema_shape(conn)
        cls._require_v4_human_request_contract(conn)
        cls._require_full_schema_catalog(conn, version=4)

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
        """Require every schema-v5 manifest relation to be a SQLite table."""

        required_tables = sorted(_V5_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master "
            f"WHERE name IN ({placeholders})",
            required_tables,
        )
        relation_types = {
            str(row["name"]): str(row["type"]).lower()
            for row in rows
        }
        invalid_relations = {
            table: relation_types.get(table, "missing")
            for table in required_tables
            if relation_types.get(table) != "table"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v5 "
                "manifest relation types: "
                f"{invalid_relations}; expected type 'table'"
            )
        super()._require_v5_schema_shape(conn)
        cls._require_v5_storage_contract(conn)
        cls._require_full_schema_catalog(conn, version=5)

    @classmethod
    def _require_v6_schema_shape(cls, conn: Any) -> None:
        """Require every schema-v6 manifest relation to be a SQLite table."""

        required_tables = sorted(_V6_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master "
            f"WHERE name IN ({placeholders})",
            required_tables,
        )
        relation_types = {
            str(row["name"]): str(row["type"]).lower()
            for row in rows
        }
        invalid_relations = {
            table: relation_types.get(table, "missing")
            for table in required_tables
            if relation_types.get(table) != "table"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v6 "
                "manifest relation types: "
                f"{invalid_relations}; expected type 'table'"
            )
        super()._require_v6_schema_shape(conn)
        cls._require_v5_storage_contract(conn)
        cls._require_v6_storage_contract(conn)
        cls._require_full_schema_catalog(conn, version=6)

    @classmethod
    def _require_v7_schema_shape(cls, conn: Any) -> None:
        """Require every schema-v7 manifest relation to be a SQLite table."""

        required_tables = sorted(_V7_REQUIRED_COLUMNS)
        placeholders = ", ".join("?" for _ in required_tables)
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master "
            f"WHERE name IN ({placeholders})",
            required_tables,
        )
        relation_types = {
            str(row["name"]): str(row["type"]).lower()
            for row in rows
        }
        invalid_relations = {
            table: relation_types.get(table, "missing")
            for table in required_tables
            if relation_types.get(table) != "table"
        }
        if invalid_relations:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS store schema v7 "
                "manifest relation types: "
                f"{invalid_relations}; expected type 'table'"
            )
        super()._require_v7_schema_shape(conn)
        cls._require_v5_storage_contract(conn)
        cls._require_v6_storage_contract(conn)
        cls._require_v7_storage_contract(conn)
        cls._require_full_schema_catalog(conn, version=7)

    @classmethod
    def _require_full_schema_catalog(cls, conn: Any, *, version: int) -> None:
        """Require every SQLite schema object to match the canonical catalog."""

        # A canonical reference is itself initialized through SQLiteStore.  Its
        # ordinary focused validators still run, but the thread-local guard
        # prevents the full comparator from recursively asking for a reference.
        if bool(getattr(_SQLITE_CANONICAL_CATALOG_BUILD, "active", False)):
            return
        expected = cls._canonical_full_schema_catalog(version)
        actual = _sqlite_full_catalog_snapshot(conn)
        if actual == expected:
            return
        raise UnsupportedStoreVersion(
            f"unsupported Agent libOS SQLite schema v{version} full catalog: "
            f"{_sqlite_catalog_difference(expected, actual)}; "
            f"expected_sha256={_sqlite_catalog_sha256(expected)}, "
            f"actual_sha256={_sqlite_catalog_sha256(actual)}"
        )

    @classmethod
    def _canonical_full_schema_catalog(cls, version: int) -> dict[str, Any]:
        if version not in {4, 5, 6, 7}:
            raise UnsupportedStoreVersion(
                f"unsupported Agent libOS SQLite schema catalog version: {version}"
            )
        cached = _SQLITE_CANONICAL_CATALOGS.get(version)
        if cached is not None:
            return cached
        with _SQLITE_CANONICAL_CATALOG_LOCK:
            cached = _SQLITE_CANONICAL_CATALOGS.get(version)
            if cached is not None:
                return cached
            _SQLITE_CANONICAL_CATALOG_BUILD.active = True
            reference: SQLiteStore | None = None
            try:
                # Build through the exact runtime DDL path, never by accepting
                # the target's definitions as the reference.  The v4 catalog
                # is the canonical v5 base with only the explicit 4->5 delta
                # reversed, matching the supported offline migration source.
                reference = SQLiteStore(":memory:")
                if version in {4, 5, 6}:
                    for table in sorted(V7_TABLES):
                        reference.conn.execute(f'DROP TABLE "{table}"')
                    changed = reference.conn.execute(
                        "UPDATE runtime_schema SET schema_version = 6 "
                        "WHERE singleton = 1 AND schema_version = 7"
                    )
                    if changed.rowcount != 1:
                        raise UnsupportedStoreVersion(
                            "unable to construct canonical SQLite schema-v6 catalog"
                        )
                if version in {4, 5}:
                    for table in sorted(V6_TABLES):
                        reference.conn.execute(f'DROP TABLE "{table}"')
                    changed = reference.conn.execute(
                        "UPDATE runtime_schema SET schema_version = 5 "
                        "WHERE singleton = 1 AND schema_version = 6"
                    )
                    if changed.rowcount != 1:
                        raise UnsupportedStoreVersion(
                            "unable to construct canonical SQLite schema-v5 catalog"
                        )
                if version == 4:
                    reference.conn.execute("DROP TABLE semantic_assessments")
                    reference.conn.execute("DROP TABLE semantic_assessment_jobs")
                    reference.conn.execute(
                        "ALTER TABLE human_requests DROP COLUMN revision"
                    )
                    changed = reference.conn.execute(
                        "UPDATE runtime_schema SET schema_version = 4 "
                        "WHERE singleton = 1 AND schema_version = 5"
                    )
                    if changed.rowcount != 1:
                        raise UnsupportedStoreVersion(
                            "unable to construct canonical SQLite schema-v4 catalog"
                        )
                    reference.conn.commit()
                catalog = _sqlite_full_catalog_snapshot(reference.conn)
                actual_digest = _sqlite_catalog_sha256(catalog)
                if actual_digest != _SQLITE_CANONICAL_CATALOG_SHA256[version]:
                    raise UnsupportedStoreVersion(
                        "runtime DDL no longer synthesizes the canonical "
                        f"Agent libOS SQLite schema-v{version} catalog"
                    )
            finally:
                try:
                    if reference is not None:
                        reference.close()
                finally:
                    _SQLITE_CANONICAL_CATALOG_BUILD.active = False
            _SQLITE_CANONICAL_CATALOGS[version] = catalog
            return catalog

    @classmethod
    def _require_v6_storage_contract(cls, conn: Any) -> None:
        sqlite_keys = dict(V6_STORAGE_KEY_CONSTRAINTS)
        # SQLite implements an exact INTEGER PRIMARY KEY as the rowid and does
        # not expose a corresponding PRAGMA index_list row.  The full catalog
        # comparator still verifies the canonical PRIMARY KEY declaration.
        sqlite_keys["semantic_control_state"] = ()
        sqlite_keys["semantic_legacy_coverage"] = ()
        problems = {
            **cls._storage_column_problems(conn, V6_STORAGE_COLUMN_CONTRACTS),
            **cls._storage_check_problems(conn, V6_STORAGE_SQLITE_CHECKS),
            **cls._storage_key_constraint_problems(
                conn,
                sqlite_keys,
            ),
        }
        if problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v6 storage contract: "
                f"{problems}"
            )

    @classmethod
    def _require_v7_storage_contract(cls, conn: Any) -> None:
        problems = {
            **cls._storage_column_problems(conn, V7_STORAGE_COLUMN_CONTRACTS),
            **cls._storage_check_problems(conn, V7_STORAGE_SQLITE_CHECKS),
            **cls._storage_key_constraint_problems(
                conn,
                V7_STORAGE_KEY_CONSTRAINTS,
            ),
        }
        if problems:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS schema v7 storage contract: "
                f"{problems}"
            )

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
            V5_STORAGE_SQLITE_CHECKS,
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
        problems: dict[str, Any] = {}
        for table, contract_items in contracts.items():
            rows = [
                dict(row) for row in conn.execute(f"PRAGMA table_xinfo({table})")
            ]
            selected_rows = rows
            expected_names = tuple(name for name, _ in contract_items)
            actual_names = tuple(str(row["name"]) for row in selected_rows)
            if actual_names != expected_names:
                problems[f"{table}.columns"] = {
                    "expected": expected_names,
                    "actual": actual_names,
                }
                continue
            ddl_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            ddl = str(ddl_row["sql"]) if ddl_row and ddl_row["sql"] else ""
            collations = _sqlite_column_collations(ddl)
            for row, (name, contract) in zip(selected_rows, contract_items):
                actual = {
                    "type": str(row["type"]).casefold(),
                    "nullable": not bool(row["notnull"]),
                    "default": _normalized_sqlite_default(row["dflt_value"]),
                    "primary_key_position": int(row["pk"]),
                    "hidden": int(row["hidden"]),
                    "collation": (
                        collations.get(name) if contract.sql_type == "text" else None
                    ),
                }
                expected = {
                    "type": contract.sql_type,
                    "nullable": (
                        True
                        if table == "human_requests" and name == "request_id"
                        else contract.nullable
                    ),
                    "default": contract.default,
                    "primary_key_position": contract.primary_key_position,
                    "hidden": 0,
                    "collation": (
                        cls.KEYSET_TEXT_COLLATION
                        if contract.sql_type == "text"
                        else None
                    ),
                }
                if table != "human_requests":
                    actual["ordinal"] = int(row["cid"]) + 1
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
        problems: dict[str, Any] = {}
        for table, expected_expressions in contracts.items():
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            ddl = str(row["sql"]) if row and row["sql"] else ""
            actual = Counter(_sqlite_check_expressions(ddl))
            expected = Counter(
                _sqlite_contract_expression(expression)
                for expression in expected_expressions
            )
            if actual != expected:
                problems[table] = {
                    "expected_count": sum(expected.values()),
                    "actual_count": sum(actual.values()),
                    "missing": sum((expected - actual).values()),
                    "extra": sum((actual - expected).values()),
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
        problems: dict[str, Any] = {}
        for table, expected_constraints in contracts.items():
            actual: Counter[tuple[str, tuple[str, ...]]] = Counter()
            for index_row in conn.execute(f"PRAGMA index_list({table})"):
                origin = str(index_row["origin"])
                if origin == "c":
                    continue
                name = str(index_row["name"])
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                    actual[("invalid_index", (name,))] += 1
                    continue
                key_rows = [
                    row
                    for row in conn.execute(f"PRAGMA index_xinfo({name})")
                    if int(row["key"]) == 1
                ]
                key_rows.sort(key=lambda row: int(row["seqno"]))
                columns = tuple(
                    str(row["name"])
                    if row["name"] is not None
                    else "<expression>"
                    for row in key_rows
                )
                kind = {
                    "pk": "primary_key",
                    "u": "unique",
                }.get(origin, f"unknown:{origin}")
                actual[(kind, columns)] += 1
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
        problems: dict[str, Any] = {}
        for table in guarded_tables:
            foreign_keys = [dict(row) for row in conn.execute(
                f"PRAGMA foreign_key_list({table})"
            )]
            if foreign_keys:
                problems[f"{table}.foreign_keys"] = len(foreign_keys)
            ddl_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            ddl = str(ddl_row["sql"]) if ddl_row and ddl_row["sql"] else ""
            tokens = _sqlite_schema_tokens(ddl)
            for left, right in zip(tokens, tokens[1:]):
                if (
                    left[0] == "word"
                    and left[1].casefold() == "on"
                    and right[0] == "word"
                    and right[1].casefold() == "conflict"
                ):
                    problems[f"{table}.conflict_clause"] = "noncanonical"
                    break
        placeholders = ", ".join("?" for _ in guarded_tables)
        trigger_rows = list(
            conn.execute(
                "SELECT name, tbl_name FROM sqlite_master "
                "WHERE type = 'trigger' "
                f"AND tbl_name IN ({placeholders})",
                guarded_tables,
            )
        )
        if trigger_rows:
            problems["triggers"] = sorted(
                f"{row['tbl_name']}.{row['name']}" for row in trigger_rows
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
            "SELECT name, sql FROM sqlite_master "
            f"WHERE type = 'table' AND name IN ({placeholders})",
            tables,
        )
        ddl_by_table = {
            str(row["name"]): str(row["sql"])
            for row in rows
        }
        result: dict[tuple[str, str], str] = {}
        for table, columns in keyset_columns.items():
            ddl = ddl_by_table.get(table)
            if ddl is None:
                continue
            declared_collations = _sqlite_column_collations(ddl)
            for column in columns:
                collation = declared_collations.get(column)
                if collation is not None:
                    result[(table, column)] = collation
        return result

    @classmethod
    def _probe_index_shapes(
        cls,
        conn: Any,
        tables: set[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        shapes: dict[str, dict[str, Any]] = {}
        for table in sorted(tables):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table) is None:
                raise UnsupportedStoreVersion("invalid table identity in v4 manifest")
            for row in conn.execute(f"PRAGMA index_list({table})"):
                name = str(row["name"])
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                    raise UnsupportedStoreVersion(
                        "invalid index identity in v4 manifest"
                    )
                key_rows = [
                    item
                    for item in conn.execute(f"PRAGMA index_xinfo({name})")
                    if int(item["key"]) == 1
                ]
                key_rows.sort(key=lambda item: int(item["seqno"]))
                definition = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (name,),
                ).fetchone()
                definition_sql = (
                    str(definition["sql"])
                    if definition is not None and definition["sql"] is not None
                    else ""
                )
                predicate_match = re.search(
                    r"\bWHERE\b(?P<predicate>.*)\Z",
                    definition_sql,
                    re.IGNORECASE | re.DOTALL,
                )
                shapes[name] = {
                    "table": table,
                    "columns": tuple(str(item["name"]) for item in key_rows),
                    "unique": bool(row["unique"]),
                    "partial": bool(row["partial"]),
                    "descending": tuple(bool(item["desc"]) for item in key_rows),
                    "collations": tuple(
                        str(item["coll"]).upper() if item["coll"] is not None else None
                        for item in key_rows
                    ),
                    "origin": (
                        "declared" if str(row["origin"]) == "c" else "constraint"
                    ),
                    "predicate": cls._canonical_index_predicate(
                        predicate_match.group("predicate")
                        if predicate_match is not None
                        else None
                    ),
                }
        return shapes

    def close(self) -> None:
        errors: list[BaseException] = []
        if not self._sqlite_connection_reports_closed():
            try:
                super().close()
            except BaseException as exc:
                errors.append(exc)
            else:
                self._sqlite_connection_closed = True
            if self._sqlite_connection_reports_closed():
                self._sqlite_connection_closed = True

        # A file lease remains the authoritative ownership barrier if closing
        # the SQLite connection failed while it was still open. Releasing it in
        # that state would let a successor start beside a retryable old owner.
        if self._sqlite_connection_reports_closed():
            try:
                self._release_runtime_lease()
            except BaseException as exc:
                errors.append(exc)

        if (
            self._sqlite_connection_reports_closed()
            and getattr(self, "_lease_handle", None) is None
        ):
            self._backend_ownership_release_observed = True

        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("SQLite store cleanup failed", errors) from None

    def _sqlite_connection_reports_closed(self) -> bool:
        if getattr(self, "_sqlite_connection_closed", False):
            return True
        # sqlite3.Connection has no public closed flag, so successful close is
        # tracked above. Test doubles and alternate DB-API adapters may expose
        # one, which also makes a close-that-raised-after-closing observable.
        conn = getattr(self, "conn", None)
        if getattr(conn, "closed", None) is True:
            return True
        if conn is None:
            return True
        # CPython sqlite3 exposes no ``closed`` flag. Reading ``in_transaction``
        # is a side-effect-free driver state probe: it returns a bool while the
        # handle is live and raises ProgrammingError only after sqlite3_close
        # has irreversibly detached it. This also covers an adapter that closes
        # the real connection and then raises a diagnostic from ``close()``.
        try:
            conn.in_transaction
        except sqlite3.ProgrammingError:
            return True
        except BaseException:
            return False
        return False

    def _runtime_ownership_released(self) -> bool:
        return (
            self._sqlite_connection_reports_closed()
            and getattr(self, "_lease_handle", None) is None
        )

    def _acquire_runtime_lease(self, db_path: Path) -> _SQLiteRuntimeLease:
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        if fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            raise ValidationError("secure file runtime leases require fcntl and O_NOFOLLOW")
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(str(lease_path), flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(f"unsafe runtime lease path: {lease_path}") from exc
            raise ValidationError(f"unable to securely open runtime lease: {lease_path}") from exc

        handle: Any | None = None
        try:
            opened_stat = os.fstat(fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValidationError(f"runtime lease must be a regular file: {lease_path}")
            self._require_owned_file(opened_stat, lease_path, label="runtime lease")
            self._require_single_link(opened_stat, lease_path, label="runtime lease")
            os.fchmod(fd, 0o600)
            opened_stat = os.fstat(fd)
            self._require_single_link(opened_stat, lease_path, label="runtime lease")
            handle = os.fdopen(fd, "r+", encoding="utf-8")
            fd = -1
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ValidationError(f"runtime store is already open: {db_path}") from exc
                raise ValidationError(f"unable to lock runtime lease: {lease_path}") from exc

            path_stat = os.stat(lease_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_dev != opened_stat.st_dev
                or path_stat.st_ino != opened_stat.st_ino
            ):
                raise ValidationError(f"unsafe runtime lease path changed while opening: {lease_path}")
            self._require_single_link(path_stat, lease_path, label="runtime lease")

            identity_handle, identity_path, database_identity = (
                self._acquire_database_identity_lease(db_path)
            )

            handle.seek(0)
            handle.truncate()
            handle.write(f"{utc_now()}\n{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            return _SQLiteRuntimeLease(
                handle,
                lease_path,
                identity_handle=identity_handle,
                identity_path=identity_path,
                database_identity=database_identity,
            )
        except BaseException:
            identity_handle = locals().get("identity_handle")
            if identity_handle is not None:
                identity_handle.close()
            if handle is not None:
                handle.close()
            elif fd >= 0:
                os.close(fd)
            raise

    def _acquire_database_identity_lease(
        self,
        db_path: Path,
    ) -> tuple[Any, Path, tuple[int, int]]:
        """Lock a private sidecar keyed by the validated database inode."""

        if fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            raise ValidationError("secure database identity leases require fcntl and O_NOFOLLOW")
        flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(str(db_path), flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(f"unsafe SQLite database path: {db_path}") from exc
            raise ValidationError(
                f"unable to securely open SQLite database lease: {db_path}"
            ) from exc
        try:
            self._tighten_open_file(fd, db_path, label="SQLite database")
            opened_stat = os.fstat(fd)
            identity = (int(opened_stat.st_dev), int(opened_stat.st_ino))
        finally:
            os.close(fd)

        identity_directory = self._secure_database_identity_lease_directory()
        identity_path = identity_directory / f"{identity[0]:x}-{identity[1]:x}.lock"
        identity_flags = (
            os.O_CREAT
            | os.O_RDWR
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            identity_fd = os.open(str(identity_path), identity_flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(
                    f"unsafe SQLite database identity lease: {identity_path}"
                ) from exc
            raise ValidationError(
                f"unable to securely open SQLite database identity lease: {identity_path}"
            ) from exc
        identity_handle: Any | None = None
        try:
            opened_identity_stat = os.fstat(identity_fd)
            if not stat.S_ISREG(opened_identity_stat.st_mode):
                raise ValidationError(
                    f"SQLite database identity lease must be a regular file: {identity_path}"
                )
            self._require_owned_file(
                opened_identity_stat,
                identity_path,
                label="SQLite database identity lease",
            )
            self._require_single_link(
                opened_identity_stat,
                identity_path,
                label="SQLite database identity lease",
            )
            os.fchmod(identity_fd, 0o600)
            identity_handle = os.fdopen(identity_fd, "r+", encoding="utf-8")
            identity_fd = -1
            try:
                fcntl.flock(
                    identity_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ValidationError(f"runtime store is already open: {db_path}") from exc
                raise ValidationError(
                    f"unable to lock SQLite database identity lease: {identity_path}"
                ) from exc
            path_stat = os.stat(identity_path, follow_symlinks=False)
            current_identity_stat = os.fstat(identity_handle.fileno())
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_dev != current_identity_stat.st_dev
                or path_stat.st_ino != current_identity_stat.st_ino
            ):
                raise ValidationError(
                    "unsafe SQLite database identity lease changed while opening: "
                    f"{identity_path}"
                )
            self._require_single_link(
                path_stat,
                identity_path,
                label="SQLite database identity lease",
            )
            identity_handle.seek(0)
            identity_handle.truncate()
            identity_handle.write(
                f"{utc_now()}\n{os.getpid()}\n{db_path}\n{identity[0]}:{identity[1]}\n"
            )
            identity_handle.flush()
            os.fsync(identity_handle.fileno())
            return identity_handle, identity_path, identity
        except BaseException:
            if identity_handle is not None:
                identity_handle.close()
            elif identity_fd >= 0:
                os.close(identity_fd)
            raise

    def _secure_database_identity_lease_directory(self) -> Path:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        directory = Path(tempfile.gettempdir()).resolve() / f"agent-libos-sqlite-leases-{uid}"
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(str(directory), flags)
        except OSError as exc:
            raise ValidationError(
                f"unable to securely open SQLite identity lease directory: {directory}"
            ) from exc
        try:
            opened_stat = os.fstat(fd)
            if not stat.S_ISDIR(opened_stat.st_mode):
                raise ValidationError(
                    f"SQLite identity lease directory must be a directory: {directory}"
                )
            self._require_owned_file(
                opened_stat,
                directory,
                label="SQLite identity lease directory",
            )
            os.fchmod(fd, 0o700)
            path_stat = os.stat(directory, follow_symlinks=False)
            if (
                not stat.S_ISDIR(path_stat.st_mode)
                or path_stat.st_dev != opened_stat.st_dev
                or path_stat.st_ino != opened_stat.st_ino
            ):
                raise ValidationError(
                    f"unsafe SQLite identity lease directory changed while opening: {directory}"
                )
        finally:
            os.close(fd)
        return directory

    def _require_database_lease_identity(self, db_path: Path) -> None:
        lease = self._lease_handle
        if lease is None:
            return
        try:
            path_stat = os.stat(db_path, follow_symlinks=False)
        except OSError as exc:
            raise ValidationError(
                f"unsafe SQLite database path changed while opening: {db_path}"
            ) from exc
        expected_device, expected_inode = lease.database_identity
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != expected_device
            or path_stat.st_ino != expected_inode
        ):
            raise ValidationError(
                f"unsafe SQLite database path changed while opening: {db_path}"
            )
        self._require_single_link(path_stat, db_path, label="SQLite database")

    def _secure_database_files(
        self,
        db_path: Path,
        *,
        tighten: bool = True,
        create_if_missing: bool = False,
    ) -> None:
        """Validate SQLite files, optionally creating/tightening them to 0600."""
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "fchmod"):
            self._secure_database_file_portable(
                db_path,
                tighten=tighten,
                create_if_missing=create_if_missing,
            )
            return
        flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(str(db_path), flags)
        except FileNotFoundError as exc:
            if not create_if_missing:
                raise ValidationError(
                    f"SQLite database path changed or disappeared while opening: {db_path}"
                ) from exc
            try:
                fd = os.open(str(db_path), flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                fd = os.open(str(db_path), flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(f"unsafe SQLite database path: {db_path}") from exc
            raise
        try:
            self._tighten_open_file(
                fd,
                db_path,
                label="SQLite database",
                tighten=tighten,
            )
        finally:
            os.close(fd)
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            try:
                sidecar_fd = os.open(str(sidecar), flags)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                    raise ValidationError(f"unsafe SQLite sidecar path: {sidecar}") from exc
                raise
            try:
                self._tighten_open_file(
                    sidecar_fd,
                    sidecar,
                    label="SQLite sidecar",
                    tighten=tighten,
                )
            finally:
                os.close(sidecar_fd)

    def _secure_database_file_portable(
        self,
        db_path: Path,
        *,
        tighten: bool,
        create_if_missing: bool,
    ) -> None:
        """Create and identity-check a database where POSIX no-follow APIs are absent."""

        flags = (
            os.O_RDWR
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(str(db_path), flags)
        except FileNotFoundError as exc:
            if not create_if_missing:
                raise ValidationError(
                    f"SQLite database path changed or disappeared while opening: {db_path}"
                ) from exc
            try:
                fd = os.open(
                    str(db_path),
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                # Another actor won the create race. Open that exact pathname
                # and apply the same descriptor/path identity checks below.
                fd = os.open(str(db_path), flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(f"unsafe SQLite database path: {db_path}") from exc
            raise
        try:
            opened_stat = os.fstat(fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValidationError(
                    f"SQLite database must be a regular file: {db_path}"
                )
            self._require_owned_file(
                opened_stat,
                db_path,
                label="SQLite database",
            )
            self._require_single_link(
                opened_stat,
                db_path,
                label="SQLite database",
            )
            if tighten and hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
                opened_stat = os.fstat(fd)
            try:
                path_stat = os.stat(db_path, follow_symlinks=False)
            except OSError as exc:
                raise ValidationError(
                    f"unsafe SQLite database path changed while opening: {db_path}"
                ) from exc
            reparse_attribute = int(
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
            )
            path_attributes = int(getattr(path_stat, "st_file_attributes", 0))
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path_attributes & reparse_attribute
                or path_stat.st_dev != opened_stat.st_dev
                or path_stat.st_ino != opened_stat.st_ino
            ):
                raise ValidationError(
                    f"unsafe SQLite database path changed while opening: {db_path}"
                )
            self._require_single_link(
                path_stat,
                db_path,
                label="SQLite database",
            )
        finally:
            os.close(fd)

    def _tighten_open_file(
        self,
        fd: int,
        path: Path,
        *,
        label: str,
        tighten: bool = True,
    ) -> None:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValidationError(f"{label} must be a regular file: {path}")
        self._require_owned_file(opened_stat, path, label=label)
        self._require_single_link(opened_stat, path, label=label)
        if tighten:
            os.fchmod(fd, 0o600)
            opened_stat = os.fstat(fd)
            self._require_single_link(opened_stat, path, label=label)
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValidationError(f"unsafe {label} path changed while opening: {path}") from exc
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != opened_stat.st_dev
            or path_stat.st_ino != opened_stat.st_ino
        ):
            raise ValidationError(f"unsafe {label} path changed while opening: {path}")
        self._require_single_link(path_stat, path, label=label)

    def _require_owned_file(self, opened_stat: os.stat_result, path: Path, *, label: str) -> None:
        if hasattr(os, "getuid") and opened_stat.st_uid != os.getuid():
            raise ValidationError(f"{label} is not owned by the current user: {path}")

    @staticmethod
    def _require_single_link(opened_stat: os.stat_result, path: Path, *, label: str) -> None:
        if opened_stat.st_nlink != 1:
            raise ValidationError(f"{label} must not have hard links: {path}")

    def _acquire_exclusive_sqlite_lease(self, conn: sqlite3.Connection, db_path: Path) -> None:
        try:
            row = conn.execute("PRAGMA locking_mode=EXCLUSIVE").fetchone()
            if row is None or str(row[0]).lower() != "exclusive":
                raise ValidationError(f"SQLite refused exclusive runtime lease mode: {db_path}")
            conn.execute("BEGIN EXCLUSIVE")
            conn.commit()
        except sqlite3.Error as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            busy_codes = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
            if getattr(exc, "sqlite_errorcode", None) in busy_codes:
                raise ValidationError(f"runtime store is already open: {db_path}") from exc
            raise ValidationError(f"unable to acquire SQLite runtime lease: {db_path}") from exc

    def _release_runtime_lease(self) -> None:
        lease = getattr(self, "_lease_handle", None)
        if lease is None:
            return
        handle = lease.handle
        try:
            handle.close()
        except BaseException as exc:
            close_error: BaseException | None = exc
        else:
            close_error = None

        # Closing the descriptor is the single irreversible lease release
        # point. An explicit LOCK_UN before close would create an ambiguous
        # acknowledgement window: unlock may have taken effect even if both
        # that call and the later close report diagnostics. File handles expose
        # whether close crossed its release point, including close-then-raise
        # adapters used by alternate runtimes.
        if close_error is not None and not getattr(handle, "closed", False):
            raise close_error

        identity_handle = lease.identity_handle
        try:
            identity_handle.close()
        except BaseException as exc:
            identity_close_error: BaseException | None = exc
        else:
            identity_close_error = None

        if (
            getattr(handle, "closed", False)
            and getattr(identity_handle, "closed", False)
        ):
            self._lease_handle = None
        errors = [
            error
            for error in (close_error, identity_close_error)
            if error is not None
        ]
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("SQLite runtime lease cleanup failed", errors) from None
