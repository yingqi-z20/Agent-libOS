from __future__ import annotations

import errno
import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Any, Mapping

from agent_libos.config import AgentLibOSConfig
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.sql import SQLRuntimeStore, _V3_KEYSET_TEXT_COLUMNS
from agent_libos.utils.ids import utc_now

try:  # pragma: no cover - Windows fallback is exercised only on non-POSIX hosts.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class _SQLiteRuntimeLease:
    def __init__(self, handle: Any, path: Path) -> None:
        self.handle = handle
        self.path = path


class SQLiteStore(SQLRuntimeStore):
    """SQLite runtime store backend.

    Connection setup, file hardening, and lease behavior remain SQLite-only;
    backend-neutral repositories live in :class:`SQLRuntimeStore`.
    """

    KEYSET_TEXT_COLLATION = "BINARY"

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        config: AgentLibOSConfig | None = None,
        initialize_schema: bool = True,
    ) -> None:
        selected_path = str(path)
        connection_path = selected_path
        self._lease_handle: Any | None = None
        self._sqlite_connection_closed = False
        use_database_lease = False
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
            if canonical_path.exists():
                fresh_store = self._preflight_existing_store(canonical_path)
                if not initialize_schema and fresh_store:
                    raise ValidationError(
                        "offline migration requires an existing initialized "
                        "Agent libOS store"
                    )
            else:
                if not initialize_schema:
                    raise ValidationError(
                        "offline migration requires an existing initialized "
                        "Agent libOS store"
                    )
                db_path.parent.mkdir(parents=True, exist_ok=True)
            connection_path = str(canonical_path)
            self._secure_database_files(canonical_path)
            if fcntl is not None and hasattr(os, "O_NOFOLLOW"):
                self._lease_handle = self._acquire_runtime_lease(canonical_path)
            else:
                # SQLite's kernel-managed EXCLUSIVE lock is crash-recoverable,
                # unlike a create-once fallback lockfile that can survive its
                # owner indefinitely.  This is also the Windows path.
                use_database_lease = True
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                connection_path,
                check_same_thread=False,
                timeout=0.0 if use_database_lease else 5.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if use_database_lease:
                self._acquire_exclusive_sqlite_lease(conn, Path(connection_path))
            self._init_store(
                selected_path,
                config=config,
                conn=conn,
                initialize_schema=initialize_schema,
            )
        except BaseException:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._release_runtime_lease()
            raise

    def _preflight_existing_store(self, db_path: Path) -> bool:
        """Reject an incompatible store through a read-only connection."""

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
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
    def _probe_text_column_collations(
        cls,
        conn: Any,
    ) -> Mapping[tuple[str, str], str]:
        tables = sorted(_V3_KEYSET_TEXT_COLUMNS)
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
        for table, columns in _V3_KEYSET_TEXT_COLUMNS.items():
            ddl = ddl_by_table.get(table)
            if ddl is None:
                continue
            for column in columns:
                declaration = re.search(
                    rf"(?im)^\s*[\"`\[]?{re.escape(column)}[\"`\]]?\s+TEXT\b(?P<tail>[^,\n]*)",
                    ddl,
                )
                if declaration is None:
                    continue
                explicit = re.search(
                    r"\bCOLLATE\s+(?:[\"`\[])?(?P<name>[A-Za-z0-9_.-]+)",
                    declaration.group("tail"),
                    re.IGNORECASE,
                )
                # SQLite TEXT columns use BINARY without an explicit clause.
                result[(table, column)] = (
                    explicit.group("name").upper()
                    if explicit is not None
                    else "BINARY"
                )
        return result

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
            os.fchmod(fd, 0o600)
            opened_stat = os.fstat(fd)
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

            handle.seek(0)
            handle.truncate()
            handle.write(f"{utc_now()}\n{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            return _SQLiteRuntimeLease(handle, lease_path)
        except BaseException:
            if handle is not None:
                handle.close()
            elif fd >= 0:
                os.close(fd)
            raise

    def _secure_database_files(self, db_path: Path) -> None:
        """Create/tighten the SQLite database and existing sidecars to 0600."""
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "fchmod"):
            return
        flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(str(db_path), flags)
        except FileNotFoundError:
            try:
                fd = os.open(str(db_path), flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                fd = os.open(str(db_path), flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EISDIR}:
                raise ValidationError(f"unsafe SQLite database path: {db_path}") from exc
            raise
        try:
            self._tighten_open_file(fd, db_path, label="SQLite database")
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
                self._tighten_open_file(sidecar_fd, sidecar, label="SQLite sidecar")
            finally:
                os.close(sidecar_fd)

    def _tighten_open_file(self, fd: int, path: Path, *, label: str) -> None:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValidationError(f"{label} must be a regular file: {path}")
        self._require_owned_file(opened_stat, path, label=label)
        os.fchmod(fd, 0o600)
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != opened_stat.st_dev
            or path_stat.st_ino != opened_stat.st_ino
        ):
            raise ValidationError(f"unsafe {label} path changed while opening: {path}")

    def _require_owned_file(self, opened_stat: os.stat_result, path: Path, *, label: str) -> None:
        if hasattr(os, "getuid") and opened_stat.st_uid != os.getuid():
            raise ValidationError(f"{label} is not owned by the current user: {path}")

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
        if getattr(handle, "closed", False):
            self._lease_handle = None
        if close_error is not None:
            raise close_error
