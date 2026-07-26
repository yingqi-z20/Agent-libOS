from __future__ import annotations

import errno
import os
import re
import sqlite3
import stat
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Mapping

from agent_libos.config import AgentLibOSConfig
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.storage.sql import SQLRuntimeStore, _V3_KEYSET_TEXT_COLUMNS
from agent_libos.utils.ids import utc_now

try:  # pragma: no cover - Windows fallback is exercised only on non-POSIX hosts.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


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
            else:
                # SQLite's kernel-managed EXCLUSIVE lock is crash-recoverable,
                # unlike a create-once fallback lockfile that can survive its
                # owner indefinitely.  This is also the Windows path.
                use_database_lease = True
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                connection_target,
                check_same_thread=False,
                timeout=0.0 if use_database_lease else 5.0,
                uri=connection_uri,
            )
            # Make the live handle visible to the state-aware cleanup path even
            # when setup fails before SQLRuntimeStore._init_store assigns it.
            self.conn = conn
            if self._lease_handle is not None:
                self._require_database_lease_identity(Path(connection_path))
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
            if not create_if_missing and not db_path.exists():
                raise ValidationError(
                    f"SQLite database path changed or disappeared while opening: {db_path}"
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
