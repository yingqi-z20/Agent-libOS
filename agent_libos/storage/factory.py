from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import SplitResult, unquote, unquote_plus, urlsplit, urlunsplit

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.base import RuntimeStore
from agent_libos.storage.sqlite import SQLiteStore

POSTGRES_SCHEMES = {"postgres", "postgresql"}
SQLITE_SCHEME = "sqlite"
USER_STORE_TARGET = "user"
USER_STORE_DIRECTORY = (".agent-libos", "runtime")
USER_STORE_FILENAME = "agent-libos.sqlite"
_LIBPQ_DSN_FIELD = re.compile(
    r"(?:^|\s)(?:dbname|host|hostaddr|options|password|port|service|sslmode|target_session_attrs|user)\s*=",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "code",
        "credential",
        "credentials",
        "key",
        "pass",
        "passfile",
        "passwd",
        "password",
        "pwd",
        "secret",
        "sig",
        "signature",
        "sslkey",
        "token",
        "user",
        "username",
    }
)
_SENSITIVE_QUERY_KEY_SUFFIXES = (
    "_api_key",
    "_auth",
    "_credential",
    "_credentials",
    "_key",
    "_pass",
    "_passwd",
    "_password",
    "_secret",
    "_sig",
    "_signature",
    "_token",
    "_user",
    "_username",
)
_SENSITIVE_QUERY_KEY_COMPACT_SUFFIXES = (
    "apikey",
    "credential",
    "credentials",
    "passwd",
    "password",
    "secret",
    "signature",
    "token",
)


@dataclass(frozen=True, slots=True)
class ResolvedStoreTarget:
    """One normalized runtime-store selection shared by every factory caller."""

    backend: Literal["sqlite", "postgres"]
    connection_target: str
    display_target: str
    uses_user_directory: bool = False
    selected_sqlite_path: Path | None = None
    user_store_home: Path | None = None

    @property
    def persistent_sqlite_path(self) -> Path | None:
        if self.backend != "sqlite" or self.connection_target == ":memory:":
            return None
        return Path(self.connection_target)


def open_store(
    target: str | Path | ResolvedStoreTarget | None = None,
    *,
    config: AgentLibOSConfig | None = None,
) -> RuntimeStore:
    return _open_store(target, config=config, initialize_schema=True)


def open_store_for_migration(
    target: str | Path | ResolvedStoreTarget | None = None,
    *,
    config: AgentLibOSConfig | None = None,
) -> RuntimeStore:
    """Open an existing canonical store without running schema initialization."""

    return _open_store(target, config=config, initialize_schema=False)


def _open_store(
    target: str | Path | ResolvedStoreTarget | None,
    *,
    config: AgentLibOSConfig | None,
    initialize_schema: bool,
) -> RuntimeStore:
    selected_config = config or DEFAULT_CONFIG
    resolved = (
        target
        if isinstance(target, ResolvedStoreTarget)
        else resolve_store_target(target, config=selected_config)
    )
    if resolved.backend == "postgres":
        from agent_libos.storage.postgres import PostgresStore

        return PostgresStore(
            resolved.connection_target,
            config=selected_config,
            initialize_schema=initialize_schema,
        )
    if resolved.backend == "sqlite":
        if resolved.uses_user_directory:
            _ensure_secure_user_store_directory(
                resolved.user_store_home,
                create_missing=initialize_schema,
            )
            _validate_user_store_leaf(
                resolved.selected_sqlite_path or _user_store_path(),
            )
        store = SQLiteStore(
            resolved.connection_target,
            config=selected_config,
            initialize_schema=initialize_schema,
            _frozen_target=True,
        )
        if resolved.selected_sqlite_path is not None:
            # The connection target is deliberately canonical and frozen, but
            # caller-owned attachment must also retain the original lexical
            # spelling so a model-visible workspace alias cannot be forgotten.
            store.lexical_path = str(resolved.selected_sqlite_path)
        return store
    raise ValidationError(f"unsupported runtime store backend: {resolved.backend}")


def display_store_target(target: str | Path | None = None, *, config: AgentLibOSConfig | None = None) -> str:
    return resolve_store_target(target, config=config).display_target


def resolve_store_target(
    target: str | Path | None = None,
    *,
    config: AgentLibOSConfig | None = None,
    base_directory: str | Path | None = None,
) -> ResolvedStoreTarget:
    """Resolve one target without opening a database or mutating the filesystem."""

    selected_config = config or DEFAULT_CONFIG
    selected_target = _selected_target(target, selected_config)
    backend = _backend_for(
        selected_target,
        selected_config,
        explicit=target is not None,
    )
    if backend == "postgres":
        dsn = _postgres_dsn(selected_target, selected_config)
        return ResolvedStoreTarget(
            backend="postgres",
            connection_target=dsn,
            display_target=redact_store_target(dsn),
        )
    if backend != "sqlite":
        raise ValidationError(f"unsupported runtime store backend: {backend}")

    uses_user_directory = str(selected_target) == USER_STORE_TARGET
    user_store_home = _resolved_user_store_home() if uses_user_directory else None
    raw_sqlite_target = (
        str(_user_store_path(user_store_home))
        if uses_user_directory
        else _sqlite_target(selected_target)
    )
    lexical_sqlite_path = (
        None
        if raw_sqlite_target == ":memory:"
        else _absolute_sqlite_target(
            raw_sqlite_target,
            base_directory=base_directory,
        )
    )
    sqlite_target = (
        raw_sqlite_target
        if raw_sqlite_target == ":memory:"
        else _canonical_sqlite_target(lexical_sqlite_path)
    )
    return ResolvedStoreTarget(
        backend="sqlite",
        connection_target=sqlite_target,
        display_target=(sqlite_target if uses_user_directory else str(selected_target)),
        uses_user_directory=uses_user_directory,
        selected_sqlite_path=lexical_sqlite_path,
        user_store_home=user_store_home,
    )


def validate_store_target_workspace_isolation(
    target: str | Path | None,
    *,
    workspace_root: str | Path,
    config: AgentLibOSConfig | None = None,
    base_directory: str | Path | None = None,
) -> ResolvedStoreTarget:
    """Reject a persistent SQLite target within a model-visible workspace."""

    resolved = resolve_store_target(
        target,
        config=config,
        base_directory=base_directory,
    )
    paths = (
        resolved.selected_sqlite_path,
        resolved.persistent_sqlite_path,
    )
    for path in paths:
        if path is not None:
            _validate_sqlite_workspace_isolation(path, workspace_root=workspace_root)
    return resolved


def validate_runtime_store_workspace_isolation(
    store: RuntimeStore,
    *,
    workspace_root: str | Path,
) -> None:
    """Apply the same boundary to a caller-owned concrete SQLite store."""

    if not isinstance(store, SQLiteStore):
        return
    paths = (
        getattr(store, "lexical_path", None),
        getattr(store, "canonical_path", None),
    )
    for selected_path in paths:
        if selected_path in {None, ":memory:"}:
            continue
        path = Path(str(selected_path))
        _validate_sqlite_workspace_isolation(
            path,
            workspace_root=workspace_root,
        )


def _validate_sqlite_workspace_isolation(
    path: Path,
    *,
    workspace_root: str | Path,
) -> None:
    """Compare ancestor identities so aliases cannot defeat containment."""

    try:
        workspace = Path(workspace_root).resolve(strict=True)
        workspace_stat = os.stat(workspace)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(
            "cannot verify local workspace identity for runtime-store isolation"
        ) from exc
    if not stat.S_ISDIR(workspace_stat.st_mode):
        raise ValidationError(
            "cannot verify local workspace identity for runtime-store isolation"
        )

    try:
        current = _absolute_sqlite_target(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(
            "cannot verify persistent SQLite runtime-store isolation"
        ) from exc

    while True:
        try:
            current_stat = os.stat(current)
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise ValidationError(
                    "cannot verify persistent SQLite runtime-store isolation"
                )
            current = parent
            continue
        except (NotADirectoryError, OSError, ValueError) as exc:
            raise ValidationError(
                "cannot verify persistent SQLite runtime-store isolation"
            ) from exc
        break

    workspace_identity = (workspace_stat.st_dev, workspace_stat.st_ino)
    while True:
        if (current_stat.st_dev, current_stat.st_ino) == workspace_identity:
            raise ValidationError(
                "persistent SQLite runtime store must be outside the local workspace"
            )
        parent = current.parent
        if parent == current:
            return
        current = parent
        try:
            current_stat = os.stat(current)
        except (OSError, ValueError) as exc:
            raise ValidationError(
                "cannot verify persistent SQLite runtime-store isolation"
            ) from exc


def _canonical_sqlite_target(target: str | Path) -> str:
    """Freeze a persistent SQLite spelling before isolation validation/open."""

    try:
        return str(Path(target).resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError(
            "cannot canonicalize persistent SQLite runtime-store target"
        ) from exc


def _absolute_sqlite_target(
    target: str | Path,
    *,
    base_directory: str | Path | None = None,
) -> Path:
    """Freeze an absolute lexical spelling without following path aliases."""

    candidate = Path(target)
    if candidate.is_absolute():
        return Path(os.path.abspath(os.fspath(candidate)))
    base = Path(base_directory) if base_directory is not None else Path.cwd()
    return Path(os.path.abspath(os.fspath(base / candidate)))


def _resolved_user_store_home() -> Path:
    try:
        return Path.home().expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError("cannot establish secure user runtime-store directory") from exc


def _user_store_path(home: Path | None = None) -> Path:
    return (home or _resolved_user_store_home()).joinpath(
        *USER_STORE_DIRECTORY,
        USER_STORE_FILENAME,
    )


def _ensure_secure_user_store_directory(
    home: Path | None,
    *,
    create_missing: bool,
) -> None:
    if home is None:
        raise ValidationError("cannot establish secure user runtime-store directory")
    try:
        home = Path(home).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError("cannot establish secure user runtime-store directory") from exc
    if (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
    ):
        _ensure_secure_posix_user_store_directory(
            home,
            create_missing=create_missing,
        )
        return
    _ensure_secure_portable_user_store_directory(
        home,
        create_missing=create_missing,
    )


def _ensure_secure_posix_user_store_directory(
    home: Path,
    *,
    create_missing: bool,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        parent_fd = os.open(home, flags)
    except OSError as exc:
        raise ValidationError("cannot establish secure user runtime-store directory") from exc
    try:
        for component in USER_STORE_DIRECTORY:
            if create_missing:
                try:
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
            child_flags = flags | getattr(os, "O_NOFOLLOW", 0)
            try:
                child_fd = os.open(component, child_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                if not create_missing:
                    return
                raise ValidationError(
                    "cannot establish secure user runtime-store directory"
                ) from None
            except OSError as exc:
                raise ValidationError(
                    "user runtime-store directory contains a symlink or unsafe component"
                ) from exc
            try:
                metadata = os.fstat(child_fd)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValidationError(
                        "user runtime-store directory contains a non-directory component"
                    )
                getuid = getattr(os, "getuid", None)
                if callable(getuid) and metadata.st_uid != getuid():
                    raise ValidationError(
                        "user runtime-store directory must be owned by the current user"
                    )
                os.fchmod(child_fd, 0o700)
                tightened = os.fstat(child_fd)
                if stat.S_IMODE(tightened.st_mode) != 0o700:
                    raise ValidationError(
                        "user runtime-store directory must have mode 0700"
                    )
            except BaseException:
                os.close(child_fd)
                raise
            os.close(parent_fd)
            parent_fd = child_fd
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("cannot establish secure user runtime-store directory") from exc
    finally:
        os.close(parent_fd)


def _ensure_secure_portable_user_store_directory(
    home: Path,
    *,
    create_missing: bool,
) -> None:
    current = home
    for component in USER_STORE_DIRECTORY:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not create_missing:
                return
            try:
                current.mkdir(mode=0o700)
                metadata = os.lstat(current)
            except OSError as exc:
                raise ValidationError(
                    "cannot establish secure user runtime-store directory"
                ) from exc
        except OSError as exc:
            raise ValidationError(
                "cannot establish secure user runtime-store directory"
            ) from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
            or _is_path_junction(current)
        ):
            raise ValidationError(
                "user runtime-store directory contains a symlink or unsafe component"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValidationError(
                "user runtime-store directory contains a non-directory component"
            )
        if os.name == "posix":
            getuid = getattr(os, "getuid", None)
            if callable(getuid) and metadata.st_uid != getuid():
                raise ValidationError(
                    "user runtime-store directory must be owned by the current user"
                )
            os.chmod(current, 0o700, follow_symlinks=False)
            if stat.S_IMODE(os.lstat(current).st_mode) != 0o700:
                raise ValidationError(
                    "user runtime-store directory must have mode 0700"
                )


def _is_path_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


def _validate_user_store_leaf(path: Path) -> None:
    """Reject an existing alias at the reserved database pathname."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValidationError("cannot verify secure user runtime-store path") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or _is_path_junction(path)
    ):
        raise ValidationError(
            "user runtime-store database path contains a symlink or unsafe component"
        )


def redact_store_target(target: str | Path) -> str:
    text = str(target)
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in POSTGRES_SCHEMES or not parsed.netloc:
        return text
    _userinfo, separator, hostinfo = parsed.netloc.rpartition("@")
    netloc = f"***@{hostinfo}" if separator else parsed.netloc
    return urlunsplit(
        SplitResult(
            parsed.scheme,
            netloc,
            parsed.path,
            _redact_sensitive_query(parsed.query),
            "",
        )
    )


def _redact_sensitive_query(query: str) -> str:
    if not query:
        return ""
    fields = re.split(r"([&;])", query)
    for index in range(0, len(fields), 2):
        field = fields[index]
        raw_key, separator, _raw_value = field.partition("=")
        if _is_sensitive_query_key(raw_key):
            fields[index] = f"{raw_key}{separator or '='}***"
    return "".join(fields)


def _is_sensitive_query_key(raw_key: str) -> bool:
    decoded = unquote_plus(raw_key).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", decoded).strip("_")
    return (
        normalized in _SENSITIVE_QUERY_KEY_NAMES
        or normalized.endswith(_SENSITIVE_QUERY_KEY_SUFFIXES)
        or normalized.endswith(_SENSITIVE_QUERY_KEY_COMPACT_SUFFIXES)
    )


def _selected_target(target: str | Path | None, config: AgentLibOSConfig) -> str | Path:
    if target is not None:
        return target
    if config.runtime.store_backend == "postgres":
        if not config.runtime.store_dsn:
            raise ValidationError("PostgreSQL runtime store requires runtime.store_dsn")
        return config.runtime.store_dsn
    return config.runtime.local_store_target


def _backend_for(target: str | Path, config: AgentLibOSConfig, *, explicit: bool = False) -> str:
    text = str(target)
    parsed = urlsplit(text)
    scheme = parsed.scheme.lower()
    inferred_backend: str | None = None
    if scheme in POSTGRES_SCHEMES:
        if "://" not in text:
            raise ValidationError("PostgreSQL runtime store targets must use a postgres:// or postgresql:// URI")
        inferred_backend = "postgres"
    elif scheme == SQLITE_SCHEME:
        inferred_backend = "sqlite"
    elif "://" in text:
        raise ValidationError(f"unsupported runtime store target scheme: {scheme or '<missing>'}")
    elif _LIBPQ_DSN_FIELD.search(text):
        raise ValidationError(
            "libpq keyword DSNs are not supported as runtime store targets; "
            "use a postgres:// or postgresql:// URI"
        )
    if inferred_backend is not None:
        if not explicit and inferred_backend != config.runtime.store_backend:
            raise ValidationError(
                "runtime store target conflicts with runtime.store_backend: "
                f"target selects {inferred_backend}, config selects {config.runtime.store_backend}"
            )
        return inferred_backend
    if explicit:
        return "sqlite"
    return config.runtime.store_backend


def _postgres_dsn(target: str | Path, config: AgentLibOSConfig) -> str:
    text = str(target)
    parsed = urlsplit(text)
    if parsed.scheme.lower() in POSTGRES_SCHEMES:
        return text
    if config.runtime.store_dsn:
        return config.runtime.store_dsn
    raise ValidationError("PostgreSQL runtime store requires a postgresql:// DSN")


def _sqlite_target(target: str | Path) -> str:
    text = str(target)
    parsed = urlsplit(text)
    if parsed.scheme.lower() == SQLITE_SCHEME:
        if parsed.netloc and parsed.path:
            return unquote(f"//{parsed.netloc}{parsed.path}")
        if parsed.path:
            path = unquote(parsed.path)
            if not parsed.netloc and path.startswith("//"):
                path = f"/{path.lstrip('/')}"
            if path.startswith("/") and len(path) > 2 and path[2] == ":":
                return path[1:]
            return path
        return ":memory:"
    return ":memory:" if text in {"local", ":memory:"} else text
