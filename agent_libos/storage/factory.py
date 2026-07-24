from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.base import RuntimeStore
from agent_libos.storage.sqlite import SQLiteStore

POSTGRES_SCHEMES = {"postgres", "postgresql"}
SQLITE_SCHEME = "sqlite"
_LIBPQ_DSN_FIELD = re.compile(
    r"(?:^|\s)(?:dbname|host|hostaddr|options|password|port|service|sslmode|target_session_attrs|user)\s*=",
    re.IGNORECASE,
)


def open_store(
    target: str | Path | None = None,
    *,
    config: AgentLibOSConfig | None = None,
) -> RuntimeStore:
    return _open_store(target, config=config, initialize_schema=True)


def open_store_for_migration(
    target: str | Path | None = None,
    *,
    config: AgentLibOSConfig | None = None,
) -> RuntimeStore:
    """Open an existing canonical store without running schema initialization."""

    return _open_store(target, config=config, initialize_schema=False)


def _open_store(
    target: str | Path | None,
    *,
    config: AgentLibOSConfig | None,
    initialize_schema: bool,
) -> RuntimeStore:
    selected_config = config or DEFAULT_CONFIG
    selected_target = _selected_target(target, selected_config)
    backend = _backend_for(selected_target, selected_config, explicit=target is not None)
    if backend == "postgres":
        from agent_libos.storage.postgres import PostgresStore

        dsn = _postgres_dsn(selected_target, selected_config)
        return PostgresStore(
            dsn,
            config=selected_config,
            initialize_schema=initialize_schema,
        )
    if backend == "sqlite":
        return SQLiteStore(
            _sqlite_target(selected_target),
            config=selected_config,
            initialize_schema=initialize_schema,
        )
    raise ValidationError(f"unsupported runtime store backend: {backend}")


def display_store_target(target: str | Path | None = None, *, config: AgentLibOSConfig | None = None) -> str:
    selected_config = config or DEFAULT_CONFIG
    selected_target = _selected_target(target, selected_config)
    backend = _backend_for(selected_target, selected_config, explicit=target is not None)
    if backend == "postgres":
        return redact_store_target(_postgres_dsn(selected_target, selected_config))
    return str(selected_target)


def redact_store_target(target: str | Path) -> str:
    text = str(target)
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in POSTGRES_SCHEMES or not parsed.netloc:
        return text
    userinfo, sep, hostinfo = parsed.netloc.rpartition("@")
    if not sep:
        return text
    user, colon, password = userinfo.partition(":")
    if not colon or not password:
        return text
    return urlunsplit(SplitResult(parsed.scheme, f"{user}:***@{hostinfo}", parsed.path, parsed.query, parsed.fragment))


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
