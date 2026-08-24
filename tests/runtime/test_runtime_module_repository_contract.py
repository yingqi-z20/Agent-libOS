from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    RuntimeModule,
    RuntimeModuleRegistration,
    RuntimeModuleStatus,
)
from agent_libos.storage import PostgresStore, SQLiteStore, UnitOfWork


BACKENDS = [
    "sqlite-memory",
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


def test_registry_projects_typed_module_record_to_public_dictionary() -> None:
    runtime = Runtime.open("local")
    try:
        persisted = runtime.uow.module_publications.get_runtime_module(
            "agent-libos-core:v0"
        )

        assert isinstance(persisted, RuntimeModule)
        assert runtime.modules.inspect_module(
            "agent-libos-core:v0"
        ) == persisted.to_public_dict()
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_runtime_module_repository_is_typed_and_transactional(
    backend: str,
    tmp_path: Path,
) -> None:
    with _unit_of_work_for_backend(backend, tmp_path) as unit:
        loaded = _runtime_module("typed-module:v0")

        persisted = unit.module_publications.upsert_runtime_module(loaded)

        assert isinstance(persisted, RuntimeModule)
        assert persisted.status == RuntimeModuleStatus.LOADED
        assert persisted.registered.tools == ("typed_tool",)
        assert persisted.updated_at is not None
        assert unit.module_publications.get_runtime_module(loaded.module_id) == persisted
        assert unit.module_publications.list_runtime_modules(limit=1) == [persisted]

        failed = replace(
            persisted,
            status=RuntimeModuleStatus.FAILED,
            loaded_at=None,
            registered=RuntimeModuleRegistration(),
            error="module failed",
        )
        failed = unit.module_publications.upsert_runtime_module(failed)
        assert failed.status == RuntimeModuleStatus.FAILED
        assert failed.error == "module failed"

        with pytest.raises(RuntimeError, match="roll back module publication"):
            with unit.transaction():
                unit.module_publications.upsert_runtime_module(
                    _runtime_module("rolled-back-module:v0")
                )
                raise RuntimeError("roll back module publication")

        assert (
            unit.module_publications.get_runtime_module("rolled-back-module:v0")
            is None
        )


def _runtime_module(module_id: str) -> RuntimeModule:
    return RuntimeModule(
        module_id=module_id,
        name="Typed module",
        version="v0",
        entrypoint="typed.module:register",
        manifest_path="/modules/typed/module.yaml",
        manifest_sha256="1" * 64,
        source_path="/modules/typed/module.py",
        source_sha256="2" * 64,
        status=RuntimeModuleStatus.LOADED,
        loaded_at="2026-01-01T00:00:00Z",
        registered=RuntimeModuleRegistration(tools=("typed_tool",)),
        metadata={"contract": True},
    )


@contextlib.contextmanager
def _unit_of_work_for_backend(
    backend: str,
    tmp_path: Path,
) -> Iterator[UnitOfWork]:
    postgres_context = contextlib.nullcontext(None)
    if backend == "sqlite-memory":
        store = SQLiteStore(":memory:")
    elif backend == "sqlite-file":
        store = SQLiteStore(tmp_path / "module-publication.sqlite")
    elif backend == "postgres":
        postgres_context = _postgres_schema_dsn()
        store = None
    else:
        raise AssertionError(f"unknown backend: {backend}")

    with postgres_context as dsn:
        if dsn is not None:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            store = PostgresStore(dsn, config=config)
        assert store is not None
        try:
            yield UnitOfWork(store)
        finally:
            store.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_module_publication_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
