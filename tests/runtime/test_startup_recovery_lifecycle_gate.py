from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.evidence.external_effects import reconcile_pending_external_effects
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.checkpoint_manager import CheckpointManager
from agent_libos.storage.repositories import ObjectRepository


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_checkpoint_payload_recovery_precedes_missing_payload_sweep(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    original_checkpoint_recovery = (
        CheckpointManager.recover_incomplete_restore_publications
    )
    original_payload_sweep = ObjectRepository.recover_missing_runtime_object_payloads

    def recover_checkpoints(manager: CheckpointManager) -> list[str]:
        order.append("checkpoint_restore")
        return original_checkpoint_recovery(manager)

    def sweep_missing_payloads(
        repository: ObjectRepository,
        *,
        require_recovery_lease: Callable[[], None],
        retained_read_capabilities: Any = None,
    ) -> object:
        order.append("missing_payload_sweep")
        return original_payload_sweep(
            repository,
            require_recovery_lease=require_recovery_lease,
            retained_read_capabilities=retained_read_capabilities,
        )

    monkeypatch.setattr(
        CheckpointManager,
        "recover_incomplete_restore_publications",
        recover_checkpoints,
    )
    monkeypatch.setattr(
        ObjectRepository,
        "recover_missing_runtime_object_payloads",
        sweep_missing_payloads,
    )
    with _runtime_for_backend(backend, tmp_path):
        assert order == ["checkpoint_restore", "missing_payload_sweep"]


class _LeaseInterrupt(BaseException):
    pass

_RECOVERY_TABLES = (
    "runtime_publications",
    "processes",
    "operations",
    "external_effects",
    "capability_use_reservations",
    "capabilities",
    "resource_usage_reservations",
    "events",
    "audit_records",
    "operation_evidence",
    "objects",
    "object_links",
    "object_tasks",
)


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_open_runtime_rejects_all_startup_recovery_entries_before_first_read(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(backend, tmp_path) as runtime:
        before = _recovery_rows(runtime)
        reads: list[str] = []

        def forbidden_read(name: str) -> Callable[..., Any]:
            def fail(*_args: Any, **_kwargs: Any) -> Any:
                reads.append(name)
                raise AssertionError(f"recovery entry read durable state: {name}")

            return fail

        monkeypatch.setattr(
            runtime.uow.protected_effects,
            "query_external_effect_recovery",
            forbidden_read("external_effects"),
        )
        monkeypatch.setattr(
            runtime.uow.authority._authority_recovery_backend,
            "abandon_stale_capability_use_reservations",
            forbidden_read("capability_reservations"),
        )
        monkeypatch.setattr(
            runtime.resources.resource_repository,
            "query_resource_usage_reservation_recovery",
            forbidden_read("resource_reservations"),
        )
        monkeypatch.setattr(
            runtime.process.publications,
            "query_runtime_publication_recovery",
            forbidden_read("process_publications"),
        )
        monkeypatch.setattr(
            runtime.operations.store,
            "stale_operation_recovery_index",
            forbidden_read("stale_operations"),
        )
        monkeypatch.setattr(
            runtime.uow.processes._process_backend,
            "recover_stale_executions",
            forbidden_read("stale_executions"),
        )
        monkeypatch.setattr(
            runtime.store,
            "_query_object_payload_recovery_page",
            forbidden_read("object_payloads"),
        )
        monkeypatch.setattr(
            runtime.object_tasks._records,
            "query_object_task_recovery",
            forbidden_read("object_tasks"),
        )

        entries: tuple[Callable[[], Any], ...] = (
            lambda: runtime.uow.objects.recover_missing_runtime_object_payloads(
                require_recovery_lease=runtime.lifecycle.require_recovery_lease,
            ),
            lambda: runtime.store.recover_missing_runtime_object_payloads(
                require_recovery_lease=runtime.lifecycle.require_recovery_lease,
            ),
            runtime.protected_operations.recover_prepared,
            lambda: reconcile_pending_external_effects(
                runtime.uow.protected_effects,
                runtime.substrate,
                require_recovery_lease=runtime.lifecycle.require_recovery_lease,
            ),
            lambda: runtime.uow.authority.abandon_stale_capability_use_reservations(
                require_recovery_lease=runtime.lifecycle.require_recovery_lease,
            ),
            runtime.resources.recover_usage_reservations,
            runtime.process.recover_incomplete_publications,
            runtime.operations.interrupt_stale_running,
            lambda: runtime.uow.processes.recover_stale_executions(
                owner_id=runtime.instance_id,
                require_recovery_lease=runtime.lifecycle.require_recovery_lease,
                on_recovered=lambda _pid: None,
            ),
            runtime.object_tasks.recover,
        )
        for entry in entries:
            with pytest.raises(RuntimeError, match="active startup recovery lease"):
                entry()

        assert reads == []
        assert _recovery_rows(runtime) == before


def test_process_recovery_propagates_lease_interrupt_before_read_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "startup-recovery-interrupt.sqlite")
    interrupt = _LeaseInterrupt("injected recovery lease interruption")
    before = _recovery_rows(runtime)
    read_attempted = False

    def interrupt_lease() -> None:
        raise interrupt

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal read_attempted
        read_attempted = True
        raise AssertionError("process recovery read before its lifecycle gate")

    try:
        monkeypatch.setattr(runtime.process, "_require_recovery_lease", interrupt_lease)
        monkeypatch.setattr(
            runtime.process.publications,
            "query_runtime_publication_recovery",
            forbidden_read,
        )
        with pytest.raises(_LeaseInterrupt) as caught:
            runtime.process.recover_incomplete_publications()
        assert caught.value is interrupt
        assert not read_attempted
        assert _recovery_rows(runtime) == before
    finally:
        runtime.close()


def _recovery_rows(runtime: Runtime) -> dict[str, list[dict[str, Any]]]:
    return {
        table: [dict(row) for row in runtime.store.select_table_rows(table)]
        for table in _RECOVERY_TABLES
    }


@contextlib.contextmanager
def _runtime_for_backend(
    backend: str,
    tmp_path: Path,
) -> Iterator[Runtime]:
    if backend == "sqlite":
        runtime = Runtime.open(tmp_path / "startup-recovery-gate.sqlite")
        try:
            yield runtime
        finally:
            runtime.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        runtime = Runtime.open(dsn, config=config)
        try:
            yield runtime
        finally:
            runtime.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_recovery_gate_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
