from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pytest

from agent_libos.runtime import (
    RuntimeAssemblyCleanupKind,
    RuntimeAssemblyCleanupRequired,
)
from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.runtime.checkpoint_manager import CheckpointManager
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import SQLiteStore


def _contains_cancellation(error: BaseException) -> bool:
    if isinstance(error, asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_cancellation(item) for item in error.exceptions)
    return False


def test_async_open_yields_caller_loop_while_opening_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_loop_advanced = threading.Event()
    observed_loop_progress: list[bool] = []

    def blocking_open_store(*_args: object, **_kwargs: object) -> SQLiteStore:
        observed_loop_progress.append(caller_loop_advanced.wait(timeout=1.0))
        raise RuntimeError("injected open-store failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        blocking_open_store,
    )

    async def exercise() -> None:
        asyncio.get_running_loop().call_soon(caller_loop_advanced.set)
        with pytest.raises(RuntimeError, match="injected open-store failure"):
            await Runtime.aopen("local")

    asyncio.run(exercise())

    assert observed_loop_progress == [True]


def test_async_open_keeps_store_open_and_assembly_on_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    open_threads: list[int] = []
    recovery_threads: list[int] = []

    def tracked_open_store(*_args: object, **_kwargs: object) -> SQLiteStore:
        open_threads.append(threading.get_ident())
        return store

    def fail_during_recovery(_host: Runtime) -> None:
        recovery_threads.append(threading.get_ident())
        raise RuntimeError("injected same-worker recovery failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        tracked_open_store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(fail_during_recovery),
    )

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError,
            match="injected same-worker recovery failure",
        ):
            await Runtime.aopen("local")

    try:
        asyncio.run(exercise())
    finally:
        if not store._runtime_ownership_released():
            store.close()

    assert len(open_threads) == 1
    assert recovery_threads == open_threads


def test_cancelled_async_open_closes_store_returned_by_open_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "cancelled-open-store.sqlite"
    store = SQLiteStore(database)
    open_entered = threading.Event()
    allow_open_to_finish = threading.Event()

    def delayed_open_store(*_args: object, **_kwargs: object) -> SQLiteStore:
        open_entered.set()
        allow_open_to_finish.wait(timeout=1.0)
        return store

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        delayed_open_store,
    )

    async def exercise() -> tuple[BaseException | None, bool]:
        opening = asyncio.create_task(Runtime.aopen("local"))
        assert await asyncio.to_thread(open_entered.wait, 2.0)
        opening.cancel()
        allow_open_to_finish.set()

        caught: BaseException | None = None
        leaked_runtime: Runtime | None = None
        try:
            leaked_runtime = await opening
        except BaseException as error:
            caught = error
        released_before_fallback = store._runtime_ownership_released()
        if leaked_runtime is not None:
            await leaked_runtime.ashutdown(reason="test fallback cleanup")
        return caught, released_before_fallback

    try:
        caught, released_before_fallback = asyncio.run(exercise())
    finally:
        if not store._runtime_ownership_released():
            store.close()

    assert caught is not None
    assert _contains_cancellation(caught)
    assert store._runtime_assembly_reservation is None
    assert released_before_fallback is True
    reopened = SQLiteStore(database)
    reopened.close()


def test_cancelled_successful_async_assembly_normally_shuts_down_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "cancelled-successful-assembly.sqlite"
    store = SQLiteStore(database)
    original_assemble = RuntimeBuilder._assemble_host
    assembly_succeeded = threading.Event()
    allow_assembly_to_return = threading.Event()
    assembled_hosts: list[Runtime] = []

    monkeypatch.setattr(
        CheckpointManager,
        "recover_incomplete_restore_publications",
        lambda _self: [],
    )
    monkeypatch.setattr(
        CheckpointManager,
        "_prepare_startup_payload_delivery",
        lambda _self: (),
    )
    monkeypatch.setattr(
        CheckpointManager,
        "_complete_startup_payload_delivery",
        lambda _self, _publication_ids: None,
    )

    def delayed_assemble(
        _cls: type[RuntimeBuilder[Runtime]],
        host: Runtime,
        selected_store: SQLiteStore,
        **kwargs: object,
    ) -> None:
        original_assemble(host, selected_store, **kwargs)  # type: ignore[arg-type]
        assembled_hosts.append(host)
        assembly_succeeded.set()
        allow_assembly_to_return.wait(timeout=1.0)

    monkeypatch.setattr(
        RuntimeBuilder,
        "_assemble_host",
        classmethod(delayed_assemble),
    )

    async def exercise() -> tuple[BaseException | None, bool]:
        opening = asyncio.create_task(
            RuntimeBuilder.configured(Runtime).afrom_store(store)
        )
        assert await asyncio.to_thread(assembly_succeeded.wait, 2.0)
        opening.cancel()
        allow_assembly_to_return.set()

        caught: BaseException | None = None
        leaked_runtime: Runtime | None = None
        try:
            leaked_runtime = await opening
        except BaseException as error:
            caught = error
        released_before_fallback = store._runtime_ownership_released()
        if leaked_runtime is not None:
            await leaked_runtime.ashutdown(reason="test fallback cleanup")
        return caught, released_before_fallback

    try:
        caught, released_before_fallback = asyncio.run(exercise())
    finally:
        if not store._runtime_ownership_released():
            store.close()

    assert caught is not None
    assert _contains_cancellation(caught)
    assert released_before_fallback is True
    assert len(assembled_hosts) == 1
    assert assembled_hosts[0].lifecycle.closed is True
    assert assembled_hosts[0].lifecycle._active_leases == 0
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.parametrize(
    ("failure_mode", "release_mode"),
    [
        ("exception", "async"),
        ("false_result", "async"),
        pytest.param("exception", "cancelled_async", id="cancelled-release"),
        pytest.param(
            "exception",
            "cancelled_incomplete",
            id="cancelled-incomplete-release",
        ),
        pytest.param("exception", "sync", id="sync-release"),
    ],
)
def test_cancelled_open_publishes_retriable_normal_shutdown_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    release_mode: str,
) -> None:
    database = tmp_path / f"cancelled-open-{failure_mode}.sqlite"
    store = SQLiteStore(database)
    original_assemble = RuntimeBuilder._assemble_host
    original_ashutdown = Runtime.ashutdown
    assembly_succeeded = threading.Event()
    allow_assembly_to_return = threading.Event()
    assembled_hosts: list[Runtime] = []
    shutdown_calls = 0
    scheduler_shutdown_calls = 0
    retry_shutdown_entered = asyncio.Event()
    allow_retry_shutdown = asyncio.Event()

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    if failure_mode == "exception":

        async def fail_first_ashutdown(
            host: Runtime,
            *,
            actor: str = "runtime",
            reason: str = "runtime.shutdown",
        ) -> dict[str, object]:
            nonlocal shutdown_calls
            shutdown_calls += 1
            if shutdown_calls == 1:
                raise RuntimeError("injected cancelled-open shutdown failure")
            if (
                shutdown_calls == 2
                and release_mode in {"cancelled_async", "cancelled_incomplete"}
            ):
                retry_shutdown_entered.set()
                await allow_retry_shutdown.wait()
                if release_mode == "cancelled_incomplete":
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": "injected incomplete retry",
                    }
            return await original_ashutdown(host, actor=actor, reason=reason)

        monkeypatch.setattr(Runtime, "ashutdown", fail_first_ashutdown)

    def delayed_assemble(
        _cls: type[RuntimeBuilder[Runtime]],
        host: Runtime,
        selected_store: SQLiteStore,
        **kwargs: object,
    ) -> None:
        nonlocal scheduler_shutdown_calls
        original_assemble(host, selected_store, **kwargs)  # type: ignore[arg-type]
        if failure_mode == "false_result":
            original_scheduler_shutdown = host.scheduler.shutdown

            def fail_first_scheduler_shutdown() -> bool:
                nonlocal scheduler_shutdown_calls
                scheduler_shutdown_calls += 1
                if scheduler_shutdown_calls == 1:
                    return False
                return original_scheduler_shutdown()

            host.scheduler.shutdown = fail_first_scheduler_shutdown  # type: ignore[method-assign]
        assembled_hosts.append(host)
        assembly_succeeded.set()
        allow_assembly_to_return.wait(timeout=1.0)

    monkeypatch.setattr(
        RuntimeBuilder,
        "_assemble_host",
        classmethod(delayed_assemble),
    )

    async def exercise() -> RuntimeAssemblyCleanupRequired:
        opening = asyncio.create_task(Runtime.aopen("local"))
        assert await asyncio.to_thread(assembly_succeeded.wait, 2.0)
        opening.cancel()
        allow_assembly_to_return.set()

        with pytest.raises(BaseExceptionGroup) as caught:
            await opening
        assert _contains_cancellation(caught.value)
        handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
        assert len(handles) == 1
        handle = handles[0]
        assert handle.cleanup_kind is RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN
        assert handle.partial_runtime is assembled_hosts[0]
        assert handle.cleanup_completed is False
        assert handle.released is False
        assert handle.cleanup_errors[0]["component"] == "runtime_shutdown"
        assert assembled_hosts[0].lifecycle.closed is False
        assert store._runtime_ownership_released() is False

        if release_mode in {"cancelled_async", "cancelled_incomplete"}:
            release = asyncio.create_task(handle.arelease())
            await retry_shutdown_entered.wait()
            release.cancel()
            allow_retry_shutdown.set()
            with pytest.raises(BaseExceptionGroup) as release_error:
                await release
            assert _contains_cancellation(release_error.value)
            if release_mode == "cancelled_incomplete":
                assert RuntimeAssemblyCleanupRequired.extract(
                    release_error.value
                ) == (handle,)
                assert handle.released is False
                assert store._runtime_ownership_released() is False
                await handle.arelease()
        elif release_mode == "async":
            await handle.arelease()
        return handle

    try:
        handle = asyncio.run(exercise())
        if release_mode == "sync":
            handle.release()
            handle.release()
        else:
            # A discharged ownership handle remains safely idempotent.
            asyncio.run(handle.arelease())
        assert handle.cleanup_completed is True
        assert handle.released is True
        assert assembled_hosts[0].lifecycle.closed is True
        assert store._runtime_ownership_released() is True
    finally:
        allow_assembly_to_return.set()
        if not store._runtime_ownership_released():
            store.close()

    if failure_mode == "exception":
        expected_shutdown_calls = {
            "sync": 1,
            "cancelled_incomplete": 3,
        }.get(release_mode, 2)
        assert shutdown_calls == expected_shutdown_calls
    else:
        assert scheduler_shutdown_calls == 2
    reopened = SQLiteStore(database)
    reopened.close()


def test_async_from_store_yields_caller_loop_during_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation_loop_advanced = threading.Event()
    recovery_loop_advanced = threading.Event()
    observed_allocation_progress: list[bool] = []
    observed_recovery_progress: list[bool] = []

    class BlockingAllocationRuntime(Runtime):
        @classmethod
        def allocate_unassembled(cls) -> BlockingAllocationRuntime:
            observed_allocation_progress.append(
                allocation_loop_advanced.wait(timeout=1.0)
            )
            host = super().allocate_unassembled()
            assert isinstance(host, cls)
            return host

    def blocking_recovery(_host: Runtime) -> None:
        observed_recovery_progress.append(
            recovery_loop_advanced.wait(timeout=1.0)
        )
        raise RuntimeError("injected recovery failure")

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(blocking_recovery),
    )
    store = SQLiteStore(":memory:")

    async def exercise() -> None:
        caller_loop = asyncio.get_running_loop()
        caller_loop.call_soon(allocation_loop_advanced.set)
        caller_loop.call_soon(recovery_loop_advanced.set)
        with pytest.raises(RuntimeError, match="injected recovery failure"):
            await RuntimeBuilder.configured(
                BlockingAllocationRuntime
            ).afrom_store(store)

    try:
        asyncio.run(exercise())
    finally:
        store.close()

    assert observed_allocation_progress == [True]
    assert observed_recovery_progress == [True]


@pytest.mark.parametrize(
    "entrypoint",
    ["aopen", "afrom_store", "aassemble_existing"],
)
def test_async_assembly_reservation_closes_readiness_to_worker_gap(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    store = SQLiteStore(":memory:")
    worker_reached_claim = threading.Event()
    allow_worker_claim = threading.Event()
    original_claim = store.claim_runtime_assembly

    @contextmanager
    def delayed_claim(
        reservation: object,
        *,
        require_unbound_runtime: bool = False,
    ):
        worker_reached_claim.set()
        assert allow_worker_claim.wait(timeout=5)
        with original_claim(
            reservation,  # type: ignore[arg-type]
            require_unbound_runtime=require_unbound_runtime,
        ):
            yield

    monkeypatch.setattr(store, "claim_runtime_assembly", delayed_claim)
    if entrypoint == "aopen":
        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )

    async def assemble() -> Runtime:
        if entrypoint == "aopen":
            return await Runtime.aopen("local")
        builder = RuntimeBuilder.configured(Runtime)
        if entrypoint == "afrom_store":
            return await builder.afrom_store(store)
        host = Runtime.allocate_unassembled()
        await builder.aassemble_existing(
            host,
            store,
            llm_client=None,
            substrate=None,
            config=None,
            startup_module_manifests=None,
            trusted_modules=None,
            trusted_module_sha256=None,
        )
        return host

    async def exercise() -> None:
        opening = asyncio.create_task(assemble())
        assert await asyncio.to_thread(worker_reached_claim.wait, 5)

        scope_errors: list[BaseException | None] = []
        for operation in (
            lambda: store.locked(),
            lambda: store.transaction(),
        ):
            try:
                with operation():
                    pass
            except BaseException as error:
                scope_errors.append(error)
            else:
                scope_errors.append(None)
        try:
            store._query("SELECT 1")
        except BaseException as error:
            scope_errors.append(error)
        else:
            scope_errors.append(None)

        allow_worker_claim.set()
        runtime = await asyncio.wait_for(opening, timeout=10)
        try:
            assert all(isinstance(error, RuntimeError) for error in scope_errors)
            assert all(
                "assembly is reserved" in str(error)
                for error in scope_errors
                if error is not None
            )
        finally:
            await runtime.ashutdown(reason="assembly reservation gap regression")

    try:
        asyncio.run(exercise())
    finally:
        allow_worker_claim.set()
        if not store._runtime_ownership_released():
            store.close()


@pytest.mark.parametrize("entrypoint", ["afrom_store", "aassemble_existing"])
def test_async_assembly_scheduling_failure_releases_reservation(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    store = SQLiteStore(":memory:")

    async def fail_before_worker_start(
        _callback: Callable[[], object],
    ) -> tuple[object | None, BaseException | None, tuple[BaseException, ...]]:
        raise RuntimeError("injected startup scheduling failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder._drain_blocking_startup_call",
        fail_before_worker_start,
    )

    async def exercise() -> None:
        builder = RuntimeBuilder.configured(Runtime)
        with pytest.raises(RuntimeError, match="startup scheduling failure"):
            if entrypoint == "afrom_store":
                await builder.afrom_store(store)
            else:
                host = Runtime.allocate_unassembled()
                await builder.aassemble_existing(
                    host,
                    store,
                    llm_client=None,
                    substrate=None,
                    config=None,
                    startup_module_manifests=None,
                    trusted_modules=None,
                    trusted_module_sha256=None,
                )
        assert store._runtime_assembly_reservation is None
        with store.locked():
            assert store._query("SELECT 1")

    try:
        asyncio.run(exercise())
    finally:
        store.close()
