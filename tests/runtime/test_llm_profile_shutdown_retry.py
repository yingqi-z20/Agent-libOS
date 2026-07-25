from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.runtime.builder import RuntimeAssemblyCleanupRequired, RuntimeBuilder
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import SQLiteStore


class _TrackingStore(SQLiteStore):
    def __init__(self) -> None:
        self.close_calls = 0
        super().__init__(":memory:")

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class _ScriptedSyncClient:
    def __init__(self, *outcomes: BaseException | None) -> None:
        self._outcomes = list(outcomes)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is not None:
            raise outcome


class _ScriptedAsyncClient:
    def __init__(self, *outcomes: BaseException | None) -> None:
        self._outcomes = list(outcomes)
        self.close_calls = 0

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.close_calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is not None:
            raise outcome


def _cache_production_client(
    registry: LLMProfileRegistry,
    profile_id: str,
    client: Any,
) -> None:
    registry._clients[profile_id] = client
    registry._client_cache_sha256[profile_id] = f"cache:{profile_id}"


def test_sync_profile_shutdown_removes_only_successful_unique_clients() -> None:
    registry = LLMProfileRegistry(object())  # type: ignore[arg-type]
    shared_success = _ScriptedSyncClient()
    transient_failure = _ScriptedSyncClient(OSError("retry close"), None)
    interrupted = _ScriptedSyncClient(KeyboardInterrupt("retry interrupt"), None)
    _cache_production_client(registry, "default", shared_success)
    registry._test_clients["fast"] = shared_success
    registry._test_clients["slow"] = transient_failure
    registry._test_clients["interrupt"] = interrupted

    with pytest.raises(BaseExceptionGroup) as caught:
        registry.shutdown()

    assert caught.value.subgroup(KeyboardInterrupt) is not None
    assert caught.value.subgroup(OSError) is not None
    assert shared_success.close_calls == 1
    assert transient_failure.close_calls == 1
    assert interrupted.close_calls == 1
    assert registry._clients == {}
    assert registry._client_cache_sha256 == {}
    assert registry._test_clients == {
        "slow": transient_failure,
        "interrupt": interrupted,
    }

    asyncio.run(registry.ashutdown())

    assert shared_success.close_calls == 1
    assert transient_failure.close_calls == 2
    assert interrupted.close_calls == 2
    assert registry._test_clients == {}


def test_async_profile_shutdown_preserves_interrupts_and_sync_fallbacks() -> None:
    async def exercise() -> None:
        registry = LLMProfileRegistry(object())  # type: ignore[arg-type]
        shared_success = _ScriptedAsyncClient()
        transient_failure = _ScriptedAsyncClient(OSError("retry async close"), None)
        interrupted = _ScriptedAsyncClient(
            asyncio.CancelledError("retry cancellation"),
            None,
        )
        sync_fallback = _ScriptedSyncClient()
        _cache_production_client(registry, "default", shared_success)
        registry._test_clients["fast"] = shared_success
        registry._test_clients["slow"] = transient_failure
        registry._test_clients["interrupt"] = interrupted
        registry._test_clients["sync"] = sync_fallback

        with pytest.raises(BaseExceptionGroup) as caught:
            await registry.ashutdown()

        assert caught.value.subgroup(asyncio.CancelledError) is not None
        assert caught.value.subgroup(OSError) is not None
        assert shared_success.close_calls == 1
        assert transient_failure.close_calls == 1
        assert interrupted.close_calls == 1
        assert sync_fallback.close_calls == 1
        assert registry._clients == {}
        assert registry._client_cache_sha256 == {}
        assert registry._test_clients == {
            "slow": transient_failure,
            "interrupt": interrupted,
        }

        registry.shutdown()

        assert shared_success.close_calls == 1
        assert transient_failure.close_calls == 2
        assert interrupted.close_calls == 2
        assert sync_fallback.close_calls == 1
        assert registry._test_clients == {}

    asyncio.run(exercise())


def test_runtime_shutdown_retries_only_the_failed_llm_client() -> None:
    store = _TrackingStore()
    runtime = RuntimeBuilder.configured(Runtime).from_store(store)
    failed = _ScriptedSyncClient(OSError("retry runtime close"), None)
    runtime.llms.set_test_client("default", failed)

    first = runtime.shutdown(actor="test", reason="transient LLM close")

    assert first["ok"] is False
    assert first["llms_stopped"] is False
    assert failed.close_calls == 1
    assert store.close_calls == 0
    assert store.list_processes() == []

    second = runtime.shutdown(actor="test", reason="retry LLM close")

    assert second["ok"] is True
    assert failed.close_calls == 2
    assert store.close_calls == 1


def test_sync_failed_assembly_cleanup_retries_llm_before_owned_store_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _TrackingStore()
    client = _ScriptedSyncClient(OSError("retry failed assembly close"), None)

    def fail_after_client_registration(host: Runtime) -> None:
        host.llms.set_test_client("default", client)
        raise RuntimeError("injected assembly failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(fail_after_client_registration),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
    assert len(handles) == 1
    handle = handles[0]
    assert handle.owns_store is True
    assert handle.released is False
    assert client.close_calls == 1
    assert store.close_calls == 0
    assert handle.partial_runtime is not None
    assert handle.partial_runtime.llms._test_clients["default"] is client

    handle.release()

    assert handle.cleanup_completed is True
    assert handle.released is True
    assert client.close_calls == 2
    assert handle.partial_runtime.llms._test_clients == {}
    assert store.close_calls == 1


def test_async_failed_assembly_cleanup_retries_llm_before_owned_store_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _TrackingStore()
    client = _ScriptedAsyncClient(
        asyncio.CancelledError("retry interrupted assembly close"),
        None,
    )

    def fail_after_client_registration(host: Runtime) -> None:
        host.llms.set_test_client("default", client)
        raise RuntimeError("injected async assembly failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(fail_after_client_registration),
    )

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        assert caught.value.subgroup(asyncio.CancelledError) is not None
        handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
        assert len(handles) == 1
        handle = handles[0]
        assert handle.owns_store is True
        assert handle.released is False
        assert client.close_calls == 1
        assert store.close_calls == 0
        assert handle.partial_runtime is not None
        assert handle.partial_runtime.llms._test_clients["default"] is client

        await handle.arelease()

        assert handle.cleanup_completed is True
        assert handle.released is True
        assert client.close_calls == 2
        assert handle.partial_runtime.llms._test_clients == {}
        assert store.close_calls == 1

    asyncio.run(exercise())


def test_async_profile_shutdown_offloads_sync_close_timer_barrier() -> None:
    entered = threading.Event()
    release = threading.Event()
    close_threads: list[int] = []

    class BlockingSyncClient:
        def close(self) -> None:
            close_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(timeout=1)

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        registry = LLMProfileRegistry(object())  # type: ignore[arg-type]
        registry._test_clients["sync"] = BlockingSyncClient()
        caller_thread = threading.get_ident()
        loop.call_later(0.01, release.set)
        await registry.ashutdown()
        assert entered.is_set()
        assert close_threads and close_threads[0] != caller_thread
        assert registry._test_clients == {}

    asyncio.run(exercise())


def test_async_failed_assembly_retains_sync_client_that_raises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _TrackingStore()
    client = _ScriptedSyncClient(
        asyncio.CancelledError("retry sync worker cancellation"),
        None,
    )

    def fail_after_client_registration(host: Runtime) -> None:
        host.llms.set_test_client("default", client)
        raise RuntimeError("injected async assembly failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(fail_after_client_registration),
    )

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        assert caught.value.subgroup(asyncio.CancelledError) is not None
        handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
        assert len(handles) == 1
        handle = handles[0]
        assert not handle.released
        assert client.close_calls == 1
        assert handle.partial_runtime is not None
        assert handle.partial_runtime.llms._test_clients["default"] is client

        await handle.arelease()

        assert handle.released
        assert client.close_calls == 2
        assert handle.partial_runtime.llms._test_clients == {}
        assert store.close_calls == 1

    asyncio.run(exercise())
