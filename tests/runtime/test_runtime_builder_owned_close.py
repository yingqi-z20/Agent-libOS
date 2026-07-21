from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Callable

import pytest

from agent_libos.runtime.builder import RuntimeAssemblyCleanupRequired, RuntimeBuilder
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.runtime import Runtime
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import (
    SQLiteStore,
    StoreAssemblyReadiness,
    StoreCloseClaimOutcome,
    StoreCloseOutcome,
)


def _install_failed_open(
    monkeypatch: pytest.MonkeyPatch,
    store: SQLiteStore,
) -> None:
    original_recover = RuntimeBuilder._recover_runtime_state
    attempts = 0

    def fail_once(host: Runtime) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected assembly failure")
        original_recover(host)

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(fail_once),
    )


def _fail_next_close(
    monkeypatch: pytest.MonkeyPatch,
    store: SQLiteStore,
) -> tuple[list[int], Callable[[], None]]:
    original_close = store.close
    close_calls: list[int] = []

    def fail_once_then_close() -> None:
        close_calls.append(len(close_calls) + 1)
        if len(close_calls) == 1:
            raise OSError("injected retained close failure")
        original_close()

    monkeypatch.setattr(store, "close", fail_once_then_close)
    return close_calls, original_close


def _extract_one(error: BaseException) -> RuntimeAssemblyCleanupRequired:
    handles = RuntimeAssemblyCleanupRequired.extract(error)
    assert len(handles) == 1
    return handles[0]


def test_owned_close_failure_blocks_successor_until_exact_sync_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "sync-owned-close.sqlite"
    store = SQLiteStore(database)
    close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    reservation = handle._owned_store_close_reservation

    assert handle.owns_store is True
    assert handle.cleanup_completed is True
    assert handle.released is False
    assert store._admission_commit_guard is reservation
    assert close_calls == [1]
    assert store.list_processes() == []
    with pytest.raises(RuntimeError, match="admission commit guard is already bound"):
        RuntimeBuilder.configured(Runtime).from_store(store)
    assert store._admission_commit_guard is reservation

    handle.release()

    assert handle.released is True
    assert close_calls == [1, 2]
    reopened = SQLiteStore(database)
    reopened.close()


def test_owned_close_failure_blocks_successor_until_exact_async_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-owned-close.sqlite"
    store = SQLiteStore(database)
    close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        reservation = handle._owned_store_close_reservation

        assert handle.owns_store is True
        assert store._admission_commit_guard is reservation
        assert close_calls == [1]
        with pytest.raises(
            RuntimeError,
            match="admission commit guard is already bound",
        ):
            await RuntimeBuilder.configured(Runtime).afrom_store(store)

        await handle.arelease()

        assert handle.released is True
        assert close_calls == [1, 2]
        handle.release()

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


def test_stale_owned_handle_never_closes_successor_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "stale-owned-close.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    reservation = handle._owned_store_close_reservation
    assert store.unbind_admission_commit_guard(reservation) is True
    successor = RuntimeBuilder.configured(Runtime).from_store(store)
    successor_guard = store._admission_commit_guard

    try:
        with pytest.raises(BaseExceptionGroup, match="ownership conflict"):
            handle.release()
        assert handle.released is False
        assert store._admission_commit_guard is successor_guard
        assert store.list_processes() == []
    finally:
        successor.close()


def test_pre_lifecycle_failed_open_reserves_exact_close_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "pre-lifecycle-owned-close.sqlite"
    store = SQLiteStore(database)
    close_calls, _original_close = _fail_next_close(monkeypatch, store)

    def fail_before_lifecycle(_host: Runtime) -> None:
        raise RuntimeError("injected pre-lifecycle failure")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        RuntimeBuilder,
        "_configure_evidence_and_authority",
        staticmethod(fail_before_lifecycle),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)

    assert handle.partial_runtime is not None
    assert not hasattr(handle.partial_runtime, "lifecycle")
    assert store._admission_commit_guard is handle._owned_store_close_reservation
    assert close_calls == [1]
    handle.release()
    assert handle.released is True
    assert close_calls == [1, 2]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows uses the SQLite connection itself as the runtime lease",
)
def test_sync_failed_open_poisoned_store_closes_and_releases_file_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "sync-poisoned-owned-close.sqlite"
    store = SQLiteStore(database)
    store._poison("injected rollback failure")
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    with pytest.raises(ValidationError, match="injected rollback failure") as caught:
        Runtime.open("ignored")

    assert RuntimeAssemblyCleanupRequired.extract(caught.value) == ()
    assert store._runtime_ownership_released() is True
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows uses the SQLite connection itself as the runtime lease",
)
def test_async_failed_open_poisoned_store_closes_and_releases_file_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-poisoned-owned-close.sqlite"
    store = SQLiteStore(database)
    store._poison("injected rollback failure")
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    async def exercise() -> None:
        with pytest.raises(
            ValidationError,
            match="injected rollback failure",
        ) as caught:
            await Runtime.aopen("ignored")

        assert RuntimeAssemblyCleanupRequired.extract(caught.value) == ()
        assert store._runtime_ownership_released() is True

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows releases its connection-backed lease when the store is poisoned",
)
def test_sync_failed_open_poisoned_store_retained_close_publishes_retry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "sync-poisoned-retained-close.sqlite"
    store = SQLiteStore(database)
    store._poison("injected rollback failure")
    original_close = store.close
    close_calls = 0

    def fail_once_then_close() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("injected retained poison close failure")
        original_close()

    monkeypatch.setattr(store, "close", fail_once_then_close)
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)

    assert handle.owns_store is True
    assert store._runtime_ownership_released() is False
    with pytest.raises(ValidationError, match="already open"):
        SQLiteStore(database)

    handle.release()

    assert close_calls == 2
    assert handle.released is True
    assert store._runtime_ownership_released() is True
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows releases its connection-backed lease when the store is poisoned",
)
def test_async_failed_open_poisoned_store_retained_close_publishes_retry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-poisoned-retained-close.sqlite"
    store = SQLiteStore(database)
    store._poison("injected rollback failure")
    original_close = store.close
    close_calls = 0

    def fail_once_then_close() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("injected retained poison close failure")
        original_close()

    monkeypatch.setattr(store, "close", fail_once_then_close)
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)

        assert handle.owns_store is True
        assert store._runtime_ownership_released() is False
        with pytest.raises(ValidationError, match="already open"):
            SQLiteStore(database)

        await handle.arelease()

        assert close_calls == 2
        assert handle.released is True
        assert store._runtime_ownership_released() is True

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows connection-backed SQLite lease regression",
)
def test_windows_poisoned_file_store_terminalizes_released_connection_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "windows-poisoned-connection-lease.sqlite"
    store = SQLiteStore(database)
    store._poison("injected Windows connection lease release")
    assert store._runtime_ownership_released() is True
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)

    assert handle.owns_store is False
    assert handle.released is True
    reopened = SQLiteStore(database)
    reopened.close()


def test_sync_failed_open_already_released_memory_store_terminalizes_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    store._poison("injected released in-memory ownership")
    assert store._runtime_ownership_released() is True
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)

    assert handle.owns_store is False
    assert handle.released is True
    handle.release()


def test_async_failed_open_already_released_memory_store_terminalizes_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    store._poison("injected released in-memory ownership")
    assert store._runtime_ownership_released() is True
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)

        assert handle.owns_store is False
        assert handle.released is True
        await handle.arelease()

    asyncio.run(exercise())


def test_sync_failed_open_already_released_session_does_not_publish_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReleasedSessionStore:
        def probe_runtime_assembly_readiness(self):
            return StoreAssemblyReadiness.READY

        def probe_admission_guard_close(self, _expected: object):
            return StoreCloseClaimOutcome.OWNERSHIP_RELEASED

    store = ReleasedSessionStore()
    failure = RuntimeError("injected allocation failure after session release")

    def fail_allocation(_builder: RuntimeBuilder[Runtime]) -> Runtime:
        raise failure

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(RuntimeBuilder, "_allocate_host", fail_allocation)

    with pytest.raises(RuntimeError) as caught:
        Runtime.open("ignored")

    assert caught.value is failure
    assert RuntimeAssemblyCleanupRequired.extract(caught.value) == ()


def test_async_failed_open_already_released_session_does_not_publish_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReleasedSessionStore:
        reservation: object | None = None

        def probe_runtime_assembly_readiness(self):
            return StoreAssemblyReadiness.READY

        def reserve_runtime_assembly(self, reservation: object):
            self.reservation = reservation
            return StoreAssemblyReadiness.READY

        def release_runtime_assembly_reservation(self, reservation: object):
            if self.reservation is not reservation:
                return False
            self.reservation = None
            return True

        def probe_admission_guard_close(self, _expected: object):
            return StoreCloseClaimOutcome.OWNERSHIP_RELEASED

    store = ReleasedSessionStore()
    failure = RuntimeError("injected allocation failure after session release")

    def fail_allocation(_builder: RuntimeBuilder[Runtime]) -> Runtime:
        raise failure

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(RuntimeBuilder, "_allocate_host", fail_allocation)

    async def exercise() -> None:
        with pytest.raises(RuntimeError) as caught:
            await Runtime.aopen("ignored")

        assert caught.value is failure
        assert RuntimeAssemblyCleanupRequired.extract(caught.value) == ()

    asyncio.run(exercise())


def test_active_transaction_reservation_failure_is_retryable_after_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "transaction-owned-close.sqlite"
    store = SQLiteStore(database)
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    with store.transaction():
        with pytest.raises(BaseExceptionGroup) as caught:
            Runtime.open("ignored")
        handle = _extract_one(caught.value)
        assert handle.owns_store is True
        assert handle.released is False
        assert any(
            item["component"] == "store_close_reservation"
            for item in handle.cleanup_errors
        )

    handle.release()

    assert handle.released is True
    reopened = SQLiteStore(database)
    reopened.close()


@pytest.mark.parametrize(
    ("scope_name", "readiness"),
    [
        ("locked", "current_thread_locked"),
        ("transaction", "active_transaction"),
    ],
)
def test_sync_owned_release_rejects_current_thread_store_scope_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_name: str,
    readiness: str,
) -> None:
    database = tmp_path / f"sync-current-thread-{scope_name}.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    with handle._lock:
        handle._cleanup_completed = False
    cleanup_calls: list[Runtime] = []

    def track_cleanup(host: Runtime) -> list[dict[str, str]]:
        cleanup_calls.append(host)
        return []

    monkeypatch.setattr(
        RuntimeBuilder,
        "_cleanup_failed_assembly",
        staticmethod(track_cleanup),
    )

    scope = getattr(store, scope_name)
    with scope():
        with pytest.raises(RuntimeError, match=readiness):
            handle.release()
        assert cleanup_calls == []
        assert handle.cleanup_completed is False
        assert handle.released is False

    handle.release()

    assert cleanup_calls == [handle.partial_runtime]
    assert handle.cleanup_completed is True
    assert handle.released is True


@pytest.mark.parametrize(
    ("scope_name", "readiness"),
    [
        ("locked", "current_thread_locked"),
        ("transaction", "active_transaction"),
    ],
)
def test_async_owned_release_rejects_current_thread_store_scope_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_name: str,
    readiness: str,
) -> None:
    database = tmp_path / f"async-current-thread-{scope_name}.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        with handle._lock:
            handle._cleanup_completed = False
        cleanup_calls: list[Runtime] = []

        async def track_cleanup(host: Runtime) -> list[dict[str, str]]:
            cleanup_calls.append(host)
            return []

        monkeypatch.setattr(
            RuntimeBuilder,
            "_drain_async_failed_assembly",
            staticmethod(track_cleanup),
        )

        scope = getattr(store, scope_name)
        with scope():
            with pytest.raises(RuntimeError, match=readiness):
                await handle.arelease()
            assert cleanup_calls == []
            assert handle.cleanup_completed is False
            assert handle.released is False

        await handle.arelease()

        assert cleanup_calls == [handle.partial_runtime]
        assert handle.cleanup_completed is True
        assert handle.released is True

    asyncio.run(exercise())


def test_async_owned_release_repairs_unbound_reservation_after_transaction(
    tmp_path: Path,
) -> None:
    database = tmp_path / "async-repair-unbound-reservation.sqlite"
    store = SQLiteStore(database)
    reservation = RuntimeBuilder._new_owned_store_close_reservation()
    handle = RuntimeAssemblyCleanupRequired(
        partial_runtime=None,
        store=store,
        cleanup_errors=[],
        cleanup_completed=True,
    )
    handle._claim_owned_store(store, reservation)

    async def exercise() -> None:
        with store.transaction():
            with pytest.raises(RuntimeError, match="active_transaction"):
                await handle.arelease()
        assert store._admission_commit_guard is None
        assert handle.released is False

        await handle.arelease()

        assert store._admission_commit_guard is None
        assert handle.released is True

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


def test_async_stale_owned_handle_never_closes_successor_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-stale-owned-close.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        reservation = handle._owned_store_close_reservation
        assert store.unbind_admission_commit_guard(reservation) is True
        successor = await RuntimeBuilder.configured(Runtime).afrom_store(store)
        successor_guard = store._admission_commit_guard

        try:
            with pytest.raises(BaseExceptionGroup, match="ownership conflict"):
                await handle.arelease()
            assert handle.released is False
            assert store._admission_commit_guard is successor_guard
            assert store.list_processes() == []
        finally:
            await successor.ashutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("scope_name", "readiness"),
    [
        ("locked", "current_thread_locked"),
        ("transaction", "active_transaction"),
    ],
)
def test_async_failed_open_publishes_handle_before_current_thread_close_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope_name: str,
    readiness: str,
) -> None:
    database = tmp_path / f"async-open-current-thread-{scope_name}.sqlite"
    store = SQLiteStore(database)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        scope = getattr(store, scope_name)
        with scope():
            with pytest.raises(BaseExceptionGroup) as caught:
                await Runtime.aopen("ignored")
            handle = _extract_one(caught.value)
            assert store._runtime_assembly_reservation is None
            assert (
                caught.value.subgroup(
                    lambda item: isinstance(item, RuntimeError)
                    and readiness in str(item)
                )
                is not None
            )
            assert handle.owns_store is True
            assert handle.released is False

        await handle.arelease()
        assert handle.released is True

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


def test_async_failed_open_publishes_handle_when_assembly_store_lock_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-open-lock-busy.sqlite"
    store = SQLiteStore(database)
    lock_entered = threading.Event()
    release_lock = threading.Event()

    def hold_store_lock() -> None:
        with store.locked():
            lock_entered.set()
            assert release_lock.wait(timeout=3)

    holder = threading.Thread(target=hold_store_lock)
    holder.start()
    assert lock_entered.wait(timeout=3)
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )

    async def exercise() -> RuntimeAssemblyCleanupRequired:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        assert caught.value.subgroup(
            lambda item: isinstance(item, RuntimeError)
            and "lock_busy" in str(item)
        ) is not None
        assert handle.cleanup_completed is True
        assert handle.owns_store is True
        assert handle.released is False
        return handle

    try:
        handle = asyncio.run(exercise())
    finally:
        release_lock.set()
        holder.join(timeout=3)
    assert not holder.is_alive()

    asyncio.run(handle.arelease())
    assert handle.released is True
    reopened = SQLiteStore(database)
    reopened.close()


def test_concurrent_sync_owned_release_has_one_preflight_and_close_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "concurrent-sync-release.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    original_probe = store.probe_admission_guard_close
    probe_entered = threading.Event()
    allow_probe = threading.Event()
    probe_calls: list[int] = []

    def blocking_probe(expected: object):
        probe_calls.append(threading.get_ident())
        probe_entered.set()
        assert allow_probe.wait(timeout=3)
        return original_probe(expected)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "probe_admission_guard_close", blocking_probe)
    leader_errors: list[BaseException] = []
    follower_errors: list[BaseException] = []

    def release_into(errors: list[BaseException]) -> None:
        try:
            handle.release()
        except BaseException as exc:
            errors.append(exc)

    leader = threading.Thread(target=release_into, args=(leader_errors,))
    follower = threading.Thread(target=release_into, args=(follower_errors,))
    leader.start()
    assert probe_entered.wait(timeout=3)
    follower.start()
    follower.join(timeout=3)
    allow_probe.set()
    leader.join(timeout=3)

    assert not leader.is_alive()
    assert not follower.is_alive()
    assert leader_errors == []
    assert len(follower_errors) == 1
    assert "already in progress" in str(follower_errors[0])
    assert len(probe_calls) == 1
    assert handle.released is True


def test_concurrent_async_owned_release_has_one_preflight_claim_and_close_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "concurrent-async-release.sqlite"
    store = SQLiteStore(database)
    _close_calls, original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        original_probe = store.probe_admission_guard_close
        original_claim = store.claim_admission_guard_close
        probe_calls: list[int] = []
        claim_calls: list[int] = []

        def track_probe(expected: object):
            probe_calls.append(threading.get_ident())
            return original_probe(expected)  # type: ignore[arg-type]

        def track_claim(expected: object):
            claim_calls.append(threading.get_ident())
            return original_claim(expected)  # type: ignore[arg-type]

        monkeypatch.setattr(store, "probe_admission_guard_close", track_probe)
        monkeypatch.setattr(store, "claim_admission_guard_close", track_claim)
        close_entered = threading.Event()
        allow_close = threading.Event()

        def blocking_close() -> None:
            close_entered.set()
            assert allow_close.wait(timeout=3)
            original_close()

        monkeypatch.setattr(store, "close", blocking_close)
        leader = asyncio.create_task(handle.arelease())
        while not close_entered.is_set():
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="already in progress"):
            await handle.arelease()
        assert len(probe_calls) == 1
        assert len(claim_calls) == 1

        allow_close.set()
        await leader
        assert handle.released is True

    asyncio.run(exercise())


@pytest.mark.parametrize("failure_mode", ["raise", "false"])
def test_reservation_failure_still_drains_started_object_task_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    database = tmp_path / f"worker-reservation-{failure_mode}.sqlite"
    store = SQLiteStore(database)
    original_replace = store.replace_admission_commit_guard
    replace_calls = 0

    def fail_initial_reservation(expected: object, replacement: object) -> bool:
        nonlocal replace_calls
        replace_calls += 1
        if failure_mode == "raise" and replace_calls == 1:
            raise RuntimeError("injected reservation failure")
        if failure_mode == "false" and replace_calls <= 3:
            return False
        return original_replace(expected, replacement)  # type: ignore[arg-type]

    def fail_after_worker_start(_self: RuntimeLifecycle) -> None:
        raise RuntimeError("injected failure after worker start")

    monkeypatch.setattr(store, "replace_admission_commit_guard", fail_initial_reservation)
    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(RuntimeLifecycle, "mark_open", fail_after_worker_start)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    assert handle.partial_runtime is not None
    assert handle.partial_runtime.object_tasks._closed is True
    assert handle.partial_runtime.object_tasks._thread.is_alive() is False
    assert store._admission_commit_guard is handle._owned_store_close_reservation

    handle.release()
    assert handle.released is True


def test_concurrent_successor_cannot_cross_owned_close_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "concurrent-owned-close.sqlite"
    store = SQLiteStore(database)
    _close_calls, original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    close_entered = threading.Event()
    allow_close = threading.Event()

    def blocking_close() -> None:
        close_entered.set()
        assert allow_close.wait(timeout=3)
        original_close()

    monkeypatch.setattr(store, "close", blocking_close)
    release_errors: list[BaseException] = []
    successor_errors: list[BaseException] = []

    def release_handle() -> None:
        try:
            handle.release()
        except BaseException as exc:
            release_errors.append(exc)

    def assemble_successor() -> None:
        try:
            RuntimeBuilder.configured(Runtime).from_store(store)
        except BaseException as exc:
            successor_errors.append(exc)

    release_thread = threading.Thread(target=release_handle)
    successor_thread = threading.Thread(target=assemble_successor)
    release_thread.start()
    assert close_entered.wait(timeout=3)
    successor_thread.start()
    assert successor_thread.is_alive()
    allow_close.set()
    release_thread.join(timeout=3)
    successor_thread.join(timeout=3)

    assert not release_thread.is_alive()
    assert not successor_thread.is_alive()
    assert release_errors == []
    assert successor_errors
    assert handle.released is True


def test_async_owned_close_is_offloaded_drained_and_cancellation_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-cancel-owned-close.sqlite"
    store = SQLiteStore(database)
    _close_calls, original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    async def exercise() -> None:
        with pytest.raises(BaseExceptionGroup) as caught:
            await Runtime.aopen("ignored")
        handle = _extract_one(caught.value)
        close_entered = threading.Event()
        allow_close = threading.Event()
        heartbeats = 0

        def blocking_close() -> None:
            close_entered.set()
            assert allow_close.wait(timeout=3)
            original_close()

        monkeypatch.setattr(store, "close", blocking_close)

        async def heartbeat() -> None:
            nonlocal heartbeats
            while not close_entered.is_set():
                heartbeats += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        release_task = asyncio.create_task(handle.arelease())
        while not close_entered.is_set():
            await asyncio.sleep(0)
        await heartbeat_task
        release_task.cancel()
        allow_close.set()

        with pytest.raises(BaseExceptionGroup) as released:
            await release_task
        assert released.value.subgroup(asyncio.CancelledError) is not None
        assert heartbeats > 0
        assert handle.released is True

    asyncio.run(exercise())


def test_async_failed_open_close_is_offloaded_drained_and_cancellation_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "async-open-cancel-owned-close.sqlite"
    store = SQLiteStore(database)
    _install_failed_open(monkeypatch, store)
    original_close = store.close
    close_entered = threading.Event()
    allow_close = threading.Event()

    def blocking_close() -> None:
        close_entered.set()
        assert allow_close.wait(timeout=3)
        original_close()

    monkeypatch.setattr(store, "close", blocking_close)

    async def exercise() -> None:
        heartbeats = 0

        async def heartbeat() -> None:
            nonlocal heartbeats
            while not close_entered.is_set():
                heartbeats += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        open_task = asyncio.create_task(Runtime.aopen("ignored"))
        while not close_entered.is_set():
            await asyncio.sleep(0)
        await heartbeat_task
        open_task.cancel()
        allow_close.set()

        with pytest.raises(BaseExceptionGroup) as cancelled:
            await open_task
        assert cancelled.value.subgroup(asyncio.CancelledError) is not None
        assert cancelled.value.subgroup(RuntimeError) is not None
        assert heartbeats > 0

    asyncio.run(exercise())
    reopened = SQLiteStore(database)
    reopened.close()


def test_released_store_warnings_terminalize_handle_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "warning-owned-close.sqlite"
    store = SQLiteStore(database)
    _close_calls, _original_close = _fail_next_close(monkeypatch, store)
    _install_failed_open(monkeypatch, store)

    with pytest.raises(BaseExceptionGroup) as caught:
        Runtime.open("ignored")
    handle = _extract_one(caught.value)
    original_release = store.release_admission_guard_and_close
    warning = OSError("released with diagnostic")

    def release_with_warning(expected: object) -> StoreCloseOutcome:
        outcome = original_release(expected)  # type: ignore[arg-type]
        assert outcome.ownership_released is True
        return StoreCloseOutcome(
            guard_matched=True,
            ownership_released=True,
            warnings=(warning,),
        )

    monkeypatch.setattr(
        store,
        "release_admission_guard_and_close",
        release_with_warning,
    )

    with pytest.raises(BaseExceptionGroup) as warned:
        handle.release()

    assert warned.value.subgroup(OSError) is not None
    assert handle.released is True
    handle.release()
