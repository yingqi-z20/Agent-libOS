from __future__ import annotations

import asyncio
import threading

import pytest

from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.runtime.blocking_work import BlockingWorkSupervisor


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


def test_run_blocking_once_propagates_worker_cancelled_error_without_spinning() -> None:
    def worker() -> None:
        raise asyncio.CancelledError("worker cancelled itself")

    async def exercise() -> None:
        with pytest.raises(asyncio.CancelledError, match="worker cancelled itself"):
            await asyncio.wait_for(run_blocking_once(worker), timeout=1)

    asyncio.run(exercise())


def test_runtime_supervisor_propagates_worker_cancelled_error_without_spinning() -> None:
    async def exercise() -> None:
        supervisor = BlockingWorkSupervisor(max_workers=1, shutdown_timeout_s=1)

        def worker() -> None:
            raise asyncio.CancelledError("supervised worker cancelled itself")

        try:
            with pytest.raises(
                asyncio.CancelledError,
                match="supervised worker cancelled itself",
            ):
                await asyncio.wait_for(supervisor.run(worker), timeout=1)
        finally:
            assert await supervisor.ashutdown()

    asyncio.run(exercise())


def test_run_blocking_once_aggregates_caller_cancellation_and_worker_failure() -> None:
    async def exercise() -> None:
        entered = threading.Event()
        release = threading.Event()

        def worker() -> None:
            entered.set()
            assert release.wait(timeout=1)
            raise RuntimeError("worker failed after cancellation")

        task = asyncio.create_task(run_blocking_once(worker))
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(BaseExceptionGroup) as caught:
            await task
        assert caught.value.subgroup(asyncio.CancelledError) is not None
        assert caught.value.subgroup(RuntimeError) is not None

    asyncio.run(exercise())


def test_supervisor_shutdown_joins_without_holding_future_registry_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = BlockingWorkSupervisor(max_workers=1, shutdown_timeout_s=1)
    worker_entered = threading.Event()
    release_worker = threading.Event()
    callback_entered = threading.Event()
    release_callback = threading.Event()
    shutdown_join_entered = threading.Event()
    callback_acquired_registry: list[bool] = []
    results: list[str] = []
    errors: list[BaseException] = []
    shutdown_results: list[bool] = []

    def worker() -> str:
        worker_entered.set()
        assert release_worker.wait(timeout=2)
        return "done"

    def observed_forget(future: object) -> None:
        callback_entered.set()
        assert release_callback.wait(timeout=2)
        acquired = supervisor._lock.acquire(timeout=0.5)
        callback_acquired_registry.append(acquired)
        if not acquired:
            return
        try:
            supervisor._futures.discard(future)  # type: ignore[arg-type]
        finally:
            supervisor._lock.release()

    original_executor_shutdown = supervisor._executor.shutdown

    def observed_executor_shutdown(*args: object, **kwargs: object) -> None:
        shutdown_join_entered.set()
        original_executor_shutdown(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(supervisor, "_forget", observed_forget)
    monkeypatch.setattr(
        supervisor._executor,
        "shutdown",
        observed_executor_shutdown,
    )

    def run_work() -> None:
        try:
            results.append(asyncio.run(supervisor.run(worker)))
        except BaseException as exc:  # pragma: no cover - assertion diagnostics
            errors.append(exc)

    def shutdown() -> None:
        try:
            shutdown_results.append(supervisor.shutdown())
        except BaseException as exc:  # pragma: no cover - assertion diagnostics
            errors.append(exc)

    worker_thread = threading.Thread(target=run_work, daemon=True)
    worker_thread.start()
    assert worker_entered.wait(timeout=2)

    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
    shutdown_thread.start()
    for _attempt in range(2_000):
        with supervisor._lock:
            if supervisor._closing:
                break
        threading.Event().wait(0.001)
    else:  # pragma: no cover - assertion diagnostics
        raise AssertionError("shutdown did not close work admission")

    release_worker.set()
    assert callback_entered.wait(timeout=2)
    assert shutdown_join_entered.wait(timeout=2)
    release_callback.set()

    worker_thread.join(timeout=2)
    shutdown_thread.join(timeout=2)
    assert not worker_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert errors == []
    assert results == ["done"]
    assert shutdown_results == [True]
    assert callback_acquired_registry == [True]


def test_supervisor_rejects_invalid_worker_count_before_executor_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_executor(
        *args: object,
        **kwargs: object,
    ) -> object:
        executor_calls.append((args, kwargs))
        raise AssertionError("invalid configuration must not create an executor")

    monkeypatch.setattr(
        "agent_libos.runtime.blocking_work.ThreadPoolExecutor",
        unexpected_executor,
    )
    invalid_worker_counts = (
        0,
        -1,
        True,
        False,
        1.0,
        "1",
        None,
        _IntSubclass(1),
    )
    for invalid_worker_count in invalid_worker_counts:
        with pytest.raises(
            ValueError,
            match="max_workers must be a positive integer",
        ):
            BlockingWorkSupervisor(
                max_workers=invalid_worker_count,  # type: ignore[arg-type]
                shutdown_timeout_s=1,
            )
    assert executor_calls == []


def test_supervisor_rejects_invalid_timeout_before_executor_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_executor(
        *args: object,
        **kwargs: object,
    ) -> object:
        executor_calls.append((args, kwargs))
        raise AssertionError("invalid configuration must not create an executor")

    monkeypatch.setattr(
        "agent_libos.runtime.blocking_work.ThreadPoolExecutor",
        unexpected_executor,
    )
    invalid_timeouts = (
        0,
        -1,
        0.0,
        -1.0,
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
        None,
        _IntSubclass(1),
        _FloatSubclass(1.0),
        10**10_000,
    )
    for invalid_timeout in invalid_timeouts:
        with pytest.raises(
            ValueError,
            match="shutdown timeout must be a positive finite number",
        ):
            BlockingWorkSupervisor(
                max_workers=1,
                shutdown_timeout_s=invalid_timeout,  # type: ignore[arg-type]
            )
    assert executor_calls == []
