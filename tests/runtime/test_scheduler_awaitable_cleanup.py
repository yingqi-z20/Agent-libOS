from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SchedulerDefaults
from agent_libos.models import ProcessStatus


def _cleanup_config(*, timeout_s: float = 0.03) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        scheduler=SchedulerDefaults(
            max_workers=1,
            poll_interval_s=0.001,
            drain_window_s=0.01,
            shutdown_join_timeout_s=timeout_s,
        )
    )


def _run_in_thread(operation: Callable[[], Any]) -> tuple[
    threading.Thread,
    threading.Event,
    list[Any],
    list[BaseException],
]:
    finished = threading.Event()
    results: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(operation())
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, finished, results, errors


def test_awaitable_cleanup_recancels_task_that_swallows_one_cancellation() -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="cooperative repeated cancellation cleanup")
    background_started = threading.Event()
    release_background = threading.Event()
    cancellations: list[int] = []
    try:
        async def background() -> None:
            background_started.set()
            while not release_background.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    cancellations.append(len(cancellations) + 1)
                    if len(cancellations) == 1:
                        continue
                    raise

        async def quantum(selected_pid: str) -> dict[str, str]:
            asyncio.create_task(background())
            while not background_started.is_set():
                await asyncio.sleep(0)
            return {"pid": selected_pid}

        thread, finished, results, errors = _run_in_thread(
            lambda: runtime.scheduler.run_pid_once(pid, quantum)
        )
        assert background_started.wait(timeout=1)
        finished_before_release = finished.wait(timeout=0.3)
        release_background.set()
        thread.join(timeout=2)

        assert finished_before_release
        assert not thread.is_alive()
        assert errors == []
        assert results == [{"pid": pid}]
        assert len(cancellations) >= 2
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
    finally:
        release_background.set()
        runtime.close()


def test_awaitable_cleanup_fails_quantum_for_cancellation_resistant_task(
    caplog: Any,
) -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="noncooperative background cleanup")
    background_started = threading.Event()
    release_background = threading.Event()
    cancellation_count = 0
    caplog.set_level(logging.ERROR, logger="asyncio")
    try:
        async def background() -> None:
            nonlocal cancellation_count
            background_started.set()
            while not release_background.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    cancellation_count += 1

        async def quantum(selected_pid: str) -> dict[str, str]:
            asyncio.create_task(background())
            while not background_started.is_set():
                await asyncio.sleep(0)
            return {"pid": selected_pid}

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            thread, finished, results, errors = _run_in_thread(
                lambda: runtime.scheduler.run_pid_until_idle(
                    pid,
                    quantum,
                    max_quanta=1,
                )
            )
            assert background_started.wait(timeout=1)
            finished_before_release = finished.wait(timeout=0.3)
            release_background.set()
            thread.join(timeout=2)
            gc.collect()

        assert finished_before_release
        assert not thread.is_alive()
        assert errors == []
        assert len(results) == 1
        assert len(results[0]) == 1
        failure = results[0][0]
        assert failure["ok"] is False
        assert failure["code"] == "internal_error"
        assert failure["error_type"] == "RuntimeError"
        assert cancellation_count >= 2
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.FAILED
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
        assert not [
            warning
            for warning in caught_warnings
            if "never awaited" in str(warning.message)
        ]
        assert not [
            record
            for record in caplog.records
            if "Task was destroyed" in record.getMessage()
        ]
    finally:
        release_background.set()
        runtime.close()


def test_cancelled_main_awaitable_cannot_hide_resistant_background_cleanup() -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="cancelled main with noncooperative background")
    main_started = threading.Event()
    background_started = threading.Event()
    release_background = threading.Event()
    try:
        async def background() -> None:
            background_started.set()
            while not release_background.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue

        async def quantum(_selected_pid: str) -> None:
            main_started.set()
            try:
                await asyncio.sleep(3600)
            finally:
                asyncio.create_task(background())
                await asyncio.sleep(0)

        thread, finished, _results, errors = _run_in_thread(
            lambda: runtime.scheduler.run_pid_once(pid, quantum)
        )
        assert main_started.wait(timeout=1)
        runtime.scheduler._cancel_awaitable(pid)
        assert background_started.wait(timeout=1)
        finished_before_release = finished.wait(timeout=0.3)
        release_background.set()
        thread.join(timeout=2)

        assert finished_before_release
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "awaitable cleanup did not complete" in str(errors[0])
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
    finally:
        release_background.set()
        runtime.close()


def test_awaitable_cleanup_bounds_resistant_async_generator_shutdown(
    caplog: Any,
) -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="noncooperative async generator cleanup")
    generator_started = threading.Event()
    release_generator = threading.Event()
    caplog.set_level(logging.ERROR, logger="asyncio")
    try:
        async def resistant_generator() -> Any:
            generator_started.set()
            try:
                yield "started"
            finally:
                while not release_generator.is_set():
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        continue

        async def quantum(selected_pid: str) -> dict[str, str]:
            generator = resistant_generator()
            assert await anext(generator) == "started"
            return {"pid": selected_pid}

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            thread, finished, results, errors = _run_in_thread(
                lambda: runtime.scheduler.run_pid_until_idle(
                    pid,
                    quantum,
                    max_quanta=1,
                )
            )
            assert generator_started.wait(timeout=1)
            finished_before_release = finished.wait(timeout=0.3)
            release_generator.set()
            thread.join(timeout=2)
            gc.collect()

        assert finished_before_release
        assert not thread.is_alive()
        assert errors == []
        assert results[0][0]["ok"] is False
        assert results[0][0]["error_type"] == "RuntimeError"
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.FAILED
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
        assert not [
            warning
            for warning in caught_warnings
            if "never awaited" in str(warning.message)
        ]
        assert not [
            record
            for record in caplog.records
            if "Task was destroyed" in record.getMessage()
        ]
    finally:
        release_generator.set()
        runtime.close()


def test_blocked_default_executor_fails_quantum_and_defers_runtime_shutdown() -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="blocked default executor cleanup")
    executor_started = threading.Event()
    executor_finished = threading.Event()
    release_executor = threading.Event()
    first_shutdown: dict[str, Any] | None = None
    second_shutdown: dict[str, Any] | None = None
    process_after_quantum: Any | None = None
    try:
        def blocking_work() -> None:
            executor_started.set()
            release_executor.wait(timeout=5)
            executor_finished.set()

        async def quantum(selected_pid: str) -> dict[str, str]:
            asyncio.get_running_loop().run_in_executor(None, blocking_work)
            return {"pid": selected_pid}

        thread, finished, results, errors = _run_in_thread(
            lambda: runtime.scheduler.run_pid_until_idle(
                pid,
                quantum,
                max_quanta=1,
            )
        )
        assert executor_started.wait(timeout=1)
        finished_before_release = finished.wait(timeout=0.3)
        if finished_before_release:
            process_after_quantum = runtime.process.get(pid)
            first_shutdown = runtime.shutdown(
                actor="test",
                reason="default executor still running",
            )
        release_executor.set()
        assert executor_finished.wait(timeout=2)
        thread.join(timeout=2)

        if first_shutdown is not None:
            deadline = time.monotonic() + 2
            while not runtime.scheduler.shutdown() and time.monotonic() < deadline:
                time.sleep(0.01)
            second_shutdown = runtime.shutdown(
                actor="test",
                reason="default executor drained",
            )

        assert finished_before_release
        assert not thread.is_alive()
        assert errors == []
        assert len(results) == 1
        assert results[0][0]["ok"] is False
        assert results[0][0]["error_type"] == "RuntimeError"
        assert process_after_quantum is not None
        assert process_after_quantum.status == ProcessStatus.FAILED
        assert process_after_quantum.execution_owner_id is None
        assert process_after_quantum.execution_lease_id is None
        assert first_shutdown is not None
        assert first_shutdown["ok"] is False
        assert first_shutdown["scheduler_stopped"] is False
        assert second_shutdown is not None
        assert second_shutdown["ok"] is True
    finally:
        release_executor.set()
        runtime.close()


def test_default_executor_shutdown_call_cannot_block_quantum_caller() -> None:
    runtime = Runtime.open("local", config=_cleanup_config())
    pid = runtime.process.spawn(goal="bounded default executor shutdown call")
    shutdown_started = threading.Event()
    release_shutdown = threading.Event()
    first_shutdown: dict[str, Any] | None = None
    second_shutdown: dict[str, Any] | None = None
    process_after_quantum: Any | None = None

    class BlockingShutdownExecutor(ThreadPoolExecutor):
        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_started.set()
            release_shutdown.wait(timeout=5)
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    try:
        async def quantum(selected_pid: str) -> dict[str, str]:
            asyncio.get_running_loop().set_default_executor(
                BlockingShutdownExecutor(max_workers=1)
            )
            return {"pid": selected_pid}

        thread, finished, results, errors = _run_in_thread(
            lambda: runtime.scheduler.run_pid_until_idle(
                pid,
                quantum,
                max_quanta=1,
            )
        )
        assert shutdown_started.wait(timeout=1)
        finished_before_release = finished.wait(timeout=0.3)
        if finished_before_release:
            process_after_quantum = runtime.process.get(pid)
            first_shutdown = runtime.shutdown(
                actor="test",
                reason="default executor shutdown call still blocked",
            )
        release_shutdown.set()
        thread.join(timeout=2)

        if first_shutdown is not None:
            deadline = time.monotonic() + 2
            while not runtime.scheduler.shutdown() and time.monotonic() < deadline:
                time.sleep(0.01)
            second_shutdown = runtime.shutdown(
                actor="test",
                reason="default executor shutdown call completed",
            )

        assert finished_before_release
        assert not thread.is_alive()
        assert errors == []
        assert results[0][0]["ok"] is False
        assert results[0][0]["error_type"] == "RuntimeError"
        assert process_after_quantum is not None
        assert process_after_quantum.status == ProcessStatus.FAILED
        assert process_after_quantum.execution_owner_id is None
        assert process_after_quantum.execution_lease_id is None
        assert first_shutdown is not None
        assert first_shutdown["ok"] is False
        assert first_shutdown["scheduler_stopped"] is False
        assert second_shutdown is not None
        assert second_shutdown["ok"] is True
    finally:
        release_shutdown.set()
        runtime.close()
