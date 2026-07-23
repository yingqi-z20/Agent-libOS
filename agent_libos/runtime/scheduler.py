from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
import time
from contextlib import contextmanager
from collections.abc import Awaitable, Callable
from concurrent.futures import FIRST_COMPLETED, CancelledError as FutureCancelledError, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    FailedProcessOutcome,
    ProcessExecutionToken,
    ProcessStatus,
    ResourceUsage,
)
from agent_libos.models.exceptions import ResourceLimitExceeded, ValidationError
from agent_libos.process_execution import bind_process_execution
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.storage import ProcessRepository


Quantum = Callable[[str], Any | Awaitable[Any]]
_SCHEDULER_DEFAULTS = DEFAULT_CONFIG.scheduler
_ACTIVE_QUANTUM: contextvars.ContextVar[tuple[int, str] | None] = contextvars.ContextVar(
    "agent_libos_active_scheduler_quantum",
    default=None,
)


class AsyncProcessScheduler:
    """Thread-backed scheduler for AgentProcess quanta.

    Async public methods are kept for host compatibility, but process quanta are
    executed by worker threads so one blocked process does not monopolize the
    runtime scheduler.
    """

    TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
    WAITING_STATUSES = {ProcessStatus.WAITING_EVENT, ProcessStatus.WAITING_TOOL, ProcessStatus.WAITING_HUMAN}

    def __init__(
        self,
        store: ProcessRepository,
        audit: AuditManager,
        poll_interval_s: float = _SCHEDULER_DEFAULTS.poll_interval_s,
        max_workers: int = _SCHEDULER_DEFAULTS.max_workers,
        drain_window_s: float = _SCHEDULER_DEFAULTS.drain_window_s,
        shutdown_join_timeout_s: float = _SCHEDULER_DEFAULTS.shutdown_join_timeout_s,
        resources: Any | None = None,
        skip_pid: Callable[[str], bool] | None = None,
        cancel_process: Callable[[str, str], None] | None = None,
        blocking_work: Any | None = None,
        owner_id: str = "scheduler.local",
        transitions: ProcessTransitionService | None = None,
    ):
        self.store = store
        self.audit = audit
        self.poll_interval_s = poll_interval_s
        self.max_workers = max_workers
        self.drain_window_s = drain_window_s
        self.shutdown_join_timeout_s = shutdown_join_timeout_s
        self.resources = resources
        self._skip_pid = skip_pid
        self._cancel_process = cancel_process
        self._blocking_work = blocking_work
        self.owner_id = str(owner_id)
        self._transitions = transitions or ProcessTransitionService(store)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-libos-scheduler")
        self._unblock_executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="agent-libos-scheduler-unblock",
        )
        self._executor_lock = threading.RLock()
        self._run_lock = threading.RLock()
        self._closed = False
        self._awaitable_lock = threading.RLock()
        self._awaitables: dict[str, _AwaitableHandle] = {}
        self._futures_lock = threading.RLock()
        self._futures: dict[Future[Any], str] = {}

    def next_runnable(self) -> str | None:
        runnable = self.store.list_processes_by_status(ProcessStatus.RUNNABLE)
        for process in runnable:
            if self._is_schedulable(process.pid):
                return process.pid
        return None

    def runnable_pids(self) -> list[str]:
        return [
            proc.pid
            for proc in self.store.list_processes_by_status(ProcessStatus.RUNNABLE)
            if self._is_schedulable(proc.pid)
        ]

    def _is_schedulable(self, pid: str) -> bool:
        return self._skip_pid is None or not self._skip_pid(pid)

    @contextmanager
    def quiescent_state(self, *, reason: str):
        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            raise ValidationError(f"{reason} refused while scheduler is running")
        try:
            active = self.active_pids()
            if active:
                raise ValidationError(f"{reason} refused while scheduler futures are active: {', '.join(active)}")
            yield
        finally:
            self._run_lock.release()

    def active_pids(self) -> list[str]:
        with self._futures_lock:
            return sorted({pid for future, pid in self._futures.items() if not future.done()})

    async def arun_once(self, quantum: Quantum) -> Any:
        return await self._run_blocking(self.run_once, quantum)

    def run_once(self, quantum: Quantum) -> Any:
        with self._run_lock:
            pid = self.next_runnable()
            if pid is None:
                return None
            future = self._submit(pid, lambda: self._run_quantum(pid, quantum))
            return future.result()

    async def arun_pid_once(self, pid: str, quantum: Quantum) -> Any:
        return await self._run_blocking(self.run_pid_once, pid, quantum)

    def run_pid_once(self, pid: str, quantum: Quantum) -> Any:
        """Advance one explicitly selected runnable process by one quantum."""
        with self._run_lock:
            process = self.store.get_process(pid)
            if process is None:
                raise ValidationError(f"process not found: {pid}")
            if process.status != ProcessStatus.RUNNABLE or not self._is_schedulable(pid):
                return {"ok": False, "skipped": True, "status": process.status.value}
            future = self._submit(pid, lambda: self._run_quantum(pid, quantum))
            return future.result()

    def is_active_quantum(self, pid: str) -> bool:
        return _ACTIVE_QUANTUM.get() == (id(self), pid)

    async def arun_until_idle(
        self,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
        *,
        cancel_inflight_on_budget_exhaustion: bool = True,
    ) -> list[Any]:
        return await self._run_blocking(
            self.run_until_idle,
            quantum,
            max_quanta=max_quanta,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )

    def run_until_idle(
        self,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
        *,
        cancel_inflight_on_budget_exhaustion: bool = True,
    ) -> list[Any]:
        with self._run_lock:
            return self._run_until_idle_locked(
                quantum,
                max_quanta=max_quanta,
                cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
            )

    def _run_until_idle_locked(
        self,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
        *,
        cancel_inflight_on_budget_exhaustion: bool = True,
    ) -> list[Any]:
        results: list[Any] = []
        futures: dict[str, Future[list[Any]]] = {}
        quanta_used = 0
        effective_max_quanta = max_quanta
        unblock_quanta_used = 0
        unblock_quanta_limit = max(1, max_quanta or 0) if max_quanta is not None else None
        drain_deadline: float | None = None
        drain_window_s = self.drain_window_s if max_quanta is not None else None
        quanta_lock = threading.Lock()

        def reserve_quantum() -> bool:
            # The quantum budget is global across process tasks, not per process.
            nonlocal quanta_used
            with quanta_lock:
                if _budget_exhausted(quanta_used, effective_max_quanta):
                    return False
                quanta_used += 1
                return True

        def budget_exhausted() -> bool:
            with quanta_lock:
                return _budget_exhausted(quanta_used, effective_max_quanta)

        def process_loop(pid: str, *, initial_reserved: bool) -> list[Any]:
            process_results: list[Any] = []
            has_reservation = initial_reserved
            while has_reservation or reserve_quantum():
                has_reservation = False
                process = self.store.get_process(pid)
                if process is None or process.status != ProcessStatus.RUNNABLE:
                    break
                try:
                    process_results.append(self._run_quantum(pid, quantum))
                except _QuantumCancelled:
                    raise
                except Exception as exc:
                    self._fail_process_task(pid, exc)
                    process_results.append({"ok": False, "pid": pid, "error": str(exc)})
                    break
                latest = self.store.get_process(pid)
                if latest is None or latest.status != ProcessStatus.RUNNABLE:
                    break
            return process_results

        while True:
            # Start one future per runnable pid. Each future keeps advancing its own
            # process until it blocks, exits, fails, or the shared budget is used.
            for pid in self.runnable_pids():
                if budget_exhausted():
                    break
                if pid not in futures and reserve_quantum():
                    futures[pid] = self._submit(pid, lambda selected_pid=pid: process_loop(selected_pid, initial_reserved=True))

            if self._collect_completed_futures(futures, results):
                drain_deadline = None
                continue

            if not futures:
                break

            done, _pending = wait(
                list(futures.values()),
                timeout=self.poll_interval_s,
                return_when=FIRST_COMPLETED,
            )
            if done:
                self._collect_completed_futures(futures, results, completed=done)
                drain_deadline = None
                continue

            if budget_exhausted():
                runnable_dependencies = [pid for pid in self.runnable_pids() if pid not in futures]
                if (
                    runnable_dependencies
                    and self._has_waiting_pending_future(futures)
                    and unblock_quanta_limit is not None
                    and unblock_quanta_used < unblock_quanta_limit
                ):
                    # A bounded run may have spent its nominal budget inside a
                    # parent quantum that is waiting for a child/message. Grant
                    # limited dependency quanta so the waiter can be unblocked.
                    unblock_quanta_used += 1
                    effective_max_quanta = (effective_max_quanta or 0) + 1
                    drain_deadline = None
                    self.audit.record(
                        actor="scheduler",
                        action="scheduler.unblock_quantum_reserved",
                        target="scheduler",
                        decision={
                            "quanta_used": quanta_used,
                            "max_quanta": max_quanta,
                            "unblock_quanta_used": unblock_quanta_used,
                            "runnable_dependencies": runnable_dependencies,
                        },
                    )
                    dependency_pid = runnable_dependencies[0]
                    if dependency_pid not in futures and reserve_quantum():
                        futures[dependency_pid] = self._submit(
                            dependency_pid,
                            lambda selected_pid=dependency_pid: process_loop(selected_pid, initial_reserved=True),
                            unblock=True,
                        )
                    continue
                keep_draining, drain_deadline = self._budget_drain_state(
                    futures=futures,
                    cancel_inflight=cancel_inflight_on_budget_exhaustion,
                    drain_window_s=drain_window_s,
                    drain_deadline=drain_deadline,
                )
                if keep_draining:
                    continue
                self._cancel_pending_futures(futures, results, reason="max_quanta_exhausted")
                break

        return results

    def _budget_drain_state(
        self,
        *,
        futures: dict[str, Future[list[Any]]],
        cancel_inflight: bool,
        drain_window_s: float | None,
        drain_deadline: float | None,
    ) -> tuple[bool, float | None]:
        # Host-managed incremental runners may use max_quanta only as a
        # completed-batch boundary. An admitted provider/tool quantum then
        # finishes normally while the budget still blocks another admission.
        if not cancel_inflight:
            return True, None
        if drain_window_s is None or not self._has_pending_future(futures):
            return False, drain_deadline
        # A wall-clock deadline avoids Windows event-loop timer granularity
        # turning a short bounded drain into a multi-second wait.
        now = time.perf_counter()
        selected_deadline = drain_deadline or now + drain_window_s
        return now < selected_deadline, selected_deadline

    async def arun_pid_until_idle(
        self,
        pid: str,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
    ) -> list[Any]:
        return await self._run_blocking(
            self.run_pid_until_idle,
            pid,
            quantum,
            max_quanta=max_quanta,
        )

    async def _run_blocking(self, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        if self._blocking_work is None:
            return await run_blocking_once(function, *args, **kwargs)
        return await self._blocking_work.run(function, *args, **kwargs)

    def run_pid_until_idle(
        self,
        pid: str,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
    ) -> list[Any]:
        """Advance one process until it blocks, exits, fails, or exhausts budget."""
        with self._run_lock:
            results: list[Any] = []
            quanta_used = 0
            while not _budget_exhausted(quanta_used, max_quanta):
                process = self.store.get_process(pid)
                if process is None or process.status != ProcessStatus.RUNNABLE:
                    break
                try:
                    quanta_used += 1
                    future = self._submit(pid, lambda: self._run_quantum(pid, quantum))
                    results.append(future.result())
                except (FutureCancelledError, _QuantumCancelled):
                    self._record_task_cancelled(pid, reason="cancelled")
                    break
                except Exception as exc:
                    self._fail_process_task(pid, exc)
                    results.append({"ok": False, "pid": pid, "error": str(exc)})
                    break
                latest = self.store.get_process(pid)
                if latest is None or latest.status != ProcessStatus.RUNNABLE:
                    break
            return results

    def shutdown(self) -> bool:
        with self._executor_lock:
            if self._closed:
                return self._all_futures_done()
            self._closed = True
        with self._futures_lock:
            pending = list(self._futures.items())
        for future, pid in pending:
            future.cancel()
            self._cancel_awaitable(pid)
        if pending:
            wait([future for future, _pid in pending], timeout=self.shutdown_join_timeout_s)
        stopped = self._all_futures_done()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._unblock_executor.shutdown(wait=False, cancel_futures=True)
        return stopped

    def _run_quantum(self, pid: str, quantum: Quantum) -> Any:
        execution_token = self._claim_runnable_process(pid)
        if execution_token is None:
            return None
        self.audit.record(actor="scheduler", action="scheduler.run_quantum", target=f"process:{pid}")
        started_at = time.perf_counter()
        result: Any = None
        error: BaseException | None = None
        resource_error: ResourceLimitExceeded | None = None
        try:
            token = _ACTIVE_QUANTUM.set((id(self), pid))
            try:
                with bind_process_execution(execution_token):
                    result = quantum(pid)
                    if inspect.isawaitable(result):
                        result = self._run_awaitable(pid, result)
            finally:
                _ACTIVE_QUANTUM.reset(token)
        except BaseException as exc:
            error = exc
        finally:
            if self.resources is not None:
                elapsed = max(0.0, time.perf_counter() - started_at)
                try:
                    self.resources.charge(
                        pid,
                        ResourceUsage(runtime_seconds=elapsed),
                        source="scheduler.quantum",
                        context={"elapsed_s": elapsed},
                        allow_overage=True,
                        kill_on_exceed=True,
                    )
                except ResourceLimitExceeded as exc:
                    resource_error = exc
            # A primitive may deliberately fence this lease by transitioning to
            # WAITING_HUMAN, EXITED, or another state.  Only the exact execution
            # token may restore RUNNABLE after a plain return.
            self.store.complete_execution(
                execution_token,
                status=ProcessStatus.RUNNABLE,
            )
        if error is not None:
            raise error
        if resource_error is not None:
            raise resource_error
        return result

    def _run_awaitable(self, pid: str, awaitable: Awaitable[Any]) -> Any:
        loop = asyncio.new_event_loop()
        task = asyncio.ensure_future(awaitable, loop=loop)
        handle = _AwaitableHandle(loop=loop, task=task)
        with self._awaitable_lock:
            self._awaitables[pid] = handle
        try:
            return loop.run_until_complete(task)
        except asyncio.CancelledError as exc:
            raise _QuantumCancelled("scheduler quantum cancelled") from exc
        finally:
            with self._awaitable_lock:
                if self._awaitables.get(pid) is handle:
                    self._awaitables.pop(pid, None)
            pending = [item for item in asyncio.all_tasks(loop) if not item.done()]
            for item in pending:
                item.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _claim_runnable_process(self, pid: str) -> ProcessExecutionToken | None:
        return self.store.claim_execution(pid, owner_id=self.owner_id)

    def _submit(self, pid: str, operation: Callable[[], Any], *, unblock: bool = False) -> Future[Any]:
        # ThreadPoolExecutor does not propagate ContextVars. Capture the host
        # run context at submission so it follows the process quantum.
        context = contextvars.copy_context()
        with self._executor_lock:
            if self._closed:
                raise RuntimeError("scheduler is shut down")
            executor = self._unblock_executor if unblock else self._executor
            future = executor.submit(context.run, operation)
        with self._futures_lock:
            self._futures[future] = pid
        future.add_done_callback(self._forget_future)
        return future

    def _all_futures_done(self) -> bool:
        with self._futures_lock:
            return all(future.done() for future in self._futures)

    def _forget_future(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.pop(future, None)

    def _fail_process_task(self, pid: str, exc: Exception) -> None:
        process = self.store.get_process(pid)
        if process is not None and process.status not in self.TERMINAL_STATUSES:
            message = f"scheduler task failed: {exc}"
            self._transitions.transition(
                pid,
                ProcessStatus.FAILED,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                outcome=FailedProcessOutcome(code="scheduler_task_failed"),
                status_message=message,
            )
        self.audit.record(
            actor="scheduler",
            action="scheduler.process_task_failed",
            target=f"process:{pid}",
            decision={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _collect_completed_futures(
        self,
        futures: dict[str, Future[list[Any]]],
        results: list[Any],
        *,
        completed: set[Future[list[Any]]] | None = None,
    ) -> bool:
        selected = completed or {future for future in futures.values() if future.done()}
        if not selected:
            return False
        for pid, future in list(futures.items()):
            if future not in selected and not future.done():
                continue
            futures.pop(pid, None)
            self._consume_future_result(pid, future, results)
        return True

    def _consume_future_result(self, pid: str, future: Future[list[Any]], results: list[Any]) -> None:
        try:
            outcome = future.result()
        except (FutureCancelledError, _QuantumCancelled):
            self._record_task_cancelled(pid, reason="cancelled")
        except Exception as exc:
            self._fail_process_task(pid, exc)
            results.append({"ok": False, "pid": pid, "error": str(exc)})
        else:
            if isinstance(outcome, list):
                results.extend(outcome)
            else:
                results.append(outcome)

    def _cancel_pending_futures(
        self,
        futures: dict[str, Future[list[Any]]],
        results: list[Any],
        *,
        reason: str,
    ) -> None:
        for pid, future in list(futures.items()):
            self._cancel_awaitable(pid)
            cancelled = future.cancel()
            futures.pop(pid, None)
            if cancelled:
                self._record_task_cancelled(pid, reason=reason)
            elif future.done():
                self._consume_future_result(pid, future, results)
            else:
                self._record_task_cancelled(pid, reason=reason, detached=True)

    def _cancel_awaitable(self, pid: str) -> None:
        with self._awaitable_lock:
            handle = self._awaitables.get(pid)
        if handle is None:
            return
        try:
            handle.loop.call_soon_threadsafe(handle.task.cancel)
        except RuntimeError:
            return

    def _record_task_cancelled(self, pid: str, *, reason: str, detached: bool = False) -> None:
        decision: dict[str, Any] = {"reason": reason}
        if detached:
            decision["detached"] = True
            if self._cancel_process is not None:
                try:
                    self._cancel_process(pid, reason)
                    decision["process_cancelled"] = True
                except Exception as exc:
                    decision["process_cancel_error"] = str(exc)
                    decision["process_cancel_error_type"] = type(exc).__name__
        self.audit.record(
            actor="scheduler",
            action="scheduler.process_task_cancelled",
            target=f"process:{pid}",
            decision=decision,
        )

    def _has_pending_future(self, futures: dict[str, Future[list[Any]]]) -> bool:
        for future in futures.values():
            if not future.done():
                return True
        return False

    def _has_waiting_pending_future(self, futures: dict[str, Future[list[Any]]]) -> bool:
        for pid, future in futures.items():
            if future.done():
                continue
            process = self.store.get_process(pid)
            if process is not None and process.status in self.WAITING_STATUSES:
                return True
        return False


class SimpleScheduler(AsyncProcessScheduler):
    pass


@dataclass(frozen=True)
class _AwaitableHandle:
    loop: asyncio.AbstractEventLoop
    task: asyncio.Task[Any]


class _QuantumCancelled(Exception):
    pass


def _budget_exhausted(quanta_used: int, max_quanta: int | None) -> bool:
    return max_quanta is not None and quanta_used >= max_quanta
