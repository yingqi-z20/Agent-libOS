from __future__ import annotations

import asyncio
import contextvars
import inspect
import math
import threading
import time
from contextlib import contextmanager
from collections.abc import Awaitable, Callable, Iterable
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
from agent_libos.models.exceptions import ValidationError
from agent_libos.process_execution import bind_process_execution
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.storage import ProcessRepository
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)


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
        terminal_cleanup: Callable[[str], Any] | None = None,
    ):
        self._validate_constructor_arguments(
            poll_interval_s=poll_interval_s,
            max_workers=max_workers,
            drain_window_s=drain_window_s,
            shutdown_join_timeout_s=shutdown_join_timeout_s,
            owner_id=owner_id,
        )
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
        self.owner_id = owner_id
        self._transitions = transitions or ProcessTransitionService(store)
        self._terminal_cleanup = terminal_cleanup
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

    def next_runnable(self, *, pids: Iterable[str] | None = None) -> str | None:
        selected_pids = self._normalize_pid_scope(pids)
        runnable = self.store.list_processes_by_status(ProcessStatus.RUNNABLE)
        for process in runnable:
            if (
                (selected_pids is None or process.pid in selected_pids)
                and self._is_schedulable(process.pid)
            ):
                return process.pid
        return None

    def runnable_pids(self, *, pids: Iterable[str] | None = None) -> list[str]:
        selected_pids = self._normalize_pid_scope(pids)
        return [
            proc.pid
            for proc in self.store.list_processes_by_status(ProcessStatus.RUNNABLE)
            if (selected_pids is None or proc.pid in selected_pids)
            and self._is_schedulable(proc.pid)
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
        pids: Iterable[str] | None = None,
    ) -> list[Any]:
        selected_pids = self._normalize_pid_scope(pids)
        self.validate_run_controls(
            max_quanta=max_quanta,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )
        return await self._run_blocking(
            self.run_until_idle,
            quantum,
            max_quanta=max_quanta,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
            pids=selected_pids,
        )

    def run_until_idle(
        self,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
        *,
        cancel_inflight_on_budget_exhaustion: bool = True,
        pids: Iterable[str] | None = None,
    ) -> list[Any]:
        self.validate_run_controls(
            max_quanta=max_quanta,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )
        selected_pids = self._normalize_pid_scope(pids)
        with self._run_lock:
            return self._run_until_idle_locked(
                quantum,
                max_quanta=max_quanta,
                cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
                pids=selected_pids,
            )

    def _run_until_idle_locked(
        self,
        quantum: Quantum,
        max_quanta: int | None = _SCHEDULER_DEFAULTS.max_quanta,
        *,
        cancel_inflight_on_budget_exhaustion: bool = True,
        pids: frozenset[str] | None = None,
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
                    process_results.append(
                        self._run_quantum(pid, quantum, pids=pids)
                    )
                except _QuantumCancelled:
                    raise
                except Exception as exc:
                    public_error = self._fail_process_task(pid, exc)
                    process_results.append(
                        self._failure_result(pid, public_error)
                    )
                    break
                latest = self.store.get_process(pid)
                if latest is None or latest.status != ProcessStatus.RUNNABLE:
                    break
            return process_results

        while True:
            # Start one future per runnable pid. Each future keeps advancing its own
            # process until it blocks, exits, fails, or the shared budget is used.
            for pid in self.runnable_pids(pids=pids):
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
                runnable_dependencies = [
                    pid
                    for pid in self.runnable_pids(pids=pids)
                    if pid not in futures
                ]
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
        self.validate_max_quanta(max_quanta)
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
        self.validate_max_quanta(max_quanta)
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
                    public_error = self._fail_process_task(pid, exc)
                    results.append(self._failure_result(pid, public_error))
                    break
                latest = self.store.get_process(pid)
                if latest is None or latest.status != ProcessStatus.RUNNABLE:
                    break
            return results

    @staticmethod
    def validate_max_quanta(max_quanta: int | None) -> None:
        if max_quanta is not None and (
            type(max_quanta) is not int or max_quanta <= 0
        ):
            raise ValidationError("scheduler max_quanta must be a positive integer or null")

    @staticmethod
    def validate_run_controls(
        *,
        max_quanta: int | None,
        cancel_inflight_on_budget_exhaustion: bool,
    ) -> None:
        AsyncProcessScheduler.validate_max_quanta(max_quanta)
        if type(cancel_inflight_on_budget_exhaustion) is not bool:
            raise ValidationError(
                "scheduler cancel_inflight_on_budget_exhaustion must be a boolean"
            )

    @staticmethod
    def _validate_constructor_arguments(
        *,
        poll_interval_s: float,
        max_workers: int,
        drain_window_s: float,
        shutdown_join_timeout_s: float,
        owner_id: str,
    ) -> None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ValidationError("scheduler max_workers must be a positive integer")
        for name, value, allow_zero in (
            ("poll_interval_s", poll_interval_s, False),
            ("drain_window_s", drain_window_s, True),
            ("shutdown_join_timeout_s", shutdown_join_timeout_s, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (value < 0 if allow_zero else value <= 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValidationError(
                    f"scheduler {name} must be a finite {qualifier} number"
                )
        if (
            not isinstance(owner_id, str)
            or not owner_id
            or owner_id != owner_id.strip()
            or "\x00" in owner_id
        ):
            raise ValidationError("scheduler owner_id must be a canonical non-empty string")

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

    def _run_quantum(
        self,
        pid: str,
        quantum: Quantum,
        *,
        pids: frozenset[str] | None = None,
    ) -> Any:
        execution_token = self._claim_runnable_process(pid, pids=pids)
        if execution_token is None:
            return None
        started_at = time.perf_counter()
        result: Any = None
        error: BaseException | None = None
        resource_error: BaseException | None = None
        try:
            # Establish the release finally immediately after the durable
            # claim. Even an observability failure must not strand RUNNING.
            self.audit.record(
                actor="scheduler",
                action="scheduler.run_quantum",
                target=f"process:{pid}",
            )
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
            try:
                if self.resources is not None:
                    elapsed = max(0.0, time.perf_counter() - started_at)
                    self.resources.charge(
                        pid,
                        ResourceUsage(runtime_seconds=elapsed),
                        source="scheduler.quantum",
                        context={"elapsed_s": elapsed},
                        allow_overage=True,
                        kill_on_exceed=True,
                    )
            except BaseException as exc:
                resource_error = exc
            finally:
                # A primitive may deliberately fence this lease by transitioning to
                # WAITING_HUMAN, EXITED, or another state.  Only the exact execution
                # token may restore RUNNABLE after a plain return.  Resource
                # accounting is deliberately outside this inner finally: even a
                # broken accounting adapter must not strand the process RUNNING.
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
        result: Any = None
        primary_error: BaseException | None = None
        with self._awaitable_lock:
            self._awaitables[pid] = handle
        try:
            try:
                result = loop.run_until_complete(task)
            except asyncio.CancelledError as exc:
                primary_error = _QuantumCancelled("scheduler quantum cancelled")
                primary_error.__cause__ = exc
            except BaseException as exc:
                primary_error = exc
        finally:
            with self._awaitable_lock:
                if self._awaitables.get(pid) is handle:
                    self._awaitables.pop(pid, None)

            cleanup_deadline = time.monotonic() + max(
                0.0,
                float(self.shutdown_join_timeout_s),
            )
            cleanup_failures: list[str] = []

            try:
                pending = _drain_asyncio_tasks(
                    loop,
                    deadline=cleanup_deadline,
                    cancellation_slice_s=max(
                        0.001,
                        min(0.01, float(self.poll_interval_s)),
                    ),
                )
            except BaseException:
                cleanup_failures.append("pending_tasks")
                pending = _pending_asyncio_tasks(loop)
            if pending:
                cleanup_failures.append("pending_tasks")
                _abandon_asyncio_tasks(pending)

            async_generators = getattr(loop, "_asyncgens", ())
            if async_generators:
                remaining = _remaining_deadline_s(cleanup_deadline)
                if remaining <= 0:
                    cleanup_failures.append("async_generators")
                else:
                    shutdown_asyncgens = loop.create_task(
                        loop.shutdown_asyncgens()
                    )
                    try:
                        done, _pending = loop.run_until_complete(
                            asyncio.wait({shutdown_asyncgens}, timeout=remaining)
                        )
                    except BaseException:
                        cleanup_failures.append("async_generators")
                        done = set()
                    _consume_asyncio_tasks(set(done))
                    if not shutdown_asyncgens.done():
                        cleanup_failures.append("async_generators")
                        _abandon_asyncio_tasks(_pending_asyncio_tasks(loop))

            default_executor = getattr(loop, "_default_executor", None)
            if default_executor is not None:
                # ``loop.shutdown_default_executor()`` joins its helper thread
                # in a coroutine ``finally`` block on Python 3.11+, so wrapping
                # it in wait_for cannot bound a non-cooperative executor. Own
                # the wait in a daemon monitor and retain a scheduler lifecycle
                # fence until every executor thread has actually stopped.
                setattr(loop, "_default_executor", None)
                executor_done, executor_errors = _begin_executor_cleanup(
                    self,
                    pid=pid,
                    executor=default_executor,
                )
                if not executor_done.wait(
                    timeout=_remaining_deadline_s(cleanup_deadline)
                ):
                    cleanup_failures.append("default_executor")
                elif executor_errors:
                    cleanup_failures.append("default_executor")

            remaining_tasks = _pending_asyncio_tasks(loop)
            if remaining_tasks:
                _abandon_asyncio_tasks(remaining_tasks)
            try:
                loop.close()
            except BaseException:
                cleanup_failures.append("event_loop")

        if cleanup_failures:
            stages = ",".join(dict.fromkeys(cleanup_failures))
            cleanup_error = RuntimeError(
                "awaitable cleanup did not complete before the scheduler "
                f"shutdown deadline (stages={stages})"
            )
            if primary_error is not None:
                raise cleanup_error from primary_error
            raise cleanup_error
        if primary_error is not None:
            raise primary_error
        return result

    def _claim_runnable_process(
        self,
        pid: str,
        *,
        pids: frozenset[str] | None = None,
    ) -> ProcessExecutionToken | None:
        if pids is not None and pid not in pids:
            return None
        return self.store.claim_execution(pid, owner_id=self.owner_id)

    @staticmethod
    def _normalize_pid_scope(
        pids: Iterable[str] | None,
    ) -> frozenset[str] | None:
        if pids is None:
            return None
        if isinstance(pids, (str, bytes)):
            raise ValidationError("scheduler pids must be a collection of process ids")
        try:
            selected = tuple(pids)
        except TypeError as exc:
            raise ValidationError(
                "scheduler pids must be a collection of process ids"
            ) from exc
        checked: list[str] = []
        for pid in selected:
            if (
                type(pid) is not str
                or not pid
                or pid != pid.strip()
                or "\x00" in pid
            ):
                raise ValidationError(
                    "scheduler pids must contain canonical non-empty process ids"
                )
            checked.append(pid)
        if len(checked) != len(set(checked)):
            raise ValidationError("scheduler pids must be unique")
        return frozenset(checked)

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

    def _fail_process_task(
        self,
        pid: str,
        exc: Exception,
    ) -> dict[str, str]:
        public_error = public_error_envelope(exc)
        process = self.store.get_process(pid)
        terminalized = False
        if process is not None and process.status not in self.TERMINAL_STATUSES:
            message = f"scheduler task failed: {public_error['message']}"
            self._transitions.transition(
                pid,
                ProcessStatus.FAILED,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                outcome=FailedProcessOutcome(code="scheduler_task_failed"),
                status_message=message,
            )
            terminalized = True
        self.audit.record(
            actor="scheduler",
            action="scheduler.process_task_failed",
            target=f"process:{pid}",
            decision={
                "public_error": dict(public_error),
                "internal_error": internal_exception_observation(exc),
            },
            correlation_id=public_error["correlation_id"],
        )
        if terminalized and self._terminal_cleanup is not None:
            try:
                self._terminal_cleanup(pid)
            except BaseException:
                # The durable cleanup row and ProcessManager audit retain the
                # secondary failure.  Never replace the scheduler task's
                # already-committed FAILED outcome or its safe public envelope.
                pass
        return public_error

    @staticmethod
    def _failure_result(
        pid: str,
        public_error: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "pid": pid,
            "error": public_error["message"],
            **{
                key: public_error[key]
                for key in ("code", "error_type", "correlation_id")
            },
        }

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
            public_error = self._fail_process_task(pid, exc)
            results.append(self._failure_result(pid, public_error))
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
        correlation_id: str | None = None
        if detached:
            decision["detached"] = True
            if self._cancel_process is not None:
                try:
                    self._cancel_process(pid, reason)
                    decision["process_cancelled"] = True
                except Exception as exc:
                    public_error = public_error_envelope(exc)
                    correlation_id = public_error["correlation_id"]
                    decision["process_cancel_public_error"] = public_error
                    decision["process_cancel_internal_error"] = (
                        internal_exception_observation(exc)
                    )
        self.audit.record(
            actor="scheduler",
            action="scheduler.process_task_cancelled",
            target=f"process:{pid}",
            decision=decision,
            correlation_id=correlation_id,
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


def _remaining_deadline_s(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _pending_asyncio_tasks(
    loop: asyncio.AbstractEventLoop,
) -> set[asyncio.Task[Any]]:
    return {
        selected_task
        for selected_task in asyncio.all_tasks(loop)
        if not selected_task.done()
    }


def _consume_asyncio_tasks(selected: set[asyncio.Task[Any]]) -> None:
    for selected_task in selected:
        if not selected_task.done():
            continue
        try:
            selected_task.exception()
        except BaseException:
            pass


def _abandon_asyncio_tasks(selected: set[asyncio.Task[Any]]) -> None:
    for selected_task in selected:
        if selected_task.done():
            _consume_asyncio_tasks({selected_task})
            continue
        selected_task.cancel()
        coroutine = selected_task.get_coro()
        try:
            if (
                inspect.iscoroutine(coroutine)
                and inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
            ):
                coroutine.close()
        except BaseException:
            pass
        # asyncio has no public force-terminal API for a Task that deliberately
        # rejects cancellation. Its loop is closed immediately after this call,
        # so it cannot execute again; suppress only the misleading destruction
        # diagnostic for that intentionally abandoned Task.
        try:
            selected_task._log_destroy_pending = False  # type: ignore[attr-defined]
        except BaseException:
            pass


def _drain_asyncio_tasks(
    loop: asyncio.AbstractEventLoop,
    *,
    deadline: float,
    cancellation_slice_s: float,
) -> set[asyncio.Task[Any]]:
    pending = _pending_asyncio_tasks(loop)
    while pending:
        remaining = _remaining_deadline_s(deadline)
        if remaining <= 0:
            break
        for selected_task in pending:
            selected_task.cancel()
        done, _still_pending = loop.run_until_complete(
            asyncio.wait(
                pending,
                timeout=min(remaining, cancellation_slice_s),
            )
        )
        _consume_asyncio_tasks(set(done))
        pending = _pending_asyncio_tasks(loop)
    return pending


def _begin_executor_cleanup(
    scheduler: AsyncProcessScheduler,
    *,
    pid: str,
    executor: Any,
) -> tuple[threading.Event, list[BaseException]]:
    executor_done = threading.Event()
    executor_errors: list[BaseException] = []
    lifecycle_fence: Future[Any] = Future()
    lifecycle_fence.set_running_or_notify_cancel()
    with scheduler._futures_lock:
        scheduler._futures[lifecycle_fence] = pid
    lifecycle_fence.add_done_callback(scheduler._forget_future)

    def finish_default_executor() -> None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as exc:
            executor_errors.append(exc)
        else:
            # A lifecycle fence may report completion only after the blocking
            # shutdown call proves that every worker has stopped.  If the join
            # itself fails, retain the running fence so Runtime shutdown stays
            # conservatively incomplete.
            lifecycle_fence.set_result(None)
        finally:
            executor_done.set()

    monitor = threading.Thread(
        target=finish_default_executor,
        name="agent-libos-awaitable-executor-cleanup",
        daemon=True,
    )
    try:
        monitor.start()
    except BaseException as exc:
        executor_errors.append(exc)
        executor_done.set()
    return executor_done, executor_errors


def _budget_exhausted(quanta_used: int, max_quanta: int | None) -> bool:
    return max_quanta is not None and quanta_used >= max_quanta
