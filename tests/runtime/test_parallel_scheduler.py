from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SchedulerDefaults
from agent_libos.models import ChildProcessWait, ProcessStatus, ResourceBudget
from agent_libos.models.exceptions import ProcessRevisionConflict


def _parallel_config(max_workers: int = 4) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        scheduler=SchedulerDefaults(
            max_workers=max_workers,
            poll_interval_s=0.001,
            drain_window_s=0.3,
            shutdown_join_timeout_s=0.5,
        )
    )


class TestParallelScheduler:
    def test_awaitable_quantum_drains_default_executor_before_closing_loop(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=1))
        pid = runtime.process.spawn(goal="default executor drain")
        executor_started = threading.Event()
        release_executor = threading.Event()
        scheduler_finished = threading.Event()
        errors: list[BaseException] = []
        try:
            def default_executor_work() -> None:
                executor_started.set()
                assert release_executor.wait(timeout=2)

            async def quantum(selected_pid: str) -> dict[str, str]:
                asyncio.get_running_loop().run_in_executor(
                    None,
                    default_executor_work,
                )
                return {"pid": selected_pid}

            def run_quantum() -> None:
                try:
                    runtime.scheduler.run_pid_once(pid, quantum)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    scheduler_finished.set()

            scheduler_thread = threading.Thread(target=run_quantum)
            scheduler_thread.start()
            assert executor_started.wait(timeout=2)
            assert not scheduler_finished.wait(timeout=0.1)

            release_executor.set()
            assert scheduler_finished.wait(timeout=2)
            scheduler_thread.join(timeout=2)

            assert not scheduler_thread.is_alive()
            assert errors == []
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            release_executor.set()
            runtime.close()

    def test_resource_charge_failure_releases_execution_lease(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(goal="resource charge failure")

            def fail_charge(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected resource accounting failure")

            runtime.resources.charge = fail_charge  # type: ignore[method-assign]

            with pytest.raises(
                RuntimeError,
                match="injected resource accounting failure",
            ):
                runtime.scheduler.run_pid_once(pid, lambda selected_pid: selected_pid)

            process = runtime.process.get(pid)
            assert process.status == ProcessStatus.RUNNABLE
            assert process.execution_owner_id is None
            assert process.execution_lease_id is None
        finally:
            runtime.close()

    def test_terminal_process_rejects_direct_tool_table_mutation(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(goal="terminal process tool fence")
            runtime.process.cancel(pid, "terminal fence regression")
            terminal = runtime.process.get(pid)

            with pytest.raises(ProcessRevisionConflict, match="terminal process"):
                runtime.tools.configure_process_tools(
                    pid,
                    ["human_output"],
                    assigned_by="late host mutation",
                )

            current = runtime.process.get(pid)
            assert current.status == ProcessStatus.KILLED
            assert current.revision == terminal.revision
            assert current.tool_table == terminal.tool_table
        finally:
            runtime.close()

    def test_concurrent_direct_single_step_calls_do_not_reenter_same_process(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=2))
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="direct single step")
            active = 0
            max_active = 0
            calls = 0
            lock = threading.Lock()
            start = threading.Barrier(2)
            errors: list[BaseException] = []

            async def quantum(selected_pid: str) -> dict[str, str]:
                nonlocal active, max_active, calls
                assert selected_pid == pid
                with lock:
                    active += 1
                    calls += 1
                    max_active = max(max_active, active)
                try:
                    await asyncio.sleep(0.05)
                    return {"pid": selected_pid}
                finally:
                    with lock:
                        active -= 1

            runtime.llm.arun_once = quantum  # type: ignore[method-assign]

            def runner() -> None:
                try:
                    start.wait(timeout=1.0)
                    runtime.run_process_once(pid)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=runner) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)

            assert errors == []
            assert calls == 2
            assert max_active == 1
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_blocking_quantum_does_not_block_other_process(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=2))
        try:
            slow = runtime.process.spawn(image="base-agent:v0", goal="slow")
            fast = runtime.process.spawn(image="base-agent:v0", goal="fast")
            slow_started = threading.Event()
            marks: dict[str, float] = {}
            lock = threading.Lock()

            def mark(name: str) -> None:
                with lock:
                    marks[name] = time.perf_counter()

            def quantum(pid: str) -> dict[str, str]:
                if pid == slow:
                    slow_started.set()
                    time.sleep(0.15)
                    runtime.process.pause(pid, "slow quantum complete")
                    mark("slow_done")
                    return {"pid": pid, "kind": "slow"}
                assert slow_started.wait(timeout=0.5)
                runtime.process.exit(pid)
                mark("fast_done")
                return {"pid": pid, "kind": "fast"}

            results = runtime.scheduler.run_until_idle(quantum, max_quanta=2)

            assert {result["kind"] for result in results if isinstance(result, dict)} == {"slow", "fast"}
            assert marks["fast_done"] < marks["slow_done"]
        finally:
            runtime.close()

    def test_same_process_quantum_is_not_reentered(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=4))
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="single process")
            active = 0
            max_active = 0
            calls = 0
            lock = threading.Lock()

            def quantum(selected_pid: str) -> dict[str, int | str]:
                nonlocal active, max_active, calls
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.02)
                    with lock:
                        calls += 1
                        call_index = calls
                    if call_index >= 2:
                        runtime.process.pause(selected_pid, "done")
                    return {"pid": selected_pid, "call": call_index}
                finally:
                    with lock:
                        active -= 1

            results = runtime.scheduler.run_until_idle(quantum, max_quanta=2)

            assert [result["call"] for result in results if isinstance(result, dict)] == [1, 2]
            assert max_active == 1
        finally:
            runtime.close()

    def test_parallel_scheduler_respects_global_quantum_budget(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=4))
        try:
            for index in range(4):
                runtime.process.spawn(image="base-agent:v0", goal=f"process {index}")

            def quantum(pid: str) -> dict[str, str]:
                runtime.process.pause(pid, "budget consumed")
                return {"pid": pid}

            results = runtime.scheduler.run_until_idle(quantum, max_quanta=2)
            run_records = [record for record in runtime.audit.trace() if record.action == "scheduler.run_quantum"]

            assert len([result for result in results if isinstance(result, dict)]) == 2
            assert len(run_records) == 2
        finally:
            runtime.close()

    def test_waiting_parent_does_not_starve_child_when_worker_pool_is_full(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=1))
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            state: dict[str, str] = {}

            async def quantum(pid: str) -> dict[str, str]:
                if pid == parent:
                    child = runtime.process.spawn_child(
                        parent,
                        goal="child",
                        resource_budget=ResourceBudget(max_tool_calls=0, max_child_processes=None),
                    )
                    state["child"] = child
                    process = runtime.process.get(parent)
                    runtime.process_transitions.transition(
                        parent,
                        ProcessStatus.WAITING_EVENT,
                        expected_revision=process.revision,
                        expected_status=process.status,
                        wait_state=ChildProcessWait(child_pid=child),
                    )
                    while runtime.process.get(child).status not in runtime.process.TERMINAL_STATUSES:
                        await asyncio.sleep(runtime.scheduler.poll_interval_s)
                    return {"pid": pid, "done": "parent"}
                runtime.process.exit(pid)
                return {"pid": pid, "done": "child"}

            results = asyncio.run(runtime.scheduler.arun_until_idle(quantum, max_quanta=1))

            assert {item["done"] for item in results if isinstance(item, dict) and "done" in item} == {
                "parent",
                "child",
            }
        finally:
            runtime.close()

    def test_concurrent_top_level_runs_do_not_reenter_same_process(self) -> None:
        runtime = Runtime.open("local", config=_parallel_config(max_workers=4))
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="single process")
            active = 0
            max_active = 0
            lock = threading.Lock()
            start = threading.Barrier(2)
            errors: list[BaseException] = []

            def quantum(selected_pid: str) -> dict[str, str]:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.05)
                    runtime.process.pause(selected_pid, "done")
                    return {"pid": selected_pid}
                finally:
                    with lock:
                        active -= 1

            def runner() -> None:
                try:
                    start.wait(timeout=1.0)
                    runtime.scheduler.run_until_idle(quantum, max_quanta=1)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=runner) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1.0)

            assert errors == []
            assert max_active == 1
        finally:
            runtime.close()

    def test_shutdown_leaves_store_open_when_sync_quantum_cannot_stop(self) -> None:
        config = AgentLibOSConfig(
            scheduler=SchedulerDefaults(
                max_workers=1,
                poll_interval_s=0.001,
                drain_window_s=0.01,
                shutdown_join_timeout_s=0.01,
            )
        )
        runtime = Runtime.open("local", config=config)
        pid = runtime.process.spawn(image="base-agent:v0", goal="slow sync")
        try:
            def quantum(selected_pid: str) -> dict[str, str]:
                time.sleep(0.1)
                runtime.process.exit(selected_pid)
                return {"pid": selected_pid}

            assert runtime.scheduler.run_until_idle(quantum, max_quanta=1) == []
            result = runtime.shutdown(actor="test", reason="detached quantum")

            assert result["ok"] is False
            assert result["scheduler_stopped"] is False
            assert runtime.store.get_process(pid) is not None
            time.sleep(0.15)
            assert runtime.shutdown(actor="test", reason="detached quantum complete")["ok"] is True
        finally:
            runtime.close()

    def test_detached_quantum_token_cannot_mutate_after_external_cancel(self) -> None:
        runtime = Runtime.open(
            "local",
            config=AgentLibOSConfig(
                scheduler=SchedulerDefaults(
                    max_workers=1,
                    poll_interval_s=0.001,
                    drain_window_s=0.01,
                    shutdown_join_timeout_s=0.5,
                )
            ),
        )
        pid = runtime.process.spawn(goal="detached execution fence")
        started = threading.Event()
        release = threading.Event()
        rejected: list[BaseException] = []
        tool_rejected: list[BaseException] = []
        scheduler_errors: list[BaseException] = []
        try:
            def quantum(selected_pid: str) -> dict[str, str]:
                started.set()
                assert release.wait(timeout=2)
                latest = runtime.process.get(selected_pid)
                latest.status_message = "stale detached worker write"
                try:
                    runtime.store.update_process(latest)
                except BaseException as error:
                    rejected.append(error)
                try:
                    runtime.tools.configure_process_tools(
                        selected_pid,
                        ['human_output'],
                        assigned_by='stale detached worker',
                    )
                except BaseException as error:
                    tool_rejected.append(error)
                return {"pid": selected_pid}

            def run_scheduler() -> None:
                try:
                    runtime.scheduler.run_until_idle(quantum, max_quanta=1)
                except BaseException as error:
                    scheduler_errors.append(error)

            scheduler_thread = threading.Thread(target=run_scheduler)
            scheduler_thread.start()
            assert started.wait(timeout=1)
            runtime.process.cancel(pid, "external cancel wins")
            cancelled = runtime.process.get(pid)
            release.set()
            scheduler_thread.join(timeout=2)

            deadline = time.monotonic() + 2
            while (not rejected or not tool_rejected) and time.monotonic() < deadline:
                time.sleep(0.01)

            process = runtime.process.get(pid)
            assert not scheduler_thread.is_alive()
            assert scheduler_errors == []
            assert len(rejected) == 1
            assert isinstance(rejected[0], ProcessRevisionConflict)
            assert len(tool_rejected) == 1
            assert isinstance(tool_rejected[0], ProcessRevisionConflict)
            assert process.status == ProcessStatus.KILLED
            assert process.status_message == cancelled.status_message
            assert process.revision >= cancelled.revision
            assert process.status_message != "stale detached worker write"
            assert process.tool_table == cancelled.tool_table
        finally:
            release.set()
            runtime.close()
