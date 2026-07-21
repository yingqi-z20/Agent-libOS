import asyncio
import copy
import inspect
import threading
import time
from contextlib import nullcontext
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from agent_libos.config import AgentLibOSConfig, SchedulerDefaults
from agent_libos.capability import (
    CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS,
    CAPABILITY_MANAGER_MIXED_PUBLIC_METHODS,
    CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS,
    CAPABILITY_MANAGER_READ_ONLY_PUBLIC_METHODS,
    CAPABILITY_MUTATION_SERVICE_PUBLIC_METHODS,
    CapabilityLeaseService,
    CapabilityManager,
    CapabilityMutationService,
)
from agent_libos.models import EventType, ObjectType, ProcessStatus, SinkTrustLevel, SinkTrustRule
from agent_libos.models.exceptions import RuntimeRecoveryRequired
from agent_libos.human.manager import HumanObjectManager
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.builder import (
    RuntimeAssemblyCleanupRequired,
    RuntimeBuilder,
)
from agent_libos.runtime.operation_manager import OperationManager
from agent_libos.runtime.boundary_descriptors import (
    CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    DATA_FLOW_COMPOSITION_PUBLIC_METHODS,
    DATA_FLOW_CONTEXT_PUBLIC_METHODS,
    DATA_FLOW_PUBLIC_MUTATION_METHODS,
    DATA_FLOW_READ_ONLY_PUBLIC_METHODS,
    EXPLAIN_BOUNDARY_DESCRIPTORS,
    HUMAN_PUBLIC_MUTATION_METHODS,
    HUMAN_READ_ONLY_PUBLIC_METHODS,
    OBJECT_TASK_LIFECYCLE_PUBLIC_METHODS,
    OBJECT_TASK_PUBLIC_MUTATION_METHODS,
    OBJECT_TASK_READ_ONLY_PUBLIC_METHODS,
    OBJECT_TASK_RECOVERY_PUBLIC_METHODS,
    PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES,
    RUNTIME_CONTEXT_PUBLIC_METHODS,
    RUNTIME_LIFECYCLE_PUBLIC_METHODS,
    RUNTIME_PUBLIC_MUTATION_METHODS,
    RUNTIME_READ_ONLY_PUBLIC_METHODS,
    SCHEDULER_COORDINATION_PUBLIC_METHODS,
    SCHEDULER_LIFECYCLE_PUBLIC_METHODS,
    SCHEDULER_PUBLIC_MUTATION_METHODS,
    SCHEDULER_READ_ONLY_PUBLIC_METHODS,
)
from agent_libos.runtime.data_flow_manager import DataFlowManager
from agent_libos.runtime.object_tasks import ObjectTaskManager
from agent_libos.runtime.scheduler import AsyncProcessScheduler
from agent_libos.modules import RuntimeModuleRegistry
from agent_libos.sdk import ProtectedOperationSDK
from agent_libos.storage import SQLiteStore


def _open_test_lifecycle(
    *,
    store: SQLiteStore | None = None,
    substrate: object | None = None,
    scheduler: object | None = None,
    object_tasks: object | None = None,
    modules: object | None = None,
    llms: object | None = None,
    blocking_work: object | None = None,
) -> tuple[SQLiteStore, RuntimeLifecycle, object]:
    selected_store = store or SQLiteStore(":memory:")
    operations = OperationManager(selected_store)
    lifecycle = RuntimeLifecycle(
        store=selected_store,
        audit=AuditManager(selected_store, operations),
        events=EventBus(selected_store, operations),
        substrate=substrate,
    )
    lifecycle.bind_components(
        scheduler=scheduler,
        object_tasks=object_tasks,
        modules=modules,
        llms=llms,
        blocking_work=blocking_work,
    )
    lifecycle.begin_recovery()
    lifecycle.begin_starting()
    lifecycle.mark_open()
    capability = lifecycle._issue_recovery_diagnostics_release_capability()
    return selected_store, lifecycle, capability


def _starting_test_lifecycle(
    *,
    lifecycle_type: type[RuntimeLifecycle] = RuntimeLifecycle,
) -> tuple[SQLiteStore, RuntimeLifecycle]:
    store = SQLiteStore(":memory:")
    operations = OperationManager(store)
    lifecycle = lifecycle_type(
        store=store,
        audit=AuditManager(store, operations),
        events=EventBus(store, operations),
        substrate=object(),
    )
    lifecycle.begin_recovery()
    lifecycle.begin_starting()
    return store, lifecycle


def _mark_test_recovery_fence(
    lifecycle: RuntimeLifecycle,
    publication_id: str,
) -> str:
    with lifecycle.admit():
        lifecycle.mark_recovery_required(publication_id=publication_id)
    return f"runtime.recovery_required:{publication_id}"


class _FirstRecoveryCloseProbeBarrierStore(SQLiteStore):
    """Delay the first close probe until a competing handoff has completed."""

    def __init__(self) -> None:
        self.first_probe_entered = threading.Event()
        self.release_first_probe = threading.Event()
        self._probe_count_lock = threading.Lock()
        self.probe_count = 0
        super().__init__(":memory:")

    def probe_admission_guard_close(self, expected_guard):
        with self._probe_count_lock:
            self.probe_count += 1
            is_first_probe = self.probe_count == 1
        if is_first_probe:
            self.first_probe_entered.set()
            assert self.release_first_probe.wait(timeout=2)
        return super().probe_admission_guard_close(expected_guard)

class TestRuntimeShutdown:

    def test_mark_open_interrupt_between_assignments_restores_starting(self) -> None:
        class _InterruptEverOpenedAssignment(RuntimeLifecycle):
            inject_interrupt = False

            def __setattr__(self, name: str, value: object) -> None:
                if (
                    name == "_ever_opened"
                    and value is True
                    and self.inject_interrupt
                ):
                    raise KeyboardInterrupt(
                        "injected between OPEN lifecycle assignments"
                    )
                super().__setattr__(name, value)

        store, lifecycle = _starting_test_lifecycle(
            lifecycle_type=_InterruptEverOpenedAssignment,
        )
        try:
            lifecycle.inject_interrupt = True
            with pytest.raises(
                KeyboardInterrupt,
                match="between OPEN lifecycle assignments",
            ):
                lifecycle.mark_open()
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
        finally:
            store.close()

    @pytest.mark.parametrize("failure_point", ["before", "after", "body"])
    def test_in_memory_open_scope_rolls_back_every_failure_point(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_point: str,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()
        original_mark_open = lifecycle.mark_open

        def interrupted_mark_open() -> None:
            if failure_point == "before":
                raise KeyboardInterrupt("injected before mark_open state")
            original_mark_open()
            if failure_point == "after":
                raise KeyboardInterrupt("injected after mark_open state")

        monkeypatch.setattr(lifecycle, "mark_open", interrupted_mark_open)
        try:
            with lifecycle.startup_lease():
                with pytest.raises(KeyboardInterrupt, match="injected"):
                    with lifecycle.in_memory_open_scope():
                        if failure_point == "body":
                            raise KeyboardInterrupt(
                                "injected in in-memory OPEN scope body"
                            )
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
        finally:
            store.close()

    def test_in_memory_open_scope_publishes_exact_open_state(self) -> None:
        store, lifecycle = _starting_test_lifecycle()
        try:
            with lifecycle.startup_lease():
                with lifecycle.in_memory_open_scope():
                    assert lifecycle.state == "open"
                    assert lifecycle._ever_opened is True
            assert lifecycle.state == "open"
            assert lifecycle._ever_opened is True
        finally:
            store.close()

    def test_in_memory_open_scope_rejects_expired_startup_context(self) -> None:
        store, lifecycle = _starting_test_lifecycle()
        try:
            with lifecycle.startup_lease():
                escaped_context = copy_context()

            def escaped_open() -> None:
                with lifecycle.in_memory_open_scope():
                    pass

            with pytest.raises(
                RuntimeError,
                match="exact active startup lease",
            ):
                escaped_context.run(escaped_open)

            def escaped_admission() -> None:
                with lifecycle.admit():
                    pass

            with pytest.raises(RuntimeError, match="state=starting"):
                escaped_context.run(escaped_admission)
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
        finally:
            store.close()

    @pytest.mark.parametrize("failure_point", ["before", "after"])
    def test_runtime_open_no_backlog_never_leaves_wrapper_failure_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure_point: str,
    ) -> None:
        target = tmp_path / f"no-backlog-open-{failure_point}.sqlite"
        original_mark_open = RuntimeLifecycle.mark_open
        observed: list[RuntimeLifecycle] = []

        def interrupted_mark_open(lifecycle: RuntimeLifecycle) -> None:
            observed.append(lifecycle)
            if failure_point == "before":
                raise KeyboardInterrupt("injected before Runtime OPEN")
            original_mark_open(lifecycle)
            raise KeyboardInterrupt("injected after Runtime OPEN")

        monkeypatch.setattr(
            RuntimeLifecycle,
            "mark_open",
            interrupted_mark_open,
        )
        with pytest.raises(KeyboardInterrupt, match="injected"):
            Runtime.open(target)
        assert len(observed) == 1
        assert observed[0].state != "open"
        assert observed[0]._ever_opened is False

        monkeypatch.setattr(RuntimeLifecycle, "mark_open", original_mark_open)
        reopened = Runtime.open(target)
        try:
            assert reopened.lifecycle.state == "open"
        finally:
            reopened.close()

    def test_startup_open_commit_is_the_durable_open_linearization_point(
        self,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()
        try:
            with lifecycle.startup_lease():
                with lifecycle.open_on_next_commit():
                    with store.transaction() as cursor:
                        cursor.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            ("startup-open", None, "{}", "test", "1", "1"),
                        )
                        assert lifecycle.state == "starting"
            assert lifecycle.state == "open"
            assert store.select_table_rows(
                "object_namespaces",
                "namespace = ?",
                ("startup-open",),
            )
        finally:
            store.close()

    def test_startup_open_commit_failure_rolls_back_open_and_sql(self) -> None:
        store, lifecycle = _starting_test_lifecycle()

        class _FailCommitConnection:
            def __init__(self, connection: object) -> None:
                self._connection = connection

            def __getattr__(self, name: str):
                return getattr(self._connection, name)

            def commit(self) -> None:
                raise RuntimeError("injected startup ack commit failure")

        store.conn = _FailCommitConnection(store.conn)
        try:
            with lifecycle.startup_lease():
                with pytest.raises(
                    RuntimeError,
                    match="injected startup ack commit failure",
                ):
                    with lifecycle.open_on_next_commit():
                        with store.transaction() as cursor:
                            cursor.execute(
                                "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                                ("startup-failed", None, "{}", "test", "1", "1"),
                            )
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
            assert store.select_table_rows(
                "object_namespaces",
                "namespace = ?",
                ("startup-failed",),
            ) == []
        finally:
            store.close()

    def test_startup_open_commit_requires_exactly_one_successful_commit(
        self,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()
        try:
            with lifecycle.startup_lease():
                with pytest.raises(RuntimeError, match="without a Store commit"):
                    with lifecycle.open_on_next_commit():
                        pass
                assert lifecycle.state == "starting"

                with pytest.raises(
                    RuntimeError,
                    match="consumed more than once",
                ):
                    with lifecycle.open_on_next_commit():
                        try:
                            with lifecycle.admission_commit_guard():
                                raise RuntimeError("injected first commit failure")
                        except RuntimeError as exc:
                            assert str(exc) == "injected first commit failure"
                        assert lifecycle.state == "starting"
                        with lifecycle.admission_commit_guard():
                            pass
                assert lifecycle.state == "starting"
                assert lifecycle._ever_opened is False
        finally:
            store.close()

    def test_startup_open_commit_rejects_expired_copied_context(self) -> None:
        store, lifecycle = _starting_test_lifecycle()
        escaped_context = None
        try:
            with lifecycle.startup_lease():
                with pytest.raises(RuntimeError, match="without a Store commit"):
                    with lifecycle.open_on_next_commit():
                        escaped_context = copy_context()
            assert escaped_context is not None

            def delayed_commit() -> None:
                with store.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                        ("escaped-startup", None, "{}", "test", "1", "1"),
                    )

            with pytest.raises(
                RuntimeError,
                match="lost its exact lifecycle scope",
            ):
                escaped_context.run(delayed_commit)
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
            assert store.select_table_rows(
                "object_namespaces",
                "namespace = ?",
                ("escaped-startup",),
            ) == []
        finally:
            store.close()

    def test_startup_open_commit_rejects_copied_context_on_other_thread(
        self,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()
        failures: list[BaseException] = []
        try:
            with lifecycle.startup_lease():
                with pytest.raises(RuntimeError, match="without a Store commit"):
                    with lifecycle.open_on_next_commit():
                        copied_context = copy_context()

                        def escaped_commit() -> None:
                            try:
                                def commit() -> None:
                                    with store.transaction():
                                        pass

                                copied_context.run(commit)
                            except BaseException as exc:
                                failures.append(exc)

                        worker = threading.Thread(target=escaped_commit)
                        worker.start()
                        worker.join(timeout=2)
                        assert not worker.is_alive()
            assert len(failures) == 1
            assert isinstance(failures[0], RuntimeError)
            assert "lost its exact lifecycle scope" in str(failures[0])
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
        finally:
            store.close()

    def test_startup_open_commit_rejects_inherited_async_child_context(
        self,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()

        async def run_attempt() -> BaseException:
            failure: BaseException | None = None
            with lifecycle.startup_lease():
                with pytest.raises(RuntimeError, match="without a Store commit"):
                    with lifecycle.open_on_next_commit():
                        async def escaped_commit() -> None:
                            with store.transaction():
                                pass

                        task = asyncio.create_task(escaped_commit())
                        try:
                            await task
                        except BaseException as exc:
                            failure = exc
            if failure is None:
                raise AssertionError(
                    "inherited startup context unexpectedly committed"
                )
            return failure

        try:
            failure = asyncio.run(run_attempt())
            assert isinstance(failure, RuntimeError)
            assert "lost its exact lifecycle scope" in str(failure)
            assert lifecycle.state == "starting"
            assert lifecycle._ever_opened is False
        finally:
            store.close()

    def test_startup_open_commit_holds_admission_until_commit_returns(
        self,
    ) -> None:
        store, lifecycle = _starting_test_lifecycle()
        commit_entered = threading.Event()
        release_commit = threading.Event()
        admission_completed = threading.Event()
        failures: list[BaseException] = []

        class _BlockingCommitConnection:
            def __init__(self, connection: object) -> None:
                self._connection = connection

            def __getattr__(self, name: str):
                return getattr(self._connection, name)

            def commit(self) -> None:
                commit_entered.set()
                assert release_commit.wait(timeout=2)
                self._connection.commit()

        store.conn = _BlockingCommitConnection(store.conn)

        def publish_open() -> None:
            try:
                with lifecycle.startup_lease():
                    with lifecycle.open_on_next_commit():
                        with store.transaction():
                            pass
            except BaseException as exc:
                failures.append(exc)

        def acquire_admission() -> None:
            try:
                with lifecycle.admit(read_only=True):
                    admission_completed.set()
            except BaseException as exc:
                failures.append(exc)

        publisher = threading.Thread(target=publish_open)
        waiter = threading.Thread(target=acquire_admission)
        try:
            publisher.start()
            assert commit_entered.wait(timeout=2)
            waiter.start()
            assert not admission_completed.wait(timeout=0.05)
            release_commit.set()
            publisher.join(timeout=2)
            waiter.join(timeout=2)
            assert not publisher.is_alive()
            assert not waiter.is_alive()
            assert failures == []
            assert admission_completed.is_set()
            assert lifecycle.state == "open"
        finally:
            release_commit.set()
            publisher.join(timeout=2)
            waiter.join(timeout=2)
            store.close()

    def test_admission_revalidation_remains_eager_non_generator(self) -> None:
        assert not inspect.isgeneratorfunction(
            RuntimeLifecycle.revalidate_current_admission_if_present
        )
        store, lifecycle, _capability = _open_test_lifecycle()
        try:
            with lifecycle.admit():
                lifecycle.mark_recovery_required(publication_id="publication_test")
                with pytest.raises(RuntimeError, match="state=close_failed"):
                    lifecycle.revalidate_current_admission_if_present()
        finally:
            store.close()

    def test_capability_internal_recovery_and_startup_leases_allow_mutation(self) -> None:
        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        audit = AuditManager(store, operations)
        events = EventBus(store, operations)
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=audit,
            events=events,
            substrate=object(),
        )
        lifecycle.begin_recovery()
        capability = CapabilityManager(
            store,
            audit,
            events,
            operations=operations,
            admission=lifecycle,
        )
        try:
            with pytest.raises(RuntimeError, match="state=recovering"):
                capability.issue_trusted(
                    "recovery-owner",
                    "custom:recovery-lease",
                    ["read"],
                    issued_by="test",
                )
            with lifecycle.recovery_lease():
                recovered = capability.issue_trusted(
                    "recovery-owner",
                    "custom:recovery-lease",
                    ["read"],
                    issued_by="test",
                )
            assert store.get_capability(recovered.cap_id) is not None

            lifecycle.begin_starting()
            with pytest.raises(RuntimeError, match="state=starting"):
                capability.issue_trusted(
                    "startup-owner",
                    "custom:startup-lease",
                    ["read"],
                    issued_by="test",
                )
            with lifecycle.startup_lease():
                started = capability.issue_trusted(
                    "startup-owner",
                    "custom:startup-lease",
                    ["read"],
                    issued_by="test",
                )
            assert store.get_capability(started.cap_id) is not None
            lifecycle.mark_open()
        finally:
            store.close()

    def test_capability_public_admission_inventory_is_complete(self) -> None:
        manager_public = {
            name
            for name, method in inspect.getmembers(
                CapabilityManager,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        manager_classes = (
            CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS,
            CAPABILITY_MANAGER_MIXED_PUBLIC_METHODS,
            CAPABILITY_MANAGER_READ_ONLY_PUBLIC_METHODS,
        )
        assert sum(len(methods) for methods in manager_classes) == 48
        assert not any(
            left & right
            for index, left in enumerate(manager_classes)
            for right in manager_classes[index + 1 :]
        )
        assert set().union(*manager_classes) == manager_public

        lease_public = {
            name
            for name, method in inspect.getmembers(
                CapabilityLeaseService,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        mutation_public = {
            name
            for name, method in inspect.getmembers(
                CapabilityMutationService,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        assert lease_public == CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS
        assert mutation_public == CAPABILITY_MUTATION_SERVICE_PUBLIC_METHODS

        runtime = Runtime.open("local")
        try:
            for service, guarded_methods in (
                (
                    runtime.capability,
                    CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS
                    | CAPABILITY_MANAGER_MIXED_PUBLIC_METHODS,
                ),
                (runtime.capability.leases, CAPABILITY_LEASE_MUTATION_PUBLIC_METHODS),
                (
                    runtime.capability.mutations,
                    CAPABILITY_MUTATION_SERVICE_PUBLIC_METHODS,
                ),
            ):
                assert all(
                    getattr(
                        getattr(service, method),
                        "__agent_libos_capability_admission_guarded__",
                        False,
                    )
                    for method in guarded_methods
                )
            assert all(
                not getattr(
                    getattr(runtime.capability, method),
                    "__agent_libos_capability_admission_guarded__",
                    False,
                )
                for method in CAPABILITY_MANAGER_READ_ONLY_PUBLIC_METHODS
            )
        finally:
            runtime.close()

    def test_human_public_mutation_inventory_is_complete(self) -> None:
        public_methods = {
            name
            for name, method in inspect.getmembers(
                HumanObjectManager,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        assert not (HUMAN_PUBLIC_MUTATION_METHODS & HUMAN_READ_ONLY_PUBLIC_METHODS)
        assert (
            HUMAN_PUBLIC_MUTATION_METHODS | HUMAN_READ_ONLY_PUBLIC_METHODS
            == public_methods
        )

    def test_lifecycle_admission_component_public_inventories_are_complete(
        self,
    ) -> None:
        inventories = (
            (
                Runtime,
                (
                    RUNTIME_PUBLIC_MUTATION_METHODS,
                    RUNTIME_READ_ONLY_PUBLIC_METHODS,
                    RUNTIME_LIFECYCLE_PUBLIC_METHODS,
                    RUNTIME_CONTEXT_PUBLIC_METHODS,
                ),
            ),
            (
                AsyncProcessScheduler,
                (
                    SCHEDULER_PUBLIC_MUTATION_METHODS,
                    SCHEDULER_READ_ONLY_PUBLIC_METHODS,
                    SCHEDULER_LIFECYCLE_PUBLIC_METHODS,
                    SCHEDULER_COORDINATION_PUBLIC_METHODS,
                ),
            ),
            (
                DataFlowManager,
                (
                    DATA_FLOW_PUBLIC_MUTATION_METHODS,
                    DATA_FLOW_READ_ONLY_PUBLIC_METHODS,
                    DATA_FLOW_CONTEXT_PUBLIC_METHODS,
                    DATA_FLOW_COMPOSITION_PUBLIC_METHODS,
                ),
            ),
            (
                ObjectTaskManager,
                (
                    OBJECT_TASK_PUBLIC_MUTATION_METHODS,
                    OBJECT_TASK_READ_ONLY_PUBLIC_METHODS,
                    OBJECT_TASK_RECOVERY_PUBLIC_METHODS,
                    OBJECT_TASK_LIFECYCLE_PUBLIC_METHODS,
                ),
            ),
        )
        for component_type, classifications in inventories:
            public_methods = {
                name
                for name, method in inspect.getmembers(
                    component_type,
                    predicate=inspect.isfunction,
                )
                if not name.startswith("_")
            }
            assert not any(
                left & right
                for index, left in enumerate(classifications)
                for right in classifications[index + 1 :]
            )
            assert set().union(*classifications) == public_methods

    def test_public_mutation_admission_inventory_is_fully_installed(self) -> None:
        runtime = Runtime.open("local")
        try:
            assert (
                runtime.mutation_admission_boundary_names
                == PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES
            )
            components = {
                "authority_manifests": runtime.authority_manifests,
                "capability": runtime.capability,
                "checkpoint": runtime.checkpoint,
                "clock": runtime.clock,
                "data_flow": runtime.data_flow,
                "filesystem": runtime.filesystem,
                "git": runtime.git,
                "human": runtime.human,
                "image_registry": runtime.image_registry,
                "image_boot": runtime.image_boot,
                "jsonrpc": runtime.jsonrpc,
                "mcp": runtime.mcp,
                "memory": runtime.memory,
                "messages": runtime.messages,
                "modules": runtime.modules,
                "object_tasks": runtime.object_tasks,
                "process": runtime.process,
                "runtime": runtime,
                "scheduler": runtime.scheduler,
                "shell": runtime.shell,
                "skills": runtime.skills,
                "tools": runtime.tools,
            }
            targets = [
                (descriptor.component, descriptor.method)
                for descriptor in EXPLAIN_BOUNDARY_DESCRIPTORS
            ] + [
                (component, method)
                for component, method, _name in CONTROL_MUTATION_ADMISSION_BOUNDARIES
            ]
            assert all(
                getattr(
                    getattr(components[component], method),
                    "__agent_libos_admission_guarded__",
                    False,
                )
                for component, method in targets
            )
        finally:
            runtime.close()

    def test_nested_runtime_scheduler_boundaries_reuse_one_admission_lease(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        observed_active_leases: list[int] = []

        async def quantum(_pid: str) -> dict[str, bool]:
            observed_active_leases.append(runtime.lifecycle._active_leases)
            return {"ok": True}

        try:
            pid = runtime.process.spawn(goal="nested scheduler admission")
            monkeypatch.setattr(runtime.llm, "arun_once", quantum)

            assert runtime.run_process_once(pid) == {"ok": True}

            assert observed_active_leases == [1]
            assert runtime.lifecycle._active_leases == 0
        finally:
            runtime.close()

    def test_close_failed_scheduler_facades_do_not_claim_process_execution(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(goal="scheduler admission fence")
            before = runtime.store.get_process(pid)
            assert before is not None
            before_identity = (
                before.status,
                before.revision,
                before.state_generation,
                before.execution_generation,
                before.execution_owner_id,
                before.execution_lease_id,
            )
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-scheduler-admission-fence",
                )

            attempts = (
                lambda: runtime.run_process_once(pid),
                lambda: runtime.scheduler.run_pid_once(pid, lambda _pid: None),
                lambda: asyncio.run(runtime.arun_process_once(pid)),
                lambda: asyncio.run(
                    runtime.scheduler.arun_pid_once(pid, lambda _pid: None)
                ),
            )
            for attempt in attempts:
                with pytest.raises(RuntimeError, match="state=close_failed"):
                    attempt()

            after = runtime.store.get_process(pid)
            assert after is not None
            assert (
                after.status,
                after.revision,
                after.state_generation,
                after.execution_generation,
                after.execution_owner_id,
                after.execution_lease_id,
            ) == before_identity
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            runtime.close()

    def test_close_failed_process_view_publishers_use_current_runtime_guard(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(goal="memory view admission fence")
            handle = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {"fenced": True},
            )
            before = runtime.store.get_process(pid)
            assert before is not None
            before_view = copy.deepcopy(before.memory_view)
            before_revision = before.revision
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-process-view-admission-fence",
                )

            publishers = (
                runtime.add_handle_to_process_view,
                runtime.object_tasks._add_handle_to_process_view,
                runtime.modules._hook_services.add_handle_to_process_view,
            )
            for publish in publishers:
                with pytest.raises(RuntimeError, match="state=close_failed"):
                    publish(pid, handle)
            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.memory._object_pin_checker(handle.oid)
            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.memory._object_change_notifier(
                    handle.oid,
                    {"event": "updated"},
                    pid,
                )

            after = runtime.store.get_process(pid)
            assert after is not None
            assert after.memory_view == before_view
            assert after.revision == before_revision
        finally:
            runtime.close()

    def test_close_failed_sink_registry_mutations_publish_no_evidence(self) -> None:
        runtime = Runtime.open("local")
        rule = SinkTrustRule(
            pattern="filesystem:workspace:lifecycle-admission.txt",
            trust_level=SinkTrustLevel.TRUSTED,
        )
        try:
            before_rules = runtime.data_flow.list_sink_trust(active_only=False)
            before_generation = runtime.store.get_sink_trust_generation()
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-sink-registry-admission-fence",
                )

            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.data_flow.register_sink_trust(
                    rule,
                    actor="test.host",
                    require_capability=False,
                )
            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.register_sink_trust(rule, actor="test.host")

            assert runtime.data_flow.list_sink_trust(active_only=False) == before_rules
            assert runtime.store.get_sink_trust_generation() == before_generation
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            runtime.close()

    def test_registry_mutation_components_share_guarded_lifecycle_barrier(self) -> None:
        runtime = Runtime.open("local")
        try:
            barrier = runtime._registry_lifecycle_lock
            assert runtime.tools._registry_lifecycle_lock_value is barrier
            assert runtime.tools.execution._registry_lifecycle_lock is barrier
            assert runtime.checkpoint._registry_lifecycle_lock is barrier
            assert runtime.skills._lifecycle_lock is barrier
            assert runtime.modules._lifecycle_lock is barrier
            assert runtime.image_registry.lifecycle_lock is barrier
            assert runtime.image_boot._publication_lock is barrier
        finally:
            runtime.close()

    def test_store_outer_commit_guard_is_bound_to_runtime_lifecycle(self) -> None:
        runtime = Runtime.open("local")
        try:
            guard = runtime.store._admission_commit_guard
            assert guard is not None
            assert getattr(guard, "__self__", None) is runtime.lifecycle
            assert getattr(guard, "__func__", None) is RuntimeLifecycle.admission_commit_guard
        finally:
            runtime.close()

    def test_admission_epoch_snapshot_is_atomic_with_active_lease_registration(
        self,
    ) -> None:
        epoch_read_entered = threading.Event()
        release_epoch_read = threading.Event()

        class InterleavedRuntimeLifecycle(RuntimeLifecycle):
            def __init__(self, **kwargs: object) -> None:
                self._controlled_recovery_fence_epoch = 0
                self.blocked_epoch_reader: int | None = None
                super().__init__(**kwargs)

            @property
            def _recovery_fence_epoch(self) -> int:
                if self.blocked_epoch_reader == threading.get_ident():
                    epoch_read_entered.set()
                    if not release_epoch_read.wait(timeout=2):
                        raise AssertionError("timed out waiting to release epoch read")
                    self.blocked_epoch_reader = None
                return self._controlled_recovery_fence_epoch

            @_recovery_fence_epoch.setter
            def _recovery_fence_epoch(self, value: int) -> None:
                self._controlled_recovery_fence_epoch = value

        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        lifecycle = InterleavedRuntimeLifecycle(
            store=store,
            audit=AuditManager(store, operations),
            events=EventBus(store, operations),
            substrate=object(),
        )
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()

        fencer_ready = threading.Event()
        trigger_fence = threading.Event()
        fence_attempted = threading.Event()
        fence_completed = threading.Event()
        writer_admitted = threading.Event()
        thread_errors: list[BaseException] = []

        def fence() -> None:
            try:
                with lifecycle.admit():
                    fencer_ready.set()
                    if not trigger_fence.wait(timeout=2):
                        raise AssertionError("timed out waiting to trigger recovery fence")
                    fence_attempted.set()
                    lifecycle.mark_recovery_required(
                        publication_id="publication-admission-epoch-race",
                    )
            except BaseException as error:
                thread_errors.append(error)
            finally:
                fence_completed.set()

        def admit_writer() -> None:
            try:
                lifecycle.blocked_epoch_reader = threading.get_ident()
                with lifecycle.admit():
                    writer_admitted.set()
            except BaseException as error:
                thread_errors.append(error)

        fence_thread = threading.Thread(target=fence)
        writer_thread = threading.Thread(target=admit_writer)
        fence_thread.start()
        try:
            assert fencer_ready.wait(timeout=1)
            writer_thread.start()
            assert epoch_read_entered.wait(timeout=1)

            trigger_fence.set()
            assert fence_attempted.wait(timeout=1)
            assert not fence_completed.wait(timeout=0.1)
        finally:
            trigger_fence.set()
            release_epoch_read.set()
            fence_thread.join(timeout=2)
            if writer_thread.ident is not None:
                writer_thread.join(timeout=2)
            store.close()

        assert not fence_thread.is_alive()
        assert not writer_thread.is_alive()
        assert thread_errors == []
        assert writer_admitted.is_set()
        assert lifecycle.state == "close_failed"

    def test_recovery_fence_rolls_back_staged_business_audit_and_event_writes(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        tables = ("object_namespaces", "audit_records", "events")
        before = {
            table: copy.deepcopy(runtime.store.select_table_rows(table))
            for table in tables
        }
        fencer_ready = threading.Event()
        writes_staged = threading.Event()
        fence_completed = threading.Event()
        writer_errors: list[BaseException] = []
        fencer_errors: list[BaseException] = []

        def fence() -> None:
            try:
                with runtime.lifecycle.admit():
                    fencer_ready.set()
                    if not writes_staged.wait(timeout=2):
                        raise AssertionError("timed out waiting for staged writes")
                    runtime.lifecycle.mark_recovery_required(
                        publication_id="publication-staged-write-fence",
                    )
            except BaseException as error:
                fencer_errors.append(error)
            finally:
                fence_completed.set()

        def write() -> None:
            try:
                with runtime.lifecycle.admit():
                    with runtime.store.transaction() as cur:
                        cur.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                "recovery-fenced-namespace",
                                None,
                                "{}",
                                "test",
                                "1",
                                "1",
                            ),
                        )
                        runtime.audit.record(
                            actor="test",
                            action="test.recovery_fenced_write",
                            target="runtime",
                        )
                        runtime.events.emit(
                            EventType.OBJECT_UPDATED,
                            source="test",
                            target="runtime",
                            payload={"recovery_fenced": True},
                        )
                        writes_staged.set()
                        if not fence_completed.wait(timeout=2):
                            raise AssertionError(
                                "timed out waiting for recovery fence"
                            )
            except BaseException as error:
                writer_errors.append(error)

        fence_thread = threading.Thread(target=fence)
        writer_thread = threading.Thread(target=write)
        fence_thread.start()
        try:
            assert fencer_ready.wait(timeout=1)
            writer_thread.start()
            writer_thread.join(timeout=3)
            fence_thread.join(timeout=3)

            assert not writer_thread.is_alive()
            assert not fence_thread.is_alive()
            assert fencer_errors == []
            assert len(writer_errors) == 1
            assert isinstance(writer_errors[0], RuntimeError)
            assert "state=close_failed" in str(writer_errors[0])
            assert {
                table: runtime.store.select_table_rows(table)
                for table in tables
            } == before
        finally:
            writes_staged.set()
            fence_completed.set()
            if writer_thread.ident is not None and writer_thread.is_alive():
                writer_thread.join(timeout=2)
            if fence_thread.is_alive():
                fence_thread.join(timeout=2)
            runtime.close()

    def test_sync_shutdown_preserves_recovery_fence_after_admission_drain(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=AuditManager(store, operations),
            events=EventBus(store, operations),
            substrate=object(),
        )
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()

        admission_entered = threading.Event()
        drain_entered = threading.Event()
        operation_errors: list[BaseException] = []
        finalizer_calls: list[str] = []
        publication_id = "publication-sync-shutdown-drain-fence"
        recovery_reason = f"runtime.recovery_required:{publication_id}"
        before_audit = store.list_audit()
        before_events = store.list_events()

        def finalizer() -> None:
            finalizer_calls.append("finalizer")

        def admitted_operation() -> None:
            try:
                with lifecycle.admit():
                    admission_entered.set()
                    if not drain_entered.wait(timeout=2):
                        raise AssertionError("timed out waiting for shutdown drain")
                    lifecycle.mark_recovery_required(
                        publication_id=publication_id,
                    )
            except BaseException as error:
                operation_errors.append(error)

        original_drain = lifecycle._drain_admission

        def observed_drain() -> bool:
            drain_entered.set()
            return original_drain()

        lifecycle.bind_finalizer(finalizer)
        monkeypatch.setattr(lifecycle, "_drain_admission", observed_drain)
        operation_thread = threading.Thread(target=admitted_operation)
        operation_thread.start()
        try:
            assert admission_entered.wait(timeout=1)
            result = lifecycle.shutdown(
                actor="test",
                reason="concurrent-sync-shutdown",
            )
            operation_thread.join(timeout=2)
            repeated = lifecycle.shutdown(
                actor="test",
                reason="must-not-overwrite-sync-recovery",
            )

            assert not operation_thread.is_alive()
            assert operation_errors == []
            assert result == {
                "ok": False,
                "already_shutdown": False,
                "reason": recovery_reason,
                "recovery_required": True,
            }
            assert repeated == result
            assert lifecycle.state == "close_failed"
            assert lifecycle.shutdown_reason == recovery_reason
            assert finalizer_calls == []
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
        finally:
            drain_entered.set()
            if operation_thread.is_alive():
                operation_thread.join(timeout=2)
            store.close()

    def test_async_shutdown_preserves_recovery_fence_after_admission_drain(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=AuditManager(store, operations),
            events=EventBus(store, operations),
            substrate=object(),
        )
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()

        admission_entered = threading.Event()
        drain_entered = threading.Event()
        operation_errors: list[BaseException] = []
        finalizer_calls: list[str] = []
        publication_id = "publication-async-shutdown-drain-fence"
        recovery_reason = f"runtime.recovery_required:{publication_id}"
        before_audit = store.list_audit()
        before_events = store.list_events()

        async def finalizer() -> None:
            finalizer_calls.append("finalizer")

        def admitted_operation() -> None:
            try:
                with lifecycle.admit():
                    admission_entered.set()
                    if not drain_entered.wait(timeout=2):
                        raise AssertionError("timed out waiting for shutdown drain")
                    lifecycle.mark_recovery_required(
                        publication_id=publication_id,
                    )
            except BaseException as error:
                operation_errors.append(error)

        original_drain = lifecycle._drain_admission

        def observed_drain() -> bool:
            drain_entered.set()
            return original_drain()

        lifecycle.bind_finalizer(finalizer)
        monkeypatch.setattr(lifecycle, "_drain_admission", observed_drain)
        operation_thread = threading.Thread(target=admitted_operation)
        operation_thread.start()
        try:
            assert admission_entered.wait(timeout=1)
            result = asyncio.run(
                lifecycle.ashutdown(
                    actor="test",
                    reason="concurrent-async-shutdown",
                )
            )
            operation_thread.join(timeout=2)
            repeated = asyncio.run(
                lifecycle.ashutdown(
                    actor="test",
                    reason="must-not-overwrite-async-recovery",
                )
            )

            assert not operation_thread.is_alive()
            assert operation_errors == []
            assert result == {
                "ok": False,
                "already_shutdown": False,
                "reason": recovery_reason,
                "recovery_required": True,
            }
            assert repeated == result
            assert lifecycle.state == "close_failed"
            assert lifecycle.shutdown_reason == recovery_reason
            assert finalizer_calls == []
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
        finally:
            drain_entered.set()
            if operation_thread.is_alive():
                operation_thread.join(timeout=2)
            store.close()

    def test_sync_admission_timeout_never_tears_down_after_recovery_fence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        component_calls: list[str] = []

        class TrackingComponent:
            def shutdown(self) -> bool:
                component_calls.append("sync")
                return True

        component = TrackingComponent()
        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=AuditManager(store, operations),
            events=EventBus(store, operations),
            substrate=component,
            admission_drain_timeout_s=0.05,
        )
        lifecycle.bind_components(
            scheduler=component,
            object_tasks=component,
            modules=component,
            llms=component,
            blocking_work=component,
        )
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()

        admission_entered = threading.Event()
        drain_entered = threading.Event()
        fence_marked = threading.Event()
        release_admission = threading.Event()
        operation_errors: list[BaseException] = []
        drain_results: list[bool] = []
        finalizer_calls: list[str] = []
        publication_id = "publication-sync-timeout-fence"
        recovery_reason = f"runtime.recovery_required:{publication_id}"
        before_audit = store.list_audit()
        before_events = store.list_events()

        def finalizer() -> None:
            finalizer_calls.append("finalizer")

        def admitted_operation() -> None:
            try:
                with lifecycle.admit():
                    admission_entered.set()
                    if not drain_entered.wait(timeout=2):
                        raise AssertionError("timed out waiting for shutdown drain")
                    lifecycle.mark_recovery_required(
                        publication_id=publication_id,
                    )
                    fence_marked.set()
                    if not release_admission.wait(timeout=2):
                        raise AssertionError("timed out waiting to release admission")
            except BaseException as error:
                operation_errors.append(error)

        original_drain = lifecycle._drain_admission

        def observed_drain() -> bool:
            drain_entered.set()
            drained = original_drain()
            drain_results.append(drained)
            return drained

        lifecycle.bind_finalizer(finalizer)
        monkeypatch.setattr(lifecycle, "_drain_admission", observed_drain)
        operation_thread = threading.Thread(target=admitted_operation)
        operation_thread.start()
        try:
            assert admission_entered.wait(timeout=1)
            result = lifecycle.shutdown(
                actor="test",
                reason="sync-timeout-must-not-teardown",
            )
            assert fence_marked.is_set()
            assert operation_thread.is_alive()
            repeated = lifecycle.shutdown(
                actor="test",
                reason="sync-timeout-must-not-overwrite",
            )

            assert drain_results == [False]
            assert result == {
                "ok": False,
                "already_shutdown": False,
                "reason": recovery_reason,
                "recovery_required": True,
            }
            assert repeated == result
            assert lifecycle.state == "close_failed"
            assert lifecycle.shutdown_reason == recovery_reason
            assert component_calls == []
            assert finalizer_calls == []
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
        finally:
            drain_entered.set()
            release_admission.set()
            operation_thread.join(timeout=2)
            store.close()

        assert not operation_thread.is_alive()
        assert operation_errors == []

    def test_async_admission_timeout_linearizes_before_late_recovery_fence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        component_calls: list[str] = []

        class TrackingComponent:
            async def ashutdown(self) -> bool:
                component_calls.append("async")
                return True

        component = TrackingComponent()
        store = SQLiteStore(":memory:")
        operations = OperationManager(store)
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=AuditManager(store, operations),
            events=EventBus(store, operations),
            substrate=component,
            admission_drain_timeout_s=0.05,
        )
        lifecycle.bind_components(
            scheduler=component,
            object_tasks=component,
            modules=component,
            llms=component,
            blocking_work=component,
        )
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()

        admission_entered = threading.Event()
        timeout_linearized = threading.Event()
        fence_marked = threading.Event()
        release_admission = threading.Event()
        operation_errors: list[BaseException] = []
        finalizer_calls: list[str] = []
        publication_id = "publication-async-post-timeout-fence"
        recovery_reason = f"runtime.recovery_required:{publication_id}"
        before_audit = store.list_audit()
        before_events = store.list_events()

        async def finalizer() -> None:
            finalizer_calls.append("finalizer")

        def admitted_operation() -> None:
            try:
                with lifecycle.admit():
                    admission_entered.set()
                    if not timeout_linearized.wait(timeout=2):
                        raise AssertionError(
                            "timed out waiting for timeout linearization"
                        )
                    lifecycle.mark_recovery_required(
                        publication_id=publication_id,
                    )
                    fence_marked.set()
                    if not release_admission.wait(timeout=2):
                        raise AssertionError("timed out waiting to release admission")
            except BaseException as error:
                operation_errors.append(error)

        original_timeout_result = lifecycle._admission_timeout_result

        def observed_timeout_result(
            reason: str,
            errors: list[dict[str, str]],
        ) -> dict[str, object]:
            result = original_timeout_result(reason, errors)
            timeout_linearized.set()
            if not fence_marked.wait(timeout=2):
                raise AssertionError("late recovery fence was not marked")
            return result

        lifecycle.bind_finalizer(finalizer)
        monkeypatch.setattr(
            lifecycle,
            "_admission_timeout_result",
            observed_timeout_result,
        )
        operation_thread = threading.Thread(target=admitted_operation)
        operation_thread.start()
        try:
            assert admission_entered.wait(timeout=1)
            result = asyncio.run(
                lifecycle.ashutdown(
                    actor="test",
                    reason="async-timeout-before-recovery-fence",
                )
            )
            assert operation_thread.is_alive()
            repeated = asyncio.run(
                lifecycle.ashutdown(
                    actor="test",
                    reason="async-timeout-must-not-overwrite",
                )
            )

            assert result == {
                "ok": False,
                "already_shutdown": False,
                "reason": "async-timeout-before-recovery-fence",
                "admission_stopped": False,
            }
            assert repeated == {
                "ok": False,
                "already_shutdown": False,
                "reason": recovery_reason,
                "recovery_required": True,
            }
            assert lifecycle.state == "close_failed"
            assert lifecycle.shutdown_reason == recovery_reason
            assert component_calls == []
            assert finalizer_calls == []
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
        finally:
            timeout_linearized.set()
            release_admission.set()
            operation_thread.join(timeout=2)
            store.close()

        assert not operation_thread.is_alive()
        assert operation_errors == []

    def test_recovery_diagnostics_release_is_explicit_idempotent_and_reopenable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "recovery-diagnostics.db"
            runtime = Runtime.open(database)
            finalizer_calls: list[str] = []
            close_snapshots: list[tuple[list[object], list[object]]] = []
            reopened: Runtime | None = None
            released = False
            publication_id = "publication-explicit-recovery-release"
            recovery_reason = f"runtime.recovery_required:{publication_id}"

            def finalizer() -> None:
                finalizer_calls.append("finalizer")

            runtime.bind_shutdown_finalizer(finalizer)
            pid = runtime.process.spawn(goal="same-process recovery handoff")
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id=publication_id,
                )
            before_audit = list(runtime.store.list_audit())
            before_events = list(runtime.store.list_events())
            recovery_result = {
                "ok": False,
                "already_shutdown": False,
                "reason": recovery_reason,
                "recovery_required": True,
            }

            original_close = runtime.store.close

            def observed_close() -> None:
                close_snapshots.append(
                    (
                        list(runtime.store.list_audit()),
                        list(runtime.store.list_events()),
                    )
                )
                original_close()

            monkeypatch.setattr(runtime.store, "close", observed_close)
            try:
                assert runtime.close() == recovery_result
                assert runtime.shutdown(
                    actor="test",
                    reason="must-not-overwrite-recovery",
                ) == recovery_result

                outcome = runtime.release_recovery_diagnostics()
                released = outcome["ok"] is True
                assert outcome == {
                    "ok": True,
                    "already_released": False,
                    "reason": recovery_reason,
                    "recovery_required": True,
                    "recovery_diagnostics_released": True,
                }
                assert runtime.release_recovery_diagnostics() == {
                    **outcome,
                    "already_released": True,
                }
                assert runtime.lifecycle.closed
                assert runtime.lifecycle.shutdown_reason == recovery_reason
                assert finalizer_calls == []
                assert close_snapshots == [(before_audit, before_events)]

                reopened = Runtime.open(database)
                assert reopened.lifecycle.state == "open"
                assert reopened.store.get_process(pid) is not None
            finally:
                if not released:
                    runtime.store.close()
                if reopened is not None:
                    reopened.close()

    def test_async_recovery_diagnostics_release_allows_same_process_reopen(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "async-recovery-diagnostics.db"
            runtime = Runtime.open(database)
            reopened: Runtime | None = None
            released = False
            publication_id = "publication-async-recovery-release"
            recovery_reason = f"runtime.recovery_required:{publication_id}"
            finalizer_calls: list[str] = []
            close_snapshots: list[tuple[list[object], list[object]]] = []

            async def finalizer() -> None:
                finalizer_calls.append("finalizer")

            runtime.bind_shutdown_finalizer(finalizer)
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id=publication_id,
                )
            before_audit = list(runtime.store.list_audit())
            before_events = list(runtime.store.list_events())
            original_close = runtime.store.close

            def observed_close() -> None:
                close_snapshots.append(
                    (
                        list(runtime.store.list_audit()),
                        list(runtime.store.list_events()),
                    )
                )
                original_close()

            monkeypatch.setattr(runtime.store, "close", observed_close)
            try:
                outcome = asyncio.run(runtime.arelease_recovery_diagnostics())
                released = outcome["ok"] is True
                assert outcome == {
                    "ok": True,
                    "already_released": False,
                    "reason": recovery_reason,
                    "recovery_required": True,
                    "recovery_diagnostics_released": True,
                }
                assert finalizer_calls == []
                assert close_snapshots == [(before_audit, before_events)]

                reopened = Runtime.open(database)
                assert reopened.lifecycle.state == "open"
            finally:
                if not released:
                    runtime.store.close()
                if reopened is not None:
                    reopened.close()

    def test_concurrent_sync_recovery_release_reads_back_after_stale_probe(
        self,
    ) -> None:
        store = _FirstRecoveryCloseProbeBarrierStore()
        store, lifecycle, capability = _open_test_lifecycle(store=store)
        _mark_test_recovery_fence(
            lifecycle,
            "publication-concurrent-sync-release",
        )
        delayed_results: list[dict[str, object]] = []
        delayed_errors: list[BaseException] = []

        def delayed_release() -> None:
            try:
                delayed_results.append(
                    lifecycle.release_recovery_diagnostics(
                        capability=capability,
                    )
                )
            except BaseException as exc:
                delayed_errors.append(exc)

        delayed_thread = threading.Thread(target=delayed_release)
        delayed_thread.start()
        assert store.first_probe_entered.wait(timeout=1)
        try:
            winner = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert winner["ok"] is True
            assert winner["already_released"] is False

            store.release_first_probe.set()
            delayed_thread.join(timeout=2)
            assert not delayed_thread.is_alive()
            assert delayed_errors == []
            assert delayed_results == [
                {**winner, "already_released": True},
            ]
            assert lifecycle.closed
        finally:
            store.release_first_probe.set()
            delayed_thread.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    def test_concurrent_async_recovery_release_reads_back_after_stale_probe(
        self,
    ) -> None:
        store = _FirstRecoveryCloseProbeBarrierStore()
        store, lifecycle, capability = _open_test_lifecycle(store=store)
        _mark_test_recovery_fence(
            lifecycle,
            "publication-concurrent-async-release",
        )
        delayed_results: list[dict[str, object]] = []
        delayed_errors: list[BaseException] = []

        def delayed_release() -> None:
            try:
                delayed_results.append(
                    asyncio.run(
                        lifecycle.arelease_recovery_diagnostics(
                            capability=capability,
                        )
                    )
                )
            except BaseException as exc:
                delayed_errors.append(exc)

        delayed_thread = threading.Thread(target=delayed_release)
        delayed_thread.start()
        assert store.first_probe_entered.wait(timeout=1)
        try:
            winner = asyncio.run(
                lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
            )
            assert winner["ok"] is True
            assert winner["already_released"] is False

            store.release_first_probe.set()
            delayed_thread.join(timeout=2)
            assert not delayed_thread.is_alive()
            assert delayed_errors == []
            assert delayed_results == [
                {**winner, "already_released": True},
            ]
            assert lifecycle.closed
        finally:
            store.release_first_probe.set()
            delayed_thread.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    def test_recovery_release_keyboard_interrupt_resets_and_retries(self) -> None:
        class InterruptingScheduler:
            def __init__(self) -> None:
                self.calls = 0

            def shutdown(self) -> bool:
                self.calls += 1
                if self.calls == 1:
                    raise KeyboardInterrupt("injected recovery cleanup interrupt")
                return True

        scheduler = InterruptingScheduler()
        store, lifecycle, capability = _open_test_lifecycle(
            scheduler=scheduler,
        )
        ordinary_finalizers: list[str] = []
        lifecycle.bind_finalizer(lambda: ordinary_finalizers.append("ordinary"))
        recovery_reason = _mark_test_recovery_fence(
            lifecycle,
            "publication-recovery-interrupt-retry",
        )
        before_audit = list(store.list_audit())
        before_events = list(store.list_events())
        try:
            with pytest.raises(KeyboardInterrupt, match="cleanup interrupt"):
                lifecycle.release_recovery_diagnostics(capability=capability)

            assert lifecycle.state == "close_failed"
            assert lifecycle.shutdown_reason == recovery_reason
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
            assert ordinary_finalizers == []

            outcome = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert outcome["ok"] is True
            assert lifecycle.closed
            assert scheduler.calls == 2
            assert ordinary_finalizers == []
        finally:
            if not lifecycle.closed:
                store.close()

    @pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
    def test_recovery_safe_callback_failure_or_interrupt_is_retryable(
        self,
        failure_type: type[BaseException],
    ) -> None:
        store, lifecycle, capability = _open_test_lifecycle()
        callback_calls = 0

        def cleanup() -> bool:
            nonlocal callback_calls
            callback_calls += 1
            if callback_calls == 1:
                raise failure_type("injected recovery callback failure")
            return True

        lifecycle.bind_finalizer(cleanup, recovery_safe=True)
        _mark_test_recovery_fence(
            lifecycle,
            f"publication-callback-{failure_type.__name__}",
        )
        try:
            if issubclass(failure_type, Exception):
                first = lifecycle.release_recovery_diagnostics(
                    capability=capability,
                )
                assert first["ok"] is False
                assert first["errors"][0]["error_type"] == failure_type.__name__
            else:
                with pytest.raises(failure_type):
                    lifecycle.release_recovery_diagnostics(
                        capability=capability,
                    )

            assert lifecycle.state == "close_failed"
            second = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert second["ok"] is True
            assert lifecycle.closed
            assert callback_calls == 2
        finally:
            if not lifecycle.closed:
                store.close()

    def test_failed_recovery_cleanup_permanently_blocks_unadmitted_writes(
        self,
    ) -> None:
        store, lifecycle, capability = _open_test_lifecycle()
        callback_calls = 0
        callback_write_errors: list[BaseException] = []
        late_write_errors: list[BaseException] = []
        ordinary_finalizers: list[str] = []
        release_late_write = threading.Event()

        def cleanup() -> bool:
            nonlocal callback_calls
            callback_calls += 1
            lifecycle.require_recovery_cleanup_lease()
            if callback_calls != 1:
                return True
            try:
                lifecycle._audit.record(
                    actor="recovery-cleanup",
                    action="forbidden.recovery.write",
                    target="runtime",
                )
            except BaseException as exc:
                callback_write_errors.append(exc)
            return False

        def delayed_background_write() -> None:
            assert release_late_write.wait(timeout=2)
            try:
                lifecycle._audit.record(
                    actor="late-worker",
                    action="forbidden.late.write",
                    target="runtime",
                )
            except BaseException as exc:
                late_write_errors.append(exc)

        lifecycle.bind_finalizer(cleanup, recovery_safe=True)
        lifecycle.bind_finalizer(lambda: ordinary_finalizers.append("ordinary"))
        _mark_test_recovery_fence(
            lifecycle,
            "publication-no-write-recovery-cleanup",
        )
        before_audit = list(store.list_audit())
        before_events = list(store.list_events())
        worker = threading.Thread(target=delayed_background_write)
        worker.start()
        try:
            with pytest.raises(
                RuntimeError,
                match="active recovery cleanup lease",
            ):
                lifecycle.require_recovery_cleanup_lease()
            first = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert first["ok"] is False
            assert callback_calls == 1
            assert len(callback_write_errors) == 1
            assert "recovery is required" in str(callback_write_errors[0])

            release_late_write.set()
            worker.join(timeout=2)
            assert not worker.is_alive()
            assert len(late_write_errors) == 1
            assert "recovery is required" in str(late_write_errors[0])
            assert store.list_audit() == before_audit
            assert store.list_events() == before_events
            assert ordinary_finalizers == []

            second = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert second["ok"] is True
            assert callback_calls == 2
            assert ordinary_finalizers == []
        finally:
            release_late_write.set()
            worker.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    @pytest.mark.parametrize(
        ("first_result", "expected_calls"),
        [(True, 1), (False, 2)],
    )
    def test_async_recovery_release_preserves_cancelled_cleanup_outcome(
        self,
        first_result: bool,
        expected_calls: int,
    ) -> None:
        async def scenario() -> None:
            cleanup_entered = threading.Event()
            release_cleanup = threading.Event()
            cleanup_threads: list[int] = []
            substrate_loops: list[asyncio.AbstractEventLoop] = []
            ordinary_finalizers: list[str] = []

            class AsyncOnlySubstrate:
                async def aclose(self) -> None:
                    substrate_loops.append(asyncio.get_running_loop())

            def recovery_cleanup() -> bool:
                lifecycle.require_recovery_cleanup_lease()
                cleanup_threads.append(threading.get_ident())
                cleanup_entered.set()
                assert release_cleanup.wait(timeout=2)
                return first_result if len(cleanup_threads) == 1 else True

            store, lifecycle, capability = _open_test_lifecycle(
                substrate=AsyncOnlySubstrate(),
            )
            lifecycle.bind_finalizer(recovery_cleanup, recovery_safe=True)
            lifecycle.bind_finalizer(
                lambda: ordinary_finalizers.append("ordinary")
            )
            _mark_test_recovery_fence(
                lifecycle,
                "publication-async-cleanup-cancel",
            )
            caller_loop = asyncio.get_running_loop()
            caller_thread = threading.get_ident()
            task = asyncio.create_task(
                lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
            )
            try:
                assert await asyncio.to_thread(cleanup_entered.wait, 2.0)
                assert cleanup_threads == [cleanup_threads[0]]
                assert cleanup_threads[0] != caller_thread

                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                release_cleanup.set()
                with pytest.raises(asyncio.CancelledError):
                    await task

                assert lifecycle.state == "close_failed"
                assert substrate_loops == []
                assert ordinary_finalizers == []
                assert store.list_audit() == []

                outcome = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert outcome["ok"] is True
                assert len(cleanup_threads) == expected_calls
                assert all(thread != caller_thread for thread in cleanup_threads)
                assert substrate_loops == [caller_loop]
                assert ordinary_finalizers == []
            finally:
                release_cleanup.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_cancelled_sync_cleanup_preserves_returned_awaitable_failure(
        self,
    ) -> None:
        async def scenario() -> None:
            cleanup_entered = threading.Event()
            release_cleanup = threading.Event()
            callback_calls = 0
            store, lifecycle, capability = _open_test_lifecycle()

            def cleanup():
                nonlocal callback_calls
                lifecycle.require_recovery_cleanup_lease()
                callback_calls += 1
                if callback_calls != 1:
                    return True
                cleanup_entered.set()
                assert release_cleanup.wait(timeout=2)

                async def fail_after_worker_return() -> None:
                    raise RuntimeError("returned awaitable failed")

                return fail_after_worker_return()

            lifecycle.bind_finalizer(cleanup, recovery_safe=True)
            _mark_test_recovery_fence(
                lifecycle,
                "publication-cancelled-returned-awaitable",
            )
            task = asyncio.create_task(
                lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
            )
            try:
                assert await asyncio.to_thread(cleanup_entered.wait, 2.0)
                task.cancel()
                release_cleanup.set()
                with pytest.raises(BaseExceptionGroup) as caught:
                    await task
                assert any(
                    isinstance(error, asyncio.CancelledError)
                    for error in caught.value.exceptions
                )
                assert any(
                    isinstance(error, RuntimeError)
                    and "returned awaitable failed" in str(error)
                    for error in caught.value.exceptions
                )
                assert lifecycle.state == "close_failed"

                retry = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert retry["ok"] is True
                assert callback_calls == 2
            finally:
                release_cleanup.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_cancelled_async_object_task_release_stays_single_flight_until_drained(
        self,
    ) -> None:
        async def scenario() -> None:
            entered = threading.Event()
            allow_return = threading.Event()
            state_lock = threading.Lock()
            calls = 0
            active = 0
            max_active = 0

            class BlockingObjectTaskHandle:
                def release_recovery_diagnostics(self) -> bool:
                    nonlocal calls, active, max_active
                    with state_lock:
                        calls += 1
                        active += 1
                        max_active = max(max_active, active)
                    entered.set()
                    try:
                        assert allow_return.wait(timeout=2)
                        return True
                    finally:
                        with state_lock:
                            active -= 1

            store, lifecycle, capability = _open_test_lifecycle(
                object_tasks=BlockingObjectTaskHandle(),
            )
            _mark_test_recovery_fence(
                lifecycle,
                "publication-object-task-release-cancel",
            )
            task = asyncio.create_task(
                lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2.0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                with pytest.raises(RuntimeError, match="already in progress"):
                    await lifecycle.arelease_recovery_diagnostics(
                        capability=capability,
                    )

                allow_return.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert lifecycle.state == "close_failed"

                outcome = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert outcome["ok"] is True
                assert calls == 2
                assert max_active == 1
            finally:
                allow_return.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_sync_recovery_release_drives_async_only_component_without_loop(
        self,
    ) -> None:
        close_loops: list[asyncio.AbstractEventLoop] = []

        class AsyncOnlySubstrate:
            async def aclose(self) -> None:
                close_loops.append(asyncio.get_running_loop())

        store, lifecycle, capability = _open_test_lifecycle(
            substrate=AsyncOnlySubstrate(),
        )
        _mark_test_recovery_fence(
            lifecycle,
            "publication-sync-async-only-substrate",
        )
        try:
            outcome = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert outcome["ok"] is True
            assert lifecycle.closed
            assert len(close_loops) == 1
        finally:
            if not lifecycle.closed:
                store.close()

    def test_sync_recovery_release_in_loop_has_no_teardown_or_latch_side_effect(
        self,
    ) -> None:
        async def scenario() -> None:
            component_calls: list[str] = []

            class AsyncOnlySubstrate:
                async def aclose(self) -> None:
                    component_calls.append("aclose")

            store, lifecycle, capability = _open_test_lifecycle(
                substrate=AsyncOnlySubstrate(),
            )
            _mark_test_recovery_fence(
                lifecycle,
                "publication-sync-release-inside-loop",
            )
            try:
                with pytest.raises(
                    RuntimeError,
                    match="active event loop",
                ):
                    lifecycle.release_recovery_diagnostics(
                        capability=capability,
                    )
                assert component_calls == []
                assert lifecycle.state == "close_failed"
                assert lifecycle._recovery_diagnostics_release_started is False

                outcome = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert outcome["ok"] is True
                assert component_calls == ["aclose"]
            finally:
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_sync_shutdown_rejects_async_only_component_inside_running_loop(
        self,
    ) -> None:
        async def scenario() -> None:
            component_calls: list[str] = []

            class AsyncOnlySubstrate:
                async def aclose(self) -> None:
                    component_calls.append("aclose")

            store, lifecycle, _capability = _open_test_lifecycle(
                substrate=AsyncOnlySubstrate(),
            )
            try:
                with pytest.raises(
                    RuntimeError,
                    match="async-only shutdown work",
                ):
                    lifecycle.shutdown(reason="sync-inside-loop")
                assert lifecycle.state == "open"
                assert component_calls == []

                outcome = await lifecycle.ashutdown(
                    reason="async-after-sync-preflight",
                )
                assert outcome["ok"] is True
                assert component_calls == ["aclose"]
            finally:
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_sync_shutdown_in_active_loop_reads_back_closed_async_runtime(
        self,
    ) -> None:
        async def scenario() -> None:
            class AsyncOnlySubstrate:
                async def aclose(self) -> None:
                    return None

            store, lifecycle, _capability = _open_test_lifecycle(
                substrate=AsyncOnlySubstrate(),
            )
            try:
                first = await lifecycle.ashutdown(
                    reason="async-close-before-sync-readback",
                )
                assert first["ok"] is True

                readback = lifecycle.shutdown(
                    reason="sync-readback-inside-active-loop",
                )
                assert readback == {
                    **first,
                    "already_shutdown": True,
                }
            finally:
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_sync_shutdown_rejects_caller_transaction_before_side_effects(
        self,
    ) -> None:
        component_calls: list[str] = []

        class TrackingComponent:
            def shutdown(self) -> bool:
                component_calls.append("shutdown")
                return True

        store, lifecycle, _capability = _open_test_lifecycle(
            substrate=TrackingComponent(),
        )
        before_audit = list(store.list_audit())
        before_events = list(store.list_events())
        try:
            with store.transaction():
                with pytest.raises(RuntimeError, match="active_transaction"):
                    lifecycle.shutdown(reason="inside-caller-transaction")
                assert lifecycle.state == "open"
                assert component_calls == []
                assert store.list_audit() == before_audit
                assert store.list_events() == before_events

            outcome = lifecycle.shutdown(reason="after-caller-transaction")
            assert outcome["ok"] is True
            assert lifecycle.closed
            assert component_calls == ["shutdown"]
        finally:
            if not lifecycle.closed:
                store.close()

    def test_async_shutdown_rejects_caller_transaction_before_side_effects(
        self,
    ) -> None:
        async def scenario() -> None:
            component_calls: list[str] = []

            class TrackingComponent:
                async def ashutdown(self) -> bool:
                    component_calls.append("ashutdown")
                    return True

            store, lifecycle, _capability = _open_test_lifecycle(
                substrate=TrackingComponent(),
            )
            before_audit = list(store.list_audit())
            before_events = list(store.list_events())
            try:
                with store.transaction():
                    with pytest.raises(RuntimeError, match="active_transaction"):
                        await lifecycle.ashutdown(
                            reason="async-inside-caller-transaction",
                        )
                    assert lifecycle.state == "open"
                    assert component_calls == []
                    assert store.list_audit() == before_audit
                    assert store.list_events() == before_events

                outcome = await lifecycle.ashutdown(
                    reason="async-after-caller-transaction",
                )
                assert outcome["ok"] is True
                assert lifecycle.closed
                assert component_calls == ["ashutdown"]
            finally:
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_sync_shutdown_terminalizes_graph_after_store_ownership_release(
        self,
    ) -> None:
        component_calls: list[str] = []

        class TrackingComponent:
            def shutdown(self) -> bool:
                component_calls.append("shutdown")
                return True

        store, lifecycle, _capability = _open_test_lifecycle(
            substrate=TrackingComponent(),
        )
        store.close()

        outcome = lifecycle.shutdown(reason="backend-already-released")

        assert outcome["ok"] is True
        assert outcome["already_shutdown"] is False
        assert outcome["reason"] == "backend-already-released"
        assert outcome["warnings"][0]["component"] == "shutdown_evidence"
        assert "already released" in outcome["warnings"][0]["error"]
        assert component_calls == ["shutdown"]
        assert lifecycle.closed
        assert store._admission_commit_guard is None

    def test_async_shutdown_terminalizes_graph_after_store_ownership_release(
        self,
    ) -> None:
        async def scenario() -> None:
            component_calls: list[str] = []

            class TrackingComponent:
                async def ashutdown(self) -> bool:
                    component_calls.append("ashutdown")
                    return True

            store, lifecycle, _capability = _open_test_lifecycle(
                substrate=TrackingComponent(),
            )
            store.close()

            outcome = await lifecycle.ashutdown(
                reason="async-backend-already-released",
            )

            assert outcome["ok"] is True
            assert outcome["already_shutdown"] is False
            assert outcome["reason"] == "async-backend-already-released"
            assert outcome["warnings"][0]["component"] == "shutdown_evidence"
            assert "already released" in outcome["warnings"][0]["error"]
            assert component_calls == ["ashutdown"]
            assert lifecycle.closed
            assert store._admission_commit_guard is None

        asyncio.run(scenario())

    def test_shutdown_follower_never_waits_while_holding_store_transaction(
        self,
    ) -> None:
        store, lifecycle, _capability = _open_test_lifecycle()
        admission_entered = threading.Event()
        release_admission = threading.Event()
        leader_results: list[dict[str, object]] = []
        leader_errors: list[BaseException] = []

        def admitted_operation() -> None:
            with lifecycle.admit():
                admission_entered.set()
                assert release_admission.wait(timeout=2)

        def shutdown_leader() -> None:
            try:
                leader_results.append(
                    lifecycle.shutdown(reason="leader-waits-for-admission")
                )
            except BaseException as exc:
                leader_errors.append(exc)

        operation_thread = threading.Thread(target=admitted_operation)
        leader_thread = threading.Thread(target=shutdown_leader)
        operation_thread.start()
        assert admission_entered.wait(timeout=1)
        leader_thread.start()
        try:
            for _ in range(1000):
                if lifecycle.state == "stopping":
                    break
                time.sleep(0.001)
            assert lifecycle.state == "stopping"

            with store.transaction():
                with pytest.raises(RuntimeError, match="active_transaction"):
                    lifecycle.shutdown(reason="follower-inside-transaction")

            release_admission.set()
            operation_thread.join(timeout=2)
            leader_thread.join(timeout=2)
            assert not operation_thread.is_alive()
            assert not leader_thread.is_alive()
            assert leader_errors == []
            assert leader_results[0]["ok"] is True
            assert lifecycle.closed
        finally:
            release_admission.set()
            operation_thread.join(timeout=2)
            leader_thread.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    def test_async_shutdown_drains_irreversible_store_close_before_cancellation(
        self,
    ) -> None:
        class BlockingCloseStore(SQLiteStore):
            def __init__(self) -> None:
                self.ownership_released = threading.Event()
                self.allow_return = threading.Event()
                self.close_threads: list[int] = []
                super().__init__(":memory:")

            def release_admission_guard_and_close(self, expected_guard):
                self.close_threads.append(threading.get_ident())
                outcome = super().release_admission_guard_and_close(
                    expected_guard
                )
                self.ownership_released.set()
                assert self.allow_return.wait(timeout=2)
                return outcome

        async def scenario() -> None:
            store = BlockingCloseStore()
            _store, lifecycle, _capability = _open_test_lifecycle(store=store)
            caller_thread = threading.get_ident()
            task = asyncio.create_task(
                lifecycle.ashutdown(reason="cancel-during-store-close")
            )
            try:
                assert await asyncio.to_thread(store.ownership_released.wait, 2.0)
                assert store.close_threads[0] != caller_thread
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                store.allow_return.set()
                with pytest.raises(asyncio.CancelledError):
                    await task

                assert lifecycle.closed
                readback = await lifecycle.ashutdown(
                    reason="idempotent-after-cancelled-close",
                )
                assert readback["ok"] is True
                assert readback["already_shutdown"] is True
            finally:
                store.allow_return.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_async_shutdown_cancel_and_warning_are_caller_local_and_readable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        warning_text = "close warning after ownership release"

        class BlockingWarningCloseStore(SQLiteStore):
            def __init__(self) -> None:
                self.ownership_released = threading.Event()
                self.allow_return = threading.Event()
                super().__init__(":memory:")

            def close(self) -> None:
                super().close()
                raise RuntimeError(warning_text)

            def release_admission_guard_and_close(self, expected_guard):
                outcome = super().release_admission_guard_and_close(
                    expected_guard
                )
                self.ownership_released.set()
                assert self.allow_return.wait(timeout=2)
                return outcome

        async def scenario() -> None:
            store = BlockingWarningCloseStore()
            _store, lifecycle, _capability = _open_test_lifecycle(store=store)
            follower_joined = threading.Event()
            original_start_attempt = lifecycle._start_attempt

            def observed_start_attempt(*, caller_task):
                outcome = original_start_attempt(caller_task=caller_task)
                attempt, is_leader, _early = outcome
                if attempt is not None and not is_leader:
                    follower_joined.set()
                return outcome

            monkeypatch.setattr(
                lifecycle,
                "_start_attempt",
                observed_start_attempt,
            )
            leader = asyncio.create_task(
                lifecycle.ashutdown(reason="cancelled-warning-leader")
            )
            follower: asyncio.Task[dict[str, object]] | None = None
            try:
                assert await asyncio.to_thread(store.ownership_released.wait, 2.0)

                follower = asyncio.create_task(
                    lifecycle.ashutdown(reason="warning-follower")
                )
                assert await asyncio.to_thread(follower_joined.wait, 2.0)

                leader.cancel()
                await asyncio.sleep(0)
                assert not leader.done()
                store.allow_return.set()
                with pytest.raises(asyncio.CancelledError):
                    await leader

                follower_result = await follower
                expected_warning = {
                    "component": "store",
                    "error": warning_text,
                    "error_type": "RuntimeError",
                }
                assert follower_result == {
                    "ok": True,
                    "already_shutdown": False,
                    "reason": "cancelled-warning-leader",
                    "warnings": [expected_warning],
                }
                assert await lifecycle.ashutdown() == {
                    **follower_result,
                    "already_shutdown": True,
                }
            finally:
                store.allow_return.set()
                if not leader.done():
                    leader.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await leader
                if follower is not None and not follower.done():
                    follower.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await follower
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_shutdown_control_warning_after_store_release_leaves_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store, lifecycle, _capability = _open_test_lifecycle()
        original_close = store.close
        warning = KeyboardInterrupt("close interrupted after ownership release")

        def close_then_interrupt() -> None:
            original_close()
            raise warning

        monkeypatch.setattr(store, "close", close_then_interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            lifecycle.shutdown(reason="control-warning-after-close")
        assert caught.value is warning
        assert lifecycle.closed
        assert store._admission_commit_guard is None

        readback = lifecycle.shutdown(reason="must-remain-closed")
        assert readback["ok"] is True
        assert readback["already_shutdown"] is True
        assert readback["warnings"] == [
            {
                "component": "store",
                "error": str(warning),
                "error_type": "KeyboardInterrupt",
            }
        ]

    def test_sync_shutdown_control_warning_is_caller_local_to_leader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        warning = KeyboardInterrupt("leader-only close control warning")

        class BlockingControlCloseStore(SQLiteStore):
            def __init__(self) -> None:
                self.ownership_released = threading.Event()
                self.allow_return = threading.Event()
                super().__init__(":memory:")

            def close(self) -> None:
                super().close()
                raise warning

            def release_admission_guard_and_close(self, expected_guard):
                outcome = super().release_admission_guard_and_close(
                    expected_guard
                )
                self.ownership_released.set()
                assert self.allow_return.wait(timeout=2)
                return outcome

        store = BlockingControlCloseStore()
        _store, lifecycle, _capability = _open_test_lifecycle(store=store)
        follower_joined = threading.Event()
        original_start_attempt = lifecycle._start_attempt

        def observed_start_attempt(*, caller_task):
            outcome = original_start_attempt(caller_task=caller_task)
            attempt, is_leader, _early = outcome
            if attempt is not None and not is_leader:
                follower_joined.set()
            return outcome

        monkeypatch.setattr(
            lifecycle,
            "_start_attempt",
            observed_start_attempt,
        )
        leader_errors: list[BaseException] = []
        follower_errors: list[BaseException] = []
        follower_results: list[dict[str, object]] = []

        def run_leader() -> None:
            try:
                lifecycle.shutdown(reason="control-warning-leader")
            except BaseException as exc:
                leader_errors.append(exc)

        def run_follower() -> None:
            try:
                follower_results.append(
                    lifecycle.shutdown(reason="control-warning-follower")
                )
            except BaseException as exc:
                follower_errors.append(exc)

        leader = threading.Thread(target=run_leader)
        follower = threading.Thread(target=run_follower)
        leader.start()
        assert store.ownership_released.wait(timeout=1)
        follower.start()
        assert follower_joined.wait(timeout=1)
        try:
            store.allow_return.set()
            leader.join(timeout=2)
            follower.join(timeout=2)
            assert not leader.is_alive()
            assert not follower.is_alive()
            assert leader_errors == [warning]
            assert follower_errors == []
            expected_warning = {
                "component": "store",
                "error": str(warning),
                "error_type": "KeyboardInterrupt",
            }
            assert follower_results == [
                {
                    "ok": True,
                    "already_shutdown": False,
                    "reason": "control-warning-leader",
                    "warnings": [expected_warning],
                }
            ]
            assert lifecycle.shutdown() == {
                **follower_results[0],
                "already_shutdown": True,
            }
        finally:
            store.allow_return.set()
            leader.join(timeout=2)
            follower.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    def test_shutdown_close_claim_loses_race_without_closing_active_store(
        self,
    ) -> None:
        store, lifecycle, _capability = _open_test_lifecycle()
        finalizer_entered = threading.Event()
        release_finalizer = threading.Event()
        transaction_entered = threading.Event()
        release_transaction = threading.Event()
        shutdown_results: list[dict[str, object]] = []
        shutdown_errors: list[BaseException] = []

        def finalizer() -> bool:
            finalizer_entered.set()
            assert release_finalizer.wait(timeout=2)
            return True

        def raw_transaction() -> None:
            assert finalizer_entered.wait(timeout=2)
            with store.transaction():
                transaction_entered.set()
                assert release_transaction.wait(timeout=2)

        def shutdown_leader() -> None:
            try:
                shutdown_results.append(
                    lifecycle.shutdown(reason="close-claim-race")
                )
            except BaseException as exc:
                shutdown_errors.append(exc)

        lifecycle.bind_finalizer(finalizer)
        transaction_thread = threading.Thread(target=raw_transaction)
        shutdown_thread = threading.Thread(target=shutdown_leader)
        transaction_thread.start()
        shutdown_thread.start()
        try:
            assert transaction_entered.wait(timeout=2)
            release_finalizer.set()
            shutdown_thread.join(timeout=2)
            assert not shutdown_thread.is_alive()
            assert shutdown_errors == []
            assert shutdown_results[0]["ok"] is False
            assert shutdown_results[0]["store_stopped"] is False
            assert lifecycle.state == "close_failed"

            release_transaction.set()
            transaction_thread.join(timeout=2)
            assert not transaction_thread.is_alive()
            assert store.list_audit()

            retry = lifecycle.shutdown(reason="close-claim-race-retry")
            assert retry["ok"] is True
            assert lifecycle.closed
        finally:
            release_finalizer.set()
            release_transaction.set()
            transaction_thread.join(timeout=2)
            shutdown_thread.join(timeout=2)
            if not lifecycle.closed:
                store.close()

    def test_async_recovery_release_rejects_caller_store_lock_before_teardown(
        self,
    ) -> None:
        async def scenario() -> None:
            component_calls: list[str] = []

            class TrackingComponent:
                async def ashutdown(self) -> bool:
                    component_calls.append("ashutdown")
                    return True

            store, lifecycle, capability = _open_test_lifecycle(
                substrate=TrackingComponent(),
            )
            _mark_test_recovery_fence(
                lifecycle,
                "publication-caller-store-lock",
            )
            try:
                with store.locked():
                    with pytest.raises(
                        RuntimeError,
                        match="current_thread_locked",
                    ):
                        await lifecycle.arelease_recovery_diagnostics(
                            capability=capability,
                        )
                    assert component_calls == []
                    assert lifecycle.state == "close_failed"
                    assert lifecycle._recovery_diagnostics_release_started is False

                outcome = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert outcome["ok"] is True
                assert lifecycle.closed
                assert component_calls == ["ashutdown"]
            finally:
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    def test_recovery_release_wrong_store_owner_never_closes_successor(
        self,
    ) -> None:
        store, lifecycle, capability = _open_test_lifecycle()
        _mark_test_recovery_fence(
            lifecycle,
            "publication-stale-recovery-owner",
        )
        lifecycle_guard = lifecycle._admission_commit_guard_binding
        successor_guard = lambda: nullcontext()
        assert store.unbind_admission_commit_guard(lifecycle_guard) is True
        store.bind_admission_commit_guard(successor_guard)
        try:
            first = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert first["ok"] is False
            assert first["store_released"] is False
            assert lifecycle.state == "close_failed"
            assert store._admission_commit_guard is successor_guard
            assert store.list_audit() == []

            assert store.unbind_admission_commit_guard(successor_guard) is True
            store.bind_admission_commit_guard(lifecycle_guard)
            second = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert second["ok"] is True
            assert lifecycle.closed
        finally:
            if not lifecycle.closed:
                store.unbind_admission_commit_guard(successor_guard)
                store.unbind_admission_commit_guard(lifecycle_guard)
                store.close()

    def test_async_cancel_during_irreversible_store_handoff_finishes_closed(
        self,
    ) -> None:
        class BlockingCloseStore(SQLiteStore):
            def __init__(self) -> None:
                self.ownership_released = threading.Event()
                self.allow_return = threading.Event()
                self.close_warning = RuntimeError(
                    "close warning after irreversible handoff"
                )
                super().__init__(":memory:")

            def close(self) -> None:
                super().close()
                raise self.close_warning

            def release_admission_guard_and_close(self, expected_guard):
                outcome = super().release_admission_guard_and_close(
                    expected_guard
                )
                self.ownership_released.set()
                assert self.allow_return.wait(timeout=2)
                return outcome

        async def scenario() -> None:
            store = BlockingCloseStore()
            _store, lifecycle, capability = _open_test_lifecycle(store=store)
            _mark_test_recovery_fence(
                lifecycle,
                "publication-cancel-at-store-commit",
            )
            task = asyncio.create_task(
                lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
            )
            try:
                assert await asyncio.to_thread(store.ownership_released.wait, 2.0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                store.allow_return.set()
                with pytest.raises(asyncio.CancelledError):
                    await task

                assert lifecycle.closed
                readback = await lifecycle.arelease_recovery_diagnostics(
                    capability=capability,
                )
                assert readback["ok"] is True
                assert readback["already_released"] is True
                assert readback["warnings"][0]["error_type"] == "RuntimeError"
            finally:
                store.allow_return.set()
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                if not lifecycle.closed:
                    store.close()

        asyncio.run(scenario())

    @pytest.mark.parametrize("warning_type", [RuntimeError, KeyboardInterrupt])
    def test_store_warning_after_ownership_release_closes_and_is_readable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        warning_type: type[BaseException],
    ) -> None:
        store, lifecycle, capability = _open_test_lifecycle()
        _mark_test_recovery_fence(
            lifecycle,
            f"publication-store-warning-{warning_type.__name__}",
        )
        original_close = store.close
        warning = warning_type("close reported after ownership release")

        def close_then_warn() -> None:
            original_close()
            raise warning

        monkeypatch.setattr(store, "close", close_then_warn)
        if issubclass(warning_type, Exception):
            first = lifecycle.release_recovery_diagnostics(
                capability=capability,
            )
            assert first["ok"] is True
            assert first["warnings"][0]["error_type"] == warning_type.__name__
        else:
            with pytest.raises(warning_type) as caught:
                lifecycle.release_recovery_diagnostics(
                    capability=capability,
                )
            assert caught.value is warning

        assert lifecycle.closed
        readback = lifecycle.release_recovery_diagnostics(
            capability=capability,
        )
        assert readback["ok"] is True
        assert readback["already_released"] is True
        assert readback["warnings"][0]["error_type"] == warning_type.__name__

    def test_recovery_release_and_failed_assembly_cleanup_reject_wrong_scope(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        finalizer_calls: list[str] = []
        released = False

        def finalizer() -> None:
            finalizer_calls.append("finalizer")

        runtime.bind_shutdown_finalizer(finalizer)
        before_audit = list(runtime.store.list_audit())
        before_events = list(runtime.store.list_events())
        try:
            with pytest.raises(
                RuntimeError,
                match="requires an active recovery fence",
            ):
                runtime.release_recovery_diagnostics()
            with pytest.raises(
                RuntimeError,
                match="failed assembly cleanup is unavailable after Runtime open",
            ):
                runtime.lifecycle.cleanup_failed_assembly()
            with pytest.raises(
                RuntimeError,
                match="failed assembly cleanup is unavailable after Runtime open",
            ):
                asyncio.run(runtime.lifecycle.acleanup_failed_assembly())

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-release-rejection",
                )
                with pytest.raises(
                    RuntimeError,
                    match="requires admission drain",
                ):
                    runtime.release_recovery_diagnostics()
                with pytest.raises(
                    RuntimeError,
                    match="invalid recovery diagnostics release capability",
                ):
                    runtime.lifecycle.release_recovery_diagnostics(
                        capability=object(),
                    )
                with pytest.raises(
                    RuntimeError,
                    match="invalid recovery diagnostics release capability",
                ):
                    asyncio.run(
                        runtime.lifecycle.arelease_recovery_diagnostics(
                            capability=object(),
                        )
                    )

            with pytest.raises(
                RuntimeError,
                match="failed assembly cleanup is unavailable after Runtime open",
            ):
                runtime.lifecycle.cleanup_failed_assembly()
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
            assert finalizer_calls == []

            outcome = runtime.release_recovery_diagnostics()
            released = outcome["ok"] is True
            assert outcome["recovery_diagnostics_released"] is True
            assert finalizer_calls == []
        finally:
            if not released:
                runtime.store.close()

    def test_read_only_admission_cannot_promote_while_stopping(self) -> None:
        runtime = Runtime.open("local")
        entered = threading.Event()
        release = threading.Event()
        outcome: list[dict[str, object]] = []

        def finalizer() -> None:
            entered.set()
            assert release.wait(timeout=2)

        runtime.bind_shutdown_finalizer(finalizer)
        thread = threading.Thread(
            target=lambda: outcome.append(
                runtime.shutdown(actor="test", reason="read-only-stopping")
            )
        )
        thread.start()
        try:
            assert entered.wait(timeout=1)
            assert runtime.lifecycle.state == "stopping"

            with runtime.lifecycle.admit(read_only=True):
                assert not runtime.store.namespace_exists("read-only-stopping")
                with pytest.raises(
                    RuntimeError,
                    match="read-only runtime admission lease cannot authorize mutation",
                ):
                    with runtime.lifecycle.admit():
                        pass
                with pytest.raises(
                    RuntimeError,
                    match="read-only runtime admission lease cannot authorize mutation",
                ):
                    with runtime.store.transaction() as cur:
                        cur.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            ("read-only-stopping", None, "{}", "test", "1", "1"),
                        )

            assert not runtime.store.namespace_exists("read-only-stopping")
        finally:
            release.set()
            thread.join(timeout=2)
            if not runtime.lifecycle.closed:
                runtime.close()
        assert not thread.is_alive()
        assert outcome == [
            {
                "ok": True,
                "already_shutdown": False,
                "reason": "read-only-stopping",
            }
        ]

    def test_read_only_admission_cannot_promote_after_close_failed(self) -> None:
        runtime = Runtime.open("local")
        attempts = 0

        def finalizer() -> bool:
            nonlocal attempts
            attempts += 1
            return attempts > 1

        runtime.bind_shutdown_finalizer(finalizer)
        try:
            first = runtime.shutdown(actor="test", reason="read-only-close-failed")
            assert first["ok"] is False
            assert runtime.lifecycle.state == "close_failed"

            with runtime.lifecycle.admit(read_only=True):
                assert not runtime.store.namespace_exists("read-only-close-failed")
                with pytest.raises(
                    RuntimeError,
                    match="read-only runtime admission lease cannot authorize mutation",
                ):
                    with runtime.lifecycle.admit():
                        pass
                with pytest.raises(
                    RuntimeError,
                    match="read-only runtime admission lease cannot authorize mutation",
                ):
                    with runtime.store.transaction() as cur:
                        cur.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                "read-only-close-failed",
                                None,
                                "{}",
                                "test",
                                "1",
                                "1",
                            ),
                        )

            assert not runtime.store.namespace_exists("read-only-close-failed")
            second = runtime.shutdown(
                actor="test",
                reason="read-only-close-failed-retry",
            )
            assert second["ok"] is True
            assert runtime.lifecycle.closed
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_recovery_terminalization_scope_is_opaque_and_publication_bound(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        publication_id = "publication-terminalization-scope"

        def insert_namespace(namespace: str) -> None:
            with runtime.store.transaction() as cur:
                cur.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    (namespace, None, "{}", "test", "1", "1"),
                )

        try:
            with runtime.lifecycle.admit():
                with pytest.raises(
                    RuntimeError,
                    match="invalid recovery terminalization capability",
                ):
                    with runtime.lifecycle.recovery_terminalization_scope(
                        publication_id,
                        capability=object(),
                    ):
                        pass

                runtime.lifecycle.mark_recovery_required(
                    publication_id=publication_id,
                )

                with pytest.raises(
                    RuntimeError,
                    match="does not match the active recovery fence",
                ):
                    with runtime.process._recovery_terminalization_scope(
                        "publication-other",
                    ):
                        pass

                with pytest.raises(
                    RuntimeError,
                    match="runtime is not accepting operations: state=close_failed",
                ):
                    insert_namespace("terminalization-denied")

                with runtime.process._recovery_terminalization_scope(
                    publication_id,
                ):
                    insert_namespace("terminalization-allowed")

            assert not runtime.store.namespace_exists("terminalization-denied")
            assert runtime.store.namespace_exists("terminalization-allowed")
        finally:
            runtime.close()

    def test_recovery_hooks_worker_and_open_are_strictly_ordered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        order: list[tuple[str, str, bool]] = []
        original_recover = ProtectedOperationSDK.recover_prepared
        original_hooks = RuntimeModuleRegistry.run_startup_hooks
        original_start = ObjectTaskManager.start_worker

        def recover(
            sdk: ProtectedOperationSDK,
            *,
            page_size: int = 500,
        ) -> object:
            result = original_recover(sdk, page_size=page_size)
            order.append(('recover', 'recovering', False))
            return result

        def hooks(registry: RuntimeModuleRegistry) -> None:
            lifecycle = registry._hook_services.lifecycle
            # Bind the ordering assertion to the Runtime being assembled.
            # Other Runtime instances in the same xdist worker may still have
            # a legitimately active ObjectTask thread with the same name.
            worker_started = any(item[0] == 'worker' for item in order)
            order.append(('hooks', lifecycle.state, worker_started))
            original_hooks(registry)

        def start(manager: ObjectTaskManager) -> None:
            host = manager._tools._tool_context_host
            order.append(('worker', host.lifecycle.state, manager._started))
            original_start(manager)

        monkeypatch.setattr(ProtectedOperationSDK, 'recover_prepared', recover)
        monkeypatch.setattr(RuntimeModuleRegistry, 'run_startup_hooks', hooks)
        monkeypatch.setattr(ObjectTaskManager, 'start_worker', start)

        unrelated_release = threading.Event()
        unrelated_worker = threading.Thread(
            target=unrelated_release.wait,
            name='agent-libos-object-tasks',
            daemon=True,
        )
        unrelated_worker.start()
        try:
            runtime = Runtime.open('local')
            try:
                assert [item[0] for item in order] == ['recover', 'hooks', 'worker']
                assert order[1] == ('hooks', 'starting', False)
                assert order[2] == ('worker', 'starting', False)
                assert runtime.lifecycle.state == 'open'
                assert runtime.object_tasks._thread.is_alive()
            finally:
                runtime.close()
        finally:
            unrelated_release.set()
            unrelated_worker.join(timeout=1)
            assert not unrelated_worker.is_alive()

    def test_stopping_admission_rejects_mutations_without_writes(self) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(goal='admission owner')
        runtime.tools.configure_process_tools(pid, ['get_working_directory'], assigned_by='test')
        entered = threading.Event()
        release = threading.Event()

        def finalizer() -> None:
            entered.set()
            assert release.wait(timeout=2)

        runtime.bind_shutdown_finalizer(finalizer)
        outcome: list[dict[str, object]] = []
        thread = threading.Thread(
            target=lambda: outcome.append(runtime.shutdown(actor='test', reason='admission')),
        )
        thread.start()
        assert entered.wait(timeout=1)
        before = {
            table: len(runtime.store.select_table_rows(table))
            for table in (
                'processes',
                'objects',
                'operations',
                'audit_records',
                'events',
                'runtime_publications',
                'capabilities',
                'tool_candidates',
                'tools',
            )
        }

        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.process.spawn(goal='rejected spawn')
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.tools.call(pid, 'get_working_directory', {})
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.memory.create_object(pid, 'artifact', {'rejected': True})
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.exec_process(pid, 'base-agent:v0', goal='rejected exec')
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.tools.configure_process_tools(
                pid,
                ['human_output'],
                assigned_by='rejected',
            )
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.tools.propose(
                pid,
                {
                    'name': 'rejected_jit',
                    'description': 'must not persist',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                'export function run() { return {}; }',
            )
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.tools.rehydrate_registered_jit_tools()
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.modules.run_startup_hooks()
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.modules.load_core_module()
        with pytest.raises(RuntimeError, match='not accepting operations'):
            runtime.checkpoint.reconcile_terminal_restore_publications()

        after = {
            table: len(runtime.store.select_table_rows(table))
            for table in before
        }
        assert after == before
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert outcome[0]['ok'] is True

    def test_checkpoint_terminal_reconciliation_rejects_recovery_fence_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        dispatched = False

        def reconcile_terminal_publications() -> list[str]:
            nonlocal dispatched
            dispatched = True
            return []

        monkeypatch.setattr(
            runtime.checkpoint._restore_reconciler,
            "reconcile_terminal_publications",
            reconcile_terminal_publications,
        )
        try:
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-checkpoint-reconciliation-fence",
                )
                with pytest.raises(
                    RuntimeError,
                    match="not accepting operations: state=close_failed",
                ):
                    runtime.checkpoint.reconcile_terminal_restore_publications()

            assert dispatched is False
        finally:
            runtime.close()

    def test_stopping_rejects_all_human_mutators_without_writes(self) -> None:
        runtime = Runtime.open("local")
        pid = runtime.process.spawn(goal="Human admission owner")
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {"type": "question", "question": "admission fixture"},
            blocking=False,
        )
        request = runtime.human.get(request_id)
        entered = threading.Event()
        release = threading.Event()

        def finalizer() -> None:
            entered.set()
            assert release.wait(timeout=5)

        runtime.bind_shutdown_finalizer(finalizer)
        outcome: list[dict[str, object]] = []
        thread = threading.Thread(
            target=lambda: outcome.append(
                runtime.shutdown(actor="test", reason="Human admission")
            ),
        )
        thread.start()
        assert entered.wait(timeout=2)

        tables = (
            "audit_records",
            "capabilities",
            "capability_use_reservations",
            "data_flow_decisions",
            "events",
            "external_effect_transitions",
            "external_effects",
            "human_requests",
            "operation_evidence",
            "operations",
            "process_messages",
            "processes",
        )
        before = {
            table: copy.deepcopy(runtime.store.select_table_rows(table))
            for table in tables
        }
        before_claims = set(runtime.human._terminal_claims)
        before_receipts = copy.copy(runtime.human.presentation._receipts)
        human = runtime.config.runtime.default_human
        effect = SimpleNamespace(
            effect_id="eff_rejected_human_recovery",
            provider_metadata={"context": {"request_id": request_id}},
        )
        provider = object()

        sync_calls = {
            "answer_for_request": lambda: runtime.human.answer_for_request(request_id),
            "approve": lambda: runtime.human.approve(request_id),
            "approve_for_presentation": lambda: runtime.human.approve_for_presentation(
                request_id,
                presentation="gui",
            ),
            "ask": lambda: runtime.human.ask(pid, "rejected question", blocking=False),
            "cancel_pending_for_process": lambda: runtime.human.cancel_pending_for_process(
                pid,
                actor="test",
                reason="rejected cancellation",
            ),
            "drain_terminal_queue": lambda: runtime.human.drain_terminal_queue(),
            "interrupt": lambda: runtime.human.interrupt(pid, "pause"),
            "list_for_presentation": lambda: runtime.human.list_for_presentation(
                presentation="gui",
                provider=provider,
            ),
            "list_for_presentation_window": lambda: runtime.human.list_for_presentation_window(
                presentation="gui",
                provider=provider,
            ),
            "output": lambda: runtime.human.output(pid, "rejected output"),
            "present_request_view": lambda: runtime.human.present_request_view(
                request,
                presentation="gui",
                provider=provider,
            ),
            "present_terminal_request": lambda: runtime.human.present_terminal_request(
                request,
                suffix="[rejected]",
            ),
            "process_next_terminal": lambda: runtime.human.process_next_terminal(),
            "query": lambda: runtime.human.query(
                pid,
                human,
                {"type": "question", "question": "rejected query"},
                blocking=False,
            ),
            "recover_prepared_output": lambda: runtime.human.recover_prepared_output(effect),
            "reject": lambda: runtime.human.reject(request_id),
            "reject_for_presentation": lambda: runtime.human.reject_for_presentation(
                request_id,
                presentation="gui",
            ),
            "request_data_release": lambda: runtime.human.request_data_release(
                pid=pid,
                human=human,
                request={
                    "type": "data_release_approval",
                    "context": {},
                    "requested_once_capability": {},
                },
                blocking=False,
            ),
            "request_permission": lambda: runtime.human.request_permission(
                pid,
                human,
                "human:default",
                ["write"],
                "rejected permission request",
                blocking=False,
            ),
            "send_process_message": lambda: runtime.human.send_process_message(
                pid,
                "rejected message",
            ),
        }
        async_calls = {
            "adrain_terminal_queue": lambda: runtime.human.adrain_terminal_queue(),
            "aprocess_next_terminal": lambda: runtime.human.aprocess_next_terminal(),
        }
        assert set(sync_calls) | set(async_calls) == HUMAN_PUBLIC_MUTATION_METHODS

        try:
            for call in sync_calls.values():
                with pytest.raises(RuntimeError, match="not accepting operations"):
                    call()
            for call in async_calls.values():
                with pytest.raises(RuntimeError, match="not accepting operations"):
                    asyncio.run(call())

            after = {
                table: runtime.store.select_table_rows(table)
                for table in tables
            }
            assert after == before
            assert runtime.human._terminal_claims == before_claims
            assert runtime.human.presentation._receipts == before_receipts
        finally:
            release.set()
            thread.join(timeout=5)
            if thread.is_alive():
                pytest.fail("Runtime shutdown did not finish")
        assert outcome[0]["ok"] is True

    def test_failed_finalizer_retry_resumes_at_first_incomplete_handle(self) -> None:
        runtime = Runtime.open('local')
        calls = {'a': 0, 'b': 0}
        b_succeeds = False

        def finalizer_a() -> bool:
            calls['a'] += 1
            return True

        def finalizer_b() -> bool:
            calls['b'] += 1
            return b_succeeds

        runtime.bind_shutdown_finalizer(finalizer_a)
        runtime.bind_shutdown_finalizer(finalizer_b)

        first = runtime.shutdown(actor='test', reason='retry-finalizer')
        assert first['ok'] is False
        assert runtime.lifecycle.closed is False
        assert runtime.store.list_processes() == []
        assert calls == {'a': 1, 'b': 1}

        b_succeeds = True
        second = runtime.shutdown(actor='test', reason='retry-finalizer')
        assert second['ok'] is True
        assert calls == {'a': 1, 'b': 2}

    def test_store_close_failure_is_reported_and_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        original_close = runtime.store.close
        attempts = 0

        def fail_once() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('injected store close failure')
            original_close()

        monkeypatch.setattr(runtime.store, 'close', fail_once)

        first = runtime.close()
        assert first['ok'] is False
        assert runtime.lifecycle.closed is False
        assert runtime.store.list_processes() == []
        second = runtime.close()
        assert second['ok'] is True
        assert attempts == 2

    def test_sync_shutdown_rejects_async_finalizer_inside_running_loop_before_stopping(self) -> None:
        async def exercise() -> None:
            runtime = await Runtime.aopen('local')

            async def async_finalizer() -> None:
                await asyncio.sleep(0)

            runtime.bind_shutdown_finalizer(async_finalizer)
            with pytest.raises(RuntimeError, match=r'use await runtime\.ashutdown'):
                runtime.shutdown(actor='test', reason='wrong-facade')

            assert runtime.lifecycle.state == 'open'
            pid = runtime.process.spawn(goal='still admitted')
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert (await runtime.ashutdown(actor='test', reason='right-facade'))['ok'] is True

        asyncio.run(exercise())

    def test_sync_shutdown_rejects_async_callable_finalizer_before_stopping(self) -> None:
        class AsyncFinalizer:
            async def __call__(self) -> None:
                await asyncio.sleep(0)

        async def exercise() -> None:
            runtime = await Runtime.aopen('local')
            runtime.bind_shutdown_finalizer(AsyncFinalizer())

            with pytest.raises(RuntimeError, match=r'use await runtime\.ashutdown'):
                runtime.shutdown(actor='test', reason='wrong-callable-facade')

            assert runtime.lifecycle.state == 'open'
            pid = runtime.process.spawn(goal='async callable still admitted')
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            result = await runtime.ashutdown(
                actor='test',
                reason='right-callable-facade',
            )
            assert result['ok'] is True

        asyncio.run(exercise())

    def test_cancelled_blocking_wrapper_waits_for_underlying_worker(self) -> None:
        async def exercise() -> None:
            runtime = await Runtime.aopen('local')
            entered = threading.Event()
            release = threading.Event()

            def blocking_work() -> str:
                entered.set()
                assert release.wait(timeout=2)
                return 'done'

            task = asyncio.create_task(runtime.blocking_work.run(blocking_work))
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime.blocking_work.active_count() == 0
            assert (await runtime.ashutdown(actor='test', reason='worker-drained'))['ok'] is True

        asyncio.run(exercise())

    def test_cancelled_human_terminal_worker_blocks_shutdown_until_drained(self) -> None:
        async def exercise() -> None:
            runtime = await Runtime.aopen(
                'local',
                config=AgentLibOSConfig(
                    scheduler=SchedulerDefaults(shutdown_join_timeout_s=0.01),
                ),
            )
            entered = threading.Event()
            release = threading.Event()

            def blocking_terminal(**_kwargs: object) -> None:
                entered.set()
                assert release.wait(timeout=2)

            runtime.human.process_next_terminal = blocking_terminal  # type: ignore[method-assign]
            runtime.blocking_work._shutdown_timeout_s = 0.01
            task = asyncio.create_task(runtime.human.aprocess_next_terminal())
            assert await asyncio.to_thread(entered.wait, 1)

            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()

            first = await runtime.ashutdown(
                actor='test',
                reason='human-worker-still-running',
            )
            assert first['ok'] is False
            assert runtime.lifecycle.state == 'close_failed'
            assert runtime.store.list_processes() == []

            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime.blocking_work.active_count() == 0

            second = await runtime.ashutdown(
                actor='test',
                reason='human-worker-drained',
            )
            assert second['ok'] is True
            assert runtime.lifecycle.state == 'closed'

        asyncio.run(exercise())

    def test_process_exec_holds_admission_through_late_image_boot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open(
            'local',
            config=AgentLibOSConfig(
                scheduler=SchedulerDefaults(shutdown_join_timeout_s=0.01),
            ),
        )
        pid = runtime.process.spawn(image='base-agent:v0', goal='before blocking exec')
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []
        original_configure = runtime.image_boot._configure_tools

        def blocking_configure(*args: object, **kwargs: object) -> object:
            entered.set()
            assert release.wait(timeout=2)
            return original_configure(*args, **kwargs)

        monkeypatch.setattr(runtime.image_boot, '_configure_tools', blocking_configure)

        def execute() -> None:
            try:
                runtime.exec_process(pid, 'base-agent:v0', goal='after blocking exec')
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=execute)
        thread.start()
        try:
            assert entered.wait(timeout=1)
            first = runtime.shutdown(actor='test', reason='exec-still-applying')

            assert first['ok'] is False
            assert runtime.lifecycle.state == 'close_failed'
            assert runtime.store.get_process(pid) is not None

            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert errors == []
            operation = [
                item
                for item in runtime.store.list_operations(pid=pid)
                if item.name == 'process.exec'
            ][-1]
            assert operation.outcome.value == 'succeeded'

            second = runtime.shutdown(actor='test', reason='exec-drained')
            assert second['ok'] is True
            assert runtime.lifecycle.state == 'closed'
        finally:
            release.set()
            if thread.is_alive():
                thread.join(timeout=2)
            runtime.close()

    def test_recovery_fence_supersedes_prior_shutdown_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open(
            "local",
            config=AgentLibOSConfig(
                scheduler=SchedulerDefaults(shutdown_join_timeout_s=0.01),
            ),
        )
        pid = runtime.process.spawn(goal="recovery fence after shutdown timeout")
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def pause_then_fail(*_args: object, **_kwargs: object) -> None:
            entered.set()
            assert release.wait(timeout=2)
            raise RuntimeError("injected late exec failure")

        def fail_restore(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected compensation failure")

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", pause_then_fail)
        monkeypatch.setattr(runtime.process_exec_state, "restore", fail_restore)

        def execute() -> None:
            try:
                runtime.exec_process(pid, "base-agent:v0")
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=execute)
        thread.start()
        try:
            assert entered.wait(timeout=1)
            first = runtime.shutdown(actor="test", reason="exec-still-applying")
            assert first["ok"] is False
            assert runtime.lifecycle.state == "close_failed"
            assert not str(runtime.lifecycle.shutdown_reason).startswith(
                "runtime.recovery_required:"
            )

            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], RuntimeRecoveryRequired)
            assert runtime.lifecycle.state == "close_failed"
            assert runtime.lifecycle.shutdown_reason == (
                f"runtime.recovery_required:{errors[0].publication_id}"
            )
        finally:
            release.set()
            if thread.is_alive():
                thread.join(timeout=2)
            runtime.close()

    def test_close_returns_shutdown_outcome(self) -> None:
        runtime = Runtime.open("local")

        result = runtime.close()

        assert result == {
            "ok": True,
            "already_shutdown": False,
            "reason": "runtime.close",
        }

    def test_concurrent_shutdown_is_single_flight(self) -> None:
        runtime = Runtime.open("local")
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def finalizer() -> None:
            nonlocal calls
            with calls_lock:
                calls += 1
            entered.set()
            assert release.wait(timeout=2.0)

        runtime.bind_shutdown_finalizer(finalizer)
        results: list[dict[str, object]] = []

        def close_runtime() -> None:
            results.append(runtime.shutdown(actor="test", reason="concurrent"))

        first = threading.Thread(target=close_runtime)
        second = threading.Thread(target=close_runtime)
        first.start()
        assert entered.wait(timeout=2.0)
        second.start()
        time.sleep(0.05)
        release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert calls == 1
        assert results == [
            {"ok": True, "already_shutdown": False, "reason": "concurrent"},
            {"ok": True, "already_shutdown": False, "reason": "concurrent"},
        ]
        assert runtime.shutdown()["already_shutdown"] is True

    def test_concurrent_async_shutdown_is_single_flight(self) -> None:
        runtime = Runtime.open("local")
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        async def finalizer() -> None:
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.to_thread(release.wait, 2.0)

        runtime.bind_shutdown_finalizer(finalizer)

        async def exercise() -> list[dict[str, object]]:
            first = asyncio.create_task(
                runtime.ashutdown(actor="test", reason="async-concurrent")
            )
            assert await asyncio.to_thread(entered.wait, 2.0)
            second = asyncio.create_task(
                runtime.ashutdown(actor="test", reason="async-concurrent")
            )
            await asyncio.sleep(0.05)
            release.set()
            return await asyncio.gather(first, second)

        results = asyncio.run(exercise())

        assert calls == 1
        assert results == [
            {"ok": True, "already_shutdown": False, "reason": "async-concurrent"},
            {"ok": True, "already_shutdown": False, "reason": "async-concurrent"},
        ]

    def test_reentrant_sync_shutdown_is_rejected_without_deadlock(self) -> None:
        runtime = Runtime.open("local")

        def finalizer() -> None:
            with pytest.raises(RuntimeError, match="reentrant runtime shutdown"):
                runtime.shutdown(actor="nested", reason="nested")

        runtime.bind_shutdown_finalizer(finalizer)

        assert runtime.shutdown(actor="test", reason="outer") == {
            "ok": True,
            "already_shutdown": False,
            "reason": "outer",
        }

    def test_reentrant_async_shutdown_is_rejected_without_deadlock(self) -> None:
        runtime = Runtime.open("local")

        async def finalizer() -> None:
            with pytest.raises(RuntimeError, match="reentrant runtime shutdown"):
                await runtime.ashutdown(actor="nested", reason="nested")

        runtime.bind_shutdown_finalizer(finalizer)

        assert asyncio.run(runtime.ashutdown(actor="test", reason="outer")) == {
            "ok": True,
            "already_shutdown": False,
            "reason": "outer",
        }

    def test_reentrant_async_shutdown_from_child_task_is_rejected(self) -> None:
        runtime = Runtime.open("local")

        async def finalizer() -> None:
            with pytest.raises(RuntimeError, match="reentrant runtime shutdown"):
                await asyncio.create_task(
                    runtime.ashutdown(actor="nested", reason="nested")
                )

        runtime.bind_shutdown_finalizer(finalizer)

        async def exercise() -> dict[str, object]:
            return await asyncio.wait_for(
                runtime.ashutdown(actor="test", reason="outer"),
                timeout=2.0,
            )

        assert asyncio.run(exercise()) == {
            "ok": True,
            "already_shutdown": False,
            "reason": "outer",
        }

    def test_async_failed_assembly_cleanup_preserves_teardown_order(self) -> None:
        order: list[str] = []

        class AsyncComponent:
            def __init__(self, name: str) -> None:
                self.name = name

            async def ashutdown(self) -> bool:
                await asyncio.sleep(0)
                order.append(self.name)
                return True

        components = {
            name: AsyncComponent(name)
            for name in (
                "scheduler",
                "object_tasks",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            )
        }
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=components["substrate"],
        )
        lifecycle.bind_components(
            scheduler=components["scheduler"],
            object_tasks=components["object_tasks"],
            modules=components["modules"],
            llms=components["llms"],
            blocking_work=components["blocking_work"],
        )

        async def finalizer() -> None:
            await asyncio.sleep(0)
            order.append("finalizer")

        lifecycle.bind_finalizer(finalizer)
        try:
            assert asyncio.run(lifecycle.acleanup_failed_assembly()) == []
            assert order == [
                "scheduler",
                "object_tasks",
                "finalizer",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
            assert lifecycle.finalizers_snapshot() == ()
            assert lifecycle.state == "close_failed"
            assert store.list_processes() == []
        finally:
            store.close()

    def test_async_failed_assembly_cleanup_collects_errors_and_continues(self) -> None:
        calls: list[str] = []
        failures_remaining = 1
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=object(),
        )

        async def failing_finalizer() -> None:
            nonlocal failures_remaining
            calls.append("failing")
            if failures_remaining:
                failures_remaining -= 1
                raise RuntimeError("injected async cleanup failure")

        async def later_finalizer() -> None:
            await asyncio.sleep(0)
            calls.append("later")

        lifecycle.bind_finalizer(failing_finalizer)
        lifecycle.bind_finalizer(later_finalizer)
        try:
            errors = asyncio.run(lifecycle.acleanup_failed_assembly())
            assert calls == ["failing", "later"]
            assert [error["error_type"] for error in errors] == ["RuntimeError"]
            assert [error["error"] for error in errors] == [
                "injected async cleanup failure"
            ]
            assert lifecycle.finalizers_snapshot() == (failing_finalizer,)
            assert lifecycle.state == "close_failed"
            assert asyncio.run(lifecycle.acleanup_failed_assembly()) == []
            assert calls == ["failing", "later", "failing"]
            assert lifecycle.finalizers_snapshot() == ()
        finally:
            store.close()

    def test_sync_failed_assembly_cleanup_drains_after_keyboard_interrupt(self) -> None:
        order: list[str] = []
        interrupts_remaining = 1

        class Component:
            def __init__(self, name: str) -> None:
                self.name = name

            def shutdown(self) -> bool:
                order.append(self.name)
                return True

        components = {
            name: Component(name)
            for name in (
                "scheduler",
                "object_tasks",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            )
        }
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=components["substrate"],
        )
        lifecycle.bind_components(
            scheduler=components["scheduler"],
            object_tasks=components["object_tasks"],
            modules=components["modules"],
            llms=components["llms"],
            blocking_work=components["blocking_work"],
        )

        def interrupted_finalizer() -> None:
            nonlocal interrupts_remaining
            order.append("interrupted")
            if interrupts_remaining:
                interrupts_remaining -= 1
                raise KeyboardInterrupt("injected cleanup interrupt")

        lifecycle.bind_finalizer(interrupted_finalizer)
        lifecycle.bind_finalizer(lambda: order.append("later"))
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                lifecycle.cleanup_failed_assembly()
            assert caught.value.subgroup(KeyboardInterrupt) is not None
            assert order == [
                "scheduler",
                "object_tasks",
                "interrupted",
                "later",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
            assert lifecycle.finalizers_snapshot() == (interrupted_finalizer,)
            assert lifecycle.state == "close_failed"
            assert lifecycle.cleanup_failed_assembly() == []
            assert order[-7:] == [
                "scheduler",
                "object_tasks",
                "interrupted",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
            assert lifecycle.finalizers_snapshot() == ()
        finally:
            store.close()

    def test_async_failed_assembly_cleanup_drains_after_cancelled_error(self) -> None:
        order: list[str] = []
        cancellations_remaining = 1

        class Component:
            def __init__(self, name: str) -> None:
                self.name = name

            async def ashutdown(self) -> bool:
                await asyncio.sleep(0)
                order.append(self.name)
                return True

        components = {
            name: Component(name)
            for name in (
                "scheduler",
                "object_tasks",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            )
        }
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=components["substrate"],
        )
        lifecycle.bind_components(
            scheduler=components["scheduler"],
            object_tasks=components["object_tasks"],
            modules=components["modules"],
            llms=components["llms"],
            blocking_work=components["blocking_work"],
        )

        async def interrupted_finalizer() -> None:
            nonlocal cancellations_remaining
            order.append("interrupted")
            if cancellations_remaining:
                cancellations_remaining -= 1
                raise asyncio.CancelledError("injected cleanup cancellation")

        async def later_finalizer() -> None:
            await asyncio.sleep(0)
            order.append("later")

        lifecycle.bind_finalizer(interrupted_finalizer)
        lifecycle.bind_finalizer(later_finalizer)

        async def exercise() -> None:
            with pytest.raises(BaseExceptionGroup) as caught:
                await lifecycle.acleanup_failed_assembly()
            assert caught.value.subgroup(asyncio.CancelledError) is not None

        try:
            asyncio.run(exercise())
            assert order == [
                "scheduler",
                "object_tasks",
                "interrupted",
                "later",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
            assert lifecycle.finalizers_snapshot() == (interrupted_finalizer,)
            assert lifecycle.state == "close_failed"

            async def retry() -> None:
                assert await lifecycle.acleanup_failed_assembly() == []

            asyncio.run(retry())
            assert order[-7:] == [
                "scheduler",
                "object_tasks",
                "interrupted",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
            assert lifecycle.finalizers_snapshot() == ()
        finally:
            store.close()

    def test_sync_failed_assembly_cleanup_retries_deferred_finalizer(self) -> None:
        attempts = 0
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=object(),
        )

        def deferred_finalizer() -> bool:
            nonlocal attempts
            attempts += 1
            return attempts > 1

        lifecycle.bind_finalizer(deferred_finalizer)
        try:
            first = lifecycle.cleanup_failed_assembly()
            assert [item["error_type"] for item in first] == ["FinalizerDeferred"]
            assert lifecycle.finalizers_snapshot() == (deferred_finalizer,)

            assert lifecycle.cleanup_failed_assembly() == []
            assert attempts == 2
            assert lifecycle.finalizers_snapshot() == ()
        finally:
            store.close()

    def test_sync_failed_assembly_cleanup_retries_only_deferred_duplicate_entry(
        self,
    ) -> None:
        attempts = 0
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=object(),
        )

        def duplicate_finalizer() -> bool:
            nonlocal attempts
            attempts += 1
            return attempts != 2

        lifecycle.bind_finalizer(duplicate_finalizer)
        lifecycle.bind_finalizer(duplicate_finalizer)
        try:
            first = lifecycle.cleanup_failed_assembly()
            assert [item["error_type"] for item in first] == ["FinalizerDeferred"]
            assert attempts == 2
            assert lifecycle.finalizers_snapshot() == (duplicate_finalizer,)

            assert lifecycle.cleanup_failed_assembly() == []
            assert attempts == 3
            assert lifecycle.finalizers_snapshot() == ()
        finally:
            store.close()

    def test_async_failed_assembly_cleanup_retries_deferred_finalizer(self) -> None:
        attempts = 0
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=object(),
        )

        async def deferred_finalizer() -> bool:
            nonlocal attempts
            await asyncio.sleep(0)
            attempts += 1
            return attempts > 1

        lifecycle.bind_finalizer(deferred_finalizer)

        async def exercise() -> None:
            first = await lifecycle.acleanup_failed_assembly()
            assert [item["error_type"] for item in first] == ["FinalizerDeferred"]
            assert lifecycle.finalizers_snapshot() == (deferred_finalizer,)

            assert await lifecycle.acleanup_failed_assembly() == []
            assert attempts == 2
            assert lifecycle.finalizers_snapshot() == ()

        try:
            asyncio.run(exercise())
        finally:
            store.close()

    def test_async_failed_assembly_cleanup_retries_interrupted_duplicate_entry(
        self,
    ) -> None:
        attempts = 0
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=object(),
        )

        async def duplicate_finalizer() -> None:
            nonlocal attempts
            await asyncio.sleep(0)
            attempts += 1
            if attempts == 2:
                raise asyncio.CancelledError("injected duplicate interruption")

        lifecycle.bind_finalizer(duplicate_finalizer)
        lifecycle.bind_finalizer(duplicate_finalizer)

        async def exercise() -> None:
            with pytest.raises(BaseExceptionGroup) as caught:
                await lifecycle.acleanup_failed_assembly()
            assert caught.value.subgroup(asyncio.CancelledError) is not None
            assert attempts == 2
            assert lifecycle.finalizers_snapshot() == (duplicate_finalizer,)

            assert await lifecycle.acleanup_failed_assembly() == []
            assert attempts == 3
            assert lifecycle.finalizers_snapshot() == ()

        try:
            asyncio.run(exercise())
        finally:
            store.close()

    def test_sync_failed_assembly_cleanup_records_component_false_and_continues(self) -> None:
        order: list[str] = []

        class Component:
            def __init__(self, name: str, *, stopped: bool = True) -> None:
                self.name = name
                self.stopped = stopped

            def shutdown(self) -> bool:
                order.append(self.name)
                return self.stopped

        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=Component("substrate"),
        )
        lifecycle.bind_components(
            scheduler=Component("scheduler", stopped=False),
            object_tasks=Component("object_tasks"),
            modules=Component("modules"),
            llms=Component("llms"),
            blocking_work=Component("blocking_work"),
        )
        try:
            errors = lifecycle.cleanup_failed_assembly()
            assert errors == [
                {
                    "component": "scheduler",
                    "error_type": "ComponentStopDeferred",
                    "error": "returned false",
                }
            ]
            assert order == [
                "scheduler",
                "object_tasks",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
        finally:
            store.close()

    def test_async_failed_assembly_cleanup_records_component_false_and_continues(self) -> None:
        order: list[str] = []

        class Component:
            def __init__(self, name: str, *, stopped: bool = True) -> None:
                self.name = name
                self.stopped = stopped

            async def ashutdown(self) -> bool:
                await asyncio.sleep(0)
                order.append(self.name)
                return self.stopped

        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=Component("substrate"),
        )
        lifecycle.bind_components(
            scheduler=Component("scheduler", stopped=False),
            object_tasks=Component("object_tasks"),
            modules=Component("modules"),
            llms=Component("llms"),
            blocking_work=Component("blocking_work"),
        )

        async def exercise() -> list[dict[str, str]]:
            return await lifecycle.acleanup_failed_assembly()

        try:
            assert asyncio.run(exercise()) == [
                {
                    "component": "scheduler",
                    "error_type": "ComponentStopDeferred",
                    "error": "returned false",
                }
            ]
            assert order == [
                "scheduler",
                "object_tasks",
                "modules",
                "llms",
                "blocking_work",
                "substrate",
            ]
        finally:
            store.close()

    def test_sync_open_fails_fast_inside_running_loop_before_store_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        opened = False

        def unexpected_open(*_args: object, **_kwargs: object) -> object:
            nonlocal opened
            opened = True
            raise AssertionError("store must not open from sync API inside a loop")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            unexpected_open,
        )

        async def exercise() -> None:
            with pytest.raises(RuntimeError, match=r"await Runtime\.aopen"):
                Runtime.open("local")

        asyncio.run(exercise())
        assert opened is False

    def test_async_failed_assembly_cleanup_uses_caller_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        finalized_on: list[asyncio.AbstractEventLoop] = []
        db_path = tmp_path / "failed-assembly.sqlite"

        async def exercise() -> None:
            caller_loop = asyncio.get_running_loop()
            caller_future = caller_loop.create_future()
            caller_loop.call_soon(caller_future.set_result, None)

            async def finalizer() -> None:
                finalized_on.append(asyncio.get_running_loop())
                await caller_future

            def fail_after_binding_finalizer(host: Runtime) -> None:
                host.lifecycle.bind_finalizer(finalizer)
                raise RuntimeError("injected late assembly failure")

            monkeypatch.setattr(
                RuntimeBuilder,
                "_recover_runtime_state",
                staticmethod(fail_after_binding_finalizer),
            )
            with pytest.raises(RuntimeError, match="injected late assembly failure"):
                await Runtime.aopen(db_path)
            assert finalized_on == [caller_loop]

        asyncio.run(exercise())
        reopened = SQLiteStore(db_path)
        reopened.close()

    def test_cancelled_async_open_waits_for_failed_assembly_cleanup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "cancelled-failed-assembly.sqlite"
        calls: list[str] = []

        async def exercise() -> None:
            entered = asyncio.Event()
            release = asyncio.Event()

            async def blocking_finalizer() -> None:
                calls.append("blocking-entered")
                entered.set()
                await release.wait()
                calls.append("blocking-finished")

            async def later_finalizer() -> None:
                await asyncio.sleep(0)
                calls.append("later")

            def fail_after_binding_finalizer(host: Runtime) -> None:
                host.lifecycle.bind_finalizer(blocking_finalizer)
                host.lifecycle.bind_finalizer(later_finalizer)
                raise RuntimeError("injected cancellable assembly failure")

            monkeypatch.setattr(
                RuntimeBuilder,
                "_recover_runtime_state",
                staticmethod(fail_after_binding_finalizer),
            )
            opening = asyncio.create_task(Runtime.aopen(db_path))
            await entered.wait()
            opening.cancel()
            release.set()
            with pytest.raises(BaseExceptionGroup) as caught:
                await opening
            assert caught.value.subgroup(asyncio.CancelledError) is not None
            assert caught.value.subgroup(RuntimeError) is not None

        asyncio.run(exercise())
        assert calls == ["blocking-entered", "blocking-finished", "later"]
        reopened = SQLiteStore(db_path)
        reopened.close()

    def test_cancelled_async_open_preserves_incomplete_cleanup_handle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "cancelled-incomplete-assembly.sqlite"

        async def exercise() -> None:
            entered = asyncio.Event()
            release_finalizer = asyncio.Event()

            async def blocking_finalizer() -> None:
                entered.set()
                await release_finalizer.wait()

            def fail_with_deferred_scheduler(host: Runtime) -> None:
                host.scheduler.shutdown = lambda: False  # type: ignore[method-assign]
                host.lifecycle.bind_finalizer(blocking_finalizer)
                raise RuntimeError("injected cancelled incomplete assembly failure")

            monkeypatch.setattr(
                RuntimeBuilder,
                "_recover_runtime_state",
                staticmethod(fail_with_deferred_scheduler),
            )
            opening = asyncio.create_task(Runtime.aopen(db_path))
            await entered.wait()
            opening.cancel()
            release_finalizer.set()
            with pytest.raises(BaseExceptionGroup) as caught:
                await opening
            assert caught.value.subgroup(asyncio.CancelledError) is not None
            handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
            assert len(handles) == 1
            handle = handles[0]
            assert handle.owns_store is True
            assert any(
                item["error_type"] == "ComponentStopDeferred"
                for item in handle.cleanup_errors
            )
            assert handle.partial_runtime is not None
            handle.partial_runtime.scheduler.shutdown = lambda: True  # type: ignore[method-assign]
            await handle.arelease()
            assert handle.released is True

        asyncio.run(exercise())
        reopened = SQLiteStore(db_path)
        reopened.close()

    def test_incomplete_async_assembly_cleanup_keeps_owned_store_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")

        def fail_with_deferred_scheduler(host: Runtime) -> None:
            host.scheduler.shutdown = lambda: False  # type: ignore[method-assign]
            raise RuntimeError("injected assembly failure with live scheduler")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )
        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_with_deferred_scheduler),
        )

        async def exercise() -> None:
            with pytest.raises(BaseExceptionGroup) as caught:
                await Runtime.aopen("ignored")
            assert caught.value.subgroup(RuntimeError) is not None
            handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
            assert len(handles) == 1
            handle = handles[0]
            assert handle.owns_store is True
            assert handle.partial_runtime is not None
            assert handle.cleanup_errors[0]["error_type"] == "ComponentStopDeferred"
            assert store.list_processes() == []
            handle.partial_runtime.scheduler.shutdown = lambda: True  # type: ignore[method-assign]
            await handle.arelease()
            assert handle.released is True

        asyncio.run(exercise())

    def test_incomplete_sync_assembly_cleanup_keeps_owned_store_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")

        def fail_with_deferred_scheduler(host: Runtime) -> None:
            host.scheduler.shutdown = lambda: False  # type: ignore[method-assign]
            raise RuntimeError("injected sync assembly failure with live scheduler")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )
        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_with_deferred_scheduler),
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            Runtime.open("ignored")
        assert caught.value.subgroup(RuntimeError) is not None
        handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
        assert len(handles) == 1
        handle = handles[0]
        assert handle.owns_store is True
        assert handle.partial_runtime is not None
        assert store.list_processes() == []
        handle.partial_runtime.scheduler.shutdown = lambda: True  # type: ignore[method-assign]
        handle.release()
        assert handle.released is True

    def test_failed_assembly_cleanup_handle_retries_finalizer_before_store_close(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TrackingStore(SQLiteStore):
            close_calls = 0

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        store = TrackingStore(":memory:")
        finalizer_calls = 0

        def deferred_finalizer() -> bool:
            nonlocal finalizer_calls
            finalizer_calls += 1
            return finalizer_calls != 2

        def fail_with_deferred_finalizer(host: Runtime) -> None:
            host.lifecycle.bind_finalizer(deferred_finalizer)
            host.lifecycle.bind_finalizer(deferred_finalizer)
            raise RuntimeError("injected assembly failure with deferred finalizer")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )
        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_with_deferred_finalizer),
        )

        with pytest.raises(BaseExceptionGroup) as caught:
            Runtime.open("ignored")
        handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
        assert len(handles) == 1
        handle = handles[0]
        assert handle.owns_store is True
        assert handle.cleanup_completed is False
        assert handle.released is False
        assert store.close_calls == 0
        assert finalizer_calls == 2
        assert [item["error_type"] for item in handle.cleanup_errors] == [
            "FinalizerDeferred"
        ]

        handle.release()

        assert finalizer_calls == 3
        assert handle.cleanup_completed is True
        assert handle.released is True
        assert store.close_calls == 1

    def test_async_failed_assembly_preserves_caller_owned_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")

        def fail_recovery(_host: Runtime) -> None:
            raise RuntimeError("injected caller-owned assembly failure")

        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_recovery),
        )

        async def exercise() -> None:
            builder = RuntimeBuilder.configured(Runtime)
            with pytest.raises(
                RuntimeError,
                match="injected caller-owned assembly failure",
            ):
                await builder.afrom_store(store)

        try:
            asyncio.run(exercise())
            assert store.list_processes() == []
        finally:
            store.close()

    def test_sync_failed_assembly_releases_caller_store_guard_for_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        builder = RuntimeBuilder.configured(Runtime)
        original_recover = RuntimeBuilder._recover_runtime_state
        attempts = 0

        with store.transaction() as cur:
            cur.execute(
                "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                ("assembly-retry", None, "{}", "test", "1", "1"),
            )

        def fail_once(host: Runtime) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected caller-owned sync assembly failure")
            original_recover(host)

        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_once),
        )

        runtime: Runtime | None = None
        try:
            with pytest.raises(
                RuntimeError,
                match="injected caller-owned sync assembly failure",
            ):
                builder.from_store(store)

            assert store.namespace_exists("assembly-retry")
            assert store._admission_commit_guard is None

            runtime = builder.from_store(store)
            assert store.namespace_exists("assembly-retry")
            assert getattr(store._admission_commit_guard, "__self__", None) is (
                runtime.lifecycle
            )
        finally:
            if runtime is None:
                store.close()
            else:
                runtime.close()

    def test_async_failed_assembly_releases_caller_store_guard_for_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        builder = RuntimeBuilder.configured(Runtime)
        original_recover = RuntimeBuilder._recover_runtime_state
        attempts = 0

        def fail_once(host: Runtime) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected caller-owned async assembly failure")
            original_recover(host)

        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_once),
        )

        async def exercise() -> None:
            with pytest.raises(
                RuntimeError,
                match="injected caller-owned async assembly failure",
            ):
                await builder.afrom_store(store)
            assert store._admission_commit_guard is None

            runtime = await builder.afrom_store(store)
            assert getattr(store._admission_commit_guard, "__self__", None) is (
                runtime.lifecycle
            )
            await runtime.ashutdown()

        asyncio.run(exercise())

    def test_failed_assembly_guard_release_error_is_retryable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class RetryUnbindStore(SQLiteStore):
            unbind_calls = 0

            def unbind_admission_commit_guard(self, guard):
                self.unbind_calls += 1
                if self.unbind_calls == 1:
                    raise RuntimeError("injected admission guard release failure")
                return super().unbind_admission_commit_guard(guard)

        store = RetryUnbindStore(":memory:")
        builder = RuntimeBuilder.configured(Runtime)
        original_recover = RuntimeBuilder._recover_runtime_state
        attempts = 0

        def fail_once(host: Runtime) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected assembly failure before guard release")
            original_recover(host)

        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_once),
        )

        runtime: Runtime | None = None
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                builder.from_store(store)
            handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
            assert len(handles) == 1
            handle = handles[0]
            assert handle.cleanup_completed is False
            assert [item["component"] for item in handle.cleanup_errors] == [
                "admission_commit_guard"
            ]
            assert store._admission_commit_guard is not None

            handle.release()
            assert handle.cleanup_completed is True
            assert handle.released is True
            assert store._admission_commit_guard is None

            runtime = builder.from_store(store)
        finally:
            if runtime is None:
                store.close()
            else:
                runtime.close()

    def test_stale_failed_assembly_cleanup_cannot_unbind_live_runtime_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        builder = RuntimeBuilder.configured(Runtime)
        original_recover = RuntimeBuilder._recover_runtime_state
        attempts = 0

        def fail_once(host: Runtime) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected assembly failure for stale cleanup")
            original_recover(host)

        monkeypatch.setattr(
            RuntimeBuilder,
            "_recover_runtime_state",
            staticmethod(fail_once),
        )

        runtime: Runtime | None = None
        try:
            with pytest.raises(RuntimeError) as caught:
                builder.from_store(store)
            failed_host = RuntimeBuilder._partial_runtime_from_error(caught.value)
            assert failed_host is not None

            runtime = builder.from_store(store)
            live_guard = store._admission_commit_guard
            assert live_guard is not None

            cleanup_result: list[list[dict[str, str]]] = []
            cleanup_thread = threading.Thread(
                target=lambda: cleanup_result.append(
                    failed_host.lifecycle.cleanup_failed_assembly()
                )
            )
            cleanup_thread.start()
            cleanup_thread.join(timeout=3)

            assert not cleanup_thread.is_alive()
            assert cleanup_result == [[]]
            assert store._admission_commit_guard is live_guard
            with pytest.raises(
                RuntimeError,
                match="admission commit guard is already bound",
            ):
                builder.from_store(store)
            assert store._admission_commit_guard is live_guard
        finally:
            if runtime is None:
                store.close()
            else:
                runtime.close()

    def test_async_foundation_failure_closes_substrate_on_caller_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_on: list[asyncio.AbstractEventLoop] = []

        class AsyncSubstrate:
            workspace_display = "."

            async def aclose(self) -> None:
                closed_on.append(asyncio.get_running_loop())

        def fail_before_lifecycle(_host: Runtime) -> None:
            raise RuntimeError("injected pre-lifecycle assembly failure")

        monkeypatch.setattr(
            RuntimeBuilder,
            "_configure_evidence_and_authority",
            staticmethod(fail_before_lifecycle),
        )

        async def exercise() -> None:
            caller_loop = asyncio.get_running_loop()
            with pytest.raises(
                RuntimeError,
                match="injected pre-lifecycle assembly failure",
            ):
                await Runtime.aopen("local", substrate=AsyncSubstrate())
            assert closed_on == [caller_loop]

        asyncio.run(exercise())

    def test_pre_lifecycle_cleanup_handle_retries_substrate_and_store_release(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TrackingStore(SQLiteStore):
            close_calls = 0

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        class DeferredSubstrate:
            workspace_display = "."

            def __init__(self) -> None:
                self.shutdown_calls = 0

            async def ashutdown(self) -> bool:
                self.shutdown_calls += 1
                return self.shutdown_calls > 1

        store = TrackingStore(":memory:")
        substrate = DeferredSubstrate()

        def fail_before_lifecycle(_host: Runtime) -> None:
            raise RuntimeError("injected pre-lifecycle cleanup retry")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )
        monkeypatch.setattr(
            RuntimeBuilder,
            "_configure_evidence_and_authority",
            staticmethod(fail_before_lifecycle),
        )

        async def exercise() -> None:
            with pytest.raises(BaseExceptionGroup) as caught:
                await Runtime.aopen("ignored", substrate=substrate)
            handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
            assert len(handles) == 1
            handle = handles[0]
            assert handle.owns_store is True
            assert handle.partial_runtime is not None
            assert not hasattr(handle.partial_runtime, "lifecycle")
            assert handle.cleanup_errors == (
                {
                    "component": "substrate",
                    "error_type": "ComponentStopDeferred",
                    "error": "returned false",
                },
            )
            assert store.close_calls == 0
            await handle.arelease()
            assert handle.released is True
            assert substrate.shutdown_calls == 2
            assert store.close_calls == 1

        asyncio.run(exercise())

    def test_async_builder_uses_runtime_allocation_hook(self) -> None:
        class HookedRuntime(Runtime):
            @classmethod
            def allocate_unassembled(cls) -> Runtime:
                host = super().allocate_unassembled()
                host.allocated_by_async_hook = True
                return host

        async def exercise() -> None:
            runtime = await RuntimeBuilder.configured(HookedRuntime).aopen("local")
            try:
                assert isinstance(runtime, HookedRuntime)
                assert runtime.allocated_by_async_hook is True
            finally:
                await runtime.ashutdown(
                    actor="test",
                    reason="async-allocation-hook",
                )

        asyncio.run(exercise())

    def test_async_builder_rejects_custom_init_without_allocation_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TrackingStore(SQLiteStore):
            close_calls = 0

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        store = TrackingStore(":memory:")

        class CustomInitRuntime(Runtime):
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.custom_invariant = True
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        open_calls = 0

        def unexpected_open_store(*_args: object, **_kwargs: object) -> SQLiteStore:
            nonlocal open_calls
            open_calls += 1
            return store

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            unexpected_open_store,
        )

        async def exercise() -> None:
            with pytest.raises(
                TypeError,
                match=r"overrides Runtime\.__init__.*override allocate_unassembled",
            ):
                await RuntimeBuilder.configured(CustomInitRuntime).aopen("ignored")

        asyncio.run(exercise())
        assert open_calls == 0
        assert store.close_calls == 0
        store.close()

    def test_async_open_preserves_assembly_and_owned_store_close_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class CloseFailingStore(SQLiteStore):
            fail_close = True

            def close(self) -> None:
                if self.fail_close:
                    raise OSError("injected owned store close failure")
                super().close()

        store = CloseFailingStore(":memory:")

        def fail_before_lifecycle(_host: Runtime) -> None:
            raise RuntimeError("injected assembly failure before store close")

        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )
        monkeypatch.setattr(
            RuntimeBuilder,
            "_configure_evidence_and_authority",
            staticmethod(fail_before_lifecycle),
        )

        async def exercise() -> None:
            with pytest.raises(BaseExceptionGroup) as caught:
                await Runtime.aopen("ignored")
            assert caught.value.subgroup(RuntimeError) is not None
            assert caught.value.subgroup(OSError) is not None
            handles = RuntimeAssemblyCleanupRequired.extract(caught.value)
            assert len(handles) == 1
            handle = handles[0]
            assert handle.owns_store is True
            assert handle.cleanup_completed is True
            store.fail_close = False
            await handle.arelease()
            assert handle.released is True

        asyncio.run(exercise())

    def test_failed_open_does_not_start_object_task_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TrackingStore(SQLiteStore):
            def __init__(self) -> None:
                self.close_calls = 0
                super().__init__(":memory:")

            def close(self) -> None:
                self.close_calls += 1
                super().close()

        existing = {
            thread.ident
            for thread in threading.enumerate()
            if thread.name == "agent-libos-object-tasks"
        }
        store = TrackingStore()
        recovery_reads = 0

        def fail_recovery(
            _store: object,
            *,
            kind: object,
            after: object,
            limit: int,
        ) -> object:
            nonlocal recovery_reads
            recovery_reads += 1
            assert kind is not None
            assert after is None
            assert limit > 0
            raise RuntimeError("injected object task recovery failure")

        monkeypatch.setattr(
            "agent_libos.storage.sql.SQLRuntimeStore.query_object_task_recovery",
            fail_recovery,
        )
        monkeypatch.setattr(
            "agent_libos.runtime.builder.open_store",
            lambda *_args, **_kwargs: store,
        )

        with pytest.raises(RuntimeError, match="injected object task recovery failure"):
            Runtime.open("local")

        assert recovery_reads == 1
        assert store.close_calls == 1
        leaked = [
            thread
            for thread in threading.enumerate()
            if thread.name == "agent-libos-object-tasks"
            and thread.ident not in existing
        ]
        assert leaked == []

    def test_shutdown_is_host_lifecycle_not_process_exit(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'runtime.sqlite')
            runtime = Runtime.open(db)
            pid = runtime.process.spawn(goal='stay runnable')
            result = runtime.shutdown(actor='test', reason='unit-test')
            assert result['ok']
            assert not result['already_shutdown']
            assert runtime.shutdown()['already_shutdown']
            reopened = Runtime.open(db)
            try:
                process = reopened.store.get_process(pid)
                assert process is not None
                assert process.status == ProcessStatus.RUNNABLE
                assert any((record.action == 'runtime.shutdown' for record in reopened.audit.trace()))
                assert any((event.type == EventType.RUNTIME_SHUTDOWN for event in reopened.events.list()))
            finally:
                reopened.shutdown(actor='test', reason='reopen-cleanup')

    def test_async_shutdown_offloads_sync_component_timer_barrier(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        callback_threads: list[int] = []

        class SyncComponent:
            def shutdown(self) -> bool:
                callback_threads.append(threading.get_ident())
                entered.set()
                assert release.wait(timeout=1)
                return True

        async def exercise() -> None:
            caller_thread = threading.get_ident()
            store, lifecycle, _ = _open_test_lifecycle(
                scheduler=SyncComponent(),
            )
            asyncio.get_running_loop().call_later(0.01, release.set)
            result = await lifecycle.ashutdown(
                actor="test",
                reason="sync-component-timer-barrier",
            )
            assert result["ok"] is True
            assert entered.is_set()
            assert callback_threads and callback_threads[0] != caller_thread
            assert lifecycle.closed

        asyncio.run(exercise())

    def test_async_failed_assembly_offloads_sync_finalizer_timer_barrier(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        callback_threads: list[int] = []
        store = SQLiteStore(":memory:")
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=object(),
            events=object(),
            substrate=None,
        )

        def finalizer() -> None:
            callback_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(timeout=1)

        lifecycle.bind_finalizer(finalizer)

        async def exercise() -> None:
            caller_thread = threading.get_ident()
            asyncio.get_running_loop().call_later(0.01, release.set)
            assert await lifecycle.acleanup_failed_assembly() == []
            assert entered.is_set()
            assert callback_threads and callback_threads[0] != caller_thread

        try:
            asyncio.run(exercise())
        finally:
            store.close()
