from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading
import time
from uuid import uuid4

import pytest
from pydantic import BaseModel

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    EventType,
    ObjectHandle,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectPatch,
    ObjectRight,
    ObjectTask,
    ObjectTaskNotificationStatus,
    ObjectTaskStatus,
    ObjectType,
    ProcessMessageKind,
    ProcessStatus,
    RelationType,
)
from agent_libos.models.exceptions import CapabilityDenied, ProcessError, ProcessMessageWaitRequired, ValidationError
from agent_libos.process_execution import bind_process_execution
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, "process:spawn", [CapabilityRight.WRITE], issued_by="test")


def _grant_delegable_clock_sleep(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, "clock:sleep", [CapabilityRight.READ], issued_by="test", delegable=True)


def _inherit_clock_sleep() -> list[dict[str, object]]:
    return [{"resource": "clock:sleep", "rights": [CapabilityRight.READ.value]}]


def _close_fenced_runtime(runtime: Runtime) -> None:
    """Release test resources without weakening the public recovery latch."""

    result = runtime.release_recovery_diagnostics()
    assert result["ok"] is True


class EmptyArgs(BaseModel):
    pass


class SideEffectThenWaitTool(SyncAgentTool[EmptyArgs]):
    name = "side_effect_then_wait"
    description = "Record a side effect before blocking on an owner-watch message."
    args_schema = EmptyArgs
    policy = ToolPolicy(side_effects=True, idempotent=False)

    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, object]:
        runtime = ctx.runtime
        assert runtime is not None
        self.counter["calls"] = self.counter.get("calls", 0) + 1
        runtime.messages.receive(ctx.pid, block=True, channel=runtime.config.object_tasks.owner_watch_channel)
        return {"ready": True}


class SlowSyncSideEffectTool(SyncAgentTool[EmptyArgs]):
    name = "slow_sync_side_effect_for_object_task"
    description = "Slow sync side-effect tool used to exercise cancellation semantics."
    args_schema = EmptyArgs
    policy = ToolPolicy(side_effects=True, idempotent=False)

    def __init__(self, release: threading.Event | None = None) -> None:
        self.release = release
        self.started = threading.Event()
        self.finished = threading.Event()

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, object]:
        self.started.set()
        try:
            if self.release is None:
                time.sleep(0.2)
            else:
                assert self.release.wait(timeout=2.0)
            return {"ok": True}
        finally:
            self.finished.set()


class TestObjectTasks:
    def test_success_publication_failure_rolls_back_before_marking_task_failed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="atomic object task terminal publication",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            original_emit = runtime.events.emit

            def fail_completed_event(event_type: EventType | str, *args: object, **kwargs: object):
                if EventType(event_type) == EventType.OBJECT_TASK_COMPLETED:
                    raise RuntimeError("injected completion event failure")
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, "emit", fail_completed_event)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})

            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.FAILED
            assert completed.result_oid is None
            assert completed.notification.status == ObjectTaskNotificationStatus.DELIVERED
            assert "injected completion event failure" in str(completed.error)
            assert not any(
                event.type == EventType.OBJECT_TASK_COMPLETED
                and event.payload.get("task_id") == task.task_id
                for event in runtime.events.list()
            )
            assert any(
                event.type == EventType.OBJECT_TASK_FAILED
                and event.payload.get("task_id") == task.task_id
                for event in runtime.events.list()
            )
            assert not any(
                record.action == "object_task.completed"
                and record.target == f"object_task:{task.task_id}"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_transient_terminal_notification_failure_is_retried_idempotently(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="retry object task notification",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            notifications = runtime.object_tasks._notifications
            original_notify = notifications.notify
            attempts = 0

            def fail_once(task: ObjectTask, *, phase: str) -> ObjectTask:
                nonlocal attempts
                attempts += 1
                if attempts == 1 and phase == "completed":
                    raise RuntimeError("injected transient notification failure")
                return original_notify(task, phase=phase)

            monkeypatch.setattr(notifications, "notify", fail_once)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})

            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.notification.status == ObjectTaskNotificationStatus.DELIVERED
            assert attempts == 2
            assert len(
                [
                    message
                    for message in runtime.messages.list(pid)
                    if message.correlation_id == task.task_id
                ]
            ) == 1
        finally:
            runtime.close()

    def test_runtime_reopen_marks_terminal_result_unavailable_when_payload_was_runtime_only(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "object-task-result-reopen.sqlite"
        runtime = Runtime.open(database)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="persist ObjectTask history")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=3)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.completed_at is not None
            assert completed.result_oid is not None
            result_oid = str(completed.result_oid)
            completed_at = completed.completed_at
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            recovered = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert recovered.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert recovered.result_oid is None
            assert recovered.completed_at == completed_at
            assert recovered.wait["result_unavailable_after_reopen"] is True
            assert recovered.wait["previous_status"] == ObjectTaskStatus.SUCCEEDED.value
            assert recovered.wait["previous_result_oid"] == result_oid
            assert "runtime reopen" in str(recovered.error)
            assert reopened.store.get_object(result_oid) is None
            assert any(
                entry.action == "object_task.result_unavailable_recovered"
                and task.task_id in entry.decision.get("task_ids", [])
                for entry in reopened.audit.trace()
            )
        finally:
            reopened.close()

        reopened_again = Runtime.open(database)
        try:
            recovered_again = reopened_again.object_tasks.get(task.task_id, actor_pid=pid)
            assert recovered_again.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert recovered_again.result_oid is None
            assert recovered_again.completed_at == completed_at
            assert recovered_again.wait["previous_result_oid"] == result_oid
            assert len(
                [
                    entry
                    for entry in reopened_again.audit.trace()
                    if entry.action == "object_task.result_unavailable_recovered"
                    and task.task_id in entry.decision.get("task_ids", [])
                ]
            ) == 1
        finally:
            reopened_again.close()

    def test_cross_actor_task_view_and_cancel_consume_one_shot_authority(self) -> None:
        runtime = Runtime.open("local")
        try:
            creator = runtime.process.spawn(image="base-agent:v0", goal="task creator")
            actor = runtime.process.spawn(image="base-agent:v0", goal="task controller")
            _grant_process_spawn(runtime, creator)
            owner = _owner(runtime, creator)
            tasks = [
                runtime.object_tasks.start(
                    creator,
                    owner,
                    "receive_process_messages",
                    {"channel": f"never-{index}"},
                )
                for index in range(2)
            ]
            for task in tasks:
                waiting = runtime.object_tasks.wait(task.task_id, actor_pid=creator, timeout=2)
                assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            write_cap = runtime.capability.issue_trusted(
                actor,
                f"object:{owner.oid}",
                [ObjectRight.WRITE],
                issued_by="test",
                uses_remaining=1,
            )
            cancelled = runtime.object_tasks.cancel(tasks[0].task_id, actor_pid=actor)
            assert cancelled.status == ObjectTaskStatus.CANCELLED
            assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.object_tasks.cancel(tasks[1].task_id, actor_pid=actor)

            read_cap = runtime.capability.issue_trusted(
                actor,
                f"object:{owner.oid}",
                [ObjectRight.READ],
                issued_by="test",
                uses_remaining=1,
            )
            assert runtime.object_tasks.get(tasks[1].task_id, actor_pid=actor).task_id == tasks[1].task_id
            assert runtime.store.get_capability(read_cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.object_tasks.get(tasks[1].task_id, actor_pid=actor)
        finally:
            runtime.close()
    def test_object_task_runs_visible_tool_links_result_and_notifies_creator(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="object task")
            _grant_process_spawn(runtime, pid)
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {"name": "owner"},
                metadata=ObjectMetadata(title="owner"),
                immutable=False,
            )

            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.result_oid is not None
            assert completed.notification.status == ObjectTaskNotificationStatus.DELIVERED
            assert runtime.store.get_object(completed.result_oid) is not None
            assert runtime.store.get_object(completed.result_oid).owner_kind == ObjectOwnerKind.OBJECT_TASK
            assert runtime.store.get_object(completed.result_oid).owner_id == task.task_id
            links = runtime.store.list_links(src=owner.oid)
            assert [(link.relation, link.dst) for link in links] == [(RelationType.PRODUCED, completed.result_oid)]
            unread = runtime.messages.unread(pid)
            assert unread[-1].sender == f"object_task:{task.task_id}"
            assert unread[-1].channel == runtime.config.object_tasks.notification_channel
            assert unread[-1].payload["status"] == ObjectTaskStatus.SUCCEEDED.value
            assert set(unread[-1].metadata["source_oids"]) == {owner.oid, completed.result_oid}
            assert {
                ref["oid"] for ref in unread[-1].metadata["data_flow_context"]["source_refs"]
            } == {owner.oid, completed.result_oid}
            lifecycle = [
                event.type
                for event in runtime.events.list(target=owner.oid)
                if event.type
                in {
                    EventType.OBJECT_TASK_STARTED,
                    EventType.OBJECT_TASK_RUNNING,
                    EventType.OBJECT_TASK_COMPLETED,
                }
            ]
            assert lifecycle == [
                EventType.OBJECT_TASK_STARTED,
                EventType.OBJECT_TASK_RUNNING,
                EventType.OBJECT_TASK_COMPLETED,
            ]
        finally:
            runtime.close()

    def test_object_task_notification_respects_recipient_identity_domain(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                authority_manifest={
                    "data_flow_policy": {
                        "schema_version": 1,
                        "allowed_tenants": ["tenant-a"],
                        "allowed_principals": [],
                    }
                },
            )
            _grant_process_spawn(runtime, parent)
            recipient = runtime.process.spawn_child(
                parent,
                "restricted notification recipient",
                authority_manifest={
                    "data_flow_policy": {
                        "schema_version": 1,
                        "allowed_tenants": [],
                        "allowed_principals": [],
                    }
                },
            )
            owner = runtime.memory.create_object(
                parent,
                ObjectType.ARTIFACT,
                {"name": "tenant owner"},
                metadata=ObjectMetadata(
                    title="tenant owner",
                    sensitivity="secret",
                    tenant="tenant-a",
                ),
                immutable=False,
            )
            grant_cap = runtime.capability.issue_trusted(
                subject=parent,
                resource="object:*",
                rights=[ObjectRight.GRANT.value],
                issued_by="test",
                uses_remaining=1,
            )

            task = runtime.object_tasks.start(
                parent,
                owner,
                "get_working_directory",
                {},
                notify_pid=recipient,
                grant_result_to_notify=True,
            )
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.result_oid is not None
            assert completed.notification.status == ObjectTaskNotificationStatus.FAILED
            assert "data_flow_policy" in (completed.notification.error or "")
            assert runtime.messages.unread(recipient) == []
            with pytest.raises(CapabilityDenied):
                runtime.memory.handle_for_oid(
                    recipient,
                    completed.result_oid,
                    required_rights={ObjectRight.READ.value},
                )
            assert runtime.store.get_capability(grant_cap.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    def test_object_task_notify_result_rolls_back_grant_when_notification_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            recipient = runtime.spawn_child_process(parent, "notify recipient")
            owner = _owner(runtime, parent)
            grant_cap = runtime.capability.issue_trusted(
                subject=parent,
                resource="object:*",
                rights=[ObjectRight.GRANT.value],
                issued_by="test",
                uses_remaining=1,
            )

            def fail_notification(*_args: object, **_kwargs: object) -> object:
                raise ProcessError("injected object task notification failure")

            monkeypatch.setattr(runtime.messages, "post", fail_notification)
            task = runtime.object_tasks.start(
                parent,
                owner,
                "get_working_directory",
                {},
                notify_pid=recipient,
                grant_result_to_notify=True,
            )

            completed = runtime.object_tasks.wait(
                task.task_id,
                actor_pid=parent,
                timeout=2,
            )

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.result_oid is not None
            assert completed.notification.status == ObjectTaskNotificationStatus.FAILED
            assert "injected object task notification failure" in (
                completed.notification.error or ""
            )
            with pytest.raises(CapabilityDenied):
                runtime.memory.handle_for_oid(
                    recipient,
                    completed.result_oid,
                    required_rights={ObjectRight.READ.value},
                )
            assert runtime.store.get_capability(grant_cap.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    def test_object_task_wait_includes_terminal_notification_delivery(self) -> None:
        runtime = Runtime.open("local")
        release_notification = threading.Event()
        notification_started = threading.Event()
        original_post = runtime.messages.post

        def delayed_post(*args: object, **kwargs: object) -> object:
            if (
                str(kwargs.get("sender") or "").startswith("object_task:")
                and kwargs.get("channel") == runtime.config.object_tasks.notification_channel
            ):
                notification_started.set()
                if not release_notification.wait(timeout=2):
                    raise TimeoutError("timed out waiting to release object task notification")
            return original_post(*args, **kwargs)

        runtime.messages.post = delayed_post  # type: ignore[method-assign]
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="object task wait notification")
            _grant_process_spawn(runtime, pid)
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {"name": "owner"},
                metadata=ObjectMetadata(title="owner"),
                immutable=False,
            )
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})

            with ThreadPoolExecutor(max_workers=1) as executor:
                waiter = executor.submit(runtime.object_tasks.wait, task.task_id, actor_pid=pid, timeout=2)
                assert notification_started.wait(timeout=2)
                time.sleep(0.05)
                assert not waiter.done()
                release_notification.set()
                completed = waiter.result(timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.notification.status == ObjectTaskNotificationStatus.DELIVERED
        finally:
            release_notification.set()
            runtime.messages.post = original_post  # type: ignore[method-assign]
            runtime.close()

    def test_object_task_cannot_run_tool_outside_creator_tool_table(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="object task")
            owner = _owner(runtime, pid)

            with pytest.raises(ValidationError, match="not in process tool table"):
                runtime.object_tasks.start(pid, owner, "parse_pytest_log", {"log": "FAILED tests/x.py::test_y"})
        finally:
            runtime.close()

    def test_object_task_start_requires_owner_write_and_link_rights(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="object task")
            read_only = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {"name": "immutable"},
                metadata=ObjectMetadata(title="immutable"),
                immutable=True,
            )

            with pytest.raises(CapabilityDenied, match="write"):
                runtime.object_tasks.start(pid, read_only, "get_working_directory", {})
        finally:
            runtime.close()

    def test_object_task_start_requires_process_spawn_authority(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="object task without spawn")
            owner = _owner(runtime, pid)
            before = len(runtime.process.list())

            with pytest.raises(CapabilityDenied, match="process:spawn"):
                runtime.object_tasks.start(pid, owner, "get_working_directory", {})

            assert len(runtime.process.list()) == before
            assert runtime.store.list_object_tasks(include_terminal=True) == []
        finally:
            runtime.close()

    def test_object_task_start_consumes_finite_spawn_attempt_before_runner_creation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="finite object task spawn attempt",
            )
            spawn_cap = runtime.capability.issue_trusted(
                subject=pid,
                resource="process:spawn",
                rights=[CapabilityRight.WRITE.value],
                issued_by="test",
                uses_remaining=1,
            )
            owner = _owner(runtime, pid)
            before_process_ids = {
                process.pid for process in runtime.store.list_processes()
            }

            def fail_before_runner(*_args: object, **_kwargs: object) -> str:
                raise RuntimeError("runner creation failed before publication")

            monkeypatch.setattr(
                runtime.object_tasks._process,
                "spawn_child",
                fail_before_runner,
            )

            with pytest.raises(
                RuntimeError,
                match="runner creation failed before publication",
            ):
                runtime.object_tasks.start(
                    pid,
                    owner,
                    "get_working_directory",
                    {},
                )

            assert {
                process.pid for process in runtime.store.list_processes()
            } == before_process_ids
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(spawn_cap.cap_id).uses_remaining == 0
            assert any(
                record.action == "capability.consume"
                and spawn_cap.cap_id in record.capability_refs
                for record in runtime.audit.trace()
            )
            assert not any(
                event.type == EventType.OBJECT_TASK_STARTED
                for event in runtime.store.list_events()
            )
            assert not any(
                record.action == "object_task.start"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_object_task_runner_does_not_inherit_external_capability_unless_explicit(self, tmp_path: Path) -> None:
        runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(tmp_path))
        try:
            path = f"agent_outputs/object_task_{uuid4().hex}.txt"
            pid = runtime.process.spawn(image="review-agent:v0", goal="write from object task")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            runtime.filesystem.grant_path(pid, path, [CapabilityRight.WRITE], issued_by="test", delegable=True)

            denied = runtime.object_tasks.start(pid, owner, "write_text_file", {"path": path, "content": "denied"})
            denied = runtime.object_tasks.wait(denied.task_id, actor_pid=pid, timeout=2)
            assert denied.status == ObjectTaskStatus.FAILED
            assert not (tmp_path / path).exists()

            allowed = runtime.object_tasks.start(
                pid,
                owner,
                "write_text_file",
                {"path": path, "content": "allowed"},
                inherit_capabilities=[
                    {"resource": runtime.filesystem.resource_for(path), "rights": [CapabilityRight.WRITE.value]}
                ],
            )
            allowed = runtime.object_tasks.wait(allowed.task_id, actor_pid=pid, timeout=2)
            assert allowed.status == ObjectTaskStatus.SUCCEEDED
            assert (tmp_path / path).read_text(encoding="utf-8") == "allowed"
        finally:
            runtime.close()

    def test_object_task_runner_is_not_scheduled_by_llm_scheduler(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="scheduler isolation")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "receive_process_messages", {"channel": "never"})
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runner = runtime.process.get(str(waiting.runner_pid))
            runtime.process_transitions.wake(
                runtime.process_transitions.wait_token(runner),
                reason="test object-task scheduler isolation",
            )

            assert str(waiting.runner_pid) not in runtime.scheduler.runnable_pids()
            results = runtime.scheduler.run_until_idle(lambda selected_pid: {"pid": selected_pid}, max_quanta=2)
            assert all(item.get("pid") != waiting.runner_pid for item in results if isinstance(item, dict))
            assert not any(
                record.action == "scheduler.run_quantum" and record.target == f"process:{waiting.runner_pid}"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_object_task_completion_survives_one_time_owner_handle(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="one time owner")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
                capability_id=cap.cap_id,
            )

            task = runtime.object_tasks.start(child, one_time_owner, "get_working_directory", {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=child, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            assert [(link.relation, link.dst) for link in runtime.store.list_links(src=owner.oid)] == [
                (RelationType.PRODUCED, completed.result_oid)
            ]
        finally:
            runtime.close()

    def test_concurrent_object_task_start_reserves_one_time_owner_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="one time owner race parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "one time owner race creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
                capability_id=cap.cap_id,
            )
            original_assert = runtime.object_tasks._assert_owner_rights
            authorized = threading.Barrier(2)
            synchronized_thread = threading.local()

            def synchronized_assert(pid: str, handle: ObjectHandle, rights: set[str]) -> list[object]:
                decisions = original_assert(pid, handle, rights)
                if not getattr(synchronized_thread, "preflight_complete", False):
                    synchronized_thread.preflight_complete = True
                    authorized.wait(timeout=2)
                return decisions

            monkeypatch.setattr(runtime.object_tasks, "_assert_owner_rights", synchronized_assert)

            def start() -> object:
                return runtime.object_tasks.start(child, one_time_owner, "get_working_directory", {})

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(start) for _ in range(2)]
                outcomes: list[object] = []
                errors: list[BaseException] = []
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=2))
                    except BaseException as exc:
                        errors.append(exc)

            assert len(outcomes) == 1
            assert len(errors) == 1
            assert isinstance(errors[0], CapabilityDenied)
            completed = runtime.object_tasks.wait(outcomes[0].task_id, actor_pid=child, timeout=2)  # type: ignore[union-attr]
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            assert len(runtime.store.list_object_tasks(include_terminal=True)) == 1
        finally:
            runtime.close()

    def test_object_task_start_reauthorizes_owner_rights_before_reservation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="owner authority race parent",
            )
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                },
                capability_id=cap.cap_id,
            )
            original_assert = runtime.object_tasks._assert_owner_rights
            deny_injected = False

            def deny_after_preflight(
                pid: str,
                handle: ObjectHandle,
                rights: set[str],
            ) -> list[object]:
                nonlocal deny_injected
                decisions = original_assert(pid, handle, rights)
                if not deny_injected:
                    runtime.capability.issue_trusted(
                        subject=pid,
                        resource=f"object:{handle.oid}",
                        rights=[ObjectRight.LINK.value],
                        issued_by="test.authority-race",
                        effect=CapabilityEffect.DENY,
                    )
                    deny_injected = True
                return decisions

            monkeypatch.setattr(
                runtime.object_tasks,
                "_assert_owner_rights",
                deny_after_preflight,
            )
            before_process_ids = {
                process.pid for process in runtime.store.list_processes()
            }

            with pytest.raises(CapabilityDenied, match="capability lacks link"):
                runtime.object_tasks.start(
                    child,
                    one_time_owner,
                    "get_working_directory",
                    {},
                )

            assert {
                process.pid for process in runtime.store.list_processes()
            } == before_process_ids
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            assert not any(
                event.type == EventType.OBJECT_TASK_STARTED
                for event in runtime.store.list_events()
            )
            assert not any(
                record.action == "object_task.start"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_object_task_start_rejects_revoked_named_handle_despite_broad_allow(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="stale owner handle parent",
            )
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            named_cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test",
                uses_remaining=1,
            )
            runtime.capability.issue_trusted(
                subject=child,
                resource="object:*",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test.broad-authority",
            )
            stale_owner = ObjectHandle(
                oid=owner.oid,
                rights={
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                },
                capability_id=named_cap.cap_id,
            )
            original_assert = runtime.object_tasks._assert_owner_rights
            revoked = False

            def revoke_after_preflight(
                pid: str,
                handle: ObjectHandle,
                rights: set[str],
            ) -> list[object]:
                nonlocal revoked
                decisions = original_assert(pid, handle, rights)
                if not revoked:
                    runtime.capability.revoke(
                        named_cap.cap_id,
                        revoked_by="test.authority-race",
                        reason="invalidate named Object handle after preflight",
                        require_authority=False,
                    )
                    revoked = True
                return decisions

            monkeypatch.setattr(
                runtime.object_tasks,
                "_assert_owner_rights",
                revoke_after_preflight,
            )
            before_process_ids = {
                process.pid for process in runtime.store.list_processes()
            }

            with pytest.raises(CapabilityDenied, match="invalid object capability"):
                runtime.object_tasks.start(
                    child,
                    stale_owner,
                    "get_working_directory",
                    {},
                )

            assert runtime.capability.check(
                child,
                f"object:{owner.oid}",
                ObjectRight.LINK,
            )
            assert {
                process.pid for process in runtime.store.list_processes()
            } == before_process_ids
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert not any(
                event.type == EventType.OBJECT_TASK_STARTED
                for event in runtime.store.list_events()
            )
            assert not any(
                record.action == "object_task.start"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_object_task_start_cleans_runner_before_restoring_one_time_owner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="one time owner rollback parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
                capability_id=cap.cap_id,
            )
            runner_pids: list[str] = []

            def fail_insert(task: object) -> None:
                runner_pids.append(str(getattr(task, "runner_pid")))
                raise RuntimeError("object task insert failed")

            monkeypatch.setattr(runtime.store, "insert_object_task", fail_insert)

            token = runtime.store.claim_execution(
                child,
                owner_id="test.object-task-start-cleanup",
            )
            assert token is not None
            with bind_process_execution(token):
                with pytest.raises(RuntimeError, match="object task insert failed"):
                    runtime.object_tasks.start(
                        child,
                        one_time_owner,
                        "get_working_directory",
                        {},
                    )
            assert runtime.store.complete_execution(
                token,
                status=ProcessStatus.RUNNABLE,
            )

            assert len(runner_pids) == 1
            assert runtime.store.get_process(runner_pids[0]) is None
            assert runtime.capability.list_subject(runner_pids[0], include_inactive=True) == []
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    def test_object_task_start_link_evidence_failure_rolls_back_and_restores_owner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="object task link evidence rollback parent",
            )
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            owner_cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                },
                capability_id=owner_cap.cap_id,
            )
            runner_pids: list[str] = []
            original_spawn_child = runtime.object_tasks._process.spawn_child
            original_link_evidence = runtime.object_tasks._operations.link_evidence

            def capture_runner(*args: object, **kwargs: object) -> str:
                runner_pid = original_spawn_child(*args, **kwargs)
                runner_pids.append(runner_pid)
                return runner_pid

            def fail_object_task_link(
                entity_type: str,
                *args: object,
                **kwargs: object,
            ) -> object:
                if entity_type == "object_task":
                    raise RuntimeError("object task operation link failed")
                return original_link_evidence(entity_type, *args, **kwargs)

            monkeypatch.setattr(
                runtime.object_tasks._process,
                "spawn_child",
                capture_runner,
            )
            monkeypatch.setattr(
                runtime.object_tasks._operations,
                "link_evidence",
                fail_object_task_link,
            )

            with pytest.raises(
                RuntimeError,
                match="object task operation link failed",
            ):
                runtime.object_tasks.start(
                    child,
                    one_time_owner,
                    "get_working_directory",
                    {},
                )

            assert len(runner_pids) == 1
            assert runtime.store.get_process(runner_pids[0]) is None
            assert runtime.capability.list_subject(
                runner_pids[0],
                include_inactive=True,
            ) == []
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(owner_cap.cap_id).uses_remaining == 1
            reservations = [
                row
                for row in runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                )
                if row["cap_id"] == owner_cap.cap_id
            ]
            assert [row["status"] for row in reservations] == ["restored"]
            actions = [record.action for record in runtime.audit.trace()]
            assert "capability.restore_reserved_use" in actions
            assert "object_task.runner_cleanup" in actions
            assert "object_task.start" not in actions
            assert not any(
                event.type == EventType.OBJECT_TASK_STARTED
                for event in runtime.store.list_events()
            )
        finally:
            runtime.close()

    def test_object_task_start_skips_owner_restore_when_runner_cleanup_is_unconfirmed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        runner_pids: list[str] = []
        original_cleanup = runtime.object_tasks._process.cleanup_failed_launch
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="object task incomplete cleanup parent",
            )
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            owner_cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                },
                capability_id=owner_cap.cap_id,
            )
            original_spawn_child = runtime.object_tasks._process.spawn_child
            original_link_evidence = runtime.object_tasks._operations.link_evidence

            def capture_runner(*args: object, **kwargs: object) -> str:
                runner_pid = original_spawn_child(*args, **kwargs)
                runner_pids.append(runner_pid)
                return runner_pid

            def preserve_runner(_runner_pid: str) -> None:
                raise RuntimeError("runner cleanup failed")

            def fail_object_task_link(
                entity_type: str,
                *args: object,
                **kwargs: object,
            ) -> object:
                if entity_type == "object_task":
                    raise RuntimeError("object task publication failed")
                return original_link_evidence(entity_type, *args, **kwargs)

            monkeypatch.setattr(
                runtime.object_tasks._process,
                "spawn_child",
                capture_runner,
            )
            monkeypatch.setattr(
                runtime.object_tasks._process,
                "cleanup_failed_launch",
                preserve_runner,
            )
            monkeypatch.setattr(
                runtime.object_tasks._operations,
                "link_evidence",
                fail_object_task_link,
            )

            with pytest.raises(
                RuntimeError,
                match="object task publication failed",
            ):
                runtime.object_tasks.start(
                    child,
                    one_time_owner,
                    "get_working_directory",
                    {},
                )

            assert len(runner_pids) == 1
            assert runtime.store.get_process(runner_pids[0]) is not None
            assert runtime.capability.list_subject(
                runner_pids[0],
                include_inactive=True,
            )
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(owner_cap.cap_id).uses_remaining == 0
            reservations = [
                row
                for row in runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                )
                if row["cap_id"] == owner_cap.cap_id
            ]
            assert [row["status"] for row in reservations] == ["reserved"]
            actions = [record.action for record in runtime.audit.trace()]
            assert "object_task.runner_cleanup_incomplete" in actions
            assert "object_task.owner_permission_restore_skipped" in actions
            assert "capability.restore_reserved_use" not in actions
            assert "object_task.start" not in actions
            assert not any(
                event.type == EventType.OBJECT_TASK_STARTED
                for event in runtime.store.list_events()
            )
        finally:
            for runner_pid in runner_pids:
                original_cleanup(runner_pid)
            runtime.close()

    def test_object_task_start_rolls_back_task_when_owner_commit_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="one time owner commit rejection parent",
            )
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "object task creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                ],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={
                    ObjectRight.READ.value,
                    ObjectRight.WRITE.value,
                    ObjectRight.LINK.value,
                },
                capability_id=cap.cap_id,
            )
            runner_pids: list[str] = []
            original_spawn_child = runtime.object_tasks._process.spawn_child

            def capture_runner(*args: object, **kwargs: object) -> str:
                runner_pid = original_spawn_child(*args, **kwargs)
                runner_pids.append(runner_pid)
                return runner_pid

            monkeypatch.setattr(
                runtime.object_tasks._process,
                "spawn_child",
                capture_runner,
            )
            monkeypatch.setattr(
                runtime.object_tasks._capabilities,
                "commit_reserved_use",
                lambda *_args, **_kwargs: False,
            )

            with pytest.raises(
                CapabilityDenied,
                match="owner permission reservation could not be committed",
            ):
                runtime.object_tasks.start(
                    child,
                    one_time_owner,
                    "get_working_directory",
                    {},
                )

            assert len(runner_pids) == 1
            assert runtime.store.get_process(runner_pids[0]) is None
            assert runtime.capability.list_subject(
                runner_pids[0],
                include_inactive=True,
            ) == []
            assert runtime.store.list_object_tasks(include_terminal=True) == []
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            actions = [record.action for record in runtime.audit.trace()]
            assert "capability.restore_reserved_use" in actions
            assert "object_task.runner_cleanup" in actions
            assert "object_task.start" not in actions
        finally:
            runtime.close()

    def test_object_task_schedule_failure_terminalizes_task_and_removes_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="schedule failure parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "schedule failure creator")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            cap = runtime.capability.issue_trusted(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value],
                issued_by="test",
                uses_remaining=1,
            )
            one_time_owner = ObjectHandle(
                oid=owner.oid,
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
                capability_id=cap.cap_id,
            )

            def fail_schedule(_task_id: str) -> object:
                raise RuntimeError("executor rejected object task")

            monkeypatch.setattr(runtime.object_tasks, "_schedule_task_locked", fail_schedule)

            with pytest.raises(RuntimeError, match="executor rejected object task"):
                runtime.object_tasks.start(child, one_time_owner, "get_working_directory", {})

            tasks = runtime.store.list_object_tasks(include_terminal=True)
            assert len(tasks) == 1
            assert tasks[0].status == ObjectTaskStatus.FAILED
            assert runtime.store.get_process(str(tasks[0].runner_pid)) is None
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            assert runtime.store.list_object_tasks(include_terminal=False) == []
        finally:
            runtime.close()

    def test_object_task_result_wiring_failure_removes_result_and_terminalizes_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="result wiring failure")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            result_oids: list[str] = []

            def fail_link(
                _actor: str,
                _src_oid: str,
                _relation: object,
                dst_oid: str,
                **_kwargs: object,
            ) -> object:
                result_oids.append(dst_oid)
                raise RuntimeError("result link failed")

            monkeypatch.setattr(runtime.memory, "link_objects_trusted", fail_link)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
            failed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert failed.status == ObjectTaskStatus.FAILED
            assert "result link failed" in (failed.error or "")
            assert len(result_oids) == 1
            assert runtime.store.get_object(result_oids[0]) is None
            runner = runtime.store.get_process(str(task.runner_pid))
            assert runner is not None
            assert runner.status in runtime.process.TERMINAL_STATUSES
            creator = runtime.process.get(pid)
            assert creator.memory_view is None or all(root.oid != result_oids[0] for root in creator.memory_view.roots)
            assert runtime.store.list_objects_owned_by(ObjectOwnerKind.OBJECT_TASK, task.task_id) == []
        finally:
            runtime.close()

    def test_object_task_success_commit_failure_discards_result_and_fails_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="success commit failure")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            original_update = runtime.store.update_object_task
            failed_success_commit = False

            def fail_success_once(task: object) -> None:
                nonlocal failed_success_commit
                if getattr(task, "status", None) == ObjectTaskStatus.SUCCEEDED and not failed_success_commit:
                    failed_success_commit = True
                    raise RuntimeError("object task success commit failed")
                original_update(task)

            monkeypatch.setattr(runtime.store, "update_object_task", fail_success_once)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
            failed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert failed_success_commit
            assert failed.status == ObjectTaskStatus.FAILED
            assert "object task success commit failed" in (failed.error or "")
            assert runtime.store.list_objects_owned_by(ObjectOwnerKind.OBJECT_TASK, task.task_id) == []
            runner = runtime.store.get_process(str(task.runner_pid))
            assert runner is not None
            assert runner.status in runtime.process.TERMINAL_STATUSES
            creator = runtime.process.get(pid)
            assert creator.memory_view is None or all(
                runtime.store.get_object(root.oid) is not None for root in creator.memory_view.roots
            )
        finally:
            runtime.close()

    def test_object_task_cancel_winning_during_result_wiring_discards_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        release_link = threading.Event()
        link_started = threading.Event()
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="cancel during result wiring")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            original_link = runtime.memory.link_objects_trusted
            result_oids: list[str] = []

            def delayed_link(
                actor: str,
                src_oid: str,
                relation: object,
                dst_oid: str,
                **kwargs: object,
            ) -> None:
                result_oids.append(dst_oid)
                link_started.set()
                assert release_link.wait(timeout=2)
                original_link(actor, src_oid, relation, dst_oid, **kwargs)  # type: ignore[arg-type]

            monkeypatch.setattr(runtime.memory, "link_objects_trusted", delayed_link)
            task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
            assert link_started.wait(timeout=2)

            cancelled = runtime.object_tasks.cancel(task.task_id, actor_pid=pid, reason="cancel won")
            assert cancelled.status == ObjectTaskStatus.CANCELLED
            release_link.set()
            settled = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert settled.status == ObjectTaskStatus.CANCELLED
            assert len(result_oids) == 1
            assert runtime.store.get_object(result_oids[0]) is None
            assert runtime.store.list_objects_owned_by(ObjectOwnerKind.OBJECT_TASK, task.task_id) == []
            creator = runtime.process.get(pid)
            assert creator.memory_view is None or all(root.oid != result_oids[0] for root in creator.memory_view.roots)
        finally:
            release_link.set()
            runtime.close()

    def test_object_task_notification_can_interrupt_and_wake_message_waiter(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "wait for task")
            owner = _owner(runtime, parent)
            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(child, block=True, channel="object-task")
            assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT

            task = runtime.object_tasks.start(
                parent,
                owner,
                "get_working_directory",
                {},
                notify_pid=child,
                notify_kind=ProcessMessageKind.INTERRUPT,
            )
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            message = runtime.messages.unread(child)[0]
            assert message.kind == ProcessMessageKind.INTERRUPT
            assert message.channel == "object-task"
            assert message.payload["task_id"] == task.task_id
        finally:
            runtime.close()

    def test_object_task_records_undelivered_notification_when_target_exits(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            _grant_delegable_clock_sleep(runtime, parent)
            child = runtime.spawn_child_process(parent, "notify me")
            owner = _owner(runtime, parent)
            task = runtime.object_tasks.start(
                parent,
                owner,
                "sleep",
                {"seconds": 0.05},
                notify_pid=child,
                inherit_capabilities=_inherit_clock_sleep(),
            )
            runtime.process.exit(child, message="done")

            completed = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.notification.status == ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL
            assert runtime.messages.unread(child) == []
        finally:
            runtime.close()

    def test_object_task_result_oid_in_notification_does_not_grant_result_authority(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "notify me")
            owner = _owner(runtime, parent)
            task = runtime.object_tasks.start(parent, owner, "get_working_directory", {}, notify_pid=child)
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)
            result_oid = completed.result_oid
            assert result_oid is not None

            with pytest.raises(CapabilityDenied):
                runtime.memory.handle_for_oid(child, result_oid, required_rights={ObjectRight.READ.value})
            parent_handle = runtime.memory.handle_for_oid(parent, result_oid, required_rights={ObjectRight.READ.value})
            assert parent_handle.oid == result_oid
        finally:
            runtime.close()

    def test_object_task_notify_result_consumes_one_time_grant_authority(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            recipient = runtime.spawn_child_process(parent, "notify recipient")
            owner = _owner(runtime, parent)
            grant_cap = runtime.capability.issue_trusted(
                subject=parent,
                resource="object:*",
                rights=[ObjectRight.GRANT.value],
                issued_by="test",
                uses_remaining=1,
            )
            first = runtime.object_tasks.start(
                parent,
                owner,
                "get_working_directory",
                {},
                notify_pid=recipient,
                grant_result_to_notify=True,
            )
            first_completed = runtime.object_tasks.wait(first.task_id, actor_pid=parent, timeout=2)
            assert first_completed.status == ObjectTaskStatus.SUCCEEDED
            assert first_completed.result_oid is not None
            recipient_handle = runtime.memory.handle_for_oid(
                recipient,
                first_completed.result_oid,
                required_rights={ObjectRight.READ.value},
            )
            assert recipient_handle.oid == first_completed.result_oid
            assert runtime.store.get_capability(grant_cap.cap_id).uses_remaining == 0

            second = runtime.object_tasks.start(
                parent,
                owner,
                "get_working_directory",
                {},
                notify_pid=recipient,
                grant_result_to_notify=True,
            )
            second_completed = runtime.object_tasks.wait(second.task_id, actor_pid=parent, timeout=2)

            assert second_completed.status == ObjectTaskStatus.FAILED
            assert "cannot grant object task result" in (second_completed.error or "")
            notify_object_grants = [
                cap
                for cap in runtime.capability.list_subject(recipient)
                if cap.resource.startswith("object:")
            ]
            assert [
                cap.resource
                for cap in notify_object_grants
                if cap.issued_by == f"object_task:{first.task_id}"
            ] == [f"object:{first_completed.result_oid}"]
            assert not any(cap.issued_by == f"object_task:{second.task_id}" for cap in notify_object_grants)
            reserve_record = next(
                record
                for record in runtime.audit.trace()
                if record.action == "capability.reserve_use" and grant_cap.cap_id in record.capability_refs
            )
            assert any(
                record.action == "capability.commit_reserved_use"
                and record.target == f"capability_reservation:{reserve_record.decision['reservation_id']}"
                for record in runtime.audit.trace()
            )
            assert runtime.store.list_objects_owned_by(ObjectOwnerKind.OBJECT_TASK, second.task_id) == []
        finally:
            runtime.close()

    def test_object_task_waiting_message_state_does_not_repeat_tool_call(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="wait task")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert waiting.wait["filters"]["channel"] == "never"
            assert runtime.process.get(str(waiting.runner_pid)).status == ProcessStatus.WAITING_EVENT
            assert len([record for record in runtime.audit.trace() if record.action == "tool.call_waiting_message"]) == 1
        finally:
            runtime.close()

    def test_posted_process_message_resumes_waiting_object_task_runner(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="message resume task")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "resume-me"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runtime.messages.post(
                sender=pid,
                recipient_pid=str(waiting.runner_pid),
                channel="resume-me",
                subject="ready",
                body="continue",
            )
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert any(record.action == "object_task.owner_watch.resume" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_posted_process_message_only_resumes_recipient_object_task_runner(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="message recipient isolation")
            _grant_process_spawn(runtime, pid)
            first_owner = _owner(runtime, pid)
            second_owner = _owner(runtime, pid)
            first = runtime.object_tasks.start(
                pid,
                first_owner,
                "receive_process_messages",
                {"channel": "shared-resume"},
            )
            second = runtime.object_tasks.start(
                pid,
                second_owner,
                "receive_process_messages",
                {"channel": "shared-resume"},
            )
            first_waiting = runtime.object_tasks.wait(first.task_id, actor_pid=pid, timeout=2)
            second_waiting = runtime.object_tasks.wait(second.task_id, actor_pid=pid, timeout=2)
            assert first_waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert second_waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            before_resumes = len([record for record in runtime.audit.trace() if record.action == "object_task.owner_watch.resume"])

            runtime.messages.post(
                sender=pid,
                recipient_pid=str(first_waiting.runner_pid),
                channel="shared-resume",
                subject="ready",
                body="continue",
            )
            first_completed = runtime.object_tasks.wait(first.task_id, actor_pid=pid, timeout=2)
            second_still_waiting = runtime.object_tasks.wait(second.task_id, actor_pid=pid, timeout=0.05)
            after_resumes = len([record for record in runtime.audit.trace() if record.action == "object_task.owner_watch.resume"])

            assert first_completed.status == ObjectTaskStatus.SUCCEEDED
            assert second_still_waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert after_resumes - before_resumes == 1
            assert len([record for record in runtime.audit.trace() if record.action == "tool.call_waiting_message"]) == 2
        finally:
            runtime.close()

    def test_child_process_exit_resumes_waiting_object_task_runner(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="process resume task")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            runner_pid = str(waiting.runner_pid)
            _grant_process_spawn(runtime, runner_pid)
            child_pid = runtime.spawn_child_process(runner_pid, "child waited by object task")
            runtime.tools.configure_process_tools(runner_pid, ["wait_child_process"], assigned_by="test")
            runtime.object_tasks._pending_args[task.task_id] = {"child_pid": child_pid}
            runtime.store.update_object_task(
                replace(
                    waiting,
                    status=ObjectTaskStatus.WAITING_PROCESS,
                    tool="wait_child_process",
                    wait={"child_pid": child_pid},
                )
            )

            runtime.process.exit(child_pid, message="child done")
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert any(record.action == "object_task.process_resume" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_object_task_request_permission_resumes_after_human_decision(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="permission task",
                authority_manifest={
                    "authorized_capabilities": [
                        {
                            "resource": "human:owner",
                            "rights": [CapabilityRight.WRITE.value],
                            "delegable": True,
                        }
                    ],
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:agent_outputs/*",
                                "rights": [CapabilityRight.WRITE.value],
                            }
                        ]
                    },
                },
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            resource = "filesystem:workspace:agent_outputs/object_task_permission.txt"

            task = runtime.object_tasks.start(
                pid,
                owner,
                "request_permission",
                {"resource": resource, "rights": [CapabilityRight.WRITE.value], "reason": "write artifact"},
                inherit_capabilities=[
                    {"resource": "human:owner", "rights": [CapabilityRight.WRITE.value]},
                ],
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_HUMAN
            assert waiting.wait["request_id"] == runtime.human.pending()[0].request_id

            runtime.human.drain_terminal_queue(auto_policy=runtime.capability.ALWAYS_ALLOW)
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            payload = runtime.store.get_object(str(completed.result_oid)).payload
            assert payload["result"]["status"] == "approved"
            assert any(record.action == "object_task.human_resume" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_object_task_owner_watch_update_resumes_waiting_runner(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="watch owner")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": runtime.config.object_tasks.owner_watch_channel},
                owner_watch=True,
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runtime.memory.update_object(pid, owner, ObjectPatch(payload={"name": "owner", "version": 2}))
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.owner_watch.enabled
            assert completed.result_oid is not None
            payload = runtime.store.get_object(completed.result_oid).payload
            message = payload["result"]["messages"][0]
            assert message["channel"] == runtime.config.object_tasks.owner_watch_channel
            assert message["payload"]["type"] == "object_task_owner_change"
            assert message["payload"]["event"] == "updated"
            assert message["payload"]["owner_oid"] == owner.oid
            assert message["payload"]["version"] == 2
            assert "payload" not in message["payload"]
            assert message["metadata"]["source_oids"] == [owner.oid]
            assert [
                ref["oid"] for ref in message["metadata"]["data_flow_context"]["source_refs"]
            ] == [owner.oid]
            assert any(record.action == "object_task.owner_watch.resume" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_object_task_owner_watch_does_not_replay_unsafe_waiting_tool(self) -> None:
        runtime = Runtime.open("local")
        try:
            counter: dict[str, int] = {}
            handle = runtime.tools.register_tool(SideEffectThenWaitTool(counter), registered_by="test", ephemeral=True)
            pid = runtime.process.spawn(image="base-agent:v0", goal="unsafe replay")
            _grant_process_spawn(runtime, pid)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "side_effect_then_wait", {}, owner_watch=True)
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert counter["calls"] == 1

            runtime.memory.update_object(pid, owner, ObjectPatch(payload={"name": "owner", "version": 2}))
            still_waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=0.1)

            assert still_waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert counter["calls"] == 1
            assert runtime.messages.unread(str(task.runner_pid), channel=runtime.config.object_tasks.owner_watch_channel)
            assert any(
                record.action == "object_task.owner_watch.resume_unsafe_replay_skipped"
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_object_task_owner_watch_link_payload_does_not_grant_dst_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        background_revalidation = threading.Event()
        release_revalidation = threading.Event()
        runtime = Runtime.open("local")
        try:
            original_notify_owner_change = runtime.object_tasks._notify_owner_change
            original_execute_task = runtime.object_tasks._execute_task
            gate_background_revalidation = False

            async def controlled_execute_task(task_id: str) -> None:
                if (
                    gate_background_revalidation
                    and not background_revalidation.is_set()
                ):
                    background_revalidation.set()
                    assert release_revalidation.wait(timeout=2), (
                        "background admission revalidation was not released"
                    )
                await original_execute_task(task_id)

            def controlled_notify_owner_change(*args: object, **kwargs: object) -> bool:
                nonlocal gate_background_revalidation
                gate_background_revalidation = True
                notified = original_notify_owner_change(*args, **kwargs)
                assert background_revalidation.wait(timeout=2), (
                    "resumed object task did not revalidate admission"
                )
                return notified

            monkeypatch.setattr(
                runtime.object_tasks,
                "_execute_task",
                controlled_execute_task,
            )
            monkeypatch.setattr(
                runtime.object_tasks,
                "_notify_owner_change",
                controlled_notify_owner_change,
            )
            parent = runtime.process.spawn(image="base-agent:v0", goal="watch link")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "watcher")
            _grant_process_spawn(runtime, child)
            owner = _owner(runtime, parent)
            dst = runtime.memory.create_object(
                parent,
                ObjectType.ARTIFACT,
                {"secret": "dst"},
                metadata=ObjectMetadata(title="dst"),
                immutable=True,
            )
            runtime.capability.grant(
                subject=child,
                resource=f"object:{owner.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value],
                issued_by="test",
            )
            child_owner = runtime.memory.handle_for_oid(
                child,
                owner.oid,
                required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
            )
            task = runtime.object_tasks.start(
                child,
                child_owner,
                "receive_process_messages",
                {"channel": "owner-link-watch"},
                owner_watch={"enabled": True, "events": ["linked"], "channel": "owner-link-watch"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=child, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runtime.memory.link_objects(parent, owner, RelationType.REFERENCES, dst)
            release_revalidation.set()
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=child, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED, completed.error
            payload = runtime.store.get_object(completed.result_oid).payload
            message = payload["result"]["messages"][0]
            assert message["payload"]["event"] == "linked"
            assert message["payload"]["relation"] == RelationType.REFERENCES.value
            assert message["payload"]["dst_oid"] == dst.oid
            with pytest.raises(CapabilityDenied):
                runtime.memory.handle_for_oid(child, dst.oid, required_rights={ObjectRight.READ.value})
        finally:
            release_revalidation.set()
            runtime.close()

    def test_object_task_owner_watch_disabled_does_not_notify_runner(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="watch disabled")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": runtime.config.object_tasks.owner_watch_channel},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runtime.memory.update_object(pid, owner, ObjectPatch(payload={"name": "owner", "version": 2}))

            still_waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=0.05)
            assert still_waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert runtime.messages.unread(str(task.runner_pid), channel=runtime.config.object_tasks.owner_watch_channel) == []
        finally:
            runtime.close()

    def test_object_task_owner_watch_terminal_task_is_not_notified(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="watch terminal")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "get_working_directory",
                {},
                owner_watch=True,
            )
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            before = len(runtime.store.list_process_messages(str(completed.runner_pid)))

            runtime.memory.update_object(pid, owner, ObjectPatch(payload={"name": "owner", "version": 2}))

            after = len(runtime.store.list_process_messages(str(completed.runner_pid)))
            assert after == before
        finally:
            runtime.close()

    def test_object_task_pins_owner_object_after_creator_exit_until_task_completes(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="owner exits")
            _grant_process_spawn(runtime, pid)
            _grant_delegable_clock_sleep(runtime, pid)
            owner = _owner(runtime, pid)
            result = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {"kept": True},
                metadata=ObjectMetadata(title="result"),
            )
            task = runtime.object_tasks.start(
                pid,
                owner,
                "sleep",
                {"seconds": 0.05},
                inherit_capabilities=_inherit_clock_sleep(),
            )
            runtime.process.exit(pid, result=result, message="creator exited")

            assert runtime.store.get_object(owner.oid) is not None
            assert runtime.store.get_object(result.oid) is not None
            completed = runtime.object_tasks.wait(task.task_id, timeout=2)

            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert runtime.store.get_object(owner.oid) is None
            assert runtime.store.get_object(result.oid) is not None
            assert runtime.store.get_object(result.oid).owner_kind == ObjectOwnerKind.PROCESS_RESULT
            assert completed.notification.status == ObjectTaskNotificationStatus.UNDELIVERED_TERMINAL
        finally:
            runtime.close()

    def test_object_task_cancel_updates_task_and_runner_process(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="cancel task")
            _grant_process_spawn(runtime, pid)
            _grant_delegable_clock_sleep(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "sleep",
                {"seconds": 1.0},
                inherit_capabilities=_inherit_clock_sleep(),
            )

            cancelled = runtime.object_tasks.cancel(task.task_id, actor_pid=pid, reason="no longer needed")

            assert cancelled.status == ObjectTaskStatus.CANCELLED
            assert runtime.process.get(str(cancelled.runner_pid)).status == ProcessStatus.KILLED
        finally:
            runtime.close()

    def test_object_task_start_bootstraps_runner_under_creator_execution_token(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="scheduler starts object task",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            token = runtime.store.claim_execution(pid, owner_id="test.object-task-start")
            assert token is not None

            with bind_process_execution(token):
                task = runtime.object_tasks.start(
                    pid,
                    owner,
                    "receive_process_messages",
                    {"channel": "never"},
                )

            assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)
            runner = runtime.process.get(str(task.runner_pid))
            assert runner.tool_table == {
                "receive_process_messages": task.tool_id,
            }
            assert any(
                capability.resource == f"object:{owner.oid}"
                for capability in runtime.capability.list_subject(str(task.runner_pid))
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            cancelled = runtime.object_tasks.cancel(
                task.task_id,
                actor_pid=pid,
                reason="test cleanup",
            )
            assert cancelled.status == ObjectTaskStatus.CANCELLED
            assert (
                runtime.process.get(str(task.runner_pid)).status
                == ProcessStatus.KILLED
            )
        finally:
            runtime.close()

    def test_object_task_cancel_terminalizes_runner_under_creator_execution_token(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="scheduler cancels object task",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            token = runtime.store.claim_execution(pid, owner_id="test.object-task-cancel")
            assert token is not None
            with bind_process_execution(token):
                cancelled = runtime.object_tasks.cancel(
                    task.task_id,
                    actor_pid=pid,
                    reason="cancel from scheduler quantum",
                )

            assert cancelled.status == ObjectTaskStatus.CANCELLED
            assert (
                runtime.process.get(str(cancelled.runner_pid)).status
                == ProcessStatus.KILLED
            )
            assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)
            settled = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert settled.status == ObjectTaskStatus.CANCELLED
            assert (
                runtime.process.get(str(settled.runner_pid)).status
                == ProcessStatus.KILLED
            )
        finally:
            runtime.close()

    def test_object_task_cancel_fallback_uses_creator_execution_control_scope(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="scheduler cancellation fallback",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            def fail_signal(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected object task signal failure")

            monkeypatch.setattr(runtime.object_tasks._state._process, "signal", fail_signal)
            token = runtime.store.claim_execution(
                pid,
                owner_id="test.object-task-cancel-fallback",
            )
            assert token is not None
            with bind_process_execution(token):
                cancelled = runtime.object_tasks.cancel(
                    task.task_id,
                    actor_pid=pid,
                    reason="cancel through fallback",
                )

            assert cancelled.status == ObjectTaskStatus.CANCELLED
            runner = runtime.process.get(str(cancelled.runner_pid))
            assert runner.status == ProcessStatus.KILLED
            assert runner.status_message == "cancel through fallback"
            assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)
            settled = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert settled.status == ObjectTaskStatus.CANCELLED
            assert runtime.process.get(str(settled.runner_pid)).status == ProcessStatus.KILLED
        finally:
            runtime.close()

    def test_object_task_cancel_fallback_takes_over_active_runner_execution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="cancel active object task runner",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(
                task.task_id,
                actor_pid=pid,
                timeout=2,
            )
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            runner_pid = str(waiting.runner_pid)
            runtime.object_tasks._state.set_runner_status(
                runner_pid,
                ProcessStatus.RUNNABLE,
                "test active cancellation",
            )
            token = runtime.store.claim_execution(
                runner_pid,
                owner_id="test.object-task-active-runner",
            )
            assert token is not None
            assert runtime.process.get(runner_pid).status == ProcessStatus.RUNNING

            def fail_signal(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected object task signal failure")

            monkeypatch.setattr(
                runtime.object_tasks._state._process,
                "signal",
                fail_signal,
            )
            cancelled = runtime.object_tasks.cancel(
                task.task_id,
                actor_pid=pid,
                reason="cancel active runner through fallback",
            )

            assert cancelled.status == ObjectTaskStatus.CANCELLED
            runner = runtime.process.get(runner_pid)
            assert runner.status == ProcessStatus.KILLED
            assert runner.status_message == "cancel active runner through fallback"
            assert not runtime.store.complete_execution(
                token,
                status=ProcessStatus.RUNNABLE,
            )
        finally:
            runtime.close()

    def test_object_task_cancel_cannot_resurrect_runner_during_running_transition(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        transition_started = threading.Event()
        release_transition = threading.Event()
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="cancel during running transition")
            _grant_process_spawn(runtime, pid)
            _grant_delegable_clock_sleep(runtime, pid)
            owner = _owner(runtime, pid)
            state = runtime.object_tasks._state
            original_set_runner_status = state.set_runner_status

            def delayed_set_runner_status(
                runner_pid: str,
                status: ProcessStatus,
                message: str | None = None,
            ) -> None:
                transition_started.set()
                assert release_transition.wait(timeout=2)
                original_set_runner_status(runner_pid, status, message)

            monkeypatch.setattr(state, "set_runner_status", delayed_set_runner_status)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "sleep",
                {"seconds": 1.0},
                inherit_capabilities=_inherit_clock_sleep(),
            )
            assert transition_started.wait(timeout=2)

            cancel_started = threading.Event()

            def cancel() -> ObjectTask:
                cancel_started.set()
                return runtime.object_tasks.cancel(task.task_id, actor_pid=pid, reason="cancel won")

            with ThreadPoolExecutor(max_workers=1) as executor:
                cancel_future = executor.submit(cancel)
                assert cancel_started.wait(timeout=2)
                assert not cancel_future.done()
                release_transition.set()
                cancelled = cancel_future.result(timeout=2)

            assert cancelled.status == ObjectTaskStatus.CANCELLED
            runner_pid = str(cancelled.runner_pid)
            assert runtime.process.get(runner_pid).status == ProcessStatus.KILLED
            settled = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert settled.status == ObjectTaskStatus.CANCELLED
            assert runtime.process.get(runner_pid).status == ProcessStatus.KILLED
        finally:
            release_transition.set()
            runtime.close()

    def test_object_task_refuses_to_cancel_running_sync_side_effect_tool(self) -> None:
        runtime = Runtime.open("local")
        try:
            handle = runtime.tools.register_tool(SlowSyncSideEffectTool(), registered_by="test", ephemeral=True)
            pid = runtime.process.spawn(image="base-agent:v0", goal="sync cancel")
            _grant_process_spawn(runtime, pid)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "slow_sync_side_effect_for_object_task", {})
            deadline = time.monotonic() + 1.0
            while runtime.object_tasks.get(task.task_id, actor_pid=pid).status != ObjectTaskStatus.RUNNING:
                if time.monotonic() >= deadline:
                    pytest.fail("object task did not enter running state")
                time.sleep(0.01)

            with pytest.raises(ValidationError, match="cannot be safely cancelled"):
                runtime.object_tasks.cancel(task.task_id, actor_pid=pid)

            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
        finally:
            runtime.close()

    def test_object_task_reconciles_external_runner_kill(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="external kill")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "receive_process_messages", {"channel": "never"})
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            runtime.process.signal(str(waiting.runner_pid), "cancel", {"reason": "external kill"})
            refreshed = runtime.object_tasks.get(task.task_id, actor_pid=pid)

            assert refreshed.status == ObjectTaskStatus.CANCELLED
            assert refreshed.error is not None and refreshed.error.startswith("result_oid:")
            reason = runtime.store.get_object(refreshed.error.split(":", 1)[1])
            assert reason is not None
            assert reason.payload == {"reason": "external kill"}
        finally:
            runtime.close()

    def test_object_task_list_limit_applies_after_visibility_filter(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, "child")
            _grant_process_spawn(runtime, child)
            child_owner = _owner(runtime, child)
            child_task = runtime.object_tasks.start(child, child_owner, "get_working_directory", {})
            child_task = runtime.object_tasks.wait(child_task.task_id, actor_pid=child, timeout=2)
            parent_owner = _owner(runtime, parent)
            parent_task = runtime.object_tasks.start(parent, parent_owner, "get_working_directory", {})
            runtime.object_tasks.wait(parent_task.task_id, actor_pid=parent, timeout=2)

            visible = runtime.object_tasks.list(actor_pid=child, limit=1)

            assert [task.task_id for task in visible] == [child_task.task_id]
        finally:
            runtime.close()

    def test_object_task_rejects_invalid_list_limit_and_wait_timeout(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="invalid object task bounds")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "receive_process_messages", {"channel": "never"})

            with pytest.raises(ValidationError, match="list limit"):
                runtime.object_tasks.list(actor_pid=pid, limit=True)
            with pytest.raises(ValidationError, match="list limit"):
                runtime.object_tasks.list(actor_pid=pid, limit=-1)
            with pytest.raises(ValidationError, match="wait timeout"):
                runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=float("nan"))
        finally:
            runtime.close()

    def test_runtime_shutdown_drains_object_task_wait_transition_before_store_close(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="shutdown object task wait")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            runtime.object_tasks.start(pid, owner, "receive_process_messages", {"channel": "never"})

            result = runtime.shutdown(actor="test", reason="object-task-wait-drain")

            assert result["ok"] is True
        finally:
            runtime.close()

    def test_runtime_shutdown_keeps_store_open_when_object_task_executor_is_still_running(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            object_tasks=replace(DEFAULT_CONFIG.object_tasks, shutdown_join_timeout_s=0.5),
        )
        runtime = Runtime.open("local", config=config)
        release = threading.Event()
        slow_tool = SlowSyncSideEffectTool(release)
        try:
            handle = runtime.tools.register_tool(slow_tool, registered_by="test", ephemeral=True)
            pid = runtime.process.spawn(image="base-agent:v0", goal="shutdown slow object task")
            _grant_process_spawn(runtime, pid)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(pid, owner, "slow_sync_side_effect_for_object_task", {})
            deadline = time.monotonic() + 1.0
            while runtime.object_tasks.get(task.task_id, actor_pid=pid).status != ObjectTaskStatus.RUNNING:
                if time.monotonic() >= deadline:
                    pytest.fail("object task did not enter running state")
                time.sleep(0.01)
            assert slow_tool.started.wait(timeout=1.0)

            result = runtime.shutdown(actor="test", reason="object-task-slow-drain")

            assert result["ok"] is False
            assert result["admission_stopped"] is False
            assert "object_tasks_stopped" not in result
            assert runtime.store.get_object_task(task.task_id) is not None

            release.set()
            assert slow_tool.finished.wait(timeout=2.0)
            deadline = time.monotonic() + 2.0
            while runtime.object_tasks._has_active_future(task.task_id):
                if time.monotonic() >= deadline:
                    pytest.fail("object task runner did not finish cleanup")
                time.sleep(0.01)
            retry = runtime.shutdown(actor="test", reason="object-task-slow-drain-retry")
            assert retry["ok"] is True
        finally:
            release.set()
            runtime.close()

    def test_recovery_fence_blocks_background_object_task_failure_publication(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        release = threading.Event()
        slow_tool = SlowSyncSideEffectTool(release)
        try:
            handle = runtime.tools.register_tool(
                slow_tool,
                registered_by="test",
                ephemeral=True,
            )
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="fence background object task",
            )
            _grant_process_spawn(runtime, pid)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "slow_sync_side_effect_for_object_task",
                {},
            )
            assert slow_tool.started.wait(timeout=1.0)
            before_task = runtime.store.get_object_task(task.task_id)
            assert before_task is not None
            assert before_task.status == ObjectTaskStatus.RUNNING
            assert before_task.runner_pid is not None
            before_runner = runtime.store.get_process(str(before_task.runner_pid))
            assert before_runner is not None
            before_runner_identity = (
                before_runner.status,
                before_runner.revision,
                before_runner.state_generation,
                before_runner.execution_generation,
                before_runner.execution_owner_id,
                before_runner.execution_lease_id,
            )
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-object-task-worker-fence",
                )
            with pytest.raises(
                RuntimeError,
                match="recovery diagnostics release requires admission drain",
            ):
                runtime.release_recovery_diagnostics()
            assert runtime.object_tasks._closing is False
            release.set()
            assert slow_tool.finished.wait(timeout=2.0)
            deadline = time.monotonic() + 2.0
            while runtime.object_tasks._has_active_future(task.task_id):
                if time.monotonic() >= deadline:
                    pytest.fail("fenced object task worker did not settle")
                time.sleep(0.01)

            after_task = runtime.store.get_object_task(task.task_id)
            assert after_task == before_task
            after_runner = runtime.store.get_process(str(before_task.runner_pid))
            assert after_runner is not None
            assert (
                after_runner.status,
                after_runner.revision,
                after_runner.state_generation,
                after_runner.execution_generation,
                after_runner.execution_owner_id,
                after_runner.execution_lease_id,
            ) == before_runner_identity
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            release.set()
            _close_fenced_runtime(runtime)

    def test_recovery_release_settles_queued_object_task_without_writes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "queued-object-task-recovery-release.sqlite"
        runtime = Runtime.open(db_path)
        loop_blocked = threading.Event()
        release_loop = threading.Event()
        stop_queued = threading.Event()
        loop_errors: list[dict[str, object]] = []
        release_result: list[dict[str, object]] = []
        release_error: list[BaseException] = []
        release_thread: threading.Thread | None = None
        fenced = False
        released = False
        try:
            loop = runtime.object_tasks._loop
            loop.call_soon_threadsafe(
                loop.set_exception_handler,
                lambda _loop, context: loop_errors.append(dict(context)),
            )

            def block_loop() -> None:
                loop_blocked.set()
                release_loop.wait(timeout=10)

            loop.call_soon_threadsafe(block_loop)
            assert loop_blocked.wait(timeout=2), "object task loop did not block"

            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="queue object task before recovery release",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "get_working_directory",
                {},
            )
            queued = runtime.store.get_object_task(task.task_id)
            assert queued is not None
            assert queued.status == ObjectTaskStatus.QUEUED
            assert queued.runner_pid is not None
            runner = runtime.store.get_process(str(queued.runner_pid))
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()
            bridge_future = runtime.object_tasks._futures[task.task_id]

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-object-task-queued-release",
                )
            fenced = True

            original_call_soon_threadsafe = loop.call_soon_threadsafe

            def observe_stop_barrier(
                callback: Callable[..., object],
                *args: object,
                **kwargs: object,
            ) -> object:
                handle = original_call_soon_threadsafe(
                    callback,
                    *args,
                    **kwargs,
                )
                if getattr(callback, "__name__", "") == "stop_if_quiescent":
                    stop_queued.set()
                return handle

            monkeypatch.setattr(
                loop,
                "call_soon_threadsafe",
                observe_stop_barrier,
            )

            def release_runtime() -> None:
                try:
                    release_result.append(runtime.release_recovery_diagnostics())
                except BaseException as exc:
                    release_error.append(exc)

            release_thread = threading.Thread(target=release_runtime)
            release_thread.start()
            assert stop_queued.wait(timeout=2), "recovery stop barrier was not queued"
            release_loop.set()
            release_thread.join(timeout=5)

            assert not release_thread.is_alive()
            assert release_error == []
            assert release_result and release_result[0]["ok"] is True
            released = True
            assert bridge_future.done()
            if not bridge_future.cancelled():
                assert isinstance(bridge_future.exception(timeout=0), RuntimeError)
            assert not any(
                "destroyed" in str(context.get("message", "")).lower()
                and "pending" in str(context.get("message", "")).lower()
                for context in loop_errors
            )
        finally:
            release_loop.set()
            if release_thread is not None and release_thread.is_alive():
                release_thread.join(timeout=5)
            if not released:
                if fenced:
                    _close_fenced_runtime(runtime)
                else:
                    runtime.close()

        reopened = SQLiteStore(db_path)
        try:
            assert reopened.get_object_task(task.task_id) == queued
            assert reopened.get_process(str(queued.runner_pid)) == runner
            assert reopened.list_audit() == before_audit
            assert reopened.list_events() == before_events
        finally:
            reopened.close()

    def test_recovery_release_retries_after_transient_executor_drain(
        self,
        tmp_path: Path,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            object_tasks=replace(
                DEFAULT_CONFIG.object_tasks,
                shutdown_join_timeout_s=0.01,
            ),
        )
        db_path = tmp_path / "object-task-recovery-release-retry.sqlite"
        runtime = Runtime.open(db_path, config=config)
        worker_started = threading.Event()
        release_worker = threading.Event()
        released = False
        fenced = False
        transient_future: Future[None] | None = None
        try:
            def blocking_transient_work() -> None:
                worker_started.set()
                release_worker.wait(timeout=10)

            async def run_transient_work() -> None:
                await runtime.object_tasks._loop.run_in_executor(
                    None,
                    blocking_transient_work,
                )

            transient_future = asyncio.run_coroutine_threadsafe(
                run_transient_work(),
                runtime.object_tasks._loop,
            )
            assert worker_started.wait(timeout=2), "transient executor work did not start"
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-object-task-release-retry",
                )
            fenced = True

            first = runtime.release_recovery_diagnostics()

            assert first == {
                "ok": False,
                "already_released": False,
                "reason": (
                    "runtime.recovery_required:"
                    "publication-object-task-release-retry"
                ),
                "recovery_required": True,
                "object_tasks_released": False,
            }
            assert runtime.object_tasks._closing is True
            assert runtime.object_tasks._closed is False

            release_worker.set()
            deadline = time.monotonic() + 2
            while runtime.object_tasks._thread.is_alive():
                if time.monotonic() >= deadline:
                    pytest.fail("transient executor did not drain after release")
                time.sleep(0.01)

            retry = runtime.release_recovery_diagnostics()
            assert retry["ok"] is True
            released = True
            assert transient_future.done()
            if not transient_future.cancelled():
                transient_future.exception(timeout=0)
        finally:
            release_worker.set()
            if not released:
                if fenced:
                    _close_fenced_runtime(runtime)
                else:
                    runtime.close()

        reopened = SQLiteStore(db_path)
        try:
            assert reopened.list_audit() == before_audit
            assert reopened.list_events() == before_events
        finally:
            reopened.close()

    def test_close_failed_get_does_not_retry_terminal_notification(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            notifications = runtime.object_tasks._notifications
            original_notify = notifications.notify

            def fail_completion_notification(
                task: ObjectTask,
                *,
                phase: str,
            ) -> ObjectTask:
                if phase == "completed":
                    raise RuntimeError("injected terminal notification failure")
                return original_notify(task, phase=phase)

            monkeypatch.setattr(
                notifications,
                "notify",
                fail_completion_notification,
            )
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="fence terminal notification retry",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "get_working_directory",
                {},
            )
            deadline = time.monotonic() + 2.0
            while True:
                terminal = runtime.store.get_object_task(task.task_id)
                if (
                    terminal is not None
                    and terminal.status == ObjectTaskStatus.SUCCEEDED
                    and not runtime.object_tasks._has_active_future(task.task_id)
                ):
                    break
                if time.monotonic() >= deadline:
                    pytest.fail("object task terminal notification did not settle")
                time.sleep(0.01)
            assert terminal.notification.status == ObjectTaskNotificationStatus.FAILED
            monkeypatch.setattr(notifications, "notify", original_notify)
            before_task = terminal
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-object-task-notification-fence",
                )

            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.object_tasks.get(task.task_id)

            assert runtime.store.get_object_task(task.task_id) == before_task
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            _close_fenced_runtime(runtime)

    def test_close_failed_public_object_task_shutdown_publishes_no_cancellation(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "object-task-recovery-release.sqlite"
        runtime = Runtime.open(db_path)
        fenced = False
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="fence public object task shutdown",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            task = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(
                task.task_id,
                actor_pid=pid,
                timeout=2,
            )
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert waiting.runner_pid is not None
            before_task = runtime.store.get_object_task(task.task_id)
            before_runner = runtime.store.get_process(str(waiting.runner_pid))
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()

            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="publication-object-task-shutdown-fence",
                )
            fenced = True

            with pytest.raises(RuntimeError, match="state=close_failed"):
                runtime.object_tasks.shutdown()
            with pytest.raises(
                RuntimeError,
                match="invalid ObjectTask lifecycle shutdown capability",
            ):
                runtime.object_tasks._shutdown_for_lifecycle(object())
            with pytest.raises(
                RuntimeError,
                match="invalid ObjectTask lifecycle shutdown capability",
            ):
                runtime.object_tasks._abandon_for_recovery(object())

            assert runtime.store.get_object_task(task.task_id) == before_task
            assert runtime.store.get_process(str(waiting.runner_pid)) == before_runner
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            if fenced:
                _close_fenced_runtime(runtime)
            else:
                runtime.close()

        reopened = SQLiteStore(db_path)
        try:
            assert reopened.get_object_task(task.task_id) == before_task
            assert reopened.get_process(str(waiting.runner_pid)) == before_runner
            assert reopened.list_audit() == before_audit
            assert reopened.list_events() == before_events
        finally:
            reopened.close()

    def test_concurrent_object_task_start_rechecks_per_object_limit_at_publish(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            object_tasks=replace(
                DEFAULT_CONFIG.object_tasks,
                max_running_per_object=1,
            ),
        )
        runtime = Runtime.open("local", config=config)
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="concurrent object task capacity",
            )
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            original_require_capacity = (
                runtime.object_tasks._require_concurrency_capacity
            )
            preflight_barrier = threading.Barrier(2)
            synchronized_thread = threading.local()

            def synchronize_preflight(owner_oid: str) -> None:
                original_require_capacity(owner_oid)
                if not getattr(
                    synchronized_thread,
                    "capacity_preflight_complete",
                    False,
                ):
                    synchronized_thread.capacity_preflight_complete = True
                    preflight_barrier.wait(timeout=2)

            monkeypatch.setattr(
                runtime.object_tasks,
                "_require_concurrency_capacity",
                synchronize_preflight,
            )

            def start() -> ObjectTask:
                return runtime.object_tasks.start(
                    pid,
                    owner,
                    "receive_process_messages",
                    {"channel": "never"},
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(start) for _ in range(2)]
                outcomes: list[ObjectTask] = []
                errors: list[BaseException] = []
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=3))
                    except BaseException as exc:
                        errors.append(exc)

            assert len(outcomes) == 1
            assert len(errors) == 1
            assert isinstance(errors[0], ValidationError)
            assert "per-object concurrency limit" in str(errors[0])
            waiting = runtime.object_tasks.wait(
                outcomes[0].task_id,
                actor_pid=pid,
                timeout=2,
            )
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
            assert len(runtime.store.list_object_tasks(include_terminal=False)) == 1
            assert len(runtime.store.list_child_processes(pid)) == 1
        finally:
            runtime.close()

    def test_object_task_per_object_concurrency_limit_is_enforced(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            object_tasks=replace(DEFAULT_CONFIG.object_tasks, max_running_per_object=1),
        )
        runtime = Runtime.open("local", config=config)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="concurrency")
            _grant_process_spawn(runtime, pid)
            owner = _owner(runtime, pid)
            first = runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
            )
            waiting = runtime.object_tasks.wait(first.task_id, actor_pid=pid, timeout=2)
            assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

            with pytest.raises(ValidationError, match="per-object concurrency limit"):
                runtime.object_tasks.start(pid, owner, "get_working_directory", {})
        finally:
            runtime.close()


def _owner(runtime: Runtime, pid: str):
    return runtime.memory.create_object(
        pid,
        ObjectType.ARTIFACT,
        {"name": "owner"},
        metadata=ObjectMetadata(title="owner"),
        immutable=False,
    )
