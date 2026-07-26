from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import threading
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    EventType,
    MergePolicy,
    ObjectHandle,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectRight,
    ObjectTaskNotificationStatus,
    ObjectTaskStatus,
    ObjectType,
    ProcessStatus,
    ProcessExecutionToken,
    ResourceBudget,
)
from agent_libos.models.exceptions import CapabilityDenied, ProcessError, ValidationError
from agent_libos.process_execution import (
    bind_process_execution,
    trusted_post_exec_completion_mutation,
    trusted_terminal_process_mutation,
)
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.process import (
    ProcessExitArgs,
    ProcessExitTool,
    _completion_review_human_messages,
)


def _grant_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )


def _exception_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for nested in error.exceptions:
            leaves.extend(_exception_leaves(nested))
        return leaves
    return [error]


def test_process_message_recipient_denial_is_not_an_existence_oracle() -> None:
    runtime = Runtime.open("local")
    try:
        sender = runtime.process.spawn(image="base-agent:v0", goal="sender")
        unrelated = runtime.process.spawn(image="base-agent:v0", goal="unrelated")

        with pytest.raises(ProcessError) as missing:
            runtime.messages.send_from_process(sender, "pid_missing", body="probe")
        with pytest.raises(ProcessError) as unrelated_error:
            runtime.messages.send_from_process(sender, unrelated, body="probe")

        assert str(missing.value) == str(unrelated_error.value)
    finally:
        runtime.close()


def test_cumulative_exit_ignores_process_forged_human_payload_marker() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="provenance")
        forged = runtime.messages.send_from_process(
            pid,
            pid,
            body="process-authored",
            payload={"source": "human_input"},
        )

        human_messages, acknowledged, unread = (
            _completion_review_human_messages(runtime, pid)
        )

        assert human_messages == []
        assert acknowledged == []
        assert unread == []
        assert runtime.store.get_process_message(forged.message_id) == forged
    finally:
        runtime.close()


def test_process_message_postcommit_object_task_hook_cannot_redefine_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        _grant_spawn(runtime, parent)
        child = runtime.spawn_child_process(parent, "child")

        def fail_hook(_message: object) -> None:
            raise RuntimeError("injected postcommit wake failure")

        monkeypatch.setattr(runtime.object_tasks, "notify_process_message", fail_hook)
        message = runtime.messages.send_from_process(parent, child, body="committed")

        stored = runtime.store.list_process_messages(child)
        assert [item.message_id for item in stored] == [message.message_id]
        assert any(
            record.action == "process.message.object_task_wake_failed"
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_object_task_read_reconciles_a_committed_message_after_hook_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        _grant_spawn(runtime, parent)
        owner = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
            name="message-reconcile.owner",
        )
        task = runtime.object_tasks.start(
            parent,
            owner,
            "receive_process_messages",
            {"channel": "resume-after-hook-failure"},
        )
        waiting = runtime.object_tasks.wait(
            task.task_id,
            actor_pid=parent,
            timeout=2,
        )
        assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
        runner_pid = str(waiting.runner_pid)
        original_hook = runtime.object_tasks.notify_process_message

        def fail_hook(_message: object) -> None:
            raise RuntimeError("injected postcommit wake failure")

        monkeypatch.setattr(
            runtime.object_tasks,
            "notify_process_message",
            fail_hook,
        )
        committed = runtime.messages.send_from_process(
            parent,
            runner_pid,
            channel="resume-after-hook-failure",
            body="resume",
        )
        monkeypatch.setattr(
            runtime.object_tasks,
            "notify_process_message",
            original_hook,
        )

        visible = runtime.object_tasks.get(task.task_id, actor_pid=parent)
        assert visible.status == ObjectTaskStatus.WAITING_MESSAGE
        completed = runtime.object_tasks.wait(
            task.task_id,
            actor_pid=parent,
            timeout=2,
        )

        assert completed.status == ObjectTaskStatus.SUCCEEDED
        stored = runtime.store.get_process_message(committed.message_id)
        assert stored is not None
    finally:
        runtime.close()


def test_process_message_unread_uses_the_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="mailbox")
        observed: list[int | None] = []
        original = runtime.store.list_process_messages

        def tracked(*args: object, **kwargs: Any) -> list[Any]:
            observed.append(kwargs.get("limit"))
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime.store, "list_process_messages", tracked)
        assert runtime.messages.unread(pid) == []
        assert observed[-1] == runtime.config.tools.message_read_limit
    finally:
        runtime.close()


def test_failed_child_preflight_restores_one_time_spawn_authority() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        spawn_cap = runtime.capability.grant_once(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )

        with pytest.raises(CapabilityDenied, match="escapes filesystem"):
            runtime.spawn_child_process(
                parent,
                "invalid child",
                working_directory="../outside",
            )

        assert runtime.store.get_capability(spawn_cap.cap_id).uses_remaining == 1
        assert runtime.process.list_children(parent) == []
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["spawn", "fork"])
def test_successful_child_publication_consumes_one_time_spawn_authority_once(
    operation: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        spawn_cap = runtime.capability.grant_once(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        image_cap = runtime.capability.grant_once(
            parent,
            runtime.image_registry.resource_for("coding-agent:v0"),
            [CapabilityRight.READ],
            issued_by="test",
        )

        child = (
            runtime.spawn_child_process(
                parent,
                "child",
                image="coding-agent:v0",
            )
            if operation == "spawn"
            else runtime.fork_child_process(
                parent,
                "child",
                image="coding-agent:v0",
            )
        )

        assert runtime.process.get(child).parent_pid == parent
        assert runtime.store.get_capability(spawn_cap.cap_id).uses_remaining == 0
        assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 0
        reservations = [
            row
            for row in runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            )
            if row["cap_id"] in {spawn_cap.cap_id, image_cap.cap_id}
        ]
        assert [row["status"] for row in reservations] == [
            "committed",
            "committed",
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["spawn", "fork"])
def test_failed_final_child_publication_rolls_back_charge_and_spawn_authority(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="parent",
            resource_budget=ResourceBudget(max_child_processes=1),
        )
        spawn_cap = runtime.capability.grant_once(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        image_cap = runtime.capability.grant_once(
            parent,
            runtime.image_registry.resource_for("coding-agent:v0"),
            [CapabilityRight.READ],
            issued_by="test",
        )

        def fail_commit(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected final publication failure")

        monkeypatch.setattr(runtime.process, "_commit_launch_publication", fail_commit)
        with pytest.raises(RuntimeError, match="final publication failure"):
            if operation == "spawn":
                runtime.spawn_child_process(
                    parent,
                    "child",
                    image="coding-agent:v0",
                )
            else:
                runtime.fork_child_process(
                    parent,
                    "child",
                    image="coding-agent:v0",
                )

        assert runtime.process.get(parent).resource_usage.child_processes == 0
        assert runtime.store.get_capability(spawn_cap.cap_id).uses_remaining == 1
        assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 1
        assert runtime.process.list_children(parent) == []
    finally:
        runtime.close()


def test_object_task_publication_rejects_deny_committed_after_owner_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="owner")
        _grant_spawn(runtime, parent)
        creator = runtime.spawn_child_process(parent, "creator")
        _grant_spawn(runtime, creator)
        owner = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
            name="raced.owner",
        )
        owner_cap = runtime.capability.issue_trusted(
            subject=creator,
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
        deny_inserted = False

        def insert_deny_after_reservation(*args: object, **kwargs: Any) -> str:
            nonlocal deny_inserted
            runner_pid = original_spawn_child(*args, **kwargs)
            if not deny_inserted:
                runtime.capability.issue_trusted(
                    subject=creator,
                    resource=f"object:{owner.oid}",
                    rights=[ObjectRight.LINK.value],
                    issued_by="test.race",
                    effect=CapabilityEffect.DENY,
                )
                deny_inserted = True
            return runner_pid

        monkeypatch.setattr(
            runtime.object_tasks._process,
            "spawn_child",
            insert_deny_after_reservation,
        )
        process_ids_before = {
            process.pid for process in runtime.store.list_processes()
        }

        with pytest.raises(CapabilityDenied, match="policy now restricts link"):
            runtime.object_tasks.start(
                creator,
                one_time_owner,
                "get_working_directory",
                {},
            )

        assert {
            process.pid for process in runtime.store.list_processes()
        } == process_ids_before
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
        assert not any(
            event.type == EventType.OBJECT_TASK_STARTED
            for event in runtime.store.list_events()
        )
    finally:
        runtime.close()


def test_object_task_runner_rejects_owner_outside_its_receive_domain() -> None:
    runtime = Runtime.open("local")
    try:
        owner_pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="tenant-a owner",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        creator = runtime.process.spawn(
            image="base-agent:v0",
            goal="tenant-b creator",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-b"],
                    "allowed_principals": [],
                }
            },
        )
        _grant_spawn(runtime, creator)
        owner = runtime.memory.create_object(
            owner_pid,
            ObjectType.ARTIFACT,
            {"tenant": "a"},
            metadata=ObjectMetadata(tenant="tenant-a"),
            immutable=False,
            name="tenant-a.owner",
        )
        owner_cap = runtime.capability.issue_trusted(
            subject=creator,
            resource=f"object:{owner.oid}",
            rights=[
                ObjectRight.READ.value,
                ObjectRight.WRITE.value,
                ObjectRight.LINK.value,
            ],
            issued_by="test",
        )
        delegated_owner = ObjectHandle(
            oid=owner.oid,
            rights={
                ObjectRight.READ.value,
                ObjectRight.WRITE.value,
                ObjectRight.LINK.value,
            },
            capability_id=owner_cap.cap_id,
        )
        process_ids_before = {
            process.pid for process in runtime.store.list_processes()
        }

        with pytest.raises(CapabilityDenied, match="data_flow_policy"):
            runtime.object_tasks.start(
                creator,
                delegated_owner,
                "get_working_directory",
                {},
            )

        assert {
            process.pid for process in runtime.store.list_processes()
        } == process_ids_before
        assert runtime.store.list_object_tasks(include_terminal=True) == []
    finally:
        runtime.close()


def test_object_task_result_rechecks_the_creator_receive_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="creator")
        _grant_spawn(runtime, creator)
        owner = runtime.memory.create_object(
            creator,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
            name="creator-domain.owner",
        )
        before_caps = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=creator)
        }
        original_assert = runtime.authority_manifests.assert_data_flow_labels

        def deny_creator_result(pid: str, labels: object) -> None:
            if pid == creator:
                raise CapabilityDenied("creator receive domain changed")
            original_assert(pid, labels)

        monkeypatch.setattr(
            runtime.authority_manifests,
            "assert_data_flow_labels",
            deny_creator_result,
        )

        task = runtime.object_tasks.start(
            creator,
            owner,
            "get_working_directory",
            {},
        )
        failed = runtime.object_tasks.wait(
            task.task_id,
            actor_pid=creator,
            timeout=2,
        )

        assert failed.status == ObjectTaskStatus.FAILED
        assert failed.result_oid is None
        assert {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=creator)
        } == before_caps
        assert runtime.store.list_objects_owned_by(
            ObjectOwnerKind.OBJECT_TASK,
            task.task_id,
        ) == []
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["spawn", "fork"])
def test_child_publication_prevents_parent_exit_during_launch_without_orphan(
    operation: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        _grant_spawn(runtime, parent)

        def terminalize_parent(pid: str, _image_id: str, _publication_id: str) -> None:
            if runtime.process.get(pid).parent_pid == parent:
                runtime.process.exit(parent)

        runtime.process.add_after_spawn_hook(terminalize_parent)
        with pytest.raises(ProcessError, match="descendants are nonterminal"):
            if operation == "spawn":
                runtime.spawn_child_process(parent, "racing child")
            else:
                runtime.fork_child_process(parent, "racing child")

        assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
        assert runtime.process.list_children(parent) == []
    finally:
        runtime.close()


def test_wait_result_delivery_rolls_back_on_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        child = runtime.process.fork(parent, goal="child")
        result = runtime.memory.create_object(
            child,
            ObjectType.SUMMARY,
            {"done": True},
            name="child.result",
        )
        runtime.process.exit(child, result=result)
        before_caps = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=parent)
        }
        original_record = runtime.audit.record

        def fail_wait_audit(*args: object, **kwargs: Any) -> Any:
            action = kwargs.get("action")
            if action == "process.wait":
                raise RuntimeError("injected wait audit failure")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", fail_wait_audit)
        with pytest.raises(RuntimeError, match="wait audit failure"):
            runtime.process.wait(parent, child)

        retained = runtime.store.get_object(result.oid)
        assert retained is not None
        assert retained.owner_kind == ObjectOwnerKind.PROCESS
        assert retained.owner_id == child
        assert {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=parent)
        } == before_caps
        assert result.oid not in {
            handle.oid for handle in runtime.process.get(parent).memory_view.roots
        }

        monkeypatch.setattr(runtime.audit, "record", original_record)
        delivered = runtime.process.wait(parent, child)
        assert delivered.result is not None and delivered.result.oid == result.oid
        assert len(
            [
                capability
                for capability in runtime.store.list_capabilities(subject=parent)
                if capability.resource == f"object:{result.oid}"
                and capability.active
                and not capability.revoked
            ]
        ) == 1
    finally:
        runtime.close()


def test_terminal_parent_cannot_receive_a_child_result() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        child = runtime.process.fork(parent, goal="child")
        result = runtime.memory.create_object(
            child,
            ObjectType.SUMMARY,
            {"done": True},
            name="terminal-parent.child-result",
        )
        runtime.process.exit(child, result=result)
        runtime.process.exit(parent)
        before_caps = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=parent)
        }

        with pytest.raises(ProcessError, match="terminated process"):
            runtime.process.wait(parent, child)

        assert {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=parent)
        } == before_caps
        parent_view = runtime.process.get(parent).memory_view
        assert parent_view is None or result.oid not in {
            handle.oid for handle in parent_view.roots
        }
    finally:
        runtime.close()


def test_cumulative_exit_acquires_ownership_before_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="verify cumulative exit lock order",
        )
        entered: list[str] = []
        original_ownership_locked = runtime.memory.ownership_locked
        original_store_locked = runtime.store.locked

        @contextmanager
        def tracked_ownership_locked():
            entered.append("ownership")
            with original_ownership_locked():
                yield

        @contextmanager
        def tracked_store_locked():
            entered.append("store")
            with original_store_locked():
                yield

        monkeypatch.setattr(
            runtime.memory,
            "ownership_locked",
            tracked_ownership_locked,
        )
        monkeypatch.setattr(runtime.store, "locked", tracked_store_locked)

        output = ProcessExitTool()._run_cumulative_exit(
            ProcessExitArgs(),
            ToolContext(
                trace_id="trace_lock_order",
                call_id="call_lock_order",
                pid=pid,
                runtime=runtime,
            ),
            runtime,
            runtime.images["coding-agent:v0"],
        )

        assert output.status == "completion_review_required"
        assert entered[:2] == ["ownership", "store"]
    finally:
        runtime.close()


def test_merge_precheck_ignores_child_roots_when_updated_objects_are_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
        child = runtime.process.fork(parent, goal="child")
        child_object = runtime.memory.create_object(
            child,
            ObjectType.SUMMARY,
            {"excluded": True},
            name="child.excluded",
        )
        runtime.process.exit(child, result=child_object)

        def reject_spurious_precheck(_pid: str, _oid: str) -> None:
            raise AssertionError("excluded child roots must not be flow-checked")

        monkeypatch.setattr(
            runtime.process,
            "_assert_object_data_flow",
            reject_spurious_precheck,
        )

        result = runtime.process.merge_child_memory(
            parent,
            child,
            policy=MergePolicy(
                include_child_created=False,
                include_updated=False,
            ),
        )

        assert result.merged_oids == []
    finally:
        runtime.close()


def test_terminal_cleanup_records_then_repropagates_control_flow_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    original = runtime.process._notify_object_task_process_terminal
    interrupt = KeyboardInterrupt("injected terminal notifier interrupt")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="terminal cleanup")

        def interrupt_notify(_pid: str) -> None:
            raise interrupt

        monkeypatch.setattr(
            runtime.process,
            "_notify_object_task_process_terminal",
            interrupt_notify,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.exit(pid)

        assert any(leaf is interrupt for leaf in _exception_leaves(caught.value))
        assert runtime.process.terminal_cleanup_state(pid)["state"] == "failed"
    finally:
        monkeypatch.setattr(
            runtime.process,
            "_notify_object_task_process_terminal",
            original,
        )
        runtime.close()


def test_terminal_cleanup_recovery_repropagates_control_flow_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    original = runtime.process._notify_object_task_process_terminal
    interrupt = SystemExit("injected recovery notifier interruption")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="cleanup recovery")

        def interrupt_notify(_pid: str) -> None:
            raise interrupt

        monkeypatch.setattr(
            runtime.process,
            "_notify_object_task_process_terminal",
            interrupt_notify,
        )
        with pytest.raises(BaseExceptionGroup):
            runtime.process.exit(pid)

        monkeypatch.setattr(
            runtime.process,
            "_require_recovery_lease",
            lambda: None,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.recover_terminal_cleanups()

        assert any(leaf is interrupt for leaf in _exception_leaves(caught.value))
        monkeypatch.setattr(
            runtime.process,
            "_notify_object_task_process_terminal",
            original,
        )
        completed = runtime.process.retry_terminal_cleanup(pid)
        assert completed["state"] == "completed"
    finally:
        monkeypatch.setattr(
            runtime.process,
            "_notify_object_task_process_terminal",
            original,
        )
        runtime.close()


def test_scheduler_audit_failure_releases_the_claimed_execution_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="scheduler claim")
        original_record = runtime.audit.record

        def fail_claim_audit(*args: object, **kwargs: Any) -> Any:
            if kwargs.get("action") == "scheduler.run_quantum":
                raise RuntimeError("injected scheduler audit failure")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", fail_claim_audit)
        with pytest.raises(RuntimeError, match="scheduler audit failure"):
            runtime.scheduler._run_quantum(pid, lambda _pid: None)

        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
    finally:
        runtime.close()


def test_scheduler_pid_scope_filters_enumeration_and_claims() -> None:
    runtime = Runtime.open("local")
    try:
        first = runtime.process.spawn(image="base-agent:v0", goal="first")
        second = runtime.process.spawn(image="base-agent:v0", goal="second")
        seen: list[str] = []

        runtime.scheduler.run_until_idle(
            lambda pid: seen.append(pid),
            max_quanta=1,
            pids=[second],
        )

        assert seen == [second]
        assert runtime.process.get(first).status == ProcessStatus.RUNNABLE
        with pytest.raises(ValidationError, match="unique"):
            runtime.scheduler.run_until_idle(
                lambda _pid: None,
                max_quanta=1,
                pids=[first, first],
            )
    finally:
        runtime.close()


def test_spawn_claimed_never_exposes_a_runnable_workflow_window() -> None:
    runtime = Runtime.open("local")
    token = None
    try:
        pid, token = runtime.process.spawn_claimed(
            owner_id="test.workflow",
            image="base-agent:v0",
            goal="claimed workflow",
        )

        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNING
        assert process.execution_owner_id == token.owner_id
        assert process.execution_lease_id == token.lease_id
        assert runtime.store.claim_execution(pid, owner_id="other") is None
    finally:
        if token is not None:
            runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)
        runtime.close()


def test_spawn_claimed_remains_unrunnable_during_concurrent_publication() -> None:
    runtime = Runtime.open("local")
    release_publication = threading.Event()
    hook_entered = threading.Event()
    spawned_pids: list[str] = []
    outcome: list[tuple[str, ProcessExecutionToken]] = []
    errors: list[BaseException] = []
    try:
        def pause_before_publication(
            pid: str,
            _image_id: str,
            _publication_id: str,
        ) -> None:
            spawned_pids.append(pid)
            hook_entered.set()
            assert release_publication.wait(timeout=5)

        runtime.process.add_after_spawn_hook(pause_before_publication)

        def spawn_workflow() -> None:
            try:
                outcome.append(
                    runtime.process.spawn_claimed(
                        owner_id="test.workflow.concurrent",
                        image="base-agent:v0",
                        goal="concurrent claimed workflow",
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=spawn_workflow, daemon=True)
        worker.start()
        assert hook_entered.wait(timeout=5)
        assert len(spawned_pids) == 1
        unpublished = runtime.store.get_process(spawned_pids[0])
        assert unpublished is not None
        assert unpublished.status == ProcessStatus.CREATED
        assert runtime.scheduler.next_runnable() is None

        release_publication.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert errors == []
        assert len(outcome) == 1
        pid, token = outcome[0]
        assert runtime.process.get(pid).status == ProcessStatus.RUNNING
        assert runtime.store.claim_execution(pid, owner_id="scheduler.race") is None
        assert runtime.store.complete_execution(
            token,
            status=ProcessStatus.RUNNABLE,
        )
    finally:
        release_publication.set()
        runtime.close()


@pytest.mark.parametrize("invalid", [True, "1", 1.0, 1.9])
def test_terminal_process_cas_scope_rejects_lossy_numeric_coercion(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="concurrency values"):
        with trusted_terminal_process_mutation(
            "pid_test",
            expected_revision=invalid,  # type: ignore[arg-type]
            expected_generation=0,
            allowed_statuses={ProcessStatus.EXITED},
            execution_token=None,
            reason="test exact CAS typing",
        ):
            pass


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("expected_revision", True),
        ("expected_revision", "1"),
        ("expected_revision", 1.9),
        ("expected_generation", True),
        ("expected_generation", "1"),
        ("expected_generation", 1.9),
    ],
)
def test_post_exec_cas_scope_rejects_lossy_numeric_coercion(
    field: str,
    invalid: object,
) -> None:
    token = ProcessExecutionToken(
        pid="pid_test",
        generation=0,
        owner_id="test.worker",
        lease_id="lease_test",
    )
    values: dict[str, object] = {
        "expected_revision": 0,
        "expected_generation": 1,
    }
    values[field] = invalid

    with bind_process_execution(token):
        with pytest.raises(ValueError, match="concurrency values"):
            with trusted_post_exec_completion_mutation(
                token.pid,
                publication_id="publication_test",
                operation_id="operation_test",
                expected_revision=values["expected_revision"],  # type: ignore[arg-type]
                expected_generation=values["expected_generation"],  # type: ignore[arg-type]
                execution_token=token,
                reason="test exact post-exec CAS typing",
            ):
                pass


@pytest.mark.parametrize("invalid", [True, "1", 1.0, 1.9])
def test_process_transition_rejects_lossy_revision_coercion(
    invalid: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exact CAS")
        before = runtime.process.get(pid)

        with pytest.raises(ValidationError, match="expected_revision"):
            runtime.process.transitions.transition(
                pid,
                ProcessStatus.RUNNABLE,
                expected_revision=invalid,  # type: ignore[arg-type]
            )

        assert runtime.process.get(pid) == before
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid", [True, "1", 1.0, 1.9])
def test_process_transition_rejects_lossy_generation_coercion(
    invalid: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exact CAS")
        before = runtime.process.get(pid)

        with pytest.raises(ValidationError, match="expected_state_generation"):
            runtime.process.transitions.transition(
                pid,
                ProcessStatus.RUNNABLE,
                expected_revision=before.revision,
                expected_state_generation=invalid,  # type: ignore[arg-type]
            )

        assert runtime.process.get(pid) == before
    finally:
        runtime.close()


def test_object_task_notification_bounds_precede_durable_mutation() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="object task bounds")
        _grant_spawn(runtime, pid)
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
            name="task.owner",
        )
        before_children = {process.pid for process in runtime.process.list_children(pid)}

        with pytest.raises(ProcessError, match="channel is too long"):
            runtime.object_tasks.start(
                pid,
                owner,
                "receive_process_messages",
                {"channel": "never"},
                notify_channel="x" * 129,
            )

        assert {process.pid for process in runtime.process.list_children(pid)} == before_children
        task = runtime.object_tasks.start(
            pid,
            owner,
            "receive_process_messages",
            {"channel": "never"},
        )
        waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
        assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE
        overlong = "x" * (runtime.config.tools.message_body_max_chars + 1)
        with pytest.raises(ValidationError, match="message body"):
            runtime.object_tasks.cancel(task.task_id, actor_pid=pid, reason=overlong)
        assert runtime.object_tasks.get(task.task_id, actor_pid=pid).status == ObjectTaskStatus.WAITING_MESSAGE
    finally:
        runtime.close()


def test_actor_scoped_object_task_listing_has_a_zero_fast_path_and_candidate_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded tasks")
        calls: list[int | None] = []
        original = runtime.object_tasks._records.list_object_tasks

        def tracked(*args: object, **kwargs: Any) -> list[Any]:
            calls.append(kwargs.get("limit"))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            runtime.object_tasks._records,
            "list_object_tasks",
            tracked,
        )

        assert runtime.object_tasks.list(actor_pid=pid, limit=0) == []
        assert calls == []
        assert runtime.object_tasks.list(actor_pid=pid, limit=1) == []
        assert calls == [
            runtime.config.runtime.object_task_recovery_page_hard_limit
        ]
    finally:
        runtime.close()


def test_terminal_notification_retry_ignores_stale_none_snapshot() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="notification retry")
        _grant_spawn(runtime, pid)
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
            name="notification.owner",
        )
        task = runtime.object_tasks.start(pid, owner, "get_working_directory", {})
        delivered = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
        assert delivered.notification.status == ObjectTaskNotificationStatus.DELIVERED
        assert delivered.notification.message_id is not None
        stale = replace(
            delivered,
            notification=replace(
                delivered.notification,
                status=ObjectTaskNotificationStatus.NONE,
                message_id=None,
            ),
        )

        retried = runtime.object_tasks._notifications.retry_terminal(stale)

        assert retried.notification.status == ObjectTaskNotificationStatus.DELIVERED
        assert retried.notification.message_id == delivered.notification.message_id
        matching = [
            message
            for message in runtime.store.list_process_messages(pid)
            if message.correlation_id == task.task_id
        ]
        assert len(matching) == 1
    finally:
        runtime.close()
