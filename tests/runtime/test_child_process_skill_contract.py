from __future__ import annotations

from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ObjectMetadata,
    ObjectRight,
    ObjectTaskStatus,
    ObjectType,
    ProcessStatus,
)


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )


@pytest.mark.parametrize(
    "specs",
    [
        [{"resource": "clock:sleep"}],
        [{"resource": "clock:sleep", "rights": []}],
        [{"resource": "clock:sleep", "rights": ["unknown"]}],
        [{"resource": " ", "rights": ["read"]}],
        [{"resource": "clock:sleep", "rights": ["read"], "constraints": []}],
        [{"resource": "clock:sleep", "rights": ["read"], "extra": True}],
        [
            {
                "resource": "clock:sleep",
                "rights": ["read"],
                "constraints": {"scope": "first"},
            },
            {
                "resource": "clock:sleep",
                "rights": ["write"],
                "constraints": {"scope": "second"},
            },
        ],
    ],
)
def test_spawn_child_rejects_ambiguous_inherited_capability_specs(
    specs: list[dict[str, Any]],
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="strict inheritance")
        _grant_process_spawn(runtime, parent)

        result = runtime.tools.call(
            parent,
            "spawn_child_process",
            {"goal": "must not launch", "inherit_capabilities": specs},
        )

        assert not result.ok
        assert runtime.process.list_children(parent) == []
    finally:
        runtime.close()


def test_duplicate_inherited_capability_specs_union_only_identical_constraints() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="safe inheritance union")
        _grant_process_spawn(runtime, parent)
        runtime.capability.grant(
            parent,
            "clock:sleep",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
            delegable=True,
        )

        result = runtime.tools.call(
            parent,
            "spawn_child_process",
            {
                "goal": "inherit a deterministic union",
                "inherit_capabilities": [
                    {"resource": "clock:sleep", "rights": ["write"]},
                    {"resource": "clock:sleep", "rights": ["read"]},
                ],
            },
        )

        assert result.ok, result.error
        assert result.payload["inherited_capabilities"] == [
            {"resource": "clock:sleep", "rights": ["read", "write"]}
        ]
    finally:
        runtime.close()


def test_launch_and_list_expose_effective_roots_and_resource_budget() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="verify launch evidence")
        _grant_process_spawn(runtime, parent)
        selected = runtime.memory.create_object(
            parent,
            ObjectType.EVIDENCE,
            {"selected": True},
            name="selected.parent.root",
        )

        launched = runtime.tools.call(
            parent,
            "fork_child_process",
            {
                "goal": "receive one exact parent root",
                "root_oids": [selected.oid],
                "resource_budget": {"max_tool_calls": 3, "max_child_processes": 0},
            },
        )

        assert launched.ok, launched.error
        child = runtime.process.get(launched.payload["child_pid"])
        expected_budget = {
            name: getattr(child.resource_budget, name)
            for name in child.resource_budget.__dataclass_fields__
        }
        assert launched.payload["selected_parent_root_oids"] == [selected.oid]
        assert set(launched.payload["memory_root_oids"]) == {
            selected.oid,
            child.goal_oid,
        }
        assert launched.payload["resource_budget"] == expected_budget

        listed = runtime.tools.call(parent, "list_child_processes", {})
        assert listed.ok, listed.error
        info = next(item for item in listed.payload["children"] if item["pid"] == child.pid)
        assert info["memory_root_oids"] == launched.payload["memory_root_oids"]
        assert info["resource_budget"] == expected_budget
    finally:
        runtime.close()


def test_failed_parent_exit_terminates_ordinary_descendants_bottom_up() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="fail safely")
        child = runtime.process.fork(parent, goal="must not become orphaned")
        grandchild = runtime.process.fork(child, goal="must terminate before its parent")

        runtime.process.exit(parent, failed=True, message="injected parent failure")

        assert runtime.process.get(parent).status == ProcessStatus.FAILED
        assert runtime.process.get(child).status == ProcessStatus.KILLED
        assert runtime.process.get(grandchild).status == ProcessStatus.KILLED
        assert child not in runtime.scheduler.runnable_pids()
        assert grandchild not in runtime.scheduler.runnable_pids()
    finally:
        runtime.close()


def test_merge_reports_object_task_pinned_child_owned_tail() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="inspect pinned merge tail")
        _grant_process_spawn(runtime, parent)
        child = runtime.process.fork(parent, goal="own a pinned non-root object")
        owner = runtime.memory.create_object(
            child,
            ObjectType.ARTIFACT,
            {"owner": True},
            metadata=ObjectMetadata(title="pinned child owner"),
            immutable=False,
            name="pinned.child.owner",
        )
        child_process = runtime.process.get(child)
        assert child_process.memory_view is not None
        child_process.memory_view.roots = [
            handle
            for handle in child_process.memory_view.roots
            if handle.oid != owner.oid
        ]
        runtime.store.update_process(child_process)

        runtime.capability.grant(
            parent,
            f"object:{owner.oid}",
            [ObjectRight.READ, ObjectRight.WRITE, ObjectRight.LINK],
            issued_by="test",
        )
        parent_owner = runtime.memory.handle_for_oid(
            parent,
            owner.oid,
            required_rights={
                ObjectRight.READ.value,
                ObjectRight.WRITE.value,
                ObjectRight.LINK.value,
            },
            issued_by="test",
        )
        task = runtime.object_tasks.start(
            parent,
            parent_owner,
            "receive_process_messages",
            {"channel": "never"},
        )
        waiting = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)
        assert waiting.status == ObjectTaskStatus.WAITING_MESSAGE

        runtime.process.exit(child)
        first = runtime.tools.call(
            parent,
            "merge_child_memory",
            {"child_pid": child, "include_child_created": False},
        )

        assert first.ok, first.error
        assert owner.oid not in first.payload["adopted_oids"]
        assert owner.oid not in first.payload["released_oids"]
        assert owner.oid in first.payload["retained_child_owned_oids"]
        assert owner.oid in first.payload["pinned_child_owned_oids"]

        runtime.messages.send_from_process(
            parent,
            str(waiting.runner_pid),
            channel="never",
            body="finish",
        )
        completed = runtime.object_tasks.wait(task.task_id, actor_pid=parent, timeout=2)
        assert completed.status == ObjectTaskStatus.SUCCEEDED
        second = runtime.tools.call(
            parent,
            "merge_child_memory",
            {"child_pid": child, "include_child_created": False},
        )
        assert second.ok, second.error
        assert owner.oid in second.payload["released_oids"]
        assert runtime.store.get_object(owner.oid) is None
    finally:
        runtime.close()
