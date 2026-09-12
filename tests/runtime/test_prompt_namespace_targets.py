"""The v2 memory targets must be executable without alias guessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.prompt import (
    RETAINED_GOAL_CONTEXT_BINDING_KEY,
    _compact_materialized_context_text,
    _memory_tool_result_projection,
    build_user_prompt,
    recover_initial_goal_context,
    retained_goal_context_binding,
    split_cache_optimized_user_prompt,
)
from agent_libos.models import MaterializedContext, ObjectType
from agent_libos.skills import get_builtin_skill_catalog


def _json_lines(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.startswith("{")
    ]


class _CopyVisibleMemoryTargetClient:
    """Copy only the name/namespace supplied by the actual v2 prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.targets: list[dict[str, Any]] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: Any) -> LLMCompletion:
        prompt = str(messages[-1]["content"])
        self.prompts.append(prompt)
        records = _json_lines(prompt)
        candidates = records + [
            item["payload"]["result"]
            for item in records
            if isinstance(item.get("payload"), dict)
            and isinstance(item["payload"].get("result"), dict)
        ]
        record = next((item for item in candidates if item.get("name") == "roundtrip-ledger"), None)
        assert record is not None, "The created memory target must be visible in the next prompt"
        target = {key: record[key] for key in ("name", "namespace")}
        self.targets.append(target)
        tool_name = "append_memory_object" if len(self.targets) == 1 else "read_memory_object"
        arguments = dict(target)
        if tool_name == "append_memory_object":
            arguments.update(entry={"step": "verified"}, list_field="entries")
        return LLMCompletion(
            content="",
            tool_calls=[{
                "id": f"namespace-roundtrip-{len(self.targets)}",
                "name": tool_name,
                "arguments": json.dumps(arguments),
            }],
            api="chat",
            model="deterministic-namespace-client",
        )


@pytest.mark.parametrize("namespace_prefix", ["process", "tenant-process"])
def test_v2_displayed_target_drives_real_memory_read_and_append(namespace_prefix: str) -> None:
    config = replace(
        DEFAULT_CONFIG,
        memory=replace(DEFAULT_CONFIG.memory, process_namespace_prefix=namespace_prefix),
        llm=replace(DEFAULT_CONFIG.llm, prompt_layout="cache_optimized_v2"),
    )
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Update the existing ledger")
        package = get_builtin_skill_catalog().get("agent-libos-object-memory")
        assert package is not None
        activated = runtime.llm.dispatch(pid, {
            "action": "activate_skill", "skill_id": package.skill_id,
            "expected_package_sha256": package.package_sha256,
        })
        assert activated["ok"], activated
        created = runtime.llm.dispatch(pid, {
            "action": "create_memory_object",
            "name": "roundtrip-ledger", "type": "summary",
            "payload": {"entries": []}, "immutable": False,
        })
        assert created["ok"], created
        client = _CopyVisibleMemoryTargetClient()
        runtime.llm.client = client

        appended = runtime.run_process_once(pid)
        read = runtime.run_process_once(pid)

        assert appended["ok"], appended
        assert read["ok"], read
        assert client.targets == [
            {"name": "roundtrip-ledger", "namespace": None},
            {"name": "roundtrip-ledger", "namespace": None},
        ]
        obj = runtime.memory.get_object_by_name(pid, "roundtrip-ledger")
        assert obj.payload == {"entries": [{"step": "verified"}]}
        assert runtime.tools.call(pid, "list_memory_namespace", {
            "namespace": client.targets[0]["namespace"],
        }).ok
        for prompt in client.prompts:
            assert '"namespace":"process:self"' not in prompt
            assert '"resource":"object_namespace:process:self"' not in prompt
            stable, dynamic = split_cache_optimized_user_prompt(prompt)
            assert '"namespace":null' in stable
            assert dynamic is not None
    finally:
        runtime.close()


def test_foreign_and_literal_self_namespaces_remain_literal_and_enforced() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Own namespace only")
        foreign_pid = runtime.process.spawn(image="base-agent:v0", goal="Foreign ledger")
        foreign = runtime.memory.resolve_namespace(foreign_pid)
        current = runtime.memory.resolve_namespace(pid)
        runtime.memory.create_object(foreign_pid, ObjectType.SUMMARY, {"entries": []}, name="ledger", immutable=False)
        current_obj = runtime.memory.create_object(pid, ObjectType.SUMMARY, {"entries": []}, name="ledger", immutable=False)

        for namespace in (foreign, "process:self", current + "/child", "process:pid_external_suffix"):
            projected = _memory_tool_result_projection(
                "read_memory_object", {"name": "ledger", "namespace": namespace},
                current_namespace=current,
            )
            assert projected["namespace"] == namespace
            denied = runtime.tools.call(pid, "append_memory_object", {
                **projected, "entry": {"must": "not appear"}, "list_field": "entries",
            })
            assert not denied.ok
        assert runtime.memory.get_object_by_name(pid, "ledger").payload == {"entries": []}
        assert runtime.store.get_object(current_obj.oid).payload == {"entries": []}
        assert any(
            record.action == "tool.call"
            and record.decision.get("ok") is False
            and record.decision.get("error", {}).get("error_type") == "CapabilityDenied"
            for record in runtime.store.list_audit()
        )

        runtime.memory.create_namespace(pid, "process:self")
        literal_obj = runtime.memory.create_object(pid, ObjectType.SUMMARY, {"literal": True}, name="ledger", namespace="process:self")
        literal = runtime.tools.call(pid, "read_memory_object", {"name": "ledger", "namespace": "process:self"})
        assert literal.ok, literal
        assert literal.payload["payload"] == {"literal": True}
        assert runtime.store.get_object(literal_obj.oid).namespace == "process:self"
        assert runtime.memory.get_object_by_name(pid, "ledger").payload == {"entries": []}
    finally:
        runtime.close()


def test_namespace_projection_preserves_business_payload_and_namespace_parent() -> None:
    current = "process:pid_current"
    foreign = "process:pid_foreign"
    payload = {"namespace": current, "resource": f"object_namespace:{foreign}", "run_id": "pid_customer"}
    for namespace, expected in [(current, None), (foreign, foreign), ("process:self", "process:self")]:
        result = _memory_tool_result_projection(
            "read_memory_object", {"name": "ledger", "namespace": namespace, "payload": payload},
            current_namespace=current,
        )
        assert result["namespace"] == expected
        assert result["payload"] == payload
    created_namespace = _memory_tool_result_projection(
        "create_memory_namespace", {"namespace": current + "/child", "parent_namespace": current, "created": True},
        current_namespace=current,
    )
    assert created_namespace["parent_namespace"] == current
    assert created_namespace["namespace"] == current + "/child"


def test_persistent_context_caps_and_events_do_not_invent_namespace_aliases() -> None:
    current = "process:pid_current"
    resources = [f"object_namespace:{value}" for value in (current, "process:pid_foreign", "process:self")]
    source = {"record_type": "object_memory_payload_entry", "entry_index": 0, "entry": {
        "kind": "capabilities_snapshot",
        "capabilities": [{"resource": resource, "rights": ["read"], "effect": "allow"} for resource in resources],
    }}
    compacted = json.loads(_compact_materialized_context_text(json.dumps(source), include_object_ids=False, current_namespace=current))
    rows = compacted["entry"]["capabilities"]
    assert rows[0] == {"resource_type": "object_namespace", "namespace": None, "rights": ["read"], "effect": "allow"}
    assert rows[1]["resource"] == resources[1]
    assert rows[2]["resource"] == resources[2]
    source["entry"] = {"kind": "events_delta", "events": [
        {"type": "capability_granted", "payload": {"resource": resource}}
        for resource in resources
    ]}
    compacted_events = json.loads(_compact_materialized_context_text(
        json.dumps(source), include_object_ids=False, current_namespace=current,
    ))["entry"]["events"]
    assert compacted_events[0]["payload"] == {"resource_type": "object_namespace", "namespace": None}
    assert compacted_events[1]["payload"]["resource"] == resources[1]
    assert compacted_events[2]["payload"]["resource"] == resources[2]


def test_new_and_persisted_old_goal_bindings_still_recover_exact_payloads() -> None:
    goal_oid = "obj_goal"
    source = {
        "record_type": "object_memory_object", "object_oid": goal_oid,
        "namespace": "tenant:pid_test", "name": f"goal:{goal_oid}",
        "type": "goal", "immutable": True, "payload": {"goal": "original"},
    }
    projected = json.loads(_compact_materialized_context_text(json.dumps(source), include_object_ids=False, current_namespace=source["namespace"]))
    binding = retained_goal_context_binding(goal_oid, source, current_namespace=source["namespace"])
    for namespace in (None, "process:self"):
        record = {**projected, "namespace": namespace}
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        selected_binding = binding if namespace is None else {
            "schema_version": 1, "goal_oid": goal_oid,
            "record_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }
        call = SimpleNamespace(messages=[{"role": "user", "content": encoded}], request_options={RETAINED_GOAL_CONTEXT_BINDING_KEY: selected_binding})
        assert recover_initial_goal_context([call], goal_oid) == encoded
        assert recover_initial_goal_context([call], "obj_foreign_goal") is None


@pytest.mark.parametrize("current_namespace", ["process:pid_child", None])
def test_shared_goal_binding_uses_callers_namespace_without_guessing(current_namespace: str | None) -> None:
    goal_oid = "obj_shared_goal"
    source = {
        "record_type": "object_memory_object", "object_oid": goal_oid,
        "namespace": "process:pid_parent", "name": f"goal:{goal_oid}",
        "type": "goal", "immutable": True, "payload": {"goal": "shared original"},
    }
    encoded = _compact_materialized_context_text(
        json.dumps(source), include_object_ids=False, current_namespace=current_namespace,
    )
    binding = retained_goal_context_binding(goal_oid, source, current_namespace=current_namespace)
    call = SimpleNamespace(
        messages=[{"role": "user", "content": encoded}],
        request_options={RETAINED_GOAL_CONTEXT_BINDING_KEY: binding},
    )
    assert json.loads(encoded)["namespace"] == "process:pid_parent"
    assert recover_initial_goal_context([call], goal_oid) == encoded


def test_capability_tool_preserves_namespace_resource_for_permission_and_delegation() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Inspect authority")
        listed = runtime.llm.dispatch(pid, {"action": "list_capabilities"})
        assert listed["ok"], listed
        expected = f"object_namespace:{runtime.memory.resolve_namespace(pid)}"
        assert any(row["resource"] == expected for row in listed["payload"]["capabilities"])
        stored = runtime.store.get_object(listed["result_oid"])
        assert stored.payload["model_projection"] == listed["payload"]
        prompt = build_user_prompt(
            runtime.process.get(pid),
            MaterializedContext(text="", object_refs=[], token_count=0, omitted_objects=[], policy_used="recency_first"),
            [], runtime.store.list_capabilities(pid),
            [{"name": "delegate_capability"}],
            prompt_layout="cache_optimized_v2",
            requestable_capabilities=[{"resource": expected, "rights": ["read"]}],
        )
        assert prompt.count(f'"resource":"{expected}"') >= 2
        assert '"resource":"object_namespace:process:self"' not in prompt
        context = MaterializedContext(
            text=json.dumps({"record_type": "object_memory_payload_entry", "entry_index": 0, "entry": {
                "kind": "capabilities_snapshot",
                "capabilities": [{"resource": expected, "rights": ["read"], "effect": "allow"}],
            }}),
            object_refs=[], token_count=0, omitted_objects=[], policy_used="llm_context_object",
        )
        persistent_prompt = build_user_prompt(
            runtime.process.get(pid), context, [], runtime.store.list_capabilities(pid),
            [{"name": "delegate_capability"}], prompt_layout="cache_optimized_v2",
        )
        stable, dynamic = split_cache_optimized_user_prompt(persistent_prompt)
        assert '"namespace":null' in stable
        assert expected not in stable
        assert dynamic is not None and f'"resource":"{expected}"' in dynamic
    finally:
        runtime.close()
