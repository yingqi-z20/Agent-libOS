from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.memory.object_memory import FEEDBACK_STUB_RECORD_TYPE
from agent_libos.models import ObjectMetadata, ObjectType, ProcessStatus, PROMPT_MODE_LIBOS_DEFAULT
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy
from tests.support.fakes import RecordingActionClient


@pytest.mark.parametrize("tool_name", ["read_process_messages", "receive_process_messages", "ask_human"])
def test_acknowledged_input_survives_plan_backlog(tool_name: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Keep every deliverable")
        goal = runtime.process.get(pid).memory_view.roots[0]
        followup = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": tool_name,
            "result": {"messages": [{"body": "PRESERVE_RELEASED_HISTORY"}]},
        })
        plans = [runtime.memory.create_object(pid, ObjectType.PLAN, {
            "pending": f"old plan {index}",
        }) for index in range(24)]
        latest = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "run_shell_command", "result": {"returncode": 0},
        })
        required = [goal, followup, latest]
        budget = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, required),
            budget_tokens=100_000, charge_resources=False,
        ).token_count

        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [goal, followup, *plans, latest]),
            policy="working_set", budget_tokens=budget, charge_resources=False,
        )

        assert context.object_refs == [handle.oid for handle in required]
        assert "PRESERVE_RELEASED_HISTORY" in context.text
        assert context.token_count <= budget
        assert all(entry["transform"] != "compacted" for entry in context.object_manifest)
    finally:
        runtime.close()


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_oversized_latest_result_is_observable_and_retrievable(layout: str) -> None:
    runtime = Runtime.open("local", config=replace(
        DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, prompt_layout=layout),
    ))
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Diagnose the failing test")
        runtime.activate_skill(pid, "agent-libos-object-memory")
        process = runtime.process.get(pid)
        goal = process.memory_view.roots[0]
        result_payload = {
            "tool_name": "run_shell_command", "ok": True,
            "result": {
                "argv": ["python", "-m", "unittest"], "returncode": 1,
                "stdout": "large diagnostic output\n" * 3_000,
                "stderr": "EXACT_FAILURE_DETAIL", "stdout_truncated": True,
                "stderr_truncated": False,
            },
        }
        feedback = runtime.memory.create_object(
            pid, ObjectType.TOOL_RESULT, result_payload, name="test-output",
        )
        process = runtime.process.get(pid)
        process.resource_budget = replace(
            process.resource_budget, max_context_materialization_tokens=2_000,
        )
        process.memory_view = runtime.memory.create_view(pid, [goal, feedback])
        runtime.store.update_process(process)
        client = RecordingActionClient([{
            "action": "read_memory_object", "name": "test-output", "json_pointer": "/result/stderr",
        }])
        runtime.llm.client = client

        observed = runtime.run_process_once(pid)

        prompt = client.user_prompts[0]
        assert FEEDBACK_STUB_RECORD_TYPE in prompt
        assert '"stub_reason":"token_budget"' in prompt
        assert '"returncode":1' in prompt
        assert '"stdout_truncated":true' in prompt
        assert '"stderr_truncated":false' in prompt
        assert "read_memory_object" in prompt
        assert "json_pointer" in prompt
        assert "every newer result is still verbatim" not in prompt
        assert "EXACT_FAILURE_DETAIL" not in prompt
        assert observed["result"]["payload"]["payload"] == "EXACT_FAILURE_DETAIL"
        assert runtime.memory.get_object(pid, feedback).payload == result_payload
        materializations = [row for row in runtime.audit.trace(actor=pid)
                            if row.action == "memory.materialize_context"]
        assert all(row.decision["tokens"] <= 2_000 for row in materializations)
    finally:
        runtime.close()


def test_old_feedback_keeps_completeness_and_memory_page_receipts() -> None:
    runtime = Runtime.open("local", config=replace(
        DEFAULT_CONFIG,
        memory=replace(DEFAULT_CONFIG.memory,
                       working_set_recent_feedback=1, working_set_verbatim_feedback_tokens=0),
    ))
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Retain incomplete evidence")
        feedback = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "read_memory_object", "ok": True,
            "result": {
                "namespace": "process:test", "name": "large-report",
                "json_pointer": "/findings", "page_offset_bytes": 100,
                "page_bytes": 500, "next_cursor": 600, "truncated": True,
                "sha256": "a" * 64, "preview": "not complete",
            },
        })
        newest = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "echo", "result": {"text": "continue"},
        })

        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [feedback, newest]),
            policy="working_set", budget_tokens=2_000, charge_resources=False,
        )

        stub = json.loads(context.text.split("\n\n", 1)[0])
        summary = stub["summary"]
        assert summary["truncated"] is True
        assert summary["json_pointer"] == "/findings"
        assert summary["page_offset_bytes"] == 100
        assert summary["page_bytes"] == 500
        assert summary["next_cursor"] == 600
        assert summary["sha256"] == "a" * 64
        assert "not complete" not in context.text
    finally:
        runtime.close()


@pytest.mark.parametrize("pinned", [False, True], ids=["oversized-result", "human-input"])
def test_pressure_projection_never_reads_a_revoked_handle(pinned: bool) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Respect revocation")
        feedback = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "ask_human" if pinned else "run_shell_command",
            "result": {"path": "PRIVATE_PATH", "stdout": "PRIVATE_OUTPUT" * 3_000},
        })
        view = runtime.memory.create_view(pid, [feedback])
        runtime.capability.revoke(
            feedback.capability_id, revoked_by="test.host", require_authority=False,
        )

        context = runtime.memory.materialize_context(
            pid, view, policy="working_set", budget_tokens=1_000, charge_resources=False,
        )

        assert context.text == ""
        assert context.object_refs == []
        assert context.object_manifest[0]["reason"] == "capability_denied"
        assert "PRIVATE" not in json.dumps(context.object_manifest)
        audit = [row for row in runtime.audit.trace(actor=pid)
                 if row.action == "memory.materialize_context"][-1]
        assert audit.output_refs == []
        assert audit.decision["omitted"] == [feedback.oid]
    finally:
        runtime.close()


@pytest.mark.parametrize("budget", [0, 1, 200])
def test_oversized_stub_cannot_exceed_the_remaining_budget(budget: int) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Observe only what fits")
        feedback = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "run_shell_command",
            "result": {"argv": ["long-argument" * 100] * 6, "stdout": "x" * 30_000},
        })
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [feedback]), policy="working_set",
            budget_tokens=budget, charge_resources=False,
        )
        assert context.token_count <= budget
        assert context.text == ""
        assert context.object_manifest[0]["reason"] == "token_budget"
    finally:
        runtime.close()


@pytest.mark.parametrize("change", ["ack", "body", "id", "partial", "labels"])
def test_message_dedup_preserves_distinct_content_pages_and_labels(change: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Keep distinct inputs")
        message = {
            "message_id": "message-1", "body": "KEEP_FIRST_REQUIREMENT",
            "subject": "customer", "status": "unread", "acked_at": None,
        }
        first = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "read_process_messages", "result": {"messages": [message]},
        })
        newer_message = {**message, "status": "acked", "acked_at": "2026-09-14T00:00:00Z"}
        if change == "body":
            newer_message["body"] = "KEEP_SECOND_REQUIREMENT"
        if change == "id":
            newer_message["message_id"] = "message-2"
        second = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "read_process_messages",
            "result": {"messages": [newer_message], "has_more": change == "partial"},
        }, metadata=ObjectMetadata(tenant="another-tenant") if change == "labels" else None)

        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [first, second]),
            policy="working_set", budget_tokens=20_000, charge_resources=False,
        )

        assert "KEEP_FIRST_REQUIREMENT" in context.text
        if change == "ack":
            assert context.object_refs == [second.oid]
            assert context.object_manifest[0]["reason"] == "superseded"
        else:
            assert context.object_refs == [first.oid, second.oid]
        assert runtime.memory.get_object(pid, first).payload["result"]["messages"][0] == message
    finally:
        runtime.close()


def test_unselected_message_superset_cannot_hide_earlier_input() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Retain the input that fits")
        message = {"message_id": "message-1", "body": "SMALL_REQUIRED_INPUT", "status": "acked"}
        first = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "read_process_messages", "result": {"messages": [message]},
        })
        second = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "read_process_messages", "result": {"messages": [message, {
                "message_id": "message-2", "body": "too large " * 5_000, "status": "acked",
            }]},
        })

        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [first, second]),
            policy="working_set", budget_tokens=1_000, charge_resources=False,
        )

        assert context.object_refs == [first.oid]
        assert "SMALL_REQUIRED_INPUT" in context.text
        assert context.object_manifest[1]["reason"] == "token_budget"
        assert FEEDBACK_STUB_RECORD_TYPE not in context.text
    finally:
        runtime.close()


@pytest.mark.parametrize("inline_limit,truncated", [(512, False), (0, False), (10, False), (512, True)])
def test_focused_read_value_survives_an_oversized_result_wrapper(
    inline_limit: int, truncated: bool,
) -> None:
    runtime = Runtime.open("local", config=replace(DEFAULT_CONFIG, memory=replace(
        DEFAULT_CONFIG.memory, working_set_inline_read_payload_chars=inline_limit,
    )))
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Use the recovered value")
        value = {"diagnostic": "EXACT_RECOVERED_VALUE", "count": 0, "pending": None}
        payload = {
            "tool_name": "read_memory_object",
            "metadata": {"large_annotation": "metadata " * 2_000},
            "result": {"name": "diagnostic", "json_pointer": "/result/stderr",
                       "representation": "json_value", "truncated": truncated, "payload": value},
        }
        feedback = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, payload)

        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [feedback]), policy="working_set",
            budget_tokens=1_000, charge_resources=False,
        )

        record = json.loads(context.text)
        assert record["record_type"] == FEEDBACK_STUB_RECORD_TYPE
        assert record["stub_reason"] == "token_budget"
        if inline_limit == 512 and not truncated:
            assert record["retrieved_payload"] == value
        else:
            assert "retrieved_payload" not in record
            assert "EXACT_RECOVERED_VALUE" not in context.text
        assert context.token_count <= 1_000
        assert runtime.memory.get_object(pid, feedback).payload == payload
    finally:
        runtime.close()


class _DiagnosticArgs(BaseModel):
    sequence: int


class _LargeDiagnostic(SyncAgentTool[_DiagnosticArgs]):
    name = "large_diagnostic"
    description = "Collect one deterministic diagnostic stream."
    args_schema = _DiagnosticArgs
    policy = ToolPolicy(side_effects=False, idempotent=True)

    def __init__(self) -> None:
        self.observed: list[int] = []

    def run(self, args: _DiagnosticArgs, ctx: ToolContext) -> dict[str, Any]:
        self.observed.append(args.sequence)
        return {
            "stdout": "diagnostic stream\n" * 3_000,
            "stderr": f"DIAGNOSTIC_{args.sequence:04d}", "returncode": 1,
            "stdout_truncated": False, "stderr_truncated": False,
        }


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_long_task_recovers_large_results_without_repeating_actions(layout: str) -> None:
    runtime = Runtime.open("local", config=replace(
        DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, prompt_layout=layout),
    ))
    try:
        diagnostic = _LargeDiagnostic()
        runtime.tools.register_tool(diagnostic, registered_by="test.host", ephemeral=True)
        image = AgentImage(
            image_id="pressure-probe:v0", name="pressure-probe", context_policy="working_set",
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
            default_tools=["large_diagnostic", "read_memory_object", "read_process_messages"],
        )
        runtime.register_image(image, actor="test.host")
        pid = runtime.process.spawn(image=image.image_id, goal="ORIGINAL_DELIVERABLE")
        runtime.human.send_process_message(pid, "KEEP_THE_CUSTOMER_CONSTRAINT")
        assert runtime.llm.dispatch(pid, {"action": "read_process_messages"})["ok"]
        plans = [runtime.memory.create_object(pid, ObjectType.PLAN, {"pending": f"old {i}"})
                 for i in range(24)]
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(
            pid, [*process.memory_view.roots, *plans],
        )
        process.resource_budget = replace(
            process.resource_budget, max_context_materialization_tokens=2_000,
        )
        runtime.store.update_process(process)

        class PressureClient:
            calls = 0

            def complete_action(self, messages: list[dict[str, Any]],
                                tools: list[dict[str, Any]]) -> LLMCompletion:
                prompt = str(messages[-1]["content"])
                assert "ORIGINAL_DELIVERABLE" in prompt
                assert "KEEP_THE_CUSTOMER_CONSTRAINT" in prompt
                step, recover = divmod(self.calls, 2)
                if recover:
                    stubs = [json.loads(line) for line in prompt.splitlines()
                             if line.startswith("{") and FEEDBACK_STUB_RECORD_TYPE in line]
                    current = [stub for stub in stubs
                               if stub.get("summary", {}).get("tool_name") == diagnostic.name][-1]
                    name = "read_memory_object"
                    args = {"name": current["name"], "namespace": current["namespace"],
                            "json_pointer": "/result/stderr", "max_payload_chars": 100}
                else:
                    if step:
                        assert f"DIAGNOSTIC_{step - 1:04d}" in prompt
                    name, args = diagnostic.name, {"sequence": step}
                self.calls += 1
                return LLMCompletion(content="", tool_calls=[{
                    "id": f"pressure_{self.calls}", "name": name, "arguments": json.dumps(args),
                }])

        client = PressureClient()
        runtime.llm.client = client
        for quantum in range(32):
            if quantum and quantum % 4 == 0:
                assert runtime.llm.dispatch(pid, {
                    "action": "read_process_messages", "include_acked": True,
                })["ok"]
            result = runtime.run_process_once(pid)
            assert result["ok"], result
        assert client.calls == 32
        assert diagnostic.observed == list(range(16)), "recover output without re-running the action"
    finally:
        runtime.close()


@pytest.mark.real_llm
@pytest.mark.timeout(360)
@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_real_model_recovers_large_feedback_and_keeps_followup(tmp_path: Path, layout: str) -> None:
    profile_id = DEFAULT_CONFIG.llm.default_profile_id
    profile = replace(
        DEFAULT_CONFIG.llm.profiles[profile_id], timeout_s=60, max_retries=0,
        logical_call_timeout_s=60,
    )
    config = replace(DEFAULT_CONFIG, llm=replace(
        DEFAULT_CONFIG.llm, prompt_layout=layout,
        profiles={**DEFAULT_CONFIG.llm.profiles, profile_id: profile},
    ))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = Runtime.open(
        tmp_path / "pressure.sqlite", config=config,
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        image = AgentImage(
            image_id="feedback-reader:v0", name="feedback-reader",
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT, context_policy="working_set",
            default_tools=["read_memory_object", "read_process_messages", "process_exit"],
        )
        runtime.register_image(image, actor="test.host")
        pid = runtime.process.spawn(image=image.image_id, goal=(
            "Inspect the diagnostic feedback already collected for this task. "
            "Return process_exit with an object payload containing failure_marker "
            "(the exact stderr string) and preserve (the exact filename from the "
            "acknowledged customer follow-up). Ground both values in retained data."
        ))
        runtime.human.send_process_message(pid, "Preserve RELEASE_ARCHIVE.md unchanged.")
        assert runtime.llm.dispatch(pid, {"action": "read_process_messages"})["ok"]
        marker = f"failure-{uuid4().hex}"
        feedback = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, {
            "tool_name": "run_shell_command", "ok": True,
            "result": {
                "argv": ["python", "-m", "unittest"], "returncode": 1,
                "stdout": "unrelated log output\n" * 3_000, "stderr": marker,
                "stdout_truncated": False, "stderr_truncated": False,
            },
        }, name="collected-diagnostic")
        plans = [runtime.memory.create_object(pid, ObjectType.PLAN, {"pending": f"old {i}"})
                 for i in range(24)]
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(pid, [
            *process.memory_view.roots, *plans, feedback,
        ])
        process.resource_budget = replace(
            process.resource_budget, max_context_materialization_tokens=2_000,
        )
        runtime.store.update_process(process)

        runtime.run_process_until_idle(pid, max_quanta=6)

        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.EXITED
        assert process.outcome is not None and process.outcome.result_oid is not None
        result = runtime.store.get_object(process.outcome.result_oid)
        assert result is not None
        assert result.payload["failure_marker"] == marker
        assert result.payload["preserve"] == "RELEASE_ARCHIVE.md"
        assert any(row.action == "tool.call" and row.decision.get("tool") == "read_memory_object"
                   for row in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()
