from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.prompt import _compact_tool_result_payload, build_user_prompt
from agent_libos.models import MaterializedContext
from agent_libos.skills import get_builtin_skill_catalog


_APPEND_HINT = "target_payload_is_not_a_container_recreate_with_object_or_array"
_SPECIALIZED_TOOLS = (
    "create_memory_object",
    "create_memory_namespace",
    "list_memory_namespace",
    "read_memory_object",
    "append_memory_object",
    "process_exit",
)


def _failure_carrier(tool_name: str) -> dict[str, Any]:
    return {
        "tool_id": "tool_private_identifier",
        "tool_name": tool_name,
        "ok": False,
        "failure": {
            "ok": False,
            "error": {
                "code": "execution_error",
                "type": "ValidationError",
                "message": "private error text corr_private_identifier",
                "correlation_id": "corr_private_identifier",
                "details": {
                    "hint": _APPEND_HINT,
                    "correlation_id": "corr_private_identifier",
                    "object_oid": "obj_private_identifier",
                    "payload": "private business payload",
                },
            },
            "internal_error": {"message": "private traceback"},
            "data": "private failure data",
        },
        "content": "private carrier content",
        "artifacts": [{"path": "/private/carrier-artifact"}],
    }


@pytest.mark.parametrize("tool_name", _SPECIALIZED_TOOLS)
def test_specialized_failure_preserves_only_public_diagnostics(tool_name: str) -> None:
    projected = _compact_tool_result_payload(_failure_carrier(tool_name))

    assert projected == {
        "tool_name": tool_name,
        "result": {
            "ok": False,
            "error": {
                "code": "execution_error",
                "type": "ValidationError",
                "retryable": False,
                "safe_message": "Tool execution failed.",
                "details": {"hint": _APPEND_HINT},
            },
        },
    }
    assert "private" not in json.dumps(projected)
    assert "error_hash" not in json.dumps(projected)


@pytest.mark.parametrize("tool_name", ("append_memory_object", "process_exit"))
def test_omitted_failure_stays_a_visible_failure(tool_name: str) -> None:
    projected = _compact_tool_result_payload({
        "tool_name": tool_name,
        "ok": False,
        "failure_omitted": True,
        "reason": "private diagnostic",
    })

    assert projected["result"]["ok"] is False
    assert projected["result"]["error"]["type"] == "ToolError"
    assert "private" not in json.dumps(projected)


def test_successful_memory_business_fields_and_explicit_projection_are_unchanged() -> None:
    business_payload = {"ok": False, "failure": {"namespace": "process:pid_customer"}}
    projected = _compact_tool_result_payload({
        "tool_name": "read_memory_object",
        "result": {
            "name": "ledger",
            "namespace": "process:pid_current",
            "payload": business_payload,
        },
    }, current_namespace="process:pid_current")
    assert projected["result"] == {
        "name": "ledger", "namespace": None, "payload": business_payload,
    }

    explicit = {"status": "review_required", "receipt": {"recoverable": True}}
    carrier = _failure_carrier("process_exit")
    carrier["model_projection"] = explicit
    carrier.pop("content")
    carrier.pop("artifacts")
    assert _compact_tool_result_payload(carrier)["result"] is explicit


def test_failure_fix_does_not_change_other_tool_receipts_or_legacy_prompt() -> None:
    other = _failure_carrier("restore_checkpoint")
    assert _compact_tool_result_payload(other)["result"] == other["failure"]

    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="legacy failure prompt")
        record = {
            "record_type": "object_memory_object", "type": "tool_result",
            "payload": _failure_carrier("append_memory_object"),
        }
        text = json.dumps(record)
        prompt = build_user_prompt(
            runtime.process.get(pid),
            MaterializedContext(
                text=text, object_refs=[], token_count=0,
                omitted_objects=[], policy_used="recency_first",
            ),
            [], [], [], prompt_layout="legacy_v1",
        )
        assert text in prompt
    finally:
        runtime.close()


class _AppendFailureClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: Any) -> LLMCompletion:
        self.prompts.append(str(messages[-1]["content"]))
        if len(self.prompts) == 1:
            name = "append_memory_object"
            args = {"name": "ledger", "entry": {"step": "checked"}, "list_field": "entries"}
        else:
            name = "read_memory_object"
            args = {"name": "ledger"}
        return LLMCompletion(
            content="",
            tool_calls=[{"id": "failure-projection-test", "name": name, "arguments": json.dumps(args)}],
            api="chat", model="deterministic-test-client",
        )


def test_v2_next_quantum_exposes_real_append_failure_hint() -> None:
    config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, prompt_layout="cache_optimized_v2"))
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Update the ledger")
        package = get_builtin_skill_catalog().get("agent-libos-object-memory")
        assert package is not None
        assert runtime.llm.dispatch(pid, {
            "action": "activate_skill", "skill_id": package.skill_id,
            "expected_package_sha256": package.package_sha256,
        })["ok"]
        assert runtime.llm.dispatch(pid, {
            "action": "create_memory_object", "name": "ledger", "type": "summary",
            "payload": '{"entries":[]}', "immutable": False,
        })["ok"]
        client = _AppendFailureClient()
        runtime.llm.client = client

        failed = runtime.run_process_once(pid)
        assert failed["ok"] and not failed["result"]["ok"]
        assert _APPEND_HINT in json.dumps(failed["result"]["payload"])
        assert runtime.run_process_once(pid)["ok"]

        next_prompt = client.prompts[1]
        records = [json.loads(line) for line in next_prompt.splitlines() if line.startswith("{")]
        failure = next(
            record["payload"]["result"] for record in records
            if isinstance(record.get("payload"), dict)
            and record["payload"].get("tool_name") == "append_memory_object"
        )
        assert failure["ok"] is False
        assert failure["error"]["type"] == "ValidationError"
        assert failure["error"]["details"]["hint"] == _APPEND_HINT
        rendered = json.dumps(failure)
        assert "correlation_id" not in rendered
        assert "internal_error" not in rendered
        assert "error_hash" not in rendered
        assert "corr_" not in rendered
        assert pid not in rendered
    finally:
        runtime.close()
