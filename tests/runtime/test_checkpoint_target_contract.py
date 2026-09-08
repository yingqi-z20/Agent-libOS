"""Keep checkpoint caller selection actionable under strict model schemas."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.builtin.checkpoint import CreateCheckpointArgs, CreateCheckpointTool
from agent_libos.tools.contracts import CURRENT_PROCESS
from agent_libos.utils.openai_schema import openai_responses_tool_schema


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_strict_checkpoint_schema_explains_its_required_nullable_pid(layout: str, api: str) -> None:
    chat = CreateCheckpointTool().to_openai_chat_tool(prompt_layout=layout)
    tool = openai_responses_tool_schema(chat) if api == "responses" else chat["function"]
    assert tool is not None
    schema = tool["parameters"]
    pid_schema = schema["properties"]["pid"]

    assert tool["strict"] is True
    assert schema["required"] == ["reason", "pid"]
    assert pid_schema["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert pid_schema["description"] == CURRENT_PROCESS.description
    assert "Pass JSON null to select the caller" in pid_schema["description"]
    assert "omission is also valid when allowed by the call schema" in pid_schema["description"]
    assert "Do not guess 'self'" in pid_schema["description"]
    assert "projected Capability resource" in pid_schema["description"]


@pytest.mark.parametrize("target", [{}, {"pid": None}])
@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_null_and_omitted_checkpoint_target_create_for_the_actual_caller(
    target: dict[str, None], layout: str,
) -> None:
    config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, prompt_layout=layout))
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="Save one verified milestone")

        result = runtime.llm.dispatch(pid, {
            "action": "create_checkpoint", "reason": "Verified milestone", **target,
        })

        assert result["ok"], result
        checkpoints = runtime.checkpoint.list(pid, actor=pid)
        assert len(checkpoints) == 1
        assert checkpoints[0]["pid"] == pid
        created = [record for record in runtime.store.list_audit() if record.action == "checkpoint.create"]
        assert len(created) == 1
        assert created[0].decision["pid"] == pid
        if layout == "cache_optimized_v2":
            assert result["payload"] == {"created": True, "reason": "Verified milestone"}
            assert pid not in json.dumps(result["payload"])
            assert "checkpoint_id" not in result["payload"]
        else:
            assert result["payload"]["pid"] == pid
            assert result["payload"]["checkpoint_id"] == checkpoints[0]["checkpoint_id"]
    finally:
        runtime.close()


def test_literal_self_and_foreign_checkpoint_targets_remain_unauthorized() -> None:
    runtime = Runtime.open("local")
    try:
        caller = runtime.process.spawn(image="base-agent:v0", goal="Save only my own state")
        foreign = runtime.process.spawn(image="base-agent:v0", goal="Separate process state")
        for target in ("self", foreign):
            assert CreateCheckpointArgs(reason="Unauthorized target", pid=target).pid == target

            result = runtime.tools.call(caller, "create_checkpoint", {
                "reason": "Unauthorized target", "pid": target,
            })

            assert not result.ok
            assert result.payload["error"]["code"] == "permission_denied"
            assert runtime.checkpoint.list(caller, actor=caller) == []
            assert runtime.checkpoint.list(foreign, actor=foreign) == []
        assert not any(record.action == "checkpoint.create" for record in runtime.store.list_audit())
        failures = [
            record for record in runtime.store.list_audit()
            if record.action == "tool.call"
            and record.decision.get("tool") == "create_checkpoint"
            and record.decision.get("ok") is False
        ]
        assert len(failures) == 2
        assert all(record.decision["error"]["error_type"] == "CapabilityDenied" for record in failures)
    finally:
        runtime.close()


def test_checkpoint_skill_accepts_compact_creation_receipt_without_demanding_an_id() -> None:
    path = Path(__file__).resolve().parents[2] / "agent_libos/skills/builtin/agent-libos-checkpoints/SKILL.md"
    instructions = " ".join(path.read_text().split())

    assert CURRENT_PROCESS.description in instructions
    assert "Do not guess 'self'" in instructions
    assert "a successful tool result with `created: true` confirms creation" in instructions
    assert "Do not create another checkpoint to obtain an id" in instructions
    assert "`created: true` is sufficient and does not require an id" in instructions
    assert "so require the id" not in instructions
    assert "Create/select one checkpoint and retain its id" not in instructions
