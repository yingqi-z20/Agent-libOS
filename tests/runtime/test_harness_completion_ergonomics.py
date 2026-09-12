from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.images.default_agents.coding import CODING_AGENT_PROMPT
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import CapabilityRight, ProcessStatus
from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog
from agent_libos.tools.builtin.process import _build_cumulative_exit_review


class _FinalResponseClient:
    def __init__(self, exit_args: dict[str, Any]) -> None:
        self.exit_args = exit_args
        self.calls = 0

    def complete_action(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMCompletion:
        self.calls += 1
        assert self.calls == 1, "finalization must not need another model response"
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "final_report",
                    "name": "human_output",
                    "arguments": json.dumps({"message": "Authority inspected."}),
                },
                {
                    "id": "final_exit",
                    "name": "process_exit",
                    "arguments": json.dumps(self.exit_args),
                },
            ],
            raw=SimpleNamespace(id="final_response"),
            api="chat",
            model="deterministic-final-response",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
@pytest.mark.parametrize("new_followup", [False, True])
def test_same_response_report_and_exit_preserves_review_and_human_input_gate(
    tmp_path: Path, layout: str, new_followup: bool
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm, prompt_layout=layout, parallel_tool_calls=True
        ),
    )
    runtime = Runtime.open(tmp_path / "completion.sqlite", config=config)
    delivered: list[str] = []
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="Inspect current authority, send a concise human-facing summary, and exit.",
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.activate_skill(pid, "agent-libos-human-collaboration")
        observed = runtime.llm.dispatch(pid, {"action": "list_capabilities"})
        assert observed["ok"] is True
        first = runtime.llm.dispatch(
            pid, {"action": "process_exit", "message": "Authority inspected."}
        )
        assert first["payload"]["status"] == "completion_review_required"
        review = _build_cumulative_exit_review(runtime, pid)
        check = {
            "status": "completed",
            "evidence_tool_calls": ["list_capabilities", "human_output"],
            "evidence_summary": "Inspected authority and delivered the concise summary.",
        }
        evidence: dict[str, Any] = {
            "acceptance_checks": [check],
            "final_verification": ["list_capabilities"],
        }
        if layout == "legacy_v1":
            evidence.update(
                goal_oid=review["goal"]["oid"],
                reviewed_message_ids=[],
            )
            check.update(
                requirement="Inspect authority and report the result.",
                source_refs=[review["goal"]["oid"]],
            )
        client = _FinalResponseClient(
            {
                "review_token": review["review_token"],
                "completion_evidence": evidence,
                "message": "Authority inspected.",
            }
        )
        runtime.llm.client = client

        def receive_output(message: str) -> None:
            delivered.append(message)
            if new_followup:
                runtime.human.send_process_message(
                    pid, "Before exiting, inspect the new requirement."
                )

        runtime.substrate.human.output_sink = receive_output

        result = runtime.run_process_once(pid)

        assert client.calls == 1
        assert delivered == ["Authority inspected."]
        passed = [
            row
            for row in runtime.audit.trace(actor=pid)
            if row.action == "process.exit_review_passed"
        ]
        if new_followup:
            assert runtime.process.get(pid).status != ProcessStatus.EXITED
            assert passed == []
            messages = runtime.store.list_process_messages(pid)
            assert len(messages) == 1
            assert _build_cumulative_exit_review(runtime, pid)["review_token"] != review[
                "review_token"
            ]
        else:
            assert result["ok"] is True
            assert result["executed_count"] == 2
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            assert len(passed) == 1
            assert passed[0].decision["review_token"] == review["review_token"]
    finally:
        runtime.close()


def test_completion_guidance_batches_terminal_pair_without_repeating_reports() -> None:
    package = get_builtin_skill_catalog().get("agent-libos-runtime-session")
    assert package is not None
    exit_guide = package.instructions.split("### `process_exit`", 1)[1].split(
        "## Recommended workflow", 1
    )[0]

    assert "Call alone" not in exit_guide
    assert "same response" in exit_guide
    assert "do not send a final result again if it was already delivered" in exit_guide
    assert "process_exit` alone with that result" in package.instructions
    assert "brevity must not omit a deliverable" in " ".join(CODING_AGENT_PROMPT.split())
    assert "Include another human_output only if" in CODING_AGENT_PROMPT


def test_edit_guidance_reuses_baseline_but_keeps_cas_and_final_readback() -> None:
    package = get_builtin_skill_catalog().get("agent-libos-workspace-editing")
    assert package is not None
    instructions = package.instructions

    assert "reuse complete content and its non-null `content_sha256`" in instructions
    assert "pass its exact digest as `expected_content_sha256`" in instructions
    assert "Never drop `expected_content_sha256` or replace it with `null`" in instructions
    assert "If conditional writes are unsupported, report" in instructions
    assert "a written file re-reads completely" in instructions
    assert "tests or validations affected by the change ran after the final write" in instructions
