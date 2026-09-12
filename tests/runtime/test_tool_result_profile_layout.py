from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile
from agent_libos.llm.client import LLMCompletion, LLMError
from agent_libos.models import ProcessStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.tools.prompt_layout import model_prompt_layout


_LAYOUT_PAIRS = [
    ("legacy_v1", "cache_optimized_v2"),
    ("cache_optimized_v2", "legacy_v1"),
]


def _open_runtime(global_layout: str, profile_layout: str) -> Runtime:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            prompt_layout=global_layout,
            profiles={
                "default": LLMProfile(),
                "selected": LLMProfile(
                    prompt_layout=profile_layout,
                    responses_replay=False,
                ),
            },
        ),
    )
    return Runtime.open(":memory:", config=config)


@pytest.mark.parametrize("global_layout,profile_layout", _LAYOUT_PAIRS)
def test_profile_review_reaches_the_model_and_allows_terminal_exit(
    global_layout: str, profile_layout: str,
) -> None:
    runtime = _open_runtime(global_layout, profile_layout)
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="Inspect current authority and finish.",
            llm_profile_id="selected",
        )
        inspected = runtime.llm.dispatch(pid, {"action": "list_capabilities"})
        assert inspected["ok"] is True
        first = runtime.llm.dispatch(pid, {"action": "process_exit"})
        assert first["ok"] is True
        review = first["payload"]["completion_review"]
        token = review["review_token"]
        check = {
            "status": "completed",
            "evidence_tool_calls": ["list_capabilities"],
            "evidence_summary": "Inspected the current authority successfully.",
        }
        evidence: dict[str, Any] = {
            "acceptance_checks": [check],
            "final_verification": ["list_capabilities"],
        }
        if profile_layout == "cache_optimized_v2":
            assert len(review["requirements"]) == 1
            for forbidden in (pid, "goal_oid", "source_refs", "reviewed_message_ids"):
                assert forbidden not in json.dumps(review)
        else:
            evidence.update(
                goal_oid=review["goal"]["oid"],
                reviewed_message_ids=review["acknowledged_human_message_ids"],
            )
            check.update(
                requirement="Inspect current authority and finish.",
                source_refs=review["completion_source_refs"],
            )

        class ConfirmingClient:
            calls = 0

            def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
                self.calls += 1
                assert token in json.dumps(messages)
                assert "obtain a fresh review_token" not in json.dumps(messages)
                schema = next(
                    tool["function"]["parameters"] for tool in tools
                    if tool["function"]["name"] == "process_exit"
                )
                expected = (
                    "CompactProcessCompletionEvidence"
                    if profile_layout == "cache_optimized_v2"
                    else "ProcessCompletionEvidence"
                )
                assert schema["properties"]["completion_evidence"]["anyOf"][0] == {
                    "$ref": f"#/$defs/{expected}"
                }
                return LLMCompletion(
                    content="",
                    tool_calls=[{
                        "id": "confirm_completion",
                        "name": "process_exit",
                        "arguments": json.dumps({
                            "review_token": token,
                            "completion_evidence": evidence,
                            "message": "Authority inspected.",
                        }),
                    }],
                )

        client = ConfirmingClient()
        runtime.llms.set_test_client("selected", client)
        completed = runtime.run_process_once(pid)
        assert completed["ok"] is True, completed
        assert client.calls == 1
        assert runtime.process.get(pid).status == ProcessStatus.EXITED
    finally:
        runtime.close()


@pytest.mark.parametrize("global_layout,profile_layout", _LAYOUT_PAIRS)
def test_checkpoint_and_message_receipts_use_updated_caller_profile(
    global_layout: str, profile_layout: str,
) -> None:
    runtime = _open_runtime(global_layout, global_layout)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0", goal="Save a milestone", llm_profile_id="selected",
        )
        runtime.llms.register_profile(
            "selected", LLMProfile(prompt_layout=profile_layout, responses_replay=False),
        )
        created = runtime.llm.dispatch(pid, {
            "action": "create_checkpoint", "reason": "Verified milestone",
        })
        listed = runtime.llm.dispatch(pid, {"action": "list_checkpoints"})
        assert created["ok"] is True, created
        assert listed["ok"] is True, listed
        if profile_layout == "cache_optimized_v2":
            assert created["payload"] == {"created": True, "reason": "Verified milestone"}
            assert listed["payload"]["checkpoints"] == [
                {"checkpoint_ref": "only", "reason": "Verified milestone"},
            ]
        else:
            assert created["payload"]["pid"] == pid
            assert listed["payload"]["checkpoints"][0]["checkpoint_id"] == created["payload"]["checkpoint_id"]

        message = runtime.human.send_process_message(pid, "Inspect the saved milestone.")
        received = runtime.llm.dispatch(pid, {"action": "read_process_messages"})
        assert received["ok"] is True, received
        assert received["payload"]["messages"][0]["message_id"] == message.message_id
        assert received["payload"]["acked_message_ids"] == (
            [] if profile_layout == "cache_optimized_v2" else [message.message_id]
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("layout,expected", [
    ("legacy_v1", "legacy_v1"),
    ("cache_optimized_v2", "cache_optimized_v2"),
    ("auto", "legacy_v1"),
])
def test_direct_tool_consumers_keep_configured_projection_without_registry(
    layout: str, expected: str,
) -> None:
    runtime = SimpleNamespace(config=SimpleNamespace(llm=SimpleNamespace(prompt_layout=layout)))
    assert model_prompt_layout(runtime, "caller") == expected
    assert model_prompt_layout(SimpleNamespace(), "caller") == "legacy_v1"


@pytest.mark.parametrize("error", [ValidationError("unknown profile"), LLMError("invalid profile")])
def test_invalid_profile_keeps_the_safe_projection_fallback(error: Exception) -> None:
    def profile_snapshot(profile_id: str) -> None:
        assert profile_id == "selected"
        raise error

    runtime = SimpleNamespace(
        config=SimpleNamespace(llm=SimpleNamespace(prompt_layout="auto", default_profile_id="default")),
        process=SimpleNamespace(get=lambda pid: SimpleNamespace(llm_profile_id="selected")),
        llms=SimpleNamespace(profile_snapshot=profile_snapshot),
    )
    assert model_prompt_layout(runtime, "caller") == "legacy_v1"
