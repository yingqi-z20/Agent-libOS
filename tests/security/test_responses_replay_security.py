from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import LLMProfile
from agent_libos.models import (
    CapabilityRight,
    ObjectMetadata,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
    SinkTrustLevel,
    SinkTrustRule,
    ViewMode,
)
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.utils.serde import dumps
from tests.runtime.test_responses_replay_executor import (
    CONFIG,
    SECRET,
    ReplayClient,
    completion,
    register,
)


def _source_view(runtime: Runtime, pid: str, *, secret: bool = False):
    source = runtime.memory.create_object(
        pid, ObjectType.EVIDENCE, {"value": "original source content"},
        metadata=ObjectMetadata(sensitivity="secret" if secret else "normal"),
    )
    process = runtime.process.get(pid)
    process.memory_view = runtime.memory.create_view(pid, [source], mode=ViewMode.READ_ONLY)
    runtime.store.update_process(process)
    return source


def _remove_source_view(runtime: Runtime, pid: str) -> None:
    process = runtime.process.get(pid)
    process.memory_view = runtime.memory.create_view(pid, [], mode=ViewMode.READ_ONLY)
    runtime.store.update_process(process)


def _sink(runtime: Runtime, level: SinkTrustLevel) -> None:
    runtime.data_flow.register_sink_trust(
        SinkTrustRule(
            pattern="llm:default", trust_level=level, max_sensitivity="secret",
            identity_sha256=runtime.llms.profile_identity_sha256("default"),
        ),
        actor="test.host", require_capability=False,
        replace=runtime.data_flow.inspect_sink_trust("llm:default") is not None,
    )


@pytest.mark.parametrize("revoke_after_prepare", [False, True])
def test_revoked_historical_source_read_denies_private_replay_before_provider(
    revoke_after_prepare: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        source = _source_view(runtime, pid)
        assert runtime.run_process_once(pid)["ok"]
        _remove_source_view(runtime, pid)
        def revoke() -> None:
            runtime.capability.revoke(source.capability_id, revoked_by="test", require_authority=False)
            assert not runtime.capability.check(pid, f"object:{source.oid}", CapabilityRight.READ)

        if revoke_after_prepare:
            invoke = runtime.llm._invoke_prepared_llm_request

            async def revoke_before_protected_dispatch(state):
                revoke()
                return await invoke(state)

            monkeypatch.setattr(runtime.llm, "_invoke_prepared_llm_request", revoke_before_protected_dispatch)
        else:
            revoke()

        denied = runtime.run_process_once(pid)

        assert not denied["ok"]
        assert len(client.inputs) == 1
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.events.list(target=pid))
        assert any(record.decision.get("allowed") is False or "error" in record.decision
                   for record in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


def test_conditional_private_replay_cannot_reuse_revoked_source_after_approval() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        _sink(runtime, SinkTrustLevel.TRUSTED)
        pid = register(runtime)
        source = _source_view(runtime, pid, secret=True)
        assert runtime.run_process_once(pid)["ok"]
        _sink(runtime, SinkTrustLevel.CONDITIONAL)
        assert runtime.run_process_once(pid)["waiting_human"]
        runtime.human.drain_terminal_queue(auto_approve=True)
        _remove_source_view(runtime, pid)
        runtime.capability.revoke(source.capability_id, revoked_by="test", require_authority=False)

        with pytest.raises(CapabilityDenied):
            runtime.run_process_once(pid)
        assert len(client.inputs) == 1
        assert runtime.process.get(pid).status.value == "failed"
        assert SECRET not in dumps(runtime.store.get_llm_pending_action(pid))
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


@pytest.mark.parametrize("parallel", [False, True])
def test_native_parallel_groups_are_complete_or_repaired_before_tool_dispatch(parallel: bool) -> None:
    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        "default": LLMProfile(model="gpt-6-astra", api_mode="responses", responses_replay=True,
                              parallel_tool_calls=parallel),
    }))
    runtime = Runtime.open("local", config=config)
    try:
        first = completion(1)
        extra = completion(2)
        first.tool_calls.extend(extra.tool_calls)
        first.response_items.extend(extra.response_items)
        client = ReplayClient([first, completion(3), completion(4)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.run_process_once(pid)["ok"]
        if parallel:
            assert len(client.inputs) == 2
            assert [item.get("call_id") for item in client.inputs[1]
                    if item.get("type") == "function_call"] == ["call_1", "call_2"]
            assert [item.get("call_id") for item in client.inputs[1]
                    if item.get("type") == "function_call_output"] == ["call_1", "call_2"]
        else:
            assert len(client.inputs) == 3
            assert not any(item.get("call_id") in {"call_1", "call_2"}
                           for item in client.inputs[2])
            assert [item.get("call_id") for item in client.inputs[2]
                    if item.get("type") == "function_call_output"] == ["call_3"]
            assert any(record.action == "llm.action_repair_requested"
                       for record in runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()


@pytest.mark.parametrize("arrival", ["queued", "waiting", "reopened_wait"])
def test_image_only_replay_delivers_host_auto_wait_followup_once(arrival: str, tmp_path: Path) -> None:
    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        "default": LLMProfile(model="gpt-6-astra", api_mode="responses", responses_replay=True,
                              auto_wait_on_empty_tool_calls=True),
    }))
    database = tmp_path / "auto-wait.sqlite"
    runtime = Runtime.open(database, config=config)
    try:
        first = completion(1)
        first.tool_calls = []
        first.content = "Waiting for your message."
        first.response_items = first.response_items[:1] + [{
            "type": "message", "id": "message_1", "role": "assistant",
            "content": [{"type": "output_text", "text": first.content, "annotations": []}],
            "phase": "final_answer", "status": "completed",
        }]
        client = ReplayClient([first, completion(2)])
        runtime.llm.client = client
        pid = register(runtime, PROMPT_MODE_IMAGE_ONLY)
        if arrival == "queued":
            runtime.human.send_process_message(pid, "FOLLOWUP_MESSAGE_MUST_REACH_MODEL")
        else:
            assert runtime.run_process_once(pid)["waiting_message"]
            if arrival == "reopened_wait":
                runtime.close()
                runtime = Runtime.open(database, config=config)
                client = ReplayClient([completion(2)])
                runtime.llm.client = client
            runtime.human.send_process_message(pid, "FOLLOWUP_MESSAGE_MUST_REACH_MODEL")
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.messages.unread(pid) == []
        assert runtime.run_process_once(pid)["ok"]

        assert len(client.inputs) == (1 if arrival == "reopened_wait" else 2)
        assert json.dumps(client.inputs[-1]).count("FOLLOWUP_MESSAGE_MUST_REACH_MODEL") == 1
        assert not any(item.get("type") == "function_call_output" for item in client.inputs[-1])
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()


@pytest.mark.parametrize("reopen", [False, True])
def test_conditional_replay_freezes_private_input_and_resumes_exactly_once(reopen: bool, tmp_path: Path) -> None:
    database = tmp_path / "runtime.sqlite"
    runtime = Runtime.open(database, config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        _sink(runtime, SinkTrustLevel.TRUSTED)
        pid = register(runtime)
        _source_view(runtime, pid, secret=True)
        assert runtime.run_process_once(pid)["ok"]
        _sink(runtime, SinkTrustLevel.CONDITIONAL)

        waiting = runtime.run_process_once(pid)

        assert waiting["waiting_human"]
        assert len(client.inputs) == 1
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None
        assert pending["wait_type"] == "llm_release"
        assert SECRET not in dumps(pending)
        assert SECRET not in dumps(runtime.human.pending())
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.events.list(target=pid))
        prepared = pending["action"]
        reference = prepared["responses_replay_request"]
        assert set(reference) == {"turn_id", "payload_sha256"}
        retained = runtime.store.get_llm_replay_turn(reference["turn_id"])
        assert retained is not None and SECRET in json.dumps(retained.payload)
        frozen = retained.payload["request"]["response_items"]
        assert "responses_items" not in prepared["egress_payload"]

        _remove_source_view(runtime, pid)
        if reopen:
            runtime.close()
            runtime = Runtime.open(database, config=CONFIG)
            client = ReplayClient([completion(2)])
            runtime.llm.client = client
        runtime.human.drain_terminal_queue(auto_approve=True)
        resumed = runtime.run_process_once(pid)

        assert resumed["ok"], resumed
        assert resumed["resumed_after_human"]
        assert client.inputs[-1] == frozen
        assert len(client.inputs) == (1 if reopen else 2)
        assert len([row for row in runtime.store.list_llm_calls(pid=pid) if row.status == "ok"]) == 2
        assert runtime.store.get_llm_pending_action(pid)["status"] == "completed"
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()


def test_first_conditional_replay_request_survives_restart_without_prior_head(tmp_path: Path) -> None:
    database = tmp_path / "first-request.sqlite"
    runtime = Runtime.open(database, config=CONFIG)
    try:
        client = ReplayClient([completion(1)])
        runtime.llm.client = client
        _sink(runtime, SinkTrustLevel.CONDITIONAL)
        pid = register(runtime)
        source = _source_view(runtime, pid, secret=True)

        waiting = runtime.run_process_once(pid)

        assert waiting["waiting_human"]
        assert client.inputs == []
        assert runtime.store.get_llm_replay_head(pid) is None
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None and pending["wait_type"] == "llm_release"
        prepared = pending["action"]
        reference = prepared["responses_replay_request"]
        retained = runtime.store.get_llm_replay_turn(reference["turn_id"])
        assert retained is not None
        assert retained.payload["request"]["expected_head"] is None
        frozen = retained.payload["request"]["response_items"]
        assert "original source content" in json.dumps(frozen)
        assert "responses_items" not in prepared["egress_payload"]
        assert SECRET not in dumps(pending)

        _remove_source_view(runtime, pid)
        runtime.close()
        runtime = Runtime.open(database, config=CONFIG)
        client = ReplayClient([completion(1)])
        runtime.llm.client = client
        assert runtime.store.get_llm_replay_head(pid) is None
        assert runtime.capability.check(pid, f"object:{source.oid}", CapabilityRight.READ)
        runtime.human.drain_terminal_queue(auto_approve=True)

        resumed = runtime.run_process_once(pid)

        assert resumed["ok"], resumed
        assert resumed["resumed_after_human"]
        assert client.inputs == [frozen]
        calls = [row for row in runtime.store.list_llm_calls(pid=pid) if row.status == "ok"]
        assert len(calls) == 1
        assert len(runtime.store.list_llm_tool_outputs(pid=pid, response_id=calls[0].call_id)) == 1
        assert runtime.store.get_llm_pending_action(pid)["status"] == "completed"
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.events.list(target=pid))
    finally:
        runtime.close()
