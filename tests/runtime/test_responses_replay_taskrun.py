from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.llm.prompt import build_system_prompt
from agent_libos.llm.replay import ReplayStateError
from agent_libos.llm.task_runs import validated_action_manifest
from agent_libos.models import LLMCallRecord, PROMPT_MODE_LIBOS_DEFAULT, TaskRunRequirementStatus, TaskRunRetention, TaskRunSpecV1, TaskRunStatus
from agent_libos.models.data_flow import DataFlowContext, DataLabels
from agent_libos.utils.ids import utc_now


IMAGE = "responses-taskrun-replay:v0"
SECRET = "PRIVATE_TASKRUN_REASONING"
LOCAL_CALL = "llmcall-committed-before-restart"
NATIVE_CALL = "native-function-call-before-restart"
CONFIG = replace(
    DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm, profiles={"default": LLMProfile(model="gpt-6-astra", api_mode="responses", responses_replay=True)}),
    task_runs=replace(DEFAULT_CONFIG.task_runs, plaintext_payloads_enabled=True),
)


class ReplayClient(LLMClient):
    def __init__(self, *, finish: bool) -> None:
        super().__init__(model="gpt-6-astra", api_key="test", api_mode="responses", responses_replay=True)
        self.finish = finish
        self.inputs: list[list[dict[str, Any]]] = []

    async def acomplete_action(self, messages: Any, tools: Any, **kwargs: Any) -> LLMCompletion:
        assert self.finish, "recovery must dispatch the committed action without a provider call"
        self.inputs.append(deepcopy(kwargs["responses_items"]))
        assert len(self.inputs) == 1, "finishing the task must require one provider call"
        arguments = json.dumps({"payload": {"done": True}})
        item = {"type": "function_call", "id": "fc-finish", "call_id": "call-finish", "name": "process_exit", "arguments": arguments, "status": "completed"}
        return LLMCompletion(
            content="", tool_calls=[dict(item)], api="responses", response_id="response-finish", model="gpt-6-astra",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "reasoning_tokens": 0},
            response_items=[item],
        )


class EchoReplayClient(LLMClient):
    def __init__(self, *, replay: bool = True) -> None:
        super().__init__(model="gpt-6-astra", api_key="test", api_mode="responses", responses_replay=replay)
        self.requests: list[dict[str, Any]] = []

    async def acomplete_action(self, messages: Any, tools: Any, **kwargs: Any) -> LLMCompletion:
        self.requests.append(deepcopy(kwargs))
        index = len(self.requests)
        arguments = json.dumps({"message": f"VISIBLE_TASKRUN_ECHO_{index}"})
        item = {"type": "function_call", "id": f"fc-echo-{index}", "call_id": f"call-echo-{index}", "name": "echo", "arguments": arguments, "status": "completed"}
        return LLMCompletion(content="", tool_calls=[dict(item)], api="responses", response_id=f"response-echo-{index}", model="gpt-6-astra", usage={"input_tokens": 100, "output_tokens": 20}, response_items=[item])


def seed_validated_wait(runtime: Runtime):
    image = AgentImage(image_id=IMAGE, name="durable Responses replay", system_prompt="Wait for a message, then finish.", default_tools=["receive_process_messages", "process_exit"])
    runtime.register_image(image, actor="test.host")
    created = runtime.task_runs.create(TaskRunSpecV1(goal="wait, then finish", display_title="Durable native replay", image_id=IMAGE, retention=TaskRunRetention.PERMANENT), client_request_id="create-native-replay")
    pid = created.root_pid
    requirement = runtime.store.list_task_run_requirements(created.run_id)[0]
    runtime.store.update_task_run_requirement_cas(requirement.requirement_id, expected_status=TaskRunRequirementStatus.PENDING, status=TaskRunRequirementStatus.IN_PROGRESS, updated_at=created.updated_at, started_at=created.updated_at)
    generation = runtime.store.get_llm_context_generation(pid)
    binding = runtime.task_runs.requirement_binding_for_prompt(pid, context_generation=generation)
    resolved = runtime.llms.resolve_for_process(pid)
    flow = DataFlowContext(labels=DataLabels(trust_level="user_asserted", integrity="checked"))
    messages = [{"role": "system", "content": build_system_prompt(image)}, {"role": "user", "content": "wait, then finish"}]
    request = runtime.llm.replay.prepare(pid=pid, run_id=created.run_id, provider_fingerprint=resolved.identity_sha256, model="gpt-6-astra", context_generation=generation, messages=messages, flow_context=flow)
    call_item = {"type": "function_call", "id": "provider-item-before-restart", "call_id": NATIVE_CALL, "name": "receive_process_messages", "arguments": "{}", "status": "completed"}
    native = [{"type": "reasoning", "id": "reasoning-before-restart", "summary": [], "encrypted_content": SECRET}, call_item]
    head = runtime.llm.replay.stage(request, call_id=LOCAL_CALL, response_items=native, usage={"output_tokens": 20, "reasoning_tokens": 12}, max_output_tokens=100, response_id="response-before-restart")
    staged = runtime.store.get_llm_replay_turn(head.turn_id)
    now = utc_now()
    runtime.store.insert_llm_call(LLMCallRecord(
        call_id=LOCAL_CALL, pid=pid, image_id=IMAGE, purpose="action_selection", status="ok", api="responses", model="gpt-6-astra", response_id="response-before-restart",
        messages=messages, tool_calls=[dict(call_item)], created_at=now, completed_at=now,
        request_options={
            "task_run_requirement_binding_v1": binding,
            "llm_context_generation": generation,
            "responses_replay": {"schema_version": 1, "enabled": True, "turn_id": staged.turn_id, "payload_sha256": staged.payload_sha256, "item_count": len(native)},
        },
    ))
    runtime.llm.replay.mark_validated(pid=pid, call_id=LOCAL_CALL)
    manifest = validated_action_manifest([{"action": "receive_process_messages"}], call_id=LOCAL_CALL, parallel_tool_calls=False, host_auto_wait=False, tool_call_count=1, data_labels=flow.labels.to_dict())
    runtime.task_runs.record_validated_transcript(pid=pid, call_id=LOCAL_CALL, action_manifest=manifest, context_generation=generation)
    return created, manifest


def test_taskrun_native_call_survives_action_recovery_then_wait_recovery(tmp_path: Path) -> None:
    target = tmp_path / "native-taskrun-replay.sqlite"
    first = Runtime.open(target, config=CONFIG)
    try:
        created, _manifest = seed_validated_wait(first)
    finally:
        first.close()
    second = Runtime.open(target, config=CONFIG)
    try:
        no_provider = ReplayClient(finish=False)
        second.llm.client = no_provider
        recovered = second.task_runs.get(created.run_id)
        waiting = second.task_runs.run_until_blocked(created.run_id, expected_revision=recovered.revision, command_id="recover-original-call")
        assert waiting.status is TaskRunStatus.WAITING_MESSAGE
        pending = second.store.get_llm_pending_action(created.root_pid)
        assert pending["response_id"] == LOCAL_CALL
        assert pending["tool_call_id"] == NATIVE_CALL
        assert no_provider.inputs == []
    finally:
        second.close()
    third = Runtime.open(target, config=CONFIG)
    try:
        client = ReplayClient(finish=True)
        third.llm.client = client
        recovered = third.task_runs.get(created.run_id)
        third.messages.post(sender="human:test", recipient_pid=created.root_pid, subject="continue", payload={"ready": True})
        completed = third.task_runs.run_until_blocked(created.run_id, expected_revision=recovered.revision, command_id="finish-recovered-wait")
        assert completed.status is TaskRunStatus.SUCCEEDED
        assert len(client.inputs) == 1
        wire = client.inputs[0]
        assert sum(item.get("encrypted_content") == SECRET for item in wire) == 1
        assert sum(item.get("type") == "function_call" and item.get("call_id") == NATIVE_CALL for item in wire) == 1
        assert sum(item.get("type") == "function_call_output" and item.get("call_id") == NATIVE_CALL for item in wire) == 1
        assert len(third.store.list_llm_calls(pid=created.root_pid)) == 2
    finally:
        third.close()


def test_taskrun_native_replay_rejects_manifest_arguments_that_differ_from_provider() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        created, manifest = seed_validated_wait(runtime)
        mismatched = [{"action": "receive_process_messages", "channel": "changed"}]
        with pytest.raises(ReplayStateError, match="function arguments changed"):
            runtime.llm._task_run_resume_completion(created.root_pid, manifest, mismatched)
        assert runtime.store.get_llm_pending_action(created.root_pid) is None
    finally:
        runtime.close()


@pytest.mark.parametrize("disabled_setting", ["responses_replay", "persist_full_io"])
def test_reenabling_replay_cannot_skip_an_intervening_successful_call(disabled_setting: str) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        runtime.register_image(AgentImage(image_id=IMAGE, name="replay continuity", system_prompt="Use echo.", default_tools=["echo"], prompt_mode=PROMPT_MODE_LIBOS_DEFAULT), actor="test.host")
        pid = runtime.process.spawn(image=IMAGE, goal="Echo three observations")
        client = EchoReplayClient()
        runtime.llm.client = client
        assert runtime.run_process_once(pid)["ok"]
        original = runtime.store.get_llm_replay_head(pid)
        if disabled_setting == "responses_replay":
            client.responses_replay = False
        else:
            runtime.llm.config = replace(CONFIG, llm=replace(CONFIG.llm, persist_full_io=False))
        ordinary = runtime.run_process_once(pid)
        assert ordinary["ok"], ordinary
        assert client.requests[1].get("responses_items") is None
        client.responses_replay = True
        runtime.llm.config = CONFIG
        outcome = runtime.run_process_once(pid)
        assert not outcome["ok"]
        assert len(client.requests) == 2
        assert runtime.store.get_llm_replay_head(pid) == original
    finally:
        runtime.close()


def test_first_taskrun_replay_request_seeds_existing_visible_transcript() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        runtime.register_image(AgentImage(image_id=IMAGE, name="TaskRun replay upgrade", system_prompt="Use echo.", default_tools=["echo"]), actor="test.host")
        created = runtime.task_runs.create(TaskRunSpecV1(goal="Echo two observations", display_title="Replay upgrade", image_id=IMAGE, retention=TaskRunRetention.PERMANENT), client_request_id="create-replay-upgrade")
        client = EchoReplayClient(replay=False)
        runtime.llm.client = client
        running = runtime.task_runs.run_until_blocked(created.run_id, expected_revision=created.revision, command_id="legacy-transcript", max_quanta=1)
        assert running.status is TaskRunStatus.RUNNING
        assert runtime.store.get_llm_replay_head(created.root_pid) is None
        client.responses_replay = True
        continued = runtime.task_runs.run_until_blocked(created.run_id, expected_revision=running.revision, command_id="first-replay-turn", max_quanta=1)
        assert continued.status is TaskRunStatus.RUNNING
        assert len(client.requests) == 2
        wire = client.requests[1]["responses_items"]
        legacy_observations = [item for item in wire if item.get("role") == "assistant" and "VISIBLE_TASKRUN_ECHO_1" in item.get("content", "")]
        assert len(legacy_observations) == 1, wire
        legacy_result = json.loads(legacy_observations[0]["content"])
        assert legacy_result["ok"] is True
        assert legacy_result["result"]["payload"]["message"] == "VISIBLE_TASKRUN_ECHO_1"
    finally:
        runtime.close()


@pytest.mark.parametrize("corruption", ["missing", "changed-payload"])
def test_corrupt_replay_run_is_isolated_during_startup(tmp_path: Path, corruption: str) -> None:
    target = tmp_path / "corrupt-replay.sqlite"
    runtime = Runtime.open(target, config=CONFIG)
    try:
        created, _ = seed_validated_wait(runtime)
        head = runtime.store.get_llm_replay_head(created.root_pid)
        unaffected_pid = runtime.process.spawn(goal="unaffected process")
    finally:
        runtime.close()
    with sqlite3.connect(target) as connection:
        if corruption == "missing":
            connection.execute("DELETE FROM llm_replay_turns WHERE turn_id = ?", (head.turn_id,))
        else:
            connection.execute("UPDATE llm_replay_turns SET payload_json = '{}' WHERE turn_id = ?", (head.turn_id,))
    reopened = Runtime.open(target, config=CONFIG)
    try:
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert any(blocker["kind"] == "payload_corrupt" for blocker in summary.blockers)
        assert reopened.process.get(unaffected_pid) is not None
        assert not reopened.store.list_llm_calls(pid=unaffected_pid)
        assert len(reopened.store.list_llm_calls(pid=created.root_pid)) == 1
    finally:
        reopened.close()
