from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import LLMProfile
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import CapabilityRight, DataFlowContext, TaskRunSpecV1, TaskRunStatus
from agent_libos.utils.serde import dumps
from tests.runtime.test_responses_replay_executor import CONFIG, IMAGE, SECRET, ReplayClient, completion, register


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_context_object_refresh_preserves_replay_across_quanta(layout: str) -> None:
    config = replace(CONFIG, llm=replace(CONFIG.llm, prompt_layout=layout),
                     llm_context=replace(CONFIG.llm_context, policy="llm_context_object"))
    runtime = Runtime.open("local", config=config)
    try:
        client = ReplayClient([completion(1), completion(2), completion(3)])
        runtime.llm.client = client
        pid = register(runtime)
        versions = []
        for _ in range(3):
            outcome = runtime.run_process_once(pid)
            assert outcome["ok"], outcome
            _head, _turn, payload = runtime.llm.replay.load_current(pid)
            context_oid = runtime.llm.context_memory.context_oid(pid)
            refs = [ref for ref in DataFlowContext.from_dict(payload["flow_context"]).source_refs if ref.oid == context_oid]
            assert len(refs) == 1
            assert refs[0].version == runtime.store.get_object(context_oid).version
            versions.append(refs[0].version)
        assert versions[0] < versions[1] < versions[2]
        assert sum(item.get("encrypted_content") == SECRET + "1" for item in client.inputs[2]) == 1
        assert sum(item.get("call_id") == "call_1" and item.get("type") == "function_call_output" for item in client.inputs[2]) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["libos_default", "image_only"])
@pytest.mark.parametrize("new_image", [False, True], ids=["same-image", "different-image"])
def test_exec_starts_replay_with_current_image_and_goal(mode: str, new_image: bool) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime, mode)
        assert runtime.run_process_once(pid)["ok"]
        old_head = runtime.store.get_llm_replay_head(pid)
        target = IMAGE
        if new_image:
            target = "responses-replay-new:v0"
            runtime.register_image(AgentImage(image_id=target, name="new replay image",
                system_prompt="Follow the replacement image instructions.", default_tools=["echo"], prompt_mode=mode), actor="test")
            runtime.capability.grant(pid, f"image:{target}", [CapabilityRight.READ], issued_by="test")
        runtime.exec_process(pid, target, goal="NEW_GOAL_AFTER_EXEC_SENTINEL")
        assert runtime.store.get_llm_replay_head(pid) is None
        assert runtime.store.get_llm_replay_turn(old_head.turn_id).payload is not None
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        wire = dumps(client.inputs[1])
        assert "NEW_GOAL_AFTER_EXEC_SENTINEL" in wire
        assert SECRET + "1" not in wire
        assert not any(item.get("call_id") == "call_1" for item in client.inputs[1])
        if mode == "image_only":
            assert "Complete both steps." not in wire
        if new_image:
            assert "Follow the replacement image instructions." in wire
    finally:
        runtime.close()


def test_failed_exec_keeps_replay_head_and_context_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime, "image_only")
        assert runtime.run_process_once(pid)["ok"]
        head = runtime.store.get_llm_replay_head(pid)
        generation = runtime.store.get_llm_context_generation(pid)
        advance = runtime.store.advance_runtime_publication

        def fail_commit(publication_id, **kwargs):
            if kwargs.get("state") == "committed":
                raise RuntimeError("injected exec commit failure")
            return advance(publication_id, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(runtime.store, "advance_runtime_publication", fail_commit)
            with pytest.raises(Exception, match="injected exec commit failure"):
                runtime.exec_process(pid, IMAGE, goal="ROLLED_BACK_GOAL")
        assert runtime.store.get_llm_replay_head(pid) == head
        assert runtime.store.get_llm_context_generation(pid) == generation
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert SECRET + "1" in dumps(client.inputs[1])
        assert "ROLLED_BACK_GOAL" not in dumps(client.inputs[1])
    finally:
        runtime.close()


def test_model_exec_settles_old_tool_output_and_starts_new_replay() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        runtime.register_image(AgentImage(image_id=IMAGE, name="model exec",
            system_prompt="Use the tools.", default_tools=["exec_process", "echo"], prompt_mode="image_only"), actor="test")
        pid = runtime.process.spawn(image=IMAGE, goal="Replace this goal.")
        client = ReplayClient([completion(1, name="exec_process", arguments={"image": IMAGE, "goal": "MODEL_EXEC_NEW_GOAL"}), completion(2)])
        runtime.llm.client = client
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert runtime.store.get_llm_replay_head(pid) is None
        call = runtime.store.get_latest_successful_llm_call(pid=pid, purpose="action_selection")
        assert runtime.store.list_llm_tool_outputs(pid=pid, response_id=call.call_id)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert "MODEL_EXEC_NEW_GOAL" in dumps(client.inputs[1])
        assert SECRET + "1" not in dumps(client.inputs[1])
    finally:
        runtime.close()


def test_taskrun_exec_reopens_without_changing_durable_context_generation(tmp_path: Path) -> None:
    config = replace(CONFIG, task_runs=replace(CONFIG.task_runs, plaintext_payloads_enabled=True))
    target = str(tmp_path / "taskrun-exec.sqlite")
    runtime = Runtime.open(target, config=config)
    try:
        runtime.register_image(AgentImage(image_id=IMAGE, name="durable exec",
            system_prompt="Use the tools.", default_tools=["exec_process", "echo"], prompt_mode="image_only"), actor="test")
        run = runtime.task_runs.create(TaskRunSpecV1(goal="Continue after exec.", display_title="Exec replay", image_id=IMAGE), client_request_id="exec-run")
        generation = runtime.store.get_llm_context_generation(run.root_pid)
        runtime.llm.client = ReplayClient([completion(1, name="exec_process", arguments={"image": IMAGE})])
        outcome = runtime.task_runs.run_until_blocked(run.run_id, expected_revision=run.revision, command_id="exec", max_quanta=1)
        assert outcome.status is TaskRunStatus.RUNNING, outcome.blockers
        assert runtime.store.get_llm_replay_head(run.root_pid) is None
        assert runtime.store.get_llm_context_generation(run.root_pid) == generation
    finally:
        runtime.close()
    reopened = Runtime.open(target, config=config)
    try:
        client = ReplayClient([completion(2)])
        reopened.llm.client = client
        current = reopened.task_runs.get(run.run_id)
        outcome = reopened.task_runs.run_until_blocked(run.run_id, expected_revision=current.revision, command_id="continue", max_quanta=1)
        assert outcome.status is TaskRunStatus.RUNNING, outcome.blockers
        assert len(client.inputs) == 1
        assert SECRET + "1" not in dumps(client.inputs[0])
        assert reopened.store.get_llm_replay_head(run.root_pid) is not None
    finally:
        reopened.close()


def test_repeated_exec_before_next_call_uses_the_latest_goal() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime, "image_only")
        assert runtime.run_process_once(pid)["ok"]
        runtime.exec_process(pid, IMAGE, goal="INTERMEDIATE_EXEC_GOAL")
        runtime.exec_process(pid, IMAGE, goal="LATEST_EXEC_GOAL")
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        wire = dumps(client.inputs[1])
        assert "LATEST_EXEC_GOAL" in wire
        assert "INTERMEDIATE_EXEC_GOAL" not in wire
        assert SECRET + "1" not in wire
    finally:
        runtime.close()


def test_non_replay_pending_wait_reopens_after_profile_removal(tmp_path: Path) -> None:
    class WaitClient:
        async def acomplete_action(self, messages, tools, **kwargs):
            return LLMCompletion(content="", tool_calls=[{"id": "wait_call", "name": "receive_process_messages", "arguments": "{}"}])

    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        "default": LLMProfile(model="legacy", responses_replay=False),
        "old": LLMProfile(model="legacy", responses_replay=False),
    }))
    target = str(tmp_path / "non-replay-wait.sqlite")
    runtime = Runtime.open(target, config=config)
    try:
        register(runtime)
        pid = runtime.process.spawn(image=IMAGE, goal="Wait for a message.", llm_profile_id="old")
        runtime.llms.set_test_client("old", WaitClient())
        outcome = runtime.run_process_once(pid)
        assert outcome.get("waiting_message"), outcome
        assert runtime.store.get_llm_replay_head(pid) is None
        assert runtime.store.get_llm_pending_action(pid)["status"] == "pending"
    finally:
        runtime.close()
    reopened = Runtime.open(target, config=replace(config, llm=replace(config.llm, profiles={"default": config.llm.profiles["default"]})))
    try:
        assert reopened.store.get_llm_pending_action(pid)["status"] == "pending"
        assert reopened.process.get(pid).llm_profile_id == "old"
    finally:
        reopened.close()
