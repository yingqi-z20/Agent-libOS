from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.models import PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT
from agent_libos.utils.serde import dumps


IMAGE = "responses-replay-test:v0"
SECRET = "OPAQUE_PROVIDER_STATE_MUST_STAY_PRIVATE"
CONFIG = replace(
    DEFAULT_CONFIG,
    llm=replace(
        DEFAULT_CONFIG.llm,
        profiles={"default": LLMProfile(model="gpt-6-astra", api_mode="responses", responses_replay=True)},
    ),
)


def completion(index: int, *, name: str = "echo", arguments: dict[str, Any] | None = None) -> LLMCompletion:
    args = json.dumps(arguments if arguments is not None else {"message": f"step {index}"})
    return LLMCompletion(
        content="",
        tool_calls=[{"id": f"fc_{index}", "call_id": f"call_{index}", "name": name, "arguments": args}],
        api="responses", response_id=f"resp_{index}", model="gpt-6-astra",
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "reasoning_tokens": 12},
        response_items=[
            {"type": "reasoning", "id": f"rs_{index}", "summary": [], "encrypted_content": SECRET + str(index)},
            {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}", "name": name, "arguments": args, "status": "completed"},
        ],
    )


class ReplayClient(LLMClient):
    def __init__(self, completions: list[LLMCompletion]) -> None:
        super().__init__(model="gpt-6-astra", api_key="test", api_mode="responses", responses_replay=True)
        self.completions = list(completions)
        self.inputs: list[list[dict[str, Any]]] = []

    async def acomplete_action(self, messages: Any, tools: Any, **kwargs: Any) -> LLMCompletion:
        assert kwargs.get("previous_response_id") is None
        assert kwargs.get("responses_items") is not None
        self.inputs.append(deepcopy(kwargs["responses_items"]))
        return self.completions.pop(0)


class LegacyResponsesClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(model="gpt-6-astra", api_key="test", api_mode="responses", responses_replay=False)

    async def acomplete_action(self, messages: Any, tools: Any, **kwargs: Any) -> LLMCompletion:
        assert kwargs.get("responses_items") is None
        return completion(1)


def register(runtime: Runtime, mode: str = PROMPT_MODE_LIBOS_DEFAULT) -> str:
    runtime.register_image(AgentImage(
        image_id=IMAGE, name="responses replay", system_prompt="Use the provided tools.",
        default_tools=["echo", "process_exit", "receive_process_messages"], prompt_mode=mode,
    ), actor="test")
    return runtime.process.spawn(image=IMAGE, goal="Complete both steps.")


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
def test_runtime_replays_reasoning_and_tool_result_once_without_public_ciphertext(mode: str) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime, mode)
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.run_process_once(pid)["ok"]
        items = client.inputs[1]
        assert sum(item.get("encrypted_content") == SECRET + "1" for item in items) == 1
        assert sum(item.get("type") == "function_call" and item.get("call_id") == "call_1" for item in items) == 1
        assert sum(item.get("type") == "function_call_output" and item.get("call_id") == "call_1" for item in items) == 1
        assert items[0]["role"] == "system"
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.events.list(target=pid))
        assert runtime.store.get_llm_replay_head(pid) is not None
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_total_tokens == 240
    finally:
        runtime.close()


def test_runtime_reopens_private_reasoning_without_repeating_provider_or_tool(tmp_path: Path) -> None:
    target = str(tmp_path / "replay.sqlite")
    runtime = Runtime.open(target, config=CONFIG)
    client = ReplayClient([completion(1)])
    runtime.llm.client = client
    pid = register(runtime)
    assert runtime.run_process_once(pid)["ok"]
    runtime.close()
    reopened = Runtime.open(target, config=CONFIG)
    try:
        later = ReplayClient([completion(2)])
        reopened.llm.client = later
        assert reopened.run_process_once(pid)["ok"]
        assert len(later.inputs) == 1
        assert any(item.get("encrypted_content") == SECRET + "1" for item in later.inputs[0])
        assert len(reopened.store.list_llm_calls(pid=pid)) == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("reopen_before_replay", [False, True], ids=["live", "reopen"])
def test_first_image_only_replay_seeds_legacy_tool_history_once(
    tmp_path: Path, reopen_before_replay: bool,
) -> None:
    target = str(tmp_path / "legacy-image-only.sqlite")
    legacy_config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        "default": replace(CONFIG.llm.profiles["default"], responses_replay=False),
    }))
    runtime = Runtime.open(target, config=legacy_config)
    try:
        runtime.llm.client = LegacyResponsesClient()
        pid = register(runtime, PROMPT_MODE_IMAGE_ONLY)
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.store.get_llm_replay_head(pid) is None
        record = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert record.request_options["image_only_transcript"]
        if reopen_before_replay:
            runtime.close()
            runtime = Runtime.open(target, config=CONFIG)
        else:
            runtime.llms.register_profile("default", CONFIG.llm.profiles["default"])
        client = ReplayClient([completion(2), completion(3)])
        runtime.llm.client = client

        for _ in range(2):
            outcome = runtime.run_process_once(pid)
            assert outcome["ok"], outcome

        assert len(client.inputs) == 2
        for index, items in enumerate(client.inputs, start=1):
            calls = [item for item in items if item.get("type") == "function_call"]
            results = [item for item in items if item.get("type") == "function_call_output"]
            expected_ids = [f"call_{turn}" for turn in range(1, index + 1)]
            assert [item["call_id"] for item in calls] == expected_ids
            assert [item["call_id"] for item in results] == expected_ids
            assert json.loads(calls[0]["arguments"]) == {"message": "step 1"}
            assert "step 1" in results[0]["output"]
            assert sum(item.get("content") == "Complete both steps." for item in items) == 1
            assert sum(item.get("role") == "system" for item in items) == 1
        assert len(runtime.store.list_llm_calls(pid=pid)) == 3
    finally:
        runtime.close()


def test_first_image_only_replay_rejects_incomplete_legacy_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        runtime.llm.client = LegacyResponsesClient()
        pid = register(runtime, PROMPT_MODE_IMAGE_ONLY)
        assert runtime.run_process_once(pid)["ok"]
        client = ReplayClient([completion(2)])
        runtime.llm.client = client
        monkeypatch.setattr(runtime.store, "list_llm_tool_outputs", lambda **_kwargs: [])

        outcome = runtime.run_process_once(pid)

        assert not outcome["ok"]
        assert "image_only" in outcome["error"]
        assert not client.inputs
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()


def test_runtime_replay_refuses_changed_provider_before_network() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        runtime.llms.register_profile("default", LLMProfile(
            model="gpt-6-astra", api_mode="responses", responses_replay=True,
            reasoning_context="current_turn",
        ))
        runtime.llm.client = client
        result = runtime.run_process_once(pid)
        assert not result["ok"]
        assert len(client.inputs) == 1
    finally:
        runtime.close()


def test_runtime_replay_repairs_invalid_action_without_orphan_tool_items() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1, name="missing_tool"), completion(2), completion(3)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.run_process_once(pid)["ok"]
        assert len(client.inputs) == 3
        assert all(item.get("call_id") != "call_1" for item in client.inputs[2])
        assert any(item.get("call_id") == "call_2" for item in client.inputs[2])
    finally:
        runtime.close()


def test_runtime_local_checkpoint_restores_replay_for_next_provider_call() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2), completion(3)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        checkpoint_id = runtime.checkpoint.create(pid, "after first call", actor=pid)
        assert runtime.run_process_once(pid)["ok"]
        restored = runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert restored["main_state_committed"]
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert any(item.get("encrypted_content") == SECRET + "1" for item in client.inputs[2])
        assert not any(item.get("encrypted_content") == SECRET + "2" for item in client.inputs[2])
    finally:
        runtime.close()


def test_runtime_no_full_io_never_persists_private_responses_state() -> None:
    class NoRetentionClient(ReplayClient):
        async def acomplete_action(self, messages: Any, tools: Any, **kwargs: Any) -> LLMCompletion:
            assert kwargs.get("responses_items") is None
            return self.completions.pop(0)

    runtime = Runtime.open("local", config=replace(CONFIG, llm=replace(CONFIG.llm, persist_full_io=False)))
    try:
        runtime.llm.client = NoRetentionClient([completion(1)])
        pid = register(runtime)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert runtime.store.get_llm_replay_head(pid) is None
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()


def test_runtime_preserves_configured_auto_cache_and_private_domain_evidence() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1)])
        client.prompt_cache_mode_configured = "auto"
        client.prompt_cache_mode = "implicit"
        client.prompt_cache_key_source = "host_generated"
        client.prompt_cache_key = "PRIVATE_HOST_CACHE_DOMAIN"
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        record = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert record.request_options["openai_prompt_cache_mode_configured"] == "auto"
        assert record.request_options["openai_prompt_cache_key_source"] == "host_generated"
        assert record.request_options["openai_prompt_cache_key_configured"] is False
        assert "PRIVATE_HOST_CACHE_DOMAIN" not in dumps(record)
    finally:
        runtime.close()


def test_terminal_replay_owner_does_not_require_retired_profile(tmp_path: Path) -> None:
    from agent_libos.models import ProcessStatus

    target = tmp_path / "retired-profile.sqlite"
    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        **CONFIG.llm.profiles, "retired": CONFIG.llm.profiles["default"],
    }))
    runtime = Runtime.open(target, config=config)
    try:
        runtime.register_image(AgentImage(
            image_id=IMAGE, name="terminal replay", system_prompt="Exit.",
            default_tools=["process_exit"],
        ), actor="test")
        pid = runtime.process.spawn(image=IMAGE, goal="finish", llm_profile_id="retired")
        runtime.llms.set_test_client("retired", ReplayClient([completion(1, name="process_exit", arguments={"payload": {"done": True}})]))
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.process.get(pid).status is ProcessStatus.EXITED
        assert runtime.store.get_llm_replay_head(pid) is not None
    finally:
        runtime.close()
    reopened = Runtime.open(target, config=CONFIG)
    try:
        assert reopened.process.get(pid).status is ProcessStatus.EXITED
        assert reopened.store.get_llm_replay_head(pid) is not None
    finally:
        reopened.close()
