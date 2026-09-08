from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile, ProviderToolsConfig
from agent_libos.models import (
    ObjectMetadata,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_LIBOS_DEFAULT,
    SinkTrustLevel,
    SinkTrustRule,
    ViewMode,
)
from agent_libos.utils.serde import dumps


_PRIVATE = "HOSTED_TOOL_PRIVATE_SENTINEL"
_IMAGE = "provider-tools-governance:v0"


class _Responses:
    """Capture the actual SDK request, with no network or custom LLM client."""

    def __init__(self, *, code: bool, repair_first: bool = False, exit_action: bool = False) -> None:
        self.code = code
        self.repair_first = repair_first
        self.exit_action = exit_action
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(deepcopy(request))
        index = len(self.requests)
        enabled = {tool["type"] for tool in request.get("tools", [])}
        output: list[dict[str, Any]] = []
        if self.code and "code_interpreter" in enabled:
            output.append({
                "type": "code_interpreter_call",
                "id": f"ci_{index}",
                "status": "completed",
                "container_id": f"cntr_private_{index}",
                "code": f"print('{_PRIVATE}')",
                "outputs": [{"type": "logs", "logs": _PRIVATE}],
            })
        elif "web_search" in enabled:
            output.append({
                "type": "web_search_call",
                "id": f"ws_{index}",
                "status": "completed",
                "action": {"type": "search", "query": _PRIVATE},
            })
        output.append({
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": "not_a_runtime_tool" if self.repair_first and index == 1 else "process_exit" if self.exit_action else "echo",
            "arguments": '{"payload":{"summary":"done"}}' if self.exit_action else '{"message":"finished provider work"}',
            "status": "completed",
        })
        return SimpleNamespace(
            id=f"resp_{index}", model="gpt-test", status="completed",
            output=output, output_text="",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


def _config(*, code: bool = False, search: bool = False, full_io: bool = True):
    return replace(DEFAULT_CONFIG, llm=replace(
        DEFAULT_CONFIG.llm,
        persist_full_io=full_io,
        profiles={"default": LLMProfile(
            model="gpt-test", api_mode="responses", responses_replay=True,
            provider_tools=(ProviderToolsConfig(
                provider="openai", web_search=search, code_interpreter=code,
            ) if code or search else None),
        )},
    ))


def _capture(runtime: Runtime, responses: _Responses) -> None:
    runtime.llms.resolve("default").client._async_client = SimpleNamespace(
        responses=responses,
    )


def _spawn(runtime: Runtime, *, mode: str = PROMPT_MODE_LIBOS_DEFAULT) -> str:
    if runtime.images.get(_IMAGE) is None:
        runtime.register_image(AgentImage(
            image_id=_IMAGE, name="provider governance",
            system_prompt="Use an available function after completing your work.",
            default_tools=["echo", "process_exit"], prompt_mode=mode,
        ), actor="test")
    return runtime.process.spawn(image=_IMAGE, goal="Calculate and report the result.")


def _assert_independent_code_request(request: dict[str, Any]) -> None:
    assert request.get("previous_response_id") is None
    code = next(tool for tool in request["tools"] if tool["type"] == "code_interpreter")
    assert code["container"]["type"] == "auto"
    serialized_input = dumps(request["input"])
    assert '"type":"code_interpreter_call"' not in serialized_input.replace(" ", "")
    assert '"type":"item_reference"' not in serialized_input.replace(" ", "")
    assert "encrypted_content" not in serialized_input


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
def test_code_requests_never_reuse_native_container_history_across_processes(mode: str) -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        responses = _Responses(code=True)
        _capture(runtime, responses)
        first, second = _spawn(runtime, mode=mode), _spawn(runtime, mode=mode)
        for pid in (first, second, first):
            outcome = runtime.run_process_once(pid)
            assert outcome["ok"], outcome
            assert runtime.store.get_llm_replay_head(pid) is None
        assert len(responses.requests) == 3
        for request in responses.requests:
            _assert_independent_code_request(request)
        assert "cntr_private_1" not in dumps(responses.requests[1]["input"])
        assert "cntr_private_2" not in dumps(responses.requests[2]["input"])
    finally:
        runtime.close()


def test_code_request_after_reopen_has_no_container_continuation(tmp_path: Path) -> None:
    database = tmp_path / "independent-code.sqlite"
    config = _config(code=True)
    runtime = Runtime.open(database, config=config)
    try:
        first = _Responses(code=True)
        _capture(runtime, first)
        pid = _spawn(runtime, mode=PROMPT_MODE_IMAGE_ONLY)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        later = _Responses(code=True)
        _capture(reopened, later)
        outcome = reopened.run_process_once(pid)
        assert outcome["ok"], outcome
        assert len(later.requests) == 1
        _assert_independent_code_request(later.requests[0])
        assert reopened.store.get_llm_replay_head(pid) is None
    finally:
        reopened.close()


@pytest.mark.parametrize("operation", ["restore", "fork"])
def test_checkpoint_recovery_never_restores_provider_container_state(operation: str) -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        responses = _Responses(code=True)
        _capture(runtime, responses)
        pid = _spawn(runtime, mode=PROMPT_MODE_IMAGE_ONLY)
        first = runtime.run_process_once(pid)
        assert first["ok"], first
        checkpoint_id = runtime.checkpoint.create(pid, "independent code history", actor=pid)
        snapshot = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert snapshot is not None
        assert not snapshot[1].get("responses_replay_refs")
        second = runtime.run_process_once(pid)
        assert second["ok"], second
        if operation == "restore":
            restored = runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
            assert restored["main_state_committed"]
            selected_pid = pid
        else:
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id, require_capability=False)
            selected_pid = forked["fork_root_pid"]
            assert selected_pid != pid
        outcome = runtime.run_process_once(selected_pid)
        assert outcome["ok"], outcome
        assert len(responses.requests) == 3
        _assert_independent_code_request(responses.requests[-1])
        assert "cntr_private_2" not in dumps(responses.requests[-1]["input"])
        assert runtime.store.get_llm_replay_head(selected_pid) is None
    finally:
        runtime.close()


def test_hosted_code_result_repair_cannot_execute_provider_tools_again() -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        responses = _Responses(code=True, repair_first=True)
        _capture(runtime, responses)
        pid = _spawn(runtime)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert len(responses.requests) == 2
        assert any(tool["type"] == "code_interpreter" for tool in responses.requests[0]["tools"])
        assert all(tool["type"] == "function" for tool in responses.requests[1]["tools"])
        assert _PRIVATE in dumps(responses.requests[1]["input"])
        assert any(row.action == "llm.action_repair_requested" for row in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


def test_changed_provider_tools_cannot_inherit_existing_secret_sink_clearance() -> None:
    runtime = Runtime.open("local", config=_config())
    try:
        old_identity = runtime.llms.profile_identity_sha256("default")
        runtime.data_flow.register_sink_trust(SinkTrustRule(
            pattern="llm:default", trust_level=SinkTrustLevel.TRUSTED,
            max_sensitivity="secret", identity_sha256=old_identity,
        ), actor="test.host", require_capability=False)
        runtime.llms.register_profile("default", _config(search=True).llm.profiles["default"])
        assert runtime.llms.profile_identity_sha256("default") != old_identity
        responses = _Responses(code=False)
        _capture(runtime, responses)
        pid = _spawn(runtime)
        source = runtime.memory.create_object(
            pid, ObjectType.EVIDENCE, {"text": _PRIVATE},
            metadata=ObjectMetadata(sensitivity="secret"),
        )
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(pid, [source], mode=ViewMode.READ_ONLY)
        runtime.store.update_process(process)
        outcome = runtime.run_process_once(pid)
        assert not outcome["ok"], outcome
        assert not responses.requests
        assert runtime.store.list_external_effects(pid=pid) == []
        assert runtime.uow.resources.list_resource_usage_reservations(pid=pid) == []
        assert _PRIVATE not in dumps(runtime.audit.trace(actor=pid))
        assert any(
            row.action == "data_flow.egress" and row.decision.get("outcome") == "deny"
            for row in runtime.audit.trace(actor=pid)
        )
        assert any(
            event.payload.get("outcome") == "deny"
            for event in runtime.events.list(target="data_flow_sink:llm:default")
        )
    finally:
        runtime.close()


def test_context_compressor_cannot_inherit_hosted_tools_from_its_profile() -> None:
    runtime = Runtime.open("local", config=_config(code=True, search=True))
    try:
        responses = _Responses(code=True, exit_action=True)
        _capture(runtime, responses)
        pid = runtime.process.spawn(image="context-compressor:v0", goal="Summarize the supplied empty context.")
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert len(responses.requests) == 1
        assert all(tool["type"] == "function" for tool in responses.requests[0]["tools"])
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()


def test_internal_text_completion_never_inherits_profile_tools() -> None:
    runtime = Runtime.open("local", config=_config(code=True, search=True))
    try:
        requests: list[dict[str, Any]] = []

        async def complete(**request: Any) -> SimpleNamespace:
            requests.append(request)
            return SimpleNamespace(
                id="resp_internal", status="completed", model="gpt-test",
                output_text="summary", output=[{
                    "type": "message", "role": "assistant", "content":[{
                        "type": "output_text", "text": "summary", "annotations": [],
                    }],
                }],
            )

        client = runtime.llms.resolve("default").client
        client._async_client = SimpleNamespace(responses=SimpleNamespace(create=complete))
        assert asyncio.run(client.acomplete(
            [{"role": "user", "content": "Summarize only the supplied text."}],
            json_mode=False,
        )) == "summary"
        assert len(requests) == 1
        assert not requests[0].get("tools")
    finally:
        runtime.close()


def test_provider_tool_private_content_is_absent_when_full_io_is_disabled() -> None:
    runtime = Runtime.open("local", config=_config(code=True, full_io=False))
    try:
        responses = _Responses(code=True)
        _capture(runtime, responses)
        pid = _spawn(runtime)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"], outcome
        assert len(responses.requests) == 1
        assert _PRIVATE not in dumps(runtime.store.list_llm_calls(pid=pid))
        assert _PRIVATE not in dumps(runtime.audit.trace(actor=pid))
        assert _PRIVATE not in dumps(runtime.events.list(target=pid))
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()
