from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile, ProviderToolsConfig
from agent_libos.models import PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT
from agent_libos.utils.serde import dumps


_IMAGE = "provider-continuation-test:v0"
_RESULT = "PROVIDER_CONTINUATION_PRIVATE_RESULT_481"


class _Responses:
    """Script wire responses while retaining the real client and executor."""

    def __init__(self, *steps: str, code: bool = False) -> None:
        self.steps = list(steps)
        self.code = code
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(deepcopy(request))
        assert self.steps, "unexpected extra provider dispatch"
        step = self.steps.pop(0)
        index = len(self.requests)
        output: list[dict[str, Any]] = []
        if step in {"hosted", "activity_only"}:
            if self.code:
                output.append({
                    "type": "code_interpreter_call", "id": f"ci_{index}",
                    "status": "completed", "container_id": f"container_{index}",
                    "code": "print(481)", "outputs": [{"type": "logs", "logs": _RESULT}],
                })
            else:
                output.append({
                    "type": "web_search_call", "id": f"ws_{index}", "status": "completed",
                    "action": {"type": "search", "query": _RESULT},
                })
        content = _RESULT if step in {"hosted", "text_only"} else ""
        annotations = []
        if step == "citation_only":
            annotations = [{
                "type": "url_citation", "url": "https://example.test/reference-only",
                "title": "Reference-only result", "start_index": 0, "end_index": 0,
            }]
        elif step == "artifact_only":
            annotations = [{
                "type": "container_file_citation", "file_id": "file-artifact-only",
                "container_id": "container-reference-only", "filename": "artifact-only.txt",
                "start_index": 0, "end_index": 0,
            }]
        if content or annotations:
            output.append({
                "type": "message", "role": "assistant", "content": [{
                    "type": "output_text", "text": content, "annotations": annotations,
                }],
            })
        if step in {"action", "invalid_action"}:
            output.append({
                "type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                "name": "missing_runtime_tool" if step == "invalid_action" else "echo",
                "arguments": '{"message":"local action completed"}', "status": "completed",
            })
        return SimpleNamespace(
            id=f"resp_{index}", model="gpt-test", status="completed", output=output,
            output_text=content,
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


class _Chat:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(deepcopy(request))
        index = len(self.requests)
        calls = [] if index == 1 else [SimpleNamespace(
            id=f"call_{index}", type="function",
            function=SimpleNamespace(name="echo", arguments='{"message":"local action completed"}'),
        )]
        return SimpleNamespace(
            id=f"chat_{index}", model="qwen-test",
            choices=[SimpleNamespace(
                finish_reason="stop" if index == 1 else "tool_calls",
                message=SimpleNamespace(content=_RESULT if index == 1 else "", tool_calls=calls),
            )],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )


def _config(*, code: bool = False, chat: bool = False, full_io: bool = True):
    return replace(DEFAULT_CONFIG, llm=replace(
        DEFAULT_CONFIG.llm,
        persist_full_io=full_io,
        profiles={"default": LLMProfile(
            model="qwen-test" if chat else "gpt-test",
            api_mode="chat" if chat else "responses",
            responses_replay=not code and not chat,
            provider_tools=ProviderToolsConfig(
                provider="aliyun" if chat else "openai",
                web_search=not code, code_interpreter=code,
            ),
        )},
    ))


def _capture(runtime: Runtime, endpoint: _Responses | _Chat) -> None:
    client = runtime.llms.resolve("default").client
    client._async_client = (
        SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        if isinstance(endpoint, _Chat) else SimpleNamespace(responses=endpoint)
    )


def _spawn(runtime: Runtime, mode: str = PROMPT_MODE_LIBOS_DEFAULT) -> str:
    runtime.register_image(AgentImage(
        image_id=_IMAGE, name="provider continuation",
        system_prompt="Research the request, then use an available local function.",
        default_tools=["echo"], prompt_mode=mode,
    ), actor="test")
    return runtime.process.spawn(image=_IMAGE, goal="Research and report the result.")


def _assert_local_only(request: dict[str, Any]) -> None:
    assert request["tools"]
    assert all(tool["type"] == "function" for tool in request["tools"])
    assert request.get("extra_body", {}).get("enable_search") is not True


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
@pytest.mark.parametrize("first_step", ["hosted", "activity_only"])
def test_provider_only_response_continues_next_quantum_with_local_functions_only(mode: str, first_step: str) -> None:
    runtime = Runtime.open("local", config=_config())
    try:
        provider = _Responses(first_step, "action", "action")
        _capture(runtime, provider)
        pid = _spawn(runtime, mode)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        assert len(provider.requests) == 1
        assert any(tool["type"] == "web_search" for tool in provider.requests[0]["tools"])
        assert not any(row.action == "llm.action_repair_requested" for row in runtime.audit.trace(actor=pid))
        second = runtime.run_process_once(pid)
        assert second["ok"] and second["action"]["action"] == "echo", second
        assert len(provider.requests) == 2
        _assert_local_only(provider.requests[1])
        assert _RESULT in dumps(provider.requests[1]["input"])
        third = runtime.run_process_once(pid)
        assert third["ok"], third
        assert any(tool["type"] == "web_search" for tool in provider.requests[2]["tools"])
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
@pytest.mark.parametrize("code", [False, True], ids=["search", "code"])
def test_provider_continuation_survives_reopen_without_repeating_hosted_tools(tmp_path: Path, mode: str, code: bool) -> None:
    database = tmp_path / "provider-continuation.sqlite"
    config = _config(code=code)
    runtime = Runtime.open(database, config=config)
    try:
        first_provider = _Responses("hosted", code=code)
        _capture(runtime, first_provider)
        pid = _spawn(runtime, mode)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"] and outcome.get("provider_continuation") is True, outcome
        assert len(first_provider.requests) == 1
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        next_provider = _Responses("action")
        _capture(reopened, next_provider)
        outcome = reopened.run_process_once(pid)
        assert outcome["ok"] and outcome["action"]["action"] == "echo", outcome
        assert len(next_provider.requests) == 1
        _assert_local_only(next_provider.requests[0])
        assert _RESULT in dumps(next_provider.requests[0]["input"])
        if code:
            assert next_provider.requests[0].get("previous_response_id") is None
            assert not any(item.get("type") == "code_interpreter_call" for item in next_provider.requests[0]["input"])
    finally:
        reopened.close()


def test_existing_native_replay_cannot_switch_to_independent_code_execution() -> None:
    runtime = Runtime.open("local", config=_config())
    try:
        search = _Responses("action")
        _capture(runtime, search)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid)["ok"]
        head = runtime.store.get_llm_replay_head(pid)
        assert head is not None
        runtime.llms.register_profile("default", _config(code=True).llm.profiles["default"])
        code = _Responses(code=True)
        _capture(runtime, code)
        outcome = runtime.run_process_once(pid)
        assert not outcome["ok"], outcome
        assert "native replay" in outcome["error"]
        assert not code.requests
        assert runtime.store.get_llm_replay_head(pid) == head
    finally:
        runtime.close()


def test_pending_stateless_result_rejects_removing_profile_tools_before_provider_dispatch() -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        provider = _Responses("hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        assert runtime.store.get_llm_replay_head(pid) is None
        source_before = dumps(runtime.store.get_llm_call(first["call_id"]))
        marker_before = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        profile = runtime.llms.profile("default")
        runtime.llms.register_profile("default", replace(profile, provider_tools=None))
        later = _Responses()
        _capture(runtime, later)

        outcome = runtime.run_process_once(pid)

        assert not outcome["ok"], outcome
        assert "provider continuation profile changed" in outcome["error"]
        assert not later.requests
        assert dumps(runtime.store.get_llm_call(first["call_id"])) == source_before
        assert runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation") == marker_before
        assert marker_before.request_options["provider_continuation"]["state"] == "pending"
    finally:
        runtime.close()


def test_aliyun_chat_text_only_result_continues_with_execution_unknown() -> None:
    runtime = Runtime.open("local", config=_config(chat=True))
    try:
        provider = _Chat()
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        record = runtime.store.get_llm_call(first["call_id"])
        assert record is not None
        attempt = record.reasoning["attempts"][record.reasoning["selected_attempt"] - 1]
        assert attempt["provider_tools"]["observed"] == "unknown"
        assert provider.requests[0]["extra_body"]["enable_search"] is True
        second = runtime.run_process_once(pid)
        assert second["ok"] and second["action"]["action"] == "echo", second
        assert len(provider.requests) == 2
        _assert_local_only(provider.requests[1])
        assert _RESULT in dumps(provider.requests[1]["messages"])
    finally:
        runtime.close()


def test_provider_continuation_repair_keeps_hosted_tools_disabled() -> None:
    runtime = Runtime.open("local", config=_config())
    try:
        provider = _Responses("hosted", "invalid_action", "action")
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"] and outcome["action"]["action"] == "echo", outcome
        assert len(provider.requests) == 3
        for request in provider.requests[1:]:
            _assert_local_only(request)
            assert _RESULT in dumps(request["input"])
    finally:
        runtime.close()


@pytest.mark.parametrize(("step", "expected_text"), [
    ("citation_only", "Source: Reference-only result https://example.test/reference-only"),
    ("artifact_only", "Artifact: artifact-only.txt (file ID: file-artifact-only)"),
])
def test_reference_only_provider_result_continues_without_repeating_hosted_tools(step: str, expected_text: str) -> None:
    config = _config(code=step == "artifact_only")
    config = replace(config, llm=replace(config.llm, profiles={
        "default": replace(config.llm.profiles["default"], responses_replay=False),
    }))
    runtime = Runtime.open("local", config=config)
    try:
        provider = _Responses(step, "action", code=step == "artifact_only")
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        assert len(provider.requests) == 1
        source = runtime.store.get_llm_call(first["call_id"])
        assert source is not None and source.response_content == "" and source.tool_calls == []
        attempt = source.reasoning["attempts"][source.reasoning["selected_attempt"] - 1]
        assert attempt["provider_tools"]["activities"] == []
        assert source.request_options["provider_continuation_required"] is True

        second = runtime.run_process_once(pid)

        assert second["ok"] and second["action"]["action"] == "echo", second
        assert len(provider.requests) == 2
        _assert_local_only(provider.requests[1])
        assert expected_text in dumps(provider.requests[1]["input"])
        assert "container-reference-only" not in dumps(provider.requests[1]["input"])
        assert not any(row.action == "llm.action_repair_requested" for row in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


def test_content_free_continuation_uses_ephemeral_result_without_durable_leak() -> None:
    runtime = Runtime.open("local", config=_config(code=True, full_io=False))
    try:
        provider = _Responses("hosted", "action", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        second = runtime.run_process_once(pid)
        assert second["ok"] and second["action"]["action"] == "echo", second
        assert len(provider.requests) == 2
        _assert_local_only(provider.requests[1])
        assert _RESULT in dumps(provider.requests[1]["input"])
        assert _RESULT not in dumps(runtime.store.list_llm_calls(pid=pid))
        assert _RESULT not in dumps(runtime.audit.trace(actor=pid))
        assert _RESULT not in dumps(runtime.events.list(target=pid))
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()


def test_content_free_continuation_after_reopen_fails_closed_without_reexecuting_code(tmp_path: Path) -> None:
    database = tmp_path / "content-free-continuation.sqlite"
    config = _config(code=True, full_io=False)
    runtime = Runtime.open(database, config=config)
    try:
        first_provider = _Responses("hosted", code=True)
        _capture(runtime, first_provider)
        pid = _spawn(runtime)
        outcome = runtime.run_process_once(pid)
        assert outcome["ok"] and outcome.get("provider_continuation") is True, outcome
        assert _RESULT not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        later = _Responses(code=True)
        _capture(reopened, later)
        outcome = reopened.run_process_once(pid)
        assert not outcome["ok"], outcome
        assert "retention" in outcome["error"]
        assert not later.requests
        assert _RESULT not in dumps(reopened.store.list_llm_calls(pid=pid))
    finally:
        reopened.close()


@pytest.mark.parametrize("reopen", [False, True], ids=["same-runtime", "reopen"])
@pytest.mark.parametrize("full_io", [False, True], ids=["content-free", "full-io"])
def test_successful_hosted_result_without_continuation_marker_blocks_repeated_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reopen: bool, full_io: bool,
) -> None:
    database = tmp_path / "continuation-settlement-gap.sqlite"
    config = _config(code=True, full_io=full_io)
    runtime = Runtime.open(database, config=config)
    try:
        def preserve_resumable_process(_pid: str, error: Exception) -> dict[str, Any]:
            # Keep the process schedulable as after an abrupt Host interruption.
            # Terminal-status rejection would otherwise mask the admission
            # guard, including its lookup past subsequently recorded errors.
            return {"ok": False, "error": str(error)}

        monkeypatch.setattr(runtime.llm, "_fail_llm_quantum", preserve_resumable_process)
        provider = _Responses("hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)

        def fail_marker_commit(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("injected provider continuation settlement failure")

        with monkeypatch.context() as patch:
            patch.setattr(runtime.llm, "_commit_provider_continuation", fail_marker_commit)
            initial = runtime.run_process_once(pid)

        assert not initial["ok"], initial
        assert len(provider.requests) == 1
        source = runtime.store.get_latest_successful_llm_call(pid=pid, purpose="action_selection")
        assert source is not None
        assert source.request_options["provider_continuation_required"] is True
        source_before = dumps(source)
        assert runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation") is None
        if reopen:
            runtime.close()
            runtime = Runtime.open(database, config=config)
            monkeypatch.setattr(runtime.llm, "_fail_llm_quantum", preserve_resumable_process)
        later = _Responses(code=True)
        _capture(runtime, later)

        for _ in range(2):
            outcome = runtime.run_process_once(pid)
            assert not outcome["ok"], outcome
            assert not later.requests
            assert "awaiting continuation settlement" in outcome.get("error", ""), outcome
            assert dumps(runtime.store.get_llm_call(source.call_id)) == source_before
            latest = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
            assert latest is not None and latest.status == "error"
            latest_success = runtime.store.get_latest_successful_llm_call(pid=pid, purpose="action_selection")
            assert latest_success is not None and latest_success.call_id == source.call_id
        assert runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation") is None
    finally:
        runtime.close()
