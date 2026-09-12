from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import LLMProfile, ProviderToolsConfig
from agent_libos.llm.client import LLMClient, LLMError
from agent_libos.llm.provider_tools import (
    PROVIDER_TOOL_MAX_ITEMS, PROVIDER_TOOL_TEXT_MAX_CHARS,
    provider_tool_result_text,
)


FUNCTION = {"type": "function", "function": {
    "name": "wait", "description": "Wait", "parameters": {"type": "object", "properties": {}},
}}
MESSAGES = [{"role": "system", "content": "Act carefully"}, {"role": "user", "content": "Research"}]


class Capture:
    def __init__(self, result: Any):
        self.result = result
        self.payloads: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(copy.deepcopy(payload))
        return self.result


def response(output: list[Any] | None = None, **extra: Any) -> Any:
    return SimpleNamespace(id="resp_1", model="gpt-test", status="completed", output_text="answer", output=output or [], **extra)


def make_client(provider: str = "openai", *, output: Any = None, **kwargs: Any) -> tuple[LLMClient, Capture]:
    capture = Capture(output or response())
    client = LLMClient(
        model="gpt-test" if provider == "openai" else "qwen3-max",
        api_key="test", api_mode="auto",
        base_url=None if provider == "openai" else "https://dashscope.aliyuncs.com/compatible-mode/v1",
        allow_custom_base_url=True, inherit_ambient_openai_sdk_config=False,
        **kwargs,
    )
    client._async_client = SimpleNamespace(responses=capture, chat=SimpleNamespace(completions=capture))
    return client, capture


def act(client: LLMClient, **kwargs: Any) -> Any:
    return asyncio.run(client.acomplete_action(MESSAGES, [FUNCTION], **kwargs))


def search(provider: str = "openai", **extra: Any) -> ProviderToolsConfig:
    return ProviderToolsConfig(provider=provider, web_search=True, **extra)


def search_item(**extra: Any) -> dict[str, Any]:
    return {"type": "web_search_call", "id": "ws_1", "status": "completed",
            "action": {"type": "search", "query": "weather"}, **extra}


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_auto_responses_merges_hosted_tools_with_functions(provider: str) -> None:
    client, captured = make_client(provider, provider_tools=search(provider))
    completion = act(client)
    assert client.api_mode == "responses"
    assert [item["type"] for item in captured.payloads[0]["tools"]] == ["function", "web_search"]
    tools = completion.provider_request_options["provider_tools"]
    assert tools["configured"] == tools["effective"] == ["web_search"]
    assert tools["observed"] == "unknown"
    if provider == "openai":
        assert captured.payloads[0]["include"] == ["web_search_call.action.sources"]
    else:
        assert "include" not in captured.payloads[0]


def test_aliyun_chat_search_keeps_explicit_search_when_thinking_rejected() -> None:
    client, captured = make_client("aliyun", provider_tools=search("aliyun"))
    client.api_mode = "chat"
    captured.result = SimpleNamespace(id="chat_1", model="qwen3-max", choices=[SimpleNamespace(
        message=SimpleNamespace(content="answer", tool_calls=[]), finish_reason="stop")])
    completion = act(client)
    assert captured.payloads[0]["extra_body"] == {"enable_search": True}
    retry = client._compatibility_retry_payload(
        {"extra_body": {"enable_search": True, "enable_thinking": False}},
        RuntimeError("unsupported enable_thinking"), "chat",
    )
    assert retry == {"extra_body": {"enable_search": True}}
    assert completion.provider_request_options["provider_tools"]["observed"] == "unknown"


@pytest.mark.parametrize("feature", ["web_extractor", "code_interpreter"])
def test_aliyun_required_thinking_cannot_be_downgraded(feature: str) -> None:
    client, captured = make_client("aliyun", provider_tools=search("aliyun", **{feature: True}))
    act(client)
    payload = captured.payloads[0]
    assert payload["extra_body"]["enable_thinking"] is True
    assert all("container" not in tool for tool in payload["tools"])
    assert client._compatibility_retry_payload(payload, RuntimeError("unsupported enable_thinking"), "responses") is None


@pytest.mark.parametrize("feature", ["web_extractor", "code_interpreter"])
def test_aliyun_required_thinking_conflict_is_rejected_at_profile_and_client_admission(feature: str) -> None:
    tools = search("aliyun", **{feature: True})
    with pytest.raises(ValueError, match="require thinking"):
        LLMProfile(provider_tools=tools, reasoning_effort="none")
    with pytest.raises(LLMError, match="require thinking"):
        make_client("aliyun", provider_tools=tools, reasoning_effort="none")


def test_code_container_is_independent_and_file_ids_only_in_tool_configuration() -> None:
    client, captured = make_client(provider_tools=ProviderToolsConfig(
        provider="openai", code_interpreter=True, file_ids=("file-first", "file-second"),
    ), responses_replay=True, store=True)
    act(client)
    act(client)
    assert client.responses_replay is False
    assert client.responses_previous_response_id is False
    assert len(captured.payloads) == 2
    for payload in captured.payloads:
        assert payload["tools"][-1] == {"type": "code_interpreter", "container": {
            "type": "auto", "file_ids": ["file-first", "file-second"],
        }}
        assert "previous_response_id" not in payload
        assert payload["include"] == ["code_interpreter_call.outputs"]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("history", [{"previous_response_id": "resp_old"}, {"responses_items": []}])
def test_code_rejects_native_history_even_when_temporarily_suppressed(enabled: bool, history: dict[str, Any]) -> None:
    client, captured = make_client(provider_tools=ProviderToolsConfig(provider="openai", code_interpreter=True))
    with pytest.raises(LLMError, match="stateless history"):
        act(client, provider_tools_enabled=enabled, **history)
    assert captured.payloads == []


def test_code_cache_marked_messages_do_not_preserve_provider_annotations() -> None:
    client, captured = make_client(provider_tools=ProviderToolsConfig(provider="openai", code_interpreter=True))
    messages = [{"role": "user", "content": [{"type": "output_text", "text": "plain",
        "prompt_cache_breakpoint": {"mode": "explicit"},
        "annotations": [{"container_id": "cntr-secret"}]}]}]
    asyncio.run(client.acomplete_action(messages, [FUNCTION]))
    assert "cntr-secret" not in json.dumps(captured.payloads)
    assert captured.payloads[0]["input"] == [{"role": "user", "content": "plain"}]


def test_internal_calls_and_suppression_do_not_inject_tools() -> None:
    client, captured = make_client(provider_tools=search())
    suppressed = act(client, provider_tools_enabled=False)
    asyncio.run(client.acomplete_with_metadata(MESSAGES, json_mode=False))
    assert [tool["type"] for tool in captured.payloads[0]["tools"]] == ["function"]
    assert "tools" not in captured.payloads[1]
    assert suppressed.provider_request_options["provider_tools"]["effective"] == []


def test_managed_results_are_separate_from_local_function_calls_and_trace() -> None:
    output = [search_item(), {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "wait", "arguments": "{}"},
        {"type": "message", "content": [{"type": "output_text", "text": "answer", "annotations": [
            {"type": "url_citation", "url": "https://example.com", "title": "Example", "start_index": 0, "end_index": 6},
        ]}]}]
    client, _ = make_client(provider_tools=search(), output=response(output, x_tools={"web_search": {"count": 1}}))
    completion = act(client)
    assert [call["name"] for call in completion.tool_calls] == ["wait"]
    assert completion.provider_tool_activities == [search_item()]
    assert completion.citations[0]["title"] == "Example"
    assert completion.provider_request_options["provider_tools"]["usage"] == {"web_search": {"count": 1}}
    trace_tools = completion.provider_trace["attempts"][0]["provider_tools"]
    assert trace_tools["observed"] == "returned"
    assert trace_tools["activities"][0]["type"] == "web_search_call"


def test_native_search_replay_accepts_provider_items_without_pairing_with_local_results() -> None:
    client, captured = make_client(provider_tools=search(), output=response([search_item()]), responses_replay=True)
    completion = act(client, responses_items=[{"role": "user", "content": "Research"}])
    assert completion.response_items == [search_item()]
    client._native_replay_input(completion.response_items)
    assert captured.payloads[0]["truncation"] == "disabled"


def test_aliyun_extractor_body_survives_completion_trace_and_history() -> None:
    item = {"type": "web_extractor_call", "id": "we_1", "status": "completed",
            "goal": "Extract weather", "urls": ["https://example.com/weather"], "output": "Rain tomorrow"}
    client, _ = make_client("aliyun", provider_tools=search("aliyun", web_extractor=True), output=response([item]))
    completion = act(client)
    assert completion.provider_tool_activities == [item]
    assert completion.provider_trace["attempts"][0]["provider_tools"]["activities"][0]["output"] == "Rain tomorrow"
    assert "Rain tomorrow" in provider_tool_result_text("", completion.provider_tool_activities, [], [])


@pytest.mark.parametrize("action", [{"type": "open_page", "url": "https://example.com"},
                                    {"type": "find_in_page", "url": "https://example.com", "pattern": "weather"}])
def test_openai_web_page_actions_are_observations(action: dict[str, Any]) -> None:
    client, _ = make_client(provider_tools=search(), output=response([search_item(action=action)]))
    completion = act(client)
    assert completion.provider_tool_activities[0]["action"] == action
    assert completion.tool_calls == []


@pytest.mark.parametrize("item", [
    {"type": "mcp_call", "id": "mcp_1"},
    search_item(status="in_progress"),
    search_item(action={"type": "unexpected"}),
    search_item(action={"type": "open_page", "query": "invalid field"}),
    search_item(unrecognized="unknown"),
    search_item(action={"type": "search", "query": "x" * (PROVIDER_TOOL_TEXT_MAX_CHARS + 1)}),
    search_item(action={"type": "search", "sources": [{"type": "url", "url": "javascript:alert(1)"}]}),
    {"type": "code_interpreter_call", "id": "code_1", "status": "completed"},
])
def test_unknown_disabled_invalid_or_oversized_managed_results_fail(item: dict[str, Any]) -> None:
    client, _ = make_client(provider_tools=search(), output=response([item]))
    with pytest.raises(LLMError):
        act(client)


def test_total_activity_count_is_bounded() -> None:
    client, _ = make_client(provider_tools=search(), output=response([search_item()] * (PROVIDER_TOOL_MAX_ITEMS + 1)))
    with pytest.raises(LLMError, match="activities exceed bounds"):
        act(client)


def test_code_output_files_and_history_preserve_results_without_container_state() -> None:
    item = {"type": "code_interpreter_call", "id": "ci_1", "status": "completed", "code": "print(42)",
            "container_id": "cntr-private", "outputs": [{"type": "logs", "logs": "42"}]}
    message = {"type": "message", "content": [{"type": "output_text", "text": "Result", "annotations": [{
        "type": "container_file_citation", "file_id": "file-result", "container_id": "cntr-private", "filename": "result.txt",
    }]}]}
    client, _ = make_client(provider_tools=ProviderToolsConfig(provider="openai", code_interpreter=True), output=response([item, message]))
    completion = act(client)
    assert completion.response_items == []
    assert completion.artifacts[0]["file_id"] == "file-result"
    history = provider_tool_result_text(completion.content, completion.provider_tool_activities, completion.citations, completion.artifacts)
    assert "42" in history and "file-result" in history
    assert "cntr-private" not in history and "print(42)" not in history


def test_cache_fingerprint_changes_for_chat_search_suppression() -> None:
    client, captured = make_client("aliyun", provider_tools=search("aliyun"), prompt_cache_mode="implicit", prompt_cache_key="host-private")
    client.api_mode = "chat"
    captured.result = SimpleNamespace(id="chat_1", model="qwen3-max", choices=[SimpleNamespace(
        message=SimpleNamespace(content="answer", tool_calls=[]), finish_reason="stop")])
    act(client)
    act(client, provider_tools_enabled=False)
    assert captured.payloads[0]["prompt_cache_key"] != captured.payloads[1]["prompt_cache_key"]


def test_host_boolean_override_is_strict() -> None:
    client, captured = make_client(provider_tools=search())
    with pytest.raises(LLMError, match="Host boolean"):
        act(client, provider_tools_enabled="false")
    assert captured.payloads == []


def test_installed_sdk_openai_managed_types_match_strict_projection() -> None:
    from openai._models import construct_type
    from openai.types.responses import Response

    output = [search_item(action={"type": "search", "queries": ["weather"], "sources": [
        {"type": "url", "url": "https://example.com/weather"},
    ]}), {"type": "code_interpreter_call", "id": "code_1", "status": "completed", "container_id": "cntr_1",
        "code": "print(42)", "outputs": [{"type": "logs", "logs": "42"}, {"type": "image", "url": "https://example.com/chart.png"}]},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{
            "type": "output_text", "text": "Result", "annotations": [{"type": "container_file_citation",
                "container_id": "cntr_1", "file_id": "file_1", "filename": "result.csv", "start_index": 0, "end_index": 6}],
        }]}]
    sdk_response = construct_type(value={"id": "resp_1", "status": "completed", "output": output}, type_=Response)
    client, _ = make_client(provider_tools=search(code_interpreter=True), output=sdk_response)
    completion = act(client)
    assert len(completion.provider_tool_activities) == 2
    assert completion.provider_tool_activities[1]["outputs"][1]["type"] == "image"
    assert completion.artifacts[0]["file_id"] == "file_1"


def test_installed_sdk_aliyun_extractor_placeholder_defaults_are_omitted() -> None:
    from openai._models import construct_type
    from openai.types.responses import Response

    item = {"type": "web_extractor_call", "id": "we_1", "status": "completed",
            "goal": "Extract weather", "urls": ["https://example.com/weather"], "output": "Rain tomorrow"}
    sdk_response = construct_type(value={"id": "resp_1", "status": "completed", "output": [item],
        "x_tools": {"web_search": {"count": 1}}}, type_=Response)
    client, _ = make_client("aliyun", provider_tools=search("aliyun", web_extractor=True), output=sdk_response)
    completion = act(client)
    assert completion.provider_tool_activities == [item]
    assert completion.provider_request_options["provider_tools"]["usage"] == {"web_search": {"count": 1}}


@pytest.mark.parametrize("extra", [{"content": None}, {"role": None}, {"phase": None}, {"unexpected": None}, {"content": "bad"}])
def test_installed_sdk_aliyun_explicit_invalid_fields_remain_rejected(extra: dict[str, Any]) -> None:
    from openai._models import construct_type
    from openai.types.responses import Response

    item = {"type": "web_extractor_call", "id": "we_1", "status": "completed",
            "goal": "Extract weather", "urls": ["https://example.com/weather"], "output": "Rain tomorrow", **extra}
    sdk_response = construct_type(value={"id": "resp_1", "status": "completed", "output": [item]}, type_=Response)
    client, _ = make_client("aliyun", provider_tools=search("aliyun", web_extractor=True), output=sdk_response)
    with pytest.raises(LLMError, match="Unsupported provider tool fields"):
        act(client)


def test_real_llm_marker_host_environment_is_independent_of_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import _has_real_llm_environment

    prefix = "AGENT_LIBOS_REAL_PROVIDER_TOOLS_OPENAI"
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_MODEL", "ambient-model")
    monkeypatch.delenv(f"{prefix}_API_KEY", raising=False)
    monkeypatch.delenv(f"{prefix}_MODEL", raising=False)
    assert _has_real_llm_environment() is True
    assert _has_real_llm_environment(prefix) is False
    monkeypatch.setenv(f"{prefix}_API_KEY", "dedicated-key")
    monkeypatch.setenv(f"{prefix}_MODEL", "dedicated-model")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert _has_real_llm_environment() is False
    assert _has_real_llm_environment(prefix) is True
