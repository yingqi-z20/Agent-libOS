from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem

from agent_libos.llm import client as client_module
from agent_libos.llm.client import LLMClient, LLMError
from agent_libos.llm.response_items import ResponseItemsError, validate_response_items
from agent_libos.utils.serde import dumps


class _Responses:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.payloads: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        response = self.outputs.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(output: list[Any], *, status: str = "completed") -> Any:
    return SimpleNamespace(
        id="resp_1", model="gpt-6-astra", status=status, output=output,
        usage=SimpleNamespace(input_tokens=10, output_tokens=8,
                              output_tokens_details={"reasoning_tokens": 6}),
    )


def _client(outputs: list[Any], **kwargs: Any) -> tuple[LLMClient, _Responses]:
    responses = _Responses(outputs)
    client = LLMClient(api_key="test", **kwargs)
    client._async_client = SimpleNamespace(
        responses=responses,
        chat=SimpleNamespace(completions=_Responses([])),
    )
    return client, responses


def _reasoning(secret: str = "OPAQUE_REASONING_SECRET") -> dict[str, Any]:
    return {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": secret}


def _call(call_id: str = "call_1") -> dict[str, Any]:
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
            "name": "read", "arguments": "{}", "status": "completed"}


def _message(text: str = "done") -> dict[str, Any]:
    return {"type": "message", "id": "msg_1", "role": "assistant",
            "phase": "final_answer", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def test_official_default_is_stateless_astra_without_sampling_options() -> None:
    client, responses = _client([_response([_message()])])
    result = client.complete_action([{"role": "user", "content": "answer"}], [])
    payload = responses.payloads[0]
    assert client.model == "gpt-6-astra"
    assert payload["model"] == "gpt-6-astra"
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "medium", "context": "all_turns"}
    assert payload["truncation"] == "disabled"
    assert "temperature" not in payload
    assert "previous_response_id" not in payload
    assert result.response_items == [_message()]


def test_explicit_reasoning_policy_overrides_official_defaults() -> None:
    client = LLMClient(api_key="test", model="gpt-6-astra", reasoning_effort="low",
                       reasoning_context="current_turn", responses_replay=False)
    payload = client._responses_payload([], 0.7, 64)
    assert payload["reasoning"] == {"effort": "low", "context": "current_turn"}
    assert client.responses_replay is False
    assert "temperature" not in payload


def test_replay_keeps_exact_order_ids_phase_and_parallel_tool_results() -> None:
    first_output = [_reasoning(), _message("working"), _call("a"), _call("b")]
    first_output[1]["phase"] = "commentary"
    client, responses = _client([_response(first_output), _response([_message()])])
    first = client.complete_action([{"role": "user", "content": "read"}], [])
    native = [{"role": "system", "content": "host prefix"},
              {"role": "user", "content": "read"}, *first.response_items,
              {"type": "function_call_output", "call_id": "b", "output": "safe b"},
              {"type": "function_call_output", "call_id": "a", "output": "safe a"},
              {"role": "user", "content": "current dynamic state"}]
    second = client.complete_action(
        [{"role": "user", "content": "PUBLIC_HISTORY_MUST_NOT_BE_DUPLICATED"}],
        [], responses_items=native,
    )
    assert first.response_items == first_output
    assert responses.payloads[1]["input"] == native
    assert responses.payloads[1]["input"] is not native
    assert "instructions" not in responses.payloads[1]
    assert "PUBLIC_HISTORY_MUST_NOT_BE_DUPLICATED" not in json.dumps(responses.payloads[1])
    assert second.content == "done"
    assert "OPAQUE_REASONING_SECRET" not in dumps(first)
    assert "OPAQUE_REASONING_SECRET" not in repr(first)
    assert "OPAQUE_REASONING_SECRET" not in json.dumps(first.raw)
    assert "OPAQUE_REASONING_SECRET" not in json.dumps(first.reasoning)
    assert "OPAQUE_REASONING_SECRET" not in json.dumps(first.provider_trace)


def test_replay_codec_roundtrips_supported_sdk_items_without_loss() -> None:
    reasoning = ResponseReasoningItem(**_reasoning())
    message = ResponseOutputMessage(**_message())
    expected = [reasoning.model_dump(), message.model_dump()]
    assert validate_response_items([reasoning, message], output=True) == expected


@pytest.mark.parametrize("as_sdk_object", [False, True], ids=["dict", "sdk"])
def test_sdk_function_calls_roundtrip_through_native_tool_loop(as_sdk_object: bool) -> None:
    sdk_calls = [ResponseFunctionToolCall(**_call(call_id)) for call_id in ("a", "b")]
    original = [call.model_dump() for call in sdk_calls]
    calls = sdk_calls if as_sdk_object else original
    client, responses = _client([
        _response([_reasoning(), *calls]), _response([_message()]),
    ])

    first = client.complete_action([{"role": "user", "content": "read"}], [])

    assert first.response_items == [_reasoning(), _call("a"), _call("b")]
    assert [call["call_id"] for call in first.tool_calls] == ["a", "b"]
    assert [call.model_dump() for call in sdk_calls] == original
    assert all(call["caller"] is None and call["namespace"] is None for call in original)
    native = [
        {"role": "user", "content": "read"}, *first.response_items,
        {"type": "function_call_output", "call_id": "b", "output": "safe b"},
        {"type": "function_call_output", "call_id": "a", "output": "safe a"},
    ]
    second = client.complete_action([], [], responses_items=native)
    assert second.content == "done"
    assert responses.payloads[1]["input"] == native
    assert responses.payloads[1]["store"] is False
    assert "previous_response_id" not in responses.payloads[1]


@pytest.mark.parametrize("metadata", [
    {"caller": {"type": "program", "caller_id": "program_1"}},
    {"namespace": "other_tools"},
    {"future_routing_field": None},
])
@pytest.mark.parametrize("as_sdk_object", [False, True], ids=["dict", "sdk"])
def test_function_call_replay_does_not_drop_unsupported_metadata(
    metadata: dict[str, Any], as_sdk_object: bool,
) -> None:
    item = {**_call(), **metadata}
    with pytest.raises(ResponseItemsError, match="unsupported item fields"):
        validate_response_items(
            [ResponseFunctionToolCall(**item) if as_sdk_object else item], output=True,
        )


@pytest.mark.parametrize("mutation", [
    {"status": "in_progress"}, {"status": "incomplete"}, {"status": "failed"},
    {"type": "web_search_call"}, {"phase": "unsupported"},
    {"role": "system"}, {"future_opaque_field": "do not silently drop"},
])
def test_replay_rejects_incomplete_or_unrepresentable_output(mutation: dict[str, Any]) -> None:
    item = _message()
    item.update(mutation)
    client, _responses = _client([_response([item])])
    with pytest.raises(LLMError, match="Responses replay"):
        client.complete_action([{"role": "user", "content": "read"}], [])


@pytest.mark.parametrize("items", [
    [{"type": "function_call_output", "call_id": "orphan", "output": "unsafe"}],
    [_call()],
    [_call(), _call()],
    [_call(), {"type": "function_call_output", "call_id": "call_1", "output": "a"},
     {"type": "function_call_output", "call_id": "call_1", "output": "b"}],
    [{"type": "item_reference", "id": "rs_server_state"}],
])
def test_input_replay_rejects_orphan_duplicate_or_unfinished_tool_groups(items: list[dict[str, Any]]) -> None:
    with pytest.raises(ResponseItemsError):
        validate_response_items(items)


def test_opaque_replay_bounds_fail_without_copying_secret_to_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_libos.llm.response_items as codec

    monkeypatch.setattr(codec, "RESPONSE_ITEMS_MAX_STRING_BYTES", 64)
    secret = "PRIVATE" * 10
    client, _responses = _client([_response([_reasoning(secret)])])
    with pytest.raises(LLMError) as failure:
        client.complete_action([{"role": "user", "content": "read"}], [])
    assert secret not in str(failure.value)
    assert "bound" in str(failure.value)


@pytest.mark.parametrize("block", [
    {"type": "input_image", "image_url": "https://untrusted.invalid/image"},
    {"type": "output_text", "text": "ok", "future_state": "opaque"},
    {"type": "output_text", "text": ["invalid"]},
])
def test_replay_rejects_unrepresentable_content_blocks(block: dict[str, Any]) -> None:
    message = _message()
    message["content"] = [block]
    with pytest.raises(ResponseItemsError):
        validate_response_items([message], output=True)


@pytest.mark.parametrize("error_message", ["reasoning is unsupported", "store is unsupported", "Responses endpoint not found"])
def test_replay_protocol_error_never_drops_state_or_falls_back(
    monkeypatch: pytest.MonkeyPatch, error_message: str,
) -> None:
    class Rejected(Exception):
        status_code = 400

    monkeypatch.setattr(client_module, "_is_openai_sdk_error", lambda _exc: True)
    client, responses = _client([Rejected(error_message)], fallback_json_actions=True)
    with pytest.raises(LLMError):
        client.complete_action([], [], responses_items=[_reasoning()])
    assert len(responses.payloads) == 1
    assert not client._async_client.chat.completions.payloads


def test_explicit_replay_and_server_response_id_are_mutually_exclusive() -> None:
    client, responses = _client([])
    with pytest.raises(LLMError, match="previous_response_id"):
        client.complete_action([], [], responses_items=[], previous_response_id="resp_old")
    assert not responses.payloads


def test_text_completion_accepts_fresh_explicit_native_replay_on_other_model() -> None:
    client, responses = _client([_response([_message()])], model="gpt-test")
    items = [{"role": "system", "content": "host"}, {"role": "user", "content": "answer"}]
    result = asyncio.run(client.acomplete_with_metadata([], json_mode=False, responses_items=items))
    assert responses.payloads[0]["input"] == items
    assert result.response_items == [_message()]


def test_native_text_replay_keeps_json_mode_host_instruction() -> None:
    client, responses = _client([_response([_message('{"ok":true}')])])
    native = [{"role": "user", "content": "answer"}]
    result = client.complete_with_metadata([], responses_items=native)
    assert responses.payloads[0]["input"] == native
    assert responses.payloads[0]["instructions"] == client.defaults.json_instruction
    assert responses.payloads[0]["text"]["format"] == {"type": "json_object"}
    assert result.content == '{"ok":true}'


def test_custom_provider_retains_chat_defaults_and_requires_explicit_model() -> None:
    client = LLMClient(base_url="https://custom.invalid/v1", allow_custom_base_url=True, api_key="test")
    assert client.model is None
    assert client._use_responses_api() is False
    assert client.reasoning_context is None
    assert client.responses_replay is False
    with pytest.raises(LLMError, match="not configured"):
        client._chat_payload([], 0.2, 64)


def test_auto_cache_retains_isolated_host_domain_during_client_lifetime() -> None:
    first = LLMClient(api_key="test", prompt_cache_mode="auto")
    second = LLMClient(api_key="test", prompt_cache_mode="auto")
    messages = [{"role": "system", "content": "stable prefix"}, {"role": "user", "content": "one"}]
    payload = first._responses_payload(messages, 0.2, 64)
    first._finalize_prompt_cache_request(payload)
    other = first._responses_payload(messages[:-1] + [{"role": "user", "content": "two"}], 0.2, 64)
    first._finalize_prompt_cache_request(other)
    assert payload["prompt_cache_key"] == other["prompt_cache_key"]
    assert first.prompt_cache_key != second.prompt_cache_key
    assert payload["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert first.prompt_cache_mode_configured == "auto"
    assert first.prompt_cache_key_source == "host_generated"


def test_candidate_replay_cache_marks_only_stable_host_prefix() -> None:
    client, responses = _client([_response([_message()])], prompt_cache_mode="auto", prompt_layout="auto")
    messages = [{"role": "system", "content": "stable instructions"},
                {"role": "user", "content": "new dynamic state"}]
    native = [messages[0], {"role": "user", "content": "old state"},
              _reasoning(), _message("earlier answer"), messages[1]]
    client.complete_action(messages, [], responses_items=native)
    payload = responses.payloads[0]
    assert payload["input"][0]["content"] == [{
        "type": "input_text", "text": "stable instructions",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }]
    assert payload["input"][1:] == native[1:]
    assert payload["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert payload["input"][-1]["content"] == "new dynamic state"
