from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai._models import construct_type
from openai.types.responses import Response, ResponseOutputMessage

from agent_libos.llm.replay import LLMReplayService, ReplayStateError
from agent_libos.llm.response_items import ResponseItemsError, validate_response_items
from tests.unit.test_llm_replay import ReplayStore, flow, prepare


def search_item(*, action=None, status="completed"):
    return {
        "type": "web_search_call", "id": "search1", "status": status,
        "action": action or {
            "type": "search", "query": "weather", "queries": ["weather today"],
            "sources": [{"type": "url", "url": "https://example.org/weather"}],
        },
    }


def extractor_item():
    return {
        "type": "web_extractor_call", "id": "extract1", "status": "completed",
        "goal": "Read forecast", "urls": ["https://example.org/weather"],
        "output": "Rain is expected.",
    }


def _sdk_extractor_response(item=None):
    return construct_type(type_=Response, value={
        "id": "response-extractor", "status": "completed", "model": "qwen3-max",
        "output": [extractor_item() if item is None else item],
    })


def test_aliyun_sdk_extractor_drops_only_synthetic_message_defaults():
    response = _sdk_extractor_response()
    item = response.output[0]
    assert isinstance(item, ResponseOutputMessage)
    assert item.__dict__["content"] is None
    assert "content" not in item.__pydantic_fields_set__
    before = deepcopy(item.__dict__)
    assert validate_response_items(response.output, output=True, provider="aliyun") == [extractor_item()]
    assert item.__dict__ == before


@pytest.mark.parametrize("field,value", [
    ("content", None), ("role", None), ("phase", None),
    ("content", []), ("role", "assistant"), ("phase", "final_answer"),
    ("unexpected", None),
])
def test_aliyun_sdk_extractor_still_rejects_explicit_unsupported_fields(field, value):
    response = _sdk_extractor_response({**extractor_item(), field: value})
    with pytest.raises(ResponseItemsError, match="unsupported item fields"):
        validate_response_items(response.output, output=True, provider="aliyun")


def test_raw_extractor_null_fields_are_not_treated_as_sdk_defaults():
    with pytest.raises(ResponseItemsError, match="unsupported item fields"):
        validate_response_items([{**extractor_item(), "role": None}], output=True, provider="aliyun")


def test_aliyun_sdk_extractor_capture_can_be_replayed_without_reenabling_tools():
    from agent_libos.config import ProviderToolsConfig
    from tests.unit.test_responses_replay_client import _client, _response

    client, responses = _client(
        [_sdk_extractor_response(), _response([])], model="qwen3-max",
        api_mode="responses", responses_replay=True,
        provider_tools=ProviderToolsConfig(provider="aliyun", web_search=True, web_extractor=True),
    )
    first = client.complete_action([{"role": "user", "content": "Extract forecast"}], [])
    assert first.response_items == [extractor_item()]
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    request = prepare(service, provider="aliyun")
    service.stage(request, call_id="extract", response_items=first.response_items, usage={}, max_output_tokens=100)
    service.mark_validated(pid="p1", call_id="extract")
    next_request = prepare(service, provider="aliyun")
    client.complete_action([], [], responses_items=next_request.response_items, provider_tools_enabled=False)
    assert responses.payloads[1]["input"] == next_request.response_items
    assert extractor_item() in responses.payloads[1]["input"]
    assert not any(tool["type"] in {"web_search", "web_extractor"} for tool in responses.payloads[1].get("tools", []))


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
@pytest.mark.parametrize("output", [True, False])
def test_hosted_search_requires_no_local_function_result(provider, output):
    items = [search_item(), search_item(status="failed")]
    assert validate_response_items(items, output=output, provider=provider) == items


@pytest.mark.parametrize("action", [
    {"type": "search"},
    {"type": "search", "query": None, "queries": None, "sources": None},
    {"type": "open_page"},
    {"type": "open_page", "url": None},
    {"type": "open_page", "url": "https://example.org"},
    {"type": "find_in_page", "url": "https://example.org", "pattern": "weather"},
])
def test_openai_documented_search_actions_and_sdk_defaults_round_trip(action):
    item = search_item(action=action)
    sdk = SimpleNamespace(**{**item, "action": SimpleNamespace(**action)})
    assert validate_response_items([sdk], output=True, provider="openai") == [item]


def test_aliyun_extractor_preserves_text_and_order_among_function_items():
    items = [
        search_item(), extractor_item(),
        {"type": "function_call", "call_id": "local", "name": "finish", "arguments": "{}"},
        search_item(action={"type": "search", "queries": ["followup"]}),
        {"type": "function_call_output", "call_id": "local", "output": "done"},
    ]
    assert validate_response_items(items, provider="aliyun") == items
    assert items[1]["output"] == "Rain is expected."


@pytest.mark.parametrize("item", [
    search_item(status="in_progress"), search_item(status="searching"),
    search_item(status="incomplete"),
    {**search_item(), "id": ""}, {**search_item(), "status": None},
    {**search_item(), "container_id": "forbidden"},
    search_item(action={"type": "search", "queries": [42]}),
    search_item(action={"type": "search", "query": []}),
    search_item(action={"type": "search", "sources": ["https://example.org"]}),
    search_item(action={"type": "search", "sources": [{"type": "file", "url": "bad"}]}),
    search_item(action={"type": "search", "sources": [{"type": "url", "url": "url", "extra": "bad"}]}),
    search_item(action={"type": "find_in_page", "url": "url"}),
    search_item(action={"type": "open_page", "url": [], "unexpected": True}),
    search_item(action={"type": "unknown", "query": "secret"}),
])
def test_invalid_hosted_search_fails_closed_without_echoing_content(item):
    with pytest.raises(ResponseItemsError) as caught:
        validate_response_items([item], output=True, provider="openai")
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("changes", [
    {"output": []}, {"goal": None}, {"urls": "https://example.org"},
    {"urls": [42]}, {"status": "failed"}, {"extra": "unsupported"},
])
def test_invalid_aliyun_extractor_fails_closed(changes):
    with pytest.raises(ResponseItemsError):
        validate_response_items([{**extractor_item(), **changes}], output=True, provider="aliyun")


def test_hosted_payloads_obey_existing_nested_and_string_bounds(monkeypatch):
    from agent_libos.llm import response_items

    monkeypatch.setattr(response_items, "RESPONSE_ITEMS_MAX_STRING_BYTES", 128)
    with pytest.raises(ResponseItemsError, match="text exceeds"):
        validate_response_items([{**extractor_item(), "output": "x" * 129}], output=True, provider="aliyun")
    monkeypatch.setattr(response_items, "RESPONSE_ITEMS_MAX_ITEMS", 1)
    with pytest.raises(ResponseItemsError, match="item count"):
        validate_response_items([search_item(), search_item()], provider="openai")


def test_legacy_profile_and_payload_remain_unchanged():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    request = prepare(service)
    assert request.payload["schema_version"] == 1
    assert "provider" not in request.payload
    with pytest.raises(ResponseItemsError, match="unsupported item type"):
        service.stage(request, call_id="search", response_items=[search_item()], usage={}, max_output_tokens=100)


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_hosted_replay_restarts_freezes_and_pairs_only_local_calls(provider):
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    first = prepare(service, provider=provider)
    outputs = [search_item()]
    if provider == "aliyun":
        outputs.append(extractor_item())
    outputs.append({"type": "function_call", "call_id": "local", "name": "echo", "arguments": "{}"})
    service.stage(first, call_id="first", response_items=outputs, usage={}, max_output_tokens=100)
    service.mark_validated(pid="p1", call_id="first")
    with pytest.raises(ReplayStateError, match="durable tool outputs"):
        prepare(service, provider=provider)
    tool_result = {"type": "function_call_output", "call_id": "local", "output": "done"}
    service.settle(pid="p1", call_id="first", outputs=[tool_result])
    restored = LLMReplayService(store, max_bytes=1_000_000)
    request = prepare(restored, "followup", provider=provider)
    assert request.response_items == [*first.response_items, *outputs, tool_result, {"role": "user", "content": "followup"}]
    reference = restored.freeze_request(request)
    loaded = restored.load_request(reference, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial", provider=provider)
    assert loaded == request
    assert loaded.payload["schema_version"] == 2
    assert loaded.payload["provider"] == provider


def test_hosted_only_turn_can_be_validated_and_replayed_without_settlement():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    request = prepare(service, provider="aliyun")
    items = [search_item(), extractor_item()]
    service.stage(request, call_id="search", response_items=items, usage={}, max_output_tokens=100)
    service.mark_validated(pid="p1", call_id="search")
    followup = prepare(service, provider="aliyun")
    assert followup.response_items[len(request.response_items):-1] == items


def test_compaction_and_checkpoint_preserve_provider_scope():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    request = prepare(service, provider="aliyun")
    service.stage(request, call_id="search", response_items=[search_item(), extractor_item()], usage={}, max_output_tokens=100)
    service.mark_validated(pid="p1", call_id="search")
    reference = service.capture_checkpoint_refs(["p1"])["p1"]
    head = service.rebind(turn_id=reference["turn_id"], pid="fork", context_generation="fork-generation", provider_fingerprint="provider", model="gpt-test", flow_context=flow())
    forked = service.validate_turn(service.store.get_llm_replay_turn(head.turn_id))
    assert forked["provider"] == "aliyun"
    assert forked["groups"][0]["output_items"] == [search_item(), extractor_item()]
    service.compact(pid="p1", context_generation="compacted", messages=[{"role": "system", "content": "stable"}, {"role": "user", "content": "summary"}], flow_context=flow())
    compacted = prepare(service, provider="aliyun", context_generation="compacted")
    assert compacted.payload["provider"] == "aliyun"
    assert not any(item.get("type") in {"web_search_call", "web_extractor_call"} for item in compacted.response_items)


@pytest.mark.parametrize("provider", [None, "openai"])
def test_existing_hosted_history_cannot_change_provider_even_with_same_fingerprint(provider):
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    request = prepare(service, provider="aliyun")
    service.stage(request, call_id="search", response_items=[search_item()], usage={}, max_output_tokens=100)
    service.mark_validated(pid="p1", call_id="search")
    head = deepcopy(service.store.get_llm_replay_head("p1"))
    with pytest.raises(ReplayStateError, match="provider changed"):
        prepare(service, provider=provider)
    assert service.store.get_llm_replay_head("p1") == head
    reference = service.freeze_request(prepare(service, provider="aliyun"))
    with pytest.raises(ReplayStateError, match="provider changed"):
        service.load_request(reference, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial", provider=provider)
