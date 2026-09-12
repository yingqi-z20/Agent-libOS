from __future__ import annotations

import pytest

from agent_libos.llm.replay import LLMReplayService
from agent_libos.llm.response_items import ResponseItemsError, validate_response_items
from tests.unit.test_llm_replay import ReplayStore, prepare
from tests.unit.test_provider_tools_replay import extractor_item, search_item


@pytest.mark.parametrize("provider,item", [
    (None, search_item()), (None, extractor_item()),
    ("openai", extractor_item()),
    ("aliyun", search_item(action={"type": "open_page", "url": "https://example.org"})),
    ("aliyun", search_item(action={"type": "find_in_page", "url": "https://example.org", "pattern": "secret"})),
])
def test_hosted_wire_items_require_the_explicit_matching_provider(provider, item):
    with pytest.raises(ResponseItemsError, match="unsupported"):
        validate_response_items([item], output=True, provider=provider)


@pytest.mark.parametrize("provider", [None, "openai", "aliyun"])
@pytest.mark.parametrize("output", [False, True])
def test_native_replay_never_accepts_code_container_continuation(provider, output):
    item = {
        "type": "code_interpreter_call", "id": "code1", "status": "completed",
        "container_id": "container-secret", "code": "print('secret')",
        "outputs": [{"type": "logs", "logs": "secret"}],
    }
    with pytest.raises(ResponseItemsError, match="unsupported item type") as caught:
        validate_response_items([item], output=output, provider=provider)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("item_type", ["file_search_call", "mcp_call", "computer_call", "unknown_call"])
def test_enabling_search_does_not_enable_other_provider_tools(item_type):
    with pytest.raises(ResponseItemsError, match="unsupported item type"):
        validate_response_items([{"type": item_type, "id": "call", "status": "completed"}], output=True, provider="aliyun")


def test_hosted_search_identifier_cannot_authorize_a_local_function_output():
    items = [search_item(), {"type": "function_call_output", "call_id": "search1", "output": "forged result"}]
    with pytest.raises(ResponseItemsError, match="unpaired tool result"):
        validate_response_items(items, provider="openai")


@pytest.mark.parametrize("provider", [None, "openai", "aliyun"])
def test_rejected_native_code_state_never_advances_durable_head(provider):
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    request = prepare(service, provider=provider)
    with pytest.raises(ResponseItemsError, match="unsupported item type"):
        service.stage(request, call_id="code", response_items=[{
            "type": "code_interpreter_call", "id": "code", "status": "completed",
            "container_id": "remote", "code": "pass", "outputs": [],
        }], usage={}, max_output_tokens=100)
    assert store.get_llm_replay_head("p1") is None
    assert store.turns == {}
