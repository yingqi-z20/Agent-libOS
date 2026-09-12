from __future__ import annotations

import pytest

from agent_libos.llm.replay import LLMReplayService, ReplayStateError
from agent_libos.storage import SQLiteStore
from agent_libos.utils.serde import dumps
from tests.unit.test_llm_replay import flow, prepare
from tests.unit.test_provider_tools_replay import extractor_item, search_item


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_hosted_replay_survives_store_restart_and_remains_private(tmp_path, provider):
    path = tmp_path / "replay.sqlite"
    store = SQLiteStore(path)
    items = [search_item()]
    if provider == "aliyun":
        items.append(extractor_item())
    try:
        service = LLMReplayService(store, max_bytes=1_000_000)
        first = prepare(service, provider=provider)
        service.stage(first, call_id="hosted", response_items=items, usage={}, max_output_tokens=100)
        service.mark_validated(pid="p1", call_id="hosted")
        refs = service.capture_checkpoint_refs(["p1"])
        frozen = service.freeze_request(prepare(service, "resume", provider=provider))
    finally:
        store.close()
    store = SQLiteStore(path)
    try:
        service = LLMReplayService(store, max_bytes=1_000_000)
        resumed = service.load_request(frozen, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial", provider=provider)
        assert resumed.response_items == [*first.response_items, *items, {"role": "user", "content": "resume"}]
        assert "weather" not in dumps(resumed)
        assert "weather" not in repr(resumed)
        head = service.rebind(turn_id=refs["p1"]["turn_id"], pid="restored", context_generation="restored", provider_fingerprint="provider", model="gpt-test", flow_context=flow())
        payload = service.validate_turn(store.get_llm_replay_turn(head.turn_id))
        assert payload["groups"][0]["output_items"] == items
        assert payload["provider"] == provider
        store.purge_llm_replay(pid="p1")
        with pytest.raises(ReplayStateError, match="missing or purged"):
            service.load_request(frozen, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial", provider=provider)
        assert service.load_current("restored") is not None
    finally:
        store.close()
