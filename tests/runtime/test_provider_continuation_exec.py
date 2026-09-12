from __future__ import annotations

import pytest

from agent_libos import Runtime
from agent_libos.models import PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT
from agent_libos.utils.serde import dumps
from tests.runtime.test_provider_tools_executor import (
    _IMAGE,
    _RESULT,
    _Responses,
    _assert_local_only,
    _capture,
    _config,
    _spawn,
)


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
def test_host_exec_preserves_pending_search_result_in_fresh_local_only_request(mode: str) -> None:
    runtime = Runtime.open("local", config=_config())
    try:
        provider = _Responses("hosted", "action")
        _capture(runtime, provider)
        pid = _spawn(runtime, mode)
        first = runtime.run_process_once(pid)
        assert first["ok"] and first.get("provider_continuation") is True, first
        assert len(provider.requests) == 1
        assert any(tool["type"] == "web_search" for tool in provider.requests[0]["tools"])
        generation = runtime.store.get_llm_context_generation(pid)
        source = runtime.store.get_llm_call(first["call_id"])
        assert runtime.store.get_llm_replay_head(pid) is not None

        runtime.exec_process(
            pid, _IMAGE, preserve_memory=True, preserve_capabilities=True,
        )

        assert runtime.store.get_llm_replay_head(pid) is None
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert runtime.store.get_llm_call(first["call_id"]) == source
        second = runtime.run_process_once(pid)
        assert second["ok"] and second["action"]["action"] == "echo", second
        assert len(provider.requests) == 2
        _assert_local_only(provider.requests[1])
        assert _RESULT in dumps(provider.requests[1]["input"])
        marker = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        assert marker.request_options["provider_continuation"]["state"] == "consumed"
    finally:
        runtime.close()
