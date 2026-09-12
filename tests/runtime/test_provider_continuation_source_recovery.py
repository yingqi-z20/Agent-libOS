from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT, CapabilityRight,
)
from agent_libos.utils.serde import dumps
from tests.runtime.test_provider_tools_executor import (
    _IMAGE, _RESULT, _Responses, _assert_local_only, _capture, _config, _spawn,
)


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
@pytest.mark.parametrize("code", [True, False], ids=["code", "search-without-replay"])
def test_provider_continuation_after_local_tool_survives_source_recovery(
    tmp_path: Path, mode: str, code: bool,
) -> None:
    config = _config(code=code)
    config = replace(config, llm=replace(config.llm, profiles={
        "default": replace(config.llm.profiles["default"], responses_replay=False),
    }))
    database = tmp_path / "continuation-source.sqlite"
    provider = _Responses("action", "hosted", "action", code=code)
    runtime = Runtime.open(database, config=config)
    try:
        _capture(runtime, provider)
        pid = _spawn(runtime, mode)
        first = runtime.run_process_once(pid)
        assert first["ok"], first
        oid = first["result"]["result_oid"]
        extra = runtime.capability.issue_trusted(
            pid, f"object:{oid}", [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test", delegable=True,
        )
        second = runtime.run_process_once(pid)
        assert second.get("provider_continuation") is True, second
        pending = runtime.llm._provider_continuation_data(pid)
        assert any(ref["oid"] == oid for ref in pending["flow_context"]["source_refs"])
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=config)
    try:
        _capture(reopened, provider)
        state = reopened.store.get_persisted_object_state(oid)
        assert state.recovered_after_reopen and not state.payload_present
        retained = reopened.store.get_capability(extra.cap_id)
        assert retained.active and retained.rights == {CapabilityRight.READ.value}
        assert not retained.delegable
        outcome = reopened.run_process_once(pid)
        assert outcome["ok"] and outcome["action"]["action"] == "echo", outcome
        assert len(provider.requests) == 3
        _assert_local_only(provider.requests[-1])
        assert _RESULT in dumps(provider.requests[-1]["input"])
        assert reopened.llm._provider_continuation_data(pid) is None
    finally:
        reopened.close()


@pytest.mark.parametrize("state", ["consumed", "obsolete-generation"])
def test_settled_continuation_does_not_require_retired_profile_on_reopen(
    tmp_path: Path, state: str,
) -> None:
    config = _config(code=True)
    configured = replace(config, llm=replace(config.llm, profiles={
        **config.llm.profiles, "retired": config.llm.profiles["default"],
    }))
    database = tmp_path / "retired-profile.sqlite"
    runtime = Runtime.open(database, config=configured)
    try:
        _spawn(runtime)  # Register the test image.
        pid = runtime.process.spawn(image=_IMAGE, goal="settled result", llm_profile_id="retired")
        provider = _Responses("hosted", "action", code=True)
        runtime.llms.resolve("retired").client._async_client = SimpleNamespace(responses=provider)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        if state == "consumed":
            assert runtime.run_process_once(pid)["ok"]
        else:
            runtime.store.set_llm_context_generation(pid, "replacement-generation")
        assert runtime.store.get_llm_replay_head(pid) is None
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        assert reopened.llm._provider_continuation_data(pid) is None
    finally:
        reopened.close()
