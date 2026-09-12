from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ObjectPatch, ObjectType
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.replay_source_recovery import LLMReplaySourceRecovery
from tests.runtime.test_provider_tools_executor import _Responses, _capture, _config, _spawn


def _recovery(runtime: Runtime) -> LLMReplaySourceRecovery:
    return LLMReplaySourceRecovery(
        runtime.uow, config=runtime.config, capabilities=runtime.capability,
        profile_snapshot=runtime.llms.profile_snapshot,
    )


@pytest.mark.parametrize("change", ["revoke-before", "revoke-after", "source-version", "source-deleted"])
def test_provider_continuation_recovery_cannot_restore_revoked_or_changed_source(
    tmp_path: Path, change: str,
) -> None:
    database = tmp_path / "continuation-authority.sqlite"
    config = _config(code=True)
    provider = _Responses("hosted", "action", code=True)
    runtime = Runtime.open(database, config=config)
    try:
        _capture(runtime, provider)
        pid = _spawn(runtime)
        source = runtime.memory.create_object(
            pid, ObjectType.ARTIFACT, {"value": "source read by hosted tool"}, immutable=False,
        )
        runtime.llm._add_to_view(pid, source)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        if change == "revoke-before":
            runtime.capability.revoke_resource_trusted(
                f"object:{source.oid}", revoked_by="test", reason="revoke before restart",
            )
        elif change == "source-version":
            runtime.memory.update_object(pid, source, ObjectPatch(payload={"value": "changed"}))
        elif change == "source-deleted":
            runtime.memory.delete_object_trusted("test", source.oid, reason="explicit deletion")
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        _capture(reopened, provider)
        if change == "revoke-after":
            assert reopened.store.get_capability(source.capability_id).active
            reopened.capability.revoke_resource_trusted(
                f"object:{source.oid}", revoked_by="test", reason="revoke after restart",
            )
        assert not reopened.store.get_capability(source.capability_id).active
        outcome = reopened.run_process_once(pid)
        assert not outcome["ok"], outcome
        assert len(provider.requests) == 1
        assert any(
            record.action == "capability.authorize"
            and record.target == f"object:{source.oid}"
            and record.decision.get("allowed") is False
            for record in reopened.audit.trace(actor=pid)
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["source-call", "payload", "profile"])
def test_provider_continuation_recovery_validates_retained_bindings_before_read_retention(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        provider = _Responses("action", "hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        oid = first["result"]["result_oid"]
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        marker = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        source_call_id = marker.request_options["provider_continuation"]["call_id"]
        original_get = runtime.store.get_llm_call
        if change == "source-call":
            def changed_call(call_id):
                call = original_get(call_id)
                return replace(call, response_content="changed") if call_id == source_call_id else call
            monkeypatch.setattr(runtime.store, "get_llm_call", changed_call)
        elif change == "payload":
            original_latest = runtime.store.get_latest_llm_call
            def changed_marker(**kwargs):
                call = original_latest(**kwargs)
                return replace(call, raw_response={}) if call is not None and call.call_id == marker.call_id else call
            monkeypatch.setattr(runtime.store, "get_latest_llm_call", changed_marker)
        else:
            profile = runtime.config.llm.profiles["default"]
            runtime.llms.register_profile("default", replace(profile, model="another-model"))
        recovery = _recovery(runtime)
        with pytest.raises(ValidationError, match="provider continuation"):
            recovery.preflight()
        with pytest.raises(ValidationError, match="completed preflight"):
            recovery.retained_read_capabilities((oid,))
        assert len(provider.requests) == 2
    finally:
        runtime.close()
