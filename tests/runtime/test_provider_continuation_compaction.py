from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT,
    CapabilityEffect, CapabilityRight, ObjectPatch,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.utils.serde import dumps
from tests.runtime.test_provider_tools_executor import (
    _RESULT, _Responses, _assert_local_only, _capture, _config, _spawn,
)
from tests.runtime.test_responses_replay_compaction import _compact_context
from tests.runtime.test_task_run_provider_continuation import CONFIG, _call, _create


def _compaction_config(*, code: bool = True, full_io: bool = True):
    config = _config(code=code, full_io=full_io)
    return replace(config, llm_context=replace(config.llm_context, policy="llm_context_object"))


@pytest.mark.parametrize("full_io", [True, False], ids=["retained", "volatile"])
@pytest.mark.parametrize("code", [True, False], ids=["code", "search"])
def test_host_context_append_preserves_pending_result_and_exact_source(
    code: bool, full_io: bool,
) -> None:
    runtime = Runtime.open("local", config=_compaction_config(code=code, full_io=full_io))
    try:
        provider = _Responses("hosted", "action", code=code)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        first = runtime.run_process_once(pid)
        assert first.get("provider_continuation") is True
        original = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        generation = runtime.store.get_llm_context_generation(pid)
        result = runtime.run_process_once(pid)
        assert result["ok"], result
        _assert_local_only(provider.requests[-1])
        assert _RESULT in dumps(provider.requests[-1]["input"])
        assert runtime.store.get_llm_context_generation(pid) == generation
        markers = runtime.store.list_llm_calls(pid=pid, limit=100)
        updated = [call for call in markers if "provider_continuation_context_update" in call.request_options]
        assert len(updated) == 1
        provenance = updated[0].request_options["provider_continuation_context_update"]
        assert provenance["source_marker_call_id"] == original.call_id
        assert provenance["context_version"] == provenance["source_version"] + 1
        if not full_io:
            assert updated[0].messages is None and updated[0].raw_response is None
            assert _RESULT not in dumps(updated[0])
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
@pytest.mark.parametrize("code", [True, False], ids=["code", "search"])
def test_certified_compaction_preserves_pending_result_and_local_only_next_call(mode: str, code: bool) -> None:
    runtime = Runtime.open("local", config=_compaction_config(code=code))
    try:
        provider = _Responses("hosted", "action", code=code)
        _capture(runtime, provider)
        pid = _spawn(runtime, mode)
        first = runtime.run_process_once(pid)
        assert first.get("provider_continuation") is True
        original = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        certificate = _compact_context(runtime, pid)
        rebound = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        assert rebound.call_id != original.call_id
        manifest = rebound.request_options["provider_continuation"]
        assert manifest["call_id"] == first["call_id"]
        assert manifest["context_generation"] == certificate["context_generation"]
        assert rebound.request_options["provider_continuation_compaction"]["summary_sha256"] == certificate["summary_sha256"]
        result = runtime.run_process_once(pid)
        assert result["ok"], result
        _assert_local_only(provider.requests[-1])
        assert _RESULT in dumps(provider.requests[-1]["input"])
        assert "container_" not in dumps(provider.requests[-1]["input"])
    finally:
        runtime.close()


def test_unretained_pending_result_refuses_compaction_without_changing_generation() -> None:
    runtime = Runtime.open("local", config=_compaction_config(full_io=False))
    try:
        provider = _Responses("hosted", "action", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        generation = runtime.store.get_llm_context_generation(pid)
        marker = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        with pytest.raises(ValidationError, match="retained payloads"):
            _compact_context(runtime, pid)
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation") == marker
        result = runtime.run_process_once(pid)
        assert result["ok"], result
        _assert_local_only(provider.requests[-1])
        assert _RESULT in dumps(provider.requests[-1]["input"])
    finally:
        runtime.close()


def test_pending_task_run_refuses_compaction_without_changing_safe_point() -> None:
    config = replace(CONFIG, llm_context=replace(CONFIG.llm_context, policy="llm_context_object"))
    runtime = Runtime.open("local", config=config)
    try:
        created = _create(runtime)
        pid = created.root_pid
        runtime.store.insert_llm_call(_call(runtime, pid))
        # Actual executor publication writes its generic marker in the same
        # transaction as the TaskRun hook.
        from agent_libos.llm.client import LLMCompletion

        runtime.llm._commit_provider_continuation(
            pid, "hosted-call", LLMCompletion(content="The retained search result.", tool_calls=[]),
        )
        generation = runtime.store.get_llm_context_generation(pid)
        point = runtime.store.get_task_run_resume_point(pid)
        with pytest.raises(ValidationError, match="pending TaskRun provider result"):
            _compact_context(runtime, pid)
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert runtime.store.get_task_run_resume_point(pid) == point
    finally:
        runtime.close()


def test_compaction_certificate_failure_rolls_back_generation_payload_and_marker(monkeypatch) -> None:
    runtime = Runtime.open("local", config=_compaction_config())
    try:
        provider = _Responses("hosted", "action", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        generation = runtime.store.get_llm_context_generation(pid)
        marker = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        context_oid = runtime.llm.context_memory.context_oid(pid)
        original = runtime.store.get_object(context_oid)
        monkeypatch.setattr(runtime.llm.context_memory, "latest_validated_compaction", lambda _pid: None)
        with pytest.raises(ValidationError, match="current certificate"):
            _compact_context(runtime, pid)
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert runtime.store.get_llm_call(marker.call_id) == marker
        assert runtime.store.get_object(context_oid).version == original.version
        assert runtime.run_process_once(pid)["ok"]
        _assert_local_only(provider.requests[-1])
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["compaction", "append"])
@pytest.mark.parametrize("change", ["version", "read_denied"])
def test_pending_result_rejects_changed_source_before_context_transition(operation: str, change: str) -> None:
    runtime = Runtime.open("local", config=_compaction_config())
    try:
        provider = _Responses("hosted", "action", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        generation = runtime.store.get_llm_context_generation(pid)
        marker = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        oid = runtime.llm.context_memory.context_oid(pid)
        obj = runtime.store.get_object(oid)
        if change == "version":
            handle = runtime.memory.handle_for_oid(pid, oid, required_rights={"write"})
            runtime.memory.update_object(pid, handle, ObjectPatch(payload={**obj.payload, "external_change": True}))
        else:
            runtime.capability.issue_trusted(
                pid, f"object:{oid}", [CapabilityRight.READ],
                issued_by="test", effect=CapabilityEffect.DENY,
            )
        changed = runtime.store.get_object(oid)
        if operation == "compaction":
            with pytest.raises((ValidationError, CapabilityDenied)):
                runtime.llm.context_memory.replace_with_compacted_summary(
                    pid, context_oid=oid, expected_version=changed.version,
                    summary={"goal": "summary"}, compaction_method="test_compaction",
                    preserve_recent_entries=0, source_tokens=1000, target_tokens=512,
                    compressor_pids=[],
                )
        else:
            result = runtime.run_process_once(pid)
            assert not result["ok"], result
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation") == marker
        assert runtime.store.get_persisted_object_state(oid).version == changed.version
        assert len(provider.requests) == 1
    finally:
        runtime.close()
