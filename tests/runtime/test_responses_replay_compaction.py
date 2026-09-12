from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.llm.replay import ReplayStateError
from agent_libos.models import CapabilityEffect, CapabilityRight
from agent_libos.models.data_flow import DataFlowContext
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.utils.serde import dumps
from tests.runtime.test_responses_replay_executor import (
    CONFIG, SECRET, ReplayClient, completion, register,
)


SUMMARY = "CERTIFIED_REPLAY_COMPACTION_SUMMARY"


def _compact_context(runtime: Runtime, pid: str) -> dict[str, Any]:
    process = runtime.process.get(pid)
    handle = runtime.llm.context_memory.ensure(
        pid, runtime.images[process.image_id], process, runtime.tools.visible_tools(pid),
    )
    context = runtime.memory.get_object(pid, handle)
    runtime.llm.context_memory.replace_with_compacted_summary(
        pid,
        context_oid=context.oid,
        expected_version=context.version,
        summary={
            "goal": SUMMARY, "constraints": [], "user_preferences": [],
            "completed": [], "pending": [], "key_references": {},
            "recent_decisions": [], "risks": [], "uncertainties": [], "next_steps": [],
        },
        compaction_method="test_compaction",
        preserve_recent_entries=0,
        source_tokens=1000,
        target_tokens=512,
        compressor_pids=[],
    )
    certificate = runtime.llm.context_memory.latest_validated_compaction(pid)
    assert certificate is not None
    return certificate


@pytest.mark.parametrize("mode", ["libos_default", "image_only"])
@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
def test_certified_compaction_replaces_whole_replay_with_one_current_summary(
    mode: str, layout: str,
) -> None:
    config = replace(
        CONFIG,
        llm=replace(CONFIG.llm, prompt_layout=layout),
        llm_context=replace(CONFIG.llm_context, policy="llm_context_object"),
    )
    runtime = Runtime.open("local", config=config)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime, mode)
        assert runtime.run_process_once(pid)["ok"]
        _old_head, old_turn, old_payload = runtime.llm.replay.load_current(pid)
        certificate = _compact_context(runtime, pid)

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        assert len(client.inputs) == 2
        current_input = dumps(client.inputs[1])
        assert current_input.count(SUMMARY) == 1
        assert SECRET not in current_input
        assert not any(item.get("call_id") == "call_1" for item in client.inputs[1])
        _new_head, new_turn, new_payload = runtime.llm.replay.load_current(pid)
        assert new_turn.context_generation == certificate["context_generation"]
        assert new_turn.context_generation != old_turn.context_generation
        old_refs = {dumps(ref) for ref in old_payload["flow_context"]["source_refs"]
                    if ref["oid"] != certificate["context_oid"]}
        new_refs = {dumps(ref) for ref in new_payload["flow_context"]["source_refs"]}
        assert old_refs <= new_refs
        assert len(new_payload["groups"]) == 1
        assert new_payload["groups"][0]["validated"] is True
    finally:
        runtime.close()


def test_arbitrary_generation_change_cannot_discard_private_replay() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        original = runtime.store.get_llm_replay_head(pid)
        runtime.store.set_llm_context_generation(pid, "uncertified-generation")

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert "validated compaction" in result["error"]
        assert len(client.inputs) == 1
        assert runtime.store.get_llm_replay_head(pid) == original
    finally:
        runtime.close()


@pytest.mark.parametrize("validated", [False, True])
def test_compaction_rejects_pending_group_without_changing_private_head(validated: bool) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        _head, turn, payload = runtime.llm.replay.load_current(pid)
        flow = DataFlowContext.from_dict(payload["flow_context"])
        request = runtime.llm.replay.prepare(
            pid=pid, provider_fingerprint=turn.provider_fingerprint, model=turn.model,
            context_generation=turn.context_generation, messages=payload["prefix"],
            flow_context=flow,
        )
        pending = completion(2)
        runtime.llm.replay.stage(
            request, call_id="pending-call", response_items=pending.response_items,
            usage=pending.usage, max_output_tokens=64,
        )
        if validated:
            runtime.llm.replay.mark_validated(pid=pid, call_id="pending-call")
        original = runtime.store.get_llm_replay_head(pid)
        _compact_context(runtime, pid)
        state = SimpleNamespace(
            pid=pid, client=client,
            resolved=SimpleNamespace(identity_sha256=turn.provider_fingerprint),
            request_messages=list(payload["prefix"]), flow_context=flow,
        )

        with pytest.raises(ReplayStateError, match="pending tool group"):
            runtime.llm._prepare_responses_replay(state)

        assert runtime.store.get_llm_replay_head(pid) == original
        assert len(client.inputs) == 1
    finally:
        runtime.close()


def test_compaction_rolls_back_private_head_when_request_source_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        original = runtime.store.get_llm_replay_head(pid)
        _compact_context(runtime, pid)

        def reject_source(_state: Any) -> None:
            raise ReplayStateError("compacted source authority revoked")

        monkeypatch.setattr(runtime.llm, "_validate_replay_request_sources", reject_source)
        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert "authority revoked" in result["error"]
        assert runtime.store.get_llm_replay_head(pid) == original
        assert len(client.inputs) == 1
    finally:
        runtime.close()


def test_compaction_requires_current_read_authority_before_replacing_context_source() -> None:
    config = replace(CONFIG, llm_context=replace(CONFIG.llm_context, policy="llm_context_object"))
    runtime = Runtime.open("local", config=config)
    try:
        client = ReplayClient([completion(1)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        original, turn, payload = runtime.llm.replay.load_current(pid)
        certificate = _compact_context(runtime, pid)
        assert any(ref["oid"] == certificate["context_oid"]
                   for ref in payload["flow_context"]["source_refs"])
        runtime.capability.issue_trusted(
            pid, f"object:{certificate['context_oid']}", [CapabilityRight.READ],
            issued_by="test", effect=CapabilityEffect.DENY,
        )
        state = SimpleNamespace(
            pid=pid, client=client,
            resolved=SimpleNamespace(identity_sha256=turn.provider_fingerprint),
            request_messages=list(payload["prefix"]),
            flow_context=DataFlowContext.from_dict(payload["flow_context"]),
        )

        with pytest.raises(CapabilityDenied):
            runtime.llm._prepare_responses_replay(state)

        assert runtime.store.get_llm_replay_head(pid) == original
        assert len(client.inputs) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("drift", ["context_version", "source_version"])
def test_compaction_certificate_must_cover_exact_old_and_current_context_versions(
    monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    config = replace(CONFIG, llm_context=replace(CONFIG.llm_context, policy="llm_context_object"))
    runtime = Runtime.open("local", config=config)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        original = runtime.store.get_llm_replay_head(pid)
        _compact_context(runtime, pid)
        lookup = runtime.llm.context_memory.latest_validated_compaction

        def drifted_certificate(selected_pid: str) -> dict[str, Any]:
            certificate = lookup(selected_pid)
            assert certificate is not None
            certificate[drift] = certificate[drift] + 1 if drift == "context_version" else 0
            return certificate

        monkeypatch.setattr(runtime.llm.context_memory, "latest_validated_compaction", drifted_certificate)
        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert "compaction" in result["error"]
        assert runtime.store.get_llm_replay_head(pid) == original
        assert len(client.inputs) == 1
    finally:
        runtime.close()
