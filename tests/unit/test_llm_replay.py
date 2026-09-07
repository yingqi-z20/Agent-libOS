from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from agent_libos.llm.replay import LLMReplayService, ReplayStateError, reasoning_token_bound
from agent_libos.models.data_flow import DataFlowContext, DataLabels, DataSourceRef
from agent_libos.models.exceptions import ValidationError
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.utils.serde import dumps


class ReplayStore:
    def __init__(self) -> None:
        self.turns = {}
        self.heads = {}

    @contextmanager
    def transaction(self):
        before = deepcopy((self.turns, self.heads))
        try:
            yield
        except Exception:
            self.turns, self.heads = before
            raise

    def insert_llm_replay_turn(self, turn):
        assert turn.turn_id not in self.turns
        self.turns[turn.turn_id] = deepcopy(turn)

    def get_llm_replay_turn(self, turn_id):
        return deepcopy(self.turns.get(turn_id))

    def get_llm_replay_head(self, pid):
        return self.heads.get(pid)

    def get_latest_successful_llm_call(self, **_kwargs):
        return None

    def get_latest_committed_exec_publication(self, pid):
        return None

    def compare_and_set_llm_replay_head(self, head, *, expected_revision):
        current = self.heads.get(head.pid)
        if (None if current is None else current.revision) != expected_revision:
            return False
        self.heads[head.pid] = head
        return True


def flow(secret=False):
    return DataFlowContext(
        labels=DataLabels(sensitivity="secret" if secret else "normal"),
        source_refs=(DataSourceRef(oid="object-a", version=1, content_sha256="a" * 64),),
    )


def prepare(service, text="hello 世界", **kwargs):
    return service.prepare(
        pid="p1", provider_fingerprint="provider", model="gpt-test",
        context_generation=kwargs.pop("context_generation", "initial"),
        messages=[{"role": "system", "content": "stable"}, {"role": "user", "content": text}],
        flow_context=kwargs.pop("flow_context", flow()), **kwargs,
    )


def outputs(*calls):
    return [
        {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "OPAQUE-SECRET"},
        *[{"type": "function_call", "id": f"fc{call}", "call_id": call, "name": "echo", "arguments": "{}"} for call in calls],
        {"type": "message", "id": "msg1", "role": "assistant", "content": [{"type": "output_text", "text": "working", "annotations": []}], "phase": "commentary"},
    ]


def stage(service, request, *calls, call_id="call-local"):
    head = service.stage(request, call_id=call_id, response_items=outputs(*calls), usage={"output_tokens_details": {"reasoning_tokens": 25}, "output_tokens": 40}, max_output_tokens=100, response_id="resp1")
    service.mark_validated(pid="p1", call_id=call_id)
    return head


def settle(service, *calls, call_id="call-local"):
    return service.settle(pid="p1", call_id=call_id, outputs=[{"type": "function_call_output", "call_id": call, "output": f"result:{call}"} for call in calls])


def test_opaque_replay_round_trip_preserves_order_and_tool_outputs_once():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    first = prepare(service)
    stage(service, first, "a", "b")
    settle(service, "b")
    with pytest.raises(ReplayStateError, match="durable tool outputs"):
        prepare(service)
    head = settle(service, "a")
    assert settle(service, "a") == head
    restored = LLMReplayService(store, max_bytes=1_000_000)
    second = prepare(restored, "next")
    assert second.response_items == [
        *first.response_items, *outputs("a", "b"),
        {"type": "function_call_output", "call_id": "a", "output": "result:a"},
        {"type": "function_call_output", "call_id": "b", "output": "result:b"},
        {"role": "user", "content": "next"},
    ]
    assert "OPAQUE-SECRET" not in repr(second)
    assert "OPAQUE-SECRET" not in dumps(second)


def test_staged_action_cannot_replay_until_validation_and_repair_discards_only_unexecuted():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    service.stage(prepare(service), call_id="bad", response_items=outputs(), usage={"output_tokens": 40}, max_output_tokens=100)
    with pytest.raises(ReplayStateError, match="durable tool outputs"):
        prepare(service)
    service.discard_staged(pid="p1", call_id="bad")
    repaired = prepare(service, "repair")
    assert not any(item.get("type") == "reasoning" for item in repaired.response_items)
    stage(service, repaired, "a")
    with pytest.raises(ReplayStateError, match="cannot discard"):
        service.discard_staged(pid="p1", call_id="call-local")


def test_scope_mismatch_and_changed_result_fail_closed():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    stage(service, prepare(service), "a")
    settle(service, "a")
    with pytest.raises(ReplayStateError, match="scope changed"):
        prepare(service, context_generation="changed")
    with pytest.raises(ReplayStateError, match="changed after publication"):
        service.settle(pid="p1", call_id="call-local", outputs=[{"type": "function_call_output", "call_id": "a", "output": "changed"}])
    with pytest.raises(ReplayStateError, match="no provider function call"):
        settle(service, "unknown")


def test_history_aggregates_labels_and_exact_source_references():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    stage(service, prepare(service, flow_context=flow(secret=True)))
    second = prepare(service)
    assert second.flow_context.labels.sensitivity.value == "secret"
    assert second.flow_context.source_refs == flow().source_refs


def test_host_wait_observation_preserves_native_history_and_deduplicates_retries():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    first = prepare(service)
    stage(service, first)
    observed = [{"role": "user", "content": "Host receive result: follow-up message"}]
    args = dict(pid="p1", call_id="call-local", input_items=observed, flow_context=flow(secret=True))
    head = service.append_host_input(**args)
    restored = LLMReplayService(store, max_bytes=1_000_000)
    assert restored.append_host_input(**args) == head
    next_request = prepare(restored, "next")
    assert next_request.response_items == [*first.response_items, *outputs(), *observed, {"role": "user", "content": "next"}]
    assert next_request.flow_context.labels.sensitivity.value == "secret"
    assert not any(item.get("type") == "function_call_output" for item in next_request.response_items)
    with pytest.raises(ReplayStateError, match="changed after publication"):
        restored.append_host_input(**{**args, "input_items": [{"role": "user", "content": "changed"}]})
    assert store.get_llm_replay_head("p1") == head


def test_host_wait_observation_cannot_replace_native_tool_output():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    stage(service, prepare(service), "native")
    with pytest.raises(ReplayStateError, match="cannot replace"):
        service.append_host_input(pid="p1", call_id="call-local", input_items=[{"role": "user", "content": "result"}])
    with pytest.raises(ReplayStateError, match="durable tool outputs"):
        prepare(service)


def test_token_estimate_uses_generation_usage_not_ciphertext_length():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    stage(service, prepare(service))
    short = prepare(service).estimated_input_tokens
    store = ReplayStore()
    other = LLMReplayService(store, max_bytes=1_000_000)
    large_items = outputs()
    large_items[0]["encrypted_content"] = "X" * 100_000
    other.stage(prepare(other), call_id="long", response_items=large_items, usage={"reasoning_tokens": 25}, max_output_tokens=100)
    other.mark_validated(pid="p1", call_id="long")
    assert prepare(other).estimated_input_tokens == short
    assert reasoning_token_bound(outputs(), {"output_tokens": 70}, 100) == 70
    assert reasoning_token_bound(outputs(), {}, 100) == 100
    assert reasoning_token_bound(outputs(), {"reasoning_tokens": 0}, 100) == 0


def test_compaction_replaces_only_complete_groups_and_requires_new_generation():
    service = LLMReplayService(ReplayStore(), max_bytes=1_000_000)
    stage(service, prepare(service), "a")
    compact_args = dict(pid="p1", context_generation="compacted", messages=[{"role": "system", "content": "stable"}, {"role": "user", "content": "validated summary"}], flow_context=flow())
    with pytest.raises(ReplayStateError, match="pending tool group"):
        service.compact(**compact_args)
    settle(service, "a")
    service.compact(**compact_args)
    request = prepare(service, "new state", context_generation="compacted")
    assert request.response_items == [{"role": "system", "content": "stable"}, {"role": "user", "content": "validated summary"}, {"role": "user", "content": "new state"}]


def test_checkpoint_refs_reject_pending_and_purged_payloads():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    stage(service, prepare(service), "a")
    with pytest.raises(ReplayStateError, match="incomplete"):
        service.capture_checkpoint_refs(["p1"])
    head = settle(service, "a")
    refs = service.capture_checkpoint_refs(["p1"])
    assert refs["p1"]["turn_id"] == head.turn_id
    original = store.turns[head.turn_id]
    store.turns[head.turn_id] = replace(original, payload=None, purged_at="now")
    with pytest.raises(ReplayStateError, match="purged"):
        service.rebind(turn_id=head.turn_id, pid="fork", context_generation="fork-generation", provider_fingerprint="provider", model="gpt-test", flow_context=flow())


def test_frozen_request_is_private_exact_and_rejects_changed_head():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    stage(service, prepare(service))
    request = prepare(service, "approval required")
    reference = service.freeze_request(request)
    assert "OPAQUE-SECRET" not in str(reference)
    loaded = service.load_request(reference, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial")
    assert loaded == request
    stage(service, request, call_id="next")
    with pytest.raises(ReplayStateError, match="head changed"):
        service.load_request(reference, pid="p1", provider_fingerprint="provider", model="gpt-test", context_generation="initial")


def test_competing_provider_turn_does_not_overwrite_committed_head():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    request = prepare(service)
    stage(service, request)
    count = len(store.turns)
    with pytest.raises(ReplayStateError, match="head changed"):
        stage(service, request, call_id="racing")
    assert len(store.turns) == count


def test_payload_byte_limit_rejects_without_publishing_partial_state():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=2_000)
    request = prepare(service)
    oversized = outputs()
    oversized[0]["encrypted_content"] = "X" * 3_000
    with pytest.raises(ReplayStateError, match="byte bound"):
        service.stage(request, call_id="large", response_items=oversized, usage={"output_tokens": 40}, max_output_tokens=100)
    assert not store.heads and not store.turns


def test_existing_conversation_never_silently_starts_without_private_head():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    store.get_latest_successful_llm_call = lambda **_kwargs: SimpleNamespace(
        call_id="call-local",
        request_options={"llm_context_generation": "initial", "responses_replay": {"enabled": True}}
    )
    with pytest.raises(ReplayStateError, match="head is missing"):
        prepare(service)
    assert prepare(service, context_generation="explicit-new-generation").response_items


def test_existing_head_cannot_skip_a_completed_call_when_replay_is_reenabled():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    stage(service, prepare(service))
    calls = []

    def latest(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(call_id="ordinary-call-while-replay-disabled", request_options={"llm_context_generation": "initial"})

    store.get_latest_successful_llm_call = latest
    head = store.get_llm_replay_head("p1")
    with pytest.raises(ReplayStateError, match="continuity is missing"):
        prepare(service)
    assert calls == [{"pid": "p1", "purpose": "action_selection"}]
    assert store.get_llm_replay_head("p1") == head
    service.compact(pid="p1", context_generation="certified-summary", messages=[{"role": "system", "content": "stable"}], flow_context=flow())
    assert prepare(service, context_generation="certified-summary").response_items


def test_continuity_accepts_a_rejected_action_tombstone():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    service.stage(prepare(service), call_id="rejected", response_items=outputs(), usage={}, max_output_tokens=100)
    service.discard_staged(pid="p1", call_id="rejected")
    store.get_latest_successful_llm_call = lambda **_kwargs: SimpleNamespace(call_id="rejected", request_options={"llm_context_generation": "initial"})
    request = prepare(service)
    assert not any(item.get("type") == "reasoning" for item in request.response_items)


def test_turn_limit_denies_admission_before_another_provider_request():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000, max_turns=1)
    stage(service, prepare(service))
    head = store.get_llm_replay_head("p1")
    with pytest.raises(ReplayStateError, match="semantic compaction"):
        prepare(service, "this must not reach the provider")
    assert store.get_llm_replay_head("p1") == head


def test_new_input_byte_bound_denies_admission_without_dropping_history():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=2_000)
    stage(service, prepare(service))
    head = store.get_llm_replay_head("p1")
    with pytest.raises(ReplayStateError, match="byte bound"):
        prepare(service, "large input" * 300)
    assert store.get_llm_replay_head("p1") == head


def recovery_manager(store):
    manager = SimpleNamespace(_store=store, config=DEFAULT_CONFIG, _member_pids=lambda _run_id: ["p1"])
    manager._prevalidate_replay_release = MethodType(TaskRunManager._prevalidate_replay_release, manager)
    manager._prevalidate_replay_call = MethodType(TaskRunManager._prevalidate_replay_call, manager)
    return manager


def test_task_run_recovery_requires_private_state_bound_to_public_call():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    request = prepare(service, run_id="run1")
    staged_head = stage(service, request)
    staged = store.get_llm_replay_turn(staged_head.turn_id)
    marker = {"schema_version": 1, "enabled": True, "turn_id": staged.turn_id, "payload_sha256": staged.payload_sha256}
    store.get_latest_successful_llm_call = lambda **_kwargs: SimpleNamespace(request_options={"responses_replay": marker}, call_id="call-local", status="ok")
    store.get_llm_pending_action = lambda _pid: None
    manager = recovery_manager(store)
    record = SimpleNamespace(run_id="run1")
    TaskRunManager._prevalidate_recoverable_llm_replay(manager, record)
    del store.heads["p1"]
    with pytest.raises(ValidationError, match="continuation is missing"):
        TaskRunManager._prevalidate_recoverable_llm_replay(manager, record)


def test_task_run_recovery_validates_pending_private_approval_request():
    store = ReplayStore()
    service = LLMReplayService(store, max_bytes=1_000_000)
    reference = service.freeze_request(prepare(service, run_id="run1"))
    store.get_latest_llm_call = lambda **_kwargs: None
    store.get_llm_pending_action = lambda _pid: {"action": {"responses_replay_request": reference}}
    manager = recovery_manager(store)
    record = SimpleNamespace(run_id="run1")
    TaskRunManager._prevalidate_recoverable_llm_replay(manager, record)
    del store.turns[reference["turn_id"]]
    with pytest.raises(ValidationError, match="release payload is missing"):
        TaskRunManager._prevalidate_recoverable_llm_replay(manager, record)
