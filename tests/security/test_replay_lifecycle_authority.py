from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.llm.replay import ReplayStateError
from agent_libos.models import CapabilityEffect, CapabilityRight, DataFlowContext, ObjectPatch, ObjectType
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps, loads
from tests.runtime.test_responses_replay_executor import CONFIG, IMAGE, SECRET, ReplayClient, completion, register


CONTEXT_CONFIG = replace(CONFIG, llm_context=replace(CONFIG.llm_context, policy="llm_context_object"))


@pytest.mark.parametrize("change", ["version", "hash", "read-denied"])
def test_context_refresh_cannot_bypass_prior_source_integrity_or_read_authority(change: str) -> None:
    runtime = Runtime.open("local", config=CONTEXT_CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        head = runtime.store.get_llm_replay_head(pid)
        oid = runtime.llm.context_memory.context_oid(pid)
        obj = runtime.store.get_object(oid)
        if change == "read-denied":
            runtime.capability.issue_trusted(pid, f"object:{oid}", [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        else:
            payload = deepcopy(obj.payload)
            payload["unexpected_write"] = "not a Host context append"
            if change == "hash":
                runtime.store.set_object_payload(oid, payload)
            else:
                handle = next(handle for handle in runtime.process.get(pid).memory_view.roots if handle.oid == oid)
                runtime.memory.update_object(pid, handle, ObjectPatch(payload=payload))
        outcome = runtime.run_process_once(pid)

        assert not outcome["ok"]
        assert len(client.inputs) == 1
        assert runtime.store.get_llm_replay_head(pid) == head
        assert any(record.action == "llm.action_failed" for record in runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.audit.trace(actor=pid))
        assert SECRET not in dumps(runtime.events.list(target=pid))
    finally:
        runtime.close()


def test_context_refresh_preserves_other_mutable_source_checks() -> None:
    runtime = Runtime.open("local", config=CONTEXT_CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        source = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"text": "retained source"}, immutable=False)
        runtime.llm._add_to_view(pid, source)
        assert runtime.run_process_once(pid)["ok"]
        retained = DataFlowContext.from_dict(runtime.llm.replay.load_current(pid)[2]["flow_context"])
        original = next(ref for ref in retained.source_refs if ref.oid == source.oid)
        runtime.memory.update_object(pid, source, ObjectPatch(payload={"text": "changed after provider read"}))

        outcome = runtime.run_process_once(pid)

        assert not outcome["ok"]
        assert "source is unavailable or changed" in outcome["error"]
        assert len(client.inputs) == 1
        current = DataFlowContext.from_dict(runtime.llm.replay.load_current(pid)[2]["flow_context"])
        assert original in current.source_refs
        assert any(record.action == "llm.action_failed" for record in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


def test_context_update_rolls_back_with_private_head_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("local", config=CONTEXT_CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        head = runtime.store.get_llm_replay_head(pid)
        oid = runtime.llm.context_memory.context_oid(pid)
        obj = deepcopy(runtime.store.get_object(oid))
        monkeypatch.setattr(runtime.store, "compare_and_set_llm_replay_head", lambda *args, **kwargs: False)
        fail_quantum = runtime.llm._fail_llm_quantum
        rolled_back = []

        def inspect_before_exit(owner, error):
            after = runtime.store.get_object(oid)
            rolled_back.append((after.version, after.payload) == (obj.version, obj.payload))
            return fail_quantum(owner, error)

        monkeypatch.setattr(runtime.llm, "_fail_llm_quantum", inspect_before_exit)

        outcome = runtime.run_process_once(pid)

        assert not outcome["ok"]
        assert len(client.inputs) == 1
        assert runtime.store.get_llm_replay_head(pid) == head
        assert rolled_back == [True]
    finally:
        runtime.close()


def test_missing_exec_receipt_cannot_discard_pending_tool_output() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        runtime.llm.client = ReplayClient([completion(1)])
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        call = runtime.store.get_latest_successful_llm_call(pid=pid, purpose="action_selection")
        output = runtime.store.list_llm_tool_outputs(pid=pid, response_id=call.call_id)[0]
        runtime.store.clear_llm_replay_head(pid)
        with pytest.raises(ReplayStateError, match="no staged provider turn"):
            runtime.llm._persist_replay_tool_output(pid=pid, call=call, result=loads(output["output_text"]),
                response_id=call.call_id, tool_call_id="call_1", tool_name="echo")
    finally:
        runtime.close()


def test_exec_receipt_does_not_excuse_missing_later_replay() -> None:
    runtime = Runtime.open("local", config=CONFIG)
    try:
        client = ReplayClient([completion(1), completion(2), completion(3)])
        runtime.llm.client = client
        pid = register(runtime)
        assert runtime.run_process_once(pid)["ok"]
        runtime.exec_process(pid, IMAGE, goal="Use the new session.")
        assert runtime.run_process_once(pid)["ok"]
        runtime.store.clear_llm_replay_head(pid)
        outcome = runtime.run_process_once(pid)
        assert not outcome["ok"]
        assert "head is missing" in outcome["error"]
        assert len(client.inputs) == 2
        assert any(record.action == "llm.action_failed" for record in runtime.audit.trace(actor=pid))
    finally:
        runtime.close()


def test_actual_replay_still_requires_its_profile_at_startup(tmp_path: Path) -> None:
    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={**CONFIG.llm.profiles, "old": CONFIG.llm.profiles["default"]}))
    target = str(tmp_path / "actual-replay-profile.sqlite")
    runtime = Runtime.open(target, config=config)
    try:
        register(runtime)
        pid = runtime.process.spawn(image=IMAGE, goal="Keep native replay.", llm_profile_id="old")
        runtime.llms.set_test_client("old", ReplayClient([completion(1)]))
        assert runtime.run_process_once(pid)["ok"]
        assert runtime.store.get_llm_replay_head(pid) is not None
    finally:
        runtime.close()
    with pytest.raises(ValidationError, match="unknown LLM profile: old"):
        Runtime.open(target, config=CONFIG)
