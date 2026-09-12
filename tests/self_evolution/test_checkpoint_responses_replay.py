from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityEffect, CapabilityRight, DataFlowContext
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.runtime.checkpoint_replay import CheckpointReplayAdapter
from agent_libos.utils.serde import dumps


_OPAQUE = "PRIVATE_ENCRYPTED_RESPONSES_STATE_MUST_STAY_LOCAL"


@contextmanager
def _runtime(**kwargs):
    runtime = Runtime.open(":memory:", **kwargs)
    try:
        yield runtime
    finally:
        runtime.shutdown(actor="test", reason="done")


def _adapter(runtime: Runtime) -> CheckpointReplayAdapter:
    adapter = CheckpointReplayAdapter(
        runtime.uow,
        config=runtime.config,
        capabilities=runtime.capability,
        data_flow=runtime.data_flow,
        filesystem=runtime.filesystem,
        profile_snapshot=runtime.llms.profile_snapshot,
    )
    runtime.checkpoint.bind_responses_replay(adapter)
    return adapter


def _seed(runtime: Runtime, adapter: CheckpointReplayAdapter, pid: str, *, source: bool = False, pending: bool = False) -> str:
    process = runtime.store.get_process(pid)
    profile = runtime.llms.profile_snapshot(process.llm_profile_id)
    context = (
        runtime.data_flow.context_from_source_oids(pid, [process.goal_oid], include_current=False)
        if source else DataFlowContext()
    )
    request = adapter.service.prepare(
        pid=pid,
        provider_fingerprint=profile.identity_sha256,
        model=profile.policy.model,
        context_generation=runtime.store.get_llm_context_generation(pid),
        messages=[{"role": "user", "content": "Continue the local conversation."}],
        flow_context=context,
    )
    output = [
        {"type": "reasoning", "id": "rs_private", "encrypted_content": _OPAQUE, "summary": []},
        {"type": "message", "id": "msg_private", "role": "assistant", "phase": "final_answer", "content": "Ready."},
    ]
    if pending:
        output.append({"type": "function_call", "call_id": "fc_pending", "name": "some_tool", "arguments": "{}"})
    adapter.service.stage(request, call_id="local_call", response_items=output, usage={"reasoning_tokens": 7}, max_output_tokens=64)
    head = adapter.service.mark_validated(pid=pid, call_id="local_call")
    return head.turn_id


def test_checkpoint_restore_rebinds_private_replay_after_restart(tmp_path: Path) -> None:
    target = str(tmp_path / "runtime.sqlite")
    runtime = Runtime.open(target)
    try:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="local replay")
        original_turn_id = _seed(runtime, adapter, pid)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        _, snapshot = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert snapshot["responses_replay_refs"][pid]["turn_id"] == original_turn_id
        assert _OPAQUE not in dumps(snapshot)
        assert "responses_replay_refs" not in dumps(runtime.checkpoint.inspect(checkpoint_id, require_capability=False))
    finally:
        runtime.shutdown(actor="test", reason="reopen")
    reopened = Runtime.open(target)
    try:
        adapter = _adapter(reopened)
        result = reopened.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert result["main_state_committed"]
        _head, turn, payload = adapter.service.load_current(pid)
        assert turn.turn_id != original_turn_id
        assert turn.context_generation == reopened.store.get_llm_context_generation(pid)
        assert _OPAQUE in dumps(payload)
    finally:
        reopened.shutdown(actor="test", reason="done")


@pytest.mark.parametrize("missing", [False, True], ids=["purged", "missing"])
def test_checkpoint_restore_refuses_purged_or_missing_replay_before_effects(monkeypatch: pytest.MonkeyPatch, missing: bool) -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="no resurrection")
        turn_id = _seed(runtime, adapter, pid)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        if missing:
            original = runtime.store.get_llm_replay_turn
            monkeypatch.setattr(runtime.store, "get_llm_replay_turn", lambda selected: None if selected == turn_id else original(selected))
        else:
            runtime.store.purge_llm_replay(pid=pid)
        called = []
        monkeypatch.setattr(runtime.checkpoint, "_prepare_durable_restore_finalizers", lambda *args, **kwargs: called.append("finalizers"))
        before = runtime.store.get_process(pid)
        with pytest.raises(ValidationError, match="(missing|purged|unavailable)"):
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert called == []
        assert runtime.store.get_process(pid) == before


def test_checkpoint_restore_refuses_changed_provider_scope_before_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="provider scope")
        _seed(runtime, adapter, pid)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        original = adapter.profile_snapshot
        adapter.profile_snapshot = lambda profile_id: SimpleNamespace(identity_sha256="f" * 64, policy=original(profile_id).policy)
        before = runtime.store.get_process(pid)
        with pytest.raises(ValidationError, match="provider scope changed"):
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert runtime.store.get_process(pid) == before


def test_checkpoint_replay_rechecks_source_read_authority_for_restore_and_fork() -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="private source")
        _seed(runtime, adapter, pid, source=True)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        goal_oid = runtime.store.get_process(pid).goal_oid
        runtime.capability.issue_trusted(pid, f"object:{goal_oid}", [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        with pytest.raises(CapabilityDenied):
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        with pytest.raises(CapabilityDenied):
            runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id, require_capability=False)


def test_checkpoint_fork_rebinds_replay_and_image_commit_excludes_it() -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="fork private state")
        original_turn_id = _seed(runtime, adapter, pid)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        fork = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id, require_capability=False)
        fork_pid = fork["fork_root_pid"]
        _head, turn, payload = adapter.service.load_current(fork_pid)
        assert turn.pid == fork_pid and turn.run_id is None
        assert turn.turn_id != original_turn_id
        assert _OPAQUE in dumps(payload)
        image = runtime.image_registry.commit_from_checkpoint(
            actor=pid, checkpoint_id=checkpoint_id, image_id="private-replay-free:v0", name="Private replay free", require_capability=False,
        )
        artifact, _ = runtime.store.get_image_artifact(image.image.boot["artifact_id"])
        encoded = dumps(artifact)
        assert "responses_replay_refs" not in encoded
        assert _OPAQUE not in encoded
        assert original_turn_id not in encoded


def test_checkpoint_refuses_incomplete_private_replay_group() -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="pending tool")
        _seed(runtime, adapter, pid, pending=True)
        with pytest.raises(ValidationError, match="(pending|unfinished|unresolved|incomplete)"):
            runtime.checkpoint.create(pid, "unsafe tool boundary", actor=pid)


def test_checkpoint_fork_preserves_authorized_source_lineage() -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="fork readable source")
        _seed(runtime, adapter, pid, source=True)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        fork = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id, require_capability=False)
        _head, turn, payload = adapter.service.load_current(fork["fork_root_pid"])
        context = DataFlowContext.from_dict(payload["flow_context"])
        assert context.source_refs[0].oid == fork["object_map"][runtime.store.get_process(pid).goal_oid]
        assert turn.source_labels == context.labels.to_dict()


def test_checkpoint_replay_rebind_rolls_back_with_failed_restore_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="atomic replay restore")
        _seed(runtime, adapter, pid)
        checkpoint_id = runtime.checkpoint.create(pid, "private replay", actor=pid)
        before_head = runtime.store.get_llm_replay_head(pid)
        before_count = runtime.store._query("SELECT count(*) AS n FROM llm_replay_turns")[0]["n"]
        def fail(**kwargs):
            raise RuntimeError("injected restore evidence failure")
        monkeypatch.setattr(runtime.checkpoint, "_record_restore_commit_evidence", fail)
        with pytest.raises(RuntimeError, match="injected restore evidence"):
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert runtime.store.get_llm_replay_head(pid) == before_head
        assert runtime.store._query("SELECT count(*) AS n FROM llm_replay_turns")[0]["n"] == before_count


def test_old_checkpoint_clears_head_without_purging_retained_replay() -> None:
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="old checkpoint")
        checkpoint_id = runtime.checkpoint.create(pid, "before replay", actor=pid)
        turn_id = _seed(runtime, adapter, pid)
        runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
        assert runtime.store.get_llm_replay_head(pid) is None
        assert runtime.store.get_llm_replay_turn(turn_id).payload is not None


def test_payload_retention_disabled_does_not_capture_private_replay() -> None:
    config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False))
    with _runtime() as runtime:
        adapter = _adapter(runtime)
        pid = runtime.process.spawn(image="base-agent:v0", goal="no persisted IO")
        _seed(runtime, adapter, pid)
        adapter.config = config
        checkpoint_id = runtime.checkpoint.create(pid, "no private replay", actor=pid)
        _, snapshot = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert "responses_replay_refs" not in snapshot
