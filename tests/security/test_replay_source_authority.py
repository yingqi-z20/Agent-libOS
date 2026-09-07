from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.replay import LLMReplayService
from agent_libos.models import CapabilityEffect, CapabilityRight, DataFlowContext, ObjectPatch, ObjectType
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.runtime.replay_source_recovery import LLMReplaySourceRecovery
from tests.support.runtime import workspace_runtime


def _validate(runtime, pid: str, context: DataFlowContext, **kwargs) -> None:
    runtime.data_flow.validate_replay_sources(
        pid, context, file_resource_resolver=runtime.filesystem.resource_for, **kwargs,
    )


def _retain_history(runtime, pid: str, context: DataFlowContext) -> None:
    service = LLMReplayService(runtime.uow.processes, max_bytes=runtime.config.llm.responses_replay_max_bytes)
    profile = runtime.llms.profile_snapshot(runtime.store.get_process(pid).llm_profile_id)
    request = service.prepare(
        pid=pid, provider_fingerprint=profile.identity_sha256, model=profile.policy.model,
        context_generation=runtime.store.get_llm_context_generation(pid),
        messages=[{"role": "user", "content": "Remember the source."}], flow_context=context,
    )
    service.stage(request, call_id="recovery-source-call", response_items=[{"type": "reasoning", "summary": [], "encrypted_content": "PRIVATE_SOURCE_RECOVERY"}], usage={"reasoning_tokens": 3}, max_output_tokens=32)
    service.mark_validated(pid=pid, call_id="recovery-source-call")


def test_private_replay_requires_current_object_read_authority() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="source authority")
        goal_oid = runtime.store.get_process(pid).goal_oid
        context = runtime.data_flow.context_from_source_oids(pid, [goal_oid], include_current=False)
        _validate(runtime, pid, context)
        runtime.capability.issue_trusted(pid, f"object:{goal_oid}", [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        with pytest.raises(CapabilityDenied):
            _validate(runtime, pid, context)


def test_private_replay_rejects_changed_object_and_validates_host_captured_snapshot() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="source integrity")
        handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "original"}, immutable=False)
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid], include_current=False)
        original = runtime.store.get_object(handle.oid)
        runtime.memory.update_object(pid, handle, ObjectPatch(payload={"value": "changed"}))
        with pytest.raises(ValidationError, match="source is unavailable or changed"):
            _validate(runtime, pid, context)
        _validate(runtime, pid, context, captured_objects={handle.oid: (original.version, original.payload)})
        with pytest.raises(ValidationError, match="does not match its snapshot"):
            _validate(runtime, pid, context, captured_objects={handle.oid: (original.version, {"value": "forged"})})
        runtime.capability.issue_trusted(pid, f"object:{handle.oid}", [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        with pytest.raises(CapabilityDenied):
            _validate(runtime, pid, context, captured_objects={handle.oid: (original.version, original.payload)})


def test_private_replay_requires_current_file_source_read_authority() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="file source authority")
        path = "replay-source.txt"
        (root / path).write_text("private source", encoding="utf-8")
        runtime.data_flow.bind_written_file(pid=pid, normalized_path=path, content=b"private source", context=DataFlowContext())
        runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by="test")
        context = runtime.data_flow.file_context(path)
        _validate(runtime, pid, context)
        resource = runtime.filesystem.resource_for(path)
        runtime.capability.issue_trusted(pid, resource, [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        with pytest.raises(CapabilityDenied):
            _validate(runtime, pid, context)


@pytest.mark.parametrize("change", ["version", "hash", "superseded"])
def test_private_replay_rejects_changed_file_binding(change: str) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="file source integrity")
        path = "replay-source.txt"
        (root / path).write_text("private source", encoding="utf-8")
        runtime.data_flow.bind_written_file(pid=pid, normalized_path=path, content=b"private source", context=DataFlowContext())
        runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by="test")
        context = runtime.data_flow.file_context(path)
        reference = context.source_refs[0]
        if change == "version":
            context = replace(context, source_refs=(replace(reference, version=reference.version + 1),))
        elif change == "hash":
            context = replace(context, source_refs=(replace(reference, content_sha256="0" * 64),))
        else:
            runtime.data_flow.bind_written_file(pid=pid, normalized_path=path, content=b"replacement", context=DataFlowContext())
        with pytest.raises(ValidationError, match="(source is unavailable or changed|binding is unavailable)"):
            _validate(runtime, pid, context)


def test_private_replay_recovery_exception_preserves_current_read_authority(tmp_path: Path) -> None:
    target = str(tmp_path / "replay-sources.sqlite")
    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="recover private source")
        handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "private volatile source"})
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid], include_current=False)
    finally:
        runtime.close()
    reopened = Runtime.open(target)
    try:
        state = reopened.store.get_persisted_object_state(handle.oid)
        assert state.recovered_after_reopen and not state.payload_present
        # General volatile recovery revokes Object grants. A Host retaining
        # certified replay must independently preserve/revalidate READ first.
        with pytest.raises(CapabilityDenied):
            _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
        reopened.capability.issue_trusted(pid, f"object:{handle.oid}", [CapabilityRight.READ], issued_by="test")
        with pytest.raises(ValidationError, match="source is unavailable or changed"):
            _validate(reopened, pid, context)
        _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
        mismatched = replace(context, source_refs=(replace(context.source_refs[0], version=context.source_refs[0].version + 1),))
        with pytest.raises(ValidationError, match="source is unavailable or changed"):
            _validate(reopened, pid, mismatched, allow_recovered_source_snapshots=True)
        reopened.capability.issue_trusted(pid, f"object:{handle.oid}", [CapabilityRight.READ], issued_by="test", effect=CapabilityEffect.DENY)
        with pytest.raises(CapabilityDenied):
            _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
    finally:
        reopened.close()


def test_private_replay_recovery_exception_does_not_accept_explicit_source_deletion() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="deleted private source")
        handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "deleted"})
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid], include_current=False)
        runtime.memory.delete_object_trusted("test", handle.oid, reason="explicit deletion")
        runtime.capability.issue_trusted(pid, f"object:{handle.oid}", [CapabilityRight.READ], issued_by="test")
        with pytest.raises(ValidationError, match="source is unavailable or changed"):
            _validate(runtime, pid, context, allow_recovered_source_snapshots=True)


@pytest.mark.parametrize("revoked_before", [False, True], ids=["active-read-only", "user-revoked"])
def test_startup_private_replay_retains_only_existing_read_authority(tmp_path: Path, revoked_before: bool) -> None:
    target = str(tmp_path / "retained-source.sqlite")
    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="retain source READ")
        handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "private"})
        extra = runtime.capability.issue_trusted(pid, f"object:{handle.oid}", [CapabilityRight.READ, CapabilityRight.WRITE], issued_by="test", delegable=True)
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid], include_current=False)
        _retain_history(runtime, pid, context)
        if revoked_before:
            runtime.capability.revoke_resource_trusted(f"object:{handle.oid}", revoked_by="test", reason="explicit revoke before restart")
    finally:
        runtime.close()
    reopened = Runtime.open(target)
    try:
        if not revoked_before:
            retained = reopened.store.get_capability(extra.cap_id)
            assert retained.active and retained.rights == {CapabilityRight.READ.value}
            assert not retained.delegable
            _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
            reopened.capability.revoke_resource_trusted(f"object:{handle.oid}", revoked_by="test", reason="explicit revoke after restart")
        else:
            assert not reopened.store.get_capability(extra.cap_id).active
        with pytest.raises(CapabilityDenied):
            _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
        denied = [row for row in reopened.audit.trace(actor=pid) if row.action == "capability.authorize" and row.target == f"object:{handle.oid}" and row.decision.get("allowed") is False]
        assert denied
    finally:
        reopened.close()


def test_startup_does_not_retain_replay_source_authority_when_full_io_is_disabled(tmp_path: Path) -> None:
    target = str(tmp_path / "no-source-retention.sqlite")
    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="disable source retention")
        handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "private"})
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid], include_current=False)
        _retain_history(runtime, pid, context)
    finally:
        runtime.close()
    config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False))
    reopened = Runtime.open(target, config=config)
    try:
        assert not reopened.store.get_capability(handle.capability_id).active
        with pytest.raises(CapabilityDenied):
            _validate(reopened, pid, context, allow_recovered_source_snapshots=True)
    finally:
        reopened.close()


def test_private_replay_recovery_decodes_history_once_across_object_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded replay recovery")
        handles = [runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": index}) for index in range(3)]
        context = runtime.data_flow.context_from_source_oids(pid, [handle.oid for handle in handles], include_current=False)
        _retain_history(runtime, pid, context)
        recovery = LLMReplaySourceRecovery(runtime.uow, config=runtime.config, capabilities=runtime.capability, profile_snapshot=runtime.llms.profile_snapshot)
        reads = []
        original = recovery.service.validate_turn
        def validate(turn):
            reads.append(turn.turn_id)
            return original(turn)
        monkeypatch.setattr(recovery.service, "validate_turn", validate)
        recovery.preflight()
        assert len(reads) == 1
        for handle in handles:
            assert recovery.retained_read_capabilities((handle.oid,))[(pid, handle.oid)]
        assert len(reads) == 1


@pytest.mark.parametrize("parent_has_history", [False, True])
def test_private_replay_recovery_never_invents_delegation_ancestor_authority(tmp_path: Path, parent_has_history: bool) -> None:
    target = str(tmp_path / "delegated-source.sqlite")
    runtime = Runtime.open(target)
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="source owner")
        child = runtime.process.spawn(image="base-agent:v0", goal="source reader")
        handle = runtime.memory.create_object(parent, ObjectType.ARTIFACT, {"value": "delegated source"})
        runtime.capability.issue_trusted(parent, f"object:{handle.oid}", [CapabilityRight.READ], issued_by="test", delegable=True)
        delegated = runtime.capability.delegate(parent, child, {"resource": f"object:{handle.oid}", "rights": ["read"]})
        context = runtime.data_flow.context_from_source_oids(child, [handle.oid], include_current=False)
        _retain_history(runtime, child, context)
        if parent_has_history:
            _retain_history(runtime, parent, context)
    finally:
        runtime.close()
    reopened = Runtime.open(target)
    try:
        assert reopened.store.get_capability(delegated.cap_id).active is parent_has_history
        if parent_has_history:
            _validate(reopened, child, context, allow_recovered_source_snapshots=True)
        else:
            with pytest.raises(CapabilityDenied):
                _validate(reopened, child, context, allow_recovered_source_snapshots=True)
    finally:
        reopened.close()
