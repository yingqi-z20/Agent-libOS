from __future__ import annotations

from dataclasses import replace
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.api.cli import (
    _handle_interactive_line,
    _show_pending_interactive_human_request,
)
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, DataFlowDefaults
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataFlowDirection,
    DataIntegrity,
    DataLabels,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequestStatus,
    ObjectMetadata,
    ObjectPatch,
    ObjectType,
    ProcessStatus,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.sdk import (
    ProtectedOperationContract,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderPhase,
    ResourcePolicy,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    HumanResponseRequired,
)
from agent_libos.substrate import LocalResourceProviderSubstrate, ProviderEffectNotStarted
from tests.support.runtime import workspace_runtime


class _IntegrityPolicyProvider:
    def classify_external_effect(
        self,
        _operation: str,
        _context: dict[str, Any],
        _result: Any,
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
        )


def _secret_source(runtime: Any, pid: str, *, mutable: bool = False):
    return runtime.memory.create_object(
        pid,
        ObjectType.EVIDENCE,
        {"value": "DATA_FLOW_SECRET_SENTINEL"},
        metadata=ObjectMetadata(sensitivity="secret"),
        immutable=not mutable,
    )


def _register_file_sink(
    runtime: Any,
    path: str,
    *,
    trust_level: SinkTrustLevel,
) -> None:
    runtime.data_flow.register_sink_trust(
        SinkTrustRule(
            pattern=runtime.filesystem.resource_for_path(path),
            trust_level=trust_level,
            max_sensitivity="secret",
        ),
        actor="test.host",
        replace=runtime.data_flow.inspect_sink_trust(
            runtime.filesystem.resource_for_path(path)
        )
        is not None,
        require_capability=False,
    )


def _count_filesystem_boundaries(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"state": 0, "write": 0}
    original_state = runtime.filesystem.provider.state
    original_write = runtime.filesystem.provider.write_text

    def state(*args: Any, **kwargs: Any):
        calls["state"] += 1
        return original_state(*args, **kwargs)

    def write(*args: Any, **kwargs: Any):
        calls["write"] += 1
        return original_write(*args, **kwargs)

    monkeypatch.setattr(runtime.filesystem.provider, "state", state)
    monkeypatch.setattr(runtime.filesystem.provider, "write_text", write)
    return calls


def test_protected_write_rejects_untrusted_integrity_before_provider_dispatch() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="block untrusted instructions from writes")
        resource = "test:integrity-protected-write"
        runtime.capability.issue_trusted(
            pid,
            resource,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=resource,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        decision = runtime.capability.require(
            pid,
            resource,
            CapabilityRight.WRITE,
            consume=False,
        )
        contract = ProtectedOperationContract(
            name="primitive.test.integrity_protected_write",
            provider="test",
            operation="write",
            evidence_roles=("audit", "event", "effect"),
            resource_policy=ResourcePolicy.NONE,
            state_mutation=True,
            information_flow=True,
            data_flow_direction=DataFlowDirection.EGRESS,
            minimum_egress_integrity=DataIntegrity.UNKNOWN,
        )
        runtime.protected_operations.register_contract(contract)
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args={"value": "payload"},
            observation={"value_sha256": "0" * 64},
            data_sink=DataSink(resource),
            data_flow_context=DataFlowContext(
                labels=DataLabels(
                    trust_level="untrusted",
                    integrity="untrusted",
                    origin="external:prompt-injection",
                )
            ),
            data_flow_payload={"value_sha256": "0" * 64},
            data_flow_operation="test.integrity_protected_write",
        )
        provider_calls = 0

        def provider_write() -> str:
            nonlocal provider_calls
            provider_calls += 1
            return "written"

        with pytest.raises(
            CapabilityDenied,
            match="data integrity untrusted is below operation minimum unknown",
        ):
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_IntegrityPolicyProvider(),
            ) as operation:
                result = operation.call(
                    ProviderPhase(
                        "write",
                        state_mutation=True,
                        information_flow=True,
                    ),
                    provider_write,
                )
                operation.complete(
                    result,
                    ProtectedOperationEvidence(
                        event_type=EventType.EXTERNAL_WRITE,
                        event_source=pid,
                        audit_action="primitive.test.integrity_protected_write",
                        audit_actor=pid,
                    ),
                )

        assert provider_calls == 0
        assert runtime.store.list_external_effects(pid=pid) == []
        denials = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert len(denials) == 1
        assert denials[0].labels.integrity is DataIntegrity.UNTRUSTED
        assert any(
            record.action == "data_flow.egress"
            and record.target == resource
            and record.decision.get("decision_id") == denials[0].decision_id
            and record.decision.get("outcome") == "deny"
            for record in runtime.audit.trace()
        )
        assert any(
            event.type == EventType.DATA_FLOW_DECISION
            and event.payload.get("decision_id") == denials[0].decision_id
            and event.payload.get("outcome") == "deny"
            for event in runtime.events.list(target=f"data_flow_sink:{resource}")
        )


def test_ordinary_data_flow_evidence_retains_source_identity_by_default() -> None:
    """Semantic classifier redaction must not alter ordinary DataFlow evidence."""

    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="preserve ordinary DataFlow provenance")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "ordinary provenance"},
            metadata=ObjectMetadata(
                origin="test:ordinary-origin",
                tenant="tenant-a",
                principal="analyst-a",
            ),
        )
        context = runtime.data_flow.context_from_source_oids(pid, [source.oid])
        sink = DataSink("test:ordinary-egress")
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=sink.identity,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=("tenant-a",),
                principals=("analyst-a",),
            ),
            actor="test.host",
            require_capability=False,
        )

        decision, release = runtime.data_flow.authorize_egress(
            pid=pid,
            sink=sink,
            context=context,
            payload={"size": 1},
            operation="test.ordinary_egress",
        )

        assert release is None
        assert decision.source_refs == context.source_refs
        assert decision.labels == context.labels
        assert context.labels.origin is not None
        assert context.labels.tenant == "tenant-a"
        assert context.labels.principal == "analyst-a"
        event = next(
            item
            for item in runtime.events.list(target=f"data_flow_sink:{sink.identity}")
            if item.payload.get("decision_id") == decision.decision_id
        )
        audit = next(
            item
            for item in runtime.audit.trace()
            if item.action == "data_flow.egress"
            and (item.decision or {}).get("decision_id") == decision.decision_id
        )
        assert event.payload["source_refs"] == [
            item.to_dict() for item in context.source_refs
        ]
        assert event.payload["labels"] == context.labels.to_dict()
        assert audit.input_refs == [source.oid]


def test_configured_integrity_floor_blocks_real_filesystem_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        data_flow=replace(
            DEFAULT_CONFIG.data_flow,
            operation_minimum_integrity={
                "primitive.filesystem.write_text": DataIntegrity.UNKNOWN,
            },
        ),
    )
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        pid = runtime.process.spawn(goal="contain untrusted write instructions")
        path = "contained.txt"
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.data_flow.observe_ingress(
            DataFlowContext(
                labels=DataLabels(
                    trust_level="untrusted",
                    integrity="untrusted",
                    origin="external:prompt-injection",
                )
            )
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(
            CapabilityDenied,
            match="data integrity untrusted is below operation minimum unknown",
        ):
            runtime.filesystem.write_text(pid, path, "blocked")

        assert calls == {"state": 0, "write": 0}
        assert not (tmp_path / path).exists()
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_untrusted_secret_file_egress_denies_before_state_and_preserves_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="deny labeled file egress")
        source = _secret_source(runtime, pid)
        path = "exports/secret.txt"
        resource = runtime.filesystem.resource_for_path(path)
        once = runtime.capability.grant_once(
            pid,
            resource,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )

        assert calls == {"state": 0, "write": 0}
        assert not (root / path).exists()
        assert runtime.store.get_capability(once.cap_id).uses_remaining == 1
        assert runtime.store.list_external_effects(pid=pid) == []
        assert runtime.human.pending() == []
        decisions = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert len(decisions) == 1
        record = next(
            item for item in runtime.audit.trace() if item.action == "data_flow.egress"
        )
        assert "DATA_FLOW_SECRET_SENTINEL" not in str(record.decision)


def test_trusted_file_sink_allows_secret_but_still_requires_ordinary_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="trusted file egress")
        source = _secret_source(runtime, pid)
        path = "exports/trusted.txt"
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises((CapabilityDenied, HumanApprovalRequired)):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )
        assert calls == {"state": 0, "write": 0}

        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        result = runtime.filesystem.write_text(
            pid,
            path,
            "DATA_FLOW_SECRET_SENTINEL",
            source_oids=[source.oid],
        )

        assert result.bytes_written > 0
        assert calls == {"state": 2, "write": 1}
        assert (root / path).read_text(encoding="utf-8") == "DATA_FLOW_SECRET_SENTINEL"
        binding = runtime.store.get_file_label_binding(path)
        assert binding is not None and binding.labels.sensitivity.value == "secret"
        effect = runtime.store.list_external_effects(pid=pid)[-1]
        assert effect.provider_metadata["data_flow"]["trust_id"] is not None


def test_trusted_sink_does_not_bypass_task_effect_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        path = "exports/effect-ceiling.txt"
        pid = runtime.process.spawn(
            goal="trusted Sink still obeys effect ceiling",
            authority_manifest={
                "authorized_capabilities": [],
                "permitted_effects": ["jsonrpc.*"],
            },
        )
        source = _secret_source(runtime, pid)
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(CapabilityDenied, match="effect class"):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )

        assert calls == {"state": 0, "write": 0}
        assert runtime.store.list_external_effects(pid=pid) == []


def test_conditional_file_sink_uses_exact_once_release_and_rejects_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="conditional file egress")
        source = _secret_source(runtime, pid)
        path = "exports/conditional.txt"
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.CONDITIONAL)
        ordinary = runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )
        assert calls == {"state": 0, "write": 0}
        assert runtime.store.get_capability(ordinary.cap_id).uses_remaining == 1
        pending = runtime.human.pending()
        assert len(pending) == 1
        assert pending[0].payload["type"] == "data_release_approval"
        assert pending[0].blocking is True
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert "DATA_FLOW_SECRET_SENTINEL" not in str(pending[0].payload)

        runtime.human.drain_terminal_queue(auto_approve=True)
        runtime.filesystem.write_text(
            pid,
            path,
            "DATA_FLOW_SECRET_SENTINEL",
            source_oids=[source.oid],
        )
        assert calls == {"state": 2, "write": 1}
        release_caps = [
            cap
            for cap in runtime.capability.capabilities_for(pid)
            if cap.resource.startswith("data_release:")
        ]
        assert len(release_caps) == 1 and release_caps[0].uses_remaining == 0

        runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )
        assert calls == {"state": 2, "write": 1}


def test_conditional_release_is_invalid_after_payload_source_or_registry_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="exact release binding")
        source = _secret_source(runtime, pid, mutable=True)
        path = "exports/exact.txt"
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.CONDITIONAL)
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid, path, "first", source_oids=[source.oid]
            )
        runtime.human.drain_terminal_queue(auto_approve=True)

        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid, path, "changed", source_oids=[source.oid]
            )
        assert calls == {"state": 0, "write": 0}
        runtime.human.drain_terminal_queue(auto_approve=True)

        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={"value": "new source version"}),
        )
        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid, path, "changed", source_oids=[source.oid]
            )
        assert calls == {"state": 0, "write": 0}
        runtime.human.drain_terminal_queue(auto_approve=True)

        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.CONDITIONAL)
        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid, path, "changed", source_oids=[source.oid]
            )
        assert calls == {"state": 0, "write": 0}


def test_source_change_during_prepare_is_revalidated_before_provider_and_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="source toctou")
        source = _secret_source(runtime, pid, mutable=True)
        path = "exports/toctou.txt"
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        once = runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)
        original_intent = runtime.filesystem._record_mutation_intent

        def mutate_source(*args: Any, **kwargs: Any):
            runtime.memory.update_object(
                pid,
                source,
                ObjectPatch(payload={"value": "changed between gate and dispatch"}),
            )
            return original_intent(*args, **kwargs)

        monkeypatch.setattr(runtime.filesystem, "_record_mutation_intent", mutate_source)
        with pytest.raises(CapabilityDenied, match="source Object changed"):
            runtime.filesystem.write_text(
                pid, path, "payload", source_oids=[source.oid]
            )

        assert calls == {"state": 0, "write": 0}
        assert runtime.store.get_capability(once.cap_id).uses_remaining == 1
        assert runtime.store.list_external_effects(pid=pid) == []


def test_same_runtime_source_release_invalidates_ordinary_egress() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="invalidate a released egress source")
        source = _secret_source(runtime, pid)
        context = runtime.data_flow.context_from_trusted_source_oids([source.oid])
        path = "exports/released-source.txt"
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        assert runtime.memory.delete_object_trusted(
            "test.host",
            source.oid,
            reason="same-runtime source invalidation regression",
        )
        assert not runtime.store.is_recovered_object_payload(source.oid)
        with runtime.data_flow.recovered_source_snapshot_access():
            assert "no longer live" in str(
                runtime.data_flow._validate_source_refs(context.source_refs)
            )

        with runtime.data_flow.activate(context):
            with pytest.raises(CapabilityDenied, match="source Object"):
                runtime.filesystem.write_text(
                    pid,
                    path,
                    "DATA_FLOW_SECRET_SENTINEL",
                )
        assert not (root / path).exists()


def test_reopen_recovery_marker_does_not_enable_ordinary_egress(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    database = tmp_path / "runtime.sqlite"
    path = "exports/recovered-source.txt"
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="keep recovered snapshots resume-only")
        source = _secret_source(runtime, pid)
        context = runtime.data_flow.context_from_trusted_source_oids([source.oid])
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        assert reopened.store.is_recovered_object_payload(source.oid)
        assert "no longer live" in str(
            reopened.data_flow._validate_source_refs(context.source_refs)
        )
        with reopened.data_flow.activate(context):
            with pytest.raises(CapabilityDenied, match="source Object"):
                reopened.filesystem.write_text(
                    pid,
                    path,
                    "DATA_FLOW_SECRET_SENTINEL",
                )
        assert not (root / path).exists()
    finally:
        reopened.close()


def test_persisted_file_label_source_allows_unchanged_egress_after_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    database = tmp_path / "runtime.sqlite"
    source_path = "vault/persisted-secret.txt"
    target_path = "exports/after-reopen.txt"
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="reuse a durable labeled file after reopen")
        source = _secret_source(runtime, pid)
        for path in (source_path, target_path):
            _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            source_path,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.grant_path(
            pid,
            target_path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.write_text(
            pid,
            source_path,
            "DATA_FLOW_SECRET_SENTINEL",
            source_oids=[source.oid],
        )
        source_binding = runtime.store.get_file_label_binding(source_path)
        assert source_binding is not None
    finally:
        runtime.close()

    reopened = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        assert reopened.store.is_recovered_object_payload(source.oid)

        observed = reopened.filesystem.read_text(pid, source_path)

        observed_refs = reopened.data_flow.current_context().source_refs
        assert source.oid not in {ref.oid for ref in observed_refs}
        assert {ref.oid for ref in observed_refs} == {
            f"{reopened.data_flow.FILE_BINDING_SOURCE_REF_PREFIX}"
            f"{source_binding.binding_id}"
        }
        written = reopened.filesystem.write_text(pid, target_path, observed.content)

        assert written.bytes_written > 0
        assert (root / target_path).read_text(encoding="utf-8") == (
            "DATA_FLOW_SECRET_SENTINEL"
        )
    finally:
        reopened.close()


def test_file_backed_source_ref_validates_immutable_binding_hash() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="validate an immutable file source binding")
        source = _secret_source(runtime, pid)
        source_path = "vault/source-generation.txt"
        target_path = "exports/durable-file-source.txt"
        for path in (source_path, target_path):
            _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            source_path,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.grant_path(
            pid,
            target_path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.write_text(
            pid,
            source_path,
            "DATA_FLOW_SECRET_SENTINEL",
            source_oids=[source.oid],
        )
        original = runtime.store.get_file_label_binding(source_path)
        assert original is not None
        observed = runtime.filesystem.read_text(pid, source_path)

        replacement_content = b"replacement"
        (root / source_path).write_bytes(replacement_content)
        replacement = runtime.data_flow.bind_written_file(
            pid="test.host",
            normalized_path=source_path,
            content=replacement_content,
            context=DataFlowContext(),
        )
        assert replacement.binding_id != original.binding_id

        file_ref = next(
            ref
            for ref in runtime.data_flow.current_context().source_refs
            if ref.oid
            == f"{runtime.data_flow.FILE_BINDING_SOURCE_REF_PREFIX}{original.binding_id}"
        )
        assert runtime.data_flow._validate_source_refs((file_ref,)) is None
        assert "source file binding changed" in str(
            runtime.data_flow._validate_source_refs(
                (replace(file_ref, content_sha256="0" * 64),)
            )
        )

        written = runtime.filesystem.write_text(pid, target_path, observed.content)
        assert written.bytes_written > 0
        assert (root / target_path).read_text(encoding="utf-8") == (
            "DATA_FLOW_SECRET_SENTINEL"
        )


def test_conditional_release_rejects_target_generation_change_during_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="target state toctou")
        source = _secret_source(runtime, pid)
        path = "exports/target-toctou.txt"
        resource = runtime.filesystem.resource_for_path(path)
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.CONDITIONAL)
        ordinary = runtime.capability.grant_once(
            pid,
            resource,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        calls = _count_filesystem_boundaries(runtime, monkeypatch)

        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )
        runtime.human.drain_terminal_queue(auto_approve=True)
        release = next(
            cap
            for cap in runtime.capability.capabilities_for(pid)
            if cap.resource.startswith("data_release:")
        )
        effect_ids_before_retry = [
            effect.effect_id for effect in runtime.store.list_external_effects(pid=pid)
        ]
        original_intent = runtime.filesystem._record_mutation_intent

        def mutate_target_generation(*args: Any, **kwargs: Any):
            runtime.data_flow.bind_written_file(
                pid=pid,
                normalized_path=path,
                content=b"raced target state",
                context=DataFlowContext(),
            )
            return original_intent(*args, **kwargs)

        monkeypatch.setattr(
            runtime.filesystem,
            "_record_mutation_intent",
            mutate_target_generation,
        )
        with pytest.raises(CapabilityDenied, match="target state version changed"):
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )

        assert calls == {"state": 0, "write": 0}
        assert not (root / path).exists()
        assert runtime.store.get_capability(ordinary.cap_id).uses_remaining == 1
        assert runtime.store.get_capability(release.cap_id).uses_remaining == 1
        assert [
            effect.effect_id for effect in runtime.store.list_external_effects(pid=pid)
        ] == effect_ids_before_retry
        denied = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert len(denied) == 1
        assert denied[0].sink == resource
        assert "target state version changed" in denied[0].reason
        assert any(
            record.action == "data_flow.egress"
            and record.target == resource
            and record.decision.get("decision_id") == denied[0].decision_id
            and record.decision.get("outcome") == "deny"
            for record in runtime.audit.trace()
        )
        assert any(
            event.type == EventType.DATA_FLOW_DECISION
            and event.payload.get("decision_id") == denied[0].decision_id
            and event.payload.get("outcome") == "deny"
            for event in runtime.events.list(target=f"data_flow_sink:{resource}")
        )


@pytest.mark.parametrize(
    ("identity", "identity_sha256"),
    [
        ("llm:corp-secure", "a" * 64),
        ("human:operator:terminal", None),
        ("jsonrpc:crm:update", "b" * 64),
        ("mcp:corp:lookup", "c" * 64),
        ("filesystem:workspace:export.txt", None),
        ("shell:/usr/bin/example", "d" * 64),
        ("pty:spawn:/usr/bin/example", "e" * 64),
        ("process:pid_child", None),
    ],
)
def test_host_trusted_clearance_applies_to_each_sink_namespace(
    identity: str,
    identity_sha256: str | None,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="trusted sink namespaces")
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=identity,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
                identity_sha256=identity_sha256,
            ),
            actor="test.host",
            require_capability=False,
        )

        decision, release = runtime.data_flow.authorize_egress(
            pid=pid,
            sink=DataSink(identity, identity_sha256),
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
            payload={"size": 1},
            operation="test.egress",
        )

        assert decision.outcome.value == "allow"
        assert release is None


@pytest.mark.parametrize(
    ("labels", "rule", "sink", "reason"),
    [
        (
            DataLabels(sensitivity="secret"),
            SinkTrustRule(
                "filesystem:workspace:limited.txt",
                trust_level="trusted",
                max_sensitivity="restricted",
            ),
            DataSink("filesystem:workspace:limited.txt"),
            "exceeds Sink maximum",
        ),
        (
            DataLabels(sensitivity="confidential", tenant="tenant-b"),
            SinkTrustRule(
                "filesystem:workspace:tenant.txt",
                trust_level="trusted",
                max_sensitivity="secret",
                tenants=("tenant-a",),
            ),
            DataSink("filesystem:workspace:tenant.txt"),
            "outside Sink clearance",
        ),
        (
            DataLabels(sensitivity="confidential", principal="analyst-b"),
            SinkTrustRule(
                "human:operator:terminal",
                trust_level="trusted",
                max_sensitivity="secret",
                principals=("analyst-a",),
            ),
            DataSink("human:operator:terminal"),
            "outside Sink clearance",
        ),
        (
            DataLabels(sensitivity="confidential", tenant="mixed"),
            SinkTrustRule(
                "filesystem:workspace:mixed.txt",
                trust_level="conditional",
                max_sensitivity="secret",
                tenants=("tenant-a", "tenant-b"),
            ),
            DataSink("filesystem:workspace:mixed.txt"),
            "must be reclassified",
        ),
        (
            DataLabels(sensitivity="confidential"),
            SinkTrustRule(
                "llm:identity-bound",
                trust_level="trusted",
                max_sensitivity="secret",
                identity_sha256="a" * 64,
            ),
            DataSink("llm:identity-bound", "b" * 64),
            "identity hash does not match",
        ),
    ],
)
def test_clearance_rejects_over_limit_identity_domain_and_identity_hash(
    labels: DataLabels,
    rule: SinkTrustRule,
    sink: DataSink,
    reason: str,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="reject Sink clearance mismatch")
        runtime.data_flow.register_sink_trust(
            rule,
            actor="test.host",
            require_capability=False,
        )

        with pytest.raises(CapabilityDenied, match=reason):
            runtime.data_flow.authorize_egress(
                pid=pid,
                sink=sink,
                context=DataFlowContext(labels=labels),
                payload={"size": 1},
                operation="test.egress",
            )

        assert runtime.human.pending() == []
        decisions = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert len(decisions) == 1


def test_trusted_human_sink_accepts_secret_output_without_exposing_registry_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="trusted Human output")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        once = runtime.capability.grant_once(
            pid,
            f"human:{human}",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)

        runtime.human.output(
            pid,
            "DATA_FLOW_SECRET_SENTINEL",
            source_oids=[source.oid],
        )

        assert delivered == ["DATA_FLOW_SECRET_SENTINEL"]
        assert runtime.store.get_capability(once.cap_id).uses_remaining == 0
        assert all(
            "sink_trust" not in str(tool)
            for tool in runtime.tools.model_visible_tools(pid)
        )


def test_conditional_human_sink_processes_metadata_release_before_secret_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="conditional Human question")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "Confirm DATA_FLOW_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )

        released = runtime.human.process_next_terminal(
            auto_approve=True,
            auto_answer="confirmed",
        )

        assert released is not None
        assert released.payload["type"] == "data_release_approval"
        assert released.status == HumanRequestStatus.APPROVED
        assert "DATA_FLOW_SECRET_SENTINEL" not in "\n".join(delivered)
        assert [item.request_id for item in runtime.human.pending()] == [request_id]

        answered = runtime.human.process_next_terminal(
            auto_approve=True,
            auto_answer="confirmed",
        )

        assert answered is not None and answered.request_id == request_id
        assert answered.status == HumanRequestStatus.APPROVED
        assert "DATA_FLOW_SECRET_SENTINEL" in delivered[-1]
        assert runtime.human.pending() == []


def test_interactive_cli_presents_conditional_question_only_after_exact_release(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="conditional interactive Human question")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "CLI DATA_FLOW_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )
        state: dict[str, str] = {}

        _show_pending_interactive_human_request(runtime, human, state)

        assert "DATA_FLOW_SECRET_SENTINEL" not in capsys.readouterr().err
        assert delivered == []
        release = next(
            item
            for item in runtime.human.pending(human=human)
            if item.payload.get("type") == "data_release_approval"
        )

        _show_pending_interactive_human_request(runtime, human, state)

        assert delivered
        assert "DATA_FLOW_SECRET_SENTINEL" not in delivered[-1]
        runtime.human.approve(
            release.request_id,
            {"approved": True, "source": "test"},
        )
        approved_only = runtime.human.public_request_view(
            runtime.human.get(request_id)
        )
        assert "DATA_FLOW_SECRET_SENTINEL" not in str(approved_only)

        _show_pending_interactive_human_request(runtime, human, state)

        assert state["shown_request_id"] == request_id
        assert "DATA_FLOW_SECRET_SENTINEL" in delivered[-1]
        completed_release = runtime.human.public_request_view(
            runtime.human.get(request_id)
        )
        assert "DATA_FLOW_SECRET_SENTINEL" in str(completed_release)


def test_conditional_external_approval_returns_post_release_preview_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="conditional external approval preview")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)
        resource = "filesystem:workspace:reports/conditional-preview.txt"
        request_id = runtime.human.query_authority_request(
            pid,
            human,
            {
                "type": "external_operation_approval",
                "question": "Approve DATA_FLOW_SECRET_SENTINEL read?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                    "constraints": {},
                },
                "context": {
                    "adapter": "filesystem",
                    "authority_operation": "filesystem.read",
                    "primitive": "runtime.filesystem.read_text",
                    "operation": "read_text",
                    "pid": pid,
                    "resource": resource,
                    "right": CapabilityRight.READ.value,
                    "target_state_version": None,
                },
            },
            authority_origin="external_operation",
            source_oids=[source.oid],
        )
        state: dict[str, Any] = {}

        _show_pending_interactive_human_request(runtime, human, state)
        release = next(
            request
            for request in runtime.human.pending(human=human)
            if request.payload.get("type") == "data_release_approval"
        )
        _show_pending_interactive_human_request(runtime, human, state)
        runtime.human.approve(
            release.request_id,
            {"approved": True, "source": "test"},
        )

        _show_pending_interactive_human_request(runtime, human, state)

        current = runtime.human.get(request_id)
        view = runtime.human.public_request_view(current)
        assert state["shown_request_id"] == request_id
        assert state["shown_request_revision"] == current.revision
        assert state["shown_preview_sha256"] == view["preview_sha256"]
        assert "Canonical operation approval preview" in delivered[-1]
        assert "DATA_FLOW_SECRET_SENTINEL" not in delivered[-1]

        assert (
            _handle_interactive_line(
                runtime,
                "/approve",
                state,
                human,
                channel,
                [],
            )
            is None
        )
        settled = runtime.human.get(request_id)
        assert settled.status is HumanRequestStatus.APPROVED


def test_interactive_cli_retains_release_outside_bounded_pending_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(DEFAULT_CONFIG.tools, human_request_list_limit=2),
    )
    runtime = Runtime.open(
        "local",
        config=config,
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="present the exact crowded release request")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "CROWDED_RELEASE_SECRET_SENTINEL",
            },
            blocking=False,
            source_oids=[source.oid],
        )
        runtime.human.query(
            pid,
            human,
            {"type": "approval", "summary": "ordinary backlog filler"},
            blocking=False,
        )
        state = {"pid": pid, "shown_request_id": ""}
        posted: list[dict[str, Any]] = []

        _show_pending_interactive_human_request(runtime, human, state)

        release_id = state["pending_release_request_id"]
        assert release_id not in {
            item.request_id for item in runtime.human.pending(human=human)
        }
        assert delivered == []

        _show_pending_interactive_human_request(runtime, human, state)

        assert state["shown_request_id"] == release_id
        assert delivered
        assert "CROWDED_RELEASE_SECRET_SENTINEL" not in delivered[-1]
        _handle_interactive_line(
            runtime,
            "yes",
            state,
            human,
            channel,
            posted,
        )
        assert runtime.human.get(release_id).status == HumanRequestStatus.APPROVED

        _show_pending_interactive_human_request(runtime, human, state)

        assert state["shown_request_id"] == request_id
        assert "CROWDED_RELEASE_SECRET_SENTINEL" in delivered[-1]
    finally:
        runtime.close()


def test_interactive_cli_does_not_apply_queued_input_to_an_unshown_release() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="bind interactive input to the shown request")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        runtime.human.provider.output_sink = delivered.append
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "QUEUED_INPUT_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )
        state = {"pid": pid, "shown_request_id": ""}
        posted: list[dict[str, Any]] = []

        _show_pending_interactive_human_request(runtime, human, state)

        release = next(
            item
            for item in runtime.human.pending(human=human)
            if item.payload.get("type") == "data_release_approval"
        )
        assert state["shown_request_id"] == ""
        assert delivered == []

        _handle_interactive_line(
            runtime,
            "yes",
            state,
            human,
            channel,
            posted,
        )

        assert runtime.human.get(release.request_id).status == HumanRequestStatus.PENDING
        assert posted and posted[-1]["body"] == "yes"

        _show_pending_interactive_human_request(runtime, human, state)
        assert state["shown_request_id"] == release.request_id
        _handle_interactive_line(
            runtime,
            "yes",
            state,
            human,
            channel,
            posted,
        )
        assert runtime.human.get(release.request_id).status == HumanRequestStatus.APPROVED

        _show_pending_interactive_human_request(runtime, human, state)
        assert state["shown_request_id"] == request_id
        assert "QUEUED_INPUT_SECRET_SENTINEL" in delivered[-1]


def test_human_query_event_contains_only_sanitized_request_evidence() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="sanitize Human query event")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )

        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "EVENT DATA_FLOW_SECRET_SENTINEL",
                "opaque": "EVENT_OPAQUE_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )

        event = next(
            item
            for item in runtime.events.list()
            if item.type == EventType.HUMAN_QUERY
            and item.payload.get("request_id") == request_id
        )
        assert event.payload["request_type"] == "question"
        assert event.payload["request"]["redacted"] is True
        assert "DATA_FLOW_SECRET_SENTINEL" not in str(event.payload)
        assert "EVENT_OPAQUE_SECRET_SENTINEL" not in str(event.payload)


def test_rejected_conditional_human_release_terminates_secret_question_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="reject conditional Human question")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        delivered: list[str] = []
        monkeypatch.setattr(runtime.human.provider, "output_sink", delivered.append)
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "Reject DATA_FLOW_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )

        processed = runtime.human.drain_terminal_queue(
            auto_approve=False,
            auto_answer="must-not-be-delivered",
        )

        assert len(processed) == 1
        assert processed[0].payload["type"] == "data_release_approval"
        assert processed[0].status == HumanRequestStatus.REJECTED
        original = runtime.human.get(request_id)
        assert original.status == HumanRequestStatus.CANCELLED
        assert original.decision is not None
        assert original.decision["data_release_outcome"] == "rejected"
        assert original.decision["sensitive_payload_delivered"] is False
        assert runtime.human.pending() == []
        assert runtime.human.drain_terminal_queue(auto_approve=False) == []
        assert sum(
            item.payload.get("type") == "data_release_approval"
            for item in runtime.human.list(pid)
        ) == 1
        assert "DATA_FLOW_SECRET_SENTINEL" not in "\n".join(delivered)


def test_pending_conditional_human_release_survives_reopen_without_duplication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    database = tmp_path / "runtime.sqlite"
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="reopen conditional Human release")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "Reopen DATA_FLOW_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )

        def not_started(_text: str) -> None:
            raise ProviderEffectNotStarted("release prompt was not delivered")

        runtime.human.provider.output_sink = not_started
        with pytest.raises(ProviderEffectNotStarted):
            runtime.human.process_next_terminal(
                auto_approve=False,
                auto_answer="must-not-be-delivered",
            )
        assert len(runtime.human.pending()) == 2
    finally:
        runtime.close()

    reopened = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        delivered: list[str] = []
        reopened.human.provider.output_sink = delivered.append
        processed = reopened.human.drain_terminal_queue(
            auto_approve=False,
            auto_answer="must-not-be-delivered",
        )

        assert len(processed) == 1
        assert processed[0].payload["type"] == "data_release_approval"
        assert processed[0].status == HumanRequestStatus.REJECTED
        assert reopened.human.get(request_id).status == HumanRequestStatus.CANCELLED
        assert reopened.human.pending() == []
        assert sum(
            item.payload.get("type") == "data_release_approval"
            for item in reopened.human.list(pid)
        ) == 1
        assert "DATA_FLOW_SECRET_SENTINEL" not in "\n".join(delivered)
    finally:
        reopened.close()


def test_ambiguous_conditional_human_release_cancels_linked_secret_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal="ambiguous conditional Human release")
        source = _secret_source(runtime, pid)
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f"human:{human}:{channel}",
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                "type": "question",
                "question": "Ambiguous DATA_FLOW_SECRET_SENTINEL",
            },
            source_oids=[source.oid],
        )
        provider_calls = 0

        def fail_release(_text: str) -> None:
            nonlocal provider_calls
            provider_calls += 1
            raise RuntimeError("ambiguous release prompt delivery")

        monkeypatch.setattr(runtime.human.provider, "output_sink", fail_release)
        with pytest.raises(RuntimeError, match="ambiguous release prompt delivery"):
            runtime.human.process_next_terminal(
                auto_approve=False,
                auto_answer="must-not-be-delivered",
            )

        assert provider_calls == 1
        original = runtime.human.get(request_id)
        assert original.status == HumanRequestStatus.CANCELLED
        assert original.decision is not None
        assert original.decision["data_release_outcome"] == "provider_outcome_unknown"
        assert original.decision["sensitive_payload_delivered"] is False
        assert runtime.human.pending() == []
        assert sum(
            item.payload.get("type") == "data_release_approval"
            for item in runtime.human.list(pid)
        ) == 1


def test_human_answer_resume_rehydrates_secret_labels_after_reopen(tmp_path: Path) -> None:
    root = tmp_path / "resume-workspace"
    root.mkdir()
    database = tmp_path / "resume-runtime.sqlite"
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    args = {
        "question": "Which window?",
        "context": {"scope": "deployment"},
        "human": runtime.config.runtime.default_human,
    }
    try:
        pid = runtime.process.spawn(image="review-agent:v0", goal="resume labeled answer")
        runtime.capability.grant(
            pid,
            f"human:{runtime.config.runtime.default_human}",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        source = _secret_source(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f"human:{runtime.config.runtime.default_human}:"
                    f"{runtime.config.runtime.terminal_channel}"
                ),
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        source_context = runtime.data_flow.context_from_source_oids(
            pid,
            [source.oid],
            include_current=False,
        )
        with pytest.raises(HumanResponseRequired) as raised:
            runtime.tools.call(
                pid,
                "ask_human",
                args,
                context_metadata={"data_flow_context": source_context},
            )
        request_id = raised.value.request_id
        runtime.human.drain_terminal_queue(auto_answer="Sunday")
    finally:
        runtime.close()

    reopened = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        result = reopened.tools.call(
            pid,
            "ask_human",
            args,
            context_metadata={
                "data_flow_context": DataFlowContext(),
                "human_resume_request_id": request_id,
            },
        )

        assert result.ok, result.error
        assert result.result_handle is not None
        persisted = reopened.store.get_object(result.result_handle.oid)
        assert persisted is not None
        assert persisted.metadata.sensitivity == "secret"
        assert persisted.metadata.trust_level == "untrusted"
        assert source.oid in persisted.provenance.parent_oids
    finally:
        reopened.close()


def test_host_sink_registry_api_requires_admin_and_is_not_model_visible() -> None:
    with workspace_runtime() as (runtime, _root):
        actor = runtime.process.spawn(goal="manage Host Sink registry")
        rule = SinkTrustRule(
            pattern="filesystem:workspace:approved.txt",
            trust_level=SinkTrustLevel.TRUSTED,
            max_sensitivity="restricted",
        )

        with pytest.raises(CapabilityDenied):
            runtime.register_sink_trust(rule, actor=actor)

        runtime.capability.grant(
            actor,
            runtime.config.data_flow.registry_resource,
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        registered = runtime.register_sink_trust(rule, actor=actor)

        assert runtime.inspect_sink_trust(rule.pattern) == registered
        assert runtime.list_sink_trust() == (registered,)
        assert all(
            "sink_trust" not in str(tool)
            for tool in runtime.tools.model_visible_tools(actor)
        )

        removed = runtime.unregister_sink_trust(rule.pattern, actor=actor)
        assert removed.pattern == rule.pattern
        assert runtime.inspect_sink_trust(rule.pattern) is None


@pytest.mark.parametrize("operation", ["register", "unregister"])
def test_sink_registry_reauthorizes_unlimited_admin_before_mutation(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        actor = runtime.process.spawn(goal=f"race Sink registry {operation}")
        rule = SinkTrustRule(
            pattern="filesystem:workspace:revocation-race.txt",
            trust_level=SinkTrustLevel.TRUSTED,
            max_sensitivity="restricted",
        )
        if operation == "unregister":
            runtime.data_flow.register_sink_trust(
                rule,
                actor="test.host",
                require_capability=False,
            )
        authority = runtime.capability.grant(
            actor,
            runtime.config.data_flow.registry_resource,
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        original_require = runtime.capability.require

        def revoke_after_outer_authorization(*args: Any, **kwargs: Any):
            decision = original_require(*args, **kwargs)
            runtime.capability.revoke(
                authority.cap_id,
                revoked_by="test.host",
                reason="registry revocation race regression",
                require_authority=False,
            )
            return decision

        monkeypatch.setattr(
            runtime.capability,
            "require",
            revoke_after_outer_authorization,
        )

        with pytest.raises(CapabilityDenied, match="authority changed"):
            if operation == "register":
                runtime.data_flow.register_sink_trust(rule, actor=actor)
            else:
                runtime.data_flow.unregister_sink_trust(rule.pattern, actor=actor)

        persisted = runtime.data_flow.inspect_sink_trust(rule.pattern)
        assert (persisted is None) is (operation == "register")


def test_file_delete_tombstones_target_binding() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="delete one labeled file")
        path = "labeled-file.txt"
        content = b"DATA_FLOW_SECRET_SENTINEL"
        target = root / path
        target.write_bytes(content)
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=path,
            content=content,
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        original = runtime.store.get_file_label_binding(path)
        assert original is not None
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )

        deleted = runtime.filesystem.delete_file(pid, path)

        assert deleted.deleted is True
        assert not target.exists()
        assert runtime.store.get_file_label_binding(path) is None
        history = runtime.store.list_file_label_bindings(
            normalized_path=path,
            include_history=True,
            include_tombstones=True,
        )
        assert history[0].tombstoned is True
        assert history[0].generation == original.generation + 1


@pytest.mark.parametrize("target_kind", ["file", "recursive-directory"])
def test_delete_serializes_pre_unlink_against_secret_label_publication(
    target_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        delete_pid = runtime.process.spawn(goal=f"delete raced {target_kind}")
        writer_pid = runtime.process.spawn(goal=f"publish secret into raced {target_kind}")
        delete_path = "pre-unlink-race"
        write_path = (
            delete_path
            if target_kind == "file"
            else f"{delete_path}/child.txt"
        )
        target = root / write_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("normal", encoding="utf-8")
        source = _secret_source(runtime, writer_pid)
        _register_file_sink(
            runtime,
            write_path,
            trust_level=SinkTrustLevel.CONDITIONAL,
        )
        runtime.filesystem.grant_path(
            writer_pid,
            write_path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        if target_kind == "file":
            runtime.filesystem.grant_path(
                delete_pid,
                delete_path,
                [CapabilityRight.DELETE],
                issued_by="test.host",
            )
            original_delete = runtime.filesystem.provider.delete_file

            def delete_target() -> None:
                runtime.filesystem.delete_file(delete_pid, delete_path)

            provider_method = "delete_file"
        else:
            runtime.filesystem.grant_directory(
                delete_pid,
                delete_path,
                [CapabilityRight.DELETE],
                issued_by="test.host",
            )
            original_delete = runtime.filesystem.provider.delete_directory_protected

            def delete_target() -> None:
                runtime.filesystem.delete_directory(
                    delete_pid,
                    delete_path,
                    recursive=True,
                )

            provider_method = "delete_directory_protected"

        with pytest.raises(HumanApprovalRequired):
            runtime.filesystem.write_text(
                writer_pid,
                write_path,
                "RACED_SECRET",
                source_oids=[source.oid],
            )
        runtime.human.drain_terminal_queue(auto_approve=True)

        before_unlink = threading.Event()
        release_unlink = threading.Event()
        writer_started = threading.Event()
        writer_provider_entered = threading.Event()
        original_write = runtime.filesystem.provider.write_text
        delete_errors: list[BaseException] = []
        writer_errors: list[BaseException] = []

        def block_before_unlink(*args: Any, **kwargs: Any) -> Any:
            before_unlink.set()
            if not release_unlink.wait(timeout=10):
                raise TimeoutError("pre-unlink race was not released")
            return original_delete(*args, **kwargs)

        def track_writer_dispatch(*args: Any, **kwargs: Any) -> Any:
            writer_provider_entered.set()
            return original_write(*args, **kwargs)

        def run_delete() -> None:
            try:
                delete_target()
            except BaseException as exc:
                delete_errors.append(exc)

        def run_writer() -> None:
            writer_started.set()
            try:
                runtime.filesystem.write_text(
                    writer_pid,
                    write_path,
                    "RACED_SECRET",
                    source_oids=[source.oid],
                )
            except BaseException as exc:
                writer_errors.append(exc)

        monkeypatch.setattr(
            runtime.filesystem.provider,
            provider_method,
            block_before_unlink,
        )
        monkeypatch.setattr(
            runtime.filesystem.provider,
            "write_text",
            track_writer_dispatch,
        )
        delete_thread = threading.Thread(target=run_delete, daemon=True)
        writer_thread = threading.Thread(target=run_writer, daemon=True)
        delete_thread.start()
        assert before_unlink.wait(timeout=10)
        writer_thread.start()
        assert writer_started.wait(timeout=10)
        try:
            acquired = runtime.filesystem._file_label_io_lock.acquire(
                blocking=False
            )
            if acquired:
                runtime.filesystem._file_label_io_lock.release()
            assert acquired is False
            assert not writer_provider_entered.wait(timeout=0.1)
        finally:
            release_unlink.set()
            delete_thread.join(timeout=10)
            writer_thread.join(timeout=10)

        assert not delete_thread.is_alive()
        assert not writer_thread.is_alive()
        assert delete_errors == []
        assert writer_errors == []
        assert writer_provider_entered.is_set()
        assert target.read_text(encoding="utf-8") == "RACED_SECRET"
        binding = runtime.store.get_file_label_binding(write_path)
        assert binding is not None
        assert binding.labels.sensitivity.value == "secret"


def test_file_delete_preserves_post_dispatch_replacement_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        delete_pid = runtime.process.spawn(goal="delete one authorized labeled file")
        writer_pid = runtime.process.spawn(goal="replace the deleted labeled file")
        source = _secret_source(runtime, writer_pid)
        path = "replacement-file.txt"
        target = root / path
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            writer_pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.write_text(
            writer_pid,
            path,
            "ORIGINAL_SECRET",
            source_oids=[source.oid],
        )
        original = runtime.store.get_file_label_binding(path)
        assert original is not None
        runtime.filesystem.grant_path(
            delete_pid,
            path,
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )

        provider_deleted = threading.Event()
        release_delete = threading.Event()
        writer_started = threading.Event()
        writer_complete = threading.Event()
        original_delete = runtime.filesystem.provider.delete_file
        delete_error: list[BaseException] = []
        writer_error: list[BaseException] = []

        def blocking_delete(*args: Any, **kwargs: Any) -> None:
            original_delete(*args, **kwargs)
            provider_deleted.set()
            if not release_delete.wait(timeout=10):
                raise TimeoutError("file delete settlement was not released")

        def run_delete() -> None:
            try:
                runtime.filesystem.delete_file(delete_pid, path)
            except BaseException as exc:
                delete_error.append(exc)

        def run_writer() -> None:
            writer_started.set()
            try:
                runtime.filesystem.write_text(
                    writer_pid,
                    path,
                    "REPLACEMENT_SECRET",
                    source_oids=[source.oid],
                )
            except BaseException as exc:
                writer_error.append(exc)
            finally:
                writer_complete.set()

        monkeypatch.setattr(
            runtime.filesystem.provider,
            "delete_file",
            blocking_delete,
        )
        delete_thread = threading.Thread(target=run_delete, daemon=True)
        writer_thread = threading.Thread(target=run_writer, daemon=True)
        delete_thread.start()
        if not provider_deleted.wait(timeout=10):
            raise TimeoutError("file delete did not reach the provider boundary")
        writer_thread.start()
        assert writer_started.wait(timeout=10)
        assert not writer_complete.wait(timeout=0.1)
        release_delete.set()
        delete_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert not delete_thread.is_alive()
        assert not writer_thread.is_alive()
        assert delete_error == []
        assert len(writer_error) == 1
        assert isinstance(writer_error[0], CapabilityDenied)
        assert "target state version changed" in str(writer_error[0])
        runtime.filesystem.write_text(
            writer_pid,
            path,
            "REPLACEMENT_SECRET",
            source_oids=[source.oid],
        )
        replacement = runtime.store.get_file_label_binding(path)
        assert replacement is not None
        assert replacement.binding_id != original.binding_id
        assert target.read_text(encoding="utf-8") == "REPLACEMENT_SECRET"
        active = runtime.store.get_file_label_binding(path)
        assert active is not None
        assert active.binding_id == replacement.binding_id
        assert active.labels.sensitivity.value == "secret"


def test_recursive_directory_delete_includes_descendant_labels() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="delete labeled directory tree")
        directory = root / "labeled-tree"
        directory.mkdir()
        child = directory / "secret-child.txt"
        child.write_text("DATA_FLOW_SECRET_SENTINEL", encoding="utf-8")
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path="labeled-tree/secret-child.txt",
            content=b"DATA_FLOW_SECRET_SENTINEL",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        runtime.filesystem.grant_directory(
            pid,
            "labeled-tree",
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )

        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.filesystem.delete_directory(
                pid,
                "labeled-tree",
                recursive=True,
            )
        assert child.exists()

        _register_file_sink(
            runtime,
            "labeled-tree",
            trust_level=SinkTrustLevel.TRUSTED,
        )
        deleted = runtime.filesystem.delete_directory(
            pid,
            "labeled-tree",
            recursive=True,
        )

        assert deleted.deleted is True
        assert not directory.exists()
        assert runtime.store.get_file_label_binding(
            "labeled-tree/secret-child.txt"
        ) is None
        history = runtime.store.list_file_label_bindings(
            normalized_path="labeled-tree/secret-child.txt",
            include_history=True,
            include_tombstones=True,
        )
        assert history[0].tombstoned


def test_nonrecursive_directory_delete_tombstones_target_binding() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="delete one labeled directory")
        directory_path = "labeled-empty-directory"
        directory = root / directory_path
        directory.mkdir()
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=directory_path,
            content=b"<agent-libos-directory>",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        original = runtime.store.get_file_label_binding(directory_path)
        assert original is not None
        _register_file_sink(
            runtime,
            directory_path,
            trust_level=SinkTrustLevel.TRUSTED,
        )
        runtime.filesystem.grant_directory(
            pid,
            directory_path,
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )

        deleted = runtime.filesystem.delete_directory(
            pid,
            directory_path,
            recursive=False,
        )

        assert deleted.deleted is True
        assert not directory.exists()
        assert runtime.store.get_file_label_binding(directory_path) is None
        history = runtime.store.list_file_label_bindings(
            normalized_path=directory_path,
            include_history=True,
            include_tombstones=True,
        )
        assert history[0].tombstoned is True
        assert history[0].generation == original.generation + 1


def test_nonrecursive_directory_delete_preserves_post_dispatch_replacement_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        delete_pid = runtime.process.spawn(goal="delete one authorized labeled directory")
        writer_pid = runtime.process.spawn(goal="replace its directory binding")
        directory_path = "replacement-empty-directory"
        directory = root / directory_path
        directory.mkdir()
        runtime.data_flow.bind_written_file(
            pid=delete_pid,
            normalized_path=directory_path,
            content=b"original-directory",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        original = runtime.store.get_file_label_binding(directory_path)
        assert original is not None
        _register_file_sink(
            runtime,
            directory_path,
            trust_level=SinkTrustLevel.TRUSTED,
        )
        runtime.filesystem.grant_directory(
            delete_pid,
            directory_path,
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )
        original_delete = runtime.filesystem.provider.delete_directory

        def delete_then_replace(*args: Any, **kwargs: Any) -> None:
            original_delete(*args, **kwargs)
            directory.mkdir()
            runtime.data_flow.bind_written_file(
                pid=writer_pid,
                normalized_path=directory_path,
                content=b"replacement-directory",
                context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
            )

        monkeypatch.setattr(
            runtime.filesystem.provider,
            "delete_directory",
            delete_then_replace,
        )

        deleted = runtime.filesystem.delete_directory(
            delete_pid,
            directory_path,
            recursive=False,
        )

        assert deleted.deleted is True
        replacement = runtime.store.get_file_label_binding(directory_path)
        assert replacement is not None
        assert replacement.binding_id != original.binding_id
        assert replacement.labels.sensitivity.value == "secret"


def test_recursive_directory_delete_uses_atomic_label_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="race recursive directory labels")
        directory = root / "raced-tree"
        directory.mkdir()
        child = directory / "child.txt"
        child.write_text("normal", encoding="utf-8")
        authority = runtime.capability.grant_once(
            pid,
            runtime.filesystem.directory_resource_for("raced-tree"),
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )
        original_list_tree = runtime.store.list_file_label_bindings_for_tree
        tree_queries = 0

        def inject_binding_between_legacy_queries(normalized_path: str):
            nonlocal tree_queries
            bindings = original_list_tree(normalized_path)
            tree_queries += 1
            if tree_queries == 1:
                runtime.data_flow.bind_written_file(
                    pid=pid,
                    normalized_path="raced-tree/child.txt",
                    content=b"normal",
                    context=DataFlowContext(
                        labels=DataLabels(sensitivity="secret")
                    ),
                )
            return bindings

        provider_calls = {"state": 0, "delete": 0}
        original_state = runtime.filesystem.provider.state
        original_delete = runtime.filesystem.provider.delete_directory_protected

        def track_state(*args: Any, **kwargs: Any):
            provider_calls["state"] += 1
            return original_state(*args, **kwargs)

        def track_delete(*args: Any, **kwargs: Any):
            provider_calls["delete"] += 1
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(
            runtime.store,
            "list_file_label_bindings_for_tree",
            inject_binding_between_legacy_queries,
        )
        monkeypatch.setattr(runtime.filesystem.provider, "state", track_state)
        monkeypatch.setattr(
            runtime.filesystem.provider,
            "delete_directory_protected",
            track_delete,
        )

        with pytest.raises(CapabilityDenied, match="target state version changed"):
            runtime.filesystem.delete_directory(
                pid,
                "raced-tree",
                recursive=True,
            )

        assert tree_queries >= 2
        assert provider_calls == {"state": 0, "delete": 0}
        assert child.exists()
        assert runtime.store.get_capability(authority.cap_id).uses_remaining == 1


def test_recursive_directory_delete_preserves_post_dispatch_replacement_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        delete_pid = runtime.process.spawn(goal="delete an authorized labeled tree")
        writer_pid = runtime.process.spawn(goal="recreate a labeled child")
        source = _secret_source(runtime, writer_pid)
        directory_path = "replacement-tree"
        child_path = f"{directory_path}/child.txt"
        directory = root / directory_path
        child = root / child_path
        directory.mkdir()
        _register_file_sink(runtime, child_path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_path(
            writer_pid,
            child_path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        runtime.filesystem.write_text(
            writer_pid,
            child_path,
            "ORIGINAL_SECRET",
            source_oids=[source.oid],
        )
        _register_file_sink(runtime, directory_path, trust_level=SinkTrustLevel.TRUSTED)
        runtime.filesystem.grant_directory(
            delete_pid,
            directory_path,
            [CapabilityRight.DELETE],
            issued_by="test.host",
        )

        provider_deleted = threading.Event()
        release_delete = threading.Event()
        writer_started = threading.Event()
        writer_complete = threading.Event()
        original_delete = runtime.filesystem.provider.delete_directory_protected
        delete_error: list[BaseException] = []
        writer_error: list[BaseException] = []

        def blocking_delete(*args: Any, **kwargs: Any) -> None:
            original_delete(*args, **kwargs)
            provider_deleted.set()
            if not release_delete.wait(timeout=10):
                raise TimeoutError("directory delete settlement was not released")

        def run_delete() -> None:
            try:
                runtime.filesystem.delete_directory(
                    delete_pid,
                    directory_path,
                    recursive=True,
                )
            except BaseException as exc:
                delete_error.append(exc)

        def run_writer() -> None:
            writer_started.set()
            try:
                runtime.filesystem.write_text(
                    writer_pid,
                    child_path,
                    "REPLACEMENT_SECRET",
                    source_oids=[source.oid],
                )
            except BaseException as exc:
                writer_error.append(exc)
            finally:
                writer_complete.set()

        monkeypatch.setattr(
            runtime.filesystem.provider,
            "delete_directory_protected",
            blocking_delete,
        )
        delete_thread = threading.Thread(target=run_delete, daemon=True)
        writer_thread = threading.Thread(target=run_writer, daemon=True)
        delete_thread.start()
        if not provider_deleted.wait(timeout=10):
            raise TimeoutError("recursive delete did not reach the provider boundary")
        writer_thread.start()
        assert writer_started.wait(timeout=10)
        assert not writer_complete.wait(timeout=0.1)
        release_delete.set()
        delete_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert not delete_thread.is_alive()
        assert not writer_thread.is_alive()
        assert delete_error == []
        assert len(writer_error) == 1
        assert isinstance(writer_error[0], CapabilityDenied)
        assert "target state version changed" in str(writer_error[0])
        runtime.filesystem.write_text(
            writer_pid,
            child_path,
            "REPLACEMENT_SECRET",
            source_oids=[source.oid],
        )
        replacement = runtime.store.get_file_label_binding(child_path)
        assert replacement is not None
        assert child.read_text(encoding="utf-8") == "REPLACEMENT_SECRET"
        active = runtime.store.get_file_label_binding(child_path)
        assert active is not None
        assert active.binding_id == replacement.binding_id
        assert active.labels.sensitivity.value == "secret"


def test_file_reads_fail_closed_when_binding_changes_after_provider_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for operation in ("read_text", "read_bytes"):
        with workspace_runtime() as (runtime, root):
            pid = runtime.process.spawn(goal=f"bind {operation} to one file generation")
            path = f"{operation}.txt"
            secret = b"READ_GENERATION_SECRET"
            (root / path).write_bytes(secret)
            runtime.data_flow.bind_written_file(
                pid=pid,
                normalized_path=path,
                content=secret,
                context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
            )
            runtime.filesystem.grant_path(
                pid,
                path,
                [CapabilityRight.READ],
                issued_by="test.host",
            )
            original_read = runtime.filesystem.provider.read_bytes

            def replace_after_read(*args: Any, **kwargs: Any) -> bytes:
                returned = original_read(*args, **kwargs)
                (root / path).write_bytes(b"replacement")
                runtime.data_flow.tombstone_file(pid="test.host", normalized_path=path)
                runtime.data_flow.bind_written_file(
                    pid="test.host",
                    normalized_path=path,
                    content=b"replacement",
                    context=DataFlowContext(),
                )
                return returned

            monkeypatch.setattr(
                runtime.filesystem.provider,
                "read_bytes",
                replace_after_read,
            )

            with pytest.raises(CapabilityDenied, match="label binding changed during read"):
                getattr(runtime.filesystem, operation)(pid, path)


def test_file_reads_wait_for_label_publication_after_provider_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for operation in ("read_text", "read_bytes"):
        with workspace_runtime() as (runtime, root):
            writer_pid = runtime.process.spawn(goal=f"write while {operation} waits")
            reader_pid = runtime.process.spawn(goal=f"read after {operation} label publication")
            source = _secret_source(runtime, writer_pid)
            path = f"{operation}-publication.txt"
            public = b"PUBLIC"
            secret_text = "SECRET_AFTER_PROVIDER_WRITE"
            (root / path).write_bytes(public)
            runtime.data_flow.bind_written_file(
                pid="test.host",
                normalized_path=path,
                content=public,
                context=DataFlowContext(),
            )
            _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
            runtime.filesystem.grant_path(
                writer_pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            runtime.filesystem.grant_path(
                reader_pid,
                path,
                [CapabilityRight.READ],
                issued_by="test.host",
            )

            provider_written = threading.Event()
            release_writer = threading.Event()
            reader_started = threading.Event()
            reader_done = threading.Event()
            writer_error: list[BaseException] = []
            reader_error: list[BaseException] = []
            read_content: list[str | bytes] = []
            read_context: list[DataFlowContext] = []
            original_write = runtime.filesystem.provider.write_text

            def write_then_pause(*args: Any, **kwargs: Any) -> None:
                original_write(*args, **kwargs)
                provider_written.set()
                if not release_writer.wait(timeout=10):
                    raise TimeoutError("writer was not released for label settlement")

            def run_writer() -> None:
                try:
                    runtime.filesystem.write_text(
                        writer_pid,
                        path,
                        secret_text,
                        source_oids=[source.oid],
                    )
                except BaseException as exc:
                    writer_error.append(exc)

            def run_reader() -> None:
                reader_started.set()
                try:
                    result = getattr(runtime.filesystem, operation)(reader_pid, path)
                    read_content.append(result.content)
                    read_context.append(runtime.data_flow.current_context())
                except BaseException as exc:
                    reader_error.append(exc)
                finally:
                    reader_done.set()

            monkeypatch.setattr(
                runtime.filesystem.provider,
                "write_text",
                write_then_pause,
            )
            writer_thread = threading.Thread(target=run_writer, daemon=True)
            writer_thread.start()
            if not provider_written.wait(timeout=10):
                raise TimeoutError("file write did not reach the provider boundary")
            reader_thread = threading.Thread(target=run_reader, daemon=True)
            reader_thread.start()
            if not reader_started.wait(timeout=10):
                raise TimeoutError("file read did not start")
            completed_before_label_publication = reader_done.wait(timeout=1)
            release_writer.set()
            writer_thread.join(timeout=10)
            reader_thread.join(timeout=10)

            assert completed_before_label_publication is False
            assert not writer_thread.is_alive()
            assert not reader_thread.is_alive()
            assert writer_error == []
            assert reader_error == []
            assert read_content == [
                secret_text if operation == "read_text" else secret_text.encode("utf-8")
            ]
            assert read_context[0].labels.sensitivity.value == "secret"
            final_binding = runtime.store.get_file_label_binding(path)
            assert final_binding is not None
            assert final_binding.labels.sensitivity.value == "secret"


@pytest.mark.parametrize(
    ("operation", "path"),
    (
        ("write_text", "secret-derived-filename.txt"),
        ("write_directory", "secret-derived-directory"),
    ),
)
def test_directory_listing_waits_for_child_label_publication(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    path: str,
) -> None:
    with workspace_runtime() as (runtime, _root):
        writer_pid = runtime.process.spawn(goal=f"publish labels for {operation}")
        reader_pid = runtime.process.spawn(goal="list after child label publication")
        source = _secret_source(runtime, writer_pid)
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        if operation == "write_text":
            runtime.filesystem.grant_path(
                writer_pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            provider_operation = "write_text"
        else:
            runtime.filesystem.grant_directory(
                writer_pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            provider_operation = "make_directory"
        runtime.filesystem.grant_directory(
            reader_pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test.host",
        )

        provider_mutated = threading.Event()
        release_writer = threading.Event()
        reader_started = threading.Event()
        reader_done = threading.Event()
        writer_error: list[BaseException] = []
        reader_error: list[BaseException] = []
        listed_names: list[list[str]] = []
        read_context: list[DataFlowContext] = []
        original_mutation = getattr(runtime.filesystem.provider, provider_operation)

        def mutate_then_pause(*args: Any, **kwargs: Any) -> Any:
            result = original_mutation(*args, **kwargs)
            provider_mutated.set()
            if not release_writer.wait(timeout=10):
                raise TimeoutError("writer was not released for child label settlement")
            return result

        def run_writer() -> None:
            try:
                if operation == "write_text":
                    runtime.filesystem.write_text(
                        writer_pid,
                        path,
                        "SECRET_FILENAME_SENTINEL",
                        source_oids=[source.oid],
                    )
                else:
                    runtime.filesystem.write_directory(
                        writer_pid,
                        path,
                        source_oids=[source.oid],
                    )
            except BaseException as exc:
                writer_error.append(exc)

        def run_reader() -> None:
            reader_started.set()
            try:
                with runtime.data_flow.activate(DataFlowContext()):
                    result = runtime.filesystem.read_directory(reader_pid, ".")
                    listed_names.append([entry.name for entry in result.entries])
                    read_context.append(runtime.data_flow.current_context())
            except BaseException as exc:
                reader_error.append(exc)
            finally:
                reader_done.set()

        monkeypatch.setattr(
            runtime.filesystem.provider,
            provider_operation,
            mutate_then_pause,
        )
        writer_thread = threading.Thread(target=run_writer, daemon=True)
        writer_thread.start()
        if not provider_mutated.wait(timeout=10):
            raise TimeoutError("filesystem mutation did not reach the provider boundary")
        reader_thread = threading.Thread(target=run_reader, daemon=True)
        reader_thread.start()
        if not reader_started.wait(timeout=10):
            raise TimeoutError("directory listing did not start")
        completed_before_label_publication = reader_done.wait(timeout=1)
        release_writer.set()
        writer_thread.join(timeout=10)
        reader_thread.join(timeout=10)

        assert completed_before_label_publication is False
        assert not writer_thread.is_alive()
        assert not reader_thread.is_alive()
        assert writer_error == []
        assert reader_error == []
        assert listed_names == [[path]]
        assert read_context[0].labels.sensitivity.value == "secret"
        final_binding = runtime.store.get_file_label_binding(path)
        assert final_binding is not None
        assert final_binding.labels.sensitivity.value == "secret"


@pytest.mark.parametrize(
    ("operation", "path", "expected_parents"),
    [
        (
            "write_text",
            "derived-text/inner/result.txt",
            ("derived-text", "derived-text/inner"),
        ),
        (
            "write_directory",
            "derived-directory/inner/final",
            ("derived-directory", "derived-directory/inner"),
        ),
    ],
    ids=("write-text", "write-directory"),
)
def test_auto_created_parent_directories_inherit_written_labels(
    operation: str,
    path: str,
    expected_parents: tuple[str, ...],
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal=f"label parents for {operation}")
        source = _secret_source(runtime, pid)
        _register_file_sink(runtime, path, trust_level=SinkTrustLevel.TRUSTED)
        if operation == "write_text":
            runtime.filesystem.grant_path(
                pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            runtime.filesystem.write_text(
                pid,
                path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )
        else:
            runtime.filesystem.grant_directory(
                pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            runtime.filesystem.write_directory(
                pid,
                path,
                parents=True,
                source_oids=[source.oid],
            )

        for parent in expected_parents:
            binding = runtime.store.get_file_label_binding(parent)
            assert binding is not None
            assert binding.labels.sensitivity.value == "secret"
            assert source.oid in {item.oid for item in binding.source_refs}

        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        with runtime.data_flow.activate(DataFlowContext()):
            runtime.filesystem.read_directory(pid, ".")
            assert (
                runtime.data_flow.current_context().labels.sensitivity.value
                == "secret"
            )


def test_directory_listing_aggregates_returned_and_truncation_child_labels() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="read labeled directory entries")
        labeled_directory = root / "exports"
        labeled_directory.mkdir()
        normal_path = labeled_directory / "a-normal.txt"
        secret_path = labeled_directory / "z-secret-name.txt"
        normal_path.write_text("normal", encoding="utf-8")
        secret_path.write_text("secret", encoding="utf-8")
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path="exports/z-secret-name.txt",
            content=b"secret",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        runtime.filesystem.grant_directory(
            pid,
            "exports",
            [CapabilityRight.READ],
            issued_by="test",
        )

        with runtime.data_flow.activate(DataFlowContext()):
            listed = runtime.filesystem.read_directory(pid, "exports")
            assert [entry.name for entry in listed.entries] == [
                "a-normal.txt",
                "z-secret-name.txt",
            ]
            assert runtime.data_flow.current_context().labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            truncated = runtime.filesystem.read_directory(pid, "exports", limit=1)
            assert [entry.name for entry in truncated.entries] == ["a-normal.txt"]
            assert truncated.truncated is True
            assert runtime.data_flow.current_context().labels.sensitivity.value == "secret"

        normal_directory = root / "public"
        normal_directory.mkdir()
        (normal_directory / "only-normal.txt").write_text("normal", encoding="utf-8")
        runtime.filesystem.grant_directory(
            pid,
            "public",
            [CapabilityRight.READ],
            issued_by="test",
        )
        with runtime.data_flow.activate(DataFlowContext()):
            runtime.filesystem.read_directory(pid, "public")
            assert runtime.data_flow.current_context().labels.sensitivity.value == "normal"


def test_directory_listing_rejects_changed_child_binding_before_label_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="directory label replacement PoC")
        directory = root / "exports"
        directory.mkdir()
        secret_path = directory / "secret-codename.txt"
        secret_path.write_text("secret", encoding="utf-8")
        normalized_path = "exports/secret-codename.txt"
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=normalized_path,
            content=b"secret",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        runtime.filesystem.grant_directory(
            pid,
            "exports",
            [CapabilityRight.READ],
            issued_by="test",
        )
        original_list = runtime.filesystem.provider.list_directory

        def lower_labels_after_list(*args: Any, **kwargs: Any):
            children = list(original_list(*args, **kwargs))
            runtime.data_flow.tombstone_file(
                pid="test.host",
                normalized_path=normalized_path,
            )
            runtime.data_flow.bind_written_file(
                pid="test.host",
                normalized_path=normalized_path,
                content=b"secret",
                context=DataFlowContext(),
            )
            return children

        monkeypatch.setattr(
            runtime.filesystem.provider,
            "list_directory",
            lower_labels_after_list,
        )

        with runtime.data_flow.activate(DataFlowContext()):
            with pytest.raises(
                CapabilityDenied,
                match="directory-child label bindings changed",
            ):
                runtime.filesystem.read_directory(pid, "exports")
            assert runtime.data_flow.current_context().labels.sensitivity.value == "secret"


def test_bootstrap_removes_only_rules_deleted_from_host_config(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    database = tmp_path / "runtime.sqlite"
    configured_rule = SinkTrustRule(
        pattern="filesystem:workspace:configured.txt",
        trust_level=SinkTrustLevel.TRUSTED,
        max_sensitivity="secret",
    )
    manual_rule = SinkTrustRule(
        pattern="filesystem:workspace:manual.txt",
        trust_level=SinkTrustLevel.TRUSTED,
        max_sensitivity="secret",
    )
    configured = AgentLibOSConfig(
        data_flow=DataFlowDefaults(sink_rules=(configured_rule,)),
    )

    runtime = Runtime.open(
        database,
        substrate=LocalResourceProviderSubstrate(root),
        config=configured,
    )
    try:
        bootstrapped = runtime.inspect_sink_trust(configured_rule.pattern)
        assert bootstrapped is not None and bootstrapped.created_by == "runtime.bootstrap"
        runtime.data_flow.register_sink_trust(
            manual_rule,
            actor="test.host",
            require_capability=False,
        )
    finally:
        runtime.close()

    reopened = Runtime.open(
        database,
        substrate=LocalResourceProviderSubstrate(root),
        config=AgentLibOSConfig(),
    )
    try:
        assert reopened.inspect_sink_trust(configured_rule.pattern) is None
        manual = reopened.inspect_sink_trust(manual_rule.pattern)
        assert manual is not None and manual.created_by == "test.host"
        history = reopened.list_sink_trust(active_only=False)
        assert any(
            item.pattern == configured_rule.pattern and not item.active
            for item in history
        )
        assert any(
            record.actor == "runtime.bootstrap"
            and record.action == "data_flow.sink_trust.unregister"
            and record.target == configured_rule.pattern
            for record in reopened.audit.trace()
        )
        assert any(
            event.type == EventType.SINK_TRUST_UNREGISTERED
            and event.source == "runtime.bootstrap"
            and event.target == configured_rule.pattern
            for event in reopened.events.list(target=configured_rule.pattern)
        )
    finally:
        reopened.close()
