from __future__ import annotations

import pytest

from agent_libos.models import (
    CapabilityRight,
    EventType,
    ObjectMetadata,
    ObjectPatch,
    ObjectType,
    ProcessSignal,
    ProcessStatus,
    Provenance,
    SinkTrustLevel,
    SinkTrustRule,
    ViewMode,
)
from agent_libos.models.exceptions import CapabilityDenied
from tests.support.fakes import RecordingActionClient
from tests.support.public_errors import assert_public_error_message
from tests.support.runtime import temporary_runtime


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sensitivity", "top-secret"),
        ("trust_level", "model_asserted"),
        ("integrity", "perfect"),
        ("sensitivity", None),
    ],
)
def test_object_metadata_rejects_unknown_label_enum_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="invalid object data label"):
        ObjectMetadata(**{field: value})


def test_derived_object_conservatively_propagates_sensitivity_and_trust() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="labels")
        trusted = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"source": "internal"},
            metadata=ObjectMetadata(
                sensitivity="confidential",
                trust_level="verified",
                integrity="checked",
                origin="internal-db",
                tenant="tenant-a",
            ),
        )
        untrusted = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"source": "remote"},
            metadata=ObjectMetadata(
                sensitivity="restricted",
                trust_level="untrusted",
                integrity="untrusted",
                origin="remote-provider",
                tenant="tenant-a",
            ),
        )
        derived = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"summary": "derived"},
            metadata=ObjectMetadata(sensitivity="normal", trust_level="trusted", integrity="verified"),
            provenance=Provenance(
                created_from_action="llm.create_memory_object",
                parent_oids=[trusted.oid, untrusted.oid],
            ),
        )
        obj = runtime.memory.get_object(pid, derived)

        assert obj.metadata.sensitivity == "restricted"
        assert obj.metadata.trust_level == "untrusted"
        assert obj.metadata.integrity == "untrusted"
        assert obj.metadata.origin == "derived"
        assert obj.metadata.tenant == "tenant-a"


def test_explicit_identity_cannot_override_parent_identity() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="mixed labels")
        parent = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"source": "tenant-a"},
            metadata=ObjectMetadata(tenant="tenant-a", principal="principal-a"),
        )

        derived = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"summary": "cross-domain"},
            metadata=ObjectMetadata(tenant="tenant-b", principal="principal-b"),
            provenance=Provenance(parent_oids=[parent.oid]),
        )

        metadata = runtime.memory.get_object(pid, derived).metadata
        assert metadata.tenant == "mixed"
        assert metadata.principal == "mixed"


def test_context_manifest_and_explain_expose_labels_without_payload() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="explain labels")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"secret": "LABEL_PAYLOAD_MUST_NOT_LEAK"},
            metadata=ObjectMetadata(sensitivity="secret", trust_level="untrusted", origin="remote"),
        )
        view = runtime.memory.create_view(pid, [handle])
        context = runtime.memory.materialize_context(pid, view, charge_resources=False)

        assert context.object_manifest[0]["labels"]["sensitivity"] == "secret"
        assert context.object_manifest[0]["labels"]["trust_level"] == "untrusted"
        assert "LABEL_PAYLOAD_MUST_NOT_LEAK" not in str(context.object_manifest)


def test_label_downgrade_requires_explicit_declassification_capability() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="declassification")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret"),
            immutable=False,
        )
        with pytest.raises(CapabilityDenied):
            runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(metadata=ObjectMetadata(sensitivity="public")),
            )

        runtime.capability.issue_trusted(
            pid,
            f"declassification:object:{handle.oid}",
            [CapabilityRight.ADMIN],
            issued_by="test",
        )
        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(metadata=ObjectMetadata(sensitivity="public")),
        )
        assert runtime.memory.get_object(pid, handle).metadata.sensitivity == "public"


def test_finite_declassification_capability_is_consumed_once() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="finite declassification")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret"),
            immutable=False,
        )
        capability = runtime.capability.issue_trusted(
            pid,
            f"declassification:object:{handle.oid}",
            [CapabilityRight.ADMIN],
            issued_by="test",
            uses_remaining=1,
        )

        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(metadata=ObjectMetadata(sensitivity="public")),
        )
        consumed = runtime.store.get_capability(capability.cap_id)
        assert consumed is not None and consumed.uses_remaining == 0

        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(metadata=ObjectMetadata(sensitivity="secret")),
        )
        with pytest.raises(CapabilityDenied):
            runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(metadata=ObjectMetadata(sensitivity="public")),
            )


@pytest.mark.parametrize(
    ("current", "proposed"),
    [
        ({"tenant": "tenant-a"}, {"tenant": "tenant-b"}),
        ({"principal": "principal-a"}, {"principal": "principal-b"}),
        (
            {"declassification_authority": "authority-a"},
            {"declassification_authority": "authority-b"},
        ),
    ],
)
def test_identity_or_declassification_authority_change_requires_declassification(
    current: dict[str, str],
    proposed: dict[str, str],
) -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="identity relabel")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(**current),
            immutable=False,
        )

        with pytest.raises(CapabilityDenied):
            runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(metadata=ObjectMetadata(**proposed)),
            )


def test_llm_created_object_unions_explicit_and_all_materialized_context_parents() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="derive from context")
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="llm:default",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
                tenants=("tenant-a",),
                identity_sha256=runtime.llms.profile_identity_sha256("default"),
            ),
            actor="test",
            require_capability=False,
        )
        context_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"secret": "materialized source"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        explicit_parent = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"public": "explicit source"},
            metadata=ObjectMetadata(sensitivity="public", tenant="tenant-a"),
        )
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(
            pid,
            [context_source],
            mode=ViewMode.READ_ONLY,
        )
        runtime.store.update_process(process)
        runtime.llm.client = RecordingActionClient(
            [
                {
                    "action": "create_memory_object",
                    "type": "summary",
                    "payload": {"derived": True},
                    "parent_oids": [explicit_parent.oid],
                }
            ]
        )
        runtime.skills.activate_skill(pid, "agent-libos-object-memory", actor=pid)

        result = runtime.run_next_process_once()
        assert result["ok"], result
        derived = next(
            obj
            for obj in runtime.store.list_objects(namespace=runtime.memory.resolve_namespace(pid))
            if obj.payload == {"derived": True}
        )
        assert explicit_parent.oid in derived.provenance.parent_oids
        assert context_source.oid in derived.provenance.parent_oids
        assert derived.metadata.sensitivity == "secret"
        assert derived.metadata.tenant == "tenant-a"


def test_llm_created_object_fails_closed_for_missing_or_unreadable_explicit_parent() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="validate parents")
        other = runtime.process.spawn(image="base-agent:v0", goal="own hidden parent")
        hidden = runtime.memory.create_object(
            other,
            ObjectType.EVIDENCE,
            {"secret": "not readable"},
            metadata=ObjectMetadata(sensitivity="secret"),
        )

        missing = runtime.tools.call(
            pid,
            "create_memory_object",
            {
                "type": "summary",
                "payload": {"invalid": "missing"},
                "parent_oids": ["obj_missing"],
            },
        )
        unreadable = runtime.tools.call(
            pid,
            "create_memory_object",
            {
                "type": "summary",
                "payload": {"invalid": "unreadable"},
                "parent_oids": [hidden.oid],
            },
        )

        assert not missing.ok
        assert_public_error_message(
            missing.error,
            code="permission_denied",
            error_type="CapabilityDenied",
            forbidden=("parent not found", "obj_missing"),
        )
        assert not unreadable.ok
        assert_public_error_message(
            unreadable.error,
            code="permission_denied",
            error_type="CapabilityDenied",
            forbidden=("parent is not readable", hidden.oid),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"trust_level": "trusted"},
        {"integrity": "verified"},
        {"declassification_authority": "model-selected"},
    ],
)
def test_llm_memory_tool_cannot_assert_trusted_labels(metadata: dict[str, str]) -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject trusted labels")

        result = runtime.tools.call(
            pid,
            "create_memory_object",
            {
                "type": "summary",
                "payload": {"value": "untrusted model output"},
                "metadata": metadata,
            },
        )

        assert not result.ok
        assert_public_error_message(
            result.error,
            code="permission_denied",
            error_type="ToolExecutionError",
            forbidden=("cannot",),
        )


def test_trusted_tool_context_sources_label_child_goal() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": ["analyst-a"],
                }
            },
        )
        runtime.capability.grant(parent, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                tenant="tenant-a",
                principal="analyst-a",
            ),
        )

        result = runtime.tools.call(
            parent,
            "spawn_child_process",
            {"goal": "summarize the observed value"},
            context_metadata={
                "data_flow_context": runtime.data_flow.context_from_source_oids(
                    parent,
                    [secret.oid],
                    include_current=False,
                )
            },
        )

        assert result.ok, result.error
        child = runtime.process.get(result.payload["child_pid"])
        goal = runtime.store.get_object(child.goal_oid)
        assert goal is not None
        assert goal.metadata.sensitivity == "secret"
        assert goal.metadata.tenant == "tenant-a"
        assert goal.metadata.principal == "analyst-a"
        assert secret.oid in goal.provenance.parent_oids


def test_secret_message_read_taints_later_child_goal_and_reply() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        runtime.capability.grant(parent, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
        child = runtime.spawn_child_process(parent, "child")
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        parent_process = runtime.process.get(parent)
        assert parent_process.memory_view is not None
        parent_process.memory_view.roots.append(secret)
        runtime.store.update_process(parent_process)

        sent = runtime.messages.send_from_process(parent, child, body="derived secret")
        persisted = runtime.store.get_process_message(sent.message_id)
        assert persisted is not None
        assert persisted.metadata["data_labels"]["sensitivity"] == "secret"
        assert secret.oid in persisted.metadata["source_oids"]
        persisted_refs = persisted.metadata["data_flow_context"]["source_refs"]
        assert any(ref["oid"] == secret.oid and len(ref["content_sha256"]) == 64 for ref in persisted_refs)

        read = runtime.tools.call(child, "read_process_messages", {})
        assert read.ok, read.error
        # The model-facing message projection carries actionable message
        # content only. Labels/provenance remain on the durable ToolResult and
        # the metadata-only carrier instead of duplicating raw message
        # metadata (including source ids) into the prompt.
        assert "metadata" not in read.payload["messages"][0]
        assert read.result_handle is not None
        stored_read = runtime.store.get_object(read.result_handle.oid)
        assert stored_read is not None
        read_flow = stored_read.payload["metadata"]["data_flow_context"]
        assert read_flow["labels"]["sensitivity"] == "secret"
        assert all(len(ref["content_sha256"]) == 64 for ref in read_flow["source_refs"])
        child_process = runtime.process.get(child)
        assert child_process.memory_view is not None
        carrier_objects = [runtime.store.get_object(handle.oid) for handle in child_process.memory_view.roots]
        assert any(obj is not None and obj.metadata.sensitivity == "secret" for obj in carrier_objects)

        reply = runtime.messages.send_from_process(child, parent, body="secret-derived reply")
        assert reply.metadata["data_labels"]["sensitivity"] == "secret"

        runtime.capability.grant(child, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
        grandchild = runtime.spawn_child_process(child, "continue secret work")
        grandchild_goal = runtime.store.get_object(runtime.process.get(grandchild).goal_oid)
        assert grandchild_goal is not None
        assert grandchild_goal.metadata.sensitivity == "secret"
        assert grandchild_goal.metadata.tenant == "tenant-a"


def test_process_message_rejects_recipient_identity_domain_mismatch() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        child = runtime.process.spawn_child(
            parent,
            "restricted child",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                }
            },
        )
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "tenant data"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        parent_process = runtime.process.get(parent)
        assert parent_process.memory_view is not None
        parent_process.memory_view.roots.append(secret)
        runtime.store.update_process(parent_process)

        with pytest.raises(CapabilityDenied, match="data_flow_policy"):
            runtime.messages.send_from_process(parent, child, body="must not cross tenant domain")

        assert runtime.messages.unread(child) == []


def test_fork_rejects_explicit_root_outside_child_identity_domain() -> None:
    with temporary_runtime() as runtime:
        owner = runtime.process.spawn(
            goal="labeled Object owner",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": ["principal-a"],
                }
            },
        )
        parent = runtime.process.spawn(
            goal="restricted fork parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                }
            },
        )
        secret = runtime.memory.create_object(
            owner,
            ObjectType.EVIDENCE,
            {"marker": "must not enter child view"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                tenant="tenant-a",
                principal="principal-a",
            ),
            name="fork.forbidden.root",
        )
        runtime.capability.grant(
            parent,
            f"object:{secret.oid}",
            [CapabilityRight.READ, CapabilityRight.MATERIALIZE],
            issued_by="test",
        )
        runtime.capability.grant(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        children_before = {child.pid for child in runtime.process.list_children(parent)}

        result = runtime.tools.call(
            parent,
            "fork_child_process",
            {
                "goal": "unlabeled child goal",
                "include_parent_roots": False,
                "root_oids": [secret.oid],
            },
        )

        assert not result.ok
        assert_public_error_message(
            result.error,
            code="permission_denied",
            error_type="CapabilityDenied",
            forbidden=("data_flow_policy does not allow", secret.oid),
        )
        assert {child.pid for child in runtime.process.list_children(parent)} == children_before


def test_exec_replacement_goal_keeps_observed_secret_label() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="before exec")
        secret = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret"),
        )
        process = runtime.process.get(pid)
        assert process.memory_view is not None
        process.memory_view.roots.append(secret)
        runtime.store.update_process(process)

        runtime.exec_process(
            pid,
            "base-agent:v0",
            goal="replacement goal",
            preserve_memory=False,
        )

        replacement = runtime.store.get_object(runtime.process.get(pid).goal_oid)
        assert replacement is not None
        assert replacement.metadata.sensitivity == "secret"
        assert secret.oid in replacement.provenance.parent_oids


def test_exit_message_becomes_labeled_result_and_wait_preserves_sources() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        child = runtime.process.spawn_child(parent, "child")
        secret = runtime.memory.create_object(
            child,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        child_process = runtime.process.get(child)
        assert child_process.memory_view is not None
        child_process.memory_view.roots.append(secret)
        runtime.store.update_process(child_process)

        raw_message = "classified completion summary"
        runtime.process.exit(child, message=raw_message)

        exited = runtime.process.get(child)
        assert exited.status == ProcessStatus.EXITED
        assert exited.status_message is not None
        assert exited.status_message.startswith("result_oid:")
        assert raw_message not in exited.status_message
        result_oid = exited.status_message.split(":", 1)[1]
        result = runtime.store.get_object(result_oid)
        assert result is not None
        assert result.payload == {"message": raw_message}
        assert result.metadata.sensitivity == "secret"
        assert result.metadata.tenant == "tenant-a"
        assert secret.oid in result.provenance.parent_oids

        waited = runtime.process.wait(parent, child, timeout=0)
        assert waited.result is not None and waited.result.oid == result_oid
        delivered = runtime.memory.get_object(parent, waited.result)
        assert delivered.metadata.sensitivity == "secret"
        assert secret.oid in delivered.provenance.parent_oids
        exit_audit = [record for record in runtime.audit.trace() if record.action == "process.exit"][-1]
        assert raw_message not in str(exit_audit.decision)


def test_self_exit_carrier_preserves_host_injected_identity_without_inbound_rejection() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(
            goal="self exit",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                }
            },
        )
        injected = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "host managed tenant data"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        process = runtime.process.get(pid)
        assert process.memory_view is not None
        process.memory_view.roots.append(injected)
        runtime.store.update_process(process)

        runtime.process.exit(pid, failed=True, message="upstream operation denied")

        exited = runtime.process.get(pid)
        assert exited.status == ProcessStatus.FAILED
        assert exited.status_message is not None and exited.status_message.startswith("result_oid:")
        result = runtime.store.get_object(exited.status_message.split(":", 1)[1])
        assert result is not None
        assert result.metadata.sensitivity == "secret"
        assert result.metadata.tenant == "tenant-a"
        assert injected.oid in result.provenance.parent_oids


def test_signal_reason_uses_labeled_carrier_and_taints_recipient() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        child = runtime.process.spawn_child(parent, "child")
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        parent_process = runtime.process.get(parent)
        assert parent_process.memory_view is not None
        parent_process.memory_view.roots.append(secret)
        runtime.store.update_process(parent_process)

        raw_reason = "pause because of classified evidence"
        paused = runtime.process.signal_child(parent, child, ProcessSignal.PAUSE, reason=raw_reason)

        assert paused.status == ProcessStatus.PAUSED
        assert paused.status_message is not None and paused.status_message.startswith("result_oid:")
        assert raw_reason not in paused.status_message
        carrier_oid = paused.status_message.split(":", 1)[1]
        carrier = runtime.store.get_object(carrier_oid)
        assert carrier is not None
        assert carrier.payload == {"reason": raw_reason}
        assert carrier.metadata.sensitivity == "secret"
        assert carrier.metadata.tenant == "tenant-a"
        assert secret.oid in carrier.provenance.parent_oids
        assert raw_reason not in str(
            [record.decision for record in runtime.audit.trace() if record.action == "process.signal_child"][-1]
        )


def test_signal_failure_rolls_back_reason_carrier_and_child_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(goal="parent")
        child = runtime.process.spawn_child(parent, "child")
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret"),
        )
        parent_process = runtime.process.get(parent)
        assert parent_process.memory_view is not None
        parent_process.memory_view.roots.append(secret)
        runtime.store.update_process(parent_process)

        child_before = runtime.process.get(child)
        assert child_before.memory_view is not None
        root_oids_before = [handle.oid for handle in child_before.memory_view.roots]
        namespace = runtime.memory.resolve_namespace(child)
        object_oids_before = {obj.oid for obj in runtime.store.list_objects(namespace=namespace)}
        signal_event_ids_before = {
            event.event_id
            for event in runtime.events.list(target=child)
            if event.type == EventType.PROCESS_SIGNAL
        }
        original_emit = runtime.events.emit

        def fail_signal_event(event_type: EventType, *args: object, **kwargs: object) -> object:
            if event_type == EventType.PROCESS_SIGNAL:
                raise RuntimeError("injected process signal event failure")
            return original_emit(event_type, *args, **kwargs)

        monkeypatch.setattr(runtime.events, "emit", fail_signal_event)
        with pytest.raises(RuntimeError, match="injected process signal event failure"):
            runtime.process.signal_child(
                parent,
                child,
                ProcessSignal.PAUSE,
                reason="classified signal reason",
                source_oids=[secret.oid],
            )

        child_after = runtime.process.get(child)
        assert child_after.status == ProcessStatus.RUNNABLE
        assert child_after.status_message is None
        assert child_after.memory_view is not None
        assert [handle.oid for handle in child_after.memory_view.roots] == root_oids_before
        assert {
            obj.oid for obj in runtime.store.list_objects(namespace=namespace)
        } == object_oids_before
        assert {
            event.event_id
            for event in runtime.events.list(target=child)
            if event.type == EventType.PROCESS_SIGNAL
        } == signal_event_ids_before


def test_host_signal_failure_rolls_back_reason_carrier_and_process_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="host-managed process")
        secret = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "classified"},
            metadata=ObjectMetadata(sensitivity="secret"),
        )
        process = runtime.process.get(pid)
        assert process.memory_view is not None
        process.memory_view.roots.append(secret)
        runtime.store.update_process(process)

        root_oids_before = [handle.oid for handle in process.memory_view.roots]
        namespace = runtime.memory.resolve_namespace(pid)
        object_oids_before = {obj.oid for obj in runtime.store.list_objects(namespace=namespace)}
        original_emit = runtime.events.emit

        def fail_signal_event(event_type: EventType, *args: object, **kwargs: object) -> object:
            if event_type == EventType.PROCESS_SIGNAL:
                raise RuntimeError("injected host process signal event failure")
            return original_emit(event_type, *args, **kwargs)

        monkeypatch.setattr(runtime.events, "emit", fail_signal_event)
        with pytest.raises(RuntimeError, match="injected host process signal event failure"):
            runtime.process.signal(
                pid,
                ProcessSignal.PAUSE,
                {"reason": "classified host signal reason"},
            )

        after = runtime.process.get(pid)
        assert after.status == ProcessStatus.RUNNABLE
        assert after.status_message is None
        assert after.memory_view is not None
        assert [handle.oid for handle in after.memory_view.roots] == root_oids_before
        assert {
            obj.oid for obj in runtime.store.list_objects(namespace=namespace)
        } == object_oids_before


def test_signal_reason_rejects_recipient_identity_domain_mismatch_before_transition() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                }
            },
        )
        child = runtime.process.spawn_child(
            parent,
            "restricted child",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                }
            },
        )
        secret = runtime.memory.create_object(
            parent,
            ObjectType.ARTIFACT,
            {"value": "tenant data"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        parent_process = runtime.process.get(parent)
        assert parent_process.memory_view is not None
        parent_process.memory_view.roots.append(secret)
        runtime.store.update_process(parent_process)

        with pytest.raises(CapabilityDenied, match="data_flow_policy"):
            runtime.process.signal_child(
                parent,
                child,
                ProcessSignal.PAUSE,
                reason="must not cross tenant domain",
            )

        unchanged = runtime.process.get(child)
        assert unchanged.status == ProcessStatus.RUNNABLE
        assert unchanged.status_message is None


def test_merge_checks_non_root_child_objects_against_parent_identity_domain() -> None:
    with temporary_runtime() as runtime:
        parent = runtime.process.spawn(
            goal="parent",
            authority_manifest={
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                }
            },
        )
        child = runtime.process.spawn_child(parent, "child")
        injected = runtime.memory.create_object(
            child,
            ObjectType.ARTIFACT,
            {"value": "host injected tenant data"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        child_process = runtime.process.get(child)
        assert child_process.memory_view is not None
        assert all(root.oid != injected.oid for root in child_process.memory_view.roots)
        runtime.process.exit(child)

        with pytest.raises(CapabilityDenied, match="data_flow_policy"):
            runtime.process.merge_child_memory(parent, child)

        with pytest.raises(CapabilityDenied):
            runtime.memory.handle_for_oid(parent, injected.oid, required_rights={"read"})
