from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import pytest

from agent_libos.models import (
    CapabilityRight,
    ContextMaterializationManifest,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectFilter,
    ObjectMetadata,
    ObjectType,
    OperationEvidenceRole,
    ResourceBudget,
)
from agent_libos.models.exceptions import NotFound
from tests.support.fakes import RecordingActionClient
from tests.support.runtime import temporary_runtime, workspace_runtime
from agent_libos.evidence.external_effects import (
    abandon_external_effect_intent,
    record_external_effect,
)
from agent_libos.runtime.runtime import Runtime
from tests.support.external_effects import begin_external_effect_intent


@pytest.mark.parametrize('evidence_kind', ['audit', 'event'])
def test_primary_evidence_and_operation_link_commit_atomically(
    monkeypatch: pytest.MonkeyPatch,
    evidence_kind: str,
) -> None:
    with temporary_runtime() as runtime:
        original_insert = runtime.operations.store.insert_operation_evidence

        def fail_selected_link(link: object) -> object:
            if getattr(link, 'evidence_type', None) == evidence_kind:
                raise RuntimeError(f'injected {evidence_kind} link failure')
            return original_insert(link)

        monkeypatch.setattr(
            runtime.operations.store,
            'insert_operation_evidence',
            fail_selected_link,
        )
        with runtime.operations.scope(
            kind='runtime',
            name=f'test.atomic-{evidence_kind}',
            actor='test',
            pid=None,
        ):
            with pytest.raises(RuntimeError, match=f'injected {evidence_kind} link failure'):
                if evidence_kind == 'audit':
                    runtime.audit.record(
                        actor='test',
                        action='test.atomic_evidence',
                        target='test:target',
                    )
                else:
                    runtime.events.emit(
                        EventType.OBJECT_UPDATED,
                        source='test',
                        target='test:target',
                        payload={'atomic': True},
                    )

        if evidence_kind == 'audit':
            assert not any(
                record.action == 'test.atomic_evidence'
                for record in runtime.audit.trace()
            )
        else:
            assert not any(
                event.payload.get('atomic') is True
                for event in runtime.events.list()
            )


def test_direct_protected_operation_is_persisted_and_explainable() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="explain spawn")
        runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"value": "direct"})
        runtime.checkpoint.create(pid, "direct checkpoint", require_capability=False)

        roots = [
            operation
            for operation in runtime.store.list_operations(pid=pid)
            if operation.parent_operation_id is None and operation.name == "process.spawn"
        ]
        assert len(roots) == 1
        explanation = runtime.explain.explain_operation(roots[0].operation_id)

        assert explanation["root"]["outcome"] == "succeeded"
        assert explanation["evidence_complete"] is True
        assert explanation["missing_evidence"] == []
        assert any(item["evidence_type"] == "audit" for item in explanation["evidence"])
        direct_roots = {
            operation.name: operation
            for operation in runtime.store.list_operations(pid=pid)
            if operation.parent_operation_id is None
        }
        assert {"process.spawn", "memory.create_object", "checkpoint.create"} <= direct_roots.keys()
        assert all(
            direct_roots[name].outcome.value == "succeeded"
            for name in ("process.spawn", "memory.create_object", "checkpoint.create")
        )


def test_process_exec_is_one_operation_covering_publication_and_evidence() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="before exec")
        runtime.capability.grant(
            pid,
            runtime.image_registry.resource_for("review-agent:v0"),
            [CapabilityRight.READ],
            issued_by="test",
        )

        runtime.exec_process(pid, "review-agent:v0", goal="after exec")

        operations = [
            operation
            for operation in runtime.store.list_operations(pid=pid)
            if operation.name == "process.exec"
        ]
        assert len(operations) == 1
        operation = operations[0]
        assert operation.parent_operation_id is None
        assert operation.outcome.value == "succeeded"

        publications = [
            publication
            for publication in runtime.store.list_runtime_publications(pid=pid)
            if publication["kind"] == "process_exec"
        ]
        assert len(publications) == 1
        publication = publications[0]
        assert publication["state"] == "committed"
        assert publication["plan"]["operation_id"] == operation.operation_id

        links = runtime.store.list_operation_evidence(
            operation_ids=[operation.operation_id]
        )
        assert {link.evidence_type for link in links} >= {"audit", "event"}
        commit_receipt = publication["receipt"]["phases"][-1]
        assert commit_receipt["phase"] == "committed"
        assert commit_receipt["audit_id"] in {
            link.evidence_id for link in links if link.evidence_type == "audit"
        }
        assert commit_receipt["event_id"] in {
            link.evidence_id for link in links if link.evidence_type == "event"
        }
        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["evidence_complete"] is True
        assert explanation["root"]["outcome"] == "succeeded"


def test_tool_primitive_capability_effect_and_resource_evidence_form_one_tree() -> None:
    with workspace_runtime() as (runtime, root):
        relative = "input.txt"
        (root / relative).write_text("explain me", encoding="utf-8")
        pid = runtime.process.spawn(image="review-agent:v0", goal="read")
        capability = runtime.capability.issue_trusted(
            pid,
            runtime.filesystem.resource_for(relative),
            [CapabilityRight.READ],
            issued_by="test",
            uses_remaining=1,
        )

        result = runtime.tools.call(pid, "read_text_file", {"path": relative})
        assert result.ok
        operation = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "tool.read_text_file" and item.parent_operation_id is None
        )
        explanation = runtime.explain.explain_operation(operation.operation_id)

        assert explanation["evidence_complete"] is True
        assert {item["kind"] for item in explanation["operations"]} >= {"tool_call", "primitive"}
        evidence_types = {item["evidence_type"] for item in explanation["evidence"]}
        assert {"tool_call", "audit", "event", "external_effect", "capability_reservation"} <= evidence_types
        assert explanation["summary"]["external_effects"][0]["state"] == "finalized"
        assert runtime.store.get_capability(capability.cap_id).uses_remaining == 0
        assert runtime.explain.resolve("call", result.call_id)["root"]["operation_id"] == operation.operation_id


def test_denied_tool_has_decision_evidence_and_no_external_effect() -> None:
    with workspace_runtime() as (runtime, root):
        (root / "secret.txt").write_text("secret", encoding="utf-8")
        pid = runtime.process.spawn(image="review-agent:v0", goal="denied read")

        result = runtime.tools.call(pid, "read_text_file", {"path": "secret.txt"})
        assert not result.ok
        operation = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "tool.read_text_file" and item.parent_operation_id is None
        )
        explanation = runtime.explain.explain_operation(operation.operation_id)

        assert explanation["root"]["outcome"] == "denied"
        assert explanation["evidence_complete"] is True
        assert explanation["summary"]["external_effects"] == []
        assert explanation["summary"]["authorization"]
        assert explanation["summary"]["authorization"][-1]["allowed"] is False


def test_tool_resource_limit_is_a_denied_complete_operation() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="resource limit",
            resource_budget=ResourceBudget(max_tool_calls=1),
        )
        runtime.tools.configure_process_tools(pid, ["get_working_directory"], assigned_by="test")
        assert runtime.tools.call(pid, "get_working_directory", {}).ok

        denied = runtime.tools.call(pid, "get_working_directory", {})

        assert not denied.ok
        operation = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "tool.get_working_directory" and item.outcome.value == "denied"
        )
        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["root"]["outcome"] == "denied"
        assert explanation["evidence_complete"] is True
        assert {item["evidence_type"] for item in explanation["evidence"]} >= {"tool_call", "audit", "event"}
        succeeded = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "tool.get_working_directory" and item.outcome.value == "succeeded"
        )
        succeeded_explanation = runtime.explain.explain_operation(succeeded.operation_id)
        assert succeeded_explanation["summary"]["resource_charge_count"] == 1
        assert succeeded_explanation["summary"]["resource_consumption"][0]["usage"]["tool_calls"] == 1


def test_pending_provider_effect_marks_primitive_and_tool_outcome_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    with workspace_runtime() as (runtime, root):
        (root / "input.txt").write_text("uncertain", encoding="utf-8")
        pid = runtime.process.spawn(image="review-agent:v0", goal="unknown read")
        runtime.filesystem.grant_path(pid, "input.txt", [CapabilityRight.READ], issued_by="test")
        emit = runtime.events.emit

        def fail_external_read(event_type, *args, **kwargs):
            if EventType(event_type) == EventType.EXTERNAL_READ:
                raise RuntimeError("event sink failed after provider read")
            return emit(event_type, *args, **kwargs)

        monkeypatch.setattr(runtime.events, "emit", fail_external_read)
        result = runtime.tools.call(pid, "read_text_file", {"path": "input.txt"})

        assert not result.ok
        tool = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "tool.read_text_file" and item.parent_operation_id is None
        )
        explanation = runtime.explain.explain_operation(tool.operation_id)
        assert explanation["root"]["outcome"] == "unknown"
        assert any(item["reason"] == "provider_outcome_unknown" for item in explanation["uncertainties"])


def test_pending_provider_effect_propagates_unknown_to_llm_root(monkeypatch: pytest.MonkeyPatch) -> None:
    with workspace_runtime() as (runtime, root):
        (root / "input.txt").write_text("uncertain", encoding="utf-8")
        runtime.llm.client = RecordingActionClient(
            [{"action": "read_text_file", "path": "input.txt"}]
        )
        pid = runtime.process.spawn(image="review-agent:v0", goal="unknown LLM read")
        runtime.tools.configure_model_tool_projection(
            pid,
            [*runtime.process.get(pid).model_tool_table, "read_text_file"],
            assigned_by="test",
        )
        runtime.filesystem.grant_path(pid, "input.txt", [CapabilityRight.READ], issued_by="test")
        emit = runtime.events.emit

        def fail_external_read(event_type, *args, **kwargs):
            if EventType(event_type) == EventType.EXTERNAL_READ:
                raise RuntimeError("event sink failed after provider read")
            return emit(event_type, *args, **kwargs)

        monkeypatch.setattr(runtime.events, "emit", fail_external_read)
        result = runtime.run_next_process_once()
        llm_root = next(
            item
            for item in runtime.store.list_operations(pid=pid)
            if item.name == "llm.action_selection" and item.parent_operation_id is None
        )
        explanation = runtime.explain.explain_operation(llm_root.operation_id)

        assert result["ok"] is True
        assert result["result"]["ok"] is False
        assert explanation["root"]["outcome"] == "unknown"
        assert explanation["summary"]["headline"] == "llm.action_selection has an unknown external outcome."


def test_certified_pre_boundary_failure_is_not_started_not_unknown() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="provider did not start")
        with runtime.operations.scope(
            kind="primitive",
            name="test.provider.not_started",
            actor=pid,
            pid=pid,
            auto_finish=False,
        ) as operation:
            intent = begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="test",
                operation="write",
                target="resource:test",
                state_mutation=True,
                information_flow=False,
            )
            runtime.events.emit(EventType.TOOL_FAILED, source="test", target=pid, payload={"phase": "pre_boundary"})
            runtime.audit.record(
                actor=pid,
                action="test.provider.not_started",
                target="resource:test",
                decision={"provider_started": False},
            )
            abandon_external_effect_intent(
                runtime.uow.protected_effects,
                intent.effect_id,
                operations=runtime.operations,
            )
            runtime.operations.finish("failed", operation_id=operation.operation_id)

        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["root"]["outcome"] == "failed"
        assert explanation["evidence_complete"] is True
        assert explanation["summary"]["external_effects"] == [
            {
                "effect_id": intent.effect_id,
                "provider": "test",
                "operation": "write",
                "state": "abandoned",
                "outcome": "not_started",
                "rollback_class": None,
                "rollback_status": None,
            }
        ]
        assert not any(item["reason"] == "provider_outcome_unknown" for item in explanation["uncertainties"])


def test_finalized_unknown_effect_marks_operation_unknown() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="finalized unknown")
        operation_id = None
        with pytest.raises(RuntimeError, match="provider failed after dispatch"):
            with runtime.operations.scope(
                kind="primitive",
                name="test.provider.unknown",
                actor=pid,
                pid=pid,
            ) as operation:
                operation_id = operation.operation_id
                intent = begin_external_effect_intent(
                    runtime,
                    pid=pid,
                    provider="test",
                    operation="write",
                    target="resource:test",
                    state_mutation=True,
                    information_flow=False,
                )
                record_external_effect(
                    runtime.uow.protected_effects,
                    pid=pid,
                    provider="test",
                    operation="write",
                    target="resource:test",
                    classification=ExternalEffectClassification(
                        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                        state_mutation=True,
                        information_flow=False,
                        metadata={"outcome": "unknown_after_provider_exception"},
                    ),
                    audit_record=None,
                    event=None,
                    intent_effect_id=intent.effect_id,
                    operations=runtime.operations,
                )
                raise RuntimeError("provider failed after dispatch")

        assert operation_id is not None
        assert runtime.store.get_operation(operation_id).outcome.value == "unknown"
        explanation = runtime.explain.explain_operation(operation_id)
        assert explanation["root"]["outcome"] == "unknown"
        assert any(item["reason"] == "operation_outcome_unknown" for item in explanation["uncertainties"])


def test_successful_return_cannot_hide_finalized_unknown_effect() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="finalized unknown without exception")
        with runtime.operations.scope(
            kind="primitive",
            name="test.provider.unknown_return",
            actor=pid,
            pid=pid,
        ) as operation:
            intent = begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="test",
                operation="write",
                target="resource:test",
                state_mutation=True,
                information_flow=False,
            )
            record_external_effect(
                runtime.uow.protected_effects,
                pid=pid,
                provider="test",
                operation="write",
                target="resource:test",
                classification=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                    rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                    state_mutation=True,
                    information_flow=False,
                    metadata={"outcome": "unknown_after_provider_boundary"},
                ),
                audit_record=None,
                event=None,
                intent_effect_id=intent.effect_id,
                operations=runtime.operations,
            )

        stored = runtime.store.get_operation(operation.operation_id)
        assert stored is not None and stored.outcome.value == "unknown"
        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["root"]["outcome"] == "unknown"
        assert any(item["reason"] == "operation_outcome_unknown" for item in explanation["uncertainties"])


def test_unknown_rollback_status_does_not_hide_known_committed_outcome() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="known outcome with unknown rollback")
        with runtime.operations.scope(
            kind="primitive",
            name="test.provider.committed_unknown_rollback",
            actor=pid,
            pid=pid,
        ) as operation:
            intent = begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="test",
                operation="write",
                target="resource:test",
                state_mutation=True,
                information_flow=False,
            )
            effect = record_external_effect(
                runtime.uow.protected_effects,
                pid=pid,
                provider="test",
                operation="write",
                target="resource:test",
                classification=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                    rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                    state_mutation=True,
                    information_flow=False,
                    metadata={"outcome": "completed"},
                ),
                audit_record=None,
                event=None,
                intent_effect_id=intent.effect_id,
                operations=runtime.operations,
            )

        stored = runtime.store.get_operation(operation.operation_id)
        assert effect.transaction_state == "committed"
        assert stored is not None and stored.outcome.value == "succeeded"
        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["root"]["outcome"] == "succeeded"
        assert explanation["uncertainties"] == []


def test_llm_request_persists_context_manifest_without_object_payloads() -> None:
    with temporary_runtime() as runtime:
        runtime.llm.client = RecordingActionClient(
            [{"action": "create_memory_object", "type": "observation", "payload": {"secret": "payload-marker"}}]
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="materialize context")

        result = runtime.run_next_process_once()
        assert result["ok"]
        manifests = runtime.store.list_context_materialization_manifests(pid=pid)
        assert len(manifests) == 1
        manifest = manifests[0]
        assert manifest.rendered_tokens > 0
        assert manifest.rendered_sha256
        assert manifest.objects
        assert "payload-marker" not in json.dumps(manifest.objects)
        explanation = runtime.explain.resolve("context", manifest.materialization_id)
        assert explanation["summary"]["context"][0]["materialization_id"] == manifest.materialization_id
        assert "payload-marker" not in json.dumps(explanation)


def test_human_wait_and_resume_reuse_llm_and_tool_operation_ids() -> None:
    with temporary_runtime() as runtime:
        runtime.llm.client = RecordingActionClient(
            [{"action": "ask_human", "question": "Continue?", "context": {"phase": "test"}}]
        )
        pid = runtime.process.spawn(image="review-agent:v0", goal="wait and resume")
        runtime.capability.grant(pid, "human:owner", [CapabilityRight.WRITE], issued_by="test")

        first = runtime.run_next_process_once()
        assert first["waiting_human"] is True
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None
        llm_operation_id = pending["llm_operation_id"]
        tool_operation_id = pending["tool_operation_id"]
        assert runtime.store.get_operation(llm_operation_id).state.value == "waiting"
        assert runtime.store.get_operation(tool_operation_id).state.value == "waiting"

        runtime.human.approve(
            first["request_id"],
            {"approved": True, "answer": "ultraviolet_human_phrase"},
        )
        resumed = runtime.run_next_process_once()

        assert resumed["ok"] is True
        assert runtime.store.get_operation(llm_operation_id).outcome.value == "succeeded"
        assert runtime.store.get_operation(tool_operation_id).outcome.value == "succeeded"
        explanation = runtime.explain.explain_operation(llm_operation_id)
        assert any(item["evidence_type"] == "human_request" for item in explanation["evidence"])
        assert "ultraviolet_human_phrase" not in json.dumps(explanation)


def test_explain_output_redacts_sensitive_audit_values() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="redact")
        with runtime.operations.scope(
            kind="runtime",
            name="test.secret",
            actor=pid,
            pid=pid,
            expected_roles=["audit"],
            metadata={
                "password": "OPERATION_METADATA_SECRET",
                "argv": ["echo", "OPERATION_ARGV_SECRET"],
                "command_line": "OPERATION_COMMAND_SECRET",
            },
        ) as operation:
            runtime.audit.record(
                actor=pid,
                action="test.secret",
                decision={
                    "password": "DO_NOT_LEAK",
                    "token": "SECRET_TOKEN",
                    "argv": ["echo", "PRIVATE_ARG_MARKER"],
                    "arguments_preview": "PRIVATE_PREVIEW_MARKER",
                    "environment": {"VISIBLE_NAME": "PRIVATE_ENV_MARKER"},
                    "canonical_args_hash": "a" * 64,
                },
            )
            runtime.audit.record(
                actor=pid,
                action="capability.authorize",
                decision={
                    "allowed": False,
                    "reason": "policy denied",
                    "resource": "filesystem:workspace:report.txt",
                    "right": "read",
                },
            )

        payload = runtime.explain.explain_operation(operation.operation_id)
        encoded = json.dumps(payload)
        assert "DO_NOT_LEAK" not in encoded
        assert "SECRET_TOKEN" not in encoded
        assert "OPERATION_METADATA_SECRET" not in encoded
        assert "OPERATION_ARGV_SECRET" not in encoded
        assert "OPERATION_COMMAND_SECRET" not in encoded
        assert "PRIVATE_ARG_MARKER" not in encoded
        assert "PRIVATE_PREVIEW_MARKER" not in encoded
        assert "PRIVATE_ENV_MARKER" not in encoded
        assert "policy denied" in encoded
        assert "a" * 64 in encoded
        assert "[redacted]" in encoded


def test_concurrent_evidence_expectation_cannot_resurrect_terminal_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with temporary_runtime() as runtime:
        operation = runtime.operations.start(
            kind="runtime",
            name="test.concurrent_expect",
            actor="host",
            pid=None,
        )
        update_started = threading.Event()
        release_update = threading.Event()
        finish_started = threading.Event()
        original_update = runtime.store.update_operation

        def delayed_update(record: object, **kwargs: object) -> bool:
            if (
                getattr(record, "operation_id", None) == operation.operation_id
                and "decision" in getattr(record, "expected_roles", [])
                and getattr(getattr(record, "state", None), "value", None) == "running"
            ):
                update_started.set()
                assert release_update.wait(timeout=2)
            return original_update(record, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(runtime.store, "update_operation", delayed_update)

        def finish() -> object:
            finish_started.set()
            return runtime.operations.finish("succeeded", operation_id=operation.operation_id)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                expect_future = executor.submit(
                    runtime.operations.expect,
                    "decision",
                    operation_id=operation.operation_id,
                )
                assert update_started.wait(timeout=2)
                finish_future = executor.submit(finish)
                assert finish_started.wait(timeout=2)
                assert not finish_future.done()
                release_update.set()
                expect_future.result(timeout=2)
                finish_future.result(timeout=2)
        finally:
            release_update.set()

        stored = runtime.store.get_operation(operation.operation_id)
        assert stored is not None
        assert stored.state.value == "terminal"
        assert stored.outcome.value == "succeeded"
        assert "decision" in stored.expected_roles


def test_running_operation_is_interrupted_on_reopen_but_waiting_is_preserved(tmp_path) -> None:
    database = tmp_path / "operations.sqlite"
    runtime = Runtime.open(database)
    running = runtime.operations.start(kind="runtime", name="test.running", actor="host", pid=None)
    waiting = runtime.operations.start(kind="runtime", name="test.waiting", actor="host", pid=None)
    runtime.operations.wait(operation_id=waiting.operation_id)
    runtime.close()

    reopened = Runtime.open(database)
    try:
        assert reopened.store.get_operation(running.operation_id).outcome.value == "interrupted"
        assert reopened.store.get_operation(waiting.operation_id).state.value == "waiting"
    finally:
        reopened.close()


def test_running_operation_with_pending_provider_effect_recovers_as_unknown(tmp_path) -> None:
    database = tmp_path / "unknown-effect.sqlite"
    runtime = Runtime.open(database)
    operation = runtime.operations.start(kind="primitive", name="test.provider", actor="host", pid=None)
    with runtime.operations.attach(operation.operation_id):
        begin_external_effect_intent(
            runtime,
            pid="host",
            provider="test",
            operation="write",
            target="resource:test",
            state_mutation=True,
            information_flow=False,
        )
    runtime.close()

    reopened = Runtime.open(database)
    try:
        recovered = reopened.store.get_operation(operation.operation_id)
        assert recovered.outcome.value == "unknown"
        explanation = reopened.explain.explain_operation(operation.operation_id)
        assert any(item["reason"] == "provider_outcome_unknown" for item in explanation["uncertainties"])
    finally:
        reopened.close()


def test_operation_evidence_is_deduplicated_and_paginated() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="paginate")
        operation = runtime.operations.start(kind="runtime", name="test.pagination", actor=pid, pid=pid)
        first = runtime.operations.link_evidence("call", "call_1", OperationEvidenceRole.INVOCATION, operation_id=operation.operation_id)
        duplicate = runtime.operations.link_evidence("call", "call_1", OperationEvidenceRole.INVOCATION, operation_id=operation.operation_id)
        runtime.operations.link_evidence("event", "event_1", OperationEvidenceRole.EVENT, operation_id=operation.operation_id)
        runtime.operations.link_evidence("call", "call_1", OperationEvidenceRole.RESULT, operation_id=operation.operation_id)
        runtime.operations.finish("succeeded", operation_id=operation.operation_id)

        assert first is not None
        assert duplicate is None
        page = runtime.explain.explain_operation(operation.operation_id, evidence_limit=1)
        assert page["presentation_truncated"] is True
        assert page["next_cursor"] is not None
        assert len(page["evidence"]) == 1
        assert page["evidence"][0]["evidence_id"] == "call_1"
        assert page["evidence"][0]["roles"] == ["invocation", "result"]
        next_page = runtime.explain.explain_operation(
            operation.operation_id,
            evidence_limit=1,
            cursor=page["next_cursor"],
        )
        assert [item["evidence_id"] for item in next_page["evidence"]] == ["event_1"]


def test_unlinked_legacy_evidence_is_not_backfilled_by_pid_or_time() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="no heuristic backfill")
        legacy = runtime.audit.record(actor=pid, action="legacy.record", target=f"process:{pid}")
        operation = runtime.operations.start(
            kind="runtime",
            name="test.unlinked",
            actor=pid,
            pid=pid,
            expected_roles=["audit"],
        )
        runtime.operations.finish("succeeded", operation_id=operation.operation_id)

        explanation = runtime.explain.explain_operation(operation.operation_id)
        assert explanation["evidence_complete"] is False
        assert explanation["missing_evidence"] == [{"operation_id": operation.operation_id, "role": "audit"}]
        with pytest.raises(NotFound):
            runtime.explain.resolve("audit", legacy.record_id)


def test_context_manifest_storage_accepts_explicit_truncated_transform() -> None:
    with temporary_runtime() as runtime:
        manifest = ContextMaterializationManifest(
            materialization_id="ctxmat_truncated",
            pid="pid_test",
            view_id="view_test",
            policy="recency_first",
            budget_tokens=10,
            rendered_tokens=10,
            rendered_sha256="0" * 64,
            context_generation="initial",
            context_oid="oid_context",
            context_version=1,
            objects=[
                {
                    "oid": "oid_large",
                    "version": 1,
                    "type": "artifact",
                    "disposition": "included",
                    "reason": "selected",
                    "transform": "truncated",
                    "tokens": 10,
                    "rendered_sha256": "1" * 64,
                    "payload": "MANIFEST_PAYLOAD_MUST_NOT_LEAK",
                }
            ],
            compaction={},
            created_at="2026-07-10T00:00:00+00:00",
        )
        runtime.store.insert_context_materialization_manifest(manifest)

        stored = runtime.store.get_context_materialization_manifest(manifest.materialization_id)
        assert stored is not None
        assert stored.objects[0]["transform"] == "truncated"
        operation = runtime.operations.start(
            kind="llm_request",
            name="test.truncated_context",
            actor="pid_test",
            pid="pid_test",
            expected_roles=["context"],
        )
        runtime.operations.link_evidence(
            "context_manifest",
            manifest.materialization_id,
            "context",
            operation_id=operation.operation_id,
        )
        runtime.operations.finish("succeeded", operation_id=operation.operation_id)
        explanation = runtime.explain.resolve("context", manifest.materialization_id)
        assert "MANIFEST_PAYLOAD_MUST_NOT_LEAK" not in json.dumps(explanation)


def test_context_materialization_records_omission_reasons() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="context reasons")
        large = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"text": "x" * 2_000},
            metadata=ObjectMetadata(tags=["large"]),
        )
        budget_view = runtime.memory.create_view(pid, [large])
        budget = runtime.memory.materialize_context(pid, budget_view, budget_tokens=1, charge_resources=False)
        assert budget.object_manifest[0]["reason"] == "token_budget"

        filtered_handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"text": "filtered"},
            metadata=ObjectMetadata(tags=["other"]),
        )
        filtered_view = runtime.memory.create_view(
            pid,
            [filtered_handle],
            filters=[ObjectFilter(tags=["wanted"])],
        )
        filtered = runtime.memory.materialize_context(pid, filtered_view, charge_resources=False)
        assert filtered.object_manifest[0]["reason"] == "filter_mismatch"

        denied_handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"text": "denied"})
        denied_view = runtime.memory.create_view(pid, [denied_handle])
        runtime.capability.revoke(denied_handle.capability_id, revoked_by="test", require_authority=False)
        denied = runtime.memory.materialize_context(pid, denied_view, charge_resources=False)
        assert denied.object_manifest[0]["reason"] == "capability_denied"

        missing_handle = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"text": "missing"})
        missing_view = runtime.memory.create_view(pid, [missing_handle])
        assert runtime.store.delete_object(missing_handle.oid)
        missing = runtime.memory.materialize_context(pid, missing_view, charge_resources=False)
        assert missing.object_manifest[0]["reason"] == "missing"


def test_protected_boundary_registry_covers_core_mutation_surfaces() -> None:
    with temporary_runtime() as runtime:
        assert {
            "process.spawn",
            "process.exec",
            "process.wait",
            "memory.create_object",
            "memory.materialize_context",
            "checkpoint.restore",
            "object_task.start",
            "primitive.filesystem.write_text",
            "primitive.shell.run",
            "primitive.jsonrpc.call",
            "primitive.mcp.call",
            "human.request_permission",
            "capability.issue",
            "capability.derive_authority",
            "authority_manifest.bind",
            "skill.activate",
            "image.commit",
        } <= runtime.explainable_boundary_names
        assert "tool_group.activate" not in runtime.explainable_boundary_names
        assert {
            "runtime.git.repository_info",
            "runtime.git.status",
            "runtime.git.stage",
            "runtime.git.fetch",
            "runtime.git.push",
            "runtime.git.create_pull_request",
        } <= runtime.explainable_boundary_names

        parent = runtime.process.spawn(goal="direct fork attribution")
        runtime.process.fork(parent, "forked child")
        fork_root = next(
            operation
            for operation in runtime.store.list_operations(pid=parent, roots_only=True)
            if operation.name == "process.fork"
        )
        assert fork_root.actor == parent
        assert fork_root.pid == parent


def test_explain_is_host_only_not_a_model_tool_or_syscall() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="host only")
        assert all("explain" not in name for name in runtime.tools.model_tool_names(pid))
        from agent_libos.runtime.syscalls import BUILTIN_SYSCALL_NAMES

        assert all("explain" not in name for name in BUILTIN_SYSCALL_NAMES)
