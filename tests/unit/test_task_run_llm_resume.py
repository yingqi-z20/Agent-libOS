from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.payload_retention import (
    PayloadRetentionTier,
    external_effect_payload_retention_tier,
    llm_call_payload_retention_tier,
    llm_call_payload_sha256,
    redact_task_run_llm_calls,
    redact_terminal_task_run_llm_call,
    redact_terminal_task_run_llm_tool_output,
    retain_llm_call_payload,
    validate_terminal_task_run_llm_redaction,
)
from agent_libos.evidence.external_effects import record_external_effect
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.executor import LLMProcessExecutor
from agent_libos.llm.pending import PENDING_METADATA_FILTER_KEY
from agent_libos.llm.task_runs import (
    completed_outcome_manifest,
    normalize_task_run_prompt_context,
    normalize_validated_action_manifest,
    task_run_contract_message,
    task_run_dynamic_state_message,
    validated_action_manifest,
)
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequest,
    HumanRequestStatus,
    LLMCallRecord,
    PROMPT_MODE_IMAGE_ONLY,
    TaskRunRetention,
    TaskRunRequirementStatus,
    TaskRunSpecV1,
    TaskRunStatus,
    canonical_task_run_json,
    task_run_payload_sha256,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps, to_jsonable
from tests.support.fakes import RecordingActionClient


def _v2_prompt_context(
    requirements: list[dict[str, str]],
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": "run_internal_123",
        "context_generation": "generation_internal_123",
        "goal_text": "Preserve the business document field run_id.",
        "requirements": requirements,
        "transcript_messages": [],
        "compressed_summary": summary,
        "data_labels": {},
    }


def test_v2_task_run_contract_is_semantic_append_only_and_state_is_dynamic() -> None:
    first_requirement = {
        "requirement_id": "req_internal_1",
        "kind": "original_goal",
        "content_text": "Create the report.",
        "status": "pending",
    }
    follow_up = {
        "requirement_id": "req_internal_2",
        "kind": "human_follow_up",
        "content_text": "Keep the user payload field run_id unchanged.",
        "status": "pending",
    }
    initial = _v2_prompt_context([first_requirement])
    appended = _v2_prompt_context([first_requirement, follow_up])

    first = task_run_contract_message(
        initial,
        prompt_layout="cache_optimized_v2",
    )
    second = task_run_contract_message(
        appended,
        prompt_layout="cache_optimized_v2",
    )

    assert second.startswith(first + "\n")
    assert "Create the report." in second
    assert "Keep the user payload field run_id unchanged." in second
    for internal in (
        "run_internal_123",
        "generation_internal_123",
        "req_internal_1",
        "req_internal_2",
        "requirement_id",
        "schema_version",
    ):
        assert internal not in second

    satisfied = _v2_prompt_context(
        [{**first_requirement, "status": "satisfied"}, follow_up],
        summary="Report generated; follow-up pending.",
    )
    assert task_run_contract_message(
        satisfied,
        prompt_layout="cache_optimized_v2",
    ) == second
    dynamic = task_run_dynamic_state_message(satisfied)
    assert "Report generated; follow-up pending." in dynamic
    assert '"order": 1' in dynamic
    assert '"status": "satisfied"' in dynamic
    assert "req_internal_1" not in dynamic

    legacy = task_run_contract_message(initial, prompt_layout="legacy_v1")
    assert "run_internal_123" in legacy
    assert "req_internal_1" in legacy


def test_task_run_result_oid_scan_ignores_non_object_argument_sentinels() -> None:
    result = {
        "action": {
            "result_oid": "None",
            "payload": {"result_oid": "external-provider-result"},
        },
        "result": {
            "result_oid": "obj_runtime_result",
            "nested": {"result_oid": "obj_nested_result"},
        },
    }

    assert LLMProcessExecutor._task_run_result_oids(result) == (
        "obj_nested_result",
        "obj_runtime_result",
    )


class _CaptureTaskRunHook:
    def __init__(
        self,
        prompt_context: Mapping[str, Any] | None = None,
        pending_action: Mapping[str, Any] | None = None,
    ) -> None:
        self.prompt_context = prompt_context
        self.pending_action = pending_action
        self.events: list[tuple[str, str, dict[str, Any], str]] = []
        self.staged: list[tuple[str, str, dict[str, Any], str]] = []

    def prompt_context_for_pid(self, pid: str) -> Mapping[str, Any] | None:
        del pid
        return self.prompt_context

    def requirement_binding_for_prompt(
        self,
        pid: str,
        *,
        context_generation: str,
    ) -> Mapping[str, Any] | None:
        del pid, context_generation
        return None

    def record_validated_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        action_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        self.events.append(
            ("validated", call_id, dict(action_manifest), context_generation)
        )

    def record_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        self.events.append(
            ("completed", call_id, dict(outcome_manifest), context_generation)
        )

    def stage_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        self.staged.append(
            (pid, call_id, dict(outcome_manifest), context_generation)
        )

    def pending_validated_action_for_pid(
        self,
        pid: str,
    ) -> Mapping[str, Any] | None:
        del pid
        return self.pending_action


class _CaptureMessagesClient:
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del tools
        self.messages.append(json.loads(json.dumps(messages)))
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "local-exit",
                    "name": "process_exit",
                    "arguments": '{"payload":{"done":true}}',
                }
            ],
            api="responses",
            response_id="provider-response-must-not-be-required",
            request_id="provider-request-observability-only",
            usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )


class _ProviderMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del messages, tools
        self.calls += 1
        raise AssertionError("local TaskRun recovery called the LLM Provider")


def _prompt_context() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": "run_local_resume",
        "context_generation": "generation-7",
        "goal_text": "DURABLE_GOAL_SENTINEL",
        "requirements": [
            {
                "requirement_id": "requirement-1",
                "kind": "initial_goal",
                "content_text": "PRESERVE_REQUIREMENT_SENTINEL",
                "status": "pending",
            }
        ],
        "transcript_messages": [
            {"role": "assistant", "content": "LOCAL_TRANSCRIPT_SENTINEL"}
        ],
        "compressed_summary": "LOCAL_SUMMARY_SENTINEL",
        "data_labels": {
            "sensitivity": "normal",
            "trust_level": "user_asserted",
            "integrity": "checked",
            "origin": "local",
            "tenant": None,
            "principal": None,
            "declassification_authority": None,
        },
    }


def _consume_initial_requirement(
    runtime: Runtime,
    created: Any,
) -> dict[str, Any]:
    requirements = runtime.store.list_task_run_requirements(created.run_id)
    assert len(requirements) == 1
    consumed = runtime.store.update_task_run_requirement_cas(
        requirements[0].requirement_id,
        expected_status=TaskRunRequirementStatus.PENDING,
        status=TaskRunRequirementStatus.IN_PROGRESS,
        updated_at=created.updated_at,
        started_at=created.updated_at,
    )
    assert consumed is not None
    assert consumed.status is TaskRunRequirementStatus.IN_PROGRESS
    context = runtime.task_runs.prompt_context_for_pid(created.root_pid)
    assert context is not None
    binding = runtime.task_runs.requirement_binding_for_prompt(
        created.root_pid,
        context_generation=str(context["context_generation"]),
    )
    assert isinstance(binding, dict)
    return binding


def test_real_task_run_uses_local_llm_resume_hook_and_purges_terminal_io(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    client = _CaptureMessagesClient()
    runtime = Runtime.open(tmp_path / "task-run-llm.sqlite", config=config)
    try:
        runtime.llm.client = client
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal={"objective": "DURABLE_E2E_GOAL_SENTINEL"},
                display_title="Durable LLM E2E",
            ),
            client_request_id="create-durable-llm-e2e",
        )

        completed = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-durable-llm-e2e",
        )

        assert completed.status is TaskRunStatus.SUCCEEDED
        assert client.messages
        assert "DURABLE_E2E_GOAL_SENTINEL" in dumps(client.messages[0])
        calls = runtime.store.list_task_run_llm_calls(
            created.run_id,
            [created.root_pid],
        )
        assert calls
        assert all(
            llm_call_payload_retention_tier(call)
            is PayloadRetentionTier.HASH_ONLY
            for call in calls
        )
        assert "DURABLE_E2E_GOAL_SENTINEL" not in dumps(
            [to_jsonable(call) for call in calls]
        )
        outputs = runtime.store.list_task_run_llm_tool_outputs(
            created.run_id,
            [created.root_pid],
        )
        assert all("done" not in row["output_text"] for row in outputs)
        assert all(
            payload.canonical_json is None
            for payload in runtime.store.list_task_run_payloads(created.run_id)
        )
    finally:
        runtime.close()


def test_real_task_run_resumes_message_wait_across_runtime_reopen(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    target = tmp_path / "task-run-message-reopen.sqlite"
    first = Runtime.open(target, config=config)
    try:
        first.llm.client = RecordingActionClient(
            [{"action": "receive_process_messages"}]
        )
        first.register_image(
            AgentImage(
                image_id="task-run-message-reopen:v0",
                name="task-run-message-reopen",
                system_prompt="Wait for one message, then finish.",
                default_tools=["receive_process_messages", "process_exit"],
                context_policy="plan_first",
            ),
            actor="test.host",
        )
        created = first.task_runs.create(
            TaskRunSpecV1(
                goal="wait for a durable message, then finish",
                display_title="Durable message wait",
                image_id="task-run-message-reopen:v0",
            ),
            client_request_id="create-message-reopen",
        )
        waiting = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="wait-before-reopen",
        )

        assert waiting.status is TaskRunStatus.WAITING_MESSAGE
        point = first.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert point is not None and point.complete
        first_epoch = point.task_run_epoch
    finally:
        first.close()

    second = Runtime.open(target, config=config)
    try:
        second_client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"done": True}}]
        )
        second.llm.client = second_client
        recovered = second.task_runs.get(created.run_id)
        assert recovered.status is TaskRunStatus.WAITING_MESSAGE
        assert second.task_runs.runtime_epoch > first_epoch
        second.messages.post(
            sender="human:test",
            recipient_pid=created.root_pid,
            subject="durable resume",
            payload={"ready": True},
        )

        completed = second.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=recovered.revision,
            command_id="finish-after-reopen",
        )

        assert completed.status is TaskRunStatus.SUCCEEDED
        assert len(second_client.user_prompts) == 1
    finally:
        second.close()


def test_task_run_certifies_compaction_generation_after_durable_child_wait(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
        llm_context=replace(
            DEFAULT_CONFIG.llm_context,
            policy="llm_context_object",
        ),
    )
    runtime = Runtime.open(tmp_path / "task-run-context-compaction.sqlite", config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="compact the local context, then finish",
                display_title="Durable context compaction",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-durable-context-compaction",
        )
        assert created.root_pid is not None
        runtime.capability.grant(
            created.root_pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.capability.grant(
            created.root_pid,
            "image:context-compressor:v0",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            created.root_pid,
            "agent-libos-runtime-session",
            actor=created.root_pid,
        )
        compact_summary = {
            "goal": "continue after locally certified compaction",
            "constraints": ["preserve durable bindings"],
            "user_preferences": [],
            "completed": [],
            "pending": ["finish the TaskRun"],
            "key_references": {},
            "recent_decisions": [],
            "risks": [],
            "uncertainties": [],
            "next_steps": ["resume the caller"],
        }
        client = RecordingActionClient(
            [
                {
                    "action": "compact_process_context",
                    "force": True,
                    "target_tokens": 512,
                    "max_chunks": 1,
                    "preserve_recent_entries": 1,
                },
                {"action": "process_exit", "payload": compact_summary},
                {"action": "process_exit", "payload": {"done": True}},
            ]
        )
        runtime.llm.client = client

        completed = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-durable-context-compaction",
            max_quanta=10,
        )

        assert completed.status is TaskRunStatus.SUCCEEDED
        current_generation = runtime.store.get_llm_context_generation(
            created.root_pid
        )
        point = runtime.store.get_task_run_resume_point(created.root_pid)
        assert point is not None
        assert point.context_generation == current_generation
        completed_wrappers = []
        for payload in runtime.store.list_task_run_payloads(created.run_id):
            if payload.role != "pending_action" or payload.canonical_json is None:
                continue
            value = json.loads(payload.canonical_json)
            if value.get("kind") == "completed_outcome":
                completed_wrappers.append(value)
        staged = [
            value
            for value in completed_wrappers
            if value.get("generation_transition") is not None
        ]
        assert len(staged) == 1
        wrapper = staged[0]
        transition = wrapper["generation_transition"]
        assert wrapper["source_context_generation"] != current_generation
        assert wrapper["result_context_generation"] == current_generation
        assert transition == {
            "schema_version": 1,
            "kind": "certified_context_compaction",
            "source_context_generation": wrapper["source_context_generation"],
            "result_context_generation": current_generation,
            "compaction_sha256": transition["compaction_sha256"],
        }
        assert len(transition["compaction_sha256"]) == 64
        assert all(
            value.get("generation_transition") is None
            for value in completed_wrappers
            if value != wrapper
        )
    finally:
        runtime.close()


def test_real_task_run_replays_committed_action_locally_after_reopen(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    target = tmp_path / "task-run-validated-action-reopen.sqlite"
    first = Runtime.open(target, config=config)
    try:
        created = first.task_runs.create(
            TaskRunSpecV1(
                goal="finish from the locally committed action",
                display_title="Local action replay",
            ),
            client_request_id="create-local-action-replay",
        )
        assert created.root_pid is not None
        requirement_binding = _consume_initial_requirement(first, created)
        call_id = "llmcall-real-local-action-replay"
        llm_record = _full_llm_record(call_id)
        first.store.insert_llm_call(
            replace(
                llm_record,
                pid=created.root_pid,
                image_id="base-agent:v0",
                response_content="",
                tool_calls=[],
                request_options={
                    **llm_record.request_options,
                    "task_run_requirement_binding_v1": requirement_binding,
                },
                raw_response=None,
            )
        )
        manifest = validated_action_manifest(
            [{"action": "process_exit", "payload": {"recovered": True}}],
            call_id=call_id,
            parallel_tool_calls=False,
            host_auto_wait=False,
            tool_call_count=1,
            data_labels=_prompt_context()["data_labels"],
        )
        first.task_runs.record_validated_transcript(
            pid=created.root_pid,
            call_id=call_id,
            action_manifest=manifest,
            context_generation=first.store.get_llm_context_generation(
                created.root_pid
            ),
        )
        point = first.store.get_task_run_resume_point(created.root_pid)
        assert point is not None and point.pending_action_payload_id is not None
    finally:
        first.close()

    second = Runtime.open(target, config=config)
    try:
        provider = _ProviderMustNotRun()
        second.llm.client = provider
        recovered = second.task_runs.get(created.run_id)

        completed = second.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=recovered.revision,
            command_id="dispatch-local-action-after-reopen",
        )

        assert completed.status is TaskRunStatus.SUCCEEDED
        assert provider.calls == 0
        assert second.process.get(created.root_pid).status.value == "exited"
        assert any(
            record.action == "llm.task_run_validated_action_recovered"
            for record in second.audit.trace()
        )
    finally:
        second.close()


def test_permanent_task_run_explicit_purge_redacts_linked_external_effect(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    runtime = Runtime.open(tmp_path / "task-run-effect-purge.sqlite", config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="finish after a locally committed safe point",
                display_title="External effect purge",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-effect-purge",
        )
        assert created.root_pid is not None
        requirement_binding = _consume_initial_requirement(runtime, created)
        requirements = runtime.store.list_task_run_requirements(created.run_id)
        assert len(requirements) == 1
        assert requirements[0].status.value == "in_progress"
        human_request = HumanRequest(
            request_id="hreq-effect-purge",
            pid=created.root_pid,
            human="host",
            payload={
                "type": "question",
                "question": "PERMANENT_HUMAN_PROMPT_SENTINEL",
            },
            status=HumanRequestStatus.APPROVED,
            decision={"answer": "PERMANENT_HUMAN_ANSWER_SENTINEL"},
            blocking=True,
            created_at="2026-07-31T00:00:00+00:00",
            updated_at="2026-07-31T00:00:01+00:00",
        )
        runtime.store.insert_human_request(human_request)
        effect = record_external_effect(
            runtime.uow.protected_effects,
            pid=created.root_pid,
            provider="test-provider",
            operation="write",
            target="record:effect-purge",
            classification=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=False,
            ),
            audit_record=None,
            event=None,
            metadata={
                "provider_receipt": {"secret": "EFFECT_RECEIPT_SENTINEL"},
                "sensitive_result": "EFFECT_RESULT_SENTINEL",
            },
        )
        call_id = "llmcall-effect-purge"
        llm_record = _full_llm_record(call_id)
        runtime.store.insert_llm_call(
            replace(
                llm_record,
                pid=created.root_pid,
                image_id="base-agent:v0",
                response_content="",
                tool_calls=[],
                request_options={
                    **llm_record.request_options,
                    "task_run_requirement_binding_v1": requirement_binding,
                },
                raw_response=None,
            )
        )
        runtime.task_runs.record_validated_transcript(
            pid=created.root_pid,
            call_id=call_id,
            action_manifest=validated_action_manifest(
                [{"action": "process_exit", "payload": {"done": True}}],
                call_id=call_id,
                parallel_tool_calls=False,
                host_auto_wait=False,
                tool_call_count=1,
                data_labels=_prompt_context()["data_labels"],
            ),
            context_generation=runtime.store.get_llm_context_generation(
                created.root_pid
            ),
        )
        provider = _ProviderMustNotRun()
        runtime.llm.client = provider
        ready = runtime.task_runs.get(created.run_id)

        completed = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=ready.revision,
            command_id="finish-effect-purge",
        )

        assert completed.status is TaskRunStatus.SUCCEEDED
        assert provider.calls == 0
        before = runtime.store.get_external_effect(effect.effect_id)
        assert before is not None
        assert (
            external_effect_payload_retention_tier(before)
            is PayloadRetentionTier.FULL
        )
        assert [
            item.effect_id
            for item in runtime.store.list_task_run_external_effects(
                created.run_id,
                [created.root_pid],
            )
        ] == [effect.effect_id]
        before_human = runtime.store.get_human_request(human_request.request_id)
        assert before_human is not None
        assert "PERMANENT_HUMAN_PROMPT_SENTINEL" in dumps(before_human.payload)
        assert "PERMANENT_HUMAN_ANSWER_SENTINEL" in dumps(before_human.decision)

        purged = runtime.task_runs.purge_payloads(
            created.run_id,
            expected_revision=completed.revision,
            command_id="purge-effect-payload",
        )
        replayed_purge = runtime.task_runs.purge_payloads(
            created.run_id,
            expected_revision=completed.revision,
            command_id="purge-effect-payload",
        )

        after = runtime.store.get_external_effect(effect.effect_id)
        after_human = runtime.store.get_human_request(human_request.request_id)
        assert after is not None
        assert after_human is not None
        assert purged.status is TaskRunStatus.SUCCEEDED
        assert replayed_purge == purged
        assert (
            external_effect_payload_retention_tier(after)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert "EFFECT_RECEIPT_SENTINEL" not in dumps(to_jsonable(after))
        assert "EFFECT_RESULT_SENTINEL" not in dumps(to_jsonable(after))
        assert "PERMANENT_HUMAN_PROMPT_SENTINEL" not in dumps(
            to_jsonable(after_human)
        )
        assert "PERMANENT_HUMAN_ANSWER_SENTINEL" not in dumps(
            to_jsonable(after_human)
        )
        assert after_human.payload[
            "$agent_libos_task_run_human_redaction"
        ]["request_type"] == "question"
        assert (after.effect_id, after.pid, after.effect_state, after.transaction_state) == (
            before.effect_id,
            before.pid,
            before.effect_state,
            before.transaction_state,
        )
    finally:
        runtime.close()


def test_task_run_manifest_is_local_canonical_and_provider_independent() -> None:
    actions = [{"action": "echo", "payload": {"value": "before"}}]
    validated = validated_action_manifest(
        actions,
        call_id="llmcall-local-manifest",
        parallel_tool_calls=False,
        host_auto_wait=False,
        tool_call_count=1,
        data_labels=_prompt_context()["data_labels"],
    )
    actions[0]["payload"]["value"] = "after"

    assert validated["actions"][0]["payload"]["value"] == "before"
    assert validated["previous_response_id_used"] is False
    assert normalize_validated_action_manifest(validated) == validated
    completed = completed_outcome_manifest(
        state="completed",
        paired_outputs_persisted=True,
        data_labels=_prompt_context()["data_labels"],
        result={"ok": True, "result": {"value": 1}},
    )
    assert completed["previous_response_id_used"] is False
    assert completed["paired_outputs_persisted"] is True
    waiting = completed_outcome_manifest(
        state="waiting",
        paired_outputs_persisted=True,
        data_labels=_prompt_context()["data_labels"],
        durable_wait={"wait_type": "message", "filters": {}},
    )
    assert waiting["durable_wait"]["wait_type"] == "message"
    with pytest.raises(ValueError, match="wait type"):
        completed_outcome_manifest(
            state="waiting",
            paired_outputs_persisted=True,
            data_labels=_prompt_context()["data_labels"],
            durable_wait={"type": "message"},
        )


def test_task_run_prompt_context_is_strict_and_bounded() -> None:
    selected = normalize_task_run_prompt_context(_prompt_context())
    assert selected["run_id"] == "run_local_resume"

    invalid = {**_prompt_context(), "provider_response_id": "remote-state"}
    with pytest.raises(ValueError, match="invalid shape"):
        normalize_task_run_prompt_context(invalid)

    duplicate = _prompt_context()
    duplicate["requirements"] = [
        duplicate["requirements"][0],
        dict(duplicate["requirements"][0]),
    ]
    with pytest.raises(ValueError, match="duplicate ids"):
        normalize_task_run_prompt_context(duplicate)


def test_task_run_prompt_binds_follow_up_role_hash_and_labels(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    runtime = Runtime.open(tmp_path / "follow-up-prompt-binding.sqlite", config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="original durable goal",
                display_title="Follow-up prompt binding",
            ),
            client_request_id="create-follow-up-prompt-binding",
        )
        followed = runtime.task_runs.follow_up(
            created.run_id,
            {"instruction": "FOLLOW_UP_PROMPT_SENTINEL"},
            expected_revision=created.revision,
            command_id="add-follow-up-prompt-binding",
        )
        requirements = runtime.store.list_task_run_requirements(created.run_id)
        follow_up = next(
            item for item in requirements if item.kind.value == "follow_up"
        )
        payload = runtime.store.get_task_run_payload(follow_up.payload_id)

        assert followed.requirement_count == 2
        assert payload is not None
        assert payload.run_id == created.run_id
        assert payload.role == "follow_up"
        assert payload.sha256 == follow_up.requirement_sha256
        decoded = json.loads(payload.canonical_json or "null")
        assert decoded["data_labels"]["origin"] == "host"
        context = runtime.task_runs.prompt_context_for_pid(created.root_pid)
        assert context is not None
        assert any(
            "FOLLOW_UP_PROMPT_SENTINEL" in item["content_text"]
            and item["requirement_id"] == follow_up.requirement_id
            for item in context["requirements"]
        )
        assert context["data_labels"]["origin"] == "host"
    finally:
        runtime.close()


@pytest.mark.parametrize("tamper", ["run_id", "role", "hash", "missing_labels"])
def test_task_run_prompt_rejects_tampered_requirement_payload_binding(
    tmp_path: Path,
    tamper: str,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    database = tmp_path / f"requirement-tamper-{tamper}.sqlite"
    runtime = Runtime.open(database, config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="integrity-bound initial requirement",
                display_title="Requirement integrity binding",
            ),
            client_request_id=f"create-requirement-tamper-{tamper}",
        )
        assert created.root_pid is not None
        initial = runtime.store.list_task_run_requirements(created.run_id)[0]
        goal_payload = runtime.store.get_task_run_payload(initial.payload_id)
        assert goal_payload is not None
        replacement_payload = None
        if tamper == "run_id":
            other = runtime.task_runs.create(
                TaskRunSpecV1(
                    goal="payload owned by another run",
                    display_title="Other payload owner",
                ),
                client_request_id="create-other-requirement-owner",
            )
            other_initial = runtime.store.list_task_run_requirements(other.run_id)[0]
            replacement_payload = runtime.store.get_task_run_payload(
                other_initial.payload_id
            )
        elif tamper == "role":
            followed = runtime.task_runs.follow_up(
                created.run_id,
                "payload has the follow_up role",
                expected_revision=created.revision,
                command_id="add-role-tamper-follow-up",
            )
            assert followed.requirement_count == 2
            follow_up = runtime.store.list_task_run_requirements(created.run_id)[1]
            replacement_payload = runtime.store.get_task_run_payload(
                follow_up.payload_id
            )
        replacement_payload_id = (
            replacement_payload.payload_id
            if replacement_payload is not None
            else None
        )
        replacement_sha256 = (
            replacement_payload.sha256
            if replacement_payload is not None
            else None
        )
    finally:
        runtime.close()

    connection = sqlite3.connect(database)
    try:
        if tamper in {"run_id", "role"}:
            assert replacement_payload_id is not None
            assert replacement_sha256 is not None
            connection.execute(
                "UPDATE task_run_requirements SET payload_id = ?, "
                "requirement_sha256 = ? WHERE requirement_id = ?",
                (
                    replacement_payload_id,
                    replacement_sha256,
                    initial.requirement_id,
                ),
            )
        elif tamper == "hash":
            connection.execute(
                "UPDATE task_run_requirements SET requirement_sha256 = ? "
                "WHERE requirement_id = ?",
                ("0" * 64, initial.requirement_id),
            )
        else:
            canonical = canonical_task_run_json(
                {"goal": "integrity-bound initial requirement"}
            )
            sha256 = task_run_payload_sha256(canonical)
            connection.execute(
                "UPDATE task_run_payloads SET canonical_json = ?, sha256 = ?, "
                "size_bytes = ? WHERE payload_id = ?",
                (
                    canonical,
                    sha256,
                    len(canonical.encode("utf-8")),
                    goal_payload.payload_id,
                ),
            )
            connection.execute(
                "UPDATE task_run_requirements SET requirement_sha256 = ? "
                "WHERE requirement_id = ?",
                (sha256, initial.requirement_id),
            )
        connection.commit()
    finally:
        connection.close()

    reopened = Runtime.open(database, config=config)
    try:
        with pytest.raises(ValidationError):
            reopened.task_runs.prompt_context_for_pid(created.root_pid)
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert "payload_corrupt" in {item["kind"] for item in summary.blockers}
        assert reopened.process.get(created.root_pid).resource_usage.llm_calls == 0
    finally:
        reopened.close()


def test_task_run_semantic_compaction_cannot_drop_later_turns_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
            payload_max_bytes=4_096,
        ),
    )
    runtime = Runtime.open(tmp_path / "single-use-compaction.sqlite", config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="keep every fact not covered by the semantic summary",
                display_title="Single-use compaction",
            ),
            client_request_id="create-single-use-compaction",
        )
        assert created.root_pid is not None
        call_id = "llmcall-single-use-compaction"
        runtime.store.insert_llm_call(
            replace(
                _full_llm_record(call_id),
                pid=created.root_pid,
                image_id="base-agent:v0",
                response_content="",
                tool_calls=[],
                raw_response=None,
            )
        )
        generation = runtime.store.get_llm_context_generation(created.root_pid)
        runtime.task_runs.record_validated_transcript(
            pid=created.root_pid,
            call_id=call_id,
            action_manifest=validated_action_manifest(
                [{"action": "process_exit", "payload": {"done": True}}],
                call_id=call_id,
                parallel_tool_calls=False,
                host_auto_wait=False,
                tool_call_count=1,
                data_labels=_prompt_context()["data_labels"],
            ),
            context_generation=generation,
        )
        prior = runtime.store.get_task_run_resume_point(created.root_pid)
        assert prior is not None and prior.summary_payload_id is None
        semantic_summary = {
            "goal": "keep the early fact",
            "constraints": [],
            "user_preferences": [],
            "completed": [],
            "pending": ["continue"],
            "key_references": {},
            "recent_decisions": [],
            "risks": [],
            "uncertainties": [],
            "next_steps": ["finish"],
        }
        summary_sha256 = hashlib.sha256(
            json.dumps(
                semantic_summary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        monkeypatch.setattr(
            runtime.llm.context_memory,
            "latest_validated_compaction",
            lambda pid: {
                "schema_version": 1,
                "context_oid": "obj-context",
                "context_version": 2,
                "context_generation": generation,
                "compacted_at": generation,
                "source_version": 1,
                "source_entry_count": 4,
                "summary": semantic_summary,
                "data_labels": _prompt_context()["data_labels"],
                "summary_sha256": summary_sha256,
            },
        )
        messages = [
            {"role": "assistant", "content": "early-a:" + "a" * 3_000},
            {"role": "assistant", "content": "early-b:" + "b" * 3_000},
            {"role": "assistant", "content": "new-action-result"},
        ]

        first = runtime.task_runs._bounded_transcript_projection(
            created.run_id,
            created.root_pid,
            messages,
            prior=prior,
            created_at="2026-07-31T00:00:00+00:00",
            context_generation=generation,
            new_message_count=1,
        )

        assert first is not None
        retained, summary_payload = first
        assert summary_payload is not None
        assert retained[-1]["content"] == "new-action-result"
        runtime.store.insert_task_run_payload(summary_payload)
        consumed = replace(
            prior,
            summary_payload_id=summary_payload.payload_id,
        )

        assert runtime.task_runs._bounded_transcript_projection(
            created.run_id,
            created.root_pid,
            messages,
            prior=consumed,
            created_at="2026-07-31T00:00:01+00:00",
            context_generation=generation,
            new_message_count=1,
        ) is None
    finally:
        runtime.close()


def test_executor_injects_local_task_run_contract_and_safe_point_order() -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            prompt_layout="cache_optimized_v2",
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        hook = _CaptureTaskRunHook(_prompt_context())
        runtime.llm._task_runs = hook
        client = _CaptureMessagesClient()
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="runtime goal marker only",
        )

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        assert [item[0] for item in hook.events] == ["validated", "completed"]
        assert hook.events[0][1] == hook.events[1][1]
        assert hook.events[0][2]["previous_response_id_used"] is False
        assert hook.events[1][2]["state"] == "completed"
        assert hook.events[1][2]["result"] == result
        assert hook.events[1][2]["paired_outputs_persisted"] is True
        assert hook.events[1][2]["data_labels"]["sensitivity"] == "normal"
        assert hook.events[1][2]["data_labels"]["origin"] == "derived"
        assert runtime.store.get_llm_call(hook.events[0][1]) is not None

        prompt = client.messages[0]
        assert [message["role"] for message in prompt] == [
            "system",
            "user",
            "assistant",
            "user",
            "user",
        ]
        rendered = dumps(prompt)
        assert "DURABLE_GOAL_SENTINEL" in rendered
        assert "PRESERVE_REQUIREMENT_SENTINEL" in rendered
        assert "LOCAL_TRANSCRIPT_SENTINEL" in rendered
        assert "LOCAL_SUMMARY_SENTINEL" in rendered
        assert "Ordered requirements (append-only)" in prompt[1]["content"]
        assert prompt[1]["_agent_libos_cache_stable"] is True
        assert "Current TaskRun state (dynamic)" in prompt[3]["content"]
        assert "run_local_resume" not in rendered
        assert "requirement-1" not in rendered
        assert "provider-response-must-not-be-required" not in prompt[1]["content"]
    finally:
        runtime.close()


def test_invalid_action_repair_never_publishes_a_safe_point() -> None:
    runtime = Runtime.open("local")
    try:
        hook = _CaptureTaskRunHook()
        runtime.llm._task_runs = hook
        client = RecordingActionClient(
            [
                {"action": "tool_that_does_not_exist"},
                {"action": "process_exit", "payload": {"done": True}},
            ]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(image="base-agent:v0", goal="repair locally")

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        assert len(client.user_prompts) == 2
        assert [item[0] for item in hook.events] == ["validated", "completed"]
        calls = runtime.store.list_llm_calls(pid=pid)
        assert len(calls) == 2
        assert hook.events[0][1] == calls[-1].call_id
    finally:
        runtime.close()


def test_integrity_checked_pending_action_replays_locally_without_provider() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="resume a locally validated action",
        )
        call_id = "llmcall-local-action-recovery"
        runtime.store.insert_llm_call(
            replace(
                _full_llm_record(call_id),
                pid=pid,
                response_content="",
                tool_calls=[],
                raw_response=None,
            )
        )
        manifest = validated_action_manifest(
            [{"action": "process_exit", "payload": {"recovered": True}}],
            call_id=call_id,
            parallel_tool_calls=False,
            host_auto_wait=False,
            tool_call_count=1,
            data_labels=_prompt_context()["data_labels"],
        )
        hook = _CaptureTaskRunHook(pending_action=manifest)
        runtime.llm._task_runs = hook
        provider = _ProviderMustNotRun()
        runtime.llm.client = provider

        result = runtime.run_process_once(pid)

        assert result["ok"] is True
        assert runtime.process.get(pid).status.value == "exited"
        assert result["action"]["payload"] == {"recovered": True}
        assert provider.calls == 0
        assert [event[0] for event in hook.events] == ["completed"]
        assert len(hook.staged) == 1
        assert any(
            record.action == "llm.task_run_validated_action_recovered"
            and record.target == f"llm_call:{call_id}"
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_corrupt_non_run_pending_action_remains_startup_fatal(
    tmp_path: Path,
) -> None:
    target = tmp_path / "corrupt-ordinary-pending.sqlite"
    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="ordinary pending corruption remains fatal",
        )
        runtime.store.upsert_llm_pending_action(
            pid,
            {
                "resume_token": "ordinary-corrupt-token",
                "wait_type": "message",
                "filters": {},
                "action": {"action": "receive_process_messages"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "",
                "tool_call_count": 1,
                "status": "pending",
            },
        )
    finally:
        runtime.close()
    with sqlite3.connect(target) as connection:
        connection.execute(
            "UPDATE llm_pending_actions SET action_json = ? WHERE pid = ?",
            ("[]", pid),
        )
        connection.commit()

    with pytest.raises(ValidationError, match="invalid persisted pending LLM action"):
        Runtime.open(target)


def test_corrupt_task_run_pending_action_isolated_to_needs_attention(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    target = tmp_path / "corrupt-task-run-pending.sqlite"
    runtime = Runtime.open(target, config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="isolate corrupt pending action",
                display_title="Corrupt pending isolation",
            ),
            client_request_id="create-corrupt-pending-isolation",
        )
        assert created.root_pid is not None
        runtime.store.upsert_llm_pending_action(
            created.root_pid,
            {
                "resume_token": "task-run-corrupt-token",
                "wait_type": "message",
                "filters": {},
                "action": {"action": "receive_process_messages"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "",
                "tool_call_count": 1,
                "status": "pending",
            },
        )
    finally:
        runtime.close()
    with sqlite3.connect(target) as connection:
        connection.execute(
            "UPDATE llm_pending_actions SET action_json = ? WHERE pid = ?",
            ("[]", created.root_pid),
        )
        connection.commit()

    reopened = Runtime.open(target, config=config)
    try:
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert any(
            blocker.get("kind") == "pending_action_unreplayable"
            for blocker in summary.blockers
        )
        assert any(
            record.action
            == "llm.pending_action_corrupt_deferred_to_task_run"
            and record.target == f"process:{created.root_pid}"
            for record in reopened.audit.trace()
        )
        assert not reopened.llm.pending.has_memory(created.root_pid, "message")
    finally:
        reopened.close()


def test_legacy_context_marker_is_never_trusted_as_task_run_resume_state(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    target = tmp_path / "task-run-legacy-context-marker.sqlite"
    runtime = Runtime.open(target, config=config)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="do not replay an unbound legacy marker",
                display_title="Strict TaskRun pending action",
            ),
            client_request_id="create-task-run-legacy-context-marker",
        )
        assert created.root_pid is not None
        runtime.store.upsert_llm_pending_action(
            created.root_pid,
            {
                "resume_token": "legacy-context-marker-token",
                "wait_type": "context_management",
                "filters": {
                    PENDING_METADATA_FILTER_KEY: {
                        "kind": "context_management_auto",
                        "schema_version": 1,
                        "source": "runtime_context_management",
                        "outcome": "attempted",
                    }
                },
                "action": {
                    "action": "compact_process_context",
                    "force": True,
                },
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "",
                "tool_call_count": 0,
                "status": "completed",
            },
        )
    finally:
        runtime.close()

    reopened = Runtime.open(target, config=config)
    try:
        summary = reopened.task_runs.get(created.run_id)

        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert "run" not in summary.allowed_actions
        assert "resume" not in summary.allowed_actions
        assert any(
            blocker.get("kind") == "pending_action_unreplayable"
            for blocker in summary.blockers
        )
        assert reopened.store.list_llm_calls(pid=created.root_pid) == []
        assert any(
            record.action
            == "llm.pending_action_corrupt_deferred_to_task_run"
            and record.target == f"process:{created.root_pid}"
            for record in reopened.audit.trace()
        )
    finally:
        reopened.close()


def test_exact_message_wait_is_a_safe_point_and_completes_same_transcript() -> None:
    runtime = Runtime.open("local")
    try:
        runtime.register_image(
            AgentImage(
                image_id="task-run-wait:v0",
                name="task-run-wait",
                system_prompt="Wait once, then finish.",
                prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                default_tools=["receive_process_messages", "process_exit"],
            ),
            actor="test",
        )
        hook = _CaptureTaskRunHook()
        runtime.llm._task_runs = hook
        client = RecordingActionClient(
            [
                {"action": "receive_process_messages"},
                {"action": "process_exit", "payload": {"done": True}},
            ]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="task-run-wait:v0",
            goal="wait for explicit input",
        )

        waiting = runtime.run_process_once(pid)
        assert waiting["waiting_message"]
        assert [item[0] for item in hook.events] == ["validated", "completed"]
        first_call_id = hook.events[0][1]
        wait_manifest = hook.events[1][2]
        assert wait_manifest["state"] == "waiting"
        assert wait_manifest["paired_outputs_persisted"] is True
        assert wait_manifest["durable_wait"]["wait_type"] == "message"
        assert runtime.store.get_llm_pending_action(pid) is not None

        runtime.messages.post(
            sender="human:test",
            recipient_pid=pid,
            subject="resume",
            payload={"ready": True},
        )
        resumed = runtime.run_process_once(pid)

        assert resumed["ok"] and resumed["resumed_after_message"]
        assert hook.events[2][0] == "completed"
        assert hook.events[2][1] == first_call_id
        assert hook.events[2][2]["state"] == "completed"
    finally:
        runtime.close()


def test_wait_safe_point_keeps_local_call_id_across_runtime_reopen(
    tmp_path: Path,
) -> None:
    target = tmp_path / "task-run-wait.sqlite"
    hook = _CaptureTaskRunHook()
    runtime = Runtime.open(target)
    try:
        runtime.register_image(
            AgentImage(
                image_id="task-run-reopen-wait:v0",
                name="task-run-reopen-wait",
                system_prompt="Wait durably once.",
                prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                default_tools=["receive_process_messages", "process_exit"],
            ),
            actor="test",
        )
        runtime.llm._task_runs = hook
        runtime.llm.client = RecordingActionClient(
            [{"action": "receive_process_messages"}]
        )
        pid = runtime.process.spawn(
            image="task-run-reopen-wait:v0",
            goal="wait across reopen",
        )
        waiting = runtime.run_process_once(pid)
        assert waiting["waiting_message"]
        call_id = hook.events[-1][1]
    finally:
        runtime.close()

    reopened = Runtime.open(target)
    try:
        reopened.llm._task_runs = hook
        reopened.messages.post(
            sender="human:test",
            recipient_pid=pid,
            subject="resume after reopen",
            payload={"ready": True},
        )
        resumed = reopened.run_process_once(pid)

        assert resumed["ok"] and resumed["resumed_after_message"]
        assert hook.events[-1][0] == "completed"
        assert hook.events[-1][1] == call_id
        assert hook.events[-1][2]["state"] == "completed"
    finally:
        reopened.close()


def _full_llm_record(call_id: str = "llmcall-redact") -> LLMCallRecord:
    return LLMCallRecord(
        call_id=call_id,
        pid="pid-redact",
        image_id="image:v0",
        purpose="action_selection",
        status="ok",
        api="responses",
        model="model-observability",
        request_id="request-observability",
        response_id="response-observability",
        messages=[{"role": "user", "content": "FULL_IO_SECRET"}],
        tools=[{"name": "secret-tool", "description": "FULL_IO_SECRET"}],
        request_options={"safe_option": True},
        response_content="FULL_IO_SECRET",
        tool_calls=[{"name": "secret-tool", "arguments": "FULL_IO_SECRET"}],
        reasoning={"text": "FULL_IO_SECRET"},
        usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
        raw_response={"secret": "FULL_IO_SECRET"},
        observability={"preview": "FULL_IO_SECRET", "bytes": 14},
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
    )


class _RedactionStore:
    def __init__(
        self,
        records: list[LLMCallRecord],
        *,
        outputs: list[dict[str, Any]] | None = None,
        conflict: bool = False,
    ) -> None:
        self.records = records
        self.outputs = list(outputs or [])
        self.conflict = conflict
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def list_task_run_llm_calls(
        self,
        run_id: str,
        linked_pids: tuple[str, ...] = (),
    ) -> list[LLMCallRecord]:
        self.calls.append((run_id, linked_pids))
        return list(self.records)

    def redact_task_run_llm_call_payload(
        self,
        record: LLMCallRecord,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_tier: str,
    ) -> bool:
        del run_id, expected_payload_sha256, expected_tier
        if self.conflict:
            return False
        self.records = [
            record if current.call_id == record.call_id else current
            for current in self.records
        ]
        return True

    def list_task_run_llm_tool_outputs(
        self,
        run_id: str,
        linked_pids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        assert run_id == "run-redact"
        assert linked_pids == ("pid-redact",)
        return [dict(row) for row in self.outputs]

    def redact_task_run_llm_tool_output(
        self,
        *,
        run_id: str,
        pid: str,
        response_id: str,
        call_id: str,
        expected_output_sha256: str,
        redacted_output: str,
    ) -> bool:
        assert run_id == "run-redact"
        if self.conflict:
            return False
        for row in self.outputs:
            if (row["pid"], row["response_id"], row["call_id"]) == (
                pid,
                response_id,
                call_id,
            ):
                actual_sha256 = hashlib.sha256(
                    row["output_text"].encode("utf-8")
                ).hexdigest()
                if actual_sha256 != expected_output_sha256:
                    return False
                row["output_text"] = redacted_output
                return True
        return False


def test_terminal_task_run_redaction_preserves_hash_usage_and_observability_ids() -> None:
    original = _full_llm_record()
    original_sha256 = llm_call_payload_sha256(original)

    redacted = redact_terminal_task_run_llm_call(original)

    assert llm_call_payload_retention_tier(redacted) is PayloadRetentionTier.HASH_ONLY
    assert llm_call_payload_sha256(redacted) == original_sha256
    assert redacted.usage == original.usage
    assert redacted.request_id == original.request_id
    assert redacted.response_id == original.response_id
    assert redacted.model == original.model
    assert "FULL_IO_SECRET" not in dumps(to_jsonable(redacted))

    store = _RedactionStore(
        [original],
        outputs=[
            {
                "pid": "pid-redact",
                "response_id": "response-observability",
                "call_id": "tool-call-redact",
                "output_text": '{"secret":"FULL_IO_SECRET"}',
            }
        ],
    )
    assert redact_task_run_llm_calls(
        store,
        "run-redact",
        ["pid-redact"],
    ) == 2
    assert store.calls == [("run-redact", ("pid-redact",))]
    assert llm_call_payload_retention_tier(
        store.records[0]
    ) is PayloadRetentionTier.HASH_ONLY
    assert "FULL_IO_SECRET" not in store.outputs[0]["output_text"]
    redacted_output = store.outputs[0]["output_text"]
    assert redact_task_run_llm_calls(
        store,
        "run-redact",
        ["pid-redact"],
    ) == 0
    assert store.outputs[0]["output_text"] == redacted_output


def test_terminal_task_run_redaction_accepts_canonical_summary_source() -> None:
    original = _full_llm_record("llmcall-summary-redact")
    summary = retain_llm_call_payload(
        original,
        PayloadRetentionTier.SUMMARY,
        provider_chain_head=False,
    )

    redacted = redact_terminal_task_run_llm_call(summary)

    assert llm_call_payload_retention_tier(redacted) is PayloadRetentionTier.HASH_ONLY
    assert llm_call_payload_sha256(redacted) == llm_call_payload_sha256(original)
    assert redacted.usage == original.usage


def test_terminal_task_run_redaction_validator_requires_exact_projection() -> None:
    original = _full_llm_record("llmcall-validate-redact")
    expected_sha256 = llm_call_payload_sha256(original)
    target = redact_terminal_task_run_llm_call(original)

    assert validate_terminal_task_run_llm_redaction(
        original,
        target,
        expected_payload_sha256=expected_sha256,
        expected_tier=PayloadRetentionTier.FULL,
    ) is PayloadRetentionTier.HASH_ONLY
    with pytest.raises(ValueError, match="not canonical"):
        validate_terminal_task_run_llm_redaction(
            original,
            replace(target, response_content="tampered"),
            expected_payload_sha256=expected_sha256,
            expected_tier=PayloadRetentionTier.FULL,
        )


def test_terminal_tool_output_redaction_is_content_free_and_idempotent() -> None:
    source = '{"secret":"TOOL_OUTPUT_SECRET"}'
    redacted, source_sha256, changed = (
        redact_terminal_task_run_llm_tool_output(source)
    )

    assert changed is True
    assert "TOOL_OUTPUT_SECRET" not in redacted
    assert len(source_sha256) == 64
    repeated, repeated_sha256, repeated_changed = (
        redact_terminal_task_run_llm_tool_output(redacted)
    )
    assert repeated == redacted
    assert repeated_sha256 == source_sha256
    assert repeated_changed is False


def test_terminal_task_run_redaction_conflict_fails_closed() -> None:
    store = _RedactionStore([_full_llm_record()], conflict=True)
    with pytest.raises(RuntimeError, match="conflicted"):
        redact_task_run_llm_calls(store, "run-redact", ["pid-redact"])


def test_terminal_task_run_redaction_rejects_nonterminal_call() -> None:
    with pytest.raises(ValueError, match="nonterminal"):
        redact_terminal_task_run_llm_call(
            replace(_full_llm_record(), completed_at=None)
        )
