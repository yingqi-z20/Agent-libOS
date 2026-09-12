from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG, LLMProfile, ProviderToolsConfig
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.llm.task_runs import (
    normalize_provider_continuation_manifest,
    normalize_validated_action_manifest,
    provider_continuation_manifest,
    validated_action_manifest,
)
from agent_libos.models import LLMCallRecord, TaskRunRetention, TaskRunSpecV1, TaskRunStatus
from agent_libos.models.exceptions import TaskRunRevisionConflict, ValidationError
from agent_libos.utils.ids import utc_now


CONFIG = replace(
    DEFAULT_CONFIG,
    llm=replace(
        DEFAULT_CONFIG.llm,
        profiles={"default": LLMProfile(
            provider_tools=ProviderToolsConfig(provider="openai", web_search=True),
        )},
    ),
    task_runs=replace(DEFAULT_CONFIG.task_runs, plaintext_payloads_enabled=True),
)


def _create(runtime: Runtime):
    created = runtime.task_runs.create(
        TaskRunSpecV1(
            goal="Use the search result to finish the report.",
            display_title="Hosted result continuation",
            retention=TaskRunRetention.PERMANENT,
        ),
        client_request_id="create-provider-continuation",
    )
    assert created.root_pid is not None
    return created


def _call(runtime: Runtime, pid: str, call_id: str = "hosted-call") -> LLMCallRecord:
    now = utc_now()
    return LLMCallRecord(
        call_id=call_id, pid=pid, image_id="base-agent:v0",
        purpose="action_selection", status="ok", api="responses", model="test-model",
        request_options={
            "provider_tools_enabled": True,
            "provider_tools_configured": {"provider": "openai", "web_search": True},
            "llm_context_generation": runtime.store.get_llm_context_generation(pid),
            "llm_profile_id": "default",
            "llm_profile_identity_sha256": runtime.llms.profile_identity_sha256("default"),
        },
        response_content="The retained search result.", tool_calls=[],
        created_at=now, completed_at=now,
    )


def _record(runtime: Runtime, call: LLMCallRecord) -> dict[str, Any]:
    runtime.store.insert_llm_call(call)
    manifest = provider_continuation_manifest(call_id=call.call_id, data_labels={})
    runtime.task_runs.record_provider_continuation(
        pid=call.pid, call_id=call.call_id, continuation_manifest=manifest,
        context_generation=runtime.store.get_llm_context_generation(call.pid),
    )
    return manifest


def test_hosted_result_commits_one_idempotent_completed_safe_point(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "continuation.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        call = _call(runtime, created.root_pid)
        manifest = _record(runtime, call)
        point = runtime.store.get_task_run_resume_point(created.root_pid)
        assert point is not None and point.pending_action_payload_id is not None
        assert runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid) == manifest
        assert runtime.task_runs.pending_validated_action_for_pid(created.root_pid) is None
        context = runtime.task_runs.prompt_context_for_pid(created.root_pid)
        assert context["transcript_messages"] == [
            {"role": "assistant", "content": call.response_content},
        ]
        runtime.task_runs.record_provider_continuation(
            pid=created.root_pid, call_id=call.call_id, continuation_manifest=manifest,
            context_generation=point.context_generation,
        )
        assert runtime.store.get_task_run_resume_point(created.root_pid) == point
        record = runtime.store.get_task_run(created.run_id)
        assert record.step_count == record.completed_step_count == 1
        ledger = runtime.store.list_task_run_ledger(created.run_id, after=None, limit=100)
        assert len([item for item in ledger.records if item.status == "provider_continuation"]) == 1
    finally:
        runtime.close()


def test_hosted_continuation_survives_reopen_and_is_consumed_by_local_action(tmp_path: Path) -> None:
    path = tmp_path / "reopen.sqlite"
    first = Runtime.open(path, config=CONFIG)
    try:
        created = _create(first)
        manifest = _record(first, _call(first, created.root_pid))
    finally:
        first.close()
    second = Runtime.open(path, config=CONFIG)
    try:
        assert second.task_runs.get(created.run_id).status is not TaskRunStatus.NEEDS_ATTENTION
        assert second.task_runs.pending_provider_continuation_for_pid(created.root_pid) == manifest
        call = _call(second, created.root_pid, "local-action-call")
        call = replace(call, request_options={**call.request_options, "provider_tools_enabled": False})
        second.store.insert_llm_call(call)
        action_manifest = validated_action_manifest(
            [{"action": "process_exit", "payload": {"done": True}}],
            call_id=call.call_id, parallel_tool_calls=False, host_auto_wait=False,
            tool_call_count=1, data_labels={},
        )
        second.task_runs.record_validated_transcript(
            pid=created.root_pid, call_id=call.call_id, action_manifest=action_manifest,
            context_generation=second.store.get_llm_context_generation(created.root_pid),
        )
        assert second.task_runs.pending_provider_continuation_for_pid(created.root_pid) is None
        context = second.task_runs.prompt_context_for_pid(created.root_pid)
        assert context["transcript_messages"][0]["content"] == "The retained search result."
        record = second.store.get_task_run(created.run_id)
        assert record.step_count == 2 and record.completed_step_count == 1
    finally:
        second.close()


@pytest.mark.parametrize("change", [
    {"provider_tools_enabled": False},
    {"provider_tools_configured": {}},
    {"llm_context_generation": "wrong"},
    {"llm_profile_id": "wrong"},
    {"llm_profile_identity_sha256": "0" * 64},
])
def test_continuation_rejects_changed_request_binding(tmp_path: Path, change: dict[str, Any]) -> None:
    runtime = Runtime.open(tmp_path / "binding.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        call = _call(runtime, created.root_pid)
        call = replace(call, request_options={**call.request_options, **change})
        with pytest.raises(ValidationError, match="request binding"):
            _record(runtime, call)
        assert runtime.store.get_task_run_resume_point(created.root_pid) is None
    finally:
        runtime.close()


def test_continuation_does_not_replace_or_repeat_pending_hosted_work(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "duplicate.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        manifest = _record(runtime, _call(runtime, created.root_pid))
        with pytest.raises(TaskRunRevisionConflict, match="pending continuation"):
            _record(runtime, _call(runtime, created.root_pid, "duplicate-hosted-call"))
        assert runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid) == manifest
    finally:
        runtime.close()


@pytest.mark.parametrize("changes", [
    {"tool_calls": [{"id": "call", "name": "process_exit", "arguments": "{}"}]},
    {"status": "error"},
    {"response_content": ""},
])
def test_continuation_requires_successful_result_without_local_dispatch(
    tmp_path: Path, changes: dict[str, Any],
) -> None:
    runtime = Runtime.open(tmp_path / "source-shape.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        with pytest.raises(ValidationError):
            _record(runtime, replace(_call(runtime, created.root_pid), **changes))
        assert runtime.store.get_task_run_resume_point(created.root_pid) is None
    finally:
        runtime.close()


def test_continuation_successor_cannot_reenable_hosted_tools(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "successor.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        manifest = _record(runtime, _call(runtime, created.root_pid))
        call = _call(runtime, created.root_pid, "unsafe-successor")
        runtime.store.insert_llm_call(call)
        action = validated_action_manifest(
            [{"action": "process_exit"}], call_id=call.call_id,
            parallel_tool_calls=False, host_auto_wait=False,
            tool_call_count=1, data_labels={},
        )
        with pytest.raises(ValidationError, match="re-enabled hosted tools"):
            runtime.task_runs.record_validated_transcript(
                pid=created.root_pid, call_id=call.call_id, action_manifest=action,
                context_generation=runtime.store.get_llm_context_generation(created.root_pid),
            )
        assert runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid) == manifest
    finally:
        runtime.close()


def test_continuation_does_not_truncate_exact_transcript_past_run_bound(tmp_path: Path) -> None:
    config = replace(CONFIG, task_runs=replace(CONFIG.task_runs, payload_max_bytes=8192))
    runtime = Runtime.open(tmp_path / "bounds.sqlite", config=config)
    try:
        created = _create(runtime)
        call = replace(_call(runtime, created.root_pid), response_content="x" * 16384)
        with pytest.raises(ValidationError, match="payload bound"):
            _record(runtime, call)
        assert runtime.store.get_task_run_resume_point(created.root_pid) is None
        assert runtime.store.get_task_run(created.run_id).step_count == 0
    finally:
        runtime.close()


def test_activity_only_result_and_pause_cancel_remain_durable(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "control.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        paused = runtime.task_runs.pause(
            created.run_id, expected_revision=created.revision, command_id="pause-before-settlement",
        )
        call = replace(_call(runtime, created.root_pid), response_content="", reasoning={
            "schema_version": 1, "kind": "provider_trace", "selected_attempt": 1,
            "attempts": [{"provider_tools": {"activities": [
                {"type": "web_search_call", "status": "completed", "query": "retained query"},
            ]}}],
        })
        manifest = _record(runtime, call)
        assert runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid) == manifest
        context = runtime.task_runs.prompt_context_for_pid(created.root_pid)
        assert context["transcript_messages"][0]["content"] == "Provider web_search activity: completed"
        current = runtime.task_runs.get(created.run_id)
        assert current.status is paused.status is TaskRunStatus.PAUSED
        cancelled = runtime.task_runs.cancel(
            created.run_id, expected_revision=current.revision, command_id="cancel-hosted-continuation",
        )
        assert cancelled.status is TaskRunStatus.CANCELLED
    finally:
        runtime.close()


def test_continuation_rejects_source_call_change_before_dispatch(tmp_path: Path, monkeypatch) -> None:
    runtime = Runtime.open(tmp_path / "source.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        call = _call(runtime, created.root_pid)
        _record(runtime, call)
        original = runtime.store.get_llm_call
        monkeypatch.setattr(runtime.store, "get_llm_call", lambda call_id: (
            replace(call, response_content="modified provider output")
            if call_id == call.call_id else original(call_id)
        ))
        with pytest.raises(ValidationError, match="source evidence changed"):
            runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid)
        with pytest.raises(ValidationError, match="source evidence changed"):
            runtime.task_runs._prevalidate_recoverable_record(
                runtime.store.get_task_run(created.run_id)
            )
        process = runtime.store.get_process(created.root_pid)
        blocker = runtime.task_runs._recover_task_run_resume_state(process)
        assert blocker["kind"] == "pending_action_unreplayable"
    finally:
        runtime.close()


def test_continuation_startup_prevalidation_uses_only_persisted_bindings(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = Runtime.open(tmp_path / "static-prevalidation.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        _record(runtime, _call(runtime, created.root_pid))

        def reject_registry_lookup(*_args):
            raise AssertionError("early recovery cannot consult live provider registry")

        monkeypatch.setattr(runtime.llms, "profile_identity_sha256", reject_registry_lookup)
        runtime.task_runs._prevalidate_recoverable_record(
            runtime.store.get_task_run(created.run_id)
        )
    finally:
        runtime.close()


def test_continuation_manifest_is_separate_from_strict_action_manifest() -> None:
    manifest = provider_continuation_manifest(call_id="hosted-call", data_labels={})
    with pytest.raises(ValueError, match="invalid shape"):
        normalize_validated_action_manifest(manifest)
    with pytest.raises(ValueError, match="disable hosted tools"):
        normalize_provider_continuation_manifest({**manifest, "provider_tools_disabled_on_resume": False})
    action = validated_action_manifest(
        [{"action": "process_exit"}], call_id="call", parallel_tool_calls=False,
        host_auto_wait=False, tool_call_count=1, data_labels={},
    )
    with pytest.raises(ValueError, match="action list is invalid"):
        normalize_validated_action_manifest({**action, "actions": []})


@pytest.mark.parametrize("replay", [False, True])
def test_runtime_task_run_continues_hosted_result_with_one_tools_disabled_call(
    tmp_path: Path, replay: bool,
) -> None:
    class HostedThenLocalClient(LLMClient):
        def __init__(self) -> None:
            super().__init__(
                model="gpt-6-astra", api_key="deterministic-test", api_mode="responses",
                provider_tools=ProviderToolsConfig(provider="openai", web_search=True),
                responses_replay=replay,
            )
            self.requests: list[dict[str, Any]] = []

        async def acomplete_action(self, messages, tools, **kwargs) -> LLMCompletion:
            self.requests.append({"messages": messages, "tools": tools, **kwargs})
            if len(self.requests) == 1:
                assert kwargs["provider_tools_enabled"] is True
                return LLMCompletion(
                    content="The retained search result for the report.", tool_calls=[],
                    api="responses", model="gpt-6-astra", response_id="hosted-response",
                    usage={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                    response_items=[{
                        "type": "message", "id": "msg-hosted-result", "role": "assistant",
                        "status": "completed", "content": [{
                            "type": "output_text", "text": "The retained search result for the report.",
                            "annotations": [],
                        }],
                    }],
                )
            assert len(self.requests) == 2, "hosted work or action repair repeated unexpectedly"
            assert kwargs["provider_tools_enabled"] is False
            assert "The retained search result for the report." in str([
                *messages, *kwargs.get("responses_items", []),
            ])
            item = {
                "type": "function_call", "id": "fc-local-exit", "call_id": "local-exit",
                "name": "process_exit", "arguments": '{"payload":{"done":true}}',
                "status": "completed",
            }
            return LLMCompletion(
                content="", tool_calls=[dict(item)], response_items=[dict(item)],
                api="responses", model="gpt-6-astra", response_id="local-response",
                usage={"input_tokens": 120, "output_tokens": 10, "total_tokens": 130},
            )

    config = replace(CONFIG, llm=replace(CONFIG.llm, profiles={
        "default": replace(CONFIG.llm.profiles["default"], responses_replay=replay),
    }))
    runtime = Runtime.open(tmp_path / "actual-executor.sqlite", config=config)
    try:
        created = _create(runtime)
        client = HostedThenLocalClient()
        runtime.llm.client = client
        result = runtime.task_runs.run_until_blocked(
            created.run_id, expected_revision=created.revision,
            command_id="run-hosted-then-local", max_quanta=4,
        )
        assert result.status is TaskRunStatus.SUCCEEDED
        assert len(client.requests) == 2
        assert runtime.task_runs.pending_provider_continuation_for_pid(created.root_pid) is None
        record = runtime.store.get_task_run(created.run_id)
        assert record.step_count == record.completed_step_count == 2
        assert any(
            entry.action == "llm.provider_continuation" for entry in runtime.audit.trace()
        )
    finally:
        runtime.close()
