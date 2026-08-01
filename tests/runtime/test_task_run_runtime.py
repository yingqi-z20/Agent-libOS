from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.external_effects import (
    mark_external_effect_unknown,
    record_external_effect,
)
from agent_libos.evidence.payload_retention import (
    PayloadRetentionTier,
    external_effect_payload_retention_tier,
    llm_call_payload_retention_tier,
)
from agent_libos.llm.task_runs import (
    completed_outcome_manifest,
    validated_action_manifest,
)
from agent_libos.models import (
    ChildProcessWait,
    DataFlowContext,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ExitedProcessOutcome,
    HumanProcessWait,
    HumanRequest,
    HumanRequestStatus,
    LLMCallRecord,
    MessageProcessWait,
    ObjectTaskStatus,
    PausedProcessWait,
    ProcessStatus,
    StaleExecutionProcessWait,
    TaskRunAction,
    TaskRunPayloadRetention,
    TaskRunRetention,
    TaskRunStatus,
    ToolProcessWait,
    process_wait_state_to_mapping,
)
from agent_libos.models.exceptions import (
    TaskRunCommandConflict,
    TaskRunRevisionConflict,
    ValidationError,
)
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps
from tests.support.fakes import RecordingActionClient
from tests.support.external_effects import begin_external_effect_intent


def _config(*, plaintext: bool = True):
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=plaintext,
            recovery_page_size=500,
            recovery_page_hard_limit=5_000,
        ),
    )


def _spec(
    title: str = "Durable integration",
    *,
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL,
    deadline_at: str | None = None,
) -> TaskRunSpecV1:
    return TaskRunSpecV1(
        goal={"goal": title},
        display_title=title,
        image_id="base-agent:v0",
        retention=retention,
        deadline_at=deadline_at,
    )


def _create(runtime: Runtime, *, request_id: str = "create-1", **spec: object):
    return runtime.task_runs.create(
        _spec(**spec),
        client_request_id=request_id,
    )


def _full_llm_call(call_id: str, pid: str, secret: str) -> LLMCallRecord:
    now = utc_now()
    return LLMCallRecord(
        call_id=call_id,
        pid=pid,
        image_id="base-agent:v0",
        purpose="action_selection",
        status="ok",
        api="responses",
        model="test-model",
        request_id=f"request-{call_id}",
        response_id=f"response-{call_id}",
        messages=[{"role": "user", "content": secret}],
        tools=[{"name": "test", "description": secret}],
        request_options={},
        response_content=secret,
        tool_calls=[{"name": "test", "arguments": {"value": secret}}],
        reasoning={"text": secret},
        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        raw_response={"value": secret},
        observability={"bytes": len(secret)},
        created_at=now,
        completed_at=now,
    )


def _human_request(request_id: str, pid: str, secret: str) -> HumanRequest:
    now = utc_now()
    return HumanRequest(
        request_id=request_id,
        pid=pid,
        human="host",
        payload={"type": "question", "question": f"prompt:{secret}"},
        status=HumanRequestStatus.APPROVED,
        decision={"answer": f"answer:{secret}"},
        blocking=True,
        created_at=now,
        updated_at=now,
    )


def _claim_test_task_run_action(
    runtime: Runtime,
    *,
    run_id: str,
    pid: str,
    call_id: str,
    action: dict[str, object],
):
    runtime.store.insert_llm_call(_full_llm_call(call_id, pid, "RECOVERY_ACTION"))
    generation = runtime.store.get_llm_context_generation(pid)
    manifest = validated_action_manifest(
        [action],
        call_id=call_id,
        parallel_tool_calls=False,
        host_auto_wait=False,
        tool_call_count=1,
        data_labels={},
    )
    runtime.task_runs.record_validated_transcript(
        pid=pid,
        call_id=call_id,
        action_manifest=manifest,
        context_generation=generation,
    )
    assert runtime.task_runs.pending_validated_action_for_pid(pid) == manifest
    point = runtime.store.get_task_run_resume_point(pid, complete_only=True)
    assert point is not None and point.pending_action_payload_id is not None
    assert point.run_id == run_id
    wrapper = runtime.task_runs._decode_pending_resume_payload(point)
    assert wrapper["state"] == "dispatching"
    return manifest, point, wrapper


def _record_test_task_run_safe_point(
    runtime: Runtime,
    *,
    run_id: str,
    pid: str,
    call_id: str,
) -> None:
    """Persist one complete provider-independent local resume point."""

    action = {"action": "get_current_time", "timezone": "UTC"}
    manifest, _point, _wrapper = _claim_test_task_run_action(
        runtime,
        run_id=run_id,
        pid=pid,
        call_id=call_id,
        action=action,
    )
    generation = runtime.store.get_llm_context_generation(pid)
    assert manifest["call_id"] == call_id
    outcome = completed_outcome_manifest(
        state="completed",
        paired_outputs_persisted=True,
        data_labels={},
        result={
            "ok": True,
            "action": action,
            "result": {
                "ok": True,
                "payload": {"timezone": "UTC", "time": "2000-01-01T00:00:00Z"},
            },
        },
    )
    runtime.task_runs.stage_completed_transcript(
        pid=pid,
        call_id=call_id,
        outcome_manifest=outcome,
        context_generation=generation,
    )
    runtime.task_runs.record_completed_transcript(
        pid=pid,
        call_id=call_id,
        outcome_manifest=outcome,
        context_generation=generation,
    )
    point = runtime.store.get_task_run_resume_point(pid, complete_only=True)
    assert point is not None
    assert point.run_id == run_id
    assert point.pending_action_payload_id is None


def _resume_point_integrity(
    runtime: Runtime,
    point: object,
    **overrides: object,
) -> str:
    """Rebuild the persisted resume envelope for corruption-boundary tests."""

    selected = {
        name: overrides.get(name, getattr(point, name))
        for name in (
            "run_id",
            "pid",
            "task_run_epoch",
            "process_revision",
            "context_generation",
            "safe_point_seq",
            "binding_hash",
            "image_binding_hash",
            "tool_binding_hash",
            "provider_binding_hash",
            "transcript_payload_id",
            "summary_payload_id",
            "pending_action_payload_id",
            "last_effect_seq",
        )
    }
    transcript = runtime.store.get_task_run_payload(
        str(selected["transcript_payload_id"])
    )
    assert transcript is not None
    summary = (
        runtime.store.get_task_run_payload(str(selected["summary_payload_id"]))
        if selected["summary_payload_id"] is not None
        else None
    )
    pending = (
        runtime.store.get_task_run_payload(
            str(selected["pending_action_payload_id"])
        )
        if selected["pending_action_payload_id"] is not None
        else None
    )
    return runtime.task_runs._sha256(  # noqa: SLF001 - integrity fault fixture
        {
            "run_id": selected["run_id"],
            "pid": selected["pid"],
            "task_run_epoch": selected["task_run_epoch"],
            "process_revision": selected["process_revision"],
            "context_generation": selected["context_generation"],
            "safe_point_seq": selected["safe_point_seq"],
            "binding_hash": selected["binding_hash"],
            "image_binding_hash": selected["image_binding_hash"],
            "tool_binding_hash": selected["tool_binding_hash"],
            "provider_binding_hash": selected["provider_binding_hash"],
            "transcript_sha256": transcript.sha256,
            "summary_payload_id": selected["summary_payload_id"],
            "summary_sha256": summary.sha256 if summary is not None else None,
            "pending_action_payload_id": selected["pending_action_payload_id"],
            "pending_action_sha256": pending.sha256 if pending is not None else None,
            "last_effect_seq": selected["last_effect_seq"],
        }
    )


def _task_run_action_effect_binding(
    *,
    pid: str,
    point,
    wrapper: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": point.run_id,
        "pid": pid,
        "call_id": wrapper.get("call_id"),
        "context_generation": wrapper.get("context_generation"),
        "action_manifest_sha256": wrapper.get("manifest_sha256"),
        "source_safe_point_seq": point.safe_point_seq,
    }


def _mark_test_run_started_and_attention(
    runtime: Runtime,
    *,
    run_id: str,
    effect_id: str,
):
    current = runtime.store.get_task_run(run_id)
    assert current is not None
    now = utc_now()
    started = runtime.store.update_task_run_cas(
        run_id,
        current.revision,
        updates={
            "status": TaskRunStatus.RUNNING,
            "started_at": now,
            "updated_at": now,
        },
        expected_runtime_epoch=runtime.task_runs.runtime_epoch,
    )
    return runtime.task_runs._mark_attention(
        started,
        runtime.task_runs._blocker(
            "unknown_effect",
            "test provider outcome is unknown",
            effect_ids=[effect_id],
        ),
    )


def test_create_requires_plaintext_opt_in_and_is_atomic_idempotent(tmp_path: Path) -> None:
    disabled = Runtime.open(tmp_path / "disabled.sqlite", config=_config(plaintext=False))
    try:
        with pytest.raises(ValidationError, match="plaintext payloads are disabled"):
            _create(disabled)
        assert disabled.store.list_task_runs(limit=1).records == ()
        assert disabled.process.list() == []
    finally:
        disabled.close()

    runtime = Runtime.open(tmp_path / "enabled.sqlite", config=_config())
    try:
        created = _create(runtime)
        replayed = _create(runtime)
        assert replayed == created
        assert len(runtime.process.list()) == 1
        process = runtime.process.get(created.root_pid or "")
        assert (
            process.task_run_id,
            process.task_run_epoch,
            process.task_run_role,
        ) == (created.run_id, runtime.task_runs.runtime_epoch, "root")
        requirements = runtime.store.list_task_run_requirements(created.run_id)
        payloads = runtime.store.list_task_run_payloads(created.run_id)
        assert len(requirements) == len(payloads) == 1
        assert requirements[0].payload_id == payloads[0].payload_id

        with pytest.raises(TaskRunRevisionConflict, match="different request"):
            runtime.task_runs.create(
                _spec("different"),
                client_request_id="create-1",
            )
        assert len(runtime.store.list_task_runs(limit=10).records) == 1
        assert len(runtime.process.list()) == 1
    finally:
        runtime.close()


def test_create_auto_run_uses_immutable_create_revision_for_pending_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "create-auto-run-identity.sqlite", config=_config())
    selected_spec = _spec(
        "Create auto-run identity",
        retention=TaskRunRetention.PERMANENT,
    )
    request_id = "create-auto-run-identity"
    dispatch_count = 0
    try:
        created = runtime.task_runs.create(
            selected_spec,
            client_request_id=request_id,
            auto_run=False,
        )
        original_complete = runtime.task_runs._complete_command_summary

        def admit_without_provider(**_kwargs: object) -> list[object]:
            nonlocal dispatch_count
            dispatch_count += 1
            return []

        def lose_run_result(
            record: object,
            command_id: str,
            command_kind: str,
            request: object,
            **kwargs: object,
        ) -> object:
            if command_kind == "run":
                raise RuntimeError("injected create auto-run response loss")
            return original_complete(
                record,
                command_id,
                command_kind,
                request,
                **kwargs,
            )

        monkeypatch.setattr(runtime, "run_until_idle", admit_without_provider)
        monkeypatch.setattr(
            runtime.task_runs,
            "_complete_command_summary",
            lose_run_result,
        )
        with pytest.raises(RuntimeError, match="auto-run response loss"):
            runtime.task_runs.create(
                selected_spec,
                client_request_id=request_id,
                auto_run=True,
            )

        create_command = runtime.store.get_task_run_command_by_client_request_id(
            request_id
        )
        run_command = runtime.store.get_task_run_command(
            created.run_id,
            f"{request_id}:run",
        )
        assert create_command is not None
        assert create_command.result_revision == created.revision
        assert run_command is not None
        assert run_command.result["settlement_state"] == "pending"
        assert run_command.request_hash == TaskRunManager._request_hash(
            "run",
            {
                "expected_revision": created.revision,
                "max_quanta": None,
            },
        )
        assert dispatch_count == 1

        monkeypatch.setattr(
            runtime.task_runs,
            "_complete_command_summary",
            original_complete,
        )
        replayed = runtime.task_runs.create(
            selected_spec,
            client_request_id=request_id,
            auto_run=True,
        )
        completed = runtime.store.get_task_run_command(
            created.run_id,
            f"{request_id}:run",
        )
        assert replayed.status is TaskRunStatus.RUNNING
        assert completed is not None
        assert completed.result["settlement_state"] == "complete"
        assert completed.result_revision == replayed.revision
        assert dispatch_count == 1
        assert runtime.task_runs.create(
            selected_spec,
            client_request_id=request_id,
            auto_run=False,
        ) == created
    finally:
        runtime.close()


def test_create_auto_run_never_rebinds_to_a_newer_queued_revision(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "create-auto-run-no-rebind.sqlite",
        config=_config(),
    )
    selected_spec = _spec(
        "Create auto-run no rebind",
        retention=TaskRunRetention.PERMANENT,
    )
    request_id = "create-auto-run-no-rebind"
    try:
        created = runtime.task_runs.create(
            selected_spec,
            client_request_id=request_id,
            auto_run=False,
        )
        advanced = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "advance the queued Run revision"},
            expected_revision=created.revision,
            command_id="advance-before-create-auto-run",
        )
        assert advanced.status is TaskRunStatus.QUEUED
        assert advanced.revision > created.revision
        before = (
            runtime.store.get_task_run(created.run_id),
            tuple(
                runtime.store.list_task_run_commands(
                    created.run_id,
                    limit=100,
                )
            ),
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
        )

        with pytest.raises(
            TaskRunRevisionConflict,
            match="auto-run result is not durably reconstructable",
        ):
            runtime.task_runs.create(
                selected_spec,
                client_request_id=request_id,
                auto_run=True,
            )

        assert runtime.store.get_task_run_command(
            created.run_id,
            f"{request_id}:run",
        ) is None
        assert (
            runtime.store.get_task_run(created.run_id),
            tuple(
                runtime.store.list_task_run_commands(
                    created.run_id,
                    limit=100,
                )
            ),
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
        ) == before
    finally:
        runtime.close()


@pytest.mark.parametrize("max_quanta", [0, -1])
def test_run_until_blocked_rejects_nonpositive_quanta_without_mutation(
    tmp_path: Path,
    max_quanta: int,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"nonpositive-quanta-{max_quanta}.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        before = runtime.store.get_task_run(created.run_id)

        with pytest.raises(
            ValidationError,
            match="max_quanta must be a positive integer",
        ):
            runtime.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=created.revision,
                command_id=f"reject-quanta-{max_quanta}",
                max_quanta=max_quanta,
            )

        assert runtime.store.get_task_run(created.run_id) == before
        assert runtime.store.get_task_run_command(
            created.run_id,
            f"reject-quanta-{max_quanta}",
        ) is None
        assert runtime.process.get(created.root_pid).resource_usage.llm_calls == 0
    finally:
        runtime.close()


def test_create_persists_the_exact_resolved_default_image(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "resolved-image.sqlite", config=_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="resolve the Host default Image",
                display_title="Resolved default Image",
                image_id=None,
            ),
            client_request_id="create-resolved-image",
        )
        persisted = runtime.store.get_task_run(created.run_id)
        assert persisted is not None
        assert persisted.image_id == runtime.config.runtime.default_image_id
        assert runtime.process.get(created.root_pid or "").image_id == persisted.image_id
    finally:
        runtime.close()


def test_queued_run_reopens_under_new_epoch_without_background_dispatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "queued-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(first)
        first_epoch = first.task_runs.runtime_epoch
        root_pid = created.root_pid
        assert root_pid is not None
        assert first.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        process = reopened.process.get(root_pid)
        assert reopened.task_runs.runtime_epoch > first_epoch
        assert process.task_run_epoch == reopened.task_runs.runtime_epoch
        assert process.status is ProcessStatus.RUNNABLE
        assert process.resource_usage.llm_calls == 0
        assert recovered.status is TaskRunStatus.QUEUED
        assert TaskRunAction.RUN in recovered.allowed_actions
        assert reopened.run_next_process_once() is None
        assert process.resource_usage.llm_calls == 0
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("column", "value", "blocker"),
    [
        ("canonical_json", '{"goal":"tampered"}', "payload_corrupt"),
        ("binding_hash", "0" * 64, "binding_drift"),
    ],
)
def test_reopen_blocks_corrupt_payload_or_binding_before_any_dispatch(
    column: str,
    value: str,
    blocker: str,
    tmp_path: Path,
) -> None:
    database = tmp_path / f"reopen-{column}.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime)
        root_pid = created.root_pid
        assert root_pid is not None
    finally:
        runtime.close()

    connection = sqlite3.connect(database)
    try:
        if column == "canonical_json":
            connection.execute(
                "UPDATE task_run_payloads SET canonical_json = ? WHERE run_id = ? AND role = 'goal'",
                (value, created.run_id),
            )
        else:
            connection.execute(
                "UPDATE task_runs SET binding_hash = ? WHERE run_id = ?",
                (value, created.run_id),
            )
        connection.commit()
    finally:
        connection.close()

    reopened = Runtime.open(database, config=_config())
    try:
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert blocker in {item["kind"] for item in summary.blockers}
        actions = {action.value for action in summary.allowed_actions}
        assert "run" not in actions and "resume" not in actions
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "corruption",
    [
        "cross_run_transcript",
        "future_epoch",
        "effect_baseline_envelope",
    ],
)
def test_resume_point_corruption_never_crosses_run_epoch_or_effect_fence(
    corruption: str,
    tmp_path: Path,
) -> None:
    database = tmp_path / f"resume-point-{corruption}.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            request_id=f"create-resume-point-{corruption}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        source_record = runtime.store.get_task_run(created.run_id)
        assert source_record is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            source_record.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        _record_test_task_run_safe_point(
            runtime,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id=f"safe-point-{corruption}",
        )
        point = runtime.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert point is not None

        if corruption == "cross_run_transcript":
            other = _create(
                runtime,
                request_id="create-cross-run-transcript-source",
                title="Cross-run transcript source",
                retention=TaskRunRetention.PERMANENT,
            )
            assert other.root_pid is not None
            other_record = runtime.store.get_task_run(other.run_id)
            assert other_record is not None
            runtime.store.update_task_run_cas(
                other.run_id,
                other_record.revision,
                updates={
                    "status": TaskRunStatus.RUNNING,
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                },
                expected_runtime_epoch=runtime.task_runs.runtime_epoch,
            )
            _record_test_task_run_safe_point(
                runtime,
                run_id=other.run_id,
                pid=other.root_pid,
                call_id="cross-run-transcript-source-safe-point",
            )
            other_point = runtime.store.get_task_run_resume_point(
                other.root_pid,
                complete_only=True,
            )
            assert other_point is not None
            integrity = _resume_point_integrity(
                runtime,
                point,
                transcript_payload_id=other_point.transcript_payload_id,
            )
            runtime.store._execute(  # noqa: SLF001 - cross-run corruption fixture
                "UPDATE task_run_resume_points "
                "SET transcript_payload_id = ?, integrity_sha256 = ? "
                "WHERE pid = ?",
                (
                    other_point.transcript_payload_id,
                    integrity,
                    created.root_pid,
                ),
            )
        elif corruption == "future_epoch":
            future_epoch = runtime.task_runs.runtime_epoch + 100
            integrity = _resume_point_integrity(
                runtime,
                point,
                task_run_epoch=future_epoch,
            )
            runtime.store._execute(  # noqa: SLF001 - future epoch corruption fixture
                "UPDATE task_run_resume_points "
                "SET task_run_epoch = ?, integrity_sha256 = ? WHERE pid = ?",
                (future_epoch, integrity, created.root_pid),
            )
        else:
            # Even a rehashed high-water mark must never claim append-only
            # effect evidence that does not exist.
            future_effect_seq = 2**62
            integrity = _resume_point_integrity(
                runtime,
                point,
                last_effect_seq=future_effect_seq,
            )
            runtime.store._execute(  # noqa: SLF001 - effect fence corruption fixture
                "UPDATE task_run_resume_points "
                "SET last_effect_seq = ?, integrity_sha256 = ? WHERE pid = ?",
                (future_effect_seq, integrity, created.root_pid),
            )
        llm_calls_before = runtime.process.get(
            created.root_pid
        ).resource_usage.llm_calls
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        summary = reopened.task_runs.get(created.run_id)
        process = reopened.process.get(created.root_pid)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in summary.blockers} == {"payload_corrupt"}
        assert TaskRunAction.RUN not in summary.allowed_actions
        assert TaskRunAction.RESUME not in summary.allowed_actions
        assert process.resource_usage.llm_calls == llm_calls_before
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_integrity_valid_resume_point_cannot_move_to_another_run_process(
    tmp_path: Path,
) -> None:
    database = tmp_path / "resume-point-cross-run-identity.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        source = _create(
            runtime,
            request_id="create-cross-run-point-source",
            title="Cross-run point source",
            retention=TaskRunRetention.PERMANENT,
        )
        target = _create(
            runtime,
            request_id="create-cross-run-point-target",
            title="Cross-run point target",
            retention=TaskRunRetention.PERMANENT,
        )
        assert source.root_pid is not None
        assert target.root_pid is not None
        for created in (source, target):
            record = runtime.store.get_task_run(created.run_id)
            assert record is not None
            runtime.store.update_task_run_cas(
                created.run_id,
                record.revision,
                updates={
                    "status": TaskRunStatus.RUNNING,
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                },
                expected_runtime_epoch=runtime.task_runs.runtime_epoch,
            )
            _record_test_task_run_safe_point(
                runtime,
                run_id=created.run_id,
                pid=created.root_pid,
                call_id=f"cross-run-point-{created.run_id}",
            )
        source_point = runtime.store.get_task_run_resume_point(
            source.root_pid,
            complete_only=True,
        )
        target_point = runtime.store.get_task_run_resume_point(
            target.root_pid,
            complete_only=True,
        )
        assert source_point is not None
        assert target_point is not None
        overrides = {
            "run_id": target_point.run_id,
            "task_run_epoch": target_point.task_run_epoch,
            "binding_hash": target_point.binding_hash,
            "image_binding_hash": target_point.image_binding_hash,
            "tool_binding_hash": target_point.tool_binding_hash,
            "provider_binding_hash": target_point.provider_binding_hash,
            "transcript_payload_id": target_point.transcript_payload_id,
            "summary_payload_id": target_point.summary_payload_id,
            "pending_action_payload_id": target_point.pending_action_payload_id,
            "last_effect_seq": target_point.last_effect_seq,
        }
        integrity = _resume_point_integrity(
            runtime,
            source_point,
            **overrides,
        )
        runtime.store._execute(  # noqa: SLF001 - cross-Run point corruption fixture
            "UPDATE task_run_resume_points SET run_id = ?, task_run_epoch = ?, "
            "binding_hash = ?, image_binding_hash = ?, tool_binding_hash = ?, "
            "provider_binding_hash = ?, transcript_payload_id = ?, "
            "summary_payload_id = ?, pending_action_payload_id = ?, "
            "last_effect_seq = ?, integrity_sha256 = ? WHERE pid = ?",
            (
                overrides["run_id"],
                overrides["task_run_epoch"],
                overrides["binding_hash"],
                overrides["image_binding_hash"],
                overrides["tool_binding_hash"],
                overrides["provider_binding_hash"],
                overrides["transcript_payload_id"],
                overrides["summary_payload_id"],
                overrides["pending_action_payload_id"],
                overrides["last_effect_seq"],
                integrity,
                source.root_pid,
            ),
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        source_summary = reopened.task_runs.get(source.run_id)
        target_summary = reopened.task_runs.get(target.run_id)
        assert source_summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert target_summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert "payload_corrupt" in {
            item["kind"] for item in target_summary.blockers
        }
        for summary in (source_summary, target_summary):
            assert TaskRunAction.RUN not in summary.allowed_actions
            assert TaskRunAction.RESUME not in summary.allowed_actions
        assert reopened.run_next_process_once() is None
        for pid in (source.root_pid, target.root_pid):
            usage = reopened.process.get(pid).resource_usage
            assert usage.llm_calls == 0
            assert usage.tool_calls == 0
    finally:
        reopened.close()


def test_resume_point_from_multiple_prior_runtime_epochs_remains_valid(
    tmp_path: Path,
) -> None:
    database = tmp_path / "resume-point-multiple-historical-epochs.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(
            first,
            request_id="create-multiple-historical-resume-epochs",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        _record_test_task_run_safe_point(
            first,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id="multiple-historical-epochs-safe-point",
        )
        point = first.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert point is not None
        safe_point_epoch = point.task_run_epoch
    finally:
        first.close()

    for reopen_index in range(2):
        reopened = Runtime.open(database, config=_config())
        try:
            summary = reopened.task_runs.get(created.run_id)
            process = reopened.process.get(created.root_pid)
            current_point = reopened.store.get_task_run_resume_point(
                created.root_pid,
                complete_only=True,
            )
            assert summary.status is TaskRunStatus.RUNNING
            assert current_point is not None
            assert current_point.task_run_epoch == safe_point_epoch
            assert current_point.task_run_epoch < process.task_run_epoch
            assert process.task_run_epoch == reopened.task_runs.runtime_epoch
            assert TaskRunAction.RUN in summary.allowed_actions
            assert reopened.run_next_process_once() is None
            if reopen_index == 1:
                assert process.task_run_epoch >= safe_point_epoch + 2
        finally:
            reopened.close()


def test_cancel_with_dispatched_unknown_effect_never_reports_cancelled(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cancel-unknown.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        effect = begin_external_effect_intent(
            runtime,
            pid=created.root_pid,
            provider="test-unknown-provider",
            operation="write",
            target="record:unknown",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key="cancel-unknown-once",
        )

        result = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-unknown",
        )

        assert result.status is TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" in {item["kind"] for item in result.blockers}
        assert TaskRunAction.RESUME not in result.allowed_actions
        assert TaskRunAction.RUN not in result.allowed_actions
        assert runtime.process.get(created.root_pid).status is not ProcessStatus.KILLED
        persisted = runtime.store.get_external_effect(effect.effect_id)
        assert persisted is not None
        assert persisted.effect_state == "pending"
        assert persisted.transaction_state == "dispatched"
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        stable = reopened.task_runs.get(created.run_id)
        persisted = reopened.store.get_external_effect(effect.effect_id)
        persisted_run = reopened.store.get_task_run(created.run_id)

        assert stable.status is TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" in {item["kind"] for item in stable.blockers}
        assert stable.completed_at is None
        assert persisted_run is not None and persisted_run.finalized_at is None
        assert persisted is not None
        assert persisted.effect_state == "pending"
        assert persisted.transaction_state == "unknown"
        assert reopened.process.get(created.root_pid).status is not ProcessStatus.KILLED
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_cancel_with_finalized_unknown_effect_never_clears_effect_blocker(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cancel-finalized-unknown.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime)
        root_pid = created.root_pid
        assert root_pid is not None
        effect = record_external_effect(
            runtime.uow.protected_effects,
            pid=root_pid,
            provider="test-finalized-unknown-provider",
            operation="write",
            target="record:finalized-unknown",
            classification=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=False,
            ),
            audit_record=None,
            event=None,
            metadata={"outcome": "unknown_after_provider_return"},
        )

        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-finalized-unknown",
        )

        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        blocker = next(
            item for item in blocked.blockers if item["kind"] == "unknown_effect"
        )
        assert blocker["effect_ids"] == [effect.effect_id]
        assert runtime.process.get(root_pid).status is not ProcessStatus.KILLED
        persisted = runtime.store.get_external_effect(effect.effect_id)
        assert persisted is not None
        assert (persisted.effect_state, persisted.transaction_state) == (
            "finalized",
            "unknown",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        stable = reopened.task_runs.get(created.run_id)
        persisted = reopened.store.get_external_effect(effect.effect_id)

        assert stable.status is TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" in {item["kind"] for item in stable.blockers}
        assert stable.completed_at is None
        assert persisted is not None
        assert (persisted.effect_state, persisted.transaction_state) == (
            "finalized",
            "unknown",
        )
        assert reopened.process.get(root_pid).status is not ProcessStatus.KILLED
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_live_not_started_receipt_rewinds_exact_action_before_explicit_dispatch(
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            effect: object,
            receipt: object,
        ) -> object:
            self.calls += 1
            assert getattr(effect, "effect_id") == unknown.effect_id
            assert receipt == {"provider_request_id": "request-safe-retry"}
            return {
                "state": "not_started",
                "provider_receipt": {
                    "provider_request_id": "request-safe-retry",
                    "dispatch_status": "not_started",
                    "certified": True,
                },
            }

    runtime = Runtime.open(tmp_path / "live-not-started-rewind.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        action = {"action": "discover_skills", "text": "workspace", "limit": 3}
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id="llmcall-live-not-started-rewind",
            action=action,
        )
        binding = _task_run_action_effect_binding(
            pid=root_pid,
            point=claimed_point,
            wrapper=wrapper,
        )
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_live_not_started",
            operation="write",
            target="record:live-not-started",
            state_mutation=True,
            information_flow=False,
            metadata={"task_run_action": binding},
            canonical_args={"value": "once"},
            idempotency_key="live-not-started-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={"provider_request_id": "request-safe-retry"},
        )
        blocked = _mark_test_run_started_and_attention(
            runtime,
            run_id=created.run_id,
            effect_id=unknown.effect_id,
        )
        verifier = ReceiptVerifier()
        setattr(runtime.substrate, "test_live_not_started", verifier)
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-safe-retry"},
            expected_revision=blocked.revision,
            command_id="recover-live-not-started",
        )

        assert recovered.status is TaskRunStatus.PAUSED
        assert recovered.blockers == ()
        assert verifier.calls == 1
        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "failed",
        )
        rewound_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert rewound_point is not None
        assert rewound_point.safe_point_seq == claimed_point.safe_point_seq + 1
        assert rewound_point.last_effect_seq > claimed_point.last_effect_seq
        rewound = runtime.task_runs._decode_pending_resume_payload(rewound_point)
        assert rewound["state"] == "validated"

        resumed = runtime.task_runs.resume(
            created.run_id,
            expected_revision=recovered.revision,
            command_id="resume-after-not-started-receipt",
        )
        assert resumed.status is TaskRunStatus.RUNNING

        class ProviderMustNotRun:
            def complete_action(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError(
                    "exact local action recovery must not ask the LLM Provider again"
                )

        runtime.llm.client = ProviderMustNotRun()
        tool_calls_before = runtime.process.get(root_pid).resource_usage.tool_calls
        continued = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=resumed.revision,
            command_id="run-rewound-not-started-action",
            max_quanta=1,
        )

        assert continued.status is TaskRunStatus.RUNNING
        assert (
            runtime.process.get(root_pid).resource_usage.tool_calls
            == tool_calls_before + 1
        )
        completed_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert completed_point is not None
        assert completed_point.pending_action_payload_id is None
        assert verifier.calls == 1
    finally:
        runtime.close()


def test_live_not_started_receipt_keeps_unbound_action_in_needs_attention(
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            self.calls += 1
            return {
                "state": "not_started",
                "provider_receipt": {
                    "provider_request_id": "request-unbound",
                    "dispatch_status": "not_started",
                    "certified": True,
                },
            }

    runtime = Runtime.open(tmp_path / "live-not-started-unbound.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        action = {"action": "discover_skills", "text": "workspace", "limit": 3}
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id="llmcall-live-not-started-unbound",
            action=action,
        )
        mismatched_binding = _task_run_action_effect_binding(
            pid=root_pid,
            point=claimed_point,
            wrapper=wrapper,
        )
        mismatched_binding["call_id"] = "llmcall-other-action"
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_live_not_started_unbound",
            operation="write",
            target="record:unbound-not-started",
            state_mutation=True,
            information_flow=False,
            metadata={"task_run_action": mismatched_binding},
            canonical_args={"value": "once"},
            idempotency_key="unbound-not-started-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={"provider_request_id": "request-unbound"},
        )
        blocked = _mark_test_run_started_and_attention(
            runtime,
            run_id=created.run_id,
            effect_id=unknown.effect_id,
        )
        verifier = ReceiptVerifier()
        setattr(runtime.substrate, "test_live_not_started_unbound", verifier)
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-unbound"},
            expected_revision=blocked.revision,
            command_id="recover-unbound-not-started",
        )

        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in recovered.blockers} == {
            "pending_action_unreplayable"
        }
        assert TaskRunAction.RUN not in recovered.allowed_actions
        assert TaskRunAction.RESUME not in recovered.allowed_actions
        assert verifier.calls == 1
        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "failed",
        )
        unchanged_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert unchanged_point == claimed_point
        assert runtime.task_runs._decode_pending_resume_payload(unchanged_point)[
            "state"
        ] == "dispatching"
        command = runtime.store.get_task_run_command(
            created.run_id,
            "recover-unbound-not-started",
        )
        assert command is not None
        assert command.result_revision == recovered.revision

        replayed = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-unbound"},
            expected_revision=blocked.revision,
            command_id="recover-unbound-not-started",
        )
        assert replayed == recovered
        assert verifier.calls == 1
    finally:
        runtime.close()


def test_live_not_started_recovery_rolls_back_settlement_and_rewind_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReceiptVerifier:
        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            return {
                "state": "not_started",
                "provider_receipt": {
                    "provider_request_id": "request-atomic-recovery",
                    "dispatch_status": "not_started",
                    "certified": True,
                },
            }

    runtime = Runtime.open(tmp_path / "atomic-not-started-rewind.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id="llmcall-atomic-not-started-rewind",
            action={"action": "discover_skills", "text": "workspace", "limit": 3},
        )
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_atomic_not_started",
            operation="write",
            target="record:atomic-not-started",
            state_mutation=True,
            information_flow=False,
            metadata={
                "task_run_action": _task_run_action_effect_binding(
                    pid=root_pid,
                    point=claimed_point,
                    wrapper=wrapper,
                )
            },
            canonical_args={"value": "once"},
            idempotency_key="atomic-not-started-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={"provider_request_id": "request-atomic-recovery"},
        )
        blocked = _mark_test_run_started_and_attention(
            runtime,
            run_id=created.run_id,
            effect_id=unknown.effect_id,
        )
        setattr(runtime.substrate, "test_atomic_not_started", ReceiptVerifier())
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )

        def fail_after_rewind(*_args: object, **kwargs: object) -> object:
            assert kwargs["not_started_action_rewound"] is True
            raise RuntimeError("injected Run CAS failure after exact rewind")

        monkeypatch.setattr(
            runtime.task_runs,
            "_apply_effect_recovery_settlement",
            fail_after_rewind,
        )
        with pytest.raises(RuntimeError, match="injected Run CAS failure"):
            runtime.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "request-atomic-recovery"},
                expected_revision=blocked.revision,
                command_id="recover-atomic-not-started",
            )

        persisted = runtime.store.get_external_effect(unknown.effect_id)
        assert persisted is not None
        assert (persisted.effect_state, persisted.transaction_state) == (
            "pending",
            "unknown",
        )
        assert runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        ) == claimed_point
        latest = runtime.store.get_task_run(created.run_id)
        assert latest is not None
        assert latest.revision == blocked.revision
        assert latest.status is TaskRunStatus.NEEDS_ATTENTION
        assert runtime.store.get_task_run_command(
            created.run_id,
            "recover-atomic-not-started",
        ) is None
    finally:
        runtime.close()


def test_cancel_recovery_settles_not_started_effect_without_replaying_action(
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            return {
                "state": "not_started",
                "provider_receipt": {
                    "provider_request_id": "request-cancel-recovery",
                    "dispatch_status": "not_started",
                    "certified": True,
                },
            }

    runtime = Runtime.open(tmp_path / "cancel-not-started-recovery.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id="llmcall-cancel-not-started-recovery",
            action={"action": "discover_skills", "text": "workspace", "limit": 3},
        )
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_cancel_not_started",
            operation="write",
            target="record:cancel-not-started",
            state_mutation=True,
            information_flow=False,
            metadata={
                "task_run_action": _task_run_action_effect_binding(
                    pid=root_pid,
                    point=claimed_point,
                    wrapper=wrapper,
                )
            },
            canonical_args={"value": "once"},
            idempotency_key="cancel-not-started-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={"provider_request_id": "request-cancel-recovery"},
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=runtime.task_runs.get(created.run_id).revision,
            command_id="cancel-before-not-started-recovery",
        )
        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert runtime.store.get_task_run(created.run_id).cancel_generation == 1
        setattr(runtime.substrate, "test_cancel_not_started", ReceiptVerifier())
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )
        tool_calls_before = runtime.process.get(root_pid).resource_usage.tool_calls

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-cancel-recovery"},
            expected_revision=blocked.revision,
            command_id="settle-cancel-not-started-recovery",
        )

        assert recovered.status is TaskRunStatus.CANCELLED
        assert recovered.blockers == ()
        assert (
            runtime.process.get(root_pid).resource_usage.tool_calls
            == tool_calls_before
        )
        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "failed",
        )
        retained_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert retained_point == claimed_point
        assert runtime.task_runs._decode_pending_resume_payload(retained_point)[
            "state"
        ] == "dispatching"
    finally:
        runtime.close()


def test_effect_receipt_replay_finishes_locally_after_post_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReceiptVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            _effect: object,
            receipt: object,
        ) -> object:
            self.calls += 1
            assert receipt == {"provider_request_id": "request-crash-once"}
            return {
                "state": "committed",
                "provider_receipt": {
                    "provider_request_id": "request-crash-once",
                    "provider_state": "committed",
                },
            }

    database = tmp_path / "effect-receipt-post-commit-crash.sqlite"
    verifier = ReceiptVerifier()
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_receipt_crash_replay",
            operation="write",
            target="record:receipt-crash-replay",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key="receipt-crash-replay-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider returned before local settlement",
            provider_receipt={"provider_request_id": "request-crash-once"},
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-before-receipt-crash",
        )
        setattr(runtime.substrate, "test_receipt_crash_replay", verifier)
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )

        def crash_before_local_cancel(_run_id: str) -> None:
            raise RuntimeError("injected crash after receipt settlement commit")

        monkeypatch.setattr(
            runtime.task_runs,
            "_continue_cancel_after_effect_settlement",
            crash_before_local_cancel,
        )
        with pytest.raises(RuntimeError, match="after receipt settlement commit"):
            runtime.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "request-crash-once"},
                expected_revision=blocked.revision,
                command_id="recover-receipt-crash",
            )

        command = runtime.store.get_task_run_command(
            created.run_id,
            "recover-receipt-crash",
        )
        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert verifier.calls == 1
        assert command is not None
        assert command.result["settlement_state"] == "pending"
        assert command.result["settlement_kind"] == "effect_receipt"
        assert command.result["effect_id"] == unknown.effect_id
        assert command.result["expected_transaction_state"] == "unknown"
        assert command.result["cancel_generation"] == 1
        assert "request-crash-once" not in json.dumps(command.result)
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "committed",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        current = reopened.store.get_task_run(created.run_id)
        assert current is not None
        before_conflict = (
            current,
            reopened.store.get_task_run_command(
                created.run_id,
                "recover-receipt-crash",
            ),
            reopened.store.get_external_effect(unknown.effect_id),
        )
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            reopened.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "request-crash-once"},
                expected_revision=current.revision,
                command_id="recover-receipt-crash",
            )
        assert (
            reopened.store.get_task_run(created.run_id),
            reopened.store.get_task_run_command(
                created.run_id,
                "recover-receipt-crash",
            ),
            reopened.store.get_external_effect(unknown.effect_id),
        ) == before_conflict
        assert verifier.calls == 1

        replayed = reopened.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-crash-once"},
            expected_revision=blocked.revision,
            command_id="recover-receipt-crash",
        )

        latest = reopened.task_runs.get(created.run_id)
        command = reopened.store.get_task_run_command(
            created.run_id,
            "recover-receipt-crash",
        )
        assert verifier.calls == 1
        assert replayed == latest
        assert replayed.status is TaskRunStatus.CANCELLED
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == latest.revision
    finally:
        reopened.close()


@pytest.mark.parametrize("reopen_before_replay", (False, True))
@pytest.mark.parametrize("tamper", ("transition_seq", "cancel_generation"))
def test_terminal_effect_receipt_tamper_never_completes_pending_command(
    reopen_before_replay: bool,
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReceiptVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            self.calls += 1
            return {
                "state": "committed",
                "provider_receipt": {
                    "provider_request_id": "terminal-tamper",
                    "provider_state": "committed",
                },
            }

    database = tmp_path / (
        f"terminal-effect-tamper-{tamper}-{reopen_before_replay}.sqlite"
    )
    first = Runtime.open(database, config=_config())
    verifier = ReceiptVerifier()
    try:
        created = _create(
            first,
            request_id=(
                f"create-terminal-effect-tamper-{tamper}-{reopen_before_replay}"
            ),
        )
        assert created.root_pid is not None
        provider_name = (
            f"test_terminal_effect_tamper_{tamper}_{reopen_before_replay}"
        )
        setattr(first.substrate, provider_name, verifier)
        dispatched = begin_external_effect_intent(
            first,
            pid=created.root_pid,
            provider=provider_name,
            operation="write",
            target="record:terminal-effect-tamper",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key=(
                f"terminal-effect-tamper-{tamper}-{reopen_before_replay}"
            ),
        )
        unknown = mark_external_effect_unknown(
            first.uow.protected_effects,
            dispatched.effect_id,
            reason="provider result awaits Host receipt",
            provider_receipt={"provider_request_id": "terminal-tamper"},
        )
        blocked = first.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id=(
                f"cancel-terminal-effect-tamper-{tamper}-{reopen_before_replay}"
            ),
        )
        option = next(
            item
            for item in first.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )
        command_id = (
            f"recover-terminal-effect-tamper-{tamper}-{reopen_before_replay}"
        )
        original_continue = first.task_runs._continue_cancel_after_effect_settlement

        def crash_before_local_cancel(_run_id: str) -> None:
            raise RuntimeError("fault after exact effect evidence commit")

        monkeypatch.setattr(
            first.task_runs,
            "_continue_cancel_after_effect_settlement",
            crash_before_local_cancel,
        )
        with pytest.raises(RuntimeError, match="exact effect evidence commit"):
            first.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "terminal-tamper"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        monkeypatch.setattr(
            first.task_runs,
            "_continue_cancel_after_effect_settlement",
            original_continue,
        )
        pending = first.store.get_task_run_command(created.run_id, command_id)
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"
        assert type(pending.result["settlement_transition_seq"]) is int
        original_continue(created.run_id)
        terminal = first.task_runs._project(
            first.store.get_task_run(created.run_id),
            allow_finalize=True,
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        assert terminal.payloads_purged_at is not None
        result = json.loads(json.dumps(pending.result))
        if tamper == "transition_seq":
            result["settlement_transition_seq"] += 1
        else:
            result["cancel_generation"] += 1
        first.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
    except BaseException:
        first.close()
        raise

    if reopen_before_replay:
        first.close()
        selected = Runtime.open(database, config=_config())
    else:
        selected = first
    try:
        before = (
            selected.store.get_task_run(created.run_id),
            selected.process.get(created.root_pid),
            selected.store.get_external_effect(unknown.effect_id),
            selected.store.get_task_run_command(created.run_id, command_id),
        )
        with pytest.raises(ValidationError):
            selected.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "terminal-tamper"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        after_command = selected.store.get_task_run_command(
            created.run_id,
            command_id,
        )
        assert (
            selected.store.get_task_run(created.run_id),
            selected.process.get(created.root_pid),
            selected.store.get_external_effect(unknown.effect_id),
            after_command,
        ) == before
        assert after_command is not None
        assert after_command.result["settlement_state"] == "pending"
        assert selected.run_next_process_once() is None
        assert verifier.calls == 1
    finally:
        selected.close()


def test_stale_runtime_cannot_complete_terminal_effect_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReceiptVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            self.calls += 1
            return {
                "state": "committed",
                "provider_receipt": {
                    "provider_request_id": "stale-terminal-effect",
                    "provider_state": "committed",
                },
            }

    runtime = Runtime.open(
        tmp_path / "stale-terminal-effect-receipt.sqlite",
        config=_config(),
    )
    verifier = ReceiptVerifier()
    command_id = "stale-terminal-effect-receipt"
    try:
        created = _create(
            runtime,
            request_id="create-stale-terminal-effect-receipt",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        provider_name = "test_stale_terminal_effect_receipt"
        setattr(runtime.substrate, provider_name, verifier)
        dispatched = begin_external_effect_intent(
            runtime,
            pid=created.root_pid,
            provider=provider_name,
            operation="write",
            target="record:stale-terminal-effect",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key="stale-terminal-effect-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider result awaits Host receipt",
            provider_receipt={"provider_request_id": "stale-terminal-effect"},
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-before-stale-terminal-effect",
        )
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )
        original_continue = runtime.task_runs._continue_cancel_after_effect_settlement

        def crash_before_local_cancel(_run_id: str) -> None:
            raise RuntimeError("fault after stale effect evidence commit")

        monkeypatch.setattr(
            runtime.task_runs,
            "_continue_cancel_after_effect_settlement",
            crash_before_local_cancel,
        )
        with pytest.raises(RuntimeError, match="stale effect evidence commit"):
            runtime.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "stale-terminal-effect"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        monkeypatch.setattr(
            runtime.task_runs,
            "_continue_cancel_after_effect_settlement",
            original_continue,
        )
        original_continue(created.run_id)
        terminal = runtime.task_runs._project(
            runtime.store.get_task_run(created.run_id),
            allow_finalize=True,
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        pending = runtime.store.get_task_run_command(created.run_id, command_id)
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"
        old_epoch = runtime.task_runs.runtime_epoch
        new_epoch = runtime.store.claim_runtime_epoch("new-runtime-after-effect")
        assert new_epoch > old_epoch
        rebound = runtime.store.claim_terminal_task_run_epoch(
            created.run_id,
            terminal.revision,
            new_epoch,
        )
        before = (
            rebound,
            runtime.process.get(created.root_pid),
            runtime.store.get_external_effect(unknown.effect_id),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        )

        with pytest.raises(TaskRunRevisionConflict):
            runtime.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "stale-terminal-effect"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_external_effect(unknown.effect_id),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        ) == before
        assert before[3].result["settlement_state"] == "pending"
        assert verifier.calls == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("provider_state", ["committed", "failed", "compensated"])
def test_cancel_recovery_accepts_terminal_effect_truth_without_local_replay(
    provider_state: str,
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            return {
                "state": provider_state,
                "provider_receipt": {
                    "provider_request_id": f"request-cancel-{provider_state}",
                    "provider_state": provider_state,
                },
            }

    database = tmp_path / f"cancel-{provider_state}-recovery.sqlite"
    provider_name = f"test_cancel_{provider_state}"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id=f"llmcall-cancel-{provider_state}-recovery",
            action={"action": "discover_skills", "text": "workspace", "limit": 3},
        )
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider=provider_name,
            operation="write",
            target=f"record:cancel-{provider_state}",
            state_mutation=True,
            information_flow=False,
            metadata={
                "task_run_action": _task_run_action_effect_binding(
                    pid=root_pid,
                    point=claimed_point,
                    wrapper=wrapper,
                )
            },
            canonical_args={"value": "once"},
            idempotency_key=f"cancel-{provider_state}-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={
                "provider_request_id": f"request-cancel-{provider_state}"
            },
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=runtime.task_runs.get(created.run_id).revision,
            command_id=f"cancel-before-{provider_state}-recovery",
        )
        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        setattr(runtime.substrate, provider_name, ReceiptVerifier())
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )
        tool_calls_before = runtime.process.get(root_pid).resource_usage.tool_calls

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={
                "provider_request_id": f"request-cancel-{provider_state}"
            },
            expected_revision=blocked.revision,
            command_id=f"settle-cancel-{provider_state}-recovery",
        )

        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert recovered.status is TaskRunStatus.CANCELLED
        assert recovered.blockers == ()
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            provider_state,
        )
        assert (
            runtime.process.get(root_pid).resource_usage.tool_calls
            == tool_calls_before
        )
        retained_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert retained_point == claimed_point
        assert runtime.task_runs._decode_pending_resume_payload(retained_point)[
            "state"
        ] == "dispatching"
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        stable = reopened.task_runs.get(created.run_id)
        settled = reopened.store.get_external_effect(unknown.effect_id)

        assert stable.status is TaskRunStatus.CANCELLED
        assert stable.blockers == ()
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            provider_state,
        )
        assert reopened.process.get(root_pid).resource_usage.tool_calls == tool_calls_before
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_cancel_recovery_does_not_apply_unbound_settlement_to_pending_action(
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def verify_external_effect_receipt(
            self,
            _effect: object,
            _receipt: object,
        ) -> object:
            return {
                "state": "committed",
                "provider_receipt": {
                    "provider_request_id": "request-cancel-unbound"
                },
            }

    runtime = Runtime.open(tmp_path / "cancel-unbound-recovery.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        _manifest, claimed_point, wrapper = _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=root_pid,
            call_id="llmcall-cancel-unbound-recovery",
            action={"action": "discover_skills", "text": "workspace", "limit": 3},
        )
        mismatched_binding = _task_run_action_effect_binding(
            pid=root_pid,
            point=claimed_point,
            wrapper=wrapper,
        )
        mismatched_binding["call_id"] = "llmcall-unrelated-action"
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test_cancel_unbound",
            operation="write",
            target="record:cancel-unbound",
            state_mutation=True,
            information_flow=False,
            metadata={"task_run_action": mismatched_binding},
            canonical_args={"value": "once"},
            idempotency_key="cancel-unbound-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider outcome unavailable before Host receipt",
            provider_receipt={"provider_request_id": "request-cancel-unbound"},
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=runtime.task_runs.get(created.run_id).revision,
            command_id="cancel-before-unbound-recovery",
        )
        setattr(runtime.substrate, "test_cancel_unbound", ReceiptVerifier())
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=option.option_id,
            receipt={"provider_request_id": "request-cancel-unbound"},
            expected_revision=blocked.revision,
            command_id="settle-cancel-unbound-recovery",
        )

        settled = runtime.store.get_external_effect(unknown.effect_id)
        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        assert "pending_action_unreplayable" in {
            item["kind"] for item in recovered.blockers
        }
        assert recovered.completed_at is None
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "committed",
        )
        assert runtime.process.get(root_pid).status is ProcessStatus.KILLED
    finally:
        runtime.close()


def test_changed_effect_fence_rejects_committed_effect_with_not_started_flags(
    tmp_path: Path,
) -> None:
    database = tmp_path / "contradictory-not-started-effect.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.store.insert_llm_call(
            _full_llm_call(
                "llmcall-contradictory-not-started",
                root_pid,
                "CONTRADICTORY_EFFECT",
            )
        )
        runtime.task_runs.record_validated_transcript(
            pid=root_pid,
            call_id="llmcall-contradictory-not-started",
            action_manifest=validated_action_manifest(
                [{"action": "discover_skills", "text": "workspace", "limit": 3}],
                call_id="llmcall-contradictory-not-started",
                parallel_tool_calls=False,
                host_auto_wait=False,
                tool_call_count=1,
                data_labels={},
            ),
            context_generation=runtime.store.get_llm_context_generation(root_pid),
        )
        point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert point is not None
        effect = record_external_effect(
            runtime.uow.protected_effects,
            pid=root_pid,
            provider="contradictory-provider",
            operation="write",
            target="record:contradictory",
            classification=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=False,
            ),
            audit_record=None,
            event=None,
            metadata={
                "provider_reconciliation_state": "not_started",
                "certified_not_started": True,
                "provider_receipt": {
                    "dispatch_status": "not_started",
                    "certified": True,
                },
            },
        )
        assert effect.effect_state == "finalized"
        assert effect.transaction_state == "committed"
        current = runtime.store.get_task_run(created.run_id)
        assert current is not None
        now = utc_now()
        runtime.store.update_task_run_cas(
            created.run_id,
            current.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": now,
                "updated_at": now,
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)

        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        blocker = next(
            item for item in recovered.blockers if item["kind"] == "unknown_effect"
        )
        assert blocker["effect_ids"] == [effect.effect_id]
        assert TaskRunAction.RUN not in recovered.allowed_actions
        assert TaskRunAction.RESUME not in recovered.allowed_actions
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        reopened.close()


def test_recover_terminate_unknown_effect_stops_execution_without_settlement(
    tmp_path: Path,
) -> None:
    class ReceiptVerifier:
        def verify_external_effect_receipt(self, *_args: object) -> object:
            raise AssertionError("terminate_run must not reconcile the provider effect")

    runtime = Runtime.open(tmp_path / "recover-terminate-unknown.sqlite", config=_config())
    try:
        created = _create(runtime)
        root_pid = created.root_pid
        assert root_pid is not None
        provider_name = "test_unknown_recovery"
        setattr(runtime.substrate, provider_name, ReceiptVerifier())
        dispatched = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider=provider_name,
            operation="write",
            target="record:unknown-recovery",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "exactly-once"},
            idempotency_key="unknown-recovery-once",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="provider acknowledged dispatch but outcome is unavailable",
            provider_receipt={
                "provider_request_id": "request-unknown-recovery",
                "opaque_status": "outcome-unavailable",
            },
        )
        blocked = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-before-terminate-recovery",
        )
        payloads_before = runtime.store.list_task_run_payloads(created.run_id)
        receipt_before = dict(unknown.provider_receipt)

        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        options = runtime.task_runs.recovery_options(created.run_id)
        terminate = next(item for item in options if item.option_id == "terminate_run")
        assert terminate.kind == "terminalize"
        assert terminate.label == "Stop old Run execution without settling effects"
        assert any(item.kind == "effect_receipt" for item in options)

        recovered = runtime.task_runs.recover(
            created.run_id,
            option_id=terminate.option_id,
            expected_revision=blocked.revision,
            command_id="terminate-unknown-recovery",
        )

        persisted_effect = runtime.store.get_external_effect(dispatched.effect_id)
        persisted_run = runtime.store.get_task_run(created.run_id)
        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" in {item["kind"] for item in recovered.blockers}
        assert runtime.process.get(root_pid).status is ProcessStatus.KILLED
        assert persisted_effect is not None
        assert (persisted_effect.effect_state, persisted_effect.transaction_state) == (
            "pending",
            "unknown",
        )
        assert persisted_effect.provider_receipt == receipt_before
        assert persisted_run is not None
        assert persisted_run.status is TaskRunStatus.NEEDS_ATTENTION
        assert persisted_run.completed_at is None
        assert persisted_run.finalized_at is None
        assert persisted_run.payloads_purged_at is None
        assert runtime.store.list_task_run_payloads(created.run_id) == payloads_before
        assert all(
            payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
            for payload in payloads_before
        )
        remaining_options = runtime.task_runs.recovery_options(created.run_id)
        assert "terminate_run" in {item.option_id for item in remaining_options}
        assert any(item.kind == "effect_receipt" for item in remaining_options)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "effect_request_with_terminalize_receipt",
        "effect_request_with_linked_fields",
        "effect_request_with_wrong_binding",
    ),
)
def test_recover_replay_rejects_cross_variant_receipt_without_settlement(
    corruption: str,
    tmp_path: Path,
) -> None:
    class ProviderMustNotVerify:
        def __init__(self) -> None:
            self.calls = 0

        def verify_external_effect_receipt(
            self,
            *_args: object,
        ) -> object:
            self.calls += 1
            raise AssertionError("corrupt replay must not verify or settle an effect")

    runtime = Runtime.open(
        tmp_path / f"recover-cross-variant-{corruption}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-recover-cross-variant-{corruption}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        provider_name = f"test_cross_variant_{corruption}"
        verifier = ProviderMustNotVerify()
        setattr(runtime.substrate, provider_name, verifier)
        dispatched = begin_external_effect_intent(
            runtime,
            pid=created.root_pid,
            provider=provider_name,
            operation="write",
            target="record:cross-variant",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key=f"cross-variant-{corruption}",
        )
        unknown = mark_external_effect_unknown(
            runtime.uow.protected_effects,
            dispatched.effect_id,
            reason="cross-variant test unknown",
            provider_receipt={"provider_request_id": "cross-variant"},
        )
        blocked = _mark_test_run_started_and_attention(
            runtime,
            run_id=created.run_id,
            effect_id=unknown.effect_id,
        )
        option = next(
            item
            for item in runtime.task_runs.recovery_options(created.run_id)
            if item.kind == "effect_receipt"
        )
        request = {
            "expected_revision": blocked.revision,
            "option_id": option.option_id,
            "receipt": {"provider_request_id": "cross-variant"},
        }
        source = runtime.store.get_task_run(created.run_id)
        assert source is not None
        if corruption == "effect_request_with_terminalize_receipt":
            result = {
                "settlement_state": "pending",
                "settlement_kind": "terminalize",
                "cancel_generation": 1,
            }
        else:
            result = {
                "settlement_state": "pending",
                "settlement_kind": "effect_receipt",
                "effect_id": (
                    "effect-another-binding"
                    if corruption == "effect_request_with_wrong_binding"
                    else unknown.effect_id
                ),
                "expected_transaction_state": "unknown",
                "cancel_generation": source.cancel_generation,
                "admission_runtime_epoch": option.runtime_epoch,
                "settlement_transition_seq": 1,
                "settlement_audit_record_id": "audit-cross-variant",
            }
            if corruption == "effect_request_with_linked_fields":
                result.update(
                    run_id=created.run_id,
                    revision=source.revision,
                    new_run_id="run-cross-variant-target",
                    new_run_summary={"run_id": "run-cross-variant-target"},
                )
        command_id = f"recover-cross-variant-{corruption}"
        runtime.task_runs._record_command(
            source,
            command_id,
            "recover",
            request,
            result=result,
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_external_effect(unknown.effect_id),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_runs(limit=10).records),
        )

        with pytest.raises(ValidationError):
            runtime.task_runs.recover(
                created.run_id,
                option_id=option.option_id,
                receipt={"provider_request_id": "cross-variant"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_external_effect(unknown.effect_id),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_runs(limit=10).records),
        ) == before
        assert verifier.calls == 0
        assert runtime.run_next_process_once() is None
    finally:
        runtime.close()


def test_expired_deadline_persists_cancel_intent_and_waits_for_effect_settlement(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "deadline-effect.sqlite", config=_config())
    try:
        deadline_at = (datetime.now(timezone.utc) + timedelta(seconds=0.2)).isoformat()
        created = _create(runtime, deadline_at=deadline_at)
        root_pid = created.root_pid
        assert root_pid is not None
        effect = begin_external_effect_intent(
            runtime,
            pid=root_pid,
            provider="test-deadline-provider",
            operation="write",
            target="record:deadline",
            state_mutation=True,
            information_flow=False,
            canonical_args={"value": "once"},
            idempotency_key="deadline-effect-once",
        )

        remaining = datetime.fromisoformat(deadline_at) - datetime.now(timezone.utc)
        time.sleep(max(0.0, remaining.total_seconds()) + 0.05)

        assert runtime.task_runs.list(
            statuses=(TaskRunStatus.QUEUED,),
            limit=10,
        ).records == ()
        attention_page = runtime.task_runs.list(
            statuses=(TaskRunStatus.NEEDS_ATTENTION,),
            limit=10,
        )
        assert len(attention_page.records) == 1
        blocked = attention_page.records[0]

        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in blocked.blockers} >= {
            "deadline_reached",
            "unknown_effect",
        }
        assert TaskRunAction.RUN not in blocked.allowed_actions
        assert TaskRunAction.RESUME not in blocked.allowed_actions
        persisted_intent = runtime.store.get_task_run(created.run_id)
        assert persisted_intent is not None
        assert persisted_intent.cancel_generation == 1
        stable_revision = persisted_intent.revision
        assert runtime.task_runs.get(created.run_id) == blocked
        assert runtime.task_runs.list(
            statuses=(TaskRunStatus.NEEDS_ATTENTION,),
            limit=10,
        ).records == (blocked,)
        repeated_intent = runtime.store.get_task_run(created.run_id)
        assert repeated_intent is not None
        assert repeated_intent.revision == stable_revision
        assert repeated_intent.cancel_generation == 1
        control_transitions = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label
            in {
                "deadline cancellation intent persisted",
                "manual attention required",
            }
        ]
        assert [
            (item.label, item.metadata)
            for item in control_transitions
        ] == [
            (
                "deadline cancellation intent persisted",
                {"from": "queued", "to": "cancelling"},
            ),
            (
                "manual attention required",
                {"from": "cancelling", "to": "needs_attention"},
            ),
        ]
        assert runtime.process.get(root_pid).status is not ProcessStatus.KILLED
        assert runtime.process.get(root_pid).resource_usage.llm_calls == 0
        assert runtime.run_next_process_once() is None
        pending = runtime.store.get_external_effect(effect.effect_id)
        assert pending is not None
        assert (pending.effect_state, pending.transaction_state) == (
            "pending",
            "dispatched",
        )

        record_external_effect(
            runtime.uow.protected_effects,
            pid=root_pid,
            provider="test-deadline-provider",
            operation="write",
            target="record:deadline",
            classification=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=False,
            ),
            audit_record=None,
            event=None,
            metadata={"provider_receipt": {"revision": "settled-once"}},
            intent_effect_id=effect.effect_id,
        )

        terminal = runtime.task_runs.get(created.run_id)
        settled = runtime.store.get_external_effect(effect.effect_id)
        assert terminal.status is TaskRunStatus.CANCELLED
        assert [item["kind"] for item in terminal.blockers] == ["deadline_reached"]
        assert settled is not None
        assert (settled.effect_state, settled.transaction_state) == (
            "finalized",
            "committed",
        )
        assert (
            external_effect_payload_retention_tier(settled)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert "settled-once" not in json.dumps(settled.provider_receipt)
        assert runtime.process.get(root_pid).status is ProcessStatus.KILLED
        assert runtime.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        runtime.close()


def test_expired_run_command_rolls_back_intent_when_receipt_cannot_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "deadline-command-atomic.sqlite", config=_config())
    try:
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(milliseconds=80)
        ).isoformat()
        created = _create(
            runtime,
            request_id="create-deadline-command-atomic",
            deadline_at=deadline_at,
        )
        time.sleep(0.12)
        original_record_command = runtime.task_runs._record_command

        def fail_receipt(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("injected deadline command receipt failure")

        monkeypatch.setattr(runtime.task_runs, "_record_command", fail_receipt)
        with pytest.raises(RuntimeError, match="receipt failure"):
            runtime.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=created.revision,
                command_id="run-expired-atomically",
            )

        rolled_back = runtime.store.get_task_run(created.run_id)
        assert rolled_back is not None
        assert rolled_back.status is TaskRunStatus.QUEUED
        assert rolled_back.revision == created.revision
        assert rolled_back.cancel_generation == 0
        assert (
            runtime.store.get_task_run_command(
                created.run_id,
                "run-expired-atomically",
            )
            is None
        )
        assert not any(
            item.label == "deadline cancellation intent persisted"
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
        )

        monkeypatch.setattr(
            runtime.task_runs,
            "_record_command",
            original_record_command,
        )
        retried = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-expired-atomically",
        )
        command = runtime.store.get_task_run_command(
            created.run_id,
            "run-expired-atomically",
        )
        assert retried.status is TaskRunStatus.CANCELLED
        assert command is not None
        assert command.result_revision == retried.revision
    finally:
        runtime.close()


def test_pending_deadline_receipt_converges_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "deadline-command-pending.sqlite", config=_config())
    try:
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(milliseconds=80)
        ).isoformat()
        created = _create(
            runtime,
            request_id="create-deadline-command-pending",
            deadline_at=deadline_at,
        )
        time.sleep(0.12)
        original_unsettled = runtime.task_runs._unsettled_effects
        failed_once = False

        def fail_after_intent(run_id: str) -> list[object]:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("injected crash after deadline intent")
            return list(original_unsettled(run_id))

        monkeypatch.setattr(
            runtime.task_runs,
            "_unsettled_effects",
            fail_after_intent,
        )
        with pytest.raises(RuntimeError, match="after deadline intent"):
            runtime.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=created.revision,
                command_id="run-expired-pending",
            )

        cancelling = runtime.store.get_task_run(created.run_id)
        command = runtime.store.get_task_run_command(
            created.run_id,
            "run-expired-pending",
        )
        assert cancelling is not None
        assert cancelling.status is TaskRunStatus.CANCELLING
        assert command is not None
        assert command.result["settlement_state"] == "pending"
        assert command.result["settlement_kind"] == "deadline"

        before_conflict = (
            cancelling,
            command,
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
        )
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            runtime.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=cancelling.revision,
                command_id="run-expired-pending",
            )
        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.store.get_task_run_command(
                created.run_id,
                "run-expired-pending",
            ),
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
        ) == before_conflict

        replayed = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-expired-pending",
        )
        command = runtime.store.get_task_run_command(
            created.run_id,
            "run-expired-pending",
        )
        assert replayed.status is TaskRunStatus.CANCELLED
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result["settlement_kind"] == "deadline"
        assert command.result_revision == replayed.revision
        assert runtime.process.get(created.root_pid or "").resource_usage.llm_calls == 0
    finally:
        runtime.close()


def test_deadline_intent_retries_a_competing_pre_intent_settlement_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "deadline-initial-cas-race.sqlite", config=_config())
    try:
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=0.2)
        ).isoformat()
        created = _create(runtime, deadline_at=deadline_at)
        original_transaction = runtime.task_runs._uow.transaction
        raced = False

        @contextmanager
        def settlement_races_before_intent(*args: object, **kwargs: object):
            nonlocal raced
            if not raced:
                raced = True
                current = runtime.store.get_task_run(created.run_id)
                assert current is not None
                runtime.store.update_task_run_cas(
                    current.run_id,
                    current.revision,
                    updates={
                        "step_count": current.step_count + 1,
                        "updated_at": utc_now(),
                    },
                    expected_runtime_epoch=runtime.task_runs.runtime_epoch,
                )
            with original_transaction(*args, **kwargs) as transaction:
                yield transaction

        monkeypatch.setattr(
            runtime.task_runs._uow,
            "transaction",
            settlement_races_before_intent,
        )
        remaining = datetime.fromisoformat(deadline_at) - datetime.now(timezone.utc)
        time.sleep(max(0.0, remaining.total_seconds()) + 0.05)

        terminal = runtime.task_runs.get(created.run_id)
        persisted = runtime.store.get_task_run(created.run_id)

        assert raced is True
        assert terminal.status is TaskRunStatus.CANCELLED
        assert persisted is not None and persisted.cancel_generation == 1
        assert sum(
            item.label == "deadline cancellation intent persisted"
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
        ) == 1
    finally:
        runtime.close()


def test_absolute_deadline_keeps_elapsing_during_wait_and_runtime_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "deadline-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=0.3)
        ).isoformat()
        created = _create(first, deadline_at=deadline_at)
        root_pid = created.root_pid
        assert root_pid is not None
        first_epoch = first.task_runs.runtime_epoch
        persisted_deadline = first.store.get_task_run(created.run_id)
        assert persisted_deadline is not None
        assert persisted_deadline.deadline_at == TaskRunSpecV1(
            goal="deadline normalization",
            display_title="Deadline normalization",
            deadline_at=deadline_at,
        ).deadline_at
        observed = first.task_runs.wait(
            created.run_id,
            after_revision=created.revision,
            timeout=0.02,
        )
        assert observed.status is TaskRunStatus.QUEUED
        assert first.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        first.close()

    remaining = datetime.fromisoformat(deadline_at) - datetime.now(timezone.utc)
    time.sleep(max(0.0, remaining.total_seconds()) + 0.05)

    reopened = Runtime.open(database, config=_config())
    try:
        summary = reopened.task_runs.get(created.run_id)
        persisted = reopened.store.get_task_run(created.run_id)
        assert persisted is not None
        assert reopened.task_runs.runtime_epoch > first_epoch
        assert persisted.deadline_at == persisted_deadline.deadline_at
        assert persisted.cancel_generation == 1
        assert summary.status is TaskRunStatus.CANCELLED
        assert summary.completed_at is not None
        assert persisted.deadline_at is not None
        assert datetime.now(timezone.utc) >= datetime.fromisoformat(
            persisted.deadline_at
        )
        assert reopened.process.get(root_pid).status is ProcessStatus.KILLED
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_follow_up_is_durable_requirement_and_run_bound_interrupt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "follow-up.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        followed = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "also verify receipts"},
            expected_revision=created.revision,
            command_id="follow-up-1",
        )
        assert followed.requirement_count == 2
        messages = runtime.store.list_process_messages(created.root_pid)
        assert len(messages) == 1
        assert messages[0].metadata["task_run_id"] == created.run_id
        assert messages[0].payload["run_id"] == created.run_id
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        requirements = reopened.store.list_task_run_requirements(created.run_id)
        messages = reopened.store.list_process_messages(created.root_pid)
        assert len(requirements) == 2
        assert len(messages) == 1
        assert messages[0].metadata["task_run_id"] == created.run_id
    finally:
        reopened.close()


def test_concurrent_follow_ups_with_one_revision_commit_exactly_one_atomic_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "follow-up-race.sqlite", config=_config())
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        barrier = threading.Barrier(2)
        original_require_payload_bound = runtime.task_runs._require_payload_bound

        def synchronize_after_local_validation(payload):
            original_require_payload_bound(payload)
            barrier.wait(timeout=5)

        monkeypatch.setattr(
            runtime.task_runs,
            "_require_payload_bound",
            synchronize_after_local_validation,
        )

        def append_follow_up(index: int):
            return runtime.task_runs.follow_up(
                created.run_id,
                {"request": f"concurrent-{index}"},
                expected_revision=created.revision,
                command_id=f"concurrent-follow-up-{index}",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(append_follow_up, index): index
                for index in range(2)
            }
            successes = []
            failures: list[tuple[int, BaseException]] = []
            for future, index in futures.items():
                try:
                    successes.append((index, future.result(timeout=10)))
                except BaseException as exc:  # assert the public error below
                    failures.append((index, exc))

        assert len(successes) == 1
        assert len(failures) == 1
        winning_index, winning_summary = successes[0]
        losing_index, conflict = failures[0]
        assert isinstance(conflict, TaskRunRevisionConflict)
        assert not isinstance(conflict, sqlite3.IntegrityError)
        assert getattr(conflict, "expected_revision") == created.revision
        assert getattr(conflict, "actual_revision") == winning_summary.revision

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        payloads = runtime.store.list_task_run_payloads(created.run_id)
        messages = runtime.store.list_process_messages(created.root_pid)
        assert [item.ordinal for item in requirements] == [0, 1]
        assert len(payloads) == 2
        assert len(messages) == 1
        assert messages[0].correlation_id == requirements[1].requirement_id
        assert runtime.store.get_task_run_command(
            created.run_id,
            f"concurrent-follow-up-{winning_index}",
        ) is not None
        assert runtime.store.get_task_run_command(
            created.run_id,
            f"concurrent-follow-up-{losing_index}",
        ) is None
        follow_up_ledger = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "follow-up appended"
        ]
        assert len(follow_up_ledger) == 1
        assert follow_up_ledger[0].requirement_id == requirements[1].requirement_id
    finally:
        runtime.close()


def test_follow_up_hard_limit_is_exact_zero_write_and_durable_across_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "follow-up-hard-limit.sqlite"
    base_config = _config()
    config = replace(
        base_config,
        task_runs=replace(
            base_config.task_runs,
            recovery_page_size=2,
            recovery_page_hard_limit=2,
        ),
    )
    runtime = Runtime.open(database, config=config)
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        original_query = runtime.store._query
        requirement_queries: list[tuple[str, tuple[object, ...]]] = []

        def tracked_query(sql: str, params: object = ()) -> list[object]:
            selected_params = tuple(params)  # type: ignore[arg-type]
            if "FROM task_run_requirements" in sql and " LIMIT ?" in sql:
                requirement_queries.append((sql, selected_params))
            return original_query(sql, selected_params)

        monkeypatch.setattr(runtime.store, "_query", tracked_query)
        at_limit = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "exact boundary"},
            expected_revision=created.revision,
            command_id="follow-up-at-hard-limit",
        )
        assert at_limit.requirement_count == 2
        assert len(requirement_queries) == 1
        assert "ORDER BY ordinal, requirement_id COLLATE BINARY LIMIT ?" in (
            requirement_queries[0][0]
        )
        assert requirement_queries[0][1][-1] == 2
        assert [
            item.ordinal
            for item in runtime.store.list_task_run_requirements(created.run_id)
        ] == [0, 1]

        mutation_calls: list[str] = []

        def reject_unexpected_write(name: str):
            def unexpected_write(*args: object, **kwargs: object) -> None:
                mutation_calls.append(name)
                raise AssertionError(f"hard-limit rejection attempted {name}")

            return unexpected_write

        for name in (
            "insert_task_run_payload",
            "insert_task_run_requirement",
            "update_task_run_cas",
            "append_task_run_ledger_item",
            "insert_task_run_command",
        ):
            monkeypatch.setattr(
                runtime.store,
                name,
                reject_unexpected_write(name),
            )
        monkeypatch.setattr(
            runtime.task_runs,
            "_post_follow_up_message",
            reject_unexpected_write("_post_follow_up_message"),
        )

        replayed = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "exact boundary"},
            expected_revision=created.revision,
            command_id="follow-up-at-hard-limit",
        )
        assert replayed == at_limit
        with pytest.raises(TaskRunRevisionConflict, match="different request"):
            runtime.task_runs.follow_up(
                created.run_id,
                {"request": "different request under a used command id"},
                expected_revision=at_limit.revision,
                command_id="follow-up-at-hard-limit",
            )
        with pytest.raises(ValidationError, match="recovery_page_hard_limit"):
            runtime.task_runs.follow_up(
                created.run_id,
                {"request": "one beyond boundary"},
                expected_revision=at_limit.revision,
                command_id="follow-up-beyond-hard-limit",
            )
        assert mutation_calls == []
        assert len(requirement_queries) == 1
        assert runtime.task_runs.get(created.run_id) == at_limit
        assert len(runtime.store.list_task_run_payloads(created.run_id)) == 2
        assert len(runtime.store.list_task_run_requirements(created.run_id)) == 2
        assert len(runtime.store.list_process_messages(created.root_pid)) == 1
        assert runtime.store.get_task_run_command(
            created.run_id,
            "follow-up-beyond-hard-limit",
        ) is None
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=config)
    try:
        durable = reopened.task_runs.get(created.run_id)
        assert durable.revision > at_limit.revision
        assert durable.status is at_limit.status
        assert durable.requirement_count == at_limit.requirement_count
        assert (
            durable.satisfied_requirement_count
            == at_limit.satisfied_requirement_count
        )
        with pytest.raises(ValidationError, match="recovery_page_hard_limit"):
            reopened.task_runs.follow_up(
                created.run_id,
                {"request": "still beyond boundary after reopen"},
                expected_revision=durable.revision,
                command_id="follow-up-beyond-hard-limit-after-reopen",
            )
        assert reopened.task_runs.get(created.run_id) == durable
        assert len(reopened.store.list_task_run_payloads(created.run_id)) == 2
        assert len(reopened.store.list_task_run_requirements(created.run_id)) == 2
        assert len(reopened.store.list_process_messages(created.root_pid)) == 1
        assert reopened.store.get_task_run_command(
            created.run_id,
            "follow-up-beyond-hard-limit-after-reopen",
        ) is None
    finally:
        reopened.close()


def test_terminal_purge_is_run_scoped_and_removes_every_resumable_payload(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "terminal-purge.sqlite", config=_config())
    try:
        created = _create(runtime)
        root_pid = created.root_pid
        assert root_pid is not None
        unrelated_pid = runtime.process.spawn(goal="unrelated process")
        followed = runtime.task_runs.follow_up(
            created.run_id,
            {"secret": "FOLLOW_UP_SECRET"},
            expected_revision=created.revision,
            command_id="follow-up-before-purge",
        )
        runtime.messages.post(
            sender="host",
            recipient_pid=root_pid,
            subject="run-bound private input",
            body="RUN_BOUND_MESSAGE_SECRET",
        )
        runtime.messages.post(
            sender="host",
            recipient_pid=unrelated_pid,
            subject="unrelated process message",
            body="UNRELATED_PROCESS_SECRET",
        )
        runtime.store.insert_llm_call(
            _full_llm_call("run-call", root_pid, "RUN_LLM_SECRET")
        )
        runtime.store.insert_llm_call(
            _full_llm_call("other-call", unrelated_pid, "OTHER_LLM_SECRET")
        )
        run_human = _human_request(
            "human-run-purge",
            root_pid,
            "RUN_HUMAN_SECRET",
        )
        other_human = _human_request(
            "human-other-purge",
            unrelated_pid,
            "OTHER_HUMAN_SECRET",
        )
        runtime.store.insert_human_request(run_human)
        runtime.store.insert_human_request(other_human)
        runtime.store.upsert_llm_tool_output(
            pid=root_pid,
            response_id="response-run-call",
            call_id="tool-run",
            tool_name="test",
            output="RUN_TOOL_SECRET",
        )
        runtime.store.upsert_llm_tool_output(
            pid=unrelated_pid,
            response_id="response-other-call",
            call_id="tool-other",
            tool_name="test",
            output="OTHER_TOOL_SECRET",
        )
        runtime.store.upsert_llm_pending_action(
            root_pid,
            {
                "wait_type": "message",
                "resume_token": "run-pending-token",
                "action": {"action": "receive_process_messages"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "RUN_PENDING_SECRET",
                "tool_call_count": 1,
                "status": "pending",
            },
        )
        runtime.store.upsert_llm_pending_action(
            unrelated_pid,
            {
                "wait_type": "message",
                "resume_token": "other-pending-token",
                "action": {"action": "receive_process_messages"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "OTHER_PENDING_SECRET",
                "tool_call_count": 1,
                "status": "pending",
            },
        )

        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=followed.revision,
            command_id="cancel-for-purge",
        )

        assert terminal.status is TaskRunStatus.CANCELLED
        payloads = runtime.store.list_task_run_payloads(created.run_id)
        assert payloads
        assert all(
            payload.retention_state is TaskRunPayloadRetention.HASH_ONLY
            and payload.canonical_json is None
            and payload.purged_at is not None
            for payload in payloads
        )
        run_call = runtime.store.get_llm_call("run-call")
        other_call = runtime.store.get_llm_call("other-call")
        assert run_call is not None and other_call is not None
        assert llm_call_payload_retention_tier(run_call) is PayloadRetentionTier.HASH_ONLY
        assert llm_call_payload_retention_tier(other_call) is PayloadRetentionTier.FULL
        run_output = runtime.store.list_llm_tool_outputs(
            pid=root_pid,
            response_id="response-run-call",
        )[0]["output_text"]
        run_output_projection = json.loads(run_output)
        assert set(run_output_projection) == {
            "$agent_libos_task_run_tool_output_redaction"
        }
        assert "RUN_TOOL_SECRET" not in run_output
        assert runtime.store.list_llm_tool_outputs(
            pid=unrelated_pid,
            response_id="response-other-call",
        )[0]["output_text"] == "OTHER_TOOL_SECRET"
        assert runtime.store.get_llm_pending_action(root_pid) is None
        assert runtime.store.get_llm_pending_action(unrelated_pid) is not None
        redacted_human = runtime.store.get_human_request(run_human.request_id)
        retained_human = runtime.store.get_human_request(other_human.request_id)
        assert redacted_human is not None and retained_human is not None
        assert "RUN_HUMAN_SECRET" not in json.dumps(
            {
                "payload": redacted_human.payload,
                "decision": redacted_human.decision,
            },
            sort_keys=True,
        )
        assert "OTHER_HUMAN_SECRET" in json.dumps(
            {
                "payload": retained_human.payload,
                "decision": retained_human.decision,
            },
            sort_keys=True,
        )
        root_messages = runtime.store.list_process_messages(root_pid)
        other_messages = runtime.store.list_process_messages(unrelated_pid)
        assert root_messages == []
        assert [message.subject for message in other_messages] == [
            "unrelated process message"
        ]
    finally:
        runtime.close()


def test_terminal_cleanup_failure_stays_finalizing_and_rolls_back_partial_purge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "cleanup-failure.sqlite", config=_config())
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        runtime.store.insert_llm_call(
            _full_llm_call("cleanup-call", created.root_pid, "CLEANUP_SECRET")
        )
        cleanup_human = _human_request(
            "human-cleanup-failure",
            created.root_pid,
            "CLEANUP_HUMAN_SECRET",
        )
        runtime.store.insert_human_request(cleanup_human)

        def fail_payload_purge(*_args: object, **_kwargs: object) -> int:
            raise RuntimeError("injected TaskRun payload purge failure")

        monkeypatch.setattr(runtime.store, "purge_task_run_payloads", fail_payload_purge)
        with pytest.raises(RuntimeError, match="injected TaskRun payload purge failure"):
            runtime.task_runs.cancel(
                created.run_id,
                expected_revision=created.revision,
                command_id="cancel-cleanup-failure",
            )

        current = runtime.task_runs.get(created.run_id)
        assert current.status is TaskRunStatus.FINALIZING
        assert "cleanup_failed" in {item["kind"] for item in current.blockers}
        call = runtime.store.get_llm_call("cleanup-call")
        assert call is not None
        assert llm_call_payload_retention_tier(call) is PayloadRetentionTier.FULL
        retained_human = runtime.store.get_human_request(cleanup_human.request_id)
        assert retained_human is not None
        assert "CLEANUP_HUMAN_SECRET" in json.dumps(
            {
                "payload": retained_human.payload,
                "decision": retained_human.decision,
            },
            sort_keys=True,
        )
        assert all(
            payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
            for payload in runtime.store.list_task_run_payloads(created.run_id)
        )
    finally:
        runtime.close()


def test_permanent_retention_does_not_auto_purge_content(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "permanent.sqlite", config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        assert created.root_pid is not None
        runtime.store.insert_llm_call(
            _full_llm_call("permanent-call", created.root_pid, "PERMANENT_SECRET")
        )
        permanent_human = _human_request(
            "human-permanent",
            created.root_pid,
            "PERMANENT_HUMAN_SECRET",
        )
        runtime.store.insert_human_request(permanent_human)
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-permanent",
        )

        assert terminal.status is TaskRunStatus.CANCELLED
        assert all(
            payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
            for payload in runtime.store.list_task_run_payloads(created.run_id)
        )
        call = runtime.store.get_llm_call("permanent-call")
        assert call is not None
        assert llm_call_payload_retention_tier(call) is PayloadRetentionTier.FULL
        retained_human = runtime.store.get_human_request(
            permanent_human.request_id
        )
        assert retained_human is not None
        assert "PERMANENT_HUMAN_SECRET" in json.dumps(
            {
                "payload": retained_human.payload,
                "decision": retained_human.decision,
            },
            sort_keys=True,
        )
    finally:
        runtime.close()


def test_reopen_abandons_active_object_task_and_blocks_owning_run_without_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "active-object-task-reopen.sqlite"
    repository = Path(__file__).resolve().parents[2]
    worker = f"""
from pathlib import Path
import os

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ObjectMetadata,
    ObjectTaskStatus,
    ObjectType,
)
from tests.runtime.test_task_run_runtime import _config, _create

runtime = Runtime.open(Path({str(database)!r}), config=_config())
created = _create(runtime, request_id="create-active-object-task")
root_pid = created.root_pid
assert root_pid is not None
runtime.capability.grant(
    root_pid,
    "process:spawn",
    [CapabilityRight.WRITE],
    issued_by="test",
)
owner = runtime.memory.create_object(
    root_pid,
    ObjectType.ARTIFACT,
    {{"name": "ObjectTask owner"}},
    metadata=ObjectMetadata(title="ObjectTask owner"),
    immutable=False,
)
task = runtime.object_tasks.start(
    root_pid,
    owner,
    "receive_process_messages",
    {{"channel": "never-replay"}},
)
waiting = runtime.object_tasks.wait(task.task_id, actor_pid=root_pid, timeout=2)
assert waiting.status is ObjectTaskStatus.WAITING_MESSAGE
os._exit(91)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", worker],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 91, crashed.stderr

    reopened = Runtime.open(database, config=_config())
    try:
        runs = reopened.store.list_task_runs(limit=10).records
        assert len(runs) == 1
        run = runs[0]
        assert run.root_pid is not None
        tasks = reopened.store.list_object_tasks(include_terminal=True)
        assert len(tasks) == 1
        task = tasks[0]
        assert task.creator_pid == run.root_pid
        assert task.status is ObjectTaskStatus.ABANDONED

        summary = reopened.task_runs.get(run.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        blocker = next(
            item for item in summary.blockers if item["kind"] == "active_object_task"
        )
        assert blocker["object_task_ids"] == [task.task_id]
        assert TaskRunAction.RUN not in summary.allowed_actions
        assert TaskRunAction.RESUME not in summary.allowed_actions
        assert reopened.process.get(run.root_pid).resource_usage.llm_calls == 0
        assert reopened.run_next_process_once() is None
        assert reopened.store.list_object_tasks(include_terminal=True) == tasks
    finally:
        reopened.close()


def test_command_replay_returns_original_summary_after_a_later_mutation(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "command-summary-replay.sqlite", config=_config())
    try:
        created = _create(runtime)
        first_result = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "preserve this command result"},
            expected_revision=created.revision,
            command_id="durable-follow-up-command",
        )
        later_result = runtime.task_runs.pause(
            created.run_id,
            expected_revision=first_result.revision,
            command_id="unrelated-later-pause",
        )
        assert first_result.status is TaskRunStatus.QUEUED
        assert later_result.status is TaskRunStatus.PAUSED
        assert later_result.revision > first_result.revision

        replayed = runtime.task_runs.follow_up(
            created.run_id,
            {"request": "preserve this command result"},
            expected_revision=created.revision,
            command_id="durable-follow-up-command",
        )

        assert replayed == first_result
        assert replayed.revision == first_result.revision
        assert replayed.status is TaskRunStatus.QUEUED
        assert runtime.task_runs.get(created.run_id) == later_result
        with pytest.raises(TaskRunRevisionConflict, match="different request"):
            runtime.task_runs.follow_up(
                created.run_id,
                {"request": "different content under the same command id"},
                expected_revision=later_result.revision,
                command_id="durable-follow-up-command",
            )
        assert runtime.task_runs.get(created.run_id) == later_result
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "mutation_kind",
    [
        "run",
        "pause",
        "resume",
        "cancel",
        "follow_up",
        "recover",
        "rerun",
        "purge_payloads",
    ],
)
def test_existing_run_command_identity_includes_original_expected_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"expected-revision-identity-{mutation_kind}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-expected-revision-{mutation_kind}",
            retention=TaskRunRetention.PERMANENT,
        )
        command_id = f"expected-revision-{mutation_kind}"
        run_id = created.run_id
        expected_revision = created.revision

        if mutation_kind == "run":
            monkeypatch.setattr(runtime, "run_until_idle", lambda **_kwargs: [])

            def invoke(revision: int):
                return runtime.task_runs.run_until_blocked(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                    max_quanta=1,
                )

        elif mutation_kind == "pause":

            def invoke(revision: int):
                return runtime.task_runs.pause(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                )

        elif mutation_kind == "resume":
            paused = runtime.task_runs.pause(
                run_id,
                expected_revision=created.revision,
                command_id="prepare-expected-revision-resume",
            )
            expected_revision = paused.revision

            def invoke(revision: int):
                return runtime.task_runs.resume(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                )

        elif mutation_kind == "cancel":

            def invoke(revision: int):
                return runtime.task_runs.cancel(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                    reason="stable cancellation request",
                )

        elif mutation_kind == "follow_up":

            def invoke(revision: int):
                return runtime.task_runs.follow_up(
                    run_id,
                    {"request": "stable follow-up"},
                    expected_revision=revision,
                    command_id=command_id,
                )

        elif mutation_kind == "recover":
            source = runtime.store.get_task_run(run_id)
            assert source is not None
            runtime.task_runs._mark_attention(
                source,
                runtime.task_runs._blocker(
                    "binding_drift",
                    "test recovery command identity",
                ),
            )
            attention = runtime.task_runs.get(run_id)
            expected_revision = attention.revision

            def invoke(revision: int):
                return runtime.task_runs.recover(
                    run_id,
                    option_id="terminate_run",
                    expected_revision=revision,
                    command_id=command_id,
                )

        elif mutation_kind == "rerun":
            terminal = runtime.task_runs.cancel(
                run_id,
                expected_revision=created.revision,
                command_id="prepare-expected-revision-rerun",
            )
            expected_revision = terminal.revision

            def invoke(revision: int):
                return runtime.task_runs.rerun(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                    client_request_id="expected-revision-rerun-create",
                    spec_overrides={"goal": "stable rerun goal"},
                )

        else:
            terminal = runtime.task_runs.cancel(
                run_id,
                expected_revision=created.revision,
                command_id="prepare-expected-revision-purge",
            )
            expected_revision = terminal.revision

            def invoke(revision: int):
                return runtime.task_runs.purge_payloads(
                    run_id,
                    expected_revision=revision,
                    command_id=command_id,
                )

        first = invoke(expected_revision)
        assert invoke(expected_revision) == first

        def durable_snapshot() -> tuple[object, ...]:
            root = runtime.store.get_task_run(run_id)
            return (
                root,
                tuple(runtime.store.list_task_runs(limit=100).records),
                tuple(runtime.store.list_task_run_commands(run_id, limit=100)),
                tuple(runtime.store.list_task_run_requirements(run_id)),
                tuple(runtime.store.list_task_run_payloads(run_id)),
                tuple(runtime.task_runs.list_ledger(run_id, limit=100).records),
                tuple(
                    runtime.store.list_process_messages(created.root_pid or "")
                ),
                tuple(runtime.process.list()),
            )

        before_conflict = durable_snapshot()
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            invoke(expected_revision + 10_000)
        assert durable_snapshot() == before_conflict
    finally:
        runtime.close()


def test_evidence_projection_uses_hard_plus_one_bounded_link_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config()
    config = replace(
        base,
        task_runs=replace(
            base.task_runs,
            recovery_page_size=10,
            recovery_page_hard_limit=50,
        ),
    )
    runtime = Runtime.open(tmp_path / "bounded-evidence-links.sqlite", config=config)
    try:
        created = _create(
            runtime,
            request_id="create-bounded-evidence-links",
            retention=TaskRunRetention.PERMANENT,
        )
        original_list_links = runtime.store.list_task_run_links
        observed_limits: list[int | None] = []

        def observe_links(
            run_id: str,
            *,
            limit: int | None = None,
        ):
            observed_limits.append(limit)
            return original_list_links(run_id, limit=limit)

        monkeypatch.setattr(
            runtime.store,
            "list_task_run_links",
            observe_links,
        )
        runtime.task_runs._project_evidence(created.run_id)
        assert observed_limits == [51]
        before = (
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
            tuple(original_list_links(created.run_id)),
        )
        seed = before[1][0]

        def overflow_links(
            _run_id: str,
            *,
            limit: int | None = None,
        ):
            assert limit == 51
            return [seed] * 51

        monkeypatch.setattr(
            runtime.store,
            "list_task_run_links",
            overflow_links,
        )
        with pytest.raises(ValidationError, match="links exceed the recovery bound"):
            runtime.task_runs._project_evidence(created.run_id)
        monkeypatch.setattr(
            runtime.store,
            "list_task_run_links",
            original_list_links,
        )
        assert (
            tuple(runtime.task_runs.list_ledger(created.run_id, limit=100).records),
            tuple(original_list_links(created.run_id)),
        ) == before
    finally:
        runtime.close()


def test_linked_recovery_replay_returns_the_same_new_run_after_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "linked-recovery-command-replay.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        source = _create(
            runtime,
            request_id="create-linked-recovery-source",
            retention=TaskRunRetention.PERMANENT,
        )
        source_record = runtime.store.get_task_run(source.run_id)
        assert source_record is not None
        runtime.task_runs._mark_attention(
            source_record,
            runtime.task_runs._blocker(
                "binding_drift",
                "test evidence requires a separately fenced Run",
            ),
        )
        blocked = runtime.task_runs.get(source.run_id)
        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert "create_linked_run" in {
            option.option_id
            for option in runtime.task_runs.recovery_options(source.run_id)
        }

        created = runtime.task_runs.recover(
            source.run_id,
            option_id="create_linked_run",
            expected_revision=blocked.revision,
            command_id="recover-as-linked-run",
        )

        assert created.run_id != source.run_id
        assert len(runtime.store.list_task_runs(limit=10).records) == 2
        command = runtime.store.get_task_run_command(
            source.run_id,
            "recover-as-linked-run",
        )
        assert command is not None
        assert command.result["new_run_id"] == created.run_id
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        replayed = reopened.task_runs.recover(
            source.run_id,
            option_id="create_linked_run",
            expected_revision=blocked.revision,
            command_id="recover-as-linked-run",
        )

        assert replayed == created
        assert replayed.run_id != source.run_id
        assert len(reopened.store.list_task_runs(limit=10).records) == 2
        with pytest.raises(TaskRunRevisionConflict, match="different request"):
            reopened.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                receipt={"unexpected": "request hash change"},
                expected_revision=blocked.revision,
                command_id="recover-as-linked-run",
            )
        assert len(reopened.store.list_task_runs(limit=10).records) == 2
    finally:
        reopened.close()


def test_linked_recovery_reconstructs_missing_parent_receipt_from_bound_nested_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "linked-recovery-parent-receipt-gap.sqlite"
    command_id = "recover-linked-parent-gap"
    first = Runtime.open(database, config=_config())
    try:
        source = _create(
            first,
            request_id="create-linked-parent-gap-source",
            retention=TaskRunRetention.PERMANENT,
        )
        source_record = first.store.get_task_run(source.run_id)
        assert source_record is not None
        first.task_runs._mark_attention(
            source_record,
            first.task_runs._blocker(
                "binding_drift",
                "test linked recovery parent receipt gap",
            ),
        )
        blocked = first.task_runs.get(source.run_id)
        original_record_command = first.task_runs._record_command

        def lose_parent_receipt(
            record: object,
            selected_command_id: str,
            command_kind: str,
            request: object,
            **kwargs: object,
        ) -> object:
            if command_kind == "recover" and selected_command_id == command_id:
                raise RuntimeError("injected linked recovery parent receipt loss")
            return original_record_command(
                record,
                selected_command_id,
                command_kind,
                request,
                **kwargs,
            )

        monkeypatch.setattr(
            first.task_runs,
            "_record_command",
            lose_parent_receipt,
        )
        with pytest.raises(RuntimeError, match="parent receipt loss"):
            first.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                receipt={"opaque": "bound-parent-request"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        nested = first.store.get_task_run_command(
            source.run_id,
            f"{command_id}:rerun",
        )
        assert nested is not None
        assert nested.command_kind == "rerun"
        assert first.store.get_task_run_command(source.run_id, command_id) is None
        assert len(first.store.list_task_runs(limit=10).records) == 2
        new_run_id = str(nested.result["new_run_id"])
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        before_conflict = (
            tuple(reopened.store.list_task_runs(limit=10).records),
            tuple(
                reopened.store.list_task_run_commands(
                    source.run_id,
                    limit=100,
                )
            ),
            tuple(reopened.store.list_task_run_links(new_run_id)),
        )
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            reopened.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                receipt={"opaque": "bound-parent-request"},
                expected_revision=blocked.revision + 1,
                command_id=command_id,
            )
        assert (
            tuple(reopened.store.list_task_runs(limit=10).records),
            tuple(
                reopened.store.list_task_run_commands(
                    source.run_id,
                    limit=100,
                )
            ),
            tuple(reopened.store.list_task_run_links(new_run_id)),
        ) == before_conflict
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            reopened.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                receipt={"opaque": "changed-parent-request"},
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        assert (
            tuple(reopened.store.list_task_runs(limit=10).records),
            tuple(
                reopened.store.list_task_run_commands(
                    source.run_id,
                    limit=100,
                )
            ),
            tuple(reopened.store.list_task_run_links(new_run_id)),
        ) == before_conflict

        original_list_links = reopened.store.list_task_run_links
        observed_limits: list[int | None] = []

        def observe_bounded_links(
            selected_run_id: str,
            *,
            limit: int | None = None,
        ):
            observed_limits.append(limit)
            return original_list_links(selected_run_id, limit=limit)

        monkeypatch.setattr(
            reopened.store,
            "list_task_run_links",
            observe_bounded_links,
        )
        recovered = reopened.task_runs.recover(
            source.run_id,
            option_id="create_linked_run",
            receipt={"opaque": "bound-parent-request"},
            expected_revision=blocked.revision,
            command_id=command_id,
        )
        parent = reopened.store.get_task_run_command(source.run_id, command_id)
        nested = reopened.store.get_task_run_command(
            source.run_id,
            f"{command_id}:rerun",
        )
        assert recovered.run_id == new_run_id
        assert parent is not None and nested is not None
        assert parent.result == nested.result
        assert parent.result_revision == nested.result_revision
        assert reopened.config.task_runs.recovery_page_hard_limit + 1 in observed_limits
        assert len(reopened.store.list_task_runs(limit=10).records) == 2
        assert reopened.task_runs.recover(
            source.run_id,
            option_id="create_linked_run",
            receipt={"opaque": "bound-parent-request"},
            expected_revision=blocked.revision,
            command_id=command_id,
        ) == recovered
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "tamper",
    ["nested_hash", "target_identity", "rerun_link"],
)
def test_linked_recovery_parent_gap_fails_closed_on_nested_evidence_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    database = tmp_path / f"linked-recovery-gap-tamper-{tamper}.sqlite"
    command_id = f"recover-linked-gap-tamper-{tamper}"
    first = Runtime.open(database, config=_config())
    try:
        source = _create(
            first,
            request_id=f"create-linked-gap-tamper-{tamper}",
            retention=TaskRunRetention.PERMANENT,
        )
        source_record = first.store.get_task_run(source.run_id)
        assert source_record is not None
        first.task_runs._mark_attention(
            source_record,
            first.task_runs._blocker(
                "binding_drift",
                "test linked recovery nested evidence tamper",
            ),
        )
        blocked = first.task_runs.get(source.run_id)
        original_record_command = first.task_runs._record_command

        def lose_parent_receipt(
            record: object,
            selected_command_id: str,
            command_kind: str,
            request: object,
            **kwargs: object,
        ) -> object:
            if command_kind == "recover" and selected_command_id == command_id:
                raise RuntimeError("injected parent receipt loss before tamper")
            return original_record_command(
                record,
                selected_command_id,
                command_kind,
                request,
                **kwargs,
            )

        monkeypatch.setattr(
            first.task_runs,
            "_record_command",
            lose_parent_receipt,
        )
        with pytest.raises(RuntimeError, match="before tamper"):
            first.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        nested = first.store.get_task_run_command(
            source.run_id,
            f"{command_id}:rerun",
        )
        assert nested is not None
        target_run_id = str(nested.result["new_run_id"])
    finally:
        first.close()

    connection = sqlite3.connect(database)
    try:
        if tamper == "nested_hash":
            connection.execute(
                "UPDATE task_run_commands SET request_hash = ? "
                "WHERE run_id = ? AND command_id = ?",
                ("0" * 64, source.run_id, f"{command_id}:rerun"),
            )
        elif tamper == "target_identity":
            row = connection.execute(
                "SELECT result_json FROM task_run_commands "
                "WHERE run_id = ? AND command_id = ?",
                (source.run_id, f"{command_id}:rerun"),
            ).fetchone()
            assert row is not None
            result = json.loads(str(row[0]))
            result["new_run_id"] = source.run_id
            connection.execute(
                "UPDATE task_run_commands SET result_json = ? "
                "WHERE run_id = ? AND command_id = ?",
                (
                    json.dumps(result, sort_keys=True, separators=(",", ":")),
                    source.run_id,
                    f"{command_id}:rerun",
                ),
            )
        else:
            connection.execute(
                "DELETE FROM task_run_links WHERE run_id = ? "
                "AND evidence_type = 'task_run' AND evidence_id = ? "
                "AND role = 'rerun_of'",
                (target_run_id, source.run_id),
            )
        connection.commit()
    finally:
        connection.close()

    reopened = Runtime.open(database, config=_config())
    try:
        before = (
            tuple(reopened.store.list_task_runs(limit=10).records),
            tuple(
                reopened.store.list_task_run_commands(
                    source.run_id,
                    limit=100,
                )
            ),
            tuple(reopened.store.list_task_run_links(target_run_id)),
        )
        error = (
            TaskRunCommandConflict
            if tamper == "nested_hash"
            else ValidationError
        )
        with pytest.raises(error):
            reopened.task_runs.recover(
                source.run_id,
                option_id="create_linked_run",
                expected_revision=blocked.revision,
                command_id=command_id,
            )
        assert reopened.store.get_task_run_command(source.run_id, command_id) is None
        assert (
            tuple(reopened.store.list_task_runs(limit=10).records),
            tuple(
                reopened.store.list_task_run_commands(
                    source.run_id,
                    limit=100,
                )
            ),
            tuple(reopened.store.list_task_run_links(target_run_id)),
        ) == before
    finally:
        reopened.close()


def test_purged_terminal_run_advertises_rerun_but_requires_replacement_goal(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "purged-rerun.sqlite", config=_config())
    try:
        created = _create(runtime)
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-before-rerun",
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        assert terminal.payloads_purged is True
        assert TaskRunAction.RERUN in terminal.allowed_actions
        goal = next(
            payload
            for payload in runtime.store.list_task_run_payloads(created.run_id)
            if payload.role == "goal"
        )
        assert goal.retention_state is TaskRunPayloadRetention.HASH_ONLY
        assert goal.canonical_json is None

        with pytest.raises(ValidationError, match="goal payload is hash-only"):
            runtime.task_runs.rerun(
                created.run_id,
                expected_revision=terminal.revision,
                command_id="rerun-without-replacement-goal",
                client_request_id="rerun-without-replacement-goal-create",
            )

        assert runtime.task_runs.get(created.run_id) == terminal
        assert len(runtime.store.list_task_runs(limit=10).records) == 1
        assert (
            runtime.store.get_task_run_command(
                created.run_id,
                "rerun-without-replacement-goal",
            )
            is None
        )
    finally:
        runtime.close()


def test_rerun_command_replay_survives_source_revision_and_epoch_advancement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rerun-command-replay.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            request_id="create-permanent-rerun-source",
            retention=TaskRunRetention.PERMANENT,
        )
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-permanent-rerun-source",
        )
        linked = runtime.task_runs.rerun(
            created.run_id,
            expected_revision=terminal.revision,
            command_id="stable-rerun-after-response-loss",
            client_request_id="stable-rerun-linked-create",
        )
        source_after_rerun = runtime.task_runs.get(created.run_id)
        purged = runtime.task_runs.purge_payloads(
            created.run_id,
            expected_revision=source_after_rerun.revision,
            command_id="purge-after-rerun-response-loss",
        )
        assert purged.revision > terminal.revision
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        replayed = reopened.task_runs.rerun(
            created.run_id,
            expected_revision=terminal.revision,
            command_id="stable-rerun-after-response-loss",
            client_request_id="stable-rerun-linked-create",
        )

        assert replayed == linked
        assert len(reopened.store.list_task_runs(limit=10).records) == 2
    finally:
        reopened.close()


def test_reopened_terminal_runs_claim_epoch_only_inside_new_rerun_or_purge(
    tmp_path: Path,
) -> None:
    database = tmp_path / "terminal-on-demand-claim.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        rerun_source = _create(
            first,
            request_id="create-terminal-rerun-source",
            retention=TaskRunRetention.PERMANENT,
        )
        purge_source = _create(
            first,
            request_id="create-terminal-purge-source",
            retention=TaskRunRetention.PERMANENT,
        )
        rerun_terminal = first.task_runs.cancel(
            rerun_source.run_id,
            expected_revision=rerun_source.revision,
            command_id="cancel-terminal-rerun-source",
        )
        purge_terminal = first.task_runs.cancel(
            purge_source.run_id,
            expected_revision=purge_source.revision,
            command_id="cancel-terminal-purge-source",
        )
        old_epoch = first.task_runs.runtime_epoch
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        assert reopened.task_runs.runtime_epoch > old_epoch
        assert (
            reopened.store.get_task_run(rerun_source.run_id).runtime_epoch
            == old_epoch
        )
        linked = reopened.task_runs.rerun(
            rerun_source.run_id,
            expected_revision=rerun_terminal.revision,
            command_id="rerun-after-terminal-reopen",
            client_request_id="create-linked-after-terminal-reopen",
        )
        claimed_rerun_source = reopened.store.get_task_run(rerun_source.run_id)
        assert claimed_rerun_source is not None
        assert claimed_rerun_source.runtime_epoch == reopened.task_runs.runtime_epoch
        assert linked.run_id != rerun_source.run_id
        assert (
            reopened.process.get(rerun_source.root_pid).task_run_epoch
            == reopened.task_runs.runtime_epoch
        )

        purged = reopened.task_runs.purge_payloads(
            purge_source.run_id,
            expected_revision=purge_terminal.revision,
            command_id="purge-after-terminal-reopen",
        )
        claimed_purge_source = reopened.store.get_task_run(purge_source.run_id)
        assert purged.payloads_purged is True
        assert claimed_purge_source is not None
        assert claimed_purge_source.runtime_epoch == reopened.task_runs.runtime_epoch
        assert (
            reopened.process.get(purge_source.root_pid).task_run_epoch
            == reopened.task_runs.runtime_epoch
        )
    finally:
        reopened.close()


def test_terminal_claim_rolls_back_with_failed_rerun_or_purge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "terminal-claim-rollback.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        rerun_source = _create(
            first,
            request_id="create-rerun-rollback-source",
            retention=TaskRunRetention.PERMANENT,
        )
        purge_source = _create(
            first,
            request_id="create-purge-rollback-source",
            retention=TaskRunRetention.PERMANENT,
        )
        rerun_terminal = first.task_runs.cancel(
            rerun_source.run_id,
            expected_revision=rerun_source.revision,
            command_id="cancel-rerun-rollback-source",
        )
        purge_terminal = first.task_runs.cancel(
            purge_source.run_id,
            expected_revision=purge_source.revision,
            command_id="cancel-purge-rollback-source",
        )
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        rerun_before = reopened.store.get_task_run(rerun_source.run_id)
        purge_before = reopened.store.get_task_run(purge_source.run_id)
        purge_payloads_before = reopened.store.list_task_run_payloads(
            purge_source.run_id
        )
        rerun_process_epoch_before = reopened.process.get(
            rerun_source.root_pid
        ).task_run_epoch
        purge_process_epoch_before = reopened.process.get(
            purge_source.root_pid
        ).task_run_epoch
        run_count_before = len(reopened.store.list_task_runs(limit=10).records)
        original_payload_by_role = reopened.task_runs._payload_by_role

        def fail_after_rerun_claim(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("fault after terminal rerun claim")

        monkeypatch.setattr(
            reopened.task_runs,
            "_payload_by_role",
            fail_after_rerun_claim,
        )
        with pytest.raises(RuntimeError, match="fault after terminal rerun claim"):
            reopened.task_runs.rerun(
                rerun_source.run_id,
                expected_revision=rerun_terminal.revision,
                command_id="rerun-fault-after-claim",
                client_request_id="rerun-fault-after-claim-create",
            )
        monkeypatch.setattr(
            reopened.task_runs,
            "_payload_by_role",
            original_payload_by_role,
        )

        def fail_after_purge_claim(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("fault after terminal purge claim")

        monkeypatch.setattr(
            reopened.task_runs,
            "_purge_run_content",
            fail_after_purge_claim,
        )
        with pytest.raises(RuntimeError, match="fault after terminal purge claim"):
            reopened.task_runs.purge_payloads(
                purge_source.run_id,
                expected_revision=purge_terminal.revision,
                command_id="purge-fault-after-claim",
            )

        assert reopened.store.get_task_run(rerun_source.run_id) == rerun_before
        assert reopened.store.get_task_run(purge_source.run_id) == purge_before
        assert (
            reopened.process.get(rerun_source.root_pid).task_run_epoch
            == rerun_process_epoch_before
        )
        assert (
            reopened.process.get(purge_source.root_pid).task_run_epoch
            == purge_process_epoch_before
        )
        assert (
            reopened.store.list_task_run_payloads(purge_source.run_id)
            == purge_payloads_before
        )
        assert len(reopened.store.list_task_runs(limit=10).records) == run_count_before
        assert (
            reopened.store.get_task_run_command(
                rerun_source.run_id,
                "rerun-fault-after-claim",
            )
            is None
        )
        assert (
            reopened.store.get_task_run_command(
                purge_source.run_id,
                "purge-fault-after-claim",
            )
            is None
        )
    finally:
        reopened.close()


def test_direct_rerun_rejects_nonterminal_and_needs_attention_sources(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "direct-rerun-source-status.sqlite", config=_config())
    try:
        created = _create(
            runtime,
            request_id="create-direct-rerun-source",
            retention=TaskRunRetention.PERMANENT,
        )
        with pytest.raises(ValidationError, match="terminal source"):
            runtime.task_runs.rerun(
                created.run_id,
                expected_revision=created.revision,
                command_id="rerun-queued-source",
            )
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None
        attention = runtime.task_runs._mark_attention(
            record,
            runtime.task_runs._blocker(
                "unknown_effect",
                "test unknown effect must not authorize direct rerun",
            ),
        )
        with pytest.raises(ValidationError, match="terminal source"):
            runtime.task_runs.rerun(
                created.run_id,
                expected_revision=attention.revision,
                command_id="rerun-unknown-effect-source",
            )
        assert len(runtime.store.list_task_runs(limit=10).records) == 1
        assert (
            runtime.store.get_task_run_command(
                created.run_id,
                "rerun-unknown-effect-source",
            )
            is None
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "timeout",
    (float("nan"), float("inf"), float("-inf"), 10**10_000),
    ids=("nan", "positive-infinity", "negative-infinity", "unrepresentable"),
)
def test_wait_rejects_non_finite_timeout(
    tmp_path: Path,
    timeout: float | int,
) -> None:
    runtime = Runtime.open(
        tmp_path / "wait-nonfinite-timeout.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime)
        with pytest.raises(ValidationError, match="finite non-negative"):
            runtime.task_runs.wait(created.run_id, timeout=timeout)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_schema",
        "boolean_schema",
        "missing_summary",
        "extra_key",
        "noncanonical_summary",
        "summary_bigint_overflow",
    ),
)
def test_exact_replay_rejects_noncanonical_base_command_envelope_without_dispatch(
    corruption: str,
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"command-base-envelope-{corruption}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-command-base-envelope-{corruption}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        command_id = f"pause-command-base-envelope-{corruption}"
        runtime.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id=command_id,
        )
        command = runtime.store.get_task_run_command(created.run_id, command_id)
        assert command is not None
        result = json.loads(json.dumps(command.result))
        if corruption == "missing_schema":
            result.pop("schema_version")
        elif corruption == "boolean_schema":
            result["schema_version"] = True
        elif corruption == "missing_summary":
            result.pop("summary")
        elif corruption == "extra_key":
            result["unexpected"] = "must not be ignored"
        elif corruption == "noncanonical_summary":
            result["summary"].pop("completed_at")
        else:
            result["summary"]["step_count"] = 2**63
        runtime.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_run_requirements(created.run_id)),
            tuple(runtime.store.list_process_messages(created.root_pid)),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        )

        with pytest.raises(ValidationError):
            runtime.task_runs.pause(
                created.run_id,
                expected_revision=created.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_run_requirements(created.run_id)),
            tuple(runtime.store.list_process_messages(created.root_pid)),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        ) == before
        assert runtime.run_next_process_once() is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "create_extra",
        "run_extra",
        "resume_extra",
        "cancel_overflow",
        "deadline_wrong_kind",
        "effect_extra",
        "terminalize_extra",
        "linked_missing_target",
        "linked_alternate_source",
        "linked_target_bigint_overflow",
    ),
)
def test_command_result_family_validator_rejects_variant_confusion(
    corruption: str,
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"command-family-confusion-{corruption}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-command-family-confusion-{corruption}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        command = runtime.store.get_task_run_command(
            created.run_id,
            f"create:create-command-family-confusion-{corruption}",
        )
        assert command is not None
        base = json.loads(json.dumps(command.result))
        kind = "create"
        result = dict(base)
        if corruption == "create_extra":
            result["unexpected"] = True
        elif corruption == "run_extra":
            kind = "run"
            result.update(
                settlement_state="pending",
                admission_revision=command.result_revision,
                unexpected=True,
            )
        elif corruption == "resume_extra":
            kind = "resume"
            result.update(
                settlement_state="complete",
                pause_generation=0,
                unexpected=True,
            )
        elif corruption == "cancel_overflow":
            kind = "cancel"
            result.update(
                settlement_state="pending",
                cancel_generation=2**63,
            )
        elif corruption == "deadline_wrong_kind":
            kind = "run"
            result.update(
                settlement_state="pending",
                settlement_kind="terminalize",
                cancel_generation=1,
            )
        elif corruption == "effect_extra":
            kind = "recover"
            result.update(
                settlement_state="pending",
                settlement_kind="effect_receipt",
                effect_id="effect-wrong-variant",
                expected_transaction_state="unknown",
                cancel_generation=0,
                admission_runtime_epoch=runtime.task_runs.runtime_epoch,
                settlement_transition_seq=1,
                settlement_audit_record_id="audit-wrong-variant",
                unexpected=True,
            )
        elif corruption == "terminalize_extra":
            kind = "recover"
            result.update(
                settlement_state="pending",
                settlement_kind="terminalize",
                cancel_generation=1,
                effect_id="effect-must-not-be-read",
            )
        else:
            kind = "rerun"
            target = json.loads(json.dumps(base["summary"]))
            target["run_id"] = "run-linked-target"
            if corruption == "linked_target_bigint_overflow":
                target["revision"] = 2**63
            result.update(
                run_id=created.run_id,
                revision=command.result_revision,
                new_run_id="run-linked-target",
                new_run_summary=target,
            )
            if corruption == "linked_missing_target":
                result.pop("new_run_id")
            elif corruption == "linked_alternate_source":
                result["alternate_run_id"] = created.run_id
        candidate = replace(
            command,
            command_kind=kind,
            result=result,
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
        )

        with pytest.raises(ValidationError):
            runtime.task_runs._validate_command_result_for_kind(candidate)

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
        ) == before
    finally:
        runtime.close()


@pytest.mark.parametrize("family", ("run", "resume"))
@pytest.mark.parametrize("terminal", (False, True))
def test_future_run_or_resume_receipt_fence_never_settles(
    family: str,
    terminal: bool,
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"future-{family}-receipt-{terminal}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-future-{family}-receipt-{terminal}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        if family == "resume":
            runtime.task_runs.pause(
                created.run_id,
                expected_revision=created.revision,
                command_id=f"pause-before-future-resume-{terminal}",
            )
        if terminal:
            runtime.process.cancel(created.root_pid, "terminal future-fence test")
            runtime.task_runs._project(
                runtime.store.get_task_run(created.run_id),
                allow_finalize=True,
            )
        current = runtime.store.get_task_run(created.run_id)
        assert current is not None
        expected_revision = current.revision
        command_id = f"future-{family}-receipt-{terminal}"
        if family == "run":
            request = {
                "expected_revision": expected_revision,
                "max_quanta": 1,
            }
            result = {
                "settlement_state": "pending",
                "admission_revision": current.revision,
            }
        else:
            request = {"expected_revision": expected_revision}
            result = {
                "settlement_state": "pending",
                "pause_generation": current.pause_generation + 1,
            }
        runtime.task_runs._record_command(
            current,
            command_id,
            family,
            request,
            result=result,
        )
        if family == "run":
            command = runtime.store.get_task_run_command(created.run_id, command_id)
            assert command is not None
            tampered = json.loads(json.dumps(command.result))
            tampered["admission_revision"] = current.revision + 1
            tampered["summary"]["revision"] = current.revision + 1
            runtime.store._execute(  # noqa: SLF001 - persisted receipt corruption
                "UPDATE task_run_commands SET result_json = ?, result_revision = ? "
                "WHERE run_id = ? AND command_id = ?",
                (
                    json.dumps(tampered, sort_keys=True, separators=(",", ":")),
                    current.revision + 1,
                    created.run_id,
                    command_id,
                ),
            )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
        )

        with pytest.raises(ValidationError):
            if family == "run":
                runtime.task_runs.run_until_blocked(
                    created.run_id,
                    expected_revision=expected_revision,
                    command_id=command_id,
                    max_quanta=1,
                )
            else:
                runtime.task_runs.resume(
                    created.run_id,
                    expected_revision=expected_revision,
                    command_id=command_id,
                )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
        ) == before
        assert before[2] is not None
        assert before[2].result["settlement_state"] == "pending"
    finally:
        runtime.close()


def test_pending_resume_receipt_converges_after_runtime_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "resume-pending-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(first, request_id="create-resume-pending-reopen")
        assert created.root_pid is not None
        paused = first.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id="pause-before-resume-crash",
        )

        def crash_before_process_resume(_pid: str) -> None:
            raise RuntimeError("fault after durable resume admission")

        monkeypatch.setattr(
            first.task_runs._process,
            "resume",
            crash_before_process_resume,
        )
        with pytest.raises(RuntimeError, match="durable resume admission"):
            first.task_runs.resume(
                created.run_id,
                expected_revision=paused.revision,
                command_id="resume-with-local-crash",
            )
        command = first.store.get_task_run_command(
            created.run_id,
            "resume-with-local-crash",
        )
        assert command is not None
        assert command.result["settlement_state"] == "pending"
        assert first.process.get(created.root_pid).status is ProcessStatus.PAUSED

        current = first.store.get_task_run(created.run_id)
        assert current is not None
        before_conflict = (
            current,
            command,
            first.process.get(created.root_pid),
        )
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            first.task_runs.resume(
                created.run_id,
                expected_revision=current.revision,
                command_id="resume-with-local-crash",
            )
        assert (
            first.store.get_task_run(created.run_id),
            first.store.get_task_run_command(
                created.run_id,
                "resume-with-local-crash",
            ),
            first.process.get(created.root_pid),
        ) == before_conflict
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        before_retry = reopened.task_runs.get(created.run_id)
        assert before_retry.status is TaskRunStatus.PAUSED
        resumed = reopened.task_runs.resume(
            created.run_id,
            expected_revision=paused.revision,
            command_id="resume-with-local-crash",
        )
        command = reopened.store.get_task_run_command(
            created.run_id,
            "resume-with-local-crash",
        )
        assert resumed.status is TaskRunStatus.RUNNING
        assert reopened.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == resumed.revision
    finally:
        reopened.close()


def test_pending_interrupt_follow_up_reopens_and_settles_without_manual_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-follow-up-pending-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "include the durable follow-up exactly once"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-follow-up-pending",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        _claim_test_task_run_action(
            first,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id="interrupt-pending-reopen-old-action",
            action={"action": "process_exit", "payload": {"message": "OLD"}},
        )
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after durable interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="durable interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-follow-up-with-local-crash",
            )

        pending = first.store.get_task_run_command(
            created.run_id,
            "interrupt-follow-up-with-local-crash",
        )
        interrupted = first.task_runs.get(created.run_id)
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"
        assert pending.result["settlement_kind"] == "interrupt"
        assert pending.result["pause_generation"] == 1
        assert pending.result["prior_status"] == TaskRunStatus.RUNNING.value
        assert interrupted.status is TaskRunStatus.PAUSED
        assert len(first.store.list_task_run_requirements(created.run_id)) == 2
        assert len(first.store.list_process_messages(created.root_pid)) == 1
        assert first.process.get(created.root_pid).status is ProcessStatus.RUNNABLE

        before_conflict = (
            first.store.get_task_run(created.run_id),
            pending,
            tuple(first.store.list_task_run_requirements(created.run_id)),
            tuple(first.store.list_process_messages(created.root_pid)),
        )
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=interrupted.revision,
                command_id="interrupt-follow-up-with-local-crash",
            )
        assert (
            first.store.get_task_run(created.run_id),
            first.store.get_task_run_command(
                created.run_id,
                "interrupt-follow-up-with-local-crash",
            ),
            tuple(first.store.list_task_run_requirements(created.run_id)),
            tuple(first.store.list_process_messages(created.root_pid)),
        ) == before_conflict
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        before_replay = reopened.task_runs.get(created.run_id)
        recovered_command = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-follow-up-with-local-crash",
        )
        assert before_replay.status is TaskRunStatus.RUNNING
        assert reopened.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert recovered_command is not None
        assert recovered_command.result["settlement_state"] == "complete"
        assert recovered_command.result_revision == before_replay.revision

        settled = reopened.task_runs.follow_up(
            created.run_id,
            body,
            kind="interrupt",
            expected_revision=running.revision,
            command_id="interrupt-follow-up-with-local-crash",
        )

        command = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-follow-up-with-local-crash",
        )
        assert settled == reopened.task_runs.get(created.run_id)
        assert settled.status is TaskRunStatus.RUNNING
        assert reopened.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == settled.revision
        assert len(reopened.store.list_task_run_requirements(created.run_id)) == 2
        assert len(reopened.store.list_process_messages(created.root_pid)) == 1
        usage = reopened.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0

        replayed = reopened.task_runs.follow_up(
            created.run_id,
            body,
            kind="interrupt",
            expected_revision=running.revision,
            command_id="interrupt-follow-up-with-local-crash",
        )
        assert replayed == settled
        assert len(reopened.store.list_task_run_requirements(created.run_id)) == 2
        assert len(reopened.store.list_process_messages(created.root_pid)) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_schema",
        "missing_summary",
        "missing_prior_status",
        "extra_key",
        "duplicate_identical_fence",
        "duplicate_pid_state_fence",
        "duplicate_pid_execution_fence",
        "pause_generation_overflow",
        "cancel_generation_overflow",
        "admission_epoch_overflow",
        "state_generation_overflow",
        "execution_generation_overflow",
    ),
)
def test_corrupt_pending_interrupt_receipt_fails_closed_without_dispatch(
    corruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / f"interrupt-pending-corrupt-{corruption}.sqlite"
    body = {"constraint": "a corrupt interrupt receipt never resumes work"}
    command_id = f"interrupt-pending-corrupt-{corruption}"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(
            first,
            request_id=f"create-{command_id}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after corrupt-receipt interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="corrupt-receipt interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )
        pending = first.store.get_task_run_command(created.run_id, command_id)
        assert pending is not None
        result = json.loads(json.dumps(pending.result))
        if corruption == "missing_schema":
            result.pop("schema_version")
        elif corruption == "missing_summary":
            result.pop("summary")
        elif corruption == "missing_prior_status":
            result.pop("prior_status")
        elif corruption == "extra_key":
            result["unexpected"] = "must fail closed"
        elif corruption == "duplicate_identical_fence":
            result["resume_fences"] = [
                [created.root_pid, 0, 1],
                [created.root_pid, 0, 1],
            ]
        elif corruption == "duplicate_pid_state_fence":
            result["resume_fences"] = [
                [created.root_pid, 0, 1],
                [created.root_pid, 1, 1],
            ]
        elif corruption == "duplicate_pid_execution_fence":
            result["resume_fences"] = [
                [created.root_pid, 0, 1],
                [created.root_pid, 0, 2],
            ]
        elif corruption == "pause_generation_overflow":
            result["pause_generation"] = 2**63
        elif corruption == "cancel_generation_overflow":
            result["cancel_generation"] = 2**63
        elif corruption == "admission_epoch_overflow":
            result["admission_runtime_epoch"] = 2**63
        elif corruption == "state_generation_overflow":
            result["resume_fences"] = [[created.root_pid, 2**63, 1]]
        else:
            result["resume_fences"] = [[created.root_pid, 0, 2**63]]
        first.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
        exact_before = (
            first.store.get_task_run(created.run_id),
            first.process.get(created.root_pid),
            first.store.get_task_run_command(created.run_id, command_id),
            tuple(first.store.list_task_run_requirements(created.run_id)),
            tuple(first.store.list_process_messages(created.root_pid)),
        )
        with pytest.raises(ValidationError):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )
        assert (
            first.store.get_task_run(created.run_id),
            first.process.get(created.root_pid),
            first.store.get_task_run_command(created.run_id, command_id),
            tuple(first.store.list_task_run_requirements(created.run_id)),
            tuple(first.store.list_process_messages(created.root_pid)),
        ) == exact_before
        assert first.run_next_process_once() is None
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        attention = reopened.task_runs.get(created.run_id)
        command = reopened.store.get_task_run_command(created.run_id, command_id)
        assert attention.status is TaskRunStatus.NEEDS_ATTENTION
        assert "manual_recovery_required" in {
            blocker["kind"] for blocker in attention.blockers
        }
        assert command is not None
        assert command.result_revision == pending.result_revision
        assert command.result.get("settlement_state") == "pending"
        assert len(reopened.store.list_task_run_requirements(created.run_id)) == 2
        assert len(reopened.store.list_process_messages(created.root_pid)) == 1
        usage = reopened.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "pending_only_fields",
        "missing_prior_status",
        "extra_key",
        "invalid_state",
        "pause_generation_overflow",
        "cancel_generation_overflow",
    ),
)
def test_exact_replay_rejects_corrupt_completed_interrupt_receipt(
    corruption: str,
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"interrupt-complete-corrupt-{corruption}.sqlite",
        config=_config(),
    )
    body = {"constraint": "completed interrupt receipts remain exact"}
    try:
        created = _create(
            runtime,
            request_id=f"create-interrupt-complete-corrupt-{corruption}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = runtime.store.get_task_run(created.run_id)
        assert queued is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        running = runtime.task_runs.get(created.run_id)
        command_id = f"interrupt-complete-corrupt-{corruption}"
        runtime.task_runs.follow_up(
            created.run_id,
            body,
            kind="interrupt",
            expected_revision=running.revision,
            command_id=command_id,
        )
        command = runtime.store.get_task_run_command(created.run_id, command_id)
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert "admission_runtime_epoch" not in command.result
        assert "resume_fences" not in command.result
        result = json.loads(json.dumps(command.result))
        if corruption == "pending_only_fields":
            result["admission_runtime_epoch"] = runtime.task_runs.runtime_epoch
            result["resume_fences"] = []
        elif corruption == "missing_prior_status":
            result.pop("prior_status")
        elif corruption == "extra_key":
            result["unexpected"] = "must fail closed"
        elif corruption == "invalid_state":
            result["settlement_state"] = "finished"
        elif corruption == "pause_generation_overflow":
            result["pause_generation"] = 2**63
        else:
            result["cancel_generation"] = 2**63
        runtime.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_run_requirements(created.run_id)),
            tuple(runtime.store.list_process_messages(created.root_pid)),
        )

        with pytest.raises(ValidationError):
            runtime.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(runtime.store.list_task_run_requirements(created.run_id)),
            tuple(runtime.store.list_process_messages(created.root_pid)),
        ) == before
    finally:
        runtime.close()


@pytest.mark.parametrize("tamper", ("prior_status", "foreign_resume_fence"))
def test_terminal_interrupt_replay_requires_immutable_admission_evidence(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"terminal-interrupt-admission-{tamper}.sqlite",
        config=_config(),
    )
    body = {"constraint": "receipt fields must match admission evidence"}
    command_id = f"terminal-interrupt-admission-{tamper}"
    try:
        created = _create(
            runtime,
            request_id=f"create-{command_id}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = runtime.store.get_task_run(created.run_id)
        assert queued is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        running = runtime.task_runs.get(created.run_id)

        def crash_after_admission(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after interrupt admission evidence")

        monkeypatch.setattr(
            runtime.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_admission,
        )
        with pytest.raises(RuntimeError, match="interrupt admission evidence"):
            runtime.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )
        monkeypatch.undo()

        runtime.process.cancel(created.root_pid, "terminal receipt validation")
        terminal = runtime.task_runs._project(
            runtime.store.get_task_run(created.run_id),
            allow_finalize=True,
        )
        assert terminal.status in {
            TaskRunStatus.FAILED,
            TaskRunStatus.CANCELLED,
        }
        pending = runtime.store.get_task_run_command(created.run_id, command_id)
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"
        result = json.loads(json.dumps(pending.result))
        if tamper == "prior_status":
            result["prior_status"] = TaskRunStatus.QUEUED.value
        else:
            result["resume_fences"] = [["pid-foreign", 0, 1]]
            result["interrupt_provenance_sha256"] = runtime.task_runs._sha256(
                {
                    "schema_version": 1,
                    "admission_runtime_epoch": result[
                        "admission_runtime_epoch"
                    ],
                    "resume_fences": result["resume_fences"],
                }
            )
        result["admission_evidence_sha256"] = runtime.task_runs._sha256(
            {
                "schema_version": 1,
                "run_id": pending.run_id,
                "command_id": pending.command_id,
                "command_kind": pending.command_kind,
                "request_hash": pending.request_hash,
                "evidence": {
                    "kind": "interrupt",
                    "pause_generation": result["pause_generation"],
                    "cancel_generation": result["cancel_generation"],
                    "prior_status": result["prior_status"],
                    "interrupt_provenance_sha256": result[
                        "interrupt_provenance_sha256"
                    ],
                },
            }
        )
        runtime.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        )

        with pytest.raises(ValidationError, match="admission"):
            runtime.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        ) == before
        assert before[2].result["settlement_state"] == "pending"
    finally:
        runtime.close()


def test_terminalize_lower_generation_tamper_cannot_complete_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "terminalize-lower-generation-admission.sqlite",
        config=_config(),
    )
    command_id = "terminalize-lower-generation-admission"
    try:
        created = _create(
            runtime,
            request_id="create-terminalize-lower-generation-admission",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        attention = runtime.task_runs._mark_attention(
            runtime.store.get_task_run(created.run_id),
            runtime.task_runs._blocker(
                "binding_drift",
                "test recovery termination evidence",
            ),
        )
        seeded = runtime.store.update_task_run_cas(
            created.run_id,
            attention.revision,
            updates={"cancel_generation": 1, "updated_at": utc_now()},
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )

        def crash_after_termination_admission(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after terminalize admission evidence")

        monkeypatch.setattr(
            runtime.task_runs,
            "_settle_recover_terminate_command",
            crash_after_termination_admission,
        )
        with pytest.raises(RuntimeError, match="terminalize admission evidence"):
            runtime.task_runs.recover(
                created.run_id,
                option_id="terminate_run",
                expected_revision=seeded.revision,
                command_id=command_id,
            )
        monkeypatch.undo()
        pending = runtime.store.get_task_run_command(created.run_id, command_id)
        assert pending is not None
        assert pending.result["cancel_generation"] == 2
        assert pending.result["settlement_state"] == "pending"

        runtime.process.cancel(created.root_pid, "terminalize receipt validation")
        terminal = runtime.task_runs._project(
            runtime.store.get_task_run(created.run_id),
            allow_finalize=True,
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        result = json.loads(json.dumps(pending.result))
        result["cancel_generation"] = 1
        result["admission_evidence_sha256"] = runtime.task_runs._sha256(
            {
                "schema_version": 1,
                "run_id": pending.run_id,
                "command_id": pending.command_id,
                "command_kind": pending.command_kind,
                "request_hash": pending.request_hash,
                "evidence": {
                    "kind": "terminalize",
                    "cancel_generation": 1,
                },
            }
        )
        runtime.store._execute(  # noqa: SLF001 - persisted receipt corruption
            "UPDATE task_run_commands SET result_json = ? "
            "WHERE run_id = ? AND command_id = ?",
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                created.run_id,
                command_id,
            ),
        )
        before = (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
        )

        with pytest.raises(ValidationError, match="admission"):
            runtime.task_runs.recover(
                created.run_id,
                option_id="terminate_run",
                expected_revision=seeded.revision,
                command_id=command_id,
            )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
        ) == before
        assert before[2].result["settlement_state"] == "pending"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "family",
    ("run", "resume", "cancel", "deadline", "interrupt", "terminalize"),
)
def test_stale_runtime_cannot_complete_any_terminal_control_receipt(
    family: str,
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"stale-terminal-control-{family}.sqlite",
        config=_config(),
    )
    command_id = f"stale-terminal-control-{family}"
    try:
        created = _create(
            runtime,
            request_id=f"create-{command_id}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        runtime.process.cancel(created.root_pid, "prepare terminal control receipt")
        terminal = runtime.task_runs._project(
            runtime.store.get_task_run(created.run_id),
            allow_finalize=True,
        )
        assert terminal.status in {
            TaskRunStatus.FAILED,
            TaskRunStatus.CANCELLED,
        }
        if family in {"cancel", "deadline", "terminalize"}:
            terminal = runtime.store.update_task_run_cas(
                created.run_id,
                terminal.revision,
                updates={"cancel_generation": 1, "updated_at": utc_now()},
                expected_runtime_epoch=runtime.task_runs.runtime_epoch,
            )
        if family in {"resume", "interrupt"}:
            terminal = runtime.store.update_task_run_cas(
                created.run_id,
                terminal.revision,
                updates={"pause_generation": 1, "updated_at": utc_now()},
                expected_runtime_epoch=runtime.task_runs.runtime_epoch,
            )

        if family == "run":
            command_kind = "run"
            request = {
                "expected_revision": terminal.revision,
                "max_quanta": None,
            }
            evidence = {
                "kind": "run",
                "admission_revision": terminal.revision,
            }
            result = {
                "settlement_state": "pending",
                "admission_revision": terminal.revision,
            }
            label = "explicit dispatch admitted"
        elif family == "resume":
            command_kind = "resume"
            request = {"expected_revision": terminal.revision}
            evidence = {"kind": "resume", "pause_generation": 1}
            result = {
                "settlement_state": "pending",
                "pause_generation": 1,
            }
            label = "resume admitted"
        elif family == "cancel":
            command_kind = "cancel"
            request = {
                "expected_revision": terminal.revision,
                "reason": "stale terminal cancel",
            }
            evidence = {"kind": "cancel", "cancel_generation": 1}
            result = {
                "settlement_state": "pending",
                "cancel_generation": 1,
            }
            label = "cancel intent persisted"
        elif family == "deadline":
            command_kind = "run"
            request = {
                "expected_revision": terminal.revision,
                "max_quanta": None,
            }
            evidence = {"kind": "deadline", "cancel_generation": 1}
            result = {
                "settlement_state": "pending",
                "settlement_kind": "deadline",
                "cancel_generation": 1,
            }
            label = "deadline command admitted"
        elif family == "interrupt":
            command_kind = "follow_up"
            request = {
                "expected_revision": terminal.revision,
                "body": {"constraint": "stale runtime must not settle"},
                "kind": "interrupt",
                "required": True,
            }
            interrupt = runtime.task_runs._interrupt_follow_up_result(
                settlement_state="pending",
                pause_generation=1,
                cancel_generation=0,
                prior_status=TaskRunStatus.RUNNING,
                admission_runtime_epoch=runtime.task_runs.runtime_epoch,
                resume_fences=(),
            )
            evidence = {
                "kind": "interrupt",
                "pause_generation": 1,
                "cancel_generation": 0,
                "prior_status": TaskRunStatus.RUNNING.value,
                "interrupt_provenance_sha256": interrupt[
                    "interrupt_provenance_sha256"
                ],
            }
            result = interrupt
            label = "interrupt generation persisted"
        else:
            command_kind = "recover"
            request = {
                "expected_revision": terminal.revision,
                "option_id": "terminate_run",
                "receipt": {},
            }
            evidence = {"kind": "terminalize", "cancel_generation": 1}
            result = {
                "settlement_state": "pending",
                "settlement_kind": "terminalize",
                "cancel_generation": 1,
            }
            label = "manual recovery termination intent persisted"

        with runtime.uow.transaction():
            result.update(
                runtime.task_runs._append_control_admission(
                    terminal,
                    terminal,
                    command_id=command_id,
                    command_kind=command_kind,
                    request=request,
                    evidence=evidence,
                    label=label,
                )
            )
            runtime.task_runs._record_command(
                terminal,
                command_id,
                command_kind,
                request,
                result=result,
            )

        old_epoch = runtime.task_runs.runtime_epoch
        new_epoch = runtime.store.claim_runtime_epoch(
            f"new-runtime-after-{family}"
        )
        assert new_epoch > old_epoch
        rebound = runtime.store.claim_terminal_task_run_epoch(
            created.run_id,
            terminal.revision,
            new_epoch,
        )
        before = (
            rebound,
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        )

        with pytest.raises(TaskRunRevisionConflict):
            if family in {"run", "deadline"}:
                runtime.task_runs.run_until_blocked(
                    created.run_id,
                    expected_revision=request["expected_revision"],
                    command_id=command_id,
                )
            elif family == "resume":
                runtime.task_runs.resume(
                    created.run_id,
                    expected_revision=request["expected_revision"],
                    command_id=command_id,
                )
            elif family == "cancel":
                runtime.task_runs.cancel(
                    created.run_id,
                    expected_revision=request["expected_revision"],
                    command_id=command_id,
                    reason="stale terminal cancel",
                )
            elif family == "interrupt":
                runtime.task_runs.follow_up(
                    created.run_id,
                    request["body"],
                    kind="interrupt",
                    expected_revision=request["expected_revision"],
                    command_id=command_id,
                )
            else:
                runtime.task_runs.recover(
                    created.run_id,
                    option_id="terminate_run",
                    expected_revision=request["expected_revision"],
                    command_id=command_id,
                )

        assert (
            runtime.store.get_task_run(created.run_id),
            runtime.process.get(created.root_pid),
            runtime.store.get_task_run_command(created.run_id, command_id),
            tuple(
                runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=500,
                ).records
            ),
        ) == before
        assert before[2].result["settlement_state"] == "pending"
    finally:
        runtime.close()


def test_startup_finalizing_run_projects_and_completes_pending_interrupt_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-pending-finalizing-reopen.sqlite"
    command_id = "interrupt-pending-finalizing-reopen"
    body = {"constraint": "finish ordinary terminal projection on reopen"}
    first = Runtime.open(database, config=_config())
    try:
        created = _create(
            first,
            request_id="create-interrupt-pending-finalizing",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after finalizing interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="finalizing interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )
        first.process.cancel(created.root_pid, "test terminal projection crash")
        interrupted = first.store.get_task_run(created.run_id)
        assert interrupted is not None
        original_update = first.store.update_task_run_cas
        failed_terminal_commit = False

        def fail_after_finalizing(
            run_id: str,
            expected_revision: int,
            *,
            updates: object,
            expected_runtime_epoch: int | None = None,
        ):
            nonlocal failed_terminal_commit
            selected_updates = dict(updates)
            if (
                selected_updates.get("status") is TaskRunStatus.FAILED
                and not failed_terminal_commit
            ):
                failed_terminal_commit = True
                raise RuntimeError("fault after finalizing status commit")
            return original_update(
                run_id,
                expected_revision,
                updates=selected_updates,
                expected_runtime_epoch=expected_runtime_epoch,
            )

        monkeypatch.setattr(
            first.store,
            "update_task_run_cas",
            fail_after_finalizing,
        )
        with pytest.raises(RuntimeError, match="finalizing status commit"):
            first.task_runs._project(interrupted, allow_finalize=True)
        finalizing = first.store.get_task_run(created.run_id)
        pending = first.store.get_task_run_command(created.run_id, command_id)
        assert finalizing is not None
        assert finalizing.status is TaskRunStatus.FINALIZING
        assert finalizing.cancel_generation == 0
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"
        assert pending.result["settlement_kind"] == "interrupt"
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        settled = reopened.task_runs.get(created.run_id)
        command = reopened.store.get_task_run_command(created.run_id, command_id)
        assert settled.status is TaskRunStatus.FAILED
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == settled.revision
        assert command.result["summary"]["revision"] == settled.revision
        usage = reopened.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        reopened.close()


def test_interrupt_follow_up_preserves_queued_explicit_dispatch_boundary(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "interrupt-follow-up-queued.sqlite", config=_config())
    try:
        created = _create(
            runtime,
            request_id="create-interrupt-follow-up-queued",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        settled = runtime.task_runs.follow_up(
            created.run_id,
            {"constraint": "remain queued"},
            kind="interrupt",
            expected_revision=created.revision,
            command_id="interrupt-follow-up-queued",
        )
        command = runtime.store.get_task_run_command(
            created.run_id,
            "interrupt-follow-up-queued",
        )
        assert settled.status is TaskRunStatus.QUEUED
        assert settled.started_at is None
        assert runtime.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        usage = runtime.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        runtime.close()


def test_pending_queued_interrupt_reopens_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-follow-up-queued-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "remain queued across reopen"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-follow-up-queued-reopen",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after queued interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="queued interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=created.revision,
                command_id="interrupt-follow-up-queued-reopen",
            )
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        settled = reopened.task_runs.get(created.run_id)
        command = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-follow-up-queued-reopen",
        )
        assert settled.status is TaskRunStatus.QUEUED
        assert settled.started_at is None
        assert reopened.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        usage = reopened.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("reopen", [False, True])
def test_interrupt_never_resumes_a_process_paused_before_admission(
    reopen: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / f"interrupt-prepaused-{reopen}.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "preserve the independent process pause"}
    try:
        created = _create(
            first,
            request_id=f"create-interrupt-prepaused-{reopen}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        first.process.pause(created.root_pid, "independent Host pause")
        running = first.task_runs.get(created.run_id)
        if reopen:
            def crash_after_interrupt_commit(
                _prior: object,
                **_kwargs: object,
            ) -> object:
                raise RuntimeError("fault after prepaused interrupt admission")

            monkeypatch.setattr(
                first.task_runs,
                "_finish_interrupt_follow_up",
                crash_after_interrupt_commit,
            )
            with pytest.raises(RuntimeError, match="prepaused interrupt admission"):
                first.task_runs.follow_up(
                    created.run_id,
                    body,
                    kind="interrupt",
                    expected_revision=running.revision,
                    command_id=f"interrupt-prepaused-{reopen}",
                )
        else:
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=f"interrupt-prepaused-{reopen}",
            )
            paused = first.process.get(created.root_pid)
            assert paused.status is ProcessStatus.PAUSED
            assert isinstance(paused.wait_state, PausedProcessWait)
            assert not isinstance(paused.wait_state, StaleExecutionProcessWait)
    finally:
        first.close()

    if not reopen:
        return
    reopened = Runtime.open(database, config=_config())
    try:
        paused = reopened.process.get(created.root_pid)
        assert paused.status is ProcessStatus.PAUSED
        assert isinstance(paused.wait_state, PausedProcessWait)
        assert not isinstance(paused.wait_state, StaleExecutionProcessWait)
        command = reopened.store.get_task_run_command(
            created.run_id,
            f"interrupt-prepaused-{reopen}",
        )
        assert command is not None
        assert command.result["settlement_state"] == "complete"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "recovery_projection",
    ["diagnostic_only", "legacy_pause", "tampered_receipt"],
)
def test_interrupt_resume_provenance_is_running_only_and_generation_bound(
    recovery_projection: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / f"interrupt-running-only-{recovery_projection}.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "resume only execution owned at admission"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-running-only-provenance",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        child_pid = first.process.spawn_child(created.root_pid, "runnable sibling")
        for index, pid in enumerate((created.root_pid, child_pid)):
            _record_test_task_run_safe_point(
                first,
                run_id=created.run_id,
                pid=pid,
                call_id=f"running-only-safe-point-{index}",
            )
        running = first.task_runs.get(created.run_id)
        assert first.store.claim_execution(
            created.root_pid,
            owner_id="interrupt-running-only-crashed-runtime",
        ) is not None
        admission_process = first.process.get(created.root_pid)
        admission_generation = admission_process.state_generation
        admission_execution_generation = (
            admission_process.execution_generation
        )

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after running-only interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="running-only interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-running-only-provenance",
            )
        pending = first.store.get_task_run_command(
            created.run_id,
            "interrupt-running-only-provenance",
        )
        assert pending is not None
        assert (
            pending.result["admission_runtime_epoch"]
            == first.task_runs.runtime_epoch
        )
        assert pending.result["resume_fences"] == [
            [
                created.root_pid,
                admission_generation,
                admission_execution_generation,
            ]
        ]
        assert child_pid not in {item[0] for item in pending.result["resume_fences"]}
    finally:
        first.close()

    original_recover_startup = TaskRunManager.recover_startup
    observed_receipts: list[StaleExecutionProcessWait] = []

    def inspect_typed_recovery(
        self: TaskRunManager,
    ) -> tuple[object, ...]:
        process = self._store.get_process(created.root_pid)  # noqa: SLF001
        assert process is not None
        assert isinstance(process.wait_state, StaleExecutionProcessWait)
        observed_receipts.append(process.wait_state)
        wait_mapping = process_wait_state_to_mapping(process.wait_state)
        assert wait_mapping is not None
        if recovery_projection == "legacy_pause":
            wait_mapping = process_wait_state_to_mapping(PausedProcessWait())
        elif recovery_projection == "tampered_receipt":
            wait_mapping["recovered_by_owner_sha256"] = wait_mapping[
                "prior_owner_sha256"
            ]
        self._store._execute(  # noqa: SLF001 - recovery projection fault injection
            "UPDATE processes SET status_message = ?, wait_state_json = ? WHERE pid = ?",
            (
                (
                    "diagnostic text unrelated to machine state"
                    if recovery_projection == "diagnostic_only"
                    else "stale_execution_recovery"
                ),
                dumps(wait_mapping),
                created.root_pid,
            ),
        )
        return original_recover_startup(self)  # type: ignore[return-value]

    monkeypatch.setattr(
        TaskRunManager,
        "recover_startup",
        inspect_typed_recovery,
    )
    reopened = Runtime.open(database, config=_config())
    try:
        settled = reopened.task_runs.get(created.run_id)
        command = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-running-only-provenance",
        )
        assert len(observed_receipts) == 1
        assert reopened.process.get(child_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert "resume_fences" not in command.result
        if recovery_projection == "diagnostic_only":
            assert settled.status is TaskRunStatus.RUNNING
            assert reopened.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        else:
            assert settled.status is TaskRunStatus.NEEDS_ATTENTION
            assert {
                item["kind"] for item in settled.blockers
            } == {"manual_recovery_required"}
            assert reopened.process.get(created.root_pid).status is ProcessStatus.PAUSED
    finally:
        reopened.close()


def test_old_stale_pause_cannot_cross_a_later_interrupt_generation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "interrupt-old-stale-generation.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(
            first,
            request_id="create-interrupt-old-stale-generation",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        child_pid = first.process.spawn_child(created.root_pid, "live sibling")
        for index, pid in enumerate((created.root_pid, child_pid)):
            _record_test_task_run_safe_point(
                first,
                run_id=created.run_id,
                pid=pid,
                call_id=f"old-stale-safe-point-{index}",
            )
        assert first.store.claim_execution(
            created.root_pid,
            owner_id="older-crashed-runtime",
        ) is not None
    finally:
        first.close()

    second = Runtime.open(database, config=_config())
    try:
        before = second.task_runs.get(created.run_id)
        stale = second.process.get(created.root_pid)
        assert before.status is TaskRunStatus.RUNNING
        assert stale.status is ProcessStatus.PAUSED
        assert stale.status_message == "stale_execution_recovery"
        assert second.process.get(child_pid).status is ProcessStatus.RUNNABLE

        blocked = second.task_runs.follow_up(
            created.run_id,
            {"constraint": "do not adopt the older stale pause"},
            kind="interrupt",
            expected_revision=before.revision,
            command_id="interrupt-after-older-stale-pause",
        )
        command = second.store.get_task_run_command(
            created.run_id,
            "interrupt-after-older-stale-pause",
        )
        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in blocked.blockers} == {
            "manual_recovery_required"
        }
        assert second.process.get(created.root_pid).status is ProcessStatus.PAUSED
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert "resume_fences" not in command.result
    finally:
        second.close()


def test_oversized_interrupt_receipt_rolls_back_every_durable_write(
    tmp_path: Path,
) -> None:
    database = tmp_path / "interrupt-receipt-byte-cap.sqlite"
    creator = Runtime.open(database, config=_config())
    try:
        created = _create(
            creator,
            request_id="create-interrupt-receipt-byte-cap",
            retention=TaskRunRetention.PERMANENT,
        )
        create_command = creator.store.get_task_run_command(
            created.run_id,
            "create:create-interrupt-receipt-byte-cap",
        )
        assert create_command is not None
        existing_result_bytes = len(
            json.dumps(
                create_command.result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    finally:
        creator.close()

    ordinary = _config()
    limited = replace(
        ordinary,
        task_runs=replace(
            ordinary.task_runs,
            # Keep already-persisted base receipts readable while proving the
            # larger interrupt provenance receipt is rejected atomically.
            command_result_max_bytes=existing_result_bytes + 32,
        ),
    )
    runtime = Runtime.open(database, config=limited)
    try:
        assert created.root_pid is not None
        queued = runtime.store.get_task_run(created.run_id)
        assert queued is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        assert runtime.store.claim_execution(
            created.root_pid,
            owner_id="oversized-interrupt-receipt",
        ) is not None
        before = runtime.store.get_task_run(created.run_id)
        requirements_before = runtime.store.list_task_run_requirements(created.run_id)
        payloads_before = runtime.store.list_task_run_payloads(created.run_id)
        ledger_before = runtime.store.list_task_run_ledger(
            created.run_id,
            after=None,
            limit=500,
        ).records
        messages_before = runtime.store.list_process_messages(created.root_pid)
        commands_before = runtime.store.list_task_run_commands(
            created.run_id,
            limit=100,
        )
        assert before is not None

        with pytest.raises(
            ValidationError,
            match="command result exceeds configured maximum",
        ):
            runtime.task_runs.follow_up(
                created.run_id,
                {"constraint": "must roll back atomically"},
                kind="interrupt",
                expected_revision=before.revision,
                command_id="oversized-interrupt-receipt",
            )

        assert runtime.store.get_task_run(created.run_id) == before
        assert runtime.store.list_task_run_requirements(
            created.run_id
        ) == requirements_before
        assert runtime.store.list_task_run_payloads(created.run_id) == payloads_before
        assert runtime.store.list_task_run_ledger(
            created.run_id,
            after=None,
            limit=500,
        ).records == ledger_before
        assert runtime.store.list_process_messages(
            created.root_pid
        ) == messages_before
        assert runtime.store.list_task_run_commands(
            created.run_id,
            limit=100,
        ) == commands_before
        assert runtime.store.get_task_run_command(
            created.run_id,
            "oversized-interrupt-receipt",
        ) is None
    finally:
        runtime.close()


def test_pending_interrupt_blocks_resume_and_repause_until_old_action_is_superseded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "interrupt-control-barrier.sqlite", config=_config())
    body = {"constraint": "replace the old validated action"}
    try:
        created = _create(
            runtime,
            request_id="create-interrupt-control-barrier",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = runtime.store.get_task_run(created.run_id)
        assert queued is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        running = runtime.task_runs.get(created.run_id)
        _claim_test_task_run_action(
            runtime,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id="old-interrupt-action",
            action={"action": "process_exit", "payload": {"message": "OLD"}},
        )
        running = runtime.task_runs.get(created.run_id)

        original_finish = runtime.task_runs._finish_interrupt_follow_up

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after control-barrier interrupt admission")

        monkeypatch.setattr(
            runtime.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="control-barrier interrupt admission"):
            runtime.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-control-barrier",
            )
        interrupted = runtime.task_runs.get(created.run_id)
        assert interrupted.status is TaskRunStatus.PAUSED

        with pytest.raises(ValidationError, match="interrupt settlement is pending"):
            runtime.task_runs.resume(
                created.run_id,
                expected_revision=interrupted.revision,
                command_id="resume-before-interrupt-settlement",
            )
        with pytest.raises(ValidationError, match="cannot pause from paused"):
            runtime.task_runs.pause(
                created.run_id,
                expected_revision=interrupted.revision,
                command_id="repause-before-interrupt-settlement",
            )
        assert runtime.store.get_task_run_command(
            created.run_id,
            "resume-before-interrupt-settlement",
        ) is None
        assert runtime.store.get_task_run_command(
            created.run_id,
            "repause-before-interrupt-settlement",
        ) is None

        monkeypatch.setattr(
            runtime.task_runs,
            "_finish_interrupt_follow_up",
            original_finish,
        )
        settled = runtime.task_runs.follow_up(
            created.run_id,
            body,
            kind="interrupt",
            expected_revision=running.revision,
            command_id="interrupt-control-barrier",
        )
        point = runtime.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert settled.status is TaskRunStatus.RUNNING
        assert point is not None
        assert point.pending_action_payload_id is None
        usage = runtime.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        runtime.close()


def test_pending_interrupt_second_reopen_finishes_after_run_status_restore_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-second-reopen-status-restored.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "finish after a second startup crash"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-second-reopen-status-restored",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        _claim_test_task_run_action(
            first,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id="interrupt-second-reopen-status-old-action",
            action={"action": "process_exit", "payload": {"message": "OLD"}},
        )
        running = first.task_runs.get(created.run_id)
        assert first.store.claim_execution(
            created.root_pid,
            owner_id="interrupt-crashed-runtime",
        ) is not None

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after second-reopen interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="second-reopen interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-second-reopen-status-restored",
            )
    finally:
        first.close()

    original_resume_phase = TaskRunManager._resume_reopened_interrupt_processes

    def crash_after_run_restore(
        self: TaskRunManager,
        current: object,
        **_kwargs: object,
    ) -> object:
        assert current.status is TaskRunStatus.RUNNING
        raise RuntimeError("fault after interrupt Run status restore")

    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        crash_after_run_restore,
    )
    with pytest.raises(RuntimeError, match="Run status restore"):
        Runtime.open(database, config=_config())
    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        original_resume_phase,
    )

    third = Runtime.open(database, config=_config())
    try:
        settled = third.task_runs.get(created.run_id)
        command = third.store.get_task_run_command(
            created.run_id,
            "interrupt-second-reopen-status-restored",
        )
        assert settled.status is TaskRunStatus.RUNNING
        assert third.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        usage = third.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        third.close()


@pytest.mark.parametrize("settled_state", ["waiting", "terminal"])
def test_same_runtime_interrupt_accepts_legitimate_quantum_settlement(
    settled_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"interrupt-same-runtime-{settled_state}.sqlite",
        config=_config(),
    )
    try:
        created = _create(
            runtime,
            request_id=f"create-same-runtime-{settled_state}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = runtime.store.get_task_run(created.run_id)
        assert queued is not None
        runtime.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        _record_test_task_run_safe_point(
            runtime,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id=f"same-runtime-{settled_state}-safe-point",
        )
        token = runtime.store.claim_execution(
            created.root_pid,
            owner_id=f"same-runtime-{settled_state}-worker",
        )
        assert token is not None
        running = runtime.task_runs.get(created.run_id)

        def settle_quantum(_run_id: str) -> None:
            if settled_state == "waiting":
                assert runtime.store.complete_execution(
                    token,
                    status=ProcessStatus.WAITING_EVENT,
                    wait_state=MessageProcessWait(
                        filters={"channel": "same-runtime"}
                    ),
                )
            else:
                assert runtime.store.complete_execution(
                    token,
                    status=ProcessStatus.EXITED,
                    outcome=ExitedProcessOutcome(),
                )

        monkeypatch.setattr(
            runtime.task_runs,
            "_wait_for_dispatch_drain",
            settle_quantum,
        )
        summary = runtime.task_runs.follow_up(
            created.run_id,
            {"constraint": f"preserve {settled_state} quantum settlement"},
            kind="interrupt",
            expected_revision=running.revision,
            command_id=f"interrupt-same-runtime-{settled_state}",
        )
        process = runtime.process.get(created.root_pid)
        command = runtime.store.get_task_run_command(
            created.run_id,
            f"interrupt-same-runtime-{settled_state}",
        )
        assert process.status is (
            ProcessStatus.WAITING_EVENT
            if settled_state == "waiting"
            else ProcessStatus.EXITED
        )
        assert "manual_recovery_required" not in {
            item["kind"] for item in summary.blockers
        }
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result["summary"]["status"] == summary.status.value
        assert process.resource_usage.llm_calls == 0
        assert process.resource_usage.tool_calls == 0
    finally:
        runtime.close()


def test_pending_interrupt_second_reopen_finishes_partial_process_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-second-reopen-partial-tree.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "resume every stale process exactly once"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-second-reopen-partial-tree",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        child_pid = first.process.spawn_child(created.root_pid, "durable child")
        for index, pid in enumerate((created.root_pid, child_pid)):
            _claim_test_task_run_action(
                first,
                run_id=created.run_id,
                pid=pid,
                call_id=f"interrupt-partial-tree-old-action-{index}",
                action={
                    "action": "process_exit",
                    "payload": {"message": f"OLD-{index}"},
                },
            )
        running = first.task_runs.get(created.run_id)
        for pid in (created.root_pid, child_pid):
            assert first.store.claim_execution(
                pid,
                owner_id="interrupt-partial-crashed-runtime",
            ) is not None

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after partial-tree interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="partial-tree interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-second-reopen-partial-tree",
            )
    finally:
        first.close()

    original_resume_phase = TaskRunManager._resume_reopened_interrupt_processes

    def crash_after_first_process_resume(
        self: TaskRunManager,
        current: object,
        **_kwargs: object,
    ) -> object:
        assert current.status is TaskRunStatus.RUNNING
        for process in self._tree_processes(current.run_id):
            if self._is_stale_execution_pause(process):
                self._process.resume(process.pid)
                raise RuntimeError("fault after first interrupt process resume")
        raise AssertionError("test expected one stale execution pause")

    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        crash_after_first_process_resume,
    )
    with pytest.raises(RuntimeError, match="first interrupt process resume"):
        Runtime.open(database, config=_config())
    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        original_resume_phase,
    )

    third = Runtime.open(database, config=_config())
    try:
        settled = third.task_runs.get(created.run_id)
        command = third.store.get_task_run_command(
            created.run_id,
            "interrupt-second-reopen-partial-tree",
        )
        assert settled.status is TaskRunStatus.RUNNING
        assert {
            third.process.get(pid).status
            for pid in (created.root_pid, child_pid)
        } == {ProcessStatus.RUNNABLE}
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        for pid in (created.root_pid, child_pid):
            usage = third.process.get(pid).resource_usage
            assert usage.llm_calls == 0
            assert usage.tool_calls == 0
    finally:
        third.close()


@pytest.mark.parametrize(
    "tamper",
    ["state_generation", "waiting_state", "execution_lease"],
)
def test_pending_interrupt_partial_resume_tamper_blocks_before_any_new_resume(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / f"interrupt-partial-resume-tamper-{tamper}.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "never trust a mutated partial resume"}
    command_id = f"interrupt-partial-resume-tamper-{tamper}"
    try:
        created = _create(
            first,
            request_id=f"create-{command_id}",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        child_pid = first.process.spawn_child(
            created.root_pid,
            "partial-resume tamper sibling",
        )
        member_pids = (created.root_pid, child_pid)
        for index, pid in enumerate(member_pids):
            _record_test_task_run_safe_point(
                first,
                run_id=created.run_id,
                pid=pid,
                call_id=f"{command_id}-safe-point-{index}",
            )
            assert first.store.claim_execution(
                pid,
                owner_id="interrupt-partial-tamper-crashed-runtime",
            ) is not None
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after partial-tamper interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="partial-tamper interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id=command_id,
            )
    finally:
        first.close()

    original_resume_phase = TaskRunManager._resume_reopened_interrupt_processes
    resumed_pid: list[str] = []

    def crash_after_first_process_resume(
        self: TaskRunManager,
        current: object,
        **_kwargs: object,
    ) -> object:
        assert current.status is TaskRunStatus.RUNNING
        for process in self._tree_processes(current.run_id):
            if self._is_stale_execution_pause(process):
                self._process.resume(process.pid)
                resumed_pid.append(process.pid)
                raise RuntimeError("fault after one partial-tamper process resume")
        raise AssertionError("test expected one stale execution pause")

    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        crash_after_first_process_resume,
    )
    with pytest.raises(RuntimeError, match="one partial-tamper process resume"):
        Runtime.open(database, config=_config())
    monkeypatch.setattr(
        TaskRunManager,
        "_resume_reopened_interrupt_processes",
        original_resume_phase,
    )
    assert len(resumed_pid) == 1
    untouched_pid = next(pid for pid in member_pids if pid != resumed_pid[0])

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        untouched_before = connection.execute(
            "SELECT status, state_generation, revision, wait_state_json "
            "FROM processes WHERE pid = ?",
            (untouched_pid,),
        ).fetchone()
        assert untouched_before is not None
        assert untouched_before["status"] == ProcessStatus.PAUSED.value
        if tamper == "state_generation":
            connection.execute(
                "UPDATE processes SET state_generation = state_generation + 1, "
                "revision = revision + 1 WHERE pid = ?",
                (resumed_pid[0],),
            )
        elif tamper == "waiting_state":
            wait_state_json = dumps(
                process_wait_state_to_mapping(
                    MessageProcessWait(filters={"channel": "tampered"})
                )
            )
            connection.execute(
                "UPDATE processes SET status = ?, status_message = ?, "
                "wait_state_json = ?, revision = revision + 1 WHERE pid = ?",
                (
                    ProcessStatus.WAITING_EVENT.value,
                    'waiting_message:{"channel": "tampered"}',
                    wait_state_json,
                    resumed_pid[0],
                ),
            )
        else:
            connection.execute(
                "UPDATE processes SET execution_owner_id = ?, "
                "execution_lease_id = ?, revision = revision + 1 WHERE pid = ?",
                (
                    "tampered-partial-owner",
                    "tampered-partial-lease",
                    resumed_pid[0],
                ),
            )
        connection.commit()

    third = Runtime.open(database, config=_config())
    try:
        settled = third.task_runs.get(created.run_id)
        command = third.store.get_task_run_command(created.run_id, command_id)
        untouched_after = third.process.get(untouched_pid)
        assert settled.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in settled.blockers} == {
            "manual_recovery_required"
        }
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result["summary"]["status"] == "needs_attention"
        assert untouched_after.status is ProcessStatus.PAUSED
        assert isinstance(
            untouched_after.wait_state,
            StaleExecutionProcessWait,
        )
        assert untouched_after.state_generation == untouched_before[
            "state_generation"
        ]
        assert untouched_after.revision == untouched_before["revision"]
        assert (
            dumps(process_wait_state_to_mapping(untouched_after.wait_state))
            == untouched_before["wait_state_json"]
        )
        assert third.run_next_process_once() is None
        for pid in member_pids:
            usage = third.process.get(pid).resource_usage
            assert usage.llm_calls == 0
            assert usage.tool_calls == 0
    finally:
        third.close()


def test_completed_startup_interrupt_receipt_is_idempotent_after_open_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-command-complete-open-crash.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "complete the command once"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-command-complete-open-crash",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        _claim_test_task_run_action(
            first,
            run_id=created.run_id,
            pid=created.root_pid,
            call_id="interrupt-command-complete-old-action",
            action={"action": "process_exit", "payload": {"message": "OLD"}},
        )
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after command-complete interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="command-complete interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-command-complete-open-crash",
            )
    finally:
        first.close()

    original_complete = TaskRunManager._complete_startup_interrupt_command

    def crash_after_command_complete(
        self: TaskRunManager,
        current: object,
        pending: object,
    ) -> object:
        original_complete(self, current, pending)
        raise RuntimeError("fault after interrupt command completion")

    monkeypatch.setattr(
        TaskRunManager,
        "_complete_startup_interrupt_command",
        crash_after_command_complete,
    )
    with pytest.raises(RuntimeError, match="interrupt command completion"):
        Runtime.open(database, config=_config())
    monkeypatch.setattr(
        TaskRunManager,
        "_complete_startup_interrupt_command",
        original_complete,
    )

    third = Runtime.open(database, config=_config())
    try:
        settled = third.task_runs.get(created.run_id)
        command = third.store.get_task_run_command(
            created.run_id,
            "interrupt-command-complete-open-crash",
        )
        point = third.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert settled.status is TaskRunStatus.RUNNING
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert point is not None and point.pending_action_payload_id is None
        usage = third.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        third.close()


def test_terminal_run_wins_over_pending_interrupt_until_exact_client_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "interrupt-pending-terminal-replay.sqlite"
    first = Runtime.open(database, config=_config())
    body = {"constraint": "terminal cancellation must win"}
    try:
        created = _create(
            first,
            request_id="create-interrupt-pending-terminal-replay",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        queued = first.store.get_task_run(created.run_id)
        assert queued is not None
        first.store.update_task_run_cas(
            created.run_id,
            queued.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=first.task_runs.runtime_epoch,
        )
        running = first.task_runs.get(created.run_id)

        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after terminal-replay interrupt admission")

        monkeypatch.setattr(
            first.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="terminal-replay interrupt admission"):
            first.task_runs.follow_up(
                created.run_id,
                body,
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-pending-terminal-replay",
            )
        interrupted = first.task_runs.get(created.run_id)
        terminal = first.task_runs.cancel(
            created.run_id,
            expected_revision=interrupted.revision,
            command_id="cancel-pending-terminal-replay",
        )
        assert terminal.status is TaskRunStatus.CANCELLED
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        pending = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-pending-terminal-replay",
        )
        assert reopened.task_runs.get(created.run_id).status is TaskRunStatus.CANCELLED
        assert pending is not None
        assert pending.result["settlement_state"] == "pending"

        replayed = reopened.task_runs.follow_up(
            created.run_id,
            body,
            kind="interrupt",
            expected_revision=running.revision,
            command_id="interrupt-pending-terminal-replay",
        )
        completed = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-pending-terminal-replay",
        )
        assert replayed.status is TaskRunStatus.CANCELLED
        assert completed is not None
        assert completed.result["settlement_state"] == "complete"
        usage = reopened.process.get(created.root_pid).resource_usage
        assert usage.llm_calls == 0
        assert usage.tool_calls == 0
    finally:
        reopened.close()


def test_pending_cancel_receipt_converges_to_terminal_result_after_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "cancel-pending-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(
            first,
            request_id="create-cancel-pending-reopen",
            retention=TaskRunRetention.PERMANENT,
        )
        original_complete = first.task_runs._complete_command_summary

        def lose_cancel_result(
            _record: object,
            _command_id: str,
            command_kind: str,
            _request: object,
            **_kwargs: object,
        ) -> object:
            if command_kind == "cancel":
                raise RuntimeError("injected cancel result loss")
            return original_complete(
                _record,
                _command_id,
                command_kind,
                _request,
                **_kwargs,
            )

        monkeypatch.setattr(
            first.task_runs,
            "_complete_command_summary",
            lose_cancel_result,
        )
        with pytest.raises(RuntimeError, match="cancel result loss"):
            first.task_runs.cancel(
                created.run_id,
                expected_revision=created.revision,
                command_id="cancel-with-pending-result",
            )
        terminal_before_reopen = first.task_runs.get(created.run_id)
        command = first.store.get_task_run_command(
            created.run_id,
            "cancel-with-pending-result",
        )
        assert terminal_before_reopen.status is TaskRunStatus.CANCELLED
        assert command is not None
        assert command.result["settlement_state"] == "pending"

        before_conflict = (terminal_before_reopen, command)
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            first.task_runs.cancel(
                created.run_id,
                expected_revision=terminal_before_reopen.revision,
                command_id="cancel-with-pending-result",
            )
        assert (
            first.task_runs.get(created.run_id),
            first.store.get_task_run_command(
                created.run_id,
                "cancel-with-pending-result",
            ),
        ) == before_conflict
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        replayed = reopened.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-with-pending-result",
        )
        command = reopened.store.get_task_run_command(
            created.run_id,
            "cancel-with-pending-result",
        )
        assert replayed == reopened.task_runs.get(created.run_id)
        assert replayed.status is TaskRunStatus.CANCELLED
        assert replayed.revision == terminal_before_reopen.revision
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == replayed.revision
    finally:
        reopened.close()


def test_pending_run_receipt_settles_locally_without_redispatch_after_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "run-pending-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    dispatch_count = 0
    try:
        created = _create(
            first,
            request_id="create-run-pending-reopen",
            retention=TaskRunRetention.PERMANENT,
        )
        assert created.root_pid is not None
        original_complete = first.task_runs._complete_command_summary

        def finish_root_without_provider_dispatch(**_kwargs: object) -> list[object]:
            nonlocal dispatch_count
            dispatch_count += 1
            first.process.cancel(created.root_pid or "", "test terminal run result")
            return []

        def lose_run_result(
            _record: object,
            _command_id: str,
            command_kind: str,
            _request: object,
            **_kwargs: object,
        ) -> object:
            if command_kind == "run":
                raise RuntimeError("injected run result loss")
            return original_complete(
                _record,
                _command_id,
                command_kind,
                _request,
                **_kwargs,
            )

        monkeypatch.setattr(
            first,
            "run_until_idle",
            finish_root_without_provider_dispatch,
        )
        monkeypatch.setattr(
            first.task_runs,
            "_complete_command_summary",
            lose_run_result,
        )
        with pytest.raises(RuntimeError, match="run result loss"):
            first.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=created.revision,
                command_id="run-with-pending-result",
                max_quanta=1,
            )
        terminal_before_reopen = first.task_runs.get(created.run_id)
        command = first.store.get_task_run_command(
            created.run_id,
            "run-with-pending-result",
        )
        assert terminal_before_reopen.status is TaskRunStatus.FAILED
        assert command is not None
        assert command.result["settlement_state"] == "pending"
        assert dispatch_count == 1

        before_conflict = (terminal_before_reopen, command, dispatch_count)
        with pytest.raises(TaskRunCommandConflict, match="different request"):
            first.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=terminal_before_reopen.revision,
                command_id="run-with-pending-result",
                max_quanta=1,
            )
        assert (
            first.task_runs.get(created.run_id),
            first.store.get_task_run_command(
                created.run_id,
                "run-with-pending-result",
            ),
            dispatch_count,
        ) == before_conflict
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        replayed = reopened.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-with-pending-result",
            max_quanta=1,
        )
        command = reopened.store.get_task_run_command(
            created.run_id,
            "run-with-pending-result",
        )
        assert replayed == reopened.task_runs.get(created.run_id)
        assert replayed.status is TaskRunStatus.FAILED
        assert replayed.revision == terminal_before_reopen.revision
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert command.result_revision == replayed.revision
        assert dispatch_count == 1
    finally:
        reopened.close()


def test_recovery_termination_cannot_bypass_active_external_dispatch_barrier(
    tmp_path: Path,
) -> None:
    database = tmp_path / "recover-active-dispatch.sqlite"
    runtime = Runtime.open(database, config=_config())
    release_dispatch = threading.Event()
    dispatch_entered = threading.Event()
    try:
        created = _create(
            runtime,
            request_id="create-recover-active-dispatch",
        )
        assert created.root_pid is not None
        current = runtime.store.get_task_run(created.run_id)
        assert current is not None
        running = runtime.store.update_task_run_cas(
            created.run_id,
            current.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )

        def hold_external_dispatch() -> None:
            with runtime.task_runs.dispatch_scope_for_pid(
                created.root_pid or "",
                "tool",
            ):
                dispatch_entered.set()
                assert release_dispatch.wait(timeout=10)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(hold_external_dispatch)
            assert dispatch_entered.wait(timeout=10)
            attention = runtime.task_runs._mark_attention(
                running,
                runtime.task_runs._blocker(
                    "binding_drift",
                    "test manual recovery boundary",
                ),
            )
            result = runtime.task_runs.recover(
                created.run_id,
                option_id="terminate_run",
                expected_revision=attention.revision,
                command_id="terminate-with-active-dispatch",
            )
            persisted = runtime.store.get_task_run(created.run_id)
            assert persisted is not None
            assert result.status is TaskRunStatus.CANCELLING
            assert persisted.status is TaskRunStatus.CANCELLING
            assert persisted.completed_at is None
            assert persisted.payloads_purged_at is None
            assert all(
                payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
                for payload in runtime.store.list_task_run_payloads(created.run_id)
            )
            release_dispatch.set()
            future.result(timeout=10)
    finally:
        release_dispatch.set()
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        converged = reopened.task_runs.get(created.run_id)
        assert converged.status is TaskRunStatus.CANCELLED
        assert converged.payloads_purged is True
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("process_status", "wait_state", "run_status"),
    (
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq-durable",)),
            TaskRunStatus.WAITING_HUMAN,
        ),
        (
            ProcessStatus.WAITING_TOOL,
            ToolProcessWait(operation_id="op-durable"),
            TaskRunStatus.WAITING_TOOL,
        ),
        (
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "durable-message"}),
            TaskRunStatus.WAITING_MESSAGE,
        ),
        (
            ProcessStatus.WAITING_EVENT,
            ChildProcessWait(child_pid="pid-durable-child"),
            TaskRunStatus.WAITING_PROCESS,
        ),
    ),
    ids=("human", "tool", "message", "child"),
)
def test_resume_projects_the_preserved_typed_wait(
    tmp_path: Path,
    process_status: ProcessStatus,
    wait_state: object,
    run_status: TaskRunStatus,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"resume-{run_status.value}.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime)
        assert created.root_pid is not None
        root = runtime.process.get(created.root_pid)
        waiting = runtime.process_transitions.transition(
            created.root_pid,
            process_status,
            expected_revision=root.revision,
            expected_status=root.status,
            expected_state_generation=root.state_generation,
            wait_state=wait_state,
        )
        paused = runtime.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"pause-{run_status.value}",
        )
        resumed = runtime.task_runs.resume(
            created.run_id,
            expected_revision=paused.revision,
            command_id=f"resume-{run_status.value}",
        )
        persisted = runtime.process.get(created.root_pid)

        assert resumed.status is run_status
        assert persisted.status is waiting.status
        assert persisted.wait_state == waiting.wait_state
        command = runtime.store.get_task_run_command(
            created.run_id,
            f"resume-{run_status.value}",
        )
        assert command is not None
        assert command.result["settlement_state"] == "complete"
    finally:
        runtime.close()


def test_list_reconciles_an_elapsed_absolute_deadline_before_returning(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "list-expired-deadline.sqlite", config=_config())
    try:
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=0.2)
        ).isoformat()
        created = _create(runtime, deadline_at=deadline_at)
        current = runtime.store.get_task_run(created.run_id)
        assert current is not None and current.root_pid is not None
        remaining = datetime.fromisoformat(deadline_at) - datetime.now(timezone.utc)
        time.sleep(max(0.0, remaining.total_seconds()) + 0.05)

        page = runtime.task_runs.list(limit=10)

        assert len(page.records) == 1
        expired = page.records[0]
        assert expired.run_id == created.run_id
        assert expired.status is TaskRunStatus.CANCELLED
        assert "deadline_reached" in {item["kind"] for item in expired.blockers}
        assert runtime.process.get(current.root_pid).status is ProcessStatus.KILLED
        persisted = runtime.store.get_task_run(created.run_id)
        assert persisted is not None
        assert persisted.cancel_generation == 1
        assert persisted.completed_at is not None
    finally:
        runtime.close()


def test_task_run_certifies_activate_skill_tool_binding_transition(
    tmp_path: Path,
) -> None:
    database = tmp_path / "certified-activate-skill.sqlite"
    skill_id = "agent-libos-workspace-navigation"
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    action = {
        "action": "activate_skill",
        "skill_id": skill_id,
        "expected_package_sha256": package.package_sha256,
    }
    runtime = Runtime.open(database, config=_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="activate one trusted navigation Skill",
                display_title="Certified binding transition",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-certified-binding-transition",
        )
        assert created.root_pid is not None
        before = runtime.process.get(created.root_pid)
        before_hash = runtime.task_runs._process_binding_hashes(before)[1]
        client = RecordingActionClient([action])
        runtime.llm.client = client

        running = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-certified-binding-transition",
            max_quanta=1,
        )

        assert running.status is TaskRunStatus.RUNNING
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None and record.completed_step_count == 1
        assert len(client.user_prompts) == 1
        after = runtime.process.get(created.root_pid)
        after_hash = runtime.task_runs._process_binding_hashes(after)[1]
        assert after_hash != before_hash
        point = runtime.store.get_task_run_resume_point(
            created.root_pid,
            complete_only=True,
        )
        assert point is not None
        assert point.pending_action_payload_id is None
        assert point.tool_binding_hash == after_hash
        assert runtime.task_runs._resume_static_integrity_valid(point)
        assert runtime.task_runs._resume_current_binding_valid(point)

        staged = [
            json.loads(payload.canonical_json)["binding_transition"]
            for payload in runtime.store.list_task_run_payloads(created.run_id)
            if payload.role == "pending_action"
            and payload.canonical_json is not None
            and "certified_activate_skill" in payload.canonical_json
        ]
        assert len(staged) == 1
        transition = staged[0]
        assert transition["kind"] == "certified_activate_skill"
        assert transition["skill_id"] == skill_id
        assert transition["package_sha256"] == package.package_sha256
        assert transition["pre_tool_binding_hash"] == before_hash
        assert transition["post_tool_binding_hash"] == after_hash
        staged_items = [
            item
            for item in runtime.store.list_task_run_ledger(
                created.run_id,
                after=None,
                limit=100,
            ).records
            if item.status == "result_staged"
            and item.metadata.get("binding_transition_sha256") is not None
        ]
        assert len(staged_items) == 1
        assert staged_items[0].metadata["binding_transition_sha256"] == (
            runtime.task_runs._sha256(transition)
        )
        llm_calls = after.resource_usage.llm_calls
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        assert recovered.status is TaskRunStatus.RUNNING
        assert {item["kind"] for item in recovered.blockers} == set()
        assert (
            reopened.process.get(created.root_pid).resource_usage.llm_calls
            == llm_calls
        )
    finally:
        reopened.close()


def test_task_run_recovers_staged_activate_skill_transition_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "staged-activate-skill-recovery.sqlite"
    skill_id = "agent-libos-workspace-navigation"
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    runtime = Runtime.open(database, config=_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="recover a fully staged binding transition",
                display_title="Recover staged binding transition",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-staged-binding-recovery",
        )
        assert created.root_pid is not None
        client = RecordingActionClient(
            [
                {
                    "action": "activate_skill",
                    "skill_id": skill_id,
                    "expected_package_sha256": package.package_sha256,
                }
            ]
        )
        runtime.llm.client = client
        monkeypatch.setattr(
            runtime.task_runs,
            "record_completed_transcript",
            lambda **_kwargs: None,
        )

        running = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="stage-binding-before-restart",
            max_quanta=1,
        )

        assert running.status is TaskRunStatus.RUNNING
        before_restart = runtime.store.get_task_run(created.run_id)
        assert before_restart is not None
        assert before_restart.completed_step_count == 0
        point = runtime.store.get_task_run_resume_point(created.root_pid)
        assert point is not None and point.pending_action_payload_id is not None
        staged = runtime.task_runs._decode_pending_resume_payload(point)
        assert staged["kind"] == "completed_outcome"
        assert staged["binding_transition"]["kind"] == "certified_activate_skill"
        def crash_after_interrupt_commit(
            _prior: object,
            **_kwargs: object,
        ) -> object:
            raise RuntimeError("fault after staged-outcome interrupt admission")

        monkeypatch.setattr(
            runtime.task_runs,
            "_finish_interrupt_follow_up",
            crash_after_interrupt_commit,
        )
        with pytest.raises(RuntimeError, match="staged-outcome interrupt admission"):
            runtime.task_runs.follow_up(
                created.run_id,
                {"constraint": "settle the staged provider result first"},
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt-staged-binding-before-restart",
            )
        llm_calls = runtime.process.get(created.root_pid).resource_usage.llm_calls
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        assert recovered.status is TaskRunStatus.RUNNING
        record = reopened.store.get_task_run(created.run_id)
        assert record is not None and record.completed_step_count == 1
        point = reopened.store.get_task_run_resume_point(created.root_pid)
        assert point is not None and point.pending_action_payload_id is None
        assert reopened.task_runs._resume_current_binding_valid(point)
        command = reopened.store.get_task_run_command(
            created.run_id,
            "interrupt-staged-binding-before-restart",
        )
        assert command is not None
        assert command.result["settlement_state"] == "complete"
        assert (
            reopened.process.get(created.root_pid).resource_usage.llm_calls
            == llm_calls
        )
    finally:
        reopened.close()


def test_startup_settles_staged_outcome_before_persisted_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "staged-outcome-before-cancel-recovery.sqlite"
    skill_id = "agent-libos-workspace-navigation"
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    first = Runtime.open(database, config=_config())
    try:
        created = first.task_runs.create(
            TaskRunSpecV1(
                goal="settle a staged result before cancellation",
                display_title="Settle staged result before cancellation",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-staged-outcome-before-cancel",
        )
        assert created.root_pid is not None
        client = RecordingActionClient(
            [
                {
                    "action": "activate_skill",
                    "skill_id": skill_id,
                    "expected_package_sha256": package.package_sha256,
                }
            ]
        )
        first.llm.client = client
        monkeypatch.setattr(
            first.task_runs,
            "record_completed_transcript",
            lambda **_kwargs: None,
        )
        running = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="stage-result-before-cancel",
            max_quanta=1,
        )
        point = first.store.get_task_run_resume_point(created.root_pid)
        assert point is not None and point.pending_action_payload_id is not None
        assert first.task_runs._decode_pending_resume_payload(point)["kind"] == (
            "completed_outcome"
        )

        def crash_after_cancel_intent(_pid: str, _reason: str) -> None:
            raise RuntimeError("fault after durable cancellation admission")

        monkeypatch.setattr(first.task_runs._process, "cancel", crash_after_cancel_intent)
        with pytest.raises(RuntimeError, match="durable cancellation admission"):
            first.task_runs.cancel(
                created.run_id,
                expected_revision=running.revision,
                command_id="cancel-with-staged-outcome",
            )
        persisted = first.store.get_task_run(created.run_id)
        command = first.store.get_task_run_command(
            created.run_id,
            "cancel-with-staged-outcome",
        )
        assert persisted is not None
        assert persisted.status is TaskRunStatus.CANCELLING
        assert persisted.completed_step_count == 0
        assert command is not None
        assert command.result["settlement_state"] == "pending"
        llm_calls = first.process.get(created.root_pid).resource_usage.llm_calls
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        record = reopened.store.get_task_run(created.run_id)
        point = reopened.store.get_task_run_resume_point(created.root_pid)
        assert recovered.status is TaskRunStatus.CANCELLED
        assert record is not None and record.completed_step_count == 1
        assert point is not None and point.pending_action_payload_id is None
        assert (
            reopened.process.get(created.root_pid).resource_usage.llm_calls
            == llm_calls
        )

        replayed = reopened.task_runs.cancel(
            created.run_id,
            expected_revision=running.revision,
            command_id="cancel-with-staged-outcome",
        )
        completed = reopened.store.get_task_run_command(
            created.run_id,
            "cancel-with-staged-outcome",
        )
        assert replayed.status is TaskRunStatus.CANCELLED
        assert completed is not None
        assert completed.result["settlement_state"] == "complete"
    finally:
        reopened.close()


def test_task_run_rejects_extra_binding_during_activate_skill_settlement(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "uncertified-activate-skill-drift.sqlite",
        config=_config(),
    )
    try:
        skill_id = "agent-libos-workspace-navigation"
        package = get_builtin_skill_catalog().get(skill_id)
        assert package is not None
        action = {
            "action": "activate_skill",
            "skill_id": skill_id,
            "expected_package_sha256": package.package_sha256,
        }
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="reject an unrelated binding during activation",
                display_title="Reject extra binding",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-uncertified-binding-drift",
        )
        assert created.root_pid is not None
        call_id = "llmcall-uncertified-binding-drift"
        runtime.store.insert_llm_call(
            replace(
                _full_llm_call(call_id, created.root_pid, "BINDING_RESULT"),
                image_id="coding-agent:v0",
            )
        )
        generation = runtime.store.get_llm_context_generation(created.root_pid)
        manifest = validated_action_manifest(
            [action],
            call_id=call_id,
            parallel_tool_calls=False,
            host_auto_wait=False,
            tool_call_count=1,
            data_labels={},
        )
        runtime.task_runs.record_validated_transcript(
            pid=created.root_pid,
            call_id=call_id,
            action_manifest=manifest,
            context_generation=generation,
        )
        assert runtime.task_runs.pending_validated_action_for_pid(
            created.root_pid
        ) == manifest
        claimed = runtime.store.get_task_run_resume_point(created.root_pid)
        assert claimed is not None
        result = runtime.llm.actions.dispatch(created.root_pid, action)
        assert result["ok"] is True

        outcome = completed_outcome_manifest(
            state="completed",
            paired_outputs_persisted=True,
            data_labels={},
            result={"ok": True, "action": action, "result": result},
        )
        tampered_outcome = json.loads(json.dumps(outcome))
        tampered_activation = tampered_outcome["result"]["result"]["payload"][
            "result"
        ]
        tampered_activation["tool_ids"]["read_text_file"] = "tool_forged"
        with pytest.raises(
            ValidationError,
            match="does not match durable state",
        ):
            runtime.task_runs.stage_completed_transcript(
                pid=created.root_pid,
                call_id=call_id,
                outcome_manifest=tampered_outcome,
                context_generation=generation,
            )
        assert runtime.store.get_task_run_resume_point(created.root_pid) == claimed

        activated = runtime.process.get(created.root_pid)
        forged_model_table = dict(activated.model_tool_table)
        assert "get_current_time" not in forged_model_table
        forged_model_table["get_current_time"] = activated.tool_table[
            "get_current_time"
        ]
        runtime.store.patch_process(
            created.root_pid,
            {"model_tool_table": forged_model_table},
            expected_revision=activated.revision,
        )
        with pytest.raises(
            ValidationError,
            match="uncertified tool or Skill binding",
        ):
            runtime.task_runs.stage_completed_transcript(
                pid=created.root_pid,
                call_id=call_id,
                outcome_manifest=outcome,
                context_generation=generation,
            )

        unchanged = runtime.store.get_task_run_resume_point(created.root_pid)
        assert unchanged == claimed
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None and record.completed_step_count == 0
        assert all(
            item.status != "result_staged"
            for item in runtime.store.list_task_run_ledger(
                created.run_id,
                after=None,
                limit=100,
            ).records
        )
    finally:
        runtime.close()


def test_task_run_exact_tool_id_rejects_alias_rebind_before_dispatch(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "exact-tool-id-race.sqlite", config=_config())
    try:
        skill_id = "agent-libos-workspace-navigation"
        package = get_builtin_skill_catalog().get(skill_id)
        assert package is not None
        action = {
            "action": "activate_skill",
            "skill_id": skill_id,
            "expected_package_sha256": package.package_sha256,
        }
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="reject a tool alias rebind after durable claim",
                display_title="Exact tool identity",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-exact-tool-id-race",
        )
        root_pid = created.root_pid
        assert root_pid is not None
        call_id = "llmcall-exact-tool-id-race"
        runtime.store.insert_llm_call(
            replace(
                _full_llm_call(call_id, root_pid, "EXACT_TOOL_ID"),
                image_id="coding-agent:v0",
            )
        )
        generation = runtime.store.get_llm_context_generation(root_pid)
        manifest = validated_action_manifest(
            [action],
            call_id=call_id,
            parallel_tool_calls=False,
            host_auto_wait=False,
            tool_call_count=1,
            data_labels={},
        )
        runtime.task_runs.record_validated_transcript(
            pid=root_pid,
            call_id=call_id,
            action_manifest=manifest,
            context_generation=generation,
        )
        assert runtime.task_runs.pending_validated_action_for_pid(root_pid) == manifest
        expected_tool_id = runtime.task_runs.expected_tool_id_for_pending_action(
            root_pid,
            action,
        )
        before = runtime.process.get(root_pid)
        assert expected_tool_id == before.tool_table["activate_skill"]
        assert skill_id not in before.loaded_skills
        tool_calls_before = before.resource_usage.tool_calls

        rebound_table = dict(before.tool_table)
        rebound_model_table = dict(before.model_tool_table)
        rebound_tool_id = before.tool_table["discover_skills"]
        assert rebound_tool_id != expected_tool_id
        rebound_table["activate_skill"] = rebound_tool_id
        rebound_model_table["activate_skill"] = rebound_tool_id
        runtime.store.patch_process(
            root_pid,
            {
                "tool_table": rebound_table,
                "model_tool_table": rebound_model_table,
            },
            expected_revision=before.revision,
        )

        with pytest.raises(ValueError, match="binding changed before dispatch"):
            runtime.llm.actions.dispatch(
                root_pid,
                action,
                expected_tool_id=expected_tool_id,
            )
        after = runtime.process.get(root_pid)
        assert skill_id not in after.loaded_skills
        assert after.resource_usage.tool_calls == tool_calls_before
        with pytest.raises(ValidationError, match="binding changed"):
            runtime.task_runs.expected_tool_id_for_pending_action(root_pid, action)
        summary = runtime.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert "binding_drift" in {item["kind"] for item in summary.blockers}
    finally:
        runtime.close()


def test_task_run_rejects_parallel_activate_skill_before_any_dispatch(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "parallel-activate-skill-barrier.sqlite",
        config=_config(),
    )
    try:
        skill_id = "agent-libos-workspace-navigation"
        package = get_builtin_skill_catalog().get(skill_id)
        assert package is not None
        activation = {
            "action": "activate_skill",
            "skill_id": skill_id,
            "expected_package_sha256": package.package_sha256,
        }
        cases = (
            ([activation], True, 1),
            (
                [activation, {"action": "discover_skills", "query": "workspace"}],
                False,
                2,
            ),
        )
        for index, (actions, parallel_tool_calls, tool_call_count) in enumerate(cases):
            created = runtime.task_runs.create(
                TaskRunSpecV1(
                    goal="reject parallel binding mutations",
                    display_title="Binding batch barrier",
                    image_id="coding-agent:v0",
                    retention=TaskRunRetention.PERMANENT,
                ),
                client_request_id=f"create-binding-batch-barrier-{index}",
            )
            root_pid = created.root_pid
            assert root_pid is not None
            call_id = f"llmcall-binding-batch-barrier-{index}"
            runtime.store.insert_llm_call(
                replace(
                    _full_llm_call(call_id, root_pid, "BINDING_BATCH"),
                    image_id="coding-agent:v0",
                )
            )
            before = runtime.process.get(root_pid)
            manifest = validated_action_manifest(
                actions,
                call_id=call_id,
                parallel_tool_calls=parallel_tool_calls,
                host_auto_wait=False,
                tool_call_count=tool_call_count,
                data_labels={},
            )
            with pytest.raises(ValidationError, match="singleton non-parallel"):
                runtime.task_runs.record_validated_transcript(
                    pid=root_pid,
                    call_id=call_id,
                    action_manifest=manifest,
                    context_generation=(
                        runtime.store.get_llm_context_generation(root_pid)
                    ),
                )
            after = runtime.process.get(root_pid)
            assert skill_id not in after.loaded_skills
            assert after.resource_usage.tool_calls == before.resource_usage.tool_calls
            assert runtime.store.get_task_run_resume_point(root_pid) is None
            summary = runtime.task_runs.get(created.run_id)
            assert summary.status is TaskRunStatus.NEEDS_ATTENTION
            assert "binding_drift" in {item["kind"] for item in summary.blockers}
    finally:
        runtime.close()


def test_task_run_provider_time_binding_drift_commits_no_action_safe_point(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "provider-time-binding-drift.sqlite",
        config=_config(),
    )
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="reject a response selected against an obsolete tool binding",
                display_title="Provider-time binding gate",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-provider-time-binding-drift",
        )
        root_pid = created.root_pid
        assert root_pid is not None

        class BindingDriftClient(RecordingActionClient):
            def complete_action(
                self,
                messages: list[dict[str, object]],
                tools: list[dict[str, object]],
            ) -> object:
                process = runtime.process.get(root_pid)
                rebound = dict(process.tool_table)
                assert rebound["get_current_time"] != rebound["sleep"]
                rebound["get_current_time"] = rebound["sleep"]
                runtime.store.patch_process(
                    root_pid,
                    {"tool_table": rebound},
                    expected_revision=process.revision,
                )
                return super().complete_action(messages, tools)

        client = BindingDriftClient(
            [{"action": "discover_skills", "text": "workspace", "limit": 5}]
        )
        runtime.llm.client = client

        blocked = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run-provider-time-binding-drift",
            max_quanta=1,
        )

        assert len(client.user_prompts) == 1
        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert "binding_drift" in {item["kind"] for item in blocked.blockers}
        process = runtime.process.get(root_pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.tool_calls == 0
        assert runtime.store.get_task_run_resume_point(root_pid) is None
        latest_call = runtime.store.get_latest_llm_call(
            pid=root_pid,
            purpose="action_selection",
        )
        assert latest_call is not None and latest_call.status == "error"
        assert isinstance(
            latest_call.request_options.get("openai_response_scope_fingerprint"),
            str,
        )
        ledger = runtime.store.list_task_run_ledger(
            created.run_id,
            after=None,
            limit=100,
        ).records
        assert not any(
            item.kind.value == "llm_turn"
            and item.status in {"validated", "dispatching", "result_staged"}
            for item in ledger
        )
    finally:
        runtime.close()


def test_task_run_binding_drift_blocks_before_next_provider_dispatch(
    tmp_path: Path,
) -> None:
    class ProviderMustNotRun:
        def __init__(self) -> None:
            self.calls = 0

        def complete_action(self, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("Provider must not run after TaskRun binding drift")

    runtime = Runtime.open(
        tmp_path / "binding-drift-before-provider.sqlite",
        config=_config(),
    )
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="detect binding drift before another model turn",
                display_title="Provider preflight binding gate",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-provider-binding-gate",
        )
        assert created.root_pid is not None
        runtime.llm.client = RecordingActionClient(
            [
                {
                    "action": "discover_skills",
                    "text": "workspace navigation",
                    "limit": 5,
                }
            ]
        )
        running = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="establish-binding-safe-point",
            max_quanta=1,
        )
        assert running.status is TaskRunStatus.RUNNING
        process = runtime.process.get(created.root_pid)
        forged_model_table = dict(process.model_tool_table)
        forged_model_table["get_current_time"] = process.tool_table[
            "get_current_time"
        ]
        runtime.store.patch_process(
            created.root_pid,
            {"model_tool_table": forged_model_table},
            expected_revision=process.revision,
        )
        provider = ProviderMustNotRun()
        runtime.llm.client = provider

        blocked = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=running.revision,
            command_id="reject-drift-before-provider",
            max_quanta=1,
        )

        assert blocked.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in blocked.blockers} == {"binding_drift"}
        assert provider.calls == 0
        assert runtime.process.get(created.root_pid).resource_usage.llm_calls == 1
    finally:
        runtime.close()


def test_recovery_preflight_is_static_but_live_resume_binding_drift_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "static-preflight-dynamic-binding.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(runtime, retention=TaskRunRetention.PERMANENT)
        root_pid = created.root_pid
        assert root_pid is not None
        call_id = "binding-safe-point-call"
        runtime.store.insert_llm_call(
            _full_llm_call(call_id, root_pid, "BINDING_SAFE_POINT")
        )
        action_manifest = {
            "schema_version": 1,
            "call_id": call_id,
            "actions": [{"action": "process_exit", "payload": {"done": True}}],
            "parallel_tool_calls": False,
            "host_auto_wait": False,
            "tool_call_count": 1,
            "data_labels": {},
            "previous_response_id_used": False,
        }
        outcome_manifest = {
            "schema_version": 1,
            "state": "completed",
            "paired_outputs_persisted": True,
            "data_labels": {},
            "result": {
                "transcript_messages": [
                    {"role": "assistant", "content": "safe local result"}
                ]
            },
            "durable_wait": None,
            "previous_response_id_used": False,
        }
        runtime.task_runs.record_validated_transcript(
            pid=root_pid,
            call_id=call_id,
            action_manifest=action_manifest,
            context_generation="1",
        )
        claimed_action = runtime.task_runs.pending_validated_action_for_pid(root_pid)
        assert claimed_action == action_manifest
        runtime.task_runs.stage_completed_transcript(
            pid=root_pid,
            call_id=call_id,
            outcome_manifest=outcome_manifest,
            context_generation="1",
        )
        runtime.task_runs.record_completed_transcript(
            pid=root_pid,
            call_id=call_id,
            outcome_manifest=outcome_manifest,
            context_generation="1",
        )
        point = runtime.store.get_task_run_resume_point(root_pid, complete_only=True)
        assert point is not None and point.complete
        before = runtime.store.get_task_run(created.run_id)
        assert before is not None
        payloads_before = runtime.store.list_task_run_payloads(created.run_id)

        def live_binding_must_not_run(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("static payload preflight consulted live bindings")

        with monkeypatch.context() as static_patch:
            static_patch.setattr(
                runtime.task_runs,
                "_process_binding_hashes",
                live_binding_must_not_run,
            )
            runtime.task_runs.validate_recoverable_payloads()
        assert created.run_id not in runtime.task_runs._prevalidated_blockers
        assert runtime.store.get_task_run(created.run_id) == before
        assert runtime.store.list_task_run_payloads(created.run_id) == payloads_before

        latest = runtime.store.get_task_run(created.run_id)
        assert latest is not None
        now = utc_now()
        runtime.store.update_task_run_cas(
            created.run_id,
            latest.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": now,
                "updated_at": now,
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
    finally:
        runtime.close()

    original_binding = TaskRunManager._process_binding_hashes
    dynamic_checks = 0

    def changed_live_binding(
        manager: TaskRunManager,
        process: object,
    ) -> tuple[str, str, str]:
        nonlocal dynamic_checks
        dynamic_checks += 1
        image_hash, tool_hash, provider_hash = original_binding(manager, process)
        changed_tool_hash = "f" * 64 if tool_hash != "f" * 64 else "e" * 64
        return image_hash, changed_tool_hash, provider_hash

    monkeypatch.setattr(
        TaskRunManager,
        "_process_binding_hashes",
        changed_live_binding,
    )
    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        assert dynamic_checks > 0
        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        assert {item["kind"] for item in recovered.blockers} == {"binding_drift"}
        assert TaskRunAction.RUN not in recovered.allowed_actions
        assert TaskRunAction.RESUME not in recovered.allowed_actions
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        reopened.close()


def test_terminal_paused_root_never_advertises_undeliverable_controls(
    tmp_path: Path,
) -> None:
    database = tmp_path / "paused-terminal-root.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            request_id="create-paused-terminal-root",
            title="Paused terminal root",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        paused = runtime.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id="pause-before-host-exit",
        )
        assert paused.status is TaskRunStatus.PAUSED

        runtime.process.exit(root_pid, payload={"host_exit": True})
        terminal_root = runtime.task_runs.get(created.run_id)

        assert terminal_root.status is TaskRunStatus.NEEDS_ATTENTION
        assert terminal_root.revision > paused.revision
        assert terminal_root.allowed_actions == (
            TaskRunAction.RECOVER,
            TaskRunAction.CANCEL,
        )
        assert runtime.task_runs.get(created.run_id) == terminal_root
        snapshot = runtime.task_runs.list(
            statuses=(TaskRunStatus.NEEDS_ATTENTION,),
            limit=10,
        )
        assert snapshot.records == (terminal_root,)
        # Exact replay keeps the immutable historical PAUSED result, whose
        # revision is now stale; the current snapshot remains revision-bound.
        assert (
            runtime.task_runs.pause(
                created.run_id,
                expected_revision=created.revision,
                command_id="pause-before-host-exit",
            )
            == paused
        )
        with pytest.raises(
            ValidationError,
            match="root process no longer accepts follow-up requirements",
        ):
            runtime.task_runs.follow_up(
                created.run_id,
                "cannot reach this root",
                expected_revision=terminal_root.revision,
                command_id="follow-up-after-paused-root-exit",
            )
        assert runtime.task_runs.get(created.run_id) == terminal_root
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)

        assert recovered.status is TaskRunStatus.NEEDS_ATTENTION
        assert recovered.allowed_actions == (
            TaskRunAction.RECOVER,
            TaskRunAction.CANCEL,
        )
        assert reopened.process.get(root_pid).status is ProcessStatus.EXITED
        cancelled = reopened.task_runs.cancel(
            created.run_id,
            expected_revision=recovered.revision,
            command_id="cancel-paused-terminal-root",
        )
        assert cancelled.status is TaskRunStatus.CANCELLED
        assert cancelled.allowed_actions == (TaskRunAction.RERUN,)
    finally:
        reopened.close()


def test_paused_message_waiter_reopens_without_losing_its_physical_wait(
    tmp_path: Path,
) -> None:
    database = tmp_path / "paused-message-waiter.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            request_id="create-paused-message-waiter",
            title="Paused durable waiter",
        )
        assert created.root_pid is not None
        runnable = runtime.process.get(created.root_pid)
        waiting_process = runtime.process_transitions.transition(
            created.root_pid,
            ProcessStatus.WAITING_EVENT,
            expected_revision=runnable.revision,
            expected_status=ProcessStatus.RUNNABLE,
            expected_state_generation=runnable.state_generation,
            wait_state=MessageProcessWait(filters={"channel": "durable-control"}),
        )
        assert waiting_process.status is ProcessStatus.WAITING_EVENT

        paused = runtime.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id="pause-message-waiter",
        )
        assert paused.status is TaskRunStatus.PAUSED
        process_after_pause = runtime.process.get(created.root_pid)
        assert process_after_pause.status is ProcessStatus.WAITING_EVENT
        assert type(process_after_pause.wait_state).__name__ == "MessageProcessWait"
        llm_calls_before_reopen = process_after_pause.resource_usage.llm_calls
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        process = reopened.process.get(created.root_pid)
        assert recovered.status is TaskRunStatus.PAUSED
        assert "manual_recovery_required" not in {
            item["kind"] for item in recovered.blockers
        }
        assert TaskRunAction.RESUME in recovered.allowed_actions
        assert process.status is ProcessStatus.WAITING_EVENT
        assert type(process.wait_state).__name__ == "MessageProcessWait"
        assert process.resource_usage.llm_calls == llm_calls_before_reopen
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()
