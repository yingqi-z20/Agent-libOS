from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import agent_libos

from agent_libos.models import (
    TASK_RUN_TERMINAL_STATUSES,
    TaskRunAction,
    TaskRunPage,
    TaskRunPayload,
    TaskRunPayloadRetention,
    TaskRunRecord,
    TaskRunResumePoint,
    TaskRunRetention,
    TaskRunSpecV1,
    TaskRunStatus,
    TaskRunSummary,
    canonical_task_run_json,
    task_run_payload_sha256,
)


def _spec(**overrides: object) -> TaskRunSpecV1:
    values: dict[str, object] = {
        "schema_version": 1,
        "goal": {"ticket": "T-42", "steps": [1, 2]},
        "display_title": "Repair T-42",
        "image_id": "coding-agent:v0",
        "launch_options": {"working_directory": "repo"},
        "authority_manifest_id": "authority-1",
        "deadline_at": "2030-01-02T03:04:05+00:00",
        "retention": "purge_on_terminal",
    }
    values.update(overrides)
    return TaskRunSpecV1(**values)


def test_task_run_status_contract_is_exact_and_terminal_subset_is_closed() -> None:
    assert {status.value for status in TaskRunStatus} == {
        "queued",
        "running",
        "waiting_human",
        "waiting_process",
        "waiting_message",
        "waiting_tool",
        "paused",
        "cancelling",
        "finalizing",
        "needs_attention",
        "succeeded",
        "failed",
        "cancelled",
    }
    assert TASK_RUN_TERMINAL_STATUSES == {
        TaskRunStatus.SUCCEEDED,
        TaskRunStatus.FAILED,
        TaskRunStatus.CANCELLED,
    }
    assert TaskRunStatus.NEEDS_ATTENTION not in TASK_RUN_TERMINAL_STATUSES


def test_versioned_task_run_public_types_are_top_level_sdk_exports() -> None:
    for name in (
        "TaskRunSpecV1",
        "TaskRunStatus",
        "TaskRunSummary",
        "TaskRunLedgerItem",
        "TaskRunAction",
        "TaskRunRetention",
    ):
        assert name in agent_libos.__all__
        assert getattr(agent_libos, name) is not None


def test_task_run_spec_has_one_canonical_versioned_serialization() -> None:
    source_goal = {"z": [1, {"b": True}], "a": "value"}
    source_launch = {"environment": {"MODE": "test"}}
    spec = _spec(goal=source_goal, launch_options=source_launch)
    source_goal["z"].append("mutated")
    source_launch["environment"]["MODE"] = "mutated"

    mapping = spec.to_mapping()
    assert list(mapping) == [
        "schema_version",
        "goal",
        "display_title",
        "image_id",
        "launch_options",
        "authority_manifest_id",
        "deadline_at",
        "retention",
    ]
    assert "objective" not in mapping
    assert "title" not in mapping
    assert mapping["goal"] == {"z": [1, {"b": True}], "a": "value"}
    assert mapping["launch_options"] == {"environment": {"MODE": "test"}}
    assert spec.canonical_json() == canonical_task_run_json(mapping)
    assert spec.canonical_json().startswith('{"authority_manifest_id"')


def test_optional_spec_image_must_be_resolved_before_persisted_record() -> None:
    spec = _spec(image_id=None)

    assert spec.image_id is None
    with pytest.raises(ValueError, match="resolved image_id"):
        TaskRunRecord.from_spec("run-unresolved", spec)

    record = TaskRunRecord.from_spec(
        "run-resolved",
        spec,
        image_id="base-agent:v0",
    )
    assert record.image_id == "base-agent:v0"


def test_task_run_spec_accepts_input_aliases_but_never_emits_them() -> None:
    spec = TaskRunSpecV1.from_mapping(
        {
            "schema_version": 1,
            "objective": "finish the work",
            "title": "Long task",
            "image_id": "base-agent:v0",
            "retention": "permanent",
        }
    )

    assert spec.goal == "finish the work"
    assert spec.display_title == "Long task"
    assert spec.retention is TaskRunRetention.PERMANENT
    assert set(spec.to_mapping()) == {
        "schema_version",
        "goal",
        "display_title",
        "image_id",
        "launch_options",
        "authority_manifest_id",
        "deadline_at",
        "retention",
    }


@pytest.mark.parametrize(
    "values",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"deadline_at": "2030-01-01T00:00:00"},
        {"deadline_at": "not-a-date"},
        {"goal": {"number": float("nan")}},
        {"goal": {1: "non-string key"}},
        {"launch_options": []},
    ],
)
def test_task_run_spec_rejects_ambiguous_or_noncanonical_input(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _spec(**values)


def test_task_run_spec_rejects_cycles_and_unknown_alias_residue() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        _spec(goal=cycle)
    with pytest.raises(ValueError, match="unknown TaskRun spec fields"):
        TaskRunSpecV1.from_mapping(
            {
                **_spec().to_mapping(),
                "provider_api_key": "must-not-be-accepted",
            }
        )


def test_task_run_spec_normalizes_equivalent_deadlines_to_canonical_utc() -> None:
    east = _spec(deadline_at="2030-01-02T11:04:05+08:00")
    utc = _spec(deadline_at="2030-01-02T03:04:05Z")

    assert east.deadline_at == "2030-01-02T03:04:05.000000+00:00"
    assert east.canonical_json() == utc.canonical_json()


def test_plaintext_payload_is_integrity_bound_and_hash_only_has_no_content() -> None:
    payload = TaskRunPayload.plaintext(
        payload_id="payload-1",
        run_id="run-1",
        role="goal",
        label="initial requirement",
        value={"b": 2, "a": 1},
        created_at="2030-01-01T00:00:00+00:00",
    )

    assert payload.canonical_json == '{"a":1,"b":2}'
    assert payload.size_bytes == len(payload.canonical_json.encode("utf-8"))
    assert payload.sha256 == hashlib.sha256(
        payload.canonical_json.encode("utf-8")
    ).hexdigest()
    purged = replace(
        payload,
        canonical_json=None,
        retention_state=TaskRunPayloadRetention.HASH_ONLY,
        purged_at="2030-01-02T00:00:00+00:00",
    )
    assert purged.canonical_json is None
    assert purged.sha256 == payload.sha256


def test_plaintext_payload_rejects_noncanonical_encoding_even_with_matching_hash() -> None:
    noncanonical = '{"b": 2, "a": 1}'
    with pytest.raises(ValueError, match="canonical"):
        TaskRunPayload(
            payload_id="payload-1",
            run_id="run-1",
            role="goal",
            label="initial requirement",
            canonical_json=noncanonical,
            sha256=task_run_payload_sha256(noncanonical),
            size_bytes=len(noncanonical.encode("utf-8")),
            retention_state=TaskRunPayloadRetention.PLAINTEXT,
            created_at="2030-01-01T00:00:00+00:00",
            updated_at="2030-01-01T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"sha256": "0" * 64},
        {"size_bytes": 999},
        {"retention_state": "hash_only", "canonical_json": "{}", "purged_at": None},
        {"retention_state": "hash_only", "canonical_json": None, "purged_at": None},
    ],
)
def test_task_run_payload_rejects_forged_retention_projection(
    changes: dict[str, object],
) -> None:
    payload = TaskRunPayload.plaintext(
        payload_id="payload-1",
        run_id="run-1",
        role="goal",
        label="initial requirement",
        value={"secret": "integrity only"},
        created_at="2030-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError):
        replace(payload, **changes)


def test_task_run_summary_actions_are_typed_unique_and_fail_closed() -> None:
    summary = TaskRunSummary(
        run_id="run-1",
        revision=3,
        status=TaskRunStatus.PAUSED,
        display_title="Paused run",
        allowed_actions=("resume", "cancel"),
    )
    assert summary.allowed_actions == (
        TaskRunAction.RESUME,
        TaskRunAction.CANCEL,
    )
    with pytest.raises(ValueError):
        replace(summary, allowed_actions=("resume", "resume"))
    with pytest.raises(ValueError):
        replace(summary, allowed_actions=("retry_unknown_effect",))
    with pytest.raises(ValueError, match="payloads_purged"):
        replace(summary, payloads_purged=1)


@pytest.mark.parametrize(
    ("step_count", "completed_step_count", "requirement_count", "satisfied_count"),
    [(0, 1, 0, 0), (0, 0, 0, 1)],
)
def test_task_run_summary_rejects_impossible_progress_counts(
    step_count: int,
    completed_step_count: int,
    requirement_count: int,
    satisfied_count: int,
) -> None:
    with pytest.raises(ValueError):
        TaskRunSummary(
            run_id="run-progress",
            revision=1,
            status=TaskRunStatus.RUNNING,
            display_title="Progress",
            step_count=step_count,
            completed_step_count=completed_step_count,
            requirement_count=requirement_count,
            satisfied_requirement_count=satisfied_count,
        )


def test_task_run_resume_point_requires_positive_epoch() -> None:
    values = {
        "run_id": "run-1",
        "pid": "pid-1",
        "task_run_epoch": 1,
        "process_revision": 0,
        "context_generation": "generation-1",
        "safe_point_seq": 0,
        "binding_hash": "binding",
        "image_binding_hash": "image",
        "tool_binding_hash": "tools",
        "provider_binding_hash": "provider",
        "transcript_payload_id": "payload-1",
        "integrity_sha256": "0" * 64,
        "created_at": "2030-01-01T00:00:00+00:00",
        "updated_at": "2030-01-01T00:00:00+00:00",
    }
    assert TaskRunResumePoint(**values).task_run_epoch == 1
    with pytest.raises(ValueError, match="must be positive"):
        TaskRunResumePoint(**{**values, "task_run_epoch": 0})


def test_task_run_page_is_explicitly_versioned() -> None:
    assert TaskRunPage(records=()).schema_version == 1
    with pytest.raises(ValueError, match="schema_version"):
        TaskRunPage(records=(), schema_version=2)
