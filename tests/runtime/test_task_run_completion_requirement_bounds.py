from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    TaskRunRequirementKind,
    TaskRunRequirementStatus,
    TaskRunRetention,
    TaskRunStatus,
)
from agent_libos.models.exceptions import ValidationError


def _config(*, recovery_hard_limit: int = 1_000):
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
            recovery_page_size=min(100, recovery_hard_limit),
            recovery_page_hard_limit=recovery_hard_limit,
        ),
    )


def _create(runtime: Runtime, title: str):
    return runtime.task_runs.create(
        TaskRunSpecV1(
            goal={"objective": title},
            display_title=title,
            image_id="base-agent:v0",
            retention=TaskRunRetention.PERMANENT,
        ),
        client_request_id=f"create:{title}",
    )


def _completion(call_id: str, action: dict[str, Any]) -> LLMCompletion:
    selected = dict(action)
    name = str(selected.pop("action"))
    return LLMCompletion(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": name,
                "arguments": json.dumps(selected, sort_keys=True),
            }
        ],
    )


class _ExitClient:
    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        return _completion(
            "bounded-requirement-exit",
            {"action": "process_exit", "payload": {"summary": "complete"}},
        )


class _DiscoverClient:
    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        return _completion(
            "bounded-requirement-discover",
            {
                "action": "discover_skills",
                "text": "local Skill catalog",
                "limit": 5,
            },
        )


def test_completion_requirement_snapshot_uses_hard_limit_lookahead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "completion-requirement-lookahead.sqlite",
        config=_config(recovery_hard_limit=2),
    )
    try:
        created = _create(runtime, "completion-requirement-lookahead")
        initial = runtime.store.list_task_run_requirements(created.run_id)[0]
        for ordinal in (1, 2):
            runtime.store.insert_task_run_requirement(
                replace(
                    initial,
                    requirement_id=f"req-corrupt-lookahead-{ordinal}",
                    ordinal=ordinal,
                    kind=TaskRunRequirementKind.FOLLOW_UP,
                    label=f"corrupt lookahead {ordinal}",
                )
            )
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None
        record_at_cap = replace(record, requirement_count=2)

        observed_limits: list[int | None] = []
        original = runtime.store.list_task_run_requirements

        def traced_list(run_id: str, **kwargs: Any):
            observed_limits.append(kwargs.get("limit"))
            return original(run_id, **kwargs)

        monkeypatch.setattr(runtime.store, "list_task_run_requirements", traced_list)
        with pytest.raises(ValidationError, match="projection is inconsistent"):
            runtime.task_runs._bounded_completion_requirements(record_at_cap)

        assert observed_limits == [3]
    finally:
        runtime.close()


def test_prompt_requirement_binding_validation_uses_bounded_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "prompt-requirement-binding-bound.sqlite",
        config=_config(recovery_hard_limit=2),
    )
    try:
        created = _create(runtime, "prompt-requirement-binding-bound")
        record = runtime.store.get_task_run(created.run_id)
        requirements = runtime.store.list_task_run_requirements(created.run_id)
        assert record is not None and len(requirements) == 1
        requirement = requirements[0]
        binding = {
            "schema_version": 1,
            "run_id": created.run_id,
            "pid": created.root_pid,
            "context_generation": "binding-generation",
            "requirements": [
                {
                    "requirement_id": requirement.requirement_id,
                    "ordinal": requirement.ordinal,
                    "requirement_sha256": requirement.requirement_sha256,
                }
            ],
        }

        observed_limits: list[int | None] = []
        original = runtime.store.list_task_run_requirements

        def traced_list(run_id: str, **kwargs: Any):
            observed_limits.append(kwargs.get("limit"))
            return original(run_id, **kwargs)

        monkeypatch.setattr(runtime.store, "list_task_run_requirements", traced_list)
        validated = runtime.task_runs._validated_prompt_requirement_binding(
            record,
            pid=str(created.root_pid),
            context_generation="binding-generation",
            value=binding,
            allowed_statuses={TaskRunRequirementStatus.PENDING},
        )

        assert validated == binding
        assert observed_limits == [3]
    finally:
        runtime.close()


def test_prompt_requirement_snapshot_refreshes_a_legitimately_stale_run(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "prompt-requirement-stale-run.sqlite",
        config=_config(recovery_hard_limit=2),
    )
    try:
        created = _create(runtime, "prompt-requirement-stale-run")
        stale = runtime.store.get_task_run(created.run_id)
        assert stale is not None and created.root_pid is not None
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Include the concurrently appended requirement.",
            expected_revision=created.revision,
            command_id="follow-up:prompt-requirement-stale-run",
        )

        visible = runtime.task_runs._requirements_visible_to_prompt(
            stale,
            pid=created.root_pid,
            context_generation="stale-run-generation",
        )

        assert followed.requirement_count == 2
        assert [item.ordinal for item in visible] == [0, 1]
    finally:
        runtime.close()


def test_frozen_llm_binding_validates_against_current_requirement_snapshot(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "llm-binding-stale-run.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime, "llm-binding-stale-run")
        assert created.root_pid is not None
        runtime.llm.client = _DiscoverClient()
        observed = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:llm-binding-stale-run",
            max_quanta=1,
        )
        stale = runtime.store.get_task_run(created.run_id)
        calls = runtime.store.list_llm_calls(pid=created.root_pid, limit=10)
        assert stale is not None and len(calls) == 1
        raw = calls[0].request_options["task_run_requirement_binding_v1"]
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "This later requirement was not in the frozen LLM prompt.",
            expected_revision=observed.revision,
            command_id="follow-up:llm-binding-stale-run",
        )
        process = runtime.store.get_process(created.root_pid)
        assert process is not None

        binding = runtime.task_runs._llm_prompt_requirement_binding(
            process,
            stale,
            call_id=calls[0].call_id,
            context_generation=str(raw["context_generation"]),
        )

        assert binding is not None
        assert len(binding["requirements"]) == 1
        assert followed.requirement_count == 2
    finally:
        runtime.close()


def test_recoverable_resume_point_prevalidation_uses_hard_limit_lookahead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "resume-point-recovery-lookahead.sqlite",
        config=_config(recovery_hard_limit=2),
    )
    try:
        created = _create(runtime, "resume-point-recovery-lookahead")
        assert len(
            runtime.store.list_task_run_resume_points(
                created.run_id,
                limit=3,
            )
        ) <= 2
        with pytest.raises(ValidationError, match="hard cap"):
            runtime.store.list_task_run_resume_points(
                created.run_id,
                limit=4,
            )
        observed_limits: list[int | None] = []

        def oversized_page(
            run_id: str,
            *,
            complete_only: bool = True,
            limit: int | None = None,
        ) -> list[object]:
            assert run_id == created.run_id
            assert complete_only is True
            observed_limits.append(limit)
            return [object(), object(), object()]

        monkeypatch.setattr(
            runtime.store,
            "list_task_run_resume_points",
            oversized_page,
        )
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None
        with pytest.raises(ValidationError, match="resume-point recovery exceeds"):
            runtime.task_runs._prevalidate_recoverable_resume_points(record)

        assert observed_limits == [3]
    finally:
        runtime.close()


def test_goal_payload_lookup_never_falls_back_to_an_unbounded_payload_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "authoritative-goal-payload.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime, "authoritative-goal-payload")

        def reject_payload_scan(_run_id: str) -> list[object]:
            raise AssertionError("authoritative goal lookup scanned all Run payloads")

        monkeypatch.setattr(
            runtime.store,
            "list_task_run_payloads",
            reject_payload_scan,
        )
        goal = runtime.task_runs._payload_by_role(created.run_id, "goal")

        assert goal.role == "goal"
        assert runtime.task_runs._payloads_retained(created.run_id) is True
        with pytest.raises(ValidationError, match="authoritative goal"):
            runtime.task_runs._payload_by_role(created.run_id, "transcript")
    finally:
        runtime.close()


def test_completion_settlement_reuses_one_bounded_requirement_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        tmp_path / "completion-requirement-single-snapshot.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime, "completion-requirement-single-snapshot")
        inside_settlement = False
        observed_limits: list[int | None] = []
        original_commit = runtime.task_runs._commit_completed_transcript_once
        original_list = runtime.store.list_task_run_requirements

        def traced_commit(**kwargs: Any) -> None:
            nonlocal inside_settlement
            inside_settlement = True
            try:
                original_commit(**kwargs)
            finally:
                inside_settlement = False

        def traced_list(run_id: str, **kwargs: Any):
            if inside_settlement:
                observed_limits.append(kwargs.get("limit"))
            return original_list(run_id, **kwargs)

        monkeypatch.setattr(
            runtime.task_runs,
            "_commit_completed_transcript_once",
            traced_commit,
        )
        monkeypatch.setattr(runtime.store, "list_task_run_requirements", traced_list)
        runtime.llm.client = _ExitClient()

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:completion-requirement-single-snapshot",
            max_quanta=1,
        )

        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert terminal.requirement_count == terminal.satisfied_requirement_count == 1
        assert observed_limits == [1_001]
    finally:
        runtime.close()
