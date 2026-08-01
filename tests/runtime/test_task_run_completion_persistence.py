from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    ProcessStatus,
    TaskRunLedgerKind,
    TaskRunRequirementStatus,
    TaskRunRetention,
    TaskRunStatus,
)


def _config():
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
            recovery_page_size=100,
            recovery_page_hard_limit=1_000,
        ),
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


def _find_completion_review(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            return _find_completion_review(json.loads(value))
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        for item in value:
            found = _find_completion_review(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    review = value.get("completion_review")
    if isinstance(review, dict) and isinstance(review.get("review_token"), str):
        return review
    for item in value.values():
        found = _find_completion_review(item)
        if found is not None:
            return found
    return None


class _RequestReviewClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        return _completion(
            "request-persisted-review",
            {"action": "process_exit", "payload": {"summary": "verified"}},
        )


class _DiscoverSkillGuidanceClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        return _completion(
            "discover-persisted-review-evidence",
            {
                "action": "discover_skills",
                "text": "available skill guidance",
                "limit": 5,
            },
        )


class _CompletePersistedReviewClient:
    def __init__(self, expected_token: str) -> None:
        self.expected_token = expected_token
        self.calls = 0

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        review = _find_completion_review(messages)
        assert review is not None
        assert review["review_token"] == self.expected_token
        source_refs = review["completion_source_refs"]
        assert isinstance(source_refs, list) and source_refs
        return _completion(
            "complete-persisted-review",
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": {
                    "goal_oid": review["goal"]["oid"],
                    "reviewed_message_ids": review[
                        "acknowledged_human_message_ids"
                    ],
                    "acceptance_checks": [
                        {
                            "requirement": "inspect available skill guidance",
                            "source_refs": source_refs,
                            "status": "completed",
                            "evidence_tool_calls": ["discover_skills"],
                            "evidence_summary": (
                                "The successful discover_skills result verifies "
                                "the requested inspection."
                            ),
                        }
                    ],
                    "final_verification": ["discover_skills"],
                },
                "payload": {"summary": "verified after reopen"},
            },
        )


def test_task_run_completion_review_token_survives_reopen_without_redispatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "task-run-completion-review-reopen.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = first.task_runs.create(
            TaskRunSpecV1(
                goal=(
                    "Inspect available skill guidance with discover_skills and "
                    "finish only after that result is verified."
                ),
                display_title="Persist completion review",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create:persist-completion-review",
        )
        root_pid = created.root_pid
        assert root_pid is not None
        discover_client = _DiscoverSkillGuidanceClient()
        first.llm.client = discover_client
        inspected = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:persisted-review-evidence",
            max_quanta=1,
        )
        assert inspected.status is TaskRunStatus.RUNNING
        assert discover_client.calls == 1

        review_client = _RequestReviewClient()
        first.llm.client = review_client
        review_pending = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=inspected.revision,
            command_id="run:request-persisted-review",
            max_quanta=1,
        )

        assert review_pending.status is TaskRunStatus.RUNNING
        assert review_client.calls == 1
        point = first.store.get_task_run_resume_point(root_pid, complete_only=True)
        assert point is not None
        assert point.pending_action_payload_id is None
        transcript = first.store.get_task_run_payload(point.transcript_payload_id)
        assert transcript is not None and transcript.canonical_json is not None
        review = _find_completion_review(transcript.canonical_json)
        assert review is not None
        review_token = review["review_token"]
        ledger_before_reopen = first.task_runs.list_ledger(
            created.run_id,
            limit=100,
        ).records
    finally:
        first.close()

    reopened = Runtime.open(database, config=_config())
    try:
        ledger_after_reopen = reopened.task_runs.list_ledger(
            created.run_id,
            limit=100,
        ).records
        assert ledger_after_reopen == ledger_before_reopen

        completion_client = _CompletePersistedReviewClient(review_token)
        reopened.llm.client = completion_client
        current = reopened.task_runs.get(created.run_id)
        terminal = reopened.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=current.revision,
            command_id="run:complete-persisted-review",
            max_quanta=1,
        )

        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert terminal.step_count == terminal.completed_step_count == 3
        assert completion_client.calls == 1
        assert reopened.process.get(root_pid).status is ProcessStatus.EXITED
        assert reopened.store.get_llm_pending_action(root_pid) is None
        requirements = reopened.store.list_task_run_requirements(created.run_id)
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.SATISFIED
        ]

        final_ledger = reopened.task_runs.list_ledger(
            created.run_id,
            limit=100,
        ).records
        assert final_ledger[: len(ledger_before_reopen)] == ledger_before_reopen
        assert len({item.item_id for item in final_ledger}) == len(final_ledger)
        assert [item.seq for item in final_ledger] == list(
            range(1, len(final_ledger) + 1)
        )
        completed_turns = [
            item
            for item in final_ledger
            if item.kind is TaskRunLedgerKind.LLM_TURN
            and item.label == "LLM action and paired outputs persisted"
        ]
        assert len(completed_turns) == 3
    finally:
        reopened.close()
