from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.external_effects import record_external_effect
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ProcessMessageKind,
    ProcessStatus,
    TaskRunAction,
    TaskRunLedgerKind,
    TaskRunPayloadRetention,
    TaskRunRequirementStatus,
    TaskRunRetention,
    TaskRunStatus,
)
from agent_libos.models.exceptions import (
    HumanResponseRequired,
    TaskRunRevisionConflict,
    ValidationError,
)
from agent_libos.process_execution import current_process_execution_token
from agent_libos.runtime.syscalls import LibOSSyscallSession


CHILD_PROCESS_SKILL = "agent-libos-child-processes"
HUMAN_COLLABORATION_SKILL = "agent-libos-human-collaboration"
REAL_DEADLINE_TEST_WINDOW_S = 5.0
THREAD_SYNC_TIMEOUT_S = 30.0
BATCH_CONTROL_SYNC_TIMEOUT_S = 90.0 if os.name == "nt" else THREAD_SYNC_TIMEOUT_S


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


def _v2_config():
    return replace(
        _config(),
        llm=replace(
            DEFAULT_CONFIG.llm,
            prompt_layout="cache_optimized_v2",
        ),
    )


def _spec(
    title: str,
    *,
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL,
    authority_manifest_id: str | None = None,
) -> TaskRunSpecV1:
    return TaskRunSpecV1(
        goal={"objective": title},
        display_title=title,
        image_id="base-agent:v0",
        authority_manifest_id=authority_manifest_id,
        retention=retention,
    )


def _create(
    runtime: Runtime,
    title: str,
    *,
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL,
    authority_manifest_id: str | None = None,
):
    return runtime.task_runs.create(
        _spec(
            title,
            retention=retention,
            authority_manifest_id=authority_manifest_id,
        ),
        client_request_id=f"create:{title}",
    )


def _create_coding_completion_run(runtime: Runtime, title: str):
    return runtime.task_runs.create(
        TaskRunSpecV1(
            goal={"objective": "Inspect the local Skill catalog and finish."},
            display_title=title,
            image_id="coding-agent:v0",
            retention=TaskRunRetention.PERMANENT,
        ),
        client_request_id=f"create:{title}",
    )


def _run_discover_quantum(runtime: Runtime, summary: Any, command_id: str):
    runtime.llm.client = _PlannedClient(
        [
            {
                "action": "discover_skills",
                "text": "local Skill catalog",
                "limit": 5,
            }
        ]
    )
    return runtime.task_runs.run_until_blocked(
        summary.run_id,
        expected_revision=summary.revision,
        command_id=command_id,
        max_quanta=1,
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


class _PlannedClient:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = [dict(action) for action in actions]
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        if not self._actions:
            raise AssertionError("no planned LLM action remains")
        return _completion(f"planned-{self.calls}", self._actions.pop(0))


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


class _UnresolvedCompletionClient:
    def __init__(self, reported_status: str) -> None:
        self.reported_status = reported_status
        self.calls = 0

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        if self.calls == 1:
            return _completion(
                "unresolved-1",
                {
                    "action": "discover_skills",
                    "text": "completion review",
                    "limit": 5,
                },
            )
        if self.calls == 2:
            return _completion(
                "unresolved-2",
                {"action": "process_exit", "payload": {"summary": "not done"}},
            )
        review = _find_completion_review(messages)
        assert review is not None
        requirements = review.get("requirements")
        assert isinstance(requirements, list) and len(requirements) == 1
        return _completion(
            "unresolved-3",
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": {
                    "acceptance_checks": [
                        {
                            "status": self.reported_status,
                            "evidence_tool_calls": [],
                            "evidence_summary": "The model cannot prove completion.",
                        }
                    ],
                    "final_verification": ["discover_skills"],
                },
                "payload": {"summary": "unresolved"},
            },
        )


class _StructuredTaskRunCompletionClient:
    def __init__(
        self,
        *,
        bundle_requirements: bool,
        stringify_completion_evidence: bool = False,
        malformed_completion_evidence: bool = False,
    ) -> None:
        self.bundle_requirements = bundle_requirements
        self.stringify_completion_evidence = stringify_completion_evidence
        self.malformed_completion_evidence = malformed_completion_evidence
        self.calls = 0
        self.review: dict[str, Any] | None = None

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        if self.calls == 1:
            return _completion(
                "structured-exit-review",
                {"action": "process_exit", "payload": {"summary": "review"}},
            )
        review = _find_completion_review(messages)
        assert review is not None
        self.review = review
        requirements = review.get("requirements")
        assert isinstance(requirements, list) and len(requirements) == 2
        if self.bundle_requirements:
            checks = [
                {
                    "status": "completed",
                    "evidence_tool_calls": ["discover_skills"],
                    "evidence_summary": "A generic catalog read was observed.",
                }
            ]
        else:
            checks = [
                {
                    "status": "completed",
                    "evidence_tool_calls": ["discover_skills"],
                    "evidence_summary": "The cited catalog read is the proof.",
                }
                for _requirement in requirements
            ]
        completion_evidence: Any = {
            "acceptance_checks": checks,
            "final_verification": ["discover_skills"],
        }
        if self.malformed_completion_evidence:
            completion_evidence = '{"acceptance_checks":'
        elif self.stringify_completion_evidence:
            completion_evidence = json.dumps(
                completion_evidence,
                sort_keys=True,
            )
        return _completion(
            "structured-exit-claim",
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": completion_evidence,
                "payload": {"summary": "structured claim"},
            },
        )


class _ExplodingClient:
    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        raise AssertionError("durable wait resume must not call the LLM provider")


class _CountingExplodingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        raise AssertionError("old TaskRun generation reached the LLM provider")


class _BlockingExitClient:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.messages: list[dict[str, Any]] = []

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        self.messages = [dict(message) for message in _messages]
        self.entered.set()
        if not self.release.wait(timeout=THREAD_SYNC_TIMEOUT_S):
            raise AssertionError("test did not release the blocked LLM provider")
        return _completion(
            f"blocked-{self.calls}",
            {"action": "process_exit", "payload": {"seen": "old-context"}},
        )


class _BlockingDiscoverClient:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=THREAD_SYNC_TIMEOUT_S):
            raise AssertionError("test did not release the blocked LLM provider")
        return _completion(
            f"blocked-discover-{self.calls}",
            {
                "action": "discover_skills",
                "text": "local Skill catalog",
                "limit": 5,
            },
        )


class _TwoToolClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "batch-first",
                    "name": "discover_skills",
                    "arguments": json.dumps({"text": "messages", "limit": 2}),
                },
                {
                    "id": "batch-second",
                    "name": "discover_skills",
                    "arguments": json.dumps({"text": "workspace", "limit": 2}),
                },
            ],
        )


class _FanOutJoinClient:
    """Route deterministic actions by the real process execution token."""

    def __init__(self, runtime: Runtime, root_pid: str) -> None:
        self.runtime = runtime
        self.root_pid = root_pid
        self.calls_by_pid: dict[str, int] = defaultdict(int)

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        token = current_process_execution_token()
        assert token is not None
        pid = token.pid
        call = self.calls_by_pid[pid]
        self.calls_by_pid[pid] += 1
        if pid != self.root_pid:
            assert call == 0
            return _completion(
                f"child-exit-{pid}",
                {"action": "process_exit", "payload": {"child": pid}},
            )

        children = sorted(
            (
                process
                for process in self.runtime.process.list()
                if process.parent_pid == self.root_pid
            ),
            key=lambda process: (process.created_at, process.pid),
        )
        if call == 0:
            action = {"action": "spawn_child_process", "goal": "fanout-child-a"}
        elif call == 1:
            action = {"action": "spawn_child_process", "goal": "fanout-child-b"}
        elif call in {2, 3}:
            assert len(children) == 2
            action = {
                "action": "wait_child_process",
                "child_pid": children[call - 2].pid,
            }
        else:
            assert call == 4
            action = {
                "action": "process_exit",
                "payload": {"joined_children": [child.pid for child in children]},
            }
        return _completion(f"root-{call}", action)


def _activate_child_tools(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    runtime.skills.activate_skill(pid, CHILD_PROCESS_SKILL, actor=pid)


def test_task_run_child_fanout_join_inherits_binding_and_converges(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "fanout.sqlite", config=_config())
    try:
        created = _create(runtime, "fanout-join")
        root_pid = created.root_pid
        assert root_pid is not None
        _activate_child_tools(runtime, root_pid)
        client = _FanOutJoinClient(runtime, root_pid)
        runtime.llm.client = client

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:fanout-join",
            max_quanta=20,
        )

        processes = runtime.store.list_processes_for_task_run(created.run_id)
        children = [process for process in processes if process.parent_pid == root_pid]
        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert len(children) == 2
        assert {process.status for process in processes} == {ProcessStatus.EXITED}
        assert all(process.task_run_id == created.run_id for process in processes)
        assert all(
            process.task_run_epoch == runtime.task_runs.runtime_epoch
            for process in processes
        )
        assert client.calls_by_pid[root_pid] == 5
        assert all(client.calls_by_pid[child.pid] == 1 for child in children)
        assert terminal.completed_step_count == terminal.step_count
        assert terminal.satisfied_requirement_count == terminal.requirement_count
    finally:
        runtime.close()


def test_integrity_bound_root_exit_satisfies_prompt_requirements(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "requirement-evidence.sqlite", config=_config())
    try:
        created = _create(
            runtime,
            "requirement-evidence",
            retention=TaskRunRetention.PERMANENT,
        )
        runtime.llm.client = _PlannedClient(
            [{"action": "process_exit", "payload": {"answer": "complete"}}]
        )

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:requirement-evidence",
            max_quanta=1,
        )

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        evidence_items = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "requirement satisfied by integrity-bound root exit"
        ]
        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert terminal.step_count == terminal.completed_step_count == 1
        assert terminal.requirement_count == terminal.satisfied_requirement_count == 1
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.SATISFIED
        ]
        assert len(evidence_items) == 1
        assert evidence_items[0].requirement_id == requirements[0].requirement_id
        assert len(evidence_items[0].metadata["completion_evidence_sha256"]) == 64
        assert len(evidence_items[0].metadata["outcome_sha256"]) == 64
    finally:
        runtime.close()


def test_one_generic_check_cannot_satisfy_two_task_run_requirements(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "bundled-requirements.sqlite",
        config=_v2_config(),
    )
    try:
        created = _create_coding_completion_run(runtime, "bundled-requirements")
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Independently confirm the follow-up requirement.",
            expected_revision=created.revision,
            command_id="follow-up:bundled-requirements",
        )
        observed = _run_discover_quantum(
            runtime,
            followed,
            "run:bundled-requirements:discover",
        )
        client = _StructuredTaskRunCompletionClient(bundle_requirements=True)
        runtime.llm.client = client

        rejected = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=observed.revision,
            command_id="run:bundled-requirements:claim",
            max_quanta=2,
        )

        assert rejected.status is not TaskRunStatus.SUCCEEDED
        assert client.review is not None
        assert created.root_pid is not None
        assert "task_run" not in client.review
        assert len(client.review["requirements"]) == 2
        assert client.review["available_evidence_tools"] == ["discover_skills"]
        assert all(
            "eligible_evidence_tools" not in item
            for item in client.review["requirements"]
        )
        assert {
            requirement.status
            for requirement in runtime.store.list_task_run_requirements(
                created.run_id
            )
        } == {TaskRunRequirementStatus.IN_PROGRESS}
        assert runtime.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
    finally:
        runtime.close()


def test_tool_receipt_from_old_llm_binding_cannot_prove_new_follow_up(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "old-follow-up-evidence.sqlite",
        config=_v2_config(),
    )
    try:
        created = _create_coding_completion_run(runtime, "old-follow-up-evidence")
        observed = _run_discover_quantum(
            runtime,
            created,
            "run:old-follow-up-evidence:discover",
        )
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Independently confirm this later follow-up.",
            expected_revision=observed.revision,
            command_id="follow-up:old-follow-up-evidence",
        )
        client = _StructuredTaskRunCompletionClient(bundle_requirements=False)
        runtime.llm.client = client

        rejected = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=followed.revision,
            command_id="run:old-follow-up-evidence:claim",
            max_quanta=2,
        )

        assert rejected.status is not TaskRunStatus.SUCCEEDED
        assert client.review is not None
        assert created.root_pid is not None
        assert "eligible_evidence_tools" not in client.review["requirements"][0]
        assert client.review["requirements"][1]["eligible_evidence_tools"] == []
        assert {
            requirement.status
            for requirement in runtime.store.list_task_run_requirements(
                created.run_id
            )
        } == {TaskRunRequirementStatus.IN_PROGRESS}
        assert runtime.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
    finally:
        runtime.close()


def test_tool_dispatched_after_follow_up_keeps_frozen_old_requirement_binding(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "frozen-binding-follow-up-race.sqlite",
        config=_v2_config(),
    )
    provider = _BlockingDiscoverClient()
    try:
        created = _create_coding_completion_run(
            runtime,
            "frozen-binding-follow-up-race",
        )
        assert created.root_pid is not None
        runtime.llm.client = provider
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:frozen-binding-before-follow-up",
                max_quanta=1,
            )
            assert provider.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            followed = runtime.task_runs.follow_up(
                created.run_id,
                "Independently confirm the follow-up added after provider admission.",
                kind="normal",
                required=True,
                expected_revision=running.revision,
                command_id="follow-up:frozen-binding-race",
            )
            provider.release.set()
            future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        tool_operations = [
            operation
            for operation in runtime.store.list_operations(pid=created.root_pid)
            if operation.name == "tool.discover_skills"
        ]
        assert len(tool_operations) == 1
        follow_up = runtime.store.list_task_run_requirements(created.run_id)[1]
        assert tool_operations[0].started_at >= follow_up.created_at

        claim_client = _StructuredTaskRunCompletionClient(
            bundle_requirements=False
        )
        runtime.llm.client = claim_client
        current = runtime.task_runs.get(created.run_id)
        rejected = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=current.revision,
            command_id="run:frozen-binding-after-follow-up",
            max_quanta=2,
        )

        assert rejected.status is not TaskRunStatus.SUCCEEDED
        assert followed.requirement_count == 2
        assert claim_client.review is not None
        assert "task_run" not in claim_client.review
        assert claim_client.review["requirements"][1]["eligible_evidence_tools"] == []
        assert {
            requirement.status
            for requirement in runtime.store.list_task_run_requirements(
                created.run_id
            )
        } == {TaskRunRequirementStatus.IN_PROGRESS}
    finally:
        provider.release.set()
        runtime.close()


def test_structured_completion_persists_causal_receipts_and_started_at(
    tmp_path: Path,
) -> None:
    database = tmp_path / "causal-completion-receipts.sqlite"
    runtime = Runtime.open(database, config=_v2_config())
    try:
        created = _create_coding_completion_run(runtime, "causal-completion-receipts")
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Independently confirm the later follow-up.",
            expected_revision=created.revision,
            command_id="follow-up:causal-completion-receipts",
        )
        observed = _run_discover_quantum(
            runtime,
            followed,
            "run:causal-completion-receipts:discover",
        )
        before = runtime.store.list_task_run_requirements(created.run_id)
        started_at = {
            requirement.requirement_id: requirement.started_at
            for requirement in before
        }
        assert all(value is not None for value in started_at.values())
        client = _StructuredTaskRunCompletionClient(bundle_requirements=False)
        runtime.llm.client = client

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=observed.revision,
            command_id="run:causal-completion-receipts:claim",
            max_quanta=2,
        )

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert all(
            requirement.status is TaskRunRequirementStatus.SATISFIED
            and requirement.started_at == started_at[requirement.requirement_id]
            for requirement in requirements
        )
        evidence_items = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "requirement satisfied by integrity-bound root exit"
        ]
        assert len(evidence_items) == 2
        receipt_ids = {
            receipt_id
            for item in evidence_items
            for receipt_id in item.metadata["evidence_receipt_ids"]
        }
        assert len(receipt_ids) == 1
        receipt_id = next(iter(receipt_ids))
        tool_operation = runtime.store.get_operation(receipt_id)
        assert tool_operation is not None
        assert tool_operation.name == "tool.discover_skills"
        assert tool_operation.parent_operation_id is not None
        llm_operation = runtime.store.get_operation(
            tool_operation.parent_operation_id
        )
        assert llm_operation is not None
        assert llm_operation.kind.value == "llm_request"
        assert llm_operation.operation_id == llm_operation.root_operation_id
        assert tool_operation.root_operation_id == llm_operation.root_operation_id
        llm_links = runtime.store.list_operation_evidence(
            operation_ids=[llm_operation.operation_id],
            evidence_types=["llm_call"],
        )
        successful_links = [
            link
            for link in llm_links
            if link.role == "invocation" and link.metadata.get("status") == "ok"
        ]
        assert len(successful_links) == 1
        frozen_call = runtime.store.get_llm_call(successful_links[0].evidence_id)
        assert frozen_call is not None
        frozen = frozen_call.request_options["task_run_requirement_binding_v1"]
        assert {item["requirement_id"] for item in frozen["requirements"]} == {
            requirement.requirement_id for requirement in requirements
        }
        completion_links = [
            link
            for link in runtime.store.list_task_run_links(created.run_id)
            if link.role.startswith("requirement_completion:")
        ]
        assert len(completion_links) == 2
        assert {link.evidence_id for link in completion_links} == {receipt_id}
        persisted_requirements = requirements
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_v2_config())
    try:
        assert (
            reopened.store.list_task_run_requirements(created.run_id)
            == persisted_requirements
        )
        assert all(
            requirement.started_at
            == started_at[requirement.requirement_id]
            for requirement in reopened.store.list_task_run_requirements(
                created.run_id
            )
        )
    finally:
        reopened.close()


def test_json_stringified_completion_evidence_satisfies_each_requirement(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "stringified-completion-evidence.sqlite",
        config=_v2_config(),
    )
    try:
        created = _create_coding_completion_run(
            runtime,
            "stringified-completion-evidence",
        )
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Independently confirm the stringified-evidence follow-up.",
            expected_revision=created.revision,
            command_id="follow-up:stringified-completion-evidence",
        )
        observed = _run_discover_quantum(
            runtime,
            followed,
            "run:stringified-completion-evidence:discover",
        )
        client = _StructuredTaskRunCompletionClient(
            bundle_requirements=False,
            stringify_completion_evidence=True,
        )
        runtime.llm.client = client

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=observed.revision,
            command_id="run:stringified-completion-evidence:claim",
            max_quanta=2,
        )

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        evidence_items = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "requirement satisfied by integrity-bound root exit"
        ]
        assert terminal.status is TaskRunStatus.SUCCEEDED
        assert terminal.step_count == terminal.completed_step_count
        assert terminal.requirement_count == terminal.satisfied_requirement_count == 2
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.SATISFIED,
            TaskRunRequirementStatus.SATISFIED,
        ]
        assert len(evidence_items) == 2
        assert all(item.metadata["evidence_receipt_ids"] for item in evidence_items)
        assert created.root_pid is not None
        assert runtime.process.get(created.root_pid).status is ProcessStatus.EXITED
    finally:
        runtime.close()


def test_malformed_stringified_completion_evidence_fails_before_terminal_exit(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        tmp_path / "malformed-stringified-completion-evidence.sqlite",
        config=_v2_config(),
    )
    try:
        created = _create_coding_completion_run(
            runtime,
            "malformed-stringified-completion-evidence",
        )
        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Independently confirm the malformed-evidence follow-up.",
            expected_revision=created.revision,
            command_id="follow-up:malformed-stringified-completion-evidence",
        )
        observed = _run_discover_quantum(
            runtime,
            followed,
            "run:malformed-stringified-completion-evidence:discover",
        )
        client = _StructuredTaskRunCompletionClient(
            bundle_requirements=False,
            malformed_completion_evidence=True,
        )
        runtime.llm.client = client

        blocked = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=observed.revision,
            command_id="run:malformed-stringified-completion-evidence:claim",
            max_quanta=2,
        )

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        record = runtime.store.get_task_run(created.run_id)
        assert record is not None
        assert blocked.status not in {
            TaskRunStatus.SUCCEEDED,
            TaskRunStatus.FAILED,
            TaskRunStatus.CANCELLED,
            TaskRunStatus.FINALIZING,
            TaskRunStatus.NEEDS_ATTENTION,
        }
        assert blocked.satisfied_requirement_count == 0
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.IN_PROGRESS,
            TaskRunRequirementStatus.IN_PROGRESS,
        ]
        assert record.completed_at is None
        assert record.finalized_at is None
        assert created.root_pid is not None
        assert runtime.process.get(created.root_pid).status is ProcessStatus.RUNNABLE
        assert not any(
            item.label == "requirement satisfied by integrity-bound root exit"
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("reported_status", ["blocked", "cancelled"])
def test_coding_exit_cannot_promote_unresolved_requirement_to_satisfied(
    tmp_path: Path,
    reported_status: str,
) -> None:
    database = tmp_path / f"unresolved-{reported_status}.sqlite"
    runtime = Runtime.open(database, config=_v2_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal=(
                    "Create the durable deliverable UNRESOLVED_DELIVERABLE.txt "
                    "with the exact text 'completed' and verify its contents."
                ),
                display_title=f"Unresolved {reported_status}",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id=f"create:unresolved:{reported_status}",
        )
        root_pid = created.root_pid
        assert root_pid is not None
        client = _UnresolvedCompletionClient(reported_status)
        runtime.llm.client = client

        terminal = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"run:unresolved:{reported_status}",
            max_quanta=5,
        )

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        unresolved_items = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "requirement unresolved by integrity-bound root exit"
        ]
        assert terminal.status is TaskRunStatus.NEEDS_ATTENTION
        assert terminal.completed_step_count == terminal.step_count
        assert terminal.step_count > 0
        assert terminal.satisfied_requirement_count == 0
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.BLOCKED
        ]
        assert [blocker["kind"] for blocker in terminal.blockers] == [
            "requirements_unsatisfied"
        ]
        assert set(terminal.allowed_actions) == {
            TaskRunAction.RECOVER,
            TaskRunAction.CANCEL,
        }
        assert len(unresolved_items) == 1
        assert unresolved_items[0].metadata["reported_status"] == reported_status
        assert runtime.process.get(root_pid).status is ProcessStatus.EXITED
        resume_point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert resume_point is not None
        assert resume_point.pending_action_payload_id is None
        assert runtime.store.get_llm_pending_action(root_pid) is None
        assert client.calls == 3

        persisted_requirements = requirements
        persisted_ledger = runtime.task_runs.list_ledger(
            created.run_id,
            limit=100,
        ).records
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        redispatch_guard = _CountingExplodingClient()
        reopened.llm.client = redispatch_guard

        recovered = reopened.task_runs.get(created.run_id)
        # Reopen advances the fenced Runtime epoch, which monotonically bumps
        # revision/updated_at without changing the terminal projection.
        assert recovered.revision == terminal.revision + 1
        assert replace(
            recovered,
            revision=terminal.revision,
            updated_at=terminal.updated_at,
        ) == terminal
        assert (
            reopened.store.list_task_run_requirements(created.run_id)
            == persisted_requirements
        )
        assert (
            reopened.task_runs.list_ledger(created.run_id, limit=100).records
            == persisted_ledger
        )
        recovered_point = reopened.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert recovered_point is not None
        assert recovered_point.pending_action_payload_id is None
        assert reopened.store.get_llm_pending_action(root_pid) is None
        assert reopened.process.get(root_pid).status is ProcessStatus.EXITED
        assert reopened.run_next_process_once() is None
        assert redispatch_guard.calls == 0

        assert set(recovered.allowed_actions) == {
            TaskRunAction.RECOVER,
            TaskRunAction.CANCEL,
        }
        requirements_before_rejected_follow_up = (
            reopened.store.list_task_run_requirements(created.run_id)
        )
        with pytest.raises(
            ValidationError,
            match="root process no longer accepts follow-up requirements",
        ):
            reopened.task_runs.follow_up(
                created.run_id,
                "This must not be delivered to an exited root.",
                expected_revision=recovered.revision,
                command_id=f"follow-up:terminal-root:{reported_status}",
            )
        assert reopened.task_runs.get(created.run_id) == recovered
        assert (
            reopened.store.list_task_run_requirements(created.run_id)
            == requirements_before_rejected_follow_up
        )
        assert (
            reopened.store.get_task_run_command(
                created.run_id,
                f"follow-up:terminal-root:{reported_status}",
            )
            is None
        )

        terminate = next(
            option
            for option in reopened.task_runs.recovery_options(created.run_id)
            if option.option_id == "terminate_run"
        )
        terminated = reopened.task_runs.recover(
            created.run_id,
            option_id=terminate.option_id,
            expected_revision=recovered.revision,
            command_id=f"recover:terminal-root:{reported_status}",
        )

        assert terminated.status is TaskRunStatus.CANCELLED
        assert terminated.allowed_actions == (TaskRunAction.RERUN,)
        assert terminated.completed_at is not None
        assert reopened.process.get(root_pid).status is ProcessStatus.EXITED
        assert [
            item.status
            for item in reopened.store.list_task_run_requirements(created.run_id)
        ] == [TaskRunRequirementStatus.BLOCKED]
        termination_transitions = [
            item
            for item in reopened.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.label == "manual recovery termination intent persisted"
        ]
        assert len(termination_transitions) == 1
        command_id = f"recover:terminal-root:{reported_status}"
        termination_command = reopened.store.get_task_run_command(
            created.run_id,
            command_id,
        )
        assert termination_command is not None
        transition = termination_transitions[0]
        assert transition.seq == termination_command.result[
            "admission_ledger_seq"
        ]
        assert transition.item_id == termination_command.result[
            "admission_ledger_item_id"
        ]
        assert transition.metadata == {
            "schema_version": 1,
            "from": "needs_attention",
            "to": "cancelling",
            "command_id": command_id,
            "command_kind": "recover",
            "request_hash": termination_command.request_hash,
            "admission_evidence_sha256": termination_command.result[
                "admission_evidence_sha256"
            ],
        }
        assert (
            reopened.task_runs.recover(
                created.run_id,
                option_id=terminate.option_id,
                expected_revision=recovered.revision,
                command_id=f"recover:terminal-root:{reported_status}",
            )
            == terminated
        )
    finally:
        reopened.close()

    terminal_reopen = Runtime.open(database, config=_config())
    try:
        stable = terminal_reopen.task_runs.get(created.run_id)
        assert stable == terminated
        assert stable.status is TaskRunStatus.CANCELLED
        assert stable.allowed_actions == (TaskRunAction.RERUN,)
        assert terminal_reopen.process.get(root_pid).status is ProcessStatus.EXITED
        assert terminal_reopen.run_next_process_once() is None
        assert (
            terminal_reopen.task_runs.recover(
                created.run_id,
                option_id="terminate_run",
                expected_revision=recovered.revision,
                command_id=f"recover:terminal-root:{reported_status}",
            )
            == terminated
        )
    finally:
        terminal_reopen.close()


def test_host_exit_after_prompt_visibility_cannot_satisfy_requirements(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "host-exit-requirement.sqlite", config=_config())
    try:
        created = _create(
            runtime,
            "host-exit-requirement",
            retention=TaskRunRetention.PERMANENT,
        )
        persisted = runtime.store.get_task_run(created.run_id)
        assert persisted is not None and created.root_pid is not None
        started = runtime.store.update_task_run_cas(
            created.run_id,
            persisted.revision,
            updates={
                "status": TaskRunStatus.RUNNING,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            expected_runtime_epoch=runtime.task_runs.runtime_epoch,
        )
        assert runtime.task_runs.prompt_context_for_pid(created.root_pid) is not None
        visible = runtime.store.list_task_run_requirements(created.run_id)
        assert [item.status for item in visible] == [
            TaskRunRequirementStatus.IN_PROGRESS
        ]

        runtime.process.exit(created.root_pid, payload={"forced": True})
        projected = runtime.task_runs._project(started, allow_finalize=True)

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        assert projected.status is TaskRunStatus.NEEDS_ATTENTION
        assert [blocker["kind"] for blocker in projected.blockers] == [
            "requirements_unsatisfied"
        ]
        assert [item.status for item in requirements] == [
            TaskRunRequirementStatus.IN_PROGRESS
        ]
        assert projected.satisfied_requirement_count == 0
        assert projected.step_count == projected.completed_step_count == 0
        assert (
            runtime.store.get_task_run_resume_point(
                created.root_pid,
                complete_only=True,
            )
            is None
        )
    finally:
        runtime.close()


def test_task_run_human_wait_survives_two_reopens_without_second_llm_call(
    tmp_path: Path,
) -> None:
    database = tmp_path / "human-two-reopens.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(first, "human-two-reopens")
        root_pid = created.root_pid
        assert root_pid is not None
        first.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        first.skills.activate_skill(
            root_pid,
            HUMAN_COLLABORATION_SKILL,
            actor=root_pid,
        )
        client = _PlannedClient(
            [
                {
                    "action": "ask_human",
                    "question": "Continue this durable run? HUMAN_PROMPT_MUST_PURGE",
                }
            ]
        )
        first.llm.client = client
        waiting = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:ask-human",
            max_quanta=1,
        )
        pending = first.store.get_llm_pending_action(root_pid)
        assert waiting.status is TaskRunStatus.WAITING_HUMAN
        assert pending is not None
        request_id = pending["request_id"]
        resume_token = pending["resume_token"]
        assert client.calls == 1
    finally:
        first.close()

    second = Runtime.open(database, config=_config())
    try:
        second.llm.client = _ExplodingClient()
        pending = second.store.get_llm_pending_action(root_pid)
        assert pending is not None
        assert (pending["request_id"], pending["resume_token"]) == (
            request_id,
            resume_token,
        )
        assert second.task_runs.get(created.run_id).status is TaskRunStatus.WAITING_HUMAN
        assert [item.request_id for item in second.human.pending()] == [request_id]
    finally:
        second.close()

    third = Runtime.open(database, config=_config())
    try:
        third.llm.client = _ExplodingClient()
        pending = third.store.get_llm_pending_action(root_pid)
        assert pending is not None
        assert (pending["request_id"], pending["resume_token"]) == (
            request_id,
            resume_token,
        )
        assert [item.request_id for item in third.human.pending()] == [request_id]
        third.human.drain_terminal_queue(
            auto_answer="yes HUMAN_ANSWER_MUST_PURGE"
        )

        current = third.task_runs.get(created.run_id)
        resumed = third.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=current.revision,
            command_id="run:resume-human",
            max_quanta=1,
        )
        assert resumed.status is TaskRunStatus.RUNNING
        completed = third.store.get_llm_pending_action(root_pid)
        assert completed is not None and completed["status"] == "completed"
        assert completed["resume_token"] == resume_token

        third.llm.client = _PlannedClient(
            [{"action": "process_exit", "payload": {"answer": "accepted"}}]
        )
        terminal = third.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=resumed.revision,
            command_id="run:finish-human",
            max_quanta=1,
        )
        assert terminal.status is TaskRunStatus.SUCCEEDED
        redacted_request = third.store.get_human_request(request_id)
        assert redacted_request is not None
        rendered_request = json.dumps(
            {
                "payload": redacted_request.payload,
                "decision": redacted_request.decision,
            },
            sort_keys=True,
        )
        assert "HUMAN_PROMPT_MUST_PURGE" not in rendered_request
        assert "HUMAN_ANSWER_MUST_PURGE" not in rendered_request
        assert redacted_request.payload[
            "$agent_libos_task_run_human_redaction"
        ]["request_type"] == "question"
    finally:
        third.close()


def test_answered_human_wait_settles_before_unread_interrupt_after_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "human-answer-before-interrupt.sqlite"
    answer = "unique-answer-after-reopen"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(first, "human-answer-before-interrupt")
        root_pid = created.root_pid
        assert root_pid is not None
        first.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        first.skills.activate_skill(
            root_pid,
            HUMAN_COLLABORATION_SKILL,
            actor=root_pid,
        )
        first.llm.client = _PlannedClient(
            [
                {
                    "action": "ask_human",
                    "question": "Which durable choice should be used?",
                }
            ]
        )
        waiting = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:human-before-interrupt",
            max_quanta=1,
        )
        pending = first.store.get_llm_pending_action(root_pid)
        assert waiting.status is TaskRunStatus.WAITING_HUMAN
        assert pending is not None
        request_id = str(pending["request_id"])
        resume_token = str(pending["resume_token"])

        followed = first.task_runs.follow_up(
            created.run_id,
            "Keep the answer internal.",
            kind="interrupt",
            required=True,
            expected_revision=waiting.revision,
            command_id="follow-up:human-before-interrupt",
        )
        assert followed.status is TaskRunStatus.WAITING_HUMAN
        assert len(
            first.messages.unread(root_pid, kind=ProcessMessageKind.INTERRUPT)
        ) == 1
    finally:
        first.close()

    second = Runtime.open(database, config=_config())
    try:
        second.llm.client = _ExplodingClient()
        assert (
            second.task_runs.get(created.run_id).status
            is TaskRunStatus.WAITING_HUMAN
        )
        pending = second.store.get_llm_pending_action(root_pid)
        assert pending is not None
        assert (pending["request_id"], pending["resume_token"]) == (
            request_id,
            resume_token,
        )
        second.human.drain_terminal_queue(auto_answer=answer)

        current = second.task_runs.get(created.run_id)
        resumed = second.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=current.revision,
            command_id="run:settle-human-before-interrupt",
            max_quanta=1,
        )
        assert resumed.status is TaskRunStatus.RUNNING
        completed = second.store.get_llm_pending_action(root_pid)
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["resume_token"] == resume_token
        requests = second.store.list_human_requests_for_pids(
            (root_pid,),
            limit=10,
        )["records"]
        assert [item.request_id for item in requests] == [request_id]
        assert len(
            second.messages.unread(root_pid, kind=ProcessMessageKind.INTERRUPT)
        ) == 1

        point = second.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert point is not None
        transcript = second.store.get_task_run_payload(point.transcript_payload_id)
        assert transcript is not None and transcript.canonical_json is not None
        assert transcript.canonical_json.count(answer) == 1

        class CaptureNextPrompt(_PlannedClient):
            def __init__(self) -> None:
                super().__init__(
                    [
                        {
                            "action": "process_exit",
                            "payload": {"should_not_exit": True},
                        }
                    ]
                )
                self.messages: list[dict[str, Any]] = []

            def complete_action(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]],
            ) -> LLMCompletion:
                self.messages = [dict(message) for message in messages]
                return super().complete_action(messages, tools)

        capture = CaptureNextPrompt()
        second.llm.client = capture
        second.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=resumed.revision,
            command_id="run:observe-answer-before-interrupt",
            max_quanta=1,
        )
        assert capture.calls == 1
        answer_transcript = [
            message
            for message in capture.messages
            if message.get("role") == "assistant"
            and answer in str(message.get("content", ""))
        ]
        assert len(answer_transcript) == 1
        assert str(answer_transcript[0]["content"]).count(answer) == 1
        assert any(
            "Pending explicit process input" in str(message.get("content", ""))
            for message in capture.messages
        )
        assert second.process.get(root_pid).status not in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        }
        assert len(
            second.messages.unread(root_pid, kind=ProcessMessageKind.INTERRUPT)
        ) == 1
        assert len(
            second.store.list_human_requests_for_pids(
                (root_pid,),
                limit=10,
            )["records"]
        ) == 1
    finally:
        second.close()


def test_task_run_large_quantum_budget_returns_at_real_human_wait(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "large-budget-human-wait.sqlite", config=_config())
    entered_provider = threading.Event()
    release_provider = threading.Event()

    class SlowHumanClient(_PlannedClient):
        def complete_action(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
        ) -> LLMCompletion:
            entered_provider.set()
            if not release_provider.wait(timeout=THREAD_SYNC_TIMEOUT_S):
                raise AssertionError("test did not release the slow LLM provider")
            return super().complete_action(messages, tools)

    try:
        created = _create(runtime, "large-budget-human-wait")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            root_pid,
            HUMAN_COLLABORATION_SKILL,
            actor=root_pid,
        )
        client = SlowHumanClient(
            [
                {
                    "action": "ask_human",
                    "question": "Choose monthly or annual billing?",
                }
            ]
        )
        runtime.llm.client = client

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:large-budget-human-wait",
                max_quanta=48,
            )
            assert entered_provider.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            release_provider.set()
            waiting = future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        assert waiting.status is TaskRunStatus.WAITING_HUMAN
        assert client.calls == 1
        assert runtime.process.get(root_pid).status is ProcessStatus.WAITING_HUMAN
        effects = runtime.store.list_external_effects(pid=root_pid)
        assert effects
        assert all(
            effect.effect_state == "finalized"
            and effect.transaction_state == "committed"
            for effect in effects
        )
    finally:
        release_provider.set()
        runtime.close()


def test_task_run_syscall_surfaces_durable_human_wait_without_in_memory_polling(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "durable-human-syscall.sqlite", config=_config())
    try:
        created = _create(runtime, "durable-human-syscall")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        session = LibOSSyscallSession(runtime, root_pid)

        with pytest.raises(HumanResponseRequired) as raised:
            asyncio.run(
                session.handle(
                    "human.ask",
                    {"question": "Choose monthly or annual billing?"},
                )
            )

        process = runtime.process.get(root_pid)
        assert process.status is ProcessStatus.WAITING_HUMAN
        assert process.task_run_id == created.run_id
        assert [item.request_id for item in runtime.human.pending()] == [
            raised.value.request_id
        ]
    finally:
        runtime.close()


def test_interrupt_accepts_settled_gui_presentation_after_human_wait_safe_point(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "human-presentation-interrupt.sqlite", config=_config())
    try:
        created = _create(runtime, "human-presentation-interrupt")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            root_pid,
            HUMAN_COLLABORATION_SKILL,
            actor=root_pid,
        )
        runtime.llm.client = _PlannedClient(
            [
                {
                    "action": "ask_human",
                    "question": "Has the charge been verified?",
                }
            ]
        )
        waiting = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:human-presentation-interrupt",
            max_quanta=1,
        )
        pending = runtime.store.get_llm_pending_action(root_pid)
        assert waiting.status is TaskRunStatus.WAITING_HUMAN
        assert pending is not None
        request_id = str(pending["request_id"])

        record_external_effect(
            runtime.uow.protected_effects,
            pid=root_pid,
            provider="human",
            operation="write",
            target="human:owner",
            classification=ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
            ),
            audit_record=None,
            event=None,
            metadata={
                "request_id": request_id,
                "channel": "gui",
                "presented": True,
            },
        )

        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Keep this draft internal and contact nobody.",
            kind="interrupt",
            required=True,
            expected_revision=waiting.revision,
            command_id="follow-up:human-presentation-interrupt",
        )

        assert followed.status is not TaskRunStatus.NEEDS_ATTENTION
        assert followed.requirement_count == 2
        assert "unknown_effect" not in {
            item["kind"] for item in followed.blockers
        }
        assert runtime.store.list_external_effects_changed_after(
            0,
            pids=(root_pid,),
        )[-1].transaction_state == "committed"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("case", "target", "classification"),
    [
        (
            "state-mutating",
            "human:owner",
            ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=True,
                information_flow=True,
            ),
        ),
        (
            "wrong-rollback-classification",
            "human:owner",
            ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=False,
                information_flow=True,
            ),
        ),
        (
            "wrong-target",
            "human:another-owner",
            ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
            ),
        ),
    ],
)
def test_interrupt_rejects_non_gui_presentation_effect_after_human_wait(
    tmp_path: Path,
    case: str,
    target: str,
    classification: ExternalEffectClassification,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"human-presentation-rejected-{case}.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime, f"human-presentation-rejected-{case}")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.capability.grant(
            root_pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.skills.activate_skill(
            root_pid,
            HUMAN_COLLABORATION_SKILL,
            actor=root_pid,
        )
        client = _PlannedClient(
            [
                {
                    "action": "ask_human",
                    "question": "Has the charge been verified?",
                }
            ]
        )
        runtime.llm.client = client
        waiting = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id=f"run:human-presentation-rejected:{case}",
            max_quanta=1,
        )
        pending = runtime.store.get_llm_pending_action(root_pid)
        assert waiting.status is TaskRunStatus.WAITING_HUMAN
        assert pending is not None

        record_external_effect(
            runtime.uow.protected_effects,
            pid=root_pid,
            provider="human",
            operation="write",
            target=target,
            classification=classification,
            audit_record=None,
            event=None,
            metadata={
                "request_id": str(pending["request_id"]),
                "channel": "gui",
                "presented": True,
            },
        )

        followed = runtime.task_runs.follow_up(
            created.run_id,
            "Keep this draft internal and contact nobody.",
            kind="interrupt",
            required=True,
            expected_revision=waiting.revision,
            command_id=f"follow-up:human-presentation-rejected:{case}",
        )

        assert followed.status is TaskRunStatus.NEEDS_ATTENTION
        assert "unknown_effect" in {
            item["kind"] for item in followed.blockers
        }
        with pytest.raises(ValidationError, match="cannot dispatch"):
            runtime.task_runs.run_until_blocked(
                created.run_id,
                expected_revision=followed.revision,
                command_id=f"run:after-rejected-presentation:{case}",
                max_quanta=1,
            )
        assert client.calls == 1
    finally:
        runtime.close()


def test_task_run_message_wait_and_pause_token_survive_two_reopens(
    tmp_path: Path,
) -> None:
    database = tmp_path / "message-two-reopens.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        created = _create(first, "message-two-reopens")
        root_pid = created.root_pid
        assert root_pid is not None
        first.skills.activate_skill(root_pid, CHILD_PROCESS_SKILL, actor=root_pid)
        client = _PlannedClient(
            [
                {
                    "action": "receive_process_messages",
                    "channel": "control",
                    "correlation_id": "durable-message",
                }
            ]
        )
        first.llm.client = client
        waiting = first.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:message-wait",
            max_quanta=1,
        )
        pending = first.store.get_llm_pending_action(root_pid)
        assert waiting.status is TaskRunStatus.WAITING_MESSAGE
        assert pending is not None
        resume_token = pending["resume_token"]
        assert client.calls == 1
        status_transitions = [
            item
            for item in first.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.kind is TaskRunLedgerKind.STATUS_TRANSITION
        ]
        projected_wait = [
            item
            for item in status_transitions
            if item.metadata
            == {
                "from": TaskRunStatus.RUNNING.value,
                "to": TaskRunStatus.WAITING_MESSAGE.value,
            }
        ]
        assert len(projected_wait) == 1
        assert projected_wait[0].label == "process state projected"
        running_transition = next(
            item
            for item in status_transitions
            if item.metadata.get("to") == TaskRunStatus.RUNNING.value
        )
        assert running_transition.seq < projected_wait[0].seq
    finally:
        first.close()

    second = Runtime.open(database, config=_config())
    try:
        second.llm.client = _ExplodingClient()
        current = second.task_runs.get(created.run_id)
        assert current.status is TaskRunStatus.WAITING_MESSAGE
        assert second.store.get_llm_pending_action(root_pid)["resume_token"] == resume_token
        paused = second.task_runs.pause(
            created.run_id,
            expected_revision=current.revision,
            command_id="pause:message-wait",
        )
        assert paused.status is TaskRunStatus.PAUSED
        assert second.store.get_llm_pending_action(root_pid)["resume_token"] == resume_token
    finally:
        second.close()

    third = Runtime.open(database, config=_config())
    try:
        third.llm.client = _ExplodingClient()
        paused = third.task_runs.get(created.run_id)
        assert paused.status is TaskRunStatus.PAUSED
        resumed = third.task_runs.resume(
            created.run_id,
            expected_revision=paused.revision,
            command_id="resume:message-wait",
        )
        pending = third.store.get_llm_pending_action(root_pid)
        assert resumed.status is TaskRunStatus.WAITING_MESSAGE
        assert pending is not None and pending["resume_token"] == resume_token
        assert third.process.get(root_pid).status is ProcessStatus.WAITING_EVENT

        message = third.messages.post(
            sender="host",
            recipient_pid=root_pid,
            kind=ProcessMessageKind.NORMAL,
            channel="control",
            correlation_id="durable-message",
            subject="resume durable wait",
            payload={"ready": True},
        )
        after_message = third.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=resumed.revision,
            command_id="run:resume-message",
            max_quanta=1,
        )
        completed = third.store.get_llm_pending_action(root_pid)
        assert completed is not None and completed["status"] == "completed"
        assert completed["resume_token"] == resume_token
        stored_message = third.store.get_process_message(message.message_id)
        assert stored_message is not None and stored_message.acked_at is not None

        third.llm.client = _PlannedClient(
            [{"action": "process_exit", "payload": {"message": "received"}}]
        )
        terminal = third.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=after_message.revision,
            command_id="run:finish-message",
            max_quanta=1,
        )
        assert terminal.status is TaskRunStatus.SUCCEEDED
    finally:
        third.close()


def test_task_run_cancel_is_deep_to_shallow(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "cancel-tree.sqlite", config=_config())
    try:
        created = _create(runtime, "cancel-tree")
        root_pid = created.root_pid
        assert root_pid is not None
        _activate_child_tools(runtime, root_pid)
        runtime.llm.client = _PlannedClient(
            [{"action": "spawn_child_process", "goal": "cancel-child"}]
        )
        running = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:spawn-cancel-child",
            max_quanta=1,
        )
        child = next(
            process
            for process in runtime.store.list_processes_for_task_run(created.run_id)
            if process.parent_pid == root_pid
        )
        runtime.process.pause(root_pid, "allow child to create a grandchild")
        _activate_child_tools(runtime, child.pid)
        runtime.llm.client = _PlannedClient(
            [{"action": "spawn_child_process", "goal": "cancel-grandchild"}]
        )
        running = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=running.revision,
            command_id="run:spawn-cancel-grandchild",
            max_quanta=1,
        )
        grandchild = next(
            process
            for process in runtime.store.list_processes_for_task_run(created.run_id)
            if process.parent_pid == child.pid
        )

        audit_start = len(runtime.audit.trace())
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=running.revision,
            command_id="cancel:tree",
            reason="test deep-to-shallow cancellation",
        )
        signals = [
            item
            for item in runtime.audit.trace()[audit_start:]
            if item.action == "process.signal"
            and item.decision.get("signal") == "cancel"
        ]
        assert [item.target for item in signals] == [
            f"process:{grandchild.pid}",
            f"process:{child.pid}",
            f"process:{root_pid}",
        ]
        assert terminal.status is TaskRunStatus.CANCELLED
        assert {
            process.status
            for process in runtime.store.list_processes_for_task_run(created.run_id)
        } == {ProcessStatus.KILLED}
    finally:
        runtime.close()


def test_task_run_cancel_during_inflight_llm_never_fakes_terminal_success(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "cancel-inflight.sqlite", config=_config())
    client = _BlockingExitClient()
    try:
        created = _create(runtime, "cancel-inflight")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:inflight",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            assert running.status is TaskRunStatus.RUNNING
            cancellation = runtime.task_runs.cancel(
                created.run_id,
                expected_revision=running.revision,
                command_id="cancel:inflight",
            )
            assert cancellation.status is not TaskRunStatus.CANCELLED
            assert cancellation.status is TaskRunStatus.NEEDS_ATTENTION
            assert "unknown_effect" in {
                blocker["kind"] for blocker in cancellation.blockers
            }
            client.release.set()
            worker_result = future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        final = runtime.task_runs.get(created.run_id)
        assert client.calls == 1
        assert final.status is not TaskRunStatus.SUCCEEDED
        assert worker_result.status is not TaskRunStatus.SUCCEEDED
        assert runtime.process.get(root_pid).status is not ProcessStatus.EXITED
    finally:
        client.release.set()
        runtime.close()


def test_task_run_quantum_budget_drains_admitted_provider_before_returning(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(),
        scheduler=replace(
            _config().scheduler,
            poll_interval_s=0.001,
            drain_window_s=0.01,
        ),
    )
    runtime = Runtime.open(tmp_path / "budget-drains-provider.sqlite", config=config)
    client = _BlockingExitClient()
    try:
        created = _create(runtime, "budget-drains-provider")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:budget-drains-provider",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            time.sleep(0.05)
            assert not future.done()

            client.release.set()
            completed = future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        assert completed.status is TaskRunStatus.SUCCEEDED
        assert runtime.process.get(root_pid).status is ProcessStatus.EXITED
        effects = runtime.store.list_external_effects(pid=root_pid)
        assert effects
        assert {
            (effect.effect_state, effect.transaction_state)
            for effect in effects
        } == {("finalized", "committed")}
    finally:
        client.release.set()
        runtime.close()


@pytest.mark.parametrize("control", ["cancel", "deadline"])
def test_inflight_settlement_cannot_strand_control_attention_on_revision_race(
    control: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": datetime.now(timezone.utc)}

    class ControlledDatetime(datetime):
        @classmethod
        def now(cls, tz: Any | None = None) -> datetime:
            selected = clock["now"]
            if tz is None:
                return selected.replace(tzinfo=None)
            return selected.astimezone(tz)

    monkeypatch.setattr("agent_libos.runtime.task_runs.datetime", ControlledDatetime)
    runtime = Runtime.open(
        tmp_path / f"{control}-attention-revision-race.sqlite",
        config=_config(),
    )
    # Keep deadline progression under test control. Runner load must not expire
    # the deadline before the Provider and Run reach the intended race point.
    deadline_at = (
        (
            clock["now"]
            + timedelta(seconds=REAL_DEADLINE_TEST_WINDOW_S)
        ).isoformat()
        if control == "deadline"
        else None
    )
    client = _BlockingExitClient()
    try:
        spec = TaskRunSpecV1(
            goal={"objective": f"{control}-attention-revision-race"},
            display_title=f"{control}-attention-revision-race",
            image_id="base-agent:v0",
            retention=TaskRunRetention.PERMANENT,
            deadline_at=deadline_at,
        )
        created = runtime.task_runs.create(
            spec,
            client_request_id=f"create:{control}-attention-revision-race",
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client

        original_update = runtime.store.update_task_run_cas
        attention_read = threading.Event()
        settlement_committed = threading.Event()
        attempt_lock = threading.Lock()
        attention_attempts = 0

        def update_task_run_cas(*args: Any, **kwargs: Any) -> Any:
            nonlocal attention_attempts
            updates = dict(kwargs.get("updates") or {})
            selected_run_id = (
                str(args[0]) if args else str(kwargs.get("run_id") or "")
            )
            is_attention = (
                selected_run_id == created.run_id
                and updates.get("status") is TaskRunStatus.NEEDS_ATTENTION
            )
            first_attention = False
            if is_attention:
                with attempt_lock:
                    attention_attempts += 1
                    first_attention = attention_attempts == 1
            if first_attention:
                # `_mark_attention` has already read this expected revision. Let
                # the admitted Provider settlement advance it before the CAS.
                attention_read.set()
                if not settlement_committed.wait(timeout=THREAD_SYNC_TIMEOUT_S):
                    raise AssertionError(
                        "admitted Provider settlement did not advance the Run revision"
                    )
            updated = original_update(*args, **kwargs)
            if selected_run_id == created.run_id and "step_count" in updates:
                settlement_committed.set()
            return updated

        monkeypatch.setattr(runtime.store, "update_task_run_cas", update_task_run_cas)

        with ThreadPoolExecutor(max_workers=2) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id=f"run:{control}-attention-revision-race",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            assert running.status is TaskRunStatus.RUNNING

            if control == "deadline":
                assert deadline_at is not None
                clock["now"] = datetime.fromisoformat(deadline_at) + timedelta(
                    microseconds=1
                )
                control_future = pool.submit(
                    runtime.task_runs.get,
                    created.run_id,
                )
            else:
                control_future = pool.submit(
                    runtime.task_runs.cancel,
                    created.run_id,
                    expected_revision=running.revision,
                    command_id="cancel:attention-revision-race",
                )

            assert attention_read.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            client.release.set()
            controlled = control_future.result(timeout=THREAD_SYNC_TIMEOUT_S)
            run_future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        assert settlement_committed.is_set()
        assert attention_attempts >= 2
        assert controlled.status is TaskRunStatus.NEEDS_ATTENTION
        persisted = runtime.task_runs.get(created.run_id)
        assert persisted.status is not TaskRunStatus.CANCELLING
        assert runtime.process.get(root_pid).status is not ProcessStatus.EXITED
        assert client.calls == 1
        if control == "cancel":
            replayed = runtime.task_runs.cancel(
                created.run_id,
                expected_revision=running.revision,
                command_id="cancel:attention-revision-race",
            )
            assert replayed == controlled
    finally:
        client.release.set()
        runtime.close()


@pytest.mark.parametrize(
    "persist_completed_transcript",
    [True, False],
    ids=("local-settlement-committed", "staged-settlement-recovered"),
)
def test_cancel_intent_dominates_an_already_admitted_process_exit(
    persist_completed_transcript: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / (
        "cancel-admitted-process-exit-"
        f"{int(persist_completed_transcript)}.sqlite"
    )
    runtime = Runtime.open(database, config=_config())
    tool_admitted = threading.Event()
    release_tool = threading.Event()
    stage_entered = threading.Event()
    release_stage = threading.Event()
    try:
        created = _create(
            runtime,
            f"cancel-admitted-process-exit-{int(persist_completed_transcript)}",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = _PlannedClient(
            [{"action": "process_exit", "payload": {"result": "late-exit"}}]
        )

        original_adispatch = runtime.llm.actions.adispatch

        async def admitted_adispatch(
            pid: str,
            action: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            if action.get("action") == "process_exit":
                tool_admitted.set()
                if not release_tool.wait(timeout=THREAD_SYNC_TIMEOUT_S):
                    raise AssertionError("test did not release admitted process_exit")
            return await original_adispatch(pid, action, **kwargs)

        monkeypatch.setattr(runtime.llm.actions, "adispatch", admitted_adispatch)
        original_stage = runtime.task_runs.stage_completed_transcript

        def blocked_stage_completed_transcript(**kwargs: Any) -> None:
            stage_entered.set()
            if not release_stage.wait(timeout=THREAD_SYNC_TIMEOUT_S):
                raise AssertionError("test did not release local result staging")
            original_stage(**kwargs)

        monkeypatch.setattr(
            runtime.task_runs,
            "stage_completed_transcript",
            blocked_stage_completed_transcript,
        )
        if not persist_completed_transcript:
            monkeypatch.setattr(
                runtime.task_runs,
                "record_completed_transcript",
                lambda **_kwargs: None,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:cancel-admitted-process-exit",
                max_quanta=1,
            )
            assert tool_admitted.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            assert running.status is TaskRunStatus.RUNNING

            cancelled = runtime.task_runs.cancel(
                created.run_id,
                expected_revision=running.revision,
                command_id="cancel:admitted-process-exit",
            )
            assert cancelled.status is TaskRunStatus.NEEDS_ATTENTION
            assert runtime.process.get(root_pid).status is ProcessStatus.RUNNING

            release_tool.set()
            assert stage_entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            raced_projection = runtime.task_runs._project(
                runtime.task_runs._require_run(created.run_id),
                allow_finalize=True,
            )
            assert raced_projection.status is TaskRunStatus.NEEDS_ATTENTION
            assert "effect_unsettled" in {
                blocker["kind"] for blocker in raced_projection.blockers
            }
            assert raced_projection.completed_at is None
            release_stage.set()
            worker_result = run_future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        current = runtime.task_runs.get(created.run_id)
        requirements = runtime.store.list_task_run_requirements(created.run_id)
        effects = runtime.store.list_external_effects(pid=root_pid)
        assert effects
        effect_ids_before_reopen = tuple(effect.effect_id for effect in effects)
        assert {
            (effect.effect_state, effect.transaction_state) for effect in effects
        } == {("finalized", "committed")}
        if persist_completed_transcript:
            assert worker_result.status is TaskRunStatus.CANCELLED
            assert current.status is TaskRunStatus.CANCELLED
            assert worker_result.blockers == ()
            assert current.blockers == ()
            assert current.satisfied_requirement_count == 1
            assert requirements[0].status is TaskRunRequirementStatus.SATISFIED
        else:
            assert worker_result.status is TaskRunStatus.NEEDS_ATTENTION
            assert current.status is TaskRunStatus.NEEDS_ATTENTION
            assert "effect_unsettled" in {
                blocker["kind"] for blocker in current.blockers
            }
            assert current.completed_at is None
            assert current.completed_step_count == 0
            assert current.satisfied_requirement_count == 0
            assert requirements[0].status is TaskRunRequirementStatus.IN_PROGRESS
            point = runtime.store.get_task_run_resume_point(
                root_pid,
                complete_only=True,
            )
            assert point is not None and point.pending_action_payload_id is not None
            staged = runtime.task_runs._decode_pending_resume_payload(point)
            assert (staged["kind"], staged["state"]) == (
                "completed_outcome",
                "staged",
            )
            assert not any(
                item.label == "LLM action and paired outputs persisted"
                for item in runtime.task_runs.list_ledger(
                    created.run_id,
                    limit=100,
                ).records
            )
        status_transitions = [
            item
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
            if item.kind is TaskRunLedgerKind.STATUS_TRANSITION
        ]
        terminal_transitions = [
            item
            for item in status_transitions
            if item.metadata.get("to")
            in {
                TaskRunStatus.FINALIZING.value,
                TaskRunStatus.CANCELLED.value,
            }
        ]
        assert bool(terminal_transitions) is persist_completed_transcript
    finally:
        release_tool.set()
        release_stage.set()
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        reopened_summary = reopened.task_runs.get(created.run_id)
        assert reopened_summary.status is TaskRunStatus.CANCELLED
        assert reopened_summary.status is not TaskRunStatus.SUCCEEDED
        assert reopened_summary.blockers == ()
        assert reopened_summary.completed_step_count == reopened_summary.step_count
        assert reopened_summary.satisfied_requirement_count == 1
        reopened_point = reopened.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert reopened_point is not None
        assert reopened_point.pending_action_payload_id is None
        assert any(
            item.label == "LLM action and paired outputs persisted"
            for item in reopened.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
        )
        reopened_effects = reopened.store.list_external_effects(pid=root_pid)
        assert reopened_effects
        assert tuple(effect.effect_id for effect in reopened_effects) == (
            effect_ids_before_reopen
        )
        assert {
            (effect.effect_state, effect.transaction_state)
            for effect in reopened_effects
        } == {("finalized", "committed")}
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 1
    finally:
        reopened.close()


def test_staged_terminal_result_recovers_locally_without_stale_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "staged-terminal-result-recovery.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            "staged-terminal-result-recovery",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = _PlannedClient(
            [{"action": "process_exit", "payload": {"result": "settle-local"}}]
        )
        monkeypatch.setattr(
            runtime.task_runs,
            "record_completed_transcript",
            lambda **_kwargs: None,
        )

        staged = runtime.task_runs.run_until_blocked(
            created.run_id,
            expected_revision=created.revision,
            command_id="run:stage-terminal-result",
            max_quanta=1,
        )

        assert staged.status is TaskRunStatus.NEEDS_ATTENTION
        assert "effect_unsettled" in {
            blocker["kind"] for blocker in staged.blockers
        }
        assert staged.completed_at is None
        assert staged.completed_step_count == 0
        assert runtime.process.get(root_pid).status is ProcessStatus.EXITED
        point = runtime.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )
        assert point is not None and point.pending_action_payload_id is not None
        wrapper = runtime.task_runs._decode_pending_resume_payload(point)
        assert (wrapper["kind"], wrapper["state"]) == (
            "completed_outcome",
            "staged",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        recovered = reopened.task_runs.get(created.run_id)
        point = reopened.store.get_task_run_resume_point(
            root_pid,
            complete_only=True,
        )

        assert recovered.status is TaskRunStatus.SUCCEEDED
        assert recovered.blockers == ()
        assert recovered.completed_step_count == recovered.step_count
        assert recovered.satisfied_requirement_count == 1
        assert point is not None and point.pending_action_payload_id is None
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 1
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_pause_persists_generation_then_drains_inflight_provider_without_tool_dispatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pause-inflight-drain.sqlite"
    runtime = Runtime.open(database, config=_config())
    client = _BlockingExitClient()
    try:
        created = _create(
            runtime,
            "pause-inflight-drain",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=3) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:pause-drain",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            before_lease = runtime.process.get(root_pid)
            assert before_lease.status is ProcessStatus.RUNNING

            pause_future = pool.submit(
                runtime.task_runs.pause,
                created.run_id,
                expected_revision=running.revision,
                command_id="pause:drain",
            )
            deadline = time.monotonic() + THREAD_SYNC_TIMEOUT_S
            persisted = runtime.store.get_task_run(created.run_id)
            while (
                persisted is not None
                and persisted.status is not TaskRunStatus.PAUSED
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                persisted = runtime.store.get_task_run(created.run_id)
            assert persisted is not None
            assert persisted.status is TaskRunStatus.PAUSED
            assert persisted.pause_generation == 1
            assert not pause_future.done()
            draining = runtime.process.get(root_pid)
            assert draining.status is ProcessStatus.RUNNING
            assert draining.execution_lease_id == before_lease.execution_lease_id
            replay_future = pool.submit(
                runtime.task_runs.pause,
                created.run_id,
                expected_revision=running.revision,
                command_id="pause:drain",
            )
            time.sleep(0.05)
            assert not replay_future.done()

            client.release.set()
            paused = pause_future.result(timeout=THREAD_SYNC_TIMEOUT_S)
            replayed = replay_future.result(timeout=THREAD_SYNC_TIMEOUT_S)
            worker_result = run_future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        assert paused.status is TaskRunStatus.PAUSED
        assert replayed == paused
        assert worker_result.status is TaskRunStatus.PAUSED
        process = runtime.process.get(root_pid)
        assert process.status is ProcessStatus.PAUSED
        assert process.resource_usage.tool_calls == 0
        point = runtime.store.get_task_run_resume_point(root_pid, complete_only=True)
        assert point is not None and point.pending_action_payload_id is not None
        pending = runtime.store.get_task_run_payload(point.pending_action_payload_id)
        assert pending is not None and pending.canonical_json is not None
        assert json.loads(pending.canonical_json)["state"] == "validated"
        assert runtime.store.list_external_effects(pid=root_pid)
    finally:
        client.release.set()
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.PAUSED
        assert reopened.process.get(root_pid).status is ProcessStatus.PAUSED
        assert reopened.process.get(root_pid).resource_usage.tool_calls == 0
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_concurrent_and_repeated_evidence_projection_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "projection-race.sqlite", config=_config())
    projection_barrier = threading.Barrier(2)
    original_tree_processes = runtime.task_runs._tree_processes

    def synchronized_tree_processes(run_id: str) -> list[Any]:
        processes = original_tree_processes(run_id)
        projection_barrier.wait(timeout=THREAD_SYNC_TIMEOUT_S)
        return processes

    try:
        created = _create(
            runtime,
            "projection-race",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        monkeypatch.setattr(
            runtime.task_runs,
            "_tree_processes",
            synchronized_tree_processes,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(runtime.task_runs.list_ledger, created.run_id, limit=100)
                for _ in range(2)
            ]
            pages = [future.result(timeout=THREAD_SYNC_TIMEOUT_S) for future in futures]
        assert pages[0].records == pages[1].records

        monkeypatch.setattr(
            runtime.task_runs,
            "_tree_processes",
            original_tree_processes,
        )
        links = runtime.store.list_task_run_links(created.run_id)
        ledger = runtime.store.list_task_run_ledger(
            created.run_id,
            after=None,
            limit=100,
        ).records
        operation_ids = {
            operation.operation_id
            for operation in runtime.uow.evidence.list_operations(
                pid=root_pid,
                limit=100,
            )
        }
        projected_operation_links = [
            link
            for link in links
            if link.evidence_type == "operation" and link.role == "operation"
        ]
        assert operation_ids
        assert {link.evidence_id for link in projected_operation_links} == operation_ids
        assert len(projected_operation_links) == len(operation_ids)
        assert len(
            [
                item
                for item in ledger
                if item.label == "TaskRun operation evidence"
                and item.operation_id in operation_ids
            ]
        ) == len(operation_ids)
        assert {
            link.role
            for link in links
            if link.evidence_type == "process" and link.evidence_id == root_pid
        } == {"process", "result"}

        runtime.task_runs.list_ledger(created.run_id, limit=100)
        assert runtime.store.list_task_run_links(created.run_id) == links
        assert runtime.store.list_task_run_ledger(
            created.run_id,
            after=None,
            limit=100,
        ).records == ledger
    finally:
        runtime.close()


@pytest.mark.parametrize("control", ["pause", "interrupt"])
def test_control_generation_is_checked_before_each_tool_in_a_batch(
    control: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config()
    runtime = Runtime.open(
        tmp_path / f"batch-{control}.sqlite",
        config=replace(
            base,
            llm=replace(base.llm, parallel_tool_calls=True),
        ),
    )
    client = _TwoToolClient()
    first_entered = threading.Event()
    release_first = threading.Event()
    tool_calls: list[str] = []
    original_acall = runtime.tools.acall

    async def controlled_acall(
        pid: str,
        tool: Any,
        args: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> Any:
        handle = runtime.tools.resolve(tool, pid=pid)
        tool_calls.append(handle.name)
        if len(tool_calls) == 1:
            first_entered.set()
            if not release_first.wait(timeout=BATCH_CONTROL_SYNC_TIMEOUT_S):
                raise AssertionError("test did not release the first tool")
        return await original_acall(
            pid,
            tool,
            args,
            context_metadata=context_metadata,
        )

    monkeypatch.setattr(runtime.tools, "acall", controlled_acall)
    try:
        created = _create(
            runtime,
            f"batch-{control}",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=2) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id=f"run:batch-{control}",
                max_quanta=1,
            )
            assert first_entered.wait(timeout=BATCH_CONTROL_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            if control == "pause":
                control_future = pool.submit(
                    runtime.task_runs.pause,
                    created.run_id,
                    expected_revision=running.revision,
                    command_id="pause:batch",
                )
            else:
                control_future = pool.submit(
                    runtime.task_runs.follow_up,
                    created.run_id,
                    {"requirement": "replace the remaining old batch action"},
                    kind="interrupt",
                    expected_revision=running.revision,
                    command_id="interrupt:batch",
                )
            deadline = time.monotonic() + BATCH_CONTROL_SYNC_TIMEOUT_S
            intent = runtime.store.get_task_run(created.run_id)
            while (
                intent is not None
                and intent.pause_generation == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                intent = runtime.store.get_task_run(created.run_id)
            assert intent is not None
            assert intent.status is TaskRunStatus.PAUSED
            assert intent.pause_generation == 1
            assert not control_future.done()

            release_first.set()
            controlled = control_future.result(timeout=BATCH_CONTROL_SYNC_TIMEOUT_S)
            worker = run_future.result(timeout=BATCH_CONTROL_SYNC_TIMEOUT_S)

        assert client.calls == 1
        assert tool_calls == ["discover_skills"]
        assert runtime.process.get(root_pid).resource_usage.tool_calls == 1
        point = runtime.store.get_task_run_resume_point(root_pid, complete_only=True)
        assert point is not None and point.pending_action_payload_id is None
        if control == "pause":
            assert controlled.status is TaskRunStatus.PAUSED
            assert worker.status is TaskRunStatus.PAUSED
        else:
            assert controlled.status is TaskRunStatus.RUNNING
            assert controlled.requirement_count == 2
            assert worker.status is not TaskRunStatus.SUCCEEDED
            assert len(runtime.store.list_process_messages(root_pid)) == 1
            links = runtime.store.list_task_run_links(created.run_id)
            natural_keys = [
                (
                    link.evidence_type,
                    link.evidence_id,
                    link.role,
                )
                for link in links
            ]
            assert len(natural_keys) == len(set(natural_keys))
            operation_links = [
                link
                for link in links
                if link.evidence_type == "operation" and link.role == "operation"
            ]
            projected_operations = [
                item
                for item in runtime.store.list_task_run_ledger(
                    created.run_id,
                    after=None,
                    limit=100,
                ).records
                if item.label == "TaskRun operation evidence"
            ]
            assert operation_links
            assert {link.ledger_seq for link in operation_links} == {
                item.seq for item in projected_operations
            }
            assert len(operation_links) == len(projected_operations)
    finally:
        release_first.set()
        runtime.close()


def test_interrupt_generation_fences_old_run_command_before_scope_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "interrupt-old-run.sqlite", config=_config())
    client = _CountingExplodingClient()
    dispatch_ready = threading.Event()
    release_old_dispatch = threading.Event()
    original_dispatch = runtime.task_runs._dispatch

    @contextmanager
    def delayed_dispatch(run_id: str, *, pause_generation: int):
        dispatch_ready.set()
        if not release_old_dispatch.wait(timeout=THREAD_SYNC_TIMEOUT_S):
            raise AssertionError("test did not release the old run command")
        with original_dispatch(
            run_id,
            pause_generation=pause_generation,
        ) as admitted:
            yield admitted

    monkeypatch.setattr(runtime.task_runs, "_dispatch", delayed_dispatch)
    try:
        created = _create(
            runtime,
            "interrupt-old-run",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=1) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:old-generation",
                max_quanta=1,
            )
            assert dispatch_ready.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            interrupted = runtime.task_runs.follow_up(
                created.run_id,
                {"requirement": "new generation requirement"},
                kind="interrupt",
                expected_revision=running.revision,
                command_id="interrupt:before-scope",
            )
            assert interrupted.status is TaskRunStatus.RUNNING
            assert interrupted.requirement_count == 2
            persisted = runtime.store.get_task_run(created.run_id)
            assert persisted is not None and persisted.pause_generation == 1

            release_old_dispatch.set()
            worker = run_future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        assert worker.status is TaskRunStatus.RUNNING
        assert client.calls == 0
        assert runtime.process.get(root_pid).resource_usage.llm_calls == 0
        assert runtime.process.get(root_pid).status is ProcessStatus.RUNNABLE
    finally:
        release_old_dispatch.set()
        runtime.close()


@pytest.mark.parametrize("drift", ["revoked", "expired"])
def test_task_run_reopen_blocks_revoked_or_expired_authority_before_dispatch(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / f"authority-{drift}.sqlite"
    first = Runtime.open(database, config=_config())
    try:
        seed_pid = first.process.spawn(
            image="base-agent:v0",
            goal="authority template",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "human:owner",
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )
        template = first.authority_manifests.get_for_process(seed_pid)
        assert template is not None
        created = _create(
            first,
            f"authority-{drift}",
            authority_manifest_id=template.manifest_id,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        if drift == "revoked":
            matching = [
                capability
                for capability in first.capability.capabilities_for(root_pid)
                if capability.resource == "human:owner"
                and CapabilityRight.WRITE.value in capability.rights
            ]
            assert matching
            for capability in matching:
                first.capability.revoke(
                    capability.cap_id,
                    revoked_by="test",
                    reason="reopen authority regression",
                    require_authority=False,
                )
    finally:
        first.close()

    if drift == "expired":
        import agent_libos.runtime.authority_manifest_manager as authority_module

        class _FutureDateTime(datetime):
            @classmethod
            def now(cls, tz: timezone | None = None) -> _FutureDateTime:
                value = cls(2100, 1, 1, tzinfo=timezone.utc)
                return value if tz is not None else value.replace(tzinfo=None)

        monkeypatch.setattr(authority_module, "datetime", _FutureDateTime)

    reopened = Runtime.open(database, config=_config())
    try:
        summary = reopened.task_runs.get(created.run_id)
        assert summary.status is TaskRunStatus.NEEDS_ATTENTION
        assert "authority_revoked" in {item["kind"] for item in summary.blockers}
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
        assert reopened.run_process_once(root_pid)["skipped"] is True
        assert reopened.process.get(root_pid).resource_usage.llm_calls == 0
    finally:
        reopened.close()


def test_direct_run_process_once_cannot_bypass_task_run_admission(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "direct-bypass.sqlite", config=_config())
    try:
        created = _create(runtime, "direct-bypass")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = _ExplodingClient()

        skipped = runtime.run_process_once(root_pid)

        assert skipped == {
            "ok": False,
            "skipped": True,
            "status": ProcessStatus.RUNNABLE.value,
        }
        assert runtime.process.get(root_pid).resource_usage.llm_calls == 0
        assert runtime.task_runs.get(created.run_id).status is TaskRunStatus.QUEUED
    finally:
        runtime.close()


def test_follow_up_interrupt_during_inflight_llm_remains_unsatisfied(
    tmp_path: Path,
) -> None:
    database = tmp_path / "follow-up-inflight.sqlite"
    runtime = Runtime.open(database, config=_config())
    client = _BlockingExitClient()
    try:
        created = _create(
            runtime,
            "follow-up-inflight",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=2) as pool:
            run_future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:before-follow-up",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            follow_future = pool.submit(
                runtime.task_runs.follow_up,
                created.run_id,
                {"requirement": "include the newly arrived constraint"},
                kind="interrupt",
                required=True,
                expected_revision=running.revision,
                command_id="follow-up:interrupt",
            )
            deadline = time.monotonic() + THREAD_SYNC_TIMEOUT_S
            intent = runtime.store.get_task_run(created.run_id)
            while (
                intent is not None
                and intent.pause_generation == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                intent = runtime.store.get_task_run(created.run_id)
            assert intent is not None
            assert intent.status is TaskRunStatus.PAUSED
            assert intent.pause_generation == 1
            assert intent.requirement_count == 2
            assert not follow_future.done()
            durable_messages = runtime.store.list_process_messages(root_pid)
            assert len(durable_messages) == 1
            assert durable_messages[0].kind is ProcessMessageKind.INTERRUPT
            assert "include the normal follow-up constraint" not in json.dumps(
                client.messages,
                sort_keys=True,
            )
            client.release.set()
            followed = follow_future.result(timeout=THREAD_SYNC_TIMEOUT_S)
            worker_result = run_future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        follow_up = requirements[-1]
        messages = runtime.store.list_process_messages(root_pid)
        assert follow_up.status is not TaskRunRequirementStatus.SATISFIED
        assert followed.requirement_count == 2
        assert followed.status is TaskRunStatus.RUNNING
        assert worker_result.status is not TaskRunStatus.SUCCEEDED
        final = runtime.task_runs.get(created.run_id)
        assert final.status is TaskRunStatus.RUNNING
        assert final.satisfied_requirement_count < 2
        assert len(messages) == 1
        assert messages[0].kind is ProcessMessageKind.INTERRUPT
        assert messages[0].correlation_id == follow_up.requirement_id
        assert client.calls == 1
        assert runtime.process.get(root_pid).resource_usage.tool_calls == 0
        point = runtime.store.get_task_run_resume_point(root_pid, complete_only=True)
        assert point is not None and point.pending_action_payload_id is None
        replayed = runtime.task_runs.follow_up(
            created.run_id,
            {"requirement": "include the newly arrived constraint"},
            kind="interrupt",
            required=True,
            expected_revision=running.revision,
            command_id="follow-up:interrupt",
        )
        assert replayed == followed
        assert len(runtime.store.list_task_run_requirements(created.run_id)) == 2
    finally:
        client.release.set()
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        reopened.llm.client = _ExplodingClient()
        recovered = reopened.task_runs.get(created.run_id)
        assert recovered.status is TaskRunStatus.RUNNING
        assert recovered.requirement_count == 2
        assert len(reopened.store.list_process_messages(root_pid)) == 1
        assert reopened.process.get(root_pid).resource_usage.tool_calls == 0
        assert reopened.run_next_process_once() is None
    finally:
        reopened.close()


def test_normal_follow_up_during_inflight_llm_cannot_be_satisfied_by_old_exit(
    tmp_path: Path,
) -> None:
    """A response cannot satisfy a requirement absent from its frozen prompt."""

    runtime = Runtime.open(tmp_path / "normal-follow-up-inflight.sqlite", config=_config())
    client = _BlockingExitClient()
    try:
        created = _create(
            runtime,
            "normal-follow-up-inflight",
            retention=TaskRunRetention.PERMANENT,
        )
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.llm.client = client
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                runtime.task_runs.run_until_blocked,
                created.run_id,
                expected_revision=created.revision,
                command_id="run:before-normal-follow-up",
                max_quanta=1,
            )
            assert client.entered.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            running = runtime.task_runs.get(created.run_id)
            followed = runtime.task_runs.follow_up(
                created.run_id,
                {"requirement": "include the normal follow-up constraint"},
                kind="normal",
                required=True,
                expected_revision=running.revision,
                command_id="follow-up:normal-inflight",
            )
            assert followed.requirement_count == 2
            client.release.set()
            worker_result = future.result(timeout=THREAD_SYNC_TIMEOUT_S)

        requirements = runtime.store.list_task_run_requirements(created.run_id)
        follow_up = requirements[-1]
        messages = runtime.store.list_process_messages(root_pid)
        final = runtime.task_runs.get(created.run_id)
        assert follow_up.status is not TaskRunRequirementStatus.SATISFIED
        assert worker_result.status is not TaskRunStatus.SUCCEEDED
        assert final.status is not TaskRunStatus.SUCCEEDED
        assert final.satisfied_requirement_count < final.requirement_count
        assert len(messages) == 1
        assert messages[0].kind is ProcessMessageKind.NORMAL
        assert messages[0].correlation_id == follow_up.requirement_id
        assert client.calls == 1
    finally:
        client.release.set()
        runtime.close()


def test_ambiguous_command_commit_replays_original_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "ambiguous-command.sqlite", config=_config())
    try:
        created = _create(runtime, "ambiguous-command")
        original_notify = runtime.task_runs._notify_updated

        def lose_response_after_commit() -> None:
            original_notify()
            raise ConnectionError("simulated response loss after durable commit")

        monkeypatch.setattr(
            runtime.task_runs,
            "_notify_updated",
            lose_response_after_commit,
        )
        with pytest.raises(ConnectionError, match="response loss"):
            runtime.task_runs.pause(
                created.run_id,
                expected_revision=created.revision,
                command_id="pause:ambiguous",
            )
        monkeypatch.setattr(runtime.task_runs, "_notify_updated", original_notify)

        replayed = runtime.task_runs.pause(
            created.run_id,
            expected_revision=created.revision,
            command_id="pause:ambiguous",
        )
        persisted = runtime.store.get_task_run(created.run_id)
        command = runtime.store.get_task_run_command(
            created.run_id,
            "pause:ambiguous",
        )
        assert persisted is not None and command is not None
        assert replayed.status is TaskRunStatus.PAUSED
        assert replayed.revision == command.result_revision == persisted.revision
        assert persisted.pause_generation == 1
        assert sum(
            item.label == "pause intent persisted"
            for item in runtime.task_runs.list_ledger(
                created.run_id,
                limit=100,
            ).records
        ) == 1
        with pytest.raises((TaskRunRevisionConflict, ValidationError)):
            runtime.task_runs.cancel(
                created.run_id,
                expected_revision=persisted.revision,
                command_id="pause:ambiguous",
            )
    finally:
        runtime.close()


def test_permanent_run_explicit_purge_is_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "explicit-purge.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = _create(
            runtime,
            "explicit-purge",
            retention=TaskRunRetention.PERMANENT,
        )
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel:explicit-purge",
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        assert any(
            payload.retention_state is TaskRunPayloadRetention.PLAINTEXT
            for payload in runtime.store.list_task_run_payloads(created.run_id)
        )

        purged = runtime.task_runs.purge_payloads(
            created.run_id,
            expected_revision=terminal.revision,
            command_id="purge:explicit",
        )
        replayed = runtime.task_runs.purge_payloads(
            created.run_id,
            expected_revision=terminal.revision,
            command_id="purge:explicit",
        )
        record = runtime.store.get_task_run(created.run_id)
        assert replayed == purged
        assert purged.payloads_purged is True
        assert record is not None and record.payloads_purged_at is not None
        assert all(
            payload.retention_state is TaskRunPayloadRetention.HASH_ONLY
            and payload.canonical_json is None
            and payload.purged_at is not None
            for payload in runtime.store.list_task_run_payloads(created.run_id)
        )
        audits = [
            item
            for item in runtime.audit.trace()
            if item.action == "task_run.payloads.purge"
            and item.target == f"task_run:{created.run_id}"
        ]
        assert len(audits) == 1
        links = runtime.store.list_task_run_links(created.run_id)
        assert any(
            link.evidence_type == "audit"
            and link.evidence_id == audits[0].record_id
            and link.role == "purge"
            for link in links
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        record = reopened.store.get_task_run(created.run_id)
        assert record is not None and record.payloads_purged_at is not None
        assert len(
            [
                item
                for item in reopened.audit.trace()
                if item.action == "task_run.payloads.purge"
                and item.target == f"task_run:{created.run_id}"
            ]
        ) == 1
    finally:
        reopened.close()
