from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ResourceReservation,
    ResourceUsage,
    TaskRunStatus,
)
from agent_libos.utils.ids import utc_now


def _config():
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )


class _ExitClient:
    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "terminal-convergence-exit",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }
            ],
        )


def _create(runtime: Runtime, title: str):
    return runtime.task_runs.create(
        TaskRunSpecV1(
            goal={"objective": title},
            display_title=title,
            image_id="base-agent:v0",
        ),
        client_request_id=f"create:{title}",
    )


def _run_to_terminal_boundary(runtime: Runtime, created: Any):
    runtime.llm.client = _ExitClient()
    summary = runtime.task_runs.run_until_blocked(
        created.run_id,
        expected_revision=created.revision,
        command_id=f"run:{created.run_id}",
        max_quanta=1,
    )
    record = runtime.store.get_task_run(created.run_id)
    assert record is not None
    return summary, record


def _assert_unsettled(
    runtime: Runtime,
    created: Any,
    *,
    blocker_kind: str,
) -> None:
    summary, record = _run_to_terminal_boundary(runtime, created)

    assert summary.status is TaskRunStatus.NEEDS_ATTENTION
    assert {item["kind"] for item in summary.blockers} == {blocker_kind}
    assert record.completed_at is None
    assert record.finalized_at is None
    assert record.payloads_purged_at is None


@pytest.mark.parametrize("publication_state", ["planning", "committed"])
def test_task_run_refuses_terminal_completion_for_unsettled_publication(
    tmp_path: Path,
    publication_state: str,
) -> None:
    runtime = Runtime.open(
        tmp_path / f"publication-{publication_state}.sqlite",
        config=_config(),
    )
    try:
        created = _create(runtime, f"publication-{publication_state}")
        root_pid = created.root_pid
        assert root_pid is not None
        publication_id = f"publication-terminal-convergence-{publication_state}"
        runtime.store.insert_runtime_publication(
            publication_id=publication_id,
            kind="process_exec",
            pid=root_pid,
            owner_instance_id=runtime.instance_id,
            plan={"operation_id": f"operation-{publication_state}"},
        )
        if publication_state == "committed":
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="committed",
                phase="committed",
                expected_states={"planning"},
            )
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == publication_state
        assert publication["operation_reconciled"] is False

        _assert_unsettled(
            runtime,
            created,
            blocker_kind="publication_unsettled",
        )
    finally:
        runtime.close()


def test_task_run_refuses_terminal_completion_for_active_resource_usage_reservation(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "resource-usage.sqlite", config=_config())
    try:
        created = _create(runtime, "resource-usage")
        root_pid = created.root_pid
        assert root_pid is not None
        runtime.uow.resources.insert_resource_usage_reservation(
            reservation_id="usage-terminal-convergence",
            pid=root_pid,
            usage=ResourceUsage(external_write_bytes=1),
            reserved_by="terminal-convergence-test",
            reason="prove TaskRun terminal convergence",
            created_at=utc_now(),
        )

        _assert_unsettled(
            runtime,
            created,
            blocker_kind="reservation_unsettled",
        )
    finally:
        runtime.close()


def test_task_run_refuses_terminal_completion_for_finite_capability_reservation(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "capability.sqlite", config=_config())
    try:
        created = _create(runtime, "capability")
        root_pid = created.root_pid
        assert root_pid is not None
        capability = runtime.capability.issue_trusted(
            root_pid,
            "custom:terminal-convergence",
            [CapabilityRight.WRITE],
            issued_by="test",
            uses_remaining=1,
        )
        runtime.capability.reserve_use(
            capability.cap_id,
            reserved_by="terminal-convergence-test",
            reason="prove TaskRun terminal convergence",
        )

        _assert_unsettled(
            runtime,
            created,
            blocker_kind="reservation_unsettled",
        )
    finally:
        runtime.close()


def test_task_run_refuses_terminal_completion_for_process_resource_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "process-resource.sqlite", config=_config())
    try:
        created = _create(runtime, "process-resource")
        root_pid = created.root_pid
        assert root_pid is not None
        original_exit: Callable[..., Any] = runtime.process.exit

        def exit_then_reserve(*args: Any, **kwargs: Any) -> Any:
            result = original_exit(*args, **kwargs)
            now = utc_now()
            runtime.uow.resources.upsert_resource_reservation(
                ResourceReservation(
                    parent_pid=root_pid,
                    child_pid="pid-unsettled-child-budget",
                    reserved={"max_tool_calls": 1.0},
                    created_at=now,
                    updated_at=now,
                )
            )
            return result

        monkeypatch.setattr(runtime.process, "exit", exit_then_reserve)

        _assert_unsettled(
            runtime,
            created,
            blocker_kind="reservation_unsettled",
        )
    finally:
        runtime.close()
