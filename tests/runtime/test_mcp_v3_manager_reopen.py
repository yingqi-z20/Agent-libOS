from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_libos.mcp.continuations import McpContinuationManager
from agent_libos.mcp.human import HumanObjectManagerMcpBridge
from agent_libos.mcp.tasks import McpRemoteTaskManager
from agent_libos.mcp.types import McpComplete, McpRemoteTaskStatus
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.runtime import Runtime

from tests.unit.test_mcp_v3_continuations import (
    _Boundary as _ContinuationBoundary,
    _Broker,
    _binding as _continuation_binding,
    _input_required,
)
from tests.unit.test_mcp_v3_tasks import (
    _Boundary as _TaskBoundary,
    _binding as _task_binding,
    _task_result,
)


def test_explicit_continuation_and_task_operations_survive_sqlite_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mcp-v3-reopen.sqlite"
    broker = _Broker()
    initial_runtime = Runtime.open(database)
    try:
        owner_id = initial_runtime.process.spawn(
            image="base-agent:v0",
            goal="durable MCP Elicitation",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "human:owner",
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ]
            },
        )
        human = HumanObjectManagerMcpBridge(initial_runtime.human)
        continuation_manager = McpContinuationManager(
            repository=initial_runtime.uow.mcp_continuations,
            side_effects=initial_runtime.uow.mcp_side_effects,
            broker=broker,
            human_requests=human,
            boundary=_ContinuationBoundary(),
            id_factory=lambda: "continuation-reopen",
        )
        continuation = continuation_manager.capture_input_required(
            _continuation_binding(owner_id=owner_id),
            _input_required(state="PRIVATE-REQUEST-STATE"),
            expires_at=None,
        )
        task_manager = McpRemoteTaskManager(
            repository=initial_runtime.uow.mcp_remote_tasks,
            side_effects=initial_runtime.uow.mcp_side_effects,
            broker=broker,
            human_requests=human,
            boundary=_TaskBoundary(),
            id_factory=lambda: "task-reopen",
        )
        task = task_manager.capture_task(_task_binding(), _task_result())
    finally:
        initial_runtime.close()

    _assert_store_files_exclude(
        database,
        b"PRIVATE-REQUEST-STATE",
        b"remote-bearer-id",
    )

    continuation_boundary = _ContinuationBoundary(
        {"resultType": "complete", "after": "reopen"}
    )
    task_boundary = _TaskBoundary()
    task_boundary.get_results.append(
        _task_result(
            status="completed",
            statusMessage="credential-value terminal status",
            result={"after": "reopen", "secret": "credential-value"},
        )
    )
    reopened = Runtime.open(database)
    try:
        human = HumanObjectManagerMcpBridge(reopened.human)
        continuation_manager = McpContinuationManager(
            repository=reopened.uow.mcp_continuations,
            side_effects=reopened.uow.mcp_side_effects,
            broker=broker,
            human_requests=human,
            boundary=continuation_boundary,
        )
        assert continuation.human_request_id is not None
        assert continuation.human_revision is not None
        assert continuation.human_preview_sha256 is not None
        reopened_question = reopened.human.get(continuation.human_request_id)
        assert reopened_question.payload["type"] == "question"
        assert (
            reopened_question.payload["context"]["mcp_local_ref"]
            == continuation.continuation_id
        )
        human.settle_answer(
            continuation.human_request_id,
            {"input-1": {"action": "decline"}},
            expected_revision=continuation.human_revision,
            preview_sha256=continuation.human_preview_sha256,
        )
        recovered_binding = continuation_manager.binding_material(
            continuation.continuation_id
        )
        assert recovered_binding.owner_id == owner_id
        complete = asyncio.run(
            continuation_manager.respond(
                continuation.continuation_id,
                expected_revision=continuation.revision,
                binding=recovered_binding,
                human_request_id=continuation.human_request_id,
                human_expected_revision=continuation.human_revision,
                human_preview_sha256=continuation.human_preview_sha256,
                deadline=100.0,
            )
        )
        assert complete == McpComplete(value={"after": "reopen"})
        assert len(continuation_boundary.calls) == 1
        with pytest.raises(ValidationError, match="revision|terminal"):
            asyncio.run(
                continuation_manager.respond(
                    continuation.continuation_id,
                    expected_revision=continuation.revision,
                    binding=recovered_binding,
                    human_request_id=continuation.human_request_id,
                    human_expected_revision=continuation.human_revision,
                    human_preview_sha256=continuation.human_preview_sha256,
                    deadline=100.0,
                )
            )
        assert len(continuation_boundary.calls) == 1

        task_manager = McpRemoteTaskManager(
            repository=reopened.uow.mcp_remote_tasks,
            side_effects=reopened.uow.mcp_side_effects,
            broker=broker,
            human_requests=human,
            boundary=task_boundary,
            sensitive_values=("credential-value",),
        )
        completed_task = asyncio.run(
            task_manager.get(
                task.task_ref,
                expected_revision=task.revision,
                binding=_task_binding(),
                deadline=100.0,
            )
        )
        assert completed_task.status is McpRemoteTaskStatus.COMPLETED
        assert completed_task.result == {
            "after": "reopen",
            "secret": "[redacted]",
        }
        assert completed_task.status_message == "[redacted] terminal status"
        assert len(task_boundary.get_calls) == 1
        assert all(b"credential-value" not in value for value in broker.values.values())
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        b"PRIVATE-REQUEST-STATE",
        b"remote-bearer-id",
        b"credential-value",
    )


def _assert_store_files_exclude(database: Path, *secrets: bytes) -> None:
    candidates = [database, *database.parent.glob(f"{database.name}-*")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = candidate.read_bytes()
        for secret in secrets:
            assert secret not in payload, (candidate, secret)
