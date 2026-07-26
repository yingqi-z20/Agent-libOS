from __future__ import annotations

import asyncio

import pytest

from agent_libos import Runtime
from agent_libos.models import ObjectType, ProcessStatus


class TestWorkflowEntry:
    def test_workflow_exception_is_text_free_across_public_and_durable_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        secret = "SECRET /private/provider/credential"

        async def fail_acall(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(secret)

        try:
            monkeypatch.setattr(runtime.tools, "acall", fail_acall)
            result = runtime.run_workflow("get_working_directory")

            assert not result.ok
            assert result.error is not None
            assert result.error.startswith("workflow_failed: RuntimeError ")
            assert "correlation_id=corr_" in result.error
            process = runtime.process.get(result.pid)
            assert process.status is ProcessStatus.FAILED
            assert process.outcome is not None
            assert process.outcome.result_oid is not None
            assert process.status_message == f"result_oid:{process.outcome.result_oid}"
            stored_error = runtime.store.get_object(process.outcome.result_oid)
            assert stored_error is not None
            assert stored_error.payload == {"message": result.error}
            persisted = (
                f"{process!r} {stored_error!r} {runtime.audit.trace()!r} "
                f"{runtime.events.list()!r}"
            )
            assert secret not in result.error
            assert secret not in persisted
        finally:
            runtime.close()

    def test_async_workflow_lease_excludes_scheduler_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")

        async def scenario() -> None:
            entered_tool = asyncio.Event()
            release_tool = asyncio.Event()
            selected_pid: list[str] = []
            scheduler_quantum_called = False
            original_acall = runtime.tools.acall

            async def gated_acall(pid: str, tool: str, args: dict[str, object]):
                selected_pid.append(pid)
                entered_tool.set()
                await release_tool.wait()
                return await original_acall(pid, tool, args)

            def scheduler_quantum(_pid: str) -> None:
                nonlocal scheduler_quantum_called
                scheduler_quantum_called = True

            monkeypatch.setattr(runtime.tools, "acall", gated_acall)
            task = asyncio.create_task(runtime.arun_workflow("get_working_directory"))
            await asyncio.wait_for(entered_tool.wait(), timeout=2)

            pid = selected_pid[0]
            claimed = runtime.process.get(pid)
            assert claimed.status == ProcessStatus.RUNNING
            assert claimed.execution_owner_id == f"{runtime.instance_id}:workflow"
            assert claimed.execution_lease_id is not None
            assert runtime.scheduler.run_pid_once(pid, scheduler_quantum) == {
                "ok": False,
                "skipped": True,
                "status": ProcessStatus.RUNNING.value,
            }
            assert scheduler_quantum_called is False

            release_tool.set()
            result = await task
            assert result.ok, result.error
            assert runtime.process.get(pid).status == ProcessStatus.EXITED

        try:
            asyncio.run(scenario())
        finally:
            runtime.close()

    def test_cancelled_async_workflow_terminalizes_spawned_process(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")

        async def scenario() -> None:
            entered_tool = asyncio.Event()
            never_release = asyncio.Event()
            selected_pid: list[str] = []

            async def blocked_acall(
                pid: str,
                _tool: str,
                _args: dict[str, object],
            ) -> None:
                selected_pid.append(pid)
                entered_tool.set()
                await never_release.wait()

            monkeypatch.setattr(runtime.tools, "acall", blocked_acall)
            task = asyncio.create_task(runtime.arun_workflow("get_working_directory"))
            await asyncio.wait_for(entered_tool.wait(), timeout=2)

            pid = selected_pid[0]
            assert runtime.process.get(pid).status == ProcessStatus.RUNNING
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            process = runtime.process.get(pid)
            assert process.status == ProcessStatus.KILLED
            assert process.execution_owner_id is None
            assert process.execution_lease_id is None

        try:
            asyncio.run(scenario())
        finally:
            runtime.close()

    def test_default_workflow_runs_complete_table_tool_hidden_from_model_projection(
        self,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow("get_working_directory")

            assert result.ok, result.error
            assert result.image == runtime.config.runtime.default_image_id
            assert result.status == ProcessStatus.EXITED.value
            assert result.payload == {"working_directory": "."}
            assert result.result_oid is not None
            process = runtime.process.get(result.pid)
            assert process.status == ProcessStatus.EXITED
            assert process.status_message == f"result_oid:{result.result_oid}"
            assert "get_working_directory" in process.tool_table
            assert "get_working_directory" not in process.model_tool_table
            assert process.memory_view is not None
            assert result.result_oid in {handle.oid for handle in process.memory_view.roots}
            stored_result = runtime.store.get_object(result.result_oid)
            assert stored_result is not None
            assert stored_result.type == ObjectType.TOOL_RESULT
            assert any(record.action == "workflow.run" and record.target == f"process:{result.pid}" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_workflow_can_use_explicit_image_tool_table(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "parse_pytest_log",
                {"log": "FAILED tests/example_test.py::test_example\nE AssertionError: boom"},
                image=runtime.config.runtime.coding_image_id,
            )

            assert result.ok, result.error
            assert result.image == runtime.config.runtime.coding_image_id
            assert result.status == ProcessStatus.EXITED.value
            assert result.payload["failed"] == ["FAILED tests/example_test.py::test_example"]
            assert result.payload["failure_count"] == 1
            assert runtime.process.get(result.pid).image_id == runtime.config.runtime.coding_image_id
        finally:
            runtime.close()

    def test_parse_pytest_log_failure_count_uses_first_nonempty_bucket(self) -> None:
        runtime = Runtime.open("local")
        try:
            mixed = runtime.run_workflow(
                "parse_pytest_log",
                {
                    "log": "\n".join(
                        [
                            "FAILED tests/test_a.py::test_a",
                            "FAILED tests/test_b.py::test_b",
                            "AssertionError: first",
                            "AssertionError: second",
                            "AssertionError: third",
                            "E first setup error",
                            "E second setup error",
                            "E third setup error",
                            "E fourth setup error",
                        ]
                    )
                },
                image=runtime.config.runtime.coding_image_id,
            )
            assertions_only = runtime.run_workflow(
                "parse_pytest_log",
                {
                    "log": "\n".join(
                        [
                            "AssertionError: first",
                            "AssertionError: second",
                            "E setup error",
                        ]
                    )
                },
                image=runtime.config.runtime.coding_image_id,
            )

            assert mixed.ok, mixed.error
            assert len(mixed.payload["failed"]) == 2
            assert len(mixed.payload["assertions"]) == 3
            assert len(mixed.payload["errors"]) == 4
            assert mixed.payload["failure_count"] == 2
            assert assertions_only.ok, assertions_only.error
            assert len(assertions_only.payload["assertions"]) == 2
            assert len(assertions_only.payload["errors"]) == 1
            assert assertions_only.payload["failure_count"] == 2
        finally:
            runtime.close()

    def test_parse_pytest_log_error_removes_only_e_and_first_whitespace(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "parse_pytest_log",
                {"log": "   E       AssertionError: retained indentation   "},
                image=runtime.config.runtime.coding_image_id,
            )

            assert result.ok, result.error
            assert result.payload["failed"] == []
            assert result.payload["assertions"] == []
            assert result.payload["errors"] == [
                "      AssertionError: retained indentation"
            ]
            assert result.payload["failure_count"] == 1
        finally:
            runtime.close()

    def test_parse_pytest_log_rejects_unknown_fields(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "parse_pytest_log",
                {
                    "log": "FAILED tests/test_example.py::test_example",
                    "filename": "must-not-be-silently-ignored.log",
                },
                image=runtime.config.runtime.coding_image_id,
            )

            assert not result.ok
            assert result.error == "Invalid arguments for tool `parse_pytest_log`."
        finally:
            runtime.close()

    def test_unknown_workflow_tool_returns_failed_result(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow("missing_workflow_tool")

            assert not result.ok
            assert result.status == ProcessStatus.FAILED.value
            assert result.tool_id is None
            assert "not in process tool table" in (result.error or "")
            assert runtime.process.get(result.pid).status == ProcessStatus.FAILED
            assert any(record.action == "workflow.run" and record.target == f"process:{result.pid}" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_workflow_waiting_for_human_returns_request_without_auto_exit(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "ask_human",
                {"question": "Continue?"},
                authority_manifest={
                    "authorized_capabilities": [
                        {"resource": "human:owner", "rights": ["write"]}
                    ],
                    "permitted_effects": ["human.*"],
                },
            )

            assert not result.ok
            assert result.waiting_human
            assert result.request_id is not None
            process = runtime.process.get(result.pid)
            assert result.status == process.status.value
            assert process.status == ProcessStatus.WAITING_HUMAN
            assert process.status_message == f"waiting for human request {result.request_id}"
            assert runtime.human.pending()[0].request_id == result.request_id
        finally:
            runtime.close()

    def test_workflow_request_permission_waits_instead_of_exiting_with_pending_request(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "request_permission",
                {
                    "resource": "filesystem:workspace:*",
                    "rights": ["write"],
                    "reason": "edit workspace",
                },
                authority_manifest={
                    "authorized_capabilities": [
                        {"resource": "human:owner", "rights": ["write"]}
                    ],
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:*",
                                "rights": ["write"],
                            }
                        ]
                    },
                },
            )

            assert not result.ok
            assert result.waiting_human
            assert result.request_id is not None
            process = runtime.process.get(result.pid)
            assert process.status == ProcessStatus.WAITING_HUMAN
            assert process.status_message == f"waiting for human request {result.request_id}"
            assert runtime.human.pending()[0].request_id == result.request_id
        finally:
            runtime.close()

    def test_workflow_does_not_auto_exit_process_exec_tool(self) -> None:
        runtime = Runtime.open("local")
        try:
            result = runtime.run_workflow(
                "exec_process",
                {"image": runtime.config.runtime.coding_image_id, "goal": "become coding workflow"},
            )

            assert result.ok, result.error
            process = runtime.process.get(result.pid)
            assert result.status == ProcessStatus.RUNNABLE.value
            assert process.status == ProcessStatus.RUNNABLE
            assert process.image_id == runtime.config.runtime.coding_image_id
        finally:
            runtime.close()
