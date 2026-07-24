from __future__ import annotations

from agent_libos import Runtime
from agent_libos.models import ObjectType, ProcessStatus


class TestWorkflowEntry:
    def test_default_workflow_runs_visible_tool_and_exits_with_result_object(self) -> None:
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
