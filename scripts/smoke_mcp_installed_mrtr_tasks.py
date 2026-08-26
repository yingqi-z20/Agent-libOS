from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import agent_libos
from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp import (
    InMemoryMcpCredentialBroker,
    MCP_TASKS_EXTENSION_ID,
    McpInputRequired,
    McpRemoteTask,
    McpServerManifestV3,
    McpTasksExtensionSpec,
)
from agent_libos.models import (
    CapabilityRight,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.substrate import LocalResourceProviderSubstrate


_TASKS_SPEC_SHA256 = "b" * 64
_PRIVATE_REQUEST_STATE = "installed-private-continuation-state"
_PRIVATE_INPUT_TASK_ID = "installed-private-input-task"
_PRIVATE_CANCEL_TASK_ID = "installed-private-cancel-task"
_PRIVATE_VALUES = (
    _PRIVATE_REQUEST_STATE,
    _PRIVATE_INPUT_TASK_ID,
    _PRIVATE_CANCEL_TASK_ID,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _InitialProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, task_created_at: dict[str, str]) -> None:
        self.result_adapter: Any | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.task_created_at = task_created_at

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpInputRequired | McpRemoteTask:
        if sensitive_values:
            raise RuntimeError("credential-free installed fixture received a secret")
        if self.result_adapter is None:
            raise RuntimeError("installed MCP result adapter was not bound")
        self.calls.append((tool_id, dict(arguments)))
        if tool_id == "review":
            raw = {
                "resultType": "input_required",
                "requestState": _PRIVATE_REQUEST_STATE,
                "inputRequests": {
                    "remote-review": {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Approve the installed artifact review?",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {"approved": {"type": "boolean"}},
                                "required": ["approved"],
                            },
                        },
                    }
                },
            }
        elif tool_id == "begin-task":
            mode = arguments.get("mode")
            if mode not in {"input", "cancel"}:
                raise RuntimeError("installed MCP Task mode is invalid")
            now = _utc_now()
            task_id = (
                _PRIVATE_INPUT_TASK_ID if mode == "input" else _PRIVATE_CANCEL_TASK_ID
            )
            self.task_created_at[task_id] = now
            raw = {
                "resultType": "task",
                "taskId": task_id,
                "status": "working",
                "createdAt": now,
                "lastUpdatedAt": now,
                "ttlMs": 60_000,
                "pollIntervalMs": 1,
            }
        else:
            raise RuntimeError(f"unexpected installed MCP tool: {tool_id}")
        return self.result_adapter.tool_result(
            raw,
            server_id=manifest.server_id,
            logical_id=tool_id,
            deadline=deadline,
        )


class _ContinuationProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self) -> None:
        self.calls = 0

    async def continue_tool(
        self,
        server: Any,
        mcp_name: str,
        arguments: dict[str, Any],
        input_responses: dict[str, Any],
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        self.calls += 1
        if deadline <= 0:
            raise RuntimeError("installed MCP continuation deadline is invalid")
        if server.server_id != "installed-durable" or mcp_name != "artifact.review":
            raise RuntimeError("installed MCP continuation binding changed")
        if arguments != {"document": "release-notes"}:
            raise RuntimeError("installed MCP continuation arguments changed")
        if request_state != _PRIVATE_REQUEST_STATE:
            raise RuntimeError("installed MCP private continuation state changed")
        if input_responses != {
            "remote-review": {
                "action": "accept",
                "content": {"approved": True},
            }
        }:
            raise RuntimeError("installed MCP continuation answer changed")
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": "installed review approved once"}],
        }

    async def continue_resource(
        self,
        server: Any,
        resource_name: str,
        logical_id: str,
        input_responses: dict[str, Any],
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del server, resource_name, logical_id, input_responses, request_state, deadline
        raise RuntimeError("installed fixture has no Resource continuation")

    async def continue_prompt(
        self,
        server: Any,
        prompt_name: str,
        logical_id: str,
        arguments: dict[str, str],
        input_responses: dict[str, Any],
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del (
            server,
            prompt_name,
            logical_id,
            arguments,
            input_responses,
            request_state,
            deadline,
        )
        raise RuntimeError("installed fixture has no Prompt continuation")


class _TasksProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, task_created_at: dict[str, str]) -> None:
        self.status = {
            _PRIVATE_INPUT_TASK_ID: "working",
            _PRIVATE_CANCEL_TASK_ID: "working",
        }
        self.task_created_at = task_created_at
        self.get_calls = 0
        self.update_calls = 0
        self.cancel_calls = 0

    def _result(self, task_id: str, status: str) -> dict[str, Any]:
        now = _utc_now()
        result: dict[str, Any] = {
            "resultType": "complete",
            "taskId": task_id,
            "status": status,
            "createdAt": self.task_created_at[task_id],
            "lastUpdatedAt": now,
            "ttlMs": 60_000,
            "pollIntervalMs": 1,
        }
        if status == "input_required":
            result["inputRequests"] = {
                "remote-task-input": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": "Approve the installed remote Task?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            }
        elif status == "completed":
            result["result"] = {"approved": True}
        return result

    async def get_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del server
        if deadline <= 0:
            raise RuntimeError("installed MCP Task deadline is invalid")
        self.get_calls += 1
        status = self.status[remote_task_id]
        if remote_task_id == _PRIVATE_INPUT_TASK_ID and status == "working":
            status = self.status[remote_task_id] = "input_required"
        elif remote_task_id == _PRIVATE_CANCEL_TASK_ID and status == "cancel_requested":
            status = self.status[remote_task_id] = "cancelled"
        return self._result(remote_task_id, status)

    async def update_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        response: dict[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del server
        if deadline <= 0:
            raise RuntimeError("installed MCP Task deadline is invalid")
        self.update_calls += 1
        if response != {
            "remote-task-input": {
                "action": "accept",
                "content": {"approved": True},
            }
        }:
            raise RuntimeError("installed MCP Task answer changed")
        self.status[remote_task_id] = "completed"
        return {"resultType": "complete"}

    async def cancel_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del server
        if deadline <= 0:
            raise RuntimeError("installed MCP Task deadline is invalid")
        self.cancel_calls += 1
        self.status[remote_task_id] = "cancel_requested"
        return {"resultType": "complete"}


class _RuntimeLease:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def shutdown(self, **_kwargs: Any) -> dict[str, bool]:
        return {"ok": True}


def _runtime_factory(runtime: Runtime) -> type[Any]:
    class RuntimeFactory:
        @staticmethod
        def open(*_args: Any, **_kwargs: Any) -> _RuntimeLease:
            return _RuntimeLease(runtime)

    return RuntimeFactory


def _run_cli(runtime: Runtime, *arguments: str) -> dict[str, Any]:
    cli_module = importlib.import_module("agent_libos.api.cli")
    stdout = io.StringIO()
    stderr = io.StringIO()
    failure: SystemExit | None = None
    with (
        patch.object(cli_module, "Runtime", _runtime_factory(runtime)),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            cli_module.cli(["--db", ":memory:", "mcp", *arguments])
        except SystemExit as error:
            failure = error
    output = stdout.getvalue()
    errors = stderr.getvalue()
    if failure is not None:
        raise RuntimeError(
            f"installed MCP CLI failed with exit={failure.code}: {output}{errors}"
        )
    if errors:
        raise RuntimeError(f"installed MCP CLI wrote stderr: {errors}")
    if any(private in output for private in _PRIVATE_VALUES):
        raise RuntimeError("installed MCP CLI exposed private Provider state")
    try:
        selected = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("installed MCP CLI did not emit one JSON result") from error
    if not isinstance(selected, dict):
        raise RuntimeError("installed MCP CLI result is not an object")
    return selected


def _manifest() -> McpServerManifestV3:
    base_tool = McpToolSpec(
        tool_id="review",
        mcp_name="artifact.review",
        right="read",
        rollback_class="no_rollback_required",
        rollback_status="not_required",
        state_mutation=False,
        information_flow=True,
        input_schema={
            "type": "object",
            "properties": {"document": {"type": "string"}},
            "required": ["document"],
            "additionalProperties": False,
        },
    )
    return McpServerManifestV3(
        schema_version=3,
        server_id="installed-durable",
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        timeout_s=5.0,
        max_request_bytes=65_536,
        max_response_bytes=65_536,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        tools=(
            base_tool,
            replace(
                base_tool,
                tool_id="begin-task",
                mcp_name="artifact.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["input", "cancel"]}
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=_TASKS_SPEC_SHA256,
        ),
    )


def _substrate(
    root: Path,
    broker: InMemoryMcpCredentialBroker,
    initial: _InitialProvider,
    continuation: _ContinuationProvider,
    tasks: _TasksProvider,
) -> LocalResourceProviderSubstrate:
    substrate = LocalResourceProviderSubstrate(root)
    substrate.mcp_credential_broker = broker
    substrate.mcp_v3_tool_provider = initial
    substrate.mcp_continuation_provider = continuation
    substrate.mcp_tasks_provider = tasks
    return substrate


def _bind_result_adapter(runtime: Runtime, provider: _InitialProvider) -> None:
    resource_provider = getattr(runtime, "_mcp_resource_provider", None)
    adapter = getattr(resource_provider, "result_adapter", None)
    if adapter is None or not callable(getattr(adapter, "tool_result", None)):
        raise RuntimeError("installed MCP Runtime did not publish its governed result adapter")
    provider.result_adapter = adapter


def _grant_initial_authority(runtime: Runtime, pid: str) -> None:
    for resource, rights in (
        ("mcp:installed-durable:review", [CapabilityRight.READ]),
        ("mcp:installed-durable:begin-task", [CapabilityRight.READ]),
        ("mcp_server:installed-durable", [CapabilityRight.EXECUTE]),
        ("human:owner", [CapabilityRight.WRITE]),
    ):
        runtime.capability.grant(
            pid,
            resource,
            rights,
            issued_by="installed-artifact-smoke",
        )


def _assert_private_values_absent(store_root: Path) -> None:
    store_bytes = b"".join(
        path.read_bytes()
        for path in store_root.glob("durable.sqlite*")
        if path.is_file()
    )
    for value in _PRIVATE_VALUES:
        if value.encode("utf-8") in store_bytes:
            raise RuntimeError("installed MCP Store persisted private Provider state")


def _exercise_cli(
    runtime: Runtime,
    review: McpInputRequired,
    cancel_review: McpInputRequired,
    input_task: McpRemoteTask,
    cancel_task: McpRemoteTask,
) -> dict[str, Any]:
    observed = _run_cli(
        runtime,
        "continuations",
        "inspect",
        review.continuation_id,
    )
    completed_review = _run_cli(
        runtime,
        "continuations",
        "respond",
        review.continuation_id,
        "--expected-revision",
        str(observed["revision"]),
        "--human-request-id",
        str(observed["human_request_id"]),
        "--human-expected-revision",
        str(observed["human_revision"]),
        "--human-preview-sha256",
        str(observed["human_preview_sha256"]),
        "--responses-json",
        '{"input-1":{"action":"accept","content":{"approved":true}}}',
    )
    if completed_review.get("kind") != "complete":
        raise RuntimeError("installed MCP continuation did not complete")

    cancel_observed = _run_cli(
        runtime,
        "continuations",
        "inspect",
        cancel_review.continuation_id,
    )
    cancelled_review = _run_cli(
        runtime,
        "continuations",
        "cancel",
        cancel_review.continuation_id,
        "--expected-revision",
        str(cancel_observed["revision"]),
    )
    if cancelled_review.get("kind") != "complete" or cancelled_review.get("value") is not None:
        raise RuntimeError("installed MCP continuation cancel result is invalid")

    waiting = _run_cli(
        runtime,
        "remote-tasks",
        "get",
        input_task.task_ref,
        "--expected-revision",
        str(input_task.revision),
    )
    if waiting.get("status") != "input_required":
        raise RuntimeError("installed MCP Task did not request Human input")
    working = _run_cli(
        runtime,
        "remote-tasks",
        "update",
        input_task.task_ref,
        "--expected-revision",
        str(waiting["revision"]),
        "--human-request-id",
        str(waiting["human_request_id"]),
        "--human-expected-revision",
        str(waiting["human_revision"]),
        "--human-preview-sha256",
        str(waiting["human_preview_sha256"]),
        "--responses-json",
        '{"input-1":{"action":"accept","content":{"approved":true}}}',
    )
    if working.get("status") != "working":
        raise RuntimeError("installed MCP Task update acknowledgement is invalid")
    completed_task = _run_cli(
        runtime,
        "remote-tasks",
        "get",
        input_task.task_ref,
        "--expected-revision",
        str(working["revision"]),
    )
    if completed_task.get("status") != "completed" or completed_task.get("result") != {
        "approved": True
    }:
        raise RuntimeError("installed MCP Task did not complete")

    cancellation_requested = _run_cli(
        runtime,
        "remote-tasks",
        "cancel",
        cancel_task.task_ref,
        "--expected-revision",
        str(cancel_task.revision),
    )
    if cancellation_requested.get("status") != "cancel_requested":
        raise RuntimeError("installed MCP Task cancellation was reported as stopped")
    cancelled_task = _run_cli(
        runtime,
        "remote-tasks",
        "get",
        cancel_task.task_ref,
        "--expected-revision",
        str(cancellation_requested["revision"]),
    )
    if cancelled_task.get("status") != "cancelled":
        raise RuntimeError("installed MCP Task cancellation was not explicitly re-observed")
    return {
        "continuation": completed_review["kind"],
        "continuation_cancel": cancelled_review["kind"],
        "task": completed_task["status"],
        "task_cancel_ack": cancellation_requested["status"],
        "task_cancel_observed": cancelled_task["status"],
    }


def _installed_smoke() -> dict[str, Any]:
    installed_module = Path(agent_libos.__file__).resolve()
    installed_prefix = Path(sys.prefix).resolve()
    if not installed_module.is_relative_to(installed_prefix):
        raise RuntimeError(
            "installed MCP durable CLI smoke imported agent_libos outside the clean "
            f"environment: module={installed_module} prefix={installed_prefix}"
        )

    with (
        tempfile.TemporaryDirectory(
            prefix="agent-libos-installed-mrtr-workspace-"
        ) as workspace_root,
        tempfile.TemporaryDirectory(
            prefix="agent-libos-installed-mrtr-store-"
        ) as store_root,
    ):
        workspace_path = Path(workspace_root)
        store_path = Path(store_root)
        database = store_path / "durable.sqlite"
        broker = InMemoryMcpCredentialBroker()
        task_created_at: dict[str, str] = {}
        initial_provider = _InitialProvider(task_created_at)
        continuation_provider = _ContinuationProvider()
        tasks_provider = _TasksProvider(task_created_at)
        config = AgentLibOSConfig(
            mcp=replace(
                DEFAULT_CONFIG.mcp,
                tasks_extension_enabled=True,
                tasks_extension_spec_sha256=_TASKS_SPEC_SHA256,
                remote_task_poll_min_interval_s=0.000001,
            )
        )
        first = Runtime.open(
            database,
            substrate=_substrate(
                workspace_path,
                broker,
                initial_provider,
                continuation_provider,
                tasks_provider,
            ),
            config=config,
        )
        try:
            _bind_result_adapter(first, initial_provider)
            first.mcp.register_server(
                _manifest(),
                actor="installed-artifact-smoke",
                require_capability=False,
            )
            pids = tuple(
                first.process.spawn(
                    image="base-agent:v0",
                    goal=f"installed MCP durable CLI artifact smoke {index}",
                    resource_budget=ResourceBudget(max_mcp_bytes=256_000),
                )
                for index in range(4)
            )
            for pid in pids:
                _grant_initial_authority(first, pid)
            review = first.mcp.call_tool(
                pids[0],
                "installed-durable",
                "review",
                {"document": "release-notes"},
            )
            cancel_review = first.mcp.call_tool(
                pids[1],
                "installed-durable",
                "review",
                {"document": "release-notes"},
            )
            input_task = first.mcp.call_tool(
                pids[2],
                "installed-durable",
                "begin-task",
                {"mode": "input"},
            )
            cancel_task = first.mcp.call_tool(
                pids[3],
                "installed-durable",
                "begin-task",
                {"mode": "cancel"},
            )
            if not isinstance(review, McpInputRequired) or not isinstance(
                cancel_review, McpInputRequired
            ):
                raise RuntimeError("installed MCP initial MRTR result is invalid")
            if not isinstance(input_task, McpRemoteTask) or not isinstance(
                cancel_task, McpRemoteTask
            ):
                raise RuntimeError("installed MCP initial Task result is invalid")
            for pid, task in zip(pids[2:], (input_task, cancel_task), strict=True):
                first.capability.grant(
                    pid,
                    f"mcp_task:{task.task_ref}",
                    [CapabilityRight.READ, CapabilityRight.WRITE],
                    issued_by="installed-artifact-smoke",
                )
        finally:
            first.close()

        _assert_private_values_absent(store_path)
        reopened = Runtime.open(
            database,
            substrate=_substrate(
                workspace_path,
                broker,
                initial_provider,
                continuation_provider,
                tasks_provider,
            ),
            config=config,
        )
        try:
            result = _exercise_cli(
                reopened,
                review,
                cancel_review,
                input_task,
                cancel_task,
            )
            if len(initial_provider.calls) != 4:
                raise RuntimeError("installed MCP original Tool call was replayed")
            if continuation_provider.calls != 1:
                raise RuntimeError("installed MCP continuation was not dispatched exactly once")
            if (
                tasks_provider.get_calls != 3
                or tasks_provider.update_calls != 1
                or tasks_provider.cancel_calls != 1
            ):
                raise RuntimeError("installed MCP Task methods were not dispatched exactly once")
            durable_projection = repr(
                reopened.uow.mcp_remote_tasks.list(limit=100)
            ) + repr(reopened.uow.mcp_continuations.list(limit=100))
            if any(private in durable_projection for private in _PRIVATE_VALUES):
                raise RuntimeError("installed MCP durable public records exposed Provider state")
        finally:
            reopened.close()
            broker.close()
        _assert_private_values_absent(store_path)
        return {
            "protocol_revision": "2026-07-28",
            "initial_tool_calls": len(initial_provider.calls),
            "continuation_dispatches": continuation_provider.calls,
            "task_get_dispatches": tasks_provider.get_calls,
            **result,
        }


def main() -> int:
    print(json.dumps({"runtime-v3-durable-cli": _installed_smoke()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
