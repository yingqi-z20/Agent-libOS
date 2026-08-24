from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.api.cli import cli as mcp_cli
from agent_libos.mcp import (
    InMemoryMcpCredentialBroker,
    McpComplete,
    McpInputRequired,
    McpRemoteTask,
    McpServerManifestV3,
    McpTasksExtensionSpec,
)
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    McpPromptSpec,
    McpResourceSpec,
)
from agent_libos.models import (
    CapabilityRight,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    ValidationError,
)
from agent_libos.substrate import (
    LocalResourceProviderSubstrate,
    ProviderEffectNotStarted,
)
from agent_libos.utils.serde import dumps, to_jsonable


_FUNCTIONAL_PROVIDER_TIMEOUT_S = 10.0


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="durable-mrtr",
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        # These tests exercise durable protected-operation bookkeeping, not
        # latency. Leave enough budget for loaded Windows CI runners; explicit
        # deadline tests below replace this value with their measured budget.
        timeout_s=_FUNCTIONAL_PROVIDER_TIMEOUT_S,
        max_request_bytes=16_384,
        max_response_bytes=16_384,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        tools=(
            McpToolSpec(
                tool_id="review",
                mcp_name="fixture.review",
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
            ),
        ),
    )


def _resource_prompt_manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="durable-host-surfaces",
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        timeout_s=_FUNCTIONAL_PROVIDER_TIMEOUT_S,
        max_request_bytes=16_384,
        max_response_bytes=16_384,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(
                resource_id="briefing",
                remote_uri="memo://provider/briefing",
            ),
        ),
        prompts=(
            McpPromptSpec(
                prompt_id="review",
                mcp_name="fixture.review_prompt",
                argument_names=("topic",),
            ),
        ),
    )


class _InitialInputRequiredProvider:
    def __init__(self, adapter: Any, runtime: Runtime | None = None) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.calls = 0

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpInputRequired:
        assert sensitive_values == ()
        self.calls += 1
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "call_tool")
        assert tool_id == "review"
        assert arguments == {"document": "release-notes"}
        return self.adapter.tool_result(
            {
                "resultType": "input_required",
                "requestState": "broker-only-round-state",
                "inputRequests": {
                    "provider-confirmation": {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Approve this review?",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {
                                    "approved": {"type": "boolean"},
                                },
                                "required": ["approved"],
                            },
                        },
                    }
                },
            },
            server_id=manifest.server_id,
            logical_id=tool_id,
            deadline=deadline,
        )


class _ContinuationProvider:
    def __init__(self, runtime: Runtime | None = None) -> None:
        self.runtime = runtime
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
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "continuation.respond")
        assert deadline > 0
        assert server.server_id == "durable-mrtr"
        assert mcp_name == "fixture.review"
        assert arguments == {"document": "release-notes"}
        assert request_state == "broker-only-round-state"
        assert input_responses == {
            "provider-confirmation": {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": "approved exactly once"}],
        }


class _SettlementFailureContinuationProvider(_ContinuationProvider):
    def __init__(self, settlement_case: str) -> None:
        super().__init__()
        self.settlement_case = settlement_case

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
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        if self.settlement_case == "malformed":
            return {"resultType": "not-a-result"}
        return {
            "resultType": "input_required",
            "requestState": "broker-only-next-round-state",
            "inputRequests": {
                "next-confirmation": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": "Confirm another round?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            },
        }


class _MaliciousNotStartedContinuationProvider(_ContinuationProvider):
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
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        raise ProviderEffectNotStarted("custom Provider forged not-started")


class _CooperativeDeadlineProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.cancellations = 0
        self.closed = 0

    async def _hang(self) -> Any:
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        finally:
            self.closed += 1


class _HangingToolProvider(_CooperativeDeadlineProvider):
    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> Any:
        del manifest, tool_id, arguments, deadline, sensitive_values
        return await self._hang()


class _CancellationSuppressingToolProvider(_HangingToolProvider):
    """Yielding contract violation retained as a shutdown regression probe."""

    async def _hang(self) -> Any:
        self.calls += 1
        try:
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
        finally:
            self.closed += 1


class _BlockingToolProvider:
    """Contract-violating custom SPI used to lock conservative post-checks."""

    def __init__(self, *, delay_s: float) -> None:
        self.calls = 0
        self.delay_s = delay_s

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpComplete[dict[str, Any]]:
        del manifest, tool_id, arguments, deadline, sensitive_values
        self.calls += 1
        # In-process Python cannot safely preempt this.  It is forbidden by the
        # public custom-SPI contract; Runtime must still reject its late result
        # and keep the entered effect UNKNOWN/no-replay.
        time.sleep(self.delay_s)
        return McpComplete(value={"content": []})


class _HangingContinuationProvider(_CooperativeDeadlineProvider):
    async def continue_tool(
        self,
        server: Any,
        mcp_name: str,
        arguments: dict[str, Any],
        input_responses: dict[str, Any],
        request_state: str | None,
        *,
        deadline: float,
    ) -> Any:
        del server, mcp_name, arguments, input_responses, request_state, deadline
        return await self._hang()


class _ContinuationTaskProvider:
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
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        return {
            "resultType": "task",
            "taskId": "broker-only-continuation-task",
            "status": "working",
            "createdAt": "2030-01-01T00:00:00Z",
            "lastUpdatedAt": "2030-01-01T00:00:01Z",
            "ttlMs": 60_000,
            "pollIntervalMs": 1,
        }


class _HangingTasksProvider(_CooperativeDeadlineProvider):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    async def _operation(self, operation: str) -> Any:
        self.operations.append(operation)
        return await self._hang()

    async def get_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> Any:
        del server, remote_task_id, deadline
        return await self._operation("get")

    async def update_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        response: dict[str, Any],
        *,
        deadline: float,
    ) -> Any:
        del server, remote_task_id, response, deadline
        return await self._operation("update")

    async def cancel_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> Any:
        del server, remote_task_id, deadline
        return await self._operation("cancel")


class _InitialResourcePromptInputRequiredProvider:
    def __init__(self, adapter: Any, runtime: Runtime) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.calls: list[str] = []

    @staticmethod
    def _input_required(surface: str) -> dict[str, Any]:
        return {
            "resultType": "input_required",
            "requestState": f"broker-only-{surface}-round-state",
            "inputRequests": {
                f"provider-{surface}-confirmation": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": f"Approve the {surface}?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            },
        }

    async def read_resource(
        self,
        server: Any,
        resource_name: str,
        variables: Any,
        *,
        deadline: float,
    ) -> McpInputRequired:
        assert server.server_id == "durable-host-surfaces"
        assert resource_name == "memo://provider/briefing"
        assert variables is None
        _assert_pending_effect(self.runtime, "resources.read")
        self.calls.append("resource")
        return self.adapter.read_resource_result(
            self._input_required("resource"),
            server_id=server.server_id,
            logical_id="briefing",
            deadline=deadline,
        )

    async def get_prompt(
        self,
        server: Any,
        prompt_name: str,
        arguments: dict[str, str],
        *,
        deadline: float,
    ) -> McpInputRequired:
        assert server.server_id == "durable-host-surfaces"
        assert prompt_name == "fixture.review_prompt"
        assert arguments == {"topic": "MCP"}
        _assert_pending_effect(self.runtime, "prompts.get")
        self.calls.append("prompt")
        return self.adapter.prompt_result(
            self._input_required("prompt"),
            server_id=server.server_id,
            logical_id="review",
            deadline=deadline,
        )


class _ResourcePromptContinuationProvider:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.calls: list[str] = []

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
        assert server.server_id == "durable-host-surfaces"
        assert resource_name == "memo://provider/briefing"
        assert logical_id == "briefing"
        assert request_state == "broker-only-resource-round-state"
        self._assert_response(input_responses)
        _assert_pending_effect(self.runtime, "continuation.respond")
        assert deadline > 0
        self.calls.append("resource")
        return {
            "resultType": "complete",
            "contents": [
                {
                    "uri": "memo://provider/briefing",
                    "mimeType": "text/plain",
                    "text": "resource approved exactly once",
                }
            ],
        }

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
        assert server.server_id == "durable-host-surfaces"
        assert prompt_name == "fixture.review_prompt"
        assert logical_id == "review"
        assert arguments == {"topic": "MCP"}
        assert request_state == "broker-only-prompt-round-state"
        self._assert_response(input_responses)
        _assert_pending_effect(self.runtime, "continuation.respond")
        assert deadline > 0
        self.calls.append("prompt")
        return {
            "resultType": "complete",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "prompt approved exactly once",
                    },
                }
            ],
        }

    @staticmethod
    def _assert_response(input_responses: dict[str, Any]) -> None:
        assert input_responses in (
            {
                "provider-resource-confirmation": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
            {
                "provider-prompt-confirmation": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
        )


def _substrate(root: Path, broker: InMemoryMcpCredentialBroker) -> Any:
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.mcp_credential_broker = broker
    return substrate


class _InitialTaskProvider:
    def __init__(self, adapter: Any, runtime: Runtime | None = None) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.calls = 0

    async def call_tool(
        self,
        manifest: McpServerManifestV3,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpRemoteTask:
        assert sensitive_values == ()
        self.calls += 1
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "call_tool")
        mode = arguments["mode"]
        return self.adapter.tool_result(
            {
                "resultType": "task",
                "taskId": f"private-{mode}-task",
                "status": "working",
                "createdAt": "2030-01-01T00:00:00Z",
                "lastUpdatedAt": "2030-01-01T00:00:01Z",
                "ttlMs": 60_000,
                "pollIntervalMs": 1,
            },
            server_id=manifest.server_id,
            logical_id=tool_id,
            deadline=deadline,
        )


class _TasksProvider:
    def __init__(self, runtime: Runtime | None = None) -> None:
        self.runtime = runtime
        self.status = {
            "private-input-task": "working",
            "private-cancel-task": "working",
            "private-error-task": "working",
        }
        self.get_calls = 0
        self.update_calls = 0
        self.cancel_calls = 0

    @staticmethod
    def _result(task_id: str, status: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "resultType": "complete",
            "taskId": task_id,
            "status": status,
            # The remote id is a broker-only bearer.  Exercise the primitive's
            # dynamic exact-secret sanitizer on every provider-controlled
            # sibling before the durable manager or CLI can observe it.
            "statusMessage": f"provider reflected {task_id}",
            "createdAt": "2030-01-01T00:00:00Z",
            "lastUpdatedAt": (
                "2030-01-01T00:00:03Z"
                if status in {"completed", "cancelled"}
                else "2030-01-01T00:00:02Z"
            ),
            "ttlMs": 60_000,
            "pollIntervalMs": 1,
        }
        if status == "input_required":
            result["inputRequests"] = {
                "provider-task-input": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": "Approve remote Task?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            }
        elif status == "completed":
            result["result"] = {
                "approved": True,
                "providerEcho": task_id,
            }
        return result

    async def get_remote_task(
        self, server: Any, remote_task_id: str, *, deadline: float
    ) -> dict[str, Any]:
        del server, deadline
        self.get_calls += 1
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "tasks.get")
        if remote_task_id == "private-error-task":
            raise RuntimeError(f"Tasks peer reflected {remote_task_id}")
        status = self.status[remote_task_id]
        if remote_task_id == "private-input-task" and status == "working":
            status = self.status[remote_task_id] = "input_required"
        return self._result(remote_task_id, status)

    async def update_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        response: dict[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del server, deadline
        self.update_calls += 1
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "tasks.update")
        assert response == {
            "provider-task-input": {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        self.status[remote_task_id] = "completed"
        return {"resultType": "complete"}

    async def cancel_remote_task(
        self, server: Any, remote_task_id: str, *, deadline: float
    ) -> dict[str, Any]:
        del server, deadline
        self.cancel_calls += 1
        if self.runtime is not None:
            _assert_pending_effect(self.runtime, "tasks.cancel")
        self.status[remote_task_id] = "cancelled"
        return {"resultType": "complete"}


def _assert_pending_effect(runtime: Runtime, operation: str) -> None:
    effects = [
        effect
        for effect in runtime.store.list_external_effects()
        if effect.provider == "mcp" and effect.operation == operation
    ]
    assert effects
    assert effects[-1].effect_state == "pending"


def _bind_cli_to_open_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: Runtime,
) -> None:
    """Exercise the real CLI parser/dispatcher without surrendering Runtime ownership."""

    monkeypatch.setattr(
        "agent_libos.api.cli.Runtime.open",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(
        "agent_libos.api.cli._shutdown_runtime_before_exit",
        lambda _runtime: None,
    )


def _run_live_mcp_cli(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> dict[str, Any]:
    mcp_cli(["--db", ":memory:", "mcp", *args])
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert isinstance(result, dict)
    assert "private-" not in captured.out
    return result


def _run_live_mcp_cli_failure(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> dict[str, Any]:
    with pytest.raises(SystemExit) as raised:
        mcp_cli(["--db", ":memory:", "mcp", *args])
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["schema_version"] == 1
    assert result["error"]["type"] == "ValidationError"
    assert "private-" not in captured.out
    return result


def _approved_input_responses() -> dict[str, Any]:
    return {
        "input-1": {
            "action": "accept",
            "content": {"approved": True},
        }
    }


def _mcp_effect_count(runtime: Runtime, operation: str) -> int:
    return sum(
        effect.provider == "mcp" and effect.operation == operation
        for effect in runtime.store.list_external_effects()
    )


def _respond_pending_continuation(
    runtime: Runtime,
    pending: McpInputRequired,
) -> Any:
    assert pending.human_request_id is not None
    assert pending.human_revision is not None
    assert pending.human_preview_sha256 is not None
    return runtime.mcp.respond_continuation(
        pending.continuation_id,
        expected_revision=pending.revision,
        responses=_approved_input_responses(),
        human_request_id=pending.human_request_id,
        human_expected_revision=pending.human_revision,
        human_preview_sha256=pending.human_preview_sha256,
    )


def _respond_to_cli_continuation(
    capsys: pytest.CaptureFixture[str],
    observed: dict[str, Any],
) -> dict[str, Any]:
    return _run_live_mcp_cli(
        capsys,
        "continuations",
        "respond",
        observed["continuation_id"],
        "--expected-revision",
        str(observed["revision"]),
        "--human-request-id",
        observed["human_request_id"],
        "--human-expected-revision",
        str(observed["human_revision"]),
        "--human-preview-sha256",
        observed["human_preview_sha256"],
        "--responses-json",
        json.dumps(
            {
                "input-1": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            }
        ),
    )


def test_cli_resource_and_prompt_initial_elicitation_use_protected_host_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "durable-host-surfaces.sqlite"
    broker = InMemoryMcpCredentialBroker()
    open_runtime = Runtime.open
    initial = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    pending: dict[str, dict[str, Any]] = {}
    try:
        initial.mcp.register_server(
            _resource_prompt_manifest(),
            actor="runtime",
            require_capability=False,
        )
        adapter = initial._mcp_v3_tool_provider.result_adapter
        provider = _InitialResourcePromptInputRequiredProvider(adapter, initial)
        initial.mcp._modern_client.resource_provider = provider  # noqa: SLF001
        initial.mcp._modern_client.prompt_provider = provider  # noqa: SLF001
        _bind_cli_to_open_runtime(monkeypatch, initial)
        capsys.readouterr()

        pending["resource"] = _run_live_mcp_cli(
            capsys,
            "resources",
            "read",
            "durable-host-surfaces",
            "briefing",
        )
        pending["prompt"] = _run_live_mcp_cli(
            capsys,
            "prompts",
            "get",
            "durable-host-surfaces",
            "review",
            "--arguments-json",
            json.dumps({"topic": "MCP"}),
        )

        assert provider.calls == ["resource", "prompt"]
        for surface, result in pending.items():
            assert result["kind"] == "input_required"
            assert result["respondable"] is True
            assert result["input_requests"] == [
                {
                    "request_id": "input-1",
                    "kind": "elicitation",
                    "mode": "form",
                    "prompt": f"Approve the {surface}?",
                    "schema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                    },
                    "inert_url": None,
                }
            ]
            binding = initial._mcp_continuation_manager.binding_material(
                result["continuation_id"]
            )
            assert binding.owner_id == "cli"
            assert binding.canonical_request["method"] == (
                "resources/read" if surface == "resource" else "prompts/get"
            )
            question = initial.human.get(result["human_request_id"])
            assert question.payload["type"] == "question"
            assert question.payload["context"]["mcp_operation"] == (
                "resources/read" if surface == "resource" else "prompts/get"
            )
            assert (
                question.payload["context"]["mcp_local_ref"]
                == result["continuation_id"]
            )
        public = json.dumps(pending, sort_keys=True)
        assert "broker-only-" not in public
        assert "provider-resource-confirmation" not in public
        assert "provider-prompt-confirmation" not in public
    finally:
        initial.close()

    database_bytes = database.read_bytes()
    assert b"broker-only-resource-round-state" not in database_bytes
    assert b"broker-only-prompt-round-state" not in database_bytes

    reopened = open_runtime(database, substrate=_substrate(tmp_path, broker))
    try:
        continuation_provider = _ResourcePromptContinuationProvider(reopened)
        reopened.mcp._modern_continuation_provider = continuation_provider  # noqa: SLF001
        _bind_cli_to_open_runtime(monkeypatch, reopened)
        capsys.readouterr()

        completed: dict[str, dict[str, Any]] = {}
        for surface in ("resource", "prompt"):
            observed = _run_live_mcp_cli(
                capsys,
                "continuations",
                "inspect",
                pending[surface]["continuation_id"],
            )
            assert observed["kind"] == "input_required"
            assert observed["human_request_id"] == pending[surface]["human_request_id"]
            record_before = reopened._mcp_continuation_manager.repository.get(
                observed["continuation_id"]
            )
            assert record_before is not None
            human_before = reopened.human.get(observed["human_request_id"])
            assert human_before.status.value == "pending"
            provider_calls_before = len(continuation_provider.calls)
            effects_before = sum(
                effect.provider == "mcp"
                and effect.operation == "continuation.respond"
                for effect in reopened.store.list_external_effects()
            )
            for invalid_responses in (
                {
                    "unknown-input": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                },
                {
                    "input-1": {
                        "action": "accept",
                        "content": {"approved": "not-a-boolean"},
                    }
                },
            ):
                failure = _run_live_mcp_cli_failure(
                    capsys,
                    "continuations",
                    "respond",
                    observed["continuation_id"],
                    "--expected-revision",
                    str(observed["revision"]),
                    "--human-request-id",
                    observed["human_request_id"],
                    "--human-expected-revision",
                    str(observed["human_revision"]),
                    "--human-preview-sha256",
                    observed["human_preview_sha256"],
                    "--responses-json",
                    json.dumps(invalid_responses),
                )
                assert "sensitive request details were omitted" in failure["error"][
                    "message"
                ]
                assert "unknown-input" not in json.dumps(failure)
                assert "not-a-boolean" not in json.dumps(failure)
                assert (
                    reopened._mcp_continuation_manager.repository.get(
                        observed["continuation_id"]
                    )
                    == record_before
                )
                human_after = reopened.human.get(observed["human_request_id"])
                assert human_after.status.value == "pending"
                assert human_after.revision == human_before.revision
                assert human_after.decision is None
                assert len(continuation_provider.calls) == provider_calls_before
                assert sum(
                    effect.provider == "mcp"
                    and effect.operation == "continuation.respond"
                    for effect in reopened.store.list_external_effects()
                ) == effects_before
            completed[surface] = _respond_to_cli_continuation(capsys, observed)

        assert completed["resource"] == {
            "kind": "complete",
            "preview_sha256": None,
            "value": {
                "contents": [
                    {
                        "uri": "memo://provider/briefing",
                        "mimeType": "text/plain",
                        "text": "resource approved exactly once",
                    }
                ]
            },
        }
        assert completed["prompt"] == {
            "kind": "complete",
            "preview_sha256": None,
            "value": {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "prompt approved exactly once",
                        },
                    }
                ]
            },
        }
        assert continuation_provider.calls == ["resource", "prompt"]
        records = reopened.audit.trace()
        assert any(
            record.actor == "cli"
            and record.action == "primitive.mcp.resources.read"
            for record in records
        )
        assert any(
            record.actor == "cli"
            and record.action == "primitive.mcp.prompts.get"
            for record in records
        )
        assert sum(
            record.actor == "cli"
            and record.action == "primitive.mcp.continuation.respond"
            for record in records
        ) == 2
    finally:
        reopened.close()
        broker.close()


def test_runtime_facade_captures_real_human_and_reopens_without_initial_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "durable-mrtr.sqlite"
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        initial.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )
        pid = initial.process.spawn(
            image="base-agent:v0",
            goal="exercise durable MCP MRTR facade",
            resource_budget=ResourceBudget(max_mcp_bytes=64_000),
        )
        for resource, rights in (
            ("mcp:durable-mrtr:review", [CapabilityRight.READ]),
            ("mcp_server:durable-mrtr", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            initial.capability.grant(pid, resource, rights, issued_by="test")
        provider = _InitialInputRequiredProvider(
            initial._mcp_v3_tool_provider.result_adapter,
            initial,
        )
        initial.mcp._modern_tool_provider = provider  # noqa: SLF001

        pending = initial.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        cancel_pending = initial.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(cancel_pending, McpInputRequired)
        assert cancel_pending.continuation_id != pending.continuation_id
        assert provider.calls == 2
        assert pending.human_request_id is not None
        question = initial.human.get(pending.human_request_id)
        assert question.payload["type"] == "question"
        assert question.payload["context"]["mcp_local_ref"] == pending.continuation_id
    finally:
        initial.close()

    assert b"broker-only-round-state" not in database.read_bytes()
    reopened = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        continuation_provider = _ContinuationProvider(reopened)
        reopened.mcp._modern_continuation_provider = continuation_provider  # noqa: SLF001
        _bind_cli_to_open_runtime(monkeypatch, reopened)
        capsys.readouterr()
        observed = _run_live_mcp_cli(
            capsys,
            "continuations",
            "inspect",
            pending.continuation_id,
        )
        assert observed["kind"] == "input_required"
        assert observed["human_request_id"] == pending.human_request_id
        result = _run_live_mcp_cli(
            capsys,
            "continuations",
            "respond",
            pending.continuation_id,
            "--expected-revision",
            str(observed["revision"]),
            "--human-request-id",
            observed["human_request_id"],
            "--human-expected-revision",
            str(observed["human_revision"]),
            "--human-preview-sha256",
            observed["human_preview_sha256"],
            "--responses-json",
            json.dumps(
                {
                    "input-1": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                }
            ),
        )
        assert result == {
            "kind": "complete",
            "preview_sha256": None,
            "value": {
                "content": [
                    {"type": "text", "text": "approved exactly once"}
                ]
            },
        }

        cancel_observed = _run_live_mcp_cli(
            capsys,
            "continuations",
            "inspect",
            cancel_pending.continuation_id,
        )
        cancelled = _run_live_mcp_cli(
            capsys,
            "continuations",
            "cancel",
            cancel_pending.continuation_id,
            "--expected-revision",
            str(cancel_observed["revision"]),
        )
        assert cancelled == {
            "kind": "complete",
            "preview_sha256": None,
            "value": None,
        }
        assert continuation_provider.calls == 1
        assert reopened._mcp_continuation_manager.repository.get(
            pending.continuation_id
        ).status == "complete"
        assert reopened._mcp_continuation_manager.repository.get(
            cancel_pending.continuation_id
        ).status == "cancelled"
        effects = reopened.store.list_external_effects()
        contracts = {
            effect.provider_metadata["protected_operation"]["contract_name"]
            for effect in effects
            if effect.provider == "mcp"
        }
        assert {
            "primitive.mcp.call",
            "primitive.mcp.continuation.respond",
            "primitive.mcp.continuation.cancel",
        }.issubset(contracts)
        continuation_effect = [
            effect
            for effect in effects
            if effect.provider == "mcp"
            and effect.operation == "continuation.respond"
        ][-1]
        assert continuation_effect.transaction_state == "committed"
        assert continuation_effect.provider_metadata["data_flow"]["sink"].startswith(
            "mcp:durable-mrtr:continuation.respond"
        )
        assert any(
            record.action == "primitive.mcp.continuation.respond"
            for record in reopened.audit.trace()
        )
        assert any(
            record.action == "primitive.mcp.continuation.cancel"
            for record in reopened.audit.trace()
        )
        usage = reopened.process.get(pid).resource_usage
        assert usage.mcp_request_bytes > 0
        assert usage.mcp_response_bytes > 0
    finally:
        reopened.close()
        broker.close()


def test_continuation_pre_provider_denial_and_ask_restore_but_provider_forgery_does_not(
    tmp_path: Path,
) -> None:
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "durable-continuation-pre-provider.sqlite",
        substrate=_substrate(tmp_path, broker),
    )
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exercise certified MCP continuation failures",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        target_resource = "mcp:durable-mrtr:review"
        target_capability = runtime.capability.grant(
            pid,
            target_resource,
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "mcp_server:durable-mrtr",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        initial_provider = _InitialInputRequiredProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        denied_pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        ask_pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        forged_pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(denied_pending, McpInputRequired)
        assert isinstance(ask_pending, McpInputRequired)
        assert isinstance(forged_pending, McpInputRequired)
        assert {
            item.payload["type"] for item in runtime.human.pending()
        } == {"question"}

        provider = _ContinuationProvider(runtime)
        runtime.mcp._modern_continuation_provider = provider  # noqa: SLF001
        before = runtime._mcp_continuation_manager.repository.get(
            denied_pending.continuation_id
        )
        assert before is not None and before.broker_ref is not None
        runtime.capability.revoke(
            target_capability.cap_id,
            revoked_by="test",
            require_authority=False,
        )
        denied_effects = _mcp_effect_count(runtime, "continuation.respond")

        with pytest.raises(CapabilityDenied):
            _respond_pending_continuation(runtime, denied_pending)

        restored = runtime._mcp_continuation_manager.repository.get(
            denied_pending.continuation_id
        )
        assert restored is not None
        assert restored.status == "input_required"
        assert restored.broker_ref == before.broker_ref
        assert provider.calls == 0
        assert _mcp_effect_count(runtime, "continuation.respond") == denied_effects
        target_capability = runtime.capability.grant(
            pid,
            target_resource,
            [CapabilityRight.READ],
            issued_by="test",
        )
        refreshed = runtime.mcp.get_continuation(denied_pending.continuation_id)
        assert isinstance(_respond_pending_continuation(runtime, refreshed), McpComplete)
        assert provider.calls == 1

        malicious = _MaliciousNotStartedContinuationProvider(runtime)
        runtime.mcp._modern_continuation_provider = malicious  # noqa: SLF001
        with pytest.raises(ValidationError, match="unknown outcome"):
            _respond_pending_continuation(runtime, forged_pending)
        unsafe = runtime._mcp_continuation_manager.repository.get(
            forged_pending.continuation_id
        )
        assert unsafe is not None and unsafe.status == "needs_attention"
        assert malicious.calls == 1
        with pytest.raises(ValidationError):
            runtime.mcp.respond_continuation(
                forged_pending.continuation_id,
                expected_revision=unsafe.revision,
                responses=_approved_input_responses(),
                human_request_id=forged_pending.human_request_id or "",
                human_expected_revision=forged_pending.human_revision or 0,
                human_preview_sha256=forged_pending.human_preview_sha256 or "",
            )
        assert malicious.calls == 1
        runtime.mcp._modern_continuation_provider = provider  # noqa: SLF001

        runtime.capability.revoke(
            target_capability.cap_id,
            revoked_by="test",
            require_authority=False,
        )
        runtime.capability.set_permission_policy(
            pid,
            target_resource,
            [CapabilityRight.READ],
            runtime.capability.ASK_EACH_TIME,
            issued_by="test",
        )
        ask_before = runtime._mcp_continuation_manager.repository.get(
            ask_pending.continuation_id
        )
        assert ask_before is not None and ask_before.broker_ref is not None
        ask_effects = _mcp_effect_count(runtime, "continuation.cancel")
        with pytest.raises(HumanApprovalRequired) as approval:
            runtime.mcp.cancel_continuation(
                ask_pending.continuation_id,
                expected_revision=ask_pending.revision,
            )
        ask_restored = runtime._mcp_continuation_manager.repository.get(
            ask_pending.continuation_id
        )
        assert ask_restored is not None
        assert ask_restored.status == "input_required"
        assert ask_restored.broker_ref == ask_before.broker_ref
        assert _mcp_effect_count(runtime, "continuation.cancel") == ask_effects
        pending_approval = approval.value
        current = ask_restored
        for index, expected_operation in enumerate(
            ("mcp.call", "mcp.continuation.cancel")
        ):
            authority_request = runtime.human.get(pending_approval.request_id)
            authority_rules = authority_request.payload[
                "requested_once_capability"
            ]["constraints"]["authority_rules"]
            assert [rule["operation"] for rule in authority_rules] == [
                expected_operation
            ]
            authority_preview = runtime.human.canonical_approval_preview(
                authority_request
            )
            approved = runtime.human.approve(
                authority_request.request_id,
                {"approved": True},
                expected_revision=authority_request.revision,
                preview_sha256=authority_preview.canonical_sha256(),
            )
            assert approved.request_id == pending_approval.request_id
            if index == 0:
                with pytest.raises(HumanApprovalRequired) as next_approval:
                    runtime.mcp.cancel_continuation(
                        ask_pending.continuation_id,
                        expected_revision=current.revision,
                    )
                current = runtime._mcp_continuation_manager.repository.get(
                    ask_pending.continuation_id
                )
                assert current is not None
                assert current.status == "input_required"
                assert current.broker_ref == ask_before.broker_ref
                assert _mcp_effect_count(
                    runtime, "continuation.cancel"
                ) == ask_effects
                pending_approval = next_approval.value
        cancelled = runtime.mcp.cancel_continuation(
            ask_pending.continuation_id,
            expected_revision=current.revision,
        )
        assert isinstance(cancelled, McpComplete)
        assert provider.calls == 1
    finally:
        runtime.close()
        broker.close()


@pytest.mark.parametrize("pin_case", ["missing-extension", "wrong-host-pin"])
def test_continuation_task_requires_current_manifest_and_host_pin_before_capture(
    tmp_path: Path,
    pin_case: str,
) -> None:
    manifest = _manifest()
    config = DEFAULT_CONFIG
    if pin_case == "wrong-host-pin":
        manifest = replace(
            manifest,
            tasks_extension=McpTasksExtensionSpec(
                extension_id=MCP_TASKS_EXTENSION_ID,
                spec_sha256="b" * 64,
            ),
        )
        config = AgentLibOSConfig(
            mcp=replace(
                DEFAULT_CONFIG.mcp,
                tasks_extension_enabled=True,
                tasks_extension_spec_sha256="b" * 64,
            )
        )
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / f"continuation-task-pin-{pin_case}.sqlite",
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        runtime.mcp.register_server(
            manifest, actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject unpinned continuation Task capture",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        for resource, rights in (
            ("mcp:durable-mrtr:review", [CapabilityRight.READ]),
            ("mcp_server:durable-mrtr", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")
        initial_provider = _InitialInputRequiredProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        if pin_case == "wrong-host-pin":
            # Registration and the initial call were valid under the original
            # Host pin.  A Host-policy change before continuation settlement
            # must be rechecked against the current manifest and fail before
            # any Task capture can observe the Provider result.
            runtime.mcp.config = AgentLibOSConfig(
                mcp=replace(
                    config.mcp,
                    tasks_extension_spec_sha256="a" * 64,
                )
            )
        provider = _ContinuationTaskProvider()
        runtime.mcp._modern_continuation_provider = provider  # noqa: SLF001

        if pin_case == "missing-extension":
            with pytest.raises(ValidationError, match="unknown outcome"):
                _respond_pending_continuation(runtime, pending)
        else:
            with pytest.raises(
                ValidationError,
                match="Tasks extension spec digest does not match Host policy pin",
            ):
                _respond_pending_continuation(runtime, pending)

        continuation = runtime._mcp_continuation_manager.repository.get(
            pending.continuation_id
        )
        assert continuation is not None
        if pin_case == "missing-extension":
            assert continuation.status == "needs_attention"
            assert provider.calls == 1
        else:
            assert continuation.status == "input_required"
            assert provider.calls == 0
        assert runtime.uow.mcp_remote_tasks.list(limit=100) == ()
        assert not any(
            item.operation_kind == "remote_task"
            for item in runtime.uow.mcp_side_effects.list()
        )
        assert not any(
            secret.namespace.startswith("mcp.remote_task.")
            for secret in broker._secrets.values()  # noqa: SLF001
        )
        effects = [
            item
            for item in runtime.store.list_external_effects()
            if item.provider == "mcp"
            and item.operation == "continuation.respond"
        ]
        if pin_case == "missing-extension":
            assert effects[-1].transaction_state == "unknown"
        else:
            assert effects == []
    finally:
        runtime.close()
        broker.close()


@pytest.mark.parametrize("settlement_case", ["malformed", "round-limit"])
def test_post_provider_continuation_settlement_failure_is_never_not_started(
    tmp_path: Path,
    settlement_case: str,
) -> None:
    config = (
        AgentLibOSConfig(mcp=replace(DEFAULT_CONFIG.mcp, mrtr_max_rounds=1))
        if settlement_case == "round-limit"
        else DEFAULT_CONFIG
    )
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / f"continuation-settlement-{settlement_case}.sqlite",
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        runtime.mcp.register_server(
            _manifest(), actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="classify continuation settlement after provider dispatch",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        for resource, rights in (
            ("mcp:durable-mrtr:review", [CapabilityRight.READ]),
            ("mcp_server:durable-mrtr", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")
        initial_provider = _InitialInputRequiredProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        provider = _SettlementFailureContinuationProvider(settlement_case)
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        runtime.mcp._modern_continuation_provider = provider  # noqa: SLF001
        pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)

        with pytest.raises(ValidationError, match="unknown outcome"):
            _respond_pending_continuation(runtime, pending)

        record = runtime._mcp_continuation_manager.repository.get(
            pending.continuation_id
        )
        assert record is not None
        assert record.status == "needs_attention"
        assert provider.calls == 1
        effect = [
            item
            for item in runtime.store.list_external_effects()
            if item.provider == "mcp"
            and item.operation == "continuation.respond"
        ][-1]
        assert effect.transaction_state == "unknown"
        with pytest.raises(ValidationError):
            runtime.mcp.respond_continuation(
                pending.continuation_id,
                expected_revision=record.revision,
                responses=_approved_input_responses(),
                human_request_id=pending.human_request_id or "",
                human_expected_revision=pending.human_revision or 0,
                human_preview_sha256=pending.human_preview_sha256 or "",
            )
        assert provider.calls == 1
    finally:
        runtime.close()
        broker.close()


def test_runtime_remote_task_facade_get_update_cancel_uses_only_local_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "b" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.000001,
        )
    )
    manifest = replace(
        _manifest(),
        server_id="durable-tasks",
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
        tools=(
            replace(
                _manifest().tools[0],
                tool_id="begin-task",
                mcp_name="fixture.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["input", "cancel", "error"],
                        }
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "durable-tasks.sqlite",
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        runtime.mcp.register_server(
            manifest, actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exercise remote Tasks facade",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        for resource, rights in (
            ("mcp:durable-tasks:begin-task", [CapabilityRight.READ]),
            ("mcp_server:durable-tasks", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")
        initial_provider = _InitialTaskProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        tasks_provider = _TasksProvider(runtime)
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        runtime.mcp._modern_tasks_provider = tasks_provider  # noqa: SLF001

        input_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "input"}
        )
        assert isinstance(input_task, McpRemoteTask)
        assert "private-input-task" not in repr(input_task)
        runtime.capability.grant(
            pid,
            f"mcp_task:{input_task.task_ref}",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
        )
        _bind_cli_to_open_runtime(monkeypatch, runtime)
        capsys.readouterr()
        waiting = _run_live_mcp_cli(
            capsys,
            "remote-tasks",
            "get",
            input_task.task_ref,
            "--expected-revision",
            str(input_task.revision),
        )
        assert waiting["kind"] == "remote_task"
        assert waiting["status"] == "input_required"
        assert waiting["human_request_id"] is not None
        task_record_before = runtime._mcp_remote_task_manager.repository.get(
            waiting["task_ref"]
        )
        assert task_record_before is not None
        task_human_before = runtime.human.get(waiting["human_request_id"])
        assert task_human_before.status.value == "pending"
        update_calls_before = tasks_provider.update_calls
        update_effects_before = sum(
            effect.provider == "mcp" and effect.operation == "tasks.update"
            for effect in runtime.store.list_external_effects()
        )
        for invalid_responses in (
            {
                "unknown-input": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
            {
                "input-1": {
                    "action": "accept",
                    "content": {"approved": "not-a-boolean"},
                }
            },
        ):
            failure = _run_live_mcp_cli_failure(
                capsys,
                "remote-tasks",
                "update",
                waiting["task_ref"],
                "--expected-revision",
                str(waiting["revision"]),
                "--human-request-id",
                waiting["human_request_id"],
                "--human-expected-revision",
                str(waiting["human_revision"]),
                "--human-preview-sha256",
                waiting["human_preview_sha256"],
                "--responses-json",
                json.dumps(invalid_responses),
            )
            assert "sensitive request details were omitted" in failure["error"][
                "message"
            ]
            assert "unknown-input" not in json.dumps(failure)
            assert "not-a-boolean" not in json.dumps(failure)
            assert (
                runtime._mcp_remote_task_manager.repository.get(waiting["task_ref"])
                == task_record_before
            )
            task_human_after = runtime.human.get(waiting["human_request_id"])
            assert task_human_after.status.value == "pending"
            assert task_human_after.revision == task_human_before.revision
            assert task_human_after.decision is None
            assert tasks_provider.update_calls == update_calls_before
            assert sum(
                effect.provider == "mcp" and effect.operation == "tasks.update"
                for effect in runtime.store.list_external_effects()
            ) == update_effects_before
        working = _run_live_mcp_cli(
            capsys,
            "remote-tasks",
            "update",
            waiting["task_ref"],
            "--expected-revision",
            str(waiting["revision"]),
            "--human-request-id",
            waiting["human_request_id"],
            "--human-expected-revision",
            str(waiting["human_revision"]),
            "--human-preview-sha256",
            waiting["human_preview_sha256"],
            "--responses-json",
            json.dumps(
                {
                    "input-1": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                }
            ),
        )
        assert working["kind"] == "remote_task"
        assert working["status"] == "working"
        completed = _run_live_mcp_cli(
            capsys,
            "remote-tasks",
            "get",
            working["task_ref"],
            "--expected-revision",
            str(working["revision"]),
        )
        assert completed["status"] == "completed"
        assert completed["result"] == {
            "approved": True,
            "providerEcho": "[redacted]",
        }
        assert "private-input-task" not in repr(completed)

        cancel_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "cancel"}
        )
        assert isinstance(cancel_task, McpRemoteTask)
        runtime.capability.grant(
            pid,
            f"mcp_task:{cancel_task.task_ref}",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
        )
        requested = _run_live_mcp_cli(
            capsys,
            "remote-tasks",
            "cancel",
            cancel_task.task_ref,
            "--expected-revision",
            str(cancel_task.revision),
        )
        assert requested["status"] == "cancel_requested"
        assert requested["status"] != "cancelled"
        cancelled = _run_live_mcp_cli(
            capsys,
            "remote-tasks",
            "get",
            requested["task_ref"],
            "--expected-revision",
            str(requested["revision"]),
        )
        assert cancelled["status"] == "cancelled"

        error_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "error"}
        )
        assert isinstance(error_task, McpRemoteTask)
        runtime.capability.grant(
            pid,
            f"mcp_task:{error_task.task_ref}",
            [CapabilityRight.READ],
            issued_by="test",
        )
        with pytest.raises(ValidationError) as reflected_error:
            runtime.mcp.get_remote_task(
                error_task.task_ref,
                expected_revision=error_task.revision,
            )
        assert "MCP provider operation failed" in str(reflected_error.value)
        assert "private-error-task" not in str(reflected_error.value)

        assert initial_provider.calls == 3
        assert tasks_provider.get_calls == 4
        assert tasks_provider.update_calls == 1
        assert tasks_provider.cancel_calls == 1
        assert all(
            "private-" not in repr(record)
            for record in runtime.uow.mcp_remote_tasks.list(limit=100)
        )
        assert "private-error-task" not in dumps(
            [to_jsonable(record) for record in runtime.audit.trace()]
        )
        effects = runtime.store.list_external_effects()
        contracts = {
            effect.provider_metadata["protected_operation"]["contract_name"]
            for effect in effects
            if effect.provider == "mcp"
            and "protected_operation" in effect.provider_metadata
        }
        assert {
            "primitive.mcp.call",
            "primitive.mcp.tasks.get",
            "primitive.mcp.tasks.update",
            "primitive.mcp.tasks.cancel",
        }.issubset(contracts)
        actions = {record.action for record in runtime.audit.trace()}
        assert {
            "primitive.mcp.tasks.get",
            "primitive.mcp.tasks.update",
            "primitive.mcp.tasks.cancel",
        }.issubset(actions)
        usage = runtime.process.get(pid).resource_usage
        assert usage.mcp_request_bytes > 0
        assert usage.mcp_response_bytes > 0
    finally:
        runtime.close()
        broker.close()


def test_remote_task_pre_provider_denial_and_ask_restore_without_dispatch(
    tmp_path: Path,
) -> None:
    digest = "b" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.000001,
        )
    )
    manifest = replace(
        _manifest(),
        server_id="durable-tasks",
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
        tools=(
            replace(
                _manifest().tools[0],
                tool_id="begin-task",
                mcp_name="fixture.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["input", "cancel", "error"],
                        }
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "durable-task-pre-provider.sqlite",
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        runtime.mcp.register_server(
            manifest, actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exercise certified remote Task failures",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        for resource, rights in (
            ("mcp:durable-tasks:begin-task", [CapabilityRight.READ]),
            ("mcp_server:durable-tasks", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")
        initial_provider = _InitialTaskProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        tasks_provider = _TasksProvider(runtime)
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        runtime.mcp._modern_tasks_provider = tasks_provider  # noqa: SLF001

        input_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "input"}
        )
        assert isinstance(input_task, McpRemoteTask)
        task_resource = f"mcp_task:{input_task.task_ref}"
        task_capability = runtime.capability.grant(
            pid,
            task_resource,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
        )
        waiting = runtime.mcp.get_remote_task(
            input_task.task_ref,
            expected_revision=input_task.revision,
        )
        assert waiting.status.value == "input_required"
        assert waiting.human_request_id is not None
        assert waiting.human_revision is not None
        assert waiting.human_preview_sha256 is not None
        update_before = runtime._mcp_remote_task_manager.repository.get(
            waiting.task_ref
        )
        assert update_before is not None and update_before.broker_ref is not None
        runtime.capability.revoke(
            task_capability.cap_id,
            revoked_by="test",
            require_authority=False,
        )
        update_effects = _mcp_effect_count(runtime, "tasks.update")
        task_responses = {
            "input-1": {
                "action": "accept",
                "content": {"approved": True},
            }
        }

        with pytest.raises(CapabilityDenied):
            runtime.mcp.update_remote_task(
                waiting.task_ref,
                expected_revision=waiting.revision,
                responses=task_responses,
                human_request_id=waiting.human_request_id,
                human_expected_revision=waiting.human_revision,
                human_preview_sha256=waiting.human_preview_sha256,
            )

        update_restored = runtime._mcp_remote_task_manager.repository.get(
            waiting.task_ref
        )
        assert update_restored is not None
        assert update_restored.status == "input_required"
        assert update_restored.broker_ref == update_before.broker_ref
        assert tasks_provider.update_calls == 0
        assert _mcp_effect_count(runtime, "tasks.update") == update_effects
        assert runtime.human.get(waiting.human_request_id).status.value == "approved"
        runtime.capability.grant(
            pid,
            task_resource,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        working = runtime.mcp.update_remote_task(
            waiting.task_ref,
            expected_revision=update_restored.revision,
            responses=task_responses,
            human_request_id=waiting.human_request_id,
            human_expected_revision=waiting.human_revision,
            human_preview_sha256=waiting.human_preview_sha256,
        )
        assert working.status.value == "working"
        assert tasks_provider.update_calls == 1

        cancel_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "cancel"}
        )
        assert isinstance(cancel_task, McpRemoteTask)
        cancel_resource = f"mcp_task:{cancel_task.task_ref}"
        cancel_capability = runtime.capability.grant(
            pid,
            cancel_resource,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.capability.revoke(
            cancel_capability.cap_id,
            revoked_by="test",
            require_authority=False,
        )
        runtime.capability.set_permission_policy(
            pid,
            cancel_resource,
            [CapabilityRight.WRITE],
            runtime.capability.ASK_EACH_TIME,
            issued_by="test",
        )
        cancel_before = runtime._mcp_remote_task_manager.repository.get(
            cancel_task.task_ref
        )
        assert cancel_before is not None and cancel_before.broker_ref is not None
        cancel_effects = _mcp_effect_count(runtime, "tasks.cancel")

        with pytest.raises(HumanApprovalRequired) as approval:
            runtime.mcp.cancel_remote_task(
                cancel_task.task_ref,
                expected_revision=cancel_task.revision,
            )

        cancel_restored = runtime._mcp_remote_task_manager.repository.get(
            cancel_task.task_ref
        )
        assert cancel_restored is not None
        assert cancel_restored.status == "working"
        assert cancel_restored.broker_ref == cancel_before.broker_ref
        assert tasks_provider.cancel_calls == 0
        assert _mcp_effect_count(runtime, "tasks.cancel") == cancel_effects
        authority_request = runtime.human.get(approval.value.request_id)
        authority_rules = authority_request.payload["requested_once_capability"][
            "constraints"
        ]["authority_rules"]
        assert [rule["operation"] for rule in authority_rules] == [
            "mcp.tasks.cancel"
        ]
        authority_preview = runtime.human.canonical_approval_preview(
            authority_request
        )
        runtime.human.approve(
            authority_request.request_id,
            {"approved": True},
            expected_revision=authority_request.revision,
            preview_sha256=authority_preview.canonical_sha256(),
        )
        requested = runtime.mcp.cancel_remote_task(
            cancel_task.task_ref,
            expected_revision=cancel_restored.revision,
        )
        assert requested.status.value == "cancel_requested"
        assert tasks_provider.cancel_calls == 1
    finally:
        runtime.close()
        broker.close()


def test_custom_tool_and_continuation_deadline_contract_and_violation_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deadline_s = 5.0
    manifest = replace(_manifest(), timeout_s=deadline_s)
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "durable-modern-provider-deadline.sqlite",
        substrate=_substrate(tmp_path, broker),
    )
    created_loops: list[asyncio.AbstractEventLoop] = []
    new_event_loop = asyncio.new_event_loop

    def tracked_event_loop() -> asyncio.AbstractEventLoop:
        loop = new_event_loop()
        created_loops.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", tracked_event_loop)
    try:
        runtime.mcp.register_server(
            manifest, actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="enforce cooperative modern Provider deadlines",
            resource_budget=ResourceBudget(max_mcp_bytes=512_000),
        )
        for resource, rights in (
            ("mcp:durable-mrtr:review", [CapabilityRight.READ]),
            ("mcp_server:durable-mrtr", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")

        hanging_tool = _HangingToolProvider()
        runtime.mcp._modern_tool_provider = hanging_tool  # noqa: SLF001
        started = time.monotonic()
        with pytest.raises(ValidationError, match="absolute deadline"):
            runtime.mcp.call_tool(
                pid,
                "durable-mrtr",
                "review",
                {"document": "release-notes"},
            )
        assert time.monotonic() - started < deadline_s + 5.0
        assert hanging_tool.calls == 1
        assert hanging_tool.cancellations >= 1
        assert hanging_tool.closed == 1
        tool_effect = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "call_tool"
        ][-1]
        assert tool_effect.transaction_state == "unknown"

        suppressing_tool = _CancellationSuppressingToolProvider()
        runtime.mcp._modern_tool_provider = suppressing_tool  # noqa: SLF001
        started = time.monotonic()
        with pytest.raises(ValidationError, match="absolute deadline"):
            runtime.mcp.call_tool(
                pid,
                "durable-mrtr",
                "review",
                {"document": "release-notes"},
            )
        # Cancellation suppression is outside the supported SPI contract. The
        # operation-local loop still prevents this yielding violation from
        # hanging sync Runtime/CLI shutdown if a Host composes it accidentally.
        assert time.monotonic() - started < deadline_s + 5.0
        assert suppressing_tool.calls == 1
        assert suppressing_tool.cancellations >= 1
        assert suppressing_tool.closed == 1

        blocking_tool = _BlockingToolProvider(delay_s=deadline_s + 0.2)
        runtime.mcp._modern_tool_provider = blocking_tool  # noqa: SLF001
        started = time.monotonic()
        with pytest.raises(ValidationError, match="absolute deadline"):
            runtime.mcp.call_tool(
                pid,
                "durable-mrtr",
                "review",
                {"document": "release-notes"},
            )
        elapsed = time.monotonic() - started
        # A blocking in-process coroutine is explicitly outside the custom-SPI
        # contract. Runtime cannot hard-preempt it, but rejects the late result
        # and never presents the operation as retry-safe.
        assert deadline_s <= elapsed < deadline_s + 5.0
        assert blocking_tool.calls == 1
        blocking_effect = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "call_tool"
        ][-1]
        assert blocking_effect.transaction_state == "unknown"

        initial_provider = _InitialInputRequiredProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        pending = runtime.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        hanging_continuation = _HangingContinuationProvider()
        runtime.mcp._modern_continuation_provider = (  # noqa: SLF001
            hanging_continuation
        )
        started = time.monotonic()
        with pytest.raises(ValidationError, match="unknown outcome"):
            _respond_pending_continuation(runtime, pending)
        assert time.monotonic() - started < deadline_s + 5.0
        record = runtime._mcp_continuation_manager.repository.get(
            pending.continuation_id
        )
        assert record is not None and record.status == "needs_attention"
        assert hanging_continuation.calls == 1
        assert hanging_continuation.cancellations >= 1
        assert hanging_continuation.closed == 1
        continuation_effect = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp"
            and effect.operation == "continuation.respond"
        ][-1]
        assert continuation_effect.transaction_state == "unknown"
        with pytest.raises(ValidationError):
            runtime.mcp.respond_continuation(
                pending.continuation_id,
                expected_revision=record.revision,
                responses=_approved_input_responses(),
                human_request_id=pending.human_request_id or "",
                human_expected_revision=pending.human_revision or 0,
                human_preview_sha256=pending.human_preview_sha256 or "",
            )
        assert hanging_continuation.calls == 1
        assert created_loops
        assert all(loop.is_closed() for loop in created_loops)
    finally:
        close_started = time.monotonic()
        runtime.close()
        assert time.monotonic() - close_started < 2.0
        broker.close()


def test_custom_task_deadlines_cancel_cooperatively_and_never_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "b" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.000001,
        )
    )
    manifest = replace(
        _manifest(),
        server_id="durable-tasks",
        # Exercise bounded cooperative cancellation without making ordinary
        # protected-operation setup depend on sub-second scheduler latency.
        timeout_s=2.0,
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
        tools=(
            replace(
                _manifest().tools[0],
                tool_id="begin-task",
                mcp_name="fixture.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["input", "cancel", "error"],
                        }
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        tmp_path / "durable-task-provider-deadline.sqlite",
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    created_loops: list[asyncio.AbstractEventLoop] = []
    new_event_loop = asyncio.new_event_loop

    def tracked_event_loop() -> asyncio.AbstractEventLoop:
        loop = new_event_loop()
        created_loops.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", tracked_event_loop)
    try:
        runtime.mcp.register_server(
            manifest, actor="runtime", require_capability=False
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="enforce cooperative Tasks Provider cancellation",
            resource_budget=ResourceBudget(max_mcp_bytes=128_000),
        )
        for resource, rights in (
            ("mcp:durable-tasks:begin-task", [CapabilityRight.READ]),
            ("mcp_server:durable-tasks", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test")
        initial_provider = _InitialTaskProvider(
            runtime._mcp_v3_tool_provider.result_adapter,
            runtime,
        )
        normal_tasks = _TasksProvider(runtime)
        runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        runtime.mcp._modern_tasks_provider = normal_tasks  # noqa: SLF001

        get_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "error"}
        )
        update_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "input"}
        )
        cancel_task = runtime.mcp.call_tool(
            pid, "durable-tasks", "begin-task", {"mode": "cancel"}
        )
        assert isinstance(get_task, McpRemoteTask)
        assert isinstance(update_task, McpRemoteTask)
        assert isinstance(cancel_task, McpRemoteTask)
        for task in (get_task, update_task, cancel_task):
            runtime.capability.grant(
                pid,
                f"mcp_task:{task.task_ref}",
                [CapabilityRight.READ, CapabilityRight.WRITE],
                issued_by="test",
            )
        waiting = runtime.mcp.get_remote_task(
            update_task.task_ref,
            expected_revision=update_task.revision,
        )
        assert waiting.status.value == "input_required"
        assert waiting.human_request_id is not None
        assert waiting.human_revision is not None
        assert waiting.human_preview_sha256 is not None

        hanging = _HangingTasksProvider()
        runtime.mcp._modern_tasks_provider = hanging  # noqa: SLF001
        started = time.monotonic()
        with pytest.raises(ValidationError, match="sensitive details"):
            runtime.mcp.get_remote_task(
                get_task.task_ref,
                expected_revision=get_task.revision,
            )
        assert time.monotonic() - started < 5.0
        get_record = runtime._mcp_remote_task_manager.repository.get(
            get_task.task_ref
        )
        assert get_record is not None and get_record.status == "working"

        responses = {
            "input-1": {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        started = time.monotonic()
        with pytest.raises(ValidationError, match="sensitive details"):
            runtime.mcp.update_remote_task(
                waiting.task_ref,
                expected_revision=waiting.revision,
                responses=responses,
                human_request_id=waiting.human_request_id,
                human_expected_revision=waiting.human_revision,
                human_preview_sha256=waiting.human_preview_sha256,
            )
        assert time.monotonic() - started < 5.0
        update_record = runtime._mcp_remote_task_manager.repository.get(
            waiting.task_ref
        )
        assert update_record is not None
        assert update_record.status == "needs_attention"

        started = time.monotonic()
        with pytest.raises(ValidationError, match="sensitive details"):
            runtime.mcp.cancel_remote_task(
                cancel_task.task_ref,
                expected_revision=cancel_task.revision,
            )
        assert time.monotonic() - started < 5.0
        cancel_record = runtime._mcp_remote_task_manager.repository.get(
            cancel_task.task_ref
        )
        assert cancel_record is not None
        assert cancel_record.status == "needs_attention"
        assert hanging.operations == ["get", "update", "cancel"]
        assert hanging.calls == 3
        assert hanging.cancellations >= 3
        assert hanging.closed == 3
        for operation in ("tasks.get", "tasks.update", "tasks.cancel"):
            effect = [
                item
                for item in runtime.store.list_external_effects()
                if item.provider == "mcp" and item.operation == operation
            ][-1]
            assert effect.transaction_state == "unknown"

        with pytest.raises(ValidationError):
            runtime.mcp.update_remote_task(
                waiting.task_ref,
                expected_revision=update_record.revision,
                responses=responses,
                human_request_id=waiting.human_request_id,
                human_expected_revision=waiting.human_revision,
                human_preview_sha256=waiting.human_preview_sha256,
            )
        with pytest.raises(ValidationError):
            runtime.mcp.cancel_remote_task(
                cancel_task.task_ref,
                expected_revision=cancel_record.revision,
            )
        assert hanging.calls == 3
        assert created_loops
        assert all(loop.is_closed() for loop in created_loops)
    finally:
        close_started = time.monotonic()
        runtime.close()
        assert time.monotonic() - close_started < 2.0
        broker.close()
