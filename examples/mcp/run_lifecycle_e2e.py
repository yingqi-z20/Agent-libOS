#!/usr/bin/env python3
"""Run the deterministic MCP MRTR, Tasks, and subscription lifecycle.

The providers below are local Host fixtures.  They make no network requests,
hold remote bearer-like state only in memory, and let the public Runtime facade
exercise its real Capability, Human, effect, audit, persistence, and recovery
boundaries.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp import (
    InMemoryMcpCredentialBroker,
    McpComplete,
    McpInputRequired,
    McpRemoteTask,
    McpResourceSpec,
    McpServerManifestV3,
    McpSubscriptionEvent,
    McpSubscriptionSession,
    McpSubscriptionStatus,
    McpTasksExtensionSpec,
)
from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.models import (
    CapabilityRight,
    McpHttpTransportSpec,
    McpProtocolMode,
    McpToolSpec,
    ResourceBudget,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate


SERVER_ID = "modern-lifecycle-demo"
HOST_ACTOR = "examples.mcp.lifecycle-host"
TASKS_SPEC_SHA256 = "b" * 64
RAW_REQUEST_STATE = "remote-request-state-MUST-NOT-PERSIST"
RAW_REVIEW_INPUT_KEY = "provider-review-input-MUST-NOT-PERSIST"
RAW_TASK_INPUT_KEY = "provider-task-input-MUST-NOT-PERSIST"
RAW_INPUT_TASK_ID = "remote-input-task-MUST-NOT-PERSIST"
RAW_CANCEL_TASK_ID = "remote-cancel-task-MUST-NOT-PERSIST"
RAW_SUBSCRIPTION_HANDLE = "remote-subscription-handle-MUST-NOT-PERSIST"
PRIVATE_SENTINELS = (
    RAW_REQUEST_STATE,
    RAW_REVIEW_INPUT_KEY,
    RAW_TASK_INPUT_KEY,
    RAW_INPUT_TASK_ID,
    RAW_CANCEL_TASK_ID,
    RAW_SUBSCRIPTION_HANDLE,
)


class ScriptedToolProvider:
    """Initial Tool fixture using the Runtime-configured durable result adapter."""

    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(
        self,
        result_adapter: Any,
        runtime: Runtime,
        *,
        task_created_at: str,
    ) -> None:
        self.result_adapter = result_adapter
        self.runtime = runtime
        self.task_created_at = task_created_at
        self.calls: dict[str, int] = {"review": 0, "begin-task": 0}

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
            raise RuntimeError("credential-free lifecycle fixture received a secret")
        _assert_pending_effect(self.runtime, "call_tool")
        self.calls[tool_id] += 1
        if tool_id == "review":
            assert arguments == {"document": "release-notes"}
            raw: dict[str, Any] = {
                "resultType": "input_required",
                "requestState": RAW_REQUEST_STATE,
                "inputRequests": {
                    RAW_REVIEW_INPUT_KEY: {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Approve this review?",
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
            mode = arguments["mode"]
            remote_id = (
                RAW_INPUT_TASK_ID if mode == "input" else RAW_CANCEL_TASK_ID
            )
            raw = {
                "resultType": "task",
                "taskId": remote_id,
                "status": "working",
                "createdAt": self.task_created_at,
                "lastUpdatedAt": self.task_created_at,
                "ttlMs": 300_000,
                "pollIntervalMs": 1,
            }
        else:  # pragma: no cover - the manifest is the closed selector set
            raise AssertionError(f"unexpected local Tool fixture selector: {tool_id}")
        return self.result_adapter.tool_result(
            raw,
            server_id=manifest.server_id,
            logical_id=tool_id,
            deadline=deadline,
            sensitive_values=sensitive_values,
        )


class RejectInitialReplayProvider:
    """Reopen sentinel: any initial Tool dispatch is a recovery contract bug."""

    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(
        self,
        _manifest: McpServerManifestV3,
        _tool_id: str,
        _arguments: dict[str, Any],
        *,
        deadline: float,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpComplete[Any]:
        del deadline, sensitive_values
        self.calls += 1
        raise RuntimeError("Runtime restart replayed an initial MCP Tool")


class ScriptedContinuationProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self) -> None:
        self.runtime: Runtime | None = None
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
        assert self.runtime is not None
        _assert_pending_effect(self.runtime, "continuation.respond")
        assert deadline > time.monotonic()
        assert server.server_id == SERVER_ID
        assert mcp_name == "demo.review"
        assert arguments == {"document": "release-notes"}
        assert request_state == RAW_REQUEST_STATE
        assert input_responses == {
            RAW_REVIEW_INPUT_KEY: {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        self.calls += 1
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": "approved exactly once"}],
        }

    async def continue_resource(
        self,
        _server: Any,
        _resource_name: str,
        _logical_id: str,
        _input_responses: dict[str, Any],
        _request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del deadline
        raise AssertionError("the lifecycle fixture has no Resource continuation")

    async def continue_prompt(
        self,
        _server: Any,
        _prompt_name: str,
        _logical_id: str,
        _arguments: dict[str, str],
        _input_responses: dict[str, Any],
        _request_state: str | None,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        del deadline
        raise AssertionError("the lifecycle fixture has no Prompt continuation")


class ScriptedTasksProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self, *, task_created_at: str) -> None:
        self.runtime: Runtime | None = None
        self.task_created_at = task_created_at
        self.statuses = {
            RAW_INPUT_TASK_ID: "working",
            RAW_CANCEL_TASK_ID: "working",
        }
        self.get_calls = 0
        self.update_calls = 0
        self.cancel_calls = 0

    async def get_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        assert self.runtime is not None
        _assert_pending_effect(self.runtime, "tasks.get")
        assert server.server_id == SERVER_ID
        assert deadline > time.monotonic()
        self.get_calls += 1
        status = self.statuses[remote_task_id]
        if remote_task_id == RAW_INPUT_TASK_ID and status == "working":
            status = self.statuses[remote_task_id] = "input_required"
        return self._result(remote_task_id, status)

    async def update_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        response: Mapping[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        assert self.runtime is not None
        _assert_pending_effect(self.runtime, "tasks.update")
        assert server.server_id == SERVER_ID
        assert deadline > time.monotonic()
        assert response == {
            RAW_TASK_INPUT_KEY: {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        self.update_calls += 1
        self.statuses[remote_task_id] = "completed"
        return {"resultType": "complete"}

    async def cancel_remote_task(
        self,
        server: Any,
        remote_task_id: str,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        assert self.runtime is not None
        _assert_pending_effect(self.runtime, "tasks.cancel")
        assert server.server_id == SERVER_ID
        assert deadline > time.monotonic()
        self.cancel_calls += 1
        self.statuses[remote_task_id] = "cancelled"
        return {"resultType": "complete"}

    def _result(self, remote_task_id: str, status: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "resultType": "complete",
            "taskId": remote_task_id,
            "status": status,
            "createdAt": self.task_created_at,
            "lastUpdatedAt": self.task_created_at,
            "ttlMs": 300_000,
            "pollIntervalMs": 1,
        }
        if status == "input_required":
            result["inputRequests"] = {
                RAW_TASK_INPUT_KEY: {
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
            result["result"] = {"approved": True}
        return result


class ScriptedSubscriptionProvider:
    mcp_manifest_schema_version = 3
    mcp_protocol_revision = "2026-07-28"

    def __init__(self) -> None:
        self.runtime: Runtime | None = None
        self.listen_count = 0
        self.receive_count = 0
        self.close_count = 0
        self.events_by_stream = {
            1: [
                McpSubscriptionEvent(
                    sequence=0,
                    event_type="resourcesListChanged",
                    payload={"changed": True, "resource_id": "status"},
                    received_at="2030-01-01T00:00:04+00:00",
                )
            ],
            2: [
                McpSubscriptionEvent(
                    sequence=0,
                    event_type="resourcesListChanged",
                    payload={"changed": True, "resource_id": "queued-before-restart"},
                    received_at="2030-01-01T00:00:05+00:00",
                )
            ],
        }

    async def listen(
        self,
        server: Any,
        filters: tuple[str, ...],
        *,
        deadline: float,
    ) -> McpSubscriptionSession:
        assert self.runtime is not None
        _assert_pending_effect(self.runtime, "subscriptions.start")
        assert server.server_id == SERVER_ID
        assert filters == ("resourcesListChanged",)
        assert deadline > time.monotonic()
        self.listen_count += 1
        stream_number = self.listen_count
        owner_task = asyncio.create_task(
            asyncio.Event().wait(),
            name="example-mcp-subscription-owner",
        )
        return McpSubscriptionSession(
            handle={"opaque": RAW_SUBSCRIPTION_HANDLE, "stream": stream_number},
            owner_task=owner_task,
            acknowledged_filters=filters,
        )

    async def receive(self, handle: Any, *, deadline: float) -> McpSubscriptionEvent:
        assert deadline > time.monotonic()
        self.receive_count += 1
        stream_number = handle["stream"]
        events = self.events_by_stream[stream_number]
        if events:
            return events.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self, _handle: Any) -> None:
        self.close_count += 1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-libos-mcp-lifecycle-") as directory:
        root = Path(directory)
        database = root / "lifecycle.sqlite"
        broker = InMemoryMcpCredentialBroker()
        task_created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        continuation_provider = ScriptedContinuationProvider()
        tasks_provider = ScriptedTasksProvider(task_created_at=task_created_at)
        subscription_provider = ScriptedSubscriptionProvider()
        config = replace(
            DEFAULT_CONFIG,
            mcp=replace(
                DEFAULT_CONFIG.mcp,
                tasks_extension_enabled=True,
                tasks_extension_spec_sha256=TASKS_SPEC_SHA256,
                remote_task_poll_min_interval_s=0.000001,
            ),
        )

        runtime = Runtime.open(
            database,
            config=config,
            substrate=_substrate(
                root,
                broker,
                continuation_provider,
                tasks_provider,
                subscription_provider,
            ),
        )
        initial_call_counts: dict[str, int]
        task_call_counts_before_restart: tuple[int, int, int]
        subscription_listens_before_restart: int
        try:
            continuation_provider.runtime = runtime
            tasks_provider.runtime = runtime
            subscription_provider.runtime = runtime
            runtime.mcp.register_server(
                _manifest(),
                actor=HOST_ACTOR,
                require_capability=False,
            )
            pid = _spawn_and_grant(runtime)
            tool_provider = _install_scripted_tool_provider(
                runtime,
                task_created_at=task_created_at,
            )

            pending = runtime.mcp.call_tool(
                pid,
                SERVER_ID,
                "review",
                {"document": "release-notes"},
            )
            if not isinstance(pending, McpInputRequired):
                raise RuntimeError("review Tool did not yield InputRequired")
            _require_human_binding(pending)

            input_task = runtime.mcp.call_tool(
                pid,
                SERVER_ID,
                "begin-task",
                {"mode": "input"},
            )
            if not isinstance(input_task, McpRemoteTask):
                raise RuntimeError("task Tool did not yield a remote Task")
            _grant_task(runtime, pid, input_task.task_ref)
            waiting = runtime.mcp.get_remote_task(
                input_task.task_ref,
                expected_revision=input_task.revision,
                actor=HOST_ACTOR,
            )
            _require_human_binding(waiting)
            updated = runtime.mcp.update_remote_task(
                waiting.task_ref,
                expected_revision=waiting.revision,
                responses=_approved_local_answer(),
                human_request_id=waiting.human_request_id,
                human_expected_revision=waiting.human_revision,
                human_preview_sha256=waiting.human_preview_sha256,
                actor=HOST_ACTOR,
            )
            completed_task = runtime.mcp.get_remote_task(
                updated.task_ref,
                expected_revision=updated.revision,
                actor=HOST_ACTOR,
            )

            cancel_task = runtime.mcp.call_tool(
                pid,
                SERVER_ID,
                "begin-task",
                {"mode": "cancel"},
            )
            if not isinstance(cancel_task, McpRemoteTask):
                raise RuntimeError("cancel fixture did not yield a remote Task")
            _grant_task(runtime, pid, cancel_task.task_ref)
            cancel_requested = runtime.mcp.cancel_remote_task(
                cancel_task.task_ref,
                expected_revision=cancel_task.revision,
                actor=HOST_ACTOR,
            )
            cancelled_task = runtime.mcp.get_remote_task(
                cancel_requested.task_ref,
                expected_revision=cancel_requested.revision,
                actor=HOST_ACTOR,
            )

            stopped_stream = runtime.mcp.start_subscription(
                SERVER_ID,
                filters=("resourcesListChanged",),
                actor=HOST_ACTOR,
            )
            events = _wait_for_events(runtime, stopped_stream.subscription_id)
            stopped_stream = runtime.mcp.stop_subscription(
                stopped_stream.subscription_id,
                actor=HOST_ACTOR,
            )
            interrupted_stream = runtime.mcp.start_subscription(
                SERVER_ID,
                filters=("resourcesListChanged",),
                actor=HOST_ACTOR,
            )
            if interrupted_stream.status is not McpSubscriptionStatus.ACTIVE:
                raise RuntimeError("restart fixture subscription did not become active")
            queued_event_before_restart = _wait_for_queued_event(
                runtime,
                interrupted_stream.subscription_id,
            )

            initial_call_counts = dict(tool_provider.calls)
            task_call_counts_before_restart = _task_call_counts(tasks_provider)
            subscription_listens_before_restart = subscription_provider.listen_count
        finally:
            runtime.close()

        missing_broker_failed_closed = _missing_broker_recovery_fails_closed(
            database,
            root=root,
            config=config,
            continuation_id=pending.continuation_id,
            task_created_at=task_created_at,
        )
        replay_provider = RejectInitialReplayProvider()
        reopened = Runtime.open(
            database,
            config=config,
            substrate=_substrate(
                root,
                broker,
                continuation_provider,
                tasks_provider,
                subscription_provider,
                tool_provider=replay_provider,
            ),
        )
        try:
            continuation_provider.runtime = reopened
            tasks_provider.runtime = reopened
            subscription_provider.runtime = reopened
            if replay_provider.calls != 0:
                raise RuntimeError("restart replayed an initial MCP Tool")
            recovered_pending = reopened.mcp.get_continuation(
                pending.continuation_id,
                actor=HOST_ACTOR,
            )
            _require_human_binding(recovered_pending)
            if continuation_provider.calls != 0:
                raise RuntimeError("restart dispatched a continuation automatically")
            if _task_call_counts(tasks_provider) != task_call_counts_before_restart:
                raise RuntimeError("restart polled or replayed a remote Task")
            if subscription_provider.listen_count != subscription_listens_before_restart:
                raise RuntimeError("restart re-opened a subscription automatically")
            task_calls_after_restart = sum(
                current - previous
                for current, previous in zip(
                    _task_call_counts(tasks_provider),
                    task_call_counts_before_restart,
                    strict=True,
                )
            )
            subscription_listens_after_restart = (
                subscription_provider.listen_count
                - subscription_listens_before_restart
            )

            lost_stream = reopened.mcp.subscription_status(
                interrupted_stream.subscription_id,
                actor=HOST_ACTOR,
            )
            try:
                reopened.mcp.subscription_events(
                    interrupted_stream.subscription_id,
                    actor=HOST_ACTOR,
                )
            except NotFound:
                reopened_events = "unavailable"
            else:
                raise RuntimeError("restart unexpectedly restored an event queue")
            if (
                lost_stream.status is not McpSubscriptionStatus.LOST
                or lost_stream.lost_reason != "runtime_restart"
            ):
                raise RuntimeError("subscription restart did not fail closed as lost")

            review_result = reopened.mcp.respond_continuation(
                recovered_pending.continuation_id,
                expected_revision=recovered_pending.revision,
                responses=_approved_local_answer(),
                human_request_id=recovered_pending.human_request_id,
                human_expected_revision=recovered_pending.human_revision,
                human_preview_sha256=recovered_pending.human_preview_sha256,
                actor=HOST_ACTOR,
            )
            if not isinstance(review_result, McpComplete):
                raise RuntimeError("explicit continuation did not complete")
            if continuation_provider.calls != 1:
                raise RuntimeError("explicit continuation was not dispatched exactly once")

            effect_operations = sorted(
                {
                    effect.operation
                    for effect in reopened.store.list_external_effects()
                    if effect.provider == "mcp"
                }
            )
            audit_actions = sorted(
                {
                    record.action
                    for record in reopened.audit.trace()
                    if record.action.startswith("primitive.mcp")
                }
            )
            evidence = json.dumps(
                {
                    "effects": [
                        effect.provider_metadata
                        for effect in reopened.store.list_external_effects()
                        if effect.provider == "mcp"
                    ],
                    "audit": [
                        record.decision
                        for record in reopened.audit.trace()
                        if record.action.startswith("primitive.mcp")
                    ],
                },
                sort_keys=True,
            )
            _assert_private_state_absent(evidence.encode("utf-8"), "evidence")

            output = {
                "schema_version": 1,
                "mrtr": {
                    "continuation_id": recovered_pending.continuation_id,
                    "human_request_id": recovered_pending.human_request_id,
                    "initial_result": pending.kind,
                    "reopened_result": recovered_pending.kind,
                    "explicit_response_result": review_result.kind,
                    "initial_tool_dispatches": initial_call_counts["review"],
                    "continuation_dispatches": continuation_provider.calls,
                    "automatic_initial_replay": replay_provider.calls != 0,
                },
                "remote_tasks": {
                    "input_task_ref": input_task.task_ref,
                    "input_flow": [
                        input_task.status.value,
                        waiting.status.value,
                        updated.status.value,
                        completed_task.status.value,
                    ],
                    "cancel_task_ref": cancel_task.task_ref,
                    "cancel_flow": [
                        cancel_task.status.value,
                        cancel_requested.status.value,
                        cancelled_task.status.value,
                    ],
                    "provider_calls_after_restart": task_calls_after_restart,
                    "automatic_poll_or_replay": task_calls_after_restart != 0,
                },
                "subscriptions": {
                    "event_sequences": [event.sequence for event in events],
                    "event_provenance": [event.provenance for event in events],
                    "explicit_stop_status": stopped_stream.status.value,
                    "reopened_status": lost_stream.status.value,
                    "lost_reason": lost_stream.lost_reason,
                    "reopened_events": reopened_events,
                    "queued_event_before_restart": queued_event_before_restart,
                    "automatic_relisten": subscription_listens_after_restart != 0,
                },
                "recovery": {
                    "missing_broker_failed_closed": missing_broker_failed_closed,
                    "raw_remote_state_in_sqlite": None,
                    "protected_effect_operations": effect_operations,
                    "audit_actions": audit_actions,
                },
            }
        finally:
            reopened.close()

        raw_remote_state_absent = _sqlite_has_no_private_state(database)
        if not raw_remote_state_absent:
            raise RuntimeError("private MCP Provider state escaped into SQLite")
        output["recovery"]["raw_remote_state_in_sqlite"] = (
            not raw_remote_state_absent
        )
        encoded = json.dumps(output, indent=2, sort_keys=True)
        _assert_private_state_absent(encoded.encode("utf-8"), "public output")
        broker.close()
        print(encoded)
    return 0


def _manifest() -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id=SERVER_ID,
        transport="streamable_http",
        http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
        timeout_s=2.0,
        max_request_bytes=16_384,
        max_response_bytes=16_384,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        tools=(
            McpToolSpec(
                tool_id="review",
                mcp_name="demo.review",
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
            McpToolSpec(
                tool_id="begin-task",
                mcp_name="demo.begin_task",
                right="execute",
                rollback_class="unknown",
                rollback_status="unknown",
                state_mutation=True,
                information_flow=True,
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
        resources=(
            McpResourceSpec(
                resource_id="status",
                remote_uri="file:///provider/status",
            ),
        ),
        subscriptions=("resourcesListChanged",),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=TASKS_SPEC_SHA256,
        ),
    )


def _substrate(
    root: Path,
    broker: InMemoryMcpCredentialBroker,
    continuation_provider: ScriptedContinuationProvider,
    tasks_provider: ScriptedTasksProvider,
    subscription_provider: ScriptedSubscriptionProvider,
    *,
    tool_provider: Any | None = None,
) -> LocalResourceProviderSubstrate:
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    selected = LocalResourceProviderSubstrate(workspace)
    selected.mcp_credential_broker = broker
    selected.mcp_continuation_provider = continuation_provider
    selected.mcp_tasks_provider = tasks_provider
    selected.mcp_subscription_provider = subscription_provider
    if tool_provider is not None:
        selected.mcp_v3_tool_provider = tool_provider
    return selected


def _spawn_and_grant(runtime: Runtime) -> str:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="exercise the deterministic MCP modern lifecycle",
        resource_budget=ResourceBudget(max_mcp_bytes=1_000_000),
    )
    for resource, rights in (
        (f"mcp:{SERVER_ID}:review", [CapabilityRight.READ]),
        (f"mcp:{SERVER_ID}:begin-task", [CapabilityRight.EXECUTE]),
        (f"mcp_server:{SERVER_ID}", [CapabilityRight.EXECUTE]),
        ("human:owner", [CapabilityRight.WRITE]),
    ):
        runtime.capability.grant(pid, resource, rights, issued_by=HOST_ACTOR)
    return pid


def _install_scripted_tool_provider(
    runtime: Runtime,
    *,
    task_created_at: str,
) -> ScriptedToolProvider:
    # Fixture-only Host composition: reuse the adapter that RuntimeBuilder has
    # already bound to its durable continuation/task managers.  Production
    # clients use the built-in governed SDK provider and never replace it.
    built_in = runtime._mcp_v3_tool_provider  # noqa: SLF001
    result_adapter = getattr(built_in, "result_adapter", None)
    if result_adapter is None:
        raise RuntimeError("the Runtime did not compose its modern result adapter")
    selected = ScriptedToolProvider(
        result_adapter,
        runtime,
        task_created_at=task_created_at,
    )
    runtime.mcp._modern_tool_provider = selected  # noqa: SLF001
    return selected


def _grant_task(runtime: Runtime, pid: str, task_ref: str) -> None:
    runtime.capability.grant(
        pid,
        f"mcp_task:{task_ref}",
        [CapabilityRight.READ, CapabilityRight.WRITE],
        issued_by=HOST_ACTOR,
    )


def _approved_local_answer() -> dict[str, Any]:
    # The Host submits only the local input id.  Provider request keys are
    # recovered inside the broker-bound continuation/task manager.
    return {
        "input-1": {
            "action": "accept",
            "content": {"approved": True},
        }
    }


def _require_human_binding(result: McpInputRequired | McpRemoteTask) -> None:
    if (
        result.human_request_id is None
        or result.human_revision is None
        or result.human_preview_sha256 is None
    ):
        raise RuntimeError("MCP input-required result lacks its Human receipt binding")


def _wait_for_events(
    runtime: Runtime,
    subscription_id: str,
) -> tuple[McpSubscriptionEvent, ...]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        selected = runtime.mcp.subscription_events(
            subscription_id,
            actor=HOST_ACTOR,
        )
        if selected:
            return selected
        time.sleep(0.005)
    raise RuntimeError("deterministic subscription event did not arrive")


def _wait_for_queued_event(runtime: Runtime, subscription_id: str) -> bool:
    """Confirm the second stream has an unread in-memory event before close."""

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        record = runtime.uow.mcp_subscriptions.get(subscription_id)
        if record is not None and record.received_count >= 1:
            return True
        time.sleep(0.005)
    raise RuntimeError("restart fixture did not queue its unread notification")


def _missing_broker_recovery_fails_closed(
    database: Path,
    *,
    root: Path,
    config: Any,
    continuation_id: str,
    task_created_at: str,
) -> bool:
    """Exercise missing volatile custody on an isolated copy of durable state."""

    selected_database = root / "missing-broker.sqlite"
    shutil.copy2(database, selected_database)
    empty_broker = InMemoryMcpCredentialBroker()
    replay_provider = RejectInitialReplayProvider()
    selected = Runtime.open(
        selected_database,
        config=config,
        substrate=_substrate(
            root,
            empty_broker,
            ScriptedContinuationProvider(),
            ScriptedTasksProvider(task_created_at=task_created_at),
            ScriptedSubscriptionProvider(),
            tool_provider=replay_provider,
        ),
    )
    try:
        if replay_provider.calls != 0:
            raise RuntimeError("missing-broker recovery replayed an initial Tool")
        try:
            selected.mcp.get_continuation(continuation_id, actor=HOST_ACTOR)
        except ValidationError as error:
            if "unavailable" not in str(error).casefold():
                raise RuntimeError(
                    "missing-broker recovery failed for an unrelated reason"
                ) from error
            return True
        raise RuntimeError("missing MCP broker state did not fail closed")
    finally:
        selected.close()
        empty_broker.close()


def _assert_pending_effect(runtime: Runtime, operation: str) -> None:
    effects = [
        effect
        for effect in runtime.store.list_external_effects()
        if effect.provider == "mcp" and effect.operation == operation
    ]
    if not effects or effects[-1].effect_state != "pending":
        raise RuntimeError(f"MCP {operation} provider ran before its pending effect")


def _task_call_counts(provider: ScriptedTasksProvider) -> tuple[int, int, int]:
    return provider.get_calls, provider.update_calls, provider.cancel_calls


def _sqlite_has_no_private_state(database: Path) -> bool:
    for path in database.parent.glob(f"{database.name}*"):
        if path.is_file() and any(
            sentinel.encode("utf-8") in path.read_bytes()
            for sentinel in PRIVATE_SENTINELS
        ):
            return False
    return True


def _assert_private_state_absent(payload: bytes, label: str) -> None:
    for sentinel in PRIVATE_SENTINELS:
        if sentinel.encode("utf-8") in payload:
            raise RuntimeError(f"private MCP Provider state escaped into {label}")


if __name__ == "__main__":
    raise SystemExit(main())
