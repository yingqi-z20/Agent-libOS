from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp.oauth import InMemoryMcpCredentialBroker
from agent_libos.mcp.resources import inert_resource_handle
from agent_libos.mcp.types import (
    McpComplete,
    McpRemoteTask,
    McpRemoteTaskStatus,
    McpSubscriptionStatus,
    McpTextContent,
)
from agent_libos.models import CapabilityRight, ResourceBudget
from agent_libos.substrate import LocalResourceProviderSubstrate


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]

ROOT = Path(__file__).resolve().parents[2]
PYTHON_FIXTURE = ROOT / "tests" / "fixtures" / "mcp_sdk_v2" / "python_server.py"
TYPESCRIPT_FIXTURE_ROOT = (
    ROOT / "tests" / "fixtures" / "mcp_sdk_v2" / "typescript_server"
)
TYPESCRIPT_FIXTURE = TYPESCRIPT_FIXTURE_ROOT / "server.mjs"
RESOURCE_URI = "fixture://document/current"
TASKS_SCHEMA = ROOT / "tests" / "fixtures" / "mcp_sdk_v2" / "tasks_extension_schema.json"


def _substrate(
    root: Path,
    broker: InMemoryMcpCredentialBroker,
) -> LocalResourceProviderSubstrate:
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.mcp_credential_broker = broker
    return substrate


@pytest.mark.parametrize("fixture", ("python-sdk-v2", "typescript-sdk-v2"))
def test_real_sdk_v2_resources_and_prompts_use_runtime_protected_path(
    fixture: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_failures: list[dict[str, object]] = []
    original_new_event_loop = asyncio.events.new_event_loop

    def monitored_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = original_new_event_loop()
        loop.set_exception_handler(
            lambda _loop, context: loop_failures.append(dict(context))
        )
        return loop

    monkeypatch.setattr(asyncio.events, "new_event_loop", monitored_new_event_loop)
    command, args, content_marker = _fixture_command(fixture)
    server_id = fixture
    runtime = Runtime.open(":memory:")
    try:
        runtime.mcp.register_server(
            _runtime_manifest(server_id, command, args),
            actor="test-host",
            require_capability=False,
        )

        resources = runtime.mcp.list_resources(server_id)
        assert [item.resource_id for item in resources.items] == ["document"]
        templates = runtime.mcp.list_resource_templates(server_id)
        assert [item.template_id for item in templates.items] == [
            "document-template"
        ]

        current = runtime.mcp.read_resource(server_id, "document")
        assert isinstance(current, McpComplete)
        assert current.value is not None
        assert current.value.resource_id == "document"
        assert len(current.value.contents) == 1
        assert isinstance(current.value.contents[0], McpTextContent)
        assert content_marker in current.value.contents[0].text
        assert "revision=1" in current.value.contents[0].text

        named = runtime.mcp.read_resource(
            server_id,
            "document-template",
            variables={"name": "Ada Lovelace"},
        )
        assert isinstance(named, McpComplete)
        assert named.value is not None
        assert named.value.resource_id == "document-template"
        assert len(named.value.contents) == 1
        assert isinstance(named.value.contents[0], McpTextContent)
        assert (
            f"{content_marker} name=Ada Lovelace"
            in named.value.contents[0].text.replace("%20", " ")
        )

        prompts = runtime.mcp.list_prompts(server_id)
        assert [item.prompt_id for item in prompts.items] == ["review"]
        preview = runtime.mcp.get_prompt(
            server_id,
            "review",
            arguments={"focus": "authority boundaries"},
        )
        assert isinstance(preview, McpComplete)
        assert preview.preview_sha256 is not None
        assert preview.value is not None
        assert preview.value.prompt_id == "review"
        assert len(preview.value.messages) == 1
        assert isinstance(preview.value.messages[0].content, McpTextContent)
        assert "authority boundaries" in preview.value.messages[0].content.text

        assert any(
            record.action == "primitive.mcp.resources.read"
            for record in runtime.audit.trace()
        )
        assert any(
            record.action == "primitive.mcp.prompts.get"
            for record in runtime.audit.trace()
        )
        assert runtime.mcp.inspect_server(
            server_id,
            require_capability=False,
        )["schema_version"] == 3
        assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()
    finally:
        runtime.close()
    assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()
    assert loop_failures == []


@pytest.mark.parametrize("fixture", ("python-sdk-v2", "typescript-sdk-v2"))
def test_real_sdk_v2_subscription_uses_runtime_protected_path(
    fixture: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    loop_failures: list[dict[str, object]] = []
    original_new_event_loop = asyncio.events.new_event_loop

    def monitored_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = original_new_event_loop()
        loop.set_exception_handler(
            lambda _loop, context: loop_failures.append(dict(context))
        )
        return loop

    monkeypatch.setattr(asyncio.events, "new_event_loop", monitored_new_event_loop)
    command, args, _content_marker = _fixture_command(fixture)
    server_id = f"{fixture}-subscription"
    runtime = Runtime.open(":memory:")
    loss_reasons: list[str] = []
    subscription_manager = runtime._mcp_subscription_manager
    original_connection_lost = subscription_manager._connection_lost

    async def record_connection_loss(subscription_id: str, reason: str) -> None:
        loss_reasons.append(reason)
        await original_connection_lost(subscription_id, reason)

    monkeypatch.setattr(
        subscription_manager,
        "_connection_lost",
        record_connection_loss,
    )
    subscription_id: str | None = None
    try:
        runtime.mcp.register_server(
            _runtime_manifest(server_id, command, args),
            actor="test-host",
            require_capability=False,
        )
        subscription = runtime.mcp.start_subscription(
            server_id,
            filters=("resourceSubscriptions",),
            actor="test-host",
        )
        subscription_id = subscription.subscription_id
        assert subscription.status is McpSubscriptionStatus.ACTIVE
        assert subscription.acknowledged_filters == ("resourceSubscriptions",)
        assert runtime.mcp.subscription_status(
            subscription_id,
            actor="test-host",
        ).status is McpSubscriptionStatus.ACTIVE
        assert len(anyio.run(runtime._mcp_connection_supervisor.snapshot)) == 1

        deadline = time.monotonic() + 5
        events = ()
        while time.monotonic() < deadline:
            events = runtime.mcp.subscription_events(
                subscription_id,
                after=0,
                limit=10,
                actor="test-host",
            )
            if events:
                break
            time.sleep(0.02)
        assert len(events) >= 1
        event = events[0]
        assert event.sequence == 1
        assert event.event_type == "resourceUpdated"
        assert event.payload == {
            "resource_handle": inert_resource_handle(server_id, RESOURCE_URI)
        }
        assert event.provenance == "untrusted_mcp_notification"

        stopped = runtime.mcp.stop_subscription(
            subscription_id,
            actor="test-host",
        )
        subscription_id = None
        assert loss_reasons == []
        assert stopped.status is McpSubscriptionStatus.CLOSED
        assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()
        actions = {record.action for record in runtime.audit.trace()}
        assert {
            "primitive.mcp.subscriptions.start",
            "primitive.mcp.subscriptions.events",
            "primitive.mcp.subscriptions.stop",
        }.issubset(actions)
    finally:
        if subscription_id is not None:
            runtime.mcp.stop_subscription(subscription_id, actor="test-host")
        runtime.close()
    assert anyio.run(runtime._mcp_connection_supervisor.snapshot) == ()
    assert loop_failures == []
    # In particular, the opening/owner futures must be consumed when the
    # Runtime closes the SDK listen scope.  An unobserved deadline exception
    # otherwise bypasses the loop handler during interpreter/loop teardown.
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("fixture", ("python-sdk-v2", "typescript-sdk-v2"))
def test_real_sdk_v2_task_subscription_projects_only_local_refs(
    fixture: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    command, fixture_args, _content_marker = _fixture_command(fixture)
    task_state_file = tmp_path / f"{fixture}-subscription-task-state.json"
    digest = hashlib.sha256(TASKS_SCHEMA.read_bytes()).hexdigest()
    secret_source = "AGENT_LIBOS_MCP_TASK_NOTIFICATION_SECRET"
    secret = "task-notification-secret-must-not-escape"
    monkeypatch.setenv(secret_source, secret)
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
        )
    )
    broker = InMemoryMcpCredentialBroker()
    substrate = _substrate(tmp_path, broker)
    runtime = Runtime.open(":memory:", substrate=substrate, config=config)
    subscription_id: str | None = None
    try:
        arguments = [*fixture_args, "--task-state-file", str(task_state_file)]
        manifest = _runtime_manifest(
            fixture,
            command,
            arguments,
            tasks_extension_sha256=digest,
        )
        stdio = manifest["stdio"]
        assert isinstance(stdio, dict)
        stdio["env"] = {"MCP_FIXTURE_TASK_SECRET": secret_source}
        runtime.mcp.register_server(
            manifest,
            actor="test-host",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="receive an inert exact-v3 Task notification",
            resource_budget=ResourceBudget(max_mcp_bytes=10_000_000),
        )
        for resource, rights in (
            (f"mcp:{fixture}:review-task", [CapabilityRight.READ]),
            (f"mcp:{fixture}:subscription:catalog", [CapabilityRight.WRITE]),
            (f"mcp_server:{fixture}", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
            ("process:spawn", [CapabilityRight.WRITE]),
            (
                runtime.mcp.stdio_resource_for_argv(
                    command,
                    arguments,
                    env={"MCP_FIXTURE_TASK_SECRET": secret_source},
                ),
                [CapabilityRight.EXECUTE],
            ),
        ):
            runtime.capability.grant(pid, resource, rights, issued_by="test-host")

        task = runtime.mcp.call_tool(
            pid,
            fixture,
            "review-task",
            {"mode": "input"},
        )
        assert isinstance(task, McpRemoteTask)
        state = json.loads(task_state_file.read_text(encoding="utf-8"))
        remote_ids = tuple(state["tasks"])
        assert len(remote_ids) == 1
        remote_id = remote_ids[0]

        subscription = runtime.mcp.start_subscription(
            fixture,
            filters=("taskIds",),
            actor=pid,
        )
        subscription_id = subscription.subscription_id
        assert subscription.acknowledged_filters == ("taskIds",)
        runtime.capability.grant(
            pid,
            f"mcp_subscription:{subscription_id}",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test-host",
        )
        deadline = time.monotonic() + 5
        events = ()
        while time.monotonic() < deadline:
            events = runtime.mcp.subscription_events(
                subscription_id,
                after=0,
                limit=1,
                actor=pid,
            )
            if events:
                break
            time.sleep(0.02)
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "taskStatus"
        assert event.payload == {
            "task_ref": task.task_ref,
            "status": "working",
            "created_at": "2030-01-01T00:00:00.000000+00:00",
            "last_updated_at": "2030-01-01T00:00:01.000000+00:00",
            "ttl_ms": 60_000,
            "poll_interval_ms": 0,
            "status_message": "fixture update [redacted]",
        }
        durable = runtime.uow.mcp_subscriptions.get(subscription_id)
        projected = repr((event, durable, runtime.audit.trace()))
        assert remote_id not in projected
        assert secret not in projected
        assert "ui/resourceUri" not in projected
        assert "ui/visibility" not in projected
        stopped = runtime.mcp.stop_subscription(subscription_id, actor=pid)
        subscription_id = None
        assert stopped.status is McpSubscriptionStatus.CLOSED
    finally:
        if subscription_id is not None:
            runtime.mcp.stop_subscription(subscription_id, actor=pid)
        runtime.close()
        broker.close()
    captured = capfd.readouterr()
    assert remote_id not in captured.out + captured.err
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("fixture", ("python-sdk-v2", "typescript-sdk-v2"))
def test_real_sdk_v2_tasks_use_runtime_store_human_and_reopen_path(
    fixture: str,
    tmp_path: Path,
) -> None:
    command, fixture_args, _content_marker = _fixture_command(fixture)
    task_state_file = tmp_path / f"{fixture}-remote-task-state.json"
    digest = hashlib.sha256(TASKS_SCHEMA.read_bytes()).hexdigest()
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.001,
        )
    )
    broker = InMemoryMcpCredentialBroker()
    substrate = _substrate(tmp_path, broker)
    database = tmp_path / f"{fixture}-runtime.sqlite"
    manifest = _runtime_manifest(
        fixture,
        command,
        [*fixture_args, "--task-state-file", str(task_state_file)],
        tasks_extension_sha256=digest,
    )

    initial = Runtime.open(database, substrate=substrate, config=config)
    try:
        initial.mcp.register_server(
            manifest,
            actor="test-host",
            require_capability=False,
        )
        pid = initial.process.spawn(
            image="base-agent:v0",
            goal="capture exact-v3 Tasks handles",
            resource_budget=ResourceBudget(max_mcp_bytes=10_000_000),
        )
        for resource, rights in (
            (f"mcp:{fixture}:review-task", [CapabilityRight.READ]),
            (f"mcp_server:{fixture}", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
            ("process:spawn", [CapabilityRight.WRITE]),
            (
                initial.mcp.stdio_resource_for_argv(
                    command,
                    [*fixture_args, "--task-state-file", str(task_state_file)],
                ),
                [CapabilityRight.EXECUTE],
            ),
        ):
            initial.capability.grant(pid, resource, rights, issued_by="test-host")
        input_task = initial.mcp.call_tool(
            pid,
            fixture,
            "review-task",
            {"mode": "input"},
        )
        cancel_task = initial.mcp.call_tool(
            pid,
            fixture,
            "review-task",
            {"mode": "cancel"},
        )
        assert isinstance(input_task, McpRemoteTask)
        assert isinstance(cancel_task, McpRemoteTask)
        assert input_task.status is McpRemoteTaskStatus.WORKING
        assert cancel_task.status is McpRemoteTaskStatus.WORKING
        assert "private-task" not in repr((input_task, cancel_task))
        for task in (input_task, cancel_task):
            initial.capability.grant(
                pid,
                f"mcp_task:{task.task_ref}",
                [CapabilityRight.READ, CapabilityRight.WRITE],
                issued_by="test-host",
            )
    finally:
        initial.close()

    state = json.loads(task_state_file.read_text(encoding="utf-8"))
    assert state["counter"] == 2
    assert len(state["tasks"]) == 2
    private_ids = tuple(state["tasks"])
    stored = database.read_bytes()
    assert all(remote_id.encode("utf-8") not in stored for remote_id in private_ids)

    reopened_substrate = _substrate(tmp_path, broker)
    reopened = Runtime.open(database, substrate=reopened_substrate, config=config)
    try:
        awaiting = reopened.mcp.get_remote_task(
            input_task.task_ref,
            expected_revision=input_task.revision,
            actor="test-host",
        )
        assert awaiting.status is McpRemoteTaskStatus.INPUT_REQUIRED
        assert len(awaiting.input_requests) == 1
        assert awaiting.human_request_id is not None
        assert awaiting.human_revision is not None
        assert awaiting.human_preview_sha256 is not None
        working = reopened.mcp.update_remote_task(
            input_task.task_ref,
            expected_revision=awaiting.revision,
            responses={
                awaiting.input_requests[0].request_id: {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
            human_request_id=awaiting.human_request_id,
            human_expected_revision=awaiting.human_revision,
            human_preview_sha256=awaiting.human_preview_sha256,
            actor="test-host",
        )
        assert working.status is McpRemoteTaskStatus.WORKING
        completed = reopened.mcp.get_remote_task(
            input_task.task_ref,
            expected_revision=working.revision,
            actor="test-host",
        )
        assert completed.status is McpRemoteTaskStatus.COMPLETED
        assert completed.result == {
            "approved": True,
            "source": f"{fixture}-tasks-extension",
        }

        cancel_requested = reopened.mcp.cancel_remote_task(
            cancel_task.task_ref,
            expected_revision=cancel_task.revision,
            actor="test-host",
        )
        assert cancel_requested.status is McpRemoteTaskStatus.CANCEL_REQUESTED
        cancelled = reopened.mcp.get_remote_task(
            cancel_task.task_ref,
            expected_revision=cancel_requested.revision,
            actor="test-host",
        )
        assert cancelled.status is McpRemoteTaskStatus.CANCELLED
        assert anyio.run(reopened._mcp_connection_supervisor.snapshot) == ()
        actions = [record.action for record in reopened.audit.trace()]
        assert actions.count("primitive.mcp.tasks.get") == 3
        assert actions.count("primitive.mcp.tasks.update") == 1
        assert actions.count("primitive.mcp.tasks.cancel") == 1
    finally:
        reopened.close()
        broker.close()
    assert anyio.run(reopened._mcp_connection_supervisor.snapshot) == ()


@pytest.mark.parametrize("fixture", ("python-sdk-v2", "typescript-sdk-v2"))
def test_real_sdk_v2_fixture_supports_modern_subscription_transport(
    fixture: str,
) -> None:
    """Keep the cross-language wire fixture honest until Runtime binds listen."""

    command, args, content_marker = _fixture_command(fixture)
    anyio.run(_exercise_subscription_fixture, command, args, content_marker)


def _fixture_command(fixture: str) -> tuple[str, list[str], str]:
    if fixture == "python-sdk-v2":
        command = sys.executable
        args = [str(PYTHON_FIXTURE)]
        content_marker = "python-sdk-v2"
    else:
        command = shutil.which("node") or ""
        assert command, "the TypeScript MCP SDK v2 fixture requires Node 24"
        installed_server = (
            TYPESCRIPT_FIXTURE_ROOT
            / "node_modules"
            / "@modelcontextprotocol"
            / "server"
            / "package.json"
        )
        assert installed_server.is_file(), (
            "the frozen TypeScript MCP SDK v2 fixture is not installed; run "
            "`npm --prefix tests/fixtures/mcp_sdk_v2/typescript_server ci "
            "--ignore-scripts --no-audit --no-fund`"
        )
        args = [str(TYPESCRIPT_FIXTURE)]
        content_marker = "typescript-sdk-v2"
    return command, args, content_marker


def _runtime_manifest(
    server_id: str,
    command: str,
    args: list[str],
    *,
    tasks_extension_sha256: str | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 3,
        "server_id": server_id,
        "transport": "stdio",
        "protocol_mode": "2026-07-28",
        "stdio": {"command": command, "args": args},
        "resources": [
            {
                "resource_id": "document",
                "remote_uri": RESOURCE_URI,
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": ["text/plain"],
            }
        ],
        "resource_templates": [
            {
                "template_id": "document-template",
                "remote_uri_template": "fixture://document/{name}",
                "variables": ["name"],
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": ["text/plain"],
            }
        ],
        "prompts": [
            {
                "prompt_id": "review",
                "mcp_name": "review_document",
                "argument_names": ["focus"],
            }
        ],
        "subscriptions": ["resourceSubscriptions"],
        "timeout_s": 10,
        "max_request_bytes": 65_536,
        "max_response_bytes": 1_048_576,
    }
    if tasks_extension_sha256 is not None:
        manifest["subscriptions"] = ["resourceSubscriptions", "taskIds"]
        manifest["tools"] = [
            {
                "tool_id": "review-task",
                "mcp_name": "begin_review_task",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "rollback_status": "not_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["input", "cancel"]}
                    },
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            }
        ]
        manifest["tasks_extension"] = {
            "extension_id": "io.modelcontextprotocol/tasks",
            "spec_sha256": tasks_extension_sha256,
        }
    return manifest


async def _exercise_subscription_fixture(
    command: str,
    args: list[str],
    content_marker: str,
) -> None:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.subscriptions import ResourceUpdated

    transport = stdio_client(
        StdioServerParameters(command=command, args=args, env={}),
        # Upstream binds its default ``errlog`` to ``sys.stderr`` at import
        # time. Runtime subscription tests intentionally import the SDK while
        # pytest capture is active; that capture object is closed at fixture
        # teardown. Bind the currently owned live stream for this independent
        # direct-client lifecycle instead of reusing the stale default object.
        errlog=sys.stderr,
    )
    client = Client(transport, mode="auto", raise_exceptions=True)
    async with client:
        assert client.session.protocol_version == "2026-07-28"

        first = await client.read_resource(RESOURCE_URI, cache_mode="reload")
        assert len(first.contents) == 1
        assert content_marker in first.contents[0].text
        assert "revision=1" in first.contents[0].text

        async with client.listen(resource_subscriptions=[RESOURCE_URI]) as subscription:
            assert subscription.honored.resource_subscriptions == [RESOURCE_URI]
            update = await client.call_tool("publish_resource_update", {})
            assert not update.is_error
            assert update.structured_content == {
                "uri": RESOURCE_URI,
                "revision": 2,
            }
            with anyio.fail_after(5):
                event = await anext(subscription)
            assert event == ResourceUpdated(uri=RESOURCE_URI)

        second = await client.read_resource(RESOURCE_URI, cache_mode="reload")
        assert "revision=2" in second.contents[0].text
