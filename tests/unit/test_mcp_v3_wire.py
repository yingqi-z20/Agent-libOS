from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import pytest

from agent_libos.mcp._input import sanitize_provider_json
from agent_libos.mcp.client import (
    bind_mcp_client_binding,
    McpClientBinding,
    McpContinuationSurfaceUnsupported,
    McpModernClient,
    McpSdkV2ResultAdapter,
    McpSdkV2SessionProvider,
    mcp_transport_spec_from_v3,
)
from agent_libos.mcp.manifest import parse_mcp_v3_manifest_mapping
from agent_libos.mcp.types import (
    McpArtifactReceipt,
    McpComplete,
    McpInputRequired,
)
from agent_libos.mcp.wire import (
    McpSdkV3ContinuationProvider,
    McpSdkV3TasksProvider,
    McpSdkV3ToolProvider,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import McpProtocolMode


_TASKS_SHA256 = hashlib.sha256(b"official pinned tasks extension").hexdigest()


def _manifest(*, tasks: bool = False):
    value: dict[str, Any] = {
        "schema_version": 3,
        "server_id": "wire-demo",
        "transport": "stdio",
        "protocol_mode": "2026-07-28",
        "timeout_s": 10,
        "max_request_bytes": 4096,
        "max_response_bytes": 8192,
        "stdio": {"command": "wire-demo", "args": [], "env": {}, "cwd": None},
        "tools": [
            {
                "tool_id": "logical-echo",
                "mcp_name": "provider.echo",
                "right": "execute",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }
        ],
        "resources": [
            {
                "resource_id": "logical-status",
                "remote_uri": "opaque://provider/status",
            }
        ],
        "prompts": [
            {
                "prompt_id": "logical-review",
                "mcp_name": "provider.review",
                "argument_names": ["subject"],
            }
        ],
    }
    if tasks:
        value["tasks_extension"] = {
            "extension_id": "io.modelcontextprotocol/tasks",
            "spec_sha256": _TASKS_SHA256,
        }
    return parse_mcp_v3_manifest_mapping(value)


class _Session:
    def __init__(self, *results: Any) -> None:
        self.protocol_version = "2026-07-28"
        self.results = list(results)
        self.tool_calls: list[dict[str, Any]] = []
        self.resource_calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []
        self.completion_calls: list[dict[str, Any]] = []
        self.raw_calls: list[dict[str, Any]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any):
        self.tool_calls.append({"name": name, "arguments": arguments, **kwargs})
        return self.results.pop(0)

    async def send_request(self, request: Any, _adapter: Any, **kwargs: Any):
        dumped = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.raw_calls.append({"request": dumped, **kwargs})
        return self.results.pop(0)

    async def read_resource(self, resource_name: str, **kwargs: Any) -> Any:
        self.resource_calls.append({"resource_name": resource_name, **kwargs})
        return self.results.pop(0)

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, str], **kwargs: Any
    ) -> Any:
        self.prompt_calls.append(
            {"prompt_name": prompt_name, "arguments": arguments, **kwargs}
        )
        return self.results.pop(0)

    async def complete(self, reference: Any, argument: Any, **kwargs: Any) -> Any:
        self.completion_calls.append(
            {"reference": reference, "argument": argument, **kwargs}
        )
        return self.results.pop(0)


class _Factory:
    def __init__(self, session: _Session, *, sensitive: tuple[str, ...] = ()) -> None:
        self.session = session
        self.sensitive = sensitive
        self.servers: list[Any] = []

    @asynccontextmanager
    async def __call__(self, server: Any, *, deadline: float):
        self.servers.append((server, deadline))
        yield self.session

    def sensitive_values(self, server_id: str) -> tuple[str, ...]:
        assert server_id == "wire-demo"
        return self.sensitive


def _deadline() -> float:
    return time.monotonic() + 10


def test_initial_tool_uses_only_manifest_name_and_exact_modern_flags() -> None:
    session = _Session(
        {
            "resultType": "complete",
            "content": [{"type": "text", "text": "credential-token result"}],
        }
    )
    factory = _Factory(session, sensitive=("credential-token",))
    provider = McpSdkV3ToolProvider(factory)

    result = asyncio.run(
        provider.call_tool(
            _manifest(),
            "logical-echo",
            {"text": "hello"},
            deadline=_deadline(),
        )
    )

    assert result == McpComplete(
        value={"content": [{"type": "text", "text": "[redacted] result"}]}
    )
    call = session.tool_calls[0]
    assert call["name"] == "provider.echo"
    assert call["arguments"] == {"text": "hello"}
    assert call["allow_input_required"] is True
    assert call["allow_claimed"] is False
    assert call["read_timeout_seconds"] > 0


def test_initial_tool_validates_allowlist_schema_and_tasks_pin_before_session() -> None:
    session = _Session({"resultType": "complete", "content": []})
    factory = _Factory(session)
    provider = McpSdkV3ToolProvider(factory)

    with pytest.raises(NotFound):
        asyncio.run(
            provider.call_tool(
                _manifest(), "provider.echo", {"text": "x"}, deadline=_deadline()
            )
        )
    with pytest.raises(ValidationError, match="required"):
        asyncio.run(
            provider.call_tool(
                _manifest(), "logical-echo", {}, deadline=_deadline()
            )
        )
    with pytest.raises(ValidationError, match="Host pin"):
        asyncio.run(
            provider.call_tool(
                _manifest(tasks=True),
                "logical-echo",
                {"text": "x"},
                deadline=_deadline(),
            )
        )
    assert factory.servers == []

    pinned_session = _Session({"resultType": "complete", "content": []})
    pinned = McpSdkV3ToolProvider(
        _Factory(pinned_session),
        host_tasks_extension_sha256=_TASKS_SHA256,
    )
    asyncio.run(
        pinned.call_tool(
            _manifest(tasks=True),
            "logical-echo",
            {"text": "x"},
            deadline=_deadline(),
        )
    )
    assert pinned_session.tool_calls[0]["allow_claimed"] is True


def test_continuation_wire_carries_bound_state_and_never_exposes_initial_api() -> None:
    session = _Session({"resultType": "complete", "content": []})
    provider = McpSdkV3ContinuationProvider(_Factory(session))
    server = mcp_transport_spec_from_v3(_manifest())

    result = asyncio.run(
        provider.continue_tool(
            server,
            "provider.echo",
            {"text": "hello"},
            {"remote-input": {"action": "decline"}},
            "opaque-state",
            deadline=_deadline(),
        )
    )

    assert result == {"resultType": "complete", "content": []}
    assert not hasattr(provider, "call_tool")
    assert session.tool_calls == [
        {
            "name": "provider.echo",
            "arguments": {"text": "hello"},
            "read_timeout_seconds": session.tool_calls[0]["read_timeout_seconds"],
            "input_responses": {"remote-input": {"action": "decline"}},
            "request_state": "opaque-state",
            "allow_input_required": True,
            "allow_claimed": False,
        }
    ]


def test_resource_continuation_resubmits_state_and_projects_safe_complete() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    session = _Session(
        mcp_types.InputRequiredResult(
            inputRequests={}, requestState="next-state"
        ),
        mcp_types.ReadResourceResult(
            contents=[
                mcp_types.TextResourceContents(
                    uri="opaque://provider/status",
                    text="credential-token status",
                )
            ]
        ),
    )
    provider = McpSdkV3ContinuationProvider(
        _Factory(session, sensitive=("credential-token",))
    )
    server = mcp_transport_spec_from_v3(_manifest())

    pending = asyncio.run(
        provider.continue_resource(
            server,
            "opaque://provider/status",
            "logical-status",
            {"confirm": {"action": "accept", "content": {"ok": True}}},
            "first-state",
            deadline=_deadline(),
        )
    )
    complete = asyncio.run(
        provider.continue_resource(
            server,
            "opaque://provider/status",
            "logical-status",
            {"confirm": {"action": "accept", "content": {"ok": True}}},
            "next-state",
            deadline=_deadline(),
        )
    )

    assert pending == {
        "resultType": "input_required",
        "inputRequests": {},
        "requestState": "next-state",
    }
    assert complete["resultType"] == "complete"
    assert complete["resource_id"] == "logical-status"
    assert complete["provenance"] == "untrusted_mcp_resource"
    assert complete["contents"][0]["text"] == "[redacted] status"
    assert session.resource_calls[0] == {
        "resource_name": "opaque://provider/status",
        "input_responses": {
            "confirm": {"action": "accept", "content": {"ok": True}}
        },
        "request_state": "first-state",
        "allow_input_required": True,
    }


def test_prompt_continuation_resubmits_arguments_and_keeps_untrusted_roles() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    session = _Session(
        mcp_types.GetPromptResult(
            description="review",
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text="Review it"),
                )
            ],
        )
    )
    provider = McpSdkV3ContinuationProvider(_Factory(session))

    complete = asyncio.run(
        provider.continue_prompt(
            mcp_transport_spec_from_v3(_manifest()),
            "provider.review",
            "logical-review",
            {"subject": "contract"},
            {"confirm": {"action": "accept", "content": {"ok": True}}},
            "opaque-state",
            deadline=_deadline(),
        )
    )

    assert complete["resultType"] == "complete"
    assert complete["prompt_id"] == "logical-review"
    assert complete["user_confirmation_required"] is True
    assert complete["messages"][0]["role"] == "user"
    assert (
        complete["messages"][0]["provenance"] == "untrusted_mcp_prompt"
    )
    assert session.prompt_calls == [
        {
            "prompt_name": "provider.review",
            "arguments": {"subject": "contract"},
            "input_responses": {
                "confirm": {"action": "accept", "content": {"ok": True}}
            },
            "request_state": "opaque-state",
            "allow_input_required": True,
        }
    ]


def test_resource_continuation_binary_returns_only_artifact_receipt() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    payload = b"private binary payload"

    class Writer:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def write_mcp_artifact(
            self,
            data: bytes,
            *,
            server_id: str,
            logical_id: str,
            mime_type: str | None,
        ) -> McpArtifactReceipt:
            assert (server_id, logical_id) == ("wire-demo", "logical-binary")
            self.payloads.append(data)
            return McpArtifactReceipt(
                artifact_id="artifact:binary",
                byte_length=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                mime_type=mime_type,
            )

    writer = Writer()
    session = _Session(
        mcp_types.ReadResourceResult(
            contents=[
                mcp_types.BlobResourceContents(
                    uri="opaque://provider/binary",
                    mimeType="application/octet-stream",
                    blob=base64.b64encode(payload).decode("ascii"),
                )
            ]
        )
    )
    provider = McpSdkV3ContinuationProvider(
        _Factory(session),
        result_adapter=McpSdkV2ResultAdapter(artifact_writer=writer),
    )

    result = asyncio.run(
        provider.continue_resource(
            mcp_transport_spec_from_v3(_manifest()),
            "opaque://provider/binary",
            "logical-binary",
            {},
            "state",
            deadline=_deadline(),
        )
    )

    assert writer.payloads == [payload]
    assert result["contents"][0]["artifact"]["artifact_id"] == "artifact:binary"
    assert base64.b64encode(payload).decode("ascii") not in str(result)


def test_completion_continuation_is_typed_unsupported_before_session() -> None:
    factory = _Factory(_Session())
    provider = McpSdkV3ContinuationProvider(factory)

    with pytest.raises(McpContinuationSurfaceUnsupported, match="completion/complete"):
        asyncio.run(
            provider.continue_completion(
                mcp_transport_spec_from_v3(_manifest()),
                deadline=_deadline(),
            )
        )
    assert factory.servers == []


def test_completion_input_required_is_rejected_before_human_capture() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class Handler:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def capture_input_required(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise AssertionError("unsupported surface must not create Human state")

    handler = Handler()
    adapter = McpSdkV2ResultAdapter(input_required_handler=handler)
    result = mcp_types.InputRequiredResult(inputRequests={}, requestState="state")

    with pytest.raises(McpContinuationSurfaceUnsupported, match="completion/complete"):
        adapter.completion_result(
            result,
            server_id="wire-demo",
            logical_id="logical-review",
            deadline=_deadline(),
        )
    assert handler.calls == []


def test_initial_sdk_capture_uses_host_logical_ids_not_remote_selectors() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class Handler:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def capture_input_required(self, **kwargs: Any) -> McpInputRequired:
            self.calls.append(kwargs)
            return McpInputRequired(continuation_id=f"c-{len(self.calls)}")

    handler = Handler()
    session = _Session(
        mcp_types.InputRequiredResult(inputRequests={}, requestState="resource-state"),
        mcp_types.InputRequiredResult(inputRequests={}, requestState="prompt-state"),
    )
    factory = _Factory(session)
    sdk_provider = McpSdkV2SessionProvider(
        factory,
        result_adapter=McpSdkV2ResultAdapter(input_required_handler=handler),
    )
    manifest = _manifest()
    client = McpModernClient(
        lambda _server_id: McpClientBinding(
            manifest=manifest,
            registry_generation=1,
        ),
        resource_provider=sdk_provider,
        prompt_provider=sdk_provider,
    )

    assert isinstance(client.read_resource("wire-demo", "logical-status"), McpInputRequired)
    assert isinstance(
        client.get_prompt(
            "wire-demo",
            "logical-review",
            {"subject": "contract"},
        ),
        McpInputRequired,
    )
    assert [call["logical_id"] for call in handler.calls] == [
        "logical-status",
        "logical-review",
    ]
    assert [call["operation"] for call in handler.calls] == [
        "resources/read",
        "prompts/get",
    ]


def test_result_adapter_merges_active_binding_secrets_before_capture() -> None:
    class Handler:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def capture_input_required(self, **kwargs: Any) -> McpInputRequired:
            self.calls.append(kwargs)
            return McpInputRequired(continuation_id="captured")

    handler = Handler()
    adapter = McpSdkV2ResultAdapter(input_required_handler=handler)
    with bind_mcp_client_binding(
        McpClientBinding(
            manifest=_manifest(),
            registry_generation=1,
            owner_id="runtime",
            sensitive_values=("dynamic-access-token",),
        )
    ):
        result = adapter.prompt_result(
            {
                "resultType": "input_required",
                "inputRequests": {},
                "requestState": "opaque-state",
            },
            server_id="wire-demo",
            logical_id="logical-review",
            deadline=_deadline(),
            sensitive_values=("static-secret",),
        )

    assert isinstance(result, McpInputRequired)
    assert handler.calls[0]["sensitive_values"] == (
        "static-secret",
        "dynamic-access-token",
    )


def test_modern_client_preserves_typed_unsupported_completion_without_capture() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    class Handler:
        def __init__(self) -> None:
            self.calls = 0

        def capture_input_required(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise AssertionError("Completion cannot create durable Human state")

    handler = Handler()
    session = _Session(
        mcp_types.InputRequiredResult(inputRequests={}, requestState="state")
    )
    provider = McpSdkV2SessionProvider(
        _Factory(session),
        result_adapter=McpSdkV2ResultAdapter(input_required_handler=handler),
    )
    manifest = _manifest()
    client = McpModernClient(
        lambda _server_id: McpClientBinding(
            manifest=manifest,
            registry_generation=1,
        ),
        prompt_provider=provider,
    )

    with pytest.raises(McpContinuationSurfaceUnsupported, match="completion/complete"):
        client.complete_prompt(
            "wire-demo",
            "prompt",
            "logical-review",
            {"name": "subject", "value": "contract"},
        )
    assert handler.calls == 0


@pytest.mark.parametrize(
    "result",
    [
        {
            "resultType": "input_required",
            "requestState": "refreshed-token",
            "inputRequests": {},
        },
        {
            "resultType": "task",
            "taskId": "refreshed-token",
            "status": "working",
        },
    ],
)
def test_dynamic_session_secret_cannot_be_reflected_as_opaque_identity(
    result: dict[str, Any],
) -> None:
    session = _Session(result)
    provider = McpSdkV3ContinuationProvider(
        _Factory(session, sensitive=("refreshed-token",)),
        host_tasks_extension_sha256=_TASKS_SHA256,
    )
    manifest = _manifest(tasks=True)
    with bind_mcp_client_binding(
        McpClientBinding(
            manifest=manifest,
            registry_generation=1,
            owner_id="test-owner",
        )
    ):
        with pytest.raises(ValidationError, match="reflected an operation secret"):
            asyncio.run(
                provider.continue_tool(
                    mcp_transport_spec_from_v3(manifest),
                    "provider.echo",
                    {"text": "x"},
                    {},
                    "state",
                    deadline=_deadline(),
                )
            )


def test_tasks_provider_uses_fixed_methods_and_has_no_list() -> None:
    session = _Session(
        {"resultType": "complete", "taskId": "remote-id", "status": "working"},
        {"resultType": "complete"},
        {"resultType": "complete"},
    )
    provider = McpSdkV3TasksProvider(
        _Factory(session), host_tasks_extension_sha256=_TASKS_SHA256
    )
    server = mcp_transport_spec_from_v3(_manifest(tasks=True))

    asyncio.run(provider.get_remote_task(server, "remote-id", deadline=_deadline()))
    asyncio.run(
        provider.update_remote_task(
            server,
            "remote-id",
            {"remote-input": {"action": "decline"}},
            deadline=_deadline(),
        )
    )
    asyncio.run(provider.cancel_remote_task(server, "remote-id", deadline=_deadline()))

    assert [item["request"] for item in session.raw_calls] == [
        {"method": "tasks/get", "params": {"taskId": "remote-id"}},
        {
            "method": "tasks/update",
            "params": {
                "taskId": "remote-id",
                "inputResponses": {"remote-input": {"action": "decline"}},
            },
        },
        {"method": "tasks/cancel", "params": {"taskId": "remote-id"}},
    ]
    assert not hasattr(provider, "list_remote_tasks")
    assert not hasattr(provider, "list")


def test_wire_rejects_downgrade_and_expired_deadline_without_dispatch() -> None:
    session = _Session({"resultType": "complete", "content": []})
    factory = _Factory(session)
    provider = McpSdkV3ContinuationProvider(factory)
    server = replace(
        mcp_transport_spec_from_v3(_manifest()),
        protocol_mode=McpProtocolMode.AUTO,
    )
    with pytest.raises(ValidationError, match="exact protocol"):
        asyncio.run(
            provider.continue_tool(
                server,
                "provider.echo",
                {"text": "x"},
                {},
                None,
                deadline=_deadline(),
            )
        )
    with pytest.raises(TimeoutError, match="deadline"):
        asyncio.run(
            provider.continue_tool(
                mcp_transport_spec_from_v3(_manifest()),
                "provider.echo",
                {"text": "x"},
                {},
                None,
                deadline=time.monotonic() - 1,
            )
        )
    assert factory.servers == []


@pytest.mark.parametrize(
    "apps_value",
    [
        "UI://malicious-app",
        "ui:opaque-app",
        'TEXT/HTML ; PROFILE = "MCP-APP"',
        "text/html; profile='mcp-app'",
    ],
)
def test_apps_policy_rejects_nested_keys_values_and_schema(apps_value: str) -> None:
    for value in (
        {"nested": [{"value": apps_value}]},
        {apps_value: "nested key"},
        {"requestedSchema": {"contentMediaType": apps_value}},
        {"statusMessage": apps_value},
        {"result": {"content": apps_value}},
    ):
        with pytest.raises(ValidationError, match="Apps"):
            sanitize_provider_json(value)


def test_sdk_completion_rejects_apps_values_before_secret_redaction() -> None:
    mcp_types = pytest.importorskip("mcp.types")
    secret = "sdk-completion-operation-secret"
    adapter = McpSdkV2ResultAdapter()

    for apps_value in (
        f"ui://{secret}",
        'TEXT/HTML ; PROFILE = "MCP-APP"',
    ):
        raw = mcp_types.CompleteResult(
            completion=mcp_types.Completion(values=[apps_value])
        )
        with pytest.raises(ValidationError, match="Apps") as captured:
            adapter.completion_result(
                raw,
                server_id="wire-demo",
                logical_id="logical-review",
                deadline=_deadline(),
                sensitive_values=(secret,),
            )
        assert secret not in str(captured.value)


def test_apps_metadata_is_discarded_case_insensitively() -> None:
    assert sanitize_provider_json(
        {
            "safe": True,
            "UI": {"resourceUri": "UI://not-evaluated"},
            "ui/resourceUri": "ui://legacy-flat",
            "UI/Visibility": ["model"],
            "ui/csp": {"connectDomains": ["https://ignored.invalid"]},
            "IO.MODELCONTEXTPROTOCOL/UI/widget": {"html": "ignored"},
        }
    ) == {"safe": True}
