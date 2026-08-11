import asyncio
import contextlib
import time
from typing import Any

import pytest

from agent_libos.mcp.wire import McpSdkV3ContinuationProvider
from agent_libos.models.mcp import (
    McpProtocolMode,
    McpServerSpec,
    McpStdioTransportSpec,
)


pytestmark = [pytest.mark.mcp, pytest.mark.mcp_transport]


def _server_spec() -> McpServerSpec:
    return McpServerSpec(
        schema_version=2,
        server_id="real-python-sdk-v2-mrtr",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="in-process-only"),
        tools=[],
        timeout_s=5.0,
        max_request_bytes=64 * 1024,
        max_response_bytes=256 * 1024,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
    )


def test_real_python_sdk_v2_resource_and_prompt_continuations_round_trip() -> None:
    # The repository-wide and PostgreSQL jobs intentionally omit the MCP extra
    # but must still collect this invariant node.  ``--run-mcp`` validates the
    # complete extra before execution, so load SDK-only symbols inside the test.
    from mcp import Client
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import (
        ElicitRequest,
        ElicitRequestFormParams,
        InputRequiredResult,
    )

    def input_required(*, request_id: str, request_state: str) -> InputRequiredResult:
        return InputRequiredResult(
            inputRequests={
                request_id: ElicitRequest(
                    params=ElicitRequestFormParams(
                        message="Approve this untrusted MCP content?",
                        requestedSchema={
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    )
                )
            },
            requestState=request_state,
        )

    async def exercise() -> None:
        sdk_server = MCPServer("agent-libos-mrtr-fixture", version="2.0.0")
        resource_calls: list[tuple[str, bool, str | None, str | None, bool | None]] = []
        prompt_calls: list[tuple[str, bool, str | None, str | None, bool | None]] = []

        @sdk_server.resource(
            "fixture://approval/{name}",
            name="approval-resource",
            mime_type="text/plain",
        )
        def approval_resource(
            name: str,
            ctx: Context,
        ) -> str | InputRequiredResult:
            responses = ctx.input_responses
            if responses is None:
                resource_calls.append(
                    (name, False, ctx.request_state, ctx.protocol_version, None)
                )
                return input_required(
                    request_id="resource-approval",
                    request_state="resource-state",
                )
            response = responses["resource-approval"]
            approved = bool(response.content and response.content["approved"])
            resource_calls.append(
                (name, True, ctx.request_state, ctx.protocol_version, approved)
            )
            return f"resource={name};approved={approved}"

        @sdk_server.prompt(name="approval_prompt")
        def approval_prompt(
            focus: str,
            ctx: Context,
        ) -> str | InputRequiredResult:
            responses = ctx.input_responses
            if responses is None:
                prompt_calls.append(
                    (focus, False, ctx.request_state, ctx.protocol_version, None)
                )
                return input_required(
                    request_id="prompt-approval",
                    request_state="prompt-state",
                )
            response = responses["prompt-approval"]
            approved = bool(response.content and response.content["approved"])
            prompt_calls.append(
                (focus, True, ctx.request_state, ctx.protocol_version, approved)
            )
            return f"prompt={focus};approved={approved}"

        server = _server_spec()
        factory_calls: list[McpServerSpec] = []
        async with Client(
            sdk_server,
            mode="2026-07-28",
            raise_exceptions=True,
        ) as client:
            assert client.session.protocol_version == "2026-07-28"

            @contextlib.asynccontextmanager
            async def session_factory(
                selected_server: McpServerSpec,
                *,
                deadline: float,
            ) -> Any:
                assert deadline > time.monotonic()
                factory_calls.append(selected_server)
                yield client.session

            provider = McpSdkV3ContinuationProvider(session_factory)

            initial_resource = await client.session.read_resource(
                "fixture://approval/ada",
                allow_input_required=True,
            )
            assert isinstance(initial_resource, InputRequiredResult)
            assert initial_resource.request_state
            resource_result = await provider.continue_resource(
                server,
                "fixture://approval/ada",
                "approved-document",
                {
                    "resource-approval": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                },
                initial_resource.request_state,
                deadline=time.monotonic() + 5.0,
            )

            initial_prompt = await client.session.get_prompt(
                "approval_prompt",
                {"focus": "accuracy"},
                allow_input_required=True,
            )
            assert isinstance(initial_prompt, InputRequiredResult)
            assert initial_prompt.request_state
            prompt_result = await provider.continue_prompt(
                server,
                "approval_prompt",
                "approved-review",
                {"focus": "accuracy"},
                {
                    "prompt-approval": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                },
                initial_prompt.request_state,
                deadline=time.monotonic() + 5.0,
            )

        assert resource_result["resultType"] == "complete"
        assert resource_result["resource_id"] == "approved-document"
        assert resource_result["provenance"] == "untrusted_mcp_resource"
        resource_contents = resource_result["contents"]
        assert isinstance(resource_contents, list)
        assert resource_contents[0]["text"] == "resource=ada;approved=True"

        assert prompt_result["resultType"] == "complete"
        assert prompt_result["prompt_id"] == "approved-review"
        assert prompt_result["user_confirmation_required"] is True
        prompt_messages = prompt_result["messages"]
        assert isinstance(prompt_messages, list)
        assert prompt_messages[0]["role"] == "user"
        assert prompt_messages[0]["provenance"] == "untrusted_mcp_prompt"
        assert prompt_messages[0]["content"]["text"] == (
            "prompt=accuracy;approved=True"
        )

        # Each SDK handler ran exactly once initially and exactly once for its
        # durable retry.  The server boundary unsealed the opaque requestState
        # and injected the accepted response only on the second invocation.
        assert resource_calls == [
            ("ada", False, None, "2026-07-28", None),
            ("ada", True, "resource-state", "2026-07-28", True),
        ]
        assert prompt_calls == [
            ("accuracy", False, None, "2026-07-28", None),
            ("accuracy", True, "prompt-state", "2026-07-28", True),
        ]
        assert factory_calls == [server, server]

    asyncio.run(exercise())
