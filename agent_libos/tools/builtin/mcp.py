from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.serde import to_jsonable

class ListMcpServersArgs(BaseModel):
    text: str | None = Field(default=None, description="Optional MCP server search text.")
    limit: int | None = Field(default=None, ge=1)


class ListMcpServersOutput(BaseModel):
    servers: list[dict[str, Any]]


class InspectMcpServerArgs(BaseModel):
    server_id: str = Field(description="Registered MCP server id returned by list_mcp_servers.")


class InspectMcpServerOutput(BaseModel):
    server: dict[str, Any]


class ListMcpToolsArgs(BaseModel):
    server_id: str = Field(description="Registered MCP server id; transport commands and ad hoc URLs are not accepted.")
    refresh: bool = Field(
        default=False,
        description=(
            "False returns the registered allowlist without contacting the server. True performs a live tools/list external "
            "read and requires execute authority, provider policy, approval when configured, and resource budget."
        ),
    )


class ListMcpToolsOutput(BaseModel):
    server_id: str
    transport: str
    tools: list[dict[str, Any]]
    refreshed: bool
    response_bytes: int


class CallMcpToolArgs(BaseModel):
    server_id: str = Field(
        description="Pre-registered MCP server id; callers cannot supply a transport command or URL."
    )
    tool_id: str = Field(
        description="Allowed logical tool id returned by list_mcp_tools, not an arbitrary MCP tool name."
    )
    arguments: dict[str, Any] = Field(default_factory=dict, description="MCP tool arguments object.")


class CallMcpToolOutput(BaseModel):
    server_id: str
    tool_id: str
    mcp_name: str
    status: str
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None
    response_bytes: int
    duration_s: float


class ListMcpServersTool(BaseAgentTool[ListMcpServersArgs]):
    name = "list_mcp_servers"
    description = (
        "List Host-registered MCP server metadata without starting or contacting a server. "
        "Use JSON-RPC discovery tools instead for a plain JSON-RPC endpoint."
    )
    args_schema = ListMcpServersArgs
    output_schema = ListMcpServersOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"mcp_server.read"})
    tags = ["mcp", "remote"]

    async def execute(self, args: ListMcpServersArgs, ctx: ToolContext) -> ListMcpServersOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        servers = runtime.mcp.list_servers(actor=ctx.pid, text=args.text, limit=args.limit)
        return ListMcpServersOutput(servers=servers)


class InspectMcpServerTool(BaseAgentTool[InspectMcpServerArgs]):
    name = "inspect_mcp_server"
    description = "Inspect one Host-registered MCP server without contacting it or exposing resolved secrets."
    args_schema = InspectMcpServerArgs
    output_schema = InspectMcpServerOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"mcp_server.read"})
    tags = ["mcp", "remote"]

    async def execute(self, args: InspectMcpServerArgs, ctx: ToolContext) -> InspectMcpServerOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return InspectMcpServerOutput(server=runtime.mcp.inspect_server(args.server_id, actor=ctx.pid))


class ListMcpToolsTool(BaseAgentTool[ListMcpToolsArgs]):
    name = "list_mcp_tools"
    description = (
        "List allowed tools for a registered MCP server. By default this reads cached registry metadata only; "
        "refresh=true makes a governed external tools/list call and may require approval."
    )
    args_schema = ListMcpToolsArgs
    output_schema = ListMcpToolsOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"mcp_server.read", "mcp_server.execute"},
    )
    tags = ["mcp", "remote"]

    async def execute(self, args: ListMcpToolsArgs, ctx: ToolContext) -> ListMcpToolsOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = await runtime.mcp.alist_tools(
            args.server_id,
            actor=ctx.pid,
            refresh=args.refresh,
        )
        return ListMcpToolsOutput(**result)


class CallMcpToolTool(BaseAgentTool[CallMcpToolArgs]):
    name = "call_mcp_tool"
    description = (
        "Call one allowed logical tool on a pre-registered MCP server; ad hoc servers and transport commands "
        "are unavailable. "
        "The primitive enforces server registration, "
        "tool capability, human approval, audit, resource limits, and external-effect classification."
    )
    args_schema = CallMcpToolArgs
    output_schema = CallMcpToolOutput
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"mcp.call"}, timeout_s=None)
    tags = ["mcp", "remote", "external"]

    async def execute(self, args: CallMcpToolArgs, ctx: ToolContext) -> CallMcpToolOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = await runtime.mcp.acall_tool(ctx.pid, args.server_id, args.tool_id, args.arguments)
        return CallMcpToolOutput(**to_jsonable(result))
