from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.serde import to_jsonable

class ListJsonRpcEndpointsArgs(BaseModel):
    text: str | None = Field(default=None, description="Optional endpoint search text.")
    limit: int | None = Field(default=None, ge=1)


class ListJsonRpcEndpointsOutput(BaseModel):
    endpoints: list[dict[str, Any]]
    has_more: bool = Field(
        description="Whether another registered endpoint matched beyond this bounded result."
    )


class InspectJsonRpcEndpointArgs(BaseModel):
    endpoint_id: str = Field(
        description="Registered endpoint id returned by list_jsonrpc_endpoints; URLs are not accepted."
    )


class InspectJsonRpcEndpointOutput(BaseModel):
    endpoint: dict[str, Any]


class CallJsonRpcMethodArgs(BaseModel):
    endpoint_id: str = Field(
        description="Pre-registered endpoint id; callers cannot supply an ad hoc URL or credentials."
    )
    method_id: str = Field(
        description="Allowed logical method id declared by the registered endpoint, not raw HTTP or argv."
    )
    params: Any = Field(default=None, description="JSON-RPC params object, array, or null.")


class CallJsonRpcMethodOutput(BaseModel):
    endpoint_id: str
    method_id: str
    rpc_method: str
    request_id: str
    status: str
    http_status: int | None
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None
    response_bytes: int
    duration_s: float


class ListJsonRpcEndpointsTool(BaseAgentTool[ListJsonRpcEndpointsArgs]):
    name = "list_jsonrpc_endpoints"
    description = (
        "List Host-registered JSON-RPC endpoint metadata and logical method ids without contacting an endpoint. "
        "Use MCP discovery tools instead for an MCP server."
    )
    args_schema = ListJsonRpcEndpointsArgs
    output_schema = ListJsonRpcEndpointsOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"jsonrpc_endpoint.read"})
    tags = ["jsonrpc", "remote"]

    async def execute(self, args: ListJsonRpcEndpointsArgs, ctx: ToolContext) -> ListJsonRpcEndpointsOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        endpoints, has_more = runtime.jsonrpc.list_endpoints_window(
            actor=ctx.pid,
            text=args.text,
            limit=args.limit,
        )
        return ListJsonRpcEndpointsOutput(endpoints=endpoints, has_more=has_more)


class InspectJsonRpcEndpointTool(BaseAgentTool[InspectJsonRpcEndpointArgs]):
    name = "inspect_jsonrpc_endpoint"
    description = (
        "Inspect one Host-registered JSON-RPC endpoint and its allowed methods without contacting it or "
        "exposing secrets."
    )
    args_schema = InspectJsonRpcEndpointArgs
    output_schema = InspectJsonRpcEndpointOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"jsonrpc_endpoint.read"})
    tags = ["jsonrpc", "remote"]

    async def execute(self, args: InspectJsonRpcEndpointArgs, ctx: ToolContext) -> InspectJsonRpcEndpointOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return InspectJsonRpcEndpointOutput(
            endpoint=runtime.jsonrpc.inspect_endpoint(args.endpoint_id, actor=ctx.pid)
        )


class CallJsonRpcMethodTool(BaseAgentTool[CallJsonRpcMethodArgs]):
    name = "call_jsonrpc_method"
    description = (
        "Call one allowed logical method on a pre-registered JSON-RPC-over-HTTP endpoint; ad hoc URLs are unavailable. "
        "The primitive enforces endpoint registration, "
        "method capability, human approval, audit, and provider external-effect classification."
    )
    args_schema = CallJsonRpcMethodArgs
    output_schema = CallJsonRpcMethodOutput
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"jsonrpc.call"}, timeout_s=None)
    tags = ["jsonrpc", "remote", "external"]

    async def execute(self, args: CallJsonRpcMethodArgs, ctx: ToolContext) -> CallJsonRpcMethodOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = await runtime.jsonrpc.acall(ctx.pid, args.endpoint_id, args.method_id, args.params)
        return CallJsonRpcMethodOutput(**to_jsonable(result))
