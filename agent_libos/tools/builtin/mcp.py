from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    StringConstraints,
    field_validator,
)

from agent_libos.mcp.types import McpComplete, McpInputRequired, McpRemoteTask
from agent_libos.models.mcp import McpCallResult
from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.serde import to_jsonable


class _McpArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _McpOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


_McpLogicalId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$",
    ),
]
_McpOpaqueCursor = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=23,
        max_length=256,
        pattern=r"^mcpcur_[A-Za-z0-9_-]+$",
    ),
]
_McpVariableName = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$",
    ),
]
_McpVariableValue = Annotated[
    str,
    StringConstraints(strict=True, max_length=65_536),
]


class ListMcpServersArgs(_McpArgs):
    text: str | None = Field(default=None, description="Optional MCP server search text.")
    limit: int | None = Field(default=None, ge=1)


class ListMcpServersOutput(BaseModel):
    servers: list[dict[str, Any]]
    has_more: bool = Field(
        description="Whether another registered server matched beyond this bounded result."
    )


class InspectMcpServerArgs(_McpArgs):
    server_id: str = Field(description="Registered MCP server id returned by list_mcp_servers.")


class InspectMcpServerOutput(BaseModel):
    server: dict[str, Any]


class ListMcpToolsArgs(_McpArgs):
    server_id: str = Field(description="Registered MCP server id; transport commands and ad hoc URLs are not accepted.")
    refresh: bool = Field(
        default=False,
        description=(
            "False returns the registered allowlist without contacting the server. True performs a live tools/list external "
            "read and requires server read+execute authority (plus process spawn and exact stdio execute authority for "
            "stdio), provider policy, and resource budget. It does not register live-only tools."
        ),
    )


class ListMcpToolsOutput(BaseModel):
    server_id: str
    transport: str
    tools: list[dict[str, Any]]
    refreshed: bool
    response_bytes: int


class CallMcpToolArgs(_McpArgs):
    server_id: str = Field(
        description="Pre-registered MCP server id; callers cannot supply a transport command or URL."
    )
    tool_id: str = Field(
        description="Allowed logical tool id returned by list_mcp_tools, not an arbitrary MCP tool name."
    )
    arguments: dict[str, Any] = Field(default_factory=dict, description="MCP tool arguments object.")


class CallMcpToolOutput(_McpOutput):
    """Stable Manifest v1/v2 model projection."""

    server_id: str
    tool_id: str
    mcp_name: str
    status: str
    ok: bool
    result: Any = None
    error: dict[str, Any] | None = None
    response_bytes: int
    duration_s: float
    dispatch_state: str
    retry_class: str
    automatic_retry_disabled: bool


_McpLocalContinuationId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=9,
        max_length=128,
        pattern=r"^mcpcont_[A-Za-z0-9_-]+$",
    ),
]
_McpLocalTaskRef = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=9,
        max_length=128,
        pattern=r"^mcptask_[A-Za-z0-9_-]+$",
    ),
]
_McpLocalHumanRequestId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=6,
        max_length=64,
        pattern=r"^hreq_[0-9a-f]{16}$",
    ),
]
_Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class CallMcpHumanReceiptOutput(_McpOutput):
    request_id: _McpLocalHumanRequestId
    revision: int = Field(ge=0)
    preview_sha256: _Sha256


class CallMcpToolCompleteOutput(_McpOutput):
    kind: Literal["complete"]
    value: dict[str, JsonValue]


class CallMcpToolInputRequiredOutput(_McpOutput):
    kind: Literal["input_required"]
    continuation_id: _McpLocalContinuationId
    human_receipt: CallMcpHumanReceiptOutput | None


class CallMcpToolRemoteTaskOutput(_McpOutput):
    kind: Literal["remote_task"]
    task_ref: _McpLocalTaskRef
    status: Literal[
        "working",
        "input_required",
        "completed",
        "failed",
        "cancelled",
        "cancel_requested",
        "needs_attention",
    ]
    result: JsonValue | None
    human_receipt: CallMcpHumanReceiptOutput | None


_CallMcpToolV3Output = Annotated[
    CallMcpToolCompleteOutput
    | CallMcpToolInputRequiredOutput
    | CallMcpToolRemoteTaskOutput,
    Field(discriminator="kind"),
]


class CallMcpToolResultOutput(
    RootModel[CallMcpToolOutput | _CallMcpToolV3Output]
):
    """Closed model projection for stable v1/v2 and minimal v3 outcomes."""


class ListMcpResourcesArgs(_McpArgs):
    server_id: _McpLogicalId = Field(
        description="Registered Manifest v3 server logical id."
    )
    kind: Literal["resource", "template"] = Field(
        default="resource",
        description="List concrete Resources or Resource Templates.",
    )
    cursor: _McpOpaqueCursor | None = Field(
        default=None,
        description="Opaque one-use cursor returned by this same list surface.",
    )


class ListMcpResourcesOutput(BaseModel):
    server_id: str
    kind: Literal["resource", "template"]
    items: list[dict[str, Any]]
    next_cursor: str | None = None
    has_more: bool
    cache_hint: dict[str, Any] | None = None


class ReadMcpResourceArgs(_McpArgs):
    server_id: _McpLogicalId = Field(
        description="Registered Manifest v3 server logical id."
    )
    resource_id: _McpLogicalId = Field(
        description="Manifest-authorized logical Resource or Template id."
    )
    variables: dict[_McpVariableName, _McpVariableValue] = Field(
        default_factory=dict,
        max_length=256,
        description="Exact string variables for a manifest Resource Template.",
        json_schema_extra={"additionalProperties": False},
    )

    @field_validator("variables")
    @classmethod
    def _validate_variable_bytes(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        total_bytes = 0
        for key, item in value.items():
            if len(item.encode("utf-8")) > 65_536:
                raise ValueError("MCP Resource variable value exceeds maximum bytes")
            total_bytes += len(key.encode("utf-8")) + len(item.encode("utf-8"))
            if total_bytes > 1_048_576:
                raise ValueError("MCP Resource variables exceed maximum bytes")
        return value


class ReadMcpResourceOutput(BaseModel):
    server_id: str
    resource_id: str
    result: dict[str, Any]


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
        servers, has_more = runtime.mcp.list_servers_window(
            actor=ctx.pid,
            text=args.text,
            limit=args.limit,
        )
        return ListMcpServersOutput(servers=servers, has_more=has_more)


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
        "refresh=true makes a governed live tools/list call and reports manifest/live schema drift without "
        "changing the registered allowlist or granting authority."
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


def _call_mcp_human_receipt(
    result: McpInputRequired | McpRemoteTask,
) -> CallMcpHumanReceiptOutput | None:
    selected = (
        result.human_request_id,
        result.human_revision,
        result.human_preview_sha256,
    )
    if all(value is None for value in selected):
        return None
    if any(value is None for value in selected):
        raise ToolExecutionError(
            "MCP protected facade returned an incomplete Human receipt.",
            code=ToolErrorCode.EXECUTION_ERROR,
        )
    return CallMcpHumanReceiptOutput(
        request_id=result.human_request_id,
        revision=result.human_revision,
        preview_sha256=result.human_preview_sha256,
    )


def _call_mcp_model_output(result: Any) -> CallMcpToolResultOutput:
    if isinstance(result, McpCallResult):
        # This branch intentionally retains the established v1/v2 payload
        # byte-for-byte while the outer RootModel adds v3 alternatives.
        serialized = to_jsonable(result)
        legacy = CallMcpToolOutput.model_validate(
            {
                field: serialized[field]
                for field in CallMcpToolOutput.model_fields
            }
        )
        return CallMcpToolResultOutput(root=legacy)
    if isinstance(result, McpComplete):
        return CallMcpToolResultOutput(
            root=CallMcpToolCompleteOutput(
                kind="complete",
                value=to_jsonable(result.value),
            )
        )
    if isinstance(result, McpInputRequired):
        return CallMcpToolResultOutput(
            root=CallMcpToolInputRequiredOutput(
                kind="input_required",
                continuation_id=result.continuation_id,
                human_receipt=_call_mcp_human_receipt(result),
            )
        )
    if isinstance(result, McpRemoteTask):
        status = str(result.status)
        return CallMcpToolResultOutput(
            root=CallMcpToolRemoteTaskOutput(
                kind="remote_task",
                task_ref=result.task_ref,
                status=status,
                # A working/failed Task may carry provider diagnostics or a
                # partial value. Only a completed, already-sanitized final
                # result is part of the model contract.
                result=(
                    to_jsonable(result.result)
                    if status == "completed"
                    else None
                ),
                human_receipt=_call_mcp_human_receipt(result),
            )
        )
    raise ToolExecutionError(
        "MCP protected facade returned an unsupported Tool outcome.",
        code=ToolErrorCode.EXECUTION_ERROR,
    )


class CallMcpToolTool(BaseAgentTool[CallMcpToolArgs]):
    name = "call_mcp_tool"
    description = (
        "Call one allowed logical tool on a pre-registered MCP server; ad hoc servers and transport commands "
        "are unavailable. "
        "The primitive enforces server registration, "
        "tool capability, human approval, audit, resource limits, and external-effect classification. "
        "Manifest v3 pending outcomes expose only local continuation/Task handles and a Host Human receipt."
    )
    args_schema = CallMcpToolArgs
    output_schema = CallMcpToolResultOutput
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"mcp.call"}, timeout_s=None)
    tags = ["mcp", "remote", "external"]

    async def execute(
        self,
        args: CallMcpToolArgs,
        ctx: ToolContext,
    ) -> CallMcpToolResultOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = await runtime.mcp.acall_tool(ctx.pid, args.server_id, args.tool_id, args.arguments)
        return _call_mcp_model_output(result)


class ListMcpResourcesTool(BaseAgentTool[ListMcpResourcesArgs]):
    name = "list_mcp_resources"
    description = (
        "List only model-visible Manifest v3 MCP Resources or Resource Templates "
        "through opaque pagination; remote URIs and provider cursors are never accepted."
    )
    args_schema = ListMcpResourcesArgs
    output_schema = ListMcpResourcesOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"mcp_resource.read"},
        timeout_s=None,
    )
    tags = ["mcp", "remote", "resource"]

    async def execute(
        self,
        args: ListMcpResourcesArgs,
        ctx: ToolContext,
    ) -> ListMcpResourcesOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError(
                "Runtime is unavailable.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        method_name = (
            "alist_resources"
            if args.kind == "resource"
            else "alist_resource_templates"
        )
        method = getattr(runtime.mcp, method_name, None)
        if not callable(method):
            raise ToolExecutionError(
                "MCP Resources protected facade is unavailable.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        page = await method(
            args.server_id,
            cursor=args.cursor,
            actor=ctx.pid,
            model_visible_only=True,
        )
        payload = to_jsonable(page)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("items"),
            list,
        ):
            raise ToolExecutionError(
                "MCP Resources protected facade returned an invalid page.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        return ListMcpResourcesOutput(
            server_id=args.server_id,
            kind=args.kind,
            items=payload["items"],
            next_cursor=payload.get("next_cursor"),
            has_more=payload.get("next_cursor") is not None,
            cache_hint=payload.get("cache_hint"),
        )


class ReadMcpResourceTool(BaseAgentTool[ReadMcpResourceArgs]):
    name = "read_mcp_resource"
    description = (
        "Read one model-visible Manifest v3 MCP Resource by logical id. Binary "
        "content stays behind Host artifact receipts and ResourceLinks stay inert."
    )
    args_schema = ReadMcpResourceArgs
    output_schema = ReadMcpResourceOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"mcp_resource.read"},
        timeout_s=None,
    )
    tags = ["mcp", "remote", "resource"]

    async def execute(
        self,
        args: ReadMcpResourceArgs,
        ctx: ToolContext,
    ) -> ReadMcpResourceOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError(
                "Runtime is unavailable.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        method = getattr(runtime.mcp, "aread_resource", None)
        if not callable(method):
            raise ToolExecutionError(
                "MCP Resources protected facade is unavailable.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        result = await method(
            args.server_id,
            args.resource_id,
            variables=dict(args.variables),
            actor=ctx.pid,
            for_model=True,
        )
        payload = to_jsonable(result)
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                "MCP Resources protected facade returned an invalid result.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        return ReadMcpResourceOutput(
            server_id=args.server_id,
            resource_id=args.resource_id,
            result=payload,
        )
