from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.memory.data_labels import flow_context_parts, flow_context_value
from agent_libos.models import DataFlowContext, ProcessMessage, ProcessMessageKind
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolPolicy,
    ToolResult,
)

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class ProcessMessageInfo(BaseModel):
    message_id: str
    sender: str
    recipient_pid: str
    kind: str
    channel: str
    correlation_id: str | None = None
    reply_to: str | None = None
    subject: str
    body: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: str
    acked_at: str | None = None


class SendProcessMessageArgs(BaseModel):
    recipient_pid: str = Field(description="Target process id. Must be self, parent, or a direct child.")
    kind: str = Field(default=ProcessMessageKind.NORMAL.value, description="Message kind: normal or interrupt.")
    channel: str = Field(default="default", description="Mailbox channel for selective receive.")
    correlation_id: str | None = Field(default=None, description="Optional conversation/request correlation id.")
    reply_to: str | None = Field(default=None, description="Optional message id this message replies to.")
    subject: str = Field(default="", max_length=_TOOL_DEFAULTS.message_subject_max_chars, description="Short message subject.")
    body: str = Field(default="", max_length=_TOOL_DEFAULTS.message_body_max_chars, description="Message body.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Structured message payload.")


class SendProcessMessageOutput(BaseModel):
    message_id: str
    recipient_pid: str
    kind: str
    channel: str
    correlation_id: str | None = None
    reply_to: str | None = None
    subject: str


class ReadProcessMessagesArgs(BaseModel):
    include_acked: bool = Field(default=False, description="Include already acknowledged messages.")
    kind: str | None = Field(default=None, description="Optional kind filter: normal or interrupt.")
    sender: str | None = Field(default=None, description="Optional sender filter.")
    channel: str | None = Field(default=None, description="Optional channel filter.")
    correlation_id: str | None = Field(default=None, description="Optional correlation id filter.")
    reply_to: str | None = Field(default=None, description="Optional reply-to message id filter.")
    message_ids: list[str] | None = Field(
        default=None,
        max_length=_TOOL_DEFAULTS.message_filter_ids_hard_limit,
        description="Optional exact message ids to return.",
    )
    limit: int | None = Field(
        default=None,
        ge=0,
        le=_TOOL_DEFAULTS.message_read_hard_limit,
        description="Maximum number of messages to return.",
    )
    ack: bool = Field(
        default=True,
        description=(
            "Acknowledge each returned unread message after reading; set false to leave it unread for a later receive."
        ),
    )


class ReadProcessMessagesOutput(BaseModel):
    ready: bool = True
    messages: list[ProcessMessageInfo]
    acked_message_ids: list[str]


class SendProcessMessageTool(SyncAgentTool[SendProcessMessageArgs]):
    name = "send_process_message"
    description = (
        "Send a message to this process, its parent, or a direct child. "
        "Interrupt messages notify the target before its next tool call; normal messages notify after a tool call."
    )
    args_schema = SendProcessMessageArgs
    output_schema = SendProcessMessageOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.message"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "message"]

    def run(self, args: SendProcessMessageArgs, ctx: ToolContext) -> SendProcessMessageOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        source_oids, source_labels, source_context = _flow_sources(ctx)
        try:
            message = runtime.messages.send_from_process(
                ctx.pid,
                args.recipient_pid,
                kind=ProcessMessageKind(args.kind),
                channel=args.channel,
                correlation_id=args.correlation_id,
                reply_to=args.reply_to,
                subject=args.subject,
                body=args.body,
                payload=args.payload,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid process message kind.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"kind": args.kind, "allowed": [kind.value for kind in ProcessMessageKind]},
            ) from exc
        return SendProcessMessageOutput(
            message_id=message.message_id,
            recipient_pid=message.recipient_pid,
            kind=message.kind.value,
            channel=message.channel,
            correlation_id=message.correlation_id,
            reply_to=message.reply_to,
            subject=message.subject,
        )


class ReadProcessMessagesTool(SyncAgentTool[ReadProcessMessagesArgs]):
    name = "read_process_messages"
    description = (
        "Take an immediate, non-blocking snapshot of this process mailbox using optional filters. "
        "Use receive_process_messages when the process should suspend until a match; returned unread messages "
        "are acknowledged by default."
    )
    args_schema = ReadProcessMessagesArgs
    output_schema = ReadProcessMessagesOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.message"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "message", "inspect"]

    def run(self, args: ReadProcessMessagesArgs, ctx: ToolContext) -> ToolResult:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            kind = ProcessMessageKind(args.kind) if args.kind is not None else None
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid process message kind.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"kind": args.kind, "allowed": [kind.value for kind in ProcessMessageKind]},
            ) from exc
        messages = runtime.messages.list(
            ctx.pid,
            include_acked=args.include_acked,
            kind=kind,
            sender=args.sender,
            channel=args.channel,
            correlation_id=args.correlation_id,
            reply_to=args.reply_to,
            message_ids=args.message_ids,
            limit=args.limit,
        )
        carrier_oids = runtime.messages.observe_labels(ctx.pid, messages)
        acked: list[ProcessMessage] = []
        if args.ack:
            unread_ids = [message.message_id for message in messages if message.status.value == "unread"]
            if unread_ids:
                acked = runtime.messages.ack(ctx.pid, unread_ids)
                acked_by_id = {message.message_id: message for message in acked}
                messages = [acked_by_id.get(message.message_id, message) for message in messages]
        output = ReadProcessMessagesOutput(
            ready=True,
            messages=[_message_info(message) for message in messages],
            acked_message_ids=[message.message_id for message in acked],
        )
        return _flow_labeled_result(runtime, ctx.pid, carrier_oids, output)


class ReceiveProcessMessagesArgs(ReadProcessMessagesArgs):
    block: bool = Field(
        default=True,
        description=(
            "Suspend until a matching unread message arrives; false returns ready=false immediately when none match."
        ),
    )


class ReceiveProcessMessagesTool(SyncAgentTool[ReceiveProcessMessagesArgs]):
    name = "receive_process_messages"
    description = (
        "Receive unread process messages using optional selective filters. "
        "Unlike read_process_messages, block=true suspends in WAITING_EVENT until a match; "
        "block=false returns immediately."
    )
    args_schema = ReceiveProcessMessagesArgs
    output_schema = ReadProcessMessagesOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.message"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "message", "ipc", "receive"]

    def run(self, args: ReceiveProcessMessagesArgs, ctx: ToolContext) -> ToolResult:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            kind = ProcessMessageKind(args.kind) if args.kind is not None else None
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid process message kind.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"kind": args.kind, "allowed": [kind.value for kind in ProcessMessageKind]},
            ) from exc
        messages = runtime.messages.receive(
            ctx.pid,
            block=args.block,
            include_acked=args.include_acked,
            kind=kind,
            sender=args.sender,
            channel=args.channel,
            correlation_id=args.correlation_id,
            reply_to=args.reply_to,
            message_ids=args.message_ids,
            limit=args.limit,
        )
        carrier_oids = runtime.messages.observe_labels(ctx.pid, messages)
        acked: list[ProcessMessage] = []
        if args.ack:
            unread_ids = [message.message_id for message in messages if message.status.value == "unread"]
            if unread_ids:
                acked = runtime.messages.ack(ctx.pid, unread_ids)
                acked_by_id = {message.message_id: message for message in acked}
                messages = [acked_by_id.get(message.message_id, message) for message in messages]
        output = ReadProcessMessagesOutput(
            ready=bool(messages),
            messages=[_message_info(message) for message in messages],
            acked_message_ids=[message.message_id for message in acked],
        )
        return _flow_labeled_result(runtime, ctx.pid, carrier_oids, output)


def _message_info(message: ProcessMessage) -> ProcessMessageInfo:
    return ProcessMessageInfo(
        message_id=message.message_id,
        sender=message.sender,
        recipient_pid=message.recipient_pid,
        kind=message.kind.value,
        channel=message.channel,
        correlation_id=message.correlation_id,
        reply_to=message.reply_to,
        subject=message.subject,
        body=message.body,
        payload=message.payload,
        metadata={
            key: value
            for key, value in message.metadata.items()
            if key in {"source_oids", "source_refs", "data_labels", "data_flow_context"}
        },
        status=message.status.value,
        created_at=message.created_at,
        acked_at=message.acked_at,
    )


def _flow_sources(ctx: ToolContext) -> tuple[list[str] | None, Any | None, DataFlowContext | None]:
    try:
        source_oids, labels = flow_context_parts(ctx.metadata)
        return source_oids, labels, flow_context_value(ctx.metadata)
    except ValueError as exc:
        raise ToolExecutionError(
            str(exc),
            code=ToolErrorCode.EXECUTION_ERROR,
        ) from exc


def _flow_labeled_result(
    runtime: Any,
    pid: str,
    carrier_oids: list[str],
    output: ReadProcessMessagesOutput,
) -> ToolResult:
    context = runtime.data_flow.context_from_source_oids(
        pid,
        carrier_oids,
        include_current=True,
    )
    return ToolResult.success(
        data=output.model_dump(),
        metadata={
            "data_flow_context": {
                "labels": context.labels.to_dict(),
                "source_refs": [ref.to_dict() for ref in context.source_refs],
                "materialization_id": context.materialization_id,
            }
        },
    )
