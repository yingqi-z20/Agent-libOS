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
from agent_libos.tools.observability import json_size_bytes

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
# One observed labelled message adds a fixed-width Object source reference.
# The larger aggregate reserve covers four 256-character data-flow identities
# under worst-case JSON Unicode escaping. These are format bounds, not runtime
# tuning defaults.
_MESSAGE_CARRIER_REF_RESERVE_BYTES = 256
_MESSAGE_FLOW_LABEL_RESERVE_BYTES = 8_192
_DEFERRED_PROCESS_MESSAGE_ACK_METADATA_KEY = "_deferred_process_message_ack_ids"
_DURABLE_MESSAGE_METADATA_KEYS = frozenset(
    {"source_oids", "source_refs", "data_labels", "data_flow_context"}
)


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
    metadata: dict[str, Any] | None = Field(
        default=None,
        exclude_if=lambda value: not value,
        description="Data-flow provenance attached to the message, when present.",
    )
    status: str
    created_at: str
    acked_at: str | None = None


class ModelProcessMessageInfo(BaseModel):
    message_id: str
    sender: str
    kind: str
    channel: str
    correlation_id: str | None = None
    reply_to: str | None = None
    subject: str
    body: str
    payload: dict[str, Any]
    status: str


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
    has_more: bool = False
    omitted_count: int = 0
    continuation: dict[str, Any] | None = None


class ModelReadProcessMessagesOutput(BaseModel):
    ready: bool = True
    messages: list[ModelProcessMessageInfo]
    acked_message_ids: list[str]
    has_more: bool = False
    omitted_count: int = 0
    continuation: dict[str, Any] | None = None


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
        "are acknowledged by default. Large matching windows may be split to fit the durable result budget; "
        "when has_more is true, use the returned continuation with the same filters."
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
        messages, matching_count = runtime.messages.list_page(
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
        return _bounded_message_result(
            runtime,
            ctx,
            tool_name=self.name,
            messages=messages,
            matching_count=matching_count,
            ready=True,
            ack=args.ack,
        )

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
        "block=false returns immediately. Large matching windows may be split to fit the durable result budget; "
        "when has_more is true, use the returned continuation with the same filters."
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
        messages, matching_count = runtime.messages.receive_page(
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
        return _bounded_message_result(
            runtime,
            ctx,
            tool_name=self.name,
            messages=messages,
            matching_count=matching_count,
            ready=matching_count > 0,
            ack=args.ack,
        )

def _message_info(
    message: ProcessMessage,
    *,
    acknowledged: bool = False,
) -> ProcessMessageInfo:
    durable_metadata = {
        key: value
        for key, value in message.metadata.items()
        if key in _DURABLE_MESSAGE_METADATA_KEYS
    }
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
        metadata=durable_metadata or None,
        status="acked" if acknowledged and message.status.value == "unread" else message.status.value,
        created_at=message.created_at,
        acked_at=message.acked_at,
    )


def _model_message_info(
    message: ProcessMessage,
    *,
    acknowledged: bool = False,
) -> ModelProcessMessageInfo:
    return ModelProcessMessageInfo(
        message_id=message.message_id,
        sender=message.sender,
        kind=message.kind.value,
        channel=message.channel,
        correlation_id=message.correlation_id,
        reply_to=message.reply_to,
        subject=message.subject,
        body=message.body,
        payload=message.payload,
        status="acked" if acknowledged and message.status.value == "unread" else message.status.value,
    )


def _bounded_message_result(
    runtime: Any,
    ctx: ToolContext,
    *,
    tool_name: str,
    messages: list[ProcessMessage],
    matching_count: int,
    ready: bool,
    ack: bool,
) -> ToolResult:
    """Select a persistable page and stage its ACK with ToolResult commit.

    Process-message reads are side effects because their default behavior is
    to acknowledge returned unread messages. ToolExecutionService applies its
    generic result-size guard only after ``run`` returns, which is too late for
    this tool: an oversized page used to be acknowledged and then replaced by
    ``result_omitted``. Size the exact model-facing projection first, retain a
    conservative allowance for message label carriers, and defer the
    destructive ACK so ToolExecutionService can commit it atomically with the
    durable ToolResult.
    """

    if matching_count < len(messages):
        raise ToolExecutionError(
            "Process message snapshot count is smaller than its returned window.",
            code=ToolErrorCode.EXECUTION_ERROR,
        )

    selected = _select_messages_for_result(
        runtime,
        ctx,
        tool_name=tool_name,
        messages=messages,
        matching_count=matching_count,
        ready=ready,
        ack=ack,
    )
    omitted_count = matching_count - len(selected)
    predicted_acked_ids = [
        message.message_id
        for message in selected
        if ack and message.status.value == "unread"
    ]
    predicted_output = _message_output(
        tool_name=tool_name,
        ready=ready,
        selected=selected,
        omitted_count=omitted_count,
        acked_message_ids=predicted_acked_ids,
        acknowledge_selected=ack,
    )
    predicted_model_output = _model_message_output(
        tool_name=tool_name,
        ready=ready,
        selected=selected,
        omitted_count=omitted_count,
        acked_message_ids=predicted_acked_ids,
        acknowledge_selected=ack,
    )

    # Label observation is evidence/provenance materialization, so perform it
    # only for messages whose bodies will actually be returned to the model.
    carrier_oids = runtime.messages.observe_labels(ctx.pid, selected)
    predicted_result = _flow_labeled_result(
        runtime,
        ctx.pid,
        carrier_oids,
        predicted_output,
        predicted_model_output,
    )
    _ensure_message_result_fits(
        runtime,
        ctx,
        tool_name=tool_name,
        result=predicted_result,
    )

    if predicted_acked_ids:
        predicted_result.metadata[_DEFERRED_PROCESS_MESSAGE_ACK_METADATA_KEY] = list(
            predicted_acked_ids
        )
    return predicted_result


def _select_messages_for_result(
    runtime: Any,
    ctx: ToolContext,
    *,
    tool_name: str,
    messages: list[ProcessMessage],
    matching_count: int,
    ready: bool,
    ack: bool,
) -> list[ProcessMessage]:
    selected: list[ProcessMessage] = []
    for message in messages:
        candidate = [*selected, message]
        omitted_count = matching_count - len(candidate)
        acked_ids = [
            item.message_id
            for item in candidate
            if ack and item.status.value == "unread"
        ]
        output = _message_output(
            tool_name=tool_name,
            ready=ready,
            selected=candidate,
            omitted_count=omitted_count,
            acked_message_ids=acked_ids,
            acknowledge_selected=ack,
        )
        estimate = _result_envelope_size(
            runtime,
            ctx,
            tool_name=tool_name,
            output=output,
        )
        model_output = _model_message_output(
            tool_name=tool_name,
            ready=ready,
            selected=candidate,
            omitted_count=omitted_count,
            acked_message_ids=acked_ids,
            acknowledge_selected=ack,
        )
        estimate = max(estimate, json_size_bytes(model_output.model_dump()))
        # A received labelled message creates one metadata-only Object carrier.
        # Its source ref is small and fixed-width; reserve more than the encoded
        # ref plus the maximum possible aggregate label identity expansion.
        carrier_reserve = (
            len(candidate) * _MESSAGE_CARRIER_REF_RESERVE_BYTES
            + _MESSAGE_FLOW_LABEL_RESERVE_BYTES
        )
        if estimate + carrier_reserve > _message_result_limit(runtime):
            break
        selected = candidate
    return selected


def _message_output(
    *,
    tool_name: str,
    ready: bool,
    selected: list[ProcessMessage],
    omitted_count: int,
    acked_message_ids: list[str],
    acknowledge_selected: bool,
) -> ReadProcessMessagesOutput:
    return ReadProcessMessagesOutput(
        ready=ready,
        messages=[
            _message_info(message, acknowledged=acknowledge_selected)
            for message in selected
        ],
        acked_message_ids=acked_message_ids,
        has_more=omitted_count > 0,
        omitted_count=omitted_count,
        continuation=(
            {
                "tool": tool_name,
                "same_filters": True,
            }
            if omitted_count > 0
            else None
        ),
    )


def _model_message_output(
    *,
    tool_name: str,
    ready: bool,
    selected: list[ProcessMessage],
    omitted_count: int,
    acked_message_ids: list[str],
    acknowledge_selected: bool,
) -> ModelReadProcessMessagesOutput:
    return ModelReadProcessMessagesOutput(
        ready=ready,
        messages=[
            _model_message_info(message, acknowledged=acknowledge_selected)
            for message in selected
        ],
        acked_message_ids=acked_message_ids,
        has_more=omitted_count > 0,
        omitted_count=omitted_count,
        continuation=(
            {"tool": tool_name, "same_filters": True}
            if omitted_count > 0
            else None
        ),
    )


def _message_result_limit(runtime: Any) -> int:
    return min(
        runtime.config.tools.tool_result_payload_hard_limit_bytes,
        runtime.config.tools.memory_payload_hard_limit_bytes,
    )


def _result_envelope_size(
    runtime: Any,
    ctx: ToolContext,
    *,
    tool_name: str,
    output: ReadProcessMessagesOutput,
    metadata: dict[str, Any] | None = None,
) -> int:
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "tool_name": tool_name,
            "tool_version": _TOOL_DEFAULTS.version,
            "trace_id": ctx.trace_id,
            "call_id": ctx.call_id,
            # Deliberately longer than any ordinary measured duration. This
            # keeps the pre-ACK estimate conservative without time dependence.
            "duration_ms": 999_999_999_999.999,
        }
    )
    current_context = runtime.data_flow.current_context()
    result_metadata.setdefault("data_flow_context", current_context.to_dict())
    return json_size_bytes(
        {
            "tool_id": str(ctx.metadata.get("tool_id") or ("tool_" + "x" * 128)),
            "tool_name": tool_name,
            "result": output.model_dump(),
            "content": "",
            "artifacts": [],
            "metadata": result_metadata,
        }
    )


def _ensure_message_result_fits(
    runtime: Any,
    ctx: ToolContext,
    *,
    tool_name: str,
    result: ToolResult,
) -> None:
    output = ReadProcessMessagesOutput.model_validate(result.data)
    size = _result_envelope_size(
        runtime,
        ctx,
        tool_name=tool_name,
        output=output,
        metadata=result.metadata,
    )
    limit = _message_result_limit(runtime)
    model_size = json_size_bytes(result.model_projection(limit_bytes=limit))
    if max(size, model_size) > limit:
        raise ToolExecutionError(
            "Process message response exceeds the durable result budget; no messages were acknowledged.",
            code=ToolErrorCode.EXECUTION_ERROR,
            details={
                "result_bytes": size,
                "model_result_bytes": model_size,
                "limit_bytes": limit,
            },
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
    model_output: ModelReadProcessMessagesOutput,
) -> ToolResult:
    context = runtime.data_flow.context_from_source_oids(
        pid,
        carrier_oids,
        include_current=True,
    )
    return ToolResult.success(
        data=output.model_dump(),
        model_data=model_output.model_dump(),
        metadata={
            "data_flow_context": {
                "labels": context.labels.to_dict(),
                "source_refs": [ref.to_dict() for ref in context.source_refs],
                "materialization_id": context.materialization_id,
            }
        },
    )
