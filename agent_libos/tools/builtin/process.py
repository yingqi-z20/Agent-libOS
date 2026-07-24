from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.prompt import recover_initial_goal_context
from agent_libos.memory.data_labels import flow_context_parts, flow_context_value
from agent_libos.models.exceptions import NotFound
from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog
from agent_libos.models import (
    AgentProcess,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    ExitedProcessOutcome,
    FailedProcessOutcome,
    ForkMode,
    MemoryViewSpec,
    MergePolicy,
    ObjectHandle,
    ObjectRight,
    ProcessMessageStatus,
    ProcessSignal,
    ResourceBudget,
    ViewMode,
    process_state_to_mapping,
)
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_CUMULATIVE_EXIT_REVIEW = "cumulative_review"
_COMPLETION_REVIEW_GOAL_FALLBACK_MAX_CHARS = 32_000


class CompletionAcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(
        min_length=1,
        description="One explicit cumulative requirement; do not bundle unrelated deliverables.",
    )
    source_refs: list[str] = Field(
        min_length=1,
        description="Goal oid or acknowledged human message ids that state this requirement.",
    )
    status: Literal["completed", "blocked", "cancelled"]
    evidence_tool_calls: list[str] = Field(
        description="Successful observed tool names that support this status. Required when completed."
    )
    evidence_summary: str = Field(
        min_length=1,
        description="Concrete verification evidence or blocker/cancellation reason.",
    )

    @field_validator("source_refs", "evidence_tool_calls", mode="before")
    @classmethod
    def parse_json_string_list(cls, value: Any) -> Any:
        return _parse_json_container(value)

    @field_validator("requirement", "evidence_summary")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")
        return value


class ProcessCompletionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_oid: str = Field(description="Exact goal oid from the cumulative completion review.")
    reviewed_message_ids: list[str] = Field(
        description="Exact acknowledged human message ids from the review; use an empty list when none."
    )
    acceptance_checks: list[CompletionAcceptanceCheck] = Field(
        min_length=1,
        description="One entry per explicit original or follow-up deliverable.",
    )
    final_verification: list[str] = Field(
        min_length=1,
        description="Successful observed tool names used for final verification.",
    )

    @field_validator(
        "reviewed_message_ids",
        "acceptance_checks",
        "final_verification",
        mode="before",
    )
    @classmethod
    def parse_json_lists(cls, value: Any) -> Any:
        return _parse_json_container(value)


class ProcessExitArgs(BaseModel):
    payload: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional structured final result mapping cumulative requested "
            "deliverables to verification evidence, blockers, and residual risks."
        ),
    )
    result_oid: str | None = Field(default=None, description="Existing object id to use as process result.")
    message: str | None = Field(
        default=None,
        description="Optional final message stored in a label-bearing result Object.",
    )
    review_token: str | None = Field(
        default=None,
        description=(
            "Fresh token returned by a prior nonterminal cumulative completion "
            "review. Supply it only after resolving every item surfaced by that review."
        ),
    )
    completion_evidence: ProcessCompletionEvidence | None = Field(
        default=None,
        description=(
            "For images using cumulative exit review: goal_oid, "
            "reviewed_message_ids, acceptance_checks, and final_verification. "
            "Each acceptance check must cite source_refs and observed tool-call evidence."
        ),
    )

    @field_validator("payload", mode="before")
    @classmethod
    def parse_json_payload(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return {"content": value}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        return value

    @field_validator("completion_evidence", mode="before")
    @classmethod
    def parse_json_completion_evidence(cls, value: Any) -> Any:
        return _parse_json_container(value)


class ProcessExitOutput(BaseModel):
    status: str
    result_oid: str | None = None
    completion_review: dict[str, Any] | None = None


class GetWorkingDirectoryArgs(BaseModel):
    pass


class GetWorkingDirectoryOutput(BaseModel):
    working_directory: str


class SetWorkingDirectoryArgs(BaseModel):
    path: str = Field(description="Workspace-relative directory, resolved from the current process working directory.")


class SetWorkingDirectoryOutput(BaseModel):
    working_directory: str


class ExecProcessArgs(BaseModel):
    image: str = Field(description="Target AgentImage id for the current process.")
    args: dict[str, Any] = Field(default_factory=dict, description="Structured exec arguments recorded in audit.")
    goal: str | dict[str, Any] | None = Field(
        default=None,
        description="Optional replacement goal for the new image.",
    )
    preserve_memory: bool = Field(default=True, description="Keep the current MemoryView unless a replacement is requested.")
    preserve_capabilities: bool = Field(
        default=False,
        description="Keep existing capabilities. Exec never grants capabilities required by the target image.",
    )


class ExecProcessOutput(BaseModel):
    pid: str
    old_image: str
    new_image: str
    status: str
    goal_oid: str | None
    preserve_memory: bool
    preserve_capabilities: bool
    active_tools: list[str] = Field(
        description=(
            "Complete post-exec process tool-table names. This is an authority/availability inventory, "
            "not the model-visible schema projection; activate the matching Skill before calling a hidden tool."
        )
    )


class ForkChildProcessArgs(BaseModel):
    goal: str | dict[str, Any] = Field(description="Goal for the child AgentProcess.")
    mode: str = Field(
        default=ForkMode.WORKER.value,
        description=(
            "Memory-view mode: copy uses copy-on-write, speculative uses an ephemeral view, "
            "and restricted/worker use read-only roots."
        ),
    )
    image: str | None = Field(
        default=None,
        description="Optional child image id. Defaults to the parent image.",
    )
    include_parent_roots: bool = Field(
        default=True,
        description=(
            "When root_oids is omitted, include all current parent MemoryView roots. "
            "Ignored when root_oids is provided."
        ),
    )
    root_oids: list[str] | None = Field(
        default=None,
        description=(
            "Optional exact Object ids that replace parent-root selection. An empty list "
            "exposes no parent roots; when provided, include_parent_roots is ignored."
        ),
    )
    inherit_read_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose read capability should be inherited by the child.",
    )
    inherit_write_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose write capability should be inherited by the child.",
    )
    inherit_read_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose read capability should be inherited by the child.",
    )
    inherit_write_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose write capability should be inherited by the child.",
    )
    inherit_capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit capability specs to inherit, each with resource and rights.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Optional child working directory. Defaults to the parent's current working directory.",
    )
    resource_budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional child ResourceBudget override. It must fit inside the parent process remaining budget.",
    )


class ForkChildProcessOutput(BaseModel):
    child_pid: str
    parent_pid: str
    image: str
    mode: str
    status: str
    goal_oid: str | None
    inherited_capabilities: list[dict[str, Any]]
    working_directory: str


class SpawnChildProcessArgs(BaseModel):
    goal: str | dict[str, Any] = Field(
        description="Goal for a fresh child whose MemoryView contains the goal but no copied parent roots."
    )
    image: str | None = Field(default=None, description="Optional child image id. Defaults to the parent image.")
    inherit_read_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose read capability should be inherited by the child.",
    )
    inherit_write_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files whose write capability should be inherited by the child.",
    )
    inherit_read_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose read capability should be inherited by the child.",
    )
    inherit_write_dirs: list[str] = Field(
        default_factory=list,
        description="Workspace-relative directories whose write capability should be inherited by the child.",
    )
    inherit_capabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit capability specs to inherit, each with resource and rights.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Optional child working directory. Defaults to the parent's current working directory.",
    )
    resource_budget: dict[str, Any] | None = Field(
        default=None,
        description="Optional child ResourceBudget override. It must fit inside the parent process remaining budget.",
    )


class SpawnChildProcessOutput(BaseModel):
    child_pid: str
    parent_pid: str
    image: str
    status: str
    goal_oid: str | None
    inherited_capabilities: list[dict[str, Any]]
    fresh_memory_view: bool
    working_directory: str


class WaitChildProcessArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id to wait for.")
    block: bool = Field(default=True, description="If false, return ready=false when the child is still running.")


class WaitChildProcessOutput(BaseModel):
    child_pid: str
    status: str
    ready: bool
    result_oid: str | None = None
    message: str | None = None
    wait_state: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    state_generation: int


class ChildProcessInfo(BaseModel):
    pid: str
    image: str
    status: str
    working_directory: str
    goal_oid: str | None
    result_oid: str | None = None
    status_message: str | None = None
    wait_state: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    state_generation: int


class ListChildProcessesArgs(BaseModel):
    include_terminal: bool = Field(default=True, description="Whether exited/failed/killed children are included.")


class ListChildProcessesOutput(BaseModel):
    children: list[ChildProcessInfo]


class SignalChildProcessArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id to signal.")
    signal: str = Field(description="Signal to send: pause, resume, cancel, or terminate.")
    reason: str | None = Field(
        default=None,
        description="Optional reason stored in a label-bearing Object referenced by child status.",
    )


class SignalChildProcessOutput(BaseModel):
    child_pid: str
    signal: str
    status: str
    wait_state: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    state_generation: int


class MergeChildMemoryArgs(BaseModel):
    child_pid: str = Field(description="Direct child process id whose memory should be merged.")
    include_child_created: bool = Field(default=True, description="Include objects created by the child.")


class MergeChildMemoryOutput(BaseModel):
    child_pid: str
    merged_oids: list[str]
    skipped_oids: list[str]


class ProcessExitTool(SyncAgentTool[ProcessExitArgs]):
    name = "process_exit"
    description = (
        "Terminally exit the current Agent Process with an optional final result. "
        "Except for an image's explicit nonterminal cumulative review, call only "
        "after the original goal and all acknowledged human follow-ups are "
        "cumulatively complete; a follow-up does not erase unmentioned requirements "
        "unless it explicitly replaces or cancels them. Before confirmed exit, "
        "verify every requested deliverable and evidence step; passing tests alone "
        "does not prove the whole goal is complete. "
        "An image with cumulative exit review returns a nonterminal review on the "
        "first attempt. This is also the supported exact-goal recovery path after "
        "reopen when a runtime-only goal Object is unavailable. Re-read its goal "
        "and human follow-ups, complete every missing "
        "item, then retry with its fresh review_token and structured "
        "completion_evidence. "
        "This does not present the result to the human; interactive images "
        "should call human_output in a prior quantum. "
        "This is a Skills/Tools Layer wrapper over process lifecycle primitives."
    )
    args_schema = ProcessExitArgs
    output_schema = ProcessExitOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.read", "object.write", "process.lifecycle"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "lifecycle"]

    def run(self, args: ProcessExitArgs, ctx: ToolContext) -> ProcessExitOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        process = runtime.process.get(ctx.pid)
        image = runtime.images[process.image_id]
        if image.metadata.get("completion_gate") == _CUMULATIVE_EXIT_REVIEW:
            return self._run_cumulative_exit(args, ctx, runtime, image)
        return self._commit_exit(args, ctx, runtime, image)

    def _run_cumulative_exit(
        self,
        args: ProcessExitArgs,
        ctx: ToolContext,
        runtime: Any,
        image: Any,
    ) -> ProcessExitOutput:
        # Human message posting and process exit use this same Runtime store
        # lock. Keep review, freshness validation, and terminal commit inside
        # one critical section so a follow-up cannot land between them.
        with runtime.store.locked():
            process = runtime.process.get(ctx.pid)
            review = _build_cumulative_exit_review(runtime, ctx.pid)
            validation_errors = _completion_evidence_errors(
                args,
                review=review,
            )
            if validation_errors:
                review["validation_errors"] = validation_errors
                runtime.audit.record(
                    actor=ctx.pid,
                    action="process.exit_review_required",
                    target=f"process:{ctx.pid}",
                    input_refs=[process.goal_oid] if process.goal_oid else [],
                    decision={
                        "review_token": review["review_token"],
                        "acknowledged_human_message_ids": review[
                            "acknowledged_human_message_ids"
                        ],
                        "unread_human_message_ids": review[
                            "unread_human_message_ids"
                        ],
                        "goal_source": review["goal"]["source"],
                        "validation_errors": validation_errors,
                        "authority_changed": False,
                    },
                )
                return ProcessExitOutput(
                    status="completion_review_required",
                    completion_review=review,
                )
            completion_evidence = args.completion_evidence
            if not isinstance(completion_evidence, ProcessCompletionEvidence):
                raise ToolExecutionError(
                    "Validated completion evidence disappeared before exit.",
                    code=ToolErrorCode.VALIDATION_ERROR,
                )
            runtime.audit.record(
                actor=ctx.pid,
                action="process.exit_review_passed",
                target=f"process:{ctx.pid}",
                input_refs=[process.goal_oid] if process.goal_oid else [],
                decision={
                    "review_token": review["review_token"],
                    "goal_source": review["goal"]["source"],
                    "evidence_sha256": _canonical_sha256(
                        completion_evidence.model_dump(mode="json")
                    ),
                    "acceptance_check_count": len(
                        completion_evidence.acceptance_checks
                    ),
                    "authority_changed": False,
                },
            )
            return self._commit_exit(args, ctx, runtime, image)

    def _commit_exit(
        self,
        args: ProcessExitArgs,
        ctx: ToolContext,
        runtime: Any,
        image: Any,
    ) -> ProcessExitOutput:
        source_oids, source_labels, source_context = _flow_sources(ctx)
        result_handle: ObjectHandle | None = None
        if args.result_oid:
            result_handle = runtime.memory.handle_for_oid(
                ctx.pid,
                args.result_oid,
                required_rights={ObjectRight.READ.value},
                optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
                issued_by="process_exit_tool",
            )
        generated_payload = (
            args.payload
            if args.payload is not None
            else {"message": args.message}
            if args.message is not None
            else None
        )
        if (
            image.metadata.get("completion_gate") == _CUMULATIVE_EXIT_REVIEW
            and result_handle is None
        ):
            generated_payload = {
                **(generated_payload or {}),
                "completion_evidence": (
                    args.completion_evidence.model_dump(mode="json")
                    if args.completion_evidence is not None
                    else None
                ),
            }
        result_handle = runtime.process.exit(
            ctx.pid,
            result=result_handle,
            payload=generated_payload if result_handle is None else None,
            source_oids=source_oids,
            source_labels=source_labels,
            source_context=source_context,
        )
        result_oid = result_handle.oid if result_handle is not None else None
        return ProcessExitOutput(status="exited", result_oid=result_oid)


def _build_cumulative_exit_review(runtime: Any, pid: str) -> dict[str, Any]:
    process = runtime.process.get(pid)
    if process.goal_oid is None:
        raise ToolExecutionError(
            "Cumulative exit review requires a durable process goal.",
            code=ToolErrorCode.VALIDATION_ERROR,
        )
    goal_oid = process.goal_oid
    (
        goal_payload,
        goal_version,
        goal_source,
        goal_reference,
        goal_fallback_required,
    ) = _completion_goal_payload(
        runtime,
        pid,
        goal_oid,
    )
    human_messages, acknowledged, unread = _completion_review_human_messages(runtime, pid)
    goal_sha256 = _canonical_sha256(goal_payload)
    acknowledged_messages_sha256 = _canonical_sequence_sha256(
        _completion_message_identity(message) for message in acknowledged
    )
    contract_identity = {
        "goal_oid": goal_oid,
        "goal_version": goal_version,
        "human_messages": [
            {
                "message_id": message.message_id,
                "status": message.status.value,
            }
            for message in human_messages
        ],
    }
    acknowledged_message_ids = [message.message_id for message in acknowledged]
    observed_tools = _successful_tool_calls(runtime, pid)
    goal_evidence: dict[str, Any] = {
        "oid": goal_oid,
        "version": goal_version,
        "payload_sha256": goal_sha256,
        "source": goal_source,
        "reference": goal_reference,
    }
    if goal_fallback_required:
        goal_evidence["fallback"] = _bounded_json_value(
            goal_payload,
            _COMPLETION_REVIEW_GOAL_FALLBACK_MAX_CHARS,
        )
    return {
        "schema_version": 2,
        "review_token": f"exitrev_{_canonical_sha256(contract_identity)}",
        "goal": goal_evidence,
        "acknowledged_human_message_count": len(acknowledged),
        "acknowledged_human_message_ids": acknowledged_message_ids,
        "acknowledged_human_messages_sha256": acknowledged_messages_sha256,
        "acknowledged_human_message_reference": {
            "kind": "process_message_ids",
            "activate_skill": "agent-libos-child-processes",
            "tool": "read_process_messages",
            "fixed_arguments": {
                "include_acked": True,
                "ack": False,
            },
            "copy_arguments": {
                "message_ids": "acknowledged_human_message_ids",
                "limit": "acknowledged_human_message_count",
            },
        },
        "unread_human_message_ids": [message.message_id for message in unread],
        "observed_successful_tool_calls": observed_tools,
        "explicit_unobserved_tool_hints": _explicit_unobserved_tool_hints(
            process,
            goal_payload,
            observed_tools,
        ),
        "required_evidence_shape": {
            "goal_oid": goal_oid,
            "reviewed_message_ids": (
                "copy acknowledged_human_message_ids exactly"
            ),
            "acceptance_checks": [
                {
                    "requirement": "one explicit cumulative requirement; do not bundle unrelated deliverables",
                    "source_refs": ["goal oid or acknowledged human message id"],
                    "status": "completed | blocked | cancelled",
                    "evidence_tool_calls": ["successful tool names that prove this status"],
                    "evidence_summary": "concise concrete evidence or blocker/cancellation reason",
                }
            ],
            "final_verification": ["successful tool names used for final verification"],
        },
        "instructions": [
            "Split every explicit goal and follow-up deliverable into its own acceptance check.",
            (
                "Use the goal and acknowledged-message references instead of "
                "treating this review as another content copy."
            ),
            (
                "If referenced content is absent from the current context, recover "
                "it with the exact reference before claiming coverage."
            ),
            "Compare the checks with observed_successful_tool_calls; complete missing work before retrying.",
            (
                "For each explicit_unobserved_tool_hint that the goal still requires, "
                "activate its Skill and call the tool before retrying."
            ),
            (
                "Cover the goal oid and every acknowledged_human_message_id as "
                "source_refs, and cite only successful tool calls actually observed."
            ),
            (
                "Send the final human-facing report only after this review is clear, "
                "then retry process_exit with review_token and completion_evidence."
            ),
        ],
    }


def _completion_review_human_messages(
    runtime: Any,
    pid: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    human_messages = [
        message
        for message in runtime.store.list_process_messages(pid)
        if message.sender.startswith("human:")
        or message.payload.get("source") == "human_input"
    ]
    message_limit = runtime.config.tools.message_read_hard_limit
    if len(human_messages) > message_limit:
        raise ToolExecutionError(
            "Cumulative exit review cannot safely represent every human message; "
            "start a successor process with a consolidated goal.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={
                "human_message_count": len(human_messages),
                "review_message_limit": message_limit,
            },
        )
    acknowledged = [
        message
        for message in human_messages
        if message.status == ProcessMessageStatus.ACKED
    ]
    unread = [
        message
        for message in human_messages
        if message.status == ProcessMessageStatus.UNREAD
    ]
    if acknowledged:
        runtime.messages.observe_labels(pid, acknowledged)
    return human_messages, acknowledged, unread


def _completion_message_identity(message: Any) -> dict[str, Any]:
    """Return the immutable, content-bearing part of a mailbox message.

    ACK timestamps and label-carrier metadata are deliberately excluded: they
    can change when the same durable message is observed again, while the
    requirement the model must review does not.
    """

    return {
        "message_id": message.message_id,
        "sender": message.sender,
        "recipient_pid": message.recipient_pid,
        "kind": message.kind.value,
        "channel": message.channel,
        "correlation_id": message.correlation_id,
        "reply_to": message.reply_to,
        "subject": message.subject,
        "body": message.body,
        "payload": message.payload,
    }


def _completion_goal_payload(
    runtime: Any,
    pid: str,
    goal_oid: str,
) -> tuple[Any, int, str, dict[str, Any], bool]:
    try:
        goal_handle = runtime.memory.handle_for_oid(
            pid,
            goal_oid,
            required_rights={ObjectRight.READ.value},
            issued_by="process_exit_review",
        )
        goal = runtime.memory.get_object(pid, goal_handle)
        return (
            goal.payload,
            goal.version,
            "object_memory",
            {
                "kind": "object_memory",
                "namespace": goal.namespace,
                "name": goal.name,
                "activate_skill": "agent-libos-object-memory",
                "tool": "read_memory_object",
                "arguments": {
                    "namespace": goal.namespace,
                    "name": goal.name,
                    "max_payload_chars": (
                        runtime.config.tools.memory_payload_hard_limit_chars
                    ),
                },
            },
            False,
        )
    except NotFound as exc:
        # Runtime Object payloads are intentionally volatile across a Host
        # reopen. Full-I/O LLM evidence already contains the exact initial
        # materialized goal, so use that existing durable copy rather than
        # introducing a second plaintext persistence channel. When the Host
        # opted out of full-I/O retention, fail closed and require a checkpoint
        # restore or explicit human restatement.
        durable_context = _initial_user_context_from_llm_evidence(
            runtime,
            pid,
            goal_oid,
        )
        if durable_context is None:
            raise ToolExecutionError(
                "The process goal payload is unavailable after Runtime reopen and "
                "no full-I/O initial LLM context is retained. Restore a checkpoint "
                "or ask the human to restate the cumulative goal before exit.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"goal_oid": goal_oid, "cause": type(exc).__name__},
            ) from exc
        persisted = runtime.uow.objects.get_persisted_object_state(goal_oid)
        version = persisted.version if persisted is not None else 0
        persisted_metadata = runtime.uow.objects.get_persisted_object_metadata(
            goal_oid
        )
        if persisted_metadata is not None:
            runtime.data_flow.observe_ingress(
                DataFlowContext(
                    labels=DataLabels.from_object_metadata(persisted_metadata)
                )
            )
        recovered_payload = _goal_payload_from_recovered_context(
            durable_context,
            goal_oid,
        )
        return (
            recovered_payload
            if recovered_payload is not None
            else {
                "recovered_from": "persisted_initial_llm_context",
                "initial_user_context": durable_context,
                "instruction": (
                    "Recover every original requirement from this initial context; "
                    "do not treat later human input as a replacement."
                ),
            },
            version,
            "persisted_initial_llm_context",
            {
                "kind": "retained_llm_evidence",
                "goal_oid": goal_oid,
                "instruction": (
                    "Use the bounded fallback because the runtime-local goal "
                    "Object is unavailable after reopen."
                ),
            },
            True,
        )


def _goal_payload_from_recovered_context(
    recovered_context: str,
    goal_oid: str,
) -> Any | None:
    """Extract the logical goal payload from retained prompt evidence.

    Current source-only and persistent-context renderings are canonical JSON.
    The literal parser preserves reopen compatibility with earlier source-only
    evidence without executing input.
    """

    candidate = recovered_context.strip()
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        referenced_oid = decoded.get("object_oid", decoded.get("oid"))
        if referenced_oid == goal_oid and "payload" in decoded:
            return decoded["payload"]

    payload_marker = " payload: "
    if payload_marker not in candidate:
        return None
    legacy_payload = candidate.split(payload_marker, 1)[1].strip()
    try:
        return ast.literal_eval(legacy_payload)
    except (SyntaxError, ValueError):
        return None


def _initial_user_context_from_llm_evidence(
    runtime: Any,
    pid: str,
    goal_oid: str,
) -> str | None:
    calls = runtime.store.list_llm_calls(
        pid=pid,
        limit=runtime.config.llm.call_record_hard_limit,
    )
    try:
        return recover_initial_goal_context(calls, goal_oid)
    except ValueError as exc:
        raise ToolExecutionError(
            "The retained goal context exceeds the cumulative review recovery "
            "limit. Restore a checkpoint or ask the human to restate a bounded "
            "cumulative goal before exit.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"goal_oid": goal_oid},
        ) from exc


def _completion_evidence_errors(
    args: ProcessExitArgs,
    *,
    review: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if review["unread_human_message_ids"]:
        errors.append(
            "read and acknowledge every unread human message before completion review"
        )
    if args.review_token != review["review_token"]:
        errors.append("review_token is missing or stale; use the token from this review")
    evidence_model = args.completion_evidence
    if not isinstance(evidence_model, ProcessCompletionEvidence):
        errors.append("completion_evidence must be a structured object")
        return errors
    errors.extend(_completion_identity_errors(evidence_model, review=review))
    observed_tools = set(review["observed_successful_tool_calls"])
    expected_sources = {
        str(review["goal"]["oid"]),
        *[
            str(message_id)
            for message_id in review["acknowledged_human_message_ids"]
        ],
    }
    covered_sources: set[str] = set()
    for index, check in enumerate(evidence_model.acceptance_checks):
        check_errors, covered = _acceptance_check_errors(
            check,
            index=index,
            expected_sources=expected_sources,
            observed_tools=observed_tools,
        )
        errors.extend(check_errors)
        covered_sources.update(covered)
    missing_sources = expected_sources - covered_sources
    if missing_sources:
        errors.append("acceptance checks do not cover every expected_source_ref")
    if set(evidence_model.final_verification) - observed_tools:
        errors.append("completion_evidence.final_verification cites unobserved tools")
    return errors


def _completion_identity_errors(
    evidence: ProcessCompletionEvidence,
    *,
    review: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if evidence.goal_oid != str(review["goal"]["oid"]):
        errors.append("completion_evidence.goal_oid must match the reviewed goal oid")
    expected_message_ids = list(review["acknowledged_human_message_ids"])
    if (
        set(evidence.reviewed_message_ids) != set(expected_message_ids)
        or len(evidence.reviewed_message_ids) != len(expected_message_ids)
    ):
        errors.append(
            "completion_evidence.reviewed_message_ids must exactly cover acknowledged human messages"
        )
    return errors


def _acceptance_check_errors(
    check: CompletionAcceptanceCheck,
    *,
    index: int,
    expected_sources: set[str],
    observed_tools: set[str],
) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    label = f"acceptance_checks[{index}]"
    source_refs = set(check.source_refs)
    if source_refs - expected_sources:
        errors.append(f"{label}.source_refs contains unknown references")
    if check.status == "completed" and not check.evidence_tool_calls:
        errors.append(
            f"{label}.evidence_tool_calls must cite evidence for completed work"
        )
    elif set(check.evidence_tool_calls) - observed_tools:
        errors.append(f"{label}.evidence_tool_calls cites unobserved tools")
    return errors, source_refs & expected_sources


def _successful_tool_calls(runtime: Any, pid: str) -> list[str]:
    names: list[str] = []
    for record in runtime.audit.trace(actor=pid):
        if record.action != "tool.call" or record.decision.get("ok") is not True:
            continue
        name = record.decision.get("tool")
        # A prior nonterminal review is not evidence that any deliverable was
        # completed. Excluding it also keeps an unchanged review stable across
        # retries.
        if name == "process_exit":
            continue
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _explicit_unobserved_tool_hints(
    process: AgentProcess,
    goal_payload: Any,
    observed_tools: list[str],
) -> list[dict[str, str]]:
    searchable = json.dumps(
        goal_payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).lower()
    observed = set(observed_tools)
    catalog = get_builtin_skill_catalog()
    hints: list[dict[str, str]] = []
    for tool_name in sorted(process.tool_table):
        if tool_name in observed:
            continue
        phrase = tool_name.lower().replace("_", " ").replace(".", " ")
        if len(phrase) < 5 or phrase not in searchable:
            continue
        skill_id = catalog.skill_for_tool(tool_name)
        if skill_id is None:
            continue
        hints.append(
            {
                "tool": tool_name,
                "activate_skill": skill_id,
                "reason": f"The cumulative goal explicitly mentions '{phrase}'.",
            }
        )
    return hints


def _bounded_json_value(value: Any, max_chars: int) -> Any:
    rendered = _canonical_json(value)
    normalized = json.loads(rendered)
    if len(rendered) <= max_chars:
        return normalized
    envelope: dict[str, Any] = {
        "truncated": True,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "preview": None,
        "instruction": (
            "Use the supplied recovery reference for omitted content before completion."
        ),
    }
    shell_chars = len(_canonical_json(envelope)) - len("null")
    preview_budget = max_chars - shell_chars
    if preview_budget < 1:
        # Completion review uses a much larger bound, but keep this helper
        # total for focused callers and never return malformed partial JSON.
        return None if max_chars >= len("null") else 0
    envelope["preview"] = _project_json_to_limit(normalized, preview_budget)
    return envelope


def _project_json_to_limit(value: Any, max_chars: int) -> Any:
    """Build a structurally valid JSON projection within ``max_chars``.

    This truncates decoded string values and whole container entries before
    serializing them.  It never slices serialized JSON, so quotes, escapes,
    arrays, and objects remain closed and parseable.
    """

    rendered = _canonical_json(value)
    if len(rendered) <= max_chars:
        return value
    item_limits = (64, 32, 16, 8, 4, 2, 1, 0)
    for item_limit in item_limits:
        smallest = _project_json_value(
            value,
            string_limit=0,
            item_limit=item_limit,
            depth=8,
        )
        if len(_canonical_json(smallest)) > max_chars:
            continue
        low = 0
        high = max_chars
        best = smallest
        while low <= high:
            candidate_limit = (low + high) // 2
            candidate = _project_json_value(
                value,
                string_limit=candidate_limit,
                item_limit=item_limit,
                depth=8,
            )
            if len(_canonical_json(candidate)) <= max_chars:
                best = candidate
                low = candidate_limit + 1
            else:
                high = candidate_limit - 1
        return best
    return None if max_chars >= len("null") else 0


def _project_json_value(
    value: Any,
    *,
    string_limit: int,
    item_limit: int,
    depth: int,
) -> Any:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return f"{value[:string_limit]}\u2026"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth <= 0:
        return {"__agent_libos_truncated__": type(value).__name__}
    if isinstance(value, list):
        projected = [
            _project_json_value(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth - 1,
            )
            for item in value[:item_limit]
        ]
        if len(value) > item_limit:
            projected.append(
                {"__agent_libos_omitted_items__": len(value) - item_limit}
            )
        return projected
    if isinstance(value, dict):
        items = list(value.items())
        projected = {
            key: _project_json_value(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth - 1,
            )
            for key, item in items[:item_limit]
        }
        if len(items) > item_limit:
            marker = "__agent_libos_omitted_items__"
            while marker in projected:
                marker = f"_{marker}"
            projected[marker] = len(items) - item_limit
        return projected
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _canonical_sha256(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sequence_sha256(values: Iterable[Any]) -> str:
    """Hash a canonical JSON array without constructing a second large list."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(_canonical_json(value).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def _parse_json_container(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded


class GetWorkingDirectoryTool(SyncAgentTool[GetWorkingDirectoryArgs]):
    name = "get_working_directory"
    description = "Return this AgentProcess working directory, relative to the runtime workspace root."
    args_schema = GetWorkingDirectoryArgs
    output_schema = GetWorkingDirectoryOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["process", "working_directory"]

    def run(self, args: GetWorkingDirectoryArgs, ctx: ToolContext) -> GetWorkingDirectoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return GetWorkingDirectoryOutput(working_directory=runtime.process.working_directory(ctx.pid))


class SetWorkingDirectoryTool(SyncAgentTool[SetWorkingDirectoryArgs]):
    name = "set_working_directory"
    description = (
        "Set this AgentProcess working directory. Relative filesystem and shell tool paths resolve from it."
    )
    args_schema = SetWorkingDirectoryArgs
    output_schema = SetWorkingDirectoryOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"filesystem.read", "process.cwd"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "working_directory"]

    def run(self, args: SetWorkingDirectoryArgs, ctx: ToolContext) -> SetWorkingDirectoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        process = runtime.set_process_working_directory(ctx.pid, args.path)
        return SetWorkingDirectoryOutput(working_directory=process.working_directory)


class ExecProcessTool(SyncAgentTool[ExecProcessArgs]):
    name = "exec_process"
    description = (
        "Replace the current AgentProcess image and tool table without changing pid. "
        "Exec is not a permission escalation path: target image required capabilities are not granted automatically."
    )
    args_schema = ExecProcessArgs
    output_schema = ExecProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "process.lifecycle", "tool.table"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "lifecycle", "exec"]

    def run(self, args: ExecProcessArgs, ctx: ToolContext) -> ExecProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        old_image = runtime.process.get(ctx.pid).image_id
        source_oids, source_labels, source_context = _flow_sources(ctx)
        try:
            process = runtime.exec_process(
                ctx.pid,
                args.image,
                args=args.args,
                goal=args.goal,
                preserve_memory=args.preserve_memory,
                preserve_capabilities=args.preserve_capabilities,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )
        except NotFound as exc:
            raise ToolExecutionError(
                "Target image does not exist.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"image": args.image},
            ) from exc
        return ExecProcessOutput(
            pid=process.pid,
            old_image=old_image,
            new_image=process.image_id,
            status=process.status.value,
            goal_oid=process.goal_oid,
            preserve_memory=args.preserve_memory,
            preserve_capabilities=args.preserve_capabilities,
            active_tools=sorted(process.tool_table),
        )


class ForkChildProcessTool(SyncAgentTool[ForkChildProcessArgs]):
    name = "fork_child_process"
    description = (
        "Fork a direct Agent libOS child that can see selected parent MemoryView roots under an attenuated view. "
        "Use spawn_child_process instead for a fresh goal-only MemoryView; neither tool creates a host OS process."
    )
    args_schema = ForkChildProcessArgs
    output_schema = ForkChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"capability.write", "filesystem.read", "object.read", "object.write", "process.spawn"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "fork"]

    def run(self, args: ForkChildProcessArgs, ctx: ToolContext) -> ForkChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        parent = runtime.process.get(ctx.pid)
        source_oids, source_labels, source_context = _flow_sources(ctx)
        image = args.image or parent.image_id
        try:
            fork_mode = ForkMode(args.mode)
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid fork mode.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"mode": args.mode, "allowed": [mode.value for mode in ForkMode]},
            ) from exc
        roots = self._selected_roots(runtime, ctx.pid, args.root_oids)
        view_spec = MemoryViewSpec(
            roots=roots,
            mode=_view_mode_for_fork(fork_mode),
            include_parent_roots=args.include_parent_roots,
        )
        inherit_specs = self._inheritance_specs(runtime, args, cwd=parent.working_directory)
        try:
            child_pid = runtime.fork_child_process(
                parent=ctx.pid,
                goal=args.goal,
                memory_view=view_spec,
                inherit_capabilities=inherit_specs,
                resource_budget=_resource_budget_from_spec(args.resource_budget),
                image=image,
                mode=fork_mode,
                working_directory=args.working_directory,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )
        except NotFound as exc:
            raise ToolExecutionError(
                "Target image does not exist.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"image": image},
            ) from exc
        child = runtime.process.get(child_pid)
        return ForkChildProcessOutput(
            child_pid=child.pid,
            parent_pid=ctx.pid,
            image=child.image_id,
            mode=fork_mode.value,
            status=child.status.value,
            goal_oid=child.goal_oid,
            inherited_capabilities=inherit_specs,
            working_directory=child.working_directory,
        )

    def _selected_roots(self, runtime: Any, pid: str, root_oids: list[str] | None) -> list[ObjectHandle] | None:
        if root_oids is None:
            return None
        process = runtime.process.get(pid)
        visible = {handle.oid: handle for handle in (process.memory_view.roots if process.memory_view else [])}
        roots: list[ObjectHandle] = []
        for oid in root_oids:
            if oid in visible:
                roots.append(visible[oid])
                continue
            roots.append(
                runtime.memory.handle_for_oid(
                    pid,
                    oid,
                    required_rights={ObjectRight.READ.value},
                    optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.DIFF.value},
                    issued_by="fork_child_process_tool",
                )
            )
        return roots

    def _inheritance_specs(
        self,
        runtime: Any,
        args: ForkChildProcessArgs,
        *,
        cwd: str,
    ) -> list[dict[str, Any]]:
        specs = [_normalize_capability_spec(spec) for spec in args.inherit_capabilities]
        for path in args.inherit_read_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]})
        for path in args.inherit_write_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]})
        for path in args.inherit_read_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]}
            )
        for path in args.inherit_write_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]}
            )
        return _coalesce_capability_specs(specs)


class SpawnChildProcessTool(SyncAgentTool[SpawnChildProcessArgs]):
    name = "spawn_child_process"
    description = (
        "Create a fresh direct Agent libOS child with a new namespace and goal-only MemoryView. "
        "Use fork_child_process when selected parent memory roots are required; explicit capabilities may still "
        "be inherited here."
    )
    args_schema = SpawnChildProcessArgs
    output_schema = SpawnChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"capability.write", "filesystem.read", "object.write", "process.spawn"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "spawn"]

    def run(self, args: SpawnChildProcessArgs, ctx: ToolContext) -> SpawnChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        source_oids, source_labels, source_context = _flow_sources(ctx)
        inherit_specs = self._inheritance_specs(runtime, args, cwd=runtime.process.working_directory(ctx.pid))
        child_pid = runtime.spawn_child_process(
            parent=ctx.pid,
            goal=args.goal,
            image=args.image,
            inherit_capabilities=inherit_specs,
            resource_budget=_resource_budget_from_spec(args.resource_budget),
            working_directory=args.working_directory,
            source_oids=source_oids,
            source_labels=source_labels,
            source_context=source_context,
        )
        child = runtime.process.get(child_pid)
        return SpawnChildProcessOutput(
            child_pid=child.pid,
            parent_pid=ctx.pid,
            image=child.image_id,
            status=child.status.value,
            goal_oid=child.goal_oid,
            inherited_capabilities=inherit_specs,
            fresh_memory_view=True,
            working_directory=child.working_directory,
        )

    def _inheritance_specs(
        self,
        runtime: Any,
        args: SpawnChildProcessArgs,
        *,
        cwd: str,
    ) -> list[dict[str, Any]]:
        specs = [_normalize_capability_spec(spec) for spec in args.inherit_capabilities]
        for path in args.inherit_read_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]})
        for path in args.inherit_write_files:
            specs.append({"resource": runtime.filesystem.resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]})
        for path in args.inherit_read_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.READ.value]}
            )
        for path in args.inherit_write_dirs:
            specs.append(
                {"resource": runtime.filesystem.directory_resource_for_path(path, cwd=cwd), "rights": [CapabilityRight.WRITE.value]}
            )
        return _coalesce_capability_specs(specs)


class WaitChildProcessTool(SyncAgentTool[WaitChildProcessArgs]):
    name = "wait_child_process"
    description = (
        "Wait for a direct child AgentProcess to exit, fail, or be killed. "
        "If the child is still running, block=true suspends this process until that event; "
        "block=false returns ready=false immediately. "
        "Do not poll with sleep."
    )
    args_schema = WaitChildProcessArgs
    output_schema = WaitChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.wait"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "wait"]

    def run(self, args: WaitChildProcessArgs, ctx: ToolContext) -> WaitChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            result = runtime.process.wait(
                ctx.pid,
                args.child_pid,
                timeout=None if args.block else 0,
            )
        except TimeoutError:
            child = runtime.process.get(args.child_pid)
            return WaitChildProcessOutput(
                child_pid=args.child_pid,
                ready=False,
                message=child.status_message,
                **process_state_to_mapping(
                    child.status.value,
                    child.wait_state,
                    child.outcome,
                    child.state_generation,
                ),
            )
        return WaitChildProcessOutput(
            child_pid=result.pid,
            ready=True,
            result_oid=result.result.oid if result.result is not None else None,
            message=result.message,
            **process_state_to_mapping(
                result.status.value,
                result.wait_state,
                result.outcome,
                result.state_generation,
            ),
        )


class ListChildProcessesTool(SyncAgentTool[ListChildProcessesArgs]):
    name = "list_child_processes"
    description = "List direct child AgentProcesses owned by the current process."
    args_schema = ListChildProcessesArgs
    output_schema = ListChildProcessesOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.standard_timeout_s)
    tags = ["process", "child", "inspect"]

    def run(self, args: ListChildProcessesArgs, ctx: ToolContext) -> ListChildProcessesOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        return ListChildProcessesOutput(
            children=[_child_info(child) for child in runtime.process.list_children(ctx.pid, args.include_terminal)]
        )


class SignalChildProcessTool(SyncAgentTool[SignalChildProcessArgs]):
    name = "signal_child_process"
    description = "Pause, resume, cancel, or terminate a direct child AgentProcess."
    args_schema = SignalChildProcessArgs
    output_schema = SignalChildProcessOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"process.signal"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "signal"]

    def run(self, args: SignalChildProcessArgs, ctx: ToolContext) -> SignalChildProcessOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        try:
            signal = ProcessSignal(args.signal)
        except ValueError as exc:
            raise ToolExecutionError(
                "Invalid process signal.",
                code=ToolErrorCode.VALIDATION_ERROR,
                details={"signal": args.signal, "allowed": ["pause", "resume", "cancel", "terminate"]},
            ) from exc
        if signal not in {ProcessSignal.PAUSE, ProcessSignal.RESUME, ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
            raise ToolExecutionError(
                "Signal is not exposed through this tool.",
                code=ToolErrorCode.PERMISSION_DENIED,
                details={"signal": signal.value},
            )
        source_oids, source_labels, source_context = _flow_sources(ctx)
        child = runtime.process.signal_child(
            ctx.pid,
            args.child_pid,
            signal,
            reason=args.reason,
            source_oids=source_oids,
            source_labels=source_labels,
            source_context=source_context,
        )
        return SignalChildProcessOutput(
            child_pid=child.pid,
            signal=signal.value,
            **process_state_to_mapping(
                child.status.value,
                child.wait_state,
                child.outcome,
                child.state_generation,
            ),
        )


class MergeChildMemoryTool(SyncAgentTool[MergeChildMemoryArgs]):
    name = "merge_child_memory"
    description = "Merge result-visible Object Memory from an exited direct child into the parent process view."
    args_schema = MergeChildMemoryArgs
    output_schema = MergeChildMemoryOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.link", "object.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["process", "child", "memory"]

    def run(self, args: MergeChildMemoryArgs, ctx: ToolContext) -> MergeChildMemoryOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = runtime.process.merge_child_memory(
            ctx.pid,
            args.child_pid,
            policy=MergePolicy(include_child_created=args.include_child_created),
        )
        return MergeChildMemoryOutput(
            child_pid=args.child_pid,
            merged_oids=result.merged_oids,
            skipped_oids=result.skipped_oids,
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


def _view_mode_for_fork(mode: ForkMode) -> ViewMode:
    if mode == ForkMode.COPY:
        return ViewMode.COPY_ON_WRITE
    if mode == ForkMode.SPECULATIVE:
        return ViewMode.EPHEMERAL
    return ViewMode.READ_ONLY


def _child_info(child: AgentProcess) -> ChildProcessInfo:
    result_oid = None
    if isinstance(child.outcome, (ExitedProcessOutcome, FailedProcessOutcome)):
        result_oid = child.outcome.result_oid
    return ChildProcessInfo(
        pid=child.pid,
        image=child.image_id,
        working_directory=child.working_directory,
        goal_oid=child.goal_oid,
        result_oid=result_oid,
        status_message=child.status_message,
        **process_state_to_mapping(
            child.status.value,
            child.wait_state,
            child.outcome,
            child.state_generation,
        ),
    )


def _normalize_capability_spec(spec: dict[str, Any]) -> dict[str, Any]:
    resource = spec.get("resource")
    if not isinstance(resource, str) or not resource:
        raise ToolExecutionError(
            "Inherited capability spec requires a non-empty resource.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"spec": spec},
        )
    rights = spec.get("rights", [CapabilityRight.READ.value])
    if not isinstance(rights, list) or not rights:
        raise ToolExecutionError(
            "Inherited capability spec requires a non-empty rights list.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"spec": spec},
        )
    normalized: dict[str, Any] = {"resource": resource, "rights": [str(right) for right in rights]}
    constraints = spec.get("constraints")
    if isinstance(constraints, dict):
        normalized["constraints"] = constraints
    return normalized


def _resource_budget_from_spec(spec: dict[str, Any] | None) -> ResourceBudget | None:
    if spec is None:
        return None
    allowed = set(ResourceBudget.__dataclass_fields__)
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ToolExecutionError(
            "Unknown resource_budget fields.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"unknown_fields": unknown},
        )
    try:
        return ResourceBudget(**{key: value for key, value in spec.items() if key in allowed})
    except ValueError as exc:
        raise ToolExecutionError(
            "Invalid resource_budget.",
            code=ToolErrorCode.VALIDATION_ERROR,
            details={"error": str(exc)},
        ) from exc


def _coalesce_capability_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_resource: dict[str, dict[str, Any]] = {}
    for spec in specs:
        resource = str(spec["resource"])
        current = by_resource.setdefault(resource, {"resource": resource, "rights": []})
        current["rights"] = sorted(set(current["rights"]) | {str(right) for right in spec.get("rights", [])})
        if isinstance(spec.get("constraints"), dict):
            current["constraints"] = dict(spec["constraints"])
    return list(by_resource.values())
