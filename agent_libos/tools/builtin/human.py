from __future__ import annotations

import json
import threading
from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import HumanRequestStatus
from agent_libos.models.exceptions import HumanResponseRequired
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class HumanOutputArgs(BaseModel):
    message: str = Field(
        description="One-way message to present to the human; no answer or permission decision is requested."
    )
    channel: str = Field(
        default=_RUNTIME_DEFAULTS.terminal_channel,
        description="Configured human-provider output channel, not an arbitrary transport destination.",
    )


class HumanOutputResult(BaseModel):
    delivered: bool
    channel: str
    chars: int


class AskHumanArgs(BaseModel):
    question: str = Field(description="Decision or missing information to ask the human; not a capability request.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context shown with the question.",
    )
    human: str = Field(default=_RUNTIME_DEFAULTS.default_human, description="Human recipient name.")


class AskHumanResult(BaseModel):
    request_id: str
    answer: str
    status: str


class HumanOutputTool(SyncAgentTool[HumanOutputArgs]):
    name = "human_output"
    description = (
        "Send a one-way message to the human operator through the configured human provider. "
        "Use it for the concise final user-facing result before process_exit "
        "when the image requires interactive reporting; use ask_human when an answer is required. "
        "This is a Skills/Tools Layer wrapper around the libOS HumanObject output primitive; "
        "the primitive enforces human write capability, audit, and events."
    )
    args_schema = HumanOutputArgs
    output_schema = HumanOutputResult
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_confirmation_required=False,
        declared_permissions={"human.output"},
        timeout_s=_TOOL_DEFAULTS.interactive_timeout_s,
    )
    tags = ["human", "terminal", "output"]

    def run(self, args: HumanOutputArgs, ctx: ToolContext) -> HumanOutputResult:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        result = runtime.human.output(
            pid=ctx.pid,
            message=args.message,
            human=runtime.config.runtime.default_human,
            channel=args.channel,
        )
        return HumanOutputResult(**result)


class AskHumanTool(SyncAgentTool[AskHumanArgs]):
    name = "ask_human"
    description = (
        "Ask the human operator for a decision or missing information and return the answer. "
        "This blocks through the HumanObject queue; use request_permission instead when capability policy must change, "
        "and human_output for a one-way update."
    )
    args_schema = AskHumanArgs
    output_schema = AskHumanResult
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_confirmation_required=False,
        declared_permissions={"human.ask"},
        timeout_s=_TOOL_DEFAULTS.interactive_timeout_s,
    )
    tags = ["human", "terminal", "question"]

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending_by_key: dict[str, str] = {}

    def run(self, args: AskHumanArgs, ctx: ToolContext) -> AskHumanResult:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        key = self._pending_key(ctx.pid, args)
        request_id = ctx.metadata.get("human_resume_request_id")
        if isinstance(request_id, str) and request_id:
            self._validate_resume_request(runtime, ctx.pid, args, request_id)
        else:
            with self._lock:
                request_id = self._pending_by_key.get(key)
                if request_id is None:
                    # The first call queues the question and raises; the resumed
                    # call with the same arguments returns the recorded answer.
                    request_id = runtime.human.ask(
                        pid=ctx.pid,
                        human=args.human,
                        question=args.question,
                        context=args.context,
                        blocking=True,
                    )
                    self._pending_by_key[key] = request_id
        if request_id is None:
            raise ToolExecutionError("Human request id was not created.", code=ToolErrorCode.EXECUTION_ERROR)
        request = runtime.human.get(request_id)
        if request.status == HumanRequestStatus.PENDING:
            raise HumanResponseRequired(
                request_id=request_id,
                message=f"{ctx.pid} is waiting for human answer to {request_id}",
            )

        try:
            answer = runtime.human.answer_for_request(request_id)
        except HumanResponseRequired:
            raise
        except Exception:
            with self._lock:
                self._pending_by_key.pop(key, None)
            raise
        with self._lock:
            self._pending_by_key.pop(key, None)
        return AskHumanResult(request_id=request_id, answer=answer, status="answered")

    def _validate_resume_request(self, runtime: Any, pid: str, args: AskHumanArgs, request_id: str) -> None:
        request = runtime.human.get(request_id)
        if request.pid != pid or request.human != args.human:
            raise ToolExecutionError(
                "Human resume request does not belong to this ask_human call.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        expected_payload = {
            "type": "question",
            "question": args.question,
            "context": args.context,
        }
        if runtime.human.public_request_payload(request) != expected_payload:
            raise ToolExecutionError(
                "Human resume request payload does not match this ask_human call.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )

    def _pending_key(self, pid: str, args: AskHumanArgs) -> str:
        payload = {
            "pid": pid,
            "human": args.human,
            "question": args.question,
            "context": args.context,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
