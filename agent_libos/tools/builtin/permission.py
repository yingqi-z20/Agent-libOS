from __future__ import annotations

import json
import threading
from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight, HumanRequestStatus
from agent_libos.models.exceptions import HumanResponseRequired
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class RequestPermissionArgs(BaseModel):
    resource: str = Field(
        description="Exact canonical capability resource to authorize, such as filesystem:workspace:path.txt."
    )
    rights: list[str] = Field(description="Smallest required capability rights, such as ['read'] or ['write'].")
    reason: str = Field(description="Brief task-specific reason shown with the proposed policy change.")
    human: str = Field(default=_RUNTIME_DEFAULTS.default_human, description="Human recipient name.")


class RequestPermissionOutput(BaseModel):
    request_id: str
    resource: str
    rights: list[str]
    status: str


class RequestPermissionTool(SyncAgentTool[RequestPermissionArgs]):
    name = "request_permission"
    description = (
        "Ask the human to set capability policy for one exact libOS resource and a minimal set of rights. "
        "Use this only to change authority, not for a general question or confirmation; the human can allow, deny, "
        "or require per-use approval, and the underlying primitive still enforces the resulting policy."
    )
    args_schema = RequestPermissionArgs
    output_schema = RequestPermissionOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"human.ask", "capability.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["permission", "human", "capability"]

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending_by_key: dict[str, str] = {}

    def run(self, args: RequestPermissionArgs, ctx: ToolContext) -> RequestPermissionOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        if not args.rights:
            raise ToolExecutionError(
                "At least one capability right is required.",
                code=ToolErrorCode.VALIDATION_ERROR,
            )
        try:
            rights = [CapabilityRight(str(right)).value for right in args.rights]
        except ValueError as exc:
            raise ToolExecutionError(
                f"Unknown capability right: {exc}",
                code=ToolErrorCode.VALIDATION_ERROR,
            ) from exc
        selected_human = self._selected_human(runtime, args)
        key = self._pending_key(ctx.pid, args, rights, human=selected_human)
        request_id = ctx.metadata.get("human_resume_request_id")
        if isinstance(request_id, str) and request_id:
            self._validate_resume_request(
                runtime,
                ctx.pid,
                args,
                rights,
                request_id,
                human=selected_human,
            )
        else:
            with self._lock:
                request_id = self._pending_by_key.get(key)
                if request_id is None:
                    request_id = runtime.human.request_permission(
                        pid=ctx.pid,
                        human=selected_human,
                        resource=args.resource,
                        rights=rights,
                        reason=args.reason,
                    )
                    self._pending_by_key[key] = request_id
        if request_id is None:
            raise ToolExecutionError("Human permission request id was not created.", code=ToolErrorCode.EXECUTION_ERROR)
        request = runtime.human.get(request_id)
        if request.status == HumanRequestStatus.PENDING:
            raise HumanResponseRequired(
                request_id=request_id,
                message=f"{ctx.pid} is waiting for human permission decision {request_id}",
            )
        with self._lock:
            self._pending_by_key.pop(key, None)
        if request.status not in {HumanRequestStatus.APPROVED, HumanRequestStatus.REJECTED}:
            raise ToolExecutionError(
                f"Human permission request {request_id} ended with status={request.status.value}.",
                code=ToolErrorCode.EXECUTION_ERROR,
            )
        return RequestPermissionOutput(
            request_id=request_id,
            resource=args.resource,
            rights=rights,
            status=request.status.value,
        )

    def _validate_resume_request(
        self,
        runtime: Any,
        pid: str,
        args: RequestPermissionArgs,
        rights: list[str],
        request_id: str,
        *,
        human: str,
    ) -> None:
        request = runtime.human.get(request_id)
        if request.pid != pid or request.human != human:
            raise ToolExecutionError(
                "Human resume request does not belong to this request_permission call.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )
        payload = request.payload
        requested = payload.get("requested_permission") if isinstance(payload, dict) else None
        context = payload.get("context") if isinstance(payload, dict) else None
        if (
            not isinstance(requested, dict)
            or not isinstance(context, dict)
            or payload.get("type") != "permission_request"
            or requested.get("subject") != pid
            or requested.get("rights") != rights
            or context.get("resource") != args.resource
            or context.get("reason") != args.reason
        ):
            raise ToolExecutionError(
                "Human resume request payload does not match this request_permission call.",
                code=ToolErrorCode.PERMISSION_DENIED,
            )

    @staticmethod
    def _selected_human(runtime: Any, args: RequestPermissionArgs) -> str:
        fields_set = getattr(args, "model_fields_set", set())
        selected = args.human.strip()
        if "human" not in fields_set or not selected:
            return runtime.config.runtime.default_human
        return selected

    def _pending_key(
        self,
        pid: str,
        args: RequestPermissionArgs,
        rights: list[str],
        *,
        human: str,
    ) -> str:
        payload = {
            "pid": pid,
            "human": human,
            "resource": args.resource,
            "rights": rights,
            "reason": args.reason,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
