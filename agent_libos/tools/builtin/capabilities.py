from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, StrictBool, StrictInt, field_validator

from agent_libos.models import Capability, CapabilityEffect
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy


class ListCapabilitiesArgs(BaseModel):
    include_inactive: bool = Field(default=False, description="Include revoked, disabled, or expired capabilities.")
    limit: StrictInt | None = Field(
        default=None,
        ge=1,
        description="Maximum records in this page; bounded by the Host capability list limit.",
    )
    after_cap_id: str | None = Field(
        default=None,
        min_length=1,
        description="Exclusive capability identifier cursor returned by the previous page.",
    )

    @field_validator("limit")
    @classmethod
    def _positive_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("limit must be a positive integer")
        return value

    @field_validator("after_cap_id")
    @classmethod
    def _non_empty_cursor(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("after_cap_id must be a non-empty string")
        return value


class ListCapabilitiesOutput(BaseModel):
    capabilities: list[dict[str, Any]]
    has_more: bool
    next_cursor: str | None = None


class InspectCapabilityArgs(BaseModel):
    cap_id: str


class InspectCapabilityOutput(BaseModel):
    capability: dict[str, Any]


class DelegateCapabilityArgs(BaseModel):
    child_pid: str
    resource: str
    rights: list[str]
    effect: str = Field(default=CapabilityEffect.ALLOW.value)
    expires_at: str | None = None
    uses_remaining: StrictInt | None = Field(default=None, ge=1)
    delegable: StrictBool = False
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("uses_remaining")
    @classmethod
    def _positive_uses_remaining(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("uses_remaining must be a positive integer")
        return value


class DelegateCapabilityOutput(BaseModel):
    capability: dict[str, Any]


class RevokeCapabilityArgs(BaseModel):
    cap_id: str
    reason: str | None = None


class RevokeCapabilityOutput(BaseModel):
    capability: dict[str, Any]


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return ctx.runtime


def _presentation_budget(runtime: Any) -> int:
    limit = min(
        runtime.config.tools.tool_result_payload_hard_limit_bytes,
        runtime.config.tools.memory_payload_hard_limit_bytes,
    )
    # Reserve half of the carrier for the durable tool envelope, telemetry,
    # and model-facing projection. The capability manager enforces this
    # budget incrementally before materializing another record.
    return max(1, limit // 2)


def _mutation_capability_receipt(
    runtime: Any,
    cap: Capability,
) -> dict[str, Any]:
    """Return success evidence without failing after the mutation committed."""

    try:
        return runtime.capability.inspect_for_presentation(
            cap.cap_id,
            max_bytes=_presentation_budget(runtime),
        )
    except ValidationError:
        # An exceptionally small Host result budget can be unable to carry the
        # complete authority identity.  The mutation has already committed at
        # this point, so return a positive, bounded settlement marker instead
        # of a validation failure that could invite an unsafe duplicate call.
        return {
            "cap_id": cap.cap_id,
            "status": cap.status.value,
            "presentation_omitted": True,
        }


class ListCapabilitiesTool(SyncAgentTool[ListCapabilitiesArgs]):
    name = "list_capabilities"
    description = "List the current process capabilities without granting new authority."
    args_schema = ListCapabilitiesArgs
    output_schema = ListCapabilitiesOutput
    policy = ToolPolicy(side_effects=False)
    tags = ["capability", "authority"]

    def run(self, args: ListCapabilitiesArgs, ctx: ToolContext) -> ListCapabilitiesOutput:
        runtime = _runtime(ctx)
        page = runtime.capability.presentation_page(
            subject=ctx.pid,
            include_inactive=args.include_inactive,
            limit=args.limit,
            after_cap_id=args.after_cap_id,
            max_bytes=_presentation_budget(runtime),
        )
        return ListCapabilitiesOutput(
            capabilities=page.capabilities,
            has_more=page.has_more,
            next_cursor=page.next_cursor,
        )


class InspectCapabilityTool(SyncAgentTool[InspectCapabilityArgs]):
    name = "inspect_capability"
    description = "Inspect one capability owned by the current process."
    args_schema = InspectCapabilityArgs
    output_schema = InspectCapabilityOutput
    policy = ToolPolicy(side_effects=False)
    tags = ["capability", "authority"]

    def run(self, args: InspectCapabilityArgs, ctx: ToolContext) -> InspectCapabilityOutput:
        runtime = _runtime(ctx)
        cap = runtime.store.get_capability(args.cap_id)
        if cap is None:
            raise ToolExecutionError("Capability not found.", code=ToolErrorCode.VALIDATION_ERROR)
        if cap.subject != ctx.pid:
            raise ToolExecutionError("Cannot inspect another process capability.", code=ToolErrorCode.PERMISSION_DENIED)
        return InspectCapabilityOutput(
            capability=runtime.capability.inspect_for_presentation(
                args.cap_id,
                max_bytes=_presentation_budget(runtime),
            )
        )


class DelegateCapabilityTool(SyncAgentTool[DelegateCapabilityArgs]):
    name = "delegate_capability"
    description = "Delegate an attenuated capability to a direct child process."
    args_schema = DelegateCapabilityArgs
    output_schema = DelegateCapabilityOutput
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"capability.write"})
    tags = ["capability", "authority"]

    def run(self, args: DelegateCapabilityArgs, ctx: ToolContext) -> DelegateCapabilityOutput:
        runtime = _runtime(ctx)
        child = runtime.process.get(args.child_pid)
        if child.parent_pid != ctx.pid:
            raise ToolExecutionError("Capabilities can only be delegated to a direct child.", code=ToolErrorCode.PERMISSION_DENIED)
        try:
            cap = runtime.capability.delegate(
                ctx.pid,
                args.child_pid,
                {
                    "resource": args.resource,
                    "rights": args.rights,
                    "effect": args.effect,
                    "expires_at": args.expires_at,
                    "uses_remaining": args.uses_remaining,
                    "delegable": args.delegable,
                    "constraints": args.constraints,
                    "metadata": args.metadata,
                },
                actor=ctx.pid,
            )
        except CapabilityDenied as exc:
            raise ToolExecutionError(str(exc), code=ToolErrorCode.PERMISSION_DENIED) from exc
        return DelegateCapabilityOutput(
            capability=_mutation_capability_receipt(runtime, cap)
        )


class RevokeCapabilityTool(SyncAgentTool[RevokeCapabilityArgs]):
    name = "revoke_capability"
    description = "Revoke a capability when the current process has holder, issuer, revoke, or admin authority."
    args_schema = RevokeCapabilityArgs
    output_schema = RevokeCapabilityOutput
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"capability.write"})
    tags = ["capability", "authority"]

    def run(self, args: RevokeCapabilityArgs, ctx: ToolContext) -> RevokeCapabilityOutput:
        runtime = _runtime(ctx)
        try:
            cap = runtime.capability.revoke(args.cap_id, revoked_by=ctx.pid, reason=args.reason)
        except CapabilityDenied as exc:
            raise ToolExecutionError(str(exc), code=ToolErrorCode.PERMISSION_DENIED) from exc
        return RevokeCapabilityOutput(
            capability=_mutation_capability_receipt(runtime, cap)
        )
