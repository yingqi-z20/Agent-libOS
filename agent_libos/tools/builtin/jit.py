from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import ToolSpec
from agent_libos.models.exceptions import ValidationError as LibOSValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.tools.observability import ensure_json_size
from agent_libos.tools.sandbox import compact_validation_diagnostic

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_MODEL_DIAGNOSTIC_PREVIEW_MULTIPLIER = 2
_MODEL_LOG_PREVIEW_MULTIPLIER = 4


class ProposeJitToolArgs(BaseModel):
    name: str = Field(description="Name of the TypeScript JIT tool to create.")
    description: str = Field(description="Human-readable tool description.")
    source_code: str = Field(
        max_length=_TOOL_DEFAULTS.jit_source_max_chars,
        description="TypeScript source exporting run(args, libos).",
    )
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    tests: list[dict[str, Any]] = Field(default_factory=list, max_length=_TOOL_DEFAULTS.jit_tests_max_count)

    @field_validator("tests")
    @classmethod
    def _validate_test_sizes(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, test in enumerate(value, start=1):
            try:
                ensure_json_size(test, _TOOL_DEFAULTS.jit_test_case_max_bytes, f"JIT test {index}")
            except LibOSValidationError as exc:
                raise ValueError(str(exc)) from exc
        return value


class ProposeJitToolOutput(BaseModel):
    candidate_id: str
    name: str
    language: str


class ValidateJitToolArgs(BaseModel):
    candidate_id: str


class ValidateJitToolOutput(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    logs: str


class RegisterJitToolArgs(BaseModel):
    candidate_id: str


class RegisterJitToolOutput(BaseModel):
    tool_id: str
    name: str
    scope: str


class ProposeJitTool(SyncAgentTool[ProposeJitToolArgs]):
    name = "propose_jit_tool"
    description = (
        "Propose a Deno/TypeScript JIT tool candidate. The source must export run(args, libos); "
        "libOS access inside the tool happens through libos.syscall()."
    )
    args_schema = ProposeJitToolArgs
    output_schema = ProposeJitToolOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "tool.write"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["jit", "tool", "typescript"]

    def run(self, args: ProposeJitToolArgs, ctx: ToolContext) -> ProposeJitToolOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        candidate_id = runtime.tools.propose(
            ctx.pid,
            ToolSpec(
                name=args.name,
                description=args.description,
                input_schema=args.input_schema,
                output_schema=args.output_schema,
                tags=["jit", "typescript"],
                metadata={"language": "typescript"},
            ),
            source_code=args.source_code,
            tests=args.tests,
        )
        return ProposeJitToolOutput(candidate_id=candidate_id, name=args.name, language="typescript")


class ValidateJitTool(SyncAgentTool[ValidateJitToolArgs]):
    name = "validate_jit_tool"
    description = "Validate a proposed Deno/TypeScript JIT tool with static checks and candidate tests."
    args_schema = ValidateJitToolArgs
    output_schema = ValidateJitToolOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"jit.validate", "tool.validate"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["jit", "tool", "typescript", "validation"]

    def run(self, args: ValidateJitToolArgs, ctx: ToolContext) -> ValidateJitToolOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        validation = runtime.tools.validate(args.candidate_id, pid=ctx.pid)
        tool_config = runtime.config.tools
        diagnostic_chars = (
            tool_config.tool_observability_preview_chars
            * _MODEL_DIAGNOSTIC_PREVIEW_MULTIPLIER
        )
        log_chars = min(
            tool_config.jit_validation_log_max_chars,
            tool_config.tool_observability_preview_chars
            * _MODEL_LOG_PREVIEW_MULTIPLIER,
        )
        return ValidateJitToolOutput(
            ok=validation.ok,
            errors=_model_validation_items(
                validation.errors,
                max_items=tool_config.jit_tests_max_count,
                max_chars=diagnostic_chars,
            ),
            warnings=_model_validation_items(
                validation.warnings,
                max_items=tool_config.jit_tests_max_count,
                max_chars=diagnostic_chars,
            ),
            logs=compact_validation_diagnostic(
                validation.logs,
                max_chars=log_chars,
                head_tail=True,
            ),
        )


class RegisterJitTool(SyncAgentTool[RegisterJitToolArgs]):
    name = "register_jit_tool"
    description = "Register a validated Deno/TypeScript JIT tool into the current process tool table."
    args_schema = RegisterJitToolArgs
    output_schema = RegisterJitToolOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"tool.write", "tool.table"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["jit", "tool", "typescript", "registration"]

    def run(self, args: RegisterJitToolArgs, ctx: ToolContext) -> RegisterJitToolOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        handle = runtime.tools.register(ctx.pid, args.candidate_id, approver=ctx.pid)
        return RegisterJitToolOutput(tool_id=handle.tool_id, name=handle.name, scope=handle.scope)


def _model_validation_items(
    values: list[str],
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    compacted = [
        compact_validation_diagnostic(
            value,
            max_chars=max_chars,
            head_tail=True,
        )
        for value in values
    ]
    if len(compacted) <= max_items:
        return compacted

    digest_input = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8", errors="replace")
    marker = (
        "[validation diagnostics omitted "
        f"count={len(compacted) - max_items} "
        f"sha256={hashlib.sha256(digest_input).hexdigest()}]"
    )
    retained_items = max(0, max_items - 1)
    head_items = (retained_items + 1) // 2
    tail_items = retained_items - head_items
    return (
        compacted[:head_items]
        + [marker]
        + (compacted[-tail_items:] if tail_items else [])
    )
