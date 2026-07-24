from __future__ import annotations

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy

_SHELL_DEFAULTS = DEFAULT_CONFIG.shell
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class RunShellCommandArgs(BaseModel):
    argv: list[str] = Field(
        min_length=1,
        description=(
            "Command argv array. Shell strings and git argv are not accepted; "
            "activate a matching Git Skill and use its typed git_* tool."
        ),
    )
    timeout_s: float = Field(
        default=_TOOL_DEFAULTS.shell_timeout_s,
        gt=0,
        le=_SHELL_DEFAULTS.timeout_hard_limit_s,
        description="Command timeout in seconds.",
    )
    max_stdout_chars: int = Field(
        default=_SHELL_DEFAULTS.max_stdout_chars,
        ge=0,
        le=_SHELL_DEFAULTS.stdout_hard_limit_chars,
        description="Maximum stdout characters returned in the tool result.",
    )
    max_stderr_chars: int = Field(
        default=_SHELL_DEFAULTS.max_stderr_chars,
        ge=0,
        le=_SHELL_DEFAULTS.stderr_hard_limit_chars,
        description="Maximum stderr characters returned in the tool result.",
    )


class RunShellCommandOutput(BaseModel):
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


class RunShellCommandTool(BaseAgentTool[RunShellCommandArgs]):
    name = "run_shell_command"
    description = (
        "Run an argv-only command through the libOS shell primitive. "
        "Git argv, including read-only status/diff, are rejected; use typed git_* tools. "
        "The primitive enforces shell execution policy, configured allow/ask lists, human approval, audit, and events."
    )
    args_schema = RunShellCommandArgs
    output_schema = RunShellCommandOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"shell.execute"},
        timeout_s=None,
    )
    tags = ["shell", "external", "side_effect"]

    async def execute(self, args: RunShellCommandArgs, ctx: ToolContext) -> RunShellCommandOutput:
        runtime = ctx.runtime
        if runtime is None:
            raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
        cwd = runtime.process.working_directory(ctx.pid)
        try:
            result = await runtime.shell.arun(ctx.pid, args.argv, timeout=args.timeout_s, cwd=cwd)
        except TimeoutError as exc:
            raise ToolExecutionError(
                "Shell command timed out.",
                code=ToolErrorCode.TIMEOUT,
                retryable=True,
                details={"argv": args.argv, "timeout_s": args.timeout_s},
            ) from exc
        stdout, stdout_truncated = _truncate(result.stdout, args.max_stdout_chars)
        stderr, stderr_truncated = _truncate(result.stderr, args.max_stderr_chars)
        return RunShellCommandOutput(
            argv=result.argv,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=result.stdout_truncated or stdout_truncated,
            stderr_truncated=result.stderr_truncated or stderr_truncated,
        )


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True
