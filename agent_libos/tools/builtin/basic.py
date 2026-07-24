from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class EchoTool(SyncAgentTool[EchoArgs]):
    name = "echo"
    description = (
        "Return the parsed argument object for tool-plumbing tests. In the LLM function-call "
        "protocol, the top-level `action` key is reserved routing metadata and is removed before "
        "this tool receives the remaining arguments."
    )
    args_schema = EchoArgs
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.interactive_timeout_s)
    tags = ["debug", "deterministic"]

    def run(self, args: EchoArgs, ctx: ToolContext) -> dict[str, Any]:
        return args.model_dump()


class ParsePytestLogArgs(BaseModel):
    log: str = Field(
        description="Captured pytest output text to summarize; this tool does not run pytest or read a log file."
    )


class ParsePytestLogOutput(BaseModel):
    failed: list[str]
    errors: list[str]
    assertions: list[str]
    failure_count: int


class ParsePytestLogTool(SyncAgentTool[ParsePytestLogArgs]):
    name = "parse_pytest_log"
    description = (
        "Heuristically extract FAILED lines, error lines, and AssertionError lines from supplied pytest output. "
        "This does not run tests and is not an authoritative pass/fail check; provide the relevant raw output."
    )
    args_schema = ParsePytestLogArgs
    output_schema = ParsePytestLogOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, timeout_s=_TOOL_DEFAULTS.interactive_timeout_s)
    tags = ["coding", "pytest", "parser"]

    def run(self, args: ParsePytestLogArgs, ctx: ToolContext) -> ParsePytestLogOutput:
        failed: list[str] = []
        errors: list[str] = []
        assertions: list[str] = []
        for line in args.log.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                failed.append(stripped)
            elif re.match(r"^E\s+", stripped):
                errors.append(stripped[2:])
            elif "AssertionError" in stripped:
                assertions.append(stripped)
        return ParsePytestLogOutput(
            failed=failed,
            errors=errors,
            assertions=assertions,
            failure_count=len(failed) or len(assertions) or len(errors),
        )
