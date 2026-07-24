from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolExecutionError, ToolPolicy

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class DiscoverSkillsArgs(BaseModel):
    text: str | None = Field(
        default=None,
        description=(
            "Optional text matched against registered and Host-catalog Skill ids, names, "
            "and descriptions; applicable built-in Skills are always returned in full."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Maximum matching registered and Host-catalog Skill summaries to return; "
            "applicable built-in Skills do not count toward this limit."
        ),
    )


class DiscoverSkillsOutput(BaseModel):
    skills: list[dict[str, Any]] = Field(
        description=(
            "The complete applicable built-in catalog followed by the requested page of "
            "registered or Host-catalog Skills."
        )
    )
    catalog_scope: str = Field(description="Authority-bounded catalog sources included in this result.")
    has_more: bool = Field(
        default=False,
        description=(
            "Whether another matching registered or Host-catalog Skill exists; the built-in "
            "catalog in this result is already complete."
        ),
    )


class ActivateSkillArgs(BaseModel):
    skill_id: str = Field(description="Exact Skill id returned by discover_skills or the built-in Skill catalog.")


class ActivateSkillOutput(BaseModel):
    result: dict[str, Any]


class ReadSkillResourceArgs(BaseModel):
    skill_id: str = Field(description="Exact id of a Skill already loaded in this process.")
    path: str = Field(
        description=(
            "Package-relative resource path from the loaded snapshot, such as references/foo.md; "
            "absolute and parent paths are invalid."
        )
    )
    max_bytes: int | None = Field(
        default=None,
        description="Optional maximum allowed resource size; a larger resource is rejected rather than truncated.",
    )


class ReadSkillResourceOutput(BaseModel):
    resource: dict[str, Any]


class UnloadSkillArgs(BaseModel):
    skill_id: str = Field(description="Exact id of a Skill currently loaded in this process.")


class UnloadSkillOutput(BaseModel):
    result: dict[str, Any]


class DiscoverSkillsTool(SyncAgentTool[DiscoverSkillsArgs]):
    name = "discover_skills"
    description = (
        "List every applicable built-in tool Skill and search registered Agent Skills visible to this process. "
        "Use the returned exact id with activate_skill; discovery does not activate tools or grant authority."
    )
    args_schema = DiscoverSkillsArgs
    output_schema = DiscoverSkillsOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"skill.read"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["skill", "inspect"]

    def run(self, args: DiscoverSkillsArgs, ctx: ToolContext) -> DiscoverSkillsOutput:
        return DiscoverSkillsOutput(
            **_runtime(ctx).skills.discover_skills_result(
                args.text,
                actor=ctx.pid,
                limit=args.limit,
            )
        )


class ActivateSkillTool(SyncAgentTool[ActivateSkillArgs]):
    name = "activate_skill"
    description = (
        "Load one Skill's instructions and tool bindings into this process by exact id. "
        "Activation changes visibility only where applicable and never bypasses primitive capability or approval checks."
    )
    args_schema = ActivateSkillArgs
    output_schema = ActivateSkillOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"skill.read", "tool.write", "tool.table"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["skill", "activate"]

    def run(self, args: ActivateSkillArgs, ctx: ToolContext) -> ActivateSkillOutput:
        return ActivateSkillOutput(result=_runtime(ctx).skills.activate_skill(ctx.pid, args.skill_id, actor=ctx.pid))


class ReadSkillResourceTool(SyncAgentTool[ReadSkillResourceArgs]):
    name = "read_skill_resource"
    description = (
        "Read a package-relative resource from an already loaded Skill's immutable snapshot. "
        "Use this only when the loaded instructions refer to a bundled reference; it does not read workspace files."
    )
    args_schema = ReadSkillResourceArgs
    output_schema = ReadSkillResourceOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"skill.read"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["skill", "resource", "inspect"]

    def run(self, args: ReadSkillResourceArgs, ctx: ToolContext) -> ReadSkillResourceOutput:
        return ReadSkillResourceOutput(
            resource=_runtime(ctx).skills.read_skill_resource(
                ctx.pid,
                args.skill_id,
                args.path,
                actor=ctx.pid,
                max_bytes=args.max_bytes,
            )
        )


class UnloadSkillTool(SyncAgentTool[UnloadSkillArgs]):
    name = "unload_skill"
    description = (
        "Unload an active Skill, removing its prompt instructions and tool bindings contributed by that activation. "
        "This does not revoke capabilities or remove bindings still owned by the image or another loaded Skill."
    )
    args_schema = UnloadSkillArgs
    output_schema = UnloadSkillOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"tool.table"},
        timeout_s=_TOOL_DEFAULTS.standard_timeout_s,
    )
    tags = ["skill", "unload"]

    def run(self, args: UnloadSkillArgs, ctx: ToolContext) -> UnloadSkillOutput:
        return UnloadSkillOutput(result=_runtime(ctx).skills.unload_skill(ctx.pid, args.skill_id, actor=ctx.pid))


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.")
    return ctx.runtime
