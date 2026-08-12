from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import SkillPackageChanged, ValidationError
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolExecutionError,
    ToolPolicy,
    ToolResult,
)
from agent_libos.utils.skill_search import SKILL_SEARCH_TEXT_MAX_CHARS

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class DiscoverSkillsArgs(BaseModel):
    text: str | None = Field(
        default=None,
        max_length=SKILL_SEARCH_TEXT_MAX_CHARS,
        description=(
            "Optional text matched against visible Skill ids, names, and descriptions. "
            "Use a task-specific query to load only relevant metadata."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Maximum matching Skill summaries to return across every visible source. "
            "Pass an unquoted JSON integer, for example 5, never the string \"5\"."
        ),
    )


class DiscoveredSkill(BaseModel):
    skill_id: str
    name: str
    version: str
    description: str
    allowed_tools: list[str]
    actions: list[str]
    jit_tools: list[str]
    required_capabilities: list[dict[str, Any]]
    package_sha256: str
    active: bool = False


class DiscoverSkillsOutput(BaseModel):
    skills: list[DiscoveredSkill] = Field(
        description="One bounded page of matching visible Skill metadata."
    )
    has_more: bool = Field(
        default=False,
        description=(
            "Whether more matching visible Skills exist. There is no cursor; refine text or "
            "raise limit within the Host maximum."
        ),
    )
    visibility_limited: bool = Field(
        default=False,
        description=(
            "Whether catalog authority prevented this process from searching every configured "
            "Skill source. This does not identify the source of any returned Skill."
        ),
    )
    next_step: Literal[
        "activate_skill",
        "use_loaded_skill",
        "refine_search",
        "catalog_access_required",
    ] = Field(
        description=(
            "Recommended next lifecycle step. Activate one plausible inactive exact id, use "
            "the loaded snapshot when every returned match is current, refine when a complete "
            "visible catalog has no match, or stop discovery and report/request exact catalog "
            "read authority when no match is visible and catalog access is limited; never "
            "repeat an unchanged search."
        )
    )


class ActivateSkillArgs(BaseModel):
    skill_id: str = Field(description="Exact Skill id returned by discover_skills.")
    expected_package_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Exact package_sha256 from the selected discover_skills result. Activation "
            "fails without mutation if the visible package changed since discovery."
        ),
    )


class ActivatedSkill(BaseModel):
    pid: str
    skill_id: str
    name: str
    version: str
    tool_names: list[str]
    tool_ids: dict[str, str]
    jit_tool_ids: dict[str, str]
    instructions_hash: str
    package_sha256: str


class ActivateSkillOutput(BaseModel):
    result: ActivatedSkill


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
        gt=0,
        strict=True,
        description="Optional maximum allowed resource size; a larger resource is rejected rather than truncated.",
    )


class ReadSkillResourceOutput(BaseModel):
    resource: dict[str, Any]


class UnloadSkillArgs(BaseModel):
    skill_id: str = Field(description="Exact id of a Skill currently loaded in this process.")


class UnloadedSkill(BaseModel):
    pid: str
    skill_id: str
    removed_tools: list[str]


class UnloadSkillOutput(BaseModel):
    result: UnloadedSkill


class DiscoverSkillsTool(SyncAgentTool[DiscoverSkillsArgs]):
    name = "discover_skills"
    description = (
        "Search visible Agent Skills by task intent and return one relevance-ranked metadata "
        "page. Start with two to four concrete domain/action terms and omit limit or use at "
        "least 5 as an unquoted JSON integer. A multi-capability query can return separate narrowly owned Skills; activate "
        "each relevant exact id with that row's package hash instead of repeating discovery. "
        "visibility_limited does not "
        "invalidate returned matches. Discovery does not load instructions, expose domain "
        "tools, or grant authority."
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
        discovered = _runtime(ctx).skills.discover_skills_result(
            args.text,
            actor=ctx.pid,
            limit=args.limit,
        )
        skills = discovered["skills"]
        visibility_limited = (
            discovered.get("catalog_scope") == "visibility_limited"
        )
        next_step: Literal[
            "activate_skill",
            "use_loaded_skill",
            "refine_search",
            "catalog_access_required",
        ]
        if not skills:
            next_step = (
                "catalog_access_required"
                if visibility_limited
                else "refine_search"
            )
        elif all(bool(skill.get("active")) for skill in skills):
            next_step = "use_loaded_skill"
        else:
            next_step = "activate_skill"
        return DiscoverSkillsOutput(
            skills=skills,
            has_more=bool(discovered["has_more"]),
            visibility_limited=visibility_limited,
            next_step=next_step,
        )


class ActivateSkillTool(SyncAgentTool[ActivateSkillArgs]):
    name = "activate_skill"
    description = (
        "Load one Skill's instructions and tool bindings into this process by exact id and "
        "the package hash returned by discovery. "
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

    def run(self, args: ActivateSkillArgs, ctx: ToolContext) -> ToolResult:
        try:
            result = _runtime(ctx).skills.activate_skill(
                ctx.pid,
                args.skill_id,
                actor=ctx.pid,
                expected_package_sha256=args.expected_package_sha256,
            )
        except SkillPackageChanged as exc:
            raise SkillPackageChanged(
                _source_neutral_error_message(str(exc))
            ) from exc
        except ValidationError as exc:
            raise ValidationError(_source_neutral_error_message(str(exc))) from exc
        output = ActivateSkillOutput(result=result)
        activated = output.result
        return ToolResult.success(
            data=output.model_dump(),
            model_data={
                "result": {
                    "skill_id": activated.skill_id,
                    "name": activated.name,
                    "version": activated.version,
                    "tool_names": activated.tool_names,
                }
            },
        )


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
        try:
            resource = _runtime(ctx).skills.read_skill_resource(
                ctx.pid,
                args.skill_id,
                args.path,
                actor=ctx.pid,
                max_bytes=args.max_bytes,
            )
        except ValidationError as exc:
            raise ValidationError(_source_neutral_error_message(str(exc))) from exc
        return ReadSkillResourceOutput(resource=resource)


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

    def run(self, args: UnloadSkillArgs, ctx: ToolContext) -> ToolResult:
        try:
            result = _runtime(ctx).skills.unload_skill(
                ctx.pid,
                args.skill_id,
                actor=ctx.pid,
            )
        except ValidationError as exc:
            raise ValidationError(_source_neutral_error_message(str(exc))) from exc
        output = UnloadSkillOutput(result=result)
        unloaded = output.result
        return ToolResult.success(
            data=output.model_dump(),
            model_data={
                "result": {
                    "skill_id": unloaded.skill_id,
                    "removed_tools": unloaded.removed_tools,
                }
            },
        )


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.")
    return ctx.runtime


def _source_neutral_error_message(message: str) -> str:
    """Remove Host-only package provenance labels from model tool failures."""

    replacements = (
        ("reserved built-in Skill", "reserved Skill"),
        ("built-in loaded Skill", "loaded Skill"),
        ("built-in tool Skill", "Skill"),
        ("built-in Skill", "Skill"),
        ("built-in tool", "tool"),
    )
    result = message
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    return result
