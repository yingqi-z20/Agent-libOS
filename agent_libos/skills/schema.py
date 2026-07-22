from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionSchema:
    name: str
    use_cases: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class JitToolSpec:
    name: str
    description: str
    source_path: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    tests: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None


@dataclass(frozen=True)
class SkillResource:
    path: str
    size_bytes: int
    sha256: str
    kind: str = "text"
    content: str | None = None
    content_base64: str | None = None


@dataclass(frozen=True)
class SkillPackage:
    """Snapshot of a standard Agent Skill package.

    A Skill's public identity is the standard SKILL.md frontmatter ``name``.
    Agent libOS stores a package snapshot so activation and resource reads do
    not keep authority to the original workspace or global filesystem path.
    """

    skill_id: str
    name: str
    description: str
    instructions: str
    version: str = "v0"
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    actions: list[ActionSchema] = field(default_factory=list)
    jit_tools: list[JitToolSpec] = field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    resources: list[SkillResource] = field(default_factory=list)
    package_sha256: str = ""
    diagnostics: list[str] = field(default_factory=list)
    schema_version: int = 1


@dataclass(frozen=True)
class LoadedSkill:
    skill_id: str
    version: str
    source: str | None
    package_sha256: str
    loaded_at: str
    tool_names: list[str]
    tool_ids: dict[str, str]
    jit_tool_ids: dict[str, str]
    instructions_hash: str
    package_snapshot: dict[str, Any] = field(default_factory=dict)
    # Bindings that existed independently of this Skill before activation.
    # They let unload restore image/manual visibility without treating a
    # shared alias as owned by whichever Skill happens to be removed first.
    base_tool_ids: dict[str, str] = field(default_factory=dict)
    base_model_tool_ids: dict[str, str] = field(default_factory=dict)
