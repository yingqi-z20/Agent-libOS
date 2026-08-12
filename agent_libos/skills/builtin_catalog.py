from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

from agent_libos.models.exceptions import ValidationError
from agent_libos.skills.schema import SkillPackage, SkillResource
from agent_libos.utils.yaml_loader import load_yaml_mapping

BUILTIN_SKILL_PREFIX = "agent-libos-"
BUILTIN_SKILL_SOURCE_TYPE = "builtin"
BUILTIN_SKILL_CATALOG_SCOPE = "builtin"
BUILTIN_SKILL_PACKAGE = "agent_libos.skills.builtin"
BUILTIN_SKILL_MAX_FILE_BYTES = 24 * 1_024
BUILTIN_SKILL_MAX_INSTRUCTION_BYTES = 16 * 1_024
BUILTIN_SKILL_MAX_TOOLS = 9
BUILTIN_SKILL_MAX_TOOL_NAME_CHARS = 128
BUILTIN_SKILL_TOOL_COUNT = 101

BUILTIN_SKILL_IDS = (
    "agent-libos-skill-navigation",
    "agent-libos-authority-basics",
    "agent-libos-capability-delegation",
    "agent-libos-human-collaboration",
    "agent-libos-runtime-session",
    "agent-libos-workspace-navigation",
    "agent-libos-workspace-editing",
    "agent-libos-command-execution",
    "agent-libos-test-log-analysis",
    "agent-libos-tool-protocol-diagnostics",
    "agent-libos-object-memory",
    "agent-libos-object-file-transfer",
    "agent-libos-object-tasks",
    "agent-libos-child-processes",
    "agent-libos-checkpoints",
    "agent-libos-agent-images",
    "agent-libos-jit-tool-authoring",
    "agent-libos-jsonrpc",
    "agent-libos-mcp",
    "agent-libos-git-inspection",
    "agent-libos-git-change-recording",
    "agent-libos-git-branches-worktrees",
    "agent-libos-git-integration-recovery",
    "agent-libos-git-patch-objects",
    "agent-libos-git-remotes",
    "agent-libos-git-pull-requests",
)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
_FRONTMATTER_FIELDS = {"name", "description", "allowed-tools"}


class BuiltinSkillCatalog:
    """Immutable catalog of package-distributed tool guidance Skills.

    Built-in packages are data bundled with Agent libOS, not records in the
    mutable workspace/global/runtime Skill registry. Returned packages are
    defensive copies so callers cannot alter the catalog's authority-neutral
    tool ownership map.
    """

    def __init__(self, package: str = BUILTIN_SKILL_PACKAGE) -> None:
        root = resources.files(package)
        discovered = self._discover_skill_ids(root)
        expected = set(BUILTIN_SKILL_IDS)
        if discovered != expected:
            missing = sorted(expected - discovered)
            unexpected = sorted(discovered - expected)
            raise ValidationError(
                "invalid built-in Skill catalog contents: "
                f"missing={missing}, unexpected={unexpected}"
            )

        packages: dict[str, SkillPackage] = {}
        owner_by_tool: dict[str, str] = {}
        for skill_id in BUILTIN_SKILL_IDS:
            package_root = root.joinpath(skill_id)
            self._validate_package_files(package_root, skill_id)
            skill = self._load_package(package_root.joinpath("SKILL.md"), skill_id)
            packages[skill_id] = skill
            for tool_name in skill.allowed_tools:
                previous = owner_by_tool.get(tool_name)
                if previous is not None:
                    raise ValidationError(
                        f"built-in tool {tool_name!r} is owned by both {previous!r} and {skill_id!r}"
                    )
                owner_by_tool[tool_name] = skill_id

        if len(owner_by_tool) != BUILTIN_SKILL_TOOL_COUNT:
            raise ValidationError(
                "built-in Skill catalog must own exactly "
                f"{BUILTIN_SKILL_TOOL_COUNT} tools, found {len(owner_by_tool)}"
            )
        self._packages = packages
        self._owner_by_tool = owner_by_tool

    def get(self, skill_id: str) -> SkillPackage | None:
        """Return a defensive copy of one built-in Skill package."""

        package = self._packages.get(skill_id)
        return _copy_package(package) if package is not None else None

    def list(self) -> tuple[SkillPackage, ...]:
        """Return all built-in Skills in stable catalog order."""

        return tuple(_copy_package(self._packages[skill_id]) for skill_id in BUILTIN_SKILL_IDS)

    def skill_for_tool(self, tool_name: str) -> str | None:
        """Return the unique built-in Skill that owns ``tool_name``."""

        return self._owner_by_tool.get(tool_name)

    def is_builtin_id(self, skill_id: str) -> bool:
        """Return whether an identifier is reserved for built-in Skills."""

        return isinstance(skill_id, str) and skill_id.startswith(BUILTIN_SKILL_PREFIX)

    def metadata(self, skill_id: str) -> dict[str, Any] | None:
        """Return discovery/provenance metadata without the instruction body."""

        package = self._packages.get(skill_id)
        if package is None:
            return None
        return {
            "skill_id": package.skill_id,
            "name": package.name,
            "description": package.description,
            "version": package.version,
            "source_type": BUILTIN_SKILL_SOURCE_TYPE,
            "source": f"builtin:{package.skill_id}",
            "package_sha256": package.package_sha256,
            "allowed_tools": list(package.allowed_tools),
            "catalog_scope": BUILTIN_SKILL_CATALOG_SCOPE,
        }

    def summary(self, skill_id: str) -> dict[str, Any] | None:
        """Alias for ``metadata`` used by discovery and prompt builders."""

        return self.metadata(skill_id)

    @staticmethod
    def _discover_skill_ids(root: Traversable) -> set[str]:
        discovered: set[str] = set()
        for child in root.iterdir():
            if child.is_dir() and child.joinpath("SKILL.md").is_file():
                discovered.add(child.name)
        return discovered

    @staticmethod
    def _validate_package_files(root: Traversable, skill_id: str) -> None:
        entries = sorted(child.name for child in root.iterdir())
        if entries != ["SKILL.md"]:
            raise ValidationError(
                f"built-in Skill {skill_id!r} may contain only SKILL.md, found {entries}"
            )

    @staticmethod
    def _load_package(skill_md: Traversable, expected_name: str) -> SkillPackage:
        with skill_md.open("rb") as handle:
            raw = handle.read(BUILTIN_SKILL_MAX_FILE_BYTES + 1)
        if len(raw) > BUILTIN_SKILL_MAX_FILE_BYTES:
            raise ValidationError(
                f"built-in Skill {expected_name!r} SKILL.md exceeds "
                f"{BUILTIN_SKILL_MAX_FILE_BYTES} bytes"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"built-in Skill {expected_name!r} is not UTF-8") from exc
        frontmatter, instructions = _parse_skill_markdown(text, expected_name=expected_name)
        if len(instructions.encode("utf-8")) > BUILTIN_SKILL_MAX_INSTRUCTION_BYTES:
            raise ValidationError(
                f"built-in Skill {expected_name!r} instructions exceed "
                f"{BUILTIN_SKILL_MAX_INSTRUCTION_BYTES} bytes"
            )
        _validate_builtin_tool_guidance(
            instructions,
            allowed_tools=frontmatter["allowed_tools"],
            name=expected_name,
        )
        resource = SkillResource(
            path="SKILL.md",
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            kind="text",
            content=text,
        )
        package = SkillPackage(
            skill_id=expected_name,
            name=expected_name,
            description=frontmatter["description"],
            instructions=instructions,
            allowed_tools=frontmatter["allowed_tools"],
            resources=[resource],
        )
        return replace(package, package_sha256=_package_hash(package))


@lru_cache(maxsize=1)
def get_builtin_skill_catalog() -> BuiltinSkillCatalog:
    """Return the process-wide immutable built-in Skill catalog."""

    return BuiltinSkillCatalog()


def _parse_skill_markdown(text: str, *, expected_name: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    data, end_index = _parse_frontmatter(lines)
    name = _validate_builtin_skill_name(data.get("name"), expected_name=expected_name)
    description = _validate_builtin_skill_description(data.get("description"), name=name)
    allowed_tools = _validate_builtin_allowed_tools(data.get("allowed-tools"), name=name)
    instructions = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    if not instructions.strip():
        raise ValidationError(f"built-in Skill {name!r} requires instructions")
    return {
        "name": name,
        "description": description,
        "allowed_tools": allowed_tools,
    }, instructions


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, Any], int]:
    if not lines or lines[0].strip() != "---":
        raise ValidationError("built-in SKILL.md must start with YAML frontmatter delimited by ---")
    end_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if end_index is None:
        raise ValidationError("built-in SKILL.md frontmatter is missing closing ---")
    data = load_yaml_mapping("\n".join(lines[1:end_index]))
    unknown = sorted(set(data) - _FRONTMATTER_FIELDS)
    if unknown:
        raise ValidationError(f"unknown built-in SKILL.md frontmatter fields: {unknown}")
    return data, end_index


def _validate_builtin_skill_name(raw_name: Any, *, expected_name: str) -> str:
    if not isinstance(raw_name, str):
        raise ValidationError(f"invalid built-in Skill name: {raw_name!r}")
    name = raw_name.strip()
    if len(name) > 64 or not _SKILL_NAME_PATTERN.fullmatch(name):
        raise ValidationError(f"invalid built-in Skill name: {name!r}")
    if name != expected_name or not name.startswith(BUILTIN_SKILL_PREFIX):
        raise ValidationError(
            f"built-in Skill directory/name mismatch: {expected_name!r} != {name!r}"
        )
    return name


def _validate_builtin_skill_description(raw_description: Any, *, name: str) -> str:
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise ValidationError(f"built-in Skill {name!r} requires a description")
    description = raw_description.strip()
    if len(description) > 1_024:
        raise ValidationError(f"built-in Skill {name!r} description exceeds 1024 characters")
    return description


def _validate_builtin_allowed_tools(raw_allowed_tools: Any, *, name: str) -> list[str]:
    # Agent Skills defines ``allowed-tools`` as one space-separated scalar, not
    # a YAML sequence. Built-ins are package-owned, so reject the historical
    # list spelling instead of silently accepting non-standard bundled data.
    if not isinstance(raw_allowed_tools, str) or not raw_allowed_tools.strip():
        raise ValidationError(
            f"built-in Skill {name!r} requires a non-empty space-separated allowed-tools string"
        )
    allowed_tools = raw_allowed_tools.split()
    if len(allowed_tools) > BUILTIN_SKILL_MAX_TOOLS:
        raise ValidationError(
            f"built-in Skill {name!r} exceeds {BUILTIN_SKILL_MAX_TOOLS} allowed tools"
        )
    if any(
        len(tool) > BUILTIN_SKILL_MAX_TOOL_NAME_CHARS
        or not _TOOL_NAME_PATTERN.fullmatch(tool)
        for tool in allowed_tools
    ):
        raise ValidationError(f"built-in Skill {name!r} has an invalid allowed-tools entry")
    if len(set(allowed_tools)) != len(allowed_tools):
        raise ValidationError(f"built-in Skill {name!r} has duplicate allowed-tools entries")
    return allowed_tools


def _validate_builtin_tool_guidance(
    instructions: str,
    *,
    allowed_tools: list[str],
    name: str,
) -> None:
    """Require built-in guidance to route every schema it projects."""

    missing = [
        tool_name
        for tool_name in allowed_tools
        if f"`{tool_name}`" not in instructions
    ]
    if missing:
        raise ValidationError(
            f"built-in Skill {name!r} instructions do not guide allowed tools: "
            f"{', '.join(missing)}"
        )


def _package_hash(package: SkillPackage) -> str:
    payload = {
        "schema_version": package.schema_version,
        "skill_id": package.skill_id,
        "name": package.name,
        "description": package.description,
        "instructions_sha256": _hash_text(package.instructions),
        "version": package.version,
        "license": package.license,
        "compatibility": package.compatibility,
        "metadata": package.metadata,
        "allowed_tools": package.allowed_tools,
        "actions": [asdict(action) for action in package.actions],
        "jit_tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "source_path": tool.source_path,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "source_sha256": _hash_text(tool.source),
                "tests": tool.tests,
                "metadata": tool.metadata,
                **({"timeout_s": tool.timeout_s} if tool.timeout_s is not None else {}),
            }
            for tool in package.jit_tools
        ],
        "required_capabilities": package.required_capabilities,
        "resources": [
            {
                "path": resource.path,
                "sha256": resource.sha256,
                "size_bytes": resource.size_bytes,
                "kind": resource.kind,
                "content_sha256": resource.sha256,
            }
            for resource in package.resources
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_package(package: SkillPackage) -> SkillPackage:
    return replace(
        package,
        metadata=dict(package.metadata),
        allowed_tools=list(package.allowed_tools),
        actions=list(package.actions),
        jit_tools=list(package.jit_tools),
        required_capabilities=[dict(spec) for spec in package.required_capabilities],
        resources=list(package.resources),
        diagnostics=list(package.diagnostics),
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
