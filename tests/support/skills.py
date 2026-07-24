from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_libos import Runtime


def write_raw_skill(root: Path, name: str, frontmatter: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n# {name}\n", encoding="utf-8")
    return skill_dir


def write_skill_package(
    root: Path,
    name: str,
    *,
    allowed_tools: list[str] | None = None,
    actions: list[dict[str, Any]] | None = None,
    required_capabilities: list[dict[str, Any]] | None = None,
    jit_tools: list[dict[str, Any]] | None = None,
    scripts: dict[str, str] | None = None,
    extra_resources: dict[str, str] | None = None,
    body: str | None = None,
) -> Path:
    skill_dir = root / name
    metadata: dict[str, str] = {"agent-libos.version": "v0"}
    if actions:
        metadata["agent-libos.actions"] = "references/agent-libos/actions.json"
    if required_capabilities:
        metadata["agent-libos.required-capabilities"] = "references/agent-libos/required-capabilities.json"
    if jit_tools:
        metadata["agent-libos.jit-tools"] = "references/agent-libos/jit-tools.json"

    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = ["---", f"name: {name}", f"description: Adds tools for {name}."]
    if allowed_tools:
        frontmatter_lines.append(f"allowed-tools: {' '.join(allowed_tools)}")
    frontmatter_lines.append("metadata:")
    for key, value in metadata.items():
        frontmatter_lines.append(f"  {key}: {value}")
    frontmatter_lines.append("---")
    selected_body = body or f"# {name}\n\nUse this skill for deterministic checks.\n"
    (skill_dir / "SKILL.md").write_text("\n".join(frontmatter_lines) + "\n\n" + selected_body, encoding="utf-8")

    refs = skill_dir / "references" / "agent-libos"
    refs.mkdir(parents=True, exist_ok=True)
    if actions:
        (refs / "actions.json").write_text(json.dumps(actions), encoding="utf-8")
    if required_capabilities:
        (refs / "required-capabilities.json").write_text(json.dumps(required_capabilities), encoding="utf-8")
    if jit_tools:
        (refs / "jit-tools.json").write_text(json.dumps(jit_tools), encoding="utf-8")
    for path, content in (scripts or {}).items():
        target = skill_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for path, content in (extra_resources or {}).items():
        target = skill_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    package = Runtime.open("local")
    try:
        package.skills.validate_package_path(skill_dir)
    finally:
        package.close()
    return skill_dir
