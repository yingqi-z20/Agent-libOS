from __future__ import annotations

import json
from copy import deepcopy
from importlib import resources
from pathlib import Path

import pytest

import agent_libos.skills.builtin_catalog as builtin_catalog_module
from agent_libos import Runtime
from agent_libos.llm.prompt import build_user_prompt
from agent_libos.models import MaterializedContext
from agent_libos.models.exceptions import ValidationError
from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_CATALOG_METADATA_MAX_BYTES,
    BUILTIN_SKILL_IDS,
    BUILTIN_SKILL_MAX_TOOLS,
    BuiltinSkillCatalog,
    get_builtin_skill_catalog,
)
from agent_libos.skills.schema import SkillPackage
from agent_libos.utils.yaml_loader import load_yaml_mapping


WORKSPACE_EDITING_SKILL = "agent-libos-workspace-editing"


def test_builtin_skill_packages_use_standard_allowed_tools_scalar() -> None:
    root = resources.files("agent_libos.skills.builtin")
    catalog = get_builtin_skill_catalog()

    for skill_id in BUILTIN_SKILL_IDS:
        raw = root.joinpath(skill_id, "SKILL.md").read_text(encoding="utf-8")
        frontmatter_text = raw.split("---", 2)[1]
        frontmatter = load_yaml_mapping(frontmatter_text)

        assert isinstance(frontmatter["allowed-tools"], str)
        assert frontmatter["allowed-tools"].split() == catalog.get(skill_id).allowed_tools


def test_builtin_skill_parser_rejects_legacy_allowed_tools_sequence() -> None:
    raw = """---
name: agent-libos-example
description: Exercise strict built-in package validation.
allowed-tools:
  - echo
---
# Example
"""

    with pytest.raises(ValidationError, match="space-separated allowed-tools string"):
        builtin_catalog_module._parse_skill_markdown(
            raw,
            expected_name="agent-libos-example",
        )


@pytest.mark.parametrize(
    "invalid_name",
    ("agent-libos-example-", "agent-libos--example"),
)
def test_builtin_skill_parser_enforces_standard_hyphen_rules(invalid_name: str) -> None:
    raw = f"""---
name: {invalid_name}
description: Exercise strict built-in package validation.
allowed-tools: echo
---
# Example
"""

    with pytest.raises(ValidationError, match="invalid built-in Skill name"):
        builtin_catalog_module._parse_skill_markdown(raw, expected_name=invalid_name)


def test_builtin_skill_catalog_has_unique_bounded_ownership() -> None:
    catalog = get_builtin_skill_catalog()
    packages = catalog.list()

    assert len(packages) == 26
    assert tuple(package.skill_id for package in packages) == BUILTIN_SKILL_IDS

    owners: dict[str, str] = {}
    for package in packages:
        assert 1 <= len(package.allowed_tools) <= BUILTIN_SKILL_MAX_TOOLS
        assert catalog.metadata(package.skill_id)["catalog_scope"] == "builtin"
        for tool_name in package.allowed_tools:
            assert tool_name not in owners
            owners[tool_name] = package.skill_id
            assert catalog.skill_for_tool(tool_name) == package.skill_id

    assert len(owners) == 99


def test_workspace_editing_skill_routes_requested_baseline_before_writes() -> None:
    instructions = get_builtin_skill_catalog().get(
        "agent-libos-workspace-editing"
    ).instructions

    assert "stop before writing" in instructions
    assert "command-execution Skill" in instructions
    assert "later passing run cannot replace this baseline" in instructions


def test_command_execution_skill_routes_all_git_argv_to_typed_tools() -> None:
    instructions = get_builtin_skill_catalog().get(
        "agent-libos-command-execution"
    ).instructions

    assert "Never pass a `git` argv" in instructions
    assert "read-only `git status` or `git diff`" in instructions
    assert "typed tool" in instructions


def test_builtin_skill_prompt_catalog_metadata_stays_within_budget() -> None:
    catalog = get_builtin_skill_catalog()
    payload = [
        {
            "skill_id": package.skill_id,
            "description": package.description,
            "active": False,
        }
        for package in catalog.list()
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert catalog.prompt_catalog_metadata_size_bytes == len(encoded)
    assert len(encoded) <= BUILTIN_SKILL_CATALOG_METADATA_MAX_BYTES


def test_builtin_skill_prompt_catalog_metadata_budget_fails_closed(monkeypatch) -> None:
    actual_size = get_builtin_skill_catalog().prompt_catalog_metadata_size_bytes
    monkeypatch.setattr(
        builtin_catalog_module,
        "BUILTIN_SKILL_CATALOG_METADATA_MAX_BYTES",
        actual_size - 1,
    )

    with pytest.raises(ValidationError, match="prompt catalog metadata exceeds"):
        BuiltinSkillCatalog()


def test_builtin_skill_catalog_exactly_covers_registered_core_tools(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "builtin-catalog-core-coverage.sqlite")
    try:
        registered = {str(row["name"]) for row in runtime.tools.list()}
        owned = {
            tool_name
            for package in get_builtin_skill_catalog().list()
            for tool_name in package.allowed_tools
        }
        assert owned == registered
        assert {"discover_tool_groups", "activate_tool_group"}.isdisjoint(registered)
    finally:
        runtime.close()


def test_registered_skills_cannot_claim_reserved_builtin_ids(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "reserved-builtin-skill.sqlite")
    try:
        package = SkillPackage(
            skill_id="agent-libos-not-a-runtime-asset",
            name="agent-libos-not-a-runtime-asset",
            description="Attempt to claim the built-in Skill namespace.",
            instructions="This package must never be registered.",
        )

        with pytest.raises(ValidationError, match="reserved|built-in"):
            runtime.skills.register_skill_package(
                package,
                actor="test.host",
                require_capability=False,
            )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("image_id", "expected_schema_count"),
    [
        ("base-agent:v0", 15),
        ("coding-agent:v0", 14),
        ("review-agent:v0", 14),
    ],
)
def test_builtin_images_start_with_bounded_skill_projection(
    tmp_path: Path,
    image_id: str,
    expected_schema_count: int,
) -> None:
    runtime = Runtime.open(tmp_path / f"{image_id.replace(':', '-')}.sqlite")
    try:
        pid = runtime.process.spawn(image=image_id, goal="inspect built-in Skill projection")
        process = runtime.process.get(pid)

        assert len(runtime.tools.openai_tool_schemas(pid)) == expected_schema_count
        assert len(process.model_tool_table) == expected_schema_count
        assert len(process.tool_table) > len(process.model_tool_table)
    finally:
        runtime.close()


def test_builtin_discovery_ignores_registered_search_and_page_limit(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "complete-builtin-discovery.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="inspect the complete supported built-in Skill catalog",
        )
        expected_ids = [
            item["skill_id"]
            for item in runtime.skills.available_builtin_prompt_context(pid)
        ]

        discovered = runtime.skills.discover_skills_result(
            text="workspace",
            actor=pid,
            limit=1,
        )

        assert len(expected_ids) > 1
        assert [item["skill_id"] for item in discovered["skills"]] == expected_ids
        assert discovered["catalog_scope"] == "builtin_only"
        assert discovered["has_more"] is False
    finally:
        runtime.close()


def test_builtin_activation_needs_no_skill_capability_and_preserves_full_tool_table(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-activation.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="activate workspace editing guidance",
        )
        before = runtime.process.get(pid)
        full_tool_table = dict(before.tool_table)
        capabilities_before = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=pid)
        }
        assert all(
            not capability.resource.startswith("skill:")
            for capability in runtime.store.list_capabilities(subject=pid)
        )

        result = runtime.skills.activate_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )

        after = runtime.process.get(pid)
        capabilities_after = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=pid)
        }
        assert result["skill_id"] == WORKSPACE_EDITING_SKILL
        assert after.tool_table == full_tool_table
        assert capabilities_after == capabilities_before
        assert set(result["tool_names"]).issubset(after.model_tool_table)
        assert after.loaded_skills[WORKSPACE_EDITING_SKILL]["activation_kind"] == "builtin_projection"
    finally:
        runtime.close()


def test_incomplete_builtin_skill_is_hidden_and_activation_fails_atomically(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "incomplete-builtin-skill.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject an incomplete built-in Skill projection",
        )
        process = runtime.process.get(pid)
        incomplete_tool_table = dict(process.tool_table)
        assert incomplete_tool_table.pop("write_text_file", None) is not None
        runtime.store.patch_process(
            pid,
            {"tool_table": incomplete_tool_table},
            expected_revision=process.revision,
        )

        before = runtime.process.get(pid)
        before_model_table = dict(before.model_tool_table)
        before_loaded_skills = deepcopy(before.loaded_skills)
        before_capabilities = {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=pid)
        }

        discovered = runtime.skills.discover_skills_result(actor=pid)
        assert discovered["catalog_scope"] == "builtin_only"
        assert WORKSPACE_EDITING_SKILL not in {
            item["skill_id"] for item in discovered["skills"]
        }

        with pytest.raises(ValidationError, match="image|tool|authorized|available"):
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )

        after = runtime.process.get(pid)
        assert after.tool_table == incomplete_tool_table
        assert after.model_tool_table == before_model_table
        assert after.loaded_skills == before_loaded_skills
        assert {
            capability.cap_id
            for capability in runtime.store.list_capabilities(subject=pid)
        } == before_capabilities
    finally:
        runtime.close()


def test_unload_builtin_skill_restores_model_projection(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "unload-builtin-skill.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="activate then unload workspace editing guidance",
        )
        before = runtime.process.get(pid)
        full_tool_table = dict(before.tool_table)
        initial_model_table = dict(before.model_tool_table)

        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        active = runtime.process.get(pid)
        assert set(
            get_builtin_skill_catalog().get(WORKSPACE_EDITING_SKILL).allowed_tools
        ).issubset(active.model_tool_table)

        runtime.skills.unload_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        after = runtime.process.get(pid)
        assert after.tool_table == full_tool_table
        assert after.model_tool_table == initial_model_table
        assert WORKSPACE_EDITING_SKILL not in after.loaded_skills
    finally:
        runtime.close()


def test_cross_image_fork_drops_incompatible_builtin_projections(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "cross-image-builtin-projection.sqlite")
    try:
        parent = runtime.process.spawn(
            image="coding-agent:v0",
            goal="prepare a broader parent projection",
        )
        runtime.skills.activate_skill(parent, WORKSPACE_EDITING_SKILL, actor=parent)

        child = runtime.process.fork(
            parent,
            goal="compress context only",
            image="context-compressor:v0",
        )
        child_process = runtime.process.get(child)

        assert child_process.tool_table == {
            "process_exit": child_process.tool_table["process_exit"]
        }
        assert child_process.model_tool_table == child_process.tool_table
        assert child_process.loaded_skills == {}
    finally:
        runtime.close()


def test_cross_image_fork_rebases_registered_skill_before_unload(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "cross-image-registered-skill-rebase.sqlite")
    ordinary_skill_id = "registered-navigation-overlap"
    overlap_tool = "discover_skills"
    try:
        parent = runtime.process.spawn(
            image="coding-agent:v0",
            goal="carry one ordinary Skill across a narrower image fork",
        )
        assert "agent-libos-skill-navigation" in runtime.process.get(parent).loaded_skills
        runtime.skills.register_skill_package(
            SkillPackage(
                skill_id=ordinary_skill_id,
                name=ordinary_skill_id,
                description="Expose one tool that overlaps built-in navigation.",
                instructions="Use discover_skills only when Skill discovery is needed.",
                allowed_tools=[overlap_tool],
            ),
            actor="test.host",
            require_capability=False,
        )
        runtime.skills.activate_skill(
            parent,
            ordinary_skill_id,
            actor="test.host",
            require_capability=False,
        )
        parent_loaded = runtime.process.get(parent).loaded_skills[ordinary_skill_id]
        assert overlap_tool in parent_loaded["base_tool_ids"]
        assert overlap_tool in parent_loaded["base_model_tool_ids"]

        child = runtime.process.fork(
            parent,
            goal="retain only the ordinary Skill on the compressor image",
            image="context-compressor:v0",
        )
        active = runtime.process.get(child)
        process_exit_id = active.tool_table["process_exit"]

        assert set(active.loaded_skills) == {ordinary_skill_id}
        assert active.loaded_skills[ordinary_skill_id]["base_tool_ids"] == {}
        assert active.loaded_skills[ordinary_skill_id]["base_model_tool_ids"] == {}
        assert overlap_tool in active.tool_table
        assert overlap_tool in active.model_tool_table

        runtime.skills.unload_skill(
            child,
            ordinary_skill_id,
            actor="test.host",
            require_capability=False,
        )
        unloaded = runtime.process.get(child)

        assert unloaded.loaded_skills == {}
        assert unloaded.tool_table == {"process_exit": process_exit_id}
        assert unloaded.model_tool_table == unloaded.tool_table
    finally:
        runtime.close()


def test_prompt_discloses_builtin_metadata_before_body(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "builtin-prompt-disclosure.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="inspect progressive Skill disclosure",
        )
        package = get_builtin_skill_catalog().get(WORKSPACE_EDITING_SKILL)
        assert package is not None
        body_marker = _distinct_instruction_line(
            package.instructions,
            description=package.description,
        )

        available_before = runtime.skills.available_builtin_prompt_context(pid)
        editing_before = next(
            item
            for item in available_before
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert editing_before == {
            "skill_id": WORKSPACE_EDITING_SKILL,
            "description": package.description,
            "active": False,
        }
        prompt_before = _render_prompt(runtime, pid)
        assert WORKSPACE_EDITING_SKILL in prompt_before
        assert package.description in prompt_before
        assert body_marker not in prompt_before

        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        editing_after = next(
            item
            for item in runtime.skills.available_builtin_prompt_context(pid)
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert editing_after["active"] is True
        prompt_after = _render_prompt(runtime, pid)
        assert prompt_after.count(body_marker) == 1
        assert "package_snapshot" not in prompt_after
    finally:
        runtime.close()


def test_prompt_does_not_render_raw_loaded_skill_state(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "builtin-prompt-loaded-state.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="render only validated built-in Skill context",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        process = runtime.process.get(pid)
        loaded_skills = deepcopy(process.loaded_skills)
        loaded = loaded_skills[WORKSPACE_EDITING_SKILL]
        skill_md_marker = "RAW_SKILL_MD_CONTENT_MUST_NOT_ENTER_PROMPT"
        resource_marker = "RAW_SKILL_RESOURCE_CONTENT_MUST_NOT_ENTER_PROMPT"
        loaded["raw_skill_md"] = skill_md_marker
        loaded["raw_resource_content"] = resource_marker
        runtime.store.patch_process(
            pid,
            {"loaded_skills": loaded_skills},
            expected_revision=process.revision,
        )

        skills = runtime.skills.prompt_context(pid)
        assert any(
            item["skill_id"] == WORKSPACE_EDITING_SKILL
            for item in skills
        )

        prompt = _render_prompt(runtime, pid)
        assert skill_md_marker not in prompt
        assert resource_marker not in prompt
        assert "raw_skill_md" not in prompt
        assert "raw_resource_content" not in prompt
        assert "package_snapshot" not in prompt
    finally:
        runtime.close()


def test_prompt_does_not_render_forged_builtin_snapshot_content(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-prompt-forged-snapshot.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject forged built-in Skill prompt content",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        process = runtime.process.get(pid)
        loaded_skills = deepcopy(process.loaded_skills)
        forged_marker = "FORGED_BUILTIN_SKILL_INSTRUCTION_MUST_NOT_ENTER_PROMPT"
        loaded_skills[WORKSPACE_EDITING_SKILL]["package_snapshot"][
            "instructions"
        ] = forged_marker
        runtime.store.patch_process(
            pid,
            {"loaded_skills": loaded_skills},
            expected_revision=process.revision,
        )

        skills = runtime.skills.prompt_context(pid)
        forged = next(
            item
            for item in skills
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert forged["invalid_snapshot"] is True
        assert "snapshot hash" in forged["error"]

        prompt = _render_prompt(runtime, pid)
        assert "'invalid_snapshot': True" in prompt
        assert "loaded Skill snapshot failed validation" in prompt
        assert forged["error"] not in prompt
        assert forged_marker not in prompt
        assert "package_snapshot" not in prompt
    finally:
        runtime.close()


def _render_prompt(runtime: Runtime, pid: str) -> str:
    return build_user_prompt(
        runtime.process.get(pid),
        MaterializedContext(
            text="progressive disclosure context",
            object_refs=[],
            token_count=3,
            omitted_objects=[],
            policy_used="test",
        ),
        events=[],
        capabilities=[],
        tools=[],
        skills=runtime.skills.prompt_context(pid),
        available_skills=runtime.skills.available_builtin_prompt_context(pid),
    )


def _distinct_instruction_line(instructions: str, *, description: str) -> str:
    candidates = sorted(
        (
            line.strip()
            for line in instructions.splitlines()
            if len(line.strip()) >= 24 and line.strip() not in description
        ),
        key=len,
        reverse=True,
    )
    assert candidates
    return candidates[0]
