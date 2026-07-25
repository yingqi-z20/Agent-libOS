from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.images import DEFAULT_IMAGES
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps


SKILL_BOOTSTRAP_TOOLS = {
    "activate_skill",
    "discover_skills",
    "process_exit",
    "read_skill_resource",
    "unload_skill",
}

AUTHORITY_TOOLS = {
    "inspect_capability",
    "list_capabilities",
    "request_permission",
}

HUMAN_TOOLS = {
    "ask_human",
    "human_output",
}

WORKSPACE_NAVIGATION_TOOLS = {
    "get_working_directory",
    "read_directory",
    "read_text_file",
    "set_working_directory",
}

OBJECT_MEMORY_TOOLS = {
    "append_memory_object",
    "create_memory_namespace",
    "create_memory_object",
    "list_memory_namespace",
    "read_memory_object",
}

INITIAL_TOOLS = SKILL_BOOTSTRAP_TOOLS


def test_review_image_projects_small_model_schema_without_removing_callable_tools(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "projection.sqlite")
    try:
        pid = runtime.process.spawn(image="review-agent:v0", goal="tool projection")
        process = runtime.process.get(pid)
        initial_schema = runtime.tools.openai_tool_schemas(pid)

        assert len(process.tool_table) > len(process.model_tool_table)
        assert set(process.model_tool_table) == INITIAL_TOOLS
        assert process.loaded_skills == {}
        assert "list_capabilities" not in process.model_tool_table
        assert "read_text_file" in process.tool_table
        assert "read_text_file" not in process.model_tool_table
        assert "write_text_file" not in process.model_tool_table
        assert "delete_file" not in process.model_tool_table
        assert len(dumps(initial_schema).encode("utf-8")) < 16_000

        full_tool_table = dict(process.tool_table)
        schema_bytes_before = len(dumps(initial_schema).encode("utf-8"))
        capabilities_before = {item.cap_id for item in runtime.store.list_capabilities(subject=pid)}
        activated = runtime.skills.activate_skill(
            pid,
            "agent-libos-workspace-editing",
            actor=pid,
        )
        after = runtime.process.get(pid)
        schema_bytes_after = len(
            dumps(runtime.tools.openai_tool_schemas(pid)).encode("utf-8")
        )
        capabilities_after = {item.cap_id for item in runtime.store.list_capabilities(subject=pid)}

        assert activated["authority_changed"] is False
        assert schema_bytes_after > schema_bytes_before
        assert capabilities_after == capabilities_before
        assert after.tool_table == full_tool_table
        assert "read_text_file" not in after.model_tool_table
        assert "write_text_file" in after.model_tool_table
        assert "delete_file" in after.model_tool_table
    finally:
        runtime.close()


def test_model_tool_projection_survives_runtime_reopen(tmp_path: Path) -> None:
    database = tmp_path / "projection-reopen.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="review-agent:v0", goal="persist projection")
        runtime.skills.activate_skill(pid, "agent-libos-mcp", actor=pid)
        expected = dict(runtime.process.get(pid).model_tool_table)
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        process = reopened.process.get(pid)
        assert process.model_tool_table == expected
        assert reopened.tools.model_tool_table(pid) == expected
        assert process.loaded_skills["agent-libos-mcp"]["activation_kind"] == "builtin_projection"
    finally:
        reopened.close()


def test_all_skill_projected_builtin_images_start_small_without_changing_authority(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projections.sqlite")
    try:
        expectations = {
            "base-agent:v0": (
                INITIAL_TOOLS,
                "agent-libos-child-processes",
                12_000,
            ),
            "coding-agent:v0": (
                INITIAL_TOOLS,
                "agent-libos-command-execution",
                12_000,
            ),
            "review-agent:v0": (
                INITIAL_TOOLS,
                "agent-libos-git-inspection",
                12_000,
            ),
        }
        for image_id, (expected_tools, activation_skill, schema_limit) in expectations.items():
            pid = runtime.process.spawn(image=image_id, goal="inspect projection")
            process = runtime.process.get(pid)
            schemas = runtime.tools.openai_tool_schemas(pid)

            assert set(process.model_tool_table) == expected_tools
            assert process.loaded_skills == {}
            assert len(process.tool_table) > len(process.model_tool_table)
            assert len(dumps(schemas).encode("utf-8")) < schema_limit

            full_tool_table = dict(process.tool_table)
            visible_count_before = len(process.model_tool_table)
            capabilities_before = {
                capability.cap_id
                for capability in runtime.store.list_capabilities(subject=pid)
            }
            activated = runtime.skills.activate_skill(
                pid,
                activation_skill,
                actor=pid,
            )
            after = runtime.process.get(pid)
            capabilities_after = {
                capability.cap_id
                for capability in runtime.store.list_capabilities(subject=pid)
            }

            assert activated["authority_changed"] is False
            assert len(after.model_tool_table) > visible_count_before
            assert after.tool_table == full_tool_table
            assert capabilities_after == capabilities_before
    finally:
        runtime.close()


@pytest.mark.parametrize("projection", [False, "groups", ["skills"], {"mode": "skills"}])
def test_projection_metadata_accepts_only_skills(
    tmp_path: Path,
    projection: object,
) -> None:
    runtime = Runtime.open(tmp_path / "strict-projection-metadata.sqlite")
    try:
        image = replace(
            DEFAULT_IMAGES["coding-agent:v0"],
            metadata={"tool_projection": projection},
        )
        with pytest.raises(ValidationError, match="tool_projection must be 'skills'"):
            runtime.tools.initial_tool_projection(image)
    finally:
        runtime.close()


@pytest.mark.parametrize("legacy_key", ["lazy_tool_groups", "initial_tool_groups"])
def test_removed_tool_group_metadata_never_falls_back_to_full_projection(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"removed-{legacy_key}.sqlite")
    try:
        source = DEFAULT_IMAGES["coding-agent:v0"]
        image = replace(
            source,
            metadata={legacy_key: True if legacy_key == "lazy_tool_groups" else ["filesystem"]},
        )

        with pytest.raises(
            ValidationError,
            match=rf"removed tool-group fields: {legacy_key}",
        ):
            runtime.tools.initial_tool_projection(image)
    finally:
        runtime.close()


@pytest.mark.parametrize("missing_tool", sorted(SKILL_BOOTSTRAP_TOOLS))
def test_skills_projection_requires_complete_bootstrap(
    tmp_path: Path,
    missing_tool: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"missing-{missing_tool}.sqlite")
    try:
        source = DEFAULT_IMAGES["coding-agent:v0"]
        image = replace(
            source,
            default_tools=[
                tool_name
                for tool_name in source.default_tools
                if tool_name != missing_tool
            ],
        )

        with pytest.raises(
            ValidationError,
            match=rf"requires all bootstrap tools; missing: {missing_tool}",
        ):
            runtime.tools.initial_tool_projection(image)
    finally:
        runtime.close()


def test_tool_group_api_is_removed_and_old_tool_names_are_unknown(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "removed-tool-groups.sqlite")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject old tool API")

        assert not hasattr(runtime.tools, "activate_tool_group")
        assert not hasattr(runtime.tools, "tool_group_for")
        assert not hasattr(runtime.tools, "tool_groups")
        assert {
            row["name"] for row in runtime.tools.visible_tools(pid)
        }.isdisjoint({"activate_tool_group", "discover_tool_groups"})

        result = runtime.tools.call(
            pid,
            "activate_tool_group",
            {"group": "filesystem"},
        )

        assert not result.ok
        assert "not in process tool table" in (result.error or "")
    finally:
        runtime.close()
