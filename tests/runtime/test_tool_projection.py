from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.images import DEFAULT_IMAGES
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps


LAZY_CORE_TOOLS = {
    "activate_tool_group",
    "append_memory_object",
    "ask_human",
    "create_memory_object",
    "discover_tool_groups",
    "get_current_time",
    "human_output",
    "list_capabilities",
    "process_exit",
    "read_memory_object",
    "request_permission",
}

FILESYSTEM_READ_TOOLS = {
    "create_object_from_file",
    "read_directory",
    "read_text_file",
}


def test_review_image_projects_small_model_schema_without_removing_callable_tools(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "projection.sqlite")
    try:
        pid = runtime.process.spawn(image="review-agent:v0", goal="tool projection")
        process = runtime.process.get(pid)
        initial_schema = runtime.tools.openai_tool_schemas(pid)

        assert len(process.tool_table) > len(process.model_tool_table)
        assert set(process.model_tool_table) == LAZY_CORE_TOOLS | FILESYSTEM_READ_TOOLS
        assert "list_capabilities" in process.model_tool_table
        assert "read_text_file" in process.tool_table
        assert "read_text_file" in process.model_tool_table
        assert "write_text_file" not in process.model_tool_table
        assert "delete_file" not in process.model_tool_table
        assert len(dumps(initial_schema).encode("utf-8")) < 16_000

        capabilities_before = {item.cap_id for item in runtime.store.list_capabilities(subject=pid)}
        activated = runtime.tools.activate_tool_group(pid, "filesystem")
        capabilities_after = {item.cap_id for item in runtime.store.list_capabilities(subject=pid)}

        assert activated["authority_changed"] is False
        assert activated["schema_bytes_after"] > activated["schema_bytes_before"]
        assert capabilities_after == capabilities_before
        assert "read_text_file" in runtime.process.get(pid).model_tool_table
        assert "write_text_file" in runtime.process.get(pid).model_tool_table
    finally:
        runtime.close()


def test_model_tool_projection_survives_runtime_reopen(tmp_path: Path) -> None:
    database = tmp_path / "projection-reopen.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="review-agent:v0", goal="persist projection")
        runtime.tools.activate_tool_group(pid, "remote")
        expected = dict(runtime.process.get(pid).model_tool_table)
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        assert reopened.process.get(pid).model_tool_table == expected
        assert reopened.tools.model_tool_table(pid) == expected
    finally:
        reopened.close()


def test_all_lazy_builtin_images_start_small_without_changing_authority(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projections.sqlite")
    try:
        expectations = {
            "base-agent:v0": ({"process", "context", "clock"}, "remote", 30_000),
            "coding-agent:v0": ({"filesystem"}, "shell", 20_000),
            "review-agent:v0": ({"filesystem_read"}, "git", 16_000),
        }
        for image_id, (initial_groups, activation_group, schema_limit) in expectations.items():
            pid = runtime.process.spawn(image=image_id, goal="inspect projection")
            process = runtime.process.get(pid)
            schemas = runtime.tools.openai_tool_schemas(pid)
            expected_tools = set(LAZY_CORE_TOOLS)
            for group in initial_groups:
                if group == "filesystem_read":
                    expected_tools.update(FILESYSTEM_READ_TOOLS)
                else:
                    expected_tools.update(
                        entry["name"]
                        for entry in runtime.tools.visible_tools(pid)
                        if runtime.tools.tool_group_for(str(entry["name"])) == group
                    )

            assert set(process.model_tool_table) == expected_tools
            assert len(process.tool_table) > len(process.model_tool_table)
            assert len(dumps(schemas).encode("utf-8")) < schema_limit

            capabilities_before = {
                capability.cap_id
                for capability in runtime.store.list_capabilities(subject=pid)
            }
            activated = runtime.tools.activate_tool_group(pid, activation_group)
            capabilities_after = {
                capability.cap_id
                for capability in runtime.store.list_capabilities(subject=pid)
            }

            assert activated["authority_changed"] is False
            assert activated["tool_count_after"] > activated["tool_count_before"]
            assert capabilities_after == capabilities_before
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "initial_groups, message",
    [
        ("filesystem", "list of non-empty strings"),
        (["filesystem", "filesystem"], "must not contain duplicates"),
        (["does-not-exist"], "unknown initial tool group"),
        (["git"], "is not authorized by image"),
    ],
)
def test_invalid_initial_tool_groups_fail_closed(
    tmp_path: Path,
    initial_groups: object,
    message: str,
) -> None:
    runtime = Runtime.open(tmp_path / "invalid-initial-groups.sqlite")
    try:
        image = replace(
            DEFAULT_IMAGES["base-agent:v0"],
            metadata={
                "lazy_tool_groups": True,
                "initial_tool_groups": initial_groups,
            },
        )
        with pytest.raises(ValidationError, match=message):
            runtime.tools.initial_tool_projection(image)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"lazy_tool_groups": "false"}, "lazy_tool_groups must be a boolean"),
        ({"initial_tool_groups": ["filesystem"]}, "requires lazy_tool_groups=true"),
    ],
)
def test_projection_metadata_uses_strict_types(
    tmp_path: Path,
    metadata: dict[str, object],
    message: str,
) -> None:
    runtime = Runtime.open(tmp_path / "strict-projection-metadata.sqlite")
    try:
        image = replace(DEFAULT_IMAGES["coding-agent:v0"], metadata=metadata)
        with pytest.raises(ValidationError, match=message):
            runtime.tools.initial_tool_projection(image)
    finally:
        runtime.close()
