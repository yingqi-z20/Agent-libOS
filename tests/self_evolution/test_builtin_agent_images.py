from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.images import DEFAULT_IMAGES, build_default_images
from agent_libos.llm.prompt import ACTION_PROTOCOL, BASE_SYSTEM_PROMPT, build_system_prompt
from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog


BUILTIN_IMAGE_IDS = {
    "base-agent:v0",
    "coding-agent:v0",
    "review-agent:v0",
    "toolmaker-agent:v0",
    "context-compressor:v0",
}
SUPPORTED_BUILTIN_CONTEXT_POLICIES = {
    "plan_first",
    "recency_first",
    "evidence_first",
    "error_debug",
}
SKILL_PROJECTION_BOOTSTRAP = {
    "activate_skill",
    "discover_skills",
    "process_exit",
    "read_skill_resource",
    "unload_skill",
}


def test_builtin_image_contracts_are_unique_bounded_and_internally_coherent() -> None:
    assert set(DEFAULT_IMAGES) == BUILTIN_IMAGE_IDS

    for image_id, image in DEFAULT_IMAGES.items():
        assert image.image_id == image_id
        assert image.system_prompt.strip()
        assert len(image.system_prompt) <= DEFAULT_CONFIG.image.prompt_max_chars
        assert len(image.default_tools) == len(set(image.default_tools))
        assert len(image.default_tools) <= DEFAULT_CONFIG.image.max_default_tools
        assert image.context_policy in SUPPORTED_BUILTIN_CONTEXT_POLICIES
        if image.metadata.get("tool_projection") == "skills":
            assert {
                "discover_skills",
                "activate_skill",
                "read_skill_resource",
                "unload_skill",
                "process_exit",
            }.issubset(image.default_tools)


@pytest.mark.parametrize(
    ("image_id", "expected_initial_schema_count"),
    [
        ("base-agent:v0", 15),
        ("coding-agent:v0", 14),
        ("review-agent:v0", 14),
    ],
)
def test_skill_projected_builtin_images_contain_complete_initial_skill_packages(
    image_id: str,
    expected_initial_schema_count: int,
) -> None:
    image = DEFAULT_IMAGES[image_id]
    catalog = get_builtin_skill_catalog()
    image_tools = set(image.default_tools)

    initially_projected = set(SKILL_PROJECTION_BOOTSTRAP)
    for skill_id in image.default_skills:
        package = catalog.get(skill_id)
        assert package is not None
        initially_projected.update(package.allowed_tools)
    assert initially_projected <= image_tools
    assert len(initially_projected) == expected_initial_schema_count


def test_builtin_image_capability_requirements_follow_runtime_identity_config() -> None:
    config = replace(
        DEFAULT_CONFIG,
        runtime=replace(
            DEFAULT_CONFIG.runtime,
            workspace_namespace="repo",
            default_human="operator",
        ),
    )
    images = build_default_images(config)
    human = {"resource": "human:operator", "rights": ["write"]}
    workspace_read = {"resource": "filesystem:repo:*", "rights": ["read"]}

    assert images[config.runtime.default_image_id].required_capabilities == [human]
    assert images[config.runtime.coding_image_id].required_capabilities == [human, workspace_read]
    assert images["review-agent:v0"].required_capabilities == [human, workspace_read]
    assert images["toolmaker-agent:v0"].required_capabilities == [human]
    assert images["context-compressor:v0"].required_capabilities == []


@pytest.mark.parametrize("colliding_id", ["review-agent:v0", "toolmaker-agent:v0", "context-compressor:v0"])
def test_configured_builtin_image_ids_fail_closed_on_collision(colliding_id: str) -> None:
    config = replace(
        DEFAULT_CONFIG,
        runtime=replace(DEFAULT_CONFIG.runtime, default_image_id=colliding_id),
    )

    with pytest.raises(ValueError, match="built-in AgentImage ids must be unique"):
        build_default_images(config)


def test_base_and_coding_ids_must_also_be_distinct() -> None:
    config = replace(
        DEFAULT_CONFIG,
        runtime=replace(
            DEFAULT_CONFIG.runtime,
            coding_image_id=DEFAULT_CONFIG.runtime.default_image_id,
        ),
    )

    with pytest.raises(ValueError, match="built-in AgentImage ids must be unique"):
        build_default_images(config)


def test_specialized_images_keep_narrow_explicit_tool_tables() -> None:
    toolmaker = DEFAULT_IMAGES["toolmaker-agent:v0"]
    compressor = DEFAULT_IMAGES["context-compressor:v0"]

    assert toolmaker.metadata["source_contract"] == "import_free"
    assert {"ask_human", "request_permission", "propose_jit_tool", "validate_jit_tool", "register_jit_tool"}.issubset(toolmaker.default_tools)
    assert compressor.default_tools == ["process_exit"]
    assert compressor.metadata["output_contract"] == [
        "goal",
        "constraints",
        "user_preferences",
        "completed",
        "pending",
        "key_references",
        "recent_decisions",
        "risks",
        "uncertainties",
        "next_steps",
    ]


def test_review_image_starts_read_only_and_defers_mutation_and_test_tools() -> None:
    review = DEFAULT_IMAGES["review-agent:v0"]

    assert review.metadata["tool_projection"] == "skills"
    assert "agent-libos-workspace-navigation" in review.default_skills
    assert "agent-libos-workspace-editing" not in review.default_skills
    assert "agent-libos-test-log-analysis" not in review.default_skills
    assert "parse_pytest_log" in review.default_tools


def test_builtin_prompts_use_real_tool_names_and_current_jit_contract() -> None:
    prompts = "\n".join(build_system_prompt(image) for image in DEFAULT_IMAGES.values())

    assert "human_query" not in ACTION_PROTOCOL
    assert "call exit with" not in BASE_SYSTEM_PROMPT
    assert "version-pinned allowlisted JSR" not in prompts
    assert "version-pinned imports" not in prompts
    assert DEFAULT_IMAGES["toolmaker-agent:v0"].default_skills == [
        "agent-libos-jit-tool-authoring"
    ]
    assert "follow the AgentImage's final reporting contract" in ACTION_PROTOCOL
    assert "Do not prepend that working-directory path" in ACTION_PROTOCOL
    assert "do not call the effect merely to elicit a denial" in ACTION_PROTOCOL


@pytest.mark.parametrize(
    "image_id",
    ["base-agent:v0", "coding-agent:v0", "review-agent:v0", "toolmaker-agent:v0"],
)
def test_interactive_builtin_images_report_a_final_user_facing_result(image_id: str) -> None:
    prompt = DEFAULT_IMAGES[image_id].system_prompt

    assert "human_output" in prompt
    assert "user-facing" in prompt
    assert "final" in prompt
    assert "machine-only" in prompt
