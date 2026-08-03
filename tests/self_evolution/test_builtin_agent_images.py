from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.images import DEFAULT_IMAGES, build_default_images
from agent_libos.llm.prompt import ACTION_PROTOCOL, BASE_SYSTEM_PROMPT, build_system_prompt
from agent_libos.skills.builtin_catalog import BUILTIN_SKILL_IDS


BUILTIN_IMAGE_IDS = {
    "analysis-agent:v0",
    "base-agent:v0",
    "coding-agent:v0",
    "maintenance-agent:v0",
    "operator-agent:v0",
    "research-agent:v0",
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
    "image_id",
    [
        "base-agent:v0",
        "coding-agent:v0",
        "review-agent:v0",
        "toolmaker-agent:v0",
    ],
)
def test_skill_projected_builtin_images_defer_every_skill_package(
    image_id: str,
) -> None:
    image = DEFAULT_IMAGES[image_id]
    image_tools = set(image.default_tools)
    system_prompt = build_system_prompt(image)

    assert image.default_skills == []
    assert SKILL_PROJECTION_BOOTSTRAP <= image_tools
    assert all(skill_id not in system_prompt for skill_id in BUILTIN_SKILL_IDS)


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
    assert images["maintenance-agent:v0"].required_capabilities == [human, workspace_read]
    assert images["research-agent:v0"].required_capabilities == [human, workspace_read]
    assert images["analysis-agent:v0"].required_capabilities == [human, workspace_read]
    assert images["operator-agent:v0"].required_capabilities == [human]
    assert images["review-agent:v0"].required_capabilities == [human, workspace_read]
    assert images["toolmaker-agent:v0"].required_capabilities == [human]
    assert images["context-compressor:v0"].required_capabilities == []


@pytest.mark.parametrize(
    "colliding_id",
    [
        "analysis-agent:v0",
        "maintenance-agent:v0",
        "operator-agent:v0",
        "research-agent:v0",
        "review-agent:v0",
        "toolmaker-agent:v0",
        "context-compressor:v0",
    ],
)
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
    assert review.default_skills == []
    assert "parse_pytest_log" in review.default_tools


def test_long_horizon_images_use_narrow_direct_projections() -> None:
    expected_contracts = {
        "maintenance-agent:v0": "repository_maintenance_v1",
        "research-agent:v0": "evidence_synthesis_v1",
        "analysis-agent:v0": "data_analysis_v1",
        "operator-agent:v0": "external_operation_v1",
    }

    for image_id, contract in expected_contracts.items():
        image = DEFAULT_IMAGES[image_id]
        assert image.metadata["projection_posture"] == "narrow_direct"
        assert image.metadata["workflow_contract"] == contract
        assert image.metadata["completion_gate"] == "cumulative_review"
        assert "tool_projection" not in image.metadata
        assert image.default_skills == []
        assert {"human_output", "process_exit"}.issubset(image.default_tools)

    maintenance = DEFAULT_IMAGES["maintenance-agent:v0"]
    assert {
        "read_text_file",
        "write_text_file",
        "run_shell_command",
        "git_status",
        "git_diff",
        "create_checkpoint",
        "read_process_messages",
    }.issubset(maintenance.default_tools)
    assert {
        "discover_skills",
        "activate_skill",
        "delete_file",
        "git_commit",
        "git_push",
    }.isdisjoint(maintenance.default_tools)

    operator = DEFAULT_IMAGES["operator-agent:v0"]
    assert {
        "list_jsonrpc_endpoints",
        "inspect_jsonrpc_endpoint",
        "call_jsonrpc_method",
        "create_checkpoint",
    }.issubset(operator.default_tools)
    assert {"run_shell_command", "write_text_file", "delete_file"}.isdisjoint(
        operator.default_tools
    )


def test_builtin_prompts_use_real_tool_names_and_current_jit_contract() -> None:
    prompts = "\n".join(build_system_prompt(image) for image in DEFAULT_IMAGES.values())

    assert "human_query" not in ACTION_PROTOCOL
    assert "call exit with" not in BASE_SYSTEM_PROMPT
    assert "version-pinned allowlisted JSR" not in prompts
    assert "version-pinned imports" not in prompts
    assert DEFAULT_IMAGES["toolmaker-agent:v0"].default_skills == []
    assert "follow the AgentImage's final reporting contract" in ACTION_PROTOCOL
    assert "Do not prepend that working-directory path" in ACTION_PROTOCOL
    assert "do not call the effect merely to elicit a denial" in ACTION_PROTOCOL
    assert '{"limit":5}' in ACTION_PROTOCOL
    assert '{"limit":"5"}' in ACTION_PROTOCOL
    assert "correct the reported field/type" in ACTION_PROTOCOL


@pytest.mark.parametrize(
    "image_id",
    [
        "analysis-agent:v0",
        "base-agent:v0",
        "coding-agent:v0",
        "maintenance-agent:v0",
        "operator-agent:v0",
        "research-agent:v0",
        "review-agent:v0",
        "toolmaker-agent:v0",
    ],
)
def test_interactive_builtin_images_report_a_final_user_facing_result(image_id: str) -> None:
    prompt = DEFAULT_IMAGES[image_id].system_prompt

    assert "human_output" in prompt
    assert "user-facing" in prompt
    assert "final" in prompt
    assert "machine-only" in prompt
