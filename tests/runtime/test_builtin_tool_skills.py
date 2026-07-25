from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from importlib import resources
from pathlib import Path
import re

import pytest

import agent_libos.skills.builtin_catalog as builtin_catalog_module
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.prompt import build_user_prompt
from agent_libos.models import CapabilityRight, MaterializedContext
from agent_libos.models.exceptions import ValidationError
from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_IDS,
    BUILTIN_SKILL_MAX_FILE_BYTES,
    BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
    BUILTIN_SKILL_MAX_TOOLS,
    get_builtin_skill_catalog,
)
from agent_libos.skills.schema import SkillPackage
from agent_libos.tools.builtin.checkpoint import (
    DiffCheckpointOutput,
    RestoreCheckpointOutput,
)
from agent_libos.tools.builtin.jsonrpc import ListJsonRpcEndpointsTool
from agent_libos.tools.builtin.mcp import ListMcpServersTool
from agent_libos.utils.yaml_loader import load_yaml_mapping


WORKSPACE_EDITING_SKILL = "agent-libos-workspace-editing"

_HIGH_RISK_SCHEMA_CONTRACTS: dict[str, dict[str, object]] = {
    "request_permission": {
        "required": {"resource", "rights", "reason"},
        "defaults": {"human": "owner"},
        "forbidden": {"cap_id", "policy"},
    },
    "delegate_capability": {
        "required": {"child_pid", "resource", "rights"},
        "defaults": {
            "delegable": False,
            "effect": "allow",
            "expires_at": None,
            "uses_remaining": None,
        },
        "forbidden": {"parent_cap_id", "max_delegation_depth"},
    },
    "revoke_capability": {
        "required": {"cap_id"},
        "defaults": {"reason": None},
    },
    "ask_human": {
        "required": {"question"},
        "defaults": {"human": "owner"},
    },
    "human_output": {
        "required": {"message"},
        "defaults": {"channel": "terminal"},
        "forbidden": {"human"},
    },
    "compact_process_context": {
        "defaults": {
            "force": False,
            "max_chunks": 8,
            "preserve_recent_entries": 8,
            "target_tokens": 4000,
        },
        "bounds": {
            "max_chunks": {"minimum": 1, "maximum": 64},
            "preserve_recent_entries": {"minimum": 0, "maximum": 128},
            "target_tokens": {"minimum": 256, "maximum": 64000},
        },
    },
    "process_exit": {
        "fields": {
            "completion_evidence",
            "message",
            "payload",
            "result_oid",
            "review_token",
        },
    },
    "get_current_time": {"defaults": {"timezone": "UTC"}},
    "sleep": {
        "required": {"seconds"},
        "bounds": {"seconds": {"minimum": 0, "maximum": 60.0}},
    },
    "read_text_file": {
        "required": {"path"},
        "defaults": {"encoding": "utf-8", "max_bytes": 65536},
    },
    "write_directory": {
        "required": {"path"},
        "defaults": {"exist_ok": True, "parents": True},
    },
    "write_text_file": {
        "required": {"path", "content"},
        "defaults": {"encoding": "utf-8", "overwrite": True},
    },
    "delete_directory": {
        "required": {"path"},
        "defaults": {"missing_ok": False, "recursive": False},
    },
    "run_shell_command": {
        "required": {"argv"},
        "defaults": {
            "max_stderr_chars": 32000,
            "max_stdout_chars": 32000,
            "timeout_s": 30.0,
        },
        "forbidden": {"cwd", "env", "stdin", "shell"},
    },
    "create_memory_object": {
        "required": {"type", "payload"},
        "defaults": {"immutable": True, "name": None, "namespace": None},
    },
    "read_memory_object": {
        "required": {"name"},
        "defaults": {
            "cursor": 0,
            "expected_sha256": None,
            "json_pointer": "",
            "max_payload_chars": 12000,
            "namespace": None,
        },
    },
    "append_memory_object": {
        "required": {"name", "entry"},
        "defaults": {"list_field": "entries", "namespace": None},
    },
    "create_object_from_file": {
        "required": {"name", "path"},
        "defaults": {
            "allow_truncated": False,
            "encoding": "utf-8",
            "max_bytes": 1048576,
            "object_type": "artifact",
        },
        "bounds": {"max_bytes": {"minimum": 1, "maximum": 1048576}},
    },
    "write_object_to_file": {
        "required": {"name", "path"},
        "defaults": {"encoding": "utf-8", "overwrite": True},
        "forbidden": {"oid"},
    },
    "start_object_task": {
        "required": {"tool"},
        "defaults": {
            "grant_result_to_notify": False,
            "owner_watch": False,
        },
        "fields": {"inherit_capabilities", "owner_name", "owner_oid"},
    },
    "wait_object_task": {
        "required": {"task_id"},
        "defaults": {"timeout_s": None},
    },
    "fork_child_process": {
        "required": {"goal"},
        "defaults": {"include_parent_roots": True, "mode": "worker"},
    },
    "wait_child_process": {
        "required": {"child_pid"},
        "defaults": {"block": True},
    },
    "read_process_messages": {
        "defaults": {"ack": True, "include_acked": False, "limit": 100},
    },
    "receive_process_messages": {
        "defaults": {
            "ack": True,
            "block": True,
            "include_acked": False,
            "limit": 100,
        },
    },
    "merge_child_memory": {
        "required": {"child_pid"},
        "defaults": {"include_child_created": True},
    },
    "exec_process": {
        "required": {"image"},
        "defaults": {"preserve_capabilities": False, "preserve_memory": True},
    },
    "list_mcp_tools": {
        "required": {"server_id"},
        "defaults": {"refresh": False},
    },
    "call_mcp_tool": {
        "required": {"server_id", "tool_id"},
        "fields": {"arguments"},
        "forbidden": {"refresh", "url"},
    },
    "git_integrate": {
        "required": {"operation", "expected_state_token"},
        "enums": {
            "operation": {"merge", "rebase", "cherry_pick", "revert", "abort"},
        },
    },
    "git_reset": {
        "required": {"target", "expected_state_token"},
        "defaults": {"mode": "mixed"},
        "enums": {"mode": {"soft", "mixed", "hard"}},
    },
    "git_pull": {
        "required": {"remote", "expected_state_token"},
        "defaults": {"strategy": "ff_only"},
        "enums": {"strategy": {"ff_only", "merge", "rebase"}},
        "forbidden": {"url"},
    },
    "git_push": {
        "required": {"remote", "remote_ref", "expected_state_token"},
        "defaults": {"delete": False, "force_with_lease_oid": None},
        "forbidden": {"force", "url"},
    },
    "git_merge_pull_request": {
        "required": {"pr_id", "expected_state_token"},
        "defaults": {"strategy": "fast_forward"},
        "enums": {"strategy": {"fast_forward", "merge", "squash"}},
    },
}


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


def test_builtin_skill_instructions_route_every_owned_tool() -> None:
    for package in get_builtin_skill_catalog().list():
        missing = [
            tool_name
            for tool_name in package.allowed_tools
            if f"`{tool_name}`" not in package.instructions
        ]

        assert missing == [], f"{package.skill_id} does not guide {missing}"


def test_builtin_skill_descriptions_and_bodies_have_progressive_guidance_structure() -> None:
    trigger_context = re.compile(
        r"\b(?:after|before|by|during|for|if|not|on|only|rather|use|using|when|while|with|without)\b",
        re.IGNORECASE,
    )
    safety_decision = re.compile(
        r"\b(?:cannot|deny|denied|do not|must|never|only|permission|requires?|stop)\b",
        re.IGNORECASE,
    )
    verification_decision = re.compile(
        r"\b(?:confirm|inspect|read back|result|stop|success|verify|verification)\b",
        re.IGNORECASE,
    )

    for package in get_builtin_skill_catalog().list():
        description_words = re.findall(r"[A-Za-z0-9_-]+", package.description)
        assert len(description_words) >= 12, package.skill_id
        assert trigger_context.search(package.description), package.skill_id

        sections = list(
            re.finditer(
                r"(?ms)^##\s+(.+?)\s*$\n(.*?)(?=^##\s+|\Z)",
                package.instructions,
            )
        )
        assert len(sections) >= 3, package.skill_id
        assert safety_decision.search(package.instructions), package.skill_id
        assert verification_decision.search(package.instructions), package.skill_id


def test_builtin_skill_bodies_have_operational_tool_guides() -> None:
    workflow_heading = re.compile(r"\b(?:guide|sequence|workflow)\b", re.IGNORECASE)
    settlement_heading = re.compile(
        r"\b(?:approval|authority|boundary|completion|evidence|failure|"
        r"recover\w*|stop\w*|uncertain|verif\w*)\b",
        re.IGNORECASE,
    )

    for package in get_builtin_skill_catalog().list():
        sections = {
            match.group(1).strip(): match.group(2)
            for match in re.finditer(
                r"(?ms)^##\s+(.+?)\s*$\n(.*?)(?=^##\s+|\Z)",
                package.instructions,
            )
        }
        assert any(workflow_heading.search(title) for title in sections), package.skill_id
        assert any(settlement_heading.search(title) for title in sections), package.skill_id

        missing_tool_guidance = [
            tool_name
            for tool_name in package.allowed_tools
            if f"`{tool_name}`" not in package.instructions
        ]
        assert missing_tool_guidance == [], (
            package.skill_id,
            missing_tool_guidance,
        )

        # Combined workflows can guide several closely related tools more
        # efficiently than repeating one fixed quota per schema. Keep a useful
        # absolute floor while the semantic and per-tool checks above guard the
        # actual content.
        assert len(package.instructions.encode("utf-8")) >= 3_000, package.skill_id


def test_effectful_builtin_skills_include_recovery_or_stop_decisions(tmp_path: Path) -> None:
    recovery_decision = re.compile(
        r"\b(?:cancel\w*|fail\w*|reconcile\w*|recover\w*|reject\w*|replay|"
        r"resume|retry|rollback|stop|suspend\w*|uncertain|unknown)\b",
        re.IGNORECASE,
    )
    runtime = Runtime.open(tmp_path / "builtin-skill-recovery-contract.sqlite")
    try:
        specs = {
            str(row["name"]): json.loads(row["spec_json"])
            for row in runtime.tools.list()
        }
        for package in get_builtin_skill_catalog().list():
            is_effectful = any(
                specs[tool]["policy"]["side_effects"]
                and not specs[tool]["policy"]["idempotent"]
                for tool in package.allowed_tools
            )
            if is_effectful:
                assert recovery_decision.search(package.instructions), package.skill_id
    finally:
        runtime.close()


def test_high_risk_builtin_tool_schemas_match_guidance_contracts(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "builtin-skill-schema-contract.sqlite")
    try:
        specs = {
            str(row["name"]): json.loads(row["spec_json"])
            for row in runtime.tools.list()
        }
        catalog = get_builtin_skill_catalog()
        assert set(_HIGH_RISK_SCHEMA_CONTRACTS) <= set(specs)
        for tool_name, contract in _HIGH_RISK_SCHEMA_CONTRACTS.items():
            schema = specs[tool_name]["input_schema"]
            properties = schema["properties"]
            assert catalog.skill_for_tool(tool_name) is not None
            assert set(contract.get("required", set())) <= set(schema.get("required", []))
            assert set(contract.get("fields", set())) <= set(properties)
            assert set(contract.get("forbidden", set())).isdisjoint(properties)
            for field, expected in dict(contract.get("defaults", {})).items():
                assert properties[field].get("default") == expected, (tool_name, field)
                assert "default" in properties[field], (tool_name, field)
            for field, expected in dict(contract.get("enums", {})).items():
                assert set(properties[field]["enum"]) == set(expected), (tool_name, field)
            for field, expected in dict(contract.get("bounds", {})).items():
                assert {
                    key: properties[field][key]
                    for key in expected
                } == expected, (tool_name, field)
    finally:
        runtime.close()


def test_builtin_tool_guidance_validation_fails_closed() -> None:
    with pytest.raises(ValidationError, match="do not guide allowed tools: echo"):
        builtin_catalog_module._validate_builtin_tool_guidance(
            "Use the diagnostic tool.",
            allowed_tools=["echo"],
            name="agent-libos-example",
        )


def test_fork_tool_schema_does_not_claim_copy_on_write_isolation(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "fork-tool-mode-description.sqlite")
    try:
        spec = next(row for row in runtime.tools.list() if row["name"] == "fork_child_process")
        schema = json.loads(spec["spec_json"])["input_schema"]
        description = schema["properties"]["mode"]["description"]

        assert "not copy-on-write isolation" in description
        assert "same Object ids" in description
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("object_default", "object_hard", "filesystem_hard", "expected_default", "expected_maximum"),
    [
        (300_000, 500_000, 200_000, 200_000, 200_000),
        (150_000, 180_000, 220_000, 150_000, 180_000),
        (15_000_000, 20_000_000, 20_000_000, 15_000_000, 20_000_000),
    ],
)
def test_create_object_from_file_schema_uses_effective_file_read_limits(
    tmp_path: Path,
    object_default: int,
    object_hard: int,
    filesystem_hard: int,
    expected_default: int,
    expected_maximum: int,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(
            DEFAULT_CONFIG.tools,
            object_file_max_bytes=object_default,
            object_file_hard_limit_bytes=object_hard,
            filesystem_read_hard_limit_bytes=filesystem_hard,
        ),
    )
    runtime = Runtime.open(
        tmp_path / f"object-file-schema-{object_default}-{filesystem_hard}.sqlite",
        config=config,
    )
    try:
        spec = next(
            row for row in runtime.tools.list()
            if row["name"] == "create_object_from_file"
        )
        max_bytes = json.loads(spec["spec_json"])["input_schema"]["properties"][
            "max_bytes"
        ]

        assert max_bytes["default"] == expected_default
        assert max_bytes["maximum"] == expected_maximum
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "skill_id",
    [
        "agent-libos-workspace-navigation",
        "agent-libos-workspace-editing",
        "agent-libos-command-execution",
        "agent-libos-test-log-analysis",
        "agent-libos-tool-protocol-diagnostics",
    ],
)
def test_workspace_tool_skill_guidance_fits_progressive_disclosure_budget(
    skill_id: str,
) -> None:
    instructions = get_builtin_skill_catalog().get(skill_id).instructions

    assert len(instructions.encode("utf-8")) <= BUILTIN_SKILL_MAX_INSTRUCTION_BYTES


def test_builtin_skill_expanded_body_budget_is_fully_model_visible() -> None:
    assert BUILTIN_SKILL_MAX_INSTRUCTION_BYTES == 16 * 1_024
    assert BUILTIN_SKILL_MAX_FILE_BYTES == 24 * 1_024
    assert BUILTIN_SKILL_MAX_FILE_BYTES > BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
    assert (
        DEFAULT_CONFIG.skills.max_prompt_instruction_chars
        >= BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
    )


def test_long_builtin_skill_body_is_injected_without_tail_truncation(
    tmp_path: Path,
) -> None:
    package = max(
        get_builtin_skill_catalog().list(),
        key=lambda item: len(item.instructions.encode("utf-8")),
    )
    assert len(package.instructions.encode("utf-8")) > 8_000

    runtime = Runtime.open(tmp_path / "long-builtin-prompt-body.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="verify complete long Skill prompt projection",
        )
        runtime.skills.activate_skill(pid, package.skill_id, actor=pid)

        prompt = _render_prompt(runtime, pid)
        assert package.instructions.strip() in prompt
    finally:
        runtime.close()


def test_runtime_rejects_builtin_prompt_body_truncation_config(tmp_path: Path) -> None:
    config = replace(
        DEFAULT_CONFIG,
        skills=replace(
            DEFAULT_CONFIG.skills,
            max_prompt_instruction_chars=64,
        ),
    )
    with pytest.raises(
        ValidationError,
        match="instructions exceeds max_prompt_instruction_chars=64",
    ):
        Runtime.open(tmp_path / "builtin-prompt-limit.sqlite", config=config)


def test_checkpoint_output_models_preserve_external_effect_evidence() -> None:
    empty_page = {
        "count": 0,
        "returned_count": 0,
        "truncated": False,
        "next_cursor": None,
    }
    diff = DiffCheckpointOutput(
        checkpoint_id="ckpt_test",
        pid="pid_test",
        tables={},
        external_effects_since_checkpoint=[],
        external_effects_page=empty_page,
        external_effect_summary={"total": 0},
        restore_external_policy="report_only",
    ).model_dump()
    restored = RestoreCheckpointOutput(
        checkpoint_id="ckpt_test",
        publication_id="pub_test",
        pid="pid_test",
        status="restored",
        main_state_committed=True,
        reconciliation_pending=False,
        post_commit_failures=[],
        restored_pids=["pid_test"],
        previous_pids=["pid_test"],
        cancelled_human_requests=["human_test"],
        superseded_messages=[],
        superseded_object_tasks=[],
        external_effects_since_checkpoint=[],
        restored_pids_page=empty_page,
        previous_pids_page=empty_page,
        cancelled_human_requests_page=empty_page,
        superseded_messages_page=empty_page,
        superseded_object_tasks_page=empty_page,
        external_effects_page=empty_page,
        external_effect_summary={"total": 0},
        restore_external_policy="report_only",
        post_commit_failures_page=empty_page,
    ).model_dump()

    assert diff["external_effect_summary"] == {"total": 0}
    assert diff["restore_external_policy"] == "report_only"
    assert restored["cancelled_human_requests"] == ["human_test"]
    assert "cancelled_human_request_ids" not in restored
    assert restored["external_effect_summary"] == {"total": 0}
    assert restored["restore_external_policy"] == "report_only"


def test_registered_integration_list_outputs_expose_completeness() -> None:
    for tool in (ListJsonRpcEndpointsTool(), ListMcpServersTool()):
        output_schema = tool.spec().output_schema
        assert output_schema is not None
        assert "has_more" in output_schema["properties"]
        assert "has_more" in output_schema["required"]


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
    "image_id",
    [
        "base-agent:v0",
        "coding-agent:v0",
        "review-agent:v0",
        "toolmaker-agent:v0",
    ],
)
def test_builtin_images_start_with_only_the_source_neutral_skill_lifecycle(
    tmp_path: Path,
    image_id: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"{image_id.replace(':', '-')}.sqlite")
    try:
        pid = runtime.process.spawn(image=image_id, goal="inspect built-in Skill projection")
        process = runtime.process.get(pid)

        assert set(process.model_tool_table) == {
            "activate_skill",
            "discover_skills",
            "process_exit",
            "read_skill_resource",
            "unload_skill",
        }
        assert process.loaded_skills == {}
        assert len(runtime.tools.openai_tool_schemas(pid)) == 5
        assert len(process.tool_table) > len(process.model_tool_table)
    finally:
        runtime.close()


def test_builtin_discovery_obeys_the_same_search_and_page_limit(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "complete-builtin-discovery.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="inspect a bounded visible Skill catalog page",
        )
        discovered = runtime.skills.discover_skills_result(
            text="workspace",
            actor=pid,
            limit=1,
        )

        assert len(discovered["skills"]) == 1
        assert discovered["has_more"] is True
        assert discovered["catalog_scope"] == "visibility_limited"
        summary = discovered["skills"][0]
        searchable = " ".join(
            str(summary[field])
            for field in ("skill_id", "name", "description")
        ).casefold()
        assert "workspace" in searchable

        intent_result = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": "ordinary workspace file write", "limit": 5},
        )
        assert intent_result.ok
        assert intent_result.payload["next_step"] == "activate_skill"
        assert intent_result.payload["skills"][0]["skill_id"] == WORKSPACE_EDITING_SKILL

        model_result = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": WORKSPACE_EDITING_SKILL, "limit": 1},
        )
        assert model_result.ok
        assert model_result.payload["has_more"] is False
        assert model_result.payload["visibility_limited"] is True
        assert model_result.payload["next_step"] == "activate_skill"
        model_summary = model_result.payload["skills"][0]
        assert model_summary["skill_id"] == WORKSPACE_EDITING_SKILL
        assert {
            "source",
            "source_type",
            "catalog_scope",
            "available_tools",
        }.isdisjoint(model_summary)

        no_match = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": "nonexistent-zebra-domain", "limit": 5},
        )
        assert no_match.ok
        assert no_match.payload["skills"] == []
        assert no_match.payload["next_step"] == "refine_search"

        oversized = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": "x" * 1_025, "limit": 5},
        )
        assert not oversized.ok
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("query", "expected_skill_id"),
    [
        ("ordinary workspace file write", "agent-libos-workspace-editing"),
        ("git status read-only inspection", "agent-libos-git-inspection"),
        ("run shell command argv execute", "agent-libos-command-execution"),
        ("checkpoint recovery point", "agent-libos-checkpoints"),
        ("mcp registry metadata", "agent-libos-mcp"),
    ],
)
def test_source_neutral_discovery_ranks_concrete_task_terms(
    tmp_path: Path,
    query: str,
    expected_skill_id: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"intent-{expected_skill_id}.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="select one Skill from task intent",
        )

        result = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": query, "limit": 5},
        )

        assert result.ok
        assert result.payload["next_step"] == "activate_skill"
        assert result.payload["skills"][0]["skill_id"] == expected_skill_id
    finally:
        runtime.close()


def test_model_skill_lifecycle_contract_is_source_neutral(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "source-neutral-skill-lifecycle.sqlite")
    registered_skill_id = "source-neutral-clock"
    try:
        runtime.skills.register_skill_package(
            SkillPackage(
                skill_id=registered_skill_id,
                name=registered_skill_id,
                description=(
                    "Read the current time for source-neutral Skill lifecycle testing."
                ),
                instructions=(
                    "Use `get_current_time` once and verify the returned timezone."
                ),
                allowed_tools=["get_current_time"],
            ),
            actor="test.host",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="compare model-visible Skill lifecycle contracts",
        )
        runtime.capability.grant(
            pid,
            runtime.config.skills.registry_resource,
            [CapabilityRight.READ],
            issued_by="source-neutral-skill-test",
        )
        runtime.capability.grant(
            pid,
            runtime.skills.resource_for(registered_skill_id),
            [CapabilityRight.EXECUTE],
            issued_by="source-neutral-skill-test",
        )

        builtin_discovery = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": WORKSPACE_EDITING_SKILL, "limit": 1},
        )
        registered_discovery = runtime.tools.call(
            pid,
            "discover_skills",
            {"text": registered_skill_id, "limit": 1},
        )
        assert builtin_discovery.ok and registered_discovery.ok
        builtin_summary = builtin_discovery.payload["skills"][0]
        registered_summary = registered_discovery.payload["skills"][0]
        assert set(builtin_summary) == set(registered_summary)

        builtin_activation = runtime.tools.call(
            pid,
            "activate_skill",
            {"skill_id": WORKSPACE_EDITING_SKILL},
        )
        registered_activation = runtime.tools.call(
            pid,
            "activate_skill",
            {"skill_id": registered_skill_id},
        )
        assert builtin_activation.ok and registered_activation.ok
        builtin_result = builtin_activation.payload["result"]
        registered_result = registered_activation.payload["result"]
        assert set(builtin_result) == set(registered_result)
        model_loaded = runtime.tools.model_loaded_skills(pid)
        assert set(model_loaded[WORKSPACE_EDITING_SKILL]) == set(
            model_loaded[registered_skill_id]
        )

        builtin_unload = runtime.tools.call(
            pid,
            "unload_skill",
            {"skill_id": WORKSPACE_EDITING_SKILL},
        )
        registered_unload = runtime.tools.call(
            pid,
            "unload_skill",
            {"skill_id": registered_skill_id},
        )
        assert builtin_unload.ok and registered_unload.ok
        assert set(builtin_unload.payload["result"]) == set(
            registered_unload.payload["result"]
        )

        source_specific_fields = {
            "activation_kind",
            "authority_changed",
            "catalog_scope",
            "registered",
            "registered_by",
            "source",
            "source_type",
        }
        for payload in (
            builtin_summary,
            registered_summary,
            builtin_result,
            registered_result,
            model_loaded[WORKSPACE_EDITING_SKILL],
            model_loaded[registered_skill_id],
            builtin_unload.payload["result"],
            registered_unload.payload["result"],
        ):
            assert source_specific_fields.isdisjoint(payload)
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
        assert discovered["catalog_scope"] == "visibility_limited"
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
        assert runtime.process.get(parent).loaded_skills == {}
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


def test_prompt_discloses_no_skill_metadata_until_activation(tmp_path: Path) -> None:
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

        prompt_before = _render_prompt(runtime, pid)
        assert WORKSPACE_EDITING_SKILL not in prompt_before
        assert package.description not in prompt_before
        assert body_marker not in prompt_before

        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        prompt_after = _render_prompt(runtime, pid)
        assert WORKSPACE_EDITING_SKILL in prompt_after
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
        assert '"invalid_snapshot":true' in prompt
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
        available_skills=[],
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
