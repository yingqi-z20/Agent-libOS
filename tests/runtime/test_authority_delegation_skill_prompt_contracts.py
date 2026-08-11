from __future__ import annotations

from pathlib import Path

from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog


def _instructions(skill_id: str) -> str:
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    return package.instructions


def test_authority_basics_preserves_ask_deny_and_request_ceiling_boundaries() -> None:
    instructions = _instructions("agent-libos-authority-basics")

    assert "explicitly provides a per-use approval bridge" in instructions
    assert "Without such a bridge the operation returns" in instructions
    assert "Never replay" in instructions
    assert "every root-prefix `<kind>:*` request" in instructions
    assert "successful publication consumes a" in instructions


def test_authority_basics_distinguishes_model_projection_from_durable_data() -> None:
    instructions = _instructions("agent-libos-authority-basics")
    normalized = " ".join(instructions.split())

    assert "Durable ToolResult data may additionally contain `subject`" in instructions
    assert "intentionally not model-visible" in instructions
    assert "Missing `constraints`/`rules` in the model projection" in normalized
    assert "absent model policy fields are not proof" in normalized


def test_capability_delegation_preserves_lease_and_settlement_boundaries() -> None:
    instructions = _instructions("agent-libos-capability-delegation")

    assert "Restrictive effects" in instructions
    assert "cannot have `uses_remaining`" in instructions
    assert "strictly in the future at delegation" in instructions
    assert "presentation_omitted:true" in instructions
    assert "positive settlement receipts" in instructions
    assert "must not be retried" in instructions


def test_capability_delegation_does_not_require_hidden_provenance() -> None:
    instructions = _instructions("agent-libos-capability-delegation")

    assert "selected `parent_cap_id`" in instructions
    assert "are not\nmodel-visible" in instructions
    assert "must not trigger another mutation" in instructions
    assert "model projection omits `issued_at`" in instructions
    assert "Validate returned `cap_id`, child `subject`" not in instructions


def test_skill_navigation_uses_the_model_lifecycle_projection() -> None:
    instructions = _instructions("agent-libos-skill-navigation")
    skills_doc = Path("docs/skills.md").read_text(encoding="utf-8")
    normalized_doc = " ".join(skills_doc.split())

    assert "nested result contains only `skill_id`, `name`" in instructions
    assert "Durable ToolResult data retained by the Host" in instructions
    assert "model receipt deliberately does not echo that hash" in instructions
    assert "Host tool-ID maps are not a model completion condition" in instructions
    assert "set(tool_names)" not in instructions
    assert "keys(tool_ids)" not in instructions
    assert "Model-facing activation contains only `skill_id`" in normalized_doc
    assert "model-facing unload contains only `skill_id`" in normalized_doc
    assert "intentionally absent from the model projection" in normalized_doc
