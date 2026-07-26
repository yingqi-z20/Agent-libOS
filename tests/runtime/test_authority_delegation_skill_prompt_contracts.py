from __future__ import annotations

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


def test_capability_delegation_preserves_lease_and_settlement_boundaries() -> None:
    instructions = _instructions("agent-libos-capability-delegation")

    assert "Restrictive effects" in instructions
    assert "cannot have `uses_remaining`" in instructions
    assert "strictly in the future at delegation" in instructions
    assert "presentation_omitted:true" in instructions
    assert "positive settlement receipts" in instructions
    assert "must not be retried" in instructions
