from __future__ import annotations

from importlib import resources

import pytest

from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_MAX_FILE_BYTES,
    BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
    get_builtin_skill_catalog,
)


_LONG_CONTEXT_SKILLS = (
    "agent-libos-runtime-session",
    "agent-libos-child-processes",
    "agent-libos-object-tasks",
)


@pytest.mark.parametrize("skill_id", _LONG_CONTEXT_SKILLS)
def test_long_context_builtin_skill_fits_prompt_budget(skill_id: str) -> None:
    raw = (
        resources.files("agent_libos.skills.builtin")
        .joinpath(skill_id, "SKILL.md")
        .read_bytes()
    )
    package = get_builtin_skill_catalog().get(skill_id)

    assert package is not None
    assert len(raw) <= BUILTIN_SKILL_MAX_FILE_BYTES
    assert (
        len(package.instructions.encode("utf-8"))
        <= BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
    )


@pytest.mark.parametrize(
    ("skill_id", "required_contracts"),
    [
        (
            "agent-libos-runtime-session",
            (
                "Never poll an event with repeated sleeps.",
                "A source-only prompt has no context Object",
                "`target_tokens=4000`",
                "`preserve_recent_entries=8`",
                "`max_chunks=8`",
                "`force=false`",
                "input precedence is exact",
                "`result_oid` reuses",
                "`payload` creates",
                "`message` creates",
                "Call a bare `process_exit`",
                "first token stale",
                "post-ACK review",
                "fails the process closed",
                "full-I/O",
            ),
        ),
        (
            "agent-libos-child-processes",
            (
                "`copy` is not copy-on-write isolation",
                "`speculative` is not automatic rollback",
                "`wait_child_process` targets one direct child",
                "There is no timeout parameter.",
                "`ready=true`",
                "`status=\"exited\"`",
                "`read_process_messages`",
                "`receive_process_messages`",
                "Merge is an irreversible collection boundary.",
                "can delete that generated result",
                "fails closed instead of replaying",
                "Object payloads are runtime-local.",
            ),
        ),
        (
            "agent-libos-object-tasks",
            (
                "with exactly one tool",
                "Start success proves admission, not execution",
                "never duplicate on timeout",
                "Auto-replay applies only",
                "No effects roll back",
                "active tasks become `abandoned`",
                "`result_unavailable_after_reopen`",
                "`superseded_by_restore`",
                "Restore does not undo completed external effects.",
            ),
        ),
    ],
)
def test_long_context_builtin_skill_keeps_recovery_contracts(
    skill_id: str,
    required_contracts: tuple[str, ...],
) -> None:
    package = get_builtin_skill_catalog().get(skill_id)

    assert package is not None
    missing = [
        contract
        for contract in required_contracts
        if contract not in package.instructions
    ]
    assert missing == []


def test_workspace_editing_skill_distinguishes_following_from_deleting_links() -> None:
    package = get_builtin_skill_catalog().get("agent-libos-workspace-editing")

    assert package is not None
    assert "never follows descendant links" in package.instructions
    assert "does remove descendant symlink/junction directory entries" in package.instructions
