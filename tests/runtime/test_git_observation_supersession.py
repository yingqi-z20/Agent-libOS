from __future__ import annotations

import hashlib
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.memory.object_memory import _observation_supersession_key
from agent_libos.models import ObjectType
from tests.support.fakes import RecordingActionClient


def _git_result(tool_name: str, **changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository_id": "repository",
        "worktree_id": "main",
    }
    if tool_name == "git_diff":
        result.update({
            "scope": "worktree", "base_oid": None, "head_oid": None,
            "paths_sha256": hashlib.sha256(b"a.py").hexdigest(),
            "max_bytes": 1024, "patch": "FIRST_PATCH", "changed_paths": [],
        })
    elif tool_name == "git_status":
        result.update({"limit": 20, "entries": []})
    elif tool_name == "git_log":
        result.update({"ref_oid": "a" * 40, "limit": 20, "commits": []})
    result.update(changes)
    return {"tool_name": tool_name, "result": result}


@pytest.mark.parametrize(
    ("tool_name", "changes"),
    [
        ("git_diff", {"paths_sha256": hashlib.sha256(b"b.py").hexdigest()}),
        ("git_diff", {"max_bytes": 128}),
        ("git_diff", {"scope": "staged"}),
        ("git_diff", {"base_oid": "b" * 40}),
        ("git_diff", {"head_oid": "b" * 40}),
        ("git_diff", {"worktree_id": "other"}),
        ("git_status", {"worktree_id": "other"}),
        ("git_status", {"limit": 1}),
        ("git_log", {"worktree_id": "other"}),
        ("git_log", {"ref_oid": "b" * 40}),
        ("git_log", {"limit": 1}),
        ("git_repository_info", {"worktree_id": "other"}),
    ],
)
def test_git_observations_keep_distinct_selections_and_extents(
    tool_name: str, changes: dict[str, Any],
) -> None:
    original = _observation_supersession_key(_git_result(tool_name))
    different = _observation_supersession_key(_git_result(tool_name, **changes))
    assert original is not None and different is not None
    assert original != different


@pytest.mark.parametrize(
    "tool_name", ["git_diff", "git_status", "git_log", "git_repository_info"],
)
def test_git_observations_require_complete_selection_provenance(tool_name: str) -> None:
    payload = _git_result(tool_name)
    expected = _observation_supersession_key(payload)
    assert expected is not None
    for field in ("repository_id", "worktree_id"):
        incomplete = {**payload, "result": dict(payload["result"])}
        incomplete["result"].pop(field)
        assert _observation_supersession_key(incomplete) is None
    if tool_name != "git_repository_info":
        field = "max_bytes" if tool_name == "git_diff" else "limit"
        for value in (None, 0, True, "20"):
            assert _observation_supersession_key(_git_result(tool_name, **{field: value})) is None
        incomplete = {**payload, "result": dict(payload["result"])}
        incomplete["result"].pop(field)
        assert _observation_supersession_key(incomplete) is None
    if tool_name == "git_diff":
        # Legacy patch records have changed_paths but no requested selection.
        payload["result"].pop("paths_sha256")
        assert _observation_supersession_key(payload) is None
    elif tool_name == "git_log":
        payload["result"].pop("ref_oid")
        assert _observation_supersession_key(payload) is None
        assert _observation_supersession_key(_git_result(tool_name, ref_oid=None)) is not None


def test_same_batch_filtered_diffs_reach_the_next_quantum_and_only_exact_repeats_supersede() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="Inspect both changed files")
        first_payload = _git_result("git_diff")
        second_payload = _git_result(
            "git_diff", paths_sha256=hashlib.sha256(b"b.py").hexdigest(),
            patch="SECOND_PATCH",
        )
        first = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, first_payload)
        second = runtime.memory.create_object(pid, ObjectType.TOOL_RESULT, second_payload)
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(
            pid, [*process.memory_view.roots, first, second],
        )
        runtime.store.update_process(process)
        client = RecordingActionClient([{"action": "discover_skills", "text": "git"}])
        runtime.llm.client = client

        runtime.run_process_once(pid)

        assert "FIRST_PATCH" in client.user_prompts[0]
        assert "SECOND_PATCH" in client.user_prompts[0]
        repeated = runtime.memory.create_object(
            pid, ObjectType.TOOL_RESULT, _git_result("git_diff", patch="FRESH_FIRST_PATCH"),
        )
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [first, second, repeated]),
            policy="working_set", budget_tokens=100_000, charge_resources=False,
        )
        assert context.object_refs == [second.oid, repeated.oid]
        assert context.omitted_objects == [first.oid]
        assert next(row for row in context.object_manifest if row["oid"] == first.oid)["reason"] == "superseded"
    finally:
        runtime.close()
