from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight
from tests.support.skills import write_skill_package


WORKSPACE_EDITING_SKILL = "agent-libos-workspace-editing"
OVERLAP_SKILL = "workspace-write-overlap"
OVERLAP_TOOL = "write_text_file"


def test_same_image_fork_inherits_builtin_projection_but_spawn_child_is_fresh(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projection-children.sqlite")
    try:
        parent = runtime.process.spawn(
            image="coding-agent:v0",
            goal="activate editing before creating children",
        )
        runtime.skills.activate_skill(parent, WORKSPACE_EDITING_SKILL, actor=parent)
        parent_process = runtime.process.get(parent)
        parent_full_table = dict(parent_process.tool_table)
        parent_model_table = dict(parent_process.model_tool_table)
        parent_snapshot = deepcopy(
            parent_process.loaded_skills[WORKSPACE_EDITING_SKILL]
        )

        forked = runtime.process.fork(
            parent,
            goal="inherit the current working context",
        )
        forked_process = runtime.process.get(forked)

        assert forked_process.tool_table == parent_full_table
        assert forked_process.model_tool_table == parent_model_table
        assert (
            forked_process.loaded_skills[WORKSPACE_EDITING_SKILL]
            == parent_snapshot
        )
        _assert_projection_prompt_uses_snapshot(runtime, forked, parent_snapshot)
        _assert_no_skill_capabilities(runtime, forked)

        spawned = runtime.process.spawn_child(
            parent,
            goal="start with a fresh child context",
        )
        spawned_process = runtime.process.get(spawned)

        assert spawned_process.tool_table == parent_full_table
        assert WORKSPACE_EDITING_SKILL not in spawned_process.loaded_skills
        assert OVERLAP_TOOL not in spawned_process.model_tool_table
        assert len(spawned_process.model_tool_table) == 5
        assert spawned_process.loaded_skills == {}
        _assert_no_skill_capabilities(runtime, spawned)
    finally:
        runtime.close()


def test_checkpoint_restore_reinstates_builtin_projection_and_snapshot(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projection-restore.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="restore an activated editing projection",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        before = runtime.process.get(pid)
        expected_full_table = dict(before.tool_table)
        expected_model_table = dict(before.model_tool_table)
        expected_loaded = deepcopy(before.loaded_skills)
        expected_capability_ids = _capability_ids(runtime, pid)

        checkpoint_id = runtime.checkpoint.create(
            pid,
            "built-in projection is active",
            actor=pid,
        )
        runtime.skills.unload_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        assert WORKSPACE_EDITING_SKILL not in runtime.process.get(pid).loaded_skills

        result = runtime.checkpoint.restore(
            "test.host",
            checkpoint_id,
            require_capability=False,
        )
        restored = runtime.process.get(pid)

        assert result["status"] == "restored"
        assert restored.tool_table == expected_full_table
        assert restored.model_tool_table == expected_model_table
        assert restored.loaded_skills == expected_loaded
        assert expected_capability_ids.issubset(_capability_ids(runtime, pid))
        _assert_no_skill_capabilities(runtime, pid)
        _assert_projection_prompt_uses_snapshot(
            runtime,
            pid,
            expected_loaded[WORKSPACE_EDITING_SKILL],
        )
    finally:
        runtime.close()


def test_repeated_builtin_activation_is_idempotent_and_unloads_once(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projection-reactivate.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="activate the same built-in projection twice",
        )
        initial = runtime.process.get(pid)
        initial_full_table = dict(initial.tool_table)
        initial_model_table = dict(initial.model_tool_table)
        initial_loaded_skill_ids = set(initial.loaded_skills)
        initial_capability_ids = _capability_ids(runtime, pid)

        first_result = runtime.skills.activate_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )
        after_first = runtime.process.get(pid)
        first_model_table = dict(after_first.model_tool_table)

        second_result = runtime.skills.activate_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )
        after_second = runtime.process.get(pid)

        assert first_result["tool_ids"] == second_result["tool_ids"]
        assert after_second.tool_table == initial_full_table
        assert after_second.model_tool_table == first_model_table
        assert _capability_ids(runtime, pid) == initial_capability_ids
        assert set(after_second.loaded_skills) == {
            *initial_loaded_skill_ids,
            WORKSPACE_EDITING_SKILL,
        }
        assert list(after_second.loaded_skills).count(WORKSPACE_EDITING_SKILL) == 1
        for name, tool_id in second_result["tool_ids"].items():
            assert after_second.model_tool_table[name] == tool_id

        runtime.skills.unload_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )
        unloaded = runtime.process.get(pid)

        assert unloaded.tool_table == initial_full_table
        assert unloaded.model_tool_table == initial_model_table
        assert set(unloaded.loaded_skills) == initial_loaded_skill_ids
        assert _capability_ids(runtime, pid) == initial_capability_ids
    finally:
        runtime.close()


def test_checkpoint_fork_remaps_and_preserves_builtin_projection(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projection-checkpoint-fork.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="fork an activated editing projection from a checkpoint",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        source = runtime.process.get(pid)
        expected_full_table = dict(source.tool_table)
        expected_model_table = dict(source.model_tool_table)
        expected_snapshot = deepcopy(
            source.loaded_skills[WORKSPACE_EDITING_SKILL]
        )

        checkpoint_id = runtime.checkpoint.create(
            pid,
            "built-in projection fork point",
            actor=pid,
        )
        runtime.capability.grant(
            pid,
            f"checkpoint:{checkpoint_id}",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        result = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
        forked_pid = result["fork_root_pid"]
        forked = runtime.process.get(forked_pid)

        assert result["status"] == "forked"
        assert forked.tool_table == expected_full_table
        assert forked.model_tool_table == expected_model_table
        assert (
            forked.loaded_skills[WORKSPACE_EDITING_SKILL]
            == expected_snapshot
        )
        _assert_projection_prompt_uses_snapshot(
            runtime,
            forked_pid,
            expected_snapshot,
        )
        _assert_no_skill_capabilities(runtime, forked_pid)
        assert runtime.process.get(pid).tool_table == expected_full_table
    finally:
        runtime.close()


def test_checkpoint_committed_image_preserves_builtin_projection_on_spawn_and_exec(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-projection-image.sqlite")
    try:
        source = runtime.process.spawn(
            image="coding-agent:v0",
            goal="commit an activated editing projection",
        )
        runtime.skills.activate_skill(
            source,
            WORKSPACE_EDITING_SKILL,
            actor=source,
        )
        source_process = runtime.process.get(source)
        expected_full_table = dict(source_process.tool_table)
        expected_model_table = dict(source_process.model_tool_table)
        expected_snapshot = deepcopy(
            source_process.loaded_skills[WORKSPACE_EDITING_SKILL]
        )

        checkpoint_id = runtime.checkpoint.create(
            source,
            "commit built-in projection",
            actor=source,
        )
        image_id = "builtin-projection-image:v0"
        runtime.image_registry.grant_register(source, image_id, issued_by="test")
        committed = runtime.image_registry.commit_from_checkpoint(
            actor=source,
            checkpoint_id=checkpoint_id,
            image_id=image_id,
            name="builtin-projection-image",
        )

        assert committed.image.metadata["tool_projection"] == "skills"
        booted = runtime.process.spawn(
            image=image_id,
            goal="boot the committed projection",
        )
        booted_process = runtime.process.get(booted)
        assert booted_process.tool_table == expected_full_table
        assert booted_process.model_tool_table == expected_model_table
        assert (
            booted_process.loaded_skills[WORKSPACE_EDITING_SKILL]
            == expected_snapshot
        )
        _assert_projection_prompt_uses_snapshot(
            runtime,
            booted,
            expected_snapshot,
        )
        _assert_no_skill_capabilities(runtime, booted)

        target = runtime.process.spawn(
            image="base-agent:v0",
            goal="exec into the committed projection",
        )
        runtime.capability.grant(
            target,
            runtime.image_registry.resource_for(image_id),
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.exec_process(
            target,
            image_id,
            goal="run the committed projection",
            preserve_capabilities=False,
        )
        executed = runtime.process.get(target)

        assert executed.tool_table == expected_full_table
        assert executed.model_tool_table == expected_model_table
        assert (
            executed.loaded_skills[WORKSPACE_EDITING_SKILL]
            == expected_snapshot
        )
        _assert_projection_prompt_uses_snapshot(
            runtime,
            target,
            expected_snapshot,
        )
        _assert_no_skill_capabilities(runtime, target)
    finally:
        runtime.close()


def test_reopen_reproduces_old_builtin_snapshot_after_catalog_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "builtin-projection-upgrade.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="retain the loaded package across an upgrade",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        expected_full_table = dict(runtime.process.get(pid).tool_table)
        loaded_snapshot = deepcopy(
            runtime.process.get(pid).loaded_skills[WORKSPACE_EDITING_SKILL]
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        catalog = reopened.skills._builtin_catalog
        current = catalog.get(WORKSPACE_EDITING_SKILL)
        assert current is not None
        upgraded = replace(
            current,
            instructions="# Upgraded catalog body\n\nUse the new workflow only.",
            package_sha256="",
        )
        upgraded = replace(
            upgraded,
            package_sha256=reopened.skills._package_hash(upgraded),
        )
        monkeypatch.setitem(
            catalog._packages,
            WORKSPACE_EDITING_SKILL,
            upgraded,
        )

        current_catalog = catalog.get(WORKSPACE_EDITING_SKILL)
        assert current_catalog is not None
        assert current_catalog.instructions == upgraded.instructions
        _assert_projection_prompt_uses_snapshot(
            reopened,
            pid,
            loaded_snapshot,
        )
        loaded_context = _loaded_context(reopened, pid)
        assert loaded_context["instructions"] != upgraded.instructions
        assert reopened.process.get(pid).tool_table == expected_full_table

        discovered = reopened.tools.call(
            pid,
            "discover_skills",
            {"text": WORKSPACE_EDITING_SKILL, "limit": 1},
        )
        assert discovered.ok
        assert discovered.payload["skills"][0]["package_sha256"] == upgraded.package_sha256
        assert discovered.payload["skills"][0]["active"] is False
        assert discovered.payload["next_step"] == "activate_skill"

        stale = reopened.tools.call(
            pid,
            "activate_skill",
            {
                "skill_id": WORKSPACE_EDITING_SKILL,
                "expected_package_sha256": loaded_snapshot["package_sha256"],
            },
        )
        assert not stale.ok
        assert stale.payload["error"]["details"]["error_type"] == "SkillPackageChanged"
        assert (
            reopened.process.get(pid).loaded_skills[WORKSPACE_EDITING_SKILL]
            == loaded_snapshot
        )

        activated = reopened.tools.call(
            pid,
            "activate_skill",
            {
                "skill_id": WORKSPACE_EDITING_SKILL,
                "expected_package_sha256": upgraded.package_sha256,
            },
        )
        assert activated.ok
        assert activated.payload["result"]["package_sha256"] == upgraded.package_sha256
        assert _loaded_context(reopened, pid)["instructions"] == upgraded.instructions
        rediscovered = reopened.tools.call(
            pid,
            "discover_skills",
            {"text": WORKSPACE_EDITING_SKILL, "limit": 1},
        )
        assert rediscovered.ok
        assert rediscovered.payload["skills"][0]["active"] is True
        assert rediscovered.payload["next_step"] == "use_loaded_skill"
        _assert_no_skill_capabilities(reopened, pid)
    finally:
        reopened.close()


def test_reopen_drops_builtin_projection_outside_replaced_image_ceiling(
    tmp_path: Path,
) -> None:
    database = tmp_path / "builtin-projection-image-ceiling-reopen.sqlite"
    runtime = Runtime.open(database)
    editing_tools: set[str]
    try:
        image_id = "narrowable-coding-agent:v0"
        runtime.image_registry.register(
            replace(
                runtime.get_image("coding-agent:v0"),
                image_id=image_id,
                name="narrowable-coding-agent",
            ),
            actor="test.host",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image=image_id,
            goal="drop guidance that no longer fits a replaced image",
        )
        activated = runtime.skills.activate_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )
        editing_tools = set(activated["tool_names"])
        image = runtime.get_image(image_id)
        runtime.image_registry.register(
            replace(
                image,
                default_tools=[
                    name for name in image.default_tools if name not in editing_tools
                ],
            ),
            actor="test.host",
            replace=True,
            require_capability=False,
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        process = reopened.process.get(pid)
        assert WORKSPACE_EDITING_SKILL not in process.loaded_skills
        assert editing_tools.isdisjoint(process.model_tool_table)
        assert editing_tools.isdisjoint(
            schema["function"]["name"]
            for schema in reopened.tools.openai_tool_schemas(pid)
        )
        assert any(
            record.actor == "runtime"
            and record.action == "skill.unload"
            and record.decision.get("skill_id") == WORKSPACE_EDITING_SKILL
            and record.decision.get("authority_changed") is False
            for record in reopened.audit.trace()
        )
    finally:
        reopened.close()


def test_cross_image_fork_rebases_builtin_model_baseline_before_unload(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-cross-image-model-baseline.sqlite")
    try:
        source_image = runtime.get_image("coding-agent:v0")
        source_metadata = dict(source_image.metadata)
        source_metadata.pop("tool_projection", None)
        source_image_id = "full-projection-coding-source:v0"
        runtime.image_registry.register(
            replace(
                source_image,
                image_id=source_image_id,
                name="full-projection-coding-source",
                default_skills=[],
                metadata=source_metadata,
            ),
            actor="test.host",
            require_capability=False,
        )
        parent = runtime.process.spawn(
            image=source_image_id,
            goal="carry one built-in projection into a skills-projected image",
        )
        activated = runtime.skills.activate_skill(
            parent,
            WORKSPACE_EDITING_SKILL,
            actor=parent,
        )
        editing_tools = set(activated["tool_names"])
        assert editing_tools.issubset(runtime.process.get(parent).model_tool_table)

        child = runtime.process.fork(
            parent,
            goal="rebase inherited model visibility",
            image="coding-agent:v0",
        )
        runtime.skills.unload_skill(
            child,
            WORKSPACE_EDITING_SKILL,
            actor=child,
        )

        child_process = runtime.process.get(child)
        assert editing_tools.isdisjoint(child_process.model_tool_table)
        assert len(child_process.model_tool_table) == 5
    finally:
        runtime.close()


@pytest.mark.parametrize("first", ["registered", "builtin"])
def test_registered_and_builtin_skill_overlap_survives_either_unload_order(
    tmp_path: Path,
    first: str,
) -> None:
    skill_root = write_skill_package(
        tmp_path,
        OVERLAP_SKILL,
        allowed_tools=[OVERLAP_TOOL],
        body="# Workspace write overlap\n\nUse one existing static write tool.\n",
    )
    runtime = Runtime.open(tmp_path / f"builtin-overlap-{first}.sqlite")
    try:
        runtime.skills.register_skill_from_path(
            skill_root,
            actor="test.host",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="exercise overlapping static Skill bindings",
        )
        runtime.capability.grant(
            pid,
            f"skill:{OVERLAP_SKILL}",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        initial = runtime.process.get(pid)
        expected_full_table = dict(initial.tool_table)
        tool_id = expected_full_table[OVERLAP_TOOL]
        assert OVERLAP_TOOL not in initial.model_tool_table

        if first == "registered":
            runtime.skills.activate_skill(pid, OVERLAP_SKILL, actor=pid)
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )
        else:
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )
            runtime.skills.activate_skill(pid, OVERLAP_SKILL, actor=pid)

        active = runtime.process.get(pid)
        assert active.tool_table == expected_full_table
        assert active.model_tool_table[OVERLAP_TOOL] == tool_id
        assert {OVERLAP_SKILL, WORKSPACE_EDITING_SKILL}.issubset(
            active.loaded_skills
        )

        first_skill = (
            OVERLAP_SKILL if first == "registered" else WORKSPACE_EDITING_SKILL
        )
        runtime.skills.unload_skill(pid, first_skill, actor=pid)

        remaining = runtime.process.get(pid)
        remaining_skill = (
            WORKSPACE_EDITING_SKILL if first == "registered" else OVERLAP_SKILL
        )
        assert remaining.tool_table == expected_full_table
        assert remaining.model_tool_table[OVERLAP_TOOL] == tool_id
        assert remaining_skill in remaining.loaded_skills

        runtime.skills.unload_skill(pid, remaining_skill, actor=pid)

        unloaded = runtime.process.get(pid)
        assert unloaded.tool_table == expected_full_table
        assert OVERLAP_TOOL not in unloaded.model_tool_table
        assert OVERLAP_SKILL not in unloaded.loaded_skills
        assert WORKSPACE_EDITING_SKILL not in unloaded.loaded_skills
    finally:
        runtime.close()


def _assert_projection_prompt_uses_snapshot(
    runtime: Runtime,
    pid: str,
    loaded_snapshot: dict[str, object],
) -> None:
    process = runtime.process.get(pid)
    loaded = process.loaded_skills[WORKSPACE_EDITING_SKILL]
    assert loaded["activation_kind"] == "builtin_projection"
    assert loaded["package_sha256"] == loaded_snapshot["package_sha256"]
    assert loaded["package_snapshot"] == loaded_snapshot["package_snapshot"]
    assert loaded["tool_ids"] == loaded_snapshot["tool_ids"]
    for name, tool_id in loaded["tool_ids"].items():
        assert process.tool_table[name] == tool_id
        assert process.model_tool_table[name] == tool_id

    prompt_context = _loaded_context(runtime, pid)
    package_snapshot = loaded_snapshot["package_snapshot"]
    assert isinstance(package_snapshot, dict)
    assert prompt_context["instructions"] == package_snapshot["instructions"]


def _loaded_context(runtime: Runtime, pid: str) -> dict[str, object]:
    return next(
        item
        for item in runtime.skills.prompt_context(pid)
        if item["skill_id"] == WORKSPACE_EDITING_SKILL
    )


def _capability_ids(runtime: Runtime, pid: str) -> set[str]:
    return {
        capability.cap_id
        for capability in runtime.store.list_capabilities(subject=pid)
    }


def _assert_no_skill_capabilities(runtime: Runtime, pid: str) -> None:
    assert all(
        not capability.resource.startswith("skill:")
        for capability in runtime.store.list_capabilities(subject=pid)
    )
