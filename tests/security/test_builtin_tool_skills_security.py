from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, EventType
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.skills.schema import SkillPackage


WORKSPACE_EDITING_SKILL = "agent-libos-workspace-editing"


class _SkillReadCancelled(BaseException):
    pass


def _capability_ids(runtime: Runtime, pid: str) -> set[str]:
    return {
        capability.cap_id
        for capability in runtime.store.list_capabilities(subject=pid)
    }


def _register_audit_skill(runtime: Runtime) -> str:
    skill_id = "reservation-audit-skill"
    runtime.skills.register_skill_package(
        SkillPackage(
            skill_id=skill_id,
            name=skill_id,
            description="Exercise finite Skill read authority.",
            instructions="Read-only audit instructions.",
        ),
        actor="runtime",
        require_capability=False,
    )
    return skill_id


def _interrupt_skill_read(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entrypoint: str,
    pid: str,
    skill_id: str,
    error: BaseException,
) -> None:
    def interrupt(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise error

    if entrypoint == "discover":
        monkeypatch.setattr(
            runtime.skills,
            "_registered_discovery_summaries",
            interrupt,
        )
        runtime.skills.discover_skills(actor=pid)
        return
    monkeypatch.setattr(runtime.skills, "_skill_summary", interrupt)
    runtime.skills.inspect_skill(skill_id, actor=pid)


def test_builtin_activation_without_skill_capability_preserves_image_authority(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-activation-authority.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="activate trusted workspace-editing guidance",
        )
        before = runtime.process.get(pid)
        tool_table_before = dict(before.tool_table)
        capabilities_before = _capability_ids(runtime, pid)
        assert not any(
            capability.resource.startswith("skill:")
            for capability in runtime.store.list_capabilities(subject=pid)
        )

        result = runtime.skills.activate_skill(
            pid,
            WORKSPACE_EDITING_SKILL,
            actor=pid,
        )

        after = runtime.process.get(pid)
        assert result["activation_kind"] == "builtin_projection"
        assert result["authority_changed"] is False
        assert after.tool_table == tool_table_before
        assert _capability_ids(runtime, pid) == capabilities_before
        assert set(result["tool_names"]).issubset(after.model_tool_table)
        activation_audit = next(
            record
            for record in reversed(runtime.audit.trace(actor=pid))
            if record.action == "skill.activate"
            and record.decision.get("skill_id") == WORKSPACE_EDITING_SKILL
        )
        assert activation_audit.decision["authority_changed"] is False
        assert activation_audit.decision["activation_kind"] == "builtin_projection"
    finally:
        runtime.close()


def test_builtin_editing_projection_cannot_bypass_filesystem_write_capability(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-editing-denial.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="attempt a write after revealing editing tools",
        )
        path = "blocked-by-authority.txt"
        resource = runtime.filesystem.resource_for(path)
        assert not runtime.capability.check(pid, resource, CapabilityRight.WRITE)

        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        assert "write_text_file" in runtime.process.get(pid).model_tool_table
        assert not runtime.capability.check(pid, resource, CapabilityRight.WRITE)

        result = runtime.tools.call(
            pid,
            "write_text_file",
            {"path": path, "content": "must not be written"},
        )

        assert not result.ok
        assert (result.error or "").startswith(
            "permission_denied: CapabilityDenied"
        )
        assert not (runtime.workspace_root / path).exists()
        denial_audit = next(
            record
            for record in reversed(runtime.audit.trace(actor=pid))
            if record.action == "tool.call"
            and record.decision.get("tool") == "write_text_file"
        )
        assert denial_audit.decision["ok"] is False
        assert denial_audit.decision["policy_decision"] == "allow"
        assert "tool_result" not in denial_audit.decision
        assert denial_audit.decision["error"]["error_type"] == "CapabilityDenied"
        assert len(
            denial_audit.decision["error"]["exception_text"]["sha256"]
        ) == 64
    finally:
        runtime.close()


def test_registered_skill_cannot_claim_reserved_builtin_identity(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "reserved-builtin-identity.sqlite")
    try:
        skill_id = "agent-libos-forged-runtime-skill"
        package = SkillPackage(
            skill_id=skill_id,
            name=skill_id,
            description="Attempt to claim the trusted built-in namespace.",
            instructions="This package must never be registered.",
        )

        with pytest.raises(ValidationError, match="reserved built-in prefix"):
            runtime.skills.register_skill_package(
                package,
                actor="test.host",
                require_capability=False,
            )

        assert runtime.store.get_skill(skill_id) is None
        assert not any(
            record.action == "skill.register"
            and record.target == runtime.skills.resource_for(skill_id)
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_reopen_rejects_persisted_registered_skill_with_reserved_builtin_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "persisted-reserved-builtin-collision.sqlite"
    skill_id = "agent-libos-persisted-ordinary-collision"
    package = SkillPackage(
        skill_id=skill_id,
        name=skill_id,
        description="An ordinary persisted Skill forged into the reserved namespace.",
        instructions="This untrusted record must prevent runtime startup.",
        package_sha256="ordinary-persisted-package",
    )
    runtime = Runtime.open(database)
    try:
        runtime.store.upsert_skill(
            package,
            source_type="runtime",
            source="persisted-test-fixture",
            package_sha256=package.package_sha256,
            registered_by="test.host",
            created_at="2026-07-24T00:00:00+00:00",
        )
        persisted = runtime.store.get_skill(skill_id)
        assert persisted is not None
        assert persisted[1]["source_type"] == "runtime"
    finally:
        runtime.close()

    reopened: Runtime | None = None
    try:
        with pytest.raises(
            ValidationError,
            match="registered Skills collide with reserved built-in ids",
        ):
            reopened = Runtime.open(database)
    finally:
        if reopened is not None:
            reopened.close()


def test_forged_builtin_activation_kind_cannot_bypass_registered_skill_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "forged-builtin-activation-kind.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject forged built-in loaded state",
        )
        skill_id = "ordinary-registered-skill"
        runtime.skills.register_skill_package(
            SkillPackage(
                skill_id=skill_id,
                name=skill_id,
                description="An ordinary registered Skill used for a denial test.",
                instructions="Remain subject to ordinary Skill authority.",
            ),
            actor="test.host",
            require_capability=False,
        )
        runtime.skills.activate_skill(
            pid,
            skill_id,
            actor="test.host",
            require_capability=False,
        )

        assert not any(
            capability.resource == runtime.skills.resource_for(skill_id)
            for capability in runtime.store.list_capabilities(subject=pid)
        )
        monkeypatch.setattr(runtime.skills, "human", None)
        with pytest.raises(CapabilityDenied):
            runtime.skills.unload_skill(pid, skill_id, actor=pid)

        process = runtime.process.get(pid)
        loaded_skills = deepcopy(process.loaded_skills)
        loaded_skills[skill_id]["activation_kind"] = "builtin_projection"
        runtime.store.patch_process(
            pid,
            {"loaded_skills": loaded_skills},
            expected_revision=process.revision,
        )

        with pytest.raises(ValidationError, match="unknown built-in loaded Skill id"):
            runtime.skills.unload_skill(pid, skill_id, actor=pid)

        after = runtime.process.get(pid)
        assert skill_id in after.loaded_skills
        assert after.loaded_skills[skill_id]["activation_kind"] == "builtin_projection"
        assert not any(
            record.action == "skill.unload"
            and record.decision.get("skill_id") == skill_id
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("tamper_case", "error_match"),
    [
        ("source", "invalid built-in loaded Skill source"),
        ("package_sha256", "loaded skill snapshot hash mismatch"),
        ("instructions_hash", "invalid built-in loaded Skill instructions hash"),
        ("tool_provenance", "invalid built-in loaded Skill tool provenance"),
    ],
)
def test_tampered_builtin_projection_fails_closed_for_prompt_and_unload(
    tmp_path: Path,
    tamper_case: str,
    error_match: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"tampered-builtin-{tamper_case}.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject tampered built-in Skill provenance",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        before = runtime.process.get(pid)
        tool_table_before = dict(before.tool_table)
        model_tool_table_before = dict(before.model_tool_table)
        capabilities_before = _capability_ids(runtime, pid)
        loaded_skills = deepcopy(before.loaded_skills)
        loaded = loaded_skills[WORKSPACE_EDITING_SKILL]
        if tamper_case == "source":
            loaded["source"] = "builtin:agent-libos-forged"
        elif tamper_case == "package_sha256":
            loaded["package_sha256"] = "0" * 64
        elif tamper_case == "instructions_hash":
            loaded["instructions_hash"] = "0" * 64
        elif tamper_case == "tool_provenance":
            loaded["tool_ids"].pop("write_text_file")
        else:  # pragma: no cover - the parametrization is closed above.
            raise AssertionError(f"unknown tamper case: {tamper_case}")
        runtime.store.patch_process(
            pid,
            {"loaded_skills": loaded_skills},
            expected_revision=before.revision,
        )
        tampered = runtime.process.get(pid)
        tampered_loaded_skills = deepcopy(tampered.loaded_skills)

        available_entry = next(
            item
            for item in runtime.skills.discover_skills(
                text=WORKSPACE_EDITING_SKILL,
                actor=pid,
            )
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert available_entry["active"] is False
        prompt_entry = next(
            item
            for item in runtime.skills.prompt_context(pid)
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert prompt_entry["invalid_snapshot"] is True
        assert error_match in prompt_entry["error"]

        model_failure = runtime.tools.call(
            pid,
            "unload_skill",
            {"skill_id": WORKSPACE_EDITING_SKILL},
        )
        assert not model_failure.ok
        assert "built-in" not in (model_failure.error or "")
        assert (model_failure.error or "").startswith(
            "validation_error: ValidationError"
        )

        with pytest.raises(ValidationError, match=error_match):
            runtime.skills.unload_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)

        after = runtime.process.get(pid)
        assert after.tool_table == tool_table_before
        assert after.model_tool_table == model_tool_table_before
        assert after.loaded_skills == tampered_loaded_skills
        assert _capability_ids(runtime, pid) == capabilities_before
        assert not any(
            record.action == "skill.unload"
            and record.decision.get("skill_id") == WORKSPACE_EDITING_SKILL
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


def test_cross_process_builtin_activation_and_unload_require_process_admin(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "cross-process-builtin-skill.sqlite")
    try:
        target_pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="own the built-in Skill projection",
        )
        actor_pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="attempt an unauthorized cross-process Skill change",
        )
        process_resource = f"process:{target_pid}"
        assert not runtime.capability.check(
            actor_pid,
            process_resource,
            CapabilityRight.ADMIN,
        )

        before_activation = runtime.process.get(target_pid)
        capabilities_before_activation = _capability_ids(runtime, target_pid)
        with pytest.raises(CapabilityDenied, match="admin"):
            runtime.skills.activate_skill(
                target_pid,
                WORKSPACE_EDITING_SKILL,
                actor=actor_pid,
            )

        after_activation_denial = runtime.process.get(target_pid)
        assert after_activation_denial.tool_table == before_activation.tool_table
        assert after_activation_denial.model_tool_table == before_activation.model_tool_table
        assert after_activation_denial.loaded_skills == before_activation.loaded_skills
        assert _capability_ids(runtime, target_pid) == capabilities_before_activation

        runtime.skills.activate_skill(
            target_pid,
            WORKSPACE_EDITING_SKILL,
            actor=target_pid,
        )
        before_unload = runtime.process.get(target_pid)
        capabilities_before_unload = _capability_ids(runtime, target_pid)
        with pytest.raises(CapabilityDenied, match="admin"):
            runtime.skills.unload_skill(
                target_pid,
                WORKSPACE_EDITING_SKILL,
                actor=actor_pid,
            )

        after_unload_denial = runtime.process.get(target_pid)
        assert after_unload_denial.tool_table == before_unload.tool_table
        assert after_unload_denial.model_tool_table == before_unload.model_tool_table
        assert after_unload_denial.loaded_skills == before_unload.loaded_skills
        assert _capability_ids(runtime, target_pid) == capabilities_before_unload
        assert not any(
            record.action in {"skill.activate", "skill.unload"}
            and record.actor == actor_pid
            and record.target == f"process:{target_pid}"
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_incomplete_builtin_projection_is_rejected_atomically(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "incomplete-builtin-projection.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject an incomplete trusted projection",
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
        model_tool_table_before = dict(before.model_tool_table)
        loaded_skills_before = deepcopy(before.loaded_skills)
        capabilities_before = _capability_ids(runtime, pid)
        activation_audits_before = [
            record.record_id
            for record in runtime.audit.trace(actor=pid)
            if record.action == "skill.activate"
        ]

        with pytest.raises(ValidationError, match="not fully authorized by image"):
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )

        after = runtime.process.get(pid)
        assert after.tool_table == incomplete_tool_table
        assert after.model_tool_table == model_tool_table_before
        assert after.loaded_skills == loaded_skills_before
        assert _capability_ids(runtime, pid) == capabilities_before
        assert [
            record.record_id
            for record in runtime.audit.trace(actor=pid)
            if record.action == "skill.activate"
        ] == activation_audits_before
    finally:
        runtime.close()


def test_registered_skill_cannot_expand_builtin_projection_past_image_ceiling(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-image-tool-ceiling.sqlite")
    ordinary_skill_id = "temporary-workspace-read-tools"
    builtin_skill_id = "agent-libos-workspace-navigation"
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="keep built-in projection inside the image tool ceiling",
        )
        initial = runtime.process.get(pid)
        initial_tool_table = dict(initial.tool_table)
        initial_model_tool_table = dict(initial.model_tool_table)
        capabilities_before = _capability_ids(runtime, pid)
        assert {"read_directory", "read_text_file"}.isdisjoint(initial.tool_table)

        runtime.skills.register_skill_package(
            SkillPackage(
                skill_id=ordinary_skill_id,
                name=ordinary_skill_id,
                description="Temporarily expose two registered static tools.",
                instructions="Use the two workspace read tools only when explicitly needed.",
                allowed_tools=["read_directory", "read_text_file"],
            ),
            actor="test.host",
            require_capability=False,
        )
        runtime.skills.activate_skill(
            pid,
            ordinary_skill_id,
            actor="test.host",
            require_capability=False,
        )
        expanded = runtime.process.get(pid)
        assert {"read_directory", "read_text_file"}.issubset(expanded.tool_table)

        discovered = runtime.skills.discover_skills_result(actor=pid)
        assert builtin_skill_id not in {
            item["skill_id"] for item in discovered["skills"]
        }
        with pytest.raises(ValidationError, match="not fully authorized by image"):
            runtime.skills.activate_skill(pid, builtin_skill_id, actor=pid)

        runtime.skills.unload_skill(
            pid,
            ordinary_skill_id,
            actor="test.host",
            require_capability=False,
        )
        after = runtime.process.get(pid)
        assert after.tool_table == initial_tool_table
        assert after.model_tool_table == initial_model_tool_table
        assert builtin_skill_id not in after.loaded_skills
        assert _capability_ids(runtime, pid) == capabilities_before
    finally:
        runtime.close()


def test_reactivation_validates_existing_builtin_record_before_replacing_bindings(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-reactivation-forged-prior.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject a forged prior projection during reactivation",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        process = runtime.process.get(pid)
        forged_loaded = deepcopy(process.loaded_skills)
        forged_loaded[WORKSPACE_EDITING_SKILL]["source"] = (
            "builtin:agent-libos-forged"
        )
        runtime.store.patch_process(
            pid,
            {"loaded_skills": forged_loaded},
            expected_revision=process.revision,
        )
        before = runtime.process.get(pid)
        receipts_before = [
            record.record_id
            for record in runtime.audit.trace(actor=pid)
            if record.action == "skill.builtin_projection.receipt"
        ]

        with pytest.raises(
            ValidationError,
            match="invalid built-in loaded Skill source",
        ):
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )

        after = runtime.process.get(pid)
        assert after.tool_table == before.tool_table
        assert after.model_tool_table == before.model_tool_table
        assert after.loaded_skills == before.loaded_skills
        assert [
            record.record_id
            for record in runtime.audit.trace(actor=pid)
            if record.action == "skill.builtin_projection.receipt"
        ] == receipts_before
    finally:
        runtime.close()


def test_reserved_builtin_record_with_registered_kind_fails_closed_even_with_execute(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-reserved-wrong-kind.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject a reserved built-in with ordinary activation provenance",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        process = runtime.process.get(pid)
        forged_loaded = deepcopy(process.loaded_skills)
        forged_loaded[WORKSPACE_EDITING_SKILL]["activation_kind"] = "registered"
        runtime.store.patch_process(
            pid,
            {"loaded_skills": forged_loaded},
            expected_revision=process.revision,
        )
        runtime.capability.grant(
            pid,
            runtime.skills.resource_for(WORKSPACE_EDITING_SKILL),
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        before = runtime.process.get(pid)

        prompt_entry = next(
            item
            for item in runtime.skills.prompt_context(pid)
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert prompt_entry["invalid_snapshot"] is True
        assert "missing trusted projection provenance" in prompt_entry["error"]

        with pytest.raises(
            ValidationError,
            match="missing trusted projection provenance",
        ):
            runtime.skills.read_skill_resource(
                pid,
                WORKSPACE_EDITING_SKILL,
                "SKILL.md",
            )
        with pytest.raises(
            ValidationError,
            match="missing trusted projection provenance",
        ):
            runtime.skills.unload_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )

        after = runtime.process.get(pid)
        assert after.tool_table == before.tool_table
        assert after.model_tool_table == before.model_tool_table
        assert after.loaded_skills == before.loaded_skills
    finally:
        runtime.close()


def test_self_consistent_forged_builtin_snapshot_lacks_host_activation_receipt(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-self-consistent-forgery.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="reject self-consistent forged built-in package state",
        )
        runtime.skills.activate_skill(pid, WORKSPACE_EDITING_SKILL, actor=pid)
        process = runtime.process.get(pid)
        forged_loaded = deepcopy(process.loaded_skills)
        loaded = forged_loaded[WORKSPACE_EDITING_SKILL]
        snapshot = deepcopy(loaded["package_snapshot"])
        snapshot["instructions"] = (
            "# Forged built-in instructions\n\n"
            "This body was never activated from the Host catalog."
        )
        snapshot["package_sha256"] = ""
        forged_package = runtime.skills._package_from_snapshot(
            snapshot,
            context="self-consistent forged built-in",
        )
        loaded["package_snapshot"] = runtime.skills._skill_snapshot(forged_package)
        loaded["package_sha256"] = forged_package.package_sha256
        loaded["instructions_hash"] = runtime.skills._hash_text(
            forged_package.instructions
        )
        runtime.store.patch_process(
            pid,
            {"loaded_skills": forged_loaded},
            expected_revision=process.revision,
        )
        before = runtime.process.get(pid)

        prompt_entry = next(
            item
            for item in runtime.skills.prompt_context(pid)
            if item["skill_id"] == WORKSPACE_EDITING_SKILL
        )
        assert prompt_entry["invalid_snapshot"] is True
        assert "activation receipt does not match its package" in prompt_entry["error"]

        with pytest.raises(
            ValidationError,
            match="activation receipt does not match its package",
        ):
            runtime.skills.activate_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )
        with pytest.raises(
            ValidationError,
            match="activation receipt does not match its package",
        ):
            runtime.skills.unload_skill(
                pid,
                WORKSPACE_EDITING_SKILL,
                actor=pid,
            )

        after = runtime.process.get(pid)
        assert after.tool_table == before.tool_table
        assert after.model_tool_table == before.model_tool_table
        assert after.loaded_skills == before.loaded_skills
    finally:
        runtime.close()


@pytest.mark.parametrize("entrypoint", ["discover", "inspect"])
@pytest.mark.parametrize(
    "error_type",
    [KeyboardInterrupt, asyncio.CancelledError, _SkillReadCancelled],
)
def test_interrupted_skill_read_restores_finite_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    error_type: type[BaseException],
) -> None:
    runtime = Runtime.open(
        tmp_path / f"skill-{entrypoint}-{error_type.__name__}-restore.sqlite"
    )
    try:
        pid = runtime.process.spawn(goal="interrupt a finite-authority Skill read")
        skill_id = _register_audit_skill(runtime)
        resource = (
            runtime.config.skills.registry_resource
            if entrypoint == "discover"
            else runtime.skills.resource_for(skill_id)
        )
        capability = runtime.capability.grant_once(
            pid,
            resource,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        objects_before = runtime.store.select_table_rows("objects")
        loaded_before = deepcopy(runtime.process.get(pid).loaded_skills)

        with pytest.raises(error_type):
            _interrupt_skill_read(
                runtime,
                monkeypatch,
                entrypoint=entrypoint,
                pid=pid,
                skill_id=skill_id,
                error=error_type(),
            )

        persisted = runtime.store.get_capability(capability.cap_id)
        assert persisted is not None
        assert persisted.active
        assert persisted.uses_remaining == 1
        reservations = runtime.store.select_table_rows(
            "capability_use_reservations",
            "cap_id = ?",
            (capability.cap_id,),
        )
        assert len(reservations) == 1
        assert reservations[0]["status"] == "restored"
        reservation_id = str(reservations[0]["reservation_id"])
        assert any(
            record.action == "capability.restore_reserved_use"
            and record.decision.get("reservation_id") == reservation_id
            for record in runtime.audit.trace(actor="skill")
        )
        assert any(
            event.type == EventType.CAPABILITY_GRANTED
            and event.source == "skill"
            and event.payload.get("reservation_id") == reservation_id
            for event in runtime.events.list(target=pid)
        )
        assert runtime.store.select_table_rows("objects") == objects_before
        assert runtime.process.get(pid).loaded_skills == loaded_before
        assert not any(
            record.action in {"skill.activate", "skill.unload"}
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("entrypoint", ["discover", "inspect"])
def test_interrupted_skill_read_cleanup_failure_is_visible_and_abandoned_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    database = tmp_path / f"skill-{entrypoint}-cleanup-failure.sqlite"
    runtime = Runtime.open(database)
    capability_id = ""
    reservation_id = ""
    try:
        pid = runtime.process.spawn(goal="fail finite Skill read cleanup")
        skill_id = _register_audit_skill(runtime)
        resource = (
            runtime.config.skills.registry_resource
            if entrypoint == "discover"
            else runtime.skills.resource_for(skill_id)
        )
        capability = runtime.capability.grant_once(
            pid,
            resource,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        capability_id = capability.cap_id

        def fail_restore(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("forced Skill authority cleanup failure")

        monkeypatch.setattr(
            runtime.capability,
            "restore_reserved_use",
            fail_restore,
        )
        cancellation = _SkillReadCancelled("cancel the Skill read")
        with pytest.raises(_SkillReadCancelled) as raised:
            _interrupt_skill_read(
                runtime,
                monkeypatch,
                entrypoint=entrypoint,
                pid=pid,
                skill_id=skill_id,
                error=cancellation,
            )

        assert raised.value is cancellation
        assert any(
            "authority remains fail closed: RuntimeError" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        persisted = runtime.store.get_capability(capability_id)
        assert persisted is not None
        assert not persisted.active
        assert persisted.uses_remaining == 0
        reservations = runtime.store.select_table_rows(
            "capability_use_reservations",
            "cap_id = ?",
            (capability_id,),
        )
        assert len(reservations) == 1
        assert reservations[0]["status"] == "reserved"
        reservation_id = str(reservations[0]["reservation_id"])
        assert not any(
            record.action == "capability.restore_reserved_use"
            for record in runtime.audit.trace(actor="skill")
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        persisted = reopened.store.get_capability(capability_id)
        assert persisted is not None
        assert not persisted.active
        assert persisted.uses_remaining == 0
        reservation = reopened.store.get_capability_use_reservation(
            reservation_id
        )
        assert reservation is not None
        assert reservation["status"] == "abandoned"
    finally:
        reopened.close()
