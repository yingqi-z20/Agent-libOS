from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SkillDefaults
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.skills.schema import SkillPackage
from agent_libos.substrate import LocalResourceProviderSubstrate


WORKSPACE_EDITING_SKILL = "agent-libos-workspace-editing"


def _write_minimal_skill(root: Path, name: str, *, description: str) -> Path:
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                f"# {name}",
                "",
                "Use this Skill for the described task.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return package


def _register_private_skill(runtime: Runtime, skill_id: str) -> None:
    runtime.skills.register_skill_package(
        SkillPackage(
            skill_id=skill_id,
            name=skill_id,
            description="Private registered Skill used for authority-order checks.",
            instructions="Do not expose these instructions without exact authority.",
        ),
        actor="test.host",
        require_capability=False,
    )


@pytest.mark.parametrize("operation", ["inspect", "activate"])
def test_registered_exact_operations_authorize_before_registry_lookup(
    tmp_path: Path,
    operation: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"skill-{operation}-error-order.sqlite")
    try:
        private_id = f"private-{operation}-skill"
        _register_private_skill(runtime, private_id)
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"exercise {operation} authority order",
        )
        # Keep the regression deterministic: a denied exact request must not
        # be converted into a pending Human approval during this test.
        runtime.skills.human = None

        if operation == "inspect":
            invoke: Callable[[str], object] = lambda selected: runtime.skills.inspect_skill(
                selected,
                actor=pid,
            )
        else:
            invoke = lambda selected: runtime.skills.activate_skill(
                pid,
                selected,
                actor=pid,
            )

        for selected in (private_id, f"missing-{operation}-skill"):
            with pytest.raises(CapabilityDenied):
                invoke(selected)
    finally:
        runtime.close()


@pytest.mark.parametrize("skill_id", ["revoked-existing-skill", "revoked-missing-skill"])
def test_registered_activation_reauthorizes_before_any_registry_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skill_id: str,
) -> None:
    runtime = Runtime.open(tmp_path / f"skill-revocation-{skill_id}.sqlite")
    try:
        _register_private_skill(runtime, "revoked-existing-skill")
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="close the activation reauthorization race",
        )
        authority = runtime.capability.grant(
            pid,
            "skill:*",
            [CapabilityRight.EXECUTE],
            issued_by="test.host",
        )
        original_require = runtime.skills._require_skill_right
        original_get = runtime.skills._get_skill
        registry_lookups = 0

        def revoke_after_preflight(
            actor: str,
            selected_skill_id: str,
            right: CapabilityRight,
        ):
            decisions = original_require(actor, selected_skill_id, right)
            runtime.capability.revoke(
                authority.cap_id,
                revoked_by="test.host",
                reason="Skill activation reauthorization regression",
                require_authority=False,
            )
            return decisions

        def count_registry_lookup(selected_skill_id: str):
            nonlocal registry_lookups
            registry_lookups += 1
            return original_get(selected_skill_id)

        monkeypatch.setattr(runtime.skills, "_require_skill_right", revoke_after_preflight)
        monkeypatch.setattr(runtime.skills, "_get_skill", count_registry_lookup)

        with pytest.raises(CapabilityDenied):
            runtime.skills.activate_skill(pid, skill_id, actor=pid)
        assert registry_lookups == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid_hash", ["A" * 64, "a" * 63, "not-a-hash"])
def test_activation_rejects_invalid_expected_hash_before_authority_or_identity_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_hash: str,
) -> None:
    runtime = Runtime.open(tmp_path / "skill-invalid-expected-hash.sqlite")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject an invalid Skill package precondition",
        )
        before_events = [event.event_id for event in runtime.events.list()]
        before_audit = [record.record_id for record in runtime.audit.trace()]

        def unexpected_authority(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("invalid hash reached Skill authority lookup")

        def unexpected_identity(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("invalid hash reached Skill identity lookup")

        monkeypatch.setattr(runtime.skills, "_require_skill_right", unexpected_authority)
        monkeypatch.setattr(runtime.skills, "_get_skill", unexpected_identity)

        with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
            runtime.skills.activate_skill(
                pid,
                "private-invalid-hash-skill",
                actor=pid,
                expected_package_sha256=invalid_hash,
            )

        assert [event.event_id for event in runtime.events.list()] == before_events
        assert [record.record_id for record in runtime.audit.trace()] == before_audit
    finally:
        runtime.close()


def test_cross_process_builtin_activation_authorizes_before_target_state(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "builtin-target-error-order.sqlite")
    try:
        actor = runtime.process.spawn(
            image="base-agent:v0",
            goal="attempt cross-process built-in activation",
        )
        compatible = runtime.process.spawn(
            image="coding-agent:v0",
            goal="compatible target",
        )
        incompatible = runtime.process.spawn(
            image="base-agent:v0",
            goal="incompatible target",
        )

        for target in ("missing-target", compatible, incompatible):
            with pytest.raises(CapabilityDenied):
                runtime.skills.activate_skill(
                    target,
                    WORKSPACE_EDITING_SKILL,
                    actor=actor,
                )
    finally:
        runtime.close()


def test_cross_process_registered_unload_authorizes_before_loaded_state(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "unload-target-error-order.sqlite")
    skill_id = "private-unload-order-skill"
    try:
        _register_private_skill(runtime, skill_id)
        actor = runtime.process.spawn(
            image="base-agent:v0",
            goal="attempt cross-process registered unload",
        )
        unloaded = runtime.process.spawn(image="base-agent:v0", goal="unloaded target")
        loaded = runtime.process.spawn(image="base-agent:v0", goal="loaded target")
        runtime.skills.activate_skill(
            loaded,
            skill_id,
            actor="test.host",
            require_capability=False,
        )
        runtime.capability.grant(
            actor,
            runtime.skills.resource_for(skill_id),
            [CapabilityRight.EXECUTE],
            issued_by="test.host",
        )

        for target in ("missing-target", unloaded, loaded):
            with pytest.raises(CapabilityDenied):
                runtime.skills.unload_skill(target, skill_id, actor=actor)
        assert skill_id in runtime.process.get(loaded).loaded_skills
    finally:
        runtime.close()


def test_host_skill_catalog_rejects_entry_beyond_scan_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "catalog"
    for index in range(3):
        _write_minimal_skill(
            catalog,
            f"bounded-catalog-{index}",
            description="Bounded catalog marker.",
        )
    config = AgentLibOSConfig(
        skills=replace(
            SkillDefaults(),
            workspace_dirs=(str(catalog),),
            global_dirs=(),
            catalog_scan_limit=3,
        )
    )
    runtime = Runtime.open(tmp_path / "bounded-catalog.sqlite", config=config)
    try:
        discovered = runtime.skills.discover_skills(
            text="bounded catalog marker",
            require_capability=False,
        )
        assert len(discovered) == 3

        _write_minimal_skill(
            catalog,
            "bounded-catalog-overflow",
            description="Overflow entry must not be parsed.",
        )
        original_load = runtime.skills._load_package_from_host_path
        loaded_paths: list[Path] = []

        def record_load(path: str | Path):
            loaded_paths.append(Path(path))
            return original_load(path)

        monkeypatch.setattr(runtime.skills, "_load_package_from_host_path", record_load)
        with pytest.raises(ValidationError, match="catalog_scan_limit=3"):
            runtime.skills.discover_skills(
                text="no matching package",
                require_capability=False,
            )
        assert loaded_paths == []
    finally:
        runtime.close()


def test_registered_skill_search_rejects_row_beyond_scan_limit(
    tmp_path: Path,
) -> None:
    config = AgentLibOSConfig(
        skills=replace(SkillDefaults(), catalog_scan_limit=3)
    )
    runtime = Runtime.open(tmp_path / "bounded-registered-catalog.sqlite", config=config)
    try:
        for index in range(3):
            runtime.skills.register_skill_package(
                SkillPackage(
                    skill_id=f"bounded-registered-{index}",
                    name=f"bounded-registered-{index}",
                    description="Registered bounded marker.",
                    instructions="Use this bounded registered Skill.",
                ),
                actor="test.host",
                require_capability=False,
            )
        discovered = runtime.skills.discover_skills(
            "registered bounded marker",
            actor="test.host",
            require_capability=False,
        )
        assert len(discovered) == 3

        runtime.skills.register_skill_package(
            SkillPackage(
                skill_id="bounded-registered-overflow",
                name="bounded-registered-overflow",
                description="Registered overflow marker.",
                instructions="This row is beyond the configured search scan ceiling.",
            ),
            actor="test.host",
            require_capability=False,
        )
        with pytest.raises(ValidationError, match="catalog_scan_limit=3"):
            runtime.skills.discover_skills(
                "registered marker",
                actor="test.host",
                require_capability=False,
            )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("limit_field", "limit_value", "expected_error"),
    [
        ("max_package_directories", 2, "max_package_directories=2"),
        ("max_package_depth", 2, "max_package_depth=2"),
    ],
)
def test_host_skill_resource_topology_is_aggregate_bounded(
    tmp_path: Path,
    limit_field: str,
    limit_value: int,
    expected_error: str,
) -> None:
    package = _write_minimal_skill(
        tmp_path,
        f"host-{limit_field}",
        description="Reject excessive directory topology.",
    )
    (package / "references" / "one" / "two").mkdir(parents=True)
    overrides: dict[str, Any] = {
        "max_package_directories": 10,
        "max_package_depth": 10,
        limit_field: limit_value,
    }
    config = AgentLibOSConfig(skills=replace(SkillDefaults(), **overrides))
    runtime = Runtime.open(tmp_path / f"{limit_field}.sqlite", config=config)
    try:
        with pytest.raises(ValidationError, match=expected_error):
            runtime.skills.validate_package_path(package)
    finally:
        runtime.close()


def test_workspace_skill_resource_topology_is_aggregate_bounded(
    tmp_path: Path,
) -> None:
    package = _write_minimal_skill(
        tmp_path,
        "workspace-topology-bound",
        description="Reject excessive workspace directory topology.",
    )
    (package / "references" / "one" / "two").mkdir(parents=True)
    config = AgentLibOSConfig(
        skills=replace(
            SkillDefaults(),
            max_package_directories=2,
            max_package_depth=10,
        )
    )
    runtime = Runtime.open(
        tmp_path / "workspace-topology.sqlite",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject excessive workspace Skill topology",
        )
        runtime.filesystem.grant_directory(
            pid,
            package.name,
            [CapabilityRight.READ],
            issued_by="test.host",
        )

        with pytest.raises(ValidationError, match="max_package_directories=2"):
            runtime.skills.register_skill_from_workspace_path(
                pid,
                package.name,
                require_capability=False,
            )
        assert runtime.store.get_skill(package.name) is None
    finally:
        runtime.close()


def test_workspace_registration_requires_write_before_descendant_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _write_minimal_skill(
        tmp_path,
        "workspace-preauthorized-traversal",
        description="Require Skill WRITE before reading package descendants.",
    )
    (package / "references" / "one" / "two").mkdir(parents=True)
    runtime = Runtime.open(
        tmp_path / "workspace-preauthorized-traversal.sqlite",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="deny package traversal before Skill WRITE",
        )
        runtime.filesystem.grant_directory(
            pid,
            package.name,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        runtime.skills.human = None
        original_read_bytes = runtime.filesystem.read_bytes
        original_read_directory = runtime.filesystem.read_directory
        file_reads: list[str] = []
        directory_reads: list[str] = []

        def record_read_bytes(*args: Any, **kwargs: Any):
            file_reads.append(str(args[1]))
            return original_read_bytes(*args, **kwargs)

        def record_read_directory(*args: Any, **kwargs: Any):
            directory_reads.append(str(args[1]))
            return original_read_directory(*args, **kwargs)

        monkeypatch.setattr(runtime.filesystem, "read_bytes", record_read_bytes)
        monkeypatch.setattr(
            runtime.filesystem,
            "read_directory",
            record_read_directory,
        )

        with pytest.raises(CapabilityDenied):
            runtime.skills.register_skill_from_workspace_path(pid, package.name)
        assert file_reads == [f"{package.name}/SKILL.md"]
        assert directory_reads == []
        assert runtime.store.get_skill(package.name) is None
    finally:
        runtime.close()


def test_workspace_topology_failure_restores_finite_write_authority(
    tmp_path: Path,
) -> None:
    package = _write_minimal_skill(
        tmp_path,
        "workspace-topology-rollback",
        description="Restore finite WRITE after a bounded topology rejection.",
    )
    (package / "references" / "one" / "two").mkdir(parents=True)
    runtime = Runtime.open(
        tmp_path / "workspace-topology-rollback.sqlite",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=AgentLibOSConfig(
            skills=replace(
                SkillDefaults(),
                max_package_directories=2,
                max_package_depth=10,
            )
        ),
    )
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="restore WRITE after invalid package traversal",
        )
        runtime.filesystem.grant_directory(
            pid,
            package.name,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        authority = runtime.capability.grant_once(
            pid,
            runtime.skills.resource_for(package.name),
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        with pytest.raises(ValidationError, match="max_package_directories=2"):
            runtime.skills.register_skill_from_workspace_path(pid, package.name)
        persisted = runtime.store.get_capability(authority.cap_id)
        assert persisted is not None
        assert persisted.active is True
        assert persisted.uses_remaining == 1
        assert runtime.store.get_skill(package.name) is None
    finally:
        runtime.close()


def test_unicode_skill_search_is_source_independent(tmp_path: Path) -> None:
    query = "strasse navigation"
    description = "Straße navigation for multilingual workspaces."

    registered_runtime = Runtime.open(tmp_path / "registered-unicode.sqlite")
    try:
        registered_runtime.skills.register_skill_package(
            SkillPackage(
                skill_id="registered-unicode-search",
                name="registered-unicode-search",
                description=description,
                instructions="Use the multilingual navigation workflow.",
            ),
            actor="test.host",
            require_capability=False,
        )
        registered = registered_runtime.skills.discover_skills(
            query,
            actor="test.host",
            require_capability=False,
        )
        assert [item["skill_id"] for item in registered] == [
            "registered-unicode-search"
        ]
    finally:
        registered_runtime.close()

    catalog = tmp_path / "unicode-catalog"
    _write_minimal_skill(
        catalog,
        "host-unicode-search",
        description=description,
    )
    host_runtime = Runtime.open(
        tmp_path / "host-unicode.sqlite",
        config=AgentLibOSConfig(
            skills=replace(
                SkillDefaults(),
                workspace_dirs=(str(catalog),),
                global_dirs=(),
            )
        ),
    )
    try:
        host = host_runtime.skills.discover_skills(
            query,
            require_capability=False,
        )
        assert [item["skill_id"] for item in host] == ["host-unicode-search"]
    finally:
        host_runtime.close()
