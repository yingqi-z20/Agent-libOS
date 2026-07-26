from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight, EventType
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate


def _checkpoint_fixture(runtime: Runtime) -> tuple[str, str, str]:
    source = runtime.process.spawn(
        image="base-agent:v0",
        goal="checkpoint reference source",
    )
    attacker = runtime.process.spawn(
        image="base-agent:v0",
        goal="checkpoint reference caller",
    )
    checkpoint_id = runtime.checkpoint.create(
        source,
        "reference boundary",
        actor=source,
    )
    return source, attacker, checkpoint_id


@pytest.mark.parametrize(
    "operation",
    ["inspect", "diff", "restore", "fork", "replay"],
)
def test_checkpoint_operations_hide_existing_ids_from_unauthorized_callers(
    operation: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        _source, attacker, checkpoint_id = _checkpoint_fixture(runtime)
        creation_event = next(
            event
            for event in runtime.events.list()
            if event.type == EventType.CHECKPOINT_CREATED
            and event.payload.get("checkpoint_id") == checkpoint_id
        )

        def invoke(selected_id: str) -> object:
            if operation == "inspect":
                return runtime.checkpoint.inspect(selected_id, actor=attacker)
            if operation == "diff":
                return runtime.checkpoint.diff(selected_id, actor=attacker)
            if operation == "restore":
                return runtime.checkpoint.restore(attacker, selected_id)
            if operation == "fork":
                return runtime.checkpoint.fork_from_checkpoint(
                    attacker,
                    selected_id,
                )
            return runtime.checkpoint.replay_to_event(
                selected_id,
                creation_event.event_id,
                actor=attacker,
            )

        observed: list[type[BaseException]] = []
        for selected_id in (checkpoint_id, "ckpt_missing_reference"):
            with pytest.raises(CapabilityDenied) as denied:
                invoke(selected_id)
            observed.append(type(denied.value))

        assert observed == [CapabilityDenied, CapabilityDenied]
    finally:
        runtime.close()


def test_authorized_checkpoint_reference_controls_still_allow_operations() -> None:
    runtime = Runtime.open("local")
    try:
        _source, actor, checkpoint_id = _checkpoint_fixture(runtime)
        creation_event = next(
            event
            for event in runtime.events.list()
            if event.type == EventType.CHECKPOINT_CREATED
            and event.payload.get("checkpoint_id") == checkpoint_id
        )
        runtime.capability.grant(
            actor,
            runtime.checkpoint.checkpoint_resource(checkpoint_id),
            [CapabilityRight.READ, CapabilityRight.ADMIN, CapabilityRight.EXECUTE],
            issued_by="test",
        )

        assert runtime.checkpoint.inspect(checkpoint_id, actor=actor)[
            "checkpoint"
        ]["checkpoint_id"] == checkpoint_id
        assert runtime.checkpoint.diff(checkpoint_id, actor=actor)[
            "checkpoint_id"
        ] == checkpoint_id
        assert runtime.checkpoint.replay_to_event(
            checkpoint_id,
            creation_event.event_id,
            actor=actor,
        )["event_id"] == creation_event.event_id
        runtime.checkpoint.preflight_checkpoint_restore(
            actor,
            checkpoint_id,
        )
        runtime.checkpoint.preflight_checkpoint_fork(
            actor,
            checkpoint_id,
        )
    finally:
        runtime.close()


def test_checkpoint_read_keeps_process_owner_authority_fallback() -> None:
    runtime = Runtime.open("local")
    try:
        source, actor, checkpoint_id = _checkpoint_fixture(runtime)
        runtime.capability.grant(
            actor,
            runtime.checkpoint.process_resource(source),
            [CapabilityRight.READ],
            issued_by="test",
        )

        inspected = runtime.checkpoint.inspect(checkpoint_id, actor=actor)

        assert inspected["checkpoint"]["pid"] == source
    finally:
        runtime.close()


def test_checkpoint_image_commit_hides_source_id_after_image_authority() -> None:
    runtime = Runtime.open("local")
    try:
        _source, actor, checkpoint_id = _checkpoint_fixture(runtime)
        runtime.image_registry.grant_register(actor, issued_by="test")

        for selected_id, image_id in (
            (checkpoint_id, "hidden-existing:v0"),
            ("ckpt_missing_reference", "hidden-missing:v0"),
        ):
            with pytest.raises(CapabilityDenied):
                runtime.image_registry.commit_from_checkpoint(
                    actor=actor,
                    checkpoint_id=selected_id,
                    image_id=image_id,
                    name=image_id,
                )
            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None

        runtime.capability.grant(
            actor,
            runtime.checkpoint.checkpoint_resource(checkpoint_id),
            [CapabilityRight.READ],
            issued_by="test",
        )
        committed = runtime.image_registry.commit_from_checkpoint(
            actor=actor,
            checkpoint_id=checkpoint_id,
            image_id="authorized-commit:v0",
            name="authorized-commit",
        )
        assert committed.image.image_id == "authorized-commit:v0"
    finally:
        runtime.close()


def test_process_exec_hides_existing_images_until_read_authorized() -> None:
    runtime = Runtime.open("local")
    try:
        target = AgentImage(
            image_id="exec-reference-target:v0",
            name="exec-reference-target",
        )
        runtime.image_registry.register(target, actor="test")
        actor = runtime.process.spawn(
            image="base-agent:v0",
            goal="exec reference caller",
        )

        for image_id in (target.image_id, "exec-reference-missing:v0"):
            with pytest.raises(CapabilityDenied):
                runtime.exec_process(actor, image_id)
            assert runtime.process.get(actor).image_id == "base-agent:v0"

        runtime.capability.grant(
            actor,
            runtime.image_registry.resource_for(target.image_id),
            [CapabilityRight.READ],
            issued_by="test",
        )
        transitioned = runtime.exec_process(actor, target.image_id)
        assert transitioned.image_id == target.image_id
    finally:
        runtime.close()


def test_process_exec_spends_one_shot_image_read_only_after_boot_preflight() -> None:
    runtime = Runtime.open("local")
    try:
        actor = runtime.process.spawn(
            image="base-agent:v0",
            goal="finite exec caller",
        )
        missing_id = "finite-exec-missing:v0"
        missing_cap = runtime.capability.grant_once(
            actor,
            runtime.image_registry.resource_for(missing_id),
            [CapabilityRight.READ],
            issued_by="test",
        )

        with pytest.raises(NotFound):
            runtime.exec_process(actor, missing_id)
        assert runtime.store.get_capability(
            missing_cap.cap_id
        ).uses_remaining == 1

        package = _register_package(runtime)
        package_cap = runtime.capability.grant_once(
            actor,
            runtime.image_registry.resource_for(package.image_id),
            [CapabilityRight.READ],
            issued_by="test",
        )
        artifact_id = str(package.boot["artifact_id"])
        found = runtime.store.get_image_artifact(artifact_id)
        assert found is not None
        artifact, _metadata = found
        runtime.store.conn.execute(
            "UPDATE image_artifacts SET artifact_json = ? WHERE artifact_id = ?",
            (
                json.dumps({**artifact, "tampered": True}, sort_keys=True),
                artifact_id,
            ),
        )
        runtime.store.conn.commit()

        with pytest.raises(RuntimeError, match="content hash mismatch"):
            runtime.exec_process(actor, package.image_id)
        assert runtime.store.get_capability(
            package_cap.cap_id
        ).uses_remaining == 1

        valid = AgentImage(
            image_id="finite-exec-valid:v0",
            name="finite-exec-valid",
        )
        runtime.image_registry.register(valid, actor="test")
        valid_cap = runtime.capability.grant_once(
            actor,
            runtime.image_registry.resource_for(valid.image_id),
            [CapabilityRight.READ],
            issued_by="test",
        )

        assert runtime.exec_process(actor, valid.image_id).image_id == valid.image_id
        assert runtime.store.get_capability(valid_cap.cap_id).uses_remaining == 0
    finally:
        runtime.close()


def _register_package(runtime: Runtime) -> AgentImage:
    return runtime.image_registry.register_from_package_files(
        {
            "IMAGE.yaml": (
                "image_id: identity-package:v0\n"
                "name: identity-package\n"
                "prompt: prompt.md\n"
            ),
            "prompt.md": "Bound artifact identity.\n",
        },
        actor="test",
    ).image


def _corrupt_persisted_image_artifact_reference(
    runtime: Runtime,
    image: AgentImage,
) -> dict[str, object]:
    manifest = asdict(image)
    manifest["boot"] = {
        "kind": "image_package",
        "artifact_id": "imgpkg_missing_reference",
        "artifact_sha256": "0" * 64,
    }
    runtime.store.conn.execute(
        "UPDATE images SET manifest_json = ? WHERE image_id = ?",
        (json.dumps(manifest, sort_keys=True), image.image_id),
    )
    runtime.store.conn.commit()
    return manifest


@pytest.mark.parametrize("failure", ["missing", "kind", "digest"])
def test_generic_registration_rejects_unbound_boot_artifacts(
    failure: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        package = _register_package(runtime)
        boot = dict(package.boot)
        if failure == "missing":
            boot["artifact_id"] = "imgpkg_missing_reference"
        elif failure == "kind":
            boot["kind"] = "checkpoint_commit"
        else:
            boot["artifact_sha256"] = "0" * 64
        image_id = f"invalid-artifact-{failure}:v0"

        with pytest.raises(ValidationError, match="artifact reference is invalid"):
            runtime.image_registry.register(
                replace(
                    package,
                    image_id=image_id,
                    name=image_id,
                    boot=boot,
                ),
                actor="test",
            )

        assert image_id not in runtime.images
        assert runtime.store.get_image(image_id) is None
    finally:
        runtime.close()


def test_persisted_image_load_quarantines_unbound_boot_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "invalid-boot.sqlite"
    runtime = Runtime.open(target)
    try:
        package = _register_package(runtime)
        invalid_reference = AgentImage(
            image_id="quarantined-image:v0",
            name="quarantined-image",
        )
        malformed_digest = AgentImage(
            image_id="malformed-boot-image:v0",
            name="malformed-boot-image",
        )
        runtime.image_registry.register(invalid_reference, actor="test")
        runtime.image_registry.register(malformed_digest, actor="test")
        invalid_manifest = asdict(invalid_reference)
        invalid_manifest["boot"] = {
            "kind": "image_package",
            "artifact_id": "imgpkg_missing_reference",
            "artifact_sha256": "0" * 64,
        }
        malformed_manifest = asdict(malformed_digest)
        malformed_manifest["boot"] = {
            "kind": "image_package",
            "artifact_id": package.boot["artifact_id"],
            "artifact_sha256": "wrong",
        }
        for image, manifest in (
            (invalid_reference, invalid_manifest),
            (malformed_digest, malformed_manifest),
        ):
            runtime.store.conn.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (json.dumps(manifest, sort_keys=True), image.image_id),
            )
        runtime.store.conn.commit()
    finally:
        runtime.close()

    reopened = Runtime.open(target)
    try:
        assert package.image_id in reopened.images
        assert invalid_reference.image_id not in reopened.images
        assert malformed_digest.image_id not in reopened.images
        listed = {
            item["image_id"]: item
            for item in reopened.image_registry.list_images()
        }
        for image in (invalid_reference, malformed_digest):
            assert listed[image.image_id]["boot_status"] == "invalid"
            assert listed[image.image_id]["invalid_boot"] == {
                "code": "image_boot_artifact_invalid",
                "message": "Image boot artifact failed identity validation.",
            }
            inspected = reopened.image_registry.inspect(image.image_id)
            assert inspected["boot_status"] == "invalid"
            assert inspected["artifact"] is None
            with pytest.raises(NotFound):
                reopened.process.spawn(
                    image=image.image_id,
                    goal="quarantined image must not execute",
                )
    finally:
        reopened.close()


def test_quarantined_durable_image_requires_explicit_admin_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "quarantined-replacement.sqlite"
    image_id = "quarantined-replacement:v0"
    runtime = Runtime.open(target)
    try:
        original = AgentImage(image_id=image_id, name="quarantined-original")
        runtime.image_registry.register(original, actor="test")
        invalid_manifest = _corrupt_persisted_image_artifact_reference(
            runtime,
            original,
        )
    finally:
        runtime.close()

    reopened = Runtime.open(target)
    try:
        assert image_id not in reopened.images
        actor = reopened.process.spawn(
            image="base-agent:v0",
            goal="replace quarantined image",
        )
        write_cap = reopened.capability.grant_once(
            actor,
            reopened.image_registry.resource_for(image_id),
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        package_files = {
            "IMAGE.yaml": (
                f"image_id: {image_id}\n"
                "name: quarantined-replacement\n"
                "prompt: prompt.md\n"
            ),
            "prompt.md": "Replacement package.\n",
        }

        with pytest.raises(ValidationError, match="already exists"):
            reopened.image_registry.register(
                AgentImage(image_id=image_id, name="generic-replacement"),
                actor=actor,
                require_capability=True,
                replace=False,
            )

        persisted = reopened.store.get_image(image_id)
        assert persisted is not None
        assert asdict(persisted[0]) == invalid_manifest
        assert reopened.store.get_capability(write_cap.cap_id).uses_remaining == 1

        with pytest.raises(ValidationError, match="already exists"):
            reopened.image_registry.register_from_package_files(
                package_files,
                actor=actor,
                require_capability=True,
                replace=False,
            )

        persisted = reopened.store.get_image(image_id)
        assert persisted is not None
        assert asdict(persisted[0]) == invalid_manifest
        assert reopened.store.get_capability(write_cap.cap_id).uses_remaining == 1

        checkpoint_id = reopened.checkpoint.create(
            actor,
            "commit must not replace quarantined image",
            actor=actor,
        )
        with pytest.raises(ValidationError, match="already exists"):
            reopened.image_registry.commit_from_checkpoint(
                actor=actor,
                checkpoint_id=checkpoint_id,
                image_id=image_id,
                name="quarantined-checkpoint-replacement",
                replace=False,
            )
        assert reopened.store.get_capability(write_cap.cap_id).uses_remaining == 1

        with pytest.raises(CapabilityDenied):
            reopened.image_registry.register_from_package_files(
                package_files,
                actor=actor,
                require_capability=True,
                replace=True,
            )

        reopened.capability.grant(
            actor,
            reopened.image_registry.resource_for(image_id),
            [CapabilityRight.ADMIN],
            issued_by="test",
        )
        result = reopened.image_registry.register_from_package_files(
            package_files,
            actor=actor,
            require_capability=True,
            replace=True,
        )
        assert result.replaced is True
        assert reopened.store.get_image(image_id) is not None
        assert image_id in reopened.images
    finally:
        reopened.close()


def test_checkpoint_fork_and_restore_treat_quarantined_image_as_existing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "quarantined-checkpoint-image.sqlite"
    image_id = "quarantined-checkpoint-image:v0"
    runtime = Runtime.open(target)
    try:
        image = AgentImage(image_id=image_id, name="checkpoint-image")
        runtime.image_registry.register(image, actor="test")
        source = runtime.process.spawn(image=image_id, goal="capture image")
        checkpoint_id = runtime.checkpoint.create(
            source,
            "quarantined image authority boundary",
            actor=source,
        )
        invalid_manifest = _corrupt_persisted_image_artifact_reference(
            runtime,
            image,
        )
    finally:
        runtime.close()

    reopened = Runtime.open(target)
    try:
        assert image_id not in reopened.images
        actor = reopened.process.spawn(
            image="base-agent:v0",
            goal="checkpoint image authority caller",
        )
        reopened.capability.grant(
            actor,
            reopened.checkpoint.checkpoint_resource(checkpoint_id),
            [CapabilityRight.EXECUTE, CapabilityRight.ADMIN],
            issued_by="test",
        )
        image_write = reopened.capability.grant_once(
            actor,
            reopened.image_registry.resource_for(image_id),
            [CapabilityRight.WRITE],
            issued_by="test",
        )

        with pytest.raises(ValidationError, match="quarantined durable"):
            reopened.checkpoint.fork_from_checkpoint(actor, checkpoint_id)
        assert reopened.store.get_capability(image_write.cap_id).uses_remaining == 1

        with pytest.raises(CapabilityDenied):
            reopened.checkpoint.restore(actor, checkpoint_id)

        persisted = reopened.store.get_image(image_id)
        assert persisted is not None
        assert asdict(persisted[0]) == invalid_manifest
    finally:
        reopened.close()


def test_workspace_package_global_path_budget_counts_directories(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    package.joinpath("IMAGE.yaml").write_text(
        "image_id: bounded-workspace:v0\n"
        "name: bounded-workspace\n"
        "prompt: prompt.md\n",
        encoding="utf-8",
    )
    package.joinpath("prompt.md").write_text("Bounded.\n", encoding="utf-8")
    package.joinpath("nested", "deeper").mkdir(parents=True)
    config = replace(
        DEFAULT_CONFIG,
        image=replace(DEFAULT_CONFIG.image, package_max_files=3),
    )
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=config,
    )
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="validate bounded workspace package",
        )
        runtime.filesystem.grant_directory(
            pid,
            "package",
            [CapabilityRight.READ],
            issued_by="test",
        )

        with pytest.raises(
            ValidationError,
            match="exceeds package_max_files=3",
        ):
            runtime.image_registry.validate_workspace_package(pid, "package")
    finally:
        runtime.close()
