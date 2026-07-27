from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    AgentImage,
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    CapabilityStatus,
    EventType,
)
from agent_libos.models.exceptions import CapabilityDenied


def _reservation_rows(runtime: Runtime) -> list[dict[str, object]]:
    return runtime.store.select_table_rows(
        "capability_use_reservations",
        order_by="reservation_id",
    )


@pytest.mark.parametrize("sink", ["event", "audit"])
def test_checkpoint_create_failure_rolls_back_one_time_authority_and_publication(
    monkeypatch: pytest.MonkeyPatch,
    sink: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        owner = runtime.process.spawn(goal=f"checkpoint create {sink} owner")
        controller = runtime.process.spawn(goal=f"checkpoint create {sink} controller")
        authority = runtime.capability.grant_once(
            controller,
            f"checkpoint:process:{owner}",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        before_process = runtime.process.get(owner)
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        before_reservations = _reservation_rows(runtime)

        if sink == "event":
            original_emit = runtime.events.emit

            def fail_after_create_event(event_type, *args, **kwargs):
                result = original_emit(event_type, *args, **kwargs)
                if event_type == EventType.CHECKPOINT_CREATED:
                    raise RuntimeError("injected checkpoint create event failure")
                return result

            monkeypatch.setattr(runtime.events, "emit", fail_after_create_event)
        else:
            original_record = runtime.audit.record

            def fail_after_create_audit(*args, **kwargs):
                result = original_record(*args, **kwargs)
                if kwargs.get("action") == "checkpoint.create":
                    raise RuntimeError("injected checkpoint create audit failure")
                return result

            monkeypatch.setattr(runtime.audit, "record", fail_after_create_audit)

        with pytest.raises(RuntimeError, match=f"create {sink} failure"):
            runtime.checkpoint.create(
                owner,
                "atomic create",
                actor=controller,
            )

        persisted = runtime.store.get_capability(authority.cap_id)
        assert persisted is not None
        assert persisted.status == CapabilityStatus.ACTIVE
        assert persisted.uses_remaining == 1
        assert _reservation_rows(runtime) == before_reservations
        assert runtime.store.list_checkpoints(pid=owner) == []
        assert runtime.process.get(owner).checkpoint_head == before_process.checkpoint_head
        assert {event.event_id for event in runtime.events.list()} == before_event_ids
        new_audits = [
            record
            for record in runtime.audit.trace()
            if record.record_id not in before_audit_ids
        ]
        # The advisory authorization decision is append-only evidence and may
        # remain, but publication and finite-use settlement evidence must roll
        # back with the failed checkpoint.
        assert all(
            record.action
            not in {
                "checkpoint.create",
                "capability.reserve_use",
                "capability.commit_reserved_use",
            }
            for record in new_audits
        )
    finally:
        runtime.close()


def test_checkpoint_create_commits_one_time_authority_with_publication() -> None:
    runtime = Runtime.open("local")
    try:
        owner = runtime.process.spawn(goal="checkpoint create owner")
        controller = runtime.process.spawn(goal="checkpoint create controller")
        authority = runtime.capability.grant_once(
            controller,
            f"checkpoint:process:{owner}",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        checkpoint_id = runtime.checkpoint.create(
            owner,
            "atomic create success",
            actor=controller,
        )

        persisted = runtime.store.get_capability(authority.cap_id)
        assert persisted is not None
        assert persisted.uses_remaining == 0
        reservations = [
            row
            for row in _reservation_rows(runtime)
            if row["cap_id"] == authority.cap_id
        ]
        assert [row["status"] for row in reservations] == ["committed"]
        assert runtime.process.get(owner).checkpoint_head == checkpoint_id
        assert [
            checkpoint.checkpoint_id
            for checkpoint in runtime.store.list_checkpoints(pid=owner)
        ] == [checkpoint_id]
    finally:
        runtime.close()


def test_object_handle_capability_preserves_publication_ownership_metadata() -> None:
    runtime = Runtime.open("local")
    try:
        subject = runtime.process.spawn(goal="publication-owned object handle")
        handle = runtime.capability.handle_for_object(
            subject,
            "obj_publication_owned",
            [CapabilityRight.READ],
            issued_by="checkpoint.image",
            metadata={
                "runtime_publication_id": "pub_exact_owner",
                "object_handle": False,
            },
        )

        cap = runtime.store.get_capability(handle.capability_id)
        assert cap is not None
        assert cap.metadata["runtime_publication_id"] == "pub_exact_owner"
        assert cap.metadata["object_handle"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("policy_change", ["deny", "revoke"])
def test_grant_transfer_recomputes_requested_rights_inside_authority_transaction(
    monkeypatch: pytest.MonkeyPatch,
    policy_change: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        actor = runtime.process.spawn(goal=f"grant transfer {policy_change} actor")
        child = runtime.process.spawn(goal=f"grant transfer {policy_change} child")
        resource = f"object:grant-transfer-{policy_change}"
        parent = runtime.capability.issue_trusted(
            actor,
            resource,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        grant_once = runtime.capability.grant_once(
            actor,
            resource,
            [CapabilityRight.GRANT],
            issued_by="test.host",
        )
        barrier = Barrier(2)
        original_require = runtime.capability._require_issue_authority

        def pause_after_preflight(who: str, spec: CapabilitySpec):
            decision = original_require(who, spec)
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
            return decision

        monkeypatch.setattr(
            runtime.capability,
            "_require_issue_authority",
            pause_after_preflight,
        )
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        before_reservations = _reservation_rows(runtime)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runtime.capability.issue,
                actor,
                child,
                CapabilitySpec(resource=resource, rights={CapabilityRight.READ.value}),
            )
            barrier.wait(timeout=5)
            if policy_change == "deny":
                runtime.capability.issue_trusted(
                    actor,
                    resource,
                    [CapabilityRight.READ],
                    issued_by="test.defender",
                    effect=CapabilityEffect.DENY,
                )
            else:
                runtime.capability.revoke(
                    parent.cap_id,
                    revoked_by="test.defender",
                    require_authority=False,
                )
            barrier.wait(timeout=5)
            with pytest.raises(CapabilityDenied):
                future.result(timeout=5)

        latest_grant = runtime.store.get_capability(grant_once.cap_id)
        assert latest_grant is not None
        assert latest_grant.status == CapabilityStatus.ACTIVE
        assert latest_grant.uses_remaining == 1
        assert _reservation_rows(runtime) == before_reservations
        assert not runtime.capability.check(child, resource, CapabilityRight.READ)
        assert not [
            cap
            for cap in runtime.capability.capabilities_for(child)
            if cap.resource == resource
        ]
        assert not [
            event
            for event in runtime.events.list()
            if event.event_id not in before_event_ids
            and event.type == EventType.CAPABILITY_GRANTED
            and event.target == child
            and event.payload.get("resource") == resource
        ]
        assert not [
            record
            for record in runtime.audit.trace()
            if record.record_id not in before_audit_ids
            and record.action == "capability.issue"
            and record.target == f"{child}:{resource}"
        ]
    finally:
        runtime.close()


def test_grant_transfer_refreshes_authority_chain_and_target_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        actor = runtime.process.spawn(goal="grant replacement actor")
        child = runtime.process.spawn(goal="grant replacement child")
        resource = "object:grant-replacement"
        old_parent = runtime.capability.issue_trusted(
            actor,
            resource,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        old_grant = runtime.capability.grant_once(
            actor,
            resource,
            [CapabilityRight.GRANT],
            issued_by="test.host",
        )
        barrier = Barrier(2)
        original_require = runtime.capability._require_issue_authority

        def pause_after_preflight(who: str, spec: CapabilitySpec):
            decision = original_require(who, spec)
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
            return decision

        monkeypatch.setattr(
            runtime.capability,
            "_require_issue_authority",
            pause_after_preflight,
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runtime.capability.issue,
                actor,
                child,
                CapabilitySpec(resource=resource, rights={CapabilityRight.READ.value}),
            )
            barrier.wait(timeout=5)
            runtime.capability.revoke(
                old_parent.cap_id,
                revoked_by="test.defender",
                require_authority=False,
            )
            runtime.capability.revoke(
                old_grant.cap_id,
                revoked_by="test.defender",
                require_authority=False,
            )
            new_parent = runtime.capability.issue_trusted(
                actor,
                resource,
                [CapabilityRight.READ],
                issued_by="test.host",
            )
            new_grant = runtime.capability.grant_once(
                actor,
                resource,
                [CapabilityRight.GRANT],
                issued_by="test.host",
            )
            current_child = runtime.process.get(child)
            runtime.store.patch_process(
                child,
                {"status_message": "concurrent target revision"},
                expected_revision=current_child.revision,
            )
            barrier.wait(timeout=5)
            issued = future.result(timeout=5)

        assert issued.issuer_cap_id == new_grant.cap_id
        assert issued.parent_cap_id == new_parent.cap_id
        assert issued.issuer_cap_id != old_grant.cap_id
        assert issued.parent_cap_id != old_parent.cap_id
        assert runtime.process.get(child).status_message == "concurrent target revision"
        assert runtime.capability.check(child, resource, CapabilityRight.READ)
        assert not [
            row
            for row in _reservation_rows(runtime)
            if row["status"] == "reserved"
        ]
    finally:
        runtime.close()


def test_one_shot_authority_has_one_concurrent_winner_without_stranded_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        actor = runtime.process.spawn(goal="one-shot authority actor")
        children = [
            runtime.process.spawn(goal="one-shot authority child one"),
            runtime.process.spawn(goal="one-shot authority child two"),
        ]
        resource = "object:one-shot-authority"
        authority = runtime.capability.grant_once(
            actor,
            resource,
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        barrier = Barrier(2)
        original_require = runtime.capability._require_issue_authority

        def synchronize_preflights(who: str, spec: CapabilitySpec):
            decision = original_require(who, spec)
            barrier.wait(timeout=5)
            return decision

        monkeypatch.setattr(
            runtime.capability,
            "_require_issue_authority",
            synchronize_preflights,
        )

        def issue(child: str):
            try:
                return runtime.capability.issue(
                    actor,
                    child,
                    CapabilitySpec(resource=resource, rights={CapabilityRight.READ.value}),
                )
            except CapabilityDenied as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(issue, children))

        assert sum(not isinstance(outcome, CapabilityDenied) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, CapabilityDenied) for outcome in outcomes) == 1
        latest = runtime.store.get_capability(authority.cap_id)
        assert latest is not None
        assert latest.status == CapabilityStatus.REVOKED
        assert latest.uses_remaining == 0
        reservations = [
            row
            for row in _reservation_rows(runtime)
            if row["cap_id"] == authority.cap_id
        ]
        assert [row["status"] for row in reservations] == ["committed"]
        assert sum(
            runtime.capability.check(child, resource, CapabilityRight.READ)
            for child in children
        ) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("sink", ["event", "audit"])
def test_checkpoint_restore_core_evidence_failure_rolls_back_authority_and_main_state(
    monkeypatch: pytest.MonkeyPatch,
    sink: str,
) -> None:
    runtime = Runtime.open("local")
    image_id = f"checkpoint-authority-atomic-{sink}:v0"
    try:
        runtime.register_image(
            AgentImage(image_id=image_id, name=image_id, system_prompt="captured"),
            actor="test.host",
        )
        owner = runtime.process.spawn(image=image_id, goal="checkpoint authority owner")
        controller = runtime.process.spawn(goal="checkpoint authority controller")
        checkpoint_id = runtime.checkpoint.create(owner, "authority atomicity", actor=owner)
        current_owner = runtime.process.get(owner)
        runtime.store.patch_process(
            owner,
            {"status_message": "current state"},
            expected_revision=current_owner.revision,
        )
        runtime.register_image(
            AgentImage(image_id=image_id, name=image_id, system_prompt="current"),
            actor="test.host",
            replace=True,
        )
        checkpoint_cap = runtime.capability.grant_once(
            controller,
            f"checkpoint:{checkpoint_id}",
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        image_cap = runtime.capability.grant_once(
            controller,
            f"image:{image_id}",
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        before_reservations = _reservation_rows(runtime)

        if sink == "event":
            original_emit = runtime.events.emit

            def fail_after_restore_event(event_type, *args, **kwargs):
                result = original_emit(event_type, *args, **kwargs)
                if event_type == EventType.ROLLBACK:
                    raise RuntimeError("injected restore event failure")
                return result

            monkeypatch.setattr(runtime.events, "emit", fail_after_restore_event)
        else:
            original_record = runtime.audit.record

            def fail_after_restore_audit(*args, **kwargs):
                result = original_record(*args, **kwargs)
                if kwargs.get("action") == "checkpoint.restore":
                    raise RuntimeError("injected restore audit failure")
                return result

            monkeypatch.setattr(runtime.audit, "record", fail_after_restore_audit)

        with pytest.raises(RuntimeError, match=f"restore {sink} failure"):
            runtime.checkpoint.restore(controller, checkpoint_id)

        assert runtime.process.get(owner).status_message == "current state"
        assert runtime.get_image(image_id).system_prompt == "current"
        for cap_id in (checkpoint_cap.cap_id, image_cap.cap_id):
            cap = runtime.store.get_capability(cap_id)
            assert cap is not None
            assert cap.status == CapabilityStatus.ACTIVE
            assert cap.uses_remaining == 1
        assert _reservation_rows(runtime) == before_reservations
        assert not [
            event
            for event in runtime.events.list()
            if event.event_id not in before_event_ids and event.type == EventType.ROLLBACK
        ]
        assert not [
            record
            for record in runtime.audit.trace()
            if record.record_id not in before_audit_ids
            and record.action == "checkpoint.restore"
        ]
    finally:
        runtime.close()


def test_checkpoint_restore_publish_failure_never_uses_fallible_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        owner = runtime.process.spawn(goal="checkpoint publish failure owner")
        controller = runtime.process.spawn(goal="checkpoint publish failure controller")
        checkpoint_id = runtime.checkpoint.create(owner, "before publish failure", actor=owner)
        current_owner = runtime.process.get(owner)
        runtime.store.patch_process(
            owner,
            {"status_message": "current state"},
            expected_revision=current_owner.revision,
        )
        authority = runtime.capability.grant_once(
            controller,
            f"checkpoint:{checkpoint_id}",
            [CapabilityRight.ADMIN],
            issued_by="test.host",
        )
        before_reservations = _reservation_rows(runtime)
        compensation_calls = 0

        def fail_publish(*_args, **_kwargs):
            raise RuntimeError("injected restore publish failure")

        def fail_compensation(*_args, **_kwargs):
            nonlocal compensation_calls
            compensation_calls += 1
            raise RuntimeError("compensation must not run")

        monkeypatch.setattr(runtime.checkpoint, "_publish_restore_rows", fail_publish)
        monkeypatch.setattr(runtime.capability, "restore_reserved_use", fail_compensation)

        with pytest.raises(RuntimeError, match="restore publish failure"):
            runtime.checkpoint.restore(controller, checkpoint_id)

        latest = runtime.store.get_capability(authority.cap_id)
        assert latest is not None
        assert latest.status == CapabilityStatus.ACTIVE
        assert latest.uses_remaining == 1
        assert runtime.process.get(owner).status_message == "current state"
        assert _reservation_rows(runtime) == before_reservations
        assert compensation_calls == 0
    finally:
        runtime.close()


def test_checkpoint_restore_settlement_failure_rolls_back_composite_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    image_id = "checkpoint-authority-settlement:v0"
    try:
        runtime.register_image(
            AgentImage(image_id=image_id, name=image_id, system_prompt="captured"),
            actor="test.host",
        )
        owner = runtime.process.spawn(image=image_id, goal="checkpoint settlement owner")
        controller = runtime.process.spawn(goal="checkpoint settlement controller")
        checkpoint_id = runtime.checkpoint.create(owner, "settlement atomicity", actor=owner)
        current_owner = runtime.process.get(owner)
        runtime.store.patch_process(
            owner,
            {"status_message": "current state"},
            expected_revision=current_owner.revision,
        )
        runtime.register_image(
            AgentImage(image_id=image_id, name=image_id, system_prompt="current"),
            actor="test.host",
            replace=True,
        )
        authorities = [
            runtime.capability.grant_once(
                controller,
                f"checkpoint:{checkpoint_id}",
                [CapabilityRight.ADMIN],
                issued_by="test.host",
            ),
            runtime.capability.grant_once(
                controller,
                f"image:{image_id}",
                [CapabilityRight.ADMIN],
                issued_by="test.host",
            ),
        ]
        before_event_ids = {event.event_id for event in runtime.events.list()}
        before_audit_ids = {record.record_id for record in runtime.audit.trace()}
        before_reservations = _reservation_rows(runtime)
        original_commit = runtime.capability.commit_reserved_use
        commit_calls = 0

        def fail_after_second_settlement(*args, **kwargs):
            nonlocal commit_calls
            result = original_commit(*args, **kwargs)
            commit_calls += 1
            if commit_calls == 2:
                raise RuntimeError("injected second settlement failure")
            return result

        monkeypatch.setattr(
            runtime.capability,
            "commit_reserved_use",
            fail_after_second_settlement,
        )

        with pytest.raises(RuntimeError, match="second settlement failure"):
            runtime.checkpoint.restore(controller, checkpoint_id)

        assert commit_calls == 2
        assert runtime.process.get(owner).status_message == "current state"
        assert runtime.get_image(image_id).system_prompt == "current"
        for authority in authorities:
            latest = runtime.store.get_capability(authority.cap_id)
            assert latest is not None
            assert latest.status == CapabilityStatus.ACTIVE
            assert latest.uses_remaining == 1
        assert _reservation_rows(runtime) == before_reservations
        assert not [
            event
            for event in runtime.events.list()
            if event.event_id not in before_event_ids and event.type == EventType.ROLLBACK
        ]
        assert not [
            record
            for record in runtime.audit.trace()
            if record.record_id not in before_audit_ids
            and record.action == "checkpoint.restore"
        ]
    finally:
        runtime.close()
