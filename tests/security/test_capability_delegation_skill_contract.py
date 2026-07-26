from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityEffect, CapabilityRight, EventType
from agent_libos.models.exceptions import ValidationError


def _delegation_runtime(*, result_limit_bytes: int | None = None) -> tuple[Runtime, str, str]:
    config = DEFAULT_CONFIG
    if result_limit_bytes is not None:
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(
                DEFAULT_CONFIG.tools,
                tool_result_payload_hard_limit_bytes=result_limit_bytes,
            ),
        )
    runtime = Runtime.open("local", config=config)
    parent = runtime.process.spawn(goal="capability delegation parent")
    child = runtime.process.spawn_child(
        parent,
        goal="capability delegation child",
    )
    runtime.capability.issue_trusted(
        parent,
        "object:delegation-contract",
        [CapabilityRight.READ],
        issued_by="test.host",
        delegable=True,
        max_delegation_depth=4,
    )
    runtime.tools.configure_process_tools(
        parent,
        ["delegate_capability", "revoke_capability"],
        assigned_by="test",
    )
    return runtime, parent, child


def _child_grant_events(runtime: Runtime, child: str) -> list[str]:
    return [
        event.event_id
        for event in runtime.events.list(target=child)
        if event.type == EventType.CAPABILITY_GRANTED
    ]


def test_mutation_tools_return_safe_success_receipts_when_full_identity_cannot_fit() -> None:
    runtime, parent, child = _delegation_runtime(result_limit_bytes=640)
    try:
        delegated = runtime.tools.call(
            parent,
            "delegate_capability",
            {
                "child_pid": child,
                "resource": "object:delegation-contract",
                "rights": [CapabilityRight.READ.value],
            },
        )

        assert delegated.ok is True
        delegated_receipt = delegated.payload["capability"]
        assert delegated_receipt["presentation_omitted"] is True
        assert delegated_receipt["status"] == "active"
        cap_id = delegated_receipt["cap_id"]
        assert runtime.capability.inspect(cap_id)["status"] == "active"

        revoked = runtime.tools.call(
            parent,
            "revoke_capability",
            {"cap_id": cap_id, "reason": "delegation contract complete"},
        )

        assert revoked.ok is True
        revoked_receipt = revoked.payload["capability"]
        assert revoked_receipt == {
            "cap_id": cap_id,
            "status": "revoked",
            "presentation_omitted": True,
        }
        assert runtime.capability.inspect(cap_id)["status"] == "revoked"
    finally:
        runtime.close()


def test_mutation_tools_keep_success_when_the_entire_result_is_omitted() -> None:
    runtime, parent, child = _delegation_runtime(result_limit_bytes=256)
    try:
        before_capability_ids = set(runtime.process.get(child).capabilities)
        delegated = runtime.tools.call(
            parent,
            "delegate_capability",
            {
                "child_pid": child,
                "resource": "object:delegation-contract",
                "rights": [CapabilityRight.READ.value],
            },
        )

        assert delegated.ok is True
        assert delegated.payload["result_omitted"] is True
        created_ids = (
            set(runtime.process.get(child).capabilities) - before_capability_ids
        )
        assert len(created_ids) == 1
        cap_id = created_ids.pop()

        revoked = runtime.tools.call(
            parent,
            "revoke_capability",
            {"cap_id": cap_id, "reason": "omitted-result contract complete"},
        )

        assert revoked.ok is True
        assert revoked.payload["result_omitted"] is True
        assert runtime.capability.inspect(cap_id)["status"] == "revoked"
    finally:
        runtime.close()


@pytest.mark.parametrize("effect", [CapabilityEffect.ASK, CapabilityEffect.DENY])
def test_delegation_rejects_finite_restrictive_effect_before_side_effects(
    effect: CapabilityEffect,
) -> None:
    runtime, parent, child = _delegation_runtime()
    try:
        with pytest.raises(
            ValidationError,
            match="uses_remaining is supported only for allow capabilities",
        ):
            runtime.capability.validate_delegation(
                parent,
                {
                    "resource": "object:delegation-contract",
                    "rights": [CapabilityRight.READ.value],
                    "effect": effect.value,
                    "uses_remaining": 1,
                },
            )
        before_capability_ids = set(runtime.process.get(child).capabilities)
        before_event_ids = _child_grant_events(runtime, child)

        result = runtime.tools.call(
            parent,
            "delegate_capability",
            {
                "child_pid": child,
                "resource": "object:delegation-contract",
                "rights": [CapabilityRight.READ.value],
                "effect": effect.value,
                "uses_remaining": 1,
            },
        )

        assert result.ok is False
        assert (result.error or "").startswith("validation_error: ValidationError")
        assert result.payload["error"]["code"] == "validation_error"
        assert set(runtime.process.get(child).capabilities) == before_capability_ids
        assert _child_grant_events(runtime, child) == before_event_ids
        assert all(
            record.action != "capability.delegate"
            for record in runtime.audit.trace()
            if record.target.startswith(f"{parent}->{child}:")
        )
    finally:
        runtime.close()


def test_delegation_rejects_past_expiry_before_side_effects() -> None:
    runtime, parent, child = _delegation_runtime()
    try:
        with pytest.raises(
            ValidationError,
            match="delegated capability expiry must be in the future",
        ):
            runtime.capability.validate_delegation(
                parent,
                {
                    "resource": "object:delegation-contract",
                    "rights": [CapabilityRight.READ.value],
                    "expires_at": "2000-01-01T00:00:00Z",
                },
            )
        before_capability_ids = set(runtime.process.get(child).capabilities)
        before_event_ids = _child_grant_events(runtime, child)

        result = runtime.tools.call(
            parent,
            "delegate_capability",
            {
                "child_pid": child,
                "resource": "object:delegation-contract",
                "rights": [CapabilityRight.READ.value],
                "expires_at": "2000-01-01T00:00:00Z",
            },
        )

        assert result.ok is False
        assert (result.error or "").startswith("validation_error: ValidationError")
        assert result.payload["error"]["code"] == "validation_error"
        assert set(runtime.process.get(child).capabilities) == before_capability_ids
        assert _child_grant_events(runtime, child) == before_event_ids
        assert all(
            record.action != "capability.delegate"
            for record in runtime.audit.trace()
            if record.target.startswith(f"{parent}->{child}:")
        )
    finally:
        runtime.close()
