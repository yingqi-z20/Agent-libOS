from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_libos import Runtime
import agent_libos.runtime.authority_manifest_manager as authority_manifest_module
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityRight,
    ResourceBudget,
    encode_permitted_effects_policy,
    upcast_permitted_effects_policy,
)
from agent_libos.models.exceptions import CapabilityDenied, ProcessError, ValidationError


def test_image_requirements_are_declared_but_not_granted_by_default() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="manifest required")
        manifest = runtime.authority_manifests.get_for_process(pid)

        assert manifest is not None
        assert manifest.metadata["launch_authority_mode"] == "manifest_required"
        assert manifest.authorized_capabilities == []
        assert manifest.permitted_effects is None
        assert manifest.required_capabilities
        assert not runtime.capability.check(pid, DEFAULT_CONFIG.runtime.default_human_resource, "write")
        assert runtime.authority_manifests.summary_for_process(pid)["missing_required_capabilities"]
    finally:
        runtime.close()


def test_permitted_effects_policy_v2_distinguishes_unrestricted_and_deny_all() -> None:
    assert encode_permitted_effects_policy(None) == {
        "schema_version": 2,
        "effects": None,
    }
    assert encode_permitted_effects_policy([]) == {
        "schema_version": 2,
        "effects": [],
    }
    assert upcast_permitted_effects_policy([]) is None
    assert upcast_permitted_effects_policy(["jsonrpc.*"]) == ["jsonrpc.*"]
    assert upcast_permitted_effects_policy(
        {"schema_version": 2, "effects": []}
    ) == []


def test_explicit_empty_effect_ceiling_denies_all_while_omission_is_unrestricted() -> None:
    runtime = Runtime.open("local")
    try:
        unrestricted = runtime.process.spawn(
            goal="unrestricted effect compatibility",
            authority_manifest={},
        )
        deny_all = runtime.process.spawn(
            goal="deny every provider effect",
            authority_manifest={"permitted_effects": []},
        )

        assert runtime.authority_manifests.get_for_process(unrestricted).permitted_effects is None
        assert runtime.authority_manifests.get_for_process(deny_all).permitted_effects == []
        runtime.authority_manifests.assert_effect(unrestricted, "jsonrpc.call")
        with pytest.raises(CapabilityDenied, match="does not permit effect class"):
            runtime.authority_manifests.assert_effect(deny_all, "jsonrpc.call")
        with pytest.raises(CapabilityDenied, match="does not permit effect class"):
            runtime.authority_manifests.assert_effect(deny_all, "human.write")
    finally:
        runtime.close()


def test_manifest_expiry_denies_unrestricted_effects_but_absent_and_live_manifests_allow() -> None:
    runtime = Runtime.open("local")
    try:
        expired_pid = "pid_expired_unrestricted_manifest"
        runtime.authority_manifests.prepare_launch(
            pid=expired_pid,
            image_id="base-agent:v0",
            goal_ref=None,
            supplied={"expires_at": "2000-01-01T00:00:00Z"},
        )
        live_pid = "pid_live_unrestricted_manifest"
        runtime.authority_manifests.prepare_launch(
            pid=live_pid,
            image_id="base-agent:v0",
            goal_ref=None,
            supplied={"expires_at": "2099-01-01T00:00:00Z"},
        )

        with pytest.raises(CapabilityDenied, match="task authority manifest expired"):
            runtime.authority_manifests.assert_effect(expired_pid, "jsonrpc.call")
        runtime.authority_manifests.assert_effect(live_pid, "jsonrpc.call")
        runtime.authority_manifests.assert_effect(
            "pid_without_authority_manifest",
            "jsonrpc.call",
        )
    finally:
        runtime.close()


def test_manifest_expiry_caps_root_and_human_issued_allow_capabilities() -> None:
    runtime = Runtime.open("local")
    try:
        resource = "filesystem:workspace:expiry-bounded.txt"
        manifest_expiry = "2099-01-01T00:00:00Z"
        pid = runtime.process.spawn(
            goal="expiry-bounded authority",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    },
                    {
                        "resource": "filesystem:workspace:root-bounded.txt",
                        "rights": [CapabilityRight.READ.value],
                        "expires_at": "2199-01-01T00:00:00Z",
                    },
                ],
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": resource,
                            "rights": [CapabilityRight.WRITE.value],
                        }
                    ]
                },
                "expires_at": manifest_expiry,
            },
        )

        request_id = runtime.human.request_permission(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            resource,
            [CapabilityRight.WRITE.value],
            "verify manifest-bounded Human grant",
        )
        runtime.human.approve(
            request_id,
            {"approved": True, "policy": CapabilityManager.ALWAYS_ALLOW},
        )

        capabilities = runtime.capability.capabilities_for(pid)
        root_capability = next(
            cap
            for cap in capabilities
            if cap.resource == "filesystem:workspace:root-bounded.txt"
        )
        human_capability = next(cap for cap in capabilities if cap.resource == resource)
        expected_expiry = "2099-01-01T00:00:00+00:00"
        assert root_capability.expires_at == expected_expiry
        assert human_capability.expires_at == expected_expiry
    finally:
        runtime.close()


@pytest.mark.parametrize("transition", ["spawn_child", "fork"])
def test_child_manifest_expiry_caps_derived_capabilities(transition: str) -> None:
    runtime = Runtime.open("local")
    try:
        resource = "object:child-manifest-expiry"
        parent = runtime.process.spawn(
            goal="parent with a later authority lease",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": resource,
                        "rights": [CapabilityRight.READ.value],
                        "delegable": True,
                    }
                ],
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )
        child_manifest = {
            "authorized_capabilities": [
                {
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                }
            ],
            "expires_at": "2030-01-01T00:00:00Z",
        }

        child = (
            runtime.process.spawn_child(
                parent,
                "child with an earlier lease",
                authority_manifest=child_manifest,
            )
            if transition == "spawn_child"
            else runtime.process.fork(
                parent,
                "fork with an earlier lease",
                authority_manifest=child_manifest,
            )
        )

        manifest = runtime.authority_manifests.get_for_process(child)
        capability = next(
            cap
            for cap in runtime.capability.capabilities_for(child)
            if cap.resource == resource
        )
        assert manifest is not None
        assert manifest.expires_at == "2030-01-01T00:00:00Z"
        assert capability.expires_at == "2030-01-01T00:00:00+00:00"
    finally:
        runtime.close()


def test_human_allow_approval_fails_closed_when_manifest_expires_while_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        resource = "filesystem:workspace:pending-expiry.txt"
        pid = runtime.process.spawn(
            goal="pending approval expiry",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": resource,
                            "rights": [CapabilityRight.WRITE.value],
                        }
                    ]
                },
                "expires_at": "2030-01-01T00:00:00Z",
            },
        )
        request_id = runtime.human.request_permission(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            resource,
            [CapabilityRight.WRITE.value],
            "manifest may expire while the Human decides",
        )

        class ExpiredDateTime(datetime):
            @classmethod
            def now(cls, tz: timezone | None = None) -> datetime:
                selected = datetime(2031, 1, 1, tzinfo=timezone.utc)
                return selected if tz is None else selected.astimezone(tz)

        monkeypatch.setattr(authority_manifest_module, "datetime", ExpiredDateTime)

        with pytest.raises(CapabilityDenied, match="task authority manifest expired"):
            runtime.human.approve(
                request_id,
                {"approved": True, "policy": CapabilityManager.ALWAYS_ALLOW},
            )
        assert runtime.human.get(request_id).status.value == "pending"
        assert not any(
            cap.resource == resource for cap in runtime.capability.capabilities_for(pid)
        )
    finally:
        runtime.close()


def test_manifest_expiry_caps_human_requested_capability_variants() -> None:
    runtime = Runtime.open("local")
    try:
        manifest_expiry = "2099-01-01T00:00:00Z"
        permanent_resource = "object:human-grant"
        once_resource = "object:human-grant-once"
        pid = runtime.process.spawn(
            goal="bound all Human capability variants",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "expires_at": manifest_expiry,
            },
        )
        request_id = runtime.human.query(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            {
                "type": "approval",
                "question": "grant bounded capabilities",
                "requested_capability": {
                    "resource": permanent_resource,
                    "rights": [CapabilityRight.READ.value],
                    "expires_at": "2199-01-01T00:00:00Z",
                },
                "requested_once_capability": {
                    "resource": once_resource,
                    "rights": [CapabilityRight.READ.value],
                    "expires_at": "2199-01-01T00:00:00Z",
                },
            },
            blocking=False,
        )
        runtime.human.approve(request_id)

        grants = {
            cap.resource: cap
            for cap in runtime.capability.capabilities_for(pid)
            if cap.resource in {permanent_resource, once_resource}
        }
        assert grants[permanent_resource].expires_at == "2099-01-01T00:00:00+00:00"
        assert grants[permanent_resource].uses_remaining is None
        assert grants[once_resource].expires_at == "2099-01-01T00:00:00+00:00"
        assert grants[once_resource].uses_remaining == 1
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("requestable_expiry", "expected_expiry"),
    [
        ("2030-01-01T00:00:00Z", "2030-01-01T00:00:00+00:00"),
        ("2199-01-01T00:00:00Z", "2099-01-01T00:00:00+00:00"),
    ],
    ids=["earlier-preserved", "later-clamped"],
)
def test_standard_permission_request_inherits_requestable_expiry_ceiling(
    requestable_expiry: str,
    expected_expiry: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        resource = f"object:requestable-expiry:{requestable_expiry[:4]}"
        pid = runtime.process.spawn(
            goal="requestable expiry lease",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": resource,
                            "rights": [CapabilityRight.READ.value],
                            "expires_at": requestable_expiry,
                        }
                    ]
                },
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )
        request_id = runtime.human.request_permission(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            resource,
            [CapabilityRight.READ.value],
            "inherit the requestable lease",
            blocking=False,
        )
        request = runtime.human.get(request_id)
        assert request.payload["requested_permission"]["expires_at"] == requestable_expiry
        assert request.payload["context"]["lease"]["expires_at"] == requestable_expiry

        runtime.human.approve(
            request_id,
            {"approved": True, "policy": CapabilityManager.ALWAYS_ALLOW},
        )
        capability = next(
            cap
            for cap in runtime.capability.capabilities_for(pid)
            if cap.resource == resource
        )
        assert capability.expires_at == expected_expiry
    finally:
        runtime.close()


@pytest.mark.parametrize("expires_at", [123, "", "not-a-time"])
def test_human_capability_grant_rejects_malformed_expiry(expires_at: object) -> None:
    runtime = Runtime.open("local")
    try:
        resource = "object:invalid-human-expiry"
        pid = runtime.process.spawn(goal="reject invalid Human lease")
        runtime.capability.grant(
            pid,
            DEFAULT_CONFIG.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        request_id = runtime.human.query(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            {
                "type": "approval",
                "question": "invalid lease must not become permanent",
                "requested_capability": {
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                    "expires_at": expires_at,
                },
            },
            blocking=False,
        )

        with pytest.raises(ValidationError, match="expires_at"):
            runtime.human.approve(request_id)
        assert runtime.human.get(request_id).status.value == "pending"
        assert not any(
            cap.resource == resource for cap in runtime.capability.capabilities_for(pid)
        )
    finally:
        runtime.close()


def test_human_capability_grant_rejects_malformed_authority_fields() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject malformed Human authority")
        runtime.capability.grant(
            pid,
            DEFAULT_CONFIG.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        malformed_specs: list[tuple[str, dict[str, object]]] = [
            ("rights-string", {"rights": "read"}),
            ("rights-empty", {"rights": []}),
            ("rights-unknown", {"rights": ["not-a-right"]}),
            ("delegable-string", {"rights": ["read"], "delegable": "false"}),
            ("constraints-string", {"rights": ["read"], "constraints": "none"}),
        ]

        for request_field in (
            "requested_capability",
            "requested_once_capability",
        ):
            for case, malformed in malformed_specs:
                resource = f"object:invalid-human-authority:{request_field}:{case}"
                request_id = runtime.human.query(
                    pid,
                    DEFAULT_CONFIG.runtime.default_human,
                    {
                        "type": "approval",
                        "question": "malformed authority must fail closed",
                        request_field: {
                            "resource": resource,
                            **malformed,
                        },
                    },
                    blocking=False,
                )

                with pytest.raises(ValidationError):
                    runtime.human.approve(request_id)
                assert runtime.human.get(request_id).status.value == "pending"
                assert not any(
                    cap.resource == resource
                    for cap in runtime.capability.capabilities_for(pid)
                )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "manifest",
    [
        {"expires_at": "not-a-time"},
        {
            "authorized_capabilities": [
                {
                    "resource": "object:invalid-expiry",
                    "rights": [CapabilityRight.READ.value],
                    "expires_at": "not-a-time",
                }
            ]
        },
    ],
    ids=["manifest", "capability"],
)
def test_manifest_rejects_malformed_expiry_before_persisting(
    manifest: dict[str, object],
) -> None:
    runtime = Runtime.open("local")
    try:
        before_processes = runtime.store.list_processes()
        before_manifests = runtime.store.list_authority_manifests()

        with pytest.raises(ValidationError, match="expires_at must be an ISO-8601"):
            runtime.process.spawn(
                goal="reject malformed manifest expiry",
                authority_manifest=manifest,
            )

        assert runtime.store.list_processes() == before_processes
        assert runtime.store.list_authority_manifests() == before_manifests
    finally:
        runtime.close()


def test_manifest_rejects_invalid_capability_use_leases_without_residue() -> None:
    runtime = Runtime.open("local")
    try:
        invalid_values: list[tuple[str, object]] = [
            ("bool", True),
            ("zero", 0),
            ("negative", -1),
            ("fraction", 1.5),
            ("nan", float("nan")),
            ("infinity", float("inf")),
        ]
        for case, uses_remaining in invalid_values:
            before_processes = runtime.store.list_processes()
            before_manifests = runtime.store.list_authority_manifests()
            before_capabilities = runtime.store.list_capabilities()

            with pytest.raises(ValidationError, match="uses_remaining"):
                runtime.process.spawn(
                    goal=f"reject invalid capability lease {case}",
                    authority_manifest={
                        "authorized_capabilities": [
                            {
                                "resource": f"object:invalid-lease:{case}",
                                "rights": [CapabilityRight.READ.value],
                                "uses_remaining": uses_remaining,
                            }
                        ]
                    },
                )

            assert runtime.store.list_processes() == before_processes
            assert runtime.store.list_authority_manifests() == before_manifests
            assert runtime.store.list_capabilities() == before_capabilities
    finally:
        runtime.close()


def test_mutated_resource_budget_instances_are_revalidated_at_boundaries() -> None:
    runtime = Runtime.open("local")
    try:
        invalid_fields: list[tuple[str, str, object]] = [
            ("bool", "max_tool_calls", True),
            ("negative", "max_tool_calls", -1),
            ("fraction", "max_tool_calls", 1.5),
            ("nan", "max_runtime_seconds", float("nan")),
            ("infinity", "max_runtime_seconds", float("inf")),
        ]
        for case, field, value in invalid_fields:
            budget = ResourceBudget()
            setattr(budget, field, value)

            with pytest.raises(ValidationError, match="invalid resource_budget"):
                runtime.authority_manifests.prepare_launch(
                    pid=f"pid_invalid_mutated_budget_{case}",
                    image_id="base-agent:v0",
                    goal_ref=None,
                    resource_budget=budget,
                )
            assert (
                runtime.authority_manifests.get_for_process(
                    f"pid_invalid_mutated_budget_{case}"
                )
                is None
            )

            before_processes = runtime.store.list_processes()
            before_manifests = runtime.store.list_authority_manifests()
            with pytest.raises(ProcessError):
                runtime.process.spawn(
                    goal=f"reject mutated process budget {case}",
                    resource_budget=budget,
                )
            assert runtime.store.list_processes() == before_processes
            assert runtime.store.list_authority_manifests() == before_manifests
    finally:
        runtime.close()


@pytest.mark.parametrize("transition", ["spawn_child", "fork"])
def test_child_manifest_budget_is_applied_before_process_reservation(
    transition: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="bounded child budget",
            authority_manifest={"resource_budget": {"max_tool_calls": 8}},
        )
        if transition == "spawn_child":
            child = runtime.process.spawn_child(
                parent,
                "one child tool",
                authority_manifest={"resource_budget": {"max_tool_calls": 1}},
            )
        else:
            child = runtime.process.fork(
                parent,
                "one fork tool",
                authority_manifest={"resource_budget": {"max_tool_calls": 1}},
            )

        manifest = runtime.authority_manifests.get_for_process(child)
        assert manifest is not None
        assert manifest.resource_budget["max_tool_calls"] == 1
        assert runtime.process.get(child).resource_budget.max_tool_calls == 1
        reservation = runtime.resources.resource_repository.get_resource_reservation(
            parent,
            child,
        )
        assert reservation is not None
        assert reservation.reserved["max_tool_calls"] == 1

        runtime.tools.configure_process_tools(
            child,
            ["get_working_directory"],
            assigned_by="test",
        )
        assert runtime.tools.call(child, "get_working_directory", {}).ok
        denied = runtime.tools.call(child, "get_working_directory", {})
        assert not denied.ok
        assert "max_tool_calls" in (denied.error or "")
    finally:
        runtime.close()


@pytest.mark.parametrize("transition", ["spawn_child", "fork"])
def test_child_budget_without_child_manifest_keeps_normal_attenuation(
    transition: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="ordinary child budget",
            authority_manifest={"resource_budget": {"max_tool_calls": 8}},
        )
        child = (
            runtime.process.spawn_child(parent, "ordinary child")
            if transition == "spawn_child"
            else runtime.process.fork(parent, "ordinary fork")
        )

        assert runtime.process.get(child).resource_budget.max_tool_calls == 4
    finally:
        runtime.close()


def test_child_manifest_budget_cannot_widen_explicit_runtime_budget() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="runtime budget remains a ceiling",
            authority_manifest={"resource_budget": {"max_tool_calls": 8}},
        )
        child = runtime.process.spawn_child(
            parent,
            "intersection budget",
            resource_budget=ResourceBudget(max_tool_calls=2),
            authority_manifest={"resource_budget": {"max_tool_calls": 7}},
        )

        manifest = runtime.authority_manifests.get_for_process(child)
        assert manifest is not None
        assert manifest.resource_budget["max_tool_calls"] == 2
        assert runtime.process.get(child).resource_budget.max_tool_calls == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("transition", ["spawn_child", "fork"])
def test_child_budget_intersects_multiple_manifest_fields_and_reservation(
    transition: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="multi-field parent budget",
            authority_manifest={
                "resource_budget": {
                    "max_tool_calls": 20,
                    "max_runtime_seconds": 20.0,
                    "max_llm_calls": 10,
                }
            },
        )
        runtime_budget = ResourceBudget(
            max_tool_calls=2,
            max_runtime_seconds=6.0,
            max_llm_calls=None,
        )
        child_manifest = {
            "resource_budget": {
                "max_tool_calls": 7,
                "max_runtime_seconds": 3.0,
                "max_llm_calls": 1,
            }
        }
        child = (
            runtime.process.spawn_child(
                parent,
                "multi-field child",
                resource_budget=runtime_budget,
                authority_manifest=child_manifest,
            )
            if transition == "spawn_child"
            else runtime.process.fork(
                parent,
                "multi-field fork",
                resource_budget=runtime_budget,
                authority_manifest=child_manifest,
            )
        )

        budget = runtime.process.get(child).resource_budget
        assert budget.max_tool_calls == 2
        assert budget.max_runtime_seconds == 3.0
        assert budget.max_llm_calls == 1
        reservation = runtime.resources.resource_repository.get_resource_reservation(
            parent,
            child,
        )
        assert reservation is not None
        assert reservation.reserved["max_tool_calls"] == 2
        assert reservation.reserved["max_runtime_seconds"] == 3.0
        assert reservation.reserved["max_llm_calls"] == 1
    finally:
        runtime.close()


def test_root_budget_uses_field_wise_runtime_and_manifest_intersection() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            goal="root budget intersection",
            resource_budget=ResourceBudget(
                max_tool_calls=2,
                max_llm_calls=None,
            ),
            authority_manifest={
                "resource_budget": {
                    "max_tool_calls": 7,
                    "max_llm_calls": 1,
                }
            },
        )

        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None
        assert manifest.resource_budget["max_tool_calls"] == 2
        assert manifest.resource_budget["max_llm_calls"] == 1
        assert runtime.process.get(pid).resource_budget.max_tool_calls == 2
        assert runtime.process.get(pid).resource_budget.max_llm_calls == 1
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("runtime_limit", "manifest_limit", "expected"),
    [
        (2, None, 2),
        (None, 1, 1),
    ],
    ids=[
        "unlimited-manifest-keeps-finite-runtime",
        "finite-manifest-bounds-unlimited-runtime",
    ],
)
def test_child_budget_none_semantics_preserve_the_narrower_limit(
    runtime_limit: int | None,
    manifest_limit: int | None,
    expected: int,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="None budget intersection",
            authority_manifest={"resource_budget": {"max_tool_calls": 8}},
        )
        child = runtime.process.spawn_child(
            parent,
            "None-aware child budget",
            resource_budget=ResourceBudget(max_tool_calls=runtime_limit),
            authority_manifest={
                "resource_budget": {"max_tool_calls": manifest_limit}
            },
        )

        manifest = runtime.authority_manifests.get_for_process(child)
        assert manifest is not None
        assert manifest.resource_budget["max_tool_calls"] == expected
        assert runtime.process.get(child).resource_budget.max_tool_calls == expected
    finally:
        runtime.close()


def test_child_budget_parent_ceiling_rejection_leaves_no_launch_state() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="strict parent budget ceiling",
            authority_manifest={"resource_budget": {"max_tool_calls": 1}},
        )
        before_processes = runtime.store.list_processes()
        before_publications = runtime.store.list_runtime_publications()
        before_usage = runtime.process.get(parent).resource_usage

        with pytest.raises(CapabilityDenied, match="resource budget exceeds"):
            runtime.process.spawn_child(
                parent,
                "over-budget child",
                resource_budget=ResourceBudget(max_tool_calls=2),
                authority_manifest={"resource_budget": {"max_tool_calls": 2}},
            )

        assert runtime.store.list_processes() == before_processes
        assert runtime.store.list_runtime_publications() == before_publications
        assert runtime.process.get(parent).resource_usage == before_usage
        assert (
            runtime.resources.resource_repository.list_resource_reservations(
                parent_pid=parent,
            )
            == []
        )
    finally:
        runtime.close()


def test_manifest_resource_budget_validation_is_closed_and_typed() -> None:
    runtime = Runtime.open("local")
    try:
        invalid_budgets = [
            {"unknown_limit": 1},
            {"max_tool_calls": True},
            {"max_tool_calls": -1},
            {"max_tool_calls": 1.5},
            {"max_runtime_seconds": float("inf")},
        ]
        for budget in invalid_budgets:
            with pytest.raises(ValidationError, match="invalid resource_budget"):
                runtime.authority_manifests.prepare_launch(
                    pid="pid_invalid_resource_budget",
                    image_id="base-agent:v0",
                    goal_ref=None,
                    supplied={"resource_budget": budget},
                )
            assert (
                runtime.authority_manifests.get_for_process(
                    "pid_invalid_resource_budget"
                )
                is None
            )
    finally:
        runtime.close()


def test_host_manifest_is_hashed_persisted_and_compiles_only_declared_authority(tmp_path: Path) -> None:
    database = tmp_path / "manifest.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="explicit manifest",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "permitted_effects": ["human.*"],
                "metadata": {"contract": "test"},
            },
        )
        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None and len(manifest.manifest_hash) == 64
        assert runtime.capability.check(pid, DEFAULT_CONFIG.runtime.default_human_resource, "write")
        assert not runtime.capability.check(pid, "filesystem:workspace:*", "read")
        manifest_id = manifest.manifest_id
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        restored = reopened.authority_manifests.get(manifest_id)
        assert restored.pid == pid
        assert reopened.authority_manifests.summary_for_process(pid)["manifest_hash"] == restored.manifest_hash
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("permitted_effect", ["filesystem.*"]),
        ("expiry_at", "2030-01-01T00:00:00Z"),
    ],
    ids=["permitted-effects-typo", "expires-at-typo"],
)
def test_manifest_rejects_unknown_top_level_fields(field: str, value: object) -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(
            ValidationError,
            match="authority manifest contains unsupported fields",
        ):
            runtime.process.spawn(
                goal="reject misspelled authority",
                authority_manifest={field: value},
            )
    finally:
        runtime.close()


def test_manifest_rejects_mixed_non_string_unknown_fields_without_side_effects() -> None:
    runtime = Runtime.open("local")
    try:
        before_processes = runtime.store.list_processes()
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(
            ValidationError,
            match="authority manifest contains unsupported fields",
        ):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_manifest",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={2: "invalid", "typo": "invalid"},  # type: ignore[dict-item]
            )
        assert runtime.store.list_processes() == before_processes
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert runtime.authority_manifests.get_for_process("pid_invalid_manifest") is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "manifest",
    [
        {
            "authorized_capabilities": [
                {
                    "resource": "filesystem:workspace:report.txt",
                    "rights": [CapabilityRight.READ.value],
                    "permitted_effects": ["filesystem.*"],
                }
            ]
        },
        {
            "approval_policy": {
                "requestable_capabilities": [
                    {
                        "resource": "filesystem:workspace:report.txt",
                        "rights": [CapabilityRight.READ.value],
                        "expire_at": "2030-01-01T00:00:00Z",
                    }
                ]
            }
        },
    ],
    ids=["authorized", "requestable"],
)
def test_manifest_rejects_unknown_capability_entry_fields(
    manifest: dict[str, object],
) -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(
            ValidationError,
            match="authority manifest capability entry contains unsupported fields",
        ):
            runtime.process.spawn(
                goal="reject misplaced capability policy",
                authority_manifest=manifest,
            )
    finally:
        runtime.close()


@pytest.mark.parametrize("field", ["delegable", "revocable"])
@pytest.mark.parametrize(
    "value",
    ["false", "true", 0, 1, 0.0, 1.0, None],
    ids=[
        "false-string",
        "true-string",
        "zero",
        "one",
        "zero-float",
        "one-float",
        "null",
    ],
)
def test_manifest_capability_boolean_fields_require_json_booleans(
    field: str,
    value: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(
            ValidationError,
            match=rf"authority manifest capability entry {field} must be a boolean",
        ):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_capability_boolean",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={
                    "authorized_capabilities": [
                        {
                            "resource": "filesystem:workspace:report.txt",
                            "rights": [CapabilityRight.READ.value],
                            field: value,
                        }
                    ]
                },
            )
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert (
            runtime.authority_manifests.get_for_process(
                "pid_invalid_capability_boolean"
            )
            is None
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("field", ["delegable", "revocable"])
def test_requestable_manifest_capability_boolean_fields_require_json_booleans(
    field: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(
            ValidationError,
            match=rf"authority manifest capability entry {field} must be a boolean",
        ):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_requestable_boolean",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:report.txt",
                                "rights": [CapabilityRight.READ.value],
                                field: "false",
                            }
                        ]
                    }
                },
            )
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert (
            runtime.authority_manifests.get_for_process(
                "pid_invalid_requestable_boolean"
            )
            is None
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("constraints", {"authority_rules": []}, "constraints must be an empty object"),
        ("constraints", None, "constraints must be an empty object"),
        ("delegable", True, "delegable must be false"),
        ("revocable", False, "revocable must be true"),
        ("uses_remaining", 1, "uses_remaining is not supported"),
        ("uses_remaining", None, "uses_remaining is not supported"),
        ("max_delegation_depth", 0, "max_delegation_depth is not supported"),
        ("max_delegation_depth", None, "max_delegation_depth is not supported"),
    ],
    ids=[
        "constraints",
        "constraints-null",
        "delegable",
        "revocable",
        "finite-use",
        "finite-use-null",
        "delegation-depth",
        "delegation-depth-null",
    ],
)
def test_requestable_capability_rejects_fields_that_cannot_propagate(
    field: str,
    value: object,
    message: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(ValidationError, match=message):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_requestable_semantics",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:report.txt",
                                "rights": [CapabilityRight.READ.value],
                                field: value,
                            }
                        ]
                    }
                },
            )
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert (
            runtime.authority_manifests.get_for_process(
                "pid_invalid_requestable_semantics"
            )
            is None
        )
    finally:
        runtime.close()


def test_requestable_capability_accepts_canonical_host_compatibility_values() -> None:
    runtime = Runtime.open("local")
    try:
        manifest = runtime.authority_manifests.prepare_launch(
            pid="pid_canonical_requestable_semantics",
            image_id="base-agent:v0",
            goal_ref=None,
            supplied={
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": "filesystem:workspace:report.txt",
                            "rights": [CapabilityRight.READ.value],
                            "constraints": {},
                            "delegable": False,
                            "revocable": True,
                            "expires_at": "2030-01-01T00:00:00Z",
                        }
                    ]
                }
            },
        )

        assert manifest.approval_policy["requestable_capabilities"] == [
            {
                "resource": "filesystem:workspace:report.txt",
                "rights": [CapabilityRight.READ.value],
                "constraints": {},
                "delegable": False,
                "revocable": True,
                "expires_at": "2030-01-01T00:00:00Z",
            }
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "max_depth",
    [True, -1, "1", 1.0, 1.5],
    ids=["bool", "negative", "string", "integral-float", "fraction"],
)
def test_manifest_max_delegation_depth_requires_non_negative_json_integer(
    max_depth: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(
            ValidationError,
            match="max_delegation_depth must be a non-negative integer",
        ):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_max_delegation_depth",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={
                    "authorized_capabilities": [
                        {
                            "resource": "filesystem:workspace:report.txt",
                            "rights": [CapabilityRight.READ.value],
                            "max_delegation_depth": max_depth,
                        }
                    ]
                },
            )
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert (
            runtime.authority_manifests.get_for_process(
                "pid_invalid_max_delegation_depth"
            )
            is None
        )
    finally:
        runtime.close()


def test_manifest_capability_boolean_fields_compile_exact_values() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            goal="compile exact capability booleans",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "filesystem:workspace:delegable.txt",
                        "rights": [CapabilityRight.READ.value],
                        "delegable": True,
                        "revocable": False,
                    }
                ]
            },
        )

        capability = next(
            item
            for item in runtime.capability.capabilities_for(pid)
            if item.resource == "filesystem:workspace:delegable.txt"
        )
        assert capability.delegable is True
        assert capability.revocable is False
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "effect",
    [None, 1, True, "", "   ", "git.*.*", "git*"],
    ids=[
        "null",
        "integer",
        "boolean",
        "empty",
        "whitespace",
        "multiple-wildcards",
        "embedded-wildcard",
    ],
)
def test_manifest_effect_entries_require_strings_and_one_terminal_wildcard(
    effect: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.store.list_events()
        before_audit = runtime.store.list_audit()
        with pytest.raises(ValidationError, match="permitted_effects"):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_effect_class",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={"permitted_effects": [effect]},
            )
        assert runtime.store.list_events() == before_events
        assert runtime.store.list_audit() == before_audit
        assert (
            runtime.authority_manifests.get_for_process("pid_invalid_effect_class")
            is None
        )
    finally:
        runtime.close()


def test_unknown_effect_entries_grant_no_current_effect_but_global_wildcard_does() -> None:
    runtime = Runtime.open("local")
    try:
        unknown = runtime.process.spawn(
            goal="future extension effect",
            authority_manifest={"permitted_effects": ["extension.future"]},
        )
        wildcard = runtime.process.spawn(
            goal="all effects",
            authority_manifest={"permitted_effects": ["*"]},
        )

        with pytest.raises(CapabilityDenied, match="does not permit effect class"):
            runtime.authority_manifests.assert_effect(unknown, "filesystem.read_bytes")
        runtime.authority_manifests.assert_effect(wildcard, "filesystem.read_bytes")
    finally:
        runtime.close()


def test_data_flow_policy_requires_lists_in_python_manifests() -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(
            ValidationError,
            match=r"data_flow_policy\.allowed_tenants must be a list",
        ):
            runtime.process.spawn(
                goal="reject tuple identity policy",
                authority_manifest={
                    "data_flow_policy": {
                        "allowed_tenants": ("tenant-a",),
                        "allowed_principals": [],
                    }
                },
            )
    finally:
        runtime.close()


def test_data_flow_policy_rejects_mixed_non_string_unknown_fields() -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(
            ValidationError,
            match="data_flow_policy contains unsupported fields",
        ):
            runtime.authority_manifests.prepare_launch(
                pid="pid_invalid_data_flow_policy",
                image_id="base-agent:v0",
                goal_ref=None,
                supplied={
                    "data_flow_policy": {
                        "allowed_tenants": [],
                        "allowed_principals": [],
                        2: "invalid",
                        "typo": "invalid",
                    }
                },
            )
        assert (
            runtime.authority_manifests.get_for_process(
                "pid_invalid_data_flow_policy"
            )
            is None
        )
    finally:
        runtime.close()


def test_model_permission_request_outside_manifest_is_denied_before_human_request() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="bounded request",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ]
            },
        )

        with pytest.raises(CapabilityDenied, match="exceeds task authority manifest"):
            runtime.human.request_permission(
                pid,
                DEFAULT_CONFIG.runtime.default_human,
                "filesystem:workspace:outside.txt",
                [CapabilityRight.WRITE.value],
                "outside launch contract",
            )

        assert runtime.human.list(pid=pid) == []
    finally:
        runtime.close()


def test_requestable_manifest_authority_allows_prompt_without_pregranting_capability() -> None:
    runtime = Runtime.open("local")
    try:
        resource = "filesystem:workspace:report.txt"
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="requestable authority",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": DEFAULT_CONFIG.runtime.default_human_resource,
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "approval_policy": {
                    "requestable_capabilities": [
                        {"resource": resource, "rights": [CapabilityRight.WRITE.value]},
                    ]
                },
            },
        )
        assert not runtime.capability.check(pid, resource, CapabilityRight.WRITE)

        request_id = runtime.human.request_permission(
            pid,
            DEFAULT_CONFIG.runtime.default_human,
            resource,
            [CapabilityRight.WRITE.value],
            "write the report",
        )

        assert runtime.human.get(request_id).status.value == "pending"
        assert not runtime.capability.check(pid, resource, CapabilityRight.WRITE)
    finally:
        runtime.close()


def test_implicit_manifest_denies_model_requests_but_preserves_host_transition_authority() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="implicit host authority")
        resource = "filesystem:workspace:reports/host-granted.txt"

        with pytest.raises(CapabilityDenied, match="exceeds task authority manifest"):
            runtime.authority_manifests.assert_capability_request(
                parent,
                resource,
                [CapabilityRight.READ.value],
            )

        runtime.capability.grant(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="host:test",
        )
        runtime.capability.grant(
            parent,
            resource,
            [CapabilityRight.READ],
            issued_by="host:test",
            delegable=True,
        )
        child = runtime.process.spawn_child(
            parent,
            "derived host authority",
            inherit_capabilities=[
                {"resource": resource, "rights": [CapabilityRight.READ.value]}
            ],
        )

        assert runtime.capability.check(child, resource, CapabilityRight.READ)
    finally:
        runtime.close()


def test_implicit_manifest_records_host_launch_capabilities() -> None:
    runtime = Runtime.open("local")
    try:
        resource = "filesystem:workspace:reports/launch-granted.txt"
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="implicit launch authority",
            capabilities=[
                {
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                }
            ],
        )

        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None
        assert manifest.authorized_capabilities[0]["resource"] == resource
        assert manifest.authorized_capabilities[0]["rights"] == [
            CapabilityRight.READ.value
        ]
        assert manifest.metadata["explicit"] is False
        assert manifest.metadata["transition_ceiling"] is False
        assert runtime.capability.check(pid, resource, CapabilityRight.READ)
        runtime.authority_manifests.assert_capability_request(
            pid,
            resource,
            [CapabilityRight.READ.value],
        )
    finally:
        runtime.close()


def test_explicit_empty_object_keeps_host_launch_capabilities_as_ceiling() -> None:
    runtime = Runtime.open("local")
    try:
        resource = "filesystem:workspace:reports/explicit-launch.txt"
        pid = runtime.process.spawn(
            goal="explicit launch authority",
            capabilities=[
                {
                    "resource": resource,
                    "rights": [CapabilityRight.READ.value],
                }
            ],
            authority_manifest={},
        )

        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None
        assert manifest.authorized_capabilities[0]["resource"] == resource
        assert manifest.metadata["explicit"] is True
        assert manifest.metadata["transition_ceiling"] is True
        assert runtime.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        runtime.close()


def test_child_manifest_and_transition_api_enforce_parent_intersection() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="parent",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "filesystem:workspace:reports/*",
                        "rights": [CapabilityRight.READ.value],
                        "delegable": True,
                    }
                ]
            },
        )
        child_spec = {
            "resource": "filesystem:workspace:reports/q1.txt",
            "rights": [CapabilityRight.READ.value],
        }
        child = runtime.process.spawn_child(
            parent,
            "child",
            authority_manifest={"authorized_capabilities": [child_spec]},
        )
        assert runtime.capability.check(child, child_spec["resource"], CapabilityRight.READ)
        child_manifest = runtime.authority_manifests.get_for_process(child)
        assert child_manifest is not None
        assert child_manifest.parent_manifest_id == runtime.authority_manifests.get_for_process(parent).manifest_id

        forked = runtime.process.fork(
            parent,
            "forked child",
            authority_manifest={"authorized_capabilities": [child_spec]},
        )
        assert runtime.capability.check(forked, child_spec["resource"], CapabilityRight.READ)
        forked_manifest = runtime.authority_manifests.get_for_process(forked)
        assert forked_manifest is not None
        assert forked_manifest.authorized_capabilities == child_manifest.authorized_capabilities

        outside = {
            "resource": "filesystem:workspace:secrets/key.txt",
            "rights": [CapabilityRight.READ.value],
        }
        with pytest.raises(CapabilityDenied, match="derived child authority"):
            runtime.process.spawn_child(
                parent,
                "outside",
                authority_manifest={"authorized_capabilities": [outside]},
            )
    finally:
        runtime.close()


def test_child_manifest_cannot_widen_parent_policy_ceilings() -> None:
    runtime = Runtime.open("local")
    try:
        parent_resource = "filesystem:workspace:reports/*"
        child_resource = "filesystem:workspace:reports/q1.txt"
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="parent policy ceiling",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": parent_resource,
                        "rights": [CapabilityRight.READ.value],
                        "delegable": True,
                    }
                ],
                "permitted_effects": ["filesystem.*"],
                "approval_policy": {
                    "mode": "operator",
                    "requestable_capabilities": [
                        {
                            "resource": parent_resource,
                            "rights": [CapabilityRight.READ.value],
                        }
                    ],
                },
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": ["tenant-a"],
                    "allowed_principals": [],
                },
                "expires_at": "2030-01-01T00:00:00Z",
            },
        )
        child_spec = {
            "resource": child_resource,
            "rights": [CapabilityRight.READ.value],
        }
        child = runtime.process.spawn_child(
            parent,
            "inherited policy ceiling",
            capabilities=[child_spec],
            authority_manifest={"authorized_capabilities": [child_spec]},
        )
        child_manifest = runtime.authority_manifests.get_for_process(child)
        assert child_manifest is not None
        assert child_manifest.permitted_effects == ["filesystem.*"]
        assert child_manifest.approval_policy["mode"] == "operator"
        assert child_manifest.data_flow_policy == {
            "schema_version": 1,
            "allowed_tenants": ["tenant-a"],
            "allowed_principals": [],
        }
        assert child_manifest.expires_at == "2030-01-01T00:00:00Z"

        deny_all_child = runtime.process.spawn_child(
            parent,
            "deny all effects",
            capabilities=[child_spec],
            authority_manifest={
                "authorized_capabilities": [child_spec],
                "permitted_effects": [],
            },
        )
        assert (
            runtime.authority_manifests.get_for_process(deny_all_child).permitted_effects
            == []
        )

        with pytest.raises(CapabilityDenied, match="effect ceiling"):
            runtime.process.spawn_child(
                parent,
                "widen effects",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "permitted_effects": ["jsonrpc.*"],
                },
            )
        with pytest.raises(CapabilityDenied, match="effect ceiling"):
            runtime.process.spawn_child(
                parent,
                "remove effect ceiling",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "permitted_effects": None,
                },
            )
        with pytest.raises(CapabilityDenied, match="requestable capability"):
            runtime.process.spawn_child(
                parent,
                "widen requestable authority",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:secrets/*",
                                "rights": [CapabilityRight.WRITE.value],
                            }
                        ]
                    },
                },
            )
        with pytest.raises(CapabilityDenied, match="expiry"):
            runtime.process.spawn_child(
                parent,
                "widen expiry",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "expires_at": "2040-01-01T00:00:00Z",
                },
            )
        with pytest.raises(CapabilityDenied, match="data_flow_policy"):
            runtime.process.spawn_child(
                parent,
                "replace data flow policy",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "data_flow_policy": {
                        "schema_version": 1,
                        "allowed_tenants": ["tenant-b"],
                        "allowed_principals": [],
                    },
                },
            )
        with pytest.raises(ValidationError, match="unsupported fields"):
            runtime.process.spawn_child(
                parent,
                "add data flow escape",
                capabilities=[child_spec],
                authority_manifest={
                    "authorized_capabilities": [child_spec],
                    "data_flow_policy": {
                        "schema_version": 1,
                        "allowed_tenants": ["tenant-a"],
                        "allowed_principals": [],
                        "allow_external": True,
                    },
                },
            )
    finally:
        runtime.close()


def test_manifest_max_delegation_depth_is_compiled_and_cannot_be_broadened() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="non-delegable manifest ceiling",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "filesystem:workspace:reports/*",
                        "rights": [CapabilityRight.READ.value],
                        "delegable": True,
                        "max_delegation_depth": 0,
                    }
                ]
            },
        )
        capability = next(
            item
            for item in runtime.capability.capabilities_for(parent)
            if item.resource == "filesystem:workspace:reports/*"
        )

        assert capability.max_delegation_depth == 0
        with pytest.raises(CapabilityDenied, match="delegation depth exhausted"):
            runtime.capability.delegate(
                parent,
                "pid_child",
                {
                    "resource": "filesystem:workspace:reports/q1.txt",
                    "rights": [CapabilityRight.READ.value],
                },
            )
        assert not runtime.capability.spec_covers(
            {
                "resource": "filesystem:workspace:reports/*",
                "rights": [CapabilityRight.READ.value],
                "delegable": True,
                "max_delegation_depth": 1,
            },
            {
                "resource": "filesystem:workspace:reports/q1.txt",
                "rights": [CapabilityRight.READ.value],
                "delegable": True,
                "max_delegation_depth": 2,
            },
        )
    finally:
        runtime.close()


def test_checkpoint_fork_preserves_explicit_manifest_policy_ceilings() -> None:
    runtime = Runtime.open("local")
    try:
        requestable = {
            "resource": "filesystem:workspace:later.txt",
            "rights": [CapabilityRight.READ.value],
        }
        parent = runtime.process.spawn(
            goal="checkpoint manifest source",
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": "jsonrpc:demo:update",
                        "rights": [CapabilityRight.WRITE.value],
                    }
                ],
                "permitted_effects": ["filesystem.*"],
                "resource_budget": {"max_tool_calls": 7},
                "approval_policy": {
                    "mode": "operator",
                    "requestable_capabilities": [requestable],
                },
                "data_flow_policy": {
                    "schema_version": 1,
                    "allowed_tenants": [],
                    "allowed_principals": [],
                },
                "expires_at": "2030-01-01T00:00:00Z",
            },
        )
        source_manifest = runtime.authority_manifests.get_for_process(parent)
        checkpoint_id = runtime.checkpoint.create(parent, "fork manifest", require_capability=False)

        fork = runtime.checkpoint.fork_from_checkpoint(
            parent,
            checkpoint_id,
            require_capability=False,
        )
        fork_pid = fork["fork_root_pid"]
        manifest = runtime.authority_manifests.get_for_process(fork_pid)

        assert manifest is not None
        assert source_manifest is not None
        assert manifest.parent_manifest_id == source_manifest.manifest_id
        assert manifest.permitted_effects == ["filesystem.*"]
        assert manifest.resource_budget["max_tool_calls"] == 7
        assert manifest.approval_policy == {
            "mode": "operator",
            "requestable_capabilities": [
                {
                    **requestable,
                    "constraints": {},
                    "delegable": False,
                    "revocable": True,
                }
            ],
        }
        assert manifest.data_flow_policy == {
            "schema_version": 1,
            "allowed_tenants": [],
            "allowed_principals": [],
        }
        assert manifest.expires_at == "2030-01-01T00:00:00Z"
        assert manifest.metadata["transition_ceiling"] is True
        assert runtime.capability.check(fork_pid, "jsonrpc:demo:update", CapabilityRight.WRITE)
        with pytest.raises(CapabilityDenied, match="does not permit effect class"):
            runtime.authority_manifests.assert_effect(fork_pid, "jsonrpc.call")
    finally:
        runtime.close()
