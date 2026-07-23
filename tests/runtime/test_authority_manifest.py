from __future__ import annotations

from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityRight,
    encode_permitted_effects_policy,
    upcast_permitted_effects_policy,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError


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
