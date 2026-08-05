from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.utils.serde import dumps


def _rule(
    *,
    rule_id: str = "workspace-read-v1",
    authority_operation: str = "filesystem.read",
    resource: str = "filesystem:workspace:reports/*",
    rights: list[str] | None = None,
) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "authority_operation": authority_operation,
        "resource": resource,
        "rights": ["read"] if rights is None else rights,
    }


def _policy(*rules: dict[str, object]) -> dict[str, object]:
    return {"schema_version": 1, "rules": list(rules)}


def test_semantic_auto_approval_is_deny_all_when_absent_null_or_empty() -> None:
    runtime = Runtime.open("local")
    try:
        pids = [
            runtime.process.spawn(goal="absent ceiling"),
            runtime.process.spawn(
                goal="null ceiling",
                authority_manifest={
                    "approval_policy": {"semantic_auto_approval": None}
                },
            ),
            runtime.process.spawn(
                goal="empty ceiling",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": _policy()
                    }
                },
            ),
        ]

        absent = runtime.authority_manifests.get_for_process(pids[0])
        null = runtime.authority_manifests.get_for_process(pids[1])
        empty = runtime.authority_manifests.get_for_process(pids[2])
        assert absent is not None and "semantic_auto_approval" not in absent.approval_policy
        assert null is not None and null.approval_policy["semantic_auto_approval"] == _policy()
        assert empty is not None and empty.approval_policy["semantic_auto_approval"] == _policy()
        for pid in pids:
            assert runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/q1.txt",
                rights=["read"],
            ) is None
    finally:
        runtime.close()


def test_semantic_auto_approval_candidate_is_read_only_host_ceiling_evidence() -> None:
    runtime = Runtime.open("local")
    try:
        rules = [
            _rule(rule_id="z-rule"),
            _rule(
                rule_id="a-rule",
                authority_operation="git.diff",
                resource="git:workspace/*",
                rights=["diff"],
            ),
        ]
        pid = runtime.process.spawn(
            goal="strict semantic ceiling",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(*rules)
                }
            },
        )
        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None
        normalized = manifest.approval_policy["semantic_auto_approval"]
        assert [rule["rule_id"] for rule in normalized["rules"]] == [
            "a-rule",
            "z-rule",
        ]

        before = {
            "capabilities": runtime.store.list_capabilities(subject=pid),
            "human_requests": runtime.store.list_human_requests(pid),
            "events": runtime.events.list(),
            "audit": runtime.store.list_audit(),
            "operations": runtime.store.list_operations(pid=pid),
        }
        candidate = runtime.authority_manifests.semantic_auto_approval_candidate(
            pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q1.txt",
            rights=["read"],
        )
        assert candidate == {
            "schema_version": 1,
            "rule_id": "z-rule",
            "authority_operation": "filesystem.read",
            "resource": "filesystem:workspace:reports/q1.txt",
            "rights": ["read"],
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.manifest_hash,
            "policy_sha256": hashlib.sha256(
                dumps(normalized).encode("utf-8")
            ).hexdigest(),
        }
        assert "allow" not in candidate
        assert "permit" not in candidate
        second_candidate = runtime.authority_manifests.semantic_auto_approval_candidate(
            pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q2.txt",
            rights=["read"],
        )
        assert second_candidate is not None
        assert second_candidate["resource"] == "filesystem:workspace:reports/q2.txt"
        assert second_candidate != candidate
        assert {
            "capabilities": runtime.store.list_capabilities(subject=pid),
            "human_requests": runtime.store.list_human_requests(pid),
            "events": runtime.events.list(),
            "audit": runtime.store.list_audit(),
            "operations": runtime.store.list_operations(pid=pid),
        } == before
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:secrets/key.txt",
            rights=["read"],
        ) is None
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            pid,
            authority_operation="shell.run",
            resource="shell:workspace:status",
            rights=["execute"],
        ) is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("semantic_policy", "match"),
    [
        ({}, "missing required fields"),
        ({"schema_version": 2, "rules": []}, "schema_version"),
        ({"schema_version": True, "rules": []}, "schema_version"),
        (
            {"schema_version": 1, "rules": [], "allow": True},
            "unsupported fields",
        ),
        (
            _policy(_rule(), _rule()),
            "duplicate rule_id",
        ),
        (
            _policy(_rule(authority_operation="filesystem.*")),
            "canonical exact string",
        ),
        (
            _policy(
                _rule(
                    authority_operation="shell.run",
                    resource="shell:workspace:status",
                    rights=["execute"],
                )
            ),
            "unsupported semantic auto-approval authority_operation",
        ),
        (
            _policy(
                _rule(
                    authority_operation="filesystem.write",
                    rights=["write"],
                )
            ),
            "unsupported semantic auto-approval authority_operation",
        ),
        (
            _policy(
                _rule(
                    authority_operation="git.write",
                    resource="git:workspace",
                    rights=["write"],
                )
            ),
            "unsupported semantic auto-approval authority_operation",
        ),
        (
            _policy(
                _rule(
                    authority_operation="jsonrpc.call",
                    resource="jsonrpc:demo:read",
                )
            ),
            "unsupported semantic auto-approval authority_operation",
        ),
        (
            _policy(
                _rule(
                    authority_operation="mcp.call",
                    resource="mcp:demo:read",
                )
            ),
            "unsupported semantic auto-approval authority_operation",
        ),
        (
            _policy(_rule(rights=["admin"])),
            "rights are unsupported",
        ),
        (
            _policy(_rule(resource="git:workspace")),
            "resource kind",
        ),
        (
            _policy({**_rule(), "decision": "allow"}),
            "unsupported fields",
        ),
    ],
)
def test_semantic_auto_approval_rejects_malformed_or_unsafe_policy(
    semantic_policy: dict[str, object],
    match: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(ValidationError, match=match):
            runtime.process.spawn(
                goal="invalid semantic ceiling",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": semantic_policy
                    }
                },
            )
    finally:
        runtime.close()


def test_child_semantic_auto_approval_requires_explicit_parent_subset() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="parent semantic ceiling",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(_rule())
                }
            },
        )

        omitted = runtime.process.spawn_child(parent, "omitted semantic ceiling")
        omitted_manifest = runtime.authority_manifests.get_for_process(omitted)
        assert omitted_manifest is not None
        assert "semantic_auto_approval" not in omitted_manifest.approval_policy
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            omitted,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q1.txt",
            rights=["read"],
        ) is None

        narrowed = runtime.process.spawn_child(
            parent,
            "narrowed semantic ceiling",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(
                        _rule(resource="filesystem:workspace:reports/q1.txt")
                    )
                }
            },
        )
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            narrowed,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q1.txt",
            rights=["read"],
        ) is not None
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            narrowed,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q2.txt",
            rights=["read"],
        ) is None

        with pytest.raises(CapabilityDenied, match="cannot widen resource or rights"):
            runtime.process.spawn_child(
                parent,
                "widened semantic ceiling",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": _policy(
                            _rule(resource="filesystem:workspace/*")
                        )
                    }
                },
            )
        with pytest.raises(CapabilityDenied, match="cannot add rule_id"):
            runtime.process.spawn_child(
                parent,
                "renamed semantic ceiling",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": _policy(
                            _rule(rule_id="replacement-rule")
                        )
                    }
                },
            )
        with pytest.raises(CapabilityDenied, match="cannot change authority_operation"):
            runtime.process.spawn_child(
                parent,
                "changed semantic operation",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": _policy(
                            _rule(
                                authority_operation="git.read",
                                resource="git:workspace",
                            )
                        )
                    }
                },
            )
    finally:
        runtime.close()


def test_child_cannot_create_semantic_ceiling_from_ordinary_capability() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="ordinary capability is not an auto ceiling",
            capabilities=[
                {
                    "resource": "filesystem:workspace:reports/*",
                    "rights": ["read"],
                }
            ],
        )
        with pytest.raises(CapabilityDenied, match="cannot add rule_id"):
            runtime.process.spawn_child(
                parent,
                "invalid auto ceiling",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": _policy(_rule())
                    }
                },
            )
    finally:
        runtime.close()


def test_checkpoint_fork_preserves_but_does_not_widen_semantic_ceiling() -> None:
    runtime = Runtime.open("local")
    try:
        source = runtime.process.spawn(
            goal="checkpoint semantic ceiling",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(_rule())
                }
            },
        )
        checkpoint_id = runtime.checkpoint.create(
            source,
            "semantic ceiling fork",
            require_capability=False,
        )
        fork = runtime.checkpoint.fork_from_checkpoint(
            source,
            checkpoint_id,
            require_capability=False,
        )
        fork_pid = fork["fork_root_pid"]
        candidate = runtime.authority_manifests.semantic_auto_approval_candidate(
            fork_pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q1.txt",
            rights=["read"],
        )
        assert candidate is not None
        assert candidate["resource"] == "filesystem:workspace:reports/q1.txt"
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            fork_pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:secrets/key.txt",
            rights=["read"],
        ) is None
    finally:
        runtime.close()


def test_semantic_ceiling_and_legacy_manifest_hashes_survive_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic-ceiling.sqlite"
    runtime = Runtime.open(database)
    try:
        legacy_pid = runtime.process.spawn(goal="legacy manifest without semantic policy")
        semantic_pid = runtime.process.spawn(
            goal="persisted semantic policy",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(_rule())
                }
            },
        )
        legacy_manifest = runtime.authority_manifests.get_for_process(legacy_pid)
        semantic_manifest = runtime.authority_manifests.get_for_process(semantic_pid)
        assert legacy_manifest is not None
        assert semantic_manifest is not None
        legacy_hash = legacy_manifest.manifest_hash
        semantic_hash = semantic_manifest.manifest_hash
        semantic_policy = semantic_manifest.approval_policy["semantic_auto_approval"]
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        restored_legacy = reopened.authority_manifests.get_for_process(legacy_pid)
        restored_semantic = reopened.authority_manifests.get_for_process(semantic_pid)
        assert restored_legacy is not None
        assert restored_semantic is not None
        assert restored_legacy.manifest_hash == legacy_hash
        assert "semantic_auto_approval" not in restored_legacy.approval_policy
        assert restored_semantic.manifest_hash == semantic_hash
        assert (
            restored_semantic.approval_policy["semantic_auto_approval"]
            == semantic_policy
        )
        candidate = reopened.authority_manifests.semantic_auto_approval_candidate(
            semantic_pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/q1.txt",
            rights=["read"],
        )
        assert candidate is not None
        assert candidate["manifest_sha256"] == semantic_hash
    finally:
        reopened.close()


def test_semantic_auto_approval_candidate_requires_exact_canonical_resource() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            goal="exact candidate only",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": _policy(_rule())
                }
            },
        )
        assert runtime.authority_manifests.semantic_auto_approval_candidate(
            pid,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/*",
            rights=["read"],
        ) is None
        with pytest.raises(ValidationError, match="canonical"):
            runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.*",
                resource="filesystem:workspace:reports/q1.txt",
                rights=["read"],
            )
    finally:
        runtime.close()
