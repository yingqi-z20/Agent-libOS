from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from agent_libos.models import (
    CanonicalApprovalPreviewV1,
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
    SemanticApprovalArgumentKind,
    SemanticApprovalArgumentProjectionV1,
    SemanticPreviewLabelsV1,
    SemanticPreviewRisk,
)
from agent_libos.semantic.preview import (
    build_host_argument_projection,
    build_host_resource_projection,
    host_preview_risk,
)


pytestmark = pytest.mark.security

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _labels() -> SemanticPreviewLabelsV1:
    return SemanticPreviewLabelsV1.from_data_labels(
        DataLabels(
            sensitivity=DataSensitivity.NORMAL,
            integrity=DataIntegrity.VERIFIED,
            trust_level=DataTrustLevel.TRUSTED,
        )
    )


def _filesystem_preview() -> CanonicalApprovalPreviewV1:
    return CanonicalApprovalPreviewV1(
        request_id="request-preview",
        revision=3,
        pid="pid-preview",
        action_id="filesystem.read",
        resource_display="<redacted>",
        resource_sha256=hashlib.sha256(
            b"filesystem:workspace:reports/report.txt"
        ).hexdigest(),
        rights=("read",),
        effect_id="effect_preview",
        canonical_args_sha256=_A,
        argument_projection=SemanticApprovalArgumentProjectionV1(
            kind=SemanticApprovalArgumentKind.FILESYSTEM,
            operation="read",
            path_sha256=_B,
            read_max_bytes=65_536,
        ),
        target_state_sha256=None,
        risk=SemanticPreviewRisk.LOW,
        source_labels=_labels(),
        expires_at=None,
    )


def test_preview_risk_uses_only_host_action_ontology_and_rights() -> None:
    assert host_preview_risk("filesystem.read", ("read",)) is SemanticPreviewRisk.LOW
    assert host_preview_risk("filesystem.write", ("write",)) is SemanticPreviewRisk.HIGH
    assert host_preview_risk("filesystem.delete", ("delete",)) is SemanticPreviewRisk.CRITICAL
    assert host_preview_risk("future.read", ("read",)) is SemanticPreviewRisk.CRITICAL
    assert host_preview_risk("filesystem.read", ("admin",)) is SemanticPreviewRisk.CRITICAL


def test_filesystem_projection_uses_resource_fallback_and_never_retains_content() -> None:
    sentinel = "SEMANTIC_PREVIEW_BODY_SECRET_SENTINEL_7f33"
    projection = build_host_argument_projection(
        action_id="filesystem.write",
        resource="filesystem:workspace:reports/report.txt",
        context={
            "operation": "write_text",
            "content_sha256": hashlib.sha256(sentinel.encode()).hexdigest(),
            "content_bytes": len(sentinel),
            "content_preview": sentinel,
            "risk": "harmless",
            "sandbox_profile": {"risk": "harmless"},
        },
    )

    encoded = json.dumps(projection.to_dict(), sort_keys=True)
    assert projection.operation == "write_text"
    assert projection.path_sha256 == hashlib.sha256(
        b"filesystem:workspace:reports/report.txt"
    ).hexdigest()
    assert projection.content_bytes == len(sentinel)
    assert sentinel not in encoded
    assert "content_preview" not in encoded


def test_resource_projection_keeps_safe_workspace_identity_and_redacts_secrets() -> None:
    safe = "filesystem:workspace:reports/report.txt"
    display, digest = build_host_resource_projection(
        resource=safe,
        action_id="filesystem.read",
        sensitivity="normal",
    )
    assert display == safe
    assert digest == hashlib.sha256(safe.encode()).hexdigest()

    sentinel = "filesystem:workspace:SECRET_SENTINEL_abcdefghijkl"
    redacted, redacted_digest = build_host_resource_projection(
        resource=sentinel,
        action_id="filesystem.read",
        sensitivity="normal",
    )
    assert redacted == "<redacted>"
    assert redacted_digest == hashlib.sha256(sentinel.encode()).hexdigest()
    high, _ = build_host_resource_projection(
        resource=safe,
        action_id="filesystem.read",
        sensitivity="confidential",
    )
    assert high == "<redacted>"


def test_filesystem_projection_rejects_stale_path_digest_and_normalizes_codec_alias() -> None:
    with pytest.raises(ValueError, match="stale"):
        build_host_argument_projection(
            action_id="filesystem.read",
            resource="filesystem:workspace:reports/report.txt",
            context={
                "operation": "read_text",
                "path": "reports/report.txt",
                "path_sha256": _A,
            },
        )
    for alias in ("utf-8", "utf 8", "utf@8"):
        projection = build_host_argument_projection(
            action_id="filesystem.read",
            resource="filesystem:workspace:reports/report.txt",
            context={"operation": "read_text", "encoding": alias},
        )
        assert projection.text_encoding == "utf-8"


@pytest.mark.parametrize("subcommand", ("status", "push", "reset", "clean"))
def test_shell_projection_distinguishes_host_catalog_subcommands_without_values(
    subcommand: str,
) -> None:
    sentinel = "SEMANTIC_PREVIEW_ARG_SECRET_SENTINEL_7f33"
    argv = ["/usr/bin/git", subcommand, "--token=" + sentinel, "customer123"]
    argv_sha256 = hashlib.sha256("\0".join(argv).encode()).hexdigest()
    projection = build_host_argument_projection(
        action_id="shell.run",
        resource="shell:git",
        context={
            "operation": "shell.run",
            "argv": argv,
            "argv_sha256": argv_sha256,
            "cwd": "/Users/customer/secret-project",
            "workspace_root": "/Users/customer/workspace",
        },
    )

    assert projection.operation == "run"
    assert projection.display_argv == (
        "git",
        subcommand,
        "<redacted>",
        "<redacted>",
    )
    assert projection.safe_cwd is None
    assert projection.argv_sha256 == argv_sha256
    assert sentinel not in json.dumps(projection.to_dict(), sort_keys=True)


def test_shell_projection_rejects_a_forged_recorded_argv_digest() -> None:
    with pytest.raises(ValueError, match="stale"):
        build_host_argument_projection(
            action_id="shell.run",
            resource="shell:git",
            context={
                "operation": "run",
                "argv": ["git", "status"],
                "argv_sha256": _A,
                "cwd": ".",
            },
        )


def test_pty_projection_accepts_zero_startup_timeout() -> None:
    argv = ["sh"]
    projection = build_host_argument_projection(
        action_id="pty.spawn",
        resource="shell:pty",
        context={
            "operation": "pty.spawn",
            "argv": argv,
            "argv_sha256": hashlib.sha256(b"sh").hexdigest(),
            "cwd": ".",
            "startup_timeout_s": 0,
            "continuous_session": True,
        },
    )
    assert projection.operation == "spawn"
    assert projection.timeout_seconds == "0"
    assert projection.continuous_session is True


def test_remote_git_and_unknown_projection_variants_are_closed() -> None:
    jsonrpc = build_host_argument_projection(
        action_id="jsonrpc.call",
        resource="jsonrpc:billing:get_invoice",
        context={
            "operation": "jsonrpc.call",
            "endpoint_id": "billing",
            "method_id": "get_invoice",
            "params_sha256": _A,
            "registry_spec_sha256": _B,
            "registry_generation": 7,
        },
    )
    mcp = build_host_argument_projection(
        action_id="mcp.call",
        resource="mcp:reports:read_report",
        context={
            "operation": "mcp.call",
            "server_id": "reports",
            "tool_id": "read_report",
            "arguments_sha256": _B,
            "registry_spec_sha256": _C,
            "registry_generation": 9,
        },
    )
    git = build_host_argument_projection(
        action_id="git.diff",
        resource="git:workspace",
        context={
            "operation": "diff",
            "worktree_id": "main",
            "paths_sha256": _C,
            "base": "refs/heads/main",
            "head": "refs/heads/review",
        },
    )
    other = build_host_argument_projection(
        action_id="future.read",
        resource="future:resource",
        context={"operation": "requester_prose"},
    )

    assert (jsonrpc.endpoint_id, jsonrpc.method_id, jsonrpc.payload_sha256) == (
        "billing",
        "get_invoice",
        _A,
    )
    assert (jsonrpc.registry_spec_sha256, jsonrpc.registry_generation) == (_B, 7)
    assert (mcp.server_id, mcp.tool_id, mcp.payload_sha256) == (
        "reports",
        "read_report",
        _B,
    )
    assert (mcp.registry_spec_sha256, mcp.registry_generation) == (_C, 9)
    assert git.operation == "diff" and git.worktree_id == "main"
    assert [(item.role, item.display) for item in git.git_references] == [
        ("base", "refs/heads/main"),
        ("head", "refs/heads/review"),
    ]
    assert other.kind is SemanticApprovalArgumentKind.OTHER
    assert other.operation == "requester_prose"


def test_git_projection_exposes_material_enums_flags_counts_and_bound_digests() -> None:
    common = {
        "worktree_id": "main",
        "target_state_version": _A,
        "source_args_sha256": _B,
        "scope_count": 3,
    }
    reset_soft = build_host_argument_projection(
        action_id="git.write",
        resource="git:workspace",
        context={"operation": "reset", "mode": "soft", **common},
    )
    reset_hard = build_host_argument_projection(
        action_id="git.write",
        resource="git:workspace",
        context={"operation": "reset", "mode": "hard", **common},
    )
    push = build_host_argument_projection(
        action_id="git.write",
        resource="git_remote:origin",
        context={
            "operation": "push",
            "delete": True,
            "force_with_lease": False,
            "local_ref": "refs/heads/main",
            "remote_ref": "refs/heads/release",
            **common,
        },
    )
    review = build_host_argument_projection(
        action_id="git.approve",
        resource="git_pr:workspace:pr_1",
        context={
            "operation": "review_pull_request",
            "decision": "request_changes",
            "body_sha256": _C,
            "body_bytes": 42,
            "pr_id": "pr_1",
            **common,
        },
    )

    assert "mode=soft" in reset_soft.git_fact_tokens
    assert "mode=hard" in reset_hard.git_fact_tokens
    assert reset_soft.git_fact_tokens != reset_hard.git_fact_tokens
    assert reset_soft.repository_state_sha256 == hashlib.sha256(
        json.dumps(_A, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert reset_soft.source_args_sha256 == _B
    assert {"delete=true", "force_with_lease=false"}.issubset(push.git_fact_tokens)
    assert {item.role: item.display for item in push.git_references} == {
        "local_ref": "refs/heads/main",
        "remote_ref": "refs/heads/release",
    }
    assert {"decision=request_changes", "body_bytes=42"}.issubset(
        review.git_fact_tokens
    )
    assert f"body_sha256={_C}" in review.git_fact_tokens
    assert [(item.role, item.display) for item in review.git_references] == [
        ("pr_id", "pr_1")
    ]
    assert review.source_args_sha256 == _B


def test_git_reference_projection_is_role_bound_readable_or_redacted() -> None:
    secret_ref = "refs/heads/ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
    projection = build_host_argument_projection(
        action_id="git.write",
        resource="git_remote:origin",
        context={
            "operation": "push",
            "worktree_id": "main",
            "local_ref": "refs/heads/main",
            "remote_ref": secret_ref,
        },
    )
    references = {item.role: item for item in projection.git_references}
    assert references["local_ref"].display == "refs/heads/main"
    assert references["remote_ref"].display == "<redacted>"
    assert references["remote_ref"].sha256 == hashlib.sha256(
        secret_ref.encode()
    ).hexdigest()
    assert secret_ref not in json.dumps(projection.to_dict(), sort_keys=True)

    wire = projection.to_dict()
    unknown_role = json.loads(json.dumps(wire))
    unknown_role["git_references"][0]["role"] = "unknown_ref"
    with pytest.raises(ValueError, match="role"):
        SemanticApprovalArgumentProjectionV1.from_dict(unknown_role)

    stale_display = json.loads(json.dumps(wire))
    stale_display["git_references"][0]["display"] = "refs/heads/release"
    with pytest.raises(ValueError, match="stale"):
        SemanticApprovalArgumentProjectionV1.from_dict(stale_display)

    duplicate_role = json.loads(json.dumps(wire))
    duplicate_role["git_references"][1]["role"] = duplicate_role["git_references"][0]["role"]
    with pytest.raises(ValueError, match="unique and ordered"):
        SemanticApprovalArgumentProjectionV1.from_dict(duplicate_role)

    reverse_order = json.loads(json.dumps(wire))
    reverse_order["git_references"].reverse()
    with pytest.raises(ValueError, match="unique and ordered"):
        SemanticApprovalArgumentProjectionV1.from_dict(reverse_order)


@pytest.mark.parametrize(
    "pollution",
    (
        {"argv_truncated": True},
        {"payload_sha256": _C},
        {"endpoint_id": "forged-endpoint"},
        {"worktree_id": "forged-worktree"},
    ),
)
def test_projection_from_dict_rejects_cross_variant_pollution(
    pollution: dict[str, object],
) -> None:
    wire = _filesystem_preview().argument_projection.to_dict()
    wire.update(pollution)
    with pytest.raises(ValueError):
        SemanticApprovalArgumentProjectionV1.from_dict(wire)


def test_preview_rejects_action_projection_substitution_and_line_controls() -> None:
    preview = _filesystem_preview()
    shell = SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.SHELL,
        operation="run",
        display_argv=("git", "status"),
        argv_count=2,
        argv_sha256=_A,
        safe_cwd=".",
        cwd_sha256=_B,
    )
    with pytest.raises(ValueError, match="does not match"):
        replace(preview, argument_projection=shell)
    for injection in ("\n", "\x1b", "\u202e", "\u2066", "\u2028", "\u2029", "\ud800"):
        with pytest.raises(ValueError):
            replace(
                preview,
                resource_display=f"filesystem:workspace:report{injection}forged",
            )


def test_projection_public_counts_match_javascript_safe_integer_bound() -> None:
    with pytest.raises(ValueError, match="bounded"):
        replace(
            _filesystem_preview().argument_projection,
            read_max_bytes=2**53,
        )
