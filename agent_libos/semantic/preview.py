from __future__ import annotations

import codecs
import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from agent_libos.models import AuthorityRisk
from agent_libos.models.semantic import (
    SemanticApprovalArgumentKind,
    SemanticApprovalArgumentProjectionV1,
    SemanticApprovalGitReferenceV1,
    SemanticPreviewRisk,
)
from agent_libos.semantic.ontology import DEFAULT_ACTION_ONTOLOGY
from agent_libos.semantic.projection import LocalDlpAccumulator
from agent_libos.utils.serde import dumps, to_jsonable


_SAFE_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SAFE_OPTION_RE = re.compile(r"^(?P<option>--?[A-Za-z][A-Za-z0-9_-]{0,63})(?:=.*)?$")
_NUMBER_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_SAFE_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_SAFE_RESOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+/*=-]{0,511}$")
_SAFE_GIT_REFERENCE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/@:+~^{}-]{0,255}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXECUTABLES = frozenset(
    {
        "bash",
        "cat",
        "echo",
        "git",
        "head",
        "ls",
        "make",
        "node",
        "npm",
        "pwd",
        "python",
        "python3",
        "pytest",
        "rg",
        "sed",
        "sh",
        "tail",
        "uv",
        "zsh",
    }
)
_SAFE_SUBCOMMANDS_BY_EXECUTABLE = {
    "git": frozenset(
        {
            "add",
            "branch",
            "checkout",
            "clean",
            "commit",
            "diff",
            "fetch",
            "log",
            "merge",
            "pull",
            "push",
            "rebase",
            "remote",
            "reset",
            "restore",
            "show",
            "status",
            "switch",
            "tag",
        }
    ),
    "npm": frozenset({"ci", "install", "run", "test"}),
    "uv": frozenset({"build", "lock", "run", "sync"}),
}
_RISK_RANK = {
    SemanticPreviewRisk.LOW: 0,
    SemanticPreviewRisk.MEDIUM: 1,
    SemanticPreviewRisk.HIGH: 2,
    SemanticPreviewRisk.CRITICAL: 3,
}
_ONTOLOGY_RISK = {
    AuthorityRisk.HARMLESS: SemanticPreviewRisk.LOW,
    AuthorityRisk.LOW: SemanticPreviewRisk.LOW,
    AuthorityRisk.MEDIUM: SemanticPreviewRisk.MEDIUM,
    AuthorityRisk.HIGH: SemanticPreviewRisk.HIGH,
    AuthorityRisk.DESTRUCTIVE: SemanticPreviewRisk.CRITICAL,
}
_RIGHT_RISK = {
    "read": SemanticPreviewRisk.LOW,
    "diff": SemanticPreviewRisk.LOW,
    "materialize": SemanticPreviewRisk.MEDIUM,
    "write": SemanticPreviewRisk.HIGH,
    "execute": SemanticPreviewRisk.HIGH,
    "link": SemanticPreviewRisk.HIGH,
    "delete": SemanticPreviewRisk.CRITICAL,
    "grant": SemanticPreviewRisk.CRITICAL,
    "revoke": SemanticPreviewRisk.CRITICAL,
    "approve": SemanticPreviewRisk.CRITICAL,
    "admin": SemanticPreviewRisk.CRITICAL,
}
_REDACTED = "<redacted>"
_GIT_BOOLEAN_FACTS = frozenset(
    {
        "amend",
        "create",
        "delete",
        "deletes_paths",
        "detach",
        "directories",
        "force",
        "force_with_lease",
        "ignored",
        "include_untracked",
        "index",
        "prune",
        "reinstate_index",
        "staged",
        "worktree",
    }
)
_GIT_ENUM_FACTS: dict[str, frozenset[str]] = {
    "action": frozenset(
        {
            "apply",
            "clear",
            "create",
            "delete",
            "drop",
            "list",
            "pop",
            "push",
            "remove",
            "rename",
            "save",
            "show",
        }
    ),
    "mode": frozenset({"hard", "keep", "merge", "mixed", "soft"}),
    "abort_kind": frozenset({"cherry_pick", "merge", "rebase", "revert"}),
    "decision": frozenset({"approve", "comment", "request_changes"}),
    "integration": frozenset({"abort", "cherry_pick", "merge", "rebase", "revert"}),
    "kind": frozenset({"all", "branches", "pull_requests", "remotes", "tags"}),
    "scope": frozenset({"range", "staged", "worktree"}),
    "status": frozenset({"closed", "merged", "open"}),
    "strategy": frozenset({"fast_forward", "ff_only", "merge", "rebase", "squash"}),
}
_GIT_COUNT_FACTS = frozenset(
    {
        "body_bytes",
        "candidate_count",
        "limit",
        "max_bytes",
        "message_bytes",
        "patch_bytes",
        "path_count",
        "preview_bytes",
        "scope_count",
        "stash_index",
    }
)
_GIT_DIGEST_FACTS = (
    "body_sha256",
    "candidate_manifest_sha256",
    "git_config_sha256",
    "git_remote_fingerprint",
    "git_remote_refs_sha256",
    "git_url_fingerprint",
    "message_sha256",
    "patch_sha256",
    "preview_sha256",
    "title_sha256",
)
_GIT_REFERENCE_KEYS = (
    "ref",
    "base",
    "head",
    "branch",
    "remote",
    "local_ref",
    "remote_ref",
    "expected_remote_oid",
    "target",
    "start",
    "new_branch",
    "new_name",
    "tag",
    "managed_worktree_id",
    "source",
    "base_ref",
    "head_ref",
    "pr_id",
    "patch_oid",
    "git_remote_ref",
    "git_old_oid",
    "git_remote",
    "base_oid",
    "head_oid",
    "index_oid",
)


def host_preview_risk(
    action_id: str,
    rights: Sequence[str],
) -> SemanticPreviewRisk:
    """Return a risk composed only from immutable Host ontology facts.

    Request context, sandbox profiles, tool prose, and model findings are not
    consulted.  Unknown actions or rights are deliberately critical so a new
    operation cannot inherit a misleading low-risk presentation.
    """

    action = DEFAULT_ACTION_ONTOLOGY.resolve(action_id)
    selected = (
        SemanticPreviewRisk.CRITICAL
        if action is None
        else _ONTOLOGY_RISK[action.risk]
    )
    for right in rights:
        candidate = _RIGHT_RISK.get(right, SemanticPreviewRisk.CRITICAL)
        if _RISK_RANK[candidate] > _RISK_RANK[selected]:
            selected = candidate
    return selected


def build_host_argument_projection(
    *,
    action_id: str,
    resource: str,
    context: Mapping[str, Any],
) -> SemanticApprovalArgumentProjectionV1:
    """Project exact Host operation arguments without retaining payload text."""

    raw_operation = context.get("operation")
    operation = (
        action_id.rsplit(".", 1)[1]
        if raw_operation == action_id
        else raw_operation
    )
    if type(operation) is not str or _SAFE_OPERATION_RE.fullmatch(operation) is None:
        raise ValueError("semantic approval Host operation is malformed")
    kind = _argument_kind(action_id)
    if kind is SemanticApprovalArgumentKind.FILESYSTEM:
        return _filesystem_projection(operation, context, resource=resource)
    if kind is SemanticApprovalArgumentKind.SHELL:
        return _shell_projection(operation, context)
    if kind is SemanticApprovalArgumentKind.GIT:
        return _git_projection(operation, context)
    if kind is SemanticApprovalArgumentKind.JSONRPC:
        return _jsonrpc_projection(operation, context)
    if kind is SemanticApprovalArgumentKind.MCP:
        return _mcp_projection(operation, context)
    return SemanticApprovalArgumentProjectionV1(kind=kind, operation=operation)


def build_host_resource_projection(
    *,
    resource: str,
    action_id: str,
    sensitivity: str,
) -> tuple[str, str]:
    """Return a bounded public display token and the exact resource digest."""

    selected = _required_text(resource, "semantic approval resource", 65_536)
    digest = _sha256_text(selected)
    if (
        sensitivity not in {"public", "normal"}
        or len(selected) > 512
        or not _resource_display_allowed(selected, action_id=action_id)
        or _has_unsafe_unicode(selected)
        or _contains_credential_data(selected)
    ):
        return _REDACTED, digest
    return selected, digest


def _filesystem_projection(
    operation: str,
    context: Mapping[str, Any],
    *,
    resource: str,
) -> SemanticApprovalArgumentProjectionV1:
    path = context.get("path")
    if path is None:
        path = resource
    path = _required_text(path, "filesystem path identity", 65_536)
    content_sha256 = context.get("content_sha256")
    content_bytes = context.get("content_bytes")
    if content_sha256 is not None and not _is_sha256(content_sha256):
        raise ValueError("filesystem content digest is malformed")
    if content_bytes is not None and not _is_count(content_bytes):
        raise ValueError("filesystem content size is malformed")
    read_max_bytes = context.get("max_bytes")
    if read_max_bytes is not None and not _is_count(read_max_bytes):
        raise ValueError("filesystem read size is malformed")
    entry_limit = context.get("limit")
    if entry_limit is not None and not _is_count(entry_limit):
        raise ValueError("filesystem entry limit is malformed")
    text_encoding = _optional_encoding(context.get("encoding"))
    expected_content_sha256 = context.get("expected_content_sha256")
    if (
        expected_content_sha256 is not None
        and expected_content_sha256 != "missing"
        and not _is_sha256(expected_content_sha256)
    ):
        raise ValueError("filesystem expected content digest is malformed")
    recorded_path_sha256 = context.get("path_sha256")
    if recorded_path_sha256 is not None and not _is_sha256(recorded_path_sha256):
        raise ValueError("filesystem path projection digest is malformed")
    if (
        context.get("path") is not None
        and recorded_path_sha256 is not None
        and recorded_path_sha256 != _sha256_text(path)
    ):
        raise ValueError("filesystem path projection digest is stale")
    return SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.FILESYSTEM,
        operation=operation,
        path_sha256=recorded_path_sha256 or _sha256_text(path),
        content_sha256=content_sha256,
        content_bytes=content_bytes,
        read_max_bytes=read_max_bytes,
        entry_limit=entry_limit,
        text_encoding=text_encoding,
        expected_content_sha256=expected_content_sha256,
        overwrite=_optional_bool(context, "overwrite"),
        parents=_optional_bool(context, "parents"),
        exist_ok=_optional_bool(context, "exist_ok"),
        recursive=_optional_bool(context, "recursive"),
        missing_ok=_optional_bool(context, "missing_ok"),
    )


def _shell_projection(
    operation: str,
    context: Mapping[str, Any],
) -> SemanticApprovalArgumentProjectionV1:
    raw_argv = context.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or any(type(item) is not str for item in raw_argv)
    ):
        raise ValueError("shell argv is malformed")
    derived_argv_sha256 = hashlib.sha256("\0".join(raw_argv).encode("utf-8")).hexdigest()
    recorded_argv_sha256 = context.get("argv_sha256")
    if recorded_argv_sha256 is not None and not _is_sha256(recorded_argv_sha256):
        raise ValueError("shell argv digest is malformed")
    if recorded_argv_sha256 is not None and recorded_argv_sha256 != derived_argv_sha256:
        raise ValueError("shell argv digest is stale")
    argv_sha256 = derived_argv_sha256
    cwd = context.get("cwd", context.get("working_directory"))
    cwd = _required_text(cwd, "shell cwd", 4_096)
    safe_cwd = _safe_cwd(cwd, workspace_root=context.get("workspace_root"))
    executable = raw_argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    display_argv = tuple(
        _safe_argv_item(item, index=index, executable=executable)
        for index, item in enumerate(raw_argv[:16])
    )
    return SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.SHELL,
        operation=operation,
        display_argv=display_argv,
        argv_count=len(raw_argv),
        argv_truncated=len(raw_argv) > len(display_argv),
        argv_sha256=argv_sha256,
        safe_cwd=safe_cwd,
        cwd_sha256=_sha256_text(cwd),
        timeout_seconds=_optional_timeout(
            context.get("timeout_s", context.get("startup_timeout_s"))
        ),
        continuous_session=_optional_bool(context, "continuous_session"),
        network_access=_optional_bool(context, "network"),
    )


def _git_projection(
    operation: str,
    context: Mapping[str, Any],
) -> SemanticApprovalArgumentProjectionV1:
    worktree_id, worktree_id_sha256 = _identity_projection(
        context.get("worktree_id", "main"),
        "Git worktree id",
    )
    state = context.get("target_state_version", context.get("expected_state_token"))
    state_sha256 = None
    if state is not None:
        if type(state) is not str or len(state) > 2_048:
            raise ValueError("Git repository state is malformed")
        state_sha256 = hashlib.sha256(
            dumps(to_jsonable(state)).encode("utf-8")
        ).hexdigest()
    path_sha256 = context.get("paths_sha256", context.get("path_sha256"))
    if path_sha256 is not None and not _is_sha256(path_sha256):
        raise ValueError("Git path projection digest is malformed")
    source_args_sha256 = context.get("source_args_sha256")
    if source_args_sha256 is not None and not _is_sha256(source_args_sha256):
        raise ValueError("Git source argument digest is malformed")
    references: list[SemanticApprovalGitReferenceV1] = []
    for key in _GIT_REFERENCE_KEYS:
        value = context.get(key)
        if value is None:
            continue
        references.append(_git_reference_projection(key, value))
    references.sort(key=lambda item: item.role)
    if len(references) > 16:
        raise ValueError("Git reference projection exceeds its item budget")
    return SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.GIT,
        operation=operation,
        path_sha256=path_sha256,
        worktree_id=worktree_id,
        worktree_id_sha256=worktree_id_sha256,
        repository_state_sha256=state_sha256,
        source_args_sha256=source_args_sha256,
        git_references=tuple(references),
        git_fact_tokens=_git_fact_tokens(context),
    )


def _jsonrpc_projection(
    operation: str,
    context: Mapping[str, Any],
) -> SemanticApprovalArgumentProjectionV1:
    payload_sha256 = context.get("params_sha256")
    if not _is_sha256(payload_sha256):
        raise ValueError("JSON-RPC parameters digest is malformed")
    endpoint_id, endpoint_id_sha256 = _identity_projection(
        context.get("endpoint_id"),
        "JSON-RPC endpoint id",
    )
    method_id, method_id_sha256 = _identity_projection(
        context.get("method_id"),
        "JSON-RPC method id",
    )
    registry_spec_sha256, registry_generation = _registry_projection(context)
    return SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.JSONRPC,
        operation=operation,
        endpoint_id=endpoint_id,
        endpoint_id_sha256=endpoint_id_sha256,
        method_id=method_id,
        method_id_sha256=method_id_sha256,
        registry_spec_sha256=registry_spec_sha256,
        registry_generation=registry_generation,
        payload_sha256=payload_sha256,
    )


def _mcp_projection(
    operation: str,
    context: Mapping[str, Any],
) -> SemanticApprovalArgumentProjectionV1:
    payload_sha256 = context.get("arguments_sha256")
    if not _is_sha256(payload_sha256):
        raise ValueError("MCP arguments digest is malformed")
    server_id, server_id_sha256 = _identity_projection(
        context.get("server_id"),
        "MCP server id",
    )
    tool_id, tool_id_sha256 = _identity_projection(
        context.get("tool_id"),
        "MCP tool id",
    )
    registry_spec_sha256, registry_generation = _registry_projection(context)
    return SemanticApprovalArgumentProjectionV1(
        kind=SemanticApprovalArgumentKind.MCP,
        operation=operation,
        server_id=server_id,
        server_id_sha256=server_id_sha256,
        tool_id=tool_id,
        tool_id_sha256=tool_id_sha256,
        registry_spec_sha256=registry_spec_sha256,
        registry_generation=registry_generation,
        payload_sha256=payload_sha256,
    )


def _safe_argv_item(value: str, *, index: int, executable: str) -> str:
    if _has_unsafe_unicode(value):
        return "<redacted>"
    if index == 0:
        basename = value.replace("\\", "/").rsplit("/", 1)[-1]
        return (
            basename
            if basename in _SAFE_EXECUTABLES and not _contains_sensitive_data(basename)
            else "<redacted>"
        )
    if _contains_sensitive_data(value):
        return "<redacted>"
    if index == 1 and value in _SAFE_SUBCOMMANDS_BY_EXECUTABLE.get(
        executable,
        frozenset(),
    ):
        return value
    option = _SAFE_OPTION_RE.fullmatch(value)
    if option is not None:
        selected = option.group("option")
        return selected if "=" not in value else f"{selected}=<redacted>"
    if _NUMBER_RE.fullmatch(value) is not None:
        return "<number>"
    return "<redacted>"


def _argument_kind(action_id: str) -> SemanticApprovalArgumentKind:
    if action_id == "pty.spawn":
        return SemanticApprovalArgumentKind.SHELL
    prefix = action_id.split(".", 1)[0]
    try:
        return SemanticApprovalArgumentKind(prefix)
    except ValueError:
        return SemanticApprovalArgumentKind.OTHER


def _safe_cwd(value: str, *, workspace_root: Any) -> str | None:
    if value == ".":
        return "."
    if type(workspace_root) is str and value == workspace_root:
        return "<workspace>"
    return None


def _identity_projection(value: Any, label: str) -> tuple[str, str]:
    selected = _required_text(value, label, 65_536)
    if (
        _SAFE_IDENTITY_RE.fullmatch(selected) is None
        or _has_unsafe_unicode(selected)
    ):
        raise ValueError(f"{label} is not safe for an approval projection")
    digest = _sha256_text(selected)
    display = (
        selected
        if len(selected) <= 256 and not _contains_sensitive_data(selected)
        else _REDACTED
    )
    return display, digest


def _git_fact_tokens(context: Mapping[str, Any]) -> tuple[str, ...]:
    facts: list[str] = []
    for key in sorted(_GIT_BOOLEAN_FACTS):
        value = context.get(key)
        if value is None:
            continue
        if type(value) is not bool:
            raise ValueError(f"Git {key} fact is malformed")
        facts.append(f"{key}={'true' if value else 'false'}")
    for key in sorted(_GIT_ENUM_FACTS):
        value = context.get(key)
        if value is None:
            continue
        if type(value) is not str or value not in _GIT_ENUM_FACTS[key]:
            raise ValueError(f"Git {key} fact is malformed")
        facts.append(f"{key}={value}")
    for key in sorted(_GIT_COUNT_FACTS):
        value = context.get(key)
        if value is None:
            continue
        if not _is_count(value):
            raise ValueError(f"Git {key} fact is malformed")
        facts.append(f"{key}={value}")
    for key in sorted(_GIT_DIGEST_FACTS):
        value = context.get(key)
        if value is None:
            continue
        if not _is_sha256(value):
            raise ValueError(f"Git {key} digest is malformed")
        facts.append(f"{key}={value}")
    return tuple(sorted(facts))


def _git_reference_projection(
    role: str,
    value: Any,
) -> SemanticApprovalGitReferenceV1:
    selected = _required_text(value, f"Git {role}", 4_096)
    digest = _sha256_text(selected)
    display = (
        selected
        if len(selected) <= 256
        and _SAFE_GIT_REFERENCE_RE.fullmatch(selected) is not None
        and not _contains_credential_data(selected)
        else _REDACTED
    )
    return SemanticApprovalGitReferenceV1(
        role=role,
        display=display,
        sha256=digest,
    )


def _optional_encoding(value: Any) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or _has_unsafe_unicode(value)
    ):
        raise ValueError("filesystem text encoding is malformed")
    try:
        selected = codecs.lookup(value).name
    except LookupError as exc:
        raise ValueError("filesystem text encoding is malformed") from exc
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", selected) is None:
        raise ValueError("filesystem text encoding is malformed")
    return selected


def _registry_projection(
    context: Mapping[str, Any],
) -> tuple[str | None, int | None]:
    digest = context.get("registry_spec_sha256")
    generation = context.get("registry_generation")
    if not _is_sha256(digest) or not _is_count(generation):
        raise ValueError("provider registry binding is malformed")
    return digest, generation


def _contains_sensitive_data(value: str) -> bool:
    digest = _sha256_text(value)
    detector = LocalDlpAccumulator(input_sha256=digest)
    detector.scan(value)
    return bool(detector.findings)


def _contains_credential_data(value: str) -> bool:
    digest = _sha256_text(value)
    detector = LocalDlpAccumulator(input_sha256=digest)
    detector.scan(value)
    return any(finding.category.value == "credential" for finding in detector.findings)


def _resource_display_allowed(value: str, *, action_id: str) -> bool:
    if _SAFE_RESOURCE_RE.fullmatch(value) is None:
        return False
    if not action_id.startswith("filesystem."):
        return True
    prefix = "filesystem:workspace:"
    if not value.startswith(prefix):
        return False
    relative = value[len(prefix):]
    return bool(relative) and not relative.startswith("/") and all(
        segment not in {"", ".", ".."}
        for segment in relative.split("/")
    )


def _has_unsafe_unicode(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    )


def _required_text(value: Any, label: str, maximum: int) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is malformed")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 2**53 - 1


def _optional_bool(context: Mapping[str, Any], key: str) -> bool | None:
    value = context.get(key)
    if value is not None and type(value) is not bool:
        raise ValueError(f"semantic approval {key} is malformed")
    return value


def _optional_timeout(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise ValueError("semantic approval timeout is malformed")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0 or selected >= 10**12:
        raise ValueError("semantic approval timeout is malformed")
    rendered = format(selected, ".9f").rstrip("0").rstrip(".")
    return rendered


__all__ = [
    "build_host_argument_projection",
    "build_host_resource_projection",
    "host_preview_risk",
]
