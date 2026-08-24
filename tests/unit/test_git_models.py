from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import GitErrorCode, GitStatusKind
from agent_libos.models.exceptions import GitError, ValidationError
from agent_libos.primitives.git import GitPrimitive
from agent_libos.primitives.git_command_policy import (
    READ_ONLY_GIT_COMMANDS,
    harden_read_only_git_argv,
    trusted_git_read_operation,
    validate_and_normalize_raw_git,
)
from agent_libos.substrate import CommandMetrics, GitCommandResult, GitRepositoryLayout, GitRepositoryState
from agent_libos.tools.builtin.git import (
    GIT_TOOL_NAMES,
    GitDiffArgs,
    GitFetchArgs,
    GitPushArgs,
)


_SHA1_A = "a" * 40
_SHA1_B = "b" * 40
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _state(status: bytes, *, head_oid: str | None = _SHA1_A) -> GitRepositoryState:
    layout = GitRepositoryLayout(
        root=Path("/workspace"),
        git_dir=Path("/workspace/.git"),
        common_dir=Path("/workspace/.git"),
        object_format="sha1",
        linked_worktree=False,
        repository_id="repository-id",
        worktree_id="worktree-id",
        git_version="2.26.0",
    )
    return GitRepositoryState(
        layout=layout,
        head_ref="refs/heads/main" if head_oid is not None else None,
        head_oid=head_oid,
        index_sha256=_EMPTY_SHA256,
        config_sha256=_EMPTY_SHA256,
        refs_sha256=_EMPTY_SHA256,
        worktrees_sha256=_EMPTY_SHA256,
        pull_requests_sha256=_EMPTY_SHA256,
        worktree_sha256=_EMPTY_SHA256,
        status_porcelain=status,
        status_sha256=hashlib.sha256(status).hexdigest(),
    )


def test_porcelain_v2_parser_handles_rename_conflict_and_byte_path() -> None:
    status = (
        b"# branch.oid " + _SHA1_A.encode("ascii") + b"\0"
        b"# branch.head main\0"
        b"# branch.upstream origin/main\0"
        b"# branch.ab +2 -1\0"
        b"1 M. N... 100644 100644 100644 "
        + _SHA1_A.encode("ascii")
        + b" "
        + _SHA1_B.encode("ascii")
        + b" tracked.txt\0"
        b"2 R. N... 100644 100644 100644 "
        + _SHA1_A.encode("ascii")
        + b" "
        + _SHA1_B.encode("ascii")
        + b" R100 renamed.txt\0old.txt\0"
        b"u UU N... 100644 100644 100644 100644 "
        + _SHA1_A.encode("ascii")
        + b" "
        + _SHA1_B.encode("ascii")
        + b" "
        + _SHA1_A.encode("ascii")
        + b" conflict.txt\0"
        b"? byte-\xff.txt\0"
    )

    parsed = GitPrimitive._parse_status(_state(status), limit=3)

    assert parsed.branch == "main"
    assert parsed.upstream == "origin/main"
    assert (parsed.ahead, parsed.behind) == (2, 1)
    assert parsed.truncated
    assert [entry.kind for entry in parsed.entries] == [
        GitStatusKind.TRACKED,
        GitStatusKind.RENAMED,
        GitStatusKind.UNMERGED,
    ]
    assert parsed.entries[1].original_path is not None
    assert parsed.entries[1].original_path.display == "old.txt"

    complete = GitPrimitive._parse_status(_state(status), limit=10)
    assert complete.entries[-1].path.lossy
    assert GitPrimitive._decode_path(complete.entries[-1].path) == b"byte-\xff.txt"


@pytest.mark.parametrize(
    "value",
    [
        "../escape",
        "a/../escape",
        "/absolute",
        ".git/config",
        "directory/.GIT/index",
        "double//separator",
    ],
)
def test_git_path_validation_rejects_metadata_and_traversal(value: str) -> None:
    with pytest.raises(GitError) as exc_info:
        GitPrimitive._decode_path(value)
    assert exc_info.value.code == GitErrorCode.INVALID_PATH.value


def test_dash_prefixed_path_is_preserved_as_a_literal_pathspec() -> None:
    assert GitPrimitive._decode_path("-option") == b"-option"
    assert GitPrimitive._path_argv((b"-option",)) == ["-option"]


def test_git_diff_treats_empty_optional_refs_as_absent() -> None:
    parsed = GitDiffArgs(scope="worktree", base="", head="   ")

    assert parsed.base is None
    assert parsed.head is None


@pytest.mark.parametrize(
    "value",
    [
        "-branch",
        "branch..name",
        "branch@{1}",
        "branch.lock",
        "topic.lock/child",
        "HEAD",
        ".hidden",
        "name with space",
        "refs/heads/-option",
        "refs/meta/arbitrary",
    ],
)
def test_git_ref_validation_rejects_option_and_ambiguous_names(value: str) -> None:
    with pytest.raises(GitError) as exc_info:
        GitPrimitive._validate_ref_name(value, branch_only=not value.startswith("refs/"))
    assert exc_info.value.code == GitErrorCode.INVALID_REF.value


def test_git_state_token_changes_for_status_and_supports_unborn_head() -> None:
    clean = GitPrimitive._state_token(_state(b""))
    dirty = GitPrimitive._state_token(_state(b"? new.txt\0"))
    unborn = GitPrimitive._state_token(_state(b"", head_oid=None))

    assert len(clean.token) == 64
    assert clean.token != dirty.token
    assert clean.token != unborn.token
    assert unborn.head_oid is None


def test_non_fast_forward_error_mapping_reads_porcelain_stdout_without_leaking_it() -> None:
    stdout = b"!\trefs/heads/main:refs/heads/main\t[rejected] (non-fast-forward)\n"
    result = GitCommandResult(
        argv=("git", "push"),
        returncode=1,
        stdout=stdout,
        stderr=b"error: failed to push some refs\n",
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(b"error: failed to push some refs\n").hexdigest(),
        metrics=CommandMetrics(),
    )

    error = GitPrimitive._error_from_result(result, "push")

    assert error.code == GitErrorCode.NON_FAST_FORWARD.value
    assert "refs/heads/main" not in str(error)
    assert error.details["stdout_sha256"] == result.stdout_sha256


def test_git_tool_surface_is_exact_and_remote_schemas_reject_urls_or_argv() -> None:
    assert len(GIT_TOOL_NAMES) == 32
    assert len(set(GIT_TOOL_NAMES)) == 32
    assert set(GIT_TOOL_NAMES) == {
        "git_repository_info",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_blame",
        "git_list_refs",
        "git_list_remotes",
        "git_list_worktrees",
        "git_stage",
        "git_unstage",
        "git_commit",
        "git_restore",
        "git_branch",
        "git_switch",
        "git_tag",
        "git_integrate",
        "git_stash",
        "git_reset",
        "git_clean",
        "git_worktree",
        "git_create_patch",
        "git_apply_patch",
        "git_fetch",
        "git_pull",
        "git_push",
        "git_create_pull_request",
        "git_list_pull_requests",
        "git_inspect_pull_request",
        "git_review_pull_request",
        "git_merge_pull_request",
        "git_close_pull_request",
    }
    with pytest.raises(PydanticValidationError):
        GitFetchArgs(
            remote="origin",
            expected_state_token="0" * 64,
            url="https://example.test/secret",
        )
    with pytest.raises(PydanticValidationError):
        GitPushArgs(
            remote="origin",
            remote_ref="refs/heads/main",
            local_ref="refs/heads/main",
            expected_state_token="0" * 64,
            argv=["--force"],
        )


@pytest.mark.parametrize("command", sorted(READ_ONLY_GIT_COMMANDS))
@pytest.mark.parametrize("spelling", ["git", "GIT", "GiT.ExE"])
def test_legacy_git_reads_share_case_insensitive_exact_hardening(
    command: tuple[str, ...],
    spelling: str,
) -> None:
    requested = [spelling, *command[1:]]
    normalized = validate_and_normalize_raw_git(requested)
    hardened = harden_read_only_git_argv(requested)

    assert normalized == list(command)
    assert hardened[0] == "git"
    assert "--no-pager" in hardened
    assert "--no-optional-locks" in hardened
    assert trusted_git_read_operation(hardened, hardened_only=True) is not None
    if command[1] == "diff":
        assert "--no-textconv" in hardened
        assert "--no-ext-diff" in hardened


@pytest.mark.parametrize(
    "requested",
    [
        ["env", "-P", ".", "git", "push"],
        ["env", "git", "branch", "wrapper-created"],
        [r"C:\Program Files\Git\usr\bin\NoHuP.ExE", "printf"],
        ["/usr/bin/NiCe", "-n", "5", "printf"],
        ["SETSID.EXE", "--", "printf"],
        ["/usr/bin/stdbuf", "-o0", "printf"],
        ["Timeout.exe", "30", "printf"],
        ["env", "FOO=bar", "nohup", "printf"],
        ["env", "-Sprintf unsafe"],
        ["env", "--unknown-option", "printf"],
    ],
)
def test_shared_argv_policy_rejects_transparent_launcher_dispatch(
    requested: list[str],
) -> None:
    with pytest.raises(ValidationError, match="launcher dispatch"):
        validate_and_normalize_raw_git(requested)
    with pytest.raises(ValidationError, match="launcher dispatch"):
        harden_read_only_git_argv(requested)


@pytest.mark.parametrize(
    "requested",
    [
        ["env"],
        ["NOHUP.EXE"],
        ["/usr/bin/nice"],
        [r"C:\Windows\System32\SetSid.ExE"],
        ["stdbuf"],
        ["TIMEOUT"],
    ],
)
def test_shared_argv_policy_allows_standalone_launcher(
    requested: list[str],
) -> None:
    assert validate_and_normalize_raw_git(requested) == requested


def test_shared_argv_policy_preserves_direct_non_launcher_arguments() -> None:
    requested = ["printf", "git"]

    assert validate_and_normalize_raw_git(requested) == requested


def test_default_git_version_supports_config_scope_inspection() -> None:
    version = tuple(int(part) for part in DEFAULT_CONFIG.git.minimum_version.split("."))

    assert version >= (2, 26, 0)
