from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_libos.models.base import OID, StrEnum


class GitErrorCode(StrEnum):
    GIT_UNAVAILABLE = "git_unavailable"
    UNSUPPORTED_GIT_VERSION = "unsupported_git_version"
    NOT_REPOSITORY = "not_repository"
    UNSAFE_REPOSITORY = "unsafe_repository"
    REPOSITORY_BUSY = "repository_busy"
    STALE_STATE = "stale_state"
    INVALID_REF = "invalid_ref"
    INVALID_PATH = "invalid_path"
    DIRTY_WORKTREE = "dirty_worktree"
    CONFLICT = "conflict"
    IDENTITY_MISSING = "identity_missing"
    UNSAFE_CONFIG = "unsafe_repository_config"
    AUTH_REQUIRED = "auth_required"
    NON_FAST_FORWARD = "non_fast_forward"
    REMOTE_REJECTED = "remote_rejected"
    OUTPUT_TOO_LARGE = "output_too_large"
    TIMEOUT = "timeout"
    UNKNOWN_EFFECT = "unknown_effect"
    COMMAND_FAILED = "command_failed"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    UNSUPPORTED = "unsupported"


class GitStatusKind(StrEnum):
    TRACKED = "tracked"
    RENAMED = "renamed"
    UNMERGED = "unmerged"
    UNTRACKED = "untracked"
    IGNORED = "ignored"


class GitPullRequestStatus(StrEnum):
    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class GitReviewDecision(StrEnum):
    COMMENT = "comment"
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"


@dataclass(frozen=True, slots=True)
class GitPath:
    """Byte-preserving repository path exposed through JSON-safe fields."""

    display: str
    path_b64: str
    lossy: bool = False


@dataclass(frozen=True, slots=True)
class GitStateToken:
    token: str
    repository_identity: str
    worktree_id: str
    head_ref: str | None
    head_oid: str | None
    index_sha256: str
    config_sha256: str
    refs_sha256: str
    worktrees_sha256: str
    pull_requests_sha256: str
    worktree_sha256: str


@dataclass(frozen=True, slots=True)
class GitRepositoryInfo:
    repository_id: str
    worktree_id: str
    root: str
    git_dir: str
    common_dir: str
    object_format: str
    bare: bool
    linked_worktree: bool
    git_version: str
    state: GitStateToken


@dataclass(frozen=True, slots=True)
class GitStatusEntry:
    path: GitPath
    kind: GitStatusKind
    index_status: str
    worktree_status: str
    original_path: GitPath | None = None
    submodule: str | None = None
    head_mode: str | None = None
    index_mode: str | None = None
    worktree_mode: str | None = None
    head_oid: str | None = None
    index_oid: str | None = None


@dataclass(frozen=True, slots=True)
class GitStatusResult:
    repository_id: str
    worktree_id: str
    branch: str | None
    upstream: str | None
    ahead: int
    behind: int
    head_oid: str | None
    entries: list[GitStatusEntry]
    state: GitStateToken
    truncated: bool
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GitDiffResult:
    repository_id: str
    worktree_id: str
    scope: str
    base_oid: str | None
    head_oid: str | None
    patch: str
    patch_b64: str
    changed_paths: list[GitPath]
    state: GitStateToken
    truncated: bool
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GitRef:
    name: str
    short_name: str
    kind: str
    oid: str
    symbolic_target: str | None = None
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None


@dataclass(frozen=True, slots=True)
class GitCommit:
    oid: str
    parents: list[str]
    author_name: str
    author_email: str
    authored_at: str
    committer_name: str
    committer_email: str
    committed_at: str
    subject: str
    body: str = ""


@dataclass(frozen=True, slots=True)
class GitRemoteInfo:
    name: str
    fetch_url: str
    push_url: str
    fetch_url_sha256: str
    push_url_sha256: str
    fetch_refspecs: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GitWorktreeInfo:
    worktree_id: str
    path: str
    head_oid: str | None
    branch: str | None
    detached: bool
    bare: bool
    locked: bool
    prunable: bool
    managed: bool


@dataclass(frozen=True, slots=True)
class GitOperationResult:
    operation: str
    repository_id: str
    worktree_id: str
    before: GitStateToken
    after: GitStateToken
    changed_paths: list[GitPath] = field(default_factory=list)
    created_oid: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GitPatchArtifact:
    oid: OID
    repository_id: str
    worktree_id: str
    base_oid: str | None
    head_oid: str | None
    index_oid: str
    patch_sha256: str
    bytes: int
    changed_paths: list[GitPath]
    state: GitStateToken


@dataclass(frozen=True, slots=True)
class GitPullRequestReview:
    review_id: str
    actor: str
    decision: GitReviewDecision
    body_sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class GitPullRequest:
    pr_id: str
    repository_id: str
    title: str
    body: str
    base_ref: str
    base_oid: str
    head_ref: str
    head_oid: str
    patch_sha256: str
    status: GitPullRequestStatus
    created_by: str
    created_at: str
    updated_at: str
    reviews: list[GitPullRequestReview] = field(default_factory=list)
    merged_oid: str | None = None
    state: GitStateToken | None = None
    truncated: bool = False
    bytes: int = 0
    sha256: str = ""


__all__ = [
    "GitCommit",
    "GitDiffResult",
    "GitErrorCode",
    "GitOperationResult",
    "GitPatchArtifact",
    "GitPath",
    "GitPullRequest",
    "GitPullRequestReview",
    "GitPullRequestStatus",
    "GitRef",
    "GitRemoteInfo",
    "GitRepositoryInfo",
    "GitReviewDecision",
    "GitStateToken",
    "GitStatusEntry",
    "GitStatusKind",
    "GitStatusResult",
    "GitWorktreeInfo",
]
