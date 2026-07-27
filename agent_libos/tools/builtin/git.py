from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import GitErrorCode
from agent_libos.models.exceptions import GitError
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolPolicy,
)
from agent_libos.utils.serde import to_jsonable

_GIT_DEFAULTS = DEFAULT_CONFIG.git
_STATE_TOKEN_PATTERN = r"^[0-9a-f]{64}$"
_EXPECTED_STATE_TOKEN_DESCRIPTION = (
    "Fresh token from the nested state.token field returned by a Git read for this repository/worktree. "
    "The mutation uses compare-and-swap and fails as stale if state changed; re-read and reconsider before retrying."
)


def _expected_state_token_field() -> Any:
    return Field(pattern=_STATE_TOKEN_PATTERN, description=_EXPECTED_STATE_TOKEN_DESCRIPTION)


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GitPathInput(_StrictArgs):
    """A UTF-8 display path or a byte-preserving token returned by a Git read."""

    path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative repository path. Set exactly one of path and "
            "path_b64; never copy both fields from a Git read result."
        ),
    )
    path_b64: str | None = Field(
        default=None,
        description=(
            "Base64 path token returned by a Git tool. Set exactly one of "
            "path_b64 and path; never copy both fields from a Git read result."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_path(self) -> "GitPathInput":
        if (self.path is None) == (self.path_b64 is None):
            raise ValueError("exactly one of path and path_b64 is required")
        return self


GitPathArgument = str | GitPathInput


class GitRepositoryInfoArgs(_StrictArgs):
    worktree_id: str = "main"


class GitStatusArgs(_StrictArgs):
    worktree_id: str = "main"
    limit: int | None = Field(default=None, gt=0)


class GitDiffArgs(_StrictArgs):
    scope: Literal["worktree", "staged", "range"] = "worktree"
    base: str | None = Field(
        default=None,
        description=(
            "Exact base ref/OID for range diffs. Set null for worktree or staged diffs; "
            "an empty string is treated as null."
        ),
    )
    head: str | None = Field(
        default=None,
        description=(
            "Exact head ref/OID for range diffs. Set null for worktree or staged diffs; "
            "an empty string is treated as null."
        ),
    )
    paths: list[GitPathArgument] = Field(default_factory=list)
    worktree_id: str = "main"
    max_bytes: int | None = Field(default=None, gt=0)

    @field_validator("base", "head", mode="before")
    @classmethod
    def _empty_ref_is_absent(cls, value: Any) -> Any:
        # Some structured-output providers populate nullable string fields with
        # an empty string. For non-range diffs that is unambiguously equivalent
        # to omission; range validation still rejects the resulting None.
        if isinstance(value, str) and not value.strip():
            return None
        return value


class GitLogArgs(_StrictArgs):
    ref: str | None = None
    limit: int | None = Field(default=None, gt=0)
    worktree_id: str = "main"


class GitShowArgs(_StrictArgs):
    ref: str
    worktree_id: str = "main"
    max_bytes: int | None = Field(default=None, gt=0)


class GitBlameArgs(_StrictArgs):
    path: GitPathArgument
    ref: str | None = None
    worktree_id: str = "main"
    max_bytes: int | None = Field(default=None, gt=0)


class GitListRefsArgs(_StrictArgs):
    kind: Literal["all", "branches", "tags", "remotes", "pull_requests"] = "all"
    limit: int = Field(default=200, gt=0)
    worktree_id: str = "main"


class GitListRemotesArgs(_StrictArgs):
    worktree_id: str = "main"


class GitListWorktreesArgs(_StrictArgs):
    pass


class GitPathsMutationArgs(_StrictArgs):
    paths: list[GitPathArgument] = Field(min_length=1)
    expected_state_token: str = _expected_state_token_field()
    worktree_id: str = "main"


class GitCommitArgs(_StrictArgs):
    message: str = Field(min_length=1, max_length=131_072)
    expected_state_token: str = _expected_state_token_field()
    amend: bool = Field(default=False, description="Replace the current tip commit instead of creating a new commit.")
    worktree_id: str = "main"


class GitRestoreArgs(GitPathsMutationArgs):
    staged: bool = Field(default=False, description="Restore the selected paths in the index.")
    worktree: bool = Field(
        default=True,
        description="Restore the selected paths in the worktree, discarding local content.",
    )
    source: str | None = Field(default=None, description="Exact source ref/OID; defaults to the index or HEAD by mode.")


class GitBranchArgs(_StrictArgs):
    operation: Literal["create", "delete", "rename"] = Field(
        validation_alias=AliasChoices("operation", "action"),
        serialization_alias="action",
    )
    name: str
    expected_state_token: str = _expected_state_token_field()
    start: str | None = None
    new_name: str | None = None
    force: bool = Field(
        default=False,
        description="Permit the requested create/delete/rename to overwrite safety checks.",
    )
    worktree_id: str = "main"


class GitSwitchArgs(_StrictArgs):
    target: str
    expected_state_token: str = _expected_state_token_field()
    create: bool = False
    start: str | None = None
    detach: bool = False
    force: bool = Field(
        default=False,
        description="Discard conflicting local changes when switching; requires destructive approval.",
    )
    worktree_id: str = "main"


class GitTagArgs(_StrictArgs):
    operation: Literal["create", "delete"] = Field(
        validation_alias=AliasChoices("operation", "action"),
        serialization_alias="action",
    )
    name: str
    expected_state_token: str = _expected_state_token_field()
    target: str | None = None
    message: str | None = Field(default=None, max_length=131_072)
    force: bool = Field(
        default=False,
        description="Replace an existing tag when creating; deleting is already destructive.",
    )
    worktree_id: str = "main"


class GitIntegrateArgs(_StrictArgs):
    operation: Literal["merge", "rebase", "cherry_pick", "revert", "abort"]
    expected_state_token: str = _expected_state_token_field()
    ref: str | None = None
    abort_kind: Literal["merge", "rebase", "cherry_pick", "revert"] | None = None
    worktree_id: str = "main"


class GitStashArgs(_StrictArgs):
    operation: Literal["push", "apply", "pop", "drop", "clear"] = Field(
        validation_alias=AliasChoices("operation", "action"),
        serialization_alias="action",
    )
    expected_state_token: str = _expected_state_token_field()
    index: int = Field(default=0, ge=0, le=100_000)
    include_untracked: bool = False
    reinstate_index: bool = False
    worktree_id: str = "main"


class GitResetArgs(_StrictArgs):
    target: str = Field(description="Exact validated ref/OID to move HEAD to.")
    expected_state_token: str = _expected_state_token_field()
    mode: Literal["soft", "mixed", "hard"] = Field(
        default="mixed",
        description="soft moves HEAD; mixed also resets the index; hard also discards worktree changes.",
    )
    worktree_id: str = "main"


class GitCleanArgs(_StrictArgs):
    expected_state_token: str = _expected_state_token_field()
    paths: list[GitPathArgument] = Field(
        default_factory=list,
        description="Optional literal path restriction; empty selects all eligible untracked paths.",
    )
    directories: bool = Field(default=False, description="Also remove eligible untracked directories.")
    ignored: bool = Field(default=False, description="Also remove ignored paths, increasing destructive scope.")
    worktree_id: str = "main"


class GitWorktreeArgs(_StrictArgs):
    operation: Literal["create", "remove"] = Field(
        validation_alias=AliasChoices("operation", "action"),
        serialization_alias="action",
    )
    expected_state_token: str = _expected_state_token_field()
    ref: str | None = None
    new_branch: str | None = None
    managed_worktree_id: str | None = Field(
        default=None,
        description="Existing managed worktree id required for remove; removal may discard that worktree's local state.",
    )


class GitCreatePatchArgs(_StrictArgs):
    scope: Literal["worktree", "staged", "range"] = "worktree"
    base: str | None = None
    head: str | None = None
    paths: list[GitPathArgument] = Field(default_factory=list)
    worktree_id: str = "main"


class GitApplyPatchArgs(_StrictArgs):
    patch_oid: str
    expected_state_token: str = _expected_state_token_field()
    index: bool = Field(default=False, description="Also apply the patch to the index instead of only the worktree.")
    worktree_id: str = "main"


class GitFetchArgs(_StrictArgs):
    remote: str
    expected_state_token: str = _expected_state_token_field()
    prune: bool = False
    worktree_id: str = "main"


class GitPullArgs(_StrictArgs):
    remote: str
    expected_state_token: str = _expected_state_token_field()
    branch: str | None = None
    strategy: Literal["ff_only", "merge", "rebase"] = "ff_only"
    worktree_id: str = "main"


class GitPushArgs(_StrictArgs):
    remote: str
    remote_ref: str = Field(
        description="Complete remote ref, restricted to refs/heads/... or refs/tags/...."
    )
    expected_state_token: str = _expected_state_token_field()
    local_ref: str | None = None
    delete: bool = Field(
        default=False,
        description=(
            "Delete remote_ref instead of updating it; local_ref must be omitted and "
            "force_with_lease_oid is still required. Requires destructive authority."
        ),
    )
    force_with_lease_oid: str | None = Field(
        default=None,
        description=(
            "Exact expected remote OID for a guarded force update or deletion; "
            "unrestricted force and unguarded deletion are unavailable."
        ),
    )
    worktree_id: str = "main"


class GitCreatePullRequestArgs(_StrictArgs):
    title: str = Field(min_length=1, max_length=4096)
    body: str = Field(default="", max_length=131_072)
    base_ref: str
    head_ref: str
    expected_state_token: str = _expected_state_token_field()


class GitListPullRequestsArgs(_StrictArgs):
    status: Literal["open", "merged", "closed"] | None = None
    limit: int = Field(default=100, gt=0)


class GitInspectPullRequestArgs(_StrictArgs):
    pr_id: str


class GitReviewPullRequestArgs(_StrictArgs):
    pr_id: str
    decision: Literal["comment", "approve", "request_changes"]
    body: str = Field(default="", max_length=131_072)
    expected_state_token: str = _expected_state_token_field()


class GitMergePullRequestArgs(_StrictArgs):
    pr_id: str
    expected_state_token: str = _expected_state_token_field()
    strategy: Literal["fast_forward", "merge", "squash"] = "fast_forward"
    worktree_id: str = "main"


class GitClosePullRequestArgs(_StrictArgs):
    pr_id: str
    expected_state_token: str = _expected_state_token_field()


_INVALID_CODES = {
    GitErrorCode.INVALID_REF.value,
    GitErrorCode.INVALID_PATH.value,
}
_UNSUPPORTED_CODES = {
    GitErrorCode.GIT_UNAVAILABLE.value,
    GitErrorCode.UNSUPPORTED_GIT_VERSION.value,
    GitErrorCode.UNSUPPORTED.value,
}
_DENIAL_CODES = {
    GitErrorCode.UNSAFE_REPOSITORY.value,
    GitErrorCode.UNSAFE_CONFIG.value,
}
_TRANSIENT_CODES = {
    GitErrorCode.REPOSITORY_BUSY.value,
    GitErrorCode.STALE_STATE.value,
    GitErrorCode.UNKNOWN_EFFECT.value,
}


def _git_tool_error(exc: GitError) -> ToolExecutionError:
    if exc.code in _INVALID_CODES:
        code = ToolErrorCode.VALIDATION_ERROR
    elif exc.code in _UNSUPPORTED_CODES:
        code = ToolErrorCode.UNSUPPORTED
    elif exc.code in _DENIAL_CODES:
        code = ToolErrorCode.PERMISSION_DENIED
    elif exc.code == GitErrorCode.TIMEOUT.value:
        code = ToolErrorCode.TIMEOUT
    elif exc.code == GitErrorCode.AUTH_REQUIRED.value:
        code = ToolErrorCode.PERMISSION_DENIED
    elif exc.code in _TRANSIENT_CODES:
        code = ToolErrorCode.TRANSIENT_ERROR
    else:
        code = ToolErrorCode.EXECUTION_ERROR
    return ToolExecutionError(
        str(exc),
        code=code,
        retryable=exc.retryable or exc.code in _TRANSIENT_CODES,
        details={
            "git_error_code": exc.code,
            "operation": exc.operation,
        },
    )


class _GitTool(SyncAgentTool[_StrictArgs]):
    method_name: ClassVar[str]
    tags = ["git"]

    def run(self, args: _StrictArgs, ctx: ToolContext) -> Any:
        if ctx.runtime is None or getattr(ctx.runtime, "git", None) is None:
            raise ToolExecutionError("Runtime Git boundary is unavailable.", code=ToolErrorCode.UNSUPPORTED)
        try:
            result = getattr(ctx.runtime.git, self.method_name)(
                pid=ctx.pid,
                **args.model_dump(exclude_none=True, by_alias=True),
            )
        except GitError as exc:
            raise _git_tool_error(exc) from exc
        return to_jsonable(result)


class _GitReadTool(_GitTool):
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"git.read"},
        timeout_s=_GIT_DEFAULTS.local_timeout_s,
    )
    tags = ["git", "inspect"]


class _GitMutationTool(_GitTool):
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"git.write", "filesystem.write"},
        timeout_s=_GIT_DEFAULTS.local_timeout_s,
    )
    tags = ["git", "mutation"]


class _GitRemoteTool(_GitMutationTool):
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"git.write", "git_remote.read", "git_remote.write"},
        timeout_s=_GIT_DEFAULTS.remote_timeout_s,
    )
    tags = ["git", "remote", "mutation"]


class _GitPullRequestTool(_GitMutationTool):
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"git.write", "git_pr.write"},
        timeout_s=_GIT_DEFAULTS.local_timeout_s,
    )
    tags = ["git", "pull_request"]


class GitRepositoryInfoTool(_GitReadTool):
    name = "git_repository_info"
    description = "Inspect the fixed Runtime workspace Git repository and return its current state token."
    args_schema = GitRepositoryInfoArgs
    method_name = "repository_info"


class GitStatusTool(_GitReadTool):
    name = "git_status"
    description = "Read byte-safe porcelain status and a state token for a Runtime Git worktree."
    args_schema = GitStatusArgs
    method_name = "status"


class GitDiffTool(_GitReadTool):
    name = "git_diff"
    description = (
        "Read a bounded, hardened Git patch for the worktree, index, or an exact commit range. "
        "Use null base/head for worktree or staged scope; range requires both exact refs/OIDs."
    )
    args_schema = GitDiffArgs
    method_name = "diff"


class GitLogTool(_GitReadTool):
    name = "git_log"
    description = "List bounded structured commits from an exact validated ref or OID."
    args_schema = GitLogArgs
    method_name = "log"


class GitShowTool(_GitReadTool):
    name = "git_show"
    description = "Inspect one commit with bounded, hardened root or per-parent patches."
    args_schema = GitShowArgs
    method_name = "show"


class GitBlameTool(_GitReadTool):
    name = "git_blame"
    description = "Read bounded porcelain blame for one literal repository path."
    args_schema = GitBlameArgs
    method_name = "blame"


class GitListRefsTool(_GitReadTool):
    name = "git_list_refs"
    description = "List bounded structured refs without accepting arbitrary Git patterns or argv."
    args_schema = GitListRefsArgs
    method_name = "list_refs"


class GitListRemotesTool(_GitReadTool):
    name = "git_list_remotes"
    description = "List existing configured remotes and their validated URL fingerprints."
    args_schema = GitListRemotesArgs
    method_name = "list_remotes"


class GitListWorktreesTool(_GitReadTool):
    name = "git_list_worktrees"
    description = "List the main and managed Git worktrees for the fixed repository."
    args_schema = GitListWorktreesArgs
    method_name = "list_worktrees"


class GitStageTool(_GitMutationTool):
    name = "git_stage"
    description = "Stage literal paths when both Git and filesystem authority permit it."
    args_schema = GitPathsMutationArgs
    method_name = "stage"


class GitUnstageTool(_GitMutationTool):
    name = "git_unstage"
    description = "Unstage literal paths using compare-and-swap repository state."
    args_schema = GitPathsMutationArgs
    method_name = "unstage"


class GitCommitTool(_GitMutationTool):
    name = "git_commit"
    description = "Commit staged content using trusted Host/repository identity; author overrides are unavailable."
    args_schema = GitCommitArgs
    method_name = "commit"


class GitRestoreTool(_GitMutationTool):
    name = "git_restore"
    description = "Restore literal paths; discarded content requires delete/admin authority and exact approval."
    args_schema = GitRestoreArgs
    method_name = "restore"


class GitBranchTool(_GitMutationTool):
    name = "git_branch"
    description = "Create, rename, or explicitly delete a strictly validated local branch."
    args_schema = GitBranchArgs
    method_name = "branch"


class GitSwitchTool(_GitMutationTool):
    name = "git_switch"
    description = "Switch the selected Runtime worktree to a validated branch or exact commit."
    args_schema = GitSwitchArgs
    method_name = "switch"


class GitTagTool(_GitMutationTool):
    name = "git_tag"
    description = "Create an unsigned local tag or explicitly delete a validated tag."
    args_schema = GitTagArgs
    method_name = "tag"


class GitIntegrateTool(_GitMutationTool):
    name = "git_integrate"
    description = "Run typed merge, rebase, cherry-pick, revert, or an explicit abort operation."
    args_schema = GitIntegrateArgs
    method_name = "integrate"


class GitStashTool(_GitMutationTool):
    name = "git_stash"
    description = "Operate on bounded local stash indexes through typed actions."
    args_schema = GitStashArgs
    method_name = "stash"


class GitResetTool(_GitMutationTool):
    name = "git_reset"
    description = "Reset to an exact resolved commit with mandatory destructive authorization."
    args_schema = GitResetArgs
    method_name = "reset"


class GitCleanTool(_GitMutationTool):
    name = "git_clean"
    description = "Preview then remove untracked paths with delete/admin authority and exact approval."
    args_schema = GitCleanArgs
    method_name = "clean"


class GitWorktreeTool(_GitMutationTool):
    name = "git_worktree"
    description = "Create a Runtime-named managed worktree or explicitly remove a known managed worktree."
    args_schema = GitWorktreeArgs
    method_name = "worktree"


class GitCreatePatchTool(_GitMutationTool):
    name = "git_create_patch"
    description = "Create a complete immutable CODE_PATCH Object with Git state and lineage metadata."
    args_schema = GitCreatePatchArgs
    method_name = "create_patch"
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"git.diff", "object.write"},
        timeout_s=_GIT_DEFAULTS.local_timeout_s,
    )
    tags = ["git", "patch", "object"]


class GitApplyPatchTool(_GitMutationTool):
    name = "git_apply_patch"
    description = "Validate and apply an immutable Git patch Object while propagating its data labels."
    args_schema = GitApplyPatchArgs
    method_name = "apply_patch"
    tags = ["git", "patch", "mutation"]


class GitFetchTool(_GitRemoteTool):
    name = "git_fetch"
    description = "Fetch from an existing validated remote; URLs and refspecs cannot be supplied."
    args_schema = GitFetchArgs
    method_name = "fetch"


class GitPullTool(_GitRemoteTool):
    name = "git_pull"
    description = "Fetch then integrate an existing remote branch, defaulting to fast-forward only."
    args_schema = GitPullArgs
    method_name = "pull"


class GitPushTool(_GitRemoteTool):
    name = "git_push"
    description = "Push an explicit local ref to an explicit remote ref; force requires an exact lease OID."
    args_schema = GitPushArgs
    method_name = "push"


class GitCreatePullRequestTool(_GitPullRequestTool):
    name = "git_create_pull_request"
    description = "Create a repository-local simulated pull request backed by immutable snapshot refs."
    args_schema = GitCreatePullRequestArgs
    method_name = "create_pull_request"


class GitListPullRequestsTool(_GitReadTool):
    name = "git_list_pull_requests"
    description = "List repository-local simulated pull requests and their current state token."
    args_schema = GitListPullRequestsArgs
    method_name = "list_pull_requests"
    tags = ["git", "pull_request", "inspect"]


class GitInspectPullRequestTool(_GitReadTool):
    name = "git_inspect_pull_request"
    description = "Inspect one repository-local simulated pull request and verify its snapshot refs."
    args_schema = GitInspectPullRequestArgs
    method_name = "inspect_pull_request"
    tags = ["git", "pull_request", "inspect"]


class GitReviewPullRequestTool(_GitPullRequestTool):
    name = "git_review_pull_request"
    description = "Record a comment, approval, or requested-changes review on a simulated pull request."
    args_schema = GitReviewPullRequestArgs
    method_name = "review_pull_request"


class GitMergePullRequestTool(_GitPullRequestTool):
    name = "git_merge_pull_request"
    description = "CAS-merge a simulated pull request by fast-forward, merge commit, or squash."
    args_schema = GitMergePullRequestArgs
    method_name = "merge_pull_request"


class GitClosePullRequestTool(_GitPullRequestTool):
    name = "git_close_pull_request"
    description = "Close an open simulated pull request without deleting its evidence."
    args_schema = GitClosePullRequestArgs
    method_name = "close_pull_request"


GIT_TOOL_TYPES: tuple[type[_GitTool], ...] = (
    GitRepositoryInfoTool,
    GitStatusTool,
    GitDiffTool,
    GitLogTool,
    GitShowTool,
    GitBlameTool,
    GitListRefsTool,
    GitListRemotesTool,
    GitListWorktreesTool,
    GitStageTool,
    GitUnstageTool,
    GitCommitTool,
    GitRestoreTool,
    GitBranchTool,
    GitSwitchTool,
    GitTagTool,
    GitIntegrateTool,
    GitStashTool,
    GitResetTool,
    GitCleanTool,
    GitWorktreeTool,
    GitCreatePatchTool,
    GitApplyPatchTool,
    GitFetchTool,
    GitPullTool,
    GitPushTool,
    GitCreatePullRequestTool,
    GitListPullRequestsTool,
    GitInspectPullRequestTool,
    GitReviewPullRequestTool,
    GitMergePullRequestTool,
    GitClosePullRequestTool,
)
GIT_TOOL_NAMES: tuple[str, ...] = tuple(tool_type.name for tool_type in GIT_TOOL_TYPES)


__all__ = [
    "GIT_TOOL_NAMES",
    "GIT_TOOL_TYPES",
    "GitApplyPatchTool",
    "GitBlameTool",
    "GitBranchTool",
    "GitCleanTool",
    "GitClosePullRequestTool",
    "GitCommitTool",
    "GitCreatePatchTool",
    "GitCreatePullRequestTool",
    "GitDiffTool",
    "GitFetchTool",
    "GitInspectPullRequestTool",
    "GitIntegrateTool",
    "GitListPullRequestsTool",
    "GitListRefsTool",
    "GitListRemotesTool",
    "GitListWorktreesTool",
    "GitLogTool",
    "GitMergePullRequestTool",
    "GitPathInput",
    "GitPullTool",
    "GitPushTool",
    "GitRepositoryInfoTool",
    "GitResetTool",
    "GitRestoreTool",
    "GitReviewPullRequestTool",
    "GitShowTool",
    "GitStageTool",
    "GitStashTool",
    "GitStatusTool",
    "GitSwitchTool",
    "GitTagTool",
    "GitUnstageTool",
    "GitWorktreeTool",
]
