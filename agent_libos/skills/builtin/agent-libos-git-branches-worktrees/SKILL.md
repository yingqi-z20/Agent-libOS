---
name: agent-libos-git-branches-worktrees
description: Manage validated local branches, tags, checkout targets, and structurally trusted Runtime-managed worktrees without accepting arbitrary host paths.
allowed-tools: git_branch git_switch git_tag git_worktree
---
# Manage Git topology and worktrees

Activate `agent-libos-git-inspection` first. These tools operate on local repository topology and selected managed checkouts; they do not fetch, push, merge, rebase, edit file content, initialize repositories, or accept an arbitrary destination path. Use exact refs/OIDs, byte-safe status evidence, and a fresh token from the worktree whose state the operation actually compares.

“Managed” means a worktree is structurally validated under the Host-configured managed root and shares the trusted common Git directory. It is not persistent proof that this Runtime instance originally created it. Select worktrees only by returned `worktree_id`; never pass their host `path` as a selector.

All tools here are side-effecting and non-idempotent. Ordinary successes return a `GitOperationResult` with post-state digest `after.token`, operation-specific `created_oid`, `details`, and affected/reconciliation paths. A state token is not consumed and need not change after a verified no-op. `created_oid` does not universally mean that a new Git object was created; interpret it with the operation and verify refs afterward. Capability and filesystem grants remain subject to data-flow policy and state revalidation.

## Tool guide

### `git_branch`

`git_branch` creates, deletes, or renames a strictly validated local branch without switching the selected worktree.

`name` and `new_name` are **short local branch names** such as `main` or `feature/x`, not `refs/heads/...` values copied from `git_list_refs`. Full refs or OIDs belong only in `start`. Passing a full ref as a branch name can address a different nested name rather than the listed branch.

- `operation="create"`: `name` is the new local branch. Optional `start` resolves to a commit; omission uses the selected worktree's current HEAD. `new_name` is irrelevant and must be omitted. `force=false` refuses an existing name; `force=true` permits updating the branch ref where Git's remaining checks allow it. Success returns the start commit in `created_oid`.
- `operation="delete"`: `name` is the existing branch. Omit `start` and `new_name`. `force=false` uses Git's merged/reachability safety check; `force=true` overrides that check. It cannot silently delete a branch checked out by a protected worktree merely because force was requested.
- `operation="rename"`: `name` is the old branch and `new_name` is required. Omit `start`. `force=false` refuses a conflicting destination; `force=true` permits replacement where Git allows it.

All variants require repository write. Delete, rename, and forced create are treated as destructive and additionally require repository delete/admin authority plus mandatory one-use Human approval. Branch metadata operations do not grant or require checkout filesystem access.

### `git_switch`

`git_switch` changes the selected `worktree_id` checkout. The modes are exclusive:

- Existing branch: `create=false`, `detach=false`, `target=<short local branch>`. Omit `start`. This resolves and checks out that branch.
- New branch: `create=true`, `detach=false`, `target=<new short branch name>`, optional `start=<exact ref/OID>`. Omission starts at current HEAD. This creates and checks out the branch. Do not copy a `refs/heads/...` row into `target`.
- Detached: `detach=true`, `create=false`, `target=<exact ref/OID>`. Omit `start`. This resolves a commit and checks it out detached.
- `create=true` with `detach=true` is invalid. A supplied `start` has meaning only for new-branch mode; do not send ignored fields.

Switch requires repository write and selected-worktree filesystem read/write/delete authority. With `force=false`, Git protects conflicting local changes but does not promise that every dirty configuration is rejected; inspect before switching. `force=true` is destructive: checkout can discard index and tracked worktree changes and remove untracked files/directories that block materialization. It requires repository delete/admin plus mandatory one-use approval. Never use force as a shortcut around an unexplained dirty status.

For detached or new-branch mode, `created_oid` is the selected start commit, not necessarily a newly created commit. Existing-branch mode can return `created_oid=null`; verify the resulting `head_ref`, `head_oid`, branch, and files with inspection.

### `git_tag`

`git_tag` manages local commit tags only.

`name` is a **short tag name** such as `v1.2.0`, never a full `refs/tags/...` value. The optional `target` is the field that accepts an exact ref/OID.

- `operation="create"`: `name` is the new tag; optional `target` resolves to a commit and omission uses current HEAD. `message=null` creates an unsigned lightweight tag. A non-empty `message` creates an unsigned annotated tag. `force=true` replaces an existing tag and is destructive.
- `operation="delete"`: `name` is deleted. Omit `target` and `message`; force adds no safer meaning to deletion and should remain false.

Creation cannot tag arbitrary trees/blobs and cannot sign. Creating an annotated
tag additionally requires a usable Host/repository tagger identity; lightweight
tag creation does not. `created_oid` is the target commit even for an annotated
tag; it is not the annotated tag object's ref OID. Use
`git_list_refs(kind="tags")` to verify the authoritative tag ref, and use exact
OIDs for later push leases. Delete or force requires repository delete/admin
and one-use approval; ordinary creation requires repository write. No checkout
filesystem authority is conferred.

### `git_worktree`

`git_worktree` always mutates through `main`, so `expected_state_token` must be a fresh main-repository token, preferably from `git_list_worktrees().state.token` or a new main status read.

- `operation="create"`: omit `managed_worktree_id`. Optional `ref` resolves through main; omission uses main HEAD. Optional `new_branch` is a **short new branch name**, not `refs/heads/...`; it creates that branch and checks it out in the new worktree, and cannot attach an existing branch. If `new_branch` is omitted, the new worktree is detached even when `ref` names a branch. The Runtime deterministically chooses an opaque id and a path under the configured managed root. Success returns the start commit in `created_oid` and the id in `details.managed_worktree_id`; obtain its observed path and status through list/status tools.
- `operation="remove"`: provide only an exact returned `managed_worktree_id`; `main` is invalid. First inspect that target worktree, then acquire a new main token. Removal has no force mode and should refuse a dirty target. It removes the managed checkout, not its shared commits or an associated branch.

Create requires repository write plus filesystem write on the Runtime-selected subtree. Remove requires repository write/delete/admin, filesystem delete on that subtree, and mandatory one-use approval. The target's own status token cannot substitute for the required main token; conversely, a main token is not proof that target worktree bytes remained clean after their last inspection, so rely on the non-forced remove check and re-inspect on failure.

The generated create id is stable for the same process, main token, ref, and new branch. This permits the exact same call after a pre-dispatch Human approval. It does not make a dispatched or unknown creation safe to replay.

## Recommended workflow

1. Read status for every potentially affected worktree, both diffs for a checkout change, and bounded refs/worktrees. Stop on truncation or unexplained local changes.
2. Choose the exact operation and omit fields that have no meaning in that mode. Resolve starts/targets with inspection and retain their full commit OIDs.
3. Decide how local state is preserved. Use workspace editing/change recording only if the user intends an edit or commit; use stash/integration recovery only with explicit temporary-preservation intent. Do not mutate state solely to make a topology command pass.
4. For branch/tag operations or switch, obtain the newest token for the selected worktree. For every worktree create/remove, obtain a new **main** token after the last topology/status observation.
5. If approval is requested before dispatch, approve only the exact name, target/start, force flag, worktree, and old token. Reissue the unchanged call after approval while that token is still current.
6. After success, call `git_list_refs` and `git_list_worktrees`, then status each affected worktree. For a switch or create, inspect both diffs and task-relevant file/tests. For an annotated tag, compare list-ref OID rather than assuming `created_oid` is the tag object.

Use `agent-libos-git-remotes` for remote-tracking synchronization and `agent-libos-git-integration-recovery` for merge/rebase/stash/reset/restore/clean. These topology tools do neither.

## Failure and recovery

- A Human approval request is a pre-dispatch pause. Exact retry after approval is expected. A changed token or changed parameters require a new inspection and approval decision.
- `stale_state` means re-read selected-worktree status, refs, and worktrees and reconstruct the decision. For worktree operations, ensure the replacement token is from main.
- `dirty_worktree` or a checkout conflict is evidence to inspect, not permission to add `force`, stash, reset, or clean. Preserve and report unrelated changes.
- Branch/tag `already_exists`, merge-safety rejection, or a branch checked out elsewhere must be resolved by intent, not force by default.
- The model-facing error does not expose every provider settlement detail. After any failure not clearly known to be pre-dispatch validation, permission/approval, or stale-state rejection, assume the ref, HEAD, index/worktree, generated directory, or worktree registry may already have changed—even when the code is plain `command_failed`. Do not replay with merely a fresh token.
- Reconcile branch/tag uncertainty by listing exact refs and comparing OIDs. Reconcile switch uncertainty with status, HEAD/ref, both diffs, and relevant file contents.
- Reconcile worktree-create uncertainty by comparing fresh worktree/ref lists with the saved pre-call lists and inspecting any newly visible managed id, HEAD, branch, and status. A failed tool call may not expose the deterministic candidate id; if concurrent changes leave more than one plausible result, stop as ambiguous. A partial directory or missing registry entry is not a usable worktree, and a host path is never a selector.
- Reconcile removal uncertainty by listing worktrees and checking the exact managed id and target path observation. Do not delete an orphan path through ordinary filesystem tools.
- Startup external-effect reconciliation is query-only and never replays Git. A locally correct topology can coexist with an `unknown` effect record; report that distinction.
- `unsafe_repository`, unsafe configuration, unmanaged paths, or missing typed support are stop conditions. Do not invoke shell Git to bypass validation.

## Completion evidence

Completion requires operation-specific readback:

- Branch create/rename/delete: exact `refs/heads/...` presence/absence and OID, plus unchanged HEADs in all other worktrees.
- Switch: selected worktree `head_ref`/`head_oid`, expected branch or detached state, untruncated status and both diffs, and preservation of unrelated local changes.
- Tag create/delete: exact `refs/tags/...` presence/absence and ref OID; distinguish an annotated tag object OID from the returned target commit.
- Worktree create: returned managed id appears once, is structurally managed, has intended detached/new-branch state and start OID, and has a fully inspected status.
- Worktree remove: the exact id is absent from a fresh list, no unintended branch/ref was deleted, and all other worktrees retain their state.
- Every mutation has a fresh post-operation read, regardless of whether its digest differs from the pre-state, and no truncation, unexplained diff, or unresolved unknown effect.

Do not claim completion from `created_oid` or `details` alone. Stop if local changes disappeared unexpectedly, a result is truncated, a target moved, the worktree is dirty/unmanaged, or only a host path rather than a typed id is available.
