---
name: agent-libos-git-change-recording
description: Stage, unstage, and commit reviewed changes in one managed Git worktree. Use only when the entire index is intended for the commit; stop if unrelated staged entries must remain staged.
allowed-tools: git_stage git_unstage git_commit
---
# Record Git changes

This Skill changes the Git index and commits; it does not edit ordinary file content, switch branches, start an integration, or contact a remote. Activate `agent-libos-git-inspection` first and use its byte-safe path, exact-ref, completeness, and state-digest contract. A plain `git_commit` can nevertheless finalize an already active merge/cherry-pick/revert state; the built-in read surface does not expose those control files. If such a state is known or cannot be ruled out after an integration failure, stop and use `agent-libos-git-integration-recovery` rather than treating the call as an ordinary commit. If content must change, use `agent-libos-workspace-editing` before recording it.

Every tool here is side-effecting and declared non-idempotent. A successful ordinary result is a `GitOperationResult`: `before.token` identifies the accepted observation and `after.token` identifies the observed post-state. These tokens are state digests, not consumed nonces; a no-op stage/unstage can return the same value. `created_oid` is operation-specific; for `git_commit` it is the resulting commit OID. `changed_paths` can be the selected path manifest even when no index bytes changed. A returned result is therefore not a substitute for reading back the tree.

## Tool guide

### `git_stage`

`git_stage` takes one or more literal `paths`, a fresh `expected_state_token`, and `worktree_id`. It runs the typed equivalent of `git add -- <paths>`:

- Existing, new, and missing tracked paths can update the index. It can therefore stage additions, modifications, and deletions.
- It does not support patch/hunk selection, interactive mode, pathspec patterns, an empty-path “stage all” mode, or arbitrary flags.
- A directory is still a literal path, but Git can stage its whole subtree. The Runtime consequently requires filesystem read authority for that subtree. Do not stage a directory until every included entry is reviewed.
- A missing/deleted input is also authorized as a filesystem **subtree** scope, not an exact-file scope, because preflight cannot prove its file kind. Request the matching directory/subtree read authority rather than repeating an exact-file grant.
- A new ignored file is not force-added: the tool exposes no `-f`. Ignored children are not included merely because their directory was selected.
- A submodule directory is different from an ordinary directory: staging it records the parent repository's gitlink OID and does not stage or review the nested checkout's files.
- A renamed status entry has a new `path` and an `original_path`. Staging only the new name can leave the old-path deletion unstaged; explicitly review and select both sides needed for the intended index.
- A non-UTF-8 path must be passed as its exact `path_b64` input, never reconstructed from `display`.

Required Capability includes repository Git write plus filesystem read for each selected exact-file or subtree scope. Every mutation also passes data-flow sink policy and target-state revalidation, so these grants are necessary but not sufficient; a data-flow denial is not fixed by requesting broader Git/filesystem rights or using shell. Staging does not grant file read authority, and filesystem authority alone does not grant index mutation. Success normally returns selected/reconciliation paths in `changed_paths`; it does not prove those paths changed or that the staged patch matches intent.

### `git_unstage`

`git_unstage` takes one or more literal `paths`, a fresh same-worktree token, and `worktree_id`. It changes only the index and preserves worktree bytes. With an existing HEAD it resets those index paths to HEAD. In an unborn repository it removes matching entries from the index while leaving their files in the worktree.

It requires repository Git write authority, data-flow clearance, and no checkout filesystem grant. It has no “unstage all” empty-path mode and no revision/source selector. Unstaging one side of a rename or only part of a directory can intentionally split index and worktree state, so inspect both diffs afterward.

### `git_commit`

`git_commit` takes `message`, a fresh token, `amend`, and `worktree_id`. The message must remain non-empty after trimming, contain no NUL, and encode to at most 131,072 UTF-8 bytes; the schema's character bound is not the final byte check. It commits the **entire current index**, including every previously staged entry. There is no path selector, partial commit, allow-empty option, author/committer override, signing option, editor, or hook execution. If any staged entry is outside the requested commit, stop and ask whether to expand the commit or change the index explicitly; this tool cannot both commit and preserve that entry as staged.

For a normal standalone commit, author/committer identity comes from validated Host/repository configuration. For `amend=true`, the command does not use `--reset-author`: it preserves the replaced commit's author while the current configured identity is the committer. Missing required identity fails. The Runtime disables hooks and GPG signing. Before either mode, stop if an active integration is known or possible: the implementation does not reject `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, rebase, or sequencer state, and plain commit can complete or alter that operation.

`amend=false` creates a normal commit from the whole index. `amend=true` replaces the current tip using that same complete index and new message. Amend is destructive even when the tree is unchanged: it requires repository write/delete/admin authority and mandatory one-use Human approval bound to the exact old state. A request for a normal commit is not consent to amend.

On success, `created_oid` is the resulting HEAD commit and `after.token` observes the successor state. The worktree can remain intentionally dirty after a commit; a clean status is not required unless the user requested one.

## Recommended workflow

1. Read `git_status` for the exact `worktree_id`; reject or narrow a truncated result. Record the current HEAD OID, especially before amend.
2. Read `git_diff(scope="worktree")` and `git_diff(scope="staged")`. Worktree diff omits untracked contents, so inspect each untracked file with workspace-reading tools. Status omits ignored content, and parent-repository reads do not inspect nested submodules. If the staged diff contains anything outside the requested commit, stop before committing: it cannot remain staged across a whole-index commit.
3. Map the reviewed intent to the smallest literal path set. For each rename include the required old/new path identities; use byte tokens for lossy names. Avoid a directory path when explicit file paths are practical.
4. Call `git_stage` with the newest same-worktree `state.token`. If Human approval is requested before dispatch, obtain it and reissue the exact same arguments only while the observed token remains current.
5. Re-read status and both diffs. Check the complete staged patch, not merely `changed_paths`. If the index contains a mistake, call `git_unstage` only for the exact mistaken paths with the newest token, then re-read both diffs again.
6. Before committing, verify the entire staged tree is intended, the byte-level message contract is satisfied, no integration state is known or suspected, and `amend` is explicitly chosen. Call `git_commit` with the latest state digest.
7. Read back the returned `created_oid` with `git_show`, then read status plus staged/worktree diffs. Confirm the commit tree and the intentionally dirty remainder separately.

Do not stage unrelated changes merely to make status clean. Do not unstage user changes as a convenience. When unrelated entries are already staged, report the whole-index constraint and obtain intent instead of promising to preserve them. Do not amend because a previous commit message seems inconvenient unless replacing that exact tip is within the request.

## Failure and recovery

- A Human approval request occurs before Host dispatch. After approval, repeat the exact operation and token; do not alter paths, message, `amend`, or worktree under the old approval binding. If state changed, start over with inspection and expect a new decision.
- `stale_state` means this invocation did not accept the old CAS observation. Re-read status and both diffs, account for user/concurrent changes, and reconstruct the path/index decision.
- `repository_busy` requires a later fresh read. Do not spin on the mutation with the same token.
- `identity_missing`, validation failure, an empty index, or an unavailable operation is not permission to use shell Git or fabricate identity. Correct the supported precondition or stop.
- A truncated staged diff is never sufficient commit evidence. Narrow inspection or stop; do not commit a tree whose complete contents were not reviewed.
- The model-facing error does not expose every provider settlement detail. After any error not clearly known to be pre-dispatch validation, permission/approval, or stale-state rejection, assume stage, unstage, or commit may already have happened. A plain `command_failed`, timeout, or `unknown_effect` code can all require reconciliation; tool-level `retryable` never authorizes blind replay. Inspect HEAD OID, status, and both diffs first.
- For uncertain `git_stage`, compare the index patch with the intended path set. If already staged, do not stage again merely to obtain a success response.
- For uncertain `git_unstage`, determine whether the paths remain in staged diff and whether worktree bytes are intact before deciding that anything remains.
- For uncertain `git_commit`, compare HEAD with the saved pre-operation OID, inspect recent exact OIDs, show the candidate commit, and inspect index/worktree. For amend, the old tip may no longer be reachable from ordinary refs; the saved old HEAD OID is essential reconciliation evidence.
- If a post-dispatch failure left state that is semantically correct but the external-effect record is unknown, report both facts. Local inspection does not rewrite the durable unknown-effect classification.

Never “repair” an uncertain operation by reset, clean, checkout, amend, or another commit without a separately reviewed request and the corresponding recovery Skill.

## Completion evidence

For staging or unstaging, completion requires:

- an untruncated status for the exact worktree;
- complete staged and worktree diffs after the operation;
- byte-identical intended path selection, including both sides of renames;
- confirmation that worktree bytes were not unexpectedly changed;
- preservation of every unrelated **unstaged** user change, with no out-of-scope staged entry silently included; if such an entry existed, record the user's explicit scope decision;
- a fresh read of the post-state; do not require the digest value to differ when the operation was a verified no-op.

For a commit, completion additionally requires:

- `git_show(created_oid)` identifies the resulting commit and an untruncated patch;
- the commit's complete tree change and parents match the requested normal/amend action;
- the commit message, author, and committer match the intended normal/amend semantics;
- status and both diffs explain every remaining change;
- amend, when used, had exact destructive authority and one-use approval;
- no uncertain effect, conflict, truncation, or unexplained index remainder is presented as success.

A clean worktree is optional. Exact recording and preservation of unrelated state are the completion criteria.
