---
name: agent-libos-git-patch-objects
description: Create immutable, byte-safe Git patch Objects and apply them to an authorized worktree in the same repository. Use for bounded patch transfer, reviewable change handoff, or cross-process application through Object Memory; do not use rendered diffs as patch payloads.
allowed-tools: git_create_patch git_apply_patch
---
# Transfer Git changes as immutable Objects

Treat a `CODE_PATCH` Object as a complete binary patch plus Runtime provenance, not as copied terminal text. Activate `agent-libos-git-inspection` for source and destination observations and `agent-libos-object-memory` when Object discovery or access must be reconciled. The Object, repository state token, Capability, data-flow decision, and Human approval are independent controls; possession of one never implies the others.

## Tool guide

### `git_create_patch`

Create one immutable `CODE_PATCH` Object from one internally consistent source-repository observation. The operation does not change Git state, but Object creation is a non-idempotent side effect and requires both Git diff and Object write authority. It deliberately has no `expected_state_token`: under the repository read lock the primitive reads the patch, computes the index tree, and rejects state drift detected during that capture. The lock is released before the Object is stored, so the returned `state.token` proves the captured patch/index observation—not that repository state remained unchanged until Object creation completed. Re-read if current source state matters after creation.

Choose exactly one scope:

- `scope="worktree"` captures unstaged tracked changes relative to the index. Leave `base` and `head` null.
- `scope="staged"` captures index changes relative to `HEAD`. Leave `base` and `head` null.
- `scope="range"` captures the exact `base..head` tree difference. Supply both `base` and `head` as validated exact refs or full OIDs.

Use `paths` only as literal byte-safe restrictions. A returned path is `{display, path_b64, lossy}`; use `path_b64` when feeding it back so unusual filesystem bytes cannot be changed by display decoding. Rename detection is disabled, so a rename is represented by its old and new paths. A worktree patch does not include untracked files because Git diff has no untracked-file payload.

Submodules remain parent-repository gitlinks: a patch can carry a gitlink OID change but never the nested checkout's file patch/history. Stop if the requested handoff depends on nested submodule content.

Patch creation is all-or-nothing. The patch must fit the configured patch hard limit, the configured artifact limit, and the Object Memory payload hard limit. It rejects a conflicted index, unsupported index modes, and intent-to-add entries because it cannot produce complete, trustworthy index provenance in those states. Resolve or deliberately restage first; never substitute a truncated `git_diff` response.

The result is a `GitPatchArtifact`, not a normal mutation wrapper. Preserve:

- `oid`: the Runtime Object identifier used by `git_apply_patch`; it is not a Git commit/blob OID.
- `repository_id` and `worktree_id`: source identity.
- `base_oid`, `head_oid`, and `index_oid`: source provenance recorded at creation.
- `patch_sha256`, `bytes`, and complete `changed_paths`: integrity and manifest evidence.
- `state.token`: the stable source Git observation token.

The provenance OIDs describe where the patch came from; they are not destination preconditions. Applying to a different state can be valid if Git's own check succeeds and the caller intentionally accepts that destination. Do not falsely require destination `HEAD` or index to equal the artifact's source fields.

### `git_apply_patch`

Apply an exact `patch_oid` to a selected managed `worktree_id` using a fresh destination `expected_state_token`. The primitive first resolves Object read authority, verifies immutable `CODE_PATCH` type, trusted Git-creation provenance, schema, byte length, digest, manifest, size limit, and same `repository_id`. It then runs a checked application with strict whitespace and recount behavior before dispatching the write.

Select index behavior deliberately:

- `index=false` applies changes to the worktree only. It does not stage them.
- `index=true` uses Git's index-aware apply and updates both index and worktree. It is not an “index only” mode.

Neither mode creates a commit. The destination need not be clean, but overlapping edits, index mismatches, whitespace errors, or context drift can make the preview fail. Inspect unrelated destination changes before application and ensure the patch manifest does not accidentally overlap them.

Application requires Object read, repository Git write, and filesystem read/write authority for affected paths. Existing regular files use exact-file scopes; directories and any target observed as missing at preflight use subtree scopes because preflight cannot prove a narrower final shape. A patch containing deletion additionally requires filesystem delete on the corresponding exact/subtree scope, exact repository delete/admin authority, and mandatory one-use Human approval. Data labels and lineage from the Object propagate to written paths; approval does not grant Object access, and Object access does not bypass filesystem or repository authority.

On success, read the normal `GitOperationResult`: `after.token` is the observed post-state digest and may equal `before.token` only for a verified no-op; tokens are not consumed. `details` contains the Object/digest/size/index/deletion facts, and `changed_paths` is the affected/reconciliation manifest rather than semantic proof. `created_oid` is normally null because applying a patch creates no commit.

### Object lifecycle boundary

The patch Object belongs to the Runtime/Object-owner lifecycle. Sending its OID to another process does not delegate `ObjectRight.READ`; explicitly share/delegate Object authority through supported Object Memory and Capability workflows. The Object may cross managed worktrees only when their `repository_id` is the same and cannot bridge unrelated repositories. An ordinary Runtime reopen loses volatile Object payloads. A checkpoint or checkpoint-derived image is different: when this Object is owned/captured inside the checkpoint subtree, its payload is eligible under the normal checkpoint payload limits and can be restored/remapped with that snapshot. An OID copied by itself, an uncaptured borrowed Object, or filesystem-only copying does not preserve it. Use the checkpoint/image contract deliberately or a separately supported durable artifact workflow for longer-lived delivery.

## Recommended workflow

1. Inspect the source with status and both relevant diffs. Resolve conflicts and intent-to-add entries, identify literal paths, and decide whether the intended unit is worktree, staged, or an exact commit range.
2. Call `git_create_patch` once with the narrowest scope and path set. Record the returned Object OID, patch digest/size, source token, provenance OIDs, and complete path manifest.
3. If another process will apply it, arrange explicit Object read authority and required repository/filesystem rights. Do not pass credentials, ad hoc transport commands, or reconstructed payload text.
4. Inspect the destination worktree immediately before application. Record its fresh `state.token`, status, worktree diff, staged diff, and any unrelated paths to preserve.
5. Decide whether the result belongs only in the worktree or in both worktree and index. Preview conceptually against the actual destination; then call `git_apply_patch` with the exact Object OID and fresh destination token.
6. Re-read destination status and both diffs. Compare the byte-safe changed-path set and patch intent, then run task-specific build/tests. Commit separately with `agent-libos-git-change-recording` only if requested.

For review-only handoff, stop after Object creation and report the OID and completeness evidence. For change transfer, do not declare success merely because apply returned: validate semantics and preservation of unrelated changes.

## Failure and recovery

- For patch application, only an **initial destination state-token CAS** `stale_state` before the first state-changing provider effect certifies that Git was not mutated. Re-inspect, reconsider scope and overlap, and use the current digest where applicable. A later or unqualified `stale_state` is possibly dispatched. Never replay merely because a hex token changed.
- The model-facing error does not expose every provider settlement detail. After any create error not clearly known to precede Object creation, search readable Object Memory for a trusted `CODE_PATCH` matching repository identity, schema, digest, size, manifest, and source provenance before creating another. A plain failure, timeout, or `unknown_effect` can all require this reconciliation; if authority prevents it, report the unknown result rather than claiming absence.
- A failed create caused by output limits means there is no safe partial artifact. Narrow literal paths or split the work into independently meaningful complete patches; never trim base64 or rendered text.
- A preview rejection from `git_apply_patch` should leave Git unchanged. Re-inspect destination conflicts, index state, whitespace, and every manifest path; choose a new destination or construct a new source patch rather than editing immutable payload bytes.
- After any apply error not clearly known to be a pre-dispatch validation, permission/approval, or an **initial** destination state-token CAS rejection, assume some or all affected paths may have been written. This includes a later `stale_state`. Inspect destination status, staged/worktree diffs, content digests where needed, and every manifest path before deciding whether retry is safe. Do not depend on seeing a literal timeout or `unknown_effect` code; blind reapplication can duplicate semantic changes or overwrite repair work.
- If deletion approval is requested before dispatch, preserve the exact `patch_oid`, `worktree_id`, index choice, and expected token for the approved retry. If repository state changed while waiting, obtain a new observation and expect a new approval decision.
- Same-repository, Object integrity, or Object authority failures are hard boundaries. Do not export the payload, forge provenance, run raw shell Git, or weaken checks as a workaround.

## Completion evidence

For creation, report the exact Object OID, repository/worktree identity, scope and literal path selection, patch SHA-256 and byte count, complete changed-path manifest, provenance OIDs, and source `state.token`. State explicitly that the OID is a Runtime Object identifier and identify any lifecycle limitation relevant to the handoff.

For application, report the destination repository/worktree, `index` choice, `before.token` and `after.token`, verified Object OID/digest/size, byte-safe affected paths, post-apply status and both diffs, and tests or validation run. Confirm that unrelated changes were preserved and whether the result is staged. Do not claim a commit, cross-repository portability, durable export, or semantic correctness unless separately demonstrated.
