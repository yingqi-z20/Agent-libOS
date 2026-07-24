---
name: agent-libos-git-integration-recovery
description: Restore selected paths, integrate or abort one history operation, manage stashes, reset state, or clean untracked content after exact Git inspection.
allowed-tools: git_restore git_integrate git_stash git_reset git_clean
---
# Integrate and recover Git state

Activate `agent-libos-git-inspection` first. Read untruncated status, worktree and staged diffs, HEAD/history, and the exact selected `worktree_id`; retain its latest `state.token` and relevant full OIDs. These tools can rewrite HEAD, index, worktree, stash, or untracked content. They do not provide a general Git command, automatic backup, rollback, integration continue/skip, or permission to erase unrelated state.

All calls are side-effecting and non-idempotent. Success returns a `GitOperationResult` with post-state digest `after.token`, operation-specific `created_oid`, reconciliation paths, and `details`. Tokens are not consumed and can remain equal across a verified no-op. `created_oid` can mean a resulting/selected OID rather than a newly allocated object. Capability grants are necessary but remain subject to data-flow policy and state revalidation. Always verify through inspection.

## Tool guide

### `git_restore`

`git_restore` requires one or more literal `paths`, a fresh token, at least one destination, and an optional exact commit `source`:

| `staged` | `worktree` | `source=null` meaning | Effect |
| --- | --- | --- | --- |
| false | true | index | Replace selected worktree bytes from the index |
| true | false | HEAD | Replace selected index entries from HEAD |
| true | true | HEAD | Replace both index and worktree from HEAD |

When `source` is supplied, staged-only or staged+worktree restore uses that resolved commit. Source-specific worktree-only restore is unsupported; set `staged=true` as well or choose a supported workflow. `staged=false,worktree=false` is invalid. An unborn repository without an available HEAD/source fails rather than inventing a source.

Every restore is classified destructive and requires repository write/delete/admin plus mandatory one-use Human approval. A worktree destination additionally requires filesystem write/delete subtree authority for each selected path; staged-only restore requires no checkout filesystem grant. Restore does not save discarded bytes and does not accept patterns. For renames and directory paths, inspect the complete affected subtree before approval.

### `git_integrate`

`git_integrate` supports exactly one operation:

- `operation="merge"`, `"rebase"`, `"cherry_pick"`, or `"revert"` requires one exact `ref` resolving to a commit and no `abort_kind`.
- `operation="abort"` requires `ref=null` and an `abort_kind` exactly matching `merge`, `rebase`, `cherry_pick`, or `revert`.

Merge has no strategy/ff selector on this tool: graph and validated effective configuration determine whether it fast-forwards or creates a merge commit. Rebase uses no autostash. Cherry-pick and revert accept only one commit and have no merge-mainline selector. There is no continue, skip, multi-ref, interactive rebase, custom strategy, conflict favoring, or message option.

All integration variants require repository write and selected-worktree filesystem read/write/delete. Rebase and every abort are destructive and additionally require repository delete/admin plus one-use approval. Merge, cherry-pick, and revert can still ask under ordinary policy. Dirty changes are not automatically stashed and may block or coexist according to Git; inspect them first.

On success, `created_oid` is the final HEAD, which may equal an existing commit after fast-forward/no-op. Merge commits, cherry-picks, reverts, and rebases that create commits require usable Host/repository identity; the tool has no identity override. After a conflict, the Host effect may already have written index/worktree/control metadata.

The built-in read schemas do **not** expose `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, rebase directories, or sequencer state, and the state digest does not certify their absence. An ordinary edit → stage → commit can finish a known one-step merge/cherry-pick/revert, but `git_commit` itself does not guard against doing so. Continue only when the active kind is known from the operation just attempted and finishing it is explicitly intended; otherwise stop. Rebase usually needs unavailable continue/skip controls, so stop or explicitly abort the known kind rather than invoking shell Git.

### `git_stash`

`git_stash` captures/applies bytes in the selected worktree, but its `refs/stash` reflog and numeric stack are **shared by every worktree and process in the repository common directory**. A push/drop/pop/clear elsewhere can shift every ordinal. `worktree_id` selects where checkout changes are read/written; it does not select an isolated stash stack.

- `operation="push"` stashes the whole tracked staged and unstaged state; there is no path selector. `include_untracked=true` broadens it to untracked content. The operation removes saved tracked state from index/worktree even when it does not require mandatory destructive approval. A clean push can succeed with `created_oid=null`.
- `operation="apply"` applies the selected numeric `index` and keeps its stash entry. `reinstate_index=true` also attempts to restore staged state.
- `operation="pop"` applies the selected entry and removes it only on success. On conflict Git normally keeps recovery evidence, but inspect rather than assume.
- `operation="drop"` deletes one numeric entry. `operation="clear"` deletes the whole stash stack.

Only apply/pop use `reinstate_index`; only push uses `include_untracked`; index has meaning only for apply/pop/drop. Omit irrelevant fields instead of relying on them being ignored.

There is no stash-list tool. `git_list_refs(kind="all")` can expose only the current shared `refs/stash` tip, not the reflog/ordinal mapping. `git_show` accepts an exact retained stash commit OID, not `stash@{n}` revision syntax. Save a successful push's `created_oid`; before every ordinal-based apply/pop/drop, account for cross-worktree/process changes. Stop if the exact intended entry cannot be identified confidently.

Push/apply/pop require repository write plus filesystem read/write/delete. Pop/drop/clear and push with untracked content require repository delete/admin and mandatory one-use approval; other variants can ask under policy.

### `git_reset`

`git_reset` resolves exact `target` to a commit:

- `mode="soft"` moves HEAD only and leaves index/worktree as they are.
- `mode="mixed"` moves HEAD and resets the index to target, leaving worktree bytes.
- `mode="hard"` moves HEAD and resets index plus tracked checkout state; obstructing untracked paths can also be lost during materialization. It is not a substitute for `git_clean` and does not promise removal of all untracked/ignored content.

All reset modes are deliberately classified destructive and require repository write/delete/admin with one-use approval. Hard additionally requires filesystem read/write/delete; soft/mixed require no checkout filesystem grant. `created_oid` is the selected target commit and `details.old_head_oid` records the previous tip; it is not a newly created commit.

### `git_clean`

`git_clean` removes eligible untracked content, never tracked modifications. `paths` are optional literal restrictions; an empty list broadens scope to every eligible untracked path in the selected worktree. `directories=true` includes untracked directories. `ignored=true` includes ignored content in addition to ordinary untracked content.

The Runtime performs an internal bounded dry-run and candidate manifest before authorization, then recomputes that manifest immediately before dispatch. Candidate drift yields `stale_state`. The digest/count in approval evidence are not a substitute for the caller inspecting status and literal paths. Success details include preview and candidate hashes/count, while `changed_paths` supplies reconciliation paths.

Clean requires repository write/delete/admin, filesystem read/delete, and mandatory one-use approval. Do not pair hard reset and clean as a generic “make clean” recipe: they affect different domains and each needs separate explicit intent, inspection, authority, and verification.

## Recommended workflow

1. Inventory HEAD/ref, known attempted operation/conflict evidence, shared stash-tip evidence, and all staged/unstaged/untracked paths. Require untruncated status and diffs and save full OIDs needed for recovery. Do not claim that these reads reveal hidden Git integration control files.
2. Choose the narrowest tool. Restore only selected paths; integrate one exact commit; stash only when temporary whole-worktree preservation is intended; reset only when moving HEAD/index is intended; clean only explicit reviewed untracked candidates.
3. Map exact before/after semantics in writing. For restore use the destination/source truth table. For reset name what remains in index/worktree. For stash retain the source state and expected stash OID/index. For integration decide what conflict outcome is acceptable.
4. Obtain a fresh token after the last observation. Request approval only for the exact paths, OID, mode/operation, broadening flags, and old token.
5. If approval pauses before dispatch, repeat the identical call after approval while state is unchanged. Do not turn a refusal into `force`, a broader path list, hard reset, clean, or stash clear.
6. After success, re-read status, both diffs, HEAD/log, and refs. Inspect every changed path and run task-specific tests. For stash, verify both the checkout and retained/deleted evidence that the typed surface can actually expose.

## Failure and recovery

- `stale_state` means re-read and re-decide. For clean, it can specifically mean candidate bytes/types changed after approval. Never replace only the token.
- `repository_busy` requires a later fresh observation, not repeated mutation calls.
- A pre-dispatch Human approval pause is the only case where exact replay of unchanged parameters/token is expected. The model-facing error omits some provider settlement details, so any error not clearly known to be pre-dispatch validation, permission/approval, or stale state—including ordinary `command_failed`—must be treated as possibly dispatched and reconciled before another mutation.
- After integration conflict/uncertainty, inspect HEAD, status, staged/worktree diffs, and the exact ref/OID effects that the typed surface can expose. Do not claim to inspect hidden operation state, and do not restart the same merge/rebase/cherry-pick/revert. Use `abort_kind` only when it matches the operation known to have started; never guess from status alone. If completion requires unavailable continue/skip/mainline control or the active kind is uncertain, stop.
- After restore/reset unknown, compare saved and current HEAD, index/staged diff, worktree diff, and every selected path. Do not restore/reset again merely to obtain a success envelope.
- After stash push unknown, inspect checkout/index and `refs/stash`; compare any saved old/new stash OIDs. Blind push can create duplicates or hide already-saved state. After apply/pop/drop/clear unknown, do not trust the old ordinal index; reconcile visible tip plus checkout/index and stop if deeper stack identity is unknowable.
- After clean unknown, inspect every precomputed candidate and status. Missing files may have been deleted; never recreate or delete further content without task evidence.
- Provider startup recovery is query-only and never rolls back or replays Git. A locally reconstructed outcome does not change a durable unknown-effect classification.
- `unsafe_repository`, unsafe configuration, truncated evidence, unsupported control, or an unidentifiable stash entry is a stop condition. Shell Git is not a supported escape hatch.

## Completion evidence

Finish only when the exact requested state is proven:

- Restore: selected index/worktree destinations match the intended index/HEAD/source commit; every unselected change remains.
- Integrate: final HEAD/parents or fast-forward target match intent, no unresolved entries remain, and relevant tests pass. State separately that hidden operation-control state is not readable; only claim it cleared when a successful typed completion/abort of the known operation plus subsequent Git behavior establishes that conclusion. Abort, if chosen, must restore the expected pre-operation state.
- Stash: the intended tracked/untracked/index state was saved/applied, retained or deleted as requested, exact observable OIDs agree, and no unrelated entry was addressed by a shifted ordinal.
- Reset: HEAD equals exact target; index/worktree preservation or rewrite matches soft/mixed/hard semantics; old HEAD evidence is retained in the report.
- Clean: only reviewed eligible candidates are absent; tracked changes and excluded untracked/ignored/directory content remain.
- All reads are untruncated, changed paths are byte-safe, successor state is freshly observed, and no conflict or unknown external effect is mislabeled as verified success.

If shared-stash identity cannot be proven, the active integration kind is unknown, an integration needs unavailable continuation, any unrelated content disappeared, or recovery evidence remains ambiguous, stop and report the exact limit instead of applying another destructive operation.
