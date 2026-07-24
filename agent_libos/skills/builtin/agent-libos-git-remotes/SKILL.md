---
name: agent-libos-git-remotes
description: Fetch, pull, or push through an existing Host-configured Git remote. Use for remote synchronization when the exact local state, remote name, branch or full destination ref, integration strategy, and guarded destructive intent are known; never supply ad hoc URLs or credentials.
allowed-tools: git_fetch git_pull git_push
---
# Synchronize through configured Git remotes

Remote operations cross a provider boundary. Activate `agent-libos-git-inspection` first and use only a remote returned by `git_list_remotes`. Its redacted URLs and fingerprints are observations, never transport inputs. These tools accept no arbitrary URL, refspec, credential, environment override, or shell command; Host configuration selects non-interactive transport and credentials.

Keep three forms of freshness separate:

- `expected_state_token` is local repository compare-and-swap (CAS). It prevents dispatch after observed local Git state changes.
- The remote fingerprint binds the authorized operation to observed local remote configuration and relevant locally recorded refs. It is checked again before dispatch, but it is not a live server-state lock.
- `force_with_lease_oid` asks the remote server to accept a destructive update only if the exact destination ref still has the expected OID. It is the only supported force/delete guard.

None substitutes for Capability, provider policy, data-flow authorization, Human approval, or post-operation verification.

Local `state.token` values are deterministic state digests, not consumed nonces. A successful push can leave all local token inputs unchanged and return `after.token == before.token`; this says nothing about remote success. Always perform the operation-specific readback below.

## Tool guide

### `git_fetch`

Fetch from an exact registered `remote` into the selected `worktree_id` using its fresh local `expected_state_token`. The command uses the remote's Host-configured fetch refspec, disables tags and submodule recursion, and accepts no branch or refspec override. Therefore the refs updated are determined by configuration, not by the model.

Keep `prune=false` by default. `prune=true` may delete stale local remote-tracking refs; it requires exact repository delete/admin authority and mandatory one-use Human approval in addition to remote read and repository write authority. It does not prune tags.

Success returns a normal `GitOperationResult`: use `after.token` for the successor local state. `details.remote_fingerprint` records the preflight fingerprint, but success does not enumerate authoritative server refs. Fetch does not integrate a branch, alter worktree files, or prove that a particular ref was covered by the configured fetch refspec.

### `git_pull`

Fetch exactly one remote branch into `refs/remotes/<remote>/<branch>`, then integrate that fetched commit in the selected worktree. Supply `remote`, a fresh local token, and a deliberate strategy:

- `ff_only` runs a fast-forward-only merge and is the safest default. It fails rather than create a merge commit or rewrite local commits.
- `merge` integrates with a merge when needed. No merge-strategy selector is exposed, signing is disabled, and dirty state is not automatically stashed.
- `rebase` rebases without autostash. It rewrites local commit topology and is destructive, so it requires exact repository delete/admin authority and mandatory one-use Human approval.

`branch` is a **short remote branch name** such as `main` or `feature/x`, never `refs/heads/...`; the implementation itself constructs `refs/heads/<branch>` and `refs/remotes/<remote>/<branch>`. If `branch` is null, the tool uses the current **local short branch name**; it does not resolve configured upstream/merge settings. Detached `HEAD` requires an explicit short branch. The argument selects the remote branch to fetch, while integration always acts on the currently checked-out local branch/worktree state. Confirm that this pairing matches user intent before dispatch.

Pull requires remote read, repository write, selected-worktree filesystem read/write/delete authority, and data-flow clearance. A clean worktree is the reliable starting point. Merge/rebase paths that create commits also need usable Host/repository identity; no override is exposed. Conflict or integration failure can occur after the fetch has already changed the tracking ref, so failure is not atomic. On success, `created_oid` is the resulting local `HEAD`, `after.token` is its post-state digest, and `details.remote_oid` identifies the fetched commit.

### `git_push`

Push one explicit local ref to one explicit full remote ref. `remote_ref` must be `refs/heads/...` or `refs/tags/...`. For an update, `local_ref` is required and may be a short local branch or a full local `refs/heads/...` or `refs/tags/...`; it is resolved to an exact local OID before push. No implicit current-branch push, wildcard refspec, follow-tags, hook execution, signing, or submodule behavior is exposed.

For an ordinary update, set `delete=false` and omit `force_with_lease_oid`. Git/remote policy still rejects non-fast-forward updates. For a guarded force update, supply the exact current remote destination OID as `force_with_lease_oid`. For deletion, set `delete=true`, omit `local_ref`, and supply that same exact old remote OID. OIDs must be full length for the repository object format; abbreviated OIDs and guessed zero values are invalid.

Force and deletion require remote delete/admin authority and mandatory one-use Human approval. All pushes require remote write and repository read authority, and data-flow policy applies to the selected local commit/ref lineage. The command disables local pre-push hooks, push signing, and automatic tag following.

Success returns local evidence: `created_oid` is the pushed local OID for an update and null for deletion; `after.token` is local CAS state; `details` records refs, lease choice, and preflight fingerprint. This is evidence that the command returned successfully, not an independently read authoritative remote snapshot.

## Recommended workflow

1. Inspect repository identity, selected worktree, status, exact refs, and registered remotes. Resolve every short local branch to an exact returned ref/OID and retain the latest `state.token`.
2. Decide whether the intent is observation-only synchronization (`git_fetch`), fetch plus local integration (`git_pull`), or publication (`git_push`). Do not use pull merely to inspect remote state.
3. For fetch, inspect the configured remote facts and understand that its configured refspec controls coverage. Decide explicitly whether local remote-tracking deletion via prune is intended.
4. For pull, name the remote branch when there is any ambiguity, confirm the current local branch/worktree, preserve or commit unrelated changes, and choose `ff_only`, `merge`, or `rebase` based on topology policy.
5. For push, provide both source and full destination refs. Resolve the source OID and inspect its diff/lineage. For force/delete, first obtain authoritative knowledge of the old destination OID; stop if the typed interface cannot supply it.
6. Call one mutation with the fresh token. After success or failure, re-inspect local status and refs before using another token.
7. Verify a pushed branch by performing a fresh `git_fetch`, then call `git_list_refs(kind="remotes")` **only when** the configured fetch refspec maps that destination branch to a known remote-tracking ref. `git_list_refs` accepts no token argument and returns its own `state.token`; compare the exact ref/OID and use that new observation for any later mutation. If the refspec does not cover the destination, say verification is unavailable instead of treating an absent/stale ref as proof.

Branch deletion verification may require a separately authorized pruning fetch so a stale tracking ref can be removed. Tags are not fetched by these tools, so the built-in surface generally cannot independently verify a pushed or deleted tag. Report that limitation. Never infer server state from the local push `created_oid` alone.

## Failure and recovery

All three tools are effectful and declared non-idempotent. The model-facing error omits some provider settlement details. Treat only a clearly pre-dispatch validation, permission/approval, remote-fingerprint, or stale-state rejection as effect-not-started; conservatively reconcile every other failure before retrying, including plain `command_failed`.

- A local `STALE_STATE` or remote-fingerprint mismatch before dispatch means preconditions changed. Re-list remotes/status/refs, reevaluate intent, and submit a fresh token. Do not patch around CAS.
- When mandatory approval is requested before dispatch, retry the exact operation only after approval and only while its token, remote fingerprint, refs, lease, and arguments remain current. Changed state requires a new inspection and decision.
- A fetch failure after possible dispatch may have updated some configured remote-tracking refs. Re-list local refs before considering another fetch; with prune, also check whether local refs disappeared.
- Any pull error after possible dispatch may mean the tracking-ref fetch completed even though the model surface does not expose `fetch_completed`. Inspect that ref, `HEAD`, status, index, and conflict evidence. The typed reads do not expose hidden merge/rebase control files; use `agent-libos-git-integration-recovery` only for the exact operation known to have started, and never blindly pull again.
- Any push failure after possible dispatch may have updated the server even when no success result arrived. Reconcile through a fresh authorized fetch/ref observation when the destination branch is covered. If authoritative typed evidence is unavailable, leave the outcome unknown and ask for external verification rather than replaying.
- Runtime external-effect recovery in query-only mode reads recorded effect state only; it does not contact the remote and does not replay the operation. An `unknown` record remains unknown until fresh authoritative evidence resolves it.
- A lease rejection is safety working as designed: obtain the new authoritative remote OID, compare histories, and ask the user to choose merge/rebase/new lease/abandonment. Never weaken to an unguarded force.
- Authentication, network, server policy, data-flow, or Capability denial does not authorize raw shell Git, embedded credentials, an ad hoc URL, or transport reconfiguration. Report the boundary and the evidence available.

## Completion evidence

For fetch, report remote, prune choice, returned `before.token`/`after.token`, preflight fingerprint, and the exact local refs/OIDs observed afterward, including whether configured refspec coverage limits the conclusion.

For pull, report remote branch, current local branch/worktree, strategy, fetched `remote_oid`, old and resulting `HEAD`, successor token, clean/conflict status, and tests run. State whether history was rewritten or a merge commit was created.

For push, report exact local and remote refs, resolved local OID (or deletion), lease OID if used, approval evidence where required, returned local token, and the independent verification method/result. If fresh fetch coverage, tag visibility, pruning authority, or external access prevents authoritative verification, explicitly label the remote outcome as unverified or unknown rather than complete.
