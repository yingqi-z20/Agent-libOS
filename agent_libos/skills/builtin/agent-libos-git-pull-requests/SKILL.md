---
name: agent-libos-git-pull-requests
description: Create, list, inspect, review, merge, or close repository-local simulated pull requests backed by Git snapshot refs and metadata. Use for Agent libOS's evidence-preserving local PR workflow; do not use it for GitHub, GitLab, or any external forge.
allowed-tools: git_create_pull_request git_list_pull_requests git_inspect_pull_request git_review_pull_request git_merge_pull_request git_close_pull_request
---
# Work with repository-local simulated pull requests

These tools implement a local review record, not a hosted pull request. They write verified JSON metadata under the repository's Git common directory and snapshot refs under `refs/agent-libos/pull-requests/...`; they never contact or update GitHub, GitLab, or another forge. The metadata retains PR and review bodies as plaintext so their hashes can be validated. Model inspection returns the PR body but only hashes for review bodies; that projection is not encryption, and common-directory administrators can read the stored text. If the user requests an external PR, stop and use a separately configured forge integration.

PR metadata is shared across managed worktrees of the same repository, but most operations intentionally observe the main repository context. Checkpoint restore and image commit do not promise to package or roll back this Git/provider state. Do not describe these records as remote, globally durable, or transactionally restored with an Agent image.

Every `state.token`/`operation.after.token` is a local deterministic state digest, not a consumed nonce. Use it only for its repository/worktree observation and verify writes by fresh reads; do not require the hex value itself to change as proof of effect.

## Tool guide

### `git_create_pull_request`

Create a frozen review record from two existing local branches. `base_ref` and `head_ref` accept a short local branch name or full `refs/heads/...`; OIDs, tags, remote-tracking refs, `HEAD`, and revision expressions are rejected. The tool resolves both branches in the main repository, hashes the complete binary no-renames diff from base OID to head OID, creates two snapshot refs, and then writes metadata.

Use a fresh main-repository `expected_state_token`. There is no `worktree_id` argument. Title must be nonblank and bounded; body is bounded and may be empty. Creation freezes `base_oid`, `head_oid`, and `patch_sha256`; later branch movement does not update the PR or its snapshot refs.

Creation requires `WRITE` on the deterministically generated PR resource, repository Git `READ` for its protected verification phase, repository Git `WRITE` for the snapshot refs, and repository data-flow sink clearance. Review, merge, and close likewise perform protected repository reads before their PR mutation. These are independent checks; a visibility/read grant, a token, or Human approval does not supply the others, and a data-flow denial is not fixed with broader Git authority or shell Git.

The PR ID is deterministically derived from actor, expected token, branches, title hash, and body hash. This makes an exact pre-dispatch approval retry target the same ID. It does not make creation generally idempotent: a post-dispatch failure can leave refs, metadata, or both, and a replay can collide with partial state.

The return shape is `{pull_request, operation}`. Read the successor token at `operation.after.token`, not at a top-level `after`. `pull_request` contains the human-readable title/body, exact refs/OIDs, patch hash, status, timestamps, and its `state`; `operation` carries mutation evidence and `created_oid=head_oid`.

### `git_list_pull_requests`

List local PR summaries with optional exact `status` (`open`, `merged`, or `closed`) and a bounded `limit`. This is a stable read of metadata plus verified snapshot refs. The result has top-level `pull_requests`, `truncated`, `bytes`, `sha256`, and `state`; the reusable read token is `state.token`.

There is no cursor or page token. `truncated=true` means the collection is incomplete; raising the limit within the configured hard bound or filtering by status may narrow the view, but the tool cannot page beyond its bounded scan. Never infer that a PR is absent or that all open work was reviewed from a truncated result.

### `git_inspect_pull_request`

Inspect one exact `pr_id`. It validates metadata schema and hashes and verifies that both snapshot refs still equal the recorded base/head OIDs. The returned `GitPullRequest` includes the PR body, snapshot facts, status, reviews, metadata size/hash, and `state.token` from the main repository.

Review entries expose `review_id`, actor, decision, `body_sha256`, and timestamp, but **not the review body text**. Do not claim that inspection reproduced the review comment. The original PR body is returned. Missing, malformed, oversized, or mismatched metadata/snapshots fail closed rather than returning a partially trusted record.

### `git_review_pull_request`

Append one review to an open PR using `decision="comment"`, `"approve"`, or `"request_changes"`, a bounded body, and a fresh main-repository token. A comment requires PR write authority; approve and request-changes require PR approve authority. Metadata update also uses its own digest CAS inside the protected mutation.

The review ID is newly generated during dispatch, so a timeout retry can create a duplicate review with a different ID. The returned review exposes only its body hash, not the body text. The return shape is `{pull_request, operation}` and the successor is `operation.after.token`.

Reviews are evidence, not an enforced merge gate. An approval does not grant Capability or mandatory Human approval, and a request-changes review does not mechanically block `git_merge_pull_request`. A review also does not refresh frozen snapshots when branches move.

### `git_merge_pull_request`

Merge an open PR into a selected managed `worktree_id`. This is the only PR tool with a worktree selector. Before dispatch, all of the following must hold:

- Both current local base/head branches still resolve to their recorded OIDs.
- The selected worktree has the exact recorded base branch checked out at `base_oid`.
- Its index and worktree are completely clean, including no untracked status entries.
- Snapshot refs still match metadata and the PR status is open.

Use a fresh token from `git_status` on the **selected merge worktree**. The token from `git_inspect_pull_request` is a main-worktree token and is suitable only when merging in `main`; it is not interchangeable with another worktree's token.

Choose one exact strategy:

- `fast_forward` rejects divergent histories and advances `HEAD` to `head_oid` when the base is behind that head. If both OIDs are equal, or the recorded head is already an ancestor of the base, Git can report already up to date and leave `HEAD` at the base; the PR can still be recorded as merged without a new commit.
- `merge` requests `--no-ff` with signing disabled and exposes no lower-level strategy selector. It normally creates a merge commit when the head contributes reachable history not already in the base, but equal OIDs or a head already contained in the base can complete as an already-up-to-date no-op. A merge commit, when one is created, requires usable Host/repository identity.
- `squash` asks Git to stage the recorded head's net changes and then attempts one new commit named for the PR. It requires usable Host/repository identity, disables hooks/signing for the commit, and does not move the head branch. An empty net delta—including `base_oid == head_oid` or otherwise identical resulting trees—leaves nothing to commit, so the operation fails rather than promising a squash commit or marking the PR merged.

Merge requires PR approve authority, mandatory one-use Human approval, repository write/delete/admin authority, and filesystem read/write/delete authority. Recorded approval reviews are separate and do not satisfy these controls. Success updates PR metadata to `merged`, records `merged_oid`, and returns `{pull_request, operation}` with the successor at `operation.after.token`.

Clean status does not prove that no merge/rebase/cherry-pick/revert control state exists: the built-in read schemas do not expose those Git control files. If a prior integration failed or such state is otherwise possible, stop and reconcile it before PR merge rather than treating an empty status as sufficient.

### `git_close_pull_request`

Close an open PR without merging it. It requires PR delete authority and a fresh main-repository token, may additionally require policy approval, and has no `worktree_id`. Closing updates metadata but deliberately preserves reviews and snapshot refs as evidence. There is no reopen tool and close does not delete either branch.

Success returns `{pull_request, operation}` with status `closed` and successor `operation.after.token`. A merged or already-closed PR cannot be closed again.

## Recommended workflow

1. Activate `agent-libos-git-inspection`. Confirm repository identity, inspect the main state, and list exact local base/head branch refs and OIDs. Review branch ancestry and status before creating a PR.
2. Compute `git_diff(scope="range", base=<base_oid>, head=<head_oid>)`. Compare its full `sha256` with the eventual PR `patch_sha256`; `patch` is lossy display text and `patch_b64` is the exact returned bytes. If content is truncated, do not claim complete review even though the full digest is available.
3. Create with the exact local branch names and fresh main token. Record `pr_id`, base/head refs and OIDs, patch/title/body hashes, snapshot refs, and `operation.after.token`.
4. Discover with `git_list_pull_requests`, but inspect the exact PR before reviewing, closing, or merging. Recompute the exact frozen range diff and compare its SHA-256 to `patch_sha256`; also check current branches separately because snapshots and live branches serve different purposes.
5. Record a review only after examining the complete change and any tests. Since review bodies are not readable through the result, keep the returned `review_id`, decision, body hash, and operation evidence if exact comment accountability matters.
6. Before merge, inspect the chosen worktree, ensure the base branch is checked out at the recorded OID and fully clean, and acquire its own fresh token. Select ff-only integration, a `--no-ff` merge request, or squash based on the requested history policy, accounting for the already-up-to-date and empty-delta boundaries above.
7. After merge, inspect PR status, snapshot refs, selected worktree status, base branch/`HEAD`, resulting commit, and exact range/content. Run task-specific tests. After close, inspect the PR and refs to prove preserved evidence and unchanged branches.

Do not treat “approved” as permission to merge or “request_changes” as a technical lock. If policy requires review-state gating, enforce it in the calling workflow and show the reviewed evidence explicitly.

## Failure and recovery

All PR writes are non-idempotent effects. The model-facing error omits some provider settlement details. Treat only a clearly pre-dispatch validation, permission/approval, or initial repository-token CAS rejection as effect-not-started; after any other failure, conservatively read and reconcile refs, metadata, and the relevant worktree before retrying. A `stale_state` alone is insufficient because PR metadata CAS can detect concurrent change after a review/close metadata effect began or after merge already changed Git state.

- On an initial token `stale_state`, no PR mutation/write effect from that invocation started, although protected read-only provider observations may already have occurred. A later `stale_state`—for example, metadata CAS after a Git merge—can follow a state-changing effect and must be reconciled. In either case, re-inspect PR metadata/snapshots, live base/head refs, the relevant worktree status, and exact diff. Use a fresh token from the operation's actual worktree context and repeat the decision, not just the call.
- On pre-dispatch Human approval, preserve exact arguments. Creation's stable ID supports the same approved retry while state is unchanged; merge approval is one-use and tied to exact scope. If state changes while waiting, obtain a new observation and approval decision.
- Creation writes snapshot refs before metadata. Any failure after possible dispatch—not only timeout or a literal `unknown_effect`—can leave snapshot refs without metadata, metadata with uncertain acknowledgement, or a complete PR. Reconcile `git_list_pull_requests`, `git_list_refs(kind="pull_requests")`, and exact inspection before retrying.
- Review writes metadata with a new random review ID. After an uncertain result, inspect the review list for actor, decision, timestamp, and body hash. Because body text is not returned and a second call creates a new ID, do not blindly replay.
- Merge can change `HEAD`, index, or worktree before PR metadata is marked merged. Inspect selected-worktree `HEAD`/branch/status, PR status, `merged_oid`, live branches, snapshots, and commit topology. Because hidden operation state is not directly readable, invoke integration recovery only for a kind known from the attempted strategy/error; otherwise stop. Do not issue a second merge merely because metadata remains open.
- Close can persist even if acknowledgement is lost. Exact inspection resolves ordinary cases; never retry an already closed PR.
- Malformed/missing metadata, orphan snapshot refs, snapshot/metadata mismatch, or a merged worktree with stale PR metadata has no typed repair operation in this Skill. Preserve the evidence, report the inconsistent state, and request an authorized administrative recovery path. Raw shell edits to refs/JSON are not an escape hatch.
- Query-only external-effect recovery performs a fresh local repository-state read, including refs, worktree state, and the simulated-PR metadata digest; it does not replay PR writes or compare that evidence into a definitive outcome. It cannot repair partial snapshot/metadata state, and `unknown` remains unknown until a caller reconciles the fresh repository evidence.

## Completion evidence

For creation, report the local-only nature of the PR, exact `pr_id`, title/body hashes as needed, base/head refs and frozen OIDs, `patch_sha256`, verified snapshot refs, and `operation.after.token`.

For review, report PR ID/status, reviewed frozen range and matching patch hash, test evidence, decision, review ID, actor, timestamp, body hash, and successor token. State that the review body itself is not returned by inspection.

For merge, report selected worktree, clean precondition, strategy, approval/authority outcome, old base/head OIDs, resulting `merged_oid`, whether HEAD advanced or the integration was already up to date, whether a new commit was actually created, commit topology appropriate to the strategy, post-merge clean status, PR status, preserved snapshots, successor token, and tests. For close, report closed status, unchanged branch OIDs, preserved reviews/snapshot refs, and successor token.

Never claim an external forge PR exists, all PRs were enumerated after a truncated list, review text was read when only its hash is visible, or checkpoint/image durability for Git common-directory metadata.
