---
name: agent-libos-git-inspection
description: Read-only inspect the fixed Runtime Git repository, managed worktrees, status, diffs, history, refs, remotes, and blame before making or verifying a Git decision.
allowed-tools: git_repository_info git_status git_diff git_log git_show git_blame git_list_refs git_list_remotes git_list_worktrees
---
# Inspect Git state

Git tools operate only on the Runtime workspace repository. They do not accept a repository path, cwd, arbitrary argv, URL, credentials, or revision expression. `worktree_id="main"` selects the root checkout; another id must come from `git_list_worktrees`. Returned host paths such as `root`, `git_dir`, `common_dir`, and a worktree `path` are observations, never selectors.

Exact commit inputs accept a full object-format OID, a validated full ref below `refs/heads/`, `refs/tags/`, `refs/remotes/`, or `refs/agent-libos/`, or a short local branch name. Do not use `HEAD`, abbreviated OIDs, `~`, `^`, `@{}`, ranges, path/ref ambiguity, or option-like values. Use two exact refs/OIDs in the dedicated `base` and `head` fields when a range is required.

Git paths are byte-preserving objects shaped as `{display,path_b64,lossy}`. For a later path input pass exactly one UTF-8 string, `{"path": display}`, or `{"path_b64": path_b64}`. Never copy the whole returned path object or send both forms. If `lossy=true`, only `path_b64` preserves identity. A renamed status entry can also have `original_path`; review both old and new paths.

## Tool guide

### Repository, status, and CAS

- `git_repository_info` reports repository/worktree identity, object format, Git version, HEAD, structural paths, and `state.token`. It does not inspect changed file contents and its returned paths cannot be fed back as a repository selector.
- `git_status` reports the selected worktree's branch, configured upstream, ahead/behind counts, HEAD OID, and bounded byte-safe entries. Interpret `index_status` and `worktree_status` independently. It requests all untracked paths but does **not** request ignored paths, so an untruncated clean result does not prove that ignored content is absent. `truncated=true` means other tracked or untracked entries may also have been omitted.
- A non-null status `submodule` field describes a parent-repository gitlink entry; it is not nested-repository status. The hardened provider disables recursive submodule inspection. Parent status/diff/show and staging a submodule path observe or record the gitlink OID, not the nested checkout's complete files or history. If nested submodule evidence is required, this fixed-repository tool surface cannot provide it; stop instead of claiming completeness.

A state token is a 64-hex local compare-and-swap **state digest**, not a nonce or one-use credential. It commits to repository and selected-worktree identity, HEAD, index, effective configuration, refs, worktree registry, simulated-PR metadata, and bounded selected-worktree status/content. The same state can produce the same token before and after a successful no-op, and a token is not consumed merely by being supplied. It is not remote network state, Capability, Human approval, Object authority, or proof about ignored, nested-submodule, or unrelated workspace files.

Return shapes differ:

- Git reads return the observation token at `state.token`.
- `git_create_patch`, from the patch Skill, is an Object side effect but also returns its source observation at `state.token`.
- Mutations returning a `GitOperationResult` expose their post-operation observation at `after.token`; it can equal `before.token` when local state did not change. Likewise, `changed_paths` can be a selected/reconciliation manifest rather than proof that every path changed.
- Simulated-PR mutations wrap that result at `operation.after.token`.

Use a token only for the same repository/worktree observation and exact decision that produced it. Any relevant edit, index/ref/config/worktree/PR mutation can make it stale. A token does not freeze state between calls; whether its hex value changed is never a substitute for readback.

### Diffs and commits

- `git_diff` with `scope="worktree"` reads tracked unstaged changes versus the index. It does not include untracked file contents. `scope="staged"` reads the index versus HEAD. Both require `base=null` and `head=null`. `scope="range"` requires both exact `base` and `head` and resolves them to commit OIDs. `paths` are literal restrictions, not patterns. Diff generation disables rename detection, external diff, and textconv and includes binary patch data, so a rename can appear as delete plus add.
- `git_log` returns bounded structured commit metadata from `ref`, or from the current HEAD OID when `ref=null`. It does not return patches or accept a path filter. `truncated=true` means more commits exist beyond the returned limit.
- `git_show` resolves exactly one commit and returns its structured metadata plus explicit root/per-parent diffs. `parent_diffs` has one entry with `parent_oid=null` for a root commit, one entry for an ordinary commit, and one entry per parent in `commit.parents` order for a merge or stash commit. Each entry has its own byte-safe `changed_paths`, `patch`/`patch_b64`, `truncated`, `bytes`, and `sha256`. The top-level patch fields are a compatibility view of the root or first-parent entry, identified by `patch_base_oid`; they are never complete evidence for every side of a multi-parent commit. A tag is peeled to a commit for this operation.
- `git_blame` returns bounded raw porcelain for one literal path. When `ref=null` and HEAD exists, it deliberately blames the current HEAD OID, not dirty worktree bytes. Set an exact ref/OID to select another committed version. Use workspace-reading tools when the task is to inspect uncommitted file content.

For `git_diff`, the top-level `git_show` root/first-parent view, and each `git_show.parent_diffs` entry, `patch` is a convenience UTF-8 rendering and `patch_b64` is the exact returned byte prefix. For `git_blame`, the analogous fields are `content` and `content_b64`. If the corresponding `truncated=true`, neither field is complete. `git_show.parent_diffs_truncated=true` means at least one parent view is incomplete; raise `max_bytes` within its hard bound or inspect that exact parent/commit pair with `git_diff(scope="range")`. Where a full capture fit under the hard bound, `bytes` describes its full byte length and `sha256` its full digest even when only a soft-limited prefix is rendered. For limit-driven listings such as log or refs, the digest covers the bounded capture, not all history or all refs.

### Refs, remotes, and worktrees

- `git_list_refs` lists bounded structured refs by `kind="branches"`, `"tags"`, `"remotes"`, `"pull_requests"`, or `"all"`. It accepts no arbitrary prefix or cursor. A tag row's `oid` can be an annotated tag object OID rather than its peeled commit. `kind="all"` can also expose other refs such as the stash tip. On truncation, narrow by kind or raise the bounded limit; never infer absence.
- `git_list_remotes` returns configured remote names, validated fetch refspecs, and redacted fetch/push URL placeholders with SHA-256 fingerprints. The refspecs reveal only which branch or wildcard maps into `refs/remotes/<remote>/`; none of these observations are usable transport inputs or grant remote authority. This tool does not contact a remote or read its current refs.
- `git_list_worktrees` accepts no worktree selector and lists only `main` plus structurally trusted worktrees under the configured managed root. Unmanaged external linked worktrees are omitted. The returned `state.token` is always a main-repository token. Structural visibility is not proof that the current Runtime originally created a worktree, and returned `path` values are not accepted by mutation tools.

All reads are bounded and can fail with `output_too_large` instead of returning a partial result when a configured hard limit is crossed. They require Git read/diff authority and data-flow clearance; seeing a tool or receiving a token grants none of the later filesystem, repository, remote, or PR rights.

## Recommended workflow

1. Select `main` or an exact returned `worktree_id`. Start with `git_repository_info` when repository identity or object format is unknown, then call `git_status`.
2. Check `truncated` before interpreting entries. For renames, retain both `path` and `original_path`; for lossy paths retain only their `path_b64` forms.
3. Read both `git_diff(scope="worktree")` and `git_diff(scope="staged")` for a change-recording decision. Inspect untracked content through workspace tools because worktree diff omits it. Inspect ignored paths separately when the task names them, and stop at a submodule gitlink when nested-repository evidence is required.
4. Resolve history questions with `git_log`, then use `git_show` on exact OIDs. For a multi-parent commit, decide whether the requested meaning is its first-parent tree delta or every parent comparison; inspect the matching complete `parent_diffs` entries and never generalize from the top-level first-parent view alone. For a proposed range, call `git_diff(scope="range", base=<exact>, head=<exact>)` and require a complete patch.
5. Use `git_list_refs`, `git_list_remotes`, or `git_list_worktrees` for topology. Re-read status for every affected worktree rather than treating a main checkout observation as its local state.
6. Immediately before a mutation, repeat the narrow status/diff/ref read after the last edit or test and carry its newest same-worktree `state.token`. Re-evaluate the operation, paths, and destructive scope; do not merely replace an old token.

## Failure and recovery

- `repository_busy` means the bounded repository lock was unavailable. Wait briefly, then perform a fresh read; do not assume the prior observation is current.
- `stale_state` during a read means repository state or data-flow lineage changed while it was observed. Discard the result and read again. For a mutation, only an initial state-token CAS rejection before the first state-changing provider effect certifies that no mutation from this invocation began; read-only preflight/provider observations may already have occurred. A multi-phase operation can instead detect drift after an earlier phase; notably, pull can fetch its tracking ref and then return `stale_state` before integration. Use the operation Skill's reconciliation procedure rather than treating the code alone as proof of no effect.
- `output_too_large` or `truncated=true` is incomplete evidence. Narrow literal paths, reduce the ref kind/history question, or use a supported complete artifact workflow. Never summarize omitted content as clean or absent.
- Lossy rendered text is not byte evidence. Use the corresponding base64 field or path token.
- `unsafe_repository`, `unsafe_repository_config`, or an unsupported operation is a stop condition. Do not bypass the typed boundary with shell Git; direct shell Git supports only a small hardened inspection subset and does not add missing typed mutations.
- A read succeeding does not make a later mutation idempotent. The model-facing Git error projection does not expose every provider settlement detail. After any mutation error that is not clearly a pre-dispatch validation, permission, approval, or **initial** state-token CAS rejection, conservatively assume dispatch may have started and use the mutation Skill's read/reconciliation procedure before any retry; do not wait for a literal timeout or `unknown_effect` code.

## Completion evidence

An inspection claim is complete only when all of the following hold:

- the intended repository and exact `worktree_id` are identified;
- status, relevant diffs/history, and all byte-safe paths are untruncated, with ignored and submodule blind spots stated explicitly;
- untracked content was not mistaken for worktree diff content;
- exact refs resolved to the intended full OIDs, with no revision-expression guesswork;
- patch/content byte fields and their `bytes`/`sha256` meanings were interpreted correctly, and every required merge/stash parent diff was present and untruncated;
- remote URLs were treated as redacted observations and no local token was called remote proof;
- the final same-worktree `state.token` was obtained after the last relevant change;
- any subsequent mutation is separately authorized and verified by a new read.

Stop rather than claim completion when state changed, any required result is truncated, a lossy path was reconstructed from display text, a worktree is omitted/unmanaged, or the typed surface cannot provide authoritative evidence.
