---
name: agent-libos-git-branches-worktrees
description: Create, rename, or delete branches; create or delete tags; switch worktree targets; and manage runtime-owned Git worktrees. Use when repository topology or the active target must change deliberately.
allowed-tools: git_branch git_switch git_tag git_worktree
---
# Manage Git branches and worktrees

## Workflow

1. Inspect status, refs, and managed worktrees, then obtain a fresh state token.
2. Use exact validated names and start refs; prefer non-force operations.
3. Re-read state after each mutation and use its new token for any next step.
4. Remove only a known `managed_worktree_id` after inspecting its local state.

## Boundaries and safety

- Branch/tag deletion, forced switching, forced replacement, and worktree removal can destroy reachability or local state and require exact authorization.
- `git_worktree` manages runtime-named worktrees only; it does not accept an arbitrary host path.
- Never switch or remove a worktree with unreviewed changes merely to make another operation succeed.

## Verify

List refs/worktrees and read status to confirm the active target, created/deleted name, HEAD, and preservation of local changes.
