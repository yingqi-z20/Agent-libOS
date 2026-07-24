---
name: agent-libos-git-integration-recovery
description: Restore paths, integrate or abort history operations, manage stashes, reset commits/index/worktree, and clean untracked content. Use for deliberate integration or recovery after inspecting exact repository state.
allowed-tools: git_restore git_integrate git_stash git_reset git_clean
---
# Integrate and recover Git state

## Workflow

1. Inspect status/diff/history and obtain a fresh state token before every operation.
2. Use typed merge, rebase, cherry-pick, revert, or matching abort; inspect conflicts before deciding the next operation.
3. Use stash for bounded temporary preservation. Restore only explicit paths and exact sources.
4. Preview the affected state before reset or clean, then verify immediately afterward.

## Boundaries and safety

- Hard reset, clean, restore-to-worktree, stash drop/clear, and some abort paths can discard data and require destructive authority/approval.
- Empty clean paths broaden scope to all eligible untracked paths; add directories/ignored content only when explicitly intended.
- On stale state, re-read and reconsider instead of replaying the operation.

## Verify

Read status, relevant diff, HEAD, and stash state. Confirm conflicts are resolved/aborted and no unrelated content disappeared.
