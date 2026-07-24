---
name: agent-libos-git-change-recording
description: Stage, unstage, and commit selected changes in the managed Git repository. Use when reviewed workspace changes should be recorded intentionally without changing branches or integrating history.
allowed-tools: git_stage git_unstage git_commit
---
# Record Git changes

## Workflow

1. Activate Git inspection and review status/diff, preserving unrelated user changes.
2. Stage only explicit literal paths with the freshest state token.
3. Re-read status and staged diff; unstage mistakes with the newly returned/read token.
4. Commit the reviewed index with a concise message and fresh token; amend only when explicitly intended.

## Boundaries and safety

- Every mutation uses compare-and-swap. On stale state, re-read and reconsider; never blindly replay an old token.
- Staging is not filesystem authority, and tool visibility is not Git write approval.
- Commit identity comes from trusted Host/repository configuration; author overrides are unavailable.

## Verify

Inspect the resulting status and commit. Confirm the recorded tree contains exactly the intended paths and remaining changes are preserved.
