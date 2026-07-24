---
name: agent-libos-git-patch-objects
description: Create an immutable CODE_PATCH Object from Git state or apply such an Object to a managed worktree. Use when changes need durable lineage, reviewable transfer, or byte-safe application across process boundaries.
allowed-tools: git_create_patch git_apply_patch
---
# Transfer Git patches as Objects

## Workflow

1. Inspect the chosen worktree, scope, range, and literal paths.
2. Create a patch Object and retain its OID and lineage metadata rather than copying rendered diff text.
3. Before applying, inspect the destination worktree and obtain a fresh state token.
4. Apply to the worktree or index as intended, then inspect status and diff.

## Boundaries and safety

- Patch creation writes Object Memory but not repository state; patch application mutates Git/filesystem state.
- Only immutable validated patch Objects are accepted, and their data labels propagate to the destination.
- A clean parse does not guarantee semantic correctness or conflict-free behavior; preserve unrelated changes.

## Verify

Confirm the patch Object is complete, then compare the applied status/diff with its intended scope and ensure no unexpected paths changed.
