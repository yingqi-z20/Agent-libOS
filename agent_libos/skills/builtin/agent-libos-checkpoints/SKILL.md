---
name: agent-libos-checkpoints
description: Capture, inspect, compare, restore, or fork durable process-subtree checkpoints. Use for recoverable internal-state milestones, isolated replay, or deliberate rollback of reconstructable Agent libOS state.
allowed-tools: create_checkpoint list_checkpoints inspect_checkpoint diff_checkpoint restore_checkpoint fork_checkpoint
---
# Manage checkpoints

## Workflow

1. Create a checkpoint only after final edits, requested verification, and Git inspection form a stable boundary; record why it exists.
2. Inspect its processes/modules and diff it from live state before a restore or fork.
3. Prefer `fork_checkpoint` for isolated exploration that must not replace the live subtree.
4. Restore only when replacing the live subtree is explicitly intended; inspect reconciliation and post-commit failures afterward.

## Boundaries and safety

- Checkpoints capture reconstructable internal state. They do not clone or roll back external provider state or completed external effects.
- Restore supersedes live messages, tasks, and human requests and requires exact image/restore authority.
- A fork remaps process and Object IDs; consume the returned maps rather than retaining old identifiers.
- A later workspace mutation is not covered by an earlier checkpoint; reverify and create a new checkpoint when the final state changed.

## Verify

Check restored/forked PID and Object mappings, state status, reconciliation flags, and the reported external effects since capture.
