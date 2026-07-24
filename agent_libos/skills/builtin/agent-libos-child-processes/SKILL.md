---
name: agent-libos-child-processes
description: Spawn or fork direct AgentProcess children, coordinate their lifecycle and messages, then merge visible results. Use for bounded parallel delegation, isolated exploration, or structured parent-child collaboration.
allowed-tools: fork_child_process spawn_child_process list_child_processes wait_child_process signal_child_process merge_child_memory send_process_message read_process_messages receive_process_messages
---
# Coordinate child processes

## Workflow

1. Spawn for a fresh goal-only MemoryView; fork when selected parent roots or copy/speculative semantics are required.
2. Delegate only explicit capabilities and a bounded budget at creation or through the authority Skill.
3. Use messages for structured coordination. `read_process_messages` is an immediate snapshot; `receive_process_messages` can suspend for a filtered unread match.
4. After acknowledging human input, merge it into the current cumulative goal. Treat it as an incremental constraint unless it explicitly replaces or cancels earlier requirements.
5. Wait for terminal state, inspect the outcome, then merge result-visible memory if needed.

## Boundaries and safety

- These are Agent libOS processes, not host OS processes. Parent/child relationships restrict messaging, signals, and merges.
- Reads acknowledge returned unread messages by default; set `ack=false` only when deliberate redelivery is needed.
- Signals and merges are mutations. Do not merge speculative or failed output without inspection.

## Verify

Confirm child terminal status, result OID, relevant messages, and exactly which object OIDs were merged or skipped.
