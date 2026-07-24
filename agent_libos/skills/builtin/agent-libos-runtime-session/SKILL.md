---
name: agent-libos-runtime-session
description: Manage the current AgentProcess session, time, deliberate waiting, context compaction, and terminal exit. Use for deadlines/backoff, long-context maintenance, or completing the current process with a durable result.
allowed-tools: process_exit compact_process_context get_current_time sleep
---
# Manage the runtime session

## Workflow

1. Read runtime time when a deadline, timestamp, or elapsed interval matters.
2. Use `sleep` only for intentional time-based waiting; use child, message, or Object-task waits for event completion.
3. Compact context when pressure threatens continuity, preserving goals, constraints, decisions, evidence, and pending work.
4. Before final reporting, re-read the original goal, acknowledged human follow-ups, and any durable acceptance ledger. Require evidence for every cumulative deliverable; passing one test or completing the latest follow-up is not whole-goal completion.
5. Follow the image's exit contract. Most images call `process_exit` once after the gate. If it returns a nonterminal cumulative review, resolve its missing items, send any required final human output, then retry with the fresh token and structured evidence.

## Boundaries and safety

- Compaction changes retained prompt context, not durable external state; inspect its result before continuing.
- Sleeping does not poll or prove that an external event completed.
- Exit is terminal. Do not call it while any original or follow-up requirement, required write, child merge, verification, checkpoint, or user-facing output remains pending.

## Verify

Before exit, check that requested work is complete, relevant results are durable, and no required wait or report remains.
