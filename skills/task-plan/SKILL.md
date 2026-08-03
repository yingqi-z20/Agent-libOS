---
name: task-plan
description: Create, read, and update revisioned task plans in Agent libOS Object Memory. Use for multi-step work that needs explicit progress states, resumable plan context across prompt compaction, stale-update detection, or an auditable in-runtime plan history.
license: Apache-2.0
metadata:
  agent-libos.version: v0
  agent-libos.jit-tools: references/agent-libos/jit-tools.json
---

# Manage a task plan

Keep one named plan as a mutable Object Memory ledger. The ledger survives
prompt changes and compaction inside the current live Runtime. It is not a
durable payload after Runtime reopen; use an authorized checkpoint or export
workflow when reopen durability is required.

## Plan contract

Represent every step as:

    {"step":"Inspect the failing path","status":"in_progress"}

Use only these statuses:

- pending: not started.
- in_progress: actively being worked; allow at most one per revision.
- blocked: cannot currently proceed.
- completed: finished and verified.
- cancelled: deliberately removed from execution.

Keep 1 to 32 steps. Trim step text and keep each step between 1 and 512
characters. Supply explanation as a string of at most 2048 characters or null.
Revisions are flexible snapshots: add, remove, reorder, reopen, or cancel steps
when reality changes. Do not infer a terminal transition matrix.

## Tool contracts

The three JIT tool names and argument contracts below are authoritative. In
direct JIT exposure, call the tool by name. In multiplexed exposure, retain
these exact contracts and call run_jit_tool with tool_name set to the exact
name and arguments set to the object shown. Never guess a hidden contract.

### create_task_plan

Create one mutable Object Memory object of type plan. All four fields are
required; use null for the process-default namespace or for no explanation.

    {
      "name": "implementation-plan",
      "namespace": null,
      "explanation": "Inspect, implement, and verify.",
      "plan": [
        {"step": "Inspect the current implementation", "status": "in_progress"},
        {"step": "Implement the change", "status": "pending"},
        {"step": "Run focused tests", "status": "pending"}
      ]
    }

Creation starts at revision 1. A duplicate name fails; it is never treated as
an upsert. Read and verify an existing plan before deciding to reuse it.

### read_task_plan

Read and validate the latest snapshot:

    {"name":"implementation-plan","namespace":null}

The result includes Object identity, memory_version, revision, explanation,
plan, and status_counts. Treat a malformed type, schema version, revision
sequence, or snapshot as an explicit blocker; do not guess a current state.

### update_task_plan

Replace the logical plan with a new complete snapshot:

    {
      "name": "implementation-plan",
      "namespace": null,
      "expected_revision": 1,
      "explanation": "Inspection finished; implementation is active.",
      "plan": [
        {"step": "Inspect the current implementation", "status": "completed"},
        {"step": "Implement the change", "status": "in_progress"},
        {"step": "Run focused tests", "status": "pending"}
      ]
    }

Use the revision returned by the latest successful create, read, or update.
The tool rejects stale revisions. An identical current snapshot is a no-op. An
immediate retry whose expected revision is exactly one behind the identical
current snapshot is also a no-op, preventing a duplicate append after an
ambiguous result.

## Workflow

1. Choose one stable, task-specific Object name. Prefer the process-default
   namespace unless an existing authorized shared namespace is intentional.
2. Create the plan once, retaining its Object identity and revision.
3. Read before acting after compaction, handoff, or any uncertain tool result.
4. Update with the complete current plan and the exact expected revision.
5. Re-read before completion and map every remaining step to a final status or
   an explicit blocker.

Serialize updates. Do not issue parallel update_task_plan calls for the same
Object: Object Memory append has no caller-supplied compare-and-swap token.
After an unknown outcome, read first and retry only if the intended snapshot is
not already current.

Skill activation changes tool visibility only. It grants no Object Memory
authority and creates no explicit namespace. Every create, read, and append
still passes normal namespace/Object Capability, data-flow, budget, event, and
audit enforcement.

## Failure recovery

- Duplicate create: read the exact name and reuse it only when its validated
  ledger belongs to this task.
- Stale revision: read again, reconcile the latest snapshot, and submit a new
  complete update.
- Permission or namespace denial: obtain the least required authority or use
  the process-default namespace; never route around policy.
- Size limit: finish or replace the plan with a deliberately summarized new
  named plan. Do not truncate or encode JSON to evade the limit.
- Runtime reopen with missing payload: recover from an authoritative
  checkpoint/export. Metadata, Object name, OID, or version alone is not plan
  content.
