---
name: agent-libos-object-tasks
description: "Run one visible tool asynchronously in a Host-managed child bound to an owner Object, then inspect, wait, watch, or cancel it. Use for single background calls; use child-processes for multi-step Agent work."
allowed-tools: start_object_task get_object_task list_object_tasks wait_object_task watch_object_task_owner cancel_object_task
---
# Run one Object-bound tool call

An ObjectTask is a durable record plus a Host-managed runner with exactly one tool. It is not a planning Agent or authority bypass. Activate the target Skill before forming `args`.

Use one intentionally mutable owner from object-memory. It supplies identity, not target-tool authority; immutable handles normally lack required write.

## Tool guide

### `start_object_task`

Queues one call and returns `{task: ...}`. Pass `owner_oid`, or `owner_name` plus optional `namespace` (process scope by default). OID wins if both appear, so pass one.

- `tool` must be visible in the creator's process tool table; `args` must match its schema. Start success proves admission, not execution or an external effect.
- Admission needs owner read/write/link, `process:spawn`, budget, and capacity. Finite owner grants can be consumed at commit.
- Runner gets the exact tool and owner read/materialize only. Delegate minimal exact specs through `inherit_capabilities`; ambient and non-delegable authority does not follow.
- `notify_pid` defaults to creator and permits self, direct parent, or direct nonterminal child. Prefer `normal`; use `interrupt` only for needed preemption and a specific channel.
- A notified `result_oid` grants nothing. `grant_result_to_notify=true` requests a handle for another recipient. Missing creator grant can fail/discard; label or delivery failure can leave `succeeded` without that handle. Inspect both records.
- `owner_watch=true` sends selected `updated`/`linked` references to the runner. Watch channel/kind are separate from completion notification and grant no payload access.

Save `task_id` and `runner_pid`. Publish/claim/lease/heartbeat are Host-internal, not tools.

### `get_object_task`

Returns full state for one `task_id`. Creator can inspect; others need owner read, possibly finite. It can reconcile a dead runner and retry terminal notification in `none`/`failed`, so status reads may perform housekeeping.

### `list_object_tasks`

Lists visible tasks; filter with `owner_oid`, `include_terminal` (default true), and `limit` 0–1000. Filtering precedes limit; non-creators need owner read. Reconciliation/grant consumption can occur. No cursor or `has_more` exists, so a full page is incomplete evidence.

### `wait_object_task`

Waits by ID. `timeout_s` is omitted or 0 through the configured maximum; omission may block, so prefer finite. Zero is nonblocking but can reconcile/retry notification.

Timeout can return `queued`/`running` without failing or cancelling. Explicit waits expose a dependency in `wait`. Terminal state may precede notification settlement; poll again if delivery matters.

### `watch_object_task_owner`

Changes owner notices for an active ID. `watch_events` accepts `updated`/`linked`; empty preserves defaults. Channel/kind target runner. Terminal tasks reject changes. It neither restarts nor grants access. Pair with target `receive_process_messages` using matching filters.

### `cancel_object_task`

Requests cancellation with optional `reason`. Creator manages; others need owner write. Terminal returns unchanged. No effects roll back, and running synchronous side effects reject unsafe cancellation. Wait for `cancelled`; after rejection, reconcile and continue waiting.

## Recommended workflow

1. Confirm one schema-known call; use child-processes for multi-step work. Activate this, target, and owner-memory Skills.
2. Select/create one mutable, stably named owner and retain its OID. Confirm read/write/link without inventing authority.
3. Validate args, minimal delegation, notification, result grant, and watch. Default to no extra grant/watch and normal notification.
4. Call `start_object_task` once; record `task_id`, `runner_pid`, owner, tool, and initial status. `queued` is not success.
5. Wait with a finite bound. On `queued`/`running`, reuse the ID with backoff; never duplicate on timeout.
6. For `waiting_human`, settle `wait.request_id`, then wait again. Rejected `request_permission` can resume and return denial; other rejected human waits normally fail.
7. For message wait, send to `runner_pid` using `wait.filters`; for process wait, resolve `wait.child_pid`. Auto-replay applies only when target itself is `receive_process_messages` or `wait_child_process`, never arbitrary effects.
8. Treat only `succeeded` as tool success. Inspect `result_oid`, `error`, notification status/error, and timestamps, then run the target Skill's authoritative verification.

Exact active statuses: `queued`, `running`, `waiting_human`, `waiting_process`, `waiting_message`. Exact terminals: `succeeded`, `failed`, `cancelled`, `abandoned`, `superseded_by_restore`, `result_unavailable_after_reopen`. There is no `completed` or generic `waiting`.

Success publishes an immutable ToolResult owned by the task, linked as produced. A live creator normally gets read/materialize/link plus a MemoryView root; task info does not inline content.

## Failure and recovery

- Ambiguous start: list exact owner before retry; start is non-idempotent. Correct owner/tool/args/delegation/capacity.
- Timeout in `queued`/`running`: wait on the same ID. A replacement can duplicate provider or filesystem effects.
- Explicit wait: satisfy only its recorded dependency, then wait. Stop if a non-replay-safe target is stranded.
- `failed`: inspect `error` and follow the target Skill. A tool may fail after an external effect, so do authoritative read-only reconciliation before another call.
- `cancelled`: report no rollback and inspect possible effects. After rejected cancellation, reconcile/wait; do not control the runner via unrelated tools.
- Notification `failed`: get/wait may retry while recipient is valid. `undelivered_terminal` means that recipient ended. Either can coexist with target success; a message proves neither consumption nor result access.
- Result grant: verify recipient handle. Fix missing authority before replacement; after label/delivery denial use an authorized handoff, never weaker labels.
- Reopen: active tasks become `abandoned`; missing live results become `result_unavailable_after_reopen`. Args/delegations are not persisted, so review evidence before recreation.
- Restore: scoped active tasks block restore; later history may become `superseded_by_restore`. Restore does not undo completed external effects.

Identity/status, notification/watch, result/error/wait are durable; args, delegation, and result-grant intent are live only. Creator exit does not cancel; active tasks pin owners.

## Completion evidence

Retain IDs, owner, creator/runner, tool, final status, and time. Success requires `status="succeeded"`; admission, runner exit, notification, or result alone is insufficient.

If a result is expected, retain `result_oid` and prove the intended reader can read it. Separately retain notification recipient, `status`, `message_id`, and error. `delivered` proves message publication only, not consumption or Object access.

For external effects, add target-Skill read-back. Task success proves tool return/result publication, not continuing external correctness.

For non-success, preserve exact status, `error` or structured `wait`, dependency, reconciliation, and duplicate-effect risk. Finish only when every required effect and access expectation has evidence; otherwise report an explicit blocker or unknown state.
