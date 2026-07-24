---
name: agent-libos-object-tasks
description: Start, inspect, wait for, watch, or cancel an asynchronous tool task owned by an Object. Use when work should continue in a background child while remaining attached to durable Object state and process notifications.
allowed-tools: start_object_task get_object_task list_object_tasks wait_object_task watch_object_task_owner cancel_object_task
---
# Run Object tasks

## Workflow

1. Select an owner object with read, write, and link authority and a tool already visible to the caller.
2. Start the task with only the capabilities its runner needs and a precise notification target/channel.
3. Inspect for a snapshot; wait when the current process should pause until a terminal or explicit waiting state.
4. Enable owner watching only when changes should notify the runner. Cancel explicitly when its result is no longer needed.

## Boundaries and safety

- Object tasks execute through child processes; starting one does not bypass tool visibility, Capability, approval, or data-label controls.
- Waiting states may still require human or external input; inspect the returned `wait` data.
- Grant result access to the notification process only when that data flow is intended.

## Verify

Require a terminal status and inspect `result_oid` or `error`; do not infer completion solely from a notification.
