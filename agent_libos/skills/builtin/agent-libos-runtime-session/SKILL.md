---
name: agent-libos-runtime-session
description: Read wall-clock time, make one bounded timer delay, compact an enabled LLM-context Object, or finish the current AgentProcess. Never use timer sleep as an event wait or treat runtime-local Object payloads as reopen-durable storage.
allowed-tools: process_exit compact_process_context get_current_time sleep
---
# Manage time, context, and process completion

Activating this Skill exposes schemas but grants no underlying authority. Make a
wait, compaction, or exit the only call in its turn because it can stop later
calls in a parallel batch.

## Tool guide

### `get_current_time`

Optional `timezone` must be an IANA name such as `UTC` or `Asia/Shanghai`;
omission uses the default shown by the current tool schema. It requires `read`
on `clock:now` and returns `iso8601`, `unix_seconds`, and `timezone`. It is wall
clock, not a monotonic timer; re-read it after a deadline delay.

### `sleep`

Required finite `seconds` is `0..` the active-schema maximum (Host default 60).
It requires `read` on `clock:sleep` and returns requested and measured elapsed
seconds, which need not match. Use it only for one known delay. It never observes
completion. Activate the owning Skill and use `wait_child_process`,
`receive_process_messages(block=true)`, or `wait_object_task` for a child,
mailbox message, or Object task. Never poll an event with repeated sleeps.

Sleep crosses an observed provider boundary. After timing begins, cancellation
or failure may mean the delay already happened and one-use authority was
consumed. Only an error explicitly certifying that the initial provider
observation did not start proves no delay. Otherwise re-read time and relevant
state; do not blindly sleep again.

### `compact_process_context`

Call manually only when the prompt contains the literal `LLM context object:`
heading. A source-only prompt has no context Object; a pressure notice alone is
not eligibility. Do not duplicate Runtime-managed automatic maintenance.

Arguments/defaults:

- `target_tokens=4000`, range `256..64000`: no-op threshold and summary goal,
  not a hard final-size limit.
- `preserve_recent_entries=8`, range `0..128`: verbatim tail-entry count, not a
  token/message count; large retained entries can keep output above target.
- `max_chunks=8`, range `1..64`: maximum sequential summary stages.
- `force=false`: bypasses only the under-target no-op, never authority, budget,
  schema, version, or child-output validation.

Each stage spawns one `context-compressor:v0` child and rolls its summary into
the next stage; eight chunks can cost eight child/LLM turns. Use the smallest
safe value. Recovery may replace a child with a vanished result, so
`compressor_pids` can exceed completed stages.

Every fresh manual call needs read/write/materialize access to the current
context Object. An unforced under-target call returns before creating a job or
child, so do not pre-emptively request namespace, spawn, image, or child/LLM
authority for that no-op path. Once actual compaction must start, creating a new
job needs write on the process Object namespace; child stages need `process:spawn` write,
`image:context-compressor:v0` read when booting that different image, and
sufficient tool/child/LLM budgets. An existing job may already have its job
Object but still needs the authority and budget for any remaining stages.
Automatic maintenance separately requires `context:maintenance/execute`; that
eligibility is distinct from the prerequisites of a direct manual call.

Runtime stores/resumes a suspended child wait; never pass `_resume_job`. Success
returns `compacted=true`, `reason="compacted"`, context OID, versions, child PIDs,
token counts, and actual preserved count. An unforced small context returns
`compacted=false`, `reason="context_under_target"`, equal versions, and is a
successful stop.

Writeback uses a source-version compare-and-swap. Durable context generation is
advanced before volatile Object writeback, so even a final version conflict can
leave the original payload unchanged while advancing generation; the next
quantum must rematerialize. Success also advances generation. Compaction is not
reopen persistence: ordinary Runtime reopen can lose runtime-local context
payload history even though generation, labels, and evidence survive.

### `process_exit`

Call alone after every write, wait, merge, requested Git/checkpoint action,
verification, queued message, and required Human output is complete. Result
input precedence is exact:

1. Nonempty `result_oid` reuses an existing readable Object and overrides everything;
   an empty string is rejected before any terminal transition.
2. Otherwise `payload` creates a structured final summary Object and overrides
   `message`.
3. Otherwise `message` creates internal `{"message": ...}` Object data; it is
   not Human output.
4. Supplying none can return `result_oid=null` on a one-phase image. Omit all
   result inputs only when an intentionally empty terminal result is the known
   contract; never do so to discover whether an exit gate exists.

`review_token` and `completion_evidence` are only for cumulative review. A
committed exit returns `status="exited"` and `terminal_committed=true`; only that
status confirms terminal completion. `status="completion_review_required"` is
nonterminal. If post-commit cleanup fails, the same exited result includes a
safe structured `error.code="terminal_cleanup_required"`, the committed
`result_oid`, cleanup state, and Host recovery instructions.

Exit atomically records terminal state, result identity, evidence, and parent
wake, then performs durable cleanup; the commit is irreversible even when that
cleanup reports an error. The result survives live-runtime process cleanup,
while other child memory awaits its direct parent. Its payload is still not
guaranteed recoverable after reopen.

Exit is rejected before terminal commit while any ordinary descendant remains
nonterminal; settle it with the child-process Skill rather than leaving an
independently runnable orphan. An already-published active ObjectTask runner is
the deliberate exception because its lifecycle is Host-managed and its owner
pin must survive creator exit; an unpublished runner is not exempt. Internal
failed-exit handling terminates ordinary descendants bottom-up while preserving
those Host-managed runners.

## Recommended workflow

1. Check authority/budgets; visibility is not authority.
2. For backoff, read time, sleep once, then re-read time/state. Use owning waits
   for events.
3. Compact only with its marker; prefer `force=false`, low sufficient chunks,
   and a deliberate tail. Call alone and yield while Runtime resumes it.
4. Before exit re-read the cumulative goal/ledger and queued messages. A notice
   pauses exit: follow source-neutral Skill discovery, activate a result that
   declares a message-read tool, read/ACK input, and merge cumulative follow-ups.
   Send required `human_output` in a prior turn, then call `process_exit` alone
   and follow Completion evidence on review.

## Failure and recovery

- Invalid timezone: use a valid IANA name. Do not repeat authority denials.
- Sleep error is unknown unless certified not-started; reconcile time/state.
- Missing context/rights: stop; `force=true` creates neither.
- A source-version conflict, rejected child output, or failure explicitly known
  to occur before context writeback leaves the original payload unchanged, but
  a final CAS conflict may already have advanced durable context generation.
  Other errors can occur after the context CAS but before job/result evidence
  finishes. Re-read context OID/version/generation and the compaction job; never
  loop or replay until readback proves no writeback committed and the cause is
  resolved. A changed generation ends the quantum for fresh rematerialization.
- Pending compaction survives reopen: Runtime can reconstruct a missing child
  goal and resume; if a completed child's result payload vanished it may rerun
  that summary stage. If a crash happened after a pending row was claimed as
  `resuming`, Runtime fails the process closed rather than replay an operation
  with unknown settlement. Never forge `_resume_job` or spawn a replacement.
- Exit validation/review: the process remains nonterminal. Use the newest review
  and never claim exit without exact `status="exited"`. Any exited result with
  `error.code="terminal_cleanup_required"` has already committed its outcome:
  stop, preserve `result_oid`, do not call `process_exit` again, and leave the
  idempotent `retry_terminal_cleanup` action to the Host.
- Goal recovery after reopen: cumulative review first reads the live goal Object.
  On persistent reopen, the Host may already have rehydrated the exact initial
  goal of a committed live root spawn from its integrity-bound full-I/O launch
  envelope. That exception does not cover child/fork or exec replacement goals.
  If the live payload is still gone, review can recover the exact initial goal
  only from retained full-I/O LLM evidence and only up to 32,000 characters. Disabled
  full-I/O, absent evidence, or oversized recovered context fails closed before
  a token is issued. Restore a suitable checkpoint or establish a supported
  bounded successor goal; never invent omitted requirements. A Human follow-up
  does not silently replace an unavailable original goal.

Loaded projection and a pending wait can survive reopen; ordinary Object payloads
do not. The narrow committed-root initial-goal envelope above is a startup aid,
not general Object storage. Use checkpoints, files, or approved storage for
durable artifacts.

## Completion evidence

Some images, notably the coding image, gate exit; others are one-phase. Do not
probe that distinction with an empty terminal call. Use this state machine.

### Prepare, call, ACK/review

1. Finish the cumulative work first and prepare the intended final `result_oid`,
   `payload`, or `message`. Call `process_exit` alone with that result, without
   a review token or invented evidence. Never use an empty `process_exit` as a
   gate probe: on a one-phase image it irreversibly exits with no result.
2. If the result-bearing call returns `exited`, stop. If it returns
   `completion_review_required`, inspect its goal, source refs, unread/ACKed IDs,
   message count/hash/reference, observed tools, hints, required shape, and errors.
3. Follow the review's source-neutral `skill_discovery` query, activate one
   returned Skill that declares `read_process_messages`, then read unread Human
   messages using exact IDs or `{}` when directed to drain the unread mailbox.
   Default `ack=true` changes each returned message status. Status is part of
   review identity, so ACK makes the first token stale.
4. After ACK, repeat `process_exit` with the intended final result but still
   without token/evidence to obtain a post-ACK review. Use only this review's
   token, goal OID, acknowledged ID set, and expected refs. If more unread input
   appears, repeat read then the same result-bearing review call. Never submit
   the pre-ACK token.

Review never inlines Human message bodies. It keeps every ACKed ID, count, and
hash, plus an `acknowledged_human_message_reference`. Execute each
`batches[].arguments` exactly; every batch already fits the active ID-count and
filter-JSON limits and uses `include_acked=true`, `ack=false`. Inspect
`has_more`/`continuation`. There is no cursor: if a durable-result bound splits a
batch, subtract returned IDs from that batch's `arguments.message_ids` and issue
the next exact-ID read with only the remainder and its count. Repeating identical
filters replays the first page. No batches means no ACKed messages. Refresh the
review after any ACK. Message-count overflow fails closed and requires a
Host-approved consolidated successor.

A live goal likewise carries a hash and Object Memory reference, not an inline
preview. Copy the reference's namespace/name arguments and read by name—never
use `goal_oid` as a name. Only when the live payload is missing after reopen may
the review include the bounded full-I/O `fallback` described above.

### Evidence shape and terminal submission

Give every explicit deliverable its own check; one source citation does not prove
all its requirements. Observed calls are unique tool names, not call IDs; cite
exact names and explain concrete results.

`completion_evidence` requires exact `goal_oid`; exact full
`reviewed_message_ids` (or `[]`); nonempty `acceptance_checks`; and nonempty
`final_verification`. Every check needs nonblank `requirement`, nonempty known
`source_refs`, `status` (`completed|blocked|cancelled`), an
`evidence_tool_calls` array, and concrete nonblank `evidence_summary`. Across
checks cover every `expected_source_ref` and no unknown ref. Completed checks
must cite observed successful tool names. Blocked/cancelled may cite none but
must state the exact reason; any cited name must still be observed. Final
verification may contain only observed successful tool names.

There is no separate “clear review” call. After the post-ACK review, complete
missing tools, prepare evidence, and send required Human output in its own turn.
Successful tools/Human output do not change the token; a goal version change,
new Human message, or message ACK does. Then call `process_exit` alone with the
latest token, evidence, and the same desired result. Validation rebuilds the
current successful-tool list. If another review returns, resolve its newest
errors and input, refresh after any ACK, and retry. Stop only on
`status="exited"`.
