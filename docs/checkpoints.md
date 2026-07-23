# Checkpoints

Checkpoints are capability-controlled durable snapshots of reconstructable
runtime state for one process subtree. They are a durable execution building
block, not a mechanism for rewinding the outside world.

## Captured State

A checkpoint captures scoped state needed to reconstruct the owner subtree:

- process rows and statuses,
- process working directories,
- Object Memory metadata and payloads owned by the subtree,
- borrowed MemoryView root references and their types, without treating the
  referenced Object as subtree-owned state,
- process namespaces and object links,
- subtree capabilities other than `checkpoint:*` control authority,
- parent/child resource-budget reservations whose two processes are both in
  the captured subtree,
- process tool tables,
- JIT candidates and registered process-local JIT tools,
- compatibility Skill registry rows associated with loaded Skills (never
  applied to the current global Skill registry by restore/fork),
- loaded Skill records,
- mailbox delivery state,
- durable `llm_pending_actions`, including conditional provider-release
  prepared state according to the active `llm.persist_full_io` retention
  policy,
- image definitions needed by the subtree,
- embedded boot artifacts needed by those image definitions, for both
  `checkpoint_commit` and `image_package` boot kinds,
- loaded startup Runtime Module ids and source hashes.

Capabilities whose resource starts with `checkpoint:` control access to a
checkpoint artifact; they are not reconstructable process state. They are
excluded from the snapshot, and restore and fork also reject them defensively
when reading an older or externally supplied artifact. Consequently,
checkpoint creation, restore, and fork cannot duplicate or resurrect
checkpoint read, execute, or admin authority.

Transient `running` state is normalized to `runnable` at snapshot time. Forking
from a checkpoint also normalizes transient wait states such as waiting for an
event, tool, or human response back to `runnable`; the forked process must
re-enter those waits explicitly under its new identity.

Checkpoint creation reads the scoped SQL rows and in-memory Object payloads and
writes the checkpoint row, owner's `checkpoint_head`, owner's initial
`checkpoint:<id>` `read` capability, creation event, and creation audit in one
store transaction. Any event/audit/capability sink failure therefore rolls the
entire create result back instead of leaving a checkpoint whose id was never
returned. A concurrent store transaction appears wholly before or wholly after
the captured snapshot; it cannot split an Object row from its payload. The
denormalized process capability index is rebuilt from the capability rows read
by that same snapshot, so those two representations cannot disagree.

Ownership defines the destructive scope. Process- and process-result-owned
Objects, plus scoped ObjectTask results, are captured as reconstructable state.
A root in a `MemoryView` is only a reference: checkpointing a borrower does not
claim ownership of the lender's Object, payload, namespace, or capability.
The canonical version-4 artifact records owned and referenced Object sets
separately, and restore/fork derive their destructive scope from captured owner
fields rather than from borrowed roots.

## Append-Only Boundary

Restore never deletes:

- audit records,
- events,
- LLM call records,
- checkpoint records,
- human interaction history,
- Sink trust registry generations/history, data-flow decisions, and file label
  binding/tombstone history.

Restore itself appends new audit and event records.

Restore and fork require the current Python runtime to have already loaded the
same startup Runtime Module ids and source hashes captured in the checkpoint.
Checkpoint restore does not import Python modules, change module trust, restore
global Skill trust rows, replace the global Skill registry, or roll back the
host module environment. Each restored/forked process continues to use the
immutable `package_snapshot` in its `loaded_skills` record. Historical
checkpoint `skills` rows remain readable compatibility data, not a host
registry mutation.

Host filesystem, Git checkout/index/refs/worktrees/simulated-PR metadata and
remote calls, shell, JSON-RPC/MCP remote calls, and provider effects are not
captured or rolled back. Git external-effect records remain reportable through
checkpoint diff/restore evidence, but restore never replays or compensates
them. Image registry rows are internal runtime metadata: snapshots
capture the image definitions needed by the checkpointed subtree and any
referenced `checkpoint_commit` or `image_package` artifact payloads.
Destructive restore re-upserts those captured image and artifact rows so the
restored subtree can run against its snapshotted image definitions. For an
image package, the captured artifact is the normalized package payload,
including its declared files and workspace/JIT metadata; restore does not
revisit or copy from the package's original host source path, and it does not
restore provider-side state.

Sink trust is Host-global external policy, not reconstructable subtree state.
Checkpoint restore neither replaces it nor lowers its active generation. Any
restored pending LLM action, provider chain, or conditional release is checked
against the current registry, current Task Authority manifest, and exact source
versions before use; an old trust record or release cannot authorize a new
dispatch. File path bindings also remain current because restore cannot rewind
the external file.

Providers classify their successful effects as:

- `irreversible`,
- `rollbackable`,
- `no_rollback_required`,
- `unknown` for an outcome whose external state cannot be proven.

The default `LocalFilesystemProvider` classifies successful writes, directory
creation, and file or directory deletion as `irreversible` with
`rollback_status=not_supported`. It stores no preimage or undo log and exposes
no compensation operation. The generic `rollbackable` class remains available
for providers that can support a real compensation layer, but checkpoint
restore itself still only reports those effects.

Filesystem mutations, clock operations, shell execution, and PTY spawn reserve
finite-use authority before entering their provider boundary.
`ProviderEffectNotStarted` restores that exact reservation only when it
certifies the current provider phase did not start and every completed earlier
phase has no mutation or information flow and explicitly does not commit
authority. Clock sleep/asleep begins its intent before
the first `monotonic()` measurement; ordinary first-measurement failure and all
failures after that observation consume the use, and elapsed-time measurement
marks the effect as information flow. Timeout, cancellation, resource-limit,
ordinary provider exception, and post-effect classifier failure otherwise
cannot prove non-execution, so the use stays consumed and an `unknown` effect
remains visible in checkpoint reports. A tool/provider error therefore does not
imply that no outside-world effect occurred. LLM, filesystem/clock/shell, human
output, PTY spawn/write/resize/close, and live JSON-RPC/MCP calls persist this
uncertainty before the provider boundary as a pending external-effect intent,
then conditionally finalize the same id. If any post-provider sink fails,
checkpoint diff/restore still reports the durable uncertainty, either as a
dispatched pending row or as a finalized `unknown` row. Cleaning up a failed
PTY session publication does not remove its uncertain spawn history.

Restore reports provider-recorded effects in
`external_effects_since_checkpoint`, summarizes them in
`external_effect_summary`, and returns `restore_external_policy:
"report_only"`. The summary includes `by_state` counts and an explicit
`pending` count in addition to rollback class and provider/operation totals. In
this report-only restore contract, `rollbackable` means the provider says a
future compensation layer could reason about the effect; restore still does
not apply external compensation. This contract is independent of the snapshot,
restore-plan, and image-artifact version numbers described below.

## Capability Model

Checkpoint authority uses these resource forms:

- `checkpoint:process:<pid>`
- `checkpoint:<checkpoint_id>`
- `checkpoint:*`

Rights map to operations:

- `write`: create a checkpoint.
- `read`: list, inspect, diff, or replay diagnostics.
- `execute`: fork from a checkpoint.
- `admin`: destructive restore.

Creating a checkpoint grants the checkpoint owner process `read` on that exact
new checkpoint. It does not automatically grant `execute` or destructive
`admin` restore authority.

## Public Operations

LLM-facing tools:

- `create_checkpoint`
- `list_checkpoints`
- `inspect_checkpoint`
- `diff_checkpoint`
- `restore_checkpoint`
- `fork_checkpoint`
- `commit_checkpoint_to_image` for checkpoint-derived AgentImage commits

The built-in base, coding, and review images expose the low-risk
create/list/inspect/diff tools; the toolmaker and context-compressor defaults do
not. Restore and fork tools are registered globally but require explicit tool
visibility plus checkpoint authority.

Checkpoint inspect process rows expose the snapshot's canonical tagged
`wait_state` and `outcome` mappings together with `state_generation`. The
manager strictly decodes the persisted snapshot JSON before projecting these
fields; CLI, GUI, Tool, and JIT callers therefore observe the frozen snapshot
state rather than a reconstructed compatibility message or current live state.

JIT syscalls:

- `checkpoint.create`
- `checkpoint.list`
- `checkpoint.inspect`
- `checkpoint.diff`
- `checkpoint.restore`
- `checkpoint.fork`
- `checkpoint.replay`
- `image.commit_checkpoint`

`checkpoint.replay_to_event` remains a compatibility alias for the canonical
`checkpoint.replay` JIT syscall. The Python manager method is named
`replay_to_event`, and the CLI subcommand is `checkpoint replay`.

CLI commands:

```bash
uv run agent-libos --db .agent_libos.sqlite checkpoint create <pid> "before risky edit"
uv run agent-libos --db .agent_libos.sqlite checkpoint list --pid <pid>
uv run agent-libos --db .agent_libos.sqlite checkpoint inspect <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint diff <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint restore <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint fork <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint replay <checkpoint_id> <event_id>
```

A relative `--db` path such as `.agent_libos.sqlite` is resolved from the CLI
process's current working directory. Use an absolute path when separate CLI
invocations must select the same database independently of their working
directory.

Passing `--actor-pid` makes the CLI enforce that process's checkpoint
capabilities. Omitting it runs as the trusted `cli` administrator and bypasses
process capability checks; it does not make every command automatically
audited. Checkpoint creation and restore publish their normal operation audit,
fork attempts a post-commit operation audit, and replay publishes a diagnostic
`checkpoint.replay_to_event` audit. `list`, `inspect`, and `diff` do not
themselves publish operation-specific audit records in either mode. In actor
mode, consuming a finite read still publishes the generic capability
reservation/settlement audit records. Deployments that require an audit record
for every administrator read must add that control at the CLI invocation or
host access boundary.

Finite read authority follows the operation shape. Actor-mode `list` consumes
the selected `checkpoint:*` read immediately, or the selected
`checkpoint:process:<pid>` read when `--pid` is supplied. `inspect`, `diff`, and
`replay` instead reserve an exact `checkpoint:<id>` read, falling back to the
checkpoint owner's process-read authority, for the duration of the diagnostic;
they restore that reservation if the diagnostic fails and commit it once on
success. Admin CLI mode skips these process capability checks.

## Restore

Restore applies at a runtime safe point and is scoped to the checkpoint owner
subtree. If the scheduler is actively running a quantum or still has active
futures, restore is rejected and the caller must retry after the runtime is
quiescent. Unrelated processes are not restored.

After scheduler quiescence, restore acquires the shared registry lifecycle lock,
then the Object ownership boundary, then the store mutation lock. It holds the
ownership/store boundary continuously from the first preflight read through the
main-state commit, and keeps the registry lock through image/JIT reconciliation.
That fixed order matches Runtime Module load/unload and prevents registry/store
lock inversion. Host-side process, capability, Object Memory, mailbox, and
ObjectTask mutations therefore linearize wholly before or after restore; they
cannot slip between validation and row replacement. Capability rows are
filtered inside that boundary against current active/expiry/restrictive policy
state. A revoke that commits before restore's linearization point wins and is
not resurrected from the snapshot.

Restore has three explicit phases:

1. **Preflight** validates checkpoint authority, required Runtime Modules,
   image artifacts, JIT metadata, and the absence of active scoped ObjectTasks.
   Checkpoint `admin` and all changed-image rights are collected as one
   deduplicated decision set. Restoring over an image id that is already registered
   requires `admin` on that exact `image:<image_id>` resource; reintroducing a
   missing image requires `write`.
2. **Commit** replaces scoped SQL rows and in-memory Object payloads in one
   transaction. Only owned Objects/namespaces are deleted or replaced; borrowed
   roots remain references to current lender state. The complete decision set is
   reauthorized and any finite-use authority is reserved inside the same outer
   AuthorityTransaction. Main state, the core restore event/audit, authority
   settlement, an exact `checkpoint_restore` runtime-publication plan, and its
   one-to-one `checkpoint.restore` operation binding commit together. The plan
   records the immutable checkpoint digest, scoped process ids, stale JIT ids,
   ordered phases, and bounded release-finalizer intents. The publication moves
   from `planning/planned` to
   `reconciliation_pending/main_state_committed` in that same transaction. A
   prepare, evidence, link, publication, or settlement failure rolls back all
   of them without a separate compensation transaction. If the authority/UoW
   exit reports an exception after that transaction actually committed, restore
   confirms the durable publication before handling the exception. A confirmed
   main commit is fenced and terminalized as recoverable failure; an uncertain
   confirmation remains nonterminal and fail-closed for startup recovery.
3. **Durable post-commit reconciliation** uses the version-2 ordered program
   `object_payload_reconciliation`, `image_reconciliation`,
   `jit_source_reconciliation`, `jit_pruning`, and
   `object_release_finalizers`. The first receipt records that the payload-aware
   main transaction published the restored Object markers and process-local
   payload cache. The remaining registry phases atomically reconcile captured
   image cache, image rows, artifact rows, and process-local JIT state while
   still holding the lifecycle lock. It never applies captured global Skill
   rows. The lifecycle lock is released before running release finalizers for
   scoped Objects removed by the commit, because finalizers may call host code.
   These operations cannot undo the committed process-state transaction. Every
   completed phase, and every completed finalizer work item, receives an ordered
   durable receipt before the next phase. Existing image artifacts must exactly
   match the captured payload, kind, digest, and metadata.
   Registry/JIT/prune/finalizer failures cannot undo the committed transaction:
   the publication remains recoverable, the linked operation is `unknown`, and
   mutation admission is fenced until the Runtime is reopened.

Restore never rewinds a process concurrency token. For every restored PID,
`revision`, `execution_generation`, and `state_generation` are each advanced to
one greater than their durable high-water floor, where that floor includes both
the current live row and the checkpoint row. `execution_owner_id` and
`execution_lease_id` are cleared. Stale process CAS writers and pre-restore
execution tokens therefore cannot publish after restore, and a second restore
advances the three values again.

Finite capability consumption is also monotonic. A captured capability is
revalidated against its current active/expiry/restrictive-policy state, and an
existing capability keeps its current `uses_remaining` rather than the larger
value in the checkpoint. Outstanding capability-use reservations in the
restored process/Object scope are invalidated before capability rows are
replaced, so a late provider completion cannot refund a consumed use. This is
distinct from captured parent/child `process_resource_reservations`, which are
reconstructable subtree state and are restored from the checkpoint.

A successful commit returns `main_state_committed: true`. If every
post-commit phase succeeds, `status` is `restored`; otherwise `status` is
`restored_with_warnings` and `post_commit_failures` identifies each failed
reconciliation/finalizer phase and error. Each recordable failure appends a
`checkpoint.restore.post_commit_failure` audit record. Callers must not retry a
`restored_with_warnings` result as though the main restore had rolled back.
Every result includes its `publication_id`. A `restored_with_warnings` result
also has `reconciliation_pending: true` and leaves the current Runtime in
`close_failed`; an all-success `restored` result has no pending reconciliation
and leaves ordinary admission open. In the warning case,
`close()`/`shutdown()` stays fail closed and deliberately retains the diagnostic
store and backend lease. After diagnostics have been captured, the owner must
call `Runtime.release_recovery_diagnostics()` (or await
`Runtime.arelease_recovery_diagnostics()`) before a fresh Runtime can reopen the
target and perform authoritative startup recovery.
Post-commit control-flow interruptions are re-raised without type conversion
after the recovery fence is installed. If reporting that fence also fails, the
primary interruption and secondary fence error are preserved together in a
`BaseExceptionGroup`. A secondary diagnostic failure is grouped with the
primary interruption and the exact pending-publication signal, so the operation
wrapper cannot independently finalize a still-recoverable restore. Conversely,
an exception reported after the final publication/operation transaction commits
is re-raised without installing a recovery fence only after strict read-only
confirmation of the complete receipt transcript, plan anchor, exact binding,
and terminal successful operation.

Startup claims incomplete checkpoint-restore publications only while the
lifecycle is `recovering` and the calling context holds its opaque internal
recovery lease. The recovery entry point fails before reading or claiming a
publication when called on an open Runtime; online restore and startup recovery
therefore cannot compete for the same phase lease. With the backend's exclusive
Runtime lease, startup recovery also takes a durable per-attempt publication
lease. It resumes only missing receipts, then commits the publication and
changes the exact linked operation from `unknown` to `succeeded` in one
transaction. Before the general missing-payload sweep, recovery reloads the
hash-bound checkpoint and rehydrates volatile payloads only for an exact live
Object row produced by that restore. A row with the same immutable creation
identity and a strictly higher version, including a versioned ownership
transfer, or an explicitly released row is treated as superseding the old
delivery and is not overwritten. A missing row, lower version, same-version
drift, creation-identity drift, or malformed payload marker fails startup
closed. A skipped newer row remains cache-missing after reopen, so the ordinary
recovery sweep releases it, removes its links, and revokes its Object
capabilities.

The terminal receipt then carries a `payload_delivery` handshake. Recovery
publishes `pending` after selective hydration. Once startup hooks and workers
are ready, startup creates one durable `preparing` delivery attempt and moves
the backlog through hard-bounded, indexed keyset pages from `pending` to
`confirmed` and then `completed`. Each page retains the exact attempt identity;
ordinary page transactions do not hold the lifecycle condition. Delivery state
and operation reconciliation are independent projections: delivery transitions
preserve `operation_reconciled`, and the separate terminal-repair pass repairs
any false marker in `pending`, `confirmed`, or `completed` before OPEN.

After every page completes, one outer Store transaction changes the exact
attempt from `preparing` to `acked`. The lifecycle commit guard publishes OPEN
while holding admission closed through that same commit, so the durable ACK and
the in-memory OPEN transition have one linearization point. If the backend
reports a commit error, startup performs an exact typed readback of that attempt
identity. `acked` means the commit crossed the linearization point: the current
cache-holding Runtime repairs its in-memory lifecycle to OPEN and succeeds,
without compensating or replaying payloads. `preparing` proves that compensation
is still authorized, so its `confirmed` or `completed` rows return to `pending`
and the attempt becomes `aborted`. A missing, `aborted`, malformed, mismatched,
or unreadable control row fails closed without changing publication rows. A
process crash after an `acked` attempt is likewise treated as a previously
completed OPEN; a later startup does not replay that consumed delivery.

When there is no payload backlog, startup has no durable ACK to publish and uses
the rollback-safe `in_memory_open_scope` under the exact startup lease. A late
failure before ACK compensates the active attempt so the next startup can replay
the checkpoint payloads. If compensation cannot be confirmed, failed-assembly
cleanup retains exact Store ownership and exposes an explicit cleanup-required
handle instead of allowing an ambiguous reopen.

This runs after process launch/exec publication recovery but before general JIT
rehydration, so a stale ephemeral tool cannot be loaded between reopen and
pruning. Committed restore receipts whose operation-reconciliation marker is
false are revalidated through exact, indexed keyset pages before generic
stale-operation interruption. Their strict plan, immutable receipt-side plan
digest, complete ordered phase/finalizer transcript, checkpoint snapshot
identity, publication PID, and exact operation binding must all still agree.
The reader accepts the exact legacy version-1 phase program as well as version
2; it verifies a version-1 anchor with the original version-1 bytes and never
rewrites that immutable plan or anchor during recovery. Any mixed version/order
shape is rejected before a recovery claim or callback. Generic Host publication
APIs reject checkpoint-restore insert, transition, recovery-claim, artifact,
plan, and operation-marker writes; only the storage-owned writer injected into
the reconciler can advance this state machine. Even out-of-band plan or receipt
corruption on a marker-false row is caught before replay or terminal operation
repair. Direct database writes that also forge or preserve a completed marker
remain outside the application-integrity boundary.
Failed/manual receipts remain in forward recovery. A second reopen observes a
terminal publication and performs no work.
Corrupt plans, invalid receipts, changed checkpoint artifacts, lost recovery
leases, and exhausted attempts fail startup closed rather than guessing.
Authorized recovery calls on one Runtime are single-flight, preventing two
internal startup workers from reusing the same recovery lease and running
finalizer work twice.
Cross-Runtime concurrency remains outside the documented one-writer boundary.

Flow metadata is fail closed, not repaired during restore. Every captured
`llm_pending_actions` row, including completed history, must contain a canonical
`data_flow_context_json`; every captured process message must contain canonical
`data_labels` and `source_oids` metadata. Restore and fork validate all such
rows before publishing any process-state mutation. Missing, incomplete, or
malformed metadata rejects the artifact instead of assigning conservative
labels. Fork additionally requires any message `label_carrier_oid` to belong to
the cloned Object scope and remaps that carrier to the forked Object id.

All post-checkpoint pending human requests for restored processes are cancelled.
Among messages created after the checkpoint, only entries that are still
**unread** are marked `superseded_by_restore`; acknowledged or
already-superseded post-checkpoint history is left unchanged. Captured message
rows are restored to their checkpoint delivery state, so a message that was
unread at capture may become unread again.

ObjectTask rows are durable execution history, but they are not append-only and
are not themselves checkpoint rows. Restore refuses any active task in the
affected process or Object scope. If a scoped task reached a terminal state
after the checkpoint, restore updates the existing row to
`superseded_by_restore`, clears the now-invalid runner/result references, and
stores the previous status, runner, result, and error in the task's wait
metadata. The restore response, event, and audit list those task ids as
`superseded_object_tasks`. A task that was already terminal at checkpoint time
keeps its ordinary status; if reopen had degraded its unavailable result,
restore can update that row and reconnect the checkpointed runner/result Object.
This boundary uses the task's terminal `completed_at`, not `updated_at`: late
notification delivery may update the row after checkpoint creation without
changing when the task actually became terminal. Legacy rows that have no
`completed_at` fall back conservatively to `updated_at`.

After restored process rows are written, restore reconciles wait states against
the restored runtime facts. A process waiting on an already-terminal child or a
now-available matching unread message is made runnable; an unresolved typed
child/message wait remains waiting, while an invalid event-wait shape is
paused. A `WAITING_TOOL` process is paused with an explanatory status because
restore rejects scoped active ObjectTasks and will not revive a missing,
terminal, or mismatched task binding. A valid human wait remains waiting while
any referenced request is pending. Once all requests are resolved, approved
requests resume; a resolved `permission_request` also resumes regardless of its
recorded decision so the permission path can re-check current capability state;
other rejection/cancellation outcomes pause the process. Missing or malformed
human wait identities also pause it. These transitions and their
`state_generation` bumps commit atomically with the restored rows.

Restored `llm_pending_actions` remain durable wait generations. Before the next
quantum, the executor synchronizes its in-memory human/child/message waiter with
the restored row and `resume_token`, so a pre-restore cached generation cannot
claim or clear it. Restore also assigns every restored process a fresh durable
LLM context generation inside the main transaction. Provider-side Responses
state is outside checkpoint rollback and is not rewound by restore, so this
generation change forces the next LLM request to reset stateless instead of
chaining to a response made from post-checkpoint local state.

When `llm.persist_full_io=false`, a conditional LLM release row contains only
hash-bound resume metadata. It can resume only while the matching prepared
request is still held by the current executor; a reopen, or a restore that
discards that in-memory generation, fails closed before provider dispatch.

Scoped Object Memory rows removed by restore use restart-safe release
finalizers. A trusted module declares stable handler ids in
`provides.durable_object_release_finalizers` and buffers each
`bind_durable_object_release_finalizer(id, prepare, finalize)` registration at
module entrypoint time, before startup publication recovery. Before deleting an
Object, the side-effect-free `prepare` callback freezes a bounded JSON intent,
Object id/version, digest, and stable idempotency key into the publication.
`finalize` may run at least once and always receives that same key. Per-item
receipts prevent already-acknowledged work from running again.

An anonymous `bind_object_release_finalizer` cannot safely survive a crash, so
restore rejects deletion while one is registered. If a persisted stable handler
cannot be reconstructed on reopen, the publication moves to `manual` while
retaining the exact work item and the linked operation remains `unknown`; it is
never silently marked committed.

If irreversible provider effects exist after the checkpoint, restore still
continues by default. Their durable external-effect evidence stays in runtime
history and in the restore report; restore neither deletes it nor compensates
the provider.

JSON-RPC endpoint and MCP server registry rows are host provider configuration
and are not captured or restored. Restored capabilities that reference a
missing endpoint or server fail closed until a host operator registers that
provider configuration explicitly.

## Commit To Image

A checkpoint can be committed into a new `AgentImage`, similar in spirit to a
Docker image commit but scoped to Agent libOS reconstructable runtime state.
The checkpoint-to-image artifact captures only the checkpoint owner root
process:

- Object Memory metadata and payloads both referenced by and owned/captured for
  that process (borrowed roots without captured payloads are excluded),
- process-local namespace state,
- loaded Skill records and package rows,
- visible static tools and process-local JIT tool sources,
- process cwd and image context settings,
- lazy model-tool projection behavior, including validated initial tool groups,
- required startup module summaries.

It does not copy the real filesystem, shell state, remote JSON-RPC/MCP state,
human UI output, network effects, resource budgets/usage, or any other
provider-side state. It also does not restore JSON-RPC endpoint registrations,
MCP server registrations, or global Skill trust rows; those are host registry
decisions, not image state. Provider effects remain in durable external-effect
history and are not packaged or rolled back by image commit.
The underlying intent row may still follow its guarded finalize or payload-
retention state machine. Resource limits for a process booted from the committed
image must come from the caller that starts that process. Use an image package
`workspace/` seed when an image needs filesystem content at boot time.

External capabilities in the checkpoint are converted into image
`required_capabilities` declarations. They are not restored as live authority
when the committed image is spawned or execed. Internal Object Memory
capabilities needed to read the baked objects are remapped into the new process.
Loaded startup module summaries are copied into image `required_modules`; the
committed image fails closed at boot unless those exact module ids and source
hashes are loaded in the current runtime.

Image registry authority is separate from the image's declared external
requirements. When child spawn selects an image different from the parent's
current image, or exec switches the process to a different image, the acting
process must hold `read` on the exact `image:<image_id>` resource in addition to
the relevant process lifecycle authority. Reusing the already active image does
not repeat that read check. A declaration such as `image:*` in
`required_capabilities` does not replace a required boot-time check.

CLI example:

```bash
uv run agent-libos --db .agent_libos.sqlite images commit <checkpoint_id> stateful-agent:v0 --name stateful-agent
uv run agent-libos --db .agent_libos.sqlite spawn --image stateful-agent:v0 --goal "use baked memory"
```

The model-visible commit path is the `commit_checkpoint_to_image` tool; the JIT
syscall path is `image.commit_checkpoint`. Actor-mode commits require `write`
on the target `image:<image_id>` and read authority on either the checkpoint or
the checkpointed process. Replacing an existing target image requires `admin`
instead of `write`. Both finite decisions settle with artifact/manifest/event/
audit publication, so a failed commit does not burn either one-shot grant.

## Fork From Checkpoint

Fork creates a new isolated process subtree from checkpoint state. It remaps
pids, owned object ids, capability ids, namespace ids, ephemeral JIT tool ids,
and JIT candidate ids. Candidate Object payloads and loaded-Skill JIT mappings
are rewritten to those new identities, so unloading or replacing a forked tool
cannot retire the source process's registration. Each forked process starts a
new concurrency identity with `revision`, `execution_generation`, and
`state_generation` set to zero and with no copied execution owner or lease.

Executable JIT handles/sources are prepared before publication. The missing
Image/artifact rows, all fork rows, Object payload cache entries, and any parent
child-budget reservation/charge then publish in one store transaction. On
failure the prepared handles and newly introduced images are discarded; the
scheduler cannot claim a fork root whose process-local JIT assets are only
partially installed.

Fork is non-destructive for the global image registry: it restores only missing
image definitions, and restores embedded `checkpoint_commit` or `image_package`
artifacts only for the missing images it reintroduces. It never replaces an
image id that already exists. When actor capability checks are enabled, fork
therefore requires both `execute` on the checkpoint and `write` on each missing
`image:<image_id>` resource it would reintroduce.

Fork also never replaces the global Skill registry. Forked processes consume
their captured `loaded_skills.package_snapshot`; checkpoint compatibility rows
are not upserted into host Skill state.

Fork intentionally drops all captured `llm_pending_actions`. A pending action
contains source-process provider/tool/request identities and resume tokens that
must not be replayed under the forked PID. Forked transient wait states are
normalized to `runnable`, and the new process must issue fresh requests under
its own identity.

The forked subtree must not gain authority wider than the checkpointed subtree
held. It does not share the original process private namespace or result
objects by reference. Checkpointed ObjectTask result objects are remapped to the
forked process-result owner boundary rather than pointing back at the original
task id. A source capability with `uses_remaining` set is never copied into the
fork, even when it still has uses left; cloning a finite grant would multiply
one-shot authority.

`EXTERNAL_REF` Objects are host-handle descriptions, not clonable runtime
resources. Fork drops both owned and borrowed `EXTERNAL_REF` roots and filters
their object capabilities; it never reconnects or aliases the source PTY or
other host handle. Destructive restore likewise mutates only subtree-owned
Objects and does not roll back a borrowed Object or its lender's capability.

Fork first performs non-consuming preparation checks: the requested parent must
exist and be non-terminal, cross-subject parent attachment needs `admin` on the
parent's checkpoint-process resource, the actor needs `execute` on the source
checkpoint, and every missing snapshot image needs `write` on its exact image
resource. All checks run again inside the same store transaction that publishes
the fork, and finite-use actor authority is consumed only there. A concurrent
revoke, newly terminal parent, missing image permission, or later transaction
failure publishes no fork/image/object/process rows and rolls back that
transaction's one-shot consumption. Only a successful publication spends the
grant.

Snapshot capability copying is also revalidated at the publication point, so a
revoke committed before it wins. Revoked, expired, and finite-use capabilities
are not copied. If a current
restrictive `deny` or `ask` capability may overlap a checkpointed `allow`
capability, the overlapping right is dropped from the forked copy. This is
conservative by design: capability records do not encode exceptions, so a fork
must not recreate broad authority that current policy has narrowed since the
checkpoint was taken.

After main-state publication, fork event/audit emission is observability rather
than rollbackable state. Success returns `status: forked` and
`main_state_committed: true`. If either post-commit sink fails, the fork remains
published and the result instead uses `status: forked_with_warnings` with
`post_commit_failures`; retrying it as an uncommitted fork would duplicate the
subtree.

## Replay To Event

`replay_to_event` is diagnostic timeline replay. It reports event history from
a checkpoint to a target event. It does not rerun LLM calls, tools, syscalls, or
external side effects.

## Limits

Checkpoint defaults live in `CheckpointDefaults`:

- checkpoint list limit,
- payload capture limit,
- snapshot hard byte limit,
- diff preview size.

Checkpoint-to-image construction has an independent `ImageCommitDefaults`
budget and does not inherit or reuse those checkpoint capture limits. Its
separate controls are:

- `artifact_hard_limit_bytes`, for the whole committed artifact;
- `payload_capture_limit_bytes`, for each captured Object payload;
- `max_required_capabilities`;
- `max_committed_tools`;
- `max_committed_jit_sources`;
- `metadata_preview_chars`.

Exceeding an image-commit limit fails the commit even when creation of the
source checkpoint succeeded. Conversely, increasing a checkpoint snapshot
limit does not increase any image-commit limit.

Snapshot version is not a `CheckpointDefaults` setting. The checkpoint snapshot
codec is the fixed `CHECKPOINT_SNAPSHOT_VERSION`, currently version 4. Version
4 requires canonical typed process wait/outcome fields and their state
generation; earlier snapshot shapes are rejected rather than interpreted using
`status_message` strings.

Independent version namespaces appear in this document:

- the durable checkpoint snapshot schema is version 4;
- the checkpoint-restore publication/reconciliation plan is currently version
  2, while startup recovery accepts an exact immutable legacy version-1 plan;
- checkpoint-to-image commits use the separately configured
  `image_commit.artifact_version`, whose default is 2;
- image-package artifacts use their own fixed `artifact_version` 1.

Changing one of these versions does not change or authorize either of the
others.

Git does not add another checkpoint schema. RuntimeStore remains schema v3 and
checkpoint snapshots remain v4. Image packages continue to reject `.git`
metadata and do not embed managed worktrees or remote state.

Checkpoint list callers may request only positive integer limits no larger
than `CheckpointDefaults.list_limit`; `0`, negative values, booleans, and larger
values are rejected rather than being passed through as unbounded SQL limits.

There is no automatic high-risk checkpoint setting. The former
`auto_high_risk_checkpoint` config field described an unimplemented idea and is
now rejected by strict config loading. Callers that need a safety snapshot must
create it explicitly before the risky operation.

Payload capture is bounded. A checkpoint can fail when reconstructable state is
larger than configured limits.
