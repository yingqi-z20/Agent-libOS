# Runtime Model

Agent libOS models work as `AgentProcess` instances. A process has identity,
status, a goal object, a memory view, a process-local working directory, a
complete callable tool table, a separate model tool projection, loaded Skills,
capabilities, children, a Task Authority Manifest, message queue state, an
`llm_profile_id`, and resource budgets.

A process can change visible tools, activate Skills, register process-local JIT
tools, register or exec AgentImages, fork children, and fork from checkpoints,
while resource authority remains separate in Capability.

Provider-backed primitive work is represented by a
[`ProtectedOperationContract`](protected_operation_sdk.md). One logical
operation may contain several ordered provider phases, but uses one durable
effect id, one deduplicated reservation set, and one explicit operation/evidence
chain. Waiting/resume behavior remains part of the enclosing logical operation.

## Durable Task Run supervision

A [`TaskRun`](durable_task_runs.md) is a Host-supervised durable envelope around
one root `AgentProcess` and its descendant process tree. It adds a versioned
goal, requirements, a current status projection, an append-only ledger, safe
resume points, idempotent commands, retention policy, and one monotonic Runtime
epoch. It is not another scheduler and is not a declarative workflow graph:
ordinary process transitions, ToolBroker calls, Human requests, messages,
Capabilities, Task Authority, data-flow decisions, resource accounting, and
protected external effects remain authoritative.

Process spawn and fork under a Run inherit its id, epoch, and process role in
the same transaction as child publication. A stale epoch cannot claim or
publish Run-owned execution after a successor Runtime takes the active-store
lease. `run_until_blocked` advances bounded scheduler quanta and then projects
the process tree into a Run status such as `waiting_human`, `waiting_process`,
`waiting_message`, `waiting_tool`, `paused`, or `needs_attention`.

A safe resume point is written only after a complete model action and all
paired tool results have committed. It binds the local context generation,
Image/tool and provider identities, authority, transcript/payload hashes, and
Runtime epoch. Startup does not re-run work merely because its Run was
`running` when the Host stopped. It first validates Run payloads, reconciles
prepared/provider effects, reservations, and publications, fences stale
operation/execution claims, and then reconciles ObjectTask and
terminal-process-cleanup facts. It automatically continues only from a valid
local resume point with no later unsettled effect,
an authoritatively certified non-dispatch, or an already durable provider
success awaiting local settlement.

Task Run command receipts fence Host retries separately from resume points.
Split-phase `run`/`resume`/cancel/deadline, interrupting follow-up, and
authoritative effect-receipt commands persist a request-bound provisional
result before their response can be mistaken for permission to start again;
local convergence CASes that receipt to its final revision-bound result. Every
split local-control pending and completed receipt retains its exact
`admission_ledger_seq`, `admission_ledger_item_id`, and
`admission_evidence_sha256` reference to the same-transaction append-only status
transition. Replay validates that ledger row and its Run, command, request,
from/to status, and semantic generation/revision evidence even on terminal or
superseded early-return paths. Interrupt completion retains its
`interrupt_provenance_sha256` while removing the raw admission epoch/fence list.
After
a crash or lost response, an exact pending-command replay performs only local
settlement and projection: it does not consume another scheduler quantum or
redispatch an LLM, tool, provider operation, or external effect. Once verified
effect settlement has committed, that replay instead validates the exact
append-only effect transition plus `host_verified_receipt` audit and does not
invoke the receipt verifier again; this evidence survives provider-body purge.
All command inserts and result updates, including a linked-recovery gap repair,
hold the global Runtime-epoch counter-row fence, so a superseded Runtime writes
nothing. Startup locally completes only a well-formed pending interrupt
for a recoverable nonterminal Run and the current pause/cancel generation. It
first settles any already-staged complete provider result and validates durable
waits, then applies ordinary binding, authority, effect, and ObjectTask gates.
Queued execution stays queued; only an integrity- and generation-matched
typed `StaleExecutionProcessWait` receipt is resumed, never an independent Host
pause. Its `stale_execution_recovery` `status_message` is compatibility display
text and is not a control input; ordinary resume blockers and finalization still
run afterwards. Cancellation
or an expired deadline outranks an older interrupt. If the Run becomes terminal
before command-result CAS, startup leaves that receipt for exact local client
replay. Ambiguous, corrupt, or over-bound recovery evidence fails closed into
`needs_attention`. The detailed receipt state machine is specified in
[Durable Task Runs](durable_task_runs.md#crash-safe-command-settlement).
Linked-Run recovery has a separate local gap protocol: if its deterministic
nested rerun and target creation committed before the outer recover receipt,
only the exact outer request may validate the parent request-hash binding,
source/target receipts, target row, and unique Run link and then copy that
immutable result into the missing outer receipt. Startup does not create or
repeat the linked Run, and incomplete evidence writes nothing.

Missing payloads, binding drift, non-replayable pending actions, abandoned
ObjectTasks, and dispatched/unknown effects produce `needs_attention` and
prevent downstream dispatch. Pause, cancel, and deadline handling are
forward-only persisted intents. An unresolved effect is never converted into
a false retry or `cancelled` state. A Run succeeds only after every descendant,
required requirement, effect, reservation, and retention cleanup has settled.
Default Run cleanup also removes readable content from linked terminal Human
request payloads/decisions while retaining their identities, types, statuses,
timestamps, hashes, and audit linkage.
The exact model and Host interfaces are documented in
[Durable Task Runs](durable_task_runs.md).

## Process Lifecycle

The current lifecycle includes:

- `created`: process row exists but has not started running.
- `runnable`: eligible for an execution claim. The scheduler may claim it, and
  a Host direct-workflow entrypoint may claim a newly published process.
- `running`: a Runtime-owned execution path is advancing the process. This can
  be a scheduler quantum, a claimed direct workflow, process-exec admission and
  publication, or a Host-managed ObjectTask runner. Scheduler, direct-workflow,
  and process-exec admission use an exact execution owner/lease tuple; an
  ObjectTask runner is managed by the ObjectTask service without a scheduler
  lease and is excluded from LLM scheduling.
- `waiting_human`: the process is blocked on a human question or approval.
- `waiting_event`: the process has exactly a `ChildProcessWait` or
  `MessageProcessWait`; it is not an untyped arbitrary-event subscription.
- `waiting_tool`: an ObjectTask runner has a `ToolProcessWait` whose
  `operation_id` is the owning durable ObjectTask id.
- `paused`: the process is not selected for normal execution.
- `suspended`: compatibility status accepted by resume/state readers; current
  production transitions use `paused` and do not create new suspended rows.
- `exited`: completed successfully.
- `failed`: completed with failure.
- `killed`: terminated by signal or runtime decision.

Terminal statuses are `exited`, `failed`, and `killed`.

The durable process state is a typed product, not a free-form status message.
`waiting_event`, `waiting_human`, `waiting_tool`, and `paused` carry a matching
tagged `wait_state`; terminal statuses carry a matching tagged `outcome`; other
statuses carry neither. `ProcessTransitionService` is the semantic write
boundary for those three fields. It increments `state_generation` on every
transition. Child/message and syscall-cleanup wake paths capture both that
generation and the exact wait payload in a `ProcessStateToken`; Human decisions
use the exact observed revision/status/generation and request-id wait in their
transaction. A stale owner therefore cannot wake or rewrite a later,
structurally identical wait after a wait/runnable/wait ABA cycle.
Registering a new child, message, Human, or Tool condition wait is also
owner-preserving: it may start from an unblocked state; an active child, message,
or Tool wait can only be renewed with an exactly equal value, while Human
request-id sets may only add or remove ids monotonically. Exact
revision/generation CAS fences each update. No condition owner can replace
another domain's active wait or a paused/Host-resume gate.

`revision` is the optimistic compare-and-swap fence for persisted process-row
updates. Scheduler execution has a separate
`(execution_generation, execution_owner_id, execution_lease_id)` identity. A
claim rotates the generation and installs an owner identity plus fresh opaque
lease id; worker writes must present that exact `ProcessExecutionToken`. Completion,
terminalization, trusted takeover, checkpoint restore, and exec rotate or clear
the lease so a detached worker from an older quantum cannot publish late state.
These counters have different jobs: `revision` rejects stale row updates,
`state_generation` rejects stale semantic wakeups, and the execution tuple
rejects stale quantum ownership.

### Principal lifecycle transitions

The following matrix describes the principal production orchestration contract,
not every internal publication or recovery staging row. It is not permission to
call a repository transition directly: each actor still has to satisfy its
listed authority, ownership, publication, and compare-and-swap preconditions.
`ProcessTransitionService` validates the status/wait/outcome product and
state-generation fence; specialized execution, exec, and restore repository
operations perform the same validation at their atomic boundary.

| From | To | Owning actor | Required precondition | Durable evidence at the successful boundary |
| --- | --- | --- | --- | --- |
| `created` | `runnable` | `ProcessManager` launch/direct-fork publication | Image boot and launch authority succeeded; exact `created` revision | Process row, committed launch/fork publication, `PROCESS_CREATED` or `PROCESS_FORKED`, and audit |
| `created` | `waiting_tool` | `ProcessManager` for `ObjectTaskManager` | Newly created ObjectTask runner; `ToolProcessWait.operation_id` names that task | Child creation evidence; the task insert is followed by `OBJECT_TASK_STARTED` evidence |
| `created` | `failed` | startup or publication recovery | Orphaned/failed launch is identified under the recovery lease | Typed failed outcome plus recovery audit; orphan recovery uses idempotent `PROCESS_EXITED` evidence |
| `runnable` | `running` | scheduler or direct-workflow admission | Exact runnable CAS; no existing execution owner/lease | Rotated execution generation, owner id, lease id, and concurrency high-water; scheduler quanta additionally audit `scheduler.run_quantum` |
| `runnable` or token-owned `running` | `running` | `ImageBootService` process-exec admission | Host path has the exact runnable concurrency tuple, or worker path presents the exact active execution token; no typed wait | The complete execution tuple rotates and the durable process-exec publication begins in the same transaction; the old worker token is invalid immediately |
| `waiting_tool` or `runnable` | `running` | `ObjectTaskStateService` | The durable task is still `queued` and owns the exact `ToolProcessWait`, or it is still in the matching Human/child/message wait status after that owner woke the runner to `runnable`; task id and runner pid match, the runner has no wait on the runnable path, and neither path has an execution owner/lease | Runner admission and the task rewrite commit atomically, followed by `OBJECT_TASK_RUNNING` and audit; this Host-managed runner has no scheduler execution tuple |
| `running` | `runnable` | execution owner or successful exec publication | Exact current execution token, or exact exec admission publication | Token is cleared/rotated and state generation advances; no generic process-state event is promised |
| active nonterminal | `waiting_event` | child wait or message manager | Exact revision/generation, no foreign-owned wait or pause gate, and a `ChildProcessWait` or `MessageProcessWait`; the empty mailbox check and message-wait registration are atomic | Typed wait row; child/message domain audit and message events where that operation defines them |
| active nonterminal | `waiting_human` | `HumanObjectManager` | At least one durable blocking Human request, exact process revision/generation, and no foreign-owned wait or pause gate | Human request, typed request-id wait, `HUMAN_QUERY`, operation link, and audit in the owning transaction |
| matching typed wait | `runnable`, same-domain wait, or `paused` | child/message condition owner, Human decision owner, or syscall-owned cleanup | Child/message and syscall cleanup use an exact `ProcessStateToken`; Human uses the exact observed revision/status/generation and acts only on its own request-id wait | Updated process generation plus the owning domain's documented evidence; there is no standalone generic wake event |
| `waiting_human` | `paused` | `HumanObjectManager` | Rejected release/general request, or an unknown provider outcome; the latter and executor finalization of a rejected exact conditional LLM release require the Host-only wait kind | Typed ordinary-pause or Host-resume state and Human response/diagnostic evidence |
| `paused` or compatibility `suspended` | `runnable` | trusted Host resume | Signal is applicable; a Host-resume marker may be cleared only by the Host path | `PROCESS_SIGNAL` and audit with the state transition |
| eligible nonterminal without a condition-owned wait | `paused` | trusted Host/Human control or startup stale-execution recovery | Exact observed state; a live execution lease requires a scoped takeover | Typed ordinary pause or hash-only Store recovery receipt, plus signal/recovery audit/event |
| eligible nonterminal | `killed` | trusted signal, resource enforcement, or ObjectTask cancellation | Exact observed state; a live execution lease requires a scoped takeover | Typed killed outcome and terminal-cleanup intent; signal/resource evidence identifies the cause |
| eligible nonterminal | `exited` or `failed` | process exit, workflow/ObjectTask owner, scheduler failure, or recovery | Exact revision/generation and a matching typed outcome | Terminal row and cleanup intent; ordinary process exit emits `PROCESS_EXITED`, while specialized failures retain their own audit/event contract |
| captured process state | reconciled restored state | `CheckpointManager` | Recovery/publication lease, exact plan anchor, scoped quiescence, and restore CAS | Checkpoint-restore publication, `ROLLBACK`, audit, operation evidence, and fresh revision/execution/state-generation fences |

“Active nonterminal” above is deliberately narrower than “any source status”:
the public manager validates the operation-specific source status. Terminal
statuses have no ordinary outgoing transition. Checkpoint restore is the
explicit reconstruction exception and never lowers concurrency high-water
marks. A restored `running` snapshot is made `runnable`; restored waits are
reconciled against live Human, child, message, and ObjectTask facts before they
remain blocked.

Checkpoint fork has a separate durable publication protocol rather than using
the ordinary `created` → `runnable` launch row above. Each new non-terminal
process row is first inserted as a `created` quarantine row with
`checkpoint_fork_pending_payload`, no wait/outcome, revision `0`, state
generation `0`, execution generation `0`, and no execution owner/lease. After
the captured Object payloads are rehydrated, publication atomically changes
those rows to their exact remapped target states while requiring every fresh
counter and quarantine marker to remain unchanged. Captured terminal rows stay
terminal. See [Checkpoints](checkpoints.md#fork-from-checkpoint) for the recovery
and failure boundary.

See [Runtime Events](events.md) for event-envelope and ordering semantics. A
process transition is not, by itself, a promise that a generic event exists;
only the owning lifecycle operation's documented evidence commits with it.

## Images And Tool Tables

An `AgentImage` defines the default process prompt, tool table, default Skills,
prompt mode, context policy, safety profile, declared required capabilities,
declared required startup modules, an optional default LLM profile, and
optional boot metadata. Fresh images boot from their manifest.
Checkpoint-commit images boot from an immutable internal runtime artifact
derived from one checkpoint root process. Image-package images boot from an
immutable directory-package artifact created from `IMAGE.yaml`, the prompt file
named by that manifest (commonly `prompt.md`), optional `tools/`, optional
`resources/`, and optional `workspace/`.

Image registration and replacement commit the durable manifest, event/audit
records, and any newly inserted package or checkpoint artifact in one
transaction. Shared-cache publication is staged and occurs under the same
registry lifecycle/registration locks only after that outer durable transaction
commits. A failure after artifact insertion therefore keeps the previous cache
entry (for replacement) or keeps the new id absent, while the transaction
restores/removes its durable rows; a caller never observes a manifest whose
registration result was reported as failed. `replace=false` is revalidated
inside the locked transaction. Two concurrent registrations for one id cannot
both win, and one caller's rollback cannot delete another caller's committed
cache entry.

The registry owns deep snapshots of image definitions. Mutating the caller's
registration object, a returned registration result, or an object returned by
`Runtime.get_image()` does not mutate the cached/durable definition. Booting a
checkpoint-committed image restores the process's captured
`loaded_skills.package_snapshot`, but does not replace the current global Skill
or Image registry with historical nested metadata.

`prompt_mode` controls prompt composition. `image_only` is a transparent
upstream-agent boundary: the system message is byte-for-byte the Image prompt,
and the first user message is the process goal (a string is unchanged; a
structured goal is canonical JSON). Later quanta send the cumulative native
`assistant(tool_calls) -> tool` transcript. Tool messages contain the bounded
model projection, not the Runtime result envelope. Object Memory, Capability,
Skill, fallback-action, pressure-notice, and Agent libOS explanatory text are
never injected. This is the default for custom images and image packages.
`minimal_runtime` omits the full `BASE_SYSTEM_PROMPT`/action-planner envelope,
but it is not otherwise transparent: it still adds the complete cumulative
completion contract, factual runtime/state sections, activated Skill bodies and
recovered-goal context, and optional Host-enabled fallback JSON guidance.
Unactivated Skill metadata is not injected; the model obtains it through
`discover_skills`.
`libos_default` additionally preserves the native Agent libOS planner envelope
used by built-in images. Only `image_only` is byte-transparent.

The transparent transcript is reconstructed from the latest complete,
successful `action_selection` call for the exact `(image_id, goal_oid, system
prompt hash, durable LLM-context generation)` anchor plus its call-id-paired
tool outputs, including after an ordinary Runtime reopen. The Responses API
receives explicit historical `function_call` and `function_call_output` input
items and does not rely on `previous_response_id`. Image-only provider errors
use a separate purpose and therefore do not shadow the last complete transcript
head; legacy same-purpose error rows are likewise ignored by the
latest-successful-head lookup. Repair prompts are request-local. A stopped
parallel batch records explicit non-effect cancellation outputs for calls that
were not dispatched. For a nonempty tool-call manifest, every exact call-id/name
output must be paired before the row is replayable. A provider success with no
tool calls is not a usable head until normalization and dispatch validation
writes its separate validation marker. A crash or exhausted repair before that
marker leaves the row unusable and preserves any first-request retry anchor
rather than treating provider success alone as a completed action.

An Image, goal, exact system prompt, or durable LLM-context-generation change
starts a new transcript anchor. Ordinary reopen preserves that generation;
checkpoint restore advances it, so a reopened restored process cannot replay
tool results produced after the checkpoint. If the first provider attempt for
an anchor fails, its exact two-message system/goal request and out-of-band flow
metadata form a separate durable retry anchor. The first successful transcript
that is action-validated, with every nonempty tool call paired, becomes
canonical first; the Runtime then attempts to append a content-free
same-purpose tombstone, after which the failed request can age through payload
retention. A tombstone failure does not discard the successful transcript or
paired tool outputs: it is audited and leaves the old request anchor
conservatively protected. The same protection remains if the Host terminalizes
the process before any successful transcript exists; this release does not
promise automatic terminal redaction for that LLM-call row.

Because a lossless head or retry request is required to avoid silently changing
or replaying an external effect, `image_only` fails before provider dispatch
when `llm.persist_full_io` is false. Payload retention protects the active head
and retry anchor while allowing superseded rows to age normally. This is a
breaking replacement of the former Object-Memory-snapshot behavior: existing
Images selecting `image_only` adopt these semantics after upgrade; there is no
legacy compatibility mode.

A persistent Runtime reopen has one separate, narrow initial-goal recovery path.
A committed root `ProcessManager.spawn` publication records a size-bounded,
integrity-bound JSON recovery envelope for its immutable initial GOAL when
`llm.persist_full_io=true`. During
startup, before the general missing-payload sweep, the Host validates the root
process identity and creation time, current goal id, Object identity and version,
and payload digest before rehydrating that exact live nonterminal goal into the
volatile cache. An exec that preserves the original root goal remains eligible
even if it changes Image; a replacement goal supplied by exec does not. Ordinary
publication reads expose only a hash-only projection.
`persist_full_io=false` writes only that hash-only projection; forks, child
spawns, ObjectTask owner payloads, and ordinary Objects do not gain this recovery
path. Terminal root cleanup redacts any still-full envelope. This exception does
not make SQL Object rows general payload storage. A failed launch enters
`rollback_pending` (and later a terminal non-committable publication state) in
the same outer transaction that reduces any full envelope to hashes. Startup
recovery claims an interrupted launch under the same rule. If that redaction
fails, the transaction leaves the earlier recoverable publication state and
full envelope together behind a recovery fence; it never commits a
non-committable launch state with reversible goal content.

Capability, approval, IFC, resource, audit, and guarded external-effect checks
remain outside the prompt and apply normally. The transcript ledger retains
trusted high-water labels and Object references for the goal and model-visible
tool results, so a historical sensitive result still gates the next LLM egress
and appears in audit `input_refs`. It does not add implicit same-argument
deduplication; mutating operations remain hard-idempotent only where their
protected-operation contract supplies an explicit idempotency key.

The default `llm_context.policy` is `source_only`: context preparation returns
the caller-selected Object Memory materialization unchanged. It records a
metadata-only Context Materialization Manifest, but does not create an
`llm_context` Object and does not append process, Capability, tool, event, or
memory deltas to the model input. Persistent delta enrichment is opt-in either
through Host configuration `llm_context.policy: llm_context_object` or a
Host-issued `context:enrichment/execute` capability for that process.

When explicitly enabled, the context helper uses the active Runtime's
`llm_context.schema_version`, `llm_context.object_name_prefix`, and
`llm_context.recent_event_limit`; these are not import-time constants. New
context Objects carry the active schema version, and an existing context Object
with a different schema fails closed before reuse. Event capture consumes the
same configured, store-bounded window. In Runtime-owned prompt modes, the
executor advances the process event cursor only after the provider successfully
returns for a request that actually carried that event projection. `image_only`
does not project or advance this cursor, and preflight, conditional-release, or
provider failure leaves it unchanged. The rendered append-only context records
one initial Capability snapshot and then
keyed Capability/tool deltas. Repetitive bookkeeping events are projected as
bounded counts and aggregate usage. Compaction resets captured signatures so
the next quantum appends a fresh baseline before later deltas.

`planner.context_management` controls model-window pressure handling. With no
entry, the image selects `auto_compact` at 80% projected occupancy. That mode
may call `compact_process_context` only when persistent context is explicitly
enabled for the process and the process independently holds
`context:maintenance/execute`. The default source-only path records the
pressure decision as not authorized and leaves the request unchanged: it does
not inject a notice or dispatch a maintenance tool. `prompt` is an explicit
Image policy that appends the image-owned literal reminder and numeric pressure
facts in Runtime-owned prompt modes. Image registration rejects that policy for
`image_only`, where Runtime-authored notices would violate transparency.
`disabled` records pressure but takes no action. For stateless requests,
projected occupancy is the deterministic conservative estimate of the complete
assembled input plus the profile's `max_tokens` output reservation. An eligible
Responses continuation supplied by a delta-oriented caller would additionally
add the provider-reported lower bound for retained history; the current
full-snapshot AgentProcess executor supplies no such continuation, and usage
from a prior stateless/chat request is not reused. In `prompt`
mode the Runtime applies a bounded fixed-point pass to the numeric notice, then
re-estimates and records the exact notice-inclusive request. If that final
request would exceed the configured context window, it records
`prompt_notice_exceeds_context_window` and fails before provider dispatch.

A configured automatic tool is never added to the process tool table: it must
already be present and remains subject to its argument schema, Capability,
resource, approval, event, and audit checks. One attempt is made per pressure
episode and policy fingerprint. Separately, an `auto_compact` image starts the
same durable maintenance path, only for explicitly enriched and independently
maintenance-authorized processes, when its persisted LLM-context payload reaches
`llm_context.storage_compaction_threshold_bytes` (96,000 bytes by default),
before Object Memory's 200,000-byte default hard limit. This proactive storage
path is enabled only when the process has explicit
`context:maintenance/execute` authority; tool, child-spawn, and image-read
checks still apply independently. The built-in compactor
is forced for this storage trigger so it cannot skip a payload that is below its
token target. Unless the Image explicitly selects another value, storage
maintenance uses at most `llm_context.storage_compaction_max_chunks` stages
(four by default) to bound latency and model cost, and retains zero verbatim
tail entries by default via
`llm_context.storage_compaction_preserve_recent_entries`; the cumulative
compressor summary remains. This avoids feeding context-maintenance artifacts
back into an immediate second compaction. After compaction, the first safe
projected payload becomes a durable baseline and the storage trigger is
re-armed at one configured waterline of additional growth (capped immediately
below the hard limit). This hysteresis prevents maintenance events and child
summaries from causing a compaction loop even when a Host selects a low valid
waterline. If that first post-compaction projection itself cannot fit below the
hard limit, the process fails instead of retrying an ineffective compaction.
The storage check uses the projected
payload after the current durable deltas are assembled but before that payload
is written, so one large delta cannot jump directly over Object Memory's hard
limit. The Runtime ends the quantum before provider dispatch, preserves
Human/child/message waits, and resumes only after the context generation
changes. A failed storage-triggered attempt fails the process with audit
evidence instead of recursively rebuilding the same oversized context. Images
using `prompt` or `disabled` retain their selected policy and do not receive
automatic storage compaction.

A successful compaction, or an initial failure
that nevertheless changes the durable context generation, ends the current
quantum so the next quantum re-materializes context. An initial failure with no
generation change is audited and the original model request continues without
an injected tool result or fallback prompt. Human, child, and message waits are
durable. For model-window-triggered maintenance, failure after such a wait
terminally completes that pending generation and immediately rebuilds the
ordinary request from the current context generation; storage-triggered failure
instead terminates as described above.

Root process spawn never grants image `required_capabilities`.
Requirements are copied into the Host-authored
Task Authority Manifest and reported as satisfied or unmet. Only the
manifest's `authorized_capabilities` compile into authority.
`exec_process`, checkpoint-commit image boot, and image-package boot never
grant requirement declarations automatically.
Image `required_modules` are always startup prerequisites only: spawn and exec
fail unless each declared module id is already loaded with the declared
`source_sha256`.

At process creation time, the runtime resolves only the image's explicit
`default_tools` as its initial static tool table. No lifecycle, Object Memory,
or other builtin tool is implicitly added. Image-package JIT publication and
later Skill/JIT activation may add process-local registrations through their
separately validated lifecycle. A registered Skill can add globally registered
static tools to both the complete and model tables under Skill authority; an
immutable built-in Skill is stricter and can only project its complete
`allowed-tools` set when every binding already exists in the Image-authorized
table. A process can call only tools in its resulting complete table, while the
LLM receives only the separate model projection. Either path still fails at
primitive use if resource authority is missing. Internal runtime paths such as
JIT syscalls may call primitives directly without exposing a corresponding
builtin tool to the model.

An image with `metadata.tool_projection: skills` initially projects only
`discover_skills`, `activate_skill`, `read_skill_resource`, `unload_skill`, and
`process_exit`, all of which must be present in the image's explicit
`default_tools` or image validation fails.
Shipped images leave `default_skills` empty. All Skill metadata is discovered
through `discover_skills`. Activating an immutable built-in Tool Skill copies
its complete exact Image-owned binding set from the existing full table into the
durable model projection. It leaves the full table and Capability set unchanged;
the Skill is hidden and activation fails if any declared Image binding is
missing, replaced, or absent from the current table. Partial projection is never
accepted.

A registered Skill follows a different governed path: after its registration,
trust, and execute checks, activation may add existing static tools and validated
process-local JIT tools to both tables. Discovery uses the same bounded
metadata-term matching, relevance ordering, and `next_step` contract for every
visible package source. An explicitly configured custom Image may list
`default_skills`; that is a boot-time activation list, and any failed activation
fails spawn/exec rather than starting a partial Image. This is Host-authored
Image assembly, so it does not require a separate Skill `execute` grant on the
new process; package integrity, trust, loadability, tool-binding, and primitive
authority checks are not bypassed.

The current built-in image contracts are:

| Image | Intended work | Initial model projection | Declared requirements |
| --- | --- | --- | --- |
| `base-agent:v0` | General runtime work and coordination | 5 Skill lifecycle/bootstrap schemas | configured Human write |
| `coding-agent:v0` | Repository inspection, editing, Git, and verification | 5 Skill lifecycle/bootstrap schemas | configured Human write and workspace read |
| `maintenance-agent:v0` | Durable repository maintenance with a fixed reproduce/edit/test/Git/checkpoint/delivery contract | 13 narrow direct schemas | configured Human write and workspace read |
| `research-agent:v0` | Source-backed evidence collection and synthesis across local and Host-registered integrations | 30 narrow direct schemas | configured Human write and workspace read |
| `analysis-agent:v0` | Reproducible data analysis with explicit quality and verification gates | 23 narrow direct schemas | configured Human write and workspace read |
| `operator-agent:v0` | Transactional customer/enterprise workflows with read-back-before-replay recovery | 23 narrow direct schemas | configured Human write |
| `review-agent:v0` | Evidence-first review; read-only unless repair is explicitly requested | 5 Skill lifecycle/bootstrap schemas | configured Human write and workspace read |
| `toolmaker-agent:v0` | Import-free Deno/TypeScript JIT proposal, validation, and registration | 5 Skill lifecycle/bootstrap schemas | configured Human write |
| `context-compressor:v0` | Structured context compaction | `process_exit` only | none |

“Initial model projection” is not the complete static/callable table size. In
particular, review's read-only posture is a prompt and launch-authority contract,
not an absence of dormant mutation bindings in its static table; those bindings
still require Skill projection and primitive authority before model use.

The prompt lists only activated Skill bodies. Discovery metadata is obtained on
demand through one source-neutral schema. Visibility remains separate from
authority: ToolBroker and direct workflow calls require a complete-table
binding; model-originated calls additionally require model projection; primitive
authorization uses Capability and policy and treats neither table as resource
authority. Skill activation itself grants no primitive authority.
Requirement declarations remain Task Authority Manifest inputs, not grants.
Configured base/coding ids must also remain distinct from the fixed maintenance,
research, analysis, operator, review, toolmaker, and context-compressor ids; a
collision fails Runtime construction instead of silently replacing an image.

The four long-horizon specialist images deliberately omit
`metadata.tool_projection`. Their complete, small `default_tools` tables are
therefore model-visible at boot without Skill discovery turns. This is a
visibility optimization, not an authority grant: every primitive still checks
Capability, Task Authority, provider policy, data flow, budgets, and durable
effect settlement. The broader general-purpose images retain source-neutral
on-demand Skill projection for tasks whose tool domain is not known at launch.

LLM selection is host-controlled and process-local. A process stores only an
`llm_profile_id`; the host Runtime resolves that id to a configured
OpenAI-compatible profile at LLM-call time. Root spawn uses an explicit host
profile, then the image default, then `config.llm.default_profile_id`. Fork and
fresh child creation inherit the parent profile by default. Exec keeps the
current profile unless the host explicitly overrides it. Model-facing process
tools do not expose LLM profile switching in v1. Only the configured default
profile inherits legacy `OPENAI_*` provider and model environment variables;
other named profiles require explicit host profile fields for non-default
routing.

The default provider posture is stateless: `llm.store=false` and
`responses_previous_response_id=false`. This limits provider-side retention; it
is not an end-to-end privacy guarantee. By default `llm.persist_full_io=true`
also retains prepared prompts and visible tools and, for successful responses,
tool/reasoning records, response content, and raw provider response fields in
the RuntimeStore subject to Host retention policy and database access controls.
A failed call's ordinary request fields follow the same policy, but provider or
extension exception text is never persisted or exposed to the model regardless
of this setting; durable failure evidence contains only a stable public
envelope and content-free internal observations such as type, length/hash, and
correlation id. Set `persist_full_io` to `false` when full local I/O evidence is
not acceptable, while accounting for image modes that require durable
transcript material.

The built-in OpenAI-compatible client materializes a versioned, bounded
Provider trace in the existing LLM-call reasoning field. Each entry represents
one network attempt made by Agent libOS and records only sanitized metadata,
Provider-returned readable reasoning/output, tool actions, diagnostic usage,
and timing. SDK-hidden retries and redirects are disabled; Agent libOS performs
its configured retries explicitly inside the same protected logical call.
Custom clients remain usable but are marked `custom_client_incomplete` because
their internal network attempts are not observable. Attempt diagnostics do not
create extra resource reservations or change the one-logical-call budget.
Trace persistence is terminal-call evidence, not a crash-safe per-attempt
journal; cancellation or process loss may therefore leave only the protected
effect and budget evidence.

The low-level OpenAI client sends an explicitly supplied
`previous_response_id` only for an official, stored Responses request whose
tool history is representable. That client does not receive or enforce Runtime
process, context, Sink, manifest, or credential fingerprints. The Runtime still
computes and records scope-sensitive provider and data-flow fingerprints, but
the current AgentProcess full-snapshot executor deliberately disables provider
continuation and replays the complete locally materialized context on every
request. Those fingerprints are observability and future-protocol inputs, not
an enabled low-level continuation guard. Enabling
`responses_previous_response_id` or provider storage therefore does not enable
AgentProcess chaining today; it only changes allowed client/profile policy and
may increase provider retention. Context compaction and checkpoint restore
still advance durable generations so a future Runtime continuation mechanism
can fence those boundaries. The LLM request remains a formal
bidirectional protected operation, and an unclassified response is aggregated
as `normal/untrusted` rather than replacing request labels.

## Data-flow state

Every runtime-mediated payload exit constructs a typed `DataSink` and a
runtime-owned `DataFlowContext`. The context contains strict labels and exact
Object id/version/content hashes from materialization, explicit Host
`source_oids`, and ambient process observations. Before resolving provider
state, some paths perform a read-only Sink-clearance precheck that neither
requests nor consumes a release; paths whose Sink and payload are already fully
bound may instead request exact release at this early gate, after visibility and
definite-denial checks. External LLM, Human, JSON-RPC, MCP, Git, filesystem
write, Shell, and PTY protected operations still authorize or revalidate the
final payload-bound Sink after ordinary authority decisions and immediately
before dispatch. Prompt-visible process events contribute their trusted labels
to that request's data-flow context before these checks, including under the
default `source_only` policy; only the opt-in persistent LLM context object also
records them in its durable label high-water. Process goals/messages/results
and Object Tasks propagate the same identity domain internally without
downgrading it.

The durable Sink registry is separate from `TaskAuthorityManifest`. The
manifest can only constrain which tenant/principal data a process may receive;
it cannot create external trust. A conditional high-sensitivity exit blocks on
a metadata-only Human release bound to the exact Sink, source versions, labels,
payload, operation, manifest, and registry generation. See
[Data Flow](data_flow.md) for trust levels, identities, persistence, and the
guarantee boundary.

When that exit is the LLM provider request itself, the default
`llm.persist_full_io=true` policy persists the fully prepared messages, tools,
request options, profile/Sink identity, provider-state scope, and flow binding
in the `llm_release` wait generation. With `persist_full_io=false`, SQL keeps
only the prepared-request and payload hashes plus non-sensitive resume
metadata, while the exact request remains in executor memory. The same runtime
can resume that hash-bound request once; after reopen or any loss of the
in-memory generation, the runtime fails closed before provider dispatch rather
than rebuilding it. Profile identity drift or an already-claimed generation
also fails closed. Rejection clears the prepared generation and installs a
Host-only pause marker. Direct-child model signals cannot resume through that
marker; an explicit Host resume begins a fresh turn rather than replaying or
silently regenerating the rejected request.

Checkpoint restore gives every restored process a fresh durable LLM-context
generation. For a full-I/O `image_only` conditional release, the exact persisted
approved messages, tools, Sink/flow binding, and resume token remain the request;
immediately before dispatch the executor may rebind only the frozen transcript
anchor and request marker from the captured generation to the restored one and
audits that the request payload did not change. This prevents a post-restore
successful head from being filed under the pre-restore epoch. It does not relax
profile, Sink, trust, flow, or single-claim checks. The redacted full-I/O opt-out
still fails closed after restore because its exact request is unavailable.

LLM-selected blocking actions use durable wait generations. Each row has a
unique `resume_token` plus the causal LLM and Tool operation ids; one executor
CASes `pending -> resuming` before crossing
the resumed primitive boundary and CASes the same generation to `completed`
afterward. Reblocking writes a new token, closing stale-worker ABA. Once the
claim succeeds, any dispatch, output-persistence, or completion exception
immediately fails the process, retains the non-replayable durable state, and
audits `llm.pending_action_resume_interrupted`; reopening an already-`resuming`
row follows the same fail-closed rule rather than replaying a possibly completed
external effect.

The Explainable Operations lifecycle mirrors this state machine. The LLM and
Tool rows become `waiting` before control returns to the scheduler. Resume
reactivates those exact rows even though the concrete retry may get a new
`call_id`. A runtime reopen preserves waiting rows but marks any orphaned
`running` row `interrupted`. Human terminal evidence is attached to the waiting
Tool operation without prematurely changing it to running. See
[explainable_operations.md](explainable_operations.md).

`jit_tool_exposure` controls how JIT tools appear to the LLM. `direct` exposes
each visible JIT as its own OpenAI tool. `multiplexed` exposes one stable
`run_jit_tool` protocol tool and maps it back to the real process-local JIT
before execution. Multiplexed mode hides individual JIT names from runtime
tool sections and event context; image prompts remain responsible for listing
any JIT catalog the model should know.

A checkpoint-commit image remaps baked Object Memory, process-local JIT tools,
loaded Skill package snapshots, and cwd into the new process. It also carries
the checkpoint's loaded startup module summaries as `required_modules`. It does
not package or restore filesystem, shell, JSON-RPC endpoints, global Skill
trust, human, network, or provider side effects.

An image-package boot materializes the package `workspace/` seed into a private
per-process directory under the configured `image.materialized_workspace_root`
(`agent_outputs/image_workspaces/` by default), sets the process
cwd from the package manifest, and grants only the manifest-declared
`workspace.grants` for that private copy. Package JIT tools live under
`tools/jit-tools.json` and `tools/scripts/*.ts`; they are registered as
process-local ephemeral tools and are not copied into the workspace. Package
artifacts persist only declared package content: `IMAGE.yaml`, the referenced
prompt, declared `workspace/` content, referenced `tools/` JIT files, and
`resources/`. Cache, VCS, likely secret, and platform-unsafe paths are rejected.
When boot/exec compensation succeeds, it removes the private workspace,
unpublished JIT tool rows and process aliases, candidate source rows, and
candidate Object Memory descriptors before returning failure. If compensation
cannot complete, the durable publication and residual artifacts remain under a
recovery-required fence; a fresh Runtime reconciles or fail-closes that exact
publication after the diagnostic store is explicitly released.

## Working Directory

Each process has its own workspace-relative working directory. Relative
filesystem paths and shell subprocess cwd resolve from that process cwd. The
runtime host process does not `chdir` into launched workspaces.

Changing a process cwd requires `read` on the selected filesystem directory.
An explicit cwd supplied to child spawn, direct fork, or PTY creation—and every
explicit `chdir`—is validated through the filesystem directory primitive after
the applicable spawn/image or shell authority gates. When those creation calls
omit cwd, they reuse the stored parent/current cwd identity without a new
directory observation. An explicit directory `state()` observation
therefore runs under a structured filesystem intent rather than acting as an
unauthorized existence oracle. Finite directory-read authority is consumed
only after an observation; an ambiguous provider failure leaves unknown effect
evidence. Root `ProcessManager.spawn(working_directory=...)` is a trusted Host
construction surface: it normalizes and stores the workspace-relative identity
but does not itself perform a provider existence/read observation.

The CLI command:

```bash
uv run agent-libos --db .agent_libos.sqlite cd <pid> <workspace-relative-dir>
```

After the Host grants that process directory-read authority for the selected
path, this updates one process working directory and leaves other processes
unchanged.

## Object-Bound PTY Sessions

The trusted `modules/pty` runtime module can add an interactive PTY surface.
It is a repository/source-distribution asset rather than a core-wheel module.
`pty_create` starts the host PTY through the shell primitive's authorization
path and returns a mutable Object Memory `EXTERNAL_REF` object id. The object
payload records descriptive metadata such as argv, cwd, backend, dimensions,
and creation time, but authorization for later interaction comes only from the
current Object capability graph and the in-memory PTY registry.

The local POSIX backend supervises the process tree for wall/CPU/RSS usage. The
current Windows `pywinpty`/ConPTY backend has no Job Object or resource
supervisor and advertises no `SubprocessLimits` support, so a budgeted PTY spawn
fails closed before creating the child.

`pty_read` requires object `read`; `pty_write` requires object `write` and the
original session owner pid; `pty_resize` requires object `write`; `pty_close`
requires object `delete`. Closing the session releases the object and revokes
related object capabilities. If the object is released by a lifetime scope,
process-owned memory cleanup, direct trusted delete, or runtime shutdown, the
module-bound Object release or shutdown finalizer closes the underlying PTY as
the object's RAII resource.

PTY spawn uses `pty:spawn:<resolved-executable>` plus the executable content
hash for data-flow clearance and revalidates both before spawn. The session
records that immutable trust identity at creation; later input, resize,
and public close use the visible `pty:session:<session-id>` Sink but resolve
trust only through the spawn identity. Successful or ambiguously applied input
and resize operations raise the session's label high-water; an ambiguous public
close does the same while the session remains unresolved. Internal lifecycle
close remains separate so cleanup cannot be blocked by Sink clearance. A
session cannot switch to a differently trusted executable. Trusting that Sink
means the Host trusts the native program to receive argv, stdin, and session
control signals; it does not constrain additional file/network I/O performed
by that program.

Finite object write/delete decisions for `pty_write`, `pty_resize`, and
`pty_close` are reserved before the host call. A certified not-started result
restores that exact reservation only when every completed earlier phase has
`state_mutation=False`, `information_flow=False`, and
`commits_authority=False`; ordinary or ambiguous failures commit the use and
retain pending/unknown evidence. Because `commits_authority` defaults to true,
even a successful otherwise non-effectful phase closes this restoration floor
unless it explicitly opts out. When the child exits on its own, the monitor
writes a close intent before reading the exit code and closing the handle. Once
the exit-code read succeeds, even a later not-started close cannot abandon that
information-flow intent.

PTY spawn, read, runtime-internal continuous ingest, write, resize, and close
are external-effect operations. Each writes a structured pending intent before
its provider boundary and conditionally finalizes the same effect id afterward;
event/audit/finalization failure leaves the row pending and unknown. A spawned
host session whose Object publication later fails is cleaned up but remains an
`unknown` spawn effect with failure-phase and cleanup metadata. Unsupported or
failing post-operation classification finalizes an unknown fallback rather
than erasing the effect. The local provider classifies spawn/write/close as
irreversible, resize as rollbackable-but-not-applied, and read/ingest as
information-flow observations for which no rollback is required; checkpoint
restore does not compensate the mutations.

PTY sessions are not checkpointed or persisted as reconnectable host handles.
A checkpoint or committed image may contain an `EXTERNAL_REF` row only as
descriptive metadata; it cannot rewind or recreate the provider resource.
Checkpoint fork drops owned and borrowed `EXTERNAL_REF` roots and object
capabilities rather than aliasing the source terminal. Reopening the runtime
releases stale PTY objects rather than trying to reconnect a host process.
If a live Runtime is recovery-fenced, its explicit diagnostics handoff invokes
only the PTY module's recovery-safe transient cleanup: it stops reader/monitor
workers and closes the process/FD handle without changing that durable Object
or any evidence row. Failure leaves the store open and the callback retryable;
success lets same-process reopen perform the normal stale-object recovery.

## External Effect Ledger

LLM, filesystem, Git, clock, shell, human output/terminal I/O, PTY, JSON-RPC,
and live MCP primitives close the crash gap around provider calls with a
durable external-effect intent.
Immediately before the provider boundary they insert an `unknown` record with
structured `effect_state: pending`. A classified success or ambiguous provider
exception CASes that same `effect_id` to `finalized`, matching its pid,
provider, operation, and target. Repeated, stale, or cross-boundary finalization
fails closed instead of adding another final record or altering unrelated
evidence.

Transaction outcome and rollback support are separate axes. Recognized
transaction states are `prepared`, `authorized`, `approved`, `dispatched`,
`committed`, `failed`, `unknown`, and `compensated`; not every operation passes
through every intermediate state. A confirmed provider completion is
`transaction_state: committed` even when rollback support remains `unknown`;
only an explicit unknown provider outcome is `transaction_state: unknown` and
propagates `unknown` to its operation tree.

`ProviderEffectNotStarted` is the only provider result that can prove its call
boundary was not crossed. The primitive conditionally deletes a still-pending
intent only when every completed earlier phase has neither mutated state nor
observed information and explicitly used `commits_authority=False`. It restores
any exact finite-use reservation and abandons the intent in one transaction. If
an earlier filesystem state read or MCP live validation succeeded, a later
not-started mutation/tool call finalizes the observed partial effect; a phase
with the default `commits_authority=True` likewise prevents restoration even if
its other two flags are false. After the provider may have run, a failure in
capability commit, resource/event/audit handling, classification, or final
effect persistence leaves the pending `unknown` record in place. Checkpoint
diff/restore and the runtime-safety benchmark consume that row conservatively,
so a process crash cannot turn an uncertain external effect into “no effect
recorded.”

Manifest v2 MCP keeps protocol discovery/initialization, every bounded
`tools/list` page, and `tools/call` inside one protected composite operation.
One absolute deadline and cumulative request/response byte reservation span all
phases; each completed phase contributes a bounded exchange receipt. Safe
legacy fallback can occur only before Tool dispatch. An unsupported modern
`input_required` response is never retried: a consequential or ambiguously
mutating call remains `unknown`, and Durable Task Run reconciliation blocks in
`needs_attention` rather than treating it as a normal resumable wait.

Terminal human reads and automatic prompt/decision writes follow the same
protocol. Their effect and audit context contains request id, purpose,
byte/character counts, and hashes only; provider exception text is not
persisted there. The complete prompt/request payload and terminal answer are
persisted separately in `human_requests.payload_json` and `decision_json` and
survive reopen. A successful human interaction is not replayed when later
event/audit/classification settlement fails: the request decision commits and
the pending intent remains the reconciliation evidence. A Human provider that
certifies `ProviderEffectNotStarted` instead abandons the intent and restores
retryable request/finite-authority state.

Startup reconciliation is isolated per pending intent. If one provider's
reconciliation hook raises, that intent remains explicitly `unknown` and the
runtime continues opening and reconciling other providers; the exception text
is not persisted.

All startup recovery runs while the new Runtime owns the lifecycle recovery
lease and before normal mutation admission opens. The builder first reconciles
prepared protected operations and stale finite capability reservations, then
pending external effects and resource-usage reservations. It next recovers
incomplete process-exec, process-launch, and checkpoint-restore publications,
rehydrates recoverable Object/JIT state, interrupts stale operation/execution
claims, and recovers Object Tasks. Each backlog is read in configured,
hard-bounded keyset pages. Runtime publications are also durably bound to their
Explainable Operation rows, so startup converges a terminal publication and its
operation outcome instead of inferring an outcome from process state alone.

The Runtime then remains non-`OPEN` while it runs startup hooks, starts the
ObjectTask worker, performs any checkpoint payload delivery handshake,
reconciles terminal restore publications again, and commits the payload
acknowledgement. Only after this STARTING phase completes does normal mutation
admission open.

For JSON-RPC and Streamable HTTP MCP, the pending intent and all finite remote
authority reservations are durable before non-local DNS resolution. A
successful DNS lookup, or an ordinary DNS failure after host observation, is
already information flow and commits the reservations. Consequently a later
transport-level `ProviderEffectNotStarted` finalizes an information-flow-only
unknown outcome instead of abandoning the intent. Local HTTP fast paths and
stdio have no DNS observation, so a certified failure at their first provider
boundary can still restore and abandon atomically.

Clock sleep is one composite observed operation. Synchronous `sleep` and async
`asleep` persist the intent before their first provider `monotonic()` call and
mark it `information_flow=true`, including on success, because the returned
elapsed time comes from provider observations. Only
`ProviderEffectNotStarted` at that first measurement can restore a finite-use
reservation and abandon the intent. Any ordinary first-measurement exception,
or any later sleep, cancellation, or second-measurement failure, consumes the
reservation and finalizes the same id as `unknown`.

## Evidence Payload Retention

Evidence identity and payload retention are separate. The Host-controlled
`PayloadRetentionMaintenance` service is disabled by default and is never run
implicitly during startup. When explicitly enabled and applied, it advances
eligible terminal LLM-call and external-effect payloads monotonically from
`full` to content-free `summary` and then `hash_only`. It retains call/effect
identity, provider and process identity, causal audit/event links, timestamps,
canonical argument hashes, transaction/classification state, and an aggregate
payload digest.

External effects are eligible only when `effect_state` is `finalized` and
`transaction_state` is `committed`, `failed`, or `compensated`;
`effect_state=pending` and the `prepared`, `authorized`, `approved`,
`dispatched`, and `unknown` transaction states are ineligible. LLM calls must
likewise be terminal with a durable completion timestamp, and
continuation/recovery heads remain protected until their executable semantics
have another durable projection.
Every request is a bounded, cursor-based preview or apply operation; applied
updates and the payload-free maintenance audit summary commit atomically. See
[Evidence and LLM Payload Retention](evidence_payload_retention.md) for the
exact tiers and eligibility rules.

## Scheduler

The scheduler is thread-backed. It enumerates and claims only `runnable` rows.
Its normal executor starts one worker task per claimed process up to
`config.scheduler.max_workers`, and each task advances only that process until
it blocks, exits, fails, or the shared quantum budget is exhausted. It never
claims a `running` or waiting row, including a direct-workflow process or an
ObjectTask runner. Public
async APIs remain available for event-loop hosts, but they are wrappers around
the same scheduler and do not mean process quanta are serialized on one asyncio
loop.

When a quantum itself returns an awaitable, its worker creates a private event
loop, tracks the main task for cross-thread cancellation, and runs it to a
result. Before closing that loop the worker repeatedly cancels and drains every
remaining background task, then attempts async-generator and default-executor
shutdown using one shared scheduler shutdown-join deadline for the caller's
bounded wait. Cleanup is not limited to the main awaitable: a
cancellation-resistant background task or async generator makes the quantum
fail after the bounded cleanup attempt. A default executor is detached from the
loop and joined by a daemon monitor so its blocking `shutdown(wait=True)` cannot
strand the quantum caller. Until that join actually succeeds, a tracked
lifecycle future remains active; scheduler shutdown reports incomplete and
ordinary Runtime shutdown leaves the store open rather than closing it under a
live executor. After the monitor proves that the executor drained, shutdown can
be retried.

Unblock quanta use a second executor with the same configured capacity. During
the bounded unblock window, normal and unblock work can therefore occupy up to
approximately twice `max_workers`; the setting is a per-pool capacity, not a
global thread ceiling. Scheduling is status/claim safe but does not promise
round-robin fairness between runnable pids.

The high-level synchronous entrypoint is:

```python
results = runtime.run_until_idle(max_quanta=10)
```

Event-loop hosts should use:

```python
results = await runtime.arun_until_idle(max_quanta=10)
```

When callers omit `max_quanta`, the Runtime uses
`runtime.run_until_idle_max_quanta`; its default is `None`, meaning no nominal
quantum limit and no bounded-run drain deadline. In all cases, a run:

1. runs runnable processes,
2. processes pending human terminal messages when work is blocked on human I/O,
3. delivers process-message notices at tool boundaries,
4. observes processes that condition-owning managers have returned to
   `runnable`,
5. stops when no runnable or human-resumable work remains, or when the quantum
   budget is exhausted.

`max_quanta` is a global budget across all process workers, not a per-process
limit. After a bounded run spends that nominal budget, it may reserve one extra
quantum when at least one submitted future is still active but its persisted
process status is `waiting_event`, `waiting_tool`, or `waiting_human`, and a
different schedulable PID is runnable but has no submitted future. These
unblock quanta use a separate executor and are capped at
`max(1, max_quanta)` for the run; each reservation is audited as
`scheduler.unblock_quantum_reserved`. This prevents a blocked future from
starving work that may unblock it without claiming that the Runtime has proved
a particular dependency relationship. Once neither an unblock reservation nor
normal progress is possible, `config.scheduler.drain_window_s` gives active
workers a short chance to finish before unfinished quanta are cancelled or
detached.

Host-managed incremental runners may set
`cancel_inflight_on_budget_exhaustion=False` when `max_quanta` is only a
completed-batch boundary. That setting still prevents admission of another
ordinary quantum, but waits for an already admitted provider/tool quantum to
finish. The GUI background controller uses this mode for its one-quantum
batches so a slow real LLM response is not mistaken for budget cancellation.
The public default remains `True`; bounded callers therefore keep the normal
drain-window cancellation behavior unless they opt into this host lifecycle
contract.

The scheduler serializes top-level `run_until_idle`, `run_pid_until_idle`, and
single-step invocations for one `Runtime` instance, so two host calls cannot
re-enter the same runnable process concurrently. Individual process claims are
also status-checked at the store boundary before a quantum changes a process
from `runnable` to `running`.

The scheduler does not convert typed waits to `runnable`. Child completion,
message posting, Human decisions, ObjectTask notification/resume, and
syscall-owned cleanup each use their own exact owner fence. Child/message and
syscall cleanup use generation tokens; Human/ObjectTask services use exact CAS
and durable service ownership. The scheduler only sees the resulting runnable
row on a later enumeration. The bounded “unblock
quantum” pool likewise runs a different runnable PID; it does not wake or claim
the blocked PID.

Single-step APIs remain available for tests and debugging:

```python
result = runtime.run_next_process_once()
result = runtime.run_process_once(pid)

result = await runtime.arun_next_process_once()
result = await runtime.arun_process_once(pid)
```

For debugging pending approval states, disable human queue processing:

```python
results = await runtime.arun_until_idle(max_quanta=1, process_human_queue=False)
```

Hosts must always close a Runtime they own. Use the synchronous API only from a
synchronous host:

```python
from agent_libos import Runtime

runtime = Runtime.open("runtime.sqlite")
try:
    run_host(runtime)
finally:
    shutdown_result = runtime.shutdown()
```

An event-loop host must use the async lifecycle pair so loop-affine finalizers
and components run on the caller loop:

```python
from agent_libos import Runtime

runtime = await Runtime.aopen("runtime.sqlite")
try:
    await run_host(runtime)
finally:
    shutdown_result = await runtime.ashutdown()
```

`ashutdown()` also drains synchronous blocking cleanup off-loop. Calling
`shutdown()` from a running event loop is rejected before admission closes when
the Runtime has async-only shutdown work; event-loop hosts should not select it
dynamically. In either form the host must inspect `shutdown_result["ok"]` and
retain the Runtime for a retry or diagnostics handoff when it is false.
Store-readiness misuse such as calling shutdown from the current store
transaction raises during preflight before a shutdown attempt or result exists.

Ordinary shutdown closes admission and drains active admission leases before it
invokes component callbacks. While store ownership remains, it next attempts to
record a `runtime.shutdown` audit row and `RUNTIME_SHUTDOWN` event; an evidence
failure returns before any component callback. After successful evidence it
stops the scheduler, stops ObjectTask runners, runs registered finalizers, stops
modules, LLM clients, supervised blocking work, and the substrate, and finally
claims and closes the store. The audit/event therefore means that a shutdown
attempt reached the evidence phase; it is not a success marker for the later
teardown stages.

Scheduler shutdown cancels and joins tracked worker futures for up to
`config.scheduler.shutdown_join_timeout_s`; ObjectTask shutdown drains its tool
executor for up to `config.object_tasks.shutdown_join_timeout_s`. A retryable
stage failure returns `ok: false`, `already_shutdown: false`, `reason`, a dynamic
`<stage>_stopped: false` key such as `admission_stopped`,
`shutdown_evidence_stopped`, `scheduler_stopped`, `object_tasks_stopped`, a
`<finalizer-handle>_stopped` key, `modules_stopped`, `llms_stopped`,
`blocking_work_stopped`, `substrate_stopped`, or `store_stopped`, and optional
structured `errors`. The store remains open rather than closing underneath live
work. Once the failed stage can stop, a later shutdown repeats the evidence when
available and continues cleanup. Success returns `ok: true`,
`already_shutdown`, and `reason`, plus optional `warnings`; a recovery fence
instead returns `recovery_required: true` and requires the explicit handoff
described below.

Persistent stores also take an active-runtime lease: SQLite holds an exclusive
lock on the database actually opened and, on the hardened POSIX path, adds a
secure path-sidecar `flock` plus an owner-only identity lease keyed by the
validated database `(st_dev, st_ino)`.
The database and lease files must be regular, current-user-owned, single-link
files; the canonical database path is checked before and after open. This
rejects ordinary hard-link aliases and replacement races, while the connection
lock preserves the one-writer boundary for the database actually opened.
Because the standard driver cannot expose that database descriptor, same-UID
path administration remains a Host trust boundary; a live database path and
parent must not be renamed or replaced. Platforms without the sidecar/identity
mechanism retain the exclusive lock without claiming POSIX path/inode hardening.
PostgreSQL uses a database/schema-scoped session advisory lock. Another
writable Runtime cannot open the same database/schema target until the active
Runtime closes and releases the lease.

SQL transactions nest through savepoints. An outer commit error triggers SQL
and Object-payload rollback only when the driver proves that the transaction is
still active. If commit may already have applied, the Runtime treats the
outcome as ambiguous: it neither reports rollback nor restores a cache-only
before-image, and it poisons/closes the store data plane unless a narrow typed
caller confirms and accepts the exact committed state. A failed nested
`RELEASE SAVEPOINT` attempts `ROLLBACK TO` followed by `RELEASE`; payload
before-images are restored only after that SQL recovery succeeds. Failure of
the rollback/savepoint cleanup poisons and closes the store, so later ordinary
reads and writes fail closed rather than observing a manufactured mixed state.

Ordinary shutdown claims the exact store guard immediately before close. Its
async form runs backend close off-loop and drains the result through caller
cancellation. After the backend release point, warnings are retained on
idempotent shutdown readback and a leader's cancellation or control-flow
diagnostic is not replayed to concurrent followers. If ownership was already
lost before shutdown begins, durable shutdown evidence is unavailable; the
runtime records that warning in memory, still drains the transient graph and
clears its stale exact guard, and ends `closed`.

A recovery-required fence is different from an ordinary drain timeout. It is
monotonic for the affected Runtime instance: every later `close()` or
`shutdown()` remains fail closed with the recovery reason and leaves the store,
commit guard, backend lease, transient components, and diagnostic state intact.
Ordinary shutdown never clears that fence, writes shutdown audit/event evidence,
runs user finalizers, or tears down components behind an active admission.

Once diagnostics have been captured, the owner may explicitly hand the store
off with `Runtime.release_recovery_diagnostics()` or, for an async host,
`await Runtime.arelease_recovery_diagnostics()`. The operation rejects an
unfenced/open Runtime, a forged internal authority, and any active admission or
shutdown attempt. It writes no audit, event, terminal, or finalizer records and
does not run ordinary finalizers; it runs only callbacks explicitly registered
for no-write recovery cleanup, stops the process-local transient worker graph,
then releases the exact commit guard and backend active-runtime lease. Failure
or cancellation before ownership release is drained and leaves the handoff
retryable. Once backend ownership is irreversibly released, this Runtime becomes
closed even if close reports a warning or caller cancellation arrives; warning
diagnostics remain on idempotent release readback and control-flow interruption
is propagated. A subsequent `Runtime.open()` or `Runtime.aopen()` of the same
target creates a fresh lifecycle and runs authoritative startup recovery before
normal mutation admission resumes.

Audit and event rows are append-only through the Runtime API. Domains that
promise atomic state/evidence publication write them in the same store
transaction as the transition they evidence. Other explicitly multi-phase
publication paths append observability after their main-state commit and report
an evidence failure as a warning or recovery input rather than retracting the
committed state. These are application contracts, not cryptographic tamper
evidence: a database or storage administrator can edit the underlying store,
and v1 does not externally anchor, sign, or hash-chain the log. Deployments that
need evidence against storage administrators must add an external immutable
audit sink or anchoring layer.

The GUI adds a service-level drain around this contract. It stops background
scheduling, rejects new runtime users, waits for tracked request handlers, and
only then calls `Runtime.shutdown()`. A timeout returns failure and reopens the
service lifecycle gate so shutdown can be retried; it does not mark the service
closed or close the store underneath live work.

## Resource Budgets

Resource limits are runtime constraints, not Capabilities. A process may have a
`ResourceBudget` covering tool calls, child processes, runtime seconds, LLM
calls and tokens, context materialization tokens, subprocess wall/CPU/RSS usage,
external filesystem bytes, JSON-RPC bytes, MCP bytes, and Deno syscalls.
Observed consumption is stored as `ResourceUsage` on the process row.

Discrete counters and byte/token quantities must be non-negative integers:
tool/child/LLM call counts, token counts, context counts, Deno syscall counts,
filesystem/JSON-RPC/MCP byte counts, and subprocess peak bytes reject floats
and booleans. Runtime duration and subprocess wall/CPU seconds are continuous,
finite non-negative numbers and may be fractional. This distinction is checked
when budgets/usages are constructed and again at the resource manager boundary.

Every charge applies to the acting process and its parent chain, so a parent can
bound an entire child tree. The complete child-to-ancestor usage update,
reservation consumption, resource event, and audit row commit in one store
transaction; a failure at any point leaves none of that charge published. If an
overage kills a process subtree, terminal Human/Object-task/finalizer callbacks
run only after the store transaction and lock are released. Per killed process,
Object finalization, `PROCESS_EXITED` publication, and terminal notification are
attempted independently across the whole subtree; aggregated cleanup failures
are recorded as `resource.limit_finalize_failed` when the warning sink remains
available. Fork and spawn may request a child budget. Cumulative/reservable
quantities must fit within the parent's remaining amount after sibling
reservations. Peak/non-reservable limits such as subprocess RSS and maximum
child count are checked against every ancestor's ceiling rather than subtracted
as remaining capacity; actual use remains subject to all ancestor limits.
`exec_process` keeps the same pid and does not reset usage or increase budget.
Checkpoint restore replays recorded process rows, including their resource
state, for the restored processes.
Checkpoint-committed images do not store or restore resource budgets or usage;
only the caller that starts the process may set launch-time resource limits.

Provider-backed work whose protected-operation invocation supplies
`reservation_usage` creates a durable reservation for that maximum envelope.
Normal settlement is exactly once and charges the measured value within the
envelope; an unknown provider outcome charges the maximum. On startup, an
active reservation whose linked effect is absent or still `prepared` is
released as certified pre-dispatch, while every reservation linked to a later
effect state is charged maximally. Recovery may take usage over budget and
atomically apply the corresponding resource-limit termination; it does not
discard ambiguous provider consumption to make the budget appear valid. This is
an operation-specific SDK policy, not a guarantee for every provider call.

The LLM protected operation declares `ResourcePolicy.REQUIRED`. Once context
management has produced the exact request, the executor checks the configured
per-call input and aggregate ceilings, then atomically prepares the external
effect and reserves one logical call plus the maximum prompt, completion, and
total-token envelope. Active reservations reduce remaining budget throughout
the process ancestry, so sibling calls cannot oversell a parent. Valid provider
usage settles exactly before the call record and any selected tool; a certified
not-started result releases; exceptions, cancellation, or ambiguous recovery
charge one call and the aggregate maximum with unknown token components left at
zero. Prepared effects release on startup, while dispatched or later effects
settle maximally and idempotently.

If a cumulative token budget exists and billable usage is absent, malformed,
or negative, the completed action maximum-settles and fails closed. An
arithmetic mismatch between reported components and total, or usage outside
any per-call envelope, does the same regardless of the cumulative limit. With
no cumulative token limit, wholly absent usage keeps the compatibility
settlement of one call and zero tokens. These guarantees cover Runtime logical
calls, not SDK retry attempts, exact provider billing, or monetary spend.
Context materialization has both a per-call cap
(`max_context_materialization_tokens`) and a separate cumulative budget
(`max_context_materialization_total_tokens`). The cumulative context token
budget is charged when Object Memory materializes prompt context and is
accounted independently from provider-reported LLM tokens. In the LLM executor,
the default source-only path charges its selected source render. When persistent
enrichment is explicitly enabled, the final rendered
`<config.llm_context.object_name_prefix>:<pid>` Object is charged instead (the
default prefix is `llm_context`), and its source materialization is not charged
a second time. Source-only selection omits Objects that do not fit the remaining
budget; an explicitly enriched final render that still exceeds the applicable
limit fails closed before the model call.

Shell and Deno subprocesses are run through provider-level monitors. On
supported POSIX hosts, the default local Shell provider samples the process tree
to enforce wall time, CPU time, and peak RSS budgets, then records metrics and
audits limit exceedance. The Windows Shell provider rejects execution when a
`SubprocessLimits` profile is supplied instead of silently running an
unaccounted budgeted command. Deno uses its separate POSIX/Windows supervisor
and budget backend and fails closed when those controls cannot be installed; its
host-lifetime supervisor is a POSIX death pipe or Windows `KILL_ON_JOB_CLOSE`
Job Object, so a hard host exit does not orphan untrusted JIT code. In-process
Python primitives are not hard CPU/RSS isolated and generic synchronous tools
are not preempted by a universal wall-time deadline. Elapsed scheduler-quantum
time is charged after completion; direct ToolBroker and ObjectTask calls have
no universal runtime-seconds enforcement. Call-count, byte, and
primitive-specific limits still apply where wired, and a hard deadline exists
only when the primitive/provider implements one explicitly.

## Human Queue

Human interaction is modeled as typed runtime objects rather than untracked
terminal strings. Request and answer payloads may be retained in their
dedicated Human records; effect and audit metadata remain content-free as
described below. For a Human record linked to a Durable Task Run, the Run's
default terminal cleanup or later explicit Host purge replaces readable
request/answer/decision content with hashes while retaining identity, type,
status, timestamps, and audit linkage.

- `ask_human` creates a blocking question.
- `request_permission` requires human write authority, creates a blocking scoped
  policy request with canonical resource, risk, resource scope, lease shape, and
  constraints shown to the human, then returns the final policy decision. Model
  requests cannot ask for broad high-risk grants such as `capability:*`
  privileged rights, `shell:*` execute, or root/global filesystem write such as
  `filesystem:/:*`; workspace write remains a human-approvable scope. The model
  prompt distinguishes active capabilities from the Task Authority manifest's
  requestable ceilings so a missing grant can be requested before an effect is
  attempted.
- `human_output` writes through the HumanObject primitive and provider. It
  commits the `delivered` request marker and structured pending external-effect
  intent before calling the provider. Event, audit, and effect finalization
  follow a successful provider call. A provider exception is finalized as
  unknown when possible and records only `provider_error_type`, never exception
  text; if delivery succeeds but classification/final persistence fails, the
  pending row remains and the call is not retried. Thus output is at-most-once:
  no post-provider failure can leave a replayable request or restore already
  committed one-shot authority.
- After successful delivery, the stored output message is an integrity-bound
  frozen payload for later GUI presentation. The GUI rechecks its captured
  labels and current Sink clearance, but a subsequent mutable LLM-context or
  source-object version no longer hides bytes that were already fixed and
  delivered. A digest mismatch fails closed. Pending questions, approvals, and
  delivery attempts without a successful receipt continue to require live
  source-reference validation.
- Questions, permission context, approval prompts, and output are also
  data-flow checked against `human:<recipient>:<channel>`. A conditional
  release request contains public metadata and hashes only; it never embeds the
  protected payload, and approving ordinary capability cannot elevate an
  untrusted Human Sink above `normal`. The release and protected Human request
  are durably linked: rejecting or ambiguously delivering the release
  terminates the protected request without exposing its payload, while a
  provider-certified not-started result remains retryable and reuses the same
  release after reopen. A resumed answer restores the persisted request labels
  and aggregates them with `normal/untrusted` Human ingress, even when the
  resume call begins with a clean context.
- Per-use approvals can create one-shot capabilities. Side-effectful primitives
  reserve the use before commit, restore it if a pre-commit failure aborts the
  operation, and leave it consumed once the operation crosses its commit or
  provider boundary.

If a primitive or human tool blocks on human approval, the process enters
`waiting_human`. Human requests are terminally decided once: only pending
requests can be approved or rejected. Store schema v5 gives every request a
durable non-negative `revision`. Response, cancellation, claim/delivery, and
retryable delivery-state updates compare the expected revision and status; a
winner increments the revision, so a stale response cannot exploit a
pending-to-pending ABA cycle. The common terminalization kernel commits the
request CAS, permission/Capability side effect when applicable, typed wait-set
change, process revision/state-generation transition, event, and audit in one
transaction. A failure in any required participant rolls the whole decision
back without a partial grant or wake.

The runtime processes a terminal message
and updates that request, but the resulting process transition depends on the
request kind and outcome. Approval may release the matching Human wait;
`request_permission` rejection installs the selected non-allow policy
(`always_deny` or `ask_each_time`) and resumes with a structured `rejected`
decision. An ordinary question/general-request rejection or a rejected
data-release request instead records the normal operation failure and pauses the
process in `PausedProcessWait` until an explicitly permitted resume; either the
Host or a direct parent with the required relationship/authority may clear that
ordinary pause. An ambiguous provider outcome installs the stronger
`HostResumeProcessWait`: neither an ordinary parent resume signal nor a later
Human decision may clear it, and the original provider call is not replayed.
Executor finalization of a rejected exact conditional LLM release likewise
uses the Host-only wait kind.
Terminal queue selection normally claims the oldest pending request in one
serialized critical section. A metadata-only `data_release_approval` is a
prerequisite for delivering its protected parent request and is therefore
selected ahead of older ordinary requests. Blocking provider input/output runs
outside the lock.
Concurrent drains therefore cannot deliver one output twice or install two
automatic permission policies from the same pending request, while process
exit/cancel can still cancel the claim without waiting for user input. A late
answer rechecks the durable pending state and is discarded after cancellation.

Permission decisions must include a JSON boolean `approved` consistent with the
terminal status and one explicit policy: `always_allow`, `always_deny`, or
`ask_each_time`. Approval cannot install `always_deny`; rejection cannot install
`always_allow`. Both an approval and a rejection may select `ask_each_time`;
rejection therefore need not install a durable deny. `allow_once` remains a
separate capability lease/API shape and is not a terminal permission-response
policy. An approved `question` must carry a non-empty string `answer`; values
are not coerced from numbers, objects, or missing fields.

Automatic terminal policy belongs to one host run invocation. It is stored in
an immutable `ContextVar`, copied into scheduler workers, and captured by a JIT
syscall session, so concurrent runs cannot overwrite one another's human,
auto-policy, or answer. Resolving one of several blocking requests normally
leaves the process in `waiting_human` until no blocking request remains. An
ambiguous provider outcome is the exception: its Host-resume gate preempts the
remaining Human-request wait set so those requests cannot make the process
runnable. After process
exit, failure, kill, or terminal cancellation, a post-commit cleanup phase
attempts to cancel all still-pending requests as part of the durable
`terminal_notify` phase. A failure leaves the process-terminal cleanup intent
retryable and may leave diagnostic pending rows, but terminal-state checks still
prevent the process from creating or receiving a new Human decision.

## Process Messages And IPC

Each process has a durable message queue. Messages include:

- sender and recipient pid,
- `kind` such as `normal` or `interrupt`,
- channel,
- correlation id,
- reply target,
- subject and body,
- structured payload,
- delivery and acknowledgement state.

Processes can send messages to themselves, their parent, or direct children.
Receivers use `read_process_messages` for non-blocking reads or
`receive_process_messages` to block until matching messages arrive. Unread-only
selection is the default. `include_acked=True` expands the mailbox-visible
status set to exactly `unread` plus `acked`; an acknowledged match is returned
immediately and therefore also satisfies a blocking receive without registering
a wait. History marked `superseded_by_restore` is excluded from returned pages,
full matching counts, and blocking readiness even when `include_acked=True`.
Filters can match kind, sender, channel, correlation id, reply target, and exact
message ids. An explicit empty message-id filter matches no messages; it never
means "all messages." Read limits bound both returned messages and
acknowledgement. A blocking receive may omit `message_ids` and wait on the other
filters (or on any unread message); if `message_ids` is supplied it must be
non-empty, and an explicitly supplied limit must be positive. Zero-size receive
windows cannot ever produce a message.
An empty blocking read registers its wait atomically with message posting, so a
matching concurrent post either satisfies the read or wakes the registered
process; it cannot disappear between the mailbox query and wait-state update.
Recipient terminal-state recheck, message insertion, event/audit evidence, and
matching waiter wakeup commit in the same store transaction. An evidence sink
failure therefore leaves neither an orphan unread message nor a false wakeup.

Interrupt messages preempt other work until they are acknowledged. The narrow
guidance exceptions are restricted `discover_skills` queries for message/mailbox
help and activation of a Skill that exposes the required message tools, so a
process can reach a reader without bypassing the mailbox gate. A read with
`ack=false` leaves the interrupt unread and therefore still preempting. Normal
messages notify after a tool call and do not block the current action.

ObjectTask completion and waiting notices use the same queue. By default they
arrive on channel `object-task` from sender `object_task:<task_id>`. A process
can block with `receive_process_messages(channel="object-task")` and will be
woken by matching task notifications. If the selected notification process has
already exited, the task records `undelivered_terminal`; task success is not
converted into failure.

ObjectTask owner-watch notices also use this queue. They are addressed to the
runner process on `object-task-owner` by default and can resume a task that is
blocked in `receive_process_messages`; the notice contains event metadata and
object ids, not Object Memory read authority. The same ObjectTask resume hook
also observes ordinary process messages delivered to a waiting runner, and
child-process termination can resume a runner blocked in `wait_child_process`.
ObjectTask runner processes are host-managed and skipped by the LLM scheduler;
message/child condition replay auto-resume is limited to tools with explicitly
safe semantics, currently `receive_process_messages` for message waits and
`wait_child_process` for child-process waits. Human-blocked ObjectTasks resume
through their durable request id after approval or a permission-policy
rejection; that continuation does not blindly replay a provider call whose
outcome may already be unknown.

CLI examples:

```bash
uv run agent-libos --db .agent_libos.sqlite message <pid> "Please inspect the result"
uv run agent-libos --db .agent_libos.sqlite interrupt <pid> "Stop and read this first"
```

## Fork, Spawn, Exec, Wait, Signal

`fork_child_process` creates a direct child with an attenuated parent
`MemoryView`. It can inherit selected capabilities only if the parent already
holds them.

`spawn_child_process` creates a fresh direct child with a new process namespace
and goal-only memory. It does not inherit parent-activated Skills or broad
external authority by default.

`exec_process` replaces the current image, complete tool table, model projection,
and Image-selected Skill state without changing pid.
It never grants the target image's declared required capabilities
automatically. Existing external capabilities are preserved only when explicitly
requested; otherwise exec shrinks external authority.
`ImageBootService.exec` owns the complete publication. Under the shared
publication lock, it preflights the target and atomically captures the exact
rollback snapshot, claims a new exec execution lease, and inserts a durable
`process_exec` publication, binding it to the current Explainable Operation
when one exists. Host exec accepts only a `runnable` process with no ambient
lease; an in-quantum exec must present the exact active `running` execution
token. Any typed wait is rejected before publication because exec has no atomic
way to supersede the child, mailbox, Human, Tool, or Host-resume dependency it
represents.

The service then advances durable phases while applying the process image and
capability shrink, configuring tools, instantiating fresh/package/checkpoint
boot state, and configuring Skills. The success boundary is one final store
transaction: it finalizes deferred capability revocations, commits the exec
epoch back to `runnable`, writes exec event/audit evidence, advances the
publication to `committed`, and reconciles the bound operation to `succeeded`.
The preceding boot phases are intentionally not one SQL transaction, but the
publication and exact admission lease fence competing process writes and make
the outcome recoverable.

On failure, artifact cleanup, exact snapshot restoration, and a durable
`compensation_applied` marker commit together. A later transaction publishes
`rolled_back`, the restored process receipt, the failed operation outcome, and
`image.boot.failed`. If that later terminal transaction fails, startup may
finish terminalization without replaying the already-applied snapshot over
newer work. If compensation itself is incomplete, the Runtime is recovery
fenced; startup claims the publication with a recovery lease, validates the
recorded exec admission token, restores only when the applied marker is absent,
and converges recoverable work to `rolled_back`. A publication that exhausts
recovery or is already `manual` fails closed for operator diagnosis. Normal
commit/rollback terminal receipts bind the image/status plus `revision`,
`execution_generation`, and `state_generation`; recovery additionally compares
the exact claim lease, preventing a stale publication owner from overwriting a
newer process epoch.

`wait_child_process` blocks the parent in `waiting_event`. Child exit wakes the
parent and resumes the original wait action without asking the model for a new
action. Terminal child state, budget release, exit evidence, and parent wakeup
commit atomically, so an evidence failure leaves the child retryable rather than
terminal with a stranded parent.

Signals can pause, resume, cancel, or terminate direct children.
`ProcessSignal.INTERRUPT` is retained as a compatibility enum value but is
rejected as a signal because it has no durable state transition. Send a durable
process message with kind `interrupt` through the message CLI/tool/API when a
recipient needs a preemption notice.
Generic child `pause`/`resume` is also rejected while the target has an active
child/message, Human, or Tool condition-owned wait (`waiting_event`,
`waiting_human`, or `waiting_tool`); only that domain may release its condition.
`cancel` and `terminate` remain terminal control operations and may terminalize
such a waiting child.
Every successful signal commits its target state and signal evidence in one
store transaction. For terminal `cancel`/`terminate`, child-budget release and
a matching parent wake join that transaction. Cancel/terminate then run
Human/ObjectTask terminal notification and Object/host finalization as
independent post-commit phases: failure in one does not skip the other or make
the durable terminal transition retryable. A process paused after rejection of
an exact conditional LLM release also carries a Host-only resume marker:
direct-child pause/resume signals cannot erase or cross it, while a Host
`ProcessManager.resume` clears it deliberately.

Every transition to `exited`, `failed`, or `killed` creates a
`process_terminal_cleanups` intent atomically with the terminal process row.
The receipt moves through `pending`, `running`, `failed`, and `completed`,
records an attempt count and ordered completed phases, and permits only the
claimant with the exact Runtime owner and fresh lease id to mark a phase. The
two current phases are `terminal_notify` (pending Human cancellation plus
ObjectTask notification, with those components attempted independently) and
`process_finalize` (Object/host memory finalization). The phase contract is
idempotent; a completed phase is never rerun, while a partially failed phase may
be invoked again and therefore must tolerate duplicate component calls.

All unfinished phases are attempted even when an earlier phase fails. The
receipt then retains only content-free error observations and the caller
receives `ProcessTerminalCleanupRequired`; a control-flow `BaseException` is
repropagated in an exception group with that typed cleanup requirement after
the independent phases and diagnostics have run. The terminal outcome remains
committed. While the receipt is incomplete, repeating terminal
cancel/terminate/exit on that process retries the cleanup instead of publishing
a second outcome, event, or wake. On reopen, startup holds the lifecycle
recovery lease, reclaims stale cleanup leases, and retries incomplete rows in
configured hard-bounded keyset pages before normal mutation admission opens.
Ordinary recovery failures remain durable and appear in
`recovered_terminal_cleanups`; they do not erase the terminal process state.

## Process Exit

The built-in `process_exit` tool requests the `exited` state and can attach a
final Object Memory result. For an image with cumulative completion review, an
attempt can instead return `status="completion_review_required"` without making
the process terminal; the caller must address the review and retry with its
fresh token and evidence. The trusted Host API
`ProcessManager.exit(..., failed=True)` is the separate path that can mark a
process `failed`; the model-facing tool has no failure-state argument. Callers
should omit `result_oid` and `review_token` until they have real values;
the model boundary treats the literal strings `None` and `null` as omitted.
Durable transcript label discovery follows only `obj_*` Runtime Object
references. A non-Object domain field that happens to be named `result_oid` is
ignored, while a referenced `obj_*` without persisted metadata still fails
closed. A root
process releases its process-owned memory on terminal exit
except for any retained result. A non-root terminal child's process-owned
memory remains available for its direct parent to merge or discard, and is
released if that parent terminates without adopting it. Cleanup follows
explicit Object Memory owner fields, not the object's creator provenance, and
release revokes stale object capabilities.

The terminal row, child-budget release, exit evidence, and parent wake commit
together with the pending terminal-cleanup intent. The `terminal_notify` and
`process_finalize` phases run after that commit because their provider cleanup
cannot be rolled back. Their durable lease, retry, idempotency, and failure
contract is the same one described above; a later cleanup error does not make
the terminal transition uncommitted. Compatibility audit warnings such as
`process.exit_finalize_failed` are best-effort diagnostics, not the
authoritative cleanup state.

When a Deno JIT tool calls `process.exit` or `process.exec`, the syscall records
a deferred lifecycle change. The runtime applies that change only after the JIT
tool returns its normal tool result. An image with cumulative completion review
rejects the direct `process.exit` syscall; it must exit through the built-in
`process_exit` tool so review, freshness, and evidence validation cannot be
bypassed. If one JIT call defers both exec and exit, the runtime checks both the
active and target images before either lifecycle change. A standalone authorized
exec still adopts the target image's completion contract for later quanta; the
source gate is not permanently inherited.
