# Durable Task Runs

Durable Task Runs, introduced in Agent libOS 1.1.0, remain the 1.4.2
Host-supervised unit for long-running
agent work. One `TaskRun` owns one root `AgentProcess` and the child-process
tree created from that root. The Run adds a durable goal, requirement ledger,
safe continuation records, idempotent control commands, and recovery policy;
it does not introduce a declarative DAG language or replace the existing
process, ToolBroker, Human, Capability, external-effect, or ObjectTask
subsystems.

This release targets one trusted Host and one writable Runtime per SQL
database/schema. The store lease and Runtime epoch fence cooperating writers;
Durable Task Runs are not a distributed workflow service.

## Enabling durable payloads

Run metadata and hashes are persisted in store schema v6. Resuming useful work
also requires readable goal, follow-up, transcript, and resume payloads. These
payloads are plaintext at rest, so the Host must explicitly enable their
persistence with `task_runs.plaintext_payloads_enabled: true`. Creation fails
closed while that opt-in is disabled. Enabling it
means that database readers and backup administrators can read retained Task
Run content; it does not enable at-rest encryption.

```yaml
task_runs:
  enabled: true
  plaintext_payloads_enabled: true
```

`enabled` controls admission of new Task Run creation. The manager and startup
recovery remain assembled so an existing v6 store cannot evade reconciliation
by toggling this setting. `enabled` does not imply consent to write readable
payloads. Hosts should keep
`plaintext_payloads_enabled` false unless their data-retention agreement covers
the SQL database and its backups.

Each `TaskRunSpecV1` selects one retention policy:

- `purge_on_terminal` is the default. The Runtime removes readable Run-owned
  goal, follow-up, resume, and persisted completion material during
  `finalizing`. It also hash-reduces the Run tree's retained LLM request,
  response, tool-call, and tool-output bodies; hash-reduces linked terminal
  Human request payloads and decisions; and deletes its pending LLM
  continuation actions and durable process messages automatically bound from a
  Run-member recipient. Callers cannot suppress or forge that recipient-derived
  binding. Because event and audit rows are append-only, a bound message's
  subject, body, and payload are projected there as separate SHA-256 hashes at
  admission time; readable delivery still uses the purgeable
  `process_messages` row and referenced Task Run payload. Ordinary non-Run
  message evidence keeps its existing projection. Human request identity,
  type, status, timestamps, hashes, and audit linkage remain,
  but readable prompt/answer/decision content does not. Linked external-effect
  provider metadata and receipt bodies are reduced to canonical hashes while
  effect identity, transaction state, and causal links remain. A purge failure
  blocks terminalization.
- `permanent` is available only to a Host/admin creator. It keeps readable
  Run payloads instead of performing that terminal cleanup; ordinary evidence
  retention policy may still reduce eligible provider evidence independently.
  After the Run is terminal, a Host/admin may call the revision- and
  command-fenced `Runtime.task_runs.purge_payloads` operation. It applies the
  same cleanup transaction and appends ledger/audit evidence; it does not
  silently rewrite the Run's original retention choice.

With `purge_on_terminal`, the completion result is available only from the
live Runtime's temporary result Object or the completion response. It is not
recoverable after that Runtime closes. Task Run retention does not silently
make ordinary Object Memory values durable. Its terminal LLM/tool redaction is
separate from age-based evidence maintenance and never deletes external-effect
identity, state, receipt hashes, or causal evidence; it does remove readable
provider receipt content.

## Public model

`TaskRunSpecV1` contains the durable goal and title, Image/launch settings,
authority binding, optional absolute `deadline_at`, and retention policy. The
accepted launch-option keys are `capabilities`, `resource_budget`,
`working_directory`, and `llm_profile_id`; unknown keys are rejected before
creation writes anything. Launch options remain metadata after payload purge,
so they may contain only Host-authored identifiers and capability/resource
specifications, never resolved credentials, provider keys, or inline authority
manifests. Authority policy is referenced through `authority_manifest_id`.
The spec version is part of the stored contract. A Run exposes a monotonic `revision`;
all mutations require the caller's expected revision and a stable command id.
A command row binds its kind and canonical request hash. Reusing its id with an
identical request returns the durable, revision-bound command result; reusing it
with a different request is a conflict. For a split-phase command, that stored
result may first be a provisional `pending` receipt and later advance by CAS to
the completed result. Once complete, replay returns that stored result rather
than silently substituting a newer Run projection. For an existing-Run
mutation, `expected_revision` is part of the canonical request identity, so
changing only that fence under the same command id is still a command conflict.

Creation is the exception to the already-existing-Run revision rule: it uses a
stable `client_request_id` (and the Host command receipt created for that
request) so a lost response can be retried without creating a second Run.
Its canonical create identity contains only that request id and the complete
spec; it has no `expected_revision`, and the Python convenience
`create(..., auto_run=True)` does not add `auto_run` to the create identity.
Instead, auto-run is a separate deterministic `<client_request_id>:run` command
whose `expected_revision` is reconstructed from the immutable create receipt.
An exact create replay may therefore request that separate run while the Run is
still at its queued create projection. If the Run has advanced and no matching
run receipt exists, the Runtime raises a conflict rather than binding the old
intent to the current revision or dispatching new work. A matching pending run
receipt follows the local-only replay rules below. The CLI and private HTTP API
keep queued creation and explicit run mutation separate; the HTTP create route
rejects create-time auto-run.

### Crash-safe command settlement

`run`, `resume`, `cancel`, command-associated deadline handling, interrupting
`follow_up`, and authoritative effect-receipt recovery use a durable admission
receipt when their local phase can outlive the first database transaction or
response. The admission transaction stores `settlement_state=pending` together
with the relevant revision or pause/cancel generation fence before work can be
mistaken for a fresh command. The interrupt receipt is atomic with its
requirement, Run-bound message, and new pause generation. For authoritative
effect recovery, the pending command and the verifier-normalized external-effect
settlement commit in the same transaction; the command result stores only the
public summary and content-free fences, not another copy of the submitted
receipt body. This is not a promise that verifier code runs at most once before
commit: if that whole transaction rolls back, no durable pending receipt or
settlement exists and a later fresh attempt may verify again.

Every persisted command result is a canonical strict version-1 object. The
complete public `TaskRunSummary` mapping must bind the command row's Run and
`result_revision`, and each command/settlement variant accepts its exact key
set only. Missing or extra keys, a noncanonical or partial summary, pending-only
fields on a completed variant, a Boolean used as an integer, or an integer
outside the signed BIGINT range is invalid. The configured
`task_runs.command_result_max_bytes` cap is applied to the raw UTF-8
`result_json` before JSON decoding and again to canonical writes, so whitespace
or another syntactically valid oversized representation cannot evade the
bound. For `recover`, the canonical request's server-issued `option_id` selects
the terminalize, authoritative-effect, or linked-Run result variant; stored
result keys cannot reclassify that request after admission.

Every split local-control receipt—`run`, `resume`, `cancel`, deadline,
terminalize, and interrupting follow-up—retains
`admission_ledger_seq`, `admission_ledger_item_id`, and
`admission_evidence_sha256` in both its pending and completed form. They
reference the append-only `STATUS_TRANSITION` ledger item written in the same
admission transaction. The item, its metadata, and the canonical evidence
digest bind the Run, command id/kind, canonical request hash, from/to status,
and the semantic fence: `run` uses `admission_revision`, `resume` uses
`pause_generation`, cancel/deadline/terminalize use `cancel_generation`, and an
interrupt binds its pause/cancel generations, prior status, and
`interrupt_provenance_sha256`. Every terminal or superseded early-return path
validates the exact receipt variant and this immutable ledger evidence before
returning the stored summary; terminal state is not permission to accept a
damaged receipt.

The interrupt's `interrupt_provenance_sha256` binds its positive
`admission_runtime_epoch` and the complete canonical `resume_fences` list.
While settlement is pending, the raw epoch and fences remain so local recovery
can revalidate them. The completed receipt removes those raw pending-only
fields but retains the provenance digest and admission-ledger binding, allowing
later exact replay to validate the historical admission without retaining a
second mutable fence projection.

All Task Run command inserts and result CAS updates acquire and verify the
global `task_run_runtime_epoch` counter-row lock, including creation and repair
of the missing outer receipt in the linked-recovery gap. A superseded Runtime
therefore writes no new command row and cannot complete an old result, even
when the Run is already terminal.

After local convergence, a separate result-revision CAS changes the receipt to
`complete`. A Host crash, lost response, or failure of that final result update
does not prove that the admitted work failed and does not roll back already
committed Run/effect state: the provisional receipt remains the replay fence.
An exact retry may finish only the following local settlement:

- `run` checks for a still-live admitted scope and returns the provisional
  result while that scope still owns settlement. After it drains, replay only
  projects already durable process/effect state; it does not consume another
  scheduler quantum.
- `resume`, `cancel`, and a deadline command apply only their persisted
  generation-fenced local process transition and projection. A newer control
  generation wins.
- an interrupting follow-up drains the prior admitted scope and may supersede
  only an integrity-bound `validated_action` left at local `dispatching` when no
  external effect changed after its safe point. It restores local process state
  from the persisted interrupt generation, message, and per-process fences.
- an effect-receipt replay validates the already committed ExternalEffect truth
  and its Run, cancellation generation, admission Runtime epoch,
  `settlement_transition_seq`, and `settlement_audit_record_id`. The transition
  must be the exact append-only finalized effect transition, and the audit row
  must be the matching `external_effect.recovery_settled` decision from
  `host_verified_receipt`, including its previous/settled state, provider
  outcome, transition sequence, and receipt digest. This chain remains usable
  after provider metadata/receipt bodies are purged and does not call the
  provider receipt verifier a second time.
- a linked-Run recovery uses deterministic nested rerun and target-create
  identities. If the nested rerun committed but the outer recover receipt was
  lost, an exact outer retry validates the nested receipt's hidden parent
  command/request-hash binding, immutable source and target summaries, target
  create receipt and Run row, and the unique `rerun_of` link before inserting
  the missing outer receipt with the same target result. It does not invoke a
  fresh rerun or create a second Run.

None of these receipt replays redispatches an LLM provider, tool, or external
effect. Startup may independently converge the Run's durable process/status
projection, but it normally leaves a provisional command result for an exact
client replay. For a given Run, its only automatic command settlement is at
most one bounded, well-formed pending interrupt when that Run was nonterminal
in the recovery page and its pause/cancel generations exactly match the current
Run. Before applying that interrupt, recovery validates the local
payload and binding, locally commits any already-staged complete provider
outcome, validates a durable wait, and applies the ordinary authority, effect,
and ObjectTask gates. A persisted cancellation or expired-deadline intent
outranks an older interrupt and may converge the Run without completing that
older command receipt.

An eligible interrupt resumes only a process carrying the Store-only typed
`StaleExecutionProcessWait` receipt. `status_message` may still project
`stale_execution_recovery` for older clients, but Runtime control never reads
or parses that compatibility string. The typed receipt records the PID, the
pre-takeover owner and lease identities as canonical SHA-256 values, the
recovering Runtime owner-id SHA-256 (distinct from the prior owner), and the
prior/recovered execution and recovered process-state generations. The
generic receipt permits absent prior hashes, but Durable Run auto-resume
requires both prior-owner and prior-lease hashes to be present. The
recovering-owner digest is a non-secret identity hash, not a cryptographic
signature or evidence independent of the trusted RuntimeStore/database
administrator boundary. Pre-takeover raw owner and lease tokens are cleared
atomically and never enter the stale wait, TaskRun summary/ledger, or error
projection. This narrow guarantee does not claim that a generic process API
omits the currently active process execution-owner or lease fields.

A later Runtime may accept the historical recovering-owner hash written by an
earlier exclusive Store recovery so a second or later crash during local
interrupt settlement remains idempotent. That historical hash alone grants no
resume authority. Control also joins the receipt to the pending interrupt's
positive admission Runtime epoch and exact
`(pid, admission_state_generation, admission_execution_generation)` fence. The
bounded fence list contains only Run members that were `running` at interrupt
admission, is sorted by PID, and contains each PID exactly once. Repeating a PID
with the same or different generation values makes the receipt malformed; a
decoder may not collapse such entries through a map overwrite. The
current Runtime epoch must be later than admission, and
`process.task_run_epoch == run.runtime_epoch == current Runtime epoch`. The
receipt's recovered state generation must equal the process generation and the
admitted state generation plus one; its prior execution generation must equal
the admitted execution generation, and its recovered/process execution
generation must be that value plus one. No execution owner or lease may remain.
The complete authoritative resume point must bind the same PID and Run;
`0 < point.task_run_epoch <= process.task_run_epoch`, its process revision may
not be from the future, and its static integrity,
absence of a pending action, Run binding, and current Image/tool/provider
binding must all validate. The generic Store recovery path does not read or
copy TaskRun resume payloads, epochs, safe-point integrity, or binding hashes
into the wait receipt.

If another crash occurs after only part of the interrupt settles, every fenced
member is classified exactly. A still-paused member must retain the valid typed
receipt above. An already resumed `runnable` member must be at admitted
state-generation plus two and execution-generation plus one, with no wait,
outcome, or pending action. A terminal member must be at admitted
state-generation plus one and execution-generation plus one, with an outcome
and no pending action. The only provider-free pre-resume settlements accepted
at admitted state-generation plus one and unchanged execution generation are
`runnable` with no pending action, or a supported event/Human/tool wait with a
typed wait and its pending action. Every class requires no live owner/lease and
the same identity-, integrity-, and binding-valid resume point; anything else
fails closed.

An independent Host pause, a legacy string-only pause, an incomplete lease
receipt, an invalid generation or recovering-owner hash, or a preexisting
process pause remains paused and moves the Run to `needs_attention` when it was
expected to be resumable. Checkpoint restore and checkpoint fork both
deliberately degrade the non-transferable receipt to an ordinary
`PausedProcessWait(reason_oid=None)` and clear the
`stale_execution_recovery` compatibility `status_message` instead of carrying
its PID/lease provenance across a concurrency identity.
A queued Run remains queued and dispatches nothing. After interrupt
restoration, the normal resume-point blockers and process/finalization
projection still run, so `finalizing` can finish in the same reopen but no
safety gate is bypassed. Startup then CASes the command result without
requiring a manual Resume. A stale generation cannot regain control.

If the Run reaches a complete terminal state before that command-result CAS,
terminal Runs are outside the startup recovery scan. The provisional receipt
remains for an exact client retry, which records the already terminal result
locally and dispatches no provider or tool work.

Startup likewise does not synthesize a missing outer linked-recovery receipt.
Only an exact retry of that original recover request may reconstruct it from
the bounded local evidence chain above. A different expected revision, missing
or malformed nested/target evidence, or an ambiguous link fails closed and
writes no replacement receipt; current source state is never used to rebind the
old command.

For an otherwise eligible recoverable Run, the startup interrupt scan is separately
bounded by `task_runs.recovery_page_hard_limit`; an oversized scan, multiple
pending interrupts for one current generation, or a malformed pending-interrupt
receipt causes startup to project `needs_attention` rather than guess or
dispatch. An exact replay likewise rejects a missing, corrupt, or
fence-mismatched command, and an effect-receipt replay whose stored effect truth
is incomplete leaves the Run in `needs_attention`. Validation failure is never
treated as permission to issue a replacement provider or tool call.

`TaskRunStatus` values are:

| Status | Meaning |
| --- | --- |
| `queued` | Created durably; no quantum is being run. |
| `running` | The active process tree may perform work. |
| `waiting_human` | A durable Human request is the current blocker. |
| `waiting_process` | The Run is waiting for a child-process outcome. |
| `waiting_message` | The Run is waiting for a matching durable message. |
| `waiting_tool` | A typed tool/operation wait is active. |
| `paused` | New dispatch is disabled by a persisted Host pause. |
| `cancelling` | Cancellation is persisted and the process tree is converging. |
| `finalizing` | Effects and retention cleanup must settle before terminal status. |
| `needs_attention` | Automatic continuation is unsafe; Host reconciliation is required. |
| `succeeded` | The process tree, requirements, effects, and cleanup all settled successfully. |
| `failed` | Execution ended unsuccessfully after safe settlement. |
| `cancelled` | Cancellation completed without an unresolved external effect. |

`TaskRunSummary` includes the Run id and revision, root and active PIDs, step
count, blockers, result reference when available, timestamps, and
server-computed `allowed_actions`. Its content-free `payloads_purged` flag is
true after either default terminal cleanup or an explicit purge of a permanent
Run. Clients must use `allowed_actions` rather than infer whether Resume,
Retry, Cancel, or recovery is safe from the status string alone, and must use
`payloads_purged` rather than the retention policy to decide whether rerun
needs a replacement goal.

The initial goal is the first required requirement. `follow_up` atomically
appends another immutable requirement and sends a durable message to the root
process. Requirements move through `pending`, `in_progress`, `satisfied`, and
`blocked`; only a Host/admin action may mark one `waived`. A Run cannot report
success while a required requirement remains unresolved.

Prompt reconstruction moves a pending requirement only to `in_progress`; mere
visibility is never completion evidence. Satisfaction commits atomically with
the complete local transcript for a validated root `process_exit`, bound to the
exact requirement ids and payload hashes frozen in that model request. A Host
lifecycle exit, an unbound/replayed call record, or a root exit whose result
cannot be staged leaves the requirement unresolved and the Run enters
`needs_attention` instead of reporting success. Once prompt admission sets a
requirement's `started_at`, later `satisfied`, `blocked`, or `waived` projection
preserves that original timestamp, including across Runtime reopen.

Some OpenAI-compatible providers reversibly encode the nested
`completion_evidence` object as a JSON string. The Process Tool schema and the
TaskRun settlement projection accept that bounded representation consistently,
while the durable action manifest and its hash retain the provider's exact raw
value. Malformed evidence is rejected before terminal exit, and decoded
evidence still passes the same requirement and causal Operation-receipt checks;
stringification does not weaken completion authority.

For an Image with cumulative completion review, the root Process goal Object is
only a non-secret TaskRun marker. The review resolves the integrity-checked
local goal and follow-up payloads for internal requirement/tool-hint matching,
but returns only requirement ids, hashes, statuses, and references to the
model. Those identities are part of the review token. Every outstanding
requirement id must have at least one independent structured acceptance check,
and one check may bind at most one outstanding requirement. All checks bound to
a requirement must report `completed` before it can map to `satisfied`;
model-reported `blocked` or `cancelled` maps to `blocked` and therefore
`needs_attention`. It never acts as the Host-only `waived` transition.

A `completed` check must name an evidence tool for which the Runtime can resolve
and persist a causally eligible, terminal-successful tool Operation receipt for
that exact requirement. Eligibility is derived from durable evidence, not from
the model's assertion: the `tool_call` Operation must be a child of the exact
terminal-successful root `llm_request` Operation; that LLM Operation must have
exactly one successful `llm_call` invocation link; and the linked durable call
must have frozen the requirement's current id, ordinal, and payload hash before
provider dispatch. A receipt from an older model turn therefore cannot satisfy
a follow-up absent from that turn's frozen binding, even if the tool happened to
dispatch after the follow-up was appended. A tool dispatched directly by the
Host has no qualifying LLM ancestry and cannot be reused as completion proof.

The completion review token also binds a canonical SHA-256 projection of each
acknowledged Human message's content-bearing identity, including its subject,
body, and payload. A message-body change therefore invalidates an earlier token;
message ids and ACK status alone are insufficient. Completion requirement,
Operation, LLM/tool evidence, and expanded requirement/receipt projections are
all fail-closed at `task_runs.recovery_page_hard_limit`. The Runtime rejects an
unbounded or inconsistent projection rather than truncating it into partial
completion evidence.

When a follow-up requests interruption, the requirement and its Run-bound
message commit before the interrupt can affect the root. The next local prompt
rebuild reads the requirement payload rather than trusting message preview
text, so a lost client response cannot turn the follow-up into an untracked
instruction. The initial requirement plus all follow-ups may not exceed
`task_runs.recovery_page_hard_limit`. A new follow-up beyond that bound, or one
addressed to an already terminal root process, is rejected before writing its
payload, requirement, Run revision, ledger/command receipt, or process message;
an exact replay of a previously accepted command still returns its recorded
idempotent result.

The paged ledger records requirement changes, process actions, model turns,
tool calls, waits, checkpoints, external effects, and state transitions. Ledger
items link to the authoritative subsystem evidence; they are a projection, not
a second source of truth.

## Execution and restart recovery

Creating a Run atomically writes its spec payload, root process, initial
requirement, current projection, Operation, event, and audit evidence. Starting
it is a separate command. `run_until_blocked` consumes scheduler quanta until
the Run is terminal or reaches a typed wait, pause, deadline, or
`needs_attention` boundary. `max_quanta` (or the Runtime's configured default)
bounds that dispatch only when it is non-null; a null value is not a hard
quantum limit.

After a complete model action and all of its paired tool results have committed, the
Runtime may publish a local safe resume point. The resume point binds the
process context generation, Image, tool table, provider identity, authority,
payload hashes, and Runtime epoch. Reopen reconstructs prompts from locally
persisted goal, requirements, transcript, and compacted context. The current
full-snapshot AgentProcess executor always sends that complete locally rebuilt
snapshot and never sends a provider `previous_response_id`, even when the
low-level client setting is enabled. It leaves
`previous_response_id_used=false` in the validated action/outcome manifest and
rejects a durable resume record that claims otherwise. Provider response ids
may remain in bounded call-observability records; they are not a Task Run
optimization, resume source, or replay authority. The low-level `LLMClient`
chaining support for explicitly supplied delta-style calls is outside the
AgentProcess and Task Run recovery contract.

Durable Task Runs v1 certifies one dynamic-binding transition: a validated
`activate_skill` call that is the sole action in a non-parallel model response.
The Image/tool/provider request binding is hashed into the local LLM request
record and checked immediately before and after the Provider call; drift rejects
the response before an action safe point or tool dispatch can be created.
Dispatch is pinned to the exact `tool_id` in the validated pre-action binding;
a same-name rebind cannot substitute a different tool. The committed activation
result and exact post-action bindings are then integrity-bound to the resume
point. Any other action that changes the Image, tool, provider, or loaded-Skill
binding is unsupported in v1: the Run fails closed into `needs_attention`
before continuation or reopen dispatch, and the action is never replayed
automatically.

Startup validates Task Run payload integrity before it admits execution. It
then reconciles prepared operations and provider effects, stale capability and
resource reservations, and incomplete publications. After publication
recovery it fences stale operation/execution claims, reconciles ObjectTasks and
terminal-process cleanup, and only then projects Task Runs before scheduling
can begin. Automatic continuation is limited to:

- a complete safe resume point with no later unsettled effect;
- a protected provider operation authoritatively classified as not dispatched;
- a provider success whose complete response is already durable and needs only
  local settlement.

A missing/corrupt payload, binding drift, non-replayable pending action,
unsettled dispatched/unknown effect, or active ObjectTask abandoned during
reopen moves the Run to `needs_attention`. No downstream tool or model dispatch
occurs from that state. The Host may choose only the recovery actions computed
from durable evidence, such as provider reconciliation, recording an
authoritative receipt, confirming certified non-dispatch, stopping the old
Run's execution without settling an ambiguous effect, or creating a linked
rerun. Stopping execution leaves that effect and the Run in `needs_attention`;
it does not manufacture a terminal status. A model or ordinary user cannot
simply label an unknown effect safe and retry it.

Effect settlement requires both `effect_state=finalized` and a transaction
state of `committed`, `failed`, or `compensated`. In particular,
`effect_state=finalized` with `transaction_state=unknown` remains unsettled and
continues to block dispatch, cancellation convergence, and terminalization.

When the root process is already terminal, the server does not advertise
`follow_up` or `resume`: neither can deliver more work to that process. A Host
`terminate_run` recovery persists a new cancellation generation and may then
converge the old Run to `cancelled` once processes, effects, reservations, and
cleanup are settled. A dispatched or unknown effect still keeps the Run in
`needs_attention`; terminal recovery never treats stopping the process as
effect settlement.

Pause and cancellation are forward-only intents. Pause first advances the Run
generation and stops new dispatch, then lets an in-flight protected operation
reach a durable boundary. Cancellation and an expired absolute deadline stop
new claims and child spawns, settle in-flight effects, and terminate descendants
before the root. An ambiguous effect produces `needs_attention`, never a false
`cancelled` result. Wall-clock deadlines continue while the Run waits for a
Human, process, message, or tool.

A terminal root or process tree is not by itself a terminal Run. Any scheduler
quantum or external dispatch admitted before that process transition must first
finish its local settlement and durably land its paired result and cleanup
evidence. Only after those scopes drain may projection enter `finalizing` and
then a terminal Run status; ambiguous or missing settlement instead requires
attention.

## Python Host API

`Runtime.task_runs` is a Host control plane. It exposes creation, inspection,
paged listing, optionally quantum-bounded execution, passive waiting,
pause/resume, cancellation,
follow-ups, evidence-constrained recovery, whole-Run rerun operations, and an
audited `purge_payloads` operation for a terminal `permanent` Run. The explicit
purge is a Python Host/admin surface in 1.4.2; it is not offered to ordinary
CLI or GUI users.
Rerun creates a new Run id and links it to the prior Run; it never rewinds the
old ledger. After either `purge_on_terminal` cleanup or an explicit Host purge
has removed the old goal, rerun requires an explicit replacement `goal` in
`spec_overrides`; only a still-retained `permanent` Run may reuse its durable
goal. Manager objects must not be handed to model code.

The common interchange types are exported from `agent_libos`. Exact signatures
and return types remain authoritative in the installed package. Every mutation
accepts `expected_revision` and `command_id`; callers should generate a command
id once and reuse it only when retrying the exact same request. Creation uses
its stable client request id instead of an expected revision.

## CLI

The `task-run` command group mirrors the ordinary Host controls (the
Host/admin-only explicit payload purge remains Python-only in 1.4.2):

```text
task-run start
task-run get <run_id>
task-run list
task-run wait <run_id>
task-run recovery-options <run_id>
task-run pause <run_id>
task-run resume <run_id>
task-run cancel <run_id>
task-run follow-up <run_id>
task-run recover <run_id>
task-run rerun <run_id>
```

`start` creates a queued Run by default; `--client-request-id` supplies the
stable create identity and omission generates one for that invocation. An
explicit `--run` performs execution in that CLI process until a typed boundary;
`--max-quanta` adds a quantum-count cap. `wait` observes state and does not
create a background daemon or dispatch scheduler quanta, provider calls, or
tools. An observing read may still persist safe deadline cancellation,
settlement projection, and terminal-retention housekeeping. Waiting and
`needs_attention` are valid structured results, not internal CLI failures.
`recovery-options` returns the server-derived,
evidence-bound choices that may be supplied to `recover`; it does not mutate
the Run. Cancel and recovery require explicit confirmation.
For any terminal Run whose goal was purged, including an explicitly purged
`permanent` Run, `rerun` needs
`--spec-overrides-json` containing a replacement `goal`.

## GUI and local HTTP API

The same-build GUI exposes a Task Runs list/detail view with requirements,
blockers, retention, allowed actions, Human requests, and paged ledger links.
The older Tasks panel is labeled Object tasks because its single-tool,
runtime-local execution contract is different. A `needs_attention` Run with an
unknown effect has no ordinary Retry or Resume button.

The private local API provides `/api/task-runs` collection/detail routes,
paged ledger and Human-request reads, and run, pause, resume, cancel, follow-up,
recover, and rerun mutations. Collection, ledger, and Human pages accept an
opaque `cursor` and return `next_cursor`; clients must not parse it. There is no
separate Task Run requirements or wait HTTP route in 1.4.2: the detail response
embeds a bounded requirements page selected with `requirements_limit` and
`requirements_cursor`, requirement changes are also ledger items, and waiting
is ordinary state observation. Cancel/recover require the same explicit
confirmation envelope used by other high-risk GUI actions. Mutation conflicts
return HTTP 409 with a stable error code. The response includes the content-free
`command_admitted` fact only when a Store lookup proves it, and includes
`current_summary` when the server's exact Run read succeeds. The client still
performs an exact detail read: a snapshot or SSE update is not command-specific
evidence. Only a `task_run_revision_conflict` for which the server proves
`command_admitted=false` and the authoritative detail read reconciles
successfully may retire the old request intent. A command conflict, an admitted
command, or an indeterminate admission result keeps the original request body
and command id for exact retry and fails closed; the client never silently
rebases that command onto a newer revision. `POST /api/task-runs` is
queued-only; execution always uses the separate, revision-fenced `/run`
mutation, where `max_quanta` is enforced. Rerun also carries a stable
`client_request_id` for the linked create operation in addition to its
source-Run revision and command id.

The Runtime/SDK command result remains the immutable historical receipt
described above. The private HTTP transport has a separate presentation rule:
after every successful mutation, including create and linked rerun/recovery, it
performs an exact `manager.get(result_run_id)` and returns and publishes the
latest summary observed by that read. For a linked mutation this is the target
or newly created Run, not the source Run. A later concurrent update may still
advance its revision. Follow-up intent is retired only by the HTTP 200 for that
exact command; equal requirement content or an SSE update is not proof that the
same command committed. The create-then-run client retains both the create
request id and the run command id/revision until the run response is
non-ambiguous.

SSE publishes only a redacted `task_run.updated` summary. Clients discard a
revision older than or equal to the one already rendered and refetch the HTTP
snapshot when the stream reports invalidation. This is a renderer/server
same-build contract, not a separately supported public REST service.

## Boundaries with existing subsystems

- An ObjectTask may run under a Task Run, but ObjectTask arguments and live
  execution remain runtime-local. On reopen the ObjectTask is abandoned and
  the supervising Run blocks; it is never replayed automatically.
- Checkpoints and Images do not package Task Run payloads or ledger history.
  Restore refuses a scope that intersects an active Run; fork does not clone
  Run ownership; restore never erases or rewinds Run/effect evidence.
- Capabilities, Task Authority, Human approval, data-flow labels, ToolBroker,
  provider policy, and protected-operation evidence continue to apply to every
  action. Task Run ownership grants none of them.
- Every logical LLM call atomically prepares its external effect and maximum
  call/token reservation before Provider dispatch, as in ordinary scheduler
  execution. Human/data-flow waits hold no reservation; resume re-admits the
  exact request under current Host ceilings. Known usage settles exactly,
  certified non-start releases, and ambiguous outcomes charge the aggregate
  maximum before any model-selected tool runs. This is not an exact physical
  request, Provider billing, currency, or monetary-spend cap.
- A completed Runtime LLM record may include a bounded Provider attempt trace.
  It remains terminal-call observability, not a new Task Run resume point or
  replay receipt. Default terminal purge reduces its readable reasoning/output
  together with the rest of the LLM payload; summary-only GUI events cannot
  retain a second plaintext copy.

See [Runtime Model](runtime_model.md), [Storage](storage.md),
[Object Memory](object_memory.md), [Checkpoints](checkpoints.md),
[Providers](providers.md), [CLI](cli.md), [GUI](gui.md), and
[Threat Model](threat_model.md) for subsystem-specific contracts.
