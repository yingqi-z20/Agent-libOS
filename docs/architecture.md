# Architecture

Agent libOS is structured around one boundary: model-visible and
self-evolving action surfaces are not resource authority. A process may see a
tool schema, activate a Skill, register a JIT tool, register or exec an image,
fork a child, restore from a checkpoint, or inspect a remote endpoint, but
protected effects are authorized only when a primitive runs under that process
id.

## Layer Model

```text
Agent personality / application
  -> Skills and tools layer
     - model-facing actions
     - prompt instructions
     - tool schemas
     - Deno/TypeScript JIT candidates
  -> Agent libOS runtime
     - trusted startup module loader
     - scheduler
     - process manager
     - Object Memory manager
     - ToolBroker
     - Skill manager
     - HumanObject manager
     - primitive managers
     - data-flow manager and Host Sink trust registry
     - capability manager
     - resource manager
     - event bus
     - checkpoint manager
     - operation/explain manager
     - protected-operation SDK and evidence retention
     - audit manager
  -> Resource Provider Substrate
     - filesystem provider
     - pinned system-Git provider
     - clock provider
     - shell provider
     - human provider
     - JSON-RPC over HTTP provider
     - MCP client provider
  -> host backend
     - local workspace filesystem
     - host clock
     - subprocess backend
     - terminal or UI human I/O
     - pre-registered remote JSON-RPC endpoints
     - pre-registered MCP servers
     - future container, WASM, or service providers
```

The Skills and tools layer exists for LLM ergonomics and self-evolution. It
presents stable action names, schemas, summaries, workflow instructions, and
process-local JIT candidates. It does not own external authority.

Image registration and `exec` are also self-evolution mechanisms. They can
change a process prompt, prompt composition mode, default tool table, default
Skills, and lifecycle shape, but image visibility and target-image metadata do
not grant resource capabilities or impose resource budgets. Launch-time callers
provide a durable [`TaskAuthorityManifest`](task_authority_manifest.md) that
owns capability, effect-class, approval-policy, and resource-budget ceilings.
Image `required_capabilities` are unmet-requirement declarations, not grants.
Image packages may seed a
private per-process workspace and process-local JIT tools, but those are scoped
to the booted process and do not expose the package source directory.
Default tool tables are exact image declarations: the runtime does not add
generic lifecycle or Object Memory tools unless the image lists them.
Images may opt into lazy model projection. The complete image tool table stays
callable and capability-enforced, while a separate durable model projection
initially exposes only discovery/core tools. Activating a group changes schemas
and visibility but never capabilities.

Startup Runtime Modules are different from Skills. A module is trusted Python
host code loaded before `Runtime.open()` returns. Modules extend the runtime
composition root by registering tools, images, syscalls, provider hooks, and
startup hooks. They may also declare buffered durable Object-release handlers;
those handlers are installed before startup publication recovery so persisted
cleanup intents can be resumed without running general startup hooks early.
Because modules run in the host interpreter, they are part of
the runtime trusted computing base and are gated by manifest hash trust rather
than by process capabilities.

The runtime owns agent-level semantics: process identity, capability checks,
source labels, Host Sink clearance, approval, event emission, audit, process
wakeups, checkpointing, and durable metadata. `DataFlowManager` resolves
runtime-owned Object source references, performs early egress clearance, issues
exact conditional releases through metadata-only Human requests, and appends
payload-free decisions. See [Data Flow](data_flow.md).

Explainable Operations overlays these managers without replacing their source
records. A `ContextVar` carries the active causal operation through async calls
and the runtime-owned blocking-work bridges; protected public boundaries create typed parent/child
rows, while audit, event, capability reservation, Human request, provider
effect, resource charge, LLM call, and context materialization code attaches
explicit evidence ids. Query code follows only those links. It does not infer
causality from pid and time proximity. See
[explainable_operations.md](explainable_operations.md).

The Resource Provider Substrate owns concrete host calls. A provider is a
backend, not a security bypass. Replacing the filesystem or shell provider must
not change tool schemas or skip primitive authorization. The concrete provider
inventory, containment guarantees, extension checklist, and environment
limitations are documented in [Providers](providers.md).

LLM, filesystem, Git, clock, shell, Human, JSON-RPC, MCP, and PTY provider boundaries use
the public [`agent_libos.sdk`](protected_operation_sdk.md) contract. The SDK is
the only provider-facing lifecycle for finite capability reservation, effect
prepare/dispatch/finalize, classification fallback, event/audit evidence, and
resource settlement. Egress/bidirectional contracts must also declare a
concrete `DataSink`, trusted `DataFlowContext`, canonical payload descriptor,
and operation. The SDK revalidates those descriptors in the same transaction
that reserves ordinary/release capabilities and prepares the effect intent.
Low-level effect-ledger functions remain Runtime-internal for the SDK and
startup reconciliation.

Providers are normally the source of truth for successful external-effect
rollback classification, and contracts require a provider classifier by
default. A trusted primitive or executor may instead supply an explicit
`classification_override` when it has stronger phase-specific knowledge that a
generic provider classifier cannot express—for example, the fixed
irreversible/information-flow result of a completed LLM request or a validated
pre-call/state-only result. The override is runtime code, not model- or
provider-supplied metadata, and the SDK still applies the completed-phase
state-mutation/information-flow floor so it cannot under-classify work already
observed. The runtime persists the selected classification for checkpoint
reports, but v1 does not apply external compensation. Filesystem mutation,
clock, shell, and PTY spawn paths use explicit finite-use reservations around
the provider boundary.
A filesystem, Git, clock, shell, human-output/terminal-I/O, PTY, JSON-RPC, or live
MCP primitive also persists a conservative `unknown` external-effect intent
immediately before entering that boundary. On a classified success or
ambiguous failure, the store conditionally updates that same `effect_id` from
`pending` to `finalized`, matching pid, provider, operation, and target. An
already finalized, abandoned, or mismatched intent cannot be settled again. If
capability commit, event/audit, classification, or final persistence fails after
the provider may have run, the pending `unknown` row remains durable and is
visible to checkpoint, Explain, and benchmark consumers instead of creating a false
absence of evidence.

Each intent also records a canonical argument hash, idempotency key, and a
transaction state (`prepared`, `authorized`, `approved`, `dispatched`,
`committed`, `failed`, `unknown`, or `compensated`); not every operation uses
every intermediate state. A unique process/idempotency-key index blocks
duplicate dispatch. Startup may call a provider reconciliation hook to query an
existing receipt; it never replays the effect. Providers without that hook
leave the transaction `unknown`.

A provider may raise `ProviderEffectNotStarted` only when it can certify that
its selected call did not begin. The primitive abandons the pending intent only
when every completed earlier provider phase has `state_mutation=False`,
`information_flow=False`, and `commits_authority=False`. In that case it
restores an exact reservation when one was reserved; filesystem/clock/shell and
PTY spawn perform restoration and abandonment in one store transaction. The
default `commits_authority=True` therefore closes the restoration floor even
for a successful phase whose other two flags are false. If an earlier
filesystem `state()` or MCP live-tool validation already returned information,
the main mutation/call being not-started still finalizes the observed partial
effect instead of erasing the intent.

Human terminal reads and automatic writes persist only request/purpose and
length/hash observations; raw prompts, answers, and provider exception text do
not enter effect or audit metadata. JSON-RPC and non-local HTTP MCP persist the
intent and reserve deduplicated finite authority before DNS. Once DNS observes
the host, a later transport PENS cannot erase that information flow or restore
the use. Endpoint/server registry item authority is checked before metadata
lookup, and registry row, stale-grant, event, and audit mutations are atomic.

Clock sleep/asleep similarly starts its intent before the first `monotonic()`
measurement. Only a not-started result from that first observation may restore
and abandon; every later sleep, cancellation, or measurement failure consumes
the use and finalizes unknown. The successful elapsed-time result is also an
information flow.

Once the first provider phase that may produce an effect is crossed, the
reservation is committed. Timeout,
cancellation, resource-limit, ordinary provider exception, or post-effect
classifier failure cannot prove non-execution, so authority stays consumed and
the primitive records or retains a conservative `unknown` effect. This makes a
failed return value distinct from a proven absence of external effects.

The PTY Runtime Module applies this pending-to-finalized protocol to spawn,
write, resize, and close. Cleanup after a spawned session fails to publish its
Object is containment, not evidence that spawn never occurred; classifier
absence or failure after a PTY operation finalizes an `unknown` fallback, while
post-provider sink failure leaves the pending row visible.

## Composition Root And Internal Dependencies

`agent_libos.runtime.builder.RuntimeBuilder` is the composition root. It opens
the store, constructs all acyclic dependencies in dependency order, and uses
named late bindings only for explicit construction cycles: Data Flow/Human,
Resource/Process and Object Task notifications, lifecycle participants, and
the checkpoint module-catalog/image-registry pair. Protected-operation recovery
and Process/Image Boot use their named recovery/hook registries for the same
reason. The builder then loads trusted extensions and cleans up a partially
assembled host on failure. Async hosts must use `await Runtime.aopen()`; failed
assembly cleanup runs on that caller loop and is shielded until teardown has
drained. Sync open refuses an active event loop before it opens storage. Both
sync and async builders allocate the host through `Runtime.allocate_unassembled`
and then run the same explicit assembly pipeline; they never wrap an already
live graph in a subclass constructor. A Runtime subclass that overrides
`__init__` must therefore override `allocate_unassembled` and initialize its
subclass-only fields there. The builder validates that contract before opening
an owned store. Custom Runtime subclasses must be created through their
`open`/`aopen` entrypoints or `RuntimeBuilder`; invoking a custom subclass
constructor directly is outside this lifecycle contract.

Before the lifecycle becomes `OPEN`, the assembled Runtime holds a dedicated
recovery lease and drains durable startup work in dependency order: prepared
protected operations, stale capability-use reservations, pending external
effects, resource-usage reservations, process-exec publications, process-launch
publications, checkpoint-restore publications, missing volatile Object
payloads, registered JIT rehydration, stale Explainable Operations, stale
process execution leases, and Object Tasks. Recovery queries use configured,
hard-bounded keyset pages. External providers may reconcile an existing receipt
but are never replayed; ambiguous resource reservations are charged to their
maximum envelope. Process/image/checkpoint publications carry durable plans,
phase receipts, exact recovery leases, and operation bindings so recovery can
compensate or terminalize a specific owner rather than infer success from
adjacent rows. A failed or manual publication keeps mutation admission closed.

Before an async entrypoint offloads allocation or assembly, its event-loop
caller atomically installs an identity-only `StoreAssemblyReservation`. New
`locked()`, `transaction()`, and query scopes from every non-claimant fail fast
while that reservation is installed; only the worker that claims the exact
token may enter the store during assembly. Worker completion and every
scheduling, cancellation, decision-error, and failure path compare-and-release
that exact token. This closes the readiness-probe-to-worker handoff window in
which an event-loop task could otherwise hold the thread-reentrant store lock
while awaiting the worker that needed the same lock.

If a component cannot stop, the raised exception group contains a public
`RuntimeAssemblyCleanupRequired` leaf with the partial Runtime and store. The
same typed handle is published when an async assembly reaches `OPEN` after its
caller was cancelled but normal Runtime shutdown raises or returns an
incomplete outcome. In that case `cleanup_kind` is
`RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN`, and `release()` or
`arelease()` retries normal Runtime shutdown rather than failed-assembly
teardown. The caller retains explicit ownership and can extract and retry it
without relying on garbage collection:

```python
from agent_libos import Runtime
from agent_libos.runtime import RuntimeAssemblyCleanupRequired

try:
    runtime = await Runtime.aopen("runtime.sqlite")
except BaseException as error:
    for handle in RuntimeAssemblyCleanupRequired.extract(error):
        await handle.arelease()
    raise
```

Synchronous callers use `handle.release()`. For
`RuntimeAssemblyCleanupKind.FAILED_ASSEMBLY`, a handle from `Runtime.open()` or
`Runtime.aopen()` closes the builder-owned store only after graph cleanup
succeeds, while a handle from `from_store()` or `afrom_store()` leaves the
supplied store caller-owned. An `OPEN_RUNTIME_SHUTDOWN` handle represents a
Runtime that was fully opened but never delivered to its cancelled caller; its
release API therefore runs ordinary Runtime shutdown, including normal store
release. Before failed-assembly cleanup can unbind its lifecycle
guard, an owned open atomically replaces that exact guard with a unique close
reservation. Successor assembly is rejected while the reservation is pending,
and the handle can close only through an exact compare-and-close operation;
a stale handle therefore cannot close a successor Runtime. Async handle and
failed-open close paths run blocking backend release off-loop, shield it through
caller cancellation, and report cancellation only after the irreversible close
outcome is known.

`agent_libos.runtime.runtime.Runtime` is the stable host facade over that
assembled graph. Its component fields are declared explicitly for static
tooling, and the architecture check rejects composition-root assignments that
are absent from that declaration. Subsystem services do not retain it as a
service locator.

The internal dependency rules are:

- services receive repositories, ports, registries, or callbacks explicitly at
  construction time unless an identified two-way construction dependency
  requires a narrow named binding;
- lower layers do not import concrete API or Runtime implementations;
- one component does not call or read another component's private members;
- identified cycles use named, narrow binding methods rather than a generic
  `bind_runtime` escape hatch;
- Runtime-facade traversal is confined to boundary adapters:
  `GuiRuntimeService`, `LibOSSyscallSession`, and the ephemeral `ToolContext`
  given to model-facing tool wrappers. Stateful subsystem services use explicit
  dependencies instead.

`scripts/check_architecture.py` enforces these rules across both the core
`agent_libos` package and repository-level Runtime Modules, including Runtime
aliases, and ratchets long-function and branch-complexity ceilings. The
checked-in allowlist contains current ceilings, not permanent exemptions:
the check rejects stale ceilings after debt shrinks, so the same change must
remove or lower the entry and the improvement cannot later regress within an
obsolete budget. Underscore-prefixed collaborator fields do not bypass private
access detection; literal `getattr` access and local aliases are normalized to
the same dependency path. The checked-in set of named composition-cycle
bindings is itself ratcheted so new late bindings cannot accumulate unnoticed.

The assembled graph includes:

- `RuntimeStore` persists metadata and append-only records through a backend
  abstraction. SQLite is the default backend; PostgreSQL is available through
  an optional extra. Both SQL backends share the same `SQLRuntimeStore`
  repository contract while backend classes own connection setup and dialect
  behavior.
- `RuntimeModuleRegistry` loads the internal core module and configured trusted
  startup modules before processes, tools, or LLM execution can run. Hook code
  receives an explicit `ModuleHookServices` snapshot and journaled registration
  methods, never the concrete Runtime.
- `CapabilityManager` coordinates separate evaluation, finite-use lease, and
  mutation services.
- `ResourceManager` validates hierarchical budgets, reserves maximum provider
  usage envelopes before dispatch, settles exact usage, and recovers ambiguous
  reservations conservatively on startup.
- `DataFlowManager` owns the versioned Host Sink registry, source/version
  validation, conditional releases, file label bindings, and append-only flow
  decisions. Registry writes require configured `data_flow_sink_registry:*`
  admin authority and are never projected as model tools.
- `ObjectMemoryManager` provides typed memory and namespace resolution.
- `HumanObjectManager` owns questions, approvals, terminal queue processing,
  and human output.
- `FilesystemAdapter`, `GitPrimitive`, `ShellAdapter`, `ClockPrimitive`,
  `JsonRpcPrimitive`, and `McpPrimitive` expose protected primitive operations over provider
  backends. `ShellExecutionPolicy` is the public protocol implemented directly
  by `ShellAdapter` for one-shot Shell and interactive PTY execution; Runtime
  Modules do not call `ShellAdapter` private methods or depend on a forwarding
  wrapper.
- `ToolBroker` is the public tool boundary; `ToolRegistry`,
  `ToolExecutionService`, and `JITToolService` own registration, dispatch, and
  JIT lifecycle respectively.
- `SkillManager` registers standard Skill packages and activates them into
  process tool tables and prompt context without granting resource authority.
- `ProcessManager` owns process lifecycle, working directories, child
  relationships, and durable spawn/fork publications; `ProcessLaunchService`
  owns launch authority and path policy.
- `ProcessTransitionService` is the ordinary semantic status/wait/outcome write
  boundary; dedicated exec and checkpoint publication paths use narrower
  transactional transitions that enforce the same typed-state contract. Row
  `revision`, wait `state_generation`, and exact scheduler
  execution-generation/owner/lease tokens separately fence stale updates,
  repeated-wait ABA wakeups, and detached quantum writes.
- `ImageBootService` owns image preflight, process-exec admission leases,
  phased boot publications, exact rollback snapshots, compensation, and
  startup reconciliation.
- `SimpleScheduler` runs runnable processes and wakes waiting work.
- `ObjectTaskManager` coordinates execution while dedicated state and
  notification services own durable transitions and wake/message publication.
- `CheckpointManager` coordinates restore/fork transactions over typed snapshot
  codecs and remappers. Image artifact loading, image-package installation,
  checkpoint image creation, and image boot are separate services.
- `LLMProcessExecutor` coordinates one process quantum using explicit process,
  repository, provider, pending-action, context-memory, and action-dispatch
  dependencies. LLM requests remain formal protected bidirectional provider
  operations; provider-chain reuse is bound to provider, Sink/trust generation,
  clearance domain, manifest, and context epoch.

The default substrate is `LocalResourceProviderSubstrate`, rooted at the current
workspace unless another substrate is injected.

The internal core module registers the built-in tool set and default images
through the same module registration path exposed to trusted external modules.
This keeps future providers, syscalls, and images from accumulating ad hoc
startup code in the composition root while the module registration journal
keeps rollback ownership explicit.

Host-facing control surfaces live under `agent_libos.api`. The CLI entrypoint
and the local GUI HTTP/SSE server are different presentations over the same
runtime managers and primitives; neither is an authority boundary by itself.
Both must close a Runtime they own: synchronous hosts call
`Runtime.shutdown()`, while event-loop hosts call and await
`Runtime.ashutdown()`. Ordinary shutdown first closes mutation admission and
drains admitted work. While store ownership remains, it then attempts to write
`runtime.shutdown` audit/event evidence; an evidence failure returns before any
component callback. After successful evidence it stops scheduler work and
ObjectTask runner work, runs registered finalizers, stops modules, LLM clients,
supervised blocking work, and the substrate, and only then claims and closes the
store. The durable record therefore evidences a shutdown attempt that reached
this phase, not successful completion of every later teardown step. If a
synchronous quantum, ObjectTask tool thread, or later component cannot be
stopped safely, shutdown reports the exact failed stage and leaves owned storage
open so live work is not racing a closed store connection. Host shutdown never
marks AgentProcess records as exited.

The final ordinary store close uses an exact nonblocking ownership claim. Async
shutdown performs the blocking backend release off-loop and drains that
irreversible step before propagating caller cancellation. Once ownership is
released, the shared shutdown attempt is successful for concurrent followers;
caller-local cancellation or control-flow diagnostics affect only the leader,
and close warnings remain available through idempotent `shutdown()` readback.
If the backend/session was already externally released, shutdown skips
unavailable durable shutdown evidence but still tears down the in-memory graph,
clears the exact guard, and terminalizes the lifecycle with a warning.

If durable publication or terminalization instead installs a recovery-required
fence, that fence is permanent for the current Runtime instance. Normal sync or
async close remains fail closed and preserves the diagnostic store; it does not
emit shutdown evidence, run user finalizers, or perform partial component
teardown. After inspection, the owner must explicitly call
`Runtime.release_recovery_diagnostics()` or await its async counterpart. This
handoff is available only while genuinely fenced and quiescent, stops the
transient worker/component graph without durable writes or ordinary finalizers,
runs only explicitly registered recovery-safe cleanup, and atomically releases
the store commit guard and backend lease. Async teardown runs loop-affine
components on the caller loop. Failures or cancellation before ownership release
reset the handoff for retry. An ownership-released outcome closes the old
lifecycle even when it carries close warnings or overlaps caller cancellation;
warnings remain readable and control-flow interruption is propagated. A fresh
open of the same target then owns startup recovery; the fenced Runtime itself is
never reopened.

## Tool Boundary

LLM-facing tools are stable wrappers over primitives. For example,
`write_text_file` can be visible in a process tool table, but the actual write
still enters the filesystem primitive, which checks:

- workspace containment,
- process working directory resolution,
- filesystem capability or permission policy,
- human approval if policy requires it,
- overwrite and content preview metadata,
- event emission,
- audit recording.

Putting a tool in a process table never grants access to files, shell,
terminal/human I/O, Object Memory, image registration, checkpoints, or other
resources.

Likewise, `call_jsonrpc_method` visibility never grants network authority. The
JSON-RPC primitive accepts only endpoint and method ids, first gates on the
derived `jsonrpc:<endpoint>:<method>` capability resource without loading the
endpoint manifest, then resolves URLs and env-backed headers from the registry
only for an authorized call.

The same split applies to MCP. `list_mcp_servers`, `inspect_mcp_server`,
`list_mcp_tools`, and `call_mcp_tool` are stable generic wrappers over a
registered MCP server registry. Remote MCP tools are not imported into the
ToolBroker as first-class tools, and a visible `call_mcp_tool` entry still
requires `mcp:<server>:<tool>` authority at primitive use. The call path also
checks that derived tool resource before loading server metadata or input
schemas, so missing authority cannot be used to enumerate provider manifests.

## Primitive Boundary

Primitives are the runtime boundary. They are responsible for:

- authorizing the caller pid against capabilities and policy,
- blocking on human approval when needed,
- validating inputs before side effects,
- constraining provider paths, argv, sizes, and timeouts,
- emitting events,
- writing audit records,
- preserving process wake/resume semantics.

JIT syscalls enter the same primitive boundary through
`LibOSSyscallSession`. They do not consult the caller's LLM-facing tool table.
Trusted startup modules may add new syscall names through the runtime syscall
router, but module syscalls still execute as libOS syscalls under the caller
pid and must call primitives for protected effects.

Deno is released only after a dedicated supervisor establishes host-lifetime
process-tree containment: an inherited death pipe plus isolated process group
on POSIX, or a `KILL_ON_JOB_CLOSE` Job Object on Windows. Sandbox execution
fails closed if that containment cannot be established.

## Persistence And Audit

The runtime store keeps durable metadata and append-only records:

- processes, working directories, loaded Skills, and tool tables,
- Object Memory metadata and namespace directories,
- capabilities, finite capability-use reservations, and object handles,
- resource usage plus durable maximum-usage reservations,
- process messages and human requests,
- tools and JIT candidates,
- Skill registry and trust rows,
- loaded Runtime Module status, source hashes, and registration summaries,
- image registry manifests and checkpoint-derived image artifacts,
- JSON-RPC endpoint registry rows,
- MCP server registry rows,
- checkpoints and checkpoint payload snapshots,
- runtime publications and phase receipts for process launch, process exec, and
  checkpoint restore,
- durable LLM pending-action generations, Responses tool outputs, and context
  generations used to validate opt-in provider chaining,
- provider-decided finalized external effects and conservative pending intents,
  including record-level payload-retention tier/digest provenance,
- events and audit records,
- LLM call records with provider ids, model/API mode, usage, errors, and
  full prompt, visible tools, output, tool calls, reasoning metadata, raw
  response, and bounded observability envelopes. Full LLM input/output
  persistence is enabled by default for self-evolution training and
  fine-tuning pipelines; this may include sensitive prompt, tool, reasoning,
  and provider payload fields. Set `llm.persist_full_io: false` to opt out and
  store only previews plus hashes for those fields. Conditional LLM release
  rows apply the same policy before Human approval: with full-I/O persistence
  enabled, SQL stores the prepared request; with it disabled, SQL receives only
  hashes and non-sensitive resume metadata while the exact pending request
  stays in executor memory.

Payload retention is an explicit Host maintenance surface, not startup
recovery. It is disabled by default. When enabled and applied, bounded keyset
pages may reduce eligible terminal LLM-call and external-effect payloads
monotonically from `full` to content-free `summary` and then `hash_only` while
preserving row identity, causal links, classifications, timestamps, and stable
payload digests. Pending/uncertain effects and LLM rows still needed for
continuation or recovery are ineligible. The applied batch and its payload-free
maintenance audit summary commit together; see
[Evidence and LLM Payload Retention](evidence_payload_retention.md).

Object payloads are not ordinary durable object rows. They live in runtime
memory, while SQL object rows store only a runtime-memory marker. Rows whose
live payload cache cannot be reconstructed are released fail-closed on reopen.
Persistent stores take an active-runtime lease so two writable Runtime
instances cannot concurrently open the same database. File-backed SQLite
canonicalizes the database path for both the connection and lease. On the
tested POSIX path, `O_NOFOLLOW` plus `fchmod` enables file-type/symlink checks
and owner-only (`0600`) tightening for the database and sidecars; current-user
ownership is also required where `getuid` is available. Separately, `fcntl`
plus `O_NOFOLLOW` enables the hardened no-follow sidecar `flock` lease. Where
that lease mechanism is unavailable, SQLite's kernel-managed exclusive database
lock is the fallback, without claiming unavailable POSIX mode/ownership
hardening. PostgreSQL derives its advisory-lock key from the current database
and schema. A clean close releases the lease and permits a later reopen.
Checkpoint and image artifact payloads are explicit durable snapshot
exceptions.

Store transactions nest through savepoints, and repository helpers defer their
commits to the outer lifecycle transaction. Commit or savepoint-release failure
is followed by rollback, including restoration of an opted-in Object payload
cache snapshot. If rollback or savepoint cleanup also fails, the store is
poisoned and closed; every later operation fails closed. See
[Runtime Storage](storage.md) for the complete recovery and lease contract.

Audit and events are append-only through the Runtime Store API, and checkpoint
restore must not delete them. This is a runtime invariant rather than a claim
that a host or database administrator cannot tamper with storage; deployments
that require independently verifiable evidence need an external immutable sink
or signed export.
Limited audit views select the latest matching records first and return that
window in chronological order, so GUI snapshots and per-process audit pages keep
showing new records as the log grows. Shell execution records an intent audit
record immediately before crossing into the shell provider; the result, timeout,
or resource-limit audit record uses the intent record as its parent and
correlation id.

Event consumers are bounded at query time rather than loading the durable log
and slicing it in application memory. Runtime/LLM context uses its configured
recent-event limit; GUI snapshots use `gui.snapshot_event_limit`; and the
per-process GUI route accepts a bounded `limit` plus a `before` event-id cursor
for older pages. Each newest or cursor-bounded SQL window is returned in
chronological order. The GUI does not expose an unbounded "all events" API.

## Module Map

```text
agent_libos/
  api/             CLI and GUI HTTP/SSE server entrypoints
  capability/      capability grant, revoke, check, and object handles
  config/          typed runtime, LLM, tool, memory, launcher, and script defaults
  evidence/        protected evidence classification and payload-retention policy
  human/           HumanObject query, approval, interrupt, and output primitives
  images/          built-in AgentImage definitions
  llm/             prompt, context, OpenAI-compatible client, executor, action parser
  memory/          typed Object Memory and MemoryView implementation
  models/          dataclass and enum models split by runtime domain
  modules/         trusted startup Runtime Module loader, registry, and core module
  ports/           narrow subsystem protocols and dependency-inversion boundaries
  primitives/      libOS primitives for filesystem, Git, clock, shell, JSON-RPC, and MCP
  runtime/         composition, syscalls, scheduler, processes, events, checkpoints, audit
  sdk/             public protected-operation lifecycle and provider-facing contracts
  skills/          Skill schema, strict loader, trust registry, and SkillManager
  substrate/       provider interfaces and local host-backed implementations
  storage/         runtime store backends
  tools/           tool base classes, ToolBroker, sandbox, and built-in tools
  utils/           shared validation, YAML loading, and helper utilities
benchmarks/        deterministic runtime-safety benchmark harness and fixtures
docs/              current implementation documentation
experiments/       benchmark entrypoints
gui/               Electron/React desktop console
images/            workspace AgentImage packages
modules/           workspace trusted Runtime Module packages
scripts/           real-model smoke and demo scripts
skills/            workspace standard Agent Skill packages
tests/             safety-boundary and regression tests
```
