# Architecture

Agent libOS is structured around one boundary: model-visible and
self-evolving action surfaces are not resource authority. A process may see a
tool schema, activate a Skill, register a JIT tool, register or exec an image,
fork a child, restore from a checkpoint, or inspect a remote endpoint, but
process-selected external effects enter a primitive and the Protected Operation
SDK under that process id. Trusted Runtime artifact publication is the narrow
TCB exception described below; it is not a model-facing authority path.

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
     - UnitOfWork and typed repository facades
  -> protected provider boundaries
     - LLM provider service/client
     - Resource Provider Substrate
       - filesystem provider
       - pinned system-Git provider
       - clock provider
       - shell provider
       - human provider
       - JSON-RPC over HTTP provider
       - MCP client provider
     - PTY provider installed by the trusted PTY Runtime Module
  -> host backend
     - local workspace filesystem
     - host clock
     - subprocess backend
     - terminal or UI human I/O
     - Host-configured LLM API
     - pre-registered remote JSON-RPC endpoints
     - pre-registered MCP servers
     - future container, WASM, or service providers
```

The built-in LLM provider boundary records a bounded terminal attempt trace
inside the logical LLM call. This is an observability projection, not another
provider/effect layer: retries and fallbacks remain within one protected
operation and one resource reservation. The GUI receives content-free call
summaries in snapshot/SSE and reads retained trace content on demand through
the authenticated loopback server.

The Skills and tools layer exists for LLM ergonomics and self-evolution. It
presents stable action names, schemas, summaries, workflow instructions, and
process-local JIT candidates. It does not own external authority.

Image registration and `exec` are also self-evolution mechanisms. They can
change a process prompt, prompt composition mode, default tool table, default
Skills, and lifecycle shape. Image visibility, `required_capabilities`, and
target-image metadata do not grant external resource capabilities or impose
resource budgets. Launch-time callers may provide an explicit durable
[`TaskAuthorityManifest`](task_authority_manifest.md). When present, that
manifest owns capability, effect-class, approval-policy, and resource-budget
ceilings for later process-controlled transitions. When the caller omits it,
the Runtime persists an implicit authority record for execution metadata; that
implicit record is not an explicit Host transition ceiling, and a later
Host-authorized grant may therefore extend beyond it.
Image `required_capabilities` are unmet-requirement declarations, not grants.
Image packages are the narrow exception: they may seed a private per-process
workspace and issue only the filesystem capabilities declared by
`IMAGE.yaml workspace.grants` for that materialized private copy. This is a
trusted-package bootstrap rule, separate from a Task Authority ceiling; it
must not be treated as a general process grant path. Those capabilities are
initially issued only to the booted process and cannot name or expose the
package source directory to it. A package may explicitly mark one as
delegable; any later child derivation must still use the ordinary Capability
attenuation path and remains confined to the materialized private copy. Package
JIT tools remain process-local. The Host image registry does retain the
absolute package source path for administration and inspection, so the stronger
non-exposure statement does not apply to Host APIs.
At Image boot, default tool tables are exact declarations: the runtime does not
implicitly add generic lifecycle or Object Memory tools. Separately registered
Skills may later expand the process tables under their own trust and
Skill-authority checks.
Images may opt into `metadata.tool_projection: skills`. The complete image tool
table stays callable and capability-enforced, while a separate durable model
projection initially contains exactly `discover_skills`, `activate_skill`,
`read_skill_resource`, `unload_skill`, and `process_exit`. This is the shipped
base, coding, review, and toolmaker contract. Static bindings for filesystem,
Git, checkpoint, Capability, JSON-RPC, MCP, and other domains are not thereby
initially model-visible. Activating an applicable built-in Skill projects its
entire owned subset from the complete table and changes prompt/schema visibility
without changing Capability authority.

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

The LLM provider service/client is assembled beside that substrate rather than
as a field of the `ResourceProviderSubstrate` protocol. PTY is likewise supplied
by its trusted Runtime Module, optionally using a Host-injected PTY provider.
Both still enter the same Protected Operation SDK lifecycle; the distinction is
composition ownership, not weaker mediation.

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

Trusted runtime artifact publication is an explicit exception, not a
model-facing provider boundary. Image-package materialization writes and removes
the Runtime-owned private workspace directly with Host filesystem APIs while a
durable runtime-publication program records recovery and compensation. Those
writes do not receive filesystem-provider data-flow/effect-ledger semantics and
must not be generalized into an extension escape hatch; untrusted or
process-selected filesystem work still goes through the filesystem primitive.

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

The PTY Runtime Module applies this pending-to-finalized protocol to every
protected PTY operation, including spawn, read, continuous ingest, write,
resize, and close. Cleanup after a spawned session fails to publish its
Object is containment, not evidence that spawn never occurred; classifier
absence or failure after a PTY operation finalizes an `unknown` fallback, while
post-provider sink failure leaves the pending row visible.

## Composition Root And Internal Dependencies

`agent_libos.runtime.builder.RuntimeBuilder` is the composition root. It either
opens a builder-owned store (`open`/`aopen`) or accepts a caller-owned store
(`from_store`/`afrom_store`), constructs all acyclic dependencies in dependency
order, and uses
named late bindings only for explicit construction cycles: Data Flow/Human,
Resource/Process and Object Task notifications, lifecycle participants, and
the checkpoint module-catalog/image-registry pair. Protected-operation recovery
and Process/Image Boot use their named recovery/hook registries for the same
reason. The builder then loads trusted extensions and cleans up a partially
assembled host on failure. Async hosts opening a builder-owned target use
`await Runtime.aopen()`; hosts supplying a caller-owned store use
`await RuntimeBuilder.afrom_store(...)`. Failed assembly cleanup runs on that
caller loop and is shielded until teardown has drained. Sync open refuses an
active event loop before it opens storage. Both sync and async builders allocate the host through
`Runtime.allocate_unassembled`
and then run the same explicit assembly pipeline; they never wrap an already
live graph in a subclass constructor. A Runtime subclass that overrides
`__init__` must therefore override `allocate_unassembled` and initialize its
subclass-only fields there. The builder validates that contract before opening
an owned store. Custom Runtime subclasses must be created through their
`open`/`aopen` entrypoints or `RuntimeBuilder`; invoking a custom subclass
constructor directly is outside this lifecycle contract.

Before the lifecycle becomes `OPEN`, the assembled Runtime holds a dedicated
recovery lease. It first validates recoverable TaskRun plaintext and integrity
bindings without dispatch, then drains durable startup work in dependency order:
prepared protected operations, pending external effects, semantic authority,
stale capability-use reservations, resource-usage reservations, process-exec
publications, process-launch publications, checkpoint-restore publications,
root-spawn initial-goal payloads, missing volatile Object payloads, registered
JIT rehydration, stale Explainable Operations, stale process execution leases,
Object Tasks, incomplete process-terminal cleanup intents, and TaskRun recovery.
Pending-effect reconciliation precedes stale capability-reservation abandonment
because a provider receipt may prove an effect never started and restore its
bound reservation. Recovery queries use configured, hard-bounded keyset pages.
External providers may reconcile an existing receipt but are never replayed;
ambiguous resource reservations are charged to their maximum envelope.
Process/image/checkpoint publications carry durable plans, phase receipts,
exact recovery leases, and operation bindings so recovery can compensate or
terminalize a specific owner rather than infer success from adjacent rows. A
failed or manual publication keeps mutation admission closed.

That recovery-lease pass is followed by a second, still pre-`OPEN` STARTING
phase. Under the startup lease the Runtime runs trusted startup hooks, starts
the ObjectTask worker, performs checkpoint payload begin/prepare/complete
delivery, reconciles terminal restore publications again, and commits the
payload acknowledgement before publishing `OPEN`. Normal mutation admission
does not open between these two phases.

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

- `RuntimeStore` is the concrete backend boundary. `UnitOfWork` supplies the
  shared transaction boundary and typed process, Object, authority, resource,
  publication, snapshot, evidence, extension, module-publication, retention,
  and protected-effect repositories injected into subsystem services. Direct
  raw-store use is reserved for Host lifecycle/storage ownership and small
  compatibility surfaces rather than being the normal subsystem dependency.
  SQLite is the default backend; PostgreSQL is available through an optional
  extra, and each backend owns connection setup and dialect behavior behind the
  same UnitOfWork backend contract.
- `RuntimeModuleRegistry` loads the internal core module and configured trusted
  startup modules before processes, tools, or LLM execution can run. Hook code
  receives an explicit `ModuleHookServices` snapshot and journaled registration
  methods, never the concrete Runtime.
- `CapabilityManager` coordinates separate evaluation, finite-use lease, and
  mutation services.
- `ResourceManager` validates hierarchical budgets, can durably reserve an
  operation-supplied maximum provider-usage envelope before dispatch, settles
  exact usage, and recovers those reservations conservatively on startup.
  The LLM executor uses this path: after exact request assembly it reserves the
  maximum call/token envelope before Provider dispatch and then exact-settles a
  completed call or maximum-settles an ambiguous outcome.
- `DataFlowManager` owns the versioned Host Sink registry, source/version
  validation, conditional releases, file label bindings, and append-only flow
  decisions. Public registry writes require configured
  `data_flow_sink_registry:*` admin authority and are never projected as model
  tools. Before `OPEN`, the trusted Host bootstrap may reconcile only the
  Host-configured rules without a process capability.
- `ObjectMemoryManager` provides typed memory and namespace resolution.
- `EventBus` validates and appends the closed `EventType` catalog. Its event
  insert and active operation-evidence link are atomic, while wider
  state/event/audit coupling remains the responsibility of the owning manager.
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
  prompt context and the model projection. Immutable built-in Skills project
  only bindings already present in the complete process table; registered
  Skills may add separately authorized static/JIT bindings to both tables.
  Neither path grants resource authority.
- `ProcessManager` owns process lifecycle, working directories, child
  relationships, and durable spawn/fork publications; `ProcessLaunchService`
  owns launch authority and path policy.
- `ProcessTransitionService` is the ordinary semantic status/wait/outcome write
  boundary; dedicated exec and checkpoint publication paths use narrower
  transactional transitions that enforce the same typed-state contract. Row
  `revision`, wait `state_generation`, and exact scheduler
  execution-generation/owner/lease tokens separately fence stale updates,
  repeated-wait ABA wakeups, and detached quantum writes. Registration by a
  different condition domain, retargeting an already active condition, and
  generic pause/resume while a condition or Host-resume gate owns the process
  fail closed.
- `ImageBootService` owns image preflight, process-exec admission leases,
  phased boot publications, exact rollback snapshots, compensation, and
  startup reconciliation.
- `SimpleScheduler` claims only `runnable` processes and advances
  scheduler-owned quanta. It does not interpret or clear typed waits. Child,
  message, Human, and ObjectTask managers own the durable condition they
  created; syscall cleanup likewise owns only the exact interrupted wait it
  recorded. Those owners release a matching revision/state generation through
  `ProcessTransitionService`, with token-based owners failing closed on stale
  wake tokens. Direct Host workflows and Host-managed ObjectTask runners can
  also advance a process without making it scheduler work.
- `TaskRunManager` supervises one root process tree through a versioned goal,
  requirements, idempotent Host commands, an append-only linked ledger, and
  integrity-bound local resume points. It asks the existing scheduler and
  managers to advance work rather than interpreting a workflow DAG. Store
  leases plus the Run's monotonic Runtime epoch fence stale claims and commits;
  unsafe external-effect or ObjectTask recovery blocks in `needs_attention`.
- `ObjectTaskManager` coordinates execution while dedicated state and
  notification services own durable transitions and wake/message publication.
- `CheckpointManager` coordinates restore/fork transactions over typed snapshot
  codecs and remappers. Image artifact loading, image-package installation,
  checkpoint image creation, and image boot are separate services.
- `LLMProcessExecutor` coordinates one process quantum using explicit process,
  repository, provider, pending-action, context-memory, and action-dispatch
  dependencies. LLM requests remain formal protected bidirectional provider
  operations. The Runtime computes and records provider-, Sink-, clearance-,
  manifest-, and context-sensitive fingerprints, but the current full-snapshot
  AgentProcess executor disables `previous_response_id` reuse and sends complete
  local context. The low-level client does not enforce those Runtime
  fingerprints; its narrower dispatch gate is an explicit id, an official
  stored Responses request, and representable tool history.

Prompt caching separates a model-visible layout from provider transport policy.
The defaults, `legacy_v1` and `provider_default`, preserve the legacy prompt and
send no v2 cache options. Opt-in `cache_optimized_v2` keeps stable instructions
and append-only TaskRun requirements ahead of volatile state and minimizes
libOS-owned metadata. `implicit` and `explicit` cache modes require a
Host-configured `prompt_cache_key`; Runtime derives the wire key from that
privacy domain plus provider, model, stable prefix, and tool fingerprint rather
than a Run or process id. Explicit mode also marks one stable text breakpoint.
The only v2 TTL is `30m`, mutually exclusive with legacy
`prompt_cache_retention`.

If an endpoint rejects a cache field, the bounded compatibility retry removes
the entire v2 cache-option group. That retry remains inside the same logical
protected operation and resource reservation. The LLM record distinguishes
configured layout/policy from the secret-free options accepted on the successful
attempt and records a content-free downgrade reason. Promoting v2 from opt-in is
guarded by the paired v1/v2 multi-provider release gate; this is release
qualification rather than a Runtime authority boundary. See the detailed
[provider policy and gate](providers.md#prompt-caching-v2-release-evidence).

The default substrate is `LocalResourceProviderSubstrate`, rooted at the current
workspace unless another substrate is injected.

The durable event envelope, closed event catalog, ordering model, and
transaction/causality limits are documented in [Runtime Events](events.md).

The internal core module registers the built-in tool set and default images
through the same module registration path exposed to trusted external modules.
This keeps future providers, syscalls, and images from accumulating ad hoc
startup code in the composition root while the module registration journal
keeps rollback ownership explicit.

Host-facing control surfaces live under `agent_libos.api`. The CLI entrypoint
and the local GUI HTTP/SSE server are different presentations over the same
runtime managers and primitives. They are not process effect-execution
boundaries, but they are trusted Host/admin authority-assignment boundaries:
documented CLI Host commands run with local Host authority, and the GUI bearer
token authenticates the holder to documented Host/admin routes. Actor/pid modes
in either surface deliberately switch to process authority. Primitive and
Protected Operation SDK checks remain the boundary for effects selected by a
process.
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

Scheduler worker threads may execute a quantum that returns an awaitable. Such
a worker owns a private event loop for that quantum; after the main awaitable
settles, it makes bounded cleanup attempts for every remaining loop task, async
generator, and default-executor worker before closing the loop. Incomplete
task/generator/executor cleanup makes the quantum fail rather than silently
claiming a clean boundary. A default executor that has not actually stopped is
also represented by a tracked scheduler lifecycle fence. Runtime shutdown then
reports `scheduler_stopped: false` and preserves the open store until that
executor drains and shutdown is retried.

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

LLM-facing tools are stable wrappers over primitives. A binding can exist in the
complete process tool table without appearing in the model projection. When
`write_text_file` is projected to the model (or invoked by a trusted direct
workflow), the actual write still enters the filesystem primitive, which checks:

- workspace containment,
- process working directory resolution,
- data-flow Sink clearance and source/carrier labels,
- filesystem capability, Task Authority effect ceiling, and permission policy,
- human approval if policy requires it,
- overwrite and content preview metadata,
- resource preflight/accounting, a finite-authority reservation, and a pending
  external-effect intent,
- source, target, payload, and policy revalidation immediately before dispatch,
- provider classification and label/resource/authority settlement,
- event emission,
- audit recording.

Putting a tool in a process table never grants access to files, shell,
terminal/human I/O, Object Memory, image registration, checkpoints, or other
resources.

Likewise, model projection or direct invocation of `call_jsonrpc_method` never
grants network authority. The
JSON-RPC primitive accepts only endpoint and method ids, first gates on the
derived `jsonrpc:<endpoint>:<method>` capability resource without loading the
endpoint manifest, then resolves URLs and env-backed headers from the registry
only for an authorized call.

The same split applies to MCP. `list_mcp_servers`, `inspect_mcp_server`,
`list_mcp_tools`, and `call_mcp_tool` are stable generic wrappers over a
registered MCP server registry. Remote MCP tools are not imported into the
ToolBroker as first-class tools, and a model-projected `call_mcp_tool` entry still
requires `mcp:<server>:<tool>` authority at primitive use. The call path also
checks that derived tool resource before loading server metadata or input
schemas, so missing authority cannot be used to enumerate provider manifests.

Manifest v1 remains on the initialize-based legacy MCP path. Manifest v2
explicitly selects `legacy`, `auto`, or `2026-07-28` and requires the optional
modern provider extension. Protocol discovery is itself a protected external
read; its server identity, negotiated revision, and advertised capabilities are
bounded observations rather than Tool registration or authority. Negotiation,
bounded Tool-catalog pagination, live validation, and call share one deadline,
cumulative byte budget, registry fence, and external-effect operation.

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

## Semantic Approval and Flow Plane

The optional semantic subsystem is Host-owned. Approval, root-goal, provider,
Tool, LLM, Object/file, and materialization capture create bounded jobs and a
payload-free FlowGraph. A lease/CAS worker executes a typed deterministic,
scripted, or explicitly configured external assessment; the Shadow broker
records `would_issue_exact_once`, `would_deny`, or `require_human`. Classifier
findings can veto or escalate but cannot supply allow predicates or terminal
instructions.

The default `semantic.mode` is `off`, which performs no capture writes, claims
no jobs, and invalidates undispatched semantic grants. `shadow` adds only
semantic evidence. `enforce_deny` may terminalize only the closed Host
hard-deny set. `canary_auto` may issue one exact, short-lived, nondelegable,
one-use Capability for catalog-v1 reads under an immutable static Host policy
epoch. The private settlement port shares the Human revision/status CAS and
transactional terminal kernel, so Human/cancel/machine races have one winner.
No Runtime facade, CLI, HTTP, GUI, model Tool, Skill, JIT, or Module exposes that
private settlement port. The Host-owned Python component graph does expose
`semantic_control`, including policy admission and authority-narrowing disable;
it is a trusted composition/control surface, not a remote or model-facing one.

Approval and provider-ingress jobs are metadata-only. A root-goal job may
temporarily contain a deterministic redacted intent only for bounded
`public`/`normal`, non-mixed-identity text that passes local secret and path
detection; every other goal falls back to metadata-only, and terminalization
scrubs the projection. The Host-owned provider-result observer is bound once at
composition. Invocation observers are additive, cannot replace it, and fail
independently. Provider ingress is captured only when the post-commit SDK can
derive a bounded canonical result digest; its descriptor is payload-free and
an unavailable digest becomes a capture failure without changing the result.
Root goals and safely traversable provider results also pass through a bounded
local Host DLP detector. It retains only closed category/reason/digest evidence,
forces metadata-only projection on a hit, and merges monotonic Host findings
into every terminal assessment without writing labels back.

An optional classifier call is itself a runtime-internal Protected Operation
named `semantic.llm.assess`, with effect class `llm.complete`, a frozen Host
profile identity and model, a profile-bound Sink identity, and normal DataFlow
preflight/revalidation. Snapshot/resolution/model/Sink drift fails closed. It
disables automatic release-request creation; conditional egress therefore
terminates the job as `egress_blocked`. Provider ambiguity is terminal and is
never automatically replayed. The general LLM process executor is not reused,
so its full-I/O persistence path cannot capture the semantic prompt or
response.

Store schema v7 keeps mutable queue/control/rate state separate from append-only
assessment, FlowGraph, policy epoch, machine-settlement, health, and review
evidence. Temporary safe projections are bounded and reduced to
hash-only at terminalization, expiry, cancellation, failure, or kill-switch
cleanup. Assessment records retain closed findings and Host provenance digests,
never prompts, raw task/provider text, classifier responses, or reasoning.
FlowGraph coverage other than `complete`, mixed tenant identity, stale input,
or any model finding forces Human review; semantic findings never write
`DataLabels`, declassify, or endorse.

At issuance, the machine transaction re-reads the request, Process, manifest,
ceiling, epoch, tenant, graph coverage, provider/tool/Sink/state identities,
classifier provenance, and budget. Protected Operation revalidates the
versioned exact binding and durable control generation during authorization,
reservation, prepare, and before each dispatch. `off`, epoch revocation, or a
safety trip blocks new settlement and every unconsumed/undispatched grant. See
[Semantic Approval and Data Identification](semantic_shadow.md) for the full
contract.

## Persistence And Audit

The runtime store keeps durable mutable state and append-only evidence,
including:

- processes, working directories, loaded Skills, complete callable tool tables,
  and separate model tool projections,
- Object Memory metadata and namespace directories,
- capabilities, finite capability-use reservations, and object handles,
- resource usage plus durable maximum-usage reservations,
- process messages and human requests,
- tools and JIT candidates,
- Skill registry and trust rows,
- loaded Runtime Module status, source hashes, and registration summaries,
- image registry manifests and package- or checkpoint-derived image artifacts,
- JSON-RPC endpoint registry rows,
- MCP server registry rows,
- checkpoints, checkpoint payload snapshots, and payload-delivery attempts,
- runtime publications and phase receipts for process launch, process exec, and
  checkpoint restore,
- Task Authority manifests and process authority bindings,
- Object Tasks and agent ratings,
- Host Sink trust rows, data-flow decisions and releases, and durable file
  labels,
- semantic assessment jobs with lease/CAS state, payload-free FlowGraph,
  immutable policy epochs, revisioned control/rate state, and append-only
  control-transition, assessment, machine-settlement, health, and review evidence,
- explainable operations, evidence links, and context manifests,
- durable LLM pending-action generations, `image_only` native transcript tool
  outputs, compatible Responses-continuation rows, and context generations,
- provider- or trusted-runtime-classified finalized external effects and
  conservative pending intents,
  their append-only transition history, and record-level payload-retention
  tier/digest provenance,
- events and audit records,
- LLM call records with provider ids, model/API mode, usage, outcome metadata,
  and bounded observability envelopes. Full ordinary LLM I/O persistence is
  enabled by default for self-evolution training and fine-tuning pipelines: it
  retains prepared prompts and visible tools and, on success, output, tool
  calls, reasoning metadata, and a bounded provider-response projection. That
  projection hashes recognized credential and opaque fields and replaces
  over-limit structure or text with bounded digest descriptors; it is not the
  provider's raw response. The retained fields may still contain sensitive
  material. A failed call's request fields follow the same setting, but provider
  or extension exception text is never durable or model-visible, regardless of
  `llm.persist_full_io`; only a stable public error envelope and content-free
  internal observations such as type, length/hash, and correlation id may cross
  those boundaries. Set `llm.persist_full_io: false` to store
  content-free byte counts, JSON-kind/item-count metadata where applicable, and
  hashes for ordinary I/O fields. That setting cannot run an `image_only`
  Image, whose next quantum requires a lossless durable transcript head and
  therefore fails before provider dispatch.
  Conditional LLM release
  rows apply the same policy before Human approval: with full-I/O persistence
  enabled, SQL stores the prepared request; with it disabled, SQL receives only
  hashes and non-sensitive resume metadata while the exact pending request
  stays in executor memory.

Each executor-level logical LLM call uses the protected-operation resource
contract: after exact request assembly, Runtime atomically persists the
prepared effect and a maximum call/token reservation before Provider dispatch.
The reservation constrains the process ancestry; exact usage settles before
the LLM row and selected tools, certified non-start releases, and ambiguous
outcomes maximum-settle during the call or startup recovery. Internal SDK and
compatibility retries remain within one logical call, so this boundary is not
an exact physical-request, Provider-billing, currency, or monetary-spend cap.

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
memory, while current SQL object writes store only a runtime-memory marker.
Accepted legacy rows from older development builds may still contain full JSON
payloads and must be migrated or recreated before claiming marker-only
historical retention. Marker rows whose live payload cache cannot be
reconstructed are released fail-closed on reopen. The one current exception is
an internal, integrity-bound recovery envelope for the immutable initial GOAL of a
committed, live root spawn when `llm.persist_full_io=true`: startup validates
the process and Object identity plus payload digest before rehydration, generic
publication reads redact the payload, and terminal cleanup reduces the envelope
to hashes. It does not apply to child/fork/ObjectTask or ordinary Object payloads,
or to a replacement goal supplied by exec; an exec that preserves the original
root goal remains eligible. `persist_full_io=false` stores no reversible goal
content. See
[Runtime Storage](storage.md#transaction-model).
Persistent stores take an active-runtime lease so two writable Runtime
instances cannot concurrently open the same file-backed SQLite target or the
same PostgreSQL `(current_database(), current_schema())` target. Separate
PostgreSQL schemas have distinct lease identities. File-backed SQLite
canonicalizes the database path for both the connection and lease. On the
tested POSIX path, `O_NOFOLLOW` plus `fchmod` enables file-type/symlink checks
and owner-only (`0600`) tightening for the database and sidecars; current-user
ownership is also required where `getuid` is available. Separately, `fcntl`
plus `O_NOFOLLOW` enables both the hardened no-follow path-sidecar `flock` and
an owner-only private identity lease keyed by the validated database
`(st_dev, st_ino)`. Database, lease, identity-lease, and SQLite sidecar files
must be regular, current-user-owned, single-link files on that path. Pre/post
path-identity checks and those leases reject ordinary aliases and replacement
races; the live connection also holds SQLite's exclusive lock on the actual
database it opened, preserving one active writer if a same-UID filesystem
administrator races the pathname at connect time. The standard SQLite driver
cannot prove that such a raced connection still names the originally selected
inode, so writable database directories are inside the Host trust boundary and
must not be renamed/replaced while the Runtime is alive. Where the POSIX lease
mechanism is unavailable, the exclusive database lock remains the single-writer
boundary without claiming unavailable path/inode or mode/ownership hardening.
PostgreSQL derives its advisory-lock
key from the current database and schema. A clean close releases the lease and
permits a later reopen. Checkpoint and image artifact payloads are explicit
durable snapshot exceptions.

Store transactions nest through savepoints, and repository helpers defer their
commits to the outer lifecycle transaction. After an outer commit error, the
store rolls back SQL and the corresponding Object-payload before-images only
when the driver proves the transaction is still active. If commit may already
have applied, the outcome is ambiguous: the store does not manufacture a
rollback, restores no cache-only before-image, and instead poisons/closes its
data plane unless a narrow typed caller proves and accepts the exact committed
state. A failed nested `RELEASE SAVEPOINT` similarly attempts
`ROLLBACK TO`/`RELEASE` and restores payload before-images only after that SQL
rollback succeeds; failure of the recovery sequence poisons and closes the
store. Every later ordinary operation then fails closed. See [Runtime
Storage](storage.md) for the complete recovery and lease contract.

Every process transition to `exited`, `failed`, or `killed` creates a durable
terminal-cleanup intent in the same transaction as the terminal process row.
Post-commit cleanup claims an exact owner/lease and records completion of the
independent `terminal_notify` and `process_finalize` phases. Completed phases
are not replayed; a failed attempt retains content-free diagnostics and raises
`ProcessTerminalCleanupRequired` (with control-flow interruptions retained in
an exception group after all phases are attempted). Startup reclaims incomplete
leases under the recovery lease, retries only unfinished idempotent phases in
bounded keyset pages, and keeps normal admission closed until that pass ends.

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

Model- and GUI-facing event consumers are bounded at query time rather than
loading the durable log and slicing it in application memory. Runtime/LLM
context uses its configured recent-event limit; GUI snapshots use
`gui.snapshot_event_limit`; and the per-process GUI route accepts a bounded
`limit` plus a `before` event-id cursor for older pages. Each newest or
cursor-bounded SQL window is returned in chronological order. The GUI does not
expose an unbounded "all events" API. Trusted Host code may call
`EventBus.list(limit=None)`, and checkpoint replay currently loads the complete
durable event table before locating the checkpoint and selecting the scoped
replay prefix; those paths are not claimed to provide a storage-work bound and
should not be exposed as model-facing list operations.

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
  runtime/         public Runtime facade, composition, syscalls, scheduler, processes,
                   events, checkpoints, audit, and recovery coordination
  sdk/             public protected-operation lifecycle and provider-facing contracts
  semantic/        Host semantic assessment, flow, control, settlement, and recovery
  skills/          Skill schema, strict loader, trust registry, and SkillManager
  substrate/       provider interfaces and local host-backed implementations
  storage/         UnitOfWork, typed repositories, store backends, and migrations
  tools/           tool base classes, ToolBroker, sandbox, and built-in tools
  utils/           shared validation, YAML loading, and helper utilities
benchmarks/        runtime-safety, practical, Skill-projection, long-horizon, and
                   other benchmark harnesses and fixtures
docs/              current implementation documentation
experiments/       benchmark entrypoints
gui/               Electron/React desktop console
images/            workspace AgentImage packages
modules/           workspace trusted Runtime Module packages
scripts/           validation, migration support, release checks, and opt-in model scripts
skills/            workspace standard Agent Skill packages
tests/             safety-boundary and regression tests
```
