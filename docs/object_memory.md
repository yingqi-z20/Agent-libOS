# Object Memory

Object Memory is typed, capability-controlled runtime memory for agent state.
It is not a plain key/value store and object names are not capabilities.

## Objects

Objects have:

- an object id (`oid`),
- type,
- namespace,
- a non-empty namespace-local name (the create argument may be omitted, in
  which case the Runtime generates `<type>:<oid>`),
- version,
- metadata,
- payload,
- creator provenance (`created_by`),
- explicit owner (`owner_kind`, `owner_id`) and lifecycle state,
- object capabilities.

Object handles carry object-specific rights such as `read`, `write`,
`materialize`, `link`, `diff`, or `delete`.

`ObjectType` is a closed enum. The accepted values are `task`, `goal`, `plan`,
`step`, `constraint`, `message`, `human_decision`, `human_request`,
`tool_result`, `observation`, `error_trace`, `code_patch`, `test_result`,
`evidence`, `claim`, `summary`, `skill`, `tool_spec`, `tool_candidate`,
`tool_artifact`, `checkpoint`, `process_state`, `external_ref`, and `artifact`.
`Runtime.memory.create_object(...)` accepts an `ObjectType` member or its exact
string value and rejects values outside this set.

Object metadata passed to `Runtime.memory.create_object(...)` is a complete
`ObjectMetadata` value, not a mapping merged with the configured defaults. If
it is omitted, the manager starts with the active Runtime's
`memory.metadata_sensitivity` and `memory.metadata_retention_policy`. The
model-facing `create_memory_object` tool accepts a partial metadata mapping and
uses those same active Runtime values when `sensitivity` or `retention_policy`
is absent. In both APIs, `token_estimate` is Runtime-managed derived metadata:
the manager computes it from the payload rather than trusting a caller-supplied
value.

`created_by` is provenance and does not drive cleanup. Runtime cleanup uses the
explicit owner pair. Ordinary process-created objects start as
`owner_kind=process`, `owner_id=<pid>`. A final result is retained by
transferring it to `owner_kind=process_result`, and ObjectTask results are
transferred to `owner_kind=object_task`.

Online Object lifecycle release is centralized in `ObjectMemoryManager`.
Releasing an object marks the row as `released`, removes its runtime payload,
deletes links that touch it, and revokes stale `object:<oid>` capabilities.
Released objects are not returned by oid lookup, name lookup, namespace
listing, or materialization; their namespace-local names can be reused by new
live objects.

Runtime startup has one deliberate recovery-only exception. Before normal
admission opens, the Object repository directly scans live rows whose durable
`runtime_memory` marker has no reconstructable process-local payload. In
bounded store transactions it marks those rows `released`, removes their
links, and revokes active Object capabilities. This recovery sweep does not
invoke `ObjectMemoryManager` release finalizers or its per-Object online audit
path. Hash-anchored payloads from an interrupted checkpoint restore are
rehydrated before the sweep. A runtime-only Object must therefore not be used
as the sole durable record of a host resource that requires finalization after
reopen.

Ownership changes are lifecycle changes and increment the Object version.
Create, update, append, transfer, and trusted delete all acquire the Object
Memory ownership lock before entering the store transaction. Updates,
transfers, and deletes condition their row write on `lifecycle_state=live` plus
the captured owner and version; a lost conditional update cannot report
success, overwrite a concurrent owner, or revive a released Object. Owner
cleanup enumerates candidates but deletes each one only if its `owner_kind`,
`owner_id`, and version still match the captured tuple. Transfer increments the
version. Every row whose captured owner/version still matches is updated, and
the successfully transferred subset plus its audit row publish in one
transaction; a missing, differently owned, or conditionally rejected row is
skipped and is not reported as transferred. Therefore a release racing an
ownership transfer cannot delete the new owner's Object, including an A-to-B-to-A
(ABA) cycle: returning to the same textual owner does not restore the old
version.

Trusted delete holds the ownership lock while it snapshots and conditionally
checks the Object. Host-resource finalizers then run outside the SQL transaction
so a provider cleanup such as PTY close can durably write its own pending effect
intent before crossing the host boundary. A finalizer failure leaves relational
release state untouched. After finalization, delete opens a store transaction,
rechecks the exact LIVE/owner/version tuple, and commits the Object release,
object-capability revocation, and delete audit together. A later relational or
audit failure rolls back the Object row, capabilities, and in-memory payload;
it cannot undo an already completed host finalizer, whose effect ledger remains
the reconciliation evidence.

## Namespaces

Object names are local to a namespace. Runtime code that omits `namespace`
uses the caller process namespace:

```python
pid = runtime.process.spawn(image="base-agent:v0", goal="collect notes")
handle = runtime.memory.create_object(
    pid=pid,
    object_type="summary",
    name="notes",
    payload={"entries": []},
    immutable=False,
)
obj = runtime.memory.get_object_by_name(pid, "notes")
assert obj.namespace == runtime.memory.process_namespace(pid)
```

Each process gets its own default namespace at spawn/fork time. A bare name
such as `notes` can exist independently in two process namespaces.

Explicit namespaces are directory-like scopes:

```python
runtime.memory.create_namespace(pid, "project")
runtime.memory.create_namespace(pid, "project/research")
runtime.memory.create_object(
    pid=pid,
    object_type="observation",
    namespace="project/research",
    name="notes",
    payload={"source": "README.md"},
)
listing = runtime.memory.list_namespace(pid, "project/research")
```

The trusted Host API returns full `AgentObject` values, including payloads, from
`list_namespace`. Model-facing tools and syscalls deliberately return metadata
projections instead. Do not expose the Host manager directly to an untrusted
caller that should only see the projected form.

The `limit` on `list_namespace` is one shared cap across visible Objects and
immediate child namespaces, not a per-kind cap. The manager fills visible
Objects first and uses only the remaining slots for child namespaces. The
current result has no cursor, `has_more`, or `truncated` marker, so a response
whose entry count reaches the limit does not prove that the namespace listing
is complete. Model-facing projections preserve this ordering and completeness
limitation while omitting Object payloads.

Namespace capabilities gate listing, lookup, Object creation, and creation of
a child namespace. Creating a child requires `write` on its existing parent;
successful creation grants the creator `read`, `write`, and `admin` on the new
namespace. Top-level namespace allocation is the explicit exception: when a
namespace has no parent, `create_namespace` has no pre-existing namespace
capability to check and grants those same rights to its creator. Consequently,
untrusted callers that must not allocate global top-level names should not be
given direct access to that API without a Host-owned naming policy.

Namespace capabilities do not replace object capabilities. Reading
`project/research/notes` requires namespace read authority and object read
authority.

## Name Resolution

A name is not itself authority. Resolution requires:

1. namespace rights for the directory-like lookup,
2. object rights for the requested operation.

This prevents a process from using a guessed object id or shared name to bypass
capabilities.

One-shot namespace grants are consumed only after a successful namespace
operation. For example, an `allow_once` `object_namespace:<ns>` `read` grant can
complete one named lookup or namespace listing, then later lookups must be
authorized again. Failed validation or missing objects do not burn the grant.

Namespace listing also consumes the finite visibility authority actually used
to construct its result. In one store transaction it settles the parent
namespace read plus each returned object's `object:<oid> read` decision and
each returned child namespace read decision, deduplicating repeated capability
ids. Thus an object visible through a one-use object-read grant appears in the
first successful listing and is absent from the next; listing cannot use a
temporary visibility grant as a reusable directory oracle. Validation, audit,
or settlement failure rolls back the whole listing consumption.

## Query API

The public Python API accepts an `ObjectQuery` and returns bounded, read-only
handles rather than Object payloads:

```python
from agent_libos import ObjectQuery, ObjectType

handles = runtime.memory.query_objects(
    pid,
    ObjectQuery(
        namespace="project/research",
        type=ObjectType.OBSERVATION,
        tags=["source"],
        text="README",
        limit=10,
    ),
)
```

`namespace` defaults to the caller's process namespace. `name` selects an exact
namespace-local name. The `type`, required `tags`, and case-insensitive `text`
filters are ANDed; text search covers the namespace, name, title, summary,
tags, and a bounded payload preview. Results use deterministic recent-first
ordering. A missing `limit` uses the active `memory.query_limit`; an explicit
limit must be an integer from 1 through that configured maximum. It is
validated before the namespace scan.

Querying requires `read` on `object_namespace:<namespace>`, and each returned
Object independently requires `read` on `object:<oid>`. Objects without that
authority are omitted. Returned handles carry only `read`; querying does not
grant `materialize`, `link`, or any write right.

Finite-use authority is settled while the handles are issued. A finite Object
read used for a result is consumed and yields a correspondingly finite derived
handle, rather than a reusable one. A full namespace query consumes the
namespace-read decision even when no Object matches because the scan itself is
the authorized operation. An exact-name query that finds no readable Object
returns an empty list without consuming the namespace-read use. Concurrent
attempts cannot turn one finite namespace use into multiple successful query
operations.

## Memory Views

Processes hold `MemoryView` objects that summarize which objects are visible as
goal, context, evidence, or result state. Fork can attenuate a parent view into
a child. Spawn creates a fresh goal-only view.

Ordinary process fork `mode="copy"` is not copy-on-write cloning. It may
derive a writable child handle to the same OID, so a child update changes the
Object the parent sees. Read-only `worker`/`restricted` modes are safer when
shared mutation is not intended. Checkpoint fork is a different operation: it
clones eligible owned reconstructable Objects and remaps their identities.

A view root is a borrowed reference unless the Object's explicit owner is that
process/subtree. Checkpoint restore uses ownership, not reachability, as its
destructive boundary: restoring a borrower does not roll back the lender's
payload, namespace, owner, or object capability. Checkpoint fork clones owned
reconstructable Objects, while `EXTERNAL_REF` roots and their capabilities are
dropped because a host handle cannot be cloned safely.

`MemoryView.filters` are applied during context materialization after
capability checks and before budget selection. Filters are ORed together; fields
inside one filter are ANDed (`type`, required `tags`, and bounded text search).
Filtered roots are audited as omitted objects.

`ObjectMemoryManager.merge_view` is low-level handle-derivation scaffolding: it
applies a merge policy and capability checks, but does not itself validate a
parent/child process relationship or adopt/release object ownership. Normal
process lifecycle callers use `ProcessManager.merge_child_memory`, which checks
the direct relationship around that scaffolding. When a non-root child exits,
its process-owned objects remain available for that parent to merge; the
lifecycle operation adopts merged child-owned objects into the parent and
releases unmerged child-owned objects. If the parent exits before merging,
terminal child-owned objects are released during parent cleanup.

When merge authority is finite-use, creation of the parent's derived handle and
consumption of the source grant are one store transaction. Failure to consume
the exact reservation removes the unpublished handle, so a failed merge cannot
leave durable authority behind.

`ObjectPatch()` leaves object payload unchanged. `ObjectPatch(payload=None)`
explicitly writes JSON `null` as the payload. Omitting `ObjectPatch.metadata`
preserves all current metadata; supplying it replaces the complete
`ObjectMetadata` value rather than merging individual fields. The manager then
recomputes the Runtime-owned `token_estimate` from the selected payload. Thus a
payload-only update preserves labels and descriptive fields while refreshing
the estimate, and a payload-plus-metadata update cannot retain an estimate for
the old payload.

## Object Links

Creating a link requires `link` on the source handle and `read` on the
destination handle. The manager holds the Object ownership transition lock and
one store transaction while it authorizes both exact handles, revalidates them
after the two-sided preflight, checks that both Objects are still `live`,
consumes finite-use decisions, and publishes the link/event/audit evidence.
Consequently, a capability revoke or Object release that linearizes first makes
link creation fail; it cannot leave a dangling link based on stale preflight
state.

## Object Tasks

An Object can own background tool tasks. A task records its owner oid, creator
pid, dedicated runner pid, tool, status, result oid, wait information, and
notification state in the runtime store, but it does not persist full tool arguments.
Arguments continue through the existing bounded, redacted tool audit path.

The runner is a child process whose tool table is narrowed to the requested
tool. Starting a task requires the creator to hold `read`, `write`, and `link`
rights on the owner object, `process:spawn` `write`, available per-object and
global ObjectTask concurrency slots, and the requested tool must already be
visible in the creator process tool table. External capabilities are inherited
only when explicitly delegated. Start validates this envelope and visibility,
then returns a queued task; the runner applies the target tool's argument schema
asynchronously through ToolBroker. Invalid target arguments therefore appear as
a terminal task failure rather than a synchronous start rejection.

Objects are immutable by default, so their initial owner handle does not include
`write`. An Object intended to own an ObjectTask must be created with
`immutable=False` or the creator must obtain an appropriate write handle by an
authorized lifecycle path.

ObjectTask rows are persisted, but active task execution is runtime-local. When
a runtime reopens an existing store, unfinished tasks are marked `abandoned`,
their runner processes are terminalized, and owner pins are cleaned up. The
original tool arguments are not persisted for replay. Ordinary Object payloads
are runtime-only as well. If a persisted `succeeded` task refers to a result
Object whose payload cannot be reconstructed on reopen, startup changes the
task to the explicit terminal status `result_unavailable_after_reopen`, clears
the live `result_oid`, and preserves the previous status and oid in wait
metadata. Checkpoint/image reconstruction is different because those formats
explicitly capture the payloads they promise to restore.

Successful tasks keep the tool result as a new Object Memory object and link
the owner object to it with `PRODUCED`. That link is part of the start-time
ObjectTask operation, so one-time owner handles are consumed at start and are
not re-used during completion. The source object payload is not rewritten.
Active tasks pin their owner object so process-exit cleanup cannot release it
before the task reaches a terminal state; once the task is terminal, normal
process-owned memory cleanup can release an owner whose creating process has
already exited.

A terminal notification carries a `result_oid`, but the message is not itself
Object authority. On success, a live creator receives a result handle with
`read`, `materialize`, and `link`; a different notification recipient does not
receive a new result handle by default. `grant_result_to_notify=True` opts into
publishing the same rights to a related recipient (the creator itself, its
parent, or one of its direct children). For a cross-process recipient the
creator must hold `grant` on the produced Object. A missing `grant` decision
fails the ObjectTask rather than silently publishing an unreadable result. The
recipient's Task Authority data-flow domain must also accept the result labels;
a label-domain denial suppresses the optional handle and normally makes the
notification fail without retracting an otherwise successful task.

Finite-use `grant` authority is reserved before terminal publication. The
recipient handle, durable `succeeded` transition, delivered terminal message,
and reservation commit are then settled under the Object ownership lock and
one store transaction. If delivery is not confirmed, the recipient handle and
transactional changes roll back, the exact still-live reservation is restored,
and the task may still publish success without the optional cross-process
handle. Cancellation or any earlier failure follows the same revoke-safe
restore and releases an unpublished result; an explicit revoke still wins and
cannot be undone by cleanup. The resulting recipient handle is an ordinary
durable capability; it is the creator's finite-use `grant` decision, not the
new handle, that is consumed once.

The `grant_result_to_notify` start option is runtime-local rather than part of
the durable ObjectTask row. This does not create a replay gap: an unfinished
task is abandoned on ordinary reopen instead of being re-executed. A recipient
handle that was atomically committed before shutdown remains an ordinary
durable capability.

Task start transactionally reauthorizes the complete owner-right decision set
before reserving any finite use, then creates and bootstraps its runner. The
durable task row, operation link, all owner-reservation commits, and start
event/audit publish in one later transaction; a rejected reservation commit
rolls that publication back. On a failure before this commit, the runtime
attempts to remove the runner and restores the exact owner reservations only
after that cleanup is confirmed. If cleanup cannot be confirmed, it leaves
those reservations fail-closed and records diagnostics instead of reactivating
authority alongside a residual runner; diagnostics are best-effort if the
audit sink itself is unavailable. The independent `process:spawn`
admission use is consumed immediately and is not refunded by either path. An
executor handoff failure occurs after the authorization commit, so it marks the
task failed, removes the unstarted runner, and does not refund the consumed
owner use. Result creation and linking use a lifetime scope: failed wiring or a
cancellation that wins the terminal transition releases the unpublished result
and derived handles, terminalizes the runner, and releases the owner pin. Once
the succeeded row is durable, later observability failures do not retract the
published result.

The creator can inspect and control its own task without a second owner-object
grant. For another process, `get` and `wait` require owner-object `read`, list
filters to rows visible with `read`, and `cancel` requires `write` after terminal
and unsafe-cancellation preflight. A finite read selected by `get`/`wait` is
claimed once; list claims each distinct finite read used by the returned rows
at most once, and its internal filtering does not spend authority for omitted
rows. Wait polling does not claim the same read repeatedly. Updating an owner
watch also requires owner-object `write` for a non-creator.

Runtime-internal multi-step writes can use an Object Memory lifetime scope.
Objects created in the scope are released automatically unless the scope is
committed or the object is transferred to another owner. This is used for
operations such as tool result creation where a later lifecycle step can still
fail after the Object has been allocated.

An ObjectTask can opt into owner watches. The Object Memory update, append, and
link primitives notify active watching tasks after the object change and audit
record are committed. Notices go to the runner process message queue, use the
`object-task-owner` channel by default, and carry only references such as owner
oid, version, event id, relation, and destination oid. A notice is not a
capability grant; the receiver still needs normal Object Memory authority to
read or materialize any referenced object. Owner-watch messages can resume a
task blocked in `receive_process_messages`; tools with arbitrary side effects
before a message wait are not replayed automatically.
The ObjectTask manager also watches ordinary process messages delivered to a
runner and terminal child-process notices. Those events can resume waiting
tasks only for tools with explicit replay-safe semantics, currently
`receive_process_messages` and `wait_child_process`.
`watch_object_task_owner(enabled=false)` disables future owner notices for an
active task. It does not remove notices already in the runner mailbox, cancel
the task, or change Object/target-tool authority.

## File/Object Bridge

Bridge tools can move content between workspace files and Object Memory without
returning full file content as a process-visible tool result:

- `create_object_from_file` reads a workspace file through the filesystem
  primitive and creates an object. The resulting Object Memory payload is
  checked against the memory payload hard limit before creation; callers must
  opt into truncation for oversize file objects. Its advertised `max_bytes`
  default and maximum are bounded by both the Object-file settings and the
  filesystem primitive's read hard limit; under defaults the effective ceiling
  is 1 MiB even though the separate Object-file absolute limit is higher.
- `write_object_to_file` materializes object payload into a workspace file
  through the filesystem primitive.

This is useful for large content movement and reduces accidental prompt
exposure. It does not bypass filesystem or object capabilities.
`write_object_to_file` accepts only a string payload or a mapping with a string
`content` field; other payload shapes fail instead of being guessed.

## Data labels and provenance

Object metadata carries `sensitivity`, `trust_level`, `integrity`, `origin`,
`tenant`, `principal`, and optional `declassification_authority` labels.
Provenance records parent Object ids and explicit source operation ids. Derived
objects conservatively take the highest source sensitivity and the lowest
source trust/integrity; mixed origins are marked `derived` and mixed tenant or
principal identities are not silently collapsed.

An update that lowers sensitivity or raises trust/integrity requires explicit
`admin` authority on `declassification:object:<oid>`. Context Materialization
Manifests and Explain expose labels alongside ids/hashes without copying Object
payloads. Finite declassification authority is consumed atomically with the
Object update, so a one-shot downgrade grant cannot be reused. Removing or
replacing tenant/principal, changing declassification authority, or raising
integrity/trust is the same privileged transition. On creation, the standard
model-facing memory tool may declare sensitivity, tenant/principal, and only
the conservative `untrusted` or `unknown` trust/integrity values. It cannot
request a privileged transition, elevate trust/integrity, or set a non-empty
declassification authority; see [Data Flow](data_flow.md).

For an LLM-derived Object, explicit `parent_oids` supplement rather than replace
all Objects included in the active context materialization. Every parent must
exist and be readable. The runtime resolves their exact versions and hashes;
a benign explicit parent cannot wash labels from other observed context.

These labels now constrain runtime-mediated egress. ToolBroker carries the
trusted `DataFlowContext` internally, and LLM, Human, JSON-RPC, MCP,
filesystem-write, Shell, and PTY primitives check Host Sink clearance and
revalidate source versions before dispatch. See [Data Flow](data_flow.md).

## Context Materialization

The LLM executor materializes prompt context from the process Memory View.
By default it uses that selected materialization unchanged: it does not create
or append an additional context Object. A metadata-only manifest still records
the exact selection and rendered hash for evidence.

Persistent context enrichment is an explicit Host/process opt-in. In that mode
the process has a mutable context object named
`<llm_context.object_name_prefix>:<pid>`. The prefix is configurable; its
default is `llm_context`, so the conventional name is `llm_context:<pid>`.

The runtime then appends new process facts and summaries to the end of that object so
repeated prompt prefixes remain stable for prompt caching.

Context state has two persistence planes. The live context Object plane
contains the runtime-local payload, append-only `entries`, captured-delta
signatures, and its working OID/version. The Object row and metadata are
durable evidence, but that OID does not remain a live, materializable context
when its runtime payload is lost. A separate durable per-process row stores the
provider-chain context generation and conservative `label_history` high-water
mark. Appending tool results, retry/repair, parallel batches, Human/message
resume, compaction, checkpoint, fork, exec, and reopen cannot reset a
previously observed higher sensitivity or narrower identity evidence to
defaults.

On an ordinary persistent-store reopen, the startup payload sweep releases the
old context Object because its payload cache is gone. The next `ensure` creates
a new context Object with a new OID and freshly initialized entries; it does
not reconstruct the previous append-only entry history. The new Object imports
the durable label high-water mark, while the durable context generation
continues to scope eligible provider continuation state. Exact payload recovery
is available only through checkpoint/image mechanisms that explicitly capture
and restore it. A durable Context Materialization Manifest records the old
OID/version/hash as evidence, but does not contain enough prompt content to
rebuild that Object.

The `compact_process_context` tool is the explicit exception to the append-only
shape: after validation it atomically replaces older entries with one
`context_compacted` summary plus the configured recent verbatim entries.
For an explicitly enriched process whose Image selects `auto_compact`, context
preparation also measures the persisted JSON payload when the process holds
`context:maintenance/execute`. At the configured storage-compaction waterline
it ends the current quantum and dispatches this same tool through the normal
Capability, approval, child-wait, event, and audit path before appending more
facts. The default source-only path performs neither that measurement nor that
dispatch. This waterline is intentionally below the generic Object Memory
payload hard limit; it prevents an opted-in long session from failing in
context preparation before model-window pressure can be assessed. After a successful compaction,
the first safe post-maintenance projection becomes a persisted byte baseline;
automatic storage compaction is re-armed only after another configured
waterline of growth, capped immediately below the hard limit. This hysteresis
prevents compressor children and their summaries from immediately retriggering
the maintenance they just completed. The projection is still checked before
persistence, and a first post-compaction projection that already reaches the
hard limit fails instead of entering an ineffective retry loop.

Materialization budgets use the final rendered object text, not stored
`metadata.token_estimate`. Object creation, payload updates, file imports, and
append-style writes still refresh that estimate as advisory metadata, but stale
or attacker-supplied estimates cannot make enlarged rendered content fit under
the prompt budget.

For explicitly enriched LLM execution, the append-only configured context
Object render is the charged context. Source Object materialization selects the
input Objects, while only this opted-in path records runtime deltas; neither is
double-charged in the same quantum. The rendered context must fit both
the per-call materialization window and the cumulative materialization budget
before the model is called.

LLM context preparation now persists a metadata-only Context Materialization
Manifest. For each source Object it records oid/version/type, included or
omitted disposition, the exact selection reason, transformation
(`verbatim`, `compacted`, or `truncated`), token count, rendered hash, and data
labels. The
final context Object/version, context generation, effective budget, rendered
tokens, and hash are recorded as well. No extra Object payload or rendered
prompt text is copied into the manifest. Direct materialization returns the
same per-Object selection metadata in memory; durable rows are created when the
final LLM context is prepared. See
[explainable_operations.md](explainable_operations.md).

## Persistence Invariant

Object metadata and namespace directories are stored in the runtime store.
Ordinary object payloads are runtime-only; current SQLite and PostgreSQL writes
store a runtime-memory marker rather than the user payload in
`objects.payload_json`. The decoder retains a compatibility path for accepted
legacy full-JSON rows from older development builds; those rows remain durable
sensitive data and are not rewritten by the marker recovery sweep. See
[Runtime Storage](storage.md#transaction-model).

If a reopen cannot materialize a live payload cache for a marker row, the
storage-layer startup recovery sweep releases the object fail-closed instead of
treating the marker as user payload. This is the recovery-only release path
described above, not an invocation of the manager's online release finalizers.

Scoped checkpoint snapshots and image artifacts can explicitly capture object
payloads needed to reconstruct a process subtree, subject to configured
snapshot limits.

Successful file writes store a separate canonical path binding with content
hash, labels, exact source refs, and generation. If a provider may have written
bytes but returns an ambiguous error, the runtime also stores the intended
binding conservatively; a certified `ProviderEffectNotStarted` does not create
that binding. Missing parent directories automatically created by a file or
directory write receive the same binding. Recursive directory deletion reads
the complete active descendant binding set and approval fingerprint atomically,
rechecks the fingerprint before dispatch, then tombstones that set after
success. `create_object_from_file`
inherits that binding (or `normal/untrusted` for an unclassified external
file), and `write_object_to_file` supplies the source Object to the filesystem
egress gate. Out-of-band file changes do not automatically downgrade a known
high-sensitivity path; delete creates a tombstone/history record.

Root process-owned memory is released on process exit unless retained as the
final process result. Non-root terminal process memory is held only until the
direct parent merges it or exits.
