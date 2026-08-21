# Glossary

This glossary defines project terms as they are used by the current Agent libOS
runtime. Capitalization is meaningful for named runtime concepts such as
`Runtime`, `Host`, `Capability`, `AgentProcess`, and `TaskRun`; ordinary English
uses of the same words do not create authority or identify a persisted record.

For the full contracts, follow the links in each definition. For a reading path,
start at the [documentation home](index.md).

## Version map

Several independent version namespaces coexist. Never infer one from another:

| Namespace | Current value | What it versions |
| --- | --- | --- |
| Agent libOS product/package | `1.5.1` | Python package, aligned GUI package, release workflow, and current product contract |
| RuntimeStore schema | `7` | Persisted SQL store shape accepted by ordinary Runtime startup |
| GUI snapshot envelope | `3` | Same-build `GET /api/snapshot` response consumed by the bundled renderer |
| GUI JSON Schema registry | `2` | The deliberately partial registry in [`gui_api_schema.json`](gui_api_schema.json); it describes selected v3 snapshot/API shapes and confirmed mutations, not a complete REST API |
| MCP manifest | `1`, `2`, or `3` | Independent client manifest contracts; v1/v2 preserve governed Tool compatibility and v3 requires the exact `2026-07-28` protocol contract |
| Runtime-safety benchmark task | `1` | Checked-in benchmark task YAML input |
| Runtime-safety run output | `2` | One runner's persisted benchmark result/effect artifact |

Checkpoint snapshots, image artifacts, events, semantic records, migration
plans, and other payloads have additional local schema versions. A field named
`schema_version` must always be interpreted in the type and subsystem that owns
it. See [Storage](storage.md), [GUI](gui.md#api-contract-boundary),
[MCP](mcp.md), and [Benchmark](benchmark.md) for their separate contracts.

## Runtime and execution

### Agent libOS

The capability-controlled Python runtime in this repository. It coordinates
long-lived agent processes, Object Memory, tools, providers, policy, resource
accounting, persistence, and evidence. It is not an operating-system sandbox or
a hosted multi-tenant service. See [Architecture](architecture.md) and the
[Threat Model](threat_model.md).

### Host

Trusted application or operator code that constructs and controls a `Runtime`.
The Host selects configuration, providers, modules, model profiles, trusted
Sinks, and launch authority. “Host-only” means a surface is deliberately absent
from model tools, Skills, JIT syscalls, generic remote APIs, and ordinary process
authority. Host code is part of the trusted computing base.

### Runtime

One opened Agent libOS composition root and its managers, providers, scheduler,
store lease, and recovery lifecycle. In Python it is represented by `Runtime`.
A Runtime is not the same as an `AgentProcess` and does not imply that every
persisted process is runnable. See [Python API](python_api.md) and
[Runtime Model](runtime_model.md).

### TCB (trusted computing base)

The code, configuration, services, operating-system behavior, and administrators
whose correctness is assumed by a stated security guarantee. In Agent libOS the
TCB includes the Runtime, primitives, selected providers and trusted modules,
the store and its administrator, relevant OS containment, and Host composition.
A malicious TCB component can bypass Runtime mediation. See
[Threat Model: trusted computing base](threat_model.md#trusted-computing-base).

### AgentProcess

The long-running, schedulable, interruptible unit that holds an identity, goal,
image, tool table, narrower model projection, Object Memory view, capabilities,
resource budget, messages, children, and wait/outcome state. It is not a chat
thread or a `TaskRun`. See [Runtime Model](runtime_model.md).

### Task

A user-facing piece of work. This generic word does not by itself identify a
particular persisted runtime type. Use `TaskRun` for the durable Host supervision
record, `ObjectTask` for an object-bound background tool task, `MCP Task` for the
optional remote protocol extension, and `AgentProcess` for scheduled execution.

### TaskRun / Durable Task Run

A durable Host-supervised envelope around one root `AgentProcess` tree. It adds a
versioned goal, requirements, idempotent commands, an append-only ledger, safe
continuation points, retention policy, and restart recovery. It is not a workflow
DSL or another scheduler. See [Durable Task Runs](durable_task_runs.md).

### ObjectTask

A Runtime-managed background tool task bound to an Object Memory owner. It may
notify a process through durable messages, but its runner child is not exposed
as an ordinary schedulable `AgentProcess`. It is distinct from a `TaskRun` and
from an MCP remote Task. See [Object Memory: Object Tasks](object_memory.md#object-tasks).

### AgentImage

An immutable process boot definition: prompt behavior, default tools and Skills,
optional packaged JIT tools/workspace seed, model-profile id, declarations, and
module prerequisites. Image requirements do not grant live authority. See
[AgentImage Authoring](agent_images.md).

### Skill

A progressively disclosed instruction/resource package that may change a
process's prompt context and tool visibility. Activating a Skill is not a
Capability grant. See [Skills](skills.md).

### Tool

A named, schema-validated action exposed through ToolBroker. A complete process
tool table determines what can be called; the model projection is a potentially
narrower visible subset. Visibility never substitutes for primitive authority.
See [Tools and Deno/TypeScript JIT](tools_and_jit.md).

### JIT tool

An agent- or package-authored Deno/TypeScript tool validated and executed under
the JIT supervisor. It reaches Agent libOS only through the allowed syscall RPC
surface and still depends on primitive authorization. “JIT” means just-in-time;
Python JIT compatibility is not claimed.

## Authority and policy

### Capability

A durable typed authority statement binding a subject to a resource pattern,
one or more rights, an `allow`, `ask`, or `deny` decision effect, constraints,
lineage, expiry/use limits, and delegation/revocation state. Knowledge of a path,
tool, object, image, or endpoint id is not a Capability. See
[Capabilities](capabilities.md).

### Resource

The canonical typed identifier against which Capability rights are checked, for
example `filesystem:workspace:README.md`, `image:review-agent:v0`, or
`mcp:demo:forecast`. Resource syntax and matching are subsystem-specific and do
not themselves grant authority.

### Right

The requested operation class on a resource, such as `read`, `write`,
`execute`, `delete`, or `admin`. Authorization checks the exact resource/right
pair together with constraints, Task Authority, policy, data flow, budgets, and
approval where applicable.

### Capability effect

The decision behavior stored on a Capability: `allow`, `ask`, or `deny`. This is
not the same thing as a provider-backed external effect. Documentation should
say “Capability effect” when that distinction matters.

### Task Authority / Task Authority Manifest

The Host-authored launch ceiling for one task/process lineage. A
`TaskAuthorityManifest` records authorized and requestable capabilities,
provider-effect ceilings, approval policy, budgets, expiry, data-flow policy,
and image requirements. Only authorized entries compile into root capabilities;
requirements remain declarations. See [Task Authority Manifests](task_authority_manifest.md).

### Human approval

A typed Host/user decision in response to a pending question or permission
request. Approval can issue only the authority allowed by the current Capability
and Task Authority contracts; it cannot bypass data-flow clearance or resource
budgets.

## Information flow

### Data labels

Structured sensitivity, trust, integrity, origin, tenant, principal, and
declassification evidence carried by Runtime-mediated data. Derivation is
conservative: sensitivity rises, trust/integrity can fall, and incompatible
identities become mixed. See [Data Flow](data_flow.md).

### Sink

The stable destination identity used by the data-flow gate, such as an LLM
profile, Human channel, filesystem destination, Shell/PTY input, JSON-RPC
method, MCP tool, Git remote, or process receive domain. A Host-trusted Sink may
receive data within its clearance, but Sink trust grants no operation
Capability.

### Exact release

A short-lived, one-shot Host authorization to send one bound payload to one
conditional Sink under the exact trust generation, data identity, arguments,
and state expected by the release. It is narrower than a general Capability and
cannot be reused for another destination or payload.

### FlowGraph

The payload-free semantic data-flow evidence graph. It records entities,
activities, edges, monotonic label assertions, source references, and coverage
without copying the underlying content. It is evidence for assessment and review,
not an authority graph. See [Semantic Approval and Data Identification](semantic_shadow.md).

## Effects and evidence

### Primitive

A trusted Runtime boundary that validates typed inputs and enforces the acting
process identity, Capability, Task Authority, policy, approval, data flow,
resource accounting, provider-effect intent, event, and audit rules applicable
to an operation. Model-facing tools are ergonomic wrappers; primitives enforce
authority.

### Provider

A Host-selected implementation of an external resource operation, such as local
filesystem, Shell, Git, Human I/O, JSON-RPC, MCP, or LLM transport. Providers do
not decide model/process authority and run only after the owning boundary admits
the operation. See [Provider Substrate](providers.md).

### Operation

A Host-visible causal record representing one logical Runtime action across its
LLM, Tool, syscall, primitive, Capability, Human, provider, resource, event, and
audit stages. An operation tree explains relationships; its existence does not
authorize a future action. See [Explainable Operations](explainable_operations.md).

### External effect

The durable pending/finalized classification of a provider-backed interaction
that may observe or change state outside the transaction. One logical protected
operation can have several ordered provider phases but preserves the required
effect identity and settlement rules. This meaning is distinct from a
Capability's `allow`/`ask`/`deny` effect.

### Evidence

Durable or bounded records showing what the Runtime checked, attempted,
observed, or settled: operations, events, audit rows, effect transitions,
receipts, hashes, and context manifests. Evidence supports explanation and
recovery; it is never an authorization credential and is not tamper-proof
against a direct store administrator.

### Event

A durable typed Runtime observation with an id, event type, producer-selected
source/target, timestamp, bounded payload, and optional correlation id. Events
are not a task queue or an authority source. See [Runtime Events](events.md).

### Audit record

A durable decision/outcome record emitted at a reviewed boundary. Audit history
is append-only through RuntimeStore APIs, but the Runtime does not claim
tamper-proof evidence against database administration.

### Receipt

A bounded result or identity/status projection used to bind, reconcile, or
explain a completed or ambiguous action. A receipt can prove what a Runtime path
recorded under its trust assumptions; it cannot grant authority or replace the
authoritative live state check.

## Memory, remote protocols, and concurrency

### Object Memory

The typed object store and namespace model used for goals, results, context,
artifacts, links, and external references. Metadata can be durable while payload
availability depends on the object, retention, and Runtime lifecycle. See
[Object Memory](object_memory.md).

### MemoryView

A bounded process context selection over Object Memory. A view determines which
objects may be materialized into context; it does not transfer ownership or
resource authority.

### MCP

Model Context Protocol. Agent libOS implements a client-only, Host-registered
surface with distinct Manifest compatibility contracts. It does not expose an
MCP server. See [MCP Client](mcp.md).

### MRTR (Multi Round-Trip Requests)

The MCP `2026-07-28` pattern in which a server returns an
`InputRequiredResult`, the client gathers the requested input, and the client
retries the original request with matching responses and any opaque
`requestState`. Agent libOS exposes only its governed Host continuation path;
v1/v2 model-facing Tool compatibility does not reinterpret an input-required
result as an authorized continuation. See [MCP lifecycle](mcp.md#subscriptions-mrtr-tasks-and-oauth-lifecycle).

### CAS (compare-and-swap)

A conditional update that succeeds only if the persisted revision, generation,
digest, token, or state still equals the caller's expected value. CAS prevents a
stale caller from overwriting newer state; it is concurrency control, not user
authentication or a cryptographic signature.

### Epoch

A monotonic Runtime or policy generation used to fence stale workers and bind
decisions to the active configuration/state generation. An epoch is not wall
clock time and is not an authority credential by itself.
