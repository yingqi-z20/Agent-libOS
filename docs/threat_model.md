# Threat Model

## Overview

Agent libOS is a local agent runtime with an optional Electron management
console. It runs long-lived `AgentProcess` instances that may use LLMs, Object
Memory, filesystem and Git operations, shell or PTY processes, Skills,
Deno/TypeScript JIT tools, registered JSON-RPC endpoints, registered MCP
servers, checkpoints, images, and Human approval. It is a runtime authority and
information-flow boundary, not a hosted multi-tenant service or a replacement
for an operating-system sandbox.

The intended security model has two independent gates:

```text
authority:        process identity + capability + Task Authority + primitive
information flow: data labels + Host-owned Sink trust + exact release
```

Tool visibility, prompt instructions, a Skill, an image requirement, a JIT
syscall, or knowledge of a resource identifier is not authority. A
runtime-mediated payload exit needs both the ordinary operation authority and
the applicable data-flow clearance. The detailed contracts are in
[Capabilities](capabilities.md), [Data Flow](data_flow.md), and the
[Protected Operation SDK](protected_operation_sdk.md).

This is a living model for the current implementation. Platform and provider
claims are limited by the [Support Matrix](support_matrix.md); historical design
documents and benchmark results do not widen the boundary described here.

## Threat Model, Trust Boundaries, and Assumptions

### Assets

The runtime is intended to protect:

- Host workspace files, the fixed Git repository, managed worktrees, refs, and
  configured remote operations;
- credentials and environment-backed provider secrets, including LLM,
  JSON-RPC, MCP, Git, and database credentials;
- Object Memory and prompt context, process messages, Human answers, provider
  inputs and outputs, and their tenant/principal labels;
- Durable Task Run goals, follow-ups, resume payloads, requirements, command
  receipts, and the integrity of their generation/epoch fences;
- external systems reachable through LLM, Human, JSON-RPC, MCP, Git, Shell,
  PTY, filesystem, and other provider-backed operations;
- process authority: capabilities, finite-use reservations, Task Authority
  Manifests, Human approvals, Sink trust, and conditional releases;
- availability budgets for model tokens, tool calls, subprocess CPU/RSS/wall
  time, external bytes, and child processes; and
- the integrity and causal identity of runtime state, audit/events,
  external-effect intents, operation trees, checkpoints, images, and recovery
  decisions within the RuntimeStore trust boundary.

### Untrusted and operator-controlled inputs

The following inputs are treated as untrusted content, even when they arrive
through an authorized channel:

- model output, prompts, user task text, retrieved content, workspace files,
  Git content, process messages, and data returned by remote providers;
- workspace and global Skill instructions, metadata, resources, action
  descriptions, and bundled JIT source. Global package trust binds an exact
  package hash; it does not turn the Skill's prompt text into authority;
- Deno/TypeScript JIT source and syscall arguments;
- JSON-RPC results, MCP tool metadata/results, LLM responses, PTY or shell
  output, Human free-text answers, and errors derived from providers; and
- identifiers and arguments selected by an `AgentProcess`, including attempts
  to request new permission or target a known resource.

An attacker may control any combination of those inputs and may induce the
model to call every visible tool repeatedly. A compromised process may exercise
all authority actually delegated to it; use within that authority is not an
authorization bypass. The runtime must prevent that process from acquiring or
using additional authority implicitly and must preserve labels across its
mediated handoffs.

The Host/operator controls runtime configuration, root capability issuance,
Task Authority Manifests, Sink rules, endpoint/server manifests, image and
module trust, provider selection, environment-secret mapping, database targets,
and Human approval decisions. Those values are policy inputs, not model
assertions. A malicious or compromised Host can deliberately authorize the
effects that the runtime otherwise denies and is outside the adversary model.

### Trusted computing base

The security properties below depend on this TCB:

- the Python runtime core, especially primitives, Capability and Task Authority
  managers, data-flow enforcement, the Protected Operation SDK, process and
  publication lifecycle, resource accounting, and recovery code;
- Host/admin configuration and bootstrap code, trusted capability issuers,
  Sink-registry administrators, and the Human operator when an approval or data
  release is accepted;
- for GUI deployments, the Electron main and preload code, React renderer,
  packaged assets, and their Node/Electron dependency supply chain. The preload
  passes the authenticated connection, including its bearer token, to the
  renderer; compromising renderer content therefore compromises the GUI
  Host/admin surface even though Node integration is disabled;
- trusted Runtime Modules and provider implementations. A module entrypoint is
  in-process Python and may register tools, syscalls, images, provider hooks,
  and lifecycle hooks. Module hash checks select TCB code; they do not sandbox
  it. An injected provider is trusted to honor its interface, accurately report
  whether a phase started, and avoid bypassing shared primitive/SDK gates;
- the RuntimeStore backend and its transaction/lease implementation, plus the
  integrity and access controls of the SQLite file or PostgreSQL
  database/schema. Runtime API append-only and CAS rules are not protection
  against a direct database administrator;
- the Host OS kernel, filesystem/process permissions, Python and native
  dependencies, Deno binary and supervisor, system Git, certificate store,
  resolver, and any isolation substrate the deployment claims; and
- transport libraries and configured remote-service identities. A remote
  response is still untrusted ingress, but TLS authentication, provider
  receipts, and effect reconciliation necessarily depend on their respective
  Host-selected implementations and services.

Runtime `admin` is a scoped capability right, not a claim that its holder is
outside all policy. An actor granted `admin` can perform the covered mutations;
the trusted decision is the Host's issuance of that authority. Operating-system
administrators, code executing inside the Python process, and actors with
direct database or process-memory access are stronger than runtime capabilities
and are inside the TCB.

### Principal trust boundaries

| Boundary | Runtime treatment | Residual trust or limitation |
| --- | --- | --- |
| Model/task/prompt → AgentProcess | Untrusted instructions and arguments; only visible tools can be selected | Prompt injection is expected and is constrained, not solved |
| Tool/Skill/image → primitive | Visibility and declarations do not grant authority | Primitive and manager correctness are TCB |
| JIT TypeScript → libOS syscall | Deno has no read/write/net/env/run/FFI permission, rejects imports, and reaches effects only through syscall RPC | Deno, its supervisor, and OS process containment are TCB; resource exhaustion is bounded, not impossible |
| Process → process/Object/message | Capabilities and process lineage gate access; labels propagate and the receiver's Task Authority receive-domain is checked | An authorized receiver may use all data it receives under its own authority |
| Runtime → filesystem/Git/native child | Canonical resources, policy, state/generation checks, budgets, and provider intents precede mediated effects | A launched native program is a Host-user process, not syscall-sandboxed by Agent libOS |
| Runtime → LLM/Human/JSON-RPC/MCP/remote Git | Registered or Host-configured identities, operation authority, Sink clearance, bounds, and durable effect state | The recipient's later storage, forwarding, and behavior are outside the boundary |
| GUI/CLI → Runtime | Host-facing control surface; actor mode uses process authority, while omitted actor may select Host/admin mode | Bearer-token or local Host compromise is an operator-boundary compromise |
| Runtime → RuntimeStore | Transactions, CAS/generations, recovery fences, and one writable Runtime lease | Direct SQL/file administration can forge, delete, or alter evidence and policy |
| Trusted module/provider → Host | Full TCB extension selected by Host configuration and source trust | Malicious TCB code can bypass runtime mediation entirely |

### Native children, PTY, and MCP stdio

Shell, PTY, and MCP stdio are local process-launch boundaries. The runtime
mediates the executable identity, argv, selected environment, cwd, authority,
Sink clearance, supported resource controls, and provider-effect record.
Process-tree supervision is backend-specific: the local POSIX Shell/PTY paths
provide it, while the current Windows Shell and ConPTY providers reject a
budgeted `SubprocessLimits` dispatch rather than claim enforcement they do not
have. Mutable workspace Shell/PTY executables and every local MCP stdio
executable are dispatched from a verified Host-owned content snapshot; that
does not attest their dependency or plugin trees. Linux binds an explicit MCP
subdirectory cwd through a stable directory handle. Other local platforms
reject that optional cwd and use the Host-owned workspace root rather than
claiming race-free subdirectory containment.

After launch, a native executable runs with the operating-system rights of the
Host user. Agent libOS does not apply a kernel syscall allowlist to it and does
not claim to control its direct filesystem, network, credential-store, child
process, or secondary IPC activity. Where supported, PTY supervision and
process-tree cleanup are availability/lifecycle controls, not hostile-code
isolation. Trusting a Shell, PTY, or MCP stdio Sink means the Host authorizes
that executable to receive the delivered bytes. Hostile native code requires a
container, VM, WASM, service provider, or comparable deployment boundary.

Deno JIT is narrower: it is launched no-permission and cached-only, imports are
rejected, and its libOS effects use syscall RPC under the caller pid. A trusted
Runtime Module that adds a syscall handler remains TCB code; the fact that an
untrusted JIT caller invokes it does not sandbox the handler.

### Remote Sinks, DNS, TLS, and proxies

Models cannot supply arbitrary JSON-RPC or MCP URLs at call time. The Host
registers closed manifests and environment-backed credential mappings. Remote
JSON-RPC and Streamable HTTP MCP default to HTTPS; plain HTTP is limited to
local development hosts. The implementations reject unsafe literal/resolved
addresses, bind validated addresses to dispatch, retain the original Host/TLS
server name, and do not follow redirects. DNS is itself a provider
information-flow phase, so an observation or ambiguous failure is recorded and
does not regain finite authority merely because no application request was
sent.

Manifest v2 adds a bounded modern MCP discovery/negotiation phase. Its
configured protocol mode is part of immutable registry/Sink identity; the
negotiated revision and advertised server capabilities are untrusted
operation-local observations and cannot grant authority. Automatic fallback is
limited to protocol-recognized legacy signals, never authentication errors,
server failures, malformed/oversized replies, or ambiguous transport failure.
The client does not advertise Sampling, Roots, Elicitation, subscriptions,
Tasks, or extensions, and rejects reverse requests without invoking Runtime
behavior. An MRTR input request is non-retryable and preserves unknown mutation
evidence rather than creating a model-controlled replay path.

MCP protocol/session/content-negotiation, trace, baggage, and reserved `_meta`
fields are Host-generated. Manifest header matching is case-insensitive and
cannot override them. Ambient OpenTelemetry context is cleared at this adapter
boundary; the release installs no exporter and does not claim OTel propagation
support. Static environment-backed Authorization values remain a Host transport
choice, not an OAuth implementation or a source of Runtime capability.

These controls reduce SSRF and DNS-rebinding risk; they are not a private
network firewall or a substitute for TLS and resolver integrity. TLS relies on
the Host certificate store and server-name validation. MCP HTTP disables
environment proxy inheritance, while LLM SDKs, system Git, custom providers,
and deployment networking have their own proxy/TLS behavior. Operators must
review CA roots, DNS, proxy settings, Git credential helpers, custom LLM base
URLs, and injected transports for the deployment. A Host-configured private or
custom LLM endpoint is a trusted routing decision, not model-controlled SSRF.

Sink trust authorizes only the immediate, canonical recipient. It does not
assert that the recipient is benevolent, erase returned ingress labels, or
control onward forwarding, logging, retention, webhook calls, plugins, or
other secondary effects. Clearance above `normal` should therefore be granted
only when those downstream behaviors are acceptable to the Host.

### Local concurrency and external mutation

Scheduler workers inside one Runtime share lifecycle admission, hierarchical
path or registry locks, transaction boundaries, revision CAS, state tokens,
and generation revalidation. Persistent stores also take an active-runtime
lease: the supported contract allows one writable Runtime per SQLite target or
PostgreSQL database/schema, not concurrent multi-Runtime writers.

An unrelated local process may still edit workspace files, run Git directly,
change provider state, or—if OS permissions allow—access the database. File,
Git, executable, registry, Object, and Sink paths revalidate the state that
their contract binds and fail closed or retain an `unknown` outcome when a
race is detected after a provider phase. Those mechanisms are not a global
transaction over the host. They do not prevent a same-user process from direct
I/O, protect confidentiality from that process, or make direct database writes
trustworthy. Deployments that treat other same-user processes as hostile need
OS-account, ACL, namespace, container, or VM isolation.

### Guarantees, assuming the TCB holds

- A runtime-mediated protected operation cannot obtain authority merely from
  tool visibility, a Skill, JIT code, an image declaration, an endpoint id, or
  model-authored policy. Typed capability, Task Authority, operation policy,
  Human approval where applicable, and budget checks remain independent.
- Deny policy and delegation attenuation are preserved; finite-use authority is
  reserved and settled with the durable operation boundary rather than being a
  replayable decision object.
- Runtime-mediated egress declares canonical primary and additional Sinks.
  Labels, identity domains, Sink-registry generation, source versions, payload,
  and exact conditional releases are revalidated before dispatch. A Human
  cannot elevate an untrusted Sink above `normal`.
- Filesystem and Git operations are scoped to the configured workspace/repository
  surfaces and use operation-specific canonicalization, containment, state, and
  locking rules. Remote Git operations use existing Host configuration rather
  than model-supplied URLs.
- JIT code cannot use ambient Deno filesystem, network, environment, process,
  or FFI permission. Its mediated syscalls still require the caller's primitive
  authority.
- Provider work has a durable intent and explicit phase/effect outcome. A
  certified no-start can restore eligible authority while no completed phase
  has mutated state, observed information, or committed authority. After such a
  phase, or after an ambiguous provider start, uncertainty remains durable and
  startup does not blindly replay an unknown effect. This is not a distributed
  transaction with the provider.
- A split-phase Task Run mutation commits a request- and generation-bound
  provisional command receipt before a lost response can be interpreted as a
  fresh dispatch. Its strict version-1 result has an exact variant schema,
  complete revision-bound public summary, raw pre-decode byte cap, and signed
  BIGINT bounds. Local-control pending and completed variants reference a
  same-transaction append-only status-transition ledger item and canonical
  evidence digest binding the Run, command identity/request hash, from/to state,
  and semantic revision/generation fence. Terminal or superseded early returns
  validate that evidence before returning. Exact replay settles only durable local state and never
  repeats scheduler, LLM, Tool, Provider, or external-effect work. A committed
  verified effect receipt is not verified again merely because completion of
  the public command result was lost; replay instead requires its exact
  append-only finalized-effect transition and matching
  `host_verified_receipt` audit, which remain after provider-body purge. All
  command inserts/result updates, including linked-gap repair, are fenced by a
  global Runtime-epoch counter-row lock. Startup first settles a staged complete
  provider result, then auto-settles only a well-formed current-generation
  pending interrupt for a recoverable nonterminal Run. It cannot start queued
  work or resume an unrelated Host pause; ordinary effect, resume-point, and
  finalization gates still apply. Corrupt, duplicate, or over-bound command
  recovery evidence produces `needs_attention` instead of truncated or guessed
  execution. Terminal command-result loss requires exact local client replay.
- Interrupt recovery recognizes only the Store-reserved typed
  `StaleExecutionProcessWait`, never the `stale_execution_recovery`
  `status_message`. Its prior owner/lease and recovering Runtime owner-id are
  projected only as canonical SHA-256 identity hashes; the recovering-owner
  hash is not a cryptographic signature and relies on RuntimeStore/database
  administrator integrity inside the TCB. A historical recovering-owner hash
  may survive another reopen, but it grants nothing by itself: the current
  Run/process epoch, the interrupt admission Runtime epoch and exact per-PID
  state/execution-generation fence, an identity- and integrity-bound complete
  safe point, no live owner/lease, and the current Image/tool/provider binding
  must all agree. Prior raw owner/lease tokens do not enter the stale wait,
  TaskRun ledger, summary, or error projection. Checkpoint restore and fork
  replace the non-transferable receipt with an ordinary pause and clear its
  compatibility message before publishing a new concurrency identity.
- A linked recovery whose nested rerun committed before its outer receipt does
  not use startup or current source state to guess whether to create again. An
  exact outer retry can copy the immutable target result only after validating
  a request-hash-bound nested command, source/target receipts, target Run, and
  unique causal link. Missing, changed, duplicate, or malformed evidence fails
  closed without a second Run.
- Checkpoint restore/fork revalidates current authority and reconstructs only
  scoped runtime state; it does not silently claim to roll back external state.
- Audit and effect history obey append/CAS/transition rules through Runtime APIs
  and remain causally linked to operations within the RuntimeStore boundary.
- A cooperating stale Runtime cannot publish Task Run claims, child processes,
  resume points, or terminal settlement after the active-store lease and
  monotonic Runtime epoch have advanced. Unknown or dispatched external
  effects block automatic Run continuation rather than being treated as safe
  retries.

### Explicit non-goals

Agent libOS does not claim to:

- eliminate prompt injection, malicious tasks, deceptive tool output, model
  mistakes, or incorrect but authorized actions;
- provide kernel-grade isolation for Shell, PTY, MCP stdio, system Git, or any
  other native executable;
- constrain trusted Runtime Modules, injected providers, the Host/admin plane,
  direct process-memory access, or direct RuntimeStore administration;
- control a Sink's secondary forwarding or a native child's direct I/O after
  the mediated delivery/launch boundary;
- make irreversible provider effects rollbackable, guarantee exactly-once
  delivery across a remote system, or infer success from an ambiguous crash;
- provide cryptographic tamper evidence, at-rest encryption, automatic secret
  deletion, or data-loss prevention outside runtime-mediated Sinks;
- snapshot or restore filesystem, Git/remote, network, native process, or other
  provider state through a checkpoint or checkpoint-derived image;
- support mutually untrusted writable Runtimes on one store, or isolate hostile
  processes sharing the same OS user; or
- provide a distributed workflow service, multi-host Run failover, exact LLM
  physical-request/provider-billing/monetary-spend caps, or automatic recovery
  of runtime-local ObjectTasks; or
- turn deterministic tests and benchmarks into a formal proof or a claim that
  every supported OS/provider environment has been release-validated.

## Attack Surface, Mitigations, and Attacker Stories

### Model-visible self-evolution

An adversarial prompt can ask the model to activate a Skill, expose more tools,
register JIT code, load an image package, fork a child, request permission, or
call a known remote id. These actions may change prompt or tool visibility, but
the resulting primitive still runs under one process identity and current
authority. Task Authority bounds model-requested expansion, and model requests
cannot install broad privileged authority merely by asking a Human.

Skill packages are bounded and snapshotted. Truncated `SKILL.md`, extension
metadata, JIT source, or optional resources are rejected rather than hashing a
prefix. Global Skill trust binds the complete package hash. Skill instructions
remain prompt-injection content; only trusted Runtime Modules execute Python in
the TCB.

### Capability and approval abuse

Relevant attacker stories include resource-pattern confusion, subject
confusion, deny/allow precedence errors, delegation widening, one-shot replay,
approval reuse after argument or target-state change, and registry-oracle
leaks. Resources are typed and canonicalized, wildcard shapes are restricted,
deny dominates, delegation attenuates parent authority, and Human operation
approval binds the exact effect/arguments and optional state. Registry call
paths gate on caller authority before loading endpoint or tool metadata.

These controls cannot make an intentionally broad Host grant least-privileged.
Capability and Sink configuration should be reviewed as security policy, and
Human approval UI must not be treated as protection against a compromised
operator.

### Data exfiltration and cross-domain flow

A malicious model may copy a secret from Object Memory, a file, Git ingress,
another process, or a provider response into an LLM request, Human message,
remote call, file write, shell stdin/argv, or PTY/MCP stdio operation. The
runtime propagates sensitivity and tenant/principal identity through mediated
objects and process handoffs, then checks every declared recipient against one
Host Sink-registry generation. Conditional high-sensitivity delivery requires
an exact one-shot release tied to source versions, labels, payload, manifest,
Sink, operation, and target state.

The defense ends when an approved recipient receives the payload. Exfiltration
performed later by a trusted native executable, provider, module, remote
service, or database administrator is a TCB/deployment failure, not a bypass of
the mediated Sink gate.

### Filesystem, Git, and executable races

Attackers may use traversal, symlink/hard-link swaps, case/Unicode aliases,
mutable executable replacement, Git config/helper injection, repository-state
races, malicious patches, or worktree escape. The local implementations use
workspace containment, no-follow and inode checks where supported,
hierarchical file-label locks, byte-safe Git parsing, fixed repository and
managed-root rules, config validation, state tokens, executable identity
hashes, and pre-dispatch snapshots for mutable executables.

A concurrent external process does not participate in all runtime locks. A
detected race denies or becomes conservative evidence; an undetected direct
same-user mutation and all direct native I/O remain deployment concerns.

### Remote and provider boundaries

Attackers may try URL injection, SSRF, DNS rebinding, redirects, oversized or
malformed responses, schema drift, credential leakage, error-message leakage,
or a crash between remote dispatch and local settlement. JSON-RPC/MCP use
Host-registered manifests, public-address policy, bounded wire payloads,
environment allowlists, schema checks, no redirects, absolute deadlines, and
sanitized public errors. The protected-operation state machine preserves
unknown outcomes rather than retrying them as definitely not started.

Remote services may still return adversarial content, lie at the application
layer, retain data, or become unavailable. Provider receipts and reconciliation
are only as trustworthy as the selected provider/service; returned content is
ingress, not a new policy source.

Provider-returned reasoning may itself contain HTML, misleading links, Unicode
direction controls, secrets copied from prompts, or instructions aimed at the
operator. The GUI fetches this material only on demand for an authenticated
selected process and renders it as inert isolated text, never Markdown or raw
HTML. Snapshot/SSE frames contain summaries only. Retention and full-I/O opt-out
can make content unavailable; the GUI must not reconstruct it from raw response
fields, previews, hashes, or legacy metadata.

### Persistence, recovery, and management surfaces

The CLI and GUI can issue capabilities, approve requests, register code and
remote manifests, inspect sensitive history, and trigger destructive
operations. They are Host control surfaces. The GUI is loopback-only with a
bearer token that is randomly generated by default and restricted origins, but
the Host may override the token and is responsible for its strength. It is not
a separate process-effect boundary; token, renderer, dependency, or Host
compromise grants the corresponding control surface. Actor-mode requests
remain capability checked, while documented admin-mode routes are intentionally
Host-authorized.

The RuntimeStore is also a confidentiality boundary. With the default
`llm.persist_full_io: true`, it retains complete prompts, visible tools,
reasoning metadata, outputs, tool calls, and raw provider responses. A database
or backup reader can therefore see that material. Operators that do not need
lossless training/debugging records should disable full-I/O persistence and
apply the separately documented payload-retention policy; neither setting is
at-rest encryption.

Durable Task payload persistence is independently disabled by default because
useful restart recovery requires readable goals, follow-ups, transcripts, and
resume bundles. Enabling it is an explicit Host decision to store that material
as plaintext. The default `purge_on_terminal` policy reduces retained content
only during safe terminal settlement, including linked LLM/tool-output bodies
and linked terminal external-effect provider metadata/receipt bodies. It also
hash-reduces Run-linked Human prompt/response/decision bodies, then deletes
pending continuations and durable messages automatically bound from a
Run-member recipient. Ordinary callers cannot suppress, override, or forge the
message binding. Human request id, type, status, timestamps, audit linkage, and
content digests remain without readable Human content. Effect identity, state,
classification, content digests, and causal links likewise remain, but readable
provider receipt content does not. Nonterminal effects are never reduced. This
is not secure erasure and backups may retain earlier copies.
`permanent` retention is Host/admin-only and deliberately skips automatic
Run-terminal cleanup, while independent evidence retention may still apply; a
Host/admin may explicitly apply the same audited cleanup to a terminal
permanent Run later. A task/model cannot choose either the global
payload-persistence setting or upgrade itself to permanent retention.

RuntimeStore transactions, recovery fences, active-runtime leases, bounded
reconciliation, and exact operation/publication links address partial commits,
duplicate recovery, and stale writers. Checkpoints do not erase append-only
history or external effects. A direct database writer can defeat these
properties; deployments needing independent evidence integrity must export
signed or remotely append-only evidence outside that administrator boundary.

## Severity Calibration

Severity assumes exploitation begins from untrusted model/task/Skill/JIT,
workspace, or remote-provider input and does not already require control of the
Host/admin plane or another TCB component.

- **Critical:** a reliable primitive/Capability bypass that gives untrusted
  input arbitrary Host-user code execution or broad filesystem/credential
  access without granted authority; a data-flow bypass that sends `secret`
  cross-tenant data to an untrusted Sink without release; or a module trust
  bypass that loads attacker Python into the TCB. The same outcome after an
  operator deliberately loads an untrusted module is outside this calibration.
- **High:** escape from workspace containment; approval or finite-authority
  replay enabling a destructive/remote effect; model-controlled SSRF to a
  protected network target; executable or remote-identity TOCTOU that changes
  the recipient after authorization; or recovery that duplicates an
  irreversible provider effect after an ambiguous dispatch.
- **Medium:** unauthorized registry/manifest enumeration, bounded provider
  error leakage, cross-process access to non-secret data, loss of important
  audit/effect linkage without an unauthorized effect, or a repeatable
  process-level resource exhaustion that remains inside configured Host and OS
  limits.
- **Low:** diagnostics-only information exposure, isolated availability or
  usability defects requiring trusted Host configuration, and issues confined
  to an explicitly unsupported environment without weakening a supported
  boundary.

Findings that require malicious Host configuration, a trusted module/provider,
direct database administration, same-user arbitrary native execution, or a
Sink's later forwarding are normally out of scope for the runtime boundary.
They may still be severe deployment vulnerabilities if the affected deployment
claims those components are untrusted; that stronger claim must be defined and
enforced by its OS, container, network, or evidence architecture.
