# Provider Substrate Reference

The Resource Provider Substrate is the host-effect layer below Agent libOS
primitives. A provider implements a concrete filesystem, Git, clock,
subprocess, Human I/O, JSON-RPC, or MCP backend; it is not an authority bypass. Process
identity, Capability reservation, Human approval, Task Authority Manifest
ceilings, pending-effect persistence, resource settlement, events, and audit
remain primitive/runtime responsibilities.

The protocol types are defined in `agent_libos/substrate/base.py`. The default
composition is `LocalResourceProviderSubstrate` from
`agent_libos/substrate/local.py`; trusted Runtime Modules can register additional
provider hooks during startup. Every real provider boundary must use the
[`ProtectedOperationContract`](protected_operation_sdk.md) lifecycle.

## Current provider inventory

| Provider | Current backend | Authority and policy | Effect/evidence contract | Bounds and containment |
| --- | --- | --- | --- | --- |
| Filesystem | `LocalFilesystemProvider` rooted at one workspace | Typed `filesystem:<namespace>:<path>` rights; cwd and state probes occur only after authority | Reads are information-flow effects; successful mutations are `irreversible`/`not_supported` because the local provider records no preimage or undo log and exposes no compensation operation; mutations prepare one pending effect and finalize the same id; created parents inherit labels and recursive delete aggregates the bound subtree | Lexical resolution followed by no-follow containment, size/list bounds, safe write/delete handling |
| Clock | `LocalClockProvider` | Clock resource/right plus process resource budget | `now`, monotonic observation, sleep/asleep cancellation, and result classification use one protected operation; later failures remain unknown rather than refunding authority | Finite sleep limit, monotonic elapsed accounting, sync and async paths |
| Shell | `LocalShellProvider` | Exact executable resource, shell policy rule, approval, cwd read authority, process budget | Intent exists before spawn; timeout/cancel/limit/classifier failures conservatively retain performed/unknown evidence | No shell command string by default, scrubbed environment, workspace cwd, stdout/stderr bounds, wall/CPU/RSS supervision, process-tree termination |
| Git | `LocalGitProvider` pinned to the Runtime workspace repository | configured repository resource (default `git:workspace`), affected filesystem resources, per-remote `git_remote:workspace:<name>`, per-PR `git_pr:workspace:<id>`, state-token CAS, and mandatory approval for destructive operations | Reads are ingress; local mutations, fetch, push, and simulated-PR transitions use distinct protected-operation descriptors and local query-only reconciliation | System Git 2.26+, byte-safe parsing, repository/config identity validation, cross-process lock, no arbitrary argv/URL, executable extensions disabled, bounded output |
| Human | `LocalHumanProvider` plus GUI/terminal host surfaces | Typed question/permission request and explicit policy decision | Terminal read/write and GUI request presentation are protected information-flow operations; conditional GUI views expose only bound metadata and reject parent responses until their exact one-shot release is consumed through presentation | Typed responses, queue state, bounded payload/output, lock-free blocking I/O with claimed request state |
| JSON-RPC | `HttpJsonRpcProvider` | Registered endpoint and exact method capability; model-supplied URLs are forbidden | Registry metadata is gated before lookup; calls prepare the reservation/intent before resolving header values, then remote DNS starts inside that intent; transport/classification settles the same effect id | Closed manifest shape, header-env allowlist, request/response hard limits, timeout, resolved-address policy, client-only JSON-RPC 2.0 |
| MCP | `SdkMcpProvider` for Streamable HTTP and stdio | Registered server/tool capability; stdio additionally requires `process:spawn` and exact `mcp_stdio:<digest>` execute authority exposed by server inspection | HTTP DNS and stdio spawn are covered by the call/validation intent; tool calls prepare before runtime env resolution, while live tool refresh resolves env first; registry lookup is gated; live discovery is bidirectional and tool results use bounded provider evidence | Tools-only v1 surface, closed manifest shape, header/stdio-env allowlists, request/response and timeout limits, contained stdio process lifecycle |
| PTY | Trusted `modules/pty` Runtime Module provider hooks | Startup hash trust plus normal process/shell authority; published sessions are Object Memory `EXTERNAL_REF` handles with Object rights | Spawn is bidirectional; read/continuous ingest are ingress; write/resize/public close are egress. Effectful operations use protected pending-to-finalized evidence; write raises the session label high-water even after an ambiguous provider outcome | Independent reader/monitor workers, output bounds, wall/CPU/RSS supervision, POSIX process groups or Windows Job Object/ConPTY support |

LLM requests are also formal bidirectional protected provider operations. Their
Sink is `llm:<profile>` and profile/model/base-URL/API-mode plus effective
provider retention policy (`store`, prompt-cache retention, and Responses
chaining) is hashed into the trusted identity. Precheck and client construction
use one frozen Host snapshot, so an already-cached client cannot drift from the
identity being authorized. The returned provider content is treated as
unclassified `normal/untrusted` input and cannot lower the request context.

Every payload-bearing provider exit declares an explicit data-flow direction
and descriptors. Human provider I/O uses `human:<recipient>:<channel>`; GUI
projection uses `human:<recipient>:gui` while resolving the configured Human
trust identity. JSON-RPC uses
`jsonrpc:<endpoint>:<method>`, MCP uses `mcp:<server>:<tool>`, Shell uses the
resolved executable, and PTY sessions retain their spawn Sink identity. Sink
clearance is checked before provider state, DNS, stdio, or spawn and rechecked
inside the SDK prepare transaction. Cached MCP tool metadata is public; a live
refresh remains a provider operation. See [Data Flow](data_flow.md).

Supplying an already-constructed `LocalResourceProviderSubstrate` does not make
its Git settings independent of the Runtime configuration. During open, the
Runtime atomically binds the active Git settings to the Local Git provider and
the Shell raw-Git guard; conflicting settings or incompatible Local provider
subclasses fail startup before a partially bound substrate can be used.

Runtime environment resolution is not uniformly outside preparation. A
JSON-RPC call and an MCP tool call first reserve finite authority and prepare a
pending intent, then resolve header/stdio values before their first provider
phase; a missing value restores and abandons that preparation because no
provider work started. MCP cached tool listing resolves nothing, while live
`list_tools(refresh=true)` resolves its environment before reservation and
intent creation. The operation-specific references below are authoritative for
these orderings.

Git repository reads are unclassified ingress from `external:git`; local
mutations and fetch are bidirectional, push is egress, and simulated pull
requests are bidirectional repository state transitions. Every mutation binds
the current repository state token. Push also binds its remote/config/ref
fingerprint as mutable target state. Patch Objects preserve source labels
through application. See [Git Provider and Primitive](git.md).

Shell and PTY derive the resolved executable Sink from Host-owned `argv[0]`,
workspace/cwd, and the safe executable path without handing the remaining argv
to provider code. Full provider canonicalization is the first protected
information-flow phase after data-flow and ordinary-authority revalidation.
The canonicalized executable must resolve to the already authorized Sink, and
its regular-file content hash is recomputed inside the shared boundary before
each provider phase. A path or content mismatch appends a payload-free
data-flow denial, records any prior resolver observation conservatively, and
refuses to start the command or PTY process. Immediately before final dispatch,
mutable workspace executables are copied into a private Host-owned snapshot;
Shell, PTY, and MCP stdio execute that snapshot instead of reopening the
authorized path. The local MCP stdio provider uses the same resolved executable
for both its configuration hash and snapshot source. The pre-existing direct
sibling set is exposed up to a configured bound beside the private copy through
links to the original locations so scripts that read resources relative to
`$0` or `__file__` retain ordinary read-following behavior. The mirror is
all-or-nothing: enumeration, limit, symlink, or Windows hard-link fallback
failure aborts before final provider execution instead of silently omitting a
possibly required resource. Only the executable bytes are pinned. Sibling
content remains live and is not package attestation; `lstat`/`O_NOFOLLOW`,
parent-relative (`../`) layouts, creating or renaming beside the snapshot, and
executable plugin/package trees are outside this compatibility boundary. Such
providers must supply a stronger package/container substrate.
An injected stdio provider must expose that Host resolver contract; otherwise
its Sink has no executable identity hash and clearance above `normal` fails
closed.
If the Host safe path has no bare Python command, only a supported `python`
alias may fall back to the already-running interpreter's Host-owned base
executable; workspace-local interpreter paths and links are rejected before
symlink resolution.

The shell provider is containment and mediation, not a general OS sandbox. It
does not make an otherwise hostile native binary safe. Policies restrict
commands and environment, primitives enforce authority, and resource monitors
terminate covered process trees; deployments needing a hostile-code boundary
must inject a container/WASM/service provider and document its own kernel and
network isolation assumptions.

Likewise, a Host rule that trusts Shell, PTY, or MCP stdio authorizes delivery
to that executable; it is not a claim that Agent libOS controls the executable's
later direct I/O or secondary forwarding. Trusted Runtime Modules and provider
extensions execute inside the TCB and must not bypass the shared SDK/data-flow
gate.

## Failure semantics

An effectful provider may raise `ProviderEffectNotStarted` only when it can
certify that no external mutation, delivery, request, spawn, or information
observation began. Every ordinary exception, timeout, cancellation, resource
limit, missing classifier, or sink failure after dispatch is ambiguous:

- finite authority remains consumed;
- the prepared effect is finalized or retained as `unknown`;
- provider reconciliation may query an existing receipt after restart;
- startup never automatically replays an unknown effect.

Host provider exceptions cross the public boundary through one
`PublicErrorEnvelope`: a stable Host-selected `code`, Host exception-class
`error_type`, and Host-generated `correlation_id`. Provider-authored exception
messages are never copied into MCP/JSON-RPC results, static Tool failures, Deno
syscall frames, ToolExecution events/audit, or durable Tool result objects.
Static Tools retain their generic Tool error category for compatibility and put
the complete provider envelope in structured error details; uncaught Deno
syscall failures are reconstructed as the same envelope before ToolExecution
persists them.

Checkpoint restore and image commit report provider-classified effects but do
not compensate or roll back provider state, including Git checkout/worktree,
ref, remote, and simulated-PR state. Audit rows are append-only through
RuntimeStore APIs; external-effect rows retain one causal identity while
guarded prepare, dispatch, finalize, and retention transitions update their
state. These guarantees exist only within the RuntimeStore trust boundary: an
operator with direct database write access can tamper with them unless the
deployment adds external append-only storage, signatures, or remote
attestation.

## Registration and visibility

JSON-RPC endpoint and MCP server registries are host configuration stored in the
runtime database. Visibility of a registry row, Skill, tool schema, image, or
Runtime Module does not grant process authority. GUI registration accepts
bounded manifest/package payloads for the surfaces it exposes; CLI/admin
registration can use explicit host paths. MCP stdio inspection returns the exact
`stdio_authority_resource` that a host must grant rather than asking users to
reconstruct a digest from command/env fields.

## Provider extension checklist

A new backend or operation is not complete until it has:

1. A typed provider protocol/result and a primitive-facing
   `ProtectedOperationContract`.
2. Explicit capability resource/right, Task Authority effect class, approval
   policy, resource charge, `data_flow_direction`, canonical Sink identity,
   identity-hash rule where provider-backed, trusted source/payload
   descriptors, and a trusted ingress context for every ingress or
   bidirectional invocation. Unclassified responses combine the request
   context with a `normal/untrusted` external origin; resource-backed reads
   capture the file binding or PTY session labels before dispatch.
3. Prepare-before-observation ordering and exact PENS boundary.
4. Bounded inputs, outputs, time, cancellation, and process/network cleanup.
5. A success classifier plus conservative fallback for classifier failure.
6. Pending-effect reconciliation semantics that query but never replay.
7. Denial, timeout, cancellation, ambiguous outcome, event, audit, and resource
   tests in the appropriate `security` or `providers` lane. Ingress tests must
   prove propagation on success and on any failure after provider start, and
   prove no propagation for both raised and structured
   `ProviderEffectNotStarted` certification of the current phase.
8. Platform coverage or an explicit gap in the
   [support matrix](support_matrix.md).
9. Static protected-operation coverage proving every ingress invocation
   supplies its context and every egress invocation supplies its Sink, source
   context, canonical payload, and operation before the provider boundary.

See [architecture.md](architecture.md) for composition,
[capabilities.md](capabilities.md) for authority resources,
[git.md](git.md) for the fixed-repository Git boundary,
[jsonrpc.md](jsonrpc.md) and [mcp.md](mcp.md) for remote manifests, and
[modules.md](modules.md) for trusted provider-hook registration.
