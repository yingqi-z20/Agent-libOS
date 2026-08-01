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

## Python composition boundary

Import the public structural protocols and default composition from
`agent_libos.substrate`:

```python
from pathlib import Path

from agent_libos import Runtime
from agent_libos.substrate import (
    JsonRpcProvider,
    LocalResourceProviderSubstrate,
    McpProvider,
    ResourceProviderSubstrate,
)


def open_with_remote_providers(
    target: str | Path,
    workspace: Path,
    *,
    jsonrpc_provider: JsonRpcProvider,
    mcp_provider: McpProvider,
) -> Runtime:
    substrate: ResourceProviderSubstrate = LocalResourceProviderSubstrate(
        workspace
    )
    substrate.jsonrpc = jsonrpc_provider
    substrate.mcp = mcp_provider
    return Runtime.open(target, substrate=substrate)
```

This is an executable composition skeleton, not a complete provider
implementation. It works only when the supplied objects implement every
dispatch and external-effect classification method in `JsonRpcProvider` and
`McpProvider`. Starting from `LocalResourceProviderSubstrate` preserves the
other required filesystem, Git, clock, Shell, and Human providers. A fully
custom composition must implement all fields of `ResourceProviderSubstrate`.
Provider code remains trusted Host code and does not inherit authority,
data-flow, resource, evidence, or error-sanitization responsibilities from the
model. See [Python API](python_api.md#provider-protocols-and-injection) for the
injection and Host-API contract.

## Current provider inventory

| Provider | Current backend | Authority and policy | Effect/evidence contract | Bounds and containment |
| --- | --- | --- | --- | --- |
| Filesystem | `LocalFilesystemProvider` rooted at one workspace | Typed `filesystem:<namespace>:<path>` rights; cwd and state probes occur only after authority | Reads are information-flow effects; successful mutations are `irreversible`/`not_supported` because the local provider records no preimage or undo log and exposes no compensation operation; mutations prepare one pending effect and finalize the same id; created parents inherit labels and recursive delete aggregates the bound subtree | Lexical resolution followed by no-follow containment, size/list bounds, safe write/delete handling |
| Clock | `LocalClockProvider` | Clock resource/right plus the configured sleep cap. The Clock protected contract itself uses `ResourcePolicy.NONE`; ToolBroker/Scheduler calls remain subject to their generic process tool-call/runtime budgets | `now`, monotonic observation, sleep/asleep cancellation, and result classification use one protected operation; later failures remain unknown rather than refunding authority | Configured finite sleep limit, monotonic elapsed accounting, sync and async paths |
| Shell | `LocalShellProvider` | Exact executable resource, shell policy rule, approval, cwd read authority, process budget | Intent exists before spawn; timeout/cancel/limit/classifier failures conservatively retain performed/unknown evidence | No shell command string by default, scrubbed environment, workspace cwd, and stdout/stderr bounds. POSIX supplies wall/CPU/RSS supervision and process-tree termination; Windows rejects `SubprocessLimits` before spawn |
| Git | `LocalGitProvider` pinned to the Runtime workspace repository | configured repository resource (default `git:workspace`), affected filesystem resources, per-remote `git_remote:workspace:<name>`, per-PR `git_pr:workspace:<id>`, state-token CAS, and mandatory approval for destructive operations | Reads are ingress; local mutations, fetch, push, and simulated-PR transitions use distinct protected-operation descriptors and local query-only reconciliation | System Git 2.26+, byte-safe parsing, repository/config identity validation, cross-process lock, no arbitrary argv/URL, executable extensions disabled, bounded output |
| Human | `LocalHumanProvider` plus GUI/terminal host surfaces | Typed question/permission request and explicit policy decision | Terminal read/write and GUI request presentation are protected information-flow operations; conditional GUI views expose only bound metadata and reject parent responses until their exact one-shot release is consumed through presentation | Typed responses, queue state, bounded payload/output, lock-free blocking I/O with claimed request state |
| JSON-RPC | `HttpJsonRpcProvider` | Registered endpoint and exact method capability; model-supplied URLs are forbidden | Registry metadata is gated before lookup; calls prepare the reservation/intent before resolving header values, then remote DNS starts inside that intent; transport/classification settles the same effect id | Closed manifest shape, header-env allowlist, request/response hard limits, timeout, resolved-address policy, client-only JSON-RPC 2.0 |
| MCP | `SdkMcpProvider` for Streamable HTTP and stdio | Registered server/tool capability; stdio additionally requires `process:spawn` and exact `mcp_stdio:<digest>` execute authority exposed by server inspection | Tool calls do a negative clearance precheck, exact tool/stdio authorization, target-only executable resolution, and exact Sink clearance before preparation; complete credentials are resolved only after the reservation/intent exists. HTTP DNS and stdio spawn are covered by that intent, while live tool refresh resolves env first; registry lookup is gated; live discovery is bidirectional and tool results use bounded provider evidence | Tools-only v1 surface, closed manifest shape, header/stdio-env allowlists, request/response and timeout limits, contained stdio process lifecycle |
| PTY | Trusted `modules/pty` Runtime Module provider hooks | Startup hash trust plus normal process/shell authority; published sessions are Object Memory `EXTERNAL_REF` handles with Object rights | Spawn is bidirectional; read/continuous ingest are ingress; write/resize/public close are egress. Effectful operations use protected pending-to-finalized evidence; write raises the session label high-water even after an ambiguous provider outcome | Output bounds on both backends; independent reader/monitor workers and process-tree wall/CPU/RSS supervision on POSIX. Windows uses `pywinpty`/ConPTY, has no Job Object or resource supervisor, and rejects `SubprocessLimits` before spawn |

Filesystem compare-and-swap is an optional provider extension, not a breaking
change to `FilesystemProvider.write_text`. Existing custom providers continue
to receive the legacy write signature for ordinary writes. A conditional write
uses `FilesystemCompareAndSwapProvider.write_text_compare_and_swap` only after
runtime feature detection; if the extension is absent, the primitive rejects
the request before path resolution, authority reservation, mutation intent, or
provider dispatch. The local provider validates the expected full-content
SHA-256 (or the `missing` creation token) again inside its path-creation lock
immediately before opening or replacing the target.

`modules/pty` is a repository/source-distribution asset, not part of the core
Python wheel. A wheel-only installation must receive that trusted module
separately before it can load the provider hooks. The optional `pty` dependency
extra installs `pywinpty` on Windows; it does not change the wheel's file
inventory.

LLM requests are also formal bidirectional protected provider operations. Their
Sink is `llm:<profile>` and profile/model/base-URL/API-mode plus effective
provider retention policy (`store`, prompt-cache retention, and Responses
continuation policy) is hashed into the trusted identity. Precheck and client
construction use one frozen Host snapshot, so an already-cached client cannot
drift from the identity being authorized. The returned provider content is
treated as unclassified `normal/untrusted` input and cannot lower the request
context.

The LLM protected-operation contract uses `ResourcePolicy.NONE`; LLM call and
token accounting is performed separately by the executor. It preflights the
call count before dispatch, then charges that count when the executor attempt
completes or fails; provider-reported token usage is charged only after a
completion. There is no durable call-count or maximum-token reservation before
dispatch, so these budgets are not hard provider-spend caps. An abrupt host
failure between dispatch and post-attempt charging can leave provider call or
token usage uncharged. Missing or invalid billable usage fails the completed
action closed when a token budget is configured, and an over-budget completion
is retained without dispatching its model-selected tools.

Durable Task Runs use that same accounting contract. A Run may persist a local
resume point after a complete LLM action and all paired tool results, but it does not
reserve provider calls, tokens, or money before dispatch. Its configured budget
therefore remains a local post-attempt enforcement control, not a hard
provider-spend cap. Product surfaces must not relabel it as a guaranteed cost
limit.

Task Run recovery also does not weaken protected-operation classification. A
provider-certified non-dispatch may be continued, and a complete durable
provider success may finish local settlement. A dispatched or unknown effect
instead blocks the Run in `needs_attention` until a Host selects a recovery
action derived from authoritative evidence. Merely reopening, pausing,
cancelling, or rerunning a Run never authorizes an automatic provider retry.
Caller-supplied receipt JSON is not authoritative evidence: receipt recovery
requires the trusted provider's verifier to authenticate and normalize the
receipt, followed by an exact Run/effect/Runtime-epoch/state compare-and-swap.
For a Durable Task Run, the provisional recovery-command receipt and that
normalized effect settlement commit atomically. If they committed but the
later public command-result update was lost, an exact replay checks the stored
settlement and finishes local Run projection without invoking the verifier or
provider effect again. Its content-free command receipt binds the admission
Runtime epoch, exact finalized external-effect transition sequence, and the
matching append-only `external_effect.recovery_settled` audit record whose
source is `host_verified_receipt`. The audit binds the prior/settled state,
provider outcome, transition sequence, and canonical provider-receipt digest.
That transition/audit chain remains available after terminal retention removes
readable provider metadata and receipt bodies, so replay never depends on
purgeable content. This is a post-commit guarantee, not unconditional
at-most-once verifier execution: if the whole verification transaction rolled
back, no durable pending command or normalized settlement exists and a later
fresh attempt may verify again.

Once a Run can safely terminalize, default Task Run retention reduces provider
metadata and receipt bodies for its linked terminal external effects through
the same canonical `full -> summary -> hash_only` state machine. The reducer
retains effect identity, state, classification, canonical-argument hash,
original payload digest, receipt digest, and causal links. It never reduces a
dispatched/unknown or otherwise nonterminal effect merely to make cancellation
or cleanup appear complete. The same Run cleanup replaces linked Human request
prompt, response, and decision bodies with content-free hash projections while
retaining request identity, type, status, timestamps, audit linkage, and
content digests; a retained Human row must not be mistaken for retained Human
plaintext.

LLM responses also cross a fixed inbound trust boundary that is independent of
the outbound `max_tokens` preference. Before provider-authored content or tool
arguments are joined or copied into durable Runtime state, content is limited
to 262,144 characters and each content or argument string to 1,048,576 UTF-8
bytes. A completion may contain at most 256 tool calls; each arguments string is
limited to 262,144 characters, and all tool arguments together are limited to
1,048,576 characters and 1,048,576 UTF-8 bytes. Responses API output is limited
to 2,048 items and each message content list to 2,048 parts. A type, encoding,
iterator, count, or size violation fails the completion closed through the same
text-free `LLMError` boundary; oversized provider content is not truncated into
an apparently valid model action.

Explicit `safety_identifier`, `prompt_cache_key`, and
`prompt_cache_retention` fields are dispatched to the Host-selected
OpenAI-compatible endpoint, including a custom base URL; bounded compatibility
retry may remove a rejected field. `previous_response_id` is different: the
low-level client admits it only for the official Responses endpoint with
provider storage enabled, and the AgentProcess executor never supplies one.

Every payload-bearing provider exit declares an explicit data-flow direction
and descriptors. Human provider I/O uses `human:<recipient>:<channel>`; GUI
projection uses `human:<recipient>:gui` while resolving the configured Human
trust identity. JSON-RPC uses
`jsonrpc:<endpoint>:<method>`, MCP uses `mcp:<server>:<tool>`, Shell uses the
resolved executable, and PTY sessions retain their spawn Sink identity. A
negative clearance check rejects impossible MCP stdio egress before executable
resolution; exact clearance uses the subsequently resolved executable-bound
identity. Other Sink clearance is checked before provider state, DNS, stdio,
or spawn, and every final Sink is rechecked inside the SDK prepare transaction.
Cached MCP tool metadata is public; a live refresh remains a provider operation.
See [Data Flow](data_flow.md).

Supplying an already-constructed `LocalResourceProviderSubstrate` does not make
its Git settings independent of the Runtime configuration. During open, the
Runtime atomically binds the active Git settings to the Local Git provider and
the Shell raw-Git guard; conflicting settings or incompatible Local provider
subclasses fail startup before a partially bound substrate can be used.

Runtime environment resolution is not uniformly outside preparation. A
JSON-RPC call and an MCP tool call first reserve finite authority and prepare a
pending intent, then resolve complete header/stdio credential values before
their first provider phase; a missing value restores and abandons that
preparation because no provider work started. Before MCP call preparation, a
negative clearance precheck and exact tool/stdio authority permit reading only
a manifest-mapped child `PATH` and, on Windows, `PATHEXT` needed to resolve and
hash the executable. Those values are pinned into the later immutable dispatch
environment, and exact Sink clearance uses the executable-bound hash. The
complete snapshot copies only manifest-referenced Host variables plus required
Windows bootstrap keys, never the ambient environment wholesale. MCP cached
tool listing resolves nothing,
while live `list_tools(refresh=true)` resolves its environment before reservation
and intent creation. The operation-specific references below are authoritative
for these orderings.

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
mutable workspace Shell/PTY executables and every local MCP stdio executable
are copied into a private Host-owned snapshot and executed there instead of
reopening the authorized path. The local MCP provider uses the same resolved
executable for both its configuration hash and snapshot source, and passes that
absolute snapshot path directly to Windows process creation without a second
SDK lookup. For MCP native executables only the executable is copied; for
shebang scripts, the pre-existing direct sibling set is exposed up to a
configured bound beside the private copy through links to the original
locations so scripts that read resources relative to `$0` or `__file__` retain
ordinary read-following behavior. The mirror is
all-or-nothing: enumeration, limit, symlink, or Windows hard-link fallback
failure aborts before final provider execution instead of silently omitting a
possibly required resource. Only the executable bytes are pinned. Sibling
content remains live and is not package attestation; `lstat`/`O_NOFOLLOW`,
parent-relative (`../`) layouts, creating or renaming beside the snapshot, and
executable plugin/package trees are outside this compatibility boundary. Such
providers must supply a stronger package/container substrate.

On Linux, an explicit MCP stdio cwd is passed through a stable
`/proc/self/fd` directory handle. The local provider rejects configured
subdirectory cwd on platforms without that mechanism; its Host-owned workspace
root remains the default. A snapshotted Python venv launcher receives
compatibility `VIRTUAL_ENV`, `PATH`, `PYTHONPATH`, `PYTHONNOUSERSITE`, and
platform launcher values. Those values expose the selected live dependency and
plugin tree, which is not covered by the executable hash.

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

`ProviderEffectNotStarted` is a certificate about the current provider callable
and phase, not about the whole composite operation. An effectful provider may
raise it only when it can certify that the current phase attempted no external
mutation, delivery, request, spawn, or information observation. The SDK then
examines the completed-phase transcript. It abandons the intent and restores
finite authority only when every earlier phase was non-mutating,
non-information-flowing, and explicitly `commits_authority=False`. Otherwise it
settles the earlier phases as a confirmed partial outcome (or leaves their
durable pending evidence if settlement itself fails), does not restore authority
they committed, preserves their information-flow consequences, and excludes
only the certified current phase.

Every ordinary exception, timeout, cancellation, resource limit, missing
classifier, or sink failure once a phase may have started is ambiguous:

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
3. Prepare-before-observation ordering and an exact, phase-local PENS boundary;
   a PENS certificate for a later phase must never erase prior provider work.
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
