# Data Labels, Egress Control, and Trusted Sinks

Agent libOS enforces data labels at runtime-mediated payload exits. A visible
tool, a normal operation capability, or a Human approval is not enough to send
classified data: the Host-owned Sink registry must also clear the exact Sink,
the data identity domain must match, and a `conditional` Sink needs an exact
one-shot release.

This control is independent of ordinary authority. A `trusted` Sink does not
grant filesystem write, shell execute, Human output, JSON-RPC method, MCP tool,
process, resource-budget, provider-registration, or Task Authority effect
permission. All of those checks still apply.

## Labels and derivation

`DataLabels` contains strictly validated fields:

- sensitivity: `public < normal < confidential < restricted < secret`;
- trust: `untrusted < unknown < user_asserted < verified < trusted`;
- integrity: `untrusted < unknown < checked < verified`;
- optional `origin`, `tenant`, `principal`, and
  `declassification_authority`.

Unknown enum values, malformed identities, and unknown fields in a typed flow
record fail closed. `mixed` is a valid conservative aggregate marker and may be
constructed or produced by aggregation, but any receive or egress carrying it
fails closed until a Host reclassifies it. Derived values take the highest
sensitivity, lowest trust and integrity, and the union of identity evidence.
Combining different non-empty tenants or
principals produces `mixed`; a mixed value can neither be sent automatically
nor released. It must first be reclassified by a Host operation.

Explicit object metadata cannot overwrite a parent tenant or principal.
Lowering sensitivity, removing or replacing identity, raising trust or
integrity, or changing declassification authority requires an exact
`declassification:object:<oid>` `admin` capability. Model-facing memory tools
do not reject all label metadata: the standard `create_memory_object` tool may
set sensitivity, tenant/principal, and only the conservative `untrusted` or
`unknown` trust/integrity values. Ambient materialization labels and every
explicit parent are then propagated conservatively, so a lower model-supplied
value cannot wash a source label and a conflicting identity becomes `mixed`.
The tool rejects trust/integrity elevation and every non-empty
`declassification_authority`; the JIT memory syscall ignores caller-supplied
label fields and derives labels from its runtime-owned flow context.

For an LLM-created Object, provenance is the union of explicit `parent_oids`
and every Object actually included in that LLM materialization. Missing or
unreadable explicit parents reject the creation. ToolBroker carries the
materialization id, Object id/version/content hash, and aggregate labels in
runtime-owned context; model arguments cannot replace that context.

When persistent context enrichment is explicitly enabled, process `llm_context`
objects keep a `label_history` high-water mark across append, compaction,
checkpoint, fork, exec, reopen, retry, and Human/message resume. The default
source-only path creates no such Object. Provider/tool input is never allowed
to reset a previously observed higher sensitivity. Unclassified external responses are `normal/untrusted`
and are aggregated with, rather than substituted for, the request context.
Synchronous tool workers merge their final runtime-owned context back before
output-schema validation; schema failures, exceptions, and JIT timeout/error
paths retain a labeled Tool Result carrier. The async JSON-RPC, MCP, and Shell
wrappers likewise return worker-thread context on both success and failure,
instead of relying on one-way `ContextVar` copying.

## Sink trust registry

Unmatched Sinks use the configured defaults and are `untrusted`. For every
`untrusted` Sink, whether unmatched or selected by an explicit rule, the
effective sensitivity ceiling is `min(selected_max_sensitivity, normal)`, where
the selected maximum comes from the matched rule or the default. A `public`
rule/default therefore tightens the ceiling to `public`, while no `untrusted`
rule can raise it above `normal`. This is a Sink egress ceiling; the
`normal/untrusted` labels applied to unclassified ingress do not select or
widen a Sink rule. Host configuration may publish exact or terminal-`*` rules:

```yaml
data_flow:
  default_trust_level: untrusted
  default_max_sensitivity: normal
  sink_rules:
    - pattern: "llm:corp-secure"
      trust_level: trusted
      max_sensitivity: restricted
      tenants: ["tenant-a"]
      principals: ["analyst-a"]
      identity_sha256: "<profile-identity-sha256>"

    - pattern: "jsonrpc:crm:*"
      trust_level: conditional
      max_sensitivity: confidential
      tenants: ["tenant-a"]
      principals: ["analyst-a"]
      identity_sha256: "<endpoint-and-method-identity-sha256>"
```

Patterns are either exact or a single trailing `*`. Longest match wins;
duplicate and equal-priority overlapping patterns are rejected. Provider-backed
LLM, JSON-RPC, MCP, Shell, and PTY clearance above `normal` requires an
`identity_sha256`. Changing the profile/model/base URL, endpoint/method
manifest, MCP server/tool/transport or stdio executable, or Shell/PTY executable
content changes that hash, so the old rule no longer matches.

| Trust level | Automatic send | One-shot release | Hard maximum |
| --- | --- | --- | --- |
| `untrusted` | within its effective ceiling | cannot elevate | `min(rule/default maximum, normal)` |
| `conditional` | `public`/`normal` | required above `normal` | rule maximum |
| `trusted` | within rule clearance | not needed | rule maximum |

A labeled tenant or principal must occur in the matched rule's explicit list.
An unlabeled value does not acquire an identity from the Sink. Identity hash,
trust level, sensitivity maximum, tenant list, and principal list together form
clearance; there is no `trusted=true` shortcut.

The registry is versioned and persisted. Each mutation advances the global
generation and emits an event and audit record. The Host-only API is:

```python
runtime.register_sink_trust(spec, actor=admin_pid, replace=False)
runtime.unregister_sink_trust(pattern, actor=admin_pid)
runtime.inspect_sink_trust(pattern)
runtime.list_sink_trust(active_only=True)
```

Writes require `admin` on `data_flow_sink_registry:*` (or the configured
`data_flow.registry_resource`). The core tool registry exposes no equivalent
model tool. Runtime bootstrap may load `data_flow.sink_rules` as Host
configuration before work starts. On reopen, bootstrap-owned rules are
reconciled with the current configuration: removed patterns are deactivated,
while rules registered independently through the Host API remain durable.
Reusable as well as finite registry authority is reauthorized inside the
registry mutation transaction immediately before the write. Revoking an
unlimited `admin` grant after the outer check therefore prevents both register
and unregister from changing the registry.

## Stable Sink identities

| Exit | Sink identity | Additional binding |
| --- | --- | --- |
| LLM | `llm:<profile-id>` | profile/model/endpoint/API-mode plus effective store, prompt-cache-retention, and Responses-continuation-policy identity hash |
| Human | `human:<recipient>:<channel>` | exact recipient and channel |
| Human GUI projection | `human:<recipient>:gui` | complete gate-independent serialized public view (including status and decision) plus GUI presentation operation; trust aliases the configured Human terminal identity |
| JSON-RPC | `jsonrpc:<endpoint-id>:<method-id>` | endpoint plus method manifest hash |
| MCP | `mcp:<server-id>:<tool-id>` | server, transport, and tool manifest hash; stdio also binds the resolved executable path/content |
| MCP live discovery | `mcp:<server-id>:list_tools` | server and transport manifest hash; stdio also binds the resolved executable path/content |
| File | `filesystem:workspace:<normalized-path>` | canonical workspace path |
| Git local mutation/fetch | configured `git.repository_resource` (default `git:workspace`) | repository/worktree identity plus expected state token; fetch also binds the existing remote fingerprint |
| Git push | `git_remote:workspace:<remote>` | fetch/push URL hashes, effective config/helper identities, selected refs, and expected old remote OID |
| Simulated Git PR | `git_pr:workspace:<pr-id>` | repository-local metadata hash plus immutable base/head snapshot OIDs; create and merge also authorize the configured `git.repository_resource` as an additional Sink because they mutate repository state |
| Shell | `shell:<resolved-executable>` | resolved path plus executable content hash; mutable workspace executables dispatch from a Host-owned content snapshot |
| PTY spawn | `pty:spawn:<resolved-executable>` | resolved path/content hash fixed at session creation; mutable workspace executables dispatch from a Host-owned content snapshot |
| PTY input/control | `pty:session:<session-id>` | aliases the immutable content-bound spawn trust identity |
| Internal process handoff | `process:<pid>` | identity-domain propagation, not external trust |

`context_window_tokens` is a local LLM scheduling bound, not a Provider/Sink
identity component. Changing only that value therefore does not invalidate an
otherwise identical trusted LLM Sink rule.

Each live PTY session keeps a monotonic data-flow high-water of labels and
Object source references, seeded by the spawn request. A successful write or
resize, and a write, resize, or close whose provider mutation may have started
but whose outcome is ambiguous, merges the caller's current context into that
high-water; a provider-certified not-started operation does not. The session
Object metadata mirrors the conservative labels, while exact source references
remain in the live session context. Later input/control egress combines the
caller's context with the session high-water; reads observe that high-water plus
`normal/untrusted` PTY ingress, and session listing observes the high-water for
every returned session. Control serialization prevents a read from returning
output produced by a labeled write before that write publishes its high-water.

MCP metadata-only cached discovery is public. A process-initiated live refresh
is a bidirectional provider operation: its current flow context is checked as
outbound request data, and returned metadata or an after-dispatch provider
error raises the caller's context with `normal/untrusted` external ingress. A
provider-certified not-started failure adds no ingress. A Host-internal refresh with
no process actor uses the runtime's public/verified metadata request context.
Deno/JIT code receives no direct external authority; its syscalls enter the
same filesystem, shell, Human, JSON-RPC, MCP, and process boundaries.

Git inspection is ingress from `external:git`. Local mutations and fetch are
bidirectional protected operations, push is egress, and simulated-PR state
transitions are bidirectional. Pull-request creation and merge have two real
recipients: the PR metadata Sink and the configured repository Sink; both must
clear the same flow context before either mutation is dispatched. The ordinary
Git/remote/PR capability and Task Authority effect ceiling remain independent
of Sink clearance. Mutation target state is the opaque repository token;
remote operations additionally revalidate URL/config/ref fingerprints after
approval and before dispatch. Patch creation
derives the immutable `CODE_PATCH` Object from a conservative lineage: observed
file bindings plus repository/index carriers and returned commits. Range patches
include the current index carrier even when unrelated staged content did not
contribute patch bytes, so labels may safely overtaint. Patch application
propagates those source labels to the operation result and affected filesystem
bindings. Audit and event payloads retain only
hashes, OIDs, counts, and bounded metadata—not patch bytes, commit messages,
credentials, or raw provider errors. See [Git Provider and Primitive](git.md).

## Enforcement order

`ProtectedOperationContract.data_flow_direction` is explicitly `none`,
`ingress`, `egress`, or `bidirectional`. The older `information_flow` flag is
not interpreted as egress because reads, DNS, and clocks also observe
information.

For egress, the runtime performs this sequence:

1. Check identifier visibility and any non-consuming capability policy needed
   to avoid registry, profile, or endpoint enumeration, and reject ordinary
   authority that is already definitely denied.
2. Resolve trusted Object sources into `DataFlowContext`, construct the
   canonical primary Sink and every additional real recipient as one ordered
   tuple, and require unique identities.
3. Before resolving provider state, an adapter may run a read-only clearance
   precheck; it cannot consume or request a release. Filesystem writes,
   JSON-RPC calls, and Shell/PTY paths whose exact Sink and payload are already
   bound instead run an early full authorization, which may request exact
   release before a still-pending ordinary Human approval. Neither form is the
   protected-operation dispatch revalidation.
4. Complete the remaining ordinary capability, Task Authority, policy, and
   approval checks. The protected-operation SDK validates the resulting
   decisions and authorizes or revalidates the final payload against Sinks in
   tuple order. The primary captures the Sink-registry generation and every
   later authorization must use that same generation. A conditional Sink with
   no matching release creates only a metadata-only Human release request and
   suspends that attempt, so later Sinks may be checked on a resumed attempt and
   multiple conditional recipients may require sequential approvals.
5. Run the remaining protected-operation preflight, including provider
   classifier and resource checks required by that contract.
6. In the protected-operation transaction, revalidate ordinary authority,
   registry generation,
   Object versions/content hashes, exact payload hash, and release binding.
7. Atomically reserve ordinary authority and every required per-Sink release
   capability and create the pending external-effect intent.
8. Immediately before each provider phase, revalidate the registry generation,
   exact source versions/content hashes, target state, payload, and release
   binding. The optional `data_sink_revalidator` recomputes a mutable primary
   Sink identity (including executable content); additional Sink identities
   must remain stable for the invocation, although their trust and authorization
   are still rechecked. A mismatch appends a payload-free denial. A release
   already reserved or committed by an earlier phase remains valid only through
   that same protected-operation reservation. For Shell and PTY executables in
   the mutable workspace, and for every local MCP stdio executable, create and
   verify a private Host-owned content snapshot before final dispatch. MCP
   native executables copy only their executable bytes; shebang scripts retain
   the bounded, all-or-nothing direct-sibling compatibility mirror. Mirrored
   resources remain live provider input rather than part of the pinned
   executable identity.
9. Only then enter DNS, provider state, filesystem state, stdio, subprocess, or
   Human payload delivery. Executable dispatch uses the snapshot rather than
   reopening the authorized source path.

An early denial does not call the provider, DNS, filesystem `state()`, Human
payload delivery, or spawn; does not consume an ordinary finite-use capability;
and does not create an external-effect intent. It does append a payload-free
`DataFlowDecision`, event, audit record, and Explain evidence containing the
Sink, label/source hashes, trusted source refs, trust record/generation, and
reason.

Successful effect metadata binds the primary and additional Sink decisions,
trust ids/hashes, shared registry generation, source Object
id/version/content hashes, label hash, and every release capability where
applicable. Additional recipients appear in `additional_egresses`. Mutable
sources and all Sink authorizations are checked again immediately before every
provider dispatch. A mutation before the first authority-committing phase
rejects and restores the complete reservation set; a mutation after a phase
that observed/mutated provider state or used the default
`commits_authority=True` prevents the later provider call and conservatively
finalizes the already-started effect. Pure coordination phases must explicitly
set `commits_authority=False` to retain the certified-not-started restoration
floor.

Direct Host primitive calls may pass `source_oids`. Those are Object references
resolved by the runtime; callers cannot submit a `DataLabels` value as payload
authority. A Host raw payload with no sources starts as `normal` inside the Host
trust boundary. Model-mediated calls also inherit their ambient materialized
context, so omitting an explicit source cannot wash a label.

### Atomicity and recovery boundary

Steps 6–7 are the atomic authority boundary: current ordinary and release
authority is revalidated, finite uses are reserved, and the prepared
external-effect intent is written in one RuntimeStore transaction. This is not
a distributed transaction with the provider. Dispatch rejection or
`ProviderEffectNotStarted` can restore the exact still-live reservations only
when every previously completed phase has
`state_mutation=False`, `information_flow=False`, and
`commits_authority=False`. The first phase is one instance of that rule; the
default `commits_authority=True` closes the restoration floor even for a
successful phase that otherwise appears non-effectful. Once the floor closes,
authority stays committed and an ambiguous outcome remains durable as
`unknown` or pending reconciliation. Prepared-intent recovery runs before
general provider-effect reconciliation on reopen. The complete contract is
documented in the
[Protected Operation SDK](protected_operation_sdk.md), with its durable state
and ordering in [Storage](storage.md).

Process launch, exec, fork, and checkpoint restore use separate durable
runtime-publication programs rather than external-effect intents. Their
publication/operation links are reconciled before generic stale-operation
terminalization, and an unresolved publication keeps mutation admission
fail-closed. See [Explainable Operations](explainable_operations.md) and
[Checkpoints](checkpoints.md) for those recovery boundaries.

## Exact conditional release

A conditional send above `normal` creates a separate requested
`data_release:<sink>` `approve` capability with `uses_remaining=1` for every
Sink that needs a release. The capability subject binds the pid. Its
`DataReleaseBinding` includes:

- Sink identity and identity hash;
- trust id/hash and current registry generation;
- Task Authority manifest hash;
- a SHA-256 over the canonical source-reference tuple (whose entries encode
  Object or file-binding identity, version, and content hash) plus the aggregate
  label hash;
- canonical payload/argument hash, operation, and target-state version.

The Human request carries bounded metadata only—not the payload—including Sink
and trust identity, registry generation, sensitivity, tenant/principal, size,
hashes, source count, manifest identity, operation, and the exact requested
one-shot capability binding. Approval does not change Object labels, does not
replace ordinary capability, and cannot exceed the Sink maximum or identity
scope. Replay, cross-Sink reuse, payload change, source mutation, manifest
change, trust replacement, or generation change fails. `untrusted` Sinks
cannot be elevated by Human approval; `trusted` Sinks need no release.
When the binding includes mutable target state, the SDK resolves its current
version again inside the prepare transaction; a change from the approved
version denies before capability reservation or provider dispatch.

All per-Sink releases share the source, payload, operation, manifest, and target
state binding, while each retains its own Sink identity and trust binding. As
described above, Human requests can be surfaced sequentially across retries.
Once the full set is authorized, prepare, success commit, certified-not-started
restoration, and unknown settlement reserve or settle the complete set in one
RuntimeStore transaction.

For Human Sinks, the metadata-only release and protected request are linked in
durable state. Rejecting/cancelling the release (or an ambiguous release-prompt
provider outcome) terminates the protected request and prevents automatic
replay. A provider-certified not-started outcome keeps the exact linked pair
pending, so reopen does not create duplicate release requests.

Conditional LLM provider releases additionally obey `llm.persist_full_io`
before approval. In opt-out mode, `llm_pending_actions` stores only the exact
prepared-request hash, payload hash, and non-sensitive resume identifiers; the
raw messages, tool schema, and egress payload remain in executor memory. The
same runtime can consume the approved one-shot release against that hash. If
the runtime reopens after losing the in-memory request, it claims the durable
generation and fails the process closed rather than reconstructing a different
prompt or sending an unbound payload. Rejecting the exact release clears that
prepared request and pauses the process behind a Host-only resume gate. A
parent/model `signal_child_process(resume)` cannot turn the rejection into an
automatic replacement request; an explicit Host resume starts a new model turn
and, if still required, a new independently bound release.

GUI serialization uses a separate `human:<recipient>:gui` presentation Sink
with the configured terminal identity as its trust identity. Before release,
the parent is metadata-only and cannot be answered through the GUI API. Release
approval must be followed by the protected GUI presentation that consumes the
exact one-shot capability; only then is the durable parent view interactive.
The exact binding hashes the complete public view passed to the GUI provider,
including payload, status, timestamps, and `decision`; internal release links
and visibility markers are gate state and are not part of that provider view.
That visible marker is accepted only while the original binding remains current
for Sink trust and registry generation, Task Authority manifest, labels, public
view, and operation. Pending questions and approvals also require current source
versions. A successfully delivered `human_output` is the narrow exception: its
stored message is a private-digest-bound frozen snapshot whose original source
references were validated at delivery. GUI presentation therefore rechecks its
captured labels and current Sink policy without requiring mutable source objects
to remain at the old version; a digest mismatch is denied. A later answer or any
other public-view change redacts the parent again and requires a new exact
release.
The freshness check is read-only: it does not record a data-flow decision,
consume authority, or create a release.
For GUI responses, that check runs inside the same Human-decision transaction
as the status change. Presentation lists are built lazily against their final
logical window, so an exact release is never consumed for an omitted lookahead
row; a pending metadata release is paired immediately before its still-redacted
parent without moving completed release history ahead of pending work. An
unchanged unrestricted view already handed to the same authenticated GUI
provider session may reuse a bounded in-memory receipt, but only after its exact
view hash and current Sink policy are checked under the Store lock; non-output
requests additionally revalidate current source state. A new provider session
cannot inherit the receipt. Presentation evidence remains
available in the full ledgers while bounded GUI causal windows exclude those
internally generated rows so polling cannot displace unrelated recent events or
audits.

For an LLM Sink, the default `llm.persist_full_io=true` policy serializes the
final messages, tool schemas, profile/Sink identity, request options,
provider-state scope, flow context, and exact payload binding as one durable
prepared request before returning `waiting_human`. With
`persist_full_io=false`, the durable row contains only the prepared-request and
payload hashes plus non-sensitive resume metadata; the exact request remains
in executor memory. Approval in that same runtime resumes the hash-bound
request once. If the in-memory request is lost, the runtime fails closed before
provider dispatch rather than rematerializing process memory or asking the
model to recreate the call. A changed profile/Sink identity also fails closed,
and the same release cannot produce a second provider request.

The protected-operation lifecycle restores an unconsumed ordinary/release use
when protected preparation aborts before its durable dispatch boundary. Once
phase dispatch is attempted, it restores only when the current phase certifies
`ProviderEffectNotStarted` and every completed earlier phase has
`state_mutation=False`, `information_flow=False`, and
`commits_authority=False`. Crossing DNS, stdio, provider, or spawn closes that
restoration floor even if a later phase fails.

## Process domains and persistence

`TaskAuthorityManifest.data_flow_policy` is only the process receive-domain
ceiling:

```json
{
  "schema_version": 1,
  "allowed_tenants": ["tenant-a"],
  "allowed_principals": ["analyst-a"]
}
```

Child manifests may inherit or narrow these sets, never widen them. Empty sets
accept only untagged process data. Goals, messages, results, Object Tasks,
memory merge, fork, and exec carry trusted labels; reading a secret message
taints later goals and replies. Runtime-carried `data_labels` from every
process event represented in a Prompt batch merge directly into that request's
egress context before provider dispatch, including under the default
`source_only` context policy. When the persistent LLM context object is
enabled, those labels also merge into its durable high-water. This policy
cannot make an external Sink trusted and cannot reduce a Host rule's clearance.
An event that carries `data_labels` must carry the complete canonical label
object; an empty, partial, non-object, or otherwise malformed value fails
closed before any provider call instead of falling back to `normal` labels.

SQLite and PostgreSQL share durable records for the active/versioned Sink
registry, append-only decisions, exact release constraints, file-path label
history/tombstones, pending LLM flow context, and provider-chain clearance
fingerprints. Successful file writes bind the canonical path, content hash,
labels, and sources. If a file write or directory creation may have crossed its
provider mutation boundary but returns an ambiguous outcome, the runtime still
publishes the intended conservative binding for the target and any auto-created
parents before surfacing the failure; a provider-certified not-started failure
does not. This can overtaint a path that ultimately remained unchanged, but it
prevents a later read from washing a source label. File ingress projects the
current active record through an opaque, durable reference to its immutable
binding ID, generation, and content hash; it does not restore runtime-only
Object references as live dependencies.
That exact historical binding remains valid after reopen or later replacement,
while a missing binding or mismatched generation/content hash fails source
revalidation before egress. Derived Object provenance keeps these opaque IDs in
`source_refs` and expands only the binding's stored Object ancestry into
`parent_oids`, so a file binding is never mistaken for an Object. When a write
creates missing parent directories, each
auto-created parent receives the same conservative binding. Recursive directory
delete obtains labels and its subtree fingerprint from one store snapshot,
rechecks that fingerprint during protected prepare, and tombstones all
descendant bindings on success. Non-recursive directory delete similarly
captures the target binding ID/generation and tombstones it with a storage-level
compare-and-swap, so a binding created after provider dispatch is preserved.
Later out-of-band modification does not silently lower the known path label.
Directory listing snapshots the directory and child bindings before provider
enumeration, rejects a changed subtree fingerprint, and labels returned
(including truncation-lookahead) names from the captured snapshot rather than a
newer lower binding. Runtime file writes and directory creation share the same
label-publication critical section with listing, so a newly visible child cannot
be returned before its conservative path binding is durable. That section is a
fair hierarchical path lock rather than one workspace-global mutex: unrelated
file reads may proceed concurrently, while ancestor/descendant operations and
create operations sharing a potentially missing top-level ancestor serialize.
Lock keys conservatively normalize Unicode and case so Host aliases cannot split
the label boundary; widening a held child scope to an ancestor is rejected
instead of risking a lock-upgrade deadlock.

Checkpoint restore does not roll back files, provider state, trust generation,
or decisions, and it cannot revive a stale release. Fork/reopen revalidates the
current registry and authority. Runtime-only Object payloads can disappear on
reopen; a durable pending action may use the Host-written row version and label
snapshot for domain validation, while materialization still fails and the
operation must recover or rerun.

The Runtime computes and records continuation-related fingerprints from the
provider identity, Sink/trust generation, clearance sensitivity/identity
domain, Task Authority manifest, and context epoch. Changing those inputs
changes the recorded fingerprint. The current full-snapshot AgentProcess
executor does not use `previous_response_id` at all; it rechecks the complete
rebuilt request on every egress. The low-level `LLMClient` does not receive or
enforce these Runtime fingerprints, so they are observability and
future-protocol inputs rather than a current low-level isolation boundary.

Data-flow enforcement controls runtime-mediated movement; it does not encrypt
stored payloads or automatically shorten their lifetime. The Host may
explicitly run the disabled-by-default [Evidence and LLM Payload Retention
maintenance](evidence_payload_retention.md), which monotonically reduces only
eligible terminal LLM-call and external-effect provider payloads from `full` to
`summary` to `hash_only`. It preserves causal identities, classifications,
links, and original payload digests, and it refuses rows still needed for
runtime continuation or recovery. This at-rest lifecycle is distinct from
write-time `llm.persist_full_io` and from Sink clearance.

## Guarantee boundary

The guarantee covers payloads that cross runtime-mediated Sinks. Marking a
Shell, PTY, or MCP stdio executable trusted means the Host deliberately trusts
that program to receive the data; it does not give Agent libOS kernel-level
control over that program's later network or filesystem I/O.

Host administrators, direct database writes, trusted Runtime Modules/provider
extensions, a Sink's secondary forwarding, and additional I/O initiated by a
native child after the mediated argv/stdin boundary are outside this guarantee.
Audit and data-flow decision rows are append-only through runtime APIs. Active
Sink-registry, file-binding, and context-label projections instead change under
generation/CAS or supersession rules while retaining the history needed for
decisions. None is tamper-proof against a database administrator. Deployments
that need that property must add external signed/append-only evidence and an
OS/container/WASM isolation layer.

The machine-checked invariant is
`data-labels-constrain-runtime-mediated-egress`; the deterministic benchmark
attack class is `data_label_exfiltration`.
