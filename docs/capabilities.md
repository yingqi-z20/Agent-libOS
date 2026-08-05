# Capabilities

Agent libOS uses capabilities as the runtime authority subsystem. A visible
tool, activated Skill, JIT tool, child-process handle, object name, path string,
image id, checkpoint id, or JSON-RPC endpoint id is not enough to perform a
protected operation. Protected effects are authorized at primitive use by
process identity, typed resource pattern, right, effect, constraints, Task
Authority, data-flow clearance, resource budgets, and Human approval where
required. Provider-effect intents, events, and audit make those decisions and
their outcomes observable; audit is settlement evidence, not an authorization
credential, and provider-success audit evidence may be written only after the
provider returns. This lets self-evolving agents change their action surface
without implicitly changing what resources they can affect.

## Capability Record

Durable `Capability` records are structured authority statements:

- `subject`: process or runtime actor that holds the authority.
- `resource`: canonical typed resource pattern.
- `rights`: operation rights such as `read`, `write`, or `execute`.
- `effect`: `allow`, `deny`, or `ask`.
- `issued_by`: actor that issued the record.
- `issuer_cap_id` and `parent_cap_id`: lineage for grant/delegation decisions.
- `delegable`, `revocable`, `delegation_depth`, and `max_delegation_depth`:
  delegation and attenuation state.
- `issued_at`, `expires_at`, `uses_remaining`, and `status`.
- `constraints` and `metadata`.

The inspection projection exposes `issued_by` as `issuer` and also derives
`rules`, `lease`, and `delegation` objects from the durable constraints, expiry,
use-count, and delegation fields. Those objects are presentation conveniences,
not additional persisted authority.

Model-facing and paged GUI presentation is count- and byte-bounded.
`list_capabilities` accepts an exclusive `after_cap_id` cursor and returns
`has_more`/`next_cursor`; its `limit` is only a maximum because the byte budget
may end a page sooner. A large record may replace `metadata` with `{}` and add
`metadata_projection={omitted:true,bytes,sha256}`. If more space is required it
also replaces `constraints` with `{}`, clears derived `rules`, and adds the
analogous `constraints_projection` receipt. Those empty containers are
placeholders, not proof of empty stored metadata or policy. The hashes bind the
canonical omitted JSON for comparison; they neither reveal it nor authorize an
operation. `inspect_capability` and delegation/revocation receipts use the same
bounded projection.

One-shot authority is not encoded as a policy string. It is an `allow`
capability with `uses_remaining=1`; committed primitive use consumes it and
revokes the capability when the count reaches zero. If one-shot Object Memory
authority is resolved through a namespace/name lookup, any handle minted from
that lookup remains one-shot; name lookup cannot turn temporary authority into
a persistent object handle.
`require(...)` atomically consumes finite-use authority by default. Multi-step
paths that need compensation opt out with `consume=False`, reserve the exact
use before their effect, and then commit or restore that reservation token.
This covers Human and provider effects, Skill mutations, and ObjectTask owner
Object authority. ObjectTask's separate `process:spawn` admission check first
reserves finite authority. It commits that reservation atomically with task
publication; a failure before publication restores the exact still-live use
only after runner cleanup is confirmed. If cleanup is uncertain, the
reservation remains fail-closed, and a post-publication failure is never
refunded. An explicit revoke invalidates outstanding tokens, so late cleanup
cannot reactivate revoked authority.
Provider subsystems do not manage these tokens directly: the
[`Protected Operation SDK`](protected_operation_sdk.md) reserves every distinct
finite decision in the same transaction as local prepare state and the effect
intent, then commits on the first effectful provider phase. The public
`CapabilityManager.restore_reserved_use(...)` operation is revoke-safe; the SDK
is its sole provider-effect caller.
On reopen, prepared protected operations first restore exact still-live
reservations only when their durable intent proves that no provider phase
began. After that reconciliation, every other stale `reserved` row is abandoned
fail-closed. Once the commit/provider boundary was crossed, the one-shot use
remains consumed even if the remote or follow-on result is a failure.
Checkpoint inspect/diff/replay reserve the selected exact-checkpoint or
checkpoint-process read lease across the diagnostic, restore it if the
diagnostic raises, and commit it once on success. Actor-mode checkpoint list
uses the ordinary immediate `require(...)` path. Cross-actor ObjectTask
get/wait consumes one selected finite read lease; list consumes each distinct
finite read lease used by its returned rows at most once; cancel consumes the
selected finite write lease after terminal/unsafe-cancellation preflight.
Internal wait polling does not repeatedly consume authority.
Capability issuance itself commits the new row, process attachment, event,
audit, and issuer reservation as one transaction.

Capability mutation admission is enforced at the durable authority-service
boundary, not only at selected Runtime façade methods. Every public
`CapabilityManager` method has a machine-checked read/mutation/mixed
classification; every public finite-use lease and capability-mutation service
method is mutation guarded as well. Mixed
authorization methods remain readable for diagnosis when `audit=false`, while
their evidence-writing `audit=true` branch is mutation admission. Therefore a
recovery-required `CLOSE_FAILED` runtime cannot be bypassed through
`manager.leases` or `manager.mutations`.

`AuthorityTransaction` acquires lifecycle admission when its context is
entered and retains it through reauthorization, caller business work,
finite-use settlement, and UnitOfWork commit. Authority services revalidate the
recovery-fence epoch after acquiring the store lock and again before commit.
If a recovery fence invalidates an already-admitted waiter, its capability,
reservation, event, and audit writes roll back together. Startup and recovery
code can still mutate under their explicit internal lifecycle lease.

Launch authority is additionally governed by a durable Host-authored
[`TaskAuthorityManifest`](task_authority_manifest.md). Its authorized entries
compile into root capabilities, while its effect, expiry, budget, approval, and
data-flow fields remain independent ceilings. Image requirements do not compile
into capabilities. Model permission requests are rejected before a Human prompt
when the requested resource/right exceeds the manifest.

Human approval for a concrete external operation adds an
`approval_binding` constraint containing an effect id, canonical argument hash,
and optional target state version. A resumed operation with changed arguments
or a changed supplied target version cannot consume that one-time capability.

`deny` records dominate matching allows. To create an exception, revoke the
broad deny and issue narrower allow/deny records explicitly. The runtime does
not implement hidden override precedence that could accidentally reopen a
blocked resource, and primitive-specific candidate filtering must reapply the
same deny-first ordering before making a decision. Candidate-based authorization
also filters every supplied capability to the requested `subject`; a primitive
cannot authorize one process by passing a matching capability owned by another
process. Human-approved capability specifications apply the same isolation:
their `subject` must equal the process that created the request (or be omitted
and default to it), otherwise validation fails and the request remains pending.

Authority derivation uses public CapabilityManager transition APIs.
`derive_authority()` applies source authority and, when `ceiling_specs` is a
non-empty iterable, an additional manifest ceiling. For this capability API,
`ceiling_specs=None` and `ceiling_specs=[]` both mean no additional ceiling;
this is separate from a Task Authority `permitted_effects=[]`, which is an
explicit provider-effect deny-all value.
`transition_allowed_rights()` reapplies expiry, finite-use duplication rules,
and current restrictive policy for checkpoint/fork/restore transitions. These
surfaces replace subsystem-local resource matching as the transition policy
boundary. A single delegation commits its capability row, process attachment,
grant event, and delegation audit together. Batch derivation validates every
requested spec before publishing the first child record, then commits all
delegations and the transition summary in one transaction. A late validation
or evidence-sink failure therefore publishes none of the batch.

## Rights

The common rights are:

- `read`: inspect or materialize a resource.
- `write`: create or modify a resource.
- `delete`: remove a resource.
- `execute`: run or load a resource.
- `materialize`: include object content in a prompt or tool result.
- `link`: create Object Memory links.
- `diff`: compare object or checkpoint state.
- `grant`: issue authority over a covered resource.
- `revoke`: revoke authority over a covered resource.
- `approve`: approve a human/request resource.
- `admin`: perform destructive or policy-changing operations.

The exact right set depends on the primitive. Unknown or unsupported rights do
not create primitive behavior by themselves. Capability records reject unknown
rights, including `*`; use explicit rights instead of all-rights wildcards.

## Resource Matching

Resources are typed and canonicalized before authorization. Matching is not a
raw `startswith` or suffix check.

Important resource conventions include:

- `filesystem:workspace:<path>` for exact workspace files and directories.
- `filesystem:workspace:<dir>/*` for a directory subtree.
- `filesystem:workspace:*` for the whole workspace namespace.
- `shell:<executable>` for direct command authority; `shell:*` for shell
  policy records when paired with `shell_policy_level`.
- `human:<name>` for human output, questions, and approvals.
- `clock:now`, `clock:sleep`, and `clock:*` for clock reads and bounded sleep.
- `object:<oid>` for Object Memory content.
- `object_namespace:<namespace>` for Object Memory namespace listing and name
  lookup.
- `process:<pid>` and `process:*` for process operations.
- `process:spawn` for child process and ObjectTask runner creation. Child launch
  checks include the selected `image_id` and operation, so an Authority Rule can
  bind a grant to one image and to `process.spawn_child` rather than fork.
- `image:<image_id>` and `image:*` for image registration.
- `skill:<skill_id>` and `skill:*` for Skill operations.
- `skill_trust:*` or `skill_trust:<sha256>` for global Skill trust.
- `checkpoint:process:<pid>`, `checkpoint:<checkpoint_id>`, and
  `checkpoint:*` for checkpoint operations.
- `jsonrpc_endpoint:<endpoint_id>` and `jsonrpc_endpoint:*` for JSON-RPC
  endpoint registry metadata.
- `jsonrpc:<endpoint_id>:<method_id>`, `jsonrpc:<endpoint_id>:*`, and
  `jsonrpc:*` for JSON-RPC method invocation.
- `mcp_server:<server_id>` and `mcp_server:*` for MCP server registry
  metadata.
- `mcp:<server_id>:<tool_id>`, `mcp:<server_id>:*`, and `mcp:*` for MCP tool
  invocation.

Wildcard syntax is terminal only. `kind:*` is a typed prefix pattern and
`kind:body/*` is a subtree pattern. Bare global `*` is rejected; authority must
stay inside an explicit resource kind. Prefix collisions are rejected: a grant
for `filesystem:workspace:src/*` covers `src/main.py`, not `src2/main.py`.

Requested resources are produced by primitives after their own normalization.
For example, the filesystem primitive resolves cwd-relative paths, enforces
workspace containment, then asks for the canonical logical resource. Shell
authority uses normalized executable identity and argv token policy, not a shell
string.

## Authorization API

The manager entry point is:

```python
authorize(subject, resource, right, context) -> CapabilityDecision
```

`require(...)` wraps `authorize(...)`, raises on denial, and claims finite-use
authority before returning. Effect adapters that need pre-commit compensation
must call `require(..., consume=False)` and use
`reserve_decision_use`/`commit_reserved_use`; raw authorization decisions are
not reusable effect tickets. Primitive code should pass operation context such
as path, argv, byte counts, hashes, risk labels, process lineage, or provider
details. The decision records:

- matched capability ids,
- selected capability id,
- issuer chain,
- effect and derived human-facing policy,
- constraint evaluation results,
- one-shot consumption id when applicable,
- operation context preview.

A `CapabilityDecision` does not itself create or own a Human request. When a
primitive acts on an `ask` decision, the Human subsystem owns the durable
request and direct Host calls receive its id through
`HumanApprovalRequired`/`HumanResponseRequired`.

New capability writes reject unknown constraint keys, explicit `null`, wrong
JSON types, and values with no defined comparison/evaluation semantics. Key
omission is not equivalent to an explicit null. A historical row containing an
unknown, null, or malformed constraint remains readable for diagnosis, but it
cannot authorize an operation or be delegated/granted onward; the runtime does
not silently clean or reinterpret it.

## Authority Rules And Profiles

Capabilities can carry deterministic `AuthorityRule` entries. A rule has:

- `operation`, such as `filesystem.read`, `shell.run`, `jsonrpc.call`,
  `mcp.call`, or `deno.syscall`;
- `effect`: `allow`, `ask`, or `deny`;
- `risk`: `harmless`, `low`, `medium`, `high`, or `destructive`;
- structured `conditions`, such as argv tokens, match mode, path/cwd intent,
  network intent, filesystem intent, and the target `image_id` for child
  launch.

Rules are not LLM judgments. They are local, deterministic policy facts. Unknown
rule shapes, unknown constraint keys, and malformed values for known conditions
fail closed. The primitive converts the final capability decision into a sandbox
profile and records that profile in approval context, audit, and external-effect
metadata where applicable. In particular, `timeout_s` and `timeout_max_s`
conditions, and the operation `timeout_s` compared against them, must be finite
non-negative numbers. Booleans, NaN, positive or negative infinity, and negative
values are malformed and fail closed; zero and fractional values are valid.

## Issue, Delegate, Revoke

All authority mutation goes through explicit operations:

- `issue(actor, subject, spec)`: `actor` must hold covering `admin` authority,
  or hold both covering `grant` authority and covering
  `allow` capabilities for every right being transferred. `grant` is not a
  capability-minting right: it can only transfer rights the actor already has,
  cannot create `deny`/`ask` policy records, and cannot transfer finite-use
  capabilities onward. Overlapping `deny` or `ask` boundaries, or malformed
  authority rules on the covering parent, fail closed before transfer. The
  selected `admin`/`grant` decision and the complete transferred-right parent
  chain are recomputed inside the issue AuthorityTransaction; its finite-use
  reservation, capability row, process attachment, event, audit, and lineage
  links therefore commit together against one authority/target generation.
- `delegate(parent, child, spec)`: `parent` must hold a covering delegable
  `allow` capability. Delegation can only attenuate resource, rights, expiry,
  constraints, and delegation depth. Finite-use capabilities are consumed by
  direct use and cannot be delegated. Delegated records keep a parent link, so a
  later parent revocation or expiry stops the child record from authorizing.
  Every parent constraint key must be explicitly present in the child.
  Binding, version, path, policy, approval, and other equality constraints must
  retain the exact typed value. `git_allowed_refs` is the one subset-ordered
  collection: a child may keep or narrow the parent's refs but cannot add a ref.
  Authority-shaping constraints such as `shell_policy_level` and
  `authority_rules` cannot be introduced by a child and must remain identical
  when inherited. A well-defined restrictive `git_allowed_refs` value may be
  introduced when the parent has no such key. These same attenuation rules
  apply when `grant` transfers authority and when manifest transitions derive
  child capabilities.
  Delegation also cannot launder an overlapping parent `deny`/`ask` or malformed
  authority rule by selecting a narrower allow. The child record is not
  observable unless its process attachment, event, and audit evidence all
  commit.
- `revoke(actor, cap_id)`: first requires the target record to be `revocable`.
  The original issuer may then revoke it; the holder may relinquish only its own
  `allow` capability, not an `ask` or `deny` record that restricts it; otherwise
  the actor needs covering `revoke` or `admin` authority. Target mutation,
  finite authority reservation/commit, the revoke event, and revoke audit all
  share one store transaction. A validation or evidence-sink failure therefore
  neither publishes the revocation nor consumes its finite authority.

Runtime bootstrap, image bootstrap, Human approval, Host-side CLI mutations,
checkpoint flows, and tests may use explicit embedding-host paths such as
`issue_trusted()`. Authority changes through that method retain their normal
capability evidence, while command/operation evidence remains specific to the
calling path: unchecked Host diagnostic reads do not promise a command-level
audit, and checkpoint fork has the documented post-commit best-effort audit
boundary. These methods, along with operations that explicitly set
`require_authority=false`, are Host API bypasses: the actor string is
attribution, not authentication, and no configured name or prefix makes an
ordinary caller trusted. They must not be exposed to an AgentProcess, model,
Skill, or JIT tool. Ordinary execution must use the checked `issue()`,
delegation, and revocation paths and has no implicit signing authority.

Fork and spawn inherit authority only through delegation/attenuation. Exec
switches the image/tool table and may shrink capabilities, but it never grants
the target image's declared `required_capabilities`.

Image-package boot is also not an external authority grant. Its
`workspace.grants` entries apply only to the package workspace seed after it is
materialized into that process's private directory under the configured
`image.materialized_workspace_root` (`agent_outputs/image_workspaces/` by
default); they cannot name arbitrary host or workspace paths. The optional
`recursive` and `delegable` fields default to `false`; when present, each must
be a YAML boolean. Strings, numbers, and null are rejected during package
registration before artifact publication, workspace materialization, or
capability issuance.

## Permission Policy And Human Approval

Stored capability policy names are:

- `always_allow` maps to `effect=allow`.
- `always_deny` maps to `effect=deny`.
- `ask_each_time` maps to `effect=ask`.
- `allow_once` maps to `effect=allow, uses_remaining=1`.

A terminal response to a `permission_request` must explicitly choose one of
`always_allow`, `always_deny`, or `ask_each_time`; it cannot choose
`allow_once`. One-shot authority is requested/issued through the separate
one-time capability lease shape so its exact use count remains explicit.
Approved responses cannot install `always_deny`, rejected responses cannot
install `always_allow`, and the JSON `approved` boolean must agree with the
terminal status. Approved ordinary questions require a non-empty string
`answer` rather than implicit coercion.

Complete GUI, CLI, Host, automatic, and terminal-provider Human decisions are
validated as finite, acyclic JSON with string object keys. The default response
limits are 131072 serialized bytes, 32 nested containers, and 4096 JSON values.
A direct invalid decision is rejected before request, process, Capability,
event, or audit mutation. A terminal-provider answer is necessarily validated
after its protected provider read is recorded, but before it can decide the
request, resume the process, or grant authority. An invalid or oversized value
is represented in that read evidence only by a bounded rejection marker; the
Human request remains pending and the process remains waiting, allowing the
Host to correct the input and service that exact request without a duplicate
question.

Model-facing `request_permission` is not a raw grant API. It first checks the
canonical resource/right request against the live Task Authority request
ceiling, then requires the caller to hold `human:<name>` write authority. No
Human request is created unless both checks pass. Before a request enters the
Human queue, the runtime normalizes rights, classifies risk, records resource
scope, attaches any deterministic constraints, and shows the selected lease
shape. Ordinary model requests cannot ask for broad high-risk authority such as
`capability:*` with privileged rights (`admin`, `grant`, `revoke`, `write`,
`execute`, or `delete`), `shell:*` execute, or root/global filesystem write such
as `filesystem:/:*` or `filesystem:*`. Workspace-level write
(`filesystem:workspace:*`) can be approved by the human, but workspace-wide
delete cannot: model requests for delete must name a concrete file or directory
subtree. Model requests for Shell execute are rejected for every non-Git command
class; `shell:git` is the only model-requestable class and is constrained to six
exact read-only argv forms (`status`, `status --short`, `branch --show-current`,
`rev-parse --show-toplevel`, `diff`, and `diff --stat`). Other Shell authority
cannot be obtained through model-facing `request_permission`; it must come from
an exact per-use Shell approval or an audited Host/admin-issued capability or
shell policy. Admin CLI and bootstrap paths can still issue broader policy
explicitly, with audit.

When `ask_each_time` applies, the primitive creates a durable Human approval
request. A model-facing Tool or JIT syscall invocation eventually receives the
final payload or a final denial error; those surfaces expose no pending/retry
protocol. A direct Python Host manager or primitive call may instead raise
`HumanApprovalRequired` or `HumanResponseRequired` with the `request_id` so the
Host can inspect or service that durable wait.

Filesystem `read_text`, `read_bytes`, directory listing, and working-directory
validation follow that same per-use path. They construct the exact logical path
and requested limits before prompting, but do not inspect existence, kind,
metadata, directory entries, or bytes until the one-shot read capability has
been approved and reserved. After the read settles, that one-shot capability is
consumed and the underlying `ask_each_time` policy remains in force.

Approval context includes path, resource, caller-declared overwrite policy,
byte count, SHA-256, argv, risk, rule id, sandbox profile, and escaped previews
when available. Filesystem target state is deliberately omitted until the
operation has received and reserved authority, so an approval prompt cannot be
used as an existence or metadata oracle.

`human_output` requires `human:<name>` write and reserves finite-use authority.
Its optional channel is normalized before dispatch: omission or the exact empty
string selects `runtime.terminal_channel`; every other value is trimmed and
must remain 1..128 characters, so whitespace-only input is invalid.
Before provider delivery, one transaction marks its request `delivered` and
persists a structured pending external-effect intent. The success event, audit
record, and effect finalization are settlement evidence and are written only
after the provider call returns; the runtime never pre-records successful
delivery. Provider failure finalizes unknown evidence when possible; successful
delivery followed by classifier failure uses conservative unknown
classification/finalization when possible. A later finalization or settlement
failure leaves the pending intent and still returns without replay. In that case
the success event or audit row may be absent, so the pending intent and delivered
request are the conservative durable evidence. The terminal queue cannot deliver
that request again, and the one-shot use is not restored after the provider
boundary.

Terminal queue questions, permission-policy prompts, and ordinary approval
prompts also cross the configured Human provider through structured `read` or
`write` intents. Interactive answers and automatic decisions settle the same
pending effect id, but their audit/effect observations persist only request id,
purpose, lengths, byte counts, and SHA-256 values. Raw prompt text, raw answers,
and Human-provider exception text are never written to those records. If the
provider interaction succeeds but later event, audit, classification, or CAS
settlement fails, the request still commits its answer/policy so draining the
queue cannot show the prompt again; the unresolved intent remains pending.
Human output provider failures likewise persist only `provider_error_type`, not
the exception message. A successfully delivered output also stores a private
SHA-256 binding for its message. GUI projection treats those fixed bytes as a
delivered snapshot while retaining the original labels and provenance: current
Sink clearance is still mandatory, but later mutation of an original source
does not create a false denial. The private binding is omitted from public Human
request payloads, and a mismatch is withheld.

## Data release is separate authority

Ordinary Human approval cannot make an external Sink trusted. Data above
`normal` sent to a `conditional` Host Sink requires a distinct
`data_release:<canonical-sink>` `approve` capability with
`uses_remaining=1`. The capability constraint binds the pid, trust id/hash and
registry generation, Task Authority manifest hash, source Object
id/version/content hashes, label hash, canonical payload hash, operation, and
target-state version. Any mismatch makes it unusable.

For GUI presentation of a successfully delivered `human_output`, the immutable
message digest and complete public-view hash are the source snapshot boundary.
The original source references remain in the private Human record as provenance
and were validated at provider delivery, but the GUI release does not require a
mutable source to stay at that historical version. Labels, Sink trust and
generation, manifest, operation, and payload/view hashes remain exact.

The release request contains bounded metadata only—Sink, trust, registry,
label/source/request identities, operation, sizes, hashes, and the exact
requested one-shot capability binding—never the payload. It does not change
the Object label and does not replace the normal capability or
`permitted_effects` check. An `untrusted` Sink cannot be released above
`normal`; a `trusted` Sink needs no release but still grants no ordinary right.
Host Sink registry writes separately require `admin` on
`data_flow_sink_registry:*` and have no model-facing tool. See
[Data Flow](data_flow.md).

## Tool, Skill, And JIT Boundary

The complete process tool table is the callable binding set. A separate model
tool projection controls which schemas the LLM receives. Capabilities control
primitive effects independently of both.

`ToolPolicy` is declaration metadata only. Fields such as
`declared_permissions` and `declared_confirmation_required` can help a GUI or a
human reviewer understand a tool, but the broker does not convert them into
grants or confirmations. Real authorization still happens in the primitive that
touches the resource.

For example, even when the model projection contains `write_text_file`, the
process still fails to write `src/app.py` if it lacks write authority for
`filesystem:workspace:src/app.py` or a covering subtree grant.

Activating an immutable built-in Skill projects an all-or-nothing set of
existing Image-authorized bindings into the model table; activating a registered
Skill can add its separately authorized static and Deno/TypeScript JIT bindings
to both tables. Either path may add instructions, but it does not grant
filesystem, shell, human, object, process, image, checkpoint, JSON-RPC, or MCP
authority, nor filesystem access or global hash trust for another Skill source.
JIT syscalls bypass the LLM-facing tool table, but they still enter the same
primitive authorization path as built-in tools.

Persistent LLM-context enrichment is also explicit authority. The default
source-only path does not create or append an `llm_context` Object.
`context:enrichment/execute` opts one process into that delta projection;
`context:maintenance/execute` separately permits the proactive storage
waterline and model-window pressure handler to invoke the configured compactor.
Enrichment authority alone therefore creates and appends the persistent context
Object. Without enrichment authority the request stays source-only; without
maintenance authority persistent deltas can still be used, but proactive
compaction is not authorized. Neither capability supplies the separate required
child-spawn or compressor-image authority.

The static default tool tables of the built-in base, coding, review, and
toolmaker images contain `list_capabilities` and `inspect_capability`; the
context-compressor image does not. Those four images use Skill projection, so
neither capability tool is model-visible in the initial five-tool bootstrap.
Activating the exact `agent-libos-authority-basics` Skill projects both tools
(and `request_permission`) for that process. `delegate_capability` and
`revoke_capability` are registered static tools but are not included in those
default image tool tables.

## Process Messages

Process messages are IPC records owned by the runtime message manager, not a
separate `message:*` capability namespace. Current authorization is based on
process relationship and target identity: a process can receive its own
messages, parents and direct children can communicate through the exposed
message tools, and filters such as channel, correlation id, or explicit message
ids limit delivery. Message tool visibility still matters, but visibility does
not grant unrelated process or Object Memory authority.

## CLI And Syscalls

The CLI supports:

```bash
uv run agent-libos capabilities list [--subject <pid>] [--include-inactive]
uv run agent-libos capabilities inspect <capability_id>
uv run agent-libos capabilities grant <subject> <resource> --rights read write
uv run agent-libos capabilities delegate <parent> <child> <resource> --rights read
uv run agent-libos capabilities revoke <capability_id> [--reason "..."]
uv run agent-libos capabilities explain <subject> <resource> <right>
```

Without `--actor-pid`, the capability CLI is in Host mode. `grant` and `revoke`
use their explicit Host-authority bypasses, while `delegate` still validates
the named parent's delegable authority and attenuation. Successful mutations
emit their normal capability event and audit evidence. In contrast, Host-mode
`list`, `inspect`, and `explain` are unchecked diagnostic reads and do not emit
a command-level audit record. With `--actor-pid`, subcommands apply their
process-scoped rules: cross-subject reads require the relevant `admin`
authority, and authority mutations use their checked paths. `--actor-pid` is a
`capabilities` command option and must appear before the subcommand, for
example:

```bash
uv run agent-libos capabilities --actor-pid <pid> list
```

Deno/TypeScript JIT tools can use the syscall names:

- `capability.list`
- `capability.inspect`
- `capability.request_permission`
- `capability.delegate`
- `capability.revoke`
- `mcp.list`, `mcp.inspect`, `mcp.tools`, and `mcp.call` for registered MCP
  servers and tools.

Syscalls do not consult the process tool table. They are authorized by pid,
capability records, primitive rules, and Human approval where required. Their
audit records are evidence of the decision and outcome, not another authority
gate.

## JSON-RPC Authority

Remote JSON-RPC calls are pre-registered endpoint resources. Agents cannot pass
URLs or secrets at call time.

Registry inspection and mutation use endpoint resources:

```text
jsonrpc_endpoint:demo-weather
jsonrpc_endpoint:*
```

Method calls use method resources:

```text
jsonrpc:demo-weather:forecast
jsonrpc:demo-weather:*
jsonrpc:*
```

The required right comes from the endpoint method spec: `read`, `write`, or
`execute`. Granting `jsonrpc_endpoint:* read` allows endpoint discovery, not
method invocation. Granting `jsonrpc:demo-weather:forecast read` allows that
specific remote method, subject to primitive validation, human approval,
runtime DNS policy, provider classification, audit, and external-effect
recording. Agent-facing inspect paths do not expose endpoint URLs or header
prefix/suffix values; those are host registry details.

## MCP Authority

MCP servers are pre-registered host provider resources. Agents cannot pass MCP
server commands, URLs, environment variable names, credentials, or arbitrary
remote tool names at call time.

Registry inspection and mutation use server resources:

```text
mcp_server:demo-tools
mcp_server:*
```

Tool calls use tool resources:

```text
mcp:demo-tools:echo
mcp:demo-tools:*
mcp:*
```

The required right comes from the server manifest's allowlisted tool spec:
`read`, `write`, or `execute`. Granting `mcp_server:* read` allows server
discovery, not tool invocation. Granting `mcp:demo-tools:echo read` allows only
that manifest-declared tool, subject to argument schema validation, live tool
schema checks, runtime DNS/secret policy for HTTP transports, provider
classification, audit, and external-effect recording. MCP Resources and
Prompts are not exposed. Manifest v2 protocol discovery is a separate protected
external read: a process needs both `read` and `execute` on the exact
`mcp_server:<server-id>` resource. Discovery metadata cannot grant Tool rights
or turn an unsupported server capability into a Runtime capability.

For `stdio` MCP transports, actor-mode server registration, protocol discovery,
live tool refresh, and tool calls additionally require both `process:spawn`
`write` and `execute`
on the exact `mcp_stdio:<sha256>` launch resource. The hash covers the canonical
command, argv, environment mapping, and cwd, so a grant for one launch surface
cannot authorize another. Registration authorizes persisting that surface;
refresh and calls are the operations that actually start a local child process.
Host/admin mutations emit their normal mutation evidence, and live refreshes
and provider calls retain their operation-specific evidence. Read-only Host
registry list/inspect/tools paths do not thereby promise a separate
admin-operation audit.

For both JSON-RPC and MCP, one-time Human approval is bound to the immutable
registry specification SHA-256 and a monotonic registry generation, as well as
the concrete call arguments. The binding also represents an endpoint/server
that is not registered yet, so an approval obtained before its first
registration cannot authorize whatever specification is registered later.
The persisted binding survives reopen and is revalidated before every provider
phase; replacement or any other generation/specification change requires a new
approval.

Provider return values are untrusted Host-boundary inputs. JSON-RPC and MCP
detach and validate the returned structure before runtime code accesses its
fields. Malformed results and unknown failures after a provider return expose
only the public code/type/correlation envelope; where response byte usage is
unknown, settlement charges the active provider phase's configured ceiling
rather than trusting provider-controlled metadata.

## Filesystem Authority

Filesystem capabilities can target exact files, directory subtrees, or the
whole workspace. Relative paths resolve from the caller process working
directory. The filesystem primitive enforces workspace containment before host
provider calls. Path strings that escape the workspace are rejected even if a
model can produce them.

Read, write, and delete are separate rights. Granting read over a directory does
not grant write or delete.

File writes also bind the canonical `filesystem:workspace:<path>` as a data
Sink. The content/path operation is checked against source labels before
`state()` or mutation, then revalidated with the exact path-label generation in
the protected transaction. Success persists a content-hash/source/label path
binding; later external modification does not silently downgrade it. A trusted
path still requires normal filesystem write authority.

Process cwd selection is a filesystem read, not ambient process metadata.
`set_working_directory`, explicit spawn/fork working directories, and explicit
PTY cwd values require `read` on the selected directory resource. Higher-level
spawn/image or shell authorization is checked before the cwd state probe, and
the probe uses the normal filesystem pending-intent and finite-use semantics.

Write/delete authority, including an `ask` decision, is resolved before the
primitive calls `state()`. An unauthorized or rejected mutation therefore
cannot probe whether a target exists, its kind, or its size. After approval, the
exact finite-use mutation right is reserved and one pending effect intent spans
both `state()` and the mutation. Read/list similarly use one reservation and
intent across their state and data/metadata reads.

That shared intent is an authorization and durable-evidence boundary, not a
serializable host-filesystem transaction. Runtime state checks and provider
mutation are separate host operations, and another host process can race them.
The provider therefore revalidates containment and no-follow conditions at the
mutation boundary, but callers must not interpret the prior `state()` result as
a globally locked filesystem snapshot.

If the first provider observation certifies `ProviderEffectNotStarted`, the
reservation and pending row are atomically restored/abandoned. If `state()`
already returned information but the main mutation then certifies not-started,
the mutation one-shot remains consumed and the same effect id is finalized as
`state_mutation=false, information_flow=true` with a conservative unknown
outcome. Ordinary state/read/mutation exceptions likewise cannot prove what was
observed or changed and finalize or retain a conservative unknown outcome.

Conditional text writes are the narrow exception to the preceding state-probe
rule. A complete `read_text` returns `content_sha256`; truncated reads return no
token. When `write_text` receives that digest, or the literal `missing`, its
pre-mutation state/parent checks are advisory phases that neither disclose
content nor commit finite authority. A provider-certified content conflict
therefore abandons the pending intent and restores the write reservation,
without changing bytes or file-label bindings. A successful conditional write
commits authority at the actual write phase. Providers without the optional
filesystem compare-and-swap extension reject conditional writes before any of
these phases begin.

The default `LocalFilesystemProvider` stores no preimage or undo log and
exposes no compensation operation. Successful writes, directory creation, and
file or directory deletion are therefore classified as
`irreversible`/`not_supported`; checkpoint restore reports them but cannot
reverse them.

The default local filesystem provider performs no-follow traversal inside the
workspace root for existing files. Existing file read, write, and delete reject
symlink or junction traversal and reject regular files with multiple hard links
(`st_nlink > 1`). Directory listings report child symlinks as symlink entries
without following them.

Bounded reads do not trust the size returned by an earlier `state()` call. If
that snapshot does not already prove the file is oversized, the primitive asks
the provider for one internal sentinel byte beyond `max_bytes`; a file that
grows between state and read is therefore returned at the caller's bound with
`truncated: true`, not mislabeled as complete. The sentinel is not exposed or
charged as information flow: `bytes_read` and `external_read_bytes` count only
the selected bytes up to `max_bytes`. If the original state already exceeds the
bound, the provider reads only the bound because truncation is already known.

Authorization is also separated from external-effect evidence. Filesystem,
clock, and shell primitives persist a pending `unknown` effect intent after
local preflight but before the first provider call. A classified result
conditionally finalizes that same `effect_id`; after information flow or
mutation may have begun, a post-provider capability/event/audit/classifier
failure cannot make the durable effect history look empty.

`clock.sleep` and async `clock.asleep` create that intent before the first
provider `monotonic()` measurement. The elapsed-time observations make the
whole composite operation `information_flow=true`. Only
`ProviderEffectNotStarted` from that first measurement can atomically restore a
reserved one-shot use and abandon the intent. An ordinary first-measurement
exception, or any sleep/cancellation/final-measurement exception after the
first observation (including a later `ProviderEffectNotStarted`), consumes the
use and finalizes a conservative `unknown` effect.

## Shell Authority

Shell execution is argv-only. The model-facing tool and syscall accept token
arrays, not shell command strings. Pipes, redirects, wildcard expansion, and
command chaining must be requested explicitly through an interpreter executable
such as `bash`, `sh`, `cmd`, or `powershell`, where policy matching can inspect
the interpreter token.

Arguments beginning with a `file:` URL are rejected before the provider runs.
For bare executables, the local provider resolves argv[0] on a safe host PATH
and refuses a resolution inside the workspace or selected process cwd. Shell
subprocesses receive a constrained environment with `HOME` and `USERPROFILE`
pointing at the workspace root instead of the host user's real home.

When the local provider has a workspace root, Shell also enforces that the cwd
and every path-like argv token remain inside it before policy or approval is
evaluated. This includes positional paths, `--option=path`, attached short-option
operands, `~`, and POSIX or foreign absolute-path syntax. URL-like arguments are
not treated as paths except that `file:` URLs are always rejected. An
`always_allow` policy does not bypass this containment check.

An authorized Shell subprocess still runs as the host user. Argv policy, safe
PATH resolution, the constrained environment, and resource monitoring are not
an operating-system filesystem or network sandbox. Filesystem and JSON-RPC
Capabilities govern their corresponding runtime primitives; they cannot mediate
direct file or network I/O performed by an authorized child executable. Use the
Deno JIT boundary for code that can operate through Deno's no-permission,
cached-only process plus the libOS syscall protocol. This is not an OS syscall
filter, seccomp profile, container, or VM; hostile native-code isolation still
requires an explicit container, WASM, VM, or service-provider boundary. On
Windows, the local Shell provider rejects budgeted execution that supplies
`SubprocessLimits` because it cannot enforce that profile; unbudgeted Shell
execution may still run, while Deno uses its separate Windows supervision and
budget backend.

Shell argv is an egress payload to `shell:<resolved-executable>`. Clearance
above `normal` must bind the executable content hash, which is revalidated
immediately before provider dispatch. Explicit Host trust may allow classified
argv within clearance, but does not bypass command
risk policy, execute capability, Human approval, cwd authority, effect ceiling,
or resource budget. More importantly, trust says the Host accepts that native
program as a recipient; it does not mediate the program's subsequent direct
filesystem or network I/O. PTY spawn uses the same content-bound
resolved-executable model and pins that trust identity for all later session
input.

Shell command risk is classified by argv-token rules before the provider runs:

- `harmless`: exact read-only status/version/inspection argv recognized after
  higher-risk executable-family checks, such as `git status --short`.
- `low`: read-only project inspection such as `git diff`.
- `medium`: project code execution such as `pytest`, pytest collection,
  `npm test`, or `uv run ...`.
- `high`: recognized package managers, network-capable tools, script
  interpreters, and the exact supported `python -m compileall` forms. Because
  executable-family rules run before exact harmless/custom rules, the current
  classifier also treats
  `python --version` as a high-risk interpreter command and asks under the two
  mixed automatic policies (`allowlist_auto_else_ask` and
  `blocklist_ask_else_auto`). An otherwise unrecognized command, including an
  unrecognized service-startup form, falls back to `medium`; the classifier
  does not infer hidden behavior from a command name.
- `destructive`: commands whose direct or nested executable matches the
  built-in delete/move/permission/system-control set. Recognized commands are
  denied even under broad shell policy. This is a deterministic executable
  classifier, not semantic proof that an arbitrary unrecognized program cannot
  perform destructive work.

The built-in shell policy levels then decide how to handle the classified rule:

- `always_deny`: reject every shell command.
- `allowlist_auto_else_ask`: allow `allow` rules and ask for `ask` rules.
- `blocklist_ask_else_auto`: use the same deterministic risk rules but is
  intended for broader local operation.
- `always_allow`: allow every command whose selected deterministic rule is not
  `deny`, including rules otherwise classified `ask`/high risk, while still
  reporting that risk. It does not expand the classifier's destructive-command
  vocabulary.

Those four strings are fixed semantic labels, not remappable config fields.
Configuration can select `default_policy_level` and replace the argv rule lists,
but an overlay cannot redefine, for example, `always_allow` to mean a different
level. Legacy `*_level` remapping fields are rejected by strict config loading.

The capability record's own effect is evaluated before its policy level, so an
`ask` or `deny` shell-policy capability cannot be converted into an automatic
allow by setting `shell_policy_level=always_allow`. A finite-use command or
policy allow is reserved after validation and intent recording, immediately
before provider execution. An abort during protected preparation, before the
durable dispatch boundary, restores the exact still-live reservation. After
dispatch begins, restoration is allowed only when the first provider phase
raises `ProviderEffectNotStarted` and certifies that no execution or earlier
information flow began. Timeout, resource-limit, cancellation, ordinary
provider failure, or a post-effect classification failure commits the use and
records a conservative `unknown` external effect; a failed tool result is not
proof that the command did nothing.

Rules match tokenized argv, not arbitrary substrings. Bare executable names do
not match path-qualified executables by accident. The local provider executes
the argv vector directly with `shell=False`; the primitive never rebuilds a shell
command string from model input. As defense in depth, non-`always_allow`
automatic allow rules downgrade to human approval when argv tokens contain shell
metasyntax such as command substitution, backticks, separators, newlines,
redirection/process substitution, brace expansion, or comments.
Direct command capabilities such as `shell:git` or `shell:git:*` can allow a
specific normalized executable, but a bare direct command grant is intentionally
limited: without `authority_rules`, it only covers argv that the deterministic
classifier already marks `allow`. Medium/high shell side effects need explicit
rules, per-use human approval, or a human/admin-issued shell policy. A bare
`shell:* allow` capability is not treated as direct command authority;
whole-shell authority must be represented as a policy capability carrying
`shell_policy_level`, so the primitive can still apply the four-level shell
policy semantics. Broad `deny` and `ask` records remain conservative
constraints.

For the exact directly invoked inspection argv `git status`,
`git status --short`, `git branch --show-current`,
`git rev-parse --show-toplevel`, `git diff`, and `git diff --stat`, Shell and PTY
apply the shared Git-provider validation and inject
no-pager/no-optional-lock/no-fsmonitor/no-external-diff/no-textconv and
no-lazy-fetch hardening. Matching is case-insensitive and accepts `git.exe`.
Authorization and returned results retain the original argv. Every other direct
Git argv—including mutation and remote commands—is deterministically rejected
before Shell policy or Human approval, even under `always_allow`; callers should
use the typed `Runtime.git` boundary for capability-, state-, and evidence-aware
Git operations. The shared Shell/PTY check also unwraps supported transparent
executable launchers. `env --split-string`/`env -S` dispatch is rejected because
its eventual executable cannot be validated as an argv token.

This is an argv compatibility policy, not an operating-system process boundary.
An authorized interpreter, script, or native program can invoke Git later or
modify repository metadata directly with the host user's permissions. Granting
that Shell/PTY authority therefore authorizes those downstream effects; use a
container/WASM/service provider when child-process filesystem or network I/O
must be isolated.

Scoped denies are supported with `AuthorityRule` constraints. An unconstrained
deny still dominates all matching grants; a constrained deny dominates only when
its rule matches the current operation context, so policy can allow read-only
`git` inspection while denying `git push`.

Block-list checks also scan nested executable-looking argv tokens such as
`bash`, `powershell`, `python`, or `curl`.

## Git Authority

Typed Git authority is separate from Shell authority. The repository resource
is configured by `git.repository_resource` (default `git:workspace`) with
`read`, `diff`, `write`, `delete`, and `admin` rights.
Existing remotes use `git_remote:workspace:<remote>`: `read` permits fetch/pull
input, `write` permits push, and `delete`/`admin` cover remote ref deletion and
force-with-lease. Simulated pull requests use
`git_pr:workspace:<pr-id>` with `read`, `write`, `approve`, and `delete`; listing
uses the wildcard read resource `git_pr:workspace:*`.

Remote capability constraints can bind `git_remote`,
`git_url_fingerprint`, `git_allowed_refs`, `git_expected_state_token`, and
`git_old_oid`. The primitive validates and canonicalizes remote/ref/state/OID
inputs before placing them in capability context; the URL fingerprint is a
Host observation of the registered remote rather than a model-supplied value.
A scoped deny is evaluated before remote metadata lookup, so denied authority
cannot enumerate a configured remote or its URL hash.

All mutations require an opaque state token from a previous Git read. The
primitive checks it after ordinary authority and again under the cross-process
repository lock immediately before dispatch. Checkout-affecting operations
also authorize exact filesystem paths, or the selected worktree subtree when a
safe preflight cannot enumerate every path. A Git write grant therefore cannot
rewrite files without matching filesystem rights, while filesystem authority
cannot directly access `.git` metadata.

The destructive action matrix is explicit. Reset, clean, amend, every restore,
rebase, every integration abort, branch/tag/stash/worktree/ref deletion, branch
rename, forced branch creation, stash pop, stash including untracked files,
forced switch/tag, fetch prune, remote-ref deletion, force-with-lease,
simulated-PR merge, and a patch application whose preview deletes files require
the applicable `delete` and `admin` rights and a one-use Human approval bound to
exact canonical arguments, state token, and relevant old OIDs. A broad allowed
capability without that approval binding cannot satisfy a mandatory approval.
Ordinary commit, integration merge/cherry-pick/revert, non-rebase pull, and
non-forced push follow the capability record's normal `allow`/`ask`/`deny`
effect.

Git tools in the coding/review static tables are not part of the initial
five-tool model projection. The corresponding all-or-nothing Git Skills must be
activated before their owned schemas become model-visible. Static binding or
later projection grants no `git:*`, `git_remote:*`, `git_pr:*`, filesystem,
Task Authority effect, or data-flow permission. Existing `shell:git` grants are
intentionally not translated. See [Git Provider and Primitive](git.md) for the
complete operation and approval matrix.
