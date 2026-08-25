# MCP Client

[Documentation home](index.md) · [Configuration](configuration.md) ·
[Troubleshooting](troubleshooting.md) · [Glossary](glossary.md)

Agent libOS is an MCP **client**, never an MCP server. Two deliberately separate
contracts coexist:

- Manifest v1/v2 is the released compatibility contract for governed MCP
  Tools. It retains its existing registry/Sink identities, provider result,
  and model-facing Tool projection.
- Manifest v3 is the exact `2026-07-28` modern Host-client contract. It adds
  governed Tools with the modern closed result union, plus closed allowlists
  for Resources, Resource Templates, Prompts, Completion, subscriptions,
  Host-owned OAuth profiles, MRTR continuations, and a digest-pinned Tasks
extension without silently reinterpreting v1/v2.

## In this guide

- Choose and author a manifest: [v1](#server-manifest-v1),
  [v2](#server-manifest-v2), or [v3](#server-manifest-v3).
- Integrate modern Host surfaces through the
  [modern Host client API](#modern-host-client-api).
- Review enforcement boundaries in [Authority](#authority),
  [Data-flow Sink](#data-flow-sink), [Security Rules](#security-rules), and
  [External Effects](#external-effects).
- Interpret results in [Call And Tool-List Results](#call-and-tool-list-results).
- Operate and diagnose through the [CLI](#cli) and
  [Host DX workflow](#host-dx-workflow).
- Review the projected surfaces in [Tools And Syscalls](#tools-and-syscalls).
- Check lifecycle boundaries in [Persistence And Checkpoints](#persistence-and-checkpoints).

Manifest schema, MCP wire protocol, SDK, product, and Store versions are
independent; see the [version map](glossary.md#version-map) before changing a
version-bearing field.

Neither contract lets an agent supply transports, commands, URLs, headers,
credentials, raw remote identifiers, or server configuration at call time.
On every model-facing Tool path, an agent passes only:

- `server_id`
- `tool_id`
- `arguments`, as a JSON object; `null` is normalized to `{}` by the primitive

The optional provider uses Python MCP SDK v2, pinned as `mcp==2.0.0`. “SDK v2”
is not a wire-protocol version: MCP protocol revisions are date strings. Manifest v3
requires the exact `2026-07-28` wire contract, while the v1/v2 compatibility
path also supports the documented legacy initialize-based revisions. Manifest
schema, SDK major, protocol revision, Agent libOS product version, and
RuntimeStore schema are independent identifiers; the Store uses schema v7.

The outbound `clientInfo` identity is part of the compatibility contract and
is selected by manifest generation, not copied mechanically from the installed
distribution metadata:

| Manifest contract | `clientInfo.name` | `clientInfo.version` |
| --- | --- | --- |
| v1 legacy wire | `mcp` | `0.1.0` |
| v2 governed Tools compatibility | `agent-libos` | `1.4.2` |
| v3 exact `2026-07-28` | `agent-libos` | `1.5.2` |

The v1 and v2 values are frozen compatibility identities. Only exact-v3 uses
the current modern product identity; changing the package version must never
silently rewrite a v1/v2 handshake.

Manifest v3 is a Host-composition surface, not a new set of automatically
model-visible tools. Remote Resources and Prompts remain untrusted input;
Resource links are inert selectors, binary content becomes a Host artifact
receipt, and prompt messages cannot become system/developer instructions.
Visibility never grants Capability, data-flow, effect, or provider authority.

The following are explicitly outside this product contract:

- an MCP server surface;
- executing or rendering MCP Apps (`ui://`, Apps HTML MIME types, and Apps
  metadata fail closed);
- OAuth Dynamic Client Registration (DCR); only Host-preconfigured
  `preregistered` or HTTPS CIMD client profiles are eligible;
- OAuth client-credentials, enterprise-managed authorization, DPoP,
  workload-identity federation, and `2025-03-26` OAuth backcompat modes; the
  current product contract is Host-preconfigured authorization-code/PKCE;
- Roots, Sampling, and Logging callbacks;
- the deprecated standalone HTTP+SSE transport; and
- automatic replay, transparent reconnect/resume, or `Last-Event-ID` recovery.

Streamable HTTP remains supported and may return a bounded
`text/event-stream` response. That response encoding is part of Streamable
HTTP and must not be confused with the excluded standalone SSE transport.
Static Host environment-backed Authorization headers remain supported on the
v1/v2 compatibility path, but they are not an OAuth login or conformance claim.

Normative upstream references for this release are the MCP
[protocol versioning rules](https://modelcontextprotocol.io/docs/learn/versioning),
the [revision changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog),
and the [Python SDK v2.0.0 release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0).

## Server Manifest V1

Manifests can be YAML or JSON, either as a direct mapping or wrapped under
`mcp_server:` or `server:`.

```yaml
schema_version: 1
server_id: demo-mcp
transport: stdio
stdio:
  command: python3
  args: ["-m", "demo_mcp_server"]
  env:
    DEMO_TOKEN: AGENT_LIBOS_MCP_DEMO_TOKEN
tools:
  - tool_id: forecast
    mcp_name: weather.forecast
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    input_schema:
      type: object
      additionalProperties: true
timeout_s: 10
max_request_bytes: 65536
max_response_bytes: 1048576
```

For Streamable HTTP, use `transport: streamable_http` and an `http:` block:

```yaml
http:
  url: https://api.example.test/mcp
  headers:
    Authorization:
      env: AGENT_LIBOS_MCP_DEMO_TOKEN
      prefix: "Bearer "
```

These blocks show the complete manifest structure, but they are not a live
demo service. `api.example.test` is a reserved documentation host,
`demo_mcp_server` is a placeholder module, and the environment variable name
must be mapped to a credential supplied by the Host. Save an adapted manifest
to a real file (the CLI section calls that user-created path `server.yaml`)
before running the registration commands.

`stdio.env` maps child process environment variable names to host environment
variable names. The runtime does not inherit the full host environment. On
Windows, it additionally forwards `SYSTEMROOT` and `WINDIR`, when present,
because they are child-process bootstrap variables rather than manifest
credentials. A bare Windows `command` additionally requires explicit child
`PATH` and `PATHEXT` mappings. Target resolution searches only non-empty,
absolute directories in that captured `PATH`, considers only `.exe`/`.com`
entries from the captured `PATHEXT`, and never searches the Runtime current
directory or reads ambient `PATH`/`PATHEXT`.

The accepted v1 shape is closed: unknown server, transport, tool, or header
fields are rejected instead of being silently ignored. `metadata` and JSON
Schema contents remain application-defined mappings. Field types are strict:
mapping/array/string fields must use that exact YAML/JSON shape and every
`stdio.args` entry must be a string. Explicit `null` is accepted only where
documented (`stdio.cwd` and `rollback_status`); it is not a default for other
fields. In particular, `args: "-m"` is rejected rather than split into
characters, and malformed `input_schema` values cannot silently become `{}`
and disable validation or live-schema pinning.

| Mapping | Required fields | Optional fields and defaults |
| --- | --- | --- |
| server | `server_id`, `transport`, non-empty `tools`, and exactly the transport block selected by `transport` | `schema_version: 1`, `timeout_s: config.mcp.timeout_s`, `max_request_bytes: config.mcp.max_request_bytes`, `max_response_bytes: config.mcp.max_response_bytes`, `metadata: {}` |
| `stdio` | non-empty single-token `command` without `~` expansion; on Windows, a path-qualified command ends in `.exe`/`.com`, while a bare command maps child `PATH` and `PATHEXT` | `args: []`, `env: {}`, `cwd: null` |
| `http` | `url` | `headers: {}` |
| tool | `tool_id`, `mcp_name`, `right`, `rollback_class`, `state_mutation`, `information_flow` | `rollback_status` as mapped below, `input_schema: {}`, `metadata: {}` |
| HTTP header | `env` | `prefix: ""`, `suffix: ""` |

`transport` is `stdio` or `streamable_http`; the inactive transport block is
forbidden. `right` is `read`, `write`, or `execute`. `rollback_class` accepts
`irreversible`, `rollbackable`, `no_rollback_required`, or `unknown`, while a
non-null `rollback_status` accepts `not_supported`, `not_applied`,
`not_required`, or `unknown`. An omitted `rollback_status` and an explicit
YAML/JSON `null` have the same meaning: the default MCP provider maps them as
follows:

| `rollback_class` | Effective omitted `rollback_status` |
| --- | --- |
| `irreversible` | `not_supported` |
| `rollbackable` | `not_applied` |
| `no_rollback_required` | `not_required` |
| `unknown` | `unknown` |

A supplied non-null `rollback_status` is preserved instead of applying this
default mapping. The durable manifest retains an omitted or explicit-null
status as `null`, while registry inspection/tool listing and the default
provider's effect classification expose the effective mapped value shown
above.

A tool cannot combine `no_rollback_required` with `state_mutation: true`. A
non-empty `input_schema` must be valid JSON Schema and is pinned exactly against
live tool metadata before calls. If the live server omits the schema or reports
`{}`, that is a mismatch rather than an opt-out; only a manifest whose own
`input_schema` is `{}` leaves the live schema unpinned.

With `DEFAULT_CONFIG`, omitted limits resolve to `timeout_s: 10`,
`max_request_bytes: 65,536`, and `max_response_bytes: 1,048,576`. Manifest
values must be positive and cannot exceed the active hard limits (60 seconds,
1,048,576 request bytes, and 8,388,608 response bytes by default). Manifest
text is capped at 262,144 bytes. Server/tool ids are capped at 96 characters,
`mcp_name` at 256, header names at 128, and resolved header values at 8,192;
deployments may customize these values through `AgentLibOSConfig.mcp`.

Manifest v1 is a compatibility contract. It must omit `protocol_mode`, always
uses the legacy initialize handshake, does not send a modern discovery probe,
does not follow `nextCursor`, and rejects any present, non-null `nextCursor`
(including the empty string). Safe v1 manifests preserve their existing
canonical identity, approval binding, and Sink digest.

## Server Manifest V2

Manifest v2 has the same closed transport and tool allowlist shape, but requires
an explicit protocol mode:

```yaml
schema_version: 2
protocol_mode: auto
server_id: demo-mcp-modern
transport: streamable_http
http:
  url: https://api.example.test/mcp
  headers:
    Authorization:
      env: AGENT_LIBOS_MCP_DEMO_TOKEN
      prefix: "Bearer "
tools:
  - tool_id: forecast
    mcp_name: weather.forecast
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    input_schema:
      type: object
      additionalProperties: true
timeout_s: 10
max_request_bytes: 65536
max_response_bytes: 1048576
```

`protocol_mode` has exactly three values:

| Mode | Wire behavior |
| --- | --- |
| `legacy` | Uses the initialize-based legacy path only. |
| `auto` | Attempts modern discovery, then performs only the transport-specific safe fallback described below. |
| `2026-07-28` | Requires modern discovery to advertise that exact revision and never falls back. |

The release-supported protocol set is an Agent libOS constant, not whatever a
later compatible SDK package happens to add. A future SDK minor therefore
cannot silently expand the negotiated protocol set.

`auto` gives a probe at most five seconds and always truncates it to the
operation's shorter remaining absolute deadline. Stdio may fall back only for a
specification-recognized non-modern response or probe timeout. Streamable HTTP
may fall back only for the recognized legacy `400` response. Authentication
failures, `5xx`, DNS/TLS/HTTP timeouts, malformed or oversized responses, and
recognized modern errors never trigger fallback. Fallback is forbidden after a
Tool call has been dispatched.

Manifest v2 follows `tools/list.nextCursor` on the same connection and within
the same deadline. The catalog is capped at 16 pages and 100 tools; malformed
or repeated cursors, duplicate tool names, and either cap being exceeded fail
closed. A call completes this bounded live catalog validation before
dispatching `tools/call`. Manifest v1 retains its single-page contract.

The v2 `input_schema` is a bounded JSON Schema 2020-12 subset. Its root remains
an object; local acyclic `$ref`/`$defs`, composition, and conditionals are
accepted, while external, dynamic, or recursive references are rejected.
Validation caps schema depth at 64, nodes at 10,000, reference hops at 128, and
composition expansion at 1,024. Server `outputSchema`, annotations, and cache
hints are diagnostic metadata only and never expand authority.

For Manifest v1, v2, and v3, `pattern` and `patternProperties` share a maximum
of 4,096 regex evaluations and one 50 ms monotonic matching deadline, and each
UTF-8 pattern is capped at 1,024 bytes. The corresponding Host settings are
`schema_regex_max_evaluations`, `schema_regex_match_timeout_s`, and
`schema_regex_pattern_max_bytes`. Invalid/oversized regex, timeout, or budget
exhaustion fails closed before provider dispatch; these limits are not weakened
to make a remote schema appear compatible.

## Server Manifest V3

Manifest v3 is a new, non-downgradable authority contract. It requires
`schema_version: 3` and exact `protocol_mode: "2026-07-28"`; `legacy` and
`auto` are invalid. It supports `stdio` and `streamable_http` only and requires
at least one declared Tool, Resource, Resource Template, or Prompt. Every
closed object layer rejects unknown fields and strict scalar/list types; open
`metadata` and Tool `input_schema` values must still be finite strict JSON.

The repository ships complete, validated examples for
[stdio](../examples/mcp/stdio-v3.yaml) and
[loopback Streamable HTTP](../examples/mcp/http-v3.yaml), plus a runnable
[end-to-end tutorial](../examples/mcp/README.md). Their common shape is:

```yaml
schema_version: 3
server_id: modern-demo
transport: streamable_http
protocol_mode: "2026-07-28"
http:
  url: http://127.0.0.1:8765/mcp
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    rollback_status: not_required
    state_mutation: false
    information_flow: true
    input_schema: {}
resources:
  - resource_id: status
    remote_uri: demo://status
    right: read
    information_flow: true
    model_visible: false
    mime_types: [application/json]
resource_templates:
  - template_id: greeting
    remote_uri_template: "demo://greeting/{name}"
    variables: [name]
    right: read
    information_flow: true
    model_visible: false
    mime_types: [text/plain]
prompts:
  - prompt_id: review
    mcp_name: demo.review
    argument_names: [subject]
subscriptions: []
timeout_s: 10
max_request_bytes: 65536
max_response_bytes: 1048576
```

V3 Resources and Resource Templates are read/information-flow surfaces. Their
manifest ids are local logical ids; `remote_uri` and
`remote_uri_template` are registered transport selectors, not permission to
open a local or arbitrary network URI. `model_visible` defaults to false and
cannot turn a declaration into authority. Prompt ids similarly map to pinned
remote names and argument-name allowlists. Returned prompts always carry
untrusted provenance and require explicit user confirmation before use.

For Streamable HTTP, optional `auth_profile_id` is an opaque reference to a
Host-constructed OAuth profile; neither client secrets nor tokens belong in the
manifest. Profiles use `registration_mode: preregistered` or `cimd`. DCR is
rejected rather than silently attempted. OAuth is exact-v3 only: the Host owns
status/begin/complete/revoke/logout, PKCE/state and token custody, while the
transport receives only the selected access token through a private broker.
An `auth_profile_id` cannot coexist with a manifest static `Authorization`
header; that ambiguous dual-credential configuration is rejected during
offline validation and registration, before provider work. On dispatch, the
selected access token reaches the governed transport only as the synthetic
`AGENT_LIBOS_INTERNAL_MCP_OAUTH_TOKEN` entry inside that operation's
provider-phase environment snapshot; a value supplied under that name through
a manifest or the Host environment is rejected by the default env allowlists,
and the entry is redacted like the underlying lease secrets.

The optional `subscriptions` array is a closed subset of
`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`,
`resourceSubscriptions`, and `taskIds`. A `taskIds` subscription requires:

```yaml
tasks_extension:
  extension_id: io.modelcontextprotocol/tasks
  spec_sha256: <64-lowercase-hex-Host-pin>
```

The digest is a Host pin for the extension specification, not a value learned
from the server. Task ids, request-state values, continuation ids, OAuth state,
and subscription transport handles remain opaque Host state. Manifest v3
rejects Apps metadata, case-insensitive `ui://` selectors, and every
media-type-equivalent spelling of `text/html; profile=mcp-app` rather than
trying to sandbox-render them. This rule also applies to each Completion
suggestion from either the SDK adapter or a custom Prompt Provider; exact-secret
redaction cannot turn Apps navigation or render instructions into public text.

Canonical v3 JSON and its SHA-256 are independent of v1/v2 canonicalization.
The ordinary Host `register_server`/YAML path accepts a typed v3 manifest under
the same create/explicit-replace authority rules as v1/v2. A reviewed
export/import or any replacement based on an earlier registry observation must
instead use the exact Host CAS bridge
`import_v3_manifest(..., expected_current_sha256=...)`; tooling that cannot
reach that bridge may validate and plan, but must fail before mutation rather
than downgrade v3 or perform a non-CAS replacement.

## Modern Host Client API

`McpModernClient` is the Host-facing v3 Resources/Prompts/Completion manager.
It owns no ambient registry, credential, or transport state. A Host supplies a
`McpClientBindingResolver` returning the exact manifest digest, registry
generation, auth generation/principal digest, owner, and an operation-local
sensitive-value snapshot, plus async Resource/Prompt providers. The binding is
resolved before and after every operation; any registry, owner, or auth fence
change fails the result closed.

The synchronous methods, with `a`-prefixed async equivalents, are:

- `list_resources` and `list_resource_templates`;
- `read_resource` using a local `resource_id` plus exactly the declared
  template variables;
- `list_prompts` and `get_prompt` using a local `prompt_id` plus only declared
  string arguments; and
- `complete_prompt` (also spelled `complete` for provider-style callers) for a
  declared Prompt or Resource Template reference. Its `argument` is exactly
  `{"name": <declared-name>, "value": <string>}`; arbitrary argument maps are
  rejected.

List methods return bounded `McpPage` values. A provider cursor is never
returned raw: the client stores it only in a bounded, in-memory vault and
returns a single-use opaque `mcpcur_*` handle bound to server, surface,
manifest/auth/owner fence, expiry, and seen-cursor digests. Restart, expiry,
wrong-surface use, replay, or a repeated provider cursor fails closed and
requires a fresh list. The vault is capped by `mcp.cursor_handle_limit`.
Cache hints are advisory, their TTL is capped by
`mcp.cache_hint_ttl_cap_ms`, and they never expand the manifest allowlist.
There is no modern MCP response-body cache or cross-principal cache reuse.

Resource read and Prompt get return exactly one of `McpComplete`,
`McpInputRequired`, or `McpRemoteTask`. Completion is Complete-only because the
released SDK has no continuation parameters; any non-Complete Completion result
fails typed before a Human request, durable row, evidence payload, GUI, or model
projection. SDK and custom Completion results share the same closed public
shape: a bounded tuple of strings, an optional non-negative integer `total`,
and a strict boolean `has_more`. Resource contents retain
`provenance: untrusted_mcp_resource`; large/binary contents require a Host
artifact writer and expose only a byte length, MIME type, and digest receipt.
Resource links remain inert. Prompt messages retain
`provenance: untrusted_mcp_prompt` and
`user_confirmation_required: true`. Provider metadata, icons, descriptions,
schemas, cache hints, completion values, and task status never grant authority.

`McpSdkV2SessionProvider` is the adapter for real Python SDK v2 sessions. Its
factory must yield an already governed, exact-`2026-07-28` session and owns DNS,
credentials, immutable stdio snapshots, transport limits, absolute deadlines,
and lifecycle fences. The adapter intentionally does not open a second direct
SDK transport. Custom providers must meet the same async provider contracts and
bounded public-result checks, but they are trusted, cooperative Host code rather
than an in-process isolation boundary. Their `async def` methods must not block
the event-loop thread, run unbounded CPU work, or suppress cancellation, and
must honor the supplied absolute deadline. Runtime checks the deadline before
and after dispatch and classifies an entered-provider overrun as unknown with no
automatic replay. Python cannot safely preempt arbitrary in-process code; a Host
that needs a hard boundary must place custom code behind a killable process or
use the built-in governed SDK/transport path, whose I/O deadline remains hard.
The synchronous Resource, Prompt, and Completion facade owns a fresh event loop
for each operation. After the ordinary task-affine cancellation drain, it
closes that loop on a fixed bound and hard-settles a suspended custom coroutine
that yielded but kept swallowing cancellation; it never leaves an
`asyncio.run` shutdown gather or live task on that private loop, and the runner
itself creates no worker thread. This cleanup does not make Python preemptible:
Provider code that blocks the calling thread before its first yield, or starts
an executor/thread that does not finish cooperatively within the deadline,
remains outside the Host SPI contract and must be isolated by the Host.

`RuntimeBuilder` discovers optional, caller-owned modern Host SPIs only from
the substrate attributes `mcp_credential_broker`, `mcp_artifact_writer`,
`mcp_resource_provider`, `mcp_prompt_provider`,
`mcp_subscription_provider`, and `mcp_tasks_provider`. Supplying one preserves
that object's identity and requires its complete callable protocol; there is
no partial or duck-typed success. If no broker is supplied, Runtime constructs
the system-keyring broker lazily, so a non-OAuth Runtime can open on a headless
Host while OAuth profile registration/use fails closed when no secure backend
is available. That default broker accepts only the exact audited classes from
the locked `keyring==25.7.0` build: macOS Keychain, Windows Credential Manager,
Linux Secret Service/libsecret, and KWallet 4/5. It verifies the exact class
object, distribution version, distribution-owned source path, and reviewed
source digest. `ChainerBackend`, a plugin/backend lookalike or subclass, an
unknown third-party backend, and an unreviewed keyring version all fail closed;
a positive keyring priority or a `keyring.*` module name is not evidence. A
Host that deliberately uses another secure credential facility must inject a
complete `McpCredentialBroker` through `substrate.mcp_credential_broker`.
If no artifact writer is supplied, binary Resource content fails closed before
bytes can enter a public result or RuntimeStore.

The optional Host-private `substrate.mcp_oauth_transport` is a trusted,
cooperative synchronous SPI. Its `request` implementation must honor the one
absolute deadline supplied by Runtime; unlike the built-in pinned transport,
Runtime cannot forcibly stop arbitrary Host code that blocks forever. Runtime
nevertheless rechecks the deadline immediately after every returned transport
response and again after the complete OAuth Provider phase (including the
non-secret Store projection), so a late return is recorded as an ambiguous or
failed operation and can never publish success. An `auth.begin` or scope-step-up
challenge that is not returned because deadline/evidence settlement fails is
discarded; ambiguous broker deletion remains manager-owned for close, expiry,
or next-begin cleanup.

Modern custom Provider SPIs declare the exact markers
`mcp_manifest_schema_version = 3` and
`mcp_protocol_revision = "2026-07-28"`. RuntimeBuilder requires every declared
method to be an `async def` with the closed public signature; a legacy provider,
synchronous lookalike, variadic signature, or partial continuation provider is
rejected during composition. The optional substrate attributes also include
`mcp_v3_tool_provider` and `mcp_continuation_provider`; a custom continuation
provider must implement Tool, Resource-read, and Prompt-get continuation. This
structural check does not turn Python code into a preemptible sandbox: the
cooperative deadline/cancellation obligations above remain part of the Host SPI
contract.

A custom Provider-returned `McpInputRequired` or `McpRemoteTask` is not proof of
Host capture. Before the protected Provider phase ends, Runtime reopens the
referenced durable record and requires its public projection and complete
server/manifest generation, owner/auth, original request/effect,
Capability/data-flow, and current Tasks-pin fences to equal this operation.
Missing, stale, cross-operation, or altered refs fail closed before result
accounting or public evidence. Custom continuation results are recursively
Apps-filtered and exact-secret-redacted under the active binding before the
continuation manager or credential broker receives them.
For each continuation or Tasks round, the broker-only `requestState` or remote
task id is added to that operation's exact-sensitive set. Provider-controlled
siblings and Provider exception text are sanitized before protected
settlement, audit/event evidence, Runtime/CLI/GUI output, or another broker
write; only the private durable-manager boundary may recover the validated
remote id needed for its own broker lookup.

Ordinary Resources/Prompts/Completion calls are stateless at the protocol
session layer: each call enters and exits the existing governed transport in
the same async task and never retains or resumes an MCP session id. The
connection supervisor holds only a no-I/O fence permit for such a call.
Long-lived subscription handles are the sole modern read surface retained by
the supervisor. A Prompt preview carries a deterministic `preview_sha256`
computed inside the same registry/auth binding used for its provider call;
confirmation must present that exact digest or the Prompt is fetched only as a
new unconfirmed preview.

When a process becomes terminal, both the subscription manager and connection
supervisor synchronously latch its owner as revoked for the rest of that
Runtime instance before either side performs Store transitions, catalog
traversal, or Provider cleanup. After reopen, MCP facade admission checks the
durable process status and rejects a terminal process id rather than treating it
as a Host actor. The managers detach the owner's local handles and independently
schedule bounded session close;
subsequent acquire, start, prepared-publication, resume, or live-stream
consumption attempts fail closed even if cleanup blocks or faults. Terminal
`lost` status and bounded events already queued before revocation remain
Host-inspectable through the existing owner- and authority-bound facade.
Asynchronous Provider `close` is best-effort because an in-memory, task-affine
handle cannot be reconstructed after restart and MCP does not make close
idempotent. Therefore terminal cleanup certifies local owner denial and catalog
detachment, not eventual remote shutdown, rollback, or a safely retryable
Provider close.

### Subscriptions, MRTR, Tasks, and OAuth lifecycle

`McpSubscriptionManager.start/status/events/stop` is async and accepts only a
non-empty subset of manifest-declared v3 filters. Connections and queues have
Host caps, absolute/idle lifetime fences, per-event byte limits, monotonically
assigned local sequence numbers, and explicit close. Loss becomes `lost`; the
manager does not reconnect or supply `Last-Event-ID`. A sanitized status/event
projection may be stored as evidence, but the live handle/queue is not restored
after restart. `status` may explain the durable `lost` record; `events` for
that record fails with `NotFound` because an empty list would falsely imply
that a live queue had been recovered.

`events(after=N, limit=L)` is one owner-fenced, single-consumer receive cursor,
not a replay snapshot. The first cursor is `0`; every later `after` must equal
the last sequence returned by the preceding successful non-empty read. A
successful batch atomically acknowledges and evicts only that returned prefix,
so unread events remain queued and a timely reader frees capacity before the
next notification. An empty batch does not advance the cursor. A stale or
future cursor, a competing reader using an already-consumed cursor, or a
different owner fails closed. Queue overflow still changes the subscription to
`lost`; Runtime never drops an unread event while reporting a false `active`
state. Terminal in-memory queues use the same drain-on-read cursor until they
are empty.

`notifications/tasks/status` is not part of the general subscription transport
allowlist. Runtime admits that one wire method only for an exact 2026-07-28
`taskIds` listen whose synchronous ingress is installed, whose Manifest Tasks
digest matches the Host review pin, and whose negotiated server advertises the
same extension identifier. It remains dropped during negotiation and on every
v1/v2, unpinned, ordinary-call, or non-Tasks subscription path. Before a Task
event becomes public, Apps keys are removed, exact operation secrets are
redacted, and the bearer-like remote Task id is resolved through its exact
registry/auth/owner fence to a local `task_ref`. The notification is only an
inert refetch hint: it cannot advance durable Task state or create Human work;
that still requires an explicit protected Task operation.

After a validated and sanitized `toolsListChanged`, `promptsListChanged`,
`resourcesListChanged`, or `resourceUpdated` event is admitted and its durable
receipt accounting succeeds, Runtime synchronously revokes that server's local
opaque catalog cursors. The next catalog read must be a full refresh with no
cursor. This callback is local-only: it performs no remote request and cannot
launch a model, Tool, TaskRun, or subscription. Unknown, unsupported Apps, and
task-status event types never reach this catalog invalidation seam.

`McpContinuationManager` captures an `input_required` result from an already
governed operation. The canonical original request, Provider request keys, and
raw `requestState` live only behind a credential broker; the Store retains the
original-request digest and authority fences. On reopen, a local continuation
reference recovers that broker-held request only after its canonical digest
matches the durable fence. The Host first strictly parses and secret-sanitizes
the complete Elicitation request set, preallocates the Human id and an exact
reserved broker slot, and commits their payload-free ownership sidecar. Only then does it
create the ordinary durable Human *question* and write the exact reserved broker
slot. The continuation-row commit atomically advances the sidecar to retirement;
normal or restart cleanup then removes only superseded state, so a crash cannot
leave an undiscoverable Human question or broker value. The Human
preview binds every local input id and schema digest. Public callers receive the local continuation id,
bounded questions, expiry/revision, and the real
`human_request_id`/Human revision/preview digest. There is no placeholder Human
id. Sampling and Roots remain typed unsupported and create neither a
respondable continuation nor a Human question.

Initial continuation and remote-Task results are published in the same
RuntimeStore transaction that settles their originating protected effect.
Every continuation response is prepared before its protected effect completes:
a next `input_required` round or terminal `complete` transition commits with
that effect, while a remote-Task result additionally commits the Task row in
the same transaction. All involved sidecars remain
`prepared` when that transaction rolls back; only after it commits may the Host
finalize their `cleaning/retire` receipts. A crash in either interval is handled
by startup cleanup and never by replaying the Provider call.

`Runtime.mcp.recover_durable_result(effect_id)` is the Host-only recovery
surface for the commit-before-return interval. It reads only a finalized,
committed MCP effect's closed local-ref receipt and then validates the complete
durable fence chain. Initial continuations retain their immutable origin-effect
fence; later rounds additionally bind the current response effect in the sealed
broker envelope. A continuation-to-Task handoff binds its local Task to that
response effect and checks the response receipt before following the original
continuation receipt. It is not a
remote `tasks/list`, does not enumerate handles, and never exposes the remote
Task id, broker reference, request state, or Provider body.
This lookup recovers durable local handles, not arbitrary completed response
bodies. A continuation `complete` result and its terminal state still commit
atomically, but its `McpComplete.value` is not a durable replay payload; after a
commit-before-return crash the Host can verify terminal completion and the
committed effect, but cannot ask Runtime to replay that value.

Respondable MRTR is deliberately surface-specific. Exact-v3 `tools/call`,
`resources/read`, and `prompts/get` have operation-specific continuation wire
methods; Resource and Prompt continuation sends the sealed `requestState` and
the Human-approved `inputResponses` through a new governed request without
reissuing the original call as an automatic retry. Catalog/list operations are
not respondable. The released SDK Completion request has no continuation
parameters, so an `input_required` result from `completion/complete` is rejected
as typed `mcp_continuation_surface_unsupported` before a continuation or Human
question is created.

For Host callers, only the exact `runtime`, `gui`, and local `cli` principals
may create the Human question without a process Capability, and only while the
matching MCP ProtectedOperation provider dispatch is active. The injected Host authorizer
rechecks the server, operation, logical reference, effect, Host-authority mode,
and canonical preview digest before the question is persisted. This is not a
synthetic `human:owner` grant: process callers still require their ordinary
`human:owner` `write` Capability.

`respond` does not accept raw Provider-bound responses. Before settling the
Human question, the Host asks the manager to validate the proposed local ids,
actions, and form values against the current broker-held request mapping and
durable revision/preview fences. An invalid answer therefore leaves the Human
question pending and the continuation unchanged. The Host then settles that
exact Human question using its revision and preview digest; the manager
consumes only the approved canonical JSON answer, repeats the same validation
against the current durable round, and claims the continuation with CAS. A new
Elicitation round creates a new Human question; an earlier round's receipt
cannot be reused across continuations or rounds. This data-entry question is
separate from the protected operation's ASK authority decision. `respond` and
`cancel` use a Host continuation boundary that re-runs Capability, ASK,
data-flow, budget, and pending-first effect checks. A locally certified
not-started dispatch returns to `input_required`; an ambiguous error or crash
becomes `needs_attention`. Pending Human questions and continuation bindings
survive Store reopen, but Runtime startup recovery only reconciles an interrupted
`dispatching` row to `needs_attention`; it never dispatches, resumes, or invokes
the original API automatically. After TaskRun preflight, recovery distinguishes
a precommit `prepared/abort` row from a postcommit `cleaning/retire` row. The
former cancels only the newly preallocated Human/slots; the latter cancels only
the superseded Human and deletes only old broker slots. Both revision-delete the
sidecar and neither constructs or replays the continuation/Task operation.

`McpRemoteTaskManager` similarly exposes a local `task_ref`, never the remote
bearer-like task id. `get` is an explicit re-observation; `update` and `cancel`
use durable revision CAS and a governed boundary. When creation or an explicit
`tasks/get` observes `input_required`, the manager strictly parses the full
request set and creates a new real Human question for that task round; a repeat
observation of the same outstanding round reuses that exact pending question.
`tasks/update` applies the same validate-before-settle and validate-again-after-
settlement fence, consumes only its approved answer, and never accepts raw
UI/model responses directly. A later input round receives a different Human id. There
is deliberately no `tasks/list`. The manifest/Host extension digests,
server/auth/owner binding, origin request/effect, and state transitions must
all match. A `tasks/cancel` acknowledgement means only local
`cancel_requested`, never `cancelled`; the caller must explicitly re-observe a
terminal state. Interrupted or ambiguous update/cancel dispatch is reconciled
during Runtime startup recovery to `needs_attention`, never replayed. The manager
never polls automatically. Remote task ids, status state, outstanding Provider
input keys, and terminal results stay in the credential broker while the Store
retains only local refs, binding/digests, sanitized diagnostics, CAS revision,
and the current real Human request id.

`McpOAuthManager.status/begin/complete/revoke/logout` operates only on
Host-added v3 Streamable HTTP profiles. `begin` returns a bounded authorization
URL/challenge; `complete` consumes the Host callback URL and challenge. The
authorization code necessarily exists transiently in the Host UI/HTTP/Runtime
request path until the single token-exchange attempt. It is never written to
RuntimeStore or credential-broker storage and never enters audit, event,
effect, output, error, or log projections. The GUI/server request cache and CLI
application-owned callback reference are cleared after that attempt; Python or
JavaScript immutable values are released rather than claiming guaranteed
physical zeroization. Client credentials, Runtime-held PKCE/state material,
refresh/access tokens, and revocation material live at rest only inside the
credential broker. The system
keyring uses deterministic profile-scoped client slots and alternating token
generation slots. A later Runtime can use them only after the Host explicitly
adds the same exact profile again; the encrypted token bundle binds the full
non-secret profile authority, its credential generation, and already validated
authorization endpoints. A changed client id, authentication method, redirect,
audience, metadata pin/URL, endpoint origin, or newer Store generation clears
the stale token/refresh credential without attempting a refresh. Browser
challenge/PKCE/state slots are never resumed and are deleted on completion,
expiry, close, logout, revoke, or the next exact profile registration after a
crash. Token and revocation requests are not retried, redirects and ambient
proxies are rejected, and ambiguous rotation/revocation becomes
`needs_attention`. DCR is unsupported by construction.

The runnable [modern lifecycle tutorial](../examples/mcp/README.md#exercise-mrtr-remote-tasks-subscriptions-and-safe-recovery)
uses one persistent Host Runtime to exercise Human-bound MRTR, remote Task
get/update/cancel, subscription start/events/stop, and reopen-to-`lost`
recovery. It also verifies pending-first evidence and scans its SQLite/output
for opaque Provider state. Separate one-shot CLI processes cannot share an
OAuth challenge or a live subscription handle. The CLI therefore completes a
login callback in one foreground Runtime; later commands must explicitly bind
the same strict profile file so the secure broker token can be revalidated and
rehydrated.

## Authority

The Tool Capability resources below apply to both the stable v1/v2 primitive
and exact-v3 Tool calls. `McpModernClient` itself is a Host composition object: its allowlists, bounds,
projections, and fence checks do not manufacture a Runtime Capability or effect
receipt. The built-in model Resource list/read wrappers therefore enter a
separate protected primitive facade before calling that client. Any other Host
exposure of a v3 operation must likewise add the operation-specific Capability,
Human approval, data-flow, budget, pending-first effect, event, and audit
checks. Prompts, Completion, subscriptions, OAuth, continuations, and remote
Tasks are not projected as model tools.

Registry metadata authority:

```text
mcp_server:<server_id>
mcp_server:*
```

Tool invocation authority:

```text
mcp:<server_id>:<tool_id>
mcp:<server_id>:*
mcp:*
```

Model-visible v3 Resource authority:

```text
# list Resources or Resource Templates
mcp_server:<server_id> read + execute

# read one local logical Resource/Template id
mcp:<server_id>:resource:<logical_id> read
mcp_server:<server_id> execute
```

Stdio Resource operations additionally require `process:spawn` `write` and
the exact `mcp_stdio:<sha256>` `execute` launch right. Catalog and read calls
are protected bidirectional operations: registered selectors/variables cross
the executable-bound Sink, provider content returns as
`external:mcp` untrusted ingress, a pending effect precedes DNS/session/provider
work, and registry/auth fence drift discards the result. Process calls reserve
the manifest byte envelope; Host-internal calls omit process authority
reservations but retain data-flow, effect, resource, event, and audit evidence.

`list_mcp_servers` requires `read` on `config.mcp.registry_resource` (default
`mcp_server:*`) and its model schema and result page are bounded by
`config.mcp.server_page_limit`, never the deprecated v1/v2 Tools
`list_limit`. `inspect_mcp_server` and cached `list_mcp_tools` require
`read` on the exact `mcp_server:<server_id>` resource when called by a process.
The process/model-safe list, inspect, syscall, and actor-mode CLI projections
omit v3 Resource and Resource Template selectors, Prompt declarations,
`auth_profile_id`, subscriptions, and the Tasks extension pin. In particular,
they never reveal a `model_visible: false` entry or its raw URI through the
legacy registry-inspection surface; a model must use the protected logical-id
Resource list/read facade. Trusted Host inspection
(`include_sensitive_fields=True`, including the non-actor CLI) may show the
registered declarations for administration, but still never resolves or
returns credential values.
`list_mcp_tools(refresh=true)` crosses the provider boundary to run
`tools/list`, so it also requires `execute` on `mcp_server:<server_id>` and is
recorded as an MCP external read effect. Host/admin refreshes that bypass
process capability checks still record the external read attempt under a host
actor. For `stdio` servers, actor-mode registration, live tool refresh, and tool
calls also require `process:spawn` `write` plus `execute` on the exact
`mcp_stdio:<sha256>` launch resource. Registration authorizes persisting that
executable launch surface; live refresh and calls are the operations that
actually start the local child process.

`discover`/`adiscover` is a protected external read available only for
Manifest v2 `auto` and `2026-07-28` servers. It requires the same exact server
`read+execute` authority and, for stdio, the same local-spawn authorities as a
live refresh. Discovery never registers a live Tool, grants a capability, or
changes a manifest Tool's effect classification.
`inspect_mcp_server` returns that value as `stdio_authority_resource`; its hash
covers the canonical command, args, environment mapping, and cwd. HTTP servers
return `null` for this field. `call_mcp_tool` requires the right declared by the
tool spec on `mcp:<server_id>:<tool_id>`.

Actor-mode registration also reads the user-supplied manifest through the
filesystem primitive, so the actor needs filesystem `read` authority for that
path in addition to the exact server registry authority (and the stdio launch
authority described above).

For a live refresh/call, every finite decision needed by that one composite
boundary is reserved together before provider work: the main tool or server
decision plus stdio `process:spawn` and exact `mcp_stdio:<hash> execute` when
applicable. Repeated selection of the same capability id is deduplicated, so a
single grant satisfying server read and execute is charged once, not twice.
For tool calls, the primitive gates on `server_id` and `tool_id` before loading
server metadata or input schemas. A process without tool invocation authority
gets a generic denial and cannot enumerate registered MCP server metadata
through call errors. This early invocation-authority gate is unrelated to model
tool projection and does not consume a one-shot
tool grant; the exact tool is then authorized after the server spec is loaded,
and any one-shot use from that decision is consumed only after pre-provider
validation has passed.

Per-use Human approval is additionally bound to the immutable SHA-256 digest
of the complete registered server spec and the durable MCP registry generation,
alongside the canonical arguments hash. Register, replace, and unregister each
advance the generation atomically with the row mutation. Approval obtained for
an unregistered id cannot authorize a server subsequently installed under that
id, and even a byte-identical re-registration defeats ABA reuse. Digest-only
binding state is read only after ASK or an already constrained invocation grant
is found, preserving the no-registry-oracle denial path for callers with no
matching authority. Tool calls and live `list_tools(refresh=True)` bind the
captured server digest/generation to the protected operation and compare it
with the live registry inside the effect transaction before every provider
phase. Replace, unregister, or byte-identical re-registration completed before
the first phase therefore calls no provider; a change after an earlier phase
prevents every later provider phase and retains conservative evidence for work
already observed. A per-registry phase guard serializes
register/replace/unregister with the interval from that live compare through
provider-call return; the runtime's single-writer store lease excludes a second
supported Runtime writer from bypassing the in-process guard.

Tool binding and model visibility are not authority. With `DEFAULT_CONFIG`,
the complete process tool tables for `base-agent:v0`, `coding-agent:v0`, and
`review-agent:v0` bind the four server/Tool entries plus the two v3 Resource
list/read entries. Their initial Skill projection contains only the five
bootstrap tools, so none of these MCP schemas is initially model-visible;
activating the exact `agent-libos-mcp` Skill projects them without changing
Capability authority. The narrow direct `research-agent:v0`,
`analysis-agent:v0`, and `operator-agent:v0` images expose the same six
schemas at boot because their workflow domain is Host-selected in advance;
calls still require the exact server/Tool or Resource Capability and Task
Authority. `maintenance-agent:v0`, `toolmaker-agent:v0`, and
`context-compressor:v0` do not bind these tools and cannot activate that
immutable built-in Skill. Custom or committed Images may choose another
complete table/projection. Neither static binding nor later projection lets a
process inspect or call a registered server without the matching capability.

## Data-flow Sink

Tool arguments are egress to `mcp:<server_id>:<tool_id>`. The Sink identity hash
covers the complete server/transport manifest and selected tool manifest, so a
command, URL, header/env mapping, cwd, tool name/schema, limit, or effect-policy
change invalidates prior high-sensitivity trust. The early tool-capability gate
still runs before server metadata lookup; after lookup,
a negative-only clearance precheck rejects impossible egress before exact tool
or stdio authority, argument-schema validation, Host environment access, or
executable resolution. For stdio, that precheck may use the Host trust record's
expected identity hash, but it cannot grant exact clearance or consume a
conditional release. After exact tool and stdio launch authority succeeds, the
runtime reads only a manifest-mapped child `PATH` and, on Windows, `PATHEXT`
when needed to select and hash the executable, then performs exact clearance
against that executable-bound Sink. HTTP exact clearance requires no
header-value resolution. No transport, live validation, DNS, stdio spawn, or
`call_tool` starts during these checks.

Cached `list_tools(refresh=false)` reads and returns registered manifest
metadata only (`refreshed: false`, `response_bytes: 0`); it does not resolve
environment variables, start a transport, create a provider-effect row, or
compare live schemas. A process-initiated live refresh is a bidirectional protected
provider operation with Sink `mcp:<server_id>:list_tools`: the caller's current
flow context is checked before DNS/stdio/provider dispatch, and live metadata
or an after-dispatch provider error is aggregated back as `normal/untrusted`
ingress. A provider-certified not-started failure adds no ingress. A
Host-internal refresh with no process actor uses a public/verified request
context. A successful refresh returns the manifest entries plus each matched
tool's live name, description, schema, and `schema_matches_manifest` result; it
does not add undeclared live tools to the callable set. A trusted MCP Sink does
not grant the MCP tool right, `process:spawn`, exact
`mcp_stdio:<hash>` execute authority, effect permission, or budget. Conditional
release is exact and one-shot; untrusted MCP cannot send above `normal`.

For stdio, Host trust means the executable is an approved recipient of the
arguments. Agent libOS supervises the registered process lifecycle but does not
claim OS-level control over other file/network I/O performed by that program.
Every executable used by the local SDK provider runs from a private Host-owned
content snapshot rather than reopening the authorized path after the final
identity check. Native executables copy only the executable bytes; shebang
scripts retain the bounded, all-or-nothing direct-sibling compatibility mirror.
See [Data Flow](data_flow.md).

## Security Rules

Only manifest-declared tools may be called. Argument-schema and request-size
validation happen before protected preparation; the preflight classifier runs
at the beginning of preparation. Finite capability reservations, the pending
effect intent, and the maximum resource-usage reservation are then created in
one transaction, so a resource-reservation failure rolls all three back.
Runtime environment resolution has operation-specific ordering:

- cached `list_tools(refresh=false)` performs no resolution;
- live `list_tools(refresh=true)` resolves and validates its HTTP headers or
  stdio environment before finite composite authority is reserved or a pending
  effect is created, so a missing value leaves neither reservation nor intent;
- `call_tool` first performs the negative clearance precheck, exact tool/stdio
  authority, target-only stdio `PATH` (plus Windows `PATHEXT`) resolution,
  executable hashing, and exact Sink clearance. It does not resolve HTTP
  headers or any other stdio child value during those gates. Only after finite
  authority has been reserved and the pending call intent prepared does it
  resolve the complete HTTP-header or stdio-child environment, before DNS,
  stdio snapshot/spawn, live validation, or tool dispatch. A missing or invalid
  complete value therefore takes the no-provider-start path, restoring all
  reservations and abandoning that intent.

Each resolving operation materializes the configured HTTP headers or stdio
child variables into an immutable, in-memory dispatch snapshot. For
`call_tool`, a mapped child `PATH` and Windows `PATHEXT` used for executable
selection are captured after exact stdio authority and pinned into that later
complete snapshot; other child values and all header credentials are first
read after protected preparation. On Windows, the complete snapshot also
includes the optional `SYSTEMROOT` and `WINDIR` child bootstrap values. The same snapshot is supplied
to all provider stages, including both `tools/list` and `call_tool` for a legacy
two-session provider. The SDK provider does not read the mapped host environment
variables or Windows bootstrap variables again, so a concurrent change cannot
replace the selected executable or a validated credential before dispatch.
Host variables not referenced by the manifest (or required as Windows bootstrap
keys) are never copied into this input snapshot. Snapshots are not persisted or
included in audit/effect observations.

For non-local Streamable HTTP, reservation and pending-effect persistence
precede DNS because host resolution is itself an external observation; an
ordinary DNS failure therefore consumes the use and finalizes information-flow
evidence even without a tool request. The primitive asks the provider for live
tool metadata and fails closed if the
server no longer exposes the tool or if a pinned `input_schema` changed; those
post-boundary failures do not restore the use.

For Manifest v1/v2 `call_tool`, one absolute deadline begins after canonical
argument handling and the visibility-gated manifest lookup. It covers bounded
schema and regex preflight, target-only stdio environment selection and
executable identity hashing, protected preparation/revalidation, complete
environment snapshotting, primitive DNS, final executable snapshotting,
protocol discovery/initialization, all live `tools/list` pages, validation, and
`call_tool`; each subsequent stage receives only the remaining time. Discovery
and live refresh likewise create their deadline before environment snapshotting
or initial executable identity selection and carry that exact value through
protected preparation, transport setup, negotiation, and every provider page.
A pre-dispatch deadline exhaustion during schema, identity, or protected
revalidation starts no provider and creates no effect record; an exhausted
deadline cannot start the next provider phase.
Executable hashing and immutable snapshot copying check this same monotonic
deadline before and after each bounded read/write chunk and sibling-mirroring
step. The default SDK provider advertises `supports_mcp_absolute_deadline` and
receives the unchanged absolute value through snapshot verification and wire
dispatch; it does not reconstruct a later deadline from remaining seconds.
Existing custom providers without that opt-in marker retain the legacy relative
`timeout_s` SPI, receive only the primitive's current remaining time, and are
checked again immediately before dispatch. They cannot claim end-to-end
absolute-deadline evidence until they adopt the marker.
Manifest v2 also applies `max_request_bytes` and `max_response_bytes`
cumulatively across discovery/initialization, every list page, and the call.
Bounded `McpExchangeReceipt` entries identify each phase, while
`call_started` records whether the consequential Tool phase was entered.
Legacy two-call providers reserve the complete request/response envelope before
dispatch, but settlement follows observed stage progress: completed response
bytes are charged exactly, an ordinary exception with unknown response size
charges only the current stage's `max_response_bytes`, and later stages that
never started charge zero. Thus a call-stage failure after a 128-byte live list
settles `128 + max_response_bytes`, while a list-stage failure settles one
`max_response_bytes`, never the two-stage reserved response maximum. A
provider-certified not-started phase retains the existing narrower release or
prior-stage settlement semantics described below.

HTTP transport shares JSON-RPC's manifest restrictions: HTTPS for remote hosts,
plain HTTP only for local development hosts, no URL userinfo or fragments, no
literal header secrets, no forbidden request headers, and environment-backed
header secrets restricted by `mcp.header_env_allowlist`. The SDK transport also
disables redirects and ambient proxy/environment routing (`trust_env=false`),
uses the platform TLS trust store, forces HTTP/1.1 with no keepalive, and sets
httpcore retries to zero. Its network backend may try another validated address
only while establishing a TCP connection; the custom HTTP transport adds no
retry for an already-issued request after a write/read failure.

The forbidden-header check is case-insensitive. All manifest versions reject
connection framing/routing headers and high-risk protocol, session, resume,
trace, and baggage controls. In particular, neither version may override
`MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, session ids,
`Last-Event-ID`, `traceparent`, `tracestate`, or `baggage`. Manifest v2/v3 also
reserves content-negotiation headers and `Mcp-Param-*`; its protocol request
`_meta` values are generated by the Host adapter. The v1 compatibility contract
continues to permit manifest `Accept` and `Mcp-Param-*` headers and application
metadata under `_meta`, but those exceptions do not permit the high-risk
controls listed above.

For non-local HTTP, the primitive first resolves every address and rejects the
operation if any result is non-public. The SDK connection backend resolves and
applies the same policy again immediately before opening a socket. Unlike the
JSON-RPC provider, it does not pin the socket to the primitive's earlier tuple;
DNS may move among public addresses, but a private, loopback, link-local,
reserved, multicast, or metadata-service destination still fails closed at
connect time.

stdio transport uses argv, not a shell string. The command must be a single
argv token, does not perform `~` home expansion, and args are separate strings.
Windows dispatch accepts only a resolved `.exe` or `.com`; the verified
absolute snapshot path is passed directly to process creation without the MCP
SDK performing a second executable lookup. Manifest-selected environment
injection is restricted by `mcp.stdio_env_allowlist`. For a snapshotted Python
virtual-environment launcher, the SDK adds compatibility values for
`VIRTUAL_ENV`, `PATH`,
`PYTHONNOUSERSITE`, available venv `site-packages` in `PYTHONPATH`, and on macOS
`__PYVENV_LAUNCHER__`. Those values keep the selected live venv usable but do
not attest its dependency or plugin tree; only executable bytes are pinned.

An explicit relative `stdio.cwd` is fixed to a directory handle and inherited
through `/proc/self/fd` on Linux, so replacing the manifest path cannot redirect
the child. The local SDK rejects configured subdirectory cwd on platforms
without that stable-handle mechanism; the Host-owned workspace root remains the
default cwd. A stdio manifest is still a local process-launch surface, so
process actors need explicit `process:spawn` `write` and exact
`mcp_stdio:<sha256>` `execute` in addition to MCP server/tool authority. Each
newline-delimited raw stdio response frame is capped at the manifest's
`max_response_bytes` before JSON parsing or SDK materialization, so an oversized
frame is rejected without first constructing an unbounded text or JSON value.

MCP call arguments and audit context are bounded and sanitized. MCP result
payloads are JSON-serializable; binary-like content is represented by bounded
metadata rather than raw bytes. The serialized-result check applies to all
transports. Streamable HTTP is also bounded before SDK materialization: ordinary
JSON/other response bodies have one cumulative `max_response_bytes` limit, while
long-lived `text/event-stream` responses reset the same limit at each raw SSE
blank-line frame boundary. Requests force `Accept-Encoding: identity`, and a
response carrying any other `Content-Encoding` is rejected before decoding to
avoid an encoded response expanding past the raw limit.

The v1/v2 Tool compatibility client advertises none of the callback or
extension capabilities. A server request for Sampling, Roots, Elicitation,
subscriptions, or an extension on that path is rejected without invoking
Runtime behavior. Its `InputRequiredResult` remains the stable
`mcp_input_required_unsupported` result with
`automatic_retry_disabled: true`.

Manifest v3 Tools use the exact `2026-07-28` provider and the same protected
Capability, Sink, resource, effect, event, and audit path as the other MCP
operations. Their locally projected list is the manifest allowlist; a call can
return only `McpComplete`, `McpInputRequired`, or `McpRemoteTask`. Raw provider
request state, remote task ids, and transport metadata never enter that public
union. Manifest v3 still never advertises Roots, Sampling, or Logging. It may
enable only the manifest-pinned subscription filters, MRTR Host continuation,
and the digest-pinned Tasks extension described above. Those requests are
captured as opaque Host-owned state; they do not call a model, become a new
model tool, or authorize automatic retry/replay.

### Provider Result Bounds

Objects returned by an MCP provider remain untrusted even when the provider is
a custom Host integration. Before a live tool list or call result reaches
accounting, evidence, or a process, the primitive detaches it into a strict JSON
tree and applies all of these bounds:

- maximum nesting depth is 128;
- maximum node count is `min(100,000, max_response_bytes)`;
- aggregate UTF-8 bytes across all string values and mapping keys cannot exceed
  `max_response_bytes`;
- a Manifest v1/v2 live `tools/list` response contains at most the deprecated
  compatibility setting `config.mcp.list_limit` tools (100 with
  `DEFAULT_CONFIG`), with unique non-empty names; Manifest v3 Tool catalogs use
  `config.mcp.tool_catalog_limit` instead;
- For Manifest v1, any present, non-null MCP `nextCursor`, including the empty
  string, is rejected as an incomplete catalog;
  Manifest v2 follows at most 16 pages and rejects malformed or repeated
  cursors;
- the canonical JSON encoding of the complete returned tool list, or of a call's
  `content` plus `structured_content`, cannot exceed `max_response_bytes`; and
- provider byte receipts cannot exceed `max_response_bytes` or under-report
  that canonical encoding.

Cycles, non-string mapping keys, non-finite numbers, unexpected Python objects,
invalid field types, or a violated byte receipt fail closed through a sanitized
provider error. They are never exposed as a partial result. These post-provider
checks complement, rather than replace, the earlier raw stdio frame/stdout and
HTTP body/SSE-frame limits.

There are consequently two distinct oversize outcomes. A provider that has
already materialized a valid call result may return a bounded receipt with
`too_large: true`; the primitive validates that receipt and exposes
`response_too_large`. By contrast, the default SDK provider cannot produce a
safe result receipt when a raw stdio frame/stdout stream or HTTP body/SSE frame
crosses its transport bound. During `call_tool`, its atomic path exposes that
as sanitized `transport_error` with an error type such as `McpStdioFrameTooLarge`,
`McpStdioStdoutTooLarge`, `McpHttpResponseTooLarge`, or
`McpHttpSseFrameTooLarge`. A live `list_tools(refresh=true)` failure is instead
raised as a sanitized provider exception and never returns a partial list.
The atomic validate-and-call path reports any present, non-null `nextCursor` as
`LiveToolValidationError` with `call_started: false` and does not invoke the
selected tool. Because an incomplete catalog has no exact complete-response
receipt, resource settlement charges the list stage's bounded
`max_response_bytes` ceiling.
Neither outcome makes a consequential call safe to replay.

### stdio Resource Limits And Provider Compatibility

When an actor process has any of `max_subprocess_wall_seconds`,
`max_subprocess_cpu_seconds`, or `max_subprocess_memory_bytes`, each stdio
provider dispatch receives the process's remaining cumulative wall/CPU budgets
and peak-memory bound as `SubprocessLimits`. The SDK provider supervises the
whole stdio process tree and terminates it on a limit. CPU/memory limits fail
closed when complete process-tree metrics cannot be obtained; a timeout or
resource-limit failure is not reclassified as a normal MCP result.

The public `McpProvider` protocol retains the legacy
`validate_and_call`/`list_tools`/`call_tool` signatures, including immutable
`runtime_environment` and optional `executable_snapshot` keyword arguments.
An unbudgeted stdio operation can therefore use a legacy custom provider. A
budgeted stdio operation additionally requires the provider to satisfy
`McpSubprocessLimitsProvider`, set `supports_subprocess_limits = True`, and
accept the optional `limits` keyword on all three dispatch methods; otherwise
the primitive rejects the operation before provider dispatch.

Custom stdio providers that resolve executable identity should accept the same
immutable environment snapshot and advertise
`supports_runtime_environment_snapshots = True`. If their
`executable_snapshot_required(...)` reports that the target is mutable, they
must also advertise `supports_executable_snapshots = True` and execute the
supplied Host-owned snapshot rather than reopening the source path. A provider
that cannot resolve an exact stdio executable may still handle
normal-sensitivity data, but the Sink remains unidentified and elevated
clearance fails closed. The bundled `SdkMcpProvider` implements all three
support contracts.

Modern support is an optional `McpModernProtocolProvider` extension rather than
a breaking change to those legacy signatures. A Manifest v2 operation requires
that extension before dispatch. SDK-native response objects are detached into
Agent libOS-owned result types at the Host boundary; they are never exposed as
the public provider contract.

## External Effects

A refreshed `list_tools` call first validates runtime environment values, then
atomically reserves its finite composite authority and persists an
`external_effects` row with provider `mcp`, operation `list_tools`, and
`effect_state: pending` before non-local DNS or the live metadata request. Its
event/audit/classification path CASes the same
`effect_id` to `finalized`. If the provider raises or a post-provider sink
fails, the operation is finalized conservatively when possible; otherwise the
pending/unknown row remains durable.

`call_tool` similarly reserves deduplicated main/stdio authority and creates one
pending row after negative clearance precheck, target-only executable identity
selection, and exact executable-bound Sink clearance. Complete runtime
environment values are resolved after this preparation but before the first
provider phase; a failure there restores/abandons the reservation and row. Once
environment resolution passes, that intent spans non-local DNS, the
mandatory live tool-metadata validation, and the actual tool call: once DNS or
either live provider boundary is crossed, schema drift, transport failure,
event/audit failure, or post-call classifier failure cannot be interpreted as
“no remote effect.” A
successful final path conditionally finalizes the same id; post-call classifier
failure falls back to a conservative classification. A
`ProviderEffectNotStarted` from the first local/stdio live boundary atomically
restores the reservations and abandons the pending row. Non-local DNS is an
earlier information flow, so a live-validation PENS after DNS cannot restore or
abandon. If live validation succeeded and the main `call_tool` then reports
not-started, the validation already flowed server metadata: the intent is finalized
`unknown` with `state_mutation=false, information_flow=true`, not abandoned.

Completed protected phases impose a conservative classification floor. A
non-local primitive DNS observation and every live provider boundary force
`information_flow=true`; the actual tool-dispatch phase, including the default
combined live-validation-and-call phase, also carries the tool's declared
mutation flag. Manifest classification cannot erase an earlier phase
already observed by the composite operation.

Checkpoint reports and benchmark evidence include both finalized and still
pending MCP effects. Neither manifest version compensates remote MCP state.

Call-effect metadata includes the data-flow decision, trust generation/hash,
label/source hashes, and exact Object source refs without persisting the raw
arguments as data-flow evidence.

## Call And Tool-List Results

For Manifest v3, the Python primitive and async facade return the closed modern
union `McpComplete | McpInputRequired | McpRemoteTask`. `McpComplete.value`
contains the detached, bounded JSON Tool result. An input request or remote
Task is captured first and exposes only a local continuation or Task handle;
the remote request state/task id stays Host-owned. The model-facing
`call_mcp_tool` projection follows the same closed union. It never exposes a
transport, raw remote identifier, callback control, or replay flag.

For Manifest v1/v2, the Python primitive, the `mcp.call` Runtime syscall, and
`mcp call` expose the complete `McpCallResult`: `server_id`, `tool_id`,
`mcp_name`, `status`, `ok`,
`result`, `error`, `response_bytes`, `duration_s`, optional negotiated
`connection`, bounded phase receipts, bounded `dispatch_state`, `retry_class`,
and `automatic_retry_disabled`. The model-facing `call_mcp_tool` output is
intentionally narrower only for operation-local negotiation evidence: it omits
`connection` and `receipts`, but includes the three bounded dispatch/retry
fields.

`dispatch_state` is exactly `not_started`, `started`, or `unknown`. Only the
released Manifest-v1 provider certificate or exact built-in wire evidence can
produce `not_started`; a custom Manifest-v2 provider cannot self-certify it.
Failure with `not_started` maps to `retry_class: "reobserve_required"`, which
requires a fresh observation and authorization path. `started` or `unknown`
failure maps to `unsafe_or_unknown`, while success uses `not_applicable`.
`automatic_retry_disabled` is always true: none of these fields authorizes the
SDK, Runtime, model, or Skill to replay `tools/call`. Exact negotiation,
fallback, or phase receipts still require Host-side Runtime/CLI or recorded
effect evidence.

| `status` | Meaning |
| --- | --- |
| `ok` | The tool returned successfully. `result` contains the bounded model-facing `content` and `structured_content` projections. |
| `mcp_error` | The MCP server returned a tool error. `error` contains the stable error envelope plus any bounded projected returned content. |
| `transport_error` | The provider or transport failed. This includes raw stdio frame/stdout and HTTP body/SSE-frame limit failures, because no safe materialized call-result receipt exists. An atomic provider also uses this status with `error_type: "LiveToolValidationError"` when combined live validation blocks dispatch. `error` is sanitized rather than exposing raw exception or credential text. |
| `invalid_response` | The legacy two-call path records this status when mandatory live tool metadata is missing, malformed, or does not match a pinned manifest schema, then raises the validation/provider exception to the caller. |
| `response_too_large` | A provider materialized a valid call result and returned a bounded, primitive-validated `too_large` receipt for canonical result content over the registered limit. Raw transport-limit failures are instead `transport_error`. |
| `input_required_unsupported` | On the v1/v2 Tool compatibility path, the server requested MRTR input, which that path cannot continue. The result is non-retryable; consequential or ambiguously mutating effects remain unknown and a linked Durable Task Run enters `needs_attention`. V3 Host continuation is a separate opaque-state API and does not reinterpret this result. |

On the Manifest v1/v2 path, local argument/schema validation, capability, Human approval, data-flow,
environment, and pre-provider resource failures are raised instead of encoded
as an `McpCallResult` status. Live absence/schema drift has two observable
forms: the legacy two-call path durably records `invalid_response` and raises,
while the atomic SDK path returns `transport_error` with
`error_type: "LiveToolValidationError"`. Status alone is not proof that dispatch
did or did not start; use bounded `dispatch_state`, and use recorded phase/effect
evidence when exact wire proof matters.

The model-facing `list_mcp_tools` output returns `server_id`, `transport`, the
manifest-declared `tools`, `refreshed`, and `response_bytes`. It deliberately
omits `schema_version`, `protocol_mode`, `connection`, and `receipts`. The
Python primitive, `mcp.tools` Runtime syscall, and `mcp tools` CLI additionally
return `schema_version` and `protocol_mode`; a successful live refresh also
includes its optional `connection` and bounded `receipts`. Cached listing has
`refreshed: false` and `response_bytes: 0`, with no operation-local connection
or receipts. A successful live refresh has `refreshed: true` and augments
declared tool entries with matched live metadata; refresh failures are raised
with a sanitized provider error instead of returning a partial tool list.

`discover` returns configured mode, protocol era and exact revision,
sessionless/fallback flags, bounded server name/version, and standard versus
unsupported capability names. The same optional connection projection appears
on successful live Tool listing and call results at the Python primitive,
Runtime syscall, and CLI surfaces, but not in the two model-tool projections
described above. It is current-operation diagnostic state: it is not written
as resumable/reusable MCP session state and is not reused across operations.
Its bounded, secret-sanitized connection and phase-receipt projections are,
however, retained with the applicable effect/event/audit evidence so operators
can diagnose the completed operation without persisting live session material.

## CLI

```bash
uv run agent-libos --db user mcp register server.yaml
uv run agent-libos --db user mcp register server.yaml --replace
uv run agent-libos --db user mcp list --text demo --limit 20
uv run agent-libos --db user mcp inspect demo-mcp
# Manifest v2 with protocol_mode auto or 2026-07-28 only:
uv run agent-libos --db user mcp discover demo-mcp
uv run agent-libos --db user mcp tools demo-mcp
uv run agent-libos --db user mcp tools demo-mcp --refresh
uv run agent-libos --db user capabilities grant <pid> process:spawn --rights write
uv run agent-libos --db user capabilities grant <pid> mcp_stdio:<sha256-from-inspect> --rights execute
uv run agent-libos --db user capabilities grant <pid> mcp:demo-mcp:forecast --rights read
uv run agent-libos --db user mcp call <pid> demo-mcp forecast --arguments-json '{"city":"Beijing"}'
uv run agent-libos --db user mcp unregister demo-mcp
```

`server.yaml` is the user-created, adapted manifest described above; it is not
shipped at the repository root. The placeholder server/module and reserved
HTTP hostname in the manifest examples will not make these commands a live
remote demo.

Exit status follows the structured envelope: every `mcp` subcommand prints its
JSON result and then exits with status 1 when that envelope reports
`ok: false` — for `mcp call` that includes `mcp_error` and `transport_error`
results, so the printed JSON names the exact outcome. The parallel
`jsonrpc call` command deliberately differs: it prints its
`JsonRpcCallResult` and exits 0 even when the result carries `jsonrpc_error`
or `transport_error`, because the structured result is the deliverable and
scripts must inspect its `status`/`ok` fields rather than the exit code. The
general CLI exit-code contract is described in [CLI](cli.md#mcp-commands).

`mcp list` accepts `--text` and `--limit` and returns an envelope
`{"servers": [...], "has_more": <bool>}`. `has_more: true` means the bounded
window is incomplete; this command has no cursor, so narrow `--text` or raise
`--limit` up to the configured maximum. Text search is a case-insensitive
literal substring match over the public `server_id` only; SQL wildcard
characters do not act as patterns, and private manifest fields are not
searchable. `mcp discover` and
`mcp tools --refresh` perform the protected live provider operations described
above. Registering with `--replace` requires `admin` rather than `write` on the
exact server resource.

Registry commands accept the group-level `--actor-pid <pid>` before the
subcommand to enforce that process's
configured registry-list capability (default `mcp_server:*`) or the exact
server capability required by the selected item operation. Without
`--actor-pid`, registry
mutations run as Host admin operations and emit their normal mutation
audit/event evidence. Read-only `list`, `inspect`, and `tools` calls do not
promise a separate admin-operation audit row; they retain the ordinary
authority, data-flow, provider-refresh, and read evidence applicable to the
specific call. Actor-mode `register` also requires filesystem `read` for the
manifest path; stdio registration additionally requires the launch rights
described above. For `mcp call <pid>`, an explicitly supplied group-level
`--actor-pid` must equal the target `<pid>` and adds no authority.

```bash
uv run agent-libos --db user mcp --actor-pid <pid> tools demo-mcp --refresh
```

For stdio actor mode, `mcp inspect` works only after a Host/admin has created the
registry row. It returns the manifest-derived `stdio_authority_resource`, not
the executable-bound data-flow Sink identity hash. A process therefore cannot
bootstrap its first stdio registration or high-sensitivity Sink trust through
`inspect`: either a Host/admin registers first and then grants the returned
launch resource for later actor operations, or a trusted Host integration
validates the manifest and precomputes that exact launch resource before an
actor-mode registration. Executable-bound Sink trust must be resolved and
registered separately by trusted Host code. Neither inspection nor trust
registration exposes resolved header or stdio environment values to the
process.

Per-server register/replace/inspect/tools/unregister authority is checked before
the store loads existing server metadata. `replace=true` always requires
server `admin`; non-replace registration requires `write`. Registration,
replacement, and unregistration commit the server row, stale tool-grant
invalidation, finite composite authority reservation/commit, event, and audit
in one store transaction. The finite composite decisions are reauthorized and
reserved when that transaction begins, before the existing-row lookup and its
duplicate/not-found check or any registry mutation. A duplicate/not-found
outcome or any later row, event, or audit failure rolls back every reservation
and cannot leave a half-published registry mutation. Validation completed
before transaction entry cannot consume finite authority.

The optional SDK-backed provider requires:

```bash
uv sync --frozen --extra mcp
```

The extra installs the reviewed Python MCP SDK exactly at `mcp==2.0.0`, the
audited credential package exactly at `keyring==25.7.0`, and the directly used
bounded `anyio`, `httpx2`, `httpcore2`, and OpenTelemetry API dependencies.
Agent libOS clears
ambient trace and baggage context at the MCP adapter boundary, installs no
exporter, and does not advertise an OpenTelemetry product capability. All
manifest versions forbid high-risk MCP protocol/session/resume and trace/baggage
headers. Manifest v2/v3 additionally reserve negotiation/content headers,
`Mcp-Param-*`, and protocol `_meta` keys for the Host. Manifest v1 retains only
the compatibility exceptions for `Accept`, `Mcp-Param-*`, and application
metadata under `_meta` described above.

### Host DX workflow

The stable MCP registry CLI remains the authority/effect surface. The
standalone Host helper `scripts/mcp_dx.py` adds offline validation and explicit
review workflows without making configuration model-facing:

```bash
# No registry mutation, DNS, stdio spawn, or provider session:
uv run python scripts/mcp_dx.py validate examples/mcp/stdio-v3.yaml
uv run python scripts/mcp_dx.py doctor examples/mcp/http-v3.yaml

# Secret-reference-only registry backup and a no-mutation import plan.
# Replace this with an absolute persistent target outside the workspace:
MCP_IMPORT_DB=/absolute/external/store/mcp-import.sqlite
uv run python scripts/mcp_dx.py export --db user --server demo-mcp > mcp-export.json
uv run python scripts/mcp_dx.py import-plan --db "$MCP_IMPORT_DB" mcp-export.json

# Apply exactly one reviewed CAS-bound entry:
uv run python scripts/mcp_dx.py import-one --db "$MCP_IMPORT_DB" mcp-export.json demo-mcp \
  --confirm-import --reviewer operator@example --reason "reviewed registry migration"
```

`validate` accepts Manifest v1, v2, and v3 and reports the canonical digest,
surface counts, transport, mode, and stdio launch-authority resource. `doctor`
adds local optional-dependency and manifest-referenced environment-name
presence checks; it never prints values or performs live provider work. For v3
it verifies that the selected Runtime exposes the exact
`import_v3_manifest(..., expected_current_sha256=...)` bridge. A custom Host
composition without that bridge is reported as not ready and apply fails
before mutation.

`export` contains canonical manifests and environment-variable **references**,
never resolved secret values. Its bundle digest covers the closed bundle.
`import-plan` reports create/replace/unchanged actions without mutation.
`import-one` rechecks the expected current digest under the registry guard and
applies one selected server; there is deliberately no partial multi-server
apply when the Store cannot provide one atomic batch transaction.

A live probe is an external observation and requires an explicit confirmation:

```bash
uv run python scripts/mcp_dx.py probe --db user examples/mcp/stdio-v3.yaml \
  --confirm-probe --reviewer operator@example --reason "review unregistered catalogs"
```

The shipped candidate probe accepts an unregistered, exact-v3 manifest. The
Runtime validates its Host policy and digest, creates a pending external-read
effect before DNS/session dispatch, then collects complete bounded Tool,
Resource, Resource Template, and Prompt catalogs inside one transport snapshot
and absolute deadline. Stdio executable pinning, Streamable HTTP SSRF policy,
secret redaction, provider-capability admission, pagination/cursor-cycle
rejection, and independent catalog limits remain active. The result has
`catalog_scope: full_catalog`; it does not register the server, grant authority,
or turn live metadata into an allowlist. Its effect, event, and audit record the
fixed `mcp-dx-probe` Host actor plus the explicit reviewer/reason.

Only that exact-digest report plus `--confirm-scaffold` can create a
deterministic candidate. Generated Tools start at `right: execute`,
`rollback_class/status: unknown`, `state_mutation: true`, and
`information_flow: true` with their live input schema pinned. Resources and
Templates start at `read`, `information_flow: true`, and
`model_visible: false`; Prompts remain Host-only. The result is a
non-registerable candidate wrapper; an operator must edit/review it and run the
separate `approve --confirm-review` transition before extracting a registerable
manifest. Confirmation never bypasses Runtime Capability, data-flow, effect,
or resource checks.

Each workflow above is also available as a built-in `agent-libos mcp`
subcommand (see the table earlier in this section): `mcp validate`,
`mcp doctor`, and `mcp probe` take the manifest path; `mcp scaffold create`
and `mcp scaffold approve` replace the flat `scaffold` and `approve` script
modes; and `mcp export`, `mcp import plan`, and `mcp import apply` replace
`export`, `import-plan`, and `import-one`. Both surfaces share the same
`agent_libos.mcp.dx` implementation, confirmation flags, and
reviewer/reason evidence.

## Tools And Syscalls

The Tool-call entries support the stable v1/v2 result contract and exact-v3
closed modern result union. V3 Prompts, Completion, OAuth, subscriptions,
continuations, and remote Tasks remain
Host-only. The built-in MCP Skill may additionally project the two read-only,
model-visible v3 Resource entries below; they still enter the protected MCP
facade and never accept a remote URI or provider cursor. LLM tool interfaces,
when bound in the complete process table and projected into the model tool
table:

- `list_mcp_servers`
- `inspect_mcp_server`
- `list_mcp_tools`
- `call_mcp_tool`
- `list_mcp_resources`
- `read_mcp_resource`

Deno/TypeScript syscalls:

- `mcp.list`
- `mcp.inspect`
- `mcp.tools`
- `mcp.call`
- `mcp.resources`
- `mcp.resource_read`

Syscalls enter the MCP primitive directly. They do not consult the LLM-facing
tool table, and they cannot pass arbitrary transports, URLs, headers, secrets,
or raw MCP tool names.

## Persistence And Checkpoints

MCP server specs are runtime store registry rows. Resolved secret values and
their per-operation immutable snapshots are not persisted. Negotiated revision,
server identity/capabilities, session handles, cursors, and protocol phase state
are operation-local and are not persisted as resumable/reusable MCP session
state. Bounded, secret-sanitized negotiation and phase-receipt projections may
be persisted as effect/event/audit evidence for the operation; that evidence
cannot resume a session, replay a cursor, or authorize a later call.

Store schema v7 adds durable, CAS-revisioned projections for v3 MRTR
continuations and remote Tasks, plus optional sanitized subscription
status/event evidence and payload-free preparation ownership rows. It stores
local continuation/task/subscription refs,
server/manifest/generation/auth/owner/origin bindings, digests, bounded status,
revision, expiry, and dispatch classification. Raw remote task ids,
`requestState`, input payloads, OAuth codes/tokens/state/PKCE material, live
subscription handles/queues, and provider cursors are not Store payloads.
Bearer-like values live only behind a Host credential broker; an unavailable or
missing broker value fails closed rather than reconstructing authority from a
digest. Preparation rows persist only preallocated local Human ids, exact opaque
broker references, hashes, authority fences, and lifecycle state. They also
precommit at most two superseded opaque refs and one old Human id/preview digest.
Atomic main-row commit advances the preparation to retirement; terminal pruning
uses a retirement-only row and atomically deletes the unchanged projection.
Restart cleanup never treats either stage as permission to replay Provider I/O.
Active continuations and remote Tasks have separate Host count ceilings. Expiry
is swept in bounded pages during capture and startup recovery; terminal and
`needs_attention` projections are retained only up to their independent
oldest-first caps. Pruning first persists an exact retirement-only sidecar,
atomically deletes the unchanged terminal projection, and only then retires its
broker slots and pending Human question. Audit, effect, and transition history
remain append-only when the bounded projection is removed.

Startup reconciliation changes a continuation or Task left in a dispatching
state to `needs_attention` with unknown/unsafe dispatch evidence. It does not
send the response/cancel/update again. Subscription records may explain that a
stream was active or lost, but restart does not recreate it. OAuth browser
login state and reusable connection sessions are likewise not reconstructed
from evidence. A token may be rehydrated only from its deterministic secure
broker slot after explicit exact Host-profile registration and a Store
generation check; RuntimeStore evidence contains no token or broker reference.

The long-lived GUI can recover safe local continuation and Task projections
after a browser or Runtime restart without reconstructing a session. An
explicit empty-body
`POST /api/mcp/continuations/{continuation_id}/inspect` calls
`Runtime.mcp.get_continuation` and performs no Provider dispatch. An explicit
remote Task `get` may omit `expected_revision` (or send `null`) only so a fresh
Host renderer can load the current durable revision; `update` and `cancel`
remain revision-CAS mutations. Once loaded, refreshes use the observed revision.
Expired or unknown refs fail closed and invalidate the renderer form. A
requestState-only continuation deliberately exposes an empty input array and
requires the Human-bound empty response `{}`; typed Sampling/Roots unsupported
results expose neither a durable continuation id nor a Human receipt.

Checkpoint snapshots preserve process capabilities that reference MCP
resources, but they do not copy or restore MCP server registry rows. Restore
and fork can still load capability records, but later inspect/call operations
fail closed if the current runtime does not have a matching registered server.
They also never restore or reuse a prior MCP discovery/session; a later live
operation performs a fresh governed negotiation against the current registry
binding.
