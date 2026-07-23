# MCP Client Tools

Agent libOS supports client-only MCP Tools for registered external MCP servers.
Agents cannot pass transports, commands, URLs, headers, or raw server
configuration at call time. They pass only:

- `server_id`
- `tool_id`
- `arguments`

v1 covers MCP Tools only. MCP Resources and Prompts are not exposed as runtime
primitives yet.

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

`stdio.env` maps child process environment variable names to host environment
variable names. The runtime does not inherit the full host environment. On
Windows, it additionally forwards `SYSTEMROOT` and `WINDIR`, when present,
because they are child-process bootstrap variables rather than manifest
credentials.

The accepted v1 shape is closed: unknown server, transport, tool, or header
fields are rejected instead of being silently ignored. `metadata` and JSON
Schema contents remain application-defined mappings.

| Mapping | Required fields | Optional fields and defaults |
| --- | --- | --- |
| server | `server_id`, `transport`, non-empty `tools`, and exactly the transport block selected by `transport` | `schema_version: 1`, `timeout_s: config.mcp.timeout_s`, `max_request_bytes: config.mcp.max_request_bytes`, `max_response_bytes: config.mcp.max_response_bytes`, `metadata: {}` |
| `stdio` | non-empty single-token `command` | `args: []`, `env: {}`, `cwd: null` |
| `http` | `url` | `headers: {}` |
| tool | `tool_id`, `mcp_name`, `right`, `rollback_class`, `state_mutation`, `information_flow` | `rollback_status` as mapped below, `input_schema: {}`, `metadata: {}` |
| HTTP header | `env` | `prefix: ""`, `suffix: ""` |

`transport` is `stdio` or `streamable_http`; the inactive transport block is
forbidden. `right` is `read`, `write`, or `execute`. `rollback_class` accepts
`irreversible`, `rollbackable`, `no_rollback_required`, or `unknown`, while an
explicit `rollback_status` accepts `not_supported`, `not_applied`,
`not_required`, or `unknown`. If `rollback_status` is omitted, the default MCP
provider maps it as follows:

| `rollback_class` | Effective omitted `rollback_status` |
| --- | --- |
| `irreversible` | `not_supported` |
| `rollbackable` | `not_applied` |
| `no_rollback_required` | `not_required` |
| `unknown` | `unknown` |

An explicitly supplied `rollback_status` is preserved instead of applying this
default mapping.

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

## Authority

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

`list_mcp_servers`, `inspect_mcp_server`, and `list_mcp_tools` without live
refresh require server metadata read authority when called by a process.
`list_mcp_tools(refresh=true)` crosses the provider boundary to run
`tools/list`, so it also requires `execute` on `mcp_server:<server_id>` and is
recorded as an MCP external read effect. Host/admin refreshes that bypass
process capability checks still record the external read attempt under a host
actor. For `stdio` servers, actor-mode registration, live tool refresh, and tool
calls also require `process:spawn` `write` plus `execute` on the exact
`mcp_stdio:<sha256>` launch resource. Registration authorizes persisting that
executable launch surface; live refresh and calls are the operations that
actually start the local child process.
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
through call errors. This early visibility gate does not consume a one-shot
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

Tool visibility is not authority. Default images can see the MCP tools, but a
process cannot call a registered MCP tool without the matching capability.

## Data-flow Sink

Tool arguments are egress to `mcp:<server_id>:<tool_id>`. The Sink identity hash
covers the complete server/transport manifest and selected tool manifest, so a
command, URL, header/env mapping, cwd, tool name/schema, limit, or effect-policy
change invalidates prior high-sensitivity trust. The early tool-capability
visibility gate still runs before server metadata lookup; after lookup,
clearance is enforced before argument-schema validation that could otherwise
start a stdio provider, runtime env resolution, live validation, DNS, stdio
spawn, or `call_tool`.

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
For an executable resolved inside the mutable workspace, both live validation
and tool dispatch run a private Host-owned content snapshot rather than
reopening the authorized path after the final identity check.
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
- `call_tool` resolves the same values after finite tool/stdio authority has
  been reserved and the pending call intent prepared, but before DNS, stdio
  snapshot/spawn, live validation, or tool dispatch. A missing or invalid value
  therefore takes the no-provider-start path, restoring all reservations and
  abandoning that intent.

Each resolving operation materializes the configured HTTP headers or stdio
child variables once into an immutable, in-memory snapshot. On Windows, the
same snapshot includes the optional `SYSTEMROOT` and `WINDIR` child bootstrap
values. The same snapshot is supplied to all provider stages, including both
`tools/list` and `call_tool` for a legacy two-session provider. The SDK provider
does not read the mapped host environment variables or Windows bootstrap
variables again, so a concurrent change cannot replace a validated value
before dispatch. Snapshots are not persisted or included in audit/effect
observations.

For non-local Streamable HTTP, reservation and pending-effect persistence
precede DNS because host resolution is itself an external observation; an
ordinary DNS failure therefore consumes the use and finalizes information-flow
evidence even without a tool request. The primitive asks the provider for live
tool metadata and fails closed if the
server no longer exposes the tool or if a pinned `input_schema` changed; those
post-boundary failures do not restore the use.

For `call_tool`, one absolute deadline begins after protected preparation and
covers environment snapshotting, primitive DNS, executable snapshotting, live
`tools/list`, validation, and `call_tool`; each subsequent stage receives only
the remaining time. For live refresh, environment snapshotting and initial
executable identity selection precede its deadline; that deadline then covers
primitive DNS, final executable snapshotting, and `tools/list`. An exhausted
deadline cannot start the next provider phase.
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

For non-local HTTP, the primitive first resolves every address and rejects the
operation if any result is non-public. The SDK connection backend resolves and
applies the same policy again immediately before opening a socket. Unlike the
JSON-RPC provider, it does not pin the socket to the primitive's earlier tuple;
DNS may move among public addresses, but a private, loopback, link-local,
reserved, multicast, or metadata-service destination still fails closed at
connect time.

stdio transport uses argv, not a shell string. The command must be a single
argv token; args are separate strings. Environment injection is explicit and
restricted by `mcp.stdio_env_allowlist`. A stdio manifest is still a local
process-launch surface, so process actors need explicit `process:spawn` `write`
and exact `mcp_stdio:<sha256>` `execute` in addition to MCP server/tool
authority. Each newline-delimited raw stdio response frame is capped at the
manifest's `max_response_bytes` before JSON parsing or SDK materialization, so
an oversized frame is rejected without first constructing an unbounded text or
JSON value.

MCP call arguments and audit context are bounded and sanitized. MCP result
payloads are JSON-serializable; binary-like content is represented by bounded
metadata rather than raw bytes. The serialized-result check applies to all
transports. Streamable HTTP is also bounded before SDK materialization: ordinary
JSON/other response bodies have one cumulative `max_response_bytes` limit, while
long-lived `text/event-stream` responses reset the same limit at each raw SSE
blank-line frame boundary. Requests force `Accept-Encoding: identity`, and a
response carrying any other `Content-Encoding` is rejected before decoding to
avoid an encoded response expanding past the raw limit.

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
pending row after local preflight. Runtime environment values are resolved
after this preparation but before the first provider phase; a failure there
restores/abandons the reservation and row. Once environment resolution passes,
that intent spans non-local DNS, the
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
`information_flow=true`; the actual tool-call phase also carries the tool's
declared mutation flag. Manifest classification cannot erase an earlier phase
already observed by the composite operation.

Checkpoint reports and benchmark evidence include both finalized and still
pending MCP effects. v1 does not compensate remote MCP state.

Call-effect metadata includes the data-flow decision, trust generation/hash,
label/source hashes, and exact Object source refs without persisting the raw
arguments as data-flow evidence.

## CLI

```bash
uv run agent-libos --db .agent_libos.sqlite mcp register server.yaml
uv run agent-libos --db .agent_libos.sqlite mcp list
uv run agent-libos --db .agent_libos.sqlite mcp inspect demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp tools demo-mcp
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> process:spawn --rights write
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> mcp_stdio:<sha256-from-inspect> --rights execute
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> mcp:demo-mcp:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite mcp call <pid> demo-mcp forecast --arguments-json '{"city":"Beijing"}'
uv run agent-libos --db .agent_libos.sqlite mcp unregister demo-mcp
```

Registry commands accept `--actor-pid <pid>` to enforce that process's
`mcp_server:*` or exact server capabilities. Without `--actor-pid`, they run as
audited admin registry operations. Actor-mode `register` also requires
filesystem `read` for the manifest path; stdio registration additionally
requires the launch rights described above. For `mcp call <pid>`, an explicitly
supplied group-level `--actor-pid` must equal the target `<pid>` and adds no
authority.

For stdio actor mode, run `mcp inspect` after host/admin registration and use
the returned `stdio_authority_resource` verbatim for the exact execute grant.

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
uv sync --extra mcp --all-groups
```

## Tools And Syscalls

LLM-facing tools:

- `list_mcp_servers`
- `inspect_mcp_server`
- `list_mcp_tools`
- `call_mcp_tool`

Deno/TypeScript syscalls:

- `mcp.list`
- `mcp.inspect`
- `mcp.tools`
- `mcp.call`

Syscalls enter the MCP primitive directly. They do not consult the LLM-facing
tool table, and they cannot pass arbitrary transports, URLs, headers, secrets,
or raw MCP tool names.

## Persistence And Checkpoints

MCP server specs are runtime store registry rows. Resolved secret values and
their per-operation immutable snapshots are not persisted.

Checkpoint snapshots preserve process capabilities that reference MCP
resources, but they do not copy or restore MCP server registry rows. Restore
and fork can still load capability records, but later inspect/call operations
fail closed if the current runtime does not have a matching registered server.
