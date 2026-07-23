# JSON-RPC Over HTTP

Agent libOS supports client-only JSON-RPC 2.0 over HTTP for remote resources.
Remote calls are libOS primitive operations, not ambient network access.

## Boundary

Agents, Skills, and JIT tools never pass URLs, credentials, raw headers, or raw
wire method names at call time. They pass only:

- `endpoint_id`
- `method_id`
- `params`

The runtime first constructs the endpoint/method capability resource from the
two public ids and applies the early visibility gate. Only an authorized caller
can cause the runtime to resolve endpoint metadata from the registry and validate
the method schema. It then performs exact authorization, optionally asks the
human, makes a primitive/provider call through the JSON-RPC provider, records
audit/events, and writes a provider-classified external-effect row. For a
remote host with several validated addresses, the default HTTP provider may
try the next pinned address after any exception during connect, TLS, request
write, response-header parsing, or response-body read. The retry window is the
single endpoint timeout, and an earlier POST may already have reached the
server. Endpoint methods therefore must not rely on a single wire-level POST
attempt for non-idempotency guarantees. A complete HTTP response, including a
non-2xx or redirect response, is returned without trying another address.
A caller without invocation authority gets a generic denial and cannot use call
errors to enumerate registered endpoint metadata. This early visibility gate
does not consume a one-shot method grant; the exact method is then authorized
after the method spec is known, and any one-shot use from that decision is
consumed only after pre-provider validation has passed.

Per-use Human approval is bound to the canonical params hash, the immutable
SHA-256 digest of the complete registered endpoint spec, and the durable
JSON-RPC registry generation. Register, replace, and unregister each advance
that generation in the same transaction as the row mutation. A prompt for an
as-yet unregistered id therefore cannot authorize whatever endpoint is later
registered under that id; the changed digest/generation requires a new prompt.
Re-registering byte-identical policy also advances the generation and defeats
ABA reuse. The primitive resolves this digest-only binding only after it finds
ASK or already constrained invocation authority, so a caller with no matching
authority still cannot probe registry state. The captured digest and generation
are compared with the live registry inside the protected effect transaction
immediately before every provider phase. Replace, unregister, or byte-identical
re-registration completed before the first phase therefore calls no provider;
a change after an earlier phase prevents every later provider phase and retains
the conservative evidence for work already observed. A per-registry phase guard
serializes register/replace/unregister with the interval from that live compare
through provider-call return; the runtime's single-writer store lease excludes
a second supported Runtime writer from bypassing the in-process guard.

## Endpoint Manifest V1

Endpoint manifests can be YAML or JSON. They may be direct mappings or wrapped
under `jsonrpc_endpoint:` or `endpoint:`.

```yaml
schema_version: 1
endpoint_id: demo-weather
url: https://api.example.test/jsonrpc
headers:
  Authorization:
    env: AGENT_LIBOS_JSONRPC_DEMO_WEATHER_TOKEN
    prefix: "Bearer "
methods:
  - method_id: forecast
    rpc_method: weather.forecast
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    params_schema:
      type: object
      additionalProperties: true
timeout_s: 10
max_request_bytes: 65536
max_response_bytes: 1048576
```

The accepted v1 shape is closed: unknown endpoint, method, or header fields are
rejected instead of being ignored. `metadata` and JSON Schema contents remain
application-defined mappings.

| Mapping | Required fields | Optional fields and defaults |
| --- | --- | --- |
| endpoint | `endpoint_id`, `url`, non-empty `methods` | `schema_version: 1`, `headers: {}`, `timeout_s: config.jsonrpc.timeout_s`, `max_request_bytes: config.jsonrpc.max_request_bytes`, `max_response_bytes: config.jsonrpc.max_response_bytes`, `metadata: {}` |
| method | `method_id`, `rpc_method`, `right`, `rollback_class`, `state_mutation`, `information_flow` | `rollback_status` as mapped below, `params_schema: {}`, `metadata: {}` |
| header | `env` | `prefix: ""`, `suffix: ""` |

`right` is one of `read`, `write`, or `execute`. `rollback_class` accepts
`irreversible`, `rollbackable`, `no_rollback_required`, or `unknown`;
`rollback_status`, when supplied, accepts `not_supported`, `not_applied`,
`not_required`, or `unknown`. When `rollback_status` is omitted, the default
provider maps it as follows:

| `rollback_class` | Effective omitted `rollback_status` |
| --- | --- |
| `irreversible` | `not_supported` |
| `rollbackable` | `not_applied` |
| `no_rollback_required` | `not_required` |
| `unknown` | `unknown` |

An explicitly supplied `rollback_status` is preserved instead of applying this
default mapping.

A method cannot combine `no_rollback_required` with
`state_mutation: true`. `params_schema`, when non-empty, must itself be a valid
JSON Schema and is enforced on every call.

With `DEFAULT_CONFIG`, omitted limits resolve to `timeout_s: 10`,
`max_request_bytes: 65,536`, and `max_response_bytes: 1,048,576`. Manifest
values must be positive and cannot exceed the active hard limits (60 seconds,
1,048,576 request bytes, and 8,388,608 response bytes by default). The manifest
text is capped at 262,144 bytes. Identifiers are capped at 96 characters,
`rpc_method` at 256, header names at 128, and resolved header values at 8,192;
deployments may lower or otherwise customize these values in
`AgentLibOSConfig.jsonrpc`.

`schema_version` defaults to `1` when omitted. Repository manifests should
include it explicitly so future migrations are visible in review.

`method_id` is the capability resource fragment. `rpc_method` is the JSON-RPC
wire method sent in the request body. This separation prevents method-name
punctuation from polluting capability resource matching.

## URL And Credential Rules

Endpoint URLs must be HTTP(S). The default rule is HTTPS only. Plain HTTP is
allowed only for local development hosts: `localhost`, `127.0.0.1`, and `::1`.

The registry rejects:

- URL userinfo,
- URL fragments,
- non-HTTP(S) schemes,
- non-local plain HTTP,
- private, link-local, reserved, multicast, or metadata-service IP targets,
- DNS results that resolve a non-local endpoint to loopback, private,
  link-local, reserved, multicast, or other non-public addresses,
- unsafe endpoint or method ids,
- literal secret header values,
- header prefixes outside the approved auth-scheme prefixes and any non-empty
  header suffix,
- forbidden request headers such as `Host` or `Content-Length`.

Headers are environment-backed. The registry stores the environment variable
name and a small approved prefix such as `Bearer `, never the resolved secret
value. Environment names and their allowlist are checked at registration, but
values are resolved only for a call. Each call resolves and validates all
configured headers once into an immutable, in-memory operation snapshot. That
snapshot is passed through DNS and transport dispatch; the default provider
does not read those environment variables again. A concurrent environment
change therefore cannot replace a credential after validation. The snapshot
is neither persisted nor included in audit/effect observations. Parameter-schema and request-size
validation occur before protected preparation; resource-budget and preflight
classifier checks run in its pre-transaction portion. All four precede finite
reservation and pending-intent creation. Header environment resolution is
later: the protected operation
has already reserved finite method authority and prepared its pending effect
intent, but no provider phase has started. A missing or invalid value therefore
contacts neither DNS nor HTTP; the no-provider-start path restores the exact
reservation and abandons the pending intent. DNS is different: it is the first
provider phase, so a successful lookup or an ordinary failure after host
observation commits the use even though no HTTP request was sent.

For remote HTTPS calls, the primitive passes the validated address set to the
default provider, which opens the socket to one of those exact addresses while
preserving the original Host header and TLS server name. This prevents a host
from passing runtime DNS policy and then being re-resolved by the HTTP client to
a different private or loopback address.

The default provider does not follow HTTP redirects. Redirects are treated as
HTTP failures so a registered endpoint cannot silently move a call to a new
host. It also disables ambient HTTP proxy discovery. Remote pinned connections
use the platform TLS trust store, preserve the registered host as the TLS SNI
and HTTP `Host`, force `Connection: close`, and speak HTTP/1.1. Local-development
HTTP uses the same no-redirect/no-proxy policy but does not need DNS pinning.

## Capability Resources

Endpoint metadata authority:

```text
jsonrpc_endpoint:<endpoint_id>
jsonrpc_endpoint:*
```

Method invocation authority:

```text
jsonrpc:<endpoint_id>:<method_id>
jsonrpc:<endpoint_id>:*
jsonrpc:*
```

Method invocation uses the right declared by the method spec. A `read` method
requires `read` on `jsonrpc:<endpoint_id>:<method_id>`. A `write` method
requires `write`, and an `execute` method requires `execute`.

Endpoint registry operations use endpoint metadata authority:

| Operation | Required capability when `--actor-pid` is used |
| --- | --- |
| list endpoints | `jsonrpc_endpoint:* read` |
| inspect endpoint | `jsonrpc_endpoint:<endpoint_id> read` |
| register new endpoint | filesystem `read` for the manifest path plus `jsonrpc_endpoint:<endpoint_id> write` |
| replace endpoint | filesystem `read` for the manifest path plus `jsonrpc_endpoint:<endpoint_id> admin` |
| unregister endpoint | `jsonrpc_endpoint:<endpoint_id> admin` |

These per-item checks occur before the store loads existing endpoint metadata,
so unauthorized register/replace/inspect/unregister attempts cannot distinguish
an existing id from a missing one. `replace=true` always requests `admin` and a
non-replace registration always requests `write`; the right does not depend on
an existence lookup. Registration, replacement, and unregistration commit the
endpoint row, stale method-grant invalidation, event, and audit in one store
transaction. Finite registry authority is reauthorized and reserved when that
transaction begins, before the existing-row lookup and its duplicate/not-found
check or any registry mutation. A duplicate/not-found outcome or any later
row, event, or audit failure rolls back the transaction and restores the exact
one-shot grant; success commits the reservation with the mutation and evidence.

Tool visibility does not grant remote authority. Default images expose
`list_jsonrpc_endpoints`, `inspect_jsonrpc_endpoint`, and
`call_jsonrpc_method`, but a call still fails without the method capability.

## Data-flow Sink

`params` is an egress payload to
`jsonrpc:<endpoint_id>:<method_id>`. After the authority-before-lookup visibility
gate, the runtime hashes the complete endpoint plus selected method manifest as
the Sink configuration identity. A Host trust rule above `normal` must bind
that hash, along with its sensitivity and tenant/principal clearance. Replacing
the URL, wire method, schema, headers, limits, or effect metadata changes the
identity and invalidates old trust.

Clearance is checked before ordinary per-use approval, environment resolution,
DNS, or transport and is revalidated with source Object versions and canonical
params in the protected-operation transaction. A trusted endpoint still needs
the exact JSON-RPC method capability, Task Authority effect permission, and
budget. A conditional high-sensitivity call needs an exact metadata-only
release; an untrusted endpoint cannot be elevated above `normal`. See
[Data Flow](data_flow.md).

## External Effects

The JSON-RPC provider starts classification from the method spec:

- `rollback_class`
- `rollback_status`
- `state_mutation`
- `information_flow`

Completed protected provider phases impose a conservative floor on that
classification. DNS observation makes `information_flow=true`. The transport
phase also makes `information_flow=true`, and is mutation-capable whenever the
method declares mutation or its right is not `read`. Manifest flags can make a
result more conservative, but cannot erase a phase the runtime already
observed.

After schema/request-size/budget/classifier preflight, the runtime atomically
reserves finite method authority and creates an `external_effects` row with
provider `jsonrpc`, operation `call`, and `effect_state: pending`. It then
resolves and validates header environment values inside that protected scope.
If resolution fails, the first provider phase has not started and the
reservation/intent are restored/abandoned together. Otherwise runtime DNS and
the live provider call follow.
Successful or failed transport results emit event/audit evidence, run the post-call
classifier, and CAS that same `effect_id` to `finalized`. A post-call classifier
failure falls back conservatively instead of dropping the effect. If event,
audit, or finalization fails after the transport may have run, the row remains a
durable pending/unknown effect for checkpoint and benchmark reporting.

`ProviderEffectNotStarted` conditionally abandons only when the first DNS
boundary itself certifies that it did not start. Once DNS returned or otherwise
observed the host, a later not-started transport finalizes
`state_mutation=false, information_flow=true` and the reservation stays
committed. Ordinary DNS/transport errors, non-2xx responses, and JSON-RPC error
results are finalized outcomes. Checkpoint restore reports finalized and
pending rows in `external_effects_since_checkpoint` and
`external_effect_summary`. v1 does not perform remote rollback or compensation.

Audit and external-effect metadata store bounded, redacted observations of
`params` with size and hash. Raw params are sent to the registered provider but
are not persisted in audit or provider-effect context.

Successful effect metadata additionally carries the data-flow decision,
registry generation, trust id/hash, label/source hashes, and exact source
Object references, never raw source payloads.

`params_schema`, when present, is validated at registration time and enforced
before each call. Parameter validation failures do not contact the provider and
do not consume one-shot method authority.
After local checks pass, a one-shot method grant is reserved in the same
transaction as pending-effect persistence. It is restored only for a certified
failure before DNS or any other information flow. DNS observation, transport
errors, non-2xx responses, JSON-RPC error results, or a later
certified-not-started transport do not mint another remote-call use.

## CLI

Register and inspect endpoints:

```bash
uv run agent-libos --db .agent_libos.sqlite jsonrpc register endpoint.yaml
uv run agent-libos --db .agent_libos.sqlite jsonrpc list
uv run agent-libos --db .agent_libos.sqlite jsonrpc inspect demo-weather
```

Grant method authority and call as a process:

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> jsonrpc:demo-weather:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite jsonrpc call <pid> demo-weather forecast --params-json '{"city":"Beijing"}'
```

Delete an endpoint:

```bash
uv run agent-libos --db .agent_libos.sqlite jsonrpc unregister demo-weather
```

`--actor-pid <pid>` on registry commands enforces that process's
`jsonrpc_endpoint:*` or exact endpoint capabilities. Method calls always run as
the target pid and are authorized by that pid's method capability. Actor-mode
registration reads the manifest through the filesystem primitive. For
`jsonrpc call <pid>`, an explicitly supplied group-level `--actor-pid` must
equal the target `<pid>` and adds no authority.

Replacing an existing endpoint requires endpoint `admin` when an actor pid is
used. A replace invalidates existing exact method grants for that endpoint so
old authority cannot silently point at a new URL or wire method. The endpoint
row replacement, stale method-grant invalidation, event, and audit happen in
one store transaction; if any part fails, the old endpoint spec remains active.
Unregistering an endpoint also invalidates exact and wildcard method grants for
that endpoint in the same transaction, so reusing the same endpoint id cannot
revive stale method authority.

## Tools And Syscalls

LLM-facing tools:

- `list_jsonrpc_endpoints`
- `inspect_jsonrpc_endpoint`
- `call_jsonrpc_method`

Deno/TypeScript syscalls:

- `jsonrpc.list`
- `jsonrpc.inspect`
- `jsonrpc.call`

Syscalls enter the JSON-RPC primitive directly. They do not consult the
LLM-facing tool table, and they cannot pass arbitrary URLs, headers, secrets,
or raw wire methods.

## Persistence And Checkpoints

Endpoint specs are stored as runtime store registry rows. Resolved header secret
values and their per-call immutable snapshots are not persisted.

Checkpoint snapshots preserve process capabilities that reference JSON-RPC
resources, but they do not copy or restore endpoint registry rows. Restore and
fork can still load the capability records, but later inspect/call operations
fail closed if the current runtime does not have a matching registered
endpoint. A host operator must register provider configuration explicitly.
