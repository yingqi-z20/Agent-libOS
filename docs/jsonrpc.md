# JSON-RPC Over HTTP

Agent libOS supports client-only JSON-RPC 2.0 over HTTP for remote resources.
Remote calls are libOS primitive operations, not ambient network access.

## Boundary

Agents, Skills, and JIT tools never pass URLs, credentials, raw headers, or raw
wire method names at call time. They pass only:

- `endpoint_id`
- `method_id`
- `params`, as a JSON object or array; `null` means that the wire request omits
  the optional `params` member

The runtime first constructs the endpoint/method capability resource from the
two public ids and applies an early invocation-authority gate before registry
lookup. This metadata-visibility check is unrelated to the LLM model-tool
projection. Only an authorized caller can cause the runtime to resolve endpoint
metadata and validate the method schema. It then performs exact authorization,
optionally asks the human, makes a primitive/provider call through the JSON-RPC
provider, records audit/events, and writes a provider-classified external-effect
row. For a remote host with several validated addresses, the default HTTP
provider may try the next pinned address only when the current address fails
before request dispatch starts, for example during TCP connect or TLS
handshake. As soon as request dispatch starts, including an exception from the
request write itself, the provider stops address failover; it never retries
after a response-header or response-body failure. All pre-dispatch address
attempts share the single endpoint timeout. A complete HTTP response, including
a non-2xx or redirect response, is likewise returned without another address
attempt. Thus one logical call can make several connection attempts, but at
most one attempt enters request dispatch. A transport failure after dispatch
remains an uncertain outcome because the server may have received part or all
of the POST, so consequential methods should still use service-level
idempotency and independent read-back rather than caller replay.
A caller without invocation authority gets a generic denial and cannot use call
errors to enumerate registered endpoint metadata. This early authority gate
does not consume a one-shot method grant; the exact method is then authorized
after the method spec is known, and any one-shot use from that decision is
consumed only after pre-provider validation has passed.

The client emits the JSON-RPC 2.0 request shape strictly. Scalar `params`
values are rejected before authority or registry lookup. Responses must be a
JSON object with `jsonrpc: "2.0"`, the matching request id, and exactly one of
`result` or `error`. An Error Object must contain an integer `code` (booleans
are not integers here) and a string `message`; malformed Error Objects produce
`invalid_response` rather than being coerced into remote errors.

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
application-defined mappings. Mapping fields must actually be YAML/JSON
objects: explicit `null`, arrays, strings, or booleans are rejected rather than
being treated as `{}`. In particular, a malformed `params_schema` cannot
silently disable parameter validation.

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

Registration does not perform DNS resolution. It validates the URL shape and
scheme, checks explicitly blocked hostnames, and validates a literal IP address
when the URL contains one. The registry rejects:

- URL userinfo,
- URL fragments,
- non-HTTP(S) schemes,
- non-local plain HTTP,
- non-global literal IP targets other than the explicit loopback development
  addresses above,
- explicitly blocked metadata-service hostnames,
- unsafe endpoint or method ids,
- literal secret header values,
- header prefixes outside the approved auth-scheme prefixes and any non-empty
  header suffix,
- forbidden request headers such as `Host` or `Content-Length`.

A hostname that later resolves to a loopback, private, link-local, reserved,
multicast, or other non-public address is therefore accepted by registration
and rejected when the method is called. Call-time DNS resolution is the first
provider phase. By then the protected operation has reserved finite/one-shot
method authority and created its pending effect row. A successful lookup, or
an ordinary lookup failure after the hostname was observable, commits that use
and records conservative information flow even when no HTTP request is sent.

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
| list endpoints | `config.jsonrpc.registry_resource read` (default `jsonrpc_endpoint:*`) |
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

Tool binding or model visibility does not grant remote authority. With
`DEFAULT_CONFIG`, the complete process tool tables for `base-agent:v0`,
`coding-agent:v0`, and `review-agent:v0` bind `list_jsonrpc_endpoints`,
`inspect_jsonrpc_endpoint`, and `call_jsonrpc_method`. Their initial Skill
projection contains only the five bootstrap tools, so none of these JSON-RPC
schemas is initially model-visible. Activating the exact
`agent-libos-jsonrpc` Skill projects all three without changing Capability
authority. `toolmaker-agent:v0` and `context-compressor:v0` do not bind them and
cannot activate that immutable built-in Skill. Custom or committed Images may
choose another complete table/projection; a projected call still fails without
the method capability.

## Bidirectional Data Flow

The protected `primitive.jsonrpc.call` contract is `BIDIRECTIONAL`. `params` is
an egress payload to
`jsonrpc:<endpoint_id>:<method_id>`. After the authority-before-lookup gate, the
runtime hashes the complete endpoint plus selected method manifest as
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

The return path is ingress. When the first information-flow provider phase is
observed (normally DNS), the runtime aggregates the request/source context with
a `normal`-sensitivity, `untrusted` trust and integrity label whose origin is
`external:jsonrpc`. It propagates that context back from the async worker on
both success and failure. Consequently, response data and provider errors must
remain untrusted input to later tools; even a DNS policy rejection may taint the
active flow because the hostname was already exposed to resolution.

Above-`normal` Sink trust requires the exact endpoint-and-method identity hash.
Actor-mode inspection deliberately redacts URL and header-policy fields, and
the current CLI does not emit this hash, so it is not a supported trust-bootstrap
path. Deployments that need elevated JSON-RPC clearance must provision the hash
from the exact registered manifest in trusted Host integration code and install
the Sink rule before agent work starts. Without that Host-side integration,
leave the Sink at its default `normal` maximum rather than trusting a digest
reported by a process.

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

## Call Result

`call_jsonrpc_method`, `jsonrpc.call`, the Python primitive, and the CLI expose
the same `JsonRpcCallResult` fields: `endpoint_id`, `method_id`, `rpc_method`,
`request_id`, `status`, `http_status`, `ok`, `result`, `error`,
`response_bytes`, and `duration_s`. Local validation, authority, approval, and
resource failures occur before such a result and are raised through the normal
runtime error boundary.

| `status` | Meaning |
| --- | --- |
| `ok` | A valid matching response carried `result`; `ok=true` and `result` preserves that JSON value. |
| `jsonrpc_error` | A valid matching response carried an Error Object; `error` preserves its integer `code`, string `message`, and optional `data`. |
| `http_error` | HTTP was non-2xx, including a refused redirect; `http_status` is present and only a bounded body observation is retained. |
| `transport_error` | No complete HTTP response was available; `error` is the public provider envelope, not the provider-authored message. |
| `invalid_response` | The body was not valid UTF-8 JSON or violated the response/Error Object shape or request-id binding. |
| `response_too_large` | The bounded provider response exceeded the endpoint limit. |

## CLI

Register and inspect endpoints:

```bash
uv run agent-libos --db .agent_libos.sqlite jsonrpc register endpoint.yaml
uv run agent-libos --db .agent_libos.sqlite jsonrpc list
uv run agent-libos --db .agent_libos.sqlite jsonrpc list --text weather --limit 20
uv run agent-libos --db .agent_libos.sqlite jsonrpc inspect demo-weather
uv run agent-libos --db .agent_libos.sqlite jsonrpc register endpoint.yaml --replace
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

`list` accepts `--text` and `--limit`; `register --replace` replaces an existing
row and therefore follows the stronger authority/invalidation rules below.
`--actor-pid <pid>` is a JSON-RPC group option and must appear before the
registry subcommand, for example `jsonrpc --actor-pid <pid> inspect
demo-weather`. It enforces that process's configured
registry-list capability (default `jsonrpc_endpoint:*`) or the exact endpoint
capability required by the selected item operation. Method calls always run as
the target pid and are authorized by that pid's method capability. Actor-mode
registration reads the manifest through the filesystem primitive. For
`jsonrpc call <pid>`, an explicitly supplied group-level `--actor-pid` must
equal the target `<pid>` and adds no authority.

Without `--actor-pid`, registry mutations run as Host admin operations and emit
their normal mutation event/audit evidence. Read-only Host `list` and `inspect`
calls do not promise a separate admin-operation audit row; they retain the
operation-specific read behavior described above.

Replacing an existing endpoint requires endpoint `admin` when an actor pid is
used. A replace invalidates every active endpoint-specific method grant: exact
resources such as `jsonrpc:<endpoint_id>:<method_id>` and the endpoint-prefix
wildcard `jsonrpc:<endpoint_id>:*`. The namespace-wide `jsonrpc:*` capability is
not endpoint-specific and remains active. The endpoint row replacement, stale
method-grant invalidation, event, and audit happen in one store transaction; if
any part fails, the old endpoint spec remains active. Unregistering an endpoint
applies the same endpoint-specific invalidation in the same transaction, so
reusing the same endpoint id cannot revive stale method authority.

## Tools And Syscalls

LLM tool interfaces, when bound in the complete process table and projected into
the model tool table:

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
