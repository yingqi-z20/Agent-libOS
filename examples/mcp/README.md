# MCP deterministic examples

These examples are credential-free and deterministic. They exercise the exact
MCP `2026-07-28` revision without external network access. Run every command
from the repository root.

## Install and validate

```bash
uv sync --frozen --extra mcp
uv run python scripts/mcp_dx.py validate examples/mcp/stdio-v3.yaml
uv run python scripts/mcp_dx.py validate examples/mcp/http-v3.yaml
uv run python scripts/mcp_dx.py doctor examples/mcp/stdio-v3.yaml
```

Validation and doctor are offline: they do not resolve DNS, start stdio, or
open an MCP session. Doctor reports optional-package versions and whether
manifest-referenced environment **names** are present, never their values. It
also reports whether this Host composition has installed the v3 registry CAS
bridge; a missing bridge prevents apply but not validation or import planning.

The two transport manifests declare one Tool (`echo`), one concrete Resource
(`status`), one Resource Template (`greeting`), and one Prompt (`review`). They
deliberately do not declare OAuth, Tasks, subscriptions, or MCP Apps; the
separate lifecycle fixture below supplies those Host-only contracts without
turning the basic transport manifests into an over-privileged default.

## Probe and scaffold without registration

```bash
uv run python examples/mcp/run_probe_scaffold_e2e.py
```

This harness probes both unregistered candidates over real stdio and ephemeral
loopback Streamable HTTP. Each probe collects all four bounded catalogs inside
one protected Runtime operation, then creates a conservative, still
non-registerable review candidate. The output proves that the registry remains
empty while two finalized external-read effects and two audit records retain
the explicit reviewer/reason. No live catalog item is registered or authorized
automatically.

## Run both real transports

```bash
uv run python examples/mcp/run_tools_e2e.py
```

The harness starts `http_server.py` on an ephemeral loopback port, registers
the two typed v3 manifests directly, grants exact Tool and stdio-launch
authority to a disposable Runtime process, and calls `demo.echo` once over
stdio and once over Streamable HTTP. It prints the bounded v3 `McpComplete`
projection from each call; the complete closed result union is
`McpComplete | McpInputRequired | McpRemoteTask`. It then closes the Runtime
and HTTP process. The v3 registry identity remains intact throughout; the Tool
call path reuses the governed Runtime boundary without rewriting the manifest
as v2.

To inspect the loopback service manually:

```bash
uv run python examples/mcp/http_server.py --port 8765
```

Stop it with `Ctrl-C`. The server binds only `127.0.0.1`; it never exposes a
public listener. The stdio server is normally launched by the harness because
its stdout is the JSON-RPC transport, not a human console.

## Exercise the modern Host contract

```bash
uv run python examples/mcp/run_modern_contract_e2e.py
```

This second harness parses the v3 manifest and runs the public
`McpModernClient` Resources, Resource Templates, Prompts, and Completion APIs
against a deterministic no-I/O provider. Its output demonstrates the local
logical-id projection, inert pagination shape, untrusted Resource/Prompt
provenance, and mandatory prompt confirmation flag.

The no-I/O provider is intentional. A production SDK provider must use
`McpSdkV2SessionProvider` with a Runtime-owned governed session factory that
already enforces DNS/transport credentials, immutable stdio snapshots,
absolute deadlines, bounds, lifecycle fences, and effect policy. Constructing
an official SDK client directly inside a Resource/Prompt provider would bypass
that boundary and is not a supported tutorial shortcut.

## Exercise Host-preconfigured OAuth and restart recovery

```bash
uv run python examples/mcp/run_oauth_e2e.py
```

The scripted local transport returns fixed metadata and token responses without
network access. The real Runtime OAuth facade performs begin/complete, records
protected evidence, keeps Runtime-held PKCE/state and tokens inside the
injected credential broker, and holds the callback authorization code only in
transient Host memory for one token-exchange attempt. The code never enters
Store, broker storage, evidence, output, errors, or logs. The harness also
demonstrates that reopening without those broker credentials becomes
`needs_attention` rather than silently restoring or refreshing them. It then
reconfigures the Host profile and logs out. The output contains no token, code,
state value, or callback URL.

Production system-keyring credentials use deterministic profile-scoped slots.
Reopening with that same secure broker and explicitly registering the exact
same Host profile can rehydrate a still-valid token after its profile digest
and durable credential generation are checked. Changing any profile authority
field fails closed and removes the old token/refresh bundle. Browser challenge,
state, and PKCE values are always ephemeral and are never resumed.

This is one persistent Host process for `begin` through `complete`; it is not a
promise that two separate one-shot CLI invocations can share an in-memory OAuth
challenge. Production Hosts must configure the profile and credential broker,
then keep the Runtime that owns the challenge alive through the callback.

## Exercise MRTR, remote Tasks, subscriptions, and safe recovery

```bash
uv run python examples/mcp/run_lifecycle_e2e.py
```

This credential-free fixture uses deterministic, caller-owned providers while
all product operations enter the public `Runtime.mcp` facade. It demonstrates:

- a Tool returning `McpInputRequired`, a real durable Human question, Runtime
  close/reopen, and one explicit continuation response without replaying the
  initial Tool;
- remote Task creation through a manifest-pinned extension, explicit
  `get`/Human-bound `update`, and `cancel_requested` followed by an explicit
  `get` that observes `cancelled`; and
- one subscription event and explicit stop, plus a second live subscription
  with an unread queued event that becomes `lost` after reopen with no
  automatic relisten and no restored event queue (`events` fails with typed
  `NotFound` instead of pretending that an empty live queue was recovered).

The script asserts pending-first effects at each protected facade's initial
Provider dispatch and verifies the resulting effect/audit evidence. It scans
for exact fixture sentinels to prove that raw request state, Provider input
keys, remote Task ids, and the live subscription handle occur in neither
SQLite nor public JSON. The caller-owned in-memory credential broker is reused
for the successful Runtime reopen; an isolated reopen with an empty broker is
also executed and must fail closed before the explicit Human continuation can
proceed. The demo keeps its subscription Runtime alive while listening.
Separate one-shot CLI processes do not preserve a live handle and must not be
presented as a foreground listener.

## What is intentionally absent

- No MCP server product surface is added to Agent libOS; these tiny servers are
  external test fixtures.
- Apps HTML, `ui://`, and Apps metadata are not rendered or executed.
- OAuth DCR is unsupported. The deterministic OAuth fixture uses a
  Host-preconfigured preregistered profile and injected scripted broker;
  production metadata, issuer identity, TLS, and secret custody remain an
  operator deployment gate.
- Roots, Sampling, and Logging callbacks are not advertised.
- The deprecated standalone SSE transport is not used. Streamable HTTP may
  legitimately return bounded `text/event-stream` frames.
- The lifecycle providers and their private result-adapter hook are repository
  test-fixture plumbing, not supported Host SPI sample code, an Agent libOS MCP
  server, or a shortcut around production transport governance. Real providers
  still need Host-owned credential custody and the governed SDK/transport
  boundary.

On transport loss or restart, do not replay a consequential request. The
legacy v1/v2 `McpCallResult` exposes `dispatch_state`, `retry_class`, and
`automatic_retry_disabled`; the exact-v3 public union does not. V3 no-replay
classification stays in protected effect and durable continuation/task
evidence, and interrupted dispatch reconciles to `needs_attention`. A new
explicit, reauthorized observation is required before deciding what to do
next.
