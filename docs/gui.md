# Electron GUI

Agent libOS includes a local desktop management console for supervising
processes, messages, human approvals, AgentImage selection/registration/commit,
checkpoints, capabilities, Skills, JSON-RPC endpoints, MCP servers, audit
records, persisted LLM calls, and human Agent ratings.

The GUI is a local-only Electron app. Electron starts
`agent-libos-gui-server`, receives a random session bearer token, and connects
to `127.0.0.1` through HTTP and Server-Sent Events. The renderer never receives
Node.js access; it uses the preload-exposed `libosApi` object.

During development the Electron main process starts the backend without a
shell. It first honors `AGENT_LIBOS_GUI_SERVER_BIN`, then tries the project
`.venv` entrypoint, and only falls back to `uv run agent-libos-gui-server` if no
local entrypoint exists.

## Architecture

```text
Electron main process
  -> starts Python agent-libos-gui-server
  -> owns the random GUI bearer token
  -> exposes limited preload IPC

React renderer
  -> calls localhost HTTP APIs
  -> subscribes to /api/events/stream
  -> renders process, message, approval, audit, and LLM state

Python GUI server
  -> owns Runtime.open(db)
  -> routes all operations through existing runtime managers
  -> never grants capability by GUI visibility
```

SSE sequence ids are scoped to one GUI-server process and the replay buffer is
bounded. A reconnecting client sends its last id as `cursor`. If that cursor is
older than the retained window, or is ahead of the newest id after a server
restart, the server emits `event.invalidated` with
`reason: sse_cursor_not_replayable`, resets the stream cursor, and then replays
the retained events. Clients must fetch `GET /api/snapshot` when they receive
that invalidation; the bundled renderer does so. This makes a replay gap
explicit instead of silently leaving the UI on stale state.

Append-event de-duplication is bounded by the GUI event-buffer configuration.
Immutable events, audit records, messages, and LLM calls use their durable ids.
Human requests use the request id together with `updated_at` and `status`, so a
pending request and its later approved/rejected/cancelled version each produce
a `human_request.updated` event without growing an unbounded in-process set.

The GUI server is not a new security boundary. It is a local admin control
surface over the same primitives, Capability checks, human approval flow,
events, and audit records used by the CLI. Its Python entrypoint lives under
`agent_libos.api.gui` with the CLI because both are host-facing API surfaces.
Only a bearer token holder on the same machine can use it; CORS is limited to
loopback HTTP(S) browser origins plus the exact packaged-renderer origin
`agent-libos://app`; it does not accept `Origin: null` or other custom-scheme
hosts. Packaged Electron serves `gui/dist` through that privileged, secure
custom protocol instead of `file://`, giving browser requests a stable origin
without broadening the server allowlist. The protocol resolver rejects other
authorities, credentials, ports, traversal, and paths outside the distribution
root.

The GUI can display data-flow Audit/Event/Explain evidence produced by runtime
operations, but it does not expose a Sink-trust mutation route or model tool.
Sink trust remains Host configuration/API state; adding a GUI view later must
not turn renderer visibility into `data_flow_sink_registry:*` authority.

For endpoints that accept an optional `actor`, omitting `actor` runs in GUI
admin mode. Supplying `actor` opts into process-authority mode and requires
that process to hold the capability needed by the underlying primitive, keeping
audit attribution aligned with the capability decision.

Closing the GUI server pauses auto-run and asks the scheduler to stop before it
calls `Runtime.shutdown()` on an owned runtime. Every request that reads or
mutates Runtime state registers as an in-flight runtime user, including the
non-serialized health and ObjectTask-wait paths. Shutdown rejects new users,
drains the registered handlers and the runtime lock within one bounded deadline,
then closes the owned Runtime only after its own scheduler/ObjectTask drain
succeeds. `POST /api/shutdown` returns `200 {ok: true, status: "stopped"}` only
after that teardown completes. A drain that has not begun closing the Runtime
returns retryable `503`, leaves the store/broadcaster open, and reopens the
transient gate; once Runtime teardown has begun the API stays closed and a
later shutdown call can continue the phased teardown. The top-level server
retries teardown and fails visibly instead of exiting successfully while the
Runtime is still open. Electron treats only the explicit `stopped` response as
graceful acknowledgement and otherwise uses its bounded process-tree kill
fallback. The sequence never closes a database handle underneath a live
worker. Host shutdown does not mark AgentProcess records as exited; process
lifecycle changes still go through the runtime `process.exit` primitive/tool
path.

## Development

Install Python and GUI dependencies:

```bash
uv sync
npm --prefix gui install
```

Run the Python server directly:

```bash
uv run agent-libos-gui-server --db .agent_libos.sqlite --port 0
```

The GUI server accepts the same runtime store targets as the CLI. SQLite paths
are the default local store; PostgreSQL DSNs require installing the `postgres`
extra and are redacted in startup and health payloads. Persistent stores use an
active-runtime lease, so a GUI server and a writable CLI Runtime cannot open
the same SQLite target or PostgreSQL database/schema concurrently. SQLite uses
the canonical target plus a no-follow sidecar `flock` where available and a
kernel exclusive database lock otherwise; PostgreSQL uses a stable
database/schema advisory key. See [Runtime Storage](storage.md).

The server prints one JSON line containing the selected local URL and bearer
token:

```json
{"url":"http://127.0.0.1:51234","token":"...","db":".agent_libos.sqlite"}
```

Run the Electron app:

```bash
npm --prefix gui run electron:dev
```

Build and type-check the GUI:

```bash
npm --prefix gui run test
npm --prefix gui run typecheck
npm --prefix gui run build
uv run python scripts/test_matrix.py --lane gui
```

`npm --prefix gui run build` removes `dist-electron` before compiling. Vitest
excludes both renderer and Electron build output, and the production Electron
TypeScript configuration excludes `*.test.ts`; generated JavaScript therefore
cannot become a second copy of the source test suite. A useful clean-build
check is:

```bash
npm --prefix gui run test
npm --prefix gui run typecheck
npm --prefix gui run build
find gui/dist-electron -name '*.test.js'
npm --prefix gui run test
```

The `find` command should print nothing, and the two Vitest runs should discover
the same source tests.

The Electron smoke path can be run headlessly with
`AGENT_LIBOS_GUI_SMOKE=1`. By default it verifies the Electron main process,
Python GUI server startup, authenticated `/api/health`, and graceful shutdown
against an in-memory `local` store without creating a BrowserWindow. Set
`AGENT_LIBOS_GUI_SMOKE_WINDOW=1` when a machine has a working desktop/GPU stack
and you specifically want to exercise the packaged custom-protocol renderer,
its API origin, and the preload bridge.

The Vite development server is bound to `127.0.0.1` and restricts file serving
to the `gui/` directory. Production dependency audit should remain clean; any
dev-server advisory must be handled with local-only exposure unless an upstream
fix is available.

## Current Workspace

The first screen is process-centered:

- left pane: process tree, status, image, cwd, resource budget/usage, unread
  message badges,
- center pane: selected process timeline with type filters and human request
  cards,
- right pane: details for overview, capabilities, tools/Skills, checkpoints,
  audit, LLM calls, Images, JSON-RPC, MCP, Object Memory summary, and selected
  process ratings,
- top bar: database, spawn, auto-run, quanta budget, run, step, pause, refresh.

The default user page exposes the image workflow without opening raw runtime
panels: users can choose a registered image for a new task, import an
AgentImage package, or save the current selected process as a checkpoint-
derived image. Import and commit both require explicit confirmation. Saving as
an image creates a checkpoint only after that confirmation, then commits the
checkpoint into an immutable image artifact.

Users can also score the selected AgentProcess from 1 to 5 and add an optional
comment. The GUI stores one current rating per process, default human, and GUI
source. Re-rating updates that current record while the audit log records each
change.

The operator console provides the fuller registry view: image list, inspect,
spawn/exec selection, package registration, checkpoint commit, and explicit
replace controls.

Manual spawn and exec controls include an LLM profile selector. Leaving it blank
uses the image default and then the runtime default; choosing a profile writes
only that profile id to the process. The selector can add, edit, and delete
user profiles for OpenAI-compatible providers. GUI-created profiles are stored
outside the runtime database in the operating system's user application config
area: Electron passes `app.getPath("userData")/llm-profiles.json` to the Python
server, while direct `agent-libos-gui-server` runs default to `%APPDATA%/Agent
libOS/llm-profiles.json` on Windows, `~/Library/Application Support/Agent
libOS/llm-profiles.json` on macOS, and the `agent-libos/llm-profiles.json`
file under `${XDG_CONFIG_HOME:-~/.config}` on Linux. The file stores model routing fields such
as profile id, model, base URL, API mode, tuning options, optional
`context_window_tokens`, and the `api_key_env` name. It never stores the API
key value. When a profile has a base URL,
`allow_custom_base_url: false` is preserved explicitly rather than inferred
away, so disabling custom-base-url use remains stable across GUI restarts.

The scheduler defaults to automatic mode. Users can pause auto-run, step a
selected process, or run the selected process with an optional quantum budget.
Leaving the budget blank runs until the process/runtime becomes idle; entering a
number bounds that run. Automatic runs after spawn/message/exec may advance all
runnable processes, but `POST /api/processes/{pid}/run` is intentionally scoped
to that pid. `POST /api/processes/{pid}/step` is synchronous: its response and
the snapshot it publishes contain the final scheduler state (`running: false`)
after the quantum has completed. Real LLM calls are still persisted in `llm_calls`, so the GUI can
show token usage, errors, full stored LLM inputs and outputs, and bounded
prompt/output observability metadata. This default supports self-evolution
training and fine-tuning pipelines under the deployment's user agreement. If
the host runtime is configured with `llm.persist_full_io=False`, the same
`llm_calls` API returns only bounded previews and hashes for sensitive prompt,
tool, reasoning, and provider payload fields.

GUI background auto-run deliberately sets `process_human_queue=false`. It may
advance runnable model work, but it never auto-approves, auto-denies, or invents
an answer for a pending human request. A human must decide through a request
card (or another explicit host terminal surface), after which normal runtime
wakeup/resume semantics apply.

GUI request serialization is itself a protected Human information-flow exit.
For a conditional high-sensitivity request, the first snapshot/list response
contains a metadata-only release card ahead of a redacted parent request. The
raw request is returned only after that GUI-specific exact release is approved
and consumed; the durable presentation binding prevents duplicate release
requests across polling or Runtime reopen. The withheld parent card has no
answer or decision controls, and the server returns `409 Conflict` if a client
tries to respond to it before protected GUI presentation consumes the release.
The release card shows only the bound Sink, sensitivity, tenant/principal,
payload size and SHA-256, source count, and operation. Arbitrary nested payload
values are not included in pre-release previews. The exact release hashes the
complete gate-independent public view handed to the GUI provider, including
status, timestamps, and `decision`; internal release-link/visibility metadata
does not perturb that view. A source, Sink-trust registry, Task Authority
manifest, public-view, or release-binding change invalidates the durable visible
marker; the parent is redacted again and requires a fresh exact release before
projection (including a newly recorded decision) or response. The freshness guard and Human
decision commit share one store transaction, so concurrent Host registry or
source mutations cannot land between them. Bounded snapshots project only the
final rows they will return: lookahead and release/parent pairing never consume
a release or mark a parent visible for a row cropped from the JSON response.
For an unchanged unrestricted view, one authenticated GUI provider session may
reuse a bounded in-memory presentation receipt after rechecking the exact view
hash and current source/Sink policy under the Store lock. A new server/provider
session never inherits that receipt, so reopen cannot silently reuse ephemeral
presentation evidence.

Pending Human requests are liveness-critical. The Human list returns every
pending request first, followed by a bounded newest-history window. A snapshot
then applies the GUI's general collection-size bound to that pending-first
sequence and reports any omission in `_truncated`: terminal history cannot
displace a pending request, although more pending requests than the collection
bound cannot all fit in one snapshot. `GET /api/human-requests` does not apply
the snapshot collection cap: it returns every pending row plus the Human
list's bounded newest-history window.

Request cards are typed. A permission card requires one of
`always_allow`, `ask_each_time`, or `always_deny`; approving with
`always_deny` and rejecting with `always_allow` are disabled and rejected by
the server. A question approval requires a non-empty string answer. While a
response is in flight the card remains visible and disabled; an HTTP error
keeps its answer/policy draft and shows an error instead of optimistically
removing the authoritative pending request.

`POST /api/processes` and `POST /api/processes/{pid}/exec` accept optional
`llm_profile` fields for host-selected per-process LLM routing. The GUI server
validates those ids before writing a process record. Snapshots expose each
process `llm_profile_id` and a non-secret `llm_profiles` summary list; profile
secrets stay in the host process environment and are not returned by the GUI
API.

Process snapshots include `resource_budget`, `resource_usage`, and
`resource_remaining` so the GUI can show quota state without treating it as a
Capability grant. Budget exhaustion is still enforced by the runtime and
providers, not by renderer visibility.
Snapshot payloads keep field shapes stable when bounding large values: long
strings are returned as truncated strings, and truncation metadata is reported
under the snapshot-level `_truncated` map.

Top-level snapshot collections are bounded before response assembly. Processes,
pending-first Human requests, tools, images, Skills, JSON-RPC endpoints, MCP
servers, Runtime Modules, and LLM profiles fetch at most
`snapshot_collection_max_items + 1`, subject to any stricter subsystem list
maximum. Skills, JSON-RPC, and MCP list APIs perform one additional internal
lookahead even when that subsystem maximum is stricter than the GUI maximum.
Either kind of lookahead becomes a `source_limited` lower-bound entry in
`_truncated` and is not serialized.
Event and audit rows persist a derived `gui_snapshot_visible` flag. Snapshot
queries filter that indexed flag before applying `LIMIT`, preventing internal
GUI-presentation evidence from displacing causal runtime rows. The flag is
required by the 0.3 schema; missing or malformed persisted visibility state is
rejected rather than repaired during open. The full event/audit histories can
still include presentation evidence when requested.
The process window orders non-terminal processes before the most recently
updated terminal history, so a full snapshot does not hide current work behind
old completed rows. If the bounded window contains a child but not its parent,
the process tree renders that child as a temporary root rather than making it
unreachable.
Process message/count, bounded LLM-call-window count/token usage, rating,
ancestor reservation, and hierarchical remaining-budget data are loaded through
batch queries. Message and LLM windows select the newest configured rows per
process; messages are returned chronologically. Snapshot construction therefore
does not issue one message, LLM-call, rating, or resource query for every listed
process.

## High-Risk Operations

The GUI requires explicit confirmation for high-risk operations before sending
the final request to the server:

- process `exec` and `exit`,
- process `signal` requests that cancel or terminate a process,
- workflow runs for side-effecting tools, custom images, or custom working
  directories,
- image package registration and checkpoint-to-image commit,
- checkpoint restore and fork,
- capability grant, delegate, and revoke,
- JSON-RPC method calls,
- MCP server registration and tool calls,
- Skill registration, activation, and unload.

The confirmation dialog shows the pid/resource/action summary. The server also
rejects high-risk requests without `confirmed: true`, so accidental direct HTTP
calls fail closed before invoking the runtime operation.

JSON-RPC endpoint and MCP server registration through the GUI accept manifest
text only. The renderer cannot ask the Python GUI server to read an arbitrary
host file path; file/path based registration remains a CLI/admin workflow.

Image package registration follows the same rule. Electron may read a package
directory selected by the user and pass bounded package file payloads to the
local GUI server, but the server rejects host file paths. The default GUI
request body limit is sized to carry the Electron 16 MiB raw package-file
limit after base64 and JSON wrapping. Registering or
committing an image changes image visibility and baked internal runtime state
only; it does not grant the target image's declared capabilities. Package
workspace grants apply only to the private materialized copy declared by the
package manifest.

GUI Skill registration requires both an `actor` pid and a workspace-relative
`path`. The server resolves that path through the actor-scoped workspace
filesystem and applies the normal filesystem, Human approval, and `skill:<id>`
authority checks. There is no GUI host/admin path-registration fallback;
registering a host path remains a CLI/admin workflow. Global Skill trust is not
exposed as a GUI endpoint and must likewise be managed through CLI/admin
configuration.

## API Contract Boundary

The Electron renderer and Python server are shipped and tested as one build.
The local `/api` surface is not a complete, independently versioned public REST
API, and compatibility for arbitrary external clients is not promised. The
machine-readable [GUI API contract subset v1](gui_api_schema.json) deliberately
covers the snapshot response, JSON error envelope, and payloads for every
operation that the server gates with explicit confirmation. It is JSON Schema,
not a complete OpenAPI document.

`tests/unit/test_gui_api_schema.py` parses that schema, validates representative
payloads, and compares its high-risk operation map with the server's
`_require_confirmed` calls. Renderer types and routes outside this subset remain
same-build implementation details and must be changed together with their
server handlers and GUI tests.

## API Summary

Important endpoints:

- `GET /api/health`
- `POST /api/shutdown`
- `GET /api/snapshot`
- `GET /api/events/stream?cursor=<id>`
- `GET /api/tools?limit=<n>` (default and maximum
  `gui.snapshot_collection_max_items`)
- `GET /api/llm-profiles`,
  `POST /api/llm-profiles`,
  `PUT /api/llm-profiles/{profile_id}`, and
  `DELETE /api/llm-profiles/{profile_id}` for user-level GUI model profiles.
- `GET /api/processes?limit=<n>` (default and maximum
  `gui.snapshot_collection_max_items`), `POST /api/processes`
- `GET /api/operations?pid=...`, `GET /api/operations/{operation_id}`, and
  `GET /api/operations/resolve?kind=...&id=...` for host-only deterministic
  operation explanations. List/detail responses support cursor pagination;
  ambiguous evidence resolution returns `409` with candidate causal roots.
- `POST /api/workflows/run`
- `POST /api/scheduler/auto`, `POST /api/scheduler/pause`
- `GET /api/processes/{pid}`
- `POST /api/processes/{pid}/run|step|pause|resume|signal|message|interrupt|cd|exec|exit`
- `GET /api/processes/{pid}/messages|human-requests|llm-calls|audit|events|capabilities|checkpoints`.
  The LLM-call route's `limit` defaults to and cannot exceed
  `gui.snapshot_process_llm_call_limit`.
  The events route accepts `limit=<n>` (default and maximum
  `gui.snapshot_event_limit`) and `before=<event_id>` for older pages. It selects
  the bounded newest/cursor window in storage and returns that page in
  chronological order; it is not an unbounded full-log endpoint.
- `GET /api/processes/{pid}/rating` and
  `POST /api/processes/{pid}/rating` for the selected process's 1-5 human
  score and optional comment.
- `GET /api/object-tasks`, `POST /api/object-tasks/start`,
  `GET /api/object-tasks/{task_id}`, and
  `POST /api/object-tasks/{task_id}/cancel|wait|watch-owner`
  (`POST /api/object-tasks/start` accepts `owner_watch`, `watch_events`,
  `watch_channel`, and `watch_kind` for owner-change runner messages; the
  `watch-owner` endpoint updates the same fields for an active task. Wait
  requests are bounded by the GUI object-task wait timeout defaults.)
- `GET /api/human-requests`
- `POST /api/human-requests/{request_id}/respond` approves or rejects only
  pending requests; terminal or cancelled requests return a conflict.
  A conditional parent that has not completed its exact GUI presentation also
  returns a conflict without changing the request or process; approving the
  metadata release alone is insufficient until a GUI snapshot/list consumes
  that release through the protected presentation operation.
  `approved` must be a JSON boolean. Permission requests require
  `decision.policy` equal to `always_allow`, `always_deny`, or
  `ask_each_time`, consistent with approval/rejection. Approved questions
  require a non-empty string `answer`; other JSON types are not coerced.
- `GET /api/checkpoints`, `POST /api/checkpoints/create`,
  `GET /api/checkpoints/{checkpoint_id}`,
  `GET /api/checkpoints/{checkpoint_id}/diff`, and
  `POST /api/checkpoints/{checkpoint_id}/restore|fork`
- `GET /api/skills`, `GET /api/skills/{skill_id}`,
  `POST /api/skills/register`, and
  `POST /api/skills/{skill_id}/activate|unload`
- `GET /api/capabilities`, `GET /api/capabilities/{capability_id}`,
  `POST /api/capabilities/grant|delegate|explain`, and
  `POST /api/capabilities/{capability_id}/revoke`
- `GET /api/images`, `GET /api/images/{image_id}`, and
  `POST /api/images/register|commit`
- `GET /api/jsonrpc`, `GET /api/jsonrpc/{endpoint_id}`,
  `POST /api/jsonrpc/register`, and
  `POST /api/jsonrpc/{endpoint_id}/call`
- `GET /api/mcp`, `GET /api/mcp/{server_id}`,
  `GET /api/mcp/{server_id}/tools`, `POST /api/mcp/register`, and
  `POST /api/mcp/{server_id}/call`
- `GET /api/modules`, `GET /api/modules/{module_id}`

All endpoints require `Authorization: Bearer <session-token>`.
Mutation endpoints validate required ids, image names, and paths as non-empty
JSON strings. Malformed enum values such as an unknown process signal, missing
required fields, non-object request bodies, and incorrectly typed booleans
return `400` without invoking the runtime mutation.

The process detail pane includes an Explain tab. It renders an outcome and
evidence-completeness summary, an explicit causal tree, and a filterable
evidence timeline. Audit, event, LLM-call, and Human-request entries in the main
timeline can open the corresponding explanation. Snapshot/SSE updates refresh
the panel against the current store. Explain serialization retains ids,
statuses, hashes, counts, rights, targets, and rollback classification while
redacting Object/LLM/Human/provider content. It is not exposed to model tools or
process syscalls. See [explainable_operations.md](explainable_operations.md).

`POST /api/processes` and `POST /api/workflows/run` accept an optional
`authority_manifest` JSON object. It is a Host/admin-plane launch contract;
the GUI does not synthesize authority from image requirements. Explain shows
the resulting id/hash, grants, unmet requirements, budget, and effect policy.
