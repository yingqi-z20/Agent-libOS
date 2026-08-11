# Electron GUI

Agent libOS includes a local desktop management console for supervising
Durable Task Runs, processes, messages, human approvals, AgentImage selection/registration/commit,
checkpoints, capabilities, Skills, JSON-RPC endpoints, MCP servers, audit
records, persisted LLM calls, semantic assessment/FlowGraph/policy evidence, and human Agent
ratings.

The GUI is a local-only Electron app. Electron starts
`agent-libos-gui-server`, receives a bearer token that is random by default,
and connects to `127.0.0.1` through HTTP and Server-Sent Events. The renderer
never receives Node.js access, but the preload-exposed `libosApi` object gives
it the connection information, including that bearer token, needed for direct
localhost HTTP and SSE calls. Renderer content and dependency integrity are
therefore part of the local GUI trusted computing base.

During development the Electron main process starts the backend without a
shell. It first honors `AGENT_LIBOS_GUI_SERVER_BIN`, then tries the project
`.venv` entrypoint, and only falls back to `uv run agent-libos-gui-server` if no
local entrypoint exists.

## Architecture

```text
Electron main process
  -> starts Python agent-libos-gui-server
  -> receives the GUI bearer token
  -> exposes limited preload IPC, including the authenticated connection

React renderer
  -> receives the bearer token through preload and calls localhost HTTP APIs
  -> subscribes to /api/events/stream
  -> renders process, message, approval, audit, LLM, and semantic evidence

Python GUI server
  -> owns Runtime.open(db)
  -> routes all operations through existing runtime managers
  -> never grants capability by GUI visibility
```

Semantic history is deliberately not embedded in `/api/snapshot`; the renderer
loads bounded keyset pages only when the Semantic tab is opened. This keeps the
existing snapshot bound independent of assessment-history size.

SSE sequence ids are scoped to one GUI-server process and the replay buffer is
bounded. A reconnecting client sends its last id as `cursor`. If that cursor is
older than the retained window, or is ahead of the newest id after a server
restart, the server emits `event.invalidated` with
`reason: sse_cursor_not_replayable`, resets the stream cursor, and then replays
the retained events. Clients must fetch `GET /api/snapshot` when they receive
that invalidation; the bundled renderer does so. This makes a replay gap
explicit instead of silently leaving the UI on stale state.

SSE frames are also bounded by `gui.sse_payload_max_bytes`. If a bounded
`snapshot` frame is still too large, the server replaces it with
`snapshot_truncated`; if any other frame is too large, it replaces it with
`event.invalidated`. Both replacements carry `invalidated: true`, identify the
original event, and require a fresh `GET /api/snapshot` rather than applying the
replacement as a delta. The bundled renderer refreshes on either event. Cursor
invalidation additionally carries `reason: sse_cursor_not_replayable` and the
available/reset cursor metadata.

`task_run.updated` carries only the redacted `TaskRunSummary`, never a goal,
follow-up, transcript, credential, or resume payload. Its `revision` is
monotonic for that Run. The renderer ignores a revision older than or equal to
the one already rendered and fetches the HTTP snapshot when the stream reports
invalidation instead of treating that marker as a delta.

Append-event de-duplication is bounded by the GUI event-buffer configuration.
Append-only events and audit records use their durable ids. Message and LLM-call
notifications also key de-duplication by durable row id, but that identity key
does not make every backing message or LLM row immutable; lifecycle updates and
payload-retention reductions remain governed by the storage contract. Human
requests use the request id together with `updated_at` and `status`, so a pending
request and its later approved/rejected/cancelled version each produce a
`human_request.updated` event without growing an unbounded in-process set.

The GUI server is not a separate process-effect boundary. It is an
authenticated local Host/admin authority boundary over the same primitives,
Capability checks, human approval flow, events, and audit records used by the
CLI. Possession of its bearer token authorizes the documented Host/admin
routes; actor-mode routes instead use the selected process's authority. Its
Python entrypoint lives under
`agent_libos.api.gui` with the CLI because both are host-facing API surfaces.
Only a bearer token holder on the same machine can use it; CORS is limited to
loopback HTTP(S) browser origins plus the exact production-renderer origin
`agent-libos://app`; it does not accept `Origin: null` or other custom-scheme
hosts. The Electron production-build path serves `gui/dist` through that
privileged, secure custom protocol instead of `file://`, giving browser
requests a stable origin without broadening the server allowlist. The protocol
resolver rejects other authorities, credentials, ports, traversal, missing or
non-file targets, and symbolic links whose canonical target is outside the
distribution root. Assets are read from a verified file descriptor so a
pathname swap after validation cannot redirect the response.

The Electron window applies the same exact-origin guard to renderer-initiated
navigations and server-side redirects. Same-origin Vite development routes
remain usable, while cross-origin, protocol-relative, credential-confused, and
malformed targets are prevented from replacing the trusted renderer. External
links use the separately protocol-filtered preload bridge or denied-window
handler instead of top-level navigation. Both child-server startup output
streams have the same 64 KiB cumulative limit; exceeding either fails startup
and runs the process-tree cleanup path without retaining the oversized output.

The GUI can display data-flow Audit/Event/Explain evidence produced by runtime
operations, but it does not expose a Sink-trust mutation route or model tool.
Sink trust remains Host configuration/API state; adding a GUI view later must
not turn renderer visibility into `data_flow_sink_registry:*` authority.

Agent-authored Markdown is treated as untrusted presentation data. Markdown
image syntax is always rendered as an escaped, accessible text placeholder;
the renderer never creates an `img`, `source`, or preload request for its
destination, whether that destination is remote, local, relative, `data:`, or
`blob:`. Production Content Security Policy remains a second boundary rather
than the control responsible for preventing Markdown-triggered network egress.

Only the checkpoint create/restore/fork, Skill activation/unload, Capability
grant/delegate/revoke, image registration/commit, JSON-RPC registration, MCP
registration, and MCP protocol-discovery endpoints accept an optional `actor`.
On those endpoints, omitting `actor` runs in GUI Host/admin mode; supplying a
non-empty process id opts into process-authority mode and requires that process
to hold the capability needed by the underlying primitive. Skill registration
is deliberately stricter: `POST /api/skills/register` requires a non-empty
process `actor` and reads the package path through that process's workspace
filesystem authority; there is no GUI-admin registration mode. Other mutation
endpoints use Host/admin authority or an explicit `pid` field and reject a
top-level `actor` instead of treating unused attribution as an authorization
boundary.

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

The server options are:

| Option | Behavior |
| --- | --- |
| `--config <path>` | Load a YAML overlay; otherwise use the project-root `config.yaml` when present. |
| `--db <target>` | Select the SQLite path, `local`, or configured PostgreSQL target; omission uses the runtime configuration default. |
| `--port <n>` | Bind the fixed loopback host on this port; `0` asks the OS for an available port and is the default. |
| `--token <value>` | Use an explicit bearer token; omission generates a random token. Treat an explicit value and the startup JSON as Host credentials. |
| `--llm-profiles-file <path>` | Select the user GUI LLM-profile JSON file; omission uses the documented user-level default. |
| `--no-auto-run` | Start with automatic scheduler runs disabled and the scheduler paused. |
| `--max-quanta <n>` | Set a positive default scheduler quantum budget; omission uses `runtime.run_until_idle_max_quanta`. |

There is no public-bind option: the server rejects non-loopback hosts.

The GUI server accepts the same runtime store targets as the CLI. SQLite paths
are the default local store; PostgreSQL DSNs require installing the `postgres`
extra and are redacted in startup and health payloads. Persistent stores use an
active-runtime lease, so a GUI server and a writable CLI Runtime cannot open
the same SQLite target or PostgreSQL database/schema concurrently. On the
hardened POSIX path, SQLite pairs its canonical target and no-follow sidecar
`flock` with an owner-only `(st_dev, st_ino)` identity lease; the database and
sidecars must be regular, current-user-owned, single-link files, rejecting
ordinary hard-link aliases and path/lockfile replacement races. Every platform
also holds an exclusive lock on the actual database connection. That preserves
one writer without POSIX identity guarantees and during a same-UID connect-time
retarget, but such filesystem administration remains inside the Host trust
boundary; do not rename or replace a live database path.
PostgreSQL uses a stable database/schema advisory key. See
[Runtime Storage](storage.md#active-runtime-leases).

The server prints one JSON line containing the selected local URL and bearer
token:

```json
{"url":"http://127.0.0.1:51234","token":"...","db":".agent_libos.sqlite"}
```

Run the Electron app:

```bash
npm --prefix gui run electron:dev
```

Unlike the general `LLMClient.from_env()` and CLI configuration path, the
Electron launcher intentionally reads `<repository-root>/.env` before it starts
`agent-libos-gui-server`. Values already present in the Electron process
environment take precedence; `.env` only fills missing keys. This is a GUI
launcher compatibility behavior, not implicit `.env` loading by the CLI or
library API.

Build and type-check the GUI:

```bash
npm --prefix gui run test
npm --prefix gui run typecheck
npm --prefix gui run build
uv run python scripts/test_matrix.py --lane gui
```

The browser end-to-end suite also needs the version-matched Chromium binary:

```bash
npm --prefix gui exec -- playwright install chromium
npm --prefix gui run test:e2e
```

The E2E runner creates a temporary SQLite store, starts the real loopback GUI
HTTP/SSE service and Vite bridge with an ephemeral bearer token, and blocks
renderer requests to non-loopback hosts. Provider content is deterministic test
data; real Provider credentials are not part of this suite.

`npm --prefix gui run build` removes `dist-electron` before compiling. Vitest
excludes both renderer and Electron build output, and the production Electron
TypeScript configuration excludes `*.test.ts`; generated JavaScript therefore
cannot become a second copy of the source test suite.

The GUI uses React and React DOM 19.2.8 with Lucide React 1.28.0. Its
development toolchain uses Electron 43.2, Vite 8 with plugin-react 6, Vitest 4,
TypeScript 7, and jsdom 30. Vite owns the Rolldown and PostCSS dependency graph;
no direct esbuild or PostCSS override is required. The supported runtime is
reflected in the package engine contract: Node `^24.15.0 || >=26.0.0` and npm
11 or newer. Use the Node 24 LTS line for parity with CI.

A useful clean-build check is:

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
against an in-memory `local` store without creating a BrowserWindow. Smoke
logging redacts the temporary bearer token. Set
`AGENT_LIBOS_GUI_SMOKE_WINDOW=1` when a machine has a working desktop/GPU stack
and you specifically want to exercise the production Vite build through the
custom-protocol BrowserWindow, its API origin, and the preload bridge. This is
a production-build/custom-protocol smoke, not an installer, packaged-app,
code-signing, or notarization test; Electron packaging is not configured here.

The Vite development server is bound to `127.0.0.1` and restricts file serving
to the `gui/` directory. Production dependency audit should remain clean; any
dev-server advisory must be handled with local-only exposure unless an upstream
fix is available.

For renderer-only browser development against a real local GUI server, set all
of `VITE_AGENT_LIBOS_GUI_URL`, `VITE_AGENT_LIBOS_GUI_TOKEN`, and optionally
`VITE_AGENT_LIBOS_GUI_DB` before starting Vite. This bridge is compiled only in
Vite development mode, accepts only an `http://` loopback URL, and is absent
from production builds. Do not reuse or publish the temporary bearer token.
Production Electron always requires the preload-provided connection.

## Current Workspace

On first launch the default screen is the streamlined user page; later launches
restore the last user/operator view when renderer storage is available. The
default user workspace uses a task sidebar and a dedicated conversation area.
The first-task state starts with an empty goal instead of silently submitting a
sample task, offers reusable starter prompts and a Ctrl/Command+Enter shortcut,
and keeps a compact read-only launch summary beside an **Edit settings** button.
The **New task settings** dialog edits the image, model profile, quanta,
working directory, workspace, Git, command, and context-maintenance choices as
one local draft. A successful launch
clears the goal draft and uses a bounded one-line goal summary as the task label
for the current GUI window. At most 200 bounded labels are retained in
`sessionStorage`, so reloads remain legible without persisting full goals across
window closes. The selected pid is retained in the same scope; if it is stale,
the GUI prefers actionable work over terminal history. Older tasks fall back to
a compact pid. After
launch, the sidebar shows localized process state and wait reason, LLM-call and
token metrics, process-scoped run/pause/stop controls, and collapsed image/rating
tools. On narrow screens it becomes an explicit task-panel toggle instead of
forcing the whole control stack above the conversation. The conversation area
prioritizes pending Human requests, Agent output, terminal outcome, and a
multiline composer. Terminal processes disable run, stop, message, and interrupt
actions and replace the disabled composer with a clear Start another task action.
Message drafts are retained per process so changing
the selection cannot send a draft to the wrong process, Enter sends, and
Shift+Enter inserts a line break.

Task-start controls include a workspace access scope, a separate local-Git
request switch, and a command-execution selector that defaults to disabled.
Persistent context enrichment and bounded automatic maintenance are disabled by
default and can be explicitly enabled in the dialog's permission group.
These controls create an
explicit Host-authored launch manifest: Human communication is granted to the
root process. Explicit persistent context receives non-delegable
`context:enrichment/execute` and `context:maintenance/execute` markers plus a `process:spawn`
capability whose Authority Rule matches only `process.spawn_child` with
`image_id=context-compressor:v0`, plus read authority for that exact built-in
image. It cannot fork, launch another coding agent, or boot any other image.
Filesystem and local-Git entries are only ceilings for
later permission requests. They grant no workspace or Git access by
themselves. Selecting reviewed commands adds only a constrained
`shell_policy_level=allowlist_auto_else_ask` Host capability: configured
allowlist diagnostics may run automatically, while every other exact command
still creates a per-use Human approval request. It never grants the
`always_allow` policy. With command execution disabled, arbitrary Shell
commands have no Host authority and fail before prompting. A
nested initial cwd narrows the filesystem ceiling to that subtree. The
normalized ceilings are shown to the model as permission-planning facts, clearly
separate from active capabilities, so an agent can request a coherent task scope
before attempting a filesystem effect. Runtime enforcement remains authoritative
if the model ignores that guidance or asks outside the ceiling. The default edit
mode omits delete, and the manage mode adds it. Image import
and commit both require explicit
confirmation. Saving as an image creates a checkpoint only after that
confirmation, then commits the checkpoint into an immutable image artifact.
The conversation follows new output only while the reader remains near the
bottom; scrolling upward pauses automatic following and exposes a Jump to latest
control when new output arrives. The log itself is keyboard-focusable.
Interactive built-in images are instructed to send one concise final
`human_output` before `process_exit`. The conversation also renders a terminal
status fallback from the process outcome, including only a result/reason Object
Memory reference when present. It does not materialize that object's payload;
normal Object Memory capability checks still apply. When an output was delivered
through another configured Human channel but its content is withheld from the
GUI presentation Sink by data-flow policy, the conversation shows an explicit
protected-output notice instead of an empty message.
Successfully delivered output messages are private-digest-bound frozen
snapshots for GUI presentation. The renderer still receives them only after the
captured labels pass the current GUI Sink policy, but a later LLM-context or
source-object version change no longer turns the already fixed message into a
protected-output false positive. Digest mismatch and uncertain delivery remain
withheld.

Durable Task Runs add a server-persisted task identity above the process tree.
When a Run is selected, the user page keys state by `selectedRunId`, displays
the persisted title, requirements, blockers, pending Human work, result
retention, and only the controls returned in `allowed_actions`. A
`needs_attention` Run with an unknown external effect never shows an ordinary
Retry or Resume action. The server-derived `payloads_purged` summary flag, not
the configured retention policy alone, controls whether rerun requires a
replacement goal; this also covers a `permanent` Run that a Host explicitly
purged. Process groups remain selectable, and the operator
console keeps their process grouping. Its **Task Runs** tab pages through the
selected Run ledger and resolves linked Operation/evidence rows through
**Explain**. The former generic Tasks panel is labeled **Object tasks** so its
runtime-local single-tool contract cannot be confused with Durable Task Runs.
After default terminal cleanup or an explicit Host purge, linked Human-history
cards may contain only request identity, type, status, timestamps, hashes, and
audit linkage; the GUI does not reconstruct or imply retention of readable
prompt/answer/decision content.

The operator console is a three-column process workspace with a grouped runtime
toolbar and progressively disclosed configuration:

- the left pane contains a collapsible launch configuration plus a searchable,
  keyboard-navigable process tree with session task label, compact pid, image,
  status, and unread-message
  indicators; the toolbar's New process action opens this configuration rather than
  immediately creating a process;
- the center pane contains the selected process timeline, Human request cards,
  localized state, and multiline message controls; the timeline defaults to
  Human-facing messages, requests, and LLM activity while Events, Audit, and the
  unfiltered evidence stream remain one click away;
- the right pane contains a collapsed process-control section with cwd,
  pause/resume, exec, and a visually isolated confirmed exit action, followed
  by tabs for a
  structured process/resource overview, ratings, capability administration,
  Skill lifecycle and workspace registration, checkpoint create/inspect/diff/
  fork/restore, Object Tasks, audit, Explain, LLM calls, Images, JSON-RPC, MCP,
  loaded module inspection, and an Object Memory reference showing the goal
  OID and capability-control note;
- the top bar groups database identity, spawn, auto-run, quanta, run, step,
  pause, refresh, scheduler/stream status, language, and the user-view switch.

The renderer reports initialization failure with an explicit retry surface,
shows refresh and live-stream connection state, preserves the last valid
snapshot while reconnecting, surfaces scheduler failures, and warns when the
bounded snapshot omitted any collection or value. That warning is a single-line,
dismissible summary which states that source data remains stored and reappears
only when the omitted-section set changes. Same-build snapshot responses
from both HTTP refreshes and SSE events are minimally shape-validated before
React consumes them. Destructive dialogs
trap keyboard focus, close with Escape when idle, restore the previous focus,
and expose linked ARIA title/description state. Detail tabs support arrow,
Home, and End navigation. At narrower widths the operator pane moves from three
columns to two rows and then to a single scrollable column; the user page,
forms, dialogs, and image rows also collapse without requiring a wide desktop
viewport. The production Electron window can be resized down to 360px so those
layouts are reachable. Reduced-motion preferences disable continuous animation.

Detailed cwd and resource budget/usage values are shown in process metadata and
detail views rather than as fields on every process-tree row. The Object Memory
tab does not materialize arbitrary objects; payload materialization remains a
runtime capability-checked operation.

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
file under `${XDG_CONFIG_HOME:-~/.config}` on Linux. The file stores model
routing fields such as profile id, model, base URL, API mode, tuning options,
optional `context_window_tokens`, `max_input_tokens_per_call`, and
`max_total_tokens_per_call`, prompt-cache/Responses-continuation policy
settings, and the
`api_key_env`/`safety_identifier_env` names. It never stores either environment
variable's value. The Python profile API validates, preserves, and returns the
core-only per-call budget fields, but this release adds no bundled GUI controls
for them. The editor continues to expose routing, reasoning effort, verbosity,
cache retention, and explicit Responses continuation policy. The current full-snapshot AgentProcess
executor records that policy but does not send `previous_response_id`; enabling
provider storage can still increase provider-side retention. Profile deletion
requires an inline second confirmation. Numeric fields reject fractional
integers, non-finite values, and values outside their displayed minimums instead
of silently coercing them. A
`PUT` is a partial update: profile fields not exposed or omitted by the current
renderer are preserved rather than silently reset. When a profile has a base URL,
`allow_custom_base_url: false` is preserved explicitly rather than inferred
away, so disabling custom-base-url use remains stable across GUI restarts.

The scheduler defaults to automatic mode. Users can pause auto-run, step a
selected process, or run the selected process with an optional quantum budget.
For a paused process, Run clears the pause fence and starts a run scoped to that
process without changing the global auto-run setting. Human-request
responses use their own per-request pending guard, so a concurrently running or
message-triggered task cannot discard an approval or rejection. Passive status
notices do not intercept controls beneath them, and long snapshot-truncation
details are bounded and dismissible.
Leaving the budget blank omits `max_quanta`, so the server uses
`runtime.run_until_idle_max_quanta`; the run is unbounded only when that
configured default is `null`. Entering a number explicitly bounds that run.
Automatic runs after spawn/message/exec may advance all
runnable processes, but `POST /api/processes/{pid}/run` is intentionally scoped
to that pid. `POST /api/processes/{pid}/step` is synchronous: its response and
the snapshot it publishes contain the final scheduler state (`running: false`)
after the quantum has completed. The background controller uses one completed
quantum as its batching boundary. Once a provider/tool quantum has been
admitted it is allowed to finish even when it takes longer than the core
scheduler drain window; that exception is local to GUI batching, while ordinary
bounded Runtime calls retain their cancellation boundary. Real LLM calls are
still persisted in `llm_calls`, so the GUI can show token usage, errors, and a
bounded Provider trace. Snapshot and SSE projections carry only content-free
call summaries. The selected process's full-retention input, output,
Provider-returned reasoning, tool actions, and sanitized raw response are
fetched through authenticated, size-bounded detail/content routes only when
the corresponding panel is opened. This default supports self-evolution
training and fine-tuning pipelines under the deployment's user agreement. If
the host runtime is configured with `llm.persist_full_io=False`, the same
detail API returns unavailable states, content-free hashes, counts, and structural envelopes
for sensitive prompt, tool, reasoning, and provider payload fields. Processes
using `image_only` cannot run under that setting because their lossless durable
transcript is unavailable; they fail before provider dispatch.

GUI background auto-run deliberately sets `process_human_queue=false`. It may
advance runnable model work, but it never auto-approves, auto-denies, or invents
an answer for a pending Human request. An ordinary/uncertain request remains for
a Human request card or another explicit Host terminal surface. Independently,
the Host semantic worker may win the shared CAS for a closed hard denial or an
eligible canary one-use grant; the GUI scheduler itself never initiates either.
Normal runtime wakeup/resume semantics follow the single terminal winner.

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
does not perturb that view. For pending questions and approvals, a source,
Sink-trust registry, Task Authority manifest, public-view, or release-binding
change invalidates the durable visible marker; the parent is redacted again and
requires a fresh exact release before projection (including a newly recorded
decision) or response. A successfully delivered output revalidates its frozen
message digest, labels, Sink policy, manifest, and public view without requiring
the original mutable sources to remain at the delivery version. The freshness
guard and Human decision commit share one store transaction, so concurrent Host
registry or source mutations cannot land between them. Bounded snapshots project only the
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
Durable Task Runs, pending-first Human requests, tools, images, Skills, JSON-RPC
endpoints, MCP servers, Runtime Modules, and LLM profiles fetch at most
`snapshot_collection_max_items + 1`, subject to any stricter subsystem list
maximum. Skills, JSON-RPC, and MCP list APIs perform one additional internal
lookahead even when that subsystem maximum is stricter than the GUI maximum.
Either kind of lookahead becomes a `source_limited` lower-bound entry in
`_truncated` and is not serialized.
Event and audit rows persist a derived `gui_snapshot_visible` flag. Snapshot
queries filter that indexed flag before applying `LIMIT`, preventing internal
GUI-presentation evidence from displacing causal runtime rows. The flag is
required by store schema v6; missing or malformed persisted visibility state is
rejected rather than repaired during open. Bounded event/audit page endpoints
can still include presentation evidence when requested.
The process window orders non-terminal processes before the most recently
updated terminal history, so a full snapshot does not hide current work behind
old completed rows. If the bounded window contains a child but not its parent,
the process tree renders that child as a temporary root rather than making it
unreachable.
Process message/count, bounded recent LLM-window activity, rating, ancestor
reservation, and hierarchical resource-counter/budget data are loaded through
batch queries. The user-facing `llm_call_count` and `token_total` take the
maximum of the durable hierarchical resource counters and the bounded
recent-window values. This prevents a long-running process from being
underreported at the window limit while still covering persisted or manually
inserted LLM-call rows that are not represented in resource counters. Message
and LLM windows select the newest configured rows per process; messages are
returned chronologically. Snapshot construction therefore does not issue one
message, LLM-call, rating, or resource query for every listed process.

## Semantic Panel

The read-only Semantic tab is available at Host scope and process scope; the
latter adds the selected `pid` filter. It fetches status, assessment, FlowGraph,
settlement, policy/control history, health, and metric pages on demand rather
than extending the Runtime snapshot. Bounded “load more” follows opaque keyset
cursors and de-duplicates by the relevant stable record id.

The panel shows queue health, aggregate success/error/OOD and Shadow outcome
counts, domain/status history filters, reason codes, normalized observed Human
outcome, action, calibration, reserved nullable input/output token and cost
fields, latency, classifier identity, matched rule/predicate codes, and input/policy/
classifier-artifact/manifest/action/resource/arguments/state/source/Sink/tool/
provider/projection/tenant digests.
It does not render a prompt, goal/provider content, raw response, job
projection, model explanation, or hidden reasoning. Structured strings are
rendered as inert text. The same-build TypeScript decoder requires status
schema v3 with complete exact-key `by_status` and `by_domain` maps, consistent
aggregate totals, control/FlowGraph/machine sections, and denominator-aware
real-auto-approval and review rates. Assessment
page/detail envelopes and rows remain schema v1. Their decoders require exact
keys and enum/digest/confidence/span shapes and reject unknown or private fields
before React receives them.

The panel shows validated external classifier token/cost counters when present
and shows deterministic/scripted, missing, conflicting, or untrusted counters
as unavailable. The API projection contains only the three nullable canonical
integers—never token aliases, unknown provider usage keys, or the raw usage
object. Each non-null value is a non-negative JavaScript-safe integer through
`2^53 - 1`. Values are provider-reported telemetry rather than authoritative
billing or classifier-quality metrics, and the status response does not
aggregate them.

The UI displays a zero-denominator or unreviewed rate as not applicable, not as
a misleading 0% unsafe rate. All controls are filters or pagination; the panel
contains no semantic policy/control, settlement, or review-import mutation.
See [Semantic Approval and Data
Identification](semantic_shadow.md#inspection-surfaces).

## High-Risk Operations

The Python GUI server requires `confirmed: true` for the following high-risk
operations before invoking the runtime:

- process `exec` and `exit`,
- process `signal` requests that cancel or terminate a process,
- workflow runs for side-effecting or unresolved tools, or requests with an
  explicit image or working directory,
- image package registration and checkpoint-to-image commit,
- checkpoint restore and fork,
- capability grant, delegate, and revoke,
- JSON-RPC endpoint registration and method calls,
- MCP server registration and tool calls,
- Skill registration, activation, and unload.
- Durable Task Run cancellation and evidence-constrained recovery.

The bundled renderer provides confirmation dialogs for process exec and exit,
image package registration and checkpoint-to-image commit, checkpoint restore
and fork, capability grant/delegate/revoke, JSON-RPC registration/calls, MCP
registration/calls, and Skill registration/activation/unload. Capability
mutation previews explicitly identify Host-admin mode. Skill and remote
registration can instead use the selected process authority; workspace Skill
registration always requires it. Process cancel/terminate signals and generic
workflow execution remain server/API-only and require a same-build client to
present its own confirmation UX. A client calling `POST /api/workflows/run`
must submit `confirmed: true` when the server classifies the request as
high-risk; the bundled `LibOSClient` has no generic workflow convenience
method. The server rejects a missing or false confirmation before invoking any
high-risk runtime operation, regardless of renderer state.

The Object Tasks panel also presents review dialogs before start and cancel,
but those dialogs are renderer UX rather than the server's `confirmed: true`
boundary. The Object Task endpoints do not consume a `confirmed` field; process,
Object, Tool, Capability, and Human-approval checks in the runtime remain their
authorization boundary. Clients must not infer server confirmation from the
presence of that dialog.

JSON-RPC endpoint and MCP server registration through the GUI accept manifest
text only. The renderer cannot ask the Python GUI server to read an arbitrary
host file path; file/path based registration remains a CLI/admin workflow.

Image package registration follows the same rule. Electron may read a package
directory selected by the user and pass bounded package file payloads to the
local GUI server, but the preload result omits the selected host path and the
server rejects host file paths. The Electron selector rejects a symbolic root,
symbolic or multiply-linked files, non-file/non-directory entries, `.git`
segments, more than 512 files, more than 512 directories, more than 1,024 total
directory entries, more than 524,288 bytes of entry names, directory depth above
32, and more than 16 MiB of raw file data. It separately caps the selected
manifest at 1 MiB.

Those directory-walk checks belong to Electron and do not apply to an API
client that already has a file map. The Python server accepts only a non-empty
`files` object whose values are text or `{ "base64": "..." }`; malformed base64
is a `400`. The Runtime then normalizes each supplied relative path, rejects
absolute/traversing/duplicate paths and `.git` segments, requires `IMAGE.yaml`,
and applies the file-map limits. Their defaults are a 1 MiB manifest hard limit,
1 MiB per file, 16 MiB total, and 512 files. The softer 256 KiB
`package_manifest_max_bytes` limit applies when the Runtime itself reads a Host
or process-workspace package; the preloaded GUI file-map path uses
`package_manifest_hard_limit_bytes`. The default GUI request-body limit is sized
to carry the raw package limit after base64 and JSON wrapping. Registering or
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
machine-readable [GUI API contract subset v2](gui_api_schema.json)
deliberately covers the snapshot's required top-level collections, the minimal
process/scheduler shape consumed during bootstrap, redacted Task Run
summary/detail state, the JSON error envelope, and payloads for every operation
that the server gates with explicit confirmation.
`GET /api/snapshot` emits `schema_version: 3`; this same-build renderer rejects
older snapshot shapes instead of treating summary-only LLM state or Task Run
state as an empty collection. Version 3 replaces full LLM rows in snapshot/SSE
with strict `LlmCallSummary` projections.
Other snapshot collection item schemas and renderer routes remain same-build
implementation details. It is JSON Schema, not a complete OpenAPI document.

The schema file is a registry, not a single instance schema: its document root
deliberately rejects every instance. A validator must select a named entry point
with a fragment such as
`./gui_api_schema.json#/$defs/processExecPayload` or
`./gui_api_schema.json#/$defs/snapshotResponse`. In particular,
`workflowRunPayload` can validate field types but cannot determine whether a
resolved Tool has side effects; `confirmed` is therefore an optional boolean in
that definition. The server still requires it to equal `true` whenever runtime
risk classification says the workflow is high risk.

The process-wait schema's `stale_execution` branch is a diagnostic projection
of `StaleExecutionProcessWait`. It contains canonical identity hashes and
generation values, not prior raw owner/lease tokens or TaskRun epoch,
safe-point, or live-binding evidence. The renderer may use its typed `kind` to
present a conservatively paused process, but it must not infer that a Task Run
is resumable or synthesize a recovery action from that receipt. Run controls
continue to follow the server's `allowed_actions` and recovery options.
`status_message`, including `stale_execution_recovery`, is presentation-only
compatibility text and is never a client control protocol.

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
- `GET /api/semantic/status`, assessment/detail, FlowGraph status/entity/edge/
  lineage, settlements, policy epochs, control/history, health, and metrics
  routes expose payload-free evidence. The assessment list accepts only `pid`,
  `request_id`, `operation_id`, `kind`,
  `status`, `domain`, `action_id`, `tenant_bucket_sha256`, `after`, and `limit`;
  the action is a dotted lower-case ontology id and the tenant bucket is an
  exact lower-case SHA-256 digest. The default is 50 and the hard HTTP maximum
  is 100. There are no semantic write routes.
- `POST /api/workflows/run`
- `GET /api/task-runs`, `POST /api/task-runs`, and
  `GET /api/task-runs/{run_id}` for Durable Task Run collection/detail state.
  Collection POST only creates queued state and requires a stable
  `client_request_id`; it rejects create-time `auto_run`/`max_quanta` so bounded
  execution always goes through the revision-fenced `/run` mutation.
  Detail includes the redacted summary, server-generated recovery options, and
  a bounded requirements page; `requirements_limit` and opaque
  `requirements_cursor` page that embedded collection.
- `GET /api/task-runs/{run_id}/ledger` and
  `GET /api/task-runs/{run_id}/human-requests` for bounded Run-scoped pages.
  Collection, ledger, and Human pages accept opaque `cursor` values and return
  `next_cursor`; clients must not parse or synthesize them. The embedded
  requirements page also returns `next_cursor`. Requirement changes are linked
  ledger items, and 1.4.2 has no independent Task Run requirements or wait HTTP
  route.
- `POST /api/task-runs/{run_id}/run|pause|resume|cancel|follow-ups|recover|rerun`.
  Every existing-Run mutation carries a command id and expected revision.
  Rerun additionally carries a stable `client_request_id` for creation of the
  linked Run; the bundled client derives it deterministically from the stable
  rerun command id when the caller does not supply one.
  Cancel/recover require explicit confirmation, and revision/idempotency
  conflicts return `409` rather than applying to newer state.
  If an HTTP mutation response is lost or otherwise ambiguous, the client must
  retry the identical request body with its original command id and expected
  revision; a new command id is not a transport retry. A matching provisional
  receipt permits only local settlement and stored-result completion, never a
  second scheduler quantum, LLM, tool, Provider, or external-effect dispatch.
  A linked recovery with a missing outer receipt is the other local-only case:
  exact retry validates its request-bound nested rerun, target create receipt,
  and causal link and returns the same target rather than creating another Run.
  The client first reconciles an authoritative detail snapshot, but that read
  does not authorize it to synthesize a new command id. In the two-stage
  create-then-run flow, it retains both the create `client_request_id` and the
  separate run command id and original Run revision until the run mutation has
  a non-ambiguous result; successful creation alone does not retire the run
  intent. Follow-up intent is cleared only by the HTTP 200 for that exact
  command; matching body content, a newly visible requirement, or SSE is not
  command-specific admission evidence.
  The Runtime/SDK exact replay remains an immutable historical command receipt.
  After every successful private-HTTP mutation, however, the server performs
  an exact `manager.get(result_run_id)` and returns and publishes the latest
  summary observed by that read. For linked rerun/recovery, `result_run_id` is
  the target/new Run rather than the source; a later concurrent mutation may
  still advance the revision.
  A Task Run `409` includes redacted `command_admitted` only when Store evidence
  proves it and includes a current summary when available; the client still
  performs an exact detail GET. Only `task_run_revision_conflict` together
  with `command_admitted=false` and successful authoritative reconciliation
  lets the client retire the old intent so a later user action can receive a
  new command id. Command conflict, admitted, and indeterminate cases preserve
  the exact old intent and fail closed; no conflict path automatically retries,
  rebases, or invents a new id.
- `GET /api/human-requests/{request_id}` for exact Human-request reconciliation
  when a bounded snapshot no longer contains that request.
- `POST /api/scheduler/auto`, `POST /api/scheduler/pause`
- `GET /api/processes/{pid}`
- `POST /api/processes/{pid}/run|step|pause|resume|signal|message|interrupt|cd|exec|exit`
- `GET /api/processes/{pid}/messages|human-requests|llm-calls|audit|events|capabilities|checkpoints`.
  The LLM-call route returns a newest-first keyset page of content-free
  summaries and an opaque cursor. `GET
  /api/processes/{pid}/llm-calls/{call_id}` returns bounded attempt metadata and
  content descriptors; its `/content` child route accepts only the documented
  field allowlist and returns at most 64 KiB per chunk. Content cursors bind the
  process, call, field, attempt, retention tier, and content hash; retention or
  content changes invalidate them with `409 content_changed`.
  The events route accepts `limit=<n>` (default and maximum
  `gui.snapshot_event_limit`) and `before=<event_id>` for older pages. It selects
  the bounded newest/cursor window in storage and returns that page in
  chronological order; it is not an unbounded full-log endpoint.
  The audit route likewise accepts `limit=<n>` (default and maximum
  `gui.snapshot_audit_limit`) and `before=<record_id>`. To page backward without
  gaps, use the oldest `record_id` in the current page as the next `before`
  cursor; cursors are opaque and excluded from the returned page.
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
  In `enforce_deny` and `canary_auto`, an external-operation response must also
  carry the exact top-level `expected_revision` and Host-rendered
  `preview_sha256` returned with the pending request. Missing, stale, malformed,
  or mismatched preview evidence is rejected before terminalization; neither
  field can authorize or weaken a hard-deny preflight.
- `GET /api/checkpoints`, `POST /api/checkpoints/create`,
  `GET /api/checkpoints/{checkpoint_id}`,
  `GET /api/checkpoints/{checkpoint_id}/diff`, and
  `POST /api/checkpoints/{checkpoint_id}/restore|fork`
- `GET /api/skills`, `GET /api/skills/{skill_id}`,
  `POST /api/skills/register`, and
  `POST /api/skills/{skill_id}/activate|unload`
- `GET /api/capabilities`, `GET /api/capabilities/{capability_id}`,
  `POST /api/capabilities/grant|delegate|explain`, and
  `POST /api/capabilities/{capability_id}/revoke`. Capability inventory clients
  should request `GET /api/capabilities?mode=page`, optionally with `subject`,
  `limit`, and the opaque `after` cursor returned by the preceding page. The
  response is `{items, next_after, has_more}`; callers must replay
  `next_after` as `after` without parsing it until `has_more` is false. `limit`
  defaults to and cannot exceed `capability.list_limit`. Omitting `mode=page`
  preserves the legacy array response for same-build older clients, but that
  response is only one bounded page and must not be treated as a complete
  inventory when it reaches the configured limit. The checked-in renderer uses
  page mode and also fails closed on missing/repeated cursors, more than 10,000
  received items, or an inventory that does not terminate within 10,000 pages.
- `GET /api/images`, `GET /api/images/{image_id}`, and
  `POST /api/images/register|commit`
- `GET /api/jsonrpc`, `GET /api/jsonrpc/{endpoint_id}`,
  `POST /api/jsonrpc/register`, and
  `POST /api/jsonrpc/{endpoint_id}/call`
- `GET /api/mcp`, `GET /api/mcp/{server_id}`,
  `GET /api/mcp/{server_id}/tools`, `POST /api/mcp/register`,
  `POST /api/mcp/{server_id}/discover`, and
  `POST /api/mcp/{server_id}/call`
- `GET /api/modules`, `GET /api/modules/{module_id}`

All non-`OPTIONS` endpoints require `Authorization: Bearer <session-token>`.
Unauthenticated `OPTIONS` is limited to CORS preflight and does not read or
mutate Runtime state.
Mutation endpoints validate required ids, image names, and paths as non-empty
JSON strings. Malformed enum values such as an unknown process signal, missing
required fields, non-object request bodies, and incorrectly typed booleans
return `400` without invoking the runtime mutation.
For process spawn and workflow run, optional `image` and `working_directory`
fields must be omitted or supplied as non-empty JSON strings; explicit `null`
is not treated as omission. Process exec defaults an omitted `args` to `{}` but
rejects any supplied non-object value. Process exit accepts `message` only as a
JSON string or `null`. These type failures also return `400` before the runtime
mutation or workflow launch.

The MCP panel exposes an explicit Discover action for Manifest v2 `auto` and
`2026-07-28` servers. It displays Manifest version/configured mode plus the
current operation's revision/era, legacy/sessionless/fallback state, bounded
server identity, standard capabilities, and unsupported capabilities. A page
reload returns to “not negotiated”; discovery/session state is neither placed
in the GUI snapshot schema nor persisted by the Runtime. Discovery is a
protected external read, not a high-risk mutation confirmation. The bundled
Discover action omits `actor` and therefore uses GUI Host/admin authority. An
API client may instead supply a non-empty process `actor`; that opts into the
process-authority path and requires the process capability for the discovery
read, without adding or bypassing a `confirmed: true` boundary. The panel does
not expose OAuth login, MRTR continuation, subscriptions, Resources, Prompts,
Tasks, or other excluded MCP surfaces.

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
the server does not synthesize authority from image requirements. The bundled
renderer submits an explicit manifest built only from its visible task-access
controls. It never copies image requirements into grants: its sole launch grant
in the baseline mode is non-delegable `human:owner` write for communication.
Enabling reviewed commands additionally grants a non-delegable `shell:*`
execute capability constrained to `allowlist_auto_else_ask`. Enabling context
maintenance adds non-delegable enrichment/maintenance, restricted child-spawn,
and context-compressor image-read capabilities. Selected filesystem, the
restricted `shell:git` class, and typed local-Git scopes remain non-delegable
`approval_policy.requestable_capabilities`, so an in-scope model request still
creates a Human decision and an out-of-scope request is rejected before the
Human queue. Explain shows the resulting id/hash, grants, unmet requirements,
budget, and effect policy.
