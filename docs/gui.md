# Electron GUI

Agent libOS includes a local desktop management console for supervising
processes, messages, human approvals, AgentImage selection/registration/commit,
checkpoints, capabilities, Skills, JSON-RPC endpoints, audit records, and
persisted LLM calls.

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

The GUI server is not a new security boundary. It is a local admin control
surface over the same primitives, Capability checks, human approval flow,
events, and audit records used by the CLI. Its Python entrypoint lives under
`agent_libos.api.gui` with the CLI because both are host-facing API surfaces.
Only a bearer token holder on the same machine can use it; CORS is limited to
loopback HTTP(S) browser origins and does not accept `Origin: null`.

For endpoints that accept an optional `actor`, omitting `actor` runs in GUI
admin mode. Supplying `actor` opts into process-authority mode and requires
that process to hold the capability needed by the underlying primitive, keeping
audit attribution aligned with the capability decision.

Closing the GUI server pauses auto-run and asks the scheduler to stop before it
calls `Runtime.shutdown()` on an owned runtime. If no scheduler worker or
ObjectTask tool thread is still inside synchronous work, shutdown closes owned
runtime resources, including the SQLite store. If a worker cannot be joined
safely, the GUI server leaves the runtime store open and relies on process
teardown instead of closing SQLite underneath the live worker. Host shutdown
does not mark AgentProcess records as exited; process lifecycle changes still
go through the runtime `process.exit` primitive/tool path.

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
npm --prefix gui run typecheck
npm --prefix gui run build
```

The Electron smoke path can be run headlessly with
`AGENT_LIBOS_GUI_SMOKE=1`. By default it verifies the Electron main process,
Python GUI server startup, authenticated `/api/health`, and graceful shutdown
without creating a BrowserWindow. Set `AGENT_LIBOS_GUI_SMOKE_WINDOW=1` when a
machine has a working desktop/GPU stack and you specifically want to exercise
the preload bridge.

The Vite development server is bound to `127.0.0.1` and restricts file serving
to the `gui/` directory. Production dependency audit should remain clean; any
dev-server advisory must be handled with local-only exposure unless an upstream
fix is available.

## Current Workspace

The first screen is process-centered:

- left pane: process tree, status, image, cwd, resource budget/usage, unread
  message badges,
- center pane: selected process timeline and human request cards,
- right pane: details for overview, capabilities, tools/Skills, checkpoints,
  audit, LLM calls, Images, JSON-RPC, and Object Memory summary,
- top bar: database, spawn, auto-run, quanta budget, run, step, pause, refresh.

The default user page exposes the image workflow without opening raw runtime
panels: users can choose a registered image for a new task, import an
AgentImage package, or save the current selected process as a checkpoint-
derived image. Import and commit both require explicit confirmation. Saving as
an image creates a checkpoint only after that confirmation, then commits the
checkpoint into an immutable image artifact.

The operator console provides the fuller registry view: image list, inspect,
spawn/exec selection, package registration, checkpoint commit, and explicit
replace controls.

The scheduler defaults to automatic mode. Users can pause auto-run, step a
selected process, or run the selected process with an optional quantum budget.
Leaving the budget blank runs until the process/runtime becomes idle; entering a
number bounds that run. Automatic runs after spawn/message/exec may advance all
runnable processes, but `POST /api/processes/{pid}/run` is intentionally scoped
to that pid. Real LLM calls are still persisted in `llm_calls`, so the GUI can
show token usage, errors, and bounded prompt/output observability metadata
without exposing raw prompt or provider payloads by default.
If the host runtime is configured with `llm.persist_full_io=True`, the same
`llm_calls` API returns full stored LLM inputs and outputs.

`POST /api/processes` and `POST /api/processes/{pid}/exec` accept optional
`llm_profile` fields for host-selected per-process LLM routing. Snapshots expose
each process `llm_profile_id`; profile secrets stay in the host process
environment and are not returned by the GUI API.

Process snapshots include `resource_budget`, `resource_usage`, and
`resource_remaining` so the GUI can show quota state without treating it as a
Capability grant. Budget exhaustion is still enforced by the runtime and
providers, not by renderer visibility.

## High-Risk Operations

The GUI requires explicit confirmation for high-risk operations before sending
the final request to the server:

- process `exec` and `exit`,
- image package registration and checkpoint-to-image commit,
- checkpoint restore and fork,
- capability grant, delegate, and revoke,
- JSON-RPC method calls,
- Skill registration and trust.

The confirmation dialog shows the pid/resource/action summary. The server also
rejects high-risk requests without `confirmed: true`, so accidental direct HTTP
calls fail closed before invoking the runtime operation.

JSON-RPC endpoint registration through the GUI accepts manifest text only. The
renderer cannot ask the Python GUI server to read an arbitrary host file path;
file/path based registration remains a CLI/admin workflow.

Image package registration follows the same rule. Electron may read a package
directory selected by the user and pass bounded package file payloads to the
local GUI server, but the server rejects host file paths. Registering or
committing an image changes image visibility and baked internal runtime state
only; it does not grant the target image's declared capabilities. Package
workspace grants apply only to the private materialized copy declared by the
package manifest.

## API Summary

Important endpoints:

- `GET /api/health`
- `POST /api/shutdown`
- `GET /api/snapshot`
- `GET /api/events/stream?cursor=<id>`
- `GET /api/processes`
- `POST /api/processes`
- `POST /api/processes/{pid}/run|step|pause|resume|signal|message|interrupt|cd|exec|exit`
- `GET /api/processes/{pid}/messages|human-requests|llm-calls|audit|events|capabilities|checkpoints`
- `GET /api/object-tasks`, `POST /api/object-tasks/start`,
  `GET /api/object-tasks/{task_id}`, and
  `POST /api/object-tasks/{task_id}/cancel|wait|watch-owner`
  (`POST /api/object-tasks/start` accepts `owner_watch`, `watch_events`,
  `watch_channel`, and `watch_kind` for owner-change runner messages; the
  `watch-owner` endpoint updates the same fields for an active task. Wait
  requests are bounded by the GUI object-task wait timeout defaults.)
- `POST /api/human-requests/{request_id}/respond` approves or rejects only
  pending requests; terminal or cancelled requests return a conflict.
- `GET/POST /api/checkpoints`, `/api/skills`, `/api/capabilities`,
  `/api/images`, `/api/jsonrpc`, and `/api/modules`

All endpoints require `Authorization: Bearer <session-token>`.
