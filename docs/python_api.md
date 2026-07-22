# Python API

This page describes the supported Python entrypoints for the current Agent
libOS release. Agent libOS is still experimental; the compatibility boundary
at the end of this page is part of the API contract.

## Imports

Applications should import the Runtime and common interchange models from the
top-level package:

```python
from agent_libos import ObjectQuery, ObjectType, Runtime
```

The names in `agent_libos.__all__` form the documented top-level import
surface. They are grouped below by purpose.

| Area | Top-level names |
| --- | --- |
| Runtime and authority | `Runtime`, `Capability`, `CapabilityRight`, `Rights`, `TaskAuthorityManifest` |
| Process and image state | `AgentImage`, `AgentProcess`, `ProcessResult`, `ProcessSignal`, `ProcessStatus`, `ForkMode` |
| Process outcomes and waits | `ProcessOutcome`, `ExitedProcessOutcome`, `FailedProcessOutcome`, `KilledProcessOutcome`, `ProcessWaitState`, `HostResumeProcessWait`, `HumanProcessWait`, `MessageProcessWait`, `PausedProcessWait`, `ToolProcessWait`, `ChildProcessWait` |
| Object Memory | `AgentObject`, `MaterializedContext`, `MemoryView`, `ObjectHandle`, `ObjectMetadata`, `ObjectNamespace`, `ObjectQuery`, `ObjectRight`, `ObjectType`, `RelationType`, `ViewMode` |
| Object tasks | `ObjectTask`, `ObjectTaskNotification`, `ObjectTaskNotificationStatus`, `ObjectTaskOwnerWatch`, `ObjectTaskStatus` |
| Operations and evidence | `ContextMaterializationManifest`, `OperationEvidenceLink`, `OperationEvidenceRole`, `OperationKind`, `OperationOutcome`, `OperationRecord`, `OperationState` |
| Human, events, tools, and workflows | `HumanRequest`, `Event`, `EventType`, `ToolCallResult`, `ToolCandidate`, `ToolHandle`, `ToolSpec`, `ValidationResult`, `WorkflowRunResult` |

`Rights` is an alias for `CapabilityRight`. `agent_libos.__version__` exposes
the installed package version but is not included in wildcard imports.

The wider `agent_libos.models` package contains subsystem and persistence
models used by the implementation. Their presence there does not add them to
the top-level compatibility surface.

## Opening and closing a Runtime

Use `Runtime.open(...)` in synchronous hosts:

```python
from agent_libos import Runtime

runtime = Runtime.open("local")
try:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="inspect the workspace",
    )
    assert runtime.process.get(pid).pid == pid
finally:
    shutdown = runtime.shutdown()
    if not shutdown["ok"]:
        raise RuntimeError(f"Runtime shutdown failed: {shutdown}")
```

Use `Runtime.aopen(...)` and `Runtime.ashutdown(...)` in an asynchronous host:

```python
import asyncio

from agent_libos import Runtime


async def main() -> None:
    runtime = await Runtime.aopen("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="inspect the workspace",
        )
        assert runtime.process.get(pid).pid == pid
    finally:
        shutdown = await runtime.ashutdown()
        if not shutdown["ok"]:
            raise RuntimeError(f"Runtime shutdown failed: {shutdown}")


asyncio.run(main())
```

`Runtime` is not a synchronous or asynchronous context manager, so callers
must use `try`/`finally`. `close()` remains a compatibility alias for
`shutdown()`; new code should use the explicit shutdown method.

### `Runtime.open(...)` and `Runtime.aopen(...)`

Both factories accept the same inputs:

- `target`: store target. `None` selects the target from the supplied config.
  `"local"` and `":memory:"` create separate ephemeral SQLite stores; a
  filesystem path or `sqlite:///...` target is persistent SQLite, and a
  `postgres://` or `postgresql://` URI selects PostgreSQL.
- `config`: an explicit `AgentLibOSConfig`. The library factories do not load
  the repository's `config.yaml` or a workspace `.env` file.
- `substrate`: an injected `ResourceProviderSubstrate`. If omitted, the
  Runtime builds the local substrate rooted at the selected workspace.
- `module_manifests`, `trusted_modules`, and `trusted_module_sha256`: trusted
  startup-module inputs. Module registration and startup hooks finish before
  the factory returns.

See [Configuration](configuration.md), [Storage](storage.md),
[Providers](providers.md), and [Runtime Modules](modules.md) for the complete
contracts behind these arguments.

`Runtime.open(...)` rejects use inside an active event loop. Use
`await Runtime.aopen(...)` there. The asynchronous factory performs blocking
store work away from the caller loop and finishes loop-affine assembly on that
loop.

### Shutdown contract

`shutdown(*, actor="runtime", reason="runtime.shutdown")` and its asynchronous
counterpart return a mapping. Callers must check `ok`; cleanup failures and an
admission-drain timeout may be reported as `ok: false` rather than raised. A
successful repeated call is idempotent and reports `already_shutdown: true`.
Warnings may be present even after store ownership was successfully released.

Shutdown closes Runtime mutation admission, records the shutdown attempt,
stops owned services and finalizers, and releases the store. It does **not**
mark live `AgentProcess` rows as exited. Process exit is a separate authorized
operation.

Synchronous shutdown refuses to drive async-only teardown from a running event
loop. Event-loop hosts should always await `ashutdown()`. A failed shutdown
keeps unreleased ownership available for a retry when safe. If a Runtime is
permanently recovery-fenced, ordinary shutdown intentionally preserves the
diagnostic store; after inspection, use `release_recovery_diagnostics()` or
await `arelease_recovery_diagnostics()` for the explicit handoff described in
[Architecture](architecture.md).

## Synchronous and asynchronous execution

Choose one style for the lifetime of a Runtime. A Runtime assembled with
`aopen()` can own loop-affine components and should be shut down with
`ashutdown()` on a compatible loop.

| Synchronous host | Asynchronous host | Purpose |
| --- | --- | --- |
| `run_process_once(pid)` | `await arun_process_once(pid)` | Run one process quantum |
| `run_next_process_once()` | `await arun_next_process_once()` | Run one scheduled quantum |
| `run_until_idle(...)` | `await arun_until_idle(...)` | Drain all runnable processes and optionally the Human queue |
| `run_process_until_idle(pid, ...)` | `await arun_process_until_idle(pid, ...)` | Drain one process |
| `run_workflow(tool, args, ...)` | `await arun_workflow(tool, args, ...)` | Run one image-visible tool through ToolBroker without an LLM turn |
| `shutdown()` | `await ashutdown()` | Release the Runtime composition root |

The synchronous `run_until_idle`, `run_process_until_idle`, and `run_workflow`
methods detect an active event loop and direct the caller to their async
counterparts. `max_quanta=None` uses the active Runtime configuration; it does
not mean an unbounded run unless that configuration is itself unbounded.

`run_workflow` returns `WorkflowRunResult`. Expected tool failures and wait
states are represented in that result rather than necessarily raised. Direct
manager and primitive calls retain their typed exception contracts.
Scheduler methods that execute an agent quantum invoke its configured LLM
profile and therefore require the corresponding Host environment and may spend
provider tokens. `run_workflow` is the direct ToolBroker path and does not add
an LLM turn.

## Runtime managers and primitives

`Runtime.open()` assembles the following public composition properties before
returning. These are Host APIs; exposing a manager object to model code would
bypass the intended Tool/Skill visibility layer even though primitives still
enforce their own authority.

| Property | Purpose |
| --- | --- |
| `config`, `workspace_root`, `instance_id` | Effective immutable configuration and Runtime identity |
| `process`, `launch`, `process_transitions`, `scheduler` | Process lifecycle, launch, status transitions, and scheduling |
| `capability`, `authority_manifests` | Capability and Task Authority control planes |
| `memory`, `object_tasks` | Object Memory and Object-bound background tasks |
| `tools`, `syscalls` | ToolBroker and the JIT syscall router |
| `messages`, `human` | Durable process messaging and Human requests |
| `resources`, `ratings` | Hierarchical resource accounting and agent ratings |
| `data_flow`, `protected_operations` | Label/Sink enforcement and protected provider-operation SDK |
| `filesystem`, `git`, `shell`, `clock`, `jsonrpc`, `mcp` | Typed primitives and registered remote-resource boundaries |
| `checkpoint`, `image_registry`, `image_artifacts`, `skills`, `modules` | Checkpoint, image, Skill, and trusted-module registries |
| `operations`, `explain`, `audit`, `events`, `payload_retention` | Causal operations, evidence, audit/events, and payload-retention maintenance |
| `llms`, `llm` | LLM profile registry and process executor |
| `substrate`, `store`, `uow` | Provider substrate and persistence composition boundary |

Manager methods generally require an actor or process id and preserve the same
capability, data-label, approval, resource, event, and audit semantics described
in the subsystem references. A property being Host-accessible does not mean its
operations are authority-free. Conversely, explicitly Host-only control-plane
methods are not model-callable merely because the Runtime exposes them.

Prefer the Runtime facade where one exists:

- process boot: `exec_process`, `spawn_child_process`, `fork_child_process`,
  `set_process_working_directory`, and `resolve_process_working_directory`;
- direct workflows and scheduling: the sync/async methods above;
- images: `register_image` and `get_image`;
- Skills: `register_skill_from_path`, `discover_skills`, `inspect_skill`,
  `activate_skill`, `unload_skill`, and `trust_skill_source`;
- Host Sink trust: `register_sink_trust`, `unregister_sink_trust`,
  `inspect_sink_trust`, and `list_sink_trust`;
- Object view publication: `add_handle_to_process_view`.

For subsystem-specific signatures and security semantics, see
[Runtime Model](runtime_model.md), [Capabilities](capabilities.md),
[Object Memory](object_memory.md), [Data Flow](data_flow.md),
[Checkpoints](checkpoints.md), [Git](git.md), and
[Tools and JIT](tools_and_jit.md).

## Common exceptions

Manager and primitive APIs raise domain exceptions from
`agent_libos.models.exceptions`:

```python
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ResourceLimitExceeded,
    ValidationError,
)
```

| Exception | Meaning |
| --- | --- |
| `ValidationError` | Invalid or unsupported input, configuration, manifest, or state transition |
| `NotFound` | Requested process, Object, registry entry, or other resource is absent or deliberately not disclosed |
| `CapabilityDenied` / `PolicyDenied` | Capability or policy authorization denied the operation |
| `HumanApprovalRequired` / `HumanResponseRequired` | The operation is waiting for a durable Human decision or answer; the exception carries `request_id` |
| `ProcessWaitRequired` / `ProcessMessageWaitRequired` | A resumable operation is waiting for a child process or process message |
| `ResourceLimitExceeded` | A configured process or hierarchical resource budget would be exceeded |
| `GitError` | Stable typed Git failure; inspect `code`, `operation`, `retryable`, and bounded `details` |
| `ProviderHostError` | Sanitized provider exception with `code`, `error_type`, and `correlation_id`; raw provider text is not part of the public error |
| `UnsupportedStoreVersion` | The target store is not compatible with the strict current schema |
| `RuntimeRecoveryRequired` | A durable publication/compensation ambiguity fenced further mutation until reopen or explicit diagnostic release |

`LibOSError` is the common base for these domain errors. ToolBroker calls
normally return a `ToolCallResult` whose `ok`, `error`, and structured payload
encode tool failure; do not assume every tool denial is raised to the Host.

Runtime assembly has an additional ownership-cleanup contract:

```python
from agent_libos.runtime import RuntimeAssemblyCleanupRequired
```

An open failure may contain one or more
`RuntimeAssemblyCleanupRequired` leaves inside a `BaseExceptionGroup`. Use
`RuntimeAssemblyCleanupRequired.extract(error)`, then call `release()` in sync
code or await `arelease()` in async code. The handle exists so a failed startup
cannot silently abandon an owned store or partially assembled component graph.

## Compatibility boundary

- Agent libOS 0.3 is experimental. The top-level `agent_libos.__all__` names
  and the Runtime entrypoints documented here are the intended application
  import surface for this release. Pin the package version when depending on
  exact signatures or dataclass fields.
- Concrete manager classes live in subsystem modules and are assembled by the
  Runtime builder. Their private methods, underscore-prefixed attributes, and
  constructor wiring are implementation details. Use the documented Runtime
  properties rather than importing those concrete classes as an extension API.
- Supported extension boundaries are explicit: the provider substrate,
  Protected Operation SDK, trusted Runtime Modules, Skills, and tool/syscall
  schemas. An arbitrary importable internal class is not an extension point.
- Persisted Runtime state has a strict schema generation. Opening an older,
  newer, incomplete, or hand-built schema may raise `UnsupportedStoreVersion`;
  no general automatic migration guarantee is implied by Python API
  compatibility.
- The CLI has its own documented command contract. The local GUI `/api` is a
  same-build renderer/server interface, not a versioned third-party REST API.
- Provider behavior, real LLM features, PostgreSQL, Deno, PTY, and platform
  support remain subject to the explicit environment gates in
  [Support Matrix](support_matrix.md). Python support for this release is
  `>=3.11,<3.15`.
