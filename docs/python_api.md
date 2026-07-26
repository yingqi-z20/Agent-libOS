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
  Runtime builds the local substrate rooted at `Path.cwd()` as observed when
  the Runtime opens; changing the process directory later does not re-root it.
- `module_manifests`, `trusted_modules`, and `trusted_module_sha256`: trusted
  startup-module inputs. Module registration and startup hooks finish before
  the factory returns.

See [Configuration](configuration.md), [Storage](storage.md),
[Providers](providers.md), and [Runtime Modules](modules.md) for the complete
contracts behind these arguments.

`Runtime.open(...)` rejects use inside an active event loop. Use
`await Runtime.aopen(...)` there. The asynchronous factory performs store open
and the complete Runtime assembly in a blocking worker. The caller loop owns
the startup authorization/cancellation handshake and drains any required async
cleanup before returning or raising; it does not run the assembly itself.

### Shutdown contract

`shutdown(*, actor="runtime", reason="runtime.shutdown")` and its asynchronous
counterpart return a mapping. Callers must check `ok`; cleanup failures and an
admission-drain timeout may be reported as `ok: false` rather than raised. A
successful repeated call is idempotent and reports `already_shutdown: true`.
Warnings may be present even after store ownership was successfully released.

An ordinary shutdown attempt that passes preflight closes Runtime mutation
admission and records audit/event evidence before stopping owned services and
finalizers and releasing the store. Preflight misuse, an admission-drain
timeout, or an active recovery-required fence can return or raise before that
durable attempt record exists. Shutdown does **not**
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
| `run_workflow(tool, args, ...)` | `await arun_workflow(tool, args, ...)` | Run one tool from the Image-bound complete process table through ToolBroker, without an LLM turn or model-projection check |
| `shutdown()` | `await ashutdown()` | Release the Runtime composition root |

The synchronous `run_until_idle`, `run_process_until_idle`, and `run_workflow`
methods detect an active event loop and direct the caller to their async
counterparts. `max_quanta=None` uses the active Runtime configuration; it does
not mean an unbounded run unless that configuration is itself unbounded.

The complete all-process drain signatures are:

```python
runtime.run_until_idle(
    max_quanta=None,
    *,
    pids=None,
    process_human_queue=True,
    cancel_inflight_on_budget_exhaustion=True,
    human=None,
    human_auto_approve=None,
    human_auto_policy=None,
    human_auto_answer=None,
)
await runtime.arun_until_idle(
    max_quanta=None,
    *,
    pids=None,
    process_human_queue=True,
    cancel_inflight_on_budget_exhaustion=True,
    human=None,
    human_auto_approve=None,
    human_auto_policy=None,
    human_auto_answer=None,
)
```

`pids` must be `None` or a nonempty iterable of distinct canonical PID strings;
it restricts both scheduler admission and the outer Human-queue drain rather
than widening to all processes. With `process_human_queue=True`, the Runtime
alternates runnable-process batches with pending requests for the selected
Human (`human`, or the configured default) until neither side makes progress.
Setting it to `False` disables that outer queue drain. When the shared quantum
budget is exhausted, `cancel_inflight_on_budget_exhaustion=True` applies the
configured bounded drain and then cancels still-pending admitted futures;
`False` lets already admitted quanta finish, while still admitting no new
ordinary quantum. Cancellation never rolls back a provider effect that may
already have started.

The three `human_auto_*` values are an immutable policy supplied by trusted
Host code for this invocation and its scheduler/JIT descendants.
`human_auto_approve` answers boolean approvals and, absent an explicit policy,
maps permission requests to `always_allow` or `always_deny`;
`human_auto_policy` selects `always_allow`, `always_deny`, or `ask_each_time`
for permission-policy requests; `human_auto_answer` supplies typed question
answers. Because these inputs can produce durable Human and permission-policy
decisions, they must not be derived from model output. They do not bypass
Capability, Task Authority, data-flow, or provider checks: each decision still
uses the ordinary Human transition, validation, event, and audit paths.

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
| `config`, `workspace_root`, `instance_id` | Effective frozen configuration and Runtime identity; authority-rule conditions are defensively copied and recursively frozen |
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

Manager methods have per-operation authority contracts; the presence of an
`actor` or `pid` parameter alone does not tell a caller whether capability
checks are enabled. A property being Host-accessible does not mean its
operations are authority-free. Conversely, explicitly Host-only control-plane
methods are not model-callable merely because the Runtime exposes them. Never
hand a Runtime manager or facade to untrusted/model code; project model access
through the process tool table or a Skill.

Prefer the Runtime facade where one exists. The following table records the
authority mode and public return shape of the less obvious facades; parameters
shown are the supported signature, with the typed source remaining
authoritative for imported model types.

| Facade signature | Return | Authority contract |
| --- | --- | --- |
| `exec_process(pid, image, *, args=None, goal=None, preserve_memory=True, preserve_capabilities=False, llm_profile_id=None, source_oids=None, source_labels=None, source_context=None)` | updated process value | Process-authorized image boot/exec, label release, resource, event, and audit path |
| `spawn_child_process(parent, goal, *, image=None, inherit_capabilities=None, resource_budget=None, working_directory=None, llm_profile_id=None, source_oids=None, source_labels=None, source_context=None)` | child pid `str` | Parent process spawn/image/cwd/data-flow authority and hierarchical budget path |
| `fork_child_process(parent, goal, *, memory_view=None, capabilities=None, inherit_capabilities=None, resource_budget=None, image=None, mode=ForkMode.RESTRICTED, working_directory=None, llm_profile_id=None, source_oids=None, source_labels=None, source_context=None)` | child pid `str` | Parent process fork, view, capability, image/cwd/data-flow authority and budget path |
| `set_process_working_directory(pid, path)` / `resolve_process_working_directory(pid, path)` | updated process / normalized `str` | Process filesystem-directory `read` authority |
| `register_image(image, *, actor="runtime", replace=False)` | `None` | Host-authorized registry facade; it deliberately calls the registry with capability enforcement disabled |
| `get_image(image_id)` | detached `AgentImage` copy | Host read; missing ids raise `KeyError`, not `NotFound` |
| `register_skill_from_path(path, *, actor="runtime", replace=False, source_type="runtime")` | Skill summary `dict` | Host-filesystem/Host-authorized facade; capability enforcement is disabled |
| `discover_skills(text=None)` / `inspect_skill(skill_id)` | summaries / summary `dict` | Host registry reads; capability enforcement is disabled |
| `activate_skill(pid, skill_id, *, expected_package_sha256=None)` / `unload_skill(pid, skill_id)` | Skill summary `dict` | Host-authorized target-process mutation; these facades deliberately disable capability enforcement. Supply the discovery hash to compare-and-swap content; omission is for trusted Host compatibility only. |
| `trust_skill_source(*, source_type, source, package_sha256, actor="runtime")` | trust summary `dict` | Host trust mutation; capability enforcement is disabled |
| `register_sink_trust(spec, *, actor, replace=False)` / `unregister_sink_trust(pattern, *, actor)` | `SinkTrustSpec` | Host-only control plane, but the supplied actor must hold Sink-registry mutation authority |
| `inspect_sink_trust(pattern)` / `list_sink_trust(*, active_only=True, generation=None)` | optional spec / tuple of specs | Host-only registry reads |
| `add_handle_to_process_view(pid, handle)` | `None` | Host publication of an already issued Object handle; it is not a model-callable acquire operation |

The scheduling and workflow signatures are listed in the previous section.
`run_workflow`/`arun_workflow` still traverse ToolBroker and process authority;
the Host-only image and Skill convenience facades above do not. Call the
underlying manager method with `require_capability=True`, or expose the normal
tool/Skill route, when a process actor rather than the Host initiates such an
operation.

For subsystem-specific signatures and security semantics, see
[Runtime Model](runtime_model.md), [Capabilities](capabilities.md),
[Object Memory](object_memory.md), [Data Flow](data_flow.md),
[Checkpoints](checkpoints.md), [Skills](skills.md), [Git](git.md), and
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
- Provider protocol evolution is additive through exported, runtime-checkable
  auxiliary protocols. In particular, the 0.3 Git provider contract retains
  its original `run(...)` signature; subprocess-aware providers may opt in via
  `GitSubprocessScopeProvider` and `GitLimitedRunProvider`. Hosts must test the
  auxiliary protocol and require its `supports_subprocess_limits` flag before
  making a budgeted call. A process with a configured Git subprocess budget
  fails closed when its provider does not implement the scoped supervision
  extension.
- The 0.3 `McpProvider` contract likewise retains the original
  `validate_and_call(...)`, `list_tools(...)`, and `call_tool(...)` signatures.
  An MCP provider whose three methods also accept `limits=` opts in through
  `McpSubprocessLimitsProvider`. The Runtime never passes that keyword to a
  legacy provider; when a stdio operation has a configured subprocess budget,
  a provider without the extension is rejected before provider dispatch.
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
