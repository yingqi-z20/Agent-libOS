# Python API

This page describes the supported Python entrypoints for the current Agent
libOS release. Agent libOS is still experimental; the compatibility boundary
at the end of this page is part of the API contract.

## In this guide

- [Import the supported public surface](#imports)
- [Open and close a Runtime](#opening-and-closing-a-runtime)
- [Choose synchronous or asynchronous execution](#synchronous-and-asynchronous-execution)
- [Use Runtime managers and primitives](#runtime-managers-and-primitives)
- [Inspect semantic evidence and settlement](#semantic-evidence-control-and-settlement-boundary)
- [Use JSON-RPC and MCP Host APIs](#json-rpc-and-mcp-host-apis)
- [Inject provider protocols](#provider-protocols-and-injection)
- [Handle common exceptions](#common-exceptions)
- [Apply the compatibility boundary](#compatibility-boundary)
- Return to the [documentation home](index.md).

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
| Process outcomes and waits | `ProcessOutcome`, `ExitedProcessOutcome`, `FailedProcessOutcome`, `KilledProcessOutcome`, `ProcessWaitState`, `HostResumeProcessWait`, `HumanProcessWait`, `MessageProcessWait`, `PausedProcessWait`, `StaleExecutionProcessWait`, `ToolProcessWait`, `ChildProcessWait` |
| Object Memory | `AgentObject`, `MaterializedContext`, `MemoryView`, `ObjectHandle`, `ObjectMetadata`, `ObjectNamespace`, `ObjectQuery`, `ObjectRight`, `ObjectType`, `RelationType`, `ViewMode` |
| Object tasks | `ObjectTask`, `ObjectTaskNotification`, `ObjectTaskNotificationStatus`, `ObjectTaskOwnerWatch`, `ObjectTaskStatus` |
| Durable Task Runs | `TaskRunSpecV1`, `TaskRunStatus`, `TaskRunAction`, `TaskRunRetention`, `TaskRunSummary`, `TaskRunLedgerItem` |
| MCP client | `McpProtocolMode`, `McpProtocolEra`, `McpDispatchState`, `McpRetryClass`, `McpConnectionInfo`, `McpDiscoveryResult`, `McpToolListResult`, `McpProviderCallResult`, `McpCallResult` |
| Operations and evidence | `ContextMaterializationManifest`, `OperationEvidenceLink`, `OperationEvidenceRole`, `OperationKind`, `OperationOutcome`, `OperationRecord`, `OperationState` |
| Human, events, tools, and workflows | `HumanRequest`, `Event`, `EventType`, `ToolCallResult`, `ToolCandidate`, `ToolHandle`, `ToolSpec`, `ValidationResult`, `WorkflowRunResult` |

`Rights` is an alias for `CapabilityRight`. `agent_libos.__version__` exposes
the installed package version but is not included in wildcard imports.

`StaleExecutionProcessWait` is exported so Hosts can type-check and serialize a
persisted process projection. It is nevertheless a Store-only recovery receipt:
application code must not construct or submit one as an ordinary transition,
and process transition/execution-completion APIs reject attempts to persist it.
Its identity hashes and generations are diagnostic; TaskRun epoch, safe-point
integrity, and current binding evidence stay in their authoritative records and
must not be inferred from this public value.

The wider `agent_libos.models` package contains subsystem and persistence
models used by the implementation. Their presence there does not add them to
the top-level compatibility surface.

Modern MCP Host-client contracts are exported from `agent_libos.mcp`, not
promoted wholesale into the package root. That surface includes the strict v3
Manifest/parser/Host-policy types; `McpModernClient` and its binding/limit
types; Resource, Prompt, Completion, content, page, MRTR, Task, subscription,
OAuth, and connection-lifecycle models/managers; provider Protocols; and the
explicit DX validation/doctor/probe/scaffold/export/import helpers. See
[MCP Client](mcp.md) for the version gates and exclusions; importing one of
these Host types does not bind it to a Runtime or make it model-visible.

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
  With the default config this is the reserved `"user"` target, which resolves
  to the persistent `~/.agent-libos/runtime/agent-libos.sqlite`. `"local"`,
  `":memory:"`, and bare `sqlite://` create separate ephemeral SQLite stores;
  a filesystem path or file SQLite URI is persistent SQLite, and a `postgres://`
  or `postgresql://` URI selects PostgreSQL.
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

A persistent SQLite database must be outside the effective local workspace.
The factories reject overlap before creating the database, lease, WAL, or SHM
file, including when the caller supplies an already-open store to
`RuntimeBuilder`. There is no unsafe override. In-memory SQLite, PostgreSQL,
and non-local substrates are not subject to this filesystem-containment check.

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

The complete all-process drain call shapes are:

```python
runtime.run_until_idle(
    max_quanta=None,
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
| `memory`, `object_tasks`, `task_runs` | Object Memory, Object-bound background tasks, and Durable Task Run supervision |
| `tools`, `syscalls` | ToolBroker and the JIT syscall router |
| `messages`, `human` | Durable process messaging and Human requests |
| `resources`, `ratings` | Hierarchical resource accounting and agent ratings |
| `data_flow`, `protected_operations` | Label/Sink enforcement and protected provider-operation SDK |
| `filesystem`, `git`, `shell`, `clock`, `jsonrpc`, `mcp` | Typed primitives and registered remote-resource boundaries |
| `checkpoint`, `image_registry`, `image_artifacts`, `skills`, `modules` | Checkpoint, image, Skill, and trusted-module registries |
| `operations`, `explain`, `audit`, `events`, `payload_retention` | Causal operations, evidence, audit/events, and payload-retention maintenance |
| `llms`, `llm` | LLM profile registry and process executor |
| `semantic` | Host evidence/query/review service plus the trusted Host-only, one-way live kill switch `set_mode("off")` |
| `semantic_control` | Trusted Host composition/control port for startup policy admission and authority-narrowing disable; never a remote or model-facing surface |
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

`Runtime.task_runs` is a Host-only control plane for one root AgentProcess tree.
It provides `create`, `get`, `list`, `run_until_blocked`, `wait`, `pause`,
`resume`, `cancel`, `follow_up`, `recover`, `rerun`, and the Host/admin-only
`purge_payloads` operation for a terminal permanent-retention Run. It also
provides paged `list_requirements`, `list_ledger`, and `list_human_requests`
reads plus server-derived `recovery_options`. Mutations require a monotonic
expected Run revision and stable command id. Rerun creates a linked new Run
rather than rewinding history; whenever default terminal cleanup or an explicit
Host purge has removed the source goal, `spec_overrides` must supply a
replacement `goal`. Only a still-retained `permanent` Run may reuse its goal.
Creation instead uses a stable client request id and has no pre-existing
revision. `TaskRunSummary.payloads_purged` is the server-derived, content-free
signal callers should use when deciding whether rerun needs that replacement. Do
not expose this manager to model code;
every process action still goes through the normal Tool/Skill and primitive
authority boundaries. See [Durable Task Runs](durable_task_runs.md) for payload
opt-in, retention, recovery, and external-effect semantics.

### Semantic evidence, control, and settlement boundary

The semantic surface has four deliberately separate layers:

1. **Evidence and review.** `Runtime.semantic` is the stable public Host facade
   for the status, query, metrics, and review-evidence methods listed below.
   `append_review_label(...)` appends evidence only; it does not authorize or
   settle an operation.
2. **Trusted Host kill switch and control.** A local Host that owns the Runtime
   may call `runtime.semantic.set_mode("off")`. This is a supported one-way live
   kill switch: it durably disables semantic authority before publishing the
   in-memory `off` mode, stops new capture and claim work, and invalidates
   unconsumed or undispatched semantic grants. It returns `None`. Re-enabling
   from `off`, or changing between active modes, requires Runtime restart and
   startup admission. The Host-owned Python component graph also exposes
   `runtime.semantic_control` for startup policy admission and
   authority-narrowing disable; callers must treat it as trusted composition
   control and never pass it to process or model code.
3. **Private settlement.** Machine allow/deny settlement ports are builder-wired
   implementation components that share the Human revision/status CAS and
   terminal kernel. They are not Runtime facade methods and are not part of the
   supported evidence/review or Host-control APIs.
4. **Remote and model boundary.** There is no HTTP, GUI, model Tool, Skill, JIT,
   Module, or other remote/model write entrypoint for semantic review, policy,
   control, or settlement. The local Host CLI's `semantic review import` command
   is only a wrapper around the evidence-only review append; it exposes no
   policy, control, or settlement mutation.

Within the first layer, the complete public Host evidence/review surface is:

| Method | Return |
| --- | --- |
| `runtime.semantic.status()` | schema-v3 status `dict[str, Any]` |
| `runtime.semantic.flow_status()` | FlowGraph status `dict[str, Any]` |
| `runtime.semantic.metrics(*, window=None, action_id=None, tenant_bucket_sha256=None, epoch_id=None, risk=None)` | metrics `dict[str, Any]` |
| `runtime.semantic.control_status()` | control-state `dict[str, Any]` |
| `runtime.semantic.query_assessments(*, pid=None, request_id=None, operation_id=None, kind=None, status=None, domain=None, action_id=None, tenant_bucket_sha256=None, after=None, limit=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.get_assessment(assessment_id)` | assessment `dict[str, Any]` or `None` |
| `runtime.semantic.query_flow_entities(*, after=None, limit=None, pid=None, kind=None, tenant_bucket_sha256=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_flow_edges(*, after=None, limit=None, pid=None, relation=None, node_id=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_flow_lineage(node_id, *, direction="upstream", after=None, limit=None, max_depth=8)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_machine_settlements(*, after=None, limit=None, pid=None, request_id=None, effect_id=None, action_id=None, tenant_bucket_sha256=None, outcome=None, epoch_id=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_policy_epochs(*, after=None, limit=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_control_history(*, after=None, limit=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.query_health_events(*, after=None, limit=None, severity=None, code=None, epoch_id=None)` | keyset page `dict[str, Any]` |
| `runtime.semantic.append_review_label(*, settlement_id, outcome, reviewer_id, evidence_sha256, reviewed_at=None)` | appended review-evidence `dict[str, Any]` |

Paged queries are hard-bounded and keyset-paged, with three limit groups.
Assessment queries (`query_assessments`) use `semantic.assessment_list_limit`
as the omitted direct-Host default; flow queries (`query_flow_entities`,
`query_flow_edges`, `query_flow_lineage`) use `semantic.flow_query_limit`; and
settlement-family queries (`query_machine_settlements`, `query_policy_epochs`,
`query_control_history`, `query_health_events`) use
`semantic.settlement_list_limit`. In each group an explicit limit cannot
exceed the smaller of the matching hard limit
(`semantic.assessment_list_hard_limit`, `semantic.flow_query_hard_limit`, or
`semantic.settlement_list_hard_limit`) and 500. The CLI and local HTTP API
apply their stricter 50-row default and 100-row maximum.
The returned mappings are payload-free projections containing typed findings,
Shadow outcome, normalized Human observation, calibration, reserved nullable
token/cost fields, and provenance digests. An external classifier may populate
`input_tokens`, `output_tokens`, and `cost_microunits` from an exact
`LLMCompletion.usage` dictionary, selecting only exact non-negative integers
through `2^53 - 1`, matching JSON/TypeScript safe-integer decoding.
`prompt_tokens` and `completion_tokens` are accepted aliases; canonical/alias
conflict invalidates only that counter. Unknown/raw
usage fields are never returned. Deterministic/scripted, missing, non-exact, or
invalid telemetry remains `None`, and populated values are not authoritative
billing evidence. `append_review_label(...)` is the only public evidence/review
write and is a strict Host review-evidence append; it cannot settle a request,
issue/revoke a Capability, activate/revoke an epoch, change control state,
mutate labels, or call a provider. Runtime-internal
capture, enforcement, and worker methods are not exported as model Tools,
Skills, JIT syscalls, Modules, HTTP writes, or Runtime facade settlement APIs.
See [Semantic Approval and Data
Identification](semantic_shadow.md).

`status()` returns schema v3 with queue, assessments, control, FlowGraph,
machine, actual-auto-approval, and review-metric sections. The dedicated
FlowGraph endpoint exposes the same typed status with its bounded graph reads.
Its
`assessments.by_status` and
`assessments.by_domain` mappings contain the complete closed enum key sets and
must each sum to `assessments.total`; inconsistent mappings are rejected by the
CLI and GUI adapters. The exact machine counter set is `eligible`, `issued`,
`consumed`, `succeeded`, `failed`, `unknown`, `expired`, `revoked`,
`race_lost`, and `denied`. `actual_auto_approval.rate` is `None` exactly when
`eligible` is zero; `review_metrics.unsafe_rate` is `None` exactly when
`reviewed` is zero; and `review_metrics.issued_review_rate` is `None` exactly
when `issued` is zero. A non-null review rate does not by itself prove complete
Host review coverage.

Embedded Hosts may pass `semantic_assessor=` to `Runtime(...)`,
`Runtime.open(...)`, or `Runtime.aopen(...)` only when the configured adapter is
`scripted`; an enabled scripted semantic mode requires that injected object to implement
the synchronous `assess()` port. The external adapter rejects an override, and
other adapters reject this injection. This constructor dependency is Host code,
not a CLI/HTTP/GUI/model/Skill/JIT/Module extension surface.

The same three constructors accept the separate Host-only
`semantic_tenant_bucketer=` callback. It receives a canonical tenant string and
must return a 64-character lower-case SHA-256-shaped digest; deployments should
derive that value with a deployment-keyed HMAC. When omitted, assessment rows
use `tenant_bucket_sha256=None` and no tenant grouping occurs. The callback is
not configurable through YAML or any CLI, HTTP, GUI, Tool, Skill, JIT, or
Module entrypoint, and its failure is isolated as a semantic capture failure.

## JSON-RPC and MCP Host APIs

The remote-resource managers are Host APIs. Their supported synchronous
signatures are listed below; `acall(...)`, `adiscover(...)`,
`alist_tools(...)`, and `acall_tool(...)` have the same arguments and return values as their
synchronous counterparts, but must be awaited.

| JSON-RPC method | Return |
| --- | --- |
| `runtime.jsonrpc.register_endpoint(endpoint, *, actor="runtime", replace=False, require_capability=True, source=None)` | inspected endpoint `dict` |
| `runtime.jsonrpc.register_endpoint_from_yaml_text(text, *, actor, replace=False, require_capability=True, source=None)` | inspected endpoint `dict` |
| `runtime.jsonrpc.list_endpoints(*, actor=None, require_capability=True, text=None, limit=None)` | bounded `list[dict]` |
| `runtime.jsonrpc.list_endpoints_window(*, actor=None, require_capability=True, text=None, limit=None)` | `(list[dict], has_more)` |
| `runtime.jsonrpc.inspect_endpoint(endpoint_id, *, actor=None, require_capability=True, include_sensitive_fields=False)` | endpoint `dict` |
| `runtime.jsonrpc.unregister_endpoint(endpoint_id, *, actor="runtime", require_capability=True)` | deletion `dict` |
| `runtime.jsonrpc.call(pid, endpoint_id, method_id, params=None, *, source_oids=None)` / `await runtime.jsonrpc.acall(...)` | `JsonRpcCallResult` |
| `runtime.jsonrpc.grant_method(pid, endpoint_id, method_id, *, right, issued_by="jsonrpc", delegable=True)` | capability value |

| MCP method | Return |
| --- | --- |
| `runtime.mcp.register_server(server, *, actor="runtime", replace=False, require_capability=True, source=None)` | inspected server `dict` |
| `runtime.mcp.register_server_from_yaml_text(text, *, actor, replace=False, require_capability=True, source=None)` | inspected server `dict` |
| `runtime.mcp.validate_server_manifest(server)` | strict typed v1/v2/v3 manifest; validation only, with no registry/provider effect |
| `runtime.mcp.get_server_manifest(server_id)` | validated typed registered manifest for trusted Host tooling |
| `runtime.mcp.import_server_manifest(server, *, expected_current_sha256, actor="runtime", replace=False, require_capability=True, source=None)` | reviewed import for any schema; v3 uses Store-atomic CAS, while v1/v2 use the Runtime-local registry fence |
| `runtime.mcp.import_v3_manifest(server, *, expected_current_sha256, actor="runtime", replace=False, require_capability=True, source=None)` | exact-CAS inspected v3 server `dict`; `None` expects no current row |
| `runtime.mcp.list_servers(*, actor=None, require_capability=True, text=None, limit=None)` | bounded `list[dict]` |
| `runtime.mcp.list_servers_window(*, actor=None, require_capability=True, text=None, limit=None)` | `(list[dict], has_more)` |
| `runtime.mcp.inspect_server(server_id, *, actor=None, require_capability=True, include_sensitive_fields=False)` | server `dict` |
| `runtime.mcp.discover(server_id, *, actor=None, require_capability=True)` / `await runtime.mcp.adiscover(...)` | `McpDiscoveryResult` for Manifest v2 modern-capable modes |
| `runtime.mcp.list_tools(server_id, *, actor=None, require_capability=True, refresh=False)` / `await runtime.mcp.alist_tools(...)` | tool-list `dict` |
| `runtime.mcp.list_resources(server_id, *, cursor=None, actor="runtime", model_visible_only=False)` / `await runtime.mcp.alist_resources(...)` | bounded v3 `McpPage[McpResource]` |
| `runtime.mcp.list_resource_templates(server_id, *, cursor=None, actor="runtime", model_visible_only=False)` / `await runtime.mcp.alist_resource_templates(...)` | bounded v3 `McpPage[McpResourceTemplate]` |
| `runtime.mcp.read_resource(server_id, resource_id, *, variables=None, actor="runtime", for_model=False)` / `await runtime.mcp.aread_resource(...)` | governed v3 `McpComplete`, `McpInputRequired`, or `McpRemoteTask` |
| `runtime.mcp.list_prompts(server_id, *, cursor=None, actor="runtime")` / `await runtime.mcp.alist_prompts(...)` | bounded v3 `McpPage[McpPrompt]` Host facade |
| `runtime.mcp.get_prompt(server_id, prompt_id, *, arguments=None, confirmed=False, expected_preview_sha256=None, actor="runtime")` / `await runtime.mcp.aget_prompt(...)` | governed Prompt result; confirmation is bound to the exact preview digest |
| `runtime.mcp.complete_prompt(server_id, reference_type, reference_id, argument, *, context=None, actor="runtime")` / `await runtime.mcp.acomplete_prompt(...)` | governed bounded Completion result |
| `runtime.mcp.unregister_server(server_id, *, actor="runtime", require_capability=True)` | deletion `dict` |
| `runtime.mcp.call_tool(pid, server_id, tool_id, arguments=None, *, source_oids=None)` / `await runtime.mcp.acall_tool(...)` | v1/v2: `McpCallResult`; v3: `McpComplete`, `McpInputRequired`, or `McpRemoteTask` |
| `runtime.mcp.grant_tool(pid, server_id, tool_id, *, right, issued_by="mcp", delegable=True)` | capability value |

`McpModernClient` is the lower-level Host composition surface for v3
Resources, Resource Templates, Prompts, and Completion. It provides sync and
`a`-prefixed async `list_resources`, `list_resource_templates`,
`read_resource`, `list_prompts`, `get_prompt`, and `complete_prompt` methods.
The Runtime methods above are the protected process/model Resource facade;
Prompts, Completion, OAuth, subscriptions, continuations, and remote Tasks are
not model tools. A Host that exposes those operations must supply the governed
binding/provider managers and preserve their Capability, data-flow, effect,
resource, event, audit, and lifecycle fences.

Exact-v3 custom Provider protocols are exported from `agent_libos.mcp`. Each
implementation declares `mcp_manifest_schema_version = 3` and
`mcp_protocol_revision = "2026-07-28"`; Runtime composition rejects legacy
lookalikes, synchronous or variadic methods, and a continuation provider that
does not implement Tool, Resource-read, and Prompt-get continuation. The modern
Tool SPI receives the operation-local `sensitive_values` snapshot. Custom
non-Complete results become public only when their local reference and full
authority/effect binding match the durable continuation or Task manager;
Completion is Complete-only.

These custom SPIs are trusted, cooperative Host composition. An implementation
must honor its absolute `deadline`, yield rather than block the event-loop
thread, keep CPU work bounded or move it behind a killable process, and propagate
cancellation. Runtime checks before and after dispatch and records an
entered-provider timeout as unknown/no-replay, but it cannot safely hard-kill
arbitrary in-process Python. The built-in governed SDK/transport providers retain
hard bounded I/O and process cleanup.

`actor`, `require_capability`, `include_sensitive_fields`, `source`, and the
`grant_*` conveniences are trusted Host control-plane inputs. Do not derive
them from model output or expose these manager methods as model tools.
`require_capability=False`, an omitted registry-read actor, and
`include_sensitive_fields=True` are Host bypass/disclosure modes, not ways for
a process to self-authorize. Calls to remote methods/tools always use the
target `pid` and still enforce that process's complete Capability, Task
Authority, Human, data-flow, resource, effect, event, and audit path.
`get_server_manifest` and sensitive inspection can reveal Host-declared URI and
environment-variable references, but never resolved secret values; keep them
off model, GUI, and untrusted extension surfaces.

The `*_from_yaml_text` methods parse supplied text; they do not open `source`
as a path. `source` is evidence metadata only. Trusted Host code must read and
bound a Host-controlled file itself. If a process supplies a manifest path,
use the CLI actor mode or the filesystem primitive so its filesystem authority
is enforced before passing the bounded text to the registry manager.

MCP public result models distinguish configured `McpProtocolMode` from
negotiated `McpProtocolEra`. `McpConnectionInfo` reports only bounded,
non-secret operation-local diagnostics; it is neither a capability nor a
persisted session. The existing `McpProvider` signatures remain compatible for
Manifest v1. Manifest v2 requires the separately feature-detected
`McpModernProtocolProvider` extension, so an older custom provider fails before
dispatch instead of receiving an unexpected call shape.

### Provider protocols and injection

Import provider protocols and the default composition from the public
`agent_libos.substrate` package, not from implementation modules. The following
is an executable composition skeleton: it runs when the caller supplies
concrete provider objects implementing the complete protocols; it is not a
provider implementation by itself.

```python
from pathlib import Path

from agent_libos import Runtime
from agent_libos.substrate import (
    JsonRpcProvider,
    LocalResourceProviderSubstrate,
    McpProvider,
    ResourceProviderSubstrate,
)


def open_with_remote_providers(
    target: str | Path,
    workspace: Path,
    *,
    jsonrpc_provider: JsonRpcProvider,
    mcp_provider: McpProvider,
) -> Runtime:
    substrate: ResourceProviderSubstrate = LocalResourceProviderSubstrate(
        workspace
    )
    substrate.jsonrpc = jsonrpc_provider
    substrate.mcp = mcp_provider
    return Runtime.open(target, substrate=substrate)
```

`ResourceProviderSubstrate` is the structural composition protocol;
`JsonRpcProvider` and `McpProvider` are the primitive-facing transport and
classification contracts. A replacement must implement every protocol method,
including conservative external-effect classification. It does not inherit
authority responsibilities from the primitive and must not perform transport
work outside the protected operation it is called from. The bundled concrete
providers are `HttpJsonRpcProvider` and `SdkMcpProvider`; the latter's optional
subprocess-budget extension is `McpSubprocessLimitsProvider`. See
[Providers](providers.md), [JSON-RPC](jsonrpc.md), and [MCP](mcp.md) before
injecting a custom backend.

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
| `ProcessRevisionConflict` / `TaskRunRevisionConflict` / `TaskRunCommandConflict` | A compare-and-swap fence rejected the mutation: the expected process or Durable Task Run revision was stale, or a Task Run command id was reused for a different canonical request; retry with the current revision and the same command id under the revision/command-id contract in [Durable Task Runs](durable_task_runs.md) |
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

- Agent libOS 1.5.2 is experimental. The top-level `agent_libos.__all__` names
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
  auxiliary protocols. In particular, the 1.0 line preserves the legacy Git
  provider `run(...)` signature introduced before 1.0; subprocess-aware
  providers may opt in via
  `GitSubprocessScopeProvider` and `GitLimitedRunProvider`. Hosts must test the
  auxiliary protocol and require its `supports_subprocess_limits` flag before
  making a budgeted call. A process with a configured Git subprocess budget
  fails closed when its provider does not implement the scoped supervision
  extension.
- The 1.0 line likewise preserves the legacy `McpProvider`
  `validate_and_call(...)`, `list_tools(...)`, and `call_tool(...)` signatures
  introduced before 1.0.
  An MCP provider whose three methods also accept `limits=` opts in through
  `McpSubprocessLimitsProvider`. The Runtime never passes that keyword to a
  legacy provider; when a stdio operation has a configured subprocess budget,
  a provider without the extension is rejected before provider dispatch.
  Agent libOS 1.2.1 added modern discovery/negotiation through the optional
  `McpModernProtocolProvider`; it did not add required parameters to those
  three methods.
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
