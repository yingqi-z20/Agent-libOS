# Tools And Deno/TypeScript JIT

LLM-facing tools are stable wrappers over libOS primitives. They provide names,
schemas, validation, and model ergonomics. Primitives enforce authority.

Tool visibility is not resource authority. A process can call only tools in its
process tool table, but filesystem, shell, JSON-RPC, human, memory, image, and
process effects are still authorized by the primitive path. `ToolPolicy`
contains declaration metadata such as `declared_permissions` and
`declared_confirmation_required`; it is shown in tool specs for humans and UI,
but it does not grant permissions or approve execution.

## Built-In Tools

The current built-in tool surface includes tools for:

- Object Memory: create, append, read, list namespaces, and bridge objects to
  files.
- Filesystem: read/write text, list/create/delete directories, and delete files.
- Human I/O: ask questions, output messages, and request permission.
- Clock: current time and async sleep.
- Process lifecycle: fork, spawn, wait, list children, signal, merge memory,
  exec, exit, cwd get/set, and process messages.
- Context: `compact_process_context` compresses the caller's
  `llm_context:<pid>` object through a `context-compressor:v0` child process.
- Shell: argv-only subprocess execution through policy.
- JSON-RPC: list/inspect registered endpoints and call registered methods.
- Image registry: load image manifests from YAML.
- Checkpoint: create, list, inspect, diff, restore, and fork.
- Skills: discover, activate, read bundled resources, and unload.
- JIT: propose, validate, and register Deno/TypeScript tools.
- Utility actions such as `echo` and `parse_pytest_log`.

Use `uv run agent-libos tools` to inspect registered tools in a runtime.

## Context Compaction

`compact_process_context` is a model-visible wrapper for bounded long-running
sessions. It reads the caller process' `llm_context:<pid>` object, spawns a
`context-compressor:v0` child image with only `process_exit` visible, and
replaces the caller context with one `context_compacted` entry plus the recent
verbatim entries requested by `preserve_recent_entries`.

The writeback path is method-neutral: different compressors may produce the
standard compact summary contract, while the LLM context helper records
`compaction_method` and `compaction_metadata` on the `context_compacted` entry
and owns the schema validation, version check, and replacement.

The tool does not grant external resource authority to the compressor. The
compressor child receives only the current chunk, prior stage summary, and stage
goal material needed for summarization; filesystem, shell, memory-write,
JSON-RPC, human, Skill, checkpoint, and process-control access remain absent
unless separately granted by normal primitives. The wrapper is visible to the
model, but Object Memory and Process primitives still enforce reads, writes,
child creation, waiting, resource budgets, audit, and lifecycle.

Compaction is fail-closed. If the compressor fails or is killed, returns an
invalid or empty schema, the source context version changes before final
writeback, resource limits are exceeded, or the durable pending state cannot be
resumed, the tool returns failure and leaves the original materialized context
unchanged. Pending child waits store the minimum resume state in
`llm_pending_actions`; after runtime reopen the compressor child goal can be
reconstructed and the final compacted context is recreated under the same
`llm_context:<pid>` name when the old runtime-only payload is no longer
materializable.

## Workflow Entry Point

A workflow is a tool that a user runs directly. `Runtime.run_workflow()` and
`uv run agent-libos workflow run <tool>` spawn a fresh AgentProcess, call one
tool through ToolBroker, and return the normal tool result JSON. The entrypoint
does not run the LLM scheduler and does not create a second authority model:
the selected image's process tool table still controls visibility, while
primitives enforce capabilities, approval, budgets, events, and audit.

Successful workflow calls append the tool result object to the workflow
process view and exit the process with that result. Failed calls mark the
process failed. Blocking human, child-process, or process-message waits are
returned as explicit waiting results so the caller can resume through the
normal runtime mechanisms. If the tool itself performs `process.exit` or
`process.exec`, the workflow runner leaves that lifecycle decision intact.

## Object Task Entry Point

Object tasks let an AgentObject hold asynchronous tool work. `start_object_task`
creates a host-managed runner child process, narrows that runner's process tool
table to the requested visible tool, and calls the tool through ToolBroker. The
runner is excluded from the LLM scheduler even if a message wakes it back to a
`RUNNABLE` process status, and it does not grant external authority unless the
creator explicitly delegates capabilities into the runner.

Successful tasks create the usual tool result object and link the owner object
to that result with `PRODUCED`. Notifications are ordinary process messages
from `object_task:<task_id>` on the `object-task` channel by default, with
`normal` or `interrupt` kind. The `result_oid` in a notification is only a
reference; it is not an object capability.

When `owner_watch` is enabled, Object Memory `updated` and outgoing `linked`
events on the owner object are delivered to the runner process as ordinary
process messages, on `object-task-owner` by default. The notice is produced by
the Object Memory primitive after the change is committed and audited, includes
only ids/version/link metadata, and may resume a task that is blocked in
`receive_process_messages`; it does not run the LLM scheduler. Tools that block
after non-trivial side effects are not automatically replayed on owner-watch
messages unless they are explicitly known to be safe.

## Writing Python Tools

Python tools should not directly access host resources. Use this pattern:

1. Define a Pydantic input schema and optional output schema.
2. Subclass `SyncAgentTool` for blocking local code or `BaseAgentTool` for
   async code.
3. Keep validation and model-facing ergonomics in the tool.
4. Call `ctx.runtime.<primitive>` for process, memory, filesystem, human,
   clock, shell, image, Skill, checkpoint, or other libOS operations.
5. Let primitives enforce capability checks, containment, audit, events, human
   approval, checkpoint semantics, and policy hooks.
6. Register the tool through the runtime composition root or ToolBroker-backed
   registry.

Do not put direct filesystem, terminal, network, shell, browser, database, or
credential access inside a model-facing tool unless that code is itself the
libOS primitive or a sandbox backend.

## JIT Tool Lifecycle

Agent-authored JIT tools use TypeScript and run under Deno. Python JIT tools
are intentionally not supported.

The manual lifecycle is:

1. `propose_jit_tool`: store candidate metadata and TypeScript source.
2. `validate_jit_tool`: run format/dependency lint, import allowlist checks,
   and configured tests under the sandbox backend.
3. `register_jit_tool`: add the validated tool only to the registering process
   tool table.

Registered JIT tools are process-local but persistent: when a runtime reopens an
existing SQLite store, it reloads executable TypeScript sources only for JIT
tool ids still referenced by a process tool table. Stale ephemeral tool
references with no recoverable registered source are removed from the process
tool table fail-closed instead of being shown to the model as broken tools.

Skill activation uses the same validation and registration path for bundled JIT
tools declared in package metadata and stored as `scripts/*.ts` resources.
Image package boot uses that same ToolBroker validation path before package JIT
tools become visible in the new process.

## LLM Exposure Strategy

Images use `jit_tool_exposure: direct` by default. In direct mode, every visible
process-local JIT tool is exposed to the model as its own OpenAI function tool.

Images may opt into `jit_tool_exposure: multiplexed`. In multiplexed mode,
static tools are still exposed normally, but all visible JIT tools are routed
through one stable OpenAI function named `run_jit_tool`:

```json
{"tool_name":"jit_tool_name","arguments":{}}
```

The runtime maps that protocol call back to the real process-local JIT tool,
validates `arguments` against the JIT tool's stored `input_schema`, and then
uses the normal ToolBroker, Deno sandbox, resource, capability, event, and audit
paths. `run_jit_tool` is not a real process tool and cannot be called through
`runtime.tools.call`.

Multiplexed mode does not inject a JIT catalog into prompt or context. The
image or loaded Skill instructions must describe the valid JIT names and
argument shapes. The name `run_jit_tool` is reserved for multiplexed images and
cannot be used as a real default tool or JIT tool in that mode.

## TypeScript Entry Point

The TypeScript module must export `run(args, libos)`:

```ts
export async function run(args, libos) {
  const file = await libos.syscall("filesystem.read_text", { path: args.path });
  return { bytes: String(file.content ?? "").length };
}
```

`run` may be synchronous or async. The only libOS access channel is:

```ts
await libos.syscall(name, args)
```

The `libos` object does not expose Python objects, `Runtime`, or
`runtime.tools`.

## RPC Protocol

Python starts a Deno subprocess and writes one NDJSON run frame:

```json
{"type":"run","args":{}}
```

TypeScript may emit syscall frames:

```json
{"type":"syscall","id":"1","name":"filesystem.read_text","args":{"path":"README.md"}}
```

Python responds with final syscall results:

```json
{"type":"syscall_result","id":"1","ok":true,"payload":{}}
{"type":"syscall_result","id":"1","ok":false,"error":"permission denied"}
```

The tool returns:

```json
{"type":"result","value":{}}
```

There is no public pending/retry state for human approval, child wait, or
message wait. Blocking is an implementation detail inside the syscall.

## Syscall Semantics

JIT syscalls enter `LibOSSyscallSession`. They are authorized by:

- caller pid,
- primitive-level capability checks,
- permission policy,
- human approval,
- provider containment,
- audit and event emission.

They do not consult the caller's LLM-facing tool table. This is deliberate:
tool visibility and resource authority are separate.

The current syscall surface covers existing primitive areas:

- filesystem read/write/list/mkdir/delete,
- memory namespace/object read/write/list/append,
- human ask/output/request permission,
- clock now/sleep,
- process cwd/fork/spawn/wait/list/signal/merge/exec/exit/messages,
- shell run,
- JSON-RPC list/inspect/call,
- image load/register,
- checkpoint create/list/inspect/diff/restore/fork/replay,
- Skill discover/inspect/register_path/activate/read_resource/unload.

Trusted startup Runtime Modules can add additional syscall names through the
runtime syscall router. They cannot override built-in syscall names, and the
handler still runs as part of the same `LibOSSyscallSession` under the caller
pid.

## Sandbox Rules

Deno is launched with `--no-prompt` and without read, write, net, env, run, or
ffi host permissions. External effects must go through syscalls.

When `tools.deno_executable` is a bare name such as `deno`, the sandbox resolves
it from absolute safe PATH entries and rejects executables under the runtime
workspace/current root. Absolute executable paths are accepted only when they do
not fall under configured forbidden roots.

Static imports are limited to configured `jsr:` packages. The default allowlist
is a small `@std/*` subset. `npm:`, `node:`, `http:`, `https:`, `file:`,
and dynamic imports are rejected. Static checking is lint, not the security
boundary: it checks that the source exports `run(args, libos)`, blocks dynamic
imports, rejects common runtime code generation forms such as `eval`,
`Function`, `AsyncFunction`, `GeneratorFunction`, `.constructor(...)`, and
`["constructor"](...)`, enforces source/test size limits, and restricts
dependencies to the JSR allowlist. It intentionally does not try to blacklist
every dangerous JavaScript spelling. Runtime safety comes from Deno
no-permission execution, the libOS syscall protocol, primitive Capability
checks, human approval, and resource budgets.

Validation and execution both use subprocess resource budgets when the process
has them. A sandbox backend that cannot accept limits or return subprocess
metrics fails closed for budgeted validation or execution.

If Deno is missing, validation returns a clear error. Python tests marked
`real_deno` run by default when `deno` is installed, skip with a clear reason
when it is missing, and can be intentionally excluded with `--skip-real-deno`.

## Observability Limits

Tool calls, failed tool results, LLM actions/results, and JIT syscall args are
recorded as bounded observable envelopes: preview, SHA-256, byte size, and
truncation status. Sensitive fields such as `content`, `body`, `payload`,
`params`, `question`, `answer`, `source_code`, `tests`, `context`, `metadata`,
`stdout`, and `stderr` are redacted before audit/event persistence.

Full tool results are stored only as Tool Result Object Memory objects and are
subject to a hard serialized payload limit. Larger content should be passed by
file or object reference rather than returned inline from a tool.

## Deferred Lifecycle

`process.exit` and `process.exec` are normal syscalls from TypeScript. Calling
them does not terminate the Deno subprocess mid-protocol. The runtime records
the lifecycle change and applies it after the JIT tool returns its normal
result.
