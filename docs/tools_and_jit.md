# Tools And Deno/TypeScript JIT

LLM-facing tools are stable wrappers over libOS primitives. They provide names,
schemas, validation, and model ergonomics. Primitives enforce authority.

Tool visibility is not resource authority. A process can call only tools in its
process tool table, but filesystem, shell, JSON-RPC, MCP, human, memory, image,
Git, clock, and process effects are still authorized by the primitive path. `ToolPolicy`
contains declaration metadata such as `declared_permissions` and
`declared_confirmation_required`; it is shown in tool specs for humans and UI,
but it does not grant permissions or approve execution.

## Built-In Tools

The current built-in tool surface includes tools for:

- Object Memory: create, append, read, list namespaces, and bridge objects to
  files.
- Filesystem: read/write text, list/create/delete directories, and delete files.
  Tool paths resolve from the process's current working directory and must stay
  contained by the runtime workspace root; they are not implicitly rooted at
  the workspace when the process uses a nested cwd. Object/file bridge tools
  use the same resolution rule. For model ergonomics, a redundant
  workspace-relative cwd prefix on a descendant path is stripped before
  resolution; this maps to the same contained target and grants no additional
  authority. A path equal to the cwd is not rewritten.
- Human I/O: ask questions, output messages, and request permission.
- Capabilities: list/inspect the current process's authority, delegate an
  attenuated delegable capability to a child, relinquish its own revocable
  `allow` capability, or revoke as the issuer or with covering
  `revoke`/`admin` authority. A holder cannot use self-revocation to remove a
  restrictive `ask` or `deny` capability.
- Clock: current time and async sleep through `clock:now`/`clock:sleep` read
  authority.
- Process lifecycle: fork, spawn, wait, list children, signal, merge memory,
  exec, exit, cwd get/set, and process messages.
- Object tasks: `start_object_task`, `get_object_task`, `list_object_tasks`,
  `wait_object_task`, `watch_object_task_owner`, and `cancel_object_task`.
- Context: `compact_process_context` compresses the caller's
  `<llm_context.object_name_prefix>:<pid>` object (default
  `llm_context:<pid>`) through a `context-compressor:v0` child process.
- Shell: argv-only subprocess execution through policy.
- Git: 32 strict tools for bounded inspection, local mutation, managed
  worktrees, immutable patch Objects, existing remotes, and repository-local
  simulated pull requests through `Runtime.git`; no arbitrary Git argv or URL.
- JSON-RPC: list/inspect registered endpoints and call registered methods.
- MCP: list/inspect registered servers, list manifest-allowed tools, and call
  registered MCP tools.
- Image registry: load workspace image packages and commit checkpoints into
  checkpoint-derived images.
- Checkpoint: create, list, inspect, diff, restore, and fork.
- Skills: discover, activate, read bundled resources, and unload.
- JIT: propose, validate, and register Deno/TypeScript tools.
- Lazy tool projection: discover and activate image-authorized tool groups
  without changing the underlying process tool table or resource authority.
- Utility actions such as `echo` and `parse_pytest_log`.

Use `uv run agent-libos tools` to inspect registered tools in a runtime.

## Lazy Tool Groups

An image with the exact boolean `metadata.lazy_tool_groups: true` starts with a small model-tool
projection instead of exposing every authorized schema at once. The initial
core is the intersection of the image's `default_tools` with:

- `discover_tool_groups`, `activate_tool_group`, and `process_exit`;
- `request_permission`, `ask_human`, and `human_output`;
- `read_memory_object`, `create_memory_object`, and
  `append_memory_object`; and
- `get_current_time` and `list_capabilities`.

An image may additionally declare `metadata.initial_tool_groups` as a unique
list of known group names. Their already-authorized tool intersections join the
core at process creation. A malformed, duplicate, unknown, or empty
intersection fails boot; it never adds a tool that is absent from
`default_tools`. String booleans are rejected, and `initial_tool_groups`
without lazy projection is rejected instead of being silently ignored.

This changes model visibility only. The process tool table remains the
image-authorized table, and activating a group does not grant capabilities.
`discover_tool_groups` reports only groups with at least one tool in that
table; its `tool_count` is the number of authorized tools in the group and
`active` means all of those tools are already in the model projection.
`activate_tool_group` adds the authorized intersection to the model projection
and records the before/after schema counts and bytes in audit evidence.

The built-in base, coding, and review images use this projection. Base starts
with the process, context, and clock groups, coding starts with filesystem, and
review starts with the narrower `filesystem_read` group. Their prompts tell the
model to activate the smallest additional group and to distinguish a hidden
schema from missing resource authority. The toolmaker and context-compressor
images already have narrow explicit tool tables and do not use lazy projection.

The built-in groups are:

| Group | Configured tool names |
| --- | --- |
| `filesystem_read` | `read_text_file`, `read_directory`, `create_object_from_file` |
| `filesystem` | `read_text_file`, `write_text_file`, `read_directory`, `write_directory`, `delete_file`, `delete_directory`, `create_object_from_file`, `write_object_to_file`, `get_working_directory`, `set_working_directory` |
| `process` | `list_child_processes`, `spawn_child_process`, `fork_child_process`, `wait_child_process`, `signal_child_process`, `merge_child_memory`, `send_process_message`, `read_process_messages`, `receive_process_messages`, `exec_process` |
| `remote` | `list_jsonrpc_endpoints`, `inspect_jsonrpc_endpoint`, `call_jsonrpc_method`, `list_mcp_servers`, `inspect_mcp_server`, `list_mcp_tools`, `call_mcp_tool` |
| `git` | `git_repository_info`, `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame`, `git_list_refs`, `git_list_remotes`, `git_list_worktrees`, `git_stage`, `git_unstage`, `git_commit`, `git_restore`, `git_branch`, `git_switch`, `git_tag`, `git_integrate`, `git_stash`, `git_reset`, `git_clean`, `git_worktree`, `git_create_patch`, `git_apply_patch`, `git_fetch`, `git_pull`, `git_push`, `git_create_pull_request`, `git_list_pull_requests`, `git_inspect_pull_request`, `git_review_pull_request`, `git_merge_pull_request`, `git_close_pull_request` |
| `checkpoint` | `create_checkpoint`, `list_checkpoints`, `inspect_checkpoint`, `diff_checkpoint`, `fork_checkpoint`, `restore_checkpoint`, `commit_checkpoint_to_image` |
| `memory` | `create_memory_namespace`, `list_memory_namespace`, `create_memory_object`, `append_memory_object`, `read_memory_object`, `create_object_from_file`, `write_object_to_file` |
| `skills` | `discover_skills`, `activate_skill`, `read_skill_resource`, `unload_skill` |
| `object_tasks` | `start_object_task`, `get_object_task`, `list_object_tasks`, `wait_object_task`, `watch_object_task_owner`, `cancel_object_task` |
| `self_evolution` | `load_image_package`, `propose_jit_tool`, `validate_jit_tool`, `register_jit_tool` |
| `authority` | `list_capabilities`, `inspect_capability`, `delegate_capability`, `revoke_capability` |
| `shell` | `run_shell_command`, `parse_pytest_log` |
| `context` | `compact_process_context` |
| `clock` | `sleep` |

A tool may appear in more than one group. Group membership is a fixed model
projection convenience, not an authority declaration.

## Context Compaction

Images may invoke compaction automatically before an LLM request by setting
`planner.context_management.mode: auto_compact` (the default). The attempt is
made only once per pressure episode and policy fingerprint. It counts as
successful only when the tool reports `compacted: true` and advances the
durable LLM-context generation; the current quantum then ends so the next one
re-materializes context. An unavailable, denied, invalid, resource-exhausted,
or failed automatic tool is audited. If it did not change the durable context
generation, the original model request continues without an injected fallback
prompt. If it changed the generation despite reporting failure, the current
quantum ends instead so the next request cannot use the stale materialization.
Human, child, and message waits remain durable and retain the automatic episode
metadata across reopen; after a resumed maintenance action fails, the pending
marker is cleared and the ordinary request is rebuilt from the current context
generation before Provider dispatch.

`compact_process_context` is a model-visible wrapper for bounded long-running
sessions. It reads the caller process' configured
`<llm_context.object_name_prefix>:<pid>` object, spawns a
`context-compressor:v0` child image with only `process_exit` visible, and
replaces the caller context with one `context_compacted` entry plus the recent
verbatim entries requested by `preserve_recent_entries`.

The writeback path is method-neutral: different compressors may produce the
standard compact summary contract, while the LLM context helper records
`compaction_method` and `compaction_metadata` on the `context_compacted` entry
and owns the schema validation, version check, and replacement.

The tool does not grant external resource authority to the compressor. The
compressor child receives only the current chunk, prior stage summary, and
stage goal material needed for summarization; filesystem, shell, memory-write,
JSON-RPC, MCP, human, Skill, checkpoint, and process-control access remain
absent unless separately granted by normal primitives. The wrapper is visible
to the model, but Object Memory and Process primitives still enforce reads,
writes, child creation, waiting, resource budgets, audit, and lifecycle.
For multi-chunk work, each stage produces one rolling cumulative summary. The
next stage receives only that summary, and the durable job replaces it rather
than retaining and re-merging cumulative intermediates, so summary state does
not duplicate earlier stages.

Compaction is fail-closed. If the compressor fails or is killed, returns an
invalid or empty schema, the source context version changes before final
writeback, resource limits are exceeded, or the durable pending state cannot be
resumed, the tool returns failure and leaves the original materialized context
unchanged. Pending child waits store the minimum resume state in
`llm_pending_actions`; after runtime reopen the compressor child goal can be
reconstructed and the final compacted context is recreated under the same
configured context-object name when the old runtime-only payload is no longer
materializable.

The same durable row protects LLM-selected human, child, and process-message
waits. Every wait generation has a unique resume token. A ready waiter must
atomically claim `pending -> resuming` for that exact token before dispatch; a
second executor sees the claimed state and cannot repeat the primitive. If the
resumed action blocks again, it writes a new token/generation, preventing a
stale completion from clearing the new wait. Reopening a store with an action
already in `resuming` fails the process and records
`llm.pending_action_resume_interrupted`; it never replays an action after an
unknown crash window. The same fail-closed transition happens immediately when
dispatch, durable output persistence, or completion raises after the claim, so
a direct `run_process_once` caller cannot spin a still-runnable process around a
non-replayable action.

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
reference by default; it is not itself an object capability.

`start_object_task` may set `grant_result_to_notify: true` when the notification
recipient differs from the creator. In that mode, result publication requires
the creator to authorize `grant` on the exact result object and the recipient's
Task Authority data-flow domain to accept the result labels. The runtime
reserves any finite-use grant, then commits the recipient's read/materialize/link
handle, terminal success row, delivered notification, and reservation
consumption in one transaction. If delivery fails, that transaction rolls back,
the exact still-live reservation is restored, and terminal success is recorded
without the recipient handle. Concurrent revoke or disable still wins and
cannot be undone by cleanup. If the creator lacks grant authority, the task
fails and its uncommitted result is discarded.

When `owner_watch` is enabled, Object Memory `updated` and outgoing `linked`
events on the owner object are delivered to the runner process as ordinary
process messages, on `object-task-owner` by default. The notice is produced by
the Object Memory primitive after the change is committed and audited, includes
only ids/version/link metadata, and may resume a task that is blocked in
`receive_process_messages`; it does not run the LLM scheduler. Tools that block
after non-trivial side effects are not automatically replayed on owner-watch
messages unless they are explicitly known to be safe.
Ordinary process messages delivered to a waiting runner use the same
message-wait resume path. Child-process termination can also resume a runner
blocked in `wait_child_process`. Auto-resume is limited to tools with explicit
safe replay semantics, currently `receive_process_messages` and
`wait_child_process`. A `WAITING_HUMAN` task is resumed separately through its
exact stored human-request id after the request settles; a rejected
`request_permission` resumes so the tool can return the denial, while rejection
of another human-waiting tool fails the task.

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

`SyncAgentTool` deliberately sets `enforce_timeout = False`: Python worker
threads cannot be killed safely, so `ToolPolicy.timeout_s` does not interrupt
blocking synchronous tool code. Every blocking operation used by a sync tool
must therefore terminate through a primitive or provider hard deadline; do not
rely on the model-facing Tool policy as containment. An async `BaseAgentTool`
with the default `enforce_timeout = True` is instead executed through
`asyncio.wait_for` when `ToolPolicy.timeout_s` is not `None`.

Do not put direct filesystem, terminal, network, shell, browser, database, or
credential access inside a model-facing tool unless that code is itself the
libOS primitive or a sandbox backend.

## JIT Tool Lifecycle

Agent-authored JIT tools use TypeScript and run under Deno. Python JIT tools
are intentionally not supported.

The manual lifecycle is:

1. `propose_jit_tool`: validate the spec and size bounds, then store candidate
   metadata, TypeScript source, and tests. The model-facing tool deliberately
   has no `requested_capabilities` argument. Host-side programmatic proposal
   APIs can preserve such declarations for inspection, but validation rejects
   every non-empty declaration because Deno JIT tools cannot acquire external
   authority directly.
2. `validate_jit_tool`: run static source checks, schema/source/test validation,
   and configured tests under the sandbox backend. Current JIT source must be
   import-free.
3. `register_jit_tool`: add the validated tool only to the registering process
   tool table.

The three transitions publish different artifacts:

- Proposal atomically inserts the durable candidate row (including source and
  tests), creates its immutable Object Memory descriptor, and records the
  proposal audit entry. It creates no tool alias or executable handle.
- Validation runs the sandbox work, then atomically updates the owned
  candidate to `validated` or `rejected` and records bounded validation audit
  evidence. It creates no tool alias or executable handle. Registering a
  merely proposed candidate invokes this validation first in its own
  transition.
- Registration atomically inserts the ephemeral durable Tool row, marks the
  candidate `registered`, updates both process-local tool tables, publishes the
  in-memory executable source/handle, and records the registration audit entry.

JIT registration and resolver calls share the runtime registry lifecycle lock;
the source/handle is installed before the durable alias commits and removed
again if commit or observability fails, so no resolver observes only one side.
Snapshot-based process exec holds that same lock through its terminal
publication, which serializes concurrent process-local candidate, Tool, and
Skill mutations after either commit or compensation. If compensation succeeds
but its terminal publication transaction fails, a durable applied marker lets
startup finish the terminal record without replaying the older exec snapshot.
If compensation fails before that marker is durable, ImageBoot fences the
entire runtime in `CLOSE_FAILED`, so Tool, Skill, memory, process, capability,
and all other public mutation boundaries reject admission without persisting
new state. The fence also revokes older admission epochs; a Tool or Skill
mutation that was already waiting for the shared registry barrier revalidates
after it acquires the lock and is rejected before publication. Ordinary
`close()`/`shutdown()` is also fail-closed while this recovery-required fence is
active. After capturing diagnostics, the owner must explicitly call
`release_recovery_diagnostics()` (or its async form) to release the store
without normal finalizers; only then can a fresh open perform startup recovery
and restore normal mutation admission.
Manual validation failures remain inspectable as rejected candidates. When
composite Skill activation or image-package compensation succeeds, it discards
candidates created by the enclosing operation, including their Object Memory
descriptors, so unpublished source and aliases do not accumulate as residue.
If compensation itself fails, the exact artifacts remain bound to the durable
publication under the recovery fence until reopen reconciliation converges or
marks the publication manual.

Checkpoint image installation also treats a committed JIT candidate and its
registered Tool as two distinct publication artifacts. Their durable rows,
executable handle, and exact candidate-then-Tool receipts commit in one unit of
work. Compensation consumes those receipt identities directly rather than
reverse-mapping a Tool id through candidate metadata.

Registered JIT tools are process-local but persistent: when a runtime reopens an
existing runtime store, it reloads executable TypeScript sources only for JIT
tool ids still referenced by a process tool table. Stale ephemeral tool
references with no recoverable registered source are removed from the process
tool table fail-closed instead of being shown to the model as broken tools.
Host-side process tool configuration applies the same durable candidate-owner
check before replacing either tool table: a loaded handle is not authority to
bind another process's JIT. Checkpoint restore also preflights every registered
JIT candidate owned by the restored scope that the restore would remove,
including owner-unbound tools, and refuses the restore if any such identity is
referenced outside that scope. This prevents a scoped restore from deleting the
owner metadata that a later startup would otherwise use to prune a foreign
binding.
Startup rehydration requires the opaque recovery lease before any durable read,
keyset-pages the normalized durable ephemeral-binding projection directly by
`(pid, tool_name)`, and looks up only each page's referenced JIT identities.
Process and Tool writes maintain an exact eligibility bit in this projection in
the same transaction. Its binary-collated partial covering index excludes
static/package bindings before traversal, so a sparse eligible backlog cannot
hide an unbounded cross-table residual scan inside one nominal page. The scan
does not decode unrelated process control state, so wait/outcome corruption
remains fail-closed on explicit process access without preventing startup. Each
page performs one typed bulk artifact lookup; its temporary lookup and
diagnostic buffers are page-bounded, and the returned summary and audit record
contain exact totals plus bounded samples. The in-memory registry itself
necessarily scales with the active JIT set. A single process's binding fanout is
traversed across bounded pages rather than materialized as one record.

Checkpoint fork never shares an ephemeral registration identity with the
source process. It allocates new tool and candidate ids, rewrites the forked
tool table, candidate descriptors, Object payloads, and loaded-Skill JIT maps,
and prepares executable handles before the fork process rows are published.
Fork failure discards those unpublished handles. The captured Skill package
snapshot remains process-local; fork does not replace the host's current global
Skill or Image registry.

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

JIT code cannot declare authoritative labels or Sink trust. Syscalls that carry
data out of the runtime inherit the caller's materialized source context and
pass through the same SDK data-flow gate as their Python primitive. Deno itself
still has no direct network, filesystem, environment, or subprocess authority;
marking a Sink trusted changes only the mediated payload decision.

Object and file reads append their trusted labels and any versioned Object
source refs to the active JIT call. Later JIT syscalls, created or appended
Objects, and the final tool-result Object inherit the full aggregate even when
a Host-classified file has no Object source ref, so a read-then-write sequence
cannot reset sensitivity to the default.

## RPC Protocol

Python starts the dedicated supervisor. Once host-lifetime containment is live,
the supervisor writes the first NDJSON frame before spawning Deno:

```json
{"type":"supervisor_ready","version":1}
```

Python rejects a missing, malformed, or unexpected readiness frame. It then
writes one run frame with a fresh per-execution proof nonce:

```json
{"type":"run","args":{},"provider_error_proof":"<host-generated nonce>"}
```

TypeScript may emit syscall frames:

```json
{"type":"syscall","id":"1","name":"filesystem.read_text","args":{"path":"README.md"}}
```

Python responds with final syscall results:

```json
{"type":"syscall_result","id":"1","ok":true,"payload":{}}
{"type":"syscall_result","id":"1","ok":false,"error":"provider failed","error_type":"RuntimeError","code":"provider_error","correlation_id":"..."}
```

Handler exceptions include `error_type`; `code` and `correlation_id` are
present only when the Host exception has a public provider-error envelope.
Protocol-level failures such as an unavailable handler or an exceeded RPC-call
limit may contain only `error`.

The runner returns either a result frame:

```json
{"type":"result","value":{}}
```

or an error frame:

```json
{"type":"error","message":"tool failed","stack":"..."}
```

When an uncaught error is the exact error created from a failed Host syscall,
the runner also returns its `code`, `error_type`, `correlation_id`, and the
matching `provider_error_proof`. The proof is held by the trusted runner wrapper
and is not passed to candidate `run(args, libos)`. Python accepts those fields
as a public `ProviderHostError` envelope only when the proof matches in constant
time and the envelope validates. Candidate-authored fields, a fabricated
proof, or a different thrown error are treated as an ordinary `SandboxError`;
they cannot impersonate a Host provider failure.

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
- capability list/inspect/request permission/delegate/revoke,
- clock now/sleep,
- process cwd/fork/spawn/wait/list/signal/merge/exec/exit/messages,
- shell run,
- JSON-RPC list/inspect/call,
- MCP list/inspect/tools/call,
- image list/inspect/load package/commit checkpoint,
- checkpoint create/list/inspect/diff/restore/fork/replay,
- Skill discover/inspect/register_path/activate/read_resource/unload.

Trusted startup Runtime Modules can add additional syscall names through the
runtime syscall router. They cannot override built-in syscall names, and the
handler still runs as part of the same `LibOSSyscallSession` under the caller
pid.

Process list, wait, and signal Tool/JIT results include canonical tagged
`wait_state` and `outcome` objects plus `state_generation`. The Host GUI signal
projection uses the same serializer. `status_message`/`message` remains a
temporary compatibility display field; TypeScript code must use the tagged
fields for wait identity, terminal result/reason Object ids, and outcome codes.

## Sandbox Rules

Deno is launched with `--no-prompt` and without read, write, net, env, run, or
ffi host permissions. Runtime JIT execution also uses Deno's cached-only mode,
so a tool call cannot implicitly fetch remote modules. External effects must go
through syscalls.

When `tools.deno_executable` is a bare name such as `deno`, the sandbox resolves
it from absolute safe PATH entries and rejects executables under the runtime
workspace/current root. Absolute executable paths are accepted only when they do
not fall under configured forbidden roots.

All static imports and re-exports from module specifiers are rejected,
including pinned `jsr:`, `npm:`, `node:`, `http:`, `https:`, and `file:`
specifiers. TypeScript `import name = require("...")`, triple-slash dependency
references, `amd-dependency`, and `@deno-types` directives are rejected as
well. Dynamic `import()` is also rejected. The configured
`tools.deno_jsr_allowlist` is retained in sandbox configuration and validation
metadata, but it is not currently an exception to the import-free policy and
must not be treated as permission to import. Runtime execution remains
cached-only.

Static checking is lint, not the security boundary: it checks that the source
exports `run(args, libos)`, rejects imports, blocks common runtime code
generation forms such as `eval`, `Function`, `AsyncFunction`,
`GeneratorFunction`, and member `constructor` access such as `.constructor` or
`["constructor"]`, and enforces source/test size limits. It intentionally does
not try to blacklist every dangerous JavaScript spelling. Runtime safety comes
from Deno no-permission cached-only execution, the libOS syscall protocol,
primitive Capability checks, human approval, and resource budgets.

Validation invokes the Deno compiler at least once. Each configured test
compiles and runs the candidate in a fresh contained process. When the test
list is empty, validation instead runs a contained, non-executing `deno check`
with a runtime-generated non-strict compiler configuration and remote/npm
resolution disabled. This rejects syntax and compiler errors that the lexical
static checker cannot see; an empty test list still provides no behavioral
evidence and should not be treated as representative test coverage.

Validation and execution both use subprocess resource budgets when the process
has them. A sandbox backend that cannot accept limits or return subprocess
metrics fails closed for budgeted validation or execution.
Cancelling a Deno execution kills its isolated process group (and any discovered
descendants) and waits for the syscall-serving and resource-monitor workers to
settle before returning. On POSIX, if the group signal is denied while the
supervisor is active, cleanup kills the discovered descendants and direct
supervisor and accepts that fallback only after both have been observed to
terminate; unavailable tree inspection or surviving processes produce a
sandbox error. Cleanup of an already-reaped supervisor is an idempotent no-op,
so a recycled process-group ID is not signalled. Deno is started only after a
dedicated supervisor has established host-lifetime containment: POSIX uses an
inherited death pipe and an isolated process group, while Windows uses a
`KILL_ON_JOB_CLOSE` Job Object. If the libOS host is hard-killed, the supervisor
or operating system terminates the untrusted process tree; if that containment
cannot be established, JIT execution fails closed before Deno is released.

If Deno is missing, validation returns a clear error. Python tests marked
`real_deno` run by default when `deno` is installed, skip with a clear reason
when it is missing, and can be intentionally excluded with `--skip-real-deno`.

## Observability Limits

Tool calls, failed tool results, LLM actions/results, and JIT syscall args are
recorded as bounded observable envelopes: preview, SHA-256, byte size, and
truncation status. Sensitive fields such as `content`, `body`, `payload`,
`params`, `question`, `answer`, `source_code`, `tests`, `context`, `metadata`,
`stdout`, and `stderr` are redacted before audit/event persistence.
JIT validation errors, validation logs, and input/output schema failure details
are persisted through the same bounded/redacted envelope; the direct tool call
result can still return the original error to the caller.

The canonical successful result carrier is a Tool Result Object Memory object,
subject to a hard serialized payload limit. It is not necessarily the only
durable copy: eligible OpenAI Responses continuation chains persist the exact
action-result envelope (including its payload) in `llm_tool_outputs`, and with
full LLM I/O persistence later call messages may retain the rendered result.
Audit and event envelopes remain bounded/redacted as described above. A failed
tool result is also stored as a Tool Result Object whenever its trusted data-flow
context differs from the default context; this labeled carrier prevents error
text derived from Object reads from becoming an untracked input to the next
LLM action. Sync worker threads and timeout-managed async tasks return their
post-call data-flow context to ToolBroker on both success and failure. If a
labeled failure is too large, the carrier keeps its labels and source refs but
omits the error body. Larger content should be passed by file or object
reference rather than returned inline from a tool.

## Deferred Lifecycle

`process.exit` and `process.exec` are normal syscalls from TypeScript. Calling
them does not terminate the Deno subprocess mid-protocol. The runtime records
the lifecycle change and applies it after the JIT tool returns its normal
result.
