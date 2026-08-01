# Tools And Deno/TypeScript JIT

LLM-facing tools are stable wrappers over libOS primitives. They provide names,
schemas, validation, and model ergonomics. Primitives enforce authority.

Tool visibility is not resource authority. A broker call can resolve only a
binding in the complete process tool table, while an LLM action must also be in
the narrower model tool projection. Filesystem, shell, JSON-RPC, MCP, human,
memory, image, Git, clock, and process effects are still authorized by the
primitive path. `ToolPolicy`
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
  use the same resolution rule. A leading segment matching the cwd is not
  treated as redundant: for a process whose cwd is `pkg`, `pkg/module.py`
  resolves to `pkg/pkg/module.py`. Callers that intend `pkg/module.py` must
  pass `module.py` from that cwd.
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
  registered MCP tools. Modern `server/discover` is intentionally a Host
  SDK/CLI/GUI operation rather than another model tool or syscall.
- Image registry: load workspace image packages and commit checkpoints into
  checkpoint-derived images.
- Checkpoint: create, list, inspect, diff, restore, and fork.
- Skills: discover, activate, read bundled resources, and unload.
- JIT: propose, validate, and register Deno/TypeScript tools.
- Tool Skills: discover intent-focused guidance on demand and project only the
  image-authorized tool schemas needed for the current task.
- Utility actions such as `echo` and `parse_pytest_log`.

Use `uv run agent-libos tools` to inspect registered tools in a runtime.

## On-Demand Tool Skills

An image with `metadata.tool_projection: skills` starts with a small
model-facing projection instead of exposing every image tool schema at once.
The fixed bootstrap requires the complete image-authorized set of
`discover_skills`, `activate_skill`, `read_skill_resource`, `unload_skill`, and
`process_exit`; a Skills-projection image missing any member is rejected. The
image's full process tool table is unchanged.

Fresh shipped images contain neither Skill catalog metadata nor Skill bodies in
the prompt. `discover_skills` searches every source visible under current
catalog authority using one common, `text`/`limit`-bounded result schema.
Concrete query terms are matched independently against id, name, and
description metadata and results are relevance-ranked. A one-term query must
match that term; a longer query requires at least two matching terms, allowing
one task intent to surface separate narrowly owned Skills without admitting a
result on one generic word alone. `next_step` tells the model to activate an
inactive plausible exact id, use a current loaded snapshot, or refine a
zero-result query. `active` is true only when loaded and catalog package hashes
match. `activate_skill` passes the discovered hash as
`expected_package_sha256`, then atomically loads the exact selected instructions
and tool bindings; stale discovery hashes fail before publication.
Model-visible discovery, activation, loaded-prompt, and unload contracts never
identify a Skill as built-in or registered.

Host enforcement still treats applicability as all-or-nothing: an immutable
packaged Skill is hidden and cannot activate
unless every declared tool already exists in the image-authorized process tool
table. Activation never resolves a missing binding from the global registry,
adds to the full table, grants Capability, or approves a primitive effect. It
records its trust provenance only in Host-private state and audit evidence;
filesystem, shell, Git, remote, checkpoint, human, and other effects still pass
through normal authority and approval boundaries. Cross-process operations
still require `process:<pid>` `admin`.

The catalog gives each of the 99 built-in static tools one intent-focused owner
across 26 Skills, with at most nine tools per Skill. The owner is available
through `SkillManager.builtin_skill_for_tool()`. This keeps automatic routing
deterministic and prevents overlapping Skills from making unload provenance
ambiguous:

| Built-in Skill | Guidance scope |
| --- | --- |
| `agent-libos-skill-navigation` | Discover, activate, inspect resources from, and unload Skills |
| `agent-libos-authority-basics` | Inspect authority and request missing permission |
| `agent-libos-capability-delegation` | Delegate or revoke Capability |
| `agent-libos-human-collaboration` | Ask a human question or emit user-facing output |
| `agent-libos-runtime-session` | Exit, compact context, read time, or sleep |
| `agent-libos-workspace-navigation` | Inspect files/directories and manage cwd |
| `agent-libos-workspace-editing` | Create, write, or delete workspace paths |
| `agent-libos-command-execution` | Run policy-governed argv-only commands |
| `agent-libos-test-log-analysis` | Parse pytest output |
| `agent-libos-tool-protocol-diagnostics` | Exercise the tool protocol with `echo` |
| `agent-libos-object-memory` | Manage Object Memory namespaces and objects |
| `agent-libos-object-file-transfer` | Transfer data between Objects and files |
| `agent-libos-object-tasks` | Start, observe, wait for, or cancel Object tasks |
| `agent-libos-child-processes` | Create, coordinate, and message child processes |
| `agent-libos-checkpoints` | Create, inspect, fork, restore, or diff checkpoints |
| `agent-libos-agent-images` | Load images, commit checkpoint-derived images, or exec |
| `agent-libos-jit-tool-authoring` | Propose, validate, and register JIT tools |
| `agent-libos-jsonrpc` | Inspect and call registered JSON-RPC endpoints |
| `agent-libos-mcp` | Inspect registered MCP servers and call allowed tools |
| `agent-libos-git-inspection` | Inspect repository state and history |
| `agent-libos-git-change-recording` | Stage, unstage, and commit changes |
| `agent-libos-git-branches-worktrees` | Manage branches, tags, switches, and worktrees |
| `agent-libos-git-integration-recovery` | Integrate or recover local Git state |
| `agent-libos-git-patch-objects` | Create or apply immutable patch Objects |
| `agent-libos-git-remotes` | Fetch, pull, or push configured remotes |
| `agent-libos-git-pull-requests` | Manage repository-local simulated pull requests |

Base, coding, review, and toolmaker each start with the same five Skill
lifecycle/bootstrap schemas and no loaded Skill. They discover and activate
navigation, authority, human collaboration, Object Memory, workspace, JIT, or
other domain guidance only when the task requires it. Context-compressor
remains a single-tool image.

The former `lazy_tool_groups`/`initial_tool_groups` image metadata and
`discover_tool_groups`/`activate_tool_group` tools are removed rather than
aliased. The storage migration converts old custom lazy images to full
projection after replacing the obsolete lifecycle tools, so an upgrade does
not silently hide an authorized tool. Unknown or malformed legacy group data
aborts migration instead of guessing; old tool calls fail as unknown tools.

Run the one-time migration while the runtime is stopped. It is a content-only,
atomic offline migration and is never run during startup. The default command
performs the complete validation and rewrite in a transaction, prints the
report, and rolls the transaction back:

```bash
agent-libos-migrate-tool-groups /path/to/runtime.sqlite
```

Source checkouts may equivalently run
`uv run python scripts/migrate_tool_groups_to_skills.py`. Pass `--config` when
the store uses a non-default config overlay; otherwise the command loads the
project-root `config.yaml`, matching the main CLI. After reviewing that report,
repeat with `--apply` to commit it. A migrated
custom lazy image intentionally omits `tool_projection` and therefore falls
back to full schema projection; the report includes this warning. Immutable
checkpoint-commit artifacts are replaced by new content-addressed artifact
rows while old rows remain intact. Raw image-package files likewise remain
unchanged and are called out in the report.

## Context Compaction

Images may invoke compaction automatically before an LLM request by setting
`planner.context_management.mode: auto_compact` (the default policy choice),
but only for a process with explicitly enabled persistent context and
`context:maintenance/execute` authority. Default source-only execution records
the pressure decision as not authorized and neither injects context nor invokes
the tool. An authorized attempt is made only once per pressure episode and
policy fingerprint. It counts as
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
the selected Image's complete process tool table controls callability, and its
model projection is not consulted or widened. Primitives still enforce
capabilities, approval, budgets, events, and audit.

Successful workflow calls append the tool result object to the workflow
process view and exit the process with that result. Failed calls mark the
process failed. Blocking human, child-process, or process-message waits are
returned as explicit waiting results so the caller can resume through the
normal runtime mechanisms. If the tool itself performs `process.exit` or
`process.exec`, the workflow runner leaves that lifecycle decision intact.

## Object Task Entry Point

Object tasks let an AgentObject hold asynchronous tool work. `start_object_task`
creates a host-managed runner child process, narrows that runner's process tool
table to the requested creator-bound tool, and calls it through ToolBroker. The
runner is excluded from the LLM scheduler even if a message wakes it back to a
`RUNNABLE` process status. It does not inherit creator-held external authority
unless the creator explicitly delegates it. Like an ordinary child it receives
its own goal/View, namespace, self-checkpoint, and image bootstrap grants; an
image package may also receive package-workspace grants, and ObjectTask adds
task-local owner read/materialization authority. Those runner-local grants are
not a transfer of the creator's ambient authority.

Start validates the ObjectTask envelope, owner/authority/capacity, and that the
named tool is visible before returning the durable queued task. The target
tool's own argument-schema validation happens later in the asynchronous runner.
Thus a returned `queued` task does not prove that `args` are valid; malformed
target arguments produce a terminal task failure that must be observed with
get/wait.

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
`watch_object_task_owner` accepts `enabled=false` to disable subsequent owner
notices for an active task; disabling a watch does not retract notices already
published, cancel the runner, or alter target-tool authority.

## Writing Python Tools

Python tools should not directly access host resources. Use this pattern:

1. Define a Pydantic input schema and optional output schema.
2. Subclass `SyncAgentTool` for blocking local code or `BaseAgentTool` for
   async code.
3. Keep validation and model-facing ergonomics in the tool.
4. Call `ctx.runtime.<primitive>` for process, memory, filesystem, human,
   clock, shell, image, Skill, checkpoint, or other libOS operations.
5. For an LLM-selected call, treat `ctx.metadata` values
   `llm_transcript_output_key`, `llm_tool_call_id`, and `llm_tool_name` as
   Host-captured optional identity. A protected provider protocol may derive a
   non-secret explicit idempotency key from them when that protocol defines
   native-call retry identity; never infer deduplication from matching args.
6. Let primitives enforce capability checks, containment, audit, events, human
   approval, checkpoint semantics, and policy hooks.
7. Register the tool through the runtime composition root or ToolBroker-backed
   registry.

The LLM identity metadata remains stable across supported Human, child, and
message wait/resume paths. It is absent for Host-generated maintenance and
other non-provider actions, grants no authority, and contains no model-supplied
payload.

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

JIT input and output schemas use a bounded, reference-free, regex-free subset
of JSON Schema 2020-12. Common object, array, scalar, composition, and numeric
keywords are supported, including `type`, `properties`, `required`,
`additionalProperties`, `items`, `prefixItems`, `enum`, `const`, `allOf`,
`anyOf`, `oneOf`, `not`, length/item/property limits, and numeric bounds.
Schemas are limited to 32 levels and 1,024 nodes and may contain only finite
JSON numbers. `$ref`, `$dynamicRef`, `$recursiveRef`, `pattern`, and
`patternProperties` are forbidden; `uniqueItems` and other unlisted keywords
are unsupported because their Host validation cost is not bounded by this
dialect. Existing tools that use references should inline a bounded schema;
tools that use regexes should replace them with `enum`, structural fields, or
fixed validation inside the sandboxed implementation. Proposal rejects an
unsupported schema before persisting or executing candidate code.

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
argument shapes. Prompt projection also omits the registered package's JIT
catalog and JIT source entries from its resource summary. This is not a resource
ACL: if trusted instructions or earlier authorized evidence supplied an exact
loaded-snapshot resource path, `read_skill_resource` can still read it. It does
not list paths, so callers must not probe guessed filenames.

The name `run_jit_tool` is hard-reserved when an image uses multiplexed
exposure: it cannot be a real default tool, manually proposed JIT tool, package
JIT tool, or image-package JIT tool in that mode. Direct-mode validation does
not reserve the spelling, but using it as a real JIT name is not portable: a
later switch to multiplexed exposure will reject the collision.

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
{"type":"syscall_result","id":"1","ok":false,"error":"syscall_error: CapabilityDenied (correlation_id=corr_...)","error_type":"CapabilityDenied","code":"syscall_error","correlation_id":"corr_..."}
```

Every exception raised by a Python syscall handler is converted to this
text-free public envelope: `error` contains only the Host-selected code and
exception-class identifier plus the Host correlation id, and `error_type`,
`code`, and `correlation_id` are always present. Provider- or
extension-authored exception text never enters the frame. Protocol-level
failures that occur before handler invocation, such as an unavailable handler
or an exceeded RPC-call limit, may contain only `error`.

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

The authoritative built-in name/alias inventory is generated from
[`BUILTIN_SYSCALL_DESCRIPTORS`](../agent_libos/runtime/syscall_descriptors.py),
and its uniqueness/completeness ratchet lives in
[`test_syscall_descriptors.py`](../tests/unit/test_syscall_descriptors.py).
Those descriptors intentionally define only canonical spelling, stable aliases,
and the routed handler; the documentation does not copy a count that would
become stale when a route changes. Aliases enter the identical canonical
handler and do not relax validation or authority.

Before a common-contract built-in handler runs, the Host validates every field
that handler consumes against its canonical JSON type and any configured hard
bound. Boolean controls accept only JSON booleans, integer limits reject
booleans and strings, identifiers and paths accept only strings, and structured
arguments keep their declared object/list shape. Duration arguments accept
finite JSON integers or floats within the applicable Host limit. An invalid
call is still resource-charged and request-audited, but it cannot reach a
primitive, publish deferred lifecycle/session state, acknowledge a message,
select destructive or replacement behavior, or emit a successful
`syscall.result`. Stable aliases share the same canonical argument contract.
`capability.delegate` keeps its capability-domain validation rather than the
common contract. Runtime-Module syscalls remain responsible for their own
trusted module-defined argument contracts.

This inventory is Host-side route discovery, not a parameter-schema API.
Neither it nor `libos` supplies runtime introspection for arguments, results,
required Capability, idempotency, or blocking. Those properties are
handler/domain-specific and must come from a trusted Image, loaded Skill, Host
contract, or the domain documentation. A model must not infer a syscall
contract from a similarly named model tool. In particular, a successful route
lookup proves none of the following: that its arguments are valid, that its
result matches another tool's output, that authority is present, or that it
will return without a Human/process/message wait. When an exact published
contract is unavailable, keep the JIT pure, use the governed model-facing tool,
or stop for Host guidance rather than probing.

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
Candidate-validation diagnostics and logs returned by `validate_jit_tool` are
separate from runtime invocation validation; their durable copies use the same
bounded/redacted envelope. At runtime, an input-schema mismatch returns the
fixed generic `InputValidationError` projection before candidate code or a
syscall runs. An output-schema mismatch is surfaced after candidate code (and
therefore possibly after syscalls) as a text-free execution-error envelope.
Neither runtime path returns the original `jsonschema` diagnostic to the model;
only a bounded internal type/length/digest observation is retained, and an
output rejection does not roll back preceding effects.

The canonical successful result carrier is a Tool Result Object Memory object,
subject to a hard serialized payload limit. It is not necessarily the only
durable copy: an `image_only` transcript persists its bounded model-facing
result projection and paired call metadata in `llm_tool_outputs`, and with full
LLM I/O persistence later call messages may retain the rendered result. The
current full-snapshot AgentProcess executor does not create provider-side
Responses-continuation output rows.
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

`process.exit` and `process.exec` are lifecycle syscalls from TypeScript.
Calling them does not terminate the Deno subprocess mid-protocol. The runtime
records the lifecycle change and applies it after the JIT tool returns its
normal result. Direct `process.exit` is unavailable when the active image (or a
deferred exec target) uses cumulative completion review; call the built-in
`process_exit` tool separately so its review/evidence gate remains authoritative.
When one JIT call requests both lifecycle changes, both images are checked
before mutation. A standalone authorized exec adopts the target image's contract
for later calls; it does not permanently carry the source image's gate.
