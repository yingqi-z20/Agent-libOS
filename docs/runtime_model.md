# Runtime Model

Agent libOS models work as `AgentProcess` instances. A process has identity,
status, a goal object, a memory view, a process-local working directory, a tool
table, loaded Skills, capabilities, children, message queue state, an
`llm_profile_id`, and resource budgets.

The paper frames this process model as the substrate for self-evolving agents:
a process can change visible tools, activate Skills, register process-local JIT
tools, register or exec AgentImages, fork children, and fork from checkpoints,
while resource authority remains separate in Capability.

## Process Lifecycle

The current lifecycle includes:

- `created`: process row exists but has not started running.
- `runnable`: scheduler may run the process.
- `running`: a quantum is currently executing.
- `waiting_human`: the process is blocked on a human question or approval.
- `waiting_event`: the process is blocked on a child, message, or event.
- `waiting_tool`: reserved waiting state for tool-level blocking.
- `paused` / `suspended`: the process is not selected for normal execution.
- `exited`: completed successfully.
- `failed`: completed with failure.
- `killed`: terminated by signal or runtime decision.

Terminal statuses are `exited`, `failed`, and `killed`.

## Images And Tool Tables

An `AgentImage` defines the default process prompt, tool table, default Skills,
prompt mode, context policy, safety profile, declared required capabilities,
an optional default LLM profile, and optional boot metadata. Fresh images boot
from their manifest.
Checkpoint-commit images boot from an immutable internal runtime artifact
derived from one checkpoint root process. Image-package images boot from an
immutable directory-package artifact created from `IMAGE.yaml`, `prompt.md`,
optional `tools/`, optional `resources/`, and optional `workspace/`.

`prompt_mode` controls prompt composition. `image_only` uses the image prompt as
the system prompt and gives the model only materialized task context; this is
the default for custom images and image packages. `minimal_runtime` adds a
short factual runtime note and state sections. `libos_default` preserves the
native Agent libOS planner envelope and fallback JSON instructions used by the
built-in images.

Root process spawn may use image `required_capabilities` as a bootstrap
declaration for ordinary fresh images. `exec_process`, checkpoint-commit image
boot, and image-package boot never grant those declarations automatically.

At process creation time, the runtime resolves only the image's explicit
`default_tools` into the process tool table. No lifecycle, Object Memory, or
other builtin tool is implicitly added. A process can call only tools in that
table, but visible tools still fail at primitive use if resource authority is
missing. If an image wants LLM-facing `process_exit`, Object Memory, filesystem,
shell, or other builtin access, it must list that tool explicitly. Internal
runtime paths such as JIT syscalls may still call primitives directly through
their syscall session without exposing the corresponding builtin tool to the
model.

LLM selection is host-controlled and process-local. A process stores only an
`llm_profile_id`; the host Runtime resolves that id to a configured
OpenAI-compatible profile at LLM-call time. Root spawn uses an explicit host
profile, then the image default, then `config.llm.default_profile_id`. Fork and
fresh child creation inherit the parent profile by default. Exec keeps the
current profile unless the host explicitly overrides it. Model-facing process
tools do not expose LLM profile switching in v1. Only the configured default
profile inherits legacy `OPENAI_*` provider and model environment variables;
other named profiles require explicit host profile fields for non-default
routing.

`jit_tool_exposure` controls how JIT tools appear to the LLM. `direct` exposes
each visible JIT as its own OpenAI tool. `multiplexed` exposes one stable
`run_jit_tool` protocol tool and maps it back to the real process-local JIT
before execution. Multiplexed mode hides individual JIT names from runtime
tool sections and event context; image prompts remain responsible for listing
any JIT catalog the model should know.

A checkpoint-commit image remaps baked Object Memory, process-local JIT tools,
loaded Skill package snapshots, and cwd into the new process. It does not package or
restore filesystem, shell, JSON-RPC endpoints, global Skill trust, human,
network, or provider side effects.

An image-package boot materializes the package `workspace/` seed into a private
per-process directory under `agent_outputs/image_workspaces/`, sets the process
cwd from the package manifest, and grants only the manifest-declared
`workspace.grants` for that private copy. Package JIT tools live under
`tools/jit-tools.json` and `tools/scripts/*.ts`; they are registered as
process-local ephemeral tools and are not copied into the workspace. Package
artifacts persist only declared package content: `IMAGE.yaml`, the referenced
prompt, declared `workspace/` content, referenced `tools/` JIT files, and
`resources/`. Cache, VCS, likely secret, and platform-unsafe paths are rejected.

## Working Directory

Each process has its own workspace-relative working directory. Relative
filesystem paths and shell subprocess cwd resolve from that process cwd. The
runtime host process does not `chdir` into launched workspaces.

The CLI command:

```bash
uv run agent-libos --db .agent_libos.sqlite cd <pid> src
```

updates one process working directory and leaves other processes unchanged.

## Object-Bound PTY Sessions

The trusted `modules/pty` runtime module can add an interactive PTY surface.
`pty_create` starts the host PTY through the shell primitive's authorization
path and returns a mutable Object Memory `EXTERNAL_REF` object id. The object
payload records descriptive metadata such as argv, cwd, backend, dimensions,
and creation time, but authorization for later interaction comes only from the
current Object capability graph and the in-memory PTY registry.

`pty_read` requires object `read`; `pty_write` and `pty_resize` require object
`write`; `pty_close` requires object `delete`. Closing the session releases the
object and revokes related object capabilities. If the object is released by a
lifetime scope, process-owned memory cleanup, direct trusted delete, or runtime
shutdown, the module-bound Object release or shutdown finalizer closes the
underlying PTY as the object's RAII resource.

PTY sessions are not checkpointed or persisted as reconnectable host handles.
A checkpoint or committed image may contain an `EXTERNAL_REF` row only as stale
metadata; reopening the runtime releases stale PTY objects rather than trying
to reconnect a host terminal process.

## Scheduler

The scheduler is thread-backed. It starts one worker task per runnable process
up to `config.scheduler.max_workers`, and each task advances only that process
until it blocks, exits, fails, or the shared quantum budget is exhausted. Public
async APIs remain available for event-loop hosts, but they are wrappers around
the same scheduler and do not mean process quanta are serialized on one asyncio
loop.

The high-level synchronous entrypoint is:

```python
results = runtime.run_until_idle(max_quanta=10)
```

Event-loop hosts should use:

```python
results = await runtime.arun_until_idle(max_quanta=10)
```

By default it:

1. runs runnable processes,
2. processes pending human terminal messages when work is blocked on human I/O,
3. delivers process-message notices at tool boundaries,
4. wakes resumed processes,
5. stops when no runnable or human-resumable work remains, or when the quantum
   budget is exhausted.

`max_quanta` is a global budget across all process workers, not a per-process
limit. A bounded run may briefly reserve extra dependency quanta when a running
process is already waiting on a runnable child or message dependency and the
main worker pool is full; that keeps parent/child waits from deadlocking behind
the nominal budget. After budget exhaustion, `config.scheduler.drain_window_s`
gives already-running workers a short chance to finish before unfinished quanta
are cancelled or detached.

The scheduler serializes top-level `run_until_idle`, `run_pid_until_idle`, and
single-step invocations for one `Runtime` instance, so two host calls cannot
re-enter the same runnable process concurrently. Individual process claims are
also status-checked at the store boundary before a quantum changes a process
from `runnable` to `running`.

Single-step APIs remain available for tests and debugging:

```python
result = runtime.run_next_process_once()
result = runtime.run_process_once(pid)

result = await runtime.arun_next_process_once()
result = await runtime.arun_process_once(pid)
```

For debugging pending approval states, disable human queue processing:

```python
results = await runtime.arun_until_idle(max_quanta=1, process_human_queue=False)
```

`Runtime.shutdown()` first asks the scheduler to cancel and join tracked worker
futures for up to `config.scheduler.shutdown_join_timeout_s`, then asks
ObjectTask runners to drain their tool executor for up to
`config.object_tasks.shutdown_join_timeout_s`. If a synchronous quantum or
ObjectTask tool thread cannot stop safely, shutdown reports `ok: false` with
`scheduler_stopped: false` or `object_tasks_stopped: false` and leaves the
SQLite store open rather than closing it underneath a live worker. Once the
worker finishes, a later shutdown can complete normal resource cleanup.

## Resource Budgets

Resource limits are runtime constraints, not Capabilities. A process may have a
`ResourceBudget` covering tool calls, child processes, runtime seconds, LLM
calls and tokens, context materialization tokens, subprocess wall/CPU/RSS usage,
external filesystem bytes, JSON-RPC bytes, and Deno syscalls. Observed
consumption is stored as `ResourceUsage` on the process row.

Every charge applies to the acting process and its parent chain, so a parent can
bound an entire child tree. Fork and spawn may request a child budget, but it
must fit within the parent's remaining budget. `exec_process` keeps the same pid
and does not reset usage or increase budget. Checkpoint restore replays recorded
process rows, including their resource state, for the restored processes.
Checkpoint-committed images do not store or restore resource budgets or usage;
only the caller that starts the process may set launch-time resource limits.

LLM token usage is charged after provider completion using provider-reported
usage. If a token budget exists and the provider does not return billable usage,
the LLM action fails closed. When an LLM completion pushes usage over budget,
the call record is retained but model-selected tools are not dispatched.
Context materialization has both a per-call cap
(`max_context_materialization_tokens`) and a separate cumulative budget
(`max_context_materialization_total_tokens`). The cumulative context token
budget is charged when Object Memory materializes prompt context and is
accounted independently from provider-reported LLM tokens. In the LLM executor,
the final rendered `llm_context:<pid>` prompt context is the charged unit;
source object materialization for delta capture does not double-charge the same
quantum, and over-budget rendered context fails closed before the model call.

Shell and Deno subprocesses are run through provider-level monitors. The
default local provider uses cross-platform process-tree sampling to enforce wall
time, CPU time, and peak RSS budgets, then records metrics and audits limit
exceedance. In-process Python primitives are not hard CPU/RSS isolated; they
remain bounded by call count, wall time, byte limits, and primitive-specific
caps.

## Human Queue

Human interaction is modeled as runtime objects, not raw prompt text.

- `ask_human` creates a blocking question.
- `request_permission` requires human write authority, creates a blocking scoped
  policy request with canonical resource, risk, resource scope, lease shape, and
  constraints shown to the human, then returns the final policy decision. Model
  requests cannot ask for broad high-risk grants such as `shell:*` execute or
  root/global filesystem write such as `filesystem:/:*`; workspace write remains
  a human-approvable scope.
- `human_output` writes through the HumanObject primitive and provider.
- Per-use approvals can create one-shot capabilities that are consumed after
  one successful primitive call.

If a primitive or human tool blocks on human approval, the process enters
`waiting_human`. Human requests are terminally decided once: only pending
requests can be approved or rejected. The runtime can process human terminal
messages, update the request, wake the process, and resume the original
operation. Rejection returns a normal failure to the process instead of crashing
the runtime, except `request_permission` rejection returns a structured
`rejected` decision after installing the selected deny policy.

## Process Messages And IPC

Each process has a durable message queue. Messages include:

- sender and recipient pid,
- `kind` such as `normal` or `interrupt`,
- channel,
- correlation id,
- reply target,
- subject and body,
- structured payload,
- delivery and acknowledgement state.

Processes can send messages to themselves, their parent, or direct children.
Receivers use `read_process_messages` for non-blocking reads or
`receive_process_messages` to block until matching unread messages arrive.
Filters can match kind, sender, channel, correlation id, reply target, and exact
message ids. An explicit empty message-id filter matches no messages; it never
means "all messages." Read limits bound both returned messages and
acknowledgement; a blocking receive must use a positive limit and a non-empty
explicit id filter because zero-size receive windows cannot ever produce a
message.

Interrupt messages preempt before non-message tool calls until read. Normal
messages notify after a tool call and do not block the current action.

ObjectTask completion and waiting notices use the same queue. By default they
arrive on channel `object-task` from sender `object_task:<task_id>`. A process
can block with `receive_process_messages(channel="object-task")` and will be
woken by matching task notifications. If the selected notification process has
already exited, the task records `undelivered_terminal`; task success is not
converted into failure.

ObjectTask owner-watch notices also use this queue. They are addressed to the
runner process on `object-task-owner` by default and can resume a task that is
blocked in `receive_process_messages`; the notice contains event metadata and
object ids, not Object Memory read authority. ObjectTask runner processes are
host-managed and skipped by the LLM scheduler; owner-watch auto-resume is
limited to message receive tools that are safe to replay.

CLI examples:

```bash
uv run agent-libos --db .agent_libos.sqlite message <pid> "Please inspect the result"
uv run agent-libos --db .agent_libos.sqlite interrupt <pid> "Stop and read this first"
```

## Fork, Spawn, Exec, Wait, Signal

`fork_child_process` creates a direct child with an attenuated parent
`MemoryView`. It can inherit selected capabilities only if the parent already
holds them.

`spawn_child_process` creates a fresh direct child with a new process namespace
and goal-only memory. It does not inherit parent-activated Skills or broad
external authority by default.

`exec_process` replaces the current image and tool table without changing pid.
It never grants the target image's declared required capabilities
automatically. Existing external capabilities are preserved only when explicitly
requested; otherwise exec shrinks external authority.

`wait_child_process` blocks the parent in `waiting_event`. Child exit wakes the
parent and resumes the original wait action without asking the model for a new
action.

Signals can pause, resume, cancel, interrupt, or terminate direct children.

## Process Exit

`process_exit` marks a process as `exited` or `failed` and can attach a final
Object Memory result. Process-owned memory is released on exit unless retained
as the process result. Cleanup follows explicit Object Memory owner fields, not
the object's creator provenance, and release revokes stale object capabilities.

When a Deno JIT tool calls `process.exit` or `process.exec`, the syscall records
a deferred lifecycle change. The runtime applies that change only after the JIT
tool returns its normal tool result.
