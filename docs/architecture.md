# Architecture

Agent libOS is structured around one boundary: model-visible and
self-evolving action surfaces are not resource authority. A process may see a
tool schema, activate a Skill, register a JIT tool, register or exec an image,
fork a child, restore from a checkpoint, or inspect a remote endpoint, but
protected effects are authorized only when a primitive runs under that process
id.

## Layer Model

```text
Agent personality / application
  -> Skills and tools layer
     - model-facing actions
     - prompt instructions
     - tool schemas
     - Deno/TypeScript JIT candidates
  -> Agent libOS runtime
     - trusted startup module loader
     - scheduler
     - process manager
     - Object Memory manager
     - ToolBroker
     - Skill manager
     - HumanObject manager
     - primitive managers
     - capability manager
     - event bus
     - checkpoint manager
     - audit manager
  -> Resource Provider Substrate
     - filesystem provider
     - clock provider
     - shell provider
     - human provider
     - JSON-RPC over HTTP provider
  -> host backend
     - local workspace filesystem
     - host clock
     - subprocess backend
     - terminal or UI human I/O
     - pre-registered remote JSON-RPC endpoints
     - future container, WASM, or service providers
```

The Skills and tools layer exists for LLM ergonomics and self-evolution. It
presents stable action names, schemas, summaries, workflow instructions, and
process-local JIT candidates. It does not own external authority.

Image registration and `exec` are also self-evolution mechanisms. They can
change a process prompt, prompt composition mode, default tool table, default
Skills, and lifecycle shape, but image visibility and target-image metadata do
not grant resource capabilities or impose resource budgets. Launch-time callers
own resource limits for newly started processes. Image packages may seed a
private per-process workspace and process-local JIT tools, but those are scoped
to the booted process and do not expose the package source directory.
Default tool tables are exact image declarations: the runtime does not add
generic lifecycle or Object Memory tools unless the image lists them.

Startup Runtime Modules are different from Skills. A module is trusted Python
host code loaded before `Runtime.open()` returns. Modules extend the runtime
composition root by registering tools, images, syscalls, provider hooks, and
startup hooks. Because modules run in the host interpreter, they are part of
the runtime trusted computing base and are gated by manifest hash trust rather
than by process capabilities.

The runtime owns agent-level semantics: process identity, capability checks,
approval, event emission, audit, process wakeups, checkpointing, and durable
metadata.

The Resource Provider Substrate owns concrete host calls. A provider is a
backend, not a security bypass. Replacing the filesystem or shell provider must
not change tool schemas or skip primitive authorization.

Providers are also the source of truth for external-effect rollback
classification. Effectful provider calls must return an external-effect
classification to the primitive; missing classification fails closed instead of
silently executing. The runtime persists those records for checkpoint reports,
but v1 does not apply external compensation.

## Composition Root

`agent_libos.runtime.runtime.Runtime` wires the runtime together:

- `SQLiteStore` persists metadata and append-only records.
- `RuntimeModuleRegistry` loads the internal core module and configured trusted
  startup modules before processes, tools, or LLM execution can run.
- `CapabilityManager` grants, checks, revokes, and consumes one-shot authority.
- `ObjectMemoryManager` provides typed memory and namespace resolution.
- `HumanObjectManager` owns questions, approvals, terminal queue processing,
  and human output.
- `FilesystemAdapter`, `ShellAdapter`, `ClockPrimitive`, and
  `JsonRpcPrimitive` expose protected primitive operations over provider
  backends.
- `ToolBroker` registers static tools and process-local JIT tools.
- `SkillManager` registers standard Skill packages and activates them into
  process tool tables and prompt context without granting resource authority.
- `ProcessManager` owns lifecycle, working directories, child relationships,
  and image transitions.
- `SimpleScheduler` runs runnable processes and wakes waiting work.
- `CheckpointManager` snapshots and restores reconstructable process-subtree
  state; checkpoint-derived image commit reuses that internal snapshot boundary.
- `LLMProcessExecutor` materializes prompt context, resolves the process
  `llm_profile_id` through the host profile registry, calls that LLM client,
  and dispatches selected tool calls.

The default substrate is `LocalResourceProviderSubstrate`, rooted at the current
workspace unless another substrate is injected.

The internal core module registers the built-in tool set and default images
through the same module registration path exposed to trusted external modules.
This keeps future providers, syscalls, and images from accumulating ad hoc
startup code in the composition root.

Host-facing control surfaces live under `agent_libos.api`. The CLI entrypoint
and the local GUI HTTP/SSE server are different presentations over the same
runtime managers and primitives; neither is an authority boundary by itself.
Both must call `Runtime.shutdown()` when they own a runtime instance. Shutdown
first stops scheduler work and ObjectTask runner work, then releases host
resources and emits runtime lifecycle audit/event records. If a synchronous
quantum or ObjectTask tool thread cannot be joined safely, shutdown reports the
component that did not stop and leaves owned storage open so a live worker is
not racing a closed SQLite connection. Host shutdown never marks AgentProcess
records as exited.

## Tool Boundary

LLM-facing tools are stable wrappers over primitives. For example,
`write_text_file` can be visible in a process tool table, but the actual write
still enters the filesystem primitive, which checks:

- workspace containment,
- process working directory resolution,
- filesystem capability or permission policy,
- human approval if policy requires it,
- overwrite and content preview metadata,
- event emission,
- audit recording.

Putting a tool in a process table never grants access to files, shell,
terminal/human I/O, Object Memory, image registration, checkpoints, or other
resources.

Likewise, `call_jsonrpc_method` visibility never grants network authority. The
JSON-RPC primitive accepts only endpoint and method ids, resolves URLs and
env-backed headers from the registry, then checks `jsonrpc:<endpoint>:<method>`
capabilities before the provider performs a POST.

## Primitive Boundary

Primitives are the runtime boundary. They are responsible for:

- authorizing the caller pid against capabilities and policy,
- blocking on human approval when needed,
- validating inputs before side effects,
- constraining provider paths, argv, sizes, and timeouts,
- emitting events,
- writing audit records,
- preserving process wake/resume semantics.

JIT syscalls enter the same primitive boundary through
`LibOSSyscallSession`. They do not consult the caller's LLM-facing tool table.
Trusted startup modules may add new syscall names through the runtime syscall
router, but module syscalls still execute as libOS syscalls under the caller
pid and must call primitives for protected effects.

## Persistence And Audit

SQLite stores durable runtime metadata and append-only records:

- processes, working directories, loaded Skills, and tool tables,
- Object Memory metadata and namespace directories,
- capabilities and object handles,
- process messages and human requests,
- tools and JIT candidates,
- Skill registry and trust rows,
- loaded Runtime Module status, source hashes, and registration summaries,
- image registry manifests and checkpoint-derived image artifacts,
- JSON-RPC endpoint registry rows,
- checkpoints and checkpoint payload snapshots,
- provider-decided external effect records,
- events and audit records,
- LLM call records with provider ids, model/API mode, usage, errors, and
  bounded observability envelopes for prompt, visible tools, output, tool
  calls, reasoning metadata, and raw response. Full LLM input/output
  persistence is an explicit `llm.persist_full_io` opt-in.

Object payloads are not ordinary durable object rows. They live in runtime
memory. Checkpoint payloads are the explicit durable snapshot exception.

Audit and events are append-only. Checkpoint restore must not delete them.
Limited audit views select the latest matching records first and return that
window in chronological order, so GUI snapshots and per-process audit pages keep
showing new records as the log grows. Shell execution records an intent audit
record immediately before crossing into the shell provider; the result, timeout,
or resource-limit audit record uses the intent record as its parent and
correlation id.

## Module Map

```text
agent_libos/
  api/             CLI and GUI HTTP/SSE server entrypoints
  capability/      capability grant, revoke, check, and object handles
  config/          typed runtime, LLM, tool, memory, launcher, and script defaults
  human/           HumanObject query, approval, interrupt, and output primitives
  images/          built-in AgentImage definitions
  llm/             prompt, context, OpenAI-compatible client, executor, action parser
  memory/          typed Object Memory and MemoryView implementation
  models/          dataclass and enum models split by runtime domain
  modules/         trusted startup Runtime Module loader, registry, and core module
  primitives/      libOS primitives for filesystem, clock, shell, and JSON-RPC
  runtime/         composition, syscalls, scheduler, processes, events, checkpoints, audit
  skills/          Skill schema, strict loader, trust registry, and SkillManager
  substrate/       provider interfaces and local host-backed implementations
  storage/         SQLite persistence
  tools/           tool base classes, ToolBroker, sandbox, and built-in tools
  utils/           shared validation, YAML loading, and helper utilities
benchmarks/        deterministic runtime-safety benchmark harness and fixtures
docs/              current implementation documentation
experiments/       benchmark entrypoints
gui/               Electron/React desktop console
images/            workspace AgentImage packages
modules/           workspace trusted Runtime Module packages
scripts/           real-model smoke and demo scripts
skills/            workspace standard Agent Skill packages
tests/             safety-boundary and regression tests
```
