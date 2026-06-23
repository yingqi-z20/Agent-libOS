# Agent libOS

Agent libOS is an experimental agent-native libOS runtime written in Python.
It supports the paper theme:

> Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents

The runtime models an agent as a long-running, schedulable, interruptible,
capability-controlled `AgentProcess`, not as a single chat request or workflow
thread. Agents may activate Skills, register Deno/TypeScript JIT tools,
register, execute, or commit new images from checkpoints, fork children,
checkpoint/fork state, and use registered remote resources, but these
self-evolution mechanisms do not grant resource authority by themselves.

The current contribution is the runtime authority boundary:

```text
process identity + capability + primitive + audit
```

LLM-facing tools, Skills, JIT tools, image definitions, child processes,
checkpoints, and remote endpoint visibility are ergonomic affordances. They are
not the security boundary.
Protected effects happen only inside libOS primitives, where process identity,
capabilities, human approval, policy, provider containment, events, and audit
records are enforced.

This project is still in active development. [agent_libos_design_doc.md](agent_libos_design_doc.md)
is a historical design archive and may describe planned or superseded
interfaces.

## Current System

The implementation currently includes:

- Agent process lifecycle: `spawn`, `fork`, `exec`, `wait`, `signal`, `pause`,
  `resume`, and `exit`.
- Hierarchical process resource budgets for tool calls, LLM token usage,
  subprocess wall/CPU/RSS usage, filesystem bytes, JSON-RPC bytes, and Deno
  syscalls.
- Thread-backed process scheduling through `Runtime.run_until_idle()` and the
  async host wrapper `Runtime.arun_until_idle()`, so blocked quanta do not
  monopolize scheduler progress.
- Process-local working directories for filesystem and shell operations.
- Optional Object-bound PTY sessions through the trusted `modules/pty` runtime
  module; when that module is loaded, `pty_create` returns an Object Memory
  `EXTERNAL_REF` handle, and read, write, resize, and close rights follow
  object capabilities.
- Durable process message queues for IPC, including interrupt delivery.
- Object-bound background tool tasks that can notify processes through the
  same durable message queues, including optional owner-change watches, without
  exposing their runner child processes to the LLM scheduler.
- Human queue integration for ordinary questions and per-use approval.
- Process-private Object Memory namespaces by default, with explicit shared
  namespaces available through capabilities.
- Structured Capability authority for filesystem, shell, clock, human,
  process, image, checkpoint, skill, and Object Memory primitives, including
  typed resource matching, deny/ask/allow effects, one-shot grants,
  attenuation, revoke, and audit lineage.
- A Resource Provider Substrate for injectable filesystem, clock, shell, and
  human I/O backends, plus a JSON-RPC over HTTP client provider for
  pre-registered remote endpoints.
- Trusted startup Runtime Modules loaded from manifest-declared Python
  entrypoints before `Runtime.open()` returns. Modules can register tools,
  images, syscalls, provider hooks, and startup hooks, but do not grant process
  resource authority.
- A direct workflow entrypoint for users to run one image-visible tool through
  ToolBroker without invoking the LLM scheduler.
- SQLite persistence for process/object metadata, capabilities, messages,
  human requests, LLM calls, events, audit records, tools, Skill/JIT metadata,
  and scoped checkpoints.
- Deno/TypeScript JIT tools that can access libOS only through `libos.syscall`.
- Declarative Skills that can add prompt instructions, visible tools, and JIT
  candidates without granting resource authority.
- Client-only JSON-RPC 2.0 over HTTP through registered endpoints, method
  capabilities, provider-classified external effects, audit, and checkpoints.
- A deterministic runtime-safety benchmark harness with 20+ checked-in tasks,
  including a self-evolution subset, baselines, side-effect oracle, and metrics
  collection.

## Documentation

Start here, then read the deeper references as needed:

- [docs/architecture.md](docs/architecture.md): runtime layers, provider
  substrate, and the tool/primitive boundary.
- [docs/runtime_model.md](docs/runtime_model.md): process lifecycle, scheduler,
  cwd, human queue, IPC, fork/spawn/exec, and waits.
- [docs/capabilities.md](docs/capabilities.md): resource naming, rights,
  one-shot grants, human approval, shell policy, and filesystem containment.
- [docs/object_memory.md](docs/object_memory.md): namespaces, object rights,
  file/object bridge, context materialization, and payload persistence.
- [docs/tools_and_jit.md](docs/tools_and_jit.md): built-in tools,
  ToolBroker, Deno/TypeScript JIT tools, syscall protocol, and sandbox rules.
- [docs/modules.md](docs/modules.md): trusted startup Runtime Module
  manifests, trust model, registration surfaces, CLI, and checkpoint behavior.
- [docs/jsonrpc.md](docs/jsonrpc.md): client-only JSON-RPC endpoint registry,
  capability resources, tools, syscalls, and checkpoint behavior.
- [docs/skills.md](docs/skills.md): standard `SKILL.md` packages,
  workspace/global sources, trust, activate/unload semantics, bundled JIT
  tools, and `swe-agent`.
- [docs/checkpoints.md](docs/checkpoints.md): scoped snapshots, restore, fork,
  replay diagnostics, append-only history, and external effects.
- [docs/cli.md](docs/cli.md): stable CLI command reference and examples.
- [docs/gui.md](docs/gui.md): Electron desktop console, local GUI server,
  HTTP/SSE APIs, and development commands.
- [docs/benchmark.md](docs/benchmark.md): M1 runtime-safety benchmark tasks,
  runners, oracle, outputs, and metrics.
- [docs/mini_swe_agent_image.md](docs/mini_swe_agent_image.md): package-only
  `mini-swe-agent` image behavior and known interface differences.
- [docs/development.md](docs/development.md): setup, tests, real LLM smoke,
  configuration defaults, and contribution rules.
- [docs/invariants.md](docs/invariants.md): current invariant-to-test map.
- [docs/artifact_anonymity.md](docs/artifact_anonymity.md): anonymous artifact
  hygiene checklist.
- [docs/paper_thesis.md](docs/paper_thesis.md): current paper thesis and
  non-goals.
- [benchmarks/runtime_safety/schema.md](benchmarks/runtime_safety/schema.md):
  benchmark task schema v0.

## Quick Start

Install dependencies:

```bash
uv sync --frozen --all-groups
```

Run tests:

```bash
uv run python scripts/test_matrix.py --lane unit
uv run python scripts/test_matrix.py --lane security
uv run python scripts/test_matrix.py --lane runtime
uv run python scripts/check_test_invariants.py
```

The `runtime` and `all` lanes use bounded pytest-xdist parallelism by default
to keep CI wall-clock time under control. Pass `--workers 1` for serial failure
diagnosis, or `--workers N` / `--workers auto` to override the worker count for
any Python lane. The GUI lane builds shared frontend artifacts and should be run
separately. Pytest removes files created under ignored `agent_outputs/` at the
end of a test session; use `--keep-agent-outputs` when debugging generated
files.

Run the deterministic local demo:

```bash
uv run agent-libos demo
```

Run the Electron GUI in development mode:

```bash
npm --prefix gui install
npm --prefix gui run electron:dev
```

The GUI starts a local `agent-libos-gui-server`, subscribes to runtime events,
and provides a process-centered console for concurrent messages, interrupts,
human approvals, scheduler control, image selection/registration/commit, audit
inspection, and LLM call visibility.

The demo does not call a real model. It exercises process spawn/fork, Object
Memory, Deno/TypeScript JIT validation when Deno is available, checkpointing,
capability denial before grant, human approval, filesystem write, final report
object creation, and audit trace generation.

Run a small deterministic benchmark smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --output .benchmark_runs/m1-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/m1-smoke
```

The benchmark defaults to mock/planned actions and does not spend model tokens.
Real-model benchmark smoke is opt-in and must be scoped with `--llm real
--limit 1` or a single `--task`.

## Persistent Runtime

Use `--db` to keep runtime state in SQLite:

```bash
uv run agent-libos --db .agent_libos.sqlite init
uv run agent-libos --db .agent_libos.sqlite spawn --image coding-agent:v0 --goal "Summarize README.md"
uv run agent-libos --db .agent_libos.sqlite run --max-quanta 10
uv run agent-libos --db .agent_libos.sqlite processes
uv run agent-libos --db .agent_libos.sqlite resources <pid>
uv run agent-libos --db .agent_libos.sqlite audit
uv run agent-libos --db .agent_libos.sqlite workflow run get_working_directory
```

Omit `--max-quanta` to run until the runtime becomes idle; provide it only when
you want a bounded run.

`workflow run <tool>` spawns a fresh process from the default image, calls one
visible tool, persists the result object, and exits that process. Pass
`--image <image_id>` to use another image's tool table. It does not bypass
primitive capability checks, resource budgets, human approval, or audit.

Every LLM action-selection call is persisted as an `llm_calls` row with prompt
messages, visible tool schemas, model output, tool calls, token usage when
available, reasoning metadata when exposed, raw provider response JSON, and
errors.

```bash
uv run agent-libos --db .agent_libos.sqlite llm-calls --pid <pid>
```

## Real LLM Configuration

Create a local `.env` file for real-model execution:

```bash
OPENAI_BASE_URL=https://example-openai-compatible-endpoint/v1
OPENAI_LANGUAGE_MODEL=your-model
OPENAI_API_KEY=...
```

The client uses the OpenAI Python SDK. It uses the Responses API for
OpenAI-hosted models by default and falls back to Chat Completions for custom
OpenAI-compatible `base_url` providers. Set `OPENAI_API_MODE=responses` or
`OPENAI_API_MODE=chat` to force a mode.

Optional knobs include `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`, `OPENAI_STORE`,
`OPENAI_REASONING_EFFORT`, `OPENAI_VERBOSITY`, and provider-specific
`OPENAI_ENABLE_THINKING`.

## Common CLI Examples

Send ordinary and interrupt messages:

```bash
uv run agent-libos --db .agent_libos.sqlite message <pid> "Please inspect the latest result"
uv run agent-libos --db .agent_libos.sqlite interrupt <pid> "Stop current work and read this first"
```

Run an interactive Codex CLI-style loop:

```bash
uv run agent-libos --db .agent_libos.sqlite run --interactive --pid <pid> --max-quanta 20
```

Manually control process cwd and lifecycle:

```bash
uv run agent-libos --db .agent_libos.sqlite cd <pid> src
uv run agent-libos --db .agent_libos.sqlite exec images/review-agent "Review README.md" --pid <pid> --run
uv run agent-libos --db .agent_libos.sqlite exit <pid> --payload '{"done":true}'
```

Commit a checkpoint into a new checkpoint-derived image:

```bash
uv run agent-libos --db .agent_libos.sqlite images commit <checkpoint_id> stateful-agent:v0 --name stateful-agent
uv run agent-libos --db .agent_libos.sqlite spawn --image stateful-agent:v0 --goal "Reuse the baked state"
```

Register and activate the SWE-Agent style Skill:

```bash
uv run agent-libos --db .agent_libos.sqlite skills validate skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills register skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills activate <pid> swe-agent
```

Register and call a preconfigured JSON-RPC endpoint:

```bash
uv run agent-libos --db .agent_libos.sqlite jsonrpc register endpoint.yaml
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> jsonrpc:demo-weather:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite jsonrpc call <pid> demo-weather forecast --params-json '{"city":"Beijing"}'
```

Inspect or change runtime authority:

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities list --subject <pid>
uv run agent-libos --db .agent_libos.sqlite capabilities explain <pid> filesystem:workspace:README.md read
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> filesystem:workspace:README.md --rights read
```

Launch a coding agent against another workspace:

```bash
uv run python scripts/run_coding_agent.py --workspace /path/to/repo --goal "Implement the requested change"
```

The launcher loads `.env` from this Agent libOS checkout before mounting the
target workspace. It does not automatically read the target workspace's `.env`;
pass `--env-file /path/to/env` when a run needs a different credential file.

On Windows PowerShell:

```powershell
uv run python scripts\run_coding_agent.py --workspace ..\some-repo --goal "Summarize the current project"
```

See [docs/cli.md](docs/cli.md) for the full command reference.

## Core Invariants

- Tool visibility is not resource authority.
- Capability records are typed authority statements with explicit
  allow/deny/ask effects, issuer lineage, delegation depth, status, expiry, and
  optional use counts.
- Skills and JIT tools do not grant filesystem, shell, human, object, process,
  image, checkpoint, or JSON-RPC remote authority.
- JIT syscalls bypass the LLM-facing tool table but not primitive capability
  checks, permission policy, human approval, or audit.
- Human approval is part of a primitive/syscall. Callers see a final success or
  final failure, not a pending/retry protocol.
- When the optional PTY module is loaded, PTY sessions are host runtime
  resources bound to mutable Object Memory `EXTERNAL_REF` handles. Shell policy
  authorizes creation, object read/write/delete rights authorize interaction
  and close, and runtime shutdown or object release closes the host PTY.
- `process.exit` and `process.exec` are ordinary syscalls from TypeScript. The
  runtime applies lifecycle changes after the JIT tool returns its normal tool
  result.
- Checkpoint restore covers reconstructable process-subtree state only. It does
  not delete append-only audit/events/LLM calls or roll back filesystem, shell,
  image, network, or provider side effects.
- Checkpoint-derived images capture internal reconstructable runtime state, not
  external provider state. Their required capabilities are declarations and are
  not granted automatically at spawn or exec.
- Providers classify external effects as `irreversible`, `rollbackable`, or
  `no_rollback_required`; checkpoint restore reports those classes with
  `restore_external_policy="report_only"`.
- Resource Provider Substrate backends perform host effects, but primitives own
  capability checks, policy decisions, events, and audit.

See [docs/invariants.md](docs/invariants.md) for test coverage.

## Development

Run the standard local checks:

```bash
uv sync --frozen --all-groups
uv run python -m compileall agent_libos tests scripts experiments benchmarks
uv run python scripts/test_matrix.py --lane all --workers 4
uv run python scripts/check_test_invariants.py
uv run python scripts/test_matrix.py --lane gui
git diff --check
```

Use `uv run python scripts/clean_agent_outputs.py` to dry-run cleanup of
already accumulated local outputs, and add `--yes` to delete them.

Deno is optional for the Python unit suite. Install `deno` or pass a runtime
config built with `dataclasses.replace(DEFAULT_CONFIG, tools=replace(...))` if
you want to validate and run real Deno/TypeScript JIT tools from another binary.

Runtime defaults live in `agent_libos.config.DEFAULT_CONFIG`, including
scheduler quantum, worker, drain, and shutdown limits; process budgets; image
ids; workspace namespace; tool limits; filesystem/Object Memory size limits;
Deno sandbox limits; ObjectTask notification and shutdown limits; JSR import
allowlists; shell policy lists; launcher presets; Skill defaults; and
checkpoint defaults. Optional modules such as `modules/pty` keep their own
module-local settings outside `AgentLibOSConfig`.
`AgentLibOSConfig` is validated at construction time, so invalid or inverted
bounds fail before a Runtime starts.

Add runtime dependencies with `uv add <package>` and development dependencies
with `uv add --dev <package>`. Commit both `pyproject.toml` and `uv.lock` after
dependency changes.
