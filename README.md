# Agent libOS

Agent libOS is an experimental agent-native libOS runtime written in Python,
with an optional Electron desktop console. It models an agent as a long-running,
schedulable, interruptible, capability-controlled `AgentProcess`, rather than a
single chat request or workflow thread.

The runtime separates three boundaries that must not be collapsed:

```text
authority        = process identity + Task Authority + Capability/effect constraints
information flow = data labels + Host Sink trust + exact release
evidence         = Operations + events + audit/settlement records
```

Authority and information-flow checks decide whether an operation may proceed.
Evidence records explain or settle that decision; an audit row, event, receipt,
hash, or prior success is never an authority credential.

LLM-facing tools, Skills, JIT tools, AgentImages, child processes, checkpoints,
and remote endpoint visibility are ergonomic or extension surfaces. They do not
grant resource authority by themselves. Protected operations enter trusted
primitive or SDK boundaries that enforce the applicable process identity,
Capability, Task Authority, policy, approval, data flow, budget, provider-effect,
event, and audit rules.

Trusted Runtime artifact publication is a narrow TCB exception: it may
materialize validated Image content only into a Runtime-owned private workspace
under durable publication recovery and resource accounting. It is not a general
filesystem bypass for a process or model.

This project is under active development. Current behavior is defined by the
code and current-contract documents in the same checkout. Start with the
[documentation home](docs/index.md); historical design material is identified
separately below.

## 3 to 5 Minute Quick Start

This deterministic path runs locally, does not call a real model, and requires
no API credentials. It needs Python 3.11–3.14,
[uv](https://docs.astral.sh/uv/), and a source checkout:

```bash
git clone https://github.com/yingqi-z20/Agent-libOS.git
cd Agent-libOS
uv sync --frozen
uv run agent-libos --db local demo
```

The final command should exit successfully, print a larger JSON summary, and
write `agent_outputs/demo_patch_preview.txt`. Process ids and Object ids vary;
look for this stable success shape:

```json
{
  "target_file_exists": true,
  "target_file_content_matches": true,
  "final_report_oid": "obj_..."
}
```

The demo covers process spawn/fork, Object Memory, capability denial and grant,
Human approval, checkpointing, a filesystem write, final-result creation, and
audit evidence. Deno/TypeScript validation runs only when the Host can use its
configured Deno executable; the rest of the demo remains token-free.

Inspect the file, then remove only this demo output:

```bash
test -f agent_outputs/demo_patch_preview.txt
uv run python -c 'from pathlib import Path; Path("agent_outputs/demo_patch_preview.txt").unlink(missing_ok=True)'
```

The repository-wide `scripts/clean_agent_outputs.py` utility is broader: it
previews, and with `--yes` removes, all ignored outputs under `agent_outputs/`.
If setup or the demo fails, use [Troubleshooting](docs/troubleshooting.md).

Choose the next path by role:

- user/operator: [CLI guide](docs/cli.md), [Python API](docs/python_api.md), or
  [Electron GUI](docs/gui.md);
- security reviewer: [threat model](docs/threat_model.md),
  [Capabilities](docs/capabilities.md), and [data flow](docs/data_flow.md);
- extension author: [AgentImages](docs/agent_images.md), [Skills](docs/skills.md),
  [Tools/JIT](docs/tools_and_jit.md), or [Runtime Modules](docs/modules.md);
- contributor/maintainer: [development](docs/development.md),
  [benchmarking](docs/benchmark.md), and [releasing](docs/releasing.md).

## Current System

### Execution, memory, and durable work

- `AgentProcess` lifecycle includes spawn, fork, exec, wait, signal, pause,
  resume, exit, cwd selection, normal messages, and interrupt messages.
- The scheduler runs process quanta through sync and async Host APIs without a
  blocked quantum monopolizing progress.
- Object Memory provides typed Objects, namespaces, ownership, links,
  `MemoryView` context selection, results, artifacts, and external references.
- Object payload durability is deliberately narrower than metadata durability.
  Missing runtime-memory payloads fail closed rather than being reconstructed
  from an untrusted guess.
- Durable Task Runs supervise one root process tree with versioned goals and
  requirements, idempotent Host commands, evidence-linked ledger entries, safe
  continuations, explicit plaintext-payload opt-in, and restart recovery.
- ObjectTasks run bounded background tool work tied to an Object owner and may
  notify processes through durable messages; they are not another scheduler.
- Process-local working directories are authority-checked before directory
  state is probed or used by filesystem, Shell, PTY, or Git operations.
- Resource budgets cover tool calls, LLM calls/tokens, subprocess wall/CPU/RSS,
  filesystem bytes, JSON-RPC/MCP bytes, and Deno syscalls, with hierarchical
  accounting and durable reservations where ambiguity matters.
- Checkpoints capture reconstructable process-subtree state and support inspect,
  diff, restore, fork, replay diagnostics, and commit to a derived Image.
- Checkpoint restore and Image commit do not roll back or package external
  filesystem, Git, network, Shell, PTY, or provider state.

### Authority, approval, and information flow

- Capabilities bind a process to typed resource patterns, rights, decision
  effects, constraints, lineage, expiry/use limits, delegation, and revocation.
- Task Authority Manifests provide a Host-authored launch ceiling for authorized
  and requestable capabilities, effects, budgets, approvals, data-flow policy,
  and expiry.
- AgentImage and Skill capability requirements are declarations. Root spawn,
  exec, checkpoint boot, registration, and activation do not turn them into
  ambient grants.
- Human approval is part of a protected operation and can issue only authority
  still allowed by Capability, Task Authority, policy, data flow, and budgets.
- Runtime-mediated egress uses data labels, stable Sink identities, Host trust,
  tenant/principal clearance, and exact one-shot release for conditional sends.
- Trusted Sink configuration authorizes receipt of eligible data; it never
  grants the underlying filesystem, Git, Shell, Human, LLM, JSON-RPC, MCP, PTY,
  or process operation.
- The optional semantic plane is default-off. Its payload-free FlowGraph and
  classifier output are review evidence; deterministic Host predicates remain
  mandatory, and the classifier is never an allow predicate or safety oracle.
- Semantic policy activation, revocation, and exact canary settlement are
  Host-composition surfaces, not CLI, HTTP, GUI, Skill, JIT, Module, process, or
  model-facing control surfaces.

### Images, Skills, tools, and modules

- AgentImages define prompt composition, the exact static tool table, optional
  default Skills, a Host profile id, requirements, module prerequisites, and
  optional packaged JIT/private-workspace content.
- Directory-package Images are validated and registered as immutable artifacts;
  checkpoint-derived Images preserve reconstructable Runtime state only.
- Image `prompt_mode` supports exact `image_only`, bounded `minimal_runtime`,
  and full `libos_default` composition contracts.
- Skills use progressive disclosure, immutable package identity, trust and
  activation lifecycle, optional bundled JIT tools, and exact tool ownership.
- ToolBroker validates schemas and dispatches the complete process tool table;
  the model projection may be narrower than that callable table.
- Deno/TypeScript JIT tools are statically checked, contract-tested, run with no
  Deno permissions and cached-only execution, and reach Runtime effects only
  through the governed syscall surface.
- Package JIT tools can use direct schemas or one stable multiplexed
  `run_jit_tool` projection without weakening ToolBroker or primitive checks.
- Trusted startup Runtime Modules extend Python composition from verified
  manifests and exact source identities. An Image may require a loaded module
  but cannot load or trust one automatically.

### Providers, protected effects, and evidence

- The Host selects provider implementations for filesystem, clock, Shell,
  Human I/O, typed Git, LLM, JSON-RPC, MCP, and optional PTY resources.
- Built-in typed Git, filesystem writes, Shell/PTY input and session control,
  and process/remote actions remain distinct effect and authority families.
- The protected-operation SDK gives extensions stable preflight, authority,
  pre-intent, provider, settlement, result, event, and audit phases.
- Provider-backed interactions record durable effect intent before a boundary
  where a crash could otherwise erase uncertainty.
- Certified-not-started, success, failure, rollback, and unknown outcomes have
  distinct settlement rules; ambiguous outcomes do not restore finite-use
  authority merely because the caller lost a response.
- Explainable Operations connect LLM, Tool, syscall, primitive, Capability,
  Human, provider, event, and audit stages into bounded causal trees.
- Runtime events have a closed typed catalog and bounded payloads; they are
  observations, not a task queue or authority channel.
- Audit history and effect-transition history are append-only through
  RuntimeStore APIs. Direct database administration remains inside the TCB, so
  this evidence is not claimed to be tamper-proof against that actor.
- Retention can summarize or hash selected LLM/effect payload projections while
  preserving protected identities, transitions, and recovery requirements.

### Interfaces, storage, and evaluation

- The Python API exposes the Runtime composition root, managers, sync/async
  lifecycle, public models, exceptions, and provider injection boundaries.
- The `agent-libos` CLI covers local demo, process/runtime control, TaskRuns,
  ObjectTasks, Capabilities, Images, Skills, checkpoints, remotes, modules,
  semantic inspection, evidence, and offline store administration.
- The Electron GUI provides user and operator workspaces over one loopback
  server with bearer authentication, same-build schema validation, SSE updates,
  Human approvals, process control, registry administration, and evidence views.
- SQLite and PostgreSQL implement one RuntimeStore contract with schema checks,
  single-active-Runtime leases, ordered recovery, and explicit offline migration.
- The runtime-safety benchmark contains 33 checked-in schema-v1 tasks with
  deterministic runners, side-effect oracles, evidence completeness, and
  explicit metric denominators.
- Practical workflow suites preserve `native-live` versus `modeled` provenance;
  native rows never silently fall back to modeled evidence.
- Additional source-bound suites cover external-effect recovery,
  runtime-publication recovery, long-horizon agents, Tool-Skill projection,
  knowledge/browser workflows, and the isolated AgentDojo harness.

## Scope and Non-goals

Agent libOS is a library/runtime and research system for composing governed
agent execution. Its current scope is deliberately narrower than several nearby
product categories:

- it is not a general operating-system sandbox for hostile native code;
- it is not a hosted multi-tenant control plane or identity provider;
- a Durable Task Run is not a workflow DSL or distributed job scheduler;
- Object Memory is not a promise that every payload survives Runtime shutdown;
- checkpointing is not transactional rollback for external provider state;
- an Explain tree is not a proof of model intent or hidden chain of thought;
- the local GUI server is a loopback Host surface, not a remote administration
  API; and
- the self-contained desktop path is an explicitly bounded internal
  distribution, not a public signed installer or update channel.

Support and release claims are conditional on the exact version, platform,
configuration, and required evidence. Consult the [support matrix](docs/support_matrix.md)
and maintained contract pages before generalizing from one demo or test run.

## Documentation

The [documentation home](docs/index.md) routes by audience and task and contains
the complete current-contract inventory. Direct entry points are grouped here.

### Use and operate Agent libOS

- [CLI Guide](docs/cli.md) for workflows and security semantics;
- [machine-generated exhaustive CLI reference](docs/cli_reference.md) for the
  checked-in parser, with the same installed version's `--help` as the live
  source;
- [Python API](docs/python_api.md) and [Electron GUI](docs/gui.md);
- [configuration guide](docs/configuration.md) plus the
  [machine-generated exhaustive field reference](docs/configuration_reference.md);
- [troubleshooting](docs/troubleshooting.md), [storage/migration](docs/storage.md),
  and [support matrix](docs/support_matrix.md);
- [Durable Task Runs](docs/durable_task_runs.md),
  [Object Memory](docs/object_memory.md), and [checkpoints](docs/checkpoints.md).

### Understand the runtime and security boundary

- [architecture](docs/architecture.md), [runtime model](docs/runtime_model.md),
  [glossary and version map](docs/glossary.md), and
  [threat model](docs/threat_model.md);
- [Capabilities](docs/capabilities.md),
  [Task Authority Manifests](docs/task_authority_manifest.md), and
  [data flow and trusted Sinks](docs/data_flow.md);
- [semantic approval/data identification](docs/semantic_shadow.md),
  [Explainable Operations](docs/explainable_operations.md), and
  [Runtime events](docs/events.md);
- [evidence payload retention](docs/evidence_payload_retention.md) and
  [runtime invariants](docs/invariants.md).

### Extend and integrate

- [AgentImage authoring](docs/agent_images.md) and the specialized
  [`mini-swe-agent` Image](docs/mini_swe_agent_image.md);
- [Skills](docs/skills.md), [Tools and Deno/TypeScript JIT](docs/tools_and_jit.md),
  and [startup Runtime Modules](docs/modules.md);
- [Protected Operation SDK](docs/protected_operation_sdk.md) and
  [provider substrate](docs/providers.md);
- [typed Git](docs/git.md), [JSON-RPC](docs/jsonrpc.md), and
  [MCP client](docs/mcp.md);
- [GUI API schema subset](docs/gui_api_schema.json).

### Evaluate, contribute, and release

- [benchmark contract](docs/benchmark.md),
  [runtime-safety schema](benchmarks/runtime_safety/schema.md),
  [external-effect recovery](benchmarks/external_effect_recovery/README.md), and
  [runtime-publication recovery](benchmarks/runtime_publication_recovery/README.md);
- [practical workflows](benchmarks/practical_agent_workflows/README.md),
  [built-in Tool Skill evaluation](benchmarks/builtin_tool_skills/README.md),
  [long-horizon evaluation](benchmarks/long_horizon_agent/README.md), and
  [AgentDojo](experiments/agentdojo/README.md);
- [development guide](docs/development.md), [release status](docs/release_status.md),
  and [maintainer release runbook](docs/releasing.md);
- [artifact anonymity](docs/artifact_anonymity.md),
  [research thesis](docs/paper_thesis.md), [security policy](SECURITY.md),
  [contribution guide](CONTRIBUTING.md), [changelog](CHANGELOG.md), and
  [agent guidance](AGENTS.md).

Repository README links are checkout-relative so a branch, tag, fork, or source
archive stays bound to its adjacent documentation. The release build generates
separate package-index README metadata whose repository links are pinned to the
immutable `v<version>` tag; the source README is not published directly with
mutable-`main` contract links.

### Historical references (not current contracts)

These files are retained for traceability. Do not use them to infer current
commands, interfaces, security guarantees, support, or release evidence:

- [historical design archive notice](agent_libos_design_doc.md);
- [historical paper roadmap notice](plan.md);
- [retired commit-bound prelaunch review](docs/prelaunch_hardening_report.md);
- [commit-bound pre-implementation semantic research](docs/semantic_permission_and_dataflow_research.md).

Each file states its baseline or retirement status and points back to maintained
entry points. Git history, rather than a floating prose claim, is the source for
a genuinely commit-bound historical audit.

## Installation and Operations

The Quick Start above is the supported checkout onboarding path. The core source
environment is installed with `uv sync --frozen`; add only the integration extra
you need, such as `mcp`, `postgres`, or `pty`.

The Python wheel contains the core `agent_libos` package, immutable built-in
Tool Skills, and three console entry points: `agent-libos`,
`agent-libos-gui-server`, and the explicit offline
`agent-libos-migrate-tool-groups`. Repository examples, benchmarks, docs, the
optional PTY module, and workspace Image/Skill packages remain checkout/source-
archive assets rather than core-wheel contents.

Use the canonical page for every workflow beyond onboarding:

| Task | Canonical page |
| --- | --- |
| Install, test, or change dependencies | [Development Guide](docs/development.md) |
| Build, validate, publish, or recover artifacts | [Release Runbook](docs/releasing.md) |
| Migrate, back up, restore, or diagnose a store | [Storage](docs/storage.md) |
| Configure fields, profiles, modules, or providers | [Configuration Guide](docs/configuration.md) |
| Operate the desktop or GUI server | [GUI Guide](docs/gui.md) |
| Run or interpret evaluations | [Benchmark Contract](docs/benchmark.md) |
| Find a CLI workflow or every parser option | [CLI Guide](docs/cli.md) / [generated reference](docs/cli_reference.md) |

Those pages are the single sources for their commands, prerequisites, platform
differences, safety gates, and expected evidence. Do not copy a release,
migration, benchmark, or test recipe from an older README or another version.

## Persistent Runtime

`--db user` selects the persistent
`~/.agent-libos/runtime/agent-libos.sqlite` and is the default for the CLI,
development GUI server, and bare `Runtime.open()`/`Runtime.aopen()`. Packaged
Electron keeps its existing explicit
`<Electron userData>/runtime/agent-libos.sqlite` target. `--db local` and
`--db :memory:` are in-memory. A filesystem path outside the effective
model-visible workspace selects persistent SQLite, while a `postgresql://` or
`postgres://` DSN selects the optional PostgreSQL backend. Persistent targets
use the same RuntimeStore contract and enforce a single active writable Runtime
lease for one store target. A custom SQLite target inside the workspace is
rejected before database or sidecar creation; there is no unsafe override.

The current Runtime creates and opens only RuntimeStore schema v7. A canonical
v6 store requires the explicit, offline, digest-bound v6-to-v7 migration; older
supported stores must traverse the documented ordered migrations. Startup never
performs an automatic migration, backfill, dual-schema bridge, or speculative
repair. Legacy project-root `.agent_libos.sqlite` databases are not moved
automatically. Stop all writers and move or restore the database to an external
owner-only location before selecting it, or use the default `user` target for a
fresh store.

Durable metadata does not make every Object payload durable. Ordinary payloads
remain Runtime-memory state. With the documented default retention setting, one
narrow identity- and hash-bound envelope can recover the immutable initial goal
of an active root spawn across CLI invocations; child/fork/exec goals and other
Object payloads do not gain that guarantee.

Store administration is an offline operator workflow. Stop all writers, make
the documented owner-only backup, preview the exact migration, verify its bound
digest/state, and only then apply it. See [Storage](docs/storage.md) and
[Troubleshooting](docs/troubleshooting.md).

### Upgrading stores that contain Tool Groups

Legacy Tool Group metadata is a separate content migration to immutable built-in
Tool Skills. Ordinary startup fails closed; use the dry-run-first procedure in
[Tools and JIT](docs/tools_and_jit.md#on-demand-tool-skills) and the offline
storage guidance. Do not infer that a schema migration performs this conversion.

## Real LLM Configuration

The Quick Start and default deterministic evaluations do not call a real model.
Commands such as `run`, `llm-once`, `exec --run`, real-model benchmarks, and the
coding-agent launcher can spend tokens and create ambiguous provider outcomes.

Select a Host-configured LLM profile, keep the API-key value only in the exact
environment variable named by `api_key_env`, and review token/resource budgets
before dispatch. The generic Runtime and CLI do not implicitly load a workspace
`.env`; any env-file loading must be explicit or specific to a documented
launcher.

Custom OpenAI-compatible endpoints require the profile or Host-wide trust opt-in.
A URL supplied by model text is never sufficient. Agent libOS disables
provider-SDK internal retries and records its own bounded transport attempts.
After retry exhaustion it may pause the process for Host inspection rather than
pretending a lost response did not start.

The provider-response evidence is a bounded projection. Retained readable fields,
hashes, status, and attempt evidence are not hidden chain of thought, proof of a
provider's complete internal state, or permission to repeat an uncertain effect.
Provider storage, prompt caching, local full-I/O retention, and response chaining
are independent controls with different privacy and recovery consequences.

Use [Configuration](docs/configuration.md) for precedence and security semantics,
the [generated field reference](docs/configuration_reference.md) for exact fields
and defaults, and [Development](docs/development.md#real-llm-smoke) for an
explicit opt-in smoke path.

## Security Summary

- Tool or model visibility is not resource authority. Every effectful path must
  still enter the owning trusted boundary.
- Capability, Task Authority, policy, approval, resource budgets, and data-flow
  clearance are independent gates; satisfying one does not bypass another.
- Skills, Images, JIT tools, modules, checkpoints, and remote manifests cannot
  smuggle ambient authority through metadata or a previous success.
- Deno JIT rejects static import/re-export and dynamic `import()` paths; Runtime
  effects remain available only through governed syscalls.
- The local Shell/PTY provider is mediation for commands, environment, cwd,
  resources, effects, and evidence. It is not a hostile-native-code sandbox.
- Shell/PTY reject `env`, `nice`, `nohup`, `setsid`, `stdbuf`, and `timeout`
  whenever the launcher argv has a trailing token, before policy, Human
  approval, data-flow, evidence, or provider resolution. A single-token
  invocation such as `env` remains available through ordinary Shell authority.
- Typed Git mutating operations bind repository state and affected filesystem
  authority; the six exact legacy direct Git reads remain the only raw Git
  compatibility surface.
- JSON-RPC and MCP calls require the exact registered endpoint/server operation
  authority before provider metadata or input schemas can be used.
- Trusted Sinks delimit Runtime-mediated egress. The guarantee ends at the Sink;
  a trusted native executable or remote provider may perform later I/O outside
  the Runtime boundary.
- External-effect intent is durable before ambiguous provider boundaries.
  Uncertainty is preserved as evidence and finite-use authority is not restored
  without a narrow certified-not-started contract.
- Checkpoint restore revalidates current authority and reports external effects;
  it does not erase provider state or resurrect a committed revoke.
- Audit, event, operation, receipt, and FlowGraph records are evidence, never
  credentials. Direct store administrators and trusted Host components remain
  within the TCB.
- Semantic classifier output may veto or escalate but can never authorize. The
  machine path never installs `always_allow`.
- Remote access is Host-registered and configuration-bound. Models do not choose
  ad hoc URLs, credentials, subprocess transports, or trust roots.

Read the [Threat Model](docs/threat_model.md) for assets, adversaries, TCB,
guarantees, and non-goals; [Runtime Invariants](docs/invariants.md) maps current
claims to tests.

## Contributing, Evaluation, and Releases

Use the [Development Guide](docs/development.md) for environment setup, test
lanes, architecture checks, real-provider opt-ins, GUI validation, dependency
changes, and documentation rules. Generated `agent_outputs/`, benchmark runs,
credentials, local databases, and GUI build artifacts must not be committed.

Evaluation output is source-bound evidence, not a standing performance or safety
claim for another checkout. Preserve each suite's metadata, results, effects,
metrics, model/provider identity, and declared evidence level together, and use
the [Benchmark Contract](docs/benchmark.md) to interpret denominators or failure.

The [Release Status](docs/release_status.md) describes conditional scope and
required gates; it is not a pass receipt. Artifact building, signing, tagging,
uploading, yanking, or changing external release state requires explicit Human
authorization and the [Release Runbook](docs/releasing.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull-request expectations and
[SECURITY.md](SECURITY.md) for confidential vulnerability reporting. The project
is licensed under the terms in [LICENSE](LICENSE).
