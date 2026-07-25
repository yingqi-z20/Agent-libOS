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
process identity + capability + data labels + Host Sink trust + primitive + audit
```

LLM-facing tools, Skills, JIT tools, image definitions, child processes,
checkpoints, and remote endpoint visibility are ergonomic affordances. They are
not the security boundary.
Process- or model-selected protected resource operations enter libOS primitive
or protected-operation SDK boundaries, where process identity, capabilities,
human approval, policy, provider containment, events, and audit records are
enforced. Trusted Runtime artifact publication is a narrow TCB exception: image
package materialization writes only the Runtime-owned private workspace through
Host filesystem APIs under durable publication recovery and resource
accounting. It does not grant a process a general filesystem bypass.

This project is still in active development. Current behavior is defined by
this README and the current references under `docs/`; historical design and
roadmap files are separated below and are not interface or security contracts.

## Current System

The implementation currently includes:

- Agent process lifecycle: `spawn`, `fork`, `exec`, `wait`, `signal`, `pause`,
  `resume`, and `exit`.
- Hierarchical process resource budgets for tool calls, LLM token usage,
  subprocess wall/CPU/RSS usage, filesystem bytes, JSON-RPC/MCP bytes, and
  Deno syscalls. Discrete counts/bytes/tokens are integers; runtime and
  subprocess wall/CPU seconds are continuous values. A charge publishes the
  full process-to-ancestor accounting chain, reservations, event, and audit as
  one transaction.
- Thread-backed process scheduling through `Runtime.run_until_idle()` and the
  async host wrapper `Runtime.arun_until_idle()`, so blocked quanta do not
  monopolize scheduler progress.
- Process-local working directories for filesystem and shell operations.
  Selecting a cwd, including an explicit child or PTY cwd, requires filesystem
  directory `read`; the directory state probe runs only after that authority
  and is covered by the filesystem external-effect intent.
- Optional Object-bound PTY sessions through the trusted `modules/pty` runtime
  module; when that module is loaded, `pty_create` returns an Object Memory
  `EXTERNAL_REF` handle, and read, write, resize, and close rights follow
  object capabilities. On POSIX, output reading and process-tree resource
  supervision use independent workers; concurrent lifecycle closes converge on
  one close transition on every supported backend. The current Windows backend
  uses optional `pywinpty`/ConPTY, has no Job Object or wall/CPU/RSS supervisor,
  and rejects a budgeted spawn that supplies `SubprocessLimits`. Install the
  optional `pty` extra from a source checkout/source distribution; the module
  itself is not included in the core wheel. See [docs/modules.md](docs/modules.md).
- Durable process message queues for IPC, including interrupt delivery.
- Object-bound background tool tasks that can notify processes through the
  same durable message queues, including optional owner-change watches, without
  exposing their runner child processes to the LLM scheduler.
- Human queue integration for typed questions and permission decisions.
  Permission responses explicitly choose `always_allow`, `always_deny`, or
  `ask_each_time`; approved questions require a non-empty string answer.
  Terminal prompt/read and automatic-response writes use structured pending
  effect intents. Their effect metadata stores only request/purpose plus
  length/hash observations, never raw prompts, answers, or provider exception
  text.
- Process-private Object Memory namespaces by default, with explicit shared
  namespaces available through capabilities.
- Structured Capability authority for filesystem, Git, shell, clock, human,
  process, image, checkpoint, skill, and Object Memory primitives, including
  typed resource matching, deny/ask/allow effects, one-shot grants,
  attenuation, revoke, and audit lineage.
- Runtime-enforced source-to-Sink data labels for LLM, Human, JSON-RPC, MCP,
  typed Git, filesystem writes, Shell/PTY input and session control, and process
  handoffs.
  Unmatched Sinks are untrusted/normal; Host-trusted Sinks accept data only
  within sensitivity and tenant/principal clearance, while conditional
  high-sensitivity sends require an exact one-shot release. See
  [docs/data_flow.md](docs/data_flow.md).
- Durable Host-authored Task Authority Manifests that compile launch authority,
  budgets, approval policy, and effect ceilings while treating image
  `required_capabilities` as declarations only. See
  [docs/task_authority_manifest.md](docs/task_authority_manifest.md).
- A Resource Provider Substrate for injectable filesystem, clock, shell, and
  human I/O backends, a pinned system-Git provider exposed as `Runtime.git`,
  plus JSON-RPC over HTTP and MCP client providers for pre-registered remote
  endpoints. Typed Git covers bounded inspection, local changes, managed
  worktrees, immutable patch Objects, existing remotes, and repository-local
  simulated pull requests without exposing arbitrary Git argv or URLs.
- Trusted startup Runtime Modules loaded from manifest-declared Python
  entrypoints before `Runtime.open()` returns. Modules can register tools,
  images, syscalls, provider hooks, and startup hooks, but do not grant process
  resource authority.
- A direct workflow entrypoint for users to run one tool bound in the selected
  image's complete process tool table through ToolBroker without invoking the
  LLM scheduler or consulting its narrower model projection.
- Runtime store persistence, backed by SQLite by default and optionally
  PostgreSQL, for process/object metadata, capabilities, messages, human
  requests, LLM calls, durable LLM wait generations and eligible Responses tool
  outputs, events, audit records, tools, Skill/JIT metadata, object tasks,
  JSON-RPC endpoints, image definitions/artifacts, Runtime Module load records,
  external effects, and scoped checkpoints.
- Host-only Explainable Operations with persistent causal trees across LLM,
  Tool, syscall, primitive, capability, Human, provider-effect, resource, event,
  and audit boundaries. Per-LLM Context Materialization Manifests retain Object
  selection reasons, versions, token counts, and hashes without copying payloads.
- Deno/TypeScript JIT tools that can access libOS only through `libos.syscall`.
  A dedicated supervisor establishes host-lifetime process-tree containment
  before Deno starts, so hard host termination cannot orphan untrusted code.
- Declarative Skills that can add prompt instructions, visible tools, and JIT
  candidates without granting resource authority. The shipped base, coding,
  review, and toolmaker Images use `metadata.tool_projection: skills`: a fresh
  process presents only `discover_skills`, `activate_skill`,
  `read_skill_resource`, `unload_skill`, and `process_exit` to the model. Their
  other static `default_tools` form the initial callable table and the ceiling
  for immutable built-in Skill projection, not an initial model-visible surface;
  an applicable built-in Skill projects its complete owned subset only after
  explicit activation. A separately registered Skill may expand both tables
  under its own trust and Skill-authority checks.
- Client-only JSON-RPC 2.0 over HTTP through registered endpoints, method
  capabilities, provider-classified external effects, audit, and checkpoints.
  Per-item registry authority is checked before metadata lookup; registry row,
  stale-grant invalidation, event, and audit changes commit atomically.
- Client-only MCP Tools through registered stdio or Streamable HTTP servers,
  tool capabilities, provider-classified external effects, audit, and resource
  accounting, with the same authority-before-lookup and transactional registry
  semantics.
- A deterministic runtime-safety benchmark harness with 32 checked-in schema-v1
  tasks, including a self-evolution subset, baselines, evidence-backed
  side-effect oracle, fail-closed output validity, and explicit metric
  denominators.
- A checked-in practical-workflow suite with exactly two evidence levels:
  `native-live` and `modeled`. Native rows never fall back to modeled evidence.

## Documentation

Start here, then read the deeper references as needed:

- [docs/release_status.md](docs/release_status.md): current-version readiness,
  validation outcomes, and remaining environment boundaries.
- [docs/python_api.md](docs/python_api.md): public Python imports, Runtime
  lifecycle, manager properties, sync/async usage, exceptions, and
  compatibility boundaries.
- [docs/architecture.md](docs/architecture.md): runtime layers, provider
  substrate, and the tool/primitive boundary.
- [docs/threat_model.md](docs/threat_model.md): assets, adversaries, TCB,
  trust boundaries, guarantees, non-goals, and severity calibration.
- [docs/runtime_model.md](docs/runtime_model.md): process lifecycle, scheduler,
  cwd, human queue, IPC, fork/spawn/exec, and waits.
- [docs/explainable_operations.md](docs/explainable_operations.md): operation
  trees, evidence completeness, Context Manifests, redaction, CLI, and GUI/API
  queries.
- [docs/protected_operation_sdk.md](docs/protected_operation_sdk.md): stable
  provider-operation contracts, phases, failure semantics, evidence, and
  extension examples.
- [docs/providers.md](docs/providers.md): provider inventory, authority/effect
  contracts, containment limits, and extension checklist.
- [docs/git.md](docs/git.md): the typed `Runtime.git` provider/primitive,
  model tools, state tokens, capabilities, hardening, remotes, patch Objects,
  managed worktrees, and simulated pull requests.
- [docs/capabilities.md](docs/capabilities.md): resource naming, rights,
  one-shot grants, human approval, shell policy, and filesystem containment.
- [docs/data_flow.md](docs/data_flow.md): label integrity, Host Sink trust,
  exact release, exit coverage, process identity domains, persistence, and
  guarantee boundaries.
- [docs/object_memory.md](docs/object_memory.md): namespaces, object rights,
  file/object bridge, context materialization, and payload persistence.
- [docs/tools_and_jit.md](docs/tools_and_jit.md): built-in tools,
  ToolBroker, Deno/TypeScript JIT tools, syscall protocol, and sandbox rules.
- [docs/modules.md](docs/modules.md): trusted startup Runtime Module
  manifests, trust model, registration surfaces, CLI, and checkpoint behavior.
- [docs/jsonrpc.md](docs/jsonrpc.md): client-only JSON-RPC endpoint registry,
  capability resources, tools, syscalls, and checkpoint behavior.
- [docs/mcp.md](docs/mcp.md): client-only MCP server registry, tools-only v1
  scope, capability resources, tools, syscalls, and checkpoint behavior.
- [docs/skills.md](docs/skills.md): standard `SKILL.md` packages,
  workspace/global sources, trust, activate/unload semantics, bundled JIT
  tools, and `swe-agent`.
- [docs/checkpoints.md](docs/checkpoints.md): scoped snapshots, restore, fork,
  replay diagnostics, retained runtime history, and external-effect reporting.
- [docs/storage.md](docs/storage.md): transaction rollback/poison semantics,
  Object payload durability, schema recovery, and active-runtime leases.
- [docs/evidence_payload_retention.md](docs/evidence_payload_retention.md):
  explicit, auditable LLM/external-effect payload retention tiers and safety
  exclusions.
- [docs/configuration.md](docs/configuration.md): load precedence, field-level
  config inventory, secrets, and bounded-window defaults.
- [docs/cli.md](docs/cli.md): stable CLI command reference and examples.
- [docs/gui.md](docs/gui.md): Electron desktop console, local GUI server,
  HTTP/SSE APIs, same-build contract boundary, and development commands.
- [docs/gui_api_schema.json](docs/gui_api_schema.json): versioned JSON Schema
  subset for snapshots, errors, and confirmed high-risk GUI mutations.
- [docs/benchmark.md](docs/benchmark.md): runtime-safety and practical-workflow
  evaluation contracts, scale gates, outputs, and metrics.
- [docs/mini_swe_agent_image.md](docs/mini_swe_agent_image.md): package-only
  `mini-swe-agent` image behavior and known interface differences.
- [docs/development.md](docs/development.md): setup, tests, real LLM smoke,
  configuration defaults, and contribution rules.
- [docs/support_matrix.md](docs/support_matrix.md): declared support, CI-covered
  environments, and explicit platform/provider release gates.
- [docs/invariants.md](docs/invariants.md): current invariant-to-test map.
- [docs/artifact_anonymity.md](docs/artifact_anonymity.md): anonymous artifact
  hygiene checklist.
- [docs/paper_thesis.md](docs/paper_thesis.md): current paper thesis and
  non-goals.
- [benchmarks/runtime_safety/schema.md](benchmarks/runtime_safety/schema.md):
  benchmark task schema v1 and run-output schema v2.
- [benchmarks/external_effect_recovery/README.md](benchmarks/external_effect_recovery/README.md):
  100k CI and one-million-record external-effect recovery scale profiles.
- [benchmarks/runtime_publication_recovery/README.md](benchmarks/runtime_publication_recovery/README.md):
  10k CI runtime-publication reopen and reconciliation scale gate.
- [benchmarks/practical_agent_workflows/README.md](benchmarks/practical_agent_workflows/README.md):
  native-live and modeled practical-workflow evidence contract and report
  schema.
- [benchmarks/builtin_tool_skills/README.md](benchmarks/builtin_tool_skills/README.md):
  opt-in paid paired evaluation of Skill projection versus the full tool schema.
- [benchmarks/long_horizon_agent/README.md](benchmarks/long_horizon_agent/README.md):
  opt-in paid long-horizon evaluation across restart, follow-up messages, and
  prompt-injection pressure.
- [experiments/agentdojo/README.md](experiments/agentdojo/README.md): isolated
  AgentDojo harness, its frozen Python 3.11–3.12 environment, deterministic CI
  scope, and opt-in real-model evaluation workflow.
- [AGENTS.md](AGENTS.md): repository structure, testing, security, and
  contribution guidance for local agents and contributors.

### Historical references (not current contracts)

These files are retained for design and project history. Do not use them to
infer current commands, interfaces, security guarantees, or release evidence:

- [agent_libos_design_doc.md](agent_libos_design_doc.md): historical design
  archive containing planned and superseded interfaces.
- [plan.md](plan.md): dated paper-submission roadmap, not an implementation
  reference.
- [docs/prelaunch_hardening_report.md](docs/prelaunch_hardening_report.md):
  historical, source-bound subsystem review and validation snapshot, not the
  current release-status source.

## Quick Start

Prerequisites are Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). The
typed Git provider and full test matrix require system Git 2.26 or newer. GUI
development requires Node `^20.19.0` or `>=22.12.0` and npm 8 or newer; CI uses
Node 24. Deno is optional unless running real TypeScript/JIT coverage.

Install dependencies:

```bash
uv sync --frozen --all-groups
```

### Distribution artifacts

The Python wheel contains the core `agent_libos` package, its immutable built-in
Tool Skills, and the `agent-libos`, `agent-libos-gui-server`, and explicit
offline `agent-libos-migrate-tool-groups` console entrypoints. Release
validation parses all 26 built-in Skill packages and their
99 uniquely owned tools from both the wheel and source archive. Repository-level
assets such as the optional PTY Runtime Module, bundled example Skill and Image,
benchmarks, tests, and documentation are distributed with the Python source
archive and source checkout, not installed into the core wheel. The
Electron sources are repository-checkout assets and are validated by their
separate GUI lane. A wheel installation may load separately supplied modules,
Skills, Images, and configuration through their normal explicit paths.

Build and validate both release artifacts with:

```bash
uv build --clear --out-dir dist
uv run python scripts/check_release_artifacts.py dist
```

Validate the installed console entrypoints from an artifact, rather than only
from the source checkout, in a disposable environment:

```bash
uv venv /tmp/agent-libos-wheel-check
uv pip install --python /tmp/agent-libos-wheel-check/bin/python dist/*.whl
uv pip check --python /tmp/agent-libos-wheel-check/bin/python
/tmp/agent-libos-wheel-check/bin/python -c "from agent_libos.skills import get_builtin_skill_catalog; assert len(get_builtin_skill_catalog().list()) == 26"
/tmp/agent-libos-wheel-check/bin/agent-libos --help
/tmp/agent-libos-wheel-check/bin/agent-libos-gui-server --help
/tmp/agent-libos-wheel-check/bin/agent-libos-migrate-tool-groups --help

uv venv /tmp/agent-libos-sdist-check
uv pip install --python /tmp/agent-libos-sdist-check/bin/python dist/*.tar.gz
uv pip check --python /tmp/agent-libos-sdist-check/bin/python
/tmp/agent-libos-sdist-check/bin/python -c "from agent_libos.skills import get_builtin_skill_catalog; assert len(get_builtin_skill_catalog().list()) == 26"
/tmp/agent-libos-sdist-check/bin/agent-libos --help
/tmp/agent-libos-sdist-check/bin/agent-libos-gui-server --help
/tmp/agent-libos-sdist-check/bin/agent-libos-migrate-tool-groups --help
```

Use fresh paths for each check. On Windows, replace `/tmp/...` with a disposable
directory and use the environment's `Scripts` directory instead of `bin`. The
source archive additionally contains repository-level examples, benchmarks,
documentation, and the optional PTY module that are intentionally absent from
the core wheel.

### Upgrading stores that contain Tool Groups

Legacy stores with the removed Tool Group image metadata fail closed until they
are migrated to built-in Tool Skills. Stop every Runtime using the store, run
the installed migration once without `--apply`, and review its report:

```bash
agent-libos-migrate-tool-groups /path/to/runtime.sqlite
```

The default is a complete transactionally rolled-back dry run; startup never
runs this content migration automatically. If the report is correct, repeat the
same command with `--apply` to commit it. Pass `--config <path>` for a
non-default config overlay or supply a PostgreSQL URI in place of the SQLite
path. See [Tool Broker, Skills, and JIT Tools](docs/tools_and_jit.md#on-demand-tool-skills)
for conversion rules and immutable-artifact behavior.

Run tests:

```bash
uv run python scripts/test_matrix.py --lane unit
uv run python scripts/test_matrix.py --lane security
uv run python scripts/test_matrix.py --lane runtime
uv run python scripts/check_architecture.py
uv run python scripts/check_test_invariants.py
```

The `runtime`, `security`, `self-evolution`, `providers`, and `all` lanes use
bounded pytest-xdist parallelism by default to keep feedback time under
control. Pass `--workers 1` for serial failure diagnosis, or `--workers N` /
`--workers auto` to override the worker count for any Python lane. Pass
`--durations 25` to report the slowest tests. Standard lanes deselect tests
marked `postgres`; the dedicated PostgreSQL service gate runs those tests with
`pytest -m postgres --run-postgres --fail-on-skip`. The GUI lane builds shared
frontend artifacts and should be run separately after `npm --prefix gui install`.
Pytest removes files created under ignored `agent_outputs/` at the end of a
test session; use `--keep-agent-outputs` when debugging generated files.

Run the deterministic local demo:

```bash
uv run agent-libos demo
```

The demo overwrites `agent_outputs/demo_patch_preview.txt` below the Runtime
workspace. Preserve or move an existing file at that path before running it.

Run the Electron GUI in development mode:

```bash
npm --prefix gui install
npm --prefix gui run electron:dev
```

The GUI starts a local `agent-libos-gui-server`, subscribes to runtime events,
and provides responsive user and operator workspaces for concurrent messages,
interrupts, human approvals, scheduler control, checkpoints, capability and
Skill administration, Object Tasks, image selection/registration/commit,
JSON-RPC/MCP registries, audit/Explain inspection, and LLM call visibility.
The Electron launcher fills missing backend environment values from this
checkout's `.env`; values already inherited by Electron take precedence. This
GUI-specific launcher behavior does not apply to the generic CLI or library API.

The demo does not call a real model. It exercises process spawn/fork, Object
Memory, Deno/TypeScript JIT validation when Deno is available, checkpointing,
capability denial before grant, human approval, filesystem write, final report
object creation, and audit trace generation.

Run a small deterministic benchmark smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/m1-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/m1-smoke
```

The benchmark defaults to mock/planned actions and does not spend model tokens.
`--require-all-passed` is the release/smoke gate; without it, expected oracle
failures are written for comparative analysis without forcing a non-zero exit.
Real-model benchmark smoke is opt-in, must select exactly one task after all
filters, and supports only one or more Agent-libOS-family runners; wrapper and
sandbox baselines cannot use `--llm real`.

The historical deterministic snapshot was produced from clean
source snapshot `c03a4ec764e02bd4df59e2769edeb1278d5ea545`; its ignored local
artifact is `.benchmark_runs/release-c03a4ec`. For that source snapshot it is
valid with 28/28 task
success and
safety pass, 122 normalized effects, unauthorized performed effects `0/97`,
allowed denials `0/97`, and zero unknown outcomes/classifications. Its
`metadata.json` SHA-256 is
`7ef7b0054f1e4fbd2bcb9b33e803016e62010254a122dffa8c692f0837ba6b54` and its
`metrics.json` SHA-256 is
`f6b3b0aa5e2a403c3ed0a7c848dcbccffa7faabe5eda7edf6cfe26ebccde53b6`.
That artifact is not evidence for the current tree: its counts do not carry
over across history consolidation or later runtime changes unless content
identity is proved and a new validation artifact records that proof. It uses
historical run-output schema v1; the current v2 collector intentionally rejects
it, so it remains archival evidence only.
The two rate denominators are qualified effect populations, not task counts,
and missing/unknown evidence invalidates rates instead of being inferred from
`result.ok`. The ignored artifact must be packaged separately. See the
[current release status](docs/release_status.md) and
[benchmark contract](docs/benchmark.md).

Run the practical workflow evidence suite separately:

```bash
uv run python experiments/run_practical_evaluation.py \
  --output .benchmark_runs/practical/report.json
```

Its two evidence-level labels, `native-live` and `modeled`, are part of the
result contract. `native-live` requires real ToolBroker, state, external-effect,
and Explain evidence; it never falls back to a modeled success.

## Persistent Runtime

Use `--db` to keep runtime state in a persistent store. A filesystem path uses
SQLite and remains the default local option. The `spawn` command prints the new
pid; substitute it for `<pid>` below. Running a quantum invokes the configured
real LLM, consumes provider tokens, and requires the model/API environment
described in the next section. Image requirement declarations do not grant
authority, so grant the exact README read capability before running this goal:

```bash
uv run agent-libos --db .agent_libos.sqlite init
uv run agent-libos --db .agent_libos.sqlite spawn --image coding-agent:v0 --goal "Summarize README.md"
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite run --max-quanta 10
uv run agent-libos --db .agent_libos.sqlite processes
uv run agent-libos --db .agent_libos.sqlite resources <pid>
uv run agent-libos --db .agent_libos.sqlite audit
uv run agent-libos --db .agent_libos.sqlite workflow run get_working_directory
```

PostgreSQL is opt-in. Install the extra dependency and either pass a DSN with
`--db` or configure `runtime.store_backend: postgres` together with
`runtime.store_dsn`; keep DSNs in environment-specific config or environment
variables so credentials are not committed. A PostgreSQL backend without
`runtime.store_dsn` is rejected at config load time:

```bash
uv sync --frozen --all-groups --extra postgres
uv run agent-libos --db "$AGENT_LIBOS_POSTGRES_DSN" init
```

Both backends implement the same runtime store contract. Process metadata,
capabilities, audit/events, messages, human requests, LLM call records,
checkpoints, and registered tools/images/skills are durable store records.
Ordinary Object Memory payloads remain runtime-only; current writes store a
runtime-memory marker in the object table, and marker rows whose payload cache
cannot be reconstructed are released fail-closed on reopen instead of being
treated as real payloads. An accepted legacy row from an older development
build may still contain full JSON payload data; migrate or recreate such a
store before claiming marker-only historical retention. See
[docs/storage.md](docs/storage.md#transaction-model).
Persistent stores take an active-runtime lease: SQLite uses a secure sidecar
`flock` where available or an exclusive database lock as fallback, and
PostgreSQL uses a session advisory lock. Two writable `Runtime` instances
cannot concurrently open the same store. Closing the first runtime releases the
lease and permits a later reopen.

SQLite resolves both its connection and lease from the canonical database
path. On platforms with `fcntl` and `O_NOFOLLOW`, the sidecar is opened
no-follow, verified as the same regular-file inode before use, and protected by
`flock`. On that secure POSIX path, the database, lease, journal, WAL, and SHM
files are created or tightened owner-only (`0600`). Where that path is
unavailable, SQLite uses its kernel-managed exclusive database lock instead of
trusting a stale sidecar. PostgreSQL advisory keys are scoped to the current
database and schema. Store transactions also fail closed: commit or
savepoint-release failure triggers rollback, and a rollback failure poisons and
closes the store rather than allowing further reads or writes. See
[docs/storage.md](docs/storage.md).

Omitting `--max-quanta` uses `runtime.run_until_idle_max_quanta`. Its default
`null` value runs until the Runtime becomes idle; configure that setting or pass
an explicit flag when a bounded run is required.

`workflow run <tool>` spawns a fresh process from the default image, calls one
tool from its complete process tool table, persists the result object, and exits
that process. This Host-directed path does not consult the model tool projection
and does not make the selected tool model-visible. Pass `--image <image_id>` to
use another image's complete table. It does not bypass primitive capability
checks, resource budgets, human approval, or audit.

Every LLM action-selection call is persisted as an `llm_calls` row with
provider ids, model/API mode, token usage when available, errors, and bounded
observability envelopes for prompts, visible tool schemas, model output, tool
calls, reasoning metadata, and raw provider responses. Full prompts, visible
tool schemas, model outputs, tool calls, reasoning metadata, and raw provider
payloads are stored by default for self-evolution training and fine-tuning
pipelines. Deployments that rely on this default should disclose that use in
the user agreement because it may include sensitive prompt, tool, reasoning, and
provider payload data; set `llm.persist_full_io: false` in the runtime config
when a user or operator opts out of full LLM input/output retention. That
opt-out cannot run an `image_only` Image (the default for custom packages),
because transparent replay requires a lossless durable transcript; execution
fails before provider dispatch. Use a Runtime-owned prompt mode when redacted
write-time persistence is required.

For deployments that keep full provider I/O initially, the optional payload
retention maintenance API can later reduce eligible terminal rows through
content-free summary and hash-only tiers. It is disabled by default, never runs
during startup, and never trims the active `image_only` transcript head,
compatible Responses-continuation anchors, or process-result recovery payloads.
See [Evidence and LLM Payload Retention](docs/evidence_payload_retention.md).

```bash
uv run agent-libos --db .agent_libos.sqlite llm-calls --pid <pid>
```

## Real LLM Configuration

The legacy `OPENAI_*` environment variables configure the profile whose id is
`llm.default_profile_id`. Export them into the Host process environment for a
single-profile setup:

```bash
export OPENAI_BASE_URL=https://example-openai-compatible-endpoint/v1
export OPENAI_LANGUAGE_MODEL=your-model
export OPENAI_API_KEY=...
export AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1
```

For multiple models or endpoints, define named profiles in a config overlay and
keep each secret in the environment variable named by `api_key_env`:

```yaml
llm:
  default_profile_id: default
  profiles:
    default: {}
    review:
      base_url: https://review-provider.example/v1
      model: review-model
      api_key_env: REVIEW_API_KEY
      api_mode: chat
      allow_custom_base_url: true
```

Select a configured profile when spawning or execing a process:

```bash
uv run agent-libos --config agent-config.yaml --db .agent_libos.sqlite \
  spawn --goal "Review the workspace" --llm-profile review
```

For a root spawn, an explicit `--llm-profile` wins over the image default and
then `llm.default_profile_id`; exec and child creation otherwise retain or
inherit the current process profile. Non-default named profiles do not inherit
ambient endpoint, model, or provider-policy `OPENAI_*` values, so declare those
fields explicitly. Every profile reads its API key only from `api_key_env`. See
[Configuration Reference](docs/configuration.md#effective-llm-profile-precedence)
for the exact precedence and supported fields.

The Runtime and generic CLI do not implicitly load a workspace `.env`. If these
values are stored as plain `KEY=value` lines in an untracked `.env`, load it in
the shell or invoke the CLI explicitly with
`uv run --env-file .env agent-libos ...`. The repository's
`scripts/run_coding_agent.py` launcher is a separate convenience path that does
support an env file. PowerShell users can set the same inherited variables with
`$env:NAME = "value"`.

The client uses the OpenAI Python SDK. It uses the Responses API for
OpenAI-hosted models by default and falls back to Chat Completions for custom
OpenAI-compatible `base_url` providers. Custom OpenAI-compatible endpoints
require `AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1`; official OpenAI endpoints do
not. Set `OPENAI_API_MODE=responses` or `OPENAI_API_MODE=chat` to force a mode.

Optional knobs include `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`, `OPENAI_STORE`,
`OPENAI_REASONING_EFFORT`, `OPENAI_VERBOSITY`, and provider-specific
`OPENAI_ENABLE_THINKING`.

Provider-side Responses storage/chaining policy is opt-in: the defaults remain
`store=false` and `responses_previous_response_id=false`. The current
AgentProcess executor nevertheless stays stateless when those settings are
enabled: it sends each complete locally rebuilt snapshot without
`previous_response_id`, records that full-snapshot disablement in the LLM call,
and does not create provider-chain tool-output rows. `store=true` can still
increase provider-side retention. The low-level `LLMClient` sends an explicitly
supplied `previous_response_id` only to the official Responses endpoint and only
with provider storage enabled. Separately, `image_only` replays paired tool
calls and outputs as a provider-native transcript without relying on
provider-side state: Responses receives `function_call`/
`function_call_output` items, while Chat receives assistant/tool messages.
Durable waiting actions remain protected by per-generation resume tokens
and non-replayable claims. See
[docs/development.md](docs/development.md#real-llm-smoke).

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
uv run agent-libos --db .agent_libos.sqlite exec review-agent:v0 "Review README.md" --pid <pid> --run
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
uv run agent-libos --db .agent_libos.sqlite jsonrpc register <path-to-endpoint-manifest.yaml>
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> jsonrpc:demo-weather:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite jsonrpc call <pid> demo-weather forecast --params-json '{"city":"Beijing"}'
```

Create the manifest from the complete schema/example in
[docs/jsonrpc.md](docs/jsonrpc.md); the angle-bracket path is a user-supplied
file, not a file shipped at the repository root.

Register and call a preconfigured MCP tool:

```bash
uv sync --frozen --all-groups --extra mcp
uv run agent-libos --db .agent_libos.sqlite mcp register <path-to-server-manifest.yaml>
uv run agent-libos --db .agent_libos.sqlite mcp inspect demo-mcp
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> process:spawn --rights write
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> <stdio_authority_resource-from-inspect> --rights execute
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> mcp:demo-mcp:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite mcp call <pid> demo-mcp forecast --arguments-json '{"city":"Beijing"}'
```

Create the manifest from [docs/mcp.md](docs/mcp.md); the angle-bracket path is
user supplied. The `mcp` extra is not installed by the core Quick Start command.

The `process:spawn` and exact `mcp_stdio:<sha256>` grants are required only for
stdio servers. Copy `stdio_authority_resource` from `mcp inspect`; do not derive
it from a model-supplied command or replace it with a wildcard. Streamable HTTP
servers need the exact MCP tool capability but do not spawn a local server.

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
- Skills and JIT tools do not grant filesystem, Git, shell, human, object,
  process, image, checkpoint, JSON-RPC, or MCP remote authority.
- JIT syscalls bypass the LLM-facing tool table but not primitive capability
  checks, permission policy, human approval, or audit.
- The local shell provider is command/environment/cwd/resource mediation, not a
  general OS sandbox for hostile native binaries. Strong hostile-code isolation
  requires a container, WASM, or service provider with an explicit deployment
  trust boundary.
- Deno JIT validation rejects every static import/re-export (including `jsr:`,
  `npm:`, `node:`, `http:`, `https:`, and `file:`) and dynamic `import()`.
  Runtime execution is no-permission and cached-only, so a tool call cannot
  implicitly fetch remote code. `tools.deno_jsr_allowlist` is retained metadata,
  not an import-policy exception.
- JSON-RPC and MCP calls gate on endpoint/method or server/tool capability
  resources before loading provider metadata or input schemas, so missing call
  authority cannot enumerate registered manifests.
- Git calls operate only on the configured, runtime-pinned workspace repository or a trusted
  managed worktree. Mutations require a prior state token plus Git and affected
  filesystem authority; destructive and remote-ref-rewriting operations bind
  one-use Human approval to exact state/OIDs. Exactly six directly invoked
  Shell/PTY Git inspection argv forms are accepted and hardened even under an
  always-allow shell policy. Any recognized transparent executable-launcher
  wrapper around Git is rejected before shell policy or Human approval. As with
  other direct I/O, an authorized general interpreter or native program remains
  a host-user process and is outside that argv-only compatibility boundary.
- Human approval is part of a primitive/syscall. Model-facing Tool and JIT
  callers see a final success or final failure, not a public pending/retry
  protocol. A direct Python Host call may instead raise
  `HumanApprovalRequired` or `HumanResponseRequired` with the durable
  `request_id`; see the Python API contract. Run-local automatic decisions are
  isolated across concurrent scheduler workers, and terminal process states
  cancel pending requests.
- Runtime-mediated egress requires both ordinary operation authority and Sink
  clearance. Trusted Sink configuration cannot grant a capability, bypass a
  Task Authority effect ceiling or budget, or be forged by a model/child
  manifest. Conditional release is bound to the exact Sink, trust generation,
  manifest, source versions, labels, payload hash, operation, and target state;
  untrusted Sinks cannot be elevated above `normal` by Human approval.
- The data-flow guarantee ends at the mediated Sink. Trusting Shell, PTY, or MCP
  stdio means the Host trusts that executable to receive the payload; it is not
  OS network/filesystem isolation and does not control later native I/O or Sink
  forwarding. Trusted modules/providers, Host administrators, and direct
  database writes remain outside this boundary.
- When the optional PTY module is loaded, PTY sessions are host runtime
  resources bound to mutable Object Memory `EXTERNAL_REF` handles. Shell policy
  authorizes creation; object read, write, and delete rights authorize read,
  resize, and close; and `pty_write` additionally requires the original session
  owner so delegated object write rights cannot drive an existing shell.
  Finite-use object rights for write/resize/close are reserved until the PTY
  provider boundary is known to have started. Automatic child-exit cleanup
  persists a close intent before reading exit state or closing the handle.
  Runtime shutdown or object release closes the host PTY, and a failed
  post-spawn setup closes the handle and removes the object before returning
  failure. PTY fork drops `EXTERNAL_REF` handles rather than cloning provider
  resources.
- `process.exit` and `process.exec` are ordinary syscalls from TypeScript. The
  runtime applies lifecycle changes after the JIT tool returns its normal tool
  result.
- Files under ignored `agent_outputs/` are generated workspace output, not a
  control channel. Agent lifecycle actions such as mini-swe-agent submission must
  come from explicit tool/syscall arguments, not parsed stdout or file content.
- Checkpoint restore covers reconstructable process-subtree state and captured
  image registry metadata needed by that state. It does not delete append-only
  audit/events/LLM calls or roll back filesystem, Git/worktree/remote state,
  shell, image-package source, network, or provider side effects. Ownership,
  not borrowed MemoryView
  reachability, defines the destructive Object scope; restore/fork revalidate
  current capability state so a committed revoke is not resurrected.
- Checkpoint-derived images capture internal reconstructable runtime state, not
  external provider state. Their required capabilities are declarations and are
  not granted automatically at spawn or exec.
- Providers classify successful external effects as `irreversible`,
  `rollbackable`, or `no_rollback_required`. Filesystem mutations, clock,
  shell, and PTY spawn reserve finite-use authority before the provider
  boundary. `ProviderEffectNotStarted` permits intent abandonment only when no
  completed earlier phase mutated provider state, observed information, or
  committed authority (the phase default); ambiguous failures consume authority
  and persist an `unknown` effect.
  Filesystem/clock/shell, human output and terminal I/O, PTY
  spawn/write/resize/close, and live JSON-RPC/MCP calls persist a pending
  unknown intent before the
  provider and conditionally finalize that same `effect_id`, so a post-provider
  crash cannot erase uncertainty or a duplicate settlement create extra final
  rows. Failed post-spawn Object publication remains unknown even after cleanup.
  Remote JSON-RPC/MCP intents are durable before non-local DNS; exact remote and
  auxiliary stdio authority is reserved together, and a DNS observation means
  a later certified-not-started transport can no longer erase the information
  flow or restore the use.
  Checkpoint restore reports all classes with
  `restore_external_policy="report_only"`.
- Resource Provider Substrate backends perform host effects, but primitives own
  capability checks, policy decisions, events, and audit.
- Audit rows are append-only through RuntimeStore APIs. External-effect intent
  rows instead move through guarded prepare/finalize/retention transitions while
  retaining their causal identity and outcome history. Neither contract is
  tamper-proof against a host administrator with direct database write access;
  deployments needing independent integrity must add signed or remote
  append-only evidence outside that trust boundary.

See [docs/invariants.md](docs/invariants.md) for test coverage.

## Development

Run the standard local checks:

```bash
uv sync --frozen --all-groups
npm --prefix gui install
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/test_matrix.py --lane all --workers 4
uv run python scripts/check_architecture.py
uv run python scripts/check_test_invariants.py
uv run python scripts/check_protected_operations.py
uv run python scripts/test_matrix.py --lane gui
git diff --check
```

Use `uv run python scripts/clean_agent_outputs.py` to dry-run cleanup of
already accumulated local outputs, and add `--yes` to delete them.

Deno-backed tests run by default when `deno` is installed and skip with a clear
pytest reason when it is missing. Use `--skip-real-deno` only when you
intentionally want to exclude tests marked `real_deno`. To validate and run
real Deno/TypeScript JIT tools from another binary, pass a runtime config built
with `dataclasses.replace(DEFAULT_CONFIG, tools=replace(...))`.

Runtime defaults live in `agent_libos.config.DEFAULT_CONFIG`, including
scheduler quantum, worker, drain, and shutdown limits; process budgets; image
ids; workspace namespace; tool limits; filesystem/Object Memory size limits;
Deno sandbox limits; ObjectTask notification and shutdown limits; reserved JSR
import-allowlist metadata (static imports remain rejected); shell policy lists;
launcher presets; Skill defaults; and
checkpoint defaults. Optional modules such as `modules/pty` keep their own
module-local settings outside `AgentLibOSConfig`.
`AgentLibOSConfig` is validated at construction time, so invalid or inverted
bounds fail before a Runtime starts.
See [docs/configuration.md](docs/configuration.md) for precedence and the
field-level inventory, including MCP and bounded event-window settings.

Add runtime dependencies with `uv add <package>` and development dependencies
with `uv add --dev <package>`. Commit both `pyproject.toml` and `uv.lock` after
dependency changes.
