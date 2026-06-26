# Development Guide

This guide covers local setup, regression checks, real Deno behavior, optional
real LLM paths, and documentation rules for Agent libOS contributors.

## Setup

Install dependencies:

```bash
uv sync --all-groups
```

Use frozen dependency resolution for artifact and CI-style checks:

```bash
uv sync --frozen --all-groups
```

Deno-backed tests run by default when `deno` is installed. If `deno` is absent,
tests marked `real_deno` skip with a clear pytest reason; use
`--skip-real-deno` only when a run intentionally excludes them. To validate and
run real Deno/TypeScript JIT tools from another binary, pass a runtime config
built with `dataclasses.replace(DEFAULT_CONFIG, tools=replace(...))`.

## Standard Checks

Run:

```bash
uv run python -m compileall agent_libos tests scripts experiments benchmarks
uv run python scripts/test_matrix.py --lane unit
uv run python scripts/test_matrix.py --lane security
uv run python scripts/test_matrix.py --lane runtime
uv run python scripts/check_test_invariants.py
uv run python scripts/test_matrix.py --lane gui
git diff --check
```

Run all deterministic Python lanes:

```bash
uv run python scripts/test_matrix.py --lane all
```

Use pytest-xdist workers for faster local Python feedback:

```bash
uv run python scripts/test_matrix.py --lane all --workers 4
uv run python scripts/test_matrix.py --lane runtime --workers auto
```

`--workers` applies only to Python lanes. The `runtime` and `all` lanes default
to bounded parallel execution with at most four workers and `--dist worksteal`,
which keeps CI runtime below the lane budget while balancing long SQLite and
runtime-reopen tests. Pass `--workers 1` for serial failure diagnosis, or set
`AGENT_LIBOS_TEST_WORKERS` / `AGENT_LIBOS_TEST_DIST` to override defaults in CI.
Run the `gui` lane separately because it writes shared frontend build artifacts.

Pytest cleans files created under the ignored `agent_outputs/` directory at the
end of each test session, while preserving anything that existed before the
session started. Use `--keep-agent-outputs` or set
`AGENT_LIBOS_KEEP_AGENT_OUTPUTS=1` when debugging generated files. To inspect or
clean already accumulated local output, run:

```bash
uv run python scripts/clean_agent_outputs.py
uv run python scripts/clean_agent_outputs.py --yes
```

Run a specific pytest lane with one of `unit`, `runtime`, `security`,
`self-evolution`, `providers`, or `benchmark`:

```bash
uv run python scripts/test_matrix.py --lane runtime
```

Useful smoke commands:

```bash
uv run agent-libos --help
uv run agent-libos checkpoint --help
uv run agent-libos skills --help
uv run agent-libos jsonrpc --help
uv run python experiments/run_benchmark.py --help
uv run python experiments/collect_metrics.py --help
```

Benchmark smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --output .benchmark_runs/docs-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/docs-smoke
```

`.benchmark_runs/` is ignored and should not be committed.

## Real LLM Smoke

Real LLM paths are opt-in because tokens are valuable.

Configure the host environment, or pass an explicit env file to scripts that
offer one. The runtime LLM client does not implicitly read a workspace `.env`.

```bash
OPENAI_BASE_URL=https://example-openai-compatible-endpoint/v1
OPENAI_LANGUAGE_MODEL=your-model
OPENAI_API_KEY=...
AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1
```

Useful optional variables:

- `OPENAI_API_MODE=responses|chat|auto`
- `OPENAI_TIMEOUT`
- `OPENAI_MAX_RETRIES`
- `OPENAI_STORE`
- `OPENAI_REASONING_EFFORT`
- `OPENAI_VERBOSITY`
- provider-specific `OPENAI_ENABLE_THINKING`

`OPENAI_BASE_URL` is optional for the OpenAI API. Custom OpenAI-compatible
endpoints require `AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1` or an explicit
`allow_custom_base_url=True` client construction.

Run a script smoke:

```bash
uv run python scripts/llm_write_goal_smoke.py
```

Run a benchmark smoke only with an explicit one-task limit:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --llm real --limit 1 --output .benchmark_runs/real-smoke
```

Every runtime LLM action-selection call must persist an `llm_calls` row with
provider ids, model/API mode, usage, errors, and bounded observability envelopes
for prompt, visible tools, output, tool calls, reasoning metadata, and raw
responses. Full prompt and raw provider payloads are intentionally not persisted
by default.

LLM providers are selected through host-configured named profiles. Processes
persist only `llm_profile_id`; the Runtime resolves that id for each quantum and
reads API keys from the profile's `api_key_env` environment variable. The
configured default profile preserves the existing `OPENAI_*` environment
behavior. Other named profiles do not inherit ambient provider/model
environment variables; set their profile fields explicitly when they should use
a non-default model or endpoint.

Set `config.llm.persist_full_io=True` to opt into full prompt, visible tool
schema, model output, tool call, reasoning, and raw response persistence. This
is intended for explicit local debugging or forensic runs because it can store
user data, object memory excerpts, and provider payloads in SQLite.

## Configuration Defaults

Non-secret runtime defaults live in `agent_libos.config.DEFAULT_CONFIG`.
`AgentLibOSConfig` uses Pydantic dataclass validation and fails fast when
numeric limits are negative, non-finite, inverted, or otherwise unsafe.

Current default groups include:

- runtime database and default ids,
- scheduler quantum, worker, drain, and shutdown limits,
- process resource budgets, usage accounting, and default cwd,
- LLM timeouts and provider compatibility knobs,
- tool limits and text encodings,
- filesystem and Object Memory size limits,
- Deno sandbox limits and JSR import allowlist,
- ObjectTask notification, owner-watch, and shutdown limits,
- shell policy allow/block lists,
- JSON-RPC endpoint manifest, timeout, and request/response limits,
- image registry limits,
- Object Memory and LLM context defaults,
- checkpoint snapshot limits,
- Skill package source, trust, resource, and `SKILL.md` limits,
- trusted startup Runtime Module manifests, hash trust, and registration limits,
- launcher presets,
- script defaults.

Do not scatter magic numbers in implementation code when a value affects
runtime behavior, policy, persistence, or test reproducibility. Add a typed
config default instead.

## Manifest YAML

Runtime YAML manifests and `SKILL.md` frontmatter are parsed through
`agent_libos.utils.yaml_loader.load_yaml_mapping`, which uses PyYAML's safe
loader plus a duplicate-key check. YAML syntax follows PyYAML, while the
runtime schema validators still restrict which fields and value shapes each
manifest accepts. Duplicate mapping keys are rejected so authority-bearing
manifests fail closed instead of silently overwriting earlier declarations.

## Documentation Rules

README is the entrypoint. Detailed implementation documentation belongs in
`docs/`.

When behavior changes, update the relevant doc and `docs/invariants.md` in the
same change. Do not describe future work as current behavior. Paper-facing
documentation should stay aligned with the fixed title:
`Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM
Agents`.

Current behavior must not claim:

- Python JIT compatibility,
- direct external framework adapters as trusted boundaries,
- real MCP/GitHub/provider integrations that are not implemented,
- provider-level compensation for rollbackable external side effects,
- Skill activation as a capability grant.

`agent_libos_design_doc.md` remains a historical archive and can be stale.
`plan.md` is a dated paper roadmap; keep it useful for planning, but do not use
it as the implementation reference for current command syntax or runtime
behavior.

## Adding Runtime Code

Preserve the boundary:

- model-facing tools call primitives;
- primitives perform Capability authorization, policy, approval, events, and audit;
- providers perform host effects only after primitive authorization;
- JIT tools access libOS only through syscalls;
- Skills change visibility and prompt materialization only;
- self-evolution mechanisms such as Skills, JIT tools, image registration,
  process exec, checkpoint forks, child processes, and JSON-RPC endpoint
  visibility must not imply resource authority or additional resource budget;
- Runtime Modules are trusted startup TCB extensions; they may register tools,
  images, syscalls, and provider hooks but must not be treated as process
  capabilities;
- JSON-RPC remote calls use registered endpoints and primitive capabilities
  rather than model-supplied URLs or secrets;
- checkpoint restore is scoped and append-only outside reconstructable state;
  provider-classified external effects are report-only in v1.

Prefer existing managers and primitives over new side channels. If a new host
effect is needed, add or extend a primitive and provider interface rather than
calling the host directly from a tool.

## Dependencies

Add runtime dependencies with:

```bash
uv add <package>
```

Add development dependencies with:

```bash
uv add --dev <package>
```

Commit both `pyproject.toml` and `uv.lock` after dependency changes.
