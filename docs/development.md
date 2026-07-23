# Development Guide

This guide covers local setup, regression checks, real Deno behavior, optional
real LLM paths, and documentation rules for Agent libOS contributors.

## Setup

Use Python 3.11–3.14 and uv. The full matrix also requires system Git 2.26 or
newer. GUI work requires Node `^20.19.0` or `>=22.12.0` with npm 8 or newer (CI
uses Node 24). Deno is optional unless validating real TypeScript/JIT execution.

Install dependencies:

```bash
uv sync --all-groups
npm --prefix gui install
```

Use frozen resolution for the locked runtime/development dependency graph in
artifact and CI-style checks:

```bash
uv sync --frozen --all-groups
npm --prefix gui ci
```

`uv sync --frozen` does not freeze the isolated PEP 517 build backend:
`pyproject.toml` currently requests an unconstrained `hatchling`. Do not describe
`uv build` as build-backend reproducible unless a build constraint or pinned
backend is added separately.

Deno-backed tests run by default when `deno` is installed. If `deno` is absent,
tests marked `real_deno` skip with a clear pytest reason; use
`--skip-real-deno` only when a run intentionally excludes them. To validate and
run real Deno/TypeScript JIT tools from another binary, pass a runtime config
built with `dataclasses.replace(DEFAULT_CONFIG, tools=replace(...))`.

## Standard Checks

Run:

```bash
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/test_matrix.py --lane unit
uv run python scripts/test_matrix.py --lane security
uv run python scripts/test_matrix.py --lane runtime
uv run python scripts/check_architecture.py
uv run python scripts/check_test_invariants.py
uv run python scripts/check_protected_operations.py
uv run python scripts/test_matrix.py --lane gui
git diff --check
```

Run all deterministic Python lanes:

```bash
uv run python scripts/test_matrix.py --lane all
```

Optional provider gates are excluded unless selected explicitly:

```bash
uv sync --frozen --all-groups --extra mcp
uv run python scripts/test_matrix.py --lane providers --run-mcp
uv run python scripts/test_matrix.py --lane all --run-real-llm
uv sync --frozen --all-groups --extra postgres
uv run python -m pytest -m postgres --run-postgres
```

The real-LLM command may consume paid provider tokens and requires the Host
environment described below. PostgreSQL requires a configured test service.

Use pytest-xdist workers for faster local Python feedback:

```bash
uv run python scripts/test_matrix.py --lane all --workers 4
uv run python scripts/test_matrix.py --lane runtime --workers auto
```

`--workers` applies only to Python lanes. The `runtime`, `security`,
`self-evolution`, `providers`, and `all` lanes default to bounded parallel
execution with at most four workers and `--dist worksteal`, which balances long
persistence and runtime-reopen tests. Pass `--workers 1` for serial failure
diagnosis, or set `AGENT_LIBOS_TEST_WORKERS` / `AGENT_LIBOS_TEST_DIST` to
override defaults. Pass `--durations 25` to report the slowest tests.
`--max-lane-seconds` is a hard process-tree timeout for each selected command;
the Python `all` lane is one command, while the GUI test, typecheck, and build
commands each receive the full timeout independently. Timeout exits with status
124 after terminating the process group/tree. Standard lanes deselect
`postgres` tests because the PostgreSQL CI
service runs them separately with `pytest -m postgres --run-postgres`. CI runs
each standard lane with a 360-second process deadline and a 15-minute outer
step deadline. Run the `gui` lane separately; it cleans Electron output before
production compilation, excludes generated `dist-electron` files from Vitest,
and never emits test files into the production Electron tree. Install GUI
dependencies first with `npm --prefix gui install`.

The architecture check covers the core package and repository-level Runtime
Modules. It rejects new lower-layer Runtime/API imports,
Runtime-as-service-locator access (including local aliases), and
cross-component private access, including dependencies stored in
underscore-prefixed fields. Literal-name `getattr` access is analyzed as the
equivalent ordinary attribute access. The check also ratchets existing
long-function and complexity hotspots, the explicit composition-cycle binding
set, and Runtime component declarations; stale ceilings are rejected when debt shrinks. Do not
refresh the allowlist merely to make growth pass; use `--print-baseline` to
inspect it, then lower or remove only entries justified by the current tree.

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
uv run python experiments/run_practical_evaluation.py --help
uv run python experiments/collect_metrics.py --help
```

Benchmark smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/docs-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/docs-smoke
```

`.benchmark_runs/` is ignored and should not be committed.

## Release Artifacts

The core Python wheel contains only the `agent_libos` package and its two
console entrypoints. The source distribution additionally contains the
repository-level PTY module, example Skill and Image packages, benchmarks,
tests, and documentation. Electron sources remain repository-checkout assets
validated by the separate GUI lane. Validate that contract before a release:

```bash
uv build --clear --out-dir dist
uv run python scripts/check_release_artifacts.py dist
```

The artifact checker requires the Python package, project metadata, and
lockfile versions to agree; when GUI sources are present, both GUI package
versions must agree as well. CI installs the built wheel and source
distribution into fresh environments, checks dependency consistency, runs both
wheel entrypoint help commands from outside the source tree, executes the
deterministic demo against an in-memory store, and preserves the validated
distributions for release use.

The deterministic matrix is not the full platform release matrix. See
[support_matrix.md](support_matrix.md) before claiming Windows/macOS, packaged
Electron, real MCP, or real LLM coverage.

The practical runner separates `native-live` from `modeled` scenarios. Native
scenarios fail when a semantic effect lacks a real ToolBroker call, state
oracle, external-effect row, or Explain-resolvable operation; there is no
modeled fallback in that lane.

## Real LLM Smoke

Real LLM paths are opt-in because tokens are valuable.

Configure the host environment, or pass an explicit env file to scripts that
offer one. The runtime LLM client does not implicitly read a workspace `.env`.

```bash
export OPENAI_BASE_URL=https://example-openai-compatible-endpoint/v1
export OPENAI_LANGUAGE_MODEL=your-model
export OPENAI_API_KEY=...
export AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1
```

Use plain `KEY=value` lines in an env file and pass it only to an entrypoint
that explicitly supports one, for example
`uv run --env-file .env agent-libos ...`. In PowerShell, set inherited values
with `$env:NAME = "value"`.

Useful optional variables:

- `OPENAI_API_MODE=responses|chat|auto`
- `OPENAI_TIMEOUT`
- `OPENAI_MAX_RETRIES`
- `OPENAI_STORE`
- `OPENAI_REASONING_EFFORT`
- `OPENAI_VERBOSITY`
- `OPENAI_SAFETY_IDENTIFIER`
- `OPENAI_PROMPT_CACHE_KEY`
- `OPENAI_PROMPT_CACHE_RETENTION=in-memory|24h`
- `OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID=true|false`
- `OPENAI_PARALLEL_TOOL_CALLS=true|false`
- provider-specific `OPENAI_ENABLE_THINKING`

`OPENAI_BASE_URL` is optional for the OpenAI API. Custom OpenAI-compatible
endpoints require `AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1` or an explicit
`allow_custom_base_url=True` client construction.

Official OpenAI Responses requests may also be configured with
privacy-preserving `safety_identifier`, prompt-cache routing fields, and
opt-in `previous_response_id` chaining. The runtime keeps `llm.store=False` and
Responses state chaining disabled by default; enable both only when retaining
provider-side response state is acceptable. These OpenAI-specific fields are
not sent to custom OpenAI-compatible endpoints. In the default stateless mode,
prior tool messages are sent back as ordinary bounded context text and no
durable provider-chain tool-output row is written.

Real asynchronous SDK transports are request-scoped. Scheduler quanta and
parallel process workers may use different short-lived event loops, so a cached
`AsyncOpenAI`/httpx keep-alive pool must never be reused across quanta. Explicit
host/test transports injected into an `LLMClient` are the exception: they are
retained across requests and are closed only when that wrapper, or the profile
registry that owns it, shuts down. The injector is responsible for using them
only with a compatible event-loop lifetime.

A provider-side chain is eligible only for the official Responses API with
`store=true`, `responses_previous_response_id=true`, and
`persist_full_io=true`. The preceding call must use the same LLM profile and
response-scope fingerprint (process, image, tool table, loaded Skill snapshot,
and durable context generation). It must also match a non-secret,
credential-keyed HMAC over the model, normalized official endpoint, API mode,
API-key environment name, credential identity, and organization/project tenant.
The credential is not stored; changing any provider/account identity input
forces a stateless reset, while an unchanged identity remains comparable after
restart. The preceding function-call manifest must contain unique non-empty
`call_id` values, and durable outputs must exist for every call and no extra
call. Only then does Agent libOS send the outputs as native
`function_call_output` items and set `previous_response_id`. This includes
sequentially executed parallel-tool batches and a wait result completed after a
runtime reopen. Missing/redacted/conflicting/partial output, changed scope or
provider identity, context compaction/restore, legacy ambiguous rows, or any
unrepresentable tool message resets unconditionally to a stateless
request/plain-context fallback; the runtime never continues a guessed provider
chain.

Blocking LLM-selected human, child, and message actions are durable. Each wait
generation has a unique `resume_token`; resume atomically claims
`pending -> resuming` by `(pid, token)`, and completion CASes that same
generation to `completed`. A resume that blocks again publishes a new pending
generation, so a stale worker cannot claim or complete it (ABA protection).
Only one executor can cross the resumed primitive boundary. If the runtime
reopens with a row already in `resuming`, or dispatch/output persistence/final
completion raises after a claim, it immediately marks the process failed,
retains the non-replayable state, and audits
`llm.pending_action_resume_interrupted` instead of automatically replaying a
tool whose external effect may already have happened.

Conditional LLM release waits follow the same claim discipline, but their
prepared provider request follows `llm.persist_full_io`. With full-I/O
retention disabled, SQL stores only non-sensitive identifiers and hashes while
the exact request remains in the current executor's memory. Approval can still
resume that exact request in the same runtime. A reopen cannot reconstruct it:
startup atomically claims the redacted generation, fails the process closed,
cancels its pending Human request, and audits that no provider replay occurred.

Protected public boundaries are registered by the runtime composition root for
Explainable Operations. New process, Object Memory, checkpoint, capability,
Human, ObjectTask, Skill/Image/remote-registry, or external primitive mutation
entrypoints must be added to that registry. Authorization and provider helpers
add decision/effect expectations only when those phases are actually reached;
do not declare provider evidence unconditionally for preflight denial paths.
Tests should cover the operation outcome, expected-role completeness, explicit
evidence resolution, and redacted output in addition to the original audit and
effect assertions. See [explainable_operations.md](explainable_operations.md).

Every new protected operation must also declare `data_flow_direction`. Egress
or bidirectional operations must provide stable Sink and trusted-source
descriptors and route their final dispatch through the SDK's transactional
data-flow revalidation. Do not infer egress from the older
`information_flow` flag: reads, DNS, and clock observation use that flag too.
Run `uv run python scripts/check_protected_operations.py`; its static checks
reject egress contracts that omit those descriptors or bypass the common SDK.
See [data_flow.md](data_flow.md).

Set `llm.parallel_tool_calls` or `OPENAI_PARALLEL_TOOL_CALLS=true` to let the
provider return multiple tool calls in one action-selection response. Agent
libOS dispatches that batch sequentially in one quantum; it does not run tools
concurrently.
Set `llm.auto_wait_on_empty_tool_calls: true` globally or on a specific LLM
profile only for providers that sometimes answer action-selection requests
without tool calls. When enabled, Agent libOS first preserves the existing
fallback JSON action parser; if the response still contains no valid action, it
synthesizes a `receive_process_messages` action with default arguments. The raw
LLM call record still stores the provider response with an empty `tool_calls`
list, and the synthetic wait listens for any unread process message.

Run a script smoke:

```bash
uv run python scripts/llm_write_goal_smoke.py
```

Run a benchmark smoke only with an explicit one-task limit:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --llm real --limit 1 --output .benchmark_runs/real-smoke
```

Every runtime LLM action-selection call persists an `llm_calls` row with
provider ids, model/API mode, usage, errors, and bounded observability
envelopes. With the default `llm.persist_full_io: true`, that row also contains
the full prompt, visible tools, output, tool calls, reasoning metadata, and raw
responses. Deployments using that default for self-evolution or fine-tuning
should disclose the retention and use in their user agreement.

LLM providers are selected through host-configured named profiles. Processes
persist only `llm_profile_id`; the Runtime resolves that id for each quantum and
reads API keys from the profile's `api_key_env` environment variable. The
configured default profile preserves the existing `OPENAI_*` environment
behavior. Other named profiles do not inherit ambient provider/model
environment variables; set their profile fields explicitly when they should use
a non-default model or endpoint.

The GUI can also create user-level profiles without editing the project config.
Those profiles are host configuration, not runtime database state. Electron
stores them at `app.getPath("userData")/llm-profiles.json`; direct Python GUI
server runs use `%APPDATA%/Agent libOS/llm-profiles.json` on Windows,
`~/Library/Application Support/Agent libOS/llm-profiles.json` on macOS, and
`${XDG_CONFIG_HOME:-~/.config}/agent-libos/llm-profiles.json` on Linux unless
`agent-libos-gui-server --llm-profiles-file <path>` is provided. The file stores
only non-secret routing fields and the `api_key_env` variable name; never put
the API key value in it. If `base_url` is set and `allow_custom_base_url` is
explicitly false, the false value is persisted so the profile does not start
using a custom base URL after reload.

Example user/config profile fields for common OpenAI-compatible providers:

```yaml
llm:
  profiles:
    gpt-5.5:
      model: gpt-5.5
      api_key_env: OPENAI_API_KEY
    qwen3.7-max:
      base_url: https://dashscope-compatible.example/v1
      model: qwen3.7-max
      api_key_env: QWEN_API_KEY
      api_mode: chat
      allow_custom_base_url: true
    glm-5.2:
      base_url: https://open.bigmodel.example/api/paas/v4
      model: glm-5.2
      api_key_env: GLM_API_KEY
      api_mode: chat
      allow_custom_base_url: true
    kimi-k2.7-code:
      base_url: https://api.moonshot.example/v1
      model: kimi-k2.7-code
      api_key_env: KIMI_API_KEY
      api_mode: chat
      allow_custom_base_url: true
```

Set `llm.persist_full_io: false` in a config overlay, or construct a replacement
`AgentLibOSConfig`, to opt out of full prompt, visible tool schema, model
output, tool call, reasoning, and raw response persistence. The config
dataclasses and their mapping fields are immutable, so do not mutate
`DEFAULT_CONFIG` in place. When full I/O persistence is disabled, the durable
row keeps content-free byte counts, JSON-kind/item-count metadata where
applicable, and hashes instead of raw values or readable previews. This policy
also applies before dispatch to conditional LLM release rows; `request_messages`,
`egress_payload`, and the rest of the prepared provider request are never
written to `llm_pending_actions.action_json` in opt-out mode.

The default remains `llm.persist_full_io: true` for deployments that use
complete LLM call records for self-evolution training or fine-tuning.

## Configuration Defaults

Non-secret runtime defaults live in `agent_libos.config.DEFAULT_CONFIG`.
`AgentLibOSConfig` uses Pydantic dataclass validation and fails fast when
numeric limits are negative, non-finite, inverted, or otherwise unsafe.
Product entrypoints read `config.yaml` from the project root when present, or
an explicit `--config <path>` overlay when provided. They do not auto-load a
`config.yaml` from the current working directory. Relative startup Runtime
Module paths in `config.modules.manifest_paths` resolve from the project root.
The loader starts from `DEFAULT_CONFIG`, recursively merges mapping fields,
replaces scalar/list/tuple fields, and then constructs a fresh
`AgentLibOSConfig`; it does not mutate `DEFAULT_CONFIG`.
See [configuration.md](configuration.md) for the complete precedence rules,
field-level group inventory, secret handling, and a command that prints the
exact defaults for the current checkout.

Library and test code should keep passing explicit config objects when a custom
runtime is required:

```python
from agent_libos.config import load_config_file

config = load_config_file("config.yaml")
runtime = Runtime.open(config=config)
```

Current default groups include:

- runtime database and default ids,
- scheduler quantum, worker, drain, and shutdown limits,
- process resource budgets, usage accounting, and default cwd,
- LLM timeouts and provider compatibility knobs,
- tool limits and text encodings,
- filesystem and Object Memory size limits,
- Deno sandbox limits and reserved JSR allowlist metadata (static imports are
  still rejected),
- ObjectTask notification, owner-watch, and shutdown limits,
- shell policy allow/block lists,
- JSON-RPC endpoint manifest, timeout, and request/response limits,
- MCP server manifest, HTTP/stdio environment allowlists, timeout, and
  request/response limits,
- data-label defaults, Host Sink trust rules, registry resource, and bounded
  registry/decision/file-binding queries,
- image registry limits,
- image commit limits,
- Object Memory and LLM context defaults,
- GUI HTTP/event/request limits,
- checkpoint snapshot limits,
- Skill package source, trust, resource, and `SKILL.md` limits,
- trusted startup Runtime Module manifests, hash trust, and registration limits,
- launcher presets,
- script defaults.

Resource budgets use integer fields for discrete calls, tokens, bytes, and peak
memory, while `max_runtime_seconds`, `max_subprocess_wall_seconds`, and
`max_subprocess_cpu_seconds` accept finite non-negative fractional seconds.
Booleans are not accepted as numbers.

Event limits are storage-selection bounds, not only renderer truncation. Each
LLM context preparation reads at most `llm_context.recent_event_limit` rows
newer than the process cursor. GUI snapshots read only the newest
`gui.snapshot_event_limit` rows, and process-event pagination uses the same
maximum with a `before` cursor. Durable event rows remain in the store.

Shell policy labels are protocol semantics, not user-remappable aliases. A
config may choose `shell.default_policy_level` and replace exact/prefix command
rules, but it cannot redefine the meanings of `always_deny`,
`allowlist_auto_else_ask`, `blocklist_ask_else_auto`, or `always_allow`.
Checkpoint defaults contain snapshot/list/payload/diff limits only; the former
`auto_high_risk_checkpoint` field was never wired to an operation and has been
removed. Strict overlays reject that legacy field instead of ignoring it.

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

When behavior changes, update the relevant doc. If the change affects a runtime
invariant, update the machine-readable source in `tests/invariants.yaml` and
then sync `docs/invariants.md` in the same change. Do not describe future work
as current behavior. Paper-facing documentation should stay aligned with the
fixed title:
`Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM
Agents`.

Current behavior must not claim:

- Python JIT compatibility,
- direct external framework adapters as trusted boundaries,
- MCP Resources/Prompts or real hosted GitHub/GitLab provider integrations that
  are not implemented (the typed local Git provider and simulated PRs are
  current),
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
- Deno JIT tool execution has no network dependency-resolution phase. Current
  validation rejects recognized static imports, including pinned JSR imports;
  the JSR allowlist is reserved metadata and does not authorize imports;
- Skills change visibility and prompt materialization only;
- self-evolution mechanisms such as Skills, JIT tools, image registration,
  process exec, checkpoint forks, child processes, and JSON-RPC endpoint
  visibility must not imply resource authority or additional resource budget;
- Runtime Modules are trusted startup TCB extensions; they may register tools,
  images, syscalls, provider hooks, and manifest-declared durable Object-release
  handlers, but must not be treated as process capabilities;
- JSON-RPC remote calls use registered endpoints and primitive capabilities
  rather than model-supplied URLs or secrets. Calls perform an exact
  endpoint/method capability gate before loading manifest metadata or schemas;
- MCP remote tool calls likewise gate on `server_id` and `tool_id` before
  loading server metadata or input schemas;
- runtime-mediated egress declares a stable Sink and trusted source descriptor,
  enforces the Host Sink registry independently of ordinary capability, and
  revalidates both at the provider boundary;
- checkpoint restore is scoped and append-only outside reconstructable state;
  provider-classified external effects are report-only in v1.

Prefer existing managers and primitives over new side channels. If a new host
effect is needed, add or extend a primitive and provider interface rather than
calling the host directly from a tool.

Every new provider-backed primitive must also register a
[`ProtectedOperationContract`](protected_operation_sdk.md) and execute each
real provider boundary through the returned handle's `call()` or `acall()`.
Declare authority mode, conservative mutation/information-flow ceiling,
event/audit/effect evidence roles, resource policy, classifier fallback,
`data_flow_direction`, and post-provider failure policy. Egress/bidirectional
contracts must also declare and populate their Sink/source/payload descriptors.
Run
`uv run python scripts/check_protected_operations.py`; direct effect-lifecycle
calls from provider subsystems fail this check. The checker also rejects a
provider-reaching helper when any call site bypasses an SDK phase and rejects
direct provider session-handle calls such as `session.handle.read()` outside a
phase. For `ResourcePolicy.REQUIRED`, provide measurable `failure_resource`
settlement or deliberately accept conservative preflight charging on a
dispatched failure.

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
