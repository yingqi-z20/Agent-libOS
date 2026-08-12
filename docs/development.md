# Development Guide

This guide covers local setup, regression checks, real Deno behavior, optional
real LLM paths, and documentation rules for Agent libOS contributors.

## Setup

Use Python 3.11–3.14 and uv. CI currently pins uv `0.11.32`; use that exact
version when reproducing a CI or release receipt. The full matrix also requires
system Git 2.26 or newer. The GUI package declares Node
`^24.15.0 || >=26.0.0` with npm 11 or newer. Per-change CI exercises the Node
24 LTS line with the npm version supplied by that toolchain. Deno is optional
unless validating real TypeScript/JIT execution.

Install dependencies:

```bash
uv sync
npm --prefix gui install
```

Use frozen resolution for the locked runtime/development dependency graph in
artifact and CI-style checks:

```bash
uv sync --frozen
npm --prefix gui ci
```

The default sync includes the development group used by standard checks.
Optional package extras remain independent: add `--extra mcp`,
`--extra postgres`, or `--extra pty` for the corresponding integration gate.
The complete Windows matrix uses `uv sync --frozen --extra pty` so native
ConPTY tests are available. Do not install the release group during ordinary
development; the release path selects it explicitly.

The release path installs its exact build backend from the frozen `release`
dependency group and invokes `uv build --no-build-isolation` with that virtual
environment's interpreter. Keep the `hatchling` version in `[build-system]` and
the `release` group aligned whenever the backend is upgraded.

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

### Isolated AgentDojo harness

AgentDojo has a separate dependency environment so its SDK graph does not enter
the root lock. Run its deterministic harness tests from the subproject:

```bash
(
  cd experiments/agentdojo
  uv sync --frozen
  uv run --frozen pytest -q
)
```

The subshell returns you to the repository root before the commands in the next
section.

The subproject's `pyproject.toml` and `uv.lock` cover Python 3.11–3.12. CI runs
the commands above on both versions, and the release-artifact job waits for that
matrix. The root `scripts/test_matrix.py` lanes do not collect these tests.
These deterministic tests make no provider calls; follow the explicit
credential/token gate in the [AgentDojo harness guide](../experiments/agentdojo/README.md)
for a real-model evaluation.

Optional provider gates are excluded unless selected explicitly:

```bash
uv sync --frozen --extra mcp
npm --prefix tests/fixtures/mcp_sdk_v2/typescript_server \
  ci --ignore-scripts --no-audit --no-fund
uv run python -m pytest -q \
  tests -m "mcp and not postgres" \
  --run-mcp --fail-on-skip
uv run python scripts/run_mcp_conformance.py
uv run python -m pytest -q \
  tests/self_evolution/test_builtin_agent_images_real_llm.py \
  --run-real-llm --fail-on-skip
uv sync --frozen --extra postgres
export AGENT_LIBOS_POSTGRES_DSN='postgresql://agent_libos:agent_libos@127.0.0.1:5432/agent_libos'
uv run python -m pytest -m postgres --run-postgres --fail-on-skip
```

The MCP integration job selects the full-tree `mcp` marker closure,
excluding only the separately gated PostgreSQL parametrizations, without a
hand-maintained file allowlist, ignored tests, or skip-tolerant fallbacks, on
Ubuntu Python 3.11 and 3.14. `scripts/check_mcp_test_closure.py` is the closure
rule: it collects every lane and fails if an MCP-named, MCP-fixtured, or
MCP-referencing node lacks the product marker. Deterministic lanes exclude that
marker to avoid duplicate execution; the dedicated job owns the complete MCP
closure. The job also
runs independent real-stdio Python and TypeScript SDK
v2 fixtures for Resources, Prompts, and modern resource subscriptions. The
TypeScript fixture is installed from its checked-in lock file with lifecycle
scripts disabled. A separate native job runs real stdio and loopback HTTP SDK
smokes on Windows and macOS. Release-artifact jobs for Python 3.11 through 3.14
clean-install both canonical wheel and sdist with their published `[mcp]` extra
and, from a temporary working directory, execute a self-contained exact-v3
stdio plus loopback Streamable HTTP server through the installed Runtime
Resource, Resource Template, Prompt, Completion, bounded resource-subscription,
and Tool protected paths. The same installed-package smoke writes its OAuth server fixture into
that temporary directory, performs a Host-pinned loopback-TLS OAuth
authorization-code, PKCE, and Bearer exchange, and runs an offline Store v6-to-v7
migration followed by a schema-v7 reopen. A second installed-package smoke
captures MRTR and remote-Task results through the protected Runtime, closes and
reopens the SQLite Store, then drives continuation inspect/respond/cancel and
Task get/update/cancel/re-observe through the installed CLI. It requires each
initial Tool call and continuation/Task dispatch count exactly, and verifies
that opaque request state and remote Task IDs never enter the Store or CLI
projection. The artifact gate rejects a source-tree
`agent_libos` import, nonempty fixture stderr, leaked supervised connections,
or missing protected audit actions, so dependency imports alone cannot satisfy
the artifact gate.

The `mcp` extra pins the reviewed SDK environment at `mcp==2.0.0` and the
audited OS-credential implementation at `keyring==25.7.0`, plus directly
declared `anyio`, `httpx2`, `httpcore2`, and `opentelemetry-api` bounds. Legacy
wire compatibility is tested through SDK v2 and raw protocol fixtures; CI does
not install SDK v1 beside it. Supplying `--run-mcp` without this complete extra
is a configuration error before collection, rather than a green run made only
of skips.
The conformance runner checks out one reviewed upstream revision in a temporary
directory and runs its explicit strict Tools/HTTP-schema allowlist, including
the Resource/Prompt request-header branches, plus MRTR and the fixed-upstream
OAuth pre-registration and CIMD scenarios. The OAuth harness
obtains the Resource and Authorization fixture origins directly from the
reviewed scenario object before constructing Runtime, pins both origins and
metadata URLs, and never enables Dynamic Client Registration or treats PRM/401
discovery as authority. Durable evidence for every upstream scenario retains
only check ID, status, bounded specification references, a deterministic
evidence digest, and the already pinned source identity in the summary; raw
names, descriptions, timestamps, logs, and protocol or authorization details
are dropped. It uses no expected-failure
baseline. The remaining reviewed OAuth scenarios do not provide a Host-pinned
expected issuer through their runner contract and are therefore listed by exact
name and reason as unavailable rather than counted as passes. The pinned
suite's `2025-03-26` OAuth backcompat, client-credentials,
enterprise-managed-authorization, DPoP, and workload-identity scenarios are
listed separately as reviewed product exclusions; they are not conflated with
the unavailable authorization-code scenarios. A separate real
loopback-TLS Runtime gate covers pinned issuer/resource/endpoints, PKCE, Bearer
injection, and secret absence. Generated results remain under ignored
`.benchmark_runs/` and are not release artifacts. Git checkout, `npm ci`, the
explicit build target, the Node runner, and its Python client start from a
minimal environment allowlist. Their HOME, temporary directories, XDG state,
and npm cache/user configuration are isolated for the run; ambient credentials,
DSNs, proxy configuration, Python import paths, telemetry context, and tokens
are not forwarded.

The real-LLM command may consume paid provider tokens and requires the Host
environment described below. PostgreSQL requires a configured disposable test
service and the `AGENT_LIBOS_POSTGRES_DSN` environment variable shown above;
without it the marked tests skip, and `--fail-on-skip` correctly makes the gate
fail. Do not point the suite at a production database.
`--fail-on-skip` is intentional on these evidence commands: a missing SDK,
credential, server, or service must not turn an all-skipped provider gate into a
green release result. Omit it only for exploratory local runs whose skips are
being reported rather than claimed as validation. `--fail-on-skip` is a pytest
option, not a `scripts/test_matrix.py` option; use the direct, scoped pytest
commands above when collecting no-skip evidence.

Use pytest-xdist workers for faster local Python feedback:

```bash
uv run python scripts/test_matrix.py --lane all --workers 4
uv run python scripts/test_matrix.py --lane runtime --workers auto --dist worksteal
uv run python scripts/test_matrix.py --lane runtime --shard-count 2 --shard-index 0
```

`--workers` applies only to Python lanes. The `runtime`, `security`,
`self-evolution`, `providers`, and `all` lanes default to bounded parallel
execution with at most four workers and `--dist worksteal`, which balances long
persistence and runtime-reopen tests. Pass `--workers 1` for serial failure
diagnosis, or set `AGENT_LIBOS_TEST_WORKERS` / `AGENT_LIBOS_TEST_DIST` to
override defaults. Pass `--durations 25` to report the slowest tests.
`--max-lane-seconds` is a hard process-tree timeout for each selected command;
its local default is 600 seconds. `--dist loadfile|loadscope|load|worksteal`
selects the xdist scheduler when more than one worker is used. Deterministic
file-weighted `--shard-count N --shard-index I` applies only to one Python lane
(not `all` or `gui`); indices are zero-based, and every shard must contain at
least one selected test file. The Python `all` lane is one command, while the
GUI test, typecheck, and build commands each receive the full timeout
independently. Timeout exits with status 124 after terminating the process
group/tree. Standard lanes deselect `postgres` tests because the PostgreSQL CI
service runs them separately with `pytest -m postgres --run-postgres`. Linux CI
uses a 480-second process deadline for the runtime lane and 360 seconds for the
other standard lanes, within a 15-minute step. The Windows matrix uses explicit
file shards and a 1,400-second process deadline within a 25-minute step; those
larger limits cover platform and runner variance rather than a different test
contract. Run the `gui` lane separately; it cleans Electron output before
production compilation, excludes generated `dist-electron` files from Vitest,
and never emits test files into the production Electron tree. Install GUI
dependencies first with `npm --prefix gui install`.

The architecture check covers the core package and repository-level Runtime
Modules. Its default function-size ceiling is 200 lines; larger existing
functions remain pinned to their exact checked-in ratchet budgets. It rejects
new lower-layer Runtime/API imports,
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

This is a bounded oracle smoke. The complete deterministic release gate omits
`--limit` and adds `--require-release-evidence` so audit completeness must be
1.0 and the false-denial numerator must be zero in addition to all success and
safety oracles passing.

`.benchmark_runs/` is ignored and should not be committed.

## Release Artifacts

The core Python wheel contains only the `agent_libos` package and its three
console entrypoints: `agent-libos`, `agent-libos-gui-server`, and the explicit
offline `agent-libos-migrate-tool-groups` migration command. The source
distribution additionally contains the
repository-level PTY module, example Skill and Image packages, benchmarks,
tests, and documentation. Electron sources are not part of either Python
artifact; they are validated by the GUI lane and used only by the separate
native internal desktop workflow described below. Validate the Python artifact
contract before a release:

```bash
uv sync --frozen --no-dev --group release
uv build --no-build-isolation --clear --out-dir dist --python .venv/bin/python --no-create-gitignore
.venv/bin/python scripts/check_release_artifacts.py dist --write-checksums
uv run --frozen --no-dev --group release twine check \
  dist/agent_libos-1.5.0-py3-none-any.whl dist/agent_libos-1.5.0.tar.gz
uv run --frozen --no-dev --group release check-wheel-contents \
  dist/agent_libos-1.5.0-py3-none-any.whl
.venv/bin/python scripts/check_release_artifacts.py dist --verify-checksums
```

The artifact checker requires the Python package, project metadata, and
lockfile versions to agree; when GUI sources are present, both GUI package
versions must agree as well. The artifact directory is a closed set: extra
files, directories, symbolic links, and other non-regular entries are rejected.
The wheel file contract closes over every `agent_libos.mcp` module plus the
schema-v7 SQLite/PostgreSQL MCP contract files. The source-distribution contract
also requires the reviewed MCP examples, conformance and installed-package
smoke scripts, and the frozen Python/TypeScript fixture sources without vendored
`node_modules` or generated caches.
CI builds the pair once, records `SHA256SUMS`, and makes all four Python-version
jobs download and verify that same pair before installing hash-checked locked
dependencies and the artifact itself into fresh
environments, checking dependency consistency, running all three entrypoint
help commands from outside the source tree, and executing the deterministic
demo against an in-memory store. The build job waits for its declared pre-build
gates; the clean-install matrix is downstream of that build, so the candidate
is not release-validated until every smoke job also succeeds.

The Python pair is separate from the self-contained internal desktop artifacts.
Native desktop builds use the exact `desktop` dependency group and GUI lock:

```bash
uv sync --frozen --group desktop --extra mcp
npm --prefix gui ci
npm --prefix gui run desktop:stage
npm --prefix gui run desktop:dist
python scripts/check_desktop_artifacts.py desktop-dist
```

`desktop:dist` is native-only and writes ignored output to `desktop-dist/`; it
must not be used to cross-freeze Python. The manual
`desktop-internal.yml` workflow builds macOS arm64 on `macos-15`, Windows x64
on `windows-2025`, and Linux x64 on `ubuntu-24.04`. It uploads only internal
Actions artifacts, SBOMs, component inventories, notices, and checksums. It has
no GitHub Release or publication permission. See [gui.md](gui.md#self-contained-internal-desktop-distribution)
for the packaged path and [support_matrix.md](support_matrix.md) before making
platform, signing, real-provider, or real-LLM claims.

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
- `OPENAI_PROMPT_CACHE_RETENTION=in_memory|24h`
- `OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID=true|false`
- `OPENAI_PARALLEL_TOOL_CALLS=true|false`
- `OPENAI_FALLBACK_JSON_ACTIONS=true|false`
- provider-specific `OPENAI_ENABLE_THINKING`

`OPENAI_BASE_URL` is optional for the OpenAI API. Custom OpenAI-compatible
endpoints require `AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1` or an explicit
`allow_custom_base_url=True` client construction.

To exercise the original five general/specialized image contracts against one
configured real model, run the scoped opt-in suite:

```bash
uv run --env-file .env python -m pytest -q \
  tests/self_evolution/test_builtin_agent_images_real_llm.py \
  --run-real-llm --fail-on-skip
```

The same suite covers the coding-image read/write/readback/exit path in an
isolated temporary workspace. The four narrow direct long-horizon images have
deterministic image-contract coverage; their end-to-end live coverage belongs
to the repository-maintenance, evidence-synthesis, analysis, and browser/customer
workflow evaluators so results retain scenario-specific utility and
zero-duplicate-effect oracles. `scripts/run_coding_agent.py` and
`scripts/llm_write_goal_smoke.py` remain useful for manual coding-image smoke
runs. The coding and multi-round toolmaker cases each have a 360-second
per-test timeout; the context-compressor case has a 600-second timeout and uses
a 240-second provider-call timeout. The base and review cases retain the
ordinary 120-second pytest timeout. These tests spend provider tokens and are
not part of the deterministic default matrix.

OpenAI-compatible Responses and Chat requests may also be configured with a
`safety_identifier` and prompt-cache routing fields. These options are sent to
the explicitly selected endpoint, including a custom OpenAI-compatible base
URL; compatibility retry may remove an option only after that provider rejects
it. Do not configure an identifier or cache key that the selected provider is
not trusted to receive. `previous_response_id` is narrower: the low-level
client sends it only for the official Responses endpoint, with `store=true`,
and only when the caller explicitly supplies an eligible id.

The AgentProcess executor is stateless even if a profile configures
`store=true` and `responses_previous_response_id=true`. It rebuilds the complete
local snapshot, records the configured-but-disabled reason in `llm_calls`, and
never combines that snapshot with `previous_response_id`. For ordinary Runtime
prompt modes, prior tool results are represented in the rebuilt bounded local
context. An `image_only` Image instead replays its paired tool transcript in the
provider's native form without provider-side chaining: Responses receives
`function_call`/`function_call_output` input items, while Chat receives
assistant/tool messages.

Real asynchronous SDK transports are request-scoped. Scheduler quanta and
parallel process workers may use different short-lived event loops, so a cached
`AsyncOpenAI`/httpx keep-alive pool must never be reused across quanta. Explicit
host/test transports injected into an `LLMClient` are the exception: they are
retained across requests and are closed only when that wrapper, or the profile
registry that owns it, shuts down. The injector is responsible for using them
only with a compatible event-loop lifetime.

The low-level `LLMClient` still supports delta-oriented callers that explicitly
pass `previous_response_id`. That path is restricted to the official Responses
API with `store=true`; representable paired tool history can be emitted as
native `function_call_output` items, while an unpaired or unsupported tool
message is rendered as bounded plain user context and prevents chaining. This
client capability is not an AgentProcess continuation contract and does not
create Runtime `llm_tool_outputs` rows by itself.

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
If the operation must not accept `untrusted` sources, declare an explicit
`minimum_egress_integrity`; do not implement the rule in a model prompt or tool
name check. Add a denial-before-provider regression and assert the floor in
effect evidence.
Run `uv run python scripts/check_protected_operations.py`; its static checks
reject egress contracts that omit those descriptors or bypass the common SDK.
See [data_flow.md](data_flow.md).

Set `llm.parallel_tool_calls` or `OPENAI_PARALLEL_TOOL_CALLS=true` to let the
provider return multiple tool calls in one action-selection response. Agent
libOS dispatches that batch sequentially in one quantum; it does not run tools
concurrently.
Set `llm.auto_wait_on_empty_tool_calls: true` globally or on a specific LLM
profile only for providers that sometimes answer action-selection requests
without tool calls. When enabled, Agent libOS synthesizes a
`receive_process_messages` action with default arguments. The raw
LLM call record still stores the provider response with an empty `tool_calls`
list, and the synthetic wait listens for any unread process message.
Set `llm.fallback_json_actions: true` or
`OPENAI_FALLBACK_JSON_ACTIONS=true` only for a provider that cannot reliably
accept native `tools`. It opts that profile into the compact text JSON action
protocol and provider retry; the default is native tool calls only.

Run a script smoke:

```bash
uv run python scripts/llm_write_goal_smoke.py
uv run python scripts/run_coding_agent.py --workspace /path/to/repo \
  --goal "Inspect the requested change" --permission-preset read-only --strict
```

The coding launcher runs the configured real LLM immediately unless
`--no-run` is supplied, with `runtime.launcher_max_quanta` (40 by default) as
its ceiling. Its default `edit` preset grants workspace-wide read and write;
the explicit read-only preset above is safer for inspection. By default it
uses a workspace-keyed persistent database outside the exposed workspace and
returns JSON even when the process fails; choose `--ephemeral-db` for an
in-memory store and `--strict` when failure or kill must make the command
non-zero.

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

The same setting governs the dedicated cross-CLI bootstrap record for a
committed root spawn. Full-I/O mode stores one bounded, identity- and
hash-bound copy of that process's initial goal; startup may rehydrate only the
exact matching active root goal before the ordinary Object-payload sweep.
Only the immutable initial GOAL created by `ProcessManager` is eligible;
mutable goal handles are never recovered. Terminal cleanup redacts the copy to
hash-only; failed launch rollback and startup compensation redact before
committing a non-committable publication state. Opt-out mode writes no
reversible goal content, so tests and examples that close after `spawn` and
execute after reopen must either enable full-I/O persistence or keep goal
creation and the first quantum in one Runtime lifetime. Do not generalize this
exception to child/fork goals, exec replacement goals, or arbitrary Object
payloads.

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

This opt-out is intentionally incompatible with `prompt_mode: image_only`.
Custom Image packages default to `image_only`, whose next quantum must recover
the exact latest native assistant/tool transcript after a Runtime reopen; such a
process fails before provider dispatch when full-I/O persistence is disabled.
Select a Runtime-owned prompt mode instead if content-free LLM-call persistence
is a deployment requirement.

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
from agent_libos import Runtime
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
The final `ResourceBudget` model and explicit resource tool/manifest inputs
reject booleans as numbers. Configuration loading is a different boundary:
ordinary non-strict Pydantic `int`/`float` fields may coerce YAML booleans to
`0`/`1`, while fields declared `StrictInt` or `StrictFloat` reject them. Do not
use booleans for numeric config values; see
[configuration.md](configuration.md#loading-and-precedence) for the
authoritative strict-field groups.

Event limits are storage-selection bounds, not only renderer truncation. Each
explicitly enriched LLM context preparation reads at most
`llm_context.recent_event_limit` rows newer than the process cursor; the
default source-only path does not capture event deltas. GUI snapshots read only the newest
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

Invariant manifest schema v2 uses `required_platform_nodes` when a claim needs
native host evidence. Each platform list must contain exact node ids from the
invariant's main `node_ids` list and those tests must carry the corresponding
`platform_darwin` or `platform_linux` marker. Release CI runs each marker in its
own native shard with `--fail-on-skip`; ordinary cross-platform collection does
not substitute for that platform evidence.

Current behavior must not claim:

- Python JIT compatibility,
- direct external framework adapters as trusted boundaries,
- MCP Apps, Roots, Sampling, Logging, OpenTelemetry product integration, OAuth
  Dynamic Client Registration, the deprecated standalone SSE transport, an
  MCP server surface, or real hosted GitHub/GitLab provider integrations. The
  client-only exact-v3 governed Tools, Resources/Prompts/Completion, MRTR, bounded
  subscriptions, Host-preconfigured OAuth, and digest-pinned Tasks extension
  are current Host surfaces; they must not be generalized into the excluded
  server- or model-authority surfaces,
- provider-level compensation for rollbackable external side effects,
- Skill activation as a capability grant.

`agent_libos_design_doc.md` and `plan.md` are migration notices for retired
historical material. They are not part of the current user-documentation
contract. Do not use prior revisions of either file for current command syntax,
authorization behavior, security guarantees, runtime semantics, or release
evidence.

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
