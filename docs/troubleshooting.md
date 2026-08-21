# Troubleshooting

Use this guide to identify the failing layer before changing authority, policy,
retention, or provider settings. Commands assume a source checkout unless the
text explicitly says “installed artifact.” The [CLI guide](cli.md),
[configuration guide](configuration.md),
[generated configuration field reference](configuration_reference.md), and
[support matrix](support_matrix.md) remain the maintained contracts. Return to
the [documentation home](index.md) for other task paths.

## Fast diagnosis

From the repository root, collect the smallest non-secret facts first:

```bash
python --version
uv --version
git --version
uv run agent-libos --help
uv run agent-libos --db local demo
```

Do not paste API keys, database DSNs, OAuth callback URLs, bearer tokens, raw
provider responses, retained goals, or Human-request payloads into a bug report.
Record the exact command, exit status, stable error `type`, relevant version,
platform, and whether the path was source, wheel, sdist, or internal desktop.

| Symptom | Check first | Canonical detail |
| --- | --- | --- |
| `uv sync` or import fails | Python version and whether the command runs at repository root | [Installation](../README.md#installation-and-operations) |
| Deno/JIT validation is skipped or fails | `deno --version`, configured executable, image requirements | [Deno and JIT](#deno-and-typescript-jit) |
| Runtime refuses a database | selected `--db`, schema version, active Runtime lease | [Database and recovery](#database-schema-lock-and-recovery) |
| LLM call does not start | selected profile, inherited environment, custom-base-URL opt-in | [LLM providers](#llm-profiles-credentials-and-provider-calls) |
| Operation is denied | exact pid, resource, right, Task Authority, data-flow Sink | [Authority and data flow](#capability-task-authority-and-data-flow-denials) |
| GUI cannot start or connect | backend startup JSON, loopback URL, token, renderer mode | [GUI](#gui-startup-and-task-creation) |
| Benchmark exits zero despite failed rows | whether `--require-all-passed` was set | [Benchmark](#benchmark-results-and-exit-status) |

## Installation and environment

### Unsupported Python or missing `uv`

The root package supports Python 3.11–3.14. Create or select a compatible
interpreter, install `uv`, and use the locked environment:

```bash
uv sync --frozen
```

System Git 2.26 or newer is required for the typed Git provider and complete
test matrix. Node and npm are needed only for GUI development; Deno is needed
only for real Deno/JIT paths and images that package JIT tools.

### Source checkout versus installed artifact

Commands prefixed with `uv run` use the checkout's project environment. An
installed wheel instead provides `agent-libos`, `agent-libos-gui-server`, and
`agent-libos-migrate-tool-groups` directly in that environment's executable
directory. If `uv run` imports the checkout while you intended to test a wheel,
activate or call the isolated artifact environment explicitly and run
`python -m pip check` or `uv pip check --python <interpreter>` there.

The core wheel intentionally omits repository-level examples, docs, benchmarks,
the optional PTY module, and workspace Image/Skill packages. Use the source
distribution or checkout when a documented path points to one of those assets.

### Optional dependency is absent

Install only the integration being exercised:

```bash
uv sync --frozen --extra mcp
uv sync --frozen --extra postgres
uv sync --frozen --extra pty
```

The `pty` extra supplies `pywinpty` on Windows; it does not place the repository
`modules/pty` package into the core wheel. The `mcp` extra is required for live
MCP provider paths, not ordinary core Runtime use.

### `uv` cache permission failure

An error creating or opening the global uv cache is an environment ownership or
sandbox problem, not an Agent libOS store failure. Prefer a correctly owned uv
cache. For one isolated diagnostic run, point uv at a private writable cache
directory rather than changing broad filesystem permissions:

```bash
UV_CACHE_DIR=/tmp/agent-libos-uv-cache uv sync --frozen
```

Do not use this workaround to hide a shared-machine ownership problem in a
release receipt.

## Deno and TypeScript JIT

### Tests say Deno is unavailable

Check the executable visible to the Host:

```bash
deno --version
```

When Deno is absent, tests marked `real_deno` skip with a reason. This does not
make an image with packaged JIT tools usable: such an image must validate its
tools at boot and fails closed if Deno is unavailable. The deterministic core
demo still runs and simply omits its optional JIT validation branch.

To select a non-default executable from Python, construct a Runtime config from
`DEFAULT_CONFIG` with the `tools.deno_executable` field replaced. Do not accept a
model-supplied executable or transport command.

### JIT validation rejects an import

Current JIT tools run no-permission and cached-only, and static imports/
re-exports are rejected. The reserved JSR allowlist is metadata for compatibility
and does not enable imports. Bundle import-free source and reach Runtime effects
only through the documented `libos.syscall` surface. See
[Tools and JIT: sandbox rules](tools_and_jit.md#sandbox-rules).

### A Deno process times out or is terminated

Three independent ceilings may apply: the package tool's `timeout_s`, Host
`tools.deno_timeout_hard_limit_s`, and the process's remaining resource budget.
Raising one does not raise the others. Inspect the process resources and the
bounded JIT result instead of retrying an ambiguous external effect blindly.

## Database, schema, lock, and recovery

### `local` unexpectedly loses state

`--db local` is in-memory and intentionally disappears when the Runtime closes.
Use an explicit SQLite path for persistence:

```bash
uv run agent-libos --db .agent_libos.sqlite init
```

The CLI loads the project-root `config.yaml` when `--config` is omitted. In this
checkout that file may select a persistent store and trusted module even if the
shell's current directory differs. Pass `--db local` or an exact `--config` when
you need an unambiguous diagnostic environment.

### Runtime reports an unsupported store version

Ordinary startup accepts only RuntimeStore schema v7 and never migrates it
implicitly. Stop every Runtime using the database, make and verify an independent
owner-only backup, then follow the exact offline migration sequence in
[Storage](storage.md#offline-v6-to-v7-migration). A v5 store must migrate to v6
before v7; v4 must migrate to v5 first. Older, malformed, and unversioned stores
remain unsupported.

Do not improvise DDL, edit the schema marker, or delete lease sidecars. Dry-run,
plan digest, expected source hash/schema, apply, and readback are one reviewed
workflow.

### Store contains legacy Tool Groups

This is a content migration separate from SQL schema migration. With all
Runtimes stopped, preview the installed migration command first:

```bash
agent-libos-migrate-tool-groups /path/to/runtime.sqlite
```

The default rolls the transaction back. Review the report before repeating the
same command with `--apply`. See
[Tool Skills](tools_and_jit.md#on-demand-tool-skills).

### Store is already active or locked

Agent libOS supports one writable Runtime per database/schema. Close the GUI
server, CLI/embedded Runtime, or PostgreSQL session that owns the active lease.
Do not bypass the lease or delete its files while the owning process may still
be alive. If a prior startup failed during recovery, preserve the diagnostics
and follow [Runtime Storage](storage.md#active-runtime-leases) rather than opening
a second writer.

### Object payload is missing after reopen

Ordinary Object payloads are Runtime-memory state. Persistence of metadata does
not imply persistence of every payload. The initial root goal has one narrow
recovery envelope under the documented retention settings; Durable Task Runs
have a separate explicit plaintext-payload opt-in. Read
[Object Memory](object_memory.md#persistence-invariant),
[Durable Task Runs](durable_task_runs.md#enabling-durable-payloads), and
[Evidence Payload Retention](evidence_payload_retention.md) before changing
retention.

## LLM profiles, credentials, and provider calls

### Environment variables are not being read

The generic CLI and Python Runtime do not load `.env` implicitly. Export values
into the Host environment or make the CLI load one explicit file:

```bash
uv run --env-file .env agent-libos --db .agent_libos.sqlite processes
```

The Electron development launcher and `scripts/run_coding_agent.py` have their
own documented env-file behavior; do not infer it for other entrypoints.

### Named profile has no model, endpoint, or key

Only the configured default profile inherits the legacy `OPENAI_*` model,
endpoint, and provider-policy variables. A non-default named profile should
declare its `model`, optional `base_url`, `api_mode`, and exact `api_key_env`.
Every profile reads the credential only from the environment variable named by
`api_key_env`; the secret value must not appear in YAML.

Inspect the precedence rules in
[Configuration](configuration.md#effective-llm-profile-precedence).

### Custom compatible endpoint is rejected

Custom `base_url` use requires either the selected profile's
`allow_custom_base_url: true` or the Host-wide
`AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1` opt-in. This is a Host trust decision;
do not set it solely because model text supplied a URL. OpenAI-hosted endpoints
need neither opt-in.

### An LLM command spends tokens or leaves the process paused

`run`, `llm-once`, `exec --run`, the coding-agent launcher, and real-model
benchmark modes can call the configured provider and spend tokens. The local
demo and default deterministic benchmark do not.

After explicit transport retries are exhausted, the Runtime records the failed
logical call and may pause the process for Host inspection. Inspect the process,
LLM call, operation, audit, and external-effect state before choosing a resume or
recovery action. Do not treat a lost response as proof that the provider did not
perform an effect.

## Capability, Task Authority, and data-flow denials

An operation may be denied even when its tool is visible. Check the exact
process, resource, and right:

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities list --subject <pid>
uv run agent-libos --db .agent_libos.sqlite capabilities explain <pid> filesystem:workspace:README.md read
```

Common independent causes are:

- no matching Capability, or a covering `ask`/`deny` rule;
- a constraint, expiry, use count, delegation depth, or current-state mismatch;
- the Task Authority Manifest does not authorize or make the requestable scope
  available;
- a process resource budget is exhausted;
- cwd selection lacks directory read authority;
- data labels exceed the exact Sink's clearance or identity domain;
- a conditional Sink lacks the exact one-shot release;
- a required Runtime Module is absent or has a different source hash.

Grant only the narrow reviewed resource/right needed. A trusted Sink does not
grant operation authority, image/Skill requirements are declarations, and Human
approval cannot bypass Task Authority or data flow. Use
[Capabilities](capabilities.md), [Task Authority](task_authority_manifest.md),
and [Data Flow](data_flow.md) for the decision order.

## GUI startup and task creation

### Development GUI cannot start the backend

Install both dependency sets, then start from the checkout:

```bash
uv sync
npm --prefix gui install
npm --prefix gui run electron:dev
```

The development launcher tries `AGENT_LIBOS_GUI_SERVER_BIN`, then the checkout
`.venv` entrypoint, then `uv run agent-libos-gui-server`. Run the server directly
to separate Python startup from Electron rendering:

```bash
uv run agent-libos-gui-server --db .agent_libos.sqlite --port 0
```

Its startup JSON includes a loopback URL and bearer token. Treat both the token
and logs as Host-local data; do not expose the server on a non-loopback address.

### Durable mode is unavailable in New task settings

Durable Task Run creation requires both `task_runs.enabled: true` and
`task_runs.plaintext_payloads_enabled: true`. The latter stores readable task
payloads in the database and backups, so the GUI reports the disabled state and
does not silently enable it. A regular process remains usable without that
plaintext opt-in.

### Packaged desktop expectations do not match development

The self-contained desktop is an internal, unsigned distribution with explicit
platform bounds. It ignores the development backend override, does not read a
checkout `.env`, and uses its bundled Python/Deno paths. Consult
[GUI: self-contained internal desktop distribution](gui.md#self-contained-internal-desktop-distribution)
and the [support matrix](support_matrix.md) before reporting a public installer,
signing, notarization, or update failure as a supported path.

## Benchmark results and exit status

### Failed task rows but command exited zero

Comparison mode records oracle failures for analysis without necessarily making
the command fail. Use the explicit oracle smoke gate:

```bash
uv run python experiments/run_benchmark.py \
  --suite benchmarks/runtime_safety \
  --runner agent_libos_full \
  --limit 3 \
  --require-all-passed \
  --output .benchmark_runs/smoke
```

The complete release gate additionally uses `--require-release-evidence`; the
two flags are independent.

### Real-model benchmark refuses the selection

After all filters and `--limit`, real mode must select exactly one task and one
or more Agent-libOS-family runners. Wrapper/sandbox baselines and `--runner all`
cannot use `--llm real`. Real mode requires a configured credential and spends
tokens.

### Metrics reject an output directory

Keep `metadata.json`, `metrics.json`, result rows, and effect evidence together.
The current collector accepts complete run-output schema v2 and fails closed on
missing, unknown, or mixed provenance. Do not copy a result directory between
source revisions and present it as evidence for the new checkout. See
[Benchmark: outputs](benchmark.md#outputs).

## Reporting a reproducible problem

Include:

- Agent libOS version and exact source commit when using a checkout;
- OS/architecture and Python, uv, Git, Deno, Node/npm versions that apply;
- install form: checkout, wheel, sdist, or internal desktop;
- the smallest command and its exit status;
- stable JSON error type and a redacted message;
- selected store backend and schema version, without its credentials;
- whether a real provider call may have started; and
- the relevant operation/effect/audit ids when safe to disclose.

Security issues and suspected secret exposure belong in [SECURITY.md](../SECURITY.md),
not a public troubleshooting transcript.
