# CLI Reference

The package installs the `agent-libos` command.

The package also installs `agent-libos-gui-server`, which is used by the
Electron desktop console described in [docs/gui.md](gui.md).
Both runtime entrypoints are implemented under `agent_libos.api`, because they are
host-facing control surfaces over the same runtime boundary.

The wheel also installs `agent-libos-migrate-tool-groups`, an explicit offline,
one-time migration command for legacy stores. It defaults to a rolled-back dry
run and is not a normal Runtime control surface; see [Tool Skills](tools_and_jit.md).

Use `--db` to select a runtime store. The sentinel target `local` is in-memory
SQLite. Any other filesystem path creates or opens a persistent SQLite
database. A `postgresql://` or `postgres://` DSN opens a PostgreSQL runtime
store. PostgreSQL support is optional; install it before using a DSN, for
example with `uv sync --frozen --extra postgres`.

```bash
uv run agent-libos --db .agent_libos.sqlite <command>
```

Global options (`--config`, `--db`, `--module-manifest`, and the two module
trust options) must appear before the command name. The startup trust options
extend the selected Host configuration; they are not process capabilities.
Management groups that support `--actor-pid` run as the audited Host admin when
that option is omitted. Supply an actor pid when the operation must be confined
to that process's authority. Other safety-sensitive opt-ins remain explicit:
payload retention and Tool-Group migration default to dry runs, exec defaults
to no scheduler run and to dropping external capabilities, and a bare
`--max-quanta` omission may use an unbounded-until-idle configured value.

Runtime domain errors are printed as a stable JSON object with
`schema_version`, `error.type`, and `error.message`, and exit with status 1;
the installed command does not expose a Python traceback for ordinary user
errors.

## Configuration File

`agent-libos` and `agent-libos-gui-server` read `config.yaml` from the project
root when it exists. They do not auto-load a `config.yaml` from the current
working directory. Pass `--config <path>` before the command name to use an
explicit YAML overlay instead:

```bash
uv run agent-libos --config ./agent-config.yaml --db local processes
```

The file is a validated overlay on `agent_libos.config.DEFAULT_CONFIG`. Mapping
fields are merged recursively, so adding `llm.profiles.coding` keeps the default
profile. Scalar fields and list/tuple fields replace the default value. Unknown
fields and unsafe values fail at startup. The Pydantic dataclass loader accepts
compatible coercions for ordinary scalar fields, so this is not a strict
JSON-type boundary. In particular, an ordinary integer or float config field
may coerce YAML `true`/`false` to `1`/`0`; do not use booleans as numeric
configuration values. Fields declared with `StrictInt` or `StrictFloat` reject
that coercion, and numeric bounds still receive explicit post-construction
validation. See [Configuration](configuration.md#loading-and-precedence) for
the authoritative strict-field groups.

```yaml
runtime:
  local_store_target: .agent_libos.sqlite
  store_backend: sqlite
  run_until_idle_max_quanta: 10
llm:
  parallel_tool_calls: false
  auto_wait_on_empty_tool_calls: false
  fallback_json_actions: false
  default_profile_id: coding
  profiles:
    coding:
      model: gpt-4.1
      parallel_tool_calls: true
      auto_wait_on_empty_tool_calls: true
      fallback_json_actions: true
```

Explicit CLI options still win over config defaults. Passing `--db local`,
`--db :memory:`, or a filesystem path selects SQLite even when the config
default backend is PostgreSQL. When `--db` is omitted, the CLI uses
`runtime.local_store_target` for SQLite or `runtime.store_dsn` for PostgreSQL.
If `runtime.store_backend: postgres` is selected without `runtime.store_dsn`,
config loading fails. A configured PostgreSQL DSN must use `postgres://` or
`postgresql://`; `runtime.store_dsn` is rejected for the SQLite backend, and a
PostgreSQL URI in `runtime.local_store_target` is rejected instead of silently
overriding `runtime.store_backend`. Explicit `--db` targets accept filesystem
paths, the SQLite sentinels/URI, or URI-form PostgreSQL DSNs; other URI schemes
and libpq `key=value` DSNs fail closed rather than becoming SQLite filenames.
Prefer environment variables or environment-specific config so DSN credentials
are not committed. SQLite and PostgreSQL implement the same runtime store
contract. Ordinary Object Memory payloads are runtime-only; SQL object rows
written by the current runtime store a runtime-memory marker, and marker rows
whose live payload cache cannot be reconstructed are released fail-closed on
reopen. With `llm.persist_full_io: true`, startup may first rehydrate the exact
active root-spawn initial goal from its separately bounded, identity-bound
committed launch envelope; this does not apply to other Objects, child/fork
goals, exec replacement goals, or mutable goal handles. Accepted legacy rows
from older development builds may still contain full JSON payload data; see
[Runtime Storage](storage.md#transaction-model).
Persistent stores also take an active-runtime lease. SQLite derives both the
connection target and path lease from the canonical database path and holds an
exclusive lock on the database actually opened. On the hardened POSIX path, the no-follow
path-sidecar `flock` is paired with an owner-only identity lease keyed by the
validated database `(st_dev, st_ino)`. The database, lease, identity-lease, and
SQLite sidecars must be regular, current-user-owned, single-link files and are
tightened to `0600`, rejecting ordinary aliases and replacement races. The
exclusive lock preserves the single-writer boundary without those POSIX
guarantees and during a same-UID connect-time retarget, but that actor is within
the Host trust boundary: keep the parent owner-only and never rename/replace a
live database path. PostgreSQL uses a session
advisory key scoped to `current_database()` plus `current_schema()`. A second
writable Runtime cannot open the same database/schema target while a live owner
holds the lease. A normal close or process/session termination releases it;
startup recovery then coordinates any durable interrupted state. See
[Runtime Storage](storage.md).
Relative `modules.manifest_paths` entries in the selected config resolve from
the project root, not the shell's current working directory.
By contrast, a relative `--config`, explicit `--module-manifest`, SQLite
`--db`, or `runtime.local_store_target` is rooted at the current working
directory. With the default local substrate that directory is also the Runtime
workspace; `git.worktree_root` is relative to that workspace and must remain
below it. See [Configuration](configuration.md) before combining these settings
from different launch directories.
`llm.parallel_tool_calls` is opt-in and can be overridden per profile. When it
is enabled, OpenAI may return multiple tool calls in one action-selection
response; Agent libOS dispatches them sequentially in the same quantum rather
than running tools concurrently.
`llm.auto_wait_on_empty_tool_calls` is also opt-in and can be overridden per
profile. It helps weaker tool-calling models by synthesizing
`receive_process_messages` when a response has no provider tool calls. The
synthesized wait uses the tool defaults, so
it waits for any unread process message and does not change the raw stored LLM
response.
`llm.fallback_json_actions` is a separate profile-level compatibility opt-in.
By default, text JSON is never executed and a provider that rejects native
`tools` fails the request. When enabled, the prompt includes a compact input-only
compatibility schema and a provider tool-protocol rejection may retry without
native tools.

Shell policy labels are fixed semantic values:
`always_deny`, `allowlist_auto_else_ask`, `blocklist_ask_else_auto`, and
`always_allow`. Configuration may select a default policy and edit exact/prefix
argv rules, but it cannot remap those labels to different meanings. The removed
`checkpoint.auto_high_risk_checkpoint` field was never an implemented
primitive; high-risk confirmation remains explicit at the invoking host/tool
surface, and checkpoints are created only by an explicit checkpoint operation.
Because config overlays reject unknown keys, this former field now fails
validation instead of being silently ignored.

Each configured shell command rule must contain an executable, its first argv
token must be non-blank, and no token may contain NUL. Invalid rules fail config
construction instead of becoming an empty or ambiguous policy match.

Use `--module-manifest` and `--trusted-module` before the command name to load a
trusted Runtime Module that is not already named by the selected config:

```bash
uv run agent-libos --db .agent_libos.sqlite \
  --module-manifest <path-to-module.yaml> \
  --trusted-module <module_id>:<manifest_sha256>:<source_sha256> \
  <command>
```

The checked-in `config.yaml` already names and trusts
`modules/pty/module.yaml`. Do not pass that manifest again while using the
default project config: duplicate startup manifest paths fail closed. Either
use the configured PTY entry as-is or select a config that omits it before
supplying the explicit global options.

`modules verify` reports `manifest_sha256`, `source_sha256`, and `trust_key`;
copy `trust_key` into `--trusted-module`. For multi-file modules,
`source_sha256` is the package digest; for single-file modules it is the
entrypoint file hash. For local development only,
`--trusted-module-sha256 <manifest_sha256>:<source_sha256>` trusts that digest
pair regardless of module id. Prefer `--trusted-module <trust_key>` when the
module id is known.

## Top-Level Commands

```text
init          initialize a runtime database
demo          run the deterministic local demo
audit         print audit records
explain       inspect evidence-backed protected-operation causal trees
llm-calls     print persisted LLM call records
payload-retention preview or apply one bounded LLM/effect retention page
processes     print process table
resources     print process resource budget, usage, and remaining budget
tools         print registered tools
workflow      run a user-facing workflow tool directly
object-task   start, get, list, cancel, wait, or watch-owner Object tasks
task-run      create, supervise, recover, or rerun a Durable Task Run
spawn         spawn a process
cd            set a process working directory
exec          replace a process image and goal
exit          exit a process
llm-once      run one LLM quantum for one process
run           run the process scheduler
message       send a normal human process message
interrupt     send a human interrupt process message
checkpoint    checkpoint subcommands
skills        Skill subcommands
capabilities  capability list, inspect, grant, delegate, revoke, and explain
images        AgentImage list, inspect, validate, register, and commit
jsonrpc       JSON-RPC endpoint and call subcommands
mcp           MCP server and tool-call subcommands
modules       startup Runtime Module inspection and verification
human         process pending human messages manually
semantic      semantic evidence inspection and Host-only review import
store         explicit offline Runtime-store administration
```

The `processes` and `resources` responses, and the resulting process state from
`exit`, include canonical `wait_state`, `outcome`, and `state_generation`
fields. A stale takeover appears as the readable typed
`StaleExecutionProcessWait`/`kind: stale_execution` projection with identity
hashes and generations; it is diagnostic evidence, not client-held permission
to resume. CLI automation must inspect the typed fields and must not parse
`status_message` (including `stale_execution_recovery`) as a control protocol.
The stale wait never contains the pre-takeover raw owner or lease token.

Run `uv run agent-libos <command> --help` for that parser's generated options
and subcommand list. For nested command groups, continue with
`uv run agent-libos <group> <subcommand> --help` to see the leaf parameter
list. The narrative sections below document workflows and security
contracts; lists labelled “useful options” are intentionally not exhaustive.

`demo` writes the deterministic preview to
`agent_outputs/demo_patch_preview.txt` below the Runtime workspace and uses
overwrite mode. Preserve or move an existing file at that path before running
the demo if it contains user data. In this source checkout, project-root
`config.yaml` loads its configured PTY Runtime Module; omitting `--db` also
opens the persistent `.agent_libos.sqlite` selected by that config. Use
`agent-libos --db local demo` when the Runtime records should be in-memory. The
preview file is still written in either mode.

`--actor-pid` is a command-group option for `checkpoint`, `skills`,
`capabilities`, `images`, `jsonrpc`, and `mcp`. Place it after the group name
and before the subcommand; placing it after the subcommand is rejected by
argparse. For example:

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities --actor-pid <actor_pid> grant <subject_pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite checkpoint --actor-pid <actor_pid> inspect <checkpoint_id>
```

## Persistent Runtime Basics

The current Runtime opens only store schema v7. A canonical schema-v6 database
is rejected before `init`, recovery, audit, or any other write until an
operator uses the explicit offline `store migrate --to 7` workflow below. A v5
store must first use `store migrate --to 6`, and v4 must first use
`store migrate --to 5`. Older,
unversioned, and malformed stores remain unsupported; there is no implicit
migration or read-only compatibility mode in ordinary Runtime startup.

`run`, `llm-once`, and `exec --run` invoke the configured real LLM and may
consume provider tokens. Process rows and capabilities survive a file-backed
store reopen, but ordinary Object payloads do not. With the default
`llm.persist_full_io: true`, a committed root `spawn` is the narrow exception:
its bounded launch envelope preserves the exact initial goal until terminal
cleanup, and startup rehydrates it only after matching the active root process,
unchanged goal id, Object identity/version, live state, and runtime-memory
marker for the immutable initial GOAL created by `ProcessManager`. Mutable goal
handles are not eligible. The envelope remains bound to the spawn publication's
initial image, but recovery does not require the process's current image to
remain unchanged.
This permits a later standalone `run`; image requirement declarations are still
declarations rather than grants.
The generic CLI reads inherited environment variables and does not implicitly
load `.env`; export them first or invoke it as
`uv run --env-file .env agent-libos ...`. See [Real LLM
Configuration](../README.md#real-llm-configuration).

```bash
uv run agent-libos --db .agent_libos.sqlite init
uv run agent-libos --db .agent_libos.sqlite spawn --image coding-agent:v0 --goal "Summarize README.md"
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> human:owner --rights write
uv run agent-libos --db .agent_libos.sqlite run --max-quanta 10
uv run agent-libos --db .agent_libos.sqlite processes
uv run agent-libos --db .agent_libos.sqlite resources <pid>
uv run agent-libos --db .agent_libos.sqlite audit
uv run agent-libos --db .agent_libos.sqlite tools
uv run agent-libos --db .agent_libos.sqlite workflow run get_working_directory
```

The initial `spawn` prints the pid to substitute for `<pid>`. The exact
Host-issued README and Human-output grants from the preceding commands satisfy
and narrow the coding image's declarations; the declarations themselves issued
no authority. With `llm.persist_full_io: false`, spawn writes only content-free
identity, hash, and size metadata and a later CLI invocation cannot recover the
goal. In that mode, use an embedded `Runtime`, the GUI server, or
`scripts/run_coding_agent.py` for one Runtime lifetime; for the built-in coding
image, the bootstrap-plus-`exec --no-preserve-memory --preserve-capabilities
--run` pattern is also available because exec creates and uses its replacement
goal before closing. Terminal cleanup changes any reversible root-goal envelope
to hash-only. Failed launch rollback and startup compensation also redact it
before committing a non-committable publication state. `human:owner` is the
checked-in `runtime.default_human: owner`; substitute the configured Human
resource when that setting changes.

`run` uses the high-level runtime scheduler, so human terminal messages are
processed as part of runtime execution. Without `--max-quanta`, it uses
`runtime.run_until_idle_max_quanta`; the checked-in config may bound that value.
Only a configured `null` value means run until the Runtime becomes idle without
a quantum limit. Pass `--max-quanta <n>` to set an explicit bound on the total
number of LLM/tool quanta across all runnable processes. Interactive run,
`exec --run`, and `message --run` use the same default.

`cd <pid> <path>` requires filesystem `read` authority for the selected
directory. Explicit working directories for child processes and PTY creation
are checked through the same directory primitive after their higher-level
spawn/image or shell authority gates.

Manual queue processing remains available:

```bash
uv run agent-libos --db .agent_libos.sqlite human
```

`spawn` and `exec` accept `--llm-profile <profile-id>` for host-selected
per-process LLM routing. The process row stores only the profile id; API keys
remain in host environment variables configured by the profile. If omitted,
spawn uses the selected image's `llm_profile` default and then
`config.llm.default_profile_id`; exec keeps the current process profile unless
overridden. Only the configured default profile inherits the documented legacy
`OPENAI_*` endpoint, model, account-routing, and provider-policy environment
variables, including `OPENAI_ENABLE_THINKING`. Other named profiles should
declare their model and endpoint explicitly; they still read their credential
from the exact variable named by `api_key_env`.

The CLI reads profiles from `config.yaml` or `--config`; it does not read the
GUI's user-level profile file. The GUI stores profiles created from its model
manager in the operating system's user application config directory and passes
that file to `agent-libos-gui-server --llm-profiles-file`. Both surfaces still
persist only `llm_profile_id` on processes, and both read real API keys from
environment variables named by `api_key_env`.

Example config entries:

```yaml
llm:
  default_profile_id: gpt-5.5
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

## Semantic Inspection and Review Evidence

The `semantic` group prints schema-versioned JSON. All runtime and policy
surfaces are read-only; `semantic review import` is the sole write and can only
append strict Host review evidence. It cannot activate/revoke an epoch, alter
control state, settle a request, issue a Capability, mutate labels, or call a
provider.

```bash
uv run agent-libos --db .agent_libos.sqlite semantic status
uv run agent-libos --db .agent_libos.sqlite semantic assessments \
  --pid <pid> --request-id <request_id> --operation-id <operation_id> \
  --kind approval --status success --domain filesystem \
  --action-id filesystem.read \
  --tenant-bucket-sha256 <lowercase_sha256> --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic assessments \
  --after <opaque_next_cursor> --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic show <assessment_id>
uv run agent-libos --db .agent_libos.sqlite semantic flow status
uv run agent-libos --db .agent_libos.sqlite semantic flow entities --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic flow edges --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic flow lineage <node_id>
uv run agent-libos --db .agent_libos.sqlite semantic settlements --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic policy epochs --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic control status
uv run agent-libos --db .agent_libos.sqlite semantic control history --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic health --limit 50
uv run agent-libos --db .agent_libos.sqlite semantic metrics
uv run agent-libos --db .agent_libos.sqlite semantic review import \
  --file <strict-json-review-evidence>
```

Every assessment filter is optional. `--limit` defaults to 50 and must be from
1 through 100. IDs and filters are bounded, and `--after` is an opaque bounded
keyset cursor returned by the previous page; clients must not parse or
synthesize it. Output contains the action id, typed findings, Shadow outcome,
normalized observed Human outcome, calibration bucket, reserved nullable
token/cost fields, latency, and provenance digests, but never a prompt,
goal/provider body, raw Human or classifier response, safe projection, or model
reasoning. The action filter is a dotted lower-case ontology id and the tenant
filter is an exact lower-case SHA-256 bucket digest; neither accepts a display
name.

The assessment-row schema-v1 token/cost fields are reserved and nullable.
External classifier rows may populate `input_tokens`, `output_tokens`, and
`cost_microunits` only from an exact Host `LLMCompletion` whose exact usage
dictionary contains bounded non-negative integer counters. The accepted token
aliases are `prompt_tokens → input_tokens` and
`completion_tokens → output_tokens`; canonical/alias disagreement or an
invalid value—including anything above `2^53 - 1`—makes only that counter JSON
`null`. Deterministic/scripted,
missing, non-exact, and otherwise untrusted telemetry remains `null`. Unknown
usage keys and the raw usage object are never emitted. Treat populated values
as provider-reported operational telemetry, not authoritative billing data.

`would_issue_exact_once` remains Shadow evidence, not an approval. Real
Phase 3/4 settlement is reachable only through the Runtime's Host-bound
machine-policy port under `enforce_deny` or `canary_auto`; no CLI, HTTP, GUI,
model Tool, Skill, JIT, or Runtime Module can invoke it. Status uses schema v3
and includes queue, assessment, control, FlowGraph, machine, real approval, and review
metrics together with complete closed-key `by_status` and `by_domain`
maps. Each must sum to the assessment total, and scalar success/error,
Shadow-outcome, and OOD counters must agree with those maps; malformed or
inconsistent service output is rejected. Real rates and safety-review rates
are `null` when their respective denominators are zero or unavailable; they
must never be displayed as “0% unsafe.” See [Semantic Approval and Data
Identification](semantic_shadow.md).

## Offline Store Migration

`store migrate` is handled before `Runtime.open()` and is the only supported
schema-v4 to v5, v5 to v6, and v6 to v7 path. Migrations are ordered; a v4
store must independently plan/apply `--to 5`, then `--to 6`, then `--to 7`.
Stop every CLI/GUI/embedded Runtime using the target first. Dry-run
validates the complete canonical source shape against a private snapshot,
performs zero writes beside the source, and returns a deterministic
`plan_sha256`.

Plan schema v2 binds that digest to the hashed database/schema and cluster
identity, canonical source catalog, source logical or locked PostgreSQL
relation-state digest, source-observation receipt, plan-bound migration-receipt
contract, migration implementation, and product version. PostgreSQL planning
and apply hold the Runtime advisory lease and schema-qualified relation locks
while capturing relation OIDs plus visible `ctid`/`xmin` identities in a
repeatable-read transaction; row payloads are not recorded. Apply writes the
exact plan-bound audit receipt atomically with DDL and the marker update. If
commit acknowledgement or post-commit validation fails after the transaction
actually committed, repeat `--apply` with the same reviewed digest and recovery
evidence. Only an exact target receipt, reconstructed source state, canonical
catalog, and postcondition return `applied=false` and `already_applied=true`;
this is uncertain-commit reconciliation, not permission to reuse a plan for
another, generic same-version, or subsequently modified database.

For the current v6-to-v7 step:

```bash
# Create an independent, quiesced, owner-only backup first.
sqlite3 .agent_libos.sqlite ".backup '.agent_libos.v6.backup.sqlite'"
chmod 600 .agent_libos.v6.backup.sqlite

uv run agent-libos --db .agent_libos.sqlite store migrate --to 7 \
  --dry-run --sqlite-backup .agent_libos.v6.backup.sqlite
uv run agent-libos --db .agent_libos.sqlite store migrate --to 7 \
  --apply --expected-plan-sha256 <plan_sha256> \
  --sqlite-backup .agent_libos.v6.backup.sqlite
```

The PostgreSQL form uses the same target/digest flow and requires
`--postgres-snapshot-confirmed` on apply. See the complete
[v6-to-v7 runbook](storage.md#offline-v6-to-v7-migration) for the MCP v7 table,
catalog, lease, backup, and rollback checks.

For the preceding v5-to-v6 step:

```bash
# Create an independent, quiesced, owner-only backup first.
sqlite3 .agent_libos.sqlite ".backup '.agent_libos.v5.backup.sqlite'"
chmod 600 .agent_libos.v5.backup.sqlite

uv run agent-libos --db .agent_libos.sqlite store migrate --to 6 \
  --dry-run --sqlite-backup .agent_libos.v5.backup.sqlite
uv run agent-libos --db .agent_libos.sqlite store migrate --to 6 \
  --apply --expected-plan-sha256 <plan_sha256> \
  --sqlite-backup .agent_libos.v5.backup.sqlite
```

The PostgreSQL v5-to-v6 form uses the same target and plan digest, plus the
explicit operator snapshot acknowledgement:

```bash
uv run agent-libos --db "$AGENT_LIBOS_POSTGRES_DSN" store migrate --to 6 --dry-run
uv run agent-libos --db "$AGENT_LIBOS_POSTGRES_DSN" store migrate --to 6 \
  --apply --expected-plan-sha256 <plan_sha256> \
  --postgres-snapshot-confirmed
```

The legacy v4-to-v5 step remains:

```bash
# Create an independent, quiesced, owner-only backup first.
sqlite3 .agent_libos.sqlite ".backup '.agent_libos.v4.backup.sqlite'"
chmod 600 .agent_libos.v4.backup.sqlite

uv run agent-libos --db .agent_libos.sqlite store migrate --to 5 \
  --dry-run --sqlite-backup .agent_libos.v4.backup.sqlite
```

Review the plan and pass its exact digest to apply:

```bash
uv run agent-libos --db .agent_libos.sqlite store migrate --to 5 \
  --apply --expected-plan-sha256 <plan_sha256> \
  --sqlite-backup .agent_libos.v4.backup.sqlite
```

For apply, both the SQLite source and independent backup must be regular files
owned by the current user, have exactly one hard link, and have mode `0600` on
POSIX. The backup must contain no live journal/WAL/SHM sidecar and must still
match the locked source's canonical logical v4 digest. Dry-run is zero-write:
it reports an insecure source but never changes its permissions, so the
operator must run `chmod 600` before apply. Apply obtains the exclusive offline
lease, repeats validation, runs DDL plus marker CAS `4 -> 5` in one transaction,
validates the complete v5 shape, then commits. Any failed check or injected
DDL/readback failure rolls back the schema and marker.

For PostgreSQL, create and verify an operator-managed snapshot of the exact
database/schema first. Dry-run needs no acknowledgement; apply additionally
requires `--postgres-snapshot-confirmed`:

```bash
uv run agent-libos --db "$AGENT_LIBOS_POSTGRES_DSN" store migrate --to 5 --dry-run
uv run agent-libos --db "$AGENT_LIBOS_POSTGRES_DSN" store migrate --to 5 \
  --apply --expected-plan-sha256 <plan_sha256> \
  --postgres-snapshot-confirmed
```

The PostgreSQL path takes the Runtime advisory lock and full frozen-relation
locks, validates canonical v4 and the locked source-state receipt, performs the
same single-transaction DDL/audit-receipt/marker CAS/v5 readback, and releases
the locks. The migration role must be able to execute
`pg_catalog.pg_control_system()`; identity fails closed when the cluster system
identifier is unavailable, and emitted plans contain no DSN or raw host value.
The role must also own, hold `MAINTAIN` on, or hold the applicable write
privilege for every frozen required table so dry-run and apply can take their
`ACCESS EXCLUSIVE` locks.
The acknowledgement records only that the operator confirms an external
snapshot exists; Agent libOS does not create or validate that snapshot artifact.
Never use `init` or an ordinary Runtime command as a migration tool.
The full runbook is in [Runtime Storage](storage.md#offline-v4-to-v5-migration).

## Workflow Run

`workflow run` is a direct user entrypoint for tools. It spawns a fresh
AgentProcess from the selected image, calls one tool bound in its complete
process tool table through the normal ToolBroker path, and returns the tool
result JSON. It does not run the LLM scheduler or consult the narrower model
tool projection; direct invocation does not make that schema model-visible. It
also does not bypass primitive capability checks, resource budgets, human
approval, result-object persistence, events, or audit.

```bash
uv run agent-libos --db .agent_libos.sqlite \
  workflow run get_working_directory

uv run agent-libos --db .agent_libos.sqlite \
  workflow run parse_pytest_log \
  --image coding-agent:v0 \
  --args-json '{"log":"FAILED tests/example.py::test_case"}'
```

By default the process goal is `workflow:<tool>`, so tool arguments are not
copied into the goal object. Use `--goal` only when the workflow process needs a
human-readable label. `ok:false` is still printed as JSON and exits the CLI with
status code 1.

## Durable Task Runs

`task-run` is the Host CLI for a first-class Durable Task Run. It is separate
from both the one-tool `workflow run` command and Object-bound background
tasks. A Run supervises a root AgentProcess tree, persists a versioned goal and
requirements, and exposes safe restart recovery through store schema v7.

```text
task-run start
task-run get <run_id>
task-run list
task-run wait <run_id>
task-run recovery-options <run_id>
task-run pause <run_id>
task-run resume <run_id>
task-run cancel <run_id>
task-run follow-up <run_id>
task-run recover <run_id>
task-run rerun <run_id>
```

Useful leaf options are:

| Command | Durable controls |
| --- | --- |
| `start` | Exactly one of `--goal`/`--goal-json`, required `--title`, optional Image/launch/authority/deadline/retention fields, stable `--client-request-id`, and explicit `--run [--max-quanta N] [--run-command-id ID]`; launch JSON accepts only `capabilities`, `resource_budget`, `working_directory`, and `llm_profile_id` and must not contain credentials |
| `list` | Repeated or comma-separated `--status`, opaque `--cursor`, and bounded `--limit` |
| `wait` | Optional finite `--timeout`; it has no run command id because it never mutates or dispatches |
| `recovery-options` | Read-only, server-derived recovery choices for the Run; use the returned opaque `option_id` with `recover` |
| `pause` / `resume` | Required `--expected-revision`; optional stable `--command-id` is generated for a one-shot invocation when omitted |
| `cancel` | Revision/command identity, optional `--reason`, and required `--confirm` |
| `follow-up` | Text or strict JSON content, optional `--interrupt` or `--optional`, and revision/command identity |
| `recover` | An exact server-provided option, optional provider-verifiable `--receipt-json`, revision/command identity, and required `--confirm` |
| `rerun` | Revision/command identity, optional linked-create `--client-request-id`, and optional `--spec-overrides-json` |

For example, after enabling `task_runs.plaintext_payloads_enabled` in the Host
configuration:

```bash
uv run agent-libos --db .agent_libos.sqlite task-run start \
  --goal "inspect the repository and fix the reported failure" \
  --title "repair reported failure" \
  --retention purge_on_terminal

uv run agent-libos --db .agent_libos.sqlite task-run start \
  --goal "continue until the acceptance checks pass" \
  --title "release validation" --run --max-quanta 20

uv run agent-libos --db .agent_libos.sqlite task-run follow-up <run_id> \
  "also document the compatibility boundary" \
  --expected-revision <revision> --command-id <stable-id>
```

`start` creates a queued Run by default. `--client-request-id` supplies the
stable create identity; omission generates one for that invocation. Creation
does not require an existing revision. Pass `--run` to
consume scheduler quanta in the current CLI process; `--max-quanta` is the
explicit per-command bound. Omitting it supplies no quantum-count cap to the
Task Run command, which then runs only until a typed wait, terminal, pause,
deadline, or `needs_attention` boundary. Creation does not install a daemon.
Likewise, `wait` cannot keep work running after the one-shot CLI exits and
never dispatches scheduler quanta, provider calls, or tools. Its observation
may still persist safe deadline cancellation, settlement projection, and
terminal-retention housekeeping. Waiting states and
`needs_attention` are successful structured command responses rather than
uncaught internal errors.

Every mutation carries the current `expected_revision` and a stable
`command_id`. Retrying the same canonical command is idempotent; reusing its id
for different arguments fails. Cancel and evidence-constrained recovery require
explicit confirmation. A dispatched or unknown external effect offers no
ordinary Retry/Resume shortcut.

After a timeout, connection loss, or CLI crash with an ambiguous result, retry
with the exact same command id and arguments; generating a new id is a new
mutation, not a safe transport retry. For operations whose durable command
receipt is still pending, that exact retry can finish only local settlement and
return the stored revision-bound result. It does not consume another quantum or
repeat an LLM, tool, Provider, or external effect. Callers that may need this
recovery should supply `--command-id` explicitly (or `--run-command-id` for
`start --run`) rather than rely on a one-shot generated value they did not
retain. A retryable `start --run` invocation must retain and reuse both its
`--client-request-id` and its distinct `--run-command-id`.
For `recover` with a linked-Run option, the same exact retry may reconstruct a
lost outer receipt only from the request-hash-bound nested rerun, target create
receipt, and causal link already in the Store. It does not create a second Run;
submitting a new command id is a new recovery request and has no such identity
guarantee.

Durable payload persistence is disabled by default. Until the Host opts into
the documented plaintext-at-rest boundary, Run creation is rejected. The
default `purge_on_terminal` retention removes readable Run payloads before the
terminal status commits; `permanent` is a Host/admin-only creation option. See
[Durable Task Runs](durable_task_runs.md) for the complete lifecycle and
recovery contract.

For rerun, use the summary's `payloads_purged` field rather than assuming that
only `purge_on_terminal` can remove content. A terminal `permanent` Run may
also have been explicitly purged; in either case supply a replacement goal,
for example `--spec-overrides-json '{"goal":"rebuild from current state"}'`.

## Object Tasks

`object-task` commands expose Object-bound background tool tasks. A task belongs
to an existing Object Memory object, runs one tool bound in the creator's
complete process tool table through a dedicated runner child process, and
reports status as JSON. The Host-managed runner does not infer model visibility
from that binding. Starting a task requires a live owner Object payload and a
live creator process in the same Runtime; an object id recorded in a database is
not sufficient.

```bash
uv run agent-libos --db .agent_libos.sqlite object-task list --pid <pid>
uv run agent-libos --db .agent_libos.sqlite object-task get <task_id> --pid <pid>
uv run agent-libos --db .agent_libos.sqlite object-task wait <task_id> --pid <pid>
```

Those one-shot reads are useful for terminal records. For completeness, the
parser exposes these lifecycle command shapes:

```text
object-task start --pid <pid> --owner-oid <live_oid> <tool> --wait
object-task cancel <task_id> --pid <pid>
object-task watch-owner <task_id> --pid <pid> --watch-events updated
```

This is syntax, not a standalone shell recipe: the installed one-shot CLI has
no command that creates an Object and then starts a task before that Runtime
closes. On reopen, a marker-only Object without its runtime payload is released
fail-closed. An embedding host can create the owner and start/supervise the
ObjectTask in one Runtime lifetime. With the GUI server, keep its Runtime alive
while a process creates the owner through normal model/tool execution, then call
`POST /api/object-tasks/start`; the server has no Host endpoint that creates an
arbitrary owner payload directly.

The task still uses the creator process tool table, ToolBroker, capabilities,
resource budgets, events, audit, and Object Memory result semantics. A
task started with `--watch-owner` receives owner `updated` and `linked` notices
as process messages in its runner process; those notices contain object ids and
event metadata, not object payloads or new capabilities. A
`watch-owner` subcommand can update or disable that watch while the task is
still active. Owner-watch auto-resume is limited to tools with safe
message-receive replay semantics, currently `receive_process_messages`.
Running synchronous side-effect tools are not force-cancelled because Python
cannot safely stop their worker thread after side effects may have started. The
CLI parser requires `--wait` on `object-task start` because it cannot preserve a
detached worker after shutdown, but that does not solve the live-owner
precondition above. For file-backed SQLite, the active-runtime lease also
prevents a separate CLI process from opening the database while a live GUI or
embedded host owns it. After that owner stops, reopening reconciles unfinished
ObjectTasks as abandoned; `list`, `get`, and `wait` may inspect terminal or
reconciled records, while active `cancel` and `watch-owner` belong on the live
Host surface.

## LLM Calls

Inspect persisted LLM action-selection calls:

```bash
uv run agent-libos --db .agent_libos.sqlite llm-calls --pid <pid>
uv run agent-libos --db .agent_libos.sqlite llm-calls --limit 20
```

With the default full-I/O policy, records include provider ids, model/API mode,
token usage when available, full prompts, visible tools, output, tool calls,
reasoning, raw responses, and bounded observability envelopes. The envelopes
contain preview, byte count, hash, and truncation metadata.
For OpenAI Responses requests, request options may show strict tool-schema
counts, whether prompt-cache or safety identifiers were configured and actually
accepted after compatibility retries, and whether
`responses_previous_response_id` was configured but disabled for full-snapshot
execution. Configured cache keys and safety-identifier values are not persisted
there. The current AgentProcess executor always leaves
`openai_previous_response_id` null and sends a complete local snapshot.
Ordinary Runtime prompt modes represent prior tool results in bounded local
context; `image_only` instead replays a provider-native paired tool transcript
without provider-side state. Responses receives `function_call`/
`function_call_output` input items, while Chat receives assistant/tool messages.
Request options also show whether `parallel_tool_calls` and
JSON action fallback were enabled, and whether the fallback was used. Canonical usage preserves provider-reported
`cache_read_tokens` and `cache_write_tokens`, including an explicit zero, even
when full-I/O persistence is disabled.
Full LLM input/output persistence is enabled by default for self-evolution
training and fine-tuning pipelines under the deployment's user agreement. Set
`config.llm.persist_full_io=False` when a user or operator opts out of storing
sensitive prompt, tool, reasoning, successful response, and raw provider
response fields. Provider and extension exception text is never persisted or
exposed to the model, regardless of this setting. The opt-out writes canonical
content-free summary envelopes with byte counts, JSON shape/count metadata when
available, and hashes; it does not keep readable previews of those fields. An
`image_only` Image cannot run under this
opt-out because it requires an exact durable transcript head; choose another
prompt mode when content-free write-time persistence is required.

## Payload Retention

Payload retention is disabled by default and never runs on startup. Configure
the Host policy under `runtime.payload_retention_*`, then preview one bounded
page before applying it:

```bash
uv run agent-libos --config config.yaml --db .agent_libos.sqlite payload-retention llm_call
uv run agent-libos --config config.yaml --db .agent_libos.sqlite payload-retention llm_call --apply
uv run agent-libos --config config.yaml --db .agent_libos.sqlite payload-retention external_effect --apply
```

The command returns counts and an optional `next_cursor`. Continue with both
`--after-created-at <value>` and `--after-record-id <value>`; providing only
one cursor component fails closed. Dry runs still write a metadata-only audit
summary, while `--apply` commits payload CAS updates and that audit in one
transaction. Live/unknown effects, active `image_only` transcript heads,
compatible Responses-continuation anchors, and process-result recovery calls
are not eligible. See
[Evidence and LLM Payload Retention](evidence_payload_retention.md).

## Explainable Operations

List causal-root operations for one process or explain one causal tree:

```bash
uv run agent-libos --db .agent_libos.sqlite explain process <pid>
uv run agent-libos --db .agent_libos.sqlite explain operation <operation_id>
```

Resolve an explicit evidence id:

```bash
uv run agent-libos --db .agent_libos.sqlite explain call <call_id>
uv run agent-libos --db .agent_libos.sqlite explain effect <effect_id>
uv run agent-libos --db .agent_libos.sqlite explain request <request_id>
uv run agent-libos --db .agent_libos.sqlite explain audit <record_id>
uv run agent-libos --db .agent_libos.sqlite explain event <event_id>
uv run agent-libos --db .agent_libos.sqlite explain reservation <reservation_id>
uv run agent-libos --db .agent_libos.sqlite explain context <materialization_id>
```

Process lists accept `--limit` and `--cursor`; detail/evidence lookups accept
`--evidence-limit` and `--cursor`. An ambiguous evidence id prints the explicit
causal-root candidates and exits with status 2. A not-found id prints a
structured `NotFound` error and exits with status 1. Detail pagination bounds
the returned presentation only: the current backend constructs the causal tree
before slicing its evidence projection, so `--evidence-limit` is not a bound on
backend traversal, memory, or query work. Explain output is metadata/redaction
oriented even when full LLM I/O persistence is enabled. See
[explainable_operations.md](explainable_operations.md).

## Task Authority Manifest at launch

`spawn` and `workflow run` accept a Host-authored manifest JSON object:

```bash
uv run agent-libos spawn --goal "review" \
  --authority-manifest-json '{"authorized_capabilities":[{"resource":"filesystem:workspace:reports/*","rights":["read"]}]}'
```

The default mode is `manifest_required`. Image requirements are reported but
not granted. Omitting `--authority-manifest-json` asks the runtime to bind an
implicit manifest. Explicitly passing `{}` instead binds an explicit empty
authority ceiling: it authorizes no launch capabilities or later model
permission requests, although omitted `permitted_effects` remains unrestricted.
Use `"permitted_effects": []` to deny every provider effect class.

For `workflow run`, omission also permits the trusted workflow controller to
construct the exact target-image read authority needed by `exec_process`.
Explicit `{}` suppresses that controller-derived authority. In both cases the
authority is part of the launch manifest rather than a grant after spawn. See
[task_authority_manifest.md](task_authority_manifest.md).

## Process Resources

Inspect one process's configured budget, observed usage, and remaining budget:

```bash
uv run agent-libos --db .agent_libos.sqlite resources <pid>
```

Resource accounting is process-tree scoped. Tool calls, LLM calls/tokens,
subprocess wall/CPU/RSS usage, filesystem bytes, JSON-RPC bytes, MCP bytes,
Deno syscalls, and child-process creation are charged to the acting process and
its ancestors. Capabilities, Skill activation, image exec, checkpoint restore,
and human approval do not increase these budgets.

For each executor-level logical LLM call, Runtime reserves one call plus the
configured prompt/completion/total token envelope before Provider dispatch.
Active reservations reduce the displayed remaining budget for the full process
ancestry. Valid usage settles exactly, certified non-start releases, and an
unknown outcome charges the aggregate maximum. Agent libOS's explicit,
separately traced transport retries remain one logical executor call; provider
SDK internal retries are disabled. These counters therefore are not exact
Provider request or monetary-cost meters.

Calls, tokens, syscalls, bytes, child counts, and peak-memory values are
non-negative integers. Runtime and subprocess wall/CPU seconds are continuous
finite non-negative values and may be fractional. The final `ResourceBudget`
model and resource-related tool/manifest inputs reject boolean values instead
of accepting them as Python integers. This is distinct from loading a config
overlay: ordinary non-strict Pydantic config fields may first coerce YAML
booleans to `0`/`1`; strict config fields reject them. Do not use booleans for
numeric configuration values.

## Interactive Run

For a Codex CLI-style loop:

```bash
uv run agent-libos --db .agent_libos.sqlite run --interactive --pid <pid> --max-quanta 20
```

`--pid` may be omitted only when the Runtime has exactly one non-terminal
process. When there are zero or multiple active processes, select the target
explicitly before entering the interactive loop.

Plain text sends a normal message unless a human question or approval is
pending, in which case it answers that request.

Interactive slash commands:

- `/help`: show available interactive commands.
- `/message <text>`: force a normal process message.
- `/interrupt <text>`: send an interrupt message.
- `/pid <pid>`: switch the default target process.
- `/answer <text>` or plain text while a human question is pending: answer the
  pending request.
- `/approve` and `/reject`: resolve an ordinary approval or a permission
  request. For a permission request, they map to `always_allow` and
  `always_deny`. In `enforce_deny` and `canary_auto`, an external-operation
  approval first renders the Host canonical preview and submits its exact
  request revision and `preview_sha256`; a stale preview leaves the request
  unchanged and must be refreshed.
- `/allow` and `/ask`: permission requests only; they map to `always_allow` and
  `ask_each_time`. `allow_once` is not a terminal response policy. A command
  that is not valid for the displayed request leaves that request pending.
- `/exit`: leave the interactive loop.

## Process Messages

```bash
uv run agent-libos --db .agent_libos.sqlite message <pid> "Please inspect the latest result"
uv run agent-libos --db .agent_libos.sqlite interrupt <pid> "Stop current work and read this first"
uv run agent-libos --db .agent_libos.sqlite message <pid> "Use this as job input" --channel human --correlation-id job-42 --run
```

Useful options:

- `message --kind normal|interrupt` (`interrupt` is already fixed to the
  interrupt kind and does not accept this option)
- `--human <name>`
- `--channel <channel>`
- `--subject <text>`
- `--correlation-id <id>`
- `--reply-to <message_id>`
- `--payload-json <object>`
- `--run`
- `--max-quanta <n>`: optional; omitted uses
  `runtime.run_until_idle_max_quanta`, whose `null` value means unbounded until
  idle.

## Process Builtins

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> 'filesystem:workspace:agent_libos/*' --rights read
uv run agent-libos --db .agent_libos.sqlite cd <pid> agent_libos
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> image:review-agent:v0 --rights read
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> human:owner --rights write
uv run agent-libos --db .agent_libos.sqlite exec review-agent:v0 "Review runtime/runtime.py" \
  --pid <pid> --no-preserve-memory --preserve-capabilities --run --max-quanta 10
```

For `exec`, the first positional argument is the target image. It can be an
already registered image id such as `coding-agent:v0`, or an image package
directory containing `IMAGE.yaml`. The second positional argument is the
replacement goal.

`cd` checks `read` on the exact canonical directory resource; the first grant
above covers the checked-in `agent_libos/` directory. An exec that changes the
image checks `read` on `image:<target_image_id>`, which is why the example grants
the review image explicitly. Exec preserves Object Memory by default, drops
external capabilities by default, and does not run unless `--run` is present.
Here `--no-preserve-memory` replaces any stale prior view with the live goal,
while `--preserve-capabilities` keeps the Host-issued directory, image, and
Human grants needed by the new process. Those grants satisfy and narrow the
review image's declarations; the declarations themselves issue no authority.
An `exit <pid> --payload '{"done":true}'` command is valid only while
the process is still non-terminal.

Useful exec options:

- `--replace-image`
- `--args-json <object>`
- `--preserve-memory` / `--no-preserve-memory`
- `--preserve-capabilities`
- `--run` / `--no-run`
- `--max-quanta <n>`: optional when `--run` is set; omitted uses
  `runtime.run_until_idle_max_quanta`, whose `null` value means unbounded until
  idle.

Exec never grants target-image `required_capabilities` automatically. If the
target is an image package, its `workspace/` seed is materialized into a private
per-process directory under `image.materialized_workspace_root` (by default
`agent_outputs/image_workspaces/`). For any target image, `required_modules` are
checked before boot; the runtime must already have loaded each declared
`(module_id, source_sha256)` pair.

Exec is admitted only for a Host-owned `runnable` process or from the process's
exact active execution lease. A child/message/Human/Tool/pause/Host-resume wait
must be resolved or resumed first; exec rejects it before publication so its
durable dependency cannot be orphaned by image replacement.

Exit accepts either `--payload` or `--result-oid`, not both. Non-JSON payload
text is wrapped as `{"content": "<text>"}`. The response includes canonical
tagged `wait_state` and `outcome` values plus `state_generation`; when exit
materializes a message-only result Object, `result_oid` is that exact durable
outcome Object id.

## AgentImage Packages

User-defined images are directory packages:

```text
images/custom-review-agent/
  IMAGE.yaml
  prompt.md
  tools/
    jit-tools.json
    scripts/
      summarize.ts
  resources/
  workspace/
    seed.txt
```

`IMAGE.yaml` holds structured metadata and references `prompt.md`:

```yaml
image_id: custom-review-agent:v0
name: custom-review-agent
prompt: prompt.md
prompt_mode: image_only
jit_tool_exposure: direct
llm_profile: review-fast
planner:
  context_management:
    mode: auto_compact
    threshold_ratio: 0.8
    tool:
      name: compact_process_context
      arguments: {}
default_tools:
  - compact_process_context
  - read_memory_object
  - human_output
required_modules:
  - module_id: example-module:v0
    source_sha256: "<source_sha256 from modules verify>"
jit_tools: tools/jit-tools.json
workspace:
  source: workspace
  working_directory: .
  grants:
    - path: .
      rights: [read, write]
      recursive: true
```

`tools/jit-tools.json` declares process-local TypeScript JIT tools whose source
files live under `tools/scripts/*.ts`. JIT tools are snapshotted as immutable
package content and are not copied into the materialized workspace.
Each entry may set a positive `timeout_s` for its outer Deno execution window;
omission uses `tools.deno_timeout_s`, and the configured value may not exceed
`tools.deno_timeout_hard_limit_s`. Process resource budgets remain an
independent, potentially lower bound.

`prompt_mode` is optional and defaults to `image_only` for custom packages.
In that mode, the Image prompt is the exact system message, the original goal
is the first user message, and later turns use a durable native tool-call
transcript. The Runtime does not inject Object Memory, Skill, Capability,
fallback-protocol, or explanatory prompt sections. String goals are preserved
exactly and structured goals use canonical JSON. Lossless replay requires
`llm.persist_full_io: true`; otherwise execution fails before the LLM provider.
An Image using `image_only` cannot select prompt-mode context management.
Use `minimal_runtime` when the image wants the cumulative completion contract,
factual runtime/state, activated-Skill and recovered-goal sections, plus optional
Host-enabled fallback JSON guidance, but not the complete native Agent libOS
base/action-planner envelope. Use `libos_default` when it wants that full
planner prompt as well. Unactivated Skill metadata remains discoverable through
`discover_skills`; it is not injected. Only `image_only` is byte-transparent.

`jit_tool_exposure` is optional and defaults to `direct`. Use `multiplexed`
when the image wants one stable OpenAI tool schema named `run_jit_tool` for all
JIT tools. Multiplexed packages must describe their JIT catalog in their own
prompt or Skill instructions; the runtime does not inject the individual JIT
names or schemas into prompt context.

`llm_profile` is optional and names a host-configured LLM profile used when a
root process is spawned from the image. It is only an id; provider API keys stay
in the host environment and are not packaged into the image.

`planner.context_management` is optional. It accepts only `auto_compact`,
`prompt`, or `disabled`, a ratio in `(0, 1]`, an OpenAI-compatible tool name,
object arguments, and a literal prompt. Other planner keys remain
image-defined and compatible. Automatic mode does not grant the named tool;
include it in `default_tools` (or otherwise in the process's complete tool
table) when the image intends it to run. The `prompt` mode is invalid with
`prompt_mode: image_only` because it would add Runtime-authored model input.

`required_modules` is optional. Each entry must contain a `module_id` and the
64-character hexadecimal `source_sha256` reported by
`uv run agent-libos modules verify <module.yaml>`. Spawn and exec check that
the current runtime has already loaded the exact trusted module source; image
registration accepts either hex case and stores the normalized lowercase
digest. Image boot never loads modules automatically.

`default_tools` is the exact initial static table. The runtime does not add
`process_exit`, `create_memory_object`, or any other builtin automatically. List
every builtin that must be bound in that complete table. Without
`metadata.tool_projection: skills`, those bindings are also the initial model
projection; with it, only the five required bootstrap tools are initially
model-visible and applicable immutable built-in Skills project their complete
owned subsets later. Package JIT tools can still use authorized libOS syscalls
internally without being mirrored as builtin tools in the process tool table.

## Image Commands

```bash
uv run agent-libos --db .agent_libos.sqlite images list
uv run agent-libos --db .agent_libos.sqlite images inspect coding-agent:v0
uv run agent-libos --db .agent_libos.sqlite images validate images/mini-swe-agent
uv run agent-libos --db .agent_libos.sqlite images register images/mini-swe-agent
uv run agent-libos --db .agent_libos.sqlite images commit <checkpoint_id> stateful-agent:v0 --name stateful-agent
```

| Image subcommand | Arguments and option defaults |
| --- | --- |
| `list` | no leaf options |
| `inspect` | required `image_id` |
| `validate` | required Host/package `path` |
| `register` | required `path`; `--replace` is off by default |
| `commit` | required `checkpoint_id`, `image_id`, and `--name`; `--version` defaults to `v0`, `--metadata-json` to `{}`, and `--replace` is off |

`images` accepts `--actor-pid <pid>` before the subcommand. Process mode uses
the following authority contract:

| Image subcommand | Process-mode authority |
| --- | --- |
| `list` | image registry `read` |
| `inspect` | exact image `read` |
| `validate` | filesystem `read` for the package manifest and package tree, resolved from the actor process working directory |
| `register` | the same filesystem reads plus exact target-image `write`; `--replace` requires target-image `admin` |
| `commit` | checkpoint or checkpoint-owner process `read` plus exact target-image `write`; `--replace` requires target-image `admin` |

`images commit` creates a checkpoint-derived image artifact from the checkpoint
owner root process. It captures owned/captured internal Object Memory (not an
uncaptured borrowed root), loaded Skills,
process-local JIT tools, tool visibility, and cwd. It does not package
filesystem/provider state. External capabilities from the checkpoint are stored
as `required_capabilities` declarations and are not granted automatically when
the committed image is spawned or execed. Loaded startup module summaries from
the checkpoint are copied into the committed image's `required_modules`, so the
image cannot boot unless those same module sources are loaded again.

For `images commit`, passing `--actor-pid <pid>` makes the CLI enforce that
process's checkpoint or checkpoint-owner process read capability plus exact
write capability on a new target image id.
`--replace` instead requires exact admin capability on the existing target
image. Without `--actor-pid`, the command runs as audited admin CLI.

## Checkpoint Commands

```bash
uv run agent-libos --db .agent_libos.sqlite checkpoint create <pid> "before risky edit"
uv run agent-libos --db .agent_libos.sqlite checkpoint list --pid <pid>
uv run agent-libos --db .agent_libos.sqlite checkpoint inspect <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint diff <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint restore <checkpoint_id>
uv run agent-libos --db .agent_libos.sqlite checkpoint fork <checkpoint_id> --parent-pid <pid>
uv run agent-libos --db .agent_libos.sqlite checkpoint replay <checkpoint_id> <event_id>
```

| Checkpoint subcommand | Arguments and option defaults |
| --- | --- |
| `create` | required `pid` and `reason`; `--metadata-json` defaults to `{}` |
| `list` | optional `--pid`; `--limit` defaults to and cannot exceed `checkpoint.list_limit` (100 by default) |
| `inspect`, `diff`, `restore` | required `checkpoint_id` |
| `fork` | required `checkpoint_id`; optional `--parent-pid` |
| `replay` | required `checkpoint_id` and `event_id` |

`--actor-pid <pid>` makes the CLI enforce that process's operation-specific
checkpoint capabilities. Restore requires checkpoint `admin`, exact image
`admin` for every existing image changed by the snapshot, and exact image
`write` for every missing image it reintroduces. Without `--actor-pid`, the
command runs as the trusted `cli` administrator and bypasses process capability
checks. Creation and restore publish their normal mutation event/audit
evidence; fork publishes its core mutation event and attempts its documented
post-commit audit; replay publishes a diagnostic audit. Read-only `list`,
`inspect`, and `diff` do not promise a separate admin-operation audit.
Deployments that require one for every Host read must enforce it at the CLI or
host-access boundary.
Restore prints `status: restored` after complete reconciliation, or
`status: restored_with_warnings` with `main_state_committed: true` and
`post_commit_failures` when image/JIT/finalizer reconciliation fails after the
scoped state transaction. Restore's finite authority, main state, core event,
and core audit share one transaction; a failure in any of those stages leaves
the restore uncommitted and returns an error. Do not retry a warning result as
an uncommitted restore.
Fork similarly returns `status: forked` after complete publication, or
`status: forked_with_warnings` with `main_state_committed: true` when its
post-commit event/audit sink fails. Do not retry that warning result as an
uncommitted fork. An actor-scoped fork requires `execute` on the exact
checkpoint and `write` for each missing captured image, plus matching startup
Modules. It does not overwrite an already registered same-ID image; the fork
can therefore combine captured state/tool bindings with that ID's current image
contract. Treat this as reviewed contract drift, not replay equivalence; see
[Checkpoint and Restore Semantics](checkpoints.md#fork-from-checkpoint).

## Skill Commands

```bash
uv run agent-libos --db .agent_libos.sqlite skills discover
uv run agent-libos --db .agent_libos.sqlite skills validate skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills register skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills inspect swe-agent
uv run agent-libos --db .agent_libos.sqlite skills discover --text swe-agent
uv run agent-libos --db .agent_libos.sqlite skills activate <pid> swe-agent --expected-package-sha256 <package_sha256-from-discover>
uv run agent-libos --db .agent_libos.sqlite skills unload <pid> swe-agent
```

| Skill subcommand | Arguments and option defaults |
| --- | --- |
| `discover` | optional `--text`; `--limit` defaults to and cannot exceed `skills.discover_limit` (100 by default) |
| `inspect` | required registered `skill_id` |
| `validate` | required Host `path`; never accepts `--actor-pid` |
| `register` | required `path`; `--replace` is off; `--source-type` is `workspace`, `global`, or `runtime`, and omission infers `global` only below a configured global root, otherwise `workspace` |
| `activate` | required `pid`, `skill_id`, and exact lowercase `--expected-package-sha256` |
| `unload` | required `pid` and `skill_id` |
| `trust`, `untrust` | required global Skill `path`; trust is bound to its current package SHA-256 |

`skills activate` compare-and-swaps against the exact lowercase
`package_sha256` returned by `skills discover`; rediscover instead of retrying
if package content changes.

Global Skills require exact package SHA-256 trust:

```bash
uv run agent-libos --db .agent_libos.sqlite skills trust ~/.agent-libos/skills/review-helper
uv run agent-libos --db .agent_libos.sqlite skills register ~/.agent-libos/skills/review-helper --source-type global
uv run agent-libos --db .agent_libos.sqlite skills untrust ~/.agent-libos/skills/review-helper
```

`--actor-pid <pid>` makes `discover`, `inspect`, `register`, `activate`,
`unload`, `trust`, and `untrust` enforce that process's applicable registry,
package-source, Skill, target-process, and trust capabilities. Static
`skills validate` deliberately reads a Host filesystem path and rejects
`--actor-pid`; use process-mode `skills register` when package bytes must be
read through the actor's workspace filesystem authority.

## Capability Commands

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities list --subject <pid>
uv run agent-libos --db .agent_libos.sqlite capabilities inspect <capability_id>
uv run agent-libos --db .agent_libos.sqlite capabilities explain <pid> filesystem:workspace:README.md read
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite capabilities delegate <parent_pid> <child_pid> 'filesystem:workspace:src/*' --rights read
uv run agent-libos --db .agent_libos.sqlite capabilities revoke <capability_id> --reason "no longer needed"
```

| Capability subcommand | Arguments and option defaults |
| --- | --- |
| `list` | `--subject` defaults to the actor in process mode and all subjects in admin mode; `--include-inactive` is off; bounded `--limit` is described below |
| `inspect` | required `capability_id` |
| `grant` | required `subject`, `resource`, and one or more `--rights`; spec defaults are `--effect allow`, non-delegable, revocable, no expiry/use ceiling, and `{}` constraints/metadata |
| `delegate` | required `parent`, `child`, and the same complete spec flags as `grant`; the requested spec must attenuate a covering delegable parent |
| `revoke` | required `capability_id`; optional `--reason` |
| `explain` | required `subject`, `resource`, and typed `right`; `--context-json` defaults to `{}` and does not itself grant authority |

For `grant` and `delegate`, `--delegable` opts in, while
`--revocable`/`--no-revocable` defaults to revocable. `--uses-remaining` and
`--expires-at` establish finite-use and time leases; `--constraints-json` and
`--metadata-json` must each decode to a JSON object. An omitted lease field is
unbounded, not zero. Effects are `allow`, `ask`, or `deny`; because `allow` is
the default, scripts that intend a restrictive record should always pass the
effect explicitly.

`capabilities list --limit` defaults to `capability.list_limit` from the active
Host configuration (100 by default). The same configured value is the maximum
accepted page size, so an explicit `--limit` must be between 1 and that value.

Capability records are structured authority statements: typed resource
pattern, rights, `allow`/`deny`/`ask` effect, issuer lineage, delegation depth,
status, expiry, use count, constraints, and metadata. One-shot approval is
represented as `effect=allow` with `uses_remaining=1`.

Without `--actor-pid`, capability mutations run as the Host admin actor and
publish their normal mutation audit; read-only `list`, `inspect`, and `explain`
do not each add a separate admin-operation audit row. Host mode changes who
authorizes `grant` and `revoke`, but does not bypass delegation attenuation.
`delegate` always requires a covering, delegable parent capability. With
`--actor-pid`, `grant` additionally requires grant/admin authority. `revoke`
allows the issuer, a covering revoke/admin holder, or the subject self-revoking
an `allow`; a subject cannot remove its own restrictive `ask` or `deny` record.

## JSON-RPC Commands

```bash
uv run agent-libos --db .agent_libos.sqlite jsonrpc register <path-to-endpoint-manifest.yaml>
uv run agent-libos --db .agent_libos.sqlite jsonrpc list
uv run agent-libos --db .agent_libos.sqlite jsonrpc inspect demo-weather
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> jsonrpc:demo-weather:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite jsonrpc call <pid> demo-weather forecast --params-json '{"city":"Beijing"}'
uv run agent-libos --db .agent_libos.sqlite jsonrpc unregister demo-weather
```

The manifest path is user supplied; copy and adapt the complete example in
[jsonrpc.md](jsonrpc.md).

Registry commands accept `--actor-pid <pid>` with the following authority
contract. Without it, mutations run as Host admin registry operations and emit
their normal mutation evidence; read-only list/inspect calls do not promise a
separate admin-operation audit.

| JSON-RPC subcommand | Process-mode authority |
| --- | --- |
| `register` | filesystem `read` for the manifest plus exact endpoint `write`; `--replace` requires endpoint `admin` |
| `list` | registry `read` |
| `inspect` | exact endpoint `read`; sensitive manifest fields remain hidden |
| `unregister` | exact endpoint `admin` |
| `call <pid> ...` | the target `<pid>` supplies the declared method right; an explicitly supplied `--actor-pid` must equal `<pid>` and does not add authority |

`jsonrpc call` always runs as the target process pid and requires that pid to
hold the method capability, such as
`jsonrpc:demo-weather:forecast read`. The CLI cannot supply arbitrary URLs,
headers, raw JSON-RPC method names, or request ids.
`--params-json`, when present, must be strict JSON; malformed input, duplicate
object keys, non-finite numbers, and excessive nesting are rejected before
provider dispatch. The value must be an object, array, or `null`; scalar JSON
values are rejected before authority or registry lookup.

## MCP Commands

```bash
uv sync --frozen --extra mcp
uv run agent-libos --db .agent_libos.sqlite mcp register <path-to-server-manifest.yaml>
uv run agent-libos --db .agent_libos.sqlite mcp list
uv run agent-libos --db .agent_libos.sqlite mcp inspect demo-mcp
# Manifest v2 with protocol_mode auto or 2026-07-28 only:
uv run agent-libos --db .agent_libos.sqlite mcp discover demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp tools demo-mcp
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> process:spawn --rights write
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> <stdio_authority_resource-from-inspect> --rights execute
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> mcp:demo-mcp:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite mcp call <pid> demo-mcp forecast --arguments-json '{"city":"Beijing"}'
uv run agent-libos --db .agent_libos.sqlite mcp unregister demo-mcp
```

Manifest v3 servers pinned to exact protocol revision `2026-07-28` expose the
modern Host client commands:

```bash
# Resources and Templates are addressed only by registered logical ids.
uv run agent-libos --db .agent_libos.sqlite mcp resources list demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp resources templates demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp resources read demo-mcp handbook --variables-json '{"locale":"zh-CN"}'

# Prompt output is a preview requiring user confirmation; it is never trusted
# as system/developer context.
uv run agent-libos --db .agent_libos.sqlite mcp prompts list demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp prompts get demo-mcp release-notes --arguments-json '{"version":"1.5.1"}'
uv run agent-libos --db .agent_libos.sqlite mcp prompts complete demo-mcp prompt release-notes version 2

# OAuth login is one foreground flow. It prints the authorization URL, waits
# for the full callback URL, and completes before this Runtime exits.
uv run agent-libos --db .agent_libos.sqlite mcp auth login work-oauth --profile-file oauth-profile.json --scope resources.read
# Every later one-shot command explicitly rebinds the same strict Host profile.
uv run agent-libos --db .agent_libos.sqlite mcp --oauth-profile-file oauth-profile.json auth status work-oauth
uv run agent-libos --db .agent_libos.sqlite mcp --oauth-profile-file oauth-profile.json resources list demo-mcp
# Confidential clients supply their secret through an inherited fd, never argv.
uv run agent-libos --db .agent_libos.sqlite mcp auth login work-oauth --profile-file oauth-profile.json --client-secret-fd 3 3<client-secret
uv run agent-libos --db .agent_libos.sqlite mcp auth logout work-oauth --profile-file oauth-profile.json

# Durable Elicitation and Tasks use local ids plus revision CAS only.
uv run agent-libos --db .agent_libos.sqlite mcp continuations inspect continuation-id
uv run agent-libos --db .agent_libos.sqlite mcp continuations respond continuation-id --expected-revision 1 --human-request-id human-request-id --human-expected-revision 3 --human-preview-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --responses-json '{"input-1":{"action":"accept","content":{"approved":true}}}'
uv run agent-libos --db .agent_libos.sqlite mcp continuations cancel continuation-id --expected-revision 1
uv run agent-libos --db .agent_libos.sqlite mcp remote-tasks get local-task-ref
uv run agent-libos --db .agent_libos.sqlite mcp remote-tasks update local-task-ref --expected-revision 1 --human-request-id human-request-id --human-expected-revision 4 --human-preview-sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb --responses-json '{"input-1":{"action":"accept","content":{"approved":true}}}'
uv run agent-libos --db .agent_libos.sqlite mcp remote-tasks cancel local-task-ref --expected-revision 2

# Notifications are inert Host ingress. The CLI owns one listener in this
# foreground Runtime, streams events to stderr, and prints a final JSON summary
# to stdout after its bound, terminal state, or Ctrl-C performs explicit stop.
uv run agent-libos --db .agent_libos.sqlite mcp subscriptions listen demo-mcp --filter resourcesListChanged --max-seconds 300
```

Manifest and registry developer-experience commands are also Host-only. Live
probe, scaffold transitions, and import apply require a named reviewer, a
reason, and the matching explicit confirmation flag:

```bash
uv run agent-libos --db .agent_libos.sqlite mcp validate server.yaml
uv run agent-libos --db .agent_libos.sqlite mcp doctor server.yaml
uv run agent-libos --db .agent_libos.sqlite mcp probe server.yaml --confirm-probe --reviewer alice --reason 'review the unregistered full catalog'
uv run agent-libos --db .agent_libos.sqlite mcp scaffold create base.yaml complete-probe.json --confirm-scaffold --reviewer alice --reason 'prepare allowlist candidate'
uv run agent-libos --db .agent_libos.sqlite mcp scaffold approve reviewed-candidate.json --confirm-review --reviewer alice --reason 'approve edited authority contract'
uv run agent-libos --db .agent_libos.sqlite mcp export --server demo-mcp > registry-export.json
uv run agent-libos --db .agent_libos.sqlite mcp import plan registry-export.json
uv run agent-libos --db .agent_libos.sqlite mcp import apply registry-export.json demo-mcp --confirm-import --reviewer alice --reason 'apply reviewed CAS plan'
```

`validate`, `doctor`, and both `scaffold` operations always open an ephemeral
in-memory Runtime and ignore `--db`; they neither require a current persistent
store schema nor read, migrate, or mutate the selected registry. `probe`,
`export`, and both `import` operations intentionally use the selected store.
`probe` is the only DX command above that crosses a provider boundary. It
validates an unregistered Manifest v3 candidate and collects complete bounded
Tool, Resource, Resource Template, and Prompt catalogs through the normal
governed Runtime primitive without registering the candidate. Its report pins
the exact source-manifest digest; `scaffold create` rejects a report produced
for any other base manifest and carries all four catalogs into the conservative
review bundle instead of silently discarding modern surfaces.
`import apply` records the fixed Host boundary actor `mcp-cli-import`; it does
not accept an actor override. The required reviewer and reason fields carry
the human review attribution without allowing the registry audit actor to be
relabeled from the command line.

The manifest path is user supplied; copy and adapt [mcp.md](mcp.md). The `mcp`
extra is not installed by the core dependency command. Host-mode registration
reads at most `mcp.manifest_max_bytes + 1` bytes and requires valid UTF-8 before
parsing; process-mode registration applies the same configured limit through
the process filesystem primitive.

`mcp list` returns `{servers, has_more}` rather than a bare array. A
`has_more: true` value means the bounded window is incomplete; narrow `--text`
or request a larger `--limit` within the configured registry ceiling instead of
treating the returned servers as a complete inventory. That ceiling is
`mcp.server_page_limit`; the deprecated `mcp.list_limit` applies only to
Manifest v1/v2 live Tool catalogs and cannot truncate the server page.

Registry commands accept `--actor-pid <pid>` with the following authority
contract. Without it, mutations run as Host admin registry operations and emit
their normal mutation evidence; read-only list/inspect calls do not promise a
separate admin-operation audit.

| MCP subcommand | Process-mode authority |
| --- | --- |
| `register` | filesystem `read` for the manifest plus exact server `write`; `--replace` requires server `admin`; stdio also requires `process:spawn write` and exact command `execute` |
| `list` | registry `read` |
| `inspect` | exact server `read`; sensitive manifest fields remain hidden |
| `discover` | exact server `read+execute`; stdio additionally requires the local-spawn rights; Manifest v2 `auto`/`2026-07-28` only |
| `tools` | exact server `read`; `--refresh` also requires server `execute` and the stdio local-spawn rights when applicable |
| `unregister` | exact server `admin` |
| `call <pid> ...` | the target `<pid>` supplies the declared tool right and any stdio local-spawn rights; an explicitly supplied `--actor-pid` must equal `<pid>` and does not add authority |

All modern groups (`resources`, `prompts`, `auth`, `continuations`,
`remote-tasks`, and `subscriptions`) and all DX commands (`validate`, `doctor`,
`probe`, `scaffold`, `export`, and `import`) reject `--actor-pid`. They are
explicit Host surfaces and are not syscalls or model tools. The only additional
model-visible surface is the separately governed Resource list/read pair when a
Manifest v3 allowlist explicitly enables it.

Resource, Prompt, and completion request objects are strict, byte-bounded JSON.
Omission selects the documented default object, while an explicit JSON `null`
is rejected. Resource variables and Prompt arguments require string values.
Pagination cursors are opaque and `has_more` is derived from `next_cursor`; do
not manufacture, decode, or reuse a cursor with another server or operation.

Resource read and Prompt get may return `complete`, `input_required`, or
`remote_task`; Prompt completion is complete-only. Continuation inspect is
input-required-only, response may advance to any of the three result kinds,
and cancellation is complete-only. Remote-Task get/update/cancel return only
the local `remote_task` projection. Prompt `get` and Prompt completion both
emit CLI-level `preview_only: true`, `user_confirmation_required: true`, and
an untrusted user-context label. The Completion value itself retains only its
Completion DTO fields; the CLI never inserts Prompt messages or completion
output into a process, system prompt, or developer prompt.

OAuth supports only Host-configured pre-registration and CIMD; DCR is not
supported. A bounded, strict, extra-forbid profile JSON supplies the non-secret
issuer/resource/client/redirect authority. `auth login` performs begin, manual
browser authorization, callback input, and completion inside one foreground
Runtime; there is deliberately no standalone `auth complete` command. A
non-interactive caller must opt in with `--callback-stdin`, and callback values
never appear in success or error JSON. The callback authorization code exists
only in the foreground process's transient input/request memory for the single
exchange attempt; the CLI releases its application-owned callback reference
afterward and never writes the code to Store, broker storage, evidence, output,
errors, or logs. A confidential client secret is accepted only through
`--oauth-client-secret-fd` (or the login-local
`--client-secret-fd`) and is never accepted as an argv or environment value.

Successful client/token bundles remain only in deterministic secure-broker
slots. Each later one-shot command must pass the same
`mcp --oauth-profile-file ...` before its subcommand, which strictly rebinds
that Host authority and can rehydrate the broker slot without putting a token,
secret, or slot reference in RuntimeStore. `auth logout` likewise requires the
profile file and explicitly purges the token. Runtime shutdown deletes every
unfinished PKCE/state challenge, so no callback can be resumed by another CLI
process. The default broker accepts only the exact reviewed `keyring==25.7.0`
macOS Keychain, Windows Credential Manager, Linux Secret Service/libsecret, and
KWallet 4/5 implementations; chained, plugin, lookalike, and unreviewed-version
backends fail closed. With no accepted and operational secure credential
backend, these operations fail closed.

Continuation response and remote-Task update require the operation's
`--expected-revision` plus the `--human-request-id`,
`--human-expected-revision`, and `--human-preview-sha256` returned by the
sanitized inspect/get view. Runtime first CAS-settles that exact HumanRequest
preview, including its local identity, then consumes only its approved local
responses. Each response is keyed by the local `input-N` id from that same
view and contains an explicit `action`; a form acceptance carries its validated
object under `content`. Cancellation requires only `--expected-revision`.
The CLI does not accept binding hashes, effect ids, registry generations, auth
principals, remote task ids, or deadlines. Runtime reconstructs and checks
those bindings internally, and never replays the original Tool call. A remote
Task cancellation acknowledgement is only a requested state, not proof that
the remote work stopped. There is intentionally no `remote-tasks list` command.

Subscription events are bounded untrusted ingress and cache-invalidation
signals only. Reading them cannot launch a model, Tool, TaskRun, or another
subscription. The CLI exposes only foreground `subscriptions listen`: start,
event reads, status checks, and stop all happen in the same Runtime, and Ctrl-C
still performs an explicit stop. It does not promise that a subscription id can
be resumed by a later process. A lost listener requires a new foreground listen
plus full refresh; the CLI neither sends `Last-Event-ID` nor performs hidden
reconnect or replay. Long-lived Host Python and GUI surfaces may expose separate
start/status/events/stop controls because they retain the owning Runtime.
Event reads are single-consumer and acknowledge the returned batch: the next
read must send the last returned local sequence as `after`. Empty reads retain
the current cursor. Stale/future cursors and competing readers fail closed;
the foreground CLI and GUI advance their cursor only after a validated batch.

`--arguments-json`, when present, must be a strict JSON object; malformed JSON,
duplicate object keys, non-finite numbers, arrays, and scalars are rejected
before provider dispatch. The encoded argument is rejected before JSON decode
when it exceeds `mcp.max_request_hard_limit_bytes`. Omitting it supplies `{}`.

Sink trust is intentionally not an MCP, model-tool, or process CLI registry.
Administrators configure it under `data_flow.sink_rules` or use the Host-only
`Runtime.register_sink_trust()` / `unregister_sink_trust()` interfaces. See
[data_flow.md](data_flow.md); registering trust does not grant the process
capabilities shown above.

`mcp call` always runs as the target process pid and requires that pid to hold
the tool capability, such as `mcp:demo-mcp:forecast read`. For a stdio server,
the pid additionally needs `process:spawn write` and `execute` on the exact
`stdio_authority_resource` returned by `mcp inspect`; do not reconstruct or
wildcard that digest. Streamable HTTP servers do not need those two local-spawn
grants. The CLI cannot supply arbitrary transports, commands, URLs, headers, or
raw MCP tool names.

`mcp call` always prints the structured call result. A returned MCP failure
(`ok: false`) exits with status 1 after printing that JSON exactly once; the CLI
does not retry the provider call.

`mcp list` and `mcp inspect` show the Manifest version and configured protocol
mode. `mcp discover`, live `mcp tools --refresh`, and `mcp call` project the
bounded negotiated protocol revision/era and server capability diagnostics for
that operation. Agent libOS does not cache those connection diagnostics across
CLI invocations.

## Runtime Module Commands

Runtime Modules are trusted Python startup extensions. They are loaded with
global arguments before the selected command runs. In this checkout the default
project config already loads and trusts the PTY module, so inspect that
configured instance without repeating its manifest:

```bash
uv run agent-libos --db .agent_libos.sqlite modules verify modules/pty/module.yaml
uv run agent-libos --db .agent_libos.sqlite modules list
uv run agent-libos --db .agent_libos.sqlite modules inspect agent-libos-pty:v0
```

| Module subcommand | Arguments and option defaults |
| --- | --- |
| `list` | `--limit` defaults to and cannot exceed `modules.discover_limit` (100 by default) |
| `inspect` | required loaded `module_id` |
| `verify` | required manifest `path`; static verification uses the selected config's bounds/trust plus any repeatable global trust options |

`modules verify` resolves the entrypoint and computes the manifest hash and
source hash without opening a Runtime, touching the selected database, loading
the core module, or loading any module named by `modules.manifest_paths` or
`--module-manifest`. The selected config still supplies verification limits and
trust entries; `--db` is accepted as a global option but ignored by this static
command. For single-file modules the source hash is the entry file hash; for
multi-file modules it is the inferred Python source package digest and includes
the covered `source_files` list. The returned
`trust_key` is the copy-paste value for `--trusted-module`. `modules list` and
`modules inspect` show persisted module load records for the opened runtime
database. `--trusted-module-sha256 <manifest_sha256>:<source_sha256>` is
accepted as a weaker local-development shortcut that trusts the digest pair for
any module id. When a selected config does not already name a custom module,
load it with `--module-manifest <path>` and `--trusted-module <trust_key>` before
the command name. Configured and explicit manifest lists are combined, and a
duplicate path is rejected rather than silently loaded twice.

## Benchmark Scripts

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/smoke
uv run python experiments/collect_metrics.py .benchmark_runs/smoke
```

Use repeated `--task` or `--attack-class` to select a subset. `--runner all`
also includes the observer-only `no_audit_linkage` ablation, which deliberately
withholds evidence and can produce invalid rows and a non-zero exit; use it for
evidence-loss diagnosis, not as a green comparison gate. For rate-bearing
comparisons, select the seven-runner command under [Benchmark: Running](benchmark.md#running).
Default comparison mode writes valid success/safety oracle failures without
making them command failures; `--require-all-passed` returns non-zero unless
every selected run passes both oracles and is the appropriate oracle smoke
gate. The complete deterministic release command also passes
`--require-release-evidence`, which requires audit completeness of 1.0 and zero
false denials for every selected runner; neither flag alone represents the full
release contract.

Real LLM mode is explicit and scoped:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --llm real --limit 1 --output .benchmark_runs/real-smoke
```

After all task filters and `--limit`, real mode must select exactly one task. It
supports only Agent-libOS-family runners; `--runner all` and wrapper/sandbox
baselines are rejected before any model call.

## Example Scripts

```bash
uv run python scripts/llm_summarize_document.py README.md --auto-approve
uv run python scripts/llm_write_goal_smoke.py
uv run python scripts/run_coding_agent.py --workspace /path/to/repo --goal "Implement the requested change"
uv run python scripts/object_memory_file_copy_smoke.py
uv run python scripts/async_clock_interleave_smoke.py --iterations 3 --interval 0.2
uv run python scripts/ask_file_then_show.py --auto-answer README.md
uv run python scripts/human_llm_chat.py --mock --auto-message hello --auto-message /exit
```

`run_coding_agent.py` loads `.env` from the Agent libOS checkout before it
mounts the target workspace. The target workspace's `.env` is not read
implicitly; use `--env-file` for an explicit alternate credential file.
Without `--no-run`, the launcher immediately invokes the configured real LLM
and allows up to `runtime.launcher_max_quanta` quanta (40 by default). Its
default `edit` permission preset grants workspace-wide read and write authority,
and its default Shell policy is `allowlist_auto_else_ask`. It stores state in a
workspace-keyed persistent database outside the exposed workspace unless
`--ephemeral-db` is selected. Use `--permission-preset read-only` when writes
are unnecessary, and use `--strict` when a failed or killed process must make
the script return non-zero; without `--strict`, inspect `process_status` in the
printed JSON.

On Windows PowerShell, use backslashes when convenient:

```powershell
uv run python scripts\run_coding_agent.py --workspace ..\some-repo --goal "Summarize the current project"
```
