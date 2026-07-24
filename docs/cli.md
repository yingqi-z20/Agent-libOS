# CLI Reference

The package installs the `agent-libos` command.

The package also installs `agent-libos-gui-server`, which is used by the
Electron desktop console described in [docs/gui.md](gui.md).
Both entrypoints are implemented under `agent_libos.api`, because they are
host-facing control surfaces over the same runtime boundary.

Use `--db` to select a runtime store. The sentinel target `local` is in-memory
SQLite. Any other filesystem path creates or opens a persistent SQLite
database. A `postgresql://` or `postgres://` DSN opens a PostgreSQL runtime
store. PostgreSQL support is optional; install it before using a DSN, for
example with `uv sync --frozen --all-groups --extra postgres`.

```bash
uv run agent-libos --db .agent_libos.sqlite <command>
```

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
uv run agent-libos --config ./agent-config.yaml spawn --goal "Inspect README.md"
```

The file is a validated overlay on `agent_libos.config.DEFAULT_CONFIG`. Mapping
fields are merged recursively, so adding `llm.profiles.coding` keeps the default
profile. Scalar fields and list/tuple fields replace the default value. Unknown
fields and unsafe values fail at startup. The Pydantic dataclass loader accepts
compatible coercions for ordinary scalar fields, so this is not a strict
JSON-type boundary; security-sensitive numeric bounds still receive explicit
post-construction validation.

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
store a runtime-memory marker, and rows whose live payload cache cannot be
reconstructed are released fail-closed on reopen.
Persistent stores also take an active-runtime lease. SQLite derives both the
connection target and lease from the canonical database path, so a symlink
alias cannot open a second writer. Where `fcntl` plus `O_NOFOLLOW` are
available, the sidecar is opened no-follow, regular-file/inode checked, and
protected by `flock`, while the database and SQLite sidecars are tightened to
owner-only (`0600`); otherwise SQLite's kernel-managed exclusive database
lock is used instead of a stale-file protocol. PostgreSQL uses a session
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

Use `--module-manifest` and `--trusted-module` before the command name to load
trusted Runtime Modules before the runtime is used:

```bash
uv run agent-libos --db .agent_libos.sqlite \
  --module-manifest modules/pty/module.yaml \
  --trusted-module agent-libos-pty:v0:<manifest_sha256>:<source_sha256> \
  <command>
```

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
```

Run `uv run agent-libos <command> --help` for the complete, argparse-generated
parameter list. The narrative sections below document workflows and security
contracts; lists labelled “useful options” are intentionally not exhaustive.

`demo` writes the deterministic preview to
`agent_outputs/demo_patch_preview.txt` below the Runtime workspace and uses
overwrite mode. Preserve or move an existing file at that path before running
the demo if it contains user data.

`--actor-pid` is a command-group option for `checkpoint`, `skills`,
`capabilities`, `images`, `jsonrpc`, and `mcp`. Place it after the group name
and before the subcommand; placing it after the subcommand is rejected by
argparse. For example:

```bash
uv run agent-libos --db .agent_libos.sqlite capabilities --actor-pid <actor_pid> grant <subject_pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite checkpoint --actor-pid <actor_pid> inspect <checkpoint_id>
```

## Persistent Runtime Basics

`run` invokes the configured real LLM and may consume provider tokens. For the
README goal below, first configure the model/API environment and grant the exact
file capability; image requirement declarations are not grants. `spawn` prints
the pid to use in the following command. The generic CLI reads inherited
environment variables and does not implicitly load `.env`; export them first or
invoke it as `uv run --env-file .env agent-libos ...`. See
[Real LLM Configuration](../README.md#real-llm-configuration).

```bash
uv run agent-libos --db .agent_libos.sqlite init
uv run agent-libos --db .agent_libos.sqlite spawn --image coding-agent:v0 --goal "Summarize README.md"
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> filesystem:workspace:README.md --rights read
uv run agent-libos --db .agent_libos.sqlite run --max-quanta 10
uv run agent-libos --db .agent_libos.sqlite processes
uv run agent-libos --db .agent_libos.sqlite resources <pid>
uv run agent-libos --db .agent_libos.sqlite audit
uv run agent-libos --db .agent_libos.sqlite tools
uv run agent-libos --db .agent_libos.sqlite workflow run get_working_directory
```

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

## Workflow Run

`workflow run` is a direct user entrypoint for tools. It spawns a fresh
AgentProcess from the selected image, calls one visible tool through the normal
ToolBroker path, and returns the tool result JSON. It does not run the LLM
scheduler and it does not bypass the image's process tool table, primitive
capability checks, resource budgets, human approval, result-object persistence,
events, or audit.

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

## Object Tasks

`object-task` commands expose Object-bound background tool tasks. A task belongs
to an existing Object Memory object, runs one visible tool through a dedicated
runner child process, and reports status as JSON.

```bash
uv run agent-libos --db .agent_libos.sqlite \
  object-task start --pid <pid> --owner-oid <oid> get_working_directory --wait

uv run agent-libos --db .agent_libos.sqlite \
  object-task start --pid <pid> --owner-oid <oid> \
  --watch-owner --watch-events updated,linked receive_process_messages \
  --args-json '{"channel":"object-task-owner"}' --wait

uv run agent-libos --db .agent_libos.sqlite object-task list --pid <pid>
uv run agent-libos --db .agent_libos.sqlite object-task wait <task_id> --pid <pid>
uv run agent-libos --db .agent_libos.sqlite object-task cancel <task_id> --pid <pid>
uv run agent-libos --db .agent_libos.sqlite \
  object-task watch-owner <task_id> --pid <pid> --watch-events updated \
  --watch-channel object-task-owner
```

The task still uses the creator process tool table, ToolBroker, capabilities,
resource budgets, events, audit, and Object Memory result semantics. A
task started with `--watch-owner` receives owner `updated` and `linked` notices
as process messages in its runner process; those notices contain object ids and
event metadata, not object payloads or new capabilities. A
`watch-owner` subcommand can update or disable that watch while the task is
still active. Owner-watch auto-resume is limited to tools with safe
message-receive replay semantics, currently `receive_process_messages`.
Running synchronous side-effect tools are not force-cancelled because Python
cannot safely stop their worker thread after side effects may have started. A
one-shot CLI invocation cannot keep detached in-memory tasks alive after the
CLI Runtime shuts down, so `object-task start` requires `--wait`. For
file-backed SQLite, the active-runtime lease prevents a separate CLI process
from opening the same database while a live GUI or embedded host owns it. After
the live owner has stopped, reopening the store reconciles unfinished
ObjectTasks as abandoned. Use GUI server APIs or the embedding host for live
ObjectTask supervision. The one-shot CLI `list|get|wait|cancel|watch-owner`
commands are intended for the Runtime opened by that CLI invocation or for
terminal task records after the live owner has stopped.

## LLM Calls

Inspect persisted LLM action-selection calls:

```bash
uv run agent-libos --db .agent_libos.sqlite llm-calls --pid <pid>
uv run agent-libos --db .agent_libos.sqlite llm-calls --limit 20
```

Records include provider ids, model/API mode, token usage when available, and
full prompts, visible tools, output, tool calls, reasoning, raw responses, and
bounded observability envelopes. The envelopes contain preview, byte count,
hash, and truncation metadata.
For OpenAI Responses requests, request options may show strict tool-schema
counts, whether prompt-cache or safety identifiers were configured, and any
non-secret `previous_response_id` chain; configured cache keys and safety
identifier values are not persisted there. When no provider-side chain is used,
historical tool outputs are plain bounded context instead of Responses-native
`function_call_output` items. A chain is continued only when official Responses
storage/chaining and full local I/O persistence are enabled, the profile/scope
fingerprint and credential-keyed provider identity fingerprint are unchanged,
and the immediately preceding function-call manifest has exactly one durable
output per unique `call_id`. The fingerprint binds model, official endpoint,
API mode, credential identity, and organization/project without storing the
credential. Otherwise the next request resets stateless. Request options also
show whether `parallel_tool_calls` and JSON action fallback were enabled, and
whether the fallback was used. Canonical usage preserves provider-reported
`cache_read_tokens` and `cache_write_tokens`, including an explicit zero, even
when full-I/O persistence is disabled.
Full LLM input/output persistence is enabled by default for self-evolution
training and fine-tuning pipelines under the deployment's user agreement. Set
`config.llm.persist_full_io=False` when a user or operator opts out of storing
sensitive prompt, tool, reasoning, provider-error, and provider payload fields.
The runtime then writes canonical content-free summary envelopes with byte
counts, JSON shape/count metadata when available, and hashes; it does not keep
readable previews of those fields.

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
transaction. Live/unknown effects, Responses-chain anchors, and process-result
recovery calls are not eligible. See
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
structured `NotFound` error and exits with status 1. Explain output is metadata/redaction oriented even when
full LLM I/O persistence is enabled. See
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

Calls, tokens, syscalls, bytes, child counts, and peak-memory values are
non-negative integers. Runtime and subprocess wall/CPU seconds are continuous
finite non-negative values and may be fractional. Boolean values are rejected
for both shapes instead of being accepted as Python integers.

## Interactive Run

For a Codex CLI-style loop:

```bash
uv run agent-libos --db .agent_libos.sqlite run --interactive --pid <pid> --max-quanta 20
```

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
  `always_deny`.
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

- `--kind normal|interrupt`
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
uv run agent-libos --db .agent_libos.sqlite cd <pid> src
uv run agent-libos --db .agent_libos.sqlite exec review-agent:v0 "Review README.md" --pid <pid> --run
uv run agent-libos --db .agent_libos.sqlite exit <pid> --payload '{"done":true}'
```

For `exec`, the first positional argument is the target image. It can be an
already registered image id such as `coding-agent:v0`, or an image package
directory containing `IMAGE.yaml`. The second positional argument is the
replacement goal.

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
images/review-agent/
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
image_id: review-agent:v0
name: review-agent
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
Use `minimal_runtime` for factual runtime state sections, or `libos_default`
only when the image intentionally wants the native Agent libOS planner prompt.

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
64-character lowercase `source_sha256` reported by
`uv run agent-libos modules verify <module.yaml>`. Spawn and exec check that
the current runtime has already loaded the exact trusted module source; image
boot never loads modules automatically.

`default_tools` is exact. The runtime does not add `process_exit`,
`create_memory_object`, or any other builtin automatically. List every
LLM-facing builtin the image should be able to call; package JIT tools can still
use authorized libOS syscalls internally without being mirrored as builtin
tools in the process tool table.

## Image Commands

```bash
uv run agent-libos --db .agent_libos.sqlite images list
uv run agent-libos --db .agent_libos.sqlite images inspect coding-agent:v0
uv run agent-libos --db .agent_libos.sqlite images validate images/mini-swe-agent
uv run agent-libos --db .agent_libos.sqlite images register images/mini-swe-agent
uv run agent-libos --db .agent_libos.sqlite images commit <checkpoint_id> stateful-agent:v0 --name stateful-agent
```

`images commit` creates a checkpoint-derived image artifact from the checkpoint
owner root process. It captures owned/captured internal Object Memory (not an
uncaptured borrowed root), loaded Skills,
process-local JIT tools, tool visibility, and cwd. It does not package
filesystem/provider state. External capabilities from the checkpoint are stored
as `required_capabilities` declarations and are not granted automatically when
the committed image is spawned or execed. Loaded startup module summaries from
the checkpoint are copied into the committed image's `required_modules`, so the
image cannot boot unless those same module sources are loaded again.

Passing `--actor-pid <pid>` makes the CLI enforce that process's checkpoint
read capability plus exact write capability on a new target image id.
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

`--actor-pid <pid>` makes the CLI enforce that process's operation-specific
checkpoint capabilities. Restore requires checkpoint `admin`, exact image
`admin` for every existing image changed by the snapshot, and exact image
`write` for every missing image it reintroduces. Without `--actor-pid`, the
command runs as an audited admin actor named `cli`.
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
uncommitted fork.

## Skill Commands

```bash
uv run agent-libos --db .agent_libos.sqlite skills discover
uv run agent-libos --db .agent_libos.sqlite skills validate skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills inspect swe-agent
uv run agent-libos --db .agent_libos.sqlite skills register skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills activate <pid> swe-agent
uv run agent-libos --db .agent_libos.sqlite skills unload <pid> swe-agent
```

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
uv run agent-libos --db .agent_libos.sqlite capabilities delegate <parent_pid> <child_pid> filesystem:workspace:src/* --rights read
uv run agent-libos --db .agent_libos.sqlite capabilities revoke <capability_id> --reason "no longer needed"
```

Capability records are structured authority statements: typed resource
pattern, rights, `allow`/`deny`/`ask` effect, issuer lineage, delegation depth,
status, expiry, use count, constraints, and metadata. One-shot approval is
represented as `effect=allow` with `uses_remaining=1`.

Without `--actor-pid`, capability commands run as an audited admin actor; that
Host mode changes who authorizes `grant` and `revoke`, but does not bypass
delegation attenuation. `delegate` always requires a covering, delegable parent
capability. With `--actor-pid`, `grant` additionally requires grant/admin
authority and `revoke` requires holder, issuer, revoke, or admin authority.

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
contract. Without it, they run as audited admin registry operations.

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

## MCP Commands

```bash
uv sync --frozen --all-groups --extra mcp
uv run agent-libos --db .agent_libos.sqlite mcp register <path-to-server-manifest.yaml>
uv run agent-libos --db .agent_libos.sqlite mcp list
uv run agent-libos --db .agent_libos.sqlite mcp inspect demo-mcp
uv run agent-libos --db .agent_libos.sqlite mcp tools demo-mcp
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> process:spawn --rights write
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> <stdio_authority_resource-from-inspect> --rights execute
uv run agent-libos --db .agent_libos.sqlite capabilities grant <pid> mcp:demo-mcp:forecast --rights read
uv run agent-libos --db .agent_libos.sqlite mcp call <pid> demo-mcp forecast --arguments-json '{"city":"Beijing"}'
uv run agent-libos --db .agent_libos.sqlite mcp unregister demo-mcp
```

The manifest path is user supplied; copy and adapt [mcp.md](mcp.md). The `mcp`
extra is not installed by the core dependency command.

Registry commands accept `--actor-pid <pid>` with the following authority
contract. Without it, they run as audited admin registry operations.

| MCP subcommand | Process-mode authority |
| --- | --- |
| `register` | filesystem `read` for the manifest plus exact server `write`; `--replace` requires server `admin`; stdio also requires `process:spawn write` and exact command `execute` |
| `list` | registry `read` |
| `inspect` | exact server `read`; sensitive manifest fields remain hidden |
| `tools` | exact server `read`; `--refresh` also requires server `execute` and the stdio local-spawn rights when applicable |
| `unregister` | exact server `admin` |
| `call <pid> ...` | the target `<pid>` supplies the declared tool right and any stdio local-spawn rights; an explicitly supplied `--actor-pid` must equal `<pid>` and does not add authority |

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

## Runtime Module Commands

Runtime Modules are trusted Python startup extensions. They are loaded with
global arguments before the selected command runs:

```bash
uv run agent-libos --db .agent_libos.sqlite modules verify modules/pty/module.yaml
uv run agent-libos --db .agent_libos.sqlite --module-manifest modules/pty/module.yaml --trusted-module agent-libos-pty:v0:<manifest_sha256>:<source_sha256> modules list
uv run agent-libos --db .agent_libos.sqlite --module-manifest modules/pty/module.yaml --trusted-module agent-libos-pty:v0:<manifest_sha256>:<source_sha256> modules inspect agent-libos-pty:v0
```

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
any module id.

## Benchmark Scripts

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/m1-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/m1-smoke
```

Use repeated `--task` or `--attack-class` to select a subset. `--runner all`
also includes the observer-only `no_audit_linkage` ablation, which deliberately
withholds evidence and can produce invalid rows and a non-zero exit; use it for
evidence-loss diagnosis, not as a green comparison gate. For rate-bearing
comparisons, select the seven-runner command under [Benchmark: Running](benchmark.md#running).
Default comparison mode writes valid success/safety oracle failures without
making them command failures; `--require-all-passed` returns non-zero unless
every selected run passes both oracles and is the appropriate release/smoke
gate.

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

On Windows PowerShell, use backslashes when convenient:

```powershell
uv run python scripts\run_coding_agent.py --workspace ..\some-repo --goal "Summarize the current project"
```
