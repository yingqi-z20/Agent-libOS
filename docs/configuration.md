# Configuration Guide

Agent libOS keeps non-secret runtime defaults in the frozen, validated
`agent_libos.config.DEFAULT_CONFIG` object. The canonical field declarations
and numeric defaults live in `agent_libos/config/defaults.py`; this document
defines loading, precedence, security handling, and subsystem semantics so
operators do not have to infer those rules from example YAML. The generated
[exact configuration reference](configuration_reference.md) records every
field path, resolved type, and default value for the current checkout; this
handwritten page remains the authority for semantics, ranges, precedence, and
security rules.

## In this guide

- [Load overlays with the documented precedence](#loading-and-precedence)
- [Apply bounded YAML input rules](#bounded-yaml-input)
- [Inspect exact defaults](#inspecting-exact-defaults)
- [Configure semantic phases](#semantic-phase-24-configuration)
- [Review security-sensitive settings](#security-sensitive-settings)
- [Understand bounded windows](#bounded-windows)
- Return to the [documentation home](index.md).

## Loading and precedence

Product entrypoints use this order:

1. Start from the frozen `DEFAULT_CONFIG` baseline.
2. If `--config <path>` is present, recursively merge that YAML mapping.
3. Otherwise load the repository/project-root `config.yaml` when it exists.
   The loader does not search the caller's current working directory.
4. Replace scalar, list, and tuple fields supplied by the overlay; recursively
   merge mapping fields such as `llm.profiles`.
5. For CLI and GUI-server entrypoints, an explicit `--db` store target overrides
   the selected runtime store target.
6. When an overlay is present, construct and validate a new frozen
   `AgentLibOSConfig`; without an overlay, the shared frozen baseline is safe
   to reuse through the public configuration API. Unknown fields and unsafe,
   inverted, or non-finite bounds fail before the Runtime opens. Pydantic
   accepts normal compatible scalar coercions for ordinary dataclass fields: a
   numeric YAML string may become a number, and an ordinary `int` or `float`
   field may coerce YAML booleans to `1` or `0`. Callers that generate overlays
   must not treat this loader as a strict JSON-type validator and should never
   use booleans for numeric values. Fields declared with Pydantic's `StrictInt`
   or `StrictFloat` are the authoritative numeric exceptions: non-null values
   must be numbers rather than strings, and booleans are rejected. `StrictBool`
   fields accept only booleans and reject integers and strings. The complete
   current strict set comprises:

   - `runtime.publication_recovery_max_attempts`,
     `runtime.publication_artifact_lookup_hard_limit`, both payload-retention
     ages, and every `page_size`/`page_hard_limit` field for publication,
     resource-usage, capability-use, Object-payload, ObjectTask, JIT,
     external-effect, operation, payload-retention, and terminal-process cleanup
     recovery;
   - `capability.regex_pattern_max_bytes`, `capability.regex_token_max_bytes`,
     `capability.regex_match_timeout_s`, `scheduler.max_workers`, and
     `process.max_tool_calls`;
   - `llm.max_tokens`, `llm.max_input_tokens_per_call`,
     `llm.max_total_tokens_per_call`, `llm.temperature`, and each profile's
     corresponding token limits and temperature;
   - every numeric field in `semantic`: `semantic.max_concurrency`,
     `semantic.assessment_timeout_s`, `semantic.job_lease_s`,
     `semantic.shutdown_join_timeout_s`, `semantic.projection_ttl_s`,
     `semantic.recovery_batch_limit`, `semantic.intent_max_chars`,
     `semantic.projection_max_bytes`, `semantic.assessment_list_limit`,
     `semantic.assessment_list_hard_limit`, `semantic.flow_query_limit`,
     `semantic.flow_query_hard_limit`, `semantic.settlement_list_limit`, and
     `semantic.settlement_list_hard_limit`;
   - `git.ref_list_limit`, `git.pull_request_list_limit`, all six
     `memory.query_scan_*`/`memory.metadata_*` bounds, and
     `skills.max_package_directories`, `skills.max_package_depth`, and
     `skills.catalog_scan_limit`;
   - every numeric field in `task_runs`: `payload_max_bytes`,
     `command_result_max_bytes`, all list/recovery/ledger page sizes and hard
     limits, and `recovery_sample_limit`.

   MCP's strict declarations are enumerated explicitly so generated overlays can
   distinguish them from its ordinary coercing numeric fields:

   - MCP `StrictInt` (25 fields): `mcp.server_page_limit`,
     `mcp.tool_catalog_limit`, `mcp.resource_catalog_limit`,
     `mcp.resource_template_limit`, `mcp.prompt_catalog_limit`,
     `mcp.provider_capability_limit`, `mcp.max_content_blocks`,
     `mcp.max_prompt_messages`, `mcp.max_completion_values`,
     `mcp.schema_regex_pattern_max_bytes`,
     `mcp.schema_regex_max_evaluations`, `mcp.connection_max_open`,
     `mcp.mrtr_max_rounds`, `mcp.mrtr_max_input_requests`,
     `mcp.mrtr_request_state_max_bytes`, `mcp.continuation_max_records`,
     `mcp.continuation_terminal_records`, `mcp.subscription_max_open`,
     `mcp.subscription_queue_events`, `mcp.subscription_event_max_bytes`,
     `mcp.remote_task_max_records`, `mcp.remote_task_terminal_records`,
     `mcp.cursor_handle_limit`, `mcp.subscription_terminal_records`, and
     `mcp.cache_hint_ttl_cap_ms`;
   - MCP `StrictFloat` (8 fields): `mcp.schema_regex_match_timeout_s`,
     `mcp.connection_idle_ttl_s`, `mcp.connection_absolute_ttl_s`,
     `mcp.continuation_ttl_s`, `mcp.subscription_max_lifetime_s`,
     `mcp.remote_task_poll_min_interval_s`, `mcp.remote_task_max_wait_s`, and
     `mcp.oauth_state_ttl_s`;
   - MCP `StrictBool` (2 fields): `mcp.oauth_enabled` and
     `mcp.tasks_extension_enabled`.

   Strict integer fields require integers; strict floating fields accept numeric
   integer or floating-point values; strict boolean fields require YAML/JSON
   booleans rather than `0`, `1`, or string spellings. Post-construction checks
   then reject non-finite, negative, zero, or inverted values as applicable. The
   typed declarations remain authoritative if a later release adds a field.

The config dataclasses are frozen and have no runtime hot reload. In addition,
security-critical `shell.rules[*].conditions` values are defensively copied and
recursively frozen, so ordinary nested mapping or sequence mutation cannot
change a Shell policy after a Runtime opens. Treat the complete configuration
as read-only; change the host configuration and open a new Runtime. LLM client
environment inputs are a narrow exception: each profile resolution captures
the applicable legacy values plus its named API-key and safety-identifier
variables in one immutable snapshot. A change to any effective
client-construction option invalidates and closes the cached client. The
private, in-memory cache fingerprint is secret-derived, but neither it nor the
underlying values are persisted, emitted in audit or GUI output, or used as the
public Sink identity hash.

Library callers should pass an explicit config object when they need a
different composition:

```python
from agent_libos import Runtime
from agent_libos.config import load_config_file

runtime = Runtime.open(config=load_config_file("agent-config.yaml"))
try:
    # Use the Runtime.
    ...
finally:
    runtime.shutdown(actor="library", reason="library.complete")
```

The public loader helpers support explicit library composition:

- `get_project_root(start=None)` finds the root that owns the installed
  `agent_libos` tree.
- `load_config_file(path, base=DEFAULT_CONFIG)` overlays an explicit YAML file
  on a selected immutable base. Relative paths are owned by the caller's current
  working directory.
- `load_config_from_project_root(filename="config.yaml", base=DEFAULT_CONFIG,
  root=None)` joins a relative filename to the detected or explicit project
  root and returns `base` unchanged when the file is absent. This trusted Host
  helper is path resolution, not a containment boundary: an absolute filename
  or a relative name containing `..` may select a file outside that root.
- `load_config_from_cwd(filename="config.yaml", base=DEFAULT_CONFIG)` is the
  explicit library opt-in to current-working-directory discovery. Product
  entrypoints do not call it.

### Bounded YAML input

Configuration files must be UTF-8 mappings and are read through the shared
bounded YAML loader. It rejects non-string mapping keys, duplicate keys in a
source mapping, recursive aliases, invalid scalar construction, and input over
1,048,576 UTF-8 bytes. Bootstrap-fixed parser ceilings are 60,000 parse events,
32,768 nodes (including the expanded alias graph), nesting depth 64, 64 aliases,
4,300 digits in an integer scalar, and 1,048,576 bytes of expanded scalar text.
These limits intentionally do not depend on the configuration being parsed.
YAML merge syntax is supported only within the same unique-key and bounded
alias rules; a document whose root is not a mapping fails before Pydantic
validation.

Loaders never mutate `base`, so callers may intentionally layer overlays:

```python
base = load_config_file("deployment-base.yaml")
config = load_config_file("host-overrides.yaml", base=base)
runtime = Runtime.open(config=config)
```

The checked-in repository `config.yaml` selects `.agent_libos.sqlite` and loads
the trusted PTY Runtime Module. Consequently, omitting `--db` while using this
checkout selects that persistent store. In an installed package or source tree
without a project-root config, `DEFAULT_CONFIG.runtime.local_store_target` is
`local`, an in-memory SQLite store. Scripts and documentation that require state
across separate CLI invocations should always pass `--db` explicitly.
That persistence covers process metadata, authority, evidence, and other SQL
records, not ordinary Object payloads. Current Object rows hold runtime-memory
markers; if their payload cache is unavailable on reopen they are released
fail-closed. The sole cross-reopen exception is a committed root spawn's
initial goal when `llm.persist_full_io` is true: a separate bounded recovery
envelope retains the exact payload, and startup rehydrates it only after
validating the same active root process and unchanged goal id, Object identity,
version, immutable flag, live state, and runtime-memory marker. Only the
immutable initial GOAL created by `ProcessManager` is eligible; mutable goal
handles are not. The envelope is independently bound to the initial image
recorded by the spawn publication; a later exec may change the process image
without invalidating an otherwise unchanged goal.
Terminal cleanup redacts the envelope to hash-only; failed launch rollback and
startup compensation do so before committing a non-committable publication
state. Child/fork goals, exec replacement goals, ObjectTask owners, and all
other Object payload producers and consumers must still execute within one
Runtime lifetime.

Relative paths intentionally have different owners. `--config` and explicit
`module_manifests=` paths resolve from the caller's current working directory;
configured `modules.manifest_paths` resolve from the Agent libOS project root.
A relative SQLite `--db` or `runtime.local_store_target` resolves from the
current working directory when the store opens. With the default local
substrate, that same directory is the workspace root, and a relative
`git.worktree_root` resolves below it. An injected substrate can select a
different workspace root, so library hosts should use explicit absolute store
and module paths when those roots must not depend on process startup location.

### Effective LLM profile precedence

For a root spawn, an explicit Host-selected profile id wins, then the selected
image's `llm_profile`, then `llm.default_profile_id`. An exec keeps the
process's current profile unless the Host supplies a replacement; a child
process likewise inherits its parent's profile unless explicitly overridden.
The CLI reads config profiles only. The GUI may dynamically register a
user-level profile, but its create/update API rejects ids owned by config as
read-only. At the lower-level Host registry API, an explicitly dynamically
registered profile id shadows a config profile while that Runtime is open.

Within a resolved profile, a non-null profile field wins. Only the profile
whose id equals `llm.default_profile_id` then inherits the matching legacy
`OPENAI_*` environment value; other named profiles do not inherit ambient
endpoint, model, or provider-policy settings. When neither is present,
`llm.timeout_s`, `llm.max_retries`, `llm.api_mode`, `llm.store`,
`llm.safety_identifier`, `llm.prompt_cache_key`,
`llm.prompt_cache_retention`, `llm.prompt_cache_mode`,
`llm.prompt_cache_ttl`, `llm.responses_previous_response_id`,
`llm.parallel_tool_calls`, `llm.auto_wait_on_empty_tool_calls`,
`llm.fallback_json_actions`,
`llm.temperature`, `llm.max_tokens`, `llm.max_input_tokens_per_call`,
`llm.max_total_tokens_per_call`, and `llm.context_window_tokens` supply their
group defaults. The
legacy mappings are
`OPENAI_BASE_URL`; `OPENAI_LANGUAGE_MODEL` then `OPENAI_MODEL`;
`OPENAI_TIMEOUT`; `OPENAI_MAX_RETRIES`; `OPENAI_API_MODE`; `OPENAI_STORE`;
`OPENAI_REASONING_EFFORT`; `OPENAI_VERBOSITY`; `OPENAI_SAFETY_IDENTIFIER`;
`OPENAI_PROMPT_CACHE_KEY`; `OPENAI_PROMPT_CACHE_RETENTION`;
`OPENAI_PROMPT_CACHE_MODE`; `OPENAI_PROMPT_CACHE_TTL`;
`OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID`; and
`OPENAI_PARALLEL_TOOL_CALLS`; and `OPENAI_FALLBACK_JSON_ACTIONS`. The default profile also snapshots
`OPENAI_ENABLE_THINKING` and the OpenAI organization/project routing variables
(`OPENAI_ORGANIZATION`/`OPENAI_ORG_ID` and
`OPENAI_PROJECT`/`OPENAI_PROJECT_ID`). Named non-default profiles suppress the
OpenAI SDK's own ambient endpoint and organization/project fallbacks. Agent
libOS does not support `OPENAI_CUSTOM_HEADERS`, `OPENAI_ADMIN_KEY`, or
`OPENAI_WEBHOOK_SECRET` as profile settings; SDK-side values for those variables
are removed before any provider request rather than becoming untracked profile
state.

The default LLM timeout is 180 seconds. This accommodates long-context final
verification without turning a slow but still-running request into multiple
ambiguous, potentially billable transport retries. Hosts can lower or raise it
with a selected profile's `timeout_s` or, for the default profile only,
`OPENAI_TIMEOUT`.

Agent libOS performs the configured `max_retries` as explicit, separately
traced transport retries; provider-SDK internal retries are disabled. After
Agent libOS exhausts those retries, timeout, connection, rate-limit, and
retryable HTTP status failures are recorded and pause the AgentProcess for
explicit Host resume. Resuming issues a new, separately accounted provider call
from the durable process context. Deterministic configuration, protocol, and
non-retryable provider errors still fail the process closed; the Runtime never
loops indefinitely on an unavailable provider.

Secrets are the exception to that fallback description: every profile reads
its API key only from the environment variable named by its `api_key_env`.
`safety_identifier_env`, when set and no literal `safety_identifier` is set,
is authoritative: an unset or empty named variable produces no safety
identifier and does not fall back to the default profile's legacy identifier.
A custom base URL is permitted when either the profile sets
`allow_custom_base_url: true` or the Host sets
`AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1`.

## Inspecting exact defaults

Defaults change with the code. Use the generated
[exact configuration reference](configuration_reference.md) for the complete
path/type/default snapshot, or print the exact values for the current checkout
instead of copying a stale sample:

```bash
uv run python -c 'import json; from agent_libos.config import DEFAULT_CONFIG; from agent_libos.utils.serde import to_jsonable; print(json.dumps(to_jsonable(DEFAULT_CONFIG), indent=2, sort_keys=True))'
```

The generated reference is the single exhaustive field inventory. It is grouped
for quick navigation:

- lifecycle and execution: [`runtime`](configuration_reference.md#runtime),
  [`scheduler`](configuration_reference.md#scheduler),
  [`process`](configuration_reference.md#process), and
  [`tools`](configuration_reference.md#tools);
- authority and provider boundaries:
  [`capability`](configuration_reference.md#capability),
  [`data_flow`](configuration_reference.md#data_flow),
  [`shell`](configuration_reference.md#shell),
  [`git`](configuration_reference.md#git),
  [`jsonrpc`](configuration_reference.md#jsonrpc), and
  [`mcp`](configuration_reference.md#mcp);
- model and semantic behavior: [`llm`](configuration_reference.md#llm),
  [`llm_context`](configuration_reference.md#llm_context), and
  [`semantic`](configuration_reference.md#semantic);
- durable artifacts and extension catalogs:
  [`memory`](configuration_reference.md#memory),
  [`object_tasks`](configuration_reference.md#object_tasks),
  [`task_runs`](configuration_reference.md#task_runs),
  [`checkpoint`](configuration_reference.md#checkpoint),
  [`image`](configuration_reference.md#image),
  [`image_commit`](configuration_reference.md#image_commit),
  [`skills`](configuration_reference.md#skills), and
  [`modules`](configuration_reference.md#modules); and
- Host/UI/default catalogs: [`gui`](configuration_reference.md#gui),
  [`launcher`](configuration_reference.md#launcher), and
  [`scripts`](configuration_reference.md#scripts).

Three fields remain accepted only so existing 1.x configuration files keep
loading: `runtime.runtime_db_filename`, `tools.sandbox_timeout_s`, and
`image_commit.metadata_preview_chars`. They are validated but ignored; new
configurations should omit them. Store targets, operation-specific timeouts,
and full bounded image metadata are authoritative. Removing these compatibility
inputs requires a major-version migration.

`launcher` and `scripts` are validated code-level default catalogs, not settings
consumed by the CLI or GUI Runtime assembly. The current standalone scripts bind
those values directly from `DEFAULT_CONFIG` and do not load project or explicit
YAML overlays. Consequently, putting either group in an overlay constructs the
requested `AgentLibOSConfig` value but does not change an existing built-in
launcher or script. Use that script's command-line options where available, or
pass the selected values explicitly from library code.

`runtime.default_image_id` and `runtime.coding_image_id` name two different
built-in images. They must also not collide with the fixed
`maintenance-agent:v0`, `research-agent:v0`, `analysis-agent:v0`,
`operator-agent:v0`, `review-agent:v0`, `toolmaker-agent:v0`, or
`context-compressor:v0` ids; startup fails on a collision rather than allowing
one built-in definition to replace another.

The generator contract in `tests/unit/test_generated_documentation.py` walks
the dataclass graph, including nested mapping/sequence templates, and compares
the resulting paths and scalar defaults with the checked-in generated page.

Skill package topology and catalog discovery have independent Host ceilings:

| Skills field | Default | Semantics |
| --- | ---: | --- |
| `skills.max_package_directories` | `256` | Maximum aggregate resource directories traversed while loading one Host or workspace Skill package. Missing configured resource roots are not charged; exceeding the bound rejects the package. |
| `skills.max_package_depth` | `32` | Maximum resource-directory depth in one Skill package, with each configured resource root at depth 1. A deeper package is rejected before descending further. |
| `skills.catalog_scan_limit` | `1000` | Maximum Host catalog entries or registered Skill rows examined for one metadata discovery. Detecting entry or row N+1 fails explicitly instead of returning an incomplete search result. |

The generated [`llm.profiles.<key>`
template](configuration_reference.md#llm) lists every accepted profile field;
the same expansion covers sequence templates such as Shell command rules and
the optional semantic policy epoch. Optional Runtime Modules may also own
module-local settings that are not fields of `AgentLibOSConfig`.

### Semantic Phase 2–4 configuration

Semantic assessment is disabled by default:

```yaml
semantic:
  mode: "off"
  adapter: deterministic
  external_profile_id: null
  policy_epoch: null
  max_concurrency: 2
  assessment_timeout_s: 30.0
  job_lease_s: 60.0
  shutdown_join_timeout_s: 2.0
  projection_ttl_s: 300
  recovery_batch_limit: 500
  intent_max_chars: 2000
  projection_max_bytes: 16384
  assessment_list_limit: 100
  assessment_list_hard_limit: 1000
  flow_query_limit: 100
  flow_query_hard_limit: 1000
  settlement_list_limit: 100
  settlement_list_hard_limit: 1000
```

`mode` is exactly `off`, `shadow`, `enforce_deny`, or `canary_auto`; off
performs no capture writes, claims no jobs, and is the global kill switch.
`enforce_deny` and `canary_auto` require an immutable static `policy_epoch`;
the latter also requires at least one auto-approval rule and the `external`
adapter. `adapter` is
`deterministic`, `scripted`, or `external`. Scripted is
for deterministic development/test fixtures. An `external_profile_id` is
forbidden for the other adapters. `max_concurrency` is positive and at most 32;
assessment, lease, and shutdown-join timeouts are positive, and the lease must
be at least as long as both timeout values.
Projection TTL and list limits are positive, and `projection_ttl_s` must be at
least `job_lease_s`; each assessment, flow, and settlement selected limit
cannot exceed its hard limit, and the new flow and settlement hard limits
cannot exceed 1,000. Recovery/cleanup pages use the positive
`recovery_batch_limit`, capped at 500. The intent bound cannot exceed 2,000
characters and the encoded projection bound must be from 512 through 16,384
bytes.
`assessment_list_limit` is the default for a direct Host service query and the
effective direct-query maximum is the smaller of
`assessment_list_hard_limit` and 500. CLI and GUI HTTP callers impose a
separate, stricter default of 50 and maximum of 100.
Approval and provider-ingress capture always set outbound intent to `null`.
Root-goal capture can include deterministic `redacted_intent` only for a
non-empty `public`/`normal`, non-mixed-identity goal no longer than
`intent_max_chars` and only when local credential/secret DLP and conservative
path detection find no match. Otherwise it falls back to metadata-only without
truncating a secret; the local Host detector freezes only closed category/
reason/evidence-digest findings for terminal merge. `intent_max_chars` tightens
that root-goal bound but cannot exceed 2,000 characters; it is not a switch
that enables approval or provider payload export.

An external adapter can be staged while mode is off without resolving a
profile. Enabling any external semantic mode requires a configured named
profile other than `llm.default_profile_id`, with an explicit model; explicit
`api_mode: chat` or `api_mode: responses`; `store: false`; `max_retries: 0`;
`responses_previous_response_id: false`; `fallback_json_actions: false`; and a
finite timeout compatible with `semantic.assessment_timeout_s`. Prompt caching
must be disabled both on that profile and in the global `llm` defaults: neither
level may set a cache key, retention, or TTL; the profile cache mode may only be
unset or `provider_default`, and the global cache mode must be
`provider_default`. The runtime freezes the profile snapshot identity and
explicit model at assembly; assessment rechecks snapshot/resolution/client
identity and model/timeout, and the Protected Operation revalidates the
profile-bound Sink. Drift fails closed instead of inheriting permissive
default-profile state. This is a classifier-only profile; ordinary processes
continue to use their own selected LLM profiles. See [Semantic Approval and Data
Identification](semantic_shadow.md#external-classifier-configuration).

Tenant bucketing is intentionally not a YAML setting. The default is no
bucketer and persisted `tenant_bucket_sha256` remains `null`. An embedded Host
may inject `semantic_tenant_bucketer=` at Runtime construction and should back
it with a deployment-keyed HMAC; no CLI, HTTP, GUI, model, Skill, JIT, or Module
surface can install or replace it. `canary_auto` additionally requires the
policy epoch to pin that exact profile id, the classifier profile identity
digest, and the classifier model digest. These three classifier identity fields
must always be supplied together; canary mode requires all three.

`SemanticPolicyEpochV1` is a closed static Host object. It contains
`schema_version`, `catalog_version`, `epoch_id`, positive `generation`,
`expected_previous_sha256`, exact `tenant_bucket_sha256s`, closed
`auto_approval_rules` and `hard_deny_rules`, the optional grouped classifier
profile id/profile digest/model digest identity, `minimum_confidence_bps`,
`required_calibration_bucket`, Capability TTL, rate/day/inflight ceilings, and
`created_at`. It requires at least one rule; auto rules require at least one
exact tenant digest. Rule ids are unique across both arrays. Catalog v1 permits
automatic candidates only for `filesystem.read`, `git.read`, and `git.diff`
with their exact single read-like right.

When generation 1 contains auto-approval rules, its tenant digest set must
contain exactly one bucket. Later generations may contain more than one exact
tenant digest, subject to the other static policy and rollout checks.

The confidence floor cannot be below 9,900 basis points, calibration is fixed
to `very_high`, Capability TTL cannot exceed 300 seconds, and the respective
rate ceilings cannot exceed 10/minute, 100/day, or 2 inflight. Static
configuration can only narrow those values. Activation compares the expected
previous policy digest and durable generation; conflicts fail assembly. There
is no CLI, HTTP, GUI, model, Skill, JIT, or Module policy activation/revocation
surface.

Checkpoint snapshot format versions are owned by the runtime codec and are not
configurable. A runtime release emits only the snapshot version it can decode.

## Security-sensitive settings

- `runtime.store_dsn` may contain PostgreSQL credentials. Prefer an
  environment-specific untracked overlay; never commit a real DSN.
- LLM profiles store an `api_key_env` variable name, never the API-key value.
  Only the selected host process reads the named environment variable.
- `llm.context_window_tokens` defaults to `131072` and `llm.max_tokens`
  defaults to `16384`; a profile may override either value. Effective
  `max_tokens` must be smaller than the effective model window. The
  lower default output reservation leaves room for multi-quantum task context;
  increase it per profile only when a task genuinely needs longer single-call
  output. The window controls local pressure management only and is deliberately
  excluded from the Provider/Sink identity hash.
- `llm.max_input_tokens_per_call` defaults to `114688` and
  `llm.max_total_tokens_per_call` defaults to `131072`. Profiles may override
  either positive integer. The effective input ceiling and `max_tokens` must
  each be no greater than the effective total ceiling. After assembling the
  exact request, Runtime rejects a local estimate above the input ceiling or an
  estimate-plus-output reservation above the total ceiling. It then reserves
  the declared maximum envelope before Provider dispatch. These local admission
  fields are excluded from Provider/Sink and client-cache identity; they do not
  claim tokenizer-exact Provider billing or monetary cost control.
- JSON-RPC/MCP header and stdio allowlists contain exact environment-variable
  names or trailing-`*` prefix patterns. MCP configuration requires each entry
  to be either a valid environment-variable name or a valid, non-empty prefix
  followed by exactly one `*`. For example, `AGENT_LIBOS_MCP_*` admits names
  beginning with `AGENT_LIBOS_MCP_`. A bare `*`, internal or repeated `*`,
  whitespace, and invalid names are rejected for both MCP allowlists; an empty
  MCP allowlist is valid and denies every manifest-selected variable.
  Manifests reference
  those names; resolved secret values must not be persisted in registry rows,
  audit metadata, benchmark provenance, or GUI responses.
  MCP Manifest v2 requires an explicit `protocol_mode`; Manifest v3 requires
  exact `"2026-07-28"`. Neither is a mutable global configuration fallback.
  Manifest v1 omits that field and is permanently legacy-wire. Header names
  are checked case-insensitively. Every version rejects
  protocol/session/resume, trace, and baggage controls. Manifest v2/v3
  additionally reject all reserved content-negotiation and `Mcp-Param-*`
  headers and protocol `_meta` keys;
  Manifest v1 retains its compatibility support for `Accept`, `Mcp-Param-*`,
  and application `_meta`, but none of those exceptions permits the common
  high-risk controls.
- `mcp.timeout_s` defaults to `10.0` seconds (with a `60.0` second hard limit)
  and is one absolute exchange deadline, not a fresh timeout for each I/O
  stage. DNS queueing and resolution, every resolved-address connect attempt,
  TLS, request writes, response headers/body chunks, stdio spawn, and stdio
  protocol work all consume that same deadline. For Manifest v2 that includes
  discovery/initialization, at most 16 `tools/list` pages, and Tool dispatch;
  the automatic modern probe receives at most five seconds and never outlives
  the remaining deadline. `mcp.max_request_bytes`
  defaults to `65536` and `mcp.max_response_bytes` to `1048576`; provider byte
  counters are cumulative across those phases, must cover the canonical JSON
  payload, and may not under-report it. The complete Manifest v1/v2 live Tool
  catalog cannot exceed the deprecated compatibility setting `mcp.list_limit`
  (100 by default).
  Stdio additionally caps each response frame at `max_response_bytes`, total
  stdout at four times that value, stderr at that value, each request frame at
  `max_request_bytes`, and total stdin at four times that value. Stdio children
  also inherit the process's remaining subprocess wall/CPU/memory budgets;
  unavailable metrics fail closed when a configured CPU or memory limit cannot
  be enforced.
- `mcp.protocol_probe_timeout_s` defaults to `5.0`; it must be positive and
  cannot exceed the release-locked `5.0` second maximum. The probe is further
  truncated by the operation's shorter remaining absolute deadline rather
  than silently extending either limit. `mcp.list_max_pages` defaults to 16.
  Manifest v2 schema safety
  uses `mcp.schema_max_depth: 64`, `mcp.schema_max_nodes: 10000`,
  `mcp.schema_max_ref_hops: 128`, and
  `mcp.schema_max_composition_expansions: 1024`. These are Host policy caps;
  a server or model cannot raise them at dispatch time.
  For Manifest v1, v2, and v3, JSON Schema `pattern` and
  `patternProperties` validation additionally uses a shared maximum of 4,096
  regex evaluations, a shared 50 ms monotonic matching deadline, and a
  1,024-byte UTF-8 cap per pattern. These defaults are configured by
  `mcp.schema_regex_max_evaluations`,
  `mcp.schema_regex_match_timeout_s`, and
  `mcp.schema_regex_pattern_max_bytes`. An exhausted budget, timeout, invalid
  regex, or oversized pattern fails closed before provider dispatch.
  `mcp.manifest_max_bytes` may be lowered from its 262,144-byte default but may
  not exceed the shared YAML loader's fixed 1,048,576-byte UTF-8 ceiling.
- `mcp.list_limit` is accepted only as a deprecated Manifest v1/v2
  `tools/list` compatibility limit. It is not a fallback for registered-server
  pages or Manifest v3 Tools. A YAML overlay that explicitly contains the
  field emits one filterable `AgentLibOSConfigDeprecationWarning` with code
  `deprecated_mcp_list_limit`; constructing or loading defaults does not emit a
  warning. Equal or different values do not couple the independent limits:
  migrate registered-server consumers to `server_page_limit` and Manifest v3
  Tool catalogs to `tool_catalog_limit`.
- MCP's modern-client bounds are purpose-specific. `server_page_limit`,
  `tool_catalog_limit`, `resource_catalog_limit`, `resource_template_limit`,
  and `prompt_catalog_limit` prevent one catalog from borrowing another's
  budget. `provider_capability_limit`, `max_content_blocks`,
  `max_prompt_messages`, and `max_completion_values` bound decoded provider
  results before they reach a Host consumer. Connections have both idle and
  absolute TTLs and a global open-count ceiling. MRTR continuations,
  subscriptions, and remote Tasks have independent count, byte, time, and
  queue limits; exhausted limits fail closed without reconnecting or replaying
  an operation. `cursor_handle_limit` bounds the opaque cursor vault,
  `subscription_terminal_records` bounds retained terminal diagnostics, and
  `cache_hint_ttl_cap_ms` caps untrusted provider cache-hint TTLs. The modern
  client does not implement a response-body cache.
- Pre-release names `cache_max_entries`, `cache_max_bytes`,
  `cache_ttl_cap_ms`, and `cache_public_cross_principal` are no longer accepted;
  the strict configuration loader rejects them as unknown keys. Move opaque
  cursor and terminal-diagnostic bounds to `cursor_handle_limit` and
  `subscription_terminal_records`, and move the provider hint TTL cap to
  `cache_hint_ttl_cap_ms`. There is no replacement response-cache setting
  because no response-body cache exists.
- OAuth and the digest-pinned Tasks extension are disabled by default.
  Enabling Tasks requires `tasks_extension_spec_sha256` to be an exact
  lowercase SHA-256 pin; a server-advertised digest cannot select or update it.
  OAuth credentials and tokens are supplied by a Host credential broker and
  are not configuration-file fields. The default system broker recognizes only
  the reviewed `keyring==25.7.0` OS implementations listed in the support
  matrix; Chainer/plugin/third-party backends and unreviewed versions fail
  closed even when they advertise a positive priority. A Host-approved custom
  secure store must be supplied explicitly as the caller-owned
  `substrate.mcp_credential_broker` SPI. Provider cache hints are diagnostics
  and never authorize reuse, cross-principal sharing, or retention of response
  bodies.
- `tools.shell_timeout_s` defaults to `30.0` seconds and
  `shell.timeout_hard_limit_s` to `300.0`. Provider timeout and subprocess-limit
  failures are recognized through their causal wrapper chain, charged once to
  the process resource ledger, and surfaced as the corresponding safe Runtime
  timeout/resource-limit result. Arbitrary nested provider text is not added to
  the public error allowlist.
- Filesystem reads default to `tools.filesystem_read_max_bytes=65536` and are
  capped by `tools.filesystem_read_hard_limit_bytes=1048576`. On Darwin,
  existing workspace entries derive capability, lock, and file-label keys from
  descriptor-backed `F_GETPATH` spelling so case and Unicode aliases cannot
  create parallel identities. Future names use the containing volume's
  reported case-sensitivity/case-preservation and APFS normalization semantics; ambiguous
  non-ASCII case folds, unknown volume comparison semantics, or unavailable
  canonical identity evidence fail closed for that future path. Existing paths
  remain usable when only future-name volume metadata is unavailable. Linux
  case- and normalization-distinct entries remain distinct.
- `git.executable` is resolved on a Host path outside the workspace. The
  default `git.minimum_version` is `2.26.0`; deployments may configure a
  different dotted numeric threshold. `git.worktree_root` must remain below the workspace, while
  `git.trusted_metadata_roots` is a Host trust decision for linked-worktree
  metadata and should be as narrow as possible. Remote URL schemes, local file
  remotes, credential-helper inheritance, and SSH-agent inheritance are Host
  policy; model tools cannot override them. Metadata protection is a fixed
  invariant: `git.protect_git_metadata` must remain `true`. Disabling Git
  or failing executable/version validation affects only Git calls, not Runtime
  startup. See [Git Provider and Primitive](git.md).
- `llm.persist_full_io` defaults to true. Set it to false when the deployment's
  user agreement does not authorize retention of full prompts, tool schemas,
  reasoning, outputs, successful response content, and the bounded
  provider-response projection stored in the `raw_response` field. That field
  is not a byte-for-byte SDK response: sensitive and opaque values are hashed
  and oversized structures are replaced by omission metadata and a digest.
  Provider and extension exception text is never persisted or exposed
  to the model, regardless of this setting. The opt-out persists canonical
  content-free summary envelopes containing byte counts,
  JSON shape/count metadata when available, and hashes rather than readable
  previews. It also redacts
  conditional LLM release resume rows before approval; exact same-runtime
  approval remains supported, while reopen fails that unrecoverable release
  closed instead of rebuilding or dispatching it. `prompt_mode: image_only`
  cannot use this opt-out: custom Images default to that mode, and it fails
  before provider dispatch unless the lossless native transcript can be written
  with `persist_full_io=true`. The setting also controls the committed root
  spawn's initial-goal recovery envelope: true retains its bounded exact payload
  while the committed root process remains active, while false writes only
  content-free identity, size, and hash fields and therefore cannot support a
  later CLI `run` after reopen. Terminal cleanup and failed-launch compensation
  reduce a reversible envelope to the same hash-only form.
- `task_runs.enabled` defaults to true and controls admission of new Run
  creation. The manager and startup reconciliation remain assembled when it is
  false so existing durable state cannot evade recovery.
  `task_runs.plaintext_payloads_enabled` defaults to false, and Durable Task Run
  creation is rejected while it is false because a
  restartable Run requires a readable goal and resume material. Enabling it
  permits bounded plaintext Task Run payloads in the SQL store; it is
  independent of `llm.persist_full_io` and provides no at-rest encryption.
  `task_runs.payload_max_bytes` defaults to `1,048,576` serialized bytes per
  canonical payload and must be positive. `task_runs.command_result_max_bytes`
  separately bounds persisted idempotent command results at `262,144` bytes.
  Normal Host lists default to `list_page_size=100` and are bounded by
  `list_hard_limit=1000`; ledger pages default to and are capped at `500`.
  `recovery_sample_limit=100` bounds the in-memory diagnostic sample returned
  after startup reconciliation, not the number of Runs reconciled. A
  version-1 Task Run title and opaque identity are fixed to 256 characters,
  while public blocker/summary metadata is fixed to 16 KiB. Those versioned
  interchange bounds are not Host configuration knobs. A Run's default
  `purge_on_terminal` policy reduces readable Run-owned payloads
  to hashes before terminalization; `permanent` is a Host/admin-only per-Run
  choice.
- Provider-side Responses storage policy remains opt-in through `llm.store`.
  `llm.responses_previous_response_id` permits low-level client chaining policy,
  but the current full-snapshot AgentProcess executor records it as configured
  and disabled and never sends `previous_response_id`. Enabling either setting
  can still change the trusted profile identity, and enabling `store` may
  increase provider retention.
- `runtime.launch_authority_mode: manifest_required` treats image capability
  requirements as declarations, not grants. It is the only accepted value in
  the current 1.x contract.
- `runtime.publication_recovery_max_attempts` bounds durable compensation
  retries. Exceeding it persists a `manual` publication disposition and fails
  every startup closed instead of silently repeating an uncertain cleanup
  forever. Under the exclusive runtime-store lease, startup may take over a
  claim left by a prior runtime instance; takeover consumes a fresh attempt and
  uses a new recovery lease.
- Runtime-publication startup recovery and launch/exec terminal-operation
  reconciliation use exact kind/state/marker keyset pages sized by
  `runtime.publication_reconciliation_page_size`. The configured hard limit is
  enforced by the repository; online terminalization marks completed rows in
  the same transaction so reopen does not rescan settled history.
- Exact publication-compensation artifact lookups reject receipt identity sets
  above `runtime.publication_artifact_lookup_hard_limit`. Tool existence reads
  are primary-key batched, and process tool escape checks use the normalized
  `process_tool_bindings` reverse index rather than scanning process JSON.
- Startup JIT rehydration keyset-pages only normalized durable ephemeral
  bindings by `(pid, tool_name)`, with every SQL page bounded by
  `runtime.jit_rehydration_page_size`. Tool and process mutations maintain an
  exact indexed eligibility bit in the binding projection in the same
  transaction. The partial covering index therefore makes database work
  proportional to eligible JIT bindings even when callable history is sparse.
  Each page performs one batched exact tool/candidate lookup rather than one
  lookup per process, and it never decodes unrelated process control state or
  materializes a process's complete binding set.
  The opaque startup recovery lease is required before the first durable read.
  Exact restored/pruned totals are returned with at most one page of samples;
  historical scans and diagnostic buffers are bounded, while the final loaded
  registry remains proportional to the number of active JIT tools.
- External-effect startup reconciliation is keyset-paged by
  `runtime.external_effect_recovery_page_size` and rejects any page above the
  configured hard limit. Active provider-usage reservations are independently
  scanned through indexed keyset pages sized by
  `runtime.resource_usage_reservation_recovery_page_size`. Recovery requires
  the opaque startup lease before its first read, atomically couples each
  settlement to its actual usage charge, and returns an exact count plus at
  most one page of sample IDs. Accordingly,
  `Runtime.recovered_resource_usage_reservations` is now a
  `ResourceUsageReservationRecoverySummary`, not an unbounded `list[str]`.
  Capability-use reservations are recovered only after prepared protected
  effects have restored their linked reservations. Remaining stale rows are
  abandoned through the status-first keyset index using
  `runtime.capability_use_reservation_recovery_page_size`; recovery requires
  the same opaque startup lease and exposes only an exact count plus one
  bounded sample page.
  Stale-running operation recovery is independently
  keyset-paged by `runtime.operation_recovery_page_size`. One store-locked,
  connection-local temporary index materializes the running ancestors of
  indexed pending/unknown effects; each page then performs only a bounded
  primary-key membership read. Recovery processes the full backlog but exposes
  only a page-bounded ID sample plus the exact total count.
  Durable Task Run startup work is independently keyset-paged by
  `task_runs.recovery_page_size`, which defaults to `500` and cannot exceed
  `task_runs.recovery_page_hard_limit` (default `5000`). These
  values bound each query and diagnostic sample, not the number of eligible
  Runs that startup must reconcile. Payload integrity and stale Runtime epochs
  are checked before any Run is returned to scheduling.
  Terminal-process cleanup recovery uses the independent
  `runtime.process_terminal_cleanup_recovery_page_size` setting, which defaults
  to `500`; its store-enforced hard limit
  `runtime.process_terminal_cleanup_recovery_page_hard_limit` defaults to
  `5000`. Both values must be positive and the page size cannot exceed the hard
  limit. Startup walks only incomplete cleanup intents through the partial
  `(created_at, pid)` keyset index, processes the complete backlog, and retains
  at most one page of recovered PID or failure samples alongside exact totals.
  These settings bound each query and diagnostic buffer, not the number of
  durable cleanup intents that recovery will finish. Failure samples contain
  only exception type, byte count, and SHA-256 fingerprint, never exception
  text.
  Payload retention is separately opt-in through
  `runtime.payload_retention_enabled`; startup never runs it implicitly. The
  summary/hash ages and page limits are validated together, and every applied
  maintenance page is lifecycle-gated, CAS-protected, and audited in the same
  transaction. See [Evidence and LLM Payload Retention](evidence_payload_retention.md).
- `data_flow.default_trust_level` is fixed to `untrusted`, and
  `default_max_sensitivity` cannot exceed `normal`. Higher clearance is valid
  only in a Host-owned `sink_rules` record. Rules accept exact or terminal-`*`
  patterns; provider-backed LLM, JSON-RPC, MCP, Shell, and PTY rules above
  `normal` require an `identity_sha256`. Duplicate or equal-priority overlapping
  patterns fail configuration loading. See [Data Flow](data_flow.md).
- `data_flow.operation_minimum_integrity` maps an exact core protected-operation
  contract name to `untrusted`, `unknown`, `checked`, or `verified`. Overrides
  may only tighten an egress/bidirectional contract; unknown names, non-egress
  targets, invalid values, and attempted weakening fail Runtime composition.
- `data_flow.registry_resource` is the `admin` capability resource for runtime
  registry mutations. `registry_list_limit`, `decision_list_limit`, and
  `file_binding_list_limit` bound active control-plane reads; they do not
  truncate append-only decision or binding history in storage.
- Human response ingress is bounded by
  `tools.human_response_payload_max_bytes` (default `131,072` serialized
  bytes), `tools.human_response_max_depth` (default `32` nested containers),
  and `tools.human_response_max_nodes` (default `4,096` JSON values). These
  limits apply to complete GUI, CLI, Host, and terminal-provider decisions
  before they can choose an approval, grant authority, or mutate request or
  process state. An invalid terminal-provider response leaves the Human request
  pending after the protected provider read is evidenced; an invalid direct
  decision has no decision side effects.
- `tools.executable_snapshot_sibling_limit` bounds the direct sibling entries
  linked beside a mutable workspace executable snapshot before Shell, MCP
  stdio, or PTY dispatch. Exceeding the limit, failing enumeration, or failing
  to link any required sibling aborts the dispatch instead of exposing a
  partial snapshot.

## Bounded windows

GUI and context limits operate at two different layers and must not be treated
as equivalent:

- `llm.tool_output_prompt_max_chars` bounds each tool-result string when the
  Responses adapter reconstructs provider input (default `32,768` characters).
  Oversized outputs carry deterministic included/omitted character counts;
  the setting does not reduce the separate durable ToolResult evidence limit.
- `llm_context.policy` accepts `source_only` (the default) or
  `llm_context_object`. `source_only` leaves the already materialized context
  unchanged and creates no persistent delta Object. A Host may opt in globally
  with `llm_context_object`, or opt in one process with explicit
  `context:enrichment/execute` authority.
- `llm_context.recent_event_limit` bounds the earliest next post-cursor event
  rows loaded from SQL for an explicitly enabled persistent-context
  preparation. This oldest-first page preserves a gap-free advancing cursor; it
  does not activate delta capture by itself.
- `llm_context.prompt_event_payload_max_chars` bounds each represented event's
  provider-neutral model payload (default `2,048` characters). Oversized
  payloads retain compact actionable fields plus deterministic omission counts
  and SHA-256 digests; original event evidence remains unchanged.
- `llm_context.storage_compaction_threshold_bytes` is the persisted-context
  waterline that starts automatic compaction before the generic Object Memory
  payload hard limit. It must be positive and strictly less than
  `tools.memory_payload_hard_limit_bytes`. The Image's
  `planner.context_management.mode` must resolve to `auto_compact`, and the
  process must also have both explicit persistent enrichment and explicit
  `context:maintenance/execute` authority; otherwise the proactive storage
  waterline does not elevate its authority or invoke the compactor.
- `llm_context.storage_compaction_max_chunks` bounds the built-in compressor
  stages used specifically for storage-waterline maintenance. The default is
  four; an Image's explicit `planner.context_management.tool.arguments.max_chunks`
  still takes precedence. Values must be between 1 and the tool-schema hard
  maximum of 64.
- `llm_context.storage_compaction_preserve_recent_entries` controls how many
  verbatim tail entries storage-waterline maintenance retains in addition to
  the cumulative summary. The default is zero because compressor children and
  their result objects can otherwise make a newly compacted context cross the
  waterline again. An Image's explicit tool argument still takes precedence;
  values must be between 0 and 128.
- `gui.snapshot_event_limit` bounds snapshot event reads and is also the maximum
  accepted `limit` for the process-events API. Older pages use its `before`
  cursor rather than loading all process events.
- The audit, global LLM-call, per-process message, and ObjectTask snapshot
  limits are passed to their store/provider queries, so those sources select a
  bounded window before response assembly. `snapshot_process_llm_call_limit`
  bounds the newest per-process LLM rows contributing to each process summary
  and is also the default and maximum size of the per-process LLM-call API.
  The snapshot selects those per-process windows in one SQL query and aggregates
  their count and token usage without materializing full call records.
- `gui.snapshot_collection_max_items` bounds process, pending-first Human,
  tool, image, Skill, JSON-RPC, MCP, Runtime Module, and LLM-profile reads at
  their source. A subsystem's own lower list maximum remains authoritative.
  Where the source permits the GUI collection limit plus one, the snapshot uses
  that lookahead, clips before assembly, and records the detected omission as a
  source-limited lower bound in `_truncated`. Process message/count, LLM-window
  count/token usage, rating, ancestor, reservation, and remaining-budget data
  are fetched in batches rather than once per pid.
- `gui.snapshot_string_max_chars` and the recursive collection bound still
  protect nested values during final response shaping. Final shaping is a
  second defense; it is no longer the only bound for top-level snapshot
  collections.
- `gui.event_buffer_limit` is the separate in-process SSE replay window; an
  evicted cursor causes explicit snapshot invalidation.

None of these windows deletes durable records. Source-level limits prevent
unbounded reads; final shaping limits protect the serialized payload but cannot
by themselves cap database or in-process assembly work.
