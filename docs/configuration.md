# Configuration Reference

Agent libOS keeps non-secret runtime defaults in the frozen, validated
`agent_libos.config.DEFAULT_CONFIG` object. The canonical field declarations
and numeric defaults live in `agent_libos/config/defaults.py`; this document
defines loading, precedence, security handling, and a field-level inventory so
operators do not have to infer those rules from example YAML.

## Loading and precedence

Product entrypoints use this order:

1. Start from the deeply immutable `DEFAULT_CONFIG` baseline.
2. If `--config <path>` is present, recursively merge that YAML mapping.
3. Otherwise load the repository/project-root `config.yaml` when it exists.
   The loader does not search the caller's current working directory.
4. Replace scalar, list, and tuple fields supplied by the overlay; recursively
   merge mapping fields such as `llm.profiles`.
5. For CLI and GUI-server entrypoints, an explicit `--db` store target overrides
   the selected runtime store target.
6. When an overlay is present, construct and validate a new frozen
   `AgentLibOSConfig`; without an overlay, the shared immutable baseline is
   safe to reuse. Unknown fields and unsafe, inverted, or non-finite bounds
   fail before the Runtime opens. Pydantic accepts its normal compatible
   scalar coercions for ordinary dataclass fields (for example a numeric YAML
   string may become a number); callers that generate overlays must not treat
   this loader as a strict JSON-type validator. Fields whose security contract
   uses strict numeric types, and the post-construction bound checks, still
   reject booleans and invalid values.

The frozen config object has no runtime hot reload. Change the host
configuration and open a new Runtime. The configured default LLM profile is a
narrow exception for its documented legacy environment mappings: each profile
resolution snapshots those values and changes invalidate the cached client.
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

The checked-in repository `config.yaml` selects `.agent_libos.sqlite` and loads
the trusted PTY Runtime Module. Consequently, omitting `--db` while using this
checkout selects that persistent store. In an installed package or source tree
without a project-root config, `DEFAULT_CONFIG.runtime.local_store_target` is
`local`, an in-memory SQLite store. Scripts and documentation that require state
across separate CLI invocations should always pass `--db` explicitly.

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
`llm.prompt_cache_retention`, `llm.responses_previous_response_id`,
`llm.parallel_tool_calls`, `llm.auto_wait_on_empty_tool_calls`,
`llm.temperature`, `llm.max_tokens`, and `llm.context_window_tokens` supply
their group defaults. The
legacy mappings are
`OPENAI_BASE_URL`; `OPENAI_LANGUAGE_MODEL` then `OPENAI_MODEL`;
`OPENAI_TIMEOUT`; `OPENAI_MAX_RETRIES`; `OPENAI_API_MODE`; `OPENAI_STORE`;
`OPENAI_REASONING_EFFORT`; `OPENAI_VERBOSITY`; `OPENAI_SAFETY_IDENTIFIER`;
`OPENAI_PROMPT_CACHE_KEY`; `OPENAI_PROMPT_CACHE_RETENTION`;
`OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID`; and
`OPENAI_PARALLEL_TOOL_CALLS`. The default profile also snapshots
`OPENAI_ENABLE_THINKING` and the OpenAI organization/project routing variables
(`OPENAI_ORGANIZATION`/`OPENAI_ORG_ID` and
`OPENAI_PROJECT`/`OPENAI_PROJECT_ID`). Named non-default profiles suppress the
OpenAI SDK's own ambient endpoint and organization/project fallbacks. Agent
libOS does not support `OPENAI_CUSTOM_HEADERS`, `OPENAI_ADMIN_KEY`, or
`OPENAI_WEBHOOK_SECRET` as profile settings; SDK-side values for those variables
are removed before any provider request rather than becoming untracked profile
state.

Secrets are the exception to that fallback description: every profile reads
its API key only from the environment variable named by its `api_key_env`.
`safety_identifier_env`, when set and no literal `safety_identifier` is set,
is authoritative: an unset or empty named variable produces no safety
identifier and does not fall back to the default profile's legacy identifier.
A custom base URL is permitted when either the profile sets
`allow_custom_base_url: true` or the Host sets
`AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1`.

## Inspecting exact defaults

Defaults change with the code. Print the exact values for the current checkout
instead of copying a stale sample:

```bash
uv run python -c 'import json; from agent_libos.config import DEFAULT_CONFIG; from agent_libos.utils.serde import to_jsonable; print(json.dumps(to_jsonable(DEFAULT_CONFIG), indent=2, sort_keys=True))'
```

The following inventory is intentionally field-level but leaves values in the
typed source and live dump above. A field addition must update this table in the
same change.

| Group | Fields |
| --- | --- |
| `runtime` | `local_store_target`, `runtime_db_filename`, `store_backend`, `store_dsn`, `workspace_namespace`, `default_image_id`, `coding_image_id`, `default_human`, `terminal_channel`, `run_until_idle_max_quanta`, `launcher_max_quanta`, `launch_authority_mode`, `publication_recovery_max_attempts`, `publication_reconciliation_page_size`, `publication_reconciliation_page_hard_limit`, `publication_artifact_lookup_hard_limit`, `resource_usage_reservation_recovery_page_size`, `resource_usage_reservation_recovery_page_hard_limit`, `capability_use_reservation_recovery_page_size`, `capability_use_reservation_recovery_page_hard_limit`, `object_payload_recovery_page_size`, `object_payload_recovery_page_hard_limit`, `object_task_recovery_page_size`, `object_task_recovery_page_hard_limit`, `jit_rehydration_page_size`, `jit_rehydration_page_hard_limit`, `external_effect_recovery_page_size`, `external_effect_recovery_page_hard_limit`, `operation_recovery_page_size`, `operation_recovery_page_hard_limit`, `payload_retention_enabled`, `payload_retention_summary_after_seconds`, `payload_retention_hash_only_after_seconds`, `payload_retention_page_size`, `payload_retention_page_hard_limit` |
| `gui` | `event_buffer_limit`, `request_body_max_bytes`, `scheduler_shutdown_join_timeout_s`, `http_shutdown_delay_s`, `object_task_wait_default_timeout_s`, `object_task_wait_max_timeout_s`, `snapshot_event_limit`, `snapshot_audit_limit`, `snapshot_llm_call_limit`, `snapshot_process_message_limit`, `snapshot_process_llm_call_limit`, `snapshot_object_task_limit`, `snapshot_collection_max_items`, `snapshot_string_max_chars`, `sse_payload_max_bytes`, `agent_rating_comment_max_chars` |
| `capability` | `default_delegation_depth`, `max_rights_per_capability`, `max_constraints_bytes`, `list_limit`, `decision_explain_preview_chars` |
| `data_flow` | `default_trust_level`, `default_max_sensitivity`, `sink_rules`, `registry_resource`, `registry_list_limit`, `decision_list_limit`, `file_binding_list_limit` |
| `scheduler` | `max_quanta`, `poll_interval_s`, `max_workers`, `drain_window_s`, `shutdown_join_timeout_s` |
| `process` | `max_tool_calls`, `max_child_processes`, `max_runtime_seconds`, `max_context_materialization_tokens`, `max_context_materialization_total_tokens`, `max_llm_calls`, `max_llm_total_tokens`, `max_subprocess_wall_seconds`, `max_subprocess_cpu_seconds`, `max_subprocess_memory_bytes`, `max_external_read_bytes`, `max_external_write_bytes`, `max_jsonrpc_bytes`, `max_mcp_bytes`, `max_deno_syscalls`, `default_goal_text`, `default_working_directory`, `fork_budget_divisor`, `fork_min_tool_calls`, `fork_min_child_processes` |
| `llm` | `default_profile_id`, `profiles`, `temperature`, `max_tokens`, `context_window_tokens`, `timeout_s`, `max_retries`, `api_mode`, `store`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention`, `responses_previous_response_id`, `parallel_tool_calls`, `auto_wait_on_empty_tool_calls`, `compatibility_retry_attempts`, `action_repair_attempts`, `content_preview_chars`, `tool_arguments_preview_chars`, `call_record_preview_chars`, `call_record_list_limit`, `call_record_hard_limit`, `persist_full_io`, `json_instruction`, `fallback_status_codes` |
| `tools` | `version`, `default_timeout_s`, `standard_timeout_s`, `interactive_timeout_s`, `default_text_encoding`, `tool_observability_preview_chars`, `tool_call_args_hard_limit_bytes`, `tool_result_payload_hard_limit_bytes`, `filesystem_read_max_bytes`, `filesystem_read_hard_limit_bytes`, `directory_entry_limit`, `directory_entry_hard_limit`, `executable_snapshot_sibling_limit`, `memory_payload_chars`, `memory_payload_hard_limit_chars`, `memory_payload_hard_limit_bytes`, `memory_append_entry_max_bytes`, `message_subject_max_chars`, `message_body_max_chars`, `message_payload_max_bytes`, `message_id_max_chars`, `message_read_limit`, `message_read_hard_limit`, `message_filter_ids_hard_limit`, `message_filter_json_max_bytes`, `message_wait_status_max_chars`, `human_request_payload_max_bytes`, `human_output_max_chars`, `human_request_list_limit`, `object_file_max_bytes`, `object_file_hard_limit_bytes`, `shell_timeout_s`, `sandbox_timeout_s`, `jit_source_max_chars`, `jit_tests_max_count`, `jit_test_case_max_bytes`, `jit_validation_timeout_s`, `jit_validation_log_max_chars`, `deno_executable`, `deno_timeout_s`, `deno_timeout_hard_limit_s`, `deno_max_rpc_calls`, `deno_max_stdout_bytes`, `deno_max_stderr_bytes`, `deno_jsr_allowlist`, `static_tool_id_digest_chars`, `approval_preview_chars`, `clock_timezone`, `max_sleep_seconds`, `sleep_timeout_grace_s` |
| `shell` | `policy_capability_key`, `policy_resource`, `default_policy_level`, `timeout_hard_limit_s`, `max_stdout_chars`, `max_stderr_chars`, `stdout_hard_limit_chars`, `stderr_hard_limit_chars`, `rules`, `whitelist`, `blacklist` |
| `git` | `enabled`, `executable`, `minimum_version`, `repository_resource`, `worktree_root`, `trusted_metadata_roots`, `local_timeout_s`, `remote_timeout_s`, `timeout_hard_limit_s`, `lock_timeout_s`, `status_entry_limit`, `status_entry_hard_limit`, `log_entry_limit`, `log_entry_hard_limit`, `output_max_bytes`, `output_hard_limit_bytes`, `patch_max_bytes`, `patch_hard_limit_bytes`, `state_content_hard_limit_bytes`, `allowed_remote_schemes`, `allow_scp_style_ssh`, `allow_file_remotes`, `inherit_credential_helpers`, `inherit_ssh_agent`, `protect_git_metadata` |
| `jsonrpc` | `registry_resource`, `endpoint_id_max_chars`, `method_id_max_chars`, `rpc_method_max_chars`, `header_name_max_chars`, `header_value_max_chars`, `manifest_max_bytes`, `timeout_s`, `timeout_hard_limit_s`, `max_request_bytes`, `max_response_bytes`, `max_request_hard_limit_bytes`, `max_response_hard_limit_bytes`, `list_limit`, `audit_preview_chars`, `header_env_allowlist` |
| `mcp` | `registry_resource`, `server_id_max_chars`, `tool_id_max_chars`, `mcp_name_max_chars`, `header_name_max_chars`, `header_value_max_chars`, `manifest_max_bytes`, `timeout_s`, `timeout_hard_limit_s`, `max_request_bytes`, `max_response_bytes`, `max_request_hard_limit_bytes`, `max_response_hard_limit_bytes`, `list_limit`, `audit_preview_chars`, `header_env_allowlist`, `stdio_env_allowlist` |
| `image` | `registry_resource`, `id_max_chars`, `name_max_chars`, `version_max_chars`, `manifest_hard_limit_bytes`, `structured_field_hard_limit_bytes`, `max_default_tools`, `max_required_capabilities`, `max_required_modules`, `package_manifest_name`, `package_workspace_dir`, `package_tools_dir`, `package_resources_dir`, `materialized_workspace_root`, `package_manifest_max_bytes`, `package_manifest_hard_limit_bytes`, `package_file_max_bytes`, `package_max_bytes`, `package_max_files`, `prompt_max_chars`, `max_package_jit_tools`, `max_workspace_grants` |
| `image_commit` | `artifact_version`, `artifact_hard_limit_bytes`, `payload_capture_limit_bytes`, `max_required_capabilities`, `max_committed_tools`, `max_committed_jit_sources`, `metadata_preview_chars` |
| `memory` | `object_schema_version`, `materialize_budget_tokens`, `query_limit`, `context_policy`, `metadata_sensitivity`, `metadata_retention_policy`, `process_namespace_prefix` |
| `object_tasks` | `max_running_global`, `max_running_per_object`, `notification_channel`, `owner_watch_channel`, `owner_watch_events`, `shutdown_join_timeout_s` |
| `llm_context` | `policy`, `schema_version`, `object_name_prefix`, `recent_event_limit` |
| `checkpoint` | `list_limit`, `payload_capture_limit_bytes`, `snapshot_hard_limit_bytes`, `diff_preview_items` |
| `skills` | `schema_version`, `registry_resource`, `trust_resource`, `global_dirs`, `workspace_dirs`, `resource_dirs`, `trusted_global_package_sha256`, `global_requires_trust`, `skill_md_max_bytes`, `skill_md_hard_limit_bytes`, `resource_read_max_bytes`, `package_max_bytes`, `max_package_files`, `max_prompt_instruction_chars`, `max_jit_source_chars`, `discover_limit`, `id_max_chars`, `name_max_chars`, `description_max_chars`, `version_max_chars`, `max_tools`, `max_actions`, `max_jit_tools`, `max_required_capabilities` |
| `modules` | `schema_version`, `manifest_paths`, `trusted_modules`, `trusted_sha256`, `manifest_max_bytes`, `manifest_hard_limit_bytes`, `source_max_bytes`, `package_max_bytes`, `max_package_files`, `load_policy`, `discover_limit`, `id_max_chars`, `name_max_chars`, `version_max_chars`, `entrypoint_max_chars`, `max_declared_tools`, `max_declared_images`, `max_declared_syscalls`, `max_declared_provider_hooks`, `max_declared_startup_hooks` |
| `launcher` | `permission_presets`, `default_permission_preset`, `read_only_preset`, `edit_preset`, `full_preset` |
| `scripts` | `ask_file_max_bytes`, `ask_file_max_quanta`, `document_summary_max_bytes`, `document_summary_max_read_bytes`, `document_summary_max_quanta`, `document_context_min_tokens`, `document_context_slack_tokens`, `document_context_max_tokens`, `object_copy_max_quanta`, `llm_write_smoke_max_quanta`, `clock_demo_iterations`, `clock_demo_interval_s`, `clock_demo_timezone`, `chat_max_turns`, `chat_context_tokens`, `chat_quanta_per_turn`, `chat_quanta_overhead` |

The table is checked against the dataclass fields by
`tests/unit/test_configuration_docs.py`.

Each entry under `llm.profiles.<profile_id>` accepts exactly these fields:

| Profile | Fields |
| --- | --- |
| `llm.profiles.<profile_id>` | `kind`, `base_url`, `model`, `api_key_env`, `api_mode`, `timeout_s`, `max_retries`, `store`, `reasoning_effort`, `verbosity`, `safety_identifier`, `safety_identifier_env`, `prompt_cache_key`, `prompt_cache_retention`, `responses_previous_response_id`, `parallel_tool_calls`, `auto_wait_on_empty_tool_calls`, `temperature`, `max_tokens`, `context_window_tokens`, `allow_custom_base_url` |

The same test checks this nested inventory. Exact values remain authoritative
in the live dump and typed source. Optional Runtime
Modules may also own module-local settings that are not fields of
`AgentLibOSConfig`.

Checkpoint snapshot format versions are owned by the runtime codec and are not
configurable. A runtime release emits only the snapshot version it can decode.

## Security-sensitive settings

- `runtime.store_dsn` may contain PostgreSQL credentials. Prefer an
  environment-specific untracked overlay; never commit a real DSN.
- LLM profiles store an `api_key_env` variable name, never the API-key value.
  Only the selected host process reads the named environment variable.
- `llm.context_window_tokens` defaults to `131072`; a profile may override it.
  Effective `max_tokens` must be smaller than the effective model window. The
  window controls local pressure management only and is deliberately excluded
  from the Provider/Sink identity hash.
- JSON-RPC/MCP header and stdio allowlists contain environment-variable names.
  Manifests reference those names; resolved secret values must not be persisted
  in registry rows, audit metadata, benchmark provenance, or GUI responses.
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
  reasoning, outputs, provider errors, and raw provider payloads. The opt-out
  persists canonical content-free summary envelopes containing byte counts,
  JSON shape/count metadata when available, and hashes rather than readable
  previews. It also redacts
  conditional LLM release resume rows before approval; exact same-runtime
  approval remains supported, while reopen fails that unrecoverable release
  closed instead of rebuilding or dispatching it.
- Provider-side Responses storage and chaining remain opt-in through
  `llm.store` and `llm.responses_previous_response_id`.
- `runtime.launch_authority_mode: manifest_required` treats image capability
  requirements as declarations, not grants; this value is fixed in 0.3.
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
  only a page-bounded ID sample plus the exact total count. Payload retention is separately opt-in through
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
- `data_flow.registry_resource` is the `admin` capability resource for runtime
  registry mutations. `registry_list_limit`, `decision_list_limit`, and
  `file_binding_list_limit` bound active control-plane reads; they do not
  truncate append-only decision or binding history in storage.
- `tools.executable_snapshot_sibling_limit` bounds the direct sibling entries
  linked beside a mutable workspace executable snapshot before Shell, MCP
  stdio, or PTY dispatch. Exceeding the limit, failing enumeration, or failing
  to link any required sibling aborts the dispatch instead of exposing a
  partial snapshot.

## Bounded windows

GUI and context limits operate at two different layers and must not be treated
as equivalent:

- `llm_context.recent_event_limit` bounds the newest post-cursor event rows
  loaded from SQL for one LLM context preparation.
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
