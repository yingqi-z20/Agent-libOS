from __future__ import annotations

from dataclasses import field
import math
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, StrictInt
from pydantic.dataclasses import dataclass

from agent_libos.models.capability import AuthorityRule
from agent_libos.models.data_flow import (
    DataSensitivity,
    SinkTrustLevel,
    SinkTrustRule,
    sensitivity_rank,
)

_PYDANTIC_CONFIG = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

ShellPolicyLevel = Literal[
    "always_deny",
    "allowlist_auto_else_ask",
    "blocklist_ask_else_auto",
    "always_allow",
]


def _validate_runtime_page_bounds(
    name: str,
    *,
    page_size: int,
    hard_limit: int,
) -> None:
    if page_size <= 0:
        raise ValueError(f"runtime.{name}_page_size must be greater than zero")
    if hard_limit <= 0:
        raise ValueError(f"runtime.{name}_page_hard_limit must be greater than zero")
    if page_size > hard_limit:
        raise ValueError(
            f"runtime.{name}_page_size must not exceed "
            f"runtime.{name}_page_hard_limit"
        )


class _ImmutableDict(dict):
    """Small immutable dict used by frozen configuration value objects."""

    @staticmethod
    def _reject_mutation(*args: object, **kwargs: object) -> None:
        raise TypeError("configuration mappings are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __copy__(self) -> "_ImmutableDict":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "_ImmutableDict":
        return self


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ShellCommandRule:
    argv: tuple[str, ...]
    match: Literal["exact", "prefix"] = "exact"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("shell command rule argv must include an executable")
        if not self.argv[0].strip():
            raise ValueError("shell command rule argv[0] must be a non-empty executable")
        if any("\x00" in token for token in self.argv):
            raise ValueError("shell command rule argv must not contain NUL bytes")


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class RuntimeDefaults:
    local_store_target: str = "local"
    runtime_db_filename: str = ".agent_libos.sqlite"
    store_backend: Literal["sqlite", "postgres"] = "sqlite"
    store_dsn: str | None = None
    workspace_namespace: str = "workspace"
    default_image_id: str = "base-agent:v0"
    coding_image_id: str = "coding-agent:v0"
    default_human: str = "owner"
    terminal_channel: str = "terminal"
    run_until_idle_max_quanta: int | None = None
    launcher_max_quanta: int = 40
    launch_authority_mode: Literal["manifest_required"] = "manifest_required"
    publication_recovery_max_attempts: StrictInt = 3
    publication_reconciliation_page_size: StrictInt = 500
    publication_reconciliation_page_hard_limit: StrictInt = 5_000
    publication_artifact_lookup_hard_limit: StrictInt = 5_000
    resource_usage_reservation_recovery_page_size: StrictInt = 500
    resource_usage_reservation_recovery_page_hard_limit: StrictInt = 5_000
    capability_use_reservation_recovery_page_size: StrictInt = 500
    capability_use_reservation_recovery_page_hard_limit: StrictInt = 5_000
    object_payload_recovery_page_size: StrictInt = 500
    object_payload_recovery_page_hard_limit: StrictInt = 5_000
    object_task_recovery_page_size: StrictInt = 500
    object_task_recovery_page_hard_limit: StrictInt = 5_000
    jit_rehydration_page_size: StrictInt = 500
    jit_rehydration_page_hard_limit: StrictInt = 5_000
    external_effect_recovery_page_size: StrictInt = 500
    external_effect_recovery_page_hard_limit: StrictInt = 5_000
    operation_recovery_page_size: StrictInt = 500
    operation_recovery_page_hard_limit: StrictInt = 5_000
    payload_retention_enabled: bool = False
    payload_retention_summary_after_seconds: StrictInt | None = None
    payload_retention_hash_only_after_seconds: StrictInt | None = None
    payload_retention_page_size: StrictInt = 100
    payload_retention_page_hard_limit: StrictInt = 1_000

    def __post_init__(self) -> None:
        if self.publication_recovery_max_attempts <= 0:
            raise ValueError(
                "runtime.publication_recovery_max_attempts must be greater than zero"
            )
        _validate_runtime_page_bounds(
            "publication_reconciliation",
            page_size=self.publication_reconciliation_page_size,
            hard_limit=self.publication_reconciliation_page_hard_limit,
        )
        if self.publication_artifact_lookup_hard_limit <= 0:
            raise ValueError(
                "runtime.publication_artifact_lookup_hard_limit must be greater than zero"
            )
        _validate_runtime_page_bounds(
            "resource_usage_reservation_recovery",
            page_size=self.resource_usage_reservation_recovery_page_size,
            hard_limit=self.resource_usage_reservation_recovery_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "capability_use_reservation_recovery",
            page_size=self.capability_use_reservation_recovery_page_size,
            hard_limit=self.capability_use_reservation_recovery_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "object_payload_recovery",
            page_size=self.object_payload_recovery_page_size,
            hard_limit=self.object_payload_recovery_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "object_task_recovery",
            page_size=self.object_task_recovery_page_size,
            hard_limit=self.object_task_recovery_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "jit_rehydration",
            page_size=self.jit_rehydration_page_size,
            hard_limit=self.jit_rehydration_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "external_effect_recovery",
            page_size=self.external_effect_recovery_page_size,
            hard_limit=self.external_effect_recovery_page_hard_limit,
        )
        _validate_runtime_page_bounds(
            "operation_recovery",
            page_size=self.operation_recovery_page_size,
            hard_limit=self.operation_recovery_page_hard_limit,
        )
        for name in (
            "payload_retention_summary_after_seconds",
            "payload_retention_hash_only_after_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(
                    f"runtime.{name} must be a non-negative integer or None"
                )
        if (
            self.payload_retention_enabled
            and self.payload_retention_summary_after_seconds is None
        ):
            raise ValueError(
                "runtime.payload_retention_enabled requires "
                "payload_retention_summary_after_seconds"
            )
        if (
            self.payload_retention_hash_only_after_seconds is not None
            and self.payload_retention_summary_after_seconds is None
        ):
            raise ValueError(
                "runtime.payload_retention_hash_only_after_seconds requires "
                "payload_retention_summary_after_seconds"
            )
        if (
            self.payload_retention_hash_only_after_seconds is not None
            and self.payload_retention_summary_after_seconds is not None
            and self.payload_retention_hash_only_after_seconds
            < self.payload_retention_summary_after_seconds
        ):
            raise ValueError(
                "runtime.payload_retention_hash_only_after_seconds must be at least "
                "payload_retention_summary_after_seconds"
            )
        if self.payload_retention_page_size <= 0:
            raise ValueError(
                "runtime.payload_retention_page_size must be greater than zero"
            )
        if self.payload_retention_page_hard_limit <= 0:
            raise ValueError(
                "runtime.payload_retention_page_hard_limit must be greater than zero"
            )
        if self.payload_retention_page_size > self.payload_retention_page_hard_limit:
            raise ValueError(
                "runtime.payload_retention_page_size must not exceed "
                "runtime.payload_retention_page_hard_limit"
            )

    @property
    def default_human_resource(self) -> str:
        return f"human:{self.default_human}"

    @property
    def default_human_actor(self) -> str:
        return f"human:{self.default_human}"


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class GuiDefaults:
    """Local desktop console defaults.

    The GUI server is a localhost-only host control surface, but it still needs
    bounded buffers and request sizes so a renderer bug cannot exhaust the
    Python runtime process.
    """

    event_buffer_limit: int = 1_000
    request_body_max_bytes: int = 25_165_824
    scheduler_shutdown_join_timeout_s: float = 2.0
    http_shutdown_delay_s: float = 0.2
    object_task_wait_default_timeout_s: float = 30.0
    object_task_wait_max_timeout_s: float = 300.0
    snapshot_event_limit: int = 200
    snapshot_audit_limit: int = 200
    snapshot_llm_call_limit: int = 100
    snapshot_process_message_limit: int = 100
    snapshot_process_llm_call_limit: int = 20
    snapshot_object_task_limit: int = 100
    snapshot_collection_max_items: int = 200
    snapshot_string_max_chars: int = 8_192
    sse_payload_max_bytes: int = 262_144
    agent_rating_comment_max_chars: int = 2_000


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class CapabilityDefaults:
    default_delegation_depth: int = 4
    max_rights_per_capability: int = 16
    max_constraints_bytes: int = 16_384
    list_limit: int = 100
    decision_explain_preview_chars: int = 2_000


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class DataFlowDefaults:
    """Host-owned defaults for runtime-mediated data egress.

    Unmatched sinks are intentionally fixed to untrusted/normal. Additional
    clearance must be attached to a concrete or trailing-wildcard sink rule.
    """

    default_trust_level: SinkTrustLevel = SinkTrustLevel.UNTRUSTED
    default_max_sensitivity: DataSensitivity = DataSensitivity.NORMAL
    sink_rules: tuple[SinkTrustRule, ...] = ()
    registry_resource: str = "data_flow_sink_registry:*"
    registry_list_limit: int = 100
    decision_list_limit: int = 1_000
    file_binding_list_limit: int = 1_000


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class SchedulerDefaults:
    max_quanta: int | None = None
    poll_interval_s: float = 0.01
    max_workers: int = 8
    drain_window_s: float = 0.5
    shutdown_join_timeout_s: float = 2.0


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ProcessDefaults:
    max_tool_calls: int = 256
    max_child_processes: int = 16
    max_runtime_seconds: float | None = None
    max_context_materialization_tokens: int = 65_536
    max_context_materialization_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_llm_total_tokens: int | None = None
    max_subprocess_wall_seconds: float | None = None
    max_subprocess_cpu_seconds: float | None = None
    max_subprocess_memory_bytes: int | None = None
    max_external_read_bytes: int | None = None
    max_external_write_bytes: int | None = None
    max_jsonrpc_bytes: int | None = None
    max_mcp_bytes: int | None = None
    max_deno_syscalls: int | None = None
    default_goal_text: str = "Run agent process"
    default_working_directory: str = "."
    fork_budget_divisor: int = 2
    fork_min_tool_calls: int = 1
    fork_min_child_processes: int = 0


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class LLMProfile:
    kind: Literal["openai_compatible"] = "openai_compatible"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_mode: Literal["auto", "responses", "chat"] | None = None
    timeout_s: float | None = None
    max_retries: int | None = None
    store: bool | None = None
    reasoning_effort: str | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    safety_identifier: str | None = None
    safety_identifier_env: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    responses_previous_response_id: bool | None = None
    parallel_tool_calls: bool | None = None
    auto_wait_on_empty_tool_calls: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_window_tokens: int | None = None
    allow_custom_base_url: bool = False


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class LLMDefaults:
    default_profile_id: str = "default"
    profiles: dict[str, LLMProfile] = field(default_factory=lambda: {"default": LLMProfile()})
    temperature: float = 0.2
    max_tokens: int = 16_384
    context_window_tokens: int = 131_072
    timeout_s: float = 60.0
    max_retries: int = 2
    api_mode: Literal["auto", "responses", "chat"] = "auto"
    store: bool = False
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    responses_previous_response_id: bool = False
    parallel_tool_calls: bool = False
    auto_wait_on_empty_tool_calls: bool = False
    compatibility_retry_attempts: int = 8
    action_repair_attempts: int = 2
    content_preview_chars: int = 500
    tool_arguments_preview_chars: int = 500
    call_record_preview_chars: int = 1_000
    call_record_list_limit: int = 100
    call_record_hard_limit: int = 1_000
    persist_full_io: bool = True
    json_instruction: str = "You must respond with a valid JSON object."
    fallback_status_codes: tuple[int, ...] = (404, 405)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", _ImmutableDict(self.profiles))


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ToolDefaults:
    version: str = "1.0.0"
    default_timeout_s: float = 30.0
    standard_timeout_s: float = 5.0
    interactive_timeout_s: float = 2.0
    default_text_encoding: str = "utf-8"
    tool_observability_preview_chars: int = 256
    tool_call_args_hard_limit_bytes: int = 1_500_000
    tool_result_payload_hard_limit_bytes: int = 1_500_000
    filesystem_read_max_bytes: int = 65_536
    filesystem_read_hard_limit_bytes: int = 1_048_576
    directory_entry_limit: int = 1_024
    directory_entry_hard_limit: int = 10_000
    executable_snapshot_sibling_limit: int = 1_024
    memory_payload_chars: int = 12_000
    memory_payload_hard_limit_chars: int = 200_000
    memory_payload_hard_limit_bytes: int = 200_000
    memory_append_entry_max_bytes: int = 32_768
    message_subject_max_chars: int = 512
    message_body_max_chars: int = 32_000
    message_payload_max_bytes: int = 131_072
    message_id_max_chars: int = 128
    message_read_limit: int = 100
    message_read_hard_limit: int = 1_000
    message_filter_ids_hard_limit: int = 1_000
    message_filter_json_max_bytes: int = 16_384
    message_wait_status_max_chars: int = 32_768
    human_request_payload_max_bytes: int = 131_072
    human_output_max_chars: int = 32_000
    human_request_list_limit: int = 1_000
    object_file_max_bytes: int = 1_048_576
    object_file_hard_limit_bytes: int = 10_485_760
    shell_timeout_s: float = 30.0
    sandbox_timeout_s: float = 5.0
    jit_source_max_chars: int = 65_536
    jit_tests_max_count: int = 32
    jit_test_case_max_bytes: int = 32_768
    jit_validation_timeout_s: float = 5.0
    jit_validation_log_max_chars: int = 131_072
    deno_executable: str = "deno"
    deno_timeout_s: float = 5.0
    deno_timeout_hard_limit_s: float = 60.0
    deno_max_rpc_calls: int = 64
    deno_max_stdout_bytes: int = 1_000_000
    deno_max_stderr_bytes: int = 100_000
    deno_jsr_allowlist: tuple[str, ...] = (
        "@std/assert",
        "@std/collections",
        "@std/encoding",
        "@std/path",
        "@std/yaml",
    )
    static_tool_id_digest_chars: int = 16
    approval_preview_chars: int = 256
    clock_timezone: str = "UTC"
    max_sleep_seconds: float = 60.0
    sleep_timeout_grace_s: float = 5.0

    @property
    def sleep_tool_timeout_s(self) -> float:
        return self.max_sleep_seconds + self.sleep_timeout_grace_s


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ShellDefaults:
    policy_capability_key: str = "shell_policy_level"
    policy_resource: str = "shell:*"
    default_policy_level: ShellPolicyLevel = "allowlist_auto_else_ask"
    timeout_hard_limit_s: float = 300.0
    max_stdout_chars: int = 32_000
    max_stderr_chars: int = 32_000
    stdout_hard_limit_chars: int = 200_000
    stderr_hard_limit_chars: int = 200_000
    rules: tuple[AuthorityRule, ...] = ()
    whitelist: tuple[ShellCommandRule, ...] = (
        ShellCommandRule(("git", "status")),
        ShellCommandRule(("git", "status", "--short")),
        ShellCommandRule(("git", "branch", "--show-current")),
        ShellCommandRule(("git", "rev-parse", "--show-toplevel")),
        ShellCommandRule(("git", "diff", "--stat")),
        ShellCommandRule(("python", "--version")),
        ShellCommandRule(("python3", "--version")),
        ShellCommandRule(("uv", "--version")),
    )
    blacklist: tuple[ShellCommandRule, ...] = (
        ShellCommandRule(("cmd",), match="prefix"),
        ShellCommandRule(("powershell",), match="prefix"),
        ShellCommandRule(("pwsh",), match="prefix"),
        ShellCommandRule(("sh",), match="prefix"),
        ShellCommandRule(("bash",), match="prefix"),
        ShellCommandRule(("zsh",), match="prefix"),
        ShellCommandRule(("fish",), match="prefix"),
        ShellCommandRule(("python",), match="prefix"),
        ShellCommandRule(("python3",), match="prefix"),
        ShellCommandRule(("py",), match="prefix"),
        ShellCommandRule(("node",), match="prefix"),
        ShellCommandRule(("npm",), match="prefix"),
        ShellCommandRule(("npx",), match="prefix"),
        ShellCommandRule(("yarn",), match="prefix"),
        ShellCommandRule(("pnpm",), match="prefix"),
        ShellCommandRule(("uv",), match="prefix"),
        ShellCommandRule(("pip",), match="prefix"),
        ShellCommandRule(("pip3",), match="prefix"),
        ShellCommandRule(("ruby",), match="prefix"),
        ShellCommandRule(("perl",), match="prefix"),
        ShellCommandRule(("php",), match="prefix"),
        ShellCommandRule(("java",), match="prefix"),
        ShellCommandRule(("cargo",), match="prefix"),
        ShellCommandRule(("docker",), match="prefix"),
        ShellCommandRule(("kubectl",), match="prefix"),
        ShellCommandRule(("ssh",), match="prefix"),
        ShellCommandRule(("scp",), match="prefix"),
        ShellCommandRule(("sftp",), match="prefix"),
        ShellCommandRule(("curl",), match="prefix"),
        ShellCommandRule(("wget",), match="prefix"),
        ShellCommandRule(("nc",), match="prefix"),
        ShellCommandRule(("ncat",), match="prefix"),
        ShellCommandRule(("netcat",), match="prefix"),
        ShellCommandRule(("rm",), match="prefix"),
        ShellCommandRule(("del",), match="prefix"),
        ShellCommandRule(("rmdir",), match="prefix"),
        ShellCommandRule(("remove-item",), match="prefix"),
        ShellCommandRule(("move-item",), match="prefix"),
        ShellCommandRule(("copy-item",), match="prefix"),
        ShellCommandRule(("chmod",), match="prefix"),
        ShellCommandRule(("chown",), match="prefix"),
        ShellCommandRule(("icacls",), match="prefix"),
        ShellCommandRule(("reg",), match="prefix"),
        ShellCommandRule(("regedit",), match="prefix"),
        ShellCommandRule(("taskkill",), match="prefix"),
        ShellCommandRule(("sudo",), match="prefix"),
        ShellCommandRule(("su",), match="prefix"),
        ShellCommandRule(("runas",), match="prefix"),
    )

    @property
    def always_deny_level(self) -> ShellPolicyLevel:
        return "always_deny"

    @property
    def allowlist_auto_else_ask_level(self) -> ShellPolicyLevel:
        return "allowlist_auto_else_ask"

    @property
    def blocklist_ask_else_auto_level(self) -> ShellPolicyLevel:
        return "blocklist_ask_else_auto"

    @property
    def always_allow_level(self) -> ShellPolicyLevel:
        return "always_allow"


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class GitDefaults:
    enabled: bool = True
    executable: str = "git"
    minimum_version: str = "2.26.0"
    repository_resource: str = "git:workspace"
    worktree_root: str = "agent_outputs/git_worktrees"
    trusted_metadata_roots: tuple[str, ...] = ()
    local_timeout_s: float = 30.0
    remote_timeout_s: float = 120.0
    timeout_hard_limit_s: float = 300.0
    lock_timeout_s: float = 10.0
    status_entry_limit: int = 2_000
    status_entry_hard_limit: int = 20_000
    log_entry_limit: int = 50
    log_entry_hard_limit: int = 1_000
    output_max_bytes: int = 262_144
    output_hard_limit_bytes: int = 8_388_608
    patch_max_bytes: int = 1_048_576
    patch_hard_limit_bytes: int = 8_388_608
    state_content_hard_limit_bytes: int = 67_108_864
    allowed_remote_schemes: tuple[str, ...] = ("https", "ssh")
    allow_scp_style_ssh: bool = True
    allow_file_remotes: bool = False
    inherit_credential_helpers: bool = True
    inherit_ssh_agent: bool = True
    protect_git_metadata: bool = True


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class JsonRpcDefaults:
    registry_resource: str = "jsonrpc_endpoint:*"
    endpoint_id_max_chars: int = 96
    method_id_max_chars: int = 96
    rpc_method_max_chars: int = 256
    header_name_max_chars: int = 128
    header_value_max_chars: int = 8_192
    manifest_max_bytes: int = 262_144
    timeout_s: float = 10.0
    timeout_hard_limit_s: float = 60.0
    max_request_bytes: int = 65_536
    max_response_bytes: int = 1_048_576
    max_request_hard_limit_bytes: int = 1_048_576
    max_response_hard_limit_bytes: int = 8_388_608
    list_limit: int = 100
    audit_preview_chars: int = 512
    header_env_allowlist: tuple[str, ...] = ("AGENT_LIBOS_JSONRPC_*",)


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class McpDefaults:
    registry_resource: str = "mcp_server:*"
    server_id_max_chars: int = 96
    tool_id_max_chars: int = 96
    mcp_name_max_chars: int = 256
    header_name_max_chars: int = 128
    header_value_max_chars: int = 8_192
    manifest_max_bytes: int = 262_144
    timeout_s: float = 10.0
    timeout_hard_limit_s: float = 60.0
    max_request_bytes: int = 65_536
    max_response_bytes: int = 1_048_576
    max_request_hard_limit_bytes: int = 1_048_576
    max_response_hard_limit_bytes: int = 8_388_608
    list_limit: int = 100
    audit_preview_chars: int = 512
    header_env_allowlist: tuple[str, ...] = ("AGENT_LIBOS_MCP_*",)
    stdio_env_allowlist: tuple[str, ...] = ("AGENT_LIBOS_MCP_*",)


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ImageDefaults:
    registry_resource: str = "image:*"
    id_max_chars: int = 128
    name_max_chars: int = 128
    version_max_chars: int = 64
    manifest_hard_limit_bytes: int = 1_048_576
    structured_field_hard_limit_bytes: int = 262_144
    max_default_tools: int = 128
    max_required_capabilities: int = 64
    max_required_modules: int = 64
    package_manifest_name: str = "IMAGE.yaml"
    package_workspace_dir: str = "workspace"
    package_tools_dir: str = "tools"
    package_resources_dir: str = "resources"
    materialized_workspace_root: str = "agent_outputs/image_workspaces"
    package_manifest_max_bytes: int = 262_144
    package_manifest_hard_limit_bytes: int = 1_048_576
    package_file_max_bytes: int = 1_048_576
    package_max_bytes: int = 16_777_216
    package_max_files: int = 512
    prompt_max_chars: int = 64_000
    max_package_jit_tools: int = 64
    max_workspace_grants: int = 64


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ImageCommitDefaults:
    artifact_version: int = 2
    artifact_hard_limit_bytes: int = 16_777_216
    payload_capture_limit_bytes: int = 1_048_576
    max_required_capabilities: int = 128
    max_committed_tools: int = 256
    max_committed_jit_sources: int = 64
    metadata_preview_chars: int = 512


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ObjectMemoryDefaults:
    object_schema_version: str = "1"
    materialize_budget_tokens: int = 8_000
    query_limit: int = 50
    context_policy: str = "plan_first"
    metadata_sensitivity: str = "normal"
    metadata_retention_policy: str = "default"
    process_namespace_prefix: str = "process"


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ObjectTaskDefaults:
    max_running_global: int = 16
    max_running_per_object: int = 4
    notification_channel: str = "object-task"
    owner_watch_channel: str = "object-task-owner"
    owner_watch_events: tuple[str, ...] = ("updated", "linked")
    shutdown_join_timeout_s: float = 2.0


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class LLMContextDefaults:
    policy: str = "source_only"
    schema_version: int = 1
    object_name_prefix: str = "llm_context"
    recent_event_limit: int = 20
    storage_compaction_threshold_bytes: int = 96_000
    storage_compaction_max_chunks: int = 4
    storage_compaction_preserve_recent_entries: int = 0


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class CheckpointDefaults:
    list_limit: int = 100
    payload_capture_limit_bytes: int = 1_048_576
    snapshot_hard_limit_bytes: int = 16_777_216
    diff_preview_items: int = 25


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class SkillDefaults:
    schema_version: int = 1
    registry_resource: str = "skill:*"
    trust_resource: str = "skill_trust:*"
    global_dirs: tuple[str, ...] = ("~/.agent-libos/skills",)
    # Keep the standard Agent Skills compatibility roots in the configured
    # default rather than adding them implicitly in discovery.  This makes a
    # host override authoritative while preserving the historical defaults.
    workspace_dirs: tuple[str, ...] = (
        "skills",
        ".agent_libos/skills",
        ".agents/skills",
        ".claude/skills",
    )
    resource_dirs: tuple[str, ...] = ("scripts", "references", "assets")
    trusted_global_package_sha256: tuple[str, ...] = ()
    global_requires_trust: bool = True
    skill_md_max_bytes: int = 262_144
    skill_md_hard_limit_bytes: int = 1_048_576
    resource_read_max_bytes: int = 262_144
    package_max_bytes: int = 2_097_152
    max_package_files: int = 256
    max_prompt_instruction_chars: int = 8_000
    max_jit_source_chars: int = 64_000
    discover_limit: int = 100
    id_max_chars: int = 128
    name_max_chars: int = 128
    description_max_chars: int = 1_024
    version_max_chars: int = 64
    max_tools: int = 128
    max_actions: int = 128
    max_jit_tools: int = 32
    max_required_capabilities: int = 64

    @property
    def manifest_max_bytes(self) -> int:
        return self.skill_md_max_bytes

    @property
    def manifest_hard_limit_bytes(self) -> int:
        return self.skill_md_hard_limit_bytes


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ModuleDefaults:
    schema_version: int = 1
    manifest_paths: tuple[str, ...] = ()
    trusted_modules: tuple[str, ...] = ()
    trusted_sha256: tuple[str, ...] = ()
    manifest_max_bytes: int = 262_144
    manifest_hard_limit_bytes: int = 1_048_576
    source_max_bytes: int = 1_048_576
    package_max_bytes: int = 8_388_608
    max_package_files: int = 256
    load_policy: Literal["fail", "warn"] = "fail"
    discover_limit: int = 100
    id_max_chars: int = 128
    name_max_chars: int = 128
    version_max_chars: int = 64
    entrypoint_max_chars: int = 512
    max_declared_tools: int = 128
    max_declared_images: int = 128
    max_declared_syscalls: int = 128
    max_declared_provider_hooks: int = 64
    max_declared_startup_hooks: int = 64


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class LauncherDefaults:
    permission_presets: tuple[str, ...] = ("read-only", "edit", "full")
    default_permission_preset: str = "edit"
    read_only_preset: str = "read-only"
    edit_preset: str = "edit"
    full_preset: str = "full"


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class ScriptDefaults:
    ask_file_max_bytes: int = 65_536
    ask_file_max_quanta: int = 6
    document_summary_max_bytes: int = 65_536
    document_summary_max_read_bytes: int = 1_048_576
    document_summary_max_quanta: int = 10
    document_context_min_tokens: int = 8_000
    document_context_slack_tokens: int = 12_000
    document_context_max_tokens: int = 120_000
    object_copy_max_quanta: int = 5
    llm_write_smoke_max_quanta: int = 5
    clock_demo_iterations: int = 3
    clock_demo_interval_s: float = 0.2
    clock_demo_timezone: str = "Asia/Shanghai"
    chat_max_turns: int = 20
    chat_context_tokens: int = 64_000
    chat_quanta_per_turn: int = 5
    chat_quanta_overhead: int = 8


@dataclass(frozen=True, config=_PYDANTIC_CONFIG)
class AgentLibOSConfig:
    runtime: RuntimeDefaults = field(default_factory=RuntimeDefaults)
    gui: GuiDefaults = field(default_factory=GuiDefaults)
    capability: CapabilityDefaults = field(default_factory=CapabilityDefaults)
    data_flow: DataFlowDefaults = field(default_factory=DataFlowDefaults)
    scheduler: SchedulerDefaults = field(default_factory=SchedulerDefaults)
    process: ProcessDefaults = field(default_factory=ProcessDefaults)
    llm: LLMDefaults = field(default_factory=LLMDefaults)
    tools: ToolDefaults = field(default_factory=ToolDefaults)
    shell: ShellDefaults = field(default_factory=ShellDefaults)
    git: GitDefaults = field(default_factory=GitDefaults)
    jsonrpc: JsonRpcDefaults = field(default_factory=JsonRpcDefaults)
    mcp: McpDefaults = field(default_factory=McpDefaults)
    image: ImageDefaults = field(default_factory=ImageDefaults)
    image_commit: ImageCommitDefaults = field(default_factory=ImageCommitDefaults)
    memory: ObjectMemoryDefaults = field(default_factory=ObjectMemoryDefaults)
    object_tasks: ObjectTaskDefaults = field(default_factory=ObjectTaskDefaults)
    llm_context: LLMContextDefaults = field(default_factory=LLMContextDefaults)
    checkpoint: CheckpointDefaults = field(default_factory=CheckpointDefaults)
    skills: SkillDefaults = field(default_factory=SkillDefaults)
    modules: ModuleDefaults = field(default_factory=ModuleDefaults)
    launcher: LauncherDefaults = field(default_factory=LauncherDefaults)
    scripts: ScriptDefaults = field(default_factory=ScriptDefaults)

    def __post_init__(self) -> None:
        _validate_config(self)


def _validate_git_config(git: GitDefaults) -> None:
    for name in ("executable", "minimum_version", "repository_resource", "worktree_root"):
        _require_non_empty(f"git.{name}", getattr(git, name))
    for name in (
        "local_timeout_s",
        "remote_timeout_s",
        "timeout_hard_limit_s",
        "lock_timeout_s",
    ):
        _positive(f"git.{name}", getattr(git, name))
    for name in (
        "status_entry_limit",
        "status_entry_hard_limit",
        "log_entry_limit",
        "log_entry_hard_limit",
        "output_max_bytes",
        "output_hard_limit_bytes",
        "patch_max_bytes",
        "patch_hard_limit_bytes",
        "state_content_hard_limit_bytes",
    ):
        _positive(f"git.{name}", getattr(git, name))
    for hard_name, hard_value, selected_name, selected_value in (
        ("git.timeout_hard_limit_s", git.timeout_hard_limit_s, "git.remote_timeout_s", git.remote_timeout_s),
        ("git.timeout_hard_limit_s", git.timeout_hard_limit_s, "git.local_timeout_s", git.local_timeout_s),
        ("git.status_entry_hard_limit", git.status_entry_hard_limit, "git.status_entry_limit", git.status_entry_limit),
        ("git.log_entry_hard_limit", git.log_entry_hard_limit, "git.log_entry_limit", git.log_entry_limit),
        ("git.output_hard_limit_bytes", git.output_hard_limit_bytes, "git.output_max_bytes", git.output_max_bytes),
        ("git.patch_hard_limit_bytes", git.patch_hard_limit_bytes, "git.patch_max_bytes", git.patch_max_bytes),
        ("git.output_hard_limit_bytes", git.output_hard_limit_bytes, "git.patch_hard_limit_bytes", git.patch_hard_limit_bytes),
    ):
        _require_at_least(hard_name, hard_value, selected_name, selected_value)
    _require_non_empty_items("git.allowed_remote_schemes", git.allowed_remote_schemes)
    _require_non_empty_items("git.trusted_metadata_roots", git.trusted_metadata_roots)
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", git.minimum_version) is None:
        raise ValueError("git.minimum_version must be a dotted numeric version")
    normalized_schemes = tuple(scheme.casefold() for scheme in git.allowed_remote_schemes)
    if len(set(normalized_schemes)) != len(normalized_schemes):
        raise ValueError("git.allowed_remote_schemes must not contain duplicates")
    if set(normalized_schemes) - {"https", "ssh"}:
        raise ValueError("git.allowed_remote_schemes may contain only https and ssh")
    if not git.protect_git_metadata:
        raise ValueError("git.protect_git_metadata must remain enabled")


def _validate_shell_config(shell: ShellDefaults, tools: ToolDefaults) -> None:
    _require_non_empty("shell.policy_capability_key", shell.policy_capability_key)
    _require_non_empty("shell.policy_resource", shell.policy_resource)
    _positive("shell.timeout_hard_limit_s", shell.timeout_hard_limit_s)
    for name in (
        "max_stdout_chars",
        "max_stderr_chars",
        "stdout_hard_limit_chars",
        "stderr_hard_limit_chars",
    ):
        _nonnegative(f"shell.{name}", getattr(shell, name))
    _require_at_least("shell.stdout_hard_limit_chars", shell.stdout_hard_limit_chars, "shell.max_stdout_chars", shell.max_stdout_chars)
    _require_at_least("shell.stderr_hard_limit_chars", shell.stderr_hard_limit_chars, "shell.max_stderr_chars", shell.max_stderr_chars)
    _require_at_least("shell.timeout_hard_limit_s", shell.timeout_hard_limit_s, "tools.shell_timeout_s", tools.shell_timeout_s)


def _validate_deno_timeout_config(tools: ToolDefaults) -> None:
    _positive("tools.deno_timeout_s", tools.deno_timeout_s)
    _positive(
        "tools.deno_timeout_hard_limit_s",
        tools.deno_timeout_hard_limit_s,
    )
    _require_at_least(
        "tools.deno_timeout_hard_limit_s",
        tools.deno_timeout_hard_limit_s,
        "tools.deno_timeout_s",
        tools.deno_timeout_s,
    )


def _validate_llm_context_config(
    llm_context: LLMContextDefaults,
    tools: ToolDefaults,
) -> None:
    if llm_context.policy not in {"source_only", "llm_context_object"}:
        raise ValueError(
            "llm_context.policy must be source_only or llm_context_object"
        )
    _positive(
        "llm_context.storage_compaction_threshold_bytes",
        llm_context.storage_compaction_threshold_bytes,
    )
    _positive(
        "llm_context.storage_compaction_max_chunks",
        llm_context.storage_compaction_max_chunks,
    )
    _nonnegative(
        "llm_context.storage_compaction_preserve_recent_entries",
        llm_context.storage_compaction_preserve_recent_entries,
    )
    if llm_context.storage_compaction_max_chunks > 64:
        raise ValueError(
            "llm_context.storage_compaction_max_chunks must not exceed 64"
        )
    if llm_context.storage_compaction_preserve_recent_entries > 128:
        raise ValueError(
            "llm_context.storage_compaction_preserve_recent_entries must not exceed 128"
        )
    if (
        llm_context.storage_compaction_threshold_bytes
        >= tools.memory_payload_hard_limit_bytes
    ):
        raise ValueError(
            "llm_context.storage_compaction_threshold_bytes must be less than "
            "tools.memory_payload_hard_limit_bytes"
        )


def _validate_runtime_config(runtime: RuntimeDefaults) -> None:
    for name in (
        "local_store_target",
        "runtime_db_filename",
        "store_backend",
        "workspace_namespace",
        "default_image_id",
        "coding_image_id",
        "launch_authority_mode",
        "default_human",
        "terminal_channel",
    ):
        _require_non_empty(name, getattr(runtime, name))
    if runtime.store_backend == "postgres":
        if runtime.store_dsn is None:
            raise ValueError("runtime.store_dsn is required when runtime.store_backend is postgres")
        _require_non_empty("store_dsn", runtime.store_dsn)
        if (
            urlsplit(runtime.store_dsn).scheme.lower() not in {"postgres", "postgresql"}
            or "://" not in runtime.store_dsn
        ):
            raise ValueError(
                "runtime.store_dsn must use a postgres:// or postgresql:// URI "
                "when runtime.store_backend is postgres"
            )
    elif runtime.store_dsn is not None:
        raise ValueError("runtime.store_dsn must be unset when runtime.store_backend is sqlite")
    if runtime.store_backend == "sqlite":
        local_store_scheme = urlsplit(runtime.local_store_target).scheme.lower()
        if local_store_scheme in {"postgres", "postgresql"}:
            raise ValueError(
                "runtime.local_store_target selects PostgreSQL while runtime.store_backend is sqlite; "
                "use runtime.store_dsn with runtime.store_backend=postgres"
            )
        if "://" in runtime.local_store_target and local_store_scheme not in {"sqlite"}:
            raise ValueError(
                "runtime.local_store_target must be a filesystem path, local, :memory:, or sqlite:// URI"
            )
    _positive_optional("runtime.run_until_idle_max_quanta", runtime.run_until_idle_max_quanta)
    _positive("runtime.launcher_max_quanta", runtime.launcher_max_quanta)


def _validate_config(config: AgentLibOSConfig) -> None:
    _validate_runtime_config(config.runtime)
    data_flow = config.data_flow
    if data_flow.default_trust_level is not SinkTrustLevel.UNTRUSTED:
        raise ValueError("data_flow.default_trust_level must remain untrusted")
    if sensitivity_rank(data_flow.default_max_sensitivity) > sensitivity_rank(DataSensitivity.NORMAL):
        raise ValueError("data_flow.default_max_sensitivity must not exceed normal")
    _require_non_empty("data_flow.registry_resource", data_flow.registry_resource)
    _positive("data_flow.registry_list_limit", data_flow.registry_list_limit)
    _positive("data_flow.decision_list_limit", data_flow.decision_list_limit)
    _positive("data_flow.file_binding_list_limit", data_flow.file_binding_list_limit)
    patterns: set[str] = set()
    priorities: dict[tuple[str, int], str] = {}
    for index, rule in enumerate(data_flow.sink_rules):
        if not isinstance(rule, SinkTrustRule):
            raise ValueError(f"data_flow.sink_rules[{index}] must be a SinkTrustRule")
        if rule.pattern in patterns:
            raise ValueError(f"data_flow.sink_rules contains duplicate pattern: {rule.pattern}")
        patterns.add(rule.pattern)
        base = rule.pattern[:-1] if rule.pattern.endswith("*") else rule.pattern
        priority = (base, len(base))
        conflict = priorities.get(priority)
        if conflict is not None:
            raise ValueError(
                "data_flow.sink_rules contains equal-priority overlapping patterns: "
                f"{conflict!r} and {rule.pattern!r}"
            )
        priorities[priority] = rule.pattern

    gui = config.gui
    for name in (
        "event_buffer_limit",
        "request_body_max_bytes",
        "snapshot_event_limit",
        "snapshot_audit_limit",
        "snapshot_llm_call_limit",
        "snapshot_process_message_limit",
        "snapshot_process_llm_call_limit",
        "snapshot_object_task_limit",
        "snapshot_collection_max_items",
        "snapshot_string_max_chars",
        "sse_payload_max_bytes",
        "agent_rating_comment_max_chars",
    ):
        _positive(f"gui.{name}", getattr(gui, name))
    _nonnegative("gui.scheduler_shutdown_join_timeout_s", gui.scheduler_shutdown_join_timeout_s)
    _nonnegative("gui.http_shutdown_delay_s", gui.http_shutdown_delay_s)
    _nonnegative("gui.object_task_wait_default_timeout_s", gui.object_task_wait_default_timeout_s)
    _nonnegative("gui.object_task_wait_max_timeout_s", gui.object_task_wait_max_timeout_s)
    _require_at_least(
        "gui.object_task_wait_max_timeout_s",
        gui.object_task_wait_max_timeout_s,
        "gui.object_task_wait_default_timeout_s",
        gui.object_task_wait_default_timeout_s,
    )

    capability = config.capability
    _positive("capability.default_delegation_depth", capability.default_delegation_depth)
    _positive("capability.max_rights_per_capability", capability.max_rights_per_capability)
    _nonnegative("capability.max_constraints_bytes", capability.max_constraints_bytes)
    _positive("capability.list_limit", capability.list_limit)
    _nonnegative("capability.decision_explain_preview_chars", capability.decision_explain_preview_chars)

    scheduler = config.scheduler
    _positive_optional("scheduler.max_quanta", scheduler.max_quanta)
    _positive("scheduler.poll_interval_s", scheduler.poll_interval_s)
    _positive("scheduler.max_workers", scheduler.max_workers)
    _nonnegative("scheduler.drain_window_s", scheduler.drain_window_s)
    _nonnegative("scheduler.shutdown_join_timeout_s", scheduler.shutdown_join_timeout_s)

    process = config.process
    _nonnegative_optional_fields(
        "process",
        process,
        (
            "max_tool_calls",
            "max_child_processes",
            "max_runtime_seconds",
            "max_context_materialization_total_tokens",
            "max_llm_calls",
            "max_llm_total_tokens",
            "max_subprocess_wall_seconds",
            "max_subprocess_cpu_seconds",
            "max_subprocess_memory_bytes",
            "max_external_read_bytes",
            "max_external_write_bytes",
            "max_jsonrpc_bytes",
            "max_mcp_bytes",
            "max_deno_syscalls",
        ),
    )
    _positive("process.max_context_materialization_tokens", process.max_context_materialization_tokens)
    _require_non_empty("process.default_goal_text", process.default_goal_text)
    _require_non_empty("process.default_working_directory", process.default_working_directory)
    _positive("process.fork_budget_divisor", process.fork_budget_divisor)
    _nonnegative("process.fork_min_tool_calls", process.fork_min_tool_calls)
    _nonnegative("process.fork_min_child_processes", process.fork_min_child_processes)

    _validate_llm_config(config.llm)

    tools = config.tools
    _require_non_empty("tools.version", tools.version)
    for name in (
        "default_timeout_s",
        "standard_timeout_s",
        "interactive_timeout_s",
        "shell_timeout_s",
        "sandbox_timeout_s",
        "jit_validation_timeout_s",
        "sleep_timeout_grace_s",
    ):
        _positive(f"tools.{name}", getattr(tools, name))
    _validate_deno_timeout_config(tools)
    _nonnegative_fields(
        "tools",
        tools,
        (
            "tool_observability_preview_chars",
            "max_sleep_seconds",
        ),
    )
    for name in (
        "tool_result_payload_hard_limit_bytes",
        "tool_call_args_hard_limit_bytes",
        "filesystem_read_max_bytes",
        "filesystem_read_hard_limit_bytes",
        "directory_entry_limit",
        "directory_entry_hard_limit",
        "executable_snapshot_sibling_limit",
        "memory_payload_chars",
        "memory_payload_hard_limit_chars",
        "memory_payload_hard_limit_bytes",
        "memory_append_entry_max_bytes",
        "message_subject_max_chars",
        "message_body_max_chars",
        "message_payload_max_bytes",
        "message_id_max_chars",
        "message_read_limit",
        "message_read_hard_limit",
        "message_filter_ids_hard_limit",
        "message_filter_json_max_bytes",
        "message_wait_status_max_chars",
        "human_request_payload_max_bytes",
        "human_output_max_chars",
        "human_request_list_limit",
        "object_file_max_bytes",
        "object_file_hard_limit_bytes",
        "jit_source_max_chars",
        "jit_tests_max_count",
        "jit_test_case_max_bytes",
        "jit_validation_log_max_chars",
        "deno_max_rpc_calls",
        "deno_max_stdout_bytes",
        "deno_max_stderr_bytes",
        "static_tool_id_digest_chars",
        "approval_preview_chars",
    ):
        _positive(f"tools.{name}", getattr(tools, name))
    _require_non_empty("tools.default_text_encoding", tools.default_text_encoding)
    _require_non_empty("tools.deno_executable", tools.deno_executable)
    _require_non_empty("tools.clock_timezone", tools.clock_timezone)
    _require_non_empty_items("tools.deno_jsr_allowlist", tools.deno_jsr_allowlist)
    _require_at_least("tools.filesystem_read_hard_limit_bytes", tools.filesystem_read_hard_limit_bytes, "tools.filesystem_read_max_bytes", tools.filesystem_read_max_bytes)
    _require_at_least("tools.directory_entry_hard_limit", tools.directory_entry_hard_limit, "tools.directory_entry_limit", tools.directory_entry_limit)
    _require_at_least("tools.memory_payload_hard_limit_chars", tools.memory_payload_hard_limit_chars, "tools.memory_payload_chars", tools.memory_payload_chars)
    _require_at_least("tools.message_read_hard_limit", tools.message_read_hard_limit, "tools.message_read_limit", tools.message_read_limit)
    _require_at_least("tools.object_file_hard_limit_bytes", tools.object_file_hard_limit_bytes, "tools.object_file_max_bytes", tools.object_file_max_bytes)

    _validate_llm_context_config(config.llm_context, tools)

    _validate_shell_config(config.shell, tools)
    _validate_git_config(config.git)

    jsonrpc = config.jsonrpc
    for name in (
        "registry_resource",
        "endpoint_id_max_chars",
        "method_id_max_chars",
        "rpc_method_max_chars",
        "header_name_max_chars",
        "header_value_max_chars",
        "manifest_max_bytes",
        "timeout_s",
        "timeout_hard_limit_s",
        "max_request_bytes",
        "max_response_bytes",
        "max_request_hard_limit_bytes",
        "max_response_hard_limit_bytes",
        "list_limit",
        "audit_preview_chars",
    ):
        _positive_or_non_empty(f"jsonrpc.{name}", getattr(jsonrpc, name))
    _require_at_least("jsonrpc.timeout_hard_limit_s", jsonrpc.timeout_hard_limit_s, "jsonrpc.timeout_s", jsonrpc.timeout_s)
    _require_at_least("jsonrpc.max_request_hard_limit_bytes", jsonrpc.max_request_hard_limit_bytes, "jsonrpc.max_request_bytes", jsonrpc.max_request_bytes)
    _require_at_least("jsonrpc.max_response_hard_limit_bytes", jsonrpc.max_response_hard_limit_bytes, "jsonrpc.max_response_bytes", jsonrpc.max_response_bytes)
    _require_non_empty_items("jsonrpc.header_env_allowlist", jsonrpc.header_env_allowlist)

    mcp = config.mcp
    for name in (
        "registry_resource",
        "server_id_max_chars",
        "tool_id_max_chars",
        "mcp_name_max_chars",
        "header_name_max_chars",
        "header_value_max_chars",
        "manifest_max_bytes",
        "timeout_s",
        "timeout_hard_limit_s",
        "max_request_bytes",
        "max_response_bytes",
        "max_request_hard_limit_bytes",
        "max_response_hard_limit_bytes",
        "list_limit",
        "audit_preview_chars",
    ):
        _positive_or_non_empty(f"mcp.{name}", getattr(mcp, name))
    _require_at_least("mcp.timeout_hard_limit_s", mcp.timeout_hard_limit_s, "mcp.timeout_s", mcp.timeout_s)
    _require_at_least("mcp.max_request_hard_limit_bytes", mcp.max_request_hard_limit_bytes, "mcp.max_request_bytes", mcp.max_request_bytes)
    _require_at_least("mcp.max_response_hard_limit_bytes", mcp.max_response_hard_limit_bytes, "mcp.max_response_bytes", mcp.max_response_bytes)
    _require_non_empty_items("mcp.header_env_allowlist", mcp.header_env_allowlist)
    _require_non_empty_items("mcp.stdio_env_allowlist", mcp.stdio_env_allowlist)

    image = config.image
    for name in (
        "registry_resource",
        "id_max_chars",
        "name_max_chars",
        "version_max_chars",
        "manifest_hard_limit_bytes",
        "structured_field_hard_limit_bytes",
        "max_default_tools",
        "max_required_capabilities",
        "max_required_modules",
        "package_manifest_name",
        "package_workspace_dir",
        "package_tools_dir",
        "package_resources_dir",
        "materialized_workspace_root",
        "package_manifest_max_bytes",
        "package_manifest_hard_limit_bytes",
        "package_file_max_bytes",
        "package_max_bytes",
        "package_max_files",
        "prompt_max_chars",
        "max_package_jit_tools",
        "max_workspace_grants",
    ):
        _positive_or_non_empty(f"image.{name}", getattr(image, name))
    _require_at_least("image.package_manifest_hard_limit_bytes", image.package_manifest_hard_limit_bytes, "image.package_manifest_max_bytes", image.package_manifest_max_bytes)
    _require_at_least("image.package_max_bytes", image.package_max_bytes, "image.package_file_max_bytes", image.package_file_max_bytes)

    image_commit = config.image_commit
    for name in ("artifact_version", "artifact_hard_limit_bytes", "payload_capture_limit_bytes", "max_required_capabilities", "max_committed_tools", "max_committed_jit_sources", "metadata_preview_chars"):
        _positive(f"image_commit.{name}", getattr(image_commit, name))
    _require_at_least("image_commit.artifact_hard_limit_bytes", image_commit.artifact_hard_limit_bytes, "image_commit.payload_capture_limit_bytes", image_commit.payload_capture_limit_bytes)

    memory = config.memory
    _require_non_empty("memory.object_schema_version", memory.object_schema_version)
    _positive("memory.materialize_budget_tokens", memory.materialize_budget_tokens)
    _positive("memory.query_limit", memory.query_limit)
    for name in ("context_policy", "metadata_sensitivity", "metadata_retention_policy", "process_namespace_prefix"):
        _require_non_empty(f"memory.{name}", getattr(memory, name))

    object_tasks = config.object_tasks
    _positive("object_tasks.max_running_global", object_tasks.max_running_global)
    _positive("object_tasks.max_running_per_object", object_tasks.max_running_per_object)
    _require_at_least("object_tasks.max_running_global", object_tasks.max_running_global, "object_tasks.max_running_per_object", object_tasks.max_running_per_object)
    _require_non_empty("object_tasks.notification_channel", object_tasks.notification_channel)
    _require_non_empty("object_tasks.owner_watch_channel", object_tasks.owner_watch_channel)
    _require_owner_watch_events(object_tasks.owner_watch_events)
    _nonnegative("object_tasks.shutdown_join_timeout_s", object_tasks.shutdown_join_timeout_s)

    llm_context = config.llm_context
    _positive("llm_context.schema_version", llm_context.schema_version)
    _positive("llm_context.recent_event_limit", llm_context.recent_event_limit)
    for name in ("policy", "object_name_prefix"):
        _require_non_empty(f"llm_context.{name}", getattr(llm_context, name))

    checkpoint = config.checkpoint
    for name in ("list_limit", "payload_capture_limit_bytes", "snapshot_hard_limit_bytes", "diff_preview_items"):
        _positive(f"checkpoint.{name}", getattr(checkpoint, name))
    _require_at_least("checkpoint.snapshot_hard_limit_bytes", checkpoint.snapshot_hard_limit_bytes, "checkpoint.payload_capture_limit_bytes", checkpoint.payload_capture_limit_bytes)

    skills = config.skills
    for name in (
        "schema_version",
        "registry_resource",
        "trust_resource",
        "skill_md_max_bytes",
        "skill_md_hard_limit_bytes",
        "resource_read_max_bytes",
        "package_max_bytes",
        "max_package_files",
        "max_prompt_instruction_chars",
        "max_jit_source_chars",
        "discover_limit",
        "id_max_chars",
        "name_max_chars",
        "description_max_chars",
        "version_max_chars",
        "max_tools",
        "max_actions",
        "max_jit_tools",
        "max_required_capabilities",
    ):
        _positive_or_non_empty(f"skills.{name}", getattr(skills, name))
    _require_non_empty_items("skills.global_dirs", skills.global_dirs)
    _require_non_empty_items("skills.workspace_dirs", skills.workspace_dirs)
    _require_non_empty_items("skills.resource_dirs", skills.resource_dirs)
    _require_at_least("skills.skill_md_hard_limit_bytes", skills.skill_md_hard_limit_bytes, "skills.skill_md_max_bytes", skills.skill_md_max_bytes)
    _require_at_least("skills.package_max_bytes", skills.package_max_bytes, "skills.resource_read_max_bytes", skills.resource_read_max_bytes)

    modules = config.modules
    for name in (
        "schema_version",
        "manifest_max_bytes",
        "manifest_hard_limit_bytes",
        "source_max_bytes",
        "package_max_bytes",
        "max_package_files",
        "discover_limit",
        "id_max_chars",
        "name_max_chars",
        "version_max_chars",
        "entrypoint_max_chars",
        "max_declared_tools",
        "max_declared_images",
        "max_declared_syscalls",
        "max_declared_provider_hooks",
        "max_declared_startup_hooks",
    ):
        _positive(f"modules.{name}", getattr(modules, name))
    _require_non_empty_items("modules.manifest_paths", modules.manifest_paths)
    _require_non_empty_items("modules.trusted_modules", modules.trusted_modules)
    _require_non_empty_items("modules.trusted_sha256", modules.trusted_sha256)
    _require_at_least("modules.manifest_hard_limit_bytes", modules.manifest_hard_limit_bytes, "modules.manifest_max_bytes", modules.manifest_max_bytes)
    _require_at_least("modules.package_max_bytes", modules.package_max_bytes, "modules.source_max_bytes", modules.source_max_bytes)

    launcher = config.launcher
    _require_non_empty_items("launcher.permission_presets", launcher.permission_presets)
    for name in ("default_permission_preset", "read_only_preset", "edit_preset", "full_preset"):
        _require_non_empty(f"launcher.{name}", getattr(launcher, name))
    if launcher.default_permission_preset not in launcher.permission_presets:
        raise ValueError("launcher.default_permission_preset must be in launcher.permission_presets")

    scripts = config.scripts
    for name in (
        "ask_file_max_bytes",
        "ask_file_max_quanta",
        "document_summary_max_bytes",
        "document_summary_max_read_bytes",
        "document_summary_max_quanta",
        "document_context_min_tokens",
        "document_context_slack_tokens",
        "document_context_max_tokens",
        "object_copy_max_quanta",
        "llm_write_smoke_max_quanta",
        "clock_demo_iterations",
        "clock_demo_interval_s",
        "chat_max_turns",
        "chat_context_tokens",
        "chat_quanta_per_turn",
        "chat_quanta_overhead",
    ):
        _positive(f"scripts.{name}", getattr(scripts, name))
    _require_non_empty("scripts.clock_demo_timezone", scripts.clock_demo_timezone)
    _require_at_least("scripts.document_summary_max_read_bytes", scripts.document_summary_max_read_bytes, "scripts.document_summary_max_bytes", scripts.document_summary_max_bytes)
    _require_at_least("scripts.document_context_max_tokens", scripts.document_context_max_tokens, "scripts.document_context_min_tokens", scripts.document_context_min_tokens)


def _validate_llm_config(llm: LLMDefaults) -> None:
    _require_non_empty("llm.default_profile_id", llm.default_profile_id)
    if llm.default_profile_id not in llm.profiles:
        raise ValueError(
            "llm.default_profile_id does not reference a configured profile: "
            f"{llm.default_profile_id}"
        )
    for profile_id, profile in llm.profiles.items():
        prefix = f"llm.profiles[{profile_id!r}]"
        _require_non_empty("llm.profiles key", profile_id)
        if profile.kind != "openai_compatible":
            raise ValueError(f"{prefix}.kind is not supported: {profile.kind}")
        if profile.base_url is not None:
            _require_non_empty(f"{prefix}.base_url", profile.base_url)
        if profile.model is not None:
            _require_non_empty(f"{prefix}.model", profile.model)
        _require_non_empty(f"{prefix}.api_key_env", profile.api_key_env)
        if profile.api_mode is not None and profile.api_mode not in {
            "auto",
            "responses",
            "chat",
        }:
            raise ValueError(f"{prefix}.api_mode is not supported: {profile.api_mode}")
        _optional_non_empty(f"{prefix}.safety_identifier", profile.safety_identifier)
        _optional_max_chars(f"{prefix}.safety_identifier", profile.safety_identifier, 64)
        _optional_non_empty(
            f"{prefix}.safety_identifier_env",
            profile.safety_identifier_env,
        )
        _optional_non_empty(f"{prefix}.prompt_cache_key", profile.prompt_cache_key)
        _positive_optional(f"{prefix}.timeout_s", profile.timeout_s)
        _nonnegative_optional(f"{prefix}.max_retries", profile.max_retries)
        _nonnegative_optional(f"{prefix}.temperature", profile.temperature)
        _positive_optional(f"{prefix}.max_tokens", profile.max_tokens)
        _positive_optional(
            f"{prefix}.context_window_tokens",
            profile.context_window_tokens,
        )
        effective_max_tokens = (
            llm.max_tokens if profile.max_tokens is None else profile.max_tokens
        )
        effective_context_window = (
            llm.context_window_tokens
            if profile.context_window_tokens is None
            else profile.context_window_tokens
        )
        if effective_max_tokens >= effective_context_window:
            raise ValueError(
                f"{prefix} effective max_tokens must be less than "
                "effective context_window_tokens"
            )
    _optional_non_empty("llm.safety_identifier", llm.safety_identifier)
    _optional_max_chars("llm.safety_identifier", llm.safety_identifier, 64)
    _optional_non_empty("llm.prompt_cache_key", llm.prompt_cache_key)
    _nonnegative("llm.temperature", llm.temperature)
    _positive("llm.max_tokens", llm.max_tokens)
    _positive("llm.context_window_tokens", llm.context_window_tokens)
    if llm.max_tokens >= llm.context_window_tokens:
        raise ValueError("llm.max_tokens must be less than llm.context_window_tokens")
    _positive("llm.timeout_s", llm.timeout_s)
    _nonnegative("llm.max_retries", llm.max_retries)
    _positive("llm.compatibility_retry_attempts", llm.compatibility_retry_attempts)
    _positive("llm.action_repair_attempts", llm.action_repair_attempts)
    _nonnegative("llm.content_preview_chars", llm.content_preview_chars)
    _nonnegative("llm.tool_arguments_preview_chars", llm.tool_arguments_preview_chars)
    _nonnegative("llm.call_record_preview_chars", llm.call_record_preview_chars)
    _positive("llm.call_record_list_limit", llm.call_record_list_limit)
    _positive("llm.call_record_hard_limit", llm.call_record_hard_limit)
    _require_at_least(
        "llm.call_record_hard_limit",
        llm.call_record_hard_limit,
        "llm.call_record_list_limit",
        llm.call_record_list_limit,
    )
    _require_non_empty("llm.json_instruction", llm.json_instruction)
    _require_status_codes("llm.fallback_status_codes", llm.fallback_status_codes)


def _positive_or_non_empty(name: str, value: object) -> None:
    if isinstance(value, str):
        _require_non_empty(name, value)
    else:
        _positive(name, value)


def _nonnegative_fields(prefix: str, obj: object, names: tuple[str, ...]) -> None:
    for name in names:
        _nonnegative(f"{prefix}.{name}", getattr(obj, name))


def _nonnegative_optional_fields(prefix: str, obj: object, names: tuple[str, ...]) -> None:
    for name in names:
        _nonnegative_optional(f"{prefix}.{name}", getattr(obj, name))


def _positive(name: str, value: object) -> None:
    _require_number(name, value)
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be > 0")


def _positive_optional(name: str, value: object | None) -> None:
    if value is None:
        return
    _positive(name, value)


def _nonnegative(name: str, value: object) -> None:
    _require_number(name, value)
    if value < 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be >= 0")


def _nonnegative_optional(name: str, value: object | None) -> None:
    if value is None:
        return
    _nonnegative(name, value)


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


def _require_non_empty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_non_empty(name: str, value: object | None) -> None:
    if value is not None:
        _require_non_empty(name, value)


def _optional_max_chars(name: str, value: object | None, max_chars: int) -> None:
    if value is not None and isinstance(value, str) and len(value) > max_chars:
        raise ValueError(f"{name} must be at most {max_chars} characters")


def _require_non_empty_items(name: str, values: tuple[str, ...]) -> None:
    for index, value in enumerate(values):
        _require_non_empty(f"{name}[{index}]", value)


def _require_at_least(max_name: str, max_value: int | float, min_name: str, min_value: int | float) -> None:
    if max_value < min_value:
        raise ValueError(f"{max_name} must be >= {min_name}")


def _require_status_codes(name: str, values: tuple[int, ...]) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 100 or value > 599:
            raise ValueError(f"{name}[{index}] must be an HTTP status code")


def _require_owner_watch_events(values: tuple[str, ...]) -> None:
    allowed = {"updated", "linked"}
    _require_non_empty_items("object_tasks.owner_watch_events", values)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"object_tasks.owner_watch_events contains unsupported events: {unknown}")


DEFAULT_CONFIG = AgentLibOSConfig()
