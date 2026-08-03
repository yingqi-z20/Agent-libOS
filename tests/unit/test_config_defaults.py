from __future__ import annotations
import pytest
import asyncio
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from pydantic import ValidationError as PydanticValidationError

import agent_libos.config.loader as config_loader
from agent_libos.config import (
    AgentLibOSConfig,
    CapabilityDefaults,
    DEFAULT_CONFIG,
    GitDefaults,
    LLMDefaults,
    LLMProfile,
    ObjectMemoryDefaults,
    RuntimeDefaults,
    ShellCommandRule,
    SkillDefaults,
    ToolDefaults,
    load_config_from_project_root,
    load_config_file,
    load_config_from_cwd,
)
from agent_libos.utils.yaml_loader import (
    YAML_MAX_NESTING_DEPTH,
    YAML_MAX_UTF8_BYTES,
)
from agent_libos.tools.builtin.jsonrpc import ListJsonRpcEndpointsArgs, ListJsonRpcEndpointsTool
from agent_libos.tools.builtin.memory import ListMemoryNamespaceTool
from agent_libos.tools.builtin.mcp import ListMcpServersArgs, ListMcpServersTool
from agent_libos.llm.client import LLMCompletion
from agent_libos.models.exceptions import HumanResponseRequired, ValidationError
from agent_libos.models import (
    AuthorityRisk,
    AuthorityRule,
    CapabilityEffect,
    CapabilityRight,
    ProcessStatus,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import SQLiteStore, display_store_target, open_store, redact_store_target

class TestConfigDefaults:

    def test_jit_validation_default_allows_cold_deno_startup(self) -> None:
        assert DEFAULT_CONFIG.tools.jit_validation_timeout_s == 15.0

    @pytest.mark.parametrize(
        ("defaults_type", "head_field_names"),
        (
            (
                RuntimeDefaults,
                (
                    "local_store_target", "runtime_db_filename", "store_backend",
                    "store_dsn", "workspace_namespace", "default_image_id",
                    "coding_image_id", "default_human", "terminal_channel",
                    "run_until_idle_max_quanta", "launcher_max_quanta",
                    "launch_authority_mode", "publication_recovery_max_attempts",
                    "publication_reconciliation_page_size",
                    "publication_reconciliation_page_hard_limit",
                    "publication_artifact_lookup_hard_limit",
                    "resource_usage_reservation_recovery_page_size",
                    "resource_usage_reservation_recovery_page_hard_limit",
                    "capability_use_reservation_recovery_page_size",
                    "capability_use_reservation_recovery_page_hard_limit",
                    "object_payload_recovery_page_size",
                    "object_payload_recovery_page_hard_limit",
                    "object_task_recovery_page_size",
                    "object_task_recovery_page_hard_limit",
                    "jit_rehydration_page_size", "jit_rehydration_page_hard_limit",
                    "external_effect_recovery_page_size",
                    "external_effect_recovery_page_hard_limit",
                    "operation_recovery_page_size",
                    "operation_recovery_page_hard_limit", "payload_retention_enabled",
                    "payload_retention_summary_after_seconds",
                    "payload_retention_hash_only_after_seconds",
                    "payload_retention_page_size", "payload_retention_page_hard_limit",
                ),
            ),
            (
                CapabilityDefaults,
                (
                    "default_delegation_depth", "max_rights_per_capability",
                    "max_constraints_bytes", "list_limit",
                    "decision_explain_preview_chars",
                ),
            ),
            (
                ToolDefaults,
                (
                    "version", "default_timeout_s", "standard_timeout_s",
                    "interactive_timeout_s", "default_text_encoding",
                    "tool_observability_preview_chars", "tool_call_args_hard_limit_bytes",
                    "tool_result_payload_hard_limit_bytes", "filesystem_read_max_bytes",
                    "filesystem_read_hard_limit_bytes", "directory_entry_limit",
                    "directory_entry_hard_limit", "executable_snapshot_sibling_limit",
                    "memory_payload_chars", "memory_payload_hard_limit_chars",
                    "memory_payload_hard_limit_bytes", "memory_append_entry_max_bytes",
                    "message_subject_max_chars", "message_body_max_chars",
                    "message_payload_max_bytes", "message_id_max_chars",
                    "message_read_limit", "message_read_hard_limit",
                    "message_filter_ids_hard_limit", "message_filter_json_max_bytes",
                    "message_wait_status_max_chars", "human_request_payload_max_bytes",
                    "human_output_max_chars", "human_request_list_limit",
                    "object_file_max_bytes", "object_file_hard_limit_bytes",
                    "shell_timeout_s", "sandbox_timeout_s", "jit_source_max_chars",
                    "jit_tests_max_count", "jit_test_case_max_bytes",
                    "jit_validation_timeout_s", "jit_validation_log_max_chars",
                    "deno_executable", "deno_timeout_s", "deno_timeout_hard_limit_s",
                    "deno_max_rpc_calls", "deno_max_stdout_bytes",
                    "deno_max_stderr_bytes", "deno_jsr_allowlist",
                    "static_tool_id_digest_chars", "approval_preview_chars",
                    "clock_timezone", "max_sleep_seconds", "sleep_timeout_grace_s",
                ),
            ),
            (
                GitDefaults,
                (
                    "enabled", "executable", "minimum_version", "repository_resource",
                    "worktree_root", "trusted_metadata_roots", "local_timeout_s",
                    "remote_timeout_s", "timeout_hard_limit_s", "lock_timeout_s",
                    "status_entry_limit", "status_entry_hard_limit", "log_entry_limit",
                    "log_entry_hard_limit", "output_max_bytes",
                    "output_hard_limit_bytes", "patch_max_bytes",
                    "patch_hard_limit_bytes", "state_content_hard_limit_bytes",
                    "allowed_remote_schemes", "allow_scp_style_ssh",
                    "allow_file_remotes", "inherit_credential_helpers",
                    "inherit_ssh_agent", "protect_git_metadata",
                ),
            ),
            (
                ObjectMemoryDefaults,
                (
                    "object_schema_version", "materialize_budget_tokens", "query_limit",
                    "context_policy", "metadata_sensitivity",
                    "metadata_retention_policy", "process_namespace_prefix",
                ),
            ),
            (
                SkillDefaults,
                (
                    "schema_version", "registry_resource", "trust_resource",
                    "global_dirs", "workspace_dirs", "resource_dirs",
                    "trusted_global_package_sha256", "global_requires_trust",
                    "skill_md_max_bytes", "skill_md_hard_limit_bytes",
                    "resource_read_max_bytes", "package_max_bytes", "max_package_files",
                    "max_prompt_instruction_chars", "max_jit_source_chars",
                    "discover_limit", "id_max_chars", "name_max_chars",
                    "description_max_chars", "version_max_chars", "max_tools",
                    "max_actions", "max_jit_tools", "max_required_capabilities",
                ),
            ),
        ),
    )
    def test_new_defaults_preserve_head_positional_constructor_abi(
        self,
        defaults_type: type[object],
        head_field_names: tuple[str, ...],
    ) -> None:
        default_instance = defaults_type()
        positional_values = tuple(
            getattr(default_instance, name) for name in head_field_names
        )

        reconstructed = defaults_type(*positional_values)

        assert tuple(field.name for field in fields(defaults_type))[
            : len(head_field_names)
        ] == head_field_names
        for name, expected in zip(head_field_names, positional_values, strict=True):
            assert getattr(reconstructed, name) == expected

    @pytest.mark.parametrize(
        "timezone_key",
        (
            "",
            " UTC",
            "UTC ",
            "Asia /Shanghai",
            "/etc/passwd",
            "../UTC",
            "Asia//Shanghai",
            "Asia\\Shanghai",
            "Mars/Olympus_Mons",
        ),
    )
    def test_tool_clock_timezone_rejects_invalid_or_unknown_iana_keys(
        self,
        timezone_key: str,
    ) -> None:
        with pytest.raises(ValueError, match="tools.clock_timezone"):
            AgentLibOSConfig(
                tools=replace(DEFAULT_CONFIG.tools, clock_timezone=timezone_key),
            )

    @pytest.mark.parametrize("timezone_key", ("UTC", "utc", "Asia/Shanghai"))
    def test_tool_clock_timezone_accepts_runtime_supported_keys(
        self,
        timezone_key: str,
    ) -> None:
        config = AgentLibOSConfig(
            tools=replace(DEFAULT_CONFIG.tools, clock_timezone=timezone_key),
        )

        assert config.tools.clock_timezone == timezone_key

    def test_script_clock_timezone_uses_the_same_key_validation(self) -> None:
        with pytest.raises(ValueError, match="scripts.clock_demo_timezone"):
            AgentLibOSConfig(
                scripts=replace(
                    DEFAULT_CONFIG.scripts,
                    clock_demo_timezone="../UTC",
                ),
            )

    def test_default_profile_map_is_immutable(self) -> None:
        with pytest.raises(TypeError, match="immutable"):
            DEFAULT_CONFIG.llm.profiles["mutated"] = LLMProfile()

        assert "mutated" not in AgentLibOSConfig().llm.profiles
        assert DEFAULT_CONFIG.llm.fallback_json_actions is False

    def test_loaded_shell_rule_conditions_are_deeply_immutable_and_runtime_stable(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "shell-rule.yaml"
        path.write_text(
            "\n".join(
                (
                    "shell:",
                    "  rules:",
                    "    - rule_id: custom.tool.inspect",
                    "      operation: shell.run",
                    "      effect: allow",
                    "      risk: harmless",
                    "      conditions:",
                    "        argv: [tool, inspect]",
                    "        match: exact",
                )
            ),
            encoding="utf-8",
        )
        config = load_config_file(path)
        runtime = Runtime.open(":memory:", config=config)
        try:
            assert (
                runtime.shell.rule_engine.classify(["tool", "inspect"]).rule.rule_id
                == "custom.tool.inspect"
            )
            assert (
                runtime.shell.rule_engine.classify(["tool", "changed"]).rule.rule_id
                != "custom.tool.inspect"
            )

            conditions = runtime.config.shell.rules[0].conditions
            with pytest.raises(TypeError):
                conditions["match"] = "prefix"
            with pytest.raises(TypeError):
                conditions["argv"][1] = "changed"

            assert (
                runtime.shell.rule_engine.classify(["tool", "inspect"]).rule.rule_id
                == "custom.tool.inspect"
            )
            assert (
                runtime.shell.rule_engine.classify(["tool", "changed"]).rule.rule_id
                != "custom.tool.inspect"
            )
            assert json.loads(json.dumps(asdict(config)))["shell"]["rules"][0]["conditions"] == {
                "argv": ["tool", "inspect"],
                "match": "exact",
            }
        finally:
            runtime.close()

    def test_authority_rule_defensively_copies_and_freezes_nested_conditions(self) -> None:
        source = {
            "argv": ["tool", "inspect"],
            "metadata": {"labels": ["original"]},
        }
        rule = AuthorityRule(
            rule_id="custom.tool.inspect",
            operation="shell.run",
            effect=CapabilityEffect.ALLOW,
            risk=AuthorityRisk.HARMLESS,
            conditions=source,
        )

        source["argv"][1] = "changed"
        source["metadata"]["labels"].append("changed")

        assert rule.conditions["argv"] == ["tool", "inspect"]
        assert rule.conditions["metadata"]["labels"] == ["original"]
        with pytest.raises(TypeError):
            rule.conditions["metadata"]["labels"].append("changed")
        with pytest.raises(TypeError):
            rule.conditions["metadata"]["new"] = True
        assert json.loads(json.dumps(asdict(rule)))["conditions"] == {
            "argv": ["tool", "inspect"],
            "metadata": {"labels": ["original"]},
        }

    def test_remote_registry_tool_schema_uses_runtime_list_limits(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            jsonrpc=replace(DEFAULT_CONFIG.jsonrpc, list_limit=173),
            mcp=replace(DEFAULT_CONFIG.mcp, list_limit=181),
            memory=replace(DEFAULT_CONFIG.memory, query_limit=191),
        )

        assert ListJsonRpcEndpointsTool().spec(config=config).input_schema["properties"]["limit"]["maximum"] == 173
        assert ListMcpServersTool().spec(config=config).input_schema["properties"]["limit"]["maximum"] == 181
        assert ListMemoryNamespaceTool().spec(config=config).input_schema["properties"]["limit"]["maximum"] == 191
        assert ListJsonRpcEndpointsArgs.model_validate({"limit": 150}).limit == 150
        assert ListMcpServersArgs.model_validate({"limit": 150}).limit == 150

    def test_object_memory_scan_and_metadata_bounds_have_independent_defaults(self) -> None:
        memory = DEFAULT_CONFIG.memory

        assert memory.query_scan_page_size == 128
        assert memory.query_scan_ceiling == 1_024
        assert memory.query_scan_ceiling > memory.query_limit
        assert memory.metadata_text_max_chars == 32_000
        assert memory.metadata_collection_max_items == 128
        assert memory.metadata_collection_item_max_chars == 2_048
        assert memory.metadata_max_bytes == 131_072

    @pytest.mark.parametrize("invalid", (True, 0, -1))
    def test_object_memory_query_scan_ceiling_rejects_bool_and_nonpositive_values(
        self,
        invalid: object,
    ) -> None:
        with pytest.raises(ValueError):
            replace(
                DEFAULT_CONFIG,
                memory=replace(DEFAULT_CONFIG.memory, query_scan_ceiling=invalid),
            )

    def test_object_memory_query_scan_ceiling_is_separate_from_page_size(self) -> None:
        with pytest.raises(ValueError, match="query_scan_ceiling"):
            replace(
                DEFAULT_CONFIG,
                memory=replace(
                    DEFAULT_CONFIG.memory,
                    query_scan_page_size=10,
                    query_scan_ceiling=9,
                ),
            )

    def test_shell_policy_labels_are_not_configurable_and_rules_require_an_executable(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match='argv'):
            ShellCommandRule((), match='prefix')

        path = tmp_path / 'semantic-shell-label.yaml'
        path.write_text(
            '\n'.join(
                [
                    'shell:',
                    '  always_allow_level: always_deny',
                ]
            ),
            encoding='utf-8',
        )
        with pytest.raises(PydanticValidationError, match='always_allow_level'):
            load_config_file(path)

        assert DEFAULT_CONFIG.shell.always_deny_level == 'always_deny'
        assert DEFAULT_CONFIG.shell.allowlist_auto_else_ask_level == 'allowlist_auto_else_ask'
        assert DEFAULT_CONFIG.shell.blocklist_ask_else_auto_level == 'blocklist_ask_else_auto'
        assert DEFAULT_CONFIG.shell.always_allow_level == 'always_allow'

    def test_load_config_from_cwd_returns_default_when_file_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert load_config_from_cwd() is DEFAULT_CONFIG

    def test_load_config_from_project_root_returns_default_when_file_is_missing(self, tmp_path: Path) -> None:
        assert load_config_from_project_root(root=tmp_path) is DEFAULT_CONFIG

    def test_load_config_from_project_root_ignores_cwd_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_root = tmp_path / "project"
        cwd_root = tmp_path / "cwd"
        project_root.mkdir()
        cwd_root.mkdir()
        project_root.joinpath("config.yaml").write_text(
            "runtime:\n  default_image_id: project-base:v0\n",
            encoding="utf-8",
        )
        cwd_root.joinpath("config.yaml").write_text(
            "runtime:\n  default_image_id: cwd-base:v0\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(cwd_root)
        config = load_config_from_project_root(root=project_root)

        assert config.runtime.default_image_id == "project-base:v0"

    def test_load_config_file_overlays_partial_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.yaml'
        path.write_text(
            '\n'.join(
                [
                    'runtime:',
                    '  default_image_id: custom-base:v0',
                    '  run_until_idle_max_quanta: 3',
                    'tools:',
                    '  filesystem_read_max_bytes: 123',
                    'scheduler:',
                    '  max_workers: 2',
                ]
            ),
            encoding='utf-8',
        )

        config = load_config_file(path)

        assert config.runtime.default_image_id == 'custom-base:v0'
        assert config.runtime.coding_image_id == DEFAULT_CONFIG.runtime.coding_image_id
        assert config.runtime.run_until_idle_max_quanta == 3
        assert config.tools.filesystem_read_max_bytes == 123
        assert config.scheduler.max_workers == 2

    def test_load_config_file_preserves_yaml_merge_override_semantics(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "merged-config.yaml"
        path.write_text(
            """llm:
  profiles:
    default: &base_profile
      model: base-model
    worker:
      <<: *base_profile
      model: worker-model
""",
            encoding="utf-8",
        )

        config = load_config_file(path)

        assert config.llm.profiles["default"].model == "base-model"
        assert config.llm.profiles["worker"].model == "worker-model"

    @pytest.mark.parametrize(
        ("body", "field"),
        (
            ("scheduler:\n  max_workers: true\n", "max_workers"),
            ("process:\n  max_tool_calls: false\n", "max_tool_calls"),
            ("llm:\n  max_tokens: true\n", "max_tokens"),
            ("llm:\n  temperature: false\n", "temperature"),
            (
                "llm:\n  profiles:\n    default:\n      max_tokens: true\n",
                "max_tokens",
            ),
            (
                "llm:\n  profiles:\n    default:\n      temperature: false\n",
                "temperature",
            ),
        ),
    )
    def test_security_sensitive_numeric_fields_reject_yaml_booleans(
        self,
        tmp_path: Path,
        body: str,
        field: str,
    ) -> None:
        path = tmp_path / f"invalid-{field}.yaml"
        path.write_text(body, encoding="utf-8")

        with pytest.raises(PydanticValidationError, match=field):
            load_config_file(path)

    def test_security_sensitive_numeric_fields_accept_numbers(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "numeric-fields.yaml"
        path.write_text(
            "\n".join(
                (
                    "scheduler:",
                    "  max_workers: 2",
                    "process:",
                    "  max_tool_calls: 0",
                    "llm:",
                    "  temperature: 1",
                    "  max_tokens: 4096",
                    "  profiles:",
                    "    default:",
                    "      temperature: 0",
                    "      max_tokens: 2048",
                )
            ),
            encoding="utf-8",
        )

        config = load_config_file(path)

        assert config.scheduler.max_workers == 2
        assert config.process.max_tool_calls == 0
        assert config.llm.temperature == 1.0
        assert config.llm.max_tokens == 4096
        assert config.llm.profiles["default"].temperature == 0.0
        assert config.llm.profiles["default"].max_tokens == 2048

    def test_checkpoint_snapshot_version_is_not_configurable(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.yaml'
        path.write_text(
            'checkpoint:\n  snapshot_version: 3\n',
            encoding='utf-8',
        )

        with pytest.raises(PydanticValidationError, match='snapshot_version'):
            load_config_file(path)

    def test_load_config_file_deep_merges_profile_maps(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.yaml'
        path.write_text(
            '\n'.join(
                [
                    'llm:',
                    '  default_profile_id: coding',
                    '  profiles:',
                    '    coding:',
                    '      model: coding-model',
                    '      temperature: 0.0',
                ]
            ),
            encoding='utf-8',
        )

        config = load_config_file(path)

        assert sorted(config.llm.profiles) == ['coding', 'default']
        assert config.llm.default_profile_id == 'coding'
        assert config.llm.profiles['coding'].model == 'coding-model'
        assert config.llm.profiles['default'].api_key_env == DEFAULT_CONFIG.llm.profiles['default'].api_key_env

    def test_default_llm_config_persists_full_io(self) -> None:
        assert DEFAULT_CONFIG.llm.persist_full_io is True
        assert DEFAULT_CONFIG.llm.timeout_s == 180.0
        assert DEFAULT_CONFIG.llm.auto_wait_on_empty_tool_calls is False
        assert DEFAULT_CONFIG.llm.fallback_json_actions is False
        assert DEFAULT_CONFIG.llm.context_window_tokens == 131_072
        assert DEFAULT_CONFIG.llm.max_tokens == 16_384
        assert DEFAULT_CONFIG.llm.max_input_tokens_per_call == 114_688
        assert DEFAULT_CONFIG.llm.max_total_tokens_per_call == 131_072
        assert DEFAULT_CONFIG.llm.max_tokens < DEFAULT_CONFIG.llm.context_window_tokens

    def test_llm_per_call_budget_envelope_is_positive_and_consistent(self) -> None:
        configured = AgentLibOSConfig(
            llm=LLMDefaults(
                max_tokens=2_000,
                max_input_tokens_per_call=8_000,
                max_total_tokens_per_call=9_000,
                profiles={
                    "default": LLMProfile(),
                    "tight": LLMProfile(
                        max_tokens=500,
                        max_input_tokens_per_call=1_500,
                        max_total_tokens_per_call=1_750,
                    ),
                },
            )
        )
        assert configured.llm.max_input_tokens_per_call == 8_000
        assert configured.llm.profiles["tight"].max_total_tokens_per_call == 1_750

        with pytest.raises(ValueError, match="max_input_tokens_per_call"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    max_input_tokens_per_call=10_001,
                    max_total_tokens_per_call=10_000,
                )
            )
        with pytest.raises(ValueError, match="max_tokens must not exceed"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    max_tokens=10_001,
                    max_input_tokens_per_call=9_999,
                    max_total_tokens_per_call=10_000,
                )
            )
        with pytest.raises(ValueError, match="effective max_input_tokens_per_call"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    profiles={
                        "default": LLMProfile(
                            max_input_tokens_per_call=10_001,
                            max_total_tokens_per_call=10_000,
                        )
                    }
                )
            )
        for field_name in (
            "max_input_tokens_per_call",
            "max_total_tokens_per_call",
        ):
            for invalid_value in (0, -1):
                with pytest.raises(ValueError, match=field_name):
                    AgentLibOSConfig(
                        llm=LLMDefaults(**{field_name: invalid_value})
                    )
                with pytest.raises(ValueError, match=field_name):
                    AgentLibOSConfig(
                        llm=LLMDefaults(
                            profiles={
                                "default": LLMProfile(
                                    **{field_name: invalid_value}
                                )
                            }
                        )
                    )
            with pytest.raises(PydanticValidationError, match=field_name):
                LLMDefaults(**{field_name: True})
            with pytest.raises(PydanticValidationError, match=field_name):
                LLMProfile(**{field_name: True})

    def test_llm_context_storage_compaction_threshold_has_hard_limit_headroom(
        self,
    ) -> None:
        assert DEFAULT_CONFIG.llm_context.policy == "source_only"
        assert DEFAULT_CONFIG.llm_context.prompt_event_payload_max_chars == 2_048
        assert DEFAULT_CONFIG.llm_context.storage_compaction_threshold_bytes == 96_000
        assert DEFAULT_CONFIG.llm_context.storage_compaction_max_chunks == 4
        assert DEFAULT_CONFIG.llm_context.storage_compaction_preserve_recent_entries == 0
        assert (
            DEFAULT_CONFIG.llm_context.storage_compaction_threshold_bytes
            < DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes
        )

        with pytest.raises(
            ValueError,
            match="llm_context.policy must be source_only or llm_context_object",
        ):
            replace(
                DEFAULT_CONFIG,
                llm_context=replace(
                    DEFAULT_CONFIG.llm_context,
                    policy="unexpected",
                ),
            )

        with pytest.raises(
            ValueError,
            match="storage_compaction_threshold_bytes must be less",
        ):
            replace(
                DEFAULT_CONFIG,
                llm_context=replace(
                    DEFAULT_CONFIG.llm_context,
                    storage_compaction_threshold_bytes=(
                        DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes
                    ),
                ),
            )

        with pytest.raises(
            ValueError,
            match="prompt_event_payload_max_chars must be at least 512",
        ):
            replace(
                DEFAULT_CONFIG,
                llm_context=replace(
                    DEFAULT_CONFIG.llm_context,
                    prompt_event_payload_max_chars=511,
                ),
            )

        with pytest.raises(
            ValueError,
            match="storage_compaction_max_chunks must not exceed 64",
        ):
            replace(
                DEFAULT_CONFIG,
                llm_context=replace(
                    DEFAULT_CONFIG.llm_context,
                    storage_compaction_max_chunks=65,
                ),
            )

        with pytest.raises(
            ValueError,
            match="storage_compaction_preserve_recent_entries must not exceed 128",
        ):
            replace(
                DEFAULT_CONFIG,
                llm_context=replace(
                    DEFAULT_CONFIG.llm_context,
                    storage_compaction_preserve_recent_entries=129,
                ),
            )

    def test_llm_context_window_requires_effective_output_reservation_to_fit(self) -> None:
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                context_window_tokens=200_000,
                profiles={
                    "default": LLMProfile(),
                    "coding": LLMProfile(
                        max_tokens=8_192,
                        context_window_tokens=32_768,
                    ),
                },
            )
        )

        assert config.llm.context_window_tokens == 200_000
        assert config.llm.profiles["coding"].context_window_tokens == 32_768

        with pytest.raises(ValueError, match="max_tokens must be less"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    max_tokens=1_000,
                    context_window_tokens=1_000,
                )
            )
        with pytest.raises(ValueError, match="effective max_tokens"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    profiles={
                        "default": LLMProfile(context_window_tokens=16_384),
                    }
                )
            )

    def test_payload_retention_is_explicit_disabled_and_bounded(self) -> None:
        defaults = DEFAULT_CONFIG.runtime
        assert defaults.payload_retention_enabled is False
        assert defaults.payload_retention_summary_after_seconds is None
        assert defaults.payload_retention_hash_only_after_seconds is None

        configured = RuntimeDefaults(
            payload_retention_enabled=True,
            payload_retention_summary_after_seconds=86_400,
            payload_retention_hash_only_after_seconds=604_800,
            payload_retention_page_size=25,
            payload_retention_page_hard_limit=50,
        )
        assert configured.payload_retention_enabled is True
        assert configured.payload_retention_page_size == 25

        with pytest.raises(ValueError, match="requires.*summary"):
            RuntimeDefaults(payload_retention_enabled=True)
        with pytest.raises(ValueError, match="must be at least"):
            RuntimeDefaults(
                payload_retention_summary_after_seconds=20,
                payload_retention_hash_only_after_seconds=10,
            )
        with pytest.raises(ValueError, match="must not exceed"):
            RuntimeDefaults(
                payload_retention_page_size=11,
                payload_retention_page_hard_limit=10,
            )

    def test_publication_artifact_lookup_has_an_independent_positive_hard_limit(
        self,
    ) -> None:
        assert DEFAULT_CONFIG.runtime.publication_artifact_lookup_hard_limit == 5_000
        with pytest.raises(ValueError, match="publication_artifact_lookup_hard_limit"):
            RuntimeDefaults(publication_artifact_lookup_hard_limit=0)

    @pytest.mark.parametrize(
        "field_name",
        (
            "publication_recovery_max_attempts",
            "publication_reconciliation_page_size",
            "publication_reconciliation_page_hard_limit",
            "publication_artifact_lookup_hard_limit",
            "resource_usage_reservation_recovery_page_size",
            "resource_usage_reservation_recovery_page_hard_limit",
            "capability_use_reservation_recovery_page_size",
            "capability_use_reservation_recovery_page_hard_limit",
            "object_payload_recovery_page_size",
            "object_payload_recovery_page_hard_limit",
            "object_task_recovery_page_size",
            "object_task_recovery_page_hard_limit",
            "jit_rehydration_page_size",
            "jit_rehydration_page_hard_limit",
            "external_effect_recovery_page_size",
            "external_effect_recovery_page_hard_limit",
            "operation_recovery_page_size",
            "operation_recovery_page_hard_limit",
            "process_terminal_cleanup_recovery_page_size",
            "process_terminal_cleanup_recovery_page_hard_limit",
            "payload_retention_summary_after_seconds",
            "payload_retention_hash_only_after_seconds",
            "payload_retention_page_size",
            "payload_retention_page_hard_limit",
        ),
    )
    @pytest.mark.parametrize("value", (True, False))
    def test_runtime_recovery_and_retention_integers_reject_python_bool(
        self,
        field_name: str,
        value: bool,
    ) -> None:
        with pytest.raises(PydanticValidationError) as exc_info:
            RuntimeDefaults(**{field_name: value})

        assert any(
            error["loc"] == (field_name,) and error["type"] == "int_type"
            for error in exc_info.value.errors()
        )

    @pytest.mark.parametrize(
        "field_name",
        (
            "publication_recovery_max_attempts",
            "publication_reconciliation_page_size",
            "publication_reconciliation_page_hard_limit",
            "publication_artifact_lookup_hard_limit",
            "resource_usage_reservation_recovery_page_size",
            "resource_usage_reservation_recovery_page_hard_limit",
            "capability_use_reservation_recovery_page_size",
            "capability_use_reservation_recovery_page_hard_limit",
            "object_payload_recovery_page_size",
            "object_payload_recovery_page_hard_limit",
            "object_task_recovery_page_size",
            "object_task_recovery_page_hard_limit",
            "jit_rehydration_page_size",
            "jit_rehydration_page_hard_limit",
            "external_effect_recovery_page_size",
            "external_effect_recovery_page_hard_limit",
            "operation_recovery_page_size",
            "operation_recovery_page_hard_limit",
            "process_terminal_cleanup_recovery_page_size",
            "process_terminal_cleanup_recovery_page_hard_limit",
            "payload_retention_summary_after_seconds",
            "payload_retention_hash_only_after_seconds",
            "payload_retention_page_size",
            "payload_retention_page_hard_limit",
        ),
    )
    @pytest.mark.parametrize("yaml_bool", ("true", "false"))
    def test_runtime_recovery_and_retention_integers_reject_yaml_bool(
        self,
        tmp_path: Path,
        field_name: str,
        yaml_bool: str,
    ) -> None:
        path = tmp_path / f"invalid-{field_name}-{yaml_bool}.yaml"
        path.write_text(
            f"runtime:\n  {field_name}: {yaml_bool}\n",
            encoding="utf-8",
        )

        with pytest.raises(PydanticValidationError) as exc_info:
            load_config_file(path)

        assert any(
            error["loc"] == ("runtime", field_name)
            and error["type"] == "int_type"
            for error in exc_info.value.errors()
        )

    def test_runtime_recovery_and_retention_strict_integers_overlay_from_yaml(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "strict-runtime-integers.yaml"
        path.write_text(
            "\n".join(
                (
                    "runtime:",
                    "  publication_recovery_max_attempts: 4",
                    "  publication_reconciliation_page_size: 125",
                    "  publication_reconciliation_page_hard_limit: 250",
                    "  publication_artifact_lookup_hard_limit: 300",
                    "  resource_usage_reservation_recovery_page_size: 40",
                    "  resource_usage_reservation_recovery_page_hard_limit: 80",
                    "  capability_use_reservation_recovery_page_size: 30",
                    "  capability_use_reservation_recovery_page_hard_limit: 60",
                    "  object_payload_recovery_page_size: 24",
                    "  object_payload_recovery_page_hard_limit: 48",
                    "  object_task_recovery_page_size: 18",
                    "  object_task_recovery_page_hard_limit: 36",
                    "  jit_rehydration_page_size: 20",
                    "  jit_rehydration_page_hard_limit: 40",
                    "  external_effect_recovery_page_size: 250",
                    "  external_effect_recovery_page_hard_limit: 500",
                    "  operation_recovery_page_size: 50",
                    "  operation_recovery_page_hard_limit: 100",
                    "  process_terminal_cleanup_recovery_page_size: 16",
                    "  process_terminal_cleanup_recovery_page_hard_limit: 32",
                    "  payload_retention_enabled: true",
                    "  payload_retention_summary_after_seconds: 60",
                    "  payload_retention_hash_only_after_seconds: null",
                    "  payload_retention_page_size: 25",
                    "  payload_retention_page_hard_limit: 50",
                )
            ),
            encoding="utf-8",
        )

        configured = load_config_file(path)

        assert configured.runtime.publication_recovery_max_attempts == 4
        assert configured.runtime.publication_reconciliation_page_size == 125
        assert configured.runtime.publication_reconciliation_page_hard_limit == 250
        assert configured.runtime.publication_artifact_lookup_hard_limit == 300
        assert configured.runtime.resource_usage_reservation_recovery_page_size == 40
        assert configured.runtime.resource_usage_reservation_recovery_page_hard_limit == 80
        assert configured.runtime.capability_use_reservation_recovery_page_size == 30
        assert configured.runtime.capability_use_reservation_recovery_page_hard_limit == 60
        assert configured.runtime.object_payload_recovery_page_size == 24
        assert configured.runtime.object_payload_recovery_page_hard_limit == 48
        assert configured.runtime.object_task_recovery_page_size == 18
        assert configured.runtime.object_task_recovery_page_hard_limit == 36
        assert configured.runtime.jit_rehydration_page_size == 20
        assert configured.runtime.jit_rehydration_page_hard_limit == 40
        assert configured.runtime.external_effect_recovery_page_size == 250
        assert configured.runtime.external_effect_recovery_page_hard_limit == 500
        assert configured.runtime.operation_recovery_page_size == 50
        assert configured.runtime.operation_recovery_page_hard_limit == 100
        assert configured.runtime.process_terminal_cleanup_recovery_page_size == 16
        assert (
            configured.runtime.process_terminal_cleanup_recovery_page_hard_limit
            == 32
        )
        assert configured.runtime.payload_retention_summary_after_seconds == 60
        assert configured.runtime.payload_retention_hash_only_after_seconds is None
        assert configured.runtime.payload_retention_page_size == 25
        assert configured.runtime.payload_retention_page_hard_limit == 50
        assert configured.runtime.default_image_id == DEFAULT_CONFIG.runtime.default_image_id

    def test_load_config_file_accepts_openai_llm_options(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.yaml'
        path.write_text(
            '\n'.join(
                [
                    'llm:',
                    '  safety_identifier: safe-session',
                    '  prompt_cache_key: project-cache',
                    '  prompt_cache_retention: 24h',
                    '  responses_previous_response_id: true',
                    '  parallel_tool_calls: true',
                    '  auto_wait_on_empty_tool_calls: true',
                    '  fallback_json_actions: true',
                    '  persist_full_io: false',
                    '  profiles:',
                    '    default:',
                    '      model: gpt-test',
                    '      safety_identifier_env: OPENAI_SAFE_ID',
                    '      parallel_tool_calls: false',
                    '      auto_wait_on_empty_tool_calls: false',
                    '      fallback_json_actions: false',
                ]
            ),
            encoding='utf-8',
        )

        config = load_config_file(path)

        assert config.llm.safety_identifier == 'safe-session'
        assert config.llm.prompt_cache_key == 'project-cache'
        assert config.llm.prompt_cache_retention == '24h'
        assert config.llm.responses_previous_response_id is True
        assert config.llm.parallel_tool_calls is True
        assert config.llm.auto_wait_on_empty_tool_calls is True
        assert config.llm.fallback_json_actions is True
        assert config.llm.persist_full_io is False
        assert config.llm.profiles['default'].model == 'gpt-test'
        assert config.llm.profiles['default'].safety_identifier_env == 'OPENAI_SAFE_ID'
        assert config.llm.profiles['default'].parallel_tool_calls is False
        assert config.llm.profiles['default'].auto_wait_on_empty_tool_calls is False
        assert config.llm.profiles['default'].fallback_json_actions is False

    def test_load_config_file_normalizes_legacy_prompt_cache_retention(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / 'legacy-prompt-cache-retention.yaml'
        path.write_text(
            '\n'.join(
                [
                    'llm:',
                    '  prompt_cache_retention: in-memory',
                    '  profiles:',
                    '    default:',
                    '      prompt_cache_retention: in-memory',
                ]
            ),
            encoding='utf-8',
        )

        config = load_config_file(path)
        serialized = asdict(config)

        assert config.llm.prompt_cache_retention == 'in_memory'
        assert config.llm.profiles['default'].prompt_cache_retention == 'in_memory'
        assert serialized['llm']['prompt_cache_retention'] == 'in_memory'
        assert (
            serialized['llm']['profiles']['default']['prompt_cache_retention']
            == 'in_memory'
        )

    def test_load_config_file_rejects_invalid_yaml_shape(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.yaml'
        path.write_text('- runtime\n', encoding='utf-8')

        with pytest.raises(ValueError, match='root must be a mapping'):
            load_config_file(path)

    def test_load_config_file_uses_shared_yaml_parser_boundaries(
        self,
        tmp_path: Path,
    ) -> None:
        duplicate = tmp_path / "duplicate.yaml"
        duplicate.write_text(
            "runtime:\n  default_image_id: first:v0\n"
            "runtime:\n  default_image_id: second:v0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate YAML key"):
            load_config_file(duplicate)

        non_string_key = tmp_path / "non-string-key.yaml"
        non_string_key.write_text(
            "runtime:\n  1: value\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            load_config_file(non_string_key)

        nested = tmp_path / "nested.yaml"
        nested.write_text(
            "runtime:\n  default_image_id: "
            + "[" * YAML_MAX_NESTING_DEPTH
            + "value"
            + "]" * YAML_MAX_NESTING_DEPTH,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="YAML_MAX_NESTING_DEPTH"):
            load_config_file(nested)

        oversized = tmp_path / "oversized.yaml"
        oversized.write_bytes(b"#" * (YAML_MAX_UTF8_BYTES + 1))
        with pytest.raises(ValueError, match="YAML_MAX_UTF8_BYTES"):
            load_config_file(oversized)

    def test_load_config_file_maps_shared_yaml_validation_to_value_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("runtime: {}\n", encoding="utf-8")
        parser_error = ValidationError("bounded YAML failure")

        def fail_parser(_text: str) -> dict[str, object]:
            raise parser_error

        monkeypatch.setattr(config_loader, "load_yaml_mapping", fail_parser)
        with pytest.raises(ValueError, match="bounded YAML failure") as raised:
            load_config_file(path)

        assert raised.value.__cause__ is parser_error

    def test_load_config_file_accepts_retired_1x_compatibility_fields(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "legacy-config.yaml"
        path.write_text(
            "runtime:\n"
            "  runtime_db_filename: legacy.sqlite\n"
            "tools:\n"
            "  sandbox_timeout_s: 7.5\n"
            "image_commit:\n"
            "  metadata_preview_chars: 640\n",
            encoding="utf-8",
        )

        config = load_config_file(path)

        assert config.runtime.runtime_db_filename == "legacy.sqlite"
        assert config.tools.sandbox_timeout_s == 7.5
        assert config.image_commit.metadata_preview_chars == 640

    def test_load_config_file_rejects_unknown_fields_and_invalid_values(self, tmp_path: Path) -> None:
        unknown = tmp_path / 'unknown.yaml'
        unknown.write_text('runtime:\n  missing_field: true\n', encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='missing_field'):
            load_config_file(unknown)

        invalid = tmp_path / 'invalid.yaml'
        invalid.write_text('runtime:\n  launcher_max_quanta: 0\n', encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='launcher_max_quanta'):
            load_config_file(invalid)

        invalid_recovery = tmp_path / 'invalid-recovery.yaml'
        invalid_recovery.write_text(
            'runtime:\n  publication_recovery_max_attempts: 0\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='publication_recovery_max_attempts',
        ):
            load_config_file(invalid_recovery)

        oversized_publication_page = tmp_path / 'oversized-publication-page.yaml'
        oversized_publication_page.write_text(
            'runtime:\n'
            '  publication_reconciliation_page_size: 11\n'
            '  publication_reconciliation_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='publication_reconciliation_page_size',
        ):
            load_config_file(oversized_publication_page)

        oversized_capability_reservation_page = (
            tmp_path / 'oversized-capability-reservation-page.yaml'
        )
        oversized_capability_reservation_page.write_text(
            'runtime:\n'
            '  capability_use_reservation_recovery_page_size: 11\n'
            '  capability_use_reservation_recovery_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='capability_use_reservation_recovery_page_size',
        ):
            load_config_file(oversized_capability_reservation_page)

        oversized_object_task_page = tmp_path / 'oversized-object-task-page.yaml'
        oversized_object_task_page.write_text(
            'runtime:\n'
            '  object_task_recovery_page_size: 11\n'
            '  object_task_recovery_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='object_task_recovery_page_size',
        ):
            load_config_file(oversized_object_task_page)

        oversized_jit_page = tmp_path / 'oversized-jit-page.yaml'
        oversized_jit_page.write_text(
            'runtime:\n'
            '  jit_rehydration_page_size: 11\n'
            '  jit_rehydration_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='jit_rehydration_page_size',
        ):
            load_config_file(oversized_jit_page)

        invalid_effect_page = tmp_path / 'invalid-effect-page.yaml'
        invalid_effect_page.write_text(
            'runtime:\n  external_effect_recovery_page_size: 0\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='external_effect_recovery_page_size',
        ):
            load_config_file(invalid_effect_page)

        oversized_effect_page = tmp_path / 'oversized-effect-page.yaml'
        oversized_effect_page.write_text(
            'runtime:\n'
            '  external_effect_recovery_page_size: 11\n'
            '  external_effect_recovery_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='external_effect_recovery_page_size',
        ):
            load_config_file(oversized_effect_page)

        oversized_operation_page = tmp_path / 'oversized-operation-page.yaml'
        oversized_operation_page.write_text(
            'runtime:\n'
            '  operation_recovery_page_size: 11\n'
            '  operation_recovery_page_hard_limit: 10\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='operation_recovery_page_size',
        ):
            load_config_file(oversized_operation_page)

        bad_retention = tmp_path / 'bad-retention.yaml'
        bad_retention.write_text('llm:\n  prompt_cache_retention: forever\n', encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='prompt_cache_retention'):
            load_config_file(bad_retention)

        bad_safety = tmp_path / 'bad-safety.yaml'
        bad_safety.write_text(f"llm:\n  safety_identifier: {'x' * 65}\n", encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='safety_identifier'):
            load_config_file(bad_safety)

        bad_parallel = tmp_path / 'bad-parallel.yaml'
        bad_parallel.write_text('llm:\n  parallel_tool_calls: []\n', encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='parallel_tool_calls'):
            load_config_file(bad_parallel)

        bad_auto_wait = tmp_path / 'bad-auto-wait.yaml'
        bad_auto_wait.write_text('llm:\n  auto_wait_on_empty_tool_calls: []\n', encoding='utf-8')
        with pytest.raises(PydanticValidationError, match='auto_wait_on_empty_tool_calls'):
            load_config_file(bad_auto_wait)

        bad_json_fallback = tmp_path / 'bad-json-fallback.yaml'
        bad_json_fallback.write_text(
            'llm:\n  fallback_json_actions: []\n',
            encoding='utf-8',
        )
        with pytest.raises(PydanticValidationError, match='fallback_json_actions'):
            load_config_file(bad_json_fallback)

        bad_tool_output_prompt_limit = tmp_path / 'bad-tool-output-prompt-limit.yaml'
        bad_tool_output_prompt_limit.write_text(
            'llm:\n  tool_output_prompt_max_chars: 0\n',
            encoding='utf-8',
        )
        with pytest.raises(
            PydanticValidationError,
            match='tool_output_prompt_max_chars',
        ):
            load_config_file(bad_tool_output_prompt_limit)

    def test_llm_profiles_validate_default_profile_reference(self) -> None:
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                default_profile_id="coding",
                profiles={
                    "default": LLMProfile(),
                    "coding": LLMProfile(model="coding-model", temperature=0.0, max_tokens=256),
                },
            )
        )
        assert config.llm.default_profile_id == "coding"
        assert config.llm.profiles["coding"].model == "coding-model"

        with pytest.raises(ValueError, match="default_profile_id"):
            AgentLibOSConfig(llm=LLMDefaults(default_profile_id="missing", profiles={"default": LLMProfile()}))

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://user:password@gateway.example/v1",
            "https://gateway.example/v1#private",
            "https://gateway.example/v1?api_key=private",
            "https://gateway.example/v1?ACCESS-TOKEN=private",
            "https://gateway.example/v1?x-amz-signature=private",
        ],
    )
    def test_llm_config_rejects_base_url_credential_components(
        self,
        base_url: str,
    ) -> None:
        with pytest.raises(ValueError, match="base_url"):
            AgentLibOSConfig(
                llm=LLMDefaults(
                    profiles={"default": LLMProfile(base_url=base_url)},
                )
            )

    def test_llm_config_allows_non_sensitive_base_url_query_parameters(self) -> None:
        base_url = "https://gateway.example/openai/v1?api-version=2026-01-01&tenant=public"

        config = AgentLibOSConfig(
            llm=LLMDefaults(
                profiles={"default": LLMProfile(base_url=base_url)},
            )
        )

        assert config.llm.profiles["default"].base_url == base_url

    def test_runtime_default_run_until_idle_is_unbounded(self) -> None:
        runtime = Runtime(SQLiteStore(':memory:'), llm_client=ScriptedActionClient())
        try:
            pid = runtime.process.spawn(image=DEFAULT_CONFIG.runtime.default_image_id, goal='two-step process')
            results = asyncio.run(runtime.arun_until_idle())
            assert len(results) == 2
            assert results[0]['action']['action'] == 'discover_skills'
            assert results[1]['action']['action'] == 'process_exit'
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
        finally:
            runtime.close()

    def test_runtime_uses_configured_default_quanta_when_present(self) -> None:
        config = AgentLibOSConfig(runtime=RuntimeDefaults(run_until_idle_max_quanta=1))
        runtime = Runtime(SQLiteStore(':memory:'), llm_client=ScriptedActionClient(), config=config)
        try:
            pid = runtime.process.spawn(image=config.runtime.default_image_id, goal='two-step process')
            results = asyncio.run(runtime.arun_until_idle())
            assert len(results) == 1
            assert results[0]['action']['action'] == 'discover_skills'
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_runtime_respects_explicit_quanta_budget(self) -> None:
        config = AgentLibOSConfig(runtime=RuntimeDefaults(run_until_idle_max_quanta=1))
        runtime = Runtime(SQLiteStore(':memory:'), llm_client=ScriptedActionClient(), config=config)
        try:
            pid = runtime.process.spawn(image=config.runtime.default_image_id, goal='two-step process')
            results = asyncio.run(runtime.arun_until_idle(max_quanta=1))
            assert len(results) == 1
            assert results[0]['action']['action'] == 'discover_skills'
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_process_until_idle_uses_configured_default_quanta_when_present(self) -> None:
        config = AgentLibOSConfig(runtime=RuntimeDefaults(run_until_idle_max_quanta=1))
        runtime = Runtime(SQLiteStore(':memory:'), llm_client=ScriptedActionClient(), config=config)
        try:
            pid = runtime.process.spawn(image=config.runtime.default_image_id, goal='two-step process')
            results = asyncio.run(runtime.arun_process_until_idle(pid))
            assert len(results) == 1
            assert results[0]['action']['action'] == 'discover_skills'
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_default_images_are_built_from_runtime_config(self) -> None:
        config = AgentLibOSConfig(runtime=RuntimeDefaults(workspace_namespace='repo'))
        runtime = Runtime(SQLiteStore(':memory:'), llm_client=ScriptedActionClient(), config=config)
        try:
            pid = runtime.process.spawn(image=config.runtime.coding_image_id, goal='inspect')
            manifest = runtime.authority_manifests.summary_for_process(pid)
            assert manifest is not None
            assert {
                (spec['resource'], tuple(spec['rights']))
                for spec in manifest['required_capabilities']
            } >= {
                ('filesystem:repo:*', (CapabilityRight.READ.value,)),
                (config.runtime.default_human_resource, (CapabilityRight.WRITE.value,)),
            }
            assert runtime.capability.permission_policy(
                pid,
                'filesystem:repo:*',
                CapabilityRight.READ,
            ) == runtime.capability.MISSING

            authorized_pid = runtime.process.spawn(
                image=config.runtime.coding_image_id,
                goal='authorized inspect',
                authority_manifest={
                    'authorized_capabilities': [
                        {
                            'resource': 'filesystem:repo:*',
                            'rights': [CapabilityRight.READ.value],
                        },
                        {
                            'resource': config.runtime.default_human_resource,
                            'rights': [CapabilityRight.WRITE.value],
                        },
                    ]
                },
            )
            assert runtime.capability.permission_policy(
                authorized_pid,
                'filesystem:repo:*',
                CapabilityRight.READ,
            ) == runtime.capability.ALWAYS_ALLOW
            assert runtime.capability.permission_policy(
                authorized_pid,
                config.runtime.default_human_resource,
                CapabilityRight.WRITE,
            ) == runtime.capability.ALWAYS_ALLOW
        finally:
            runtime.close()

    def test_runtime_open_uses_default_local_store_sentinel_as_memory(self) -> None:
        runtime = Runtime.open()
        try:
            assert runtime.store.path == ':memory:'
        finally:
            runtime.close()

    def test_runtime_open_uses_configured_local_store_target_when_target_is_omitted(self, tmp_path: Path) -> None:
        db = tmp_path / 'configured.sqlite'
        config = AgentLibOSConfig(runtime=RuntimeDefaults(local_store_target=str(db)))
        runtime = Runtime.open(config=config)
        try:
            assert runtime.store.path == str(db)
            assert db.exists()
        finally:
            runtime.close()

    def test_runtime_store_defaults_to_sqlite_backend(self) -> None:
        assert DEFAULT_CONFIG.runtime.store_backend == 'sqlite'
        assert DEFAULT_CONFIG.runtime.store_dsn is None
        assert DEFAULT_CONFIG.gui.agent_rating_comment_max_chars > 0
        assert DEFAULT_CONFIG.gui.request_body_max_bytes > (16_777_216 * 4 // 3) + 1_024

    def test_runtime_open_accepts_sqlite_uri(self, tmp_path: Path) -> None:
        db = tmp_path / 'uri.sqlite'
        runtime = Runtime.open(f'sqlite:///{db.as_posix()}')
        try:
            assert Path(runtime.store.path) == db
            assert db.exists()
        finally:
            runtime.close()

    def test_postgres_backend_requires_configured_store_dsn(self) -> None:
        with pytest.raises(ValueError, match='runtime.store_dsn is required'):
            AgentLibOSConfig(runtime=RuntimeDefaults(store_backend='postgres'))

    def test_runtime_store_config_rejects_backend_target_conflicts(self) -> None:
        with pytest.raises(ValueError, match='must be unset'):
            AgentLibOSConfig(
                runtime=RuntimeDefaults(
                    store_backend='sqlite',
                    store_dsn='postgresql://agent:secret@localhost/agent_libos',
                )
            )
        with pytest.raises(ValueError, match='must use a postgres:// or postgresql:// URI'):
            AgentLibOSConfig(
                runtime=RuntimeDefaults(
                    store_backend='postgres',
                    store_dsn='runtime.sqlite',
                )
            )
        with pytest.raises(ValueError, match='must use a postgres:// or postgresql:// URI'):
            AgentLibOSConfig(
                runtime=RuntimeDefaults(
                    store_backend='postgres',
                    store_dsn='postgresql:runtime',
                )
            )
        with pytest.raises(ValueError, match='local_store_target selects PostgreSQL'):
            AgentLibOSConfig(
                runtime=RuntimeDefaults(
                    store_backend='sqlite',
                    local_store_target='postgresql://agent:secret@localhost/agent_libos',
                )
            )

    def test_explicit_runtime_store_target_rejects_unknown_uri_scheme(self) -> None:
        with pytest.raises(ValidationError, match='unsupported runtime store target scheme'):
            open_store('mysql://agent:secret@localhost/agent_libos')
        with pytest.raises(ValidationError, match='unsupported runtime store target scheme'):
            display_store_target('https://example.com/runtime.sqlite')
        with pytest.raises(ValidationError, match='libpq keyword DSNs are not supported'):
            open_store('host=localhost dbname=agent_libos user=agent password=secret')
        with pytest.raises(ValidationError, match='PostgreSQL runtime store targets must use'):
            open_store('postgresql:agent_libos')

    def test_configured_postgres_backend_uses_store_dsn_when_target_is_omitted(self) -> None:
        dsn = 'postgresql://agent:secret@localhost/agent_libos'
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend='postgres',
                store_dsn=dsn,
            )
        )

        assert display_store_target(config=config) == 'postgresql://***@localhost/agent_libos'

    def test_explicit_local_store_target_overrides_configured_postgres_backend(self) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend='postgres',
                store_dsn='postgresql://agent:secret@localhost/agent_libos',
            )
        )
        store = open_store('local', config=config)
        try:
            assert isinstance(store, SQLiteStore)
            assert store.path == ':memory:'
        finally:
            store.close()

    def test_postgres_store_target_display_redacts_password(self) -> None:
        dsn = 'postgresql://agent:secret@localhost:5432/agent_libos?sslmode=disable'
        config = AgentLibOSConfig(runtime=RuntimeDefaults(store_backend='postgres', store_dsn=dsn))

        assert redact_store_target(dsn) == 'postgresql://***@localhost:5432/agent_libos?sslmode=disable'
        assert display_store_target(dsn, config=config) == 'postgresql://***@localhost:5432/agent_libos?sslmode=disable'

    @pytest.mark.parametrize(
        ("dsn", "expected"),
        [
            (
                "postgresql://credential@localhost/db?sslmode=require#private",
                "postgresql://***@localhost/db?sslmode=require",
            ),
            (
                "postgresql://localhost/db?password=query-secret&application_name=agent",
                "postgresql://localhost/db?password=***&application_name=agent",
            ),
            (
                "postgresql://localhost/db?ACCESS_TOKEN=query-secret&sslmode=require",
                "postgresql://localhost/db?ACCESS_TOKEN=***&sslmode=require",
            ),
        ],
    )
    def test_store_target_redaction_covers_userinfo_sensitive_query_and_fragment(
        self,
        dsn: str,
        expected: str,
    ) -> None:
        redacted = redact_store_target(dsn)

        assert redacted == expected
        assert "credential" not in redacted
        assert "query-secret" not in redacted
        assert "private" not in redacted

    def test_spawn_without_image_uses_configured_default_image(self) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(default_image_id='custom-base:v0', coding_image_id='custom-coding:v0')
        )
        runtime = Runtime.open(config=config)
        try:
            pid = runtime.process.spawn(goal='custom image')
            assert runtime.process.get(pid).image_id == 'custom-base:v0'
            assert 'custom-base:v0' in runtime.images
            assert 'custom-coding:v0' in runtime.images
        finally:
            runtime.close()

    def test_config_validation_rejects_invalid_numeric_bounds(self) -> None:
        with pytest.raises(PydanticValidationError, match='fork_budget_divisor'):
            AgentLibOSConfig(runtime=RuntimeDefaults(), process=replace(DEFAULT_CONFIG.process, fork_budget_divisor=0))
        with pytest.raises(PydanticValidationError, match='object_task_wait_max_timeout_s'):
            AgentLibOSConfig(gui=replace(DEFAULT_CONFIG.gui, object_task_wait_default_timeout_s=5, object_task_wait_max_timeout_s=4))
        with pytest.raises(PydanticValidationError, match='agent_rating_comment_max_chars'):
            AgentLibOSConfig(gui=replace(DEFAULT_CONFIG.gui, agent_rating_comment_max_chars=0))
        with pytest.raises(PydanticValidationError, match='executable_snapshot_sibling_limit'):
            AgentLibOSConfig(
                tools=replace(
                    DEFAULT_CONFIG.tools,
                    executable_snapshot_sibling_limit=0,
                )
            )
        with pytest.raises(PydanticValidationError, match='deno_timeout_hard_limit_s'):
            AgentLibOSConfig(
                tools=replace(
                    DEFAULT_CONFIG.tools,
                    deno_timeout_s=61,
                    deno_timeout_hard_limit_s=60,
                )
            )

    def test_config_rejects_unknown_image_grant_mode(self) -> None:
        with pytest.raises(PydanticValidationError, match="launch_authority_mode"):
            RuntimeDefaults(launch_authority_mode="image_grants")

    def test_sqlite_store_llm_call_limits_use_runtime_config(self) -> None:
        config = AgentLibOSConfig(llm=LLMDefaults(call_record_list_limit=1, call_record_hard_limit=2))
        runtime = Runtime.open(config=config)
        try:
            assert runtime.store.list_llm_calls() == []
            with pytest.raises(ValidationError, match='hard cap 2'):
                runtime.store.list_llm_calls(limit=3)
        finally:
            runtime.close()

    def test_static_tool_specs_are_generated_from_runtime_config(self) -> None:
        config = AgentLibOSConfig(
            tools=replace(DEFAULT_CONFIG.tools, shell_timeout_s=6.0, standard_timeout_s=2.5),
            shell=replace(DEFAULT_CONFIG.shell, timeout_hard_limit_s=7.0, max_stdout_chars=9, stdout_hard_limit_chars=11),
        )
        runtime = Runtime.open(config=config)
        try:
            row = next(item for item in runtime.tools.list() if item['name'] == 'read_text_file')
            spec = json.loads(row['spec_json'])
            props = spec['input_schema']['properties']
            assert props['max_bytes']['default'] == config.tools.filesystem_read_max_bytes
            assert props['max_bytes']['maximum'] == config.tools.filesystem_read_hard_limit_bytes
            assert spec['policy']['timeout_s'] == 2.5
        finally:
            runtime.close()

    def test_runtime_tool_argument_defaults_use_runtime_config(self) -> None:
        config = AgentLibOSConfig(runtime=RuntimeDefaults(default_human='admin'))
        runtime = Runtime.open(config=config)
        try:
            pid = runtime.process.spawn(
                goal='configured human default',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'human:admin', 'rights': [CapabilityRight.WRITE.value]}
                    ],
                    'approval_policy': {
                        'requestable_capabilities': [
                            {
                                'resource': 'filesystem:workspace:configured.txt',
                                'rights': [CapabilityRight.WRITE.value],
                            }
                        ]
                    },
                },
            )

            with pytest.raises(HumanResponseRequired):
                runtime.tools.call(
                    pid,
                    'request_permission',
                    {'resource': 'filesystem:workspace:configured.txt', 'rights': ['write'], 'reason': 'test default'},
                )

            pending = runtime.human.pending()
            assert len(pending) == 1
            assert pending[0].human == 'admin'

            output_pid = runtime.process.spawn(
                goal='configured human output',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'human:admin', 'rights': [CapabilityRight.WRITE.value]}
                    ]
                },
            )
            output = runtime.tools.call(output_pid, 'human_output', {'message': 'hello admin'})
            assert output.ok
            output_request = runtime.human.list(pid=output_pid)[0]
            assert output_request.human == 'admin'
        finally:
            runtime.close()

class ScriptedActionClient:

    def __init__(self) -> None:
        self.actions = [
            {'action': 'discover_skills', 'text': 'memory', 'limit': 4},
            {'action': 'process_exit', 'payload': {'done': True}},
        ]

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        action = self.actions.pop(0)
        name = str(action['action'])
        args = {key: value for key, value in action.items() if key != 'action'}
        return LLMCompletion(content='', tool_calls=[{'id': f'config_{len(self.actions)}', 'name': name, 'arguments': json.dumps(args)}])
