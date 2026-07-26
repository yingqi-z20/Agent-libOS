from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_libos.config import (
    DEFAULT_CONFIG,
    AgentLibOSConfig,
    load_config_file,
    load_config_from_project_root,
)
from agent_libos.images import DEFAULT_IMAGES
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.base import StoreAssemblyReadiness
from agent_libos.tools.builtin import (
    ActivateSkillTool,
    DiscoverSkillsTool,
    ProcessExitTool,
    ReadSkillResourceTool,
    UnloadSkillTool,
)
from agent_libos.tools.registry import stable_static_tool_id
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable


LEGACY_TOOL_GROUP_TOOLS = ("discover_tool_groups", "activate_tool_group")
SKILL_LIFECYCLE_TOOLS = (
    "discover_skills",
    "activate_skill",
    "read_skill_resource",
    "unload_skill",
)
SKILL_PROJECTION_BOOTSTRAP = (*SKILL_LIFECYCLE_TOOLS, "process_exit")
_LEGACY_TOOL_GROUP_TOOLS: dict[str, tuple[str, ...]] = {
    "filesystem_read": (
        "read_text_file",
        "read_directory",
        "create_object_from_file",
    ),
    "filesystem": (
        "read_text_file",
        "write_text_file",
        "read_directory",
        "write_directory",
        "delete_file",
        "delete_directory",
        "create_object_from_file",
        "write_object_to_file",
        "get_working_directory",
        "set_working_directory",
    ),
    "git": (
        "git_repository_info",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_blame",
        "git_list_refs",
        "git_list_remotes",
        "git_list_worktrees",
        "git_stage",
        "git_unstage",
        "git_commit",
        "git_restore",
        "git_branch",
        "git_switch",
        "git_tag",
        "git_integrate",
        "git_stash",
        "git_reset",
        "git_clean",
        "git_worktree",
        "git_create_patch",
        "git_apply_patch",
        "git_fetch",
        "git_pull",
        "git_push",
        "git_create_pull_request",
        "git_list_pull_requests",
        "git_inspect_pull_request",
        "git_review_pull_request",
        "git_merge_pull_request",
        "git_close_pull_request",
    ),
    "process": (
        "list_child_processes",
        "spawn_child_process",
        "fork_child_process",
        "wait_child_process",
        "signal_child_process",
        "merge_child_memory",
        "send_process_message",
        "read_process_messages",
        "receive_process_messages",
        "exec_process",
    ),
    "remote": (
        "list_jsonrpc_endpoints",
        "inspect_jsonrpc_endpoint",
        "call_jsonrpc_method",
        "list_mcp_servers",
        "inspect_mcp_server",
        "list_mcp_tools",
        "call_mcp_tool",
    ),
    "checkpoint": (
        "create_checkpoint",
        "list_checkpoints",
        "inspect_checkpoint",
        "diff_checkpoint",
        "fork_checkpoint",
        "restore_checkpoint",
        "commit_checkpoint_to_image",
    ),
    "memory": (
        "create_memory_namespace",
        "list_memory_namespace",
        "create_memory_object",
        "append_memory_object",
        "read_memory_object",
        "create_object_from_file",
        "write_object_to_file",
    ),
    "skills": SKILL_LIFECYCLE_TOOLS,
    "object_tasks": (
        "start_object_task",
        "get_object_task",
        "list_object_tasks",
        "wait_object_task",
        "watch_object_task_owner",
        "cancel_object_task",
    ),
    "self_evolution": (
        "load_image_package",
        "propose_jit_tool",
        "validate_jit_tool",
        "register_jit_tool",
    ),
    "authority": (
        "list_capabilities",
        "inspect_capability",
        "delegate_capability",
        "revoke_capability",
    ),
    "shell": ("run_shell_command", "parse_pytest_log"),
    "context": ("compact_process_context",),
    "clock": ("sleep",),
}
LEGACY_TOOL_GROUPS = frozenset(_LEGACY_TOOL_GROUP_TOOLS)
_LEGACY_METADATA_KEYS = frozenset({"lazy_tool_groups", "initial_tool_groups"})
_TERMINAL_OBJECT_TASK_STATES = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "abandoned",
        "superseded_by_restore",
        "result_unavailable_after_reopen",
    }
)
_TERMINAL_PUBLICATION_STATES = frozenset({"committed", "rolled_back"})
_MIGRATED_STATIC_TOOL_CREATED_AT = "2026-07-24T00:00:00+00:00"
_CORE_STATIC_TOOL_SCOPE = "module:agent-libos-core:v0"
# Legacy checkpoint documents already share this default hard ceiling. Apply
# the same explicit bound to every JSON column before migration planning so an
# operator gets one deterministic error instead of unbounded parsing/copying.
_LEGACY_JSON_HARD_LIMIT_BYTES = 16_777_216


class ToolSkillMigrationError(ValidationError):
    """The legacy store cannot be transformed without guessing."""


@dataclass(slots=True)
class ToolSkillMigrationReport:
    applied: bool = False
    builtin_images_rewritten: int = 0
    custom_images_migrated: int = 0
    processes_migrated: int = 0
    checkpoints_migrated: int = 0
    checkpoint_artifacts_created: int = 0
    new_static_tools_inserted: int = 0
    static_tools_canonicalized: int = 0
    old_static_tools_deleted: int = 0
    process_bindings_rebuilt: int = 0
    artifact_id_remaps: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(
            (
                self.builtin_images_rewritten,
                self.custom_images_migrated,
                self.processes_migrated,
                self.checkpoints_migrated,
                self.checkpoint_artifacts_created,
                self.new_static_tools_inserted,
                self.static_tools_canonicalized,
                self.old_static_tools_deleted,
                self.process_bindings_rebuilt,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["changed"] = self.changed
        result["mode"] = "apply" if self.applied else "dry-run"
        return result


class _DryRunRollback(Exception):
    def __init__(self, report: ToolSkillMigrationReport):
        super().__init__("rollback successful tool-group migration dry-run")
        self.report = report


@dataclass(slots=True)
class _StaticToolPlan:
    config: AgentLibOSConfig
    created_at: str
    updated_at: str
    new_rows: dict[str, dict[str, Any]]
    expected_ids: dict[str, str]

    @classmethod
    def build(cls, config: AgentLibOSConfig, created_at: str) -> _StaticToolPlan:
        tool_instances = (
            DiscoverSkillsTool(),
            ActivateSkillTool(),
            ReadSkillResourceTool(),
            UnloadSkillTool(),
            ProcessExitTool(),
        )
        expected_ids = {
            name: stable_static_tool_id(
                name,
                digest_chars=config.tools.static_tool_id_digest_chars,
            )
            for name in (*LEGACY_TOOL_GROUP_TOOLS, *SKILL_PROJECTION_BOOTSTRAP)
        }
        rows: dict[str, dict[str, Any]] = {}
        for tool in tool_instances:
            tool_id = expected_ids[tool.name]
            rows[tool.name] = {
                "tool_id": tool_id,
                "name": tool.name,
                "spec_json": dumps(tool.spec(config=config)),
                "scope": _CORE_STATIC_TOOL_SCOPE,
                "registered_by": _CORE_STATIC_TOOL_SCOPE,
                "created_at": _MIGRATED_STATIC_TOOL_CREATED_AT,
                "ephemeral": 0,
            }
        return cls(
            config=config,
            # Tool rows become part of content-addressed checkpoint artifacts.
            # Their timestamp must not make a dry-run and subsequent apply
            # derive different hashes from identical input.
            created_at=_MIGRATED_STATIC_TOOL_CREATED_AT,
            updated_at=created_at,
            new_rows=rows,
            expected_ids=expected_ids,
        )


def migrate_tool_groups_to_skills(
    store: Any,
    *,
    apply: bool = False,
    config: AgentLibOSConfig | None = None,
) -> ToolSkillMigrationReport:
    """Migrate one closed runtime store from Tool Groups to built-in Tool Skills.

    This is deliberately an explicit content migration.  It is never invoked by
    runtime startup and it does not change the storage schema version.  Dry-run
    is the default and executes the complete mutation/postcondition program in a
    transaction which is then rolled back.  Apply mode requires ownership of the
    top-level transaction so ``applied=True`` means its commit has completed.
    """

    if type(apply) is not bool:
        raise TypeError("apply must be a bool")
    selected_config = config or getattr(store, "config", None) or DEFAULT_CONFIG
    report = ToolSkillMigrationReport(applied=apply)
    static_tools = _StaticToolPlan.build(selected_config, utc_now())
    try:
        # Keep the top-level check and transaction entry under one store lock.
        # Otherwise another thread could open an outer transaction after the
        # check and make this migration commit only a nested savepoint.
        with store.locked():
            if (
                apply
                and store.probe_runtime_assembly_readiness()
                is StoreAssemblyReadiness.ACTIVE_TRANSACTION
            ):
                raise ToolSkillMigrationError(
                    "cannot apply the Tool Skills migration inside an active "
                    "store transaction; finish the outer transaction first"
                )
            with store.transaction() as cursor:
                _run_migration(cursor, report=report, static_tools=static_tools)
                if not apply:
                    raise _DryRunRollback(report)
    except _DryRunRollback as rollback:
        rollback.report.applied = False
        return rollback.report
    return report


def cli(argv: Sequence[str] | None = None) -> int:
    """Installed entry point for the explicit offline migration."""

    parser = argparse.ArgumentParser(
        prog="agent-libos-migrate-tool-groups",
        description=(
            "Offline one-time migration from removed Tool Groups to built-in "
            "Tool Skills. The default is a transactionally rolled-back dry-run."
        ),
    )
    parser.add_argument(
        "store",
        nargs="?",
        help=(
            "SQLite store path or PostgreSQL URI. Omit to use the selected "
            "configuration's runtime store target."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config overlay; defaults to project-root config.yaml when present.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the migration. Without this flag every planned write is rolled back.",
    )
    args = parser.parse_args(argv)
    try:
        config = (
            load_config_file(args.config)
            if args.config is not None
            else load_config_from_project_root()
        )
        # Import through the package only after its initialization is complete;
        # the migration module itself must remain absent from runtime startup.
        from agent_libos.storage.factory import (
            display_store_target,
            open_store_for_migration,
        )

        store = open_store_for_migration(args.store, config=config)
        try:
            report = migrate_tool_groups_to_skills(
                store,
                apply=bool(args.apply),
                config=config,
            )
        finally:
            store.close()
        output = {
            "store": display_store_target(args.store, config=config),
            **report.to_dict(),
        }
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _run_migration(
    cursor: Any,
    *,
    report: ToolSkillMigrationReport,
    static_tools: _StaticToolPlan,
) -> None:
    _preflight_inflight_activity(cursor, static_tools)
    _preflight_legacy_tool_rows(cursor, static_tools)
    active_images = _active_image_rows(cursor)
    checkpoint_rows = [
        dict(row)
        for row in _rows(cursor, "SELECT * FROM checkpoints ORDER BY checkpoint_id")
    ]
    planned_images = _plan_active_images(
        active_images,
        report=report,
        static_tools=static_tools,
    )
    planned_processes = _plan_active_processes(cursor, static_tools=static_tools)
    artifact_migrator = _ArtifactMigrator(
        cursor=cursor,
        report=report,
        static_tools=static_tools,
    )
    _cascade_active_image_artifacts(
        active_images,
        planned_images,
        artifact_migrator=artifact_migrator,
    )
    planned_checkpoints = _plan_checkpoints(
        checkpoint_rows,
        artifact_migrator=artifact_migrator,
        static_tools=static_tools,
    )
    _preflight_checkpoint_restore_references(
        cursor,
        checkpoint_ids=frozenset(planned_checkpoints),
        static_tools=static_tools,
    )
    migration_needed = bool(
        planned_images
        or planned_processes
        or planned_checkpoints
        or report.artifact_id_remaps
        or _old_tool_rows_exist(cursor, static_tools)
    )
    if migration_needed:
        _ensure_new_static_tools(cursor, static_tools, report)
    _apply_migration_plans(
        cursor,
        active_images=planned_images,
        active_processes=planned_processes,
        checkpoint_rows=checkpoint_rows,
        checkpoints=planned_checkpoints,
        report=report,
        static_tools=static_tools,
    )
    _assert_mutable_references_clean(
        cursor,
        static_tools=static_tools,
        allow_planned_checkpoint_artifacts=frozenset(report.artifact_id_remaps),
    )
    if migration_needed:
        _delete_old_static_tools(cursor, static_tools, report)


def _active_image_rows(cursor: Any) -> dict[str, dict[str, Any]]:
    return {
        str(row["image_id"]): dict(row)
        for row in _rows(cursor, "SELECT * FROM images ORDER BY image_id")
    }


def _plan_active_images(
    rows: Mapping[str, dict[str, Any]],
    *,
    report: ToolSkillMigrationReport,
    static_tools: _StaticToolPlan,
) -> dict[str, dict[str, Any]]:
    planned: dict[str, dict[str, Any]] = {}
    for image_id, row in rows.items():
        manifest = _json_object(row.get("manifest_json"), f"image {image_id} manifest")
        if manifest.get("image_id") != image_id:
            raise ToolSkillMigrationError(
                f"images[{image_id}] key does not match manifest.image_id"
            )
        migrated, kind = _migrate_image_definition(
            manifest,
            path=f"images[{image_id}]",
            static_tools=static_tools,
        )
        if migrated == manifest:
            continue
        planned[image_id] = migrated
        if kind == "builtin":
            report.builtin_images_rewritten += 1
        elif kind == "custom":
            report.custom_images_migrated += 1
            _record_custom_image_warning(
                report,
                path=f"images[{image_id}]",
                image_package=_image_boot_kind(manifest) == "image_package",
            )
    return planned


def _record_custom_image_warning(
    report: ToolSkillMigrationReport,
    *,
    path: str,
    image_package: bool,
) -> None:
    report.warnings.append(
        f"{path}: custom legacy image now uses full tool projection; set "
        "metadata.tool_projection='skills' explicitly after selecting default_skills"
    )
    if image_package:
        report.warnings.append(
            f"{path}: preserved immutable image_package raw files"
        )


def _plan_active_processes(
    cursor: Any,
    *,
    static_tools: _StaticToolPlan,
) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    planned: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    for row in _rows(cursor, "SELECT * FROM processes ORDER BY pid"):
        pid = str(row["pid"])
        full = _tool_map(row.get("tool_table_json"), f"process {pid} tool_table_json")
        model = _tool_map(
            row.get("model_tool_table_json"),
            f"process {pid} model_tool_table_json",
        )
        _validate_model_tool_subset(full, model, path=f"processes[{pid}]")
        migrated_full = _replace_tool_map(
            full,
            path=f"processes[{pid}].tool_table",
            static_tools=static_tools,
        )
        migrated_model = _replace_tool_map(
            model,
            path=f"processes[{pid}].model_tool_table",
            static_tools=static_tools,
        )
        if migrated_full != full or migrated_model != model:
            planned[pid] = (migrated_full, migrated_model)
    return planned


def _cascade_active_image_artifacts(
    rows: Mapping[str, dict[str, Any]],
    planned: dict[str, dict[str, Any]],
    *,
    artifact_migrator: _ArtifactMigrator,
) -> None:
    for image_id, row in rows.items():
        manifest = planned.get(
            image_id,
            _json_object(row.get("manifest_json"), f"image {image_id} manifest"),
        )
        migrated = artifact_migrator.cascade_image_artifact(
            manifest,
            path=f"images[{image_id}]",
        )
        if migrated != manifest:
            planned[image_id] = migrated


def _plan_checkpoints(
    rows: list[dict[str, Any]],
    *,
    artifact_migrator: _ArtifactMigrator,
    static_tools: _StaticToolPlan,
) -> dict[str, dict[str, Any]]:
    planned: dict[str, dict[str, Any]] = {}
    for row in rows:
        checkpoint_id = str(row["checkpoint_id"])
        snapshot = _json_object(
            row.get("snapshot_json"),
            f"checkpoint {checkpoint_id} snapshot",
        )
        migrated = _migrate_checkpoint_snapshot(
            snapshot,
            path=f"checkpoints[{checkpoint_id}]",
            artifact_migrator=artifact_migrator,
            static_tools=static_tools,
        )
        if migrated != snapshot:
            planned[checkpoint_id] = migrated
    return planned


def _old_tool_rows_exist(cursor: Any, static_tools: _StaticToolPlan) -> bool:
    return bool(
        _rows(
            cursor,
            "SELECT tool_id FROM tools WHERE tool_id IN (?, ?) LIMIT 1",
            tuple(
                static_tools.expected_ids[name]
                for name in LEGACY_TOOL_GROUP_TOOLS
            ),
        )
    )


def _apply_migration_plans(
    cursor: Any,
    *,
    active_images: Mapping[str, dict[str, Any]],
    active_processes: Mapping[str, tuple[dict[str, str], dict[str, str]]],
    checkpoint_rows: list[dict[str, Any]],
    checkpoints: Mapping[str, dict[str, Any]],
    report: ToolSkillMigrationReport,
    static_tools: _StaticToolPlan,
) -> None:
    for image_id, manifest in active_images.items():
        cursor.execute(
            "UPDATE images SET manifest_json = ?, updated_at = ? WHERE image_id = ?",
            (dumps(manifest), static_tools.updated_at, image_id),
        )
    for pid, (full, model) in active_processes.items():
        _apply_process_plan(
            cursor,
            pid=pid,
            full=full,
            model=model,
            report=report,
            static_tools=static_tools,
        )
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoint_rows}
    for checkpoint_id, snapshot in checkpoints.items():
        _apply_checkpoint_plan(
            cursor,
            checkpoint_id=checkpoint_id,
            row=checkpoint_by_id.get(checkpoint_id),
            snapshot=snapshot,
            report=report,
            static_tools=static_tools,
        )


def _apply_process_plan(
    cursor: Any,
    *,
    pid: str,
    full: Mapping[str, str],
    model: Mapping[str, str],
    report: ToolSkillMigrationReport,
    static_tools: _StaticToolPlan,
) -> None:
    current = _rows(cursor, "SELECT revision FROM processes WHERE pid = ?", (pid,))
    if len(current) != 1:
        raise ToolSkillMigrationError(f"process disappeared during migration: {pid}")
    revision = _reserve_migration_process_revision(
        cursor,
        pid=pid,
        current_revision=int(current[0]["revision"]),
    )
    cursor.execute(
        "UPDATE processes SET tool_table_json = ?, model_tool_table_json = ?, "
        "revision = ?, updated_at = ? WHERE pid = ?",
        (dumps(full), dumps(model), revision, static_tools.updated_at, pid),
    )
    _replace_process_bindings(cursor, pid=pid, full=full, model=model)
    report.processes_migrated += 1
    report.process_bindings_rebuilt += 1


def _apply_checkpoint_plan(
    cursor: Any,
    *,
    checkpoint_id: str,
    row: dict[str, Any] | None,
    snapshot: dict[str, Any],
    report: ToolSkillMigrationReport,
    static_tools: _StaticToolPlan,
) -> None:
    if row is None:
        raise ToolSkillMigrationError(
            f"checkpoint disappeared during migration: {checkpoint_id}"
        )
    serialized_snapshot = dumps(snapshot)
    snapshot_bytes = len(serialized_snapshot.encode("utf-8"))
    hard_limit = static_tools.config.checkpoint.snapshot_hard_limit_bytes
    if snapshot_bytes > hard_limit:
        raise ToolSkillMigrationError(
            f"migrated checkpoint {checkpoint_id} exceeds snapshot_hard_limit_bytes="
            f"{hard_limit}"
        )
    metadata = _json_object(
        row.get("metadata_json"),
        f"checkpoint {checkpoint_id} metadata",
    )
    metadata["snapshot_bytes"] = snapshot_bytes
    cursor.execute(
        "UPDATE checkpoints SET snapshot_json = ?, metadata_json = ? "
        "WHERE checkpoint_id = ?",
        (serialized_snapshot, dumps(metadata), checkpoint_id),
    )
    report.checkpoints_migrated += 1


def _preflight_legacy_tool_rows(
    cursor: Any,
    static_tools: _StaticToolPlan,
) -> None:
    old_names = tuple(LEGACY_TOOL_GROUP_TOOLS)
    old_ids = tuple(static_tools.expected_ids[name] for name in old_names)
    rows = _rows(
        cursor,
        "SELECT * FROM tools WHERE name IN (?, ?) OR tool_id IN (?, ?) "
        "ORDER BY name, tool_id",
        (*old_names, *old_ids),
    )
    found: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "")
        if name not in LEGACY_TOOL_GROUP_TOOLS:
            raise ToolSkillMigrationError(
                f"legacy static tool id collision: {row.get('tool_id')} belongs to {name}"
            )
        if str(row.get("tool_id")) != static_tools.expected_ids[name]:
            raise ToolSkillMigrationError(f"legacy static tool id mismatch for {name}")
        if (
            bool(row.get("ephemeral"))
            or str(row.get("scope")) not in {"static", _CORE_STATIC_TOOL_SCOPE}
            or str(row.get("registered_by"))
            not in {"runtime.core", _CORE_STATIC_TOOL_SCOPE}
        ):
            raise ToolSkillMigrationError(f"legacy lifecycle tool row is malformed: {name}")
        if name in found:
            raise ToolSkillMigrationError(f"duplicate legacy lifecycle tool row: {name}")
        found.add(name)
    expected = set(LEGACY_TOOL_GROUP_TOOLS)
    if found and found != expected:
        raise ToolSkillMigrationError(
            "legacy lifecycle durable rows must contain the complete removed pair"
        )
    if _store_has_legacy_references(cursor, static_tools) and found != expected:
        raise ToolSkillMigrationError(
            "legacy lifecycle references require both canonical durable tool rows"
        )


def _store_has_legacy_references(
    cursor: Any,
    static_tools: _StaticToolPlan,
) -> bool:
    tokens = _legacy_tokens(static_tools)
    queries = (
        ("SELECT tool_table_json, model_tool_table_json FROM processes", ()),
        ("SELECT manifest_json FROM images", ()),
        ("SELECT snapshot_json FROM checkpoints", ()),
    )
    for sql, params in queries:
        for row in _rows(cursor, sql, params):
            for raw in row.values():
                if isinstance(raw, str) and _contains_token(
                    _json_value(raw, "legacy reference preflight"),
                    tokens,
                ):
                    return True
    return False


def _ensure_new_static_tools(
    cursor: Any,
    static_tools: _StaticToolPlan,
    report: ToolSkillMigrationReport,
) -> None:
    relevant_names = (*LEGACY_TOOL_GROUP_TOOLS, *SKILL_PROJECTION_BOOTSTRAP)
    placeholders = ", ".join("?" for _ in relevant_names)
    rows = _rows(
        cursor,
        f"SELECT * FROM tools WHERE name IN ({placeholders}) ORDER BY name, tool_id",
        relevant_names,
    )
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(str(row["name"]), []).append(row)
    for name, matches in by_name.items():
        expected_id = static_tools.expected_ids[name]
        if len(matches) != 1 or str(matches[0]["tool_id"]) != expected_id:
            raise ToolSkillMigrationError(
                f"durable static tool identity mismatch for {name}: expected {expected_id}"
            )
        if bool(matches[0].get("ephemeral")):
            raise ToolSkillMigrationError(f"lifecycle tool must not be ephemeral: {name}")
        if name in SKILL_PROJECTION_BOOTSTRAP:
            canonical, changed = _canonicalize_bootstrap_tool_row(
                matches[0],
                name=name,
                path=f"tools[{name}]",
                static_tools=static_tools,
            )
            if changed:
                cursor.execute(
                    "UPDATE tools SET spec_json = ?, scope = ?, registered_by = ?, "
                    "ephemeral = ? WHERE tool_id = ?",
                    (
                        canonical["spec_json"],
                        canonical["scope"],
                        canonical["registered_by"],
                        canonical["ephemeral"],
                        canonical["tool_id"],
                    ),
                )
                report.static_tools_canonicalized += 1
        elif str(matches[0].get("scope")) not in {
            "static",
            _CORE_STATIC_TOOL_SCOPE,
        }:
            raise ToolSkillMigrationError(
                f"legacy lifecycle tool has invalid scope: {name}"
            )
    all_expected_ids = tuple(static_tools.expected_ids.values())
    id_placeholders = ", ".join("?" for _ in all_expected_ids)
    for row in _rows(
        cursor,
        f"SELECT tool_id, name FROM tools WHERE tool_id IN ({id_placeholders})",
        all_expected_ids,
    ):
        tool_id = str(row["tool_id"])
        expected_name = next(
            name for name, selected_id in static_tools.expected_ids.items()
            if selected_id == tool_id
        )
        if str(row["name"]) != expected_name:
            raise ToolSkillMigrationError(
                f"durable tool id collision for {tool_id}: {row['name']} != {expected_name}"
            )
    for name in SKILL_PROJECTION_BOOTSTRAP:
        if name in by_name:
            continue
        row = static_tools.new_rows[name]
        columns = tuple(row)
        cursor.execute(
            f"INSERT INTO tools ({', '.join(columns)}) VALUES "
            f"({', '.join('?' for _ in columns)})",
            tuple(row[column] for column in columns),
        )
        report.new_static_tools_inserted += 1


def _canonicalize_bootstrap_tool_row(
    row: Mapping[str, Any],
    *,
    name: str,
    path: str,
    static_tools: _StaticToolPlan,
) -> tuple[dict[str, Any], bool]:
    expected = static_tools.new_rows[name]
    if (
        str(row.get("tool_id")) != str(expected["tool_id"])
        or str(row.get("name")) != name
        or bool(row.get("ephemeral"))
    ):
        raise ToolSkillMigrationError(f"{path} is not a canonical static core tool row")
    if str(row.get("scope")) not in {"static", _CORE_STATIC_TOOL_SCOPE} or str(
        row.get("registered_by")
    ) not in {"runtime.core", _CORE_STATIC_TOOL_SCOPE}:
        raise ToolSkillMigrationError(f"{path} has external or untrusted provenance")
    # Old releases used the same stable identity with older schemas and
    # descriptions.  Proven core rows are refreshed instead of rejected.
    _json_object(row.get("spec_json"), f"{path}.spec_json")
    canonical = deepcopy(expected)
    created_at = row.get("created_at")
    if isinstance(created_at, str) and created_at:
        canonical["created_at"] = created_at
    changed = any(
        row.get(field) != canonical[field]
        for field in ("spec_json", "scope", "registered_by", "ephemeral")
    )
    return canonical, changed


def _migrate_image_definition(
    manifest: dict[str, Any],
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> tuple[dict[str, Any], str | None]:
    selected = deepcopy(manifest)
    image_id = selected.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise ToolSkillMigrationError(f"{path}.image_id must be a non-empty string")
    metadata = selected.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ToolSkillMigrationError(f"{path}.metadata must be an object")
    default_tools = _string_list(selected.get("default_tools", []), f"{path}.default_tools")
    legacy = _validate_legacy_metadata(
        metadata,
        path=f"{path}.metadata",
        default_tools=default_tools,
        image_id=image_id,
    )
    if legacy:
        legacy_lifecycle = set(default_tools) & set(LEGACY_TOOL_GROUP_TOOLS)
        if (
            legacy_lifecycle or metadata.get("lazy_tool_groups") is True
        ) and legacy_lifecycle != set(LEGACY_TOOL_GROUP_TOOLS):
            raise ToolSkillMigrationError(
                f"{path}.default_tools must contain both removed lifecycle tools "
                "for a legacy lazy image"
            )

    current_builtin = DEFAULT_IMAGES.get(image_id)
    if current_builtin is not None:
        replacement = to_jsonable(current_builtin)
        return replacement, "builtin" if replacement != manifest else None

    if legacy:
        if "tool_projection" in metadata:
            raise ToolSkillMigrationError(
                f"{path}.metadata cannot combine legacy Tool Groups with tool_projection"
            )
        migrated_metadata = dict(metadata)
        for key in _LEGACY_METADATA_KEYS:
            migrated_metadata.pop(key, None)
        # Omission means full projection.  This is intentional for custom
        # images so an upgrade cannot silently hide an already-authorized tool.
        migrated_metadata.pop("tool_projection", None)
        selected["metadata"] = migrated_metadata
        selected["default_tools"] = _replace_tool_name_list(
            default_tools,
            path=f"{path}.default_tools",
        )
        return selected, "custom" if selected != manifest else None

    if any(name in LEGACY_TOOL_GROUP_TOOLS for name in default_tools):
        raise ToolSkillMigrationError(
            f"{path}.default_tools contains removed Tool Group lifecycle tools "
            "without legacy metadata"
        )
    _reject_legacy_ids_in_mapping(
        {name: static_tools.expected_ids.get(name, "") for name in default_tools},
        path=f"{path}.default_tools",
        static_tools=static_tools,
        check_values=False,
    )
    return selected, None


def _validate_legacy_metadata(
    metadata: dict[str, Any],
    *,
    path: str,
    default_tools: Sequence[str],
    image_id: str,
) -> bool:
    has_lazy = "lazy_tool_groups" in metadata
    has_initial = "initial_tool_groups" in metadata
    if has_lazy and type(metadata["lazy_tool_groups"]) is not bool:
        raise ToolSkillMigrationError(f"{path}.lazy_tool_groups must be a bool")
    if has_initial and metadata.get("lazy_tool_groups") is not True:
        raise ToolSkillMigrationError(
            f"{path}.initial_tool_groups requires lazy_tool_groups=true"
        )
    if has_initial:
        groups = metadata["initial_tool_groups"]
        if not isinstance(groups, list) or any(
            not isinstance(group, str) or not group.strip() for group in groups
        ):
            raise ToolSkillMigrationError(
                f"{path}.initial_tool_groups must be a list of non-empty strings"
            )
        normalized_groups = [group.strip() for group in groups]
        if len(normalized_groups) != len(set(normalized_groups)):
            raise ToolSkillMigrationError(f"{path}.initial_tool_groups contains duplicates")
        unknown = sorted(set(normalized_groups) - LEGACY_TOOL_GROUPS)
        if unknown:
            raise ToolSkillMigrationError(
                f"{path}.initial_tool_groups contains unknown groups: {unknown}"
            )
        allowed = set(default_tools)
        unauthorized = [
            group
            for group in normalized_groups
            if not allowed.intersection(_LEGACY_TOOL_GROUP_TOOLS[group])
        ]
        if unauthorized:
            raise ToolSkillMigrationError(
                f"{path}.initial_tool_groups are not authorized by image "
                f"{image_id}: {unauthorized}"
            )
    if (has_lazy or has_initial) and "tool_projection" in metadata:
        raise ToolSkillMigrationError(
            f"{path} cannot contain both legacy Tool Group fields and tool_projection"
        )
    return has_lazy or has_initial


def _replace_tool_name_list(
    names: list[str],
    *,
    path: str,
) -> list[str]:
    legacy_names = set(names) & set(LEGACY_TOOL_GROUP_TOOLS)
    if legacy_names and legacy_names != set(LEGACY_TOOL_GROUP_TOOLS):
        raise ToolSkillMigrationError(
            f"{path} must contain both removed lifecycle tools or neither"
        )
    found_legacy = bool(legacy_names)
    kept: list[str] = []
    for name in names:
        if name in LEGACY_TOOL_GROUP_TOOLS:
            continue
        if name not in kept:
            kept.append(name)
    if found_legacy:
        for name in SKILL_LIFECYCLE_TOOLS:
            if name not in kept:
                kept.append(name)
    if any(not isinstance(name, str) or not name for name in kept):
        raise ToolSkillMigrationError(f"{path} contains an invalid tool name")
    return kept


def _replace_tool_map(
    mapping: dict[str, str],
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> dict[str, str]:
    _reject_legacy_ids_in_mapping(mapping, path=path, static_tools=static_tools)
    legacy_names = set(mapping) & set(LEGACY_TOOL_GROUP_TOOLS)
    if legacy_names and legacy_names != set(LEGACY_TOOL_GROUP_TOOLS):
        raise ToolSkillMigrationError(
            f"{path} must contain both removed lifecycle bindings or neither"
        )
    found_legacy = bool(legacy_names)
    if not found_legacy:
        return dict(mapping)
    result = {
        name: tool_id
        for name, tool_id in mapping.items()
        if name not in LEGACY_TOOL_GROUP_TOOLS
    }
    for name in SKILL_LIFECYCLE_TOOLS:
        expected_id = static_tools.expected_ids[name]
        existing = result.get(name)
        if existing is not None and existing != expected_id:
            raise ToolSkillMigrationError(
                f"{path}.{name} has unexpected tool id {existing}; expected {expected_id}"
            )
        result[name] = expected_id
    return result


def _reject_legacy_ids_in_mapping(
    mapping: Mapping[str, str],
    *,
    path: str,
    static_tools: _StaticToolPlan,
    check_values: bool = True,
) -> None:
    old_ids = {
        static_tools.expected_ids[name]: name for name in LEGACY_TOOL_GROUP_TOOLS
    }
    for name, tool_id in mapping.items():
        if name in LEGACY_TOOL_GROUP_TOOLS:
            expected = static_tools.expected_ids[name]
            if tool_id != expected:
                raise ToolSkillMigrationError(
                    f"{path}.{name} has unexpected tool id {tool_id}; expected {expected}"
                )
        elif check_values and tool_id in old_ids:
            raise ToolSkillMigrationError(
                f"{path}.{name} aliases removed lifecycle id for {old_ids[tool_id]}"
            )


class _ArtifactMigrator:
    def __init__(
        self,
        *,
        cursor: Any,
        report: ToolSkillMigrationReport,
        static_tools: _StaticToolPlan,
    ) -> None:
        self.cursor = cursor
        self.report = report
        self.static_tools = static_tools
        self._global = {
            str(row["artifact_id"]): dict(row)
            for row in _rows(
                cursor,
                "SELECT * FROM image_artifacts ORDER BY artifact_id",
            )
        }
        self._entry_cache: dict[str, dict[str, Any]] = {}

    def cascade_image_artifact(
        self,
        manifest: dict[str, Any],
        *,
        path: str,
        embedded: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = deepcopy(manifest)
        boot = selected.get("boot", {"kind": "fresh"})
        if not isinstance(boot, dict):
            raise ToolSkillMigrationError(f"{path}.boot must be an object")
        kind = str(boot.get("kind") or "fresh")
        if kind not in {"checkpoint_commit", "image_package"}:
            return selected
        artifact_id = boot.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ToolSkillMigrationError(f"{path}.boot.artifact_id must be a string")
        new_id, entry = self.migrate(artifact_id, embedded=embedded, path=path)
        if str(entry.get("kind") or "") != kind:
            raise ToolSkillMigrationError(
                f"{path}.boot.kind {kind} does not match artifact kind "
                f"{entry.get('kind')}: {artifact_id}"
            )
        if kind == "image_package":
            return selected
        if new_id == artifact_id:
            return selected
        new_boot = dict(boot)
        new_boot["artifact_id"] = new_id
        new_boot["artifact_sha256"] = entry["sha256"]
        selected["boot"] = new_boot
        metadata = selected.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ToolSkillMigrationError(f"{path}.metadata must be an object")
        new_metadata = dict(metadata)
        new_metadata["artifact_sha256"] = entry["sha256"]
        new_metadata["artifact_bytes"] = len(
            dumps(entry["artifact"]).encode("utf-8")
        )
        selected["metadata"] = new_metadata
        return selected

    def migrate(
        self,
        artifact_id: str,
        *,
        embedded: dict[str, Any] | None,
        path: str,
    ) -> tuple[str, dict[str, Any]]:
        mapped = self.report.artifact_id_remaps.get(artifact_id)
        if mapped is not None:
            if embedded is not None:
                _artifact_entry(
                    row=self._global.get(artifact_id),
                    embedded=embedded,
                    artifact_id=artifact_id,
                    path=path,
                )
            return mapped, deepcopy(self._entry_cache[mapped])
        row = self._global.get(artifact_id)
        entry = _artifact_entry(row=row, embedded=embedded, artifact_id=artifact_id, path=path)
        artifact = entry["artifact"]
        kind = str(entry["kind"])
        if kind == "image_package":
            return artifact_id, entry
        if kind != "checkpoint_commit":
            raise ToolSkillMigrationError(
                f"{path} references unsupported image artifact kind {kind}: {artifact_id}"
            )
        migrated = _migrate_checkpoint_artifact(
            artifact,
            path=f"{path}.artifact[{artifact_id}]",
            static_tools=self.static_tools,
        )
        if migrated == artifact:
            return artifact_id, entry
        serialized = dumps(migrated)
        artifact_bytes = len(serialized.encode("utf-8"))
        hard_limit = self.static_tools.config.image_commit.artifact_hard_limit_bytes
        if artifact_bytes > hard_limit:
            raise ToolSkillMigrationError(
                f"migrated checkpoint artifact {artifact_id} exceeds "
                f"artifact_hard_limit_bytes={hard_limit}"
            )
        sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        new_id = f"imgart_{sha256[:24]}"
        new_entry = {
            **entry,
            "artifact_id": new_id,
            "artifact": migrated,
            "sha256": sha256,
            "metadata": {
                **dict(entry.get("metadata", {})),
                "migrated_from_artifact_id": artifact_id,
                "artifact_bytes": artifact_bytes,
            },
        }
        existing = self._global.get(new_id)
        if existing is not None:
            existing_entry = _artifact_entry(
                row=existing,
                embedded=None,
                artifact_id=new_id,
                path=path,
            )
            if (
                existing_entry["artifact"] != migrated
                or str(existing_entry["sha256"]) != sha256
                or str(existing_entry["kind"]) != "checkpoint_commit"
            ):
                raise ToolSkillMigrationError(
                    f"canonical migrated artifact id collision: {new_id}"
                )
            new_entry = existing_entry
        else:
            cursor_row = {
                "artifact_id": new_id,
                "kind": "checkpoint_commit",
                "artifact_json": serialized,
                "sha256": sha256,
                "created_by": str(entry.get("created_by") or "tool-skill-migration"),
                "created_at": str(entry.get("created_at") or self.static_tools.updated_at),
                "metadata_json": dumps(new_entry["metadata"]),
            }
            columns = tuple(cursor_row)
            self.cursor.execute(
                f"INSERT INTO image_artifacts ({', '.join(columns)}) VALUES "
                f"({', '.join('?' for _ in columns)})",
                tuple(cursor_row[column] for column in columns),
            )
            self._global[new_id] = cursor_row
            self.report.checkpoint_artifacts_created += 1
        self.report.artifact_id_remaps[artifact_id] = new_id
        self._entry_cache[new_id] = deepcopy(new_entry)
        return new_id, new_entry


def _artifact_entry(
    *,
    row: dict[str, Any] | None,
    embedded: dict[str, Any] | None,
    artifact_id: str,
    path: str,
) -> dict[str, Any]:
    global_entry = _global_artifact_entry(row, artifact_id=artifact_id)
    embedded_entry = _embedded_artifact_entry(
        embedded,
        artifact_id=artifact_id,
        path=path,
    )
    if global_entry is None and embedded_entry is None:
        raise ToolSkillMigrationError(f"{path} references missing artifact {artifact_id}")
    if global_entry is not None and embedded_entry is not None:
        if (
            global_entry["artifact"] != embedded_entry["artifact"]
            or global_entry["kind"] != embedded_entry["kind"]
            or global_entry["sha256"] != embedded_entry["sha256"]
        ):
            raise ToolSkillMigrationError(
                f"{path} embedded artifact conflicts with durable artifact {artifact_id}"
            )
    selected = deepcopy(global_entry or embedded_entry)
    assert selected is not None
    actual_sha = hashlib.sha256(dumps(selected["artifact"]).encode("utf-8")).hexdigest()
    if selected["sha256"] != actual_sha:
        raise ToolSkillMigrationError(f"{path} artifact hash mismatch: {artifact_id}")
    if not isinstance(selected["metadata"], dict):
        raise ToolSkillMigrationError(f"{path} artifact metadata must be an object")
    payload_kind = str(selected["artifact"].get("kind") or "")
    if payload_kind != str(selected["kind"]):
        raise ToolSkillMigrationError(
            f"{path} artifact row kind does not match payload kind: {artifact_id}"
        )
    return selected


def _global_artifact_entry(
    row: dict[str, Any] | None,
    *,
    artifact_id: str,
) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "artifact_id": artifact_id,
        "kind": str(row["kind"]),
        "artifact": _json_object(
            row.get("artifact_json"),
            f"image artifact {artifact_id}",
        ),
        "sha256": str(row["sha256"]),
        "created_by": str(row["created_by"]),
        "created_at": str(row["created_at"]),
        "metadata": _json_object(
            row.get("metadata_json"),
            f"image artifact {artifact_id} metadata",
        ),
    }


def _embedded_artifact_entry(
    embedded: dict[str, Any] | None,
    *,
    artifact_id: str,
    path: str,
) -> dict[str, Any] | None:
    if embedded is None:
        return None
    if not isinstance(embedded, dict):
        raise ToolSkillMigrationError(f"{path} embedded artifact must be an object")
    artifact = embedded.get("artifact")
    if not isinstance(artifact, dict):
        raise ToolSkillMigrationError(f"{path} embedded artifact payload must be an object")
    return {
        "artifact_id": artifact_id,
        "kind": str(embedded.get("kind") or artifact.get("kind") or ""),
        "artifact": deepcopy(artifact),
        "sha256": str(embedded.get("sha256") or ""),
        "created_by": str(embedded.get("created_by") or "checkpoint.restore"),
        "created_at": str(embedded.get("created_at") or ""),
        "metadata": deepcopy(embedded.get("metadata") or {}),
    }


def _migrate_checkpoint_artifact(
    artifact: dict[str, Any],
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> dict[str, Any]:
    selected = deepcopy(artifact)
    if str(selected.get("kind") or "") != "checkpoint_commit":
        raise ToolSkillMigrationError(f"{path}.kind must be checkpoint_commit")
    source_process = selected.get("source_process")
    if not isinstance(source_process, dict):
        raise ToolSkillMigrationError(f"{path}.source_process must be an object")
    migrated_process, changed = _migrate_snapshot_process_row(
        source_process,
        path=f"{path}.source_process",
        static_tools=static_tools,
    )
    selected["source_process"] = migrated_process
    top_table = selected.get("tool_table", {})
    if not isinstance(top_table, dict) or any(
        not isinstance(name, str) or not isinstance(tool_id, str)
        for name, tool_id in top_table.items()
    ):
        raise ToolSkillMigrationError(f"{path}.tool_table must be a string mapping")
    migrated_table = _replace_tool_map(
        dict(top_table),
        path=f"{path}.tool_table",
        static_tools=static_tools,
    )
    source_full = _tool_map(
        source_process.get("tool_table_json"),
        f"{path}.source_process.tool_table_json",
    )
    if dict(top_table) != source_full:
        raise ToolSkillMigrationError(
            f"{path}.tool_table does not match source_process.tool_table_json"
        )
    selected["tool_table"] = migrated_table
    static_defaults = _string_list(
        selected.get("static_default_tools", []),
        f"{path}.static_default_tools",
    )
    selected["static_default_tools"] = _replace_tool_name_list(
        static_defaults,
        path=f"{path}.static_default_tools",
    )
    rows = selected.get("rows")
    if not isinstance(rows, dict):
        raise ToolSkillMigrationError(f"{path}.rows must be an object")
    _reject_legacy_pending_action_rows(
        rows.get("llm_pending_actions", []),
        path=f"{path}.rows.llm_pending_actions",
        static_tools=static_tools,
    )
    migrated_rows = dict(rows)
    migrated_rows["tools"] = _migrate_tool_rows(
        rows.get("tools", []),
        referenced_ids=set(migrated_table.values()),
        path=f"{path}.rows.tools",
        static_tools=static_tools,
    )
    selected["rows"] = migrated_rows
    counts = selected.get("counts", {})
    if not isinstance(counts, dict):
        raise ToolSkillMigrationError(f"{path}.counts must be an object")
    migrated_counts = dict(counts)
    migrated_counts["tools"] = len(migrated_table)
    selected["counts"] = migrated_counts
    jit_sources = selected.get("jit_sources", {})
    if not isinstance(jit_sources, dict):
        raise ToolSkillMigrationError(f"{path}.jit_sources must be an object")
    legacy_ids = {
        static_tools.expected_ids[name] for name in LEGACY_TOOL_GROUP_TOOLS
    }
    if any(str(tool_id) in legacy_ids for tool_id in jit_sources):
        raise ToolSkillMigrationError(f"{path}.jit_sources references a removed static tool")
    del changed
    return selected


def _migrate_checkpoint_snapshot(
    snapshot: dict[str, Any],
    *,
    path: str,
    artifact_migrator: _ArtifactMigrator,
    static_tools: _StaticToolPlan,
) -> dict[str, Any]:
    selected = deepcopy(snapshot)
    rows = selected.get("rows")
    if not isinstance(rows, dict):
        raise ToolSkillMigrationError(f"{path}.rows must be an object")
    _reject_legacy_pending_action_rows(
        rows.get("llm_pending_actions", []),
        path=f"{path}.rows.llm_pending_actions",
        static_tools=static_tools,
    )
    migrated_processes, referenced_ids = _migrate_snapshot_process_rows(
        rows.get("processes", []),
        path=f"{path}.rows.processes",
        static_tools=static_tools,
    )
    migrated_rows = dict(rows)
    migrated_rows["processes"] = migrated_processes
    migrated_rows["tools"] = _migrate_tool_rows(
        rows.get("tools", []),
        referenced_ids=referenced_ids,
        path=f"{path}.rows.tools",
        static_tools=static_tools,
    )
    selected["rows"] = migrated_rows
    selected["images"], selected["image_artifacts"] = (
        _migrate_snapshot_images_and_artifacts(
            selected.get("images", {}),
            selected.get("image_artifacts", {}),
            path=path,
            artifact_migrator=artifact_migrator,
            static_tools=static_tools,
        )
    )
    jit_sources = selected.get("jit_sources", {})
    if not isinstance(jit_sources, dict):
        raise ToolSkillMigrationError(f"{path}.jit_sources must be an object")
    legacy_ids = {
        static_tools.expected_ids[name] for name in LEGACY_TOOL_GROUP_TOOLS
    }
    if any(str(tool_id) in legacy_ids for tool_id in jit_sources):
        raise ToolSkillMigrationError(f"{path}.jit_sources references a removed static tool")
    return selected


def _migrate_snapshot_process_rows(
    process_rows: Any,
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not isinstance(process_rows, list) or any(
        not isinstance(row, dict) for row in process_rows
    ):
        raise ToolSkillMigrationError(f"{path} must be a list of objects")
    migrated_processes: list[dict[str, Any]] = []
    referenced_ids: set[str] = set()
    for index, row in enumerate(process_rows):
        migrated, _changed = _migrate_snapshot_process_row(
            row,
            path=f"{path}[{index}]",
            static_tools=static_tools,
        )
        migrated_processes.append(migrated)
        referenced_ids.update(
            _tool_map(
                migrated.get("tool_table_json"),
                f"{path}[{index}].tool_table_json",
            ).values()
        )
    return migrated_processes, referenced_ids


def _migrate_snapshot_images_and_artifacts(
    images: Any,
    embedded_artifacts: Any,
    *,
    path: str,
    artifact_migrator: _ArtifactMigrator,
    static_tools: _StaticToolPlan,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(images, dict):
        raise ToolSkillMigrationError(f"{path}.images must be an object")
    migrated_images: dict[str, dict[str, Any]] = {}
    for image_id, manifest in images.items():
        if not isinstance(image_id, str) or not isinstance(manifest, dict):
            raise ToolSkillMigrationError(f"{path}.images must map ids to objects")
        if manifest.get("image_id") != image_id:
            raise ToolSkillMigrationError(
                f"{path}.images[{image_id}] key does not match manifest.image_id"
            )
        migrated, _kind = _migrate_image_definition(
            manifest,
            path=f"{path}.images[{image_id}]",
            static_tools=static_tools,
        )
        if (
            _kind == "custom"
        ):
            fallback_warning = (
                f"{path}.images[{image_id}]: custom legacy image now uses full tool "
                "projection; set metadata.tool_projection='skills' explicitly "
                "after selecting default_skills"
            )
            if fallback_warning not in artifact_migrator.report.warnings:
                artifact_migrator.report.warnings.append(fallback_warning)
            if _image_boot_kind(manifest) == "image_package":
                package_warning = (
                    f"{path}.images[{image_id}]: preserved immutable image_package raw files"
                )
                if package_warning not in artifact_migrator.report.warnings:
                    artifact_migrator.report.warnings.append(package_warning)
        migrated_images[image_id] = migrated

    if not isinstance(embedded_artifacts, dict):
        raise ToolSkillMigrationError(f"{path}.image_artifacts must be an object")
    migrated_artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id, entry in embedded_artifacts.items():
        if not isinstance(artifact_id, str):
            raise ToolSkillMigrationError(f"{path}.image_artifacts keys must be strings")
        new_id, migrated_entry = artifact_migrator.migrate(
            artifact_id,
            embedded=entry,
            path=f"{path}.image_artifacts[{artifact_id}]",
        )
        migrated_artifacts[new_id] = migrated_entry

    for image_id, manifest in list(migrated_images.items()):
        boot = manifest.get("boot", {})
        old_artifact_id = str(boot.get("artifact_id") or "") if isinstance(boot, dict) else ""
        embedded = embedded_artifacts.get(old_artifact_id)
        migrated_images[image_id] = artifact_migrator.cascade_image_artifact(
            manifest,
            path=f"{path}.images[{image_id}]",
            embedded=embedded,
        )
    return migrated_images, migrated_artifacts


def _migrate_snapshot_process_row(
    row: dict[str, Any],
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> tuple[dict[str, Any], bool]:
    selected = deepcopy(row)
    full = _tool_map(selected.get("tool_table_json"), f"{path}.tool_table_json")
    model = _tool_map(
        selected.get("model_tool_table_json"),
        f"{path}.model_tool_table_json",
    )
    _validate_model_tool_subset(full, model, path=path)
    migrated_full = _replace_tool_map(
        full,
        path=f"{path}.tool_table",
        static_tools=static_tools,
    )
    migrated_model = _replace_tool_map(
        model,
        path=f"{path}.model_tool_table",
        static_tools=static_tools,
    )
    selected["tool_table_json"] = dumps(migrated_full)
    selected["model_tool_table_json"] = dumps(migrated_model)
    return selected, migrated_full != full or migrated_model != model


def _migrate_tool_rows(
    rows: Any,
    *,
    referenced_ids: set[str],
    path: str,
    static_tools: _StaticToolPlan,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ToolSkillMigrationError(f"{path} must be a list of objects")
    old_ids = {
        static_tools.expected_ids[name] for name in LEGACY_TOOL_GROUP_TOOLS
    }
    bootstrap_by_id = {
        static_tools.expected_ids[name]: name
        for name in SKILL_PROJECTION_BOOTSTRAP
    }
    selected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        tool_id = row.get("tool_id")
        name = row.get("name")
        if not isinstance(tool_id, str) or not isinstance(name, str):
            raise ToolSkillMigrationError(f"{path}[{index}] has invalid tool identity")
        if tool_id in old_ids or name in LEGACY_TOOL_GROUP_TOOLS:
            expected = static_tools.expected_ids.get(name)
            if expected != tool_id:
                raise ToolSkillMigrationError(f"{path}[{index}] has mismatched legacy identity")
            continue
        bootstrap_name = (
            name
            if name in SKILL_PROJECTION_BOOTSTRAP
            else bootstrap_by_id.get(tool_id)
        )
        if bootstrap_name is not None:
            row, _changed = _canonicalize_bootstrap_tool_row(
                row,
                name=bootstrap_name,
                path=f"{path}[{index}]",
                static_tools=static_tools,
            )
        if tool_id in selected and selected[tool_id] != row:
            raise ToolSkillMigrationError(f"{path} contains conflicting rows for {tool_id}")
        selected[tool_id] = deepcopy(row)
    _add_referenced_bootstrap_rows(
        selected,
        referenced_ids=referenced_ids,
        path=path,
        static_tools=static_tools,
    )
    return [selected[tool_id] for tool_id in sorted(selected)]


def _add_referenced_bootstrap_rows(
    selected: dict[str, dict[str, Any]],
    *,
    referenced_ids: set[str],
    path: str,
    static_tools: _StaticToolPlan,
) -> None:
    for name in SKILL_PROJECTION_BOOTSTRAP:
        tool_id = static_tools.expected_ids[name]
        if tool_id not in referenced_ids:
            continue
        existing = selected.get(tool_id)
        if existing is not None:
            if str(existing.get("name")) != name:
                raise ToolSkillMigrationError(f"{path} contains a tool id collision for {tool_id}")
            continue
        selected[tool_id] = deepcopy(static_tools.new_rows[name])


def _validate_model_tool_subset(
    full: Mapping[str, str],
    model: Mapping[str, str],
    *,
    path: str,
) -> None:
    mismatches = {
        name: {"full": full.get(name), "model": tool_id}
        for name, tool_id in model.items()
        if full.get(name) != tool_id
    }
    if mismatches:
        raise ToolSkillMigrationError(
            f"{path}.model_tool_table must be an exact-ID subset of tool_table: "
            f"{mismatches}"
        )


def _reserve_migration_process_revision(
    cursor: Any,
    *,
    pid: str,
    current_revision: int,
) -> int:
    counter_name = f"process_revision:{pid}"
    cursor.execute(
        "INSERT INTO runtime_counters (counter_name, value) VALUES (?, ?) "
        "ON CONFLICT(counter_name) DO NOTHING",
        (counter_name, max(0, current_revision)),
    )
    cursor.execute(
        "UPDATE runtime_counters SET value = CASE WHEN value < ? THEN ? ELSE value END "
        "WHERE counter_name = ?",
        (current_revision, current_revision, counter_name),
    )
    cursor.execute(
        "UPDATE runtime_counters SET value = value + 1 WHERE counter_name = ?",
        (counter_name,),
    )
    rows = _rows(
        cursor,
        "SELECT value FROM runtime_counters WHERE counter_name = ?",
        (counter_name,),
    )
    if len(rows) != 1 or int(rows[0]["value"]) <= current_revision:
        raise ToolSkillMigrationError(
            f"failed to reserve a monotonic process revision for {pid}"
        )
    return int(rows[0]["value"])


def _replace_process_bindings(
    cursor: Any,
    *,
    pid: str,
    full: Mapping[str, str],
    model: Mapping[str, str],
) -> None:
    cursor.execute("DELETE FROM process_tool_bindings WHERE pid = ?", (pid,))
    rows = [
        (pid, kind, name, tool_id, 0)
        for kind, mapping in (("callable", full), ("model", model))
        for name, tool_id in mapping.items()
    ]
    if rows:
        cursor.executemany(
            "INSERT INTO process_tool_bindings "
            "(pid, binding_kind, tool_name, tool_id, jit_rehydration_eligible) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        cursor.execute(
            "UPDATE process_tool_bindings SET jit_rehydration_eligible = 1 "
            "WHERE pid = ? AND binding_kind = 'callable' AND EXISTS ("
            "SELECT 1 FROM tools WHERE tools.tool_id = process_tool_bindings.tool_id "
            "AND tools.ephemeral = 1)",
            (pid,),
        )


def _preflight_inflight_activity(cursor: Any, static_tools: _StaticToolPlan) -> None:
    tokens = _legacy_tokens(static_tools)
    for row in _rows(cursor, "SELECT * FROM llm_pending_actions ORDER BY pid"):
        if _contains_token(
            _pending_action_structure(
                row,
                path=f"llm_pending_actions[{row['pid']}]",
            ),
            tokens,
        ):
            raise ToolSkillMigrationError(
                f"pending LLM action for {row['pid']} references removed Tool Group tools"
            )
    for row in _rows(cursor, "SELECT * FROM runtime_publications ORDER BY publication_id"):
        publication = dict(row)
        operation_reconciled = _persisted_boolean(
            publication.get("operation_reconciled"),
            path=(
                "runtime_publications["
                f"{publication.get('publication_id')!r}].operation_reconciled"
            ),
        )
        unresolved = (
            str(publication.get("state")) not in _TERMINAL_PUBLICATION_STATES
            or not operation_reconciled
        )
        if unresolved and _contains_token(
            {
                "plan": _json_value(publication.get("plan_json"), "runtime publication plan"),
                "receipt": _json_value(publication.get("receipt_json"), "runtime publication receipt"),
            },
            tokens,
        ):
            raise ToolSkillMigrationError(
                "unresolved runtime publication references removed Tool Group tools: "
                f"{publication['publication_id']}"
            )
    for row in _rows(
        cursor,
        "SELECT task_id, tool, tool_id, status FROM object_tasks ORDER BY task_id",
    ):
        if str(row.get("status")) in _TERMINAL_OBJECT_TASK_STATES:
            continue
        if str(row.get("tool")) in tokens or str(row.get("tool_id")) in tokens:
            raise ToolSkillMigrationError(
                f"active object task references removed Tool Group tool: {row['task_id']}"
            )


def _preflight_checkpoint_restore_references(
    cursor: Any,
    *,
    checkpoint_ids: frozenset[str],
    static_tools: _StaticToolPlan,
) -> None:
    """Fence restore recovery before mutating its checkpoint digest anchor."""

    tokens = _legacy_tokens(static_tools)
    for row in _rows(
        cursor,
        "SELECT publications.*, attempts.state AS exact_delivery_attempt_state "
        "FROM runtime_publications AS publications "
        "LEFT JOIN checkpoint_payload_delivery_attempts AS attempts ON "
        "attempts.attempt_id = publications.payload_delivery_attempt_id AND "
        "attempts.owner_instance_id = publications.owner_instance_id AND "
        "attempts.started_at = publications.payload_delivery_started_at "
        "WHERE publications.kind = 'checkpoint_restore' "
        "ORDER BY publications.publication_id",
    ):
        plan = _json_value(
            row.get("plan_json"),
            "checkpoint restore publication plan",
        )
        payload = {
            "plan": plan,
            "receipt": _json_value(
                row.get("receipt_json"),
                "checkpoint restore publication receipt",
            ),
        }
        planned_checkpoint_id = (
            plan.get("checkpoint_id") if isinstance(plan, Mapping) else None
        )
        changes_planned_snapshot = (
            isinstance(planned_checkpoint_id, str)
            and planned_checkpoint_id in checkpoint_ids
        )
        operation_reconciled = _persisted_boolean(
            row.get("operation_reconciled"),
            path=(
                "runtime_publications["
                f"{row.get('publication_id')!r}].operation_reconciled"
            ),
        )
        unresolved = (
            str(row.get("state")) not in _TERMINAL_PUBLICATION_STATES
            or not operation_reconciled
        )
        if unresolved and changes_planned_snapshot:
            raise ToolSkillMigrationError(
                "unresolved checkpoint restore references a checkpoint snapshot "
                "requiring migration: "
                f"{row['publication_id']} -> {planned_checkpoint_id}"
            )
        delivery_incomplete = not _checkpoint_payload_delivery_is_resolved(row)
        if delivery_incomplete and (
            _contains_token(payload, tokens) or changes_planned_snapshot
        ):
            raise ToolSkillMigrationError(
                "incomplete checkpoint payload delivery references state requiring migration: "
                f"{row['publication_id']}"
            )


def _checkpoint_payload_delivery_is_resolved(
    publication: Mapping[str, Any],
) -> bool:
    delivery_state = publication.get("payload_delivery_state")
    if delivery_state is None:
        return True
    if delivery_state != "completed":
        return False
    return publication.get("exact_delivery_attempt_state") == "acked"


def _assert_mutable_references_clean(
    cursor: Any,
    *,
    static_tools: _StaticToolPlan,
    allow_planned_checkpoint_artifacts: frozenset[str],
) -> None:
    del allow_planned_checkpoint_artifacts
    tokens = _legacy_tokens(static_tools)
    for row in _rows(cursor, "SELECT pid, tool_table_json, model_tool_table_json FROM processes"):
        for field in ("tool_table_json", "model_tool_table_json"):
            if _contains_token(_json_value(row[field], field), tokens):
                raise ToolSkillMigrationError(
                    f"postcondition failed: process {row['pid']} retains removed lifecycle reference"
                )
    for row in _rows(cursor, "SELECT pid, binding_kind, tool_name, tool_id FROM process_tool_bindings"):
        if str(row["tool_name"]) in tokens or str(row["tool_id"]) in tokens:
            raise ToolSkillMigrationError(
                f"postcondition failed: process binding for {row['pid']} retains removed lifecycle reference"
            )
    for row in _rows(cursor, "SELECT image_id, manifest_json FROM images"):
        manifest = _json_object(row["manifest_json"], f"image {row['image_id']} manifest")
        _assert_image_definition_clean(
            manifest,
            expected_image_id=str(row["image_id"]),
            path=f"images[{row['image_id']}]",
            tokens=tokens,
        )
    for row in _rows(cursor, "SELECT checkpoint_id, snapshot_json FROM checkpoints"):
        checkpoint_id = str(row["checkpoint_id"])
        _assert_checkpoint_snapshot_clean(
            _json_object(
                row["snapshot_json"],
                f"checkpoint {checkpoint_id} snapshot",
            ),
            path=f"checkpoints[{checkpoint_id}]",
            tokens=tokens,
        )


def _assert_image_definition_clean(
    manifest: Mapping[str, Any],
    *,
    expected_image_id: str,
    path: str,
    tokens: frozenset[str],
) -> None:
    if manifest.get("image_id") != expected_image_id:
        raise ToolSkillMigrationError(
            f"postcondition failed: {path} key does not match manifest.image_id"
        )
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ToolSkillMigrationError(f"postcondition failed: {path}.metadata is invalid")
    if _LEGACY_METADATA_KEYS & set(metadata):
        raise ToolSkillMigrationError(
            f"postcondition failed: {path} retains legacy metadata"
        )
    if _contains_token(manifest.get("default_tools", []), tokens):
        raise ToolSkillMigrationError(
            f"postcondition failed: {path} retains removed lifecycle tool"
        )


def _assert_checkpoint_snapshot_clean(
    snapshot: Mapping[str, Any],
    *,
    path: str,
    tokens: frozenset[str],
) -> None:
    rows = snapshot.get("rows", {})
    if not isinstance(rows, dict):
        raise ToolSkillMigrationError(f"postcondition failed: {path}.rows is invalid")
    _assert_checkpoint_pending_actions_clean(
        rows.get("llm_pending_actions", []),
        path=f"{path}.rows.llm_pending_actions",
        tokens=tokens,
    )

    for index, process in enumerate(rows.get("processes", [])):
        if not isinstance(process, dict):
            raise ToolSkillMigrationError(
                f"postcondition failed: {path}.rows.processes[{index}] is invalid"
            )
        for field in ("tool_table_json", "model_tool_table_json"):
            if _contains_token(
                _json_value(process.get(field), f"{path}.{field}"),
                tokens,
            ):
                raise ToolSkillMigrationError(
                    f"postcondition failed: {path}.rows.processes[{index}] "
                    "retains removed lifecycle reference"
                )
    for index, tool in enumerate(rows.get("tools", [])):
        if not isinstance(tool, dict) or _contains_token(
            {"name": tool.get("name"), "tool_id": tool.get("tool_id")}
            if isinstance(tool, dict)
            else tool,
            tokens,
        ):
            raise ToolSkillMigrationError(
                f"postcondition failed: {path}.rows.tools[{index}] is invalid or legacy"
            )
    images = snapshot.get("images", {})
    if not isinstance(images, dict):
        raise ToolSkillMigrationError(f"postcondition failed: {path}.images is invalid")
    for image_id, manifest in images.items():
        if not isinstance(image_id, str) or not isinstance(manifest, dict):
            raise ToolSkillMigrationError(f"postcondition failed: {path}.images is invalid")
        _assert_image_definition_clean(
            manifest,
            expected_image_id=image_id,
            path=f"{path}.images[{image_id}]",
            tokens=tokens,
        )
    artifacts = snapshot.get("image_artifacts", {})
    if not isinstance(artifacts, dict):
        raise ToolSkillMigrationError(
            f"postcondition failed: {path}.image_artifacts is invalid"
        )
    for artifact_id, entry in artifacts.items():
        if not isinstance(entry, dict):
            raise ToolSkillMigrationError(
                f"postcondition failed: {path}.image_artifacts[{artifact_id}] is invalid"
            )
        artifact = entry.get("artifact", {})
        if isinstance(artifact, dict) and artifact.get("kind") == "checkpoint_commit":
            _assert_checkpoint_artifact_clean(
                artifact,
                path=f"{path}.image_artifacts[{artifact_id}]",
                tokens=tokens,
            )


def _assert_checkpoint_pending_actions_clean(
    pending_actions: Any,
    *,
    path: str,
    tokens: frozenset[str],
) -> None:
    if not isinstance(pending_actions, list) or any(
        not isinstance(action, dict) for action in pending_actions
    ):
        raise ToolSkillMigrationError(
            f"postcondition failed: {path} is invalid"
        )
    for index, action in enumerate(pending_actions):
        if _contains_token(
            _pending_action_structure(
                action,
                path=f"{path}[{index}]",
            ),
            tokens,
        ):
            raise ToolSkillMigrationError(
                f"postcondition failed: {path}[{index}] "
                "retains removed lifecycle reference"
            )


def _assert_checkpoint_artifact_clean(
    artifact: Mapping[str, Any],
    *,
    path: str,
    tokens: frozenset[str],
) -> None:
    structural = {
        "tool_table": artifact.get("tool_table"),
        "static_default_tools": artifact.get("static_default_tools"),
        "source_process_tool_table": (
            artifact.get("source_process", {}).get("tool_table_json")
            if isinstance(artifact.get("source_process"), dict)
            else None
        ),
        "source_process_model_tool_table": (
            artifact.get("source_process", {}).get("model_tool_table_json")
            if isinstance(artifact.get("source_process"), dict)
            else None
        ),
        "tool_rows": [
            {"name": row.get("name"), "tool_id": row.get("tool_id")}
            for row in artifact.get("rows", {}).get("tools", [])
            if isinstance(row, dict)
        ] if isinstance(artifact.get("rows"), dict) else None,
    }
    for field in ("source_process_tool_table", "source_process_model_tool_table"):
        raw = structural[field]
        if isinstance(raw, str):
            structural[field] = _json_value(raw, f"{path}.{field}")
    if _contains_token(structural, tokens):
        raise ToolSkillMigrationError(
            f"postcondition failed: {path} retains removed lifecycle reference"
        )


def _delete_old_static_tools(
    cursor: Any,
    static_tools: _StaticToolPlan,
    report: ToolSkillMigrationReport,
) -> None:
    for name in LEGACY_TOOL_GROUP_TOOLS:
        tool_id = static_tools.expected_ids[name]
        cursor.execute(
            "DELETE FROM tools WHERE tool_id = ? AND name = ? AND ephemeral = 0",
            (tool_id, name),
        )
        if int(cursor.rowcount) > 0:
            report.old_static_tools_deleted += int(cursor.rowcount)


def _legacy_tokens(static_tools: _StaticToolPlan) -> frozenset[str]:
    return frozenset(
        {
            *LEGACY_TOOL_GROUP_TOOLS,
            *(static_tools.expected_ids[name] for name in LEGACY_TOOL_GROUP_TOOLS),
        }
    )


def _pending_action_structure(row: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    selected = {
        key: row.get(key)
        for key in (
            "pid",
            "tool_name",
            "wait_type",
            "tool_call_id",
            "tool_operation_id",
        )
    }
    for field in ("filters_json", "action_json", "data_flow_context_json"):
        raw = row.get(field)
        selected[field] = _json_value(raw, f"{path}.{field}")
    return selected


def _reject_legacy_pending_action_rows(
    rows: Any,
    *,
    path: str,
    static_tools: _StaticToolPlan,
) -> None:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ToolSkillMigrationError(f"{path} must be a list of objects")
    tokens = _legacy_tokens(static_tools)
    for index, row in enumerate(rows):
        if _contains_token(
            _pending_action_structure(row, path=f"{path}[{index}]"),
            tokens,
        ):
            raise ToolSkillMigrationError(
                f"{path}[{index}] references a removed tool"
            )


def _contains_token(value: Any, tokens: Iterable[str]) -> bool:
    selected = frozenset(tokens)
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if current in selected:
                return True
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            pending.extend(current)
    return False


def _image_boot_kind(manifest: Mapping[str, Any]) -> str:
    boot = manifest.get("boot", {})
    return str(boot.get("kind") or "fresh") if isinstance(boot, dict) else ""


def _rows(cursor: Any, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.execute(sql, tuple(params))]


def _json_value(value: Any, path: str) -> Any:
    if not isinstance(value, str):
        raise ToolSkillMigrationError(f"{path} must contain JSON text")
    try:
        return bounded_json_loads(
            value,
            max_bytes=_LEGACY_JSON_HARD_LIMIT_BYTES,
        )
    except (TypeError, ValueError) as exc:
        raise ToolSkillMigrationError(f"{path} contains malformed JSON") from exc


def _persisted_boolean(value: Any, *, path: str) -> bool:
    """Decode a backend boolean without laundering malformed persisted values."""

    if type(value) is bool:
        return value
    if type(value) is int and value in {0, 1}:
        return value == 1
    raise ToolSkillMigrationError(f"{path} must be stored as 0 or 1")


def _json_object(value: Any, path: str) -> dict[str, Any]:
    selected = _json_value(value, path) if isinstance(value, str) else value
    if not isinstance(selected, dict):
        raise ToolSkillMigrationError(f"{path} must be a JSON object")
    return deepcopy(selected)


def _tool_map(value: Any, path: str) -> dict[str, str]:
    selected = _json_value(value, path) if isinstance(value, str) else value
    if not isinstance(selected, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(tool_id, str)
        or not tool_id
        for name, tool_id in selected.items()
    ):
        raise ToolSkillMigrationError(f"{path} must be a non-empty-string mapping")
    return dict(selected)


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ToolSkillMigrationError(f"{path} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ToolSkillMigrationError(f"{path} contains duplicate entries")
    return list(value)


__all__ = [
    "LEGACY_TOOL_GROUPS",
    "LEGACY_TOOL_GROUP_TOOLS",
    "SKILL_LIFECYCLE_TOOLS",
    "SKILL_PROJECTION_BOOTSTRAP",
    "ToolSkillMigrationError",
    "ToolSkillMigrationReport",
    "cli",
    "migrate_tool_groups_to_skills",
]
