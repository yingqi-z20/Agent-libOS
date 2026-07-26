from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from agent_libos.utils.yaml_loader import load_yaml_mapping
from benchmarks.runtime_safety.models import (
    BENCHMARK_EFFECT_OBSERVATION_FIELDS,
    BenchmarkTask,
    BenchmarkValidationError,
    VALID_EFFECT_OUTCOMES,
    VALID_EFFECT_TYPES,
)

REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "title",
    "goal",
    "workspace",
    "attack_class",
    "allowed_effects",
    "forbidden_effects",
    "success_oracle",
    "safety_oracle",
}
OPTIONAL_FIELDS = {
    "setup",
    "capabilities",
    "policy",
    "human_responses",
    "expected_audit",
    "mock_actions",
    "notes",
}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
SAFETY_ORACLE_TYPES = {"no_forbidden_effects", "no_unknown_effects"}
SUCCESS_ORACLE_TYPES = {
    "action_succeeded",
    "image_requires_capability",
    "object_exists",
    "process_exited",
    "expected_effects",
    "file_contains",
    "file_exists",
    "python_syntax_valid",
    "managed_git_worktree",
    "git_patch_artifact_lineage",
}
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def load_tasks(suite_root: str | Path) -> list[BenchmarkTask]:
    root = Path(suite_root)
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        raise BenchmarkValidationError(f"benchmark tasks directory does not exist: {tasks_dir}")
    tasks = [load_task_file(path) for path in sorted(tasks_dir.glob("*.yaml"))]
    seen: set[str] = set()
    duplicates: list[str] = []
    for task in tasks:
        if task.id in seen:
            duplicates.append(task.id)
        seen.add(task.id)
    if duplicates:
        raise BenchmarkValidationError(f"duplicate benchmark task ids: {sorted(set(duplicates))}")
    return tasks


def load_task_file(path: str | Path) -> BenchmarkTask:
    source = Path(path)
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    data = load_yaml_mapping(source_bytes.decode("utf-8"))
    _validate_required(data, source)
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise BenchmarkValidationError(f"{source}: unsupported schema_version {schema_version!r}")
    task_id = _string_field(data, "id", source)
    if not _TASK_ID_RE.match(task_id):
        raise BenchmarkValidationError(f"{source}: id must be lowercase snake_case, got {task_id!r}")
    allowed = _validate_effect_list(data.get("allowed_effects"), source, "allowed_effects")
    forbidden = _validate_effect_list(data.get("forbidden_effects"), source, "forbidden_effects")
    mock_actions = _validate_mapping_list(data.get("mock_actions", []), source, "mock_actions")
    setup = _optional_mapping(data.get("setup", {}), source, "setup")
    success_oracle_items = _validate_mapping_list(
        data.get("success_oracle"), source, "success_oracle"
    )
    # Resolve source-object identity before validating the remaining lineage
    # fields so an ambiguous provenance anchor cannot be hidden behind a later
    # shape error (for example, a missing changed-path constraint).
    _validate_git_source_references(setup, success_oracle_items, source)
    success_oracle = _validate_success_oracle(success_oracle_items, source)
    safety_oracle = _validate_safety_oracle(data.get("safety_oracle"), source)
    for index, action in enumerate(mock_actions):
        if not isinstance(action.get("action"), str) or not action["action"]:
            raise BenchmarkValidationError(f"{source}: mock_actions[{index}] requires non-empty action")
        _validate_action_paths(action, source, index)
    _validate_success_action_references(success_oracle, mock_actions, source)
    _validate_success_setup_references(setup, success_oracle, source)
    return BenchmarkTask(
        id=task_id,
        title=_string_field(data, "title", source),
        goal=_string_field(data, "goal", source),
        workspace=_safe_relative_path(_string_field(data, "workspace", source), source, "workspace"),
        attack_class=_string_field(data, "attack_class", source),
        allowed_effects=allowed,
        forbidden_effects=forbidden,
        success_oracle=success_oracle,
        safety_oracle=safety_oracle,
        schema_version=1,
        setup=setup,
        capabilities=_optional_mapping(data.get("capabilities", {}), source, "capabilities"),
        policy=_optional_mapping(data.get("policy", {}), source, "policy"),
        human_responses=_validate_mapping_list(data.get("human_responses", []), source, "human_responses"),
        expected_audit=_validate_mapping_list(data.get("expected_audit", []), source, "expected_audit"),
        mock_actions=mock_actions,
        notes=_optional_string(data.get("notes"), source, "notes"),
        source_path=source,
        source_sha256=source_sha256,
    )


def _validate_required(data: dict[str, Any], source: Path) -> None:
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise BenchmarkValidationError(f"{source}: missing required fields: {missing}")
    unknown = sorted(
        repr(key)
        for key in data
        if not isinstance(key, str) or key not in ALLOWED_FIELDS
    )
    if unknown:
        raise BenchmarkValidationError(f"{source}: unknown top-level fields: {unknown}")


def _optional_string(value: Any, source: Path, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkValidationError(f"{source}: {field} must be a string")
    return value


def _string_field(data: dict[str, Any], field: str, source: Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(f"{source}: {field} must be a non-empty string")
    return value.strip()


def _optional_mapping(value: Any, source: Path, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{source}: {field} must be a mapping")
    return value


def _validate_mapping_list(value: Any, source: Path, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise BenchmarkValidationError(f"{source}: {field} must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"{source}: {field}[{index}] must be a mapping")
        result.append(dict(item))
    return result


def _validate_effect_list(value: Any, source: Path, field: str) -> list[dict[str, Any]]:
    effects = _validate_mapping_list(value, source, field)
    for index, effect in enumerate(effects):
        effect_type = effect.get("type")
        if not isinstance(effect_type, str) or effect_type not in VALID_EFFECT_TYPES:
            raise BenchmarkValidationError(
                f"{source}: {field}[{index}].type must be one of {sorted(VALID_EFFECT_TYPES)}, got {effect_type!r}"
            )
        if str(effect_type).startswith("filesystem."):
            if "path" not in effect:
                raise BenchmarkValidationError(f"{source}: {field}[{index}] requires path")
            effect["path"] = _safe_relative_path(str(effect["path"]), source, f"{field}[{index}].path")
            _validate_match_mode(
                effect,
                source,
                f"{field}[{index}].match",
                allowed={"exact", "prefix", "glob"},
            )
            has_wildcard = any(marker in effect["path"] for marker in ("*", "?", "["))
            if has_wildcard and effect.get("match") != "glob":
                raise BenchmarkValidationError(
                    f"{source}: {field}[{index}] wildcard paths require match: glob"
                )
        if effect_type == "shell.exec":
            _validate_argv(effect.get("argv"), source, f"{field}[{index}].argv")
            _validate_match_mode(
                effect,
                source,
                f"{field}[{index}].match",
                allowed={"exact", "prefix"},
            )
        if str(effect_type).startswith("object."):
            namespace = effect.get("namespace")
            if namespace is not None and (not isinstance(namespace, str) or ".." in namespace.replace("\\", "/").split("/")):
                raise BenchmarkValidationError(f"{source}: {field}[{index}].namespace is unsafe")
        if effect_type in {"process.spawn", "process.fork", "process.exec"} and "image" in effect:
            if not isinstance(effect["image"], str) or not effect["image"]:
                raise BenchmarkValidationError(f"{source}: {field}[{index}].image must be a string")
        if effect_type == "skill.activate":
            _validate_non_empty_string(effect, "skill_id", source, f"{field}[{index}].skill_id")
        if effect_type == "jit.register":
            _validate_non_empty_string(effect, "tool", source, f"{field}[{index}].tool")
        if effect_type in {"image.register", "image.commit"}:
            _validate_non_empty_string(effect, "image", source, f"{field}[{index}].image")
        if effect_type in {"checkpoint.create", "checkpoint.fork"}:
            _validate_non_empty_string(effect, "checkpoint", source, f"{field}[{index}].checkpoint", required=False)
        if effect_type == "jsonrpc.call":
            _validate_non_empty_string(effect, "endpoint", source, f"{field}[{index}].endpoint")
            _validate_non_empty_string(effect, "method", source, f"{field}[{index}].method")
    return effects


def _validate_success_oracle(value: Any, source: Path) -> list[dict[str, Any]]:
    checks = _validate_mapping_list(value, source, "success_oracle")
    if not checks:
        raise BenchmarkValidationError(f"{source}: success_oracle must be non-empty")
    for index, check in enumerate(checks):
        check_type = check.get("type")
        if not isinstance(check_type, str) or check_type not in SUCCESS_ORACLE_TYPES:
            raise BenchmarkValidationError(
                f"{source}: success_oracle[{index}].type must be one of "
                f"{sorted(SUCCESS_ORACLE_TYPES)}, got {check_type!r}"
            )
        if check_type in {"file_contains", "file_exists", "python_syntax_valid"}:
            path = _string_field(check, "path", source)
            check["path"] = _safe_relative_path(
                path,
                source,
                f"success_oracle[{index}].path",
            )
            if check_type == "file_contains" and not isinstance(check.get("text"), str):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].text must be a string"
                )
            continue
        if check_type == "object_exists":
            _validate_non_empty_string(
                check,
                "namespace",
                source,
                f"success_oracle[{index}].namespace",
            )
            _validate_non_empty_string(
                check,
                "name",
                source,
                f"success_oracle[{index}].name",
            )
            if ".." in str(check["namespace"]).replace("\\", "/").split("/"):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].namespace is unsafe"
                )
            continue
        if check_type == "image_requires_capability":
            for key in ("image", "resource"):
                _validate_non_empty_string(
                    check,
                    key,
                    source,
                    f"success_oracle[{index}].{key}",
                )
            rights = check.get("rights")
            if (
                not isinstance(rights, list)
                or not rights
                or any(not isinstance(right, str) or not right for right in rights)
                or len(rights) != len(set(rights))
            ):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].rights must be a "
                    "non-empty unique string list"
                )
            continue
        if check_type == "action_succeeded":
            _validate_non_empty_string(
                check,
                "action",
                source,
                f"success_oracle[{index}].action",
            )
            arguments = check.get("arguments", {})
            if not isinstance(arguments, dict):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].arguments must be a mapping"
                )
            check["arguments"] = dict(arguments)
            continue
        if check_type == "managed_git_worktree":
            root = check.get("root")
            if root is not None:
                check["root"] = _safe_relative_path(
                    str(root),
                    source,
                    f"success_oracle[{index}].root",
                )
            continue
        if check_type == "git_patch_artifact_lineage":
            for key in ("source_object", "sensitivity"):
                _validate_non_empty_string(
                    check,
                    key,
                    source,
                    f"success_oracle[{index}].{key}",
                )
            _validate_non_empty_string(
                check,
                "artifact_origin",
                source,
                f"success_oracle[{index}].artifact_origin",
                required=False,
            )
            _validate_non_empty_string(
                check,
                "source_origin",
                source,
                f"success_oracle[{index}].source_origin",
                required=False,
            )
            min_bytes = check.get("min_bytes", 1)
            if (
                isinstance(min_bytes, bool)
                or not isinstance(min_bytes, int)
                or min_bytes < 1
            ):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].min_bytes must be positive"
                )
            check["min_bytes"] = min_bytes
            changed_paths = check.get("changed_paths_exact")
            if not isinstance(changed_paths, list) or not changed_paths:
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].changed_paths_exact "
                    "must be non-empty"
                )
            normalized_paths = [
                _safe_relative_path(
                    str(path),
                    source,
                    f"success_oracle[{index}].changed_paths_exact",
                )
                for path in changed_paths
            ]
            if len(normalized_paths) != len(set(normalized_paths)):
                raise BenchmarkValidationError(
                    f"{source}: success_oracle[{index}].changed_paths_exact "
                    "must be unique"
                )
            check["changed_paths_exact"] = normalized_paths
            continue
        if check_type != "expected_effects":
            continue
        field = f"success_oracle[{index}].effects"
        effects = _validate_effect_list(check.get("effects"), source, field)
        if not effects:
            raise BenchmarkValidationError(f"{source}: {field} must be non-empty")
        for effect_index, effect in enumerate(effects):
            outcomes = effect.get("outcomes", ["performed"])
            if not isinstance(outcomes, list) or not outcomes:
                raise BenchmarkValidationError(
                    f"{source}: {field}[{effect_index}].outcomes must be a non-empty list"
                )
            invalid = [
                outcome
                for outcome in outcomes
                if not isinstance(outcome, str)
                or outcome not in VALID_EFFECT_OUTCOMES
            ]
            if invalid:
                raise BenchmarkValidationError(
                    f"{source}: {field}[{effect_index}].outcomes contains invalid values {invalid!r}"
                )
            effect["outcomes"] = list(dict.fromkeys(outcomes))
        check["effects"] = effects
    return checks


def _validate_safety_oracle(value: Any, source: Path) -> list[dict[str, Any]]:
    checks = _validate_mapping_list(value, source, "safety_oracle")
    if not checks:
        raise BenchmarkValidationError(f"{source}: safety_oracle must be non-empty")
    seen: set[str] = set()
    for index, check in enumerate(checks):
        check_type = check.get("type")
        if not isinstance(check_type, str) or check_type not in SAFETY_ORACLE_TYPES:
            raise BenchmarkValidationError(
                f"{source}: safety_oracle[{index}].type must be one of "
                f"{sorted(SAFETY_ORACLE_TYPES)}, got {check_type!r}"
            )
        unknown = sorted(
            repr(key)
            for key in check
            if not isinstance(key, str) or key != "type"
        )
        if unknown:
            raise BenchmarkValidationError(
                f"{source}: safety_oracle[{index}] has unknown fields: {unknown}"
            )
        if check_type in seen:
            raise BenchmarkValidationError(
                f"{source}: duplicate safety oracle type {check_type!r}"
            )
        seen.add(check_type)
    if "no_unknown_effects" not in seen:
        raise BenchmarkValidationError(
            f"{source}: safety_oracle must include no_unknown_effects"
        )
    return checks


def _validate_git_source_references(
    setup: dict[str, Any],
    success_oracle: list[dict[str, Any]],
    source: Path,
) -> None:
    """Require name-based Git lineage references to resolve unambiguously."""

    raw_objects = setup.get("memory_objects", []) or []
    if not isinstance(raw_objects, list):
        raise BenchmarkValidationError(f"{source}: setup.memory_objects must be a list")
    object_names = [
        item.get("name")
        for item in raw_objects
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    references: list[tuple[str, Any]] = []
    git_setup = setup.get("git", {}) or {}
    if not isinstance(git_setup, dict):
        raise BenchmarkValidationError(f"{source}: setup.git must be a mapping")
    file_labels = git_setup.get("file_labels", []) or []
    if not isinstance(file_labels, list):
        raise BenchmarkValidationError(f"{source}: setup.git.file_labels must be a list")
    for index, label in enumerate(file_labels):
        if not isinstance(label, dict):
            raise BenchmarkValidationError(
                f"{source}: setup.git.file_labels[{index}] must be a mapping"
            )
        if "source_object" in label:
            references.append(
                (f"setup.git.file_labels[{index}].source_object", label.get("source_object"))
            )
    for index, check in enumerate(success_oracle):
        if check.get("type") == "git_patch_artifact_lineage":
            references.append(
                (f"success_oracle[{index}].source_object", check.get("source_object"))
            )
    for field, name in references:
        if not isinstance(name, str) or not name.strip():
            raise BenchmarkValidationError(f"{source}: {field} must be a non-empty string")
        matches = sum(candidate == name for candidate in object_names)
        if matches != 1:
            raise BenchmarkValidationError(
                f"{source}: {field} must reference exactly one setup.memory_objects "
                f"entry named {name!r}; found {matches}"
            )


def _validate_match_mode(
    effect: dict[str, Any],
    source: Path,
    field: str,
    *,
    allowed: set[str],
) -> None:
    if "match" not in effect:
        return
    value = effect.get("match")
    if not isinstance(value, str) or value not in allowed:
        raise BenchmarkValidationError(
            f"{source}: {field} match must be one of {sorted(allowed)}, got {value!r}"
        )


def _validate_action_paths(action: dict[str, Any], source: Path, index: int) -> None:
    name = str(action.get("action"))
    process_goal = action.get("process_goal")
    if process_goal is not None and (
        not isinstance(process_goal, str) or not process_goal.strip()
    ):
        raise BenchmarkValidationError(
            f"{source}: mock_actions[{index}].process_goal must be a non-empty string"
        )
    if name in {"read_text_file", "write_text_file", "delete_file", "delete_directory", "read_directory", "write_directory", "skill_syscall_read"}:
        if "path" not in action:
            raise BenchmarkValidationError(f"{source}: mock_actions[{index}] {name} requires path")
        action["path"] = _safe_relative_path(str(action["path"]), source, f"mock_actions[{index}].path")
    if name == "run_shell_command":
        _validate_argv(action.get("argv"), source, f"mock_actions[{index}].argv")
    effects = action.get("benchmark_effects")
    if name == "skill_syscall_read" and effects is None:
        raise BenchmarkValidationError(
            f"{source}: mock_actions[{index}].benchmark_effects must contain "
            "exactly one exact filesystem.read bound to the "
            "skill_syscall_read path"
        )
    if effects is not None:
        field = f"mock_actions[{index}].benchmark_effects"
        validated_effects = _validate_effect_list(effects, source, field)
        for effect_index, effect in enumerate(validated_effects):
            observed = sorted(BENCHMARK_EFFECT_OBSERVATION_FIELDS & set(effect))
            if observed:
                raise BenchmarkValidationError(
                    f"{source}: {field}[{effect_index}] may not declare "
                    f"runner-observed fields: {observed}"
                )
        action["benchmark_effects"] = validated_effects
        if name == "skill_syscall_read":
            if (
                len(validated_effects) != 1
                or validated_effects[0].get("type") != "filesystem.read"
                or validated_effects[0].get("path") != action.get("path")
                or validated_effects[0].get("match", "exact") != "exact"
            ):
                raise BenchmarkValidationError(
                    f"{source}: {field} must contain exactly one exact "
                    "filesystem.read bound to the skill_syscall_read path"
                )
    checkpoint_ref = action.get("checkpoint_ref")
    if checkpoint_ref is not None and (not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip()):
        raise BenchmarkValidationError(f"{source}: mock_actions[{index}].checkpoint_ref must be a non-empty string")


def _validate_success_action_references(
    checks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    source: Path,
) -> None:
    for index, check in enumerate(checks):
        if check.get("type") != "action_succeeded":
            continue
        expected_action = check.get("action")
        expected_arguments = check.get("arguments", {})
        matches = [
            action
            for action in actions
            if action.get("action") == expected_action
            and all(action.get(key) == value for key, value in expected_arguments.items())
        ]
        if len(matches) != 1:
            raise BenchmarkValidationError(
                f"{source}: success_oracle[{index}] must identify exactly one "
                "mock action"
            )


def _validate_success_setup_references(
    setup: dict[str, Any],
    checks: list[dict[str, Any]],
    source: Path,
) -> None:
    memory_objects = setup.get("memory_objects", []) or []
    for index, check in enumerate(checks):
        if check.get("type") != "object_exists":
            continue
        matches = [
            item
            for item in memory_objects
            if isinstance(item, dict)
            and str(item.get("namespace") or "process") == check.get("namespace")
            and str(item.get("name") or "") == check.get("name")
        ]
        if len(matches) != 1:
            raise BenchmarkValidationError(
                f"{source}: success_oracle[{index}] must identify exactly one "
                "setup.memory_objects target"
            )


def _safe_relative_path(value: str, source: Path, field: str) -> str:
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise BenchmarkValidationError(f"{source}: {field} must be non-empty")
    if raw.startswith("/") or raw.startswith("~") or re.match(r"^[A-Za-z]:", raw):
        raise BenchmarkValidationError(f"{source}: {field} must be workspace-relative: {value!r}")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise BenchmarkValidationError(f"{source}: {field} may not escape workspace: {value!r}")
    return "/".join(parts) if parts else "."


def _validate_argv(value: Any, source: Path, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{source}: {field} must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise BenchmarkValidationError(f"{source}: {field}[{index}] must be a string")
        if index == 0 and not item.strip():
            raise BenchmarkValidationError(f"{source}: {field}[0] must be non-empty")
        if "\x00" in item:
            raise BenchmarkValidationError(f"{source}: {field}[{index}] may not contain NUL")


def _validate_non_empty_string(
    mapping: dict[str, Any],
    key: str,
    source: Path,
    field: str,
    *,
    required: bool = True,
) -> None:
    if key not in mapping:
        if required:
            raise BenchmarkValidationError(f"{source}: {field} is required")
        return
    if not isinstance(mapping[key], str) or not mapping[key].strip():
        raise BenchmarkValidationError(f"{source}: {field} must be a non-empty string")
