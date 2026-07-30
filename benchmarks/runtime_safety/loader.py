from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from agent_libos.utils.yaml_loader import load_yaml_mapping
from benchmarks.runtime_safety.models import (
    BENCHMARK_EFFECT_OBSERVATION_FIELDS,
    BenchmarkTask,
    BenchmarkValidationError,
    VALID_EFFECT_TYPES,
)
from benchmarks.runtime_safety.schemas import (
    EFFECT_IDENTITY_FIELDS,
    EXPECTED_EFFECT_OUTCOMES,
    MOCK_ACTION_SCHEMAS,
    POLICY_DEFAULTS,
    policy_schema,
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
    try:
        validated_root = root.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkValidationError(
            f"benchmark suite root does not exist: {root}"
        ) from exc
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        raise BenchmarkValidationError(f"benchmark tasks directory does not exist: {tasks_dir}")
    tasks = [
        load_task_file(path, suite_root=validated_root)
        for path in sorted(tasks_dir.glob("*.yaml"))
    ]
    seen: set[str] = set()
    duplicates: list[str] = []
    for task in tasks:
        if task.id in seen:
            duplicates.append(task.id)
        seen.add(task.id)
    if duplicates:
        raise BenchmarkValidationError(f"duplicate benchmark task ids: {sorted(set(duplicates))}")
    return tasks


def load_task_file(
    path: str | Path,
    *,
    suite_root: str | Path | None = None,
) -> BenchmarkTask:
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
    mock_actions = _validate_mock_actions(data.get("mock_actions", []), source)
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
        _validate_action_paths(action, source, index)
    _validate_success_action_references(success_oracle, mock_actions, source)
    _validate_success_setup_references(setup, success_oracle, source)
    workspace = _safe_relative_path(
        _string_field(data, "workspace", source),
        source,
        "workspace",
    )
    if suite_root is not None:
        _validate_suite_workspace(workspace, source, Path(suite_root))
    return BenchmarkTask(
        id=task_id,
        title=_string_field(data, "title", source),
        goal=_string_field(data, "goal", source),
        workspace=workspace,
        attack_class=_string_field(data, "attack_class", source),
        allowed_effects=allowed,
        forbidden_effects=forbidden,
        success_oracle=success_oracle,
        safety_oracle=safety_oracle,
        schema_version=1,
        setup=setup,
        capabilities=_optional_mapping(data.get("capabilities", {}), source, "capabilities"),
        policy=_validate_policy(data.get("policy", {}), source),
        human_responses=_validate_mapping_list(data.get("human_responses", []), source, "human_responses"),
        expected_audit=_validate_mapping_list(data.get("expected_audit", []), source, "expected_audit"),
        mock_actions=mock_actions,
        notes=_optional_string(data.get("notes"), source, "notes"),
        source_path=source,
        source_sha256=source_sha256,
    )


def validate_task_execution_inputs(task: BenchmarkTask) -> None:
    """Validate untrusted task fields consumed directly by a runner.

    ``BenchmarkTask`` remains a convenient dataclass for targeted tests and
    integrations, so callers can construct one without using the YAML loader.
    Execution must nevertheless enforce the same closed action and policy
    contracts; otherwise a typo can become a successful wrapper no-op.
    """

    source = task.source_path or Path(f"<benchmark-task:{task.id}>")
    actions = _validate_mock_actions(task.mock_actions, source)
    for index, action in enumerate(actions):
        _validate_action_paths(action, source, index)
    _validate_policy(task.policy, source)


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


def _validate_mock_actions(
    value: Any,
    source: Path,
) -> list[dict[str, Any]]:
    actions = _validate_mapping_list(value, source, "mock_actions")
    for index, action in enumerate(actions):
        raw_effects = action.get("benchmark_effects")
        if isinstance(raw_effects, list):
            for effect_index, effect in enumerate(raw_effects):
                if not isinstance(effect, dict):
                    continue
                observed = sorted(BENCHMARK_EFFECT_OBSERVATION_FIELDS & set(effect))
                if observed:
                    raise BenchmarkValidationError(
                        f"{source}: mock_actions[{index}].benchmark_effects"
                        f"[{effect_index}] may not declare runner-observed fields: "
                        f"{observed}"
                    )
        name = action.get("action")
        if not isinstance(name, str) or not name:
            raise BenchmarkValidationError(
                f"{source}: mock_actions[{index}] requires non-empty action"
            )
        if name == "skill_syscall_read":
            if (
                not isinstance(raw_effects, list)
                or len(raw_effects) != 1
                or not isinstance(raw_effects[0], dict)
                or raw_effects[0].get("type") != "filesystem.read"
                or raw_effects[0].get("match", "exact") != "exact"
            ):
                raise BenchmarkValidationError(
                    f"{source}: mock_actions[{index}].benchmark_effects must "
                    "contain exactly one exact filesystem.read bound to the "
                    "skill_syscall_read path"
                )
        schema = MOCK_ACTION_SCHEMAS.get(name)
        if schema is None:
            raise BenchmarkValidationError(
                f"{source}: mock_actions[{index}].action must be one of "
                f"{sorted(MOCK_ACTION_SCHEMAS)}, got {name!r}"
            )
        errors = sorted(
            Draft202012Validator(schema).iter_errors(action),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
        if errors:
            error = errors[0]
            suffix = "".join(f"[{item!r}]" for item in error.absolute_path)
            raise BenchmarkValidationError(
                f"{source}: mock_actions[{index}]{suffix} violates the "
                f"{name} contract: {error.message}"
            )
    return actions


def _validate_policy(value: Any, source: Path) -> dict[str, Any]:
    policy = _optional_mapping(value, source, "policy")
    errors = sorted(
        Draft202012Validator(policy_schema()).iter_errors(policy),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        suffix = "".join(f"[{item!r}]" for item in error.absolute_path)
        raise BenchmarkValidationError(
            f"{source}: policy{suffix} violates the closed policy contract: "
            f"{error.message}"
        )
    return {**POLICY_DEFAULTS, **policy}


def _validate_effect_list(
    value: Any,
    source: Path,
    field: str,
    *,
    allow_outcomes: bool = False,
) -> list[dict[str, Any]]:
    effects = _validate_mapping_list(value, source, field)
    for index, effect in enumerate(effects):
        effect_type = effect.get("type")
        if not isinstance(effect_type, str) or effect_type not in VALID_EFFECT_TYPES:
            raise BenchmarkValidationError(
                f"{source}: {field}[{index}].type must be one of {sorted(VALID_EFFECT_TYPES)}, got {effect_type!r}"
            )
        required_fields, optional_fields = EFFECT_IDENTITY_FIELDS[effect_type]
        allowed_fields = {
            "type",
            *required_fields,
            *optional_fields,
            *({"outcomes"} if allow_outcomes else set()),
        }
        unknown = sorted(
            repr(key)
            for key in effect
            if not isinstance(key, str) or key not in allowed_fields
        )
        if unknown:
            raise BenchmarkValidationError(
                f"{source}: {field}[{index}] has unknown fields: {unknown}"
            )
        missing = sorted(required_fields - set(effect))
        if missing:
            raise BenchmarkValidationError(
                f"{source}: {field}[{index}] is missing required identity fields: "
                f"{missing}"
            )
        for identity_field in sorted(
            (required_fields | optional_fields) - {"argv", "match"}
        ):
            _validate_non_empty_string(
                effect,
                identity_field,
                source,
                f"{field}[{index}].{identity_field}",
                required=identity_field in required_fields,
            )
        if str(effect_type).startswith("filesystem."):
            effect["path"] = _safe_relative_path(
                effect["path"],
                source,
                f"{field}[{index}].path",
            )
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
            if namespace is not None and ".." in namespace.replace("\\", "/").split("/"):
                raise BenchmarkValidationError(f"{source}: {field}[{index}].namespace is unsafe")
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
        allowed_fields = {
            "process_exited": {"type"},
            "file_contains": {"type", "path", "text"},
            "file_exists": {"type", "path"},
            "python_syntax_valid": {"type", "path"},
            "object_exists": {"type", "namespace", "name"},
            "action_succeeded": {"type", "action", "arguments"},
            "image_requires_capability": {"type", "image", "resource", "rights"},
            "managed_git_worktree": {"type", "root"},
            "git_patch_artifact_lineage": {
                "type",
                "source_object",
                "sensitivity",
                "artifact_origin",
                "source_origin",
                "min_bytes",
                "changed_paths_exact",
            },
            "expected_effects": {"type", "effects"},
        }[check_type]
        unknown = sorted(
            repr(key)
            for key in check
            if not isinstance(key, str) or key not in allowed_fields
        )
        if unknown:
            raise BenchmarkValidationError(
                f"{source}: success_oracle[{index}] has unknown fields: {unknown}"
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
        effects = _validate_effect_list(
            check.get("effects"),
            source,
            field,
            allow_outcomes=True,
        )
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
                or outcome not in EXPECTED_EFFECT_OUTCOMES
            ]
            if invalid:
                raise BenchmarkValidationError(
                    f"{source}: {field}[{effect_index}].outcomes contains invalid values "
                    f"that cannot prove an observed effect: {invalid!r}; expected one "
                    f"of {sorted(EXPECTED_EFFECT_OUTCOMES)}"
                )
            if len(outcomes) != len(set(outcomes)):
                raise BenchmarkValidationError(
                    f"{source}: {field}[{effect_index}].outcomes must be unique"
                )
            effect["outcomes"] = list(outcomes)
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
    if name in {
        "read_text_file",
        "write_text_file",
        "delete_file",
        "delete_directory",
        "skill_syscall_read",
        "load_image_package",
    }:
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


def _validate_suite_workspace(
    workspace: str,
    source: Path,
    suite_root: Path,
) -> None:
    try:
        root = suite_root.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkValidationError(
            f"{source}: benchmark suite root does not exist: {suite_root}"
        ) from exc
    candidate = root / workspace
    if candidate.is_symlink():
        raise BenchmarkValidationError(
            f"{source}: workspace must be a non-symlink directory relative to "
            f"the suite root: {workspace!r}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BenchmarkValidationError(
            f"{source}: workspace must name an existing directory relative to "
            f"the suite root: {workspace!r}"
        ) from exc
    if not resolved.is_dir() or (
        resolved != root and root not in resolved.parents
    ):
        raise BenchmarkValidationError(
            f"{source}: workspace must name a directory inside the suite root: "
            f"{workspace!r}"
        )


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
