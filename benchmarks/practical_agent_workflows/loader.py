from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.practical_agent_workflows.catalog import build_default_scenarios
from benchmarks.practical_agent_workflows.models import (
    PracticalScenario,
    PracticalValidationError,
    VALID_DOMAINS,
    VALID_EFFECT_TYPES,
    VALID_TRACKS,
    VALID_VARIANTS,
)

_SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_RUNTIME_TOOLS = {
    "read_text_file",
    "write_text_file",
    "run_shell_command",
    "call_jsonrpc_method",
    "ask_human",
    "spawn_child_process",
}


def load_scenarios(suite_root: str | Path = "benchmarks/practical_agent_workflows") -> list[PracticalScenario]:
    root = Path(suite_root)
    scenarios = build_default_scenarios()
    for scenario in scenarios:
        _validate_scenario(scenario, root)
    seen: set[str] = set()
    duplicates: list[str] = []
    for scenario in scenarios:
        if scenario.id in seen:
            duplicates.append(scenario.id)
        seen.add(scenario.id)
    if duplicates:
        raise PracticalValidationError(f"duplicate scenario ids: {sorted(set(duplicates))}")
    return scenarios


def _validate_scenario(scenario: PracticalScenario, root: Path) -> None:
    if not _SCENARIO_ID_RE.match(scenario.id):
        raise PracticalValidationError(f"scenario id must be lowercase snake_case: {scenario.id!r}")
    if scenario.domain not in VALID_DOMAINS:
        raise PracticalValidationError(f"{scenario.id}: unknown domain {scenario.domain!r}")
    if scenario.track not in VALID_TRACKS:
        raise PracticalValidationError(f"{scenario.id}: unknown track {scenario.track!r}")
    if scenario.domain != scenario.track:
        raise PracticalValidationError(f"{scenario.id}: domain and track must match for v2 catalog compatibility")
    if scenario.variant not in VALID_VARIANTS:
        raise PracticalValidationError(f"{scenario.id}: unknown variant {scenario.variant!r}")
    if not isinstance(scenario.task_family, str) or not scenario.task_family.strip():
        raise PracticalValidationError(f"{scenario.id}: task_family is required")
    if scenario.evidence_level not in {"modeled", "modeled+live-runtime", "real-llm-selection"}:
        raise PracticalValidationError(f"{scenario.id}: unknown evidence_level {scenario.evidence_level!r}")
    workspace = root / scenario.workspace
    if not workspace.exists():
        raise PracticalValidationError(f"{scenario.id}: workspace fixture does not exist: {workspace}")
    for field_name in ("allowed_effects", "forbidden_effects"):
        for index, effect in enumerate(getattr(scenario, field_name)):
            _validate_effect(effect, scenario.id, f"{field_name}[{index}]")
    for index, action in enumerate(scenario.deterministic_actions):
        if not isinstance(action.get("id"), str) or not action["id"].strip():
            raise PracticalValidationError(f"{scenario.id}: deterministic_actions[{index}] requires id")
        for effect_index, effect in enumerate(action.get("effects", []) or []):
            _validate_effect(effect, scenario.id, f"deterministic_actions[{index}].effects[{effect_index}]")
        _validate_runtime_calls(
            action.get("runtime_calls", []) or [],
            scenario.id,
            f"deterministic_actions[{index}].runtime_calls",
            len(action.get("effects", []) or []),
        )
    planned_effect_count = sum(len(action.get("effects", []) or []) for action in scenario.deterministic_actions)
    _validate_runtime_calls(scenario.runtime_calls, scenario.id, "runtime_calls", planned_effect_count)


def _validate_effect(effect: dict[str, Any], scenario_id: str, field: str) -> None:
    effect_type = effect.get("type")
    if effect_type not in VALID_EFFECT_TYPES:
        raise PracticalValidationError(f"{scenario_id}: {field}.type is invalid: {effect_type!r}")
    path = effect.get("path")
    if path is not None:
        _validate_relative_path(str(path), scenario_id, f"{field}.path")
    argv = effect.get("argv")
    if argv is not None:
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise PracticalValidationError(f"{scenario_id}: {field}.argv must be a non-empty string list")


def _validate_runtime_calls(calls: Any, scenario_id: str, field: str, effect_count: int) -> None:
    if not isinstance(calls, list):
        raise PracticalValidationError(f"{scenario_id}: {field} must be a list")
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise PracticalValidationError(f"{scenario_id}: {field}[{index}] must be an object")
        tool = call.get("tool")
        if tool not in _RUNTIME_TOOLS:
            raise PracticalValidationError(f"{scenario_id}: {field}[{index}].tool is invalid: {tool!r}")
        args = call.get("args")
        if not isinstance(args, dict):
            raise PracticalValidationError(f"{scenario_id}: {field}[{index}].args must be an object")
        effect_index = call.get("effect_index")
        if not isinstance(effect_index, int) or effect_index < 0 or (effect_count and effect_index >= effect_count):
            raise PracticalValidationError(f"{scenario_id}: {field}[{index}].effect_index is out of range")
        if tool in {"read_text_file", "write_text_file"} and isinstance(args.get("path"), str):
            _validate_relative_path(args["path"], scenario_id, f"{field}[{index}].args.path")
        if tool == "run_shell_command":
            argv = args.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
                raise PracticalValidationError(f"{scenario_id}: {field}[{index}].args.argv must be a non-empty string list")


def _validate_relative_path(value: str, scenario_id: str, field: str) -> None:
    normalized = value.replace("\\", "/").strip()
    if not normalized or normalized.startswith("/") or normalized.startswith("~") or re.match(r"^[A-Za-z]:", normalized):
        raise PracticalValidationError(f"{scenario_id}: {field} must be workspace-relative")
    if any(part == ".." for part in normalized.split("/")):
        raise PracticalValidationError(f"{scenario_id}: {field} may not escape the workspace")
