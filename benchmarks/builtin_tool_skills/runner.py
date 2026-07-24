from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from agent_libos import Runtime
from agent_libos.llm.usage import aggregate_cache_usage
from agent_libos.models import CapabilityRight, ProcessStatus
from agent_libos.skills import get_builtin_skill_catalog
from agent_libos.substrate import LocalResourceProviderSubstrate


EVALUATION_REPETITIONS = 3
WITH_SKILLS = "with_skills"
WITHOUT_SKILLS = "without_skills"
EVALUATION_VARIANTS = (WITH_SKILLS, WITHOUT_SKILLS)
_IMAGE_IDS = {
    WITH_SKILLS: "builtin-tool-skill-evaluator:v0",
    WITHOUT_SKILLS: "builtin-tool-skill-full-projection-baseline:v0",
}
_IMAGE_TEMPLATE_ID = "coding-agent:v0"
_MAX_QUANTA = 8
_SKILL_LIFECYCLE_TOOLS = frozenset(
    {"discover_skills", "activate_skill", "read_skill_resource", "unload_skill"}
)


@dataclass(frozen=True)
class HeldOutScenario:
    scenario_id: str
    goal: str
    expected_skill_id: str
    expected_probe_tool: str
    adjacent_skill_ids: tuple[str, ...]
    setup_kind: str


# The goals describe user intent and boundaries without exposing the expected
# Skill id or the exact probe tool name. This keeps routing behavior, rather
# than string copying, as the evaluated decision.
HELD_OUT_SCENARIOS: tuple[HeldOutScenario, ...] = (
    HeldOutScenario(
        scenario_id="ordinary_workspace_write",
        goal=(
            "Machine-only evaluation: create the ordinary workspace file skill-eval.txt "
            "with exactly `routed` followed by one newline. This is a normal workspace "
            "edit, not an Object Memory payload transfer. Use the smallest appropriate "
            "built-in capability, perform the operation, verify completion, and exit "
            "without calling human_output."
        ),
        expected_skill_id="agent-libos-workspace-editing",
        expected_probe_tool="write_text_file",
        adjacent_skill_ids=("agent-libos-object-file-transfer",),
        setup_kind="workspace_write",
    ),
    HeldOutScenario(
        scenario_id="read_only_git_state",
        goal=(
            "Machine-only evaluation: inspect the current repository change state and "
            "report whether it is clean. Do not stage, commit, switch branches, or use a "
            "shell command. Use the smallest appropriate built-in capability for the "
            "read-only inspection, perform it, then exit without calling human_output."
        ),
        expected_skill_id="agent-libos-git-inspection",
        expected_probe_tool="git_status",
        adjacent_skill_ids=(
            "agent-libos-git-change-recording",
            "agent-libos-git-branches-worktrees",
            "agent-libos-command-execution",
        ),
        setup_kind="git_read",
    ),
    HeldOutScenario(
        scenario_id="approved_git_status_command",
        goal=(
            "Machine-only evaluation: run the approved argv-only command `git status "
            "--short` and report its output. The request explicitly requires live command "
            "execution rather than the typed Git inspection interface. Use the smallest "
            "appropriate built-in capability, run the command, and exit without calling "
            "human_output."
        ),
        expected_skill_id="agent-libos-command-execution",
        expected_probe_tool="run_shell_command",
        adjacent_skill_ids=("agent-libos-git-inspection",),
        setup_kind="shell_git_read",
    ),
    HeldOutScenario(
        scenario_id="capture_process_checkpoint",
        goal=(
            "Machine-only evaluation: capture the current process's reconstructable "
            "internal state as a recovery point. Do not build, load, or commit an Agent "
            "Image. Use the smallest appropriate built-in capability, create the recovery "
            "point with a concise reason, and exit without calling human_output."
        ),
        expected_skill_id="agent-libos-checkpoints",
        expected_probe_tool="create_checkpoint",
        adjacent_skill_ids=("agent-libos-agent-images",),
        setup_kind="checkpoint",
    ),
    HeldOutScenario(
        scenario_id="cached_mcp_registry",
        goal=(
            "Machine-only evaluation: list Host-registered MCP server metadata using only "
            "the local registry. Do not refresh tools, contact a server, or inspect a plain "
            "JSON-RPC endpoint. Use the smallest appropriate built-in capability, list the "
            "metadata even if it is empty, and exit without calling human_output."
        ),
        expected_skill_id="agent-libos-mcp",
        expected_probe_tool="list_mcp_servers",
        adjacent_skill_ids=("agent-libos-jsonrpc",),
        setup_kind="mcp_registry_read",
    ),
)


def run_evaluation(
    workspace_root: str | Path,
    *,
    scenario_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run three paired Skills/full-projection trials for every scenario."""

    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scenarios = _select_scenarios(scenario_ids)
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for repetition in range(1, EVALUATION_REPETITIONS + 1):
            pair_id = f"{scenario.scenario_id}:{repetition}"
            for variant in EVALUATION_VARIANTS:
                workspace = (
                    root
                    / scenario.scenario_id
                    / f"pair-{repetition}"
                    / variant
                )
                workspace.mkdir(parents=True, exist_ok=False)
                runs.append(
                    _run_once(
                        scenario,
                        repetition,
                        workspace,
                        variant=variant,
                        pair_id=pair_id,
                    )
                )
    return {
        "schema_version": 2,
        "evaluation": "builtin_tool_skill_routing",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_ids": dict(_IMAGE_IDS),
        "variants": list(EVALUATION_VARIANTS),
        "repetitions_per_scenario": EVALUATION_REPETITIONS,
        "pairs_per_scenario": EVALUATION_REPETITIONS,
        "scenario_count": len(scenarios),
        "scenario_catalog": [asdict(scenario) for scenario in scenarios],
        "runs": runs,
        "metrics": aggregate_runs(runs),
    }


def aggregate_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(runs)
    total = len(selected)
    skill_runs = [run for run in selected if _variant(run) == WITH_SKILLS]
    correct = sum(bool(run.get("correct_skill_activation")) for run in skill_runs)
    completed = sum(bool(run.get("completed")) for run in selected)
    successful = sum(bool(run.get("task_outcome_success")) for run in selected)
    correct_routes = sum(bool(run.get("correct_route")) for run in selected)
    invalid = sum(
        _plain_non_negative_int(run.get("invalid_tool_calls")) for run in selected
    )
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(
        {str(run.get("scenario_id") or "") for run in selected}
    ):
        scenario_runs = [
            run for run in selected if run.get("scenario_id") == scenario_id
        ]
        scenario_correct = sum(
            bool(run.get("correct_route")) for run in scenario_runs
        )
        by_scenario[scenario_id] = {
            "runs": len(scenario_runs),
            "completed_runs": sum(
                bool(run.get("completed")) for run in scenario_runs
            ),
            "successful_task_outcomes": sum(
                bool(run.get("task_outcome_success")) for run in scenario_runs
            ),
            "correct_routes": scenario_correct,
            "correct_route_rate": _rate(scenario_correct, len(scenario_runs)),
            "invalid_tool_calls": sum(
                _plain_non_negative_int(run.get("invalid_tool_calls"))
                for run in scenario_runs
            ),
            "by_variant": {
                variant: _aggregate_variant(
                    [run for run in scenario_runs if _variant(run) == variant]
                )
                for variant in EVALUATION_VARIANTS
            },
        }
    by_variant = {
        variant: _aggregate_variant(
            [run for run in selected if _variant(run) == variant]
        )
        for variant in EVALUATION_VARIANTS
    }
    cache_metrics = _aggregate_run_cache_metrics(selected)
    return {
        "runs": total,
        "completed_runs": completed,
        "successful_task_outcomes": successful,
        "task_outcome_success_rate": _rate(successful, total),
        "correct_routes": correct_routes,
        "correct_route_rate": _rate(correct_routes, total),
        "correct_skill_activation_eligible_runs": len(skill_runs),
        "correct_skill_activations": correct,
        "correct_skill_activation_rate": _rate(correct, len(skill_runs)),
        "invalid_tool_calls": invalid,
        "mean_llm_calls": _mean(selected, "llm_calls"),
        "mean_catalog_metadata_bytes": _mean(
            selected, "catalog_metadata_bytes"
        ),
        "mean_initial_schema_bytes": _mean(selected, "initial_schema_bytes"),
        "mean_authorized_schema_bytes": _mean(selected, "authorized_schema_bytes"),
        "mean_cumulative_schema_bytes": _mean(
            selected, "cumulative_schema_bytes"
        ),
        "mean_initial_schema_token_estimate": _mean(
            selected, "initial_schema_token_estimate"
        ),
        "mean_authorized_schema_token_estimate": _mean(
            selected, "authorized_schema_token_estimate"
        ),
        "mean_cumulative_schema_token_estimate": _mean(
            selected, "cumulative_schema_token_estimate"
        ),
        "mean_prompt_tokens": _mean(selected, "prompt_tokens"),
        "mean_cumulative_prompt_bytes": _mean(
            selected, "cumulative_prompt_bytes"
        ),
        "mean_initial_projection_reduction_rate": _mean(
            selected, "initial_projection_reduction_rate"
        ),
        **cache_metrics,
        "by_variant": by_variant,
        "comparison": _comparison(by_variant),
        "by_scenario": by_scenario,
    }


def report_all_correct(report: dict[str, Any]) -> bool:
    """Require verified task state/results and correct routing for every pair arm."""

    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        return False
    pairs: dict[str, set[str]] = {}
    for run in runs:
        if not isinstance(run, dict):
            return False
        pair_id = run.get("pair_id")
        variant = _variant(run)
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or variant not in EVALUATION_VARIANTS
        ):
            return False
        variants = pairs.setdefault(pair_id, set())
        if variant in variants:
            return False
        variants.add(variant)
        if not (
            run.get("task_outcome_success") is True
            and run.get("completed") is True
            and run.get("correct_route") is True
            and run.get("probe_tool_result_success") is True
        ):
            return False
        if (
            _variant(run) == WITH_SKILLS
            and run.get("correct_skill_activation") is not True
        ):
            return False
    return all(variants == set(EVALUATION_VARIANTS) for variants in pairs.values())


def _aggregate_variant(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    successful = sum(bool(run.get("task_outcome_success")) for run in runs)
    correct_routes = sum(bool(run.get("correct_route")) for run in runs)
    eligible = [
        run for run in runs if isinstance(run.get("correct_skill_activation"), bool)
    ]
    correct_activations = sum(
        bool(run.get("correct_skill_activation")) for run in eligible
    )
    return {
        "runs": total,
        "completed_runs": sum(bool(run.get("completed")) for run in runs),
        "successful_task_outcomes": successful,
        "task_outcome_success_rate": _rate(successful, total),
        "correct_routes": correct_routes,
        "correct_route_rate": _rate(correct_routes, total),
        "correct_skill_activation_eligible_runs": len(eligible),
        "correct_skill_activations": correct_activations,
        "correct_skill_activation_rate": _rate(correct_activations, len(eligible)),
        "mean_invalid_tool_calls": _mean(runs, "invalid_tool_calls"),
        "mean_initial_schema_bytes": _mean(runs, "initial_schema_bytes"),
        "mean_cumulative_schema_bytes": _mean(runs, "cumulative_schema_bytes"),
        "mean_initial_schema_token_estimate": _mean(
            runs, "initial_schema_token_estimate"
        ),
        "mean_cumulative_schema_token_estimate": _mean(
            runs, "cumulative_schema_token_estimate"
        ),
        "mean_prompt_tokens": _mean(runs, "prompt_tokens"),
        "mean_cumulative_prompt_bytes": _mean(runs, "cumulative_prompt_bytes"),
        "mean_catalog_metadata_bytes": _mean(runs, "catalog_metadata_bytes"),
        **_aggregate_run_cache_metrics(runs),
    }


def _comparison(by_variant: dict[str, dict[str, Any]]) -> dict[str, Any]:
    skills = by_variant[WITH_SKILLS]
    baseline = by_variant[WITHOUT_SKILLS]
    keys = (
        "task_outcome_success_rate",
        "correct_route_rate",
        "mean_invalid_tool_calls",
        "mean_initial_schema_bytes",
        "mean_cumulative_schema_bytes",
        "mean_initial_schema_token_estimate",
        "mean_cumulative_schema_token_estimate",
        "mean_prompt_tokens",
        "mean_cumulative_prompt_bytes",
        "mean_catalog_metadata_bytes",
    )
    return {
        "baseline": WITHOUT_SKILLS,
        "treatment": WITH_SKILLS,
        "treatment_correct_skill_activation_rate": skills[
            "correct_skill_activation_rate"
        ],
        "baseline_expected_tool_route_rate": baseline["correct_route_rate"],
        "with_skills_minus_without_skills": {
            key: float(skills[key]) - float(baseline[key]) for key in keys
        },
    }


def _run_once(
    scenario: HeldOutScenario,
    repetition: int,
    workspace: Path,
    *,
    variant: str,
    pair_id: str,
) -> dict[str, Any]:
    if variant not in EVALUATION_VARIANTS:
        raise ValueError(f"unsupported evaluation variant: {variant}")
    _prepare_workspace(scenario, workspace)
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        _register_evaluation_image(runtime, variant)
        pid = runtime.process.spawn(image=_IMAGE_IDS[variant], goal=scenario.goal)
        _grant_scenario_authority(runtime, pid, scenario)
        initial_schemas = runtime.tools.openai_tool_schemas(pid)
        initial_schema_bytes = _json_bytes(initial_schemas)
        authorized_schema_bytes = _authorized_schema_bytes(runtime, pid)
        catalog_metadata_bytes = _json_bytes(
            runtime.skills.available_builtin_prompt_context(pid)
        )

        results = runtime.run_process_until_idle(pid, max_quanta=_MAX_QUANTA)
        process = runtime.process.get(pid)
        actions = _action_sequence(results)
        observations = _action_observations(results)
        activated_skills = [
            str(action.get("skill_id") or "")
            for action in actions
            if action.get("action") == "activate_skill"
        ]
        expected_activation_index = _first_action_index(
            actions,
            "activate_skill",
            skill_id=scenario.expected_skill_id,
        )
        expected_activation_succeeded = _first_successful_observation(
            observations,
            "activate_skill",
            skill_id=scenario.expected_skill_id,
        ) is not None
        probe_index = _first_action_index(actions, scenario.expected_probe_tool)
        probe_observation = _first_successful_observation(
            observations,
            scenario.expected_probe_tool,
        )
        first_activation = activated_skills[0] if activated_skills else None
        adjacent_activations = sorted(
            set(activated_skills).intersection(scenario.adjacent_skill_ids)
        )
        adjacent_tool_calls = sorted(
            {
                str(action.get("action") or "")
                for action in actions
            }.intersection(_adjacent_tool_names(scenario))
        )
        correct_activation: bool | None = None
        if variant == WITH_SKILLS:
            correct_activation = (
                first_activation == scenario.expected_skill_id
                and expected_activation_index is not None
                and expected_activation_succeeded
                and probe_index is not None
                and expected_activation_index < probe_index
                and not adjacent_activations
            )
        correct_route = _correct_route(
            variant=variant,
            correct_activation=correct_activation,
            actions=actions,
            scenario=scenario,
            adjacent_tool_calls=adjacent_tool_calls,
        )
        outcome = _evaluate_task_outcome(
            runtime,
            pid=pid,
            scenario=scenario,
            workspace=workspace,
            probe=probe_observation,
        )

        calls = sorted(
            runtime.store.list_llm_calls(
                pid=pid,
                limit=runtime.config.llm.call_record_hard_limit,
            ),
            key=lambda call: (call.created_at, call.call_id),
        )
        cumulative_schema_bytes = sum(_json_bytes(call.tools) for call in calls)
        cumulative_prompt_bytes = sum(_json_bytes(call.messages) for call in calls)
        prompt_tokens = sum(
            _plain_non_negative_int(call.usage.get("prompt_tokens")) for call in calls
        )
        cache_metrics = aggregate_cache_usage(calls)
        invalid_tool_calls = _invalid_tool_call_count(runtime, pid)
        status = process.status.value
        exited_via_tool = _first_successful_action_index(
            observations, "process_exit"
        ) is not None
        return {
            "scenario_id": scenario.scenario_id,
            "repetition": repetition,
            "pair_id": pair_id,
            "variant": variant,
            "image_id": _IMAGE_IDS[variant],
            "pid": pid,
            "status": status,
            "completed": (
                process.status == ProcessStatus.EXITED
                and exited_via_tool
                and outcome["passed"]
            ),
            "task_outcome_success": outcome["passed"],
            "task_outcome_oracle": outcome,
            "probe_tool_result_success": probe_observation is not None,
            "expected_skill_id": scenario.expected_skill_id,
            "expected_probe_tool": scenario.expected_probe_tool,
            "activated_skills": activated_skills,
            "adjacent_skill_activations": adjacent_activations,
            "adjacent_tool_calls": adjacent_tool_calls,
            "correct_skill_activation": correct_activation,
            "correct_route": correct_route,
            "actions": [str(action.get("action") or "") for action in actions],
            "invalid_tool_calls": invalid_tool_calls,
            "llm_calls": len(calls),
            "models": sorted(
                {str(call.model) for call in calls if call.model is not None}
            ),
            "catalog_metadata_bytes": catalog_metadata_bytes,
            "initial_schema_bytes": initial_schema_bytes,
            "authorized_schema_bytes": authorized_schema_bytes,
            "cumulative_schema_bytes": cumulative_schema_bytes,
            "initial_schema_token_estimate": math.ceil(initial_schema_bytes / 4),
            "authorized_schema_token_estimate": math.ceil(
                authorized_schema_bytes / 4
            ),
            "cumulative_schema_token_estimate": math.ceil(
                cumulative_schema_bytes / 4
            ),
            "cumulative_prompt_bytes": cumulative_prompt_bytes,
            "prompt_tokens": prompt_tokens,
            **cache_metrics,
            "initial_projection_reduction_rate": (
                1.0 - (initial_schema_bytes / authorized_schema_bytes)
                if authorized_schema_bytes
                else 0.0
            ),
            "status_message": process.status_message,
        }
    finally:
        runtime.close()


def _select_scenarios(
    scenario_ids: Iterable[str] | None,
) -> tuple[HeldOutScenario, ...]:
    if scenario_ids is None:
        return HELD_OUT_SCENARIOS
    requested = tuple(dict.fromkeys(str(value) for value in scenario_ids))
    known = {scenario.scenario_id: scenario for scenario in HELD_OUT_SCENARIOS}
    unknown = sorted(set(requested).difference(known))
    if unknown:
        raise ValueError(f"unknown built-in Tool Skill scenarios: {', '.join(unknown)}")
    return tuple(known[scenario_id] for scenario_id in requested)


def _prepare_workspace(scenario: HeldOutScenario, workspace: Path) -> None:
    if scenario.setup_kind not in {"git_read", "shell_git_read"}:
        return
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    workspace.joinpath("tracked-intent.txt").write_text(
        "held-out routing fixture\n",
        encoding="utf-8",
    )


def _register_evaluation_image(runtime: Runtime, variant: str) -> None:
    """Register the paired Skills treatment or no-Skills full projection."""

    catalog = get_builtin_skill_catalog()
    default_tools = [
        tool_name
        for skill in catalog.list()
        for tool_name in skill.allowed_tools
    ]
    template = runtime.get_image(_IMAGE_TEMPLATE_ID)
    metadata = {
        **template.metadata,
        "role": "builtin_tool_skill_evaluator",
        "evaluation_only": True,
        "evaluation_variant": variant,
    }
    default_skills = list(template.default_skills)
    if variant == WITH_SKILLS:
        metadata["tool_projection"] = "skills"
    elif variant == WITHOUT_SKILLS:
        metadata.pop("tool_projection", None)
        default_tools = [
            tool_name
            for tool_name in default_tools
            if tool_name not in _SKILL_LIFECYCLE_TOOLS
        ]
        default_skills = []
    else:
        raise ValueError(f"unsupported evaluation variant: {variant}")
    runtime.register_image(
        replace(
            template,
            image_id=_IMAGE_IDS[variant],
            name=f"builtin-tool-skill-evaluator-{variant}",
            default_tools=default_tools,
            default_skills=default_skills,
            metadata=metadata,
        ),
        actor="builtin-tool-skill-evaluation",
    )


def _grant_scenario_authority(
    runtime: Runtime,
    pid: str,
    scenario: HeldOutScenario,
) -> None:
    issued_by = "builtin-tool-skill-evaluation"
    if scenario.setup_kind == "workspace_write":
        runtime.filesystem.grant_path(
            pid,
            "skill-eval.txt",
            [CapabilityRight.WRITE],
            issued_by=issued_by,
        )
    elif scenario.setup_kind == "git_read":
        runtime.capability.issue_trusted(
            pid,
            runtime.config.git.repository_resource,
            [CapabilityRight.READ],
            issued_by=issued_by,
        )
    elif scenario.setup_kind == "shell_git_read":
        runtime.shell.grant_policy(
            pid,
            runtime.config.shell.allowlist_auto_else_ask_level,
            issued_by=issued_by,
        )
    elif scenario.setup_kind == "mcp_registry_read":
        runtime.capability.issue_trusted(
            pid,
            runtime.config.mcp.registry_resource,
            [CapabilityRight.READ],
            issued_by=issued_by,
        )
    elif scenario.setup_kind != "checkpoint":
        raise ValueError(f"unsupported scenario setup kind: {scenario.setup_kind}")


def _authorized_schema_bytes(runtime: Runtime, pid: str) -> int:
    process = runtime.process.get(pid)
    schemas: list[dict[str, Any]] = []
    for tool_id in sorted(set(process.tool_table.values())):
        implementation = runtime.tools.registry.implementation(tool_id)
        if implementation is not None:
            schemas.append(implementation.to_openai_chat_tool(config=runtime.config))
    return _json_bytes(schemas)


def _action_sequence(results: Iterable[Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        batch = result.get("actions")
        if isinstance(batch, list):
            actions.extend(dict(item) for item in batch if isinstance(item, dict))
            continue
        action = result.get("action")
        if isinstance(action, dict):
            actions.append(dict(action))
    return actions


def _action_observations(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Pair dispatched actions with their structured tool results."""

    observations: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        batch_actions = result.get("actions")
        batch_results = result.get("results")
        if isinstance(batch_actions, list) and isinstance(batch_results, list):
            for action, tool_result in zip(batch_actions, batch_results):
                if isinstance(action, dict) and isinstance(tool_result, dict):
                    observations.append(
                        {"action": dict(action), "result": dict(tool_result)}
                    )
            continue
        action = result.get("action")
        tool_result = result.get("result")
        if isinstance(action, dict) and isinstance(tool_result, dict):
            observations.append(
                {"action": dict(action), "result": dict(tool_result)}
            )
    return observations


def _first_successful_observation(
    observations: list[dict[str, Any]],
    action_name: str,
    **expected_fields: Any,
) -> dict[str, Any] | None:
    for observation in observations:
        action = observation["action"]
        result = observation["result"]
        if action.get("action") != action_name or result.get("ok") is not True:
            continue
        if all(action.get(key) == value for key, value in expected_fields.items()):
            return observation
    return None


def _first_successful_action_index(
    observations: list[dict[str, Any]],
    action_name: str,
    **expected_fields: Any,
) -> int | None:
    for index, observation in enumerate(observations):
        action = observation["action"]
        result = observation["result"]
        if action.get("action") != action_name or result.get("ok") is not True:
            continue
        if all(action.get(key) == value for key, value in expected_fields.items()):
            return index
    return None


def _adjacent_tool_names(scenario: HeldOutScenario) -> set[str]:
    catalog = get_builtin_skill_catalog()
    names: set[str] = set()
    for skill_id in scenario.adjacent_skill_ids:
        skill = catalog.get(skill_id)
        if skill is None:
            raise ValueError(f"unknown adjacent built-in Skill: {skill_id}")
        names.update(skill.allowed_tools)
    return names


def _correct_route(
    *,
    variant: str,
    correct_activation: bool | None,
    actions: list[dict[str, Any]],
    scenario: HeldOutScenario,
    adjacent_tool_calls: list[str],
) -> bool:
    if adjacent_tool_calls:
        return False
    if variant == WITH_SKILLS:
        return correct_activation is True
    expected_index = _first_action_index(actions, scenario.expected_probe_tool)
    return variant == WITHOUT_SKILLS and expected_index is not None


def _evaluate_task_outcome(
    runtime: Runtime,
    *,
    pid: str,
    scenario: HeldOutScenario,
    workspace: Path,
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate observable task state plus the successful structured result."""

    if probe is None:
        return {
            "passed": False,
            "checks": {"successful_probe_result": False},
            "failure": "expected probe tool did not return a successful result",
        }
    result = probe["result"]
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return {
            "passed": False,
            "checks": {"successful_probe_result": True, "structured_payload": False},
            "failure": "probe tool result payload was not an object",
        }
    if scenario.setup_kind == "workspace_write":
        return _workspace_write_outcome(workspace, payload)
    if scenario.setup_kind == "git_read":
        return _git_read_outcome(payload)
    if scenario.setup_kind == "shell_git_read":
        return _shell_git_read_outcome(payload)
    if scenario.setup_kind == "checkpoint":
        return _checkpoint_outcome(runtime, pid, payload)
    if scenario.setup_kind == "mcp_registry_read":
        return _mcp_registry_outcome(payload)
    return {
        "passed": False,
        "checks": {"supported_oracle": False},
        "failure": f"unsupported task oracle: {scenario.setup_kind}",
    }


def _workspace_write_outcome(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = workspace / "skill-eval.txt"
    exact_content = target.is_file() and target.read_bytes() == b"routed\n"
    payload_path = str(payload.get("path") or "")
    path_matches = payload_path in {"skill-eval.txt", "./skill-eval.txt"}
    checks = {
        "successful_probe_result": True,
        "exact_file_content": exact_content,
        "result_path_matches": path_matches,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _git_read_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    entries = payload.get("entries")
    state = payload.get("state")
    state_token = state.get("token") if isinstance(state, dict) else None
    checks = {
        "successful_probe_result": True,
        "reported_dirty_fixture": isinstance(entries, list) and bool(entries),
        "returned_state_token": isinstance(state_token, str) and bool(state_token),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_entry_count": len(entries) if isinstance(entries, list) else None,
    }


def _shell_git_read_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    stdout = payload.get("stdout")
    checks = {
        "successful_probe_result": True,
        "exact_argv": payload.get("argv") == ["git", "status", "--short"],
        "zero_returncode": payload.get("returncode") == 0,
        "reported_dirty_fixture": (
            isinstance(stdout, str) and "tracked-intent.txt" in stdout
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _checkpoint_outcome(
    runtime: Runtime,
    pid: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_id = payload.get("checkpoint_id")
    checkpoint_matches = False
    if isinstance(checkpoint_id, str) and checkpoint_id:
        try:
            inspected = runtime.checkpoint.inspect(
                checkpoint_id,
                actor=pid,
                require_capability=False,
            )
            checkpoint_matches = inspected["checkpoint"]["pid"] == pid
        except Exception:
            checkpoint_matches = False
    checks = {
        "successful_probe_result": True,
        "returned_checkpoint_id": isinstance(checkpoint_id, str) and bool(checkpoint_id),
        "durable_checkpoint_matches_process": checkpoint_matches,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _mcp_registry_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "successful_probe_result": True,
        "returned_server_metadata_list": isinstance(payload.get("servers"), list),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_server_count": (
            len(payload["servers"]) if isinstance(payload.get("servers"), list) else None
        ),
    }


def _first_action_index(
    actions: list[dict[str, Any]],
    action_name: str,
    **expected_fields: Any,
) -> int | None:
    for index, action in enumerate(actions):
        if action.get("action") != action_name:
            continue
        if all(action.get(key) == value for key, value in expected_fields.items()):
            return index
    return None


def _invalid_tool_call_count(runtime: Runtime, pid: str) -> int:
    count = 0
    for record in runtime.audit.trace():
        if record.actor != pid or record.action != "llm.action_repair_requested":
            continue
        decision = record.decision if isinstance(record.decision, dict) else {}
        count += _plain_non_negative_int(decision.get("tool_call_count"))
    return count


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _plain_non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _variant(run: dict[str, Any]) -> str:
    value = run.get("variant", WITH_SKILLS)
    return str(value)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(runs: list[dict[str, Any]], key: str) -> float:
    values = [
        float(run[key])
        for run in runs
        if isinstance(run.get(key), (int, float))
        and not isinstance(run.get(key), bool)
    ]
    return fmean(values) if values else 0.0


def _aggregate_run_cache_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    read_tokens = sum(
        _plain_non_negative_int(run.get("cache_read_tokens")) for run in runs
    )
    write_tokens = sum(
        _plain_non_negative_int(run.get("cache_write_tokens")) for run in runs
    )
    reported_calls = sum(
        _plain_non_negative_int(run.get("cache_reported_calls")) for run in runs
    )
    input_tokens = sum(
        _plain_non_negative_int(run.get("cache_metric_input_tokens"))
        for run in runs
    )
    uncached_tokens = sum(
        _plain_non_negative_int(run.get("uncached_input_tokens")) for run in runs
    )
    return {
        "cache_read_tokens": read_tokens,
        "cache_write_tokens": write_tokens,
        "cache_reported_calls": reported_calls,
        "cache_metric_input_tokens": input_tokens,
        "uncached_input_tokens": uncached_tokens,
        "cache_hit_rate": (
            (input_tokens - uncached_tokens) / input_tokens
            if reported_calls and input_tokens > 0
            else None
        ),
    }
