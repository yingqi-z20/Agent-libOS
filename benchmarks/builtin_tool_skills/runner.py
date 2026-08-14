from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.usage import aggregate_cache_usage
from agent_libos.models import CapabilityRight, ProcessStatus
from agent_libos.skills import get_builtin_skill_catalog
from agent_libos.substrate import LocalResourceProviderSubstrate
from benchmarks.live_evaluation_provenance import (
    build_evaluation_provenance,
    capture_evaluation_provenance,
    evaluation_provenance_identity,
    valid_evaluation_provenance,
)


REPORT_SCHEMA_VERSION = 3
EVALUATION_REPETITIONS = 3
WITH_SKILLS = "with_skills"
WITHOUT_SKILLS = "without_skills"
EVALUATION_VARIANTS = (WITH_SKILLS, WITHOUT_SKILLS)
PAIR_ORDER_METHOD = "deterministic_alternating_ab_ba_v1"
_IMAGE_IDS = {
    WITH_SKILLS: "builtin-tool-skill-evaluator:v0",
    WITHOUT_SKILLS: "builtin-tool-skill-full-projection-baseline:v0",
}
_IMAGE_TEMPLATE_ID = "coding-agent:v0"
_MAX_QUANTA = 8
_SKILL_LIFECYCLE_TOOLS = frozenset(
    {"discover_skills", "activate_skill", "read_skill_resource", "unload_skill"}
)
_PAIRED_BINARY_FIELDS = (
    "task_outcome_success",
    "completed",
    "correct_route",
    "probe_tool_result_success",
)
_PAIRED_NUMERIC_FIELDS = (
    "invalid_tool_calls",
    "llm_calls",
    "provider_attempts",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "initial_schema_bytes",
    "authorized_schema_bytes",
    "cumulative_schema_bytes",
    "initial_schema_token_estimate",
    "authorized_schema_token_estimate",
    "cumulative_schema_token_estimate",
    "cumulative_prompt_bytes",
)
_PUBLICATION_EVIDENCE_COUNT_FIELDS = (
    "provider_attempt_reported_calls",
    "prompt_token_reported_calls",
    "completion_token_reported_calls",
    "cache_total_calls",
    "cache_reported_calls",
    "cache_read_reported_calls",
    "cache_write_reported_calls",
    "cache_metric_reported_calls",
    "cache_metric_input_tokens",
    "uncached_input_tokens",
)
_PUBLICATION_PROCESS_STATUSES = frozenset(status.value for status in ProcessStatus)
_FULL_ORACLE_CHECK_FIELDS = {
    "workspace_write": frozenset(
        {"successful_probe_result", "exact_file_content", "result_path_matches"}
    ),
    "git_read": frozenset(
        {
            "successful_probe_result",
            "exact_fixture_entry",
            "complete_status_result",
            "valid_state_token",
        }
    ),
    "shell_git_read": frozenset(
        {
            "successful_probe_result",
            "exact_argv",
            "zero_returncode",
            "reported_dirty_fixture",
        }
    ),
    "checkpoint": frozenset(
        {
            "successful_probe_result",
            "checkpoint_creation_acknowledged",
            "host_resolved_checkpoint",
            "durable_checkpoint_matches_process",
        }
    ),
    "mcp_registry_read": frozenset(
        {"successful_probe_result", "exact_empty_registry"}
    ),
}


@dataclass(frozen=True)
class HeldOutScenario:
    scenario_id: str
    goal: str
    expected_skill_id: str
    expected_probe_tool: str
    adjacent_skill_ids: tuple[str, ...]
    setup_kind: str


@dataclass(frozen=True)
class SkillRoutingCase:
    """One source-neutral intent with an explicit adjacent routing boundary."""

    scenario_id: str
    intent: str
    expected_skill_id: str
    adjacent_skill_ids: tuple[str, ...]


# This catalog is deliberately broader than the five end-to-end probe
# scenarios below.  It gives every distributed built-in Skill one opt-in
# real-LLM activation case while keeping the expensive paired, effect-verified
# benchmark focused on its smaller set of representative workflows.
REAL_LLM_ROUTING_CATALOG: tuple[SkillRoutingCase, ...] = (
    SkillRoutingCase(
        "skill_navigation",
        "discover applicable guidance, activate the loaded Skill, read its resource, then unload it",
        "agent-libos-skill-navigation",
        ("agent-libos-authority-basics",),
    ),
    SkillRoutingCase(
        "authority_basics",
        "inspect missing Capability authority and request one exact permission decision",
        "agent-libos-authority-basics",
        ("agent-libos-capability-delegation",),
    ),
    SkillRoutingCase(
        "capability_delegation",
        "delegate one attenuated Capability to a direct child or revoke one exact Capability",
        "agent-libos-capability-delegation",
        ("agent-libos-child-processes",),
    ),
    SkillRoutingCase(
        "human_collaboration",
        "ask a Human one blocking intent question or send a one-way update",
        "agent-libos-human-collaboration",
        ("agent-libos-authority-basics",),
    ),
    SkillRoutingCase(
        "runtime_session",
        "read wall-clock time, compact context, finish the current process, or make one bounded delay",
        "agent-libos-runtime-session",
        ("agent-libos-child-processes",),
    ),
    SkillRoutingCase(
        "workspace_navigation",
        "inspect the working directory and read a bounded workspace text file",
        "agent-libos-workspace-navigation",
        ("agent-libos-workspace-editing",),
    ),
    SkillRoutingCase(
        "workspace_editing",
        "write, replace, or delete an ordinary workspace text file or directory",
        "agent-libos-workspace-editing",
        ("agent-libos-object-file-transfer",),
    ),
    SkillRoutingCase(
        "command_execution",
        "run an approved argv-only non-interactive shell command for a build, test, or utility",
        "agent-libos-command-execution",
        ("agent-libos-git-inspection",),
    ),
    SkillRoutingCase(
        "test_log_analysis",
        "triage an already captured pytest log for failed, assertion, and error lines",
        "agent-libos-test-log-analysis",
        ("agent-libos-command-execution",),
    ),
    SkillRoutingCase(
        "tool_protocol_diagnostics",
        "round-trip a small top-level JSON object through echo to isolate model/tool argument and result plumbing",
        "agent-libos-tool-protocol-diagnostics",
        ("agent-libos-jit-tool-authoring",),
    ),
    SkillRoutingCase(
        "object_memory",
        "create, discover, read, or append named JSON Object Memory in a namespace",
        "agent-libos-object-memory",
        ("agent-libos-object-file-transfer",),
    ),
    SkillRoutingCase(
        "object_file_transfer",
        "import workspace text into Object Memory or export Object text to a file",
        "agent-libos-object-file-transfer",
        ("agent-libos-workspace-editing",),
    ),
    SkillRoutingCase(
        "object_tasks",
        "run one tool asynchronously as a background task, then wait, watch, or cancel it by owner Object",
        "agent-libos-object-tasks",
        ("agent-libos-child-processes",),
    ),
    SkillRoutingCase(
        "child_processes",
        "spawn or fork a direct child process, coordinate messages, and merge selected results",
        "agent-libos-child-processes",
        ("agent-libos-object-tasks",),
    ),
    SkillRoutingCase(
        "checkpoints",
        "capture, inspect, compare, restore, or fork a durable process recovery checkpoint",
        "agent-libos-checkpoints",
        ("agent-libos-agent-images",),
    ),
    SkillRoutingCase(
        "agent_images",
        "register an AgentImage package, publish a checkpoint-derived image, or replace the current process image",
        "agent-libos-agent-images",
        ("agent-libos-checkpoints",),
    ),
    SkillRoutingCase(
        "jit_tool_authoring",
        "propose, validate, and register a process-local Deno TypeScript bounded tool",
        "agent-libos-jit-tool-authoring",
        ("agent-libos-tool-protocol-diagnostics",),
    ),
    SkillRoutingCase(
        "jsonrpc",
        "call a Host-registered plain JSON-RPC HTTP endpoint method integration",
        "agent-libos-jsonrpc",
        ("agent-libos-mcp",),
    ),
    SkillRoutingCase(
        "mcp",
        "discover cached MCP server metadata, refresh it, or call a registered MCP tool",
        "agent-libos-mcp",
        ("agent-libos-jsonrpc",),
    ),
    SkillRoutingCase(
        "git_inspection",
        "read-only inspect Git status, diffs, history, refs, remotes, or blame",
        "agent-libos-git-inspection",
        ("agent-libos-git-change-recording",),
    ),
    SkillRoutingCase(
        "git_change_recording",
        "stage, unstage, and commit reviewed changes from the entire intended Git index",
        "agent-libos-git-change-recording",
        ("agent-libos-git-integration-recovery",),
    ),
    SkillRoutingCase(
        "git_branches_worktrees",
        "manage local Git branches, tags, switch targets, and trusted managed worktrees",
        "agent-libos-git-branches-worktrees",
        ("agent-libos-git-integration-recovery",),
    ),
    SkillRoutingCase(
        "git_integration_recovery",
        "restore paths, integrate or abort history, stash, reset, or clean Git state",
        "agent-libos-git-integration-recovery",
        ("agent-libos-git-branches-worktrees",),
    ),
    SkillRoutingCase(
        "git_patch_objects",
        "create an immutable byte-safe Git patch Object and apply it to an authorized worktree",
        "agent-libos-git-patch-objects",
        ("agent-libos-git-change-recording",),
    ),
    SkillRoutingCase(
        "git_remotes",
        "fetch, pull, or push through an existing configured Git remote and exact destination ref",
        "agent-libos-git-remotes",
        ("agent-libos-git-inspection",),
    ),
    SkillRoutingCase(
        "git_pull_requests",
        "create, inspect, review, merge, or close a repository-local simulated pull request",
        "agent-libos-git-pull-requests",
        ("agent-libos-git-remotes",),
    ),
)


# The goals describe user intent and boundaries without exposing the expected
# Skill id or the exact probe tool name. This keeps routing behavior, rather
# than string copying, as the evaluated decision.
HELD_OUT_SCENARIOS: tuple[HeldOutScenario, ...] = (
    HeldOutScenario(
        scenario_id="ordinary_workspace_write",
        goal=(
            "Machine-only evaluation: create the ordinary workspace file routing-eval.txt "
            "with exactly `routed` followed by one newline. This is a normal workspace "
            "edit, not an Object Memory payload transfer. Perform the operation, verify "
            "completion, and exit without calling human_output."
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
            "shell command. Perform the read-only inspection, then exit without calling "
            "human_output."
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
            "execution rather than the typed Git inspection interface. Run the command "
            "and exit without calling human_output."
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
            "Image. Create the recovery point with a concise reason, and exit without "
            "calling human_output."
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
            "JSON-RPC endpoint. List the metadata even if it is empty, and exit without "
            "calling human_output."
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

    provenance_start = capture_evaluation_provenance(DEFAULT_CONFIG)
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scenarios = _select_scenarios(scenario_ids)
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for repetition in range(1, EVALUATION_REPETITIONS + 1):
            pair_id = f"{scenario.scenario_id}:{repetition}"
            pair_index = _pair_index(scenario.scenario_id, repetition)
            pair_order = _pair_order(pair_index)
            for pair_position, variant in enumerate(pair_order, start=1):
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
                        pair_index=pair_index,
                        pair_position=pair_position,
                        pair_order=pair_order,
                    )
                )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation": "builtin_tool_skill_routing",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_provenance": build_evaluation_provenance(
            provenance_start,
            capture_evaluation_provenance(DEFAULT_CONFIG),
        ),
        "image_ids": dict(_IMAGE_IDS),
        "variants": list(EVALUATION_VARIANTS),
        "order_design": _order_design(scenarios),
        "repetitions_per_scenario": EVALUATION_REPETITIONS,
        "pairs_per_scenario": EVALUATION_REPETITIONS,
        "scenario_count": len(scenarios),
        "scenario_catalog": [_scenario_record(scenario) for scenario in scenarios],
        "runs": runs,
        "metrics": aggregate_runs(runs),
    }


def _pair_index(scenario_id: str, repetition: int) -> int:
    scenario_ids = [scenario.scenario_id for scenario in HELD_OUT_SCENARIOS]
    try:
        scenario_offset = scenario_ids.index(scenario_id)
    except ValueError as error:
        raise ValueError(f"unknown scenario for pair order: {scenario_id}") from error
    if repetition < 1 or repetition > EVALUATION_REPETITIONS:
        raise ValueError(f"invalid evaluation repetition: {repetition}")
    return scenario_offset * EVALUATION_REPETITIONS + repetition


def _pair_order(pair_index: int) -> tuple[str, str]:
    if pair_index < 1 or pair_index > len(HELD_OUT_SCENARIOS) * EVALUATION_REPETITIONS:
        raise ValueError(f"invalid evaluation pair index: {pair_index}")
    if pair_index % 2 == 1:
        return EVALUATION_VARIANTS
    return (WITHOUT_SKILLS, WITH_SKILLS)


def _order_design(scenarios: Iterable[HeldOutScenario]) -> dict[str, Any]:
    orders = [
        _pair_order(_pair_index(scenario.scenario_id, repetition))
        for scenario in scenarios
        for repetition in range(1, EVALUATION_REPETITIONS + 1)
    ]
    return {
        "method": PAIR_ORDER_METHOD,
        "pair_count": len(orders),
        "treatment_first_pairs": sum(order[0] == WITH_SKILLS for order in orders),
        "baseline_first_pairs": sum(order[0] == WITHOUT_SKILLS for order in orders),
    }


def _scenario_record(scenario: HeldOutScenario) -> dict[str, Any]:
    record = asdict(scenario)
    record["adjacent_skill_ids"] = list(scenario.adjacent_skill_ids)
    return record


def evaluation_pair_plan(
    scenario_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the deterministic AB/BA order without running a provider."""

    plan: list[dict[str, Any]] = []
    for scenario in _select_scenarios(scenario_ids):
        for repetition in range(1, EVALUATION_REPETITIONS + 1):
            pair_index = _pair_index(scenario.scenario_id, repetition)
            plan.append(
                {
                    "pair_id": f"{scenario.scenario_id}:{repetition}",
                    "pair_index": pair_index,
                    "scenario_id": scenario.scenario_id,
                    "repetition": repetition,
                    "pair_order": list(_pair_order(pair_index)),
                }
            )
    return plan


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
        "mean_provider_attempts": _mean(selected, "provider_attempts"),
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
        "mean_completion_tokens": _mean(selected, "completion_tokens"),
        "mean_cumulative_prompt_bytes": _mean(
            selected, "cumulative_prompt_bytes"
        ),
        "mean_initial_projection_reduction_rate": _mean(
            selected, "initial_projection_reduction_rate"
        ),
        **cache_metrics,
        "by_variant": by_variant,
        "comparison": _comparison(by_variant),
        "paired": _paired_metrics(selected),
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


def report_publication_ready(report: dict[str, Any]) -> bool:
    """Validate the complete source/model-bound 15-pair publication artifact.

    Schema-v1/v2 reports remain ordinary readable JSON and continue to work with
    :func:`report_all_correct`, but only the fully counterbalanced schema-v3
    contract is eligible for publication.
    """

    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("evaluation") != "builtin_tool_skill_routing"
        or report.get("variants") != list(EVALUATION_VARIANTS)
        or report.get("repetitions_per_scenario") != EVALUATION_REPETITIONS
        or report.get("pairs_per_scenario") != EVALUATION_REPETITIONS
        or report.get("scenario_count") != len(HELD_OUT_SCENARIOS)
    ):
        return False
    provenance = report.get("evaluation_provenance")
    if not valid_evaluation_provenance(provenance):
        return False
    identity = evaluation_provenance_identity(provenance)
    if not _clean_evaluation_identity(identity):
        return False

    catalog = report.get("scenario_catalog")
    if not _valid_publication_catalog(catalog):
        return False
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != _publication_run_count():
        return False
    llm = identity.get("llm") if isinstance(identity, dict) else None
    provenance_model = llm.get("model") if isinstance(llm, dict) else None
    if not isinstance(provenance_model, str) or not _valid_publication_runs(
        runs,
        expected_model=provenance_model,
    ):
        return False
    expected_metrics = aggregate_runs(runs)
    if report.get("metrics") != expected_metrics:
        return False
    return report.get("order_design") == _order_design(HELD_OUT_SCENARIOS)


def _publication_run_count() -> int:
    return (
        len(HELD_OUT_SCENARIOS)
        * EVALUATION_REPETITIONS
        * len(EVALUATION_VARIANTS)
    )


def _clean_evaluation_identity(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return False
    source = value.get("source")
    llm = value.get("llm")
    return bool(
        isinstance(source, dict)
        and source.get("available") is True
        and source.get("dirty") is False
        and isinstance(source.get("commit"), str)
        and bool(source["commit"])
        and isinstance(source.get("working_tree_sha256"), str)
        and len(source["working_tree_sha256"]) == 64
        and isinstance(llm, dict)
        and llm.get("available") is True
        and llm.get("credential_present") is True
        and isinstance(llm.get("model"), str)
        and bool(llm["model"])
        and isinstance(llm.get("config_sha256"), str)
        and len(llm["config_sha256"]) == 64
    )


def _valid_publication_catalog(value: Any) -> bool:
    expected = [_scenario_record(scenario) for scenario in HELD_OUT_SCENARIOS]
    if value != expected:
        return False
    for scenario in HELD_OUT_SCENARIOS:
        goal = scenario.goal
        if (
            re.search(r"\bskills?\b", goal, flags=re.IGNORECASE) is not None
            or scenario.expected_skill_id in goal
            or scenario.expected_probe_tool in goal
        ):
            return False
    return True


def _valid_publication_runs(
    runs: list[Any],
    *,
    expected_model: str,
) -> bool:
    expected_sequence: list[
        tuple[HeldOutScenario, int, int, int, str, tuple[str, str]]
    ] = []
    for scenario in HELD_OUT_SCENARIOS:
        for repetition in range(1, EVALUATION_REPETITIONS + 1):
            pair_index = _pair_index(scenario.scenario_id, repetition)
            pair_order = _pair_order(pair_index)
            for pair_position, variant in enumerate(pair_order, start=1):
                expected_sequence.append(
                    (
                        scenario,
                        repetition,
                        pair_index,
                        pair_position,
                        variant,
                        pair_order,
                    )
                )
    observed_models: set[str] = set()
    for run, expected in zip(runs, expected_sequence, strict=True):
        if not isinstance(run, dict):
            return False
        scenario, repetition, pair_index, pair_position, variant, pair_order = expected
        if (
            run.get("scenario_id") != scenario.scenario_id
            or run.get("repetition") != repetition
            or run.get("pair_id") != f"{scenario.scenario_id}:{repetition}"
            or run.get("pair_index") != pair_index
            or run.get("pair_position") != pair_position
            or run.get("pair_order") != list(pair_order)
            or run.get("variant") != variant
            or run.get("goal_sha256")
            != hashlib.sha256(scenario.goal.encode("utf-8")).hexdigest()
        ):
            return False
        oracle = run.get("task_outcome_oracle")
        oracle_passed = oracle.get("passed") if isinstance(oracle, dict) else None
        oracle_checks = oracle.get("checks") if isinstance(oracle, dict) else None
        if not all(type(run.get(field)) is bool for field in _PAIRED_BINARY_FIELDS):
            return False
        status = run.get("status")
        completed = run["completed"]
        exited_via_tool = run.get("exited_via_tool")
        expected_completed = (
            status == ProcessStatus.EXITED.value
            and exited_via_tool is True
            and oracle_passed is True
        )
        if (
            not _valid_publication_oracle(oracle, scenario=scenario)
            or run.get("task_outcome_success") is not oracle_passed
            or run.get("probe_tool_result_success")
            is not oracle_checks.get("successful_probe_result")
            or run.get("run_evidence_complete") is not True
            or type(exited_via_tool) is not bool
            or status not in _PUBLICATION_PROCESS_STATUSES
            or (
                exited_via_tool is True
                and status != ProcessStatus.EXITED.value
            )
            or completed is not expected_completed
        ):
            return False
        correct_activation = run.get("correct_skill_activation")
        if (variant == WITH_SKILLS and type(correct_activation) is not bool) or (
            variant == WITHOUT_SKILLS and correct_activation is not None
        ):
            return False
        logical_calls = _strict_non_negative_int(run.get("llm_calls"))
        provider_attempts = _strict_non_negative_int(run.get("provider_attempts"))
        if (
            logical_calls is None
            or logical_calls < 1
            or provider_attempts is None
            or provider_attempts < logical_calls
        ):
            return False
        for field in _PAIRED_NUMERIC_FIELDS:
            if _strict_non_negative_int(run.get(field)) is None:
                return False
        for field in _PUBLICATION_EVIDENCE_COUNT_FIELDS:
            if _strict_non_negative_int(run.get(field)) is None:
                return False
        if any(
            run[field] != logical_calls
            for field in (
                "provider_attempt_reported_calls",
                "prompt_token_reported_calls",
                "completion_token_reported_calls",
                "cache_total_calls",
                "cache_reported_calls",
                "cache_read_reported_calls",
                "cache_metric_reported_calls",
            )
        ):
            return False
        cache_write_reported_calls = run["cache_write_reported_calls"]
        cache_write_tokens = run.get("cache_write_tokens")
        if (
            "cache_write_tokens" not in run
            or cache_write_reported_calls > logical_calls
            or (
                cache_write_reported_calls == logical_calls
                and _strict_non_negative_int(cache_write_tokens) is None
            )
            or (
                cache_write_reported_calls < logical_calls
                and cache_write_tokens is not None
            )
            or run["uncached_input_tokens"] > run["cache_metric_input_tokens"]
        ):
            return False
        if (
            run["initial_schema_token_estimate"]
            != math.ceil(run["initial_schema_bytes"] / 4)
            or run["authorized_schema_token_estimate"]
            != math.ceil(run["authorized_schema_bytes"] / 4)
            or run["cumulative_schema_token_estimate"]
            != math.ceil(run["cumulative_schema_bytes"] / 4)
        ):
            return False
        models = run.get("models")
        if (
            not isinstance(models, list)
            or len(models) != 1
            or not isinstance(models[0], str)
            or not models[0]
        ):
            return False
        observed_models.add(models[0])
    return observed_models == {expected_model}


def _valid_publication_oracle(
    value: Any,
    *,
    scenario: HeldOutScenario,
) -> bool:
    if not isinstance(value, dict):
        return False
    passed = value.get("passed")
    checks = value.get("checks")
    if (
        type(passed) is not bool
        or not isinstance(checks, dict)
        or not checks
        or not all(type(check) is bool for check in checks.values())
        or passed is not all(checks.values())
    ):
        return False
    check_fields = frozenset(checks)
    if check_fields == frozenset({"successful_probe_result"}):
        return (
            checks["successful_probe_result"] is False
            and isinstance(value.get("failure"), str)
            and bool(value["failure"].strip())
        )
    if check_fields == frozenset(
        {"successful_probe_result", "structured_payload"}
    ):
        return (
            checks["successful_probe_result"] is True
            and checks["structured_payload"] is False
            and isinstance(value.get("failure"), str)
            and bool(value["failure"].strip())
        )
    return (
        check_fields == _FULL_ORACLE_CHECK_FIELDS.get(scenario.setup_kind)
        and checks.get("successful_probe_result") is True
    )


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
        "mean_llm_calls": _mean(runs, "llm_calls"),
        "mean_provider_attempts": _mean(runs, "provider_attempts"),
        "mean_initial_schema_bytes": _mean(runs, "initial_schema_bytes"),
        "mean_cumulative_schema_bytes": _mean(runs, "cumulative_schema_bytes"),
        "mean_initial_schema_token_estimate": _mean(
            runs, "initial_schema_token_estimate"
        ),
        "mean_cumulative_schema_token_estimate": _mean(
            runs, "cumulative_schema_token_estimate"
        ),
        "mean_prompt_tokens": _mean(runs, "prompt_tokens"),
        "mean_completion_tokens": _mean(runs, "completion_tokens"),
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
        "mean_llm_calls",
        "mean_provider_attempts",
        "mean_initial_schema_bytes",
        "mean_cumulative_schema_bytes",
        "mean_initial_schema_token_estimate",
        "mean_cumulative_schema_token_estimate",
        "mean_prompt_tokens",
        "mean_completion_tokens",
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


def _paired_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        pair_id = run.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            grouped.setdefault(pair_id, []).append(run)
    samples: list[dict[str, Any]] = []
    for pair_id, pair_runs in grouped.items():
        variants = {_variant(run): run for run in pair_runs}
        treatment = variants.get(WITH_SKILLS)
        baseline = variants.get(WITHOUT_SKILLS)
        representative = min(
            pair_runs,
            key=lambda run: (
                _plain_positive_int(run.get("pair_index")) or (1 << 30),
                _plain_positive_int(run.get("pair_position")) or (1 << 30),
            ),
        )
        arms = {
            variant: _paired_arm_values(variants.get(variant))
            for variant in EVALUATION_VARIANTS
        }
        samples.append(
            {
                "pair_id": pair_id,
                "pair_index": _plain_positive_int(representative.get("pair_index")),
                "scenario_id": str(representative.get("scenario_id") or ""),
                "repetition": _plain_positive_int(representative.get("repetition")),
                "pair_order": _string_list(representative.get("pair_order")),
                "complete": (
                    len(pair_runs) == len(EVALUATION_VARIANTS)
                    and treatment is not None
                    and baseline is not None
                ),
                "arms": arms,
                "with_skills_minus_without_skills": _paired_deltas(
                    arms[WITH_SKILLS],
                    arms[WITHOUT_SKILLS],
                ),
            }
        )
    samples.sort(
        key=lambda sample: (
            sample["pair_index"] if sample["pair_index"] is not None else 1 << 30,
            sample["pair_id"],
        )
    )
    return {
        "binary_fields": list(_PAIRED_BINARY_FIELDS),
        "numeric_fields": list(_PAIRED_NUMERIC_FIELDS),
        "observed_pairs": len(samples),
        "complete_pairs": sum(sample["complete"] is True for sample in samples),
        "samples": samples,
    }


def _paired_arm_values(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    values: dict[str, Any] = {}
    for field in _PAIRED_BINARY_FIELDS:
        value = run.get(field)
        values[field] = value if isinstance(value, bool) else None
    for field in _PAIRED_NUMERIC_FIELDS:
        value = run.get(field)
        values[field] = (
            value
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
    return values


def _paired_deltas(
    treatment: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, int | float | None]:
    deltas: dict[str, int | float | None] = {}
    for field in (*_PAIRED_BINARY_FIELDS, *_PAIRED_NUMERIC_FIELDS):
        treatment_value = treatment.get(field) if treatment is not None else None
        baseline_value = baseline.get(field) if baseline is not None else None
        if isinstance(treatment_value, bool) and isinstance(baseline_value, bool):
            deltas[field] = int(treatment_value) - int(baseline_value)
        elif (
            isinstance(treatment_value, (int, float))
            and not isinstance(treatment_value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
        ):
            deltas[field] = treatment_value - baseline_value
        else:
            deltas[field] = None
    return deltas


def _run_once(
    scenario: HeldOutScenario,
    repetition: int,
    workspace: Path,
    *,
    variant: str,
    pair_id: str,
    pair_index: int,
    pair_position: int,
    pair_order: tuple[str, str],
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
        # Both arms begin without an eagerly injected Skill catalog. Treatment
        # discovers metadata through the same model tool used for every Skill.
        catalog_metadata_bytes = 0

        results = runtime.run_process_until_idle(pid, max_quanta=_MAX_QUANTA)
        process = runtime.process.get(pid)
        actions = _action_sequence(results)
        observations = _action_observations(results)
        discovery_trace = _discovery_trace(observations)
        exit_review_trace = _exit_review_trace(observations)
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
        prompt_token_counts = [
            _usage_counter_or_none(call.usage, "prompt_tokens", "input_tokens")
            for call in calls
        ]
        completion_token_counts = [
            _usage_counter_or_none(
                call.usage,
                "completion_tokens",
                "output_tokens",
            )
            for call in calls
        ]
        provider_attempt_counts = [_provider_attempt_count(call) for call in calls]
        prompt_tokens = _complete_counter_sum(prompt_token_counts)
        completion_tokens = _complete_counter_sum(completion_token_counts)
        provider_attempts = _complete_counter_sum(provider_attempt_counts)
        cache_metrics = aggregate_cache_usage(calls)
        invalid_tool_calls = _invalid_tool_call_count(runtime, pid)
        status = process.status.value
        exited_via_tool = _terminal_exit_receipt_observed(observations)
        return {
            "scenario_id": scenario.scenario_id,
            "repetition": repetition,
            "pair_id": pair_id,
            "pair_index": pair_index,
            "pair_position": pair_position,
            "pair_order": list(pair_order),
            "variant": variant,
            "goal_sha256": hashlib.sha256(scenario.goal.encode("utf-8")).hexdigest(),
            "image_id": _IMAGE_IDS[variant],
            "pid": pid,
            "status": status,
            "run_evidence_complete": True,
            "exited_via_tool": exited_via_tool,
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
            "discovery_trace": discovery_trace,
            "exit_review_trace": exit_review_trace,
            "invalid_tool_calls": invalid_tool_calls,
            "llm_calls": len(calls),
            "provider_attempts": provider_attempts,
            "provider_attempt_reported_calls": sum(
                count is not None for count in provider_attempt_counts
            ),
            "prompt_token_reported_calls": sum(
                count is not None for count in prompt_token_counts
            ),
            "completion_token_reported_calls": sum(
                count is not None for count in completion_token_counts
            ),
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
            "completion_tokens": completion_tokens,
            **cache_metrics,
            "initial_projection_reduction_rate": (
                1.0 - (initial_schema_bytes / authorized_schema_bytes)
                if authorized_schema_bytes
                else 0.0
            ),
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
            "routing-eval.txt",
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


def _discovery_trace(
    observations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain bounded routing evidence without storing arbitrary task arguments."""

    trace: list[dict[str, Any]] = []
    for observation in observations:
        action = observation.get("action")
        result = observation.get("result")
        if not isinstance(action, dict) or action.get("action") != "discover_skills":
            continue
        payload = result.get("payload") if isinstance(result, dict) else None
        skills = payload.get("skills") if isinstance(payload, dict) else None
        trace.append(
            {
                "text": action.get("text"),
                "limit": action.get("limit"),
                "ok": result.get("ok") if isinstance(result, dict) else None,
                "skill_ids": [
                    str(item.get("skill_id") or "")
                    for item in skills
                    if isinstance(item, dict)
                ]
                if isinstance(skills, list)
                else [],
                "has_more": payload.get("has_more")
                if isinstance(payload, dict)
                else None,
                "next_step": payload.get("next_step")
                if isinstance(payload, dict)
                else None,
            }
        )
    return trace


def _exit_review_trace(
    observations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain bounded completion diagnostics without prompts or result payloads."""

    max_trace_entries = 16
    max_list_entries = 16
    max_text_chars = 256

    def bounded_text(value: str) -> str:
        if len(value) <= max_text_chars:
            return value
        return value[: max_text_chars - 3] + "..."

    trace: list[dict[str, Any]] = []
    for observation in observations:
        if len(trace) >= max_trace_entries:
            break
        action = observation.get("action")
        result = observation.get("result")
        if not isinstance(action, dict) or action.get("action") != "process_exit":
            continue
        payload = result.get("payload") if isinstance(result, dict) else None
        review = (
            payload.get("completion_review")
            if isinstance(payload, dict)
            else None
        )
        trace.append(
            {
                "ok": result.get("ok") if isinstance(result, dict) else None,
                "result_status": (
                    payload.get("status") if isinstance(payload, dict) else None
                ),
                "has_review_token": bool(action.get("review_token")),
                "has_completion_evidence": action.get("completion_evidence")
                is not None,
                "validation_errors": [
                    bounded_text(value)
                    for value in review.get("validation_errors", [])[
                        :max_list_entries
                    ]
                    if isinstance(value, str)
                ]
                if isinstance(review, dict)
                else [],
                "explicit_unobserved_tools": [
                    bounded_text(item["tool"])
                    for item in review.get("explicit_unobserved_tool_hints", [])[
                        :max_list_entries
                    ]
                    if isinstance(item, dict) and isinstance(item.get("tool"), str)
                ]
                if isinstance(review, dict)
                else [],
            }
        )
    return trace


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


def _terminal_exit_receipt_observed(
    observations: Iterable[dict[str, Any]],
) -> bool:
    """Recognize only the current process-exit contract's committed receipt."""

    for observation in observations:
        action = observation.get("action")
        result = observation.get("result")
        payload = result.get("payload") if isinstance(result, dict) else None
        if (
            isinstance(action, dict)
            and action.get("action") == "process_exit"
            and isinstance(result, dict)
            and result.get("ok") is True
            and isinstance(payload, dict)
            and payload.get("status") == ProcessStatus.EXITED.value
            and payload.get("terminal_committed") is True
        ):
            return True
    return False


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
    target = workspace / "routing-eval.txt"
    exact_content = target.is_file() and target.read_bytes() == b"routed\n"
    payload_path = str(payload.get("path") or "")
    path_matches = payload_path in {"routing-eval.txt", "./routing-eval.txt"}
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
    exact_entry = False
    if isinstance(entries, list) and len(entries) == 1:
        entry = entries[0]
        path = entry.get("path") if isinstance(entry, dict) else None
        exact_entry = (
            isinstance(entry, dict)
            and isinstance(path, dict)
            and path.get("display") == "tracked-intent.txt"
            and path.get("path_b64") == "dHJhY2tlZC1pbnRlbnQudHh0"
            and path.get("lossy") is False
            and entry.get("kind") == "untracked"
            and entry.get("index_status") == "?"
            and entry.get("worktree_status") == "?"
        )
    checks = {
        "successful_probe_result": True,
        "exact_fixture_entry": exact_entry,
        "complete_status_result": payload.get("truncated") is False,
        "valid_state_token": (
            isinstance(state_token, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", state_token) is not None
        ),
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
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        try:
            candidates = runtime.checkpoint.list(
                pid,
                actor=pid,
                limit=2,
                require_capability=False,
            )
        except Exception:
            candidates = []
        if len(candidates) == 1:
            selected = candidates[0]
            if isinstance(selected, dict):
                checkpoint_id = selected.get("checkpoint_id")
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
        "checkpoint_creation_acknowledged": (
            payload.get("created") is True
            or (isinstance(payload.get("checkpoint_id"), str) and bool(payload["checkpoint_id"]))
        ),
        "host_resolved_checkpoint": isinstance(checkpoint_id, str) and bool(checkpoint_id),
        "durable_checkpoint_matches_process": checkpoint_matches,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _mcp_registry_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "successful_probe_result": True,
        "exact_empty_registry": (
            set(payload) == {"servers", "has_more"}
            and payload.get("servers") == []
            and payload.get("has_more") is False
        ),
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


def _strict_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _plain_positive_int(value: Any) -> int | None:
    selected = _strict_non_negative_int(value)
    return selected if selected is not None and selected > 0 else None


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _usage_counter_or_none(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = _strict_non_negative_int(usage.get(key))
        if value is not None:
            return value
    return None


def _provider_attempt_count(call: Any) -> int | None:
    request_options = getattr(call, "request_options", None)
    if not isinstance(request_options, dict):
        return None
    summary = request_options.get("provider_trace_summary")
    if not isinstance(summary, dict):
        return None
    return _strict_non_negative_int(summary.get("attempt_count"))


def _complete_counter_sum(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


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
    known_write_tokens = sum(
        _plain_non_negative_int(run.get("cache_write_tokens")) for run in runs
    )
    reported_calls = sum(
        _plain_non_negative_int(run.get("cache_reported_calls")) for run in runs
    )
    total_calls = sum(
        _plain_non_negative_int(run.get("cache_total_calls")) for run in runs
    )
    read_reported_calls = sum(
        _plain_non_negative_int(
            run.get(
                "cache_read_reported_calls",
                run.get("cache_reported_calls")
                if isinstance(run.get("cache_read_tokens"), int)
                else 0,
            )
        )
        for run in runs
    )
    write_reported_calls = sum(
        _plain_non_negative_int(
            run.get(
                "cache_write_reported_calls",
                run.get("cache_reported_calls")
                if isinstance(run.get("cache_write_tokens"), int)
                else 0,
            )
        )
        for run in runs
    )
    metric_reported_calls = sum(
        _plain_non_negative_int(
            run.get(
                "cache_metric_reported_calls",
                run.get(
                    "cache_read_reported_calls",
                    run.get("cache_reported_calls"),
                )
                if isinstance(run.get("cache_metric_input_tokens"), int)
                else 0,
            )
        )
        for run in runs
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
        "cache_write_tokens": (
            known_write_tokens
            if total_calls > 0 and write_reported_calls == total_calls
            else None
        ),
        "cache_total_calls": total_calls,
        "cache_reported_calls": reported_calls,
        "cache_read_reported_calls": read_reported_calls,
        "cache_write_reported_calls": write_reported_calls,
        "cache_metric_reported_calls": metric_reported_calls,
        "cache_metric_input_tokens": input_tokens,
        "uncached_input_tokens": uncached_tokens,
        "cache_hit_rate": (
            (input_tokens - uncached_tokens) / input_tokens
            if reported_calls and input_tokens > 0
            else None
        ),
    }
