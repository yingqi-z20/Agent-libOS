from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.task_suite.load_suites import get_suite, get_suites

import agent_libos as agent_libos_package
from agent_libos.config import AgentLibOSConfig
from agent_libos.llm.client import read_dotenv
from agent_libos.models import PROMPT_MODE_MINIMAL_RUNTIME, PROMPT_MODES
from agent_libos.utils.openai_schema import normalize_openai_chat_tool_schema
from agent_libos.utils.serde import to_jsonable

from agent_libos_dojo.metrics import aggregate_results
from agent_libos_dojo.pipeline import (
    HIDDEN_TERMINAL_TOOL,
    AgentLibOSAmbientPipeline,
    ControlPipeline,
    ExplicitDotenvSnapshot,
    capture_explicit_dotenv_environment,
    evaluation_config,
    make_terminal_client_factory,
)


BENCHMARK_VERSION = "v1.2.2"
ARMS = ("upstream_control", "libos_ambient")
CASE_MODES = ("benign", "attacked", "injection_as_user")
PILOT_USER_TASK = "user_task_0"
PILOT_INJECTION_TASKS = {
    "workspace": "injection_task_0",
    "travel": "injection_task_0",
    "banking": "injection_task_0",
    # Slack removed injection_task_0 before benchmark v1.2.2.
    "slack": "injection_task_1",
}
_MAX_VERIFY_FILE_BYTES = 256 * 1024 * 1024
_MAX_VERIFY_TREE_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class PlannedCase:
    ordinal: int
    arm: str
    suite: str
    case_mode: str
    user_task_id: str | None
    injection_task_id: str | None
    attack: str | None
    repetition: int

    @property
    def case_id(self) -> str:
        user = self.user_task_id or "none"
        injection = self.injection_task_id or "none"
        attack = self.attack or "none"
        return (
            f"{self.ordinal:04d}-{self.suite}-{self.case_mode}-{user}-"
            f"{injection}-{attack}-r{self.repetition}-{self.arm}"
        )


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path
    env_file: Path
    benchmark_version: str = BENCHMARK_VERSION
    attack: str = "injecagent"
    suites: tuple[str, ...] = ("workspace", "travel", "banking", "slack")
    arms: tuple[str, ...] = ARMS
    modes: tuple[str, ...] = CASE_MODES
    user_tasks: tuple[str, ...] = (PILOT_USER_TASK,)
    # Empty selects the per-suite pilot task above. Explicit values are applied
    # to every selected suite and therefore must exist in each of them.
    injection_tasks: tuple[str, ...] = ()
    repetitions: int = 1
    max_output_tokens: int = 4096
    max_quanta: int = 16
    libos_prompt_mode: str = PROMPT_MODE_MINIMAL_RUNTIME
    observed_token_budget: int = 20_000_000
    case_limit: int | None = None
    fail_on_invalid: bool = False


def catalog(benchmark_version: str = BENCHMARK_VERSION) -> dict[str, Any]:
    suites = get_suites(benchmark_version)
    return {
        "agentdojo_package_version": importlib.metadata.version("agentdojo"),
        "benchmark_version": benchmark_version,
        "suites": {
            name: {
                "tools": len(suite.tools),
                "user_tasks": len(suite.user_tasks),
                "injection_tasks": len(suite.injection_tasks),
                "attacked_pairs": len(suite.user_tasks) * len(suite.injection_tasks),
                "user_task_ids": sorted(suite.user_tasks),
                "injection_task_ids": sorted(suite.injection_tasks),
            }
            for name, suite in suites.items()
        },
    }


def plan_pilot(options: RunOptions) -> list[PlannedCase]:
    _validate_options(options)
    cases: list[PlannedCase] = []
    ordinal = 0
    for suite_name in options.suites:
        suite = get_suite(options.benchmark_version, suite_name)
        injection_tasks = options.injection_tasks or (
            PILOT_INJECTION_TASKS[suite_name],
        )
        for repetition in range(1, options.repetitions + 1):
            for mode in options.modes:
                if mode == "injection_as_user":
                    for injection_task_id in injection_tasks:
                        suite.get_injection_task_by_id(injection_task_id)
                        for arm in options.arms:
                            ordinal += 1
                            cases.append(
                                PlannedCase(
                                    ordinal=ordinal,
                                    arm=arm,
                                    suite=suite_name,
                                    case_mode=mode,
                                    user_task_id=None,
                                    injection_task_id=injection_task_id,
                                    attack=None,
                                    repetition=repetition,
                                )
                            )
                    continue
                for user_task_id in options.user_tasks:
                    suite.get_user_task_by_id(user_task_id)
                    injection_ids: tuple[str | None, ...] = (
                        injection_tasks
                        if mode == "attacked"
                        else (None,)
                    )
                    for injection_task_id in injection_ids:
                        if injection_task_id is not None:
                            suite.get_injection_task_by_id(injection_task_id)
                        for arm in options.arms:
                            ordinal += 1
                            cases.append(
                                PlannedCase(
                                    ordinal=ordinal,
                                    arm=arm,
                                    suite=suite_name,
                                    case_mode=mode,
                                    user_task_id=user_task_id,
                                    injection_task_id=injection_task_id,
                                    attack=(options.attack if mode == "attacked" else None),
                                    repetition=repetition,
                                )
                            )
    semantic_keys = [_planned_case_semantic_key(case) for case in cases]
    if len(set(semantic_keys)) != len(semantic_keys):
        raise ValueError("planned cases contain duplicate semantic cases")
    if options.case_limit is not None:
        effective_limit = min(options.case_limit, len(cases))
        if options.arms and effective_limit % len(options.arms) != 0:
            raise ValueError(
                "case_limit must preserve complete selected-arm groups "
                f"(a multiple of {len(options.arms)})"
            )
        cases = cases[:effective_limit]
    return cases


def run(options: RunOptions) -> dict[str, Any]:
    config = evaluation_config(max_output_tokens=options.max_output_tokens)
    environment_snapshot = capture_explicit_dotenv_environment(
        options.env_file,
        config=config,
    )
    cases = plan_pilot(options)
    if not cases:
        raise ValueError("AgentDojo run requires at least one planned case")
    output = options.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    output.mkdir(parents=True)
    traces_dir = output / "traces"
    runtimes_dir = output / "runtimes"
    traces_dir.mkdir()
    runtimes_dir.mkdir()

    metadata = _metadata(
        options,
        cases,
        status="in_progress",
        environment_snapshot=environment_snapshot,
    )
    _atomic_json(output / "metadata.json", metadata)
    results_path = output / "results.jsonl"
    rows: list[dict[str, Any]] = []
    observed_tokens = 0
    stopped_for_budget = False
    with results_path.open("x", encoding="utf-8") as stream:
        for case in cases:
            if observed_tokens >= options.observed_token_budget:
                stopped_for_budget = True
                break
            environment_snapshot.assert_unchanged()
            row, trace = _run_case(
                options,
                case,
                runtime_dir=runtimes_dir / case.case_id,
                config=config,
                environment_snapshot=environment_snapshot,
            )
            environment_snapshot.assert_unchanged()
            trace_path = traces_dir / f"{case.case_id}.json"
            _atomic_json(trace_path, trace)
            row["trace_path"] = str(trace_path.relative_to(output))
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(row)
            observed_tokens += _observed_total_tokens(row)
            metrics = aggregate_results(rows)
            _atomic_json(output / "metrics.json", metrics)
            metadata.update(
                {
                    "completed_cases": len(rows),
                    "observed_total_tokens": observed_tokens,
                }
            )
            _atomic_json(output / "metadata.json", metadata)

    metrics = aggregate_results(rows)
    final_status = "partial_budget_exhausted" if stopped_for_budget else "complete"
    metadata.update(
        {
            "status": final_status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "completed_cases": len(rows),
            "observed_total_tokens": observed_tokens,
            "invalid_cases": metrics["invalid_rows"],
        }
    )
    _atomic_json(output / "metrics.json", metrics)
    _atomic_json(output / "metadata.json", metadata)
    manifest = _manifest(output, metadata, metrics, rows)
    _atomic_json(output / "manifest.json", manifest)

    if options.fail_on_invalid and metrics["invalid_rows"]:
        raise RuntimeError(
            f"AgentDojo run completed with {metrics['invalid_rows']} invalid trajectories"
        )
    return {
        "output_dir": str(output),
        "metadata": metadata,
        "metrics": metrics,
        "manifest": manifest,
    }


def verify_run(
    output_dir: str | Path,
    *,
    env_file: str | Path | None = None,
    require_complete: bool = False,
    require_all_valid: bool = False,
) -> dict[str, Any]:
    """Verify a run without trusting its manifest or favorable metrics."""

    output = Path(output_dir).resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}

    required = ("metadata.json", "metrics.json", "results.jsonl", "manifest.json")
    missing = [name for name in required if not (output / name).is_file()]
    checks["required_artifacts_present"] = not missing
    if missing:
        errors.append(f"missing required artifacts: {', '.join(missing)}")
        return _verification_result(output, checks, errors, observations={})

    artifact_tree = _artifact_tree_preflight(output)
    checks["artifact_tree"] = artifact_tree
    if not artifact_tree["valid"]:
        errors.extend(artifact_tree["errors"])
        return _verification_result(output, checks, errors, observations={})

    try:
        metadata = _read_json_object(output / "metadata.json")
        metrics = _read_json_object(output / "metrics.json")
        manifest = _read_json_object(output / "manifest.json")
        rows = _read_json_lines(output / "results.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks["primary_artifacts_parse"] = False
        errors.append(f"failed to parse primary artifacts: {type(exc).__name__}: {exc}")
        return _verification_result(output, checks, errors, observations={})
    checks["primary_artifacts_parse"] = True

    planned_count = metadata.get("planned_cases")
    planned_count_valid = (
        isinstance(planned_count, int)
        and not isinstance(planned_count, bool)
        and planned_count > 0
    )
    raw_planned_cases = metadata.get("cases")
    planned_case_maps = (
        raw_planned_cases
        if isinstance(raw_planned_cases, list)
        and all(isinstance(case, dict) for case in raw_planned_cases)
        else []
    )
    planned_case_ids = (
        [case.get("case_id") for case in planned_case_maps]
    )
    planned_semantic_keys = [
        _case_semantic_key(case) for case in planned_case_maps
    ]
    planned_semantics_valid = (
        all(key is not None for key in planned_semantic_keys)
        and len(set(planned_semantic_keys)) == len(planned_semantic_keys)
    )
    planned_cases_valid = (
        planned_count_valid
        and len(planned_case_maps) == planned_count
        and all(isinstance(case_id, str) and case_id for case_id in planned_case_ids)
        and len(set(planned_case_ids)) == len(planned_case_ids)
        and planned_semantics_valid
    )
    checks["positive_planned_case_count"] = planned_count_valid
    checks["planned_case_manifest"] = planned_cases_valid
    if not planned_count_valid:
        errors.append("metadata requires a positive planned_cases count")
    if not planned_cases_valid:
        errors.append("metadata cases do not define the unique planned case manifest")

    expected_artifacts = manifest.get("artifacts")
    artifact_matches: dict[str, bool] = {}
    if isinstance(expected_artifacts, dict):
        for name, expected in expected_artifacts.items():
            path = output / str(name)
            artifact_matches[str(name)] = (
                path.is_file()
                and isinstance(expected, str)
                and _sha256_file(path) == expected
            )
    checks["artifact_hashes"] = artifact_matches
    if set(artifact_matches) != {"metadata.json", "metrics.json", "results.jsonl"}:
        errors.append("manifest artifact set is incomplete or unexpected")
    if not artifact_matches or not all(artifact_matches.values()):
        errors.append("one or more primary artifact hashes do not match the manifest")

    trace_dir = output / "traces"
    trace_files = sorted(trace_dir.glob("*.json")) if trace_dir.is_dir() else []
    trace_entries = [
        {
            "path": str(path.relative_to(output)),
            "sha256": _sha256_file(path),
        }
        for path in trace_files
    ]
    trace_set_matches = (
        manifest.get("trace_set_sha256") == _sha256_json(trace_entries)
    )
    checks["trace_set_hash"] = trace_set_matches
    if not trace_set_matches:
        errors.append("trace-set hash does not match the manifest")

    row_count_matches = manifest.get("row_count") == len(rows)
    trace_count_matches = manifest.get("trace_count") == len(trace_files)
    manifest_status_matches = manifest.get("status") == metadata.get("status")
    checks["row_count"] = row_count_matches
    checks["trace_count"] = trace_count_matches
    checks["manifest_status"] = manifest_status_matches
    if not row_count_matches:
        errors.append("manifest row count does not match results.jsonl")
    if not trace_count_matches:
        errors.append("manifest trace count does not match traces directory")
    if not manifest_status_matches:
        errors.append("manifest status does not match metadata status")

    case_ids = [row.get("case_id") for row in rows]
    unique_case_ids = (
        all(isinstance(case_id, str) and case_id for case_id in case_ids)
        and len(set(case_ids)) == len(case_ids)
    )
    checks["unique_case_ids"] = unique_case_ids
    if not unique_case_ids:
        errors.append("results contain missing or duplicate case IDs")
    row_semantic_keys = [_case_semantic_key(row) for row in rows]
    row_semantics_unique = (
        all(key is not None for key in row_semantic_keys)
        and len(set(row_semantic_keys)) == len(row_semantic_keys)
    )
    checks["unique_case_semantics"] = row_semantics_unique
    if not row_semantics_unique:
        errors.append("results contain missing or duplicate semantic cases")
    plan_row_alignment = (
        planned_cases_valid
        and len(rows) <= len(planned_case_maps)
        and all(
            _case_manifest_projection(row)
            == _case_manifest_projection(planned_case_maps[index])
            for index, row in enumerate(rows)
        )
    )
    checks["row_plan_alignment"] = plan_row_alignment
    if not plan_row_alignment:
        errors.append("result rows do not match the recorded planned-case semantics")
    completed_plan_matches = (
        plan_row_alignment
        and len(rows) == len(planned_case_maps)
        and case_ids == planned_case_ids
    )
    checks["completed_plan_matches"] = completed_plan_matches
    if metadata.get("status") == "complete" and not completed_plan_matches:
        errors.append("complete run results do not match the planned case manifest")

    traces: dict[str, dict[str, Any]] = {}
    trace_parse_ok = True
    for path in trace_files:
        try:
            traces[path.stem] = _read_json_object(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            trace_parse_ok = False
    checks["trace_parse"] = trace_parse_ok
    if not trace_parse_ok:
        errors.append("one or more trace files are not valid JSON objects")

    row_trace_alignment = True
    hidden_terminal_absent = True
    provider_api_values: dict[str, set[str]] = defaultdict(set)
    provider_role_shapes: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            row_trace_alignment = False
            continue
        expected_trace_path = f"traces/{case_id}.json"
        trace = traces.get(case_id)
        if row.get("trace_path") != expected_trace_path or trace is None:
            row_trace_alignment = False
            continue
        expected_row = dict(row)
        expected_row.pop("trace_path", None)
        if trace.get("row_without_trace_path") != expected_row:
            row_trace_alignment = False
        case = trace.get("case")
        if not isinstance(case, dict) or any(
            case.get(field) != row.get(field)
            for field in (
                "ordinal",
                "arm",
                "suite",
                "case_mode",
                "user_task_id",
                "injection_task_id",
                "attack",
                "repetition",
            )
        ):
            row_trace_alignment = False
        evidence = trace.get("pipeline_evidence")
        if not isinstance(evidence, dict):
            row_trace_alignment = False
            continue
        arm = str(row.get("arm") or "unknown")
        provider_calls = evidence.get("provider_calls")
        if not isinstance(provider_calls, list):
            row_trace_alignment = False
            continue
        for provider_call in provider_calls:
            if not isinstance(provider_call, dict):
                row_trace_alignment = False
                continue
            provider_api_values[arm].add(str(provider_call.get("api") or ""))
            request = provider_call.get("request")
            if not isinstance(request, dict):
                row_trace_alignment = False
                continue
            roles = request.get("message_roles")
            if isinstance(roles, list):
                provider_role_shapes[arm].add(tuple(str(role) for role in roles))
            if HIDDEN_TERMINAL_TOOL in (request.get("tool_names") or []):
                hidden_terminal_absent = False
            tool_calls = provider_call.get("tool_calls")
            if isinstance(tool_calls, list) and any(
                isinstance(call, dict)
                and call.get("function") == HIDDEN_TERMINAL_TOOL
                for call in tool_calls
            ):
                hidden_terminal_absent = False
    checks["row_trace_alignment"] = row_trace_alignment
    checks["hidden_terminal_absent_from_provider_surface"] = hidden_terminal_absent
    if not row_trace_alignment:
        errors.append("one or more result rows do not align with their trace")
    if not hidden_terminal_absent:
        errors.append("runtime-only terminal tool leaked into provider evidence")

    recomputed_metrics = aggregate_results(rows)
    metrics_match = recomputed_metrics == metrics
    checks["metrics_recomputed"] = metrics_match
    if not metrics_match:
        errors.append("metrics.json does not equal a fresh aggregation of results")
    observed_tokens = recomputed_metrics["observed_total_tokens"]
    token_totals_match = (
        metadata.get("observed_total_tokens") == observed_tokens
        and manifest.get("observed_total_tokens") == observed_tokens
    )
    checks["token_totals"] = token_totals_match
    if not token_totals_match:
        errors.append("observed token totals disagree across artifacts")
    completed_count_matches = metadata.get("completed_cases") == len(rows)
    checks["metadata_completed_count"] = completed_count_matches
    if not completed_count_matches:
        errors.append("metadata completed-case count does not match results")

    paired = _verify_paired_surfaces(rows, traces)
    checks["paired_injection_hashes"] = paired["injection_hashes_equal"]
    checks["paired_tool_name_sets"] = paired["tool_name_sets_equal"]
    checks["paired_normalized_chat_tool_schemas"] = paired[
        "normalized_chat_tool_schemas_equal"
    ]
    checks["paired_provider_apis"] = paired["provider_apis_equal"]
    checks["paired_compatibility_fallbacks"] = paired[
        "compatibility_fallbacks_equal"
    ]
    complete_pair_present = paired["complete_pairs_compared"] > 0
    checks["complete_pair_present"] = complete_pair_present
    checks["all_semantic_cases_paired"] = paired["all_semantic_cases_paired"]
    if not paired["injection_hashes_equal"]:
        errors.append("paired arms did not receive identical attacked injections")
    if not paired["tool_name_sets_equal"]:
        errors.append("paired arms did not expose the same provider tool-name set")
    if not paired["normalized_chat_tool_schemas_equal"]:
        errors.append("paired arms differ after chat provider-schema normalization")
    if not paired["provider_apis_equal"]:
        errors.append("paired arms used different realized provider APIs")
    if not paired["compatibility_fallbacks_equal"]:
        errors.append("paired arms used different provider compatibility fallbacks")
    if (require_complete or require_all_valid) and not paired[
        "all_semantic_cases_paired"
    ]:
        errors.append(
            "strict verification requires every semantic case to contain both "
            "evaluation arms"
        )

    credential_scan = _scan_credentials(output, env_file)
    checks["credential_scan"] = credential_scan
    if credential_scan["requested"] and not credential_scan["env_file_present"]:
        errors.append("credential scan was requested but the dotenv file is missing")
    if credential_scan["raw_secret_hit_count"]:
        errors.append("raw API credential or endpoint appears in run artifacts")

    run_complete = metadata.get("status") == "complete"
    all_rows_valid = metrics.get("invalid_rows") == 0
    checks["run_complete"] = run_complete
    checks["all_rows_valid"] = all_rows_valid
    if require_complete and not run_complete:
        errors.append("run is not complete")
    if require_complete and not completed_plan_matches:
        errors.append("run did not complete its planned case manifest")
    if require_all_valid and not all_rows_valid:
        errors.append("run contains invalid trajectories")

    observations = {
        "rows": len(rows),
        "traces": len(trace_files),
        "observed_total_tokens": observed_tokens,
        "invalid_rows": metrics.get("invalid_rows"),
        "complete_pairs_compared": paired["complete_pairs_compared"],
        "incomplete_pair_count": paired["incomplete_pair_count"],
        "attacked_pairs_compared": paired["attacked_pairs_compared"],
        "pre_client_tool_order_equal_pairs": paired[
            "pre_client_tool_order_equal_pairs"
        ],
        "provider_api_values": {
            arm: sorted(values) for arm, values in sorted(provider_api_values.items())
        },
        "provider_message_role_shapes": {
            arm: [list(shape) for shape in sorted(shapes)]
            for arm, shapes in sorted(provider_role_shapes.items())
        },
    }
    return _verification_result(output, checks, errors, observations)


def _run_case(
    options: RunOptions,
    case: PlannedCase,
    *,
    runtime_dir: Path,
    config: AgentLibOSConfig,
    environment_snapshot: ExplicitDotenvSnapshot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = get_suite(options.benchmark_version, case.suite)
    system_message = load_system_message(None)
    pipeline: ControlPipeline | AgentLibOSAmbientPipeline
    if case.arm == "upstream_control":
        pipeline = ControlPipeline(
            client=environment_snapshot.new_client(),
            system_message=system_message,
            max_output_tokens=options.max_output_tokens,
            max_tool_iterations=max(1, options.max_quanta - 1),
        )
    elif case.arm == "libos_ambient":
        pipeline = AgentLibOSAmbientPipeline(
            client_factory=make_terminal_client_factory(environment_snapshot),
            system_message=system_message,
            runtime_dir=runtime_dir,
            config=config,
            max_quanta=options.max_quanta,
            prompt_mode=options.libos_prompt_mode,
        )
    else:
        raise ValueError(f"unknown evaluation arm: {case.arm}")

    started = time.perf_counter()
    injections: dict[str, str] = {}
    user_utility: bool | None = None
    attack_success: bool | None = None
    direct_success: bool | None = None
    error: BaseException | None = None
    injection_task = (
        suite.get_injection_task_by_id(case.injection_task_id)
        if case.injection_task_id is not None
        else None
    )
    target_tools = _target_tool_names(suite, injection_task)
    try:
        if case.case_mode == "injection_as_user":
            assert injection_task is not None
            direct_success, _ = suite.run_task_with_pipeline(
                pipeline,
                injection_task,
                injection_task=None,
                injections={},
            )
        else:
            assert case.user_task_id is not None
            user_task = suite.get_user_task_by_id(case.user_task_id)
            if case.case_mode == "attacked":
                assert injection_task is not None and case.attack is not None
                attack = load_attack(case.attack, suite, pipeline)
                injections = attack.attack(user_task, injection_task)
                user_utility, attack_success = suite.run_task_with_pipeline(
                    pipeline,
                    user_task,
                    injection_task=injection_task,
                    injections=injections,
                )
            else:
                user_utility, _ = suite.run_task_with_pipeline(
                    pipeline,
                    user_task,
                    injection_task=None,
                    injections={},
                )
    except Exception as exc:
        error = exc
    finally:
        if isinstance(pipeline, ControlPipeline):
            pipeline.close()

    duration = time.perf_counter() - started
    redactions = environment_snapshot.redactions()
    run_evidence = _redact_value(dict(pipeline.last_run), redactions)
    usage = run_evidence.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    status = "valid" if error is None else "infra_error"
    safe_and_useful = (
        bool(user_utility and not attack_success)
        if case.case_mode == "attacked"
        and isinstance(user_utility, bool)
        and isinstance(attack_success, bool)
        else None
    )
    injection_exposed = (
        _injection_exposed(run_evidence, injections)
        if case.case_mode == "attacked"
        else None
    )
    attempted_calls = _assistant_tool_calls(run_evidence)
    attempted_tools = [str(call["function"]) for call in attempted_calls]
    target_tool_set = set(target_tools)
    call_fingerprints = Counter(
        _sha256_json({"function": call["function"], "args": call.get("args", {})})
        for call in attempted_calls
    )
    provider_calls = run_evidence.get("provider_calls")
    row = {
        "schema_version": 1,
        "case_id": case.case_id,
        "ordinal": case.ordinal,
        "arm": case.arm,
        "suite": case.suite,
        "case_mode": case.case_mode,
        "user_task_id": case.user_task_id,
        "injection_task_id": case.injection_task_id,
        "attack": case.attack,
        "repetition": case.repetition,
        "status": status,
        "user_utility": user_utility,
        "attack_success": attack_success,
        "security_pass": (
            not attack_success if isinstance(attack_success, bool) else None
        ),
        "safe_and_useful": safe_and_useful,
        "injection_goal_success": direct_success,
        "injection_exposed": injection_exposed,
        "target_tool_names": target_tools,
        "target_tool_name_attempted": bool(
            target_tool_set.intersection(attempted_tools)
        ),
        "target_tool_name_call_count": sum(
            name in target_tool_set for name in attempted_tools
        ),
        "attempted_tool_names": attempted_tools,
        "provider_call_count": (
            len(provider_calls) if isinstance(provider_calls, list) else None
        ),
        "tool_call_count": len(attempted_calls),
        "repeated_identical_tool_call_count": sum(
            count - 1 for count in call_fingerprints.values()
        ),
        "max_identical_tool_call_multiplicity": max(
            call_fingerprints.values(), default=0
        ),
        "usage": usage,
        "duration_s": round(duration, 6),
        "injections_sha256": _sha256_json(injections) if injections else None,
        "error_type": type(error).__name__ if error is not None else None,
        "error": (
            _redact_text(str(error), redactions) if error is not None else None
        ),
    }
    trace = {
        "case": asdict(case),
        "metric_direction": {
            "attack_success_true_means": "targeted injection goal executed",
            "security_pass": "not attack_success",
        },
        "injections": injections,
        "row_without_trace_path": row,
        "pipeline_evidence": to_jsonable(run_evidence),
    }
    return row, trace


def _validate_options(options: RunOptions) -> None:
    if options.attack not in ATTACKS:
        raise ValueError(f"unknown AgentDojo attack: {options.attack}")
    known_suites = get_suites(options.benchmark_version)
    unknown_suites = sorted(set(options.suites).difference(known_suites))
    if unknown_suites:
        raise ValueError(f"unknown AgentDojo suites: {', '.join(unknown_suites)}")
    unknown_arms = sorted(set(options.arms).difference(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown arms: {', '.join(unknown_arms)}")
    unknown_modes = sorted(set(options.modes).difference(CASE_MODES))
    if unknown_modes:
        raise ValueError(f"unknown case modes: {', '.join(unknown_modes)}")
    for label, values in (
        ("suites", options.suites),
        ("arms", options.arms),
        ("modes", options.modes),
        ("user_tasks", options.user_tasks),
        ("injection_tasks", options.injection_tasks),
    ):
        duplicates = sorted(
            value for value, count in Counter(values).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"{label} contains duplicate selectors: {', '.join(duplicates)}"
            )
    if options.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if options.max_quanta < 2:
        raise ValueError("max_quanta must be at least 2")
    if options.max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if options.libos_prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"unknown Agent libOS prompt mode: {options.libos_prompt_mode}"
        )
    if options.observed_token_budget < 1:
        raise ValueError("observed_token_budget must be positive")
    if options.case_limit is not None and (
        isinstance(options.case_limit, bool) or options.case_limit < 1
    ):
        raise ValueError("case_limit must be positive")


def _planned_case_semantic_key(case: PlannedCase) -> tuple[Any, ...]:
    return (
        case.suite,
        case.case_mode,
        case.user_task_id,
        case.injection_task_id,
        case.attack,
        case.repetition,
        case.arm,
    )


def _metadata(
    options: RunOptions,
    cases: list[PlannedCase],
    *,
    status: str,
    environment_snapshot: ExplicitDotenvSnapshot | None = None,
) -> dict[str, Any]:
    snapshot = environment_snapshot or capture_explicit_dotenv_environment(
        options.env_file,
        config=evaluation_config(max_output_tokens=options.max_output_tokens),
    )
    resolved_client = snapshot.new_client()
    try:
        base_url = resolved_client.base_url or "https://api.openai.com/v1"
        effective_llm_config = {
            "model": resolved_client.model,
            "api_mode": resolved_client.api_mode,
            "endpoint_kind": _endpoint_kind(base_url),
            "endpoint_sha256": hashlib.sha256(
                base_url.encode("utf-8")
            ).hexdigest(),
            "credential_present": bool(resolved_client.api_key),
            "timeout_s": resolved_client.timeout,
            "max_retries": resolved_client.max_retries,
            "store": resolved_client.store,
            "reasoning_effort": resolved_client.reasoning_effort,
            "verbosity": resolved_client.verbosity,
            "safety_identifier_sha256": _optional_text_sha256(
                resolved_client.safety_identifier
            ),
            "prompt_cache_key_sha256": _optional_text_sha256(
                resolved_client.prompt_cache_key
            ),
            "prompt_cache_retention": resolved_client.prompt_cache_retention,
            "responses_previous_response_id": (
                resolved_client.responses_previous_response_id
            ),
            "fallback_json_actions": resolved_client.fallback_json_actions,
            "enable_thinking": resolved_client.enable_thinking,
            "organization_sha256": _optional_text_sha256(
                resolved_client.organization
            ),
            "project_sha256": _optional_text_sha256(resolved_client.project),
            "allow_custom_base_url": resolved_client.allow_custom_base_url,
            "inherit_ambient_openai_sdk_config": (
                resolved_client.inherit_ambient_openai_sdk_config
            ),
            "temperature": 0.0,
            "parallel_tool_calls": False,
            "max_output_tokens_per_call": options.max_output_tokens,
        }
    finally:
        resolved_client.close()
    root = Path(__file__).resolve().parents[4]
    lock = root / "experiments" / "agentdojo" / "uv.lock"
    source_entries = _harness_source_entries(root / "experiments" / "agentdojo")
    agent_libos_source_entries = _agent_libos_source_entries(root)
    harness_source_sha256 = _sha256_json(source_entries)
    agent_libos_source_sha256 = _sha256_json(agent_libos_source_entries)
    return {
        "schema_version": 1,
        "evaluation": "agentdojo_native_semantics_pilot",
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "agentdojo_package_version": importlib.metadata.version("agentdojo"),
        "agentdojo_benchmark_version": options.benchmark_version,
        "agent_libos_package_version": importlib.metadata.version("agent-libos"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "lock_sha256": _sha256_file(lock),
        "harness_source_sha256": harness_source_sha256,
        "harness_source_file_count": len(source_entries),
        "agent_libos_source_sha256": agent_libos_source_sha256,
        "agent_libos_source_file_count": len(agent_libos_source_entries),
        "agent_libos_source_scope": "pyproject.toml plus agent_libos/**/* excluding bytecode caches",
        "evaluation_source_sha256": _sha256_json(
            {
                "harness": harness_source_sha256,
                "editable_agent_libos": agent_libos_source_sha256,
            }
        ),
        "dependency_model": (
            "isolated AgentDojo subproject with Agent-libOS editable source; "
            "not the upstream AgentDojo reference lock"
        ),
        "model": effective_llm_config["model"],
        "api_mode": effective_llm_config["api_mode"],
        "endpoint_kind": effective_llm_config["endpoint_kind"],
        "endpoint_sha256": effective_llm_config["endpoint_sha256"],
        "credential_source": "explicit dotenv whitelist",
        "credential_present": effective_llm_config["credential_present"],
        "effective_llm_config": effective_llm_config,
        "effective_llm_config_sha256": _sha256_json(effective_llm_config),
        "temperature": 0.0,
        "parallel_tool_calls": False,
        "max_output_tokens_per_call": options.max_output_tokens,
        "max_quanta": options.max_quanta,
        "libos_prompt_mode": options.libos_prompt_mode,
        "observed_token_budget": options.observed_token_budget,
        "planned_cases": len(cases),
        "completed_cases": 0,
        "arms": list(options.arms),
        "suites": list(options.suites),
        "case_modes": list(options.modes),
        "attack": options.attack,
        "repetitions": options.repetitions,
        "semantics": {
            "upstream_control": (
                "AgentDojo native FunctionsRuntime/tool loop using Agent-libOS LLMClient"
            ),
            "libos_ambient": (
                "AgentDojo function contracts through Agent-libOS scheduler and "
                "ToolBroker with ambient suite-wide authority; provider-normalized "
                "schema parity is verified separately"
            ),
            "not_claimed": (
                "capability, approval, IFC, or protected external-effect containment"
            ),
            "hidden_terminal_shim": (
                "runtime-only; removed before every provider request and excluded "
                "from tool-call metrics"
            ),
        },
        "cases": [asdict(case) | {"case_id": case.case_id} for case in cases],
    }


def _manifest(
    output: Path,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_files = sorted((output / "traces").glob("*.json"))
    return {
        "schema_version": 1,
        "status": metadata["status"],
        "row_count": len(rows),
        "trace_count": len(trace_files),
        "artifacts": {
            "metadata.json": _sha256_file(output / "metadata.json"),
            "metrics.json": _sha256_file(output / "metrics.json"),
            "results.jsonl": _sha256_file(output / "results.jsonl"),
        },
        "trace_set_sha256": _sha256_json(
            [
                {
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256_file(path),
                }
                for path in trace_files
            ]
        ),
        "observed_total_tokens": metrics["observed_total_tokens"],
    }


_CASE_MANIFEST_FIELDS = (
    "case_id",
    "ordinal",
    "arm",
    "suite",
    "case_mode",
    "user_task_id",
    "injection_task_id",
    "attack",
    "repetition",
)


def _case_manifest_projection(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(value.get(field) for field in _CASE_MANIFEST_FIELDS)


def _case_semantic_key(value: dict[str, Any]) -> tuple[Any, ...] | None:
    ordinal = value.get("ordinal")
    arm = value.get("arm")
    suite = value.get("suite")
    case_mode = value.get("case_mode")
    repetition = value.get("repetition")
    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or ordinal < 1
        or arm not in ARMS
        or not isinstance(suite, str)
        or not suite
        or case_mode not in CASE_MODES
        or not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or repetition < 1
    ):
        return None
    optional_fields = (
        value.get("user_task_id"),
        value.get("injection_task_id"),
        value.get("attack"),
    )
    if any(item is not None and not isinstance(item, str) for item in optional_fields):
        return None
    return (
        suite,
        case_mode,
        *optional_fields,
        repetition,
        arm,
    )


def _verify_paired_surfaces(
    rows: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs: dict[tuple[Any, ...], dict[str, tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for row in rows:
        case_id = row.get("case_id")
        arm = row.get("arm")
        if not isinstance(case_id, str) or not isinstance(arm, str):
            continue
        trace = traces.get(case_id)
        if trace is None:
            continue
        key = (
            row.get("suite"),
            row.get("case_mode"),
            row.get("user_task_id"),
            row.get("injection_task_id"),
            row.get("attack"),
            row.get("repetition"),
        )
        pairs[key][arm] = (row, trace)

    complete = [
        pair
        for pair in pairs.values()
        if set(pair) == set(ARMS)
    ]
    incomplete_pair_count = len(pairs) - len(complete)
    injection_hashes_equal = True
    tool_name_sets_equal = True
    normalized_schemas_equal = True
    provider_apis_equal = True
    compatibility_fallbacks_equal = True
    order_equal = 0
    attacked_pairs = 0
    for pair in complete:
        control_row, control_trace = pair["upstream_control"]
        ambient_row, ambient_trace = pair["libos_ambient"]
        control_provider = _provider_execution_observation(control_trace)
        ambient_provider = _provider_execution_observation(ambient_trace)
        provider_apis_equal = provider_apis_equal and (
            control_provider is not None
            and ambient_provider is not None
            and len(control_provider[0]) == 1
            and control_provider[0] == ambient_provider[0]
        )
        compatibility_fallbacks_equal = compatibility_fallbacks_equal and (
            control_provider is not None
            and ambient_provider is not None
            and control_provider[1:] == ambient_provider[1:]
        )
        if control_row.get("case_mode") == "attacked":
            attacked_pairs += 1
            injection_hashes_equal = injection_hashes_equal and (
                control_row.get("injections_sha256")
                == ambient_row.get("injections_sha256")
                and control_row.get("injections_sha256") is not None
            )
        control_tools = _first_provider_tools(control_trace)
        ambient_tools = _first_provider_tools(ambient_trace)
        if control_tools is None or ambient_tools is None:
            tool_name_sets_equal = False
            normalized_schemas_equal = False
            continue
        control_names = [_tool_name(tool) for tool in control_tools]
        ambient_names = [_tool_name(tool) for tool in ambient_tools]
        tool_name_sets_equal = tool_name_sets_equal and (
            set(control_names) == set(ambient_names)
            and "" not in control_names
            and "" not in ambient_names
        )
        if control_names == ambient_names:
            order_equal += 1
        control_map = _normalized_chat_tool_map(control_tools)
        ambient_map = _normalized_chat_tool_map(ambient_tools)
        normalized_schemas_equal = normalized_schemas_equal and (
            control_map == ambient_map
        )
    return {
        "complete_pairs_compared": len(complete),
        "incomplete_pair_count": incomplete_pair_count,
        "all_semantic_cases_paired": bool(pairs) and incomplete_pair_count == 0,
        "attacked_pairs_compared": attacked_pairs,
        "injection_hashes_equal": injection_hashes_equal,
        "tool_name_sets_equal": tool_name_sets_equal,
        "normalized_chat_tool_schemas_equal": normalized_schemas_equal,
        "provider_apis_equal": provider_apis_equal,
        "compatibility_fallbacks_equal": compatibility_fallbacks_equal,
        "pre_client_tool_order_equal_pairs": order_equal,
    }


def _provider_execution_observation(
    trace: dict[str, Any],
) -> tuple[frozenset[str], frozenset[str], bool] | None:
    evidence = trace.get("pipeline_evidence")
    if not isinstance(evidence, dict):
        return None
    calls = evidence.get("provider_calls")
    if not isinstance(calls, list) or not calls:
        return None
    apis: set[str] = set()
    removed: set[str] = set()
    json_fallback_used = False
    for call in calls:
        if not isinstance(call, dict):
            return None
        api = call.get("api")
        if not isinstance(api, str) or not api:
            return None
        apis.add(api)
        raw_removed = call.get("compatibility_removed_options", [])
        if not isinstance(raw_removed, list) or not all(
            isinstance(item, str) for item in raw_removed
        ):
            return None
        removed.update(raw_removed)
        json_fallback_used = json_fallback_used or (
            call.get("fallback_json_action_used") is True
        )
    return frozenset(apis), frozenset(removed), json_fallback_used


def _first_provider_tools(trace: dict[str, Any]) -> list[dict[str, Any]] | None:
    evidence = trace.get("pipeline_evidence")
    if not isinstance(evidence, dict):
        return None
    calls = evidence.get("provider_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
        return None
    request = calls[0].get("request")
    if not isinstance(request, dict):
        return None
    tools = request.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        return None
    return tools


def _normalized_chat_tool_map(
    tools: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = _tool_name(tool)
        if not name or name in selected:
            return {}
        selected[name] = normalize_openai_chat_tool_schema(tool)
    return selected


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def _scan_credentials(
    output: Path,
    env_file: str | Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": env_file is not None,
        "env_file_present": False,
        "files_scanned": 0,
        "raw_secret_hit_count": 0,
        "hit_paths": {"api_key": [], "base_url": []},
    }
    if env_file is None:
        return result
    env_path = Path(env_file)
    if not env_path.is_file():
        return result
    result["env_file_present"] = True
    env = read_dotenv(env_path)
    needles = {
        "api_key": env.get("OPENAI_API_KEY"),
        "base_url": env.get("OPENAI_BASE_URL"),
    }
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    result["files_scanned"] = len(files)
    for label, value in needles.items():
        if not value:
            continue
        needle = value.encode("utf-8")
        for path in files:
            if _file_contains(path, needle):
                result["hit_paths"][label].append(str(path.relative_to(output)))
    result["raw_secret_hit_count"] = sum(
        len(paths) for paths in result["hit_paths"].values()
    )
    return result


def _file_contains(path: Path, needle: bytes) -> bool:
    overlap = max(0, len(needle) - 1)
    retained = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            selected = retained + chunk
            if needle in selected:
                return True
            retained = selected[-overlap:] if overlap else b""
    return False


def _verification_result(
    output: Path,
    checks: dict[str, Any],
    errors: list[str],
    observations: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "output_dir": str(output),
        "checks": checks,
        "observations": observations,
        "errors": errors,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    selected = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selected, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return selected


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"blank JSONL row at line {line_number}")
        selected = json.loads(line)
        if not isinstance(selected, dict):
            raise TypeError(f"expected JSON object at JSONL line {line_number}")
        rows.append(selected)
    return rows


def _artifact_tree_preflight(output: Path) -> dict[str, Any]:
    """Reject mutable links, special files, and unbounded verifier input."""

    errors: list[str] = []
    file_count = 0
    total_bytes = 0
    try:
        entries = sorted(output.rglob("*"), key=lambda path: str(path))
        for path in entries:
            relative = path.relative_to(output).as_posix()
            selected = path.lstat()
            if stat.S_ISLNK(selected.st_mode):
                errors.append(f"run artifact tree contains a symbolic link: {relative}")
                continue
            if stat.S_ISDIR(selected.st_mode):
                continue
            if not stat.S_ISREG(selected.st_mode):
                errors.append(f"run artifact tree contains a special file: {relative}")
                continue
            file_count += 1
            total_bytes += selected.st_size
            if selected.st_size > _MAX_VERIFY_FILE_BYTES:
                errors.append(
                    f"run artifact exceeds the per-file verification limit: {relative}"
                )
        if total_bytes > _MAX_VERIFY_TREE_BYTES:
            errors.append("run artifact tree exceeds the total verification limit")
    except OSError as exc:
        errors.append(
            f"failed to inspect run artifact tree: {type(exc).__name__}: {exc}"
        )
    return {
        "valid": not errors,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "max_file_bytes": _MAX_VERIFY_FILE_BYTES,
        "max_tree_bytes": _MAX_VERIFY_TREE_BYTES,
        "errors": errors,
    }


def _harness_source_entries(root: Path) -> list[dict[str, str]]:
    paths = sorted(
        [
            root / "pyproject.toml",
            root / "uv.lock",
            *(root / "src").rglob("*.py"),
            *(root / "tests").rglob("*.py"),
        ],
        key=lambda path: str(path.relative_to(root)),
    )
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_source_path(path),
        }
        for path in paths
    ]


def _agent_libos_source_entries(root: Path) -> list[dict[str, str]]:
    package_root = root / "agent_libos"
    imported_package_root = Path(agent_libos_package.__file__).resolve().parent
    if imported_package_root != package_root.resolve():
        raise RuntimeError(
            "AgentDojo evaluation requires the configured editable Agent-libOS "
            f"source at {package_root}, imported {imported_package_root}"
        )
    paths = [
        root / "pyproject.toml",
        *(
            path
            for path in package_root.rglob("*")
            if (path.is_file() or path.is_symlink())
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ),
    ]
    selected = sorted(paths, key=lambda path: str(path.relative_to(root)))
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_source_path(path),
        }
        for path in selected
    ]


def _target_tool_names(suite: Any, injection_task: Any | None) -> list[str]:
    if injection_task is None:
        return []
    env = suite.load_and_inject_default_environment({})
    return [call.function for call in injection_task.ground_truth(env)]


def _assistant_tool_calls(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    provider_calls = evidence.get("provider_calls")
    if isinstance(provider_calls, list):
        for provider_call in provider_calls:
            if not isinstance(provider_call, dict):
                continue
            for call in provider_call.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("function"):
                    selected.append(
                        {
                            "function": str(call["function"]),
                            "args": to_jsonable(call.get("args") or {}),
                        }
                    )
    if selected:
        return selected
    messages = evidence.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = call.get("function") or call.get("name")
                    if name:
                        selected.append(
                            {
                                "function": str(name),
                                "args": to_jsonable(call.get("args") or {}),
                            }
                        )
    return selected


def _injection_exposed(evidence: dict[str, Any], injections: dict[str, str]) -> bool:
    if not injections:
        return False
    values = [_normalize_exposure_text(value) for value in injections.values() if value]
    messages = evidence.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            if _contains_normalized_text(message.get("content"), values):
                return True
    tool_executions = evidence.get("tool_executions")
    return isinstance(tool_executions, list) and _contains_normalized_text(
        tool_executions, values
    )


def _contains_normalized_text(value: Any, needles: list[str]) -> bool:
    if isinstance(value, str):
        rendered = _normalize_exposure_text(value)
        return any(needle in rendered for needle in needles)
    if isinstance(value, dict):
        return any(_contains_normalized_text(item, needles) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_normalized_text(item, needles) for item in value)
    return False


def _normalize_exposure_text(value: str) -> str:
    # YAML folds long lines and doubles quotes inside quoted scalars. Exposure
    # is about whether the payload reached the model, not byte-for-byte output
    # formatting, so normalize those reversible presentation differences.
    return re.sub(r"\s+", " ", value.replace("''", "'")).strip().casefold()


def _observed_total_tokens(row: dict[str, Any]) -> int:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source_path(path: Path) -> str:
    if path.is_symlink():
        raise RuntimeError(f"evaluation source scope contains a symbolic link: {path}")
    return _sha256_file(path)


def _optional_text_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _endpoint_kind(base_url: str) -> str:
    parsed = urlparse(
        base_url if "://" in base_url else f"https://{base_url}"
    )
    if parsed.scheme.lower() == "https" and parsed.hostname == "api.openai.com":
        return "openai"
    return "custom_openai_compatible"


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _redact_value(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, replacements)
    if isinstance(value, list):
        return [_redact_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_value(item, replacements)
            for key, item in value.items()
        }
    return value


def _redact_text(value: str, replacements: dict[str, str]) -> str:
    selected = value
    for secret, replacement in replacements.items():
        selected = selected.replace(secret, replacement)
    return selected
