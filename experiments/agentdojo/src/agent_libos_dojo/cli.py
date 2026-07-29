from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo.attacks.attack_registry import ATTACKS
from agent_libos.models import PROMPT_MODE_IMAGE_ONLY, PROMPT_MODES
from agent_libos_dojo.pipeline import (
    EVALUATION_ENABLE_THINKING,
    EVALUATION_MAX_COMPLETION_TOKENS,
    EVALUATION_MAX_RETRIES,
    EVALUATION_TIMEOUT_S,
    PipelineRunError,
    normalize_model_override,
)

from agent_libos_dojo.runner import (
    ARM_ORDER_POLICY,
    ARMS,
    BENCHMARK_VERSION,
    CASE_MODES,
    LOGICAL_MODEL_INVOCATION_UNIT,
    MAX_QUERY_INVOCATIONS_PER_TRAJECTORY,
    RunOptions,
    SEMANTIC_SHARD_POLICY,
    _arm_position_counts,
    _catalog_expected_counts,
    _load_protocol_snapshot,
    _planning_provenance,
    _selected_model_override,
    catalog,
    plan_pilot,
    register_campaign,
    run,
    verify_run,
    verify_shard_coverage,
)


def _positive_int(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if selected <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return selected


def _nonnegative_int(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if selected < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return selected


def _model_override(value: str) -> str:
    try:
        selected = normalize_model_override(value)
    except PipelineRunError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert selected is not None
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated AgentDojo native-semantics evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_env_file = str(Path(__file__).resolve().parents[4] / ".env")

    catalog_parser = subparsers.add_parser("catalog", help="List benchmark inventory.")
    catalog_parser.add_argument("--benchmark-version", default=BENCHMARK_VERSION)

    register_parser = subparsers.add_parser(
        "register-campaign",
        help=(
            "Exclusively register one fresh generation-3 campaign before any "
            "provider call."
        ),
    )
    register_parser.add_argument(
        "--campaign-root",
        required=True,
        type=Path,
        help="Nonexistent external directory to create with mode 0700.",
    )
    register_parser.add_argument("--protocol", required=True, type=Path)
    register_parser.add_argument(
        "--source-manifest",
        required=True,
        type=Path,
        help=(
            "Canonical anonymous-stage byte manifest stored outside both the "
            "source stage and campaign root."
        ),
    )

    run_parser = subparsers.add_parser("run", help="Run a bounded real-LLM pilot.")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument(
        "--env-file",
        default=default_env_file,
        help="Explicit dotenv source; it is never copied to run artifacts.",
    )
    run_parser.add_argument("--benchmark-version", default=BENCHMARK_VERSION)
    run_parser.add_argument("--attack", choices=sorted(ATTACKS), default="injecagent")
    run_parser.add_argument("--suite", action="append", dest="suites")
    run_parser.add_argument("--arm", action="append", choices=ARMS, dest="arms")
    run_parser.add_argument("--mode", action="append", choices=CASE_MODES, dest="modes")
    run_parser.add_argument("--user-task", action="append", dest="user_tasks")
    run_parser.add_argument("--injection-task", action="append", dest="injection_tasks")
    run_parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Select every version-resolved user and injection task in each suite.",
    )
    run_parser.add_argument(
        "--shard-index",
        type=_nonnegative_int,
        default=0,
        help="Zero-based deterministic semantic shard index.",
    )
    run_parser.add_argument(
        "--shard-count",
        type=_positive_int,
        default=1,
        help="Number of semantic shards; every selected arm stays in one shard.",
    )
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        choices=(EVALUATION_MAX_COMPLETION_TOKENS,),
        default=EVALUATION_MAX_COMPLETION_TOKENS,
        help=(
            "Fixed per-invocation completion-token ceiling for this protocol "
            f"({EVALUATION_MAX_COMPLETION_TOKENS})."
        ),
    )
    run_parser.add_argument(
        "--model",
        dest="model_override",
        type=_model_override,
        help=(
            "Non-secret model-label override applied after the explicit dotenv "
            "snapshot; never read from ambient process state."
        ),
    )
    run_parser.add_argument(
        "--protocol",
        type=Path,
        help=(
            "Frozen JSON protocol inside the repository; its relative path and "
            "SHA-256 digest are bound into run metadata."
        ),
    )
    run_parser.add_argument(
        "--campaign-registration",
        type=Path,
        help=(
            "Fixed campaign_registration.json created by register-campaign; "
            "required for generation-3 formal execution."
        ),
    )
    run_parser.add_argument(
        "--max-quanta",
        type=int,
        default=16,
        help=(
            "Maximum harness-level logical model invocations per query "
            "(minimum 2); this excludes retries and API fallbacks inside one "
            "LLMClient call, and AgentDojo may issue up to three queries per "
            "trajectory."
        ),
    )
    run_parser.add_argument(
        "--libos-prompt-mode",
        choices=sorted(PROMPT_MODES),
        default=PROMPT_MODE_IMAGE_ONLY,
        help="Prompt envelope for the libos_ambient arm; recorded in metadata.",
    )
    run_parser.add_argument(
        "--observed-token-budget",
        type=int,
        default=250_000_000,
    )
    run_parser.add_argument("--case-limit", type=_positive_int)
    run_parser.add_argument("--confirm-real-llm", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--fail-on-invalid", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify hashes, traces, metrics, paired surfaces, and redaction."
    )
    verify_parser.add_argument("--output", required=True)
    verify_parser.add_argument("--env-file")
    verify_parser.add_argument("--require-complete", action="store_true")
    verify_parser.add_argument("--require-all-valid", action="store_true")

    shard_verify_parser = subparsers.add_parser(
        "verify-shards",
        help="Strictly verify a complete, disjoint all-catalog semantic shard set.",
    )
    shard_verify_parser.add_argument(
        "--output",
        action="append",
        required=True,
        dest="outputs",
        help="Shard output directory; repeat once per shard.",
    )
    shard_verify_parser.add_argument("--env-file")
    shard_verify_parser.add_argument("--require-all-valid", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(json.dumps(catalog(args.benchmark_version), ensure_ascii=False, indent=2))
        return
    if args.command == "register-campaign":
        result = register_campaign(
            args.campaign_root,
            args.protocol,
            args.source_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "verify":
        result = verify_run(
            args.output,
            env_file=args.env_file,
            require_complete=args.require_complete,
            require_all_valid=args.require_all_valid,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
        return
    if args.command == "verify-shards":
        result = verify_shard_coverage(
            args.outputs,
            env_file=args.env_file,
            require_all_valid=args.require_all_valid,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
        return

    if args.all_tasks and (args.user_tasks or args.injection_tasks):
        parser.error("--all-tasks cannot be combined with explicit task selectors")
    options = RunOptions(
        output_dir=Path(args.output),
        env_file=Path(args.env_file),
        benchmark_version=args.benchmark_version,
        attack=args.attack,
        suites=tuple(args.suites or ("workspace", "travel", "banking", "slack")),
        arms=tuple(args.arms or ARMS),
        modes=tuple(args.modes or CASE_MODES),
        user_tasks=tuple(args.user_tasks or ("user_task_0",)),
        injection_tasks=tuple(args.injection_tasks or ()),
        all_tasks=args.all_tasks,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        repetitions=args.repetitions,
        max_output_tokens=args.max_output_tokens,
        model_override=args.model_override,
        protocol_path=args.protocol,
        campaign_registration_path=args.campaign_registration,
        max_quanta=args.max_quanta,
        libos_prompt_mode=args.libos_prompt_mode,
        observed_token_budget=args.observed_token_budget,
        case_limit=args.case_limit,
        fail_on_invalid=args.fail_on_invalid,
    )
    cases = plan_pilot(options)
    if args.dry_run:
        protocol_snapshot = _load_protocol_snapshot(options.protocol_path)
        selected_model = _selected_model_override(options, protocol_snapshot)
        planning_provenance = _planning_provenance(options, cases)
        print(
            json.dumps(
                {
                    "real_llm_calls": False,
                    "planned_cases": len(cases),
                    "all_tasks": options.all_tasks,
                    "semantic_shard_policy": SEMANTIC_SHARD_POLICY,
                    "shard_index": options.shard_index,
                    "shard_count": options.shard_count,
                    "arm_order_policy": ARM_ORDER_POLICY,
                    "arm_ordinal_position_counts": _arm_position_counts(
                        cases, options.arms
                    ),
                    "catalog_expected_counts": _catalog_expected_counts(options),
                    **planning_provenance,
                    "max_query_invocations_per_trajectory": (
                        MAX_QUERY_INVOCATIONS_PER_TRAJECTORY
                    ),
                    "logical_model_invocation_unit": (
                        LOGICAL_MODEL_INVOCATION_UNIT
                    ),
                    "max_logical_model_invocations_per_query": (
                        options.max_quanta
                    ),
                    "max_logical_model_invocations_per_trajectory": (
                        options.max_quanta
                        * MAX_QUERY_INVOCATIONS_PER_TRAJECTORY
                    ),
                    "max_output_tokens_per_logical_model_invocation": (
                        options.max_output_tokens
                    ),
                    "max_completion_tokens_per_logical_invocation": (
                        options.max_output_tokens
                    ),
                    "model_override": selected_model,
                    "protocol_path": (
                        protocol_snapshot.relative_path
                        if protocol_snapshot is not None
                        else None
                    ),
                    "protocol_sha256": (
                        protocol_snapshot.sha256
                        if protocol_snapshot is not None
                        else None
                    ),
                    "api_mode": "chat",
                    "timeout_s": EVALUATION_TIMEOUT_S,
                    "enable_thinking": EVALUATION_ENABLE_THINKING,
                    "max_retries": EVALUATION_MAX_RETRIES,
                    "libos_prompt_mode": options.libos_prompt_mode,
                    "observed_token_budget": options.observed_token_budget,
                    "cases": [
                        {**case.__dict__, "case_id": case.case_id}
                        for case in cases
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.confirm_real_llm:
        parser.error("--confirm-real-llm is required to spend provider tokens")
    if not options.env_file.is_file():
        parser.error(f"dotenv file does not exist: {options.env_file}")
    result = run(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
