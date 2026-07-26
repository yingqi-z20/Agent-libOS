from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdojo.attacks.attack_registry import ATTACKS
from agent_libos.models import PROMPT_MODE_MINIMAL_RUNTIME, PROMPT_MODES

from agent_libos_dojo.runner import (
    ARMS,
    BENCHMARK_VERSION,
    CASE_MODES,
    LOGICAL_MODEL_INVOCATION_UNIT,
    MAX_QUERY_INVOCATIONS_PER_TRAJECTORY,
    RunOptions,
    catalog,
    plan_pilot,
    run,
    verify_run,
)


def _positive_int(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if selected <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated AgentDojo native-semantics evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_env_file = str(Path(__file__).resolve().parents[4] / ".env")

    catalog_parser = subparsers.add_parser("catalog", help="List benchmark inventory.")
    catalog_parser.add_argument("--benchmark-version", default=BENCHMARK_VERSION)

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
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument("--max-output-tokens", type=int, default=4096)
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
        default=PROMPT_MODE_MINIMAL_RUNTIME,
        help="Prompt envelope for the libos_ambient arm; recorded in metadata.",
    )
    run_parser.add_argument("--observed-token-budget", type=int, default=20_000_000)
    run_parser.add_argument("--case-limit", type=_positive_int)
    run_parser.add_argument("--confirm-real-llm", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--fail-on-invalid", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify hashes, traces, metrics, paired surfaces, and redaction."
    )
    verify_parser.add_argument("--output", required=True)
    verify_parser.add_argument("--env-file", default=default_env_file)
    verify_parser.add_argument("--require-complete", action="store_true")
    verify_parser.add_argument("--require-all-valid", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(json.dumps(catalog(args.benchmark_version), ensure_ascii=False, indent=2))
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
        repetitions=args.repetitions,
        max_output_tokens=args.max_output_tokens,
        max_quanta=args.max_quanta,
        libos_prompt_mode=args.libos_prompt_mode,
        observed_token_budget=args.observed_token_budget,
        case_limit=args.case_limit,
        fail_on_invalid=args.fail_on_invalid,
    )
    cases = plan_pilot(options)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "real_llm_calls": False,
                    "planned_cases": len(cases),
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
