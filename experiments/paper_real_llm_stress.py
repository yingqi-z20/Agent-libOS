from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_libos.llm.client import read_dotenv
from agent_libos.utils.serde import to_jsonable
from benchmarks.runtime_safety.loader import load_tasks
from benchmarks.runtime_safety.metrics import write_metrics
from benchmarks.runtime_safety.runners import run_task, write_run_outputs

DEFAULT_TASKS = (
    "shell_allowed_version_001",
    "fs_secret_read_001",
    "fs_write_forbidden_001",
    "shell_curl_001",
    "skill_jit_secret_read_001",
    "jsonrpc_visibility_no_method_authority_001",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small real-LLM runtime-safety stress set.")
    parser.add_argument("--suite", default="benchmarks/runtime_safety", help="Benchmark suite root.")
    parser.add_argument("--task", action="append", default=[], help="Task id to include, repeated.")
    parser.add_argument("--output", default=".benchmark_runs/real-llm-stress", help="Output run directory.")
    parser.add_argument("--max-quanta", type=int, default=3, help="Maximum scheduler quanta per task.")
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file loaded without printing values.")
    parser.add_argument(
        "--allow-token-spend",
        action="store_true",
        help="Required guardrail: acknowledge that this command may call a real LLM provider.",
    )
    args = parser.parse_args(argv)
    if not args.allow_token_spend:
        raise SystemExit("real LLM stress runs require --allow-token-spend")
    _load_env(args.env_file)
    _require_real_llm_env()
    suite = Path(args.suite)
    selected = tuple(args.task) if args.task else DEFAULT_TASKS
    tasks_by_id = {task.id: task for task in load_tasks(suite)}
    missing = [task_id for task_id in selected if task_id not in tasks_by_id]
    if missing:
        raise SystemExit(f"unknown benchmark task ids: {missing}")
    tasks = [tasks_by_id[task_id] for task_id in selected]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "suite": str(suite),
        "tasks": [task.id for task in tasks],
        "runners": ["agent_libos_full"],
        "llm_mode": "real",
        "max_quanta": args.max_quanta,
        "env": {
            "has_openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "has_openai_model": bool(os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL")),
            "has_custom_base_url": bool(os.getenv("OPENAI_BASE_URL")),
        },
        "pid": os.getpid(),
    }
    (output / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    runs = [
        run_task(task, suite, output, runner="agent_libos_full", llm_mode="real", max_quanta=args.max_quanta)
        for task in tasks
    ]
    write_run_outputs(runs, output)
    metrics = write_metrics(output)
    print(json.dumps(to_jsonable({"output": str(output), "results": len(runs), "metrics": metrics}), indent=2, ensure_ascii=False))


def _load_env(env_file: str) -> None:
    path = Path(env_file)
    if not path.exists():
        return
    for key, value in read_dotenv(path).items():
        os.environ.setdefault(key, value)


def _require_real_llm_env() -> None:
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL")):
        missing.append("OPENAI_LANGUAGE_MODEL or OPENAI_MODEL")
    if missing:
        raise SystemExit(f"real LLM environment is missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
