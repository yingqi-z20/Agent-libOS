from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import GitError
from benchmarks.runtime_safety.loader import load_task_file, load_tasks
from benchmarks.runtime_safety.runners import (
    AGENT_LIBOS_RUNNERS,
    RUNNER_INTERVENTIONS,
    RUNNER_NAMES,
    run_suite,
    write_run_outputs,
)
from benchmarks.runtime_safety.metrics import write_metrics
from agent_libos.utils.serde import to_jsonable
from agent_libos.substrate import LocalGitProvider

MAX_FAILURE_PREVIEW = 20
REPO_ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE_DISTRIBUTIONS = (
    "agent-libos",
    "openai",
    "psutil",
    "pydantic",
    "jsonschema",
    "PyYAML",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the runtime-safety benchmark suite.")
    parser.add_argument("--suite", default="benchmarks/runtime_safety", help="Benchmark suite root.")
    parser.add_argument("--runner", action="append", default=[], help="Runner name, repeated; use 'all' for every runner.")
    parser.add_argument("--task", action="append", default=[], help="Task id to include, repeated.")
    parser.add_argument("--attack-class", action="append", default=[], help="Attack class to include, repeated.")
    parser.add_argument("--limit", type=_positive_int, help="Maximum number of tasks after filtering.")
    parser.add_argument("--output", default=".benchmark_runs/m1", help="Output run directory.")
    parser.add_argument(
        "--llm",
        choices=["mock", "real"],
        default="mock",
        help="LLM mode; real mode requires exactly one task and only Agent libOS runners.",
    )
    parser.add_argument(
        "--max-quanta",
        type=_positive_int,
        help="Maximum scheduler quanta per Agent libOS task.",
    )
    parser.add_argument(
        "--require-all-passed",
        action="store_true",
        help=(
            "Return non-zero unless every selected task passes both success and safety oracles; "
            "by default oracle failures are reported without changing an otherwise valid run's exit status."
        ),
    )
    args = parser.parse_args(argv)

    suite = Path(args.suite)
    tasks = load_tasks(suite)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.id in wanted]
    if args.attack_class:
        wanted_classes = set(args.attack_class)
        tasks = [task for task in tasks if task.attack_class in wanted_classes]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no benchmark tasks selected")
    runners = _selected_runners(args.runner)
    if args.llm == "real":
        if len(tasks) != 1:
            raise SystemExit(
                "--llm real must select exactly one task after filtering to avoid accidental token spend"
            )
        unsupported_runners = [
            runner for runner in runners if runner not in AGENT_LIBOS_RUNNERS
        ]
        if unsupported_runners:
            raise SystemExit(
                "--llm real supports only Agent libOS runners; unsupported: "
                + ", ".join(unsupported_runners)
            )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "output_schema_version": 1,
        "suite": str(suite),
        "tasks": [task.id for task in tasks],
        "runners": runners,
        "llm_mode": args.llm,
        "pid": os.getpid(),
        "provenance": _build_provenance(
            suite,
            tasks,
            runners=runners,
            llm_mode=args.llm,
            max_quanta=args.max_quanta,
        ),
    }
    (output / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    runs = run_suite(tasks, suite, output, runners=runners, llm_mode=args.llm, max_quanta=args.max_quanta)
    write_run_outputs(runs, output)
    metrics = write_metrics(output)
    runner_failures = [
        {
            "task_id": run.result.task_id,
            "runner": run.result.runner,
            "failure_type": run.result.metadata.get("failure_type"),
        }
        for run in runs
        if run.result.metadata.get("runner_failed")
    ]
    invalid_runs = [
        {
            "task_id": run.result.task_id,
            "runner": run.result.runner,
            "invalid_reasons": list(run.result.invalid_reasons),
        }
        for run in runs
        if not run.result.valid
    ]
    oracle_failures = [
        {
            "task_id": run.result.task_id,
            "runner": run.result.runner,
            "task_success": run.result.task_success,
            "safety_passed": run.result.safety_passed,
        }
        for run in runs
        if not (run.result.task_success and run.result.safety_passed)
    ]
    print(
        json.dumps(
            to_jsonable(
                {
                    "output": str(output),
                    "results": len(runs),
                    "runner_failure_count": len(runner_failures),
                    "runner_failures": runner_failures[:MAX_FAILURE_PREVIEW],
                    "runner_failures_truncated": len(runner_failures) > MAX_FAILURE_PREVIEW,
                    "invalid_run_count": len(invalid_runs),
                    "invalid_runs": invalid_runs[:MAX_FAILURE_PREVIEW],
                    "invalid_runs_truncated": len(invalid_runs) > MAX_FAILURE_PREVIEW,
                    "oracle_failure_count": len(oracle_failures),
                    "oracle_failures": oracle_failures[:MAX_FAILURE_PREVIEW],
                    "oracle_failures_truncated": len(oracle_failures) > MAX_FAILURE_PREVIEW,
                    "metrics": metrics,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    if runner_failures:
        raise SystemExit(
            f"{len(runner_failures)} benchmark runner failure(s); outputs were written to {output}"
        )
    if not metrics.get("valid", False):
        raise SystemExit(
            f"benchmark outputs are invalid; inspect metrics invalid_reasons in {output}"
        )
    if args.require_all_passed and oracle_failures:
        raise SystemExit(
            f"{len(oracle_failures)} benchmark oracle failure(s); outputs were written to {output}"
        )


def _selected_runners(values: list[str]) -> list[str]:
    if not values:
        return ["agent_libos_full"]
    selected: list[str] = []
    for value in values:
        if value == "all":
            selected.extend(RUNNER_NAMES)
            continue
        if value not in RUNNER_NAMES:
            raise SystemExit(f"unknown runner {value!r}; choose one of {list(RUNNER_NAMES)} or 'all'")
        selected.append(value)
    return list(dict.fromkeys(selected))


def _positive_int(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if selected <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return selected


def _build_provenance(
    suite: Path,
    tasks: list[Any],
    *,
    runners: list[str],
    llm_mode: str,
    max_quanta: int | None,
) -> dict[str, Any]:
    task_entries, fixture_entries = _workload_provenance(suite, tasks)
    runner_sources = [
        REPO_ROOT / "benchmarks" / "runtime_safety" / name
        for name in (
            "ablations.py",
            "runners.py",
            "oracle.py",
            "metrics.py",
            "models.py",
            "loader.py",
            "fixtures.py",
        )
    ]
    return {
        "schema_version": 1,
        "git": _git_provenance(),
        "workload": {
            "tasks": task_entries,
            "fixtures": fixture_entries,
            "selected_workload_sha256": _sha256_json(
                {"tasks": task_entries, "fixtures": fixture_entries}
            ),
        },
        "config": {
            "default_config_sha256": _sha256_json(DEFAULT_CONFIG),
            "llm_mode": llm_mode,
            "max_quanta": max_quanta,
        },
        "runners": {
            "selected": runners,
            "interventions": {runner: RUNNER_INTERVENTIONS[runner] for runner in runners},
            "source_sha256": _hash_files(runner_sources, relative_to=REPO_ROOT),
        },
        "environment": _environment_provenance(llm_mode=llm_mode),
    }


def _workload_provenance(suite: Path, tasks: list[Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected_ids = {task.id for task in tasks}
    paths_by_id: dict[str, Path] = {}
    tasks_root = suite / "tasks"
    for path in sorted([*tasks_root.glob("*.yaml"), *tasks_root.glob("*.yml")]):
        loaded = load_task_file(path)
        if loaded.id in selected_ids:
            paths_by_id[loaded.id] = path
    missing = sorted(selected_ids - paths_by_id.keys())
    if missing:
        raise RuntimeError(f"cannot locate selected benchmark task files: {missing}")
    task_entries = [
        {
            "task_id": task.id,
            "path": paths_by_id[task.id].relative_to(suite).as_posix(),
            "sha256": _sha256_file(paths_by_id[task.id]),
        }
        for task in tasks
    ]
    workspaces = sorted({str(task.workspace) for task in tasks})
    fixture_entries = [
        {
            "path": workspace,
            "sha256": _hash_path(suite / workspace),
        }
        for workspace in workspaces
    ]
    return task_entries, fixture_entries


def _git_provenance() -> dict[str, Any]:
    guard = LocalGitProvider(REPO_ROOT, config=DEFAULT_CONFIG.git)
    try:
        for operation in ("repository_info", "list_refs", "status", "diff"):
            guard.validate_read_only_operation(operation)
    except GitError as exc:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "working_tree_sha256": None,
            "error_code": exc.code,
        }
    commit_result = _run_git(guard, "rev-parse", "HEAD")
    status_result = _run_git(
        guard,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    diff_result = _run_git(
        guard,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "HEAD",
        "--",
    )
    untracked_result = _run_git(
        guard,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if commit_result is None or status_result is None or diff_result is None or untracked_result is None:
        return {"available": False, "commit": None, "dirty": None, "working_tree_sha256": None}
    commit = commit_result.decode("utf-8", errors="replace").strip()
    digest = hashlib.sha256()
    digest.update(commit.encode("utf-8"))
    digest.update(b"\0status\0")
    digest.update(status_result)
    digest.update(b"\0diff\0")
    digest.update(diff_result)
    for raw_path in sorted(item for item in untracked_result.split(b"\0") if item):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        path = REPO_ROOT / relative
        if path.is_file() or path.is_symlink():
            digest.update(_path_bytes(path))
    return {
        "available": True,
        "commit": commit or None,
        "dirty": bool(status_result),
        "working_tree_sha256": digest.hexdigest(),
    }


def _run_git(provider: LocalGitProvider, *args: str) -> bytes | None:
    try:
        result = provider.run(
            args,
            read_only=True,
            max_output_bytes=DEFAULT_CONFIG.git.output_hard_limit_bytes,
        )
    except GitError:
        return None
    return result.stdout if result.returncode == 0 else None


def _environment_provenance(*, llm_mode: str) -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for distribution in _PROVENANCE_DISTRIBUTIONS:
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "benchmark_deno_backend": "deterministic-fake-deno" if llm_mode == "mock" else "runtime-selected",
        "real_llm_credentials_present": bool(
            os.getenv("OPENAI_API_KEY")
            and (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL"))
        ),
        "python_executable_kind": Path(sys.executable).name,
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_path(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        raise RuntimeError(f"benchmark provenance path does not exist: {path}")
    if path.is_file() or path.is_symlink():
        return hashlib.sha256(_path_bytes(path)).hexdigest()
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() or item.is_symlink()
    )
    return _hash_files(files, relative_to=path)


def _hash_files(paths: list[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(relative_to).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_path_bytes(path))
        digest.update(b"\0")
    return digest.hexdigest()


def _path_bytes(path: Path) -> bytes:
    if path.is_symlink():
        return b"symlink\0" + os.readlink(path).encode("utf-8", errors="surrogateescape")
    return b"file\0" + path.read_bytes()


if __name__ == "__main__":
    main()
