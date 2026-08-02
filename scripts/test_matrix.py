from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import signal
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LANE_PATHS = {
    "unit": ("tests/unit",),
    "runtime": ("tests/runtime",),
    "security": ("tests/security",),
    "self-evolution": ("tests/self_evolution",),
    "providers": ("tests/providers",),
    "benchmark": ("tests/benchmarks",),
}
PYTHON_LANES = tuple(LANE_PATHS)
# Standard lanes target five minutes on the bounded-parallel development
# baseline. Keep a larger local default for serial diagnosis and host variance;
# CI supplies explicit 360-second deadlines for most lanes and 480 seconds for
# the larger runtime lane.
DEFAULT_MAX_LANE_SECONDS = 600.0
DEFAULT_WORKERS = "1"
DEFAULT_PARALLEL_WORKER_CAP = 4
PROCESS_TIMEOUT_EXIT_CODE = 124
PROCESS_TERMINATION_GRACE_SECONDS = 2.0
PARALLEL_BY_DEFAULT_LANES = {
    "runtime",
    "security",
    "self-evolution",
    "providers",
    "all",
}
DEFAULT_SERIAL_DIST = "loadfile"
DEFAULT_PARALLEL_DIST = "worksteal"
WORKERS_ENV = "AGENT_LIBOS_TEST_WORKERS"
DIST_ENV = "AGENT_LIBOS_TEST_DIST"
XDIST_DISTS = ("loadfile", "loadscope", "load", "worksteal")


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]
    env: dict[str, str] | None = None
    enforce_timeout: bool = True
    invariant_test_paths: tuple[str, ...] | None = None
    invariant_marker_expression: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent-libOS test lanes.")
    parser.add_argument(
        "--lane",
        choices=[*PYTHON_LANES, "gui", "all"],
        required=True,
        help="test lane to run",
    )
    parser.add_argument("--skip-real-deno", action="store_true", help="exclude tests marked real_deno")
    parser.add_argument("--run-real-llm", action="store_true", help="include tests marked real_llm")
    parser.add_argument("--run-mcp", action="store_true", help="include tests marked mcp")
    parser.add_argument(
        "--keep-agent-outputs",
        action="store_true",
        help="preserve files created under agent_outputs during pytest cleanup",
    )
    parser.add_argument(
        "--max-lane-seconds",
        type=_positive_seconds,
        default=DEFAULT_MAX_LANE_SECONDS,
        help="hard process-tree timeout applied independently to each selected command",
    )
    parser.add_argument(
        "--durations",
        type=_nonnegative_integer,
        default=None,
        metavar="N",
        help="report the N slowest pytest durations; use 0 to report all durations",
    )
    parser.add_argument(
        "--shard-count",
        type=_positive_integer,
        default=1,
        help="split one Python lane into this many deterministic file-weighted shards",
    )
    parser.add_argument(
        "--shard-index",
        type=_nonnegative_integer,
        default=0,
        help="zero-based shard to execute when --shard-count is greater than one",
    )
    parser.add_argument(
        "-n",
        "--workers",
        type=_worker_count,
        default=None,
        help=(
            "number of pytest-xdist workers for Python lanes; use 1 to run serially, or auto/logical "
            f"(default: bounded parallel for {', '.join(sorted(PARALLEL_BY_DEFAULT_LANES))})"
        ),
    )
    parser.add_argument(
        "--dist",
        choices=XDIST_DISTS,
        default=None,
        help="pytest-xdist scheduling strategy used when --workers is greater than 1",
    )
    args = parser.parse_args(argv)
    _resolve_defaults(parser, args)
    _validate_args(parser, args)

    commands = _commands_for(args)
    with tempfile.TemporaryDirectory(prefix="agent-libos-invariant-receipts-") as receipt_dir:
        for index, command in enumerate(commands):
            receipt_path: Path | None = None
            if _is_pytest_command(command):
                receipt_path = Path(receipt_dir) / f"command-{index}.json"
                command = _with_invariant_receipt(command, receipt_path)
            status = _run(command, max_seconds=args.max_lane_seconds)
            if status != 0:
                return status
            if receipt_path is not None:
                status = _validate_invariant_receipt(
                    receipt_path,
                    lane=None if args.lane == "all" else args.lane,
                    selected_test_paths=command.invariant_test_paths,
                    marker_expression=command.invariant_marker_expression,
                )
                if status != 0:
                    return status
    return 0


def _commands_for(args: argparse.Namespace) -> list[Command]:
    if args.lane == "gui":
        npm = _required_tool("npm")
        return [
            Command("gui unit tests", [npm, "--prefix", "gui", "run", "test"]),
            Command("gui typecheck", [npm, "--prefix", "gui", "run", "typecheck"]),
            Command("gui build", [npm, "--prefix", "gui", "run", "build"]),
        ]
    if args.lane == "all":
        return [
            Command(
                f"pytest all deterministic lanes{_worker_suffix(args)}",
                _pytest_args(("tests",), args),
                env=_pytest_env(args),
                invariant_marker_expression=_pytest_marker_expression(args),
            )
        ]
    selected_paths = _sharded_lane_paths(
        LANE_PATHS[args.lane],
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    shard_suffix = (
        ""
        if args.shard_count == 1
        else f" shard {args.shard_index + 1}/{args.shard_count}"
    )
    return [
        Command(
            f"pytest {args.lane}{shard_suffix}{_worker_suffix(args)}",
            _pytest_args(selected_paths, args),
            env=_pytest_env(args),
            invariant_test_paths=(
                selected_paths if args.shard_count > 1 else None
            ),
            invariant_marker_expression=_pytest_marker_expression(args),
        )
    ]


def _sharded_lane_paths(
    paths: tuple[str, ...],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[str, ...]:
    if shard_count == 1:
        return paths
    files = sorted(
        {
            candidate.relative_to(ROOT).as_posix()
            for raw_path in paths
            for candidate in (
                sorted((ROOT / raw_path).rglob("test_*.py"))
                if (ROOT / raw_path).is_dir()
                else (ROOT / raw_path,)
            )
            if candidate.is_file()
        }
    )
    if shard_count > len(files):
        raise ValueError(
            f"shard count {shard_count} exceeds the {len(files)} selected test files"
        )
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    weights = [0 for _ in range(shard_count)]
    weighted_files = sorted(
        ((ROOT / path).stat().st_size, path) for path in files
    )
    for size, path in reversed(weighted_files):
        selected = min(range(shard_count), key=lambda index: (weights[index], index))
        buckets[selected].append(path)
        weights[selected] += size
    return tuple(sorted(buckets[shard_index]))


def _pytest_args(paths: tuple[str, ...], args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "pytest", *paths]
    if _workers_enabled(args):
        command.extend(["-n", args.workers, "--dist", args.dist])
    if args.durations is not None:
        command.extend(["--durations", str(args.durations)])
    if args.skip_real_deno:
        command.append("--skip-real-deno")
    if args.run_real_llm:
        command.append("--run-real-llm")
    if getattr(args, "run_mcp", False):
        command.append("--run-mcp")
    command.extend(["-m", _pytest_marker_expression(args)])
    return command


def _pytest_marker_expression(args: argparse.Namespace) -> str:
    marker_filters: list[str] = ["not postgres"]
    if args.skip_real_deno:
        marker_filters.append("not real_deno")
    if not args.run_real_llm:
        marker_filters.append("not real_llm")
    if not getattr(args, "run_mcp", False):
        marker_filters.append("not mcp")
    return " and ".join(marker_filters)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.lane == "gui" and _workers_enabled(args):
        parser.error("--workers only applies to pytest lanes; run the gui lane separately")
    if _workers_enabled(args) and importlib.util.find_spec("xdist") is None:
        parser.error("pytest-xdist is required for --workers; run `uv sync --frozen` first")
    if args.lane == "gui" and args.keep_agent_outputs:
        parser.error("--keep-agent-outputs only applies to pytest lanes")
    if args.shard_index >= args.shard_count:
        parser.error("--shard-index must be less than --shard-count")
    if args.lane in {"gui", "all"} and (
        args.shard_count != 1 or args.shard_index != 0
    ):
        parser.error("test sharding applies only to an individual Python lane")
    if args.shard_count > 1:
        try:
            _sharded_lane_paths(
                LANE_PATHS[args.lane],
                shard_count=args.shard_count,
                shard_index=args.shard_index,
            )
        except ValueError as exc:
            parser.error(str(exc))


def _resolve_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.workers is None:
        try:
            args.workers = _default_workers_for_lane(args.lane)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"{WORKERS_ENV}: {exc}")
    if args.dist is None:
        args.dist = _default_dist(parser, args)


def _default_workers_for_lane(lane: str) -> str:
    env_workers = os.getenv(WORKERS_ENV)
    if env_workers:
        return _worker_count(env_workers)
    if lane in PARALLEL_BY_DEFAULT_LANES:
        cpu_count = os.cpu_count() or 1
        return str(max(1, min(DEFAULT_PARALLEL_WORKER_CAP, cpu_count)))
    return DEFAULT_WORKERS


def _default_dist(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    env_dist = os.getenv(DIST_ENV)
    if env_dist:
        if env_dist not in XDIST_DISTS:
            parser.error(f"{DIST_ENV} must be one of {', '.join(XDIST_DISTS)}")
        return env_dist
    if _workers_enabled(args):
        return DEFAULT_PARALLEL_DIST
    return DEFAULT_SERIAL_DIST


def _worker_count(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"auto", "logical"}:
        return text
    try:
        count = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer, auto, or logical") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("workers must be >= 1")
    return str(count)


def _workers_enabled(args: argparse.Namespace) -> bool:
    return str(args.workers) != DEFAULT_WORKERS


def _worker_suffix(args: argparse.Namespace) -> str:
    if not _workers_enabled(args):
        return ""
    return f" ({args.workers} workers, dist={args.dist})"


def _pytest_env(args: argparse.Namespace) -> dict[str, str] | None:
    env: dict[str, str] = {}
    if args.run_real_llm:
        env["AGENT_LIBOS_RUN_REAL_LLM_BENCHMARK"] = "1"
    if args.keep_agent_outputs:
        env["AGENT_LIBOS_KEEP_AGENT_OUTPUTS"] = "1"
    return env or None


def _is_pytest_command(command: Command) -> bool:
    return len(command.argv) >= 3 and command.argv[1:3] == ["-m", "pytest"]


def _with_invariant_receipt(command: Command, path: Path) -> Command:
    if not _is_pytest_command(command):
        return command
    argv = list(command.argv)
    argv[3:3] = ["-p", "scripts.check_test_invariants"]
    env = dict(command.env or {})
    from scripts.check_test_invariants import INVARIANT_EXECUTION_RECEIPT_ENV

    env[INVARIANT_EXECUTION_RECEIPT_ENV] = str(path)
    return replace(command, argv=argv, env=env)


def _validate_invariant_receipt(
    path: Path,
    *,
    lane: str | None,
    selected_test_paths: tuple[str, ...] | None = None,
    marker_expression: str | None = None,
) -> int:
    from scripts import check_test_invariants as checker

    try:
        executed = checker.load_execution_receipt(path)
        manifest = checker._load_manifest(checker.MANIFEST)
        # A file assigned to this shard may contain only nodes excluded by the
        # active marker expression (for example an MCP-only integration file
        # in the default deterministic providers lane).  Collect the exact
        # marker-selected nodes so those files do not create false evidence
        # failures while selected skips still fail closed.
        selected_nodeids = (
            None
            if selected_test_paths is None
            else checker._collect_pytest_nodeids(
                marker_expression,
                test_paths=selected_test_paths,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"invariant execution evidence failed: {exc}", file=sys.stderr)
        return 1
    errors: list[str] = []
    checker._check_invariant_execution(
        manifest,
        executed,
        errors,
        lane=lane,
        selected_test_paths=selected_test_paths,
        selected_nodeids=selected_nodeids,
    )
    if errors:
        for error in errors:
            print(f"invariant execution evidence failed: {error}", file=sys.stderr)
        return 1
    selected = lane or "all deterministic"
    print(
        f"==> validated non-skipped invariant execution evidence for {selected} "
        f"({len(executed)} passed pytest nodes)",
        flush=True,
    )
    return 0


def _run(command: Command, *, max_seconds: float) -> int:
    print(f"==> {command.name}", flush=True)
    env = os.environ.copy()
    if command.env:
        env.update(command.env)
    started = time.perf_counter()
    windows_job: Any | None = None
    if os.name == "nt":
        from agent_libos.substrate.local import WindowsJobObject

        windows_job = WindowsJobObject.create()
    try:
        process = subprocess.Popen(
            command.argv,
            cwd=ROOT,
            env=env,
            **_process_group_options(),
        )
        if windows_job is not None:
            windows_job.assign(process)
    except BaseException:
        if windows_job is not None:
            windows_job.close()
        raise
    try:
        returncode = process.wait(timeout=max_seconds if command.enforce_timeout else None)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process, windows_job=windows_job)
        elapsed = time.perf_counter() - started
        print(
            f"{command.name} timed out after {elapsed:.2f}s (limit {max_seconds:.2f}s); process tree terminated",
            file=sys.stderr,
        )
        return PROCESS_TIMEOUT_EXIT_CODE
    # A successful root command is not sufficient release evidence when a test
    # or build helper left descendants in the dedicated group.  Reuse the same
    # bounded tree cleanup as the timeout path before reporting the root code.
    _terminate_process_tree(process, windows_job=windows_job)
    elapsed = time.perf_counter() - started
    print(f"==> {command.name} finished in {elapsed:.2f}s", flush=True)
    return returncode


def _process_group_options() -> dict[str, object]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    windows_job: Any | None = None,
) -> None:
    if windows_job is not None:
        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE remains effective after the root
        # command exits, unlike taskkill /T which can no longer discover an
        # orphaned descendant from a dead parent PID.
        windows_job.close()
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    parent_exited = process.poll() is not None
    if parent_exited:
        if os.name == "posix" and not _posix_process_group_exists(process.pid):
            return
        if os.name not in {"posix", "nt"}:
            return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
        else:
            deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
            while _posix_process_group_exists(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.02)
            if _posix_process_group_exists(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
            try:
                process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
            return
    elif os.name == "nt" and process.pid is not None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=PROCESS_TERMINATION_GRACE_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _posix_process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _positive_seconds(value: str) -> float:
    try:
        selected = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not selected > 0 or not selected < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return selected


def _nonnegative_integer(value: str) -> int:
    try:
        selected = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if selected < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return selected


def _positive_integer(value: str) -> int:
    selected = _nonnegative_integer(value)
    if selected < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return selected


def _required_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"required tool is not on PATH: {name}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
