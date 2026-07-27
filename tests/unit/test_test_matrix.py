from __future__ import annotations

import argparse
from pathlib import Path
import time

import psutil
import pytest

from scripts import test_matrix


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "lane": "runtime",
        "run_real_deno": False,
        "skip_real_deno": False,
        "run_real_llm": False,
        "run_mcp": False,
        "keep_agent_outputs": False,
        "workers": "1",
        "dist": "loadfile",
        "max_lane_seconds": test_matrix.DEFAULT_MAX_LANE_SECONDS,
        "durations": None,
        "shard_count": 1,
        "shard_index": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestTestMatrix:

    def test_default_lane_timeout_has_headroom_for_full_deterministic_matrix(self) -> None:
        assert test_matrix.DEFAULT_MAX_LANE_SECONDS == 600.0

    def test_pytest_args_default_to_serial_execution(self) -> None:
        command = test_matrix._pytest_args(("tests/runtime",), _args())

        assert command[:4] == [test_matrix.sys.executable, "-m", "pytest", "tests/runtime"]
        assert "-n" not in command
        assert "--dist" not in command
        assert command[-2:] == ["-m", "not postgres and not real_llm and not mcp"]
        assert "not real_deno" not in command
        assert "--skip-real-deno" not in command

    def test_pytest_args_can_skip_real_deno_tests(self) -> None:
        command = test_matrix._pytest_args(("tests/security",), _args(skip_real_deno=True))

        assert "--skip-real-deno" in command
        assert command[-2:] == [
            "-m",
            "not postgres and not real_deno and not real_llm and not mcp",
        ]

    def test_pytest_args_can_run_mcp_tests(self) -> None:
        command = test_matrix._pytest_args(("tests/providers",), _args(run_mcp=True))

        assert "--run-mcp" in command
        assert command[-2:] == ["-m", "not postgres and not real_llm"]

    def test_keep_agent_outputs_sets_pytest_environment(self) -> None:
        assert test_matrix._pytest_env(_args(keep_agent_outputs=True)) == {
            "AGENT_LIBOS_KEEP_AGENT_OUTPUTS": "1"
        }

    def test_pytest_command_receives_execution_receipt_plugin(
        self,
        tmp_path: Path,
    ) -> None:
        command = test_matrix._commands_for(_args(lane="unit"))[0]
        receipt = tmp_path / "receipt.json"

        wrapped = test_matrix._with_invariant_receipt(command, receipt)

        assert wrapped.argv[3:5] == ["-p", "scripts.check_test_invariants"]
        assert wrapped.env == {
            "AGENT_LIBOS_INVARIANT_EXECUTION_RECEIPT": str(receipt)
        }
        assert command.env is None

    def test_pytest_environment_combines_real_llm_and_output_retention(self) -> None:
        assert test_matrix._pytest_env(
            _args(run_real_llm=True, keep_agent_outputs=True)
        ) == {
            "AGENT_LIBOS_RUN_REAL_LLM_BENCHMARK": "1",
            "AGENT_LIBOS_KEEP_AGENT_OUTPUTS": "1",
        }

    def test_pytest_args_include_xdist_workers_when_requested(self) -> None:
        command = test_matrix._pytest_args(("tests",), _args(workers="4", dist="load"))

        assert command[:4] == [test_matrix.sys.executable, "-m", "pytest", "tests"]
        assert command[4:8] == ["-n", "4", "--dist", "load"]

    def test_pytest_args_include_requested_slowest_durations(self) -> None:
        command = test_matrix._pytest_args(("tests/security",), _args(durations=25))

        assert command[4:6] == ["--durations", "25"]

    def test_runtime_shards_are_complete_disjoint_and_weight_balanced(self) -> None:
        first = test_matrix._commands_for(
            _args(lane="runtime", shard_count=2, shard_index=0)
        )[0]
        second = test_matrix._commands_for(
            _args(lane="runtime", shard_count=2, shard_index=1)
        )[0]
        first_paths = set(first.invariant_test_paths or ())
        second_paths = set(second.invariant_test_paths or ())
        expected = {
            path.relative_to(test_matrix.ROOT).as_posix()
            for path in (test_matrix.ROOT / "tests/runtime").glob("test_*.py")
        }

        assert first.name == "pytest runtime shard 1/2"
        assert second.name == "pytest runtime shard 2/2"
        assert first_paths
        assert second_paths
        assert first_paths.isdisjoint(second_paths)
        assert first_paths | second_paths == expected
        weights = [
            sum((test_matrix.ROOT / path).stat().st_size for path in paths)
            for paths in (first_paths, second_paths)
        ]
        assert max(weights) - min(weights) <= max(
            (test_matrix.ROOT / path).stat().st_size for path in expected
        )

    def test_runtime_lane_defaults_to_bounded_parallel_worksteal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = argparse.ArgumentParser()
        args = _args(workers=None, dist=None)
        monkeypatch.setattr(test_matrix.os, "cpu_count", lambda: 12)

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == str(test_matrix.DEFAULT_PARALLEL_WORKER_CAP)
        assert args.dist == "worksteal"

    def test_all_lane_defaults_to_bounded_parallel_worksteal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = argparse.ArgumentParser()
        args = _args(lane="all", workers=None, dist=None)
        monkeypatch.setattr(test_matrix.os, "cpu_count", lambda: 2)

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == "2"
        assert args.dist == "worksteal"

    def test_all_lane_uses_the_same_hard_timeout_contract(self) -> None:
        command = test_matrix._commands_for(_args(lane="all"))[0]

        assert command.enforce_timeout is True

    def test_gui_commands_each_use_the_hard_timeout_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(test_matrix, "_required_tool", lambda _name: "npm")

        commands = test_matrix._commands_for(_args(lane="gui"))

        assert [command.name for command in commands] == [
            "gui unit tests",
            "gui typecheck",
            "gui build",
        ]
        assert all(command.enforce_timeout for command in commands)

    @pytest.mark.parametrize("lane", ["security", "self-evolution", "providers"])
    def test_long_lane_defaults_to_bounded_parallel_worksteal(
        self,
        lane: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parser = argparse.ArgumentParser()
        args = _args(lane=lane, workers=None, dist=None)
        monkeypatch.setattr(test_matrix.os, "cpu_count", lambda: 12)

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == str(test_matrix.DEFAULT_PARALLEL_WORKER_CAP)
        assert args.dist == "worksteal"

    @pytest.mark.parametrize("lane", ["unit", "benchmark"])
    def test_short_lane_defaults_to_serial_execution(self, lane: str) -> None:
        parser = argparse.ArgumentParser()
        args = _args(lane=lane, workers=None, dist=None)

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == "1"
        assert args.dist == "loadfile"

    def test_explicit_workers_override_parallel_default(self) -> None:
        parser = argparse.ArgumentParser()
        args = _args(workers="1", dist=None)

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == "1"
        assert args.dist == "loadfile"

    def test_worker_env_overrides_lane_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = argparse.ArgumentParser()
        args = _args(lane="security", workers=None, dist=None)
        monkeypatch.setenv(test_matrix.WORKERS_ENV, "3")
        monkeypatch.setenv(test_matrix.DIST_ENV, "load")

        test_matrix._resolve_defaults(parser, args)

        assert args.workers == "3"
        assert args.dist == "load"

    def test_invalid_worker_env_reports_parser_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = argparse.ArgumentParser()
        args = _args(workers=None, dist=None)
        monkeypatch.setenv(test_matrix.WORKERS_ENV, "maybe")

        with pytest.raises(SystemExit):
            test_matrix._resolve_defaults(parser, args)

    def test_worker_count_accepts_positive_int_auto_and_logical(self) -> None:
        assert test_matrix._worker_count("4") == "4"
        assert test_matrix._worker_count("auto") == "auto"
        assert test_matrix._worker_count("logical") == "logical"

    def test_worker_count_rejects_invalid_values(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            test_matrix._worker_count("0")
        with pytest.raises(argparse.ArgumentTypeError):
            test_matrix._worker_count("maybe")

    def test_max_lane_seconds_requires_a_finite_positive_value(self) -> None:
        assert test_matrix._positive_seconds("0.25") == 0.25
        for value in ("0", "-1", "nan", "inf", "not-a-number"):
            with pytest.raises(argparse.ArgumentTypeError):
                test_matrix._positive_seconds(value)

    def test_durations_requires_a_nonnegative_integer(self) -> None:
        assert test_matrix._nonnegative_integer("0") == 0
        assert test_matrix._nonnegative_integer("25") == 25
        for value in ("-1", "1.5", "not-a-number"):
            with pytest.raises(argparse.ArgumentTypeError):
                test_matrix._nonnegative_integer(value)

    def test_shard_count_requires_a_positive_integer(self) -> None:
        assert test_matrix._positive_integer("2") == 2
        for value in ("0", "-1", "1.5", "not-a-number"):
            with pytest.raises(argparse.ArgumentTypeError):
                test_matrix._positive_integer(value)

    def test_individual_lane_timeout_terminates_the_command(self) -> None:
        started = time.monotonic()

        status = test_matrix._run(
            test_matrix.Command(
                "timeout regression child",
                [test_matrix.sys.executable, "-c", "import time; time.sleep(30)"],
            ),
            max_seconds=0.05,
        )

        assert status == test_matrix.PROCESS_TIMEOUT_EXIT_CODE
        assert time.monotonic() - started < 5

    def test_timeout_terminates_a_spawned_descendant(self, tmp_path: Path) -> None:
        child_pid_file = tmp_path / "child.pid"
        child_ready_file = tmp_path / "child.ready"
        child_code = """
import pathlib
import signal
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text("ready")
time.sleep(30)
"""
        parent_code = """
import pathlib
import subprocess
import sys
import time

ready = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", sys.argv[2], str(ready)])
deadline = time.monotonic() + 5
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not ready.exists():
    raise RuntimeError("descendant did not become ready")
pathlib.Path(sys.argv[3]).write_text(str(child.pid))
time.sleep(30)
"""

        status = test_matrix._run(
            test_matrix.Command(
                "process-tree timeout regression",
                [
                    test_matrix.sys.executable,
                    "-c",
                    parent_code,
                    str(child_ready_file),
                    child_code,
                    str(child_pid_file),
                ],
            ),
            max_seconds=1,
        )

        assert status == test_matrix.PROCESS_TIMEOUT_EXIT_CODE
        assert child_pid_file.exists(), "parent did not spawn the descendant before timeout"
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        try:
            final_status = psutil.Process(child_pid).status()
        except psutil.NoSuchProcess:
            final_status = None
        assert final_status in {None, psutil.STATUS_ZOMBIE}

    def test_successful_command_terminates_a_spawned_descendant(
        self,
        tmp_path: Path,
    ) -> None:
        child_pid_file = tmp_path / "successful-child.pid"
        child_code = """
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""
        parent_code = """
import pathlib
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", sys.argv[1]])
pathlib.Path(sys.argv[2]).write_text(str(child.pid), encoding="utf-8")
"""

        status = test_matrix._run(
            test_matrix.Command(
                "successful process-tree regression",
                [
                    test_matrix.sys.executable,
                    "-c",
                    parent_code,
                    child_code,
                    str(child_pid_file),
                ],
            ),
            max_seconds=2,
        )

        assert status == 0
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        try:
            final_status = psutil.Process(child_pid).status()
        except psutil.NoSuchProcess:
            final_status = None
        assert final_status in {None, psutil.STATUS_ZOMBIE}

    @pytest.mark.skipif(test_matrix.os.name != "posix", reason="POSIX process-group assertion")
    def test_timeout_commands_start_in_a_new_posix_session(self) -> None:
        assert test_matrix._process_group_options() == {"start_new_session": True}

    def test_gui_lane_rejects_workers(self) -> None:
        parser = argparse.ArgumentParser()

        with pytest.raises(SystemExit):
            test_matrix._validate_args(parser, _args(lane="gui", workers="2"))

    def test_gui_lane_rejects_agent_output_retention_flag(self) -> None:
        parser = argparse.ArgumentParser()

        with pytest.raises(SystemExit):
            test_matrix._validate_args(
                parser,
                _args(lane="gui", keep_agent_outputs=True),
            )

    def test_sharding_rejects_invalid_index_and_aggregate_lanes(self) -> None:
        parser = argparse.ArgumentParser()

        with pytest.raises(SystemExit):
            test_matrix._validate_args(
                parser,
                _args(shard_count=2, shard_index=2),
            )
        with pytest.raises(SystemExit):
            test_matrix._validate_args(
                parser,
                _args(lane="all", shard_count=2, shard_index=0),
            )
