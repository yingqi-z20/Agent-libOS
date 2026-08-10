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
        assert wrapped.invariant_marker_expression == (
            "not postgres and not real_llm and not mcp"
        )
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

    @pytest.mark.parametrize(
        ("lane", "selected_path", "has_monitor_phase"),
        [
            ("runtime", "tests/runtime", False),
            ("security", "tests/security", True),
            ("self-evolution", "tests/self_evolution", True),
            ("providers", "tests/providers", False),
            ("all", "tests", True),
        ],
    )
    def test_parallel_lane_runs_real_deno_in_a_separate_serial_phase(
        self,
        lane: str,
        selected_path: str,
        has_monitor_phase: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            test_matrix.shutil,
            "which",
            lambda name: "/usr/bin/deno" if name == "deno" else None,
        )

        commands = test_matrix._commands_for(
            _args(lane=lane, workers="4", dist="load")
        )

        assert len(commands) == (3 if has_monitor_phase else 2)
        parallel = commands[1] if has_monitor_phase else commands[0]
        real_deno = commands[-1]
        assert parallel.argv[4:8] == ["-n", "4", "--dist", "load"]
        assert parallel.argv[-2:] == [
            "-m",
            "not postgres and not real_deno and not real_llm and not mcp",
        ]
        expected_general_marker = "not postgres and real_deno"
        if has_monitor_phase:
            expected_general_marker += " and not deno_resource_monitor"
        expected_general_marker += " and not real_llm and not mcp"
        assert real_deno.argv[-2:] == [
            "-m",
            expected_general_marker,
        ]
        assert parallel.invariant_test_paths == (selected_path,)
        assert real_deno.invariant_test_paths == (selected_path,)
        assert parallel.invariant_marker_expression == parallel.argv[-1]
        assert real_deno.invariant_marker_expression == real_deno.argv[-1]
        assert not parallel.allow_no_tests
        assert real_deno.allow_no_tests
        assert real_deno.argv[4:6] == ["-n", "0"]
        assert "--dist" not in real_deno.argv
        if has_monitor_phase:
            monitor = commands[0]
            assert monitor.argv[4:6] == ["-n", "0"]
            assert "--dist" not in monitor.argv
            assert monitor.argv[-2:] == [
                "-m",
                "not postgres and real_deno and deno_resource_monitor "
                "and not real_llm and not mcp",
            ]
            assert monitor.invariant_test_paths == (selected_path,)
            assert monitor.invariant_marker_expression == monitor.argv[-1]
            assert monitor.allow_no_tests

    @pytest.mark.parametrize(
        "overrides",
        [
            {"skip_real_deno": True, "workers": "4"},
            {"skip_real_deno": False, "workers": "1"},
        ],
    )
    def test_real_deno_phase_is_not_added_when_skipped_or_already_serial(
        self,
        overrides: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(test_matrix.shutil, "which", lambda _name: "/usr/bin/deno")

        commands = test_matrix._commands_for(
            _args(lane="self-evolution", **overrides)
        )

        assert len(commands) == 1
        if overrides["skip_real_deno"]:
            assert "--skip-real-deno" in commands[0].argv
            assert "not real_deno" in commands[0].argv[-1]

    def test_real_deno_phase_is_not_added_when_deno_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(test_matrix.shutil, "which", lambda _name: None)

        commands = test_matrix._commands_for(
            _args(lane="security", workers="4", dist="worksteal")
        )

        assert len(commands) == 1
        assert "not real_deno" not in commands[0].argv[-1]

    def test_real_deno_phase_preserves_mcp_and_real_llm_filters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(test_matrix.shutil, "which", lambda _name: "/usr/bin/deno")

        commands = test_matrix._commands_for(
            _args(
                lane="providers",
                workers="3",
                dist="worksteal",
                run_real_llm=True,
                run_mcp=True,
            )
        )

        assert commands[0].argv[-2:] == [
            "-m",
            "not postgres and not real_deno",
        ]
        assert commands[1].argv[-2:] == [
            "-m",
            "not postgres and real_deno",
        ]
        for command in commands:
            assert "--run-real-llm" in command.argv
            assert "--run-mcp" in command.argv

    def test_real_deno_shard_reuses_paths_and_names_the_shard(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(test_matrix.shutil, "which", lambda _name: "/usr/bin/deno")
        args = _args(
            lane="self-evolution",
            workers="4",
            dist="worksteal",
            shard_count=2,
            shard_index=1,
        )

        commands = test_matrix._commands_for(args)

        assert len(commands) == 3
        selected_paths = commands[0].invariant_test_paths
        assert selected_paths
        assert all(command.invariant_test_paths == selected_paths for command in commands)
        assert all("shard 2/2" in command.name for command in commands)
        assert "workers" not in commands[0].name
        assert "workers" not in commands[2].name
        assert "workers" in commands[1].name

    def test_optional_real_deno_phase_accepts_only_pytest_no_tests_status(self) -> None:
        optional = test_matrix.Command(
            "optional real Deno",
            [test_matrix.sys.executable, "-m", "pytest"],
            invariant_marker_expression="real_deno",
            allow_no_tests=True,
        )
        required = test_matrix.Command(
            "required pytest",
            [test_matrix.sys.executable, "-m", "pytest"],
        )
        non_pytest = test_matrix.Command(
            "unrelated helper",
            ["helper"],
            allow_no_tests=True,
        )

        assert test_matrix._normalize_command_status(optional, 5) == 0
        assert test_matrix._normalize_command_status(optional, 1) == 1
        assert test_matrix._normalize_command_status(required, 5) == 5
        assert test_matrix._normalize_command_status(non_pytest, 5) == 5

    def test_main_propagates_parallel_phase_failure_before_serial_phase(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        commands = [
            test_matrix.Command(
                "parallel",
                [test_matrix.sys.executable, "-m", "pytest", "tests/unit"],
            ),
            test_matrix.Command(
                "real Deno",
                [test_matrix.sys.executable, "-m", "pytest", "tests/unit"],
                allow_no_tests=True,
            ),
        ]
        invoked: list[str] = []
        monkeypatch.setattr(test_matrix, "_commands_for", lambda _args: commands)

        def run(command: test_matrix.Command, *, max_seconds: float) -> int:
            del max_seconds
            invoked.append(command.name)
            return 1

        monkeypatch.setattr(test_matrix, "_run", run)

        assert test_matrix.main(["--lane", "unit"]) == 1
        assert invoked == ["parallel"]

    def test_main_accepts_no_tests_only_for_optional_serial_phase(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        command = test_matrix.Command(
            "real Deno",
            [test_matrix.sys.executable, "-m", "pytest", "tests/unit"],
            invariant_test_paths=("tests/unit",),
            invariant_marker_expression="real_deno",
            allow_no_tests=True,
        )
        invoked: list[str] = []
        monkeypatch.setattr(test_matrix, "_commands_for", lambda _args: [command])

        def run(command: test_matrix.Command, *, max_seconds: float) -> int:
            del max_seconds
            invoked.append(command.name)
            assert command.env is not None
            receipt = command.env["AGENT_LIBOS_INVARIANT_EXECUTION_RECEIPT"]
            Path(receipt).write_text(
                '{"passed_node_ids": [], "schema_version": 1}',
                encoding="utf-8",
            )
            return test_matrix.PYTEST_NO_TESTS_EXIT_CODE

        monkeypatch.setattr(test_matrix, "_run", run)

        assert test_matrix.main(["--lane", "unit"]) == 0
        assert invoked == ["real Deno"]

    def test_pytest_args_include_requested_slowest_durations(self) -> None:
        command = test_matrix._pytest_args(("tests/security",), _args(durations=25))

        assert command[4:6] == ["--durations", "25"]

    @pytest.mark.parametrize(
        ("lane", "shard_count"),
        [("runtime", 2), ("providers", 3)],
    )
    def test_lane_shards_are_complete_disjoint_and_weight_balanced(
        self,
        lane: str,
        shard_count: int,
    ) -> None:
        commands = [
            test_matrix._commands_for(
                _args(
                    lane=lane,
                    shard_count=shard_count,
                    shard_index=shard_index,
                )
            )[0]
            for shard_index in range(shard_count)
        ]
        shard_paths = [
            set(command.invariant_test_paths or ()) for command in commands
        ]
        expected = {
            path.relative_to(test_matrix.ROOT).as_posix()
            for path in (
                test_matrix.ROOT / test_matrix.LANE_PATHS[lane][0]
            ).glob("test_*.py")
        }

        assert [command.name for command in commands] == [
            f"pytest {lane} shard {index + 1}/{shard_count}"
            for index in range(shard_count)
        ]
        assert all(shard_paths)
        combined = set().union(*shard_paths)
        assert sum(len(paths) for paths in shard_paths) == len(combined)
        assert combined == expected
        weights = [
            sum((test_matrix.ROOT / path).stat().st_size for path in paths)
            for paths in shard_paths
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
