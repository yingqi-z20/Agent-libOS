from __future__ import annotations

import argparse

import pytest

from scripts import test_matrix


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "lane": "runtime",
        "run_real_deno": False,
        "skip_real_deno": False,
        "run_real_llm": False,
        "workers": "1",
        "dist": "loadfile",
        "max_lane_seconds": test_matrix.DEFAULT_MAX_LANE_SECONDS,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestTestMatrix:
    def test_pytest_args_default_to_serial_execution(self) -> None:
        command = test_matrix._pytest_args(("tests/runtime",), _args())

        assert command[:4] == [test_matrix.sys.executable, "-m", "pytest", "tests/runtime"]
        assert "-n" not in command
        assert "--dist" not in command
        assert "not real_deno" not in command
        assert "--skip-real-deno" not in command

    def test_pytest_args_can_skip_real_deno_tests(self) -> None:
        command = test_matrix._pytest_args(("tests/security",), _args(skip_real_deno=True))

        assert "--skip-real-deno" in command
        assert command[-2:] == ["-m", "not real_deno and not real_llm"]

    def test_pytest_args_include_xdist_workers_when_requested(self) -> None:
        command = test_matrix._pytest_args(("tests",), _args(workers="4", dist="load"))

        assert command[:4] == [test_matrix.sys.executable, "-m", "pytest", "tests"]
        assert command[4:8] == ["-n", "4", "--dist", "load"]

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

    def test_non_runtime_lane_defaults_to_serial_execution(self) -> None:
        parser = argparse.ArgumentParser()
        args = _args(lane="security", workers=None, dist=None)

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

    def test_gui_lane_rejects_workers(self) -> None:
        parser = argparse.ArgumentParser()

        with pytest.raises(SystemExit):
            test_matrix._validate_args(parser, _args(lane="gui", workers="2"))
