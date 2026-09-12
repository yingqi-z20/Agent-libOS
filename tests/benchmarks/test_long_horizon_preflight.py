from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos.config import DEFAULT_CONFIG
from benchmarks.long_horizon_agent import runner
from experiments import run_long_horizon_evaluation as long_horizon_cli


def _successful_probe() -> dict[str, Any]:
    return {
        "completed": True,
        "returncode": 0,
        "stdout": "agent-libos-host-oracle-preflight-v1\n",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "limit_kind": None,
        "argv_is_absolute": True,
    }


def _fixture_without_shell(workspace: Path) -> None:
    workspace.mkdir()
    # The preflight must not need candidate Python imports or execution.
    (workspace / "sitecustomize.py").write_text(
        "raise RuntimeError('candidate code must not run during preflight')\n"
    )


@pytest.mark.parametrize(
    "failure",
    [
        {
            "completed": False,
            "limit_kind": "host_oracle_error",
            "error_type": "ValidationError",
        },
        {"returncode": 1, "stderr": "private Host error canary"},
        {"stdout_truncated": True},
        {"stdout": "wrong probe output"},
    ],
)
def test_failed_preflight_stops_before_runtime_or_llm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: dict[str, Any]
) -> None:
    def probe(self: runner.HostOracleRunner, source: str) -> dict[str, Any]:
        assert self.workspace.is_dir()
        assert source == "print('agent-libos-host-oracle-preflight-v1')"
        return {**_successful_probe(), **failure}

    monkeypatch.setattr(runner.HostOracleRunner, "run_isolated_python", probe)
    monkeypatch.setattr(
        runner.Runtime,
        "open",
        lambda *args, **kwargs: pytest.fail(
            "Runtime/LLM must not start after failed preflight"
        ),
    )
    scenario = replace(
        runner.DEFAULT_SCENARIO, prepare_workspace=_fixture_without_shell
    )
    with pytest.raises(RuntimeError, match="before any LLM call") as error:
        runner._run_once(
            tmp_path / "run",
            repetition=1,
            phase_one_quanta=1,
            max_quanta=2,
            config=DEFAULT_CONFIG,
            scenario=scenario,
        )
    assert "process-tree monitoring" in str(error.value)
    assert "private Host error canary" not in str(error.value)
    assert not (tmp_path / "run" / "state" / "runtime.sqlite").exists()


def test_successful_preflight_continues_to_runtime_with_same_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def probe(self: runner.HostOracleRunner, source: str) -> dict[str, Any]:
        assert source == "print('agent-libos-host-oracle-preflight-v1')"
        events.append("preflight")
        return _successful_probe()

    class RuntimeReached(Exception):
        pass

    def open_runtime(database: Path, **kwargs: Any) -> None:
        assert events == ["preflight"]
        assert database == tmp_path / "run" / "state" / "runtime.sqlite"
        assert kwargs["config"] is DEFAULT_CONFIG
        assert isinstance(kwargs["substrate"], runner.LocalResourceProviderSubstrate)
        events.append("runtime")
        raise RuntimeReached

    monkeypatch.setattr(runner.HostOracleRunner, "run_isolated_python", probe)
    monkeypatch.setattr(runner.Runtime, "open", open_runtime)
    scenario = replace(
        runner.DEFAULT_SCENARIO, prepare_workspace=_fixture_without_shell
    )
    with pytest.raises(RuntimeReached):
        runner._run_once(
            tmp_path / "run",
            repetition=1,
            phase_one_quanta=1,
            max_quanta=2,
            config=DEFAULT_CONFIG,
            scenario=scenario,
        )
    assert events == ["preflight", "runtime"]


def test_preflight_setup_exception_is_sanitized_and_cli_marks_report_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_setup(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("private Host path canary")

    monkeypatch.setattr(runner, "HostOracleRunner", denied_setup)
    monkeypatch.setattr(
        runner.Runtime,
        "open",
        lambda *args, **kwargs: pytest.fail(
            "Runtime/LLM must not start after failed preflight"
        ),
    )
    monkeypatch.setitem(
        runner.SCENARIOS,
        runner.SCENARIO_ID,
        replace(runner.DEFAULT_SCENARIO, prepare_workspace=_fixture_without_shell),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-no-provider-call")
    monkeypatch.setenv("OPENAI_MODEL", "test-only-model")
    output = tmp_path / "report.json"
    output.write_text('{"runs": [{"passed": true}]}\n')

    with pytest.raises(RuntimeError, match="before any LLM call") as error:
        long_horizon_cli.main(
            [
                "--confirm-real-llm",
                "--output",
                str(output),
                "--artifacts-root",
                str(tmp_path / "artifacts"),
            ]
        )

    assert "private Host path canary" not in str(error.value)
    assert error.value.__suppress_context__ is True
    marker = json.loads(output.read_text())["evaluation_artifact"]
    assert marker["completion_state"] == "failed"
    previous = output.with_name(marker["previous_artifact"])
    assert json.loads(previous.read_text()) == {"runs": [{"passed": True}]}
