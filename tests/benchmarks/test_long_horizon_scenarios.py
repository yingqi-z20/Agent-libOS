from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.models import ProcessStatus
from benchmarks.long_horizon_agent import (
    DEFAULT_SCENARIO,
    SCENARIO_ID,
    SCENARIOS,
    LongHorizonScenario,
    evaluate_run,
    prepare_workspace,
    run_evaluation,
)
from benchmarks.long_horizon_agent import runner
from benchmarks.long_horizon_agent.runner import (
    DEFAULT_MAX_QUANTA,
    DEFAULT_PHASE_ONE_QUANTA,
    GOAL,
    MIDFLIGHT_MESSAGE,
    REQUIRED_ACTIONS,
    REQUIRED_SKILLS,
    UNITTEST_ARGV,
    _action_batch_summary,
    _annotate_workflow_evidence,
    _grant_authority,
    _json_bytes,
    _llm_call_diagnostics,
    _normalize_unittest_argv,
    _pricing_behavior_probe_source,
    _stop_reason,
    _test_regression_coverage,
    _workflow_order_checks,
)
from experiments import run_long_horizon_evaluation as long_horizon_cli
from experiments.inspect_long_horizon_run import split_prompt_sections

ROOT = Path(__file__).resolve().parents[2]

# The default scenario's per-run check identifiers, in report order.  Any
# change here is a report-contract change for existing consumers.
EXPECTED_DEFAULT_CHECKS = (
    "exited",
    "restart_survived",
    "full_tests_pass",
    "exact_threshold_regression",
    "zero_quantity_regression",
    "exact_threshold_behavior",
    "zero_quantity_behavior",
    "public_signature_stable",
    "required_skills_activated",
    "required_actions_observed",
    "required_actions_successful",
    "baseline_reproduced_before_edit",
    "finalization_evidence_fresh",
    "checkpoint_created",
    "only_expected_files_changed",
    "untrusted_delete_instruction_ignored",
)

# Literal top-level headings emitted by ``agent_libos.llm.prompt`` and the
# inspector section each must be attributed to.
PROMPT_HEADINGS = {
    "Retained original goal contract (authoritative across restarts):": (
        "original_goal"
    ),
    "Loaded skills:": "loaded_skills",
    "Materialized context:": "materialized_context",
    "Current runtime state (volatile; applies only to this quantum):": (
        "runtime_state_heading"
    ),
    "Process facts:": "process_facts",
    "Materialized context metadata (volatile):": "context_metadata",
    "Capabilities:": "capabilities",
    "Recent events:": "recent_events",
    "Pending explicit process input (mandatory control action):": (
        "pending_process_input"
    ),
}

_NOT_WINDOWS = pytest.mark.skipif(
    os.name == "nt",
    reason="Windows Host oracle fails closed without SubprocessLimits",
)


def test_default_scenario_aliases_the_module_level_contract() -> None:
    assert isinstance(DEFAULT_SCENARIO, LongHorizonScenario)
    assert DEFAULT_SCENARIO.scenario_id == SCENARIO_ID == "durable_pricing_maintenance"
    assert DEFAULT_SCENARIO.goal is GOAL
    assert DEFAULT_SCENARIO.midflight_message is MIDFLIGHT_MESSAGE
    assert DEFAULT_SCENARIO.required_skills is REQUIRED_SKILLS
    assert DEFAULT_SCENARIO.required_actions is REQUIRED_ACTIONS
    assert DEFAULT_SCENARIO.verification_argv is UNITTEST_ARGV
    assert DEFAULT_SCENARIO.prepare_workspace is prepare_workspace
    assert DEFAULT_SCENARIO.grant_authority is _grant_authority
    assert DEFAULT_SCENARIO.behavior_probe_source is _pricing_behavior_probe_source
    assert DEFAULT_SCENARIO.image_id == "coding-agent:v0"
    assert DEFAULT_SCENARIO.midflight_subject == "Customer follow-up"
    assert DEFAULT_SCENARIO.midflight_check_id == "zero_quantity_regression"
    assert DEFAULT_SCENARIO.default_phase_one_quanta == DEFAULT_PHASE_ONE_QUANTA == 6
    assert DEFAULT_SCENARIO.default_max_quanta == DEFAULT_MAX_QUANTA == 96
    assert DEFAULT_SCENARIO.baseline_failure_markers == ("FAILED", "119.90", "108.00")
    assert DEFAULT_SCENARIO.expected_changed_files == frozenset(
        {"src/pricing.py", "tests/test_pricing.py"}
    )
    assert DEFAULT_SCENARIO.behavior_check_ids == (
        "exact_threshold",
        "zero_quantity",
        "public_signature",
    )
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SCENARIO.goal = "mutated"  # type: ignore[misc]


def test_default_scenario_callables_read_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    source = workspace.joinpath("tests", "test_pricing.py").read_text(encoding="utf-8")

    assert (
        DEFAULT_SCENARIO.regression_coverage(workspace)
        == _test_regression_coverage(source)
        == {"exact_threshold": False, "zero_quantity": False}
    )
    assert DEFAULT_SCENARIO.scenario_checks(workspace) == {
        "untrusted_delete_instruction_ignored": True
    }


def test_scenario_registry_contains_the_default_scenario() -> None:
    assert SCENARIOS[SCENARIO_ID] is DEFAULT_SCENARIO
    assert "durable_pricing_maintenance" in SCENARIOS
    for key, scenario in SCENARIOS.items():
        assert isinstance(scenario, LongHorizonScenario)
        assert scenario.scenario_id == key
        assert scenario.default_max_quanta > scenario.default_phase_one_quanta >= 1


def test_cli_lists_scenarios_before_the_real_llm_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_LANGUAGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        long_horizon_cli,
        "run_evaluation",
        lambda *_args, **_kwargs: pytest.fail("listing must not run an evaluation"),
    )

    assert long_horizon_cli.main(["--list-scenarios"]) is None

    captured = capsys.readouterr()
    assert captured.out.splitlines() == sorted(SCENARIOS)
    assert "durable_pricing_maintenance" in captured.out.splitlines()
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--confirm-real-llm"],
        ["--confirm-real-llm", "--output", "report.json", "--scenario", "missing"],
        [
            "--confirm-real-llm",
            "--output",
            "report.json",
            "--phase-one-quanta",
            "4",
            "--max-quanta",
            "4",
        ],
    ],
)
def test_cli_rejects_invalid_scenario_arguments_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        long_horizon_cli,
        "run_evaluation",
        lambda *_args, **_kwargs: pytest.fail("invalid arguments must not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        long_horizon_cli.main(argv)

    assert exc_info.value.code == 2


def test_cli_applies_scenario_defaults_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    synthetic = replace(
        DEFAULT_SCENARIO,
        scenario_id="synthetic",
        default_phase_one_quanta=2,
        default_max_quanta=5,
    )
    monkeypatch.setitem(runner.SCENARIOS, "synthetic", synthetic)
    captured: list[dict[str, Any]] = []

    def fake_run_evaluation(_root: object, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        kwargs["progress"]("[run 1] phase_one: quanta=1")
        return {"schema_version": 1, "runs": [{"passed": True}]}

    monkeypatch.setattr(long_horizon_cli, "run_evaluation", fake_run_evaluation)
    output = tmp_path / "report.json"

    long_horizon_cli.main(
        ["--confirm-real-llm", "--output", str(output), "--scenario", "synthetic", "--progress"]
    )

    assert len(captured) == 1
    assert captured[0]["scenario_id"] == "synthetic"
    assert captured[0]["phase_one_quanta"] == 2
    assert captured[0]["max_quanta"] == 5
    assert callable(captured[0]["progress"])
    streams = capsys.readouterr()
    assert "[run 1] phase_one: quanta=1" in streams.err
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_run_evaluation_threads_the_selected_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_run_once(_run_root: Path, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {
            "passed": True,
            "checks": {"custom_regression": True, "zero_quantity_regression": False},
            "wall_seconds": 12.0,
            "llm_latency_seconds": 10.0,
            "max_llm_call_seconds": 4.0,
            "overhead_llm_calls": 2,
        }

    monkeypatch.setattr(runner, "_run_once", fake_run_once)
    synthetic = replace(
        DEFAULT_SCENARIO,
        scenario_id="synthetic",
        default_phase_one_quanta=2,
        default_max_quanta=5,
        midflight_check_id="custom_regression",
    )
    monkeypatch.setitem(runner.SCENARIOS, "synthetic", synthetic)
    progress_lines: list[str] = []
    progress = progress_lines.append

    report = run_evaluation(
        tmp_path / "evaluation",
        scenario_id="synthetic",
        progress=progress,
    )

    assert captured[0]["scenario"] is synthetic
    assert captured[0]["phase_one_quanta"] == 2
    assert captured[0]["max_quanta"] == 5
    assert captured[0]["progress"] is progress
    assert report["scenario_id"] == "synthetic"
    assert report["phase_one_quanta"] == 2
    assert report["max_quanta"] == 5
    assert report["metrics"]["midflight_constraint_rate"] == 1.0
    assert report["metrics"]["mean_wall_seconds"] == 12.0
    assert report["metrics"]["mean_llm_latency_seconds"] == 10.0
    assert report["metrics"]["mean_max_llm_call_seconds"] == 4.0
    assert report["metrics"]["mean_overhead_llm_calls"] == 2.0


def test_run_evaluation_defaults_to_the_default_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_run_once(_run_root: Path, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"passed": False, "checks": {}}

    monkeypatch.setattr(runner, "_run_once", fake_run_once)

    report = run_evaluation(tmp_path / "evaluation")

    assert captured[0]["scenario"] is DEFAULT_SCENARIO
    assert captured[0]["phase_one_quanta"] == DEFAULT_PHASE_ONE_QUANTA
    assert captured[0]["max_quanta"] == DEFAULT_MAX_QUANTA
    assert report["scenario_id"] == SCENARIO_ID
    assert report["metrics"]["midflight_constraint_rate"] == 0.0
    assert report["metrics"]["mean_wall_seconds"] == 0.0


def test_run_evaluation_rejects_unknown_scenario_before_opening_a_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "Runtime",
        SimpleNamespace(
            open=lambda *_args, **_kwargs: pytest.fail("Runtime must not open")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_once",
        lambda *_args, **_kwargs: pytest.fail("no run may start"),
    )
    root = tmp_path / "evaluation"

    with pytest.raises(ValueError, match="unknown long-horizon scenario"):
        run_evaluation(root, scenario_id="missing")

    assert not root.exists()


def test_split_prompt_sections_attributes_chars_to_runtime_prompt_headings() -> None:
    prompt_source = (ROOT / "agent_libos" / "llm" / "prompt.py").read_text(
        encoding="utf-8"
    )
    paragraphs = ["Intro paragraph."]
    expected = {"preamble": len("Intro paragraph.") + 2}
    for index, (heading, section) in enumerate(PROMPT_HEADINGS.items()):
        assert heading in prompt_source, heading
        body = f"{heading}\n" + ("x" * (10 + index))
        paragraphs.append(body)
        expected[section] = len(body) + 2

    sizes = split_prompt_sections("\n\n".join(paragraphs))

    assert sizes == expected
    assert sum(sizes.values()) == len("\n\n".join(paragraphs)) + 2


def test_llm_call_diagnostics_summarize_sizes_counts_and_seconds_only() -> None:
    base = datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc)

    def stamp(offset: float) -> str:
        return (base + timedelta(seconds=offset)).isoformat()

    prompt = "Intro\n\nLoaded skills:\nSKILL_TEXT\n\nMaterialized context:\nCTX"
    calls = [
        SimpleNamespace(
            created_at=stamp(0),
            completed_at=stamp(2.5),
            usage={"cache_read_tokens": 4096, "reasoning_tokens": 10},
            messages=[
                {"role": "system", "content": "system text"},
                {"role": "user", "content": prompt},
            ],
            tool_calls=[
                {"name": "activate_skill", "arguments": '{"skill_id":"x"}'},
                {"function": {"name": "read_process_messages", "arguments": "{}"}},
            ],
            tools=[{"name": "a"}],
        ),
        SimpleNamespace(
            created_at=stamp(3),
            completed_at=stamp(4),
            usage={"cache_read_tokens": 1024, "reasoning_tokens": 5},
            messages=[{"role": "user", "content": prompt + "\n\nRecent events:\nEVT"}],
            tool_calls=[{"name": "read_text_file"}, {"name": "write_text_file"}],
            tools=[{"name": "a"}, {"name": "b"}],
        ),
        SimpleNamespace(
            created_at=stamp(5),
            completed_at=None,
            usage={"cache_read_tokens": 8192},
            messages={"$agent_libos_payload_retention": {"tier": "summary"}},
            tool_calls=[],
            tools={"$agent_libos_payload_retention": {"tier": "summary"}},
        ),
    ]

    diagnostics = _llm_call_diagnostics(calls)

    assert diagnostics["llm_call_seconds"] == [2.5, 1.0]
    assert diagnostics["llm_latency_seconds"] == 3.5
    assert diagnostics["max_llm_call_seconds"] == 2.5
    assert diagnostics["reasoning_tokens"] == 15
    assert diagnostics["cache_reset_count"] == 1
    assert diagnostics["overhead_llm_calls"] == 1
    assert diagnostics["tool_calls_by_category"] == {
        "messages": 1,
        "mutate": 1,
        "observe": 1,
        "skill_lifecycle": 1,
    }
    assert diagnostics["tools_bytes_max"] == _json_bytes([{"name": "a"}, {"name": "b"}])
    assert diagnostics["prompt_sections"]["preamble"] == {
        "total_chars": 14,
        "mean_chars": 7.0,
        "max_chars": 7,
    }
    assert diagnostics["prompt_sections"]["recent_events"] == {
        "total_chars": 20,
        "mean_chars": 20.0,
        "max_chars": 20,
    }
    assert set(diagnostics["prompt_sections"]) == {
        "preamble",
        "loaded_skills",
        "materialized_context",
        "recent_events",
    }
    rendered = json.dumps(diagnostics)
    assert "SKILL_TEXT" not in rendered
    assert "skill_id" not in rendered
    assert "system text" not in rendered


def test_llm_call_diagnostics_handle_no_calls() -> None:
    assert _llm_call_diagnostics([]) == {
        "llm_latency_seconds": 0.0,
        "max_llm_call_seconds": 0.0,
        "llm_call_seconds": [],
        "reasoning_tokens": 0,
        "prompt_sections": {},
        "tools_bytes_max": 0,
        "tool_calls_by_category": {},
        "overhead_llm_calls": 0,
        "cache_reset_count": 0,
    }


def test_action_batch_summary_counts_batches_without_copying_payloads() -> None:
    records = [
        SimpleNamespace(
            action="llm.action_batch",
            decision={
                "requested_count": 3,
                "executed_count": 3,
                "stop_reason": "completed",
                "actions": [{"action": "read_text_file", "path": "SECRET_PATH"}],
            },
        ),
        SimpleNamespace(
            action="llm.action_batch",
            decision={"requested_count": 4, "executed_count": 1, "stop_reason": "tool_failed"},
        ),
        SimpleNamespace(action="llm.action_batch", decision=None),
        SimpleNamespace(action="llm.action", decision={"requested_count": 9}),
    ]

    summary = _action_batch_summary(records)

    assert summary == {
        "count": 3,
        "requested": 7,
        "executed": 4,
        "stop_reasons": {"completed": 1, "tool_failed": 1, "unknown": 1},
    }
    assert "SECRET_PATH" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("status", "wait_kind", "budget_exhausted", "expected"),
    [
        (ProcessStatus.EXITED, None, False, "exited"),
        ("failed", None, True, "failed"),
        (ProcessStatus.KILLED, None, False, "killed"),
        (ProcessStatus.RUNNABLE, None, True, "budget_exhausted"),
        (ProcessStatus.WAITING_HUMAN, "human", False, "waiting_human"),
        (ProcessStatus.WAITING_EVENT, "message", False, "waiting_event"),
        (ProcessStatus.WAITING_EVENT, "child", False, "waiting_event"),
        (ProcessStatus.PAUSED, "host_resume", False, "paused"),
        (ProcessStatus.PAUSED, None, False, "paused"),
        (ProcessStatus.WAITING_TOOL, "tool", False, "other"),
        (ProcessStatus.RUNNABLE, None, False, "other"),
    ],
)
def test_stop_reason_classifies_final_process_state(
    status: object,
    wait_kind: str | None,
    budget_exhausted: bool,
    expected: str,
) -> None:
    wait_state = SimpleNamespace(kind=wait_kind) if wait_kind else None

    assert _stop_reason(status, wait_state, budget_exhausted=budget_exhausted) == expected


@_NOT_WINDOWS
def test_default_scenario_check_ids_and_report_keys_are_unchanged(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)

    result = evaluate_run(
        workspace,
        status="runnable",
        actions=[],
        activated_skills=[],
        checkpoint_count=0,
        restart_survived=False,
    )

    assert tuple(result["checks"]) == EXPECTED_DEFAULT_CHECKS
    assert tuple(result) == (
        "passed",
        "checks",
        "changed_files",
        "test_returncode",
        "test_output_tail",
        "behavior_probe",
        "host_oracle",
        "workflow_evidence",
    )
    assert result["checks"]["untrusted_delete_instruction_ignored"] is True
    assert result["checks"]["public_signature_stable"] is True


@_NOT_WINDOWS
def test_evaluate_run_builds_checks_generically_for_another_scenario(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    seen: dict[str, Path] = {}

    def coverage(root: Path) -> dict[str, bool]:
        seen["coverage"] = root
        return {"custom": True}

    def extra_checks(root: Path) -> dict[str, bool]:
        seen["checks"] = root
        return {"fixture_intact": root.joinpath("AGENTS.md").is_file()}

    synthetic = replace(
        DEFAULT_SCENARIO,
        scenario_id="synthetic",
        regression_coverage=coverage,
        behavior_check_ids=("custom", "public_signature"),
        scenario_checks=extra_checks,
        required_skills=frozenset({"agent-libos-checkpoints"}),
        expected_changed_files=frozenset({"src/pricing.py"}),
    )

    result = evaluate_run(
        workspace,
        status="runnable",
        actions=[],
        activated_skills=["agent-libos-checkpoints"],
        checkpoint_count=0,
        restart_survived=False,
        scenario=synthetic,
    )

    checks = result["checks"]
    assert checks["custom_regression"] is True
    assert checks["custom_behavior"] is False
    assert checks["public_signature_stable"] is True
    assert checks["fixture_intact"] is True
    assert checks["required_skills_activated"] is True
    assert "exact_threshold_regression" not in checks
    assert "zero_quantity_behavior" not in checks
    assert "untrusted_delete_instruction_ignored" not in checks
    assert seen == {"coverage": workspace.resolve(), "checks": workspace.resolve()}


def test_workflow_oracle_uses_the_scenario_verification_argv_and_markers() -> None:
    pytest_argv = ("python", "-m", "pytest", "-q")
    pytest_scenario = replace(
        DEFAULT_SCENARIO,
        scenario_id="pytest_variant",
        verification_argv=pytest_argv,
        baseline_failure_markers=("FAILED", "AssertionError"),
    )
    baseline_output = "1 failed\nFAILED tests/test_x.py::test_x - AssertionError"
    evidence = _workflow_evidence(pytest_argv, baseline_output)

    assert _workflow_order_checks(evidence) == {
        "baseline_reproduced_before_edit": False,
        "finalization_evidence_fresh": False,
    }
    assert _workflow_order_checks(evidence, scenario=pytest_scenario) == {
        "baseline_reproduced_before_edit": True,
        "finalization_evidence_fresh": True,
    }
    missing_marker = _workflow_evidence(pytest_argv, "1 failed")
    assert (
        _workflow_order_checks(missing_marker, scenario=pytest_scenario)[
            "baseline_reproduced_before_edit"
        ]
        is False
    )
    assert _normalize_unittest_argv(list(pytest_argv), expected_argv=pytest_argv) == pytest_argv
    assert _normalize_unittest_argv(list(pytest_argv)) is None
    assert _normalize_unittest_argv(list(UNITTEST_ARGV)) == UNITTEST_ARGV

    annotated = _annotate_workflow_evidence(evidence, scenario=pytest_scenario)
    expectations = [
        receipt.get("semantic_expectation")
        for receipt in annotated
        if receipt["action"] == "run_shell_command"
    ]
    assert expectations == ["baseline_known_defect", "final_full_suite"]
    assert all(
        "semantic_expectation" not in receipt
        for receipt in _annotate_workflow_evidence(evidence)
    )


def _workflow_evidence(
    argv: tuple[str, ...],
    baseline_output: str,
) -> list[dict[str, Any]]:
    names = (
        "read_text_file",
        "run_shell_command",
        "write_text_file",
        "run_shell_command",
        "git_status",
        "git_diff",
        "create_checkpoint",
        "human_output",
        "process_exit",
    )
    evidence: list[dict[str, Any]] = []
    shell_index = 0
    for index, name in enumerate(names):
        receipt: dict[str, Any] = {
            "sequence_index": index,
            "action": name,
            "ok": True,
            "tool_id": f"tool:{name}",
            "result_oid": f"result:{index}",
        }
        if name == "process_exit":
            receipt.update({"status": "exited", "terminal_committed": True})
        if name == "run_shell_command":
            baseline = shell_index == 0
            receipt.update(
                {
                    "requested_argv": list(argv),
                    "observed_argv": list(argv),
                    "returncode": 1 if baseline else 0,
                    "stdout": baseline_output if baseline else "OK",
                    "stderr": "",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "limit_kind": None,
                }
            )
            shell_index += 1
        evidence.append(receipt)
    return evidence
