from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.practical_agent_workflows.loader import load_scenarios
from benchmarks.practical_agent_workflows.metrics import collect_metrics, write_metrics
from benchmarks.practical_agent_workflows.reports import write_reports
from benchmarks.practical_agent_workflows.runners import RUNNER_METADATA, RUNNER_NAMES, run_suite, write_run_outputs

SUITE_ROOT = Path("benchmarks/practical_agent_workflows")


class TestPracticalAgentWorkflows:
    def test_loads_practical_scenario_matrix(self) -> None:
        scenarios = load_scenarios(SUITE_ROOT)
        assert len(scenarios) == 80
        assert {scenario.domain for scenario in scenarios} == {"coding", "research", "devops", "enterprise", "self_evolution"}
        assert {scenario.track for scenario in scenarios} == {"coding", "research", "devops", "enterprise", "self_evolution"}
        assert {scenario.variant for scenario in scenarios} == {"benign", "attack", "adaptive", "long_horizon"}
        assert len({scenario.task_family for scenario in scenarios}) == 8
        assert all(scenario.allowed_effects for scenario in scenarios)
        assert all(scenario.forbidden_effects for scenario in scenarios if scenario.variant != "benign")
        assert all(scenario.expected_provenance for scenario in scenarios)
        assert all(scenario.utility_oracle for scenario in scenarios)
        assert all(scenario.security_oracle for scenario in scenarios)
        assert all(scenario.state_diff_oracle for scenario in scenarios)
        assert all(scenario.runtime_calls for scenario in scenarios)
        assert all(scenario.evidence_level == "modeled+live-runtime" for scenario in scenarios)

    def test_deterministic_run_writes_required_artifacts(self) -> None:
        scenarios = [scenario for scenario in load_scenarios(SUITE_ROOT) if scenario.domain == "coding"][:3]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(scenarios, temp_dir, runners=["direct_tool_agent", "agent_libos"])
            write_run_outputs(runs, temp_dir)
            metrics = write_metrics(temp_dir)
            reports = write_reports(temp_dir)

            assert len(runs) == 6
            for name in [
                "results.jsonl",
                "effects.jsonl",
                "audit_trace.jsonl",
                "external_effects.jsonl",
                "human_requests.jsonl",
                "llm_calls.jsonl",
                "replay_trace.jsonl",
                "service_state_before_after.json",
                "failure_cases.json",
                "metrics.json",
                "metrics.csv",
            ]:
                assert (Path(temp_dir) / name).exists()
            assert reports["summary"].exists()
            assert reports["live_runtime"].exists()
            assert reports["case_studies"].exists()
            assert reports["failure_taxonomy"].exists()
            rows = {row["runner"]: row for row in metrics["rows"]}
            assert rows["agent_libos"]["forbidden_committed"] == 0
            assert rows["direct_tool_agent"]["forbidden_committed"] > 0

    def test_agent_libos_traces_sensitive_effects_and_denials(self) -> None:
        scenario = next(
            scenario for scenario in load_scenarios(SUITE_ROOT)
            if scenario.id == "research_tool_extension_attack"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            [run] = run_suite([scenario], temp_dir, runners=["agent_libos"])
            assert run.result.forbidden_committed == 0
            assert run.result.trace_coverage == 1.0
            assert run.result.denial_explanation_coverage == 1.0
            assert any(effect.denied and effect.classification == "forbidden" for effect in run.effects)
            assert run.audit

    def test_replay_mode_uses_prior_trace(self) -> None:
        scenarios = load_scenarios(SUITE_ROOT)[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(scenarios, temp_dir, runners=["agent_libos"])
            write_run_outputs(runs, temp_dir)
            replay = Path(temp_dir) / "replay_trace.jsonl"
            replayed = run_suite(scenarios, temp_dir, runners=["agent_libos"], mode="replay", replay_trace=replay)

            assert [run.replay_actions for run in replayed] == [run.replay_actions for run in runs]

    def test_real_mode_requires_explicit_token_spend(self) -> None:
        scenario = load_scenarios(SUITE_ROOT)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="allow_token_spend"):
                run_suite([scenario], temp_dir, runners=["agent_libos"], mode="real")

    def test_runner_names_include_required_system_controls(self) -> None:
        assert {
            "direct_tool_agent",
            "confirmation_agent",
            "sandbox_agent",
            "prompt_defense_agent",
            "agent_libos",
            "agent_libos_live",
            "agent_libos_no_audit",
            "agent_libos_no_fork_attenuation",
            "agent_libos_no_human_approval",
            "agent_libos_no_remote_method_caps",
        } <= set(RUNNER_NAMES)
        categories = {runner: RUNNER_METADATA[runner]["category"] for runner in RUNNER_NAMES}
        assert categories["agent_libos"] == "primary_system"
        assert categories["agent_libos_live"] == "primary_system"
        assert categories["direct_tool_agent"] == "external_baseline"
        assert categories["agent_libos_no_audit"] == "ablation"

    def test_confirmation_baseline_records_explicit_human_decisions(self) -> None:
        scenario = next(
            scenario for scenario in load_scenarios(SUITE_ROOT)
            if scenario.id == "enterprise_state_update_attack"
        )
        scenario = replace(scenario, deterministic_actions=[scenario.deterministic_actions[2]])
        with tempfile.TemporaryDirectory() as temp_dir:
            [run] = run_suite([scenario], temp_dir, runners=["confirmation_agent"])

            assert run.human_requests
            assert {request["decision"] for request in run.human_requests} == {"rejected"}
            assert any(effect.denied and effect.classification == "forbidden" for effect in run.effects)
            assert run.result.forbidden_committed == 0

    def test_no_audit_ablation_preserves_human_gate_but_removes_trace(self) -> None:
        scenario = next(
            scenario for scenario in load_scenarios(SUITE_ROOT)
            if scenario.id == "coding_remote_action_adaptive"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            [run] = run_suite([scenario], temp_dir, runners=["agent_libos_no_audit"])

            assert run.human_requests
            assert run.result.forbidden_committed == 0
            assert run.result.trace_coverage == 0.0
            assert not run.audit

    def test_live_runtime_runner_executes_tools_and_records_audit(self) -> None:
        scenario = next(
            scenario for scenario in load_scenarios(SUITE_ROOT)
            if scenario.id == "coding_core_task_attack"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            [run] = run_suite([scenario], temp_dir, runners=["agent_libos_live"])

            assert run.result.forbidden_committed == 0
            assert run.result.task_success
            assert run.result.state_diff_success
            assert run.result.trace_coverage == 1.0
            assert run.result.denial_explanation_coverage == 1.0
            assert run.result.metadata["db"]
            assert run.audit
            assert any(effect.denied and effect.classification == "forbidden" for effect in run.effects)

    def test_metrics_json_is_round_trippable(self) -> None:
        scenarios = load_scenarios(SUITE_ROOT)[:1]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(scenarios, temp_dir, runners=["agent_libos"])
            write_run_outputs(runs, temp_dir)
            write_metrics(temp_dir)
            collected = collect_metrics(temp_dir)
            disk = json.loads((Path(temp_dir) / "metrics.json").read_text(encoding="utf-8"))

            assert disk["columns"] == collected["columns"]
            assert disk["result_count"] == 1
