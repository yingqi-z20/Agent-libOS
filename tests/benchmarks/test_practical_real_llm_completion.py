from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from benchmarks.practical_agent_workflows.real_completion import (
    PROMPT_VERSION,
    build_completion_goal,
    build_real_completion_scenarios,
    run_completion_suite,
    write_completion_outputs,
)


def _real_llm_env_configured() -> bool:
    if os.getenv("OPENAI_API_KEY") and (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL")):
        return True
    env_path = Path(".env")
    if not env_path.exists():
        return False
    text = env_path.read_text(encoding="utf-8")
    has_key = "OPENAI_API_KEY=" in text
    has_model = "OPENAI_LANGUAGE_MODEL=" in text or "OPENAI_MODEL=" in text
    return has_key and has_model


class TestPracticalRealLlmCompletion:
    def test_catalog_has_eight_realistic_tasks_across_five_tracks(self) -> None:
        scenarios = build_real_completion_scenarios()

        assert len(scenarios) == 8
        assert {scenario.track for scenario in scenarios} == {
            "coding",
            "research",
            "enterprise",
            "devops",
            "self_evolution",
        }
        assert {scenario.id for scenario in scenarios} >= {
            "coding_bugfix_benign",
            "coding_test_output_injection_attack",
            "research_rag_injection_attack",
            "enterprise_vendor_email_attack",
            "devops_incident_summary_benign",
            "self_evolution_capability_laundering_attack",
        }
        assert all(scenario.allowed_effects for scenario in scenarios)
        assert all(scenario.completion_oracle for scenario in scenarios)
        assert all(scenario.planned_actions for scenario in scenarios)

    def test_prompt_builder_marks_boundaries_without_embedding_secret_contents(self) -> None:
        scenario = next(scenario for scenario in build_real_completion_scenarios() if scenario.id == "research_rag_injection_attack")

        prompt = build_completion_goal(scenario)
        decoded = json.loads(prompt)

        assert decoded["prompt_version"] == PROMPT_VERSION
        assert decoded["trusted_sources"] == scenario.trusted_sources
        assert decoded["untrusted_sources"] == scenario.untrusted_sources
        assert "trusted_boundary" in decoded["completion_contract"]
        assert "PRIVATE_RESEARCH_NOTE" not in prompt
        assert "RC_REPO_TOKEN" not in prompt

    def test_deterministic_completion_run_writes_required_artifacts(self) -> None:
        selected_ids = {"coding_bugfix_benign", "enterprise_ticket_update_benign"}
        scenarios = [scenario for scenario in build_real_completion_scenarios() if scenario.id in selected_ids]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_completion_suite(scenarios, temp_dir, mode="deterministic", max_quanta=8)
            reports = write_completion_outputs(runs, temp_dir)
            root = Path(temp_dir)

            assert len(runs) == 2
            for name in [
                "results.jsonl",
                "effects.jsonl",
                "audit_trace.jsonl",
                "external_effects.jsonl",
                "human_requests.jsonl",
                "llm_calls.jsonl",
                "tool_transcript.jsonl",
                "replay_trace.jsonl",
                "service_state_before_after.json",
                "failure_cases.json",
                "summary.json",
            ]:
                assert (root / name).exists()
            assert reports["summary"].exists()
            assert reports["cases"].exists()
            assert reports["failures"].exists()
            assert all(run.result.task_success for run in runs)
            assert all(run.result.forbidden_committed == 0 for run in runs)
            enterprise = next(run for run in runs if run.result.scenario_id == "enterprise_ticket_update_benign")
            assert enterprise.result.state_diff_success
            assert enterprise.service_state_before != enterprise.service_state_after

    def test_attack_completion_distinguishes_requested_denied_committed(self) -> None:
        scenario = next(scenario for scenario in build_real_completion_scenarios() if scenario.id == "research_rag_injection_attack")
        with tempfile.TemporaryDirectory() as temp_dir:
            [run] = run_completion_suite([scenario], temp_dir, mode="deterministic", max_quanta=8)

            forbidden = [effect for effect in run.effects if effect.classification == "forbidden"]
            assert forbidden
            assert any(effect.requested and effect.denied for effect in forbidden)
            assert run.result.forbidden_committed == 0
            assert run.result.task_success
            assert run.result.trace_coverage == 1.0

    def test_replay_reproduces_captured_tool_actions(self) -> None:
        scenario = next(scenario for scenario in build_real_completion_scenarios() if scenario.id == "enterprise_vendor_email_attack")
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_completion_suite([scenario], temp_dir, mode="deterministic", max_quanta=8)
            write_completion_outputs(runs, temp_dir)
            replayed = run_completion_suite(
                [scenario],
                temp_dir,
                mode="replay",
                replay_trace=Path(temp_dir) / "replay_trace.jsonl",
                max_quanta=8,
            )

            assert replayed[0].replay_actions == runs[0].replay_actions
            assert replayed[0].result.forbidden_committed == runs[0].result.forbidden_committed
            assert replayed[0].service_state_after == runs[0].service_state_after

    def test_real_mode_requires_explicit_token_spend(self) -> None:
        scenario = build_real_completion_scenarios()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="allow_token_spend"):
                run_completion_suite([scenario], temp_dir, mode="real")

    @pytest.mark.real_llm
    def test_real_llm_completion_smoke_is_opt_in(self) -> None:
        if os.getenv("AGENT_LIBOS_RUN_REAL_COMPLETION_BENCHMARK") != "1":
            pytest.skip("real completion benchmark smoke is opt-in")
        if not _real_llm_env_configured():
            pytest.skip("real LLM environment is not configured")
        selected_ids = {"coding_bugfix_benign", "research_rag_injection_attack"}
        scenarios = [scenario for scenario in build_real_completion_scenarios() if scenario.id in selected_ids]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_completion_suite(scenarios, temp_dir, mode="real", allow_token_spend=True, max_quanta=8)

            assert len(runs) == 2
            assert all(run.result.forbidden_committed == 0 for run in runs)
            assert all(run.llm_calls for run in runs)
            assert all(isinstance(run.result.metadata.get("process_output"), dict) for run in runs)
