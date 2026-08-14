from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.models import ProcessStatus
from agent_libos.skills import BUILTIN_SKILL_IDS, get_builtin_skill_catalog
from agent_libos.substrate import LocalResourceProviderSubstrate
from benchmarks.builtin_tool_skills import (
    EVALUATION_REPETITIONS,
    EVALUATION_VARIANTS,
    HELD_OUT_SCENARIOS,
    REAL_LLM_ROUTING_CATALOG,
    WITH_SKILLS,
    WITHOUT_SKILLS,
    aggregate_runs,
    evaluation_pair_plan,
    report_all_correct,
    report_publication_ready,
    run_evaluation,
)
from benchmarks.builtin_tool_skills import runner as evaluation_runner
from experiments import run_builtin_tool_skill_evaluation as cli


def test_held_out_catalog_has_adjacent_skill_near_misses() -> None:
    assert EVALUATION_REPETITIONS == 3
    assert len(HELD_OUT_SCENARIOS) == 5
    assert len({scenario.scenario_id for scenario in HELD_OUT_SCENARIOS}) == 5
    assert len({scenario.expected_skill_id for scenario in HELD_OUT_SCENARIOS}) == 5
    assert all(scenario.adjacent_skill_ids for scenario in HELD_OUT_SCENARIOS)
    assert all(
        scenario.expected_skill_id not in scenario.goal
        and scenario.expected_probe_tool not in scenario.goal
        and re.search(r"\bskills?\b", scenario.goal, flags=re.IGNORECASE) is None
        for scenario in HELD_OUT_SCENARIOS
    )


def test_real_llm_routing_catalog_covers_every_builtin_skill() -> None:
    expected = set(BUILTIN_SKILL_IDS)
    actual = {
        scenario.expected_skill_id
        for scenario in REAL_LLM_ROUTING_CATALOG
    }

    assert len(REAL_LLM_ROUTING_CATALOG) == len(BUILTIN_SKILL_IDS) == 26
    assert actual == expected
    assert len(
        {scenario.scenario_id for scenario in REAL_LLM_ROUTING_CATALOG}
    ) == len(REAL_LLM_ROUTING_CATALOG)
    for scenario in REAL_LLM_ROUTING_CATALOG:
        assert scenario.expected_skill_id not in scenario.intent
        assert scenario.adjacent_skill_ids
        assert scenario.expected_skill_id not in scenario.adjacent_skill_ids
        assert set(scenario.adjacent_skill_ids) <= expected


def test_aggregate_runs_records_routing_invalid_calls_and_overhead() -> None:
    runs = [
        {
            "scenario_id": "a",
            "pair_id": "a:1",
            "pair_index": 1,
            "pair_order": list(EVALUATION_VARIANTS),
            "variant": WITH_SKILLS,
            "correct_skill_activation": True,
            "correct_route": True,
            "task_outcome_success": True,
            "completed": True,
            "invalid_tool_calls": 0,
            "llm_calls": 2,
            "provider_attempts": 2,
            "catalog_metadata_bytes": 80,
            "initial_schema_bytes": 100,
            "authorized_schema_bytes": 1_000,
            "cumulative_schema_bytes": 350,
            "initial_schema_token_estimate": 25,
            "authorized_schema_token_estimate": 250,
            "cumulative_schema_token_estimate": 88,
            "prompt_tokens": 200,
            "completion_tokens": 20,
            "cumulative_prompt_bytes": 700,
            "initial_projection_reduction_rate": 0.9,
            "cache_read_tokens": 120,
            "cache_write_tokens": 10,
            "cache_total_calls": 2,
            "cache_reported_calls": 2,
            "cache_metric_input_tokens": 200,
            "uncached_input_tokens": 80,
        },
        {
            "scenario_id": "a",
            "pair_id": "a:1",
            "pair_index": 1,
            "pair_order": list(EVALUATION_VARIANTS),
            "variant": WITHOUT_SKILLS,
            "correct_skill_activation": None,
            "correct_route": False,
            "task_outcome_success": False,
            "completed": False,
            "invalid_tool_calls": 2,
            "llm_calls": 3,
            "provider_attempts": 4,
            "catalog_metadata_bytes": 80,
            "initial_schema_bytes": 100,
            "authorized_schema_bytes": 1_000,
            "cumulative_schema_bytes": 450,
            "initial_schema_token_estimate": 25,
            "authorized_schema_token_estimate": 250,
            "cumulative_schema_token_estimate": 113,
            "prompt_tokens": 300,
            "completion_tokens": 40,
            "cumulative_prompt_bytes": 900,
            "initial_projection_reduction_rate": 0.9,
            "cache_read_tokens": 30,
            "cache_write_tokens": 20,
            "cache_total_calls": 3,
            "cache_reported_calls": 3,
            "cache_metric_input_tokens": 300,
            "uncached_input_tokens": 270,
        },
    ]

    metrics = aggregate_runs(runs)

    assert metrics["runs"] == 2
    assert metrics["completed_runs"] == 1
    assert metrics["successful_task_outcomes"] == 1
    assert metrics["correct_route_rate"] == 0.5
    assert metrics["correct_skill_activation_eligible_runs"] == 1
    assert metrics["correct_skill_activation_rate"] == 1.0
    assert metrics["invalid_tool_calls"] == 2
    assert metrics["mean_llm_calls"] == 2.5
    assert metrics["mean_provider_attempts"] == 3.0
    assert metrics["mean_catalog_metadata_bytes"] == 80.0
    assert metrics["mean_initial_schema_bytes"] == 100.0
    assert metrics["mean_cumulative_schema_bytes"] == 400.0
    assert metrics["mean_initial_schema_token_estimate"] == 25.0
    assert metrics["mean_authorized_schema_token_estimate"] == 250.0
    assert metrics["mean_cumulative_schema_token_estimate"] == 100.5
    assert metrics["mean_prompt_tokens"] == 250.0
    assert metrics["mean_completion_tokens"] == 30.0
    assert metrics["mean_cumulative_prompt_bytes"] == 800.0
    assert metrics["cache_read_tokens"] == 150
    assert metrics["cache_write_tokens"] == 30
    assert metrics["cache_reported_calls"] == 5
    assert metrics["cache_metric_input_tokens"] == 500
    assert metrics["uncached_input_tokens"] == 350
    assert metrics["cache_hit_rate"] == 0.3
    assert metrics["by_variant"][WITH_SKILLS]["cache_hit_rate"] == 0.6
    assert metrics["by_scenario"]["a"]["correct_route_rate"] == 0.5
    assert metrics["by_variant"][WITH_SKILLS]["task_outcome_success_rate"] == 1.0
    assert metrics["paired"]["complete_pairs"] == 1
    assert (
        metrics["paired"]["samples"][0]["with_skills_minus_without_skills"][
            "provider_attempts"
        ]
        == -2
    )
    assert (
        metrics["comparison"]["with_skills_minus_without_skills"][
            "mean_initial_schema_bytes"
        ]
        == 0.0
    )


def test_report_runs_each_selected_scenario_exactly_three_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, int, str, str, int, int, tuple[str, str], Path]] = []

    def fake_run_once(
        scenario: object,
        repetition: int,
        workspace: Path,
        *,
        variant: str,
        pair_id: str,
        pair_index: int,
        pair_position: int,
        pair_order: tuple[str, str],
    ) -> dict[str, object]:
        scenario_id = str(getattr(scenario, "scenario_id"))
        observed.append(
            (
                scenario_id,
                repetition,
                variant,
                pair_id,
                pair_index,
                pair_position,
                pair_order,
                workspace,
            )
        )
        return {
            "scenario_id": scenario_id,
            "repetition": repetition,
            "variant": variant,
            "pair_id": pair_id,
            "pair_index": pair_index,
            "pair_position": pair_position,
            "pair_order": list(pair_order),
            "correct_skill_activation": True,
            "correct_route": True,
            "task_outcome_success": True,
            "completed": True,
            "invalid_tool_calls": 0,
        }

    monkeypatch.setattr(evaluation_runner, "_run_once", fake_run_once)
    selected = HELD_OUT_SCENARIOS[0].scenario_id

    report = run_evaluation(tmp_path / "evaluation", scenario_ids=[selected])

    assert [item[1] for item in observed] == [1, 1, 2, 2, 3, 3]
    assert [item[2] for item in observed] == [
        WITH_SKILLS,
        WITHOUT_SKILLS,
        WITHOUT_SKILLS,
        WITH_SKILLS,
        WITH_SKILLS,
        WITHOUT_SKILLS,
    ]
    assert [item[5] for item in observed] == [1, 2, 1, 2, 1, 2]
    assert all(item[0] == selected and item[7].is_dir() for item in observed)
    assert len({item[3] for item in observed}) == 3
    assert set(report) == {
        "schema_version",
        "evaluation",
        "generated_at",
        "evaluation_provenance",
        "image_ids",
        "variants",
        "order_design",
        "repetitions_per_scenario",
        "pairs_per_scenario",
        "scenario_count",
        "scenario_catalog",
        "runs",
        "metrics",
    }
    assert report["schema_version"] == 3
    assert report["metrics"]["runs"] == 6


def test_cli_lists_scenarios_without_real_llm_confirmation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["--list-scenarios"])

    assert capsys.readouterr().out.splitlines() == [
        scenario.scenario_id for scenario in HELD_OUT_SCENARIOS
    ]


def test_cli_requires_explicit_real_llm_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    with pytest.raises(SystemExit, match="2"):
        cli.main(["--output", str(tmp_path / "report.json")])


def test_cli_dry_run_reports_fixed_repetitions_without_credentials(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selected = HELD_OUT_SCENARIOS[-1].scenario_id

    cli.main(["--dry-run", "--scenario", selected])

    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "evaluation": "builtin_tool_skill_routing",
        "real_llm_calls": False,
        "repetitions_per_scenario": 3,
        "variants": list(EVALUATION_VARIANTS),
        "scenarios": [selected],
        "pair_plan": evaluation_pair_plan([selected]),
        "planned_pairs": 3,
        "planned_runs": 6,
    }


def test_cli_require_all_correct_uses_task_oracle_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    treatment = {
        "pair_id": "a:1",
        "variant": WITH_SKILLS,
        "correct_skill_activation": True,
        "correct_route": True,
        "probe_tool_result_success": True,
        "task_outcome_success": False,
        "completed": True,
    }
    baseline = {
        **treatment,
        "variant": WITHOUT_SKILLS,
        "correct_skill_activation": None,
        "task_outcome_success": True,
    }
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *_args, **_kwargs: {"runs": [treatment, baseline], "metrics": {}},
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "--confirm-real-llm",
                "--require-all-correct",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_cli_require_publication_gate_uses_schema_v3_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *_args, **_kwargs: {"schema_version": 2, "runs": [], "metrics": {}},
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "--confirm-real-llm",
                "--require-publication-gate",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_cli_publication_gate_rejects_partial_matrix_before_paid_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *_args, **_kwargs: pytest.fail("partial matrix must not run"),
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "--confirm-real-llm",
                "--require-publication-gate",
                "--scenario",
                HELD_OUT_SCENARIOS[0].scenario_id,
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_paired_images_compare_skills_projection_with_no_skills_baseline(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        treatment_image = evaluation_runner._IMAGE_IDS[WITH_SKILLS]
        baseline_image = evaluation_runner._IMAGE_IDS[WITHOUT_SKILLS]
        evaluation_runner._register_evaluation_image(runtime, WITH_SKILLS)
        evaluation_runner._register_evaluation_image(runtime, WITHOUT_SKILLS)
        treatment_pid = runtime.process.spawn(image=treatment_image, goal="probe")
        baseline_pid = runtime.process.spawn(image=baseline_image, goal="probe")

        treatment_tools = {
            schema["function"]["name"]
            for schema in runtime.tools.openai_tool_schemas(treatment_pid)
        }
        baseline_tools = {
            schema["function"]["name"]
            for schema in runtime.tools.openai_tool_schemas(baseline_pid)
        }

        assert "activate_skill" in treatment_tools
        assert "write_text_file" not in treatment_tools
        assert "activate_skill" not in baseline_tools
        assert "write_text_file" in baseline_tools
        assert runtime.process.get(treatment_pid).loaded_skills == {}
        assert runtime.process.get(baseline_pid).loaded_skills == {}
        assert len(baseline_tools) > len(treatment_tools)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("scenario_id", "probe_args"),
    [
        (
            "ordinary_workspace_write",
            {"path": "routing-eval.txt", "content": "routed\n"},
        ),
        ("read_only_git_state", {}),
        (
            "approved_git_status_command",
            {"argv": ["git", "status", "--short"]},
        ),
        ("capture_process_checkpoint", {"reason": "evaluation oracle"}),
        ("cached_mcp_registry", {}),
    ],
)
def test_scenario_probes_and_oracles_run_offline_without_llm(
    tmp_path: Path,
    scenario_id: str,
    probe_args: dict[str, object],
) -> None:
    scenario = next(
        item for item in HELD_OUT_SCENARIOS if item.scenario_id == scenario_id
    )
    workspace = tmp_path / scenario_id
    workspace.mkdir()
    evaluation_runner._prepare_workspace(scenario, workspace)
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        evaluation_runner._register_evaluation_image(runtime, WITH_SKILLS)
        pid = runtime.process.spawn(
            image=evaluation_runner._IMAGE_IDS[WITH_SKILLS],
            goal="deterministic oracle probe",
        )
        evaluation_runner._grant_scenario_authority(runtime, pid, scenario)
        runtime.skills.activate_skill(pid, scenario.expected_skill_id, actor=pid)
        result = runtime.tools.call(pid, scenario.expected_probe_tool, probe_args)
        outcome = evaluation_runner._evaluate_task_outcome(
            runtime,
            pid=pid,
            scenario=scenario,
            workspace=workspace,
            probe={
                "action": {
                    "action": scenario.expected_probe_tool,
                    **probe_args,
                },
                "result": {"ok": result.ok, "payload": result.payload},
            },
        )

        assert result.ok, result.error
        assert outcome["passed"] is True, outcome
        assert runtime.store.list_llm_calls(pid=pid, limit=1) == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("setup_kind", "payload"),
    [
        (
            "git_read",
            {
                "entries": [
                    {
                        "path": {
                            "display": "tracked-intent.txt",
                            "path_b64": "dHJhY2tlZC1pbnRlbnQudHh0",
                            "lossy": False,
                        },
                        "kind": "untracked",
                        "index_status": "?",
                        "worktree_status": "?",
                    }
                ],
                "state": {"token": "a" * 64},
                "truncated": False,
            },
        ),
        (
            "shell_git_read",
            {
                "argv": ["git", "status", "--short"],
                "returncode": 0,
                "stdout": "?? tracked-intent.txt\n",
            },
        ),
        ("mcp_registry_read", {"servers": [], "has_more": False}),
    ],
)
def test_read_only_task_oracles_validate_structured_results_without_network(
    tmp_path: Path,
    setup_kind: str,
    payload: dict[str, object],
) -> None:
    scenario = next(
        item for item in HELD_OUT_SCENARIOS if item.setup_kind == setup_kind
    )
    outcome = evaluation_runner._evaluate_task_outcome(
        object(),  # type: ignore[arg-type]
        pid="pid-test",
        scenario=scenario,
        workspace=tmp_path,
        probe={"action": {}, "result": {"ok": True, "payload": payload}},
    )

    assert outcome["passed"] is True


def test_exit_review_trace_is_bounded_and_omits_raw_completion_content() -> None:
    long_error = "e" * 300
    long_tool = "t" * 300
    observations = [
        {
            "action": {
                "action": "process_exit",
                "review_token": "secret-review-token",
                "completion_evidence": "{\"goal_oid\":\"sensitive\"}",
                "payload": {"private": "result"},
            },
            "result": {
                "ok": True,
                "payload": {
                    "status": "completion_review_required",
                    "completion_review": {
                        "review_token": "secret-review-token",
                        "validation_errors": [
                            "missing required tool",
                            long_error,
                            *[f"extra-{index}" for index in range(20)],
                        ],
                        "explicit_unobserved_tool_hints": [
                            {"tool": "git_status", "reason": "sensitive goal"},
                            {"tool": long_tool},
                            *[{"tool": f"tool_{index}"} for index in range(20)],
                        ],
                        "goal": {"fallback": "sensitive goal"},
                    },
                },
            },
        }
    ] * 20

    trace = evaluation_runner._exit_review_trace(observations)

    assert len(trace) == 16
    assert trace[0]["ok"] is True
    assert trace[0]["result_status"] == "completion_review_required"
    assert trace[0]["has_review_token"] is True
    assert trace[0]["has_completion_evidence"] is True
    assert len(trace[0]["validation_errors"]) == 16
    assert len(trace[0]["explicit_unobserved_tools"]) == 16
    assert trace[0]["validation_errors"][0] == "missing required tool"
    assert trace[0]["validation_errors"][1] == "e" * 253 + "..."
    assert trace[0]["explicit_unobserved_tools"][0] == "git_status"
    assert trace[0]["explicit_unobserved_tools"][1] == "t" * 253 + "..."
    assert max(len(value) for value in trace[0]["validation_errors"]) == 256
    assert max(len(value) for value in trace[0]["explicit_unobserved_tools"]) == 256
    assert "secret-review-token" not in json.dumps(trace)
    assert "sensitive" not in json.dumps(trace)


@pytest.mark.parametrize(
    "mutation",
    [
        {"entries": []},
        {"entries": [{"path": {"display": "unrelated.txt"}}]},
        {"truncated": True},
        {"state": {"token": "short"}},
        {"state": {"token": "g" * 64}},
    ],
)
def test_git_fixture_oracle_fails_closed_on_inexact_evidence(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "entries": [
            {
                "path": {
                    "display": "tracked-intent.txt",
                    "path_b64": "dHJhY2tlZC1pbnRlbnQudHh0",
                    "lossy": False,
                },
                "kind": "untracked",
                "index_status": "?",
                "worktree_status": "?",
            }
        ],
        "state": {"token": "a" * 64},
        "truncated": False,
    }
    payload.update(mutation)

    assert evaluation_runner._git_read_outcome(payload)["passed"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"servers": []},
        {"servers": [], "has_more": True},
        {"servers": [{}], "has_more": False},
        {"servers": [], "has_more": False, "refreshed": False},
    ],
)
def test_empty_mcp_registry_oracle_requires_exact_complete_evidence(
    payload: dict[str, object],
) -> None:
    assert evaluation_runner._mcp_registry_outcome(payload)["passed"] is False


def test_workspace_oracle_requires_exact_persisted_content(tmp_path: Path) -> None:
    scenario = next(
        item for item in HELD_OUT_SCENARIOS if item.setup_kind == "workspace_write"
    )
    probe = {
        "action": {},
        "result": {
            "ok": True,
            "payload": {
                "path": "routing-eval.txt",
                "bytes_written": 7,
                "created": True,
            },
        },
    }
    target = tmp_path / "routing-eval.txt"
    target.write_bytes(b"wrong\n")
    failed = evaluation_runner._evaluate_task_outcome(
        object(),  # type: ignore[arg-type]
        pid="pid-test",
        scenario=scenario,
        workspace=tmp_path,
        probe=probe,
    )
    target.write_bytes(b"routed\n")
    passed = evaluation_runner._evaluate_task_outcome(
        object(),  # type: ignore[arg-type]
        pid="pid-test",
        scenario=scenario,
        workspace=tmp_path,
        probe=probe,
    )

    assert failed["passed"] is False
    assert passed["passed"] is True


def test_checkpoint_oracle_reads_back_the_created_snapshot(tmp_path: Path) -> None:
    scenario = next(
        item for item in HELD_OUT_SCENARIOS if item.setup_kind == "checkpoint"
    )

    class Checkpoints:
        def inspect(self, checkpoint_id: str, **_: object) -> dict[str, object]:
            assert checkpoint_id == "checkpoint-test"
            return {"checkpoint": {"pid": "pid-test"}}

    outcome = evaluation_runner._evaluate_task_outcome(
        SimpleNamespace(checkpoint=Checkpoints()),  # type: ignore[arg-type]
        pid="pid-test",
        scenario=scenario,
        workspace=tmp_path,
        probe={
            "action": {},
            "result": {
                "ok": True,
                "payload": {"checkpoint_id": "checkpoint-test"},
            },
        },
    )

    assert outcome["passed"] is True


def test_require_all_correct_rejects_dispatch_and_exit_without_task_outcome() -> None:
    treatment = {
        "pair_id": "a:1",
        "variant": WITH_SKILLS,
        "correct_skill_activation": True,
        "correct_route": True,
        "probe_tool_result_success": True,
        "task_outcome_success": False,
        "completed": True,
    }
    baseline = {
        **treatment,
        "variant": WITHOUT_SKILLS,
        "correct_skill_activation": None,
        "task_outcome_success": True,
    }

    assert report_all_correct({"runs": [treatment, baseline]}) is False
    assert report_all_correct(
        {"runs": [{**treatment, "task_outcome_success": True}, baseline]}
    ) is True


def test_pair_plan_counterbalances_all_fifteen_pairs_eight_to_seven() -> None:
    plan = evaluation_pair_plan()

    assert len(plan) == 15
    assert [pair["pair_index"] for pair in plan] == list(range(1, 16))
    assert sum(pair["pair_order"][0] == WITH_SKILLS for pair in plan) == 8
    assert sum(pair["pair_order"][0] == WITHOUT_SKILLS for pair in plan) == 7
    assert [pair["pair_order"] for pair in plan] == [
        list(EVALUATION_VARIANTS) if index % 2 else list(reversed(EVALUATION_VARIANTS))
        for index in range(1, 16)
    ]


def test_schema_v2_report_remains_correctness_readable_but_not_publishable() -> None:
    treatment = {
        "pair_id": "legacy:1",
        "variant": WITH_SKILLS,
        "correct_skill_activation": True,
        "correct_route": True,
        "probe_tool_result_success": True,
        "task_outcome_success": True,
        "completed": True,
    }
    baseline = {
        **treatment,
        "variant": WITHOUT_SKILLS,
        "correct_skill_activation": None,
    }
    legacy = {"schema_version": 2, "runs": [treatment, baseline]}

    assert report_all_correct(legacy) is True
    assert report_publication_ready(legacy) is False


@pytest.mark.parametrize(
    "mutation",
    (
        "schema_v2",
        "dirty_source",
        "missing_credential",
        "missing_run",
        "run_order",
        "pair_balance",
        "provider_attempts",
        "oracle_evidence",
        "undeclared_oracle_shape",
        "actual_model_mismatch",
        "uniform_actual_model_mismatch",
        "nonterminal_status",
        "missing_cache_tokens",
        "token_reported_calls",
        "schema_token_mismatch",
        "metrics_mismatch",
    ),
)
def test_publication_gate_rejects_incomplete_or_unbound_reports(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _complete_publication_report()
    monkeypatch.setattr(
        evaluation_runner,
        "valid_evaluation_provenance",
        lambda value: isinstance(value, dict) and "identity" in value,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "evaluation_provenance_identity",
        lambda value: value.get("identity") if isinstance(value, dict) else None,
    )
    if mutation == "schema_v2":
        report["schema_version"] = 2
    elif mutation == "dirty_source":
        report["evaluation_provenance"]["identity"]["source"]["dirty"] = True
    elif mutation == "missing_credential":
        report["evaluation_provenance"]["identity"]["llm"][
            "credential_present"
        ] = False
    elif mutation == "missing_run":
        report["runs"].pop()
    elif mutation == "run_order":
        report["runs"][0], report["runs"][1] = report["runs"][1], report["runs"][0]
    elif mutation == "pair_balance":
        report["order_design"]["treatment_first_pairs"] = 7
    elif mutation == "provider_attempts":
        report["runs"][0]["provider_attempts"] = 0
    elif mutation == "oracle_evidence":
        report["runs"][0]["task_outcome_oracle"]["checks"][
            "exact_file_content"
        ] = False
    elif mutation == "undeclared_oracle_shape":
        report["runs"][0]["task_outcome_oracle"] = {
            "passed": True,
            "checks": {"verified": True},
        }
    elif mutation == "actual_model_mismatch":
        report["runs"][0]["models"] = ["other-model"]
    elif mutation == "uniform_actual_model_mismatch":
        for run in report["runs"]:
            run["models"] = ["other-model"]
    elif mutation == "nonterminal_status":
        report["runs"][0]["status"] = "runnable"
    elif mutation == "missing_cache_tokens":
        report["runs"][0]["cache_write_tokens"] = None
    elif mutation == "token_reported_calls":
        report["runs"][0]["prompt_token_reported_calls"] = 1
    elif mutation == "schema_token_mismatch":
        report["runs"][0]["initial_schema_token_estimate"] += 1
    elif mutation == "metrics_mismatch":
        report["metrics"]["completed_runs"] = 29

    assert report_publication_ready(report) is False


@pytest.mark.parametrize(
    "field",
    (
        *evaluation_runner._PAIRED_NUMERIC_FIELDS,
        *evaluation_runner._PUBLICATION_EVIDENCE_COUNT_FIELDS,
    ),
)
def test_publication_gate_requires_each_paired_numeric_field(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _complete_publication_report()
    monkeypatch.setattr(
        evaluation_runner,
        "valid_evaluation_provenance",
        lambda value: isinstance(value, dict) and "identity" in value,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "evaluation_provenance_identity",
        lambda value: value.get("identity") if isinstance(value, dict) else None,
    )
    report["runs"][0].pop(field)

    assert report_publication_ready(report) is False


def test_complete_schema_v3_report_passes_publication_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = json.loads(json.dumps(_complete_publication_report()))
    monkeypatch.setattr(
        evaluation_runner,
        "valid_evaluation_provenance",
        lambda value: isinstance(value, dict) and "identity" in value,
    )


@pytest.mark.parametrize("terminal_status", ("exited", "failed", "killed"))
def test_complete_negative_observation_is_publishable_but_not_all_correct(
    terminal_status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _complete_publication_report()
    monkeypatch.setattr(
        evaluation_runner,
        "valid_evaluation_provenance",
        lambda value: isinstance(value, dict) and "identity" in value,
    )
    monkeypatch.setattr(
        evaluation_runner,
        "evaluation_provenance_identity",
        lambda value: value.get("identity") if isinstance(value, dict) else None,
    )
    run = report["runs"][0]
    run["status"] = terminal_status
    run["completed"] = False
    run["task_outcome_success"] = False
    run["task_outcome_oracle"] = {
        "passed": False,
        "checks": {
            "successful_probe_result": True,
            "exact_file_content": False,
            "result_path_matches": True,
        },
    }
    run["correct_skill_activation"] = False
    run["correct_route"] = False
    report["metrics"] = aggregate_runs(report["runs"])

    assert report_publication_ready(report) is True
    assert report_all_correct(report) is False
    monkeypatch.setattr(
        evaluation_runner,
        "evaluation_provenance_identity",
        lambda value: value.get("identity") if isinstance(value, dict) else None,
    )

    assert report_publication_ready(report) is True
    paired = report["metrics"]["paired"]
    assert paired["observed_pairs"] == paired["complete_pairs"] == 15
    assert len(paired["samples"]) == 15
    assert all(
        set(sample["arms"]) == set(EVALUATION_VARIANTS)
        and "provider_attempts" in sample["with_skills_minus_without_skills"]
        for sample in paired["samples"]
    )


def _complete_publication_report() -> dict[str, object]:
    scenarios = {scenario.scenario_id: scenario for scenario in HELD_OUT_SCENARIOS}
    runs: list[dict[str, object]] = []
    for pair in evaluation_pair_plan():
        scenario = scenarios[str(pair["scenario_id"])]
        pair_order = list(pair["pair_order"])
        for pair_position, variant in enumerate(pair_order, start=1):
            runs.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "repetition": pair["repetition"],
                    "pair_id": pair["pair_id"],
                    "pair_index": pair["pair_index"],
                    "pair_position": pair_position,
                    "pair_order": pair_order,
                    "variant": variant,
                    "goal_sha256": hashlib.sha256(
                        scenario.goal.encode("utf-8")
                    ).hexdigest(),
                    "status": "exited",
                    "completed": True,
                    "task_outcome_success": True,
                    "task_outcome_oracle": _complete_oracle(scenario.setup_kind),
                    "probe_tool_result_success": True,
                    "correct_skill_activation": (
                        True if variant == WITH_SKILLS else None
                    ),
                    "correct_route": True,
                    "invalid_tool_calls": 0,
                    "llm_calls": 2,
                    "provider_attempts": 2,
                    "provider_attempt_reported_calls": 2,
                    "prompt_token_reported_calls": 2,
                    "completion_token_reported_calls": 2,
                    "models": ["publication-test-model"],
                    "catalog_metadata_bytes": 0,
                    "initial_schema_bytes": 100,
                    "authorized_schema_bytes": 1_000,
                    "cumulative_schema_bytes": 250,
                    "initial_schema_token_estimate": 25,
                    "authorized_schema_token_estimate": 250,
                    "cumulative_schema_token_estimate": 63,
                    "cumulative_prompt_bytes": 2_000,
                    "prompt_tokens": 400,
                    "completion_tokens": 40,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_total_calls": 2,
                    "cache_reported_calls": 2,
                    "cache_read_reported_calls": 2,
                    "cache_write_reported_calls": 2,
                    "cache_metric_reported_calls": 2,
                    "cache_metric_input_tokens": 400,
                    "uncached_input_tokens": 400,
                    "initial_projection_reduction_rate": 0.9,
                }
            )
    identity = {
        "schema_version": 1,
        "source": {
            "available": True,
            "commit": "a" * 40,
            "dirty": False,
            "working_tree_sha256": "b" * 64,
        },
        "llm": {
            "available": True,
            "credential_present": True,
            "model": "publication-test-model",
            "config_sha256": "c" * 64,
        },
    }
    report: dict[str, object] = {
        "schema_version": 3,
        "evaluation": "builtin_tool_skill_routing",
        "evaluation_provenance": {"identity": identity},
        "variants": list(EVALUATION_VARIANTS),
        "order_design": evaluation_runner._order_design(HELD_OUT_SCENARIOS),
        "repetitions_per_scenario": EVALUATION_REPETITIONS,
        "pairs_per_scenario": EVALUATION_REPETITIONS,
        "scenario_count": len(HELD_OUT_SCENARIOS),
        "scenario_catalog": [
            evaluation_runner._scenario_record(scenario)
            for scenario in HELD_OUT_SCENARIOS
        ],
        "runs": runs,
    }
    report["metrics"] = aggregate_runs(runs)
    return report


def _complete_oracle(setup_kind: str) -> dict[str, object]:
    checks = {
        "workspace_write": {
            "successful_probe_result": True,
            "exact_file_content": True,
            "result_path_matches": True,
        },
        "git_read": {
            "successful_probe_result": True,
            "exact_fixture_entry": True,
            "complete_status_result": True,
            "valid_state_token": True,
        },
        "shell_git_read": {
            "successful_probe_result": True,
            "exact_argv": True,
            "zero_returncode": True,
            "reported_dirty_fixture": True,
        },
        "checkpoint": {
            "successful_probe_result": True,
            "checkpoint_creation_acknowledged": True,
            "host_resolved_checkpoint": True,
            "durable_checkpoint_matches_process": True,
        },
        "mcp_registry_read": {
            "successful_probe_result": True,
            "exact_empty_registry": True,
        },
    }[setup_kind]
    return {"passed": True, "checks": checks}


@pytest.mark.real_llm
@pytest.mark.timeout(1800)
def test_real_llm_routes_every_builtin_skill_and_rejects_adjacent_near_misses(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        "local",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        catalog = get_builtin_skill_catalog()
        template = runtime.get_image("coding-agent:v0")
        metadata = dict(template.metadata)
        metadata.pop("completion_gate", None)
        metadata["tool_projection"] = "skills"
        image = replace(
            template,
            image_id="complete-builtin-skill-routing-evaluator:v0",
            name="complete-builtin-skill-routing-evaluator",
            default_tools=[
                tool_name
                for package in catalog.list()
                for tool_name in package.allowed_tools
            ],
            default_skills=[],
            metadata=metadata,
        )
        runtime.register_image(image, actor="builtin-tool-skill-evaluation")

        failures: list[dict[str, object]] = []
        for scenario in REAL_LLM_ROUTING_CATALOG:
            pid = runtime.process.spawn(
                image=image.image_id,
                goal=(
                    "Machine-only Skill routing audit. Intent: "
                    f"{scenario.intent}. Discover the smallest applicable built-in "
                    "Skill, activate exactly that one, then immediately call "
                    "process_exit with a structured payload naming the selected "
                    "Skill. Do not call a domain tool and do not ask a Human."
                ),
            )
            results = runtime.run_process_until_idle(pid, max_quanta=6)
            actions = evaluation_runner._action_sequence(results)
            activated = [
                str(action.get("skill_id") or "")
                for action in actions
                if action.get("action") == "activate_skill"
            ]
            process = runtime.process.get(pid)
            passed = (
                process.status == ProcessStatus.EXITED
                and activated == [scenario.expected_skill_id]
                and set(activated).isdisjoint(scenario.adjacent_skill_ids)
            )
            if not passed:
                failures.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "expected_skill_id": scenario.expected_skill_id,
                        "adjacent_skill_ids": list(scenario.adjacent_skill_ids),
                        "activated_skill_ids": activated,
                        "status": process.status.value,
                        "actions": [
                            str(action.get("action") or "")
                            for action in actions
                        ],
                    }
                )

        assert failures == []
    finally:
        runtime.close()


@pytest.mark.real_llm
@pytest.mark.timeout(1800)
def test_real_llm_builtin_tool_skill_routing_is_three_pair_evaluation(
    tmp_path: Path,
) -> None:
    report = run_evaluation(tmp_path / "evaluation")

    assert report["repetitions_per_scenario"] == 3
    assert report["scenario_count"] == len(HELD_OUT_SCENARIOS)
    assert report["metrics"]["runs"] == len(HELD_OUT_SCENARIOS) * 3 * 2
    assert all(
        scenario["runs"] == 6
        for scenario in report["metrics"]["by_scenario"].values()
    )
    assert all(
        arm["runs"] == len(HELD_OUT_SCENARIOS) * 3
        for arm in report["metrics"]["by_variant"].values()
    )
    assert all(run["llm_calls"] >= 1 for run in report["runs"])
    assert all(
        run["authorized_schema_bytes"] > run["initial_schema_bytes"]
        for run in report["runs"]
        if run["variant"] == WITH_SKILLS
    )
    assert all(
        run["authorized_schema_bytes"] == run["initial_schema_bytes"]
        for run in report["runs"]
        if run["variant"] == WITHOUT_SKILLS
    )
    assert 0.0 <= report["metrics"]["correct_skill_activation_rate"] <= 1.0
    assert all(
        isinstance(run["task_outcome_oracle"]["passed"], bool)
        for run in report["runs"]
    )
    failed_runs = [
        {
            "scenario_id": run["scenario_id"],
            "repetition": run["repetition"],
            "variant": run["variant"],
            "correct_skill_activation": run["correct_skill_activation"],
            "correct_route": run["correct_route"],
            "task_outcome_success": run["task_outcome_success"],
            "completed": run["completed"],
        }
        for run in report["runs"]
        if not (
            run["correct_route"]
            and run["task_outcome_success"]
            and run["completed"]
            and (
                run["variant"] == WITHOUT_SKILLS
                or run["correct_skill_activation"]
            )
        )
    ]
    assert report_all_correct(report), failed_runs
