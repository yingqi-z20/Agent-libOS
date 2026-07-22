from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmarks.practical_agent_workflows import (
    EvidenceLevel,
    PracticalScenario,
    SemanticEffect,
    build_modeled_scenarios,
    default_scenarios,
    run_practical_evaluation,
)
from benchmarks.practical_agent_workflows.models import PracticalRunReport
from experiments import run_practical_evaluation as practical_cli


REPORT_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "practical_agent_workflows"
    / "report.schema.json"
)


def test_native_live_workflows_have_no_modeled_fallback_and_resolve_operations(tmp_path) -> None:
    report = run_practical_evaluation(default_scenarios(), work_dir=tmp_path)

    assert report.native_live_ok
    assert report.modeled_fallback == 0
    assert report.scenario_counts == {"native-live": 3, "modeled": 80}
    assert report.modeled_suite_ok
    native = [item for item in report.results if item.evidence_level == EvidenceLevel.NATIVE_LIVE]
    assert sum(item.semantic_effects for item in native) == 3
    assert sum(item.tool_calls for item in native) == 3
    assert all(item.external_effect_ids and item.operation_ids for item in native)


def test_practical_report_matches_published_json_schema(tmp_path) -> None:
    report = run_practical_evaluation(default_scenarios(), work_dir=tmp_path).to_dict()
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    ("native_live_ok", "modeled_suite_ok", "modeled_fallback", "exit_code"),
    [
        (True, True, 0, None),
        (False, True, 0, 1),
        (True, False, 0, 1),
        (True, True, 1, 1),
    ],
)
def test_practical_cli_exit_contract_writes_completed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_live_ok: bool,
    modeled_suite_ok: bool,
    modeled_fallback: int,
    exit_code: int | None,
) -> None:
    report = PracticalRunReport(
        schema_version=1,
        results=[],
        scenario_counts={"native-live": 0, "modeled": 0},
        semantic_effect_counts={"native-live": 0, "modeled": 0},
        native_tool_calls=0,
        native_operations=0,
        modeled_fallback=modeled_fallback,
        native_live_ok=native_live_ok,
        modeled_suite_ok=modeled_suite_ok,
    )
    monkeypatch.setattr(practical_cli, "run_practical_evaluation", lambda: report)
    output = tmp_path / "report.json"

    if exit_code is None:
        practical_cli.main(["--output", str(output)])
    else:
        with pytest.raises(SystemExit) as exc_info:
            practical_cli.main(["--output", str(output)])
        assert exc_info.value.code == exit_code

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def test_eva_scenario_matrix_is_migrated_as_design_only_modeled_evidence() -> None:
    scenarios = build_modeled_scenarios()

    assert len(scenarios) == 80
    assert {item.modeled_claim["track"] for item in scenarios} == {
        "coding",
        "research",
        "enterprise",
        "devops",
        "self_evolution",
    }
    assert len({item.modeled_claim["task_family"] for item in scenarios}) == 8
    assert all(item.evidence_level == EvidenceLevel.MODELED for item in scenarios)
    assert all(not item.native_actions for item in scenarios)


def test_modeled_oracle_failure_does_not_become_runtime_evidence() -> None:
    invalid = PracticalScenario(
        scenario_id="invalid-modeled",
        title="invalid modeled claim",
        evidence_level=EvidenceLevel.MODELED,
        effects=(SemanticEffect("filesystem.read", "public.txt", "modeled"),),
        modeled_claim={"variant": "benign"},
    )

    report = run_practical_evaluation([invalid])

    assert not report.modeled_suite_ok
    assert not report.results[0].ok
    assert report.native_tool_calls == 0
    assert report.native_operations == 0


def test_native_live_scenario_cannot_smuggle_a_modeled_effect() -> None:
    with pytest.raises(ValueError, match="requires runtime tool actions"):
        PracticalScenario(
            scenario_id="invalid-native",
            title="invalid",
            evidence_level=EvidenceLevel.NATIVE_LIVE,
            effects=(SemanticEffect("mail.send", "message"),),
        )


def test_native_live_rejects_semantic_effect_not_bound_to_provider_receipt(tmp_path) -> None:
    scenario = next(
        item for item in default_scenarios()
        if item.evidence_level == EvidenceLevel.NATIVE_LIVE
    )
    mislabeled = replace(
        scenario,
        effects=(SemanticEffect("connector.calendar.delete", "wrong-target"),),
    )

    report = run_practical_evaluation([mislabeled], work_dir=tmp_path)

    assert not report.native_live_ok
    assert not report.results[0].ok
    assert any("semantic effect mismatch" in error for error in report.results[0].errors)
