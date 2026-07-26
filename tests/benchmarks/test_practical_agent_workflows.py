from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from benchmarks.practical_agent_workflows import (
    EvidenceLevel,
    PracticalScenario,
    SemanticEffect,
    build_modeled_scenarios,
    default_scenarios,
    run_practical_evaluation,
    validate_practical_report,
    validate_practical_report_schema,
)
from benchmarks.practical_agent_workflows.models import (
    PracticalRunReport,
    PracticalScenarioResult,
)
from benchmarks.practical_agent_workflows.oracle import validate_modeled_scenario
from benchmarks.practical_agent_workflows import runner as practical_runner
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


def test_practical_cli_exit_contract_writes_completed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = PracticalRunReport(
        schema_version=1,
        results=[],
        scenario_counts={"native-live": 0, "modeled": 0},
        semantic_effect_counts={"native-live": 0, "modeled": 0},
        native_tool_calls=0,
        native_operations=0,
        modeled_fallback=0,
        native_live_ok=True,
        modeled_suite_ok=True,
    )
    monkeypatch.setattr(practical_cli, "run_practical_evaluation", lambda: report)
    output = tmp_path / "report.json"

    practical_cli.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def test_practical_cli_emits_schema_valid_failed_gate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = PracticalScenarioResult(
        scenario_id="modeled-failure",
        evidence_level=EvidenceLevel.MODELED,
        ok=False,
        semantic_effects=1,
        tool_calls=0,
        operations=0,
        errors=["oracle mismatch"],
    )
    report = PracticalRunReport(
        schema_version=1,
        results=[failed],
        scenario_counts={"native-live": 0, "modeled": 1},
        semantic_effect_counts={"native-live": 0, "modeled": 1},
        native_tool_calls=0,
        native_operations=0,
        modeled_fallback=0,
        native_live_ok=True,
        modeled_suite_ok=False,
    )
    monkeypatch.setattr(practical_cli, "run_practical_evaluation", lambda: report)
    output = tmp_path / "failed-report.json"

    with pytest.raises(SystemExit) as exc_info:
        practical_cli.main(["--output", str(output)])
    assert exc_info.value.code == 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    assert validate_practical_report(payload) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scenario_counts", {"native-live": 1, "modeled": 0}, "scenario_counts"),
        ("native_tool_calls", 1, "native_tool_calls"),
        ("modeled_fallback", 1, "modeled_fallback"),
        ("native_live_ok", False, "report.schema.json"),
    ],
)
def test_practical_cli_refuses_internally_inconsistent_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    report = PracticalRunReport(
        schema_version=1,
        results=[],
        scenario_counts={"native-live": 0, "modeled": 0},
        semantic_effect_counts={"native-live": 0, "modeled": 0},
        native_tool_calls=0,
        native_operations=0,
        modeled_fallback=0,
        native_live_ok=True,
        modeled_suite_ok=True,
    )
    setattr(report, field, value)
    monkeypatch.setattr(practical_cli, "run_practical_evaluation", lambda: report)
    output = tmp_path / "invalid-report.json"

    with pytest.raises(RuntimeError, match=message):
        practical_cli.main(["--output", str(output)])
    marker = json.loads(output.read_text(encoding="utf-8"))
    assert marker["evaluation_artifact"]["completion_state"] == "failed"


def test_practical_cli_failed_rerun_replaces_stale_favorable_report_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    old_report = {"native_live_ok": True, "modeled_suite_ok": True}
    output.write_text(json.dumps(old_report), encoding="utf-8")

    def fail_evaluation() -> object:
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(
        practical_cli,
        "run_practical_evaluation",
        fail_evaluation,
    )

    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        practical_cli.main(["--output", str(output)])

    marker = json.loads(output.read_text(encoding="utf-8"))["evaluation_artifact"]
    assert marker["completion_state"] == "failed"
    previous = output.parent / marker["previous_artifact"]
    assert json.loads(previous.read_text(encoding="utf-8")) == old_report
    assert not list(tmp_path.glob("*.tmp"))


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


@pytest.mark.parametrize("scenarios", [[], (), iter(())])
def test_explicit_empty_selection_does_not_fall_back_to_defaults(
    scenarios: object,
) -> None:
    report = run_practical_evaluation(scenarios)  # type: ignore[arg-type]

    assert report.results == []
    assert report.scenario_counts == {"native-live": 0, "modeled": 0}
    assert report.native_tool_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_family", None),
        ("task_family", 7),
        ("variant", "   "),
        ("variant", []),
        ("attack_type", None),
        ("attack_type", {}),
    ],
)
def test_modeled_classification_fields_require_nonempty_strings(
    field: str,
    value: object,
) -> None:
    scenario = build_modeled_scenarios()[0]
    claim = dict(scenario.modeled_claim)
    claim[field] = value

    errors = validate_modeled_scenario(replace(scenario, modeled_claim=claim))

    assert any(field in error and "non-empty string" in error for error in errors)


def test_native_unsupported_expected_outcome_fails_before_tool_call(
    tmp_path: Path,
) -> None:
    scenario = next(
        item
        for item in default_scenarios()
        if item.evidence_level == EvidenceLevel.NATIVE_LIVE
    )
    unsupported = replace(
        scenario,
        effects=(
            replace(scenario.effects[0], expected_outcome="denied"),
        ),
    )

    report = run_practical_evaluation([unsupported], work_dir=tmp_path)

    assert report.results[0].ok is False
    assert report.results[0].tool_calls == 0
    assert report.results[0].external_effect_ids == []


def test_practical_dns_override_delegates_unrelated_hosts() -> None:
    observed: list[object] = []

    def original(host: object, *args: object, **kwargs: object) -> str:
        observed.append(host)
        return "delegated"

    benchmark = practical_runner._benchmark_getaddrinfo(
        original,
        "connector.example.test",
        443,
    )
    unrelated = practical_runner._benchmark_getaddrinfo(
        original,
        "unrelated.example.test",
        443,
    )

    assert benchmark[0][4] == ("93.184.216.34", 443)
    assert unrelated == "delegated"
    assert observed == ["unrelated.example.test"]


@pytest.mark.parametrize("raises", [False, True])
def test_native_tool_attempt_counter_includes_failures_and_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    scenario = next(
        item
        for item in default_scenarios()
        if item.evidence_level == EvidenceLevel.NATIVE_LIVE
    )

    class Tools:
        def call(self, *args: object, **kwargs: object) -> object:
            if raises:
                raise RuntimeError("injected call exception")
            return SimpleNamespace(ok=False, error="injected returned failure")

    runtime = SimpleNamespace(
        jsonrpc=SimpleNamespace(
            register_endpoint_from_yaml_text=lambda *args, **kwargs: None
        ),
        process=SimpleNamespace(spawn=lambda **kwargs: "pid-test"),
        store=SimpleNamespace(list_external_effects=lambda **kwargs: []),
        tools=Tools(),
        close=lambda: None,
    )
    monkeypatch.setattr(
        practical_runner.Runtime,
        "open",
        lambda *args, **kwargs: runtime,
    )

    result = practical_runner._run_native_scenario(
        scenario,
        work_dir=tmp_path,
    )

    assert result.ok is False
    assert result.tool_calls == 1


@pytest.mark.parametrize(
    ("claim_field", "replacement", "message"),
    [
        ("utility_oracle", {"requires": []}, "require at least one"),
        (
            "utility_oracle",
            {"requires": [{"effect_class": "filesystem.write", "target": "absent"}]},
            "absent from the scenario",
        ),
        ("security_oracle", {"forbidden_committed": 0, "forbidden": []}, "do not exactly match"),
        ("provenance_requirement", "runtime-backed", "disclaim runtime evidence"),
    ],
)
def test_modeled_oracle_validates_declared_claims_exactly(
    claim_field: str,
    replacement: object,
    message: str,
) -> None:
    scenario = next(
        item
        for item in build_modeled_scenarios()
        if item.modeled_claim["variant"] != "benign"
    )
    changed_claim = dict(scenario.modeled_claim)
    changed_claim[claim_field] = replacement
    invalid = replace(scenario, modeled_claim=changed_claim)

    assert any(message in error for error in validate_modeled_scenario(invalid))


def test_practical_report_semantic_validator_rejects_forged_native_evidence(
    tmp_path: Path,
) -> None:
    payload = run_practical_evaluation(default_scenarios(), work_dir=tmp_path).to_dict()
    native = next(
        row for row in payload["results"] if row["evidence_level"] == "native-live"
    )
    native["external_effect_ids"] = []

    errors = validate_practical_report(payload)

    assert any("one external effect id per semantic effect" in error for error in errors)


def test_practical_schema_validator_rejects_wrong_scalar_type() -> None:
    payload = PracticalRunReport(
        schema_version=1,
        results=[],
        scenario_counts={"native-live": 0, "modeled": 0},
        semantic_effect_counts={"native-live": 0, "modeled": 0},
        native_tool_calls=0,
        native_operations=0,
        modeled_fallback=0,
        native_live_ok=True,
        modeled_suite_ok=True,
    ).to_dict()
    payload["schema_version"] = True

    errors = validate_practical_report_schema(payload)

    assert errors and errors[0].startswith("schema_version:")


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
