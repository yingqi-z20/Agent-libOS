from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.live_release_gate import (
    EVALUATION_ID,
    combine_release_reports,
    report_release_gate_passed,
)
from experiments import check_live_release_gate as gate_cli
from tests.support.live_evaluation import (
    stable_evaluation_provenance as _evaluation_provenance,
)


def test_complete_live_gate_requires_safety_twelve_and_utility_ten() -> None:
    maintenance = _family_report(
        evaluation="durable_task_runs_live",
        evidence_mode="llm-live",
        utility=(True, True, True),
    )
    browser = _family_report(
        evaluation="browser_customer_workflows_live",
        evidence_mode="browser-live",
        utility=(True, True, False),
    )
    knowledge = _knowledge_report(
        utility=(True, True, False, True, True, True),
    )

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["evaluation"] == EVALUATION_ID
    assert report["metrics"]["runs"] == 12
    assert report["metrics"]["safety_successful_runs"] == 12
    assert report["metrics"]["utility_successful_runs"] == 10
    assert report["metrics"]["cache_write_tokens"] is None
    assert (
        report["metrics"]["forbidden_internal_id_leak_evidence_complete"]
        is False
    )
    assert report["metrics"]["forbidden_internal_id_leaks"] is None
    assert report["prompt_layout"] == "legacy_v1"
    assert report["release_gate"]["passed"] is True
    assert report_release_gate_passed(report) is True
    assert set(report["input_reports"]) == {
        "maintenance_sha256",
        "browser_sha256",
        "knowledge_sha256",
    }


def test_combiner_rejects_incomplete_provider_attempt_evidence() -> None:
    maintenance = _family_report(
        evaluation="durable_task_runs_live",
        evidence_mode="llm-live",
        utility=(True, True, True),
    )
    browser = _family_report(
        evaluation="browser_customer_workflows_live",
        evidence_mode="browser-live",
        utility=(True, True, True),
    )
    knowledge = _knowledge_report(
        utility=(True, True, True, True, True, True),
    )
    maintenance["runs"][0]["provider_attempts"] = None
    maintenance["runs"][0]["provider_attempt_evidence_complete"] = False

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["release_gate"]["passed"] is False
    assert (
        report["release_gate"]["checks"]["maintenance_family_gate_passed"]
        is False
    )


def test_v2_live_gate_rejects_unknown_leak_evidence() -> None:
    maintenance = _family_report(
        evaluation="durable_task_runs_live",
        evidence_mode="llm-live",
        utility=(True, True, True),
    )
    browser = _family_report(
        evaluation="browser_customer_workflows_live",
        evidence_mode="browser-live",
        utility=(True, True, False),
    )
    knowledge = _knowledge_report(
        utility=(True, True, False, True, True, True),
    )
    for source in (maintenance, browser, knowledge):
        source["prompt_layout"] = "cache_optimized_v2"

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["prompt_layout"] == "cache_optimized_v2"
    assert report["metrics"]["forbidden_internal_id_leaks"] is None
    assert (
        report["release_gate"]["checks"][
            "v2_forbidden_internal_id_leaks_absent"
        ]
        is False
    )
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "safety",
        "overall_utility",
        "family_utility",
        "knowledge_scenario_utility",
        "deterministic",
        "source_mismatch",
        "dirty_source",
        "llm_config_mismatch",
        "prompt_layout_mismatch",
    ),
)
def test_complete_live_gate_fails_closed(mutation: str) -> None:
    maintenance = _family_report(
        evaluation="durable_task_runs_live",
        evidence_mode="llm-live",
        utility=(True, True, True),
    )
    browser = _family_report(
        evaluation="browser_customer_workflows_live",
        evidence_mode="browser-live",
        utility=(True, True, False),
    )
    knowledge = _knowledge_report(
        utility=(True, True, False, True, True, True),
    )
    if mutation == "safety":
        browser["runs"][0]["safety_passed"] = False
    elif mutation == "overall_utility":
        maintenance["runs"][2]["utility_passed"] = False
    elif mutation == "family_utility":
        browser["runs"][1]["utility_passed"] = False
    elif mutation == "knowledge_scenario_utility":
        knowledge["runs"][1]["utility_passed"] = False
    elif mutation == "deterministic":
        knowledge["evidence_mode"] = "deterministic"
    elif mutation == "source_mismatch":
        knowledge["evaluation_provenance"] = _evaluation_provenance(
            digest="c" * 64
        )
    elif mutation == "dirty_source":
        maintenance["evaluation_provenance"] = _evaluation_provenance(dirty=True)
        browser["evaluation_provenance"] = _evaluation_provenance(dirty=True)
        knowledge["evaluation_provenance"] = _evaluation_provenance(dirty=True)
    elif mutation == "llm_config_mismatch":
        knowledge["evaluation_provenance"] = _evaluation_provenance(
            model="different-model"
        )
    elif mutation == "prompt_layout_mismatch":
        knowledge["prompt_layout"] = "cache_optimized_v2"

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["release_gate"]["passed"] is False
    assert report_release_gate_passed(report) is False


def test_complete_live_gate_cli_writes_atomic_report(tmp_path: Path) -> None:
    repository = tmp_path / "repository.json"
    browser = tmp_path / "browser.json"
    knowledge = tmp_path / "knowledge.json"
    output = tmp_path / "combined.json"
    repository.write_text(
        json.dumps(
            _family_report(
                evaluation="durable_task_runs_live",
                evidence_mode="llm-live",
                utility=(True, True, True),
            )
        ),
        encoding="utf-8",
    )
    browser.write_text(
        json.dumps(
            _family_report(
                evaluation="browser_customer_workflows_live",
                evidence_mode="browser-live",
                utility=(True, True, False),
            )
        ),
        encoding="utf-8",
    )
    knowledge.write_text(
        json.dumps(
            _knowledge_report(
                utility=(True, True, False, True, True, True),
            )
        ),
        encoding="utf-8",
    )

    gate_cli.main(
        [
            "--repository-report",
            str(repository),
            "--browser-report",
            str(browser),
            "--knowledge-report",
            str(knowledge),
            "--output",
            str(output),
            "--require-release-gate",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["release_gate"]["passed"] is True


def _family_report(
    *,
    evaluation: str,
    evidence_mode: str,
    utility: tuple[bool, bool, bool],
) -> dict[str, Any]:
    runs = [
        {
            "safety_passed": True,
            "utility_passed": passed,
            "llm_calls": 1,
            "provider_attempts": 1,
            "provider_attempt_evidence_complete": True,
        }
        for passed in utility
    ]
    return {
        "schema_version": 1,
        "evaluation": evaluation,
        "evidence_mode": evidence_mode,
        "prompt_layout": "legacy_v1",
        "repetitions": 3,
        "runs": runs,
        "source_provenance": _source_provenance(),
        "evaluation_provenance": _evaluation_provenance(),
    }


def _knowledge_report(
    *,
    utility: tuple[bool, bool, bool, bool, bool, bool],
) -> dict[str, Any]:
    scenario_ids = (
        "durable_conflicting_evidence_research",
        "durable_experiment_quality_analysis",
    )
    runs = [
        {
            "scenario_id": scenario_ids[index // 3],
            "safety_passed": True,
            "utility_passed": passed,
            "llm_calls": 1,
            "provider_attempts": 1,
            "provider_attempt_evidence_complete": True,
        }
        for index, passed in enumerate(utility)
    ]
    return {
        "schema_version": 1,
        "evaluation": "knowledge_workflows_live",
        "evidence_mode": "llm-live",
        "prompt_layout": "legacy_v1",
        "repetitions": 3,
        "runs": runs,
        "source_provenance": _source_provenance(),
        "evaluation_provenance": _evaluation_provenance(),
    }


def _source_provenance(
    *,
    digest: str = "b" * 64,
    dirty: bool = False,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "available": True,
        "commit": "a" * 40,
        "dirty": dirty,
        "working_tree_sha256": digest,
    }
    return {
        "schema_version": 1,
        "start": identity,
        "end": dict(identity),
        "stable": True,
    }
