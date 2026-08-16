from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import DEFAULT_CONFIG
from benchmarks.live_release_gate import (
    EVALUATION_ID,
    combine_release_reports,
    report_release_gate_passed,
)
from benchmarks.live_evaluation_provenance import _safe_llm_config_digest
from benchmarks.live_release_evidence import (
    collect_complete_checkpoints,
    collect_complete_llm_call_records,
    terminal_receipt_summary,
    validate_live_evidence_capture_capacity,
)
from benchmarks.prompt_cache_evidence import aggregate_prompt_cache_run_evidence
from experiments import check_live_release_gate as gate_cli
from tests.support.live_evaluation import (
    stable_evaluation_provenance as _evaluation_provenance,
)
from tests.support.live_release_reports import (
    browser_report,
    knowledge_report,
    maintenance_report,
    synchronize_report_metrics,
    synchronize_run_outcomes,
)


@pytest.mark.parametrize("ledger", ("llm", "checkpoint"))
def test_capture_capacity_reserves_a_non_truncated_slot(ledger: str) -> None:
    max_quanta = 4
    config = DEFAULT_CONFIG
    if ledger == "llm":
        config = replace(
            config,
            llm=replace(
                config.llm,
                call_record_list_limit=8,
                call_record_hard_limit=8,
            ),
        )
    else:
        config = replace(
            config,
            checkpoint=replace(config.checkpoint, list_limit=max_quanta),
        )

    with pytest.raises(ValueError, match="complete publication evidence"):
        validate_live_evidence_capture_capacity(config, max_quanta=max_quanta)


@pytest.mark.parametrize("count", (2, 3))
def test_bounded_collectors_reject_an_ambiguous_full_page(count: int) -> None:
    class _Store:
        def list_llm_calls(self, **_kwargs: Any) -> list[object]:
            return [object()] * count

    class _Checkpoint:
        def list(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            return [{}] * count

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            llm=SimpleNamespace(call_record_hard_limit=3),
            checkpoint=SimpleNamespace(list_limit=3),
        ),
        store=_Store(),
        checkpoint=_Checkpoint(),
    )
    if count == 2:
        assert len(collect_complete_llm_call_records(runtime, ["pid-1"])) == 2
        assert len(
            collect_complete_checkpoints(runtime, "pid-1", actor="pid-1")
        ) == 2
    else:
        with pytest.raises(RuntimeError, match="may be truncated"):
            collect_complete_llm_call_records(runtime, ["pid-1"])
        with pytest.raises(RuntimeError, match="may be truncated"):
            collect_complete_checkpoints(runtime, "pid-1", actor="pid-1")


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
    assert report["metrics"]["logical_llm_calls"] == 12
    assert report["metrics"]["provider_attempts"] == 12
    assert report["metrics"]["provider_attempt_evidence_complete"] is True
    assert report["metrics"]["prompt_tokens"] == 120
    assert report["metrics"]["completion_tokens"] == 24
    assert report["metrics"]["completion_evidence_successful_runs"] == 12
    assert report["metrics"]["cache_write_tokens"] is None
    assert report["metrics"]["forbidden_internal_id_leak_evidence_complete"] is True
    assert report["metrics"]["forbidden_internal_id_leaks"] == 0
    assert report["prompt_layout"] == "legacy_v1"
    assert report["release_gate"]["publication_ready"] is True
    assert report["families"]["browser_customer_workflow"]["valid"] is True
    assert report["families"]["knowledge_workflows"]["valid"] is True
    assert any(
        item["utility_passed"] is False
        for name in ("browser_customer_workflow", "knowledge_workflows")
        for item in report["families"][name]["run_assessments"]
    )
    assert report["release_gate"]["passed"] is True
    assert report_release_gate_passed(report) is True
    assert set(report["input_reports"]) == {
        "maintenance_sha256",
        "browser_sha256",
        "knowledge_sha256",
    }
    embedded_run = report["input_report_evidence"]["maintenance"]["runs"][0]
    assert embedded_run["task_run_requirement_count"] == 2
    assert embedded_run["task_run_satisfied_requirement_count"] == 2


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


@pytest.mark.parametrize(
    ("family_name", "evidence_key"),
    (
        ("repository_maintenance", "terminal"),
        ("repository_maintenance", "oracle"),
        ("repository_maintenance", "effects"),
        ("browser_customer_workflow", "receipts"),
        ("knowledge_workflows", "telemetry"),
    ),
)
def test_combiner_rejects_missing_run_evidence(
    family_name: str,
    evidence_key: str,
) -> None:
    maintenance = maintenance_report()
    browser = browser_report()
    knowledge = knowledge_report()
    selected = {
        "repository_maintenance": maintenance,
        "browser_customer_workflow": browser,
        "knowledge_workflows": knowledge,
    }[family_name]
    selected["runs"][0]["publication_evidence"].pop(evidence_key)

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["families"][family_name]["valid"] is False
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("safety_passed", False),
        ("utility_passed", False),
        ("passed", False),
        ("conclusion", "utility_failed"),
    ),
)
def test_combiner_rejects_tampered_derived_outcome(
    field: str,
    replacement: object,
) -> None:
    maintenance = maintenance_report()
    maintenance["runs"][0][field] = replacement

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


def test_combiner_rejects_tampered_family_metrics() -> None:
    maintenance = maintenance_report()
    maintenance["metrics"]["safety_successful_runs"] = 2

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize("mutation", ("duplicate", "missing"))
def test_combiner_rejects_nonexact_run_grid(mutation: str) -> None:
    browser = browser_report()
    if mutation == "duplicate":
        duplicate = copy.deepcopy(browser["runs"][0])
        duplicate["run_id"] = "run-duplicate-slot"
        duplicate["root_pid"] = "pid-duplicate-slot"
        browser["runs"][2] = duplicate
    else:
        browser["runs"].pop()
        browser["metrics"]["runs"] = 2
        browser["metrics"]["safety_successful_runs"] = 2
        browser["metrics"]["utility_successful_runs"] = 2
        browser["metrics"]["successful_runs"] = 2
        browser["metrics"]["provider_attempts"] = 2

    report = combine_release_reports(
        maintenance_report(),
        browser,
        knowledge_report(),
    )

    assert report["families"]["browser_customer_workflow"]["valid"] is False
    assert report["release_gate"]["passed"] is False


def test_combiner_keeps_complete_safety_negative_valid_but_fails_gate() -> None:
    maintenance = maintenance_report(safety=(True, True, False))

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    family = report["families"]["repository_maintenance"]
    assert family["valid"] is True
    assert family["safety_successes"] == 2
    assert report["metrics"]["runs"] == 12
    assert report["metrics"]["safety_successful_runs"] == 11
    assert report["release_gate"]["publication_ready"] is True
    assert report["release_gate"]["passed"] is False


def test_explicit_absent_terminal_receipt_is_a_complete_negative() -> None:
    maintenance = maintenance_report()
    run = maintenance["runs"][0]
    evidence = run["publication_evidence"]
    run["final_status"] = "failed"
    run["final_process_status"] = "failed"
    evidence["terminal"]["task_run_status"] = "failed"
    evidence["terminal"]["process_status"] = "failed"
    run["workflow_evidence"] = []
    run["actions"] = []
    run["successful_actions"] = []
    evidence["receipts"]["workflow_evidence"] = []
    evidence["receipts"]["terminal_receipt_summary"] = (
        terminal_receipt_summary([])
    )
    evidence["oracle"]["safety_checks"]["task_run_succeeded"] = False
    evidence["oracle"]["utility_checks"]["exited"] = False
    evidence["oracle"]["utility_checks"]["finalization_evidence_fresh"] = False
    evidence["oracle"]["utility_checks"]["required_actions_observed"] = False
    evidence["oracle"]["utility_checks"]["required_actions_successful"] = False
    evidence["oracle"]["utility_checks"]["checkpoint_created"] = False
    synchronize_run_outcomes(run)
    synchronize_report_metrics(maintenance)

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    family = report["families"]["repository_maintenance"]
    assert family["valid"] is True
    assert family["run_assessments"][0]["safety_passed"] is False
    assert family["run_assessments"][0]["utility_passed"] is False
    assert report["release_gate"]["publication_ready"] is True
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize("family", ("maintenance", "browser", "knowledge"))
def test_combiner_rejects_legacy_family_schema(family: str) -> None:
    maintenance = maintenance_report()
    browser = browser_report()
    knowledge = knowledge_report()
    {"maintenance": maintenance, "browser": browser, "knowledge": knowledge}[
        family
    ]["schema_version"] = 1

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["release_gate"]["passed"] is False


def test_combined_legacy_schema_cannot_pass_current_gate() -> None:
    report = combine_release_reports(
        maintenance_report(),
        browser_report(utility=(True, True, False)),
        knowledge_report(utility=(True, True, False, True, True, True)),
    )
    report["schema_version"] = 2

    assert report_release_gate_passed(report) is False


@pytest.mark.parametrize(
    "mutation",
    (
        "input_hash",
        "embedded_input",
        "source_identity",
        "llm_identity",
        "evaluation_identity",
        "prompt_layout",
        "extra_check",
        "missing_check",
        "threshold",
        "telemetry_total",
        "generated_at",
        "extra_top_level",
        "synchronized_family_projection",
    ),
)
def test_persisted_combined_report_is_a_closed_recomputable_projection(
    mutation: str,
) -> None:
    report = combine_release_reports(
        maintenance_report(),
        browser_report(),
        knowledge_report(),
    )
    assert report_release_gate_passed(report) is True

    if mutation == "input_hash":
        report["input_reports"]["maintenance_sha256"] = "0" * 64
    elif mutation == "embedded_input":
        report["input_report_evidence"]["maintenance"]["runs"][0][
            "utility_passed"
        ] = False
    elif mutation == "source_identity":
        report["source_identity"]["commit"] = "f" * 40
    elif mutation == "llm_identity":
        report["llm_identity"]["model"] = "tampered-model"
    elif mutation == "evaluation_identity":
        report["evaluation_identity"]["source"]["commit"] = "f" * 40
    elif mutation == "prompt_layout":
        report["prompt_layout"] = "cache_optimized_v2"
    elif mutation == "extra_check":
        report["release_gate"]["checks"]["unregistered_check"] = True
    elif mutation == "missing_check":
        report["release_gate"]["checks"].pop("source_worktree_clean")
    elif mutation == "threshold":
        report["release_gate"]["required_runs"] = 11
    elif mutation == "telemetry_total":
        report["metrics"]["logical_llm_calls"] = 11
    elif mutation == "generated_at":
        report["generated_at"] = "not-a-timestamp"
    elif mutation == "extra_top_level":
        report["unregistered_projection"] = True
    else:
        family = report["families"]["repository_maintenance"]
        family["run_assessments"][0]["utility_passed"] = False
        family["utility_successes"] = 2
        report["metrics"]["utility_successful_runs"] = 11

    assert report_release_gate_passed(report) is False


@pytest.mark.parametrize("family", ("maintenance", "browser", "knowledge"))
def test_family_grid_rejects_boolean_repetition(family: str) -> None:
    maintenance = maintenance_report()
    browser = browser_report()
    knowledge = knowledge_report()
    selected = {
        "maintenance": maintenance,
        "browser": browser,
        "knowledge": knowledge,
    }[family]
    selected["runs"][0]["repetition"] = True

    report = combine_release_reports(maintenance, browser, knowledge)

    family_key = {
        "maintenance": "repository_maintenance",
        "browser": "browser_customer_workflow",
        "knowledge": "knowledge_workflows",
    }[family]
    assert report["families"][family_key]["valid"] is False
    assert report_release_gate_passed(report) is False


@pytest.mark.parametrize(
    "mutation",
    (
        "run_evidence_schema_bool",
        "scenario_contract_schema_bool",
        "workflow_extra_field",
        "workflow_shell_types",
        "fabricated_terminal_status",
        "completed_steps_exceed_steps",
        "succeeded_completed_steps_missing",
        "revision_precedes_steps",
        "uncommitted_exited_receipt",
        "successful_receipt_without_status",
        "fabricated_action_receipt",
        "fabricated_effect_state",
        "missing_effect_dispatch_count",
        "leak_call_extra_field",
        "family_extra_field",
        "run_extra_field",
        "release_gate_extra_field",
    ),
)
def test_family_publication_contract_is_closed(mutation: str) -> None:
    maintenance = maintenance_report()
    run = maintenance["runs"][0]
    evidence = run["publication_evidence"]
    if mutation == "run_evidence_schema_bool":
        evidence["schema_version"] = True
    elif mutation == "scenario_contract_schema_bool":
        maintenance["scenario_contracts"][0]["schema_version"] = True
    elif mutation == "workflow_extra_field":
        run["workflow_evidence"][0]["secret"] = "canary"
        evidence["receipts"]["workflow_evidence"][0]["secret"] = "canary"
    elif mutation == "workflow_shell_types":
        shell = {
            "sequence_index": 0,
            "action": "run_shell_command",
            "ok": True,
            "tool_id": "tool-shell",
            "result_oid": "obj-shell",
            "returncode": "secret",
            "stdout_truncated": "secret",
            "stderr_truncated": {},
            "resource_limited": 7,
        }
        exit_receipt = copy.deepcopy(run["workflow_evidence"][0])
        exit_receipt["sequence_index"] = 1
        workflow = [shell, exit_receipt]
        run["workflow_evidence"] = copy.deepcopy(workflow)
        run["actions"] = ["run_shell_command", "process_exit"]
        run["successful_actions"] = ["run_shell_command", "process_exit"]
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(workflow)
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(workflow)
        )
    elif mutation == "fabricated_terminal_status":
        run["final_status"] = "fabricated-terminal"
        run["final_process_status"] = "fabricated-terminal"
        evidence["terminal"]["task_run_status"] = "fabricated-terminal"
        evidence["terminal"]["process_status"] = "fabricated-terminal"
        run["workflow_evidence"] = []
        run["actions"] = []
        run["successful_actions"] = []
        evidence["receipts"]["workflow_evidence"] = []
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary([])
        )
        evidence["oracle"]["safety_checks"]["task_run_succeeded"] = False
        evidence["oracle"]["utility_checks"]["exited"] = False
        evidence["oracle"]["utility_checks"][
            "finalization_evidence_fresh"
        ] = False
        synchronize_run_outcomes(run)
        synchronize_report_metrics(maintenance)
    elif mutation == "completed_steps_exceed_steps":
        run["task_run_completed_step_count"] = 3
        evidence["terminal"]["task_run_completed_step_count"] = 3
    elif mutation == "succeeded_completed_steps_missing":
        run["task_run_completed_step_count"] = 0
        evidence["terminal"]["task_run_completed_step_count"] = 0
    elif mutation == "revision_precedes_steps":
        run["task_run_revision"] = 0
        evidence["terminal"]["task_run_revision"] = 0
    elif mutation == "uncommitted_exited_receipt":
        run["workflow_evidence"][0]["terminal_committed"] = False
        evidence["receipts"]["workflow_evidence"][0][
            "terminal_committed"
        ] = False
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(run["workflow_evidence"])
        )
    elif mutation == "successful_receipt_without_status":
        run["workflow_evidence"][0]["status"] = None
        run["workflow_evidence"][0]["terminal_committed"] = False
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(
            run["workflow_evidence"]
        )
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(run["workflow_evidence"])
        )
    elif mutation == "fabricated_action_receipt":
        fabricated = {
            "sequence_index": 0,
            "action": "fabricated-action",
            "ok": True,
            "tool_id": None,
            "result_oid": None,
        }
        exit_receipt = copy.deepcopy(run["workflow_evidence"][0])
        exit_receipt["sequence_index"] = 1
        workflow = [fabricated, exit_receipt]
        run["workflow_evidence"] = copy.deepcopy(workflow)
        run["actions"] = ["fabricated-action", "process_exit"]
        run["successful_actions"] = ["fabricated-action", "process_exit"]
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(workflow)
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(workflow)
        )
    elif mutation == "fabricated_effect_state":
        summary = {
            "by_transaction_state": {"fabricated": 1},
            "by_provider": {"test-provider": 1},
            "unsettled_by_provider_operation": {"test-provider:effect": 1},
        }
        run["external_effect_state_summary"] = copy.deepcopy(summary)
        evidence["effects"]["external_effect_state_summary"] = copy.deepcopy(
            summary
        )
        evidence["oracle"]["safety_checks"]["all_external_effects_settled"] = (
            False
        )
        synchronize_run_outcomes(run)
        synchronize_report_metrics(maintenance)
    elif mutation == "missing_effect_dispatch_count":
        run["maximum_dispatches_per_effect"] = 0
        evidence["effects"]["maximum_dispatches_per_effect"] = 0
    elif mutation == "leak_call_extra_field":
        categories = run["forbidden_internal_id_leaks_by_category"]
        categories["host_contract_fields"] = 1
        run["forbidden_internal_id_leaks"] = 1
        run["forbidden_internal_id_leak_call_count"] = 1
        run["forbidden_internal_id_leak_calls"] = [
            {
                "call_ordinal": 1,
                "categories": {"host_contract_fields": 1},
                "surfaces": {"messages": 1},
                "response_tools": [],
                "secret": "canary",
            }
        ]
        maintenance["metrics"].update(
            aggregate_prompt_cache_run_evidence(maintenance["runs"])
        )
    elif mutation == "family_extra_field":
        maintenance["secret"] = "canary"
    elif mutation == "run_extra_field":
        run["secret"] = "canary"
    else:
        maintenance["release_gate"] = {
            "passed": False,
            "publication_ready": False,
            "forged": True,
        }

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "maintenance_behavior",
        "browser_refund",
        "browser_unique_mutation",
        "research_sources",
        "analysis_artifact",
        "analysis_recommendation",
    ),
)
def test_fixed_oracle_checks_are_recomputed_from_raw_fields(
    mutation: str,
) -> None:
    maintenance = maintenance_report()
    browser = browser_report()
    knowledge = knowledge_report()
    if mutation == "maintenance_behavior":
        run = maintenance["runs"][0]
        run["behavior_probe"]["exact_threshold"] = False
        run["publication_evidence"]["oracle"]["fields"]["behavior_probe"][
            "exact_threshold"
        ] = False
    elif mutation == "browser_refund":
        run = browser["runs"][0]
        run["portal"]["order"]["refunded_amount"] = "0.00"
        run["publication_evidence"]["oracle"]["fields"]["portal"]["order"][
            "refunded_amount"
        ] = "0.00"
    elif mutation == "browser_unique_mutation":
        run = browser["runs"][0]
        run["portal"]["telemetry"]["unique_refund_mutations"] = 0
        run["publication_evidence"]["oracle"]["fields"]["portal"][
            "telemetry"
        ]["unique_refund_mutations"] = 0
    elif mutation == "research_sources":
        run = knowledge["runs"][0]
        run["oracle"]["sources_read"] = 0
        run["publication_evidence"]["oracle"]["fields"]["oracle"][
            "sources_read"
        ] = 0
    else:
        run = next(
            item
            for item in knowledge["runs"]
            if item["scenario_id"] == "durable_experiment_quality_analysis"
        )
        projection = run["publication_evidence"]["oracle"]["fields"]["oracle"]
        if mutation == "analysis_artifact":
            run["oracle"]["artifact_semantics_valid"] = False
            run["oracle"]["artifact_validation_errors"] = [
                "value_mismatch:recommendation"
            ]
            projection["artifact_semantics_valid"] = False
            projection["artifact_validation_errors"] = [
                "value_mismatch:recommendation"
            ]
        else:
            run["oracle"]["recommendation"] = None
            projection["recommendation"] = None

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["release_gate"]["publication_ready"] is False
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "non_contiguous_receipts",
        "unknown_successful_action",
        "successful_shell_without_result",
        "boolean_checkpoint_count",
        "forged_llm_error_categories",
        "forged_tool_failure",
        "missing_live_output_usage",
    ),
)
def test_family_rejects_noncausal_or_unredacted_run_evidence(
    mutation: str,
) -> None:
    maintenance = maintenance_report()
    run = maintenance["runs"][0]
    evidence = run["publication_evidence"]
    if mutation == "non_contiguous_receipts":
        for receipt in run["workflow_evidence"]:
            receipt["sequence_index"] += 99
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(
            run["workflow_evidence"]
        )
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(run["workflow_evidence"])
        )
    elif mutation == "unknown_successful_action":
        workflow = copy.deepcopy(run["workflow_evidence"])
        exit_receipt = workflow.pop()
        workflow.append(
            {
                "sequence_index": len(workflow),
                "action": "fabricated_success",
                "ok": True,
                "tool_id": "tool-forged",
                "result_oid": "obj-forged",
            }
        )
        exit_receipt["sequence_index"] = len(workflow)
        workflow.append(exit_receipt)
        run["workflow_evidence"] = copy.deepcopy(workflow)
        run["actions"] = [item["action"] for item in workflow]
        run["successful_actions"] = [item["action"] for item in workflow]
        run["initial_model_tools"] = sorted(
            {*run["initial_model_tools"], "fabricated_success"}
        )
        run["final_model_tools"] = sorted(
            {*run["final_model_tools"], "fabricated_success"}
        )
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(workflow)
        evidence["receipts"]["terminal_receipt_summary"] = (
            terminal_receipt_summary(workflow)
        )
    elif mutation == "successful_shell_without_result":
        shell = next(
            item
            for item in run["workflow_evidence"]
            if item["action"] == "run_shell_command"
        )
        shell["returncode"] = None
        shell["resource_limited"] = True
        evidence["receipts"]["workflow_evidence"] = copy.deepcopy(
            run["workflow_evidence"]
        )
    elif mutation == "boolean_checkpoint_count":
        run["checkpoint_count"] = True
    elif mutation == "forged_llm_error_categories":
        run["llm_error_categories"] = {"secret-category": 99}
    elif mutation == "forged_tool_failure":
        run["tool_failures"] = [{"secret": "canary"}]
    else:
        run["completion_tokens"] = 0
        run["total_output_tokens"] = 0
        evidence["telemetry"]["completion_tokens"] = 0
        maintenance["metrics"].update(
            aggregate_prompt_cache_run_evidence(maintenance["runs"])
        )

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


def test_complete_raw_oracle_negative_remains_publishable() -> None:
    browser = browser_report()
    run = browser["runs"][0]
    run["portal"]["order"]["refunded_amount"] = "0.00"
    run["publication_evidence"]["oracle"]["fields"]["portal"]["order"][
        "refunded_amount"
    ] = "0.00"
    run["publication_evidence"]["oracle"]["utility_checks"][
        "exact_refund_observed"
    ] = False
    synchronize_run_outcomes(run)
    synchronize_report_metrics(browser)

    report = combine_release_reports(
        maintenance_report(),
        browser,
        knowledge_report(),
    )

    assert report["families"]["browser_customer_workflow"]["valid"] is True
    assert report["families"]["browser_customer_workflow"][
        "utility_successes"
    ] == 2
    assert report["release_gate"]["publication_ready"] is True
    assert report["release_gate"]["passed"] is True


def test_browser_predispatch_failures_remain_publishable_negatives() -> None:
    browser = browser_report()
    for run in browser["runs"][:2]:
        method_calls = run["method_calls"]
        failed_call = next(
            item for item in method_calls if item["method_id"] == "get-order"
        )
        failed_call["ok"] = False
        evidence_fields = run["publication_evidence"]["oracle"]["fields"]
        evidence_call = next(
            item
            for item in evidence_fields["method_calls"]
            if item["sequence_index"] == failed_call["sequence_index"]
        )
        evidence_call["ok"] = False
        receipt = run["workflow_evidence"][failed_call["sequence_index"]]
        receipt["ok"] = False
        receipt["tool_id"] = None
        receipt["result_oid"] = None
        run["successful_actions"] = [
            item["action"]
            for item in run["workflow_evidence"]
            if item["ok"] is True
        ]
        receipts = run["publication_evidence"]["receipts"]
        receipts["workflow_evidence"] = copy.deepcopy(run["workflow_evidence"])
        receipts["terminal_receipt_summary"] = terminal_receipt_summary(
            run["workflow_evidence"]
        )
        run["portal"]["telemetry"]["rpc_calls"]["portal.get_order"] = 1
        evidence_fields["portal"]["telemetry"]["rpc_calls"][
            "portal.get_order"
        ] = 1
        run["tool_failure_count"] = 1
        run["tool_failures"] = [
            {"action": "call_jsonrpc_method", "category": "authorization"}
        ]
        run["publication_evidence"]["telemetry"]["tool_failure_count"] = 1
        run["publication_evidence"]["oracle"]["utility_checks"][
            "fresh_order_read_before_mutation"
        ] = False
        synchronize_run_outcomes(run)
    synchronize_report_metrics(browser)

    report = combine_release_reports(
        maintenance_report(),
        browser,
        knowledge_report(),
    )

    family = report["families"]["browser_customer_workflow"]
    assert family["valid"] is True
    assert family["utility_successes"] == 1
    assert report["release_gate"]["publication_ready"] is True
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    "missing_field",
    (
        "cache_total_calls",
        "total_input_tokens",
        "total_output_tokens",
        "forbidden_internal_id_leak_evidence_complete",
        "forbidden_internal_id_leak_calls",
        "forbidden_internal_id_leak_call_count",
    ),
)
def test_family_requires_complete_per_run_cache_telemetry(
    missing_field: str,
) -> None:
    maintenance = maintenance_report()
    maintenance["runs"][0].pop(missing_field)
    maintenance["metrics"].update(
        aggregate_prompt_cache_run_evidence(maintenance["runs"])
    )

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "read_without_reported_union",
        "metric_without_read",
        "read_tokens_without_read_call",
        "metric_input_exceeds_total",
    ),
)
def test_family_rejects_internally_inconsistent_cache_coverage(
    mutation: str,
) -> None:
    maintenance = maintenance_report()
    run = maintenance["runs"][0]
    if mutation == "read_without_reported_union":
        run["cache_read_reported_calls"] = 1
    elif mutation == "metric_without_read":
        run["cache_metric_reported_calls"] = 1
    elif mutation == "read_tokens_without_read_call":
        run["cache_read_tokens"] = 1
    else:
        run["cache_reported_calls"] = 1
        run["cache_read_reported_calls"] = 1
        run["cache_metric_reported_calls"] = 1
        run["cache_metric_input_tokens"] = 100
        run["uncached_input_tokens"] = 100
        run["cache_hit_rate"] = 0.0
    maintenance["metrics"].update(
        aggregate_prompt_cache_run_evidence(maintenance["runs"])
    )

    report = combine_release_reports(
        maintenance,
        browser_report(),
        knowledge_report(),
    )

    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["passed"] is False


def test_combined_schema_rejects_boolean_version() -> None:
    report = combine_release_reports(
        maintenance_report(), browser_report(), knowledge_report()
    )
    report["schema_version"] = True

    assert report_release_gate_passed(report) is False


def test_knowledge_by_scenario_metrics_are_closed() -> None:
    knowledge = knowledge_report()
    scenario_id = knowledge["scenario_contracts"][0]["scenario_id"]
    knowledge["metrics"]["by_scenario"][scenario_id]["secret"] = "canary"

    report = combine_release_reports(
        maintenance_report(), browser_report(), knowledge
    )

    assert report["families"]["knowledge_workflows"]["valid"] is False
    assert report["release_gate"]["passed"] is False
    assert "canary" not in json.dumps(report)


def test_complete_v2_leak_observation_is_publishable_negative() -> None:
    maintenance = maintenance_report()
    browser = browser_report()
    knowledge = knowledge_report()
    for family in (maintenance, browser, knowledge):
        family["prompt_layout"] = "cache_optimized_v2"
        for boundary in ("start", "end"):
            llm = family["evaluation_provenance"][boundary]["llm"]
            llm["prompt"]["layout"] = "cache_optimized_v2"
            safe = dict(llm)
            safe.pop("config_sha256")
            llm["config_sha256"] = _safe_llm_config_digest(safe)
    run = maintenance["runs"][0]
    run["forbidden_internal_id_leaks"] = 1
    run["forbidden_internal_id_leaks_by_category"][
        "host_contract_fields"
    ] = 1
    run["forbidden_internal_id_leak_call_count"] = 1
    run["forbidden_internal_id_leak_calls"] = [
        {
            "call_ordinal": 1,
            "categories": {"host_contract_fields": 1},
            "surfaces": {"messages": 1},
            "response_tools": [],
        }
    ]
    maintenance["metrics"].update(
        aggregate_prompt_cache_run_evidence(maintenance["runs"])
    )

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["release_gate"]["checks"][
        "v2_forbidden_internal_id_leaks_absent"
    ] is False
    assert report["release_gate"]["publication_ready"] is True
    assert report["release_gate"]["passed"] is False
    assert report_release_gate_passed(report) is False


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
        source["evaluation_provenance"] = _evaluation_provenance(
            prompt_layout="cache_optimized_v2"
        )
    maintenance["runs"][0].pop("forbidden_internal_id_leaks_by_category")

    report = combine_release_reports(maintenance, browser, knowledge)

    assert report["prompt_layout"] is None
    assert report["families"]["repository_maintenance"]["valid"] is False
    assert report["release_gate"]["publication_ready"] is False
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
    if evaluation == "durable_task_runs_live":
        report = maintenance_report(utility=utility)
    elif evaluation == "browser_customer_workflows_live":
        report = browser_report(utility=utility)
    else:
        raise AssertionError(f"unexpected test evaluation: {evaluation}")
    report["evidence_mode"] = evidence_mode
    return report


def _knowledge_report(
    *,
    utility: tuple[bool, bool, bool, bool, bool, bool],
) -> dict[str, Any]:
    return knowledge_report(utility=utility)
