from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_libos.images import DEFAULT_IMAGES
from benchmarks.browser_customer_workflows.evaluation import (
    EVALUATION_ID as BROWSER_EVALUATION_ID,
    scenario_contract as browser_scenario_contract,
)
from benchmarks.durable_task_runs.live_evaluation import (
    EVALUATION_ID as MAINTENANCE_EVALUATION_ID,
    scenario_contract as maintenance_scenario_contract,
)
from benchmarks.knowledge_workflows.evaluation import (
    EVALUATION_ID as KNOWLEDGE_EVALUATION_ID,
    scenario_contracts as knowledge_scenario_contracts,
)
from benchmarks.live_release_evidence import (
    FAMILY_REPORT_SCHEMA_VERSION,
    build_run_evidence,
)
from benchmarks.prompt_cache_evidence import (
    FORBIDDEN_MODEL_TEXT_CATEGORIES,
    aggregate_prompt_cache_run_evidence,
)
from tests.support.live_evaluation import stable_evaluation_provenance


def maintenance_report(
    *,
    utility: Sequence[bool] = (True, True, True),
    safety: Sequence[bool] = (True, True, True),
) -> dict[str, Any]:
    return _family_report(
        evaluation=MAINTENANCE_EVALUATION_ID,
        evidence_mode="llm-live",
        contracts=[maintenance_scenario_contract()],
        utility=utility,
        safety=safety,
    )


def browser_report(
    *,
    utility: Sequence[bool] = (True, True, True),
    safety: Sequence[bool] = (True, True, True),
) -> dict[str, Any]:
    return _family_report(
        evaluation=BROWSER_EVALUATION_ID,
        evidence_mode="browser-live",
        contracts=[browser_scenario_contract()],
        utility=utility,
        safety=safety,
    )


def knowledge_report(
    *,
    utility: Sequence[bool] = (True, True, True, True, True, True),
    safety: Sequence[bool] = (True, True, True, True, True, True),
) -> dict[str, Any]:
    return _family_report(
        evaluation=KNOWLEDGE_EVALUATION_ID,
        evidence_mode="llm-live",
        contracts=knowledge_scenario_contracts(),
        utility=utility,
        safety=safety,
    )


def synchronize_run_outcomes(run: dict[str, Any]) -> None:
    oracle = run["publication_evidence"]["oracle"]
    safety = all(oracle["safety_checks"].values())
    utility = all(oracle["utility_checks"].values())
    run["safety_checks"] = dict(oracle["safety_checks"])
    run["utility_checks"] = dict(oracle["utility_checks"])
    run["safety_passed"] = safety
    run["utility_passed"] = utility
    run["passed"] = safety and utility
    run["conclusion"] = (
        "passed"
        if safety and utility
        else "safety_failed"
        if not safety
        else "utility_failed"
    )


def synchronize_report_metrics(report: dict[str, Any]) -> None:
    runs = report["runs"]
    safety = sum(run["safety_passed"] for run in runs)
    utility = sum(run["utility_passed"] for run in runs)
    successful = sum(run["passed"] for run in runs)
    attempts = sum(run["provider_attempts"] for run in runs)
    updated = {
        "runs": len(runs),
        "safety_successful_runs": safety,
        "utility_successful_runs": utility,
        "successful_runs": successful,
        "provider_attempts": attempts,
        "mean_provider_attempts": attempts / len(runs),
        "provider_attempt_evidence_complete": True,
    }
    if len(report.get("scenario_contracts", [])) == 1:
        updated["safety_success_rate"] = safety / len(runs)
        updated["utility_success_rate"] = utility / len(runs)
    report["metrics"].update(updated)
    report["metrics"].update(aggregate_prompt_cache_run_evidence(runs))


def _family_report(
    *,
    evaluation: str,
    evidence_mode: str,
    contracts: Sequence[dict[str, Any]],
    utility: Sequence[bool],
    safety: Sequence[bool],
) -> dict[str, Any]:
    expected_runs = len(contracts) * 3
    if len(utility) != expected_runs or len(safety) != expected_runs:
        raise ValueError("one outcome is required for each frozen run slot")
    runs: list[dict[str, Any]] = []
    ordinal = 0
    for contract in contracts:
        for repetition in range(1, 4):
            run = _run(
                contract,
                repetition=repetition,
                ordinal=ordinal,
                safety_passed=bool(safety[ordinal]),
                utility_passed=bool(utility[ordinal]),
            )
            runs.append(run)
            ordinal += 1
    cache = aggregate_prompt_cache_run_evidence(runs)
    attempts = sum(run["provider_attempts"] for run in runs)
    by_scenario = {
        contract["scenario_id"]: _scenario_metrics(
            runs,
            contract["scenario_id"],
        )
        for contract in contracts
    }
    report = {
        "schema_version": FAMILY_REPORT_SCHEMA_VERSION,
        "evaluation": evaluation,
        "evidence_mode": evidence_mode,
        "prompt_layout": "legacy_v1",
        "repetitions": 3,
        "scenario_contracts": [dict(contract) for contract in contracts],
        "runs": runs,
        "metrics": {
            "runs": len(runs),
            "safety_successful_runs": sum(run["safety_passed"] for run in runs),
            "utility_successful_runs": sum(run["utility_passed"] for run in runs),
            "successful_runs": sum(run["passed"] for run in runs),
            "provider_attempts": attempts,
            "mean_provider_attempts": attempts / len(runs),
            "provider_attempt_evidence_complete": True,
            "mean_llm_calls": 1.0,
            "mean_external_effects": 1.0,
            **cache,
            **(
                {
                    "safety_success_rate": (
                        sum(run["safety_passed"] for run in runs) / len(runs)
                    ),
                    "utility_success_rate": (
                        sum(run["utility_passed"] for run in runs) / len(runs)
                    ),
                }
                if len(contracts) == 1
                else {}
            ),
            **({"by_scenario": by_scenario} if len(contracts) > 1 else {}),
        },
        "evaluation_provenance": stable_evaluation_provenance(),
    }
    return report


def _scenario_metrics(
    runs: Sequence[dict[str, Any]],
    scenario_id: str,
) -> dict[str, Any]:
    selected = [run for run in runs if run["scenario_id"] == scenario_id]
    attempts = sum(run["provider_attempts"] for run in selected)
    return {
        "runs": len(selected),
        "safety_successful_runs": sum(run["safety_passed"] for run in selected),
        "utility_successful_runs": sum(run["utility_passed"] for run in selected),
        "successful_runs": sum(run["passed"] for run in selected),
        "provider_attempts": attempts,
        "mean_provider_attempts": attempts / len(selected),
        "provider_attempt_evidence_complete": True,
    }


def _run(
    contract: dict[str, Any],
    *,
    repetition: int,
    ordinal: int,
    safety_passed: bool,
    utility_passed: bool,
) -> dict[str, Any]:
    safety_checks = {key: True for key in contract["safety_check_ids"]}
    utility_checks = {key: True for key in contract["utility_check_ids"]}
    if not safety_passed:
        safety_checks["same_run_id_after_reopen"] = False
    if not utility_passed:
        candidate = {
            "repository-maintenance-oracle-v2": "restart_survived",
            "browser-customer-refund-oracle-v2": "human_result_delivered",
            "knowledge-research-oracle-v2": "mandatory_residency_reasoned",
            "knowledge-analysis-oracle-v2": "quality_counts_reported",
        }[contract["oracle_contract_id"]]
        utility_checks[candidate] = False
    workflow = _workflow_evidence(contract, ordinal=ordinal)
    if not utility_passed and candidate == "human_result_delivered":
        workflow = [
            item for item in workflow if item["action"] != "human_output"
        ]
        for index, item in enumerate(workflow):
            item["sequence_index"] = index
    observed_actions = {str(item["action"]) for item in workflow}
    successful_action_set = {
        str(item["action"]) for item in workflow if item["ok"] is True
    }
    required_actions = set(contract["required_action_ids"])
    if "required_actions_observed" in utility_checks:
        utility_checks["required_actions_observed"] = (
            required_actions <= observed_actions
        )
    if "required_actions_successful" in utility_checks:
        utility_checks["required_actions_successful"] = (
            required_actions <= successful_action_set
        )
    if "checkpoint_created" in utility_checks:
        utility_checks["checkpoint_created"] = (
            "create_checkpoint" in successful_action_set
        )
    if "required_authorized_tools_succeeded" in safety_checks:
        safety_checks["required_authorized_tools_succeeded"] = (
            observed_actions & required_actions
        ) <= successful_action_set
    oracle_fields = _oracle_fields(contract, workflow=workflow)
    model_tools = sorted(DEFAULT_IMAGES[contract["image_id"]].default_tools)
    state_summary = {
        "by_transaction_state": {"committed": 1},
        "by_provider": {"test-provider": 1},
        "unsettled_by_provider_operation": {},
    }
    run: dict[str, Any] = {
        "scenario_id": contract["scenario_id"],
        "image_id": contract["image_id"],
        "repetition": repetition,
        "run_id": f"run-{ordinal}",
        "root_pid": f"pid-{ordinal}",
        "final_status": "succeeded",
        "final_process_status": "exited",
        "safety_checks": safety_checks,
        "utility_checks": utility_checks,
        **oracle_fields,
        "workflow_evidence": workflow,
        "actions": [str(item["action"]) for item in workflow],
        "successful_actions": [
            str(item["action"]) for item in workflow if item["ok"] is True
        ],
        "activated_skills": [
            str(item["skill_id"])
            for item in workflow
            if item["action"] == "activate_skill" and item["ok"] is True
        ],
        "initial_model_tools": model_tools,
        "final_model_tools": model_tools,
        "llm_calls": 1,
        "provider_attempts": 1,
        "provider_attempt_evidence_complete": True,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "invalid_tool_calls": 0,
        "llm_error_count": 0,
        "llm_error_categories": {},
        "tool_failure_count": 0,
        "tool_failures": [],
        "external_effect_count": 1,
        "external_effect_state_summary": state_summary,
        "external_effect_transition_count": 2,
        "maximum_dispatches_per_effect": 1,
        "task_run_revision": 3,
        "task_run_step_count": 2,
        "task_run_completed_step_count": 2,
        "task_run_requirement_count": 2,
        "task_run_satisfied_requirement_count": 2,
        "attention_blocker_kinds": [],
        "forbidden_internal_id_leak_evidence_complete": True,
        "forbidden_internal_id_leaks": 0,
        "forbidden_internal_id_leaks_by_category": {
            category: 0 for category in FORBIDDEN_MODEL_TEXT_CATEGORIES
        },
        "forbidden_internal_id_leak_calls": [],
        "forbidden_internal_id_leak_call_count": 0,
        "cache_total_calls": 1,
        "cache_reported_calls": 0,
        "cache_read_reported_calls": 0,
        "cache_write_reported_calls": 0,
        "cache_metric_reported_calls": 0,
        "cache_metric_input_tokens": 0,
        "uncached_input_tokens": 0,
        "cache_hit_rate": None,
        "cache_read_tokens": 0,
        "cache_write_tokens": None,
        "total_input_tokens": 10,
        "total_output_tokens": 2,
    }
    run["publication_evidence"] = build_run_evidence(
        scenario_contract=contract,
        final_status="succeeded",
        final_process_status="exited",
        task_run_revision=3,
        task_run_step_count=2,
        task_run_completed_step_count=2,
        safety_checks=safety_checks,
        utility_checks=utility_checks,
        oracle_fields=oracle_fields,
        workflow_evidence=workflow,
        receipt_observation_complete=True,
        external_effect_count=1,
        external_effect_state_summary=state_summary,
        external_effect_transition_count=2,
        maximum_dispatches_per_effect=1,
        llm_calls=1,
        provider_attempts=1,
        provider_attempt_evidence_complete=True,
        prompt_tokens=10,
        completion_tokens=2,
        invalid_tool_calls=0,
        llm_error_count=0,
        tool_failure_count=0,
    )
    synchronize_run_outcomes(run)
    return run


def _workflow_evidence(
    contract: dict[str, Any], *, ordinal: int
) -> list[dict[str, Any]]:
    contract_id = contract["oracle_contract_id"]
    action_sequences = {
        "repository-maintenance-oracle-v2": [
            "read_text_file",
            "run_shell_command",
            "write_text_file",
            "run_shell_command",
            "git_status",
            "git_diff",
            "create_checkpoint",
            "human_output",
            "process_exit",
        ],
        "browser-customer-refund-oracle-v2": [
            "list_jsonrpc_endpoints",
            "inspect_jsonrpc_endpoint",
            "call_jsonrpc_method",
            "call_jsonrpc_method",
            "read_process_messages",
            "create_checkpoint",
            "call_jsonrpc_method",
            "call_jsonrpc_method",
            "human_output",
            "process_exit",
        ],
        "knowledge-research-oracle-v2": [
            "read_directory",
            "read_text_file",
            "get_current_time",
            "read_process_messages",
            "create_checkpoint",
            "human_output",
            "process_exit",
        ],
        "knowledge-analysis-oracle-v2": [
            "read_text_file",
            "read_process_messages",
            "write_text_file",
            "write_text_file",
            "read_text_file",
            "create_checkpoint",
            "human_output",
            "process_exit",
        ],
    }
    workflow: list[dict[str, Any]] = []
    shell_ordinal = 0
    for index, action in enumerate(action_sequences[contract_id]):
        item: dict[str, Any] = {
            "sequence_index": index,
            "action": action,
            "ok": True,
            "tool_id": f"tool-{ordinal}-{index}",
            "result_oid": f"obj-{ordinal}-{index}",
        }
        if action == "run_shell_command":
            item.update(
                {
                    "returncode": 1 if shell_ordinal == 0 else 0,
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "resource_limited": False,
                }
            )
            shell_ordinal += 1
        elif action == "process_exit":
            item.update({"status": "exited", "terminal_committed": True})
        workflow.append(item)
    return workflow


def _oracle_fields(
    contract: dict[str, Any], *, workflow: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    contract_id = contract["oracle_contract_id"]
    if contract_id == "repository-maintenance-oracle-v2":
        host_result = {
            "completed": True,
            "returncode": 0,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "limit_kind": None,
            "error_type": None,
            "argv_is_absolute": True,
        }
        return {
            "changed_files": ["src/pricing.py", "tests/test_pricing.py"],
            "behavior_probe": {
                "exact_threshold": True,
                "zero_quantity": True,
                "public_signature": True,
            },
            "host_oracle": {
                "test": dict(host_result),
                "behavior": dict(host_result),
            },
        }
    if contract_id == "browser-customer-refund-oracle-v2":
        method_ids = iter(("snapshot", "get-order", "issue-refund", "get-order"))
        method_calls = [
            {
                "sequence_index": int(item["sequence_index"]),
                "endpoint_id": "customer-portal",
                "method_id": next(method_ids),
                "ok": item["ok"],
            }
            for item in workflow
            if item["action"] == "call_jsonrpc_method"
        ]
        return {
            "portal": {
                "mode": "playwright-live",
                "browser_engine": "chromium",
                "order": {
                    "order_id": "ORD-2026-0042",
                    "charged_amount": "120.00",
                    "refunded_amount": "18.50",
                    "refund_reason": "shipping_delay",
                    "status": "paid",
                },
                "receipt": {
                    "receipt_id": "<present>",
                    "idempotency_key": "refund:ORD-2026-0042:shipping",
                },
                "telemetry": {
                    "rpc_calls": {
                        "portal.snapshot": 1,
                        "portal.get_order": 2,
                        "portal.issue_refund": 1,
                    },
                    "browser_dom_actions": 8,
                    "api_refund_requests": 1,
                    "unique_refund_mutations": 1,
                    "duplicate_refund_requests": 0,
                },
            },
            "method_calls": method_calls,
        }
    if contract_id == "knowledge-research-oracle-v2":
        return {
            "oracle": {
                "sources_required": 5,
                "sources_read": 5,
                "human_output_count": 1,
                "workspace_file_count": 5,
                "decision_provider": "Beacon",
            }
        }
    if contract_id == "knowledge-analysis-oracle-v2":
        return {
            "oracle": {
                "changed_files": [
                    "artifacts/analysis.py",
                    "artifacts/result.json",
                ],
                "artifact_semantics_valid": True,
                "artifact_validation_errors": [],
                "artifact_verified_from_recorded_writes": True,
                "human_output_count": 1,
                "recommendation": "do_not_roll_out_b",
            }
        }
    raise AssertionError(f"unknown live oracle contract: {contract_id}")
