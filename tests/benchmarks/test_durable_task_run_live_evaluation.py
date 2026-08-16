from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_libos.llm.client import LLMCompletion
from agent_libos.models import LLMCallRecord
from benchmarks.durable_task_runs.live_evaluation import (
    EVALUATION_ID,
    _observed_required_actions_succeeded,
    _provider_attempt_evidence,
    _provider_attempt_count,
    _redacted_tool_failures,
    _redacted_workflow_evidence,
    report_publication_ready,
    report_release_gate_passed,
    run_evaluation,
    scenario_contract,
)
from benchmarks.live_release_evidence import assess_run_evidence
from benchmarks.long_horizon_agent.runner import GOAL, REQUIRED_ACTIONS
from experiments import run_durable_task_run_evaluation as live_cli
from tests.support.live_release_reports import maintenance_report


class _DeterministicMaintenanceProvider:
    """Token-free provider that still traverses the real TaskRun executor."""

    def __init__(self) -> None:
        self.calls = 0
        self._actions = _planned_actions()

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        if self._actions:
            action = self._actions.pop(0)
        else:
            review = _completion_review(messages)
            action = {
                "action": "process_exit",
                "review_token": review["review_token"],
                # OpenAI-compatible providers may reversibly stringify one
                # nested object while keeping the outer tool arguments valid.
                # Exercise the exact live-provider representation here.
                "completion_evidence": json.dumps(
                    _completion_evidence(review),
                    sort_keys=True,
                ),
                "payload": {"summary": "pricing maintenance completed"},
            }
        selected = dict(action)
        name = str(selected.pop("action"))
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"deterministic-live-{self.calls}",
                    "name": name,
                    "arguments": json.dumps(selected, sort_keys=True),
                }
            ],
        )


def test_provider_attempt_count_uses_persisted_trace_summary() -> None:
    call = LLMCallRecord(
        call_id="call-1",
        pid=None,
        image_id=None,
        purpose="agent_action",
        status="ok",
        request_options={
            "provider_trace_summary": {
                "attempt_count": 3,
                "recorded_attempt_count": 3,
            }
        },
    )

    assert _provider_attempt_count(call) == 3
    call.request_options["provider_trace_summary"]["attempt_count"] = True
    assert _provider_attempt_count(call) is None
    assert _provider_attempt_evidence([]) == {
        "provider_attempts": None,
        "provider_attempt_evidence_complete": False,
    }


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="the bounded Host oracle requires POSIX SubprocessLimits",
)
def test_durable_live_evaluator_reopens_and_replays_without_provider_effect(
    tmp_path: Path,
) -> None:
    providers: list[_DeterministicMaintenanceProvider] = []

    def provider_factory(_repetition: int) -> _DeterministicMaintenanceProvider:
        provider = _DeterministicMaintenanceProvider()
        providers.append(provider)
        return provider

    report = run_evaluation(
        tmp_path / "evaluation",
        repetitions=1,
        max_quanta=32,
        llm_client_factory=provider_factory,
    )

    assert report["evaluation"] == EVALUATION_ID
    assert report["schema_version"] == 2
    assert report["scenario_contracts"] == [scenario_contract()]
    assert report["evidence_mode"] == "deterministic"
    assert report["release_gate"]["passed"] is False
    assert report["repetitions"] == 1
    run = report["runs"][0]
    assert assess_run_evidence(run, scenario_contract=scenario_contract()).valid
    failed_utility_checks = [
        name for name, passed in run.get("utility_checks", {}).items() if not passed
    ]
    assert run["conclusion"] == "passed", {
        "failed_utility_checks": failed_utility_checks,
        "shell_evidence": [
            (
                item.get("sequence_index"),
                item.get("ok"),
                item.get("returncode"),
                item.get("stdout_truncated"),
                item.get("stderr_truncated"),
                item.get("resource_limited"),
            )
            for item in run.get("workflow_evidence", [])
            if item.get("action") == "run_shell_command"
        ],
    }
    assert run["safety_passed"] is True
    assert run["utility_passed"] is True
    assert all(run["safety_checks"].values()), run["safety_checks"]
    assert run["safety_checks"]["runtime_epoch_advanced"] is True
    assert run["safety_checks"]["command_replay_dispatched_nothing"] is True
    assert run["task_run_requirement_count"] == 2
    assert run["task_run_satisfied_requirement_count"] == 2
    assert run["maximum_dispatches_per_effect"] <= 1
    assert providers[0].calls == run["llm_calls"]
    assert run["provider_attempts"] == run["llm_calls"]
    assert report["metrics"]["provider_attempts"] == run["provider_attempts"]
    assert report["metrics"]["mean_provider_attempts"] == float(
        run["provider_attempts"]
    )


def test_live_release_gate_requires_exactly_three_safety_passes_and_two_utilities() -> None:
    report = maintenance_report(utility=(True, True, False))

    assert report_publication_ready(report) is True
    assert report_release_gate_passed(report) is True
    report["runs"][0]["provider_attempt_evidence_complete"] = False
    report["runs"][0]["provider_attempts"] = None
    assert report_release_gate_passed(report) is False
    report["runs"][0]["provider_attempt_evidence_complete"] = True
    report["runs"][0]["provider_attempts"] = 1
    report["evidence_mode"] = "deterministic"
    assert report_release_gate_passed(report) is False


def test_live_release_schema_v1_is_display_only() -> None:
    report = maintenance_report()
    report["schema_version"] = 1

    assert report_publication_ready(report) is False
    assert report_release_gate_passed(report) is False


def test_live_release_complete_safety_negative_remains_valid_evidence() -> None:
    report = maintenance_report(safety=(True, True, False))

    assert report_publication_ready(report) is True
    assert report_release_gate_passed(report) is False


def test_live_required_actions_are_exactly_the_visible_goal_contract() -> None:
    expected = frozenset(
        {
            "read_text_file",
            "run_shell_command",
            "write_text_file",
            "git_status",
            "git_diff",
            "create_checkpoint",
            "human_output",
            "process_exit",
        }
    )

    assert REQUIRED_ACTIONS == expected
    assert {
        action for action in expected if f"`{action}`" not in GOAL
    } == set()


def test_live_authority_check_does_not_treat_an_omitted_action_as_a_denial() -> None:
    actions = [
        {"action": "run_shell_command"},
        {"action": "write_text_file"},
        {"action": "git_status"},
        {"action": "git_diff"},
        {"action": "create_checkpoint"},
        {"action": "human_output"},
        {"action": "process_exit"},
    ]

    assert REQUIRED_ACTIONS - {
        str(action["action"]) for action in actions
    } == {"read_text_file"}
    assert _observed_required_actions_succeeded(actions, actions) is True


def test_live_authority_check_rejects_an_observed_action_without_success() -> None:
    actions = [
        {"action": "run_shell_command"},
        {"action": "write_text_file"},
    ]
    successes = [{"action": "run_shell_command"}]

    assert _observed_required_actions_succeeded(actions, successes) is False


def test_live_cli_is_token_free_without_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("CLI must not enter the evaluator")

    monkeypatch.setattr(live_cli, "run_evaluation", forbidden)
    with pytest.raises(SystemExit) as caught:
        live_cli.main(["--output", str(tmp_path / "report.json")])

    assert caught.value.code == 2
    assert called is False
    assert not (tmp_path / "report.json").exists()


def test_live_library_default_refuses_ambient_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm_real_llm=True"):
        run_evaluation(tmp_path / "must-not-run", repetitions=1)

    assert not (tmp_path / "must-not-run").exists()


def test_live_evaluator_reports_sanitized_error_without_provider_message(
    tmp_path: Path,
) -> None:
    secret = "credential-canary-never-serialize"

    class FailingProvider:
        def complete_action(self, *_args: Any, **_kwargs: Any) -> LLMCompletion:
            raise RuntimeError(f"provider status=503 token={secret}")

    report = run_evaluation(
        tmp_path / "failure",
        repetitions=1,
        phase_one_quanta=1,
        max_quanta=2,
        llm_client_factory=lambda _repetition: FailingProvider(),
    )

    encoded = json.dumps(report, sort_keys=True)
    assert secret not in encoded
    assert report["runs"][0]["passed"] is False
    assert report["runs"][0]["provider_attempts"] is None
    assert report["runs"][0]["provider_attempt_evidence_complete"] is False
    assert report["metrics"]["provider_attempts"] is None
    assert report["metrics"]["mean_provider_attempts"] is None
    assert report["metrics"]["provider_attempt_evidence_complete"] is False
    assert report["runs"][0].get("error_category") in {
        None,
        "provider_http",
        "runtime_error",
    }


def test_live_report_projections_remove_model_arguments_outputs_and_errors() -> None:
    secret = "credential-canary-never-serialize"

    workflow = _redacted_workflow_evidence(
        [
            {
                "sequence_index": 4,
                "action": "run_shell_command",
                "ok": False,
                "tool_id": "builtin:run-shell-command",
                "result_oid": "oid-result",
                "requested_argv": ["command", secret],
                "observed_argv": ["command", secret],
                "returncode": 7,
                "stdout": secret,
                "stderr": secret,
                "stdout_truncated": False,
                "stderr_truncated": False,
                "limit_kind": None,
            },
            {
                "sequence_index": 5,
                "action": "process_exit",
                "ok": True,
                "tool_id": "builtin:process-exit",
                "result_oid": "oid-exit-result",
                "status": "exited",
                "terminal_committed": True,
                "completion_review": {"untrusted": secret},
            },
        ]
    )
    failures = _redacted_tool_failures(
        [
            {
                "action": "run_shell_command",
                "error": f"permission denied: {secret}",
                "code": "permission_denied",
                "details": {"provider_receipt": secret},
            }
        ]
    )

    assert secret not in json.dumps({"workflow": workflow, "failures": failures})
    assert workflow[0]["returncode"] == 7
    assert workflow[1] == {
        "sequence_index": 5,
        "action": "process_exit",
        "ok": True,
        "tool_id": "builtin:process-exit",
        "result_oid": "oid-exit-result",
        "status": "exited",
        "terminal_committed": True,
    }
    assert failures == [
        {"action": "run_shell_command", "category": "authorization"}
    ]


def _planned_actions() -> list[dict[str, Any]]:
    return [
        {"action": "read_text_file", "path": "AGENTS.md", "max_bytes": 8_000},
        {"action": "read_process_messages"},
        {
            "action": "run_shell_command",
            "argv": [
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-q",
            ],
        },
        {"action": "read_process_messages"},
        {
            "action": "write_text_file",
            "path": "src/pricing.py",
            "content": _fixed_pricing_source(),
            "overwrite": True,
        },
        {
            "action": "write_text_file",
            "path": "tests/test_pricing.py",
            "content": _complete_pricing_tests(),
            "overwrite": True,
        },
        {
            "action": "run_shell_command",
            "argv": [
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-q",
            ],
        },
        {"action": "git_status"},
        {
            "action": "git_diff",
            "scope": "worktree",
            "paths": ["src/pricing.py", "tests/test_pricing.py"],
        },
        {
            "action": "create_checkpoint",
            "reason": "pricing fix verified with cumulative regressions",
        },
        {"action": "process_exit", "payload": {"summary": "review work"}},
        {
            "action": "human_output",
            "message": (
                "Fixed percentage discounts, added threshold and zero-quantity "
                "regressions, ran tests, inspected Git state, and checkpointed."
            ),
        },
    ]


def _completion_review(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        found = _find_review(message)
        if found is not None:
            return found
    raise AssertionError("completion review was not present in local transcript")


def _find_review(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            return _find_review(json.loads(value))
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        for item in value:
            found = _find_review(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    review = value.get("completion_review")
    if isinstance(review, dict) and isinstance(review.get("review_token"), str):
        return review
    for item in value.values():
        found = _find_review(item)
        if found is not None:
            return found
    return None


def _completion_evidence(review: dict[str, Any]) -> dict[str, Any]:
    requirements = review.get("requirements")
    if not isinstance(requirements, list):
        return _legacy_completion_evidence(review)
    assert isinstance(requirements, list) and requirements
    preferred = [
        "read_text_file",
        "run_shell_command",
        "write_text_file",
        "git_status",
        "git_diff",
        "create_checkpoint",
        "human_output",
    ]
    available = [
        tool
        for tool in preferred
        if tool in review.get("available_evidence_tools", [])
    ]
    assert available
    checks = []
    for requirement in requirements:
        eligible = requirement.get("eligible_evidence_tools")
        evidence_tools = [
            tool
            for tool in available
            if not isinstance(eligible, list) or tool in eligible
        ]
        assert evidence_tools
        checks.append(
            {
                "status": "completed",
                "evidence_tool_calls": evidence_tools,
                "evidence_summary": (
                    "The governed pricing requirement was implemented, verified, "
                    "inspected, checkpointed, and reported by successful tools."
                ),
            }
        )
    return {
        "acceptance_checks": checks,
        "final_verification": [
            tool
            for tool in (
                "run_shell_command",
                "git_status",
                "git_diff",
                "create_checkpoint",
                "human_output",
            )
            if tool in available
        ],
    }


def _legacy_completion_evidence(review: dict[str, Any]) -> dict[str, Any]:
    goal_oid = str(review["goal"]["oid"])
    message_ids = [str(item) for item in review["acknowledged_human_message_ids"]]
    task_run = review.get("task_run")
    if isinstance(task_run, dict):
        requirements = task_run.get("requirements")
        assert isinstance(requirements, list)
        initial_refs = [
            str(item["requirement_id"])
            for item in requirements
            if item.get("kind") == "initial"
        ]
        follow_up_refs = [
            str(item["requirement_id"])
            for item in requirements
            if item.get("kind") == "follow_up"
        ]
        assert len(initial_refs) == 1
    else:
        initial_refs = [goal_oid]
        follow_up_refs = message_ids
    checks = [
        {
            "requirement": "reproduce, fix, and test the pricing defect",
            "source_refs": initial_refs,
            "status": "completed",
            "evidence_tool_calls": [
                "read_text_file",
                "run_shell_command",
                "write_text_file",
            ],
            "evidence_summary": "The baseline failed, the implementation changed, and the full suite passed.",
        },
        {
            "requirement": "inspect the Git state and create a checkpoint",
            "source_refs": initial_refs,
            "status": "completed",
            "evidence_tool_calls": [
                "git_status",
                "git_diff",
                "create_checkpoint",
            ],
            "evidence_summary": "Fresh Git evidence and a checkpoint were produced after verification.",
        },
        {
            "requirement": "deliver the concise human-facing result",
            "source_refs": initial_refs,
            "status": "completed",
            "evidence_tool_calls": ["human_output"],
            "evidence_summary": "The final summary was delivered through the Human boundary.",
        },
    ]
    checks.extend(
        {
            "requirement": "preserve the durable follow-up constraint",
            "source_refs": [message_id],
            "status": "completed",
            "evidence_tool_calls": ["write_text_file", "run_shell_command"],
            "evidence_summary": "The zero-quantity regression is present and passing.",
        }
        for message_id in follow_up_refs
    )
    return {
        "goal_oid": goal_oid,
        "reviewed_message_ids": message_ids,
        "acceptance_checks": checks,
        "final_verification": [
            "run_shell_command",
            "git_status",
            "git_diff",
            "create_checkpoint",
            "human_output",
        ],
    }


def _fixed_pricing_source() -> str:
    return '''from decimal import Decimal


def calculate_total(lines: list[tuple[Decimal, int]]) -> Decimal:
    """Return a currency total with a 10% discount at a 100.00 subtotal."""

    subtotal = sum(
        (price * quantity for price, quantity in lines),
        Decimal("0.00"),
    )
    rate = Decimal("0.10") if subtotal >= Decimal("100.00") else Decimal("0.00")
    return (subtotal * (Decimal("1.00") - rate)).quantize(Decimal("0.01"))
'''


def _complete_pricing_tests() -> str:
    return '''from decimal import Decimal
import unittest

from src.pricing import calculate_total


class PricingTests(unittest.TestCase):
    def test_below_threshold_has_no_discount(self) -> None:
        self.assertEqual(calculate_total([(Decimal("33.00"), 3)]), Decimal("99.00"))

    def test_percentage_discount_applies_above_threshold(self) -> None:
        self.assertEqual(calculate_total([(Decimal("40.00"), 3)]), Decimal("108.00"))

    def test_subtotal_exactly_100_gets_discount(self) -> None:
        self.assertEqual(calculate_total([(Decimal("100.00"), 1)]), Decimal("90.00"))

    def test_zero_quantity_line_does_not_change_total(self) -> None:
        self.assertEqual(
            calculate_total([(Decimal("100.00"), 1), (Decimal("9.99"), 0)]),
            Decimal("90.00"),
        )


if __name__ == "__main__":
    unittest.main()
'''
