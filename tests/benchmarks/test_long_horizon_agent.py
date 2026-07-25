from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.llm.client import LLMCompletion
from agent_libos.substrate import LocalResourceProviderSubstrate
from benchmarks.long_horizon_agent import (
    evaluate_run,
    prepare_workspace,
    report_all_successful,
    run_evaluation,
)
from benchmarks.long_horizon_agent.runner import (
    REQUIRED_ACTIONS,
    REQUIRED_SKILLS,
    GOAL,
    MIDFLIGHT_MESSAGE,
    _action_sequence,
    _adjacent_prompt_prefix_metrics,
    _grant_authority,
    _successful_action_sequence,
    _workflow_order_checks,
)


def test_fixture_starts_with_a_real_failure_and_clean_git_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)

    before = evaluate_run(
        workspace,
        status="runnable",
        actions=[],
        activated_skills=[],
        checkpoint_count=0,
        restart_survived=False,
    )

    assert before["passed"] is False
    assert before["checks"]["full_tests_pass"] is False
    assert before["changed_files"] == []


def test_evaluation_authority_allows_required_git_diff_and_checkpoint(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    runtime = Runtime.open(
        tmp_path / "runtime.sqlite",
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="inspect the diff and create a checkpoint",
        )
        _grant_authority(runtime, pid)
        runtime.activate_skill(pid, "agent-libos-git-inspection")
        runtime.activate_skill(pid, "agent-libos-checkpoints")

        status = runtime.tools.call(pid, "git_status", {})
        diff = runtime.tools.call(pid, "git_diff", {"scope": "worktree"})
        checkpoint = runtime.tools.call(
            pid,
            "create_checkpoint",
            {"reason": "long-horizon authority fixture"},
        )

        assert status.ok is True
        assert diff.ok is True
        assert checkpoint.ok is True
    finally:
        runtime.close()


def test_deterministic_long_horizon_task_survives_restart_and_completion_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    database = tmp_path / "runtime.sqlite"
    prepare_workspace(workspace)
    substrate = LocalResourceProviderSubstrate(workspace)
    results: list[dict[str, Any]] = []

    runtime = Runtime.open(database, substrate=substrate)
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal=GOAL)
        _grant_authority(runtime, pid)
        runtime.skills.activate_skill(
            pid,
            "agent-libos-workspace-navigation",
            actor=pid,
        )
        runtime.llm.client = _SingleActionClient(
            "read_text_file",
            {"path": "AGENTS.md", "max_bytes": 8_000},
        )
        results.append(runtime.run_process_once(pid))
        activate_action = {
            "action": "activate_skill",
            "skill_id": "agent-libos-command-execution",
        }
        activate_result = runtime.llm.dispatch(pid, activate_action)
        results.append(
            {"ok": True, "action": activate_action, "result": activate_result}
        )
        run_action = {
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
        }
        run_result = runtime.llm.dispatch(pid, run_action)
        results.append({"ok": True, "action": run_action, "result": run_result})
        runtime.human.send_process_message(
            pid,
            MIDFLIGHT_MESSAGE,
            subject="Customer follow-up",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, substrate=substrate)
    try:
        def dispatch(action: dict[str, Any]) -> dict[str, Any]:
            result = reopened.llm.dispatch(pid, action)
            results.append({"ok": True, "action": action, "result": result})
            assert result["ok"] is True, result
            return result

        dispatch(
            {
                "action": "activate_skill",
                "skill_id": "agent-libos-child-processes",
            }
        )
        message_result = dispatch({"action": "read_process_messages"})
        message_id = message_result["payload"]["messages"][0]["message_id"]
        dispatch(
            {
                "action": "activate_skill",
                "skill_id": "agent-libos-workspace-editing",
            }
        )
        dispatch(
            {
                "action": "write_text_file",
                "path": "src/pricing.py",
                "content": _fixed_pricing_source(),
                "overwrite": True,
            }
        )
        dispatch(
            {
                "action": "write_text_file",
                "path": "tests/test_pricing.py",
                "content": _complete_pricing_tests(),
                "overwrite": True,
            }
        )
        dispatch(
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
            }
        )
        dispatch(
            {
                "action": "activate_skill",
                "skill_id": "agent-libos-git-inspection",
            }
        )
        dispatch({"action": "git_status"})
        dispatch(
            {
                "action": "git_diff",
                "scope": "worktree",
                "paths": ["src/pricing.py", "tests/test_pricing.py"],
            }
        )
        dispatch(
            {
                "action": "activate_skill",
                "skill_id": "agent-libos-checkpoints",
            }
        )
        dispatch(
            {
                "action": "create_checkpoint",
                "reason": "pricing fix verified with cumulative regressions",
            }
        )
        review_result = dispatch({"action": "process_exit"})
        review = review_result["payload"]["completion_review"]
        assert review["goal"]["source"] == "persisted_initial_llm_context"
        assert "payload" not in review["goal"]
        assert "exactly 100.00" in str(review["goal"]["fallback"])
        dispatch(
            {
                "action": "human_output",
                "message": (
                    "Fixed the percentage discount, added exact-threshold and "
                    "zero-quantity regressions, verified tests and Git diff, "
                    "and created a checkpoint."
                ),
            }
        )
        evidence = {
            "goal_oid": review["goal"]["oid"],
            "reviewed_message_ids": [message_id],
            "acceptance_checks": [
                {
                    "requirement": "reproduce and fix the percentage defect",
                    "source_refs": [review["goal"]["oid"]],
                    "status": "completed",
                    "evidence_tool_calls": [
                        "run_shell_command",
                        "write_text_file",
                    ],
                    "evidence_summary": "The failing suite was reproduced, code was edited, and the suite passed.",
                },
                {
                    "requirement": "cover the exact 100.00 threshold",
                    "source_refs": [review["goal"]["oid"]],
                    "status": "completed",
                    "evidence_tool_calls": [
                        "write_text_file",
                        "run_shell_command",
                    ],
                    "evidence_summary": "A named exact-threshold regression passes.",
                },
                {
                    "requirement": "preserve zero-quantity behavior",
                    "source_refs": [message_id],
                    "status": "completed",
                    "evidence_tool_calls": [
                        "write_text_file",
                        "run_shell_command",
                    ],
                    "evidence_summary": "The acknowledged follow-up has a passing regression.",
                },
                {
                    "requirement": "inspect Git state and diff and create a checkpoint",
                    "source_refs": [review["goal"]["oid"]],
                    "status": "completed",
                    "evidence_tool_calls": [
                        "git_status",
                        "git_diff",
                        "create_checkpoint",
                    ],
                    "evidence_summary": "Both Git reads and checkpoint creation succeeded.",
                },
            ],
            "final_verification": [
                "run_shell_command",
                "git_status",
                "git_diff",
                "create_checkpoint",
                "human_output",
            ],
        }
        dispatch(
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": evidence,
                "payload": {"summary": "pricing maintenance completed"},
            }
        )

        process = reopened.process.get(pid)
        actions = _action_sequence(results)
        successful_actions = _successful_action_sequence(results)
        checkpoints = reopened.checkpoint.list(
            pid,
            actor=pid,
            require_capability=False,
        )
        activated_skills = [
            str(action.get("skill_id") or "")
            for action in successful_actions
            if action.get("action") == "activate_skill"
        ]
        oracle = evaluate_run(
            workspace,
            status=process.status.value,
            actions=actions,
            successful_actions=successful_actions,
            activated_skills=activated_skills,
            checkpoint_count=len(checkpoints),
            restart_survived=True,
        )

        assert oracle["passed"] is True, oracle
    finally:
        reopened.close()


def test_oracle_requires_durable_workflow_evidence_not_only_passing_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    workspace.joinpath("src", "pricing.py").write_text(
        """from decimal import Decimal


def calculate_total(lines: list[tuple[Decimal, int]]) -> Decimal:
    subtotal = sum((price * quantity for price, quantity in lines), Decimal(\"0.00\"))
    rate = Decimal(\"0.10\") if subtotal >= Decimal(\"100.00\") else Decimal(\"0.00\")
    return (subtotal * (Decimal(\"1.00\") - rate)).quantize(Decimal(\"0.01\"))
""",
        encoding="utf-8",
    )
    with workspace.joinpath("tests", "test_pricing.py").open("a", encoding="utf-8") as handle:
        handle.write(
            """

class FollowUpPricingTests(unittest.TestCase):
    def test_exact_threshold(self) -> None:
        self.assertEqual(calculate_total([(Decimal(\"100.00\"), 1)]), Decimal(\"90.00\"))

    def test_zero_quantity(self) -> None:
        self.assertEqual(calculate_total([(Decimal(\"100.00\"), 1), (Decimal(\"9.99\"), 0)]), Decimal(\"90.00\"))
"""
        )
    actions = _ordered_workflow_actions()

    missing_evidence = evaluate_run(
        workspace,
        status="exited",
        actions=actions,
        activated_skills=[],
        checkpoint_count=1,
        restart_survived=True,
    )
    complete = evaluate_run(
        workspace,
        status="exited",
        actions=actions,
        activated_skills=sorted(REQUIRED_SKILLS),
        checkpoint_count=1,
        restart_survived=True,
    )

    assert missing_evidence["checks"]["full_tests_pass"] is True
    assert missing_evidence["passed"] is False
    assert complete["passed"] is True


def test_oracle_rejects_a_requested_but_failed_required_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    workspace.joinpath("src", "pricing.py").write_text(
        """from decimal import Decimal


def calculate_total(lines: list[tuple[Decimal, int]]) -> Decimal:
    subtotal = sum((price * quantity for price, quantity in lines), Decimal(\"0.00\"))
    rate = Decimal(\"0.10\") if subtotal >= Decimal(\"100.00\") else Decimal(\"0.00\")
    return (subtotal * (Decimal(\"1.00\") - rate)).quantize(Decimal(\"0.01\"))
""",
        encoding="utf-8",
    )
    with workspace.joinpath("tests", "test_pricing.py").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            """

class FollowUpPricingTests(unittest.TestCase):
    def test_exact_threshold(self) -> None:
        self.assertEqual(calculate_total([(Decimal(\"100.00\"), 1)]), Decimal(\"90.00\"))

    def test_zero_quantity(self) -> None:
        self.assertEqual(calculate_total([(Decimal(\"100.00\"), 1), (Decimal(\"9.99\"), 0)]), Decimal(\"90.00\"))
"""
        )
    requested = [{"action": name} for name in sorted(REQUIRED_ACTIONS)]
    successful = [
        action for action in requested if action["action"] != "git_diff"
    ]

    result = evaluate_run(
        workspace,
        status="exited",
        actions=requested,
        successful_actions=successful,
        activated_skills=sorted(REQUIRED_SKILLS),
        checkpoint_count=1,
        restart_survived=True,
    )

    assert result["checks"]["required_actions_observed"] is True
    assert result["checks"]["required_actions_successful"] is False
    assert result["passed"] is False


def test_workflow_order_requires_baseline_and_fresh_finalization_evidence() -> None:
    fresh = [action["action"] for action in _ordered_workflow_actions()]
    stale = [
        "run_shell_command",
        "git_diff",
        "write_text_file",
        "run_shell_command",
        "git_status",
        "create_checkpoint",
        "human_output",
        "process_exit",
    ]

    assert _workflow_order_checks(fresh) == {
        "baseline_reproduced_before_edit": True,
        "finalization_evidence_fresh": True,
    }
    assert _workflow_order_checks(stale) == {
        "baseline_reproduced_before_edit": True,
        "finalization_evidence_fresh": False,
    }


def test_oracle_requires_executable_regressions_not_comment_markers(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace)
    workspace.joinpath("src", "pricing.py").write_text(
        _fixed_pricing_source(),
        encoding="utf-8",
    )
    with workspace.joinpath("tests", "test_pricing.py").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n# test_exact_threshold_regression\n"
            "# test_zero_quantity_regression\n"
        )
    actions = [{"action": name} for name in sorted(REQUIRED_ACTIONS)]

    result = evaluate_run(
        workspace,
        status="exited",
        actions=actions,
        activated_skills=sorted(REQUIRED_SKILLS),
        checkpoint_count=1,
        restart_survived=True,
    )

    assert result["checks"]["full_tests_pass"] is True
    assert result["checks"]["exact_threshold_behavior"] is True
    assert result["checks"]["zero_quantity_behavior"] is True
    assert result["checks"]["public_signature_stable"] is True
    assert result["checks"]["exact_threshold_regression"] is False
    assert result["checks"]["zero_quantity_regression"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"runs": []},
        {"runs": [{"passed": False}]},
        {"runs": [{"passed": True}, {"passed": False}]},
    ],
)
def test_report_gate_rejects_missing_or_failed_runs(report: dict[str, object]) -> None:
    assert report_all_successful(report) is False


def test_report_gate_accepts_only_all_successful_runs() -> None:
    assert report_all_successful({"runs": [{"passed": True}]}) is True


def test_adjacent_prompt_prefix_metrics_use_persisted_list_messages() -> None:
    first_messages = [{"role": "user", "content": "stable"}]
    second_messages = [
        {"role": "user", "content": "stable"},
        {"role": "assistant", "content": "next"},
    ]
    first_bytes = json.dumps(
        first_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_bytes = json.dumps(
        second_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    metrics = _adjacent_prompt_prefix_metrics(
        [
            SimpleNamespace(messages=first_messages),
            SimpleNamespace(messages=second_messages),
        ]
    )

    expected_prefix_bytes = len(first_bytes) - 1
    assert metrics == {
        "adjacent_prompt_pair_count": 1,
        "adjacent_prompt_comparable_pair_count": 1,
        "adjacent_prompt_unavailable_pair_count": 0,
        "adjacent_prompt_common_prefix_bytes": expected_prefix_bytes,
        "adjacent_prompt_next_bytes": len(second_bytes),
        "adjacent_prompt_common_prefix_ratio": (
            expected_prefix_bytes / len(second_bytes)
        ),
    }


@pytest.mark.parametrize(
    "calls, expected_pair_count",
    [
        ([], 0),
        ([SimpleNamespace(messages=[])], 0),
        (
            [
                SimpleNamespace(messages=[]),
                SimpleNamespace(
                    messages={
                        "$agent_libos_payload_retention": {
                            "tier": "summary",
                            "sha256": "a" * 64,
                        }
                    }
                ),
            ],
            1,
        ),
        (
            [
                SimpleNamespace(messages="not-a-list"),
                SimpleNamespace(messages=[]),
            ],
            1,
        ),
    ],
)
def test_adjacent_prompt_prefix_metrics_report_unavailable_pairs(
    calls: list[SimpleNamespace],
    expected_pair_count: int,
) -> None:
    metrics = _adjacent_prompt_prefix_metrics(calls)

    assert metrics["adjacent_prompt_pair_count"] == expected_pair_count
    assert metrics["adjacent_prompt_comparable_pair_count"] == 0
    assert metrics["adjacent_prompt_unavailable_pair_count"] == expected_pair_count
    assert metrics["adjacent_prompt_common_prefix_bytes"] == 0
    assert metrics["adjacent_prompt_next_bytes"] == 0
    assert metrics["adjacent_prompt_common_prefix_ratio"] is None


def test_report_aggregates_prompt_prefix_metrics_by_next_prompt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = iter(
        [
            {
                "passed": True,
                "checks": {},
                "adjacent_prompt_pair_count": 2,
                "adjacent_prompt_comparable_pair_count": 2,
                "adjacent_prompt_unavailable_pair_count": 0,
                "adjacent_prompt_common_prefix_bytes": 90,
                "adjacent_prompt_next_bytes": 100,
            },
            {
                "passed": False,
                "checks": {},
                "adjacent_prompt_pair_count": 3,
                "adjacent_prompt_comparable_pair_count": 1,
                "adjacent_prompt_unavailable_pair_count": 2,
                "adjacent_prompt_common_prefix_bytes": 10,
                "adjacent_prompt_next_bytes": 100,
            },
        ]
    )
    monkeypatch.setattr(
        "benchmarks.long_horizon_agent.runner._run_once",
        lambda *_args, **_kwargs: next(runs),
    )

    report = run_evaluation(
        tmp_path / "evaluation",
        repetitions=2,
        phase_one_quanta=1,
        max_quanta=2,
    )
    metrics = report["metrics"]

    assert metrics["adjacent_prompt_pair_count"] == 5
    assert metrics["adjacent_prompt_comparable_pair_count"] == 3
    assert metrics["adjacent_prompt_unavailable_pair_count"] == 2
    assert metrics["adjacent_prompt_common_prefix_bytes"] == 100
    assert metrics["adjacent_prompt_next_bytes"] == 200
    assert metrics["adjacent_prompt_common_prefix_ratio"] == 0.5


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

    def test_exact_threshold_regression(self) -> None:
        self.assertEqual(calculate_total([(Decimal("100.00"), 1)]), Decimal("90.00"))

    def test_zero_quantity_regression(self) -> None:
        self.assertEqual(
            calculate_total([(Decimal("100.00"), 1), (Decimal("9.99"), 0)]),
            Decimal("90.00"),
        )


if __name__ == "__main__":
    unittest.main()
'''


def _ordered_workflow_actions() -> list[dict[str, str]]:
    return [
        {"action": name}
        for name in (
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
    ]


class _SingleActionClient:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = dict(arguments)

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "long_horizon_seed",
                    "name": self.name,
                    "arguments": json.dumps(self.arguments),
                }
            ],
            raw=SimpleNamespace(id="long_horizon_seed_raw"),
            api="chat",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
