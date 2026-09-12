from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import ObjectType, PROMPT_MODE_LIBOS_DEFAULT


def test_working_set_keeps_goal_plan_and_latest_feedback_under_pressure() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="retain the goal")
        goal = runtime.process.get(pid).memory_view.roots[0]
        plan = runtime.memory.create_object(
            pid, ObjectType.PLAN, {"pending": ["verify", "inspect diff", "report"]},
        )
        stale = [
            runtime.memory.create_object(
                pid, ObjectType.TOOL_RESULT,
                {"tool_name": "echo", "result": {"next": "OLD_FEEDBACK"}},
            )
            for _ in range(12)
        ]
        latest = runtime.memory.create_object(
            pid, ObjectType.TOOL_RESULT,
            {"tool_name": "echo", "result": {"next": "LATEST_FEEDBACK"}},
        )
        required = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [goal, plan, latest]),
            budget_tokens=100_000, charge_resources=False,
        )
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [goal, plan, *stale, latest]),
            policy="working_set", budget_tokens=required.token_count,
            charge_resources=False,
        )

        assert context.object_refs == [goal.oid, plan.oid, latest.oid]
        assert set(context.omitted_objects) == {item.oid for item in stale}
        assert context.token_count <= required.token_count
        assert context.text == required.text
    finally:
        runtime.close()


@pytest.mark.parametrize("policy", ["plan_first", "error_debug", "evidence_first"])
def test_priority_policies_prefer_recent_objects_within_the_same_priority(policy: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="recent feedback")
        older, newer = [
            runtime.memory.create_object(
                pid, ObjectType.TOOL_RESULT,
                {"tool_name": "echo", "result": {"step": step}},
            )
            for step in (1, 2)
        ]
        budget = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [newer]),
            charge_resources=False,
        ).token_count
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [older, newer]),
            policy=policy, budget_tokens=budget, charge_resources=False,
        )
        assert context.object_refs == [newer.oid]
        assert context.omitted_objects == [older.oid]
    finally:
        runtime.close()


def test_working_set_preserves_current_constraints_and_feedback_before_old_plans() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="original deliverables")
        goal = runtime.process.get(pid).memory_view.roots[0]
        old_plans = [
            runtime.memory.create_object(pid, ObjectType.PLAN, {"pending": f"old plan {i}"})
            for i in range(8)
        ]
        current_plan = runtime.memory.create_object(
            pid, ObjectType.PLAN, {"pending": "verify the additional regression"},
        )
        constraint = runtime.memory.create_object(
            pid, ObjectType.CONSTRAINT, {"requirement": "zero quantity remains valid"},
        )
        result = runtime.memory.create_object(
            pid, ObjectType.TOOL_RESULT,
            {"tool_name": "run_shell_command", "result": {"returncode": 0}},
        )
        required = [goal, current_plan, constraint, result]
        budget = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, required), charge_resources=False,
        ).token_count
        selected = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [goal, *old_plans, *required[1:]]),
            policy="working_set", budget_tokens=budget, charge_resources=False,
        )
        assert selected.object_refs == [handle.oid for handle in required]
        assert set(selected.omitted_objects) == {handle.oid for handle in old_plans}
    finally:
        runtime.close()


@pytest.mark.parametrize("prompt_layout", ["legacy_v1", "cache_optimized_v2"])
def test_bounded_long_task_observes_each_previous_result_and_cumulative_plan(
    prompt_layout: str,
) -> None:
    """A feedback-dependent 32-turn task must survive a rolling 2k context."""
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, prompt_layout=prompt_layout),
    )
    runtime = Runtime.open("local", config=config)
    try:
        image = AgentImage(
            image_id="long-context-probe:v0",
            name="long-context-probe",
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
            default_tools=["echo", "create_memory_object", "process_exit"],
            context_policy="working_set",
        )
        runtime.register_image(image, actor="test")
        pid = runtime.process.spawn(image=image.image_id, goal="ORIGINAL_GOAL_SENTINEL")
        process = runtime.process.get(pid)
        process.resource_budget = replace(
            process.resource_budget, max_context_materialization_tokens=2_000,
        )
        runtime.store.update_process(process)
        ledger = runtime.llm.dispatch(pid, {
            "action": "create_memory_object",
            "type": "plan",
            "name": "acceptance-ledger",
            "payload": {"pending": ["PRESERVE_PUBLIC_SIGNATURE", "VERIFY_BEFORE_DELIVERY"]},
        })
        assert ledger["ok"], ledger
        observed: list[int] = []

        class FeedbackClient:
            def complete_action(
                self, messages: list[dict[str, object]], tools: list[dict[str, object]],
            ) -> LLMCompletion:
                prompt = str(messages[-1]["content"])
                step = len(observed)
                assert "ORIGINAL_GOAL_SENTINEL" in prompt
                assert "PRESERVE_PUBLIC_SIGNATURE" in prompt
                assert "VERIFY_BEFORE_DELIVERY" in prompt
                if step:
                    assert f"FEEDBACK_{step:04d}" in prompt
                observed.append(step)
                return LLMCompletion(
                    content="",
                    tool_calls=[{
                        "id": f"probe_{step}", "name": "echo",
                        "arguments": json.dumps({
                            "feedback": f"FEEDBACK_{step + 1:04d}",
                            "detail": "bounded diagnostic output " * 40,
                        }),
                    }],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

        runtime.llm.client = FeedbackClient()
        for step in range(32):
            result = runtime.run_process_once(pid)
            assert result["ok"], (step, result)
        assert len(observed) == 32
        assert runtime.process.get(pid).status.value == "runnable"
    finally:
        runtime.close()
