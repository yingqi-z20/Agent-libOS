from __future__ import annotations

from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.llm.context_management import ContextPressureAssessment
from agent_libos.models import ResourceBudget
from agent_libos.storage import SQLiteStore
from tests.support.fakes import RecordingActionClient


def _forced_pressure(**kwargs: Any) -> ContextPressureAssessment:
    window = int(kwargs["context_window_tokens"])
    reserved = int(kwargs["reserved_output_tokens"])
    estimated = max(1, window - reserved)
    return ContextPressureAssessment(
        context_window_tokens=window,
        local_input_estimate_tokens=estimated,
        provider_usage_lower_bound_tokens=0,
        estimated_input_tokens=estimated,
        reserved_output_tokens=reserved,
        projected_tokens=estimated + reserved,
        utilization_ratio=(estimated + reserved) / window,
        threshold_ratio=float(kwargs["threshold_ratio"]),
        triggered=True,
        profile_id=str(kwargs["profile_id"]),
        context_generation=str(kwargs["context_generation"]),
    )


def test_automatic_context_management_does_not_expand_process_tool_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime(SQLiteStore(":memory:"))
    image = AgentImage(
        image_id="authority-context:v0",
        name="authority-context",
        default_tools=["process_exit"],
        planner={
            "context_management": {
                "mode": "auto_compact",
                "tool": {"name": "ungranted_context_compactor", "arguments": {}},
            }
        },
    )
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(image=image.image_id, goal="finish safely")
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        before = runtime.process.get(pid)
        assert "ungranted_context_compactor" not in before.tool_table

        result = runtime.run_next_process_once()

        assert result["ok"] is True
        after = runtime.process.get(pid)
        assert "ungranted_context_compactor" not in after.tool_table
        assert len(client.user_prompts) == 1
        records = runtime.audit.trace(actor=pid)
        failure = next(
            record
            for record in records
            if record.action == "llm.context_pressure_failed"
        )
        assert failure.decision["reason"] == "tool_failed"
    finally:
        runtime.close()


def test_automatic_context_management_cannot_bypass_tool_argument_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime(SQLiteStore(":memory:"))
    image = AgentImage(
        image_id="schema-context:v0",
        name="schema-context",
        default_tools=["compact_process_context", "process_exit"],
        planner={
            "context_management": {
                "tool": {
                    "name": "compact_process_context",
                    "arguments": {"target_tokens": "not-an-integer"},
                }
            }
        },
    )
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(image=image.image_id, goal="respect schema")
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert len(client.user_prompts) == 1
        assert runtime.store.get_llm_context_generation(pid) == "initial"
        failures = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.context_pressure_failed"
        ]
        assert failures, [
            (record.action, record.decision)
            for record in runtime.audit.trace(actor=pid)
        ]
        assert failures[-1].decision["reason"] in {
            "tool_failed",
            "ValidationError",
            "PydanticValidationError",
        }
    finally:
        runtime.close()


def test_automatic_context_management_cannot_bypass_capability_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime(SQLiteStore(":memory:"))
    image = AgentImage(
        image_id="capability-context:v0",
        name="capability-context",
        default_tools=["read_text_file", "process_exit"],
        planner={
            "context_management": {
                "tool": {
                    "name": "read_text_file",
                    "arguments": {"path": "ungranted.txt"},
                }
            }
        },
    )
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(image=image.image_id, goal="respect capability")
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert len(client.user_prompts) == 1
        assert any(
            record.action == "llm.context_pressure_failed"
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


def test_automatic_context_management_cannot_bypass_resource_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime(SQLiteStore(":memory:"))
    image = AgentImage(
        image_id="budget-context:v0",
        name="budget-context",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="respect resource budget",
        resource_budget=ResourceBudget(max_tool_calls=0),
    )
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        runtime.run_next_process_once()

        assert len(client.user_prompts) == 1
        assert runtime.process.get(pid).resource_usage.tool_calls == 0
        assert any(
            record.action == "llm.context_pressure_failed"
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()
