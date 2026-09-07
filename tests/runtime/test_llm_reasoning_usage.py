from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import ResourceBudget, ResourceUsageReservationStatus


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("persist_full_io", [True, False])
@pytest.mark.parametrize("reasoning_tokens", [0, 2, 3])
def test_reasoning_usage_is_persisted_as_output_subset_without_extra_charge(
    api: str,
    persist_full_io: bool,
    reasoning_tokens: int,
) -> None:
    input_key = "input_tokens" if api == "responses" else "prompt_tokens"
    output_key = "output_tokens" if api == "responses" else "completion_tokens"

    class ReasoningClient:
        def complete_action(
            self,
            _messages: list[dict[str, Any]],
            _tools: list[dict[str, Any]],
        ) -> LLMCompletion:
            return LLMCompletion(
                content="",
                api=api,
                model="test-model",
                tool_calls=[{
                    "id": "exit_call",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }],
                usage={
                    input_key: 5,
                    output_key: 2,
                    "total_tokens": 7,
                    f"{output_key}_details": {"reasoning_tokens": reasoning_tokens},
                },
            )

    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            max_tokens=5,
            max_input_tokens_per_call=100_000,
            max_total_tokens_per_call=100_005,
            persist_full_io=persist_full_io,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        runtime.llm.client = ReasoningClient()
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="record reasoning accounting",
            resource_budget=ResourceBudget(max_llm_calls=1, max_llm_total_tokens=100_005),
        )

        result = runtime.run_process_once(pid)

        assert result["ok"]
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_prompt_tokens == 5
        assert process.resource_usage.llm_completion_tokens == 2
        assert process.resource_usage.llm_total_tokens == 7
        reservation = runtime.uow.resources.list_resource_usage_reservations(pid=pid)[0]
        assert reservation.status is ResourceUsageReservationStatus.SETTLED
        assert reservation.settled_usage is not None
        assert reservation.settled_usage.llm_total_tokens == 7
        record = runtime.store.list_llm_calls(pid=pid)[0]
        assert record.usage[input_key] == 5
        assert record.usage[output_key] == 2
        assert record.usage["total_tokens"] == 7
        assert f"{output_key}_details" not in record.usage
        if reasoning_tokens <= 2:
            assert record.usage["reasoning_tokens"] == reasoning_tokens
            assert "invalid_usage_fields" not in record.request_options
        else:
            assert "reasoning_tokens" not in record.usage
            assert record.request_options["invalid_usage_fields"] == ["reasoning_tokens"]
    finally:
        runtime.close()
