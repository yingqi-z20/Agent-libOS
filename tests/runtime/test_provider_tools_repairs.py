from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.utils.serde import dumps
from tests.runtime.test_provider_tools_executor import (
    _RESULT,
    _Responses,
    _capture,
    _config,
    _spawn,
)


class _HostedAndInvalidResponses(_Responses):
    async def create(self, **request: Any) -> SimpleNamespace:
        response = await super().create(**request)
        if len(self.requests) == 1:
            response.output.append({
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "missing_runtime_tool",
                "arguments": '{"message":"invalid local action"}', "status": "completed",
            })
        return response


@pytest.mark.parametrize(
    ("first_step", "code"),
    [("hosted", False), ("activity_only", True)],
    ids=["search-and-text", "code-logs-only"],
)
def test_hosted_result_survives_two_local_action_repairs(first_step: str, code: bool) -> None:
    config = _config(code=code)
    config = replace(config, llm=replace(config.llm, action_repair_attempts=3))
    runtime = Runtime.open("local", config=config)
    try:
        provider = _HostedAndInvalidResponses(first_step, "invalid_action", "action", code=code)
        _capture(runtime, provider)
        pid = _spawn(runtime)

        outcome = runtime.run_process_once(pid)

        assert outcome["ok"] and outcome["action"]["action"] == "echo", outcome
        assert not outcome.get("provider_continuation")
        assert len(provider.requests) == 3
        hosted_type = "code_interpreter" if code else "web_search"
        assert any(tool["type"] == hosted_type for tool in provider.requests[0]["tools"])
        for request in provider.requests[1:]:
            assert request["tools"]
            assert all(tool["type"] == "function" for tool in request["tools"])
            assert request.get("extra_body", {}).get("enable_search") is not True
            assert _RESULT in dumps(request["input"])
            if code:
                assert request.get("previous_response_id") is None
                assert not any(item.get("type") == "code_interpreter_call" for item in request["input"])
        repairs = [row for row in runtime.audit.trace(actor=pid) if row.action == "llm.action_repair_requested"]
        assert len(repairs) == 2
    finally:
        runtime.close()
