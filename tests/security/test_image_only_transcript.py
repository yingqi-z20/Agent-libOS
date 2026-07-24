from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent_libos import AgentImage, Runtime
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    ObjectMetadata,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
)


def test_historical_sensitive_tool_output_denies_next_image_only_egress() -> None:
    runtime = Runtime.open("local")
    try:
        runtime.register_image(
            AgentImage(
                image_id="security-transparent-ifc:v0",
                name="security-transparent-ifc",
                system_prompt="Exact security test prompt.",
                prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                default_tools=["read_memory_object", "process_exit"],
                context_policy="recency_first",
            ),
            actor="test",
        )
        client = _OneReadClient()
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="security-transparent-ifc:v0",
            goal="read a classified Object",
        )
        runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"secret": "IMAGE_ONLY_TRANSCRIPT_SECRET"},
            metadata=ObjectMetadata(sensitivity="secret"),
            name="classified",
        )

        first = runtime.run_process_once(pid)
        second = runtime.run_process_once(pid)

        assert first["ok"], first
        assert not second["ok"]
        assert "data-flow denied egress" in second["error"]
        assert client.call_count == 1
        result_oid = first["result"]["result_oid"]
        request = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.request"
        ][-1]
        assert runtime.process.get(pid).goal_oid in request.input_refs
        assert result_oid in request.input_refs
        decisions = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert decisions[-1].labels.sensitivity.value == "secret"
    finally:
        runtime.close()


class _OneReadClient:
    def __init__(self) -> None:
        self.call_count = 0

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del messages, tools
        self.call_count += 1
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "classified_read",
                    "name": "read_memory_object",
                    "arguments": json.dumps({"name": "classified"}),
                }
            ],
            raw=SimpleNamespace(id="security_transcript_response"),
            api="chat",
            response_id="security_transcript_response",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
