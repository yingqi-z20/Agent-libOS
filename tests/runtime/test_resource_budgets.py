from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import ProcessStatus, ResourceBudget


class TestResourceBudgets:
    def test_tool_call_budget_is_consumed_and_denies_next_tool_before_execution(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="one tool",
                resource_budget=ResourceBudget(max_tool_calls=1),
            )
            runtime.tools.configure_process_tools(pid, ["get_working_directory"], assigned_by="test")

            first = runtime.tools.call(pid, "get_working_directory", {})
            second = runtime.tools.call(pid, "get_working_directory", {})

            assert first.ok, first.error
            assert not second.ok
            assert "max_tool_calls" in (second.error or "")
            assert runtime.process.get(pid).resource_usage.tool_calls == 1
        finally:
            runtime.close()

    def test_parallel_tool_batch_budget_denial_leaves_no_partial_tool_execution(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True, action_repair_attempts=1),
        )
        runtime = Runtime.open("local", config=config)
        try:
            runtime.llm.client = ParallelBudgetClient(
                [
                    {"action": "create_memory_object", "type": "observation", "payload": {"should_not_run": True}},
                    {"action": "process_exit", "payload": {"done": True}},
                ]
            )
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="budget batch",
                resource_budget=ResourceBudget(max_tool_calls=1),
            )
            runtime.skills.activate_skill(pid, "agent-libos-object-memory", actor=pid)

            result = runtime.run_next_process_once()
            process = runtime.process.get(pid)

            assert not result["ok"]
            assert "parallel tool call batch exceeds remaining tool-call budget" in result["error"]
            assert process.status == ProcessStatus.FAILED
            assert process.resource_usage.tool_calls == 0
            assert not any(record.action == "tool.call" for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_llm_token_overage_kills_process_before_dispatching_tool(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.llm.client = UsageClient(total_tokens=11)
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="exit but over budget",
                resource_budget=ResourceBudget(max_llm_total_tokens=10),
            )

            result = runtime.run_next_process_once()
            process = runtime.process.get(pid)

            assert not result["ok"]
            assert result["resource_limit_exceeded"]
            assert process.status == ProcessStatus.KILLED
            assert process.resource_usage.llm_total_tokens == 11
            assert not any(record.action == "process.exit" and record.actor == pid for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_llm_missing_usage_with_token_budget_kills_process(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.llm.client = UsageClient(total_tokens=None)
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="missing usage",
                resource_budget=ResourceBudget(max_llm_total_tokens=10),
            )

            result = runtime.run_next_process_once()
            process = runtime.process.get(pid)

            assert not result["ok"]
            assert result["resource_limit_exceeded"]
            assert process.status == ProcessStatus.KILLED
            assert process.resource_usage.llm_calls == 1
            assert process.resource_usage.llm_total_tokens == 0
        finally:
            runtime.close()

    def test_llm_malformed_usage_cannot_bypass_token_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.llm.client = MalformedUsageClient()
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="malformed usage",
                resource_budget=ResourceBudget(max_llm_total_tokens=10),
            )

            result = runtime.run_next_process_once()
            process = runtime.process.get(pid)

            assert not result["ok"]
            assert result["resource_limit_exceeded"]
            assert "invalid total_tokens" in result["error"]
            assert process.status == ProcessStatus.KILLED
            assert process.resource_usage.llm_calls == 1
            assert process.resource_usage.llm_total_tokens == 0
            assert not any(record.action == "process.exit" and record.actor == pid for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_child_llm_usage_counts_against_parent_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.llm.client = UsageClient(total_tokens=7)
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_llm_total_tokens=20),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="child",
                resource_budget=ResourceBudget(max_llm_total_tokens=10),
            )

            runtime.run_process_once(child)

            assert runtime.process.get(child).resource_usage.llm_total_tokens == 7
            assert runtime.process.get(parent).resource_usage.llm_total_tokens == 7
        finally:
            runtime.close()

    def test_failed_llm_provider_call_consumes_call_budget_and_persists_sanitized_error_record(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False))
        runtime = Runtime.open("local", config=config)
        try:
            runtime.llm.client = FailingClient()
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="provider fails with secret-token",
                resource_budget=ResourceBudget(max_llm_calls=1),
            )

            result = runtime.run_next_process_once()
            process = runtime.process.get(pid)
            calls = runtime.store.list_llm_calls(pid)

            assert not result["ok"]
            assert process.status == ProcessStatus.FAILED
            assert process.resource_usage.llm_calls == 1
            assert len(calls) == 1
            assert calls[0].status == "error"
            retention_key = "$agent_libos_payload_retention"
            assert calls[0].messages[retention_key]["sha256"]
            assert json.loads(calls[0].error or "{}")[retention_key]["sha256"]
            serialized = json.dumps(calls[0].__dict__, sort_keys=True)
            assert "secret-token" not in serialized
            assert "PROVIDER_ERROR_SECRET" not in serialized
            assert "provider unavailable" not in serialized
            assert "preview" not in serialized

            assert "PROVIDER_ERROR_SECRET" not in result["error"]
            assert process.outcome is not None
            assert process.outcome.result_oid is not None
            result_object = runtime.store.get_object(process.outcome.result_oid)
            assert result_object is not None
            durable_sinks = json.dumps(
                {
                    "audit": [record.__dict__ for record in runtime.audit.trace()],
                    "events": [event.__dict__ for event in runtime.store.list_events()],
                    "result": result_object.__dict__,
                },
                sort_keys=True,
                default=str,
            )
            assert "PROVIDER_ERROR_SECRET" not in durable_sinks
            assert "provider unavailable" not in durable_sinks
            assert "sha256=" in str(result_object.payload.get("message"))
        finally:
            runtime.close()


class UsageClient:
    def __init__(self, total_tokens: int | None) -> None:
        self.total_tokens = total_tokens

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        usage = {} if self.total_tokens is None else {"prompt_tokens": 5, "completion_tokens": self.total_tokens - 5, "total_tokens": self.total_tokens}
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "tool_1",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }
            ],
            raw=SimpleNamespace(id="raw"),
            api="chat",
            response_id="resp_1",
            request_id="req_1",
            model="test-model",
            usage=usage,
        )


class MalformedUsageClient:
    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "tool_malformed_usage",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": "unknown"},
        )


class ParallelBudgetClient:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = actions

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        tool_calls = []
        for index, action in enumerate(self.actions, start=1):
            name = str(action["action"])
            args = {key: value for key, value in action.items() if key != "action"}
            tool_calls.append({"id": f"budget_{index}", "name": name, "arguments": json.dumps(args)})
        return LLMCompletion(content="", tool_calls=tool_calls)


class FailingClient:
    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        raise RuntimeError("provider unavailable: PROVIDER_ERROR_SECRET")
