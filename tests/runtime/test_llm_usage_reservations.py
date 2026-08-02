from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
import agent_libos.llm.client as llm_client_module
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.models import (
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    ResourceUsageReservationStatus,
)
from agent_libos.models.exceptions import ResourceLimitExceeded
from agent_libos.substrate import ProviderEffectNotStarted


_MAX_INPUT_TOKENS = 100_000
_MAX_OUTPUT_TOKENS = 5
_MAX_TOTAL_TOKENS = 100_005


def _hard_budget_config():
    return replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            max_tokens=_MAX_OUTPUT_TOKENS,
            max_input_tokens_per_call=_MAX_INPUT_TOKENS,
            max_total_tokens_per_call=_MAX_TOTAL_TOKENS,
        ),
    )


def _exit_completion() -> LLMCompletion:
    return LLMCompletion(
        content="",
        tool_calls=[
            {
                "id": "tool_exit",
                "name": "process_exit",
                "arguments": json.dumps({"payload": {"done": True}}),
            }
        ],
        api="chat",
        model="test-model",
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )


class InspectingClient:
    def __init__(self, runtime: Runtime, pid: str) -> None:
        self.runtime = runtime
        self.pid = pid
        self.calls = 0
        self.active_usage: ResourceUsage | None = None
        self.remaining_calls: int | None = None
        self.remaining_tokens: int | None = None

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        active = [
            reservation
            for reservation in self.runtime.uow.resources.list_resource_usage_reservations(
                pid=self.pid
            )
            if reservation.status is ResourceUsageReservationStatus.ACTIVE
        ]
        assert len(active) == 1
        assert active[0].reason == "llm.request"
        self.active_usage = active[0].usage
        effects = self.runtime.store.list_external_effects(pid=self.pid)
        assert len(effects) == 1
        effect_context = effects[0].provider_metadata["context"]
        assert effect_context["call_id"]
        assert effect_context["profile_id"] == "default"
        assert effect_context["max_input_tokens_per_call"] == _MAX_INPUT_TOKENS
        assert effect_context["max_output_tokens_per_call"] == _MAX_OUTPUT_TOKENS
        assert effect_context["max_total_tokens_per_call"] == _MAX_TOTAL_TOKENS
        assert effect_context["resource_envelope_sha256"]
        remaining = self.runtime.resources.remaining_budget(self.pid)
        self.remaining_calls = remaining.max_llm_calls
        self.remaining_tokens = remaining.max_llm_total_tokens
        return _exit_completion()


class NotStartedClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        raise ProviderEffectNotStarted("provider request was certified not started")


class CancelledClient:
    def __init__(self) -> None:
        self.calls = 0

    async def acomplete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        self.calls += 1
        raise asyncio.CancelledError


class InconsistentUsageClient:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        completion = _exit_completion()
        completion.usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": self.total_tokens,
        }
        return completion


class MalformedUsageClient:
    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        completion = _exit_completion()
        completion.usage = {"total_tokens": "unknown"}
        return completion


@pytest.mark.parametrize(
    ("estimated_input", "max_input", "max_total", "reason"),
    [
        (101, 100, 105, "estimated_input_exceeds_per_call_limit"),
        (100, 100, 104, "estimated_total_exceeds_per_call_limit"),
    ],
)
def test_local_per_call_envelope_denies_without_provider_evidence(
    monkeypatch: pytest.MonkeyPatch,
    estimated_input: int,
    max_input: int,
    max_total: int,
    reason: str,
) -> None:
    monkeypatch.setattr(
        "agent_libos.llm.executor.estimate_request_input_tokens",
        lambda _messages, _tools: estimated_input,
    )
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            max_tokens=_MAX_OUTPUT_TOKENS,
            max_input_tokens_per_call=max_input,
            max_total_tokens_per_call=max_total,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        client = InspectingClient(runtime, "")
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="local envelope denial",
        )
        client.pid = pid

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert result["resource_limit_exceeded"]
        assert client.calls == 0
        assert runtime.store.list_external_effects(pid=pid) == []
        assert runtime.store.list_llm_calls(pid=pid) == []
        assert runtime.uow.resources.list_resource_usage_reservations(pid=pid) == []
        denial = next(
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.budget_admission_denied"
        )
        assert denial.decision["reason"] == reason
    finally:
        runtime.close()


def test_llm_reservation_is_active_before_provider_and_settles_exactly() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="inspect hard LLM reservation",
            resource_budget=ResourceBudget(
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )
        client = InspectingClient(runtime, pid)
        runtime.llm.client = client

        result = runtime.run_process_once(pid)

        assert result["ok"]
        assert client.calls == 1
        assert client.active_usage == ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=_MAX_INPUT_TOKENS,
            llm_completion_tokens=_MAX_OUTPUT_TOKENS,
            llm_total_tokens=_MAX_TOTAL_TOKENS,
        )
        assert client.remaining_calls == 0
        assert client.remaining_tokens == 0
        process = runtime.process.get(pid)
        assert process.status is ProcessStatus.EXITED
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 5
        assert process.resource_usage.llm_completion_tokens == 2
        assert process.resource_usage.llm_total_tokens == 7
        reservations = runtime.uow.resources.list_resource_usage_reservations(pid=pid)
        assert len(reservations) == 1
        assert reservations[0].status is ResourceUsageReservationStatus.SETTLED
        assert reservations[0].settled_usage == ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=5,
            llm_completion_tokens=2,
            llm_total_tokens=7,
        )
    finally:
        runtime.close()


def test_builtin_transport_retry_is_one_logical_child_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RateLimited(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "0"})

    class SequencedChatCompletions:
        def __init__(self) -> None:
            self.items: list[Any] = [
                RateLimited("private rate-limit body"),
                SimpleNamespace(
                    id="chat_retry",
                    _request_id="req_retry",
                    model="gpt-test",
                    usage=SimpleNamespace(
                        prompt_tokens=5,
                        completion_tokens=2,
                        total_tokens=7,
                    ),
                    choices=[
                        SimpleNamespace(
                            finish_reason="tool_calls",
                            message=SimpleNamespace(
                                content="",
                                tool_calls=[
                                    SimpleNamespace(
                                        id="tool_exit",
                                        function=SimpleNamespace(
                                            name="process_exit",
                                            arguments=json.dumps(
                                                {"payload": {"done": True}}
                                            ),
                                        ),
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ]
            self.payloads: list[dict[str, Any]] = []

        async def create(self, **payload: Any) -> Any:
            self.payloads.append(payload)
            item = self.items.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    config = _hard_budget_config()
    completions = SequencedChatCompletions()
    client = LLMClient(
        model="gpt-test",
        api_key="key",
        api_mode="chat",
        max_retries=1,
        defaults=config.llm,
    )
    assert type(client) is LLMClient
    client._async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)
    runtime = Runtime.open("local", config=config)
    try:
        runtime.llm.client = client
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="parent logical budget",
            resource_budget=ResourceBudget(
                max_child_processes=1,
                max_llm_calls=2,
                max_llm_total_tokens=2 * _MAX_TOTAL_TOKENS,
            ),
        )
        child = runtime.process.spawn_child(
            parent,
            goal="child physical retry",
            resource_budget=ResourceBudget(
                max_child_processes=0,
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )

        result = runtime.run_process_once(child)

        assert result["ok"]
        assert len(completions.payloads) == 2
        child_usage = runtime.process.get(child).resource_usage
        parent_usage = runtime.process.get(parent).resource_usage
        assert child_usage.llm_calls == parent_usage.llm_calls == 1
        assert child_usage.llm_prompt_tokens == parent_usage.llm_prompt_tokens == 5
        assert child_usage.llm_completion_tokens == parent_usage.llm_completion_tokens == 2
        assert child_usage.llm_total_tokens == parent_usage.llm_total_tokens == 7

        calls = runtime.store.list_llm_calls(pid=child)
        assert len(calls) == 1
        assert runtime.store.list_llm_calls(pid=parent) == []
        trace = calls[0].reasoning
        assert trace["coverage"] == "complete"
        assert trace["selected_attempt"] == 2
        assert [
            (attempt["kind"], attempt["status"])
            for attempt in trace["attempts"]
        ] == [("initial", "error"), ("transport_retry", "ok")]
        assert calls[0].request_options["provider_trace_summary"]["attempt_count"] == 2

        reservations = runtime.uow.resources.list_resource_usage_reservations(pid=child)
        assert len(reservations) == 1
        assert reservations[0].status is ResourceUsageReservationStatus.SETTLED
        assert reservations[0].settled_usage == ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=5,
            llm_completion_tokens=2,
            llm_total_tokens=7,
        )
    finally:
        runtime.close()


def test_call_budget_denial_rolls_back_effect_and_reservation() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="deny exhausted logical call budget",
            resource_budget=ResourceBudget(
                max_llm_calls=0,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )
        client = InspectingClient(runtime, pid)
        runtime.llm.client = client

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert result["resource_limit_exceeded"]
        assert client.calls == 0
        assert runtime.process.get(pid).resource_usage.llm_calls == 0
        assert runtime.store.list_external_effects(pid=pid) == []
        assert runtime.store.list_llm_calls(pid=pid) == []
        assert runtime.uow.resources.list_resource_usage_reservations(pid=pid) == []
    finally:
        runtime.close()


def test_concurrent_sibling_reservations_cannot_oversell_parent_llm_budget() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    reservation_ids: list[str] = []
    try:
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="shared parent budget",
            resource_budget=ResourceBudget(
                max_tool_calls=None,
                max_child_processes=2,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )
        child_budget = ResourceBudget(
            max_tool_calls=None,
            max_child_processes=0,
            max_llm_total_tokens=None,
        )
        first = runtime.process.spawn_child(
            parent,
            goal="first child",
            resource_budget=child_budget,
        )
        second = runtime.process.spawn_child(
            parent,
            goal="second child",
            resource_budget=child_budget,
        )

        envelope = ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=_MAX_INPUT_TOKENS,
            llm_completion_tokens=_MAX_OUTPUT_TOKENS,
            llm_total_tokens=_MAX_TOTAL_TOKENS,
        )
        barrier = threading.Barrier(2)

        def reserve(pid: str) -> tuple[str, str | None]:
            barrier.wait()
            try:
                reservation_id = runtime.resources.reserve_usage(
                    pid,
                    envelope,
                    source="llm.request",
                    reserved_by=f"effect:test:{pid}",
                )
            except ResourceLimitExceeded:
                return "denied", None
            return "reserved", reservation_id

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(reserve, (first, second)))

        assert sorted(status for status, _ in outcomes) == ["denied", "reserved"]
        reservation_ids = [
            reservation_id
            for _status, reservation_id in outcomes
            if reservation_id is not None
        ]
        assert len(reservation_ids) == 1
        assert (
            sum(
                len(
                    runtime.uow.resources.list_resource_usage_reservations(
                        pid=child,
                        status="active",
                    )
                )
                for child in (first, second)
            )
            == 1
        )

        assert runtime.process.get(first).resource_usage.llm_total_tokens == 0
        assert runtime.process.get(second).resource_usage.llm_total_tokens == 0
        assert runtime.process.get(parent).resource_usage.llm_total_tokens == 0
    finally:
        for reservation_id in reservation_ids:
            runtime.resources.settle_usage_reservation(
                reservation_id,
                release=True,
                source="test.cleanup",
            )
        runtime.close()


def test_finite_child_llm_budget_covers_active_reservation_at_parent() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    reservation_ids: list[str] = []
    try:
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="parent with one allocated child budget",
            resource_budget=ResourceBudget(
                max_tool_calls=None,
                max_child_processes=2,
                max_llm_calls=None,
                max_llm_total_tokens=2 * _MAX_TOTAL_TOKENS,
            ),
        )
        allocated = runtime.process.spawn_child(
            parent,
            goal="child with an allocated LLM budget",
            resource_budget=ResourceBudget(
                max_tool_calls=None,
                max_child_processes=0,
                max_llm_calls=None,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )
        unallocated = runtime.process.spawn_child(
            parent,
            goal="child sharing the parent LLM budget",
            resource_budget=ResourceBudget(
                max_tool_calls=None,
                max_child_processes=0,
                max_llm_calls=None,
                max_llm_total_tokens=None,
            ),
        )
        envelope = ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=_MAX_INPUT_TOKENS,
            llm_completion_tokens=_MAX_OUTPUT_TOKENS,
            llm_total_tokens=_MAX_TOTAL_TOKENS,
        )

        reservation_ids.append(
            runtime.resources.reserve_usage(
                allocated,
                envelope,
                source="llm.request",
                reserved_by="effect:test:allocated",
            )
        )

        assert (
            runtime.resources.remaining_budget(parent).max_llm_total_tokens
            == _MAX_TOTAL_TOKENS
        )
        reservation_ids.append(
            runtime.resources.reserve_usage(
                unallocated,
                envelope,
                source="llm.request",
                reserved_by="effect:test:unallocated",
            )
        )
        assert runtime.resources.remaining_budget(parent).max_llm_total_tokens == 0

        runtime.resources.settle_usage_reservation(
            reservation_ids.pop(0),
            release=True,
            source="test.release_allocated",
        )
        # The idle allocation remains exclusive to the first child, so the
        # second child's active reservation still consumes the other half.
        assert runtime.resources.remaining_budget(parent).max_llm_total_tokens == 0

        client = NotStartedClient()
        runtime.llm.client = client
        result = runtime.run_process_once(unallocated)

        denial = next(
            record
            for record in runtime.audit.trace(actor=unallocated)
            if record.action == "llm.budget_admission_denied"
        )
        assert not result["ok"]
        assert result["resource_limit_exceeded"]
        assert client.calls == 0
        assert denial.decision["reason"] == "resource_envelope_unavailable"
        assert len(
            runtime.uow.resources.list_resource_usage_reservations(
                pid=unallocated,
                status="active",
            )
        ) == 1
    finally:
        for reservation_id in reservation_ids:
            runtime.resources.settle_usage_reservation(
                reservation_id,
                release=True,
                source="test.cleanup",
            )
        runtime.close()


def test_provider_not_started_certificate_releases_llm_reservation() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        client = NotStartedClient()
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="provider does not start",
            resource_budget=ResourceBudget(
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert client.calls == 1
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_calls == 0
        assert process.resource_usage.llm_total_tokens == 0
        reservations = runtime.uow.resources.list_resource_usage_reservations(pid=pid)
        assert len(reservations) == 1
        assert reservations[0].status is ResourceUsageReservationStatus.RELEASED
        assert reservations[0].settled_usage == ResourceUsage()
    finally:
        runtime.close()


def test_async_provider_cancellation_charges_unknown_llm_maximum() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        client = CancelledClient()
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="cancel provider request",
            resource_budget=ResourceBudget(
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )

        with pytest.raises(BaseException) as exc_info:
            runtime.run_process_once(pid)

        assert type(exc_info.value).__name__ == "_QuantumCancelled"
        assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)
        assert client.calls == 1
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert process.resource_usage.llm_total_tokens == _MAX_TOTAL_TOKENS
        reservations = runtime.uow.resources.list_resource_usage_reservations(pid=pid)
        assert len(reservations) == 1
        assert reservations[0].status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
    finally:
        runtime.close()


@pytest.mark.parametrize("reported_total", [6, 8])
def test_inconsistent_provider_usage_charges_maximum_before_tool_dispatch(
    reported_total: int,
) -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        runtime.llm.client = InconsistentUsageClient(reported_total)
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject inconsistent usage",
            resource_budget=ResourceBudget(
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert result["resource_limit_exceeded"]
        assert "does not equal" in result["error"]
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert process.resource_usage.llm_total_tokens == _MAX_TOTAL_TOKENS
        assert not any(
            record.action == "process.exit" and record.actor == pid
            for record in runtime.audit.trace()
        )
        reservation = runtime.uow.resources.list_resource_usage_reservations(
            pid=pid
        )[0]
        assert reservation.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
    finally:
        runtime.close()


def test_malformed_usage_is_not_treated_as_compatible_missing_usage() -> None:
    runtime = Runtime.open("local", config=_hard_budget_config())
    try:
        runtime.llm.client = MalformedUsageClient()
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject malformed usage without a cumulative token budget",
        )

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert result["resource_limit_exceeded"]
        assert "without a valid compatible counter" in result["error"]
        process = runtime.process.get(pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert process.resource_usage.llm_total_tokens == _MAX_TOTAL_TOKENS
        assert not any(
            record.action == "process.exit" and record.actor == pid
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_committed_llm_effect_with_interrupted_settlement_recovers_maximum(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedCrash(BaseException):
        pass

    database = tmp_path / "llm-budget-recovery.sqlite"
    runtime = Runtime.open(database, config=_hard_budget_config())
    client = InspectingClient(runtime, "")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="crash after provider commit",
            resource_budget=ResourceBudget(
                max_llm_calls=1,
                max_llm_total_tokens=_MAX_TOTAL_TOKENS,
            ),
        )
        client.pid = pid
        runtime.llm.client = client

        def interrupt_settlement(*_args: Any, **_kwargs: Any) -> None:
            raise SimulatedCrash("settlement interrupted after effect commit")

        monkeypatch.setattr(
            runtime.resources,
            "settle_usage_reservation",
            interrupt_settlement,
        )
        with pytest.raises(SimulatedCrash):
            runtime.run_process_once(pid)

        pending = runtime.uow.resources.list_resource_usage_reservations(pid=pid)
        assert len(pending) == 1
        assert pending[0].status is ResourceUsageReservationStatus.ACTIVE
        assert client.calls == 1
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_hard_budget_config())
    try:
        recovered = reopened.uow.resources.list_resource_usage_reservations(pid=pid)
        assert len(recovered) == 1
        assert recovered[0].status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
        assert recovered[0].settled_usage == ResourceUsage(
            llm_calls=1,
            llm_total_tokens=_MAX_TOTAL_TOKENS,
        )
        process = reopened.process.get(pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert process.resource_usage.llm_total_tokens == _MAX_TOTAL_TOKENS
    finally:
        reopened.close()

    second_reopen = Runtime.open(database, config=_hard_budget_config())
    try:
        process = second_reopen.process.get(pid)
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_total_tokens == _MAX_TOTAL_TOKENS
        assert client.calls == 1
    finally:
        second_reopen.close()
