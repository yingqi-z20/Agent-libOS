from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMClient
from agent_libos.models import ProcessStatus, ResourceBudget, ResourceUsageReservationStatus


def test_scheduler_logical_deadline_pauses_and_charges_unknown_provider_effect() -> None:
    max_input_tokens = 100_000
    max_output_tokens = 5
    max_total_tokens = max_input_tokens + max_output_tokens
    config = replace(DEFAULT_CONFIG, llm=replace(
        DEFAULT_CONFIG.llm,
        max_tokens=max_output_tokens,
        max_input_tokens_per_call=max_input_tokens,
        max_total_tokens_per_call=max_total_tokens,
        logical_call_timeout_s=0.04,
    ))
    runtime = Runtime.open("local", config=config)
    provider_calls = 0
    provider_thread: int | None = None
    cancelled_thread: int | None = None
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="pause after the Host logical LLM deadline",
            resource_budget=ResourceBudget(
                max_llm_calls=2,
                max_llm_total_tokens=2 * max_total_tokens,
            ),
        )

        async def create(**_payload: Any) -> Any:
            nonlocal provider_calls, provider_thread, cancelled_thread
            provider_calls += 1
            provider_thread = threading.get_ident()
            assert asyncio.get_running_loop().is_running()
            assert runtime.scheduler.is_active_quantum(pid)
            active = runtime.uow.resources.list_resource_usage_reservations(
                pid=pid, status="active",
            )
            assert len(active) == 1 and active[0].reason == "llm.request"
            assert len(runtime.store.list_external_effects(pid=pid)) == 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_thread = threading.get_ident()
                raise

        client = LLMClient(
            model="gpt-test", api_key="test-key", api_mode="chat",
            max_retries=5, timeout=10.0, defaults=config.llm,
            inherit_ambient_openai_sdk_config=False,
        )
        client._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        runtime.llm.client = client

        result = runtime.run_next_process_once()

        assert result["ok"] is False
        assert result["retryable"] is True and result["paused"] is True
        process = runtime.process.get(pid)
        assert process.status is ProcessStatus.PAUSED and process.outcome is None
        assert provider_calls == 1 and provider_thread == cancelled_thread

        call, = runtime.store.list_llm_calls(pid=pid)
        assert call.status == "error" and call.error == result["error"]
        assert call.usage == {}
        assert call.observability["failure"]["internal_error"]["error_type"] == "_LogicalCallTimeoutError"
        assert call.observability["failure"]["public_error"] == result["error_details"]
        trace = call.reasoning
        assert trace["coverage"] == "complete" and trace["selected_attempt"] is None
        attempt, = trace["attempts"]
        assert attempt["kind"] == "initial" and attempt["status"] == "error"
        assert attempt["error"]["error_type"] == "_LogicalCallTimeoutError"
        assert attempt["duration_ms"] > 0
        assert attempt["completed_at"] != attempt["started_at"]
        assert attempt["usage"] == {}
        assert call.request_options["provider_trace_summary"]["attempt_count"] == 1

        # A timed-out remote request may still have incurred usage. Preserve
        # its unknown effect and conservatively charge the reserved envelope.
        effect, = runtime.store.list_external_effects(pid=pid)
        assert effect.transaction_state == "unknown"
        reservation, = runtime.uow.resources.list_resource_usage_reservations(pid=pid)
        assert reservation.status is ResourceUsageReservationStatus.CHARGED_MAXIMUM
        assert process.resource_usage.llm_calls == 1
        assert process.resource_usage.llm_total_tokens == max_total_tokens
        assert process.resource_usage.llm_prompt_tokens == 0
        assert process.resource_usage.llm_completion_tokens == 0
        assert runtime.resources.remaining_budget(pid).max_llm_calls == 1

        audit = runtime.audit.trace(actor=pid)
        failure = next(record for record in audit if record.action == "primitive.llm.complete.failed")
        assert failure.decision["effect_outcome"] == "unknown"
        assert not any(record.action == "primitive.llm.complete" for record in audit)
        assert any(record.action == "llm.action_retryable_failure" for record in audit)
        assert runtime.run_next_process_once() is None
        assert provider_calls == 1 and len(runtime.store.list_llm_calls(pid=pid)) == 1
    finally:
        runtime.close()
