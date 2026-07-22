from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.llm.context_management import (
    DEFAULT_CONTEXT_PRESSURE_PROMPT,
    ContextPressureAssessment,
)
from agent_libos.llm.pending import pending_metadata
from agent_libos.models import (
    CapabilityRight,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_LIBOS_DEFAULT,
    PROMPT_MODE_MINIMAL_RUNTIME,
)
from agent_libos.models.exceptions import ProcessMessageWaitRequired
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.fakes import RecordingActionClient


def _forced_pressure(**kwargs: Any) -> ContextPressureAssessment:
    window = int(kwargs["context_window_tokens"])
    reserved = int(kwargs["reserved_output_tokens"])
    threshold = float(kwargs["threshold_ratio"])
    projected = max(reserved + 1, math.ceil(window * threshold))
    estimated = projected - reserved
    return ContextPressureAssessment(
        context_window_tokens=window,
        local_input_estimate_tokens=estimated,
        provider_usage_lower_bound_tokens=int(
            kwargs.get("provider_lower_bound_tokens", 0)
        ),
        estimated_input_tokens=estimated,
        reserved_output_tokens=reserved,
        projected_tokens=projected,
        utilization_ratio=projected / window,
        threshold_ratio=threshold,
        triggered=True,
        profile_id=str(kwargs["profile_id"]),
        context_generation=str(kwargs["context_generation"]),
    )


def _pressure_assessment(triggered: bool, **kwargs: Any) -> ContextPressureAssessment:
    window = int(kwargs["context_window_tokens"])
    reserved = int(kwargs["reserved_output_tokens"])
    threshold = float(kwargs["threshold_ratio"])
    boundary = math.ceil(window * threshold)
    projected = boundary if triggered else boundary - 1
    estimated = max(1, projected - reserved)
    projected = estimated + reserved
    return ContextPressureAssessment(
        context_window_tokens=window,
        local_input_estimate_tokens=estimated,
        provider_usage_lower_bound_tokens=0,
        estimated_input_tokens=estimated,
        reserved_output_tokens=reserved,
        projected_tokens=projected,
        utilization_ratio=projected / window,
        threshold_ratio=threshold,
        triggered=triggered,
        profile_id=str(kwargs["profile_id"]),
        context_generation=str(kwargs["context_generation"]),
    )


def _runtime_with_image(
    image: AgentImage,
    *,
    actions: list[dict[str, Any]] | None = None,
) -> tuple[Runtime, RecordingActionClient, str]:
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(actions or [{"action": "process_exit", "payload": {"done": True}}])
    runtime.llm.client = client
    pid = runtime.process.spawn(image=image.image_id, goal="finish under context pressure")
    return runtime, client, pid


def test_auto_compaction_ends_quantum_and_deduplicates_pressure_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="auto-context:v0",
        name="auto-context",
        default_tools=["compact_process_context", "process_exit"],
        planner={
            "context_management": {
                "mode": "auto_compact",
                "tool": {
                    "name": "compact_process_context",
                    "arguments": {"target_tokens": 1234},
                },
            }
        },
    )
    runtime, client, pid = _runtime_with_image(image)
    original_dispatch = runtime.llm.adispatch
    calls: list[dict[str, Any]] = []

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        calls.append(dict(action))
        runtime.store.set_llm_context_generation(selected_pid, "compacted-generation")
        return {
            "ok": True,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": {"compacted": True},
            "error": None,
        }

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        first = runtime.run_next_process_once()

        assert first["context_compacted"] is True
        assert client.user_prompts == []
        assert runtime.store.list_llm_calls(pid=pid) == []
        assert calls == [
            {"action": "compact_process_context", "target_tokens": 1234}
        ]
        marker = runtime.store.get_llm_pending_action(pid)
        assert marker is not None and marker["status"] == "completed"
        assert pending_metadata(marker)["source"] == "runtime_context_management"

        second = runtime.run_next_process_once()

        assert second["ok"] is True
        assert len(client.user_prompts) == 1
        assert len(calls) == 1
        call = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert call is not None
        assert call.request_options["context_pressure"]["action"] == "deduplicated"
    finally:
        runtime.close()


def test_auto_compaction_failure_is_silent_and_model_call_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="failed-context:v0",
        name="failed-context",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(image)
    original_dispatch = runtime.llm.adispatch

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        return {
            "ok": False,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": None,
            "error": "compaction refused",
        }

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert len(client.user_prompts) == 1
        assert DEFAULT_CONTEXT_PRESSURE_PROMPT not in client.user_prompts[0]
        call = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert call is not None
        assert call.request_options["context_pressure"]["action"] == "failed"
        assert call.request_options["context_pressure"]["auto_attempted"] is True
        assert "llm.context_pressure_failed" in {
            record.action for record in runtime.audit.trace(actor=pid)
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("failure_kind", ["result", "exception"])
def test_auto_compaction_generation_change_rebuilds_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    image = AgentImage(
        image_id=f"generation-change-{failure_kind}:v0",
        name=f"generation-change-{failure_kind}",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(image)
    original_dispatch = runtime.llm.adispatch

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        runtime.store.set_llm_context_generation(
            selected_pid,
            "generation-after-failed-maintenance",
        )
        if failure_kind == "exception":
            raise RuntimeError("failure after changing context generation")
        return {
            "ok": False,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": None,
            "error": "failure after changing context generation",
        }

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        changed = runtime.run_next_process_once()

        assert changed["context_generation_changed"] is True
        assert changed["context_management_failed"] is True
        assert client.user_prompts == []
        assert runtime.store.list_llm_calls(pid=pid) == []

        rebuilt = runtime.run_next_process_once()

        assert rebuilt["ok"] is True
        assert len(client.user_prompts) == 1
        call = runtime.store.get_latest_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        assert call is not None
        assert (
            call.request_options["llm_context_generation"]
            == "generation-after-failed-maintenance"
        )
    finally:
        runtime.close()


def test_completed_context_marker_does_not_reuse_terminal_llm_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="completed-marker-operation:v0",
        name="completed-marker-operation",
        default_tools=[
            "compact_process_context",
            "create_memory_object",
            "process_exit",
        ],
    )
    runtime, _client, pid = _runtime_with_image(
        image,
        actions=[
            {
                "action": "create_memory_object",
                "type": "observation",
                "payload": {"step": 1},
            },
            {"action": "process_exit", "payload": {"done": True}},
        ],
    )
    original_dispatch = runtime.llm.adispatch

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action.get("action") == "compact_process_context":
            return {
                "ok": False,
                "tool_id": "tool_context",
                "result_oid": None,
                "payload": None,
                "error": "test failure",
            }
        return await original_dispatch(
            selected_pid,
            action,
            context_metadata=context_metadata,
        )

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        first = runtime.run_next_process_once()
        assert first["ok"] is True
        marker = runtime.store.get_llm_pending_action(pid)
        assert marker is not None and marker["status"] == "completed"
        marker_operation_id = marker["llm_operation_id"]

        second = runtime.run_next_process_once()

        assert second["ok"] is True
        llm_operations = [
            operation
            for operation in runtime.store.list_operations(pid=pid)
            if operation.name == "llm.action_selection"
        ]
        assert len(llm_operations) == 2
        assert len({operation.operation_id for operation in llm_operations}) == 2
        assert marker_operation_id in {
            operation.operation_id for operation in llm_operations
        }
    finally:
        runtime.close()


def test_auto_attempt_marker_prevents_replay_after_interrupted_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedInterruption(BaseException):
        pass

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    with tempfile.TemporaryDirectory() as temp_dir:
        db = f"{temp_dir}/runtime.sqlite"
        image = AgentImage(
            image_id="durable-attempt-context:v0",
            name="durable-attempt-context",
            default_tools=["compact_process_context", "process_exit"],
        )
        runtime = Runtime.open(
            db,
            substrate=LocalResourceProviderSubstrate(temp_dir),
        )
        runtime.register_image(image, actor="test")
        initial_client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"must_not_run": True}}]
        )
        runtime.llm.client = initial_client
        pid = runtime.process.spawn(image=image.image_id, goal="survive interruption")
        compact_calls = 0
        original_dispatch = runtime.llm.adispatch

        async def dispatch(
            selected_pid: str,
            action: dict[str, Any],
            *,
            context_metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal compact_calls
            if action.get("action") != "compact_process_context":
                return await original_dispatch(
                    selected_pid,
                    action,
                    context_metadata=context_metadata,
                )
            compact_calls += 1
            return {
                "ok": False,
                "tool_id": "tool_context",
                "result_oid": None,
                "payload": None,
                "error": "test failure",
            }

        def interrupt_before_provider(_pid: str) -> None:
            raise SimulatedInterruption("simulated Runtime interruption")

        monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
        monkeypatch.setattr(
            runtime.llm,
            "_preflight_llm_call",
            interrupt_before_provider,
        )
        try:
            with pytest.raises(SimulatedInterruption):
                runtime.run_next_process_once()

            marker = runtime.store.get_llm_pending_action(pid)
            assert marker is not None and marker["status"] == "completed"
            assert pending_metadata(marker)["outcome"] == "attempted"
            assert runtime.process.get(pid).status.value == "runnable"
            assert runtime.store.list_llm_calls(pid=pid) == []
            assert initial_client.user_prompts == []
            assert compact_calls == 1
        finally:
            runtime.close()

        reopened = Runtime.open(
            db,
            substrate=LocalResourceProviderSubstrate(temp_dir),
        )
        client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"done": True}}]
        )
        reopened.llm.client = client
        reopened_dispatch = reopened.llm.adispatch

        async def dispatch_after_reopen(
            selected_pid: str,
            action: dict[str, Any],
            *,
            context_metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal compact_calls
            if action.get("action") == "compact_process_context":
                compact_calls += 1
            return await reopened_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )

        monkeypatch.setattr(reopened.llm, "adispatch", dispatch_after_reopen)
        try:
            result = reopened.run_next_process_once()

            assert result["ok"] is True
            assert compact_calls == 1
            assert len(client.user_prompts) == 1
            call = reopened.store.get_latest_llm_call(
                pid=pid,
                purpose="action_selection",
            )
            assert call is not None
            assert call.request_options["context_pressure"]["action"] == "deduplicated"
        finally:
            reopened.close()


def test_auto_compaction_message_wait_hydrates_without_repeating_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="waiting-context:v0",
        name="waiting-context",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(image)
    original_dispatch = runtime.llm.adispatch
    compact_calls = 0

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal compact_calls
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        compact_calls += 1
        if compact_calls == 1:
            raise ProcessMessageWaitRequired(
                selected_pid,
                {"channel": "resume-compaction"},
                "wait for compaction input",
            )
        runtime.store.set_llm_context_generation(selected_pid, "resumed-generation")
        return {
            "ok": True,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": {"compacted": True},
            "error": None,
        }

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        waiting = runtime.run_next_process_once()

        assert waiting["waiting_message"] is True
        durable = runtime.store.get_llm_pending_action(pid)
        assert durable is not None and durable["wait_type"] == "message"
        metadata = pending_metadata(durable)
        assert metadata["episode_id"]
        assert metadata["source"] == "runtime_context_management"

        # Exercise the same durable hydration path used by Runtime reopen.
        runtime.llm._clear_in_memory_pending_action(pid)
        runtime.llm._hydrate_pending_action(durable)
        runtime.messages.post(
            sender="test",
            recipient_pid=pid,
            channel="resume-compaction",
            subject="resume",
        )
        resumed = runtime.run_next_process_once()

        assert resumed["context_compacted"] is True
        assert resumed["resumed_context_management"] is True
        assert client.user_prompts == []
        assert compact_calls == 2

        completed = runtime.run_next_process_once()

        assert completed["ok"] is True
        assert len(client.user_prompts) == 1
        assert compact_calls == 2
    finally:
        runtime.close()


def test_auto_compaction_resume_exception_is_soft_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="resume-failure-context:v0",
        name="resume-failure-context",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(image)
    original_dispatch = runtime.llm.adispatch
    compact_calls = 0

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal compact_calls
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        compact_calls += 1
        if compact_calls == 1:
            raise ProcessMessageWaitRequired(
                selected_pid,
                {"channel": "resume-compaction"},
                "wait for compaction input",
            )
        raise RuntimeError("resumed context management failed")

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        waiting = runtime.run_next_process_once()
        assert waiting["waiting_message"] is True
        runtime.messages.post(
            sender="test",
            recipient_pid=pid,
            channel="resume-compaction",
            subject="resume",
        )

        resumed = runtime.run_next_process_once()

        assert resumed["ok"] is True
        assert compact_calls == 2
        assert len(client.user_prompts) == 1
        assert DEFAULT_CONTEXT_PRESSURE_PROMPT not in client.user_prompts[0]
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None and pending["status"] == "completed"
        failure = next(
            record
            for record in reversed(runtime.audit.trace(actor=pid))
            if record.action == "llm.context_pressure_failed"
        )
        assert failure.decision["reason"] == "RuntimeError"
    finally:
        runtime.close()


def test_builtin_auto_compaction_waits_for_child_then_rebuilds_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="builtin-auto-context:v0",
        name="builtin-auto-context",
        default_tools=["compact_process_context", "process_exit"],
        planner={
            "context_management": {
                "mode": "auto_compact",
                "tool": {
                    "name": "compact_process_context",
                    "arguments": {
                        "force": True,
                        "target_tokens": 512,
                        "max_chunks": 1,
                        "preserve_recent_entries": 1,
                    },
                },
            }
        },
    )
    runtime, client, pid = _runtime_with_image(
        image,
        actions=[
            {
                "action": "process_exit",
                "payload": {
                    "goal": "compressed automatically",
                    "constraints": [],
                    "user_preferences": [],
                    "completed": [],
                    "pending": ["continue"],
                    "key_references": {},
                    "recent_decisions": [],
                    "risks": [],
                    "uncertainties": [],
                    "next_steps": ["resume"],
                },
            },
            {"action": "process_exit", "payload": {"done": True}},
        ],
    )
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        "image:context-compressor:v0",
        [CapabilityRight.READ],
        issued_by="test",
    )
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        results = runtime.run_until_idle(max_quanta=4)

        assert any(result.get("waiting_event") for result in results)
        compacted = next(
            result for result in results if result.get("context_compacted") is True
        )
        assert compacted["ok"] is True
        assert runtime.store.get_llm_context_generation(pid) != "initial"
        parent_calls = runtime.store.list_llm_calls(pid=pid)
        assert len(parent_calls) == 1
        assert parent_calls[0].request_options["context_pressure"]["action"] == "deduplicated"
        assert len(client.user_prompts) == 2
        assert any(
            record.action == "llm.context_pressure_compacted"
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


def test_auto_context_approval_reopens_and_failure_continues_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    with tempfile.TemporaryDirectory() as temp_dir:
        db = f"{temp_dir}/runtime.sqlite"
        substrate = LocalResourceProviderSubstrate(temp_dir)
        path = "agent_outputs/auto-context-approval.txt"
        runtime = Runtime.open(db, substrate=substrate)
        image = AgentImage(
            image_id="approval-context:v0",
            name="approval-context",
            default_tools=["write_text_file", "process_exit"],
            planner={
                "context_management": {
                    "mode": "auto_compact",
                    "tool": {
                        "name": "write_text_file",
                        "arguments": {
                            "path": path,
                            "content": "approved maintenance",
                        },
                    },
                }
            },
        )
        runtime.register_image(image, actor="test")
        initial_client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"must_not_run": True}}]
        )
        runtime.llm.client = initial_client
        pid = runtime.process.spawn(image=image.image_id, goal="resume approval")
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=runtime.filesystem.resource_for(path),
            rights=[CapabilityRight.WRITE],
            policy="ask_each_time",
            issued_by="test",
        )
        try:
            waiting = runtime.run_next_process_once()

            assert waiting["waiting_human"] is True
            assert initial_client.user_prompts == []
            pending = runtime.store.get_llm_pending_action(pid)
            assert pending is not None and pending["wait_type"] == "human"
            assert pending_metadata(pending)["source"] == "runtime_context_management"
        finally:
            runtime.close()

        reopened = Runtime.open(db, substrate=LocalResourceProviderSubstrate(temp_dir))
        client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"done": True}}]
        )
        reopened.llm.client = client
        try:
            reopened.human.drain_terminal_queue(auto_approve=True)
            resumed = reopened.run_next_process_once()

            assert resumed["ok"] is True
            assert len(client.user_prompts) == 1
            assert DEFAULT_CONTEXT_PRESSURE_PROMPT not in client.user_prompts[0]
            assert (reopened.workspace_root / path).read_text(encoding="utf-8") == "approved maintenance"
            assert reopened.store.get_llm_pending_action(pid)["status"] == "completed"
            assert any(
                record.action == "llm.context_pressure_failed"
                for record in reopened.audit.trace(actor=pid)
            )
        finally:
            reopened.close()


def test_pressure_episode_resets_after_occupancy_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="episode-context:v0",
        name="episode-context",
        default_tools=[
            "compact_process_context",
            "create_memory_object",
            "process_exit",
        ],
    )
    runtime, client, pid = _runtime_with_image(
        image,
        actions=[
            {"action": "create_memory_object", "type": "observation", "payload": {"step": 1}},
            {"action": "create_memory_object", "type": "observation", "payload": {"step": 2}},
            {"action": "process_exit", "payload": {"done": True}},
        ],
    )
    original_dispatch = runtime.llm.adispatch
    assessments = iter([True, False, True])
    compact_calls = 0

    def assess(**kwargs: Any) -> ContextPressureAssessment:
        return _pressure_assessment(next(assessments), **kwargs)

    async def dispatch(
        selected_pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal compact_calls
        if action.get("action") != "compact_process_context":
            return await original_dispatch(
                selected_pid,
                action,
                context_metadata=context_metadata,
            )
        compact_calls += 1
        return {
            "ok": False,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": None,
            "error": "test failure",
        }

    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", assess)
    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        first = runtime.run_next_process_once()
        second = runtime.run_next_process_once()
        third = runtime.run_next_process_once()

        assert first["ok"] and second["ok"] and third["ok"]
        calls = runtime.store.list_llm_calls(pid=pid)
        assert len(calls) == 3
        first_pressure = calls[0].request_options["context_pressure"]
        second_pressure = calls[1].request_options["context_pressure"]
        third_pressure = calls[2].request_options["context_pressure"]
        assert first_pressure["active"] is True
        assert second_pressure["active"] is False
        assert second_pressure["action"] == "recovered"
        assert third_pressure["active"] is True
        assert third_pressure["episode_id"] != first_pressure["episode_id"]
        assert compact_calls == 2
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "prompt_mode",
    [
        PROMPT_MODE_LIBOS_DEFAULT,
        PROMPT_MODE_MINIMAL_RUNTIME,
        PROMPT_MODE_IMAGE_ONLY,
    ],
)
def test_prompt_policy_applies_to_every_prompt_mode(
    monkeypatch: pytest.MonkeyPatch,
    prompt_mode: str,
) -> None:
    configured_prompt = "Preserve the current implementation checklist."
    image = AgentImage(
        image_id=f"prompt-context-{prompt_mode}:v0",
        name=f"prompt-context-{prompt_mode}",
        prompt_mode=prompt_mode,
        default_tools=["process_exit"],
        planner={
            "unrelated": {"preserved": True},
            "context_management": {
                "mode": "prompt",
                "prompt": configured_prompt,
            },
        },
    )
    runtime, client, _pid = _runtime_with_image(image)
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert configured_prompt in client.user_prompts[0]
        assert "Context window pressure details:" in client.user_prompts[0]
        assert "context window:" in client.user_prompts[0]
        assert "estimated input:" in client.user_prompts[0]
        assert "reserved output:" in client.user_prompts[0]
        assert "projected occupancy:" in client.user_prompts[0]
        assert "utilization:" in client.user_prompts[0]
    finally:
        runtime.close()


def test_disabled_policy_detects_pressure_without_prompt_or_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="disabled-context:v0",
        name="disabled-context",
        default_tools=["process_exit"],
        planner={"context_management": {"mode": "disabled"}},
    )
    runtime, client, pid = _runtime_with_image(image)
    monkeypatch.setattr("agent_libos.llm.executor.assess_context_pressure", _forced_pressure)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert DEFAULT_CONTEXT_PRESSURE_PROMPT not in client.user_prompts[0]
        call = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert call is not None
        assert call.request_options["context_pressure"]["action"] == "disabled"
    finally:
        runtime.close()
