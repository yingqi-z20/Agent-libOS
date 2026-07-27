from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.llm.context_management import (
    DEFAULT_CONTEXT_PRESSURE_PROMPT,
    ContextPressureAssessment,
    assess_context_pressure,
)
from agent_libos.llm.context_memory import (
    LLM_CONTEXT_ENRICHMENT_RESOURCE,
    LLM_CONTEXT_MAINTENANCE_RESOURCE,
    context_object_name,
)
from agent_libos.llm.pending import pending_metadata
from agent_libos.models import (
    CapabilityRight,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_LIBOS_DEFAULT,
    PROMPT_MODE_MINIMAL_RUNTIME,
    ProcessStatus,
    ResourceBudget,
    ViewMode,
)
from agent_libos.models.exceptions import ProcessMessageWaitRequired, ValidationError
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.fakes import RecordingActionClient
from tests.support.public_errors import assert_public_error_message


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
    config: AgentLibOSConfig | None = None,
) -> tuple[Runtime, RecordingActionClient, str]:
    runtime = Runtime(SQLiteStore(":memory:"), config=config)
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(actions or [{"action": "process_exit", "payload": {"done": True}}])
    runtime.llm.client = client
    pid = runtime.process.spawn(image=image.image_id, goal="finish under context pressure")
    _grant_persistent_context(runtime, pid)
    return runtime, client, pid


def _grant_persistent_context(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(
        pid,
        LLM_CONTEXT_ENRICHMENT_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        LLM_CONTEXT_MAINTENANCE_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )


def test_default_context_uses_only_materialized_source_without_delta_object() -> None:
    image = AgentImage(
        image_id="source-only-context:v0",
        name="source-only-context",
        default_tools=["get_current_time"],
    )
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    client = RecordingActionClient([{"action": "get_current_time"}])
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="use only the already materialized source context",
    )
    try:
        result = runtime.run_next_process_once()

        assert result["action"]["action"] == "get_current_time"
        assert DEFAULT_CONFIG.llm_context.policy == "source_only"
        assert "LLM context object:" not in client.user_prompts[0]
        assert "capabilities_delta" not in client.user_prompts[0]
        assert runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        ) is None
        manifest = runtime.store.list_context_materialization_manifests(pid=pid)[0]
        assert manifest.context_oid is None
        assert manifest.context_version is None
        assert manifest.compaction["mode"] == "source_only"
    finally:
        runtime.close()


def test_default_source_only_context_is_charged_to_cumulative_budget() -> None:
    image = AgentImage(
        image_id="source-only-budget:v0",
        name="source-only-budget",
        default_tools=["process_exit"],
    )
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"must_not_run": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="charge this selected source context before calling the provider",
        resource_budget=ResourceBudget(
            max_context_materialization_tokens=100_000,
            max_context_materialization_total_tokens=100_000,
        ),
    )
    source = runtime.memory.create_object(
        pid,
        ObjectType.EVIDENCE,
        {"text": "selected source content " * 32},
        name="source-only-budget-evidence",
    )
    process = runtime.process.get(pid)
    process.memory_view = runtime.memory.create_view(
        pid,
        [source],
        mode=ViewMode.READ_ONLY,
    )
    runtime.store.update_process(process)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True, result
        assert len(client.user_prompts) == 1
        process = runtime.process.get(pid)
        manifest = runtime.store.list_context_materialization_manifests(pid=pid)[0]
        assert manifest.rendered_tokens > 0
        assert (
            process.resource_usage.context_materialized_tokens
            == manifest.rendered_tokens
        )
        assert process.status == ProcessStatus.EXITED
        assert runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        ) is None
    finally:
        runtime.close()


def test_default_source_only_pressure_does_not_inject_or_run_context_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="source-only-pressure:v0",
        name="source-only-pressure",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="keep source-only context under pressure",
    )
    monkeypatch.setattr(
        "agent_libos.llm.executor.assess_context_pressure",
        _forced_pressure,
    )
    try:
        result = runtime.run_next_process_once()

        assert result["action"]["action"] == "process_exit"
        assert DEFAULT_CONTEXT_PRESSURE_PROMPT not in client.user_prompts[0]
        assert runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        ) is None
        call = runtime.store.get_latest_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        assert call is not None
        assert call.request_options["context_pressure"]["action"] == "not_authorized"
        audit_actions = {
            record.action for record in runtime.audit.trace(actor=pid)
        }
        assert "llm.context_pressure_maintenance_not_authorized" in audit_actions
        assert "llm.context_pressure_auto_attempted" not in audit_actions
        assert "llm.context_pressure_prompted" not in audit_actions
    finally:
        runtime.close()


def test_explicit_context_enrichment_authority_enables_delta_object() -> None:
    image = AgentImage(
        image_id="explicit-context-enrichment:v0",
        name="explicit-context-enrichment",
        default_tools=["get_current_time"],
        prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
    )
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    client = RecordingActionClient([{"action": "get_current_time"}])
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="explicitly enable persistent context enrichment",
    )
    runtime.capability.grant(
        pid,
        LLM_CONTEXT_ENRICHMENT_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    try:
        result = runtime.run_next_process_once()

        assert result["action"]["action"] == "get_current_time"
        assert "LLM context object:" in client.user_prompts[0]
        context = runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        )
        assert context is not None
        assert any(
            entry.get("kind") == "capabilities_snapshot"
            for entry in context.payload["entries"]
        )
    finally:
        runtime.close()


def test_finite_context_enrichment_authority_is_consumed_by_materialization() -> None:
    image = AgentImage(
        image_id="finite-context-enrichment:v0",
        name="finite-context-enrichment",
        default_tools=["get_current_time"],
        prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
    )
    runtime = Runtime(SQLiteStore(":memory:"))
    runtime.register_image(image, actor="test")
    runtime.llm.client = RecordingActionClient([{"action": "get_current_time"}])
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="consume one persistent context materialization",
    )
    capability = runtime.capability.grant_once(
        pid,
        LLM_CONTEXT_ENRICHMENT_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    try:
        result = runtime.run_next_process_once()

        assert result["action"]["action"] == "get_current_time"
        persisted = runtime.store.get_capability(capability.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0
        assert runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        ) is not None
    finally:
        runtime.close()


def test_finite_context_maintenance_authority_is_consumed_by_auto_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="finite-context-maintenance:v0",
        name="finite-context-maintenance",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(image)
    unlimited = next(
        capability
        for capability in runtime.capability.capabilities_for(pid)
        if capability.resource == LLM_CONTEXT_MAINTENANCE_RESOURCE
        and capability.active
    )
    runtime.capability.revoke(
        unlimited.cap_id,
        revoked_by="test",
        require_authority=False,
    )
    finite = runtime.capability.grant_once(
        pid,
        LLM_CONTEXT_MAINTENANCE_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    assert runtime.capability.check(
        pid,
        LLM_CONTEXT_MAINTENANCE_RESOURCE,
        CapabilityRight.EXECUTE,
    )
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
            "finite-maintenance-complete",
        )
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
        result = runtime.run_next_process_once()

        latest_call = runtime.store.get_latest_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        failed_audits = [
            record.decision
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.context_pressure_failed"
        ]
        assert not failed_audits, [
            {
                "reason": audit.get("reason"),
                "error_type": audit.get("error_type"),
                "internal_error": audit.get("internal_error"),
            }
            for audit in failed_audits
        ]
        assert result.get("context_compacted") is True, {
            "result": result,
            "pressure": (
                latest_call.request_options.get("context_pressure")
                if latest_call is not None
                else None
            ),
        }
        assert client.user_prompts == []
        persisted = runtime.store.get_capability(finite.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0
    finally:
        runtime.close()


def _storage_pressure_config(threshold_bytes: int = 5_000) -> AgentLibOSConfig:
    return replace(
        DEFAULT_CONFIG,
        llm_context=replace(
            DEFAULT_CONFIG.llm_context,
            storage_compaction_threshold_bytes=threshold_bytes,
        ),
    )


def test_storage_pressure_compacts_before_provider_or_object_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold_bytes = 8_000
    image = AgentImage(
        image_id="storage-pressure:v0",
        name="storage-pressure",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(
        image,
        config=_storage_pressure_config(threshold_bytes),
    )
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
        runtime.store.set_llm_context_generation(
            selected_pid,
            "storage-compacted-generation",
        )
        return {
            "ok": True,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": {"compacted": True},
            "error": None,
        }

    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        result = runtime.run_next_process_once()

        assert result["context_compacted"] is True
        assert result["context_storage_pressure"] is True
        assert calls == [
            {
                "action": "compact_process_context",
                "force": True,
                "max_chunks": 4,
                "preserve_recent_entries": 0,
            }
        ]
        assert client.user_prompts == []
        assert runtime.store.list_llm_calls(pid=pid) == []
        detected = next(
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.context_pressure_detected"
        )
        assert detected.decision["trigger"] == "storage_payload"
        assert detected.decision["payload_bytes"] >= threshold_bytes
        assert detected.decision["projected_payload_bytes"] >= threshold_bytes
        assert detected.decision["persisted_payload_bytes"] < threshold_bytes
        assert detected.decision["threshold_bytes"] == threshold_bytes
        assert (
            detected.decision["hard_limit_bytes"]
            == DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes
        )
    finally:
        runtime.close()


def test_storage_pressure_compaction_failure_fails_without_provider_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="storage-pressure-failure:v0",
        name="storage-pressure-failure",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(
        image,
        config=_storage_pressure_config(),
    )
    compact_calls = 0

    async def dispatch(
        _selected_pid: str,
        _action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal compact_calls
        compact_calls += 1
        return {
            "ok": False,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": None,
            "error": "compaction refused",
        }

    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        failed = runtime.run_next_process_once()

        assert failed["ok"] is False
        assert_public_error_message(
            failed["error"],
            code="llm_context_management_error",
            error_type="RuntimeError",
            forbidden=["compaction refused", "storage compaction failed"],
        )
        assert runtime.process.get(pid).status.value == "failed"
        assert compact_calls == 1
        assert client.user_prompts == []
        assert runtime.store.list_llm_calls(pid=pid) == []

        skipped = runtime.run_next_process_once()
        assert skipped is None
        assert compact_calls == 1
    finally:
        runtime.close()


def test_storage_pressure_requires_explicit_context_maintenance_authority() -> None:
    image = AgentImage(
        image_id="storage-pressure-not-authorized:v0",
        name="storage-pressure-not-authorized",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime = Runtime(
        SQLiteStore(":memory:"),
        config=_storage_pressure_config(),
    )
    runtime.register_image(image, actor="test")
    client = RecordingActionClient(
        [{"action": "process_exit", "payload": {"done": True}}]
    )
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="do not auto-compact without explicit authority",
    )
    runtime.capability.grant(
        pid,
        LLM_CONTEXT_ENRICHMENT_RESOURCE,
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    try:
        result = runtime.run_next_process_once()

        assert result["action"]["action"] == "process_exit"
        assert runtime.process.get(pid).status.value == "exited"
        assert len(runtime.store.list_llm_calls(pid=pid)) == 1
        assert not any(
            record.action.startswith("llm.context_pressure_")
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


def test_storage_pressure_rearms_above_post_compaction_artifact_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="storage-pressure-hysteresis:v0",
        name="storage-pressure-hysteresis",
        default_tools=[
            "compact_process_context",
            "get_current_time",
            "process_exit",
        ],
    )
    runtime, client, pid = _runtime_with_image(
        image,
        actions=[
            {"action": "get_current_time"},
            {"action": "process_exit", "payload": {"unexpected": True}},
        ],
        config=_storage_pressure_config(),
    )
    process = runtime.process.get(pid)
    context_handle = runtime.llm.context_memory.ensure(
        pid,
        runtime.images[process.image_id],
        process,
        runtime.tools.model_visible_tools(pid),
    )
    context = runtime.memory.get_object(pid, context_handle)
    compacted_payload = deepcopy(context.payload)
    compacted_payload["cache_strategy"] = {
        **dict(compacted_payload["cache_strategy"]),
        "mode": "compacted_stable_prefix",
        "compacted_at": "test-compaction-generation",
        "storage_compaction_rearm_pending": True,
        "storage_compaction_baseline_bytes": None,
        "storage_compaction_rearm_at_bytes": None,
    }
    runtime.memory.update_object(
        pid,
        runtime.memory.handle_for_name(
            pid,
            context_object_name(pid),
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
        ),
        ObjectPatch(payload=compacted_payload),
    )
    compact_calls: list[dict[str, Any]] = []
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
        compact_calls.append(dict(action))
        runtime.store.set_llm_context_generation(
            selected_pid,
            "hysteresis-compacted-generation",
        )
        return {
            "ok": True,
            "tool_id": "tool_context",
            "result_oid": None,
            "payload": {"compacted": True},
            "error": None,
        }

    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        first = runtime.run_next_process_once()

        assert first["action"]["action"] == "get_current_time"
        assert compact_calls == []
        assert len(client.user_prompts) == 1
        armed = runtime.store.get_object_by_name(
            context_object_name(pid),
            namespace=runtime.memory.resolve_namespace(pid),
        )
        assert armed is not None
        cache_strategy = armed.payload["cache_strategy"]
        baseline = cache_strategy["storage_compaction_baseline_bytes"]
        rearm_at = cache_strategy["storage_compaction_rearm_at_bytes"]
        assert cache_strategy["storage_compaction_rearm_pending"] is False
        assert baseline >= 5_000
        assert rearm_at == baseline + 5_000

        oversized = deepcopy(armed.payload)
        oversized["entries"].append(
            {
                "kind": "growth_after_compaction",
                "content": "x" * (rearm_at - baseline + 2_000),
            }
        )
        runtime.memory.update_object(
            pid,
            runtime.memory.handle_for_name(
                pid,
                context_object_name(pid),
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
            ),
            ObjectPatch(payload=oversized),
        )

        second = runtime.run_next_process_once()

        assert second["context_storage_pressure"] is True
        assert len(compact_calls) == 1
        assert len(client.user_prompts) == 1
        detected = next(
            record
            for record in reversed(runtime.audit.trace(actor=pid))
            if record.action == "llm.context_pressure_detected"
        )
        assert detected.decision["compaction_baseline_bytes"] == baseline
        assert detected.decision["rearm_at_bytes"] == rearm_at
        assert detected.decision["effective_threshold_bytes"] == rearm_at
    finally:
        runtime.close()


def test_storage_pressure_resume_failure_fails_closed_without_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="storage-pressure-resume-failure:v0",
        name="storage-pressure-resume-failure",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime, client, pid = _runtime_with_image(
        image,
        config=_storage_pressure_config(),
    )
    compact_calls = 0

    async def dispatch(
        selected_pid: str,
        _action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal compact_calls
        compact_calls += 1
        if compact_calls == 1:
            raise ProcessMessageWaitRequired(
                selected_pid,
                {"channel": "resume-storage-compaction"},
                "wait for storage compaction input",
            )
        raise RuntimeError("resumed storage compaction failed")

    monkeypatch.setattr(runtime.llm, "adispatch", dispatch)
    try:
        waiting = runtime.run_next_process_once()
        assert waiting["waiting_message"] is True

        runtime.messages.post(
            sender="test",
            recipient_pid=pid,
            channel="resume-storage-compaction",
            subject="resume",
        )
        failed = runtime.run_next_process_once()

        assert failed["ok"] is False
        assert_public_error_message(
            failed["error"],
            code="llm_error",
            error_type="RuntimeError",
            forbidden=[
                "resumed storage compaction failed",
                "storage compaction failed",
            ],
        )
        assert runtime.process.get(pid).status.value == "failed"
        assert compact_calls == 2
        assert client.user_prompts == []
        assert runtime.store.list_llm_calls(pid=pid) == []
    finally:
        runtime.close()


def test_storage_pressure_builtin_compactor_uses_image_bound_spawn_authority() -> None:
    image = AgentImage(
        image_id="storage-pressure-builtin:v0",
        name="storage-pressure-builtin",
        default_tools=["compact_process_context", "process_exit"],
    )
    runtime = Runtime(
        SQLiteStore(":memory:"),
        config=_storage_pressure_config(30_000),
    )
    runtime.register_image(image, actor="test")
    summary_action = {
        "action": "process_exit",
        "payload": {
            "goal": "finish after bounded compaction",
            "constraints": [],
            "user_preferences": [],
            "completed": [],
            "pending": ["continue parent task"],
            "key_references": {},
            "recent_decisions": [],
            "risks": [],
            "uncertainties": [],
            "next_steps": ["resume parent"],
        },
    }
    client = RecordingActionClient([summary_action for _ in range(3)])
    runtime.llm.client = client
    pid = runtime.process.spawn(
        image=image.image_id,
        goal="compact through a bounded child",
        authority_manifest={
            "authorized_capabilities": [
                {
                    "resource": LLM_CONTEXT_ENRICHMENT_RESOURCE,
                    "rights": ["execute"],
                },
                {
                    "resource": LLM_CONTEXT_MAINTENANCE_RESOURCE,
                    "rights": ["execute"],
                },
                {
                    "resource": "process:spawn",
                    "rights": ["write"],
                    "constraints": {
                        "authority_rules": [
                            {
                                "rule_id": "test.context-maintenance.spawn",
                                "operation": "process.spawn_child",
                                "effect": "allow",
                                "risk": "low",
                                "conditions": {
                                    "image_id": "context-compressor:v0"
                                },
                            }
                        ]
                    },
                },
                {
                    "resource": "image:context-compressor:v0",
                    "rights": ["read"],
                },
            ]
        },
    )
    try:
        process = runtime.process.get(pid)
        context_handle = runtime.llm.context_memory.ensure(
            pid,
            runtime.images[process.image_id],
            process,
            runtime.tools.model_visible_tools(pid),
        )
        context = runtime.memory.get_object(pid, context_handle)
        payload = deepcopy(context.payload)
        payload["entries"].extend(
            {
                "kind": "seed_entry",
                "index": index,
                "content": "x" * 7_000,
            }
            for index in range(4)
        )
        write_handle = runtime.memory.handle_for_name(
            pid,
            context_object_name(pid),
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
        )
        runtime.memory.update_object(
            pid,
            write_handle,
            ObjectPatch(payload=payload),
        )

        results = runtime.run_until_idle(max_quanta=7)

        assert any(result.get("waiting_event") for result in results)
        compacted_results = [
            result
            for result in results
            if result.get("context_storage_pressure") is True
        ]
        assert compacted_results, results
        compacted = compacted_results[0]
        assert compacted["context_compacted"] is True
        children = runtime.process.list_children(pid)
        assert len(children) == 3
        assert all(child.image_id == "context-compressor:v0" for child in children)
        assert runtime.store.list_llm_calls(pid=pid) == []
        assert all(
            len(runtime.store.list_llm_calls(pid=child.pid)) == 1
            for child in children
        )
        assert any(
            record.action == "llm.context_pressure_compacted"
            and record.decision["trigger"] == "storage_payload"
            for record in runtime.audit.trace(actor=pid)
        )
    finally:
        runtime.close()


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
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
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
        _grant_persistent_context(runtime, pid)
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
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
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
        _grant_persistent_context(runtime, pid)
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
    ],
)
def test_prompt_policy_applies_to_runtime_owned_prompt_modes(
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


def test_image_only_rejects_prompt_context_management() -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(
            ValidationError,
            match="image_only does not allow prompt-mode context management",
        ):
            runtime.register_image(
                AgentImage(
                    image_id="transparent-no-runtime-prompt:v0",
                    name="transparent-no-runtime-prompt",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    planner={"context_management": {"mode": "prompt"}},
                ),
                actor="test",
            )
    finally:
        runtime.close()


def test_prompt_policy_accounts_for_the_exact_notice_bearing_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="prompt-context-exact-accounting:v0",
        name="prompt-context-exact-accounting",
        prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
        default_tools=["process_exit"],
        planner={
            "context_management": {
                "mode": "prompt",
                "threshold_ratio": 0.001,
                "prompt": "Preserve exact context pressure accounting.",
            }
        },
    )
    runtime, client, pid = _runtime_with_image(image)
    captured_messages: list[list[dict[str, Any]]] = []
    original_complete = client.complete_action

    def capture_complete(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        captured_messages.append([dict(message) for message in messages])
        return original_complete(messages, tools)

    monkeypatch.setattr(client, "complete_action", capture_complete)
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is True
        assert len(captured_messages) == 1
        call = runtime.store.get_latest_llm_call(pid=pid, purpose="action_selection")
        assert call is not None
        resolved = runtime.llms.resolve("default")
        expected = assess_context_pressure(
            messages=captured_messages[0],
            tools=client.tool_batches[0],
            context_window_tokens=resolved.context_window_tokens,
            reserved_output_tokens=resolved.max_tokens,
            threshold_ratio=0.001,
            profile_id=resolved.profile_id,
            context_generation=call.request_options["llm_context_generation"],
        )
        pressure = call.request_options["context_pressure"]
        assert pressure["local_input_estimate_tokens"] == expected.local_input_estimate_tokens
        assert pressure["estimated_input_tokens"] == expected.estimated_input_tokens
        assert pressure["projected_tokens"] == expected.projected_tokens
        assert pressure["utilization_ratio"] == expected.utilization_ratio
        assert pressure["prompt_notice_estimate_tokens"] > 0
        prompted = next(
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.context_pressure_prompted"
        )
        assert prompted.decision["projected_tokens"] == expected.projected_tokens
        assert prompted.decision["prompt_notice_estimate_tokens"] > 0
        assert prompted.decision["prompt_notice_sha256"]
    finally:
        runtime.close()


def test_prompt_policy_does_not_dispatch_when_notice_would_cross_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = AgentImage(
        image_id="prompt-context-overflow:v0",
        name="prompt-context-overflow",
        prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
        default_tools=["process_exit"],
        planner={"context_management": {"mode": "prompt"}},
    )
    runtime, client, pid = _runtime_with_image(image)
    assessments = 0

    def crosses_window(**kwargs: Any) -> ContextPressureAssessment:
        nonlocal assessments
        assessments += 1
        window = int(kwargs["context_window_tokens"])
        reserved = int(kwargs["reserved_output_tokens"])
        projected = window - 1 if assessments == 1 else window + 1
        estimated = projected - reserved
        provider_lower_bound = int(kwargs.get("provider_lower_bound_tokens", 0))
        return ContextPressureAssessment(
            context_window_tokens=window,
            local_input_estimate_tokens=max(1, estimated - provider_lower_bound),
            provider_usage_lower_bound_tokens=provider_lower_bound,
            estimated_input_tokens=estimated,
            reserved_output_tokens=reserved,
            projected_tokens=projected,
            utilization_ratio=projected / window,
            threshold_ratio=float(kwargs["threshold_ratio"]),
            triggered=True,
            profile_id=str(kwargs["profile_id"]),
            context_generation=str(kwargs["context_generation"]),
        )

    monkeypatch.setattr(
        "agent_libos.llm.executor.assess_context_pressure",
        crosses_window,
    )
    try:
        result = runtime.run_next_process_once()

        assert result["ok"] is False
        assert result["resource_limit_exceeded"] is True
        assert client.user_prompts == []
        assert runtime.process.get(pid).status.value == "killed"
        failure = next(
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.context_pressure_failed"
        )
        assert failure.decision["reason"] == "prompt_notice_exceeds_context_window"
        assert failure.decision["projected_tokens"] == (
            failure.decision["context_window_tokens"] + 1
        )
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
