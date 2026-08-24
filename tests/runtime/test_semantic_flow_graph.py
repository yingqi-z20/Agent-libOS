from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SemanticDefaults
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    DataFlowContext,
    DataLabels,
    DataSensitivity,
    ObjectType,
    Provenance,
)
from agent_libos.models.semantic import (
    SemanticDataCategory,
    SemanticDataFinding,
    SemanticDataLocator,
)
from agent_libos.semantic.flow import (
    FlowDataLocator,
    FlowEligibilityReason,
    FlowEdgeRelation,
    FlowInputEdge,
    FlowMemoryGateReason,
    FlowNodeRef,
    FlowNodeType,
    SemanticFlowService,
)
from agent_libos.semantic import runtime_flow as runtime_flow_module
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.sdk.protected_operations import (
    _post_commit_result_identity,
    visit_bounded_host_result_text,
)


_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64
_D4 = "4" * 64
_D5 = "5" * 64
_TENANT = "a" * 64


class _FlowGraphLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete_action(self, _messages: object, _tools: object) -> LLMCompletion:
        return LLMCompletion(
            content=self.content,
            tool_calls=[
                {
                    "id": "flow-lineage-exit",
                    "name": "process_exit",
                    "arguments": '{"payload":{"done":true}}',
                }
            ],
            raw=object(),
            reasoning=object(),
            provider_trace={"opaque": object()},
            model="flow-test-model",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        )


def test_runtime_shadow_wires_root_and_provider_flow_capture(tmp_path: Path) -> None:
    sentinel = "SEMANTIC_RUNTIME_FLOW_SECRET_SENTINEL"
    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal=f"inspect {sentinel}")
        root_page = runtime.semantic.query_flow_entities(
            pid=pid,
            kind="root_goal",
            limit=10,
        )
        assert len(root_page["items"]) == 1

        observation = SimpleNamespace(
            contract_name="primitive.test.read",
            pid=pid,
            provider="jsonrpc",
            operation="call",
            effect_id="effect-runtime-flow",
            result_sha256=_D1,
            result_descriptor={"digest_mode": "canonical_bounded"},
            target="test:item",
            data_flow_direction="ingress",
            data_labels=DataLabels(
                trust_level="untrusted",
                integrity="untrusted",
            ),
            source_refs_sha256=_D5,
            tool_schema_sha256=_D2,
            provider_spec_sha256=_D3,
        )
        job = runtime.semantic.capture_provider_ingress(
            {"status": "ok"},
            observation,
        )
        assert job is not None
        provider_page = runtime.semantic.query_flow_entities(
            pid=pid,
            kind="provider_result",
            limit=10,
        )
        assert len(provider_page["items"]) == 1
        assert provider_page["items"][0]["content_sha256"] == _D1

        retained_flow = json.dumps(
            {"root": root_page, "provider": provider_page},
            ensure_ascii=False,
            sort_keys=True,
        )
        assert sentinel not in retained_flow
    finally:
        runtime.close()


def test_runtime_shadow_captures_object_tool_materialization_and_model_lineage(
    tmp_path: Path,
) -> None:
    action_sentinel = "TOOL_ACTION_SECRET_SENTINEL_28c7"
    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="Exercise payload-free runtime lineage.")
        source = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"value": "source"},
            name="flow.source",
        )
        source_context = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [source]),
            charge_resources=False,
        )
        tool_result = runtime.memory.create_object(
            pid,
            ObjectType.TOOL_RESULT,
            {"ok": True},
            provenance=Provenance(
                created_from_action=f"tool.{action_sentinel}/private"
            ),
            name="flow.tool-result",
        )
        tool_context = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [tool_result]),
            charge_resources=False,
        )
        runtime.semantic_runtime_flow.observe_model_output(
            SimpleNamespace(
                pid=pid,
                flow_context=DataFlowContext(
                    labels=DataLabels(),
                    materialization_id=source_context.materialization_id,
                ),
                profile_id="profile-test",
                call_id="call-test",
                attempt=1,
            ),
            SimpleNamespace(
                content="select example",
                tool_calls=({"name": "example", "arguments": {}},),
            ),
            SimpleNamespace(model="model-test"),
        )

        entities = runtime.semantic_flow.list_entities(pid=pid, limit=100)["items"]
        activities = runtime.semantic_flow.list_activities(pid=pid, limit=100)[
            "items"
        ]
        entity_kinds = {item["kind"] for item in entities}
        activity_kinds = {item["kind"] for item in activities}

        assert {
            "object_version",
            "tool_result",
            "materialization",
            "model_output",
        } <= entity_kinds
        assert {
            "object_create",
            "tool_call",
            "object_materialize",
            "memory_retrieval",
            "llm_call",
            "tool_selection",
            "conditional",
        } <= activity_kinds
        assert any(
            item["kind"] == "materialization"
            and item["content_sha256"]
            and item["coverage"] == "partial"
            for item in entities
        ), "tool-derived materialization must inherit unproven Tool coverage"
        assert tool_context.text
        retained = json.dumps(
            {
                "entities": entities,
                "activities": activities,
                "edges": runtime.semantic_flow.list_edges(pid=pid, limit=100),
            },
            sort_keys=True,
        )
        assert action_sentinel not in retained
    finally:
        runtime.close()


def test_runtime_flow_observers_stop_at_live_off_kill_switch(tmp_path: Path) -> None:
    relative = "after-off.txt"
    (tmp_path / relative).write_text("business result remains available\n")
    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="Turn semantic capture off.")
        before = runtime.semantic_flow.status()
        runtime.semantic.set_mode("off")

        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"after": "off"},
            name="flow.after-off",
        )
        context = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [handle]),
            charge_resources=False,
        )
        runtime.semantic_runtime_flow.observe_model_output(
            SimpleNamespace(
                pid=pid,
                flow_context=DataFlowContext(
                    labels=DataLabels(),
                    materialization_id=context.materialization_id,
                ),
                profile_id="off-profile",
                call_id="off-call",
                attempt=1,
            ),
            SimpleNamespace(content="ignored lineage", tool_calls=()),
            SimpleNamespace(model="off-model"),
        )
        runtime.capability.grant(
            subject=pid,
            resource=runtime.filesystem.resource_for(relative),
            rights=("read",),
            issued_by="test.host",
        )
        assert "business result" in runtime.filesystem.read_text(
            pid,
            relative,
        ).content

        after = runtime.semantic_flow.status()
        assert after["counts"] == before["counts"]
        assert after["capture_failures"] == before["capture_failures"]
    finally:
        runtime.close()


def test_runtime_flow_observer_caches_are_globally_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_flow_module, "_MAX_TRACKED_OBJECT_VERSIONS", 2)
    monkeypatch.setattr(runtime_flow_module, "_MAX_TRACKED_MATERIALIZATIONS", 2)
    monkeypatch.setattr(runtime_flow_module, "_MAX_PENDING_SELECTION_KEYS", 2)
    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="Bound semantic observer caches.")
        contexts = []
        for index in range(4):
            handle = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {"index": index},
                name=f"flow.bound-{index}",
            )
            contexts.append(
                runtime.memory.materialize_context(
                    pid,
                    runtime.memory.create_view(pid, [handle]),
                    charge_resources=False,
                )
            )
        runtime.semantic_runtime_flow.observe_model_output(
            SimpleNamespace(
                pid=pid,
                flow_context=DataFlowContext(
                    labels=DataLabels(),
                    materialization_id=contexts[-1].materialization_id,
                ),
                profile_id="bounded-profile",
                call_id="bounded-call",
                attempt=1,
            ),
            SimpleNamespace(
                content="bounded",
                tool_calls=tuple(
                    {"name": f"tool-{index}", "arguments": {}}
                    for index in range(4)
                ),
            ),
            SimpleNamespace(model="bounded-model"),
        )

        status = runtime.semantic_runtime_flow.cache_status()
        assert status["object_versions"] <= status["object_version_limit"] == 2
        assert status["materializations"] <= status["materialization_limit"] == 2
        assert status["selection_keys"] <= status["selection_key_limit"] == 2
    finally:
        runtime.close()


def test_model_output_binds_all_bounded_tool_calls_before_selection_truncation(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    try:
        pid = runtime.process.spawn(goal="Bind the complete model tool-call set.")
        state = SimpleNamespace(
            pid=pid,
            flow_context=DataFlowContext(labels=DataLabels()),
            profile_id="profile-test",
            call_id="call-many-tools",
            attempt=1,
        )
        common = tuple(
            {"name": f"tool-{index}", "arguments": {"index": index}}
            for index in range(128)
        )
        runtime.semantic_runtime_flow.observe_model_output(
            state,
            SimpleNamespace(
                content="same content",
                tool_calls=(*common, {"name": "tail-a", "arguments": {}}),
            ),
            SimpleNamespace(model="model-test"),
        )
        first_ref = runtime.semantic_runtime_flow._latest_model_output[pid].ref
        first = runtime.semantic_flow.get_entity(first_ref.node_id)
        runtime.semantic_runtime_flow.observe_model_output(
            state,
            SimpleNamespace(
                content="same content",
                tool_calls=(*common, {"name": "tail-b", "arguments": {}}),
            ),
            SimpleNamespace(model="model-test"),
        )
        second_ref = runtime.semantic_runtime_flow._latest_model_output[pid].ref
        second = runtime.semantic_flow.get_entity(second_ref.node_id)

        assert first is not None and second is not None
        assert first.content_sha256 != second.content_sha256
        assert first.coverage == second.coverage == "partial"
    finally:
        runtime.close()


def test_model_provenance_requires_frozen_profile_model_and_sink_identity() -> None:
    complete = SimpleNamespace(
        profile_id="profile-a",
        resolved=SimpleNamespace(
            profile_id="profile-a",
            identity_sha256=_D1,
            profile=SimpleNamespace(model="model-a"),
        ),
        sink=SimpleNamespace(identity_sha256=_D1),
    )
    _digest_value, coverage = (
        runtime_flow_module.SemanticRuntimeFlowObserver._model_provenance(
            complete,
            SimpleNamespace(model="model-a"),
        )
    )
    assert coverage.value == "complete"

    _digest_value, missing = (
        runtime_flow_module.SemanticRuntimeFlowObserver._model_provenance(
            SimpleNamespace(profile_id="profile-a", resolved=None, sink=None),
            SimpleNamespace(model=None),
        )
    )
    assert missing.value == "partial"

    _digest_value, drifted = (
        runtime_flow_module.SemanticRuntimeFlowObserver._model_provenance(
            complete,
            SimpleNamespace(model="model-b"),
        )
    )
    assert drifted.value == "conflict"


def test_llm_completion_identity_excludes_opaque_provider_fields() -> None:
    first = LLMCompletion(
        content="stable model output",
        tool_calls=[{"id": "call-1", "name": "echo", "arguments": "{}"}],
        raw={"private": "RAW_PROVIDER_SECRET_A"},
        reasoning={"private": "RAW_REASONING_SECRET_A"},
        provider_trace={"private": "RAW_TRACE_SECRET_A"},
        model="model-a",
        usage={"total_tokens": 3},
    )
    second = LLMCompletion(
        content=first.content,
        tool_calls=list(first.tool_calls),
        raw=object(),
        reasoning=object(),
        provider_trace={"opaque": object()},
        model=first.model,
        usage=dict(first.usage),
    )

    first_digest, first_descriptor = _post_commit_result_identity(
        first,
        contract_name="primitive.llm.complete",
    )
    second_digest, second_descriptor = _post_commit_result_identity(
        second,
        contract_name="primitive.llm.complete",
    )
    visited: list[str | bytes] = []
    visit_bounded_host_result_text(
        first,
        contract_name="primitive.llm.complete",
        visitor=visited.append,
    )

    assert first_digest == second_digest
    assert first_digest is not None
    assert first_descriptor["digest_mode"] == "canonical_bounded"
    assert second_descriptor["digest_mode"] == "canonical_bounded"
    retained = json.dumps(visited, ensure_ascii=False, default=str)
    assert "stable model output" in retained
    assert "RAW_PROVIDER_SECRET_A" not in retained
    assert "RAW_REASONING_SECRET_A" not in retained
    assert "RAW_TRACE_SECRET_A" not in retained


def test_real_llm_provider_result_links_to_model_output_and_tightens_labels(
    tmp_path: Path,
) -> None:
    sentinel = "FLOW_REAL_LLM_PASSWORD_SENTINEL_781e"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "provider-model-lineage.db"
    runtime = Runtime.open(
        database,
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        runtime.llm.client = _FlowGraphLLMClient(f"password={sentinel}")
        pid = runtime.process.spawn(goal="Exercise real protected LLM lineage.")
        root_items = runtime.semantic_flow.list_entities(
            pid=pid,
            kind="root_goal",
            limit=10,
        )["items"]
        assert len(root_items) == 1
        root_entity_id = str(root_items[0]["entity_id"])
        runtime.run_process_once(pid)

        provider_items: list[dict[str, object]] = []
        model_entity_id: str | None = None
        for _attempt in range(200):
            provider_items = runtime.semantic_flow.list_entities(
                pid=pid,
                kind="provider_result",
                limit=20,
            )["items"]
            activities = runtime.semantic_flow.list_activities(pid=pid, limit=100)[
                "items"
            ]
            llm_activities = [
                item
                for item in activities
                if item["kind"] == "llm_call" and item["effect_id"] is not None
            ]
            if provider_items and llm_activities:
                llm_activity = llm_activities[0]
                edges = runtime.semantic_flow.list_edges(
                    node_id=llm_activity["activity_id"],
                    limit=100,
                )["items"]
                outputs = [
                    item
                    for item in edges
                    if item["source_node_id"] == llm_activity["activity_id"]
                    and item["target_node_type"] == "entity"
                ]
                if outputs:
                    model_entity_id = str(outputs[0]["target_node_id"])
                    if (
                        runtime.semantic_flow.coverage(model_entity_id)
                        .effective_labels.sensitivity
                        is DataSensitivity.SECRET
                    ):
                        break
            time.sleep(0.01)

        assert len(provider_items) == 1
        assert model_entity_id is not None
        provider_id = str(provider_items[0]["entity_id"])
        llm_activity = llm_activities[0]
        provider_activities = [
            item
            for item in activities
            if item["kind"] == "provider_call"
            and item["effect_id"] == llm_activity["effect_id"]
        ]
        assert len(provider_activities) == 1
        assert (
            provider_activities[0]["provider_spec_sha256"]
            == llm_activity["provider_spec_sha256"]
        )
        edges = runtime.semantic_flow.list_edges(
            node_id=llm_activity["activity_id"],
            limit=100,
        )["items"]
        assert any(
            item["source_node_id"] == provider_id
            and item["target_node_id"] == llm_activity["activity_id"]
            and item["relation"] == "direct"
            for item in edges
        )
        coverage = runtime.semantic_flow.coverage(model_entity_id)
        assert coverage.effective_labels.sensitivity is DataSensitivity.SECRET
        assert coverage.assertion_count >= 1
        retained = json.dumps(
            {
                "provider": provider_items,
                "activities": activities,
                "edges": edges,
                "coverage": coverage.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        assert sentinel not in retained
    finally:
        runtime.close()

    reopened = SQLiteStore(database)
    try:
        persisted_flow = SemanticFlowService(reopened)
        persisted_lineage = persisted_flow.query_flow_lineage(
            model_entity_id,
            direction="upstream",
            limit=100,
            max_depth=8,
        )
        assert any(
            item["node_type"] == "entity"
            and item["node"] is not None
            and item["node"].get("entity_id") == provider_id
            for item in persisted_lineage["items"]
        )
        assert any(
            item["node_type"] == "entity"
            and item["node"] is not None
            and item["node"].get("entity_id") == root_entity_id
            for item in persisted_lineage["items"]
        )
        assert any(
            item["node_type"] == "entity"
            and item["node"] is not None
            and item["node"].get("kind") == "materialization"
            for item in persisted_lineage["items"]
        )
        assert sentinel not in json.dumps(
            persisted_lineage,
            ensure_ascii=False,
            sort_keys=True,
        )
    finally:
        reopened.close()


def test_runtime_file_capture_dlp_tightens_labels_without_retaining_secret(
    tmp_path: Path,
) -> None:
    sentinel = "SEMANTIC_FILE_DLP_SENTINEL_82d9"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relative = "reports/secret.txt"
    target = workspace / relative
    target.parent.mkdir(parents=True)
    target.write_text(f"password={sentinel}\n", encoding="utf-8")
    database = tmp_path / "runtime.db"
    runtime = Runtime.open(
        database,
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        pid = runtime.process.spawn(goal="Read one locally classified file.")
        runtime.capability.grant(
            subject=pid,
            resource=runtime.filesystem.resource_for(relative),
            rights=("read",),
            issued_by="test.host",
        )
        observed = runtime.filesystem.read_text(pid, relative)
        assert sentinel in observed.content

        page = runtime.semantic_flow.list_entities(
            pid=pid,
            kind="file_binding_version",
            limit=20,
        )
        assert page["items"]
        entity = page["items"][-1]
        effective = runtime.semantic_flow.effective_labels(entity["entity_id"])
        assert effective.labels.sensitivity is DataSensitivity.SECRET
        assert effective.assertion_count >= 1
        assert entity["coverage"] != "complete"
        retained = json.dumps(
            {
                "entities": page,
                "lineage": runtime.semantic_flow.query_flow_lineage(
                    entity["entity_id"],
                    limit=50,
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        assert sentinel not in retained
    finally:
        runtime.close()

    assert sentinel.encode("utf-8") not in database.read_bytes()


def test_capture_failure_health_is_digest_only_and_survives_reopen(
    tmp_path: Path,
) -> None:
    sentinel = "CAPTURE_FAILURE_SECRET_SENTINEL_b73f"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "capture-health.db"
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    runtime = Runtime.open(
        database,
        config=config,
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        runtime.semantic_runtime_flow.observe_memory(
            f"invalid-{sentinel}",
            payload=sentinel,
        )
        assert runtime.semantic_flow.status()["capture_failures"] >= 1
    finally:
        runtime.close()

    assert sentinel.encode("utf-8") not in database.read_bytes()
    reopened = Runtime.open(
        database,
        config=config,
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        assert reopened.semantic_flow.status()["capture_failures"] >= 1
    finally:
        reopened.close()


def test_sqlite_flow_graph_reopens_without_raw_identity_or_payload(tmp_path: Path) -> None:
    sentinel = "SEMANTIC_FLOW_SECRET_SENTINEL_7f33"
    path = tmp_path / "runtime.db"
    store = SQLiteStore(path)
    flow = SemanticFlowService(store)

    bundle = flow.capture_provider_ingress(
        pid="pid-1",
        effect_id="effect-1",
        action_id="mcp.call",
        result_sha256=_D1,
        state_sha256=_D2,
        provider_spec_sha256=_D3,
        tool_schema_sha256=_D4,
        labels=DataLabels(
            sensitivity="normal",
            trust_level="untrusted",
            integrity="untrusted",
            origin=sentinel,
            tenant=f"tenant-{sentinel}",
            principal=f"principal-{sentinel}",
        ),
        tenant_bucket_sha256=_TENANT,
        created_at="2026-08-07T00:00:00Z",
    )
    assert bundle is not None
    entity_id = bundle.entities[0].entity_id
    store.close()

    assert sentinel.encode() not in path.read_bytes()
    reopened = SQLiteStore(path)
    try:
        record = reopened.get_semantic_flow_entity(entity_id)
        assert record is not None
        assert dict(record.baseline_labels) == {
            "sensitivity": "normal",
            "trust_level": "untrusted",
            "integrity": "untrusted",
        }
        assert record.tenant_bucket_sha256 == _TENANT
    finally:
        reopened.close()


def test_sqlite_derived_lineage_and_memory_gate_are_bounded() -> None:
    store = SQLiteStore(":memory:")
    flow = SemanticFlowService(store)
    try:
        root = flow.capture_root_goal(
            pid="pid-1",
            goal_oid="goal-1",
            goal_version=1,
            content_sha256=_D1,
            state_sha256=_D2,
            provenance_sha256=_D3,
            labels=DataLabels(
                trust_level="user_asserted",
                integrity="checked",
                tenant="tenant-a",
            ),
            tenant_bucket_sha256=_TENANT,
            created_at="2026-08-07T00:00:00Z",
        )
        assert root is not None
        derived = flow.capture_tool_result(
            pid="pid-1",
            action_id="runtime.tool_result",
            content_sha256=_D2,
            version_sha256=_D3,
            state_sha256=_D4,
            provenance_sha256=_D5,
            labels=DataLabels(
                trust_level="user_asserted",
                integrity="checked",
                tenant="tenant-a",
            ),
            tenant_bucket_sha256=_TENANT,
            inputs=(
                FlowInputEdge(
                    FlowNodeRef(
                        root.entities[0].entity_id,
                        FlowNodeType.ENTITY,
                    ),
                    FlowEdgeRelation.DIRECT,
                ),
            ),
            tool_schema_sha256=_D1,
            created_at="2026-08-07T00:00:01Z",
        )
        assert derived is not None

        lineage = flow.query_flow_lineage(
            derived.entities[0].entity_id,
            direction="upstream",
            limit=20,
            max_depth=4,
        )
        assert any(
            item["node"] is not None
            and item["node"].get("entity_id") == root.entities[0].entity_id
            for item in lineage["items"]
        )
        gate = flow.memory_gate(
            root.entities[0].entity_id,
            target_tenant_bucket_sha256=_TENANT,
            relation="control",
        )
        assert gate.allowed
        assert gate.reason is FlowMemoryGateReason.ALLOW
    finally:
        store.close()


def test_sqlite_flow_assertion_tightens_labels_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "flow.db"
    store = SQLiteStore(path)
    flow = SemanticFlowService(store)
    root = flow.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(
            sensitivity="normal",
            trust_level="verified",
            integrity="verified",
        ),
        tenant_bucket_sha256=None,
        created_at="2026-08-07T00:00:00Z",
    )
    assert root is not None
    entity_id = root.entities[0].entity_id
    flow.append_assessment_findings(
        entity_id=entity_id,
        assessment_id="assessment-1",
        findings=(
            SemanticDataFinding(
                category=SemanticDataCategory.CREDENTIAL,
                field=SemanticDataLocator.ROOT_GOAL,
                span_start=None,
                span_end=None,
                sensitivity_floor="secret",
                integrity_ceiling="checked",
                trust_ceiling="user_asserted",
                confidence_bps=10_000,
                evidence_sha256=_D5,
            ),
        ),
        created_at="2026-08-07T00:00:01Z",
    )
    store.close()

    reopened = SQLiteStore(path)
    try:
        effective = SemanticFlowService(reopened).effective_labels(entity_id)
        assert effective.labels.sensitivity is DataSensitivity.SECRET
        assert effective.assertion_count == 1
        assert effective.coverage.value == "complete"
    finally:
        reopened.close()


def test_sqlite_flow_field_and_chunk_locators_reopen_payload_free(
    tmp_path: Path,
) -> None:
    sentinel = "FLOW_PRIVATE_JSON_KEY_SENTINEL_918a"
    path = tmp_path / "flow-locators.db"
    store = SQLiteStore(path)
    flow = SemanticFlowService(store)
    root = flow.capture_root_goal(
        pid="pid-locator",
        goal_oid="goal-locator",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(trust_level="verified", integrity="verified"),
        tenant_bucket_sha256=None,
        created_at="2026-08-07T00:00:00Z",
    )
    assert root is not None
    entity_id = root.entities[0].entity_id
    common = {
        "entity_id": entity_id,
        "source": "deterministic",
        "assessment_id": "assessment-locator",
    }
    json_locator = FlowDataLocator.json_field(
        ("record", sentinel, 7),
        value_sha256=_D4,
    )
    chunk_locator = FlowDataLocator.text_chunk(
        ordinal=3,
        offset_start=128,
        offset_end=160,
        content_sha256=_D5,
    )
    finding = SemanticDataFinding(
        category=SemanticDataCategory.BUSINESS_SECRET,
        field=SemanticDataLocator.ROOT_GOAL,
        span_start=None,
        span_end=None,
        sensitivity_floor="secret",
        integrity_ceiling="checked",
        trust_ceiling="user_asserted",
        confidence_bps=10_000,
        evidence_sha256=_D5,
    )
    flow.append_label_assertion(**common, finding=finding, locator=json_locator)
    flow.append_label_assertion(
        **common,
        finding=SemanticDataFinding(
            category=finding.category,
            field=finding.field,
            span_start=None,
            span_end=None,
            sensitivity_floor=finding.sensitivity_floor,
            integrity_ceiling=finding.integrity_ceiling,
            trust_ceiling=finding.trust_ceiling,
            confidence_bps=finding.confidence_bps,
            evidence_sha256=_D4,
        ),
        locator=chunk_locator,
    )
    store.close()

    assert sentinel.encode("utf-8") not in path.read_bytes()
    reopened = SQLiteStore(path)
    try:
        page = reopened.query_semantic_flow_label_assertions(
            entity_id=entity_id,
            source=None,
            after=None,
            limit=10,
        )
        assert len(page.records) == 2
        by_kind = {record.locator_kind: record for record in page.records}
        field_record = by_kind["json_field"]
        chunk_record = by_kind["text_chunk"]
        assert field_record.path_sha256s == json_locator.path_sha256s
        assert field_record.value_sha256 == _D4
        assert field_record.ordinal is None
        assert chunk_record.path_sha256s == ()
        assert chunk_record.value_sha256 == _D5
        assert (
            chunk_record.ordinal,
            chunk_record.offset_start,
            chunk_record.offset_end,
        ) == (3, 128, 160)
    finally:
        reopened.close()


def test_sqlite_flow_entity_keyset_cursor_has_no_duplicates() -> None:
    store = SQLiteStore(":memory:")
    flow = SemanticFlowService(store)
    try:
        for index, digest in enumerate((_D1, _D2, _D3)):
            flow.capture_provider_ingress(
                pid="pid-1",
                effect_id=f"effect-{index}",
                action_id="git.read",
                result_sha256=digest,
                state_sha256=_D4,
                provider_spec_sha256=_D5,
                tool_schema_sha256=None,
                labels=DataLabels(),
                tenant_bucket_sha256=None,
                created_at=f"2026-08-07T00:00:0{index}Z",
            )

        first = flow.query_flow_entities(limit=2)
        second = flow.query_flow_entities(limit=2, after=first["next_cursor"])

        ids = [item["entity_id"] for item in first["items"] + second["items"]]
        assert len(ids) == 3
        assert len(set(ids)) == 3
        assert second["next_cursor"] is None
    finally:
        store.close()


@pytest.mark.parametrize(
    "action_id,capture",
    (
        ("filesystem.read", "file"),
        ("git.read", "git"),
        ("git.diff", "git"),
    ),
)
def test_phase4_flow_eligibility_requires_exact_current_binding(
    action_id: str,
    capture: str,
) -> None:
    store = SQLiteStore(":memory:")
    flow = SemanticFlowService(store)
    facts = {
        "pid": "pid-1",
        "action_id": action_id,
        "content_sha256": _D1,
        "version_sha256": _D2,
        "state_sha256": _D3,
        "provenance_sha256": _D4,
        "labels": DataLabels(
            sensitivity="normal",
            trust_level="unknown",
            integrity="unknown",
            tenant="tenant-a",
        ),
        "tenant_bucket_sha256": _TENANT,
        "created_at": "2026-08-07T00:00:00Z",
    }
    try:
        if capture == "file":
            bundle = flow.capture_file_version(operation="read", **facts)
        else:
            bundle = flow.capture_git_snapshot(**facts)
        assert bundle is not None
        entity_id = bundle.entities[0].entity_id

        exact = flow.approval_eligibility(
            action_id=action_id,
            entity_id=entity_id,
            tenant_bucket_sha256=_TENANT,
            current_content_sha256=_D1,
            current_version_sha256=_D2,
            current_state_sha256=_D3,
        )
        stale = flow.approval_eligibility(
            action_id=action_id,
            entity_id=entity_id,
            tenant_bucket_sha256=_TENANT,
            current_content_sha256=_D5,
            current_version_sha256=_D2,
            current_state_sha256=_D3,
        )

        assert exact.eligible
        assert exact.reason_codes == (FlowEligibilityReason.ELIGIBLE,)
        assert not stale.eligible
        assert FlowEligibilityReason.CONTENT_DRIFT in stale.reason_codes
    finally:
        store.close()


@pytest.mark.parametrize("coverage", ("partial", "unknown", "conflict", "stale"))
def test_phase4_flow_eligibility_rejects_every_non_complete_coverage(
    coverage: str,
) -> None:
    store = SQLiteStore(":memory:")
    flow = SemanticFlowService(store)
    try:
        bundle = flow.capture_file_version(
            operation="read",
            pid="pid-1",
            action_id="filesystem.read",
            content_sha256=_D1,
            version_sha256=_D2,
            state_sha256=_D3,
            provenance_sha256=_D4,
            labels=DataLabels(tenant="tenant-a"),
            tenant_bucket_sha256=_TENANT,
            coverage=coverage,
            created_at="2026-08-07T00:00:00Z",
        )
        assert bundle is not None

        decision = flow.approval_eligibility(
            action_id="filesystem.read",
            entity_id=bundle.entities[0].entity_id,
            tenant_bucket_sha256=_TENANT,
            current_content_sha256=_D1,
            current_version_sha256=_D2,
            current_state_sha256=_D3,
        )

        assert not decision.eligible
        assert FlowEligibilityReason.COVERAGE_INCOMPLETE in decision.reason_codes
        assert decision.coverage.value == coverage
    finally:
        store.close()
