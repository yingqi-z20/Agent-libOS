from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from agent_libos.models import (
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import (
    SemanticApprovalBindingV2,
    SemanticDataCategory,
    SemanticDataFinding,
    SemanticDataLocator,
    SemanticTripCode,
)
from agent_libos.config import SemanticDefaults
from agent_libos.semantic.flow import (
    FLOW_NO_TENANT_BUCKET_SHA256,
    FLOW_UNBUCKETED_IDENTITY_SHA256,
    FlowActivityKind,
    FlowCoverageStatus,
    FlowDataLocator,
    FlowEdgeRelation,
    FlowEligibilityReason,
    FlowEntityKind,
    FlowInputEdge,
    FlowLabelSource,
    FlowLabelVector,
    FlowMemoryGateReason,
    FlowNodeRef,
    FlowNodeType,
    SemanticFlowProvenanceValidator,
    SemanticFlowService,
    compute_effective_labels,
    flow_record_to_dict,
)
from agent_libos.semantic.enforcement import SemanticSafetyTripRequired
from agent_libos.semantic.service import SemanticManager
from agent_libos.storage.semantic_v6 import (
    SemanticFlowActivityRecord,
    SemanticFlowBundle,
    SemanticFlowEdgeRecord,
    SemanticFlowEntityRecord,
    SemanticFlowLabelAssertionRecord,
    SemanticFlowPage,
    records_page,
)


_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64
_D4 = "4" * 64
_D5 = "5" * 64
_NOW = "2026-08-07T00:00:00Z"


class _FlowRepository:
    def __init__(self) -> None:
        self.entities: dict[str, SemanticFlowEntityRecord] = {}
        self.activities: dict[str, SemanticFlowActivityRecord] = {}
        self.edges: dict[str, SemanticFlowEdgeRecord] = {}
        self.assertions: dict[str, SemanticFlowLabelAssertionRecord] = {}

    def append_semantic_flow_bundle(
        self,
        *,
        entities: tuple[SemanticFlowEntityRecord, ...] = (),
        activities: tuple[SemanticFlowActivityRecord, ...] = (),
        edges: tuple[SemanticFlowEdgeRecord, ...] = (),
        assertions: tuple[SemanticFlowLabelAssertionRecord, ...] = (),
    ) -> SemanticFlowBundle:
        for record in (*entities, *activities, *edges, *assertions):
            if isinstance(record, SemanticFlowEntityRecord):
                selected = self.entities
                identity = record.entity_id
            elif isinstance(record, SemanticFlowActivityRecord):
                selected = self.activities
                identity = record.activity_id
            elif isinstance(record, SemanticFlowEdgeRecord):
                selected = self.edges
                identity = record.edge_id
            else:
                selected = self.assertions
                identity = record.assertion_id
            if identity in selected and selected[identity] != record:
                raise ValidationError("append-only flow identity conflict")
            selected[identity] = record
        return SemanticFlowBundle(entities, activities, edges, assertions)

    def get_semantic_flow_entity(
        self, entity_id: str
    ) -> SemanticFlowEntityRecord | None:
        return self.entities.get(entity_id)

    def get_semantic_flow_activity(
        self, activity_id: str
    ) -> SemanticFlowActivityRecord | None:
        return self.activities.get(activity_id)

    def query_semantic_flow_entities(self, *, limit: int, **_filters: object):
        return records_page(
            sorted(self.entities.values(), key=lambda item: (item.created_at, item.entity_id)),
            limit=limit,
            id_field="entity_id",
        )

    def query_semantic_flow_activities(self, *, limit: int, **_filters: object):
        return records_page(
            sorted(self.activities.values(), key=lambda item: (item.created_at, item.activity_id)),
            limit=limit,
            id_field="activity_id",
        )

    def query_semantic_flow_edges(
        self,
        *,
        limit: int,
        node_id: str | None = None,
        **_filters: object,
    ):
        records = tuple(
            record
            for record in sorted(
                self.edges.values(), key=lambda item: (item.created_at, item.edge_id)
            )
            if node_id is None
            or record.source_node_id == node_id
            or record.target_node_id == node_id
        )
        return records_page(records, limit=limit, id_field="edge_id")

    def query_semantic_flow_label_assertions(
        self,
        *,
        limit: int,
        entity_id: str | None = None,
        **_filters: object,
    ) -> SemanticFlowPage:
        records = tuple(
            record
            for record in sorted(
                self.assertions.values(),
                key=lambda item: (item.created_at, item.assertion_id),
            )
            if entity_id is None or record.entity_id == entity_id
        )
        return records_page(records, limit=limit, id_field="assertion_id")


def _finding(
    *,
    sensitivity: DataSensitivity = DataSensitivity.SECRET,
    integrity: DataIntegrity = DataIntegrity.CHECKED,
    trust: DataTrustLevel = DataTrustLevel.USER_ASSERTED,
    evidence: str = _D5,
) -> SemanticDataFinding:
    return SemanticDataFinding(
        category=SemanticDataCategory.BUSINESS_SECRET,
        field=SemanticDataLocator.PROVIDER_RESULT,
        span_start=None,
        span_end=None,
        sensitivity_floor=sensitivity,
        integrity_ceiling=integrity,
        trust_ceiling=trust,
        confidence_bps=9_900,
        evidence_sha256=evidence,
    )


def test_flow_label_vector_is_identity_safe_and_tightens_monotonically() -> None:
    labels = DataLabels(
        sensitivity="normal",
        trust_level="verified",
        integrity="verified",
        origin="SECRET_SENTINEL",
        tenant="tenant-secret",
        principal="principal-secret",
        declassification_authority="release-secret",
    )
    vector = FlowLabelVector.from_data_labels(labels)

    assert vector.to_dict() == {
        "sensitivity": "normal",
        "trust_level": "verified",
        "integrity": "verified",
    }
    assert "SECRET_SENTINEL" not in json.dumps(vector.to_dict())
    assert "tenant-secret" not in json.dumps(vector.to_dict())

    assertion = SemanticFlowLabelAssertionRecord(
        assertion_id="assertion-1",
        entity_id="entity-1",
        source="model",
        sensitivity_floor="secret",
        integrity_ceiling="checked",
        trust_ceiling="user_asserted",
        evidence_sha256=_D1,
        assessment_id="assessment-1",
        locator_sha256=None,
        category="business_secret",
        coverage="complete",
        created_at=_NOW,
    )
    effective = compute_effective_labels(vector, (assertion,))

    assert effective.labels == FlowLabelVector(
        sensitivity="secret",
        integrity="checked",
        trust_level="user_asserted",
    )
    assert effective.coverage is FlowCoverageStatus.COMPLETE
    assert effective.conflict_count == 0


def test_effective_labels_marks_declassification_or_endorsement_conflict() -> None:
    baseline = FlowLabelVector(
        sensitivity="confidential",
        trust_level="unknown",
        integrity="unknown",
    )
    invalid = SemanticFlowLabelAssertionRecord(
        assertion_id="assertion-invalid",
        entity_id="entity-1",
        source="model",
        sensitivity_floor="public",
        integrity_ceiling="verified",
        trust_ceiling="trusted",
        evidence_sha256=_D1,
        assessment_id=None,
        locator_sha256=None,
        category=None,
        coverage="complete",
        created_at=_NOW,
    )

    result = compute_effective_labels(baseline, (invalid,))

    assert result.labels == baseline
    assert result.coverage is FlowCoverageStatus.CONFLICT
    assert result.conflict_count == 1


def test_json_and_text_locators_persist_only_digests_and_coordinates() -> None:
    locator = FlowDataLocator.json_field(
        ("customer", "private_value", 3),
        value_sha256=_D1,
    )
    encoded = json.dumps(locator.to_dict(), sort_keys=True)

    assert "customer" not in encoded
    assert "private_value" not in encoded
    assert locator.value_sha256 == _D1
    assert len(locator.path_sha256s) == 3
    assert all(len(item) == 64 for item in locator.path_sha256s)

    chunk = FlowDataLocator.text_chunk(
        ordinal=2,
        offset_start=64,
        offset_end=128,
        content_sha256=_D2,
    )
    assert chunk.to_dict()["ordinal"] == 2
    assert chunk.to_dict()["value_sha256"] == _D2
    assert chunk.path_sha256s == ()
    with pytest.raises(ValidationError, match="end must exceed"):
        FlowDataLocator.text_chunk(
            ordinal=0,
            offset_start=2,
            offset_end=2,
            content_sha256=_D2,
        )


def test_derived_activity_identity_binds_content_version_and_provenance() -> None:
    repository = _FlowRepository()
    flow = SemanticFlowService(repository)
    common = {
        "operation": "create",
        "pid": "pid-1",
        "action_id": "memory.object.create",
        "state_sha256": _D3,
        "labels": DataLabels(),
        "tenant_bucket_sha256": None,
        "created_at": _NOW,
    }

    first = flow.capture_object_version(
        **common,
        content_sha256=_D1,
        version_sha256=_D2,
        provenance_sha256=_D4,
    )
    second = flow.capture_object_version(
        **common,
        content_sha256=_D4,
        version_sha256=_D5,
        provenance_sha256=_D1,
    )

    assert first is not None and second is not None
    assert first.activities[0].activity_id != second.activities[0].activity_id
    assert first.entities[0].entity_id != second.entities[0].entity_id
    output_edges = {
        (edge.source_node_id, edge.target_node_id)
        for edge in repository.edges.values()
        if edge.source_node_type == "activity"
        and edge.target_node_type == "entity"
    }
    assert output_edges == {
        (first.activities[0].activity_id, first.entities[0].entity_id),
        (second.activities[0].activity_id, second.entities[0].entity_id),
    }


def test_root_and_provider_capture_are_payload_free_idempotent_bundles() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    labels = DataLabels(
        trust_level="untrusted",
        integrity="untrusted",
        origin="SECRET_SENTINEL",
    )

    first = service.capture_root_goal(
        pid="pid-1",
        goal_oid="SECRET_SENTINEL-goal",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=labels,
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    repeated = service.capture_root_goal(
        pid="pid-1",
        goal_oid="SECRET_SENTINEL-goal",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=labels,
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    provider = service.capture_provider_ingress(
        pid="pid-1",
        effect_id="effect-1",
        action_id="mcp.call",
        result_sha256=_D4,
        state_sha256=_D5,
        provider_spec_sha256=_D1,
        tool_schema_sha256=_D2,
        labels=labels,
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )

    assert first is not None and repeated is not None and provider is not None
    assert first.entities[0].entity_id == repeated.entities[0].entity_id
    assert len(repository.entities) == 2
    assert len(repository.activities) == 2
    assert len(repository.edges) == 2
    persisted = json.dumps(
        [flow_record_to_dict(item) for item in (
            *repository.entities.values(),
            *repository.activities.values(),
            *repository.edges.values(),
        )],
        sort_keys=True,
    )
    assert "SECRET_SENTINEL" not in persisted
    assert first.entities[0].tenant_bucket_sha256 == FLOW_NO_TENANT_BUCKET_SHA256
    assert provider.entities[0].coverage == "complete"


def test_root_goal_object_binding_is_exact_idempotent_and_rejects_ambiguity() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    labels = DataLabels(tenant="tenant-a")
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=labels,
        tenant_bucket_sha256=_D4,
        created_at=_NOW,
    )
    initial = service.capture_object_version(
        operation="create",
        pid="pid-1",
        action_id="memory.object.create",
        content_sha256=_D1,
        version_sha256=_D2,
        state_sha256=_D3,
        provenance_sha256=_D4,
        labels=labels,
        tenant_bucket_sha256=_D4,
        created_at="2026-08-07T00:00:01Z",
    )
    assert root is not None and initial is not None
    facts = {
        "pid": "pid-1",
        "root_entity_id": root.entities[0].entity_id,
        "object_entity_id": initial.entities[0].entity_id,
        "root_state_sha256": _D2,
        "object_content_sha256": _D1,
        "object_version_sha256": _D2,
        "object_provenance_sha256": _D4,
        "tenant_bucket_sha256": _D4,
    }

    first = service.bind_root_goal_object(**facts)
    repeated = service.bind_root_goal_object(**facts)

    assert first is not None and repeated is not None
    assert first.activities[0] == repeated.activities[0]
    assert first.edges == repeated.edges
    bindings = [
        item
        for item in repository.activities.values()
        if item.action_id == "runtime.root_goal_object_binding"
    ]
    assert len(bindings) == 1
    assert root.entities[0].baseline_labels == initial.entities[0].baseline_labels

    with pytest.raises(ValidationError, match="version digest changed"):
        service.bind_root_goal_object(
            **{**facts, "object_version_sha256": _D5}
        )

    duplicate = service.capture_object_version(
        operation="read",
        pid="pid-1",
        action_id="memory.object.read",
        content_sha256=_D1,
        version_sha256=_D5,
        state_sha256=_D3,
        provenance_sha256=_D4,
        labels=labels,
        tenant_bucket_sha256=_D4,
        created_at="2026-08-07T00:00:02Z",
    )
    assert duplicate is not None
    with pytest.raises(ValidationError, match="already bound"):
        service.bind_root_goal_object(
            **{
                **facts,
                "object_entity_id": duplicate.entities[0].entity_id,
                "object_version_sha256": _D5,
            }
        )


def test_root_goal_object_binding_rejects_cross_tenant_target() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(tenant="tenant-a"),
        tenant_bucket_sha256=_D4,
        created_at=_NOW,
    )
    foreign = service.capture_object_version(
        operation="create",
        pid="pid-1",
        action_id="memory.object.create",
        content_sha256=_D1,
        version_sha256=_D2,
        state_sha256=_D3,
        provenance_sha256=_D4,
        labels=DataLabels(tenant="tenant-b"),
        tenant_bucket_sha256=_D5,
        created_at="2026-08-07T00:00:01Z",
    )
    assert root is not None and foreign is not None

    with pytest.raises(ValidationError, match="cross tenant"):
        service.bind_root_goal_object(
            pid="pid-1",
            root_entity_id=root.entities[0].entity_id,
            object_entity_id=foreign.entities[0].entity_id,
            root_state_sha256=_D2,
            object_content_sha256=_D1,
            object_version_sha256=_D2,
            object_provenance_sha256=_D4,
            tenant_bucket_sha256=_D4,
        )
    assert not any(
        item.action_id == "runtime.root_goal_object_binding"
        for item in repository.activities.values()
    )


def test_unbucketed_identity_is_unknown_and_blocks_memory_influence() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    labels = DataLabels(
        trust_level="verified",
        integrity="verified",
        tenant="raw-tenant-must-not-persist",
    )
    bundle = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=labels,
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    assert bundle is not None
    entity = bundle.entities[0]

    assert entity.tenant_bucket_sha256 == FLOW_UNBUCKETED_IDENTITY_SHA256
    assert entity.coverage == "unknown"
    gate = service.memory_gate(
        entity.entity_id,
        target_tenant_bucket_sha256=FLOW_UNBUCKETED_IDENTITY_SHA256,
        relation=FlowEdgeRelation.CONTROL,
    )
    assert not gate.allowed
    assert gate.reason is FlowMemoryGateReason.UNKNOWN_COVERAGE


def test_derived_capture_rejects_cross_tenant_input_before_append() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    source = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(tenant="tenant-a"),
        tenant_bucket_sha256=_D4,
        created_at=_NOW,
    )
    assert source is not None

    with pytest.raises(ValidationError, match="cross tenant"):
        service.capture_tool_result(
            pid="pid-1",
            action_id="runtime.tool_result",
            content_sha256=_D2,
            version_sha256=_D3,
            state_sha256=_D4,
            provenance_sha256=_D5,
            labels=DataLabels(tenant="tenant-b"),
            tenant_bucket_sha256=_D5,
            inputs=(
                FlowInputEdge(
                    source=FlowNodeRef(
                        source.entities[0].entity_id,
                        FlowNodeType.ENTITY,
                    ),
                    relation=FlowEdgeRelation.DIRECT,
                ),
            ),
            tool_schema_sha256=_D1,
            created_at=_NOW,
        )
    assert len(repository.entities) == 1


@pytest.mark.parametrize(
    "method,extra,expected_entity,expected_activity,relation",
    (
        (
            "capture_object_version",
            {"operation": "update"},
            FlowEntityKind.OBJECT_VERSION,
            FlowActivityKind.OBJECT_UPDATE,
            FlowEdgeRelation.DIRECT,
        ),
        (
            "capture_file_version",
            {"operation": "read"},
            FlowEntityKind.FILE_BINDING_VERSION,
            FlowActivityKind.FILE_READ,
            FlowEdgeRelation.INDIRECT,
        ),
        (
            "capture_tool_result",
            {"tool_schema_sha256": _D1},
            FlowEntityKind.TOOL_RESULT,
            FlowActivityKind.TOOL_CALL,
            FlowEdgeRelation.CONTROL,
        ),
        (
            "capture_materialization",
            {},
            FlowEntityKind.MATERIALIZATION,
            FlowActivityKind.OBJECT_MATERIALIZE,
            FlowEdgeRelation.DIRECT,
        ),
        (
            "capture_model_output",
            {"model_artifact_sha256": _D2},
            FlowEntityKind.MODEL_OUTPUT,
            FlowActivityKind.LLM_CALL,
            FlowEdgeRelation.INDIRECT,
        ),
        (
            "capture_git_snapshot",
            {"provider_spec_sha256": _D3},
            FlowEntityKind.FILE_BINDING_VERSION,
            FlowActivityKind.PROVIDER_CALL,
            FlowEdgeRelation.CONTROL,
        ),
    ),
)
def test_generic_capture_catalog_is_payload_free_and_relation_typed(
    method: str,
    extra: dict[str, str],
    expected_entity: FlowEntityKind,
    expected_activity: FlowActivityKind,
    relation: FlowEdgeRelation,
) -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(),
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    assert root is not None

    capture = getattr(service, method)
    bundle = capture(
        pid="pid-1",
        action_id=f"test.{method}",
        content_sha256=_D2,
        version_sha256=_D3,
        state_sha256=_D4,
        provenance_sha256=_D5,
        labels=DataLabels(),
        tenant_bucket_sha256=None,
        inputs=(
            FlowInputEdge(
                FlowNodeRef(root.entities[0].entity_id, FlowNodeType.ENTITY),
                relation,
            ),
        ),
        created_at="2026-08-07T00:00:01Z",
        **extra,
    )

    assert bundle is not None
    assert bundle.entities[0].kind == expected_entity.value
    assert bundle.entities[0].coverage == FlowCoverageStatus.COMPLETE.value
    assert bundle.activities[0].kind == expected_activity.value
    assert [edge.relation for edge in bundle.edges] == [
        relation.value,
        FlowEdgeRelation.DIRECT.value,
    ]


def test_append_assessment_findings_is_monotonic_and_bounded() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(
            sensitivity="normal",
            integrity="verified",
            trust_level="verified",
        ),
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    assert root is not None
    entity_id = root.entities[0].entity_id

    records = service.append_assessment_findings(
        entity_id=entity_id,
        assessment_id="assessment-1",
        findings=(_finding(),),
        source=FlowLabelSource.MODEL,
        created_at=_NOW,
    )

    assert len(records) == 1
    effective = service.effective_labels(entity_id)
    assert effective.labels.sensitivity is DataSensitivity.SECRET
    assert effective.labels.integrity is DataIntegrity.CHECKED
    assert effective.labels.trust_level is DataTrustLevel.USER_ASSERTED

    weakening = replace(
        _finding(),
        sensitivity_floor=DataSensitivity.PUBLIC,
        integrity_ceiling=DataIntegrity.VERIFIED,
        trust_ceiling=DataTrustLevel.TRUSTED,
    )
    with pytest.raises(ValidationError, match="cannot declassify or endorse"):
        service.append_assessment_findings(
            entity_id=entity_id,
            assessment_id="assessment-2",
            findings=(weakening,),
            created_at=_NOW,
        )


def test_effective_labels_fail_closed_when_assertion_page_is_truncated() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(),
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    assert root is not None
    entity_id = root.entities[0].entity_id
    for index in range(257):
        assertion_id = f"assertion-{index:03d}"
        repository.assertions[assertion_id] = SemanticFlowLabelAssertionRecord(
            assertion_id=assertion_id,
            entity_id=entity_id,
            source="deterministic",
            sensitivity_floor="normal",
            integrity_ceiling="unknown",
            trust_ceiling="unknown",
            evidence_sha256=hashlib.sha256(assertion_id.encode("utf-8")).hexdigest(),
            assessment_id=None,
            locator_sha256=None,
            category=None,
            coverage="complete",
            created_at=_NOW,
        )

    effective = service.effective_labels(entity_id)

    assert effective.coverage is FlowCoverageStatus.PARTIAL
    assert effective.assertion_count == 256


def test_phase4_eligibility_joins_upstream_monotonic_labels() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    provider = service.capture_provider_ingress(
        pid="pid-1",
        effect_id="effect-upstream",
        action_id="git.diff",
        result_sha256=_D1,
        state_sha256=_D2,
        provider_spec_sha256=_D3,
        tool_schema_sha256=None,
        labels=DataLabels(
            sensitivity="normal",
            trust_level="unknown",
            integrity="unknown",
            tenant="tenant-a",
        ),
        tenant_bucket_sha256=_D5,
        created_at=_NOW,
    )
    assert provider is not None
    service.append_label_assertion(
        entity_id=provider.entities[0].entity_id,
        finding=SemanticDataFinding(
            category=SemanticDataCategory.CREDENTIAL,
            field=SemanticDataLocator.PROVIDER_RESULT,
            span_start=None,
            span_end=None,
            sensitivity_floor="secret",
            integrity_ceiling="unknown",
            trust_ceiling="unknown",
            confidence_bps=10_000,
            evidence_sha256=_D4,
        ),
        source=FlowLabelSource.HOST,
        assessment_id=None,
        created_at="2026-08-07T00:00:01Z",
    )
    snapshot = service.capture_git_snapshot(
        pid="pid-1",
        action_id="git.diff",
        content_sha256=_D2,
        version_sha256=_D3,
        state_sha256=_D4,
        provenance_sha256=_D5,
        labels=DataLabels(
            sensitivity="normal",
            trust_level="unknown",
            integrity="unknown",
            tenant="tenant-a",
        ),
        tenant_bucket_sha256=_D5,
        inputs=(
            FlowInputEdge(
                FlowNodeRef(
                    provider.entities[0].entity_id,
                    FlowNodeType.ENTITY,
                ),
                FlowEdgeRelation.DIRECT,
            ),
        ),
        provider_spec_sha256=_D3,
        created_at="2026-08-07T00:00:02Z",
    )
    assert snapshot is not None

    coverage = service.coverage(snapshot.entities[0].entity_id)
    decision = service.approval_eligibility(
        action_id="git.diff",
        entity_id=snapshot.entities[0].entity_id,
        tenant_bucket_sha256=_D5,
        current_content_sha256=_D2,
        current_version_sha256=_D3,
        current_state_sha256=_D4,
    )

    assert coverage.effective_labels.sensitivity is DataSensitivity.SECRET
    assert coverage.assertion_count == 1
    assert not decision.eligible
    assert FlowEligibilityReason.LABEL_TOO_SENSITIVE in decision.reason_codes


def test_flow_queries_are_keyset_bounded() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    service.capture_provider_ingress(
        pid="pid-1",
        effect_id="effect-1",
        action_id="git.read",
        result_sha256=_D1,
        state_sha256=_D2,
        provider_spec_sha256=_D3,
        tool_schema_sha256=_D4,
        labels=DataLabels(),
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )

    page = service.query_flow_entities(limit=1)
    lineage = service.query_flow_lineage(page["items"][0]["entity_id"], limit=10)

    assert len(page["items"]) == 1
    assert lineage["schema_version"] == 1
    assert lineage["coverage"] == "complete"
    with pytest.raises(ValidationError, match="between 1"):
        service.query_flow_entities(limit=0)


def test_lineage_cycle_is_bounded_and_coverage_is_conflict() -> None:
    repository = _FlowRepository()
    service = SemanticFlowService(repository)
    root = service.capture_root_goal(
        pid="pid-1",
        goal_oid="goal-1",
        goal_version=1,
        content_sha256=_D1,
        state_sha256=_D2,
        provenance_sha256=_D3,
        labels=DataLabels(),
        tenant_bucket_sha256=None,
        created_at=_NOW,
    )
    assert root is not None
    entity = root.entities[0]
    activity = root.activities[0]
    repository.append_semantic_flow_bundle(
        edges=(
            SemanticFlowEdgeRecord(
                edge_id="edge-cycle",
                relation="direct",
                source_node_id=activity.activity_id,
                source_node_type="activity",
                target_node_id=entity.entity_id,
                target_node_type="entity",
                pid="pid-1",
                provenance_sha256=_D5,
                created_at="2026-08-07T00:00:01Z",
            ),
        )
    )

    lineage = service.query_flow_lineage(
        entity.entity_id,
        direction="upstream",
        limit=20,
        max_depth=16,
    )
    coverage = service.coverage(entity.entity_id, max_depth=16)

    assert len(lineage["items"]) == 2
    assert lineage["coverage"] == "conflict"
    assert coverage.status is FlowCoverageStatus.CONFLICT


def test_flow_capture_failure_is_counted_exactly_once() -> None:
    class _FailingRepository(_FlowRepository):
        def append_semantic_flow_bundle(self, **_values: object):
            raise RuntimeError("injected flow append failure")

    service = SemanticFlowService(_FailingRepository())

    with pytest.raises(RuntimeError, match="injected"):
        service.capture_provider_ingress(
            pid="pid-1",
            effect_id="effect-1",
            action_id="git.read",
            result_sha256=_D1,
            state_sha256=_D2,
            provider_spec_sha256=_D3,
            tool_schema_sha256=_D4,
            labels=DataLabels(),
            tenant_bucket_sha256=None,
            created_at=_NOW,
        )

    assert service.flow_status()["capture_failures"] == 1


def test_flow_status_preserves_legacy_v5_history_as_unknown_coverage() -> None:
    class _LegacyRepository(_FlowRepository):
        @staticmethod
        def semantic_flow_status_aggregate() -> dict[str, object]:
            return {
                "counts": {},
                "coverage": {},
                "capture_failures": 0,
                "legacy_history": {
                    "present": True,
                    "source_schema_version": 5,
                    "assessment_count": 17,
                    "coverage": "unknown",
                    "evidence_sha256": _D1,
                    "created_at": _NOW,
                },
            }

    status = SemanticFlowService(_LegacyRepository()).flow_status()

    assert status["coverage"]["complete"] == 0
    assert status["coverage"]["unknown"] == 0
    assert status["legacy_history"] == {
        "present": True,
        "source_schema_version": 5,
        "assessment_count": 17,
        "coverage": "unknown",
        "evidence_sha256": _D1,
        "created_at": _NOW,
    }


def test_off_mode_never_invokes_flow_capture() -> None:
    repository = _FlowRepository()
    flow = SemanticFlowService(repository)
    manager = SemanticManager(
        repository,
        config=SemanticDefaults(mode="off"),
        flow_graph=flow,
    )

    assert manager.capture_root_goal("pid-1", goal={"text": "not captured"}) is None
    assert not repository.entities
    assert not repository.activities
    assert not repository.edges


def test_phase4_provenance_validator_rechecks_live_binding_and_snapshot() -> None:
    repository = _FlowRepository()
    flow = SemanticFlowService(repository)
    bundle = flow.capture_file_version(
        operation="read",
        pid="pid-1",
        action_id="filesystem.read",
        content_sha256=_D1,
        version_sha256=_D2,
        state_sha256=_D3,
        provenance_sha256=_D4,
        labels=DataLabels(
            sensitivity="normal",
            trust_level="unknown",
            integrity="unknown",
            tenant="tenant-a",
        ),
        tenant_bucket_sha256=_D5,
        created_at=_NOW,
    )
    assert bundle is not None
    entity_id = bundle.entities[0].entity_id
    live = {
        "entity_id": entity_id,
        "current_content_sha256": _D1,
        "current_version_sha256": _D2,
        "current_state_sha256": _D3,
        "source_labels_sha256": _D3,
        "source_refs_sha256": _D4,
    }
    validator = SemanticFlowProvenanceValidator(
        flow,
        live_binding_resolver=lambda **_facts: dict(live),
    )
    snapshot = validator.snapshot(
        action_id="filesystem.read",
        tenant_bucket_sha256=_D5,
        entity_id=live["entity_id"],
        current_content_sha256=live["current_content_sha256"],
        current_version_sha256=live["current_version_sha256"],
        current_state_sha256=live["current_state_sha256"],
    )
    assert snapshot.eligible
    binding = SemanticApprovalBindingV2(
        request_id="request-flow-v2",
        request_revision=0,
        pid="pid-1",
        operation_id="operation-flow-v2",
        effect_id="effect-flow-v2",
        authority_operation="filesystem.read",
        resource="filesystem:workspace:reports/result.txt",
        right="read",
        canonical_args_hash=_D1,
        target_state_version="file-version-1",
        manifest_id="manifest-flow-v2",
        manifest_sha256=_D2,
        ceiling_sha256=_D3,
        policy_epoch_id="epoch-flow-v2",
        policy_epoch_sha256=_D4,
        control_generation=1,
        assessment_id="assessment-flow-v2",
        assessment_sha256=_D5,
        classifier_profile_sha256=_D1,
        classifier_model_sha256=_D2,
        tenant_bucket_sha256=_D5,
        source_labels_sha256=_D3,
        source_refs_sha256=_D4,
        flow_snapshot_sha256=snapshot.canonical_sha256(),
        sink_identity_sha256=None,
        tool_schema_sha256=None,
        provider_spec_sha256=None,
        nonce="nonce-flow-v2",
        issued_at="2026-08-07T00:00:00Z",
        expires_at="2026-08-07T00:01:00Z",
    )

    validator(
        binding=binding,
        phase="dispatch",
        capability=object(),
        context={},
        effect_id=binding.effect_id,
        control=object(),
        epoch=object(),
    )

    with pytest.raises(SemanticSafetyTripRequired) as trip:
        validator(
            binding=replace(binding, flow_snapshot_sha256=_D1),
            phase="dispatch",
            capability=object(),
            context={},
            effect_id=binding.effect_id,
            control=object(),
            epoch=object(),
        )
    assert trip.value.trip_code is SemanticTripCode.BINDING_MISMATCH

    for key in ("source_labels_sha256", "source_refs_sha256"):
        original = live[key]
        live[key] = _D1 if original != _D1 else _D2
        with pytest.raises(SemanticSafetyTripRequired) as source_trip:
            validator(
                binding=binding,
                phase="dispatch",
                capability=object(),
                context={},
                effect_id=binding.effect_id,
                control=object(),
                epoch=object(),
            )
        assert source_trip.value.trip_code is SemanticTripCode.BINDING_MISMATCH
        live[key] = original

    live["current_content_sha256"] = _D4
    with pytest.raises(CapabilityDenied, match="incomplete, stale, or ineligible"):
        validator(
            binding=binding,
            phase="dispatch",
            capability=object(),
            context={},
            effect_id=binding.effect_id,
            control=object(),
            epoch=object(),
        )
