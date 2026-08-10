from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest

from agent_libos.config import SemanticDefaults
from agent_libos.models import DataFlowContext, DataLabels, DataSourceRef
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
    SemanticAssessment,
    SemanticAssessmentKind,
    SemanticAssessmentRequest,
    SemanticAssessmentStatus,
    SemanticDomain,
)
from agent_libos.semantic.service import (
    SemanticManager,
    _identity_safe_labels_sha256,
    _sha256,
)
from agent_libos.semantic.external import HostSemanticAssessmentInvocation
from agent_libos.storage import (
    SQLiteStore,
    SemanticAssessmentCursor,
    SemanticAssessmentJobRecord,
    SemanticAssessmentJobStatus,
    SemanticAssessmentRecord,
    SemanticProjectionRetention,
    SemanticStatusAggregate,
    UnitOfWork,
)


_DIGEST = "1" * 64


def _job(job_id: str = "job-1", *, created_at: str = "2026-01-01T00:00:00Z") -> SemanticAssessmentJobRecord:
    projection = {"action_id": "filesystem.read", "resource_count": 1}
    projection_sha256 = hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SemanticAssessmentJobRecord(
        job_id=job_id,
        kind="approval",
        status=SemanticAssessmentJobStatus.QUEUED,
        domain="filesystem",
        pid="pid-1",
        request_id=f"request-{job_id}",
        operation_id=f"operation-{job_id}",
        bindings={
            "artifact_sha256": _DIGEST,
            "input_sha256": _DIGEST,
            "feature_snapshot_sha256": _DIGEST,
            "policy_sha256": _DIGEST,
            "manifest_sha256": _DIGEST,
            "action_sha256": _DIGEST,
            "resource_sha256": _DIGEST,
            "args_sha256": _DIGEST,
            "state_sha256": _DIGEST,
            "sink_identity_sha256": None,
            "tenant_bucket_sha256": _DIGEST,
        },
        projection=projection,
        projection_sha256=projection_sha256,
        projection_retention=SemanticProjectionRetention.REDACTED,
        projection_expires_at="2026-01-01T02:00:00Z",
        created_at=created_at,
        updated_at=created_at,
    )


def _assessment(
    job: SemanticAssessmentJobRecord,
    *,
    assessment_id: str | None = None,
) -> SemanticAssessmentRecord:
    return SemanticAssessmentRecord(
        assessment_id=assessment_id or f"assessment-{job.job_id}",
        job_id=job.job_id,
        kind=job.kind,
        status="success",
        domain=job.domain,
        action_id="filesystem.read",
        pid=job.pid,
        request_id=job.request_id,
        operation_id=job.operation_id,
        effect_id=job.effect_id,
        shadow_outcome="require_human",
        reason_codes=("abstained",),
        ood=False,
        abstain=False,
        confidence_bps=7500,
        calibration_bucket="high",
        classifier_id="scripted",
        classifier_version="1",
        artifact_sha256=_DIGEST,
        input_sha256=_DIGEST,
        feature_snapshot_sha256=_DIGEST,
        policy_sha256=_DIGEST,
        manifest_sha256=_DIGEST,
        action_sha256=_DIGEST,
        resource_sha256=_DIGEST,
        args_sha256=_DIGEST,
        state_sha256=_DIGEST,
        projection_sha256=job.projection_sha256,
        created_at=job.created_at,
        completed_at="2026-01-01T00:00:02Z",
        latency_ms=2,
        input_tokens=17,
        output_tokens=5,
        cost_microunits=23,
        tenant_bucket_sha256=_DIGEST,
        findings=(
            {
                "code": "risk_detected",
                "severity": "medium",
                "confidence_bps": 7500,
                "evidence_sha256": _DIGEST,
                "source": "deterministic",
            },
        ),
        missing_predicates=("profile_pinned",),
    )


def _job_with_request_revision(
    job_id: str,
    *,
    request_id: str,
    request_revision: int,
) -> SemanticAssessmentJobRecord:
    job = _job(job_id)
    projection = {**job.projection, "request_revision": request_revision}
    projection_sha256 = hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        job,
        request_id=request_id,
        projection=projection,
        projection_sha256=projection_sha256,
    )


def test_semantic_job_exact_request_revision_lookup_is_bounded_and_ambiguous_fails_closed() -> None:
    store = SQLiteStore(":memory:")
    unit = UnitOfWork(store)
    try:
        captured = _job_with_request_revision(
            "exact-request-job",
            request_id="exact-request",
            request_revision=7,
        )
        unit.semantic.enqueue_semantic_assessment_job(captured)

        assert unit.semantic.get_semantic_assessment_job_for_request(
            "exact-request", 7
        ) == captured
        assert unit.semantic.get_semantic_assessment_job_for_request(
            "exact-request", 8
        ) is None

        duplicate = _job_with_request_revision(
            "duplicate-request-job",
            request_id="exact-request",
            request_revision=7,
        )
        unit.semantic.enqueue_semantic_assessment_job(duplicate)
        with pytest.raises(ValidationError, match="ambiguous"):
            unit.semantic.get_semantic_assessment_job_for_request(
                "exact-request", 7
            )
    finally:
        store.close()


def _claim(unit: UnitOfWork) -> SemanticAssessmentJobRecord:
    claimed = unit.semantic.claim_next_semantic_assessment_job(
        lease_owner_id="worker-1",
        lease_id="lease-1",
        lease_expires_at="2026-01-01T01:00:00Z",
        updated_at="2026-01-01T00:00:01Z",
    )
    assert claimed is not None
    return claimed


def _target(
    claimed: SemanticAssessmentJobRecord,
    assessment: SemanticAssessmentRecord,
) -> SemanticAssessmentJobRecord:
    return replace(
        claimed,
        assessment_id=assessment.assessment_id,
        status=SemanticAssessmentJobStatus.SUCCEEDED,
        revision=claimed.revision + 1,
        lease_owner_id=None,
        lease_id=None,
        lease_expires_at=None,
        projection={},
        projection_retention=SemanticProjectionRetention.HASH_ONLY,
        projection_expires_at=None,
        updated_at=assessment.completed_at,
        completed_at=assessment.completed_at,
    )


def test_semantic_job_claim_terminalization_is_single_winner_and_scrubs_projection() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        queued = _job()
        assert unit.semantic.enqueue_semantic_assessment_job(queued) == queued
        claimed = _claim(unit)
        assert claimed.revision == 1
        assert claimed.attempt_count == 1
        assert unit.semantic.claim_next_semantic_assessment_job(
            lease_owner_id="worker-2",
            lease_id="lease-2",
            lease_expires_at="2026-01-01T01:00:00Z",
            updated_at="2026-01-01T00:00:01Z",
        ) is None

        assessment = _assessment(claimed)
        target = _target(claimed, assessment)
        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed, target, assessment
        ) is True
        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed, target, assessment
        ) is False

        persisted = unit.semantic.get_semantic_assessment_job(claimed.job_id)
        assert persisted is not None
        assert persisted.projection == {}
        assert persisted.projection_retention is SemanticProjectionRetention.HASH_ONLY
        assert persisted.projection_expires_at is None
        assert unit.semantic.get_semantic_assessment(assessment.assessment_id) == assessment
    finally:
        store.close()


@pytest.mark.parametrize(
    ("adapter", "assessor_outcome", "expected_status"),
    (
        ("deterministic", "success", "success"),
        ("deterministic", "error", "provider_error"),
        ("scripted", "success", "success"),
        ("scripted", "error", "provider_error"),
        ("external", "success", "success"),
        ("external", "error", "provider_outcome_unknown"),
        ("external", "egress_blocked", "egress_blocked"),
    ),
)
def test_worker_merges_frozen_host_dlp_findings_for_every_adapter_terminal_path(
    adapter: str,
    assessor_outcome: str,
    expected_status: str,
) -> None:
    sentinel = "ghp_semanticLocalDlpSentinel123456789"
    seen_requests: list[SemanticAssessmentRequest] = []

    class Assessor:
        def _evaluate(
            self,
            request: SemanticAssessmentRequest,
        ) -> SemanticAssessment:
            seen_requests.append(request)
            if assessor_outcome == "success":
                return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)
            if assessor_outcome == "egress_blocked":
                raise CapabilityDenied("blocked without retaining provider payload")
            raise RuntimeError("provider failed without returning a payload")

        def assess(self, request: SemanticAssessmentRequest) -> SemanticAssessment:
            return self._evaluate(request)

        def assess_host(
            self,
            invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            return self._evaluate(invocation.request)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        labels = DataLabels(sensitivity="normal", origin="external:root-test")
        flow = DataFlowContext(labels=labels)
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(
                mode="shadow",
                adapter=adapter,
                external_profile_id="classifier" if adapter == "external" else None,
            ),
            assessor=Assessor(),
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )
        request = SemanticAssessmentRequest(
            kind=SemanticAssessmentKind.ROOT_GOAL,
            domain=SemanticDomain.RUNTIME,
            action_id="runtime.root_goal",
            input_sha256=_DIGEST,
            deadline_at="2099-01-01T00:00:00+00:00",
            data_labels=labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            redacted_intent=f"Use {sentinel} for the classifier",
            pid=f"pid-{adapter}-{assessor_outcome}",
            policy_sha256=_DIGEST,
            source_refs_sha256=flow.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
        )
        queued = manager._enqueue(  # noqa: SLF001 - exercise the worker contract
            request,
            candidate=None,
            hard_violations=(),
            data_flow_context=flow if adapter == "external" else None,
        )

        assert queued.projection["projection_mode"] == "metadata_only"
        assert queued.projection["dlp_findings"]
        assert sentinel not in json.dumps(queued.projection, sort_keys=True)
        assert manager.process_one()
        assert len(seen_requests) == 1
        assert seen_requests[0].data_labels.sensitivity.value == "secret"
        assert request.data_labels == labels

        records = unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
            pid=request.pid,
        ).records
        assert len(records) == 1
        record = records[0]
        assert record.status == expected_status
        assert any(
            item["code"] == "credential_material" and item["source"] == "host"
            for item in record.findings
        )
        local_data = next(
            item
            for item in record.data_findings
            if item["category"] == "credential"
        )
        assert local_data == {
            "category": "credential",
            "field": "root_goal",
            "span_start": None,
            "span_end": None,
            "sensitivity_floor": "secret",
            "integrity_ceiling": labels.integrity.value,
            "trust_ceiling": labels.trust_level.value,
            "confidence_bps": 10_000,
            "evidence_sha256": local_data["evidence_sha256"],
        }
        terminal = unit.semantic.get_semantic_assessment_job(queued.job_id)
        assert terminal is not None and terminal.projection == {}
        returned = manager.get_assessment(record.assessment_id)
        assert sentinel not in json.dumps(returned, sort_keys=True)
        assert request.data_labels == labels
    finally:
        store.close()


@pytest.mark.parametrize("reopen", (False, True), ids=("missing", "reopen"))
def test_external_worker_without_same_process_transient_flow_is_egress_blocked(
    tmp_path: Path,
    reopen: bool,
) -> None:
    provider_calls: list[str] = []

    class Assessor:
        def assess_host(
            self,
            invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            provider_calls.append(invocation.request.action_id)
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    config = SemanticDefaults(
        mode="shadow",
        adapter="external",
        external_profile_id="classifier",
    )

    def manager_for(selected_store: SQLiteStore) -> SemanticManager:
        return SemanticManager(
            UnitOfWork(selected_store).semantic,
            config=config,
            assessor=Assessor(),
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )

    database = str(tmp_path / "semantic-transient.sqlite") if reopen else ":memory:"
    store = SQLiteStore(database)
    manager = manager_for(store)
    labels = DataLabels(origin="derived")
    flow = DataFlowContext(labels=labels)
    request = SemanticAssessmentRequest(
        kind=SemanticAssessmentKind.ROOT_GOAL,
        domain=SemanticDomain.RUNTIME,
        action_id="runtime.root_goal",
        input_sha256=_DIGEST,
        deadline_at="2099-01-01T00:00:00+00:00",
        data_labels=labels,
        features=AuthoritativeApprovalFacts(schema_valid=True),
        redacted_intent="review the quarterly report",
        pid="pid-transient-reopen" if reopen else "pid-transient-missing",
        policy_sha256=_DIGEST,
        source_refs_sha256=flow.source_refs_hash(),
        data_labels_sha256=_identity_safe_labels_sha256(labels),
    )
    queued = manager._enqueue(  # noqa: SLF001 - exact worker privacy contract
        request,
        candidate=None,
        hard_violations=(),
        data_flow_context=flow if reopen else None,
    )
    if reopen:
        store.close()
        store = SQLiteStore(database)
        manager = manager_for(store)
    try:
        assert manager.process_one()
        persisted = store.get_semantic_assessment_job(queued.job_id)
        assert persisted is not None and persisted.assessment_id is not None
        terminal = manager.get_assessment(persisted.assessment_id)
        assert terminal is not None
        assert terminal["status"] == "egress_blocked"
        assert provider_calls == []
        assert persisted.projection == {}
        assert persisted.projection_retention is SemanticProjectionRetention.HASH_ONLY
    finally:
        store.close()


def _test_tenant_bucket(value: str) -> str:
    return hashlib.sha256(f"semantic-test-key\0{value}".encode("utf-8")).hexdigest()


def test_external_provider_worker_uses_verified_live_identity_and_origin() -> None:
    invocations: list[HostSemanticAssessmentInvocation] = []

    class Assessor:
        def assess_host(
            self,
            invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            invocations.append(invocation)
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        labels = DataLabels(
            sensitivity="normal",
            integrity="verified",
            trust_level="trusted",
            origin="external:provider-fixture",
            tenant="tenant-fixture",
            principal="principal-fixture",
            declassification_authority="host-release-fixture",
        )
        source = DataSourceRef(
            oid="obj-provider-source",
            version=1,
            content_sha256="2" * 64,
        )
        flow = DataFlowContext(labels=labels, source_refs=(source,))
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(
                mode="shadow",
                adapter="external",
                external_profile_id="classifier",
            ),
            assessor=Assessor(),
            tenant_bucketer=_test_tenant_bucket,
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )
        request = SemanticAssessmentRequest(
            kind=SemanticAssessmentKind.PROVIDER_INGRESS,
            domain=SemanticDomain.JSONRPC,
            action_id="jsonrpc.provider_ingress",
            input_sha256=_DIGEST,
            deadline_at="2099-01-01T00:00:00+00:00",
            data_labels=labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            pid="pid-provider-live-labels",
            effect_id="effect-provider-live-labels",
            policy_sha256=_DIGEST,
            source_refs_sha256=flow.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
        )
        manager._enqueue(  # noqa: SLF001 - exact transient envelope contract
            request,
            candidate=None,
            hard_violations=(),
            data_flow_context=flow,
        )

        assert manager.process_one()
        assert len(invocations) == 1
        effective = invocations[0].request.data_labels
        assert effective.origin == labels.origin
        assert effective.tenant == labels.tenant
        assert effective.principal == labels.principal
        assert effective.declassification_authority is None
        assert invocations[0].data_flow_context.labels == effective
        records = unit.semantic.query_semantic_assessments(
            after=None,
            limit=2,
            pid=request.pid,
        ).records
        assert len(records) == 1 and records[0].status == "success"
    finally:
        store.close()


@pytest.mark.parametrize(
    "scenario",
    ("normal", "persisted-double-null", "live-null", "bucketer-failure"),
)
def test_external_tenant_bucket_requires_exact_three_way_binding(
    scenario: str,
) -> None:
    provider_calls: list[str] = []
    bucket_calls = 0

    class Assessor:
        def assess_host(
            self,
            invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            provider_calls.append(invocation.request.action_id)
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    def bucketer(value: str) -> str:
        nonlocal bucket_calls
        bucket_calls += 1
        if scenario == "bucketer-failure" and bucket_calls > 1:
            raise RuntimeError("Host tenant bucketer unavailable")
        return _test_tenant_bucket(value)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        labels = DataLabels(
            origin="external:tenant-binding-fixture",
            tenant="tenant-exact",
            principal="principal-exact",
        )
        source = DataSourceRef(
            oid="obj-tenant-binding-source",
            version=1,
            content_sha256="5" * 64,
        )
        flow = DataFlowContext(labels=labels, source_refs=(source,))
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(
                mode="shadow",
                adapter="external",
                external_profile_id="classifier",
            ),
            assessor=Assessor(),
            tenant_bucketer=bucketer,
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )
        request = SemanticAssessmentRequest(
            kind=SemanticAssessmentKind.ROOT_GOAL,
            domain=SemanticDomain.RUNTIME,
            action_id="runtime.root_goal",
            input_sha256=_DIGEST,
            deadline_at="2099-01-01T00:00:00+00:00",
            data_labels=labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            pid=f"pid-tenant-binding-{scenario}",
            policy_sha256=_DIGEST,
            source_refs_sha256=flow.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
        )
        queued = manager._enqueue(  # noqa: SLF001 - adversarial binding matrix
            request,
            candidate=None,
            hard_violations=(),
            data_flow_context=flow,
        )
        if scenario == "persisted-double-null":
            projection = dict(queued.projection)
            projection["tenant_bucket_sha256"] = None
            bindings = dict(queued.bindings)
            bindings["tenant_bucket_sha256"] = None
            store.conn.execute(
                "UPDATE semantic_assessment_jobs "
                "SET projection_json = ?, bindings_json = ?, "
                "projection_sha256 = ? WHERE job_id = ?",
                (
                    json.dumps(
                        projection,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        bindings,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    _sha256(projection),
                    queued.job_id,
                ),
            )
            store.conn.commit()
        elif scenario == "live-null":
            snapshot = manager._transient_contexts[queued.job_id]  # noqa: SLF001
            changed_flow = replace(
                flow,
                labels=replace(labels, tenant=None),
            )
            manager._transient_contexts[queued.job_id] = replace(  # noqa: SLF001
                snapshot,
                context=changed_flow,
                exact_labels_sha256=_sha256(changed_flow.labels.to_dict()),
            )

        assert manager.process_one()
        records = unit.semantic.query_semantic_assessments(
            after=None,
            limit=2,
            pid=request.pid,
        ).records
        assert len(records) == 1
        if scenario == "normal":
            assert records[0].status == "success"
            assert provider_calls == ["runtime.root_goal"]
        else:
            assert records[0].status == "egress_blocked"
            assert provider_calls == []
    finally:
        store.close()


@pytest.mark.parametrize("drift", ("tenant", "principal", "source_ref"))
def test_external_worker_rejects_transient_identity_or_source_drift(
    drift: str,
) -> None:
    provider_calls: list[str] = []

    class Assessor:
        def assess_host(
            self,
            invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            provider_calls.append(invocation.request.action_id)
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        labels = DataLabels(
            origin="external:drift-fixture",
            tenant="tenant-original",
            principal="principal-original",
        )
        source = DataSourceRef(
            oid="obj-original-source",
            version=1,
            content_sha256="3" * 64,
        )
        flow = DataFlowContext(labels=labels, source_refs=(source,))
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(
                mode="shadow",
                adapter="external",
                external_profile_id="classifier",
            ),
            assessor=Assessor(),
            tenant_bucketer=_test_tenant_bucket,
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )
        request = SemanticAssessmentRequest(
            kind=SemanticAssessmentKind.ROOT_GOAL,
            domain=SemanticDomain.RUNTIME,
            action_id="runtime.root_goal",
            input_sha256=_DIGEST,
            deadline_at="2099-01-01T00:00:00+00:00",
            data_labels=labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            pid=f"pid-transient-{drift}",
            policy_sha256=_DIGEST,
            source_refs_sha256=flow.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
        )
        queued = manager._enqueue(  # noqa: SLF001 - adversarial transient drift
            request,
            candidate=None,
            hard_violations=(),
            data_flow_context=flow,
        )
        snapshot = manager._transient_contexts[queued.job_id]  # noqa: SLF001
        if drift == "tenant":
            changed = replace(flow, labels=replace(labels, tenant="tenant-drifted"))
        elif drift == "principal":
            changed = replace(
                flow,
                labels=replace(labels, principal="principal-drifted"),
            )
        else:
            changed = replace(
                flow,
                source_refs=(
                    DataSourceRef(
                        oid="obj-drifted-source",
                        version=1,
                        content_sha256="4" * 64,
                    ),
                ),
            )
        manager._transient_contexts[queued.job_id] = replace(  # noqa: SLF001
            snapshot,
            context=changed,
        )

        assert manager.process_one()
        assert provider_calls == []
        records = unit.semantic.query_semantic_assessments(
            after=None,
            limit=2,
            pid=request.pid,
        ).records
        assert len(records) == 1
        assert records[0].status == "egress_blocked"
    finally:
        store.close()


@pytest.mark.parametrize(
    "live_artifact_sha256",
    ("2" * 64, "3" * 64),
    ids=("profile-drift", "model-drift"),
)
def test_reopened_worker_refuses_classifier_artifact_drift_before_call(
    tmp_path: Path,
    live_artifact_sha256: str,
) -> None:
    """A persisted projection cannot be reassigned to a new profile/model."""

    database = str(tmp_path / f"semantic-{live_artifact_sha256[0]}-drift.sqlite")
    config = SemanticDefaults(
        mode="shadow",
        adapter="external",
        external_profile_id="classifier",
    )

    class Assessor:
        def __init__(self) -> None:
            self.calls = 0

        def assess_host(
            self,
            _invocation: HostSemanticAssessmentInvocation,
        ) -> SemanticAssessment:
            self.calls += 1
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    def manager_for(
        selected_store: SQLiteStore,
        *,
        assessor: Assessor,
        artifact_sha256: str,
    ) -> SemanticManager:
        return SemanticManager(
            UnitOfWork(selected_store).semantic,
            config=config,
            assessor=assessor,
            artifact_sha256=artifact_sha256,
            request_capture_registrar=lambda _callback: None,
            spawn_observer_registrar=lambda _callback: None,
            result_observer_registrar=lambda _callback: None,
            request_capture=lambda _request: None,
            spawn_observer=lambda *_args, **_kwargs: None,
            result_observer=lambda *_args, **_kwargs: None,
        )

    captured_store = SQLiteStore(database)
    captured_assessor = Assessor()
    captured_manager = manager_for(
        captured_store,
        assessor=captured_assessor,
        artifact_sha256=_DIGEST,
    )
    labels = DataLabels(origin="derived")
    request = SemanticAssessmentRequest(
        kind=SemanticAssessmentKind.ROOT_GOAL,
        domain=SemanticDomain.RUNTIME,
        action_id="runtime.root_goal",
        input_sha256=_DIGEST,
        deadline_at="2099-01-01T00:00:00+00:00",
        data_labels=labels,
        features=AuthoritativeApprovalFacts(schema_valid=True),
        pid="pid-artifact-drift",
        policy_sha256=_DIGEST,
        source_refs_sha256=DataFlowContext(labels=labels).source_refs_hash(),
    )
    captured_manager._enqueue(  # noqa: SLF001 - freeze capture provenance
        request,
        candidate=None,
        hard_violations=(),
    )
    captured_store.close()

    reopened_store = SQLiteStore(database)
    live_assessor = Assessor()
    reopened_manager = manager_for(
        reopened_store,
        assessor=live_assessor,
        artifact_sha256=live_artifact_sha256,
    )
    try:
        assert reopened_manager.process_one() is True
        assert captured_assessor.calls == 0
        assert live_assessor.calls == 0

        records = UnitOfWork(
            reopened_store
        ).semantic.query_semantic_assessments(after=None, limit=2).records
        assert len(records) == 1
        assert records[0].status == SemanticAssessmentStatus.STALE_INPUT.value
        assert records[0].shadow_outcome == "require_human"
        assert records[0].artifact_sha256 == _DIGEST
    finally:
        reopened_store.close()


@pytest.mark.parametrize(
    "injected",
    (
        {
            "category": "credential",
            "code": "credential_material",
            "evidence_sha256": _DIGEST,
            "field": "cmF3X3Byb3ZpZGVyX3NlY3JldA",
        },
        {
            "category": "free_form_category",
            "code": "credential_material",
            "evidence_sha256": _DIGEST,
        },
        {
            "category": "credential",
            "code": "sensitive_data",
            "evidence_sha256": _DIGEST,
        },
    ),
)
def test_worker_rejects_non_allowlisted_frozen_dlp_evidence(
    injected: dict[str, str],
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        base = _job("dlp-injection")
        projection = {
            **base.projection,
            "dlp_findings": [injected],
        }
        queued = replace(
            base,
            projection=projection,
            projection_sha256=hashlib.sha256(
                json.dumps(
                    projection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        unit.semantic.enqueue_semantic_assessment_job(queued)
        claimed = _claim(unit)
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(mode="off", adapter="deterministic"),
        )

        with pytest.raises(ValidationError, match="local DLP evidence"):
            manager._terminalize(  # noqa: SLF001 - adversarial frozen evidence
                claimed,
                SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS),
                "2026-01-01T00:00:02Z",
                1,
            )

        assert unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
        ).records == ()
        persisted = unit.semantic.get_semantic_assessment_job(claimed.job_id)
        assert persisted is not None
        assert persisted.status is SemanticAssessmentJobStatus.CLAIMED
    finally:
        store.close()


@pytest.mark.parametrize("backlog_size", (501, 1001))
def test_off_recovery_is_bounded_per_tick_and_drains_incrementally(
    backlog_size: int,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        for index in range(backlog_size):
            unit.semantic.enqueue_semantic_assessment_job(
                _job(f"bounded-recovery-{index:04d}")
            )
        manager = SemanticManager(
            unit.semantic,
            config=SemanticDefaults(
                mode="off",
                adapter="deterministic",
                recovery_batch_limit=100,
            ),
        )

        assert manager._run_maintenance_batch("off") is True
        queued = unit.semantic.query_semantic_assessment_jobs(
            statuses=(SemanticAssessmentJobStatus.QUEUED.value,),
            projection_expires_before=None,
            limit=500,
        )
        cancelled = unit.semantic.query_semantic_assessment_jobs(
            statuses=(SemanticAssessmentJobStatus.CANCELLED.value,),
            projection_expires_before=None,
            limit=500,
        )
        assert len(queued) == min(backlog_size - 100, 500)
        assert len(cancelled) == 100

        manager.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if manager.status()["queue"]["cancelled"] == backlog_size:
                break
            time.sleep(0.01)
        assert manager.status()["queue"]["cancelled"] == backlog_size
        assert unit.semantic.query_semantic_assessment_jobs(
            statuses=(SemanticAssessmentJobStatus.QUEUED.value,),
            projection_expires_before=None,
            limit=500,
        ) == ()
        terminal = unit.semantic.query_semantic_assessment_jobs(
            statuses=(SemanticAssessmentJobStatus.CANCELLED.value,),
            projection_expires_before=None,
            limit=500,
        )
        # The storage page itself is capped at 500; the aggregate proves all
        # Every record reached the terminal state across bounded ticks.
        assert len(terminal) == 500
        assert manager.shutdown()
    finally:
        store.close()


def test_semantic_assessment_append_is_idempotent_but_never_overwrites() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        record = _assessment(_job())
        assert unit.semantic.append_semantic_assessment(record) == record
        assert unit.semantic.append_semantic_assessment(record) == record
        with pytest.raises(ValidationError, match="conflicts"):
            unit.semantic.append_semantic_assessment(
                replace(record, reason_codes=("provider_error",))
            )
        with pytest.raises(ValidationError, match="cannot be generically deleted"):
            store.delete_table_rows(
                "semantic_assessments",
                "assessment_id = ?",
                (record.assessment_id,),
            )
        assert unit.semantic.get_semantic_assessment(record.assessment_id) == record
    finally:
        store.close()


def test_semantic_projection_rejects_content_fields_and_requires_terminal_hash_only() -> None:
    with pytest.raises(ValidationError, match="forbidden content"):
        replace(_job(), projection={"prompt": "do not persist me"})

    queued = _job()
    with pytest.raises(ValidationError, match="active semantic job cannot have"):
        replace(queued, error_code="provider_error")
    with pytest.raises(ValidationError, match="terminal semantic job projection"):
        replace(
            queued,
            assessment_id="assessment-1",
            status=SemanticAssessmentJobStatus.CANCELLED,
            revision=1,
            completed_at="2026-01-01T00:00:01Z",
        )

    with pytest.raises(ValidationError, match="bounded strict JSON"):
        replace(
            _assessment(queued),
            matched_rule_ids=("x" * 4096,) * 256,
        )


def test_semantic_projection_digest_mismatch_never_enqueues_a_job() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        with pytest.raises(ValidationError, match="digest does not match"):
            unit.semantic.enqueue_semantic_assessment_job(
                replace(_job("mismatched"), projection_sha256="2" * 64)
            )
        assert unit.semantic.get_semantic_assessment_job("mismatched") is None
    finally:
        store.close()


def test_semantic_job_requires_a_frozen_action_digest() -> None:
    bindings = dict(_job("missing-action-binding").bindings)
    del bindings["action_sha256"]
    with pytest.raises(ValidationError, match="missing=.*action_sha256"):
        replace(_job("missing-action-binding"), bindings=bindings)

    with pytest.raises(ValidationError, match="action_sha256 cannot be null"):
        bindings["action_sha256"] = None
        replace(_job("null-action-binding"), bindings=bindings)


def test_semantic_evidence_rejects_secret_bearing_free_text_fields() -> None:
    sentinel = "SEMANTIC_SECRET_SENTINEL_do-not-store"
    with pytest.raises(ValidationError, match="error code is invalid"):
        replace(_job("secret-job"), error_code=sentinel)
    with pytest.raises(ValidationError, match="human outcome is invalid"):
        replace(_assessment(_job("secret-assessment")), human_outcome=sentinel)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        job = _job("safe")
        assessment = _assessment(job)
        unit.semantic.enqueue_semantic_assessment_job(job)
        unit.semantic.append_semantic_assessment(assessment)
        rows = (
            *store.select_table_rows("semantic_assessment_jobs"),
            *store.select_table_rows("semantic_assessments"),
        )
        assert sentinel not in json.dumps(rows, ensure_ascii=False, sort_keys=True)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"action_id": "filesystem"}, "dotted lower-case identifier"),
        ({"action_id": "filesystem.*"}, "dotted lower-case identifier"),
        ({"calibration_bucket": "extreme"}, "kind/domain/status/calibration is invalid"),
        ({"input_tokens": True}, "exact integer from 0 through"),
        ({"output_tokens": -1}, "exact integer from 0 through"),
        ({"cost_microunits": False}, "exact integer from 0 through"),
        ({"input_tokens": 1 << 53}, "exact integer from 0 through"),
        ({"output_tokens": 1 << 53}, "exact integer from 0 through"),
        ({"cost_microunits": 1 << 53}, "exact integer from 0 through"),
        ({"latency_ms": 1 << 53}, "latency_ms is invalid"),
        ({"tenant_bucket_sha256": "A" * 64}, "lowercase SHA-256 digest"),
        ({"manifest_sha256": "A" * 64}, "lowercase SHA-256 digest"),
        ({"action_sha256": None}, "lowercase SHA-256 digest"),
        ({"projection_sha256": None}, "lowercase SHA-256 digest"),
        ({"ood": True}, "OOD status and flag must match"),
        ({"abstain": True}, "abstained status and flag must match"),
        ({"status": "ood"}, "OOD status and flag must match"),
        ({"status": "abstained"}, "abstained status and flag must match"),
        (
            {"status": "ood", "ood": True, "abstain": True},
            "abstained status and flag must match",
        ),
    ],
)
def test_semantic_assessment_metrics_are_strictly_validated(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        replace(_assessment(_job("strict-metrics")), **changes)


def test_semantic_assessment_record_accepts_canonical_uncertainty_status_flags() -> None:
    base = _assessment(_job("canonical-uncertainty"))

    ood = replace(base, status="ood", ood=True)
    abstained = replace(base, status="abstained", abstain=True)

    assert SemanticAssessmentRecord.from_dict(ood.to_dict()) == ood
    assert SemanticAssessmentRecord.from_dict(abstained.to_dict()) == abstained


def test_semantic_assessment_record_accepts_json_safe_integer_ceiling() -> None:
    ceiling = (1 << 53) - 1
    record = replace(
        _assessment(_job("safe-integer-ceiling")),
        input_tokens=ceiling,
        output_tokens=ceiling,
        cost_microunits=ceiling,
        latency_ms=ceiling,
    )

    assert SemanticAssessmentRecord.from_dict(record.to_dict()) == record


def test_semantic_assessment_oversized_integer_rejection_leaves_no_write() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        for field in ("input_tokens", "output_tokens", "cost_microunits", "latency_ms"):
            with pytest.raises(ValidationError):
                unit.semantic.append_semantic_assessment(
                    replace(
                        _assessment(_job(f"oversized-{field}")),
                        **{field: 1 << 53},
                    )
                )
        assert unit.semantic.query_semantic_assessments(
            after=None,
            limit=100,
        ).records == ()
    finally:
        store.close()


@pytest.mark.parametrize(
    "required_field",
    [
        "action_id",
        "calibration_bucket",
        "manifest_sha256",
        "action_sha256",
        "resource_sha256",
        "args_sha256",
        "state_sha256",
        "projection_sha256",
    ],
)
def test_semantic_assessment_decode_requires_action_and_calibration(
    required_field: str,
) -> None:
    payload = _assessment(_job("required-metrics")).to_dict()
    del payload[required_field]
    with pytest.raises(ValidationError, match="fields are invalid"):
        SemanticAssessmentRecord.from_dict(payload)


def test_semantic_assessment_decode_rejects_unknown_metric_fields() -> None:
    payload = _assessment(_job("unknown-metrics")).to_dict()
    payload["raw_cost_response"] = "must-not-persist"
    with pytest.raises(ValidationError, match="fields are invalid"):
        SemanticAssessmentRecord.from_dict(payload)


def test_semantic_assessment_record_rejects_span_on_coarse_locator() -> None:
    invalid = {
        "category": "source_code",
        "field": "provider.result",
        "span_start": 0,
        "span_end": 1,
        "sensitivity_floor": "confidential",
        "integrity_ceiling": "unknown",
        "trust_ceiling": "untrusted",
        "confidence_bps": 9_200,
        "evidence_sha256": "f" * 64,
    }

    with pytest.raises(ValidationError, match="semantic data findings item is invalid"):
        replace(
            _assessment(_job("coarse-span")),
            data_findings=(invalid,),
        )


@pytest.mark.parametrize("field", ["findings", "data_findings"])
def test_semantic_assessment_record_rejects_more_than_64_findings(
    field: str,
) -> None:
    finding = (
        {
            "code": "risk_detected",
            "severity": "low",
            "confidence_bps": 1,
            "evidence_sha256": "e" * 64,
            "source": "model",
        }
        if field == "findings"
        else {
            "category": "source_code",
            "field": "provider.result",
            "span_start": None,
            "span_end": None,
            "sensitivity_floor": "confidential",
            "integrity_ceiling": "unknown",
            "trust_ceiling": "untrusted",
            "confidence_bps": 1,
            "evidence_sha256": "f" * 64,
        }
    )

    with pytest.raises(ValidationError, match="must be a bounded list"):
        replace(
            _assessment(_job(f"too-many-{field}")),
            **{field: tuple(dict(finding) for _ in range(65))},
        )


def test_successful_classifier_cannot_add_frozen_data_flow_allow_predicate() -> None:
    store = SQLiteStore(":memory:")
    try:
        manager = SemanticManager(
            UnitOfWork(store).semantic,
            config=SemanticDefaults(mode="off", adapter="deterministic"),
        )
        frozen = AuthoritativeApprovalFacts(
            **{
                name: name != "data_flow_allowed"
                for name in AuthoritativeApprovalFacts.__dataclass_fields__
            }
        )
        base = _job("model-cannot-allow")
        projection = {**base.projection, "features": frozen.to_dict()}
        projection_sha256 = hashlib.sha256(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        job = replace(
            base,
            projection=projection,
            projection_sha256=projection_sha256,
        )

        facts = manager._facts_from_job(
            job,
            SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS),
            "pending",
        )

        assert not facts.data_flow_allowed
    finally:
        store.close()


def test_semantic_assessment_metrics_survive_sqlite_reopen_and_index_projection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic-metrics.sqlite"
    record = replace(
        _assessment(_job("metrics-reopen")),
        input_tokens=None,
        output_tokens=None,
        cost_microunits=None,
    )
    store = SQLiteStore(path)
    try:
        UnitOfWork(store).semantic.append_semantic_assessment(record)
    finally:
        store.close()

    reopened = SQLiteStore(path)
    try:
        unit = UnitOfWork(reopened)
        assert unit.semantic.get_semantic_assessment(record.assessment_id) == record
        rows = reopened.select_table_rows(
            "semantic_assessments",
            "assessment_id = ?",
            (record.assessment_id,),
        )
        assert len(rows) == 1
        assert rows[0]["action_id"] == record.action_id
        assert rows[0]["tenant_bucket_sha256"] == record.tenant_bucket_sha256
        assert unit.semantic.query_semantic_assessments(
            after=None,
            limit=1,
            action_id=record.action_id,
            tenant_bucket_sha256=record.tenant_bucket_sha256,
        ).records == (record,)
    finally:
        reopened.close()


def test_semantic_claim_requires_a_future_lease_without_mutating_queue() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        queued = _job("bad-lease")
        unit.semantic.enqueue_semantic_assessment_job(queued)
        with pytest.raises(ValidationError, match="lease expiry must follow"):
            unit.semantic.claim_next_semantic_assessment_job(
                lease_owner_id="worker-1",
                lease_id="lease-1",
                lease_expires_at="2026-01-01T00:00:01Z",
                updated_at="2026-01-01T00:00:01Z",
            )
        assert unit.semantic.get_semantic_assessment_job(queued.job_id) == queued
    finally:
        store.close()


def test_semantic_terminal_cas_rejects_transitions_without_claim_provenance() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        queued = _job("unclaimed")
        unit.semantic.enqueue_semantic_assessment_job(queued)
        assessment = _assessment(queued)
        target = replace(
            queued,
            assessment_id=assessment.assessment_id,
            status=SemanticAssessmentJobStatus.SUCCEEDED,
            revision=1,
            projection={},
            projection_retention=SemanticProjectionRetention.HASH_ONLY,
            projection_expires_at=None,
            updated_at=assessment.completed_at,
            completed_at=assessment.completed_at,
        )
        with pytest.raises(ValidationError, match="claim/attempt provenance"):
            unit.semantic.terminalize_semantic_assessment_job(
                queued,
                target,
                assessment,
            )
        assert unit.semantic.get_semantic_assessment_job(queued.job_id) == queued

        claimed = _claim(unit)
        skipped = replace(
            _assessment(claimed),
            status="skipped_policy",
        )
        cancelled = replace(
            _target(claimed, skipped),
            status=SemanticAssessmentJobStatus.CANCELLED,
        )
        cancelled = replace(cancelled, error_code="disabled")
        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed,
            cancelled,
            skipped,
        )
        persisted = unit.semantic.get_semantic_assessment_job(claimed.job_id)
        assert persisted == cancelled
        assert persisted.attempt_count == 1
    finally:
        store.close()


def test_expired_semantic_claim_can_only_record_unknown_provider_outcome() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("expired-claim"))
        claimed = _claim(unit)
        late = replace(
            _assessment(claimed),
            completed_at="2026-01-01T01:00:01Z",
        )
        with pytest.raises(ValidationError, match="unknown provider outcome"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                _target(claimed, late),
                late,
            )
        assert unit.semantic.get_semantic_assessment_job(claimed.job_id) == claimed

        unknown = replace(
            late,
            status="provider_outcome_unknown",
            reason_codes=("provider_outcome_unknown",),
        )
        unknown_target = replace(
            _target(claimed, unknown),
            status=SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN,
            error_code="provider_outcome_unknown",
        )
        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed,
            unknown_target,
            unknown,
        )
    finally:
        store.close()


def test_semantic_assessment_timestamps_and_terminal_snapshot_are_bound() -> None:
    with pytest.raises(ValidationError, match="completion cannot precede"):
        replace(
            _assessment(_job("backwards-time")),
            created_at="2026-01-01T00:00:03Z",
            completed_at="2026-01-01T00:00:02Z",
        )

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("snapshot-time"))
        claimed = _claim(unit)
        valid = _assessment(claimed)

        changed_creation = replace(
            valid,
            created_at="2026-01-01T00:00:01Z",
        )
        with pytest.raises(ValidationError, match="creation does not match"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                _target(claimed, changed_creation),
                changed_creation,
            )

        mismatched_target = replace(
            _target(claimed, valid),
            completed_at="2026-01-01T00:00:03Z",
            updated_at="2026-01-01T00:00:03Z",
        )
        with pytest.raises(ValidationError, match="completion does not match"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                mismatched_target,
                valid,
            )
        assert unit.semantic.get_semantic_assessment_job(claimed.job_id) == claimed
        assert unit.semantic.get_semantic_assessment(valid.assessment_id) is None
    finally:
        store.close()


def test_semantic_terminal_error_code_is_closed_and_matches_outcome() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("error-map"))
        claimed = _claim(unit)
        success = _assessment(claimed)
        with pytest.raises(ValidationError, match="error code does not match"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                replace(_target(claimed, success), error_code="provider_error"),
                success,
            )

        timed_out = replace(
            success,
            status="timeout",
            reason_codes=("timeout",),
        )
        failed = replace(
            _target(claimed, timed_out),
            status=SemanticAssessmentJobStatus.FAILED,
            error_code="timeout",
        )
        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed,
            failed,
            timed_out,
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("artifact_sha256", "2" * 64),
        ("manifest_sha256", "2" * 64),
        ("action_sha256", "2" * 64),
        ("resource_sha256", "2" * 64),
        ("args_sha256", "2" * 64),
        ("state_sha256", "2" * 64),
        ("projection_sha256", "2" * 64),
        ("sink_identity_sha256", "2" * 64),
    ],
)
def test_semantic_terminalization_binds_frozen_provenance_digests(
    field_name: str,
    value: str,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job())
        claimed = _claim(unit)
        valid = _assessment(claimed)
        mismatched = replace(valid, **{field_name: value})
        target = _target(claimed, mismatched)
        with pytest.raises(ValidationError, match="frozen job"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                target,
                mismatched,
            )
        assert unit.semantic.get_semantic_assessment(mismatched.assessment_id) is None
        assert unit.semantic.get_semantic_assessment_job(claimed.job_id) == claimed
    finally:
        store.close()


@pytest.mark.parametrize("tampered_column", ["bindings_json", "projection_sha256"])
def test_semantic_terminal_cas_rejects_persisted_provenance_tampering(
    tampered_column: str,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("persisted-tamper"))
        claimed = _claim(unit)
        assessment = _assessment(claimed)
        target = _target(claimed, assessment)
        if tampered_column == "bindings_json":
            tampered_bindings = dict(claimed.bindings)
            tampered_bindings["action_sha256"] = "2" * 64
            tampered_value = json.dumps(
                tampered_bindings,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
        else:
            tampered_value = "2" * 64
        store.conn.execute(
            f"UPDATE semantic_assessment_jobs SET {tampered_column} = ? WHERE job_id = ?",
            (tampered_value, claimed.job_id),
        )
        store.conn.commit()

        assert unit.semantic.terminalize_semantic_assessment_job(
            claimed,
            target,
            assessment,
        ) is False
        assert unit.semantic.get_semantic_assessment(assessment.assessment_id) is None
        rows = store.select_table_rows(
            "semantic_assessment_jobs",
            "job_id = ?",
            (claimed.job_id,),
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "claimed"
        assert rows[0][tampered_column] == tampered_value
    finally:
        store.close()


def test_semantic_terminal_append_conflict_rolls_back_job_cas() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job())
        claimed = _claim(unit)
        candidate = _assessment(claimed)
        unit.semantic.append_semantic_assessment(
            replace(candidate, human_outcome="pending")
        )
        target = _target(claimed, candidate)
        with pytest.raises(ValidationError, match="conflicts"):
            unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                target,
                candidate,
            )
        assert unit.semantic.get_semantic_assessment_job(claimed.job_id) == claimed
    finally:
        store.close()


def test_semantic_query_is_bounded_and_unicode_keyset_is_lossless() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        records = tuple(
            _assessment(
                _job(job_id),
                assessment_id=f"assessment-{job_id}",
            )
            for job_id in ("a", "é", "中")
        )
        for record in records:
            unit.semantic.append_semantic_assessment(record)

        first = unit.semantic.query_semantic_assessments(after=None, limit=2)
        assert len(first.records) == 2
        assert isinstance(first.next_cursor, SemanticAssessmentCursor)
        second = unit.semantic.query_semantic_assessments(
            after=first.next_cursor,
            limit=2,
        )
        assert len(second.records) == 1
        assert {record.assessment_id for record in (*first.records, *second.records)} == {
            record.assessment_id for record in records
        }
        with pytest.raises(ValidationError, match="hard cap"):
            unit.semantic.query_semantic_assessments(after=None, limit=501)
    finally:
        store.close()


def test_semantic_job_off_and_expiry_candidates_are_bounded_without_claiming() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("one"))
        unit.semantic.enqueue_semantic_assessment_job(_job("two"))
        candidates = unit.semantic.query_semantic_assessment_jobs(
            statuses=("queued",),
            projection_expires_before="2026-01-01T03:00:00Z",
            limit=1,
        )
        assert len(candidates) == 1
        assert candidates[0].status is SemanticAssessmentJobStatus.QUEUED
        assert candidates[0].attempt_count == 0
    finally:
        store.close()


def test_semantic_projection_expiry_uses_canonical_utc_not_raw_offset_text() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        # This is 2025-12-31T10:00Z even though its raw text begins with 2026.
        queued = replace(
            _job("offset-expiry"),
            projection_expires_at="2026-01-01T00:00:00+14:00",
        )
        assert queued.projection_expires_at == "2025-12-31T10:00:00.000000+00:00"
        unit.semantic.enqueue_semantic_assessment_job(queued)
        assert unit.semantic.claim_next_semantic_assessment_job(
            lease_owner_id="worker-offset",
            lease_id="lease-offset",
            lease_expires_at="2025-12-31T12:00:00Z",
            updated_at="2025-12-31T11:00:00Z",
        ) is None
        candidates = unit.semantic.query_semantic_assessment_jobs(
            statuses=("queued",),
            projection_expires_before="2026-01-01T01:00:00+14:00",
            limit=1,
        )
        assert candidates == (queued,)

        # A legacy/corrupt raw offset must still fail closed after selection,
        # even though its unnormalized text sorts after the claim boundary.
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE semantic_assessment_jobs SET projection_expires_at = ? "
                "WHERE job_id = ?",
                ("2026-01-01T00:00:00+14:00", queued.job_id),
            )
        assert unit.semantic.claim_next_semantic_assessment_job(
            lease_owner_id="worker-offset",
            lease_id="lease-offset-raw",
            lease_expires_at="2025-12-31T12:00:00Z",
            updated_at="2025-12-31T11:00:00Z",
        ) is None
    finally:
        store.close()


def test_semantic_lease_recovery_uses_canonical_utc_not_raw_offset_text() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        unit.semantic.enqueue_semantic_assessment_job(_job("offset-lease"))
        claimed = unit.semantic.claim_next_semantic_assessment_job(
            lease_owner_id="worker-offset",
            lease_id="lease-offset",
            lease_expires_at="2026-01-01T00:00:00+14:00",
            updated_at="2025-12-31T09:00:00Z",
        )
        assert claimed is not None
        assert claimed.lease_expires_at == "2025-12-31T10:00:00.000000+00:00"
        assert unit.semantic.query_expired_semantic_assessment_jobs(
            expired_before="2025-12-31T11:00:00-00:00",
            limit=1,
        ) == (claimed,)
    finally:
        store.close()


def test_semantic_keyset_normalizes_offsets_and_fractional_precision() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        later = _assessment(
            _job("fraction-later", created_at="2026-01-01T00:00:00.1Z")
        )
        earlier = _assessment(
            _job("fraction-earlier", created_at="2026-01-01T01:00:00.01+01:00")
        )
        assert earlier.created_at == "2026-01-01T00:00:00.010000+00:00"
        assert later.created_at == "2026-01-01T00:00:00.100000+00:00"
        unit.semantic.append_semantic_assessment(later)
        unit.semantic.append_semantic_assessment(earlier)

        first = unit.semantic.query_semantic_assessments(after=None, limit=1)
        assert first.records == (earlier,)
        assert first.next_cursor is not None
        assert first.next_cursor.created_at == earlier.created_at
        second = unit.semantic.query_semantic_assessments(
            after=SemanticAssessmentCursor(
                created_at="2026-01-01T01:00:00.01+01:00",
                assessment_id=earlier.assessment_id,
            ),
            limit=1,
        )
        assert second.records == (later,)
    finally:
        store.close()


def test_semantic_status_aggregate_is_unbounded_complete_and_one_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        for index in range(501):
            unit.semantic.enqueue_semantic_assessment_job(
                _job(f"aggregate-job-{index:04d}")
            )

        statuses = tuple(SemanticAssessmentStatus)
        domains = tuple(SemanticDomain)
        outcomes = (
            "would_issue_exact_once",
            "require_human",
            "would_deny",
        )
        for index, status in enumerate(statuses):
            job = _job(f"aggregate-assessment-job-{index:02d}")
            unit.semantic.append_semantic_assessment(
                replace(
                    _assessment(job),
                    assessment_id=f"aggregate-assessment-{index:02d}",
                    status=status.value,
                    domain=domains[index % len(domains)].value,
                    shadow_outcome=outcomes[index % len(outcomes)],
                    ood=status is SemanticAssessmentStatus.OOD,
                    abstain=status is SemanticAssessmentStatus.ABSTAINED,
                )
            )

        query_count = 0
        original_query = store._query

        def counted_query(sql: str, params: object = ()) -> list[object]:
            nonlocal query_count
            query_count += 1
            return original_query(sql, params)  # type: ignore[arg-type]

        monkeypatch.setattr(store, "_query", counted_query)
        aggregate = unit.semantic.semantic_status_aggregate()

        assert query_count == 1
        assert aggregate.job_total == 501
        assert aggregate.job_counts == {
            status.value: (501 if status is SemanticAssessmentJobStatus.QUEUED else 0)
            for status in SemanticAssessmentJobStatus
        }
        assert aggregate.assessment_total == len(statuses)
        assert aggregate.assessment_status_counts == {
            status.value: 1 for status in statuses
        }
        assert aggregate.assessment_domain_counts == {
            domain.value: sum(
                domains[index % len(domains)] is domain
                for index in range(len(statuses))
            )
            for domain in domains
        }
        assert aggregate.assessment_ood_count == 1
        assert aggregate.shadow_outcome_counts == {
            outcome: sum(
                outcomes[index % len(outcomes)] == outcome
                for index in range(len(statuses))
            )
            for outcome in outcomes
        }
    finally:
        store.close()


def test_semantic_status_aggregate_strictly_rejects_incomplete_counts() -> None:
    job_counts = {status.value: 0 for status in SemanticAssessmentJobStatus}
    status_counts = {status.value: 0 for status in SemanticAssessmentStatus}
    domain_counts = {domain.value: 0 for domain in SemanticDomain}
    outcome_counts = {
        "would_issue_exact_once": 0,
        "require_human": 0,
        "would_deny": 0,
    }
    aggregate = SemanticStatusAggregate(
        job_total=0,
        job_counts=job_counts,
        assessment_total=0,
        assessment_status_counts=status_counts,
        assessment_domain_counts=domain_counts,
        assessment_ood_count=0,
        shadow_outcome_counts=outcome_counts,
    )

    with pytest.raises(ValidationError, match="job status counts do not match"):
        replace(aggregate, job_total=1)
    with pytest.raises(ValidationError, match="every canonical outcome"):
        replace(aggregate, assessment_domain_counts={"filesystem": 0})
    with pytest.raises(ValidationError, match="non-negative exact integers"):
        replace(
            aggregate,
            assessment_status_counts={**status_counts, "success": True},
        )
    with pytest.raises(ValidationError, match="does not match OOD status count"):
        replace(aggregate, assessment_ood_count=1)


def test_semantic_manager_status_consumes_large_typed_snapshot_without_scans() -> None:
    job_counts = {status.value: 0 for status in SemanticAssessmentJobStatus}
    job_counts[SemanticAssessmentJobStatus.QUEUED.value] = 501
    status_counts = {status.value: 0 for status in SemanticAssessmentStatus}
    status_counts[SemanticAssessmentStatus.SUCCESS.value] = 1
    status_counts[SemanticAssessmentStatus.OOD.value] = 70_000
    status_counts[SemanticAssessmentStatus.PROVIDER_ERROR.value] = 30_000
    domain_counts = {domain.value: 0 for domain in SemanticDomain}
    domain_counts[SemanticDomain.FILESYSTEM.value] = 100_001
    aggregate = SemanticStatusAggregate(
        job_total=501,
        job_counts=job_counts,
        assessment_total=100_001,
        assessment_status_counts=status_counts,
        assessment_domain_counts=domain_counts,
        assessment_ood_count=70_000,
        shadow_outcome_counts={
            "would_issue_exact_once": 1,
            "require_human": 100_000,
            "would_deny": 0,
        },
    )

    class AggregateOnlyRepository:
        calls = 0

        def semantic_status_aggregate(self) -> SemanticStatusAggregate:
            self.calls += 1
            return aggregate

        def query_semantic_assessments(self, **_kwargs: object) -> None:
            raise AssertionError("status must not scan assessment pages")

        def query_semantic_assessment_jobs(self, **_kwargs: object) -> None:
            raise AssertionError("status must not scan queue pages")

    repository = AggregateOnlyRepository()
    manager = SemanticManager(
        repository,
        config=SemanticDefaults(mode="off", adapter="deterministic"),
    )

    status = manager.status()

    assert repository.calls == 1
    assert status["queue"]["queued"] == 501
    assert status["assessments"]["total"] == 100_001
    assert status["assessments"]["success"] == 1
    assert status["assessments"]["error"] == 100_000
    assert status["assessments"]["ood"] == 70_000
    assert status["assessments"]["by_status"] == status_counts
    assert status["assessments"]["by_domain"] == domain_counts
