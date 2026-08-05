from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SemanticDefaults
from agent_libos.models import (
    AuthoritativeApprovalFacts,
    CapabilityRight,
    HumanRequestStatus,
    SemanticApprovalCandidate,
    SemanticAssessment,
    SemanticAssessmentStatus,
    ShadowPolicyOutcome,
)
from agent_libos.models.exceptions import HumanApprovalRequired, ValidationError
from agent_libos.semantic import DeterministicApprovalBroker
from agent_libos.storage import (
    SQLiteStore,
    SemanticAssessmentJobRecord,
    SemanticAssessmentJobStatus,
    SemanticProjectionRetention,
    UnitOfWork,
)
from agent_libos.substrate import LocalResourceProviderSubstrate


pytestmark = pytest.mark.security


def _semantic_jobs(runtime: Runtime) -> tuple[Any, ...]:
    return runtime.uow.semantic.query_semantic_assessment_jobs(
        statuses=tuple(status.value for status in SemanticAssessmentJobStatus),
        projection_expires_before=None,
        limit=500,
    )


def _semantic_assessments(runtime: Runtime) -> tuple[Any, ...]:
    return tuple(
        runtime.uow.semantic.query_semantic_assessments(
            after=None,
            limit=500,
        ).records
    )


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


def _file_binding_snapshot(binding: Any) -> dict[str, Any] | None:
    if binding is None:
        return None
    return {
        "normalized_path": binding.normalized_path,
        "content_sha256": binding.content_sha256,
        "labels": binding.labels.to_dict(),
        "source_refs": [
            {
                "version": ref.version,
                "content_sha256": ref.content_sha256,
            }
            for ref in binding.source_refs
        ],
        "generation": binding.generation,
        "tombstoned": binding.tombstoned,
        "active": binding.active,
    }


def _data_flow_snapshot(context: Any) -> dict[str, Any]:
    return {
        "labels": context.labels.to_dict(),
        "source_refs": [
            {
                "version": ref.version,
                "content_sha256": ref.content_sha256,
            }
            for ref in context.source_refs
        ],
        "has_materialization": context.materialization_id is not None,
    }


def _authority_resource_snapshot(resource: str, pid: str) -> str:
    selected = resource.replace(pid, "<pid>")
    return "object:<runtime-object>" if selected.startswith("object:obj_") else selected


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class _DelayedApprovalAssessor:
    def __init__(self) -> None:
        self.approval_started = threading.Event()
        self.release = threading.Event()

    def assess(self, request: Any) -> SemanticAssessment:
        if request.kind.value == "approval":
            self.approval_started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release delayed assessment")
        return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)


def _run_approval_workload(runtime: Runtime, root: Path) -> dict[str, Any]:
    path = "reports/shadow-result.txt"
    target = root / path
    pid = runtime.process.spawn(
        goal="Write the reviewed report and preserve the Human decision boundary.",
        authority_manifest={
            "approval_policy": {
                "semantic_auto_approval": {
                    "schema_version": 1,
                    "rules": [
                        {
                            "rule_id": "reports-read-v1",
                            "authority_operation": "filesystem.read",
                            "resource": "filesystem:workspace:reports/*",
                            "rights": ["read"],
                        }
                    ],
                }
            }
        },
    )
    resource = runtime.filesystem.resource_for(path)
    runtime.capability.set_permission_policy(
        subject=pid,
        resource=resource,
        rights=[CapabilityRight.WRITE],
        policy=runtime.capability.ASK_EACH_TIME,
        issued_by="test.host",
    )

    with pytest.raises(HumanApprovalRequired):
        runtime.filesystem.write_text(pid, path, "approved business result\n")
    pending = runtime.human.pending()
    assert len(pending) == 1
    request_id = pending[0].request_id
    approved = runtime.human.drain_terminal_queue(auto_approve=True)
    assert [request.status for request in approved] == [HumanRequestStatus.APPROVED]

    result = runtime.filesystem.write_text(pid, path, "approved business result\n")
    request = runtime.human.requests.get(request_id)
    process = runtime.process.get(pid)
    binding = runtime.store.get_file_label_binding(path)
    capabilities = sorted(
        (
            _authority_resource_snapshot(capability.resource, pid),
            tuple(
                sorted(getattr(right, "value", str(right)) for right in capability.rights)
            ),
            capability.effect.value,
            capability.active,
            capability.uses_remaining,
        )
        for capability in runtime.capability.list_subject(pid)
    )
    business_effects = sorted(
        (
            effect.provider,
            effect.operation,
            effect.effect_state,
            effect.transaction_state,
        )
        for effect in runtime.store.list_external_effects(pid=pid)
        if effect.operation != "semantic.llm.assess"
    )
    release_requests = [
        item
        for item in runtime.human.list(pid)
        if item.payload.get("type") == "data_release_approval"
    ]
    return {
        "pid": pid,
        "path": path,
        "file_bytes": target.read_bytes(),
        "human_status": request.status.value if request is not None else None,
        "human_decision": dict(request.decision or {}) if request is not None else None,
        "human_revision": request.revision if request is not None else None,
        "process_status": process.status.value,
        "process_revision": process.revision,
        "capabilities": capabilities,
        "permission_policy": runtime.capability.permission_policy(
            pid,
            resource,
            CapabilityRight.WRITE,
        ),
        "file_binding": _file_binding_snapshot(binding),
        "data_flow_context": _data_flow_snapshot(runtime.data_flow.current_context()),
        "release_request_count": len(release_requests),
        "business_effects": business_effects,
        "write_result": _plain(result),
    }


def _wait_for_assessments(
    runtime: Runtime,
    *,
    minimum: int,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        raw = runtime.semantic.status()
        latest = raw if isinstance(raw, dict) else _plain(raw)
        queue = latest.get("queue") if isinstance(latest, dict) else None
        assessments = latest.get("assessments") if isinstance(latest, dict) else None
        if (
            isinstance(queue, dict)
            and isinstance(assessments, dict)
            and queue.get("queued") == 0
            and queue.get("leased") == 0
            and int(assessments.get("total") or 0) >= minimum
        ):
            return latest
        time.sleep(0.01)
    raise AssertionError(f"semantic worker did not become idle: {latest}")


def test_default_off_has_zero_semantic_capture_and_preserves_human_authority() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="off", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
        )
        try:
            outcome = _run_approval_workload(runtime, root)

            assert outcome["human_status"] == HumanRequestStatus.APPROVED.value
            assert outcome["file_bytes"] == b"approved business result\n"
            assert outcome["release_request_count"] == 0
            assert _semantic_jobs(runtime) == ()
            assert _semantic_assessments(runtime) == ()
        finally:
            runtime.close()


def test_shadow_and_off_have_identical_authority_business_and_label_outcomes() -> None:
    off_config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="off", adapter="deterministic")
    )
    shadow_config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as off_dir, TemporaryDirectory() as shadow_dir:
        off_root = Path(off_dir)
        shadow_root = Path(shadow_dir)
        off_runtime = Runtime.open(
            "local",
            config=off_config,
            substrate=LocalResourceProviderSubstrate(off_root),
        )
        shadow_runtime = Runtime.open(
            "local",
            config=shadow_config,
            substrate=LocalResourceProviderSubstrate(shadow_root),
        )
        try:
            off_outcome = _run_approval_workload(off_runtime, off_root)
            shadow_outcome = _run_approval_workload(shadow_runtime, shadow_root)
            status = _wait_for_assessments(shadow_runtime, minimum=3)

            assert {
                key: value for key, value in shadow_outcome.items() if key != "pid"
            } == {
                key: value for key, value in off_outcome.items() if key != "pid"
            }
            assert _semantic_jobs(off_runtime) == ()
            assert _semantic_assessments(off_runtime) == ()
            assert len(_semantic_jobs(shadow_runtime)) >= 3
            assessments = _semantic_assessments(shadow_runtime)
            assert len(assessments) >= 3
            assert {
                assessment.kind for assessment in assessments
            }.issuperset({"approval", "root_goal", "provider_ingress"})
            assert status["actual_auto_approval"] == {
                "numerator": 0,
                "denominator": 0,
                "rate": None,
            }
        finally:
            shadow_runtime.close()
            off_runtime.close()


def test_classifier_failure_and_capture_failure_have_zero_authority_delta() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="off", adapter="deterministic")
    )
    with TemporaryDirectory() as baseline_dir, TemporaryDirectory() as failure_dir:
        baseline_root = Path(baseline_dir)
        failure_root = Path(failure_dir)
        baseline = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(baseline_root),
        )
        failure = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(failure_root),
        )
        try:
            def fail_capture(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("injected semantic capture failure")

            failure.human.set_request_capture(fail_capture)
            failure.process.add_post_commit_spawn_observer(fail_capture)
            baseline_outcome = _run_approval_workload(baseline, baseline_root)
            failure_outcome = _run_approval_workload(failure, failure_root)
            assert {
                key: value for key, value in failure_outcome.items() if key != "pid"
            } == {
                key: value for key, value in baseline_outcome.items() if key != "pid"
            }
            assert failure.process.post_commit_spawn_observer_failures

            pid = str(failure_outcome["pid"])
            candidate_data = (
                failure.authority_manifests.semantic_auto_approval_candidate(
                    pid,
                    authority_operation="filesystem.read",
                    resource="filesystem:workspace:reports/shadow-result.txt",
                    rights=["read"],
                )
            )
            assert candidate_data is not None
            candidate = SemanticApprovalCandidate.from_dict(candidate_data)
            facts = AuthoritativeApprovalFacts(
                **{
                    name: True
                    for name in AuthoritativeApprovalFacts.__dataclass_fields__
                }
            )
            capabilities_before = tuple(failure.capability.list_subject(pid))
            human_before = tuple(failure.human.list(pid))
            decision = DeterministicApprovalBroker().decide(
                assessment=SemanticAssessment(
                    status=SemanticAssessmentStatus.PROVIDER_ERROR,
                ),
                facts=facts,
                policy_sha256=candidate.policy_sha256,
                candidate=candidate,
            )

            assert decision.outcome is ShadowPolicyOutcome.REQUIRE_HUMAN
            assert tuple(failure.capability.list_subject(pid)) == capabilities_before
            assert tuple(failure.human.list(pid)) == human_before
        finally:
            failure.close()
            baseline.close()


def test_mode_off_scrubs_legacy_unclaimed_projection_without_claiming() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "runtime.sqlite"
        projection = {
            "action_id": "filesystem.read",
            "projection_mode": "redacted",
            "redacted_intent": "review the report",
        }
        queued = SemanticAssessmentJobRecord(
            job_id="legacy-queued-job",
            kind="approval",
            status=SemanticAssessmentJobStatus.QUEUED,
            domain="filesystem",
            pid="pid-legacy",
            request_id="request-legacy",
            operation_id="operation-legacy",
            bindings={
                "artifact_sha256": "a" * 64,
                "input_sha256": "b" * 64,
                "feature_snapshot_sha256": "c" * 64,
                "policy_sha256": "d" * 64,
                "action_sha256": _canonical_sha256("filesystem.read"),
            },
            projection=projection,
            projection_sha256=_canonical_sha256(projection),
            projection_retention=SemanticProjectionRetention.REDACTED,
            projection_expires_at="2027-01-01T00:00:00Z",
            created_at="2026-08-05T00:00:00Z",
            updated_at="2026-08-05T00:00:00Z",
        )
        seed = SQLiteStore(db_path)
        UnitOfWork(seed).semantic.enqueue_semantic_assessment_job(queued)
        seed.close()

        workspace = root / "workspace"
        workspace.mkdir()
        runtime = Runtime(
            SQLiteStore(db_path),
            config=AgentLibOSConfig(
                semantic=SemanticDefaults(mode="off", adapter="deterministic")
            ),
            substrate=LocalResourceProviderSubstrate(workspace),
        )
        try:
            persisted = runtime.uow.semantic.get_semantic_assessment_job(
                queued.job_id
            )
            assert persisted is not None
            assert persisted.status is SemanticAssessmentJobStatus.CANCELLED
            assert persisted.attempt_count == 0
            assert persisted.projection == {}
            assert (
                persisted.projection_retention
                is SemanticProjectionRetention.HASH_ONLY
            )
            assert persisted.projection_sha256 == queued.projection_sha256
            assessment = runtime.uow.semantic.get_semantic_assessment(
                str(persisted.assessment_id)
            )
            assert assessment is not None
            assert assessment.status == "skipped_policy"
            assert assessment.shadow_outcome == "require_human"
            assert not any(
                effect.operation == "semantic.llm.assess"
                for effect in runtime.store.list_external_effects()
            )
        finally:
            runtime.close()


def test_late_assessment_records_but_never_overwrites_human_outcome() -> None:
    delayed = _DelayedApprovalAssessor()
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="scripted")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
            semantic_assessor=delayed,
        )
        try:
            pid = runtime.process.spawn(goal="Human decides while Shadow is running.")
            path = "reports/late-human.txt"
            resource = runtime.filesystem.resource_for(path)
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.WRITE],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.filesystem.write_text(pid, path, "human remains authoritative\n")
            request = runtime.human.pending()[0]
            assert delayed.approval_started.wait(timeout=5)

            approved = runtime.human.drain_terminal_queue(auto_approve=True)
            assert approved[0].status is HumanRequestStatus.APPROVED
            result = runtime.filesystem.write_text(
                pid,
                path,
                "human remains authoritative\n",
            )
            assert result.bytes_written > 0
            delayed.release.set()
            _wait_for_assessments(runtime, minimum=3)

            persisted_request = runtime.human.requests.get(request.request_id)
            assert persisted_request is not None
            assert persisted_request.status is HumanRequestStatus.APPROVED
            assessments = [
                item
                for item in _semantic_assessments(runtime)
                if item.kind == "approval" and item.request_id == request.request_id
            ]
            assert len(assessments) == 1
            assert assessments[0].human_outcome == HumanRequestStatus.APPROVED.value
            assert (root / path).read_text(encoding="utf-8") == (
                "human remains authoritative\n"
            )
        finally:
            delayed.release.set()
            runtime.close()


@pytest.mark.parametrize(
    (
        "resolution",
        "expected_human_status",
        "expected_recorded_outcome",
        "expected_shadow_outcome",
    ),
    (
        (
            "pending",
            HumanRequestStatus.PENDING,
            HumanRequestStatus.PENDING.value,
            ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE,
        ),
        (
            "approve",
            HumanRequestStatus.APPROVED,
            HumanRequestStatus.APPROVED.value,
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            "reject",
            HumanRequestStatus.REJECTED,
            HumanRequestStatus.REJECTED.value,
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            "cancel",
            HumanRequestStatus.CANCELLED,
            HumanRequestStatus.CANCELLED.value,
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            "reader_none",
            HumanRequestStatus.PENDING,
            None,
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            "reader_raises",
            HumanRequestStatus.PENDING,
            None,
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
    ),
)
def test_late_exact_read_assessment_requires_live_pending_human_binding(
    resolution: str,
    expected_human_status: HumanRequestStatus,
    expected_recorded_outcome: str | None,
    expected_shadow_outcome: ShadowPolicyOutcome,
) -> None:
    delayed = _DelayedApprovalAssessor()
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="scripted")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target = root / "reports" / "exact-read.txt"
        target.parent.mkdir(parents=True)
        target.write_text("human-owned read result\n", encoding="utf-8")
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
            semantic_assessor=delayed,
        )
        try:
            pid = runtime.process.spawn(
                goal="Read the reviewed report only after Human authorization.",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": {
                            "schema_version": 1,
                            "rules": [
                                {
                                    "rule_id": "reports-exact-read-v1",
                                    "authority_operation": "filesystem.read",
                                    "resource": "filesystem:workspace:reports/*",
                                    "rights": ["read"],
                                }
                            ],
                        }
                    }
                },
            )
            resource = runtime.filesystem.resource_for("reports/exact-read.txt")
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.filesystem.read_text(pid, "reports/exact-read.txt")
            request = runtime.human.pending()[0]
            assert delayed.approval_started.wait(timeout=5)

            claimed_jobs = [
                item
                for item in _semantic_jobs(runtime)
                if item.request_id == request.request_id
            ]
            assert len(claimed_jobs) == 1
            captured_facts = claimed_jobs[0].projection["features"]
            assert captured_facts["schema_valid"] is True
            assert captured_facts["request_is_exact_external_operation"] is True
            assert captured_facts["binding_current"] is True
            assert captured_facts["ceiling_matched"] is True

            if resolution == "approve":
                settled = runtime.human.drain_terminal_queue(auto_approve=True)
                assert settled[0].status is expected_human_status
            elif resolution == "reject":
                settled = runtime.human.drain_terminal_queue(auto_approve=False)
                assert settled[0].status is expected_human_status
            elif resolution == "cancel":
                assert runtime.human.cancel_pending_for_process(
                    pid,
                    actor="test.host",
                    reason="exercise stale Shadow binding",
                ) == [request.request_id]
            elif resolution == "reader_none":
                runtime.semantic._human_outcome_reader = lambda _request_id: None
            elif resolution == "reader_raises":
                runtime.semantic._human_outcome_reader = (
                    lambda _request_id: (_ for _ in ()).throw(
                        RuntimeError("Human reader unavailable")
                    )
                )
            authority_after_human = tuple(runtime.capability.list_subject(pid))
            policy_after_human = runtime.capability.permission_policy(
                pid,
                resource,
                CapabilityRight.READ,
            )

            delayed.release.set()
            _wait_for_assessments(runtime, minimum=2)

            assessment = next(
                item
                for item in _semantic_assessments(runtime)
                if item.request_id == request.request_id
            )
            assert assessment.human_outcome == expected_recorded_outcome
            assert assessment.shadow_outcome == expected_shadow_outcome.value
            if expected_shadow_outcome is ShadowPolicyOutcome.REQUIRE_HUMAN:
                assert "binding_current" in assessment.missing_predicates
            else:
                assert "binding_current" in assessment.proven_predicates
            persisted_request = runtime.human.requests.get(request.request_id)
            assert persisted_request is not None
            assert persisted_request.status is expected_human_status
            assert tuple(runtime.capability.list_subject(pid)) == authority_after_human
            assert runtime.capability.permission_policy(
                pid,
                resource,
                CapabilityRight.READ,
            ) == policy_after_human
            if persisted_request.status is HumanRequestStatus.PENDING:
                runtime.human.cancel_pending_for_process(
                    pid,
                    actor="test.cleanup",
                    reason="finish pending-only Shadow test",
                )
        finally:
            delayed.release.set()
            runtime.close()


@pytest.mark.parametrize("live_manifest_failure", ("missing", "raises"))
def test_late_exact_read_never_falls_back_to_frozen_manifest_candidate(
    live_manifest_failure: str,
) -> None:
    delayed = _DelayedApprovalAssessor()
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="scripted")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target = root / "reports" / "manifest-stale.txt"
        target.parent.mkdir(parents=True)
        target.write_text("manifest-bound read\n", encoding="utf-8")
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
            semantic_assessor=delayed,
        )
        try:
            pid = runtime.process.spawn(
                goal="Read only under the live reviewed manifest.",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": {
                            "schema_version": 1,
                            "rules": [
                                {
                                    "rule_id": "manifest-live-read-v1",
                                    "authority_operation": "filesystem.read",
                                    "resource": "filesystem:workspace:reports/*",
                                    "rights": ["read"],
                                }
                            ],
                        }
                    }
                },
            )
            resource = runtime.filesystem.resource_for("reports/manifest-stale.txt")
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.filesystem.read_text(pid, "reports/manifest-stale.txt")
            request = runtime.human.pending()[0]
            assert delayed.approval_started.wait(timeout=5)
            job = next(
                item
                for item in _semantic_jobs(runtime)
                if item.request_id == request.request_id
            )
            assert job.projection["candidate"] is not None

            class MissingLiveAuthority:
                def get_for_process(self, _pid: str) -> Any:
                    if live_manifest_failure == "raises":
                        raise RuntimeError("live manifest reader unavailable")
                    return None

            runtime.semantic._authority = MissingLiveAuthority()
            delayed.release.set()
            _wait_for_assessments(runtime, minimum=2)

            assessment = next(
                item
                for item in _semantic_assessments(runtime)
                if item.request_id == request.request_id
            )
            assert assessment.human_outcome == HumanRequestStatus.PENDING.value
            assert assessment.shadow_outcome == ShadowPolicyOutcome.REQUIRE_HUMAN.value
            assert "manifest_current" in assessment.missing_predicates
            assert "policy_current" in assessment.missing_predicates
            assert runtime.human.requests.get(request.request_id).status is (
                HumanRequestStatus.PENDING
            )
            runtime.human.cancel_pending_for_process(
                pid,
                actor="test.cleanup",
                reason="finish manifest-stale Shadow test",
            )
        finally:
            delayed.release.set()
            runtime.close()


def test_semantic_surface_has_no_machine_settlement_or_mutation_entrypoint() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target = root / "reports" / "candidate.txt"
        target.parent.mkdir(parents=True)
        target.write_text("candidate", encoding="utf-8")
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
        )
        try:
            pid = runtime.process.spawn(
                goal="Request an exact report read.",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": {
                            "schema_version": 1,
                            "rules": [
                                {
                                    "rule_id": "candidate-read-v1",
                                    "authority_operation": "filesystem.read",
                                    "resource": "filesystem:workspace:reports/*",
                                    "rights": ["read"],
                                }
                            ],
                        }
                    }
                },
            )
            resource = runtime.filesystem.resource_for("reports/candidate.txt")
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.filesystem.read_text(pid, "reports/candidate.txt")
            pending = runtime.human.pending()
            assert len(pending) == 1

            raw_candidate = runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.read",
                resource=resource,
                rights=["read"],
            )
            assert raw_candidate is not None
            assert raw_candidate["resource"] == resource
            assert runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.write",
                resource=resource,
                rights=["read"],
            ) is None
            assert runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/*",
                rights=["read"],
            ) is None
            assert runtime.authority_manifests.semantic_auto_approval_candidate(
                pid,
                authority_operation="filesystem.read",
                resource=resource,
                rights=["write"],
            ) is None
            with pytest.raises(ValidationError):
                runtime.authority_manifests.semantic_auto_approval_candidate(
                    pid,
                    authority_operation="filesystem.*",
                    resource=resource,
                    rights=["read"],
                )
            candidate = SemanticApprovalCandidate.from_dict(raw_candidate)
            facts = AuthoritativeApprovalFacts(
                **{
                    name: True
                    for name in AuthoritativeApprovalFacts.__dataclass_fields__
                }
            )
            before_capabilities = tuple(runtime.capability.list_subject(pid))
            decision = DeterministicApprovalBroker().decide(
                assessment=SemanticAssessment(
                    status=SemanticAssessmentStatus.SUCCESS,
                ),
                facts=facts,
                policy_sha256=candidate.policy_sha256,
                candidate=candidate,
            )

            assert decision.outcome is ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE
            assert tuple(runtime.capability.list_subject(pid)) == before_capabilities
            assert runtime.human.requests.get(pending[0].request_id).status is HumanRequestStatus.PENDING
            semantic = runtime.semantic
            for forbidden_name in (
                "approve",
                "deny",
                "reject",
                "settle",
                "issue_capability",
                "grant_capability",
                "terminalize_human_request",
            ):
                assert not hasattr(semantic, forbidden_name)
        finally:
            runtime.close()
