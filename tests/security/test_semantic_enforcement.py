from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.capability.effect_binding import APPROVAL_BINDING_KEY
from agent_libos.models import (
    CapabilityRight,
    Capability,
    DeterministicDenyDecision,
    EventType,
    DataLabels,
    HumanRequestStatus,
    ProcessStatus,
    SemanticApprovalCandidate,
    SemanticApprovalBindingV2,
    SemanticApprovalRule,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticControlStateV1,
    SemanticFinding,
    SemanticFindingSeverity,
    SemanticFindingSource,
    SemanticReasonCode,
    SemanticPolicyEpochV1,
    SemanticRuntimeMode,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    ValidationError,
)
from agent_libos.semantic.enforcement import (
    HostSemanticAuthorityValidator,
    HostSemanticAutoApprovalSettlement,
    HostSemanticControlFence,
    HostSemanticRateBudget,
    SemanticAuthorityControlView,
)
from agent_libos.semantic.flow import (
    FlowCoverageStatus,
    SemanticFlowService,
)
from agent_libos.semantic.settlement import HostSemanticDenySettlement
from agent_libos.storage import SQLiteStore
from agent_libos.storage.semantic_v6 import SemanticFlowEdgeRecord
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.ids import utc_now


pytestmark = pytest.mark.security

_A = "a" * 64
_B = "b" * 64


def _candidate(
    *,
    operation: str = "filesystem.read",
    right: str = "read",
) -> SemanticApprovalCandidate:
    resource_kind = operation.split(".", 1)[0]
    resource = (
        "filesystem:workspace:reports/semantic-cas.txt"
        if resource_kind == "filesystem"
        else f"{resource_kind}:security-test"
    )
    return SemanticApprovalCandidate(
        rule_id="security-test-rule",
        authority_operation=operation,
        resource=resource,
        rights=(right,),
        manifest_id="manifest-security-test",
        manifest_sha256=_A,
        policy_sha256=_B,
    )


def _successful_assessment() -> SemanticAssessment:
    return SemanticAssessment(
        status=SemanticAssessmentStatus.SUCCESS,
        confidence_bps=10_000,
        calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
    )


def _authority_fixture(
    provenance_validator: Any,
) -> tuple[
    SemanticPolicyEpochV1,
    SemanticControlStateV1,
    SemanticApprovalBindingV2,
    Capability,
    dict[str, Any],
    HostSemanticAuthorityValidator,
]:
    issued = datetime.now(timezone.utc)
    issued_at = issued.isoformat()
    expires_at = (issued + timedelta(seconds=60)).isoformat()
    tenant = "3" * 64
    resource = "filesystem:workspace:reports/semantic-cas.txt"
    classifier_profile_id = "security-classifier-profile"
    classifier_profile_sha256 = hashlib.sha256(
        json.dumps(
            classifier_profile_id,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    classifier_model_sha256 = "5" * 64
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-security-test",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(tenant,),
        auto_approval_rules=(
            SemanticApprovalRule(
                rule_id="security-test-rule",
                authority_operation="filesystem.read",
                resource=resource,
                rights=("read",),
            ),
        ),
        hard_deny_rules=(),
        classifier_profile_id=classifier_profile_id,
        classifier_profile_sha256=classifier_profile_sha256,
        classifier_model_sha256=classifier_model_sha256,
        created_at=issued_at,
    )
    epoch_sha256 = epoch.canonical_sha256()
    control = SemanticControlStateV1(
        revision=0,
        generation=1,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch_sha256,
        tripped=False,
        trip_code=None,
        updated_at=issued_at,
    )
    binding = SemanticApprovalBindingV2(
        request_id="request-security-test",
        request_revision=0,
        pid="pid-security-test",
        operation_id="operation-security-test",
        effect_id="eff_security-test",
        authority_operation="filesystem.read",
        resource=resource,
        right="read",
        canonical_args_hash=_A,
        target_state_version=1,
        manifest_id="manifest-security-test",
        manifest_sha256=_A,
        ceiling_sha256=_B,
        policy_epoch_id=epoch.epoch_id,
        policy_epoch_sha256=epoch_sha256,
        control_generation=control.generation,
        assessment_id="assessment-security-test",
        assessment_sha256=_A,
        classifier_profile_sha256=classifier_profile_sha256,
        classifier_model_sha256=classifier_model_sha256,
        tenant_bucket_sha256=tenant,
        source_labels_sha256=_A,
        source_refs_sha256=_B,
        flow_snapshot_sha256="4" * 64,
        sink_identity_sha256=None,
        tool_schema_sha256=None,
        provider_spec_sha256=None,
        nonce="nonce-security-test",
        issued_at=issued_at,
        expires_at=expires_at,
    )
    capability = Capability(
        cap_id="cap_semantic-security-test",
        subject=binding.pid,
        resource=binding.resource,
        rights={binding.right},
        constraints={APPROVAL_BINDING_KEY: binding.to_dict()},
        issued_by=f"policy:semantic:{epoch.epoch_id}",
        issued_at=issued_at,
        expires_at=expires_at,
        delegable=False,
        revocable=True,
        uses_remaining=1,
        metadata={
            "semantic_auto_approval": {
                "schema_version": 1,
                "binding_sha256": binding.canonical_sha256(),
                "request_id": binding.request_id,
                "assessment_id": binding.assessment_id,
                "policy_epoch_id": binding.policy_epoch_id,
                "settlement_id": "settlement-security-test",
                "budget_bucket_id": HostSemanticRateBudget.bucket_id_for(
                    epoch_id=epoch.epoch_id,
                    tenant_bucket_sha256=tenant,
                    rule_id="security-test-rule",
                ),
                "matched_rule_id": "security-test-rule",
            }
        },
    )
    current: dict[str, Any] = {
        "control": control,
        "fence_allowed": True,
        "fence_calls": 0,
        "local_latch_calls": 0,
    }

    class _FenceRepository:
        def fence_semantic_control_state(self, _expected: Any) -> bool:
            current["fence_calls"] += 1
            return bool(current["fence_allowed"])

    control_resolver = lambda: SemanticAuthorityControlView(  # noqa: E731
        current["control"],
        epoch,
    )

    def local_safety_latch() -> None:
        current["local_latch_calls"] += 1

    validator = HostSemanticAuthorityValidator(
        control_resolver=control_resolver,
        control_fence=HostSemanticControlFence(
            _FenceRepository(),
            control_resolver=control_resolver,
        ),
        local_safety_latch=local_safety_latch,
        provenance_validator=provenance_validator,
        now=lambda: issued + timedelta(seconds=1),
    )
    return epoch, control, binding, capability, current, validator


class _AuthorityProbe:
    def __init__(self) -> None:
        self.grants = 0
        self.terminalizations = 0
        self.binding_factories = 0

    def grant_once(self, **_kwargs: Any) -> Any:
        self.grants += 1
        raise AssertionError("negative semantic test reached Capability issuance")

    def terminalize(self, *_args: Any, **_kwargs: Any) -> Any:
        self.terminalizations += 1
        raise AssertionError("negative semantic test reached Human terminalization")

    def binding(self, *_args: Any, **_kwargs: Any) -> Any:
        self.binding_factories += 1
        raise AssertionError("negative semantic test reached BindingV2 creation")


def _settlement(probe: _AuthorityProbe) -> HostSemanticAutoApprovalSettlement:
    return HostSemanticAutoApprovalSettlement(
        probe,
        transaction=nullcontext,
        machine_terminalizer=probe.terminalize,
        live_binding_factory=probe.binding,
    )


@pytest.mark.parametrize(
    "assessment",
    (
        SemanticAssessment(
            status=SemanticAssessmentStatus.SUCCESS,
            findings=(
                SemanticFinding(
                    code=SemanticReasonCode.RISK_DETECTED,
                    severity=SemanticFindingSeverity.LOW,
                    confidence_bps=10_000,
                    evidence_sha256=_A,
                    source=SemanticFindingSource.MODEL,
                ),
            ),
            confidence_bps=10_000,
            calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
        ),
        SemanticAssessment(
            status=SemanticAssessmentStatus.OOD,
            confidence_bps=10_000,
            calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
            ood=True,
        ),
        SemanticAssessment(status=SemanticAssessmentStatus.PROVIDER_ERROR),
        SemanticAssessment(status=SemanticAssessmentStatus.INVALID_SCHEMA),
        SemanticAssessment(status=SemanticAssessmentStatus.TIMEOUT),
    ),
    ids=("model-finding", "ood", "provider-error", "invalid-schema", "timeout"),
)
def test_model_findings_ood_and_errors_cannot_terminalize_or_issue_authority(
    assessment: SemanticAssessment,
) -> None:
    probe = _AuthorityProbe()

    with pytest.raises(
        CapabilityDenied,
        match="classifier evidence cannot satisfy an allow predicate",
    ):
        _settlement(probe).settle_exact_once(
            request_id="request-security-test",
            expected_revision=0,
            job_id="job-security-test",
            assessment_id="assessment-security-test",
            assessment=assessment,
            candidate=_candidate(),
            semantic_terminalizer=(
                lambda _capability, _binding, _settlement: True
            ),
        )

    assert probe.grants == 0
    assert probe.terminalizations == 0
    assert probe.binding_factories == 0


@pytest.mark.parametrize(
    ("operation", "right"),
    (
        ("filesystem.write", "write"),
        ("filesystem.delete", "delete"),
        ("shell.run", "execute"),
        ("jsonrpc.call", "execute"),
        ("mcp.call", "execute"),
    ),
)
def test_high_risk_actions_are_structurally_unreachable_from_machine_settlement(
    operation: str,
    right: str,
) -> None:
    probe = _AuthorityProbe()

    with pytest.raises(CapabilityDenied, match="outside action catalog v1"):
        _settlement(probe).settle_exact_once(
            request_id="request-security-test",
            expected_revision=0,
            job_id="job-security-test",
            assessment_id="assessment-security-test",
            assessment=_successful_assessment(),
            candidate=_candidate(operation=operation, right=right),
            semantic_terminalizer=(
                lambda _capability, _binding, _settlement: True
            ),
        )

    assert probe.grants == 0
    assert probe.terminalizations == 0
    assert probe.binding_factories == 0


def test_deterministic_machine_deny_reasons_are_a_closed_non_model_set() -> None:
    executable = {
        SemanticReasonCode.MALFORMED_REQUEST,
        SemanticReasonCode.STALE_BINDING,
        SemanticReasonCode.STALE_MANIFEST,
        SemanticReasonCode.STALE_POLICY,
        SemanticReasonCode.DATA_FLOW_DENIED,
        SemanticReasonCode.POLICY_HARD_DENY,
        SemanticReasonCode.DIGEST_DRIFT,
    }

    for reason in SemanticReasonCode:
        kwargs = {
            "request_id": "request-hard-deny-security-test",
            "request_revision": 0,
            "pid": "pid-hard-deny-security-test",
            "effect_id": "eff_hard-deny-security-test",
            "reason_codes": (reason,),
            "policy_sha256": _A,
            "evidence_sha256": _B,
            "decided_at": utc_now(),
        }
        if reason in executable:
            assert DeterministicDenyDecision(**kwargs).reason_codes == (reason,)
        else:
            with pytest.raises(ValueError, match="human-overridable reason"):
                DeterministicDenyDecision(**kwargs)

    assert {
        SemanticReasonCode.RISK_DETECTED,
        SemanticReasonCode.OUT_OF_DISTRIBUTION,
        SemanticReasonCode.PROVIDER_ERROR,
        SemanticReasonCode.TIMEOUT,
        SemanticReasonCode.CEILING_MISS,
        SemanticReasonCode.HIGH_RISK_ACTION,
        SemanticReasonCode.HARD_POLICY_VIOLATION,
    }.isdisjoint(executable)


def test_epoch_revoke_and_binding_drift_block_unconsumed_machine_authority() -> None:
    provenance_calls: list[str] = []

    def provenance_validator(**facts: Any) -> None:
        provenance_calls.append(str(facts["phase"]))
        binding = facts["binding"]
        context = facts["context"]
        if context.get("args_sha256") != binding.canonical_args_hash:
            raise CapabilityDenied("semantic operation arguments changed")
        if context.get("flow_snapshot_sha256") != binding.flow_snapshot_sha256:
            raise CapabilityDenied("semantic flow snapshot changed")

    epoch, control, binding, capability, current, validator = _authority_fixture(
        provenance_validator
    )
    valid_context = {
        "args_sha256": binding.canonical_args_hash,
        "flow_snapshot_sha256": binding.flow_snapshot_sha256,
    }
    validator(
        binding=binding.to_dict(),
        phase="authorize",
        capability=capability,
        context=valid_context,
        effect_id=None,
    )
    assert provenance_calls == ["authorize"]

    with pytest.raises(CapabilityDenied, match="arguments changed"):
        validator(
            binding=binding.to_dict(),
            phase="prepare",
            capability=capability,
            context={**valid_context, "args_sha256": _B},
            effect_id=binding.effect_id,
        )
    assert provenance_calls == ["authorize", "prepare"]

    current["control"] = SemanticControlStateV1(
        revision=control.revision + 1,
        generation=control.generation,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
        tripped=True,
        trip_code="binding_mismatch",
        updated_at=utc_now(),
    )
    with pytest.raises(
        CapabilityDenied,
        match="control no longer permits settlement",
    ):
        validator(
            binding=binding.to_dict(),
            phase="dispatch",
            capability=capability,
            context=valid_context,
            effect_id=binding.effect_id,
        )
    # Epoch/control failure is checked before potentially expensive live
    # provenance and before provider dispatch.
    assert provenance_calls == ["authorize", "prepare"]

    current["control"] = SemanticControlStateV1(
        revision=control.revision + 2,
        generation=control.generation + 1,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
        tripped=False,
        trip_code=None,
        updated_at=utc_now(),
    )
    with pytest.raises(CapabilityDenied, match="epoch or control generation changed"):
        validator(
            binding=binding.to_dict(),
            phase="dispatch",
            capability=capability,
            context=valid_context,
            effect_id=binding.effect_id,
        )
    assert provenance_calls == ["authorize", "prepare"]


def test_dispatch_control_fence_failure_prevents_provider_execution() -> None:
    """A lost durable fence stops before provenance and provider dispatch."""

    provenance_calls: list[str] = []
    provider_calls: list[str] = []
    _epoch, _control, binding, capability, current, validator = (
        _authority_fixture(
            lambda **facts: provenance_calls.append(str(facts["phase"]))
        )
    )
    context = {
        "args_sha256": binding.canonical_args_hash,
        "flow_snapshot_sha256": binding.flow_snapshot_sha256,
    }

    validator(
        binding=binding.to_dict(),
        phase="authorize",
        capability=capability,
        context=context,
        effect_id=None,
    )
    current["fence_allowed"] = False

    def dispatch() -> None:
        validator(
            binding=binding.to_dict(),
            phase="dispatch",
            capability=capability,
            context=context,
            effect_id=binding.effect_id,
        )
        provider_calls.append("dispatched")

    with pytest.raises(
        CapabilityDenied,
        match="control changed before settlement",
    ):
        dispatch()

    assert current["fence_calls"] == 1
    assert provenance_calls == ["authorize"]
    assert provider_calls == []


def test_unknown_flow_coverage_blocks_machine_authority_after_sqlite_reopen() -> None:
    with TemporaryDirectory() as temp_dir:
        database = Path(temp_dir) / "unknown-flow.sqlite"
        store = SQLiteStore(database)
        store.close()

        reopened = SQLiteStore(database)
        try:
            flow = SemanticFlowService(reopened)

            def require_complete_flow(**_facts: Any) -> None:
                report = flow.coverage("flowent_missing-security-evidence")
                if report.status is not FlowCoverageStatus.COMPLETE:
                    raise CapabilityDenied(
                        "semantic flow coverage is not complete"
                    )

            _epoch, _control, binding, capability, _current, validator = (
                _authority_fixture(require_complete_flow)
            )
            with pytest.raises(CapabilityDenied, match="coverage is not complete"):
                validator(
                    binding=binding.to_dict(),
                    phase="authorize",
                    capability=capability,
                    context={
                        "args_sha256": binding.canonical_args_hash,
                        "flow_snapshot_sha256": binding.flow_snapshot_sha256,
                    },
                    effect_id=None,
                )
        finally:
            reopened.close()


def test_flow_graph_never_persists_secret_payload_and_is_tenant_isolated() -> None:
    sentinel = "SEMANTIC_FLOW_SECRET_SENTINEL_never-persist"
    with TemporaryDirectory() as temp_dir:
        database = Path(temp_dir) / "semantic-flow.sqlite"
        tenant_a = "1" * 64
        tenant_b = "2" * 64
        store = SQLiteStore(database)
        try:
            flow = SemanticFlowService(store)
            first = flow.capture_root_goal(
                pid="pid-flow-a",
                content_sha256=_A,
                state_sha256=_B,
                provenance_sha256="c" * 64,
                labels=DataLabels(
                    tenant=f"tenant-a-{sentinel}",
                    principal=f"principal-a-{sentinel}",
                    origin=sentinel,
                ),
                tenant_bucket_sha256=tenant_a,
                created_at=utc_now(),
            )
            second = flow.capture_root_goal(
                # Reuse the same process identifier so the rejected edge is
                # specifically a tenant-boundary violation, not merely a PID
                # mismatch.
                pid="pid-flow-a",
                content_sha256="d" * 64,
                state_sha256="e" * 64,
                provenance_sha256="f" * 64,
                labels=DataLabels(
                    tenant=f"tenant-b-{sentinel}",
                    principal=f"principal-b-{sentinel}",
                    origin=sentinel,
                ),
                tenant_bucket_sha256=tenant_b,
                created_at=utc_now(),
            )
            assert first is not None and second is not None
            assert first.entities and first.activities
            assert second.entities and second.activities

            tenant_a_page = flow.list_entities(
                tenant_bucket_sha256=tenant_a,
                limit=100,
            )
            assert tenant_a_page["items"]
            assert {
                item["tenant_bucket_sha256"]
                for item in tenant_a_page["items"]
            } == {tenant_a}
            assert sentinel not in json.dumps(
                tenant_a_page,
                ensure_ascii=False,
                sort_keys=True,
            )

            cross_tenant = SemanticFlowEdgeRecord(
                edge_id="flowedge_cross-tenant-security-test",
                relation="direct",
                source_node_id=first.entities[0].entity_id,
                source_node_type="entity",
                target_node_id=second.activities[0].activity_id,
                target_node_type="activity",
                pid="pid-flow-a",
                provenance_sha256="9" * 64,
                created_at=utc_now(),
            )
            with pytest.raises(ValidationError, match="cross tenant"):
                store.append_semantic_flow_bundle(edges=(cross_tenant,))
            assert cross_tenant.edge_id not in {
                item["edge_id"]
                for item in flow.list_edges(
                    node_id=cross_tenant.source_node_id,
                    limit=100,
                )["items"]
            }
        finally:
            store.close()

        assert sentinel.encode("utf-8") not in database.read_bytes()

        reopened = SQLiteStore(database)
        try:
            flow = SemanticFlowService(reopened)
            tenant_a_page = flow.list_entities(
                tenant_bucket_sha256=tenant_a,
                limit=100,
            )
            tenant_b_page = flow.list_entities(
                tenant_bucket_sha256=tenant_b,
                limit=100,
            )
            assert len(tenant_a_page["items"]) == 1
            assert len(tenant_b_page["items"]) == 1
            assert (
                flow.coverage("flowent_missing-security-evidence").status
                is FlowCoverageStatus.UNKNOWN
            )
            assert sentinel not in json.dumps(
                {"tenant_a": tenant_a_page, "tenant_b": tenant_b_page},
                ensure_ascii=False,
                sort_keys=True,
            )
        finally:
            reopened.close()


def _pending_exact_read(runtime: Runtime, root: Path) -> tuple[str, str, str]:
    relative_path = "reports/semantic-cas.txt"
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("machine/Human CAS\n", encoding="utf-8")
    pid = runtime.process.spawn(goal="Read one exact local report.")
    resource = runtime.filesystem.resource_for(relative_path)
    runtime.capability.set_permission_policy(
        subject=pid,
        resource=resource,
        rights=[CapabilityRight.READ],
        policy=runtime.capability.ASK_EACH_TIME,
        issued_by="test.host",
    )
    with pytest.raises(HumanApprovalRequired):
        runtime.filesystem.read_text(pid, relative_path)
    request = runtime.human.pending()[0]
    assert request.payload["type"] == "external_operation_approval"
    return pid, resource, request.request_id


def test_human_and_machine_terminalization_share_single_cas_winner() -> None:
    """The narrow Host machine port and Human surface share one terminal CAS."""

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime = Runtime.open(
            "local",
            substrate=LocalResourceProviderSubstrate(root),
        )
        try:
            pid, _resource, request_id = _pending_exact_read(runtime, root)
            expected = runtime.human.get(request_id)
            binding = dict(expected.payload["effect_binding"])
            fence_epoch = SemanticPolicyEpochV1(
                epoch_id="epoch-security-cas-fence",
                generation=1,
                expected_previous_sha256=None,
                tenant_bucket_sha256s=("4" * 64,),
                auto_approval_rules=(
                    SemanticApprovalRule(
                        rule_id="security-cas-fence-read",
                        authority_operation="filesystem.read",
                        resource="filesystem:workspace:reports/*",
                        rights=("read",),
                    ),
                ),
                hard_deny_rules=(),
                created_at=utc_now(),
            )
            fence_policy_sha256 = fence_epoch.canonical_sha256()
            fence_control = SemanticControlStateV1(
                revision=0,
                generation=1,
                mode=SemanticRuntimeMode.CANARY_AUTO,
                active_epoch_id=fence_epoch.epoch_id,
                active_policy_sha256=fence_policy_sha256,
                tripped=False,
                trip_code=None,
                updated_at=utc_now(),
            )

            class _ControlFenceRepository:
                calls = 0

                def fence_semantic_control_state(self, _expected: Any) -> bool:
                    self.calls += 1
                    return True

            fence_repository = _ControlFenceRepository()
            hard_deny = DeterministicDenyDecision(
                request_id=expected.request_id,
                request_revision=expected.revision,
                pid=expected.pid,
                effect_id=binding["effect_id"],
                reason_codes=(SemanticReasonCode.POLICY_HARD_DENY,),
                policy_sha256=fence_policy_sha256,
                evidence_sha256=_B,
                decided_at=utc_now(),
            )
            machine_port = runtime.human._issue_host_semantic_machine_settlement_port()
            deny_port = HostSemanticDenySettlement(
                transaction=machine_port.transaction,
                machine_terminalizer=machine_port.terminalize,
                control_fence=HostSemanticControlFence(
                    fence_repository,
                    control_resolver=lambda: SemanticAuthorityControlView(
                        fence_control,
                        fence_epoch,
                    ),
                ),
            )

            # A valid deterministic proof cannot bypass the global off switch.
            with pytest.raises(
                ValidationError,
                match="machine rejection requires active semantic enforcement",
            ):
                deny_port.settle_deny(
                    request_id=request_id,
                    expected_revision=expected.revision,
                    decision=hard_deny,
                    semantic_terminalizer=lambda: True,
                )
            assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
            assert fence_repository.calls == 1

            # Explicit unit seam for an admitted canary control snapshot. The
            # production root supplies these Host-only callbacks from durable
            # control and append-only settlement evidence.
            runtime.human._host_semantic_settlement_recorder = (  # noqa: SLF001
                lambda *_args, **_kwargs: {
                    "settlement_id": "semantic-settlement-security-test"
                }
            )
            runtime.human._host_semantic_policy_preflight = (  # noqa: SLF001
                lambda request: hard_deny
                if request.request_id == request_id
                else None
            )
            runtime.human._host_semantic_mode_reader = (  # noqa: SLF001
                lambda: "canary_auto"
            )
            before_event_ids = {
                event.event_id for event in runtime.events.list(target=pid)
            }
            before_audit_ids = {
                record.record_id for record in runtime.audit.trace()
            }
            with pytest.raises(
                ValidationError,
                match="job/assessment deny CAS was not committed",
            ):
                deny_port.settle_deny(
                    request_id=request_id,
                    expected_revision=expected.revision,
                    decision=hard_deny,
                    semantic_terminalizer=lambda: False,
                )
            assert runtime.human.get(request_id) == expected
            assert runtime.process.get(pid).status is ProcessStatus.WAITING_HUMAN
            assert {
                event.event_id for event in runtime.events.list(target=pid)
            } == before_event_ids
            assert {
                record.record_id for record in runtime.audit.trace()
            } == before_audit_ids

            preview_sha256 = runtime.human.canonical_approval_preview(
                expected
            ).canonical_sha256()
            barrier = threading.Barrier(2)

            def human_settle() -> tuple[str, object]:
                barrier.wait(timeout=3)
                try:
                    return (
                        "ok",
                        runtime.human.reject(
                            request_id,
                            {"approved": False, "reason": "operator rejected"},
                            responder="human:gui",
                            expected_revision=expected.revision,
                            preview_sha256=preview_sha256,
                        ),
                    )
                except Exception as exc:  # the losing CAS is asserted below
                    return "error", exc

            def machine_settle() -> tuple[str, object]:
                barrier.wait(timeout=3)
                try:
                    return (
                        "ok",
                        deny_port.settle_deny(
                            request_id=request_id,
                            expected_revision=expected.revision,
                            decision=hard_deny,
                            semantic_terminalizer=lambda: True,
                        ),
                    )
                except Exception as exc:  # the losing CAS is asserted below
                    return "error", exc

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = [
                    executor.submit(human_settle),
                    executor.submit(machine_settle),
                ]
                results = [future.result(timeout=5) for future in outcomes]

            assert [kind for kind, _value in results].count("ok") == 1
            errors = [value for kind, value in results if kind == "error"]
            assert len(errors) == 1
            assert isinstance(errors[0], ValidationError)
            assert "not pending" in str(errors[0]) or "changed concurrently" in str(
                errors[0]
            )

            persisted = runtime.human.get(request_id)
            assert persisted.status is HumanRequestStatus.REJECTED
            assert persisted.revision == expected.revision + 1
            once_grants = [
                capability
                for capability in runtime.capability.list_subject(pid)
                if capability.effect.value == "allow"
                and capability.uses_remaining == 1
            ]
            assert once_grants == []

            response_events = [
                event
                for event in runtime.events.list(target=pid)
                if event.type
                in {EventType.HUMAN_RESPONSE, EventType.SEMANTIC_POLICY_RESPONSE}
                and event.payload.get("request_id") == request_id
            ]
            response_audits = [
                record
                for record in runtime.audit.trace(
                    target=f"human_request:{request_id}"
                )
                if record.action
                in {"human.response", "semantic.policy.deny"}
            ]
            assert len(response_events) == 1
            assert len(response_audits) == 1
        finally:
            runtime.close()


def test_machine_settlement_port_acquires_terminal_lock_before_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Host machine UoW must preserve Human's terminal->Store order."""

    runtime = Runtime.open("local")
    try:
        port = runtime.human._issue_host_semantic_machine_settlement_port()
        original_transaction = runtime.human.requests.transaction
        observations: list[bool] = []

        @contextmanager
        def checked_store_transaction() -> Any:
            owned = getattr(runtime.human._terminal_lock, "_is_owned", None)
            observations.append(bool(callable(owned) and owned()))
            with original_transaction():
                yield

        monkeypatch.setattr(
            runtime.human.requests,
            "transaction",
            checked_store_transaction,
        )

        with port.transaction():
            pass

        assert observations == [True]
    finally:
        runtime.close()


def test_deny_settlement_rejects_cross_request_or_revision_proof() -> None:
    decision = DeterministicDenyDecision(
        request_id="request-bound-deny-proof",
        request_revision=4,
        pid="pid-bound-deny-proof",
        effect_id="effect-bound-deny-proof",
        reason_codes=(SemanticReasonCode.POLICY_HARD_DENY,),
        policy_sha256=_A,
        evidence_sha256=_B,
        decided_at=utc_now(),
    )
    settlement = HostSemanticDenySettlement(
        transaction=nullcontext,
        machine_terminalizer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched deny proof reached Human terminalization")
        ),
        control_fence=None,
    )

    for request_id, revision in (
        ("different-request", decision.request_revision),
        (decision.request_id, decision.request_revision + 1),
    ):
        with pytest.raises(
            ValidationError,
            match="does not match the requested Human CAS",
        ):
            settlement.settle_deny(
                request_id=request_id,
                expected_revision=revision,
                decision=decision,
                semantic_terminalizer=lambda: True,
            )
