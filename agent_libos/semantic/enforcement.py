from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    approval_binding_sha256,
)
from agent_libos.models.capability import Capability
from agent_libos.models.exceptions import (
    CapabilityDenied,
    SemanticAuthorityTripDeferred,
    ValidationError,
)
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.models.semantic import (
    MachinePolicySettlementV1,
    SEMANTIC_ACTION_CATALOG_V1,
    SemanticApprovalBindingV2,
    SemanticApprovalCandidate,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticControlStateV1,
    SemanticMachineSettlementOutcome,
    SemanticPolicyEpochV1,
    SemanticReasonCode,
    SemanticRuntimeMode,
    SemanticTripCode,
)
from agent_libos.storage.semantic_v6 import (
    SemanticRateBudgetRecord,
    control_state_storage_record,
    semantic_rate_budget_bucket_id,
)
from agent_libos.semantic.exact_request import (
    ExactSemanticApprovalRequest,
    decode_exact_semantic_approval_request,
)
from agent_libos.utils.ids import new_id, utc_now


AUTO_APPROVAL_ACTION_CATALOG_V1 = SEMANTIC_ACTION_CATALOG_V1

_VALIDATION_PHASES = frozenset({"authorize", "prepare", "reserve", "dispatch"})


@dataclass(frozen=True, slots=True)
class SemanticAuthorityControlView:
    """One transaction-local view used for every machine authority check."""

    control: SemanticControlStateV1
    epoch: SemanticPolicyEpochV1

    def __post_init__(self) -> None:
        if not isinstance(self.control, SemanticControlStateV1):
            raise TypeError("semantic authority control view requires control state")
        if not isinstance(self.epoch, SemanticPolicyEpochV1):
            raise TypeError("semantic authority control view requires a policy epoch")


class SemanticSettlementConflict(ValidationError):
    """The job/request CAS lost; no machine authority may survive the UoW."""


class SemanticRateBudgetExceeded(CapabilityDenied):
    """The exact tenant/rule canary budget cannot admit another issuance."""


class SemanticSafetyTripRequired(CapabilityDenied):
    """Typed Host provenance violation that must revoke the active epoch."""

    def __init__(self, trip_code: SemanticTripCode, message: str) -> None:
        if not isinstance(trip_code, SemanticTripCode):
            raise TypeError("semantic safety trip requires a typed trip code")
        super().__init__(message)
        self.trip_code = trip_code


@dataclass(frozen=True, slots=True)
class SemanticRateBudgetReservation:
    """Payload-free receipt for one transaction-local inflight reservation."""

    bucket_id: str
    epoch_id: str
    tenant_bucket_sha256: str
    rule_id: str
    revision: int


class HostSemanticControlFence:
    """Lock an exact active control pointer through the caller's commit."""

    def __init__(
        self,
        repository: Any,
        *,
        control_resolver: Callable[[], SemanticAuthorityControlView],
    ) -> None:
        if repository is None:
            raise TypeError("semantic control fence requires a repository")
        if not callable(control_resolver):
            raise TypeError("semantic control fence resolver must be callable")
        self._repository = repository
        self._control_resolver = control_resolver

    def fence(
        self,
        *,
        expected_policy_sha256: str,
        allowed_modes: tuple[SemanticRuntimeMode, ...],
    ) -> SemanticAuthorityControlView:
        if (
            not isinstance(expected_policy_sha256, str)
            or len(expected_policy_sha256) != 64
        ):
            raise CapabilityDenied("semantic control fence policy digest is invalid")
        if not allowed_modes or any(
            not isinstance(mode, SemanticRuntimeMode) for mode in allowed_modes
        ):
            raise TypeError("semantic control fence modes must be typed")
        view = self._control_resolver()
        if not isinstance(view, SemanticAuthorityControlView):
            raise CapabilityDenied("semantic control fence view is invalid")
        control = view.control
        epoch = view.epoch
        if control.mode not in allowed_modes or control.tripped:
            raise CapabilityDenied("semantic control no longer permits settlement")
        if (
            control.active_policy_sha256 != expected_policy_sha256
            or epoch.canonical_sha256() != expected_policy_sha256
        ):
            raise CapabilityDenied("semantic policy epoch changed before settlement")
        fence = getattr(self._repository, "fence_semantic_control_state", None)
        if not callable(fence):
            raise CapabilityDenied("semantic control commit fence is unavailable")
        if not fence(control_state_storage_record(control)):
            raise CapabilityDenied("semantic control changed before settlement")
        return view


class HostSemanticRateBudget:
    """Atomic v1 canary budget backed by the Store's revision-CAS row.

    ``reserve`` is called only from the auto-settlement Unit of Work.  Its
    minute/day counters therefore commit with the Human request, Capability,
    assessment, and settlement receipt, or all roll back together.  A
    terminal lifecycle observer must first win its append-if-absent outcome
    claim and then call ``release`` in that same transaction; a bucket alone
    cannot identify which of several inflight settlements a duplicate event
    belongs to.  The bucket identity is stable across policy epochs: an epoch
    is settlement provenance and a live control fence, never a way to shard
    the per-tenant/logical-rule minute, day, or inflight limits.
    """

    def __init__(
        self,
        repository: Any,
        *,
        control_resolver: Callable[[], SemanticAuthorityControlView],
        control_fence: HostSemanticControlFence | None = None,
        now: Callable[[], datetime] | None = None,
        max_cas_attempts: int = 16,
    ) -> None:
        for operation in (
            "get_semantic_rate_budget",
            "compare_and_set_semantic_rate_budget",
        ):
            if not callable(getattr(repository, operation, None)):
                raise TypeError(
                    f"semantic rate budget repository lacks {operation}"
                )
        if not callable(control_resolver):
            raise TypeError("semantic rate budget control resolver must be callable")
        if now is not None and not callable(now):
            raise TypeError("semantic rate budget clock must be callable")
        if (
            isinstance(max_cas_attempts, bool)
            or not isinstance(max_cas_attempts, int)
            or not 1 <= max_cas_attempts <= 64
        ):
            raise ValueError("semantic rate budget CAS attempts must be in [1, 64]")
        self._repository = repository
        self._control_resolver = control_resolver
        if control_fence is not None and not isinstance(
            control_fence,
            HostSemanticControlFence,
        ):
            raise TypeError("semantic rate budget control fence is invalid")
        self._control_fence = control_fence or HostSemanticControlFence(
            repository,
            control_resolver=control_resolver,
        )
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._max_cas_attempts = max_cas_attempts

    def reserve(
        self,
        binding: SemanticApprovalBindingV2,
        *,
        rule_id: str,
        assessment: SemanticAssessment,
    ) -> SemanticRateBudgetReservation:
        """Reserve one issuance and one inflight slot using revision CAS."""

        if not isinstance(binding, SemanticApprovalBindingV2):
            raise TypeError("semantic rate budget requires a BindingV2")
        if not isinstance(assessment, SemanticAssessment):
            raise TypeError("semantic rate budget requires a typed assessment")
        if not isinstance(rule_id, str) or not rule_id:
            raise TypeError("semantic rate budget requires an exact rule_id")
        view = self._control_fence.fence(
            expected_policy_sha256=binding.policy_epoch_sha256,
            allowed_modes=(SemanticRuntimeMode.CANARY_AUTO,),
        )
        _require_active_budget_scope(view, binding=binding, rule_id=rule_id)
        _require_epoch_assessment_threshold(view.epoch, assessment)
        epoch = view.epoch
        bucket_id = _rate_bucket_id(
            epoch_id=binding.policy_epoch_id,
            tenant_bucket_sha256=binding.tenant_bucket_sha256,
            rule_id=rule_id,
        )
        now = _aware_now(self._now)
        now_text = _canonical_timestamp(now)
        for _attempt in range(self._max_cas_attempts):
            current = self._repository.get_semantic_rate_budget(bucket_id)
            if current is None:
                target = SemanticRateBudgetRecord(
                    bucket_id=bucket_id,
                    epoch_id=binding.policy_epoch_id,
                    tenant_bucket_sha256=binding.tenant_bucket_sha256,
                    rule_id=rule_id,
                    minute_window_started_at=now_text,
                    day_window_started_at=now_text,
                    minute_count=1,
                    day_count=1,
                    inflight_count=1,
                    revision=0,
                    updated_at=now_text,
                )
            else:
                _require_budget_record_scope(
                    current,
                    binding=binding,
                    rule_id=rule_id,
                    bucket_id=bucket_id,
                )
                minute_start = _timestamp(current.minute_window_started_at)
                day_start = _timestamp(current.day_window_started_at)
                if now < minute_start or now < day_start:
                    raise CapabilityDenied(
                        "semantic rate budget Host clock moved backwards"
                    )
                minute_count = current.minute_count
                day_count = current.day_count
                if now - minute_start >= timedelta(minutes=1):
                    minute_start = now
                    minute_count = 0
                if now - day_start >= timedelta(days=1):
                    day_start = now
                    day_count = 0
                if (
                    minute_count >= epoch.per_rule_per_minute_limit
                    or day_count >= epoch.per_rule_per_day_limit
                    or current.inflight_count >= epoch.max_inflight
                ):
                    raise SemanticRateBudgetExceeded(
                        "semantic exact tenant/rule canary budget is exhausted"
                    )
                target = SemanticRateBudgetRecord(
                    bucket_id=current.bucket_id,
                    epoch_id=current.epoch_id,
                    tenant_bucket_sha256=current.tenant_bucket_sha256,
                    rule_id=current.rule_id,
                    minute_window_started_at=_canonical_timestamp(minute_start),
                    day_window_started_at=_canonical_timestamp(day_start),
                    minute_count=minute_count + 1,
                    day_count=day_count + 1,
                    inflight_count=current.inflight_count + 1,
                    revision=current.revision + 1,
                    updated_at=now_text,
                )
            if self._repository.compare_and_set_semantic_rate_budget(
                current,
                target,
            ):
                return SemanticRateBudgetReservation(
                    bucket_id=bucket_id,
                    epoch_id=binding.policy_epoch_id,
                    tenant_bucket_sha256=binding.tenant_bucket_sha256,
                    rule_id=rule_id,
                    revision=target.revision,
                )
        raise SemanticSettlementConflict(
            "semantic rate budget CAS did not converge"
        )

    @staticmethod
    def bucket_id_for(
        *,
        epoch_id: str,
        tenant_bucket_sha256: str,
        rule_id: str,
    ) -> str:
        """Return the deterministic durable logical-rule bucket identity.

        ``epoch_id`` remains in this compatibility signature because callers
        also bind settlement provenance.  It is deliberately excluded from
        the bucket digest so a generation rotation cannot reset hard limits.
        """

        return _rate_bucket_id(
            epoch_id=epoch_id,
            tenant_bucket_sha256=tenant_bucket_sha256,
            rule_id=rule_id,
        )

    def release(self, bucket_id: str) -> SemanticRateBudgetRecord:
        """Release one inflight slot after a unique terminal outcome claim.

        This method deliberately rejects an already-empty bucket.  Treating a
        duplicate notification as a generic no-op would be unsafe when the
        same bucket contains a different live settlement; append-only outcome
        identity is the idempotency boundary.
        """

        if not isinstance(bucket_id, str) or not bucket_id:
            raise TypeError("semantic rate budget release requires bucket_id")
        now_text = _canonical_timestamp(_aware_now(self._now))
        for _attempt in range(self._max_cas_attempts):
            current = self._repository.get_semantic_rate_budget(bucket_id)
            if current is None:
                raise SemanticSettlementConflict(
                    "semantic rate budget bucket does not exist"
                )
            if current.inflight_count < 1:
                raise SemanticSettlementConflict(
                    "semantic rate budget has no inflight issuance to release"
                )
            target = SemanticRateBudgetRecord(
                bucket_id=current.bucket_id,
                epoch_id=current.epoch_id,
                tenant_bucket_sha256=current.tenant_bucket_sha256,
                rule_id=current.rule_id,
                minute_window_started_at=current.minute_window_started_at,
                day_window_started_at=current.day_window_started_at,
                minute_count=current.minute_count,
                day_count=current.day_count,
                inflight_count=current.inflight_count - 1,
                revision=current.revision + 1,
                updated_at=now_text,
            )
            if self._repository.compare_and_set_semantic_rate_budget(
                current,
                target,
            ):
                return target
        raise SemanticSettlementConflict(
            "semantic rate budget release CAS did not converge"
        )


class HostSemanticAuthorityValidator:
    """Fail-closed live epoch/kill-switch validator for BindingV2 grants.

    ``control_resolver`` and ``provenance_validator`` are Host composition
    dependencies.  The former must read control and epoch state in the caller's
    current Store transaction.  The latter verifies live FlowGraph, manifest,
    tool/provider/Sink, and target-state evidence.  Neither is model supplied.
    """

    def __init__(
        self,
        *,
        control_resolver: Callable[[], SemanticAuthorityControlView],
        provenance_validator: Callable[..., None],
        control_fence: HostSemanticControlFence,
        local_safety_latch: Callable[[], None],
        safety_trip: Callable[..., None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(control_resolver) or not callable(provenance_validator):
            raise TypeError("semantic authority validators must be callable")
        self._control_resolver = control_resolver
        self._provenance_validator = provenance_validator
        if not isinstance(control_fence, HostSemanticControlFence):
            raise TypeError("semantic authority validator requires a control fence")
        self._control_fence = control_fence
        if not callable(local_safety_latch):
            raise TypeError("semantic authority validator requires a local safety latch")
        self._local_safety_latch = local_safety_latch
        if safety_trip is not None and not callable(safety_trip):
            raise TypeError("semantic authority safety-trip observer must be callable")
        self._safety_trip = safety_trip
        self._now = now or (lambda: datetime.now(timezone.utc))

    def __call__(
        self,
        *,
        binding: Mapping[str, Any],
        phase: str,
        capability: Capability,
        context: Mapping[str, Any],
        effect_id: str | None,
    ) -> None:
        if phase not in _VALIDATION_PHASES:
            raise CapabilityDenied("unknown semantic authority validation phase")
        selected = SemanticApprovalBindingV2.from_dict(binding)
        view = (
            self._control_resolver()
            if phase == "authorize"
            else self._control_fence.fence(
                expected_policy_sha256=selected.policy_epoch_sha256,
                allowed_modes=(SemanticRuntimeMode.CANARY_AUTO,),
            )
        )
        if not isinstance(view, SemanticAuthorityControlView):
            raise CapabilityDenied("semantic control resolver returned an invalid view")
        _require_active_authority_view(
            view,
            selected,
            capability=capability,
            phase=phase,
            safety_trip=self._safety_trip,
            local_safety_latch=self._local_safety_latch,
        )
        control = view.control
        epoch = view.epoch
        expires_at = _timestamp(selected.expires_at)
        if _aware_now(self._now) >= expires_at:
            raise CapabilityDenied("semantic exact-once authority expired")
        try:
            self._provenance_validator(
                binding=selected,
                phase=phase,
                capability=capability,
                context=dict(context),
                effect_id=effect_id,
                control=control,
                epoch=epoch,
            )
        except SemanticSafetyTripRequired as violation:
            _trip_and_deny(
                self._safety_trip,
                violation.trip_code,
                binding=selected,
                capability=capability,
                phase=phase,
                message=str(violation),
                local_safety_latch=self._local_safety_latch,
            )


class HostSemanticAutoApprovalSettlement:
    """Host-only atomic minting path for one exact low-risk capability."""

    def __init__(
        self,
        capabilities: Any,
        *,
        transaction: Callable[[], AbstractContextManager[Any]],
        machine_terminalizer: Callable[..., tuple[HumanRequest, dict[str, Any]]],
        live_binding_factory: Callable[
            [
                HumanRequest,
                str,
                SemanticAssessment,
                SemanticApprovalCandidate,
            ],
            SemanticApprovalBindingV2,
        ],
        budget: HostSemanticRateBudget | None = None,
    ) -> None:
        for label, callback in (
            ("transaction", transaction),
            ("machine terminalizer", machine_terminalizer),
            ("live BindingV2 factory", live_binding_factory),
        ):
            if not callable(callback):
                raise TypeError(f"semantic settlement {label} must be callable")
        if budget is not None and not isinstance(budget, HostSemanticRateBudget):
            raise TypeError(
                "semantic auto settlement requires a HostSemanticRateBudget"
            )
        self._capabilities = capabilities
        self._transaction = transaction
        self._machine_terminalizer = machine_terminalizer
        self._live_binding_factory = live_binding_factory
        self._budget = budget

    def settle_exact_once(
        self,
        *,
        request_id: str,
        expected_revision: int,
        job_id: str,
        assessment_id: str,
        assessment: SemanticAssessment,
        candidate: SemanticApprovalCandidate,
        semantic_terminalizer: Callable[
            [Capability, SemanticApprovalBindingV2, MachinePolicySettlementV1],
            bool,
        ],
    ) -> tuple[HumanRequest, dict[str, Any]]:
        _validate_machine_assessment(assessment)
        _validate_candidate(candidate)
        if not callable(semantic_terminalizer):
            raise TypeError("semantic evidence terminalizer must be callable")

        def apply_authority(request: HumanRequest) -> Mapping[str, Any]:
            if self._budget is None:
                raise CapabilityDenied(
                    "semantic auto settlement has no durable rate budget"
                )
            binding = self._live_binding_factory(
                request,
                assessment_id,
                assessment,
                candidate,
            )
            if not isinstance(binding, SemanticApprovalBindingV2):
                raise ValidationError(
                    "semantic live binding factory returned an invalid BindingV2"
                )
            spec = _validate_live_request(
                request,
                binding=binding,
                assessment_id=assessment_id,
                assessment=assessment,
                candidate=candidate,
            )
            constraints = dict(spec.get("constraints") or {})
            constraints[APPROVAL_BINDING_KEY] = binding.to_dict()
            budget = self._budget.reserve(
                binding,
                rule_id=candidate.rule_id,
                assessment=assessment,
            )
            settlement_id = new_id("semantic-settlement")
            metadata = {
                "semantic_auto_approval": {
                    "schema_version": 1,
                    "binding_sha256": binding.canonical_sha256(),
                    "request_id": binding.request_id,
                    "assessment_id": binding.assessment_id,
                    "policy_epoch_id": binding.policy_epoch_id,
                    "matched_rule_id": candidate.rule_id,
                    "settlement_id": settlement_id,
                    "budget_bucket_id": budget.bucket_id,
                }
            }
            capability = self._capabilities.grant_once(
                subject=binding.pid,
                resource=binding.resource,
                rights=[binding.right],
                issued_by=f"policy:semantic:{binding.policy_epoch_id}",
                constraints=constraints,
                expires_at=binding.expires_at,
                metadata=metadata,
            )
            if (
                capability.uses_remaining != 1
                or capability.delegable
                or not capability.revocable
            ):
                raise CapabilityDenied(
                    "semantic authority issuer did not create an exact one-shot grant"
                )
            binding_sha256 = approval_binding_sha256(binding.to_dict())
            decision_projection = {
                "schema_version": 1,
                "outcome": "issued",
                "assessment_id": assessment_id,
                "job_id": job_id,
                "request_id": binding.request_id,
                "request_revision": binding.request_revision,
                "capability_id": capability.cap_id,
                "binding_sha256": binding_sha256,
                "matched_rule_id": candidate.rule_id,
                "policy_epoch_id": binding.policy_epoch_id,
                "policy_epoch_sha256": binding.policy_epoch_sha256,
                "budget_bucket_id": budget.bucket_id,
            }
            settlement = MachinePolicySettlementV1(
                settlement_id=settlement_id,
                assessment_id=assessment_id,
                job_id=job_id,
                request_id=binding.request_id,
                request_revision=binding.request_revision,
                pid=binding.pid,
                operation_id=binding.operation_id,
                effect_id=binding.effect_id,
                epoch_id=binding.policy_epoch_id,
                policy_sha256=binding.policy_epoch_sha256,
                tenant_bucket_sha256=binding.tenant_bucket_sha256,
                action_id=binding.authority_operation,
                outcome=SemanticMachineSettlementOutcome.ISSUED,
                capability_id=capability.cap_id,
                binding_sha256=binding_sha256,
                decision_sha256=_value_sha256(decision_projection),
                matched_rule_id=candidate.rule_id,
                reason_codes=(SemanticReasonCode.POLICY_MATCH,),
                created_at=utc_now(),
            )
            if semantic_terminalizer(capability, binding, settlement) is not True:
                raise SemanticSettlementConflict(
                    "semantic job/assessment/settlement CAS was not committed"
                )
            return {
                "schema_version": 1,
                "outcome": "issued",
                "assessment_id": assessment_id,
                "assessment_sha256": assessment.canonical_sha256(),
                "capability_id": capability.cap_id,
                "settlement_id": settlement.settlement_id,
                "binding_sha256": binding_sha256,
                "budget_bucket_id": budget.bucket_id,
                "policy_epoch_id": binding.policy_epoch_id,
                "policy_epoch_sha256": binding.policy_epoch_sha256,
                "control_generation": binding.control_generation,
                "expires_at": binding.expires_at,
                "uses": 1,
                "delegable": False,
                "revocable": True,
            }

        with self._transaction():
            return self._machine_terminalizer(
                request_id,
                expected_revision=expected_revision,
                status=HumanRequestStatus.APPROVED,
                decision={
                    "schema_version": 1,
                    "assessment_id": assessment_id,
                    "assessment_sha256": assessment.canonical_sha256(),
                    "matched_rule_id": candidate.rule_id,
                },
                responder="policy:semantic:auto",
                authority_applier=apply_authority,
                audit_action="semantic.policy.auto_approve",
            )


def _validate_machine_assessment(assessment: SemanticAssessment) -> None:
    if not isinstance(assessment, SemanticAssessment):
        raise TypeError("semantic settlement assessment has an invalid type")
    if (
        assessment.status is not SemanticAssessmentStatus.SUCCESS
        or assessment.ood
        or assessment.abstain
        or assessment.findings
        or assessment.data_findings
        or assessment.confidence_bps < 9_900
        or assessment.calibration_bucket
        is not SemanticCalibrationBucket.VERY_HIGH
    ):
        raise CapabilityDenied(
            "semantic classifier evidence cannot satisfy an allow predicate"
        )


def _validate_candidate(candidate: SemanticApprovalCandidate) -> None:
    if not isinstance(candidate, SemanticApprovalCandidate):
        raise TypeError("semantic settlement candidate has an invalid type")
    allowed = AUTO_APPROVAL_ACTION_CATALOG_V1.get(candidate.authority_operation)
    if (
        allowed is None
        or len(candidate.rights) != 1
        or candidate.rights[0] not in allowed
    ):
        raise CapabilityDenied(
            "semantic candidate is structurally outside action catalog v1"
        )


def _validate_live_request(
    request: HumanRequest,
    *,
    binding: SemanticApprovalBindingV2,
    assessment_id: str,
    assessment: SemanticAssessment,
    candidate: SemanticApprovalCandidate,
) -> Mapping[str, Any]:
    _validate_request_assessment_identity(
        request,
        binding=binding,
        assessment_id=assessment_id,
        assessment=assessment,
    )
    exact = decode_exact_semantic_approval_request(request)
    _validate_exact_request_binding(
        exact,
        binding=binding,
    )
    _validate_live_ceiling(candidate, binding=binding)
    _validate_requested_expiry(exact.capability, binding=binding)
    return exact.capability


def _validate_request_assessment_identity(
    request: HumanRequest,
    *,
    binding: SemanticApprovalBindingV2,
    assessment_id: str,
    assessment: SemanticAssessment,
) -> None:
    if request.status is not HumanRequestStatus.PENDING:
        raise SemanticSettlementConflict("semantic approval request is no longer pending")
    if (
        binding.request_id != request.request_id
        or binding.request_revision != request.revision
        or binding.pid != request.pid
        or binding.assessment_id != assessment_id
        or binding.assessment_sha256 != assessment.canonical_sha256()
    ):
        raise ValidationError("semantic BindingV2 request or assessment changed")


def _validate_exact_request_binding(
    exact: ExactSemanticApprovalRequest,
    *,
    binding: SemanticApprovalBindingV2,
) -> None:
    if (
        exact.action_id != binding.authority_operation
        or exact.resource != binding.resource
        or exact.right != binding.right
        or binding.operation_id
        != (
            exact.context.get("operation_id")
            if isinstance(exact.context.get("operation_id"), str)
            else None
        )
    ):
        raise CapabilityDenied("semantic exact operation identity changed")
    if exact.binding != binding.to_legacy_effect_binding():
        raise CapabilityDenied("semantic legacy and BindingV2 effects do not agree")


def _validate_live_ceiling(
    candidate: SemanticApprovalCandidate,
    *,
    binding: SemanticApprovalBindingV2,
) -> None:
    if (
        candidate.authority_operation != binding.authority_operation
        or candidate.rights != (binding.right,)
        or candidate.manifest_id != binding.manifest_id
        or candidate.manifest_sha256 != binding.manifest_sha256
        or candidate.policy_sha256 != binding.ceiling_sha256
    ):
        raise CapabilityDenied("semantic live binding no longer matches its ceiling")
    projected_resource = candidate.resource
    if projected_resource.startswith("digest:"):
        if projected_resource != "digest:" + _value_sha256(binding.resource):
            raise CapabilityDenied("semantic live resource changed since capture")
    elif projected_resource != binding.resource:
        raise CapabilityDenied("semantic live resource changed since capture")


def _validate_requested_expiry(
    spec: Mapping[str, Any],
    *,
    binding: SemanticApprovalBindingV2,
) -> None:
    requested_expiry = spec.get("expires_at")
    if requested_expiry is not None and (
        not isinstance(requested_expiry, str)
        or _timestamp(binding.expires_at) > _timestamp(requested_expiry)
    ):
        raise CapabilityDenied("semantic capability expiry exceeds the request")


def _require_active_authority_view(
    view: SemanticAuthorityControlView,
    binding: SemanticApprovalBindingV2,
    *,
    capability: Capability,
    phase: str,
    safety_trip: Callable[..., None] | None,
    local_safety_latch: Callable[[], None],
) -> None:
    control = view.control
    epoch = view.epoch
    if control.mode is not SemanticRuntimeMode.CANARY_AUTO:
        raise CapabilityDenied("semantic auto approval is disabled")
    if control.tripped:
        raise CapabilityDenied("semantic policy epoch is safety-tripped")
    if not _control_epoch_identity_matches_binding(view, binding):
        raise CapabilityDenied("semantic policy epoch or control generation changed")
    if (
        epoch.classifier_profile_id is None
        or epoch.classifier_profile_sha256 is None
        or epoch.classifier_model_sha256 is None
    ):
        raise CapabilityDenied(
            "semantic canary auto approval requires a pinned external classifier"
        )
    if (
        binding.classifier_profile_sha256 != epoch.classifier_profile_sha256
        or binding.classifier_model_sha256 != epoch.classifier_model_sha256
    ):
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.BINDING_MISMATCH,
            binding=binding,
            capability=capability,
            phase=phase,
            message="semantic classifier profile or model binding changed",
            local_safety_latch=local_safety_latch,
        )
    if not _control_epoch_matches_binding(view, binding):
        raise CapabilityDenied("semantic policy epoch or control generation changed")
    if binding.tenant_bucket_sha256 not in epoch.tenant_bucket_sha256s:
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.CROSS_TENANT,
            binding=binding,
            capability=capability,
            phase=phase,
            message="semantic tenant bucket is not in the active canary",
            local_safety_latch=local_safety_latch,
        )
    issuance = capability.metadata.get("semantic_auto_approval")
    matched_rule_id = (
        issuance.get("matched_rule_id")
        if isinstance(issuance, Mapping)
        else None
    )
    if (
        not isinstance(matched_rule_id, str)
        or not _epoch_rule_matches(epoch, binding, rule_id=matched_rule_id)
    ):
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.UNAUTHORIZED_EFFECT,
            binding=binding,
            capability=capability,
            phase=phase,
            message="semantic BindingV2 is outside the active exact rule",
            local_safety_latch=local_safety_latch,
        )
    expected_bucket_id = _rate_bucket_id(
        epoch_id=binding.policy_epoch_id,
        tenant_bucket_sha256=binding.tenant_bucket_sha256,
        rule_id=matched_rule_id,
    )
    if issuance.get("budget_bucket_id") != expected_bucket_id:
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.BINDING_MISMATCH,
            binding=binding,
            capability=capability,
            phase=phase,
            message="semantic capability budget provenance changed",
            local_safety_latch=local_safety_latch,
        )
    if binding.sink_identity_sha256 is not None:
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.SECRET_EGRESS,
            binding=binding,
            capability=capability,
            phase=phase,
            message="semantic catalog v1 forbids egress-bound authority",
            local_safety_latch=local_safety_latch,
        )
    if (
        binding.tool_schema_sha256 is not None
        or binding.provider_spec_sha256 is not None
    ):
        _trip_and_deny(
            safety_trip,
            SemanticTripCode.BINDING_MISMATCH,
            binding=binding,
            capability=capability,
            phase=phase,
            message=(
                "semantic catalog v1 local-read authority has unexpected "
                "tool or provider provenance"
            ),
            local_safety_latch=local_safety_latch,
        )


def _trip_and_deny(
    safety_trip: Callable[..., None] | None,
    trip_code: SemanticTripCode,
    *,
    binding: SemanticApprovalBindingV2,
    capability: Capability,
    phase: str,
    message: str,
    local_safety_latch: Callable[[], None],
) -> None:
    evidence_sha256 = _value_sha256(
        {
            "schema_version": 1,
            "trip_code": trip_code.value,
            "phase": phase,
            "capability_id": capability.cap_id,
            "binding_sha256": binding.canonical_sha256(),
            "policy_epoch_id": binding.policy_epoch_id,
            "policy_epoch_sha256": binding.policy_epoch_sha256,
        }
    )
    if phase != "authorize":
        try:
            local_safety_latch()
        except Exception as exc:
            raise CapabilityDenied(
                "semantic local safety latch failed closed"
            ) from exc
        raise SemanticAuthorityTripDeferred(
            message,
            trip_code=trip_code.value,
            evidence_sha256=evidence_sha256,
            tenant_bucket_sha256=binding.tenant_bucket_sha256,
        )
    if safety_trip is not None:
        try:
            safety_trip(
                trip_code=trip_code,
                evidence_sha256=evidence_sha256,
                tenant_bucket_sha256=binding.tenant_bucket_sha256,
            )
        except Exception as exc:
            raise CapabilityDenied(
                "semantic safety trip failed closed"
            ) from exc
    raise CapabilityDenied(message)


def _control_epoch_matches_binding(
    view: SemanticAuthorityControlView,
    binding: SemanticApprovalBindingV2,
) -> bool:
    if not _control_epoch_identity_matches_binding(view, binding):
        return False
    control = view.control
    epoch = view.epoch
    return (
        control.active_policy_sha256 == binding.policy_epoch_sha256
        and epoch.canonical_sha256() == binding.policy_epoch_sha256
    )


def _control_epoch_identity_matches_binding(
    view: SemanticAuthorityControlView,
    binding: SemanticApprovalBindingV2,
) -> bool:
    control = view.control
    epoch = view.epoch
    return (
        control.active_epoch_id == binding.policy_epoch_id
        and control.generation == binding.control_generation
        and epoch.epoch_id == binding.policy_epoch_id
        and epoch.generation == binding.control_generation
    )


def _epoch_rule_matches(
    epoch: SemanticPolicyEpochV1,
    binding: SemanticApprovalBindingV2,
    *,
    rule_id: str,
) -> bool:
    for rule in epoch.auto_approval_rules:
        if (
            rule.rule_id != rule_id
            or rule.authority_operation != binding.authority_operation
            or rule.rights != (binding.right,)
        ):
            continue
        if rule.resource.endswith("*"):
            if binding.resource.startswith(rule.resource[:-1]) and "*" not in binding.resource:
                return True
        elif rule.resource == binding.resource:
            return True
    return False


def _require_active_budget_scope(
    view: SemanticAuthorityControlView,
    *,
    binding: SemanticApprovalBindingV2,
    rule_id: str,
) -> None:
    control = view.control
    epoch = view.epoch
    if control.mode is not SemanticRuntimeMode.CANARY_AUTO or control.tripped:
        raise CapabilityDenied("semantic canary budget is not active")
    if (
        control.active_epoch_id != binding.policy_epoch_id
        or control.active_policy_sha256 != binding.policy_epoch_sha256
        or control.generation != binding.control_generation
        or epoch.epoch_id != binding.policy_epoch_id
        or epoch.canonical_sha256() != binding.policy_epoch_sha256
        or epoch.generation != binding.control_generation
        or binding.tenant_bucket_sha256 not in epoch.tenant_bucket_sha256s
    ):
        raise CapabilityDenied("semantic canary budget scope changed")
    selected = next(
        (rule for rule in epoch.auto_approval_rules if rule.rule_id == rule_id),
        None,
    )
    if selected is None:
        raise CapabilityDenied("semantic canary budget rule is not active")
    if (
        selected.authority_operation != binding.authority_operation
        or selected.rights != (binding.right,)
        or not _rule_resource_matches(selected.resource, binding.resource)
    ):
        raise CapabilityDenied("semantic canary budget rule binding changed")


def _require_epoch_assessment_threshold(
    epoch: SemanticPolicyEpochV1,
    assessment: SemanticAssessment,
) -> None:
    if (
        assessment.confidence_bps < epoch.minimum_confidence_bps
        or assessment.calibration_bucket
        is not epoch.required_calibration_bucket
    ):
        raise CapabilityDenied(
            "semantic classifier evidence is below the active epoch threshold"
        )


def _rule_resource_matches(rule_resource: str, exact_resource: str) -> bool:
    if rule_resource.endswith("*"):
        return (
            exact_resource.startswith(rule_resource[:-1])
            and "*" not in exact_resource
        )
    return rule_resource == exact_resource


def _require_budget_record_scope(
    record: SemanticRateBudgetRecord,
    *,
    binding: SemanticApprovalBindingV2,
    rule_id: str,
    bucket_id: str,
) -> None:
    if (
        record.bucket_id != bucket_id
        or record.tenant_bucket_sha256 != binding.tenant_bucket_sha256
        or record.rule_id != rule_id
    ):
        raise CapabilityDenied("semantic persisted rate budget scope is inconsistent")


def _rate_bucket_id(
    *,
    epoch_id: str,
    tenant_bucket_sha256: str,
    rule_id: str,
) -> str:
    if not isinstance(epoch_id, str) or not epoch_id:
        raise ValueError("semantic rate budget epoch provenance is invalid")
    return semantic_rate_budget_bucket_id(
        tenant_bucket_sha256=tenant_bucket_sha256,
        rule_id=rule_id,
    )


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CapabilityDenied("semantic Host clock returned an invalid timestamp")
    return value.astimezone(timezone.utc)


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValidationError("semantic timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError("semantic timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AUTO_APPROVAL_ACTION_CATALOG_V1",
    "HostSemanticAuthorityValidator",
    "HostSemanticAutoApprovalSettlement",
    "HostSemanticControlFence",
    "HostSemanticRateBudget",
    "SemanticAuthorityControlView",
    "SemanticRateBudgetExceeded",
    "SemanticRateBudgetReservation",
    "SemanticSafetyTripRequired",
    "SemanticSettlementConflict",
]
