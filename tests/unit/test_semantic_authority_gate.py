from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_libos.sdk.protected_operations as protected_operations_module
from agent_libos.capability.effect_binding import (
    approval_binding_sha256,
    is_semantic_approval_binding,
    normalize_approval_binding,
)
from agent_libos.models import Capability, CapabilityEffect
from agent_libos.models import CapabilityDecision
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.exceptions import SemanticAuthorityTripDeferred
from agent_libos.models.semantic import (
    SemanticApprovalBindingV2,
    SemanticApprovalRule,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticControlStateV1,
    SemanticPolicyEpochV1,
    SemanticRuntimeMode,
    SemanticTripCode,
)
from agent_libos.semantic.enforcement import (
    HostSemanticAuthorityValidator,
    HostSemanticControlFence,
    HostSemanticRateBudget,
    SemanticAuthorityControlView,
    SemanticRateBudgetExceeded,
    SemanticSettlementConflict,
)
from agent_libos.semantic.control import SemanticRuntimeControl
from agent_libos.storage import SQLiteStore
from agent_libos.storage.semantic_v6 import SemanticRateBudgetRecord


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _epoch() -> SemanticPolicyEpochV1:
    return SemanticPolicyEpochV1(
        epoch_id="epoch-unit-v1",
        generation=3,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(_A,),
        auto_approval_rules=(
            SemanticApprovalRule(
                rule_id="read-reports",
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/*",
                rights=("read",),
            ),
        ),
        hard_deny_rules=(),
        classifier_profile_id="semantic-external-prod",
        classifier_profile_sha256=_A,
        classifier_model_sha256=_B,
        created_at="2026-08-07T00:00:00+00:00",
    )


def _control(epoch: SemanticPolicyEpochV1) -> SemanticControlStateV1:
    return SemanticControlStateV1(
        revision=2,
        generation=epoch.generation,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
        tripped=False,
        trip_code=None,
        updated_at="2026-08-07T00:00:01+00:00",
    )


def _binding(epoch: SemanticPolicyEpochV1) -> SemanticApprovalBindingV2:
    return SemanticApprovalBindingV2(
        request_id="human-request-unit-v2",
        request_revision=0,
        pid="pid-unit-v2",
        operation_id="operation-unit-v2",
        effect_id="eff_unit-v2",
        authority_operation="filesystem.read",
        resource="filesystem:workspace:reports/result.txt",
        right="read",
        canonical_args_hash=_C,
        target_state_version="file-version-1",
        manifest_id="manifest-unit-v2",
        manifest_sha256=_D,
        ceiling_sha256=_E,
        policy_epoch_id=epoch.epoch_id,
        policy_epoch_sha256=epoch.canonical_sha256(),
        control_generation=epoch.generation,
        assessment_id="assessment-unit-v2",
        assessment_sha256=_F,
        classifier_profile_sha256=epoch.classifier_profile_sha256,
        classifier_model_sha256=_B,
        tenant_bucket_sha256=_A,
        source_labels_sha256=_C,
        source_refs_sha256=_D,
        flow_snapshot_sha256=_E,
        sink_identity_sha256=None,
        tool_schema_sha256=None,
        provider_spec_sha256=None,
        nonce="nonce-unit-v2",
        issued_at="2026-08-07T00:00:10+00:00",
        expires_at="2026-08-07T00:01:10+00:00",
    )


def _capability(binding: SemanticApprovalBindingV2) -> Capability:
    return Capability(
        cap_id="cap-unit-v2",
        subject=binding.pid,
        resource=binding.resource,
        rights={binding.right},
        constraints={"approval_binding": binding.to_dict()},
        issued_by=f"policy:semantic:{binding.policy_epoch_id}",
        issued_at=binding.issued_at,
        expires_at=binding.expires_at,
        delegable=False,
        revocable=True,
        effect=CapabilityEffect.ALLOW,
        uses_remaining=1,
        metadata={
            "semantic_auto_approval": {
                "schema_version": 1,
                "binding_sha256": binding.canonical_sha256(),
                "request_id": binding.request_id,
                "assessment_id": binding.assessment_id,
                "policy_epoch_id": binding.policy_epoch_id,
                "matched_rule_id": "read-reports",
                "settlement_id": "semantic-settlement-unit-v2",
                "budget_bucket_id": HostSemanticRateBudget.bucket_id_for(
                    epoch_id=binding.policy_epoch_id,
                    tenant_bucket_sha256=binding.tenant_bucket_sha256,
                    rule_id="read-reports",
                ),
            }
        },
    )


def _assessment(confidence_bps: int = 10_000) -> SemanticAssessment:
    return SemanticAssessment(
        status=SemanticAssessmentStatus.SUCCESS,
        confidence_bps=confidence_bps,
        calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
    )


class _BudgetRepository:
    def __init__(self) -> None:
        self.record: SemanticRateBudgetRecord | None = None
        self.fence_allowed = True

    def get_semantic_rate_budget(
        self,
        bucket_id: str,
    ) -> SemanticRateBudgetRecord | None:
        current = self.record
        return current if current is not None and current.bucket_id == bucket_id else None

    def fence_semantic_control_state(self, _expected: object) -> bool:
        return self.fence_allowed

    def compare_and_set_semantic_rate_budget(
        self,
        expected: SemanticRateBudgetRecord | None,
        target: SemanticRateBudgetRecord,
    ) -> bool:
        if self.record != expected:
            return False
        self.record = target
        return True


def _control_fence(
    control: SemanticControlStateV1,
    epoch: SemanticPolicyEpochV1,
    *,
    repository: _BudgetRepository | None = None,
) -> HostSemanticControlFence:
    return HostSemanticControlFence(
        repository or _BudgetRepository(),
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
    )


def test_binding_v2_uses_strict_round_trip_and_digest() -> None:
    binding = _binding(_epoch())
    raw = binding.to_dict()

    assert is_semantic_approval_binding(raw)
    assert normalize_approval_binding(raw) == raw
    assert approval_binding_sha256(raw) == binding.canonical_sha256()

    with pytest.raises(ValidationError, match="fields must be exactly"):
        normalize_approval_binding({**raw, "permit": True})
    with pytest.raises(ValidationError, match="schema_version"):
        normalize_approval_binding({**raw, "schema_version": None})
    with pytest.raises(ValidationError, match="fields must be exactly"):
        removed = dict(raw)
        removed.pop("control_generation")
        normalize_approval_binding(removed)


def test_live_authority_validator_rechecks_control_epoch_and_classifier() -> None:
    epoch = _epoch()
    binding = _binding(epoch)
    control = _control(epoch)
    provenance_phases: list[str] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **facts: provenance_phases.append(facts["phase"]),
        control_fence=_control_fence(control, epoch),
        local_safety_latch=lambda: None,
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    validator(
        binding=binding.to_dict(),
        phase="dispatch",
        capability=_capability(binding),
        context={},
        effect_id=binding.effect_id,
    )
    assert provenance_phases == ["dispatch"]

    for invalid_control in (
        replace(control, mode=SemanticRuntimeMode.SHADOW, active_epoch_id=None, active_policy_sha256=None),
        replace(control, tripped=True, trip_code="unsafe_review"),
        replace(control, generation=control.generation + 1),
    ):
        denied = HostSemanticAuthorityValidator(
            control_resolver=lambda selected=invalid_control: SemanticAuthorityControlView(selected, epoch),
            provenance_validator=lambda **_facts: None,
            control_fence=_control_fence(invalid_control, epoch),
            local_safety_latch=lambda: None,
            now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
        )
        with pytest.raises(CapabilityDenied):
            denied(
                binding=binding.to_dict(),
                phase="dispatch",
                capability=_capability(binding),
                context={},
                effect_id=binding.effect_id,
            )


def test_live_authority_validator_stale_epoch_does_not_trip_rotated_classifier() -> None:
    previous_epoch = _epoch()
    stale_binding = _binding(previous_epoch)
    active_epoch = replace(
        previous_epoch,
        epoch_id="epoch-unit-v2",
        generation=previous_epoch.generation + 1,
        expected_previous_sha256=previous_epoch.canonical_sha256(),
        classifier_profile_sha256=_E,
        classifier_model_sha256=_F,
    )
    active_control = _control(active_epoch)
    trips: list[dict[str, object]] = []
    local_latches: list[bool] = []
    provenance_calls: list[str] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(
            active_control,
            active_epoch,
        ),
        provenance_validator=lambda **facts: provenance_calls.append(facts["phase"]),
        control_fence=_control_fence(active_control, active_epoch),
        local_safety_latch=lambda: local_latches.append(True),
        safety_trip=lambda **facts: trips.append(facts),
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(
        CapabilityDenied,
        match="policy epoch or control generation changed",
    ):
        validator(
            binding=stale_binding.to_dict(),
            phase="authorize",
            capability=_capability(stale_binding),
            context={},
            effect_id=None,
        )

    assert trips == []
    assert local_latches == []
    assert provenance_calls == []


def test_live_authority_validator_trips_current_epoch_classifier_mismatch() -> None:
    epoch = _epoch()
    control = _control(epoch)
    mismatched_binding = replace(
        _binding(epoch),
        classifier_profile_sha256=_F,
    )
    trips: list[dict[str, object]] = []
    provenance_calls: list[str] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **facts: provenance_calls.append(facts["phase"]),
        control_fence=_control_fence(control, epoch),
        local_safety_latch=lambda: None,
        safety_trip=lambda **facts: trips.append(facts),
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(
        CapabilityDenied,
        match="classifier profile or model binding changed",
    ):
        validator(
            binding=mismatched_binding.to_dict(),
            phase="authorize",
            capability=_capability(mismatched_binding),
            context={},
            effect_id=None,
        )

    assert [trip["trip_code"] for trip in trips] == [
        SemanticTripCode.BINDING_MISMATCH
    ]
    assert provenance_calls == []


def test_live_authority_validator_trips_classifier_and_policy_digest_mismatch() -> None:
    epoch = _epoch()
    control = _control(epoch)
    mismatched_binding = replace(
        _binding(epoch),
        policy_epoch_sha256=_E,
        classifier_profile_sha256=_F,
    )
    trips: list[dict[str, object]] = []
    provenance_calls: list[str] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **facts: provenance_calls.append(facts["phase"]),
        control_fence=_control_fence(control, epoch),
        local_safety_latch=lambda: None,
        safety_trip=lambda **facts: trips.append(facts),
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(
        CapabilityDenied,
        match="classifier profile or model binding changed",
    ):
        validator(
            binding=mismatched_binding.to_dict(),
            phase="authorize",
            capability=_capability(mismatched_binding),
            context={},
            effect_id=None,
        )

    assert [trip["trip_code"] for trip in trips] == [
        SemanticTripCode.BINDING_MISMATCH
    ]
    assert provenance_calls == []


def test_live_authority_validator_fences_dispatch_against_control_commit() -> None:
    epoch = _epoch()
    binding = _binding(epoch)
    control = _control(epoch)
    repository = _BudgetRepository()
    repository.fence_allowed = False
    provenance_calls: list[str] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **facts: provenance_calls.append(facts["phase"]),
        control_fence=_control_fence(control, epoch, repository=repository),
        local_safety_latch=lambda: None,
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(CapabilityDenied, match="control changed"):
        validator(
            binding=binding.to_dict(),
            phase="dispatch",
            capability=_capability(binding),
            context={},
            effect_id=binding.effect_id,
        )

    assert provenance_calls == []


def test_live_authority_validator_denies_expired_or_unpinned_epoch() -> None:
    epoch = _epoch()
    binding = _binding(epoch)
    control = _control(epoch)
    expired = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **_facts: None,
        control_fence=_control_fence(control, epoch),
        local_safety_latch=lambda: None,
        now=lambda: datetime(2026, 8, 7, 0, 2, tzinfo=timezone.utc),
    )
    with pytest.raises(CapabilityDenied, match="expired"):
        expired(
            binding=binding.to_dict(),
            phase="authorize",
            capability=_capability(binding),
            context={},
            effect_id=None,
        )

    unpinned = replace(
        epoch,
        classifier_profile_id=None,
        classifier_profile_sha256=None,
        classifier_model_sha256=None,
    )
    unpinned_control = _control(unpinned)
    denied = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(
            unpinned_control,
            unpinned,
        ),
        provenance_validator=lambda **_facts: None,
        control_fence=_control_fence(unpinned_control, unpinned),
        local_safety_latch=lambda: None,
        now=lambda: datetime.now(timezone.utc) - timedelta(days=1),
    )
    with pytest.raises(CapabilityDenied, match="pinned external classifier"):
        denied(
            binding=replace(
                binding,
                policy_epoch_sha256=unpinned.canonical_sha256(),
            ).to_dict(),
            phase="prepare",
            capability=_capability(binding),
            context={},
            effect_id=None,
        )


def test_live_authority_validator_durably_classifies_immediate_trip_reasons() -> None:
    epoch = _epoch()
    control = _control(epoch)
    binding = _binding(epoch)
    trips: list[dict[str, object]] = []
    local_latches: list[bool] = []
    validator = HostSemanticAuthorityValidator(
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        provenance_validator=lambda **_facts: None,
        control_fence=_control_fence(control, epoch),
        local_safety_latch=lambda: local_latches.append(True),
        safety_trip=lambda **facts: trips.append(facts),
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    cases = (
        (
            replace(binding, tenant_bucket_sha256=_B),
            SemanticTripCode.CROSS_TENANT,
        ),
        (
            replace(binding, sink_identity_sha256=_F),
            SemanticTripCode.SECRET_EGRESS,
        ),
        (
            replace(binding, classifier_profile_sha256=_F),
            SemanticTripCode.BINDING_MISMATCH,
        ),
        (
            replace(binding, tool_schema_sha256=_F),
            SemanticTripCode.BINDING_MISMATCH,
        ),
        (
            replace(binding, provider_spec_sha256=_F),
            SemanticTripCode.BINDING_MISMATCH,
        ),
    )
    for changed, code in cases:
        changed = replace(
            changed,
            policy_epoch_sha256=epoch.canonical_sha256(),
        )
        with pytest.raises(SemanticAuthorityTripDeferred) as raised:
            validator(
                binding=changed.to_dict(),
                phase="dispatch",
                capability=_capability(changed),
                context={},
                effect_id=changed.effect_id,
            )
        assert raised.value.trip_code == code.value
        assert len(raised.value.evidence_sha256) == 64
        assert "reports/result.txt" not in json.dumps(
            raised.value.__dict__,
            default=str,
        )
    assert len(local_latches) == len(cases)
    assert trips == []

    capability = _capability(binding)
    issuance = dict(capability.metadata["semantic_auto_approval"])
    issuance["budget_bucket_id"] = "semantic-budget:" + _F
    with pytest.raises(SemanticAuthorityTripDeferred) as raised:
        validator(
            binding=binding.to_dict(),
            phase="dispatch",
            capability=replace(
                capability,
                metadata={"semantic_auto_approval": issuance},
            ),
            context={},
            effect_id=binding.effect_id,
        )
    assert raised.value.trip_code == SemanticTripCode.BINDING_MISMATCH.value


def test_host_rate_budget_reserves_counts_and_releases_only_once() -> None:
    epoch = replace(
        _epoch(),
        max_inflight=1,
        per_rule_per_minute_limit=2,
        per_rule_per_day_limit=3,
    )
    control = _control(epoch)
    binding = _binding(epoch)
    repository = _BudgetRepository()
    now = [datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc)]
    budget = HostSemanticRateBudget(
        repository,
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        now=lambda: now[0],
    )

    first = budget.reserve(
        binding,
        rule_id="read-reports",
        assessment=_assessment(),
    )
    assert repository.record is not None
    assert repository.record.minute_count == 1
    assert repository.record.day_count == 1
    assert repository.record.inflight_count == 1

    with pytest.raises(SemanticRateBudgetExceeded, match="exhausted"):
        budget.reserve(
            binding,
            rule_id="read-reports",
            assessment=_assessment(),
        )

    released = budget.release(first.bucket_id)
    assert released.minute_count == 1
    assert released.day_count == 1
    assert released.inflight_count == 0
    with pytest.raises(SemanticSettlementConflict, match="no inflight"):
        budget.release(first.bucket_id)

    second = budget.reserve(
        binding,
        rule_id="read-reports",
        assessment=_assessment(),
    )
    assert second.bucket_id == first.bucket_id
    assert repository.record is not None
    assert repository.record.minute_count == 2
    assert repository.record.day_count == 2
    budget.release(second.bucket_id)

    now[0] += timedelta(minutes=1, seconds=1)
    budget.reserve(
        binding,
        rule_id="read-reports",
        assessment=_assessment(),
    )
    assert repository.record is not None
    assert repository.record.minute_count == 1
    assert repository.record.day_count == 3


def test_host_rate_budget_cannot_reset_through_epoch_rotation_after_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic-rate-budget-rotation.db"
    now = datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc)
    first_epoch = replace(
        _epoch(),
        epoch_id="budget-epoch-1",
        generation=1,
        expected_previous_sha256=None,
        per_rule_per_minute_limit=2,
        per_rule_per_day_limit=2,
        max_inflight=2,
        created_at="2026-08-07T00:00:00+00:00",
    )
    store = SQLiteStore(database)
    first_control = SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first_epoch,
        now=lambda: "2026-08-07T00:00:00+00:00",
    )
    first_control.admit()
    first_budget = HostSemanticRateBudget(
        store,
        control_resolver=first_control.authority_view,
        now=lambda: now,
    )
    with store.transaction():
        first_reservation = first_budget.reserve(
            _binding(first_epoch),
            rule_id="read-reports",
            assessment=_assessment(),
        )
        first_budget.release(first_reservation.bucket_id)
    store.close()

    second_epoch = replace(
        first_epoch,
        epoch_id="budget-epoch-2",
        generation=2,
        expected_previous_sha256=first_epoch.canonical_sha256(),
        per_rule_per_minute_limit=1,
        per_rule_per_day_limit=1,
        max_inflight=1,
        created_at="2026-08-07T00:00:10+00:00",
    )
    reopened = SQLiteStore(database)
    second_control = SemanticRuntimeControl(
        reopened,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second_epoch,
        now=lambda: "2026-08-07T00:00:20+00:00",
    )
    second_control.admit()
    second_budget = HostSemanticRateBudget(
        reopened,
        control_resolver=second_control.authority_view,
        now=lambda: now,
    )
    with reopened.transaction():
        with pytest.raises(SemanticRateBudgetExceeded, match="exhausted"):
            second_budget.reserve(
                _binding(second_epoch),
                rule_id="read-reports",
                assessment=_assessment(),
            )
    assert first_reservation.bucket_id == HostSemanticRateBudget.bucket_id_for(
        epoch_id=second_epoch.epoch_id,
        tenant_bucket_sha256=_A,
        rule_id="read-reports",
    )
    rows = reopened.conn.execute(
        "SELECT minute_count, day_count, inflight_count "
        "FROM semantic_rate_budgets"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(1, 1, 0)]
    reopened.close()


def test_rotated_rate_budget_reservation_rolls_back_and_has_one_cas_winner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic-rate-budget-cas.db"
    now = datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc)
    first_epoch = replace(
        _epoch(),
        epoch_id="budget-cas-epoch-1",
        generation=1,
        expected_previous_sha256=None,
        per_rule_per_minute_limit=2,
        per_rule_per_day_limit=2,
        max_inflight=2,
    )
    store = SQLiteStore(database)
    first_control = SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first_epoch,
        now=lambda: "2026-08-07T00:00:00+00:00",
    )
    first_control.admit()
    first_budget = HostSemanticRateBudget(
        store,
        control_resolver=first_control.authority_view,
        now=lambda: now,
    )
    with store.transaction():
        reservation = first_budget.reserve(
            _binding(first_epoch),
            rule_id="read-reports",
            assessment=_assessment(),
        )
        first_budget.release(reservation.bucket_id)

    second_epoch = replace(
        first_epoch,
        epoch_id="budget-cas-epoch-2",
        generation=2,
        expected_previous_sha256=first_epoch.canonical_sha256(),
    )
    second_control = SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second_epoch,
        now=lambda: "2026-08-07T00:00:10+00:00",
    )
    second_control.admit()
    second_budget = HostSemanticRateBudget(
        store,
        control_resolver=second_control.authority_view,
        now=lambda: now,
    )
    with pytest.raises(RuntimeError, match="injected budget rollback"):
        with store.transaction():
            second_budget.reserve(
                _binding(second_epoch),
                rule_id="read-reports",
                assessment=_assessment(),
            )
            raise RuntimeError("injected budget rollback")
    persisted = store.get_semantic_rate_budget(reservation.bucket_id)
    assert persisted is not None
    assert (persisted.minute_count, persisted.day_count, persisted.inflight_count) == (
        1,
        1,
        0,
    )
    store.close()

    reopened = SQLiteStore(database)
    reopened_control = SemanticRuntimeControl(
        reopened,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second_epoch,
        now=lambda: "2026-08-07T00:00:20+00:00",
    )
    reopened_control.admit()
    reopened_budget = HostSemanticRateBudget(
        reopened,
        control_resolver=reopened_control.authority_view,
        now=lambda: now,
    )

    def reserve_once() -> str:
        try:
            with reopened.transaction():
                selected = reopened_budget.reserve(
                    _binding(second_epoch),
                    rule_id="read-reports",
                    assessment=_assessment(),
                )
                reopened_budget.release(selected.bucket_id)
            return "issued"
        except SemanticRateBudgetExceeded:
            return "exhausted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _index: reserve_once(), range(2)))
    assert outcomes == ["exhausted", "issued"]
    final = reopened.get_semantic_rate_budget(reservation.bucket_id)
    assert final is not None
    assert (final.minute_count, final.day_count, final.inflight_count) == (2, 2, 0)
    reopened.close()


def test_cross_epoch_inflight_is_released_before_one_new_reservation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic-rate-budget-inflight.db"
    now = datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc)
    first_epoch = replace(
        _epoch(),
        epoch_id="budget-inflight-epoch-1",
        generation=1,
        expected_previous_sha256=None,
        per_rule_per_minute_limit=10,
        per_rule_per_day_limit=100,
        max_inflight=2,
    )
    store = SQLiteStore(database)
    first_control = SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first_epoch,
        now=lambda: "2026-08-07T00:00:00+00:00",
    )
    first_control.admit()
    first_budget = HostSemanticRateBudget(
        store,
        control_resolver=first_control.authority_view,
        now=lambda: now,
    )
    with store.transaction():
        old_reservations = tuple(
            first_budget.reserve(
                _binding(first_epoch),
                rule_id="read-reports",
                assessment=_assessment(),
            )
            for _index in range(2)
        )
    assert old_reservations[0].bucket_id == old_reservations[1].bucket_id
    store.close()

    second_epoch = replace(
        first_epoch,
        epoch_id="budget-inflight-epoch-2",
        generation=2,
        expected_previous_sha256=first_epoch.canonical_sha256(),
    )
    reopened = SQLiteStore(database)
    second_control = SemanticRuntimeControl(
        reopened,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second_epoch,
        now=lambda: "2026-08-07T00:00:10+00:00",
    )
    second_control.admit()
    second_budget = HostSemanticRateBudget(
        reopened,
        control_resolver=second_control.authority_view,
        now=lambda: now,
    )
    with reopened.transaction():
        with pytest.raises(SemanticRateBudgetExceeded, match="exhausted"):
            second_budget.reserve(
                _binding(second_epoch),
                rule_id="read-reports",
                assessment=_assessment(),
            )

    def release_old() -> str:
        with reopened.transaction():
            second_budget.release(old_reservations[0].bucket_id)
        return "released"

    def reserve_new() -> str:
        try:
            with reopened.transaction():
                second_budget.reserve(
                    _binding(second_epoch),
                    rule_id="read-reports",
                    assessment=_assessment(),
                )
            return "issued"
        except SemanticRateBudgetExceeded:
            return "exhausted"

    with ThreadPoolExecutor(max_workers=2) as executor:
        release_future = executor.submit(release_old)
        reserve_future = executor.submit(reserve_new)
        concurrent_outcomes = {release_future.result(), reserve_future.result()}
    assert "released" in concurrent_outcomes
    if "issued" not in concurrent_outcomes:
        assert reserve_new() == "issued"

    final = reopened.get_semantic_rate_budget(old_reservations[0].bucket_id)
    assert final is not None
    assert (final.minute_count, final.day_count, final.inflight_count) == (3, 3, 2)
    with reopened.transaction():
        with pytest.raises(SemanticRateBudgetExceeded, match="exhausted"):
            second_budget.reserve(
                _binding(second_epoch),
                rule_id="read-reports",
                assessment=_assessment(),
            )
    reopened.close()


def test_host_rate_budget_denies_scope_or_clock_drift_without_mutation() -> None:
    epoch = _epoch()
    control = _control(epoch)
    binding = _binding(epoch)
    repository = _BudgetRepository()
    now = [datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc)]
    budget = HostSemanticRateBudget(
        repository,
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        now=lambda: now[0],
    )
    with pytest.raises(CapabilityDenied, match="rule"):
        budget.reserve(
            binding,
            rule_id="other-rule",
            assessment=_assessment(),
        )
    assert repository.record is None

    repository.fence_allowed = False
    with pytest.raises(CapabilityDenied, match="control changed"):
        budget.reserve(
            binding,
            rule_id="read-reports",
            assessment=_assessment(),
        )
    assert repository.record is None
    repository.fence_allowed = True

    reservation = budget.reserve(
        binding,
        rule_id="read-reports",
        assessment=_assessment(),
    )
    budget.release(reservation.bucket_id)
    snapshot = repository.record
    now[0] -= timedelta(seconds=1)
    with pytest.raises(CapabilityDenied, match="clock moved backwards"):
        budget.reserve(
            binding,
            rule_id="read-reports",
            assessment=_assessment(),
        )
    assert repository.record == snapshot
    assert repository.record is not None
    assert repository.record.minute_count == 1


def test_host_rate_budget_enforces_narrowed_live_epoch_classifier_threshold() -> None:
    epoch = replace(_epoch(), minimum_confidence_bps=10_000)
    control = _control(epoch)
    binding = _binding(epoch)
    repository = _BudgetRepository()
    budget = HostSemanticRateBudget(
        repository,
        control_resolver=lambda: SemanticAuthorityControlView(control, epoch),
        now=lambda: datetime(2026, 8, 7, 0, 0, 20, tzinfo=timezone.utc),
    )

    with pytest.raises(CapabilityDenied, match="active epoch threshold"):
        budget.reserve(
            binding,
            rule_id="read-reports",
            assessment=_assessment(9_999),
        )
    assert repository.record is None


def test_semantic_lifecycle_notifications_are_payload_free_and_idempotent() -> None:
    events: list[dict[str, object]] = []

    class _Capabilities:
        def __init__(self) -> None:
            self.trips = 0

        @staticmethod
        def semantic_approval_lifecycle_evidence(
            _decision: CapabilityDecision,
        ) -> dict[str, object]:
            return {
                "capability_id": "cap-unit-lifecycle",
                "binding_sha256": _A,
                "policy_epoch_id": "epoch-unit-lifecycle",
                "policy_epoch_sha256": _B,
                "tenant_bucket_sha256": _C,
                "assessment_id": "assessment-unit-lifecycle",
                "settlement_id": "settlement-unit-lifecycle",
                "budget_bucket_id": "budget-unit-lifecycle",
                "matched_rule_id": "read-reports",
                "issued_at": "2026-08-07T00:00:00Z",
                "expires_at": "2026-08-07T00:01:00Z",
            }

        def trip_semantic_authority_locally(self) -> None:
            self.trips += 1

    capabilities = _Capabilities()
    sdk = SimpleNamespace(
        capabilities=capabilities,
        report_semantic_authority_lifecycle=(
            lambda event: events.append(dict(event)) or True
        ),
    )
    decision = CapabilityDecision(
        subject="pid-unit-lifecycle",
        resource="filesystem:workspace:reports/result.txt",
        right="read",
        allowed=True,
        effect=CapabilityEffect.ALLOW,
        reason="semantic unit authority",
        selected_capability_id="cap-unit-lifecycle",
        consume_capability_id="cap-unit-lifecycle",
    )

    def operation() -> object:
        selected = protected_operations_module.ProtectedOperation(
            sdk,
            SimpleNamespace(name="primitive.unit.lifecycle"),
            SimpleNamespace(
                pid="pid-unit-lifecycle",
                canonical_args={"secret": "SEMANTIC_LIFECYCLE_SECRET_SENTINEL"},
                target="/private/SEMANTIC_LIFECYCLE_SECRET_SENTINEL",
            ),
            object(),
        )
        selected.effect_id = "effect-unit-lifecycle"
        selected._authority_decisions = (decision,)
        return selected

    consumed = operation()
    consumed._report_semantic_authority_lifecycle(
        "consumed",
        phase="capability_commit",
    )
    consumed._report_semantic_authority_lifecycle(
        "consumed",
        phase="capability_commit",
    )
    consumed._report_semantic_authority_lifecycle(
        "succeeded",
        phase="completion",
    )
    unknown = operation()
    unknown._report_semantic_authority_lifecycle(
        "provider_outcome_unknown",
        phase="dispatch",
        error_type="RuntimeError",
    )

    assert len(events) == 3
    consumed_item = events[0]["authority"][0]
    succeeded_item = events[1]["authority"][0]
    unknown_item = events[2]["authority"][0]
    assert consumed_item["outcome_id"] != succeeded_item["outcome_id"]
    assert succeeded_item["outcome_id"] == unknown_item["outcome_id"]
    assert capabilities.trips == 1
    encoded = json.dumps(events, ensure_ascii=False, sort_keys=True)
    assert "SEMANTIC_LIFECYCLE_SECRET_SENTINEL" not in encoded
    for forbidden in (
        '"argv"',
        '"command"',
        '"content"',
        '"path"',
        '"prompt"',
        '"response"',
        '"secret"',
        '"text"',
    ):
        assert forbidden not in encoded


def test_repeated_dispatch_revalidation_preserves_authority_cardinality() -> None:
    decision = CapabilityDecision(
        subject="pid-unit-cardinality",
        resource="filesystem:workspace:reports/result.txt",
        right="read",
        allowed=True,
        effect=CapabilityEffect.ALLOW,
        reason="semantic unit authority",
        selected_capability_id="cap-unit-cardinality",
        consume_capability_id="cap-unit-cardinality",
    )

    class _Capabilities:
        @staticmethod
        def is_semantic_approval_decision(
            selected: CapabilityDecision,
        ) -> bool:
            return selected.selected_capability_id == "cap-unit-cardinality"

    selected = protected_operations_module.ProtectedOperation(
        SimpleNamespace(capabilities=_Capabilities()),
        SimpleNamespace(
            name="primitive.unit.cardinality",
            authority_mode=protected_operations_module.AuthorityMode.CAPABILITY,
        ),
        SimpleNamespace(
            pid="pid-unit-cardinality",
            decisions=(decision,),
        ),
        object(),
    )
    selected._authority_decisions = (decision,)
    selected._reservation_ids_by_capability = {
        "cap-unit-cardinality": "reservation-unit-cardinality"
    }
    selected._reservations_committed = True

    for _attempt in range(16):
        selected._revalidate_dispatch_authority()
        assert selected._authority_decisions == (decision,)
