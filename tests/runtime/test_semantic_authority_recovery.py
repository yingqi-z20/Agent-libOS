from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.models import Capability, CapabilityEffect, CapabilityStatus
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.models.semantic import (
    SemanticApprovalBindingV2,
    SemanticRuntimeMode,
)
from agent_libos.semantic.enforcement import HostSemanticRateBudget
from agent_libos.semantic.recovery import SemanticMachineAuthorityRecovery
from agent_libos.storage import SemanticFlowPage


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
# Keep parametrized values byte-for-byte stable across xdist collectors.
# Recovery expiry is injected by ``_recovery``; these timestamps only bind the
# synthetic Capability/BindingV2 shape and do not need wall-clock time.
_NOW = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class _Repository:
    def __init__(self, settlements: list[Any]) -> None:
        self.settlements = settlements
        self.outcomes: list[Any] = []
        self.rollback_lists: list[list[Any]] = []
        self.fail_outcome_append = False
        self.unresolved_query_calls = 0

    def _terminal_ids(self) -> set[str]:
        return {
            item.settlement_id
            for item in self.outcomes
            if item.outcome
            in {"succeeded", "failed", "outcome_unknown", "expired", "revoked", "race_lost"}
        }

    def query_unresolved_semantic_machine_settlements(
        self,
        *,
        after: str | None,
        limit: int,
        **_filters: Any,
    ) -> SemanticFlowPage:
        self.unresolved_query_calls += 1
        start = 0
        if after is not None:
            start = next(
                index + 1
                for index, item in enumerate(self.settlements)
                if item.settlement_id == after
            )
        unresolved = [
            item
            for item in self.settlements[start:]
            if item.settlement_id not in self._terminal_ids()
        ]
        records = tuple(unresolved[:limit])
        remaining = unresolved[len(records) :]
        return SemanticFlowPage(
            records=records,
            next_cursor=(
                records[-1].settlement_id if records and remaining else None
            ),  # type: ignore[arg-type]
        )

    def query_semantic_machine_outcomes(
        self,
        *,
        settlement_id: str,
        **_filters: Any,
    ) -> SemanticFlowPage:
        return SemanticFlowPage(
            records=tuple(
                item for item in self.outcomes if item.settlement_id == settlement_id
            ),
            next_cursor=None,
        )

    @contextmanager
    def transaction(self):
        before = list(self.outcomes)
        external = [list(items) for items in self.rollback_lists]
        try:
            yield
        except BaseException:
            self.outcomes = before
            for items, snapshot in zip(self.rollback_lists, external, strict=True):
                items[:] = snapshot
            raise

    def append_semantic_machine_outcome_if_absent(self, record: Any) -> bool:
        if self.fail_outcome_append:
            raise RuntimeError("injected recovery outcome append failure")
        if any(item.outcome_id == record.outcome_id for item in self.outcomes):
            return False
        self.outcomes.append(record)
        return True


class _Budget:
    bucket_id_for = staticmethod(HostSemanticRateBudget.bucket_id_for)

    def __init__(self) -> None:
        self.released: list[str] = []
        self.fail_release = False

    def release(self, bucket_id: str) -> None:
        self.released.append(bucket_id)
        if self.fail_release:
            raise RuntimeError("injected recovery budget failure")


class _Control:
    def __init__(self, mode: SemanticRuntimeMode = SemanticRuntimeMode.CANARY_AUTO) -> None:
        self.state = SimpleNamespace(
            mode=mode,
            tripped=False,
            active_epoch_id="epoch-recovery",
            active_policy_sha256=_A,
        )
        self.trips: list[Any] = []

    def current(self) -> Any:
        return self.state

    def trip(self, code: Any, **facts: Any) -> None:
        self.trips.append((code, facts))
        self.state.tripped = True


def _binding(*, effect_id: str = "effect-recovery") -> SemanticApprovalBindingV2:
    return SemanticApprovalBindingV2(
        request_id="request-recovery",
        request_revision=0,
        pid="pid-recovery",
        operation_id="operation-recovery",
        effect_id=effect_id,
        authority_operation="filesystem.read",
        resource="filesystem:workspace:report.txt",
        right="read",
        canonical_args_hash=_B,
        target_state_version="state-v1",
        manifest_id="manifest-recovery",
        manifest_sha256=_B,
        ceiling_sha256=_C,
        policy_epoch_id="epoch-recovery",
        policy_epoch_sha256=_A,
        control_generation=1,
        assessment_id="assessment-recovery",
        assessment_sha256=_B,
        classifier_profile_sha256=_B,
        classifier_model_sha256=_C,
        tenant_bucket_sha256=_C,
        source_labels_sha256=_B,
        source_refs_sha256=_C,
        flow_snapshot_sha256=_B,
        sink_identity_sha256=None,
        tool_schema_sha256=None,
        provider_spec_sha256=None,
        nonce="nonce-recovery",
        issued_at=(_NOW - timedelta(seconds=5)).isoformat(),
        expires_at=(_NOW + timedelta(seconds=60)).isoformat(),
    )


def _settlement(index: int = 0) -> Any:
    binding = _binding(effect_id=f"effect-recovery-{index}")
    return SimpleNamespace(
        settlement_id=f"settlement-recovery-{index}",
        outcome="issued",
        request_id=binding.request_id,
        request_revision=binding.request_revision,
        assessment_id=binding.assessment_id,
        pid=binding.pid,
        action_id=binding.authority_operation,
        capability_id=f"cap-recovery-{index}",
        effect_id=binding.effect_id,
        epoch_id=binding.policy_epoch_id,
        policy_sha256=binding.policy_epoch_sha256,
        tenant_bucket_sha256=binding.tenant_bucket_sha256,
        binding_sha256=binding.canonical_sha256(),
        matched_rule_id="read-report",
        created_at=(_NOW - timedelta(seconds=5)).isoformat(),
        binding=binding,
    )


def _capability(settlement: Any, *, uses: int = 1) -> Capability:
    binding = settlement.binding
    return Capability(
        cap_id=settlement.capability_id,
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
        uses_remaining=uses,
        metadata={
            "semantic_auto_approval": {
                "schema_version": 1,
                "binding_sha256": binding.canonical_sha256(),
                "request_id": binding.request_id,
                "assessment_id": binding.assessment_id,
                "policy_epoch_id": binding.policy_epoch_id,
                "matched_rule_id": settlement.matched_rule_id,
                "settlement_id": settlement.settlement_id,
                "budget_bucket_id": HostSemanticRateBudget.bucket_id_for(
                    epoch_id=binding.policy_epoch_id,
                    tenant_bucket_sha256=binding.tenant_bucket_sha256,
                    rule_id=settlement.matched_rule_id,
                ),
            }
        },
    )


def _recovery(
    repository: _Repository,
    capabilities: dict[str, Capability],
    effects: dict[str, Any],
    *,
    control: _Control | None = None,
    expired_capability_ids: set[str] | None = None,
) -> tuple[SemanticMachineAuthorityRecovery, _Budget, _Control, list[str], list[bool]]:
    budget = _Budget()
    selected_control = control or _Control()
    revoked: list[str] = []
    local_trips: list[bool] = []
    repository.rollback_lists.extend((budget.released, revoked))
    return (
        SemanticMachineAuthorityRecovery(
            repository,
            capability_reader=capabilities.get,
            capability_revoker=lambda cap_id: revoked.append(cap_id),
            capability_expired=lambda cap: cap.cap_id
            in (expired_capability_ids or set()),
            effect_reader=effects.get,
            rate_budget=budget,
            control=selected_control,
            local_trip=lambda: local_trips.append(True),
        ),
        budget,
        selected_control,
        revoked,
        local_trips,
    )


def test_recovery_retains_only_exact_not_started_authority() -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    repository = _Repository([settlement])
    recovery, budget, _control, revoked, _trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {},
    )

    first = recovery.recover()
    second = recovery.recover()

    assert first.retained_not_started == second.retained_not_started == 1
    assert repository.outcomes == []
    assert budget.released == []
    assert revoked == []


@pytest.mark.parametrize(
    "effect",
    (
        SimpleNamespace(
            effect_state="prepared",
            transaction_state="prepared",
        ),
        SimpleNamespace(
            effect_state="finalized",
            transaction_state="failed",
            provider_receipt={
                "dispatch_status": "not_started",
                "certified": True,
            },
            provider_metadata={},
        ),
    ),
    ids=("prepared", "reconciled-not-started"),
)
def test_recovery_retains_certified_not_started_effect(effect: Any) -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    repository = _Repository([settlement])
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {settlement.effect_id: effect},
    )

    assert recovery.recover().retained_not_started == 1
    assert repository.outcomes == []
    assert budget.released == []
    assert control.trips == []
    assert revoked == []
    assert local_trips == []


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        (SimpleNamespace(effect_state="finalized", transaction_state="committed"), "succeeded"),
        (
            SimpleNamespace(
                effect_state="finalized",
                transaction_state="failed",
                rollback_status="rolled_back",
            ),
            "failed",
        ),
    ],
)
def test_recovery_terminalizes_known_consumed_outcome_once(
    effect: Any,
    expected: str,
) -> None:
    settlement = _settlement()
    capability = _capability(settlement, uses=0)
    repository = _Repository([settlement])
    recovery, budget, _control, _revoked, _trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {settlement.effect_id: effect},
    )

    summary = recovery.recover()
    assert getattr(summary, expected) == 1
    assert [item.outcome for item in repository.outcomes] == ["consumed", expected]
    assert len(budget.released) == 1
    assert recovery.recover().scanned == 0
    assert len(budget.released) == 1
    assert repository.outcomes[-1].created_at >= settlement.created_at


def test_contradictory_not_started_flag_trips_and_exhausts_authority() -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    effect = SimpleNamespace(
        effect_state="finalized",
        transaction_state="committed",
        provider_receipt={"certified_not_started": True},
        provider_metadata={"certified_not_started": True},
    )
    repository = _Repository([settlement])
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {settlement.effect_id: effect},
    )

    assert recovery.recover().outcome_unknown == 1
    assert [item.outcome for item in repository.outcomes] == ["outcome_unknown"]
    assert revoked == [capability.cap_id]
    assert len(budget.released) == 1
    assert local_trips == [True]
    assert len(control.trips) == 1


def test_recovery_revokes_old_authority_when_control_is_off() -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    repository = _Repository([settlement])
    recovery, budget, _control, revoked, _trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {},
        control=_Control(SemanticRuntimeMode.OFF),
    )

    assert recovery.recover().revoked == 1
    assert revoked == [capability.cap_id]
    assert [item.outcome for item in repository.outcomes] == ["revoked"]
    assert len(budget.released) == 1


@pytest.mark.parametrize("reason", ("expired", "rotated", "tripped"))
def test_recovery_expires_or_revokes_authority_after_control_change(
    reason: str,
) -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    repository = _Repository([settlement])
    control = _Control()
    expired = set()
    expected = "revoked"
    if reason == "expired":
        expired.add(capability.cap_id)
        expected = "expired"
    elif reason == "rotated":
        control.state.active_epoch_id = "epoch-rotated"
    else:
        control.state.tripped = True
    recovery, budget, _control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {},
        control=control,
        expired_capability_ids=expired,
    )

    assert getattr(recovery.recover(), expected) == 1
    assert [item.outcome for item in repository.outcomes] == [expected]
    assert revoked == [capability.cap_id]
    assert len(budget.released) == 1
    assert local_trips == []


def test_recovery_capability_metadata_mismatch_is_unknown_and_trips_once() -> None:
    settlements = [_settlement(index) for index in range(2)]
    capabilities = {}
    for settlement in settlements:
        capability = _capability(settlement)
        capabilities[capability.cap_id] = replace(
            capability,
            metadata={
                "semantic_auto_approval": {
                    **capability.metadata["semantic_auto_approval"],
                    "assessment_id": "assessment-forged",
                }
            },
        )
    repository = _Repository(settlements)
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        capabilities,
        {},
    )

    summary = recovery.recover()
    assert summary.outcome_unknown == 2
    assert len(control.trips) == 1
    assert len(local_trips) == 2
    assert sorted(revoked) == sorted(capabilities)
    assert len(budget.released) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("subject", "pid-forged"),
        ("resource", "filesystem:workspace:forged.txt"),
        ("rights", {"write"}),
        ("effect", CapabilityEffect.ASK),
        ("issued_by", "policy:semantic:forged"),
        ("delegable", True),
        ("revocable", False),
        ("expires_at", (_NOW + timedelta(seconds=120)).isoformat()),
    ),
)
def test_recovery_rejects_forged_capability_shape(
    field: str,
    value: Any,
) -> None:
    settlement = _settlement()
    capability = replace(_capability(settlement), **{field: value})
    repository = _Repository([settlement])
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {},
    )

    assert recovery.recover().outcome_unknown == 1
    assert [item.outcome for item in repository.outcomes] == ["outcome_unknown"]
    assert revoked == [capability.cap_id]
    assert len(budget.released) == 1
    assert len(control.trips) == 1
    assert local_trips == [True]


def test_recovery_consumed_partial_lifecycle_completes_without_duplicate() -> None:
    settlement = _settlement()
    capability = _capability(settlement, uses=0)
    repository = _Repository([settlement])
    recovery, budget, _control, _revoked, _trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {
            settlement.effect_id: SimpleNamespace(
                effect_state="finalized",
                transaction_state="committed",
            )
        },
    )
    recovery._append_outcome(  # noqa: SLF001 - model crash after consumed append
        settlement,
        "consumed",
        slot="consumed",
    )

    assert recovery.recover().succeeded == 1
    assert [item.outcome for item in repository.outcomes] == [
        "consumed",
        "succeeded",
    ]
    assert len(budget.released) == 1


@pytest.mark.parametrize("failure", ("outcome", "budget"))
def test_recovery_terminal_transaction_rolls_back_on_failure(failure: str) -> None:
    settlement = _settlement()
    capability = _capability(settlement, uses=0)
    if failure == "outcome":
        # A structural mismatch trips before the terminal transaction.  The
        # durable trip remains even when revoke/outcome/release rolls back.
        capability = replace(capability, issued_by="policy:semantic:forged")
    repository = _Repository([settlement])
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {
            settlement.effect_id: SimpleNamespace(
                effect_state="finalized",
                transaction_state="committed",
            )
        },
    )
    repository.fail_outcome_append = failure == "outcome"
    budget.fail_release = failure == "budget"

    with pytest.raises(RuntimeError, match="injected recovery"):
        recovery.recover()

    assert repository.outcomes == []
    assert budget.released == []
    assert revoked == []
    if failure == "outcome":
        assert len(control.trips) == 1
        assert local_trips == [True]
    else:
        assert control.trips == []
        assert local_trips == []


def test_recovery_retry_preserves_unknown_after_durable_trip() -> None:
    settlement = _settlement()
    capability = _capability(settlement)
    repository = _Repository([settlement])
    recovery, budget, control, revoked, local_trips = _recovery(
        repository,
        {capability.cap_id: capability},
        {
            settlement.effect_id: SimpleNamespace(
                effect_state="finalized",
                transaction_state="committed",
                provider_receipt={},
                provider_metadata={},
            )
        },
    )
    repository.fail_outcome_append = True
    with pytest.raises(RuntimeError, match="outcome append"):
        recovery.recover()
    assert len(control.trips) == 1
    assert repository.outcomes == []
    assert revoked == []
    assert budget.released == []

    repository.fail_outcome_append = False
    assert recovery.recover().outcome_unknown == 1
    assert [item.outcome for item in repository.outcomes] == ["outcome_unknown"]
    assert len(control.trips) == 1
    assert local_trips == [True, True]
    assert revoked == [capability.cap_id]
    assert len(budget.released) == 1


def test_recovery_bound_counts_only_unresolved_issuance() -> None:
    settlements = [
        SimpleNamespace(settlement_id=f"terminal-history-{index:05d}")
        for index in range(10_001)
    ]
    repository = _Repository(settlements)
    for settlement in settlements:
        repository.outcomes.append(
            SimpleNamespace(
                outcome_id=f"terminal-{settlement.settlement_id}",
                settlement_id=settlement.settlement_id,
                outcome="succeeded",
            )
        )
    recovery, _budget, _control, _revoked, _trips = _recovery(
        repository,
        {},
        {},
    )
    assert recovery.recover(max_records=1).scanned == 0

    unresolved_settlements = [_settlement(index) for index in range(3)]
    unresolved = _Repository(unresolved_settlements)
    capabilities = {
        item.capability_id: _capability(item) for item in unresolved_settlements
    }
    recovery, _budget, control, _revoked, local_trips = _recovery(
        unresolved,
        capabilities,
        {},
    )
    with pytest.raises(CapabilityDenied, match="exceeded its bound"):
        recovery.recover(page_size=2, max_records=2)
    assert local_trips == [True]
    assert len(control.trips) == 1

    exact_settlements = [_settlement(index + 10) for index in range(4)]
    exact_repository = _Repository(exact_settlements)
    exact_capabilities = {
        item.capability_id: _capability(item) for item in exact_settlements
    }
    exact_recovery, _budget, exact_control, _revoked, exact_trips = _recovery(
        exact_repository,
        exact_capabilities,
        {},
    )
    exact_summary = exact_recovery.recover(page_size=2, max_records=4)
    assert exact_summary.scanned == 4
    assert exact_summary.retained_not_started == 4
    assert exact_repository.unresolved_query_calls == 2
    assert exact_control.trips == []
    assert exact_trips == []
