from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import (
    SemanticApprovalRule,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
)
from agent_libos.semantic.control import (
    SemanticControlConflict,
    SemanticRuntimeControl,
)
from agent_libos.semantic.service import SemanticManager
from agent_libos.storage import (
    SQLiteStore,
    SemanticAssessmentRepository,
    SemanticHealthEventRecord,
    SemanticMachineSettlementRecord,
    SemanticReviewLabelRecord,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_NOW = "2026-08-07T00:00:00+00:00"
_ACTIVATED = datetime(2026, 7, 1, tzinfo=timezone.utc)
_ROLLOUT_READY = _ACTIVATED + timedelta(days=7, seconds=1)


def _epoch() -> SemanticPolicyEpochV1:
    return SemanticPolicyEpochV1(
        epoch_id="review-epoch-1",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="deny-reviewed-read",
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/*",
                rights=("read",),
            ),
        ),
        created_at=_NOW,
    )


def _canary_epoch(
    generation: int,
    *,
    tenants: tuple[str, ...] = (_A,),
    resource: str = "filesystem:workspace:reports/*",
    previous: str | None = None,
) -> SemanticPolicyEpochV1:
    return SemanticPolicyEpochV1(
        epoch_id=f"canary-review-epoch-{generation}",
        generation=generation,
        expected_previous_sha256=previous,
        tenant_bucket_sha256s=tenants,
        auto_approval_rules=(
            SemanticApprovalRule(
                rule_id="read-reviewed-reports",
                authority_operation="filesystem.read",
                resource=resource,
                rights=("read",),
            ),
        ),
        hard_deny_rules=(),
        created_at=_NOW,
    )


def _manager(
    store: SQLiteStore | SemanticAssessmentRepository,
    control: SemanticRuntimeControl | None,
) -> SemanticManager:
    manager = SemanticManager.__new__(SemanticManager)
    manager._repository = store
    manager._control = control
    return manager


def _append_settlement(
    store: SQLiteStore,
    *,
    settlement_id: str,
    epoch_id: str,
    policy_sha256: str,
) -> None:
    store.append_semantic_machine_settlement(
        SemanticMachineSettlementRecord(
            settlement_id=settlement_id,
            assessment_id=None,
            job_id=None,
            request_id=f"request:{settlement_id}",
            request_revision=0,
            pid="pid-review",
            operation_id="operation-review",
            effect_id=f"effect:{settlement_id}",
            epoch_id=epoch_id,
            policy_sha256=policy_sha256,
            tenant_bucket_sha256=_A,
            action_id="filesystem.read",
            outcome="denied",
            capability_id=None,
            binding_sha256=_B,
            decision_sha256=_C,
            matched_rule_id=None,
            reason_codes=("policy_hard_deny",),
            created_at=_NOW,
        )
    )


def _append_issued_settlement(
    store: SQLiteStore,
    *,
    settlement_id: str,
    epoch: SemanticPolicyEpochV1,
    created_at: str,
) -> None:
    store.append_semantic_machine_settlement(
        SemanticMachineSettlementRecord(
            settlement_id=settlement_id,
            assessment_id=f"assessment:{settlement_id}",
            job_id=f"job:{settlement_id}",
            request_id=f"request:{settlement_id}",
            request_revision=0,
            pid=f"pid:{settlement_id}",
            operation_id=f"operation:{settlement_id}",
            effect_id=f"effect:{settlement_id}",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
            tenant_bucket_sha256=_A,
            action_id="filesystem.read",
            outcome="issued",
            capability_id=f"capability:{settlement_id}",
            binding_sha256=_B,
            decision_sha256=_C,
            matched_rule_id="read-reviewed-reports",
            reason_codes=("policy_match",),
            created_at=created_at,
        )
    )


def _import_unsafe(manager: SemanticManager, settlement_id: str) -> dict[str, object]:
    return manager.append_review_label(
        settlement_id=settlement_id,
        outcome="unsafe",
        reviewer_id="operator-reviewer",
        evidence_sha256=_D,
        reviewed_at=_NOW,
    )


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_unsafe_review_is_retained_while_semantic_authority_is_inactive(
    mode: str,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        control = SemanticRuntimeControl(store, mode=mode, policy_epoch=None)
        before = control.admit()
        _append_settlement(
            store,
            settlement_id=f"settlement-{mode}",
            epoch_id="inactive-epoch",
            policy_sha256=_A,
        )

        result = _import_unsafe(_manager(store, control), f"settlement-{mode}")

        assert result["outcome"] == "unsafe"
        labels = store.query_semantic_review_labels(limit=10).records
        assert len(labels) == 1
        assert labels[0].outcome == "unsafe"
        assert control.current() == before
    finally:
        store.close()


def test_active_unsafe_review_atomically_trips_the_current_epoch() -> None:
    store = SQLiteStore(":memory:")
    try:
        epoch = _epoch()
        control = SemanticRuntimeControl(
            store,
            mode="enforce_deny",
            policy_epoch=epoch,
        )
        control.admit()
        _append_settlement(
            store,
            settlement_id="settlement-active",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
        )

        _import_unsafe(_manager(store, control), "settlement-active")

        current = control.current()
        assert current.tripped is True
        assert current.trip_code is not None
        assert current.trip_code.value == "unsafe_review"
        assert len(store.query_semantic_review_labels(limit=10).records) == 1
    finally:
        store.close()


def test_active_unsafe_review_control_race_retains_label_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        epoch = _epoch()
        control = SemanticRuntimeControl(
            store,
            mode="enforce_deny",
            policy_epoch=epoch,
        )
        control.admit()
        _append_settlement(
            store,
            settlement_id="settlement-race",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
        )
        monkeypatch.setattr(
            store,
            "fence_semantic_control_state",
            lambda _expected: False,
        )

        manager = _manager(store, control)
        with pytest.raises(
            ValidationError,
            match="evidence was retained after durable control race",
        ):
            _import_unsafe(manager, "settlement-race")

        labels = store.query_semantic_review_labels(
            limit=10,
            settlement_id="settlement-race",
            outcome="unsafe",
        ).records
        assert len(labels) == 1
        assert manager._unsafe_review_latched is True
        assert control.current().tripped is True
        health = store.query_semantic_health_events(
            limit=10,
            event_kind="semantic_unsafe_review_fallback_trip",
        ).records
        assert len(health) == 1
    finally:
        store.close()


def test_active_unsafe_review_transient_control_race_retries_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        epoch = _epoch()
        control = SemanticRuntimeControl(
            store,
            mode="enforce_deny",
            policy_epoch=epoch,
        )
        control.admit()
        _append_settlement(
            store,
            settlement_id="settlement-transient-race",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
        )
        original_fence = store.fence_semantic_control_state
        attempts = 0

        def transient_fence(expected: object) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return False
            return original_fence(expected)

        monkeypatch.setattr(
            store,
            "fence_semantic_control_state",
            transient_fence,
        )
        manager = _manager(store, control)

        result = _import_unsafe(manager, "settlement-transient-race")

        assert result["outcome"] == "unsafe"
        assert attempts == 3
        labels = store.query_semantic_review_labels(
            limit=10,
            settlement_id="settlement-transient-race",
            outcome="unsafe",
        ).records
        assert len(labels) == 1
        assert getattr(manager, "_unsafe_review_latched", False) is False
        assert control.current().tripped is True
    finally:
        store.close()


def test_active_unsafe_review_append_failure_rolls_back_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        epoch = _epoch()
        control = SemanticRuntimeControl(
            store,
            mode="enforce_deny",
            policy_epoch=epoch,
        )
        control.admit()
        _append_settlement(
            store,
            settlement_id="settlement-append-failure",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
        )

        def fail_append(_record: object) -> None:
            raise ValidationError("injected review append failure")

        monkeypatch.setattr(store, "append_semantic_review_label", fail_append)
        with pytest.raises(ValidationError, match="injected review append failure"):
            _import_unsafe(
                _manager(store, control),
                "settlement-append-failure",
            )

        assert store.query_semantic_review_labels(limit=10).records == ()
        assert control.current().tripped is False
    finally:
        store.close()


def test_off_imported_old_epoch_unsafe_evidence_blocks_rollout_expansion() -> None:
    """An inactive import remains global evidence against the old canary.

    The 1,001st issuance is intentional: the first 1,000 retain complete safe
    reviews, proving that cohort expansion is rejected specifically by the
    global ``unsafe_count == 0`` predicate rather than by review completeness.
    """

    store = SQLiteStore(":memory:")
    try:
        repository = SemanticAssessmentRepository(store)
        first = _canary_epoch(1)
        SemanticRuntimeControl(
            repository,
            mode="canary_auto",
            policy_epoch=first,
            now=lambda: _ACTIVATED.isoformat(),
        ).admit()
        with store.transaction():
            for index in range(1_001):
                settlement_id = f"rollout-settlement-{index:04d}"
                created_at = (
                    _ACTIVATED + timedelta(seconds=index + 1)
                ).isoformat()
                _append_issued_settlement(
                    store,
                    settlement_id=settlement_id,
                    epoch=first,
                    created_at=created_at,
                )
                if index < 1_000:
                    store.append_semantic_review_label(
                        SemanticReviewLabelRecord(
                            review_id=f"safe-review-{index:04d}",
                            settlement_id=settlement_id,
                            outcome="safe",
                            reviewer_sha256=_C,
                            evidence_sha256=_D,
                            created_at=created_at,
                        )
                    )

        inactive = SemanticRuntimeControl(
            repository,
            mode="off",
            policy_epoch=None,
            now=lambda: (_ACTIVATED + timedelta(days=1)).isoformat(),
        )
        inactive.admit()
        off_state = inactive.current()
        off_record = store.get_semantic_control_state()
        assert off_record is not None
        imported = _import_unsafe(
            _manager(repository, inactive),
            "rollout-settlement-1000",
        )

        assert imported["outcome"] == "unsafe"
        unsafe_labels = store.query_semantic_review_labels(
            limit=10,
            settlement_id="rollout-settlement-1000",
            outcome="unsafe",
        ).records
        assert len(unsafe_labels) == 1
        evidence = store.semantic_rollout_review_evidence(
            first.epoch_id,
            "filesystem.read",
            limit=1_000,
        )
        assert datetime.fromisoformat(evidence["activated_at"]) == _ACTIVATED
        assert {
            key: value for key, value in evidence.items() if key != "activated_at"
        } == {
            "schema_version": 1,
            "issued_count": 1_001,
            "required_count": 1_000,
            "completely_safe_count": 1_000,
            "unsafe_count": 1,
        }
        assert inactive.current() == off_state

        expanded = _canary_epoch(
            2,
            tenants=(_A, _B),
            previous=first.canonical_sha256(),
        )
        with pytest.raises(
            SemanticControlConflict,
            match="unsafe review evidence",
        ):
            SemanticRuntimeControl(
                repository,
                mode="canary_auto",
                policy_epoch=expanded,
                now=lambda: _ROLLOUT_READY.isoformat(),
            ).admit()

        assert store.get_semantic_control_state() == off_record
        assert store.get_semantic_policy_epoch(expanded.epoch_id) is None
        assert (
            store.query_semantic_review_labels(
                limit=10,
                settlement_id="rollout-settlement-1000",
                outcome="unsafe",
            ).records
            == unsafe_labels
        )
    finally:
        store.close()


def test_old_epoch_unsafe_import_trips_new_active_epoch_and_retains_evidence() -> None:
    store = SQLiteStore(":memory:")
    try:
        repository = SemanticAssessmentRepository(store)
        first = _canary_epoch(1)
        SemanticRuntimeControl(
            repository,
            mode="canary_auto",
            policy_epoch=first,
            now=lambda: _ACTIVATED.isoformat(),
        ).admit()
        _append_issued_settlement(
            store,
            settlement_id="old-epoch-settlement",
            epoch=first,
            created_at=(_ACTIVATED + timedelta(seconds=1)).isoformat(),
        )

        # A strict resource narrowing is not a rollout expansion and therefore
        # permits generation 2 without manufacturing review evidence.
        second = _canary_epoch(
            2,
            resource="filesystem:workspace:reports/summary.txt",
            previous=first.canonical_sha256(),
        )
        active = SemanticRuntimeControl(
            repository,
            mode="canary_auto",
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
        )
        admitted = active.admit()
        assert admitted.active_epoch_id == second.epoch_id
        assert admitted.tripped is False

        imported = _import_unsafe(
            _manager(repository, active),
            "old-epoch-settlement",
        )

        current = active.current()
        assert current.generation == second.generation
        assert current.active_epoch_id == second.epoch_id
        assert current.tripped is True
        assert current.trip_code is not None
        assert current.trip_code.value == "unsafe_review"
        labels = store.query_semantic_review_labels(
            limit=10,
            settlement_id="old-epoch-settlement",
            outcome="unsafe",
        ).records
        assert len(labels) == 1
        assert labels[0].review_id == imported["review_id"]
        old_settlement = store.get_semantic_machine_settlement(
            "old-epoch-settlement"
        )
        assert old_settlement is not None
        assert old_settlement.epoch_id == first.epoch_id
        old_evidence = store.semantic_rollout_review_evidence(
            first.epoch_id,
            "filesystem.read",
            limit=1_000,
        )
        assert old_evidence["unsafe_count"] == 1
    finally:
        store.close()


def test_startup_admission_trips_active_epoch_with_direct_unsafe_label() -> None:
    """Reopen repairs a bypassed label/control gap through one control CAS."""

    store = SQLiteStore(":memory:")
    try:
        repository = SemanticAssessmentRepository(store)
        epoch = _epoch()
        original = SemanticRuntimeControl(
            repository,
            mode="enforce_deny",
            policy_epoch=epoch,
        ).admit()
        _append_settlement(
            store,
            settlement_id="startup-direct-unsafe",
            epoch_id=epoch.epoch_id,
            policy_sha256=epoch.canonical_sha256(),
        )
        label = SemanticReviewLabelRecord(
            review_id="review-startup-direct-unsafe",
            settlement_id="startup-direct-unsafe",
            outcome="unsafe",
            reviewer_sha256=_C,
            evidence_sha256=_D,
            created_at=_NOW,
        )
        store.append_semantic_review_label(label)
        policy_before = store.get_semantic_policy_epoch(epoch.epoch_id)

        reopened = SemanticRuntimeControl(
            repository,
            mode="enforce_deny",
            policy_epoch=epoch,
        ).admit()

        assert reopened.revision == original.revision + 1
        assert reopened.generation == original.generation
        assert reopened.active_epoch_id == original.active_epoch_id
        assert reopened.active_policy_sha256 == original.active_policy_sha256
        assert reopened.tripped is True
        assert reopened.trip_code is not None
        assert reopened.trip_code.value == "unsafe_review"
        assert store.get_semantic_policy_epoch(epoch.epoch_id) == policy_before
        assert store.query_semantic_review_labels(
            limit=10,
            settlement_id=label.settlement_id,
            outcome="unsafe",
        ).records == (label,)
        history = store.query_semantic_control_history(limit=10).records
        assert history[-1].revision == reopened.revision
        assert history[-1].active_epoch_id == reopened.active_epoch_id
        assert history[-1].tripped is True
    finally:
        store.close()


def test_startup_admission_trips_on_unsettled_old_epoch_review_health() -> None:
    """A durable fallback latch survives restart without rewriting evidence."""

    store = SQLiteStore(":memory:")
    try:
        repository = SemanticAssessmentRepository(store)
        epoch = _epoch()
        original = SemanticRuntimeControl(
            repository,
            mode="canary_auto",
            policy_epoch=epoch,
        ).admit()
        _append_settlement(
            store,
            settlement_id="startup-old-epoch-unsafe",
            epoch_id="older-review-epoch",
            policy_sha256=_A,
        )
        label = SemanticReviewLabelRecord(
            review_id="review-startup-old-epoch-unsafe",
            settlement_id="startup-old-epoch-unsafe",
            outcome="unsafe",
            reviewer_sha256=_C,
            evidence_sha256=_D,
            created_at=_NOW,
        )
        health = SemanticHealthEventRecord(
            event_id="health-startup-old-epoch-unsafe",
            event_kind="semantic_unsafe_review_control_unsettled",
            severity="critical",
            epoch_id=epoch.epoch_id,
            tenant_bucket_sha256=_A,
            evidence_sha256=_B,
            created_at=_NOW,
        )
        store.append_semantic_review_label(label)
        store.append_semantic_health_event(health)

        reopened = SemanticRuntimeControl(
            repository,
            mode="canary_auto",
            policy_epoch=epoch,
        ).admit()

        assert reopened.revision == original.revision + 1
        assert reopened.generation == original.generation
        assert reopened.active_epoch_id == original.active_epoch_id
        assert reopened.active_policy_sha256 == original.active_policy_sha256
        assert reopened.tripped is True
        assert reopened.trip_code is not None
        assert reopened.trip_code.value == "unsafe_review"
        assert store.query_semantic_review_labels(
            limit=10,
            settlement_id=label.settlement_id,
            outcome="unsafe",
        ).records == (label,)
        assert store.query_semantic_health_events(
            limit=10,
            epoch_id=epoch.epoch_id,
            event_kind=health.event_kind,
        ).records == (health,)
        history = store.query_semantic_control_history(limit=10).records
        assert history[-1].revision == reopened.revision
        assert history[-1].active_epoch_id == reopened.active_epoch_id
        assert history[-1].tripped is True
    finally:
        store.close()
