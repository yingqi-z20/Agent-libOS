from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from agent_libos.models.semantic import (
    SemanticApprovalRule,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
    SemanticRuntimeMode,
)
from agent_libos.semantic.control import (
    SemanticControlConflict,
    SemanticRuntimeControl,
)
from agent_libos.storage.semantic_v6 import (
    SemanticControlStateRecord,
    SemanticMachineSettlementRecord,
    SemanticPolicyEpochRecord,
    SemanticReviewLabelRecord,
    control_state_storage_record,
    policy_epoch_storage_record,
)
from agent_libos.storage import SQLiteStore


_TENANT_A = "a" * 64
_TENANT_B = "b" * 64
_ACTIVATED = datetime(2026, 7, 1, tzinfo=timezone.utc)
_READY = _ACTIVATED + timedelta(days=7, seconds=1)
_PROFILE_ID = "host-semantic-classifier"
_PROFILE_SHA256 = "1" * 64
_MODEL_SHA256 = "2" * 64


def _strict_allow_parameters() -> dict[str, Any]:
    return {
        "classifier_profile_id": _PROFILE_ID,
        "classifier_profile_sha256": _PROFILE_SHA256,
        "classifier_model_sha256": _MODEL_SHA256,
        "minimum_confidence_bps": 10_000,
        "capability_ttl_s": 1,
        "per_rule_per_minute_limit": 1,
        "per_rule_per_day_limit": 1,
        "max_inflight": 1,
    }


def _scope(epoch: SemanticPolicyEpochV1) -> dict[str, Any]:
    return dict(policy_epoch_storage_record(epoch).rollout_scope)


class _RolloutRepository:
    def __init__(self) -> None:
        self.epochs: dict[str, Any] = {}
        self.control: SemanticControlStateRecord | None = None
        self.evidence: dict[str, dict[str, Any]] = {}
        self.evidence_calls: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        epochs = dict(self.epochs)
        control = self.control
        try:
            yield
        except BaseException:
            self.epochs = epochs
            self.control = control
            raise

    def append_semantic_policy_epoch(self, epoch: SemanticPolicyEpochV1) -> Any:
        record = SimpleNamespace(
            epoch_id=epoch.epoch_id,
            generation=epoch.generation,
            catalog_version=epoch.catalog_version,
            policy_sha256=epoch.canonical_sha256(),
            expected_previous_sha256=epoch.expected_previous_sha256,
            rollout_scope=_scope(epoch),
            created_at=epoch.created_at,
            schema_version=epoch.schema_version,
        )
        existing = self.epochs.get(epoch.epoch_id)
        if existing is not None and vars(existing) != vars(record):
            raise RuntimeError("immutable epoch conflict")
        self.epochs[epoch.epoch_id] = record
        return record

    def get_semantic_policy_epoch(self, epoch_id: str) -> Any | None:
        return self.epochs.get(epoch_id)

    def query_semantic_policy_epochs(
        self,
        *,
        limit: int,
        after: object | None = None,
    ) -> Any:
        assert limit > 0
        assert after is None
        return SimpleNamespace(
            records=tuple(
                sorted(self.epochs.values(), key=lambda item: item.generation)
            ),
            next_cursor=None,
        )

    def get_semantic_control_state(self) -> SemanticControlStateRecord | None:
        return self.control

    def compare_and_set_semantic_control_state(
        self,
        expected: SemanticControlStateRecord | None,
        target: SemanticControlStateRecord,
    ) -> bool:
        if self.control != expected:
            return False
        self.control = control_state_storage_record(target)
        return True

    def semantic_rollout_review_evidence(
        self,
        *,
        epoch_id: str,
        action_id: str,
        limit: int,
    ) -> dict[str, Any]:
        assert epoch_id == "epoch-1"
        assert limit == 1_000
        self.evidence_calls.append(action_id)
        return self.evidence[action_id]

    def semantic_unsafe_review_count(
        self,
        *,
        epoch_id: str | None = None,
    ) -> int:
        # This fake records rollout evidence only for the preceding epoch.
        # Startup checks for the candidate epoch therefore have no direct
        # unsafe labels, while the global expansion gate sees every action.
        if epoch_id is not None:
            return 0
        return sum(
            int(item.get("unsafe_count", 0))
            for item in self.evidence.values()
        )

    def query_semantic_health_events(
        self,
        *,
        limit: int,
        after: object | None = None,
        epoch_id: str | None = None,
        severity: str | None = None,
        event_kind: str | None = None,
    ) -> Any:
        assert limit > 0
        assert after is None
        assert severity is None
        return SimpleNamespace(records=(), next_cursor=None)


def _rule(
    resource: str,
    *,
    action: str = "filesystem.read",
    rule_id: str = "read-reports",
) -> SemanticApprovalRule:
    right = "diff" if action == "git.diff" else "read"
    return SemanticApprovalRule(
        rule_id=rule_id,
        authority_operation=action,
        resource=resource,
        rights=(right,),
    )


def _deny_rule(
    resource: str,
    *,
    action: str = "filesystem.read",
    rule_id: str = "deny-secret-reports",
    rights: tuple[str, ...] | None = None,
) -> SemanticHardDenyRuleV1:
    selected_rights = rights or (
        ("diff",) if action == "git.diff" else ("read",)
    )
    return SemanticHardDenyRuleV1(
        rule_id=rule_id,
        authority_operation=action,
        resource=resource,
        rights=selected_rights,
    )


def _epoch(
    generation: int,
    *,
    tenants: tuple[str, ...],
    rules: tuple[SemanticApprovalRule, ...],
    hard_denies: tuple[SemanticHardDenyRuleV1, ...] = (),
    previous: str | None = None,
    **policy_updates: Any,
) -> SemanticPolicyEpochV1:
    values: dict[str, Any] = {
        "epoch_id": f"epoch-{generation}",
        "generation": generation,
        "expected_previous_sha256": previous,
        "tenant_bucket_sha256s": tenants,
        "auto_approval_rules": rules,
        "hard_deny_rules": hard_denies,
        "created_at": datetime(
            2020, 1, generation, tzinfo=timezone.utc
        ).isoformat(),
    }
    values.update(policy_updates)
    return SemanticPolicyEpochV1(**values)


@pytest.mark.parametrize(
    "candidate_denies",
    [
        (),
        (_deny_rule("filesystem:workspace:reports/secret/private/*"),),
    ],
    ids=("deleted", "narrowed"),
)
def test_rollout_hard_deny_weakening_is_review_gated_after_sqlite_reopen(
    tmp_path: Any,
    candidate_denies: tuple[SemanticHardDenyRuleV1, ...],
) -> None:
    database = tmp_path / "semantic-hard-deny-rollout.db"
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(_deny_rule("filesystem:workspace:reports/secret/*"),),
    )
    store = SQLiteStore(database)
    SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    initial_control = store.get_semantic_control_state()
    assert initial_control is not None
    assert initial_control.generation == 1
    store.close()

    reopened = SQLiteStore(database)
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=candidate_denies,
        previous=first.canonical_sha256(),
    )
    with pytest.raises(SemanticControlConflict, match="seven days"):
        SemanticRuntimeControl(
            reopened,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
        ).admit()

    # The rejected epoch append and control-pointer generation CAS share the
    # same transaction: neither may survive the failed rollout admission.
    assert reopened.get_semantic_control_state() == initial_control
    assert reopened.get_semantic_policy_epoch(second.epoch_id) is None
    reopened.close()

    final_reopen = SQLiteStore(database)
    assert final_reopen.get_semantic_control_state() == initial_control
    assert final_reopen.get_semantic_policy_epoch(second.epoch_id) is None
    final_reopen.close()


@pytest.mark.parametrize(
    "relaxation",
    [
        {"minimum_confidence_bps": 9_900},
        {"capability_ttl_s": 2},
        {"per_rule_per_minute_limit": 2},
        {"per_rule_per_day_limit": 2},
        {"max_inflight": 2},
        {"classifier_profile_id": "replacement-classifier"},
        {"classifier_profile_sha256": "3" * 64},
        {"classifier_model_sha256": "4" * 64},
    ],
    ids=(
        "confidence",
        "ttl",
        "minute-budget",
        "day-budget",
        "inflight-budget",
        "profile-id",
        "profile-artifact",
        "model-artifact",
    ),
)
def test_rollout_allow_shaping_relaxation_is_review_gated_after_sqlite_reopen(
    tmp_path: Any,
    relaxation: dict[str, Any],
) -> None:
    database = tmp_path / "semantic-allow-parameter-rollout.db"
    first_parameters = _strict_allow_parameters()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        **first_parameters,
    )
    store = SQLiteStore(database)
    SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    initial_control = store.get_semantic_control_state()
    assert initial_control is not None
    store.close()

    candidate_parameters = {**first_parameters, **relaxation}
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        previous=first.canonical_sha256(),
        **candidate_parameters,
    )
    reopened = SQLiteStore(database)
    with pytest.raises(SemanticControlConflict, match="seven days"):
        SemanticRuntimeControl(
            reopened,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
        ).admit()
    assert reopened.get_semantic_control_state() == initial_control
    assert reopened.get_semantic_policy_epoch(second.epoch_id) is None
    reopened.close()


def test_legacy_rollout_scope_v1_reopens_but_cannot_authorize_rotation(
    tmp_path: Any,
) -> None:
    database = tmp_path / "semantic-legacy-rollout-scope.db"
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(_deny_rule("filesystem:workspace:reports/secret/*"),),
    )
    current_record = policy_epoch_storage_record(first)
    current_scope = current_record.to_dict()["rollout_scope"]
    legacy_record = SemanticPolicyEpochRecord(
        epoch_id=current_record.epoch_id,
        generation=current_record.generation,
        catalog_version=current_record.catalog_version,
        policy_sha256=current_record.policy_sha256,
        expected_previous_sha256=current_record.expected_previous_sha256,
        rollout_scope={
            "schema_version": 1,
            "tenant_bucket_sha256s": current_scope["tenant_bucket_sha256s"],
            "auto_approval_rules": current_scope["auto_approval_rules"],
        },
        created_at=current_record.created_at,
        schema_version=current_record.schema_version,
    )
    initial_control = SemanticControlStateRecord(
        revision=0,
        generation=1,
        mode=SemanticRuntimeMode.CANARY_AUTO.value,
        active_epoch_id=first.epoch_id,
        active_policy_sha256=first.canonical_sha256(),
        tripped=False,
        trip_code=None,
        updated_at=_ACTIVATED.isoformat(),
    )
    store = SQLiteStore(database)
    store.append_semantic_policy_epoch(legacy_record)
    assert store.compare_and_set_semantic_control_state(None, initial_control)
    store.close()

    reopened = SQLiteStore(database)
    persisted = reopened.get_semantic_policy_epoch(first.epoch_id)
    assert persisted == legacy_record
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        previous=first.canonical_sha256(),
    )
    with pytest.raises(SemanticControlConflict, match="unsupported contract"):
        SemanticRuntimeControl(
            reopened,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: _READY.isoformat(),
        ).admit()
    assert reopened.get_semantic_control_state() == initial_control
    assert reopened.get_semantic_policy_epoch(second.epoch_id) is None
    reopened.close()


def test_rollout_rule_id_budget_shard_replacement_requires_reviews() -> None:
    repository = _RolloutRepository()
    parameters = _strict_allow_parameters()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        **parameters,
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    repository.evidence["filesystem.read"] = _complete_evidence()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(
            _rule(
                "filesystem:workspace:reports/*",
                rule_id="replacement-budget-shard",
            ),
        ),
        previous=first.canonical_sha256(),
        **parameters,
    )

    with pytest.raises(SemanticControlConflict, match="seven days"):
        SemanticRuntimeControl(
            repository,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
        ).admit()

    assert repository.control is not None
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs


def test_rollout_allow_shaping_tightening_and_new_wider_deny_are_immediate() -> None:
    repository = _RolloutRepository()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(
            _deny_rule("filesystem:workspace:reports/secret/private/*"),
        ),
        classifier_profile_id=_PROFILE_ID,
        classifier_profile_sha256=_PROFILE_SHA256,
        classifier_model_sha256=_MODEL_SHA256,
        minimum_confidence_bps=9_900,
        capability_ttl_s=60,
        per_rule_per_minute_limit=10,
        per_rule_per_day_limit=100,
        max_inflight=2,
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(
            _deny_rule("filesystem:workspace:reports/secret/*"),
            _deny_rule(
                "filesystem:workspace:reports/internal/*",
                rule_id="deny-internal-reports",
            ),
        ),
        previous=first.canonical_sha256(),
        classifier_profile_id=_PROFILE_ID,
        classifier_profile_sha256=_PROFILE_SHA256,
        classifier_model_sha256=_MODEL_SHA256,
        minimum_confidence_bps=10_000,
        capability_ttl_s=1,
        per_rule_per_minute_limit=1,
        per_rule_per_day_limit=1,
        max_inflight=1,
    )

    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second,
        now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
    ).admit()

    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.evidence_calls == []


@pytest.mark.parametrize(
    "first_denies,candidate_denies",
    [
        (
            (),
            (_deny_rule("filesystem:workspace:reports/secret/*"),),
        ),
        (
            (_deny_rule("filesystem:workspace:reports/secret/private/*"),),
            (_deny_rule("filesystem:workspace:reports/secret/*"),),
        ),
        (
            (_deny_rule("filesystem:workspace:reports/secret/*"),),
            (
                _deny_rule(
                    "filesystem:workspace:reports/secret/*",
                    rights=("read", "write"),
                ),
            ),
        ),
        (
            (_deny_rule("filesystem:workspace:reports/secret/*"),),
            (
                _deny_rule(
                    "filesystem:workspace:reports/secret/*",
                    rule_id="renamed-equivalent-deny",
                ),
            ),
        ),
    ],
    ids=("new", "resource-wider", "rights-wider", "deny-id-renamed"),
)
def test_rollout_new_or_stronger_hard_deny_is_immediate(
    first_denies: tuple[SemanticHardDenyRuleV1, ...],
    candidate_denies: tuple[SemanticHardDenyRuleV1, ...],
) -> None:
    repository = _RolloutRepository()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=first_denies,
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=candidate_denies,
        previous=first.canonical_sha256(),
    )

    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second,
        now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
    ).admit()

    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.evidence_calls == []


@pytest.mark.parametrize(
    "first_deny,candidate_denies",
    [
        (
            _deny_rule(
                "filesystem:workspace:reports/secret/*",
                rights=("read", "write"),
            ),
            (_deny_rule("filesystem:workspace:reports/secret/*"),),
        ),
        (
            _deny_rule("filesystem:workspace:reports/secret/*"),
            (
                _deny_rule(
                    "git:workspace/reports/secret/*",
                    action="git.read",
                    rule_id="moved-deny-action",
                ),
            ),
        ),
    ],
    ids=("rights-narrowed", "action-moved"),
)
def test_rollout_hard_deny_right_or_action_weakening_requires_reviews(
    first_deny: SemanticHardDenyRuleV1,
    candidate_denies: tuple[SemanticHardDenyRuleV1, ...],
) -> None:
    repository = _RolloutRepository()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(first_deny,),
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    repository.evidence["filesystem.read"] = _complete_evidence()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=candidate_denies,
        previous=first.canonical_sha256(),
    )

    with pytest.raises(SemanticControlConflict, match="seven days"):
        SemanticRuntimeControl(
            repository,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
        ).admit()
    assert repository.control is not None
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs


@pytest.mark.parametrize(
    "inactive_mode",
    [SemanticRuntimeMode.OFF, SemanticRuntimeMode.SHADOW],
)
def test_rollout_hard_deny_weakening_cannot_bypass_inactive_high_water_mark(
    inactive_mode: SemanticRuntimeMode,
) -> None:
    repository = _RolloutRepository()
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        hard_denies=(_deny_rule("filesystem:workspace:reports/secret/*"),),
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    _clear_authority(repository, inactive_mode)
    repository.evidence["filesystem.read"] = _complete_evidence()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        previous=first.canonical_sha256(),
    )

    with pytest.raises(SemanticControlConflict, match="seven days"):
        SemanticRuntimeControl(
            repository,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: (_ACTIVATED + timedelta(minutes=2)).isoformat(),
        ).admit()
    assert repository.control is not None
    assert repository.control.mode == inactive_mode.value
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs


def test_rollout_deny_only_epoch_cannot_seed_a_new_auto_action() -> None:
    repository = _RolloutRepository()
    first = _epoch(
        1,
        tenants=(),
        rules=(),
        hard_denies=(
            _deny_rule(
                "filesystem:workspace:reports/*",
                rule_id="deny-only-read",
            ),
        ),
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.ENFORCE_DENY,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    second = _epoch(
        2,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
        previous=first.canonical_sha256(),
    )

    with pytest.raises(SemanticControlConflict, match="without a preceding canary"):
        SemanticRuntimeControl(
            repository,
            mode=SemanticRuntimeMode.CANARY_AUTO,
            policy_epoch=second,
            now=lambda: _READY.isoformat(),
        ).admit()
    assert repository.control is not None
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs


def _admit_first(
    repository: _RolloutRepository,
    *,
    rule: SemanticApprovalRule | None = None,
) -> SemanticPolicyEpochV1:
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(
            rule or _rule("filesystem:workspace:reports/*"),
        ),
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    return first


def _complete_evidence(**changes: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "activated_at": _ACTIVATED.isoformat(),
        "issued_count": 1_000,
        "required_count": 1_000,
        "completely_safe_count": 1_000,
        "unsafe_count": 0,
        **changes,
    }


def _rotate(
    repository: _RolloutRepository,
    first: SemanticPolicyEpochV1,
    *,
    tenants: tuple[str, ...],
    rules: tuple[SemanticApprovalRule, ...],
    now: datetime = _READY,
) -> None:
    second = _epoch(
        2,
        tenants=tenants,
        rules=rules,
        previous=first.canonical_sha256(),
    )
    SemanticRuntimeControl(
        repository,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second,
        now=lambda: now.isoformat(),
    ).admit()


def _clear_authority(
    repository: _RolloutRepository,
    mode: SemanticRuntimeMode,
) -> None:
    SemanticRuntimeControl(
        repository,
        mode=mode,
        policy_epoch=None,
        now=lambda: (_ACTIVATED + timedelta(minutes=1)).isoformat(),
    ).admit()


def test_rollout_narrowing_does_not_require_reviews() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)

    _rotate(
        repository,
        first,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/summary.txt"),),
        now=_ACTIVATED + timedelta(minutes=1),
    )

    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.evidence_calls == []


@pytest.mark.parametrize(
    "evidence, message",
    [
        (
            _complete_evidence(activated_at=(_ACTIVATED + timedelta(seconds=2)).isoformat()),
            "seven days",
        ),
        (_complete_evidence(issued_count=999, required_count=999), "fewer than 1000"),
        (_complete_evidence(completely_safe_count=999), "1000 complete safe reviews"),
        (_complete_evidence(unsafe_count=1), "unsafe review evidence"),
    ],
)
def test_rollout_expansion_fails_closed_without_complete_proof(
    evidence: dict[str, Any],
    message: str,
) -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    repository.evidence["filesystem.read"] = evidence

    with pytest.raises(SemanticControlConflict, match=message):
        _rotate(
            repository,
            first,
            tenants=(_TENANT_A, _TENANT_B),
            rules=(_rule("filesystem:workspace:reports/*"),),
        )

    assert repository.control is not None
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs


def test_rollout_same_cardinality_tenant_swap_is_an_expansion() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    repository.evidence["filesystem.read"] = _complete_evidence(issued_count=999)

    with pytest.raises(SemanticControlConflict, match="fewer than 1000"):
        _rotate(
            repository,
            first,
            tenants=(_TENANT_B,),
            rules=(_rule("filesystem:workspace:reports/*"),),
        )


def test_rollout_expansion_uses_durable_activation_and_complete_reviews() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    repository.evidence["filesystem.read"] = _complete_evidence()

    _rotate(
        repository,
        first,
        tenants=(_TENANT_A, _TENANT_B),
        rules=(_rule("filesystem:workspace:reports/*"),),
    )

    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.evidence_calls == ["filesystem.read"]


def test_rollout_rule_widening_requires_proof_for_its_action() -> None:
    repository = _RolloutRepository()
    first = _admit_first(
        repository,
        rule=_rule("filesystem:workspace:reports/summary.txt"),
    )
    repository.evidence["filesystem.read"] = _complete_evidence()

    _rotate(
        repository,
        first,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
    )

    assert repository.evidence_calls == ["filesystem.read"]


def test_rollout_cannot_introduce_action_without_preceding_canary() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)

    with pytest.raises(SemanticControlConflict, match="without a preceding canary"):
        _rotate(
            repository,
            first,
            tenants=(_TENANT_A,),
            rules=(
                _rule("filesystem:workspace:reports/*"),
                _rule("git:workspace:repo/*", action="git.read", rule_id="git-read"),
            ),
        )


def test_rollout_ignores_backdated_candidate_created_at() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    repository.evidence["filesystem.read"] = _complete_evidence(
        activated_at=(_READY - timedelta(days=1)).isoformat()
    )

    with pytest.raises(SemanticControlConflict, match="seven days"):
        _rotate(
            repository,
            first,
            tenants=(_TENANT_A, _TENANT_B),
            rules=(_rule("filesystem:workspace:reports/*"),),
        )


@pytest.mark.parametrize(
    "inactive_mode",
    [SemanticRuntimeMode.OFF, SemanticRuntimeMode.SHADOW],
)
def test_rollout_can_reactivate_new_narrower_epoch_after_authority_clear(
    inactive_mode: SemanticRuntimeMode,
) -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    _clear_authority(repository, inactive_mode)

    _rotate(
        repository,
        first,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/summary.txt"),),
        now=_ACTIVATED + timedelta(minutes=2),
    )

    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.control.active_epoch_id == "epoch-2"
    assert repository.evidence_calls == []


def test_rollout_reactivation_after_off_still_gates_expansion() -> None:
    repository = _RolloutRepository()
    first = _admit_first(repository)
    _clear_authority(repository, SemanticRuntimeMode.OFF)
    repository.evidence["filesystem.read"] = _complete_evidence(issued_count=999)

    with pytest.raises(SemanticControlConflict, match="fewer than 1000"):
        _rotate(
            repository,
            first,
            tenants=(_TENANT_A, _TENANT_B),
            rules=(_rule("filesystem:workspace:reports/*"),),
        )
    assert repository.control is not None
    assert repository.control.mode == SemanticRuntimeMode.OFF.value
    assert repository.control.generation == 1
    assert "epoch-2" not in repository.epochs

    repository.evidence["filesystem.read"] = _complete_evidence()
    _rotate(
        repository,
        first,
        tenants=(_TENANT_A, _TENANT_B),
        rules=(_rule("filesystem:workspace:reports/*"),),
    )
    assert repository.control is not None
    assert repository.control.generation == 2
    assert repository.control.active_epoch_id == "epoch-2"


def test_rollout_expansion_uses_real_sqlite_immutable_evidence() -> None:
    store = SQLiteStore(":memory:")
    first = _epoch(
        1,
        tenants=(_TENANT_A,),
        rules=(_rule("filesystem:workspace:reports/*"),),
    )
    SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=first,
        now=lambda: _ACTIVATED.isoformat(),
    ).admit()
    with store.transaction():
        for index in range(1_000):
            suffix = f"{index:04d}"
            settlement = SemanticMachineSettlementRecord(
                settlement_id=f"settlement-{suffix}",
                assessment_id=f"assessment-{suffix}",
                job_id=f"job-{suffix}",
                request_id=f"request-{suffix}",
                request_revision=0,
                pid=f"pid-{suffix}",
                operation_id=f"operation-{suffix}",
                effect_id=f"effect-{suffix}",
                epoch_id=first.epoch_id,
                policy_sha256=first.canonical_sha256(),
                tenant_bucket_sha256=_TENANT_A,
                action_id="filesystem.read",
                outcome="issued",
                capability_id=f"capability-{suffix}",
                binding_sha256="c" * 64,
                decision_sha256="d" * 64,
                matched_rule_id="read-reports",
                reason_codes=("policy_match",),
                created_at=(
                    _ACTIVATED + timedelta(seconds=index + 1)
                ).isoformat(),
            )
            store.append_semantic_machine_settlement(settlement)
            store.append_semantic_review_label(
                SemanticReviewLabelRecord(
                    review_id=f"review-{suffix}",
                    settlement_id=settlement.settlement_id,
                    outcome="safe",
                    reviewer_sha256="e" * 64,
                    evidence_sha256="f" * 64,
                    created_at=(
                        _ACTIVATED + timedelta(seconds=index + 1)
                    ).isoformat(),
                )
            )

    SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.OFF,
        policy_epoch=None,
        now=lambda: (_ACTIVATED + timedelta(days=1)).isoformat(),
    ).admit()

    second = _epoch(
        2,
        tenants=(_TENANT_A, _TENANT_B),
        rules=(_rule("filesystem:workspace:reports/*"),),
        previous=first.canonical_sha256(),
    )
    SemanticRuntimeControl(
        store,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        policy_epoch=second,
        now=lambda: _READY.isoformat(),
    ).admit()

    control = store.get_semantic_control_state()
    assert control is not None
    assert control.generation == 2
    assert control.active_epoch_id == second.epoch_id
    persisted = store.get_semantic_policy_epoch(first.epoch_id)
    assert persisted is not None
    assert "filesystem:workspace:reports" not in str(persisted.rollout_scope)
    store.close()
