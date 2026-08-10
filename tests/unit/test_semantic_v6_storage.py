from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import (
    SemanticApprovalRule,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
)
from agent_libos.storage import (
    SQLiteStore,
    SemanticControlStateRecord,
    SemanticFlowActivityRecord,
    SemanticFlowEdgeRecord,
    SemanticFlowEntityRecord,
    SemanticFlowLabelAssertionRecord,
    SemanticHumanOutcomeLinkRecord,
    SemanticMachineOutcomeRecord,
    SemanticMachineSettlementRecord,
    SemanticRateBudgetRecord,
    SemanticReviewLabelRecord,
)
from agent_libos.storage.semantic_v6 import semantic_rate_budget_bucket_id


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_NOW = "2026-08-07T00:00:00+00:00"


def _entity(identity: str, *, tenant: str = _A) -> SemanticFlowEntityRecord:
    return SemanticFlowEntityRecord(
        entity_id=identity,
        kind="provider_result",
        pid="进程-é",
        tenant_bucket_sha256=tenant,
        content_sha256=_B,
        version_sha256=_C,
        provenance_sha256=_D,
        baseline_labels={
            "sensitivity": "normal",
            "trust_level": "verified",
            "integrity": "checked",
        },
        identity_present=True,
        identity_mixed=False,
        coverage="complete",
        created_at=_NOW,
    )


def _activity(identity: str, *, tenant: str = _A) -> SemanticFlowActivityRecord:
    return SemanticFlowActivityRecord(
        activity_id=identity,
        kind="provider_call",
        pid="进程-é",
        action_id="filesystem.read",
        effect_id="effect-é",
        state_sha256=_B,
        provider_spec_sha256=_C,
        tool_schema_sha256=None,
        model_artifact_sha256=None,
        tenant_bucket_sha256=tenant,
        created_at=_NOW,
    )


def _issued_settlement(identity: str) -> SemanticMachineSettlementRecord:
    return SemanticMachineSettlementRecord(
        settlement_id=identity,
        assessment_id=f"assessment-{identity}",
        job_id=f"job-{identity}",
        request_id=f"request-{identity}",
        request_revision=0,
        pid="pid-recovery",
        operation_id=f"operation-{identity}",
        effect_id=f"effect-{identity}",
        epoch_id="epoch-recovery",
        policy_sha256=_A,
        tenant_bucket_sha256=_B,
        action_id="filesystem.read",
        outcome="issued",
        capability_id=f"capability-{identity}",
        binding_sha256=_C,
        decision_sha256=_D,
        matched_rule_id="rule-recovery",
        reason_codes=("policy_match",),
        created_at=_NOW,
    )


def test_v6_flow_append_is_exact_atomic_and_unicode_keyset(tmp_path: Path) -> None:
    database = tmp_path / "v6.sqlite"
    store = SQLiteStore(database)
    entity = _entity("实体-é")
    activity = _activity("活动-中")
    edge = SemanticFlowEdgeRecord(
        edge_id="边-ß",
        relation="direct",
        source_node_id=entity.entity_id,
        source_node_type="entity",
        target_node_id=activity.activity_id,
        target_node_type="activity",
        pid="进程-é",
        provenance_sha256=_D,
        created_at=_NOW,
    )
    bundle = store.append_semantic_flow_bundle(
        entities=(entity,), activities=(activity,), edges=(edge,)
    )
    assert bundle.entities == (entity,)
    assert store.append_semantic_flow_bundle(
        entities=(entity,), activities=(activity,), edges=(edge,)
    ) == bundle
    assert store.query_semantic_flow_entities(limit=10).records == (entity,)
    assert store.semantic_flow_status_aggregate()["counts"] == {
        "entities": 1,
        "activities": 1,
        "edges": 1,
        "label_assertions": 0,
    }
    assert store.semantic_flow_status_aggregate()["legacy_history"] == {
        "present": False,
        "source_schema_version": None,
        "assessment_count": 0,
        "coverage": None,
        "evidence_sha256": None,
        "created_at": None,
    }
    with pytest.raises(ValidationError, match="cannot be generically deleted"):
        store.delete_table_rows(
            "semantic_flow_entities",
            "entity_id = ?",
            (entity.entity_id,),
        )
    with pytest.raises(ValidationError, match="typed repository"):
        store.insert_table_row("semantic_control_state", {})

    secret_labels = replace(
        entity,
        entity_id="实体-secret-label",
        baseline_labels={
            "sensitivity": "secret",
            "trust_level": "verified",
            "integrity": "checked",
        },
    )
    store.append_semantic_flow_bundle(entities=(secret_labels,))
    assert store.get_semantic_flow_entity(secret_labels.entity_id) == secret_labels

    cross_tenant = _entity("实体-cross", tenant="e" * 64)
    store.append_semantic_flow_bundle(entities=(cross_tenant,))
    bad_edge = SemanticFlowEdgeRecord(
        edge_id="边-cross",
        relation="direct",
        source_node_id=entity.entity_id,
        source_node_type="entity",
        target_node_id=cross_tenant.entity_id,
        target_node_type="entity",
        pid="进程-é",
        provenance_sha256=_D,
        created_at=_NOW,
    )
    with pytest.raises(ValidationError, match="cross tenant"):
        store.append_semantic_flow_bundle(edges=(bad_edge,))
    assert store.query_semantic_flow_edges(limit=10).records == (edge,)
    store.close()

    reopened = SQLiteStore(database)
    assert reopened.get_semantic_flow_entity("实体-é") == entity
    reopened.close()


def test_v6_flow_locator_carrier_is_digest_only_and_reopens(tmp_path: Path) -> None:
    database = tmp_path / "v6-locators.sqlite"
    store = SQLiteStore(database)
    entity = _entity("entity-locators")
    json_locator = SemanticFlowLabelAssertionRecord(
        assertion_id="assertion-json",
        entity_id=entity.entity_id,
        source="model",
        sensitivity_floor="confidential",
        integrity_ceiling="unknown",
        trust_ceiling="unknown",
        evidence_sha256=_A,
        assessment_id="assessment-json",
        locator_sha256=_B,
        locator_kind="json_field",
        path_sha256s=(_C, _D),
        value_sha256=_A,
        category="personal_data",
        coverage="complete",
        created_at=_NOW,
    )
    text_locator = SemanticFlowLabelAssertionRecord(
        assertion_id="assertion-text",
        entity_id=entity.entity_id,
        source="deterministic",
        sensitivity_floor="restricted",
        integrity_ceiling="untrusted",
        trust_ceiling="untrusted",
        evidence_sha256=_B,
        assessment_id=None,
        locator_sha256=_C,
        locator_kind="text_chunk",
        value_sha256=_D,
        ordinal=2,
        offset_start=64,
        offset_end=128,
        category="credential",
        coverage="partial",
        created_at=_NOW,
    )
    store.append_semantic_flow_bundle(
        entities=(entity,), assertions=(json_locator, text_locator)
    )
    rows = store.conn.execute(
        "SELECT path_sha256s_json FROM semantic_flow_label_assertions "
        "ORDER BY assertion_id"
    ).fetchall()
    assert [row["path_sha256s_json"] for row in rows] == [
        f'["{_C}","{_D}"]',
        "[]",
    ]
    assert store.query_semantic_flow_label_assertions(
        entity_id=entity.entity_id, limit=10
    ).records == (json_locator, text_locator)
    store.close()

    reopened = SQLiteStore(database)
    assert reopened.query_semantic_flow_label_assertions(
        entity_id=entity.entity_id, limit=10
    ).records == (json_locator, text_locator)
    reopened.close()

    with pytest.raises(ValidationError, match="path segment"):
        replace(json_locator, path_sha256s=("customer",))
    with pytest.raises(ValidationError, match="end must exceed"):
        replace(text_locator, offset_end=64)


def test_v6_human_outcome_link_is_append_only_and_request_joined(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v6-human-outcome.sqlite"
    store = SQLiteStore(database)
    link = SemanticHumanOutcomeLinkRecord(
        link_id="human-link-1",
        request_id="human-request-1",
        request_revision=2,
        pid="pid-human-link",
        assessment_id=None,
        job_id=None,
        settlement_id=None,
        outcome="approved",
        source="human",
        decision_sha256=_A,
        created_at=_NOW,
    )
    assert store.append_semantic_human_outcome_link(link) == link
    assert store.append_semantic_human_outcome_link(link) == link
    assert store.get_semantic_human_outcome_link_for_request(
        link.request_id
    ) == link
    assert store.query_semantic_human_outcome_links(
        limit=10, pid=link.pid
    ).records == (link,)
    assert store.semantic_human_outcome_link_counts() == {
        "total": 1,
        "outcomes": {"approved": 1, "rejected": 0, "cancelled": 0},
        "sources": {"human": 1, "machine_policy": 0, "cancel": 0},
    }
    with pytest.raises(ValidationError, match="conflicts"):
        store.append_semantic_human_outcome_link(
            replace(link, link_id="human-link-conflict", outcome="rejected")
        )
    with pytest.raises(ValidationError, match="cannot be generically deleted"):
        store.delete_table_rows(
            "semantic_human_outcome_links", "link_id = ?", (link.link_id,)
        )
    unicode_link = SemanticHumanOutcomeLinkRecord(
        link_id="结果-é",
        request_id="请求-é",
        request_revision=4,
        pid="进程-é",
        assessment_id=None,
        job_id=None,
        settlement_id=None,
        outcome="cancelled",
        source="cancel",
        decision_sha256=_B,
        created_at=_NOW,
    )
    store.append_semantic_human_outcome_link(unicode_link)
    first_page = store.query_semantic_human_outcome_links(limit=1)
    assert first_page.records == (link,)
    assert first_page.next_cursor is not None
    assert store.query_semantic_human_outcome_links(
        limit=1, after=first_page.next_cursor
    ).records == (unicode_link,)

    store.conn.execute(
        "INSERT INTO semantic_assessments "
        "(assessment_id, job_id, kind, status, domain, action_id, "
        "tenant_bucket_sha256, pid, request_id, operation_id, effect_id, "
        "shadow_outcome, ood, record_json, created_at, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "assessment-human-link",
            "job-human-link",
            "approval",
            "succeeded",
            "filesystem",
            "filesystem.read",
            _B,
            link.pid,
            link.request_id,
            None,
            None,
            "require_human",
            0,
            "{}",
            _NOW,
            _NOW,
        ),
    )
    store.conn.commit()
    settlement = replace(
        _issued_settlement("human-link"),
        request_id=link.request_id,
        pid=link.pid,
    )
    store.append_semantic_machine_settlement(settlement)
    assert store.semantic_human_outcome_links_for_assessments(
        ("assessment-human-link",)
    ) == {"assessment-human-link": link}
    assert store.semantic_human_outcome_links_for_settlements(
        (settlement.settlement_id,)
    ) == {settlement.settlement_id: link}
    store.close()

    reopened = SQLiteStore(database)
    assert reopened.get_semantic_human_outcome_link_for_request(
        link.request_id
    ) == link
    reopened.close()


def test_v6_control_cas_outcome_idempotence_and_metrics() -> None:
    store = SQLiteStore(":memory:")
    control = SemanticControlStateRecord(
        revision=0,
        generation=0,
        mode="off",
        active_epoch_id=None,
        active_policy_sha256=None,
        tripped=False,
        trip_code=None,
        updated_at=_NOW,
    )
    assert store.compare_and_set_semantic_control_state(None, control)
    assert not store.compare_and_set_semantic_control_state(None, control)
    assert len(store.query_semantic_control_history(limit=10).records) == 1
    with pytest.raises(ValidationError, match="outer UnitOfWork transaction"):
        store.fence_semantic_control_state(control)
    with store.transaction():
        assert store.fence_semantic_control_state(control)
        assert not store.fence_semantic_control_state(
            replace(control, revision=1)
        )
    assert store.get_semantic_control_state() == control

    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-é",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(_B,),
        auto_approval_rules=(
            SemanticApprovalRule(
                rule_id="rule-é",
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/*",
                rights=("read",),
            ),
        ),
        hard_deny_rules=(),
        created_at=_NOW,
    )
    persisted_epoch = store.append_semantic_policy_epoch(epoch)
    assert store.get_semantic_policy_epoch(epoch.epoch_id) == persisted_epoch
    rollout_rule = persisted_epoch.rollout_scope["auto_approval_rules"][0]
    assert rollout_rule["rule_id_sha256"] == hashlib.sha256(
        "rule-é".encode("utf-8")
    ).hexdigest()
    assert rollout_rule["action_id"] == "filesystem.read"
    assert "filesystem:workspace:reports" not in str(
        persisted_epoch.rollout_scope
    )
    active_control = replace(
        control,
        revision=1,
        generation=1,
        mode="canary_auto",
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
    )
    assert store.compare_and_set_semantic_control_state(control, active_control)

    settlement = SemanticMachineSettlementRecord(
        settlement_id="settlement-é",
        assessment_id="assessment-é",
        job_id="job-é",
        request_id="request-é",
        request_revision=0,
        pid="pid-é",
        operation_id="operation-é",
        effect_id="effect-é",
        epoch_id="epoch-é",
        policy_sha256=_A,
        tenant_bucket_sha256=_B,
        action_id="filesystem.read",
        outcome="issued",
        capability_id="capability-é",
        binding_sha256=_C,
        decision_sha256=_D,
        matched_rule_id="rule-é",
        reason_codes=("policy_match",),
        created_at=_NOW,
    )
    store.append_semantic_machine_settlement(settlement)
    outcome = SemanticMachineOutcomeRecord(
        outcome_id="outcome-é",
        settlement_id=settlement.settlement_id,
        effect_id=settlement.effect_id,
        outcome="issued",
        evidence_sha256=_A,
        created_at=_NOW,
    )
    assert store.append_semantic_machine_outcome_if_absent(outcome)
    assert not store.append_semantic_machine_outcome_if_absent(outcome)
    denied = replace(
        settlement,
        settlement_id="settlement-denied-é",
        request_id="request-denied-é",
        effect_id="effect-denied-é",
        action_id="git.read",
        outcome="denied",
        capability_id=None,
    )
    store.append_semantic_machine_settlement(denied)
    store.append_semantic_review_label(
        SemanticReviewLabelRecord(
            review_id="review-denied-é",
            settlement_id=denied.settlement_id,
            outcome="safe",
            reviewer_sha256=_C,
            evidence_sha256=_D,
            created_at=_NOW,
        )
    )
    denied_only_review = store.semantic_metrics(action_id="git.read")
    assert denied_only_review["machine"]["eligible"] == 0
    assert denied_only_review["actual_auto_approval"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    assert denied_only_review["review_metrics"] == {
        "reviewed": 1,
        "safe": 1,
        "unsafe": 0,
        "issued_reviewed": 0,
        "issued_review_rate": None,
        "unsafe_rate": 0.0,
    }
    assert store.semantic_rollout_review_evidence(
        epoch.epoch_id, "filesystem.read"
    ) == {
        "schema_version": 1,
        "activated_at": "2026-08-07T00:00:00.000000+00:00",
        "issued_count": 1,
        "required_count": 1,
        "completely_safe_count": 0,
        "unsafe_count": 0,
    }
    with pytest.raises(ValidationError, match="outside catalog"):
        store.semantic_rollout_review_evidence(epoch.epoch_id, "shell.run")
    store.append_semantic_review_label(
        SemanticReviewLabelRecord(
            review_id="review-é",
            settlement_id=settlement.settlement_id,
            outcome="safe",
            reviewer_sha256=_C,
            evidence_sha256=_D,
            created_at=_NOW,
        )
    )
    assert store.semantic_metrics() == {
        "machine": {
            "eligible": 1,
            "issued": 1,
            "consumed": 0,
            "succeeded": 0,
            "failed": 0,
            "unknown": 0,
            "expired": 0,
            "revoked": 0,
            "race_lost": 0,
            "denied": 1,
        },
        "actual_auto_approval": {
            "numerator": 1,
            "denominator": 1,
            "rate": 1.0,
        },
        "review_metrics": {
            "reviewed": 2,
            "safe": 2,
            "unsafe": 0,
            "issued_reviewed": 1,
            "issued_review_rate": 1.0,
            "unsafe_rate": 0.0,
        },
    }
    assert store.semantic_rollout_review_evidence(
        epoch.epoch_id, "filesystem.read"
    )["completely_safe_count"] == 1

    for review_id, review_outcome in (
        ("review-inconclusive-é", "inconclusive"),
        ("review-unsafe-é", "unsafe"),
        ("review-unsafe-duplicate-é", "unsafe"),
        ("review-safe-duplicate-é", "safe"),
    ):
        store.append_semantic_review_label(
            SemanticReviewLabelRecord(
                review_id=review_id,
                settlement_id=settlement.settlement_id,
                outcome=review_outcome,
                reviewer_sha256=_C,
                evidence_sha256=_D,
                created_at=_NOW,
            )
        )
    conflicting_reviews = store.semantic_metrics()
    assert conflicting_reviews["review_metrics"] == {
        "reviewed": 2,
        "safe": 1,
        "unsafe": 1,
        "issued_reviewed": 1,
        "issued_review_rate": 1.0,
        "unsafe_rate": 0.5,
    }
    conflicting_rollout = store.semantic_rollout_review_evidence(
        epoch.epoch_id, "filesystem.read"
    )
    assert conflicting_rollout["completely_safe_count"] == 0
    assert conflicting_rollout["unsafe_count"] == 1
    assert store.semantic_unsafe_review_count() == 1
    assert store.semantic_unsafe_review_count(epoch_id=epoch.epoch_id) == 1
    assert store.semantic_unsafe_review_count(epoch_id="missing-epoch") == 0
    store.close()


def test_v6_git_diff_epoch_rollout_scope_is_digest_only_and_round_trips() -> None:
    store = SQLiteStore(":memory:")
    try:
        epoch = SemanticPolicyEpochV1(
            epoch_id="git-diff-epoch",
            generation=1,
            expected_previous_sha256=None,
            tenant_bucket_sha256s=(_A,),
            auto_approval_rules=(
                SemanticApprovalRule(
                    rule_id="git-diff-rule",
                    authority_operation="git.diff",
                    resource="git:workspace/*",
                    rights=("diff",),
                ),
            ),
            hard_deny_rules=(),
            created_at=_NOW,
        )
        persisted = store.append_semantic_policy_epoch(epoch)
        assert store.get_semantic_policy_epoch(epoch.epoch_id) == persisted
        rule = persisted.rollout_scope["auto_approval_rules"][0]
        assert rule["action_id"] == "git.diff"
        assert rule["rights"] == ("diff",)
        stored_json = store.conn.execute(
            "SELECT rollout_scope_json FROM semantic_policy_epochs "
            "WHERE epoch_id = ?",
            (epoch.epoch_id,),
        ).fetchone()[0]
        assert "git:workspace" not in stored_json
        assert "git-diff-rule" not in stored_json
    finally:
        store.close()


def test_v6_rollout_scope_commits_deny_and_allow_shaping_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v6-rollout-scope.sqlite"
    profile_id = "host-classifier-sensitive-name"
    deny_id = "deny-secret-report"
    deny_resource = "filesystem:workspace:reports/secret/*"
    epoch = SemanticPolicyEpochV1(
        epoch_id="rollout-scope-epoch",
        generation=1,
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
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id=deny_id,
                authority_operation="filesystem.read",
                resource=deny_resource,
                rights=("read",),
            ),
        ),
        classifier_profile_id=profile_id,
        classifier_profile_sha256=_B,
        classifier_model_sha256=_C,
        minimum_confidence_bps=10_000,
        capability_ttl_s=1,
        per_rule_per_minute_limit=1,
        per_rule_per_day_limit=1,
        max_inflight=1,
        created_at=_NOW,
    )
    store = SQLiteStore(database)
    persisted = store.append_semantic_policy_epoch(epoch)
    deny = persisted.rollout_scope["hard_deny_rules"][0]
    assert deny["rule_id_sha256"] == hashlib.sha256(
        deny_id.encode("utf-8")
    ).hexdigest()
    assert deny["action_id"] == "filesystem.read"
    parameters = persisted.rollout_scope["allow_parameters"]
    assert parameters["classifier_profile_id_sha256"] == hashlib.sha256(
        profile_id.encode("utf-8")
    ).hexdigest()
    assert parameters["minimum_confidence_bps"] == 10_000
    assert parameters["capability_ttl_s"] == 1
    stored_json = store.conn.execute(
        "SELECT rollout_scope_json FROM semantic_policy_epochs WHERE epoch_id = ?",
        (epoch.epoch_id,),
    ).fetchone()[0]
    assert deny_id not in stored_json
    assert deny_resource not in stored_json
    assert profile_id not in stored_json
    store.close()

    reopened = SQLiteStore(database)
    assert reopened.get_semantic_policy_epoch(epoch.epoch_id) == persisted
    reopened.close()


def test_v6_rate_budget_scope_is_unique_across_origin_epochs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v6-rate-budget-scope.sqlite"
    bucket_id = semantic_rate_budget_bucket_id(
        tenant_bucket_sha256=_A,
        rule_id="read-reports",
    )
    record = SemanticRateBudgetRecord(
        bucket_id=bucket_id,
        epoch_id="origin-epoch-1",
        tenant_bucket_sha256=_A,
        rule_id="read-reports",
        minute_window_started_at=_NOW,
        day_window_started_at=_NOW,
        minute_count=1,
        day_count=1,
        inflight_count=0,
        revision=0,
        updated_at=_NOW,
    )
    store = SQLiteStore(database)
    assert store.compare_and_set_semantic_rate_budget(None, record)
    with pytest.raises(ValidationError, match="stable tenant/rule scope"):
        replace(record, bucket_id="semantic-budget-v2:" + _D)
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO semantic_rate_budgets "
            "(bucket_id, epoch_id, tenant_bucket_sha256, rule_id, "
            "minute_window_started_at, day_window_started_at, minute_count, "
            "day_count, inflight_count, revision, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "direct-db-alternate-bucket",
                "origin-epoch-2",
                _A,
                "read-reports",
                _NOW,
                _NOW,
                0,
                0,
                0,
                0,
                _NOW,
            ),
        )
    store.close()

    reopened = SQLiteStore(database)
    assert reopened.get_semantic_rate_budget(bucket_id) == record
    reopened.close()


def test_unresolved_settlement_query_filters_terminal_history_before_keyset_limit() -> None:
    store = SQLiteStore(":memory:")
    try:
        with store.transaction():
            for index in range(501):
                identity = f"terminal-{index:04d}"
                settlement = _issued_settlement(identity)
                store.append_semantic_machine_settlement(settlement)
                store.append_semantic_machine_outcome(
                    SemanticMachineOutcomeRecord(
                        outcome_id=f"outcome-{identity}",
                        settlement_id=identity,
                        effect_id=settlement.effect_id,
                        outcome="succeeded",
                        evidence_sha256=_A,
                        created_at=_NOW,
                    )
                )
            first = _issued_settlement("unresolved-a")
            second = _issued_settlement("unresolved-b")
            store.append_semantic_machine_settlement(first)
            store.append_semantic_machine_settlement(second)
            store.append_semantic_machine_outcome(
                SemanticMachineOutcomeRecord(
                    outcome_id="outcome-unresolved-b-consumed",
                    settlement_id=second.settlement_id,
                    effect_id=second.effect_id,
                    outcome="consumed",
                    evidence_sha256=_A,
                    created_at=_NOW,
                )
            )

        first_page = store.query_unresolved_semantic_machine_settlements(
            limit=1,
            epoch_id="epoch-recovery",
            action_id="filesystem.read",
        )
        assert first_page.records == (first,)
        assert first_page.next_cursor is not None
        second_page = store.query_unresolved_semantic_machine_settlements(
            limit=1,
            after=first_page.next_cursor,
            epoch_id="epoch-recovery",
            action_id="filesystem.read",
        )
        assert second_page.records == (second,)
        assert second_page.next_cursor is None
        assert store.query_unresolved_semantic_machine_settlements(
            limit=10,
            action_id="git.diff",
        ).records == ()
    finally:
        store.close()
