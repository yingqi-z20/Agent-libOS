from __future__ import annotations

import hashlib
import json

import pytest

from agent_libos.config import AgentLibOSConfig, LLMDefaults, LLMProfile, SemanticDefaults
from agent_libos.capability.effect_binding import APPROVAL_BINDING_KEY, canonical_effect_hash
from agent_libos.models import (
    AuthoritativeApprovalFacts,
    CanonicalApprovalPreviewV1,
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
    SemanticApprovalArgumentKind,
    SemanticApprovalArgumentProjectionV1,
    SemanticApprovalBindingV2,
    SemanticApprovalCandidate,
    SemanticApprovalCandidateSnapshotV1,
    SemanticApprovalRule,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticControlStateV1,
    SemanticDomain,
    SemanticFlowCoverage,
    SemanticFlowStatusV1,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
    SemanticPreviewLabelsV1,
    SemanticPreviewRisk,
    SemanticPublicControlState,
    SemanticRatioV1,
    SemanticReviewMetricsV1,
    SemanticRuntimeMode,
    SemanticStatusControlV3,
    SemanticStatusV3,
)
from agent_libos.semantic import ActionOntology, DeterministicApprovalBroker
from agent_libos.semantic.ontology import SemanticActionDefinition
from agent_libos.models import AuthorityRisk
from agent_libos.semantic.service import SemanticManager


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_MODEL_SHA256 = hashlib.sha256(b"semantic-classifier-v1").hexdigest()


def _rule() -> SemanticApprovalRule:
    return SemanticApprovalRule(
        rule_id="workspace-read-v1",
        authority_operation="filesystem.read",
        resource="filesystem:workspace:reports/*",
        rights=("read",),
    )


def _epoch(**updates: object) -> SemanticPolicyEpochV1:
    values: dict[str, object] = {
        "epoch_id": "semantic-epoch-1",
        "generation": 1,
        "expected_previous_sha256": None,
        "tenant_bucket_sha256s": (_A,),
        "auto_approval_rules": (_rule(),),
        "hard_deny_rules": (),
        "classifier_profile_id": "semantic",
        "classifier_profile_sha256": _C,
        "classifier_model_sha256": _MODEL_SHA256,
        "created_at": "2027-01-01T00:00:00+00:00",
    }
    values.update(updates)
    return SemanticPolicyEpochV1(**values)  # type: ignore[arg-type]


def _binding(**updates: object) -> SemanticApprovalBindingV2:
    values: dict[str, object] = {
        "request_id": "request-1",
        "request_revision": 0,
        "pid": "pid-1",
        "operation_id": "operation-1",
        "effect_id": "eff_1",
        "authority_operation": "filesystem.read",
        "resource": "filesystem:workspace:reports/q1.txt",
        "right": "read",
        "canonical_args_hash": _A,
        "target_state_version": 7,
        "manifest_id": "manifest-1",
        "manifest_sha256": _B,
        "ceiling_sha256": _C,
        "policy_epoch_id": "semantic-epoch-1",
        "policy_epoch_sha256": _D,
        "control_generation": 1,
        "assessment_id": "assessment-1",
        "assessment_sha256": _A,
        "classifier_profile_sha256": _B,
        "classifier_model_sha256": _C,
        "tenant_bucket_sha256": _D,
        "source_labels_sha256": _A,
        "source_refs_sha256": _B,
        "flow_snapshot_sha256": _C,
        "sink_identity_sha256": None,
        "tool_schema_sha256": None,
        "provider_spec_sha256": None,
        "nonce": "nonce-1",
        "issued_at": "2027-01-01T00:00:00+00:00",
        "expires_at": "2027-01-01T00:01:00+00:00",
    }
    values.update(updates)
    return SemanticApprovalBindingV2(**values)  # type: ignore[arg-type]


def test_policy_epoch_is_digest_only_closed_catalog_and_round_trips() -> None:
    epoch = _epoch()

    assert SemanticPolicyEpochV1.from_dict(epoch.to_dict()) == epoch
    assert len(epoch.canonical_sha256()) == 64
    assert epoch.to_dict()["tenant_bucket_sha256s"] == [_A]
    assert "tenant_buckets" not in epoch.to_dict()

    with pytest.raises(ValueError, match="tenant bucket"):
        _epoch(tenant_bucket_sha256s=("tenant-a",))
    non_json_epoch = epoch.to_dict()
    non_json_epoch["tenant_bucket_sha256s"] = (_A,)
    with pytest.raises(TypeError, match="must be an array"):
        SemanticPolicyEpochV1.from_dict(non_json_epoch)
    with pytest.raises(ValueError, match="catalog v1"):
        _epoch(
            auto_approval_rules=(
                SemanticApprovalRule(
                    "shell-auto",
                    "shell.run",
                    "shell:workspace:*",
                    ("execute",),
                ),
            )
        )
    with pytest.raises(ValueError, match="resource kind"):
        _epoch(
            auto_approval_rules=(
                SemanticApprovalRule(
                    "wrong-resource-kind",
                    "filesystem.read",
                    "git:workspace",
                    ("read",),
                ),
            )
        )
    with pytest.raises(ValueError, match="canonical terminal segment"):
        _epoch(
            auto_approval_rules=(
                SemanticApprovalRule(
                    "ambiguous-prefix",
                    "filesystem.read",
                    "filesystem:workspace:report*",
                    ("read",),
                ),
            )
        )
    with pytest.raises(ValueError, match="must not exceed 10"):
        _epoch(per_rule_per_minute_limit=11)
    with pytest.raises(ValueError, match="supplied together"):
        _epoch(classifier_profile_sha256=None)
    with pytest.raises(ValueError, match="exactly one tenant bucket"):
        _epoch(tenant_bucket_sha256s=(_A, _B))

    deny_only = _epoch(
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="deny-shell",
                authority_operation="shell.run",
                resource="shell:workspace",
                rights=("execute",),
            ),
        ),
        classifier_profile_id=None,
        classifier_profile_sha256=None,
        classifier_model_sha256=None,
    )
    assert deny_only.tenant_bucket_sha256s == ()
    assert AgentLibOSConfig(
        semantic=SemanticDefaults(
            mode="enforce_deny",
            policy_epoch=deny_only,
        )
    ).semantic.mode == "enforce_deny"

    second_rule = SemanticApprovalRule(
        rule_id="workspace-read-v2",
        authority_operation="filesystem.read",
        resource="filesystem:workspace:archive/*",
        rights=("read",),
    )
    ordered = _epoch(
        generation=2,
        expected_previous_sha256=_D,
        tenant_bucket_sha256s=(_A, _B),
        auto_approval_rules=(_rule(), second_rule),
    )
    reversed_order = _epoch(
        generation=2,
        expected_previous_sha256=_D,
        tenant_bucket_sha256s=(_B, _A),
        auto_approval_rules=(second_rule, _rule()),
    )
    assert ordered == reversed_order
    assert ordered.canonical_sha256() == reversed_order.canonical_sha256()


def test_active_config_requires_epoch_and_canary_requires_external_pin() -> None:
    with pytest.raises(ValueError, match="requires a static policy_epoch"):
        AgentLibOSConfig(semantic=SemanticDefaults(mode="enforce_deny"))
    with pytest.raises(ValueError, match="external classifier adapter"):
        AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode="canary_auto",
                adapter="scripted",
                policy_epoch=_epoch(),
            )
        )

    profiles = {
        "default": LLMProfile(),
        "semantic": LLMProfile(
            model="semantic-classifier-v1",
            api_mode="responses",
            timeout_s=5.0,
            max_retries=0,
            store=False,
            responses_previous_response_id=False,
            fallback_json_actions=False,
        ),
    }
    configured = AgentLibOSConfig(
        llm=LLMDefaults(profiles=profiles),
        semantic=SemanticDefaults(
            mode="canary_auto",
            adapter="external",
            external_profile_id="semantic",
            policy_epoch=_epoch(),
        ),
    )
    assert configured.semantic.mode == "canary_auto"
    with pytest.raises(ValueError, match="model digest does not match"):
        AgentLibOSConfig(
            llm=LLMDefaults(profiles=profiles),
            semantic=SemanticDefaults(
                mode="canary_auto",
                adapter="external",
                external_profile_id="semantic",
                policy_epoch=_epoch(classifier_model_sha256=_B),
            ),
        )


def test_inactive_control_cannot_retain_epoch_or_trip_authority() -> None:
    with pytest.raises(ValueError, match="inactive semantic modes"):
        SemanticControlStateV1(
            revision=1,
            generation=1,
            mode=SemanticRuntimeMode.OFF,
            active_epoch_id="semantic-epoch-1",
            active_policy_sha256=_A,
            tripped=False,
            trip_code=None,
            updated_at="2027-01-01T00:00:00+00:00",
        )

    with pytest.raises(ValueError, match="inactive semantic modes"):
        SemanticControlStateV1(
            revision=1,
            generation=1,
            mode=SemanticRuntimeMode.SHADOW,
            active_epoch_id=None,
            active_policy_sha256=None,
            tripped=True,
            trip_code="binding_mismatch",
            updated_at="2027-01-01T00:00:00+00:00",
        )


def test_binding_v2_is_exact_short_lived_and_legacy_projection_is_one_way() -> None:
    binding = _binding()

    assert SemanticApprovalBindingV2.from_dict(binding.to_dict()) == binding
    assert binding.to_legacy_effect_binding() == {
        "effect_id": "eff_1",
        "canonical_args_hash": _A,
        "target_state_version": 7,
    }
    with pytest.raises(ValueError, match="fields must be exactly"):
        SemanticApprovalBindingV2.from_dict(binding.to_legacy_effect_binding())
    with pytest.raises(ValueError, match="lifetime"):
        _binding(expires_at="2027-01-01T00:05:01+00:00")
    with pytest.raises(ValueError, match="catalog v1"):
        _binding(authority_operation="shell.run", right="execute")
    with pytest.raises(ValueError, match="resource kind"):
        _binding(resource="git:workspace:reports/q1.txt")


def test_shadow_candidate_snapshot_is_digest_only_and_never_canary_authority() -> None:
    resource = "filesystem:workspace:reports/q1.txt"
    exact = SemanticApprovalCandidate(
        rule_id="workspace-read-v1",
        authority_operation="filesystem.read",
        resource=resource,
        rights=("read",),
        manifest_id="manifest-1",
        manifest_sha256=_C,
        policy_sha256=_D,
    )
    snapshot = SemanticApprovalCandidateSnapshotV1.from_candidate(exact)
    wire = snapshot.to_dict()

    assert "resource" not in wire
    assert resource not in json.dumps(wire, sort_keys=True)
    assert SemanticApprovalCandidateSnapshotV1.from_dict(wire) == snapshot
    with pytest.raises(ValueError, match="fields must be exactly"):
        SemanticApprovalCandidateSnapshotV1.from_dict(
            {**wire, "resource": resource}
        )

    facts = AuthoritativeApprovalFacts(
        **{name: True for name in AuthoritativeApprovalFacts.__dataclass_fields__}
    )
    assessment = SemanticAssessment(
        status=SemanticAssessmentStatus.SUCCESS,
        confidence_bps=10_000,
        calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
    )
    broker = DeterministicApprovalBroker()
    shadow = broker.decide(
        assessment=assessment,
        facts=facts,
        policy_sha256=_D,
        candidate=snapshot,
    )
    assert shadow.outcome.value == "would_issue_exact_once"
    assert shadow.matched_rule_id == exact.rule_id

    epoch = _epoch()
    control = SemanticControlStateV1(
        revision=1,
        generation=1,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
        tripped=False,
        trip_code=None,
        updated_at="2027-01-01T00:00:00+00:00",
    )
    with pytest.raises(TypeError, match="exact SemanticApprovalCandidate"):
        broker.decide_canary(
            assessment=assessment,
            facts=facts,
            policy_sha256=_D,
            candidate=snapshot,  # type: ignore[arg-type]
            epoch=epoch,
            control=control,
            tenant_bucket_sha256=_A,
            flow_coverage=SemanticFlowCoverage.COMPLETE,
            classifier_profile_sha256=_C,
            classifier_model_sha256=_MODEL_SHA256,
        )


def test_preview_wire_does_not_expose_source_identity() -> None:
    sentinel = "SEMANTIC_PREVIEW_TENANT_SENTINEL"
    labels = SemanticPreviewLabelsV1.from_data_labels(
        DataLabels(
            sensitivity=DataSensitivity.NORMAL,
            integrity=DataIntegrity.VERIFIED,
            trust_level=DataTrustLevel.TRUSTED,
            tenant=f"tenant-{sentinel}",
            principal=f"principal-{sentinel}",
            declassification_authority=sentinel,
        )
    )
    preview = CanonicalApprovalPreviewV1(
        request_id="request-1",
        revision=0,
        pid="pid-1",
        action_id="filesystem.read",
        resource_display="<redacted>",
        resource_sha256=hashlib.sha256(
            b"filesystem:workspace:reports/q1.txt"
        ).hexdigest(),
        rights=("read",),
        effect_id="eff_1",
        canonical_args_sha256=_A,
        argument_projection=SemanticApprovalArgumentProjectionV1(
            kind=SemanticApprovalArgumentKind.FILESYSTEM,
            operation="read",
            path_sha256=_B,
            read_max_bytes=65_536,
        ),
        target_state_sha256=None,
        risk=SemanticPreviewRisk.LOW,
        source_labels=labels,
        expires_at=None,
    )

    encoded = json.dumps(preview.to_dict(), sort_keys=True)
    assert sentinel not in encoded
    assert set(preview.to_dict()["source_labels"]) == {
        "sensitivity",
        "integrity",
        "trust_level",
        "identity_present",
        "identity_mixed",
    }
    assert CanonicalApprovalPreviewV1.from_dict(preview.to_dict()) == preview


def test_canary_broker_requires_epoch_flow_tenant_and_classifier_provenance() -> None:
    epoch = _epoch()
    policy_sha256 = epoch.canonical_sha256()
    control = SemanticControlStateV1(
        revision=1,
        generation=1,
        mode=SemanticRuntimeMode.CANARY_AUTO,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=policy_sha256,
        tripped=False,
        trip_code=None,
        updated_at="2027-01-01T00:00:00+00:00",
    )
    candidate = SemanticApprovalCandidate(
        rule_id="workspace-read-v1",
        authority_operation="filesystem.read",
        resource="filesystem:workspace:reports/q1.txt",
        rights=("read",),
        manifest_id="manifest-1",
        manifest_sha256=_C,
        policy_sha256=_D,
    )
    assessment = SemanticAssessment(
        status=SemanticAssessmentStatus.SUCCESS,
        confidence_bps=10_000,
        calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
    )
    facts = AuthoritativeApprovalFacts(
        **{name: True for name in AuthoritativeApprovalFacts.__dataclass_fields__}
    )
    broker = DeterministicApprovalBroker()
    profile_sha256 = _C

    eligible = broker.decide_canary(
        assessment=assessment,
        facts=facts,
        policy_sha256=_D,
        candidate=candidate,
        epoch=epoch,
        control=control,
        tenant_bucket_sha256=_A,
        flow_coverage=SemanticFlowCoverage.COMPLETE,
        classifier_profile_sha256=profile_sha256,
        classifier_model_sha256=_MODEL_SHA256,
    )
    assert eligible.outcome.value == "would_issue_exact_once"

    incomplete = broker.decide_canary(
        assessment=assessment,
        facts=facts,
        policy_sha256=_D,
        candidate=candidate,
        epoch=epoch,
        control=control,
        tenant_bucket_sha256=_A,
        flow_coverage=SemanticFlowCoverage.UNKNOWN,
        classifier_profile_sha256=profile_sha256,
        classifier_model_sha256=_MODEL_SHA256,
    )
    assert incomplete.outcome.value == "require_human"


def test_status_v3_matches_the_strict_public_wire() -> None:
    by_status = {item.value: 0 for item in SemanticAssessmentStatus}
    by_domain = {item.value: 0 for item in SemanticDomain}
    status = SemanticStatusV3(
        mode=SemanticRuntimeMode.OFF,
        adapter="deterministic",
        profile_id=None,
        queue={
            "queued": 0,
            "leased": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
            "capture_failures": 0,
        },
        assessments={
            "total": 0,
            "success": 0,
            "error": 0,
            "ood": 0,
            "would_issue_exact_once": 0,
            "would_deny": 0,
            "require_human": 0,
            "by_status": by_status,
            "by_domain": by_domain,
        },
        control=SemanticStatusControlV3(
            catalog_version=None,
            active_epoch_id=None,
            active_epoch_sha256=None,
            generation=0,
            state=SemanticPublicControlState.INACTIVE,
            trip_reason_code=None,
        ),
        flow=SemanticFlowStatusV1(
            available=False,
            counts={
                "entities": 0,
                "activities": 0,
                "edges": 0,
                "label_assertions": 0,
            },
            coverage={item.value: 0 for item in SemanticFlowCoverage},
            capture_failures=0,
        ),
        machine={
            "eligible": 0,
            "issued": 0,
            "consumed": 0,
            "succeeded": 0,
            "failed": 0,
            "unknown": 0,
            "expired": 0,
            "revoked": 0,
            "race_lost": 0,
            "denied": 0,
        },
        actual_auto_approval=SemanticRatioV1(0, 0, None),
        review_metrics=SemanticReviewMetricsV1(0, 0, 0, None, 0, None),
    )

    assert status.to_dict()["schema_version"] == 3
    assert SemanticStatusV3.from_dict(status.to_dict()) == status

    inconsistent = status.to_dict()
    inconsistent["review_metrics"]["issued_review_rate"] = 0.0
    with pytest.raises(ValueError, match="zero denominator"):
        SemanticStatusV3.from_dict(inconsistent)


def test_custom_ontology_cannot_expand_the_frozen_auto_catalog() -> None:
    with pytest.raises(ValueError, match="catalog v1"):
        SemanticActionDefinition(
            action_id="shell.run",
            domain=SemanticDomain.SHELL,
            authority_operation="shell.run",
            allowed_rights=("execute",),
            risk=AuthorityRisk.LOW,
            auto_approval_eligible=True,
            requires_data_flow_egress=False,
        )

    with pytest.raises(ValueError, match="exactly catalog v1"):
        ActionOntology(
            actions=(
                SemanticActionDefinition(
                    action_id="filesystem.read",
                    domain=SemanticDomain.FILESYSTEM,
                    authority_operation="filesystem.read",
                    allowed_rights=("read",),
                    risk=AuthorityRisk.LOW,
                    auto_approval_eligible=True,
                    requires_data_flow_egress=False,
                ),
            )
        )


def test_unsupported_exact_action_remains_well_formed_and_with_human() -> None:
    context = {
        "pid": "pid-1",
        "authority_operation": "custom.inspect",
        "resource": "custom:workspace:item-1",
        "right": "read",
    }
    binding = {
        "effect_id": "eff_custom",
        "canonical_args_hash": canonical_effect_hash(context),
        "target_state_version": None,
    }
    capability = {
        "subject": "pid-1",
        "resource": "custom:workspace:item-1",
        "rights": ["read"],
        "constraints": {APPROVAL_BINDING_KEY: binding},
    }

    schema_valid, request_exact, binding_current = (
        SemanticManager._approval_binding_facts(
            "pid-1",
            "custom.inspect",
            "custom:workspace:item-1",
            ("read",),
            capability,
            context,
            binding,
        )
    )

    assert schema_valid
    assert request_exact
    assert binding_current

    decision = DeterministicApprovalBroker().decide(
        assessment=SemanticAssessment(
            status=SemanticAssessmentStatus.SUCCESS,
            confidence_bps=10_000,
            calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
        ),
        facts=AuthoritativeApprovalFacts(
            **{
                name: True
                for name in AuthoritativeApprovalFacts.__dataclass_fields__
            }
        ),
        policy_sha256=_A,
        candidate=SemanticApprovalCandidate(
            rule_id="custom-inspect",
            authority_operation="custom.inspect",
            resource="custom:workspace:item-1",
            rights=("read",),
            manifest_id="manifest-1",
            manifest_sha256=_B,
            policy_sha256=_A,
        ),
    )
    assert decision.outcome.value == "require_human"

    high_risk = DeterministicApprovalBroker().decide(
        assessment=SemanticAssessment(
            status=SemanticAssessmentStatus.SUCCESS,
            confidence_bps=10_000,
            calibration_bucket=SemanticCalibrationBucket.VERY_HIGH,
        ),
        facts=AuthoritativeApprovalFacts(
            **{
                name: True
                for name in AuthoritativeApprovalFacts.__dataclass_fields__
            }
        ),
        policy_sha256=_A,
        candidate=SemanticApprovalCandidate(
            rule_id="shell-run",
            authority_operation="shell.run",
            resource="shell:workspace",
            rights=("execute",),
            manifest_id="manifest-1",
            manifest_sha256=_B,
            policy_sha256=_A,
        ),
    )
    assert high_risk.outcome.value == "require_human"


def test_hard_deny_rule_allows_host_operation_but_not_wildcard_operation() -> None:
    rule = SemanticHardDenyRuleV1(
        rule_id="host-custom-deny",
        authority_operation="custom.inspect",
        resource="custom:workspace:item-1",
        rights=("read",),
    )
    assert rule.authority_operation == "custom.inspect"

    with pytest.raises(ValueError, match="exact dotted operation"):
        SemanticHardDenyRuleV1(
            rule_id="host-wildcard-deny",
            authority_operation="custom.*",
            resource="custom:workspace:item-1",
            rights=("read",),
        )
