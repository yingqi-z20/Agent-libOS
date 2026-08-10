from __future__ import annotations

from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
    DeterministicDenyDecision,
    SemanticApprovalCandidate,
    SemanticApprovalCandidateSnapshotV1,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticControlStateV1,
    SemanticFindingSeverity,
    SemanticFlowCoverage,
    SemanticPolicyEpochV1,
    SemanticPredicate,
    SemanticReasonCode,
    SemanticRuntimeMode,
    ShadowPolicyDecision,
    ShadowPolicyOutcome,
)
from agent_libos.semantic.ontology import ActionOntology, DEFAULT_ACTION_ONTOLOGY


_STATUS_REASONS = {
    SemanticAssessmentStatus.SKIPPED_POLICY: SemanticReasonCode.ABSTAINED,
    SemanticAssessmentStatus.EGRESS_BLOCKED: SemanticReasonCode.EGRESS_BLOCKED,
    SemanticAssessmentStatus.TIMEOUT: SemanticReasonCode.TIMEOUT,
    SemanticAssessmentStatus.PROVIDER_ERROR: SemanticReasonCode.PROVIDER_ERROR,
    SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN: SemanticReasonCode.PROVIDER_OUTCOME_UNKNOWN,
    SemanticAssessmentStatus.INVALID_SCHEMA: SemanticReasonCode.SCHEMA_INVALID,
    SemanticAssessmentStatus.OOD: SemanticReasonCode.OUT_OF_DISTRIBUTION,
    SemanticAssessmentStatus.ABSTAINED: SemanticReasonCode.ABSTAINED,
    SemanticAssessmentStatus.STALE_INPUT: SemanticReasonCode.STALE_BINDING,
}
_RISK_SEVERITIES = {
    SemanticFindingSeverity.LOW,
    SemanticFindingSeverity.MEDIUM,
    SemanticFindingSeverity.HIGH,
    SemanticFindingSeverity.CRITICAL,
}


def _epoch_rule_matches(
    epoch: SemanticPolicyEpochV1,
    candidate: SemanticApprovalCandidate,
) -> bool:
    return any(
        rule.rule_id == candidate.rule_id
        and rule.authority_operation == candidate.authority_operation
        and (
            rule.resource == candidate.resource
            or (
                rule.resource.endswith("*")
                and candidate.resource.startswith(rule.resource[:-1])
            )
        )
        and set(candidate.rights).issubset(rule.rights)
        for rule in epoch.auto_approval_rules
    )


class DeterministicApprovalBroker:
    """Pure closed-world Shadow policy evaluator; it never grants authority."""

    def __init__(self, ontology: ActionOntology = DEFAULT_ACTION_ONTOLOGY):
        self._ontology = ontology

    def decide(
        self,
        *,
        assessment: SemanticAssessment,
        facts: AuthoritativeApprovalFacts,
        policy_sha256: str,
        candidate: (
            SemanticApprovalCandidate | SemanticApprovalCandidateSnapshotV1 | None
        ) = None,
        hard_violations: tuple[SemanticReasonCode, ...] = (),
    ) -> ShadowPolicyDecision:
        if not isinstance(assessment, SemanticAssessment):
            raise TypeError("assessment must be SemanticAssessment")
        if not isinstance(facts, AuthoritativeApprovalFacts):
            raise TypeError("facts must be AuthoritativeApprovalFacts")
        if candidate is not None and not isinstance(
            candidate,
            (SemanticApprovalCandidate, SemanticApprovalCandidateSnapshotV1),
        ):
            raise TypeError(
                "candidate must be exact or digest-only semantic approval evidence"
            )
        hard = _reason_tuple(hard_violations)
        assessment_sha256 = assessment.canonical_sha256()
        proven = tuple(predicate for predicate, value in facts.predicates() if value)
        missing = tuple(predicate for predicate, value in facts.predicates() if not value)
        manifest_sha256 = candidate.manifest_sha256 if candidate is not None else None

        if hard:
            return _decision(ShadowPolicyOutcome.WOULD_DENY, hard, candidate, proven, missing, policy_sha256, manifest_sha256, assessment_sha256)

        candidate_outcome = self._candidate_outcome(candidate, policy_sha256)
        if candidate_outcome is not None:
            outcome, reason = candidate_outcome
            return _decision(outcome, (reason,), candidate, proven, missing, policy_sha256, manifest_sha256, assessment_sha256)

        assessment_reasons = _assessment_reasons(assessment)
        if assessment_reasons:
            return _decision(ShadowPolicyOutcome.REQUIRE_HUMAN, assessment_reasons, candidate, proven, missing, policy_sha256, manifest_sha256, assessment_sha256)
        if candidate is None:
            return _decision(ShadowPolicyOutcome.REQUIRE_HUMAN, (SemanticReasonCode.CEILING_MISS,), None, proven, missing, policy_sha256, None, assessment_sha256)
        if missing:
            return _decision(ShadowPolicyOutcome.REQUIRE_HUMAN, (SemanticReasonCode.MISSING_AUTHORITATIVE_PREDICATE,), candidate, proven, missing, policy_sha256, manifest_sha256, assessment_sha256)
        return _decision(ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE, (SemanticReasonCode.POLICY_MATCH,), candidate, proven, (), policy_sha256, manifest_sha256, assessment_sha256)

    def _candidate_outcome(
        self,
        candidate: (
            SemanticApprovalCandidate | SemanticApprovalCandidateSnapshotV1 | None
        ),
        policy_sha256: str,
    ) -> tuple[ShadowPolicyOutcome, SemanticReasonCode] | None:
        if candidate is None:
            return None
        action = self._ontology.resolve(candidate.authority_operation)
        if action is None:
            return (
                ShadowPolicyOutcome.REQUIRE_HUMAN,
                SemanticReasonCode.UNSUPPORTED_ACTION,
            )
        if not action.auto_approval_eligible:
            return (
                ShadowPolicyOutcome.REQUIRE_HUMAN,
                SemanticReasonCode.HIGH_RISK_ACTION,
            )
        if not set(candidate.rights).issubset(action.allowed_rights):
            return (
                ShadowPolicyOutcome.REQUIRE_HUMAN,
                SemanticReasonCode.CONTROL_RIGHT,
            )
        if candidate.policy_sha256 != policy_sha256:
            return ShadowPolicyOutcome.REQUIRE_HUMAN, SemanticReasonCode.STALE_POLICY
        return None

    def decide_canary(
        self,
        *,
        assessment: SemanticAssessment,
        facts: AuthoritativeApprovalFacts,
        policy_sha256: str,
        candidate: SemanticApprovalCandidate | None,
        epoch: SemanticPolicyEpochV1,
        control: SemanticControlStateV1,
        tenant_bucket_sha256: str,
        flow_coverage: SemanticFlowCoverage,
        classifier_profile_sha256: str,
        classifier_model_sha256: str,
    ) -> ShadowPolicyDecision:
        """Evaluate all canary predicates without issuing authority.

        The classifier can only turn a Host-qualified candidate into
        ``require_human``.  It cannot repair a missing Host predicate, select a
        policy rule, or produce an authority-bearing result.
        """

        if not isinstance(epoch, SemanticPolicyEpochV1):
            raise TypeError("epoch must be SemanticPolicyEpochV1")
        if not isinstance(control, SemanticControlStateV1):
            raise TypeError("control must be SemanticControlStateV1")
        if candidate is not None and not isinstance(
            candidate, SemanticApprovalCandidate
        ):
            raise TypeError(
                "canary evaluation requires an exact SemanticApprovalCandidate"
            )
        selected_coverage = _flow_coverage(flow_coverage)
        base = self.decide(
            assessment=assessment,
            facts=facts,
            policy_sha256=policy_sha256,
            candidate=candidate,
        )
        if base.outcome is not ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE:
            return base
        assert candidate is not None
        reason = (
            _canary_assessment_reason(assessment, epoch)
            or _canary_control_reason(control, epoch)
            or _canary_policy_reason(
                epoch=epoch,
                candidate=candidate,
                tenant_bucket_sha256=tenant_bucket_sha256,
                flow_coverage=selected_coverage,
                classifier_profile_sha256=classifier_profile_sha256,
                classifier_model_sha256=classifier_model_sha256,
            )
        )
        if reason is None:
            return base
        return _decision(
            ShadowPolicyOutcome.REQUIRE_HUMAN,
            (reason,),
            candidate,
            base.proven_predicates,
            base.missing_predicates,
            policy_sha256,
            candidate.manifest_sha256,
            assessment.canonical_sha256(),
        )

    @staticmethod
    def deterministic_deny(
        *,
        request_id: str,
        request_revision: int,
        pid: str,
        effect_id: str,
        reason_codes: tuple[SemanticReasonCode, ...],
        policy_sha256: str,
        evidence_sha256: str,
        decided_at: str,
    ) -> DeterministicDenyDecision | None:
        """Create a closed-set Host deny proof, or ``None`` for no hard deny."""

        if not isinstance(reason_codes, tuple):
            raise TypeError("deterministic deny reason_codes must be a tuple")
        if not reason_codes:
            return None
        return DeterministicDenyDecision(
            request_id=request_id,
            request_revision=request_revision,
            pid=pid,
            effect_id=effect_id,
            reason_codes=reason_codes,
            policy_sha256=policy_sha256,
            evidence_sha256=evidence_sha256,
            decided_at=decided_at,
        )


def _flow_coverage(value: SemanticFlowCoverage) -> SemanticFlowCoverage:
    try:
        return (
            value
            if isinstance(value, SemanticFlowCoverage)
            else SemanticFlowCoverage(value)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported semantic flow coverage") from exc


def _canary_assessment_reason(
    assessment: SemanticAssessment,
    epoch: SemanticPolicyEpochV1,
) -> SemanticReasonCode | None:
    if assessment.findings or assessment.data_findings:
        return SemanticReasonCode.RISK_DETECTED
    if assessment.confidence_bps < epoch.minimum_confidence_bps:
        return SemanticReasonCode.CONFIDENCE_TOO_LOW
    if assessment.calibration_bucket is not epoch.required_calibration_bucket:
        return SemanticReasonCode.CALIBRATION_TOO_LOW
    return None


def _canary_control_reason(
    control: SemanticControlStateV1,
    epoch: SemanticPolicyEpochV1,
) -> SemanticReasonCode | None:
    if control.mode is not SemanticRuntimeMode.CANARY_AUTO:
        return SemanticReasonCode.CONTROL_DISABLED
    if control.tripped:
        return SemanticReasonCode.CONTROL_TRIPPED
    if (
        control.active_epoch_id != epoch.epoch_id
        or control.active_policy_sha256 != epoch.canonical_sha256()
        or control.generation != epoch.generation
    ):
        return SemanticReasonCode.STALE_POLICY
    return None


def _canary_policy_reason(
    *,
    epoch: SemanticPolicyEpochV1,
    candidate: SemanticApprovalCandidate,
    tenant_bucket_sha256: str,
    flow_coverage: SemanticFlowCoverage,
    classifier_profile_sha256: str,
    classifier_model_sha256: str,
) -> SemanticReasonCode | None:
    if tenant_bucket_sha256 not in epoch.tenant_bucket_sha256s:
        return SemanticReasonCode.TENANT_NOT_ALLOWED
    if flow_coverage is not SemanticFlowCoverage.COMPLETE:
        return SemanticReasonCode.FLOW_COVERAGE_INCOMPLETE
    if not _epoch_rule_matches(epoch, candidate):
        return SemanticReasonCode.CEILING_MISS
    if (
        epoch.classifier_profile_id is None
        or epoch.classifier_profile_sha256 is None
        or epoch.classifier_model_sha256 is None
    ):
        return SemanticReasonCode.DIGEST_DRIFT
    if (
        classifier_profile_sha256 != epoch.classifier_profile_sha256
        or classifier_model_sha256 != epoch.classifier_model_sha256
    ):
        return SemanticReasonCode.DIGEST_DRIFT
    return None


def _reason_tuple(values: tuple[SemanticReasonCode, ...]) -> tuple[SemanticReasonCode, ...]:
    if not isinstance(values, tuple):
        raise TypeError("hard_violations must be a tuple")
    result: list[SemanticReasonCode] = []
    for value in values:
        selected = value if isinstance(value, SemanticReasonCode) else SemanticReasonCode(value)
        if selected not in result:
            result.append(selected)
    return tuple(result)


def _assessment_reasons(
    assessment: SemanticAssessment,
) -> tuple[SemanticReasonCode, ...]:
    if assessment.status is not SemanticAssessmentStatus.SUCCESS:
        return (_STATUS_REASONS[assessment.status],)
    if assessment.ood:
        return (SemanticReasonCode.OUT_OF_DISTRIBUTION,)
    if assessment.abstain:
        return (SemanticReasonCode.ABSTAINED,)
    risks = tuple(
        dict.fromkeys(
            finding.code
            for finding in assessment.findings
            if finding.severity in _RISK_SEVERITIES
        )
    )
    if risks:
        return risks
    if assessment.data_findings:
        return (SemanticReasonCode.RISK_DETECTED,)
    return ()


def _decision(
    outcome: ShadowPolicyOutcome,
    reasons: tuple[SemanticReasonCode, ...],
    candidate: SemanticApprovalCandidate | SemanticApprovalCandidateSnapshotV1 | None,
    proven: tuple[SemanticPredicate, ...],
    missing: tuple[SemanticPredicate, ...],
    policy_sha256: str,
    manifest_sha256: str | None,
    assessment_sha256: str,
) -> ShadowPolicyDecision:
    return ShadowPolicyDecision(
        outcome=outcome,
        reason_codes=reasons,
        matched_rule_id=candidate.rule_id if candidate is not None else None,
        proven_predicates=proven,
        missing_predicates=missing,
        policy_sha256=policy_sha256,
        manifest_sha256=manifest_sha256,
        assessment_sha256=assessment_sha256,
    )


__all__ = ["DeterministicApprovalBroker"]
