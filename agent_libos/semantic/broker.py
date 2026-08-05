from __future__ import annotations

from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
    SemanticApprovalCandidate,
    SemanticAssessment,
    SemanticAssessmentStatus,
    SemanticFindingSeverity,
    SemanticPredicate,
    SemanticReasonCode,
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
        candidate: SemanticApprovalCandidate | None = None,
        hard_violations: tuple[SemanticReasonCode, ...] = (),
    ) -> ShadowPolicyDecision:
        if not isinstance(assessment, SemanticAssessment):
            raise TypeError("assessment must be SemanticAssessment")
        if not isinstance(facts, AuthoritativeApprovalFacts):
            raise TypeError("facts must be AuthoritativeApprovalFacts")
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
        candidate: SemanticApprovalCandidate | None,
        policy_sha256: str,
    ) -> tuple[ShadowPolicyOutcome, SemanticReasonCode] | None:
        if candidate is None:
            return None
        action = self._ontology.resolve(candidate.authority_operation)
        if action is None:
            return ShadowPolicyOutcome.WOULD_DENY, SemanticReasonCode.UNSUPPORTED_ACTION
        if not action.auto_approval_eligible:
            return ShadowPolicyOutcome.WOULD_DENY, SemanticReasonCode.HIGH_RISK_ACTION
        if not set(candidate.rights).issubset(action.allowed_rights):
            return ShadowPolicyOutcome.WOULD_DENY, SemanticReasonCode.CONTROL_RIGHT
        if candidate.policy_sha256 != policy_sha256:
            return ShadowPolicyOutcome.REQUIRE_HUMAN, SemanticReasonCode.STALE_POLICY
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
    candidate: SemanticApprovalCandidate | None,
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
