from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_libos.config import (
    AgentLibOSConfig,
    DEFAULT_CONFIG,
    LLMDefaults,
    LLMProfile,
    SemanticDefaults,
)
from agent_libos.models import (
    AuthoritativeApprovalFacts,
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
    SEMANTIC_PROVIDER_RESPONSE_SCHEMA,
    SemanticApprovalCandidate,
    SemanticAssessment,
    SemanticAssessmentKind,
    SemanticAssessmentRequest,
    SemanticAssessmentStatus,
    SemanticCalibrationBucket,
    SemanticDataCategory,
    SemanticDataFinding,
    SemanticDataLocator,
    SemanticDomain,
    SemanticFinding,
    SemanticFindingSeverity,
    SemanticFindingSource,
    SemanticReasonCode,
    ShadowPolicyOutcome,
)
from agent_libos.semantic import (
    DEFAULT_ACTION_ONTOLOGY,
    DeterministicApprovalBroker,
    build_external_projection,
    conservative_label_suggestion,
    validate_monotonic_data_findings,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64


def _facts(value: bool = True) -> AuthoritativeApprovalFacts:
    return AuthoritativeApprovalFacts(
        **{name: value for name in AuthoritativeApprovalFacts.__dataclass_fields__}
    )


def _request(**updates: object) -> SemanticAssessmentRequest:
    values: dict[str, object] = {
        "kind": SemanticAssessmentKind.APPROVAL,
        "domain": SemanticDomain.FILESYSTEM,
        "action_id": "filesystem.read",
        "input_sha256": _A,
        "deadline_at": "2027-01-01T00:00:00+00:00",
        "data_labels": DataLabels(),
        "features": _facts(),
        "redacted_intent": "Read the quarterly report",
        "pid": "pid-1",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "effect_id": "effect-1",
        "manifest_id": "manifest-1",
        "manifest_sha256": _A,
        "policy_sha256": _B,
        "resource_sha256": _C,
        "args_sha256": _A,
        "state_sha256": _B,
        "source_refs_sha256": _C,
        "data_labels_sha256": _A,
        "sink_identity_sha256": _B,
        "tool_schema_sha256": _C,
        "provider_spec_sha256": _A,
    }
    values.update(updates)
    return SemanticAssessmentRequest(**values)  # type: ignore[arg-type]


def _assessment(**updates: object) -> SemanticAssessment:
    values: dict[str, object] = {
        "status": SemanticAssessmentStatus.SUCCESS,
        "findings": (),
        "data_findings": (),
        "confidence_bps": 9_500,
        "calibration_bucket": SemanticCalibrationBucket.HIGH,
        "ood": False,
        "abstain": False,
    }
    values.update(updates)
    return SemanticAssessment(**values)  # type: ignore[arg-type]


def _candidate(**updates: object) -> SemanticApprovalCandidate:
    values: dict[str, object] = {
        "rule_id": "workspace-read-v1",
        "authority_operation": "filesystem.read",
        "resource": "filesystem:workspace:reports/q1.txt",
        "rights": ("read",),
        "manifest_id": "manifest-1",
        "manifest_sha256": _A,
        "policy_sha256": _B,
    }
    values.update(updates)
    return SemanticApprovalCandidate(**values)  # type: ignore[arg-type]


def test_semantic_models_are_strict_payload_free_and_round_trip() -> None:
    finding = SemanticFinding(
        code=SemanticReasonCode.PROMPT_INJECTION,
        severity=SemanticFindingSeverity.HIGH,
        confidence_bps=9_999,
        evidence_sha256=_A,
        source=SemanticFindingSource.MODEL,
    )
    data_finding = SemanticDataFinding(
        category=SemanticDataCategory.CREDENTIAL,
        field="redacted_intent",
        span_start=4,
        span_end=12,
        sensitivity_floor=DataSensitivity.RESTRICTED,
        integrity_ceiling=DataIntegrity.UNTRUSTED,
        trust_ceiling=DataTrustLevel.UNTRUSTED,
        confidence_bps=10_000,
        evidence_sha256=_B,
    )
    assessment = _assessment(findings=(finding,), data_findings=(data_finding,))

    assert SemanticAssessment.from_dict(assessment.to_dict()) == assessment
    assert data_finding.field is SemanticDataLocator.REDACTED_INTENT
    assert _request().from_dict(_request().to_dict()) == _request()
    assert "permit" not in json.dumps(SEMANTIC_PROVIDER_RESPONSE_SCHEMA)
    with pytest.raises(ValueError, match="fields must be exactly"):
        SemanticAssessment.from_dict({**assessment.to_dict(), "decision": "allow"})
    with pytest.raises(ValueError, match="confidence_bps"):
        SemanticFinding(
            code=SemanticReasonCode.RISK_DETECTED,
            severity=SemanticFindingSeverity.LOW,
            confidence_bps=True,  # type: ignore[arg-type]
            evidence_sha256=_A,
            source=SemanticFindingSource.MODEL,
        )
    with pytest.raises(ValueError, match="span_end"):
        SemanticDataFinding(
            category=SemanticDataCategory.OTHER,
            field="redacted_intent",
            span_start=2,
            span_end=2,
            sensitivity_floor=DataSensitivity.NORMAL,
            integrity_ceiling=DataIntegrity.UNKNOWN,
            trust_ceiling=DataTrustLevel.UNKNOWN,
            confidence_bps=1,
            evidence_sha256=_A,
        )
    with pytest.raises(ValueError, match="coarse data finding locators"):
        SemanticDataFinding(
            category=SemanticDataCategory.OTHER,
            field="provider.result",
            span_start=0,
            span_end=1,
            sensitivity_floor=DataSensitivity.NORMAL,
            integrity_ceiling=DataIntegrity.UNKNOWN,
            trust_ceiling=DataTrustLevel.UNKNOWN,
            confidence_bps=1,
            evidence_sha256=_A,
        )
    with pytest.raises(ValueError, match="require a complete span"):
        SemanticDataFinding(
            category=SemanticDataCategory.OTHER,
            field="redacted_intent",
            span_start=None,
            span_end=None,
            sensitivity_floor=DataSensitivity.NORMAL,
            integrity_ceiling=DataIntegrity.UNKNOWN,
            trust_ceiling=DataTrustLevel.UNKNOWN,
            confidence_bps=1,
            evidence_sha256=_A,
        )
    with pytest.raises(ValueError, match="exceeds redacted-intent bounds"):
        SemanticDataFinding(
            category=SemanticDataCategory.OTHER,
            field="redacted_intent",
            span_start=0,
            span_end=2_001,
            sensitivity_floor=DataSensitivity.NORMAL,
            integrity_ceiling=DataIntegrity.UNKNOWN,
            trust_ceiling=DataTrustLevel.UNKNOWN,
            confidence_bps=1,
            evidence_sha256=_A,
        )
    with pytest.raises(ValueError, match="field locator"):
        SemanticDataFinding(
            category=SemanticDataCategory.OTHER,
            field="XcHJvamVjdGVkX2ludGVudF9zZW50aW5lbA",
            span_start=None,
            span_end=None,
            sensitivity_floor=DataSensitivity.NORMAL,
            integrity_ceiling=DataIntegrity.UNKNOWN,
            trust_ceiling=DataTrustLevel.UNKNOWN,
            confidence_bps=1,
            evidence_sha256=_A,
        )


def test_external_projection_redacts_or_drops_sensitive_intent_and_raw_ids() -> None:
    secret = "sk-test_12345678901234567890"
    projected = build_external_projection(
        _request(redacted_intent=f"Read /private/company/report.txt using token={secret}")
    )

    encoded = json.dumps(projected.payload, sort_keys=True)
    assert projected.metadata_only
    assert projected.dlp_matched
    assert secret not in encoded
    assert "/private/company/report.txt" not in encoded
    assert "pid-1" not in encoded
    assert projected.utf8_bytes <= 16_384

    restricted = build_external_projection(
        _request(
            data_labels=DataLabels(sensitivity=DataSensitivity.RESTRICTED),
            redacted_intent="otherwise harmless prose",
        )
    )
    assert restricted.metadata_only
    assert "redacted_intent" not in restricted.payload


@pytest.mark.parametrize(
    ("intent", "category", "code"),
    (
        (
            "Use ghp_12345678901234567890 for the request",
            "credential",
            "credential_material",
        ),
        (
            "Use password=semantic-credential-value",
            "credential",
            "credential_material",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\nQUJDREVGRw==\n-----END PRIVATE KEY-----",
            "credential",
            "credential_material",
        ),
        (
            "Inspect reports/payroll.csv",
            "business_secret",
            "sensitive_data",
        ),
    ),
)
def test_external_projection_freezes_payload_free_local_dlp_evidence(
    intent: str,
    category: str,
    code: str,
) -> None:
    projected = build_external_projection(_request(redacted_intent=intent))

    assert projected.metadata_only
    assert projected.dlp_matched
    assert projected.payload["projection_mode"] == "metadata_only"
    assert "redacted_intent" not in projected.payload
    assert intent not in json.dumps(projected.payload, sort_keys=True)
    assert projected.dlp_findings
    assert projected.data_flow_labels.sensitivity.value == (
        "secret" if category == "credential" else "confidential"
    )
    assert (
        projected.data_flow_labels.integrity
        == _request().data_labels.integrity
    )
    assert (
        projected.data_flow_labels.trust_level
        == _request().data_labels.trust_level
    )
    assert _request().data_labels.sensitivity.value == "normal"
    assert any(
        item.category.value == category and item.code.value == code
        for item in projected.dlp_findings
    )
    for item in projected.payload["dlp_findings"]:
        assert set(item) == {"category", "code", "evidence_sha256"}
        assert len(item["evidence_sha256"]) == 64


def test_action_ontology_is_closed_and_structurally_excludes_high_risk_actions() -> None:
    assert DEFAULT_ACTION_ONTOLOGY.resolve("filesystem.read").auto_approval_eligible
    assert DEFAULT_ACTION_ONTOLOGY.resolve("git.diff").auto_approval_eligible
    assert not DEFAULT_ACTION_ONTOLOGY.resolve("filesystem.read").requires_data_flow_egress
    assert not DEFAULT_ACTION_ONTOLOGY.resolve("git.read").requires_data_flow_egress
    assert not DEFAULT_ACTION_ONTOLOGY.resolve("git.diff").requires_data_flow_egress
    assert DEFAULT_ACTION_ONTOLOGY.resolve("filesystem.write").requires_data_flow_egress
    assert DEFAULT_ACTION_ONTOLOGY.resolve("shell.run").requires_data_flow_egress
    assert DEFAULT_ACTION_ONTOLOGY.resolve("mcp.call").requires_data_flow_egress
    assert not DEFAULT_ACTION_ONTOLOGY.resolve("shell.run").auto_approval_eligible
    assert not DEFAULT_ACTION_ONTOLOGY.resolve("mcp.call").auto_approval_eligible
    assert DEFAULT_ACTION_ONTOLOGY.resolve("invented.call") is None


def test_shadow_broker_requires_host_predicates_and_never_returns_a_permit() -> None:
    broker = DeterministicApprovalBroker()
    assessment = _assessment()
    candidate = _candidate()

    matched = broker.decide(
        assessment=assessment,
        facts=_facts(),
        policy_sha256=_B,
        candidate=candidate,
    )
    assert matched.outcome is ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE
    assert matched.missing_predicates == ()
    assert "permit" not in matched.to_dict()

    missing = broker.decide(
        assessment=assessment,
        facts=AuthoritativeApprovalFacts(),
        policy_sha256=_B,
        candidate=candidate,
    )
    assert missing.outcome is ShadowPolicyOutcome.REQUIRE_HUMAN
    assert SemanticReasonCode.MISSING_AUTHORITATIVE_PREDICATE in missing.reason_codes

    denied = broker.decide(
        assessment=assessment,
        facts=_facts(),
        policy_sha256=_B,
        hard_violations=(SemanticReasonCode.DATA_RELEASE,),
    )
    assert denied.outcome is ShadowPolicyOutcome.WOULD_DENY


@pytest.mark.parametrize(
    "assessment",
    (
        _assessment(status=SemanticAssessmentStatus.OOD, ood=True),
        _assessment(status=SemanticAssessmentStatus.ABSTAINED, abstain=True),
        _assessment(status=SemanticAssessmentStatus.TIMEOUT),
        _assessment(
            findings=(
                SemanticFinding(
                    code=SemanticReasonCode.RISK_DETECTED,
                    severity=SemanticFindingSeverity.LOW,
                    confidence_bps=1,
                    evidence_sha256=_A,
                    source=SemanticFindingSource.MODEL,
                ),
            )
        ),
    ),
)
def test_shadow_broker_fails_closed_on_uncertain_or_risky_assessment(
    assessment: SemanticAssessment,
) -> None:
    decision = DeterministicApprovalBroker().decide(
        assessment=assessment,
        facts=_facts(),
        policy_sha256=_B,
        candidate=_candidate(),
    )
    assert decision.outcome is ShadowPolicyOutcome.REQUIRE_HUMAN


@pytest.mark.parametrize(
    ("assessment", "candidate", "hard_violations", "expected"),
    [
        (
            _assessment(status=SemanticAssessmentStatus.OOD, ood=True),
            _candidate(),
            (SemanticReasonCode.DATA_RELEASE,),
            ShadowPolicyOutcome.WOULD_DENY,
        ),
        (
            _assessment(status=SemanticAssessmentStatus.PROVIDER_ERROR),
            _candidate(),
            (),
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            _assessment(status=SemanticAssessmentStatus.INVALID_SCHEMA),
            _candidate(),
            (),
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            _assessment(),
            _candidate(policy_sha256=_C),
            (),
            ShadowPolicyOutcome.REQUIRE_HUMAN,
        ),
        (
            _assessment(),
            _candidate(),
            (),
            ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE,
        ),
    ],
    ids=(
        "hard-violation-beats-ood",
        "provider-error",
        "malformed-response",
        "stale-policy",
        "exact-positive",
    ),
)
def test_shadow_broker_priority_is_closed_and_fail_closed(
    assessment: SemanticAssessment,
    candidate: SemanticApprovalCandidate,
    hard_violations: tuple[SemanticReasonCode, ...],
    expected: ShadowPolicyOutcome,
) -> None:
    decision = DeterministicApprovalBroker().decide(
        assessment=assessment,
        facts=_facts(),
        policy_sha256=_B,
        candidate=candidate,
        hard_violations=hard_violations,
    )

    assert decision.outcome is expected


@pytest.mark.parametrize(
    ("status", "ood", "abstain"),
    [
        (SemanticAssessmentStatus.SUCCESS, True, False),
        (SemanticAssessmentStatus.SUCCESS, False, True),
        (SemanticAssessmentStatus.OOD, False, False),
        (SemanticAssessmentStatus.ABSTAINED, False, False),
        (SemanticAssessmentStatus.OOD, True, True),
    ],
)
def test_semantic_assessment_requires_canonical_uncertainty_status_flags(
    status: SemanticAssessmentStatus,
    ood: bool,
    abstain: bool,
) -> None:
    with pytest.raises(ValueError, match="status and flag must match"):
        _assessment(status=status, ood=ood, abstain=abstain)


def test_data_finding_suggestion_is_monotonic_and_does_not_mutate_input() -> None:
    original = DataLabels(
        sensitivity=DataSensitivity.NORMAL,
        integrity=DataIntegrity.VERIFIED,
        trust_level=DataTrustLevel.TRUSTED,
        tenant="tenant-a",
    )
    finding = SemanticDataFinding(
        category=SemanticDataCategory.UNTRUSTED_CONTENT,
        field="provider.result",
        span_start=None,
        span_end=None,
        sensitivity_floor=DataSensitivity.CONFIDENTIAL,
        integrity_ceiling=DataIntegrity.UNTRUSTED,
        trust_ceiling=DataTrustLevel.UNTRUSTED,
        confidence_bps=9_000,
        evidence_sha256=_C,
    )

    suggested = conservative_label_suggestion(original, (finding,))

    assert suggested.sensitivity is DataSensitivity.CONFIDENTIAL
    assert suggested.integrity is DataIntegrity.UNTRUSTED
    assert suggested.trust_level is DataTrustLevel.UNTRUSTED
    assert suggested.tenant == "tenant-a"
    assert original == DataLabels(
        sensitivity=DataSensitivity.NORMAL,
        integrity=DataIntegrity.VERIFIED,
        trust_level=DataTrustLevel.TRUSTED,
        tenant="tenant-a",
    )

    declassifying = SemanticDataFinding(
        category=SemanticDataCategory.OTHER,
        field="provider.result",
        span_start=None,
        span_end=None,
        sensitivity_floor=DataSensitivity.PUBLIC,
        integrity_ceiling=DataIntegrity.VERIFIED,
        trust_ceiling=DataTrustLevel.TRUSTED,
        confidence_bps=1,
        evidence_sha256=_A,
    )
    with pytest.raises(ValueError, match="cannot declassify"):
        validate_monotonic_data_findings(original, (declassifying,))


def test_semantic_config_defaults_off_and_external_shadow_requires_safe_profile() -> None:
    assert DEFAULT_CONFIG.semantic == SemanticDefaults()
    assert DEFAULT_CONFIG.semantic.mode == "off"
    assert DEFAULT_CONFIG.semantic.adapter == "deterministic"
    AgentLibOSConfig(semantic=SemanticDefaults(mode="shadow", adapter="scripted"))

    with pytest.raises(ValueError, match="external_profile_id"):
        AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="external")
        )

    profiles = {
        "default": LLMProfile(),
        "semantic": LLMProfile(
            model="semantic-classifier-v1",
            api_mode="responses",
            timeout_s=15.0,
            max_retries=0,
            store=False,
            responses_previous_response_id=False,
            fallback_json_actions=False,
        ),
    }
    configured = AgentLibOSConfig(
        llm=LLMDefaults(profiles=profiles),
        semantic=SemanticDefaults(
            mode="shadow",
            adapter="external",
            external_profile_id="semantic",
        ),
    )
    assert configured.semantic.external_profile_id == "semantic"


def test_semantic_config_rejects_projection_ttl_shorter_than_job_lease() -> None:
    with pytest.raises(
        ValueError,
        match=r"semantic\.projection_ttl_s must be >= semantic\.job_lease_s",
    ):
        AgentLibOSConfig(
            semantic=SemanticDefaults(
                job_lease_s=60.0,
                projection_ttl_s=59,
            )
        )


def test_semantic_config_allows_projection_ttl_equal_to_job_lease() -> None:
    configured = AgentLibOSConfig(
        semantic=SemanticDefaults(
            job_lease_s=60.0,
            projection_ttl_s=60,
        )
    )

    assert configured.semantic.job_lease_s == 60.0
    assert configured.semantic.projection_ttl_s == 60


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_cache_key", "ambient-cache"),
        ("prompt_cache_retention", "24h"),
    ],
)
def test_semantic_external_config_rejects_ambient_prompt_cache_defaults(
    field: str,
    value: str,
) -> None:
    profiles = {
        "default": LLMProfile(),
        "semantic": LLMProfile(
            model="semantic-classifier-v1",
            api_mode="chat",
            timeout_s=5.0,
            max_retries=0,
            store=False,
            prompt_cache_key=None,
            prompt_cache_retention=None,
            responses_previous_response_id=False,
            fallback_json_actions=False,
        ),
    }
    llm = replace(LLMDefaults(profiles=profiles), **{field: value})

    with pytest.raises(ValueError, match="disable prompt caching"):
        AgentLibOSConfig(
            llm=llm,
            semantic=SemanticDefaults(
                mode="shadow",
                adapter="external",
                external_profile_id="semantic",
            ),
        )
