from __future__ import annotations

from agent_libos.models import DataIntegrity, DataLabels, DataSensitivity, DataTrustLevel
from agent_libos.models.data_flow import integrity_rank, sensitivity_rank
from agent_libos.models.semantic import SemanticDataFinding


_TRUST_ORDER = (
    DataTrustLevel.UNTRUSTED,
    DataTrustLevel.UNKNOWN,
    DataTrustLevel.USER_ASSERTED,
    DataTrustLevel.VERIFIED,
    DataTrustLevel.TRUSTED,
)


def conservative_label_suggestion(
    labels: DataLabels,
    findings: tuple[SemanticDataFinding, ...],
) -> DataLabels:
    """Return a suggestion that can only tighten labels; callers must not write it back."""

    if not isinstance(labels, DataLabels):
        raise TypeError("labels must be DataLabels")
    if not isinstance(findings, tuple) or any(not isinstance(item, SemanticDataFinding) for item in findings):
        raise TypeError("findings must be a tuple of SemanticDataFinding")
    validate_monotonic_data_findings(labels, findings)
    sensitivity = labels.sensitivity
    integrity = labels.integrity
    trust = labels.trust_level
    for finding in findings:
        if sensitivity_rank(finding.sensitivity_floor) > sensitivity_rank(sensitivity):
            sensitivity = finding.sensitivity_floor
        if integrity_rank(finding.integrity_ceiling) < integrity_rank(integrity):
            integrity = finding.integrity_ceiling
        if _TRUST_ORDER.index(finding.trust_ceiling) < _TRUST_ORDER.index(trust):
            trust = finding.trust_ceiling
    return DataLabels(
        sensitivity=sensitivity,
        trust_level=trust,
        integrity=integrity,
        origin=labels.origin,
        tenant=labels.tenant,
        principal=labels.principal,
        declassification_authority=labels.declassification_authority,
    )


def validate_monotonic_data_findings(
    labels: DataLabels,
    findings: tuple[SemanticDataFinding, ...],
) -> None:
    """Reject semantic output that attempts declassification or endorsement."""

    if not isinstance(labels, DataLabels):
        raise TypeError("labels must be DataLabels")
    if not isinstance(findings, tuple) or any(not isinstance(item, SemanticDataFinding) for item in findings):
        raise TypeError("findings must be a tuple of SemanticDataFinding")
    for finding in findings:
        if sensitivity_rank(finding.sensitivity_floor) < sensitivity_rank(labels.sensitivity):
            raise ValueError("semantic finding sensitivity_floor cannot declassify")
        if integrity_rank(finding.integrity_ceiling) > integrity_rank(labels.integrity):
            raise ValueError("semantic finding integrity_ceiling cannot endorse")
        if _TRUST_ORDER.index(finding.trust_ceiling) > _TRUST_ORDER.index(labels.trust_level):
            raise ValueError("semantic finding trust_ceiling cannot endorse")


__all__ = ["conservative_label_suggestion", "validate_monotonic_data_findings"]
