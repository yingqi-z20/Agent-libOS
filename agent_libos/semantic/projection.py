from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from agent_libos.models import DataLabels, DataSensitivity
from agent_libos.models.data_flow import sensitivity_rank
from agent_libos.models.semantic import (
    SEMANTIC_REDACTED_INTENT_MAX_CHARS,
    SemanticAssessmentRequest,
    SemanticDataCategory,
    SemanticReasonCode,
)


_CREDENTIAL_PATTERNS = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?P<key_label>(?:[A-Z0-9]+ )*PRIVATE KEY)-----.*?"
            r"-----END (?P=key_label)-----",
            re.DOTALL,
        ),
    ),
    (
        "credential_token",
        re.compile(
            r"\b(?:sk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b",
            re.IGNORECASE,
        ),
    ),
    ("credential_token", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "credential_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
    ),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:api[_-]?key|token|secret|password|passwd|pwd)"
            r"\s*[:=]\s*[^\s,;]+",
            re.IGNORECASE,
        ),
    ),
)
_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])(?:/[^\s'\"`]+){2,}"),
    re.compile(r"(?<![A-Za-z0-9])(?:~[/\\]|[A-Za-z]:[\\/])[^\s'\"`]+"),
    # Treat every path-like token conservatively, including relative paths,
    # parent traversal, dot-directories, Windows separators, and URLs.
    re.compile(r"(?<![A-Za-z0-9])[^\s'\"`]*[/\\][^\s'\"`]*"),
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOCAL_DLP_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RedactedIntent:
    text: str
    input_sha256: str
    output_sha256: str
    dlp_matched: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class LocalDlpFinding:
    """Payload-free evidence frozen by the Host-owned local DLP detector."""

    category: SemanticDataCategory
    code: SemanticReasonCode
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", SemanticDataCategory(self.category))
        object.__setattr__(self, "code", SemanticReasonCode(self.code))
        if re.fullmatch(r"[0-9a-f]{64}", self.evidence_sha256) is None:
            raise ValueError("local DLP evidence must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category.value,
            "code": self.code.value,
            "evidence_sha256": self.evidence_sha256,
        }


class LocalDlpAccumulator:
    """Incremental Host DLP detector that retains no scanned text."""

    __slots__ = ("_input_sha256", "_matched_detectors", "_remaining_bytes")

    def __init__(self, *, input_sha256: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
            raise ValueError("local DLP input identity must be a lowercase SHA-256")
        self._input_sha256 = input_sha256
        self._matched_detectors: list[
            tuple[str, SemanticDataCategory, SemanticReasonCode]
        ] = []
        self._remaining_bytes = _LOCAL_DLP_MAX_BYTES

    def scan(self, value: str | bytes) -> None:
        if type(value) is str:
            raw = value.encode("utf-8")
            selected = value
        elif type(value) is bytes:
            raw = value
            selected = value.decode("utf-8", errors="replace")
        else:
            raise TypeError("local DLP scans only exact strings or bytes")
        self._remaining_bytes -= len(raw)
        if self._remaining_bytes < 0:
            raise ValueError("local DLP input exceeds its bounded byte budget")
        for detector, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(selected) is not None:
                self._remember(
                    detector,
                    SemanticDataCategory.CREDENTIAL,
                    SemanticReasonCode.CREDENTIAL_MATERIAL,
                )
        if any(pattern.search(selected) is not None for pattern in _PATH_PATTERNS):
            self._remember(
                "path",
                SemanticDataCategory.BUSINESS_SECRET,
                SemanticReasonCode.SENSITIVE_DATA,
            )

    @property
    def findings(self) -> tuple[LocalDlpFinding, ...]:
        return tuple(
            LocalDlpFinding(
                category=category,
                code=code,
                evidence_sha256=hashlib.sha256(
                    _canonical_json(
                        {
                            "schema_version": 1,
                            "detector": detector,
                            "input_sha256": self._input_sha256,
                        }
                    )
                ).hexdigest(),
            )
            for detector, category, code in self._matched_detectors
        )

    def _remember(
        self,
        detector: str,
        category: SemanticDataCategory,
        code: SemanticReasonCode,
    ) -> None:
        item = (detector, category, code)
        if item not in self._matched_detectors:
            self._matched_detectors.append(item)


@dataclass(frozen=True, slots=True)
class SemanticExternalProjection:
    payload: dict[str, Any]
    projection_sha256: str
    utf8_bytes: int
    metadata_only: bool
    dlp_matched: bool
    dlp_findings: tuple[LocalDlpFinding, ...]
    data_flow_labels: DataLabels


def redact_intent(
    value: str,
    *,
    max_chars: int = SEMANTIC_REDACTED_INTENT_MAX_CHARS,
) -> RedactedIntent:
    if type(value) is not str:
        raise TypeError("semantic intent must be a string")
    if (
        type(max_chars) is not int
        or max_chars <= 0
        or max_chars > SEMANTIC_REDACTED_INTENT_MAX_CHARS
    ):
        raise ValueError("semantic intent max_chars must be from 1 through 2000")
    original_sha256 = hashlib.sha256(value.encode("utf-8")).hexdigest()
    selected = _CONTROL_RE.sub(" ", value)
    matched = False
    for _detector, pattern in _CREDENTIAL_PATTERNS:
        selected, count = pattern.subn("[REDACTED]", selected)
        matched = matched or count > 0
    for pattern in _PATH_PATTERNS:
        selected, count = pattern.subn("[REDACTED]", selected)
        matched = matched or count > 0
    selected = " ".join(selected.split())
    truncated = len(selected) > max_chars
    if truncated:
        selected = selected[:max_chars]
    return RedactedIntent(
        text=selected,
        input_sha256=original_sha256,
        output_sha256=hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        dlp_matched=matched,
        truncated=truncated,
    )


def build_external_projection(
    request: SemanticAssessmentRequest,
    *,
    labels: DataLabels | None = None,
    intent_max_chars: int = SEMANTIC_REDACTED_INTENT_MAX_CHARS,
    projection_max_bytes: int = 16_384,
) -> SemanticExternalProjection:
    if not isinstance(request, SemanticAssessmentRequest):
        raise TypeError("semantic projection requires SemanticAssessmentRequest")
    selected_labels = request.data_labels if labels is None else labels
    if not isinstance(selected_labels, DataLabels):
        raise TypeError("semantic projection labels must be DataLabels")
    if type(projection_max_bytes) is not int or not 512 <= projection_max_bytes <= 16_384:
        raise ValueError("semantic projection_max_bytes must be from 512 through 16384")

    redacted = redact_intent(request.redacted_intent or "", max_chars=intent_max_chars)
    dlp_findings = _local_dlp_findings(
        request.redacted_intent or "",
        input_sha256=request.input_sha256,
    )
    metadata_only = (
        selected_labels.sensitivity not in {DataSensitivity.PUBLIC, DataSensitivity.NORMAL}
        or selected_labels.is_mixed_identity
        or redacted.dlp_matched
        or not redacted.text
    )
    digests = {
        name: value
        for name, value in (
            ("manifest_sha256", request.manifest_sha256),
            ("policy_sha256", request.policy_sha256),
            ("resource_sha256", request.resource_sha256),
            ("args_sha256", request.args_sha256),
            ("state_sha256", request.state_sha256),
            ("source_refs_sha256", request.source_refs_sha256),
            ("data_labels_sha256", request.data_labels_sha256),
            ("sink_identity_sha256", request.sink_identity_sha256),
            ("tool_schema_sha256", request.tool_schema_sha256),
            ("provider_spec_sha256", request.provider_spec_sha256),
        )
        if value is not None
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "projection_mode": "metadata_only" if metadata_only else "redacted",
        "kind": request.kind.value,
        "domain": request.domain.value,
        "action_id": request.action_id,
        "input_sha256": request.input_sha256,
        "digests": digests,
        "features": request.features.to_dict(),
        "labels": {
            "sensitivity": selected_labels.sensitivity.value,
            "integrity": selected_labels.integrity.value,
            "trust_level": selected_labels.trust_level.value,
        },
        "dlp_findings": [item.to_dict() for item in dlp_findings],
    }
    if not metadata_only and redacted.text:
        payload["redacted_intent"] = redacted.text
        payload["redacted_intent_sha256"] = redacted.output_sha256
        payload["redacted_intent_truncated"] = redacted.truncated
    encoded = _canonical_json(payload)
    if len(encoded) > projection_max_bytes and "redacted_intent" in payload:
        payload.pop("redacted_intent", None)
        payload.pop("redacted_intent_sha256", None)
        payload.pop("redacted_intent_truncated", None)
        payload["projection_mode"] = "metadata_only"
        metadata_only = True
        encoded = _canonical_json(payload)
    if len(encoded) > projection_max_bytes:
        raise ValueError("semantic metadata projection exceeds the hard byte limit")
    return SemanticExternalProjection(
        payload=payload,
        projection_sha256=hashlib.sha256(encoded).hexdigest(),
        utf8_bytes=len(encoded),
        metadata_only=metadata_only,
        dlp_matched=redacted.dlp_matched,
        dlp_findings=dlp_findings,
        data_flow_labels=_local_dlp_data_flow_labels(
            selected_labels,
            dlp_findings,
        ),
    )


def _local_dlp_data_flow_labels(
    labels: DataLabels,
    findings: tuple[LocalDlpFinding, ...],
) -> DataLabels:
    sensitivity = labels.sensitivity
    minimum_by_category = {
        SemanticDataCategory.CREDENTIAL: DataSensitivity.SECRET,
        SemanticDataCategory.BUSINESS_SECRET: DataSensitivity.CONFIDENTIAL,
    }
    for finding in findings:
        minimum = minimum_by_category[finding.category]
        if sensitivity_rank(minimum) > sensitivity_rank(sensitivity):
            sensitivity = minimum
    return DataLabels(
        sensitivity=sensitivity,
        integrity=labels.integrity,
        trust_level=labels.trust_level,
        origin=labels.origin,
        tenant=labels.tenant,
        principal=labels.principal,
        declassification_authority=labels.declassification_authority,
    )


def _local_dlp_findings(
    value: str,
    *,
    input_sha256: str,
) -> tuple[LocalDlpFinding, ...]:
    """Return one bounded finding per detector class without matched text."""

    detector = LocalDlpAccumulator(input_sha256=input_sha256)
    detector.scan(value)
    return detector.findings


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


__all__ = [
    "LocalDlpFinding",
    "LocalDlpAccumulator",
    "RedactedIntent",
    "SemanticExternalProjection",
    "build_external_projection",
    "redact_intent",
]
