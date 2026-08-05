from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, TypeVar

from agent_libos.models.base import StrEnum
from agent_libos.models.data_flow import (
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
)


SEMANTIC_SCHEMA_VERSION = 1
SEMANTIC_REDACTED_INTENT_MAX_CHARS = 2_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_E = TypeVar("_E", bound=StrEnum)


class SemanticAssessmentKind(StrEnum):
    APPROVAL = "approval"
    ROOT_GOAL = "root_goal"
    PROVIDER_INGRESS = "provider_ingress"


class SemanticDomain(StrEnum):
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    GIT = "git"
    JSONRPC = "jsonrpc"
    MCP = "mcp"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class SemanticAssessmentStatus(StrEnum):
    SUCCESS = "success"
    SKIPPED_POLICY = "skipped_policy"
    EGRESS_BLOCKED = "egress_blocked"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    INVALID_SCHEMA = "invalid_schema"
    OOD = "ood"
    ABSTAINED = "abstained"
    STALE_INPUT = "stale_input"


class SemanticFindingSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SemanticFindingSource(StrEnum):
    MODEL = "model"
    DETERMINISTIC = "deterministic"
    HOST = "host"


class SemanticReasonCode(StrEnum):
    POLICY_MATCH = "policy_match"
    HARD_POLICY_VIOLATION = "hard_policy_violation"
    MALFORMED_REQUEST = "malformed_request"
    STALE_BINDING = "stale_binding"
    STALE_MANIFEST = "stale_manifest"
    STALE_POLICY = "stale_policy"
    UNSUPPORTED_ACTION = "unsupported_action"
    HIGH_RISK_ACTION = "high_risk_action"
    CONTROL_RIGHT = "control_right"
    DATA_RELEASE = "data_release"
    CEILING_MISS = "ceiling_miss"
    MISSING_AUTHORITATIVE_PREDICATE = "missing_authoritative_predicate"
    SCHEMA_INVALID = "schema_invalid"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    TIMEOUT = "timeout"
    EGRESS_BLOCKED = "egress_blocked"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    ABSTAINED = "abstained"
    RISK_DETECTED = "risk_detected"
    SENSITIVE_DATA = "sensitive_data"
    CREDENTIAL_MATERIAL = "credential_material"
    PROMPT_INJECTION = "prompt_injection"
    MIXED_IDENTITY = "mixed_identity"
    LOW_INTEGRITY = "low_integrity"


class SemanticCalibrationBucket(StrEnum):
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SemanticDataCategory(StrEnum):
    CREDENTIAL = "credential"
    PERSONAL = "personal"
    FINANCIAL = "financial"
    HEALTH = "health"
    LEGAL = "legal"
    SOURCE_CODE = "source_code"
    BUSINESS_SECRET = "business_secret"
    INSTRUCTION_ATTACK = "instruction_attack"
    UNTRUSTED_CONTENT = "untrusted_content"
    OTHER = "other"


class SemanticDataLocator(StrEnum):
    APPROVAL_REQUEST = "approval.request"
    ROOT_GOAL = "root_goal"
    PROVIDER_RESULT = "provider.result"
    REDACTED_INTENT = "redacted_intent"


class ShadowPolicyOutcome(StrEnum):
    WOULD_ISSUE_EXACT_ONCE = "would_issue_exact_once"
    REQUIRE_HUMAN = "require_human"
    WOULD_DENY = "would_deny"


class SemanticPredicate(StrEnum):
    SCHEMA_VALID = "schema_valid"
    EXACT_EXTERNAL_OPERATION = "exact_external_operation"
    BINDING_CURRENT = "binding_current"
    MANIFEST_CURRENT = "manifest_current"
    POLICY_CURRENT = "policy_current"
    ACTION_KNOWN = "action_known"
    ACTION_AUTO_ELIGIBLE = "action_auto_eligible"
    LOW_RISK = "low_risk"
    RESOURCE_EXACT = "resource_exact"
    SINGLE_NON_CONTROL_RIGHT = "single_non_control_right"
    CEILING_MATCHED = "ceiling_matched"
    DATA_FLOW_ALLOWED = "data_flow_allowed"
    PROFILE_PINNED = "profile_pinned"


@dataclass(frozen=True, slots=True)
class SemanticFinding:
    code: SemanticReasonCode
    severity: SemanticFindingSeverity
    confidence_bps: int
    evidence_sha256: str
    source: SemanticFindingSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(SemanticReasonCode, self.code, "finding code"))
        object.__setattr__(self, "severity", _enum(SemanticFindingSeverity, self.severity, "finding severity"))
        object.__setattr__(self, "source", _enum(SemanticFindingSource, self.source, "finding source"))
        _confidence(self.confidence_bps)
        _sha256("finding evidence_sha256", self.evidence_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "confidence_bps": self.confidence_bps,
            "evidence_sha256": self.evidence_sha256,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticFinding:
        _exact(value, {"code", "severity", "confidence_bps", "evidence_sha256", "source"}, "semantic finding")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticDataFinding:
    category: SemanticDataCategory
    field: SemanticDataLocator
    span_start: int | None
    span_end: int | None
    sensitivity_floor: DataSensitivity
    integrity_ceiling: DataIntegrity
    trust_ceiling: DataTrustLevel
    confidence_bps: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _enum(SemanticDataCategory, self.category, "data finding category"))
        object.__setattr__(self, "field", _enum(SemanticDataLocator, self.field, "data finding field locator"))
        object.__setattr__(self, "sensitivity_floor", _enum(DataSensitivity, self.sensitivity_floor, "sensitivity floor"))
        object.__setattr__(self, "integrity_ceiling", _enum(DataIntegrity, self.integrity_ceiling, "integrity ceiling"))
        object.__setattr__(self, "trust_ceiling", _enum(DataTrustLevel, self.trust_ceiling, "trust ceiling"))
        if self.field is SemanticDataLocator.REDACTED_INTENT:
            if self.span_start is None or self.span_end is None:
                raise ValueError(
                    "redacted-intent data findings require a complete span"
                )
            _nonnegative_int("data finding span_start", self.span_start)
            _nonnegative_int("data finding span_end", self.span_end)
            if self.span_end <= self.span_start:
                raise ValueError("data finding span_end must be greater than span_start")
            if self.span_end > SEMANTIC_REDACTED_INTENT_MAX_CHARS:
                raise ValueError("data finding span exceeds redacted-intent bounds")
        elif self.span_start is not None or self.span_end is not None:
            raise ValueError("coarse data finding locators must not include spans")
        _confidence(self.confidence_bps)
        _sha256("data finding evidence_sha256", self.evidence_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "field": self.field.value,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "sensitivity_floor": self.sensitivity_floor.value,
            "integrity_ceiling": self.integrity_ceiling.value,
            "trust_ceiling": self.trust_ceiling.value,
            "confidence_bps": self.confidence_bps,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticDataFinding:
        _exact(value, {"category", "field", "span_start", "span_end", "sensitivity_floor", "integrity_ceiling", "trust_ceiling", "confidence_bps", "evidence_sha256"}, "semantic data finding")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AuthoritativeApprovalFacts:
    schema_valid: bool = False
    request_is_exact_external_operation: bool = False
    binding_current: bool = False
    manifest_current: bool = False
    policy_current: bool = False
    action_known: bool = False
    action_auto_eligible: bool = False
    low_risk: bool = False
    resource_exact: bool = False
    single_non_control_right: bool = False
    ceiling_matched: bool = False
    data_flow_allowed: bool = False
    profile_pinned: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"authoritative fact {name} must be a boolean")

    def to_dict(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuthoritativeApprovalFacts:
        keys = set(cls.__dataclass_fields__)
        _exact(value, keys, "authoritative approval facts")
        return cls(**dict(value))

    def predicates(self) -> tuple[tuple[SemanticPredicate, bool], ...]:
        return (
            (SemanticPredicate.SCHEMA_VALID, self.schema_valid),
            (SemanticPredicate.EXACT_EXTERNAL_OPERATION, self.request_is_exact_external_operation),
            (SemanticPredicate.BINDING_CURRENT, self.binding_current),
            (SemanticPredicate.MANIFEST_CURRENT, self.manifest_current),
            (SemanticPredicate.POLICY_CURRENT, self.policy_current),
            (SemanticPredicate.ACTION_KNOWN, self.action_known),
            (SemanticPredicate.ACTION_AUTO_ELIGIBLE, self.action_auto_eligible),
            (SemanticPredicate.LOW_RISK, self.low_risk),
            (SemanticPredicate.RESOURCE_EXACT, self.resource_exact),
            (SemanticPredicate.SINGLE_NON_CONTROL_RIGHT, self.single_non_control_right),
            (SemanticPredicate.CEILING_MATCHED, self.ceiling_matched),
            (SemanticPredicate.DATA_FLOW_ALLOWED, self.data_flow_allowed),
            (SemanticPredicate.PROFILE_PINNED, self.profile_pinned),
        )


@dataclass(frozen=True, slots=True)
class SemanticAssessmentRequest:
    kind: SemanticAssessmentKind
    domain: SemanticDomain
    action_id: str
    input_sha256: str
    deadline_at: str
    data_labels: DataLabels = field(default_factory=DataLabels)
    features: AuthoritativeApprovalFacts = field(default_factory=AuthoritativeApprovalFacts)
    redacted_intent: str | None = None
    pid: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    effect_id: str | None = None
    manifest_id: str | None = None
    manifest_sha256: str | None = None
    policy_sha256: str | None = None
    resource_sha256: str | None = None
    args_sha256: str | None = None
    state_sha256: str | None = None
    source_refs_sha256: str | None = None
    data_labels_sha256: str | None = None
    sink_identity_sha256: str | None = None
    tool_schema_sha256: str | None = None
    provider_spec_sha256: str | None = None
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        object.__setattr__(self, "kind", _enum(SemanticAssessmentKind, self.kind, "assessment kind"))
        object.__setattr__(self, "domain", _enum(SemanticDomain, self.domain, "semantic domain"))
        if not isinstance(self.action_id, str) or _ACTION_RE.fullmatch(self.action_id) is None:
            raise ValueError("semantic action_id must be a dotted lower-case identifier")
        _sha256("semantic input_sha256", self.input_sha256)
        _timestamp("semantic deadline_at", self.deadline_at)
        if not isinstance(self.data_labels, DataLabels):
            raise TypeError("semantic data_labels must be DataLabels")
        if not isinstance(self.features, AuthoritativeApprovalFacts):
            raise TypeError("semantic features must be AuthoritativeApprovalFacts")
        _optional_text(
            "semantic redacted_intent",
            self.redacted_intent,
            SEMANTIC_REDACTED_INTENT_MAX_CHARS,
        )
        for name in ("pid", "request_id", "operation_id", "effect_id", "manifest_id"):
            _optional_text(f"semantic {name}", getattr(self, name), 512)
        for name in ("manifest_sha256", "policy_sha256", "resource_sha256", "args_sha256", "state_sha256", "source_refs_sha256", "data_labels_sha256", "sink_identity_sha256", "tool_schema_sha256", "provider_spec_sha256"):
            _optional_sha256(f"semantic {name}", getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "domain": self.domain.value,
            "action_id": self.action_id,
            "input_sha256": self.input_sha256,
            "deadline_at": self.deadline_at,
            "data_labels": self.data_labels.to_dict(),
            "features": self.features.to_dict(),
            "redacted_intent": self.redacted_intent,
            "pid": self.pid,
            "request_id": self.request_id,
            "operation_id": self.operation_id,
            "effect_id": self.effect_id,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "policy_sha256": self.policy_sha256,
            "resource_sha256": self.resource_sha256,
            "args_sha256": self.args_sha256,
            "state_sha256": self.state_sha256,
            "source_refs_sha256": self.source_refs_sha256,
            "data_labels_sha256": self.data_labels_sha256,
            "sink_identity_sha256": self.sink_identity_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "provider_spec_sha256": self.provider_spec_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticAssessmentRequest:
        keys = {"schema_version", "kind", "domain", "action_id", "input_sha256", "deadline_at", "data_labels", "features", "redacted_intent", "pid", "request_id", "operation_id", "effect_id", "manifest_id", "manifest_sha256", "policy_sha256", "resource_sha256", "args_sha256", "state_sha256", "source_refs_sha256", "data_labels_sha256", "sink_identity_sha256", "tool_schema_sha256", "provider_spec_sha256"}
        _exact(value, keys, "semantic assessment request")
        selected = dict(value)
        selected["data_labels"] = DataLabels.from_dict(selected["data_labels"])
        selected["features"] = AuthoritativeApprovalFacts.from_dict(selected["features"])
        return cls(**selected)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticAssessment:
    status: SemanticAssessmentStatus
    findings: tuple[SemanticFinding, ...] = ()
    data_findings: tuple[SemanticDataFinding, ...] = ()
    confidence_bps: int = 0
    calibration_bucket: SemanticCalibrationBucket = SemanticCalibrationBucket.UNKNOWN
    ood: bool = False
    abstain: bool = False
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        object.__setattr__(self, "status", _enum(SemanticAssessmentStatus, self.status, "assessment status"))
        object.__setattr__(self, "calibration_bucket", _enum(SemanticCalibrationBucket, self.calibration_bucket, "calibration bucket"))
        if not isinstance(self.findings, tuple) or any(not isinstance(item, SemanticFinding) for item in self.findings):
            raise TypeError("semantic findings must be a tuple of SemanticFinding")
        if not isinstance(self.data_findings, tuple) or any(not isinstance(item, SemanticDataFinding) for item in self.data_findings):
            raise TypeError("semantic data_findings must be a tuple of SemanticDataFinding")
        if len(self.findings) > 64 or len(self.data_findings) > 64:
            raise ValueError("semantic assessment may contain at most 64 findings of each kind")
        _confidence(self.confidence_bps)
        if type(self.ood) is not bool or type(self.abstain) is not bool:
            raise TypeError("semantic ood and abstain must be booleans")
        if self.ood != (self.status is SemanticAssessmentStatus.OOD):
            raise ValueError("semantic OOD status and flag must match")
        if self.abstain != (self.status is SemanticAssessmentStatus.ABSTAINED):
            raise ValueError("semantic abstained status and flag must match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "findings": [item.to_dict() for item in self.findings],
            "data_findings": [item.to_dict() for item in self.data_findings],
            "confidence_bps": self.confidence_bps,
            "calibration_bucket": self.calibration_bucket.value,
            "ood": self.ood,
            "abstain": self.abstain,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticAssessment:
        keys = {"schema_version", "status", "findings", "data_findings", "confidence_bps", "calibration_bucket", "ood", "abstain"}
        _exact(value, keys, "semantic assessment")
        findings = value["findings"]
        data_findings = value["data_findings"]
        if not isinstance(findings, list) or not isinstance(data_findings, list):
            raise TypeError("semantic assessment findings must be arrays")
        selected = dict(value)
        selected["findings"] = tuple(SemanticFinding.from_dict(item) for item in findings)
        selected["data_findings"] = tuple(SemanticDataFinding.from_dict(item) for item in data_findings)
        return cls(**selected)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticApprovalRule:
    rule_id: str
    authority_operation: str
    resource: str
    rights: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("semantic rule_id", self.rule_id, 128)
        if _ACTION_RE.fullmatch(self.authority_operation) is None or "*" in self.authority_operation:
            raise ValueError("semantic authority_operation must be an exact dotted operation")
        _text("semantic resource", self.resource, 2_048)
        if self.resource.count("*") > 1 or ("*" in self.resource and not self.resource.endswith("*")):
            raise ValueError("semantic resource allows only one trailing wildcard")
        if not isinstance(self.rights, tuple) or not self.rights:
            raise ValueError("semantic rights must be a non-empty tuple")
        if len(set(self.rights)) != len(self.rights):
            raise ValueError("semantic rights must not contain duplicates")
        allowed = {"read", "write", "execute", "link", "diff", "materialize", "delete"}
        if any(type(right) is not str or right not in allowed for right in self.rights):
            raise ValueError("semantic rights contain an unsupported or control right")

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "authority_operation": self.authority_operation, "resource": self.resource, "rights": list(self.rights)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticApprovalRule:
        _exact(value, {"rule_id", "authority_operation", "resource", "rights"}, "semantic approval rule")
        rights = value["rights"]
        if not isinstance(rights, list):
            raise TypeError("semantic approval rule rights must be an array")
        return cls(rule_id=value["rule_id"], authority_operation=value["authority_operation"], resource=value["resource"], rights=tuple(rights))


@dataclass(frozen=True, slots=True)
class SemanticApprovalCeiling:
    rules: tuple[SemanticApprovalRule, ...] = ()
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        if not isinstance(self.rules, tuple) or any(not isinstance(rule, SemanticApprovalRule) for rule in self.rules):
            raise TypeError("semantic ceiling rules must be a tuple of SemanticApprovalRule")
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic ceiling rule_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "rules": [rule.to_dict() for rule in self.rules]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticApprovalCeiling:
        _exact(value, {"schema_version", "rules"}, "semantic approval ceiling")
        rules = value["rules"]
        if not isinstance(rules, list):
            raise TypeError("semantic approval ceiling rules must be an array")
        return cls(rules=tuple(SemanticApprovalRule.from_dict(rule) for rule in rules), schema_version=value["schema_version"])


@dataclass(frozen=True, slots=True)
class SemanticApprovalCandidate:
    rule_id: str
    authority_operation: str
    resource: str
    rights: tuple[str, ...]
    manifest_id: str
    manifest_sha256: str
    policy_sha256: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        SemanticApprovalRule(self.rule_id, self.authority_operation, self.resource, self.rights)
        _text("semantic candidate manifest_id", self.manifest_id, 512)
        _sha256("semantic candidate manifest_sha256", self.manifest_sha256)
        _sha256("semantic candidate policy_sha256", self.policy_sha256)

    @property
    def rule(self) -> SemanticApprovalRule:
        return SemanticApprovalRule(self.rule_id, self.authority_operation, self.resource, self.rights)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "rule_id": self.rule_id, "authority_operation": self.authority_operation, "resource": self.resource, "rights": list(self.rights), "manifest_id": self.manifest_id, "manifest_sha256": self.manifest_sha256, "policy_sha256": self.policy_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticApprovalCandidate:
        _exact(value, {"schema_version", "rule_id", "authority_operation", "resource", "rights", "manifest_id", "manifest_sha256", "policy_sha256"}, "semantic approval candidate")
        rights = value["rights"]
        if not isinstance(rights, (list, tuple)):
            raise TypeError("semantic approval candidate rights must be an array")
        return cls(**{**dict(value), "rights": tuple(rights)})


@dataclass(frozen=True, slots=True)
class ShadowPolicyDecision:
    outcome: ShadowPolicyOutcome
    reason_codes: tuple[SemanticReasonCode, ...]
    matched_rule_id: str | None
    proven_predicates: tuple[SemanticPredicate, ...]
    missing_predicates: tuple[SemanticPredicate, ...]
    policy_sha256: str
    manifest_sha256: str | None
    assessment_sha256: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        object.__setattr__(self, "outcome", _enum(ShadowPolicyOutcome, self.outcome, "shadow outcome"))
        object.__setattr__(self, "reason_codes", _enum_tuple(SemanticReasonCode, self.reason_codes, "shadow reason_codes"))
        object.__setattr__(self, "proven_predicates", _enum_tuple(SemanticPredicate, self.proven_predicates, "proven_predicates"))
        object.__setattr__(self, "missing_predicates", _enum_tuple(SemanticPredicate, self.missing_predicates, "missing_predicates"))
        _optional_text("matched_rule_id", self.matched_rule_id, 128)
        _sha256("shadow policy_sha256", self.policy_sha256)
        _optional_sha256("shadow manifest_sha256", self.manifest_sha256)
        _sha256("shadow assessment_sha256", self.assessment_sha256)
        if self.outcome is ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE and (self.matched_rule_id is None or self.missing_predicates):
            raise ValueError("would_issue_exact_once requires a matched rule and no missing predicates")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "outcome": self.outcome.value, "reason_codes": [item.value for item in self.reason_codes], "matched_rule_id": self.matched_rule_id, "proven_predicates": [item.value for item in self.proven_predicates], "missing_predicates": [item.value for item in self.missing_predicates], "policy_sha256": self.policy_sha256, "manifest_sha256": self.manifest_sha256, "assessment_sha256": self.assessment_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShadowPolicyDecision:
        _exact(value, {"schema_version", "outcome", "reason_codes", "matched_rule_id", "proven_predicates", "missing_predicates", "policy_sha256", "manifest_sha256", "assessment_sha256"}, "shadow policy decision")
        for name in ("reason_codes", "proven_predicates", "missing_predicates"):
            if not isinstance(value[name], list):
                raise TypeError(f"shadow {name} must be an array")
        return cls(**{**dict(value), "reason_codes": tuple(value["reason_codes"]), "proven_predicates": tuple(value["proven_predicates"]), "missing_predicates": tuple(value["missing_predicates"])})


SEMANTIC_PROVIDER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "status", "findings", "data_findings", "confidence_bps", "calibration_bucket", "ood", "abstain"],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "status": {"type": "string", "enum": [item.value for item in SemanticAssessmentStatus]},
        "findings": {"type": "array", "maxItems": 64, "items": {"type": "object", "additionalProperties": False, "required": ["code", "severity", "confidence_bps", "evidence_sha256", "source"], "properties": {"code": {"type": "string", "enum": [item.value for item in SemanticReasonCode]}, "severity": {"type": "string", "enum": [item.value for item in SemanticFindingSeverity]}, "confidence_bps": {"type": "integer", "minimum": 0, "maximum": 10000}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "source": {"type": "string", "enum": [item.value for item in SemanticFindingSource]}}}},
        "data_findings": {"type": "array", "maxItems": 64, "items": {"type": "object", "additionalProperties": False, "required": ["category", "field", "span_start", "span_end", "sensitivity_floor", "integrity_ceiling", "trust_ceiling", "confidence_bps", "evidence_sha256"], "properties": {"category": {"type": "string", "enum": [item.value for item in SemanticDataCategory]}, "field": {"type": "string", "enum": [item.value for item in SemanticDataLocator]}, "span_start": {"type": ["integer", "null"], "minimum": 0}, "span_end": {"type": ["integer", "null"], "minimum": 0}, "sensitivity_floor": {"type": "string", "enum": [item.value for item in DataSensitivity]}, "integrity_ceiling": {"type": "string", "enum": [item.value for item in DataIntegrity]}, "trust_ceiling": {"type": "string", "enum": [item.value for item in DataTrustLevel]}, "confidence_bps": {"type": "integer", "minimum": 0, "maximum": 10000}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}},
        "confidence_bps": {"type": "integer", "minimum": 0, "maximum": 10000},
        "calibration_bucket": {"type": "string", "enum": [item.value for item in SemanticCalibrationBucket]},
        "ood": {"type": "boolean"},
        "abstain": {"type": "boolean"},
    },
}


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(f"{label} fields must be exactly {sorted(keys)}; got {sorted(actual)}")


def _enum(enum_type: type[_E], value: Any, label: str) -> _E:
    if isinstance(value, enum_type):
        return value
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {label}: {value}") from exc


def _enum_tuple(enum_type: type[_E], values: Any, label: str) -> tuple[_E, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_enum(enum_type, item, label) for item in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _version(value: Any) -> None:
    if type(value) is not int or value != SEMANTIC_SCHEMA_VERSION:
        raise ValueError("unsupported semantic schema_version")


def _confidence(value: Any) -> None:
    if type(value) is not int or not 0 <= value <= 10_000:
        raise ValueError("semantic confidence_bps must be an integer from 0 through 10000")


def _nonnegative_int(label: str, value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _text(label: str, value: Any, max_chars: int) -> None:
    if type(value) is not str or not value or value != value.strip() or len(value) > max_chars or "\x00" in value:
        raise ValueError(f"{label} must be non-empty, trimmed, NUL-free, and at most {max_chars} characters")


def _optional_text(label: str, value: Any, max_chars: int) -> None:
    if value is not None:
        _text(label, value, max_chars)


def _sha256(label: str, value: Any) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _optional_sha256(label: str, value: Any) -> None:
    if value is not None:
        _sha256(label, value)


def _timestamp(label: str, value: Any) -> None:
    _text(label, value, 64)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuthoritativeApprovalFacts",
    "SEMANTIC_PROVIDER_RESPONSE_SCHEMA",
    "SEMANTIC_REDACTED_INTENT_MAX_CHARS",
    "SEMANTIC_SCHEMA_VERSION",
    "SemanticApprovalCandidate",
    "SemanticApprovalCeiling",
    "SemanticApprovalRule",
    "SemanticAssessment",
    "SemanticAssessmentKind",
    "SemanticAssessmentRequest",
    "SemanticAssessmentStatus",
    "SemanticCalibrationBucket",
    "SemanticDataCategory",
    "SemanticDataFinding",
    "SemanticDataLocator",
    "SemanticDomain",
    "SemanticFinding",
    "SemanticFindingSeverity",
    "SemanticFindingSource",
    "SemanticPredicate",
    "SemanticReasonCode",
    "ShadowPolicyDecision",
    "ShadowPolicyOutcome",
]
