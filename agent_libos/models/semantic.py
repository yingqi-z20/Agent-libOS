from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, TypeVar

from agent_libos.models.base import StrEnum
from agent_libos.models.data_flow import (
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataTrustLevel,
)


SEMANTIC_SCHEMA_VERSION = 1
SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION = 2
SEMANTIC_STATUS_SCHEMA_VERSION = 3
SEMANTIC_ACTION_CATALOG_VERSION = 1
SEMANTIC_REDACTED_INTENT_MAX_CHARS = 2_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_PREVIEW_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_PREVIEW_PUBLIC_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+/*=-]{0,511}$")
_PREVIEW_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]{0,255}$")
_PREVIEW_GIT_REFERENCE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/@:+~^{}-]{0,255}$"
)
_PREVIEW_REDACTED = "<redacted>"
_SEMANTIC_APPROVAL_GIT_REFERENCE_ROLES = frozenset(
    {
        "base",
        "base_oid",
        "base_ref",
        "branch",
        "expected_remote_oid",
        "git_old_oid",
        "git_remote",
        "git_remote_ref",
        "head",
        "head_oid",
        "head_ref",
        "index_oid",
        "local_ref",
        "managed_worktree_id",
        "new_branch",
        "new_name",
        "patch_oid",
        "pr_id",
        "ref",
        "remote",
        "remote_ref",
        "source",
        "start",
        "tag",
        "target",
    }
)
SEMANTIC_ACTION_CATALOG_V1: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "filesystem.read": frozenset({"read"}),
        "git.read": frozenset({"read"}),
        "git.diff": frozenset({"diff"}),
    }
)
_CATALOG_V1_AUTO_RIGHTS = SEMANTIC_ACTION_CATALOG_V1
_CAPABILITY_RIGHTS = frozenset(
    {
        "read",
        "write",
        "execute",
        "link",
        "diff",
        "materialize",
        "delete",
        "grant",
        "revoke",
        "approve",
        "admin",
    }
)
_EXECUTABLE_DENY_REASONS = frozenset(
    {
        "malformed_request",
        "stale_binding",
        "stale_manifest",
        "stale_policy",
        "data_flow_denied",
        "policy_hard_deny",
        "digest_drift",
    }
)
_E = TypeVar("_E", bound=StrEnum)


class SemanticAssessmentKind(StrEnum):
    APPROVAL = "approval"
    ROOT_GOAL = "root_goal"
    PROVIDER_INGRESS = "provider_ingress"


class SemanticRuntimeMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE_DENY = "enforce_deny"
    CANARY_AUTO = "canary_auto"


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
    DATA_FLOW_DENIED = "data_flow_denied"
    FLOW_COVERAGE_INCOMPLETE = "flow_coverage_incomplete"
    POLICY_HARD_DENY = "policy_hard_deny"
    TENANT_NOT_ALLOWED = "tenant_not_allowed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONTROL_DISABLED = "control_disabled"
    CONTROL_TRIPPED = "control_tripped"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    CALIBRATION_TOO_LOW = "calibration_too_low"
    DIGEST_DRIFT = "digest_drift"
    REVISION_RACE_LOST = "revision_race_lost"
    CAPABILITY_EXPIRED = "capability_expired"
    CAPABILITY_REVOKED = "capability_revoked"


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


class SemanticFlowCoverage(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    STALE = "stale"


class SemanticMachineSettlementOutcome(StrEnum):
    ISSUED = "issued"
    DENIED = "denied"
    REQUIRE_HUMAN = "require_human"
    RACE_LOST = "race_lost"
    STALE = "stale"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REVOKED = "revoked"
    EXPIRED = "expired"
    FAILED = "failed"


class SemanticReviewOutcome(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    INCONCLUSIVE = "inconclusive"


class SemanticTripCode(StrEnum):
    UNSAFE_REVIEW = "unsafe_review"
    CRITICAL_HIGH_GRANT = "critical_high_grant"
    CROSS_TENANT = "cross_tenant"
    SECRET_EGRESS = "secret_egress"
    REPLAY_DETECTED = "replay_detected"
    BINDING_MISMATCH = "binding_mismatch"
    UNAUTHORIZED_EFFECT = "unauthorized_effect"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"


class SemanticPreviewRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SemanticApprovalArgumentKind(StrEnum):
    """Closed Host projection variants shown on approval surfaces."""

    FILESYSTEM = "filesystem"
    SHELL = "shell"
    GIT = "git"
    JSONRPC = "jsonrpc"
    MCP = "mcp"
    OTHER = "other"


class SemanticPublicControlState(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    TRIPPED = "tripped"
    REVOKED = "revoked"


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
        resource_kind, separator, resource_body = self.resource.partition(":")
        if (
            not separator
            or not resource_kind
            or not resource_body
            or resource_kind != self.authority_operation.split(".", 1)[0]
        ):
            raise ValueError(
                "semantic resource kind must match the authority operation"
            )
        if "*" in self.resource and (
            self.resource.count("*") != 1
            or not self.resource.endswith((":*", "/*"))
        ):
            raise ValueError(
                "semantic resource wildcard must be one canonical terminal segment"
            )
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
class SemanticApprovalCandidateSnapshotV1:
    """Digest-only evidence for replaying a frozen Shadow evaluation.

    A snapshot deliberately omits the raw resource needed to match an active
    policy epoch.  It can therefore preserve the rule selected at capture time
    after the Human request has reached a terminal state, but it cannot be used
    as an exact candidate by Phase 4 settlement.
    """

    rule_id: str
    authority_operation: str
    rights: tuple[str, ...]
    manifest_id: str
    manifest_sha256: str
    policy_sha256: str
    resource_sha256: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _text("semantic snapshot rule_id", self.rule_id, 128)
        _text(
            "semantic snapshot authority_operation",
            self.authority_operation,
            128,
        )
        if not _ACTION_RE.fullmatch(self.authority_operation):
            raise ValueError("semantic snapshot authority_operation is invalid")
        if not isinstance(self.rights, tuple) or not self.rights:
            raise ValueError("semantic snapshot rights must be a non-empty tuple")
        if len(set(self.rights)) != len(self.rights):
            raise ValueError("semantic snapshot rights must not contain duplicates")
        allowed = {
            "read",
            "write",
            "execute",
            "link",
            "diff",
            "materialize",
            "delete",
        }
        if any(
            type(right) is not str or right not in allowed for right in self.rights
        ):
            raise ValueError(
                "semantic snapshot rights contain an unsupported or control right"
            )
        _text("semantic snapshot manifest_id", self.manifest_id, 512)
        _sha256("semantic snapshot manifest_sha256", self.manifest_sha256)
        _sha256("semantic snapshot policy_sha256", self.policy_sha256)
        _sha256("semantic snapshot resource_sha256", self.resource_sha256)

    @classmethod
    def from_candidate(
        cls,
        candidate: SemanticApprovalCandidate,
    ) -> SemanticApprovalCandidateSnapshotV1:
        if not isinstance(candidate, SemanticApprovalCandidate):
            raise TypeError("candidate must be SemanticApprovalCandidate")
        return cls(
            rule_id=candidate.rule_id,
            authority_operation=candidate.authority_operation,
            rights=candidate.rights,
            manifest_id=candidate.manifest_id,
            manifest_sha256=candidate.manifest_sha256,
            policy_sha256=candidate.policy_sha256,
            resource_sha256=_canonical_sha256(candidate.resource),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "authority_operation": self.authority_operation,
            "rights": list(self.rights),
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "policy_sha256": self.policy_sha256,
            "resource_sha256": self.resource_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> SemanticApprovalCandidateSnapshotV1:
        _exact(
            value,
            {
                "schema_version",
                "rule_id",
                "authority_operation",
                "rights",
                "manifest_id",
                "manifest_sha256",
                "policy_sha256",
                "resource_sha256",
            },
            "semantic approval candidate snapshot",
        )
        rights = value["rights"]
        if not isinstance(rights, list):
            raise TypeError("semantic approval candidate snapshot rights must be an array")
        return cls(**{**dict(value), "rights": tuple(rights)})


@dataclass(frozen=True, slots=True)
class SemanticHardDenyRuleV1:
    """Host-authored deterministic deny rule.

    This model deliberately cannot express a model-supplied reason, an allow
    effect, or a wildcard operation.  Resource matching uses the same bounded
    trailing-prefix form as the task authority ceiling.
    """

    rule_id: str
    authority_operation: str
    resource: str
    rights: tuple[str, ...]
    reason_code: SemanticReasonCode = SemanticReasonCode.POLICY_HARD_DENY
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        SemanticApprovalRule(
            self.rule_id,
            self.authority_operation,
            self.resource,
            self.rights,
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(SemanticReasonCode, self.reason_code, "hard-deny reason code"),
        )
        if self.reason_code is not SemanticReasonCode.POLICY_HARD_DENY:
            raise ValueError("semantic hard-deny rules require policy_hard_deny")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "authority_operation": self.authority_operation,
            "resource": self.resource,
            "rights": list(self.rights),
            "reason_code": self.reason_code.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticHardDenyRuleV1:
        _exact(
            value,
            {
                "schema_version",
                "rule_id",
                "authority_operation",
                "resource",
                "rights",
                "reason_code",
            },
            "semantic hard-deny rule",
        )
        rights = value["rights"]
        if not isinstance(rights, list):
            raise TypeError("semantic hard-deny rule rights must be an array")
        return cls(**{**dict(value), "rights": tuple(rights)})


@dataclass(frozen=True, slots=True)
class SemanticPolicyEpochV1:
    """Immutable Host policy epoch for deterministic deny and canary grants."""

    epoch_id: str
    generation: int
    expected_previous_sha256: str | None
    tenant_bucket_sha256s: tuple[str, ...]
    auto_approval_rules: tuple[SemanticApprovalRule, ...]
    hard_deny_rules: tuple[SemanticHardDenyRuleV1, ...]
    created_at: str
    classifier_profile_id: str | None = None
    classifier_profile_sha256: str | None = None
    classifier_model_sha256: str | None = None
    minimum_confidence_bps: int = 9_900
    required_calibration_bucket: SemanticCalibrationBucket = (
        SemanticCalibrationBucket.VERY_HIGH
    )
    capability_ttl_s: int = 60
    per_rule_per_minute_limit: int = 10
    per_rule_per_day_limit: int = 100
    max_inflight: int = 2
    catalog_version: int = SEMANTIC_ACTION_CATALOG_VERSION
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _semantic_policy_epoch_identity(self)
        _semantic_policy_tenants(self)
        _semantic_policy_rules(self)
        _semantic_policy_classifier(self)
        _semantic_policy_thresholds(self)
        _timestamp("semantic policy created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch_id": self.epoch_id,
            "generation": self.generation,
            "catalog_version": self.catalog_version,
            "expected_previous_sha256": self.expected_previous_sha256,
            "tenant_bucket_sha256s": list(self.tenant_bucket_sha256s),
            "auto_approval_rules": [
                rule.to_dict() for rule in self.auto_approval_rules
            ],
            "hard_deny_rules": [rule.to_dict() for rule in self.hard_deny_rules],
            "classifier_profile_id": self.classifier_profile_id,
            "classifier_profile_sha256": self.classifier_profile_sha256,
            "classifier_model_sha256": self.classifier_model_sha256,
            "minimum_confidence_bps": self.minimum_confidence_bps,
            "required_calibration_bucket": self.required_calibration_bucket.value,
            "capability_ttl_s": self.capability_ttl_s,
            "per_rule_per_minute_limit": self.per_rule_per_minute_limit,
            "per_rule_per_day_limit": self.per_rule_per_day_limit,
            "max_inflight": self.max_inflight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticPolicyEpochV1:
        keys = {
            "schema_version",
            "epoch_id",
            "generation",
            "catalog_version",
            "expected_previous_sha256",
            "tenant_bucket_sha256s",
            "auto_approval_rules",
            "hard_deny_rules",
            "classifier_profile_id",
            "classifier_profile_sha256",
            "classifier_model_sha256",
            "minimum_confidence_bps",
            "required_calibration_bucket",
            "capability_ttl_s",
            "per_rule_per_minute_limit",
            "per_rule_per_day_limit",
            "max_inflight",
            "created_at",
        }
        _exact(value, keys, "semantic policy epoch")
        tenants = value["tenant_bucket_sha256s"]
        auto_rules = value["auto_approval_rules"]
        deny_rules = value["hard_deny_rules"]
        if not isinstance(tenants, list):
            raise TypeError("semantic policy tenant_bucket_sha256s must be an array")
        if not isinstance(auto_rules, list) or not isinstance(
            deny_rules, list
        ):
            raise TypeError("semantic policy rules must be arrays")
        selected = dict(value)
        selected["tenant_bucket_sha256s"] = tuple(tenants)
        selected["auto_approval_rules"] = tuple(
            SemanticApprovalRule.from_dict(item)
            for item in auto_rules
        )
        selected["hard_deny_rules"] = tuple(
            SemanticHardDenyRuleV1.from_dict(item)
            for item in deny_rules
        )
        return cls(**selected)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def _semantic_policy_epoch_identity(epoch: SemanticPolicyEpochV1) -> None:
    if (
        type(epoch.catalog_version) is not int
        or epoch.catalog_version != SEMANTIC_ACTION_CATALOG_VERSION
    ):
        raise ValueError("unsupported semantic action catalog_version")
    _text("semantic policy epoch_id", epoch.epoch_id, 512)
    _positive_int("semantic policy generation", epoch.generation)
    _optional_sha256(
        "semantic expected_previous_sha256", epoch.expected_previous_sha256
    )


def _semantic_policy_tenants(epoch: SemanticPolicyEpochV1) -> None:
    tenants = epoch.tenant_bucket_sha256s
    if not isinstance(tenants, tuple):
        raise TypeError("semantic policy tenant_bucket_sha256s must be a tuple")
    if len(tenants) != len(set(tenants)):
        raise ValueError(
            "semantic policy tenant_bucket_sha256s must not contain duplicates"
        )
    for bucket_sha256 in tenants:
        _sha256("semantic policy tenant bucket", bucket_sha256)
    object.__setattr__(
        epoch,
        "tenant_bucket_sha256s",
        tuple(sorted(tenants)),
    )


def _semantic_policy_rules(epoch: SemanticPolicyEpochV1) -> None:
    if not isinstance(epoch.auto_approval_rules, tuple) or any(
        not isinstance(rule, SemanticApprovalRule)
        for rule in epoch.auto_approval_rules
    ):
        raise TypeError(
            "semantic policy auto_approval_rules must be a tuple of SemanticApprovalRule"
        )
    if not isinstance(epoch.hard_deny_rules, tuple) or any(
        not isinstance(rule, SemanticHardDenyRuleV1)
        for rule in epoch.hard_deny_rules
    ):
        raise TypeError(
            "semantic policy hard_deny_rules must be a tuple of SemanticHardDenyRuleV1"
        )
    _semantic_policy_rule_set(epoch)
    object.__setattr__(
        epoch,
        "auto_approval_rules",
        tuple(sorted(epoch.auto_approval_rules, key=lambda rule: rule.rule_id)),
    )
    object.__setattr__(
        epoch,
        "hard_deny_rules",
        tuple(sorted(epoch.hard_deny_rules, key=lambda rule: rule.rule_id)),
    )


def _semantic_policy_rule_set(epoch: SemanticPolicyEpochV1) -> None:
    if not epoch.auto_approval_rules and not epoch.hard_deny_rules:
        raise ValueError("semantic policy epoch must contain at least one rule")
    if epoch.auto_approval_rules and not epoch.tenant_bucket_sha256s:
        raise ValueError(
            "semantic auto-approval rules require exact tenant_bucket_sha256s"
        )
    if (
        epoch.generation == 1
        and epoch.auto_approval_rules
        and len(epoch.tenant_bucket_sha256s) != 1
    ):
        raise ValueError(
            "the first semantic canary epoch requires exactly one tenant bucket"
        )
    rule_ids = [
        *(rule.rule_id for rule in epoch.auto_approval_rules),
        *(rule.rule_id for rule in epoch.hard_deny_rules),
    ]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("semantic policy rule_id values must be unique")
    for rule in epoch.auto_approval_rules:
        allowed_rights = _CATALOG_V1_AUTO_RIGHTS.get(rule.authority_operation)
        if allowed_rights is None:
            raise ValueError(
                "semantic auto-approval rule references an operation outside catalog v1"
            )
        if len(rule.rights) != 1 or not set(rule.rights).issubset(allowed_rights):
            raise ValueError(
                "semantic auto-approval rule must request the catalog's exact single right"
            )


def _semantic_policy_classifier(epoch: SemanticPolicyEpochV1) -> None:
    _optional_text(
        "semantic classifier_profile_id", epoch.classifier_profile_id, 512
    )
    _optional_sha256(
        "semantic classifier_profile_sha256", epoch.classifier_profile_sha256
    )
    _optional_sha256(
        "semantic classifier_model_sha256", epoch.classifier_model_sha256
    )
    identity = (
        epoch.classifier_profile_id,
        epoch.classifier_profile_sha256,
        epoch.classifier_model_sha256,
    )
    if any(item is None for item in identity) and any(
        item is not None for item in identity
    ):
        raise ValueError(
            "semantic classifier profile id, profile digest, and model digest must be supplied together"
        )


def _semantic_policy_thresholds(epoch: SemanticPolicyEpochV1) -> None:
    _confidence(epoch.minimum_confidence_bps)
    if epoch.minimum_confidence_bps < 9_900:
        raise ValueError("semantic minimum_confidence_bps cannot be lower than 9900")
    object.__setattr__(
        epoch,
        "required_calibration_bucket",
        _enum(
            SemanticCalibrationBucket,
            epoch.required_calibration_bucket,
            "required calibration bucket",
        ),
    )
    if epoch.required_calibration_bucket is not SemanticCalibrationBucket.VERY_HIGH:
        raise ValueError(
            "semantic auto approval requires the very_high calibration bucket"
        )
    _bounded_positive_int("semantic capability_ttl_s", epoch.capability_ttl_s, 300)
    _bounded_positive_int(
        "semantic per_rule_per_minute_limit", epoch.per_rule_per_minute_limit, 10
    )
    _bounded_positive_int(
        "semantic per_rule_per_day_limit", epoch.per_rule_per_day_limit, 100
    )
    _bounded_positive_int("semantic max_inflight", epoch.max_inflight, 2)


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


@dataclass(frozen=True, slots=True)
class SemanticControlStateV1:
    """Durable kill-switch and active-epoch pointer, updated through CAS."""

    revision: int
    generation: int
    mode: SemanticRuntimeMode
    active_epoch_id: str | None
    active_policy_sha256: str | None
    tripped: bool
    trip_code: SemanticTripCode | None
    updated_at: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _nonnegative_int("semantic control revision", self.revision)
        _nonnegative_int("semantic control generation", self.generation)
        object.__setattr__(
            self, "mode", _enum(SemanticRuntimeMode, self.mode, "semantic mode")
        )
        _optional_text("semantic active_epoch_id", self.active_epoch_id, 512)
        _optional_sha256(
            "semantic active_policy_sha256", self.active_policy_sha256
        )
        if (self.active_epoch_id is None) != (
            self.active_policy_sha256 is None
        ):
            raise ValueError("semantic control active epoch identity must be complete")
        if type(self.tripped) is not bool:
            raise TypeError("semantic control tripped must be a boolean")
        if self.trip_code is not None:
            object.__setattr__(
                self,
                "trip_code",
                _enum(SemanticTripCode, self.trip_code, "semantic trip code"),
            )
        if self.tripped != (self.trip_code is not None):
            raise ValueError("semantic control trip state and code must agree")
        if (
            self.mode
            in {SemanticRuntimeMode.ENFORCE_DENY, SemanticRuntimeMode.CANARY_AUTO}
            and self.active_epoch_id is None
        ):
            raise ValueError("active semantic modes require a policy epoch")
        if self.mode in {SemanticRuntimeMode.OFF, SemanticRuntimeMode.SHADOW} and (
            self.active_epoch_id is not None or self.tripped
        ):
            raise ValueError(
                "inactive semantic modes cannot retain active or tripped authority"
            )
        _timestamp("semantic control updated_at", self.updated_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "generation": self.generation,
            "mode": self.mode.value,
            "active_epoch_id": self.active_epoch_id,
            "active_policy_sha256": self.active_policy_sha256,
            "tripped": self.tripped,
            "trip_code": self.trip_code.value if self.trip_code is not None else None,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticControlStateV1:
        _exact(
            value,
            {
                "schema_version",
                "revision",
                "generation",
                "mode",
                "active_epoch_id",
                "active_policy_sha256",
                "tripped",
                "trip_code",
                "updated_at",
            },
            "semantic control state",
        )
        return cls(**dict(value))

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticApprovalBindingV2:
    """Exact, non-delegable semantic approval binding.

    The first three effect fields intentionally preserve the legacy approval
    binding vocabulary.  Every remaining field is mandatory Host evidence for
    machine issuance; a legacy three-field binding cannot be upcast safely.
    """

    request_id: str
    request_revision: int
    pid: str
    operation_id: str | None
    effect_id: str
    authority_operation: str
    resource: str
    right: str
    canonical_args_hash: str
    target_state_version: str | int | None
    manifest_id: str
    manifest_sha256: str
    ceiling_sha256: str
    policy_epoch_id: str
    policy_epoch_sha256: str
    control_generation: int
    assessment_id: str
    assessment_sha256: str
    classifier_profile_sha256: str
    classifier_model_sha256: str
    tenant_bucket_sha256: str
    source_labels_sha256: str
    source_refs_sha256: str
    flow_snapshot_sha256: str
    sink_identity_sha256: str | None
    tool_schema_sha256: str | None
    provider_spec_sha256: str | None
    nonce: str
    issued_at: str
    expires_at: str
    schema_version: int = SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION
        ):
            raise ValueError("unsupported semantic approval binding schema_version")
        for label, value, maximum in (
            ("request_id", self.request_id, 512),
            ("pid", self.pid, 512),
            ("effect_id", self.effect_id, 512),
            ("manifest_id", self.manifest_id, 512),
            ("policy_epoch_id", self.policy_epoch_id, 512),
            ("assessment_id", self.assessment_id, 512),
            ("nonce", self.nonce, 512),
        ):
            _text(f"semantic binding {label}", value, maximum)
        _optional_text("semantic binding operation_id", self.operation_id, 512)
        _nonnegative_int("semantic binding request_revision", self.request_revision)
        _positive_int("semantic binding control_generation", self.control_generation)
        if (
            _ACTION_RE.fullmatch(self.authority_operation) is None
            or self.authority_operation not in _CATALOG_V1_AUTO_RIGHTS
        ):
            raise ValueError(
                "semantic binding authority_operation must be in action catalog v1"
            )
        _text("semantic binding resource", self.resource, 2_048)
        if "*" in self.resource:
            raise ValueError("semantic binding resource must be exact")
        resource_kind, separator, resource_body = self.resource.partition(":")
        if (
            separator != ":"
            or not resource_kind
            or not resource_body
            or resource_kind != self.authority_operation.split(".", 1)[0]
        ):
            raise ValueError(
                "semantic binding resource kind must match the authority operation"
            )
        if self.right not in _CATALOG_V1_AUTO_RIGHTS[self.authority_operation]:
            raise ValueError(
                "semantic binding right must be the catalog's exact operation right"
            )
        for label, digest in (
            ("canonical_args_hash", self.canonical_args_hash),
            ("manifest_sha256", self.manifest_sha256),
            ("ceiling_sha256", self.ceiling_sha256),
            ("policy_epoch_sha256", self.policy_epoch_sha256),
            ("assessment_sha256", self.assessment_sha256),
            ("classifier_profile_sha256", self.classifier_profile_sha256),
            ("classifier_model_sha256", self.classifier_model_sha256),
            ("tenant_bucket_sha256", self.tenant_bucket_sha256),
            ("source_labels_sha256", self.source_labels_sha256),
            ("source_refs_sha256", self.source_refs_sha256),
            ("flow_snapshot_sha256", self.flow_snapshot_sha256),
        ):
            _sha256(f"semantic binding {label}", digest)
        for label, digest in (
            ("sink_identity_sha256", self.sink_identity_sha256),
            ("tool_schema_sha256", self.tool_schema_sha256),
            ("provider_spec_sha256", self.provider_spec_sha256),
        ):
            _optional_sha256(f"semantic binding {label}", digest)
        if self.target_state_version is not None:
            if type(self.target_state_version) is str:
                _text(
                    "semantic binding target_state_version",
                    self.target_state_version,
                    512,
                )
            elif type(self.target_state_version) is int:
                _nonnegative_int(
                    "semantic binding target_state_version",
                    self.target_state_version,
                )
            else:
                raise TypeError(
                    "semantic binding target_state_version must be a string, integer, or null"
                )
        issued = _timestamp_value("semantic binding issued_at", self.issued_at)
        expires = _timestamp_value("semantic binding expires_at", self.expires_at)
        if expires <= issued:
            raise ValueError("semantic binding expires_at must be after issued_at")
        if (expires - issued).total_seconds() > 300:
            raise ValueError("semantic binding lifetime must not exceed 300 seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_revision": self.request_revision,
            "pid": self.pid,
            "operation_id": self.operation_id,
            "effect_id": self.effect_id,
            "authority_operation": self.authority_operation,
            "resource": self.resource,
            "right": self.right,
            "canonical_args_hash": self.canonical_args_hash,
            "target_state_version": self.target_state_version,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "ceiling_sha256": self.ceiling_sha256,
            "policy_epoch_id": self.policy_epoch_id,
            "policy_epoch_sha256": self.policy_epoch_sha256,
            "control_generation": self.control_generation,
            "assessment_id": self.assessment_id,
            "assessment_sha256": self.assessment_sha256,
            "classifier_profile_sha256": self.classifier_profile_sha256,
            "classifier_model_sha256": self.classifier_model_sha256,
            "tenant_bucket_sha256": self.tenant_bucket_sha256,
            "source_labels_sha256": self.source_labels_sha256,
            "source_refs_sha256": self.source_refs_sha256,
            "flow_snapshot_sha256": self.flow_snapshot_sha256,
            "sink_identity_sha256": self.sink_identity_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "provider_spec_sha256": self.provider_spec_sha256,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticApprovalBindingV2:
        keys = {
            "schema_version",
            "request_id",
            "request_revision",
            "pid",
            "operation_id",
            "effect_id",
            "authority_operation",
            "resource",
            "right",
            "canonical_args_hash",
            "target_state_version",
            "manifest_id",
            "manifest_sha256",
            "ceiling_sha256",
            "policy_epoch_id",
            "policy_epoch_sha256",
            "control_generation",
            "assessment_id",
            "assessment_sha256",
            "classifier_profile_sha256",
            "classifier_model_sha256",
            "tenant_bucket_sha256",
            "source_labels_sha256",
            "source_refs_sha256",
            "flow_snapshot_sha256",
            "sink_identity_sha256",
            "tool_schema_sha256",
            "provider_spec_sha256",
            "nonce",
            "issued_at",
            "expires_at",
        }
        _exact(value, keys, "semantic approval binding v2")
        return cls(**dict(value))

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_legacy_effect_binding(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "canonical_args_hash": self.canonical_args_hash,
            "target_state_version": self.target_state_version,
        }


@dataclass(frozen=True, slots=True)
class DeterministicDenyDecision:
    """Host-only denial proof; it cannot represent an approval."""

    request_id: str
    request_revision: int
    pid: str
    effect_id: str
    reason_codes: tuple[SemanticReasonCode, ...]
    policy_sha256: str
    evidence_sha256: str
    decided_at: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        for label, value in (
            ("request_id", self.request_id),
            ("pid", self.pid),
            ("effect_id", self.effect_id),
        ):
            _text(f"semantic deny {label}", value, 512)
        _nonnegative_int("semantic deny request_revision", self.request_revision)
        object.__setattr__(
            self,
            "reason_codes",
            _enum_tuple(
                SemanticReasonCode,
                self.reason_codes,
                "semantic deny reason_codes",
            ),
        )
        if not self.reason_codes:
            raise ValueError("semantic deterministic deny requires a reason code")
        if any(code.value not in _EXECUTABLE_DENY_REASONS for code in self.reason_codes):
            raise ValueError(
                "semantic deterministic deny contains a human-overridable reason"
            )
        _sha256("semantic deny policy_sha256", self.policy_sha256)
        _sha256("semantic deny evidence_sha256", self.evidence_sha256)
        _timestamp("semantic deny decided_at", self.decided_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "request_revision": self.request_revision,
            "pid": self.pid,
            "effect_id": self.effect_id,
            "reason_codes": [code.value for code in self.reason_codes],
            "policy_sha256": self.policy_sha256,
            "evidence_sha256": self.evidence_sha256,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeterministicDenyDecision:
        _exact(
            value,
            {
                "schema_version",
                "request_id",
                "request_revision",
                "pid",
                "effect_id",
                "reason_codes",
                "policy_sha256",
                "evidence_sha256",
                "decided_at",
            },
            "deterministic deny decision",
        )
        reasons = value["reason_codes"]
        if not isinstance(reasons, list):
            raise TypeError("semantic deny reason_codes must be an array")
        return cls(**{**dict(value), "reason_codes": tuple(reasons)})

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticPreviewLabelsV1:
    """Identity-safe label projection for Human approval surfaces."""

    sensitivity: DataSensitivity
    integrity: DataIntegrity
    trust_level: DataTrustLevel
    identity_present: bool
    identity_mixed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensitivity",
            _enum(DataSensitivity, self.sensitivity, "preview sensitivity"),
        )
        object.__setattr__(
            self,
            "integrity",
            _enum(DataIntegrity, self.integrity, "preview integrity"),
        )
        object.__setattr__(
            self,
            "trust_level",
            _enum(DataTrustLevel, self.trust_level, "preview trust level"),
        )
        if type(self.identity_present) is not bool or type(self.identity_mixed) is not bool:
            raise TypeError("semantic preview identity flags must be booleans")
        if self.identity_mixed and not self.identity_present:
            raise ValueError("mixed semantic preview identity requires identity presence")

    @classmethod
    def from_data_labels(cls, labels: DataLabels) -> SemanticPreviewLabelsV1:
        if not isinstance(labels, DataLabels):
            raise TypeError("semantic preview labels source must be DataLabels")
        return cls(
            sensitivity=labels.sensitivity,
            integrity=labels.integrity,
            trust_level=labels.trust_level,
            identity_present=labels.tenant is not None or labels.principal is not None,
            identity_mixed=labels.is_mixed_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity": self.sensitivity.value,
            "integrity": self.integrity.value,
            "trust_level": self.trust_level.value,
            "identity_present": self.identity_present,
            "identity_mixed": self.identity_mixed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticPreviewLabelsV1:
        _exact(
            value,
            {
                "sensitivity",
                "integrity",
                "trust_level",
                "identity_present",
                "identity_mixed",
            },
            "semantic preview labels",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticApprovalGitReferenceV1:
    """One role-labelled, digest-bound Git reference for Human display."""

    role: str
    display: str
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in _SEMANTIC_APPROVAL_GIT_REFERENCE_ROLES:
            raise ValueError("semantic approval Git reference role is unknown")
        _approval_display_text("semantic approval Git reference display", self.display, 256)
        _sha256("semantic approval Git reference sha256", self.sha256)
        if self.display == _PREVIEW_REDACTED:
            return
        if _PREVIEW_GIT_REFERENCE_RE.fullmatch(self.display) is None:
            raise ValueError("semantic approval Git reference display is malformed")
        if hashlib.sha256(self.display.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("semantic approval Git reference digest is stale")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "display": self.display, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticApprovalGitReferenceV1:
        _exact(value, {"role", "display", "sha256"}, "semantic approval Git reference")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticApprovalArgumentProjectionV1:
    """Bounded, prose-free Host facts for one exact approval argument set.

    Every wire instance has one exact key set.  Variant-specific validation
    prevents a producer from smuggling an arbitrary context field into a
    Human-facing surface.  Raw payloads, file contents, prompts, responses,
    command strings, and remote parameters are intentionally unrepresentable.
    """

    kind: SemanticApprovalArgumentKind
    operation: str
    display_argv: tuple[str, ...] = ()
    argv_count: int | None = None
    argv_truncated: bool = False
    argv_sha256: str | None = None
    safe_cwd: str | None = None
    cwd_sha256: str | None = None
    endpoint_id: str | None = None
    endpoint_id_sha256: str | None = None
    method_id: str | None = None
    method_id_sha256: str | None = None
    server_id: str | None = None
    server_id_sha256: str | None = None
    tool_id: str | None = None
    tool_id_sha256: str | None = None
    registry_spec_sha256: str | None = None
    registry_generation: int | None = None
    payload_sha256: str | None = None
    path_sha256: str | None = None
    content_sha256: str | None = None
    content_bytes: int | None = None
    read_max_bytes: int | None = None
    entry_limit: int | None = None
    text_encoding: str | None = None
    expected_content_sha256: str | None = None
    overwrite: bool | None = None
    parents: bool | None = None
    exist_ok: bool | None = None
    recursive: bool | None = None
    missing_ok: bool | None = None
    timeout_seconds: str | None = None
    continuous_session: bool | None = None
    network_access: bool | None = None
    worktree_id: str | None = None
    worktree_id_sha256: str | None = None
    repository_state_sha256: str | None = None
    source_args_sha256: str | None = None
    git_references: tuple[SemanticApprovalGitReferenceV1, ...] = ()
    git_fact_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(
                SemanticApprovalArgumentKind,
                self.kind,
                "semantic approval argument kind",
            ),
        )
        if _PREVIEW_OPERATION_RE.fullmatch(self.operation) is None:
            raise ValueError("semantic approval operation must be a Host operation id")
        self._validate_common()
        self._validate_variant()

    def _validate_common(self) -> None:
        self._validate_command_fields()
        self._validate_identity_fields()
        self._validate_file_fields_common()
        self._validate_boolean_fields()
        self._validate_git_evidence_fields()

    def _validate_command_fields(self) -> None:
        if not isinstance(self.display_argv, tuple) or len(self.display_argv) > 16:
            raise ValueError("semantic approval display argv must be a tuple of at most 16 items")
        if sum(len(item) for item in self.display_argv) > 1_024:
            raise ValueError("semantic approval display argv exceeds its character budget")
        for item in self.display_argv:
            _approval_display_text("semantic approval argv item", item, 128)
        if self.argv_count is not None:
            _bounded_preview_count("semantic approval argv_count", self.argv_count)
        if type(self.argv_truncated) is not bool:
            raise TypeError("semantic approval argv_truncated must be a boolean")
        _optional_sha256("semantic approval argv_sha256", self.argv_sha256)
        if self.safe_cwd is not None:
            _approval_display_text("semantic approval safe_cwd", self.safe_cwd, 512)
        _optional_sha256("semantic approval cwd_sha256", self.cwd_sha256)

    def _validate_identity_fields(self) -> None:
        for label, value, digest in (
            ("endpoint_id", self.endpoint_id, self.endpoint_id_sha256),
            ("method_id", self.method_id, self.method_id_sha256),
            ("server_id", self.server_id, self.server_id_sha256),
            ("tool_id", self.tool_id, self.tool_id_sha256),
            ("worktree_id", self.worktree_id, self.worktree_id_sha256),
        ):
            _approval_identity_pair(label, value, digest)
        for label, value in (
            ("endpoint_id_sha256", self.endpoint_id_sha256),
            ("method_id_sha256", self.method_id_sha256),
            ("server_id_sha256", self.server_id_sha256),
            ("tool_id_sha256", self.tool_id_sha256),
            ("registry_spec_sha256", self.registry_spec_sha256),
            ("worktree_id_sha256", self.worktree_id_sha256),
            ("payload_sha256", self.payload_sha256),
            ("path_sha256", self.path_sha256),
            ("content_sha256", self.content_sha256),
            ("repository_state_sha256", self.repository_state_sha256),
            ("source_args_sha256", self.source_args_sha256),
        ):
            _optional_sha256(f"semantic approval {label}", value)
        for label, value in (
            ("content_bytes", self.content_bytes),
            ("read_max_bytes", self.read_max_bytes),
            ("entry_limit", self.entry_limit),
            ("registry_generation", self.registry_generation),
        ):
            if value is not None:
                _bounded_preview_count(f"semantic approval {label}", value)
        if (self.registry_spec_sha256 is None) != (self.registry_generation is None):
            raise ValueError(
                "semantic approval registry spec digest and generation must be paired"
            )

    def _validate_file_fields_common(self) -> None:
        if self.text_encoding is not None:
            _approval_display_text(
                "semantic approval text_encoding",
                self.text_encoding,
                64,
            )
        if self.expected_content_sha256 not in {None, "missing"}:
            _sha256(
                "semantic approval expected_content_sha256",
                self.expected_content_sha256,
            )
        if self.timeout_seconds is not None and re.fullmatch(
            r"(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?",
            self.timeout_seconds,
        ) is None:
            raise ValueError("semantic approval timeout_seconds is malformed")

    def _validate_boolean_fields(self) -> None:
        for label, value in (
            ("overwrite", self.overwrite),
            ("parents", self.parents),
            ("exist_ok", self.exist_ok),
            ("recursive", self.recursive),
            ("missing_ok", self.missing_ok),
            ("continuous_session", self.continuous_session),
            ("network_access", self.network_access),
        ):
            if value is not None and type(value) is not bool:
                raise TypeError(f"semantic approval {label} must be a boolean or null")

    def _validate_git_evidence_fields(self) -> None:
        if not isinstance(self.git_references, tuple) or len(self.git_references) > 16:
            raise ValueError("semantic approval Git references must be a tuple of at most 16 items")
        if any(not isinstance(item, SemanticApprovalGitReferenceV1) for item in self.git_references):
            raise TypeError("semantic approval Git references must be typed Host facts")
        roles = tuple(item.role for item in self.git_references)
        if len(roles) != len(set(roles)) or roles != tuple(sorted(roles)):
            raise ValueError("semantic approval Git reference roles must be unique and ordered")
        if not isinstance(self.git_fact_tokens, tuple) or len(self.git_fact_tokens) > 32:
            raise ValueError("semantic approval Git facts must be a tuple of at most 32 tokens")
        if len(self.git_fact_tokens) != len(set(self.git_fact_tokens)):
            raise ValueError("semantic approval Git fact tokens must be unique")
        if tuple(sorted(self.git_fact_tokens)) != self.git_fact_tokens:
            raise ValueError("semantic approval Git fact tokens must be canonically ordered")
        for value in self.git_fact_tokens:
            if re.fullmatch(
                r"[a-z][a-z0-9_]{0,31}=(?:true|false|[a-z][a-z0-9_]{0,31}|[0-9]{1,16}|[0-9a-f]{64})",
                value,
            ) is None:
                raise ValueError("semantic approval Git fact token is malformed")

    def _validate_variant(self) -> None:
        validators = {
            SemanticApprovalArgumentKind.FILESYSTEM: self._validate_filesystem,
            SemanticApprovalArgumentKind.SHELL: self._validate_shell,
            SemanticApprovalArgumentKind.GIT: self._validate_git,
            SemanticApprovalArgumentKind.JSONRPC: self._validate_jsonrpc,
            SemanticApprovalArgumentKind.MCP: self._validate_mcp,
            SemanticApprovalArgumentKind.OTHER: self._validate_other,
        }
        validators[self.kind]()

    def _has_remote_fields(self) -> bool:
        return any(
            (
                self.endpoint_id,
                self.endpoint_id_sha256,
                self.method_id,
                self.method_id_sha256,
                self.server_id,
                self.server_id_sha256,
                self.tool_id,
                self.tool_id_sha256,
                self.registry_spec_sha256,
                self.registry_generation,
            )
        )

    def _has_file_fields(self) -> bool:
        return self.path_sha256 is not None or self._has_filesystem_only_fields()

    def _has_filesystem_only_fields(self) -> bool:
        return any(
            (
                self.content_sha256,
                self.content_bytes,
                self.read_max_bytes,
                self.entry_limit,
                self.text_encoding,
                self.expected_content_sha256,
            )
        ) or any(
            value is not None
            for value in (
                self.overwrite,
                self.parents,
                self.exist_ok,
                self.recursive,
                self.missing_ok,
            )
        )

    def _has_shell_fields(self) -> bool:
        return any(
            (
                self.display_argv,
                self.argv_count,
                self.argv_truncated,
                self.argv_sha256,
                self.safe_cwd,
                self.cwd_sha256,
                self.timeout_seconds,
            )
        ) or self.continuous_session is not None or self.network_access is not None

    def _has_git_fields(self) -> bool:
        return any(
            (
                self.worktree_id,
                self.worktree_id_sha256,
                self.repository_state_sha256,
                self.source_args_sha256,
                self.git_references,
                self.git_fact_tokens,
            )
        )

    def _validate_filesystem(self) -> None:
        invalid = (
            self.path_sha256 is None,
            self.payload_sha256 is not None,
            self._has_remote_fields(),
            self._has_shell_fields(),
            self._has_git_fields(),
        )
        if any(invalid):
            raise ValueError("filesystem approval argument projection has invalid fields")
        if (self.content_sha256 is None) != (self.content_bytes is None):
            raise ValueError("filesystem content digest and byte count must be paired")

    def _validate_shell(self) -> None:
        invalid = (
            self.argv_count is None,
            self.argv_count is not None and self.argv_count <= 0,
            self.argv_sha256 is None,
            self.cwd_sha256 is None,
            self._has_remote_fields(),
            self._has_file_fields(),
            self._has_git_fields(),
            self.payload_sha256 is not None,
        )
        if any(invalid):
            raise ValueError("shell approval argument projection has invalid fields")
        if self.argv_truncated != (self.argv_count > len(self.display_argv)):
            raise ValueError("shell display argv truncation marker is inconsistent")

    def _validate_jsonrpc(self) -> None:
        invalid = (
            self.endpoint_id is None,
            self.endpoint_id_sha256 is None,
            self.method_id is None,
            self.method_id_sha256 is None,
            self.registry_spec_sha256 is None,
            self.registry_generation is None,
            self.payload_sha256 is None,
            self.server_id is not None,
            self.server_id_sha256 is not None,
            self.tool_id is not None,
            self.tool_id_sha256 is not None,
            self._has_shell_fields(),
            self._has_file_fields(),
            self._has_git_fields(),
        )
        if any(invalid):
            raise ValueError("JSON-RPC approval argument projection has invalid fields")

    def _validate_mcp(self) -> None:
        invalid = (
            self.server_id is None,
            self.server_id_sha256 is None,
            self.tool_id is None,
            self.tool_id_sha256 is None,
            self.registry_spec_sha256 is None,
            self.registry_generation is None,
            self.payload_sha256 is None,
            self.endpoint_id is not None,
            self.endpoint_id_sha256 is not None,
            self.method_id is not None,
            self.method_id_sha256 is not None,
            self._has_shell_fields(),
            self._has_file_fields(),
            self._has_git_fields(),
        )
        if any(invalid):
            raise ValueError("MCP approval argument projection has invalid fields")

    def _validate_git(self) -> None:
        invalid = (
            self.worktree_id is None,
            self.worktree_id_sha256 is None,
            self._has_remote_fields(),
            self._has_shell_fields(),
            self._has_filesystem_only_fields(),
            self.payload_sha256 is not None,
        )
        if any(invalid):
            raise ValueError("Git approval argument projection has invalid fields")

    def _validate_other(self) -> None:
        invalid = (
            self._has_remote_fields(),
            self._has_shell_fields(),
            self._has_file_fields(),
            self._has_git_fields(),
            self.payload_sha256 is not None,
        )
        if any(invalid):
            raise ValueError("other approval argument projection must be digest-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "operation": self.operation,
            "display_argv": list(self.display_argv),
            "argv_count": self.argv_count,
            "argv_truncated": self.argv_truncated,
            "argv_sha256": self.argv_sha256,
            "safe_cwd": self.safe_cwd,
            "cwd_sha256": self.cwd_sha256,
            "endpoint_id": self.endpoint_id,
            "endpoint_id_sha256": self.endpoint_id_sha256,
            "method_id": self.method_id,
            "method_id_sha256": self.method_id_sha256,
            "server_id": self.server_id,
            "server_id_sha256": self.server_id_sha256,
            "tool_id": self.tool_id,
            "tool_id_sha256": self.tool_id_sha256,
            "registry_spec_sha256": self.registry_spec_sha256,
            "registry_generation": self.registry_generation,
            "payload_sha256": self.payload_sha256,
            "path_sha256": self.path_sha256,
            "content_sha256": self.content_sha256,
            "content_bytes": self.content_bytes,
            "read_max_bytes": self.read_max_bytes,
            "entry_limit": self.entry_limit,
            "text_encoding": self.text_encoding,
            "expected_content_sha256": self.expected_content_sha256,
            "overwrite": self.overwrite,
            "parents": self.parents,
            "exist_ok": self.exist_ok,
            "recursive": self.recursive,
            "missing_ok": self.missing_ok,
            "timeout_seconds": self.timeout_seconds,
            "continuous_session": self.continuous_session,
            "network_access": self.network_access,
            "worktree_id": self.worktree_id,
            "worktree_id_sha256": self.worktree_id_sha256,
            "repository_state_sha256": self.repository_state_sha256,
            "source_args_sha256": self.source_args_sha256,
            "git_references": [item.to_dict() for item in self.git_references],
            "git_fact_tokens": list(self.git_fact_tokens),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> SemanticApprovalArgumentProjectionV1:
        keys = {
            "kind",
            "operation",
            "display_argv",
            "argv_count",
            "argv_truncated",
            "argv_sha256",
            "safe_cwd",
            "cwd_sha256",
            "endpoint_id",
            "endpoint_id_sha256",
            "method_id",
            "method_id_sha256",
            "server_id",
            "server_id_sha256",
            "tool_id",
            "tool_id_sha256",
            "registry_spec_sha256",
            "registry_generation",
            "payload_sha256",
            "path_sha256",
            "content_sha256",
            "content_bytes",
            "read_max_bytes",
            "entry_limit",
            "text_encoding",
            "expected_content_sha256",
            "overwrite",
            "parents",
            "exist_ok",
            "recursive",
            "missing_ok",
            "timeout_seconds",
            "continuous_session",
            "network_access",
            "worktree_id",
            "worktree_id_sha256",
            "repository_state_sha256",
            "source_args_sha256",
            "git_references",
            "git_fact_tokens",
        }
        _exact(value, keys, "semantic approval argument projection")
        display_argv = value["display_argv"]
        references = value["git_references"]
        git_facts = value["git_fact_tokens"]
        if (
            not isinstance(display_argv, list)
            or not isinstance(references, list)
            or any(not isinstance(item, Mapping) for item in references)
            or not isinstance(git_facts, list)
        ):
            raise TypeError("semantic approval argument arrays are malformed")
        selected = dict(value)
        selected["display_argv"] = tuple(display_argv)
        selected["git_references"] = tuple(
            SemanticApprovalGitReferenceV1.from_dict(item) for item in references
        )
        selected["git_fact_tokens"] = tuple(git_facts)
        return cls(**selected)


@dataclass(frozen=True, slots=True)
class CanonicalApprovalPreviewV1:
    """Host-rendered approval facts bound to a Human response digest."""

    request_id: str
    revision: int
    pid: str
    action_id: str
    resource_display: str
    resource_sha256: str
    rights: tuple[str, ...]
    effect_id: str
    canonical_args_sha256: str
    argument_projection: SemanticApprovalArgumentProjectionV1
    target_state_sha256: str | None
    risk: SemanticPreviewRisk
    source_labels: SemanticPreviewLabelsV1
    expires_at: str | None
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        for label, value in (
            ("request_id", self.request_id),
            ("pid", self.pid),
            ("effect_id", self.effect_id),
        ):
            _approval_display_text(f"semantic preview {label}", value, 512)
        _bounded_preview_count("semantic preview revision", self.revision)
        if (
            len(self.action_id) > 128
            or _ACTION_RE.fullmatch(self.action_id) is None
            or any(
                unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                for character in self.action_id
            )
        ):
            raise ValueError("semantic preview action_id must be an exact operation")
        _sha256("semantic preview resource_sha256", self.resource_sha256)
        _approval_resource_pair(self.resource_display, self.resource_sha256)
        if not isinstance(self.rights, tuple) or not self.rights:
            raise TypeError("semantic preview rights must be a non-empty tuple")
        if len(self.rights) != len(set(self.rights)) or any(
            type(right) is not str or right not in _CAPABILITY_RIGHTS
            for right in self.rights
        ):
            raise ValueError("semantic preview rights must be unique capability rights")
        _sha256(
            "semantic preview canonical_args_sha256",
            self.canonical_args_sha256,
        )
        if not isinstance(
            self.argument_projection,
            SemanticApprovalArgumentProjectionV1,
        ):
            raise TypeError(
                "semantic preview argument_projection must be SemanticApprovalArgumentProjectionV1"
            )
        expected_kind = (
            SemanticApprovalArgumentKind.SHELL
            if self.action_id == "pty.spawn"
            else SemanticApprovalArgumentKind(self.action_id.split(".", 1)[0])
            if self.action_id.split(".", 1)[0]
            in {item.value for item in SemanticApprovalArgumentKind if item is not SemanticApprovalArgumentKind.OTHER}
            else SemanticApprovalArgumentKind.OTHER
        )
        if self.argument_projection.kind is not expected_kind:
            raise ValueError(
                "semantic preview argument projection does not match its Host action"
            )
        _optional_sha256(
            "semantic preview target_state_sha256", self.target_state_sha256
        )
        object.__setattr__(
            self,
            "risk",
            _enum(SemanticPreviewRisk, self.risk, "semantic preview risk"),
        )
        if not isinstance(self.source_labels, SemanticPreviewLabelsV1):
            raise TypeError(
                "semantic preview source_labels must be SemanticPreviewLabelsV1"
            )
        if self.expires_at is not None:
            _timestamp("semantic preview expires_at", self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "revision": self.revision,
            "pid": self.pid,
            "action_id": self.action_id,
            "resource_display": self.resource_display,
            "resource_sha256": self.resource_sha256,
            "rights": list(self.rights),
            "effect_id": self.effect_id,
            "canonical_args_sha256": self.canonical_args_sha256,
            "argument_projection": self.argument_projection.to_dict(),
            "target_state_sha256": self.target_state_sha256,
            "risk": self.risk.value,
            "source_labels": self.source_labels.to_dict(),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalApprovalPreviewV1:
        keys = {
            "schema_version",
            "request_id",
            "revision",
            "pid",
            "action_id",
            "resource_display",
            "resource_sha256",
            "rights",
            "effect_id",
            "canonical_args_sha256",
            "argument_projection",
            "target_state_sha256",
            "risk",
            "source_labels",
            "expires_at",
        }
        _exact(value, keys, "canonical semantic approval preview")
        rights = value["rights"]
        if not isinstance(rights, list):
            raise TypeError("semantic preview rights must be an array")
        selected = dict(value)
        selected["rights"] = tuple(rights)
        labels = selected["source_labels"]
        selected["source_labels"] = SemanticPreviewLabelsV1.from_dict(labels)
        selected["argument_projection"] = SemanticApprovalArgumentProjectionV1.from_dict(
            selected["argument_projection"]
        )
        return cls(**selected)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MachinePolicySettlementV1:
    """Append-only receipt for a Host machine-policy settlement attempt."""

    settlement_id: str
    assessment_id: str | None
    job_id: str | None
    request_id: str
    request_revision: int
    pid: str
    operation_id: str | None
    effect_id: str
    epoch_id: str
    policy_sha256: str
    tenant_bucket_sha256: str
    action_id: str
    outcome: SemanticMachineSettlementOutcome
    capability_id: str | None
    binding_sha256: str
    decision_sha256: str
    matched_rule_id: str | None
    reason_codes: tuple[SemanticReasonCode, ...]
    created_at: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        for label, value in (
            ("settlement_id", self.settlement_id),
            ("request_id", self.request_id),
            ("pid", self.pid),
            ("effect_id", self.effect_id),
            ("epoch_id", self.epoch_id),
        ):
            _text(f"machine settlement {label}", value, 512)
        for label, value in (
            ("assessment_id", self.assessment_id),
            ("job_id", self.job_id),
            ("operation_id", self.operation_id),
            ("capability_id", self.capability_id),
        ):
            _optional_text(f"machine settlement {label}", value, 512)
        _nonnegative_int(
            "machine settlement request_revision", self.request_revision
        )
        if _ACTION_RE.fullmatch(self.action_id) is None:
            raise ValueError("machine settlement action_id must be an exact operation")
        for label, digest in (
            ("policy_sha256", self.policy_sha256),
            ("tenant_bucket_sha256", self.tenant_bucket_sha256),
            ("binding_sha256", self.binding_sha256),
            ("decision_sha256", self.decision_sha256),
        ):
            _sha256(f"machine settlement {label}", digest)
        object.__setattr__(
            self,
            "outcome",
            _enum(
                SemanticMachineSettlementOutcome,
                self.outcome,
                "machine settlement outcome",
            ),
        )
        _optional_text(
            "machine settlement matched_rule_id", self.matched_rule_id, 128
        )
        object.__setattr__(
            self,
            "reason_codes",
            _enum_tuple(
                SemanticReasonCode,
                self.reason_codes,
                "machine settlement reason_codes",
            ),
        )
        issued = self.outcome is SemanticMachineSettlementOutcome.ISSUED
        if issued != (self.capability_id is not None):
            raise ValueError("only an issued settlement may bind a capability")
        if issued and self.matched_rule_id is None:
            raise ValueError("issued settlement requires a matched rule")
        if self.outcome is SemanticMachineSettlementOutcome.DENIED:
            if not self.reason_codes or any(
                code.value not in _EXECUTABLE_DENY_REASONS
                for code in self.reason_codes
            ):
                raise ValueError(
                    "denied settlement requires deterministic non-overridable reasons"
                )
        _timestamp("machine settlement created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "settlement_id": self.settlement_id,
            "assessment_id": self.assessment_id,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "request_revision": self.request_revision,
            "pid": self.pid,
            "operation_id": self.operation_id,
            "effect_id": self.effect_id,
            "epoch_id": self.epoch_id,
            "policy_sha256": self.policy_sha256,
            "tenant_bucket_sha256": self.tenant_bucket_sha256,
            "action_id": self.action_id,
            "outcome": self.outcome.value,
            "capability_id": self.capability_id,
            "binding_sha256": self.binding_sha256,
            "decision_sha256": self.decision_sha256,
            "matched_rule_id": self.matched_rule_id,
            "reason_codes": [code.value for code in self.reason_codes],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MachinePolicySettlementV1:
        keys = {
            "schema_version",
            "settlement_id",
            "assessment_id",
            "job_id",
            "request_id",
            "request_revision",
            "pid",
            "operation_id",
            "effect_id",
            "epoch_id",
            "policy_sha256",
            "tenant_bucket_sha256",
            "action_id",
            "outcome",
            "capability_id",
            "binding_sha256",
            "decision_sha256",
            "matched_rule_id",
            "reason_codes",
            "created_at",
        }
        _exact(value, keys, "machine policy settlement")
        reasons = value["reason_codes"]
        if not isinstance(reasons, list):
            raise TypeError("machine settlement reason_codes must be an array")
        return cls(**{**dict(value), "reason_codes": tuple(reasons)})

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticReviewLabelV1:
    """Append-only Host review evidence for a machine settlement."""

    review_id: str
    settlement_id: str
    outcome: SemanticReviewOutcome
    reviewer_sha256: str
    evidence_sha256: str
    created_at: str
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _text("semantic review_id", self.review_id, 512)
        _text("semantic review settlement_id", self.settlement_id, 512)
        object.__setattr__(
            self,
            "outcome",
            _enum(SemanticReviewOutcome, self.outcome, "semantic review outcome"),
        )
        _sha256("semantic reviewer_sha256", self.reviewer_sha256)
        _sha256("semantic review evidence_sha256", self.evidence_sha256)
        _timestamp("semantic review created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "settlement_id": self.settlement_id,
            "outcome": self.outcome.value,
            "reviewer_sha256": self.reviewer_sha256,
            "evidence_sha256": self.evidence_sha256,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticReviewLabelV1:
        _exact(
            value,
            {
                "schema_version",
                "review_id",
                "settlement_id",
                "outcome",
                "reviewer_sha256",
                "evidence_sha256",
                "created_at",
            },
            "semantic review label",
        )
        return cls(**dict(value))

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SemanticRatioV1:
    numerator: int
    denominator: int
    rate: float | None

    def __post_init__(self) -> None:
        _nonnegative_int("semantic ratio numerator", self.numerator)
        _nonnegative_int("semantic ratio denominator", self.denominator)
        if self.numerator > self.denominator:
            raise ValueError("semantic ratio numerator must not exceed denominator")
        if self.denominator == 0:
            if self.rate is not None:
                raise ValueError("semantic ratio rate must be null for a zero denominator")
            return
        expected = self.numerator / self.denominator
        if (
            type(self.rate) not in {int, float}
            or isinstance(self.rate, bool)
            or not 0.0 <= float(self.rate) <= 1.0
            or abs(float(self.rate) - expected) > 1e-12
        ):
            raise ValueError("semantic ratio rate must equal numerator / denominator")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticRatioV1:
        _exact(value, {"numerator", "denominator", "rate"}, "semantic ratio")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticStatusControlV3:
    catalog_version: int | None
    active_epoch_id: str | None
    active_epoch_sha256: str | None
    generation: int
    state: SemanticPublicControlState
    trip_reason_code: SemanticTripCode | None

    def __post_init__(self) -> None:
        if self.catalog_version is not None and (
            type(self.catalog_version) is not int or self.catalog_version != 1
        ):
            raise ValueError("semantic status catalog_version must be 1 or null")
        _optional_text(
            "semantic status active_epoch_id", self.active_epoch_id, 512
        )
        _optional_sha256(
            "semantic status active_epoch_sha256", self.active_epoch_sha256
        )
        if (self.active_epoch_id is None) != (self.active_epoch_sha256 is None):
            raise ValueError("semantic status active epoch identity must be complete")
        _nonnegative_int("semantic status control generation", self.generation)
        object.__setattr__(
            self,
            "state",
            _enum(
                SemanticPublicControlState,
                self.state,
                "semantic public control state",
            ),
        )
        if self.trip_reason_code is not None:
            object.__setattr__(
                self,
                "trip_reason_code",
                _enum(
                    SemanticTripCode,
                    self.trip_reason_code,
                    "semantic trip reason code",
                ),
            )
        if self.state is SemanticPublicControlState.INACTIVE:
            if self.active_epoch_id is not None or self.trip_reason_code is not None:
                raise ValueError("inactive semantic control must not expose an epoch or trip")
        elif self.active_epoch_id is None or self.catalog_version is None:
            raise ValueError("non-inactive semantic control requires an active epoch")
        if (self.state is SemanticPublicControlState.TRIPPED) != (
            self.trip_reason_code is not None
        ):
            raise ValueError("semantic tripped state and reason must agree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "active_epoch_id": self.active_epoch_id,
            "active_epoch_sha256": self.active_epoch_sha256,
            "generation": self.generation,
            "state": self.state.value,
            "trip_reason_code": (
                self.trip_reason_code.value
                if self.trip_reason_code is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticStatusControlV3:
        _exact(
            value,
            {
                "catalog_version",
                "active_epoch_id",
                "active_epoch_sha256",
                "generation",
                "state",
                "trip_reason_code",
            },
            "semantic status control",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticLegacyFlowHistoryV1:
    """Explicit coverage boundary for assessments created before FlowGraph v6."""

    present: bool = False
    source_schema_version: int | None = None
    assessment_count: int = 0
    coverage: SemanticFlowCoverage | None = None
    evidence_sha256: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise TypeError("semantic legacy flow present must be a boolean")
        _nonnegative_int(
            "semantic legacy flow assessment_count",
            self.assessment_count,
        )
        selected_coverage = (
            None
            if self.coverage is None
            else _enum(
                SemanticFlowCoverage,
                self.coverage,
                "semantic legacy flow coverage",
            )
        )
        object.__setattr__(self, "coverage", selected_coverage)
        if self.present:
            if self.source_schema_version != 5:
                raise ValueError("semantic legacy flow source schema must be 5")
            if selected_coverage is not SemanticFlowCoverage.UNKNOWN:
                raise ValueError("semantic legacy flow coverage must be unknown")
            _sha256("semantic legacy flow evidence", self.evidence_sha256)
            _timestamp("semantic legacy flow created_at", self.created_at)
            return
        if (
            self.source_schema_version is not None
            or self.assessment_count != 0
            or selected_coverage is not None
            or self.evidence_sha256 is not None
            or self.created_at is not None
        ):
            raise ValueError("absent semantic legacy flow history must be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "source_schema_version": self.source_schema_version,
            "assessment_count": self.assessment_count,
            "coverage": self.coverage.value if self.coverage is not None else None,
            "evidence_sha256": self.evidence_sha256,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticLegacyFlowHistoryV1:
        _exact(
            value,
            {
                "present",
                "source_schema_version",
                "assessment_count",
                "coverage",
                "evidence_sha256",
                "created_at",
            },
            "semantic legacy flow history",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticFlowStatusV1:
    available: bool
    counts: Mapping[str, int]
    coverage: Mapping[str, int]
    capture_failures: int
    legacy_history: SemanticLegacyFlowHistoryV1 = field(
        default_factory=SemanticLegacyFlowHistoryV1
    )
    schema_version: int = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        if type(self.available) is not bool:
            raise TypeError("semantic flow available must be a boolean")
        object.__setattr__(
            self,
            "counts",
            _count_mapping(
                self.counts,
                {"entities", "activities", "edges", "label_assertions"},
                "semantic flow counts",
            ),
        )
        object.__setattr__(
            self,
            "coverage",
            _count_mapping(
                self.coverage,
                {item.value for item in SemanticFlowCoverage},
                "semantic flow coverage",
            ),
        )
        _nonnegative_int("semantic flow capture_failures", self.capture_failures)
        if isinstance(self.legacy_history, Mapping):
            object.__setattr__(
                self,
                "legacy_history",
                SemanticLegacyFlowHistoryV1.from_dict(self.legacy_history),
            )
        elif not isinstance(self.legacy_history, SemanticLegacyFlowHistoryV1):
            raise TypeError("semantic flow legacy_history is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "counts": dict(self.counts),
            "coverage": dict(self.coverage),
            "capture_failures": self.capture_failures,
            "legacy_history": self.legacy_history.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticFlowStatusV1:
        required = {
            "schema_version",
            "available",
            "counts",
            "coverage",
            "capture_failures",
        }
        keys = frozenset(value)
        if keys not in {
            frozenset(required),
            frozenset((*required, "legacy_history")),
        }:
            raise ValueError("semantic flow status fields are invalid")
        selected = dict(value)
        selected.setdefault("legacy_history", SemanticLegacyFlowHistoryV1())
        return cls(**selected)


@dataclass(frozen=True, slots=True)
class SemanticReviewMetricsV1:
    reviewed: int
    safe: int
    unsafe: int
    unsafe_rate: float | None
    issued_reviewed: int
    issued_review_rate: float | None

    def __post_init__(self) -> None:
        for label, count in (
            ("reviewed", self.reviewed),
            ("safe", self.safe),
            ("unsafe", self.unsafe),
            ("issued_reviewed", self.issued_reviewed),
        ):
            _nonnegative_int(f"semantic review metrics {label}", count)
        if self.safe + self.unsafe > self.reviewed:
            raise ValueError("semantic reviewed outcomes exceed reviewed count")
        if self.issued_reviewed > self.reviewed:
            raise ValueError("issued semantic reviews exceed all reviewed settlements")
        SemanticRatioV1(
            numerator=self.unsafe,
            denominator=self.reviewed,
            rate=self.unsafe_rate,
        )
        if self.issued_review_rate is not None and (
            type(self.issued_review_rate) not in {int, float}
            or isinstance(self.issued_review_rate, bool)
            or not 0.0 <= float(self.issued_review_rate) <= 1.0
        ):
            raise ValueError(
                "semantic issued review rate must be null or a finite ratio"
            )
        if self.issued_reviewed > 0 and self.issued_review_rate is None:
            raise ValueError(
                "reviewed issued settlements require a non-null review rate"
            )
        if (
            self.issued_reviewed == 0
            and self.issued_review_rate is not None
            and float(self.issued_review_rate) != 0.0
        ):
            raise ValueError(
                "zero reviewed issued settlements require a zero or null rate"
            )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "reviewed": self.reviewed,
            "safe": self.safe,
            "unsafe": self.unsafe,
            "unsafe_rate": self.unsafe_rate,
            "issued_reviewed": self.issued_reviewed,
            "issued_review_rate": self.issued_review_rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticReviewMetricsV1:
        _exact(
            value,
            {
                "reviewed",
                "safe",
                "unsafe",
                "unsafe_rate",
                "issued_reviewed",
                "issued_review_rate",
            },
            "semantic review metrics",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class SemanticStatusV3:
    """Versioned public queue, policy, machine, and review status snapshot."""

    mode: SemanticRuntimeMode
    adapter: str
    profile_id: str | None
    control: SemanticStatusControlV3
    queue: Mapping[str, int]
    assessments: Mapping[str, Any]
    flow: SemanticFlowStatusV1
    machine: Mapping[str, int]
    actual_auto_approval: SemanticRatioV1
    review_metrics: SemanticReviewMetricsV1
    schema_version: int = SEMANTIC_STATUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SEMANTIC_STATUS_SCHEMA_VERSION
        ):
            raise ValueError("unsupported semantic status schema_version")
        object.__setattr__(
            self, "mode", _enum(SemanticRuntimeMode, self.mode, "semantic mode")
        )
        _text("semantic status adapter", self.adapter, 128)
        if self.adapter not in {"deterministic", "scripted", "external"}:
            raise ValueError("semantic status adapter is unsupported")
        _optional_text("semantic status profile_id", self.profile_id, 512)
        if not isinstance(self.control, SemanticStatusControlV3):
            raise TypeError("semantic status control must be SemanticStatusControlV3")
        object.__setattr__(
            self,
            "queue",
            _count_mapping(
                self.queue,
                {
                    "queued",
                    "leased",
                    "succeeded",
                    "failed",
                    "cancelled",
                    "capture_failures",
                },
                "semantic status queue",
            ),
        )
        object.__setattr__(
            self,
            "machine",
            _count_mapping(
                self.machine,
                {
                    "eligible",
                    "issued",
                    "consumed",
                    "succeeded",
                    "failed",
                    "unknown",
                    "expired",
                    "revoked",
                    "race_lost",
                    "denied",
                },
                "semantic status machine counters",
            ),
        )
        object.__setattr__(
            self,
            "assessments",
            _assessment_status_mapping(self.assessments),
        )
        if not isinstance(self.flow, SemanticFlowStatusV1):
            raise TypeError("semantic status flow must be SemanticFlowStatusV1")
        if not isinstance(self.actual_auto_approval, SemanticRatioV1):
            raise TypeError(
                "semantic status actual_auto_approval must be SemanticRatioV1"
            )
        if (
            self.actual_auto_approval.numerator != self.machine["issued"]
            or self.actual_auto_approval.denominator != self.machine["eligible"]
        ):
            raise ValueError(
                "semantic auto-approval ratio must use issued / eligible"
            )
        if not isinstance(self.review_metrics, SemanticReviewMetricsV1):
            raise TypeError(
                "semantic status review_metrics must be SemanticReviewMetricsV1"
            )
        SemanticRatioV1(
            numerator=self.review_metrics.issued_reviewed,
            denominator=self.machine["issued"],
            rate=self.review_metrics.issued_review_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "adapter": self.adapter,
            "profile_id": self.profile_id,
            "control": self.control.to_dict(),
            "queue": dict(self.queue),
            "assessments": dict(self.assessments),
            "flow": self.flow.to_dict(),
            "machine": dict(self.machine),
            "actual_auto_approval": self.actual_auto_approval.to_dict(),
            "review_metrics": self.review_metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticStatusV3:
        _exact(
            value,
            {
                "schema_version",
                "mode",
                "adapter",
                "profile_id",
                "control",
                "queue",
                "assessments",
                "flow",
                "machine",
                "actual_auto_approval",
                "review_metrics",
            },
            "semantic status v3",
        )
        selected = dict(value)
        selected["control"] = SemanticStatusControlV3.from_dict(selected["control"])
        selected["flow"] = SemanticFlowStatusV1.from_dict(selected["flow"])
        selected["actual_auto_approval"] = SemanticRatioV1.from_dict(
            selected["actual_auto_approval"]
        )
        selected["review_metrics"] = SemanticReviewMetricsV1.from_dict(
            selected["review_metrics"]
        )
        return cls(**selected)


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


def _bounded_preview_count(label: str, value: Any) -> None:
    if type(value) is not int or value < 0 or value > 2**53 - 1:
        raise ValueError(f"{label} must be a bounded non-negative integer")


def _approval_display_text(label: str, value: Any, max_chars: int) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > max_chars
        or any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            for character in value
        )
    ):
        raise ValueError(
            f"{label} must be bounded, trimmed, and free of control or format characters"
        )


def _approval_identity_pair(label: str, value: Any, digest: Any) -> None:
    if value is None and digest is None:
        return
    if type(value) is not str or type(digest) is not str:
        raise ValueError(f"semantic approval {label} display and digest must be paired")
    _approval_display_text(f"semantic approval {label}", value, 256)
    _sha256(f"semantic approval {label}_sha256", digest)
    if value != _PREVIEW_REDACTED:
        if _PREVIEW_IDENTITY_RE.fullmatch(value) is None:
            raise ValueError(f"semantic approval {label} display is malformed")
        if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"semantic approval {label} display digest is stale")


def _approval_resource_pair(value: Any, digest: Any) -> None:
    _approval_display_text("semantic preview resource_display", value, 512)
    _sha256("semantic preview resource_sha256", digest)
    if value != _PREVIEW_REDACTED:
        if _PREVIEW_PUBLIC_TOKEN_RE.fullmatch(value) is None:
            raise ValueError("semantic preview resource_display is malformed")
        if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
            raise ValueError("semantic preview resource_display digest is stale")


def _positive_int(label: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _bounded_positive_int(label: str, value: Any, maximum: int) -> None:
    _positive_int(label, value)
    if value > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")


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
    _timestamp_value(label, value)


def _timestamp_value(label: str, value: Any) -> datetime:
    _text(label, value, 64)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _count_mapping(
    value: Mapping[str, int],
    keys: set[str],
    label: str,
) -> dict[str, int]:
    _exact(value, keys, label)
    selected: dict[str, int] = {}
    for key in sorted(keys):
        count = value[key]
        _nonnegative_int(f"{label}.{key}", count)
        selected[key] = count
    return selected


def _assessment_status_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "total",
        "success",
        "error",
        "ood",
        "would_issue_exact_once",
        "would_deny",
        "require_human",
        "by_status",
        "by_domain",
    }
    _exact(value, keys, "semantic status assessments")
    selected: dict[str, Any] = {}
    for key in keys - {"by_status", "by_domain"}:
        count = value[key]
        _nonnegative_int(f"semantic status assessments.{key}", count)
        selected[key] = count
    by_status = _count_mapping(
        value["by_status"],
        {item.value for item in SemanticAssessmentStatus},
        "semantic status assessments.by_status",
    )
    by_domain = _count_mapping(
        value["by_domain"],
        {item.value for item in SemanticDomain},
        "semantic status assessments.by_domain",
    )
    total = selected["total"]
    if sum(by_status.values()) != total or sum(by_domain.values()) != total:
        raise ValueError("semantic assessment status/domain counts must match total")
    if selected["success"] + selected["error"] != total:
        raise ValueError("semantic assessment success/error counts must match total")
    if selected["ood"] != by_status[SemanticAssessmentStatus.OOD.value]:
        raise ValueError("semantic assessment OOD count must match status counts")
    if (
        selected["would_issue_exact_once"]
        + selected["would_deny"]
        + selected["require_human"]
        != total
    ):
        raise ValueError("semantic assessment shadow outcomes must match total")
    selected["by_status"] = by_status
    selected["by_domain"] = by_domain
    return selected


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuthoritativeApprovalFacts",
    "CanonicalApprovalPreviewV1",
    "DeterministicDenyDecision",
    "MachinePolicySettlementV1",
    "SEMANTIC_ACTION_CATALOG_V1",
    "SEMANTIC_ACTION_CATALOG_VERSION",
    "SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION",
    "SEMANTIC_PROVIDER_RESPONSE_SCHEMA",
    "SEMANTIC_REDACTED_INTENT_MAX_CHARS",
    "SEMANTIC_SCHEMA_VERSION",
    "SEMANTIC_STATUS_SCHEMA_VERSION",
    "SemanticApprovalArgumentKind",
    "SemanticApprovalArgumentProjectionV1",
    "SemanticApprovalBindingV2",
    "SemanticApprovalCandidate",
    "SemanticApprovalCandidateSnapshotV1",
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
    "SemanticFlowCoverage",
    "SemanticLegacyFlowHistoryV1",
    "SemanticFlowStatusV1",
    "SemanticHardDenyRuleV1",
    "SemanticMachineSettlementOutcome",
    "SemanticPolicyEpochV1",
    "SemanticPredicate",
    "SemanticPreviewLabelsV1",
    "SemanticPreviewRisk",
    "SemanticPublicControlState",
    "SemanticRatioV1",
    "SemanticReasonCode",
    "SemanticReviewLabelV1",
    "SemanticReviewMetricsV1",
    "SemanticReviewOutcome",
    "SemanticRuntimeMode",
    "SemanticStatusControlV3",
    "SemanticStatusV3",
    "SemanticTripCode",
    "SemanticControlStateV1",
    "ShadowPolicyDecision",
    "ShadowPolicyOutcome",
]
