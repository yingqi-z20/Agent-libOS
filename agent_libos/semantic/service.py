from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    canonical_effect_hash,
    normalize_approval_binding,
)
from agent_libos.config import SemanticDefaults
from agent_libos.models import (
    DataFlowContext,
    DataLabels,
    DataSensitivity,
    HumanRequest,
    HumanRequestStatus,
    ObjectMetadata,
)
from agent_libos.models.data_flow import sensitivity_rank
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
    DeterministicDenyDecision,
    MachinePolicySettlementV1,
    SemanticControlStateV1,
    SemanticApprovalCandidate,
    SemanticApprovalCandidateSnapshotV1,
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
    SemanticReviewLabelV1,
    SemanticReviewOutcome,
    SemanticReviewMetricsV1,
    SemanticRatioV1,
    SemanticRuntimeMode,
    SemanticMachineSettlementOutcome,
    SemanticPolicyEpochV1,
    SemanticLegacyFlowHistoryV1,
    SemanticFlowStatusV1,
    SemanticStatusControlV3,
    SemanticStatusV3,
    SemanticTripCode,
    ShadowPolicyDecision,
    ShadowPolicyOutcome,
)
from agent_libos.semantic.broker import DeterministicApprovalBroker
from agent_libos.semantic.external import (
    HostSemanticAssessmentInvocation,
    SemanticAssessmentDeadlineExceeded,
    SemanticProviderCallError,
    SemanticProviderResponseError,
    SemanticUsageTelemetry,
)
from agent_libos.semantic.enforcement import SemanticRateBudgetExceeded
from agent_libos.semantic.exact_request import (
    ExactSemanticApprovalRequest,
    decode_exact_semantic_approval_request,
    decode_host_human_approval_request,
    semantic_effect_identity,
)
from agent_libos.semantic.labels import validate_monotonic_data_findings
from agent_libos.semantic.ontology import DEFAULT_ACTION_ONTOLOGY
from agent_libos.semantic.flow import (
    provider_result_entity_id,
    root_goal_entity_id,
)
from agent_libos.semantic.projection import (
    LocalDlpAccumulator,
    LocalDlpFinding,
    build_external_projection,
)
from agent_libos.sdk.protected_operations import visit_bounded_host_result_text
from agent_libos.storage import (
    SemanticAssessmentCursor,
    SemanticAssessmentJobRecord,
    SemanticAssessmentJobStatus,
    SemanticAssessmentRecord,
    SemanticHealthEventRecord,
    SemanticHumanOutcomeLinkRecord,
    SemanticProjectionRetention,
    SemanticMachineOutcomeRecord,
    SemanticReviewLabelRecord,
    SemanticV6Cursor,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable


_ACTION_PART = re.compile(r"[^a-z0-9_]+")
_ROOT_INTENT_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_EMPTY_POLICY = {"schema_version": 1, "rules": []}
_ACTIVE_MODES = frozenset({"shadow", "enforce_deny", "canary_auto"})
_ENFORCEMENT_MODES = frozenset({"enforce_deny", "canary_auto"})
_LIFECYCLE_AUTHORITY_FIELDS = frozenset(
    {
        "outcome_id",
        "capability_id",
        "binding_sha256",
        "policy_epoch_id",
        "policy_epoch_sha256",
        "tenant_bucket_sha256",
        "assessment_id",
        "issued_at",
        "expires_at",
        "settlement_id",
        "budget_bucket_id",
        "matched_rule_id",
    }
)
_UNSAFE_REVIEW_CONTROL_MAX_ATTEMPTS = 8
_UNSAFE_REVIEW_UNSETTLED_EVENT = "semantic_unsafe_review_control_unsettled"
_LIVE_DENY_FACT_KEYS = frozenset(
    {
        "schema_version",
        "binding_current",
        "target_state_current",
        "manifest_current",
        "policy_current",
        "data_flow_allowed",
        "evidence_sha256",
    }
)
_ALL_JOB_STATUSES = tuple(item.value for item in SemanticAssessmentJobStatus)
_TERMINAL_ERROR_CODE = {
    SemanticAssessmentStatus.SKIPPED_POLICY: "disabled",
    SemanticAssessmentStatus.EGRESS_BLOCKED: "egress_blocked",
    SemanticAssessmentStatus.TIMEOUT: "timeout",
    SemanticAssessmentStatus.PROVIDER_ERROR: "provider_error",
    SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN: "provider_outcome_unknown",
    SemanticAssessmentStatus.INVALID_SCHEMA: "invalid_schema",
    SemanticAssessmentStatus.OOD: "out_of_distribution",
    SemanticAssessmentStatus.ABSTAINED: "abstained",
    SemanticAssessmentStatus.STALE_INPUT: "stale_input",
}


class _UnsafeReviewControlRace(RuntimeError):
    """The exact durable control row changed before an unsafe review commit."""


class _UnsafeReviewControlFailure(RuntimeError):
    """Unsafe evidence could not be atomically coupled to the control fence."""


@dataclass(frozen=True, slots=True)
class _TransientFlowSnapshot:
    """Process-local external-classifier provenance, never persisted."""

    context: DataFlowContext
    exact_labels_sha256: str


class DeterministicSemanticAssessor:
    """Token-free Host assessor used for Shadow pipeline verification."""

    def assess(self, _request: SemanticAssessmentRequest) -> SemanticAssessment:
        return SemanticAssessment(
            status=SemanticAssessmentStatus.SUCCESS,
            confidence_bps=0,
            calibration_bucket=SemanticCalibrationBucket.UNKNOWN,
            ood=False,
            abstain=False,
        )


def _take_usage_telemetry(assessor: Any) -> SemanticUsageTelemetry | None:
    """Consume optional Host-adapter usage without affecting assessment flow."""

    try:
        take = getattr(assessor, "take_last_usage_telemetry", None)
        if not callable(take):
            return None
        value = take()
    except Exception:
        return None
    return value if type(value) is SemanticUsageTelemetry else None


def _canonical_bytes(value: Any) -> bytes:
    selected = to_jsonable(value)
    return json.dumps(
        selected,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _policy_sha256(value: Mapping[str, Any]) -> str:
    """Match AuthorityManifestManager's persisted policy canonicalization."""

    return hashlib.sha256(dumps(dict(value)).encode("utf-8")).hexdigest()


def _identity_safe_labels_sha256(labels: DataLabels) -> str:
    """Digest semantic label facts without dictionary-guessable identities."""

    return _sha256(
        {
            "schema_version": 1,
            "sensitivity": labels.sensitivity.value,
            "integrity": labels.integrity.value,
            "trust_level": labels.trust_level.value,
            "identity_present": labels.tenant is not None or labels.principal is not None,
            "identity_mixed": labels.is_mixed_identity,
        }
    )


def _root_goal_intent(payload: Any, *, max_chars: int) -> str | None:
    """Extract only the bounded text field from a trusted stored root goal."""

    if type(payload) is str:
        raw = payload
    elif type(payload) is dict and type(payload.get("text")) is str:
        raw = payload["text"]
    else:
        return None
    # The generic projection performs the final whitespace normalization and
    # DLP scan. Never truncate a longer secret before that scan (for example a
    # PEM whose END marker lies after the boundary); downgrade it to
    # metadata-only instead.
    selected = _ROOT_INTENT_CONTROL.sub(" ", raw).strip()
    if len(selected) > max_chars:
        return None
    return selected or None


def _root_goal_dlp_findings(
    payload: Any,
    *,
    input_sha256: str,
) -> tuple[LocalDlpFinding, ...]:
    """Scan only the exact Host-stored root-goal text without retaining it."""

    if type(payload) is str:
        selected = payload
    elif type(payload) is dict and type(payload.get("text")) is str:
        selected = payload["text"]
    else:
        return ()
    detector = LocalDlpAccumulator(input_sha256=input_sha256)
    detector.scan(selected)
    return detector.findings


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("semantic timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _lifecycle_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not value or len(value) > maximum:
        raise ValidationError(f"semantic lifecycle {label} is malformed")
    return value


def _lifecycle_outcome(value: Any) -> tuple[str, str]:
    raw = _lifecycle_text(value, label="outcome", maximum=64)
    assert isinstance(raw, str)
    selected = "outcome_unknown" if raw == "provider_outcome_unknown" else raw
    if selected not in {"consumed", "succeeded", "failed", "outcome_unknown"}:
        raise ValidationError("semantic lifecycle outcome is unsupported")
    return raw, selected


def _lifecycle_authority(value: Any) -> list[Mapping[str, Any]]:
    if (
        type(value) is not list
        or not value
        or len(value) > 16
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise ValidationError("semantic lifecycle authority is malformed")
    return value


def _lifecycle_notification_id(value: Any) -> str:
    selected = _lifecycle_text(
        value,
        label="notification identity",
        maximum=len("semantic-lifecycle:") + 64,
    )
    assert isinstance(selected, str)
    prefix = "semantic-lifecycle:"
    digest = selected.removeprefix(prefix)
    if (
        not selected.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValidationError("semantic lifecycle notification identity is malformed")
    return selected


def _future_timestamp(now: str, seconds: float) -> str:
    return (_parse_time(now) + timedelta(seconds=float(seconds))).isoformat()


def _domain(value: str | None) -> SemanticDomain:
    selected = str(value or "").strip().lower()
    aliases = {"mcp_stdio": "mcp", "llm": "runtime"}
    selected = aliases.get(selected, selected)
    try:
        return SemanticDomain(selected)
    except ValueError:
        return SemanticDomain.UNKNOWN


def _action_id(domain: SemanticDomain, operation: str | None) -> str:
    prefix = domain.value if domain is not SemanticDomain.UNKNOWN else "runtime"
    suffix = _ACTION_PART.sub("_", str(operation or "provider_ingress").lower()).strip("_")
    return f"{prefix}.{suffix or 'provider_ingress'}"


def _candidate_projection(
    candidate: SemanticApprovalCandidate | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return SemanticApprovalCandidateSnapshotV1.from_candidate(
        candidate
    ).to_dict()


def _conservative_labels(projection: Mapping[str, Any]) -> DataLabels:
    labels = projection.get("labels")
    selected = dict(labels) if isinstance(labels, Mapping) else {}
    identity_present = projection.get("identity_present") is True
    sensitivity = DataSensitivity(selected.get("sensitivity", "normal"))
    for _category, _code, _evidence, _severity, minimum in (
        _validated_local_dlp_items(projection)
    ):
        if sensitivity_rank(minimum) > sensitivity_rank(sensitivity):
            sensitivity = minimum
    return DataLabels(
        sensitivity=sensitivity,
        integrity=selected.get("integrity", "unknown"),
        trust_level=selected.get("trust_level", "unknown"),
        origin="derived",
        tenant="mixed" if identity_present else None,
        principal="mixed" if identity_present else None,
    )


def _request_from_job(job: SemanticAssessmentJobRecord) -> SemanticAssessmentRequest:
    projection = job.projection
    digests = projection.get("digests")
    selected_digests = dict(digests) if isinstance(digests, Mapping) else {}
    features = projection.get("features")
    if not isinstance(features, Mapping):
        raise ValidationError("semantic job has no authoritative feature snapshot")
    candidate = projection.get("candidate")
    manifest_id = candidate.get("manifest_id") if isinstance(candidate, Mapping) else None
    return SemanticAssessmentRequest(
        kind=job.kind,
        domain=job.domain,
        action_id=projection.get("action_id"),
        input_sha256=job.bindings["input_sha256"],
        deadline_at=projection.get("deadline_at"),
        data_labels=_conservative_labels(projection),
        features=AuthoritativeApprovalFacts.from_dict(features),
        redacted_intent=projection.get("redacted_intent"),
        pid=job.pid,
        request_id=job.request_id,
        operation_id=job.operation_id,
        effect_id=job.effect_id,
        manifest_id=manifest_id,
        manifest_sha256=job.bindings.get("manifest_sha256"),
        policy_sha256=job.bindings["policy_sha256"],
        resource_sha256=job.bindings.get("resource_sha256"),
        args_sha256=job.bindings.get("args_sha256"),
        state_sha256=job.bindings.get("state_sha256"),
        source_refs_sha256=job.bindings.get("source_refs_sha256"),
        data_labels_sha256=job.bindings.get("data_labels_sha256"),
        sink_identity_sha256=job.bindings.get("sink_identity_sha256"),
        tool_schema_sha256=job.bindings.get("tool_schema_sha256"),
        provider_spec_sha256=job.bindings.get("provider_spec_sha256"),
    )


def _job_status(status: SemanticAssessmentStatus) -> SemanticAssessmentJobStatus:
    if status is SemanticAssessmentStatus.SUCCESS:
        return SemanticAssessmentJobStatus.SUCCEEDED
    if status is SemanticAssessmentStatus.SKIPPED_POLICY:
        # A worker may claim a job and then prove, from Host-owned metadata,
        # that the request is outside the frozen auto-approval catalog before
        # any classifier call.  Preserve that no-call terminal as a
        # cancellation instead of misclassifying it as a provider failure.
        return SemanticAssessmentJobStatus.CANCELLED
    if status is SemanticAssessmentStatus.EGRESS_BLOCKED:
        return SemanticAssessmentJobStatus.EGRESS_BLOCKED
    if status is SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN:
        return SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN
    return SemanticAssessmentJobStatus.FAILED


def _normalize_assessment(assessment: SemanticAssessment) -> SemanticAssessment:
    if assessment.status is not SemanticAssessmentStatus.SUCCESS:
        return assessment
    if assessment.ood:
        return replace(assessment, status=SemanticAssessmentStatus.OOD)
    if assessment.abstain:
        return replace(assessment, status=SemanticAssessmentStatus.ABSTAINED)
    return assessment


_LOCAL_DLP_PROFILES = {
    (
        SemanticDataCategory.CREDENTIAL,
        SemanticReasonCode.CREDENTIAL_MATERIAL,
    ): (SemanticFindingSeverity.HIGH, DataSensitivity.SECRET),
    (
        SemanticDataCategory.BUSINESS_SECRET,
        SemanticReasonCode.SENSITIVE_DATA,
    ): (SemanticFindingSeverity.MEDIUM, DataSensitivity.CONFIDENTIAL),
}
_LOCAL_DLP_LOCATORS = {
    SemanticAssessmentKind.APPROVAL.value: SemanticDataLocator.APPROVAL_REQUEST,
    SemanticAssessmentKind.ROOT_GOAL.value: SemanticDataLocator.ROOT_GOAL,
    SemanticAssessmentKind.PROVIDER_INGRESS.value: SemanticDataLocator.PROVIDER_RESULT,
}


def _validated_local_dlp_items(
    projection: Mapping[str, Any],
) -> tuple[
    tuple[
        SemanticDataCategory,
        SemanticReasonCode,
        str,
        SemanticFindingSeverity,
        DataSensitivity,
    ],
    ...,
]:
    raw = projection.get("dlp_findings", [])
    if not isinstance(raw, list) or len(raw) > 4:
        raise ValidationError("semantic local DLP evidence is malformed")
    selected: list[
        tuple[
            SemanticDataCategory,
            SemanticReasonCode,
            str,
            SemanticFindingSeverity,
            DataSensitivity,
        ]
    ] = []
    seen: set[tuple[SemanticDataCategory, SemanticReasonCode, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "category",
            "code",
            "evidence_sha256",
        }:
            raise ValidationError("semantic local DLP evidence is malformed")
        try:
            category = SemanticDataCategory(item["category"])
            code = SemanticReasonCode(item["code"])
            severity, minimum_sensitivity = _LOCAL_DLP_PROFILES[(category, code)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("semantic local DLP evidence is not allowlisted") from exc
        evidence_sha256 = item["evidence_sha256"]
        if (
            not isinstance(evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
        ):
            raise ValidationError("semantic local DLP evidence digest is malformed")
        key = (category, code, evidence_sha256)
        if key in seen:
            raise ValidationError("semantic local DLP evidence is duplicated")
        seen.add(key)
        selected.append(
            (category, code, evidence_sha256, severity, minimum_sensitivity)
        )
    return tuple(selected)


def _local_dlp_assessment_findings(
    job: SemanticAssessmentJobRecord,
) -> tuple[tuple[SemanticFinding, ...], tuple[SemanticDataFinding, ...]]:
    try:
        locator = _LOCAL_DLP_LOCATORS[job.kind]
    except KeyError as exc:  # pragma: no cover - storage validates job kinds
        raise ValidationError("semantic local DLP locator is unavailable") from exc
    labels = _conservative_labels(job.projection)
    findings: list[SemanticFinding] = []
    data_findings: list[SemanticDataFinding] = []
    for category, code, evidence_sha256, severity, minimum_sensitivity in (
        _validated_local_dlp_items(job.projection)
    ):
        sensitivity_floor = (
            labels.sensitivity
            if sensitivity_rank(labels.sensitivity)
            >= sensitivity_rank(minimum_sensitivity)
            else minimum_sensitivity
        )
        findings.append(
            SemanticFinding(
                code=code,
                severity=severity,
                confidence_bps=10_000,
                evidence_sha256=evidence_sha256,
                source=SemanticFindingSource.HOST,
            )
        )
        data_findings.append(
            SemanticDataFinding(
                category=category,
                field=locator,
                span_start=None,
                span_end=None,
                sensitivity_floor=sensitivity_floor,
                integrity_ceiling=labels.integrity,
                trust_ceiling=labels.trust_level,
                confidence_bps=10_000,
                evidence_sha256=evidence_sha256,
            )
        )
    return tuple(findings), tuple(data_findings)


def _merge_local_dlp_assessment(
    job: SemanticAssessmentJobRecord,
    assessment: SemanticAssessment,
) -> SemanticAssessment:
    host_findings, host_data_findings = _local_dlp_assessment_findings(job)
    if not host_findings:
        return assessment

    def merged(host: tuple[Any, ...], reported: tuple[Any, ...]) -> tuple[Any, ...]:
        selected = list(host)
        for item in reported:
            if item not in selected and len(selected) < 64:
                selected.append(item)
        return tuple(selected)

    result = replace(
        assessment,
        findings=merged(host_findings, assessment.findings),
        data_findings=merged(host_data_findings, assessment.data_findings),
    )
    validate_monotonic_data_findings(
        _conservative_labels(job.projection),
        result.data_findings,
    )
    return result


class SemanticManager:
    """Host-owned durable semantic worker and read-only evidence facade.

    ``shadow`` and the two enforcement modes share one capture/assessment
    pipeline.  Authority is deliberately outside this class: an injected
    Host-only settlement port may consume a completed assessment after all
    live predicates have been revalidated.  This keeps classifier adapters,
    Skills, and runtime modules unable to reach Human or Capability mutation
    surfaces directly.
    """

    def __init__(
        self,
        repository: Any,
        *,
        config: SemanticDefaults,
        assessor: Any | None = None,
        broker: DeterministicApprovalBroker | None = None,
        authority: Any | None = None,
        processes: Any | None = None,
        objects: Any | None = None,
        human_outcome_reader: Callable[[str], str | None] | None = None,
        human_request_reader: Callable[[str], HumanRequest | None] | None = None,
        root_goal_reader: Callable[[str], Any | None] | None = None,
        root_flow_resolver: Callable[[Any], DataFlowContext] | None = None,
        flow_graph: Any | None = None,
        auto_settlement: Any | None = None,
        deny_settlement: Any | None = None,
        control: Any | None = None,
        rate_budget: Any | None = None,
        classifier_id: str | None = None,
        classifier_version: str = "1",
        artifact_sha256: str | None = None,
        owner_id: str | None = None,
        tenant_bucketer: Callable[[str], str] | None = None,
        shutdown_registrar: Callable[[Callable[[], bool]], None] | None = None,
        request_capture_registrar: Callable[[Callable[[Any], None]], None] | None = None,
        spawn_observer_registrar: Callable[[Callable[..., None]], None] | None = None,
        result_observer_registrar: Callable[[Callable[..., None]], None] | None = None,
        request_capture: Callable[[Any], None] | None = None,
        spawn_observer: Callable[..., None] | None = None,
        result_observer: Callable[..., None] | None = None,
        hard_deny_facts_resolver: Callable[
            [HumanRequest, SemanticAssessmentJobRecord | None],
            Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._assessor = assessor or DeterministicSemanticAssessor()
        self._broker = broker or DeterministicApprovalBroker()
        self._authority = authority
        self._processes = processes
        self._objects = objects
        self._human_outcome_reader = human_outcome_reader
        self._human_request_reader = human_request_reader
        self._root_goal_reader = root_goal_reader
        if root_flow_resolver is not None and not callable(root_flow_resolver):
            raise TypeError("semantic root flow resolver must be callable")
        self._root_flow_resolver = root_flow_resolver
        self._flow_graph = flow_graph
        self._auto_settlement = auto_settlement
        self._deny_settlement = deny_settlement
        self._control = control
        self._rate_budget = rate_budget
        if hard_deny_facts_resolver is not None and not callable(
            hard_deny_facts_resolver
        ):
            raise TypeError("semantic hard-deny facts resolver must be callable")
        self._hard_deny_facts_resolver = hard_deny_facts_resolver
        self._mode = config.mode
        self._adapter = config.adapter
        self._profile_id = config.external_profile_id
        self._classifier_id = classifier_id or f"semantic.{config.adapter}"
        self._classifier_version = str(classifier_version)
        self._artifact_sha256 = artifact_sha256 or _sha256(
            {"adapter": config.adapter, "schema_version": 1}
        )
        self._owner_id = owner_id or new_id("semantic-worker")
        if tenant_bucketer is not None and not callable(tenant_bucketer):
            raise TypeError("semantic tenant bucketer must be callable")
        self._tenant_bucketer = tenant_bucketer
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self._maintenance_lock = threading.Lock()
        # FlowGraph observers can run while a business-store transaction is
        # open.  Their kill-switch read must therefore never acquire the
        # manager lock: workers intentionally hold that lock while claiming a
        # job from the same store.  A threading.Event gives those observers a
        # lock-order-independent, fail-closed capture fence.
        self._capture_enabled_event = threading.Event()
        if config.mode in _ACTIVE_MODES:
            self._capture_enabled_event.set()
        # Capture diagnostics use a separate lock for the same reason.  A
        # failed observer must not turn an already committed business
        # transaction into a manager-lock/store-lock inversion.
        self._capture_failure_lock = threading.Lock()
        # Full source references are intentionally process-local and bounded.
        # They never enter job projections, assessments, events, or audit.
        self._transient_context_limit = max(
            64,
            int(config.recovery_batch_limit),
        )
        self._transient_contexts: OrderedDict[
            str,
            _TransientFlowSnapshot,
        ] = OrderedDict()
        self._capture_failures = 0
        # A process-local fail-closed fence for the narrow failure window in
        # which unsafe review evidence was durably retained but its control
        # trip could not be linearized.  Durable health evidence makes the
        # same condition fail closed again during the next startup admission.
        self._unsafe_review_latched = False
        if shutdown_registrar is not None:
            shutdown_registrar(self.shutdown)
        if config.mode in _ACTIVE_MODES:
            registrations = (
                (request_capture_registrar, request_capture),
                (spawn_observer_registrar, spawn_observer),
                (result_observer_registrar, result_observer),
            )
            if any(registrar is None or callback is None for registrar, callback in registrations):
                raise RuntimeError(
                    "semantic active runtime observers are not fully configured"
                )
            for registrar, callback in registrations:
                assert registrar is not None and callback is not None
                registrar(callback)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def capture_enabled(self) -> bool:
        """Return the live observational capture fence without manager locks."""

        return self._capture_enabled_event.is_set()

    def start(self) -> None:
        with self._lock:
            mode = self._mode
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            if mode in _ACTIVE_MODES and self._threads:
                return
            if mode in _ACTIVE_MODES:
                self._stop.clear()
        maintained = self._run_maintenance_batch(mode)
        with self._lock:
            if self._mode not in _ACTIVE_MODES:
                if self._mode == "off" and maintained and not self._threads:
                    janitor = threading.Thread(
                        target=self._worker,
                        name="agent-libos-semantic-janitor",
                        daemon=True,
                    )
                    janitor.start()
                    self._threads.append(janitor)
                return
            if self._threads:
                return
            try:
                for index in range(self._config.max_concurrency):
                    thread = threading.Thread(
                        target=self._worker,
                        name=f"agent-libos-semantic-{index}",
                        daemon=True,
                    )
                    thread.start()
                    self._threads.append(thread)
            except BaseException:
                # Any already-started workers observe off before they can
                # claim. Unstarted Thread objects are never retained/joined.
                self._mode = "off"
                self._capture_enabled_event.clear()
                self._wake.set()
                raise

    def shutdown(self) -> bool:
        with self._lock:
            threads = tuple(self._threads)
            self._stop.set()
            self._wake.set()
        timeout = float(self._config.shutdown_join_timeout_s)
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        stopped = all(not thread.is_alive() for thread in threads)
        if stopped:
            with self._lock:
                self._threads.clear()
        with self._lock:
            self._transient_contexts.clear()
        return stopped

    def set_mode(self, mode: str) -> None:
        if mode not in {"off", *_ACTIVE_MODES}:
            raise ValueError(
                "semantic mode must be off, shadow, enforce_deny, or canary_auto"
            )
        # Durable authority must be disabled before the in-memory mode becomes
        # observable.  Otherwise an unconsumed BindingV2 could pass a live
        # validator between the local kill switch and control-pointer CAS.
        if mode == "off" and self._control is not None:
            disable = getattr(self._control, "disable", None)
            if not callable(disable):
                raise RuntimeError("semantic durable kill switch is unavailable")
            disable("off")
        if mode == "off":
            # Clear before acquiring the manager lock so a FlowGraph observer
            # already running under a store transaction can never wait on a
            # worker that is itself waiting for that transaction.
            self._capture_enabled_event.clear()
        with self._lock:
            previous = self._mode
            if mode == "off":
                self._transient_contexts.clear()
            if previous == "off" and mode in _ACTIVE_MODES:
                raise RuntimeError(
                    "semantic enablement requires Runtime restart and admission"
                )
            if previous in _ACTIVE_MODES and mode in _ACTIVE_MODES and mode != previous:
                raise RuntimeError(
                    "semantic active mode changes require Runtime restart and admission"
                )
            self._mode = mode
            if mode == "off":
                # This is the linearization point: capture and claim both
                # recheck mode under this lock and are disabled immediately.
                self._wake.set()
            elif mode != previous:
                self.start()
        if mode == "off":
            self._run_maintenance_batch("off")

    def degrade_after_startup_failure(self) -> None:
        """Disable capture/claim after a worker or recovery startup failure."""

        # Shadow startup degradation is a durable kill-switch transition so
        # status/reopen cannot disagree with the in-memory capture mode.  An
        # inability to clear durable authority is itself fail-closed.
        if self._control is not None:
            self._control.disable("off")
        self._capture_enabled_event.clear()
        with self._lock:
            self._mode = "off"
            self._wake.set()
        self._increment_capture_failure()
        try:
            maintained = self._run_maintenance_batch("off")
        except Exception:
            self.record_capture_failure(source="startup_recovery")
            return
        if maintained:
            with self._lock:
                self._threads = [
                    thread for thread in self._threads if thread.is_alive()
                ]
                if not self._threads:
                    try:
                        janitor = threading.Thread(
                            target=self._worker,
                            name="agent-libos-semantic-janitor",
                            daemon=True,
                        )
                        janitor.start()
                        self._threads.append(janitor)
                    except Exception:
                        self._increment_capture_failure()

    def record_capture_failure(self, **metadata: Any) -> None:
        self._increment_capture_failure()
        source = metadata.get("source")
        error_type = metadata.get("error_type")
        source_value = source if type(source) is str else type(source).__name__
        error_value = (
            error_type
            if type(error_type) is str
            else type(error_type).__name__
        )
        evidence_sha256 = _sha256(
            {
                "schema_version": 1,
                "source_sha256": _sha256(source_value[:128]),
                "error_type_sha256": _sha256(error_value[:128]),
            }
        )
        append = getattr(
            self._repository,
            "append_semantic_health_event",
            None,
        )
        if not callable(append):
            return
        try:
            # This diagnostic is commonly emitted from an already-open
            # business UoW.  Isolate it in a nested transaction so a failed
            # PostgreSQL statement is rolled back to a savepoint before the
            # failure is swallowed; otherwise the outer Human transaction
            # would remain aborted and observational health evidence could
            # veto the real terminal outcome.
            with self._repository.transaction():
                append(
                    SemanticHealthEventRecord(
                        event_id=new_id("semantic_health"),
                        event_kind="capture_failed",
                        severity="warning",
                        epoch_id=None,
                        tenant_bucket_sha256=None,
                        evidence_sha256=evidence_sha256,
                        created_at=utc_now(),
                    )
                )
        except Exception:
            # Capture diagnostics are observational and cannot mask the
            # already committed business operation or original evidence error.
            return

    def host_tenant_bucket(self, tenant: str | None) -> str | None:
        """Narrow Host composition port for keyed tenant pseudonyms."""

        return self._tenant_bucket(tenant)

    def prepare_host_approval(
        self,
        request: HumanRequest,
        input_sha256: str,
    ) -> tuple[
        SemanticAssessmentRequest,
        SemanticApprovalCandidate | None,
        tuple[SemanticReasonCode, ...],
    ]:
        """Narrow Host-only live approval parser used by settlement wiring."""

        return self._prepare_human_approval(request, input_sha256)

    def host_human_flow(self, payload: Mapping[str, Any]) -> DataFlowContext:
        """Return the authoritative Human request flow context to Host wiring."""

        return self._human_flow(payload)

    def record_machine_lifecycle(self, observation: Any) -> None:
        """Persist one BindingV2 lifecycle notification and settle its budget.

        Consumed evidence owns a separate append-only slot.  All terminal
        outcomes share the SDK-provided terminal ``outcome_id``; only the
        first exact append may release an inflight budget reservation.
        """

        raw_outcome, outcome, effect_id, pid, authority = self._decode_lifecycle(
            observation
        )
        evidence_sha256 = _sha256(
            {
                "schema_version": 1,
                "notification_id": observation.get("notification_id"),
                "outcome": raw_outcome,
                "effect_id": effect_id,
                "phase": observation.get("phase"),
                "error_type": observation.get("error_type"),
            }
        )
        if raw_outcome == "provider_outcome_unknown":
            self._trip_unknown_lifecycle(authority, evidence_sha256)
        with self._repository.transaction():
            for item in authority:
                self._record_lifecycle_authority(
                    item,
                    effect_id=effect_id,
                    pid=pid,
                    outcome=outcome,
                    evidence_sha256=evidence_sha256,
                )

    @staticmethod
    def _decode_lifecycle(
        observation: Any,
    ) -> tuple[str, str, str, str, list[Any]]:
        if not isinstance(observation, Mapping) or set(observation) != {
            "schema_version",
            "outcome",
            "effect_id",
            "authority",
            "notification_id",
            "pid",
            "contract_name",
            "phase",
            "error_type",
        }:
            raise ValidationError("semantic lifecycle notification is malformed")
        if type(observation.get("schema_version")) is not int or observation.get(
            "schema_version"
        ) != 1:
            raise ValidationError("semantic lifecycle schema is unsupported")
        raw_outcome, outcome = _lifecycle_outcome(observation.get("outcome"))
        effect_id = _lifecycle_text(
            observation.get("effect_id"),
            label="effect identity",
            maximum=512,
        )
        pid = _lifecycle_text(
            observation.get("pid"),
            label="process identity",
            maximum=512,
        )
        authority = _lifecycle_authority(observation.get("authority"))
        notification_id = _lifecycle_notification_id(
            observation.get("notification_id")
        )
        _lifecycle_text(
            observation.get("contract_name"),
            label="contract name",
            maximum=256,
        )
        _lifecycle_text(
            observation.get("phase"),
            label="phase",
            maximum=128,
        )
        _lifecycle_text(
            observation.get("error_type"),
            label="error type",
            maximum=128,
            optional=True,
        )
        assert isinstance(effect_id, str) and isinstance(pid, str)
        expected_notification_id = "semantic-lifecycle:" + _sha256(
            {
                "schema_version": 1,
                "outcome": raw_outcome,
                "effect_id": effect_id,
                "authority": [dict(item) for item in authority],
            }
        )
        if notification_id != expected_notification_id:
            raise ValidationError("semantic lifecycle notification identity changed")
        return raw_outcome, outcome, effect_id, pid, authority

    def _trip_unknown_lifecycle(
        self,
        authority: list[Any],
        evidence_sha256: str,
    ) -> None:
        if self._control is None:
            raise ValidationError(
                "semantic unknown outcome cannot trip missing control"
            )
        first = authority[0]
        tenant = (
            first.get("tenant_bucket_sha256")
            if isinstance(first, Mapping)
            else None
        )
        # Commit the kill switch before touching the terminal outcome slot.
        self._control.trip(
            SemanticTripCode.PROVIDER_OUTCOME_UNKNOWN,
            evidence_sha256=evidence_sha256,
            tenant_bucket_sha256=tenant if isinstance(tenant, str) else None,
        )

    def _record_lifecycle_authority(
        self,
        item: Any,
        *,
        effect_id: str,
        pid: str,
        outcome: str,
        evidence_sha256: str,
    ) -> None:
        if (
            not isinstance(item, Mapping)
            or set(item) != _LIFECYCLE_AUTHORITY_FIELDS
        ):
            raise ValidationError("semantic lifecycle authority item is malformed")
        values = tuple(
            item.get(key)
            for key in (
                "outcome_id",
                "settlement_id",
                "budget_bucket_id",
                "issued_at",
            )
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ValidationError(
                "semantic lifecycle authority identity is incomplete"
            )
        outcome_id, settlement_id, budget_bucket_id, issued_at = values
        if outcome_id != self._lifecycle_outcome_id(
            item,
            effect_id=effect_id,
            outcome=outcome,
        ):
            raise ValidationError("semantic lifecycle outcome identity changed")
        try:
            issued = _parse_time(issued_at)
            expires = _parse_time(item["expires_at"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "semantic lifecycle authority lifetime is malformed"
            ) from exc
        if expires <= issued or (expires - issued).total_seconds() > 300:
            raise ValidationError(
                "semantic lifecycle authority lifetime is malformed"
            )
        settlement = self._repository.get_semantic_machine_settlement(
            settlement_id
        )
        if not self._lifecycle_settlement_matches(
            settlement,
            item=item,
            effect_id=effect_id,
            pid=pid,
            budget_bucket_id=budget_bucket_id,
        ):
            raise ValidationError(
                "semantic lifecycle does not match its issued settlement"
            )
        appended = self._repository.append_semantic_machine_outcome_if_absent(
            SemanticMachineOutcomeRecord(
                outcome_id=outcome_id,
                settlement_id=settlement_id,
                effect_id=effect_id,
                outcome=outcome,
                evidence_sha256=evidence_sha256,
                created_at=settlement.created_at,
            )
        )
        if appended and outcome != "consumed":
            if self._rate_budget is None:
                raise ValidationError(
                    "semantic lifecycle budget bridge is unavailable"
                )
            self._rate_budget.release(budget_bucket_id)

    def _lifecycle_settlement_matches(
        self,
        settlement: Any,
        *,
        item: Mapping[str, Any],
        effect_id: str,
        pid: str,
        budget_bucket_id: str,
    ) -> bool:
        if settlement is None or self._rate_budget is None:
            return False
        matched_rule_id = settlement.matched_rule_id
        if not isinstance(matched_rule_id, str):
            return False
        expected_budget_id = self._rate_budget.bucket_id_for(
            epoch_id=settlement.epoch_id,
            tenant_bucket_sha256=settlement.tenant_bucket_sha256,
            rule_id=matched_rule_id,
        )
        return bool(
            settlement.outcome == "issued"
            and settlement.pid == pid
            and settlement.effect_id == effect_id
            and settlement.capability_id == item.get("capability_id")
            and settlement.binding_sha256 == item.get("binding_sha256")
            and settlement.assessment_id == item.get("assessment_id")
            and settlement.epoch_id == item.get("policy_epoch_id")
            and settlement.policy_sha256 == item.get("policy_epoch_sha256")
            and settlement.tenant_bucket_sha256
            == item.get("tenant_bucket_sha256")
            and matched_rule_id == item.get("matched_rule_id")
            and budget_bucket_id == expected_budget_id
        )

    @staticmethod
    def _lifecycle_outcome_id(
        item: Mapping[str, Any],
        *,
        effect_id: str,
        outcome: str,
    ) -> str:
        slot = "consumed" if outcome == "consumed" else "terminal"
        return "semantic-outcome:" + _sha256(
            {
                "schema_version": 1,
                "lifecycle_slot": slot,
                "effect_id": effect_id,
                "settlement_id": item.get("settlement_id"),
                "capability_id": item.get("capability_id"),
                "binding_sha256": item.get("binding_sha256"),
            }
        )

    def capture_approval(
        self,
        request: HumanRequest,
    ) -> SemanticAssessmentJobRecord | None:
        return self.capture_human_request(request)

    def deterministic_deny_preflight(
        self,
        request: HumanRequest,
    ) -> DeterministicDenyDecision | None:
        """Return a live Host-only hard-deny proof, never an allow decision.

        Unsupported/high-risk actions, ceiling misses, classifier outcomes,
        and uncertainty intentionally return ``None`` so the request remains
        with Human.  The only executable reasons are exact malformed/stale
        bindings and an explicit immutable Host hard-deny rule.
        """

        if not isinstance(request, HumanRequest):
            raise TypeError("semantic deny preflight requires HumanRequest")
        epoch = self._active_deny_epoch()
        if epoch is None:
            return None
        payload = request.payload
        if not self._is_external_approval_payload(payload):
            return None
        try:
            input_sha256 = _sha256(payload)
        except Exception:
            # A value that cannot be canonically bound is uncertainty at this
            # boundary. Human request admission already bounds persisted JSON;
            # never translate an unrelated encoder failure into a machine
            # denial.
            return None
        try:
            exact = decode_exact_semantic_approval_request(request)
        except ValidationError:
            try:
                # A Host-bound scoped request is valid Human authority but is
                # deliberately outside the Phase 4 exact-resource catalog.
                # Non-eligibility is uncertainty/require-human, never a
                # deterministic malformed-request rejection.
                decode_host_human_approval_request(request)
            except ValidationError:
                pass
            else:
                return None
            exact = None
            selected_reasons = (SemanticReasonCode.MALFORMED_REQUEST,)
            live_proof_sha256 = None
        except Exception:
            return None
        if exact is not None:
            hard_deny = self._hard_deny_reason(epoch, exact)
            try:
                reasons = self._live_deny_reasons(
                    request=request,
                    input_sha256=input_sha256,
                    exact=exact,
                )
            except Exception:
                reasons = None
            if reasons is None:
                # Live repository/registry/DataFlow uncertainty cannot become
                # a deny predicate.  An exact immutable Host hard-deny rule is
                # independently complete and remains enforceable.
                if not hard_deny:
                    return None
                selected_reasons = ()
                live_proof_sha256 = None
            else:
                selected_reasons, live_proof_sha256 = reasons
        else:
            hard_deny = ()
        selected_reasons = tuple(
            dict.fromkeys((*selected_reasons, *hard_deny))
        )
        if not selected_reasons:
            return None
        return self._deterministic_deny_decision(
            request=request,
            input_sha256=input_sha256,
            exact=exact,
            reasons=selected_reasons,
            epoch=epoch,
            live_proof_sha256=live_proof_sha256,
        )

    def _active_deny_epoch(self) -> Any | None:
        """Return one exact live static/durable epoch or disable machine deny."""

        # Human approval calls this preflight while it owns the shared Store
        # transaction.  Workers deliberately take ``self._lock`` before
        # claiming a job from that Store, so acquiring the manager lock here
        # would invert the order (Store -> manager vs manager -> Store) and
        # could deadlock Human settlement.  These two process-local reads are
        # only an early fail-closed filter: the durable control pointer is
        # re-read below and the terminal settlement fences the same pointer in
        # the caller's transaction before any denial can commit.
        local_mode = self._mode
        unsafe_review_latched = self._unsafe_review_latched
        if (
            unsafe_review_latched
            or local_mode not in _ENFORCEMENT_MODES
            or self._control is None
        ):
            return None
        resolver = getattr(self._control, "authority_view", None)
        if not callable(resolver):
            return None
        try:
            view = resolver()
            control = view.control
            epoch = view.epoch
            static = getattr(self._config, "policy_epoch", None)
            if (
                not isinstance(control, SemanticControlStateV1)
                or not isinstance(epoch, SemanticPolicyEpochV1)
                or not isinstance(static, SemanticPolicyEpochV1)
            ):
                return None
            policy_sha256 = epoch.canonical_sha256()
            static_sha256 = static.canonical_sha256()
        except Exception:
            return None
        if (
            control.tripped
            or control.mode.value != local_mode
            or control.active_epoch_id != epoch.epoch_id
            or control.active_policy_sha256 != policy_sha256
            or control.generation != epoch.generation
            or static.epoch_id != epoch.epoch_id
            or static.generation != epoch.generation
            or static_sha256 != policy_sha256
        ):
            return None
        return epoch

    @staticmethod
    def _is_external_approval_payload(payload: Any) -> bool:
        return bool(
            isinstance(payload, Mapping)
            and payload.get("type") == "external_operation_approval"
        )

    def _live_deny_reasons(
        self,
        *,
        request: HumanRequest,
        input_sha256: str,
        exact: ExactSemanticApprovalRequest,
    ) -> tuple[tuple[SemanticReasonCode, ...], str | None] | None:
        prepared, _candidate, _shadow_violations = self._prepare_human_approval(
            request,
            input_sha256,
        )
        facts = prepared.features
        reasons: list[SemanticReasonCode] = []
        if not facts.schema_valid or not facts.request_is_exact_external_operation:
            reasons.append(SemanticReasonCode.MALFORMED_REQUEST)
        elif not facts.binding_current:
            reasons.append(SemanticReasonCode.STALE_BINDING)
        elif not facts.manifest_current:
            reasons.append(SemanticReasonCode.STALE_MANIFEST)
        elif not facts.policy_current:
            reasons.append(SemanticReasonCode.STALE_POLICY)
        # Keep the strict decode live and explicit even though preparation
        # independently re-decodes it; neither model findings nor ontology
        # eligibility can add an executable denial reason.
        if exact.action_id != prepared.action_id:
            reasons.append(SemanticReasonCode.DIGEST_DRIFT)
        live = self._resolved_live_deny_facts(request)
        if live is None:
            return None
        live_reasons, live_proof_sha256 = live
        return (
            tuple(dict.fromkeys((*reasons, *live_reasons))),
            live_proof_sha256,
        )

    def _resolved_live_deny_facts(
        self,
        request: HumanRequest,
    ) -> tuple[tuple[SemanticReasonCode, ...], str | None] | None:
        resolver = self._hard_deny_facts_resolver
        if resolver is None:
            return (), None
        getter = getattr(
            self._repository,
            "get_semantic_assessment_job_for_request",
            None,
        )
        try:
            job = (
                getter(request.request_id, request.revision)
                if callable(getter)
                else None
            )
            raw = resolver(request, job)
            return self._decode_live_deny_facts(raw)
        except Exception:
            return None

    @staticmethod
    def _decode_live_deny_facts(
        value: Any,
    ) -> tuple[tuple[SemanticReasonCode, ...], str]:
        if not isinstance(value, Mapping) or set(value) != _LIVE_DENY_FACT_KEYS:
            raise ValidationError("semantic live deny facts are malformed")
        if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
            raise ValidationError("semantic live deny facts version is unsupported")
        fields = {
            "binding_current": SemanticReasonCode.STALE_BINDING,
            "target_state_current": SemanticReasonCode.DIGEST_DRIFT,
            "manifest_current": SemanticReasonCode.STALE_MANIFEST,
            "policy_current": SemanticReasonCode.STALE_POLICY,
            "data_flow_allowed": SemanticReasonCode.DATA_FLOW_DENIED,
        }
        if any(
            value[key] is not None and type(value[key]) is not bool
            for key in fields
        ):
            raise ValidationError("semantic live deny fact must be boolean or null")
        evidence_sha256 = value.get("evidence_sha256")
        if (
            not isinstance(evidence_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
        ):
            raise ValidationError("semantic live deny evidence digest is malformed")
        reasons = tuple(reason for key, reason in fields.items() if value[key] is False)
        return reasons, evidence_sha256

    def _hard_deny_reason(
        self,
        epoch: Any,
        exact: ExactSemanticApprovalRequest | None,
    ) -> tuple[SemanticReasonCode, ...]:
        if epoch is None or exact is None:
            return ()
        matched = any(
            self._hard_deny_rule_matches(
                rule,
                action=exact.action_id,
                resource=exact.resource,
                rights=exact.rights,
            )
            for rule in getattr(epoch, "hard_deny_rules", ())
        )
        return (SemanticReasonCode.POLICY_HARD_DENY,) if matched else ()

    def _deterministic_deny_decision(
        self,
        *,
        request: HumanRequest,
        input_sha256: str,
        exact: ExactSemanticApprovalRequest | None,
        reasons: tuple[SemanticReasonCode, ...],
        epoch: Any,
        live_proof_sha256: str | None,
    ) -> DeterministicDenyDecision:
        policy_sha256 = (
            epoch.canonical_sha256()
            if epoch is not None
            else _policy_sha256(_EMPTY_POLICY)
        )
        effect_id = (
            exact.binding["effect_id"]
            if exact is not None
            else semantic_effect_identity(request)
        )
        action = exact.action_id if exact is not None else None
        resource = exact.resource if exact is not None else None
        rights = exact.rights if exact is not None else ()
        return DeterministicDenyDecision(
            request_id=request.request_id,
            request_revision=request.revision,
            pid=request.pid,
            effect_id=effect_id,
            reason_codes=reasons,
            policy_sha256=policy_sha256,
            evidence_sha256=_sha256(
                {
                    "schema_version": 1,
                    "request_id": request.request_id,
                    "request_revision": request.revision,
                    "pid": request.pid,
                    "effect_id": effect_id,
                    "action_sha256": _sha256(action),
                    "resource_sha256": _sha256(resource),
                    "rights_sha256": _sha256(rights),
                    "reason_codes": [item.value for item in reasons],
                    "policy_sha256": policy_sha256,
                    "live_proof_sha256": live_proof_sha256,
                }
            ),
            decided_at=utc_now(),
        )

    @staticmethod
    def _hard_deny_rule_matches(
        rule: Any,
        *,
        action: str,
        resource: str,
        rights: tuple[str, ...],
    ) -> bool:
        if (
            getattr(rule, "authority_operation", None) != action
            or not set(rights).intersection(getattr(rule, "rights", ()))
        ):
            return False
        pattern = getattr(rule, "resource", None)
        if not isinstance(pattern, str):
            return False
        if pattern.endswith("*"):
            return "*" not in resource and resource.startswith(pattern[:-1])
        return pattern == resource

    def capture_human_request(
        self,
        request: HumanRequest,
    ) -> SemanticAssessmentJobRecord | None:
        with self._lock:
            if self._mode not in _ACTIVE_MODES:
                return None
            try:
                return self._capture_human_request(request)
            except Exception:
                self.record_capture_failure(source="approval")
                return None

    def capture_root_goal(
        self,
        pid: str,
        *,
        image_id: str | None = None,
        publication_id: str | None = None,
        goal: Any | None = None,
    ) -> SemanticAssessmentJobRecord | None:
        return self.capture_root_process(
            pid,
            image_id=image_id,
            publication_id=publication_id,
            goal=goal,
        )

    def capture_root_process(
        self,
        pid: str,
        *,
        image_id: str | None = None,
        publication_id: str | None = None,
        goal: Any | None = None,
    ) -> SemanticAssessmentJobRecord | None:
        with self._lock:
            if self._mode not in _ACTIVE_MODES:
                return None
            try:
                return self._capture_root_process(
                    pid,
                    image_id=image_id,
                    publication_id=publication_id,
                    goal=goal,
                )
            except Exception:
                self.record_capture_failure(source="root_goal")
                return None

    def capture_provider_ingress(
        self,
        result: Any,
        observation: Any,
        *,
        data_labels: DataLabels | None = None,
        provider_spec_sha256: str | None = None,
        tool_schema_sha256: str | None = None,
    ) -> SemanticAssessmentJobRecord | None:
        return self.capture_provider_result(
            result,
            observation,
            data_labels=data_labels,
            provider_spec_sha256=provider_spec_sha256,
            tool_schema_sha256=tool_schema_sha256,
        )

    def capture_provider_result(
        self,
        result: Any,
        observation: Any,
        *,
        data_labels: DataLabels | None = None,
        provider_spec_sha256: str | None = None,
        tool_schema_sha256: str | None = None,
    ) -> SemanticAssessmentJobRecord | None:
        with self._lock:
            if self._mode not in _ACTIVE_MODES:
                return None
            try:
                return self._capture_provider_result(
                    result,
                    observation,
                    data_labels=data_labels,
                    provider_spec_sha256=provider_spec_sha256,
                    tool_schema_sha256=tool_schema_sha256,
                )
            except Exception:
                self.record_capture_failure(source="provider_ingress")
                return None

    def process_one(self) -> bool:
        with self._lock:
            mode = self._mode
        maintained = self._run_maintenance_batch(mode)
        # Claim and kill-switch transition are serialized: after ``off`` wins
        # this lock, no worker can durably claim another job.
        with self._lock:
            if self._mode not in _ACTIVE_MODES:
                return maintained
            now = utc_now()
            claimed = self._repository.claim_next_semantic_assessment_job(
                lease_owner_id=self._owner_id,
                lease_id=new_id("semantic-lease"),
                lease_expires_at=_future_timestamp(now, self._config.job_lease_s),
                updated_at=now,
            )
        if claimed is None:
            return maintained
        self._assess_claimed(claimed)
        return True

    def status(self) -> dict[str, Any]:
        aggregate = self._repository.semantic_status_aggregate()
        job_counts = aggregate.job_counts
        queue = {
            "queued": job_counts[SemanticAssessmentJobStatus.QUEUED.value],
            "leased": job_counts[SemanticAssessmentJobStatus.CLAIMED.value],
            "succeeded": job_counts[SemanticAssessmentJobStatus.SUCCEEDED.value],
            "failed": sum(
                job_counts[item.value]
                for item in (
                    SemanticAssessmentJobStatus.FAILED,
                    SemanticAssessmentJobStatus.EGRESS_BLOCKED,
                    SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN,
                    SemanticAssessmentJobStatus.EXPIRED,
                )
            ),
            "cancelled": job_counts[SemanticAssessmentJobStatus.CANCELLED.value],
            "capture_failures": self._capture_failure_count(),
        }
        assessments = {
            "total": aggregate.assessment_total,
            "success": aggregate.assessment_status_counts["success"],
            "error": (
                aggregate.assessment_total
                - aggregate.assessment_status_counts["success"]
            ),
            "ood": aggregate.assessment_ood_count,
            "would_issue_exact_once": aggregate.shadow_outcome_counts[
                "would_issue_exact_once"
            ],
            "would_deny": aggregate.shadow_outcome_counts["would_deny"],
            "require_human": aggregate.shadow_outcome_counts["require_human"],
            "by_status": {
                status.value: aggregate.assessment_status_counts[status.value]
                for status in SemanticAssessmentStatus
            },
            "by_domain": {
                domain.value: aggregate.assessment_domain_counts[domain.value]
                for domain in SemanticDomain
            },
        }
        metrics = self.metrics()
        reviews = metrics["review_metrics"]
        machine = metrics["machine"]
        actual = metrics["actual_auto_approval"]
        reviewed = int(reviews["reviewed"])
        unsafe = int(reviews["unsafe"])
        flow = self.flow_status()
        return SemanticStatusV3(
            mode=SemanticRuntimeMode(self.mode),
            adapter=self._adapter,
            profile_id=self._profile_id,
            control=self._status_control_model(),
            queue=queue,
            assessments=assessments,
            flow=SemanticFlowStatusV1.from_dict(flow),
            machine=machine,
            actual_auto_approval=SemanticRatioV1(
                numerator=int(actual["numerator"]),
                denominator=int(actual["denominator"]),
                rate=actual["rate"],
            ),
            review_metrics=SemanticReviewMetricsV1(
                reviewed=reviewed,
                safe=int(reviews["safe"]),
                unsafe=unsafe,
                unsafe_rate=(None if reviewed == 0 else unsafe / reviewed),
                issued_reviewed=int(reviews["issued_reviewed"]),
                issued_review_rate=reviews["issued_review_rate"],
            ),
        ).to_dict()

    def flow_status(self) -> dict[str, Any]:
        flow = self._flow_graph
        if flow is None:
            return {
                "schema_version": 1,
                "available": False,
                "counts": {
                    "entities": 0,
                    "activities": 0,
                    "edges": 0,
                    "label_assertions": 0,
                },
                "coverage": {
                    "complete": 0,
                    "partial": 0,
                    "unknown": 0,
                    "conflict": 0,
                    "stale": 0,
                },
                "capture_failures": self._capture_failure_count(),
                "legacy_history": SemanticLegacyFlowHistoryV1().to_dict(),
            }
        status = flow.flow_status()
        if not isinstance(status, Mapping):
            raise TypeError("semantic flow status must be a mapping")
        return dict(status)

    def metrics(
        self,
        *,
        window: str | None = None,
        action_id: str | None = None,
        tenant_bucket_sha256: str | None = None,
        epoch_id: str | None = None,
        risk: str | None = None,
    ) -> dict[str, Any]:
        provider = getattr(self._repository, "semantic_metrics", None)
        if callable(provider):
            value = provider(
                window=window,
                action_id=action_id,
                tenant_bucket_sha256=tenant_bucket_sha256,
                epoch_id=epoch_id,
                risk=risk,
            )
            if not isinstance(value, Mapping):
                raise TypeError("semantic metrics repository returned an invalid value")
            selected = dict(value)
            return {
                "schema_version": 1,
                "window": window,
                "action_id": action_id,
                "tenant_bucket_sha256": tenant_bucket_sha256,
                "epoch_id": epoch_id,
                "risk": risk,
                "machine": dict(selected["machine"]),
                "actual_auto_approval": dict(
                    selected["actual_auto_approval"]
                ),
                "review_metrics": dict(selected["review_metrics"]),
            }
        machine = {
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
        }
        return {
            "schema_version": 1,
            "window": window,
            "action_id": action_id,
            "tenant_bucket_sha256": tenant_bucket_sha256,
            "epoch_id": epoch_id,
            "risk": risk,
            "machine": machine,
            "actual_auto_approval": {
                "numerator": 0,
                "denominator": 0,
                "rate": None,
            },
            "review_metrics": {
                "reviewed": 0,
                "safe": 0,
                "unsafe": 0,
                "unsafe_rate": None,
                "issued_reviewed": 0,
                "issued_review_rate": None,
            },
        }

    def control_status(self) -> dict[str, Any]:
        control = self._control
        if control is not None:
            reader = getattr(control, "control_status", None)
            if not callable(reader):
                reader = getattr(control, "status", None)
            if not callable(reader):
                reader = getattr(control, "current", None)
            if callable(reader):
                value = reader()
                to_dict = getattr(value, "to_dict", None)
                if callable(to_dict):
                    value = to_dict()
                if not isinstance(value, Mapping):
                    raise TypeError("semantic control status must be a mapping")
                return dict(value)
        return {
            "schema_version": 1,
            "revision": 0,
            "generation": 0,
            "mode": self.mode,
            "active_epoch_id": None,
            "active_policy_sha256": None,
            "tripped": False,
            "trip_code": None,
            "updated_at": utc_now(),
        }

    def _control_state_model(self) -> SemanticControlStateV1:
        raw = self.control_status()
        try:
            control = SemanticControlStateV1.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError("semantic durable control state is malformed") from exc
        if control.mode.value != self.mode:
            raise ValidationError(
                "semantic runtime mode does not match durable control state"
            )
        return control

    def _status_control_model(self) -> SemanticStatusControlV3:
        control = self._control_state_model()
        epoch = getattr(self._config, "policy_epoch", None)
        with self._lock:
            unsafe_review_latched = self._unsafe_review_latched
        if control.tripped or unsafe_review_latched:
            state = "tripped"
        elif control.mode is SemanticRuntimeMode.CANARY_AUTO:
            state = "active"
        elif control.mode is SemanticRuntimeMode.ENFORCE_DENY:
            state = "active"
        else:
            state = "inactive"
        active_epoch_id = control.active_epoch_id
        active_epoch_sha256 = control.active_policy_sha256
        catalog_version = getattr(epoch, "catalog_version", None)
        if state == "inactive":
            active_epoch_id = None
            active_epoch_sha256 = None
            catalog_version = None
        return SemanticStatusControlV3(
            catalog_version=catalog_version,
            active_epoch_id=active_epoch_id,
            active_epoch_sha256=active_epoch_sha256,
            generation=control.generation,
            state=state,
            trip_reason_code=(
                SemanticTripCode.UNSAFE_REVIEW
                if unsafe_review_latched and control.trip_code is None
                else control.trip_code
            ),
        )

    def query_assessments(
        self,
        *,
        pid: str | None = None,
        request_id: str | None = None,
        operation_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        domain: str | None = None,
        action_id: str | None = None,
        tenant_bucket_sha256: str | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        selected_limit = self._config.assessment_list_limit if limit is None else limit
        hard_limit = min(self._config.assessment_list_hard_limit, 500)
        if (
            isinstance(selected_limit, bool)
            or not isinstance(selected_limit, int)
            or selected_limit < 1
        ):
            raise ValueError("semantic assessment limit must be a positive integer")
        if selected_limit > hard_limit:
            raise ValueError(
                f"semantic assessment limit exceeds hard cap {hard_limit}"
            )
        page = self._repository.query_semantic_assessments(
            after=self._decode_cursor(after),
            limit=selected_limit,
            pid=pid,
            request_id=request_id,
            operation_id=operation_id,
            kind=kind,
            status=status,
            domain=domain,
            action_id=action_id,
            tenant_bucket_sha256=tenant_bucket_sha256,
        )
        links = self._human_outcome_links_for_records(
            page.records,
            resolver_name="semantic_human_outcome_links_for_assessments",
            identity_field="assessment_id",
        )
        items: list[dict[str, Any]] = []
        for record in page.records:
            item = record.to_dict()
            link = links.get(record.assessment_id)
            if link is not None:
                item["human_outcome"] = link.outcome
            items.append(item)
        return {
            "items": items,
            "next_cursor": self._encode_cursor(page.next_cursor),
        }

    def query_flow_entities(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        pid: str | None = None,
        kind: str | None = None,
        tenant_bucket_sha256: str | None = None,
    ) -> dict[str, Any]:
        selected_limit = self._bounded_flow_limit(limit)
        if self._flow_graph is None:
            return {"schema_version": 1, "items": [], "next_cursor": None}
        return self._flow_graph.query_flow_entities(
            after=after,
            limit=selected_limit,
            pid=pid,
            kind=kind,
            tenant_bucket_sha256=tenant_bucket_sha256,
        )

    def query_flow_edges(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        pid: str | None = None,
        relation: str | None = None,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        selected_limit = self._bounded_flow_limit(limit)
        if self._flow_graph is None:
            return {"schema_version": 1, "items": [], "next_cursor": None}
        return self._flow_graph.query_flow_edges(
            after=after,
            limit=selected_limit,
            pid=pid,
            relation=relation,
            node_id=node_id,
        )

    def query_flow_lineage(
        self,
        node_id: str,
        *,
        direction: str = "upstream",
        after: str | None = None,
        limit: int | None = None,
        max_depth: int = 8,
    ) -> dict[str, Any]:
        selected_limit = self._bounded_flow_limit(limit)
        if self._flow_graph is None:
            return {
                "schema_version": 1,
                "root_node_id": node_id,
                "direction": direction,
                "items": [],
                "effective_labels": None,
                "coverage": "unknown",
                "next_cursor": None,
                "truncated": False,
            }
        return self._flow_graph.query_flow_lineage(
            node_id,
            direction=direction,
            after=after,
            limit=selected_limit,
            max_depth=max_depth,
        )

    def _bounded_flow_limit(self, limit: int | None) -> int:
        selected = (
            getattr(self._config, "flow_query_limit", 100)
            if limit is None
            else limit
        )
        hard = min(getattr(self._config, "flow_query_hard_limit", 1_000), 500)
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
            raise ValueError("semantic flow query limit must be a positive integer")
        if selected > hard:
            raise ValueError(f"semantic flow query limit exceeds hard cap {hard}")
        return selected

    def query_machine_settlements(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        pid: str | None = None,
        request_id: str | None = None,
        effect_id: str | None = None,
        action_id: str | None = None,
        tenant_bucket_sha256: str | None = None,
        outcome: str | None = None,
        epoch_id: str | None = None,
    ) -> dict[str, Any]:
        selected_limit = self._bounded_settlement_limit(limit)
        page = self._repository.query_semantic_machine_settlements(
            after=self._decode_v6_cursor(after),
            limit=selected_limit,
            pid=pid,
            request_id=request_id,
            effect_id=effect_id,
            action_id=action_id,
            tenant_bucket_sha256=tenant_bucket_sha256,
            outcome=outcome,
            epoch_id=epoch_id,
        )
        links = self._human_outcome_links_for_records(
            page.records,
            resolver_name="semantic_human_outcome_links_for_settlements",
            identity_field="settlement_id",
        )
        items: list[dict[str, Any]] = []
        for record in page.records:
            item = record.to_dict()
            item.update(self._settlement_human_outcome(links.get(record.settlement_id)))
            items.append(item)
        return {
            "schema_version": 1,
            "items": items,
            "next_cursor": self._encode_v6_cursor(page.next_cursor),
        }

    def query_policy_epochs(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        page = self._repository.query_semantic_policy_epochs(
            after=self._decode_v6_cursor(after),
            limit=self._bounded_settlement_limit(limit),
        )
        return self._v6_page(page)

    def query_control_history(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        page = self._repository.query_semantic_control_history(
            after=self._decode_v6_cursor(after),
            limit=self._bounded_settlement_limit(limit),
        )
        items = []
        for record in page.records:
            raw = record.to_dict()
            items.append(
                {
                    "schema_version": 1,
                    "revision": raw["revision"],
                    "generation": raw["generation"],
                    "mode": raw["mode"],
                    "active_epoch_id": raw["active_epoch_id"],
                    "active_policy_sha256": raw["active_policy_sha256"],
                    "tripped": raw["tripped"],
                    "trip_code": raw["trip_code"],
                    "updated_at": raw["created_at"],
                }
            )
        return {
            "schema_version": 1,
            "items": items,
            "next_cursor": self._encode_v6_cursor(page.next_cursor),
        }

    def query_health_events(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        severity: str | None = None,
        code: str | None = None,
        epoch_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "after": self._decode_v6_cursor(after),
            "limit": self._bounded_settlement_limit(limit),
            "severity": severity,
            "epoch_id": epoch_id,
        }
        if code is not None:
            kwargs["event_kind"] = code
        page = self._repository.query_semantic_health_events(**kwargs)
        return self._v6_page(page)

    def append_review_label(
        self,
        *,
        settlement_id: str,
        outcome: str,
        reviewer_id: str,
        evidence_sha256: str,
        reviewed_at: str | None = None,
    ) -> dict[str, Any]:
        getter = getattr(
            self._repository,
            "get_semantic_machine_settlement",
            None,
        )
        if not callable(getter):
            raise ValidationError(
                "semantic settlement exact lookup is unavailable"
            )
        settlement = getter(settlement_id)
        if settlement is None:
            raise ValidationError("semantic settlement does not exist")
        # Reviewer identities are never stored in clear.  The domain-separated
        # digest is evidence identity, not a tenant pseudonym and is not reused
        # for authorization decisions.
        label = SemanticReviewLabelV1(
            review_id=new_id("semantic-review"),
            settlement_id=settlement_id,
            outcome=SemanticReviewOutcome(outcome),
            reviewer_sha256=_sha256(
                {
                    "domain": "agent-libos.semantic.review.reviewer.v1",
                    "reviewer_id": reviewer_id,
                }
            ),
            evidence_sha256=evidence_sha256,
            created_at=reviewed_at or utc_now(),
        )
        record = SemanticReviewLabelRecord(
            review_id=label.review_id,
            settlement_id=label.settlement_id,
            outcome=label.outcome.value,
            reviewer_sha256=label.reviewer_sha256,
            evidence_sha256=label.evidence_sha256,
            created_at=label.created_at,
            schema_version=label.schema_version,
        )
        if label.outcome is SemanticReviewOutcome.UNSAFE:
            self._append_unsafe_review_label(record, settlement=settlement)
        else:
            with self._repository.transaction():
                self._repository.append_semantic_review_label(record)
        return label.to_dict()

    def _append_unsafe_review_label(
        self,
        record: SemanticReviewLabelRecord,
        *,
        settlement: Any,
    ) -> None:
        """Linearize an unsafe label against the exact durable control row.

        A control CAS race rolls the whole attempt back and starts again from
        a fresh read.  If bounded contention never converges, the evidence is
        appended independently before this call reports failure: accepted
        unsafe evidence must not disappear merely because the authority fence
        is busy.  The degraded path installs both a process-local authority
        latch and durable critical health evidence for restart admission.
        """

        last_race: _UnsafeReviewControlRace | None = None
        for _attempt in range(_UNSAFE_REVIEW_CONTROL_MAX_ATTEMPTS):
            try:
                with self._repository.transaction():
                    try:
                        self._fence_unsafe_review_control()
                    except _UnsafeReviewControlRace:
                        raise
                    except Exception as exc:
                        raise _UnsafeReviewControlFailure from exc
                    self._repository.append_semantic_review_label(record)
                return
            except _UnsafeReviewControlRace as exc:
                last_race = exc
                continue
            except _UnsafeReviewControlFailure as exc:
                self._retain_unsettled_unsafe_review(
                    record,
                    settlement=settlement,
                    failure=exc.__cause__ or exc,
                )
                raise ValidationError(
                    "unsafe semantic review evidence was retained but durable "
                    "control settlement failed"
                ) from (exc.__cause__ or exc)

        self._retain_unsettled_unsafe_review(
            record,
            settlement=settlement,
            failure=last_race or _UnsafeReviewControlRace(),
        )
        raise ValidationError(
            "unsafe semantic review evidence was retained after durable control race"
        ) from last_race

    def _fence_unsafe_review_control(self) -> None:
        reader = getattr(self._repository, "get_semantic_control_state", None)
        fence = getattr(self._repository, "fence_semantic_control_state", None)
        if not callable(reader) or not callable(fence):
            raise ValidationError(
                "unsafe semantic review cannot verify durable control"
            )
        expected = reader()
        if expected is None:
            raise ValidationError(
                "unsafe semantic review requires admitted durable control"
            )
        if fence(expected) is not True:
            raise _UnsafeReviewControlRace
        try:
            mode = SemanticRuntimeMode(expected.mode)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValidationError(
                "unsafe semantic review found invalid durable control"
            ) from exc
        if mode in {SemanticRuntimeMode.OFF, SemanticRuntimeMode.SHADOW}:
            return
        if mode not in {
            SemanticRuntimeMode.ENFORCE_DENY,
            SemanticRuntimeMode.CANARY_AUTO,
        }:
            raise ValidationError(
                "unsafe semantic review found unsupported durable control"
            )
        if self._control is None:
            raise ValidationError(
                "unsafe semantic review cannot trip missing control"
            )
        trip = getattr(self._control, "trip", None)
        if not callable(trip):
            raise ValidationError(
                "unsafe semantic review cannot trip semantic control"
            )
        tripped = trip("unsafe_review")
        confirmed = reader()
        if (
            not isinstance(tripped, SemanticControlStateV1)
            or not tripped.tripped
            or tripped.generation != expected.generation
            or tripped.active_epoch_id != expected.active_epoch_id
            or tripped.active_policy_sha256 != expected.active_policy_sha256
            or confirmed is None
            or confirmed.tripped is not True
            or confirmed.generation != expected.generation
            or confirmed.active_epoch_id != expected.active_epoch_id
            or confirmed.active_policy_sha256 != expected.active_policy_sha256
        ):
            raise ValidationError(
                "unsafe semantic review did not durably trip the active epoch"
            )

    def _retain_unsettled_unsafe_review(
        self,
        record: SemanticReviewLabelRecord,
        *,
        settlement: Any,
        failure: BaseException,
    ) -> None:
        """Retain unsafe evidence first, then fail closed as far as possible."""

        # This append is deliberately independent from the failed control
        # transaction.  If the evidence append itself fails, propagate that
        # storage failure: there is no truthful success/error receipt to give
        # the operator without a durable review row.
        with self._repository.transaction():
            self._repository.append_semantic_review_label(record)

        control_record = self._read_control_for_unsafe_review_health()
        self._install_unsafe_review_local_latch()
        tripped = self._best_effort_trip_after_unsafe_review(control_record)
        self._record_unsettled_unsafe_review_health(
            record,
            settlement=settlement,
            control_record=control_record,
            failure=failure,
            tripped=tripped,
        )

    def _read_control_for_unsafe_review_health(self) -> Any | None:
        reader = getattr(self._repository, "get_semantic_control_state", None)
        if not callable(reader):
            return None
        try:
            return reader()
        except Exception:
            return None

    def _install_unsafe_review_local_latch(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._unsafe_review_latched = True
        else:
            with lock:
                self._unsafe_review_latched = True
        self._increment_capture_failure()
        latch = getattr(self._control, "latch_unsafe_review", None)
        if callable(latch):
            try:
                latch()
            except Exception:
                # The manager latch still prevents new settlements.  The
                # durable health record below restores fail-closed admission
                # after restart even when the control object cannot latch.
                return

    def _best_effort_trip_after_unsafe_review(self, control_record: Any | None) -> bool:
        try:
            mode = SemanticRuntimeMode(getattr(control_record, "mode", None))
        except (TypeError, ValueError):
            return False
        if mode not in {
            SemanticRuntimeMode.ENFORCE_DENY,
            SemanticRuntimeMode.CANARY_AUTO,
        }:
            return False
        trip = getattr(self._control, "trip", None)
        if not callable(trip):
            return False
        try:
            value = trip(SemanticTripCode.UNSAFE_REVIEW)
        except Exception:
            return False
        return isinstance(value, SemanticControlStateV1) and value.tripped

    def _record_unsettled_unsafe_review_health(
        self,
        record: SemanticReviewLabelRecord,
        *,
        settlement: Any,
        control_record: Any | None,
        failure: BaseException,
        tripped: bool,
    ) -> None:
        append = getattr(self._repository, "append_semantic_health_event", None)
        if not callable(append):
            return
        event_kind = (
            "semantic_unsafe_review_fallback_trip"
            if tripped
            else _UNSAFE_REVIEW_UNSETTLED_EVENT
        )
        epoch_id = getattr(control_record, "active_epoch_id", None)
        tenant_bucket_sha256 = getattr(settlement, "tenant_bucket_sha256", None)
        if not isinstance(epoch_id, str) or not epoch_id:
            epoch_id = None
        if (
            not isinstance(tenant_bucket_sha256, str)
            or len(tenant_bucket_sha256) != 64
        ):
            tenant_bucket_sha256 = None
        evidence_sha256 = _sha256(
            {
                "schema_version": 1,
                "event_kind": event_kind,
                "review_id": record.review_id,
                "settlement_id": record.settlement_id,
                "failure_type": type(failure).__name__,
                "control_revision": getattr(control_record, "revision", None),
                "control_generation": getattr(control_record, "generation", None),
                "active_epoch_id": epoch_id,
            }
        )
        try:
            append(
                SemanticHealthEventRecord(
                    event_id=new_id("semantic_health"),
                    event_kind=event_kind,
                    severity="critical",
                    epoch_id=epoch_id,
                    tenant_bucket_sha256=tenant_bucket_sha256,
                    evidence_sha256=evidence_sha256,
                    created_at=utc_now(),
                )
            )
        except Exception:
            # Evidence is already durable and both live authority paths are
            # latched.  A health-write failure must not erase the review.
            return

    def _bounded_settlement_limit(self, limit: int | None) -> int:
        selected = (
            getattr(self._config, "settlement_list_limit", 100)
            if limit is None
            else limit
        )
        hard = min(
            getattr(self._config, "settlement_list_hard_limit", 1_000),
            500,
        )
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 1
            or selected > hard
        ):
            raise ValueError(
                f"semantic settlement query limit must be between 1 and {hard}"
            )
        return selected

    @classmethod
    def _v6_page(cls, page: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [record.to_dict() for record in page.records],
            "next_cursor": cls._encode_v6_cursor(page.next_cursor),
        }

    def _human_outcome_links_for_records(
        self,
        records: Any,
        *,
        resolver_name: str,
        identity_field: str,
    ) -> dict[str, SemanticHumanOutcomeLinkRecord]:
        selected = tuple(records)
        if len(selected) > 500:
            raise ValidationError("semantic Human outcome join exceeds hard limit")
        identities = tuple(getattr(record, identity_field, None) for record in selected)
        if any(not isinstance(identity, str) or not identity for identity in identities):
            raise ValidationError("semantic Human outcome join identity is invalid")
        if len(identities) != len(set(identities)):
            raise ValidationError("semantic Human outcome join identity is ambiguous")
        resolver = getattr(self._repository, resolver_name, None)
        if not callable(resolver):
            # Compatibility for narrow pre-v6 test repositories.  Production
            # v6 repositories expose the bounded request-id join.
            return {}
        joined = resolver(identities)
        if not isinstance(joined, Mapping) or any(
            identity not in identities for identity in joined
        ):
            raise ValidationError("semantic Human outcome join result is malformed")
        result: dict[str, SemanticHumanOutcomeLinkRecord] = {}
        records_by_id = dict(zip(identities, selected, strict=True))
        for identity, link in joined.items():
            if not isinstance(link, SemanticHumanOutcomeLinkRecord):
                raise ValidationError("semantic Human outcome join record is untyped")
            record = records_by_id[identity]
            self._validate_human_outcome_link_binding(
                record,
                link,
                identity_field=identity_field,
            )
            result[identity] = link
        return result

    @staticmethod
    def _validate_human_outcome_link_binding(
        record: Any,
        link: SemanticHumanOutcomeLinkRecord,
        *,
        identity_field: str,
    ) -> None:
        if (
            link.request_id != getattr(record, "request_id", None)
            or link.pid != getattr(record, "pid", None)
        ):
            raise ValidationError("semantic Human outcome request binding is invalid")
        identity = getattr(record, identity_field)
        linked_identity = (
            link.assessment_id
            if identity_field == "assessment_id"
            else link.settlement_id
        )
        if linked_identity is not None and linked_identity != identity:
            raise ValidationError("semantic Human outcome record binding is invalid")
        record_job_id = getattr(record, "job_id", None)
        if (
            link.job_id is not None
            and record_job_id is not None
            and link.job_id != record_job_id
        ):
            raise ValidationError("semantic Human outcome job binding is invalid")
        record_assessment_id = getattr(record, "assessment_id", None)
        if (
            identity_field == "settlement_id"
            and link.assessment_id is not None
            and record_assessment_id is not None
            and link.assessment_id != record_assessment_id
        ):
            raise ValidationError(
                "semantic Human outcome assessment binding is invalid"
            )

    @staticmethod
    def _settlement_human_outcome(
        link: SemanticHumanOutcomeLinkRecord | None,
    ) -> dict[str, Any]:
        if link is None:
            return {
                "human_outcome": None,
                "human_outcome_source": None,
                "human_outcome_request_revision": None,
                "human_outcome_decision_sha256": None,
                "human_outcome_created_at": None,
            }
        return {
            "human_outcome": link.outcome,
            "human_outcome_source": link.source,
            "human_outcome_request_revision": link.request_revision,
            "human_outcome_decision_sha256": link.decision_sha256,
            "human_outcome_created_at": link.created_at,
        }

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        record = self._repository.get_semantic_assessment(assessment_id)
        if record is None:
            return None
        links = self._human_outcome_links_for_records(
            (record,),
            resolver_name="semantic_human_outcome_links_for_assessments",
            identity_field="assessment_id",
        )
        result = record.to_dict()
        link = links.get(record.assessment_id)
        if link is not None:
            result["human_outcome"] = link.outcome
        return result

    def _capture_human_request(
        self,
        request: HumanRequest,
    ) -> SemanticAssessmentJobRecord | None:
        if not isinstance(request, HumanRequest):
            raise TypeError("semantic approval capture requires HumanRequest")
        payload = request.payload
        if not isinstance(payload, Mapping):
            raise TypeError("semantic approval payload must be a JSON object")
        if payload.get("type") != "external_operation_approval":
            return None
        # The Human observer receives the committed request, but retain only a
        # canonical digest. If the value cannot be canonically encoded, do not
        # invent a malformed assessment whose provenance cannot be verified.
        input_sha256 = _sha256(payload)
        try:
            prepared = self._prepare_human_approval(request, input_sha256)
        except ValidationError:
            # This enqueue deliberately sits outside the validation region.
            # A repository/schema failure while persisting the fallback must
            # become one capture failure, never a recursive fallback attempt.
            return self._enqueue_malformed_human_approval(
                request,
                input_sha256=input_sha256,
            )
        request_model, selected_candidate, violations = prepared
        return self._enqueue(
            request_model,
            candidate=selected_candidate,
            hard_violations=violations,
            request_revision=request.revision,
            data_flow_context=self._human_flow(payload),
        )

    def _prepare_human_approval(
        self,
        request: HumanRequest,
        input_sha256: str,
    ) -> tuple[
        SemanticAssessmentRequest,
        SemanticApprovalCandidate | None,
        tuple[SemanticReasonCode, ...],
    ]:
        payload = request.payload
        exact = decode_exact_semantic_approval_request(request)
        context = exact.context
        capability = exact.capability
        binding = exact.binding
        action = exact.action_id
        resource = exact.resource
        rights = list(exact.rights)
        flow = self._human_flow(payload)
        sink_identity_sha256 = self._optional_digest(
            context.get("sink_identity_sha256")
        )
        tool_schema_sha256 = self._optional_digest(
            context.get("tool_schema_sha256")
        )
        raw_provider_spec_sha256 = context.get("provider_spec_sha256")
        if raw_provider_spec_sha256 is None:
            raw_provider_spec_sha256 = context.get("registry_spec_sha256")
        provider_spec_sha256 = self._optional_digest(
            raw_provider_spec_sha256
        )

        selected_candidate = self._authority_candidate(
            request.pid,
            action,
            resource,
            rights,
        )
        manifest_sha256, policy_sha256 = self._manifest_policy(
            request.pid,
            selected_candidate,
        )
        selected_facts = self._approval_facts(
            request.pid,
            action,
            resource,
            rights,
            capability,
            context,
            binding,
            selected_candidate,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
        )
        request_model = self._request_model(
            kind=SemanticAssessmentKind.APPROVAL,
            domain=_domain(context.get("adapter") or action.split(".", 1)[0]),
            action_id=action,
            input_value=None,
            frozen_input_sha256=input_sha256,
            pid=request.pid,
            request_id=request.request_id,
            operation_id=(
                context.get("operation_id")
                if isinstance(context.get("operation_id"), str)
                else binding.get("operation_id")
                if isinstance(binding.get("operation_id"), str)
                else None
            ),
            effect_id=binding.get("effect_id"),
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            data_labels=flow.labels,
            features=selected_facts,
            resource_sha256=_sha256(resource),
            args_sha256=_sha256(context),
            state_sha256=_sha256(binding),
            source_refs_sha256=flow.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(flow.labels),
            sink_identity_sha256=sink_identity_sha256,
            tool_schema_sha256=tool_schema_sha256,
            provider_spec_sha256=provider_spec_sha256,
        )
        return (
            request_model,
            selected_candidate,
            (),
        )

    def _enqueue_malformed_human_approval(
        self,
        request: HumanRequest,
        *,
        input_sha256: str,
    ) -> SemanticAssessmentJobRecord:
        malformed = self._request_model(
            kind=SemanticAssessmentKind.APPROVAL,
            domain=SemanticDomain.RUNTIME,
            action_id="runtime.malformed_external_operation",
            input_value=None,
            frozen_input_sha256=input_sha256,
            pid=request.pid,
            request_id=request.request_id,
            manifest_sha256=None,
            policy_sha256=_policy_sha256(_EMPTY_POLICY),
            data_labels=DataLabels(),
            features=AuthoritativeApprovalFacts(),
        )
        return self._enqueue(
            malformed,
            candidate=None,
            hard_violations=(),
            request_revision=request.revision,
        )

    def _capture_root_process(
        self,
        pid: str,
        *,
        image_id: str | None,
        publication_id: str | None,
        goal: Any | None,
    ) -> SemanticAssessmentJobRecord:
        selected_goal = goal
        if selected_goal is None and self._root_goal_reader is not None:
            selected_goal = self._root_goal_reader(pid)
        if selected_goal is None:
            raise ValidationError("semantic root goal is unavailable")
        process = self._processes.get_process(pid) if self._processes is not None else None
        if process is not None and process.parent_pid is not None:
            raise ValidationError("semantic root goal capture received a child process")
        payload = getattr(selected_goal, "payload", selected_goal)
        metadata = getattr(selected_goal, "metadata", None)
        labels = (
            DataLabels.from_object_metadata(metadata)
            if isinstance(metadata, ObjectMetadata)
            else DataLabels()
        )
        provenance = getattr(selected_goal, "provenance", None)
        if self._root_flow_resolver is not None:
            flow_context = self._root_flow_resolver(selected_goal)
            if (
                not isinstance(flow_context, DataFlowContext)
                or flow_context.labels.to_dict() != labels.to_dict()
            ):
                raise ValidationError(
                    "semantic root goal DataFlow snapshot is inconsistent"
                )
        elif self._adapter == "external":
            raise ValidationError(
                "semantic external root goal has no Host DataFlow snapshot"
            )
        else:
            flow_context = DataFlowContext(labels=labels)
        manifest_sha256, policy_sha256 = self._manifest_policy(pid, None)
        request = self._request_model(
            kind=SemanticAssessmentKind.ROOT_GOAL,
            domain=SemanticDomain.RUNTIME,
            action_id="runtime.root_goal",
            input_value=payload,
            pid=pid,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            data_labels=labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            resource_sha256=_sha256(
                {"goal_oid": getattr(selected_goal, "oid", None)}
            ),
            args_sha256=_sha256(
                {"image_id": image_id, "publication_id": publication_id}
            ),
            state_sha256=_sha256(
                {
                    "goal_version": getattr(selected_goal, "version", None),
                    "process_revision": getattr(process, "revision", None),
                }
            ),
            source_refs_sha256=flow_context.source_refs_hash(),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
            redacted_intent=_root_goal_intent(
                payload,
                max_chars=self._config.intent_max_chars,
            ),
        )
        queued = self._enqueue(
            request,
            candidate=None,
            hard_violations=(),
            local_dlp_findings=_root_goal_dlp_findings(
                payload,
                input_sha256=request.input_sha256,
            ),
            data_flow_context=flow_context,
        )
        self._capture_flow_root_goal(
            pid=pid,
            goal=selected_goal,
            request=request,
            labels=labels,
        )
        return queued

    def _capture_provider_result(
        self,
        result: Any,
        observation: Any,
        *,
        data_labels: DataLabels | None,
        provider_spec_sha256: str | None,
        tool_schema_sha256: str | None,
    ) -> SemanticAssessmentJobRecord | None:
        if getattr(observation, "contract_name", None) == "semantic.llm.assess":
            return None
        pid = getattr(observation, "pid", None)
        provider = getattr(observation, "provider", None)
        operation = getattr(observation, "operation", None)
        effect_id = getattr(observation, "effect_id", None)
        if not all(isinstance(item, str) and item for item in (pid, provider, operation, effect_id)):
            raise ValidationError("semantic provider observation is incomplete")
        selected_labels = data_labels or getattr(observation, "data_labels", None)
        if not isinstance(selected_labels, DataLabels):
            selected_labels = DataLabels(
                trust_level="untrusted",
                integrity="untrusted",
                origin=f"external:{provider}",
            )
        source_refs_sha256 = getattr(observation, "source_refs_sha256", None)
        flow_context = getattr(observation, "data_flow_context", None)
        if flow_context is not None and not isinstance(
            flow_context,
            DataFlowContext,
        ):
            raise ValidationError(
                "semantic provider observation DataFlow context is malformed"
            )
        if flow_context is None:
            empty_context = DataFlowContext(labels=selected_labels)
            if source_refs_sha256 == empty_context.source_refs_hash():
                flow_context = empty_context
        result_sha256 = self._optional_digest(
            getattr(observation, "result_sha256", None)
        )
        if result_sha256 is None:
            raise ValidationError(
                "semantic provider observation has no Host result digest"
            )
        descriptor = getattr(observation, "result_descriptor", {})
        if not isinstance(descriptor, Mapping):
            raise ValidationError("semantic provider result descriptor is malformed")
        detector = LocalDlpAccumulator(input_sha256=result_sha256)
        visit_bounded_host_result_text(
            result,
            contract_name=getattr(observation, "contract_name", None),
            visitor=detector.scan,
        )
        manifest_sha256, policy_sha256 = self._manifest_policy(pid, None)
        selected_domain = _domain(provider)
        request = self._request_model(
            kind=SemanticAssessmentKind.PROVIDER_INGRESS,
            domain=selected_domain,
            action_id=_action_id(selected_domain, operation),
            input_value=None,
            frozen_input_sha256=result_sha256,
            pid=pid,
            effect_id=effect_id,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            data_labels=selected_labels,
            features=AuthoritativeApprovalFacts(schema_valid=True),
            state_sha256=_sha256(
                {
                    "provider": provider,
                    "operation": operation,
                    "target_sha256": _sha256(getattr(observation, "target", None)),
                    "contract_name": getattr(observation, "contract_name", None),
                    "data_flow_direction": getattr(
                        observation,
                        "data_flow_direction",
                        None,
                    ),
                    "result_descriptor": dict(descriptor),
                }
            ),
            source_refs_sha256=self._optional_digest(source_refs_sha256),
            data_labels_sha256=_identity_safe_labels_sha256(selected_labels),
            tool_schema_sha256=self._optional_digest(tool_schema_sha256)
            or self._optional_digest(
                getattr(observation, "tool_schema_sha256", None)
            ),
            provider_spec_sha256=self._optional_digest(provider_spec_sha256)
            or self._optional_digest(
                getattr(observation, "provider_spec_sha256", None)
            ),
        )
        queued = self._enqueue(
            request,
            candidate=None,
            hard_violations=(),
            local_dlp_findings=detector.findings,
            data_flow_context=flow_context,
        )
        self._capture_flow_provider_ingress(
            request=request,
            labels=selected_labels,
        )
        return queued

    def _capture_flow_root_goal(
        self,
        *,
        pid: str,
        goal: Any,
        request: SemanticAssessmentRequest,
        labels: DataLabels,
    ) -> None:
        flow = self._flow_graph
        if flow is None:
            return
        try:
            flow.capture_root_goal(
                pid=pid,
                goal_oid=(
                    str(getattr(goal, "oid"))
                    if getattr(goal, "oid", None) is not None
                    else None
                ),
                goal_version=getattr(goal, "version", None),
                content_sha256=request.input_sha256,
                state_sha256=request.state_sha256,
                provenance_sha256=request.source_refs_sha256,
                labels=labels,
                tenant_bucket_sha256=self._tenant_bucket(labels.tenant),
                created_at=getattr(goal, "created_at", None) or utc_now(),
            )
        except Exception:
            # Flow evidence is advisory at capture time.  It must never roll
            # back a committed root process or the independently durable
            # assessment job; missing coverage later makes auto approval
            # ineligible.
            return

    def _capture_flow_provider_ingress(
        self,
        *,
        request: SemanticAssessmentRequest,
        labels: DataLabels,
    ) -> None:
        flow = self._flow_graph
        if flow is None:
            return
        try:
            assert request.pid is not None
            assert request.effect_id is not None
            flow.capture_provider_ingress(
                pid=request.pid,
                effect_id=request.effect_id,
                action_id=request.action_id,
                result_sha256=request.input_sha256,
                state_sha256=request.state_sha256,
                provider_spec_sha256=request.provider_spec_sha256,
                tool_schema_sha256=request.tool_schema_sha256,
                labels=labels,
                tenant_bucket_sha256=self._tenant_bucket(labels.tenant),
                created_at=utc_now(),
            )
        except Exception:
            # The actual provider result has already committed.  Evidence
            # capture therefore degrades to unknown coverage without changing
            # or masking the business result.
            return

    def _request_model(
        self,
        *,
        kind: SemanticAssessmentKind,
        domain: SemanticDomain,
        action_id: str,
        input_value: Any,
        frozen_input_sha256: str | None = None,
        pid: str,
        manifest_sha256: str | None,
        policy_sha256: str,
        data_labels: DataLabels,
        features: AuthoritativeApprovalFacts,
        request_id: str | None = None,
        operation_id: str | None = None,
        effect_id: str | None = None,
        resource_sha256: str | None = None,
        args_sha256: str | None = None,
        state_sha256: str | None = None,
        source_refs_sha256: str | None = None,
        data_labels_sha256: str | None = None,
        sink_identity_sha256: str | None = None,
        tool_schema_sha256: str | None = None,
        provider_spec_sha256: str | None = None,
        redacted_intent: str | None = None,
    ) -> SemanticAssessmentRequest:
        now = utc_now()
        return SemanticAssessmentRequest(
            kind=kind,
            domain=domain,
            action_id=action_id,
            input_sha256=frozen_input_sha256 or _sha256(input_value),
            deadline_at=_future_timestamp(now, self._config.assessment_timeout_s),
            data_labels=data_labels,
            features=features,
            redacted_intent=redacted_intent,
            pid=pid,
            request_id=request_id,
            operation_id=operation_id,
            effect_id=effect_id,
            manifest_sha256=manifest_sha256,
            policy_sha256=policy_sha256,
            resource_sha256=resource_sha256,
            args_sha256=args_sha256,
            state_sha256=state_sha256,
            source_refs_sha256=source_refs_sha256,
            data_labels_sha256=data_labels_sha256,
            sink_identity_sha256=sink_identity_sha256,
            tool_schema_sha256=tool_schema_sha256,
            provider_spec_sha256=provider_spec_sha256,
        )

    def _enqueue(
        self,
        request: SemanticAssessmentRequest,
        *,
        candidate: SemanticApprovalCandidate | None,
        hard_violations: tuple[SemanticReasonCode, ...],
        local_dlp_findings: tuple[LocalDlpFinding, ...] = (),
        request_revision: int | None = None,
        data_flow_context: DataFlowContext | None = None,
    ) -> SemanticAssessmentJobRecord:
        if (
            not isinstance(local_dlp_findings, tuple)
            or any(
                not isinstance(item, LocalDlpFinding)
                for item in local_dlp_findings
            )
        ):
            raise TypeError("semantic local DLP findings must be frozen Host evidence")
        external = build_external_projection(
            request,
            labels=request.data_labels,
            intent_max_chars=self._config.intent_max_chars,
            projection_max_bytes=self._config.projection_max_bytes,
        )
        frozen_dlp_findings = list(external.dlp_findings)
        for item in local_dlp_findings:
            if item not in frozen_dlp_findings:
                frozen_dlp_findings.append(item)
        if len(frozen_dlp_findings) > 4:
            raise ValidationError("semantic local DLP evidence exceeds its finite limit")
        external_payload = dict(external.payload)
        external_payload["dlp_findings"] = [
            item.to_dict() for item in frozen_dlp_findings
        ]
        identity_present = any(
            value is not None
            for value in (
                request.data_labels.tenant,
                request.data_labels.principal,
            )
        )
        if frozen_dlp_findings or identity_present:
            external_payload["projection_mode"] = "metadata_only"
            external_payload.pop("redacted_intent", None)
            external_payload.pop("redacted_intent_sha256", None)
            external_payload.pop("redacted_intent_truncated", None)
        projection = {
            **external_payload,
            "deadline_at": request.deadline_at,
            "candidate": _candidate_projection(candidate),
            "hard_violations": [item.value for item in hard_violations],
            "identity_present": identity_present,
            "identity_mixed": request.data_labels.is_mixed_identity,
            # Raw tenant identifiers are never hashed without a Host-owned
            # keyed bucketer; an ordinary SHA-256 remains dictionary-guessable.
            "tenant_bucket_sha256": self._tenant_bucket(
                request.data_labels.tenant
            ),
            "request_revision": request_revision,
        }
        created_at = utc_now()
        projection_sha256 = _sha256(projection)
        bindings = {
            "artifact_sha256": self._artifact_sha256,
            "input_sha256": request.input_sha256,
            "feature_snapshot_sha256": _sha256(request.features.to_dict()),
            "policy_sha256": request.policy_sha256 or _policy_sha256(_EMPTY_POLICY),
            "manifest_sha256": request.manifest_sha256,
            "action_sha256": _sha256(request.action_id),
            "resource_sha256": request.resource_sha256,
            "args_sha256": request.args_sha256,
            "state_sha256": request.state_sha256,
            "source_refs_sha256": request.source_refs_sha256,
            "data_labels_sha256": request.data_labels_sha256,
            "sink_identity_sha256": request.sink_identity_sha256,
            "tool_schema_sha256": request.tool_schema_sha256,
            "provider_spec_sha256": request.provider_spec_sha256,
            "tenant_bucket_sha256": projection["tenant_bucket_sha256"],
        }
        record = SemanticAssessmentJobRecord(
            job_id=new_id("semantic-job"),
            kind=request.kind.value,
            status=SemanticAssessmentJobStatus.QUEUED,
            domain=request.domain.value,
            pid=request.pid,
            request_id=request.request_id,
            operation_id=request.operation_id,
            effect_id=request.effect_id,
            bindings=bindings,
            projection=projection,
            projection_sha256=projection_sha256,
            projection_retention=SemanticProjectionRetention.REDACTED,
            projection_expires_at=_future_timestamp(
                created_at,
                self._config.projection_ttl_s,
            ),
            created_at=created_at,
            updated_at=created_at,
        )
        # Claiming workers and the off transition use the same lock.  Publish
        # the durable job and its non-durable source refs as one process-local
        # visibility step so polling workers cannot observe only one half.
        with self._lock:
            self._validate_transient_context(
                request,
                job_id=record.job_id,
                data_flow_context=data_flow_context,
            )
            persisted = self._repository.enqueue_semantic_assessment_job(record)
            if self._adapter == "external" and data_flow_context is not None:
                self._transient_contexts[persisted.job_id] = _TransientFlowSnapshot(
                    context=data_flow_context,
                    exact_labels_sha256=_sha256(data_flow_context.labels.to_dict()),
                )
                self._transient_contexts.move_to_end(persisted.job_id)
            self._wake.set()
        return persisted

    def _validate_transient_context(
        self,
        request: SemanticAssessmentRequest,
        *,
        job_id: str,
        data_flow_context: DataFlowContext | None,
    ) -> None:
        if data_flow_context is None:
            return
        if not isinstance(data_flow_context, DataFlowContext):
            raise TypeError("semantic transient DataFlow context is invalid")
        if (
            data_flow_context.labels.to_dict() != request.data_labels.to_dict()
            or request.source_refs_sha256 is None
            or data_flow_context.source_refs_hash()
            != request.source_refs_sha256
        ):
            raise ValidationError(
                "semantic transient DataFlow provenance does not match its request"
            )
        if (
            self._adapter == "external"
            and job_id not in self._transient_contexts
            and len(self._transient_contexts) >= self._transient_context_limit
        ):
            raise ValidationError(
                "semantic transient DataFlow context capacity is exhausted"
            )

    def _tenant_bucket(self, tenant: str | None) -> str | None:
        if tenant is None or self._tenant_bucketer is None:
            return None
        selected = self._tenant_bucketer(tenant)
        if not isinstance(selected, str) or re.fullmatch(r"[0-9a-f]{64}", selected) is None:
            raise ValidationError(
                "Host semantic tenant bucketer must return a lowercase SHA-256"
            )
        return selected

    def _approval_facts(
        self,
        pid: str,
        action_id: str,
        resource: str,
        rights: list[Any],
        capability: Mapping[str, Any],
        context: Mapping[str, Any],
        binding: Mapping[str, Any],
        candidate: SemanticApprovalCandidate | None,
        *,
        manifest_sha256: str | None,
        policy_sha256: str,
    ) -> AuthoritativeApprovalFacts:
        action = DEFAULT_ACTION_ONTOLOGY.resolve(action_id)
        exact_rights = tuple(item for item in rights if isinstance(item, str))
        schema_valid, request_exact, binding_current = self._approval_binding_facts(
            pid,
            action_id,
            resource,
            exact_rights,
            capability,
            context,
            binding,
        )
        candidate_matches = (
            candidate is not None
            and candidate.authority_operation == action_id
            and candidate.resource == resource
            and candidate.rights == exact_rights
        )
        if candidate is not None:
            manifest_current = candidate.manifest_sha256 == manifest_sha256
            policy_current = candidate.policy_sha256 == policy_sha256
        else:
            manifest_current = manifest_sha256 is not None
            policy_current = True
        right_allowed = (
            action is not None
            and len(exact_rights) == 1
            and exact_rights[0] in action.allowed_rights
        )
        return AuthoritativeApprovalFacts(
            schema_valid=schema_valid,
            request_is_exact_external_operation=request_exact,
            binding_current=binding_current,
            manifest_current=manifest_current,
            policy_current=policy_current,
            action_known=action is not None,
            action_auto_eligible=bool(action and action.auto_approval_eligible),
            low_risk=bool(action and action.auto_approval_eligible),
            resource_exact="*" not in resource,
            single_non_control_right=right_allowed,
            ceiling_matched=candidate_matches,
            # This is a Host ontology fact about the target operation, frozen
            # at capture. Classifier egress and classifier success are unrelated.
            data_flow_allowed=bool(
                action
                and request_exact
                and not action.requires_data_flow_egress
            ),
            profile_pinned=(self._adapter != "external" or bool(self._profile_id)),
        )

    @staticmethod
    def _normalized_approval_bindings(
        capability: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        constraints = capability.get("constraints")
        nested_raw = (
            constraints.get(APPROVAL_BINDING_KEY)
            if isinstance(constraints, Mapping)
            else None
        )
        try:
            return (
                normalize_approval_binding(dict(binding)),
                normalize_approval_binding(dict(nested_raw)),
            )
        except (TypeError, ValueError, ValidationError):
            return None

    @staticmethod
    def _expected_approval_resource(
        action_id: str,
        context: Mapping[str, Any],
    ) -> str | None:
        remote_fields = {
            "jsonrpc.call": ("jsonrpc", "endpoint_id", "method_id"),
            "mcp.call": ("mcp", "server_id", "tool_id"),
        }
        remote = remote_fields.get(action_id)
        if remote is None:
            value = context.get("resource")
            return value if type(value) is str else None
        prefix, owner_key, member_key = remote
        owner = context.get(owner_key)
        member = context.get(member_key)
        if type(owner) is not str or type(member) is not str:
            return None
        return f"{prefix}:{owner}:{member}"

    @staticmethod
    def _approval_binding_facts(
        pid: str,
        action_id: str,
        resource: str,
        rights: tuple[str, ...],
        capability: Mapping[str, Any],
        context: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> tuple[bool, bool, bool]:
        normalized = SemanticManager._normalized_approval_bindings(
            capability,
            binding,
        )
        if normalized is None:
            return False, False, False
        top, nested = normalized
        expected_resource = SemanticManager._expected_approval_resource(
            action_id,
            context,
        )
        context_right = context.get("right")
        context_matches = (
            context.get("pid") == pid
            and context.get("authority_operation") == action_id
            and expected_resource == resource
            and context_right in rights
        )
        subject_matches = capability.get("subject") == pid
        request_exact = (
            subject_matches
            and context_matches
            and "*" not in resource
            and len(rights) == 1
        )
        schema_valid = subject_matches and context_matches
        binding_current = (
            schema_valid
            and top == nested
            and top["canonical_args_hash"] == canonical_effect_hash(dict(context))
        )
        return schema_valid, request_exact, binding_current

    @staticmethod
    def _human_flow(payload: Mapping[str, Any]) -> DataFlowContext:
        raw = payload.get("_agent_libos_data_flow_context")
        if raw is None:
            return DataFlowContext()
        if not isinstance(raw, Mapping):
            raise ValidationError("semantic Human DataFlow context is malformed")
        try:
            return DataFlowContext.from_dict(raw)
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ValidationError("semantic Human DataFlow context is malformed") from exc

    def _manifest_policy(
        self,
        pid: str | None,
        _candidate: SemanticApprovalCandidate | None,
    ) -> tuple[str | None, str]:
        manifest_sha256, policy_sha256 = self._read_live_manifest_policy(pid)
        return (
            manifest_sha256,
            policy_sha256 or _policy_sha256(_EMPTY_POLICY),
        )

    def _live_manifest_policy(
        self,
        pid: str | None,
    ) -> tuple[str | None, str | None]:
        try:
            return self._read_live_manifest_policy(pid)
        except Exception:
            return None, None

    def _read_live_manifest_policy(
        self,
        pid: str | None,
    ) -> tuple[str | None, str | None]:
        manifest = (
            self._authority.get_for_process(pid)
            if self._authority is not None and pid is not None
            else None
        )
        if manifest is None:
            return None, None
        approval_policy = getattr(manifest, "approval_policy", {})
        raw = (
            approval_policy.get("semantic_auto_approval")
            if isinstance(approval_policy, Mapping)
            else None
        )
        policy = dict(raw) if isinstance(raw, Mapping) else _EMPTY_POLICY
        return getattr(manifest, "manifest_hash", None), _policy_sha256(policy)

    @staticmethod
    def _coerce_candidate(candidate: Mapping[str, Any]) -> SemanticApprovalCandidate:
        return SemanticApprovalCandidate.from_dict(candidate)

    def _authority_candidate(
        self,
        pid: str,
        action_id: str,
        resource: str,
        rights: list[Any],
    ) -> SemanticApprovalCandidate | None:
        if self._authority is None:
            return None
        raw = self._authority.semantic_auto_approval_candidate(
            pid,
            authority_operation=action_id,
            resource=resource,
            rights=rights,
        )
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValidationError("semantic authority candidate is malformed")
        try:
            return self._coerce_candidate(raw)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ValidationError(
                "semantic authority candidate is malformed"
            ) from exc

    @staticmethod
    def _optional_digest(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValidationError("semantic provenance digest is malformed")
        return value

    def _assess_claimed(self, job: SemanticAssessmentJobRecord) -> None:
        started = time.monotonic()
        # Drain any abandoned same-thread value before binding telemetry to a
        # new job. Only the external Host adapter has a trusted usage source.
        collect_usage = self._adapter == "external"
        if collect_usage:
            _take_usage_telemetry(self._assessor)
        reported_usage: SemanticUsageTelemetry | None = None
        try:
            assessment = self._evaluate_claimed(job)
        finally:
            if collect_usage:
                reported_usage = _take_usage_telemetry(self._assessor)
        latency_ms = max(0, int((time.monotonic() - started) * 1000))
        completed_at = utc_now()
        if (
            job.lease_expires_at is None
            or _parse_time(completed_at) >= _parse_time(job.lease_expires_at)
        ):
            assessment = SemanticAssessment(
                status=SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN
            )
        self._terminalize(
            job,
            assessment,
            completed_at,
            latency_ms,
            usage_telemetry=reported_usage,
        )

    def _evaluate_claimed(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> SemanticAssessment:
        try:
            if job.bindings.get("artifact_sha256") != self._artifact_sha256:
                # A queued job is immutable evidence from its capture-time
                # classifier profile.  Reusing its projection with a newly
                # configured profile/model would make the assessment record
                # claim the old artifact while actually calling the new one.
                # Keep the captured provenance and fail closed before any
                # provider call; deterministic Host deny proof is still
                # independently re-evaluated during terminalization.
                return SemanticAssessment(
                    status=SemanticAssessmentStatus.STALE_INPUT
                )
            with self._lock:
                canary_auto = self._mode == "canary_auto"
            if canary_auto and self._job_is_catalog_outside(job):
                # Unsupported and high-risk actions are Human-owned.  They do
                # not need model evidence for Phase 4 settlement eligibility,
                # and classifier output must never turn them into an
                # authority-bearing candidate. Shadow/enforce modes still run
                # their full five-domain observational assessment contract.
                return SemanticAssessment(
                    status=SemanticAssessmentStatus.SKIPPED_POLICY
                )
            if self._deny_for_job(job) is not None:
                # Deterministic Host evidence is complete without classifier
                # egress.  The live proof is recomputed again inside the
                # terminal settlement transaction.
                return SemanticAssessment(
                    status=SemanticAssessmentStatus.SKIPPED_POLICY
                )
            request, live_flow = self._assessment_request_for_job(job)
            if not request.features.schema_valid:
                raise SemanticProviderResponseError(
                    "semantic approval request schema is invalid"
                )
            if _parse_time(utc_now()) >= _parse_time(request.deadline_at):
                raise SemanticAssessmentDeadlineExceeded(
                    "semantic assessment deadline expired before dispatch"
                )
            if self._adapter == "external":
                assess_host = getattr(self._assessor, "assess_host", None)
                if live_flow is None or not callable(assess_host):
                    raise CapabilityDenied(
                        "semantic external classifier has no live DataFlow context"
                    )
                assessment = assess_host(
                    HostSemanticAssessmentInvocation(
                        request=request,
                        data_flow_context=live_flow,
                    )
                )
            else:
                assessment = self._assessor.assess(request)
            if not isinstance(assessment, SemanticAssessment):
                raise SemanticProviderResponseError(
                    "semantic assessor returned an invalid result"
                )
            validate_monotonic_data_findings(
                request.data_labels,
                assessment.data_findings,
            )
            assessment = _normalize_assessment(assessment)
        except CapabilityDenied:
            assessment = SemanticAssessment(
                status=SemanticAssessmentStatus.EGRESS_BLOCKED
            )
        except SemanticAssessmentDeadlineExceeded:
            assessment = SemanticAssessment(status=SemanticAssessmentStatus.TIMEOUT)
        except SemanticProviderCallError:
            assessment = SemanticAssessment(
                status=SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN
            )
        except SemanticProviderResponseError:
            assessment = SemanticAssessment(
                status=SemanticAssessmentStatus.INVALID_SCHEMA
            )
        except TimeoutError:
            assessment = SemanticAssessment(status=SemanticAssessmentStatus.TIMEOUT)
        except (TypeError, ValueError, ValidationError):
            assessment = SemanticAssessment(
                status=SemanticAssessmentStatus.INVALID_SCHEMA
            )
        except Exception:
            status = (
                SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN
                if self._adapter == "external"
                else SemanticAssessmentStatus.PROVIDER_ERROR
            )
            assessment = SemanticAssessment(status=status)
        return assessment

    def _assessment_request_for_job(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> tuple[SemanticAssessmentRequest, DataFlowContext | None]:
        request = _request_from_job(job)
        if self._adapter != "external":
            return request, None
        if job.kind == SemanticAssessmentKind.APPROVAL.value:
            live, _candidate, flow = self._live_approval_assessment_for_job(job)
            return replace(request, data_labels=live.data_labels), flow
        with self._lock:
            snapshot = self._transient_contexts.get(job.job_id)
        if not isinstance(snapshot, _TransientFlowSnapshot):
            raise CapabilityDenied(
                "semantic transient DataFlow provenance is unavailable"
            )
        flow = snapshot.context
        if (
            _sha256(flow.labels.to_dict()) != snapshot.exact_labels_sha256
            or request.source_refs_sha256 is None
            or flow.source_refs_hash() != request.source_refs_sha256
            or job.bindings.get("data_labels_sha256")
            != _identity_safe_labels_sha256(flow.labels)
        ):
            raise CapabilityDenied(
                "semantic transient DataFlow provenance drifted"
            )
        try:
            live_bucket = self._tenant_bucket(flow.labels.tenant)
        except Exception as exc:
            # A capture-time bucket is only meaningful when the same
            # Host-owned bucketer can reproduce it at dispatch.  Treat a
            # missing/failing live bucketer as provenance loss, never as an
            # external provider outcome whose call may have happened.
            raise CapabilityDenied(
                "semantic transient tenant bucket is unavailable"
            ) from exc
        projection_bucket = job.projection.get("tenant_bucket_sha256")
        if (
            projection_bucket != job.bindings.get("tenant_bucket_sha256")
            or projection_bucket != live_bucket
            or job.projection.get("identity_present")
            is not (
                flow.labels.tenant is not None
                or flow.labels.principal is not None
            )
            or job.projection.get("identity_mixed")
            is not flow.labels.is_mixed_identity
        ):
            raise CapabilityDenied(
                "semantic transient DataFlow identity drifted"
            )
        joined = DataLabels.aggregate((flow.labels, request.data_labels))
        effective_labels = replace(
            joined,
            origin=flow.labels.origin,
            tenant=flow.labels.tenant,
            principal=flow.labels.principal,
            declassification_authority=None,
        )
        effective_flow = replace(flow, labels=effective_labels)
        return replace(request, data_labels=effective_labels), effective_flow

    def _live_approval_assessment_for_job(
        self,
        job: SemanticAssessmentJobRecord,
        *,
        require_pending: bool = True,
    ) -> tuple[
        SemanticAssessmentRequest,
        SemanticApprovalCandidate | None,
        DataFlowContext,
    ]:
        """Re-read live approval evidence without asserting Phase 4 eligibility.

        Shadow classification covers every exact Host-owned approval domain,
        including actions that are intentionally absent from the frozen
        auto-approval catalog and requests whose Task ceiling does not match.
        Candidate existence, exact tenant bucketing, and catalog eligibility
        are settlement predicates; treating them as classifier-ingress
        predicates would silently erase Shadow coverage for Human-owned work.
        """

        if self._human_request_reader is None or job.request_id is None:
            raise CapabilityDenied(
                "semantic live Human request reader is unavailable"
            )
        request = self._human_request_reader(job.request_id)
        expected_revision = job.projection.get("request_revision")
        if (
            not isinstance(request, HumanRequest)
            or type(expected_revision) is not int
            or request.revision != expected_revision
            or _sha256(request.payload) != job.bindings.get("input_sha256")
        ):
            raise CapabilityDenied("semantic live Human request changed")
        if require_pending and request.status is not HumanRequestStatus.PENDING:
            raise CapabilityDenied("semantic Human request is already terminal")
        try:
            live, candidate, violations = self._prepare_human_approval(
                request,
                job.bindings["input_sha256"],
            )
        except Exception as exc:
            raise CapabilityDenied(
                "semantic live approval provenance is unavailable"
            ) from exc
        if violations:
            raise CapabilityDenied(
                "semantic live Human request has deterministic violations"
            )
        self._require_live_assessment_job_bindings(job, live)
        flow = self._human_flow(request.payload)
        if (
            flow.labels.to_dict() != live.data_labels.to_dict()
            or flow.source_refs_hash() != live.source_refs_sha256
        ):
            raise CapabilityDenied("semantic live DataFlow context drifted")
        return live, candidate, flow

    def _live_approval_for_job(
        self,
        job: SemanticAssessmentJobRecord,
        *,
        require_pending: bool = True,
    ) -> tuple[
        SemanticAssessmentRequest,
        SemanticApprovalCandidate,
        DataFlowContext,
    ]:
        """Return only a live Phase 4 authority candidate.

        This stricter path is used by the broker/settlement boundary, never by
        classifier capture.  It deliberately retains the exact Task ceiling,
        tenant, projection, and frozen-candidate checks.
        """

        live, candidate, flow = self._live_approval_assessment_for_job(
            job,
            require_pending=require_pending,
        )
        if candidate is None:
            raise CapabilityDenied(
                "semantic live Human request is not a catalog candidate"
            )
        self._require_live_authority_job_bindings(job, live, candidate)
        return live, candidate, flow

    def _require_live_assessment_job_bindings(
        self,
        job: SemanticAssessmentJobRecord,
        live: SemanticAssessmentRequest,
    ) -> None:
        expected = {
            "manifest_sha256": live.manifest_sha256,
            "policy_sha256": live.policy_sha256,
            "resource_sha256": live.resource_sha256,
            "args_sha256": live.args_sha256,
            "state_sha256": live.state_sha256,
            "source_refs_sha256": live.source_refs_sha256,
            "data_labels_sha256": live.data_labels_sha256,
            "sink_identity_sha256": live.sink_identity_sha256,
            "tool_schema_sha256": live.tool_schema_sha256,
            "provider_spec_sha256": live.provider_spec_sha256,
        }
        if any(job.bindings.get(key) != value for key, value in expected.items()):
            raise CapabilityDenied("semantic live approval provenance drifted")
        if (
            job.projection.get("projection_mode") != "metadata_only"
            or "redacted_intent" in job.projection
        ):
            raise CapabilityDenied(
                "semantic approval classifier projection is not metadata-only"
            )

    def _require_live_authority_job_bindings(
        self,
        job: SemanticAssessmentJobRecord,
        live: SemanticAssessmentRequest,
        candidate: SemanticApprovalCandidate,
    ) -> None:
        bucket = self._tenant_bucket(live.data_labels.tenant)
        if (
            live.data_labels.is_mixed_identity
            or live.data_labels.tenant is None
            or not isinstance(bucket, str)
            or bucket != job.bindings.get("tenant_bucket_sha256")
            or bucket != job.projection.get("tenant_bucket_sha256")
        ):
            raise CapabilityDenied("semantic live approval tenant is not exact")
        if not self._candidate_matches_job(candidate, job):
            raise CapabilityDenied("semantic live Task ceiling drifted")

    @staticmethod
    def _candidate_matches_job(
        candidate: SemanticApprovalCandidate,
        job: SemanticAssessmentJobRecord,
    ) -> bool:
        frozen = job.projection.get("candidate")
        try:
            return (
                SemanticApprovalCandidateSnapshotV1.from_candidate(
                    candidate
                ).to_dict()
                == frozen
            )
        except (TypeError, ValueError):
            return False

    def _terminalize(
        self,
        job: SemanticAssessmentJobRecord,
        assessment: SemanticAssessment,
        completed_at: str,
        latency_ms: int,
        *,
        target_status: SemanticAssessmentJobStatus | None = None,
        error_code: str | None = None,
        usage_telemetry: SemanticUsageTelemetry | None = None,
    ) -> bool:
        changed = self._terminalize_impl(
            job,
            assessment,
            completed_at,
            latency_ms,
            target_status=target_status,
            error_code=error_code,
            usage_telemetry=usage_telemetry,
        )
        with self._lock:
            self._transient_contexts.pop(job.job_id, None)
        return changed

    def _terminalize_impl(
        self,
        job: SemanticAssessmentJobRecord,
        assessment: SemanticAssessment,
        completed_at: str,
        latency_ms: int,
        *,
        target_status: SemanticAssessmentJobStatus | None = None,
        error_code: str | None = None,
        usage_telemetry: SemanticUsageTelemetry | None = None,
    ) -> bool:
        assessment = _merge_local_dlp_assessment(job, assessment)
        human_outcome = self._human_outcome(job.request_id)
        facts = self._facts_from_job(job, assessment, human_outcome)
        candidate = self._candidate_for_job(job)
        decision = self._broker.decide(
            assessment=assessment,
            facts=facts,
            policy_sha256=job.bindings["policy_sha256"],
            candidate=candidate,
            hard_violations=self._hard_violations(job),
        )
        assessment_id = new_id("semantic-assessment")
        selected_status = target_status or _job_status(assessment.status)
        selected_error = error_code
        if selected_error is None:
            selected_error = _TERMINAL_ERROR_CODE.get(assessment.status)
        target = replace(
            job,
            assessment_id=assessment_id,
            status=selected_status,
            revision=job.revision + 1,
            lease_owner_id=None,
            lease_id=None,
            lease_expires_at=None,
            projection={},
            projection_retention=SemanticProjectionRetention.HASH_ONLY,
            projection_expires_at=None,
            error_code=selected_error,
            updated_at=completed_at,
            completed_at=completed_at,
        )
        record = self._assessment_record(
            job,
            assessment,
            decision,
            assessment_id=assessment_id,
            completed_at=completed_at,
            latency_ms=latency_ms,
            human_outcome=human_outcome,
            usage_telemetry=usage_telemetry,
        )
        deny = self._deny_for_job(job)
        if deny is not None and self._deny_settlement is not None:
            try:
                return self._settle_deterministic_deny(
                    job,
                    target,
                    record,
                    deny,
                )
            except ValidationError:
                latest_human_outcome = self._human_outcome(job.request_id)
                record = replace(record, human_outcome=latest_human_outcome)
        if self._should_attempt_auto_settlement(
            job,
            assessment,
            decision,
            target_status=selected_status,
        ):
            try:
                return self._settle_auto_exact_once(
                    job,
                    target,
                    record,
                    assessment,
                    self._authority_candidate_for_job(job),
                )
            except SemanticRateBudgetExceeded:
                latest_human_outcome = self._human_outcome(job.request_id)
                record = replace(record, human_outcome=latest_human_outcome)
                return self._terminalize_non_authority_auto_attempt(
                    job,
                    target,
                    record,
                    candidate,
                    outcome=SemanticMachineSettlementOutcome.BUDGET_EXHAUSTED,
                    reason=SemanticReasonCode.BUDGET_EXHAUSTED,
                )
            except CapabilityDenied:
                latest_human_outcome = self._human_outcome(job.request_id)
                record = replace(record, human_outcome=latest_human_outcome)
                return self._terminalize_non_authority_auto_attempt(
                    job,
                    target,
                    record,
                    candidate,
                    outcome=SemanticMachineSettlementOutcome.REQUIRE_HUMAN,
                    reason=SemanticReasonCode.MISSING_AUTHORITATIVE_PREDICATE,
                )
            except ValidationError:
                # A Human response/cancel, policy drift, budget rejection, or
                # job CAS loss is a safe race loss.  The settlement owns a
                # shared outer transaction, so no Capability or Human state
                # survives this exception.  Preserve the late assessment as
                # historical evidence without retrying the classifier or
                # overwriting the Human winner.
                latest_human_outcome = self._human_outcome(job.request_id)
                record = replace(
                    record,
                    human_outcome=latest_human_outcome,
                )
                return self._terminalize_non_authority_auto_attempt(
                    job,
                    target,
                    record,
                    candidate,
                    outcome=(
                        SemanticMachineSettlementOutcome.RACE_LOST
                        if latest_human_outcome not in {None, "pending"}
                        else SemanticMachineSettlementOutcome.STALE
                    ),
                    reason=(
                        SemanticReasonCode.REVISION_RACE_LOST
                        if latest_human_outcome not in {None, "pending"}
                        else SemanticReasonCode.DIGEST_DRIFT
                    ),
                )
        terminalized = self._repository.terminalize_semantic_assessment_job(
            job,
            target,
            record,
        )
        if terminalized:
            self._append_flow_assessment_findings(
                job,
                assessment_id=assessment_id,
                findings=assessment.data_findings,
            )
        return terminalized

    def _append_flow_assessment_findings(
        self,
        job: SemanticAssessmentJobRecord,
        *,
        assessment_id: str,
        findings: tuple[SemanticDataFinding, ...],
    ) -> None:
        """Link monotonic classifier findings to their payload-free entity.

        The assessment/job ledger is authoritative and commits first.  Flow
        assertion capture is append-only observational evidence; failure must
        make future auto approval ineligible through incomplete coverage, not
        alter a Human decision or provider result.
        """

        flow = self._flow_graph
        if flow is None or not findings or job.pid is None:
            return
        input_sha256 = job.bindings.get("input_sha256")
        state_sha256 = job.bindings.get("state_sha256")
        if not isinstance(input_sha256, str) or not isinstance(state_sha256, str):
            return
        try:
            if job.kind == SemanticAssessmentKind.ROOT_GOAL.value:
                entity_id = root_goal_entity_id(
                    pid=job.pid,
                    content_sha256=input_sha256,
                    state_sha256=state_sha256,
                )
            elif (
                job.kind == SemanticAssessmentKind.PROVIDER_INGRESS.value
                and job.effect_id is not None
            ):
                entity_id = provider_result_entity_id(
                    pid=job.pid,
                    effect_id=job.effect_id,
                    result_sha256=input_sha256,
                    state_sha256=state_sha256,
                )
            else:
                return
            _host_findings, deterministic_findings = (
                _local_dlp_assessment_findings(job)
            )
            model_findings = tuple(
                finding
                for finding in findings
                if finding not in deterministic_findings
            )
            if deterministic_findings:
                flow.append_assessment_findings(
                    entity_id=entity_id,
                    assessment_id=assessment_id,
                    findings=deterministic_findings,
                    source="deterministic",
                )
            if model_findings:
                flow.append_assessment_findings(
                    entity_id=entity_id,
                    assessment_id=assessment_id,
                    findings=model_findings,
                    source="model",
                )
        except Exception:
            # The flow service already records exactly one capture failure.
            return

    def _deny_for_job(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> DeterministicDenyDecision | None:
        if (
            job.kind != SemanticAssessmentKind.APPROVAL.value
            or job.request_id is None
            or self._human_request_reader is None
        ):
            return None
        with self._lock:
            if self._mode not in _ENFORCEMENT_MODES:
                return None
        try:
            request = self._human_request_reader(job.request_id)
        except Exception:
            return None
        if not isinstance(request, HumanRequest):
            return None
        expected_revision = job.projection.get("request_revision")
        if expected_revision != request.revision:
            return None
        return self.deterministic_deny_preflight(request)

    def _settle_deterministic_deny(
        self,
        job: SemanticAssessmentJobRecord,
        target: SemanticAssessmentJobRecord,
        record: SemanticAssessmentRecord,
        decision: DeterministicDenyDecision,
    ) -> bool:
        settlement = self._deny_settlement
        if settlement is None or job.request_id is None:
            raise ValidationError("semantic deny settlement is unavailable")
        rejected_record = replace(record, human_outcome="rejected")

        def terminalize() -> bool:
            return self._repository.terminalize_semantic_assessment_job(
                job,
                target,
                rejected_record,
            )

        settlement.settle_deny(
            request_id=job.request_id,
            expected_revision=decision.request_revision,
            decision=decision,
            semantic_terminalizer=terminalize,
        )
        return True

    def _should_attempt_auto_settlement(
        self,
        job: SemanticAssessmentJobRecord,
        assessment: SemanticAssessment,
        decision: ShadowPolicyDecision,
        *,
        target_status: SemanticAssessmentJobStatus,
    ) -> bool:
        with self._lock:
            mode = self._mode
            unsafe_review_latched = self._unsafe_review_latched
        return bool(
            not unsafe_review_latched
            and mode == "canary_auto"
            and self._auto_settlement is not None
            and job.kind == SemanticAssessmentKind.APPROVAL.value
            and job.request_id is not None
            and target_status is SemanticAssessmentJobStatus.SUCCEEDED
            and assessment.status is SemanticAssessmentStatus.SUCCESS
            and not self._job_is_catalog_outside(job)
            and decision.outcome is ShadowPolicyOutcome.WOULD_ISSUE_EXACT_ONCE
        )

    def _settle_auto_exact_once(
        self,
        job: SemanticAssessmentJobRecord,
        target: SemanticAssessmentJobRecord,
        record: SemanticAssessmentRecord,
        assessment: SemanticAssessment,
        candidate: SemanticApprovalCandidate | None,
    ) -> bool:
        if candidate is None or job.request_id is None:
            raise ValidationError("semantic auto settlement lost its exact candidate")
        request_revision = job.projection.get("request_revision")
        if (
            isinstance(request_revision, bool)
            or not isinstance(request_revision, int)
            or request_revision < 0
        ):
            raise ValidationError(
                "semantic auto settlement has no frozen request revision"
            )
        settlement = self._auto_settlement
        if settlement is None:  # pragma: no cover - guarded by caller
            raise ValidationError("semantic auto settlement is unavailable")
        approved_record = replace(record, human_outcome="approved")

        def terminalize(
            _capability: Any,
            _binding: Any,
            settlement_record: Any,
        ) -> bool:
            # The settlement implementation invokes this callback inside its
            # outer shared UnitOfWork transaction, before Human CAS commit.  A
            # false CAS is promoted to a settlement conflict by the port and
            # rolls the entire transaction back.
            append_settlement = getattr(
                self._repository,
                "append_semantic_machine_settlement",
                None,
            )
            if not callable(append_settlement):
                raise ValidationError(
                    "semantic machine settlement repository is unavailable"
                )
            append_settlement(settlement_record)
            issued_outcome = SemanticMachineOutcomeRecord(
                outcome_id=(
                    "semantic-outcome:"
                    + _sha256(
                        {
                            "schema_version": 1,
                            "lifecycle_slot": "issued",
                            "settlement_id": settlement_record.settlement_id,
                            "effect_id": settlement_record.effect_id,
                            "capability_id": settlement_record.capability_id,
                            "binding_sha256": settlement_record.binding_sha256,
                        }
                    )
                ),
                settlement_id=settlement_record.settlement_id,
                effect_id=settlement_record.effect_id,
                outcome="issued",
                evidence_sha256=settlement_record.decision_sha256,
                created_at=settlement_record.created_at,
            )
            self._repository.append_semantic_machine_outcome_if_absent(
                issued_outcome
            )
            return self._repository.terminalize_semantic_assessment_job(
                job,
                target,
                approved_record,
            )

        settlement.settle_exact_once(
            request_id=job.request_id,
            expected_revision=request_revision,
            job_id=job.job_id,
            assessment_id=record.assessment_id,
            assessment=assessment,
            candidate=candidate,
            semantic_terminalizer=terminalize,
        )
        return True

    def _terminalize_non_authority_auto_attempt(
        self,
        job: SemanticAssessmentJobRecord,
        target: SemanticAssessmentJobRecord,
        record: SemanticAssessmentRecord,
        candidate: SemanticApprovalCandidate | None,
        *,
        outcome: SemanticMachineSettlementOutcome,
        reason: SemanticReasonCode,
    ) -> bool:
        epoch = getattr(self._config, "policy_epoch", None)
        tenant_bucket_sha256 = job.bindings.get("tenant_bucket_sha256")
        action_id = job.projection.get("action_id")
        if (
            candidate is None
            or epoch is None
            or job.request_id is None
            or job.pid is None
            or job.effect_id is None
            or not isinstance(tenant_bucket_sha256, str)
            or not isinstance(action_id, str)
        ):
            return self._repository.terminalize_semantic_assessment_job(
                job,
                target,
                record,
            )
        request_revision = job.projection.get("request_revision")
        if isinstance(request_revision, bool) or not isinstance(request_revision, int):
            return self._repository.terminalize_semantic_assessment_job(
                job,
                target,
                record,
            )
        policy_sha256 = epoch.canonical_sha256()
        settlement = MachinePolicySettlementV1(
            settlement_id=new_id("semantic-settlement"),
            assessment_id=record.assessment_id,
            job_id=job.job_id,
            request_id=job.request_id,
            request_revision=request_revision,
            pid=job.pid,
            operation_id=job.operation_id,
            effect_id=job.effect_id,
            epoch_id=epoch.epoch_id,
            policy_sha256=policy_sha256,
            tenant_bucket_sha256=tenant_bucket_sha256,
            action_id=action_id,
            outcome=outcome,
            capability_id=None,
            binding_sha256=_sha256(
                {
                    "schema_version": 2,
                    "job_id": job.job_id,
                    "projection_sha256": job.projection_sha256,
                }
            ),
            decision_sha256=_sha256(
                {
                    "schema_version": 1,
                    "outcome": outcome.value,
                    "reason": reason.value,
                    "assessment_id": record.assessment_id,
                }
            ),
            matched_rule_id=candidate.rule_id,
            reason_codes=(reason,),
            created_at=record.completed_at,
        )
        with self._repository.transaction():
            self._repository.append_semantic_machine_settlement(settlement)
            changed = self._repository.terminalize_semantic_assessment_job(
                job,
                target,
                record,
            )
            if changed is not True:
                raise ValidationError(
                    "semantic non-authority settlement lost its job CAS"
                )
        return True

    def _assessment_record(
        self,
        job: SemanticAssessmentJobRecord,
        assessment: SemanticAssessment,
        decision: ShadowPolicyDecision,
        *,
        assessment_id: str,
        completed_at: str,
        latency_ms: int,
        human_outcome: str | None,
        usage_telemetry: SemanticUsageTelemetry | None,
    ) -> SemanticAssessmentRecord:
        action_id = job.projection.get("action_id")
        if not isinstance(action_id, str):
            raise ValidationError("semantic job action provenance is missing")
        return SemanticAssessmentRecord(
            assessment_id=assessment_id,
            job_id=job.job_id,
            kind=job.kind,
            status=assessment.status.value,
            domain=job.domain,
            action_id=action_id,
            pid=job.pid,
            request_id=job.request_id,
            operation_id=job.operation_id,
            effect_id=job.effect_id,
            shadow_outcome=decision.outcome.value,
            reason_codes=tuple(item.value for item in decision.reason_codes),
            ood=assessment.ood,
            abstain=assessment.abstain,
            confidence_bps=assessment.confidence_bps,
            calibration_bucket=assessment.calibration_bucket.value,
            classifier_id=self._classifier_id,
            classifier_version=self._classifier_version,
            artifact_sha256=job.bindings["artifact_sha256"],
            input_sha256=job.bindings["input_sha256"],
            feature_snapshot_sha256=job.bindings["feature_snapshot_sha256"],
            policy_sha256=job.bindings["policy_sha256"],
            manifest_sha256=job.bindings.get("manifest_sha256"),
            action_sha256=job.bindings.get("action_sha256") or _sha256(action_id),
            resource_sha256=job.bindings.get("resource_sha256"),
            args_sha256=job.bindings.get("args_sha256"),
            state_sha256=job.bindings.get("state_sha256"),
            projection_sha256=job.projection_sha256,
            created_at=job.created_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            human_outcome=human_outcome,
            findings=tuple(item.to_dict() for item in assessment.findings),
            data_findings=tuple(item.to_dict() for item in assessment.data_findings),
            matched_rule_ids=(
                (decision.matched_rule_id,)
                if decision.matched_rule_id is not None
                else ()
            ),
            proven_predicates=tuple(item.value for item in decision.proven_predicates),
            missing_predicates=tuple(item.value for item in decision.missing_predicates),
            source_refs_sha256=job.bindings.get("source_refs_sha256"),
            data_labels_sha256=job.bindings.get("data_labels_sha256"),
            sink_identity_sha256=job.bindings.get("sink_identity_sha256"),
            tool_schema_sha256=job.bindings.get("tool_schema_sha256"),
            provider_spec_sha256=job.bindings.get("provider_spec_sha256"),
            input_tokens=(
                usage_telemetry.input_tokens
                if usage_telemetry is not None
                else None
            ),
            output_tokens=(
                usage_telemetry.output_tokens
                if usage_telemetry is not None
                else None
            ),
            cost_microunits=(
                usage_telemetry.cost_microunits
                if usage_telemetry is not None
                else None
            ),
            tenant_bucket_sha256=job.bindings.get("tenant_bucket_sha256"),
        )

    def _facts_from_job(
        self,
        job: SemanticAssessmentJobRecord,
        _assessment: SemanticAssessment,
        human_outcome: str | None,
    ) -> AuthoritativeApprovalFacts:
        raw = job.projection.get("features")
        try:
            facts = (
                AuthoritativeApprovalFacts.from_dict(raw)
                if isinstance(raw, Mapping)
                else AuthoritativeApprovalFacts()
            )
        except (TypeError, ValueError):
            return AuthoritativeApprovalFacts()
        if job.kind != SemanticAssessmentKind.APPROVAL.value:
            return facts
        if not facts.schema_valid:
            # Malformed durable fallbacks carry no asserted authoritative
            # predicates and never consult live authority while terminalizing.
            return AuthoritativeApprovalFacts()
        manifest_sha256, policy_sha256 = self._live_manifest_policy(job.pid)
        return replace(
            facts,
            manifest_current=(
                job.bindings.get("manifest_sha256") is not None
                and manifest_sha256 == job.bindings.get("manifest_sha256")
            ),
            policy_current=(
                policy_sha256 is not None
                and policy_sha256 == job.bindings.get("policy_sha256")
            ),
            # Preserve the frozen Host fact. Classifier success proves neither
            # target-operation egress safety nor any other allow predicate.
            data_flow_allowed=facts.data_flow_allowed,
            binding_current=(
                facts.binding_current
                and human_outcome == "pending"
                and self._approval_request_is_current(job)
            ),
        )

    def _approval_request_is_current(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> bool:
        reader = self._human_request_reader
        expected_revision = job.projection.get("request_revision")
        if (
            reader is None
            or job.request_id is None
            or type(expected_revision) is not int
        ):
            return False
        try:
            request = reader(job.request_id)
            return bool(
                isinstance(request, HumanRequest)
                and request.status is HumanRequestStatus.PENDING
                and request.revision == expected_revision
                and _sha256(request.payload)
                == job.bindings.get("input_sha256")
            )
        except Exception:
            return False

    def _candidate_for_job(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> (
        SemanticApprovalCandidate
        | SemanticApprovalCandidateSnapshotV1
        | None
    ):
        if self._human_request_reader is not None and job.request_id is not None:
            try:
                _live, candidate, _flow = self._live_approval_for_job(
                    job,
                    require_pending=False,
                )
                return candidate
            except Exception:
                # The typed digest-only snapshot is valid only for Shadow
                # replay. Currentness remains a separate Host predicate and
                # Phase 4 uses _authority_candidate_for_job exclusively.
                return self._shadow_candidate_from_job(job)
        return self._shadow_candidate_from_job(job)

    def _authority_candidate_for_job(
        self,
        job: SemanticAssessmentJobRecord,
    ) -> SemanticApprovalCandidate | None:
        try:
            _live, candidate, _flow = self._live_approval_for_job(
                job,
                require_pending=True,
            )
            return candidate
        except Exception:
            return None

    @staticmethod
    def _shadow_candidate_from_job(
        job: SemanticAssessmentJobRecord,
    ) -> SemanticApprovalCandidateSnapshotV1 | None:
        frozen = job.projection.get("candidate")
        if not isinstance(frozen, Mapping):
            return None
        try:
            return SemanticApprovalCandidateSnapshotV1.from_dict(frozen)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hard_violations(
        job: SemanticAssessmentJobRecord,
    ) -> tuple[SemanticReasonCode, ...]:
        raw = job.projection.get("hard_violations", [])
        if not isinstance(raw, list):
            return ()
        try:
            return tuple(dict.fromkeys(SemanticReasonCode(item) for item in raw))
        except (TypeError, ValueError):
            return ()

    @staticmethod
    def _job_is_catalog_outside(job: SemanticAssessmentJobRecord) -> bool:
        """Return whether one well-formed approval is outside catalog v1."""

        if job.kind != SemanticAssessmentKind.APPROVAL.value:
            return False
        raw_features = job.projection.get("features")
        action_id = job.projection.get("action_id")
        if not isinstance(raw_features, Mapping) or not isinstance(action_id, str):
            return False
        try:
            facts = AuthoritativeApprovalFacts.from_dict(raw_features)
        except (TypeError, ValueError):
            return False
        if not (
            facts.schema_valid
            and facts.request_is_exact_external_operation
        ):
            return False
        action = DEFAULT_ACTION_ONTOLOGY.resolve(action_id)
        return action is None or not action.auto_approval_eligible

    def _human_outcome(self, request_id: str | None) -> str | None:
        if request_id is None or self._human_outcome_reader is None:
            return None
        try:
            outcome = self._human_outcome_reader(request_id)
        except Exception:
            return None
        return outcome if isinstance(outcome, str) else None

    def _run_maintenance_batch(self, mode: str) -> bool:
        if not self._maintenance_lock.acquire(blocking=False):
            return False
        try:
            budget = self._config.recovery_batch_limit
            if mode == "off":
                attempted, changed = self._terminalize_job_status_batch(
                    source=SemanticAssessmentJobStatus.CLAIMED,
                    assessment_status=(
                        SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN
                    ),
                    target=(
                        SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN
                    ),
                    error_code="provider_outcome_unknown",
                    limit=budget,
                )
                remaining = max(0, budget - attempted)
                if remaining:
                    _attempted, queued_changed = self._terminalize_job_status_batch(
                        source=SemanticAssessmentJobStatus.QUEUED,
                        assessment_status=SemanticAssessmentStatus.SKIPPED_POLICY,
                        target=SemanticAssessmentJobStatus.CANCELLED,
                        error_code="disabled",
                        limit=remaining,
                    )
                    changed += queued_changed
                return changed > 0
            if mode not in _ACTIVE_MODES:
                return False
            attempted, changed = self._terminalize_expired_claims(limit=budget)
            remaining = max(0, budget - attempted)
            if remaining:
                _attempted, queued_changed = self._terminalize_expired_queued(
                    limit=remaining
                )
                changed += queued_changed
            return changed > 0
        finally:
            self._maintenance_lock.release()

    def _terminalize_expired_claims(self, *, limit: int | None = None) -> tuple[int, int]:
        selected_limit = min(limit or self._config.recovery_batch_limit, 500)
        now = utc_now()
        jobs = self._repository.query_expired_semantic_assessment_jobs(
            expired_before=now,
            limit=selected_limit,
        )
        changed = sum(
            bool(
                self._terminalize(
                    job,
                    SemanticAssessment(
                        status=SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN
                    ),
                    now,
                    0,
                    target_status=(
                        SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN
                    ),
                    error_code="provider_outcome_unknown",
                )
            )
            for job in jobs
        )
        return len(jobs), changed

    def _terminalize_expired_queued(self, *, limit: int | None = None) -> tuple[int, int]:
        selected_limit = min(limit or self._config.recovery_batch_limit, 500)
        now = utc_now()
        jobs = self._repository.query_semantic_assessment_jobs(
            statuses=(SemanticAssessmentJobStatus.QUEUED.value,),
            projection_expires_before=now,
            limit=selected_limit,
        )
        changed = sum(
            bool(
                self._terminalize(
                    job,
                    SemanticAssessment(status=SemanticAssessmentStatus.STALE_INPUT),
                    now,
                    0,
                    target_status=SemanticAssessmentJobStatus.EXPIRED,
                    error_code="projection_expired",
                )
            )
            for job in jobs
        )
        return len(jobs), changed

    def _terminalize_claimed_unknown(self) -> tuple[int, int]:
        return self._terminalize_job_status_batch(
            source=SemanticAssessmentJobStatus.CLAIMED,
            assessment_status=SemanticAssessmentStatus.PROVIDER_OUTCOME_UNKNOWN,
            target=SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN,
            error_code="provider_outcome_unknown",
            limit=self._config.recovery_batch_limit,
        )

    def _cancel_queued_disabled(self) -> tuple[int, int]:
        return self._terminalize_job_status_batch(
            source=SemanticAssessmentJobStatus.QUEUED,
            assessment_status=SemanticAssessmentStatus.SKIPPED_POLICY,
            target=SemanticAssessmentJobStatus.CANCELLED,
            error_code="disabled",
            limit=self._config.recovery_batch_limit,
        )

    def _terminalize_job_status_batch(
        self,
        *,
        source: SemanticAssessmentJobStatus,
        assessment_status: SemanticAssessmentStatus,
        target: SemanticAssessmentJobStatus,
        error_code: str,
        limit: int,
    ) -> tuple[int, int]:
        selected_limit = min(limit, 500)
        jobs = self._repository.query_semantic_assessment_jobs(
            statuses=(source.value,),
            projection_expires_before=None,
            limit=selected_limit,
        )
        completed_at = utc_now()
        changed = sum(
            bool(
                self._terminalize(
                    job,
                    SemanticAssessment(status=assessment_status),
                    completed_at,
                    0,
                    target_status=target,
                    error_code=error_code,
                )
            )
            for job in jobs
        )
        return len(jobs), changed

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                progressed = self.process_one()
            except Exception:
                self.record_capture_failure(source="worker")
                progressed = False
            if progressed:
                continue
            with self._lock:
                if self._mode == "off":
                    return
            self._wake.wait(0.05)
            self._wake.clear()

    def _capture_failure_count(self) -> int:
        lock = getattr(self, "_capture_failure_lock", None)
        if lock is None:
            return getattr(self, "_capture_failures", 0)
        with lock:
            return self._capture_failures

    def _increment_capture_failure(self) -> None:
        lock = getattr(self, "_capture_failure_lock", None)
        if lock is None:
            self._capture_failures = getattr(self, "_capture_failures", 0) + 1
            return
        with lock:
            self._capture_failures += 1

    @staticmethod
    def _encode_cursor(cursor: SemanticAssessmentCursor | None) -> str | None:
        if cursor is None:
            return None
        payload = _canonical_bytes(
            {
                "assessment_id": cursor.assessment_id,
                "created_at": cursor.created_at,
            }
        )
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _encode_v6_cursor(cursor: SemanticV6Cursor | None) -> str | None:
        if cursor is None:
            return None
        payload = _canonical_bytes(
            {
                "created_at": cursor.created_at,
                "record_id": cursor.record_id,
            }
        )
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_v6_cursor(value: str | None) -> SemanticV6Cursor | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise ValueError("semantic v6 cursor is invalid")
        try:
            raw = base64.b64decode(
                value + "=" * (-len(value) % 4),
                altchars=b"-_",
                validate=True,
            )
            decoded = bounded_json_loads(raw, max_bytes=2048)
            if not isinstance(decoded, dict) or set(decoded) != {
                "created_at",
                "record_id",
            }:
                raise ValueError
            return SemanticV6Cursor(
                created_at=decoded["created_at"],
                record_id=decoded["record_id"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("semantic v6 cursor is invalid") from exc

    @staticmethod
    def _decode_cursor(value: str | None) -> SemanticAssessmentCursor | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise ValueError("semantic assessment cursor is invalid")
        try:
            raw = base64.b64decode(
                value + "=" * (-len(value) % 4),
                altchars=b"-_",
                validate=True,
            )
            decoded = bounded_json_loads(raw, max_bytes=2048)
            if not isinstance(decoded, dict) or set(decoded) != {
                "assessment_id",
                "created_at",
            }:
                raise ValueError
            return SemanticAssessmentCursor(
                created_at=decoded["created_at"],
                assessment_id=decoded["assessment_id"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("semantic assessment cursor is invalid") from exc


__all__ = ["DeterministicSemanticAssessor", "SemanticManager"]
