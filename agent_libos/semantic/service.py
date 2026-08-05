from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
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
    ObjectMetadata,
)
from agent_libos.models.data_flow import sensitivity_rank
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
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
    ShadowPolicyDecision,
)
from agent_libos.semantic.broker import DeterministicApprovalBroker
from agent_libos.semantic.external import (
    SemanticAssessmentDeadlineExceeded,
    SemanticProviderCallError,
    SemanticProviderResponseError,
    SemanticUsageTelemetry,
)
from agent_libos.semantic.labels import validate_monotonic_data_findings
from agent_libos.semantic.ontology import DEFAULT_ACTION_ONTOLOGY
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
    SemanticProjectionRetention,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable


_ACTION_PART = re.compile(r"[^a-z0-9_]+")
_ROOT_INTENT_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_EMPTY_POLICY = {"schema_version": 1, "rules": []}
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
    return {
        "rule_id": candidate.rule_id,
        "authority_operation": candidate.authority_operation,
        "rights": list(candidate.rights),
        "manifest_id": candidate.manifest_id,
        "manifest_sha256": candidate.manifest_sha256,
        "policy_sha256": candidate.policy_sha256,
        "resource_sha256": _sha256(candidate.resource),
    }


def _candidate_from_projection(
    value: Any,
) -> SemanticApprovalCandidate | None:
    if not isinstance(value, Mapping):
        return None
    resource_sha256 = value.get("resource_sha256")
    if not isinstance(resource_sha256, str):
        return None
    return SemanticApprovalCandidate(
        rule_id=value.get("rule_id"),
        authority_operation=value.get("authority_operation"),
        resource=f"digest:{resource_sha256}",
        rights=tuple(value.get("rights") or ()),
        manifest_id=value.get("manifest_id"),
        manifest_sha256=value.get("manifest_sha256"),
        policy_sha256=value.get("policy_sha256"),
    )


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
    """Host-owned durable Shadow classifier and read-only evidence facade."""

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
        root_goal_reader: Callable[[str], Any | None] | None = None,
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
    ) -> None:
        self._repository = repository
        self._config = config
        self._assessor = assessor or DeterministicSemanticAssessor()
        self._broker = broker or DeterministicApprovalBroker()
        self._authority = authority
        self._processes = processes
        self._objects = objects
        self._human_outcome_reader = human_outcome_reader
        self._root_goal_reader = root_goal_reader
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
        self._capture_failures = 0
        if shutdown_registrar is not None:
            shutdown_registrar(self.shutdown)
        if config.mode == "shadow":
            registrations = (
                (request_capture_registrar, request_capture),
                (spawn_observer_registrar, spawn_observer),
                (result_observer_registrar, result_observer),
            )
            if any(registrar is None or callback is None for registrar, callback in registrations):
                raise RuntimeError(
                    "semantic Shadow runtime observers are not fully configured"
                )
            for registrar, callback in registrations:
                assert registrar is not None and callback is not None
                registrar(callback)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def start(self) -> None:
        with self._lock:
            mode = self._mode
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            if mode == "shadow" and self._threads:
                return
            if mode == "shadow":
                self._stop.clear()
        maintained = self._run_maintenance_batch(mode)
        with self._lock:
            if self._mode != "shadow":
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
        return stopped

    def set_mode(self, mode: str) -> None:
        if mode not in {"off", "shadow"}:
            raise ValueError("semantic mode must be off or shadow")
        with self._lock:
            previous = self._mode
            if previous == "off" and mode == "shadow":
                raise RuntimeError(
                    "semantic Shadow enablement requires Runtime restart and admission"
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

        with self._lock:
            self._mode = "off"
            self._capture_failures += 1
            self._wake.set()
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
                        self._capture_failures += 1

    def record_capture_failure(self, **_metadata: Any) -> None:
        with self._lock:
            self._capture_failures += 1

    def capture_approval(
        self,
        request: HumanRequest,
    ) -> SemanticAssessmentJobRecord | None:
        return self.capture_human_request(request)

    def capture_human_request(
        self,
        request: HumanRequest,
    ) -> SemanticAssessmentJobRecord | None:
        with self._lock:
            if self._mode != "shadow":
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
            if self._mode != "shadow":
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
            if self._mode != "shadow":
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
            if self._mode != "shadow":
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
        return {
            "schema_version": 2,
            "mode": self.mode,
            "adapter": self._adapter,
            "profile_id": self._profile_id,
            "queue": queue,
            "assessments": assessments,
            "actual_auto_approval": {
                "numerator": 0,
                "denominator": 0,
                "rate": None,
            },
        }

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
        return {
            "items": [record.to_dict() for record in page.records],
            "next_cursor": self._encode_cursor(page.next_cursor),
        }

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        record = self._repository.get_semantic_assessment(assessment_id)
        return record.to_dict() if record is not None else None

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
        context = payload.get("context")
        capability = payload.get("requested_once_capability")
        binding = payload.get("effect_binding")
        if not all(isinstance(item, Mapping) for item in (context, capability, binding)):
            raise ValidationError("semantic approval request is not fully bound")
        assert isinstance(context, Mapping)
        assert isinstance(capability, Mapping)
        assert isinstance(binding, Mapping)
        action, resource, rights = self._approval_identity(context, capability)
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
        violations: list[SemanticReasonCode] = []
        action_definition = DEFAULT_ACTION_ONTOLOGY.resolve(action)
        if action_definition is None:
            violations.append(SemanticReasonCode.UNSUPPORTED_ACTION)
        elif not action_definition.auto_approval_eligible:
            violations.append(SemanticReasonCode.HIGH_RISK_ACTION)
        return (
            request_model,
            selected_candidate,
            tuple(dict.fromkeys(violations)),
        )

    @staticmethod
    def _approval_identity(
        context: Mapping[str, Any],
        capability: Mapping[str, Any],
    ) -> tuple[str, str, list[Any]]:
        action = context.get("authority_operation")
        resource = capability.get("resource")
        rights = capability.get("rights")
        if (
            not isinstance(action, str)
            or re.fullmatch(
                r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+",
                action,
            )
            is None
            or not isinstance(resource, str)
            or not resource
            or resource != resource.strip()
            or "\x00" in resource
        ):
            raise ValidationError("semantic approval request identity is malformed")
        if (
            not isinstance(rights, list)
            or not rights
            or any(
                not isinstance(right, str)
                or not right
                or right != right.strip()
                or "\x00" in right
                for right in rights
            )
        ):
            raise ValidationError("semantic approval request rights are malformed")
        return action, resource, rights

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
            source_refs_sha256=_sha256(
                {
                    "source_refs": getattr(provenance, "source_refs", ()),
                    "parent_oids": getattr(provenance, "parent_oids", ()),
                }
            ),
            data_labels_sha256=_identity_safe_labels_sha256(labels),
            redacted_intent=_root_goal_intent(
                payload,
                max_chars=self._config.intent_max_chars,
            ),
        )
        return self._enqueue(
            request,
            candidate=None,
            hard_violations=(),
            local_dlp_findings=_root_goal_dlp_findings(
                payload,
                input_sha256=request.input_sha256,
            ),
        )

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
        return self._enqueue(
            request,
            candidate=None,
            hard_violations=(),
            local_dlp_findings=detector.findings,
        )

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
        if frozen_dlp_findings:
            external_payload["projection_mode"] = "metadata_only"
            external_payload.pop("redacted_intent", None)
            external_payload.pop("redacted_intent_sha256", None)
            external_payload.pop("redacted_intent_truncated", None)
        projection = {
            **external_payload,
            "deadline_at": request.deadline_at,
            "candidate": _candidate_projection(candidate),
            "hard_violations": [item.value for item in hard_violations],
            "identity_present": any(
                value is not None
                for value in (
                    request.data_labels.tenant,
                    request.data_labels.principal,
                )
            ),
            "identity_mixed": request.data_labels.is_mixed_identity,
            # Raw tenant identifiers are never hashed without a Host-owned
            # keyed bucketer; an ordinary SHA-256 remains dictionary-guessable.
            "tenant_bucket_sha256": self._tenant_bucket(
                request.data_labels.tenant
            ),
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
        persisted = self._repository.enqueue_semantic_assessment_job(record)
        self._wake.set()
        return persisted

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
        action = DEFAULT_ACTION_ONTOLOGY.resolve(action_id)
        context_matches = (
            context.get("pid") == pid
            and context.get("authority_operation") == action_id
            and expected_resource == resource
            and context_right in rights
            and action is not None
            and context_right in action.allowed_rights
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
            request = _request_from_job(job)
            if not request.features.schema_valid:
                raise SemanticProviderResponseError(
                    "semantic approval request schema is invalid"
                )
            if _parse_time(utc_now()) >= _parse_time(request.deadline_at):
                raise SemanticAssessmentDeadlineExceeded(
                    "semantic assessment deadline expired before dispatch"
                )
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
        return self._repository.terminalize_semantic_assessment_job(
            job,
            target,
            record,
        )

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
            ),
        )

    @staticmethod
    def _candidate_for_job(
        job: SemanticAssessmentJobRecord,
    ) -> SemanticApprovalCandidate | None:
        try:
            return _candidate_from_projection(job.projection.get("candidate"))
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
            if mode != "shadow":
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
        with self._lock:
            return self._capture_failures

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
