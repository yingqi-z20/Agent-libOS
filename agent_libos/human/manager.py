from __future__ import annotations

import builtins
import contextlib
import hashlib
import hmac
import math
import threading
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Mapping

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.evidence.message_projection import (
    task_run_message_evidence_projection,
)
from agent_libos.models import (
    AuthorityRisk,
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataFlowOutcome,
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataSink,
    DataTrustLevel,
    ProcessMessage,
    ProcessMessageKind,
    SinkTrustLevel,
    sensitivity_rank,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    HumanResponseRequired,
    NotFound,
    ProcessError,
    ValidationError,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.models import (
    EventType,
    HostResumeProcessWait,
    HumanRequest,
    HumanRequestStatus,
    HumanProcessWait,
    KilledProcessOutcome,
    PausedProcessWait,
    ProcessExecutionToken,
    ProcessSignal,
    ProcessStatus,
)
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.process_execution import trusted_process_execution_takeover
from agent_libos.capability.effect_binding import canonical_effect_hash
from agent_libos.storage import AuthorityRepository, ProcessRepository
from agent_libos.substrate import HumanProvider, ProviderEffectNotStarted
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
)
from agent_libos.ports import (
    AuditPort,
    BlockingWorkPort,
    EventPort,
    HumanDataFlowPort,
    OperationPort,
    ProcessMessagePort,
)
from agent_libos.human.delivery import HumanDeliveryService
from agent_libos.human.presentation import HumanPresentationService
from agent_libos.human.requests import HumanRequestService
from agent_libos.utils.serde import dumps, to_jsonable

_SENSITIVE_HUMAN_AUDIT_KEYS = frozenset({"answer", "context", "decision", "message", "payload", "question", "reason"})
_TERMINAL_RETRY_FENCE_KEYS = frozenset(
    {
        "provider_outcome",
        "automatic_retry_disabled",
        "manual_recovery_required",
        "process_reconciliation_required",
    }
)
_DATA_FLOW_CONTEXT_KEY = "_agent_libos_data_flow_context"
_DATA_RELEASE_FOR_REQUEST_KEY = "_agent_libos_data_release_for_request_id"
_DATA_RELEASE_REQUEST_KEY = "_agent_libos_data_release_request_id"
_DATA_RELEASE_REQUESTS_KEY = "_agent_libos_data_release_request_ids"
_DATA_RELEASE_PRESENTATION_KEY = "_agent_libos_data_release_presentation"
_DATA_RELEASE_VISIBLE_KEY = "_agent_libos_data_release_visible"
_DATA_RELEASE_TERMINAL_COMMITTED_KEY = (
    "_agent_libos_data_release_terminal_committed_request_id"
)
_OUTPUT_SNAPSHOT_SHA256_KEY = "_agent_libos_output_snapshot_sha256"
_PRESENTATION_RECEIPT_PER_REQUEST_MULTIPLIER = 4


def _json_size_bytes(value: Any) -> int:
    return len(dumps(to_jsonable(value)).encode("utf-8"))


def _ensure_json_size(value: Any, limit_bytes: int, label: str) -> int:
    size = _json_size_bytes(value)
    if size > limit_bytes:
        raise ValidationError(f"{label} exceeds {limit_bytes} bytes (got {size})")
    return size


def _ensure_json_node_budget(
    nodes: int,
    max_nodes: int,
    label: str,
    *,
    additional_nodes: int = 0,
) -> None:
    if nodes > max_nodes or additional_nodes > max_nodes - nodes:
        raise ValidationError(f"{label} exceeds maximum JSON nodes={max_nodes}")


def _ensure_json_depth(parent_depth: int, max_depth: int, label: str) -> int:
    depth = parent_depth + 1
    if depth > max_depth:
        raise ValidationError(f"{label} exceeds maximum JSON depth={max_depth}")
    return depth


def _ensure_minimum_json_size(
    minimum_text_bytes: int,
    limit_bytes: int,
    label: str,
) -> None:
    if minimum_text_bytes > limit_bytes:
        raise ValidationError(f"{label} exceeds {limit_bytes} bytes")


def _validate_json_value_type(
    value: Any,
    *,
    nodes: int,
    max_nodes: int,
    label: str,
) -> tuple[tuple[Any, ...] | None, int]:
    """Validate the JSON type and return its children and byte delta."""

    if isinstance(value, dict):
        _ensure_json_node_budget(
            nodes,
            max_nodes,
            label,
            additional_nodes=len(value),
        )
        if any(not isinstance(key, str) for key in value):
            raise ValidationError(f"{label} must use string JSON object keys")
        key_bytes = sum(len(key) for key in value)
        children = tuple(value.values())
        return children, key_bytes
    if isinstance(value, list):
        _ensure_json_node_budget(
            nodes,
            max_nodes,
            label,
            additional_nodes=len(value),
        )
        return tuple(value), 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{label} contains a non-finite JSON number")
        return None, 0
    if isinstance(value, str):
        return None, len(value)
    if isinstance(value, bool) or value is None:
        return None, 0
    if isinstance(value, int):
        # floor(log10(2) * bit_length) is a conservative digit floor and
        # avoids converting an attacker-sized integer to text first.
        integer_bytes = max(
            1,
            (abs(value).bit_length() * 30_102) // 100_000,
        ) + int(value < 0)
        return None, integer_bytes
    raise ValidationError(
        f"{label} contains a non-JSON value: {type(value).__name__}"
    )


def _ensure_bounded_json_value(
    value: Any,
    *,
    limit_bytes: int,
    max_depth: int,
    max_nodes: int,
    label: str,
) -> int:
    """Validate an externally supplied JSON value without recursive descent."""

    # Each frame is (value, parent container depth, leaving).  Leaving frames
    # make cycle detection path-local, so an ordinary shared child is counted
    # once per JSON occurrence while an actual cycle fails closed.
    pending: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active_containers: set[int] = set()
    nodes = 0
    minimum_text_bytes = 0
    while pending:
        current, parent_depth, leaving = pending.pop()
        if leaving:
            active_containers.remove(id(current))
            continue

        nodes += 1
        _ensure_json_node_budget(nodes, max_nodes, label)
        children, minimum_byte_delta = _validate_json_value_type(
            current,
            nodes=nodes,
            max_nodes=max_nodes,
            label=label,
        )
        minimum_text_bytes += minimum_byte_delta
        if children is None:
            _ensure_minimum_json_size(minimum_text_bytes, limit_bytes, label)
            continue

        depth = _ensure_json_depth(parent_depth, max_depth, label)
        _ensure_minimum_json_size(minimum_text_bytes, limit_bytes, label)
        identity = id(current)
        if identity in active_containers:
            raise ValidationError(f"{label} contains a cyclic JSON value")
        active_containers.add(identity)
        pending.append((current, parent_depth, True))
        pending.extend((child, depth, False) for child in reversed(children))

    try:
        return _ensure_json_size(value, limit_bytes, label)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not bounded JSON: {type(exc).__name__}") from exc


def _sanitize_human_observability(
    value: Any,
    *,
    preview_chars: int = 256,
    metadata_only: bool = False,
) -> dict[str, Any]:
    jsonable = to_jsonable(value)
    redacted = _redact_human_value(jsonable)
    encoded = dumps(jsonable).encode("utf-8")
    preview = "<redacted protected payload>" if metadata_only else dumps(redacted)
    return {
        "preview": preview[: max(0, preview_chars)],
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "truncated": metadata_only or len(preview) > max(0, preview_chars),
        "redacted": metadata_only or redacted != jsonable,
        "metadata_only": metadata_only,
    }


def _redact_human_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if str(key).lower() in _SENSITIVE_HUMAN_AUDIT_KEYS else _redact_human_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_human_value(item) for item in value]
    return value


class HumanObjectManager:
    """HumanObject primitive: terminal queue, approvals, questions, and output."""

    TERMINAL_PROCESS_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}

    def __init__(
        self,
        processes: ProcessRepository,
        authority: AuthorityRepository,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        provider: HumanProvider,
        protected_operations: ProtectedOperationSDK,
        authority_policy: Any,
        operations: OperationPort,
        requests: ProcessRepository,
        messages: ProcessMessagePort,
        data_flow: HumanDataFlowPort,
        blocking_work: BlockingWorkPort,
        config: AgentLibOSConfig | None = None,
        transitions: ProcessTransitionService | None = None,
        process_terminal_cleanup: Callable[[str], Any] | None = None,
        request_capture: Callable[[HumanRequest], Any] | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.processes = processes
        self.authority = authority
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.provider = provider
        self.protected_operations = protected_operations
        self.authority_policy = authority_policy
        self.operations = operations
        self.data_flow = data_flow
        self._blocking_work = blocking_work
        self._transitions = transitions or ProcessTransitionService(processes)
        self._process_terminal_cleanup = process_terminal_cleanup
        self._request_capture = request_capture
        self._host_request_capture: Callable[[HumanRequest], Any] | None = None
        self.requests = HumanRequestService(requests)
        self.delivery = HumanDeliveryService(provider)
        self.presentation = HumanPresentationService(
            receipt_limit=(
                self.config.gui.snapshot_collection_max_items
                * _PRESENTATION_RECEIPT_PER_REQUEST_MULTIPLIER
            )
        )
        self._messages = messages
        # A terminal is a single human decision stream. Serialize queue-head
        # claims and durable transitions so concurrent drains cannot act on the
        # same request, but never hold this lock across blocking provider I/O.
        # It is re-entrant because approve()/reject() commit under the same
        # transition lock.
        self._terminal_lock = threading.RLock()
        self._terminal_claims: set[str] = set()
        # If even the minimal durable unknown-outcome marker cannot be written,
        # retain a process-lifetime fence so this manager never redispatches a
        # non-idempotent Human provider operation with an ambiguous outcome.
        self._terminal_retry_fences: set[str] = set()
        self._data_release_parent_request: ContextVar[str | None] = ContextVar(
            f"agent_libos_human_release_parent_{id(self)}",
            default=None,
        )
        self._data_release_presentation: ContextVar[str | None] = ContextVar(
            f"agent_libos_human_release_presentation_{id(self)}",
            default=None,
        )

    def set_request_capture(
        self,
        callback: Callable[[HumanRequest], Any] | None,
    ) -> None:
        """Set the optional compatibility observer for persisted requests."""

        if callback is not None and not callable(callback):
            raise TypeError("Human request capture observer must be callable")
        self._request_capture = callback

    def bind_host_request_capture(
        self,
        callback: Callable[[HumanRequest], Any],
    ) -> None:
        """One-shot bind of the Host observer that compatibility APIs cannot clear."""

        if not callable(callback):
            raise TypeError("Host Human request capture observer must be callable")
        if self._host_request_capture is not None:
            raise RuntimeError("Host Human request capture observer is already bound")
        self._host_request_capture = callback

    def _capture_persisted_request(self, request_id: str) -> None:
        callbacks = tuple(
            callback
            for callback in (
                self._host_request_capture,
                self._request_capture,
            )
            if callback is not None
        )
        if not callbacks:
            return
        persisted = self.requests.get(request_id)
        if persisted is None:
            return
        for callback in callbacks:
            try:
                callback(persisted)
            except Exception:
                # Semantic/diagnostic capture is observational. It must never
                # affect publication of the Human request or scheduler state,
                # nor suppress another independently registered observer.
                continue

    def query(
        self,
        pid: str,
        human: str,
        request: dict[str, Any],
        blocking: bool = True,
        *,
        _trusted_data_release: bool = False,
        source_oids: Iterable[str] | None = None,
    ) -> str:
        if request.get("type") == "data_release_approval" and not _trusted_data_release:
            raise ValidationError(
                "data release approvals can only be created by the Host data-flow gate"
            )
        request = dict(request)
        if not _trusted_data_release:
            request.pop(_DATA_RELEASE_FOR_REQUEST_KEY, None)
            request.pop(_DATA_RELEASE_REQUEST_KEY, None)
            request.pop(_DATA_RELEASE_REQUESTS_KEY, None)
            request.pop(_DATA_RELEASE_PRESENTATION_KEY, None)
            request.pop(_DATA_RELEASE_VISIBLE_KEY, None)
            request.pop(_DATA_RELEASE_TERMINAL_COMMITTED_KEY, None)
        request = self._bind_external_operation_approval(request)
        request.pop(_DATA_FLOW_CONTEXT_KEY, None)
        flow = self._request_source_context(
            pid,
            source_oids=source_oids,
            public_metadata=_trusted_data_release,
        )
        request[_DATA_FLOW_CONTEXT_KEY] = flow.to_dict()
        self._precheck_human_egress(
            pid=pid,
            human=human,
            channel=self.config.runtime.terminal_channel,
            context=flow,
            payload=request,
        )
        _ensure_json_size(request, self.config.tools.human_request_payload_max_bytes, "human request payload")
        now = utc_now()
        human_request = HumanRequest(
            request_id=new_id("hreq"),
            pid=pid,
            human=human,
            payload=request,
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=blocking,
            created_at=now,
            updated_at=now,
        )
        request_observation = _sanitize_human_observability(
            self.public_request_payload(human_request),
            metadata_only=(
                sensitivity_rank(flow.labels.sensitivity)
                > sensitivity_rank(DataSensitivity.NORMAL)
            ),
        )
        # Request persistence, scheduler suspension, and observability are one
        # commit. Callers may also enclose this transaction in the same Store
        # unit that reserves and settles one-shot authority.
        with self.requests.transaction():
            process = self.processes.get_process(pid)
            if process is not None and process.status in self.TERMINAL_PROCESS_STATUSES:
                raise ValidationError(
                    f"terminal process cannot create human requests: {pid} status={process.status.value}"
                )
            self.requests.insert(human_request)
            release_parent_id = request.get(_DATA_RELEASE_FOR_REQUEST_KEY)
            if _trusted_data_release and isinstance(release_parent_id, str):
                release_parent = self.requests.get(release_parent_id)
                if release_parent is None:
                    raise ValidationError(
                        f"data release parent Human request not found: {release_parent_id}"
                    )
                presentation = request.get(_DATA_RELEASE_PRESENTATION_KEY)
                presentation_release = isinstance(presentation, str) and bool(presentation)
                if (
                    release_parent.pid != pid
                    or release_parent.human != human
                    or (
                        release_parent.status != HumanRequestStatus.PENDING
                        and not presentation_release
                    )
                ):
                    raise ValidationError(
                        "data release parent Human request is not eligible for this release"
                    )
                parent_payload = dict(release_parent.payload)
                parent_payload[_DATA_RELEASE_REQUEST_KEY] = human_request.request_id
                if isinstance(presentation, str) and presentation:
                    raw_links = parent_payload.get(_DATA_RELEASE_REQUESTS_KEY)
                    links = dict(raw_links) if isinstance(raw_links, Mapping) else {}
                    previous_id = links.get(presentation)
                    if isinstance(previous_id, str) and previous_id != human_request.request_id:
                        previous = self.requests.get(previous_id)
                        if previous is not None and previous.status == HumanRequestStatus.PENDING:
                            self.requests.replace_current(
                                previous,
                                status=HumanRequestStatus.CANCELLED,
                                decision={
                                    "data_release_outcome": "superseded",
                                    "automatic_retry_disabled": True,
                                },
                                updated_at=utc_now(),
                            )
                    links[presentation] = human_request.request_id
                    parent_payload[_DATA_RELEASE_REQUESTS_KEY] = links
                # A presentation release is internal gate state.  It must not
                # mutate the public view whose exact hash the release binds,
                # otherwise creating the release would invalidate itself.
                release_parent = self.requests.replace_current(
                    release_parent,
                    payload=parent_payload,
                    updated_at=(
                        release_parent.updated_at
                        if isinstance(presentation, str) and presentation
                        else utc_now()
                    ),
                )
            self.operations.expect("approval")
            self.operations.link_evidence(
                "human_request",
                human_request.request_id,
                "approval",
                metadata={"status": human_request.status.value, "blocking": blocking},
            )
            if blocking:
                # Blocking human requests suspend scheduling for this process until
                # a terminal queue decision moves it back to RUNNABLE.
                if process is not None:
                    request_ids = tuple(
                        pending.request_id
                        for pending in self.requests.list(
                            pid=pid,
                            status=HumanRequestStatus.PENDING,
                        )
                        if pending.blocking
                    )
                    self._transitions.transition(
                        process.pid,
                        ProcessStatus.WAITING_HUMAN,
                        expected_revision=process.revision,
                        expected_status=process.status,
                        expected_state_generation=process.state_generation,
                        wait_state=HumanProcessWait(request_ids=request_ids),
                    )
            self.events.emit(
                EventType.HUMAN_QUERY,
                source=pid,
                target=f"human:{human}",
                payload={
                    "request_id": human_request.request_id,
                    "request_type": str(request.get("type") or "approval"),
                    "request": request_observation,
                    "blocking": blocking,
                },
            )
            self.audit.record(
                actor=pid,
                action="human.query",
                target=f"human:{human}",
                decision={
                    "request_id": human_request.request_id,
                    "blocking": blocking,
                    "request": request_observation,
                },
            )
        self._capture_persisted_request(human_request.request_id)
        return human_request.request_id

    def request_data_release(
        self,
        *,
        pid: str,
        human: str,
        request: dict[str, Any],
        blocking: bool = True,
    ) -> str:
        """Create the metadata-only Human request owned by DataFlowManager."""

        request = dict(request)
        request.pop(_DATA_RELEASE_FOR_REQUEST_KEY, None)
        request.pop(_DATA_RELEASE_REQUEST_KEY, None)
        request.pop(_DATA_RELEASE_REQUESTS_KEY, None)
        request.pop(_DATA_RELEASE_PRESENTATION_KEY, None)
        request.pop(_DATA_RELEASE_VISIBLE_KEY, None)
        request.pop(_DATA_RELEASE_TERMINAL_COMMITTED_KEY, None)
        if request.get("type") != "data_release_approval":
            raise ValidationError("trusted data release request has an invalid type")
        context = request.get("context")
        once = request.get("requested_once_capability")
        if not isinstance(context, dict) or not isinstance(once, dict):
            raise ValidationError(
                "trusted data release request requires metadata context and an exact capability"
            )
        forbidden = {
            "content",
            "content_preview",
            "payload",
            "params",
            "arguments",
            "question_context",
        }
        if any(key in context for key in forbidden):
            raise ValidationError("data release Human request must not contain payload content")
        parent_request_id = self._data_release_parent_request.get()
        presentation = self._data_release_presentation.get()
        presentation_release = isinstance(presentation, str) and bool(presentation)
        if parent_request_id is not None:
            parent = self.requests.get(parent_request_id)
            if parent is None or (
                parent.status != HumanRequestStatus.PENDING
                and not presentation_release
            ):
                raise CapabilityDenied(
                    "data release parent Human request is no longer pending"
                )
            raw_links = parent.payload.get(_DATA_RELEASE_REQUESTS_KEY)
            links = dict(raw_links) if isinstance(raw_links, Mapping) else {}
            existing_id = (
                links.get(presentation)
                if isinstance(presentation, str) and presentation
                else parent.payload.get(_DATA_RELEASE_REQUEST_KEY)
            )
            if isinstance(existing_id, str):
                existing = self.requests.get(existing_id)
                if (
                    existing is not None
                    and existing.status == HumanRequestStatus.PENDING
                    and existing.payload.get("type") == "data_release_approval"
                    and existing.payload.get("requested_once_capability")
                    == request.get("requested_once_capability")
                ):
                    return existing.request_id
                if existing is not None and existing.status in {
                    HumanRequestStatus.REJECTED,
                    HumanRequestStatus.CANCELLED,
                }:
                    raise CapabilityDenied(
                        "data release for this Human request was already denied"
                    )
            request[_DATA_RELEASE_FOR_REQUEST_KEY] = parent_request_id
            if isinstance(presentation, str) and presentation:
                request[_DATA_RELEASE_PRESENTATION_KEY] = presentation
        return self.query(
            pid=pid,
            human=human,
            request=request,
            blocking=blocking and not presentation_release,
            _trusted_data_release=True,
        )

    def request_permission(
        self,
        pid: str,
        human: str,
        resource: str,
        rights: list[str],
        reason: str,
        blocking: bool = True,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> str:
        selected_human = (human or "").strip() or self.config.runtime.default_human
        authority_spec = self.authority_policy.assert_capability_request(
            pid,
            resource,
            rights,
        )
        request = self._permission_request_payload(
            pid,
            resource,
            rights,
            reason,
            expires_at=authority_spec.get("expires_at"),
        )
        decision = self.capabilities.require(
            pid,
            f"human:{selected_human}",
            CapabilityRight.WRITE,
            consume=False,
        )
        # Reservation, request publication, scheduler wait state, evidence, and
        # settlement are one durable unit.  Nested component transactions join
        # this Store transaction through savepoints, so any failure restores
        # the exact authority and leaves no observable request behind.
        with self.requests.transaction():
            reservation_id = self._reserve_one_time_decision(decision, used_by="human")
            request_id = self.query(
                pid=pid,
                human=selected_human,
                request=request,
                blocking=blocking,
                source_oids=source_oids,
            )
            self._require_one_time_decision_commit(reservation_id)
        return request_id

    def _bind_external_operation_approval(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("type") != "external_operation_approval":
            return request
        context = request.get("context")
        if not isinstance(context, dict):
            raise ValidationError("external operation approval requires an object context")
        selected = dict(request)
        binding = {
            "effect_id": new_id("eff"),
            "canonical_args_hash": canonical_effect_hash(context),
            "target_state_version": context.get("target_state_version"),
        }
        selected["effect_binding"] = binding
        once = selected.get("requested_once_capability")
        if isinstance(once, dict):
            constrained = dict(once)
            constraints = dict(constrained.get("constraints") or {})
            constraints[CapabilityManager.APPROVAL_BINDING_KEY] = binding
            constrained["constraints"] = constraints
            selected["requested_once_capability"] = constrained
        return selected

    def _permission_request_payload(
        self,
        pid: str,
        resource: str,
        rights: list[str],
        reason: str,
        *,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        pattern = self.capabilities.parse_resource_pattern(resource)
        try:
            normalized_rights = [CapabilityRight(str(right)).value for right in rights]
        except ValueError as exc:
            raise ValidationError(f"unknown capability right: {exc}") from exc
        if not normalized_rights:
            raise ValidationError("permission request must include at least one right")
        self._reject_broad_model_permission_request(pattern.raw, pattern.kind, pattern.body, pattern.scope.value, normalized_rights)
        constraints = self._permission_constraints(resource, pattern.kind, normalized_rights)
        risk = self._permission_risk(pattern.kind, normalized_rights, constraints)
        lease = {
            "type": "human_selected_policy",
            "choices": [
                CapabilityManager.ALWAYS_ALLOW,
                CapabilityManager.ASK_EACH_TIME,
                CapabilityManager.ALWAYS_DENY,
            ],
            "default_if_unanswered": CapabilityManager.ALWAYS_DENY,
            "expires_at": expires_at,
            "uses_remaining": None,
        }
        context = {
            "reason": reason,
            "risk": risk.value,
            "resource": resource,
            "canonical_resource": pattern.raw,
            "resource_kind": pattern.kind,
            "resource_scope": pattern.scope.value,
            "resource_body": pattern.body,
            "rights": normalized_rights,
            "lease": lease,
            "constraints": constraints,
            "request_origin": "model",
        }
        return {
            "type": "permission_request",
            "question": f"Set permission policy for {pattern.raw} rights={normalized_rights}: {reason}",
            "requested_permission": {
                "subject": pid,
                "resource": pattern.raw,
                "rights": normalized_rights,
                "constraints": constraints,
                **({"expires_at": expires_at} if expires_at is not None else {}),
            },
            "context": context,
        }

    def _reject_broad_model_permission_request(
        self,
        resource: str,
        kind: str,
        body: str,
        scope: str,
        rights: list[str],
    ) -> None:
        rights_set = set(rights)
        privileged_rights = {
            CapabilityRight.ADMIN.value,
            CapabilityRight.GRANT.value,
            CapabilityRight.REVOKE.value,
            CapabilityRight.WRITE.value,
            CapabilityRight.EXECUTE.value,
            CapabilityRight.DELETE.value,
        }
        if rights_set & privileged_rights:
            if kind == "capability" or (scope == "prefix" and not body):
                raise ValidationError(
                    "model permission requests cannot ask for broad privileged capability authority; request a concrete non-meta resource instead"
                )
        if kind == "shell" and CapabilityRight.EXECUTE.value in rights_set:
            if resource == self.config.shell.policy_resource or (scope == "prefix" and not body):
                raise ValidationError(
                    "model permission requests cannot ask for broad shell execute authority; request a concrete command class instead"
                )
        if kind == "filesystem" and rights_set & {CapabilityRight.WRITE.value, CapabilityRight.DELETE.value}:
            if resource == "filesystem:*" or body in {"", "/"}:
                raise ValidationError(
                    "model permission requests cannot ask for root/global filesystem write/delete authority; request a workspace, concrete file, or directory subtree"
                )
        if kind == "filesystem" and CapabilityRight.DELETE.value in rights_set:
            if resource == "filesystem:workspace:*" or (scope == "prefix" and body == "workspace"):
                raise ValidationError(
                    "model permission requests cannot ask for workspace-wide delete authority; request a concrete file or directory subtree"
                )

    def _permission_constraints(
        self,
        resource: str,
        kind: str,
        rights: list[str],
    ) -> dict[str, Any]:
        if kind != "shell" or CapabilityRight.EXECUTE.value not in set(rights):
            return {}
        if resource == "shell:git":
            return {AUTHORITY_RULES_KEY: self._git_read_only_authority_rules()}
        raise ValidationError(
            "model shell execute policy requests only support the canonical exact "
            "resource shell:git; other shell authority must be approved through "
            "an exact per-use shell operation"
        )

    def _permission_risk(self, kind: str, rights: list[str], constraints: dict[str, Any]) -> AuthorityRisk:
        rights_set = set(rights)
        if CapabilityRight.DELETE.value in rights_set:
            return AuthorityRisk.DESTRUCTIVE
        if rights_set & {CapabilityRight.ADMIN.value, CapabilityRight.GRANT.value, CapabilityRight.REVOKE.value}:
            return AuthorityRisk.HIGH
        if kind == "shell" and CapabilityRight.EXECUTE.value in rights_set:
            return AuthorityRisk.LOW if constraints else AuthorityRisk.HIGH
        if CapabilityRight.WRITE.value in rights_set or CapabilityRight.EXECUTE.value in rights_set:
            return AuthorityRisk.HIGH
        return AuthorityRisk.LOW

    def _git_read_only_authority_rules(self) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for subcommand in [
            "push",
            "clean",
            "reset",
            "checkout",
            "switch",
            "restore",
            "commit",
            "merge",
            "rebase",
            "tag",
            "remote",
            "fetch",
            "pull",
            "clone",
        ]:
            rules.append(
                {
                    "rule_id": f"shell.git.deny.{subcommand}",
                    "operation": "shell.run",
                    "effect": CapabilityEffect.DENY.value,
                    "risk": AuthorityRisk.HIGH.value,
                    "conditions": {"argv": ["git", subcommand], "match": "prefix"},
                    "description": f"deny git {subcommand} from read-only git command authority",
                }
            )
        for argv in [
            ["git", "status"],
            ["git", "status", "--short"],
            ["git", "branch", "--show-current"],
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "diff"],
            ["git", "diff", "--stat"],
        ]:
            rules.append(
                {
                    "rule_id": f"shell.git.allow.{'.'.join(argv[1:])}",
                    "operation": "shell.run",
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": AuthorityRisk.LOW.value,
                    "conditions": {"argv": argv, "match": "exact"},
                    "description": "allow read-only git inspection command",
                }
            )
        return rules

    def ask(
        self,
        pid: str,
        question: str,
        human: str | None = None,
        context: dict[str, Any] | None = None,
        blocking: bool = True,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> str:
        selected_human = human or self.config.runtime.default_human
        resource = f"human:{selected_human}"
        decision = self.capabilities.require(pid, resource, CapabilityRight.WRITE, consume=False)
        with self.requests.transaction():
            reservation_id = self._reserve_one_time_decision(decision, used_by="human")
            request_id = self.query(
                pid=pid,
                human=selected_human,
                request={
                    "type": "question",
                    "question": question,
                    "context": context or {},
                },
                blocking=blocking,
                source_oids=source_oids,
            )
            self._require_one_time_decision_commit(reservation_id)
        return request_id

    def answer_for_request(self, request_id: str) -> str:
        request = self.get(request_id)
        if request.payload.get("type") != "question":
            raise ValidationError(f"human request is not a question: {request_id}")
        if request.status == HumanRequestStatus.PENDING:
            raise HumanResponseRequired(
                request_id=request_id,
                message=f"{request.pid} is waiting for human answer to {request_id}",
            )
        if request.status != HumanRequestStatus.APPROVED:
            raise CapabilityDenied(f"human question {request_id} was not answered: {request.status.value}")
        decision = request.decision or {}
        if "answer" not in decision:
            raise ValidationError(f"human question {request_id} has no answer")
        # A resumed tool call may run in a fresh thread or after reopening the
        # runtime, so its ambient context cannot be trusted to retain the
        # original question's labels. Rehydrate the Host-persisted request
        # context and conservatively aggregate the normal/untrusted Human
        # response before ToolBroker creates the result Object.
        self._observe_human_response(self._request_data_flow_context(request))
        return str(decision["answer"])

    def approve(
        self,
        request_id: str,
        decision: dict[str, Any] | None = None,
        responder: str | None = None,
    ) -> HumanRequest:
        selected_decision: Any = {"approved": True} if decision is None else decision
        if not isinstance(selected_decision, dict):
            raise ValidationError("human decision must be a JSON object")
        return self._decide(
            request_id,
            HumanRequestStatus.APPROVED,
            selected_decision,
            responder or self.config.runtime.default_human_actor,
        )

    def approve_for_presentation(
        self,
        request_id: str,
        *,
        presentation: str,
        decision: dict[str, Any] | None = None,
        responder: str | None = None,
    ) -> HumanRequest:
        """Approve only if the request is currently visible on a Host surface."""

        selected_decision: Any = {"approved": True} if decision is None else decision
        if not isinstance(selected_decision, dict):
            raise ValidationError("human decision must be a JSON object")
        return self._decide(
            request_id,
            HumanRequestStatus.APPROVED,
            selected_decision,
            responder or self.config.runtime.default_human_actor,
            required_presentation=presentation,
        )

    def reject(
        self,
        request_id: str,
        decision: dict[str, Any] | None = None,
        responder: str | None = None,
    ) -> HumanRequest:
        selected_decision: Any = {"approved": False} if decision is None else decision
        if not isinstance(selected_decision, dict):
            raise ValidationError("human decision must be a JSON object")
        return self._decide(
            request_id,
            HumanRequestStatus.REJECTED,
            selected_decision,
            responder or self.config.runtime.default_human_actor,
        )

    def reject_for_presentation(
        self,
        request_id: str,
        *,
        presentation: str,
        decision: dict[str, Any] | None = None,
        responder: str | None = None,
    ) -> HumanRequest:
        """Reject only if the request is currently visible on a Host surface."""

        selected_decision: Any = {"approved": False} if decision is None else decision
        if not isinstance(selected_decision, dict):
            raise ValidationError("human decision must be a JSON object")
        return self._decide(
            request_id,
            HumanRequestStatus.REJECTED,
            selected_decision,
            responder or self.config.runtime.default_human_actor,
            required_presentation=presentation,
        )

    def interrupt(self, pid: str, signal: ProcessSignal | str, payload: dict[str, Any] | None = None) -> str:
        sig = ProcessSignal(signal)
        if sig == ProcessSignal.INTERRUPT:
            raise ProcessError(
                "process interrupt signals are not state transitions; "
                "send a durable interrupt process message instead"
            )
        wait_state = None
        outcome = None
        if sig == ProcessSignal.PAUSE:
            selected_status = ProcessStatus.PAUSED
            wait_state = PausedProcessWait()
        elif sig == ProcessSignal.RESUME:
            selected_status = ProcessStatus.RUNNABLE
        elif sig in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
            selected_status = ProcessStatus.KILLED
            outcome = KilledProcessOutcome(code=sig.value)
        else:  # pragma: no cover - ProcessSignal currently has no other value
            raise ProcessError(f"unsupported process signal: {sig.value}")
        # Keep the terminal lock ordering (terminal -> Store) used by Human
        # decisions, then commit the state CAS, pending-request cancellation,
        # event, and audit as one unit.  Faults in any evidence sink therefore
        # cannot leave a killed process with live Human requests or vice versa.
        with self._terminal_lock:
            process = self.processes.get_process(pid)
            if process is None:
                raise NotFound(f"process not found: {pid}")
            self._transitions.require_signal_preserves_condition_wait(process, sig)
            takeover_scope = contextlib.nullcontext()
            if (
                process.status == ProcessStatus.RUNNING
                and selected_status in {ProcessStatus.PAUSED, ProcessStatus.KILLED}
            ):
                if process.execution_owner_id is None or process.execution_lease_id is None:
                    raise ProcessError(f"running process has no execution lease: {pid}")
                takeover_scope = trusted_process_execution_takeover(
                    pid,
                    source_revision=process.revision,
                    source_state_generation=process.state_generation,
                    source_execution_token=ProcessExecutionToken(
                        pid=pid,
                        generation=process.execution_generation,
                        owner_id=process.execution_owner_id,
                        lease_id=process.execution_lease_id,
                    ),
                    intended_status=selected_status,
                    reason="human interrupt takes over an execution lease",
                    nonce=new_id("process_takeover"),
                    wait_kind=(
                        "paused" if selected_status == ProcessStatus.PAUSED else None
                    ),
                    outcome_code=(
                        sig.value if selected_status == ProcessStatus.KILLED else None
                    ),
                )
            with takeover_scope:
                with self.requests.transaction():
                    process = self._transitions.transition(
                        pid,
                        selected_status,
                        expected_revision=process.revision,
                        expected_status=process.status,
                        expected_state_generation=process.state_generation,
                        wait_state=wait_state,
                        outcome=outcome,
                        status_message=(payload or {}).get("reason"),
                    )
                    if process.status in self.TERMINAL_PROCESS_STATUSES:
                        self.cancel_pending_for_process(
                            pid,
                            actor="human",
                            reason=(payload or {}).get("reason")
                            or f"process interrupted with {sig.value}",
                        )
                    event = self.events.emit(
                        EventType.PROCESS_SIGNAL,
                        source="human",
                        target=pid,
                        payload={"signal": sig.value, "payload": payload or {}},
                    )
                    self.audit.record(
                        actor="human",
                        action="human.interrupt",
                        target=f"process:{pid}",
                        decision={"signal": sig.value, "payload": payload or {}},
                    )
        if (
            process.status in self.TERMINAL_PROCESS_STATUSES
            and self._process_terminal_cleanup is not None
        ):
            self._process_terminal_cleanup(pid)
        return event.event_id

    def send_process_message(
        self,
        recipient_pid: str,
        body: str,
        *,
        kind: ProcessMessageKind | str = ProcessMessageKind.NORMAL,
        human: str | None = None,
        channel: str = "human",
        correlation_id: str | None = None,
        reply_to: str | None = None,
        subject: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ProcessMessage:
        selected_human = human or self.config.runtime.default_human
        selected_kind = ProcessMessageKind(kind)
        message_payload = dict(payload or {})
        message_payload.setdefault("source", "human_input")
        message_payload.setdefault("human", selected_human)
        # ProcessMessageManager already commits the message, wake transition,
        # and its primary evidence together.  This outer shared transaction
        # also includes the Human-specific audit, eliminating a return path
        # that raises after the message has durably become visible.
        with self.requests.transaction():
            message = self._messages.post(
                sender=f"human:{selected_human}",
                recipient_pid=recipient_pid,
                kind=selected_kind,
                channel=channel,
                correlation_id=correlation_id,
                reply_to=reply_to,
                subject=(
                    subject
                    if subject is not None
                    else self._default_message_subject(selected_kind)
                ),
                body=body,
                payload=message_payload,
            )
            self.audit.record(
                actor=f"human:{selected_human}",
                action="human.message",
                target=f"process:{recipient_pid}",
                decision=(
                    task_run_message_evidence_projection(message)
                    or {
                        "message_id": message.message_id,
                        "kind": message.kind.value,
                        "channel": message.channel,
                        "correlation_id": message.correlation_id,
                        "reply_to": message.reply_to,
                        "subject": message.subject,
                    }
                ),
            )
        return message

    def output(
        self,
        pid: str,
        message: str,
        human: str | None = None,
        channel: str | None = None,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected_human = human or self.config.runtime.default_human
        selected_channel = self._normalize_output_channel(channel)
        if len(message) > self.config.tools.human_output_max_chars:
            raise ValidationError(
                f"human output message exceeds max characters={self.config.tools.human_output_max_chars}"
            )
        resource = f"human:{selected_human}"
        decision = self.capabilities.require(pid, resource, CapabilityRight.WRITE, consume=False)
        flow = self._request_source_context(pid, source_oids=source_oids)
        self._precheck_human_egress(
            pid=pid,
            human=selected_human,
            channel=selected_channel,
            context=flow,
            payload=message,
        )
        reservation_id: str | None = None
        request = HumanRequest(
            request_id=new_id("hreq"),
            pid=pid,
            human=selected_human,
            payload={
                "type": "output",
                "message": message,
                "channel": selected_channel,
                _OUTPUT_SNAPSHOT_SHA256_KEY: hashlib.sha256(
                    message.encode("utf-8")
                ).hexdigest(),
                _DATA_FLOW_CONTEXT_KEY: flow.to_dict(),
            },
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=False,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        # Claim the just-inserted row against terminal queue drains, then cross
        # the potentially blocking provider boundary without holding the
        # terminal lock. Exit/cancel can proceed and a late delivery rechecks
        # the durable pending state.
        with self._terminal_lock:
            with self.requests.transaction():
                reservation_id = self._reserve_one_time_decision(decision, used_by="human")
                self.requests.insert(request)
                self.operations.expect("approval")
                self.operations.link_evidence(
                    "human_request",
                    request.request_id,
                    "approval",
                    metadata={"status": request.status.value, "blocking": request.blocking},
                )
            # Only publish the in-memory claim after the durable transaction
            # containing authority, request, and evidence has committed.
            self._terminal_claims.add(request.request_id)
        try:
            try:
                delivered = self._deliver_output_request(request)
            except Exception:
                with self._terminal_lock:
                    with self.requests.transaction():
                        latest = self.requests.get(request.request_id)
                        if latest is None or latest.status == HumanRequestStatus.PENDING:
                            # A pre-effect failure must not leave a retryable
                            # request after returning one-shot authority.
                            if latest is not None:
                                self.requests.replace_current(
                                    latest,
                                    status=HumanRequestStatus.CANCELLED,
                                    decision={
                                        "delivery_committed": False,
                                        "cancelled_before_delivery": True,
                                    },
                                    updated_at=utc_now(),
                                )
                            self._restore_one_time_decision(reservation_id)
                        else:
                            # The provider boundary was crossed or its outcome
                            # is unknown. Never resurrect one-shot authority.
                            self._commit_one_time_decision(reservation_id)
                raise
            with self.requests.transaction():
                self._commit_one_time_decision(reservation_id)
        finally:
            with self._terminal_lock:
                self._terminal_claims.discard(request.request_id)
        return {
            "delivered": True,
            "request_id": delivered.request_id,
            "channel": selected_channel,
            "chars": len(message),
        }

    def _normalize_output_channel(self, channel: str | None) -> str:
        selected = (channel or self.config.runtime.terminal_channel).strip()
        if not selected:
            raise ValidationError("human output channel must be non-empty")
        if len(selected) > 128:
            raise ValidationError("human output channel is too long")
        return selected

    def get(self, request_id: str) -> HumanRequest:
        request = self.requests.get(request_id)
        if request is None:
            raise NotFound(f"human request not found: {request_id}")
        return request

    @staticmethod
    def public_request_payload(request: HumanRequest) -> dict[str, Any]:
        """Return the caller-visible request payload without Host provenance."""

        payload = dict(request.payload)
        payload.pop(_DATA_FLOW_CONTEXT_KEY, None)
        payload.pop(_DATA_RELEASE_FOR_REQUEST_KEY, None)
        payload.pop(_DATA_RELEASE_REQUEST_KEY, None)
        payload.pop(_DATA_RELEASE_REQUESTS_KEY, None)
        payload.pop(_DATA_RELEASE_PRESENTATION_KEY, None)
        payload.pop(_DATA_RELEASE_VISIBLE_KEY, None)
        payload.pop(_DATA_RELEASE_TERMINAL_COMMITTED_KEY, None)
        payload.pop(_OUTPUT_SNAPSHOT_SHA256_KEY, None)
        return payload

    def list_for_presentation(
        self,
        *,
        presentation: str,
        provider: Any,
        pid: str | None = None,
        limit: int | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Project Human requests through one release-aware provider boundary."""

        views, _has_more = self.list_for_presentation_window(
            presentation=presentation,
            provider=provider,
            pid=pid,
            limit=limit,
        )
        return views

    def list_for_presentation_window(
        self,
        *,
        presentation: str,
        provider: Any,
        pid: str | None = None,
        limit: int | None = None,
    ) -> tuple[builtins.list[dict[str, Any]], bool]:
        """Return one bounded presentation window and an exact-more signal.

        A newly created metadata release is included in this same result, ahead
        of its withheld parent, so the first GUI observation is immediately
        actionable. Pending release links are durable and reused across polling
        and Runtime reopen. Projection is deliberately lazy: once the final
        logical window is full, later raw rows are not presented. Therefore a
        protected operation can never consume a release or mark a parent visible
        for a row that the caller will crop from this response.
        """

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValidationError("Human presentation limit must be a positive integer")
        selected = self.list(
            pid=pid,
            limit=None if limit is None else limit + 1,
        )
        views: builtins.list[dict[str, Any]] = []
        emitted: set[str] = set()
        has_more = False

        for index, request in enumerate(selected):
            if request.request_id in emitted:
                continue
            if limit is not None and len(views) >= limit:
                has_more = True
                break
            view = self.present_request_view(
                request,
                presentation=presentation,
                provider=provider,
            )
            payload = view.get("payload", {})
            if (
                isinstance(payload, Mapping)
                and payload.get("type") != "data_release_approval"
                and payload.get("release_required") is True
            ):
                release_id = view.get("release_request_id")
                release = (
                    self.requests.get(release_id)
                    if isinstance(release_id, str) and release_id
                    else None
                )
                if (
                    release is not None
                    and release.request_id not in emitted
                    and release.status == HumanRequestStatus.PENDING
                    and (pid is None or release.pid == pid)
                ):
                    views.append(self.public_request_view(release))
                    emitted.add(release.request_id)
                    if limit is not None and len(views) >= limit:
                        # Only a still-withheld parent can be displaced by its
                        # newly created/reused release. No protected payload was
                        # emitted and no exact release was consumed for it.
                        has_more = True
                        break

            views.append(view)
            emitted.add(request.request_id)
            if limit is not None and len(views) >= limit:
                has_more = any(
                    candidate.request_id not in emitted
                    for candidate in selected[index + 1 :]
                )
                break

        return views, has_more

    def present_request_view(
        self,
        request: HumanRequest,
        *,
        presentation: str,
        provider: Any,
    ) -> dict[str, Any]:
        """Return one request only after its presentation Sink authorizes it."""

        selected_presentation = self.presentation.normalize(presentation)
        # Replayed views are not a new Host-to-provider handoff: the exact
        # bytes were already delivered to this provider session.  Still hold
        # the Store lock through the final current-policy/source check and the
        # replay boundary so a concurrent registry/source mutation cannot
        # slip between the guard and the returned view.  The first delivery
        # continues through ProtectedOperation below and performs its own
        # dispatch-time revalidation.
        with self.processes.locked():
            fresh = self.get(request.request_id)
            if (
                self._has_terminal_retry_fence_marker(fresh)
                or fresh.request_id in self._terminal_retry_fences
            ):
                # A provider may already have observed the request.  GUI/API
                # presentation is another provider boundary, so expose only a
                # metadata-only recovery view until the Host reconciles it.
                return self._withheld_public_request_view(fresh)
            raw = self._raw_public_request_view(fresh)
            if fresh.payload.get("type") == "data_release_approval":
                return raw

            context = self._presentation_data_flow_context(fresh)
            if context is None:
                return self._withheld_public_request_view(fresh)
            manager = self.data_flow
            sink = self._presentation_sink(fresh, selected_presentation)
            view_sha256 = hashlib.sha256(
                dumps(to_jsonable(raw)).encode("utf-8")
            ).hexdigest()
            outcome = manager.classify_egress_snapshot(
                sink=sink,
                context=context,
                allow_recovered_source_snapshots=True,
            )
            release_required = outcome is DataFlowOutcome.RELEASE_REQUIRED
            if (
                outcome is DataFlowOutcome.ALLOW
                and self._presentation_was_delivered(
                    fresh,
                    presentation=selected_presentation,
                    view_sha256=view_sha256,
                    provider=provider,
                )
            ):
                return raw
            if release_required:
                if self._presentation_is_visible(
                    fresh,
                    presentation=selected_presentation,
                    view_sha256=view_sha256,
                ):
                    return raw

                release = self._linked_presentation_release(fresh, selected_presentation)
                if release is not None and release.status in {
                    HumanRequestStatus.PENDING,
                    HumanRequestStatus.REJECTED,
                    HumanRequestStatus.CANCELLED,
                }:
                    return self.public_request_view(fresh)

                # A successful first delivery persists the hidden visibility
                # receipt through one more HumanRequest CAS.  Test an approved
                # release against that exact settled public revision, not the
                # pre-delivery row currently in the Store.
                settled_raw = dict(raw)
                settled_raw["revision"] = fresh.revision + 1
                settled_sha256 = hashlib.sha256(
                    dumps(to_jsonable(settled_raw)).encode("utf-8")
                ).hexdigest()
                if self._presentation_release_can_authorize(
                    fresh,
                    release=release,
                    presentation=selected_presentation,
                    view_sha256=settled_sha256,
                ):
                    raw = settled_raw
                    view_sha256 = settled_sha256
                else:
                    # Creating or replacing a presentation release persists a
                    # hidden link on the parent through HumanRequest CAS.  The
                    # link and the later visibility receipt are excluded from
                    # the public view, but both revision advances are not.
                    # Bind the release to the exact settled revision that the
                    # provider will receive after both CAS transitions;
                    # otherwise the approval immediately becomes stale and
                    # polling creates an endless chain of replacement releases.
                    raw = dict(raw)
                    raw["revision"] = fresh.revision + 2
                    view_sha256 = hashlib.sha256(
                        dumps(to_jsonable(raw)).encode("utf-8")
                    ).hexdigest()

        try:
            return self._protected_presentation_view(
                fresh,
                presentation=selected_presentation,
                provider=provider,
                sink=sink,
                context=context,
                public_view=raw,
                view_sha256=view_sha256,
                release_required=release_required,
            )
        except (HumanApprovalRequired, CapabilityDenied):
            return self._withheld_public_request_view(self.get(fresh.request_id))

    def is_request_withheld_for_presentation(
        self,
        request: HumanRequest | str,
        *,
        presentation: str,
    ) -> bool:
        """Return whether a presentation must refuse a decision for this request.

        This is a read-only check of the same durable exact-release state used
        by :meth:`present_request_view`. It deliberately never calls a provider
        or creates a release request, so response endpoints can fail closed
        without changing the parent request or its process.
        """

        selected_presentation = self.presentation.normalize(presentation)
        request_id = request.request_id if isinstance(request, HumanRequest) else request
        fresh = self.get(request_id)
        if fresh.payload.get("type") == "data_release_approval":
            return False

        context = self._presentation_data_flow_context(fresh)
        if context is None:
            return True
        manager = self.data_flow
        sink = self._presentation_sink(fresh, selected_presentation)
        outcome = manager.classify_egress_snapshot(
            sink=sink,
            context=context,
            allow_recovered_source_snapshots=True,
        )
        if outcome is DataFlowOutcome.ALLOW:
            return False
        if outcome is DataFlowOutcome.DENY:
            return True

        raw = self._raw_public_request_view(fresh)
        view_sha256 = hashlib.sha256(
            dumps(to_jsonable(raw)).encode("utf-8")
        ).hexdigest()
        return not self._presentation_is_visible(
            fresh,
            presentation=selected_presentation,
            view_sha256=view_sha256,
        )

    def _raw_public_request_view(self, request: HumanRequest) -> dict[str, Any]:
        selected = to_jsonable(request)
        if not isinstance(selected, dict):
            raise ValidationError("Human request could not be projected")
        selected["payload"] = self.public_request_payload(request)
        parent_id = request.payload.get(_DATA_RELEASE_FOR_REQUEST_KEY)
        if isinstance(parent_id, str) and parent_id:
            selected["release_for_request_id"] = parent_id
        # Release links are gate metadata, not part of the view handed to the
        # GUI provider.  Withheld projections expose the current release ID so
        # a client can approve it; the released view is independent of that
        # internal link and can therefore be bound without a circular hash.
        return selected

    def _presentation_sink(self, request: HumanRequest, presentation: str) -> DataSink:
        trust_identity = (
            f"human:{request.human}:{self.config.runtime.terminal_channel}"
        )
        return DataSink(
            identity=f"human:{request.human}:{presentation}",
            trust_identity=trust_identity,
        )

    def _linked_presentation_release_id(
        self,
        request: HumanRequest,
        presentation: str,
    ) -> str | None:
        raw_links = request.payload.get(_DATA_RELEASE_REQUESTS_KEY)
        if not isinstance(raw_links, Mapping):
            return None
        selected = raw_links.get(presentation)
        return selected if isinstance(selected, str) and selected else None

    def _linked_presentation_release(
        self,
        request: HumanRequest,
        presentation: str,
    ) -> HumanRequest | None:
        release_id = self._linked_presentation_release_id(request, presentation)
        return self.requests.get(release_id) if release_id is not None else None

    def _presentation_is_visible(
        self,
        request: HumanRequest,
        *,
        presentation: str,
        view_sha256: str,
    ) -> bool:
        raw_visible = request.payload.get(_DATA_RELEASE_VISIBLE_KEY)
        if not isinstance(raw_visible, Mapping):
            return False
        state = raw_visible.get(presentation)
        if not isinstance(state, Mapping):
            return False
        release_id = state.get("release_request_id")
        if (
            state.get("view_sha256") != view_sha256
            or not isinstance(release_id, str)
            or release_id != self._linked_presentation_release_id(request, presentation)
        ):
            return False
        release = self.requests.get(release_id)
        return self._presentation_release_authority_matches(
            request,
            release=release,
            presentation=presentation,
            view_sha256=view_sha256,
            uses_remaining=0,
        )

    def _presentation_release_can_authorize(
        self,
        request: HumanRequest,
        *,
        release: HumanRequest | None,
        presentation: str,
        view_sha256: str,
    ) -> bool:
        """Return whether the current one-shot release can authorize this view."""

        return self._presentation_release_authority_matches(
            request,
            release=release,
            presentation=presentation,
            view_sha256=view_sha256,
            uses_remaining=1,
        )

    def _presentation_release_authority_matches(
        self,
        request: HumanRequest,
        *,
        release: HumanRequest | None,
        presentation: str,
        view_sha256: str,
        uses_remaining: int,
    ) -> bool:
        """Validate one linked release against current Host state and authority."""

        if release is None or release.status != HumanRequestStatus.APPROVED:
            return False
        once = release.payload.get("requested_once_capability")
        if not isinstance(once, Mapping):
            return False
        resource = once.get("resource")
        constraints = once.get("constraints")
        manager = self.data_flow
        if not isinstance(resource, str) or not isinstance(constraints, Mapping):
            return False
        binding = constraints.get(manager.RELEASE_BINDING_KEY)
        context = self._presentation_data_flow_context(request)
        if context is None or not manager.is_release_binding_current(
            pid=request.pid,
            sink=self._presentation_sink(request, presentation),
            context=context,
            payload_hash=view_sha256,
            operation=f"human.{presentation}.present",
            target_state_version=None,
            binding=binding,
            allow_recovered_source_snapshots=True,
        ):
            return False
        return any(
            capability.resource == resource
            and capability.constraints == dict(constraints)
            and capability.uses_remaining == uses_remaining
            for capability in self.authority.list_capabilities(subject=request.pid)
        )

    def _presentation_was_delivered(
        self,
        request: HumanRequest,
        *,
        presentation: str,
        view_sha256: str,
        provider: Any,
    ) -> bool:
        """Return whether this exact unrestricted view was already delivered.

        The current Sink is still classified before this check.  The receipt
        only suppresses a duplicate provider/evidence operation when the
        current policy remains ALLOW and the public view hash is unchanged.
        """

        return self.presentation.was_delivered(
            presentation=presentation,
            request_id=request.request_id,
            provider=provider,
            view_sha256=view_sha256,
        )

    def _mark_presentation_delivered(
        self,
        request: HumanRequest,
        *,
        presentation: str,
        view_sha256: str,
        provider: Any,
    ) -> None:
        self.presentation.mark_delivered(
            presentation=presentation,
            request_id=request.request_id,
            provider=provider,
            view_sha256=view_sha256,
        )

    def _protected_presentation_view(
        self,
        request: HumanRequest,
        *,
        presentation: str,
        provider: Any,
        sink: DataSink,
        context: DataFlowContext,
        public_view: dict[str, Any],
        view_sha256: str,
        release_required: bool,
    ) -> dict[str, Any]:
        release_request_id = (
            self._linked_presentation_release_id(request, presentation)
            if release_required
            else None
        )
        public_payload = public_view.get("payload")
        request_kind = (
            str(public_payload.get("type") or "approval")
            if isinstance(public_payload, Mapping)
            else "approval"
        )
        # Provider evidence uses a cross-presentation semantic identity rather
        # than the GUI payload schema spelling.  Terminal permission prompts
        # already record ``approval``; keep GUI presentation evidence aligned
        # so benchmark matching and audit consumers do not depend on channel.
        if request_kind == "permission_request":
            request_kind = "approval"
        observation = _sanitize_human_observability(public_view, metadata_only=True)
        presentation_attempt_id = new_id("hpres")
        effect_context = {
            "request_id": request.request_id,
            "request_kind": request_kind,
            "purpose": f"{presentation}_presentation",
            "operation": "write",
            "channel": presentation,
            "chars": observation["bytes"],
            "prompt_observation": observation,
        }
        invocation = ProtectedOperationInvocation(
            pid=request.pid,
            actor=request.pid,
            target=f"human:{request.human}",
            canonical_args={
                "request_id": request.request_id,
                "presentation": presentation,
                "view_sha256": view_sha256,
                "release_request_id": release_request_id,
                "presentation_attempt_id": presentation_attempt_id,
            },
            observation=effect_context,
            idempotency_key=(
                f"human:{presentation}:present:{request.request_id}:"
                f"{release_request_id or 'without-release'}:{view_sha256}:"
                f"{presentation_attempt_id}"
            ),
            data_sink=sink,
            data_flow_context=context,
            data_flow_payload=public_view,
            data_flow_operation=f"human.{presentation}.present",
            data_flow_allow_recovered_source_snapshots=True,
            failure_evidence=lambda error, phase: self._protected_terminal_evidence(
                request,
                operation="write",
                resource=f"human:{request.human}",
                purpose=f"{presentation}_presentation",
                prompt_observation={"chars": observation["bytes"], **observation},
                result_observation={"type": type(error).__name__},
                failed=True,
                phase=phase,
            ),
        )

        def mark_visible() -> None:
            latest = self.requests.get(request.request_id)
            if latest is None:
                raise NotFound(f"human request not found: {request.request_id}")
            latest_view = self._raw_public_request_view(latest)
            # The receipt commit itself advances HumanRequest revision.  The
            # provider payload and release binding are for that durable target
            # view, not the transient pre-receipt row.
            latest_view["revision"] = latest.revision + 1
            latest_sha256 = hashlib.sha256(
                dumps(to_jsonable(latest_view)).encode("utf-8")
            ).hexdigest()
            if latest_sha256 != view_sha256:
                raise CapabilityDenied("Human GUI view changed before presentation")
            release_id = self._linked_presentation_release_id(latest, presentation)
            if release_id is None:
                raise CapabilityDenied("Human GUI release link is missing")
            raw_visible = latest.payload.get(_DATA_RELEASE_VISIBLE_KEY)
            visible = dict(raw_visible) if isinstance(raw_visible, Mapping) else {}
            visible[presentation] = {
                "release_request_id": release_id,
                "view_sha256": view_sha256,
            }
            latest_payload = dict(latest.payload)
            latest_payload[_DATA_RELEASE_VISIBLE_KEY] = visible
            self.requests.replace_current(latest, payload=latest_payload)

        parent_token = self._data_release_parent_request.set(request.request_id)
        presentation_token = self._data_release_presentation.set(presentation)
        try:
            with self._protected().start(
                "primitive.human.write",
                invocation,
                provider=provider,
            ) as protected:
                presented = protected.call(
                    ProviderPhase(
                        "gui_presentation",
                        state_mutation=False,
                        information_flow=True,
                    ),
                    provider.present,
                    public_view,
                )
                if not isinstance(presented, dict):
                    raise ValidationError("Human presentation provider returned an invalid view")
                if presented != public_view:
                    raise ValidationError(
                        "Human GUI presentation provider altered the release-bound view"
                    )
                return protected.complete(
                    presented,
                    self._protected_terminal_evidence(
                        request,
                        operation="write",
                        resource=f"human:{request.human}",
                        purpose=f"{presentation}_presentation",
                        prompt_observation={"chars": observation["bytes"], **observation},
                        result_observation={"type": "dict", "chars": observation["bytes"]},
                    ),
                    classification_context=effect_context,
                    classification_result={"completed": True, "presentation": presentation},
                    settle_success=(
                        mark_visible
                        if release_required
                        else lambda: self._mark_presentation_delivered(
                            request,
                            presentation=presentation,
                            view_sha256=view_sha256,
                            provider=provider,
                        )
                    ),
                )
        finally:
            self._data_release_presentation.reset(presentation_token)
            self._data_release_parent_request.reset(parent_token)

    def public_request_view(self, request: HumanRequest) -> dict[str, Any]:
        """Return an observer-safe Human request projection.

        Raw request state remains Host-owned and durable. High-sensitivity
        payloads headed to a conditional Human Sink are replaced with stable,
        metadata-only evidence until the linked exact release is approved.
        """

        return self._project_public_request_view(request, force_withhold=False)

    def _withheld_public_request_view(self, request: HumanRequest) -> dict[str, Any]:
        return self._project_public_request_view(request, force_withhold=True)

    def _project_public_request_view(
        self,
        request: HumanRequest,
        *,
        force_withhold: bool,
    ) -> dict[str, Any]:

        selected = to_jsonable(request)
        if not isinstance(selected, dict):
            raise ValidationError("Human request could not be projected")
        payload = self.public_request_payload(request)
        release_parent_id = request.payload.get(_DATA_RELEASE_FOR_REQUEST_KEY)
        if isinstance(release_parent_id, str) and release_parent_id:
            selected["release_for_request_id"] = release_parent_id
        release_request_id = request.payload.get(_DATA_RELEASE_REQUEST_KEY)
        if isinstance(release_request_id, str) and release_request_id:
            selected["release_request_id"] = release_request_id
        if not force_withhold and not self._withhold_request_payload_from_observers(request):
            selected["payload"] = payload
            return selected

        request_type = payload.get("type")
        selected["payload"] = {
            "type": request_type if isinstance(request_type, str) else "approval",
            "question": "Protected Human request awaiting exact data release.",
            "release_required": True,
            "release_request_id": (
                release_request_id
                if isinstance(release_request_id, str) and release_request_id
                else None
            ),
            "payload_observation": _sanitize_human_observability(
                payload,
                metadata_only=True,
            ),
        }
        if request.decision is not None:
            selected["decision"] = _sanitize_human_observability(request.decision)
        return selected

    def _withhold_request_payload_from_observers(self, request: HumanRequest) -> bool:
        context = self._request_data_flow_context(request)
        if sensitivity_rank(context.labels.sensitivity) <= sensitivity_rank(
            DataSensitivity.NORMAL
        ):
            return False
        manager = self.data_flow
        try:
            trust = manager.resolve_sink_trust(
                DataSink(
                    identity=(
                        f"human:{request.human}:"
                        f"{self.config.runtime.terminal_channel}"
                    )
                )
            )
        except Exception:
            return True
        if trust is not None and trust.trust_level is SinkTrustLevel.TRUSTED:
            return False
        return not (
            trust is not None
            and trust.trust_level is SinkTrustLevel.CONDITIONAL
            and self._has_completed_linked_release(request)
        )

    def _has_completed_linked_release(self, request: HumanRequest) -> bool:
        release_request_id = request.payload.get(_DATA_RELEASE_REQUEST_KEY)
        if not isinstance(release_request_id, str) or not release_request_id:
            return False
        if (
            request.payload.get(_DATA_RELEASE_TERMINAL_COMMITTED_KEY)
            != release_request_id
        ):
            # A reserved one-shot lease is already represented as zero uses.
            # Require an independent settlement marker so observers do not
            # unredact while the provider call is still in flight.
            return False
        release = self.requests.get(release_request_id)
        if release is None:
            return False
        decision = release.decision or {}
        linked_and_approved = bool(
            release.pid == request.pid
            and release.human == request.human
            and release.payload.get("type") == "data_release_approval"
            and release.payload.get(_DATA_RELEASE_FOR_REQUEST_KEY)
            == request.request_id
            and release.status == HumanRequestStatus.APPROVED
            and decision.get("approved") is True
        )
        if not linked_and_approved:
            return False
        once = release.payload.get("requested_once_capability")
        if not isinstance(once, dict):
            return False
        resource = once.get("resource")
        constraints = once.get("constraints")
        if not isinstance(resource, str) or not isinstance(constraints, dict):
            return False
        # Approval grants an exact one-shot capability, but approval alone is
        # not egress. Only unredact after the protected terminal operation has
        # consumed that exact binding and therefore completed the release.
        return any(
            capability.resource == resource
            and capability.constraints == constraints
            and capability.uses_remaining == 0
            for capability in self.authority.list_capabilities(subject=request.pid)
        )

    def _mark_terminal_release_completed(self, request_id: str) -> None:
        """Bind observer visibility to a successfully settled terminal egress."""

        latest = self.requests.get(request_id)
        if latest is None:
            raise NotFound(f"human request not found: {request_id}")
        release_request_id = latest.payload.get(_DATA_RELEASE_REQUEST_KEY)
        if not isinstance(release_request_id, str) or not release_request_id:
            return
        release = self.requests.get(release_request_id)
        if (
            release is None
            or release.status != HumanRequestStatus.APPROVED
            or (release.decision or {}).get("approved") is not True
        ):
            raise CapabilityDenied(
                "linked Human data release is not approved at terminal settlement"
            )
        latest_payload = dict(latest.payload)
        latest_payload[_DATA_RELEASE_TERMINAL_COMMITTED_KEY] = release_request_id
        self.requests.replace_current(
            latest,
            payload=latest_payload,
            updated_at=utc_now(),
        )

    def list(self, pid: str | None = None, *, limit: int | None = None) -> builtins.list[HumanRequest]:
        # Pending decisions are liveness-critical and must never fall behind a
        # bounded history window.  Put every pending request first, followed by
        # the newest historical window for observability.
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValidationError("Human request list limit must be a positive integer")
        pending = self.requests.list(
            pid=pid,
            status=HumanRequestStatus.PENDING,
            limit=limit,
        )
        if limit is not None and len(pending) >= limit:
            return pending
        recent_limit = self.config.tools.human_request_list_limit
        if limit is not None:
            # The recent window can overlap every selected pending row. Fetch
            # only enough extra rows to fill the requested distinct window.
            recent_limit = min(recent_limit, limit + len(pending))
        recent = self.requests.list(
            pid=pid,
            limit=recent_limit,
            newest=True,
        )
        pending_ids = {request.request_id for request in pending}
        combined = [*pending, *(request for request in recent if request.request_id not in pending_ids)]
        return combined if limit is None else combined[:limit]

    def pending(self, human: str | None = None) -> builtins.list[HumanRequest]:
        return self.requests.list(
            human=human,
            status=HumanRequestStatus.PENDING,
            limit=self.config.tools.human_request_list_limit,
        )

    def cancel_pending_for_process(self, pid: str, *, actor: str, reason: str) -> builtins.list[str]:
        """Cancel every pending request owned by a terminal process."""
        cancelled: builtins.list[str] = []
        with self._terminal_lock:
            with self.requests.transaction():
                for request in self.requests.list(
                    pid=pid,
                    status=HumanRequestStatus.PENDING,
                ):
                    decision = {"cancelled_by": actor, "reason": reason}
                    if self._has_terminal_retry_fence_marker(request):
                        # A process-terminal cancellation is authoritative and
                        # needs no later process reconciliation, but it must not
                        # erase evidence that provider completion was unknown.
                        decision = {
                            **dict(request.decision or {}),
                            **decision,
                            "process_reconciliation_required": False,
                        }
                    settled, _release_parent = self._terminalize_pending_request(
                        request,
                        status=HumanRequestStatus.CANCELLED,
                        decision=decision,
                        responder=actor,
                        validate_and_apply_authority=False,
                        transition_process=False,
                        cancel_linked_release=False,
                        event_payload={
                            "request_id": request.request_id,
                            "status": HumanRequestStatus.CANCELLED.value,
                            "reason": reason,
                        },
                        audit_action="human.request_cancelled",
                        audit_decision={"pid": pid, "reason": reason},
                    )
                    cancelled.append(settled.request_id)
        return cancelled

    def _pending_terminal_requests(
        self,
        *,
        human: str,
        pids: frozenset[str] | None,
    ) -> builtins.list[HumanRequest]:
        if pids is None:
            return self.pending(human=human)
        if not pids:
            return []
        # Query each scoped process directly. Filtering a bounded global page
        # after the read could let unrelated processes starve the scheduler's
        # scope even though those rows must never be settled by this drain.
        selected: builtins.list[HumanRequest] = []
        for pid in sorted(pids):
            selected.extend(
                self.requests.list(
                    pid=pid,
                    human=human,
                    status=HumanRequestStatus.PENDING,
                    limit=self.config.tools.human_request_list_limit,
                )
            )
        selected.sort(key=lambda request: (request.created_at, request.request_id))
        return selected[: self.config.tools.human_request_list_limit]

    def process_next_terminal(
        self,
        human: str | None = None,
        auto_approve: bool | None = None,
        auto_policy: str | None = None,
        auto_answer: str | None = None,
        *,
        pids: frozenset[str] | None = None,
    ) -> HumanRequest | None:
        selected_human = human or self.config.runtime.default_human
        with self._terminal_lock:
            pending = self._pending_terminal_requests(
                human=selected_human,
                pids=pids,
            )
            # A durable ambiguous-provider fence blocks only its owning
            # process.  Other processes may keep using the shared Human
            # terminal, while no request from the fenced process can cross a
            # provider boundary before Host reconciliation.  The persisted
            # decision marker makes this hold after Runtime reopen; the
            # in-memory set is the last-resort fence when even that write
            # failed.
            pending = self._without_terminal_retry_fenced_pids(pending)
            if not pending:
                return None
            # The terminal is the human's message queue. Host-created,
            # metadata-only data-release approvals are prerequisites for
            # delivering an older labeled request, so they must run before the
            # request whose provider gate created them. All other requests
            # retain creation order. A second drain may not skip a claimed
            # selected request.
            request = next(
                (
                    item
                    for item in pending
                    if item.payload.get("type") == "data_release_approval"
                ),
                pending[0],
            )
            if self._terminal_request_is_claimed_or_fenced(request):
                return None
            self._terminal_claims.add(request.request_id)
        try:
            try:
                return self._process_claimed_terminal_request(
                    request=request,
                    auto_approve=auto_approve,
                    auto_policy=auto_policy,
                    auto_answer=auto_answer,
                )
            except HumanApprovalRequired as exc:
                # A conditional Human Sink discovers the exact provider text
                # only while formatting the selected request. Process the
                # metadata-only release request immediately, then let the next
                # queue iteration retry the original payload with the exact
                # one-shot release. Never let that implementation detail escape
                # as an endlessly duplicated queue item.
                with self._terminal_lock:
                    release = self.requests.get(exc.request_id)
                    if (
                        release is None
                        or release.human != selected_human
                        or (pids is not None and release.pid not in pids)
                        or release.status != HumanRequestStatus.PENDING
                        or release.payload.get("type") != "data_release_approval"
                    ):
                        raise
                    if self._terminal_request_is_claimed_or_fenced(release):
                        return None
                    self._terminal_claims.add(release.request_id)
                try:
                    return self._process_claimed_terminal_request(
                        request=release,
                        auto_approve=auto_approve,
                        auto_policy=auto_policy,
                        auto_answer=auto_answer,
                    )
                finally:
                    with self._terminal_lock:
                        self._terminal_claims.discard(release.request_id)
        finally:
            with self._terminal_lock:
                self._terminal_claims.discard(request.request_id)

    def _without_terminal_retry_fenced_pids(
        self,
        pending: builtins.list[HumanRequest],
    ) -> builtins.list[HumanRequest]:
        fenced_pids = {
            item.pid
            for item in pending
            if self._has_terminal_retry_fence_marker(item)
            or item.request_id in self._terminal_retry_fences
        }
        if not fenced_pids:
            return pending
        return [item for item in pending if item.pid not in fenced_pids]

    def _terminal_request_is_claimed_or_fenced(self, request: HumanRequest) -> bool:
        return (
            request.request_id in self._terminal_claims
            or request.request_id in self._terminal_retry_fences
            or self._has_terminal_retry_fence_marker(request)
        )

    def _process_claimed_terminal_request(
        self,
        *,
        request: HumanRequest,
        auto_approve: bool | None,
        auto_policy: str | None,
        auto_answer: str | None,
    ) -> HumanRequest:
        # The queue selection snapshot may be stale after a concurrent Host
        # transition.  Re-read immediately before any provider boundary and
        # fail closed on both canonical and malformed durable retry markers.
        with self._terminal_lock:
            latest = self.requests.get(request.request_id)
            if latest is None:
                raise NotFound(f"human request not found: {request.request_id}")
            if latest.status != HumanRequestStatus.PENDING:
                raise ValidationError(
                    "human request is not pending: "
                    f"{latest.request_id} status={latest.status.value}"
                )
            if (
                self._has_terminal_retry_fence_marker(latest)
                or latest.request_id in self._terminal_retry_fences
            ):
                raise ValidationError(
                    "human request has an ambiguous provider outcome and "
                    f"requires Host reconciliation: {latest.request_id}"
                )
            request = latest
        request_type = request.payload.get("type")
        if request_type == "output":
            return self._deliver_output_request(request)
        question = self._terminal_question(request)
        if request_type == "question":
            answer = self._select_text_answer(
                request=request,
                question=question,
                auto_answer=auto_answer,
            )
            return self.approve(
                request.request_id,
                {"approved": True, "answer": answer, "source": "terminal_queue"},
            )
        if request_type == "permission_request":
            policy = self._select_permission_policy(
                request=request,
                question=question,
                auto_policy=auto_policy,
                auto_approve=auto_approve,
            )
            decision = {"policy": policy, "source": "terminal_queue"}
            if policy == CapabilityManager.ALWAYS_DENY:
                return self.reject(request.request_id, {"approved": False, **decision})
            return self.approve(request.request_id, {"approved": True, **decision})

        approved = self._select_boolean_approval(
            request=request,
            question=question,
            auto_approve=auto_approve,
        )
        if approved:
            return self.approve(request.request_id, {"approved": True, "source": "terminal_queue"})
        return self.reject(request.request_id, {"approved": False, "source": "terminal_queue"})

    async def aprocess_next_terminal(
        self,
        human: str | None = None,
        auto_approve: bool | None = None,
        auto_policy: str | None = None,
        auto_answer: str | None = None,
        *,
        pids: frozenset[str] | None = None,
    ) -> HumanRequest | None:
        return await self._blocking_work.run(
            self.process_next_terminal,
            human=human,
            auto_approve=auto_approve,
            auto_policy=auto_policy,
            auto_answer=auto_answer,
            pids=pids,
        )

    def drain_terminal_queue(
        self,
        human: str | None = None,
        auto_approve: bool | None = None,
        auto_policy: str | None = None,
        auto_answer: str | None = None,
        *,
        pids: frozenset[str] | None = None,
    ) -> builtins.list[HumanRequest]:
        processed: builtins.list[HumanRequest] = []
        while True:
            request = self.process_next_terminal(
                human=human,
                auto_approve=auto_approve,
                auto_policy=auto_policy,
                auto_answer=auto_answer,
                pids=pids,
            )
            if request is None:
                return processed
            processed.append(request)

    async def adrain_terminal_queue(
        self,
        human: str | None = None,
        auto_approve: bool | None = None,
        auto_policy: str | None = None,
        auto_answer: str | None = None,
        *,
        pids: frozenset[str] | None = None,
    ) -> builtins.list[HumanRequest]:
        processed: builtins.list[HumanRequest] = []
        while True:
            request = await self.aprocess_next_terminal(
                human=human,
                auto_approve=auto_approve,
                auto_policy=auto_policy,
                auto_answer=auto_answer,
                pids=pids,
            )
            if request is None:
                return processed
            processed.append(request)

    def _decide(
        self,
        request_id: str,
        status: HumanRequestStatus,
        decision: dict[str, Any],
        responder: str,
        *,
        required_presentation: str | None = None,
    ) -> HumanRequest:
        # GUI/CLI/provider decisions are untrusted ingress.  Bound their full
        # structure before permission grants, request transitions, events, or
        # audit serialization can observe them.
        self._validate_human_response(decision, label="human decision")
        selected_decision = dict(decision)
        candidates = self.operations.operation_for_evidence(("human_request",), request_id)
        if len(candidates) == 1:
            with self.operations.attach(candidates[0].operation_id):
                return self._decide_impl(
                    request_id,
                    status,
                    selected_decision,
                    responder,
                    required_presentation=required_presentation,
                )
        return self._decide_impl(
            request_id,
            status,
            selected_decision,
            responder,
            required_presentation=required_presentation,
        )

    def _decide_impl(
        self,
        request_id: str,
        status: HumanRequestStatus,
        decision: dict[str, Any],
        responder: str,
        *,
        required_presentation: str | None = None,
    ) -> HumanRequest:
        with self._terminal_lock:
            with self.requests.transaction():
                request = self.requests.get(request_id)
                if request is None:
                    raise NotFound(f"human request not found: {request_id}")
                if (
                    self._has_terminal_retry_fence_marker(request)
                    or request.request_id in self._terminal_retry_fences
                ):
                    raise ValidationError(
                        "human request has an ambiguous provider outcome and "
                        f"requires Host reconciliation: {request_id}"
                    )
                if request.status != HumanRequestStatus.PENDING:
                    raise ValidationError(f"human request is not pending: {request_id} status={request.status.value}")
                process = self.processes.get_process(request.pid)
                if process is not None and process.status in self.TERMINAL_PROCESS_STATUSES:
                    raise ValidationError(
                        f"terminal process cannot receive a human decision: {request.pid} status={process.status.value}"
                    )
                if required_presentation is not None and self.is_request_withheld_for_presentation(
                    request,
                    presentation=required_presentation,
                ):
                    release_id = self._linked_presentation_release_id(
                        request,
                        str(required_presentation).strip().lower(),
                    )
                    raise HumanApprovalRequired(
                        release_id or request.request_id,
                        "human request payload has not been released for "
                        f"{required_presentation} presentation",
                    )
                request, _release_parent = self._terminalize_pending_request(
                    request,
                    status=status,
                    decision=decision,
                    responder=responder,
                    validate_and_apply_authority=True,
                    transition_process=True,
                    cancel_linked_release=status != HumanRequestStatus.APPROVED,
                )
        return request

    def _terminalize_pending_request(
        self,
        request: HumanRequest,
        *,
        status: HumanRequestStatus,
        decision: dict[str, Any],
        responder: str,
        validate_and_apply_authority: bool,
        transition_process: bool,
        cancel_linked_release: bool,
        provider_outcome_unknown: bool = False,
        event_payload: dict[str, Any] | None = None,
        event_source: str | None = None,
        audit_action: str = "human.response",
        audit_decision: dict[str, Any] | None = None,
    ) -> tuple[HumanRequest, HumanRequest | None]:
        """Commit one terminal Human transition inside the caller's unit.

        Every caller owns an enclosing Store transaction.  The exact request
        revision/status CAS, authority effects, linked-request handling,
        process wait-set transition, event, and audit therefore succeed or
        roll back together.  Building a distinct target is intentional: a
        losing CAS can never leak an in-memory mutation into retry logic.
        """

        if request.status != HumanRequestStatus.PENDING:
            raise ValidationError(
                "human request is not pending: "
                f"{request.request_id} status={request.status.value}"
            )
        if validate_and_apply_authority:
            self._validate_decision_side_effects(request, status, decision)
            self._apply_decision_side_effects(request, status, decision, responder)

        # Capability/permission side effects may advance the process revision.
        # Re-read after them and never write the pre-side-effect snapshot back.
        process = self.processes.get_process(request.pid)
        permission_related = any(
            isinstance(request.payload.get(key), dict)
            for key in ("requested_permission", "requested_once_capability")
        )
        settled = self.requests.replace_current(
            request,
            status=status,
            decision=dict(decision),
            updated_at=utc_now(),
        )
        release_parent = (
            self._cancel_linked_request_for_release(
                settled,
                outcome=(
                    "provider_outcome_unknown"
                    if provider_outcome_unknown
                    else status.value
                ),
                actor=responder,
            )
            if cancel_linked_release
            else None
        )
        if transition_process:
            self._transition_after_human_decision(
                process,
                settled,
                status,
                permission_related=permission_related,
                release_parent=release_parent,
                provider_outcome_unknown=provider_outcome_unknown,
            )

        response_evidence = (
            dict(event_payload)
            if event_payload is not None
            else {
                "request_id": settled.request_id,
                "status": status.value,
                "decision": _sanitize_human_observability(decision),
            }
        )
        if release_parent is not None:
            response_evidence.update(
                {
                    "linked_request_id": release_parent.request_id,
                    "linked_request_status": release_parent.status.value,
                }
            )
        self.events.emit(
            EventType.HUMAN_RESPONSE,
            source=event_source or responder,
            target=settled.pid,
            payload=response_evidence,
        )
        self.audit.record(
            actor=responder,
            action=audit_action,
            target=f"human_request:{settled.request_id}",
            decision=(
                dict(audit_decision)
                if audit_decision is not None
                else response_evidence
            ),
        )
        return settled, release_parent

    def _transition_after_human_decision(
        self,
        process: Any | None,
        request: HumanRequest,
        status: HumanRequestStatus,
        *,
        permission_related: bool,
        release_parent: HumanRequest | None,
        provider_outcome_unknown: bool = False,
    ) -> None:
        if process is None or process.status != ProcessStatus.WAITING_HUMAN:
            return
        if provider_outcome_unknown:
            # A Human provider phase may have completed even though the caller
            # observed an exception. Preserve that uncertainty as a Host-only
            # resume gate; a model parent must not convert manual recovery into
            # an ordinary resume. A process able to enter WAITING_HUMAN already
            # has a durable goal Object, and checkpoint restore remaps that
            # reference, so it is a stable reason OID without creating another
            # Object inside the Human terminal transaction.
            reason_oid = str(process.goal_oid or "").strip()
            if not reason_oid:
                raise ProcessError(
                    "ambiguous Human provider outcome cannot be gated without "
                    f"a durable process goal Object: {process.pid}"
                )
            selected_status = ProcessStatus.PAUSED
            wait_state = HostResumeProcessWait(reason_oid=reason_oid)
            if release_parent is not None:
                status_message = (
                    "data release provider outcome unknown for Human request "
                    f"{release_parent.request_id}; manual recovery required"
                )
            else:
                status_message = (
                    f"human provider outcome unknown for request {request.request_id}; "
                    "manual recovery required"
                )
        else:
            remaining = [
                pending
                for pending in self.requests.list(
                    pid=request.pid,
                    status=HumanRequestStatus.PENDING,
                )
                if pending.blocking
            ]
            if remaining:
                selected_status = ProcessStatus.WAITING_HUMAN
                wait_state = HumanProcessWait(
                    request_ids=tuple(pending.request_id for pending in remaining)
                )
                status_message = None
            elif release_parent is not None:
                selected_status = ProcessStatus.PAUSED
                wait_state = PausedProcessWait()
                status_message = (
                    f"data release {status.value} for Human request "
                    f"{release_parent.request_id}"
                )
            else:
                # Permission denials wake the process so it can observe the
                # structured failed operation. Generic rejections remain paused.
                selected_status = (
                    ProcessStatus.RUNNABLE
                    if status == HumanRequestStatus.APPROVED or permission_related
                    else ProcessStatus.PAUSED
                )
                wait_state = (
                    PausedProcessWait()
                    if selected_status == ProcessStatus.PAUSED
                    else None
                )
                status_message = (
                    None
                    if status == HumanRequestStatus.APPROVED
                    else f"human rejected {request.request_id}"
                )
        self._transitions.transition(
            process.pid,
            selected_status,
            expected_revision=process.revision,
            expected_status=ProcessStatus.WAITING_HUMAN,
            expected_state_generation=process.state_generation,
            wait_state=wait_state,
            status_message=status_message,
        )

    def _cancel_linked_request_for_release(
        self,
        release: HumanRequest,
        *,
        outcome: str,
        actor: str,
    ) -> HumanRequest | None:
        """Fail closed when an exact Human-Sink release is not approved.

        The caller owns the surrounding store transaction. The internal link
        is persisted with the release request, so this also works after a
        runtime reopen and never needs to reconstruct or inspect the sensitive
        provider payload.
        """

        if release.payload.get("type") != "data_release_approval":
            return None
        parent_id = release.payload.get(_DATA_RELEASE_FOR_REQUEST_KEY)
        if not isinstance(parent_id, str) or not parent_id:
            return None
        parent = self.requests.get(parent_id)
        if parent is None or parent.status != HumanRequestStatus.PENDING:
            return None
        return self.requests.replace_current(
            parent,
            status=HumanRequestStatus.CANCELLED,
            decision={
                "data_release_outcome": outcome,
                "data_release_request_id": release.request_id,
                "terminated_by": actor,
                "automatic_retry_disabled": True,
                "sensitive_payload_delivered": False,
            },
            updated_at=utc_now(),
        )

    def _validate_decision_side_effects(
        self,
        request: HumanRequest,
        status: HumanRequestStatus,
        decision: dict[str, Any],
    ) -> None:
        approved = decision.get("approved")
        expected_approved = status == HumanRequestStatus.APPROVED
        if not isinstance(approved, bool):
            raise ValidationError("human decision approved must be a JSON boolean")
        if approved is not expected_approved:
            raise ValidationError(
                f"human decision approved={approved} conflicts with status={status.value}"
            )
        request_type = request.payload.get("type")
        if request_type == "question" and status == HumanRequestStatus.APPROVED:
            if "answer" not in decision:
                raise ValidationError("approved human question requires an answer")
            if not isinstance(decision["answer"], str):
                raise ValidationError("human question answer must be a string")
            if not decision["answer"].strip():
                raise ValidationError("human question answer must be non-empty")
        permission_spec = request.payload.get("requested_permission")
        if isinstance(permission_spec, dict):
            self._permission_decision_spec(permission_spec, request.pid, status, decision)
        once_spec = request.payload.get("requested_once_capability")
        if isinstance(once_spec, dict) and status == HumanRequestStatus.APPROVED:
            _subject, _resource, _rights, _constraints, expires_at, _delegable = (
                self._capability_request_spec(
                    once_spec,
                    request.pid,
                    label="requested one-time capability",
                )
            )
            self.authority_policy.bound_capability_expiry(request.pid, expires_at)
        cap_spec = request.payload.get("requested_capability")
        if isinstance(cap_spec, dict) and status == HumanRequestStatus.APPROVED:
            _subject, _resource, _rights, _constraints, expires_at, _delegable = (
                self._capability_request_spec(
                    cap_spec,
                    request.pid,
                    label="requested capability",
                )
            )
            self.authority_policy.bound_capability_expiry(request.pid, expires_at)

    def _apply_decision_side_effects(
        self,
        request: HumanRequest,
        status: HumanRequestStatus,
        decision: dict[str, Any],
        responder: str,
    ) -> None:
        permission_spec = request.payload.get("requested_permission")
        if isinstance(permission_spec, dict):
            subject, resource, rights, constraints, policy, expires_at = (
                self._permission_decision_spec(
                    permission_spec,
                    request.pid,
                    status,
                    decision,
                )
            )
            self.capabilities.set_permission_policy(
                subject=subject,
                resource=resource,
                rights=rights,
                policy=policy,
                issued_by=responder,
                constraints=constraints,
                expires_at=expires_at,
            )

        once_spec = request.payload.get("requested_once_capability")
        if isinstance(once_spec, dict) and status == HumanRequestStatus.APPROVED:
            subject, resource, rights, constraints, requested_expiry, _delegable = (
                self._capability_request_spec(
                    once_spec,
                    request.pid,
                    label="requested one-time capability",
                )
            )
            expires_at = self.authority_policy.bound_capability_expiry(
                request.pid,
                requested_expiry,
            )
            self.capabilities.grant_once(
                subject=subject,
                resource=resource,
                rights=rights,
                issued_by=responder,
                constraints=constraints,
                expires_at=expires_at,
            )

        cap_spec = request.payload.get("requested_capability")
        if isinstance(cap_spec, dict) and status == HumanRequestStatus.APPROVED:
            subject, resource, rights, constraints, expires_at, delegable = (
                self._capability_request_spec(
                    cap_spec,
                    request.pid,
                    label="requested capability",
                )
            )
            expires_at = self.authority_policy.bound_capability_expiry(
                request.pid,
                expires_at,
            )
            self.capabilities.grant(
                subject=subject,
                resource=resource,
                rights=rights,
                issued_by=responder,
                constraints=constraints,
                expires_at=expires_at,
                delegable=delegable,
            )

    def _permission_decision_spec(
        self,
        spec: dict[str, Any],
        default_subject: str,
        status: HumanRequestStatus,
        decision: dict[str, Any],
    ) -> tuple[str, str, list[str], dict[str, Any] | None, str, str | None]:
        subject, resource, rights, constraints, requested_expiry, _delegable = (
            self._capability_request_spec(
                spec,
                default_subject,
                label="requested permission",
            )
        )
        policy_value = decision.get("policy")
        if not isinstance(policy_value, str):
            raise ValidationError("permission decisions require an explicit policy")
        policy = policy_value
        if policy not in {
            CapabilityManager.ALWAYS_ALLOW,
            CapabilityManager.ALWAYS_DENY,
            CapabilityManager.ASK_EACH_TIME,
        }:
            raise ValidationError(f"unknown permission policy: {policy}")
        if status == HumanRequestStatus.REJECTED and policy == CapabilityManager.ALWAYS_ALLOW:
            raise ValidationError("rejected permission requests cannot install always_allow policy")
        if status == HumanRequestStatus.APPROVED and policy == CapabilityManager.ALWAYS_DENY:
            raise ValidationError("approved permission requests cannot install always_deny policy")
        expires_at = (
            self.authority_policy.bound_capability_expiry(
                default_subject,
                requested_expiry,
            )
            if policy == CapabilityManager.ASK_EACH_TIME
            or (
                status == HumanRequestStatus.APPROVED
                and policy == CapabilityManager.ALWAYS_ALLOW
            )
            else None
        )
        return subject, resource, rights, constraints, policy, expires_at

    def _capability_request_spec(
        self,
        spec: dict[str, Any],
        default_subject: str,
        *,
        label: str,
    ) -> tuple[str, str, list[str], dict[str, Any] | None, str | None, bool]:
        resource = spec.get("resource")
        if not isinstance(resource, str):
            raise ValidationError(f"{label} must include a string resource")
        subject = spec.get("subject", default_subject)
        if not isinstance(subject, str):
            raise ValidationError(f"{label} subject must be a string")
        if subject != default_subject:
            raise ValidationError(
                f"{label} subject must match request process: {default_subject}"
            )
        rights = spec.get("rights", [CapabilityRight.EXECUTE.value])
        if not isinstance(rights, list) or not rights:
            raise ValidationError(f"{label} rights must be a non-empty list")
        try:
            normalized_rights = [
                CapabilityRight(str(right)).value
                for right in rights
            ]
        except ValueError as exc:
            raise ValidationError(
                f"{label} contains an unknown capability right"
            ) from exc
        constraints = spec.get("constraints")
        if constraints is not None and not isinstance(constraints, dict):
            raise ValidationError(f"{label} constraints must be an object")
        delegable = spec.get("delegable", False)
        if not isinstance(delegable, bool):
            raise ValidationError(f"{label} delegable must be a JSON boolean")
        raw_expires_at = spec.get("expires_at")
        if raw_expires_at is not None and not isinstance(raw_expires_at, str):
            raise ValidationError(f"{label} expires_at must be an ISO-8601 datetime string")
        expires_at = raw_expires_at
        return (
            subject,
            resource,
            normalized_rights,
            dict(constraints) if isinstance(constraints, dict) else None,
            expires_at if isinstance(expires_at, str) else None,
            delegable,
        )

    def _reserve_one_time_decision(self, decision: Any, *, used_by: str) -> str | None:
        return self.capabilities.reserve_decision_use(
            decision,
            used_by=used_by,
            reason="one-time human permission reserved",
        )

    def _commit_one_time_decision(self, reservation_id: str | None) -> bool:
        return self.capabilities.commit_reserved_use(
            reservation_id,
            committed_by="human",
            reason="one-time human permission committed",
        )

    def _require_one_time_decision_commit(self, reservation_id: str | None) -> None:
        if reservation_id is None:
            return
        if not self._commit_one_time_decision(reservation_id):
            raise CapabilityDenied(
                "one-time human permission reservation could not be committed"
            )

    def _restore_one_time_decision(self, reservation_id: str | None) -> None:
        self.capabilities.restore_reserved_use(
            reservation_id,
            restored_by="human",
            reason="one-time human permission restored before request commit",
        )

    def _validate_human_response(self, value: Any, *, label: str) -> int:
        return _ensure_bounded_json_value(
            value,
            limit_bytes=self.config.tools.human_response_payload_max_bytes,
            max_depth=self.config.tools.human_response_max_depth,
            max_nodes=self.config.tools.human_response_max_nodes,
            label=label,
        )

    def _select_permission_policy(
        self,
        request: HumanRequest,
        question: str,
        auto_policy: str | None,
        auto_approve: bool | None,
    ) -> str:
        choices = {
            CapabilityManager.ALWAYS_ALLOW,
            CapabilityManager.ALWAYS_DENY,
            CapabilityManager.ASK_EACH_TIME,
        }
        if auto_policy is not None:
            if auto_policy not in choices:
                raise ValueError(f"unknown permission policy: {auto_policy}")
            self._terminal_provider_io(
                request,
                operation="write",
                text=f"{question} [policy={auto_policy}]",
                purpose="permission_policy_auto",
            )
            return auto_policy
        if auto_approve is not None:
            policy = CapabilityManager.ALWAYS_ALLOW if auto_approve else CapabilityManager.ALWAYS_DENY
            self._terminal_provider_io(
                request,
                operation="write",
                text=f"{question} [policy={policy}]",
                purpose="permission_policy_auto",
            )
            return policy
        answer = str(
            self._terminal_provider_io(
                request,
                operation="read",
                text=f"{question} [a=always allow, d=always deny, e=ask each time; default=d]: ",
                purpose="permission_policy",
            )
        ).strip().lower()
        return {
            "a": CapabilityManager.ALWAYS_ALLOW,
            "allow": CapabilityManager.ALWAYS_ALLOW,
            "always_allow": CapabilityManager.ALWAYS_ALLOW,
            "y": CapabilityManager.ALWAYS_ALLOW,
            "yes": CapabilityManager.ALWAYS_ALLOW,
            "d": CapabilityManager.ALWAYS_DENY,
            "deny": CapabilityManager.ALWAYS_DENY,
            "always_deny": CapabilityManager.ALWAYS_DENY,
            "n": CapabilityManager.ALWAYS_DENY,
            "no": CapabilityManager.ALWAYS_DENY,
            "e": CapabilityManager.ASK_EACH_TIME,
            "each": CapabilityManager.ASK_EACH_TIME,
            "ask": CapabilityManager.ASK_EACH_TIME,
            "ask_each_time": CapabilityManager.ASK_EACH_TIME,
        }.get(answer, CapabilityManager.ALWAYS_DENY)

    def format_terminal_request(self, request: HumanRequest) -> str:
        return self._terminal_question(request)

    def present_terminal_request(
        self,
        request: HumanRequest,
        *,
        suffix: str,
    ) -> None:
        """Present an interactive request through the protected Human Sink."""

        with self._terminal_lock:
            latest = self.requests.get(request.request_id)
            if latest is None:
                raise NotFound(f"human request not found: {request.request_id}")
            if latest.status != HumanRequestStatus.PENDING:
                raise ValidationError(
                    "human request is not pending: "
                    f"{latest.request_id} status={latest.status.value}"
                )
            if (
                self._has_terminal_retry_fence_marker(latest)
                or latest.request_id in self._terminal_retry_fences
            ):
                raise ValidationError(
                    "human request has an ambiguous provider outcome and "
                    f"requires Host reconciliation: {latest.request_id}"
                )
            if latest.request_id in self._terminal_claims:
                raise ValidationError(
                    f"human request is already being presented: {latest.request_id}"
                )
            self._terminal_claims.add(latest.request_id)
        try:
            question = self._terminal_question(latest)
            text = f"\nHuman request {latest.request_id}: {question}\n{suffix}"
            self._terminal_provider_io(
                latest,
                operation="write",
                text=text,
                purpose="interactive_cli_presentation",
            )
        finally:
            with self._terminal_lock:
                self._terminal_claims.discard(latest.request_id)

    def _terminal_question(self, request: HumanRequest) -> str:
        raw_question = request.payload.get("question")
        question = (
            str(raw_question)
            if raw_question
            else dumps(to_jsonable(self.public_request_payload(request)))
        )
        if request.payload.get("type") == "permission_request":
            return self._permission_terminal_question(request, question)
        if request.payload.get("type") == "question":
            context = request.payload.get("context")
            if not isinstance(context, dict) or not context:
                return question
            lines = [question, "Context:"]
            for key in sorted(context):
                lines.append(f"- {key}: {context[key]!r}")
            return "\n".join(lines)
        if request.payload.get("type") == "data_release_approval":
            context = request.payload.get("context")
            if not isinstance(context, dict):
                return question
            lines = ["Data release details:"]
            for label, key in [
                ("sink", "sink"),
                ("sink identity sha256", "sink_identity_sha256"),
                ("sensitivity", "sensitivity"),
                ("tenant", "tenant"),
                ("principal", "principal"),
                ("payload bytes", "payload_bytes"),
                ("payload sha256", "payload_sha256"),
                ("labels sha256", "labels_sha256"),
                ("source refs sha256", "source_refs_sha256"),
                ("source count", "source_count"),
                ("trust id", "trust_id"),
                ("trust sha256", "trust_sha256"),
                ("registry generation", "registry_generation"),
                ("manifest sha256", "manifest_sha256"),
                ("operation", "operation"),
            ]:
                if context.get(key) is not None:
                    lines.append(f"- {label}: {context[key]}")
            lines.append(question)
            return "\n".join(lines)
        if request.payload.get("type") != "external_operation_approval":
            return question
        context = request.payload.get("context")
        if not isinstance(context, dict):
            return question
        # External-operation prompts show structured facts, not tool prose, so
        # the human can judge the primitive-level side effect safely.
        capability = request.payload.get("requested_once_capability")
        lines = ["Operation details:"]
        for label, key in [
            ("process", "pid"),
            ("primitive", "primitive"),
            ("operation", "operation"),
            ("path", "path"),
            ("absolute path", "absolute_path"),
            ("resource", "resource"),
            ("grant scope", "grant_scope"),
            ("encoding", "encoding"),
            ("overwrite flag", "overwrite"),
            ("parents flag", "parents"),
            ("exist ok", "exist_ok"),
            ("recursive", "recursive"),
            ("missing ok", "missing_ok"),
            ("will create", "will_create"),
            ("will overwrite", "will_overwrite"),
            ("content bytes", "content_bytes"),
            ("content sha256", "content_sha256"),
            ("working directory", "working_directory"),
            ("argv", "argv"),
            ("command", "command"),
            ("timeout seconds", "timeout_s"),
            ("policy level", "policy_level"),
            ("policy reason", "policy_reason"),
            ("matched rule", "matched_rule"),
            ("high risk", "high_risk"),
            ("risk", "risk"),
            ("rule id", "rule_id"),
            ("rule effect", "rule_effect"),
        ]:
            if key in context:
                lines.append(f"- {label}: {context[key]}")
        profile = context.get("sandbox_profile")
        if isinstance(profile, dict):
            lines.append("- sandbox profile:")
            for key in ["operation", "resource", "effect", "risk", "rule_id"]:
                if key in profile:
                    lines.append(f"  - {key}: {profile[key]}")
        target = context.get("target")
        if isinstance(target, dict):
            lines.append("- target:")
            for key in ["exists", "kind", "size_bytes", "modified_at"]:
                if key in target:
                    lines.append(f"  - {key}: {target[key]}")
        if isinstance(capability, dict):
            lines.append("- one-time capability:")
            lines.append(f"  - resource: {capability.get('resource')}")
            lines.append(f"  - rights: {capability.get('rights')}")
        preview = context.get("content_preview")
        if isinstance(preview, str):
            truncated = bool(context.get("content_preview_truncated"))
            lines.append(f"- content preview{' (truncated)' if truncated else ''}:")
            lines.append(self._indent_block(preview))
        lines.append(question)
        return "\n".join(lines)

    def _permission_terminal_question(self, request: HumanRequest, question: str) -> str:
        context = request.payload.get("context")
        permission = request.payload.get("requested_permission")
        if not isinstance(context, dict):
            return question
        lines = ["Permission request details:"]
        for label, key in [
            ("process", "pid"),
            ("reason", "reason"),
            ("risk", "risk"),
            ("requested resource", "resource"),
            ("canonical resource", "canonical_resource"),
            ("resource kind", "resource_kind"),
            ("resource scope", "resource_scope"),
            ("resource body", "resource_body"),
            ("rights", "rights"),
            ("origin", "request_origin"),
        ]:
            value = request.pid if key == "pid" else context.get(key)
            if value is not None:
                lines.append(f"- {label}: {value}")
        lease = context.get("lease")
        if isinstance(lease, dict):
            lines.append("- lease:")
            for key in ["type", "choices", "default_if_unanswered", "expires_at", "uses_remaining"]:
                if key in lease:
                    lines.append(f"  - {key}: {lease[key]}")
        constraints = context.get("constraints")
        if isinstance(constraints, dict):
            lines.append("- constraints:")
            rules = constraints.get(AUTHORITY_RULES_KEY)
            if isinstance(rules, list) and rules:
                lines.append("  - authority_rules:")
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    lines.append(
                        "    - "
                        f"{rule.get('rule_id')} "
                        f"effect={rule.get('effect')} "
                        f"risk={rule.get('risk')} "
                        f"conditions={rule.get('conditions')}"
                    )
            elif constraints:
                for key in sorted(constraints):
                    lines.append(f"  - {key}: {constraints[key]}")
            else:
                lines.append("  - <none>")
        if isinstance(permission, dict):
            lines.append("- requested policy target:")
            lines.append(f"  - resource: {permission.get('resource')}")
            lines.append(f"  - rights: {permission.get('rights')}")
        lines.append(question)
        return "\n".join(lines)

    def _indent_block(self, text: str) -> str:
        if not text:
            return "  <empty>"
        return "\n".join(f"  {line}" for line in text.splitlines() or [text])

    def _terminal_provider_io(
        self,
        request: HumanRequest,
        *,
        operation: str,
        text: str,
        purpose: str,
    ) -> str | None:
        candidates = self.operations.operation_for_evidence(
            ("human_request",),
            request.request_id,
        )
        if len(candidates) == 1:
            with self.operations.attach(candidates[0].operation_id):
                return self._terminal_provider_io_impl(
                    request,
                    operation=operation,
                    text=text,
                    purpose=purpose,
                )
        return self._terminal_provider_io_impl(
            request,
            operation=operation,
            text=text,
            purpose=purpose,
        )

    def _terminal_provider_io_impl(
        self,
        request: HumanRequest,
        *,
        operation: str,
        text: str,
        purpose: str,
    ) -> str | None:
        if operation not in {"read", "write"}:
            raise ValidationError(f"unsupported terminal human provider operation: {operation}")
        resource = f"human:{request.human}"
        channel = self.config.runtime.terminal_channel
        flow = self._request_data_flow_context(request)
        sink = DataSink(identity=f"human:{request.human}:{channel}")
        prompt_observation = self._terminal_text_observation(text)
        request_kind = (
            "approval"
            if purpose.startswith(("permission_policy", "boolean_approval"))
            else "question"
            if purpose.startswith("text_answer")
            else purpose
        )
        effect_context = {
            "request_id": request.request_id,
            "request_kind": request_kind,
            "purpose": purpose,
            "operation": operation,
            "chars": prompt_observation["chars"],
            "prompt_observation": prompt_observation,
        }
        invocation = ProtectedOperationInvocation(
            pid=request.pid,
            actor=request.pid,
            target=resource,
            canonical_args={
                "request_id": request.request_id,
                "operation": operation,
                "purpose": purpose,
                "text": text,
            },
            observation=effect_context,
            data_sink=sink,
            data_flow_context=flow,
            data_flow_ingress_context=(
                self._human_response_context(flow)
                if operation == "read"
                else None
            ),
            data_flow_payload=text,
            data_flow_operation=f"human.{operation}",
            data_flow_allow_recovered_source_snapshots=True,
            failure_evidence=lambda error, phase: self._protected_terminal_evidence(
                request,
                operation=operation,
                resource=resource,
                purpose=purpose,
                prompt_observation=prompt_observation,
                result_observation={"type": type(error).__name__},
                failed=True,
                phase=phase,
            ),
        )
        provider_attempted = False

        def provider_call() -> str | None:
            nonlocal provider_attempted
            provider_attempted = True
            if operation == "read":
                return self.delivery.read(text)
            self.delivery.write(text)
            return None

        release_parent_token = self._data_release_parent_request.set(request.request_id)
        completed_result: str | None
        response_validation_error: ValidationError | None = None
        try:
            with self._protected().start(
                f"primitive.human.{operation}", invocation, provider=self.provider
            ) as protected:
                result = protected.call(
                    ProviderPhase(
                        "terminal_io",
                        state_mutation=operation == "write",
                        information_flow=True,
                    ),
                    provider_call,
                )
                if operation == "read":
                    try:
                        self._validate_human_response(
                            result,
                            label="human provider response",
                        )
                    except ValidationError as exc:
                        # The provider boundary completed, so settle it as a
                        # known read while keeping the oversized value out of
                        # hashes/evidence and all request/authority mutations.
                        response_validation_error = exc
                        result_observation = {
                            "type": type(result).__name__,
                            "rejected": "response_bounds",
                        }
                    else:
                        result_observation = self._terminal_text_observation(result)
                else:
                    result_observation = (
                        self._terminal_text_observation(result)
                        if isinstance(result, str)
                        else {"type": type(result).__name__}
                    )
                completed_result = protected.complete(
                    result,
                    self._protected_terminal_evidence(
                        request,
                        operation=operation,
                        resource=resource,
                        purpose=purpose,
                        prompt_observation=prompt_observation,
                        result_observation=result_observation,
                    ),
                    classification_context=effect_context,
                    classification_result={
                        "completed": True,
                        "result_observation": result_observation,
                    },
                    settle_success=lambda: self._mark_terminal_release_completed(
                        request.request_id
                    ),
                )
        except ProviderEffectNotStarted:
            # A provider-certified pre-boundary failure is safe to retry and
            # therefore intentionally keeps the request pending.
            raise
        except BaseException as error:
            if provider_attempted:
                self._mark_terminal_provider_outcome_unknown(
                    request,
                    operation=operation,
                    purpose=purpose,
                    error=error,
                )
            raise
        finally:
            self._data_release_parent_request.reset(release_parent_token)
        if response_validation_error is not None:
            raise response_validation_error
        return completed_result

    def _mark_terminal_provider_outcome_unknown(
        self,
        request: HumanRequest,
        *,
        operation: str,
        purpose: str,
        error: BaseException,
    ) -> None:
        """Fence redispatch even if the composite outcome transition fails."""

        try:
            self._mark_terminal_provider_outcome_unknown_once(
                request,
                operation=operation,
                purpose=purpose,
                error=error,
            )
            return
        except Exception:
            # The primary transaction also reconciles linked requests and the
            # process wait state. If any of those writes fails, retry the
            # smallest durable safety boundary independently: retain this
            # exact request as PENDING but poison every automatic/provider
            # response path. This deliberately does not retry provider I/O.
            if self._persist_terminal_retry_fence(
                request.request_id,
                operation=operation,
                purpose=purpose,
                error=error,
            ):
                return
        with self._terminal_lock:
            self._terminal_retry_fences.add(request.request_id)

    def _persist_terminal_retry_fence(
        self,
        request_id: str,
        *,
        operation: str,
        purpose: str,
        error: BaseException,
    ) -> bool:
        try:
            with self.requests.transaction():
                latest = self.requests.get(request_id)
                if latest is None or latest.status != HumanRequestStatus.PENDING:
                    return True
                self.requests.replace_current(
                    latest,
                    decision={
                        "provider_outcome": "unknown",
                        "automatic_retry_disabled": True,
                        "manual_recovery_required": True,
                        "process_reconciliation_required": True,
                        "operation": operation,
                        "purpose": purpose,
                        "error_type": type(error).__name__,
                    },
                    updated_at=utc_now(),
                )
            return True
        except Exception:
            return False

    def reconcile_terminal_retry_fence(self, request_id: str) -> HumanRequest:
        """Host recovery for one durable ambiguous-provider retry fence.

        Provider I/O is never retried here.  The request CAS, linked request,
        process Host-resume gate, response event, and audit evidence all use
        the ordinary Human terminal kernel and therefore commit or roll back
        together.  A failed attempt leaves the PENDING fence intact.
        """

        with self._terminal_lock:
            with self.requests.transaction():
                latest = self.requests.get(request_id)
                if latest is None:
                    raise NotFound(f"human request not found: {request_id}")
                if (
                    latest.status == HumanRequestStatus.CANCELLED
                    and (latest.decision or {}).get("provider_outcome") == "unknown"
                ):
                    return latest
                if not self._is_terminal_retry_fence(latest):
                    raise ValidationError(
                        "human request is not a canonical ambiguous-provider retry fence: "
                        f"{request_id} status={latest.status.value}"
                    )
                decision = {
                    **dict(latest.decision or {}),
                    "process_reconciliation_required": False,
                }
                event_payload = {
                    "request_id": latest.request_id,
                    "status": HumanRequestStatus.CANCELLED.value,
                    "provider_outcome": "unknown",
                    "automatic_retry_disabled": True,
                    "operation": decision.get("operation"),
                    "purpose": decision.get("purpose"),
                    "error_type": decision.get("error_type"),
                }
                settled, _release_parent = self._terminalize_pending_request(
                    latest,
                    status=HumanRequestStatus.CANCELLED,
                    decision=decision,
                    responder="runtime:human-provider",
                    validate_and_apply_authority=False,
                    transition_process=True,
                    cancel_linked_release=True,
                    provider_outcome_unknown=True,
                    event_payload=event_payload,
                    event_source=f"human:{latest.human}",
                    audit_action="human.request.provider_outcome_unknown",
                )
        return settled

    @staticmethod
    def _is_terminal_retry_fence(request: HumanRequest) -> bool:
        decision = request.decision
        return (
            request.status == HumanRequestStatus.PENDING
            and isinstance(decision, dict)
            and decision.get("provider_outcome") == "unknown"
            and decision.get("automatic_retry_disabled") is True
            and decision.get("manual_recovery_required") is True
            and decision.get("process_reconciliation_required") is True
            and isinstance(decision.get("operation"), str)
            and bool(str(decision.get("operation")).strip())
            and isinstance(decision.get("purpose"), str)
            and bool(str(decision.get("purpose")).strip())
            and isinstance(decision.get("error_type"), str)
            and bool(str(decision.get("error_type")).strip())
        )

    @staticmethod
    def _has_terminal_retry_fence_marker(request: HumanRequest) -> bool:
        """Detect any reserved retry-fence state and fail closed if malformed."""

        decision = request.decision
        return (
            request.status == HumanRequestStatus.PENDING
            and isinstance(decision, Mapping)
            and bool(_TERMINAL_RETRY_FENCE_KEYS.intersection(decision))
        )

    def _mark_terminal_provider_outcome_unknown_once(
        self,
        request: HumanRequest,
        *,
        operation: str,
        purpose: str,
        error: BaseException,
    ) -> None:
        """Persist a non-retryable terminal request after ambiguous provider I/O."""

        latest: HumanRequest | None = None
        with self._terminal_lock:
            with self.requests.transaction():
                latest = self.requests.get(request.request_id)
                if latest is None or latest.status != HumanRequestStatus.PENDING:
                    return
                decision = {
                    "provider_outcome": "unknown",
                    "automatic_retry_disabled": True,
                    "manual_recovery_required": True,
                    "operation": operation,
                    "purpose": purpose,
                    "error_type": type(error).__name__,
                }
                latest, _release_parent = self._terminalize_pending_request(
                    latest,
                    status=HumanRequestStatus.CANCELLED,
                    decision=decision,
                    responder="runtime:human-provider",
                    validate_and_apply_authority=False,
                    transition_process=True,
                    cancel_linked_release=True,
                    provider_outcome_unknown=True,
                    event_payload={
                        "request_id": latest.request_id,
                        "status": HumanRequestStatus.CANCELLED.value,
                        "provider_outcome": "unknown",
                        "automatic_retry_disabled": True,
                        "operation": operation,
                        "purpose": purpose,
                        "error_type": type(error).__name__,
                    },
                    event_source=f"human:{latest.human}",
                    audit_action="human.request.provider_outcome_unknown",
                )

    def _protected(self) -> Any:
        return self.protected_operations

    def _protected_terminal_evidence(
        self,
        request: HumanRequest,
        *,
        operation: str,
        resource: str,
        purpose: str,
        prompt_observation: dict[str, Any],
        result_observation: dict[str, Any],
        failed: bool = False,
        phase: str | None = None,
    ) -> ProtectedOperationEvidence:
        result_chars = result_observation.get("chars", prompt_observation["chars"])
        event_payload = {
            "request_id": request.request_id,
            "purpose": purpose,
            "operation": operation,
            "chars": result_chars,
        }
        decision = {
            "request_id": request.request_id,
            "purpose": purpose,
            "operation": operation,
            "prompt_observation": prompt_observation,
            "result_observation": result_observation,
        }
        if failed:
            event_payload.update({"outcome": "unknown", "phase": phase})
            decision.update({"effect_outcome": "unknown", "phase": phase})
        return ProtectedOperationEvidence(
            event_type=(
                EventType.HUMAN_RESPONSE if operation == "read" else EventType.HUMAN_OUTPUT
            ),
            event_source=request.human if operation == "read" else request.pid,
            event_target=resource,
            event_payload=event_payload,
            audit_action=f"human.terminal.{operation}{'.failed' if failed else ''}",
            audit_actor=request.pid,
            audit_target=resource,
            audit_decision=decision,
            effect_metadata={"result_observation": result_observation},
        )

    def _protected_output_evidence(
        self,
        request: HumanRequest,
        resource: str,
        channel: str,
        *,
        failed: bool = False,
        phase: str | None = None,
        error: BaseException | None = None,
    ) -> ProtectedOperationEvidence:
        event_payload: dict[str, Any] = {
            "request_id": request.request_id,
            "channel": channel,
            "chars": int(len(str(request.payload.get("message", "")))),
        }
        decision: dict[str, Any] = {
            **event_payload,
            "delivery_committed": True,
        }
        if failed:
            event_payload.update(
                {"outcome": "unknown", "phase": phase, "error_type": type(error).__name__}
            )
            decision.update(
                {"effect_outcome": "unknown", "phase": phase, "error_type": type(error).__name__}
            )
        return ProtectedOperationEvidence(
            event_type=EventType.HUMAN_OUTPUT,
            event_source=request.pid,
            event_target=resource,
            event_payload=event_payload,
            audit_action="human.output.failed" if failed else "human.output",
            audit_actor=request.pid,
            audit_target=resource,
            audit_decision=decision,
            effect_metadata={
                "delivery_committed": True,
                "delivered": not failed,
                **({"phase": phase, "error_type": type(error).__name__} if failed else {}),
            },
        )

    def _terminal_text_observation(self, text: str) -> dict[str, Any]:
        encoded = text.encode("utf-8")
        return {
            "chars": len(text),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def _select_boolean_approval(
        self,
        request: HumanRequest,
        question: str,
        auto_approve: bool | None,
    ) -> bool:
        if auto_approve is None:
            answer = str(
                self._terminal_provider_io(
                    request,
                    operation="read",
                    text=f"{question} [y/N]: ",
                    purpose="boolean_approval",
                )
            ).strip().lower()
            return answer in {"y", "yes"}
        self._terminal_provider_io(
            request,
            operation="write",
            text=f"{question} [{'approved' if auto_approve else 'rejected'}]",
            purpose="boolean_approval_auto",
        )
        return auto_approve

    def _select_text_answer(
        self,
        request: HumanRequest,
        question: str,
        auto_answer: str | None,
    ) -> str:
        if auto_answer is not None:
            self._validate_human_response(
                {
                    "approved": True,
                    "answer": auto_answer,
                    "source": "terminal_queue",
                },
                label="human decision",
            )
            self._terminal_provider_io(
                request,
                operation="write",
                text=f"{question} [answer={auto_answer!r}]",
                purpose="text_answer_auto",
            )
            return auto_answer
        return str(
            self._terminal_provider_io(
                request,
                operation="read",
                text=f"{question} ",
                purpose="text_answer",
            )
        )

    def _deliver_output_request(self, request: HumanRequest) -> HumanRequest:
        message = str(request.payload.get("message", ""))
        channel = str(request.payload.get("channel", self.config.runtime.terminal_channel))
        effect_context = {
            "channel": channel,
            "chars": len(message),
            "request_id": request.request_id,
            "request_kind": "output",
        }
        resource = f"human:{request.human}"

        def prepare() -> None:
            nonlocal request
            latest = self.requests.get(request.request_id)
            if latest is None:
                raise NotFound(f"human request not found: {request.request_id}")
            if latest.status != HumanRequestStatus.PENDING:
                raise ValidationError(
                    f"human output request is not pending: {request.request_id} status={latest.status.value}"
                )
            process = self.processes.get_process(latest.pid)
            if process is not None and process.status in self.TERMINAL_PROCESS_STATUSES:
                raise ValidationError(
                    f"terminal process cannot deliver human output: {latest.pid} status={process.status.value}"
                )
            request = self.requests.replace_current(
                latest,
                status=HumanRequestStatus.DELIVERED,
                decision={"delivery_committed": True},
                updated_at=utc_now(),
            )

        def restore_not_started() -> None:
            latest = self.requests.get(request.request_id)
            if latest is not None and latest.status == HumanRequestStatus.DELIVERED:
                self.requests.replace_current(
                    latest,
                    status=HumanRequestStatus.PENDING,
                    decision={
                        "delivery_committed": False,
                        "provider_not_started": True,
                    },
                    updated_at=utc_now(),
                )

        def settle_success() -> None:
            latest = self.requests.get(request.request_id)
            if latest is None:
                raise NotFound(f"human request not found: {request.request_id}")
            if latest.status != HumanRequestStatus.DELIVERED:
                raise ValidationError(
                    "human output delivery changed concurrently: "
                    f"{latest.request_id} status={latest.status.value}"
                )
            self.requests.replace_current(
                latest,
                decision={"delivery_committed": True, "delivered": True},
                updated_at=utc_now(),
            )

        def update_latest_delivery_decision(
            decision: dict[str, Any],
        ) -> HumanRequest | None:
            # Provider I/O happens outside the Store transaction. Always
            # refetch before the trailing best-effort update so concurrent GUI
            # release/presentation metadata is not overwritten by the snapshot
            # captured in prepare().
            with self.requests.transaction():
                latest = self.requests.get(request.request_id)
                if latest is None or latest.status != HumanRequestStatus.DELIVERED:
                    return latest
                return self.requests.replace_current(
                    latest,
                    decision=dict(decision),
                    updated_at=utc_now(),
                )

        invocation = ProtectedOperationInvocation(
            pid=request.pid,
            actor=request.pid,
            target=resource,
            canonical_args={
                "request_id": request.request_id,
                "channel": channel,
                "message": message,
            },
            observation=effect_context,
            data_sink=DataSink(identity=f"human:{request.human}:{channel}"),
            data_flow_context=self._request_data_flow_context(request),
            data_flow_payload=message,
            data_flow_operation="human.output",
            data_flow_allow_recovered_source_snapshots=True,
            prepare=prepare,
            restore_not_started=restore_not_started,
            failure_evidence=lambda error, phase: self._protected_output_evidence(
                request, resource, channel, failed=True, phase=phase, error=error
            ),
        )
        provider_attempted = False
        try:
            with self._protected().start(
                "primitive.human.write", invocation, provider=self.provider
            ) as protected:
                def write_once() -> None:
                    nonlocal provider_attempted
                    provider_attempted = True
                    self.delivery.write(message)

                protected.call(
                    ProviderPhase("output", state_mutation=True, information_flow=True),
                    write_once,
                )
                result = protected.complete(
                    request,
                    self._protected_output_evidence(request, resource, channel),
                    classification_context=effect_context,
                    classification_result={"delivery_committed": True, "delivered": True},
                    settle_success=settle_success,
                )
        except ProviderEffectNotStarted:
            raise
        except BaseException as error:
            if not provider_attempted:
                raise
            try:
                update_latest_delivery_decision(
                    {
                        "delivery_committed": True,
                        "provider_error_type": type(error).__name__,
                    }
                )
            except Exception:
                pass
            raise

        # PRESERVE_RESULT means bookkeeping failures after the terminal write
        # are intentionally not retryable. Keep the request delivered even if
        # the durable pending effect is the only surviving evidence.
        latest_result: HumanRequest | None = None
        try:
            latest_result = update_latest_delivery_decision(
                {"delivery_committed": True, "delivered": True}
            )
        except Exception:
            pass
        return latest_result or result

    def _request_source_context(
        self,
        pid: str,
        *,
        source_oids: Iterable[str] | None = None,
        public_metadata: bool = False,
    ) -> DataFlowContext:
        if public_metadata:
            return DataFlowContext(
                labels=DataLabels(
                    sensitivity=DataSensitivity.PUBLIC,
                    trust_level=DataTrustLevel.VERIFIED,
                    integrity=DataIntegrity.VERIFIED,
                    origin="runtime:data-release-metadata",
                )
            )
        return self.data_flow.context_from_source_oids(pid, source_oids)

    def _request_data_flow_context(self, request: HumanRequest) -> DataFlowContext:
        raw = request.payload.get(_DATA_FLOW_CONTEXT_KEY)
        if raw is None:
            return DataFlowContext()
        if not isinstance(raw, Mapping):
            raise ValidationError("Human request has invalid trusted data-flow context")
        try:
            return DataFlowContext.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Human request has invalid trusted data-flow context: {exc}") from exc

    def _presentation_data_flow_context(
        self,
        request: HumanRequest,
    ) -> DataFlowContext | None:
        """Return the context for presenting a frozen Human-request payload.

        A successfully delivered output has already crossed a protected
        provider boundary while every original source reference was current.
        Its stored message is therefore a frozen payload snapshot: later
        updates to a mutable LLM context or another source cannot change those
        bytes.  Presentation must still enforce the captured labels and the
        current Sink policy, but rechecking source *freshness* would turn an
        ordinary post-output process transition into a false denial.

        New requests carry an internal message digest so accidental payload
        mutation fails closed.  Legacy delivered rows predate that marker and
        are accepted through the same durable delivered/committed receipt.
        Pending questions, approvals, and uncertain deliveries retain the
        ordinary live-source checks.
        """

        context = self._request_data_flow_context(request)
        decision = request.decision
        if not (
            request.payload.get("type") == "output"
            and request.status == HumanRequestStatus.DELIVERED
            and isinstance(decision, Mapping)
            and decision.get("delivery_committed") is True
            and decision.get("delivered") is True
        ):
            return context

        message = request.payload.get("message")
        if not isinstance(message, str):
            return None
        expected = request.payload.get(_OUTPUT_SNAPSHOT_SHA256_KEY)
        if expected is not None:
            actual = hashlib.sha256(message.encode("utf-8")).hexdigest()
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or not hmac.compare_digest(expected, actual)
            ):
                return None

        return DataFlowContext(
            labels=context.labels,
            materialization_id=context.materialization_id,
        )

    def _precheck_human_egress(
        self,
        *,
        pid: str,
        human: str,
        channel: str,
        context: DataFlowContext,
        payload: Any,
    ) -> None:
        self.data_flow.precheck_egress_clearance(
            pid=pid,
            sink=DataSink(identity=f"human:{human}:{channel}"),
            context=context,
            payload=payload,
        )

    def _observe_human_response(self, request_context: DataFlowContext) -> None:
        self.data_flow.observe_ingress(
            self._human_response_context(request_context)
        )

    @staticmethod
    def _human_response_context(request_context: DataFlowContext) -> DataFlowContext:
        external = DataFlowContext(
            labels=DataLabels(
                sensitivity=DataSensitivity.NORMAL,
                trust_level=DataTrustLevel.UNTRUSTED,
                integrity=DataIntegrity.UNTRUSTED,
                origin="external:human",
            )
        )
        return DataFlowContext.aggregate((request_context, external))

    def recover_prepared_output(self, effect: Any) -> None:
        """Undo a durable output claim that never reached the Human provider."""

        context = effect.provider_metadata.get("context")
        request_id = context.get("request_id") if isinstance(context, dict) else None
        if not isinstance(request_id, str) or not request_id:
            raise ValidationError(
                f"prepared Human output is missing request identity: {effect.effect_id}"
            )
        request = self.requests.get(request_id)
        if request is None:
            raise NotFound(f"human request not found during prepared recovery: {request_id}")
        if request.status == HumanRequestStatus.PENDING:
            return
        if request.status != HumanRequestStatus.DELIVERED:
            raise ValidationError(
                f"prepared Human output recovery found incompatible status: {request_id} "
                f"status={request.status.value}"
            )
        self.requests.replace_current(
            request,
            status=HumanRequestStatus.PENDING,
            decision={
                "delivery_committed": False,
                "provider_not_dispatched": True,
                "startup_recovered": True,
            },
            updated_at=utc_now(),
        )

    def _default_message_subject(self, kind: ProcessMessageKind) -> str:
        if kind == ProcessMessageKind.INTERRUPT:
            return "Human interrupt"
        return "Human message"
