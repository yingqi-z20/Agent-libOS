"""Host-owned client manager for the pinned MCP Tasks extension.

Remote task identifiers are treated as bearer-like secrets and exist only in
an :class:`~agent_libos.mcp.providers.McpCredentialBroker`.  Public callers use
the local ``task_ref``.  There is deliberately no ``tasks/list`` operation.

Every network operation crosses a Host-supplied boundary which must perform
Capability/Human/data-flow/budget checks and pending-first effect accounting.
Mutation calls are claimed with a durable CAS before dispatch.  An ambiguous
failure or restart becomes ``needs_attention`` and is never replayed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import math
import threading
from typing import Any, Protocol, runtime_checkable

from agent_libos.mcp._input import (
    canonical_json_bytes,
    decode_broker_json,
    json_sha256,
    parse_input_requests,
    reject_opaque_secret_reflection,
    sanitize_provider_json,
    sdk_json_mapping,
    strict_json_value,
    validate_input_responses,
)
from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.mcp.human import (
    McpHumanRequestBridge,
    McpHumanRequestReceipt,
    mcp_human_preview,
    require_human_receipt,
)
from agent_libos.mcp.providers import McpCredentialBroker
from agent_libos.mcp.side_effects import (
    McpSideEffectRepository,
    cleanup_mcp_preparation,
    commit_mcp_preparation,
    commit_mcp_preparation_deferred,
    commit_terminal_mcp_preparation,
    finalize_mcp_preparation,
    prepare_mcp_side_effects,
    reconcile_mcp_preparations,
    write_mcp_prepared_secrets,
)
from agent_libos.mcp.supervisor import McpConnectionFence
from agent_libos.mcp.types import (
    JsonValue,
    McpInputRequest,
    McpInputRequestKind,
    McpRemoteTask,
    McpRemoteTaskStatus,
    McpSubscriptionEvent,
)
from agent_libos.models.base import StrEnum
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.storage.mcp_v7 import (
    McpRemoteTaskRecord,
    McpSideEffectPreparationRecord,
)


_SHA_CHARS = frozenset("0123456789abcdef")
_TERMINAL = frozenset(
    {
        McpRemoteTaskStatus.COMPLETED,
        McpRemoteTaskStatus.FAILED,
        McpRemoteTaskStatus.CANCELLED,
    }
)
_MAX_REMOTE_ID_BYTES = 8 * 1024
_MAX_STATUS_CHARS = 8 * 1024
_MAX_TTL_MS = 365 * 24 * 60 * 60 * 1_000
_MAX_POLL_INTERVAL_MS = 24 * 60 * 60 * 1_000
_MAX_INPUT_REQUESTS = 16
_TASK_RESULT_FIELDS = frozenset(
    {
        "resultType",
        "taskId",
        "status",
        "statusMessage",
        "createdAt",
        "lastUpdatedAt",
        "ttlMs",
        "pollIntervalMs",
        "inputRequests",
        "result",
        "error",
        "_meta",
    }
)
_TASK_NOTIFICATION_FIELDS = frozenset(
    {
        "taskId",
        "status",
        "statusMessage",
        "createdAt",
        "lastUpdatedAt",
        "ttlMs",
        "pollIntervalMs",
    }
)


class McpRemoteTaskRecordStatus(StrEnum):
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCEL_REQUESTED = "cancel_requested"
    UPDATE_DISPATCHING = "update_dispatching"
    CANCEL_DISPATCHING = "cancel_dispatching"
    NEEDS_ATTENTION = "needs_attention"


class McpRemoteTaskDispatchNotStarted(ValidationError):
    """Trusted boundary signal that no Tasks Provider request was dispatched."""


@dataclass(frozen=True)
class McpRemoteTaskBinding:
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    origin_request_sha256: str
    origin_effect_id: str
    extension_id: str
    tasks_extension_sha256: str
    host_tasks_extension_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("server id", self.server_id),
            ("owner id", self.owner_id),
            ("origin effect id", self.origin_effect_id),
            ("extension id", self.extension_id),
        ):
            _require_text(value, f"MCP remote Task {label}")
        for label, value in (
            ("server spec", self.server_spec_sha256),
            ("auth principal", self.auth_principal_sha256),
            ("auth scope", self.auth_scope_sha256),
            ("origin request", self.origin_request_sha256),
            ("extension spec", self.tasks_extension_sha256),
            ("Host extension pin", self.host_tasks_extension_sha256),
        ):
            _require_sha256(value, f"MCP remote Task {label}")
        if type(self.server_generation) is not int or self.server_generation < 0:
            raise ValidationError("MCP remote Task server generation is invalid")

    def require_extension_pin(self) -> None:
        if self.extension_id != MCP_TASKS_EXTENSION_ID:
            raise ValidationError("MCP Tasks extension is not enabled")
        if self.tasks_extension_sha256 != self.host_tasks_extension_sha256:
            raise ValidationError("MCP Tasks extension does not match the Host pin")


@runtime_checkable
class McpRemoteTaskRepository(Protocol):
    def insert(self, record: McpRemoteTaskRecord) -> None: ...

    def get(self, task_ref: str) -> McpRemoteTaskRecord | None: ...

    def get_by_remote_id_sha256(
        self,
        server_id: str,
        remote_id_sha256: str,
    ) -> McpRemoteTaskRecord | None: ...

    def list(self, **filters: object) -> list[McpRemoteTaskRecord]: ...

    def count(self, *, owner_id: str | None = None) -> int: ...

    def count_active(self, *, owner_id: str | None = None) -> int: ...

    def list_terminal(
        self,
        *,
        owner_id: str | None = None,
        limit: int = 100,
    ) -> tuple[McpRemoteTaskRecord, ...]: ...

    def delete_terminal(
        self,
        task_ref: str,
        *,
        expected_revision: int,
    ) -> bool: ...

    def compare_and_swap(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        replacement: McpRemoteTaskRecord,
    ) -> bool: ...


@runtime_checkable
class McpRemoteTaskBoundary(Protocol):
    async def get_remote_task(
        self,
        *,
        record: McpRemoteTaskRecord,
        binding: McpRemoteTaskBinding,
        remote_task_id: str,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...

    async def update_remote_task(
        self,
        *,
        record: McpRemoteTaskRecord,
        binding: McpRemoteTaskBinding,
        remote_task_id: str,
        input_responses: Mapping[str, JsonValue],
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...

    async def cancel_remote_task(
        self,
        *,
        record: McpRemoteTaskRecord,
        binding: McpRemoteTaskBinding,
        remote_task_id: str,
        deadline: float,
    ) -> Mapping[str, JsonValue]: ...


class McpRemoteTaskCaptureBindingResolver(Protocol):
    def __call__(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpRemoteTaskBinding: ...


class McpContinuationTaskBindingResolver(Protocol):
    def __call__(
        self,
        binding: Any,
        *,
        origin_effect_id: str,
    ) -> McpRemoteTaskBinding: ...


class McpSdkRemoteTaskCaptureAdapter:
    """Bridge an official SDK ``CreateTaskResult`` to a local task ref."""

    def __init__(
        self,
        manager: McpRemoteTaskManager,
        binding_resolver: McpRemoteTaskCaptureBindingResolver,
    ) -> None:
        self.manager = manager
        self.binding_resolver = binding_resolver

    def capture_remote_task(
        self,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
        result: Any,
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> McpRemoteTask:
        _deadline(deadline)
        binding = self.binding_resolver(server_id, operation, logical_id)
        if binding.server_id != server_id:
            raise CapabilityDenied("MCP remote Task resolver changed server binding")
        raw = sdk_json_mapping(result, label="MCP SDK CreateTaskResult")
        return self.manager.prepare_initial_task(
            binding,
            raw,
            sensitive_values=sensitive_values,
        )


class McpContinuationRemoteTaskCaptureAdapter:
    """Capture a Task returned by an already-governed continuation call.

    The continuation binding carries the immutable server and original-effect
    fences.  The protected continuation boundary has already revalidated that
    exact Manifest against the Host Tasks pin before Provider dispatch, so no
    registry lookup or replay occurs during settlement.
    """

    def __init__(
        self,
        manager: McpRemoteTaskManager,
        binding_resolver: McpContinuationTaskBindingResolver,
    ) -> None:
        if not callable(binding_resolver):
            raise TypeError("MCP continuation Task binding resolver is required")
        self.manager = manager
        self.binding_resolver = binding_resolver

    def __call__(
        self,
        binding: Any,
        result: Mapping[str, JsonValue],
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpRemoteTask:
        task_binding = self._resolved_binding(binding)
        return self.manager.capture_task(
            task_binding,
            result,
            sensitive_values=sensitive_values,
        )

    def prepare_capture(
        self,
        binding: Any,
        result: Mapping[str, JsonValue],
        *,
        origin_effect_id: str,
        sensitive_values: tuple[str, ...] = (),
    ) -> tuple[McpRemoteTask, "McpRemoteTaskCaptureSettlement"]:
        """Prepare, but do not publish, a continuation-returned Task."""

        task_binding = self._resolved_binding(
            binding,
            origin_effect_id=origin_effect_id,
        )
        public = self.manager.prepare_initial_task(
            task_binding,
            result,
            sensitive_values=sensitive_values,
        )
        return public, self.manager.claim_initial_capture(
            public,
            binding=task_binding,
        )

    def _resolved_binding(
        self,
        binding: Any,
        *,
        origin_effect_id: str | None = None,
    ) -> McpRemoteTaskBinding:
        selected_effect_id = (
            binding.effect_id if origin_effect_id is None else origin_effect_id
        )
        task_binding = self.binding_resolver(
            binding,
            origin_effect_id=selected_effect_id,
        )
        if not isinstance(task_binding, McpRemoteTaskBinding):
            raise TypeError("MCP continuation Task resolver returned an invalid binding")
        task_binding.require_extension_pin()
        return task_binding


@dataclass(frozen=True)
class _ParsedTask:
    remote_id: str
    status: McpRemoteTaskStatus
    status_message: str | None
    created_at: str
    updated_at: str
    ttl_ms: int | None
    poll_interval_ms: int | None
    result: JsonValue | None
    input_requests: tuple[McpInputRequest, ...]
    raw_state: dict[str, JsonValue]


@dataclass
class _PendingRemoteTaskCapture:
    preparation: McpSideEffectPreparationRecord
    record: McpRemoteTaskRecord
    parsed: _ParsedTask
    public: McpRemoteTask
    binding: McpRemoteTaskBinding
    claimed: bool = False
    retirement: McpSideEffectPreparationRecord | None = None


@dataclass(frozen=True)
class McpRemoteTaskCaptureSettlement:
    """Host-only settlement token for one prepared initial Task capture."""

    manager: "McpRemoteTaskManager"
    task_ref: str

    def commit_deferred(self) -> None:
        self.manager.commit_prepared_capture_deferred(self.task_ref)

    def finalize(self) -> None:
        self.manager.finalize_prepared_capture(self.task_ref)

    def abort(self, *, reason: str = "task_capture_failed") -> None:
        self.manager.abort_prepared_capture(self.task_ref, reason=reason)


class McpRemoteTaskManager:
    def __init__(
        self,
        *,
        repository: McpRemoteTaskRepository,
        side_effects: McpSideEffectRepository,
        broker: McpCredentialBroker,
        human_requests: McpHumanRequestBridge,
        boundary: McpRemoteTaskBoundary,
        now: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
        sensitive_values: tuple[str, ...] = (),
        max_input_requests: int = _MAX_INPUT_REQUESTS,
        poll_min_interval_s: float = 0.0,
        max_wait_s: float = 3_600.0,
        max_records: int = 1_000,
        terminal_records: int = 256,
        reconcile_on_start: bool = True,
    ) -> None:
        if (
            repository is None
            or side_effects is None
            or broker is None
            or human_requests is None
            or boundary is None
        ):
            raise TypeError("MCP remote Task dependencies are required")
        _validate_remote_task_policy(
            max_input_requests=max_input_requests,
            poll_min_interval_s=poll_min_interval_s,
            max_wait_s=max_wait_s,
            max_records=max_records,
            terminal_records=terminal_records,
            reconcile_on_start=reconcile_on_start,
        )
        self.repository = repository
        self.side_effects = side_effects
        self.broker = broker
        self.human_requests = human_requests
        self.boundary = boundary
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or _new_task_ref
        self._sensitive_values = tuple(
            item for item in sensitive_values if type(item) is str and item
        )
        self._max_input_requests = max_input_requests
        self._poll_min_interval_s = float(poll_min_interval_s)
        self._max_wait = timedelta(seconds=float(max_wait_s))
        self._max_records = max_records
        self._terminal_records = terminal_records
        self._pending_capture_lock = threading.RLock()
        self._pending_captures: dict[str, _PendingRemoteTaskCapture] = {}
        if reconcile_on_start:
            self.reconcile_after_restart()

    def capture_task(
        self,
        binding: McpRemoteTaskBinding,
        result: Mapping[str, JsonValue],
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpRemoteTask:
        """Eager compatibility capture outside a protected initial operation."""

        public = self.prepare_initial_task(
            binding,
            result,
            sensitive_values=sensitive_values,
        )
        settlement = self.claim_initial_capture(public, binding=binding)
        try:
            settlement.commit_deferred()
        except Exception:
            settlement.abort(reason="task_capture_failed")
            raise
        try:
            settlement.finalize()
        except Exception:
            if self.repository.get(public.task_ref) is None:
                raise
        return public

    def prepare_initial_task(
        self,
        binding: McpRemoteTaskBinding,
        result: Mapping[str, JsonValue],
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpRemoteTask:
        """Prepare Human/broker ownership without publishing the Task row."""

        self._require_binding_type(binding)
        binding.require_extension_pin()
        parsed = self._parse_task(
            result,
            expected_result_type="task",
            sensitive_values=_merge_sensitive_values(
                self._sensitive_values,
                sensitive_values,
            ),
        )
        self._reconcile_expired(limit=500)
        self._prune_terminal_records()
        with self._pending_capture_lock:
            pending_count = len(self._pending_captures)
        if self.repository.count_active() + pending_count >= self._max_records:
            raise ValidationError("MCP remote Task record limit is exhausted")
        task_ref = self._id_factory()
        _require_text(task_ref, "MCP remote Task local reference")
        expires_at = self._bounded_task_expiry(parsed)
        human_receipt: McpHumanRequestReceipt | None = None
        human_preview_sha256: str | None = None
        human_request_id: str | None = None
        human_preview: dict[str, JsonValue] | None = None
        selected_status = _record_status(parsed.status)
        if parsed.status is McpRemoteTaskStatus.INPUT_REQUIRED:
            if _has_unsupported_input(parsed):
                selected_status = McpRemoteTaskRecordStatus.NEEDS_ATTENTION
            else:
                human_preview, human_preview_sha256 = mcp_human_preview(
                    server_id=binding.server_id,
                    operation="tasks/update",
                    local_ref=task_ref,
                    input_requests=parsed.input_requests,
                )
                human_request_id = self.human_requests.reserve_question_id()
                _require_text(human_request_id, "MCP Human request id")
        remote_secret = parsed.remote_id.encode("utf-8")
        state_secret = canonical_json_bytes(
            parsed.raw_state,
            label="MCP remote Task broker state",
        )
        remote_namespace = (
            None
            if parsed.status in _TERMINAL
            else f"mcp.remote-task.id.{task_ref}.r0"
        )
        state_namespace = f"mcp.remote-task.state.{task_ref}.r0"
        now = _utc(self._now())
        preparation = prepare_mcp_side_effects(
            repository=self.side_effects,
            broker=self.broker,
            operation_kind="remote_task",
            operation_id=task_ref,
            operation_revision=None,
            server_id=binding.server_id,
            server_spec_sha256=binding.server_spec_sha256,
            server_generation=binding.server_generation,
            owner_id=binding.owner_id,
            auth_principal_sha256=binding.auth_principal_sha256,
            auth_scope_sha256=binding.auth_scope_sha256,
            human_request_id=human_request_id,
            human_preview_sha256=human_preview_sha256,
            broker_namespace=remote_namespace,
            broker_value_sha256=(
                sha256(remote_secret).hexdigest()
                if remote_namespace is not None
                else None
            ),
            result_namespace=state_namespace,
            result_sha256=sha256(state_secret).hexdigest(),
            expires_at=expires_at,
            created_at=_timestamp(now),
        )
        try:
            if human_request_id is not None:
                if human_preview is None or human_preview_sha256 is None:
                    raise ValidationError("MCP Task Human preview is unavailable")
                human_receipt = require_human_receipt(
                    self.human_requests.create_question(
                        owner_id=binding.owner_id,
                        server_id=binding.server_id,
                        operation="tasks/update",
                        local_ref=task_ref,
                        preview=human_preview,
                        preview_sha256=human_preview_sha256,
                        expires_at=expires_at,
                        request_id=human_request_id,
                    ),
                    preview_sha256=human_preview_sha256,
                )
                if human_receipt.request_id != human_request_id:
                    raise ValidationError("MCP Human request reservation changed")
            write_mcp_prepared_secrets(
                preparation,
                broker=self.broker,
                broker_namespace=remote_namespace,
                broker_value=(
                    remote_secret if remote_namespace is not None else None
                ),
                result_namespace=state_namespace,
                result_value=state_secret,
            )
            record = McpRemoteTaskRecord(
                task_ref=task_ref,
                server_id=binding.server_id,
                server_spec_sha256=binding.server_spec_sha256,
                server_generation=binding.server_generation,
                owner_id=binding.owner_id,
                auth_principal_sha256=binding.auth_principal_sha256,
                auth_scope_sha256=binding.auth_scope_sha256,
                origin_request_sha256=binding.origin_request_sha256,
                origin_effect_id=binding.origin_effect_id,
                human_request_id=(
                    human_receipt.request_id if human_receipt is not None else None
                ),
                broker_ref=preparation.broker_ref,
                remote_id_sha256=sha256(remote_secret).hexdigest(),
                status=selected_status,
                revision=0,
                expires_at=expires_at,
                poll_interval_ms=parsed.poll_interval_ms,
                status_message_sha256=(
                    sha256(parsed.status_message.encode("utf-8")).hexdigest()
                    if parsed.status_message is not None
                    else None
                ),
                result_ref=preparation.result_ref,
                result_sha256=sha256(state_secret).hexdigest(),
                metadata=_task_metadata("not_started", "reobserve_required", parsed),
                created_at=_timestamp(now),
                updated_at=_timestamp(now),
            )
            public = self._public(record, parsed)
            pending = _PendingRemoteTaskCapture(
                preparation=preparation,
                record=record,
                parsed=parsed,
                public=deepcopy(public),
                binding=binding,
            )
            with self._pending_capture_lock:
                if task_ref in self._pending_captures:
                    raise ValidationError("MCP remote Task capture id was reused")
                self._pending_captures[task_ref] = pending
        except Exception:
            cleanup_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason="task_capture_failed",
            )
            raise
        return deepcopy(public)

    def claim_initial_capture(
        self,
        public: McpRemoteTask,
        *,
        binding: McpRemoteTaskBinding,
    ) -> McpRemoteTaskCaptureSettlement:
        """Claim one exact prepared capture for protected success settlement."""

        if not isinstance(public, McpRemoteTask):
            raise TypeError("MCP remote Task public capture is required")
        self._require_binding_type(binding)
        binding.require_extension_pin()
        with self._pending_capture_lock:
            pending = self._pending_captures.get(public.task_ref)
            if (
                pending is None
                or pending.claimed
                or pending.binding != binding
                or pending.public != public
                or pending.record.origin_effect_id != binding.origin_effect_id
            ):
                raise ValidationError("MCP remote Task prepared provenance changed")
            current = self.side_effects.get(pending.preparation.preparation_id)
            if current != pending.preparation:
                raise ValidationError("MCP remote Task preparation changed")
            pending.claimed = True
        return McpRemoteTaskCaptureSettlement(self, public.task_ref)

    def has_prepared_effect(self, effect_id: str) -> bool:
        with self._pending_capture_lock:
            return any(
                pending.record.origin_effect_id == effect_id
                for pending in self._pending_captures.values()
            )

    def abort_prepared_effect(self, effect_id: str) -> None:
        with self._pending_capture_lock:
            refs = tuple(
                task_ref
                for task_ref, pending in self._pending_captures.items()
                if pending.record.origin_effect_id == effect_id
            )
        for task_ref in refs:
            self.abort_prepared_capture(task_ref, reason="protected_operation_failed")

    def commit_prepared_capture_deferred(self, task_ref: str) -> None:
        with self._pending_capture_lock:
            pending = self._pending_captures.get(task_ref)
            if pending is None or not pending.claimed or pending.retirement is not None:
                raise ValidationError("MCP remote Task settlement token is invalid")
            pending.retirement = commit_mcp_preparation_deferred(
                self.side_effects,
                pending.preparation,
                pending.record,
            )

    def finalize_prepared_capture(self, task_ref: str) -> None:
        with self._pending_capture_lock:
            pending = self._pending_captures.get(task_ref)
            if pending is None or pending.retirement is None:
                raise ValidationError("MCP remote Task settlement is not committed")
            retirement = pending.retirement
        try:
            finalize_mcp_preparation(
                self.side_effects,
                retirement,
                broker=self.broker,
                human_requests=self.human_requests,
            )
        finally:
            if self.repository.get(task_ref) == pending.record:
                with self._pending_capture_lock:
                    self._pending_captures.pop(task_ref, None)
        if _status(pending.record) in {
            McpRemoteTaskRecordStatus.COMPLETED,
            McpRemoteTaskRecordStatus.FAILED,
            McpRemoteTaskRecordStatus.CANCELLED,
            McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
        }:
            self._prune_terminal_records()

    def abort_prepared_capture(self, task_ref: str, *, reason: str) -> None:
        with self._pending_capture_lock:
            pending = self._pending_captures.get(task_ref)
        if pending is None:
            return
        persisted = self.repository.get(task_ref)
        if persisted is not None:
            if persisted != pending.record:
                raise ValidationError("MCP remote Task committed capture changed")
            with self._pending_capture_lock:
                self._pending_captures.pop(task_ref, None)
            return
        cleanup_mcp_preparation(
            self.side_effects,
            pending.preparation,
            broker=self.broker,
            human_requests=self.human_requests,
            updated_at=_timestamp(_utc(self._now())),
            reason=reason,
        )
        with self._pending_capture_lock:
            self._pending_captures.pop(task_ref, None)

    def binding_material(
        self,
        task_ref: str,
        *,
        tasks_extension_sha256: str,
        host_tasks_extension_sha256: str,
    ) -> McpRemoteTaskBinding:
        """Recover only the durable authority fence for a local Task ref.

        The bearer-like remote task id and broker references remain private.
        The protected Runtime facade supplies the extension digest observed in
        the current registered Manifest and the current Host pin; equality is
        checked again by every manager operation.
        """

        record = self._load(task_ref)
        binding = McpRemoteTaskBinding(
            server_id=record.server_id,
            server_spec_sha256=record.server_spec_sha256,
            server_generation=record.server_generation,
            owner_id=record.owner_id,
            auth_principal_sha256=record.auth_principal_sha256,
            auth_scope_sha256=record.auth_scope_sha256,
            origin_request_sha256=record.origin_request_sha256,
            origin_effect_id=record.origin_effect_id,
            extension_id=MCP_TASKS_EXTENSION_ID,
            tasks_extension_sha256=tasks_extension_sha256,
            host_tasks_extension_sha256=host_tasks_extension_sha256,
        )
        binding.require_extension_pin()
        return binding

    def inspect(
        self,
        task_ref: str,
        *,
        binding: McpRemoteTaskBinding,
    ) -> McpRemoteTask:
        """Return the durable local projection without polling or dispatch."""

        record = self._load(task_ref)
        self._require_binding(record, binding)
        return self._public(record, self._load_state(record))

    def project_task_notification(
        self,
        *,
        event: McpSubscriptionEvent,
        fence: McpConnectionFence,
        sensitive_values: tuple[str, ...],
    ) -> McpSubscriptionEvent:
        """Validate a full Task notification and expose only its local ref.

        Notifications are untrusted cache-invalidation hints.  This method is
        intentionally read-only: it neither advances the durable Task state
        nor creates a HumanRequest.  A caller must still explicitly execute
        protected ``tasks/get`` before using any new remote state.
        """

        if not isinstance(fence, McpConnectionFence):
            raise TypeError("MCP Task notification fence is required")
        if type(event) is not McpSubscriptionEvent or event.event_type != "taskStatus":
            raise ValidationError("MCP Task notification event is invalid")
        parsed = _parse_task_notification(
            event.payload,
            sensitive_values=_merge_sensitive_values(
                self._sensitive_values,
                sensitive_values,
            ),
        )
        remote_digest = sha256(parsed.remote_id.encode("utf-8")).hexdigest()
        record = self.repository.get_by_remote_id_sha256(
            fence.server_id,
            remote_digest,
        )
        if record is None:
            raise ValidationError("MCP Task notification identity is unknown")
        self._require_notification_fence(record, fence)
        broker_remote_id = self._read_remote_id_without_mutation(record)
        if not hmac.compare_digest(broker_remote_id, parsed.remote_id):
            raise ValidationError("MCP Task notification identity changed")
        payload: dict[str, JsonValue] = {
            "task_ref": record.task_ref,
            "status": parsed.status.value,
            "created_at": parsed.created_at,
            "last_updated_at": parsed.updated_at,
            "ttl_ms": parsed.ttl_ms,
            "poll_interval_ms": parsed.poll_interval_ms,
        }
        if parsed.status_message is not None:
            payload["status_message"] = parsed.status_message
        return McpSubscriptionEvent(
            sequence=0,
            event_type="taskStatus",
            payload=payload,
            received_at=_timestamp(_utc(self._now())),
        )

    def subscription_targets(
        self,
        *,
        fence: McpConnectionFence,
    ) -> tuple[str, ...]:
        """Resolve current local Tasks to bearer filters for one wire session.

        The returned values are an internal Provider target only.  Rows outside
        the exact connection/auth fence are skipped, terminal Tasks are never
        subscribed, and a possibly truncated repository page fails closed.
        """

        if not isinstance(fence, McpConnectionFence):
            raise TypeError("MCP Task subscription fence is required")
        remote_ids: list[str] = []
        for status in (
            McpRemoteTaskRecordStatus.WORKING,
            McpRemoteTaskRecordStatus.INPUT_REQUIRED,
            McpRemoteTaskRecordStatus.CANCEL_REQUESTED,
        ):
            records = self.repository.list(
                owner_id=fence.owner,
                server_id=fence.server_id,
                server_generation=fence.registry_generation,
                status=status.value,
                limit=500,
            )
            if len(records) >= 500:
                raise ValidationError("MCP Task subscription target page is truncated")
            for record in records:
                if _status(record) is not status:
                    continue
                try:
                    self._require_notification_fence(record, fence)
                except CapabilityDenied:
                    continue
                remote_ids.append(self._read_remote_id_without_mutation(record))
        return tuple(remote_ids)

    async def get(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        binding: McpRemoteTaskBinding,
        deadline: float,
    ) -> McpRemoteTask:
        selected_deadline = _deadline(deadline)
        record = self._prepare(
            task_ref,
            expected_revision=expected_revision,
            binding=binding,
            allowed={
                McpRemoteTaskRecordStatus.WORKING,
                McpRemoteTaskRecordStatus.INPUT_REQUIRED,
                McpRemoteTaskRecordStatus.CANCEL_REQUESTED,
                McpRemoteTaskRecordStatus.COMPLETED,
                McpRemoteTaskRecordStatus.FAILED,
                McpRemoteTaskRecordStatus.CANCELLED,
            },
        )
        status = _status(record)
        if _public_status(status) in _TERMINAL:
            return self._public(record, self._load_state(record))
        self._require_poll_ready(record)
        remote_id = self._load_remote_id(record)
        try:
            raw = await self.boundary.get_remote_task(
                record=record,
                binding=binding,
                remote_task_id=remote_id,
                deadline=selected_deadline,
            )
        except Exception:
            # tasks/get is an observation, not a mutation.  It is safe for a
            # caller to explicitly re-observe; do not manufacture task state.
            raise ValidationError(
                "MCP provider operation failed; sensitive details were omitted"
            ) from None
        parsed = self._parse_task(raw, expected_result_type="complete")
        if parsed.remote_id != remote_id:
            raise ValidationError("MCP remote Task Provider returned another task identity")
        previous = self._load_state(record)
        if parsed.created_at != previous.created_at:
            raise ValidationError("MCP remote Task creation binding changed")
        if _parse_timestamp(parsed.updated_at) < _parse_timestamp(previous.updated_at):
            raise ValidationError("MCP remote Task update timestamp moved backwards")
        self._validate_transition(status, parsed.status)
        return self._settle_observation(record, parsed)

    def prevalidate_update(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        binding: McpRemoteTaskBinding,
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        responses: Mapping[str, JsonValue],
    ) -> None:
        """Validate one Host Task answer without consuming its Human request."""

        record = self._prepare(
            task_ref,
            expected_revision=expected_revision,
            binding=binding,
            allowed={McpRemoteTaskRecordStatus.INPUT_REQUIRED},
        )
        if human_request_id != record.human_request_id:
            raise CapabilityDenied("MCP Human request belongs to another remote Task")
        parsed = self._load_state(record)
        requests = parse_input_requests(
            parsed.raw_state.get("inputRequests"),
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        _preview, expected_preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation="tasks/update",
            local_ref=record.task_ref,
            input_requests=requests.public,
        )
        if human_preview_sha256 != expected_preview_sha256:
            raise CapabilityDenied("MCP Task Human response preview binding changed")
        _require_human_response_fence(
            human_expected_revision,
            human_preview_sha256,
        )
        validate_input_responses(requests, responses)

    async def update(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        binding: McpRemoteTaskBinding,
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        deadline: float,
    ) -> McpRemoteTask:
        selected_deadline = _deadline(deadline)
        record = self._prepare(
            task_ref,
            expected_revision=expected_revision,
            binding=binding,
            allowed={McpRemoteTaskRecordStatus.INPUT_REQUIRED},
        )
        if human_request_id != record.human_request_id:
            raise CapabilityDenied("MCP Human request belongs to another remote Task")
        parsed = self._load_state(record)
        raw_requests = parsed.raw_state.get("inputRequests")
        requests = parse_input_requests(
            raw_requests,
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        _preview, expected_preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation="tasks/update",
            local_ref=record.task_ref,
            input_requests=requests.public,
        )
        if human_preview_sha256 != expected_preview_sha256:
            raise CapabilityDenied("MCP Task Human response preview binding changed")
        _require_human_response_fence(
            human_expected_revision,
            human_preview_sha256,
        )
        if record.human_request_id is None:
            raise ValidationError("MCP input-required Task has no Human request binding")
        responses = self.human_requests.consume_approved_answer(
            record.human_request_id,
            presented_revision=human_expected_revision,
            preview_sha256=expected_preview_sha256,
        )
        input_responses = validate_input_responses(requests, responses)
        remote_id = self._load_remote_id(record)
        claimed = replace(
            record,
            status=McpRemoteTaskRecordStatus.UPDATE_DISPATCHING,
            revision=record.revision + 1,
            metadata=_task_metadata("not_started", "unsafe_or_unknown", parsed),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._cas_or_conflict(record, claimed)
        try:
            acknowledgement = await self.boundary.update_remote_task(
                record=claimed,
                binding=binding,
                remote_task_id=remote_id,
                input_responses=input_responses,
                deadline=selected_deadline,
            )
            _validate_empty_ack(acknowledgement, "tasks/update")
        except McpRemoteTaskDispatchNotStarted as exc:
            restored = replace(
                claimed,
                status=McpRemoteTaskRecordStatus.INPUT_REQUIRED,
                revision=claimed.revision + 1,
                metadata=_task_metadata("not_started", "reobserve_required", parsed),
                updated_at=_timestamp(_utc(self._now())),
            )
            self._cas_or_unknown(claimed, restored)
            _raise_certified_not_started(exc)
        except Exception:
            self._mark_attention(claimed, reason="update_unknown")
            raise ValidationError(
                "MCP provider operation failed; sensitive details were omitted"
            ) from None
        working_state = replace(
            parsed,
            status=McpRemoteTaskStatus.WORKING,
            status_message=None,
            result=None,
            input_requests=(),
            raw_state={
                "taskId": parsed.remote_id,
                "status": "working",
                "createdAt": parsed.created_at,
                "lastUpdatedAt": parsed.updated_at,
                "ttlMs": parsed.ttl_ms,
                **(
                    {"pollIntervalMs": parsed.poll_interval_ms}
                    if parsed.poll_interval_ms is not None
                    else {}
                ),
            },
        )
        return self._settle_mutation(
            claimed,
            working_state,
            status=McpRemoteTaskRecordStatus.WORKING,
        )

    async def cancel(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        binding: McpRemoteTaskBinding,
        deadline: float,
    ) -> McpRemoteTask:
        selected_deadline = _deadline(deadline)
        record = self._prepare(
            task_ref,
            expected_revision=expected_revision,
            binding=binding,
            allowed={
                McpRemoteTaskRecordStatus.WORKING,
                McpRemoteTaskRecordStatus.INPUT_REQUIRED,
            },
        )
        parsed = self._load_state(record)
        remote_id = self._load_remote_id(record)
        claimed = replace(
            record,
            status=McpRemoteTaskRecordStatus.CANCEL_DISPATCHING,
            revision=record.revision + 1,
            metadata=_task_metadata("not_started", "unsafe_or_unknown", parsed),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._cas_or_conflict(record, claimed)
        try:
            acknowledgement = await self.boundary.cancel_remote_task(
                record=claimed,
                binding=binding,
                remote_task_id=remote_id,
                deadline=selected_deadline,
            )
            _validate_empty_ack(acknowledgement, "tasks/cancel")
        except McpRemoteTaskDispatchNotStarted as exc:
            restored = replace(
                claimed,
                status=record.status,
                revision=claimed.revision + 1,
                metadata=_task_metadata("not_started", "reobserve_required", parsed),
                updated_at=_timestamp(_utc(self._now())),
            )
            self._cas_or_unknown(claimed, restored)
            _raise_certified_not_started(exc)
        except Exception:
            self._mark_attention(claimed, reason="cancel_unknown")
            raise ValidationError(
                "MCP provider operation failed; sensitive details were omitted"
            ) from None
        cancel_requested = replace(
            parsed,
            status=McpRemoteTaskStatus.CANCEL_REQUESTED,
            status_message="Cancellation requested; remote execution may continue",
            result=None,
            input_requests=(),
            raw_state={
                "taskId": parsed.remote_id,
                "status": "cancel_requested",
                "statusMessage": "Cancellation requested; remote execution may continue",
                "createdAt": parsed.created_at,
                "lastUpdatedAt": parsed.updated_at,
                "ttlMs": parsed.ttl_ms,
                **(
                    {"pollIntervalMs": parsed.poll_interval_ms}
                    if parsed.poll_interval_ms is not None
                    else {}
                ),
            },
        )
        return self._settle_mutation(
            claimed,
            cancel_requested,
            status=McpRemoteTaskRecordStatus.CANCEL_REQUESTED,
        )

    def reconcile_after_restart(self) -> int:
        changed = reconcile_mcp_preparations(
            self.side_effects,
            operation_kind="remote_task",
            broker=self.broker,
            human_requests=self.human_requests,
            updated_at=_timestamp(_utc(self._now())),
        )
        changed += self._reconcile_expired(limit=500)
        dispatching = (
            McpRemoteTaskRecordStatus.UPDATE_DISPATCHING,
            McpRemoteTaskRecordStatus.CANCEL_DISPATCHING,
        )
        for selected_status in dispatching:
            while True:
                batch = self.repository.list(
                    status=selected_status.value,
                    limit=500,
                )
                candidates = [
                    record for record in batch if _status(record) is selected_status
                ]
                if not candidates:
                    break
                changed_this_round = 0
                for record in candidates:
                    target = replace(
                        record,
                        status=McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
                        revision=record.revision + 1,
                        broker_ref=None,
                        result_ref=None,
                        result_sha256=None,
                        metadata=_plain_metadata(
                            "unknown",
                            "unsafe_or_unknown",
                            "runtime_restart",
                        ),
                        updated_at=_timestamp(_utc(self._now())),
                    )
                    self._commit_retirement_transition(
                        record,
                        target,
                        reason="remote_task_dispatch_interrupted",
                    )
                    changed += 1
                    changed_this_round += 1
                if changed_this_round == 0:
                    break
        self._prune_terminal_records()
        return changed

    def _reconcile_expired(self, *, limit: int) -> int:
        now = _utc(self._now())
        changed = 0
        for selected_status in (
            McpRemoteTaskRecordStatus.WORKING,
            McpRemoteTaskRecordStatus.INPUT_REQUIRED,
            McpRemoteTaskRecordStatus.CANCEL_REQUESTED,
        ):
            for record in self.repository.list(
                status=selected_status.value,
                limit=limit,
            ):
                if (
                    _status(record) is not selected_status
                    or record.expires_at is None
                    or now < _parse_timestamp(record.expires_at)
                ):
                    continue
                target = replace(
                    record,
                    status=McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
                    revision=record.revision + 1,
                    broker_ref=None,
                    result_ref=None,
                    result_sha256=None,
                    metadata=_plain_metadata(
                        "not_started",
                        "not_applicable",
                        "expired",
                    ),
                    updated_at=_timestamp(now),
                )
                self._commit_retirement_transition(
                    record,
                    target,
                    reason="remote_task_expired",
                )
                changed += 1
        return changed

    def _prune_terminal_records(self) -> int:
        removed = 0
        while True:
            records = self.repository.list_terminal(
                limit=self._terminal_records + 1,
            )
            excess = len(records) - self._terminal_records
            if excess <= 0:
                return removed
            changed = 0
            for record in records[:excess]:
                self._commit_terminal_retirement(
                    record,
                    reason="remote_task_retention_expired",
                )
                removed += 1
                changed += 1
            if changed == 0:
                return removed

    def _prepare(
        self,
        task_ref: str,
        *,
        expected_revision: int,
        binding: McpRemoteTaskBinding,
        allowed: set[McpRemoteTaskRecordStatus],
    ) -> McpRemoteTaskRecord:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValidationError("MCP remote Task expected_revision is invalid")
        record = self._load(task_ref)
        self._require_binding(record, binding)
        if record.revision != expected_revision:
            raise ValidationError("MCP remote Task revision conflict")
        if _status(record) not in allowed:
            raise ValidationError("MCP remote Task operation is invalid in its current state")
        if record.expires_at is not None and _utc(self._now()) >= _parse_timestamp(record.expires_at):
            self._mark_attention(record, reason="expired")
            raise ValidationError("MCP remote Task expired")
        return record

    def _parse_task(
        self,
        result: Mapping[str, JsonValue],
        *,
        expected_result_type: str,
        allow_local_cancel_requested: bool = False,
        sensitive_values: tuple[str, ...] | None = None,
    ) -> _ParsedTask:
        selected = _validated_task_result(result, expected_result_type)
        selected_sensitive = (
            self._sensitive_values
            if sensitive_values is None
            else sensitive_values
        )
        remote_id = _validated_remote_task_id(selected.get("taskId"), selected_sensitive)
        content_sensitive = _merge_sensitive_values(
            selected_sensitive,
            (remote_id,),
        )
        status = _validated_remote_task_status(
            selected.get("status"),
            allow_local_cancel_requested=allow_local_cancel_requested,
        )
        created, updated, ttl, poll = _validated_task_timing(selected)
        public_message = _sanitized_task_status_message(
            selected.get("statusMessage"),
            content_sensitive,
        )
        input_requests, public_result, normalized_requests = _validated_task_payload(
            selected,
            status=status,
            sensitive_values=content_sensitive,
            max_input_requests=self._max_input_requests,
        )
        raw_state = _sanitized_task_state(
            selected,
            sensitive_values=content_sensitive,
            normalized_requests=normalized_requests,
            remote_id=remote_id,
        )
        return _ParsedTask(
            remote_id=remote_id,
            status=status,
            status_message=public_message,
            created_at=created,
            updated_at=updated,
            ttl_ms=ttl,
            poll_interval_ms=poll,
            result=public_result,
            input_requests=input_requests,
            raw_state=raw_state,
        )

    def _settle_observation(
        self,
        record: McpRemoteTaskRecord,
        parsed: _ParsedTask,
    ) -> McpRemoteTask:
        target_status = _record_status(parsed.status)
        human_receipt: McpHumanRequestReceipt | None = None
        human_preview_sha256: str | None = None
        create_human = False
        if parsed.status is McpRemoteTaskStatus.INPUT_REQUIRED:
            if _has_unsupported_input(parsed):
                target_status = McpRemoteTaskRecordStatus.NEEDS_ATTENTION
            elif self._same_input_round(record, parsed):
                human_receipt, human_preview_sha256 = self._inspect_human_question(
                    record,
                    parsed,
                )
            else:
                create_human = True
        return self._replace_state(
            record,
            parsed,
            status=target_status,
            human_receipt=human_receipt,
            human_preview_sha256=human_preview_sha256,
            create_human=create_human,
        )

    def _settle_mutation(
        self,
        claimed: McpRemoteTaskRecord,
        parsed: _ParsedTask,
        *,
        status: McpRemoteTaskRecordStatus,
    ) -> McpRemoteTask:
        return self._replace_state(
            claimed,
            parsed,
            status=status,
            human_receipt=None,
            human_preview_sha256=None,
            create_human=False,
        )

    def _replace_state(
        self,
        record: McpRemoteTaskRecord,
        parsed: _ParsedTask,
        *,
        status: McpRemoteTaskRecordStatus,
        human_receipt: McpHumanRequestReceipt | None,
        human_preview_sha256: str | None,
        create_human: bool,
    ) -> McpRemoteTask:
        previous_human_state = (
            self._load_state_for_human_cleanup(record)
            if record.human_request_id is not None
            and (
                human_receipt is None
                or record.human_request_id != human_receipt.request_id
            )
            else None
        )
        retire_human_request_id: str | None = None
        retire_human_preview_sha256: str | None = None
        if previous_human_state is not None and record.human_request_id is not None:
            _preview, retire_human_preview_sha256 = mcp_human_preview(
                server_id=record.server_id,
                operation="tasks/update",
                local_ref=record.task_ref,
                input_requests=previous_human_state.input_requests,
            )
            retire_human_request_id = record.human_request_id
        state_secret = canonical_json_bytes(
            parsed.raw_state,
            label="MCP remote Task broker state",
        )
        expires_at = self._bounded_task_expiry(
            parsed,
            existing=record.expires_at,
        )
        human_request_id, human_preview, human_preview_sha256 = (
            self._reserve_task_human(record, parsed)
            if create_human
            else (None, None, human_preview_sha256)
        )
        state_namespace = (
            f"mcp.remote-task.state.{record.task_ref}."
            f"r{record.revision + 1}"
        )
        retire_refs = tuple(
            sorted(
                reference
                for reference in (
                    record.result_ref,
                    record.broker_ref if parsed.status in _TERMINAL else None,
                )
                if reference is not None
            )
        )
        preparation = prepare_mcp_side_effects(
            repository=self.side_effects,
            broker=self.broker,
            operation_kind="remote_task",
            operation_id=record.task_ref,
            operation_revision=record.revision,
            server_id=record.server_id,
            server_spec_sha256=record.server_spec_sha256,
            server_generation=record.server_generation,
            owner_id=record.owner_id,
            auth_principal_sha256=record.auth_principal_sha256,
            auth_scope_sha256=record.auth_scope_sha256,
            human_request_id=human_request_id,
            human_preview_sha256=(
                human_preview_sha256 if create_human else None
            ),
            broker_namespace=None,
            broker_value_sha256=None,
            result_namespace=state_namespace,
            result_sha256=sha256(state_secret).hexdigest(),
            expires_at=expires_at,
            created_at=_timestamp(_utc(self._now())),
            retire_human_request_id=retire_human_request_id,
            retire_human_preview_sha256=retire_human_preview_sha256,
            retire_refs=retire_refs,
        )
        try:
            human_receipt = self._materialize_prepared_task_human(
                record=record,
                expires_at=expires_at,
                create_human=create_human,
                request_id=human_request_id,
                preview=human_preview,
                preview_sha256=human_preview_sha256,
                existing_receipt=human_receipt,
            )
            write_mcp_prepared_secrets(
                preparation,
                broker=self.broker,
                broker_namespace=None,
                broker_value=None,
                result_namespace=state_namespace,
                result_value=state_secret,
            )
            target = replace(
                record,
                broker_ref=(None if parsed.status in _TERMINAL else record.broker_ref),
                status=status,
                human_request_id=(
                    human_receipt.request_id if human_receipt is not None else None
                ),
                revision=record.revision + 1,
                expires_at=expires_at,
                poll_interval_ms=parsed.poll_interval_ms,
                status_message_sha256=(
                    sha256(parsed.status_message.encode("utf-8")).hexdigest()
                    if parsed.status_message is not None
                    else None
                ),
                result_ref=preparation.result_ref,
                result_sha256=sha256(state_secret).hexdigest(),
                metadata=_task_metadata("started", "reobserve_required", parsed),
                updated_at=_timestamp(_utc(self._now())),
            )
            commit_mcp_preparation(
                self.side_effects,
                preparation,
                target,
                broker=self.broker,
                human_requests=self.human_requests,
            )
        except Exception:
            cleanup_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason="task_settlement_failed",
            )
            raise
        public = self._public(target, parsed)
        if _status(target) in {
            McpRemoteTaskRecordStatus.COMPLETED,
            McpRemoteTaskRecordStatus.FAILED,
            McpRemoteTaskRecordStatus.CANCELLED,
            McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
        }:
            self._prune_terminal_records()
        return public

    def _reserve_task_human(
        self,
        record: McpRemoteTaskRecord,
        parsed: _ParsedTask,
    ) -> tuple[str, dict[str, JsonValue], str]:
        preview, preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation="tasks/update",
            local_ref=record.task_ref,
            input_requests=parsed.input_requests,
        )
        request_id = self.human_requests.reserve_question_id()
        _require_text(request_id, "MCP Human request id")
        return request_id, preview, preview_sha256

    def _materialize_prepared_task_human(
        self,
        *,
        record: McpRemoteTaskRecord,
        expires_at: str,
        create_human: bool,
        request_id: str | None,
        preview: dict[str, JsonValue] | None,
        preview_sha256: str | None,
        existing_receipt: McpHumanRequestReceipt | None,
    ) -> McpHumanRequestReceipt | None:
        if not create_human:
            return existing_receipt
        if request_id is None or preview is None or preview_sha256 is None:
            raise ValidationError("MCP Task Human preview is unavailable")
        receipt = require_human_receipt(
            self.human_requests.create_question(
                owner_id=record.owner_id,
                server_id=record.server_id,
                operation="tasks/update",
                local_ref=record.task_ref,
                preview=preview,
                preview_sha256=preview_sha256,
                expires_at=expires_at,
                request_id=request_id,
            ),
            preview_sha256=preview_sha256,
        )
        if receipt.request_id != request_id:
            raise ValidationError("MCP Human request reservation changed")
        return receipt

    def _bounded_task_expiry(
        self,
        parsed: _ParsedTask,
        *,
        existing: str | None = None,
    ) -> str:
        candidates = [_utc(self._now()) + self._max_wait]
        remote = _task_expiry(parsed.created_at, parsed.ttl_ms)
        if remote is not None:
            candidates.append(_parse_timestamp(remote))
        if existing is not None:
            candidates.append(_parse_timestamp(existing))
        return _timestamp(min(candidates))

    def _require_poll_ready(self, record: McpRemoteTaskRecord) -> None:
        if self._poll_min_interval_s == 0:
            return
        minimum = max(
            self._poll_min_interval_s,
            (
                record.poll_interval_ms / 1_000.0
                if record.poll_interval_ms is not None
                else 0.0
            ),
        )
        elapsed = (_utc(self._now()) - _parse_timestamp(record.updated_at)).total_seconds()
        if elapsed < minimum:
            raise ValidationError("MCP remote Task poll interval has not elapsed")

    def _same_input_round(
        self,
        record: McpRemoteTaskRecord,
        parsed: _ParsedTask,
    ) -> bool:
        if (
            _status(record) is not McpRemoteTaskRecordStatus.INPUT_REQUIRED
            or record.human_request_id is None
        ):
            return False
        previous = self._load_state_for_human_cleanup(record)
        return json_sha256(
            previous.raw_state.get("inputRequests", {}),
            label="MCP previous Task input requests",
        ) == json_sha256(
            parsed.raw_state.get("inputRequests", {}),
            label="MCP current Task input requests",
        )

    def _create_human_question(
        self,
        *,
        task_ref: str,
        parsed: _ParsedTask,
        expires_at: str | None,
        binding: McpRemoteTaskBinding | None = None,
        record: McpRemoteTaskRecord | None = None,
    ) -> tuple[McpHumanRequestReceipt, str]:
        if (binding is None) == (record is None):
            raise TypeError("MCP Task Human question requires one authority binding")
        server_id = binding.server_id if binding is not None else record.server_id
        owner_id = binding.owner_id if binding is not None else record.owner_id
        preview, preview_sha256 = mcp_human_preview(
            server_id=server_id,
            operation="tasks/update",
            local_ref=task_ref,
            input_requests=parsed.input_requests,
        )
        receipt = require_human_receipt(
            self.human_requests.create_question(
                owner_id=owner_id,
                server_id=server_id,
                operation="tasks/update",
                local_ref=task_ref,
                preview=preview,
                preview_sha256=preview_sha256,
                expires_at=expires_at,
            ),
            preview_sha256=preview_sha256,
        )
        return receipt, preview_sha256

    def _inspect_human_question(
        self,
        record: McpRemoteTaskRecord,
        parsed: _ParsedTask,
    ) -> tuple[McpHumanRequestReceipt, str]:
        if record.human_request_id is None:
            raise ValidationError("MCP input-required Task has no Human request binding")
        _preview, preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation="tasks/update",
            local_ref=record.task_ref,
            input_requests=parsed.input_requests,
        )
        receipt = require_human_receipt(
            self.human_requests.inspect_question(
                record.human_request_id,
                preview_sha256=preview_sha256,
            ),
            preview_sha256=preview_sha256,
        )
        return receipt, preview_sha256

    def _cancel_human_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
        reason: str,
    ) -> None:
        self.human_requests.cancel_question(
            request_id,
            preview_sha256=preview_sha256,
            reason=reason,
        )

    def _load_state_for_human_cleanup(
        self,
        record: McpRemoteTaskRecord,
    ) -> _ParsedTask:
        if not self.broker.available() or record.result_ref is None:
            raise ValidationError("MCP remote Task Human state is unavailable")
        raw = self.broker.get_secret(record.result_ref)
        if record.result_sha256 is None or sha256(raw).hexdigest() != record.result_sha256:
            raise ValidationError("MCP remote Task Human state integrity check failed")
        decoded = decode_broker_json(raw, label="MCP remote Task broker state")
        return self._parse_task(
            {"resultType": "complete", **decoded},
            expected_result_type="complete",
            allow_local_cancel_requested=True,
        )

    def _load_remote_id(self, record: McpRemoteTaskRecord) -> str:
        if not self.broker.available() or record.broker_ref is None:
            self._mark_attention(record, reason="broker_unavailable")
            raise ValidationError("MCP credential broker is unavailable")
        try:
            raw = self.broker.get_secret(record.broker_ref)
        except Exception as exc:
            self._mark_attention(record, reason="broker_missing")
            raise ValidationError("MCP credential broker value is unavailable") from exc
        if sha256(raw).hexdigest() != record.remote_id_sha256:
            self._mark_attention(record, reason="remote_id_integrity")
            raise ValidationError("MCP remote Task identity integrity check failed")
        try:
            remote_id = raw.decode("utf-8")
        except UnicodeError as exc:
            self._mark_attention(record, reason="remote_id_integrity")
            raise ValidationError("MCP remote Task identity is invalid") from exc
        if not remote_id or len(raw) > _MAX_REMOTE_ID_BYTES or "\x00" in remote_id:
            self._mark_attention(record, reason="remote_id_integrity")
            raise ValidationError("MCP remote Task identity is invalid")
        return remote_id

    def _read_remote_id_without_mutation(
        self,
        record: McpRemoteTaskRecord,
    ) -> str:
        """Read a bearer for notification correlation without recovery writes."""

        if not self.broker.available() or record.broker_ref is None:
            raise ValidationError("MCP credential broker is unavailable")
        try:
            raw = self.broker.get_secret(record.broker_ref)
        except Exception as exc:
            raise ValidationError("MCP credential broker value is unavailable") from exc
        if sha256(raw).hexdigest() != record.remote_id_sha256:
            raise ValidationError("MCP remote Task identity integrity check failed")
        try:
            remote_id = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ValidationError("MCP remote Task identity is invalid") from exc
        if not remote_id or len(raw) > _MAX_REMOTE_ID_BYTES or "\x00" in remote_id:
            raise ValidationError("MCP remote Task identity is invalid")
        return remote_id

    def _load_state(self, record: McpRemoteTaskRecord) -> _ParsedTask:
        if not self.broker.available() or record.result_ref is None:
            self._mark_attention(record, reason="state_unavailable")
            raise ValidationError("MCP credential broker is unavailable")
        try:
            raw = self.broker.get_secret(record.result_ref)
        except Exception as exc:
            self._mark_attention(record, reason="state_missing")
            raise ValidationError("MCP remote Task state is unavailable") from exc
        if record.result_sha256 is None or sha256(raw).hexdigest() != record.result_sha256:
            self._mark_attention(record, reason="state_integrity")
            raise ValidationError("MCP remote Task state integrity check failed")
        decoded = decode_broker_json(raw, label="MCP remote Task broker state")
        parsed = self._parse_task(
            {"resultType": "complete", **decoded},
            expected_result_type="complete",
            allow_local_cancel_requested=True,
        )
        expected_status = _public_status(_status(record))
        if (
            expected_status is not McpRemoteTaskStatus.NEEDS_ATTENTION
            and parsed.status is not expected_status
        ):
            self._mark_attention(record, reason="state_binding")
            raise ValidationError("MCP remote Task state binding changed")
        return parsed

    def _load(self, task_ref: str) -> McpRemoteTaskRecord:
        _require_text(task_ref, "MCP remote Task local reference")
        record = self.repository.get(task_ref)
        if record is None:
            raise NotFound("MCP remote Task not found")
        return record

    @staticmethod
    def _require_binding_type(binding: McpRemoteTaskBinding) -> None:
        if not isinstance(binding, McpRemoteTaskBinding):
            raise TypeError("MCP remote Task binding is required")

    def _require_binding(
        self,
        record: McpRemoteTaskRecord,
        binding: McpRemoteTaskBinding,
    ) -> None:
        self._require_binding_type(binding)
        binding.require_extension_pin()
        actual = (
            binding.server_id,
            binding.server_spec_sha256,
            binding.server_generation,
            binding.owner_id,
            binding.auth_principal_sha256,
            binding.auth_scope_sha256,
            binding.origin_request_sha256,
            binding.origin_effect_id,
        )
        expected = (
            record.server_id,
            record.server_spec_sha256,
            record.server_generation,
            record.owner_id,
            record.auth_principal_sha256,
            record.auth_scope_sha256,
            record.origin_request_sha256,
            record.origin_effect_id,
        )
        if actual != expected:
            raise CapabilityDenied("MCP remote Task binding changed")

    @staticmethod
    def _require_notification_fence(
        record: McpRemoteTaskRecord,
        fence: McpConnectionFence,
    ) -> None:
        principal_sha256 = fence.auth_principal_sha256 or sha256(b"null").hexdigest()
        scope_sha256 = fence.auth_scope_sha256 or sha256(b"[]").hexdigest()
        actual = (
            record.server_id,
            record.server_spec_sha256,
            record.server_generation,
            record.owner_id,
            record.auth_principal_sha256,
            record.auth_scope_sha256,
        )
        expected = (
            fence.server_id,
            fence.server_spec_sha256,
            fence.registry_generation,
            fence.owner,
            principal_sha256,
            scope_sha256,
        )
        if actual != expected:
            raise CapabilityDenied("MCP Task notification binding changed")

    @staticmethod
    def _validate_transition(
        current: McpRemoteTaskRecordStatus,
        target: McpRemoteTaskStatus,
    ) -> None:
        if _public_status(current) in _TERMINAL and target is not _public_status(current):
            raise ValidationError("MCP remote Task terminal state cannot change")

    def _public(self, record: McpRemoteTaskRecord, parsed: _ParsedTask) -> McpRemoteTask:
        human_receipt: McpHumanRequestReceipt | None = None
        human_preview_sha256: str | None = None
        if _status(record) is McpRemoteTaskRecordStatus.INPUT_REQUIRED:
            human_receipt, human_preview_sha256 = self._inspect_human_question(
                record,
                parsed,
            )
        return McpRemoteTask(
            task_ref=record.task_ref,
            status=_public_status(_status(record)),
            status_message=parsed.status_message,
            result=parsed.result,
            input_requests=parsed.input_requests,
            created_at=parsed.created_at,
            updated_at=parsed.updated_at,
            ttl_ms=parsed.ttl_ms,
            poll_interval_ms=parsed.poll_interval_ms,
            revision=record.revision,
            human_request_id=(
                human_receipt.request_id if human_receipt is not None else None
            ),
            human_revision=(
                human_receipt.revision if human_receipt is not None else None
            ),
            human_preview_sha256=human_preview_sha256,
        )

    def _recovery_human_preview(
        self,
        record: McpRemoteTaskRecord,
    ) -> str | None:
        if record.human_request_id is None:
            return None
        return self.human_requests.question_preview_sha256_for_recovery(
            record.human_request_id
        )

    @staticmethod
    def _retired_task_refs(
        record: McpRemoteTaskRecord,
        target: McpRemoteTaskRecord | None,
    ) -> tuple[str, ...]:
        current_refs = {
            reference
            for reference in (record.broker_ref, record.result_ref)
            if reference is not None
        }
        retained_refs = (
            {
                reference
                for reference in (target.broker_ref, target.result_ref)
                if reference is not None
            }
            if target is not None
            else set()
        )
        return tuple(sorted(current_refs - retained_refs))

    def _retirement_preparation(
        self,
        record: McpRemoteTaskRecord,
        *,
        target: McpRemoteTaskRecord | None,
    ) -> McpSideEffectPreparationRecord:
        retire_human = record.human_request_id is not None and (
            target is None
            or target.human_request_id != record.human_request_id
            or _status(target)
            in {
                McpRemoteTaskRecordStatus.COMPLETED,
                McpRemoteTaskRecordStatus.FAILED,
                McpRemoteTaskRecordStatus.CANCELLED,
                McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
            }
        )
        human_preview_sha256 = (
            self._recovery_human_preview(record) if retire_human else None
        )
        now = _timestamp(_utc(self._now()))
        return prepare_mcp_side_effects(
            repository=self.side_effects,
            broker=self.broker,
            operation_kind="remote_task",
            operation_id=record.task_ref,
            operation_revision=record.revision,
            server_id=record.server_id,
            server_spec_sha256=record.server_spec_sha256,
            server_generation=record.server_generation,
            owner_id=record.owner_id,
            auth_principal_sha256=record.auth_principal_sha256,
            auth_scope_sha256=record.auth_scope_sha256,
            human_request_id=None,
            human_preview_sha256=None,
            broker_namespace=None,
            broker_value_sha256=None,
            result_namespace=None,
            result_sha256=None,
            expires_at=record.expires_at or now,
            created_at=now,
            retire_refs=self._retired_task_refs(record, target),
            retire_human_request_id=(
                record.human_request_id if retire_human else None
            ),
            retire_human_preview_sha256=human_preview_sha256,
        )

    def _commit_retirement_transition(
        self,
        record: McpRemoteTaskRecord,
        target: McpRemoteTaskRecord,
        *,
        reason: str,
    ) -> None:
        if not self._has_task_side_effect_ownership(record):
            self._cas_or_conflict(record, target)
            return
        preparation = self._retirement_preparation(record, target=target)
        try:
            commit_mcp_preparation(
                self.side_effects,
                preparation,
                target,
                broker=self.broker,
                human_requests=self.human_requests,
            )
        except Exception:
            cleanup_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason=reason,
            )
            raise

    def _commit_terminal_retirement(
        self,
        record: McpRemoteTaskRecord,
        *,
        reason: str,
    ) -> None:
        if not self._has_task_side_effect_ownership(record):
            if not self.repository.delete_terminal(
                record.task_ref,
                expected_revision=record.revision,
            ):
                raise ValidationError("MCP remote Task retention revision conflict")
            return
        preparation = self._retirement_preparation(record, target=None)
        try:
            commit_terminal_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
            )
        except Exception:
            cleanup_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason=reason,
            )
            raise

    @staticmethod
    def _has_task_side_effect_ownership(record: McpRemoteTaskRecord) -> bool:
        return any(
            value is not None
            for value in (
                record.human_request_id,
                record.broker_ref,
                record.result_ref,
            )
        )

    def _mark_attention(self, record: McpRemoteTaskRecord, *, reason: str) -> None:
        current = self.repository.get(record.task_ref)
        if current is None or current.revision != record.revision:
            return
        if _status(current) in {
            McpRemoteTaskRecordStatus.COMPLETED,
            McpRemoteTaskRecordStatus.FAILED,
            McpRemoteTaskRecordStatus.CANCELLED,
            McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
        }:
            return
        target = replace(
            current,
            status=McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
            revision=current.revision + 1,
            broker_ref=None,
            result_ref=None,
            result_sha256=None,
            metadata=_plain_metadata("unknown", "unsafe_or_unknown", reason),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._commit_retirement_transition(
            current,
            target,
            reason=reason,
        )

    def _cas_or_conflict(
        self,
        expected: McpRemoteTaskRecord,
        target: McpRemoteTaskRecord,
    ) -> None:
        if not self.repository.compare_and_swap(
            expected.task_ref,
            expected_revision=expected.revision,
            replacement=target,
        ):
            raise ValidationError("MCP remote Task revision conflict")

    def _cas_or_unknown(
        self,
        expected: McpRemoteTaskRecord,
        target: McpRemoteTaskRecord,
    ) -> None:
        if not self.repository.compare_and_swap(
            expected.task_ref,
            expected_revision=expected.revision,
            replacement=target,
        ):
            self._mark_attention(expected, reason="settlement_conflict")
            raise ValidationError("MCP remote Task settlement is unknown")


def _raise_certified_not_started(error: McpRemoteTaskDispatchNotStarted) -> None:
    cause = error.__cause__
    if isinstance(cause, Exception):
        raise cause
    raise error


def _validated_task_result(
    result: Mapping[str, JsonValue],
    expected_result_type: str,
) -> dict[str, JsonValue]:
    if not isinstance(result, Mapping):
        raise ValidationError("MCP remote Task result is invalid")
    selected = dict(result)
    strict_json_value(selected, label="MCP remote Task result")
    if selected.get("resultType") != expected_result_type:
        raise ValidationError("MCP remote Task resultType is invalid")
    if set(selected) - _TASK_RESULT_FIELDS:
        raise ValidationError("MCP remote Task result contains unsupported fields")
    return selected


def _parse_task_notification(
    value: JsonValue,
    *,
    sensitive_values: tuple[str, ...],
) -> _ParsedTask:
    """Parse the TaskStatusNotification state shape without Task payloads."""

    if type(value) is not dict:
        raise ValidationError("MCP Task notification must carry an exact object")
    selected = dict(value)
    strict_json_value(selected, label="MCP Task notification")
    if set(selected) - _TASK_NOTIFICATION_FIELDS:
        raise ValidationError("MCP Task notification contains unsupported fields")
    remote_id = _validated_remote_task_id(
        selected.get("taskId"),
        sensitive_values,
    )
    content_sensitive = _merge_sensitive_values(sensitive_values, (remote_id,))
    status = _validated_remote_task_status(
        selected.get("status"),
        allow_local_cancel_requested=False,
    )
    created, updated, ttl, poll = _validated_task_timing(selected)
    status_message = _sanitized_task_status_message(
        selected.get("statusMessage"),
        content_sensitive,
    )
    raw_state = _sanitized_task_state(
        selected,
        sensitive_values=content_sensitive,
        normalized_requests=None,
        remote_id=remote_id,
    )
    return _ParsedTask(
        remote_id=remote_id,
        status=status,
        status_message=status_message,
        created_at=created,
        updated_at=updated,
        ttl_ms=ttl,
        poll_interval_ms=poll,
        result=None,
        input_requests=(),
        raw_state=raw_state,
    )


def _validate_remote_task_policy(
    *,
    max_input_requests: Any,
    poll_min_interval_s: Any,
    max_wait_s: Any,
    max_records: Any,
    terminal_records: Any,
    reconcile_on_start: Any,
) -> None:
    if type(reconcile_on_start) is not bool:
        raise ValidationError("MCP remote Task reconcile_on_start is invalid")
    if type(max_input_requests) is not int or not 1 <= max_input_requests <= _MAX_INPUT_REQUESTS:
        raise ValidationError("MCP remote Task input request limit is invalid")
    _validate_nonnegative_finite_policy(
        poll_min_interval_s,
        label="poll interval",
        allow_zero=True,
    )
    _validate_nonnegative_finite_policy(
        max_wait_s,
        label="maximum wait",
        allow_zero=False,
    )
    if type(max_records) is not int or max_records < 1:
        raise ValidationError("MCP remote Task record limit is invalid")
    if type(terminal_records) is not int or not 1 <= terminal_records <= 499:
        raise ValidationError("MCP remote Task terminal retention is invalid")


def _validate_nonnegative_finite_policy(
    value: Any,
    *,
    label: str,
    allow_zero: bool,
) -> None:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
        or (not allow_zero and value == 0)
    ):
        raise ValidationError(f"MCP remote Task {label} is invalid")


def _validated_remote_task_id(
    value: Any,
    sensitive_values: tuple[str, ...],
) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > _MAX_REMOTE_ID_BYTES
        or "\x00" in value
    ):
        raise ValidationError("MCP remote Task identity is invalid")
    return reject_opaque_secret_reflection(
        value,
        sensitive_values=sensitive_values,
        label="MCP remote Task identity",
    )


def _validated_remote_task_status(
    value: Any,
    *,
    allow_local_cancel_requested: bool,
) -> McpRemoteTaskStatus:
    try:
        status = McpRemoteTaskStatus(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("MCP remote Task status is invalid") from exc
    if status is McpRemoteTaskStatus.NEEDS_ATTENTION or (
        status is McpRemoteTaskStatus.CANCEL_REQUESTED
        and not allow_local_cancel_requested
    ):
        raise ValidationError("MCP Provider returned a local-only Task status")
    return status


def _validated_task_timing(
    selected: Mapping[str, JsonValue],
) -> tuple[str, str, int | None, int | None]:
    created = _timestamp(
        _parse_timestamp_value(selected.get("createdAt"), "createdAt")
    )
    updated = _timestamp(
        _parse_timestamp_value(selected.get("lastUpdatedAt"), "lastUpdatedAt")
    )
    if updated < created:
        raise ValidationError("MCP remote Task timestamps are inconsistent")
    ttl = selected.get("ttlMs")
    if ttl is not None and (type(ttl) is not int or ttl < 0 or ttl > _MAX_TTL_MS):
        raise ValidationError("MCP remote Task TTL is invalid")
    poll = selected.get("pollIntervalMs")
    if poll is not None and (
        type(poll) is not int or poll < 0 or poll > _MAX_POLL_INTERVAL_MS
    ):
        raise ValidationError("MCP remote Task poll interval is invalid")
    return created, updated, ttl, poll


def _sanitized_task_status_message(
    value: Any,
    sensitive_values: tuple[str, ...],
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > _MAX_STATUS_CHARS:
        raise ValidationError("MCP remote Task status message is invalid")
    selected = sanitize_provider_json(
        value,
        sensitive_values=sensitive_values,
        label="MCP remote Task status message",
    )
    if type(selected) is not str:  # pragma: no cover - scalar invariant
        raise ValidationError("MCP remote Task status message is invalid")
    return selected


def _validated_task_payload(
    selected: Mapping[str, JsonValue],
    *,
    status: McpRemoteTaskStatus,
    sensitive_values: tuple[str, ...],
    max_input_requests: int,
) -> tuple[
    tuple[McpInputRequest, ...],
    JsonValue | None,
    dict[str, JsonValue] | None,
]:
    payload_fields = set(selected) & {"inputRequests", "result", "error"}
    if status is McpRemoteTaskStatus.INPUT_REQUIRED:
        if payload_fields != {"inputRequests"}:
            raise ValidationError("MCP input-required Task result is invalid")
        requests = parse_input_requests(
            selected["inputRequests"],
            sensitive_values=sensitive_values,
            max_requests=max_input_requests,
        )
        if not requests.public:
            raise ValidationError("MCP input-required Task has no input requests")
        return requests.public, None, requests.raw
    if status is McpRemoteTaskStatus.COMPLETED:
        if payload_fields != {"result"}:
            raise ValidationError("MCP completed Task result is invalid")
        return (), sanitize_provider_json(
            selected["result"],
            sensitive_values=sensitive_values,
            label="MCP remote Task result",
        ), None
    if status is McpRemoteTaskStatus.FAILED:
        if payload_fields != {"error"}:
            raise ValidationError("MCP failed Task result is invalid")
        return (), {
            "error": sanitize_provider_json(
                selected["error"],
                sensitive_values=sensitive_values,
                label="MCP remote Task error",
            )
        }, None
    if payload_fields:
        raise ValidationError("MCP non-terminal Task carries invalid payload")
    return (), None, None


def _sanitized_task_state(
    selected: Mapping[str, JsonValue],
    *,
    sensitive_values: tuple[str, ...],
    normalized_requests: dict[str, JsonValue] | None,
    remote_id: str,
) -> dict[str, JsonValue]:
    siblings = {key: value for key, value in selected.items() if key != "taskId"}
    sanitized = sanitize_provider_json(
        siblings,
        sensitive_values=sensitive_values,
        label="MCP remote Task broker state",
    )
    if type(sanitized) is not dict:
        raise ValidationError("MCP remote Task broker state is invalid")
    if normalized_requests is not None:
        sanitized["inputRequests"] = normalized_requests
    sanitized["taskId"] = remote_id
    raw_state = {
        key: value
        for key, value in sanitized.items()
        if key not in {"resultType", "_meta"}
    }
    canonical_json_bytes(raw_state, label="MCP remote Task broker state")
    return raw_state


def _validate_empty_ack(value: Mapping[str, JsonValue], operation: str) -> None:
    if not isinstance(value, Mapping):
        raise ValidationError(f"MCP {operation} acknowledgement is invalid")
    selected = dict(value)
    strict_json_value(selected, label=f"MCP {operation} acknowledgement")
    if selected.get("resultType") != "complete" or set(selected) - {
        "resultType",
        "_meta",
    }:
        raise ValidationError(f"MCP {operation} acknowledgement is invalid")


def _has_unsupported_input(parsed: _ParsedTask) -> bool:
    return any(
        request.kind is not McpInputRequestKind.ELICITATION
        for request in parsed.input_requests
    )


def _require_human_response_fence(revision: Any, preview_sha256: Any) -> None:
    if type(revision) is not int or revision < 0:
        raise ValidationError("MCP Human expected revision is invalid")
    _require_sha256(preview_sha256, "Human preview")


def _merge_sensitive_values(*groups: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for group in groups:
        if type(group) is not tuple or any(
            type(item) is not str or not item for item in group
        ):
            raise ValidationError("MCP sensitive value snapshot is invalid")
        selected.extend(group)
    return tuple(dict.fromkeys(selected))


def _record_status(status: McpRemoteTaskStatus) -> McpRemoteTaskRecordStatus:
    try:
        return McpRemoteTaskRecordStatus(status.value)
    except ValueError as exc:
        raise ValidationError("MCP remote Task status cannot be persisted") from exc


def _status(record: McpRemoteTaskRecord) -> McpRemoteTaskRecordStatus:
    try:
        return McpRemoteTaskRecordStatus(record.status)
    except ValueError as exc:
        raise ValidationError("MCP remote Task durable status is invalid") from exc


def _public_status(status: McpRemoteTaskRecordStatus) -> McpRemoteTaskStatus:
    if status in {
        McpRemoteTaskRecordStatus.UPDATE_DISPATCHING,
        McpRemoteTaskRecordStatus.CANCEL_DISPATCHING,
        McpRemoteTaskRecordStatus.NEEDS_ATTENTION,
    }:
        return McpRemoteTaskStatus.NEEDS_ATTENTION
    return McpRemoteTaskStatus(status.value)


def _task_metadata(
    dispatch_state: str,
    retry_class: str,
    parsed: _ParsedTask,
) -> dict[str, Any]:
    selected = _plain_metadata(dispatch_state, retry_class, None)
    if parsed.input_requests:
        selected["input_schema_sha256"] = json_sha256(
            parsed.raw_state.get("inputRequests", {}),
            label="MCP remote Task input requests",
        )
    return selected


def _plain_metadata(
    dispatch_state: str,
    retry_class: str,
    reason: str | None,
) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "automatic_retry_disabled": True,
        "dispatch_state": dispatch_state,
        "retry_class": retry_class,
    }
    if reason is not None:
        selected["reason_code"] = reason
    return selected


def _task_expiry(created_at: str, ttl_ms: int | None) -> str | None:
    if ttl_ms is None:
        return None
    return _timestamp(_parse_timestamp(created_at) + timedelta(milliseconds=ttl_ms))


def _require_text(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValidationError(f"{label} is invalid")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValidationError(f"{label} is invalid")
    return value


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("MCP clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_timestamp_value(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ValidationError(f"MCP remote Task {label} is invalid")
    return _parse_timestamp(value)


def _parse_timestamp(value: str) -> datetime:
    if type(value) is not str or len(value) > 128:
        raise ValidationError("MCP remote Task timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("MCP remote Task timestamp is invalid") from exc
    return _utc(parsed)


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds")


def _deadline(value: float) -> float:
    if type(value) not in {int, float} or not value == value or value <= 0:
        raise ValidationError("MCP remote Task deadline is invalid")
    return float(value)


def _new_task_ref() -> str:
    import secrets

    return f"mcptask_{secrets.token_urlsafe(24)}"


__all__ = [
    "McpRemoteTaskBinding",
    "McpRemoteTaskBoundary",
    "McpRemoteTaskDispatchNotStarted",
    "McpRemoteTaskManager",
    "McpRemoteTaskRecord",
    "McpRemoteTaskRecordStatus",
    "McpRemoteTaskRepository",
    "McpSdkRemoteTaskCaptureAdapter",
]
