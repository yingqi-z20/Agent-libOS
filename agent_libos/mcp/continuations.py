"""Durable, exactly-once orchestration for MCP 2026-07-28 MRTR.

The manager never invokes an initial operation.  It accepts an
``input_required`` result produced by an already-governed operation and later
dispatches only through a Host-supplied continuation boundary.  That boundary
must re-run Capability, Human, data-flow, budget and pending-first effect checks
for every explicit response/cancel operation.

Although the wire protocol represents a continuation as the original method
plus ``inputResponses`` and byte-exact ``requestState``, this class never calls
the initial Tool API again.  A durable CAS claim fences the dedicated
continuation dispatch so a crash or ambiguous error cannot cause replay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
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
from agent_libos.mcp.types import (
    JsonValue,
    McpComplete,
    McpInputRequired,
    McpOperationResult,
    McpRemoteTask,
)
from agent_libos.models.base import StrEnum
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.storage.mcp_v7 import (
    McpContinuationRecord,
    McpSideEffectPreparationRecord,
)


_SHA_CHARS = frozenset("0123456789abcdef")
_DEFAULT_LIFETIME = timedelta(minutes=15)
_MAX_ROUNDS = 10
_MAX_INPUT_REQUESTS = 16
_MAX_REQUEST_STATE_BYTES = 65_536


class McpContinuationStatus(StrEnum):
    INPUT_REQUIRED = "input_required"
    DISPATCHING = "dispatching"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    NEEDS_ATTENTION = "needs_attention"


class McpContinuationDispatchNotStarted(ValidationError):
    """Trusted boundary signal that no Provider dispatch was attempted.

    Only the Host continuation boundary should raise this type.  A raw/custom
    MCP Provider is never caught directly by this manager, preventing an
    untrusted provider from spoofing a retry-safe local failure.
    """


@dataclass(frozen=True)
class McpContinuationBinding:
    server_id: str
    server_spec_sha256: str
    server_generation: int
    owner_id: str
    auth_principal_sha256: str
    auth_scope_sha256: str
    canonical_request: dict[str, JsonValue]
    effect_id: str
    capability_sha256: str
    data_flow_sha256: str
    _canonical_request_bytes: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("server id", self.server_id),
            ("owner id", self.owner_id),
            ("effect id", self.effect_id),
        ):
            _require_text(value, f"MCP continuation {label}")
        for label, value in (
            ("server spec", self.server_spec_sha256),
            ("auth principal", self.auth_principal_sha256),
            ("auth scope", self.auth_scope_sha256),
            ("Capability", self.capability_sha256),
            ("data-flow", self.data_flow_sha256),
        ):
            _require_sha256(value, f"MCP continuation {label}")
        if type(self.server_generation) is not int or self.server_generation < 0:
            raise ValidationError("MCP continuation server generation is invalid")
        if type(self.canonical_request) is not dict:
            raise ValidationError("MCP continuation canonical request must be an object")
        encoded = canonical_json_bytes(
            self.canonical_request,
            label="MCP continuation canonical request",
        )
        detached = decode_broker_json(
            encoded,
            label="MCP continuation canonical request",
        )
        object.__setattr__(self, "canonical_request", detached)
        object.__setattr__(self, "_canonical_request_bytes", encoded)
        if type(detached.get("method")) is not str or not detached["method"]:
            raise ValidationError("MCP continuation canonical request method is invalid")

    @property
    def request_sha256(self) -> str:
        return sha256(self._canonical_request_bytes).hexdigest()

    def detached_request(self) -> dict[str, JsonValue]:
        return decode_broker_json(
            self._canonical_request_bytes,
            label="MCP continuation canonical request",
        )


@runtime_checkable
class McpContinuationRepository(Protocol):
    def insert(self, record: McpContinuationRecord) -> None: ...

    def get(self, continuation_id: str) -> McpContinuationRecord | None: ...

    def list(self, **filters: object) -> list[McpContinuationRecord]: ...

    def count_active(self, *, owner_id: str | None = None) -> int: ...

    def list_terminal(
        self,
        *,
        owner_id: str | None = None,
        limit: int = 100,
    ) -> tuple[McpContinuationRecord, ...]: ...

    def delete_terminal(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
    ) -> bool: ...

    def compare_and_swap(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        replacement: McpContinuationRecord,
    ) -> bool: ...


@runtime_checkable
class McpContinuationBoundary(Protocol):
    """Governed continuation-only dispatch boundary.

    Implementations reauthorize and account the operation, create a pending
    effect before Provider I/O, and must translate only a locally proven
    pre-dispatch failure to :class:`McpContinuationDispatchNotStarted`.
    """

    async def continue_request(
        self,
        *,
        record: McpContinuationRecord,
        binding: McpContinuationBinding,
        original_request: dict[str, JsonValue],
        input_responses: dict[str, JsonValue],
        request_state: str | None,
        deadline: float,
        result_settler: "McpContinuationResultSettler",
    ) -> McpOperationResult[Any]: ...

    async def cancel_continuation(
        self,
        *,
        record: McpContinuationRecord,
        binding: McpContinuationBinding,
        deadline: float,
    ) -> None: ...


class McpRemoteTaskCapture(Protocol):
    def prepare_capture(
        self,
        binding: McpContinuationBinding,
        result: Mapping[str, JsonValue],
        *,
        origin_effect_id: str,
        sensitive_values: tuple[str, ...],
    ) -> tuple[McpRemoteTask, Any]: ...


class McpContinuationResultSettler(Protocol):
    """Prepare one Provider result for the current protected-effect commit."""

    def __call__(
        self,
        result: Mapping[str, JsonValue],
        effect_id: str,
    ) -> tuple[McpOperationResult[Any], Any]: ...


class McpContinuationCaptureBindingResolver(Protocol):
    def __call__(
        self,
        server_id: str,
        operation: str,
        logical_id: str,
    ) -> McpContinuationBinding: ...


class McpSdkContinuationCaptureAdapter:
    """Bridge ``McpSdkV2ResultAdapter`` to the durable manager.

    The resolver, not SDK/provider data, constructs the authority/effect/auth
    binding.  This adapter only detaches the official SDK payload and never
    dispatches an operation.
    """

    def __init__(
        self,
        manager: McpContinuationManager,
        binding_resolver: McpContinuationCaptureBindingResolver,
        *,
        expires_at: Callable[[McpContinuationBinding], str | None] | None = None,
    ) -> None:
        self.manager = manager
        self.binding_resolver = binding_resolver
        self._expires_at = expires_at or (lambda _binding: None)

    def capture_input_required(
        self,
        *,
        server_id: str,
        operation: str,
        logical_id: str,
        request_state: str | None,
        input_requests: Mapping[str, Any],
        deadline: float,
        sensitive_values: tuple[str, ...],
    ) -> McpInputRequired:
        _deadline(deadline)
        binding = self.binding_resolver(server_id, operation, logical_id)
        if binding.server_id != server_id:
            raise CapabilityDenied("MCP continuation resolver changed server binding")
        raw_requests = sdk_json_mapping(
            input_requests,
            label="MCP SDK inputRequests",
        )
        result: dict[str, JsonValue] = {
            "resultType": "input_required",
            "inputRequests": raw_requests,
        }
        if request_state is not None:
            if type(request_state) is not str:
                raise ValidationError("MCP SDK requestState is invalid")
            result["requestState"] = request_state
        return self.manager.prepare_initial_input_required(
            binding,
            result,
            expires_at=self._expires_at(binding),
            sensitive_values=sensitive_values,
        )


@dataclass
class _PendingInitialContinuationCapture:
    preparation: McpSideEffectPreparationRecord
    record: McpContinuationRecord
    public: McpInputRequired
    binding: McpContinuationBinding
    claimed: bool = False
    retirement: McpSideEffectPreparationRecord | None = None
    closed: bool = False


@dataclass(frozen=True)
class McpContinuationCaptureSettlement:
    """Host-only settlement token for one prepared initial MRTR result.

    ``commit_deferred`` performs only the RuntimeStore transition, so a caller
    may compose it with the originating protected effect in one outer
    transaction.  ``finalize`` must run only after that outer transaction has
    committed.  ``abort`` is the matching rollback path and inspects durable
    state rather than trusting an in-memory retirement receipt.
    """

    manager: "McpContinuationManager" = field(repr=False)
    capture: _PendingInitialContinuationCapture = field(repr=False)

    @property
    def continuation_id(self) -> str:
        return self.capture.record.continuation_id

    @property
    def effect_id(self) -> str:
        return self.capture.record.effect_id

    def commit_deferred(self) -> None:
        self.manager.commit_initial_capture_deferred(self)

    def finalize(self) -> None:
        self.manager.finalize_initial_capture(self)

    def abort(self, *, reason: str = "capture_failed") -> None:
        self.manager.abort_initial_capture(self, reason=reason)


@dataclass
class _PreparedContinuationResult:
    preparation: McpSideEffectPreparationRecord
    claimed: McpContinuationRecord
    target: McpContinuationRecord
    public: McpOperationResult[Any]
    response_effect_id: str
    task_settlement: Any | None = None
    retirement: McpSideEffectPreparationRecord | None = None
    closed: bool = False


@dataclass(frozen=True)
class McpContinuationResultSettlement:
    """Atomic continuation-result settlement owned by a protected effect.

    Provider I/O has already finished when this token is created.  The token
    owns only prepared Human/broker side effects until ``commit_deferred`` is
    called from the *current* continuation protected effect transaction.  A
    Task result composes its Task preparation in the same hook.
    """

    manager: "McpContinuationManager" = field(repr=False)
    prepared: _PreparedContinuationResult = field(repr=False)

    def commit_deferred(self) -> None:
        self.manager.commit_response_result_deferred(self)

    def finalize(self) -> None:
        self.manager.finalize_response_result(self)

    def abort(self, *, reason: str = "continuation_result_failed") -> None:
        self.manager.abort_response_result(self, reason=reason)

    def durable_receipt(self) -> dict[str, str]:
        public = self.prepared.public
        continuation_id = self.prepared.target.continuation_id
        if isinstance(public, McpInputRequired) and public.respondable:
            return {
                "kind": "input_required",
                "continuation_id": continuation_id,
            }
        if isinstance(public, McpRemoteTask):
            return {"kind": "remote_task", "task_ref": public.task_ref}
        return {}


class McpContinuationManager:
    def __init__(
        self,
        *,
        repository: McpContinuationRepository,
        side_effects: McpSideEffectRepository,
        broker: McpCredentialBroker,
        human_requests: McpHumanRequestBridge,
        boundary: McpContinuationBoundary,
        now: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
        sensitive_values: tuple[str, ...] = (),
        max_rounds: int = _MAX_ROUNDS,
        max_input_requests: int = _MAX_INPUT_REQUESTS,
        request_state_max_bytes: int = _MAX_REQUEST_STATE_BYTES,
        continuation_ttl_s: float = _DEFAULT_LIFETIME.total_seconds(),
        max_records: int = 1_000,
        terminal_records: int = 256,
        remote_task_capture: McpRemoteTaskCapture | None = None,
        reconcile_on_start: bool = True,
    ) -> None:
        if (
            repository is None
            or side_effects is None
            or broker is None
            or human_requests is None
            or boundary is None
        ):
            raise TypeError("MCP continuation dependencies are required")
        _validate_continuation_policy(
            max_rounds=max_rounds,
            max_input_requests=max_input_requests,
            request_state_max_bytes=request_state_max_bytes,
            continuation_ttl_s=continuation_ttl_s,
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
        self._id_factory = id_factory or _new_continuation_id
        self._sensitive_values = tuple(
            item for item in sensitive_values if type(item) is str and item
        )
        self._max_rounds = max_rounds
        self._max_input_requests = max_input_requests
        self._request_state_max_bytes = request_state_max_bytes
        self._continuation_ttl = timedelta(seconds=float(continuation_ttl_s))
        self._max_records = max_records
        self._terminal_records = terminal_records
        self._remote_task_capture = remote_task_capture
        self._pending_capture_lock = threading.RLock()
        self._pending_initial_captures: dict[
            str, _PendingInitialContinuationCapture
        ] = {}
        if reconcile_on_start:
            self.reconcile_after_restart()

    def capture_input_required(
        self,
        binding: McpContinuationBinding,
        result: Mapping[str, JsonValue],
        *,
        expires_at: str | None,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpInputRequired:
        """Eager compatibility wrapper for an initial MRTR result.

        New protected-operation integrations use
        :meth:`prepare_initial_input_required` and settle the capture inside the
        originating effect transaction.  This wrapper preserves the historical
        direct-manager contract for custom Host integrations.
        """

        public = self.prepare_initial_input_required(
            binding,
            result,
            expires_at=expires_at,
            sensitive_values=sensitive_values,
        )
        if not public.respondable or not public.continuation_id:
            return public
        settlement: McpContinuationCaptureSettlement | None = None
        try:
            settlement = self.claim_initial_capture(public, binding)
            settlement.commit_deferred()
            settlement.finalize()
        except Exception:
            if (
                settlement is not None
                and self.repository.get(public.continuation_id)
                == settlement.capture.record
            ):
                # The public local ref is recoverable once the exact main row
                # exists.  A post-commit cleanup failure leaves a durable
                # ``cleaning`` receipt for restart and must not turn success
                # into an ambiguous Provider-visible failure.
                with self._pending_capture_lock:
                    self._close_initial_capture(settlement.capture)
                return public
            try:
                if settlement is None:
                    self.abort_prepared_effect(
                        binding.effect_id,
                        reason="capture_failed",
                    )
                else:
                    settlement.abort(reason="capture_failed")
            except Exception:
                # The durable prepared/cleaning row remains the restart cleanup
                # receipt when an external cleanup backend is unavailable.
                pass
            raise
        return public

    def prepare_initial_input_required(
        self,
        binding: McpContinuationBinding,
        result: Mapping[str, JsonValue],
        *,
        expires_at: str | None,
        sensitive_values: tuple[str, ...] = (),
    ) -> McpInputRequired:
        """Prepare, but do not commit, an initial ``input_required`` result.

        Human and broker slots are owned first by a durable side-effect
        preparation.  The Continuation main row is intentionally absent until
        a trusted caller claims the exact public result and commits its
        settlement from the originating protected-effect transaction.
        """

        if not isinstance(binding, McpContinuationBinding):
            raise TypeError("MCP continuation binding is required")
        operation_sensitive = _merge_sensitive_values(
            self._sensitive_values,
            sensitive_values,
        )
        envelope, public, respondable = self._parse_input_required(
            result,
            round_number=1,
            sensitive_values=operation_sensitive,
        )
        if not respondable:
            return replace(public, respondable=False)
        with self._pending_capture_lock:
            self._reconcile_expired(limit=500)
            self._prune_terminal_records()
            if (
                self.repository.count_active()
                + len(self._pending_initial_captures)
                >= self._max_records
            ):
                raise ValidationError("MCP continuation record limit is exhausted")
            now = _utc(self._now())
            envelope["originalRequest"] = binding.detached_request()
            envelope["resultEffectId"] = binding.effect_id
            canonical_json_bytes(envelope, label="MCP continuation broker envelope")
            host_expiry = now + self._continuation_ttl
            expires = (
                min(_parse_timestamp(expires_at), host_expiry)
                if expires_at is not None
                else host_expiry
            )
            continuation_id = self._id_factory()
            _require_text(continuation_id, "MCP continuation id")
            if (
                continuation_id in self._pending_initial_captures
                or self.repository.get(continuation_id) is not None
            ):
                raise ValidationError("MCP continuation id is already in use")
            if any(
                pending.record.effect_id == binding.effect_id
                and not pending.closed
                for pending in self._pending_initial_captures.values()
            ):
                raise ValidationError("MCP effect already owns a prepared continuation")
            selected_expiry = _timestamp(expires)
            preview, preview_sha256 = mcp_human_preview(
                server_id=binding.server_id,
                operation=self._binding_operation(binding),
                local_ref=continuation_id,
                input_requests=public.input_requests,
            )
            human_request_id = self.human_requests.reserve_question_id()
            _require_text(human_request_id, "MCP Human request id")
            secret = canonical_json_bytes(
                envelope,
                label="MCP continuation broker envelope",
            )
            secret_sha256 = sha256(secret).hexdigest()
            broker_namespace = f"mcp.continuation.{continuation_id}.r0"
            preparation = prepare_mcp_side_effects(
                repository=self.side_effects,
                broker=self.broker,
                operation_kind="continuation",
                operation_id=continuation_id,
                operation_revision=None,
                server_id=binding.server_id,
                server_spec_sha256=binding.server_spec_sha256,
                server_generation=binding.server_generation,
                owner_id=binding.owner_id,
                auth_principal_sha256=binding.auth_principal_sha256,
                auth_scope_sha256=binding.auth_scope_sha256,
                human_request_id=human_request_id,
                human_preview_sha256=preview_sha256,
                broker_namespace=broker_namespace,
                broker_value_sha256=secret_sha256,
                result_namespace=None,
                result_sha256=None,
                expires_at=selected_expiry,
                created_at=_timestamp(now),
            )
            try:
                human_receipt = require_human_receipt(
                    self.human_requests.create_question(
                        owner_id=binding.owner_id,
                        server_id=binding.server_id,
                        operation=self._binding_operation(binding),
                        local_ref=continuation_id,
                        preview=preview,
                        preview_sha256=preview_sha256,
                        expires_at=selected_expiry,
                        request_id=human_request_id,
                    ),
                    preview_sha256=preview_sha256,
                )
                if human_receipt.request_id != human_request_id:
                    raise ValidationError("MCP Human request reservation changed")
                write_mcp_prepared_secrets(
                    preparation,
                    broker=self.broker,
                    broker_namespace=broker_namespace,
                    broker_value=secret,
                    result_namespace=None,
                    result_value=None,
                )
                record = McpContinuationRecord(
                    continuation_id=continuation_id,
                    server_id=binding.server_id,
                    server_spec_sha256=binding.server_spec_sha256,
                    server_generation=binding.server_generation,
                    owner_id=binding.owner_id,
                    auth_principal_sha256=binding.auth_principal_sha256,
                    auth_scope_sha256=binding.auth_scope_sha256,
                    request_sha256=binding.request_sha256,
                    effect_id=binding.effect_id,
                    capability_sha256=binding.capability_sha256,
                    data_flow_sha256=binding.data_flow_sha256,
                    human_request_id=human_receipt.request_id,
                    broker_ref=preparation.broker_ref,
                    broker_value_sha256=secret_sha256,
                    status=McpContinuationStatus.INPUT_REQUIRED,
                    revision=0,
                    expires_at=selected_expiry,
                    metadata=_metadata(
                        "not_started",
                        "reobserve_required",
                        envelope,
                    ),
                    created_at=_timestamp(now),
                    updated_at=_timestamp(now),
                )
                captured_public = replace(
                    public,
                    continuation_id=continuation_id,
                    expires_at=record.expires_at,
                    revision=record.revision,
                    respondable=True,
                    human_request_id=human_receipt.request_id,
                    human_revision=human_receipt.revision,
                    human_preview_sha256=preview_sha256,
                )
                pending = _PendingInitialContinuationCapture(
                    preparation=preparation,
                    record=record,
                    public=deepcopy(captured_public),
                    binding=_detached_binding(binding),
                )
                self._validate_initial_capture_snapshot(pending)
                self._pending_initial_captures[continuation_id] = pending
            except Exception:
                cleanup_mcp_preparation(
                    self.side_effects,
                    preparation,
                    broker=self.broker,
                    human_requests=self.human_requests,
                    updated_at=_timestamp(_utc(self._now())),
                    reason="capture_failed",
                )
                raise
            return deepcopy(captured_public)

    def claim_initial_capture(
        self,
        public: McpInputRequired,
        binding: McpContinuationBinding,
    ) -> McpContinuationCaptureSettlement:
        """Claim the exact prepared public result for protected settlement."""

        if type(public) is not McpInputRequired:
            raise TypeError("MCP prepared continuation result is required")
        if type(binding) is not McpContinuationBinding:
            raise TypeError("MCP continuation binding is required")
        continuation_id = _require_text(
            public.continuation_id,
            "MCP continuation id",
        )
        with self._pending_capture_lock:
            capture = self._pending_initial_captures.get(continuation_id)
            if capture is None or capture.closed:
                raise ValidationError("MCP initial continuation capture is unavailable")
            if capture.claimed:
                raise ValidationError("MCP initial continuation capture is already claimed")
            self._validate_initial_capture_snapshot(
                capture,
                supplied_public=public,
                supplied_binding=binding,
            )
            capture.claimed = True
            return McpContinuationCaptureSettlement(self, capture)

    def commit_initial_capture_deferred(
        self,
        settlement: McpContinuationCaptureSettlement,
    ) -> None:
        """Commit only the prepared Continuation/sidecar database transition."""

        with self._pending_capture_lock:
            capture = self._capture_for_settlement(settlement)
            if not capture.claimed:
                raise ValidationError("MCP initial continuation capture is not claimed")
            if capture.retirement is not None:
                current = self.side_effects.get(
                    capture.preparation.preparation_id
                )
                if current == capture.retirement:
                    return
                raise ValidationError(
                    "MCP initial continuation deferred state changed"
                )
            self._validate_initial_capture_snapshot(capture)
            retirement = commit_mcp_preparation_deferred(
                self.side_effects,
                capture.preparation,
                capture.record,
            )
            if self.repository.get(capture.record.continuation_id) != capture.record:
                raise ValidationError("MCP initial continuation commit changed")
            if self.side_effects.get(retirement.preparation_id) != retirement:
                raise ValidationError("MCP initial continuation retirement changed")
            capture.retirement = retirement

    def finalize_initial_capture(
        self,
        settlement: McpContinuationCaptureSettlement,
    ) -> None:
        """Finalize external cleanup after the outer transaction commits."""

        with self._pending_capture_lock:
            capture = self._capture_for_settlement(settlement)
            if not capture.claimed or capture.retirement is None:
                raise ValidationError(
                    "MCP initial continuation was not committed for finalization"
                )
            if self.repository.get(capture.record.continuation_id) != capture.record:
                raise ValidationError(
                    "MCP initial continuation commit is not durable"
                )
            current = self.side_effects.get(capture.preparation.preparation_id)
            if current != capture.retirement:
                raise ValidationError(
                    "MCP initial continuation retirement receipt changed"
                )
            finalize_mcp_preparation(
                self.side_effects,
                capture.retirement,
                broker=self.broker,
                human_requests=self.human_requests,
            )
            self._close_initial_capture(capture)

    def abort_initial_capture(
        self,
        settlement: McpContinuationCaptureSettlement,
        *,
        reason: str = "capture_failed",
    ) -> None:
        """Abort a rolled-back preparation or finish an already-committed one."""

        _require_text(reason, "MCP continuation abort reason")
        with self._pending_capture_lock:
            capture = self._capture_for_settlement(
                settlement,
                allow_closed=True,
            )
            if capture.closed:
                return
            self._abort_initial_capture_locked(capture, reason=reason)

    def abort_prepared_effect(
        self,
        effect_id: str,
        *,
        reason: str = "origin_effect_failed",
    ) -> int:
        """Abort in-memory initial captures belonging to one exact effect."""

        selected_effect = _require_text(effect_id, "MCP continuation effect id")
        _require_text(reason, "MCP continuation abort reason")
        with self._pending_capture_lock:
            captures = tuple(
                capture
                for capture in self._pending_initial_captures.values()
                if not capture.closed and capture.record.effect_id == selected_effect
            )
            for capture in captures:
                self._abort_initial_capture_locked(capture, reason=reason)
            return len(captures)

    def _capture_for_settlement(
        self,
        settlement: McpContinuationCaptureSettlement,
        *,
        allow_closed: bool = False,
    ) -> _PendingInitialContinuationCapture:
        if (
            type(settlement) is not McpContinuationCaptureSettlement
            or settlement.manager is not self
        ):
            raise TypeError("MCP continuation capture settlement is invalid")
        capture = settlement.capture
        current = self._pending_initial_captures.get(
            capture.record.continuation_id
        )
        if current is not capture:
            if allow_closed and capture.closed:
                return capture
            raise ValidationError("MCP initial continuation capture is unavailable")
        if capture.closed and not allow_closed:
            raise ValidationError("MCP initial continuation capture is closed")
        return capture

    def _abort_initial_capture_locked(
        self,
        capture: _PendingInitialContinuationCapture,
        *,
        reason: str,
    ) -> None:
        current_sidecar = self.side_effects.get(
            capture.preparation.preparation_id
        )
        current_record = self.repository.get(capture.record.continuation_id)
        if current_sidecar is None:
            if current_record is not None and current_record != capture.record:
                raise ValidationError("MCP initial continuation state changed")
            self._close_initial_capture(capture)
            return
        if current_sidecar.status == "prepared":
            if current_record is not None:
                raise ValidationError(
                    "MCP prepared continuation unexpectedly has a main row"
                )
            cleanup_mcp_preparation(
                self.side_effects,
                capture.preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason=reason,
            )
            self._close_initial_capture(capture)
            return
        if current_sidecar.status == "cleaning":
            if current_record != capture.record:
                raise ValidationError(
                    "MCP committed continuation state is unavailable"
                )
            finalize_mcp_preparation(
                self.side_effects,
                current_sidecar,
                broker=self.broker,
                human_requests=self.human_requests,
            )
            self._close_initial_capture(capture)
            return
        raise ValidationError("MCP initial continuation sidecar state is invalid")

    def _close_initial_capture(
        self,
        capture: _PendingInitialContinuationCapture,
    ) -> None:
        current = self._pending_initial_captures.get(
            capture.record.continuation_id
        )
        if current is capture:
            del self._pending_initial_captures[capture.record.continuation_id]
        elif current is not None:
            raise ValidationError("MCP initial continuation capture changed")
        capture.closed = True

    def _validate_initial_capture_snapshot(
        self,
        capture: _PendingInitialContinuationCapture,
        *,
        supplied_public: McpInputRequired | None = None,
        supplied_binding: McpContinuationBinding | None = None,
    ) -> None:
        public = capture.public if supplied_public is None else supplied_public
        binding = capture.binding if supplied_binding is None else supplied_binding
        self._validate_initial_capture_projection(capture, public, binding)
        self._validate_initial_capture_sidecar(capture, public)
        envelope = self._validate_initial_capture_broker(capture, public)
        if dict(capture.record.metadata) != _metadata(
            "not_started",
            "reobserve_required",
            envelope,
        ):
            raise ValidationError("MCP initial continuation metadata changed")
        self._validate_initial_capture_human(capture, public)

    def _validate_initial_capture_projection(
        self,
        capture: _PendingInitialContinuationCapture,
        public: McpInputRequired,
        binding: McpContinuationBinding,
    ) -> None:
        record = capture.record
        if type(public) is not McpInputRequired or public != capture.public:
            raise CapabilityDenied("MCP prepared continuation result changed")
        if type(binding) is not McpContinuationBinding or binding != capture.binding:
            raise CapabilityDenied("MCP prepared continuation binding changed")
        self._require_binding(record, binding)
        if (
            _status(record) is not McpContinuationStatus.INPUT_REQUIRED
            or record.revision != 0
            or record.continuation_id != public.continuation_id
            or record.expires_at != public.expires_at
            or record.human_request_id != public.human_request_id
            or public.revision != 0
            or not public.respondable
            or public.human_revision is None
            or public.human_preview_sha256 is None
        ):
            raise ValidationError("MCP prepared continuation projection changed")

    def _validate_initial_capture_sidecar(
        self,
        capture: _PendingInitialContinuationCapture,
        public: McpInputRequired,
    ) -> None:
        record = capture.record
        preparation = capture.preparation
        expected_sidecar = (
            "continuation",
            record.continuation_id,
            None,
            record.server_id,
            record.server_spec_sha256,
            record.server_generation,
            record.owner_id,
            record.auth_principal_sha256,
            record.auth_scope_sha256,
            record.human_request_id,
            public.human_preview_sha256,
            record.broker_ref,
            record.broker_value_sha256,
            None,
            None,
            "prepared",
            0,
            record.expires_at,
        )
        actual_sidecar = (
            preparation.operation_kind,
            preparation.operation_id,
            preparation.operation_revision,
            preparation.server_id,
            preparation.server_spec_sha256,
            preparation.server_generation,
            preparation.owner_id,
            preparation.auth_principal_sha256,
            preparation.auth_scope_sha256,
            preparation.human_request_id,
            preparation.human_preview_sha256,
            preparation.broker_ref,
            preparation.broker_value_sha256,
            preparation.result_ref,
            preparation.result_sha256,
            preparation.status,
            preparation.revision,
            preparation.expires_at,
        )
        if actual_sidecar != expected_sidecar or dict(preparation.metadata) != {
            "automatic_retry_disabled": True,
            "cleanup_mode": "abort",
            "retire_refs": (),
        }:
            raise ValidationError("MCP initial continuation sidecar changed")
        if self.side_effects.get(preparation.preparation_id) != preparation:
            raise ValidationError("MCP initial continuation sidecar is unavailable")
        if self.repository.get(record.continuation_id) is not None:
            raise ValidationError("MCP initial continuation was already committed")

    def _validate_initial_capture_broker(
        self,
        capture: _PendingInitialContinuationCapture,
        public: McpInputRequired,
    ) -> dict[str, JsonValue]:
        record = capture.record
        if record.broker_ref is None or record.broker_value_sha256 is None:
            raise ValidationError("MCP initial continuation broker binding is missing")
        try:
            secret = self.broker.get_secret(record.broker_ref)
        except Exception:
            raise ValidationError(
                "MCP initial continuation broker value is unavailable"
            ) from None
        if (
            not isinstance(secret, bytes)
            or sha256(secret).hexdigest() != record.broker_value_sha256
        ):
            raise ValidationError("MCP initial continuation broker value changed")
        envelope = decode_broker_json(
            secret,
            label="MCP continuation broker envelope",
        )
        original_request = envelope.get("originalRequest")
        if type(original_request) is not dict or json_sha256(
            original_request,
            label="MCP continuation original request",
        ) != record.request_sha256:
            raise ValidationError("MCP initial continuation request binding changed")
        raw_requests = envelope.get("inputRequests")
        parsed = parse_input_requests(
            raw_requests,
            sensitive_values=(),
            max_requests=self._max_input_requests,
        )
        if parsed.has_unsupported or parsed.public != public.input_requests:
            raise ValidationError("MCP initial continuation input projection changed")
        return envelope

    def _validate_initial_capture_human(
        self,
        capture: _PendingInitialContinuationCapture,
        public: McpInputRequired,
    ) -> None:
        record = capture.record
        preview_sha256 = public.human_preview_sha256
        if preview_sha256 is None:
            raise ValidationError("MCP initial continuation Human preview is missing")
        try:
            human_receipt = require_human_receipt(
                self.human_requests.inspect_question(
                    record.human_request_id,
                    preview_sha256=preview_sha256,
                ),
                preview_sha256=preview_sha256,
            )
        except Exception:
            raise ValidationError(
                "MCP initial continuation Human binding is unavailable"
            ) from None
        if (
            human_receipt.request_id != public.human_request_id
            or human_receipt.revision != public.human_revision
        ):
            raise ValidationError("MCP initial continuation Human binding changed")

    def get(
        self,
        continuation_id: str,
        *,
        binding: McpContinuationBinding,
    ) -> McpInputRequired:
        record = self._load(continuation_id)
        self._require_binding(record, binding)
        self._require_status(record, McpContinuationStatus.INPUT_REQUIRED)
        if _utc(self._now()) >= _parse_timestamp(record.expires_at):
            self._expire(record, binding)
        envelope = self._load_envelope(record)
        parsed = parse_input_requests(
            envelope["inputRequests"],
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        preview, preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation=self._binding_operation(binding),
            local_ref=record.continuation_id,
            input_requests=parsed.public,
        )
        del preview
        receipt = require_human_receipt(
            self.human_requests.inspect_question(
                record.human_request_id,
                preview_sha256=preview_sha256,
            ),
            preview_sha256=preview_sha256,
        )
        return McpInputRequired(
            continuation_id=record.continuation_id,
            input_requests=parsed.public,
            expires_at=record.expires_at,
            revision=record.revision,
            respondable=True,
            human_request_id=receipt.request_id,
            human_revision=receipt.revision,
            human_preview_sha256=preview_sha256,
        )

    def binding_material(self, continuation_id: str) -> McpContinuationBinding:
        """Recover the broker-held original operation binding for Host dispatch.

        The public local reference is sufficient for a Host facade after a
        restart.  The canonical original request is never stored in the
        RuntimeStore; it is read from the credential broker and its canonical
        hash is checked against the durable fence before it is returned.
        """

        record = self._load(continuation_id)
        envelope = self._load_envelope(record)
        original_request = self._original_request(record, envelope)
        return McpContinuationBinding(
            server_id=record.server_id,
            server_spec_sha256=record.server_spec_sha256,
            server_generation=record.server_generation,
            owner_id=record.owner_id,
            auth_principal_sha256=record.auth_principal_sha256,
            auth_scope_sha256=record.auth_scope_sha256,
            canonical_request=original_request,
            effect_id=record.effect_id,
            capability_sha256=record.capability_sha256,
            data_flow_sha256=record.data_flow_sha256,
        )

    def prevalidate_response(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        binding: McpContinuationBinding,
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        responses: Mapping[str, JsonValue],
    ) -> None:
        """Validate one Host answer without consuming its Human request."""

        record = self._prepare_pending(
            continuation_id,
            expected_revision=expected_revision,
            binding=binding,
        )
        if human_request_id != record.human_request_id:
            raise CapabilityDenied("MCP Human request belongs to another continuation")
        envelope = self._load_envelope(record)
        parsed = parse_input_requests(
            envelope["inputRequests"],
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        _preview, expected_preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation=self._binding_operation(binding),
            local_ref=record.continuation_id,
            input_requests=parsed.public,
        )
        if human_preview_sha256 != expected_preview_sha256:
            raise CapabilityDenied("MCP Human response preview binding changed")
        _require_human_response_fence(
            human_expected_revision,
            human_preview_sha256,
        )
        validate_input_responses(parsed, responses)

    async def respond(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        binding: McpContinuationBinding,
        human_request_id: str,
        human_expected_revision: int,
        human_preview_sha256: str,
        deadline: float,
    ) -> McpOperationResult[Any]:
        selected_deadline = _deadline(deadline)
        record = self._prepare_pending(
            continuation_id,
            expected_revision=expected_revision,
            binding=binding,
        )
        if human_request_id != record.human_request_id:
            raise CapabilityDenied("MCP Human request belongs to another continuation")
        envelope = self._load_envelope(record)
        parsed = parse_input_requests(
            envelope["inputRequests"],
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        _preview, expected_preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation=self._binding_operation(binding),
            local_ref=record.continuation_id,
            input_requests=parsed.public,
        )
        if human_preview_sha256 != expected_preview_sha256:
            raise CapabilityDenied("MCP Human response preview binding changed")
        _require_human_response_fence(
            human_expected_revision,
            human_preview_sha256,
        )
        responses = self.human_requests.consume_approved_answer(
            record.human_request_id,
            presented_revision=human_expected_revision,
            preview_sha256=expected_preview_sha256,
        )
        input_responses = validate_input_responses(parsed, responses)
        request_state = envelope.get("requestState")
        if request_state is not None and type(request_state) is not str:
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation broker payload is invalid")
        round_number = envelope.get("round")
        if type(round_number) is not int or not (1 <= round_number <= self._max_rounds):
            self._mark_attention(record, reason="round_integrity")
            raise ValidationError("MCP continuation round binding is invalid")
        claimed = replace(
            record,
            status=McpContinuationStatus.DISPATCHING,
            revision=record.revision + 1,
            metadata=_metadata("not_started", "unsafe_or_unknown", envelope),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._cas_or_conflict(record, claimed)
        dispatch_sensitive = _merge_sensitive_values(
            self._sensitive_values,
            (request_state,) if request_state else (),
        )

        def settle_result(
            raw_result: Mapping[str, JsonValue],
            response_effect_id: str,
        ) -> tuple[McpOperationResult[Any], McpContinuationResultSettlement]:
            return self.prepare_response_result(
                claimed,
                binding,
                raw_result,
                previous_broker_ref=record.broker_ref,
                next_round=round_number + 1,
                sensitive_values=dispatch_sensitive,
                retire_human_preview_sha256=expected_preview_sha256,
                response_effect_id=response_effect_id,
            )

        try:
            result = await self.boundary.continue_request(
                record=claimed,
                binding=binding,
                original_request=binding.detached_request(),
                input_responses=input_responses,
                request_state=request_state,
                deadline=selected_deadline,
                result_settler=settle_result,
            )
        except McpContinuationDispatchNotStarted as exc:
            restored = replace(
                claimed,
                status=McpContinuationStatus.INPUT_REQUIRED,
                revision=claimed.revision + 1,
                metadata=_metadata("not_started", "reobserve_required", envelope),
                updated_at=_timestamp(_utc(self._now())),
            )
            self._cas_or_unknown(claimed, restored)
            _raise_certified_not_started(exc)
        except Exception:
            self._mark_attention(claimed, reason="dispatch_unknown")
            raise ValidationError(
                "MCP continuation dispatch failed with an unknown outcome"
            ) from None
        if not isinstance(result, (McpComplete, McpInputRequired, McpRemoteTask)):
            self._mark_attention(claimed, reason="settlement_invalid")
            raise ValidationError("MCP continuation boundary result is invalid")
        return result

    async def cancel(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        binding: McpContinuationBinding,
        deadline: float,
    ) -> McpComplete[None]:
        selected_deadline = _deadline(deadline)
        record = self._prepare_pending(
            continuation_id,
            expected_revision=expected_revision,
            binding=binding,
        )
        envelope = self._load_envelope(record)
        parsed = parse_input_requests(
            envelope["inputRequests"],
            sensitive_values=self._sensitive_values,
            max_requests=self._max_input_requests,
        )
        _preview, preview_sha256 = mcp_human_preview(
            server_id=record.server_id,
            operation=self._binding_operation(binding),
            local_ref=record.continuation_id,
            input_requests=parsed.public,
        )
        claimed = replace(
            record,
            status=McpContinuationStatus.DISPATCHING,
            revision=record.revision + 1,
            metadata=_metadata("not_started", "unsafe_or_unknown", None),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._cas_or_conflict(record, claimed)
        try:
            await self.boundary.cancel_continuation(
                record=claimed,
                binding=binding,
                deadline=selected_deadline,
            )
        except McpContinuationDispatchNotStarted as exc:
            restored = replace(
                claimed,
                status=McpContinuationStatus.INPUT_REQUIRED,
                revision=claimed.revision + 1,
                metadata=_metadata("not_started", "reobserve_required", None),
                updated_at=_timestamp(_utc(self._now())),
            )
            self._cas_or_unknown(claimed, restored)
            _raise_certified_not_started(exc)
        except Exception:
            self._mark_attention(claimed, reason="cancel_unknown")
            raise
        terminal = replace(
            claimed,
            status=McpContinuationStatus.CANCELLED,
            revision=claimed.revision + 1,
            broker_ref=None,
            broker_value_sha256=None,
            metadata=_metadata("started", "unsafe_or_unknown", None),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._commit_retirement_transition(
            claimed,
            terminal,
            human_preview_sha256=preview_sha256,
            reason="continuation_cancelled",
        )
        self._prune_terminal_records()
        return McpComplete(value=None)

    def reconcile_after_restart(self) -> int:
        """Fence crash-interrupted dispatches; never continue them automatically."""

        changed = reconcile_mcp_preparations(
            self.side_effects,
            operation_kind="continuation",
            broker=self.broker,
            human_requests=self.human_requests,
            updated_at=_timestamp(_utc(self._now())),
        )
        changed += self._reconcile_expired(limit=500)
        while True:
            batch = self.repository.list(
                status=McpContinuationStatus.DISPATCHING.value,
                limit=500,
            )
            candidates = [
                record
                for record in batch
                if _status(record) is McpContinuationStatus.DISPATCHING
            ]
            if not candidates:
                break
            changed_this_round = 0
            for record in candidates:
                target = replace(
                    record,
                    status=McpContinuationStatus.NEEDS_ATTENTION,
                    revision=record.revision + 1,
                    broker_ref=None,
                    broker_value_sha256=None,
                    metadata=_metadata("unknown", "unsafe_or_unknown", None),
                    updated_at=_timestamp(_utc(self._now())),
                )
                self._commit_retirement_transition(
                    record,
                    target,
                    human_preview_sha256=self._recovery_human_preview(record),
                    reason="continuation_dispatch_interrupted",
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
        for record in self.repository.list(
            status=McpContinuationStatus.INPUT_REQUIRED.value,
            limit=limit,
        ):
            if (
                _status(record) is not McpContinuationStatus.INPUT_REQUIRED
                or now < _parse_timestamp(record.expires_at)
            ):
                continue
            target = replace(
                record,
                status=McpContinuationStatus.EXPIRED,
                revision=record.revision + 1,
                broker_ref=None,
                broker_value_sha256=None,
                metadata=_metadata("not_started", "not_applicable", None),
                updated_at=_timestamp(now),
            )
            self._commit_retirement_transition(
                record,
                target,
                human_preview_sha256=self._recovery_human_preview(record),
                reason="continuation_expired",
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
                    reason="continuation_retention_expired",
                )
                removed += 1
                changed += 1
            if changed == 0:
                return removed

    def _prepare_pending(
        self,
        continuation_id: str,
        *,
        expected_revision: int,
        binding: McpContinuationBinding,
    ) -> McpContinuationRecord:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValidationError("MCP continuation expected_revision is invalid")
        record = self._load(continuation_id)
        self._require_binding(record, binding)
        if record.revision != expected_revision:
            raise ValidationError("MCP continuation revision conflict")
        self._require_status(record, McpContinuationStatus.INPUT_REQUIRED)
        if _utc(self._now()) >= _parse_timestamp(record.expires_at):
            self._expire(record, binding)
        return record

    def _expire(
        self,
        record: McpContinuationRecord,
        binding: McpContinuationBinding,
    ) -> None:
        self._require_binding(record, binding)
        preview_sha256 = self._recovery_human_preview(record)
        expired = replace(
            record,
            status=McpContinuationStatus.EXPIRED,
            revision=record.revision + 1,
            broker_ref=None,
            broker_value_sha256=None,
            metadata=_metadata("not_started", "not_applicable", None),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._commit_retirement_transition(
            record,
            expired,
            human_preview_sha256=preview_sha256,
            reason="continuation_expired",
        )
        self._prune_terminal_records()
        raise ValidationError("MCP continuation expired")

    def _parse_input_required(
        self,
        result: Mapping[str, JsonValue],
        *,
        round_number: int,
        sensitive_values: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, JsonValue], McpInputRequired, bool]:
        if not isinstance(result, Mapping):
            raise ValidationError("MCP input_required result is invalid")
        selected = dict(result)
        strict_json_value(selected, label="MCP input_required result")
        if selected.get("resultType") != "input_required":
            raise ValidationError("MCP result is not input_required")
        if set(selected) - {"resultType", "inputRequests", "requestState", "_meta"}:
            raise ValidationError("MCP input_required result contains unsupported fields")
        state = selected.get("requestState")
        if state is not None and (
            type(state) is not str
            or len(state.encode("utf-8")) > self._request_state_max_bytes
        ):
            raise ValidationError("MCP requestState is invalid")
        selected_sensitive = (
            self._sensitive_values
            if sensitive_values is None
            else sensitive_values
        )
        if state is not None:
            reject_opaque_secret_reflection(
                state,
                sensitive_values=selected_sensitive,
                label="MCP requestState",
            )
        raw_requests = selected.get("inputRequests")
        if raw_requests is None:
            # requestState-only load shedding still requires an explicit Host
            # action.  Represent it as an empty input set; no automatic replay.
            if state is None:
                raise ValidationError(
                    "MCP input_required requires inputRequests or requestState"
                )
            parsed_public = ()
            respondable = True
            normalized_requests: dict[str, JsonValue] = {}
        else:
            parsed = parse_input_requests(
                raw_requests,
                sensitive_values=selected_sensitive,
                max_requests=self._max_input_requests,
            )
            parsed_public = parsed.public
            respondable = not parsed.has_unsupported
            normalized_requests = parsed.raw
        envelope: dict[str, JsonValue] = {
            "round": round_number,
            "inputRequests": normalized_requests,
        }
        if state is not None:
            envelope["requestState"] = state
        canonical_json_bytes(envelope, label="MCP continuation broker envelope")
        return (
            envelope,
            McpInputRequired(
                input_requests=tuple(parsed_public),
                respondable=respondable,
            ),
            respondable,
        )

    def prepare_response_result(
        self,
        claimed: McpContinuationRecord,
        binding: McpContinuationBinding,
        raw_result: Mapping[str, JsonValue],
        *,
        previous_broker_ref: str | None,
        next_round: int,
        sensitive_values: tuple[str, ...],
        retire_human_preview_sha256: str,
        response_effect_id: str,
    ) -> tuple[McpOperationResult[Any], McpContinuationResultSettlement]:
        """Prepare a continuation result for its protected-effect transaction.

        This method deliberately performs no Continuation/Task main-row
        mutation.  It may create a Human request and broker slots only after a
        durable preparation owns them.  The returned settlement commits every
        main row from ``ProtectedOperation.complete``'s outer UnitOfWork.
        """

        if not isinstance(raw_result, Mapping):
            raise ValidationError("MCP continuation Provider result is invalid")
        _require_text(response_effect_id, "MCP continuation response effect id")
        if previous_broker_ref != claimed.broker_ref:
            raise ValidationError("MCP continuation broker binding changed")
        selected = dict(raw_result)
        strict_json_value(selected, label="MCP continuation Provider result")
        result_type = selected.get("resultType")
        if result_type == "input_required":
            if next_round > self._max_rounds:
                raise ValidationError("MCP continuation round limit exceeded")
            return self._prepare_next_input_required(
                claimed,
                binding,
                selected,
                next_round=next_round,
                previous_broker_ref=previous_broker_ref,
                sensitive_values=sensitive_values,
                retire_human_preview_sha256=retire_human_preview_sha256,
                response_effect_id=response_effect_id,
            )
        if result_type == "task":
            if self._remote_task_capture is None:
                raise ValidationError("MCP remote Task handler is unavailable")
            return self._prepare_remote_task_result(
                claimed,
                binding,
                selected,
                previous_broker_ref=previous_broker_ref,
                sensitive_values=sensitive_values,
                retire_human_preview_sha256=retire_human_preview_sha256,
                response_effect_id=response_effect_id,
            )
        if result_type != "complete":
            raise ValidationError("MCP continuation resultType is unsupported")
        sanitized = sanitize_provider_json(
            {key: value for key, value in selected.items() if key != "resultType"},
            sensitive_values=sensitive_values,
            label="MCP continuation result",
        )
        public = McpComplete(value=sanitized)
        return self._prepare_terminal_result(
            claimed,
            public,
            status=McpContinuationStatus.COMPLETE,
            previous_broker_ref=previous_broker_ref,
            retire_human_preview_sha256=retire_human_preview_sha256,
            response_effect_id=response_effect_id,
        )

    def _prepare_next_input_required(
        self,
        claimed: McpContinuationRecord,
        binding: McpContinuationBinding,
        selected: Mapping[str, JsonValue],
        *,
        next_round: int,
        previous_broker_ref: str | None,
        sensitive_values: tuple[str, ...],
        retire_human_preview_sha256: str,
        response_effect_id: str,
    ) -> tuple[McpOperationResult[Any], McpContinuationResultSettlement]:
        envelope, public, respondable = self._parse_input_required(
            selected,
            round_number=next_round,
            sensitive_values=sensitive_values,
        )
        envelope["originalRequest"] = binding.detached_request()
        envelope["resultEffectId"] = response_effect_id
        canonical_json_bytes(envelope, label="MCP continuation broker envelope")
        if not respondable:
            return self._prepare_terminal_result(
                claimed,
                replace(public, respondable=False),
                status=McpContinuationStatus.NEEDS_ATTENTION,
                previous_broker_ref=previous_broker_ref,
                retire_human_preview_sha256=retire_human_preview_sha256,
                response_effect_id=response_effect_id,
            )
        secret = canonical_json_bytes(
            envelope,
            label="MCP continuation broker envelope",
        )
        preview, preview_sha256 = mcp_human_preview(
            server_id=binding.server_id,
            operation=self._binding_operation(binding),
            local_ref=claimed.continuation_id,
            input_requests=public.input_requests,
        )
        human_request_id = self.human_requests.reserve_question_id()
        _require_text(human_request_id, "MCP Human request id")
        broker_namespace = (
            f"mcp.continuation.{claimed.continuation_id}."
            f"r{claimed.revision + 1}"
        )
        preparation = self._prepare_response_side_effects(
            claimed,
            previous_broker_ref=previous_broker_ref,
            retire_human_preview_sha256=retire_human_preview_sha256,
            human_request_id=human_request_id,
            human_preview_sha256=preview_sha256,
            broker_namespace=broker_namespace,
            broker_value=secret,
        )
        try:
            human_receipt = require_human_receipt(
                self.human_requests.create_question(
                    owner_id=binding.owner_id,
                    server_id=binding.server_id,
                    operation=self._binding_operation(binding),
                    local_ref=claimed.continuation_id,
                    preview=preview,
                    preview_sha256=preview_sha256,
                    expires_at=claimed.expires_at,
                    request_id=human_request_id,
                ),
                preview_sha256=preview_sha256,
            )
            if human_receipt.request_id != human_request_id:
                raise ValidationError("MCP Human request reservation changed")
            write_mcp_prepared_secrets(
                preparation,
                broker=self.broker,
                broker_namespace=broker_namespace,
                broker_value=secret,
                result_namespace=None,
                result_value=None,
            )
            target = replace(
                claimed,
                status=McpContinuationStatus.INPUT_REQUIRED,
                revision=claimed.revision + 1,
                human_request_id=human_receipt.request_id,
                broker_ref=preparation.broker_ref,
                broker_value_sha256=sha256(secret).hexdigest(),
                metadata=_metadata("started", "reobserve_required", envelope),
                updated_at=_timestamp(_utc(self._now())),
            )
            captured = replace(
                public,
                continuation_id=target.continuation_id,
                expires_at=target.expires_at,
                revision=target.revision,
                respondable=True,
                human_request_id=human_receipt.request_id,
                human_revision=human_receipt.revision,
                human_preview_sha256=preview_sha256,
            )
            settlement = self._response_settlement(
                preparation,
                claimed,
                target,
                captured,
                response_effect_id=response_effect_id,
            )
        except Exception:
            cleanup_mcp_preparation(
                self.side_effects,
                preparation,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason="settlement_failed",
            )
            raise
        return captured, settlement

    def _prepare_remote_task_result(
        self,
        claimed: McpContinuationRecord,
        binding: McpContinuationBinding,
        selected: Mapping[str, JsonValue],
        *,
        previous_broker_ref: str | None,
        sensitive_values: tuple[str, ...],
        retire_human_preview_sha256: str,
        response_effect_id: str,
    ) -> tuple[McpOperationResult[Any], McpContinuationResultSettlement]:
        capture = self._remote_task_capture
        if capture is None:
            raise ValidationError("MCP continuation Task settlement is unavailable")
        task, task_settlement = capture.prepare_capture(
            binding,
            selected,
            origin_effect_id=response_effect_id,
            sensitive_values=sensitive_values,
        )
        receipt_value = canonical_json_bytes(
            {
                "kind": "remote_task",
                "schemaVersion": 1,
                "taskRef": task.task_ref,
                "responseEffectId": response_effect_id,
            },
            label="MCP continuation result receipt",
        )
        now = _timestamp(_utc(self._now()))
        receipt_namespace = (
            f"mcp.continuation.{claimed.continuation_id}."
            f"result.r{claimed.revision + 1}"
        )
        preparation: McpSideEffectPreparationRecord | None = None
        try:
            preparation = self._prepare_response_side_effects(
                claimed,
                previous_broker_ref=previous_broker_ref,
                retire_human_preview_sha256=retire_human_preview_sha256,
                human_request_id=None,
                human_preview_sha256=None,
                broker_namespace=receipt_namespace,
                broker_value=receipt_value,
            )
            write_mcp_prepared_secrets(
                preparation,
                broker=self.broker,
                broker_namespace=receipt_namespace,
                broker_value=receipt_value,
                result_namespace=None,
                result_value=None,
            )
            target = replace(
                claimed,
                status=McpContinuationStatus.COMPLETE,
                revision=claimed.revision + 1,
                broker_ref=preparation.broker_ref,
                broker_value_sha256=sha256(receipt_value).hexdigest(),
                metadata=_metadata("started", "not_applicable", None),
                updated_at=now,
            )
            settlement = self._response_settlement(
                preparation,
                claimed,
                target,
                task,
                response_effect_id=response_effect_id,
                task_settlement=task_settlement,
            )
        except Exception:
            try:
                task_settlement.abort(reason="continuation_task_settlement_failed")
            finally:
                if preparation is not None:
                    cleanup_mcp_preparation(
                        self.side_effects,
                        preparation,
                        broker=self.broker,
                        human_requests=self.human_requests,
                        updated_at=_timestamp(_utc(self._now())),
                        reason="continuation_task_settlement_failed",
                    )
            raise
        return task, settlement

    def _prepare_terminal_result(
        self,
        claimed: McpContinuationRecord,
        public: McpOperationResult[Any],
        *,
        status: McpContinuationStatus,
        previous_broker_ref: str | None,
        retire_human_preview_sha256: str,
        response_effect_id: str,
    ) -> tuple[McpOperationResult[Any], McpContinuationResultSettlement]:
        if status not in {
            McpContinuationStatus.COMPLETE,
            McpContinuationStatus.NEEDS_ATTENTION,
        }:
            raise ValidationError("MCP continuation terminal result is invalid")
        preparation = self._prepare_response_side_effects(
            claimed,
            previous_broker_ref=previous_broker_ref,
            retire_human_preview_sha256=retire_human_preview_sha256,
            human_request_id=None,
            human_preview_sha256=None,
            broker_namespace=None,
            broker_value=None,
        )
        metadata = (
            _metadata("started", "not_applicable", None)
            if status is McpContinuationStatus.COMPLETE
            else _metadata(
                "unknown",
                "unsafe_or_unknown",
                None,
                reason="unsupported_input_request",
            )
        )
        target = replace(
            claimed,
            status=status,
            revision=claimed.revision + 1,
            broker_ref=None,
            broker_value_sha256=None,
            metadata=metadata,
            updated_at=_timestamp(_utc(self._now())),
        )
        return public, self._response_settlement(
            preparation,
            claimed,
            target,
            public,
            response_effect_id=response_effect_id,
        )

    def _prepare_response_side_effects(
        self,
        claimed: McpContinuationRecord,
        *,
        previous_broker_ref: str | None,
        retire_human_preview_sha256: str,
        human_request_id: str | None,
        human_preview_sha256: str | None,
        broker_namespace: str | None,
        broker_value: bytes | None,
    ) -> McpSideEffectPreparationRecord:
        if previous_broker_ref != claimed.broker_ref:
            raise ValidationError("MCP continuation broker binding changed")
        return prepare_mcp_side_effects(
            repository=self.side_effects,
            broker=self.broker,
            operation_kind="continuation",
            operation_id=claimed.continuation_id,
            operation_revision=claimed.revision,
            server_id=claimed.server_id,
            server_spec_sha256=claimed.server_spec_sha256,
            server_generation=claimed.server_generation,
            owner_id=claimed.owner_id,
            auth_principal_sha256=claimed.auth_principal_sha256,
            auth_scope_sha256=claimed.auth_scope_sha256,
            human_request_id=human_request_id,
            human_preview_sha256=human_preview_sha256,
            broker_namespace=broker_namespace,
            broker_value_sha256=(
                sha256(broker_value).hexdigest()
                if broker_value is not None
                else None
            ),
            result_namespace=None,
            result_sha256=None,
            expires_at=claimed.expires_at,
            created_at=_timestamp(_utc(self._now())),
            retire_refs=(previous_broker_ref,) if previous_broker_ref else (),
            retire_human_request_id=claimed.human_request_id,
            retire_human_preview_sha256=retire_human_preview_sha256,
        )

    def _response_settlement(
        self,
        preparation: McpSideEffectPreparationRecord,
        claimed: McpContinuationRecord,
        target: McpContinuationRecord,
        public: McpOperationResult[Any],
        *,
        response_effect_id: str,
        task_settlement: Any | None = None,
    ) -> McpContinuationResultSettlement:
        if target.continuation_id != claimed.continuation_id or (
            target.revision != claimed.revision + 1
        ):
            raise ValidationError("MCP continuation result target is invalid")
        return McpContinuationResultSettlement(
            self,
            _PreparedContinuationResult(
                preparation=preparation,
                claimed=claimed,
                target=target,
                public=deepcopy(public),
                response_effect_id=response_effect_id,
                task_settlement=task_settlement,
            ),
        )

    def commit_response_result_deferred(
        self,
        settlement: McpContinuationResultSettlement,
    ) -> None:
        """Join the result transition to the current protected transaction."""

        prepared = self._prepared_response_settlement(settlement)
        if prepared.retirement is not None:
            if self.side_effects.get(prepared.preparation.preparation_id) == (
                prepared.retirement
            ):
                return
            raise ValidationError("MCP continuation result retirement changed")
        if prepared.task_settlement is not None:
            prepared.task_settlement.commit_deferred()
        prepared.retirement = commit_mcp_preparation_deferred(
            self.side_effects,
            prepared.preparation,
            prepared.target,
        )

    def finalize_response_result(
        self,
        settlement: McpContinuationResultSettlement,
    ) -> None:
        """Clean retired ownership only after the outer transaction commits."""

        prepared = self._prepared_response_settlement(settlement)
        retirement = prepared.retirement
        if retirement is None:
            raise ValidationError("MCP continuation result was not committed")
        if self.repository.get(prepared.target.continuation_id) != prepared.target:
            raise ValidationError("MCP continuation result commit changed")
        if prepared.task_settlement is not None:
            prepared.task_settlement.finalize()
        finalize_mcp_preparation(
            self.side_effects,
            retirement,
            broker=self.broker,
            human_requests=self.human_requests,
        )
        prepared.closed = True
        if _status(prepared.target) in {
            McpContinuationStatus.COMPLETE,
            McpContinuationStatus.NEEDS_ATTENTION,
        }:
            self._prune_terminal_records()

    def abort_response_result(
        self,
        settlement: McpContinuationResultSettlement,
        *,
        reason: str,
    ) -> None:
        """Abort a rolled-back result or finish an already committed one."""

        prepared = self._prepared_response_settlement(
            settlement,
            allow_closed=True,
        )
        if prepared.closed:
            return
        task_error: Exception | None = None
        if prepared.task_settlement is not None:
            try:
                prepared.task_settlement.abort(reason=reason)
            except Exception as exc:
                task_error = exc
        current_sidecar = self.side_effects.get(
            prepared.preparation.preparation_id
        )
        current_record = self.repository.get(prepared.claimed.continuation_id)
        if current_sidecar is None:
            if (
                current_record != prepared.claimed
                and current_record != prepared.target
            ):
                raise ValidationError("MCP continuation result state changed")
        elif current_sidecar.status == "prepared":
            if current_record != prepared.claimed:
                raise ValidationError("MCP prepared continuation result changed")
            cleanup_mcp_preparation(
                self.side_effects,
                current_sidecar,
                broker=self.broker,
                human_requests=self.human_requests,
                updated_at=_timestamp(_utc(self._now())),
                reason=reason,
            )
        elif current_sidecar.status == "cleaning":
            if current_record != prepared.target:
                raise ValidationError("MCP committed continuation result changed")
            finalize_mcp_preparation(
                self.side_effects,
                current_sidecar,
                broker=self.broker,
                human_requests=self.human_requests,
            )
        else:
            raise ValidationError("MCP continuation result sidecar is invalid")
        prepared.closed = True
        if task_error is not None:
            raise task_error

    def _prepared_response_settlement(
        self,
        settlement: McpContinuationResultSettlement,
        *,
        allow_closed: bool = False,
    ) -> _PreparedContinuationResult:
        if (
            type(settlement) is not McpContinuationResultSettlement
            or settlement.manager is not self
        ):
            raise TypeError("MCP continuation result settlement is invalid")
        prepared = settlement.prepared
        if prepared.closed and not allow_closed:
            raise ValidationError("MCP continuation result settlement is closed")
        return prepared

    def completed_remote_task_handoff(self, continuation_id: str) -> tuple[str, str]:
        """Read the safe local Task ref and its response-effect fence."""

        record = self._load(continuation_id)
        self._require_status(record, McpContinuationStatus.COMPLETE)
        if not self.broker.available() or record.broker_ref is None:
            raise ValidationError("MCP continuation result receipt is unavailable")
        try:
            secret = self.broker.get_secret(record.broker_ref)
        except Exception as exc:
            raise ValidationError("MCP continuation result receipt is unavailable") from exc
        if (
            record.broker_value_sha256 is None
            or sha256(secret).hexdigest() != record.broker_value_sha256
        ):
            raise ValidationError("MCP continuation result receipt integrity failed")
        receipt = decode_broker_json(secret, label="MCP continuation result receipt")
        if set(receipt) != {
            "kind",
            "schemaVersion",
            "taskRef",
            "responseEffectId",
        } or (
            receipt.get("kind") != "remote_task"
            or receipt.get("schemaVersion") != 1
        ):
            raise ValidationError("MCP continuation result receipt is invalid")
        return (
            _require_text(receipt.get("taskRef"), "MCP remote Task local reference"),
            _require_text(
                receipt.get("responseEffectId"),
                "MCP continuation response effect id",
            ),
        )

    def completed_remote_task_ref(self, continuation_id: str) -> str:
        """Read only the safe local Task ref from a completed continuation."""

        task_ref, _effect_id = self.completed_remote_task_handoff(continuation_id)
        return task_ref

    def effect_id_for_recovery(self, continuation_id: str) -> str:
        """Return only the immutable origin-effect fence for Host recovery."""

        return self._load(continuation_id).effect_id

    def accepts_recovery_effect(self, continuation_id: str, effect_id: str) -> bool:
        """Verify an origin or latest-result effect against durable state."""

        selected_effect = _require_text(effect_id, "MCP continuation effect id")
        record = self._load(continuation_id)
        if record.effect_id == selected_effect:
            return True
        status = _status(record)
        if status is McpContinuationStatus.INPUT_REQUIRED:
            envelope = self._load_envelope(record)
            return envelope.get("resultEffectId") == selected_effect
        if status is McpContinuationStatus.COMPLETE:
            _task_ref, response_effect_id = self.completed_remote_task_handoff(
                continuation_id
            )
            return response_effect_id == selected_effect
        return False

    def recover_local_result(self, continuation_id: str) -> McpInputRequired | str:
        """Recover a pending round or its atomically handed-off local Task ref."""

        record = self._load(continuation_id)
        status = _status(record)
        if status is McpContinuationStatus.INPUT_REQUIRED:
            binding = self.binding_material(continuation_id)
            return self.get(continuation_id, binding=binding)
        if status is McpContinuationStatus.COMPLETE:
            return self.completed_remote_task_ref(continuation_id)
        raise NotFound("MCP continuation has no recoverable durable result")

    def _load_envelope(self, record: McpContinuationRecord) -> dict[str, JsonValue]:
        if not self.broker.available() or record.broker_ref is None:
            self._mark_attention(record, reason="broker_unavailable")
            raise ValidationError("MCP credential broker is unavailable")
        try:
            secret = self.broker.get_secret(record.broker_ref)
        except Exception as exc:
            self._mark_attention(record, reason="broker_missing")
            raise ValidationError("MCP credential broker value is unavailable") from exc
        if (
            record.broker_value_sha256 is None
            or sha256(secret).hexdigest() != record.broker_value_sha256
        ):
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation broker integrity check failed")
        envelope = decode_broker_json(secret, label="MCP continuation broker envelope")
        if set(envelope) - {
            "round",
            "inputRequests",
            "requestState",
            "originalRequest",
            "resultEffectId",
        }:
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation broker payload is invalid")
        if type(envelope.get("inputRequests")) is not dict:
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation broker payload is invalid")
        self._original_request(record, envelope)
        return envelope

    def _original_request(
        self,
        record: McpContinuationRecord,
        envelope: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        original_request = envelope.get("originalRequest")
        if type(original_request) is not dict:
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation original request is unavailable")
        encoded = canonical_json_bytes(
            original_request,
            label="MCP continuation original request",
        )
        if sha256(encoded).hexdigest() != record.request_sha256:
            self._mark_attention(record, reason="broker_integrity")
            raise ValidationError("MCP continuation original request binding changed")
        return decode_broker_json(
            encoded,
            label="MCP continuation original request",
        )

    def _create_human_question(
        self,
        *,
        binding: McpContinuationBinding,
        continuation_id: str,
        public: McpInputRequired,
        expires_at: str,
    ) -> tuple[McpHumanRequestReceipt, str]:
        preview, preview_sha256 = mcp_human_preview(
            server_id=binding.server_id,
            operation=self._binding_operation(binding),
            local_ref=continuation_id,
            input_requests=public.input_requests,
        )
        receipt = require_human_receipt(
            self.human_requests.create_question(
                owner_id=binding.owner_id,
                server_id=binding.server_id,
                operation=self._binding_operation(binding),
                local_ref=continuation_id,
                preview=preview,
                preview_sha256=preview_sha256,
                expires_at=expires_at,
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

    @staticmethod
    def _binding_operation(binding: McpContinuationBinding) -> str:
        operation = binding.canonical_request.get("method")
        if type(operation) is not str or not operation:
            raise ValidationError("MCP continuation operation binding is invalid")
        return operation

    def _load(self, continuation_id: str) -> McpContinuationRecord:
        _require_text(continuation_id, "MCP continuation id")
        record = self.repository.get(continuation_id)
        if record is None:
            raise NotFound("MCP continuation not found")
        return record

    @staticmethod
    def _require_binding(
        record: McpContinuationRecord,
        binding: McpContinuationBinding,
    ) -> None:
        if not isinstance(binding, McpContinuationBinding):
            raise TypeError("MCP continuation binding is required")
        actual = (
            binding.server_id,
            binding.server_spec_sha256,
            binding.server_generation,
            binding.owner_id,
            binding.auth_principal_sha256,
            binding.auth_scope_sha256,
            binding.request_sha256,
            binding.effect_id,
            binding.capability_sha256,
            binding.data_flow_sha256,
        )
        expected = (
            record.server_id,
            record.server_spec_sha256,
            record.server_generation,
            record.owner_id,
            record.auth_principal_sha256,
            record.auth_scope_sha256,
            record.request_sha256,
            record.effect_id,
            record.capability_sha256,
            record.data_flow_sha256,
        )
        if actual != expected:
            raise CapabilityDenied("MCP continuation binding changed")

    @staticmethod
    def _require_status(
        record: McpContinuationRecord,
        expected: McpContinuationStatus,
    ) -> None:
        if _status(record) is not expected:
            raise ValidationError("MCP continuation is terminal or in an invalid state")

    def _recovery_human_preview(self, record: McpContinuationRecord) -> str:
        return self.human_requests.question_preview_sha256_for_recovery(
            record.human_request_id
        )

    def _retirement_preparation(
        self,
        record: McpContinuationRecord,
        *,
        human_preview_sha256: str,
    ) -> McpSideEffectPreparationRecord:
        return prepare_mcp_side_effects(
            repository=self.side_effects,
            broker=self.broker,
            operation_kind="continuation",
            operation_id=record.continuation_id,
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
            expires_at=record.expires_at,
            created_at=_timestamp(_utc(self._now())),
            retire_refs=((record.broker_ref,) if record.broker_ref else ()),
            retire_human_request_id=record.human_request_id,
            retire_human_preview_sha256=human_preview_sha256,
        )

    def _commit_retirement_transition(
        self,
        record: McpContinuationRecord,
        target: McpContinuationRecord,
        *,
        human_preview_sha256: str,
        reason: str,
    ) -> None:
        preparation = self._retirement_preparation(
            record,
            human_preview_sha256=human_preview_sha256,
        )
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
        record: McpContinuationRecord,
        *,
        reason: str,
    ) -> None:
        preparation = self._retirement_preparation(
            record,
            human_preview_sha256=self._recovery_human_preview(record),
        )
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

    def _mark_attention(self, record: McpContinuationRecord, *, reason: str) -> None:
        current = self.repository.get(record.continuation_id)
        if current is None or current.revision != record.revision:
            return
        if _status(current) is McpContinuationStatus.NEEDS_ATTENTION:
            return
        target = replace(
            current,
            status=McpContinuationStatus.NEEDS_ATTENTION,
            revision=current.revision + 1,
            broker_ref=None,
            broker_value_sha256=None,
            metadata=_metadata("unknown", "unsafe_or_unknown", None, reason=reason),
            updated_at=_timestamp(_utc(self._now())),
        )
        self._commit_retirement_transition(
            current,
            target,
            human_preview_sha256=self._recovery_human_preview(current),
            reason=reason,
        )

    def _cas_or_conflict(
        self,
        expected: McpContinuationRecord,
        target: McpContinuationRecord,
    ) -> None:
        if not self.repository.compare_and_swap(
            expected.continuation_id,
            expected_revision=expected.revision,
            replacement=target,
        ):
            raise ValidationError("MCP continuation revision conflict")

    def _cas_or_unknown(
        self,
        expected: McpContinuationRecord,
        target: McpContinuationRecord,
    ) -> None:
        if not self.repository.compare_and_swap(
            expected.continuation_id,
            expected_revision=expected.revision,
            replacement=target,
        ):
            self._mark_attention(expected, reason="settlement_conflict")
            raise ValidationError("MCP continuation settlement is unknown")


def _metadata(
    dispatch_state: str,
    retry_class: str,
    envelope: Mapping[str, JsonValue] | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "automatic_retry_disabled": True,
        "dispatch_state": dispatch_state,
        "retry_class": retry_class,
    }
    if envelope is not None:
        selected["input_schema_sha256"] = json_sha256(
            envelope.get("inputRequests", {}),
            label="MCP input request schema",
        )
    if reason is not None:
        selected["reason_code"] = reason
    return selected


def _raise_certified_not_started(
    error: McpContinuationDispatchNotStarted,
) -> None:
    cause = error.__cause__
    if isinstance(cause, Exception):
        raise cause
    raise error


def _status(record: McpContinuationRecord) -> McpContinuationStatus:
    try:
        return McpContinuationStatus(record.status)
    except ValueError as exc:  # pragma: no cover - storage record validates
        raise ValidationError("MCP continuation status is invalid") from exc


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


def _require_human_response_fence(revision: Any, preview_sha256: Any) -> None:
    if type(revision) is not int or revision < 0:
        raise ValidationError("MCP Human expected revision is invalid")
    _require_sha256(preview_sha256, "MCP Human preview")


def _merge_sensitive_values(*groups: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for group in groups:
        if type(group) is not tuple or any(
            type(item) is not str or not item for item in group
        ):
            raise ValidationError("MCP sensitive value snapshot is invalid")
        selected.extend(group)
    return tuple(dict.fromkeys(selected))


def _detached_binding(binding: McpContinuationBinding) -> McpContinuationBinding:
    return McpContinuationBinding(
        server_id=binding.server_id,
        server_spec_sha256=binding.server_spec_sha256,
        server_generation=binding.server_generation,
        owner_id=binding.owner_id,
        auth_principal_sha256=binding.auth_principal_sha256,
        auth_scope_sha256=binding.auth_scope_sha256,
        canonical_request=binding.detached_request(),
        effect_id=binding.effect_id,
        capability_sha256=binding.capability_sha256,
        data_flow_sha256=binding.data_flow_sha256,
    )


def _validate_continuation_policy(
    *,
    max_rounds: Any,
    max_input_requests: Any,
    request_state_max_bytes: Any,
    continuation_ttl_s: Any,
    max_records: Any,
    terminal_records: Any,
    reconcile_on_start: Any,
) -> None:
    if type(max_rounds) is not int or not 1 <= max_rounds <= 100:
        raise ValidationError("MCP continuation max_rounds is invalid")
    if type(max_input_requests) is not int or not 1 <= max_input_requests <= _MAX_INPUT_REQUESTS:
        raise ValidationError("MCP continuation input request limit is invalid")
    if (
        type(request_state_max_bytes) is not int
        or not 1 <= request_state_max_bytes <= _MAX_REQUEST_STATE_BYTES
    ):
        raise ValidationError("MCP continuation requestState limit is invalid")
    if (
        type(continuation_ttl_s) not in {int, float}
        or isinstance(continuation_ttl_s, bool)
        or not math.isfinite(float(continuation_ttl_s))
        or continuation_ttl_s <= 0
    ):
        raise ValidationError("MCP continuation TTL is invalid")
    if type(reconcile_on_start) is not bool:
        raise ValidationError("MCP continuation reconcile_on_start is invalid")
    if type(max_records) is not int or max_records < 1:
        raise ValidationError("MCP continuation record limit is invalid")
    if type(terminal_records) is not int or not 1 <= terminal_records <= 499:
        raise ValidationError("MCP continuation terminal retention is invalid")


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("MCP clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    if type(value) is not str or len(value) > 128:
        raise ValidationError("MCP continuation expiry is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("MCP continuation expiry is invalid") from exc
    return _utc(parsed)


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds")


def _deadline(value: float) -> float:
    if type(value) not in {int, float} or not value == value or value <= 0:
        raise ValidationError("MCP continuation deadline is invalid")
    return float(value)


def _new_continuation_id() -> str:
    import secrets

    return f"mcpcont_{secrets.token_urlsafe(24)}"


__all__ = [
    "McpContinuationBinding",
    "McpContinuationBoundary",
    "McpContinuationDispatchNotStarted",
    "McpContinuationManager",
    "McpContinuationRecord",
    "McpContinuationRepository",
    "McpContinuationStatus",
    "McpSdkContinuationCaptureAdapter",
]
