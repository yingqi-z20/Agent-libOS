from __future__ import annotations

import asyncio
import inspect
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from agent_libos.models import EventType
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.storage.base import StoreCloseClaimOutcome, StoreCloseOutcome
from agent_libos.utils.ids import new_id
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)


class _LifecycleState(str, Enum):
    NEW = "new"
    RECOVERING = "recovering"
    STARTING = "starting"
    OPEN = "open"
    STOPPING = "stopping"
    CLOSE_FAILED = "close_failed"
    CLOSED = "closed"


_RECOVERY_REQUIRED_REASON_PREFIX = "runtime.recovery_required:"


@dataclass(slots=True)
class _ShutdownAttempt:
    owner_thread_id: int
    owner_task: asyncio.Task[Any] | None
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _FailedAssemblyCleanupAttempt:
    owner_thread_id: int
    owner_task: asyncio.Task[Any] | None
    done: threading.Event = field(default_factory=threading.Event)
    result: list[dict[str, str]] | None = None


@dataclass(slots=True)
class _FinalizerEntry:
    handle: str
    callback: Any
    recovery_safe: bool = False
    completed: bool = False


@dataclass(slots=True)
class _AdmissionLease:
    recovery_fence_epoch: int
    read_only: bool
    active: bool = True


@dataclass(slots=True)
class _StartupOpenCommit:
    owner_thread_id: int
    owner_task: asyncio.Task[Any] | None
    active: bool = True
    consumed: bool = False
    committed: bool = False


@dataclass(slots=True)
class _StartupLeaseScope:
    owner_thread_id: int
    owner_task: asyncio.Task[Any] | None
    active: bool = True


@dataclass(slots=True)
class _RecoveryLeaseScope:
    owner_thread_id: int
    owner_task: asyncio.Task[Any] | None
    active: bool = True


class RuntimeRegistryLock:
    """Re-entrant registry barrier that revokes pre-fence mutation leases."""

    def __init__(self, lifecycle: RuntimeLifecycle) -> None:
        self._lock = threading.RLock()
        self._lifecycle = lifecycle
        self._local = threading.local()

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        acquired = self._lock.acquire(*args, **kwargs)
        if not acquired:
            return False
        depth = int(getattr(self._local, "depth", 0))
        try:
            if depth == 0:
                self._lifecycle.revalidate_current_admission_if_present()
        except BaseException:
            self._lock.release()
            raise
        self._local.depth = depth + 1
        return True

    def release(self) -> None:
        depth = int(getattr(self._local, "depth", 0))
        if depth <= 0:
            raise RuntimeError("cannot release an unowned registry lifecycle lock")
        self._local.depth = depth - 1
        self._lock.release()

    def __enter__(self) -> RuntimeRegistryLock:
        self.acquire()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        self.release()


class RuntimeLifecycle:
    """Single-source Runtime lifecycle, admission gate, and teardown engine."""

    def __init__(
        self,
        *,
        store: Any,
        audit: Any,
        events: Any,
        substrate: Any,
        admission_drain_timeout_s: float = 2.0,
    ) -> None:
        self._store = store
        self._audit = audit
        self._events = events
        self._substrate = substrate
        self._scheduler: Any | None = None
        self._object_tasks: Any | None = None
        self._modules: Any | None = None
        self._llms: Any | None = None
        self._blocking_work: Any | None = None
        self._components_bound = False
        self._finalizers: list[_FinalizerEntry] = []
        self._lock = threading.RLock()
        self._admission_condition = threading.Condition(self._lock)
        self._state = _LifecycleState.NEW
        self._shutdown_reason: str | None = None
        self._active_attempt: _ShutdownAttempt | None = None
        self._active_failed_assembly_cleanup: (
            _FailedAssemblyCleanupAttempt | None
        ) = None
        self._active_leases = 0
        self._recovery_fence_epoch = 0
        self._admission_drain_timeout_s = max(0.0, float(admission_drain_timeout_s))
        self._ever_opened = False
        self._recovery_diagnostics_release_started = False
        self._recovery_diagnostics_releasing = False
        self._recovery_diagnostics_released = False
        self._recovery_diagnostics_release_warnings: list[dict[str, str]] = []
        self._shutdown_warnings: list[dict[str, str]] = []
        self._store_ownership_already_released = False
        self._recovery_token = object()
        self._recovery_cleanup_token = object()
        self._startup_token = object()
        self.__recovery_terminalization_capability = object()
        self.__recovery_diagnostics_release_capability = object()
        self._internal_admission: ContextVar[object | None] = ContextVar(
            f"agent_libos_internal_admission_{id(self)}",
            default=None,
        )
        self._current_admission: ContextVar[_AdmissionLease | None] = ContextVar(
            f"agent_libos_runtime_admission_{id(self)}",
            default=None,
        )
        self._startup_open_commit: ContextVar[_StartupOpenCommit | None] = (
            ContextVar(
                f"agent_libos_startup_open_commit_{id(self)}",
                default=None,
            )
        )
        self._startup_lease_scope: ContextVar[_StartupLeaseScope | None] = (
            ContextVar(
                f"agent_libos_startup_lease_scope_{id(self)}",
                default=None,
            )
        )
        self._recovery_lease_scope: ContextVar[_RecoveryLeaseScope | None] = (
            ContextVar(
                f"agent_libos_recovery_lease_scope_{id(self)}",
                default=None,
            )
        )
        self._recovery_cleanup_admission: ContextVar[object | None] = ContextVar(
            f"agent_libos_recovery_cleanup_{id(self)}",
            default=None,
        )
        self._shutdown_attempt_context: ContextVar[_ShutdownAttempt | None] = ContextVar(
            f"agent_libos_runtime_shutdown_attempt_{id(self)}",
            default=None,
        )
        self._failed_assembly_cleanup_context: ContextVar[
            _FailedAssemblyCleanupAttempt | None
        ] = ContextVar(
            f"agent_libos_failed_assembly_cleanup_{id(self)}",
            default=None,
        )
        self._recovery_terminalization_publication: ContextVar[str | None] = (
            ContextVar(
                f"agent_libos_recovery_terminalization_{id(self)}",
                default=None,
            )
        )
        bind_commit_guard = getattr(self._store, "bind_admission_commit_guard", None)
        unbind_commit_guard = getattr(
            self._store,
            "unbind_admission_commit_guard",
            None,
        )
        release_guard_and_close = getattr(
            self._store,
            "release_admission_guard_and_close",
            None,
        )
        probe_guard_close = getattr(
            self._store,
            "probe_admission_guard_close",
            None,
        )
        claim_guard_close = getattr(
            self._store,
            "claim_admission_guard_close",
            None,
        )
        if (
            not callable(bind_commit_guard)
            or not callable(unbind_commit_guard)
            or not callable(release_guard_and_close)
            or not callable(probe_guard_close)
            or not callable(claim_guard_close)
        ):
            raise RuntimeError(
                "runtime store does not support owned lifecycle admission commit fencing "
                "and atomic recovery handoff"
            )
        # Bound-method attribute access creates a fresh object.  Retain the
        # exact callable given to the store so failed-assembly cleanup can use
        # identity-CAS release without ever clearing a successor Runtime's
        # guard.
        self._admission_commit_guard_binding = self.admission_commit_guard
        self._unbind_admission_commit_guard = unbind_commit_guard
        self._release_admission_guard_and_close = release_guard_and_close
        self._probe_admission_guard_close = probe_guard_close
        self._claim_admission_guard_close = claim_guard_close
        bind_commit_guard(self._admission_commit_guard_binding)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._state is _LifecycleState.CLOSED

    @property
    def components_bound(self) -> bool:
        with self._lock:
            return self._components_bound

    @property
    def shutdown_reason(self) -> str | None:
        with self._lock:
            return self._shutdown_reason

    def begin_recovery(self) -> None:
        self._transition(_LifecycleState.NEW, _LifecycleState.RECOVERING)

    def begin_starting(self) -> None:
        self._transition(_LifecycleState.RECOVERING, _LifecycleState.STARTING)

    def mark_open(self) -> None:
        with self._lock:
            if self._state is not _LifecycleState.STARTING:
                raise RuntimeError(
                    "invalid runtime lifecycle transition: "
                    f"{self._state.value} -> {_LifecycleState.OPEN.value}"
                )
            previous_ever_opened = self._ever_opened
            try:
                self._state = _LifecycleState.OPEN
                # Failed-assembly cleanup is permanently unavailable once this
                # lifecycle has become a live Runtime. Recovery diagnostics use a
                # separate, no-write handoff path.
                self._ever_opened = True
            except BaseException:
                self._state = _LifecycleState.STARTING
                self._ever_opened = previous_ever_opened
                raise

    def mark_recovery_required(self, *, publication_id: str) -> None:
        """Fail mutation admission closed while preserving the diagnostic store."""

        selected_publication = self._require_publication_id(publication_id)
        with self._admission_condition:
            lease = self._current_admission.get()
            if lease is None or not lease.active:
                raise RuntimeError(
                    "runtime recovery fence requires an active admission lease"
                )
            recovery_reason = _RECOVERY_REQUIRED_REASON_PREFIX + selected_publication
            if (
                self._state is _LifecycleState.CLOSE_FAILED
                and self._shutdown_reason == recovery_reason
            ):
                return
            self._revalidate_admission(lease, read_only=False)
            if self._state not in {
                _LifecycleState.OPEN,
                _LifecycleState.STOPPING,
                _LifecycleState.CLOSE_FAILED,
            }:
                raise RuntimeError(
                    "runtime recovery fence requires an active runtime, got "
                    f"{self._state.value}"
                )
            self._recovery_fence_epoch += 1
            self._state = _LifecycleState.CLOSE_FAILED
            self._shutdown_reason = recovery_reason
            self._admission_condition.notify_all()

    def _transition(self, expected: _LifecycleState, selected: _LifecycleState) -> None:
        with self._lock:
            if self._state is not expected:
                raise RuntimeError(
                    f"invalid runtime lifecycle transition: {self._state.value} -> {selected.value}"
                )
            self._state = selected

    @staticmethod
    def _require_publication_id(publication_id: Any) -> str:
        if (
            type(publication_id) is not str
            or not publication_id
            or publication_id != publication_id.strip()
        ):
            raise TypeError(
                "runtime publication id must be an exact non-empty string"
            )
        return publication_id

    @contextmanager
    def recovery_lease(self) -> Iterator[None]:
        with self._lock:
            if self._state is not _LifecycleState.RECOVERING:
                raise RuntimeError(
                    "internal lifecycle lease requires recovering, got "
                    f"{self._state.value}"
                )
            inherited = self._recovery_lease_scope.get()
            if inherited is not None and inherited.active:
                raise RuntimeError("recovery lifecycle lease cannot be nested")
        scope = _RecoveryLeaseScope(
            owner_thread_id=threading.get_ident(),
            owner_task=self._current_task(),
        )
        scope_reset = self._recovery_lease_scope.set(scope)
        try:
            with self._internal_lease(
                self._recovery_token,
                _LifecycleState.RECOVERING,
            ):
                yield
        finally:
            with self._lock:
                # ContextVar values are copied into child tasks.  Mutating the
                # shared scope invalidates every copied context before the
                # parent resets its own binding.
                scope.active = False
            self._recovery_lease_scope.reset(scope_reset)

    def require_recovery_lease(self) -> None:
        """Reject recovery work outside the startup recovery context."""

        with self._lock:
            if (
                self._state is not _LifecycleState.RECOVERING
                or not self._current_recovery_lease_is_valid_locked()
            ):
                raise RuntimeError(
                    "runtime recovery requires the active startup recovery lease"
                )

    def _current_recovery_lease_is_valid_locked(self) -> bool:
        scope = self._recovery_lease_scope.get()
        return bool(
            scope is not None
            and scope.active
            and scope.owner_thread_id == threading.get_ident()
            and scope.owner_task is self._current_task()
            and self._internal_admission.get() is self._recovery_token
        )

    def require_recovery_cleanup_lease(self) -> None:
        """Reject raw transient cleanup outside an explicit handoff callback."""

        with self._admission_condition:
            if (
                not self._recovery_diagnostics_releasing
                or self._recovery_cleanup_admission.get()
                is not self._recovery_cleanup_token
            ):
                raise RuntimeError(
                    "raw recovery cleanup requires the active recovery cleanup lease"
                )

    @contextmanager
    def _recovery_cleanup_scope(self) -> Iterator[None]:
        reset = self._recovery_cleanup_admission.set(
            self._recovery_cleanup_token
        )
        try:
            yield
        finally:
            self._recovery_cleanup_admission.reset(reset)

    @contextmanager
    def startup_lease(self) -> Iterator[None]:
        with self._lock:
            if self._state is not _LifecycleState.STARTING:
                raise RuntimeError(
                    "internal lifecycle lease requires starting, got "
                    f"{self._state.value}"
                )
            inherited = self._startup_lease_scope.get()
            if inherited is not None and inherited.active:
                raise RuntimeError("startup lifecycle lease cannot be nested")
        scope = _StartupLeaseScope(
            owner_thread_id=threading.get_ident(),
            owner_task=self._current_task(),
        )
        scope_reset = self._startup_lease_scope.set(scope)
        try:
            with self._internal_lease(self._startup_token, _LifecycleState.STARTING):
                yield
        finally:
            with self._lock:
                scope.active = False
            self._startup_lease_scope.reset(scope_reset)

    def _current_startup_lease_is_valid_locked(self) -> bool:
        scope = self._startup_lease_scope.get()
        return bool(
            scope is not None
            and scope.active
            and scope.owner_thread_id == threading.get_ident()
            and scope.owner_task is self._current_task()
            and self._internal_admission.get() is self._startup_token
        )

    @contextmanager
    def in_memory_open_scope(self) -> Iterator[None]:
        """Publish a no-backlog OPEN transition as one rollback-safe scope."""

        with self._admission_condition:
            if (
                self._state is not _LifecycleState.STARTING
                or not self._current_startup_lease_is_valid_locked()
                or self._startup_open_commit.get() is not None
                or self._current_admission.get() is not None
            ):
                raise RuntimeError(
                    "in-memory OPEN requires the exact active startup lease"
                )
            if (
                self._recovery_diagnostics_release_started
                or self._recovery_required_result_locked() is not None
            ):
                raise RuntimeError(
                    "in-memory OPEN is unavailable while recovery is required"
                )
            previous_ever_opened = self._ever_opened
            try:
                self.mark_open()
                if (
                    self._state is not _LifecycleState.OPEN
                    or not self._ever_opened
                ):
                    raise RuntimeError(
                        "in-memory OPEN transition did not publish exact lifecycle state"
                    )
                yield
                if (
                    self._state is not _LifecycleState.OPEN
                    or not self._ever_opened
                ):
                    raise RuntimeError(
                        "in-memory OPEN scope lost exact lifecycle state"
                    )
            except BaseException:
                if self._state in {
                    _LifecycleState.STARTING,
                    _LifecycleState.OPEN,
                }:
                    self._state = _LifecycleState.STARTING
                    self._ever_opened = previous_ever_opened
                raise

    @contextmanager
    def open_on_next_commit(self) -> Iterator[None]:
        """Linearize durable startup acknowledgement with lifecycle OPEN.

        The caller must already hold the opaque startup lease and perform
        exactly one outer Store commit inside this scope. The Store owns its
        lock before entering :meth:`admission_commit_guard`, preserving the
        fixed store-to-lifecycle lock order.
        """

        with self._lock:
            if (
                self._state is not _LifecycleState.STARTING
                or not self._current_startup_lease_is_valid_locked()
            ):
                raise RuntimeError(
                    "startup OPEN commit requires the active startup lease"
                )
            if self._startup_open_commit.get() is not None:
                raise RuntimeError("startup OPEN commit scope cannot be nested")
        attempt = _StartupOpenCommit(
            owner_thread_id=threading.get_ident(),
            owner_task=self._current_task(),
        )
        reset = self._startup_open_commit.set(attempt)
        try:
            try:
                yield
            except BaseException:
                raise
            else:
                if not attempt.consumed:
                    raise RuntimeError(
                        "startup OPEN commit scope completed without a Store commit"
                    )
                if not attempt.committed:
                    raise RuntimeError(
                        "startup OPEN commit was consumed without committing"
                    )
        finally:
            with self._lock:
                attempt.active = False
            self._startup_open_commit.reset(reset)

    @contextmanager
    def _internal_lease(self, token: object, expected: _LifecycleState) -> Iterator[None]:
        with self._lock:
            if self._state is not expected:
                raise RuntimeError(
                    f"internal lifecycle lease requires {expected.value}, got {self._state.value}"
                )
        reset = self._internal_admission.set(token)
        try:
            yield
        finally:
            self._internal_admission.reset(reset)

    @contextmanager
    def admit(self, *, read_only: bool = False) -> Iterator[None]:
        """Acquire an operation lease before any operation/effect mutation."""

        if type(read_only) is not bool:
            raise TypeError("runtime admission read_only must be an exact boolean")
        inherited = self._current_admission.get()
        if inherited is not None and inherited.active:
            self._revalidate_admission(inherited, read_only=read_only)
            yield
            return

        with self._admission_condition:
            internal = self._internal_admission.get()
            allowed_internal = (
                (
                    self._state is _LifecycleState.RECOVERING
                    and internal is self._recovery_token
                    and self._current_recovery_lease_is_valid_locked()
                )
                or (
                    self._state is _LifecycleState.STARTING
                    and internal is self._startup_token
                    and self._current_startup_lease_is_valid_locked()
                )
            )
            allowed = (
                not self._recovery_diagnostics_releasing
                and (
                    self._state is _LifecycleState.OPEN
                    or allowed_internal
                    or (read_only and self._state is not _LifecycleState.CLOSED)
                )
            )
            if not allowed:
                raise RuntimeError(
                    f"runtime is not accepting operations: state={self._state.value}"
                )
            self._active_leases += 1
            # The fence epoch belongs to the same admission decision as the
            # active-lease increment.  Capturing it after releasing the
            # condition would let a concurrent recovery fence advance first
            # and make this pre-fence admission look current.
            lease = _AdmissionLease(
                recovery_fence_epoch=self._recovery_fence_epoch,
                read_only=read_only,
            )
        reset = self._current_admission.set(lease)
        try:
            yield
        finally:
            lease.active = False
            self._current_admission.reset(reset)
            with self._admission_condition:
                self._active_leases -= 1
                if self._active_leases == 0:
                    self._admission_condition.notify_all()

    def revalidate_current_admission_if_present(self) -> None:
        """Reject a mutation lease revoked while it waited on a barrier."""

        lease = self._current_admission.get()
        if lease is None:
            return
        self._revalidate_admission(lease, read_only=False)

    def current_mutation_admission_is_stale(self) -> bool:
        """Return whether recovery fencing revoked the current mutation lease."""

        lease = self._current_admission.get()
        with self._admission_condition:
            return bool(
                lease is not None
                and lease.active
                and lease.recovery_fence_epoch != self._recovery_fence_epoch
            )

    def _issue_recovery_diagnostics_release_capability(self) -> object:
        """Return the opaque Runtime facade capability for recovery handoff."""

        return self.__recovery_diagnostics_release_capability

    def release_recovery_diagnostics(
        self,
        *,
        capability: object,
    ) -> dict[str, Any]:
        """Release a fenced Runtime for same-process startup recovery.

        This is deliberately separate from ordinary shutdown. It publishes no
        shutdown evidence and never runs ordinary finalizers. Only callbacks
        explicitly registered as no-write recovery-safe cleanup may run before
        transient components and store ownership are released for the next
        Runtime instance.
        """

        preflight = self._preflight_recovery_diagnostics_release(
            capability,
            sync=True,
        )
        if preflight is not None:
            return preflight
        recovery_reason, early = self._begin_recovery_diagnostics_release(
            capability
        )
        if early is not None:
            return early
        self._mark_recovery_diagnostics_release_started()
        errors: list[dict[str, str]] = []
        completed = False
        try:
            failed = self._release_recovery_components_sync(
                (
                    ("scheduler", self._scheduler),
                    ("object_tasks", self._object_tasks),
                ),
                recovery_reason,
                errors,
            )
            if failed is not None:
                return failed

            failed_finalizer = self._run_sync_recovery_finalizers(errors)
            if failed_finalizer is not None:
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    failed_finalizer,
                    errors,
                )

            failed = self._release_recovery_components_sync(
                (
                    ("modules", self._modules),
                    ("llms", self._llms),
                    ("blocking_work", self._blocking_work),
                    ("substrate", self._substrate),
                ),
                recovery_reason,
                errors,
            )
            if failed is not None:
                return failed

            if not self._claim_recovery_store_close(errors):
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )
            try:
                store_released = self._release_admission_guard_and_close(
                    self._admission_commit_guard_binding
                )
            except Exception as exc:
                self._record_error(errors, "store", exc)
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )
            store_ok, control_warnings = self._store_close_outcome_released(
                store_released,
                errors,
            )
            if not store_ok:
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )

            result = self._finish_recovery_diagnostics_release(
                recovery_reason,
                already_released=False,
                warnings=errors,
            )
            completed = True
            self._raise_close_interrupts(
                interrupted=False,
                controls=control_warnings,
            )
            return result
        finally:
            if not completed:
                self._reset_recovery_diagnostics_release()

    async def arelease_recovery_diagnostics(
        self,
        *,
        capability: object,
    ) -> dict[str, Any]:
        """Native async recovery handoff preserving component loop affinity."""

        preflight = self._preflight_recovery_diagnostics_release(
            capability,
            sync=False,
        )
        if preflight is not None:
            return preflight
        recovery_reason, early = self._begin_recovery_diagnostics_release(
            capability
        )
        if early is not None:
            return early
        self._mark_recovery_diagnostics_release_started()
        errors: list[dict[str, str]] = []
        completed = False
        try:
            failed = await self._release_recovery_components_async(
                (
                    ("scheduler", self._scheduler),
                    ("object_tasks", self._object_tasks),
                ),
                recovery_reason,
                errors,
            )
            if failed is not None:
                return failed

            failed_finalizer = await self._run_async_recovery_finalizers(errors)
            if failed_finalizer is not None:
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    failed_finalizer,
                    errors,
                )

            failed = await self._release_recovery_components_async(
                (
                    ("modules", self._modules),
                    ("llms", self._llms),
                    ("blocking_work", self._blocking_work),
                    ("substrate", self._substrate),
                ),
                recovery_reason,
                errors,
            )
            if failed is not None:
                return failed

            if not self._claim_recovery_store_close(errors):
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )
            try:
                store_released, interrupted = (
                    await self._await_recovery_release_commit(
                        run_blocking_once(
                            self._release_admission_guard_and_close,
                            self._admission_commit_guard_binding,
                        )
                    )
                )
            except Exception as exc:
                self._record_error(errors, "store", exc)
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )
            store_ok, control_warnings = self._store_close_outcome_released(
                store_released,
                errors,
            )
            if not store_ok:
                if interrupted:
                    raise asyncio.CancelledError()
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    "store",
                    errors,
                )

            result = self._finish_recovery_diagnostics_release(
                recovery_reason,
                already_released=False,
                warnings=errors,
            )
            completed = True
            self._raise_close_interrupts(
                interrupted=interrupted,
                controls=control_warnings,
            )
            return result
        finally:
            if not completed:
                self._reset_recovery_diagnostics_release()

    def _begin_recovery_diagnostics_release(
        self,
        capability: object,
    ) -> tuple[str, dict[str, Any] | None]:
        if capability is not self.__recovery_diagnostics_release_capability:
            raise RuntimeError("invalid recovery diagnostics release capability")
        with self._admission_condition:
            recovery_reason = self._shutdown_reason
            if self._recovery_diagnostics_released:
                assert recovery_reason is not None
                result = self._recovery_diagnostics_release_result(
                    recovery_reason,
                    already_released=True,
                )
                if self._recovery_diagnostics_release_warnings:
                    result["warnings"] = list(
                        self._recovery_diagnostics_release_warnings
                    )
                return recovery_reason, result
            recovery_reason = self._validate_recovery_diagnostics_release_locked()
            self._recovery_diagnostics_releasing = True
            return recovery_reason, None

    def _preflight_recovery_diagnostics_release(
        self,
        capability: object,
        *,
        sync: bool,
    ) -> dict[str, Any] | None:
        if capability is not self.__recovery_diagnostics_release_capability:
            raise RuntimeError("invalid recovery diagnostics release capability")
        with self._admission_condition:
            if self._recovery_diagnostics_released:
                assert self._shutdown_reason is not None
                result = self._recovery_diagnostics_release_result(
                    self._shutdown_reason,
                    already_released=True,
                )
                if self._recovery_diagnostics_release_warnings:
                    result["warnings"] = list(
                        self._recovery_diagnostics_release_warnings
                    )
                return result
            recovery_reason = self._validate_recovery_diagnostics_release_locked()
        if sync and self._running_loop() is not None:
            raise RuntimeError(
                "synchronous recovery diagnostics release cannot run inside an "
                "active event loop; use await runtime.arelease_recovery_diagnostics()"
            )
        try:
            readiness = self._probe_admission_guard_close(
                self._admission_commit_guard_binding
            )
        except BaseException:
            # Another release can finish after this caller validates the
            # lifecycle but before it probes the store.  A closed backend then
            # rejects the stale probe; prefer the lifecycle's completed,
            # idempotent readback once ownership has actually transferred.
            with self._admission_condition:
                if not self._recovery_diagnostics_released:
                    raise
                assert self._shutdown_reason is not None
                result = self._recovery_diagnostics_release_result(
                    self._shutdown_reason,
                    already_released=True,
                )
                if self._recovery_diagnostics_release_warnings:
                    result["warnings"] = list(
                        self._recovery_diagnostics_release_warnings
                    )
                return result
        if readiness in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.LOCK_BUSY,
        }:
            return None
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            with self._lock:
                self._store_ownership_already_released = True
            return None
        if readiness is StoreCloseClaimOutcome.GUARD_MISMATCH:
            return self._recovery_diagnostics_release_failed(
                recovery_reason,
                "store",
                [],
            )
        raise RuntimeError(
            "recovery diagnostics store close is not ready: "
            f"{readiness.value}"
        )

    def _mark_recovery_diagnostics_release_started(self) -> None:
        with self._admission_condition:
            if not self._recovery_diagnostics_releasing:
                raise RuntimeError(
                    "recovery diagnostics release lost its active ownership"
                )
            self._recovery_diagnostics_release_started = True

    def _validate_recovery_diagnostics_release_locked(self) -> str:
        if self._recovery_required_result_locked() is None:
            raise RuntimeError(
                "recovery diagnostics release requires an active recovery fence"
            )
        if self._active_attempt is not None:
            raise RuntimeError(
                "recovery diagnostics release requires shutdown attempt completion"
            )
        if self._active_leases != 0:
            raise RuntimeError(
                "recovery diagnostics release requires admission drain"
            )
        if self._recovery_diagnostics_releasing:
            raise RuntimeError("recovery diagnostics release is already in progress")
        assert self._shutdown_reason is not None
        return self._shutdown_reason

    def _finish_recovery_diagnostics_release(
        self,
        recovery_reason: str,
        *,
        already_released: bool,
        warnings: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        with self._admission_condition:
            self._recovery_diagnostics_releasing = False
            self._recovery_diagnostics_released = True
            self._recovery_diagnostics_release_warnings = list(warnings or ())
            self._state = _LifecycleState.CLOSED
            self._admission_condition.notify_all()
        result = self._recovery_diagnostics_release_result(
            recovery_reason,
            already_released=already_released,
        )
        if warnings:
            result["warnings"] = list(warnings)
        return result

    def _store_close_outcome_released(
        self,
        outcome: StoreCloseOutcome,
        errors: list[dict[str, str]],
    ) -> tuple[bool, list[BaseException]]:
        if not outcome.guard_matched or not outcome.ownership_released:
            return False, []
        controls: list[BaseException] = []
        for warning in outcome.warnings:
            self._record_error(errors, "store", warning)
            if not isinstance(warning, Exception):
                controls.append(warning)
        return True, controls

    def _claim_recovery_store_close(
        self,
        errors: list[dict[str, str]],
    ) -> bool:
        outcome = self._claim_admission_guard_close(
            self._admission_commit_guard_binding
        )
        if outcome in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.OWNERSHIP_RELEASED,
        }:
            return True
        if outcome is not StoreCloseClaimOutcome.GUARD_MISMATCH:
            self._record_error(
                errors,
                "store",
                RuntimeError(
                    "recovery diagnostics store close claim failed: "
                    f"{outcome.value}"
                ),
            )
        return False

    @staticmethod
    def _raise_close_interrupts(
        *,
        interrupted: bool,
        controls: list[BaseException],
    ) -> None:
        pending: list[BaseException] = []
        if interrupted:
            pending.append(
                asyncio.CancelledError(
                    "store close completed after cancellation"
                )
            )
        pending.extend(controls)
        if not pending:
            return
        if len(pending) == 1:
            raise pending[0]
        raise BaseExceptionGroup(
            "store close completed with control-flow warnings",
            pending,
        )

    def _reset_recovery_diagnostics_release(self) -> None:
        with self._admission_condition:
            self._recovery_diagnostics_releasing = False
            self._admission_condition.notify_all()

    def _release_recovery_component_sync(
        self,
        name: str,
        component: Any,
    ) -> bool:
        if name == "object_tasks" and component is not None:
            release = getattr(component, "release_recovery_diagnostics", None)
            if not callable(release):
                raise RuntimeError(
                    "ObjectTask lifecycle handle does not support recovery diagnostics release"
                )
            return release() is not False
        if self._component_requires_async_shutdown(component):
            return asyncio.run(self.ashutdown_component(component))
        return self.shutdown_component(component)

    def _release_recovery_components_sync(
        self,
        components: tuple[tuple[str, Any], ...],
        recovery_reason: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        for name, component in components:
            try:
                released = self._release_recovery_component_sync(name, component)
            except Exception as exc:
                self._record_error(errors, name, exc)
                released = False
            if not released:
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    name,
                    errors,
                )
        return None

    async def _release_recovery_component_async(
        self,
        name: str,
        component: Any,
    ) -> bool:
        if name != "object_tasks" or component is None:
            released, interrupted = await self._await_recovery_release_step(
                self.ashutdown_component(component)
            )
            if interrupted:
                raise asyncio.CancelledError()
            return released
        release = getattr(component, "release_recovery_diagnostics", None)
        if not callable(release):
            raise RuntimeError(
                "ObjectTask lifecycle handle does not support recovery diagnostics release"
            )
        released, interrupted = await self._await_recovery_release_step(
            run_blocking_once(release)
        )
        if interrupted:
            raise asyncio.CancelledError()
        return released is not False

    async def _release_recovery_components_async(
        self,
        components: tuple[tuple[str, Any], ...],
        recovery_reason: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        for name, component in components:
            try:
                released = await self._release_recovery_component_async(
                    name,
                    component,
                )
            except Exception as exc:
                self._record_error(errors, name, exc)
                released = False
            if not released:
                return self._recovery_diagnostics_release_failed(
                    recovery_reason,
                    name,
                    errors,
                )
        return None

    @classmethod
    def _component_requires_async_shutdown(cls, component: Any) -> bool:
        if component is None:
            return False
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            return cls._is_async_callable(shutdown)
        close = getattr(component, "close", None)
        if callable(close):
            return cls._is_async_callable(close)
        return any(
            callable(getattr(component, name, None))
            for name in ("ashutdown", "aclose")
        )

    def _run_sync_recovery_finalizers(
        self,
        errors: list[dict[str, str]],
    ) -> str | None:
        for entry in self._finalizers:
            if entry.completed or not entry.recovery_safe:
                continue
            try:
                with self._recovery_cleanup_scope():
                    result = entry.callback()
                    if inspect.isawaitable(result):
                        if self._running_loop() is not None:
                            close = getattr(result, "close", None)
                            if callable(close):
                                close()
                            raise RuntimeError(
                                "async recovery cleanup requires await "
                                "runtime.arelease_recovery_diagnostics()"
                            )
                        result = asyncio.run(result)
                if result is False:
                    return entry.handle
                entry.completed = True
            except Exception as exc:
                self._record_error(errors, entry.handle, exc)
                return entry.handle
        return None

    async def _run_async_recovery_finalizers(
        self,
        errors: list[dict[str, str]],
    ) -> str | None:
        for entry in self._finalizers:
            if entry.completed or not entry.recovery_safe:
                continue
            try:
                with self._recovery_cleanup_scope():
                    interrupted = False
                    if self._is_async_callable(entry.callback):
                        result = entry.callback()
                    else:
                        result, interrupted = (
                            await self._await_recovery_release_step(
                                run_blocking_once(entry.callback)
                            )
                        )
                    if inspect.isawaitable(result):
                        try:
                            result, await_interrupted = (
                                await self._await_recovery_release_step(result)
                            )
                        except BaseException as exc:
                            if interrupted:
                                raise BaseExceptionGroup(
                                    "recovery cleanup was cancelled before its "
                                    "awaitable failed",
                                    [asyncio.CancelledError(), exc],
                                ) from None
                            raise
                        interrupted = interrupted or await_interrupted
                if interrupted and result is False:
                    raise asyncio.CancelledError()
                if result is False:
                    return entry.handle
                entry.completed = True
                if interrupted:
                    raise asyncio.CancelledError()
            except Exception as exc:
                self._record_error(errors, entry.handle, exc)
                return entry.handle
        return None

    @staticmethod
    async def _await_recovery_release_step(
        awaitable: Any,
    ) -> tuple[Any, bool]:
        """Drain one loop-affine cleanup before propagating caller cancellation."""

        task = asyncio.ensure_future(awaitable)
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                interrupted = True
                if task.done():
                    try:
                        return task.result(), True
                    except BaseException as exc:
                        raise BaseExceptionGroup(
                            "recovery cleanup was cancelled while its child failed",
                            [asyncio.CancelledError(), exc],
                        ) from None
                continue
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        "recovery cleanup was cancelled while its child failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            return result, interrupted

    @staticmethod
    async def _await_recovery_release_commit(
        awaitable: Any,
    ) -> tuple[Any, bool]:
        """Drain the irreversible store handoff and report deferred cancellation."""

        task = asyncio.ensure_future(awaitable)
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                interrupted = True
                if task.done():
                    try:
                        return task.result(), True
                    except BaseException as exc:
                        raise BaseExceptionGroup(
                            "store handoff was cancelled while close failed",
                            [asyncio.CancelledError(), exc],
                        ) from None
                continue
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        "store handoff was cancelled while close failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            return result, interrupted

    def _recovery_diagnostics_release_failed(
        self,
        recovery_reason: str,
        component: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "already_released": False,
            "reason": recovery_reason,
            "recovery_required": True,
            f"{component}_released": False,
        }
        if errors:
            result["errors"] = list(errors)
        return result

    @staticmethod
    def _recovery_diagnostics_release_result(
        recovery_reason: str,
        *,
        already_released: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "already_released": already_released,
            "reason": recovery_reason,
            "recovery_required": True,
            "recovery_diagnostics_released": True,
        }

    def _issue_recovery_terminalization_capability(self) -> object:
        """Return the opaque builder capability for fenced terminalization."""

        return self.__recovery_terminalization_capability

    @contextmanager
    def recovery_terminalization_scope(
        self,
        publication_id: str,
        *,
        capability: object,
    ) -> Iterator[None]:
        """Permit only trusted terminalization for the publication that fenced.

        The scope records intent in a ContextVar but deliberately releases the
        lifecycle condition before caller work begins.  The store keeps the
        canonical store -> lifecycle lock order and revalidates the exact
        publication binding again at each outer commit.
        """

        if capability is not self.__recovery_terminalization_capability:
            raise RuntimeError("invalid recovery terminalization capability")
        selected_publication = self._require_publication_id(publication_id)
        lease = self._current_admission.get()
        with self._admission_condition:
            self._validate_recovery_terminalization_locked(
                lease,
                selected_publication,
            )
        inherited = self._recovery_terminalization_publication.get()
        if inherited is not None and inherited != selected_publication:
            raise RuntimeError(
                "nested recovery terminalization changed publication identity"
            )
        reset = self._recovery_terminalization_publication.set(
            selected_publication
        )
        try:
            yield
        finally:
            self._recovery_terminalization_publication.reset(reset)

    @contextmanager
    def recovery_terminalization_scope_if_fenced(
        self,
        publication_id: str,
        *,
        capability: object,
    ) -> Iterator[None]:
        """Enable trusted terminalization only when this lease was fenced.

        Operation evidence scopes can run both before and after a recovery
        fence.  Before a fence this context grants nothing and the ordinary
        admission commit guard remains authoritative.  After a fence it uses
        the same exact-publication validation as the strict recovery scope.
        """

        if capability is not self.__recovery_terminalization_capability:
            raise RuntimeError("invalid recovery terminalization capability")
        selected_publication = self._require_publication_id(publication_id)
        lease = self._current_admission.get()
        with self._admission_condition:
            fenced = bool(
                lease is not None
                and lease.active
                and lease.recovery_fence_epoch != self._recovery_fence_epoch
            )
            if fenced:
                self._validate_recovery_terminalization_locked(
                    lease,
                    selected_publication,
                )
        if not fenced:
            yield
            return
        inherited = self._recovery_terminalization_publication.get()
        if inherited is not None and inherited != selected_publication:
            raise RuntimeError(
                "nested recovery terminalization changed publication identity"
            )
        reset = self._recovery_terminalization_publication.set(
            selected_publication
        )
        try:
            yield
        finally:
            self._recovery_terminalization_publication.reset(reset)

    @contextmanager
    def admission_commit_guard(self) -> Iterator[None]:
        """Atomically revalidate an admission epoch with an outer store commit.

        SQLRuntimeStore enters this guard while it already owns the store/UoW
        lock and keeps it held through ``commit``.  The fixed lock order is
        therefore store -> lifecycle condition.  Recovery fencing only needs
        the lifecycle condition and never takes the store lock, so the two
        paths cannot form a lock cycle: either the fence advances first and the
        transaction rolls back, or the admitted commit completes first.
        """

        startup_open = self._startup_open_commit.get()
        if startup_open is not None:
            with self._startup_open_commit_guard(startup_open):
                yield
            return
        lease = self._current_admission.get()
        if lease is None:
            with self._admission_condition:
                if (
                    self._recovery_diagnostics_release_started
                    or self._recovery_required_result_locked() is not None
                ):
                    raise RuntimeError(
                        "durable mutation without an admitted recovery scope is "
                        "unavailable while recovery is required"
                    )
                yield
                return
        with self._admission_condition:
            if self._recovery_diagnostics_release_started:
                raise RuntimeError(
                    "durable mutation is unavailable after recovery diagnostics "
                    "release starts"
                )
            terminalization = self._recovery_terminalization_publication.get()
            if terminalization is None:
                self._revalidate_admission(lease, read_only=False)
            else:
                self._validate_recovery_terminalization_locked(
                    lease,
                    terminalization,
                )
            yield

    @contextmanager
    def _startup_open_commit_guard(
        self,
        attempt: _StartupOpenCommit,
    ) -> Iterator[None]:
        """Keep admission closed while one Store commit publishes OPEN."""

        with self._admission_condition:
            if (
                not attempt.active
                or attempt.owner_thread_id != threading.get_ident()
                or attempt.owner_task is not self._current_task()
                or self._state is not _LifecycleState.STARTING
                or not self._current_startup_lease_is_valid_locked()
                or self._startup_open_commit.get() is not attempt
                or self._current_admission.get() is not None
            ):
                raise RuntimeError(
                    "startup OPEN commit lost its exact lifecycle scope"
                )
            if attempt.consumed:
                raise RuntimeError(
                    "startup OPEN commit scope was consumed more than once"
                )
            if (
                self._recovery_diagnostics_release_started
                or self._recovery_required_result_locked() is not None
            ):
                raise RuntimeError(
                    "startup OPEN commit is unavailable while recovery is required"
                )
            attempt.consumed = True
            try:
                # The Store lock is already held. mark_open is a pure in-memory
                # transition, and this condition remains held through commit so
                # no admission can observe OPEN before the durable ack commits.
                self.mark_open()
                if (
                    self._state is not _LifecycleState.OPEN
                    or not self._ever_opened
                ):
                    raise RuntimeError(
                        "startup OPEN transition did not publish exact lifecycle state"
                    )
                yield
            except BaseException:
                if self._state is _LifecycleState.OPEN:
                    self._state = _LifecycleState.STARTING
                    self._ever_opened = False
                elif self._state is not _LifecycleState.STARTING:
                    raise RuntimeError(
                        "startup OPEN commit failed after lifecycle state drift"
                    )
                raise
            else:
                attempt.committed = True

    def _validate_recovery_terminalization_locked(
        self,
        lease: _AdmissionLease | None,
        publication_id: str,
    ) -> None:
        if lease is None or not lease.active:
            raise RuntimeError(
                "recovery terminalization requires an active admission lease"
            )
        if lease.read_only:
            raise RuntimeError(
                "read-only runtime admission lease cannot authorize mutation"
            )
        if lease.recovery_fence_epoch == self._recovery_fence_epoch:
            raise RuntimeError(
                "recovery terminalization requires a fenced admission lease"
            )
        expected_reason = f"runtime.recovery_required:{publication_id}"
        if (
            self._state is not _LifecycleState.CLOSE_FAILED
            or self._shutdown_reason != expected_reason
        ):
            raise RuntimeError(
                "recovery terminalization does not match the active recovery fence"
            )

    def _revalidate_admission(
        self,
        lease: _AdmissionLease,
        *,
        read_only: bool,
    ) -> None:
        with self._admission_condition:
            if not lease.active:
                raise RuntimeError("runtime admission lease is no longer active")
            if not read_only and lease.read_only:
                raise RuntimeError(
                    "read-only runtime admission lease cannot authorize mutation"
                )
            if read_only and self._state is not _LifecycleState.CLOSED:
                return
            if lease.recovery_fence_epoch == self._recovery_fence_epoch:
                return
            raise RuntimeError(
                f"runtime is not accepting operations: state={self._state.value}"
            )

    def bind_components(
        self,
        *,
        scheduler: Any,
        object_tasks: Any,
        modules: Any,
        llms: Any,
        blocking_work: Any | None = None,
    ) -> None:
        with self._lock:
            if self._components_bound:
                raise RuntimeError("runtime lifecycle components are already bound")
            if self._active_attempt is not None or self._state in {
                _LifecycleState.STOPPING,
                _LifecycleState.CLOSE_FAILED,
                _LifecycleState.CLOSED,
            }:
                raise RuntimeError("cannot bind runtime components after shutdown has started")
            self._scheduler = scheduler
            self._object_tasks = object_tasks
            self._modules = modules
            self._llms = llms
            self._blocking_work = blocking_work
            self._components_bound = True

    def finalizers_snapshot(self) -> tuple[Any, ...]:
        with self._lock:
            return tuple(entry.callback for entry in self._finalizers)

    def bind_finalizer(
        self,
        finalizer: Any,
        *,
        recovery_safe: bool = False,
    ) -> None:
        if type(recovery_safe) is not bool:
            raise TypeError(
                "runtime finalizer recovery_safe must be an exact boolean"
            )
        with self._lock:
            if self._active_attempt is not None or self._state in {
                _LifecycleState.STOPPING,
                _LifecycleState.CLOSE_FAILED,
                _LifecycleState.CLOSED,
            }:
                raise RuntimeError("cannot bind a shutdown finalizer after shutdown has started")
            self._finalizers.append(
                _FinalizerEntry(
                    handle=new_id("finalizer"),
                    callback=finalizer,
                    recovery_safe=recovery_safe,
                )
            )

    def unbind_finalizer(self, finalizer: Any) -> bool:
        with self._lock:
            for index in range(len(self._finalizers) - 1, -1, -1):
                if self._finalizers[index].callback is finalizer:
                    del self._finalizers[index]
                    return True
        return False

    def _unbind_finalizer_entry(self, target: _FinalizerEntry) -> bool:
        """Remove one exact registration without changing public callback semantics."""

        with self._lock:
            for index in range(len(self._finalizers) - 1, -1, -1):
                if self._finalizers[index] is target:
                    del self._finalizers[index]
                    return True
        return False

    def shutdown(
        self,
        *,
        actor: str = "runtime",
        reason: str = "runtime.shutdown",
    ) -> dict[str, Any]:
        self._preflight_ordinary_shutdown_store()
        with self._lock:
            if self._state is _LifecycleState.CLOSED:
                return self._closed_shutdown_result_locked(
                    already_shutdown=True,
                )
        # Coroutine finalizers cannot be synchronously driven by the thread
        # that already owns an event loop. Refuse before closing admission.
        if self._running_loop() is not None and (
            any(
                not entry.completed and self._is_async_callable(entry.callback)
                for entry in self._finalizers
            )
            or any(
                self._component_requires_async_shutdown(component)
                for component in (
                    self._scheduler,
                    self._object_tasks,
                    self._modules,
                    self._llms,
                    self._blocking_work,
                    self._substrate,
                )
            )
        ):
            recovery_required = self._recovery_required_result()
            if recovery_required is not None:
                return recovery_required
            raise RuntimeError(
                "runtime has async-only shutdown work; use await runtime.ashutdown()"
            )
        caller_task = self._current_task()
        attempt, is_leader, early = self._start_attempt(caller_task=caller_task)
        if early is not None:
            return early
        assert attempt is not None
        if not is_leader:
            self._reject_reentrant_wait(attempt, caller_task=caller_task, async_wait=False)
            attempt.done.wait()
            return self._attempt_result(attempt)
        context_token = self._shutdown_attempt_context.set(attempt)
        try:
            result = self._shutdown_sync(actor=actor, reason=reason)
        except BaseException as exc:
            with self._lock:
                if self._state is _LifecycleState.CLOSED:
                    closed_result = self._closed_shutdown_result_locked(
                        already_shutdown=False,
                    )
                else:
                    self._state = _LifecycleState.CLOSE_FAILED
                    closed_result = None
            if closed_result is None:
                self._complete_attempt(attempt, error=exc)
            else:
                # Control-flow exceptions reported after irreversible backend
                # release belong only to the initiating caller. Followers
                # observe the shared terminal shutdown result.
                self._complete_attempt(attempt, result=closed_result)
            raise
        else:
            self._complete_attempt(attempt, result=result)
            return result
        finally:
            self._shutdown_attempt_context.reset(context_token)

    async def ashutdown(
        self,
        *,
        actor: str = "runtime",
        reason: str = "runtime.shutdown",
    ) -> dict[str, Any]:
        self._preflight_ordinary_shutdown_store()
        caller_task = self._current_task()
        attempt, is_leader, early = self._start_attempt(caller_task=caller_task)
        if early is not None:
            return early
        assert attempt is not None
        if not is_leader:
            self._reject_reentrant_wait(attempt, caller_task=caller_task, async_wait=True)
            await run_blocking_once(attempt.done.wait)
            return self._attempt_result(attempt)
        context_token = self._shutdown_attempt_context.set(attempt)
        try:
            result = await self._shutdown_async(actor=actor, reason=reason)
        except BaseException as exc:
            with self._lock:
                if self._state is _LifecycleState.CLOSED:
                    closed_result = self._closed_shutdown_result_locked(
                        already_shutdown=False,
                    )
                else:
                    self._state = _LifecycleState.CLOSE_FAILED
                    closed_result = None
            if closed_result is None:
                self._complete_attempt(attempt, error=exc)
            else:
                self._complete_attempt(attempt, result=closed_result)
            raise
        else:
            self._complete_attempt(attempt, result=result)
            return result
        finally:
            self._shutdown_attempt_context.reset(context_token)

    def _shutdown_sync(self, *, actor: str, reason: str) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        if not self._drain_admission():
            # Active admissions may still establish a recovery fence after
            # the timeout.  Component shutdown can write cancellation evidence,
            # so it cannot be linearized safely against such a future fence
            # without holding the lifecycle lock across arbitrary callbacks.
            # Fail this attempt closed and let the caller retry after leases
            # have actually drained.
            return self._admission_timeout_result(reason, errors)
        recovery_required = self._recovery_required_result()
        if recovery_required is not None:
            return recovery_required
        if self._store_ownership_released_before_shutdown(reason, errors):
            pass
        elif not self._record_shutdown(actor=actor, reason=reason, errors=errors):
            return self._failed(reason, "shutdown_evidence", errors)
        for name, component in (("scheduler", self._scheduler), ("object_tasks", self._object_tasks)):
            if not self._stop_sync_component(name, component, errors):
                return self._failed(reason, name, errors)
        failed_finalizer = self._run_sync_finalizers(errors)
        if failed_finalizer is not None:
            return self._failed(reason, failed_finalizer, errors)
        for name, component in (
            ("modules", self._modules),
            ("llms", self._llms),
            ("blocking_work", self._blocking_work),
            ("substrate", self._substrate),
        ):
            if not self._stop_sync_component(name, component, errors):
                return self._failed(reason, name, errors)
        return self._close_store(reason, errors)

    async def _shutdown_async(self, *, actor: str, reason: str) -> dict[str, Any]:
        errors: list[dict[str, str]] = []
        if not await run_blocking_once(self._drain_admission):
            # Keep the async timeout path identical to the sync linearization:
            # no component callback is safe while an admitted operation can
            # still advance the recovery fence.
            return self._admission_timeout_result(reason, errors)
        recovery_required = self._recovery_required_result()
        if recovery_required is not None:
            return recovery_required
        if self._store_ownership_released_before_shutdown(reason, errors):
            pass
        elif not self._record_shutdown(actor=actor, reason=reason, errors=errors):
            return self._failed(reason, "shutdown_evidence", errors)
        for name, component in (("scheduler", self._scheduler), ("object_tasks", self._object_tasks)):
            if not await self._stop_async_component(name, component, errors):
                return self._failed(reason, name, errors)
        failed_finalizer = await self._run_async_finalizers(errors)
        if failed_finalizer is not None:
            return self._failed(reason, failed_finalizer, errors)
        for name, component in (
            ("modules", self._modules),
            ("llms", self._llms),
            ("blocking_work", self._blocking_work),
            ("substrate", self._substrate),
        ):
            if not await self._stop_async_component(name, component, errors):
                return self._failed(reason, name, errors)
        return await self._aclose_store(reason, errors)

    def cleanup_failed_assembly(self) -> list[dict[str, str]]:
        """Run failed-assembly teardown without closing the diagnostic store.

        A synchronous caller cannot safely drive loop-affine async finalizers
        while its thread already owns an event loop. Async hosts must use the
        awaited assembly API, which calls :meth:`acleanup_failed_assembly` on
        that same loop.
        """

        self._require_failed_assembly_cleanup_eligible()
        if self._running_loop() is not None:
            raise RuntimeError(
                "failed assembly cleanup requires await acleanup_failed_assembly() "
                "inside an active event loop"
            )
        caller_task = self._current_task()
        attempt, is_leader = self._start_failed_assembly_cleanup_attempt(
            caller_task=caller_task,
        )
        if not is_leader:
            self._reject_reentrant_failed_assembly_cleanup_wait(
                attempt,
                caller_task=caller_task,
                async_wait=False,
            )
            attempt.done.wait()
            return self._failed_assembly_cleanup_attempt_result(attempt)
        context_token = self._failed_assembly_cleanup_context.set(attempt)
        try:
            try:
                errors, caught = self._cleanup_failed_assembly_sync()
            except BaseException as exc:
                errors = []
                self._record_error(errors, "failed_assembly_cleanup", exc)
                self._finish_failed_assembly_cleanup()
                self._complete_failed_assembly_cleanup_attempt(
                    attempt,
                    errors,
                )
                raise
            self._complete_failed_assembly_cleanup_attempt(attempt, errors)
            self._raise_failed_assembly_interrupts(caught)
            return errors
        finally:
            self._failed_assembly_cleanup_context.reset(context_token)

    def _cleanup_failed_assembly_sync(
        self,
    ) -> tuple[list[dict[str, str]], list[BaseException]]:
        """Synchronous failed-assembly cleanup for ordinary sync hosts."""

        errors: list[dict[str, str]] = []
        caught: list[BaseException] = []
        for name, component in (("scheduler", self._scheduler), ("object_tasks", self._object_tasks)):
            self._stop_failed_assembly_sync_component(
                name,
                component,
                errors,
                caught,
            )
        # A completed finalizer is removed from the partial graph. Deferred or
        # failed finalizers remain bound so RuntimeAssemblyCleanupRequired can
        # retry them before an owned store is closed.
        for entry in list(self._finalizers):
            if entry.completed:
                self._unbind_finalizer_entry(entry)
                continue
            try:
                result = entry.callback()
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            except BaseException as exc:
                self._record_error(errors, entry.handle, exc)
                caught.append(exc)
                continue
            if result is False:
                errors.append(
                    {
                        "component": entry.handle,
                        "error_type": "FinalizerDeferred",
                        "error": "returned false",
                    }
                )
                continue
            entry.completed = True
            self._unbind_finalizer_entry(entry)
        for name, component in (
            ("modules", self._modules),
            ("llms", self._llms),
            ("blocking_work", self._blocking_work),
            ("substrate", self._substrate),
        ):
            self._stop_failed_assembly_sync_component(
                name,
                component,
                errors,
                caught,
            )
        self._finish_failed_assembly_cleanup()
        if not errors:
            self._release_failed_assembly_admission_commit_guard(
                errors,
                caught,
            )
        return errors, caught

    async def acleanup_failed_assembly(self) -> list[dict[str, str]]:
        """Await failed-assembly teardown in an async host.

        This mirrors :meth:`ashutdown` component selection but intentionally
        leaves the store open.  A builder that opened the store owns the later
        close; callers that supplied a store retain diagnostic access.
        """

        caller_task = self._current_task()
        attempt, is_leader = self._start_failed_assembly_cleanup_attempt(
            caller_task=caller_task,
        )
        if not is_leader:
            self._reject_reentrant_failed_assembly_cleanup_wait(
                attempt,
                caller_task=caller_task,
                async_wait=True,
            )
            await run_blocking_once(attempt.done.wait)
            return self._failed_assembly_cleanup_attempt_result(attempt)
        context_token = self._failed_assembly_cleanup_context.set(attempt)
        try:
            try:
                errors, caught = await self._cleanup_failed_assembly_async()
            except BaseException as exc:
                errors = []
                self._record_error(errors, "failed_assembly_cleanup", exc)
                self._finish_failed_assembly_cleanup()
                self._complete_failed_assembly_cleanup_attempt(
                    attempt,
                    errors,
                )
                raise
            self._complete_failed_assembly_cleanup_attempt(attempt, errors)
            self._raise_failed_assembly_interrupts(caught)
            return errors
        finally:
            self._failed_assembly_cleanup_context.reset(context_token)

    async def _cleanup_failed_assembly_async(
        self,
    ) -> tuple[list[dict[str, str]], list[BaseException]]:
        errors: list[dict[str, str]] = []
        caught: list[BaseException] = []
        for name, component in (("scheduler", self._scheduler), ("object_tasks", self._object_tasks)):
            await self._stop_failed_assembly_async_component(
                name,
                component,
                errors,
                caught,
            )
        for entry in list(self._finalizers):
            if entry.completed:
                self._unbind_finalizer_entry(entry)
                continue
            try:
                result, interrupted = await self._invoke_async_cleanup(
                    entry.callback,
                    description="failed assembly finalizer",
                )
            except BaseException as exc:
                self._record_error(errors, entry.handle, exc)
                caught.append(exc)
                continue
            if result is False:
                errors.append(
                    {
                        "component": entry.handle,
                        "error_type": "FinalizerDeferred",
                        "error": "returned false",
                    }
                )
                if interrupted:
                    caught.append(asyncio.CancelledError())
                continue
            entry.completed = True
            self._unbind_finalizer_entry(entry)
            if interrupted:
                caught.append(asyncio.CancelledError())
        for name, component in (
            ("modules", self._modules),
            ("llms", self._llms),
            ("blocking_work", self._blocking_work),
            ("substrate", self._substrate),
        ):
            await self._stop_failed_assembly_async_component(
                name,
                component,
                errors,
                caught,
            )
        self._finish_failed_assembly_cleanup()
        if not errors:
            self._release_failed_assembly_admission_commit_guard(
                errors,
                caught,
            )
        return errors, caught

    def _start_failed_assembly_cleanup_attempt(
        self,
        *,
        caller_task: asyncio.Task[Any] | None,
    ) -> tuple[_FailedAssemblyCleanupAttempt, bool]:
        with self._lock:
            if self._ever_opened:
                raise RuntimeError(
                    "failed assembly cleanup is unavailable after Runtime open; "
                    "use release_recovery_diagnostics() for a recovery fence"
                )
            active = self._active_failed_assembly_cleanup
            if active is not None:
                return active, False
            attempt = _FailedAssemblyCleanupAttempt(
                owner_thread_id=threading.get_ident(),
                owner_task=caller_task,
            )
            self._active_failed_assembly_cleanup = attempt
            if self._state is not _LifecycleState.CLOSED:
                self._state = _LifecycleState.STOPPING
            return attempt, True

    def _require_failed_assembly_cleanup_eligible(self) -> None:
        with self._lock:
            if self._ever_opened:
                raise RuntimeError(
                    "failed assembly cleanup is unavailable after Runtime open; "
                    "use release_recovery_diagnostics() for a recovery fence"
                )

    def _reject_reentrant_failed_assembly_cleanup_wait(
        self,
        attempt: _FailedAssemblyCleanupAttempt,
        *,
        caller_task: asyncio.Task[Any] | None,
        async_wait: bool,
    ) -> None:
        same_thread = attempt.owner_thread_id == threading.get_ident()
        same_task = caller_task is not None and caller_task is attempt.owner_task
        different_async_task = (
            async_wait
            and same_thread
            and caller_task is not None
            and attempt.owner_task is not None
            and not same_task
        )
        inherited_attempt = (
            self._failed_assembly_cleanup_context.get() is attempt
        )
        if inherited_attempt or same_task or (same_thread and not different_async_task):
            raise RuntimeError(
                "reentrant failed assembly cleanup cannot wait for its own attempt"
            )

    def _complete_failed_assembly_cleanup_attempt(
        self,
        attempt: _FailedAssemblyCleanupAttempt,
        result: list[dict[str, str]],
    ) -> None:
        with self._lock:
            attempt.result = [dict(item) for item in result]
            if self._active_failed_assembly_cleanup is attempt:
                self._active_failed_assembly_cleanup = None
            attempt.done.set()

    @staticmethod
    def _failed_assembly_cleanup_attempt_result(
        attempt: _FailedAssemblyCleanupAttempt,
    ) -> list[dict[str, str]]:
        if attempt.result is None:
            raise RuntimeError(
                "failed assembly cleanup completed without a shared outcome"
            )
        return [dict(item) for item in attempt.result]

    def _finish_failed_assembly_cleanup(self) -> None:
        with self._lock:
            if self._state is not _LifecycleState.CLOSED:
                self._state = _LifecycleState.CLOSE_FAILED

    def _release_failed_assembly_admission_commit_guard(
        self,
        errors: list[dict[str, str]],
        caught: list[BaseException],
    ) -> None:
        """Return this failed graph's exact store guard after full teardown."""

        try:
            self._unbind_admission_commit_guard(
                self._admission_commit_guard_binding
            )
        except BaseException as exc:
            self._record_error(errors, "admission_commit_guard", exc)
            caught.append(exc)

    def _stop_failed_assembly_sync_component(
        self,
        name: str,
        component: Any,
        errors: list[dict[str, str]],
        caught: list[BaseException],
    ) -> bool:
        try:
            if self._component_requires_async_shutdown(component):
                stopped = asyncio.run(self.ashutdown_component(component))
            else:
                stopped = self.shutdown_component(component)
            if not stopped:
                errors.append(
                    {
                        "component": name,
                        "error_type": "ComponentStopDeferred",
                        "error": "returned false",
                    }
                )
            return stopped
        except BaseException as exc:
            self._record_error(errors, name, exc)
            caught.append(exc)
            return False

    async def _stop_failed_assembly_async_component(
        self,
        name: str,
        component: Any,
        errors: list[dict[str, str]],
        caught: list[BaseException],
    ) -> bool:
        try:
            stopped = await self.ashutdown_component(component)
            if not stopped:
                errors.append(
                    {
                        "component": name,
                        "error_type": "ComponentStopDeferred",
                        "error": "returned false",
                    }
                )
            return stopped
        except BaseException as exc:
            self._record_error(errors, name, exc)
            caught.append(exc)
            return False

    @staticmethod
    def _raise_failed_assembly_interrupts(caught: list[BaseException]) -> None:
        if any(not isinstance(exc, Exception) for exc in caught):
            raise BaseExceptionGroup(
                "failed assembly cleanup was interrupted after full teardown",
                caught,
            )

    def _start_attempt(
        self,
        *,
        caller_task: asyncio.Task[Any] | None,
    ) -> tuple[_ShutdownAttempt | None, bool, dict[str, Any] | None]:
        with self._lock:
            if self._state is _LifecycleState.CLOSED:
                return None, False, self._closed_shutdown_result_locked(
                    already_shutdown=True,
                )
            if self._active_attempt is not None:
                return self._active_attempt, False, None
            recovery_required = self._recovery_required_result_locked()
            if recovery_required is not None:
                return None, False, recovery_required
            attempt = _ShutdownAttempt(
                owner_thread_id=threading.get_ident(),
                owner_task=caller_task,
            )
            self._active_attempt = attempt
            self._state = _LifecycleState.STOPPING
            self._shutdown_reason = self._shutdown_reason or "runtime.shutdown"
            return attempt, True, None

    def _preflight_ordinary_shutdown_store(self) -> None:
        with self._admission_condition:
            if (
                self._state is _LifecycleState.CLOSED
                or self._recovery_required_result_locked() is not None
            ):
                return
        try:
            readiness = self._probe_admission_guard_close(
                self._admission_commit_guard_binding
            )
        except BaseException:
            with self._admission_condition:
                if (
                    self._state is _LifecycleState.CLOSED
                    or self._active_attempt is not None
                    or self._recovery_required_result_locked() is not None
                ):
                    return
            raise
        if readiness in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.LOCK_BUSY,
        }:
            return
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            with self._lock:
                self._store_ownership_already_released = True
            return
        if readiness in {
            StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED,
            StoreCloseClaimOutcome.ACTIVE_TRANSACTION,
        }:
            raise RuntimeError(
                "runtime shutdown store close is not ready: "
                f"{readiness.value}"
            )
        with self._admission_condition:
            if self._active_attempt is not None:
                return
        raise RuntimeError(
            "runtime shutdown store close is not ready: "
            f"{readiness.value}"
        )

    def _closed_shutdown_result_locked(
        self,
        *,
        already_shutdown: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True,
            "already_shutdown": already_shutdown,
            "reason": self._shutdown_reason,
        }
        if self._shutdown_warnings:
            result["warnings"] = list(self._shutdown_warnings)
        return result

    def _store_ownership_released_before_shutdown(
        self,
        reason: str,
        errors: list[dict[str, str]],
    ) -> bool:
        with self._lock:
            released = self._store_ownership_already_released
            if released:
                self._shutdown_reason = reason
        if not released:
            return False
        self._record_error(
            errors,
            "shutdown_evidence",
            RuntimeError(
                "runtime store ownership was already released before shutdown evidence"
            ),
        )
        return True

    def _drain_admission(self) -> bool:
        deadline = time.monotonic() + self._admission_drain_timeout_s
        with self._admission_condition:
            while self._active_leases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._admission_condition.wait(timeout=remaining)
            return True

    def _recovery_required_result(self) -> dict[str, Any] | None:
        """Read the monotonic recovery fence before any teardown action."""

        with self._admission_condition:
            return self._recovery_required_result_locked()

    def _recovery_required_result_locked(self) -> dict[str, Any] | None:
        """Return the fail-closed result while the lifecycle lock is owned.

        A recovery fence is monotonic for this RuntimeLifecycle instance.  A
        later ordinary shutdown attempt must not relabel CLOSE_FAILED as
        STOPPING and thereby erase the diagnostic reason or close its store.
        Opening a recovered Runtime creates a new lifecycle and latch.
        """

        recovery_reason = self._shutdown_reason
        if (
            self._recovery_fence_epoch <= 0
            or self._state is not _LifecycleState.CLOSE_FAILED
            or recovery_reason is None
            or not recovery_reason.startswith(_RECOVERY_REQUIRED_REASON_PREFIX)
        ):
            return None
        return {
            "ok": False,
            "already_shutdown": False,
            "reason": recovery_reason,
            "recovery_required": True,
        }

    def _admission_timeout_result(
        self,
        reason: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Linearize timeout failure against a late recovery fence.

        The condition protects both outcomes: either the recovery latch wins
        and its diagnostic result is returned, or this attempt becomes an
        ordinary retryable admission timeout before a later lease can fence.
        Neither outcome invokes a component callback while admissions remain.
        """

        with self._admission_condition:
            recovery_required = self._recovery_required_result_locked()
            if recovery_required is not None:
                return recovery_required
            return self._failed(reason, "admission", errors)

    def _record_shutdown(
        self,
        *,
        actor: str,
        reason: str,
        errors: list[dict[str, str]],
    ) -> bool:
        self._shutdown_reason = reason
        try:
            with self._store.transaction():
                self._audit.record(
                    actor=actor,
                    action="runtime.shutdown",
                    target="runtime",
                    decision={"reason": reason},
                )
                self._events.emit(
                    EventType.RUNTIME_SHUTDOWN,
                    source=actor,
                    target="runtime",
                    payload={"reason": reason},
                )
        except Exception as exc:
            self._record_error(errors, "shutdown_evidence", exc)
            return False
        return True

    def _run_sync_finalizers(self, errors: list[dict[str, str]]) -> str | None:
        for entry in self._finalizers:
            if entry.completed:
                continue
            try:
                result = entry.callback()
                if inspect.isawaitable(result):
                    if self._running_loop() is not None:
                        close = getattr(result, "close", None)
                        if callable(close):
                            close()
                        raise RuntimeError(
                            "async shutdown finalizer requires await runtime.ashutdown()"
                        )
                    result = asyncio.run(result)
                if result is False:
                    return entry.handle
                entry.completed = True
            except Exception as exc:
                self._record_error(errors, entry.handle, exc)
                return entry.handle
        return None

    async def _run_async_finalizers(self, errors: list[dict[str, str]]) -> str | None:
        for entry in self._finalizers:
            if entry.completed:
                continue
            try:
                result, interrupted = await self._invoke_async_cleanup(
                    entry.callback,
                    description="runtime shutdown finalizer",
                )
                if interrupted and result is False:
                    raise asyncio.CancelledError()
                if result is False:
                    return entry.handle
                entry.completed = True
                if interrupted:
                    raise asyncio.CancelledError()
            except Exception as exc:
                self._record_error(errors, entry.handle, exc)
                return entry.handle
        return None

    @classmethod
    async def _invoke_async_cleanup(
        cls,
        callback: Any,
        *,
        description: str,
    ) -> tuple[Any, bool]:
        """Invoke one cleanup callback without blocking or abandoning its work.

        Native async callbacks are created on the caller's loop so loop-affine
        clients retain their ownership semantics. Synchronous callbacks run in
        a one-call owned executor. Caller cancellation is deferred until the
        selected callback (and any awaitable it returns) has settled.
        """

        interrupted = False
        if cls._is_async_callable(callback):
            result = callback()
        else:
            result, interrupted = await cls._await_cleanup_step(
                run_blocking_once(callback),
                description=description,
            )
        if inspect.isawaitable(result):
            try:
                result, await_interrupted = await cls._await_cleanup_step(
                    result,
                    description=description,
                )
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        f"{description} was cancelled before its awaitable failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            interrupted = interrupted or await_interrupted
        return result, interrupted

    @staticmethod
    async def _await_cleanup_step(
        awaitable: Any,
        *,
        description: str,
    ) -> tuple[Any, bool]:
        """Drain one cleanup awaitable before reporting caller cancellation."""

        task = asyncio.ensure_future(awaitable)
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                interrupted = True
                if task.done():
                    try:
                        return task.result(), True
                    except BaseException as exc:
                        raise BaseExceptionGroup(
                            f"{description} was cancelled while its child failed",
                            [asyncio.CancelledError(), exc],
                        ) from None
                continue
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        f"{description} was cancelled while its child failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            return result, interrupted

    def _close_store(self, reason: str, errors: list[dict[str, str]]) -> dict[str, Any]:
        claim = self._claim_admission_guard_close(
            self._admission_commit_guard_binding
        )
        if claim not in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.OWNERSHIP_RELEASED,
        }:
            self._record_error(
                errors,
                "store",
                RuntimeError(f"runtime store close claim failed: {claim.value}"),
            )
            return self._failed(reason, "store", errors)
        try:
            outcome = self._release_admission_guard_and_close(
                self._admission_commit_guard_binding
            )
        except Exception as exc:
            self._record_error(errors, "store", exc)
            return self._failed(reason, "store", errors)
        released, controls = self._store_close_outcome_released(outcome, errors)
        if not released:
            return self._failed(reason, "store", errors)
        with self._lock:
            self._state = _LifecycleState.CLOSED
            self._shutdown_warnings = list(errors)
            result = self._closed_shutdown_result_locked(
                already_shutdown=False,
            )
        self._raise_close_interrupts(interrupted=False, controls=controls)
        return result

    async def _aclose_store(
        self,
        reason: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        claim = self._claim_admission_guard_close(
            self._admission_commit_guard_binding
        )
        if claim not in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.OWNERSHIP_RELEASED,
        }:
            self._record_error(
                errors,
                "store",
                RuntimeError(f"runtime store close claim failed: {claim.value}"),
            )
            return self._failed(reason, "store", errors)
        try:
            outcome, interrupted = await self._await_recovery_release_commit(
                run_blocking_once(
                    self._release_admission_guard_and_close,
                    self._admission_commit_guard_binding,
                )
            )
        except Exception as exc:
            self._record_error(errors, "store", exc)
            return self._failed(reason, "store", errors)
        released, controls = self._store_close_outcome_released(outcome, errors)
        if not released:
            if interrupted:
                raise asyncio.CancelledError()
            return self._failed(reason, "store", errors)
        with self._lock:
            self._state = _LifecycleState.CLOSED
            self._shutdown_warnings = list(errors)
            result = self._closed_shutdown_result_locked(
                already_shutdown=False,
            )
        self._raise_close_interrupts(
            interrupted=interrupted,
            controls=controls,
        )
        return result

    def _failed(
        self,
        reason: str,
        component: str,
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        with self._lock:
            self._state = _LifecycleState.CLOSE_FAILED
        result: dict[str, Any] = {
            "ok": False,
            "already_shutdown": False,
            "reason": reason,
            f"{component}_stopped": False,
        }
        if errors:
            result["errors"] = list(errors)
        return result

    def _stop_sync_component(
        self,
        name: str,
        component: Any,
        errors: list[dict[str, str]],
    ) -> bool:
        try:
            if self._component_requires_async_shutdown(component):
                return asyncio.run(self.ashutdown_component(component))
            return self.shutdown_component(component)
        except Exception as exc:
            self._record_error(errors, name, exc)
            return False

    async def _stop_async_component(
        self,
        name: str,
        component: Any,
        errors: list[dict[str, str]],
    ) -> bool:
        try:
            return await self.ashutdown_component(component)
        except Exception as exc:
            self._record_error(errors, name, exc)
            return False

    @staticmethod
    def _record_error(
        errors: list[dict[str, str]],
        component: str,
        exc: BaseException,
    ) -> None:
        public_error = public_error_envelope(exc)
        internal_error = internal_exception_observation(
            exc,
            correlation_id=public_error["correlation_id"],
        )
        exception_text = internal_error["exception_text"]
        errors.append(
            {
                "component": component,
                "error": public_error["message"],
                "error_type": public_error["error_type"],
                "code": public_error["code"],
                "correlation_id": public_error["correlation_id"],
                "error_text_bytes": str(exception_text["bytes"]),
                "error_text_sha256": str(exception_text["sha256"]),
            }
        )

    @staticmethod
    def _running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    @staticmethod
    def _is_async_callable(callback: Any) -> bool:
        return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
            getattr(callback, "__call__", None)
        )

    @staticmethod
    def _current_task() -> asyncio.Task[Any] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def _reject_reentrant_wait(
        self,
        attempt: _ShutdownAttempt,
        *,
        caller_task: asyncio.Task[Any] | None,
        async_wait: bool,
    ) -> None:
        same_thread = attempt.owner_thread_id == threading.get_ident()
        same_task = caller_task is not None and caller_task is attempt.owner_task
        different_async_task = (
            async_wait
            and same_thread
            and caller_task is not None
            and attempt.owner_task is not None
            and not same_task
        )
        inherited_attempt = self._shutdown_attempt_context.get() is attempt
        if inherited_attempt or same_task or (same_thread and not different_async_task):
            raise RuntimeError(
                "reentrant runtime shutdown cannot wait for its own shutdown attempt"
            )

    def _complete_attempt(
        self,
        attempt: _ShutdownAttempt,
        *,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            attempt.result = dict(result) if result is not None else None
            attempt.error = error
            if self._active_attempt is attempt:
                self._active_attempt = None
            attempt.done.set()

    @staticmethod
    def _attempt_result(attempt: _ShutdownAttempt) -> dict[str, Any]:
        if attempt.error is not None:
            raise attempt.error
        if attempt.result is None:
            raise RuntimeError("runtime shutdown attempt completed without an outcome")
        return dict(attempt.result)

    @staticmethod
    def shutdown_component(component: Any) -> bool:
        if component is None:
            return True
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if inspect.isawaitable(result):
                close_awaitable = getattr(result, "close", None)
                if callable(close_awaitable):
                    close_awaitable()
                raise RuntimeError(
                    "async component shutdown requires an async lifecycle path"
                )
            return result is not False
        close = getattr(component, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                close_awaitable = getattr(result, "close", None)
                if callable(close_awaitable):
                    close_awaitable()
                raise RuntimeError(
                    "async component close requires an async lifecycle path"
                )
        return True

    @classmethod
    async def ashutdown_component(cls, component: Any) -> bool:
        if component is None:
            return True
        ashutdown = getattr(component, "ashutdown", None)
        if callable(ashutdown):
            result, interrupted = await cls._invoke_async_cleanup(
                ashutdown,
                description="component ashutdown",
            )
            if interrupted:
                raise asyncio.CancelledError()
            return result is not False
        aclose = getattr(component, "aclose", None)
        if callable(aclose):
            _, interrupted = await cls._invoke_async_cleanup(
                aclose,
                description="component aclose",
            )
            if interrupted:
                raise asyncio.CancelledError()
            return True
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            result, interrupted = await cls._invoke_async_cleanup(
                shutdown,
                description="component shutdown",
            )
            if interrupted:
                raise asyncio.CancelledError()
            return result is not False
        close = getattr(component, "close", None)
        if callable(close):
            _, interrupted = await cls._invoke_async_cleanup(
                close,
                description="component close",
            )
            if interrupted:
                raise asyncio.CancelledError()
        return True


__all__ = ["RuntimeLifecycle"]
