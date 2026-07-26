from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.evidence import (
    PayloadRetentionMaintenance,
    PayloadRetentionPolicy,
    reconcile_pending_external_effects,
)
from agent_libos.primitives import (
    ClockPrimitive,
    FilesystemAdapter,
    GitPrimitive,
    JsonRpcPrimitive,
    McpPrimitive,
    ShellAdapter,
)
from agent_libos.human.manager import HumanObjectManager
from agent_libos.llm.executor import LLMProcessExecutor
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptState,
    EventType,
)
from agent_libos.modules import RuntimeModuleRegistry
from agent_libos.modules.host import ModuleHookServices, ModuleStateRegistry
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.blocking_work import BlockingWorkSupervisor
from agent_libos.runtime.authority_manifest_manager import AuthorityManifestManager
from agent_libos.runtime.boundary_descriptors import (
    CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    EXPLAIN_BOUNDARY_DESCRIPTORS,
    PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES,
)
from agent_libos.runtime.boundary_installer import (
    install_control_mutation_admission_boundaries,
    install_explain_boundaries,
)
from agent_libos.runtime.checkpoint_manager import CheckpointManager
from agent_libos.runtime.checkpoint_image import CheckpointImageInstaller
from agent_libos.runtime.data_flow_manager import DataFlowManager
from agent_libos.runtime.descriptor_catalog import (
    register_protected_operation_descriptors,
)
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.explain_manager import ExplainManager
from agent_libos.runtime.image_boot import ImageBootService
from agent_libos.runtime.image_artifact import ImageArtifactLoader
from agent_libos.runtime.image_package import ImagePackageInstaller
from agent_libos.runtime.image_registry import ImageRegistryPrimitive
from agent_libos.runtime.lifecycle import RuntimeLifecycle, RuntimeRegistryLock
from agent_libos.runtime.message_manager import ProcessMessageManager
from agent_libos.runtime.object_tasks import ObjectTaskManager
from agent_libos.runtime.operation_manager import OperationManager
from agent_libos.runtime.process_launch import ProcessLaunchService
from agent_libos.runtime.process_manager import ProcessManager
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.runtime.ratings import AgentRatingManager
from agent_libos.runtime.resource_manager import ResourceManager
from agent_libos.runtime.scheduler import SimpleScheduler
from agent_libos.runtime.syscall_router import SyscallRouter
from agent_libos.runtime.syscalls import BUILTIN_SYSCALL_NAMES, LibOSSyscallSession
from agent_libos.runtime.snapshots import ProcessExecStateService
from agent_libos.sdk import ProtectedOperationSDK
from agent_libos.skills.manager import SkillManager
from agent_libos.storage import (
    RuntimeStore,
    StoreAssemblyReadiness,
    StoreAssemblyReservation,
    StoreCloseClaimOutcome,
    StoreCloseOutcome,
    UnitOfWork,
    open_store,
)
from agent_libos.substrate import (
    HttpJsonRpcProvider,
    LocalGitProvider,
    LocalResourceProviderSubstrate,
    ResourceProviderSubstrate,
    SdkMcpProvider,
)
from agent_libos.tools.broker import ToolBroker
from agent_libos.utils.ids import new_id

if TYPE_CHECKING:
    from agent_libos.llm.client import LLMClient
    from agent_libos.runtime.runtime import Runtime


RuntimeT = TypeVar("RuntimeT", bound="Runtime")
BlockingT = TypeVar("BlockingT")


@dataclass(frozen=True, slots=True)
class _CapturedRuntimeAssembly(Generic[RuntimeT]):
    store: RuntimeStore | None
    host: RuntimeT | None
    error: BaseException | None


@dataclass(slots=True)
class _OwnedStartupHandshake:
    phase_one_ready: asyncio.Event
    assembly_decided: threading.Event
    assembly_reservation: StoreAssemblyReservation
    store: RuntimeStore | None = None
    open_error: BaseException | None = None
    assembly_reserved: bool = False
    assembly_allowed: bool = False
    decision_error: BaseException | None = None


async def _drain_startup_task(
    worker_task: asyncio.Task[BlockingT],
    *,
    initial_cancellations: tuple[BaseException, ...] = (),
) -> tuple[BlockingT | None, BaseException | None, tuple[BaseException, ...]]:
    """Drain one started worker and return cancellation as explicit outcome data."""

    caller_task = asyncio.current_task()
    cancellations = list(initial_cancellations)
    while not worker_task.done():
        try:
            await asyncio.shield(worker_task)
        except asyncio.CancelledError as exc:
            if worker_task.cancelled():
                return None, exc, tuple(cancellations)
            if caller_task is None or caller_task.cancelling() == 0:
                raise
            caller_task.uncancel()
            cancellations.append(exc)
        except BaseException:
            if not worker_task.done():
                raise
            break
    try:
        result = worker_task.result()
    except BaseException as worker_error:
        return None, worker_error, tuple(cancellations)
    return result, None, tuple(cancellations)


async def _drain_blocking_startup_call(
    callback: Callable[[], BlockingT],
) -> tuple[BlockingT | None, BaseException | None, tuple[BaseException, ...]]:
    """Drain one startup worker while preserving caller cancellation as data."""

    worker_task = asyncio.create_task(run_blocking_once(callback))
    return await _drain_startup_task(worker_task)


def _release_runtime_assembly_reservation_error(
    store: RuntimeStore,
    reservation: StoreAssemblyReservation,
) -> BaseException | None:
    release = getattr(store, "release_runtime_assembly_reservation", None)
    if not callable(release):
        return RuntimeError(
            "runtime store does not support atomic assembly reservation release"
        )
    try:
        released = release(reservation)
    except BaseException as release_error:
        return release_error
    if not isinstance(released, bool):
        return RuntimeError(
            "runtime store returned an invalid assembly-reservation release outcome"
        )
    return None


async def _drain_reserved_blocking_startup_call(
    callback: Callable[[], BlockingT],
    *,
    store: RuntimeStore,
    reservation: StoreAssemblyReservation,
) -> tuple[BlockingT | None, BaseException | None, tuple[BaseException, ...]]:
    """Drain a reserved worker and clear its token if scheduling never starts."""

    try:
        captured, worker_error, cancellations = (
            await _drain_blocking_startup_call(callback)
        )
    except BaseException as handoff_error:
        release_error = _release_runtime_assembly_reservation_error(
            store,
            reservation,
        )
        if release_error is not None:
            raise BaseExceptionGroup(
                "runtime assembly handoff and reservation release failed",
                [handoff_error, release_error],
            ) from handoff_error
        raise
    release_error = _release_runtime_assembly_reservation_error(
        store,
        reservation,
    )
    if release_error is not None:
        worker_error = (
            release_error
            if worker_error is None
            else BaseExceptionGroup(
                "runtime assembly worker and reservation release failed",
                [worker_error, release_error],
            )
        )
    return captured, worker_error, cancellations


def _combine_startup_failure(
    description: str,
    error: BaseException,
    cancellations: tuple[BaseException, ...],
) -> BaseException:
    if not cancellations:
        return error
    return BaseExceptionGroup(
        f"{description} failed after caller cancellation",
        [*cancellations, error],
    )


class _CompletedCleanupCancellation(BaseException):
    """Internal outcome preserving cancellation and cleanup results together."""

    def __init__(
        self,
        cancellations: list[BaseException],
        cleanup_errors: list[dict[str, str]],
        cleanup_exception: BaseException | None = None,
    ) -> None:
        super().__init__("caller cancelled while failed assembly cleanup drained")
        self.cancellations = tuple(cancellations)
        self.cleanup_errors = tuple(dict(item) for item in cleanup_errors)
        self.cleanup_exception = cleanup_exception


class _CompletedAssemblyCancellation(BaseExceptionGroup):
    """Typed internal boundary for cancellation after Runtime reached OPEN."""


class RuntimeAssemblyCleanupKind(str, Enum):
    """Public strategy used by a retriable Runtime assembly cleanup handle."""

    FAILED_ASSEMBLY = "failed_assembly"
    OPEN_RUNTIME_SHUTDOWN = "open_runtime_shutdown"


class RuntimeAssemblyCleanupRequired(RuntimeError):
    """Retriable ownership handle for an incompletely released Runtime graph.

    This exception is a leaf in the ``BaseExceptionGroup`` raised by
    :meth:`Runtime.open` or :meth:`Runtime.aopen`. This also covers an async
    assembly that reached ``OPEN`` after its caller was cancelled but whose
    normal Runtime shutdown did not complete. Call
    ``RuntimeAssemblyCleanupRequired.extract(error)`` on the caught exception,
    then call :meth:`release` (sync code) or await :meth:`arelease` (async
    code). A failed-assembly handle closes a builder-owned store only after
    graph cleanup completes and leaves a store passed to ``from_store``
    caller-owned. An ``OPEN_RUNTIME_SHUTDOWN`` handle instead retries the
    Runtime's ordinary shutdown contract, including its normal store release.
    """

    def __init__(
        self,
        *,
        partial_runtime: Runtime | None,
        store: RuntimeStore,
        cleanup_errors: list[dict[str, str]],
        cleanup_completed: bool = False,
        cleanup_kind: RuntimeAssemblyCleanupKind = (
            RuntimeAssemblyCleanupKind.FAILED_ASSEMBLY
        ),
    ) -> None:
        super().__init__(
            "runtime assembly resources remain owned; extract this cleanup "
            "handle and call release() or await arelease()"
        )
        self.partial_runtime = partial_runtime
        self.store = store
        self._cleanup_errors = [dict(item) for item in cleanup_errors]
        self._cleanup_completed = cleanup_completed
        self._cleanup_kind = cleanup_kind
        self._owns_store = False
        self._owned_store_close_reservation: Any | None = None
        self._released = False
        self._release_in_progress = False
        self._lock = threading.Lock()

    @property
    def cleanup_errors(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            return tuple(dict(item) for item in self._cleanup_errors)

    @property
    def cleanup_completed(self) -> bool:
        with self._lock:
            return self._cleanup_completed

    @property
    def cleanup_kind(self) -> RuntimeAssemblyCleanupKind:
        """Return the public cleanup strategy retained by this handle."""

        return self._cleanup_kind

    @property
    def owns_store(self) -> bool:
        with self._lock:
            return self._owns_store

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    @classmethod
    def extract(
        cls,
        error: BaseException,
    ) -> tuple[RuntimeAssemblyCleanupRequired, ...]:
        """Return every cleanup handle nested in an exception group."""

        found: list[RuntimeAssemblyCleanupRequired] = []
        seen: set[int] = set()

        def visit(item: BaseException) -> None:
            if isinstance(item, cls):
                if id(item) not in seen:
                    seen.add(id(item))
                    found.append(item)
                return
            if isinstance(item, BaseExceptionGroup):
                for nested in item.exceptions:
                    visit(nested)

        visit(error)
        return tuple(found)

    def release(self) -> None:
        """Retry synchronous graph cleanup and close an owned store."""

        if self.released:
            return
        if not self.cleanup_completed or self.owns_store:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    "Runtime assembly cleanup is running inside an event loop; "
                    "use await handle.arelease()"
                )
        if not self._begin_release():
            return
        try:
            if self.cleanup_kind is RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN:
                self._release_open_runtime()
                return
            self._preflight_owned_store_release()
            if not self.cleanup_completed:
                host = self._require_partial_runtime()
                try:
                    cleanup_errors = RuntimeBuilder._cleanup_failed_assembly(host)
                except BaseException as cleanup_error:
                    self._replace_cleanup_errors(
                        [self._error_record("runtime_graph", cleanup_error)]
                    )
                    raise BaseExceptionGroup(
                        "runtime assembly release cleanup failed",
                        [self, cleanup_error],
                    ) from cleanup_error
                if cleanup_errors:
                    self._replace_cleanup_errors(cleanup_errors)
                    raise self
                self._mark_cleanup_completed()
            try:
                close_outcome = self._release_owned_store()
            except BaseException as close_error:
                self._replace_cleanup_errors(
                    [self._error_record("store", close_error)]
                )
                raise BaseExceptionGroup(
                    "runtime assembly release store close failed",
                    [self, close_error],
                ) from close_error
            ownership_error = self._owned_store_close_outcome_error(close_outcome)
            if ownership_error is not None:
                self._replace_cleanup_errors(
                    [self._error_record("store", ownership_error)]
                )
                raise BaseExceptionGroup(
                    "runtime assembly release store ownership conflict",
                    [self, ownership_error],
                ) from ownership_error
            self._mark_released()
            if close_outcome is not None and close_outcome.warnings:
                self._replace_cleanup_errors(
                    [self._error_record("store", item) for item in close_outcome.warnings]
                )
                raise BaseExceptionGroup(
                    "runtime assembly store ownership was released with cleanup warnings",
                    list(close_outcome.warnings),
                )
        finally:
            self._finish_release()

    async def arelease(self) -> None:
        """Retry loop-affine graph cleanup and close an owned store."""

        if self.released:
            return
        if not self._begin_release():
            return
        cancellations: tuple[BaseException, ...] = ()
        try:
            if self.cleanup_kind is RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN:
                await self._arelease_open_runtime()
                return
            self._preflight_owned_store_release()
            if not self.cleanup_completed:
                host = self._require_partial_runtime()
                try:
                    cleanup_errors = await RuntimeBuilder._drain_async_failed_assembly(
                        host
                    )
                except _CompletedCleanupCancellation as outcome:
                    cancellations = outcome.cancellations
                    if outcome.cleanup_exception is not None:
                        self._replace_cleanup_errors(
                            [
                                self._error_record(
                                    "runtime_graph",
                                    outcome.cleanup_exception,
                                )
                            ]
                        )
                        raise BaseExceptionGroup(
                            "runtime assembly release cleanup failed after cancellation",
                            [self, *cancellations, outcome.cleanup_exception],
                        ) from outcome.cleanup_exception
                    cleanup_errors = [dict(item) for item in outcome.cleanup_errors]
                except BaseException as cleanup_error:
                    self._replace_cleanup_errors(
                        [self._error_record("runtime_graph", cleanup_error)]
                    )
                    raise BaseExceptionGroup(
                        "runtime assembly release cleanup failed",
                        [self, cleanup_error],
                    ) from cleanup_error
                if cleanup_errors:
                    self._replace_cleanup_errors(cleanup_errors)
                    if cancellations:
                        raise BaseExceptionGroup(
                            "runtime assembly release remains incomplete after cancellation",
                            [self, *cancellations],
                        )
                    raise self
                self._mark_cleanup_completed()
            (
                close_outcome,
                close_error,
                close_cancellations,
            ) = await self._await_owned_store_release()
            cancellations = (*cancellations, *close_cancellations)
            if close_error is not None:
                self._replace_cleanup_errors(
                    [self._error_record("store", close_error)]
                )
                raise BaseExceptionGroup(
                    "runtime assembly release store close failed",
                    [self, *cancellations, close_error],
                ) from close_error
            ownership_error = self._owned_store_close_outcome_error(close_outcome)
            if ownership_error is not None:
                self._replace_cleanup_errors(
                    [self._error_record("store", ownership_error)]
                )
                raise BaseExceptionGroup(
                    "runtime assembly release store ownership conflict",
                    [self, *cancellations, ownership_error],
                ) from ownership_error
            self._mark_released()
            if close_outcome is not None and close_outcome.warnings:
                self._replace_cleanup_errors(
                    [self._error_record("store", item) for item in close_outcome.warnings]
                )
                raise BaseExceptionGroup(
                    "runtime assembly store ownership was released with cleanup warnings",
                    [*cancellations, *close_outcome.warnings],
                )
            if cancellations:
                raise BaseExceptionGroup(
                    "runtime assembly release completed after caller cancellation",
                    list(cancellations),
                )
        finally:
            self._finish_release()

    def _release_open_runtime(self) -> None:
        """Retry ordinary shutdown for an assembly that already reached OPEN."""

        host = self._require_partial_runtime()
        try:
            shutdown_result = host.shutdown(
                actor="runtime.builder.cleanup",
                reason="runtime.aopen.cancelled_after_assembly.retry",
            )
            if shutdown_result.get("recovery_required") is True:
                shutdown_result = host.release_recovery_diagnostics()
        except BaseException as shutdown_error:
            if host.lifecycle.closed:
                self._mark_cleanup_completed()
                self._mark_released()
                raise
            self._replace_cleanup_errors(
                [self._error_record("runtime_shutdown", shutdown_error)]
            )
            raise BaseExceptionGroup(
                "cancelled Runtime assembly release shutdown failed",
                [self, shutdown_error],
            ) from shutdown_error
        shutdown_error = self._open_runtime_shutdown_error(host, shutdown_result)
        if shutdown_error is not None:
            self._replace_cleanup_errors(
                [self._error_record("runtime_shutdown", shutdown_error)]
            )
            raise self
        self._mark_cleanup_completed()
        self._mark_released()

    async def _arelease_open_runtime(self) -> None:
        """Drain ordinary async shutdown while retaining ownership on cancellation."""

        host = self._require_partial_runtime()
        shutdown_task = asyncio.create_task(
            host.ashutdown(
                actor="runtime.builder.cleanup",
                reason="runtime.aopen.cancelled_after_assembly.retry",
            )
        )
        shutdown_result, shutdown_error, cancellations = await _drain_startup_task(
            shutdown_task
        )
        if (
            shutdown_error is None
            and shutdown_result is not None
            and shutdown_result.get("recovery_required") is True
        ):
            recovery_release_task = asyncio.create_task(
                host.arelease_recovery_diagnostics()
            )
            (
                shutdown_result,
                shutdown_error,
                cancellations,
            ) = await _drain_startup_task(
                recovery_release_task,
                initial_cancellations=cancellations,
            )
        if shutdown_error is not None:
            if host.lifecycle.closed:
                self._mark_cleanup_completed()
                self._mark_released()
                failures = [*cancellations, shutdown_error]
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup(
                    "cancelled Runtime assembly release completed with control-flow errors",
                    failures,
                ) from shutdown_error
            self._replace_cleanup_errors(
                [self._error_record("runtime_shutdown", shutdown_error)]
            )
            raise BaseExceptionGroup(
                "cancelled Runtime assembly release shutdown failed",
                [self, *cancellations, shutdown_error],
            ) from shutdown_error
        if shutdown_result is None:
            shutdown_error = RuntimeError(
                "cancelled Runtime assembly release returned no shutdown outcome"
            )
        else:
            shutdown_error = self._open_runtime_shutdown_error(
                host,
                shutdown_result,
            )
        if shutdown_error is not None:
            self._replace_cleanup_errors(
                [self._error_record("runtime_shutdown", shutdown_error)]
            )
            if cancellations:
                raise BaseExceptionGroup(
                    "cancelled Runtime assembly release remains incomplete after cancellation",
                    [self, *cancellations],
                )
            raise self
        self._mark_cleanup_completed()
        self._mark_released()
        if cancellations:
            raise BaseExceptionGroup(
                "cancelled Runtime assembly release completed after caller cancellation",
                list(cancellations),
            )

    @staticmethod
    def _open_runtime_shutdown_error(
        host: Runtime,
        shutdown_result: dict[str, Any],
    ) -> RuntimeError | None:
        if shutdown_result.get("ok") is True and host.lifecycle.closed:
            return None
        return RuntimeError(
            "runtime assembly completed after cancellation but normal shutdown "
            f"remains incomplete: {shutdown_result!r}"
        )

    def _claim_owned_store(
        self,
        store: RuntimeStore,
        close_reservation: Any,
    ) -> None:
        if self.store is not store:
            raise RuntimeError("cleanup handle store does not match builder-owned store")
        with self._lock:
            if self._owns_store:
                if self._owned_store_close_reservation is not close_reservation:
                    raise RuntimeError(
                        "cleanup handle is already bound to another store close reservation"
                    )
                return
            self._owns_store = True
            self._owned_store_close_reservation = close_reservation

    def _begin_release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            if self._release_in_progress:
                raise RuntimeError("runtime assembly release is already in progress")
            self._release_in_progress = True
            return True

    def _finish_release(self) -> None:
        with self._lock:
            self._release_in_progress = False

    def _mark_cleanup_completed(self) -> None:
        with self._lock:
            self._cleanup_completed = True
            self._cleanup_errors = []

    def _mark_released(self) -> None:
        with self._lock:
            self._released = True

    def _terminalize_released_store_ownership(self) -> None:
        """Drop a store claim after the backend reports ownership already gone."""

        with self._lock:
            self._owns_store = False
            self._owned_store_close_reservation = None
            if self._cleanup_completed:
                self._released = True

    def _replace_cleanup_errors(self, errors: list[dict[str, str]]) -> None:
        with self._lock:
            self._cleanup_errors = [dict(item) for item in errors]

    def _require_partial_runtime(self) -> Runtime:
        if self.partial_runtime is None:
            raise RuntimeError("partial Runtime is unavailable for cleanup retry")
        return self.partial_runtime

    def _release_owned_store(self) -> StoreCloseOutcome | None:
        if not self.owns_store:
            return None
        close_reservation = self._require_owned_store_close_reservation()
        if not RuntimeBuilder._ensure_owned_store_close_reservation(
            self.partial_runtime,
            self.store,
            close_reservation,
        ):
            return StoreCloseOutcome(
                guard_matched=False,
                ownership_released=False,
            )
        return self._close_owned_store_reservation(close_reservation)

    def _release_claimed_owned_store(self) -> StoreCloseOutcome | None:
        """Consume the exact close claim installed on the event-loop thread."""

        if not self.owns_store:
            return None
        close_reservation = self._require_owned_store_close_reservation()
        return self._close_owned_store_reservation(close_reservation)

    def _close_owned_store_reservation(
        self,
        close_reservation: Any,
    ) -> StoreCloseOutcome:
        outcome = self.store.release_admission_guard_and_close(close_reservation)
        if not isinstance(outcome, StoreCloseOutcome):
            raise RuntimeError(
                "runtime store returned an invalid structured close outcome"
            )
        return outcome

    def _require_owned_store_close_reservation(self) -> Any:
        with self._lock:
            close_reservation = self._owned_store_close_reservation
        if close_reservation is None:
            raise RuntimeError(
                "builder-owned cleanup handle has no exact store close reservation"
            )
        return close_reservation

    def _preflight_owned_store_release(self) -> None:
        """Reject caller-thread store scopes before any graph cleanup begins."""

        if not self.owns_store:
            return
        close_reservation = self._require_owned_store_close_reservation()
        readiness = self._probe_owned_store_close(close_reservation)
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            self._terminalize_released_store_ownership()
            return
        if readiness is StoreCloseClaimOutcome.GUARD_MISMATCH:
            readiness = RuntimeBuilder._try_repair_owned_store_close_reservation(
                self.partial_runtime,
                self.store,
                close_reservation,
            )
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            self._terminalize_released_store_ownership()
            return
        if readiness in {
            StoreCloseClaimOutcome.READY,
            StoreCloseClaimOutcome.LOCK_BUSY,
        }:
            return
        readiness_error = self._owned_store_close_claim_error(
            readiness,
            phase="preflight",
        )
        assert readiness_error is not None
        if readiness is StoreCloseClaimOutcome.GUARD_MISMATCH:
            self._replace_cleanup_errors(
                [self._error_record("store", readiness_error)]
            )
            raise BaseExceptionGroup(
                "runtime assembly release store ownership conflict",
                [self, readiness_error],
            ) from readiness_error
        raise readiness_error

    def _probe_owned_store_close(
        self,
        close_reservation: Any,
    ) -> StoreCloseClaimOutcome:
        probe = getattr(self.store, "probe_admission_guard_close", None)
        if not callable(probe):
            raise RuntimeError(
                "runtime store does not support nonblocking guarded-close preflight"
            )
        readiness = probe(close_reservation)
        if not isinstance(readiness, StoreCloseClaimOutcome):
            raise RuntimeError(
                "runtime store returned an invalid guarded-close preflight outcome"
            )
        return readiness

    def _claim_owned_store_close(
        self,
    ) -> StoreCloseClaimOutcome:
        close_reservation = self._require_owned_store_close_reservation()
        claim = getattr(self.store, "claim_admission_guard_close", None)
        if not callable(claim):
            raise RuntimeError(
                "runtime store does not support atomic guarded-close claims"
            )
        readiness = claim(close_reservation)
        if not isinstance(readiness, StoreCloseClaimOutcome):
            raise RuntimeError(
                "runtime store returned an invalid guarded-close claim outcome"
            )
        return readiness

    def _capture_owned_store_release(
        self,
    ) -> tuple[StoreCloseOutcome | None, BaseException | None]:
        """Return worker-thread close control flow as data for async drain."""

        try:
            return self._release_claimed_owned_store(), None
        except BaseException as exc:
            return None, exc

    async def _await_owned_store_release(
        self,
    ) -> tuple[
        StoreCloseOutcome | None,
        BaseException | None,
        tuple[BaseException, ...],
    ]:
        if not self.owns_store:
            return None, None, ()
        try:
            readiness = self._claim_owned_store_close()
        except BaseException as claim_error:
            return None, claim_error, ()
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            self._terminalize_released_store_ownership()
            return None, None, ()
        readiness_error = self._owned_store_close_claim_error(
            readiness,
            phase="claim",
        )
        if readiness is StoreCloseClaimOutcome.GUARD_MISMATCH:
            return (
                StoreCloseOutcome(
                    guard_matched=False,
                    ownership_released=False,
                ),
                None,
                (),
            )
        if readiness_error is not None:
            return None, readiness_error, ()
        close_task = asyncio.create_task(
            run_blocking_once(self._capture_owned_store_release)
        )
        caller_task = asyncio.current_task()
        cancellations: list[BaseException] = []
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                if caller_task is None or caller_task.cancelling() == 0:
                    break
                caller_task.uncancel()
                cancellations.append(exc)
        try:
            outcome, error = close_task.result()
        except BaseException as unexpected:
            outcome, error = None, unexpected
        return outcome, error, tuple(cancellations)

    @staticmethod
    def _owned_store_close_claim_error(
        outcome: StoreCloseClaimOutcome,
        *,
        phase: str,
    ) -> RuntimeError | None:
        if outcome is StoreCloseClaimOutcome.READY:
            return None
        if not isinstance(outcome, StoreCloseClaimOutcome):
            return RuntimeError(
                "runtime store returned an invalid guarded-close claim outcome"
            )
        return RuntimeError(
            f"builder-owned store close {phase} is not ready: {outcome.value}"
        )

    @staticmethod
    def _owned_store_close_outcome_error(
        outcome: StoreCloseOutcome | None | Any,
    ) -> RuntimeError | None:
        if outcome is None:
            return None
        if not isinstance(outcome, StoreCloseOutcome):
            return RuntimeError(
                "runtime store returned an invalid structured close outcome"
            )
        if outcome.ownership_released:
            return None
        if not outcome.guard_matched:
            return RuntimeError(
                "builder-owned store close reservation no longer matches"
            )
        if not outcome.ownership_released:
            return RuntimeError(
                "builder-owned store close retained backend ownership"
            )
        return None

    @staticmethod
    def _error_record(component: str, error: BaseException) -> dict[str, str]:
        return {
            "component": component,
            "error_type": type(error).__name__,
            "error": str(error),
        }


@dataclass(frozen=True, slots=True)
class RuntimeBuilder(Generic[RuntimeT]):
    """Open and assemble one Runtime from explicit host dependencies."""

    runtime_type: type[RuntimeT]
    config: AgentLibOSConfig = DEFAULT_CONFIG
    substrate: ResourceProviderSubstrate | None = None
    module_manifests: tuple[str | Path, ...] | None = None
    trusted_modules: tuple[str, ...] | None = None
    trusted_module_sha256: tuple[str, ...] | None = None

    def open(self, target: str | Path | None = None) -> RuntimeT:
        self._require_sync_assembly_context()
        self._validate_runtime_allocation_contract()
        store = open_store(target, config=self.config)
        close_reservation = self._new_owned_store_close_reservation()
        try:
            return self._from_store(
                store,
                llm_client=None,
                owned_store_close_reservation=close_reservation,
            )
        except BaseException as original:
            self._close_owned_store_after_failed_open(
                store,
                original,
                close_reservation=close_reservation,
            )
            raise

    async def aopen(self, target: str | Path | None = None) -> RuntimeT:
        """Open and assemble a Runtime on the caller's event loop."""

        self._validate_runtime_allocation_contract()
        close_reservation = self._new_owned_store_close_reservation()
        captured, worker_error, cancellations = (
            await self._acapture_owned_runtime_assembly(target)
        )
        if worker_error is not None:
            raise _combine_startup_failure(
                "runtime open worker",
                worker_error,
                cancellations,
            ) from worker_error
        if captured is None:
            raise RuntimeError("runtime open worker returned no captured outcome")
        if captured.store is None:
            open_error = captured.error or RuntimeError(
                "runtime store open returned no store"
            )
            raise _combine_startup_failure(
                "runtime store open",
                open_error,
                cancellations,
            ) from open_error
        store = captured.store
        if captured.host is None:
            allocation_error = captured.error or RuntimeError(
                "runtime allocation returned no host"
            )
            original = _combine_startup_failure(
                "runtime allocation",
                allocation_error,
                cancellations,
            )
            await self._aclose_owned_store_after_failed_open(
                store,
                original,
                close_reservation=close_reservation,
            )
            raise original
        try:
            return await self._afinalize_captured_assembly(
                captured,
                cancellations=cancellations,
                owned_store_close_reservation=close_reservation,
            )
        except BaseException as original:
            if isinstance(original, _CompletedAssemblyCancellation):
                raise
            await self._aclose_owned_store_after_failed_open(
                store,
                original,
                close_reservation=close_reservation,
            )
            raise

    def from_store(
        self,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None = None,
    ) -> RuntimeT:
        self._require_sync_assembly_context()
        return self._from_store(
            store,
            llm_client=llm_client,
            owned_store_close_reservation=None,
        )

    def _from_store(
        self,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        owned_store_close_reservation: Any | None,
    ) -> RuntimeT:
        host = self._allocate_host()
        self.assemble_existing(
            host,
            store,
            llm_client=llm_client,
            substrate=self.substrate,
            config=self.config,
            startup_module_manifests=self.module_manifests,
            trusted_modules=self.trusted_modules,
            trusted_module_sha256=self.trusted_module_sha256,
            owned_store_close_reservation=owned_store_close_reservation,
        )
        return host

    async def afrom_store(
        self,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None = None,
    ) -> RuntimeT:
        return await self._afrom_store(
            store,
            llm_client=llm_client,
            owned_store_close_reservation=None,
        )

    async def _afrom_store(
        self,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        owned_store_close_reservation: Any | None,
    ) -> RuntimeT:
        assembly_reservation = self._new_runtime_assembly_reservation()
        readiness_error = self._runtime_assembly_reservation_error(
            store,
            assembly_reservation,
        )
        if readiness_error is not None:
            raise readiness_error
        captured, worker_error, cancellations = (
            await _drain_reserved_blocking_startup_call(
                partial(
                    self._capture_store_runtime_assembly,
                    store,
                    llm_client,
                    assembly_reservation,
                ),
                store=store,
                reservation=assembly_reservation,
            )
        )
        if worker_error is not None:
            raise _combine_startup_failure(
                "runtime assembly worker",
                worker_error,
                cancellations,
            ) from worker_error
        if captured is None:
            raise RuntimeError("runtime assembly worker returned no captured outcome")
        if captured.host is None:
            allocation_error = captured.error or RuntimeError(
                "runtime allocation returned no host"
            )
            raise _combine_startup_failure(
                "runtime allocation",
                allocation_error,
                cancellations,
            ) from allocation_error
        return await self._afinalize_captured_assembly(
            captured,
            cancellations=cancellations,
            owned_store_close_reservation=owned_store_close_reservation,
        )

    async def _acapture_owned_runtime_assembly(
        self,
        target: str | Path | None,
    ) -> tuple[
        _CapturedRuntimeAssembly[RuntimeT] | None,
        BaseException | None,
        tuple[BaseException, ...],
    ]:
        caller_loop = asyncio.get_running_loop()
        assembly_reservation = self._new_runtime_assembly_reservation()
        handshake = _OwnedStartupHandshake(
            phase_one_ready=asyncio.Event(),
            assembly_decided=threading.Event(),
            assembly_reservation=assembly_reservation,
        )
        worker_task = asyncio.create_task(
            run_blocking_once(
                self._capture_owned_runtime_assembly,
                target,
                handshake,
                caller_loop,
            )
        )
        worker_task.add_done_callback(
            lambda _task: handshake.phase_one_ready.set()
        )
        phase_one_wait = asyncio.create_task(handshake.phase_one_ready.wait())
        caller_task = asyncio.current_task()
        cancellations: list[BaseException] = []
        try:
            while not phase_one_wait.done():
                try:
                    await asyncio.shield(phase_one_wait)
                except asyncio.CancelledError as exc:
                    if caller_task is None or caller_task.cancelling() == 0:
                        raise
                    caller_task.uncancel()
                    cancellations.append(exc)
            phase_one_wait.result()
            if handshake.store is not None:
                if cancellations:
                    handshake.decision_error = RuntimeError(
                        "runtime assembly was skipped after caller cancellation"
                    )
                else:
                    handshake.decision_error = (
                        self._runtime_assembly_reservation_error(
                            handshake.store,
                            assembly_reservation,
                        )
                    )
                    handshake.assembly_reserved = (
                        handshake.decision_error is None
                    )
                handshake.assembly_allowed = handshake.decision_error is None
        finally:
            # The worker never crosses into allocation or assembly until the
            # caller has performed the typed nonblocking store probe. Always
            # release the handshake, including cancellation and probe failure.
            handshake.assembly_decided.set()
        try:
            captured, worker_error, drained_cancellations = (
                await _drain_startup_task(
                    worker_task,
                    initial_cancellations=tuple(cancellations),
                )
            )
        except BaseException as handoff_error:
            if not handshake.assembly_reserved:
                raise
            release_error = _release_runtime_assembly_reservation_error(
                handshake.store,
                assembly_reservation,
            )
            if release_error is not None:
                raise BaseExceptionGroup(
                    "owned runtime assembly handoff and reservation release failed",
                    [handoff_error, release_error],
                ) from handoff_error
            raise
        if handshake.assembly_reserved:
            release_error = _release_runtime_assembly_reservation_error(
                handshake.store,
                assembly_reservation,
            )
            if release_error is not None:
                worker_error = (
                    release_error
                    if worker_error is None
                    else BaseExceptionGroup(
                        "owned runtime assembly worker and reservation release failed",
                        [worker_error, release_error],
                    )
                )
        return captured, worker_error, drained_cancellations

    def _capture_owned_runtime_assembly(
        self,
        target: str | Path | None,
        handshake: _OwnedStartupHandshake,
        caller_loop: asyncio.AbstractEventLoop,
    ) -> _CapturedRuntimeAssembly[RuntimeT]:
        try:
            store = open_store(target, config=self.config)
        except BaseException as open_error:
            handshake.open_error = open_error
            caller_loop.call_soon_threadsafe(handshake.phase_one_ready.set)
            return _CapturedRuntimeAssembly(
                store=None,
                host=None,
                error=open_error,
            )
        handshake.store = store
        try:
            caller_loop.call_soon_threadsafe(handshake.phase_one_ready.set)
        except BaseException as signal_error:
            return _CapturedRuntimeAssembly(
                store=store,
                host=None,
                error=signal_error,
            )
        handshake.assembly_decided.wait()
        if not handshake.assembly_allowed:
            return _CapturedRuntimeAssembly(
                store=store,
                host=None,
                error=handshake.decision_error
                or RuntimeError("runtime assembly was not authorized by its caller"),
            )
        return self._capture_store_runtime_assembly(
            store,
            None,
            handshake.assembly_reservation,
        )

    @staticmethod
    def _runtime_assembly_reservation_error(
        store: RuntimeStore,
        reservation: StoreAssemblyReservation,
    ) -> BaseException | None:
        reserve = getattr(store, "reserve_runtime_assembly", None)
        if not callable(reserve):
            return RuntimeError(
                "runtime store does not support atomic assembly reservations"
            )
        try:
            readiness = reserve(reservation)
        except BaseException as reservation_error:
            decision_error: BaseException = reservation_error
        else:
            if readiness is StoreAssemblyReadiness.READY:
                return None
            if isinstance(readiness, StoreAssemblyReadiness):
                # A valid non-READY outcome is the store's guarantee that no
                # token was installed. Do not turn a nonblocking readiness
                # rejection into a blocking compare-and-clear attempt.
                return RuntimeError(
                    f"runtime store assembly is not ready: {readiness.value}"
                )
            decision_error = RuntimeError(
                "runtime store returned an invalid assembly-reservation outcome"
            )
        release_error = _release_runtime_assembly_reservation_error(
            store,
            reservation,
        )
        if release_error is None:
            return decision_error
        return BaseExceptionGroup(
            "runtime assembly decision and reservation release failed",
            [decision_error, release_error],
        )

    def _capture_store_runtime_assembly(
        self,
        store: RuntimeStore,
        llm_client: LLMClient | None,
        assembly_reservation: StoreAssemblyReservation,
    ) -> _CapturedRuntimeAssembly[RuntimeT]:
        host: RuntimeT | None = None
        assembly_error: BaseException | None = None
        try:
            host = self._allocate_host()
            claim = getattr(store, "claim_runtime_assembly", None)
            if not callable(claim):
                raise RuntimeError(
                    "runtime store does not support atomic assembly claims"
                )
            with claim(assembly_reservation):
                self._assemble_host(
                    host,
                    store,
                    substrate=self.substrate,
                    config=self.config,
                    llm_client=llm_client,
                    startup_module_manifests=self.module_manifests,
                    trusted_modules=self.trusted_modules,
                    trusted_module_sha256=self.trusted_module_sha256,
                )
        except BaseException as error:
            assembly_error = error
        release_error = _release_runtime_assembly_reservation_error(
            store,
            assembly_reservation,
        )
        if release_error is not None:
            assembly_error = (
                release_error
                if assembly_error is None
                else BaseExceptionGroup(
                    "runtime assembly and reservation release failed",
                    [assembly_error, release_error],
                )
            )
        return _CapturedRuntimeAssembly(
            store=store,
            host=host,
            error=assembly_error,
        )

    @staticmethod
    def _new_runtime_assembly_reservation() -> StoreAssemblyReservation:
        return StoreAssemblyReservation(new_id("runtime_assembly"))

    @staticmethod
    def _new_owned_store_close_reservation() -> Any:
        @contextmanager
        def close_reservation():
            yield

        return close_reservation

    def _allocate_host(self) -> RuntimeT:
        self._validate_runtime_allocation_contract()
        host = self.runtime_type.allocate_unassembled()
        if not isinstance(host, self.runtime_type):
            raise TypeError("Runtime allocation hook returned the wrong type")
        return host

    def _validate_runtime_allocation_contract(self) -> None:
        from agent_libos.runtime.runtime import Runtime as BaseRuntime

        allocation_hook = getattr(
            self.runtime_type.allocate_unassembled,
            "__func__",
            self.runtime_type.allocate_unassembled,
        )
        base_allocation_hook = getattr(
            BaseRuntime.allocate_unassembled,
            "__func__",
            BaseRuntime.allocate_unassembled,
        )
        if (
            self.runtime_type.__init__ is not BaseRuntime.__init__
            and allocation_hook is base_allocation_hook
        ):
            raise TypeError(
                f"{self.runtime_type.__qualname__} overrides Runtime.__init__; "
                "it must also override allocate_unassembled for builder assembly"
            )

    @classmethod
    def assemble_existing(
        cls,
        host: Runtime,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        substrate: ResourceProviderSubstrate | None,
        config: AgentLibOSConfig | None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
        owned_store_close_reservation: Any | None = None,
    ) -> None:
        cls._require_sync_assembly_context()
        try:
            cls._assemble_host(
                host,
                store,
                substrate=substrate,
                config=config,
                llm_client=llm_client,
                startup_module_manifests=startup_module_manifests,
                trusted_modules=trusted_modules,
                trusted_module_sha256=trusted_module_sha256,
            )
        except BaseException as original:
            cls._attach_partial_runtime(original, host)
            reservation_failures = cls._owned_store_close_reservation_failures(
                host,
                store,
                owned_store_close_reservation,
            )
            try:
                cleanup_errors = cls._cleanup_failed_assembly(host)
            except BaseException as cleanup_error:
                cleanup_records = [
                    RuntimeAssemblyCleanupRequired._error_record(
                        "store_close_reservation",
                        item,
                    )
                    for item in reservation_failures
                ]
                cleanup_records.append(
                    RuntimeAssemblyCleanupRequired._error_record(
                        "runtime_graph",
                        cleanup_error,
                    )
                )
                handle = RuntimeAssemblyCleanupRequired(
                    partial_runtime=host,
                    store=store,
                    cleanup_errors=cleanup_records,
                )
                raise BaseExceptionGroup(
                    "runtime assembly and cleanup failed",
                    [original, handle, *reservation_failures, cleanup_error],
                ) from original
            reservation_records = [
                RuntimeAssemblyCleanupRequired._error_record(
                    "store_close_reservation",
                    item,
                )
                for item in reservation_failures
            ]
            cls._raise_assembly_cleanup_errors(
                original,
                [*reservation_records, *cleanup_errors],
                host=host,
                store=store,
                cleanup_completed=bool(reservation_failures) and not cleanup_errors,
                secondary_failures=reservation_failures,
            )
            raise

    @classmethod
    def _capture_existing_runtime_assembly(
        cls,
        host: RuntimeT,
        store: RuntimeStore,
        assembly_reservation: StoreAssemblyReservation,
        *,
        llm_client: LLMClient | None,
        substrate: ResourceProviderSubstrate | None,
        config: AgentLibOSConfig | None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
    ) -> _CapturedRuntimeAssembly[RuntimeT]:
        assembly_error: BaseException | None = None
        try:
            claim = getattr(store, "claim_runtime_assembly", None)
            if not callable(claim):
                raise RuntimeError(
                    "runtime store does not support atomic assembly claims"
                )
            with claim(assembly_reservation):
                cls._assemble_host(
                    host,
                    store,
                    substrate=substrate,
                    config=config,
                    llm_client=llm_client,
                    startup_module_manifests=startup_module_manifests,
                    trusted_modules=trusted_modules,
                    trusted_module_sha256=trusted_module_sha256,
                )
        except BaseException as error:
            assembly_error = error
        release_error = _release_runtime_assembly_reservation_error(
            store,
            assembly_reservation,
        )
        if release_error is not None:
            assembly_error = (
                release_error
                if assembly_error is None
                else BaseExceptionGroup(
                    "runtime assembly and reservation release failed",
                    [assembly_error, release_error],
                )
            )
        return _CapturedRuntimeAssembly(
            store=store,
            host=host,
            error=assembly_error,
        )

    @classmethod
    async def aassemble_existing(
        cls,
        host: RuntimeT,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        substrate: ResourceProviderSubstrate | None,
        config: AgentLibOSConfig | None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
        owned_store_close_reservation: Any | None = None,
    ) -> None:
        assembly_reservation = cls._new_runtime_assembly_reservation()
        readiness_error = cls._runtime_assembly_reservation_error(
            store,
            assembly_reservation,
        )
        if readiness_error is not None:
            raise readiness_error
        captured, worker_error, cancellations = (
            await _drain_reserved_blocking_startup_call(
                partial(
                    cls._capture_existing_runtime_assembly,
                    host,
                    store,
                    assembly_reservation,
                    substrate=substrate,
                    config=config,
                    llm_client=llm_client,
                    startup_module_manifests=startup_module_manifests,
                    trusted_modules=trusted_modules,
                    trusted_module_sha256=trusted_module_sha256,
                ),
                store=store,
                reservation=assembly_reservation,
            )
        )
        if captured is None:
            captured = _CapturedRuntimeAssembly(
                store=store,
                host=host,
                error=worker_error
                or RuntimeError("runtime assembly worker returned no captured outcome"),
            )
        await cls._afinalize_captured_assembly(
            captured,
            cancellations=cancellations,
            owned_store_close_reservation=owned_store_close_reservation,
        )

    @classmethod
    async def _afinalize_captured_assembly(
        cls,
        captured: _CapturedRuntimeAssembly[RuntimeT],
        *,
        cancellations: tuple[BaseException, ...],
        owned_store_close_reservation: Any | None,
    ) -> RuntimeT:
        host = captured.host
        store = captured.store
        if host is None or store is None:
            raise RuntimeError("captured Runtime assembly is missing its host or store")
        if captured.error is None:
            if cancellations:
                await cls._drain_cancelled_open_runtime(host, cancellations)
            return host
        original = _combine_startup_failure(
            "runtime assembly",
            captured.error,
            cancellations,
        )
        await cls._araise_failed_async_assembly(
            host,
            store,
            original,
            owned_store_close_reservation=owned_store_close_reservation,
        )
        raise AssertionError("failed Runtime assembly unexpectedly returned")

    @classmethod
    async def _araise_failed_async_assembly(
        cls,
        host: Runtime,
        store: RuntimeStore,
        original: BaseException,
        *,
        owned_store_close_reservation: Any | None,
    ) -> None:
        cls._attach_partial_runtime(original, host)
        reservation_failures = cls._owned_store_close_reservation_failures(
            host,
            store,
            owned_store_close_reservation,
        )
        try:
            cleanup_errors = await cls._drain_async_failed_assembly(host)
        except _CompletedCleanupCancellation as outcome:
            failures: list[BaseException] = [
                original,
                *outcome.cancellations,
                *reservation_failures,
            ]
            if (
                reservation_failures
                or outcome.cleanup_errors
                or outcome.cleanup_exception is not None
            ):
                cleanup_records = [
                    RuntimeAssemblyCleanupRequired._error_record(
                        "store_close_reservation",
                        item,
                    )
                    for item in reservation_failures
                ]
                cleanup_records.extend(dict(item) for item in outcome.cleanup_errors)
                if outcome.cleanup_exception is not None:
                    cleanup_records.append(
                        RuntimeAssemblyCleanupRequired._error_record(
                            "runtime_graph",
                            outcome.cleanup_exception,
                        )
                    )
                failures.append(
                    RuntimeAssemblyCleanupRequired(
                        partial_runtime=host,
                        store=store,
                        cleanup_errors=cleanup_records,
                        cleanup_completed=(
                            bool(reservation_failures)
                            and not outcome.cleanup_errors
                            and outcome.cleanup_exception is None
                        ),
                    )
                )
                if outcome.cleanup_exception is not None:
                    failures.append(outcome.cleanup_exception)
            raise BaseExceptionGroup(
                "runtime assembly failed and caller cancelled after cleanup",
                failures,
            ) from original
        except BaseException as cleanup_error:
            cleanup_records = [
                RuntimeAssemblyCleanupRequired._error_record(
                    "store_close_reservation",
                    item,
                )
                for item in reservation_failures
            ]
            cleanup_records.append(
                RuntimeAssemblyCleanupRequired._error_record(
                    "runtime_graph",
                    cleanup_error,
                )
            )
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=host,
                store=store,
                cleanup_errors=cleanup_records,
            )
            raise BaseExceptionGroup(
                "runtime assembly and cleanup failed",
                [original, handle, *reservation_failures, cleanup_error],
            ) from original
        reservation_records = [
            RuntimeAssemblyCleanupRequired._error_record(
                "store_close_reservation",
                item,
            )
            for item in reservation_failures
        ]
        cls._raise_assembly_cleanup_errors(
            original,
            [*reservation_records, *cleanup_errors],
            host=host,
            store=store,
            cleanup_completed=bool(reservation_failures) and not cleanup_errors,
            secondary_failures=reservation_failures,
        )
        raise original

    @classmethod
    async def _drain_cancelled_open_runtime(
        cls,
        host: Runtime,
        assembly_cancellations: tuple[BaseException, ...],
    ) -> None:
        """Normally shut down a Runtime that reached OPEN after cancellation."""

        shutdown_task = asyncio.create_task(
            host.ashutdown(
                actor="runtime.builder",
                reason="runtime.aopen.cancelled_after_assembly",
            )
        )
        (
            shutdown_result,
            shutdown_error,
            drained_cancellations,
        ) = await _drain_startup_task(
            shutdown_task,
            initial_cancellations=assembly_cancellations,
        )
        cancellations = list(drained_cancellations)
        if shutdown_error is not None:
            failures: list[BaseException] = [*cancellations]
            if not host.lifecycle.closed:
                failures.append(
                    RuntimeAssemblyCleanupRequired(
                        partial_runtime=host,
                        store=host.store,
                        cleanup_errors=[
                            RuntimeAssemblyCleanupRequired._error_record(
                                "runtime_shutdown",
                                shutdown_error,
                            )
                        ],
                        cleanup_kind=(
                            RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN
                        ),
                    )
                )
            failures.append(shutdown_error)
            failure = _CompletedAssemblyCancellation(
                "runtime assembly completed after cancellation but normal shutdown failed",
                failures,
            )
            raise failure from shutdown_error
        if not (
            isinstance(shutdown_result, dict)
            and shutdown_result.get("ok") is True
            and host.lifecycle.closed
        ):
            shutdown_error = RuntimeError(
                "runtime assembly completed after cancellation but normal shutdown "
                f"remains incomplete: {shutdown_result!r}"
            )
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=host,
                store=host.store,
                cleanup_errors=[
                    RuntimeAssemblyCleanupRequired._error_record(
                        "runtime_shutdown",
                        shutdown_error,
                    )
                ],
                cleanup_kind=RuntimeAssemblyCleanupKind.OPEN_RUNTIME_SHUTDOWN,
            )
            failure = _CompletedAssemblyCancellation(
                "runtime assembly completed after cancellation but remains owned",
                [*cancellations, handle, shutdown_error],
            )
            raise failure from shutdown_error
        cancellation = _CompletedAssemblyCancellation(
            "runtime assembly completed and was shut down after caller cancellation",
            cancellations,
        )
        raise cancellation

    @classmethod
    async def _drain_async_failed_assembly(
        cls,
        host: Runtime,
    ) -> list[dict[str, str]]:
        """Finish cleanup even when the async open caller is cancelled."""

        cleanup_task = asyncio.create_task(cls._acleanup_failed_assembly(host))
        caller_task = asyncio.current_task()
        cancellations: list[BaseException] = []
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                if caller_task is None or caller_task.cancelling() == 0:
                    break
                caller_task.uncancel()
                cancellations.append(exc)
        try:
            cleanup_errors = cleanup_task.result()
        except BaseException as cleanup_error:
            if cancellations:
                raise _CompletedCleanupCancellation(
                    cancellations,
                    [],
                    cleanup_error,
                ) from cleanup_error
            raise
        if cancellations:
            raise _CompletedCleanupCancellation(
                cancellations,
                cleanup_errors,
            )
        return cleanup_errors

    @classmethod
    def _assemble_host(
        cls,
        host: Runtime,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        substrate: ResourceProviderSubstrate | None,
        config: AgentLibOSConfig | None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
    ) -> None:
        cls._configure_foundation(host, store, substrate=substrate, config=config)
        cls._configure_evidence_and_authority(host)
        cls._configure_host_services(host)
        cls._configure_human_and_primitives(host)
        cls._configure_execution_services(host)
        cls._configure_tail(
            host,
            store,
            llm_client=llm_client,
            startup_module_manifests=startup_module_manifests,
            trusted_modules=trusted_modules,
            trusted_module_sha256=trusted_module_sha256,
        )

    @staticmethod
    def _require_sync_assembly_context() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "Runtime.open cannot assemble inside an active event loop; "
            "use await Runtime.aopen(...)"
        )

    @staticmethod
    def _owned_store_close_reservation_failures(
        host: Runtime,
        store: RuntimeStore,
        close_reservation: Any | None,
    ) -> list[BaseException]:
        try:
            RuntimeBuilder._reserve_owned_store_close(
                host,
                store,
                close_reservation,
            )
        except BaseException as reservation_error:
            return [reservation_error]
        return []

    @staticmethod
    def _reserve_owned_store_close(
        host: Runtime,
        store: RuntimeStore,
        close_reservation: Any | None,
    ) -> None:
        if close_reservation is None:
            return
        replace_guard = getattr(store, "replace_admission_commit_guard", None)
        if not callable(replace_guard):
            raise RuntimeError(
                "runtime store does not support exact failed-open close reservation"
            )
        if RuntimeBuilder._ensure_owned_store_close_reservation(
            host,
            store,
            close_reservation,
        ):
            return
        raise RuntimeError(
            "failed-open store ownership changed before close reservation"
        )

    @staticmethod
    def _ensure_owned_store_close_reservation(
        host: Runtime | None,
        store: RuntimeStore,
        close_reservation: Any,
    ) -> bool:
        replace_guard = getattr(store, "replace_admission_commit_guard", None)
        if not callable(replace_guard):
            raise RuntimeError(
                "runtime store does not support exact failed-open close reservation"
            )
        # Idempotent retry first: an exact reservation already installed by the
        # failed assembly must remain the same callable identity.
        if replace_guard(close_reservation, close_reservation):
            return True
        lifecycle = getattr(host, "lifecycle", None)
        expected_guard = (
            getattr(lifecycle, "_admission_commit_guard_binding", None)
            if lifecycle is not None
            else None
        )
        if replace_guard(expected_guard, close_reservation):
            return True
        # A completed graph cleanup may already have identity-unbound the old
        # lifecycle guard after an earlier reservation attempt failed.
        return expected_guard is not None and replace_guard(None, close_reservation)

    @staticmethod
    def _try_repair_owned_store_close_reservation(
        host: Runtime | None,
        store: RuntimeStore,
        close_reservation: Any,
    ) -> StoreCloseClaimOutcome:
        """Nonblockingly repair an exact failed-open reservation."""

        replace_guard = getattr(
            store,
            "try_replace_admission_commit_guard",
            None,
        )
        if not callable(replace_guard):
            raise RuntimeError(
                "runtime store does not support nonblocking close-reservation repair"
            )
        lifecycle = getattr(host, "lifecycle", None)
        expected_guard = (
            getattr(lifecycle, "_admission_commit_guard_binding", None)
            if lifecycle is not None
            else None
        )
        candidates = [expected_guard]
        if expected_guard is not None:
            candidates.append(None)
        outcome = StoreCloseClaimOutcome.GUARD_MISMATCH
        for candidate in candidates:
            outcome = replace_guard(candidate, close_reservation)
            if not isinstance(outcome, StoreCloseClaimOutcome):
                raise RuntimeError(
                    "runtime store returned an invalid close-reservation repair outcome"
                )
            if outcome is not StoreCloseClaimOutcome.GUARD_MISMATCH:
                return outcome
        return outcome

    @staticmethod
    def _close_owned_store_after_failed_open(
        store: RuntimeStore,
        original: BaseException,
        *,
        close_reservation: Any,
        close_preclaimed: bool = False,
    ) -> None:
        handles = tuple(
            handle
            for handle in RuntimeAssemblyCleanupRequired.extract(original)
            if handle.store is store
        )
        if RuntimeBuilder._terminalize_released_store_handles(
            store,
            close_reservation,
            handles,
        ):
            return
        if RuntimeBuilder._publish_owned_store_cleanup_handles(
            store,
            original,
            close_reservation=close_reservation,
            handles=handles,
        ):
            return

        # Allocation can fail before assembly installs the reservation. Claim
        # the still-unowned guard slot here; a false result is deliberately not
        # treated as permission to close because it can also mean that a
        # successor lifecycle already owns the store.
        if not close_preclaimed:
            replace_guard = getattr(store, "replace_admission_commit_guard", None)
            if not callable(replace_guard):
                ownership_error = RuntimeError(
                    "runtime store does not support exact failed-open close reservation"
                )
                raise BaseExceptionGroup(
                    "runtime open failed without safe owned-store cleanup",
                    [original, ownership_error],
                ) from original
            try:
                replace_guard(None, close_reservation)
            except BaseException as reservation_error:
                handle = RuntimeAssemblyCleanupRequired(
                    partial_runtime=RuntimeBuilder._partial_runtime_from_error(
                        original
                    ),
                    store=store,
                    cleanup_errors=[
                        RuntimeAssemblyCleanupRequired._error_record(
                            "store_close_reservation",
                            reservation_error,
                        )
                    ],
                    cleanup_completed=True,
                )
                handle._claim_owned_store(store, close_reservation)
                raise BaseExceptionGroup(
                    "runtime open failed and store close reservation remains retryable",
                    [original, handle, reservation_error],
                ) from original
        RuntimeBuilder._finish_owned_store_close_after_failed_open(
            store,
            original,
            close_reservation=close_reservation,
        )

    @staticmethod
    def _terminalize_released_store_handles(
        store: RuntimeStore,
        close_reservation: Any,
        handles: tuple[RuntimeAssemblyCleanupRequired, ...],
    ) -> bool:
        probe = getattr(store, "probe_admission_guard_close", None)
        if callable(probe):
            try:
                readiness = probe(close_reservation)
            except BaseException:
                readiness = None
            if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
                for handle in handles:
                    handle._terminalize_released_store_ownership()
                return True
        return False

    @staticmethod
    def _publish_owned_store_cleanup_handles(
        store: RuntimeStore,
        original: BaseException,
        *,
        close_reservation: Any,
        handles: tuple[RuntimeAssemblyCleanupRequired, ...],
    ) -> bool:
        if not handles:
            return False
        for handle in handles:
            handle._claim_owned_store(store, close_reservation)
        partial_runtime = RuntimeBuilder._partial_runtime_from_error(original)
        try:
            reserved = RuntimeBuilder._ensure_owned_store_close_reservation(
                partial_runtime,
                store,
                close_reservation,
            )
        except BaseException as reservation_error:
            raise BaseExceptionGroup(
                "runtime open failed and store close reservation remains retryable",
                [original, *handles, reservation_error],
            ) from original
        if reserved:
            return True
        if RuntimeBuilder._terminalize_released_store_handles(
            store,
            close_reservation,
            handles,
        ):
            return True
        ownership_error = RuntimeError(
            "failed-open store ownership changed before cleanup handle publication"
        )
        raise BaseExceptionGroup(
            "runtime open lost its exact store close reservation",
            [original, *handles, ownership_error],
        ) from original

    @staticmethod
    def _finish_owned_store_close_after_failed_open(
        store: RuntimeStore,
        original: BaseException,
        *,
        close_reservation: Any,
    ) -> None:
        try:
            close_outcome = store.release_admission_guard_and_close(
                close_reservation
            )
        except BaseException as close_error:
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=RuntimeBuilder._partial_runtime_from_error(original),
                store=store,
                cleanup_errors=[
                    RuntimeAssemblyCleanupRequired._error_record(
                        "store",
                        close_error,
                    )
                ],
                cleanup_completed=True,
            )
            handle._claim_owned_store(store, close_reservation)
            raise BaseExceptionGroup(
                "runtime open and owned store cleanup failed",
                [original, handle, close_error],
            ) from original
        ownership_error = RuntimeAssemblyCleanupRequired._owned_store_close_outcome_error(
            close_outcome
        )
        if ownership_error is not None:
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=RuntimeBuilder._partial_runtime_from_error(original),
                store=store,
                cleanup_errors=[
                    RuntimeAssemblyCleanupRequired._error_record(
                        "store",
                        ownership_error,
                    )
                ],
                cleanup_completed=True,
            )
            handle._claim_owned_store(store, close_reservation)
            raise BaseExceptionGroup(
                "runtime open lost its exact owned-store close reservation",
                [original, handle, ownership_error],
            ) from original
        if close_outcome.warnings:
            raise BaseExceptionGroup(
                "runtime open failed and store ownership was released with warnings",
                [original, *close_outcome.warnings],
            ) from original

    @staticmethod
    def _prepare_async_owned_store_close_after_failed_open(
        store: RuntimeStore,
        original: BaseException,
        *,
        close_reservation: Any,
    ) -> bool:
        """Publish existing handles or atomically claim a worker-thread close."""

        handles = tuple(
            handle
            for handle in RuntimeAssemblyCleanupRequired.extract(original)
            if handle.store is store
        )
        partial_runtime = RuntimeBuilder._partial_runtime_from_error(original)
        probe = getattr(store, "probe_admission_guard_close", None)
        if not callable(probe):
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=RuntimeError(
                    "runtime store does not support nonblocking guarded-close preflight"
                ),
            )
        try:
            readiness = probe(close_reservation)
        except BaseException as probe_error:
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=probe_error,
            )
        if not isinstance(readiness, StoreCloseClaimOutcome):
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=RuntimeError(
                    "runtime store returned an invalid guarded-close preflight outcome"
                ),
            )
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            for handle in handles:
                handle._terminalize_released_store_ownership()
            return False
        for handle in handles:
            handle._claim_owned_store(store, close_reservation)
        if readiness is StoreCloseClaimOutcome.GUARD_MISMATCH:
            try:
                readiness = RuntimeBuilder._try_repair_owned_store_close_reservation(
                    partial_runtime,
                    store,
                    close_reservation,
                )
            except BaseException as repair_error:
                RuntimeBuilder._raise_async_owned_store_close_not_ready(
                    store,
                    original,
                    handles=handles,
                    close_reservation=close_reservation,
                    error=repair_error,
                )
        if readiness is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            for handle in handles:
                handle._terminalize_released_store_ownership()
            return False
        if readiness is not StoreCloseClaimOutcome.READY:
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=RuntimeError(
                    "runtime open owned-store close is not ready: "
                    f"{readiness.value}"
                ),
            )
        if handles:
            return False
        claim = getattr(store, "claim_admission_guard_close", None)
        if not callable(claim):
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=RuntimeError(
                    "runtime store does not support atomic guarded-close claims"
                ),
            )
        try:
            claim_outcome = claim(close_reservation)
        except BaseException as claim_error:
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=claim_error,
            )
        if claim_outcome is StoreCloseClaimOutcome.OWNERSHIP_RELEASED:
            return False
        if claim_outcome is not StoreCloseClaimOutcome.READY:
            error = RuntimeError(
                "runtime open owned-store close claim is not ready: "
                f"{getattr(claim_outcome, 'value', claim_outcome)!s}"
            )
            RuntimeBuilder._raise_async_owned_store_close_not_ready(
                store,
                original,
                handles=handles,
                close_reservation=close_reservation,
                error=error,
            )
        return True

    @staticmethod
    def _raise_async_owned_store_close_not_ready(
        store: RuntimeStore,
        original: BaseException,
        *,
        handles: tuple[RuntimeAssemblyCleanupRequired, ...],
        close_reservation: Any,
        error: BaseException,
    ) -> None:
        published_handles = handles
        if not published_handles:
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=RuntimeBuilder._partial_runtime_from_error(original),
                store=store,
                cleanup_errors=[
                    RuntimeAssemblyCleanupRequired._error_record("store", error)
                ],
                cleanup_completed=True,
            )
            handle._claim_owned_store(store, close_reservation)
            published_handles = (handle,)
        else:
            for handle in published_handles:
                handle._claim_owned_store(store, close_reservation)
        raise BaseExceptionGroup(
            "runtime open failed and owned-store close remains retryable",
            [original, *published_handles, error],
        ) from original

    @staticmethod
    async def _aclose_owned_store_after_failed_open(
        store: RuntimeStore,
        original: BaseException,
        *,
        close_reservation: Any,
    ) -> None:
        should_close = RuntimeBuilder._prepare_async_owned_store_close_after_failed_open(
            store,
            original,
            close_reservation=close_reservation,
        )
        if not should_close:
            return

        def capture_close() -> BaseException | None:
            try:
                RuntimeBuilder._close_owned_store_after_failed_open(
                    store,
                    original,
                    close_reservation=close_reservation,
                    close_preclaimed=True,
                )
            except BaseException as close_error:
                return close_error
            return None

        close_task = asyncio.create_task(run_blocking_once(capture_close))
        caller_task = asyncio.current_task()
        cancellations: list[BaseException] = []
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                if caller_task is None or caller_task.cancelling() == 0:
                    break
                caller_task.uncancel()
                cancellations.append(exc)
        try:
            close_error = close_task.result()
        except BaseException as unexpected:
            close_error = unexpected
        if close_error is not None:
            if cancellations:
                raise BaseExceptionGroup(
                    "runtime open and owned-store cleanup were interrupted",
                    [close_error, *cancellations],
                ) from original
            raise close_error
        if cancellations:
            raise BaseExceptionGroup(
                "runtime open failed and owned-store cleanup completed after cancellation",
                [original, *cancellations],
            ) from original

    @staticmethod
    def _attach_partial_runtime(error: BaseException, host: Runtime) -> None:
        try:
            setattr(error, "_agent_libos_partial_runtime", host)
        except BaseException:
            return

    @staticmethod
    def _partial_runtime_from_error(error: BaseException) -> Runtime | None:
        host = getattr(error, "_agent_libos_partial_runtime", None)
        if host is not None:
            return host
        if isinstance(error, BaseExceptionGroup):
            for item in error.exceptions:
                host = RuntimeBuilder._partial_runtime_from_error(item)
                if host is not None:
                    return host
        return None

    @staticmethod
    def _raise_assembly_cleanup_errors(
        original: BaseException,
        cleanup_errors: list[dict[str, str]],
        *,
        host: Runtime,
        store: RuntimeStore,
        cleanup_completed: bool = False,
        secondary_failures: list[BaseException] | None = None,
    ) -> None:
        if not cleanup_errors:
            return
        handle = RuntimeAssemblyCleanupRequired(
            partial_runtime=host,
            store=store,
            cleanup_errors=cleanup_errors,
            cleanup_completed=cleanup_completed,
        )
        raise BaseExceptionGroup(
            "runtime assembly and cleanup failed",
            [original, handle, *(secondary_failures or [])],
        ) from original

    @staticmethod
    def _configure_foundation(
        host: Runtime,
        store: RuntimeStore,
        *,
        substrate: ResourceProviderSubstrate | None,
        config: AgentLibOSConfig | None,
    ) -> None:
        host.config = config or DEFAULT_CONFIG
        if substrate is None:
            host.substrate = LocalResourceProviderSubstrate(
                Path.cwd().resolve(),
                namespace=host.config.runtime.workspace_namespace,
                git_config=host.config.git,
            )
        else:
            host.substrate = substrate
            if isinstance(substrate, LocalResourceProviderSubstrate):
                substrate.bind_runtime_git_config(host.config.git)
        host.workspace_root = Path(
            getattr(
                host.substrate,
                "workspace_root",
                host.substrate.workspace_display,
            )
        )
        host.store = store
        host.instance_id = new_id("runtime")
        host.store.config = host.config
        host.images = {}
        host.module_state = ModuleStateRegistry()
        host.blocking_work = BlockingWorkSupervisor(
            max_workers=max(
                host.config.scheduler.max_workers,
                host.config.object_tasks.max_running_global,
            ),
            shutdown_timeout_s=max(
                host.config.scheduler.shutdown_join_timeout_s,
                host.config.object_tasks.shutdown_join_timeout_s,
            ),
        )

    @staticmethod
    def _configure_evidence_and_authority(host: Runtime) -> None:
        host.uow = UnitOfWork(host.store)
        host.process_transitions = ProcessTransitionService(host.uow.processes)
        operation_terminalization_capability: object | None = None

        def operation_terminalization_scope(publication_id: str):
            capability = operation_terminalization_capability
            if capability is None:
                raise RuntimeError(
                    "operation recovery terminalization is not configured"
                )
            return host.lifecycle.recovery_terminalization_scope_if_fenced(
                publication_id,
                capability=capability,
            )

        host.operations = OperationManager(
            host.uow.evidence,
            host.uow.publications,
            recovery_page_size=host.config.runtime.operation_recovery_page_size,
            require_recovery_lease=(
                lambda: host.lifecycle.require_recovery_lease()
            ),
            recovery_terminalization_scope=operation_terminalization_scope,
            current_mutation_admission_is_stale=(
                lambda: host.lifecycle.current_mutation_admission_is_stale()
            ),
        )
        host.audit = AuditManager(host.uow.evidence, host.operations)
        host.events = EventBus(host.uow.evidence, host.operations)
        host.lifecycle = RuntimeLifecycle(
            store=host.store,
            audit=host.audit,
            events=host.events,
            substrate=host.substrate,
            admission_drain_timeout_s=min(
                host.config.scheduler.shutdown_join_timeout_s,
                host.config.object_tasks.shutdown_join_timeout_s,
            ),
        )
        operation_terminalization_capability = (
            host.lifecycle._issue_recovery_terminalization_capability()
        )
        host._recovery_diagnostics_release_capability = (
            host.lifecycle._issue_recovery_diagnostics_release_capability()
        )
        host.payload_retention = PayloadRetentionMaintenance(
            host.uow.retention,
            host.audit,
            PayloadRetentionPolicy.from_runtime_defaults(host.config.runtime),
            admission=host.lifecycle,
        )
        host._registry_lifecycle_lock = RuntimeRegistryLock(host.lifecycle)
        host.lifecycle.begin_recovery()
        host.capability = CapabilityManager(
            host.uow.authority,
            host.audit,
            host.events,
            config=host.config,
            operations=host.operations,
            admission=host.lifecycle,
        )

    @staticmethod
    def _configure_host_services(host: Runtime) -> None:
        host.llms = LLMProfileRegistry(host.uow.processes, config=host.config)
        host.ratings = AgentRatingManager(
            host.uow.processes,
            host.audit,
            config=host.config,
        )
        host.resources = ResourceManager(
            host.uow,
            host.audit,
            host.events,
            require_recovery_lease=host.lifecycle.require_recovery_lease,
            transitions=host.process_transitions,
            config=host.config,
        )
        host.syscalls = SyscallRouter(
            host.audit,
            reserved_names=BUILTIN_SYSCALL_NAMES,
        )
        host.provider_hooks = {}
        host.authority_manifests = AuthorityManifestManager(
            host.uow.authority,
            host.capability,
            host.audit,
            host.events,
            host.images,
            config=host.config,
        )
        host.explain = ExplainManager(host.store, host.authority_manifests)
        host.memory = ObjectMemoryManager(
            host.uow,
            host.capability,
            host.audit,
            host.events,
            config=host.config,
            resources=host.resources,
            operations=host.operations,
        )
        host.data_flow = DataFlowManager(
            host.uow.authority,
            host.capability,
            host.audit,
            host.events,
            host.authority_manifests,
            host.uow.objects,
            memory=host.memory,
            config=host.config,
            blocking_work_supervisor=host.blocking_work,
        )
        host.protected_operations = ProtectedOperationSDK(
            effects=host.uow.protected_effects,
            authority_policy=host.authority_manifests,
            capabilities=host.capability,
            audit=host.audit,
            events=host.events,
            resources=host.resources,
            operations=host.operations,
            require_recovery_lease=host.lifecycle.require_recovery_lease,
            data_flow=host.data_flow,
        )
        host.external_primitive_boundary_names = (
            register_protected_operation_descriptors(
                host.protected_operations,
                minimum_integrity=host.config.data_flow.operation_minimum_integrity,
            )
        )
        host.process = ProcessManager(
            host.uow,
            host.memory,
            host.capability,
            host.audit,
            host.events,
            host.lifecycle.require_recovery_lease,
            config=host.config,
            resources=host.resources,
            llm_profile_resolver=host._resolve_launch_llm_profile_id,
            authority_manifests=host.authority_manifests,
            data_flow=host.data_flow,
            object_task_terminal_notifier=host._notify_process_terminal,
            failed_launch_artifact_cleanup=(
                lambda publication: host.image_boot.cleanup_failed_launch_artifacts(publication)
            ),
            owner_instance_id=host.instance_id,
            recovery_required_callback=host.lifecycle.mark_recovery_required,
            recovery_terminalization_scope=partial(
                host.lifecycle.recovery_terminalization_scope,
                capability=(
                    host.lifecycle._issue_recovery_terminalization_capability()
                ),
            ),
            transitions=host.process_transitions,
        )
        host.resources.bind_process_kill_finalizer(
            host.process.finalize_killed_processes
        )
        host.messages = ProcessMessageManager(
            host.uow.processes,
            host.audit,
            host.events,
            host.authority_manifests,
            process_manager=host.process,
            config=host.config,
            transitions=host.process_transitions,
        )

    @staticmethod
    def _configure_human_and_primitives(host: Runtime) -> None:
        host.human = HumanObjectManager(
            host.uow.processes,
            host.uow.authority,
            host.capability,
            host.audit,
            host.events,
            provider=host.substrate.human,
            protected_operations=host.protected_operations,
            authority_policy=host.authority_manifests,
            operations=host.operations,
            requests=host.uow.processes,
            messages=host.messages,
            data_flow=host.data_flow,
            blocking_work=host.blocking_work,
            config=host.config,
            transitions=host.process_transitions,
        )
        host.data_flow.bind_human(host.human)
        with host.lifecycle.recovery_lease():
            host.data_flow.bootstrap_configured_rules()
        host.protected_operations.register_prepared_recovery(
            "human_output_delivery",
            host.human.recover_prepared_output,
        )
        host.clock = ClockPrimitive(
            host.capability,
            host.audit,
            host.events,
            max_sleep_seconds=host.config.tools.max_sleep_seconds,
            provider=host.substrate.clock,
            protected_operations=host.protected_operations,
        )
        host.filesystem = FilesystemAdapter(
            host.capability,
            host.audit,
            host.events,
            protected_operations=host.protected_operations,
            human=host.human,
            provider=host.substrate.filesystem,
            resources=host.resources,
            config=host.config,
        )
        host.shell = ShellAdapter(
            host.capability,
            host.audit,
            host.events,
            protected_operations=host.protected_operations,
            cwd=host.workspace_root,
            human=host.human,
            provider=host.substrate.shell,
            config=host.config,
            resources=host.resources,
        )
        git_provider = getattr(host.substrate, "git", None)
        if git_provider is None:
            git_provider = LocalGitProvider(
                host.workspace_root,
                config=host.config.git,
            )
        host.git = GitPrimitive(
            host.capability,
            host.audit,
            host.events,
            protected_operations=host.protected_operations,
            human=host.human,
            provider=git_provider,
            filesystem=host.filesystem,
            memory=host.memory,
            data_flow=host.data_flow,
            config=host.config,
        )
        host.jsonrpc = JsonRpcPrimitive(
            host.uow,
            host.capability,
            host.audit,
            host.events,
            protected_operations=host.protected_operations,
            human=host.human,
            provider=getattr(
                host.substrate,
                "jsonrpc",
                HttpJsonRpcProvider(),
            ),
            config=host.config,
            resources=host.resources,
        )
        host.mcp = McpPrimitive(
            host.uow,
            host.capability,
            host.audit,
            host.events,
            protected_operations=host.protected_operations,
            human=host.human,
            provider=getattr(
                host.substrate,
                "mcp",
                SdkMcpProvider(host.workspace_root),
            ),
            config=host.config,
            resources=host.resources,
        )

    @staticmethod
    def _configure_execution_services(host: Runtime) -> None:
        host.tools = ToolBroker(
            host.uow,
            host.memory,
            host.capability,
            host.human,
            host.audit,
            host.events,
            workspace_root=host.workspace_root,
            config=host.config,
            resources=host.resources,
            operations=host.operations,
            data_flow=host.data_flow,
            jit_session_factory=lambda pid: LibOSSyscallSession(
                host,
                pid,
                config=host.config,
                transitions=host.process_transitions,
            ),
            tool_context_host=host,
            images=host.images,
            registry_lifecycle_lock=host._registry_lifecycle_lock,
            lifecycle=host.lifecycle,
        )
        host.object_tasks = ObjectTaskManager(
            host.uow.processes,
            host.uow.objects,
            host.process,
            host.tools,
            host.memory,
            host.capability,
            host.audit,
            host.events,
            host.operations,
            host.messages,
            host.authority_manifests,
            host.human,
            # Resolve the Runtime facade at publication time. Operation
            # boundaries are installed after service construction, so
            # capturing the bound method here would permanently retain the
            # pre-admission implementation.
            lambda pid, handle: host.add_handle_to_process_view(pid, handle),
            config=host.config,
            admission=host.lifecycle,
            require_recovery_lease=host.lifecycle.require_recovery_lease,
            autostart=False,
        )
        host.memory.bind_object_pin_checker(
            lambda owner_oid: host.object_tasks.has_active_for_owner(owner_oid)
        )
        host.memory.bind_object_change_notifier(
            lambda owner_oid, change, actor_pid: host.object_tasks.notify_owner_changed(
                owner_oid,
                change,
                actor_pid,
            )
        )
        host.messages.bind_object_tasks(host.object_tasks)
        host.resources.bind_object_task_terminal_notifier(
            host._notify_process_terminal
        )

    @classmethod
    def _configure_tail(
        cls,
        host: Runtime,
        store: RuntimeStore,
        *,
        llm_client: LLMClient | None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
    ) -> None:
        host.scheduler = SimpleScheduler(
            host.uow.processes,
            host.audit,
            poll_interval_s=host.config.scheduler.poll_interval_s,
            max_workers=host.config.scheduler.max_workers,
            drain_window_s=host.config.scheduler.drain_window_s,
            shutdown_join_timeout_s=host.config.scheduler.shutdown_join_timeout_s,
            resources=host.resources,
            skip_pid=host.object_tasks.is_runner_pid,
            cancel_process=host.process.cancel,
            blocking_work=host.blocking_work,
            owner_id=host.instance_id,
            transitions=host.process_transitions,
        )
        cls._configure_checkpoint(host)
        host.skills = SkillManager(
            host.uow,
            host.capability,
            host.audit,
            host.events,
            host.tools,
            host.filesystem,
            host.process,
            host.images,
            host._registry_lifecycle_lock,
            human=host.human,
            config=host.config,
        )
        cls._configure_image_services(host)
        host.modules = RuntimeModuleRegistry(
            host.uow.extensions,
            host.uow.module_publications,
            host.tools,
            host.images,
            host.image_registry,
            host.syscalls,
            host.provider_hooks,
            host.audit,
            host.events,
            ModuleHookServices.from_host(host),
            host._registry_lifecycle_lock,
            lifecycle=host.lifecycle,
            config=host.config,
        )
        host.checkpoint.bind_modules(host.modules)
        cls._configure_image_boot(host)
        with host.lifecycle.recovery_lease():
            host.llm = LLMProcessExecutor(
                unit_of_work=host.uow,
                process=host.process,
                operations=host.operations,
                data_flow=host.data_flow,
                tools=host.tools,
                resources=host.resources,
                llms=host.llms,
                memory=host.memory,
                audit=host.audit,
                events=host.events,
                images=host.images,
                messages=host.messages,
                human=host.human,
                skills=host.skills,
                protected_operations=host.protected_operations,
                authority_manifests=host.authority_manifests,
                capabilities=host.capability,
                client=llm_client,
                config=host.config,
                blocking_work=host.blocking_work,
            )
        host.lifecycle.bind_components(
            scheduler=host.scheduler,
            object_tasks=host.object_tasks._issue_lifecycle_shutdown_handle(),
            modules=host.modules,
            llms=host.llms,
            blocking_work=host.blocking_work,
        )
        cls._load_extensions_during_recovery(
            host,
            startup_module_manifests=startup_module_manifests,
            trusted_modules=trusted_modules,
            trusted_module_sha256=trusted_module_sha256,
        )
        cls._finish_startup(host)

    @staticmethod
    def _commit_startup_payload_delivery_ack(
        host: Runtime,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> tuple[BaseException | None, bool]:
        """Commit one ACK and classify an ambiguous failure without replay."""

        try:
            with host.lifecycle.open_on_next_commit():
                # The acknowledgement CAS must join this exact outer
                # transaction. A false CAS raises before its sole commit can
                # consume the lifecycle OPEN scope.
                with host.uow.transaction():
                    host.checkpoint._ack_startup_payload_delivery(attempt)
        except BaseException as acknowledgement_error:
            try:
                attempt_state = (
                    host.checkpoint._get_startup_payload_delivery_attempt_state(
                        attempt
                    )
                )
            except BaseException as confirmation_error:
                return (
                    BaseExceptionGroup(
                        "checkpoint payload acknowledgement failed and its "
                        "exact durable state could not be confirmed",
                        [acknowledgement_error, confirmation_error],
                    ),
                    False,
                )

            if attempt_state is CheckpointPayloadDeliveryAttemptState.PREPARING:
                # The acknowledgement did not commit. The compensation path
                # may safely reopen this exact token.
                return acknowledgement_error, True
            if attempt_state is not CheckpointPayloadDeliveryAttemptState.ACKED:
                state_error = RuntimeError(
                    "checkpoint payload acknowledgement exact-state "
                    "confirmation was not compensable: "
                    f"{attempt_state!r}"
                )
                return (
                    BaseExceptionGroup(
                        "checkpoint payload acknowledgement failed closed after "
                        "an inconclusive durable-state confirmation",
                        [acknowledgement_error, state_error],
                    ),
                    False,
                )

            # The commit completed even though the backend raised afterwards.
            # Preserve the hydrated payload cache and repair only the in-memory
            # lifecycle projection.
            try:
                if host.lifecycle.state == "starting":
                    with host.lifecycle.in_memory_open_scope():
                        pass
                elif host.lifecycle.state != "open":
                    raise RuntimeError(
                        "durably acknowledged checkpoint payload delivery has "
                        "an invalid runtime lifecycle state: "
                        f"{host.lifecycle.state}"
                    )
            except BaseException as lifecycle_error:
                return (
                    BaseExceptionGroup(
                        "checkpoint payload acknowledgement committed but "
                        "lifecycle OPEN could not be confirmed",
                        [acknowledgement_error, lifecycle_error],
                    ),
                    False,
                )
            return None, False
        return None, False

    @classmethod
    def _finish_startup(cls, host: Runtime) -> None:
        """Install recovered boundaries and publish the final OPEN transition."""

        cls._install_operation_boundaries(host)
        cls._recover_runtime_state(host)
        host.lifecycle.begin_starting()
        payload_delivery_attempt = None
        payload_delivery_should_compensate = False

        try:
            with host.lifecycle.startup_lease():
                host.modules.run_startup_hooks()
                host.object_tasks.start_worker()
                payload_delivery_attempt = (
                    host.checkpoint._begin_startup_payload_delivery()
                )
                if payload_delivery_attempt is not None:
                    payload_delivery_should_compensate = True
                    host.checkpoint._prepare_startup_payload_delivery(
                        payload_delivery_attempt
                    )
                    host.checkpoint._complete_startup_payload_delivery(
                        payload_delivery_attempt
                    )
                # Delivery and operation truth are independent projections.
                # Hooks can dirty a historical terminal operation even when
                # this startup has no payload backlog, so repair immediately
                # before either OPEN path.
                host.checkpoint.reconcile_terminal_restore_publications()
                if payload_delivery_attempt is None:
                    # With no durable payload backlog there is no
                    # acknowledgement transaction to linearize. Keep the pure
                    # memory transition inside this exact startup lease so an
                    # exception can roll it back to STARTING.
                    with host.lifecycle.in_memory_open_scope():
                        pass
                    return

                (
                    acknowledgement_error,
                    payload_delivery_should_compensate,
                ) = cls._commit_startup_payload_delivery_ack(
                    host,
                    payload_delivery_attempt,
                )
                if acknowledgement_error is not None:
                    raise acknowledgement_error
        except BaseException as startup_error:
            if (
                payload_delivery_attempt is not None
                and payload_delivery_should_compensate
                and host.lifecycle.state == "starting"
            ):
                try:
                    with host.lifecycle.startup_lease():
                        host.checkpoint._reopen_startup_payload_delivery(
                            payload_delivery_attempt
                        )
                except BaseException as compensation_error:
                    handle = RuntimeAssemblyCleanupRequired(
                        partial_runtime=host,
                        store=host.store,
                        cleanup_errors=[
                            RuntimeAssemblyCleanupRequired._error_record(
                                "checkpoint_payload_delivery",
                                compensation_error,
                            )
                        ],
                    )
                    raise BaseExceptionGroup(
                        "runtime startup and checkpoint payload delivery compensation failed",
                        [startup_error, compensation_error, handle],
                    ) from startup_error
            raise

    @staticmethod
    def _configure_checkpoint(host: Runtime) -> None:
        host.checkpoint = CheckpointManager(
            host.uow,
            host.audit,
            host.events,
            host.capability,
            scheduler=host.scheduler,
            registry_lifecycle_lock=host._registry_lifecycle_lock,
            memory=host.memory,
            images=host.images,
            authority_manifests=host.authority_manifests,
            tools=host.tools,
            resources=host.resources,
            messages=host.messages,
            operations=host.operations,
            owner_instance_id=host.instance_id,
            checkpoint_publication_writer=host.uow.checkpoint_restore_publications,
            recovery_required_callback=host.lifecycle.mark_recovery_required,
            require_recovery_lease=host.lifecycle.require_recovery_lease,
            recovery_terminalization_scope=partial(
                host.lifecycle.recovery_terminalization_scope,
                capability=(
                    host.lifecycle._issue_recovery_terminalization_capability()
                ),
            ),
            transitions=host.process_transitions,
            config=host.config,
        )

    @staticmethod
    def _recover_runtime_state(host: Runtime) -> None:
        with host.lifecycle.recovery_lease():
            recovery_page_size = (
                host.config.runtime.external_effect_recovery_page_size
            )
            host.recovered_prepared_operations = host.protected_operations.recover_prepared(
                page_size=recovery_page_size,
            )
            host.recovered_capability_use_reservations = (
                host.uow.authority.abandon_stale_capability_use_reservations(
                    require_recovery_lease=host.lifecycle.require_recovery_lease,
                )
            )
            host.reconciled_external_effects = reconcile_pending_external_effects(
                host.uow.protected_effects,
                host.substrate,
                require_recovery_lease=host.lifecycle.require_recovery_lease,
                page_size=recovery_page_size,
                provider_overrides={"git": host.git.provider},
            )
            host.recovered_resource_usage_reservations = host.resources.recover_usage_reservations()
            host.recovered_exec_publications = host.image_boot.recover_incomplete_publications()
            host.recovered_runtime_publications = host.process.recover_incomplete_publications()
            host.recovered_checkpoint_restore_publications = (
                host.checkpoint.recover_incomplete_restore_publications()
            )
            # A pending checkpoint restore rehydrates its hash-anchored Object
            # payloads before the general volatile-payload sweep releases rows
            # whose process-local cache was genuinely lost on reopen.
            host.recovered_missing_object_payloads = (
                host.uow.objects.recover_missing_runtime_object_payloads(
                    require_recovery_lease=host.lifecycle.require_recovery_lease,
                )
            )
            host.tools.rehydrate_registered_jit_tools()
            host.recovered_stale_operations = host.operations.interrupt_stale_running()
            host.recovered_stale_executions = host.uow.processes.recover_stale_executions(
                owner_id=host.instance_id,
                require_recovery_lease=host.lifecycle.require_recovery_lease,
                on_recovered=partial(
                    RuntimeBuilder._record_stale_execution_recovery,
                    host,
                ),
            )
            host.recovered_object_tasks = host.object_tasks.recover()

    @staticmethod
    def _record_stale_execution_recovery(host: Runtime, pid: str) -> None:
        host.events.emit(
            EventType.PROCESS_SIGNAL,
            source="runtime.recovery",
            target=pid,
            payload={"pid": pid, "reason": "stale_execution_recovery"},
        )
        host.audit.record(
            actor="runtime.recovery",
            action="stale_execution_recovery",
            target=f"process:{pid}",
            decision={"status": "paused", "owner_instance_id": host.instance_id},
        )

    @staticmethod
    def _configure_image_services(host: Runtime) -> None:
        host.process_exec_state = ProcessExecStateService(
            host.uow.snapshots,
            host.memory,
            host.tools,
        )
        host.image_artifacts = ImageArtifactLoader(
            host.uow.extensions,
            host.config,
        )
        host.checkpoint_image_installer = CheckpointImageInstaller(
            loader=host.image_artifacts,
            unit_of_work=host.uow,
            memory=host.memory,
            capabilities=host.capability,
            authority_manifests=host.authority_manifests,
            checkpoint=host.checkpoint,
            tools=host.tools,
            audit=host.audit,
        )
        host.image_package_installer = ImagePackageInstaller(
            loader=host.image_artifacts,
            processes=host.uow.processes,
            publications=host.uow.publications,
            extensions=host.uow.extensions,
            tools=host.tools,
            filesystem=host.filesystem,
            resources=host.resources,
            audit=host.audit,
            workspace_root=host.workspace_root,
            config=host.config,
        )
        host.image_registry = ImageRegistryPrimitive(
            host.images,
            host.capability,
            host.audit,
            host.events,
            host.tools,
            host.checkpoint,
            host.filesystem,
            host.process.working_directory,
            host._registry_lifecycle_lock,
            store=host.uow.extensions,
            config=host.config,
        )
        host.checkpoint.bind_image_registry(host.image_registry)
        host.launch = ProcessLaunchService(
            process=host.process,
            capabilities=host.capability,
            filesystem=host.filesystem,
            images=host.images,
            image_resource=host.image_registry.resource_for,
            config=host.config,
        )

    @staticmethod
    def _configure_image_boot(host: Runtime) -> None:
        host.image_boot = ImageBootService(
            process=host.process,
            launch=host.launch,
            audit=host.audit,
            checkpoint=host.checkpoint,
            authority_manifests=host.authority_manifests,
            modules=host.modules,
            tools=host.tools,
            skills=host.skills,
            exec_state=host.process_exec_state,
            checkpoint_installer=host.checkpoint_image_installer,
            package_installer=host.image_package_installer,
            unit_of_work=host.uow,
            operations=host.operations,
            owner_instance_id=host.instance_id,
            recovery_max_attempts=(
                host.config.runtime.publication_recovery_max_attempts
            ),
            reconciliation_page_size=(
                host.config.runtime.publication_reconciliation_page_size
            ),
            publication_lock=host._registry_lifecycle_lock,
            recovery_required_callback=host.lifecycle.mark_recovery_required,
            require_recovery_lease=host.lifecycle.require_recovery_lease,
        )
        host.process.add_before_spawn_hook(host.image_boot.preflight_id)
        host.process.add_after_spawn_hook(host.image_boot.configure_spawn)

    @staticmethod
    def _load_extensions(
        host: Runtime,
        *,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
        rehydrate_jit: bool = True,
    ) -> None:
        host.modules.load_core_module()
        host.modules.load_startup_modules(
            startup_module_manifests,
            trusted_modules=trusted_modules,
            trusted_sha256=trusted_module_sha256,
        )
        host.image_registry.load_persisted_images()
        if rehydrate_jit:
            host.tools.rehydrate_registered_jit_tools()

    @classmethod
    def _load_extensions_during_recovery(
        cls,
        host: Runtime,
        *,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None,
        trusted_modules: list[str] | tuple[str, ...] | None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None,
    ) -> None:
        """Run admission-fenced module publication under the recovery token."""

        with host.lifecycle.recovery_lease():
            cls._load_extensions(
                host,
                startup_module_manifests=startup_module_manifests,
                trusted_modules=trusted_modules,
                trusted_module_sha256=trusted_module_sha256,
                rehydrate_jit=False,
            )
            host.skills.reconcile_builtin_projection_image_ceilings()

    @staticmethod
    def _install_operation_boundaries(host: Runtime) -> None:
        components = {
            "authority_manifests": host.authority_manifests,
            "capability": host.capability,
            "checkpoint": host.checkpoint,
            "clock": host.clock,
            "data_flow": host.data_flow,
            "filesystem": host.filesystem,
            "git": host.git,
            "human": host.human,
            "image_registry": host.image_registry,
            "image_boot": host.image_boot,
            "jsonrpc": host.jsonrpc,
            "mcp": host.mcp,
            "memory": host.memory,
            "messages": host.messages,
            "modules": host.modules,
            "object_tasks": host.object_tasks,
            "process": host.process,
            "runtime": host,
            "scheduler": host.scheduler,
            "shell": host.shell,
            "skills": host.skills,
            "tools": host.tools,
        }
        installed = install_explain_boundaries(
            components=components,
            operations=host.operations,
            descriptors=EXPLAIN_BOUNDARY_DESCRIPTORS,
            admission=host.lifecycle,
        )
        installed_control = install_control_mutation_admission_boundaries(
            components=components,
            boundaries=CONTROL_MUTATION_ADMISSION_BOUNDARIES,
            admission=host.lifecycle,
        )
        host.mutation_admission_boundary_names = frozenset(
            installed | installed_control
        )
        if host.mutation_admission_boundary_names != PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES:
            raise RuntimeError("public mutation admission inventory drift")
        host.explainable_boundary_names = frozenset(
            installed | host.external_primitive_boundary_names
        )

    @staticmethod
    def _cleanup_failed_assembly(host: Runtime) -> list[dict[str, str]]:
        """Synchronously drain a partial graph after failed assembly."""

        lifecycle = getattr(host, "lifecycle", None)
        if lifecycle is not None:
            RuntimeBuilder._bind_partial_cleanup_components(host, lifecycle)
            return lifecycle.cleanup_failed_assembly()
        errors: list[dict[str, str]] = []
        caught: list[BaseException] = []
        for name in ("blocking_work", "substrate"):
            try:
                stopped = RuntimeLifecycle.shutdown_component(
                    getattr(host, name, None)
                )
                if not stopped:
                    errors.append(
                        {
                            "component": name,
                            "error_type": "ComponentStopDeferred",
                            "error": "returned false",
                        }
                    )
            except BaseException as exc:
                RuntimeBuilder._append_cleanup_error(errors, name, exc)
                caught.append(exc)
        RuntimeBuilder._raise_cleanup_interrupts(caught)
        return errors

    @staticmethod
    async def _acleanup_failed_assembly(host: Runtime) -> list[dict[str, str]]:
        """Drain a partial graph on the async assembly caller's loop."""

        lifecycle = getattr(host, "lifecycle", None)
        if lifecycle is not None:
            RuntimeBuilder._bind_partial_cleanup_components(host, lifecycle)
            return await lifecycle.acleanup_failed_assembly()
        errors: list[dict[str, str]] = []
        caught: list[BaseException] = []
        for name in ("blocking_work", "substrate"):
            try:
                stopped = await RuntimeLifecycle.ashutdown_component(
                    getattr(host, name, None)
                )
                if not stopped:
                    errors.append(
                        {
                            "component": name,
                            "error_type": "ComponentStopDeferred",
                            "error": "returned false",
                        }
                    )
            except BaseException as exc:
                RuntimeBuilder._append_cleanup_error(errors, name, exc)
                caught.append(exc)
        RuntimeBuilder._raise_cleanup_interrupts(caught)
        return errors

    @staticmethod
    def _append_cleanup_error(
        errors: list[dict[str, str]],
        component: str,
        exc: BaseException,
    ) -> None:
        errors.append(
            {
                "component": component,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    @staticmethod
    def _bind_partial_cleanup_components(
        host: Runtime,
        lifecycle: RuntimeLifecycle,
    ) -> None:
        if lifecycle.components_bound:
            return
        object_tasks = getattr(host, "object_tasks", None)
        lifecycle_object_tasks = (
            object_tasks._issue_lifecycle_shutdown_handle()
            if object_tasks is not None
            and hasattr(object_tasks, "_issue_lifecycle_shutdown_handle")
            else object_tasks
        )
        lifecycle.bind_components(
            scheduler=getattr(host, "scheduler", None),
            object_tasks=lifecycle_object_tasks,
            modules=getattr(host, "modules", None),
            llms=getattr(host, "llms", None),
            blocking_work=getattr(host, "blocking_work", None),
        )

    @staticmethod
    def _raise_cleanup_interrupts(caught: list[BaseException]) -> None:
        if any(not isinstance(exc, Exception) for exc in caught):
            raise BaseExceptionGroup(
                "runtime foundation cleanup was interrupted after full teardown",
                caught,
            )

    @classmethod
    def configured(
        cls,
        runtime_type: type[RuntimeT],
        *,
        config: AgentLibOSConfig | None = None,
        substrate: ResourceProviderSubstrate | None = None,
        module_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> "RuntimeBuilder[RuntimeT]":
        return cls(
            runtime_type=runtime_type,
            config=config or DEFAULT_CONFIG,
            substrate=substrate,
            module_manifests=(
                tuple(module_manifests)
                if module_manifests is not None
                else None
            ),
            trusted_modules=(
                tuple(trusted_modules)
                if trusted_modules is not None
                else None
            ),
            trusted_module_sha256=(
                tuple(trusted_module_sha256)
                if trusted_module_sha256 is not None
                else None
            ),
        )


__all__ = [
    "RuntimeAssemblyCleanupKind",
    "RuntimeAssemblyCleanupRequired",
    "RuntimeBuilder",
]
