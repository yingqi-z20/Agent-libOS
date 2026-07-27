from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from agent_libos.storage.engine import SqlSession


class StoreTransaction(SqlSession, Protocol):
    """Backend-neutral cursor used inside a UnitOfWork transaction."""


@dataclass(frozen=True, slots=True)
class StoreCloseOutcome:
    """Exact ownership result of an admission-guard close handoff.

    ``warnings`` contains close diagnostics only after backend ownership has
    crossed its irreversible release point.  A close failure which leaves
    ownership held is raised instead, after restoring the exact guard, so the
    same owner can retry.  ``guard_matched=False`` guarantees that the backend
    was not touched; ``ownership_released=True`` may still report the terminal
    fact that ownership had already been released before a stale-owner call.
    """

    guard_matched: bool
    ownership_released: bool
    warnings: tuple[BaseException, ...] = ()


class StoreCloseClaimOutcome(str, Enum):
    """Nonblocking readiness/result for an exact guarded store close.

    ``OWNERSHIP_RELEASED`` is terminal, not a retry failure: callers may finish
    transient graph teardown without publishing a store-cleanup handle.
    """

    READY = "ready"
    OWNERSHIP_RELEASED = "ownership_released"
    LOCK_BUSY = "lock_busy"
    CURRENT_THREAD_LOCKED = "current_thread_locked"
    ACTIVE_TRANSACTION = "active_transaction"
    GUARD_MISMATCH = "guard_mismatch"


class StoreAssemblyReadiness(str, Enum):
    """Nonblocking readiness for off-thread Runtime graph assembly."""

    READY = "ready"
    LOCK_BUSY = "lock_busy"
    CURRENT_THREAD_LOCKED = "current_thread_locked"
    ACTIVE_TRANSACTION = "active_transaction"


@dataclass(frozen=True, slots=True, eq=False)
class StoreAssemblyReservation:
    """Opaque identity token for one async Runtime assembly handoff.

    Equality is deliberately disabled: only the exact token installed by the
    event-loop thread may be claimed by its startup worker or released by its
    cancellation path.
    """

    reservation_id: str


class RuntimeStore(Protocol):
    """Narrow host boundary for a concrete runtime store.

    Domain persistence is intentionally absent. Callers that need process,
    object, authority, evidence, or extension records consume the matching
    repository from :class:`UnitOfWork`.
    """

    config: Any
    path: str

    def close(self) -> None:
        ...

    def locked(self) -> AbstractContextManager[None]:
        ...

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[StoreTransaction]:
        ...

    def uncertain_commit_confirmation(self) -> AbstractContextManager[None]:
        """Fence one exact post-commit readback on the owning thread."""

        ...

    def accept_uncertain_commit_confirmation(self) -> bool:
        """Restore a poisoned data plane after exact durable confirmation."""

        ...

    def probe_runtime_assembly_readiness(self) -> StoreAssemblyReadiness:
        """Inspect lock/transaction readiness without waiting or mutating state."""

        ...

    def reserve_runtime_assembly(
        self,
        reservation: StoreAssemblyReservation,
    ) -> StoreAssemblyReadiness:
        """Atomically check readiness and fence ordinary store scopes."""

        ...

    def claim_runtime_assembly(
        self,
        reservation: StoreAssemblyReservation,
        *,
        require_unbound_runtime: bool = False,
    ) -> AbstractContextManager[None]:
        """Activate an exact reservation for the calling startup worker.

        Builders set ``require_unbound_runtime`` so an already-published
        Runtime is rejected before any dependency graph is attached. Direct
        Store reservation users retain re-entrant diagnostic/query access.
        """

        ...

    def release_runtime_assembly_reservation(
        self,
        reservation: StoreAssemblyReservation,
    ) -> bool:
        """Release an exact unclaimed reservation after startup cancellation."""

        ...

    def bind_admission_commit_guard(
        self,
        guard: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """Bind the lifecycle fence held across each outer durable commit."""

        ...

    def unbind_admission_commit_guard(
        self,
        guard: Callable[[], AbstractContextManager[None]],
    ) -> bool:
        """Release one exact failed-assembly guard without replacing its owner."""

        ...

    def replace_admission_commit_guard(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]] | None,
        replacement_guard: Callable[[], AbstractContextManager[None]],
    ) -> bool:
        """Atomically replace an exact guard with a failed-open close claim."""

        ...

    def try_replace_admission_commit_guard(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]] | None,
        replacement_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Nonblockingly replace an exact guard for failed-open close."""

        ...

    def claim_admission_guard_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Nonblockingly reserve one exact guard for worker-thread close."""

        ...

    def probe_admission_guard_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Nonblockingly inspect close readiness without reserving it."""

        ...

    def release_admission_guard_and_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseOutcome:
        """Atomically release an exact guard and report backend ownership."""

        ...

    def validate_table_identifier(self, table: str) -> str:
        ...

    def validate_column_identifier(self, table: str, column: str) -> str:
        ...
