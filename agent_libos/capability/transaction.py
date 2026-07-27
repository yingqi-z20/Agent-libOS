from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType
from typing import Any, Protocol

from agent_libos.capability.admission import (
    CapabilityAdmissionPort,
    admission_scope,
    revalidate_admission,
)
from agent_libos.models import CapabilityDecision
from agent_libos.models.exceptions import CapabilityDenied


class _AuthorityTransactionStore(Protocol):
    def transaction(self, *, include_object_payloads: bool = False) -> AbstractContextManager[Any]: ...


class AuthorityTransaction:
    """Atomically revalidate, reserve, mutate, and settle authority.

    The caller performs its durable mutation inside this context.  Every
    preflight decision is re-evaluated after the store transaction begins;
    finite-use authority is reserved before the mutation and committed before
    the transaction is allowed to commit.  Any exception rolls all of those
    steps back with the mutation.
    """

    def __init__(
        self,
        store: _AuthorityTransactionStore,
        decisions: Iterable[CapabilityDecision | None],
        *,
        actor: str,
        operation: str,
        reauthorize: Callable[[CapabilityDecision], CapabilityDecision],
        reserve: Callable[..., str | None],
        commit: Callable[..., bool],
        admission: CapabilityAdmissionPort | None = None,
    ) -> None:
        self._store = store
        self._original = tuple(decision for decision in decisions if decision is not None)
        self._actor = actor
        self._operation = operation
        self._reauthorize = reauthorize
        self._reserve = reserve
        self._commit = commit
        self._admission = admission
        self._stack: ExitStack | None = None
        self._decisions: tuple[CapabilityDecision, ...] = ()
        self._reservations: dict[str, str] = {}

    def __enter__(self) -> tuple[CapabilityDecision, ...]:
        if self._stack is not None:
            raise RuntimeError("authority transaction is already active")
        stack = ExitStack()
        self._stack = stack
        try:
            # Admission is entered first and therefore unwound last.  It
            # covers reauthorization, reservation, the caller's business
            # mutation, finite-use settlement, and the durable UoW commit.
            stack.enter_context(admission_scope(self._admission))
            stack.enter_context(self._store.transaction())
            revalidate_admission(self._admission)
            current = tuple(self._reauthorize(decision) for decision in self._original)
            for decision in current:
                cap_id = decision.consume_capability_id
                if cap_id is None or str(cap_id) in self._reservations:
                    continue
                reservation_id = self._reserve(
                    decision,
                    used_by=self._actor,
                    reason=f"one-time {self._operation} authority reserved",
                )
                if reservation_id is None:
                    raise CapabilityDenied(
                        f"{self._operation} authority reservation was not created"
                    )
                self._reservations[str(cap_id)] = reservation_id
            self._decisions = current
            return current
        except BaseException as exc:
            try:
                stack.__exit__(type(exc), exc, exc.__traceback__)
            finally:
                self._stack = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        stack = self._stack
        if stack is None:
            return None
        if exc_type is not None:
            try:
                return stack.__exit__(exc_type, exc, traceback)
            finally:
                self._stack = None
        try:
            # The caller may have waited on the UoW lock or a recovery fence
            # may have invalidated its lease while business code ran.  A stale
            # transaction must roll back before finite-use settlement or UoW
            # commit can make any of those writes durable.
            revalidate_admission(self._admission)
            for cap_id, reservation_id in self._reservations.items():
                committed = self._commit(
                    reservation_id,
                    committed_by=self._actor,
                    reason=f"one-time {self._operation} authority committed: {cap_id}",
                )
                if not committed:
                    raise CapabilityDenied(
                        f"{self._operation} authority reservation is no longer active"
                    )
            revalidate_admission(self._admission)
        except BaseException as commit_error:
            try:
                stack.__exit__(
                    type(commit_error),
                    commit_error,
                    commit_error.__traceback__,
                )
            finally:
                self._stack = None
            raise
        try:
            return stack.__exit__(None, None, None)
        finally:
            self._stack = None


__all__ = ["AuthorityTransaction"]
