from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_libos.models.exceptions import ProviderHostError
from agent_libos.utils.ids import new_id

UndoRegistration = Callable[[], None]


@dataclass(frozen=True)
class RegistrationEntry:
    """One reversible mutation made while installing a Runtime Module."""

    kind: str
    target: str
    undo: UndoRegistration


class RegistrationRollbackError(ProviderHostError):
    """Raised after a journal attempted every inverse operation and some failed."""

    def __init__(self, module_id: str, failures: list[tuple[RegistrationEntry, BaseException]]) -> None:
        self.module_id = module_id
        self.failures = tuple(failures)
        super().__init__(
            code="module_rollback_failed",
            error_type=type(self).__name__,
            correlation_id=new_id("corr"),
        )


class RegistrationJournal:
    """Append-only inverse-operation log for one Runtime Module.

    Registrations are undone in strict reverse publication order. A failing
    inverse does not prevent the remaining entries from being attempted.
    """

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self._entries: list[RegistrationEntry] = []
        self._recorded_targets: dict[str, list[str]] = {}
        self._rollback_started = False
        self._rolled_back = False

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def rolled_back(self) -> bool:
        return self._rolled_back

    def recorded_targets(self, kind: str) -> tuple[str, ...]:
        """Return append-only publication identities, including rolled-back entries."""

        return tuple(self._recorded_targets.get(kind, ()))

    def record(self, *, kind: str, target: str, undo: UndoRegistration) -> None:
        if self._rolled_back:
            raise RuntimeError(
                f"cannot append to rolled-back Runtime Module journal: {self.module_id}"
            )
        if self._rollback_started:
            raise RuntimeError(
                "cannot append after Runtime Module rollback has started: "
                f"{self.module_id}"
            )
        self._entries.append(RegistrationEntry(kind=kind, target=target, undo=undo))
        self._recorded_targets.setdefault(kind, []).append(target)

    def rollback(self) -> None:
        if self._rolled_back:
            return
        self._rollback_started = True
        failures: list[tuple[RegistrationEntry, BaseException]] = []
        failed_entries: list[RegistrationEntry] = []
        while self._entries:
            entry = self._entries.pop()
            try:
                entry.undo()
            except BaseException as exc:
                failures.append((entry, exc))
                failed_entries.append(entry)
        if failures:
            # Keep the original reverse-publication retry order while removing
            # entries whose inverse already completed successfully.
            self._entries.extend(reversed(failed_entries))
            raise RegistrationRollbackError(self.module_id, failures)
        self._rolled_back = True
