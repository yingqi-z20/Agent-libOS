from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_libos.models import AgentProcess, ProcessStatus
from agent_libos.models.exceptions import ProcessRevisionConflict, ValidationError
from agent_libos.models.process_state import (
    ProcessOutcome,
    ProcessWaitState,
    validate_process_state_fields,
)
from agent_libos.ports.processes import ProcessTransitionRepositoryPort

_WAITING_STATUSES = frozenset(
    {
        ProcessStatus.WAITING_EVENT,
        ProcessStatus.WAITING_HUMAN,
        ProcessStatus.WAITING_TOOL,
        ProcessStatus.PAUSED,
    }
)


@dataclass(frozen=True, slots=True)
class ProcessStateToken:
    """Identity of one persisted process wait generation.

    The generation fences wakeups against ABA: a stale observer cannot wake a
    later wait merely because its status and payload happen to look identical.
    """

    pid: str
    state_generation: int
    wait_state: ProcessWaitState


def validate_process_state(
    status: ProcessStatus | str,
    wait_state: ProcessWaitState | None,
    outcome: ProcessOutcome | None,
) -> None:
    """Enforce the cross-field process-state invariant at the write boundary."""

    selected_status = ProcessStatus(status)
    validate_process_state_fields(selected_status.value, wait_state, outcome)


class ProcessTransitionService:
    """Single semantic write boundary for status, wait, and outcome changes."""

    def __init__(self, process_repository: ProcessTransitionRepositoryPort):
        self.store = process_repository

    def transition(
        self,
        pid: str,
        status: ProcessStatus | str,
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
        expected_state_generation: int | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
        status_message: str | None = None,
        control: bool = False,
        allowed_statuses: Iterable[ProcessStatus | str] | None = None,
        reason: str | None = None,
    ) -> AgentProcess:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValidationError(
                "process transition expected_revision must be a non-negative integer"
            )
        if expected_state_generation is not None and (
            type(expected_state_generation) is not int
            or expected_state_generation < 0
        ):
            raise ValidationError(
                "process transition expected_state_generation must be a non-negative integer"
            )
        selected_status = ProcessStatus(status)
        validate_process_state(selected_status, wait_state, outcome)
        selected_allowed = tuple(allowed_statuses or ())
        if control and not selected_allowed:
            raise ValidationError(
                "control process transition requires allowed statuses"
            )
        return self.store.apply_process_state_transition(
            pid,
            selected_status,
            expected_revision=expected_revision,
            expected_status=expected_status,
            expected_state_generation=expected_state_generation,
            wait_state=wait_state,
            outcome=outcome,
            status_message=status_message,
            control=control,
            allowed_statuses=selected_allowed if control else None,
            reason=(
                reason or "semantic process state transition"
                if control
                else reason
            ),
        )

    @staticmethod
    def wait_token(process: AgentProcess) -> ProcessStateToken:
        if process.wait_state is None:
            raise ProcessRevisionConflict(
                f"process has no active wait state: {process.pid}"
            )
        return ProcessStateToken(
            pid=process.pid,
            state_generation=process.state_generation,
            wait_state=process.wait_state,
        )

    def wake(
        self,
        token: ProcessStateToken,
        *,
        control: bool = True,
        reason: str = "process wait condition satisfied",
    ) -> AgentProcess:
        current = self.store.get_process(token.pid)
        if current is None:
            raise ProcessRevisionConflict(f"process no longer exists: {token.pid}")
        if current.state_generation != token.state_generation:
            raise ProcessRevisionConflict(
                f"stale process wait token for {token.pid}: "
                f"expected state generation {token.state_generation}, "
                f"found {current.state_generation}"
            )
        if current.wait_state != token.wait_state:
            raise ProcessRevisionConflict(
                f"stale process wait token for {token.pid}: wait state changed"
            )
        if current.status not in _WAITING_STATUSES:
            raise ProcessRevisionConflict(
                f"process is no longer waiting: {token.pid} ({current.status.value})"
            )
        return self.transition(
            token.pid,
            ProcessStatus.RUNNABLE,
            expected_revision=current.revision,
            expected_status=current.status,
            expected_state_generation=token.state_generation,
            control=control,
            allowed_statuses={current.status} if control else None,
            reason=reason,
        )


__all__ = [
    "ProcessStateToken",
    "ProcessTransitionService",
    "validate_process_state",
]
