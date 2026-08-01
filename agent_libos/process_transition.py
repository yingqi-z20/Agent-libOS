from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_libos.models import AgentProcess, ProcessSignal, ProcessStatus
from agent_libos.models.exceptions import (
    ProcessError,
    ProcessRevisionConflict,
    ValidationError,
)
from agent_libos.models.process_state import (
    ChildProcessWait,
    HumanProcessWait,
    MessageProcessWait,
    ProcessOutcome,
    ProcessWaitState,
    StaleExecutionProcessWait,
    ToolProcessWait,
    validate_process_state_fields,
)
from agent_libos.ports.processes import ProcessTransitionRepositoryPort

_CONDITION_WAITING_STATUSES = frozenset(
    {
        ProcessStatus.WAITING_EVENT,
        ProcessStatus.WAITING_HUMAN,
        ProcessStatus.WAITING_TOOL,
    }
)

_CONDITION_WAIT_TYPES = (
    ChildProcessWait,
    MessageProcessWait,
    HumanProcessWait,
    ToolProcessWait,
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

    @staticmethod
    def require_signal_preserves_condition_wait(
        process: AgentProcess,
        signal: ProcessSignal | str,
    ) -> None:
        """Reject generic pause/resume signals that would erase an owned wait.

        Child/message and syscall-cleanup wakes use an exact
        ``ProcessStateToken``; Human and ObjectTask paths use their own exact
        generation/CAS owner checks. A generic control signal must not bypass
        those fences by replacing or clearing the wait state.
        """

        selected_signal = ProcessSignal(signal)
        if selected_signal not in {ProcessSignal.PAUSE, ProcessSignal.RESUME}:
            return
        if process.status not in {
            ProcessStatus.WAITING_EVENT,
            ProcessStatus.WAITING_HUMAN,
            ProcessStatus.WAITING_TOOL,
        }:
            return
        raise ProcessError(
            f"cannot {selected_signal.value} waiting process: "
            f"{process.pid} status={process.status.value}"
        )

    @staticmethod
    def require_wait_registration_preserves_owner(
        process: AgentProcess,
        wait_state: ProcessWaitState,
    ) -> None:
        """Reject a condition owner that would replace another owner's wait.

        A condition wait may be registered from an unblocked process or updated
        by the same wait domain.  Another condition domain, a paused process, or
        a Host-resume gate must first be resolved by its existing owner.
        """

        if not isinstance(wait_state, _CONDITION_WAIT_TYPES):
            return
        current_wait = process.wait_state
        if current_wait is None:
            if process.status == ProcessStatus.SUSPENDED:
                raise ProcessError(
                    "cannot register condition wait for suspended process: "
                    f"{process.pid} requested={wait_state.kind}"
                )
            return
        if type(current_wait) is type(wait_state):
            if isinstance(current_wait, HumanProcessWait) and isinstance(
                wait_state,
                HumanProcessWait,
            ):
                current_ids = set(current_wait.request_ids)
                requested_ids = set(wait_state.request_ids)
                if current_ids <= requested_ids or requested_ids <= current_ids:
                    return
                raise ProcessError(
                    "cannot replace active human process wait with a mixed "
                    f"request set: {process.pid}"
                )
            if current_wait == wait_state:
                return
            raise ProcessError(
                "cannot retarget an active condition wait: "
                f"{process.pid} owner={current_wait.kind}"
            )
        raise ProcessError(
            "cannot replace process wait owned by another condition: "
            f"{process.pid} current={current_wait.kind} requested={wait_state.kind}"
        )

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
        if isinstance(wait_state, StaleExecutionProcessWait):
            raise ValidationError(
                "stale-execution recovery receipts are reserved for Store recovery"
            )
        if isinstance(wait_state, _CONDITION_WAIT_TYPES):
            current = self.store.get_process(pid)
            if current is not None:
                if current.revision != expected_revision:
                    raise ProcessRevisionConflict(
                        f"stale process wait registration for {pid}: "
                        f"expected revision {expected_revision}, found {current.revision}"
                    )
                if (
                    expected_state_generation is not None
                    and current.state_generation != expected_state_generation
                ):
                    raise ProcessRevisionConflict(
                        f"stale process wait registration for {pid}: expected state "
                        f"generation {expected_state_generation}, found "
                        f"{current.state_generation}"
                    )
                self.require_wait_registration_preserves_owner(current, wait_state)
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
        if not isinstance(process.wait_state, _CONDITION_WAIT_TYPES):
            raise ProcessRevisionConflict(
                f"process has no condition-owned wait state: {process.pid}"
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
        if not isinstance(token.wait_state, _CONDITION_WAIT_TYPES):
            raise ProcessRevisionConflict(
                f"process wait token is not condition-owned: {token.pid}"
            )
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
        if (
            current.status not in _CONDITION_WAITING_STATUSES
            or not isinstance(current.wait_state, _CONDITION_WAIT_TYPES)
        ):
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
