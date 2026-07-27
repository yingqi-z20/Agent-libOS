from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator

from agent_libos.models import ProcessExecutionToken, ProcessStatus


@dataclass(frozen=True, slots=True)
class ProcessControlMutation:
    """Explicit Host control scope for a worker-originated cross-PID write."""

    target_pid: str
    allowed_statuses: frozenset[str]
    reason: str


@dataclass(slots=True)
class ProcessExecutionTakeoverIntent:
    """Exact one-transaction intent for emergency control of an exec lease."""

    target_pid: str
    source_revision: int
    source_state_generation: int
    source_execution_token: ProcessExecutionToken
    intended_status: str
    reason: str
    nonce: str
    reason_text: str | None = None
    reason_action: str | None = None
    wait_kind: str | None = None
    outcome_code: str | None = None
    stage: int = 0
    reason_capability_id: str | None = None
    reason_oid: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalProcessMutation:
    """Exact scope for append-only bookkeeping on one terminal process row."""

    target_pid: str
    expected_revision: int
    expected_generation: int
    allowed_statuses: frozenset[str]
    execution_token: ProcessExecutionToken | None
    reason: str


@dataclass(frozen=True, slots=True)
class PostExecCompletionMutation:
    """One exact ToolResult capability append after a committed exec epoch."""

    target_pid: str
    publication_id: str
    operation_id: str
    expected_revision: int
    expected_generation: int
    execution_token: ProcessExecutionToken
    reason: str


_CURRENT_PROCESS_EXECUTION: ContextVar[ProcessExecutionToken | None] = ContextVar(
    "agent_libos_current_process_execution",
    default=None,
)
_CURRENT_TERMINAL_PROCESS_MUTATION: ContextVar[TerminalProcessMutation | None] = ContextVar(
    "agent_libos_current_terminal_process_mutation",
    default=None,
)
_CURRENT_PROCESS_CONTROL_MUTATION: ContextVar[ProcessControlMutation | None] = (
    ContextVar(
        "agent_libos_current_process_control_mutation",
        default=None,
    )
)
_CURRENT_PROCESS_EXECUTION_TAKEOVER: ContextVar[
    ProcessExecutionTakeoverIntent | None
] = ContextVar(
    "agent_libos_current_process_execution_takeover",
    default=None,
)
_CURRENT_POST_EXEC_COMPLETION_MUTATION: ContextVar[
    PostExecCompletionMutation | None
] = ContextVar(
    "agent_libos_current_post_exec_completion_mutation",
    default=None,
)


def current_process_execution_token() -> ProcessExecutionToken | None:
    return _CURRENT_PROCESS_EXECUTION.get()


def current_terminal_process_mutation() -> TerminalProcessMutation | None:
    return _CURRENT_TERMINAL_PROCESS_MUTATION.get()


def current_process_control_mutation() -> ProcessControlMutation | None:
    return _CURRENT_PROCESS_CONTROL_MUTATION.get()


def current_process_execution_takeover_intent() -> (
    ProcessExecutionTakeoverIntent | None
):
    return _CURRENT_PROCESS_EXECUTION_TAKEOVER.get()


def current_post_exec_completion_mutation() -> PostExecCompletionMutation | None:
    return _CURRENT_POST_EXEC_COMPLETION_MUTATION.get()


@contextmanager
def bind_process_execution(
    token: ProcessExecutionToken,
) -> Iterator[ProcessExecutionToken]:
    reset = _CURRENT_PROCESS_EXECUTION.set(token)
    try:
        yield token
    finally:
        _CURRENT_PROCESS_EXECUTION.reset(reset)


@contextmanager
def trusted_terminal_process_mutation(
    target_pid: str,
    *,
    expected_revision: int,
    expected_generation: int,
    allowed_statuses: Iterable[ProcessStatus | str],
    execution_token: ProcessExecutionToken | None,
    reason: str,
) -> Iterator[TerminalProcessMutation]:
    """Authorize one exact terminal-row bookkeeping CAS.

    Terminal rows remain immutable by default. The narrow exception names the
    row version and generation observed by trusted runtime code, the terminal
    source status, the ambient worker token (if any), and an auditable reason.
    The SQL store revalidates every field before permitting the write.
    """

    selected_pid = str(target_pid).strip()
    selected_reason = str(reason).strip()
    statuses = frozenset(ProcessStatus(status).value for status in allowed_statuses)
    terminal_statuses = {
        ProcessStatus.EXITED.value,
        ProcessStatus.FAILED.value,
        ProcessStatus.KILLED.value,
    }
    if not selected_pid:
        raise ValueError("terminal process mutation target pid must be non-empty")
    if not statuses or not statuses <= terminal_statuses:
        raise ValueError("terminal process mutation statuses must be terminal")
    if (
        type(expected_revision) is not int
        or type(expected_generation) is not int
        or expected_revision < 0
        or expected_generation < 0
    ):
        raise ValueError("terminal process mutation concurrency values must be non-negative")
    if not selected_reason:
        raise ValueError("terminal process mutation reason must be non-empty")
    ambient_token = current_process_execution_token()
    if execution_token != ambient_token:
        raise ValueError("terminal process mutation token must match the ambient worker token")
    mutation = TerminalProcessMutation(
        target_pid=selected_pid,
        expected_revision=expected_revision,
        expected_generation=expected_generation,
        allowed_statuses=statuses,
        execution_token=execution_token,
        reason=selected_reason,
    )
    reset = _CURRENT_TERMINAL_PROCESS_MUTATION.set(mutation)
    try:
        yield mutation
    finally:
        _CURRENT_TERMINAL_PROCESS_MUTATION.reset(reset)


@contextmanager
def trusted_process_control_mutation(
    target_pid: str,
    *,
    allowed_statuses: Iterable[ProcessStatus | str],
    reason: str,
) -> Iterator[ProcessControlMutation]:
    """Declare one narrow Host control scope for a cross-process mutation.

    Ordinary worker writes remain bound to their exact execution token.  A
    runtime manager that intentionally changes another process must name the
    target, the source statuses it accepts, and an auditable reason before the
    store will treat the write as Host control.
    """

    selected_pid = str(target_pid).strip()
    selected_reason = str(reason).strip()
    statuses = frozenset(ProcessStatus(status).value for status in allowed_statuses)
    if not selected_pid:
        raise ValueError("process control target pid must be non-empty")
    if not statuses:
        raise ValueError("process control allowed statuses must be non-empty")
    if not selected_reason:
        raise ValueError("process control reason must be non-empty")
    mutation = ProcessControlMutation(
        target_pid=selected_pid,
        allowed_statuses=statuses,
        reason=selected_reason,
    )
    reset = _CURRENT_PROCESS_CONTROL_MUTATION.set(mutation)
    try:
        yield mutation
    finally:
        _CURRENT_PROCESS_CONTROL_MUTATION.reset(reset)


@contextmanager
def trusted_process_execution_takeover(
    target_pid: str,
    *,
    source_revision: int,
    source_state_generation: int,
    source_execution_token: ProcessExecutionToken,
    intended_status: ProcessStatus | str,
    reason: str,
    nonce: str,
    reason_text: str | None = None,
    reason_action: str | None = None,
    wait_kind: str | None = None,
    outcome_code: str | None = None,
) -> Iterator[ProcessExecutionTakeoverIntent]:
    """Authorize only one exact RUNNING exec takeover transaction."""

    pid = str(target_pid).strip()
    selected_reason = str(reason).strip()
    selected_nonce = str(nonce).strip()
    status = ProcessStatus(intended_status)
    if (
        not pid
        or source_execution_token.pid != pid
        or isinstance(source_revision, bool)
        or isinstance(source_state_generation, bool)
        or not isinstance(source_revision, int)
        or not isinstance(source_state_generation, int)
        or source_revision < 0
        or source_state_generation < 0
        or status not in {ProcessStatus.PAUSED, ProcessStatus.KILLED}
        or not selected_reason
        or not selected_nonce
    ):
        raise ValueError("invalid process execution takeover intent")
    if (reason_text is None) != (reason_action is None):
        raise ValueError("takeover reason text and action must be provided together")
    if current_process_execution_takeover_intent() is not None:
        raise ValueError("nested process execution takeover intents are forbidden")
    intent = ProcessExecutionTakeoverIntent(
        target_pid=pid,
        source_revision=int(source_revision),
        source_state_generation=int(source_state_generation),
        source_execution_token=source_execution_token,
        intended_status=status.value,
        reason=selected_reason,
        nonce=selected_nonce,
        reason_text=reason_text,
        reason_action=reason_action,
        wait_kind=wait_kind,
        outcome_code=outcome_code,
    )
    reset = _CURRENT_PROCESS_EXECUTION_TAKEOVER.set(intent)
    try:
        yield intent
    except BaseException:
        raise
    else:
        if intent.stage != 3:
            raise RuntimeError("process execution takeover did not commit its final state")
    finally:
        _CURRENT_PROCESS_EXECUTION_TAKEOVER.reset(reset)


@contextmanager
def trusted_post_exec_completion_mutation(
    target_pid: str,
    *,
    publication_id: str,
    operation_id: str,
    expected_revision: int,
    expected_generation: int,
    execution_token: ProcessExecutionToken,
    reason: str,
) -> Iterator[PostExecCompletionMutation]:
    """Authorize only the ToolResult handle append of one committed exec.

    A successful exec deliberately fences its caller's worker token before the
    tool wrapper has persisted the successful ToolResult.  This scope does not
    revive that token: the SQL store also requires the exact committed
    publication receipt, ``RUNNABLE`` generation ``token + 1``, cleared lease,
    exact row revision, and a single ToolResult object-handle capability append.
    """

    selected_pid = str(target_pid).strip()
    selected_publication = str(publication_id).strip()
    selected_operation = str(operation_id).strip()
    selected_reason = str(reason).strip()
    if not all(
        (selected_pid, selected_publication, selected_operation, selected_reason)
    ):
        raise ValueError("post-exec completion fields must be non-empty")
    if (
        type(expected_revision) is not int
        or type(expected_generation) is not int
        or expected_revision < 0
        or expected_generation < 0
    ):
        raise ValueError("post-exec completion concurrency values must be non-negative")
    ambient_token = current_process_execution_token()
    if execution_token != ambient_token or execution_token.pid != selected_pid:
        raise ValueError("post-exec completion token must match the target worker")
    if expected_generation != execution_token.generation + 1:
        raise ValueError("post-exec completion generation must be token generation + 1")
    mutation = PostExecCompletionMutation(
        target_pid=selected_pid,
        publication_id=selected_publication,
        operation_id=selected_operation,
        expected_revision=expected_revision,
        expected_generation=expected_generation,
        execution_token=execution_token,
        reason=selected_reason,
    )
    reset = _CURRENT_POST_EXEC_COMPLETION_MUTATION.set(mutation)
    try:
        yield mutation
    finally:
        _CURRENT_POST_EXEC_COMPLETION_MUTATION.reset(reset)


__all__ = [
    "PostExecCompletionMutation",
    "ProcessControlMutation",
    "ProcessExecutionTakeoverIntent",
    "TerminalProcessMutation",
    "bind_process_execution",
    "current_process_control_mutation",
    "current_process_execution_takeover_intent",
    "current_process_execution_token",
    "current_post_exec_completion_mutation",
    "current_terminal_process_mutation",
    "trusted_process_control_mutation",
    "trusted_process_execution_takeover",
    "trusted_post_exec_completion_mutation",
    "trusted_terminal_process_mutation",
]
