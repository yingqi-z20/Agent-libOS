from __future__ import annotations


class LibOSError(Exception):
    """Base exception for Agent libOS runtime errors."""


class NotFound(LibOSError):
    pass


class CapabilityDenied(LibOSError):
    pass


class HumanApprovalRequired(LibOSError):
    def __init__(self, request_id: str, message: str):
        super().__init__(message)
        self.request_id = request_id


class HumanResponseRequired(HumanApprovalRequired):
    pass


class ProcessWaitRequired(LibOSError):
    def __init__(self, child_pid: str, message: str, resume_action: dict | None = None):
        super().__init__(message)
        self.child_pid = child_pid
        self.resume_action = dict(resume_action) if resume_action is not None else None


class ProcessMessageWaitRequired(LibOSError):
    def __init__(self, recipient_pid: str, filters: dict, message: str):
        super().__init__(message)
        self.recipient_pid = recipient_pid
        self.filters = dict(filters)


class PolicyDenied(LibOSError):
    pass


class ProcessError(LibOSError):
    pass


class ProcessRevisionConflict(ProcessError):
    """A process mutation lost its compare-and-swap race."""

    pass


class TaskRunRevisionConflict(ProcessError):
    """A durable TaskRun mutation lost its revision or epoch fence."""

    pass


class TaskRunCommandConflict(TaskRunRevisionConflict):
    """A TaskRun idempotency key was reused for a different request."""

    pass


class ProcessTerminalCleanupRequired(ProcessError):
    """A terminal outcome committed, but its durable cleanup is incomplete."""

    def __init__(
        self,
        *,
        pid: str,
        phase: str,
        attempt: int,
    ) -> None:
        self.pid = str(pid)
        self.phase = str(phase)
        self.attempt = int(attempt)
        super().__init__(
            "process terminal cleanup remains incomplete: "
            f"{self.pid} phase={self.phase} attempt={self.attempt}"
        )


class RuntimePublicationPending(ProcessError):
    """A linked runtime publication still owns an operation's final outcome."""

    def __init__(
        self,
        *,
        publication_id: str,
        operation_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.publication_id = str(publication_id)
        self.operation_id = str(operation_id)
        self.state = str(state)
        self.phase = str(phase)
        super().__init__(
            "runtime publication outcome remains pending: "
            f"{self.publication_id} ({self.state}/{self.phase})"
        )


class RuntimeRecoveryRequired(ProcessError):
    """A failed exec compensation requires reopen before further mutation."""

    def __init__(
        self,
        *,
        publication_id: str,
        operation_id: str,
        pid: str,
        state: str,
        phase: str,
        _issuer_token: object | None = None,
    ) -> None:
        self.publication_id = str(publication_id)
        self.operation_id = str(operation_id)
        self.pid = str(pid)
        self.state = str(state)
        self.phase = str(phase)
        self._issuer_token = _issuer_token
        super().__init__(
            "runtime recovery is required before further mutation: "
            f"{self.publication_id} ({self.state}/{self.phase})"
        )


class ResourceLimitExceeded(ProcessError):
    pass


class ValidationError(LibOSError):
    pass


class TaskRunCompletionContractError(ValidationError):
    """A Durable TaskRun completion contract failed local integrity checks.

    The exception deliberately carries only the Run identity and an already
    sanitized blocker projection.  Callers may validate the contract while a
    Store critical section is held, then persist ``needs_attention`` only after
    that critical section has unwound.
    """

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        blocker: dict,
    ) -> None:
        self.run_id = str(run_id)
        self.blocker = dict(blocker)
        super().__init__(message)


class SkillPackageChanged(ValidationError):
    """A hash-pinned Skill activation no longer matches visible content."""

    pass


class GitError(LibOSError):
    """Stable domain error raised by the first-class Git boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        operation: str | None = None,
        retryable: bool = False,
        details: dict | None = None,
    ) -> None:
        self.code = str(code)
        self.operation = str(operation) if operation is not None else None
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "operation": self.operation,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


class DurableObjectFinalizerUnavailable(ValidationError):
    """A durable cleanup intent has no restart-stable handler."""

    pass


class ProviderHostError(LibOSError, RuntimeError):
    """Stable public replacement for an exception raised by a Host provider."""

    def __init__(self, *, code: str, error_type: str, correlation_id: str):
        self.code = str(code)
        self.error_type = str(error_type)
        self.correlation_id = str(correlation_id)
        super().__init__(
            f"{self.code}: {self.error_type} (correlation_id={self.correlation_id})"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "error_type": self.error_type,
            "correlation_id": self.correlation_id,
        }


class UnsupportedStoreVersion(ValidationError):
    """The runtime store belongs to an unsupported on-disk schema generation."""

    pass


class SandboxError(LibOSError):
    pass
