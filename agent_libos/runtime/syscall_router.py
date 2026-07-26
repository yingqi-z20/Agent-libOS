from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, TYPE_CHECKING

from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.audit_manager import AuditManager

if TYPE_CHECKING:
    from agent_libos.runtime.syscalls import LibOSSyscallSession

SyscallHandler = Callable[["LibOSSyscallSession", dict[str, Any]], Any]


@dataclass(frozen=True)
class RegisteredSyscall:
    name: str
    handler: SyscallHandler
    registered_by: str


class SyscallRouter:
    """Registry for module-provided libOS syscalls.

    Core syscalls still live in LibOSSyscallSession. This router is the
    extension point for trusted startup modules and deliberately rejects names
    reserved by the built-in syscall surface.
    """

    def __init__(self, audit: AuditManager, *, reserved_names: set[str] | None = None) -> None:
        self.audit = audit
        self.reserved_names = set(reserved_names or set())
        self._handlers: dict[str, RegisteredSyscall] = {}
        self._lock = RLock()

    def register(self, name: str, handler: SyscallHandler, *, registered_by: str) -> RegisteredSyscall:
        normalized = name.strip()
        if not normalized:
            raise ValidationError("syscall name must be non-empty")
        if not callable(handler):
            raise ValidationError(f"syscall handler is not callable: {normalized}")
        registered = RegisteredSyscall(name=normalized, handler=handler, registered_by=registered_by)
        with self._lock:
            if normalized in self.reserved_names:
                raise ValidationError(f"cannot register module syscall over built-in syscall: {normalized}")
            if normalized in self._handlers:
                raise ValidationError(f"syscall already registered: {normalized}")
            self._handlers[normalized] = registered
            try:
                self.audit.record(
                    actor=registered_by,
                    action="syscall.register",
                    target=f"syscall:{normalized}",
                    decision={"name": normalized},
                )
            except BaseException:
                # Restore only the exact tentative publication owned by this
                # call; this stays correct if reads later become lock-free.
                if self._handlers.get(normalized) is registered:
                    self._handlers.pop(normalized, None)
                raise
        return registered

    def unregister(self, name: str, *, registered_by: str | None = None) -> bool:
        normalized = name.strip()
        with self._lock:
            registered = self._handlers.get(normalized)
            if registered is None:
                return False
            if registered_by is not None and registered.registered_by != registered_by:
                return False
            self._handlers.pop(normalized, None)
            try:
                self.audit.record(
                    actor=registered_by or "runtime",
                    action="syscall.unregister",
                    target=f"syscall:{normalized}",
                    decision={"name": normalized},
                )
            except BaseException:
                if normalized not in self._handlers:
                    self._handlers[normalized] = registered
                raise
        return True

    def get(self, name: str) -> RegisteredSyscall | None:
        with self._lock:
            return self._handlers.get(name.strip())

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            handlers = tuple(self._handlers.values())
        return [
            {"name": item.name, "registered_by": item.registered_by}
            for item in sorted(handlers, key=lambda value: value.name)
        ]
