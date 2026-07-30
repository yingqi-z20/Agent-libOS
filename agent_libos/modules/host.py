from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
import threading
from typing import Any

from agent_libos.models import AgentImage
from agent_libos.models.exceptions import ValidationError
from agent_libos.modules.context import ProviderHook, SyscallHandler
from agent_libos.modules.journal import RegistrationJournal
from agent_libos.tools.base import BaseAgentTool


_MISSING = object()
_MODULE_STATE_PREFIX = "_agent_libos_"


def _remove_identity(items: list[Any], value: Any) -> bool:
    for index in range(len(items) - 1, -1, -1):
        if items[index] is value:
            del items[index]
            return True
    return False


def _restore_attribute(target: Any, name: str, installed: Any, previous: Any) -> None:
    if getattr(target, name, _MISSING) is not installed:
        return
    if previous is _MISSING:
        delattr(target, name)
    else:
        setattr(target, name, previous)


class ModuleStateRegistry:
    """Thread-safe host state owned by loaded Runtime Modules."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(name, default)

    def install(self, name: str, value: Any) -> None:
        with self._lock:
            if name in self._values:
                raise ValidationError(
                    f"Runtime Module cannot replace existing runtime_attribute: {name}"
                )
            self._values[name] = value

    def remove_if_same(self, name: str, value: Any) -> bool:
        with self._lock:
            if self._values.get(name, _MISSING) is not value:
                return False
            self._values.pop(name, None)
            return True


@dataclass(frozen=True, slots=True)
class ModuleHookServices:
    """Explicit services available to trusted Runtime Module hooks."""

    config: Any
    workspace_root: Any
    audit: Any
    events: Any
    shell: Any
    human: Any
    resources: Any
    data_flow: Any
    operations: Any
    memory: Any
    process: Any
    capability: Any
    protected_operations: Any
    store: Any
    substrate: Any
    images: dict[str, AgentImage]
    tools: Any
    image_registry: Any
    syscalls: Any
    provider_hooks: dict[str, list[Any]]
    lifecycle: Any
    state: ModuleStateRegistry
    add_handle_to_process_view: Callable[[str, Any], None]

    @classmethod
    def from_host(cls, host: Any) -> "ModuleHookServices":
        """Snapshot a Runtime facade at an assembly or test boundary."""

        return cls(
            config=host.config,
            workspace_root=host.workspace_root,
            audit=host.audit,
            events=host.events,
            shell=host.shell,
            human=host.human,
            resources=host.resources,
            data_flow=host.data_flow,
            operations=host.operations,
            memory=host.memory,
            process=host.process,
            capability=host.capability,
            protected_operations=host.protected_operations,
            store=host.store,
            substrate=host.substrate,
            images=host.images,
            tools=host.tools,
            image_registry=host.image_registry,
            syscalls=host.syscalls,
            provider_hooks=host.provider_hooks,
            lifecycle=host.lifecycle,
            state=host.module_state,
            # Runtime installs lifecycle wrappers after constructing module
            # services. Resolve this facade lazily so hooks cannot retain the
            # pre-admission bound method.
            add_handle_to_process_view=(
                lambda pid, handle: host.add_handle_to_process_view(pid, handle)
            ),
        )


class _SubstrateView:
    """Read-only provider view; module-owned additions use the journal API."""

    __slots__ = ("_host", "_substrate")

    def __init__(self, host: "ModuleHookContext", substrate: Any) -> None:
        object.__setattr__(self, "_host", host)
        object.__setattr__(self, "_substrate", substrate)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise ValidationError(
                f"Runtime Module hooks cannot access private substrate state: {name}"
            )
        return getattr(self._substrate, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        raise ValidationError(
            "Runtime Module hooks must use set_substrate_attribute for journaled state"
        )


class _MemoryView:
    """Operational Object Memory view with journaled hook registrations."""

    __slots__ = ("_host", "_memory")

    def __init__(self, host: "ModuleHookContext", memory: Any) -> None:
        object.__setattr__(self, "_host", host)
        object.__setattr__(self, "_memory", memory)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name.startswith("bind_"):
            raise ValidationError(
                f"Runtime Module hooks cannot access non-journaled memory state: {name}"
            )
        return getattr(self._memory, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        raise ValidationError(
            "Runtime Module hooks cannot replace Object Memory state"
        )

    def bind_object_release_finalizer(
        self,
        finalizer: Callable[..., None],
    ) -> None:
        self._host.bind_object_release_finalizer(finalizer)

    def bind_durable_object_release_finalizer(
        self,
        finalizer_id: str,
        prepare: Callable[..., Any],
        finalize: Callable[..., None],
    ) -> None:
        self._host.bind_durable_object_release_finalizer(
            finalizer_id,
            prepare,
            finalize,
        )


class ModuleHookContext:
    """Explicit, journaled host surface for trusted Runtime Module hooks.

    Hooks can use the named operational services required to initialize their
    provider adapter, but they cannot traverse the concrete Runtime or mutate a
    registry through a general-purpose delegation fallback. Every supported
    registration has a matching journal entry owned by this module.
    """

    __slots__ = (
        "_services",
        "_journal",
        "_module_id",
        "_actor",
        "_active",
        "_memory_view",
        "_substrate_view",
    )

    def __init__(
        self,
        services: ModuleHookServices,
        module_id: str,
        journal: RegistrationJournal,
    ) -> None:
        object.__setattr__(self, "_services", services)
        object.__setattr__(self, "_journal", journal)
        object.__setattr__(self, "_module_id", module_id)
        object.__setattr__(self, "_actor", f"module:{module_id}")
        object.__setattr__(self, "_active", True)
        object.__setattr__(
            self,
            "_memory_view",
            _MemoryView(self, services.memory),
        )
        object.__setattr__(
            self,
            "_substrate_view",
            _SubstrateView(self, services.substrate),
        )

    @property
    def module_id(self) -> str:
        return self._module_id

    @property
    def actor(self) -> str:
        return self._actor

    @property
    def config(self) -> Any:
        return self._services.config

    @property
    def workspace_root(self) -> Any:
        return self._services.workspace_root

    @property
    def audit(self) -> Any:
        return self._services.audit

    @property
    def events(self) -> Any:
        return self._services.events

    @property
    def shell(self) -> Any:
        return self._services.shell

    @property
    def human(self) -> Any:
        return self._services.human

    @property
    def resources(self) -> Any:
        return self._services.resources

    @property
    def data_flow(self) -> Any:
        return self._services.data_flow

    @property
    def operations(self) -> Any:
        return self._services.operations

    @property
    def memory(self) -> _MemoryView:
        return self._memory_view

    @property
    def process(self) -> Any:
        return self._services.process

    @property
    def capability(self) -> Any:
        return self._services.capability

    @property
    def protected_operations(self) -> Any:
        return self._services.protected_operations

    @property
    def store(self) -> Any:
        return self._services.store

    @property
    def substrate(self) -> _SubstrateView:
        return self._substrate_view

    @property
    def images(self) -> MappingProxyType[str, AgentImage]:
        return MappingProxyType(deepcopy(self._services.images))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        raise ValidationError(
            "Runtime Module hooks must use an explicit journaled registration method"
        )

    def deactivate(self) -> None:
        object.__setattr__(self, "_active", False)

    def register_tool(
        self,
        tool: BaseAgentTool,
        *,
        scope: str | None = None,
        ephemeral: bool = False,
    ) -> Any:
        self._require_active()
        handle = self._services.tools.register_tool(
            tool,
            registered_by=self.actor,
            scope=scope or f"module:{self.module_id}",
            ephemeral=ephemeral,
        )
        self._journal.record(
            kind="tool",
            target=handle.name,
            undo=lambda: self._unregister_tool(handle),
        )
        return handle

    def register_image(
        self,
        image: AgentImage | dict[str, Any],
        *,
        source: str | None = None,
    ) -> Any:
        self._require_active()
        result = self._services.image_registry.register(
            image,
            actor=self.actor,
            replace=False,
            require_capability=False,
            source=source,
        )
        registered = result.image
        self._journal.record(
            kind="image",
            target=registered.image_id,
            undo=lambda: self._unregister_image(registered),
        )
        return result

    def register_syscall(self, name: str, handler: SyscallHandler) -> Any:
        self._require_active()
        registered = self._services.syscalls.register(
            name,
            handler,
            registered_by=self.actor,
        )
        self._journal.record(
            kind="syscall",
            target=registered.name,
            undo=lambda: self._services.syscalls.unregister(
                registered.name,
                registered_by=self.actor,
            ),
        )
        return registered

    def register_provider_hook(self, kind: str, hook: ProviderHook) -> None:
        self._require_active()
        if not isinstance(kind, str) or not kind.strip() or not callable(hook):
            raise ValidationError(
                "provider hook requires a non-empty kind and callable hook"
            )
        normalized = kind.strip()
        hooks = self._services.provider_hooks.setdefault(normalized, [])
        hooks.append(hook)
        self._journal.record(
            kind="provider_hook",
            target=normalized,
            undo=lambda: self._unregister_provider_hook(normalized, hooks, hook),
        )

    def bind_shutdown_finalizer(self, finalizer: Callable[[], Any]) -> None:
        self._require_active()
        if not callable(finalizer):
            raise ValidationError("shutdown finalizer must be callable")
        self._services.lifecycle.bind_finalizer(finalizer)
        self._journal.record(
            kind="shutdown_finalizer",
            target=getattr(finalizer, "__name__", type(finalizer).__name__),
            undo=lambda: self._services.lifecycle.unbind_finalizer(finalizer),
        )

    def bind_recovery_cleanup(self, cleanup: Callable[[], Any]) -> None:
        """Bind an explicitly recovery-safe, process-local cleanup callback.

        Recovery diagnostics handoff cannot run ordinary shutdown finalizers:
        they may publish durable evidence or perform unrelated user effects.
        Trusted modules use this narrower surface only for idempotent teardown
        of transient handles and workers.  The callback must not access Object
        Memory, the runtime store, audit, or events, and must return ``False``
        when its process-local cleanup has not fully converged so the lifecycle
        can retry it without closing the store.
        """

        self._require_active()
        if not callable(cleanup):
            raise ValidationError("recovery cleanup must be callable")
        self._services.lifecycle.bind_finalizer(cleanup, recovery_safe=True)
        self._journal.record(
            kind="recovery_cleanup",
            target=getattr(cleanup, "__name__", type(cleanup).__name__),
            undo=lambda: self._services.lifecycle.unbind_finalizer(cleanup),
        )

    def require_recovery_cleanup_lease(self) -> None:
        """Require the lifecycle-scoped lease for transient provider teardown.

        Unlike registration methods, this check is intentionally callable by
        the retained module host after startup-hook deactivation.  It grants no
        authority itself: RuntimeLifecycle installs its opaque ContextVar only
        around an explicitly registered recovery-safe callback.
        """

        self._services.lifecycle.require_recovery_cleanup_lease()

    def bind_object_release_finalizer(
        self,
        finalizer: Callable[..., None],
    ) -> None:
        self._require_active()
        if not callable(finalizer):
            raise ValidationError("object release finalizer must be callable")
        self._services.memory.bind_object_release_finalizer(finalizer)
        self._journal.record(
            kind="object_release_finalizer",
            target=getattr(finalizer, "__name__", type(finalizer).__name__),
            undo=lambda: self._services.memory.unbind_object_release_finalizer(finalizer),
        )

    def bind_durable_object_release_finalizer(
        self,
        finalizer_id: str,
        prepare: Callable[..., Any],
        finalize: Callable[..., None],
    ) -> None:
        self._require_active()
        if not callable(prepare) or not callable(finalize):
            raise ValidationError(
                "durable object release finalizer callbacks must be callable"
            )
        self._services.memory.bind_durable_object_release_finalizer(
            finalizer_id,
            prepare,
            finalize,
        )
        self._journal.record(
            kind="durable_object_release_finalizer",
            target=str(finalizer_id),
            undo=lambda: self._services.memory.unbind_durable_object_release_finalizer(
                finalizer_id
            ),
        )

    def add_handle_to_process_view(self, pid: str, handle: Any) -> None:
        """Publish a handle through the Runtime-owned process CAS boundary."""

        self._services.add_handle_to_process_view(str(pid), handle)

    def get_runtime_attribute(self, name: str, default: Any = None) -> Any:
        selected = self._module_state_name(name)
        return self._services.state.get(selected, default)

    def set_runtime_attribute(self, name: str, value: Any) -> None:
        self._require_active()
        selected = self._module_state_name(name)
        self._services.state.install(selected, value)
        self._journal.record(
            kind="runtime_attribute",
            target=selected,
            undo=lambda: self._services.state.remove_if_same(selected, value),
        )

    def set_substrate_attribute(self, name: str, value: Any) -> None:
        self._require_active()
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("_")
            or hasattr(self._services.substrate, name)
        ):
            raise ValidationError(
                "module substrate attribute must be a new public name"
            )
        self._install_new_attribute(
            self._services.substrate,
            name,
            value,
            kind="substrate_attribute",
        )

    def _install_new_attribute(
        self,
        target: Any,
        name: str,
        value: Any,
        *,
        kind: str,
    ) -> None:
        previous = getattr(target, name, _MISSING)
        if previous is not _MISSING:
            raise ValidationError(
                f"Runtime Module cannot replace existing {kind}: {name}"
            )
        setattr(target, name, value)
        self._journal.record(
            kind=kind,
            target=name,
            undo=lambda: _restore_attribute(target, name, value, previous),
        )

    def _unregister_tool(self, handle: Any) -> None:
        try:
            self._services.tools.unregister_tool(
                handle,
                registered_by=self.actor,
            )
        finally:
            self._services.tools.discard_tool_registration(handle)

    def _unregister_image(self, image: AgentImage) -> None:
        current = self._services.images.get(image.image_id)
        if current == image:
            self._services.image_registry.stage_image_removal(
                image.image_id,
                expected=image,
            )
        stored = self._services.store.get_image(image.image_id)
        stored_actor = (
            stored[1].get("registered_by")
            if stored is not None
            else None
        )
        if stored_actor == self.actor:
            self._services.store.delete_image(
                image.image_id,
                registered_by=self.actor,
            )

    def _unregister_provider_hook(
        self,
        kind: str,
        hooks: list[Any],
        hook: ProviderHook,
    ) -> None:
        if self._services.provider_hooks.get(kind) is not hooks:
            return
        _remove_identity(hooks, hook)
        if not hooks:
            self._services.provider_hooks.pop(kind, None)

    def _require_active(self) -> None:
        if not self._active:
            raise ValidationError(
                f"Runtime Module hook registration is closed: {self.module_id}"
            )

    @staticmethod
    def _module_state_name(name: str) -> str:
        if (
            not isinstance(name, str)
            or not name.startswith(_MODULE_STATE_PREFIX)
            or name == _MODULE_STATE_PREFIX
        ):
            raise ValidationError(
                f"module runtime state must use the {_MODULE_STATE_PREFIX!r} prefix"
            )
        return name


__all__ = ["ModuleHookContext", "ModuleHookServices", "ModuleStateRegistry"]
