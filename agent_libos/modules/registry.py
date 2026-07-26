from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, get_project_root
from agent_libos.models import (
    EventType,
    RuntimeModule,
    RuntimeModuleRegistration,
    RuntimeModuleStatus,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.modules.context import ModuleContext, ModuleRuntimeView, StartupHook
from agent_libos.modules.host import ModuleHookContext, ModuleHookServices
from agent_libos.modules.journal import RegistrationJournal
from agent_libos.modules.loader import ModuleLoader
from agent_libos.modules.schema import ModuleManifest, ModuleProvides, ModuleSource
from agent_libos.ports import AuditPort, EventPort
from agent_libos.storage import ExtensionRepository, RuntimeModuleRepositoryProtocol
from agent_libos.utils.ids import utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope_for_type,
)

class RuntimeModuleRegistry:
    """Transactional startup module registration and rollback coordinator."""

    def __init__(
        self,
        extensions: ExtensionRepository,
        module_publications: RuntimeModuleRepositoryProtocol,
        tools: Any,
        images: dict[str, Any],
        image_registry: Any,
        syscalls: Any,
        provider_hooks: dict[str, list[Any]],
        audit: AuditPort,
        events: EventPort,
        hook_services: ModuleHookServices,
        lifecycle_lock: Any,
        *,
        lifecycle: Any,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self._extensions = extensions
        self._module_publications = module_publications
        self._tools = tools
        self._images = images
        self._image_registry = image_registry
        self._syscalls = syscalls
        self._provider_hooks_by_kind = provider_hooks
        self._audit = audit
        self._events = events
        self._hook_services = hook_services
        self.config = config or DEFAULT_CONFIG
        self._lifecycle_lock = lifecycle_lock
        self._lifecycle = lifecycle
        self._provider_hooks: list[tuple[str, str, StartupHook]] = []
        self._startup_hooks: list[tuple[str, str, StartupHook]] = []
        self._loaded_modules: dict[str, dict[str, Any]] = {}
        self._applied_contexts: dict[str, ModuleContext] = {}
        self._applied_sources: dict[str, ModuleSource] = {}
        self._registration_journals: dict[str, RegistrationJournal] = {}
        self._module_import_cleanups: dict[str, tuple[str, Any]] = {}
        self._startup_hooks_ran = False

    def load_core_module(self) -> dict[str, Any]:
        with self._lifecycle.admit(), self._lifecycle_lock:
            return self._load_core_module_locked()

    def _load_core_module_locked(self) -> dict[str, Any]:
        from agent_libos.modules.core import register_module

        source_path_text = inspect.getsourcefile(register_module)
        if source_path_text is None:
            raise ValidationError("internal core module source file is unavailable")
        source_path = Path(source_path_text).resolve()
        if not source_path.is_file():
            raise ValidationError(
                f"internal core module source file is unavailable: {source_path}"
            )
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()

        manifest = ModuleManifest(
            schema_version=self.config.modules.schema_version,
            module_id="agent-libos-core:v0",
            name="Agent libOS core",
            version="v0",
            entrypoint="agent_libos.modules.core:register_module",
            provides=ModuleProvides(),
            sha256=source_sha256,
            metadata={"internal": True},
        )
        source = ModuleSource(
            manifest=manifest,
            manifest_path="<internal>",
            manifest_sha256="0" * 64,
            source_path="agent_libos.modules.core",
            source_sha256=source_sha256,
            entrypoint_object="register_module",
            source_bytes=source_bytes,
        )
        return self._load_from_entrypoint(source, register_module, enforce_provides=False)

    def load_startup_modules(
        self,
        manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        *,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lifecycle.admit(), self._lifecycle_lock:
            return self._load_startup_modules_locked(
                manifests,
                trusted_modules=trusted_modules,
                trusted_sha256=trusted_sha256,
            )

    def _load_startup_modules_locked(
        self,
        manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        *,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        selected = [self._resolve_config_manifest_path(item) for item in self.config.modules.manifest_paths]
        selected.extend(self._resolve_explicit_manifest_path(item) for item in (manifests or ()))
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for manifest_path in selected:
            resolved = str(manifest_path)
            if resolved in seen:
                raise ValidationError(f"duplicate startup module manifest: {resolved}")
            seen.add(resolved)
            results.append(
                self.load_module_manifest(
                    resolved,
                    trusted_modules=trusted_modules,
                    trusted_sha256=trusted_sha256,
                )
            )
        return results

    def _resolve_config_manifest_path(self, manifest_path: str | Path) -> Path:
        path = Path(manifest_path).expanduser()
        if not path.is_absolute():
            path = get_project_root() / path
        return path.absolute()

    def _resolve_explicit_manifest_path(self, manifest_path: str | Path) -> Path:
        return Path(manifest_path).expanduser().absolute()

    def load_module_manifest(
        self,
        manifest_path: str | Path,
        *,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        with self._lifecycle.admit(), self._lifecycle_lock:
            return self._load_module_manifest_locked(
                manifest_path,
                trusted_modules=trusted_modules,
                trusted_sha256=trusted_sha256,
            )

    def _load_module_manifest_locked(
        self,
        manifest_path: str | Path,
        *,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        loader = ModuleLoader(
            self.config,
            trusted_modules=tuple(trusted_modules or ()),
            trusted_sha256=tuple(trusted_sha256 or ()),
        )
        try:
            source = loader.resolve(manifest_path)
            if not loader.is_trusted(source.manifest.module_id, source.source_sha256, source.manifest_sha256):
                raise CapabilityDenied(
                    "startup module is not trusted: "
                    f"{source.manifest.module_id}:{source.manifest_sha256}:{source.source_sha256}"
                )
            self._require_module_id_available(source.manifest.module_id, source.source_sha256)
            entrypoint = loader.import_entrypoint(source)
            import_cleanup = loader.import_cleanup_for_entrypoint(entrypoint)
            return self._load_from_entrypoint(source, entrypoint, enforce_provides=True, import_cleanup=import_cleanup)
        except Exception as exc:
            failure = self._record_failed_manifest(manifest_path, exc)
            if self.config.modules.load_policy == "warn":
                return {
                    "status": "failed",
                    "manifest_path": str(manifest_path),
                    "error": failure["message"],
                    "error_type": failure["error_type"],
                    "correlation_id": failure["correlation_id"],
                }
            raise

    def verify_manifest(
        self,
        manifest_path: str | Path,
        *,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        with self._lifecycle_lock:
            loader = ModuleLoader(
                self.config,
                trusted_modules=tuple(trusted_modules or ()),
                trusted_sha256=tuple(trusted_sha256 or ()),
            )
            return loader.verify(manifest_path)

    def list_modules(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lifecycle_lock:
            selected_limit = self._bounded_discover_limit(limit)
            return [
                module.to_public_dict()
                for module in self._module_publications.list_runtime_modules(
                    limit=selected_limit
                )
            ]

    def _bounded_discover_limit(self, limit: int | None) -> int:
        selected = self.config.modules.discover_limit if limit is None else limit
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValidationError("Runtime Module discover limit must be an integer")
        if selected < 1:
            raise ValidationError("Runtime Module discover limit must be >= 1")
        if selected > self.config.modules.discover_limit:
            raise ValidationError(
                f"Runtime Module discover limit exceeds configured maximum {self.config.modules.discover_limit}"
            )
        return selected

    def inspect_module(self, module_id: str) -> dict[str, Any]:
        with self._lifecycle_lock:
            found = self._module_publications.get_runtime_module(module_id)
            if found is None:
                raise NotFound(f"runtime module not found: {module_id}")
            return found.to_public_dict()

    def is_loaded(self, module_id: str, source_sha256: str | None = None) -> bool:
        with self._lifecycle_lock:
            found = self._loaded_modules.get(module_id)
            if found is None:
                return False
            return source_sha256 is None or found.get("source_sha256") == source_sha256

    def loaded_module_summaries(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValidationError("Runtime Module list limit must be a positive integer")
        with self._lifecycle_lock:
            selected = sorted(self._loaded_modules.values(), key=lambda value: value["module_id"])
            if limit is not None:
                selected = selected[:limit]
            return [dict(item) for item in selected]

    def run_startup_hooks(self) -> None:
        with self._lifecycle.admit(), self._lifecycle_lock:
            self._run_startup_hooks_locked()

    def _run_startup_hooks_locked(self) -> None:
        if self._startup_hooks_ran:
            return
        failed_hook: tuple[str, str] | None = None
        try:
            with self._module_publications.transaction(
                include_object_payloads=True
            ):
                for module_id, hook_name, hook in list(self._provider_hooks):
                    failed_hook = (module_id, hook_name)
                    self._run_module_hook(module_id, hook_name, hook, kind="provider")
                for module_id, hook_name, hook in list(self._startup_hooks):
                    failed_hook = (module_id, hook_name)
                    self._run_module_hook(module_id, hook_name, hook, kind="startup")
        except BaseException as exc:
            if failed_hook is not None:
                module_id, hook_name = failed_hook
                self._rollback_external_modules_after_hook_failure(module_id, exc, hook_name=hook_name)
            raise
        self._startup_hooks_ran = True

    def shutdown(self) -> bool:
        with self._lifecycle_lock:
            self._cleanup_all_module_imports()
            return True

    def _load_from_entrypoint(
        self,
        source: ModuleSource,
        entrypoint: Any,
        *,
        enforce_provides: bool,
        import_cleanup: tuple[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._load_from_entrypoint_locked(
                source,
                entrypoint,
                enforce_provides=enforce_provides,
                import_cleanup=import_cleanup,
            )

    def _load_from_entrypoint_locked(
        self,
        source: ModuleSource,
        entrypoint: Any,
        *,
        enforce_provides: bool,
        import_cleanup: tuple[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._retry_pending_rollback(source.manifest.module_id)
        ctx = ModuleContext(ModuleRuntimeView(config=self.config), source.manifest, enforce_provides=enforce_provides)
        journal = RegistrationJournal(source.manifest.module_id)
        self._registration_journals[source.manifest.module_id] = journal
        try:
            with self._module_publications.transaction(
                include_object_payloads=True
            ):
                result = entrypoint(ctx)
                if inspect.isawaitable(result):
                    raise ValidationError(f"module entrypoint must be synchronous: {source.manifest.module_id}")
                self._preflight_context(ctx)
                self._apply_context(ctx, journal)
                now = utc_now()
                summary = ctx.registered_summary()
                self._module_publications.upsert_runtime_module(
                    RuntimeModule(
                        module_id=source.manifest.module_id,
                        name=source.manifest.name,
                        version=source.manifest.version,
                        entrypoint=source.manifest.entrypoint,
                        manifest_path=source.manifest_path,
                        manifest_sha256=source.manifest_sha256,
                        source_path=source.source_path,
                        source_sha256=source.source_sha256,
                        status=RuntimeModuleStatus.LOADED,
                        loaded_at=now,
                        registered=RuntimeModuleRegistration.from_mapping(summary),
                        error=None,
                        metadata=source.manifest.metadata,
                    )
                )
                self._loaded_modules[source.manifest.module_id] = {
                    "module_id": source.manifest.module_id,
                    "name": source.manifest.name,
                    "version": source.manifest.version,
                    "manifest_sha256": source.manifest_sha256,
                    "source_sha256": source.source_sha256,
                    "source_kind": source.source_kind,
                    "source_root": source.source_root,
                    "source_files": [
                        {
                            "path": item.path,
                            "module_path": item.module_path,
                            "size_bytes": item.size_bytes,
                            "sha256": item.sha256,
                        }
                        for item in source.source_files
                    ],
                    "entrypoint": source.manifest.entrypoint,
                    "registered": summary,
                }
                self._applied_contexts[source.manifest.module_id] = ctx
                self._applied_sources[source.manifest.module_id] = source
                if import_cleanup is not None:
                    self._module_import_cleanups[source.manifest.module_id] = import_cleanup
                self._events.emit(
                    EventType.MODULE_LOADED,
                    source="runtime",
                    target=f"module:{source.manifest.module_id}",
                    payload={"module_id": source.manifest.module_id, "registered": summary},
                )
                self._audit.record(
                    actor="runtime",
                    action="module.load",
                    target=f"module:{source.manifest.module_id}",
                    decision={
                        "module_id": source.manifest.module_id,
                        "version": source.manifest.version,
                        "manifest_sha256": source.manifest_sha256,
                        "source_sha256": source.source_sha256,
                        "source_kind": source.source_kind,
                        "registered": summary,
                    },
                )
                if self._startup_hooks_ran and not self._is_internal_module(source.manifest.module_id):
                    self._run_context_hooks(ctx)
                loaded = self._module_publications.get_runtime_module(
                    source.manifest.module_id
                )
                if loaded is None:
                    raise ValidationError(
                        "runtime module publication disappeared after load: "
                        f"{source.manifest.module_id}"
                    )
            return loaded.to_public_dict()
        except BaseException as exc:
            rollback_error: BaseException | None = None
            try:
                journal.rollback()
            except BaseException as rollback_exc:
                rollback_error = rollback_exc
            if rollback_error is None:
                self._registration_journals.pop(source.manifest.module_id, None)
            self._loaded_modules.pop(source.manifest.module_id, None)
            self._applied_contexts.pop(source.manifest.module_id, None)
            self._applied_sources.pop(source.manifest.module_id, None)
            self._cleanup_module_import(source.manifest.module_id, fallback=import_cleanup)
            if rollback_error is not None:
                raise rollback_error from exc
            raise

    def _retry_pending_rollback(self, module_id: str) -> None:
        journal = self._registration_journals.get(module_id)
        if journal is None:
            return
        if module_id in self._loaded_modules:
            raise ValidationError(f"startup module already loaded: {module_id}")
        journal.rollback()
        self._registration_journals.pop(module_id, None)

    def _require_module_id_available(self, module_id: str, source_sha256: str) -> None:
        loaded = self._loaded_modules.get(module_id)
        if loaded is None:
            return
        loaded_sha = loaded.get("source_sha256")
        if loaded_sha == source_sha256:
            raise ValidationError(f"startup module already loaded: {module_id}:{source_sha256}")
        raise ValidationError(
            "startup module id already loaded with a different source hash: "
            f"{module_id}: loaded={loaded_sha}, requested={source_sha256}"
        )

    def _preflight_context(self, ctx: ModuleContext) -> None:
        ctx.validate_registration_buffers()
        pending_tools = set(ctx.registered_tool_names)
        for tool_name in pending_tools:
            try:
                self._tools.resolve(tool_name)
            except NotFound:
                pass
            else:
                raise ValidationError(f"module tool already exists: {tool_name}")
        for image in ctx.images:
            if image.image_id in self._images:
                raise ValidationError(f"module image already exists: {image.image_id}")
            self._image_registry.validate_image(
                image,
                additional_tool_names=pending_tools,
            )
            for tool_name in image.default_tools:
                if tool_name in pending_tools:
                    continue
                try:
                    self._tools.resolve(tool_name)
                except NotFound as exc:
                    raise ValidationError(f"module image references unknown tool: {tool_name}") from exc
        for syscall in ctx.syscalls:
            if self._syscalls.get(syscall) is not None or syscall in self._syscalls.reserved_names:
                raise ValidationError(f"module syscall already exists or is built-in: {syscall}")

    def _apply_context(self, ctx: ModuleContext, journal: RegistrationJournal) -> None:
        ctx.validate_registration_buffers()
        for expected_name, tool in ctx.registered_tools:
            handle = self._tools.register_tool(tool, registered_by=ctx.actor, scope=f"module:{ctx.module_id}")
            journal.record(
                kind="tool",
                target=handle.name,
                undo=lambda handle=handle: self._unregister_tool(handle, actor=ctx.actor),
            )
            if handle.name != expected_name:
                raise ValidationError(
                    "module tool name changed after registration: "
                    f"expected {expected_name}, got {handle.name}"
                )
            self._audit.record(
                actor=ctx.actor,
                action="module.register_tool",
                target=f"tool:{handle.tool_id}",
                decision={"module_id": ctx.module_id, "tool": handle.name},
            )
        for image in ctx.images:
            result = self._image_registry.register_module_image(image, actor=ctx.actor)
            if result.disposition == "created":
                journal.record(
                    kind="image",
                    target=result.image.image_id,
                    undo=lambda image=result.image: self._unregister_image(image, actor=ctx.actor),
                )
            elif result.disposition == "rehydrated":
                journal.record(
                    kind="image_cache",
                    target=result.image.image_id,
                    undo=lambda image=result.image: self._discard_rehydrated_image(image),
                )
            else:
                raise ValidationError(
                    "module image registration returned an invalid disposition: "
                    f"{result.disposition}"
                )
            self._audit.record(
                actor=ctx.actor,
                action=(
                    "module.rehydrate_image"
                    if result.disposition == "rehydrated"
                    else "module.register_image"
                ),
                target=f"image:{result.image.image_id}",
                decision={
                    "module_id": ctx.module_id,
                    "image_id": result.image.image_id,
                    "disposition": result.disposition,
                },
            )
        for name, handler in ctx.syscalls.items():
            registered = self._syscalls.register(name, handler, registered_by=ctx.actor)
            journal.record(
                kind="syscall",
                target=registered.name,
                undo=lambda name=registered.name: self._syscalls.unregister(name, registered_by=ctx.actor),
            )
            self._audit.record(
                actor=ctx.actor,
                action="module.register_syscall",
                target=f"syscall:{name}",
                decision={"module_id": ctx.module_id, "syscall": name},
            )
        for finalizer_id, (
            prepare,
            finalize,
        ) in ctx.durable_object_release_finalizers.items():
            self._hook_services.memory.bind_durable_object_release_finalizer(
                finalizer_id,
                prepare,
                finalize,
            )
            journal.record(
                kind="durable_object_release_finalizer",
                target=finalizer_id,
                undo=lambda finalizer_id=finalizer_id: self._hook_services.memory.unbind_durable_object_release_finalizer(
                    finalizer_id
                ),
            )
            self._audit.record(
                actor=ctx.actor,
                action="module.bind_durable_object_release_finalizer",
                target=f"object_release_finalizer:{finalizer_id}",
                decision={
                    "module_id": ctx.module_id,
                    "finalizer_id": finalizer_id,
                },
            )
        for kind, hooks in ctx.provider_hooks.items():
            runtime_hooks = self._provider_hooks_by_kind.setdefault(kind, [])
            for index, hook in enumerate(hooks):
                runtime_hooks.append(hook)
                routed_hook = (ctx.module_id, f"{kind}:{index}", hook)
                self._provider_hooks.append(routed_hook)
                journal.record(
                    kind="provider_hook",
                    target=f"{kind}:{index}",
                    undo=lambda kind=kind, runtime_hooks=runtime_hooks, hook=hook, routed_hook=routed_hook: self._unregister_context_provider_hook(
                        kind,
                        runtime_hooks,
                        hook,
                        routed_hook,
                    ),
                )
            self._audit.record(
                actor=ctx.actor,
                action="module.register_provider_hook",
                target=f"provider_hook:{kind}",
                decision={"module_id": ctx.module_id, "kind": kind, "count": len(hooks)},
            )
        for name, hook in ctx.startup_hooks.items():
            routed_hook = (ctx.module_id, name, hook)
            self._startup_hooks.append(routed_hook)
            journal.record(
                kind="startup_hook",
                target=name,
                undo=lambda routed_hook=routed_hook: self._remove_identity(self._startup_hooks, routed_hook),
            )

    def _run_context_hooks(self, ctx: ModuleContext) -> None:
        ctx.validate_registration_buffers()
        for kind, hooks in ctx.provider_hooks.items():
            for index, hook in enumerate(hooks):
                self._run_module_hook(ctx.module_id, f"{kind}:{index}", hook, kind="provider")
        for name, hook in ctx.startup_hooks.items():
            self._run_module_hook(ctx.module_id, name, hook, kind="startup")

    def _run_module_hook(self, module_id: str, hook_name: str, hook: StartupHook, *, kind: str) -> None:
        journal = self._registration_journals.get(module_id)
        if journal is None:
            raise ValidationError(f"Runtime Module has no registration journal: {module_id}")
        host = ModuleHookContext(self._hook_services, module_id, journal)
        try:
            result = hook(host)
        finally:
            host.deactivate()
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise ValidationError(f"{kind} hook must be synchronous: {module_id}:{hook_name}")
        self._audit.record(
            actor=f"module:{module_id}",
            action=f"module.{kind}_hook",
            target=f"module:{module_id}:{hook_name}",
            decision={"hook": hook_name},
        )

    def _rollback_external_modules_after_hook_failure(self, failed_module_id: str, exc: BaseException, *, hook_name: str) -> None:
        module_ids = [
            module_id
            for module_id in reversed(list(self._applied_contexts))
            if not self._is_internal_module(module_id)
        ]
        for module_id in module_ids:
            source = self._applied_sources.get(module_id)
            ctx = self._applied_contexts.get(module_id)
            registered = ctx.registered_summary() if ctx is not None else {}
            self._rollback_module(module_id)
            if source is None:
                continue
            if module_id == failed_module_id:
                failure = exc
            else:
                failure = ValidationError(
                    f"startup module rolled back because {failed_module_id}:{hook_name} failed"
                )
            self._record_failed_source(source, failure, registered=registered)

    def _rollback_module(self, module_id: str) -> None:
        ctx = self._applied_contexts.get(module_id)
        journal = self._registration_journals.get(module_id)
        try:
            with self._module_publications.transaction(
                include_object_payloads=True
            ):
                if journal is not None:
                    journal.rollback()
                if ctx is not None:
                    self._audit.record(
                        actor="runtime",
                        action="module.rollback",
                        target=f"module:{ctx.module_id}",
                        decision={"registered": ctx.registered_summary()},
                    )
        except BaseException as rollback_exc:
            try:
                self._recover_durable_module_rollback(module_id, rollback_exc)
            except BaseException as recovery_exc:
                raise recovery_exc from rollback_exc
            if journal is None or not journal.rolled_back:
                raise
        self._cleanup_module_import(module_id)
        self._loaded_modules.pop(module_id, None)
        self._applied_contexts.pop(module_id, None)
        self._applied_sources.pop(module_id, None)
        self._registration_journals.pop(module_id, None)

    def _recover_durable_module_rollback(
        self,
        module_id: str,
        rollback_exc: BaseException,
    ) -> None:
        """Fail closed when the journal ran but its enclosing transaction failed."""

        actor = f"module:{module_id}"
        source = self._applied_sources.get(module_id)
        ctx = self._applied_contexts.get(module_id)
        journal = self._registration_journals.get(module_id)
        registered = ctx.registered_summary() if ctx is not None else {}
        created_image_ids = (
            set(journal.recorded_targets("image"))
            if journal is not None
            else set()
        )
        failure, observation = self._failure_evidence(
            rollback_exc,
            code="module_rollback_recovery_failed",
        )
        with self._module_publications.transaction(include_object_payloads=True):
            for row in self._extensions.list_tools():
                if row.get("registered_by") == actor:
                    self._extensions.delete_tool(
                        str(row["tool_id"]),
                        registered_by=actor,
                    )
            for image, metadata in self._extensions.list_images():
                if (
                    image.image_id in created_image_ids
                    and metadata.get("registered_by") == actor
                ):
                    self._extensions.delete_image(
                        image.image_id,
                        registered_by=actor,
                    )
            if source is not None:
                self._module_publications.upsert_runtime_module(
                    RuntimeModule(
                        module_id=source.manifest.module_id,
                        name=source.manifest.name,
                        version=source.manifest.version,
                        entrypoint=source.manifest.entrypoint,
                        manifest_path=source.manifest_path,
                        manifest_sha256=source.manifest_sha256,
                        source_path=source.source_path,
                        source_sha256=source.source_sha256,
                        status=RuntimeModuleStatus.FAILED,
                        loaded_at=None,
                        registered=RuntimeModuleRegistration.from_mapping(
                            registered
                        ),
                        error=failure["message"],
                        metadata=self._metadata_with_failure(
                            source.manifest.metadata,
                            failure,
                            observation,
                        ),
                    )
                )

        # The original failure may be the audit sink itself. Durable cleanup
        # must not depend on a second audit write, so recovery evidence is best
        # effort after the fail-closed state has committed.
        try:
            self._audit.record(
                actor="runtime",
                action="module.rollback_recovered",
                target=f"module:{module_id}",
                decision={
                    "error": failure["message"],
                    "error_type": failure["error_type"],
                    "error_code": failure["code"],
                    "correlation_id": failure["correlation_id"],
                    "error_observation": observation,
                    "registered": registered,
                },
            )
        except Exception:
            pass

    def _cleanup_all_module_imports(self) -> None:
        for module_id in reversed(list(self._module_import_cleanups)):
            self._cleanup_module_import(module_id)

    def _cleanup_module_import(self, module_id: str, *, fallback: tuple[str, Any] | None = None) -> None:
        cleanup = self._module_import_cleanups.pop(module_id, None) or fallback
        ModuleLoader.cleanup_imported_package(cleanup)

    def _unregister_tool(self, handle: Any, *, actor: str) -> None:
        try:
            self._tools.unregister_tool(handle, registered_by=actor)
        finally:
            self._tools.discard_tool_registration(handle)

    def _unregister_image(self, image: Any, *, actor: str) -> None:
        if self._images.get(image.image_id) == image:
            self._images.pop(image.image_id, None)
        stored = self._extensions.get_image(image.image_id)
        registered_by = stored[1].get("registered_by") if stored is not None else None
        if registered_by == actor:
            self._extensions.delete_image(image.image_id, registered_by=actor)

    def _discard_rehydrated_image(self, image: Any) -> None:
        """Undo cache-only module replay without touching its durable row."""

        if self._images.get(image.image_id) == image:
            self._images.pop(image.image_id, None)

    def _unregister_context_provider_hook(
        self,
        kind: str,
        runtime_hooks: list[Any],
        hook: StartupHook,
        routed_hook: tuple[str, str, StartupHook],
    ) -> None:
        self._remove_identity(self._provider_hooks, routed_hook)
        if self._provider_hooks_by_kind.get(kind) is not runtime_hooks:
            return
        self._remove_identity(runtime_hooks, hook)
        if not runtime_hooks:
            self._provider_hooks_by_kind.pop(kind, None)

    @staticmethod
    def _remove_identity(items: list[Any], value: Any) -> bool:
        for index in range(len(items) - 1, -1, -1):
            if items[index] is value:
                del items[index]
                return True
        return False

    def _is_internal_module(self, module_id: str) -> bool:
        return module_id == "agent-libos-core:v0"

    def _record_failed_manifest(
        self,
        manifest_path: str | Path,
        exc: BaseException,
    ) -> dict[str, str]:
        path = str(manifest_path)
        module_id = f"failed:{Path(path).name}"
        failure, observation = self._failure_evidence(
            exc,
            code="module_load_failed",
        )
        try:
            verification = ModuleLoader(self.config).verify(manifest_path)
            module_id = verification["module_id"]
            name = verification["name"]
            version = verification["version"]
            entrypoint = verification["entrypoint"]
            manifest_sha256 = verification["manifest_sha256"]
            source_path = verification["source_path"]
            source_sha256 = verification["source_sha256"]
            metadata = {"verification": verification}
        except Exception:
            name = module_id
            version = "unknown"
            entrypoint = ""
            manifest_sha256 = ""
            source_path = ""
            source_sha256 = ""
            metadata = {}
        metadata = self._metadata_with_failure(
            metadata,
            failure,
            observation,
        )
        if module_id in self._loaded_modules:
            self._audit.record(
                actor="runtime",
                action="module.load_failed",
                target=f"module:{module_id}",
                decision={
                    "manifest_path": path,
                    "error": failure["message"],
                    "error_type": failure["error_type"],
                    "error_code": failure["code"],
                    "correlation_id": failure["correlation_id"],
                    "error_observation": observation,
                    "preserved_loaded": True,
                },
            )
            return failure
        self._module_publications.upsert_runtime_module(
            RuntimeModule(
                module_id=module_id,
                name=name,
                version=version,
                entrypoint=entrypoint,
                manifest_path=path,
                manifest_sha256=manifest_sha256,
                source_path=source_path,
                source_sha256=source_sha256,
                status=RuntimeModuleStatus.FAILED,
                loaded_at=None,
                registered=RuntimeModuleRegistration(),
                error=failure["message"],
                metadata=metadata,
            )
        )
        self._audit.record(
            actor="runtime",
            action="module.load_failed",
            target=f"module:{module_id}",
            decision={
                "manifest_path": path,
                "error": failure["message"],
                "error_type": failure["error_type"],
                "error_code": failure["code"],
                "correlation_id": failure["correlation_id"],
                "error_observation": observation,
            },
        )
        return failure

    def _record_failed_source(
        self,
        source: ModuleSource,
        exc: BaseException,
        *,
        registered: dict[str, Any] | None = None,
    ) -> None:
        failure, observation = self._failure_evidence(
            exc,
            code="module_load_failed",
        )
        self._module_publications.upsert_runtime_module(
            RuntimeModule(
                module_id=source.manifest.module_id,
                name=source.manifest.name,
                version=source.manifest.version,
                entrypoint=source.manifest.entrypoint,
                manifest_path=source.manifest_path,
                manifest_sha256=source.manifest_sha256,
                source_path=source.source_path,
                source_sha256=source.source_sha256,
                status=RuntimeModuleStatus.FAILED,
                loaded_at=None,
                registered=RuntimeModuleRegistration.from_mapping(
                    registered or {}
                ),
                error=failure["message"],
                metadata=self._metadata_with_failure(
                    source.manifest.metadata,
                    failure,
                    observation,
                ),
            )
        )
        self._audit.record(
            actor="runtime",
            action="module.load_failed",
            target=f"module:{source.manifest.module_id}",
            decision={
                "manifest_path": source.manifest_path,
                "error": failure["message"],
                "error_type": failure["error_type"],
                "error_code": failure["code"],
                "correlation_id": failure["correlation_id"],
                "error_observation": observation,
            },
        )

    @staticmethod
    def _failure_evidence(
        exc: BaseException,
        *,
        code: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        # Module code can construct ProviderHostError itself, so this boundary
        # must not trust exception-supplied code/type/correlation fields.
        failure = public_error_envelope_for_type(
            type(exc).__name__,
            code=code,
        )
        observation = internal_exception_observation(
            exc,
            correlation_id=failure["correlation_id"],
        )
        return failure, observation

    @staticmethod
    def _metadata_with_failure(
        metadata: dict[str, Any] | Any,
        failure: dict[str, str],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **dict(metadata),
            "failure": {
                "code": failure["code"],
                "error_type": failure["error_type"],
                "correlation_id": failure["correlation_id"],
                "observation": observation,
            },
        }
