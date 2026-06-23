from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.primitives import ClockPrimitive, FilesystemAdapter, JsonRpcPrimitive, ShellAdapter
from agent_libos.human.manager import HumanObjectManager
from agent_libos.llm.client import LLMClient
from agent_libos.llm.executor import LLMProcessExecutor
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    EventType,
    ObjectHandle,
    ObjectOwnerKind,
    ProcessStatus,
    ResourceUsage,
    ToolCandidate,
    ToolCandidateStatus,
    ToolHandle,
    ToolSpec,
    ViewMode,
    WorkflowRunResult,
)
from agent_libos.models.exceptions import HumanApprovalRequired, NotFound, ProcessMessageWaitRequired, ProcessWaitRequired, ValidationError
from agent_libos.modules import RuntimeModuleRegistry
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.checkpoint_manager import CheckpointManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.image_registry import ImageRegistryPrimitive
from agent_libos.runtime.message_manager import ProcessMessageManager
from agent_libos.runtime.object_tasks import ObjectTaskManager
from agent_libos.runtime.process_manager import ProcessManager
from agent_libos.runtime.resource_manager import ResourceManager
from agent_libos.runtime.scheduler import SimpleScheduler
from agent_libos.runtime.syscall_router import SyscallRouter
from agent_libos.runtime.syscalls import BUILTIN_SYSCALL_NAMES
from agent_libos.skills.manager import SkillManager
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import HttpJsonRpcProvider, LocalResourceProviderSubstrate, ResourceProviderSubstrate
from agent_libos.tools.broker import ToolBroker
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, loads

class Runtime:
    """Composition root for Agent libOS.

    Runtime wires storage, capability checks, primitives, providers, process
    scheduling, ToolBroker, Skills, checkpoints, audit, and LLM execution. Host
    effects should enter through primitives and provider interfaces, not through
    model-facing tools.
    """

    def __init__(
        self,
        store: SQLiteStore,
        llm_client: LLMClient | None = None,
        substrate: ResourceProviderSubstrate | None = None,
        config: AgentLibOSConfig | None = None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.substrate = substrate or LocalResourceProviderSubstrate(
            Path.cwd().resolve(),
            namespace=self.config.runtime.workspace_namespace,
        )
        self.workspace_root = Path(getattr(self.substrate, "workspace_root", self.substrate.workspace_display))
        self.store = store
        self.store.config = self.config
        self.llms = LLMProfileRegistry(self, config=self.config)
        self.audit = AuditManager(store)
        self.events = EventBus(store)
        self.resources = ResourceManager(store, self.audit, self.events)
        self.syscalls = SyscallRouter(self.audit, reserved_names=BUILTIN_SYSCALL_NAMES)
        self.provider_hooks: dict[str, list[Any]] = {}
        self._shutdown_finalizers: list[Any] = []
        self.capability = CapabilityManager(store, self.audit, self.events, config=self.config)
        self.memory = ObjectMemoryManager(
            store,
            self.capability,
            self.audit,
            self.events,
            config=self.config,
            resources=self.resources,
        )
        self.human = HumanObjectManager(
            store,
            self.capability,
            self.audit,
            self.events,
            provider=self.substrate.human,
            config=self.config,
        )
        self.messages = ProcessMessageManager(store, self.audit, self.events, config=self.config)
        self.human.bind_messages(self.messages)
        self.clock = ClockPrimitive(
            self.audit,
            self.events,
            max_sleep_seconds=self.config.tools.max_sleep_seconds,
            provider=self.substrate.clock,
        )
        self.filesystem = FilesystemAdapter(
            self.capability,
            self.audit,
            self.events,
            human=self.human,
            provider=self.substrate.filesystem,
            resources=self.resources,
            config=self.config,
        )
        self.shell = ShellAdapter(
            self.capability,
            self.audit,
            self.events,
            human=self.human,
            provider=self.substrate.shell,
            config=self.config,
            resources=self.resources,
        )
        self.jsonrpc = JsonRpcPrimitive(
            store,
            self.capability,
            self.audit,
            self.events,
            human=self.human,
            provider=getattr(self.substrate, "jsonrpc", HttpJsonRpcProvider()),
            config=self.config,
            resources=self.resources,
        )
        self.images: dict[str, AgentImage] = {}
        self.tools = ToolBroker(
            store,
            self.memory,
            self.capability,
            self.human,
            self.audit,
            self.events,
            workspace_root=self.workspace_root,
            config=self.config,
            resources=self.resources,
        )
        self.tools.runtime = self
        self.process = ProcessManager(
            store,
            self.memory,
            self.capability,
            self.audit,
            self.events,
            config=self.config,
            resources=self.resources,
            llm_profile_resolver=self._resolve_launch_llm_profile_id,
        )
        self.resources.bind_process_kill_finalizer(self.process.finalize_killed_processes)
        self.process.add_after_spawn_hook(self._configure_process_tools_and_capabilities)
        self.object_tasks = ObjectTaskManager(self, config=self.config)
        self.memory.bind_object_pin_checker(self.object_tasks.has_active_for_owner)
        self.memory.bind_object_change_notifier(self.object_tasks.notify_owner_changed)
        self.scheduler = SimpleScheduler(
            store,
            self.audit,
            poll_interval_s=self.config.scheduler.poll_interval_s,
            max_workers=self.config.scheduler.max_workers,
            drain_window_s=self.config.scheduler.drain_window_s,
            shutdown_join_timeout_s=self.config.scheduler.shutdown_join_timeout_s,
            resources=self.resources,
            skip_pid=self.object_tasks.is_runner_pid,
            cancel_process=self.process.cancel,
        )
        self.checkpoint = CheckpointManager(store, self.audit, self.events, self.capability, config=self.config)
        self.checkpoint.bind_runtime(self)
        self.skills = SkillManager(
            store,
            self.capability,
            self.audit,
            self.events,
            human=self.human,
            config=self.config,
        )
        self.skills.bind_runtime(self)
        self.image_registry = ImageRegistryPrimitive(
            self.images,
            self.capability,
            self.audit,
            self.events,
            self.tools.resolve,
            store=self.store,
            config=self.config,
        )
        self.image_registry.bind_runtime(self)
        self.llm = LLMProcessExecutor(self, llm_client, config=self.config)
        self._current_human_auto_approve: bool | None = None
        self._current_human_auto_policy: str | None = None
        self._current_human_auto_answer: str | None = None
        self.modules = RuntimeModuleRegistry(self, config=self.config)
        self.modules.load_core_module()
        self.modules.load_startup_modules(
            startup_module_manifests,
            trusted_modules=trusted_modules,
            trusted_sha256=trusted_module_sha256,
        )
        self.image_registry.load_persisted_images()
        self._rehydrate_registered_jit_tools()
        self.modules.run_startup_hooks()
        self._closed = False
        self._shutdown_reason: str | None = None

    @classmethod
    def open(
        cls,
        target: str | Path | None = None,
        substrate: ResourceProviderSubstrate | None = None,
        config: AgentLibOSConfig | None = None,
        module_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> "Runtime":
        selected_config = config or DEFAULT_CONFIG
        selected_target = selected_config.runtime.local_store_target if target is None else target
        store_target = ":memory:" if str(selected_target) == selected_config.runtime.local_store_target else str(selected_target)
        store = SQLiteStore(store_target, config=selected_config)
        try:
            return cls(
                store,
                substrate=substrate,
                config=selected_config,
                startup_module_manifests=module_manifests,
                trusted_modules=trusted_modules,
                trusted_module_sha256=trusted_module_sha256,
            )
        except Exception:
            store.close()
            raise

    def _rehydrate_registered_jit_tools(self) -> None:
        """Restore process-local JIT implementations after a normal runtime reopen.

        SQLite persists process tool tables and tool rows, while the executable
        TypeScript sources live in ToolBroker's in-memory sandbox table. A
        registered JIT is valid only when a process table still references its
        tool id; orphan candidates remain inert and stale references are pruned
        fail-closed so the model is not shown a tool that cannot run.
        """

        ephemeral_tool_rows = {
            str(row["tool_id"]): row
            for row in self.store.list_tools()
            if bool(row.get("ephemeral"))
        }
        if not ephemeral_tool_rows:
            return

        candidate_rows = self.store.select_table_rows(
            "tool_candidates",
            "status = ?",
            [ToolCandidateStatus.REGISTERED.value],
            order_by="updated_at, candidate_id",
        )
        candidates_by_tool_id: dict[str, dict[str, Any]] = {}
        fallback_by_owner_name: dict[tuple[str, str], dict[str, Any]] = {}
        for row in candidate_rows:
            spec = loads(row.get("spec_json"), {})
            name = str(spec.get("name") or "")
            registered_tool_id = str(row.get("registered_tool_id") or "")
            if registered_tool_id:
                candidates_by_tool_id[registered_tool_id] = row
            elif name:
                # Best-effort recovery for pre-registered_tool_id rows. New
                # rows use the exact id binding above to avoid name reuse bugs.
                fallback_by_owner_name[(str(row["pid"]), name)] = row

        restored: list[dict[str, str]] = []
        pruned: list[dict[str, str]] = []
        for process in self.store.list_processes():
            changed = False
            for name, raw_tool_id in list(process.tool_table.items()):
                tool_id = str(raw_tool_id)
                row = ephemeral_tool_rows.get(tool_id)
                if row is None:
                    continue
                if tool_id in self.tools._tools:
                    continue
                candidate = candidates_by_tool_id.get(tool_id) or fallback_by_owner_name.get((process.pid, str(name)))
                source = str(candidate.get("source_code") or "") if candidate is not None else ""
                if candidate is None or not source or str(row.get("name") or "") != str(name):
                    process.tool_table.pop(name, None)
                    changed = True
                    pruned.append({"pid": process.pid, "tool_id": tool_id, "name": str(name)})
                    continue
                self.tools._jit_sources[tool_id] = source
                self.tools._handles[tool_id] = ToolHandle(
                    tool_id=tool_id,
                    name=str(name),
                    capability_id=None,
                    scope=str(row.get("scope") or "ephemeral_process"),
                )
                restored.append({"pid": process.pid, "tool_id": tool_id, "name": str(name)})
            if changed:
                process.updated_at = utc_now()
                self.store.update_process(process)

        if restored or pruned:
            self.audit.record(
                actor="runtime",
                action="runtime.jit.rehydrate",
                target="tool:jits",
                decision={"restored": restored, "pruned_stale": pruned},
            )

    def shutdown(self, *, actor: str = "runtime", reason: str = "runtime.shutdown") -> dict[str, Any]:
        """Shut down this host Runtime instance.

        Shutdown is a host lifecycle operation. It stops accepting further use
        of this composition root and releases owned handles, but it does not
        change AgentProcess lifecycle state. A process must still exit through
        the process primitive/tool path, which keeps process authority and audit
        semantics separate from host resource cleanup.
        """
        if self._closed:
            return {"ok": True, "already_shutdown": True, "reason": self._shutdown_reason}
        self._shutdown_reason = reason
        errors: list[dict[str, str]] = []
        self.audit.record(
            actor=actor,
            action="runtime.shutdown",
            target="runtime",
            decision={"reason": reason},
        )
        self.events.emit(
            EventType.RUNTIME_SHUTDOWN,
            source=actor,
            target="runtime",
            payload={"reason": reason},
        )
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and not scheduler.shutdown():
            return {
                "ok": False,
                "already_shutdown": False,
                "reason": reason,
                "scheduler_stopped": False,
            }
        for name, component in [
            ("object_tasks", getattr(self, "object_tasks", None)),
        ]:
            try:
                if not self._shutdown_component(component):
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"{name}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": name, "error": str(exc), "error_type": type(exc).__name__})
        for index, finalizer in enumerate(list(self._shutdown_finalizers)):
            try:
                if finalizer() is False:
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"shutdown_finalizer_{index}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": f"shutdown_finalizer:{index}", "error": str(exc), "error_type": type(exc).__name__})
        for name, component in [
            ("llms", getattr(self, "llms", None)),
            ("substrate", self.substrate),
        ]:
            try:
                if not self._shutdown_component(component):
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"{name}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": name, "error": str(exc), "error_type": type(exc).__name__})
        self._closed = True
        self.store.close()
        if errors:
            raise RuntimeError(f"runtime shutdown completed with component errors: {errors}")
        return {"ok": True, "already_shutdown": False, "reason": reason}

    async def ashutdown(self, *, actor: str = "runtime", reason: str = "runtime.shutdown") -> dict[str, Any]:
        """Async shutdown variant for event-loop hosts."""
        if self._closed:
            return {"ok": True, "already_shutdown": True, "reason": self._shutdown_reason}
        self._shutdown_reason = reason
        errors: list[dict[str, str]] = []
        self.audit.record(
            actor=actor,
            action="runtime.shutdown",
            target="runtime",
            decision={"reason": reason},
        )
        self.events.emit(
            EventType.RUNTIME_SHUTDOWN,
            source=actor,
            target="runtime",
            payload={"reason": reason},
        )
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and not scheduler.shutdown():
            return {
                "ok": False,
                "already_shutdown": False,
                "reason": reason,
                "scheduler_stopped": False,
            }
        for name, component in [
            ("object_tasks", getattr(self, "object_tasks", None)),
        ]:
            try:
                if not await self._ashutdown_component(component):
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"{name}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": name, "error": str(exc), "error_type": type(exc).__name__})
        for index, finalizer in enumerate(list(self._shutdown_finalizers)):
            try:
                result = finalizer()
                if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                    result = await result
                if result is False:
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"shutdown_finalizer_{index}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": f"shutdown_finalizer:{index}", "error": str(exc), "error_type": type(exc).__name__})
        for name, component in [
            ("llms", getattr(self, "llms", None)),
            ("substrate", self.substrate),
        ]:
            try:
                if not await self._ashutdown_component(component):
                    return {
                        "ok": False,
                        "already_shutdown": False,
                        "reason": reason,
                        f"{name}_stopped": False,
                    }
            except Exception as exc:
                errors.append({"component": name, "error": str(exc), "error_type": type(exc).__name__})
        self._closed = True
        self.store.close()
        if errors:
            raise RuntimeError(f"runtime shutdown completed with component errors: {errors}")
        return {"ok": True, "already_shutdown": False, "reason": reason}

    def close(self) -> None:
        """Compatibility alias for shutdown(); prefer Runtime.shutdown()."""
        self.shutdown(actor="runtime.close", reason="runtime.close")

    def _shutdown_component(self, component: Any) -> bool:
        if component is None:
            return True
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            return shutdown() is not False
        close = getattr(component, "close", None)
        if callable(close):
            close()
        return True

    def bind_shutdown_finalizer(self, finalizer: Any) -> None:
        self._shutdown_finalizers.append(finalizer)

    async def _ashutdown_component(self, component: Any) -> bool:
        if component is None:
            return True
        ashutdown = getattr(component, "ashutdown", None)
        if callable(ashutdown):
            result = ashutdown()
            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                result = await result
            return result is not False
        aclose = getattr(component, "aclose", None)
        if callable(aclose):
            result = aclose()
            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                await result
            return True
        shutdown = getattr(component, "shutdown", None)
        if callable(shutdown):
            return shutdown() is not False
        close = getattr(component, "close", None)
        if callable(close):
            close()
        return True

    def run_process_once(self, pid: str) -> dict[str, Any]:
        return self.llm.run_once(pid)

    async def arun_process_once(self, pid: str) -> dict[str, Any]:
        return await self.llm.arun_once(pid)

    def run_next_process_once(self) -> Any:
        return self.scheduler.run_once(self.arun_process_once)

    async def arun_next_process_once(self) -> Any:
        return await self.scheduler.arun_once(self.arun_process_once)

    def run_until_idle(
        self,
        max_quanta: int | None = None,
        *,
        process_human_queue: bool = True,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> list[Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._run_until_idle_sync(
                max_quanta=max_quanta,
                process_human_queue=process_human_queue,
                human=human,
                human_auto_approve=human_auto_approve,
                human_auto_policy=human_auto_policy,
                human_auto_answer=human_auto_answer,
            )
        raise RuntimeError("Cannot call run_until_idle() inside a running event loop. Use await arun_until_idle(...).")

    async def arun_until_idle(
        self,
        max_quanta: int | None = None,
        *,
        process_human_queue: bool = True,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> list[Any]:
        return await asyncio.to_thread(
            self._run_until_idle_sync,
            max_quanta=max_quanta,
            process_human_queue=process_human_queue,
            human=human,
            human_auto_approve=human_auto_approve,
            human_auto_policy=human_auto_policy,
            human_auto_answer=human_auto_answer,
        )

    def _run_until_idle_sync(
        self,
        max_quanta: int | None = None,
        *,
        process_human_queue: bool = True,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> list[Any]:
        results: list[Any] = []
        remaining = self.config.runtime.run_until_idle_max_quanta if max_quanta is None else max_quanta
        selected_human = human or self.config.runtime.default_human
        previous_human_context = (
            self._current_human_auto_approve,
            self._current_human_auto_policy,
            self._current_human_auto_answer,
        )
        self._current_human_auto_approve = human_auto_approve
        self._current_human_auto_policy = human_auto_policy
        self._current_human_auto_answer = human_auto_answer
        try:
            while remaining is None or remaining > 0:
                # Run all currently runnable processes first. Human queue work below
                # may wake a process, so this loop intentionally alternates between
                # process execution and terminal queue draining.
                batch = self.scheduler.run_until_idle(self.arun_process_once, max_quanta=remaining)
                results.extend(batch)
                if remaining is not None:
                    remaining -= len(batch)
                if not process_human_queue:
                    break
                processed = self.human.drain_terminal_queue(
                    human=selected_human,
                    auto_approve=human_auto_approve,
                    auto_policy=human_auto_policy,
                    auto_answer=human_auto_answer,
                )
                if not processed:
                    break
                self.audit.record(
                    actor="runtime",
                    action="runtime.human_queue_drained",
                    target=f"human:{selected_human}",
                    decision={"request_ids": [request.request_id for request in processed]},
                )
        finally:
            (
                self._current_human_auto_approve,
                self._current_human_auto_policy,
                self._current_human_auto_answer,
            ) = previous_human_context
        return results

    def run_process_until_idle(self, pid: str, *, max_quanta: int | None = None) -> list[Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_process_until_idle(pid, max_quanta=max_quanta))
        raise RuntimeError(
            "Cannot call run_process_until_idle() inside a running event loop. "
            "Use await arun_process_until_idle(...)."
        )

    async def arun_process_until_idle(self, pid: str, *, max_quanta: int | None = None) -> list[Any]:
        selected_quanta = self.config.runtime.run_until_idle_max_quanta if max_quanta is None else max_quanta
        return await self.scheduler.arun_pid_until_idle(
            pid,
            self.arun_process_once,
            max_quanta=selected_quanta,
        )

    def run_workflow(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        image: str | None = None,
        goal: dict[str, Any] | str | None = None,
        working_directory: str | None = None,
    ) -> WorkflowRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun_workflow(
                    tool,
                    args,
                    image=image,
                    goal=goal,
                    working_directory=working_directory,
                )
            )
        raise RuntimeError("Cannot call run_workflow() inside a running event loop. Use await arun_workflow(...).")

    async def arun_workflow(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        image: str | None = None,
        goal: dict[str, Any] | str | None = None,
        working_directory: str | None = None,
    ) -> WorkflowRunResult:
        tool_name = str(tool).strip()
        if not tool_name:
            raise ValidationError("workflow tool name is required")
        if args is None:
            tool_args: dict[str, Any] = {}
        elif isinstance(args, dict):
            tool_args = dict(args)
        else:
            raise ValidationError("workflow args must be a JSON object")

        selected_image = image or self.config.runtime.default_image_id
        pid = self.process.spawn(
            image=selected_image,
            goal=goal if goal is not None else f"workflow:{tool_name}",
            working_directory=working_directory,
        )
        initial = self.process.get(pid)
        initial_image = initial.image_id
        initial_goal_oid = initial.goal_oid
        try:
            result = await self.tools.acall(pid, tool_name, tool_args)
        except HumanApprovalRequired as exc:
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=str(exc),
                waiting_human=True,
                request_id=exc.request_id,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except ProcessWaitRequired as exc:
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=str(exc),
                waiting_process=True,
                child_pid=exc.child_pid,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except ProcessMessageWaitRequired as exc:
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=str(exc),
                waiting_message=True,
                filters=exc.filters,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except Exception as exc:
            error = str(exc)
            if not self._workflow_tool_controlled_lifecycle(pid, initial_image=initial_image, initial_goal_oid=initial_goal_oid):
                self.process.exit(pid, failed=True, message=error or f"workflow failed: {tool_name}")
            workflow_result = self._workflow_failure_result(pid, selected_image, tool_name, error=error)
            self._record_workflow_run(workflow_result)
            return workflow_result

        if result.result_handle is not None:
            self._add_handle_to_process_view(pid, result.result_handle)
        if not self._workflow_tool_controlled_lifecycle(pid, initial_image=initial_image, initial_goal_oid=initial_goal_oid):
            if result.ok:
                self.process.exit(
                    pid,
                    result=result.result_handle,
                    message=None if result.result_handle is not None else f"workflow completed: {tool_name}",
                )
            else:
                self.process.exit(pid, failed=True, message=result.error or f"workflow failed: {tool_name}")
        workflow_result = self._workflow_result_from_tool_call(pid, selected_image, tool_name, result)
        self._record_workflow_run(workflow_result)
        return workflow_result

    def _workflow_wait_result(
        self,
        pid: str,
        image: str,
        tool: str,
        *,
        error: str,
        waiting_human: bool = False,
        request_id: str | None = None,
        waiting_process: bool = False,
        child_pid: str | None = None,
        waiting_message: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> WorkflowRunResult:
        process = self.process.get(pid)
        return WorkflowRunResult(
            pid=pid,
            image=image,
            tool=tool,
            ok=False,
            status=process.status.value,
            tool_id=self._workflow_resolve_tool_id(pid, tool),
            error=error,
            waiting_human=waiting_human,
            request_id=request_id,
            waiting_process=waiting_process,
            child_pid=child_pid,
            waiting_message=waiting_message,
            filters=dict(filters or {}) if filters is not None else None,
        )

    def _workflow_result_from_tool_call(
        self,
        pid: str,
        image: str,
        tool: str,
        result: Any,
    ) -> WorkflowRunResult:
        process = self.process.get(pid)
        return WorkflowRunResult(
            pid=pid,
            image=image,
            tool=tool,
            ok=bool(result.ok),
            status=process.status.value,
            call_id=result.call_id,
            tool_id=result.tool_id,
            result_oid=result.result_handle.oid if result.result_handle is not None else None,
            payload=result.payload,
            error=result.error,
        )

    def _workflow_failure_result(self, pid: str, image: str, tool: str, *, error: str) -> WorkflowRunResult:
        process = self.process.get(pid)
        return WorkflowRunResult(
            pid=pid,
            image=image,
            tool=tool,
            ok=False,
            status=process.status.value,
            tool_id=self._workflow_resolve_tool_id(pid, tool),
            error=error,
        )

    def _workflow_resolve_tool_id(self, pid: str, tool: str) -> str | None:
        try:
            return self.tools.resolve(tool, pid=pid).tool_id
        except Exception:
            return None

    def _workflow_tool_controlled_lifecycle(
        self,
        pid: str,
        *,
        initial_image: str,
        initial_goal_oid: str | None,
    ) -> bool:
        process = self.process.get(pid)
        if process.status in self.process.TERMINAL_STATUSES:
            return True
        if process.image_id != initial_image or process.goal_oid != initial_goal_oid:
            return True
        return any(
            record.actor == pid and record.action == "process.exec" and record.target == f"process:{pid}"
            for record in self.audit.trace(actor=pid, target=f"process:{pid}")
        )

    def _add_handle_to_process_view(self, pid: str, handle: ObjectHandle) -> None:
        process = self.process.get(pid)
        if process.memory_view is None:
            process.memory_view = self.memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
        elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
            process.memory_view.roots.append(handle)
        process.updated_at = utc_now()
        self.store.update_process(process)

    def _record_workflow_run(self, result: WorkflowRunResult) -> None:
        decision = {
            "tool": result.tool,
            "image": result.image,
            "ok": result.ok,
            "status": result.status,
            "call_id": result.call_id,
            "tool_id": result.tool_id,
            "result_oid": result.result_oid,
            "waiting_human": result.waiting_human,
            "waiting_process": result.waiting_process,
            "waiting_message": result.waiting_message,
        }
        if result.request_id is not None:
            decision["request_id"] = result.request_id
        if result.child_pid is not None:
            decision["child_pid"] = result.child_pid
        if result.filters is not None:
            decision["filters"] = result.filters
        self.audit.record(
            actor="workflow",
            action="workflow.run",
            target=f"process:{result.pid}",
            output_refs=[result.result_oid] if result.result_oid is not None else [],
            decision=decision,
        )

    def register_image(self, image: AgentImage | dict[str, Any], *, actor: str = "runtime", replace: bool = False) -> None:
        self.image_registry.register(image, actor=actor, replace=replace)

    def get_image(self, image_id: str) -> AgentImage:
        return self.images[image_id]

    def register_skill_from_path(
        self,
        path: str | os.PathLike[str],
        *,
        actor: str = "runtime",
        replace: bool = False,
        source_type: str = "runtime",
    ) -> dict[str, Any]:
        return self.skills.register_skill_from_path(
            path,
            actor=actor,
            replace=replace,
            require_capability=False,
            source_type=source_type,
        )

    def discover_skills(self, text: str | None = None) -> list[dict[str, Any]]:
        return self.skills.discover_skills(text, require_capability=False)

    def inspect_skill(self, skill_id: str) -> dict[str, Any]:
        return self.skills.inspect_skill(skill_id, require_capability=False)

    def activate_skill(self, pid: str, skill_id: str) -> dict[str, Any]:
        return self.skills.activate_skill(pid, skill_id, actor=pid, require_capability=False)

    def unload_skill(self, pid: str, skill_id: str) -> dict[str, Any]:
        return self.skills.unload_skill(pid, skill_id, actor=pid, require_capability=False)

    def trust_skill_source(self, *, source_type: str, source: str, package_sha256: str, actor: str = "runtime") -> dict[str, Any]:
        return self.skills.trust_skill_source(
            actor=actor,
            source_type=source_type,
            source=source,
            package_sha256=package_sha256,
            require_capability=False,
        )

    def exec_process(
        self,
        pid: str,
        image: str,
        *,
        args: dict[str, Any] | None = None,
        goal: dict[str, Any] | str | None = None,
        preserve_memory: bool = True,
        preserve_capabilities: bool = False,
        llm_profile_id: str | None = None,
    ) -> Any:
        selected_image = self._require_image(image)
        self._preflight_process_image_boot(selected_image)
        previous_state = self._snapshot_process_exec_state(pid)
        self.process.exec(
            pid,
            image,
            args=args,
            goal=goal,
            preserve_memory=preserve_memory,
            preserve_capabilities=preserve_capabilities,
            llm_profile_id=llm_profile_id,
        )
        try:
            # Exec swaps the process image and tool table, but deliberately does
            # not apply image required_capabilities. Exec may preserve existing
            # capabilities or shrink them; it never grants new external
            # authority. Package workspaces are private materialized state, so
            # they are instantiated here just as they are during spawn.
            self._configure_process_tools_for_image(pid, image, assigned_by=f"process.exec:{image}")
            boot_kind = selected_image.boot.get("kind", "fresh")
            if boot_kind == "checkpoint_commit":
                self._instantiate_checkpoint_commit_image(pid, selected_image)
            elif boot_kind == "image_package":
                self._instantiate_image_package(pid, selected_image)
            self._configure_process_skills_for_image(pid, image, assigned_by=f"process.exec:{image}")
        except Exception as exc:
            self._restore_process_exec_state(previous_state)
            self.audit.record(
                actor="runtime",
                action="image.boot.failed",
                target=f"process:{pid}",
                decision={"image": image, "phase": "process.exec", "error": str(exc), "rolled_back": True},
            )
            raise
        return self.process.get(pid)

    def spawn_child_process(
        self,
        parent: str,
        goal: dict[str, Any] | str,
        *,
        image: str | None = None,
        inherit_capabilities: list[dict[str, Any]] | None = None,
        resource_budget: Any | None = None,
        working_directory: str | None = None,
        llm_profile_id: str | None = None,
    ) -> str:
        parent_process = self.process.get(parent)
        selected_image = image or parent_process.image_id
        self._require_image(selected_image)
        selected_cwd = (
            self.resolve_process_working_directory(parent, working_directory)
            if working_directory is not None
            else parent_process.working_directory
        )
        return self.process.spawn_child(
            parent=parent,
            goal=goal,
            image=selected_image,
            inherit_capabilities=inherit_capabilities,
            resource_budget=resource_budget,
            working_directory=selected_cwd,
            llm_profile_id=llm_profile_id,
        )

    def set_process_working_directory(self, pid: str, path: str) -> Any:
        relative = self.resolve_process_working_directory(pid, path)
        return self.process.set_working_directory(pid, relative)

    def _resolve_launch_llm_profile_id(self, image_id: str, explicit_profile_id: str | None) -> str:
        if explicit_profile_id is not None:
            selected = str(explicit_profile_id).strip()
            if not selected:
                raise ValidationError("LLM profile id must be a non-empty string")
            return selected
        image = self.images.get(image_id)
        if image is not None and image.llm_profile_id:
            return image.llm_profile_id
        return self.config.llm.default_profile_id

    def resolve_process_working_directory(self, pid: str, path: str) -> str:
        current_cwd = self.process.working_directory(pid)
        target, relative = self.filesystem.resolve_path(path, cwd=current_cwd)
        state = self.filesystem.provider.state(target)
        if not state.exists:
            raise NotFound(f"working directory does not exist: {relative}")
        if state.kind != "directory":
            raise NotFound(f"working directory is not a directory: {relative}")
        return relative or "."

    def _configure_process_tools_and_capabilities(self, pid: str, image_id: str) -> None:
        try:
            image = self._require_image(image_id)
        except Exception as exc:
            self._fail_process_image_boot(pid, image_id, exc, phase="process.spawn")
            raise
        try:
            boot_kind = image.boot.get("kind", "fresh")
            is_checkpoint_commit = boot_kind == "checkpoint_commit"
            is_image_package = boot_kind == "image_package"
            # Tool visibility is fixed from the AgentImage at process creation time.
            # External-resource authority is still enforced later by the primitives.
            self._configure_process_tools_for_image(pid, image.image_id, assigned_by=f"image:{image_id}")
            if is_checkpoint_commit:
                self._instantiate_checkpoint_commit_image(pid, image)
            elif is_image_package:
                self._instantiate_image_package(pid, image)
            process = self.store.get_process(pid)
        except Exception as exc:
            self._fail_process_image_boot(pid, image_id, exc, phase="process.spawn")
            raise
        try:
            self._configure_process_skills_for_image(pid, image.image_id, assigned_by=f"image:{image_id}")
        except Exception as exc:
            self._fail_process_image_boot(pid, image_id, exc, phase="image.default_skills")
            raise
        if process is not None:
            self.checkpoint.grant_process_defaults(pid, issued_by=f"image:{image_id}")
        if process is not None and process.parent_pid is not None:
            self.audit.record(
                actor="runtime",
                action="image.default_capability_skipped_for_child",
                target=f"process:{pid}",
                decision={"image": image_id, "parent_pid": process.parent_pid},
            )
            return
        if is_checkpoint_commit or is_image_package:
            self.audit.record(
                actor="runtime",
                action="image.required_capabilities_declared_only",
                target=f"process:{pid}",
                decision={
                    "image": image_id,
                    "required_capabilities": len(image.required_capabilities),
                    "reason": f"{boot_kind} images never grant external authority automatically",
                },
            )
            return
        for spec in image.required_capabilities:
            try:
                self.capability.grant(
                    subject=pid,
                    resource=spec["resource"],
                    rights=spec.get("rights", []),
                    issued_by=f"image:{image_id}",
                    constraints=spec.get("constraints"),
                    expires_at=spec.get("expires_at"),
                    delegable=spec.get("delegable", False),
                    revocable=spec.get("revocable", True),
                )
            except Exception as exc:
                self.audit.record(
                    actor="runtime",
                    action="image.default_capability_grant_failed",
                    target=f"process:{pid}",
                    decision={"capability": spec, "error": str(exc)},
                )

    def _require_image(self, image_id: str) -> AgentImage:
        image = self.images.get(image_id)
        if image is None:
            raise NotFound(f"agent image not found: {image_id}")
        return image

    def _preflight_process_image_boot(self, image: AgentImage) -> None:
        boot_kind = image.boot.get("kind", "fresh")
        if boot_kind == "checkpoint_commit":
            artifact = self._load_image_artifact(image, expected_kind="checkpoint_commit")
            self.checkpoint._require_snapshot_modules({"modules": artifact.get("modules", [])})
        elif boot_kind == "image_package":
            self._load_image_artifact(image, expected_kind="image_package")

    def _snapshot_process_exec_state(self, pid: str) -> dict[str, Any]:
        process_rows = self.store.select_table_rows("processes", "pid = ?", (pid,))
        if not process_rows:
            raise NotFound(f"process not found: {pid}")
        object_rows = self.store.select_table_rows(
            "objects",
            "owner_kind = ? AND owner_id = ? AND lifecycle_state = ?",
            (ObjectOwnerKind.PROCESS.value, pid, "live"),
            order_by="oid",
        )
        object_oids = [str(row["oid"]) for row in object_rows]
        namespace_rows = self.store.select_table_rows(
            "object_namespaces",
            "created_by = ? OR namespace = ?",
            (pid, self.memory.process_namespace(pid)),
            order_by="namespace",
        )
        tool_ids = set(loads(process_rows[0].get("tool_table_json"), {}).values())
        return {
            "pid": pid,
            "object_oids": object_oids,
            "namespace_names": [str(row["namespace"]) for row in namespace_rows],
            "tables": {
                "processes": process_rows,
                "object_namespaces": namespace_rows,
                "objects": object_rows,
                "object_links": self._exec_object_link_rows(object_oids),
                "capabilities": self.store.select_table_rows("capabilities", "subject = ?", (pid,), order_by="cap_id"),
                "llm_pending_actions": self.store.select_table_rows("llm_pending_actions", "pid = ?", (pid,)),
                "tool_candidates": self.store.select_table_rows("tool_candidates", "pid = ?", (pid,), order_by="candidate_id"),
                "process_resource_reservations": self.store.select_table_rows(
                    "process_resource_reservations",
                    "parent_pid = ? OR child_pid = ?",
                    (pid, pid),
                    order_by="parent_pid, child_pid",
                ),
            },
            "object_payloads": {
                oid: deepcopy(self.store._object_payloads[oid])
                for oid in object_oids
                if oid in self.store._object_payloads
            },
            "tool_ids": tool_ids,
            "tool_handles": {
                tool_id: deepcopy(getattr(self.tools, "_handles", {}).get(tool_id))
                for tool_id in tool_ids
                if tool_id in getattr(self.tools, "_handles", {})
            },
            "jit_sources": {
                tool_id: deepcopy(getattr(self.tools, "_jit_sources", {}).get(tool_id))
                for tool_id in tool_ids
                if tool_id in getattr(self.tools, "_jit_sources", {})
            },
        }

    def _restore_process_exec_state(self, state: dict[str, Any]) -> None:
        pid = state["pid"]
        tables = state["tables"]
        current_process = self.store.get_process(pid)
        current_object_rows = self.store.select_table_rows(
            "objects",
            "owner_kind = ? AND owner_id = ? AND lifecycle_state = ?",
            (ObjectOwnerKind.PROCESS.value, pid, "live"),
            order_by="oid",
        )
        current_object_oids = [str(row["oid"]) for row in current_object_rows]
        object_oids = sorted(set(state["object_oids"]) | set(current_object_oids))
        namespace_names = sorted(
            set(state["namespace_names"])
            | {
                str(row["namespace"])
                for row in self.store.select_table_rows(
                    "object_namespaces",
                    "created_by = ? OR namespace = ?",
                    (pid, self.memory.process_namespace(pid)),
                    order_by="namespace",
                )
            }
        )
        current_tool_ids = set(current_process.tool_table.values()) if current_process is not None else set()
        stale_tool_ids = current_tool_ids - set(state["tool_ids"])
        with self.store.transaction(include_object_payloads=True) as cur:
            if object_oids:
                placeholders = ", ".join("?" for _ in object_oids)
                cur.execute(
                    f"DELETE FROM object_links WHERE src_oid IN ({placeholders}) OR dst_oid IN ({placeholders})",
                    [*object_oids, *object_oids],
                )
                cur.execute(f"DELETE FROM objects WHERE oid IN ({placeholders})", object_oids)
                cur.execute(
                    f"DELETE FROM capabilities WHERE resource IN ({placeholders})",
                    [f"object:{oid}" for oid in object_oids],
                )
                for oid in object_oids:
                    self.store.forget_object_payload(oid)
            if namespace_names:
                placeholders = ", ".join("?" for _ in namespace_names)
                cur.execute(f"DELETE FROM object_namespaces WHERE namespace IN ({placeholders})", namespace_names)
            cur.execute("DELETE FROM capabilities WHERE subject = ?", (pid,))
            cur.execute("DELETE FROM llm_pending_actions WHERE pid = ?", (pid,))
            cur.execute("DELETE FROM tool_candidates WHERE pid = ?", (pid,))
            cur.execute("DELETE FROM process_resource_reservations WHERE parent_pid = ? OR child_pid = ?", (pid, pid))
            cur.execute("DELETE FROM processes WHERE pid = ?", (pid,))
            for row in tables["object_namespaces"]:
                self.checkpoint._insert_row(cur, "object_namespaces", row)
            for row in tables["objects"]:
                item = dict(row)
                item["payload_json"] = dumps(self.store._memory_payload_marker(present=True))
                self.checkpoint._insert_row(cur, "objects", item)
                oid = str(item["oid"])
                if oid in state["object_payloads"]:
                    self.store.set_object_payload(oid, deepcopy(state["object_payloads"][oid]))
            for table in [
                "object_links",
                "capabilities",
                "llm_pending_actions",
                "tool_candidates",
                "process_resource_reservations",
                "processes",
            ]:
                for row in tables[table]:
                    self.checkpoint._insert_row(cur, table, row)
        for tool_id in stale_tool_ids:
            if not self._tool_id_used_by_other_process(tool_id, pid):
                getattr(self.tools, "_handles", {}).pop(tool_id, None)
                getattr(self.tools, "_jit_sources", {}).pop(tool_id, None)
        for tool_id, handle in state["tool_handles"].items():
            self.tools._handles[tool_id] = deepcopy(handle)
        for tool_id, source in state["jit_sources"].items():
            self.tools._jit_sources[tool_id] = deepcopy(source)

    def _exec_object_link_rows(self, object_oids: list[str]) -> list[dict[str, Any]]:
        if not object_oids:
            return []
        placeholders = ", ".join("?" for _ in object_oids)
        return self.store.select_table_rows(
            "object_links",
            f"src_oid IN ({placeholders}) OR dst_oid IN ({placeholders})",
            [*object_oids, *object_oids],
            order_by="id",
        )

    def _tool_id_used_by_other_process(self, tool_id: str, pid: str) -> bool:
        for process in self.store.list_processes():
            if process.pid == pid:
                continue
            if tool_id in process.tool_table.values():
                return True
        return False

    def _fail_process_image_boot(self, pid: str, image_id: str, exc: Exception, *, phase: str) -> None:
        process = self.store.get_process(pid)
        if process is not None:
            process.status = ProcessStatus.FAILED
            process.status_message = str(exc)
            process.updated_at = utc_now()
            self.store.update_process(process)
        self.audit.record(
            actor="runtime",
            action="image.boot.failed",
            target=f"process:{pid}",
            decision={"image": image_id, "phase": phase, "error": str(exc)},
        )

    def _configure_process_tools_for_image(self, pid: str, image_id: str, assigned_by: str) -> dict[str, str]:
        image = self._require_image(image_id)
        return self.tools.configure_process_tools(pid, sorted(image.default_tools), assigned_by=assigned_by)

    def _configure_process_skills_for_image(self, pid: str, image_id: str, assigned_by: str) -> None:
        process = self.store.get_process(pid)
        if process is None:
            return
        self._apply_loaded_skill_tool_table(pid)
        image = self._require_image(image_id)
        for skill_id in image.default_skills:
            process = self.store.get_process(pid)
            if process is not None and skill_id in process.loaded_skills:
                continue
            self.skills.activate_skill(pid, skill_id, actor=assigned_by, require_capability=False)

    def _instantiate_checkpoint_commit_image(self, pid: str, image: AgentImage) -> None:
        artifact = self._load_image_artifact(image, expected_kind="checkpoint_commit")
        self.checkpoint._require_snapshot_modules({"modules": artifact.get("modules", [])})
        remapped = self._remap_image_artifact_for_process(pid, artifact)
        self._insert_committed_memory_rows(remapped)
        self._restore_committed_registry_rows(artifact)
        tool_table = self._restore_committed_tool_table(pid, artifact)
        process = self.process.get(pid)
        process.working_directory = str(artifact.get("working_directory") or process.working_directory or ".")
        process.loaded_skills = self._remap_loaded_skills(artifact.get("loaded_skills", {}), tool_table)
        process.tool_table = tool_table
        self._merge_committed_memory_view(process, artifact, remapped)
        process.updated_at = utc_now()
        self.store.update_process(process)
        self.audit.record(
            actor=f"image:{image.image_id}",
            action="image.boot.checkpoint_commit",
            target=f"process:{pid}",
            decision={
                "image_id": image.image_id,
                "artifact_id": image.boot.get("artifact_id"),
                "source_checkpoint_id": artifact.get("source_checkpoint_id"),
                "objects": len(remapped["object_payloads"]),
                "tools": sorted(tool_table),
            },
        )

    def _instantiate_image_package(self, pid: str, image: AgentImage) -> None:
        artifact = self._load_image_artifact(image, expected_kind="image_package")
        workspace_paths = self._materialize_image_package_workspace(pid, image, artifact)
        registered_jit = self._register_image_package_jit_tools(pid, image, artifact)
        process = self.process.get(pid)
        workspace_root = None
        working_directory = None
        if workspace_paths is not None:
            workspace_root, working_directory = workspace_paths
            process.working_directory = working_directory
            process.updated_at = utc_now()
            self.store.update_process(process)
            self._grant_image_package_workspace(pid, image, artifact, workspace_root)
        self.audit.record(
            actor=f"image:{image.image_id}",
            action="image.boot.package",
            target=f"process:{pid}",
            decision={
                "image_id": image.image_id,
                "artifact_id": image.boot.get("artifact_id"),
                "package_sha256": artifact.get("package_sha256"),
                "workspace_root": workspace_root,
                "working_directory": working_directory,
                "jit_tools": registered_jit,
            },
        )

    def _materialize_image_package_workspace(
        self,
        pid: str,
        image: AgentImage,
        artifact: dict[str, Any],
    ) -> tuple[str, str] | None:
        workspace = artifact.get("workspace") or {}
        source = workspace.get("source")
        if not source:
            return None
        artifact_id = self._safe_materialized_segment(str(image.boot.get("artifact_id") or "image"))
        pid_segment = self._safe_materialized_segment(pid)
        boot_segment = self._safe_materialized_segment(new_id("boot"))
        root_relative = Path(self.config.image.materialized_workspace_root) / pid_segment / boot_segment / artifact_id / "workspace"
        root = (self.workspace_root / root_relative).resolve()
        if self.workspace_root not in root.parents and root != self.workspace_root:
            raise RuntimeError("image workspace materialization escaped workspace root")
        files = [
            record for record in artifact.get("files", [])
            if self._artifact_path_under(str(record.get("path", "")), str(source))
        ]
        total_bytes = sum(int(record.get("size_bytes") or 0) for record in files)
        usage = ResourceUsage(external_write_bytes=total_bytes)
        context = {
            "image_id": image.image_id,
            "artifact_id": image.boot.get("artifact_id"),
            "workspace_root": root_relative.as_posix(),
            "files": len(files),
            "bytes": total_bytes,
        }
        self.resources.preflight(pid, usage, source="image.workspace.materialize", context=context)
        root.mkdir(parents=True, exist_ok=True)
        for record in files:
            package_path = str(record["path"])
            relative = self._relative_artifact_path(package_path, str(source))
            target = (root / relative).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"image workspace file escaped materialized root: {package_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._artifact_file_bytes(record))
        working_directory = str(workspace.get("working_directory") or ".")
        cwd = (root / "" if working_directory == "." else root / working_directory).resolve()
        if root not in cwd.parents and cwd != root:
            raise RuntimeError("image workspace working_directory escaped materialized root")
        cwd.mkdir(parents=True, exist_ok=True)
        self.resources.charge(pid, usage, source="image.workspace.materialize", context=context)
        return root.relative_to(self.workspace_root).as_posix(), cwd.relative_to(self.workspace_root).as_posix()

    def _grant_image_package_workspace(
        self,
        pid: str,
        image: AgentImage,
        artifact: dict[str, Any],
        workspace_root: str,
    ) -> None:
        workspace = artifact.get("workspace") or {}
        granted: list[dict[str, Any]] = []
        for grant in workspace.get("grants", []):
            relative = str(grant.get("path") or ".")
            target = workspace_root if relative == "." else f"{workspace_root.rstrip('/')}/{relative}"
            rights = [CapabilityRight(right) for right in grant.get("rights", [])]
            if grant.get("recursive"):
                cap = self.filesystem.grant_directory(
                    pid,
                    target,
                    rights,
                    issued_by=f"image.package:{image.image_id}",
                    delegable=bool(grant.get("delegable", False)),
                )
            else:
                cap = self.filesystem.grant_path(
                    pid,
                    target,
                    rights,
                    issued_by=f"image.package:{image.image_id}",
                    delegable=bool(grant.get("delegable", False)),
                )
            granted.append({"capability_id": cap.cap_id, "resource": cap.resource, "rights": sorted(cap.rights)})
        if granted:
            self.audit.record(
                actor=f"image:{image.image_id}",
                action="image.workspace.grants",
                target=f"process:{pid}",
                decision={"grants": granted},
            )

    def _register_image_package_jit_tools(self, pid: str, image: AgentImage, artifact: dict[str, Any]) -> list[str]:
        process = self.process.get(pid)
        registered: list[str] = []
        prepared: list[tuple[str, str]] = []
        for item in artifact.get("jit_tools", []):
            name = str(item.get("name") or "")
            if not name:
                continue
            if name in process.tool_table:
                raise RuntimeError(f"image package JIT tool conflicts with visible tool: {name}")
            if self.tools._name_collides_with_static_tool(name):
                raise RuntimeError(f"image package JIT tool conflicts with static tool: {name}")
            spec = ToolSpec(
                name=name,
                description=str(item.get("description") or ""),
                input_schema=dict(item.get("input_schema") or {}),
                output_schema=dict(item.get("output_schema") or {}),
                tags=["image", "jit", "package"],
                metadata={
                    "image_id": image.image_id,
                    "artifact_id": image.boot.get("artifact_id"),
                    "source_path": item.get("source_path"),
                    **dict(item.get("metadata") or {}),
                },
            )
            candidate_id = self.tools.propose(
                pid,
                spec,
                source_code=str(item.get("source") or ""),
                tests=[dict(test) for test in item.get("tests", [])],
            )
            validation = self.tools.validate(candidate_id, pid=pid)
            if not validation.ok:
                raise ValidationError(f"image package JIT tool {name} failed validation: {'; '.join(validation.errors)}")
            prepared.append((name, candidate_id))
        handles: dict[str, ToolHandle] = {}
        try:
            for name, candidate_id in prepared:
                handle = self.tools.register(pid, candidate_id, approver=f"image.package:{image.image_id}")
                handles[name] = handle
                registered.append(name)
        except Exception:
            self._remove_registered_image_package_jit_tools(pid, handles)
            raise
        if registered:
            current_process = self.store.get_process(pid)
            if current_process is not None:
                current_process.updated_at = utc_now()
                self.store.update_process(current_process)
            self.audit.record(
                actor=f"image:{image.image_id}",
                action="image.package_jit.register",
                target=f"process:{pid}",
                decision={"tools": sorted(registered)},
            )
        return sorted(registered)

    def _remove_registered_image_package_jit_tools(self, pid: str, handles: dict[str, ToolHandle]) -> None:
        if not handles:
            return
        process = self.store.get_process(pid)
        if process is not None:
            for name, handle in handles.items():
                if process.tool_table.get(name) == handle.tool_id:
                    process.tool_table.pop(name, None)
            process.updated_at = utc_now()
            self.store.update_process(process)
        for handle in handles.values():
            getattr(self.tools, "_jit_sources", {}).pop(handle.tool_id, None)
            getattr(self.tools, "_handles", {}).pop(handle.tool_id, None)

    def _load_image_artifact(self, image: AgentImage, *, expected_kind: str | None = None) -> dict[str, Any]:
        artifact_id = str(image.boot.get("artifact_id") or "")
        expected_sha256 = str(image.boot.get("artifact_sha256") or "")
        expected_kind = expected_kind or str(image.boot.get("kind") or "")
        found = self.store.get_image_artifact(artifact_id)
        if found is None:
            raise NotFound(f"image artifact not found: {artifact_id}")
        artifact, metadata = found
        if expected_kind and artifact.get("kind") != expected_kind:
            raise RuntimeError(f"image artifact kind mismatch: {artifact.get('kind')} != {expected_kind}")
        expected_version = self.config.image_commit.artifact_version if artifact.get("kind") == "checkpoint_commit" else 1
        if artifact.get("artifact_version") != expected_version:
            raise RuntimeError(
                "image artifact version mismatch: "
                f"{artifact.get('artifact_version')} != {expected_version}"
            )
        actual_sha256 = hashlib.sha256(dumps(artifact).encode("utf-8")).hexdigest()
        if expected_sha256 and metadata.get("sha256") != expected_sha256:
            raise RuntimeError(f"image artifact hash mismatch for {artifact_id}")
        if metadata.get("sha256") != actual_sha256:
            raise RuntimeError(f"image artifact content hash mismatch for {artifact_id}")
        return artifact

    def _artifact_file_bytes(self, record: dict[str, Any]) -> bytes:
        if record.get("kind") == "base64":
            return base64.b64decode(str(record.get("content_base64") or ""))
        return str(record.get("content") or "").encode("utf-8")

    def _artifact_path_under(self, path: str, root: str) -> bool:
        return path == root or path.startswith(f"{root.rstrip('/')}/")

    def _relative_artifact_path(self, path: str, root: str) -> Path:
        root = root.rstrip("/")
        if path == root:
            return Path()
        if not path.startswith(f"{root}/"):
            raise RuntimeError(f"artifact path is outside root: {path}")
        return Path(path[len(root) + 1 :])

    def _safe_materialized_segment(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.@+-]", "_", value)[:160] or "image"

    def _remap_image_artifact_for_process(self, pid: str, artifact: dict[str, Any]) -> dict[str, Any]:
        source_pid = str(artifact["source_pid"])
        old_oids = list(artifact.get("object_oids", []))
        oid_map = {oid: new_id("obj") for oid in old_oids}
        namespace_map = {
            namespace: self._remap_image_artifact_namespace(pid, source_pid, namespace)
            for namespace in artifact.get("namespaces", [])
        }
        cap_rows = artifact.get("rows", {}).get("capabilities", [])
        cap_map = {row["cap_id"]: new_id("cap") for row in cap_rows}
        now = utc_now()
        object_rows = [
            self._remap_committed_object_row(row, pid, oid_map, namespace_map, now)
            for row in artifact.get("rows", {}).get("objects", [])
            if row["oid"] in oid_map
        ]
        namespace_rows = [
            self._remap_committed_namespace_row(row, pid, namespace_map, now)
            for row in artifact.get("rows", {}).get("object_namespaces", [])
            if row["namespace"] in namespace_map
        ]
        link_rows = [
            self._remap_committed_link_row(row, oid_map, now)
            for row in artifact.get("rows", {}).get("object_links", [])
            if row["src_oid"] in oid_map and row["dst_oid"] in oid_map
        ]
        capability_rows = [
            self._remap_committed_capability_row(row, pid, oid_map, namespace_map, cap_map, now)
            for row in cap_rows
            if row["subject"] == source_pid
        ]
        payloads = {
            oid_map[oid]: deepcopy(payload)
            for oid, payload in artifact.get("object_payloads", {}).items()
            if oid in oid_map
        }
        return {
            "oid_map": oid_map,
            "namespace_map": namespace_map,
            "capability_map": cap_map,
            "object_namespaces": namespace_rows,
            "objects": object_rows,
            "object_links": link_rows,
            "capabilities": capability_rows,
            "object_payloads": payloads,
        }

    def _remap_image_artifact_namespace(self, pid: str, source_pid: str, namespace: str) -> str:
        source_process_namespace = self.memory.process_namespace(source_pid)
        if namespace == source_process_namespace:
            return self.memory.process_namespace(pid)
        return f"image_commit/{pid}/{namespace}"

    def _remap_committed_namespace_row(
        self,
        row: dict[str, Any],
        pid: str,
        namespace_map: dict[str, str],
        now: str,
    ) -> dict[str, Any]:
        item = dict(row)
        item["namespace"] = namespace_map[item["namespace"]]
        if item.get("parent_namespace") in namespace_map:
            item["parent_namespace"] = namespace_map[item["parent_namespace"]]
        elif item["namespace"] == self.memory.process_namespace(pid):
            item["parent_namespace"] = None
        item["created_by"] = pid
        metadata = loads(item.get("metadata_json"), {})
        if metadata.get("kind") == "process":
            metadata["pid"] = pid
        item["metadata_json"] = dumps(metadata)
        item["updated_at"] = now
        return item

    def _remap_committed_object_row(
        self,
        row: dict[str, Any],
        pid: str,
        oid_map: dict[str, str],
        namespace_map: dict[str, str],
        now: str,
    ) -> dict[str, Any]:
        item = dict(row)
        old_oid = item["oid"]
        item["oid"] = oid_map[old_oid]
        if item.get("name") == old_oid:
            item["name"] = item["oid"]
        item["namespace"] = namespace_map.get(item["namespace"], item["namespace"])
        item["created_by"] = pid
        item["owner_kind"] = ObjectOwnerKind.PROCESS.value
        item["owner_id"] = pid
        item["lifecycle_state"] = "live"
        item["deleted_at"] = None
        provenance = loads(item.get("provenance_json"), {})
        provenance["parent_oids"] = [oid_map.get(oid, oid) for oid in provenance.get("parent_oids", [])]
        item["provenance_json"] = dumps(provenance)
        item["payload_json"] = dumps(self.store._memory_payload_marker(present=True))
        item["created_at"] = now
        item["updated_at"] = now
        return item

    def _remap_committed_link_row(self, row: dict[str, Any], oid_map: dict[str, str], now: str) -> dict[str, Any]:
        item = dict(row)
        item["id"] = new_id("link")
        item["src_oid"] = oid_map[item["src_oid"]]
        item["dst_oid"] = oid_map[item["dst_oid"]]
        item["created_at"] = now
        return item

    def _remap_committed_capability_row(
        self,
        row: dict[str, Any],
        pid: str,
        oid_map: dict[str, str],
        namespace_map: dict[str, str],
        cap_map: dict[str, str],
        now: str,
    ) -> dict[str, Any]:
        item = dict(row)
        item["cap_id"] = cap_map[item["cap_id"]]
        item["subject"] = pid
        item["issuer_cap_id"] = cap_map.get(item.get("issuer_cap_id")) if item.get("issuer_cap_id") else None
        item["parent_cap_id"] = cap_map.get(item.get("parent_cap_id")) if item.get("parent_cap_id") else None
        resource = str(item["resource"])
        if resource.startswith("object:"):
            item["resource"] = f"object:{oid_map[resource.split(':', 1)[1]]}"
        elif resource.startswith("object_namespace:"):
            namespace = resource.split(":", 1)[1]
            item["resource"] = f"object_namespace:{namespace_map[namespace]}"
        item["issued_by"] = f"image.commit:{item['issued_by']}"
        item["issued_at"] = now
        return item

    def _insert_committed_memory_rows(self, remapped: dict[str, Any]) -> None:
        with self.store._lock:
            cur = self.store.conn.cursor()
            for row in remapped["object_namespaces"]:
                exists = cur.execute("SELECT 1 FROM object_namespaces WHERE namespace = ?", (row["namespace"],)).fetchone()
                if exists is None:
                    self.checkpoint._insert_row(cur, "object_namespaces", row)
            for row in remapped["objects"]:
                self.checkpoint._insert_row(cur, "objects", row)
                self.store.set_object_payload(row["oid"], deepcopy(remapped["object_payloads"][row["oid"]]))
            for table in ["object_links", "capabilities"]:
                for row in remapped[table]:
                    self.checkpoint._insert_row(cur, table, row)
            self.store.conn.commit()

    def _restore_committed_registry_rows(self, artifact: dict[str, Any]) -> None:
        rows = artifact.get("rows", {})
        with self.store._lock:
            cur = self.store.conn.cursor()
            for row in rows.get("skills", []):
                self.checkpoint._upsert_row(cur, "skills", row, "skill_id")
            # Checkpoint-derived images restore only internal process runtime
            # state. External/provider registries and global trust decisions are
            # host state, so even legacy artifacts carrying those rows must not
            # resurrect them during image boot.
            self.store.conn.commit()

    def _restore_committed_tool_table(self, pid: str, artifact: dict[str, Any]) -> dict[str, str]:
        tool_rows = {row["tool_id"]: row for row in artifact.get("rows", {}).get("tools", [])}
        old_to_new: dict[str, str] = {}
        table: dict[str, str] = {}
        jit_sources = artifact.get("jit_sources", {})
        for name, old_tool_id in artifact.get("tool_table", {}).items():
            if old_tool_id in jit_sources:
                row = tool_rows.get(old_tool_id)
                if row is None:
                    raise RuntimeError(f"committed JIT tool row is missing: {old_tool_id}")
                new_tool_id = new_id("tool")
                old_to_new[old_tool_id] = new_tool_id
                spec = ToolSpec(**loads(row["spec_json"], {}))
                handle = ToolHandle(tool_id=new_tool_id, name=row["name"], capability_id=None, scope=row["scope"])
                now = utc_now()
                self.tools._jit_sources[new_tool_id] = jit_sources[old_tool_id]
                self.tools._handles[new_tool_id] = handle
                self.store.insert_tool(handle, spec, registered_by=f"image.commit:{pid}", created_at=now, ephemeral=True)
                self.store.insert_tool_candidate(
                    ToolCandidate(
                        candidate_id=new_id("tcand"),
                        pid=pid,
                        spec=spec,
                        source_code=jit_sources[old_tool_id],
                        tests=[],
                        requested_capabilities=[],
                        status=ToolCandidateStatus.REGISTERED,
                        validation={"ok": True, "source": "image.commit"},
                        created_at=now,
                        updated_at=now,
                        registered_tool_id=new_tool_id,
                    )
                )
                table[name] = new_tool_id
                continue
            handle = self.tools.resolve(name)
            old_to_new[old_tool_id] = handle.tool_id
            table[name] = handle.tool_id
        artifact["_tool_id_map"] = old_to_new
        return table

    def _remap_loaded_skills(self, loaded_skills: dict[str, Any], tool_table: dict[str, str]) -> dict[str, Any]:
        updated = deepcopy(loaded_skills or {})
        for loaded in updated.values():
            if not isinstance(loaded, dict):
                continue
            for key in ["tool_ids", "jit_tool_ids"]:
                mapping = loaded.get(key)
                if not isinstance(mapping, dict):
                    continue
                loaded[key] = {
                    name: tool_table[name]
                    for name in mapping
                    if name in tool_table
                }
        return updated

    def _merge_committed_memory_view(self, process: Any, artifact: dict[str, Any], remapped: dict[str, Any]) -> None:
        source = loads(artifact.get("source_process", {}).get("memory_view_json"), {})
        if not source:
            return
        existing_roots = list(process.memory_view.roots) if process.memory_view is not None else []
        roots = []
        cap_map = remapped["capability_map"]
        oid_map = remapped["oid_map"]
        for root in source.get("roots", []):
            old_oid = root.get("oid")
            if old_oid not in oid_map:
                continue
            old_cap = root.get("capability_id")
            new_oid = oid_map[old_oid]
            rights = set(root.get("rights", []))
            new_cap = cap_map.get(old_cap)
            if new_cap is None:
                handle = self.capability.handle_for_object(subject=process.pid, oid=new_oid, rights=rights, issued_by="image.commit")
                new_cap = handle.capability_id
            roots.append(ObjectHandle(oid=new_oid, rights=rights, capability_id=new_cap, expires_at=root.get("expires_at")))
        for handle in existing_roots:
            if all(item.oid != handle.oid for item in roots):
                roots.append(handle)
        if process.memory_view is None:
            process.memory_view = self.memory.create_view(process.pid, roots, mode="mutable")
        else:
            process.memory_view.roots = roots

    def _apply_loaded_skill_tool_table(self, pid: str) -> None:
        process = self.store.get_process(pid)
        if process is None or not process.loaded_skills:
            return
        updated = dict(process.tool_table)
        for loaded in process.loaded_skills.values():
            if not isinstance(loaded, dict):
                continue
            for mapping_key in ["tool_ids", "jit_tool_ids"]:
                mapping = loaded.get(mapping_key)
                if not isinstance(mapping, dict):
                    continue
                for name, tool_id in mapping.items():
                    if isinstance(name, str) and isinstance(tool_id, str):
                        updated[name] = tool_id
        process.tool_table = updated
        process.updated_at = utc_now()
        self.store.update_process(process)
