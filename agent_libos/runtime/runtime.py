from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from agent_libos.config import AgentLibOSConfig
from agent_libos.llm.client import LLMClient
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    CapabilityUseReservationRecoverySummary,
    DataFlowContext,
    DataLabels,
    ExternalEffectRecoverySummary,
    ForkMode,
    MemoryView,
    MemoryViewSpec,
    ObjectHandle,
    ObjectMetadata,
    ObjectPayloadRecoverySummary,
    ObjectTaskRecoverySummary,
    ProcessStatus,
    SinkTrustRule,
    SinkTrustSpec,
    ResourceUsageReservationRecoverySummary,
    StaleExecutionRecoverySummary,
    StaleOperationRecoverySummary,
    TaskRunSummary,
    WorkflowRunResult,
)
from agent_libos.models.exceptions import (
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.process_execution import (
    bind_process_execution,
    current_process_execution_token,
)
from agent_libos.storage import RuntimeStore
from agent_libos.substrate import ResourceProviderSubstrate
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)

if TYPE_CHECKING:
    from agent_libos.capability.manager import CapabilityManager
    from agent_libos.evidence import PayloadRetentionMaintenance
    from agent_libos.human.manager import HumanObjectManager
    from agent_libos.llm.executor import LLMProcessExecutor
    from agent_libos.llm.profiles import LLMProfileRegistry
    from agent_libos.memory.object_memory import ObjectMemoryManager
    from agent_libos.modules import RuntimeModuleRegistry
    from agent_libos.modules.host import ModuleStateRegistry
    from agent_libos.primitives import (
        ClockPrimitive,
        FilesystemAdapter,
        GitPrimitive,
        JsonRpcPrimitive,
        McpPrimitive,
        ShellAdapter,
    )
    from agent_libos.runtime.audit_manager import AuditManager
    from agent_libos.runtime.authority_manifest_manager import AuthorityManifestManager
    from agent_libos.runtime.checkpoint_image import CheckpointImageInstaller
    from agent_libos.runtime.checkpoint_manager import CheckpointManager
    from agent_libos.runtime.data_flow_manager import DataFlowManager
    from agent_libos.runtime.event_bus import EventBus
    from agent_libos.runtime.explain_manager import ExplainManager
    from agent_libos.runtime.image_artifact import ImageArtifactLoader
    from agent_libos.runtime.image_boot import ImageBootService
    from agent_libos.runtime.image_package import ImagePackageInstaller
    from agent_libos.runtime.image_registry import ImageRegistryPrimitive
    from agent_libos.runtime.lifecycle import RuntimeLifecycle
    from agent_libos.runtime.message_manager import ProcessMessageManager
    from agent_libos.runtime.object_tasks import ObjectTaskManager
    from agent_libos.runtime.operation_manager import OperationManager
    from agent_libos.runtime.process_launch import ProcessLaunchService
    from agent_libos.runtime.process_manager import ProcessManager
    from agent_libos.process_transition import ProcessTransitionService
    from agent_libos.runtime.ratings import AgentRatingManager
    from agent_libos.runtime.resource_manager import ResourceManager
    from agent_libos.runtime.scheduler import SimpleScheduler
    from agent_libos.runtime.snapshots import ProcessExecStateService
    from agent_libos.runtime.syscall_router import SyscallRouter
    from agent_libos.runtime.task_runs import TaskRunManager
    from agent_libos.sdk import ProtectedOperationSDK
    from agent_libos.skills.manager import SkillManager
    from agent_libos.storage import UnitOfWork
    from agent_libos.tools.broker import ToolBroker


@dataclass(frozen=True, slots=True)
class HumanRunContext:
    """Immutable terminal policy attached to one host run invocation."""

    runtime_identity: int | None = None
    human: str | None = None
    auto_approve: bool | None = None
    auto_policy: str | None = None
    auto_answer: str | None = None


_EMPTY_HUMAN_RUN_CONTEXT = HumanRunContext()
_HUMAN_RUN_CONTEXT: ContextVar[HumanRunContext] = ContextVar(
    "agent_libos_human_run_context",
    default=_EMPTY_HUMAN_RUN_CONTEXT,
)


class Runtime:
    """Public host facade over services assembled by :class:`RuntimeBuilder`.

    Host effects enter through primitives and provider interfaces. Runtime keeps
    the stable host API while subsystem services own their state machines.
    """

    config: AgentLibOSConfig
    substrate: ResourceProviderSubstrate
    workspace_root: Path
    store: RuntimeStore
    instance_id: str
    images: dict[str, AgentImage]
    module_state: ModuleStateRegistry
    _registry_lifecycle_lock: Any
    uow: UnitOfWork
    operations: OperationManager
    audit: AuditManager
    events: EventBus
    lifecycle: RuntimeLifecycle
    blocking_work: Any
    capability: CapabilityManager
    llms: LLMProfileRegistry
    ratings: AgentRatingManager
    resources: ResourceManager
    syscalls: SyscallRouter
    provider_hooks: dict[str, list[Any]]
    authority_manifests: AuthorityManifestManager
    explain: ExplainManager
    memory: ObjectMemoryManager
    data_flow: DataFlowManager
    protected_operations: ProtectedOperationSDK
    external_primitive_boundary_names: frozenset[str]
    process: ProcessManager
    process_transitions: ProcessTransitionService
    messages: ProcessMessageManager
    human: HumanObjectManager
    clock: ClockPrimitive
    filesystem: FilesystemAdapter
    git: GitPrimitive
    shell: ShellAdapter
    jsonrpc: JsonRpcPrimitive
    mcp: McpPrimitive
    tools: ToolBroker
    object_tasks: ObjectTaskManager
    task_runs: TaskRunManager
    scheduler: SimpleScheduler
    checkpoint: CheckpointManager
    skills: SkillManager
    process_exec_state: ProcessExecStateService
    image_artifacts: ImageArtifactLoader
    checkpoint_image_installer: CheckpointImageInstaller
    image_package_installer: ImagePackageInstaller
    image_registry: ImageRegistryPrimitive
    launch: ProcessLaunchService
    modules: RuntimeModuleRegistry
    image_boot: ImageBootService
    llm: LLMProcessExecutor
    recovered_prepared_operations: ExternalEffectRecoverySummary
    recovered_missing_object_payloads: ObjectPayloadRecoverySummary
    recovered_object_tasks: ObjectTaskRecoverySummary
    recovered_capability_use_reservations: CapabilityUseReservationRecoverySummary
    recovered_resource_usage_reservations: ResourceUsageReservationRecoverySummary
    recovered_exec_publications: list[str]
    recovered_runtime_publications: list[str]
    recovered_checkpoint_restore_publications: list[str]
    recovered_root_spawn_initial_goal_payloads: tuple[str, ...]
    recovered_stale_operations: StaleOperationRecoverySummary
    recovered_stale_executions: StaleExecutionRecoverySummary
    recovered_terminal_cleanups: dict[str, Any]
    recovered_task_runs: tuple[TaskRunSummary, ...]
    recovered_task_run_count: int
    reconciled_external_effects: ExternalEffectRecoverySummary
    payload_retention: PayloadRetentionMaintenance
    explainable_boundary_names: frozenset[str]
    mutation_admission_boundary_names: frozenset[str]
    _recovery_diagnostics_release_capability: object

    @classmethod
    def allocate_unassembled(cls) -> "Runtime":
        """Allocate a builder-owned host without running Runtime assembly.

        RuntimeBuilder uses this hook for both synchronous and asynchronous
        construction. Subclasses that override ``__init__`` must override this
        hook as well and initialize subclass-only fields here; builder assembly
        intentionally never calls a custom constructor around a live Runtime.
        """

        return cls.__new__(cls)

    def __init__(
        self,
        store: RuntimeStore,
        llm_client: LLMClient | None = None,
        substrate: ResourceProviderSubstrate | None = None,
        config: AgentLibOSConfig | None = None,
        startup_module_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None = None,
    ):
        from agent_libos.runtime.builder import RuntimeBuilder

        RuntimeBuilder.assemble_existing(
            self,
            store,
            llm_client=llm_client,
            substrate=substrate,
            config=config,
            startup_module_manifests=startup_module_manifests,
            trusted_modules=trusted_modules,
            trusted_module_sha256=trusted_module_sha256,
        )

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
        from agent_libos.runtime.builder import RuntimeBuilder

        return RuntimeBuilder.configured(
            cls,
            config=config,
            substrate=substrate,
            module_manifests=module_manifests,
            trusted_modules=trusted_modules,
            trusted_module_sha256=trusted_module_sha256,
        ).open(target)

    @classmethod
    async def aopen(
        cls,
        target: str | Path | None = None,
        substrate: ResourceProviderSubstrate | None = None,
        config: AgentLibOSConfig | None = None,
        module_manifests: list[str | Path] | tuple[str | Path, ...] | None = None,
        trusted_modules: list[str] | tuple[str, ...] | None = None,
        trusted_module_sha256: list[str] | tuple[str, ...] | None = None,
    ) -> "Runtime":
        """Assemble a Runtime in a blocking worker with caller-loop coordination."""

        from agent_libos.runtime.builder import RuntimeBuilder

        return await RuntimeBuilder.configured(
            cls,
            config=config,
            substrate=substrate,
            module_manifests=module_manifests,
            trusted_modules=trusted_modules,
            trusted_module_sha256=trusted_module_sha256,
        ).aopen(target)

    def shutdown(self, *, actor: str = "runtime", reason: str = "runtime.shutdown") -> dict[str, Any]:
        """Shut down this host Runtime instance.

        Shutdown is a host lifecycle operation. It stops accepting further use
        of this composition root and releases owned handles, but it does not
        change AgentProcess lifecycle state. A process must still exit through
        the process primitive/tool path, which keeps process authority and audit
        semantics separate from host resource cleanup.
        """
        return self.lifecycle.shutdown(actor=actor, reason=reason)

    async def ashutdown(self, *, actor: str = "runtime", reason: str = "runtime.shutdown") -> dict[str, Any]:
        """Async shutdown variant for event-loop hosts."""
        return await self.lifecycle.ashutdown(actor=actor, reason=reason)

    def close(self) -> dict[str, Any]:
        """Compatibility alias for shutdown(); prefer Runtime.shutdown()."""
        return self.shutdown(actor="runtime.close", reason="runtime.close")

    def release_recovery_diagnostics(self) -> dict[str, Any]:
        """Release a recovery-fenced Runtime for same-process reopen.

        Ordinary shutdown remains fail-closed while recovery is required. This
        explicit handoff preserves durable diagnostics and releases only the
        current Runtime instance's transient workers and store ownership.
        """

        return self.lifecycle.release_recovery_diagnostics(
            capability=self._recovery_diagnostics_release_capability,
        )

    async def arelease_recovery_diagnostics(self) -> dict[str, Any]:
        """Async recovery diagnostics handoff variant."""

        return await self.lifecycle.arelease_recovery_diagnostics(
            capability=self._recovery_diagnostics_release_capability,
        )

    def _shutdown_component(self, component: Any) -> bool:
        return self.lifecycle.shutdown_component(component)

    def bind_shutdown_finalizer(self, finalizer: Any) -> None:
        self.lifecycle.bind_finalizer(finalizer)

    def _notify_process_terminal(self, pid: str) -> None:
        errors: list[tuple[str, BaseException]] = []
        try:
            self.human.cancel_pending_for_process(
                pid,
                actor="runtime.process_terminal",
                reason="process reached a terminal state",
            )
        except BaseException as exc:
            errors.append(("human", exc))
        try:
            self.object_tasks.notify_process_terminal(pid)
        except BaseException as exc:
            errors.append(("object_tasks", exc))
        if not errors:
            return

        aggregate = RuntimeError("terminal process notification failed")
        envelope = public_error_envelope(
            aggregate,
            code="terminal_cleanup_failed",
        )
        # Keep the raised RuntimeError and every downstream public projection on
        # the exact same cached envelope/correlation identity.  Per-component
        # diagnostic text is fingerprinted only; it is never embedded in the
        # exception or the structured cleanup receipt.
        aggregate.args = (envelope["message"],)
        failures: list[dict[str, Any]] = []
        interruptions: list[BaseException] = []
        for phase, error in errors:
            observation = internal_exception_observation(
                error,
                correlation_id=envelope["correlation_id"],
            )
            failures.append(
                {
                    "phase": phase,
                    "error_type": observation["error_type"],
                    "correlation_id": envelope["correlation_id"],
                    "exception_text": observation["exception_text"],
                }
            )
            if not isinstance(error, Exception):
                interruptions.append(error)
        aggregate.cleanup_failures = tuple(failures)  # type: ignore[attr-defined]
        if interruptions:
            raise BaseExceptionGroup(
                "terminal process notification interrupted after full cleanup",
                [aggregate, *interruptions],
            )
        raise aggregate

    async def _ashutdown_component(self, component: Any) -> bool:
        return await self.lifecycle.ashutdown_component(component)

    def run_process_once(self, pid: str) -> dict[str, Any]:
        if self.scheduler.is_active_quantum(pid):
            return self.llm.run_once(pid)
        return self.scheduler.run_pid_once(pid, self.llm.arun_once)

    async def arun_process_once(self, pid: str) -> dict[str, Any]:
        if self.scheduler.is_active_quantum(pid):
            return await self.llm.arun_once(pid)
        return await self.scheduler.arun_pid_once(pid, self.llm.arun_once)

    def run_next_process_once(self) -> Any:
        return self.scheduler.run_once(self.arun_process_once)

    async def arun_next_process_once(self) -> Any:
        return await self.scheduler.arun_once(self.arun_process_once)

    def current_human_run_context(self) -> HumanRunContext:
        """Return the terminal policy belonging to this Runtime's current run."""
        context = _HUMAN_RUN_CONTEXT.get()
        if context.runtime_identity != id(self):
            return _EMPTY_HUMAN_RUN_CONTEXT
        return context

    @contextmanager
    def human_run_context(
        self,
        *,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> Iterator[HumanRunContext]:
        """Install one immutable run policy for scheduler and JIT descendants."""
        context = HumanRunContext(
            runtime_identity=id(self),
            human=human,
            auto_approve=human_auto_approve,
            auto_policy=human_auto_policy,
            auto_answer=human_auto_answer,
        )
        token = _HUMAN_RUN_CONTEXT.set(context)
        try:
            yield context
        finally:
            _HUMAN_RUN_CONTEXT.reset(token)

    def run_until_idle(
        self,
        max_quanta: int | None = None,
        *,
        pids: Iterable[str] | None = None,
        process_human_queue: bool = True,
        cancel_inflight_on_budget_exhaustion: bool = True,
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
                pids=pids,
                process_human_queue=process_human_queue,
                cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
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
        pids: Iterable[str] | None = None,
        process_human_queue: bool = True,
        cancel_inflight_on_budget_exhaustion: bool = True,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> list[Any]:
        selected_pids = self._normalize_run_until_idle_pids(pids)
        self._validate_run_until_idle_controls(
            max_quanta=max_quanta,
            process_human_queue=process_human_queue,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )
        return await self.blocking_work.run(
            self._run_until_idle_sync,
            max_quanta=max_quanta,
            pids=selected_pids,
            process_human_queue=process_human_queue,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
            human=human,
            human_auto_approve=human_auto_approve,
            human_auto_policy=human_auto_policy,
            human_auto_answer=human_auto_answer,
        )

    def _run_until_idle_sync(
        self,
        max_quanta: int | None = None,
        *,
        pids: Iterable[str] | None = None,
        process_human_queue: bool = True,
        cancel_inflight_on_budget_exhaustion: bool = True,
        human: str | None = None,
        human_auto_approve: bool | None = None,
        human_auto_policy: str | None = None,
        human_auto_answer: str | None = None,
    ) -> list[Any]:
        selected_pids = self._normalize_run_until_idle_pids(pids)
        self._validate_run_until_idle_controls(
            max_quanta=max_quanta,
            process_human_queue=process_human_queue,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )
        results: list[Any] = []
        remaining = self.config.runtime.run_until_idle_max_quanta if max_quanta is None else max_quanta
        selected_human = human or self.config.runtime.default_human
        with self.human_run_context(
            human=selected_human,
            human_auto_approve=human_auto_approve,
            human_auto_policy=human_auto_policy,
            human_auto_answer=human_auto_answer,
        ):
            while remaining is None or remaining > 0:
                # Run all currently runnable processes first. Human queue work below
                # may wake a process, so this loop intentionally alternates between
                # process execution and terminal queue draining.
                batch = self.scheduler.run_until_idle(
                    self.arun_process_once,
                    max_quanta=remaining,
                    cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
                    pids=selected_pids,
                )
                results.extend(batch)
                if remaining is not None:
                    remaining -= len(batch)
                if not process_human_queue:
                    break
                human_filters: dict[str, Any] = {
                    "human": selected_human,
                    "auto_approve": human_auto_approve,
                    "auto_policy": human_auto_policy,
                    "auto_answer": human_auto_answer,
                }
                if selected_pids is not None:
                    human_filters["pids"] = selected_pids
                processed = self.human.drain_terminal_queue(**human_filters)
                if not processed:
                    break
                self.audit.record(
                    actor="runtime",
                    action="runtime.human_queue_drained",
                    target=f"human:{selected_human}",
                    decision={"request_ids": [request.request_id for request in processed]},
                )
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
        self.scheduler.validate_max_quanta(selected_quanta)
        return await self.scheduler.arun_pid_until_idle(
            pid,
            self.arun_process_once,
            max_quanta=selected_quanta,
        )

    def _validate_run_until_idle_controls(
        self,
        *,
        max_quanta: int | None,
        process_human_queue: bool,
        cancel_inflight_on_budget_exhaustion: bool,
    ) -> None:
        selected_quanta = (
            self.config.runtime.run_until_idle_max_quanta
            if max_quanta is None
            else max_quanta
        )
        self.scheduler.validate_run_controls(
            max_quanta=selected_quanta,
            cancel_inflight_on_budget_exhaustion=cancel_inflight_on_budget_exhaustion,
        )
        if type(process_human_queue) is not bool:
            raise ValidationError("process_human_queue must be a boolean")

    @staticmethod
    def _normalize_run_until_idle_pids(
        pids: Iterable[str] | None,
    ) -> frozenset[str] | None:
        """Normalize an explicit scheduler scope without silently widening it."""

        if pids is None:
            return None
        if isinstance(pids, (str, bytes)):
            raise ValidationError("run_until_idle pids must be an iterable of PIDs")
        try:
            selected = list(pids)
        except TypeError as exc:
            raise ValidationError(
                "run_until_idle pids must be an iterable of PIDs"
            ) from exc
        if not selected:
            raise ValidationError("run_until_idle pids must not be empty")
        normalized: list[str] = []
        for pid in selected:
            if (
                not isinstance(pid, str)
                or not pid
                or pid != pid.strip()
                or "\x00" in pid
            ):
                raise ValidationError(
                    "run_until_idle pids must contain canonical non-empty strings"
                )
            normalized.append(pid)
        if len(set(normalized)) != len(normalized):
            raise ValidationError("run_until_idle pids must not contain duplicates")
        return frozenset(normalized)

    def run_workflow(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        image: str | None = None,
        goal: dict[str, Any] | str | None = None,
        working_directory: str | None = None,
        authority_manifest: dict[str, Any] | None = None,
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
                    authority_manifest=authority_manifest,
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
        authority_manifest: dict[str, Any] | None = None,
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
        workflow_manifest = self._workflow_authority_manifest(
            selected_image,
            tool_name,
            tool_args,
            authority_manifest,
        )
        pid, execution_token = self.process.spawn_claimed(
            owner_id=f"{self.instance_id}:workflow",
            image=selected_image,
            goal=goal if goal is not None else f"workflow:{tool_name}",
            working_directory=working_directory,
            authority_manifest=workflow_manifest,
        )
        initial = self.process.get(pid)
        initial_image = initial.image_id
        initial_goal_oid = initial.goal_oid
        try:
            try:
                with bind_process_execution(execution_token):
                    return await self._arun_claimed_workflow(
                        pid,
                        selected_image=selected_image,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        initial_image=initial_image,
                        initial_goal_oid=initial_goal_oid,
                    )
            except asyncio.CancelledError:
                # Cancel as Host control after dropping the ambient worker
                # token.  The tool may already have rotated or fenced that
                # token while entering a typed wait or committing exec.
                process = self.process.get(pid)
                if process.status not in self.process.TERMINAL_STATUSES:
                    self.process.cancel(pid, "workflow task cancelled")
                raise
        finally:
            # Wait/exit/exec transitions fence or rotate this token themselves.
            # A plain or exceptional return still releases the exact workflow
            # lease so no process can remain RUNNING without an owner.
            self.uow.processes.complete_execution(
                execution_token,
                status=ProcessStatus.RUNNABLE,
            )

    async def _arun_claimed_workflow(
        self,
        pid: str,
        *,
        selected_image: str,
        tool_name: str,
        tool_args: dict[str, Any],
        initial_image: str,
        initial_goal_oid: str | None,
    ) -> WorkflowRunResult:
        try:
            result = await self.tools.acall(pid, tool_name, tool_args)
        except HumanApprovalRequired as exc:
            error = public_error_envelope(
                exc,
                code="workflow_wait_required",
            )["message"]
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=error,
                waiting_human=True,
                request_id=exc.request_id,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except ProcessWaitRequired as exc:
            error = public_error_envelope(
                exc,
                code="workflow_wait_required",
            )["message"]
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=error,
                waiting_process=True,
                child_pid=exc.child_pid,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except ProcessMessageWaitRequired as exc:
            error = public_error_envelope(
                exc,
                code="workflow_wait_required",
            )["message"]
            workflow_result = self._workflow_wait_result(
                pid,
                selected_image,
                tool_name,
                error=error,
                waiting_message=True,
                filters=exc.filters,
            )
            self._record_workflow_run(workflow_result)
            return workflow_result
        except Exception as exc:
            error = public_error_envelope(
                exc,
                code="workflow_failed",
            )["message"]
            if not self._workflow_tool_controlled_lifecycle(pid, initial_image=initial_image, initial_goal_oid=initial_goal_oid):
                self.process.exit(pid, failed=True, message=error or f"workflow failed: {tool_name}")
            workflow_result = self._workflow_failure_result(pid, selected_image, tool_name, error=error)
            self._record_workflow_run(workflow_result)
            return workflow_result

        tool_controlled_lifecycle = self._workflow_tool_controlled_lifecycle(
            pid,
            initial_image=initial_image,
            initial_goal_oid=initial_goal_oid,
        )
        if result.result_handle is not None and not tool_controlled_lifecycle:
            self._add_handle_to_process_view(pid, result.result_handle)
        if not tool_controlled_lifecycle:
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

    def _workflow_authority_manifest(
        self,
        selected_image: str,
        tool_name: str,
        tool_args: dict[str, Any],
        supplied: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if supplied is not None:
            return dict(supplied)
        authorized: list[dict[str, Any]] = []
        target_image = tool_args.get("image")
        if (
            tool_name == "exec_process"
            and isinstance(target_image, str)
            and target_image != selected_image
            and target_image in self.images
        ):
            authorized.append(
                {
                    "resource": self.image_registry.resource_for(target_image),
                    "rights": [CapabilityRight.READ.value],
                }
            )
        if not authorized:
            return None
        return {
            "authorized_capabilities": authorized,
            "metadata": {"provided_by": "workflow_controller", "tool": tool_name},
        }

    def _workflow_result_from_tool_call(
        self,
        pid: str,
        image: str,
        tool: str,
        result: Any,
    ) -> WorkflowRunResult:
        process = self.process.get(pid)
        resolved_tool_id = self._workflow_resolve_tool_id(pid, tool)
        # ToolBroker returns the requested name in its structured denial when
        # no process-visible tool exists, because ToolCallResult historically
        # requires a non-optional identifier.  Do not expose that placeholder
        # as if it were a registered workflow tool id.
        workflow_tool_id = (
            None
            if not result.ok and resolved_tool_id is None and result.tool_id == tool
            else result.tool_id
        )
        return WorkflowRunResult(
            pid=pid,
            image=image,
            tool=tool,
            ok=bool(result.ok),
            status=process.status.value,
            call_id=result.call_id,
            tool_id=workflow_tool_id,
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
        self.process.add_handle_to_process_view(pid, handle)

    def add_handle_to_process_view(self, pid: str, handle: ObjectHandle) -> None:
        """Publish an object handle into a process view."""

        self._add_handle_to_process_view(pid, handle)

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
        return deepcopy(self.images[image_id])

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

    def activate_skill(
        self,
        pid: str,
        skill_id: str,
        *,
        expected_package_sha256: str | None = None,
    ) -> dict[str, Any]:
        return self.skills.activate_skill(
            pid,
            skill_id,
            actor=pid,
            require_capability=False,
            expected_package_sha256=expected_package_sha256,
        )

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

    def register_sink_trust(
        self,
        spec: SinkTrustRule | dict[str, Any],
        *,
        actor: str,
        replace: bool = False,
    ) -> SinkTrustSpec:
        """Host control-plane API; never projected into a model tool table."""

        return self.data_flow.register_sink_trust(
            spec,
            actor=actor,
            replace=replace,
            require_capability=True,
        )

    def unregister_sink_trust(self, pattern: str, *, actor: str) -> SinkTrustSpec:
        """Remove an active Host Sink trust rule under registry authority."""

        return self.data_flow.unregister_sink_trust(
            pattern,
            actor=actor,
            require_capability=True,
        )

    def inspect_sink_trust(self, pattern: str) -> SinkTrustSpec | None:
        return self.data_flow.inspect_sink_trust(pattern)

    def list_sink_trust(
        self,
        *,
        active_only: bool = True,
        generation: int | None = None,
    ) -> tuple[SinkTrustSpec, ...]:
        return self.data_flow.list_sink_trust(
            active_only=active_only,
            generation=generation,
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
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> Any:
        with self.lifecycle.admit():
            try:
                return self.image_boot.exec(
                    pid,
                    image,
                    execution_token=current_process_execution_token(),
                    args=args,
                    goal=goal,
                    preserve_memory=preserve_memory,
                    preserve_capabilities=preserve_capabilities,
                    llm_profile_id=llm_profile_id,
                    source_oids=source_oids,
                    source_labels=source_labels,
                    source_context=source_context,
                )
            except RuntimeRecoveryRequired as error:
                if self.image_boot.fence_recovery_required_signal(
                    error,
                    pid=pid,
                ):
                    assert self.lifecycle.state == "close_failed"
                raise

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
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> str:
        return self.launch.spawn_child(
            parent,
            goal,
            image=image,
            inherit_capabilities=inherit_capabilities,
            resource_budget=resource_budget,
            working_directory=working_directory,
            llm_profile_id=llm_profile_id,
            source_oids=source_oids,
            source_labels=source_labels,
            source_context=source_context,
        )

    def fork_child_process(
        self,
        parent: str,
        goal: dict[str, Any] | str | ObjectHandle,
        *,
        memory_view: MemoryView | MemoryViewSpec | None = None,
        capabilities: list[dict[str, Any]] | None = None,
        inherit_capabilities: list[dict[str, Any]] | None = None,
        resource_budget: Any | None = None,
        image: str | None = None,
        mode: ForkMode | str = ForkMode.RESTRICTED,
        working_directory: str | None = None,
        llm_profile_id: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> str:
        return self.launch.fork_child(
            parent,
            goal,
            memory_view=memory_view,
            capabilities=capabilities,
            inherit_capabilities=inherit_capabilities,
            resource_budget=resource_budget,
            image=image,
            mode=mode,
            working_directory=working_directory,
            llm_profile_id=llm_profile_id,
            source_oids=source_oids,
            source_labels=source_labels,
            source_context=source_context,
        )

    def set_process_working_directory(self, pid: str, path: str) -> Any:
        return self.launch.set_working_directory(pid, path)

    def _require_process_spawn_authority(self, pid: str) -> None:
        self.launch.require_spawn_authority(pid)

    def _require_process_image_boot_authority(self, pid: str, image_id: str) -> None:
        self.launch.require_image_boot_authority(pid, image_id)

    def _resolve_launch_llm_profile_id(self, image_id: str, explicit_profile_id: str | None) -> str:
        return self.launch.resolve_llm_profile_id(image_id, explicit_profile_id)

    def resolve_process_working_directory(self, pid: str, path: str) -> str:
        return self.launch.resolve_working_directory(pid, path)
