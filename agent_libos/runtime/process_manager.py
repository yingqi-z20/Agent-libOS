from __future__ import annotations

import builtins
import contextlib
import hashlib
import inspect
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, NoReturn

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.evidence.initial_goal_recovery import (
    INITIAL_GOAL_RECOVERY_KEY,
    InitialGoalRecoveryEnvelope,
    build_initial_goal_recovery_envelope,
    decode_initial_goal_recovery_envelope,
    initial_goal_object_identity,
    initial_goal_object_identity_sha256,
)
from agent_libos.memory.data_labels import metadata_from_labels, propagate_object_labels
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    ProcessError,
    ProcessTerminalCleanupRequired,
    ProcessWaitRequired,
    RuntimePublicationPending,
    ValidationError,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)
from agent_libos.models import (
    AgentProcess,
    CapabilityRight,
    ChildProcessWait,
    DataFlowContext,
    DataLabels,
    EventType,
    ExitedProcessOutcome,
    FailedProcessOutcome,
    ForkMode,
    HostResumeProcessWait,
    KilledProcessOutcome,
    MemoryView,
    MemoryViewSpec,
    MergePolicy,
    MergeResult,
    ObjectHandle,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectType,
    OperationOutcome,
    OperationState,
    Provenance,
    ProcessMessage,
    ProcessCursor,
    ProcessExecutionToken,
    ProcessResult,
    ProcessSignal,
    ProcessStatus,
    ProcessWaitState,
    PausedProcessWait,
    ResourceBudget,
    ResourceUsage,
    RuntimePublicationCursor,
    ViewMode,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.process_execution import (
    current_process_execution_token,
    trusted_process_control_mutation,
    trusted_process_execution_takeover,
    trusted_terminal_process_mutation,
)
from agent_libos.storage import UnitOfWork
from agent_libos.utils.serde import to_jsonable


class _SanitizedFinalizationInterruption(BaseException):
    """Public aggregate leaf that carries no original exception text."""


class ProcessManager:
    """Process lifecycle primitive."""

    TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
    TERMINAL_CLEANUP_PHASES = ("terminal_notify", "process_finalize")
    _TERMINAL_CLEANUP_LOCK_STRIPES = 64

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        memory: ObjectMemoryManager,
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus,
        require_recovery_lease: Callable[[], None],
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
        llm_profile_resolver: Callable[[str, str | None], str] | None = None,
        authority_manifests: Any | None = None,
        data_flow: Any | None = None,
        object_task_terminal_notifier: Callable[[str], None] | None = None,
        failed_launch_artifact_cleanup: Callable[[dict[str, Any]], None] | None = None,
        owner_instance_id: str = "runtime.local",
        recovery_required_callback: Callable[..., None] | None = None,
        recovery_terminalization_scope: (
            Callable[[str], AbstractContextManager[Any]] | None
        ) = None,
        transitions: ProcessTransitionService | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = unit_of_work.processes
        self._authority = unit_of_work.authority
        self._evidence = unit_of_work.evidence
        self.publications = unit_of_work.publications
        self.objects = unit_of_work.objects
        self.memory = memory
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.resources = resources
        self._llm_profile_resolver = llm_profile_resolver
        self.authority_manifests = authority_manifests
        self.data_flow = data_flow
        self._before_spawn_hooks: builtins.list[Callable[[str], None]] = []
        self._after_spawn_hooks: builtins.list[Callable[[str, str, str], None]] = []
        self._object_task_terminal_notifier = object_task_terminal_notifier
        self._host_managed_runner_checker: Callable[[str], bool] | None = None
        self._failed_launch_artifact_cleanup = failed_launch_artifact_cleanup
        self.owner_instance_id = str(owner_instance_id)
        self._terminal_cleanup_locks = tuple(
            threading.RLock()
            for _ in range(self._TERMINAL_CLEANUP_LOCK_STRIPES)
        )
        self._require_recovery_lease = require_recovery_lease
        self._recovery_required_callback = recovery_required_callback
        # Preserve the builder-bound lifecycle method as an immutable local
        # fallback. Tests and adapters may wrap the primary callback; an
        # interruption before that wrapper marks the lifecycle must not leave
        # mutation admission open.
        self._recovery_required_fallback = recovery_required_callback
        if (
            recovery_required_callback is not None
            and recovery_terminalization_scope is None
        ):
            raise RuntimeError(
                "recovery fencing requires a terminalization scope"
            )
        self._recovery_terminalization_scope = (
            recovery_terminalization_scope
            if recovery_terminalization_scope is not None
            else lambda _publication_id: contextlib.nullcontext()
        )
        self.transitions = transitions or ProcessTransitionService(self.store)

    def bind_host_managed_runner_checker(
        self,
        checker: Callable[[str], bool],
    ) -> None:
        """Identify published ObjectTask runners with Host-owned lifecycle."""

        self._host_managed_runner_checker = checker

    def add_after_spawn_hook(self, hook: Callable[..., None]) -> None:
        """Register a spawn hook, adapting the legacy two-argument shape."""

        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            self._after_spawn_hooks.append(hook)
            return
        try:
            signature.bind("pid", "image_id", "publication_id")
        except TypeError as three_argument_error:
            try:
                signature.bind("pid", "image_id")
            except TypeError:
                raise three_argument_error

            def legacy_hook(
                pid: str,
                image_id: str,
                _publication_id: str,
                *,
                _hook: Callable[..., None] = hook,
            ) -> None:
                _hook(pid, image_id)

            self._after_spawn_hooks.append(legacy_hook)
            return
        self._after_spawn_hooks.append(hook)

    def add_before_spawn_hook(self, hook: Callable[[str], None]) -> None:
        self._before_spawn_hooks.append(hook)

    def preflight_spawn(self, image: str | None = None) -> None:
        """Validate root image artifacts before an operation row is opened."""

        self._run_before_spawn_hooks(image or self.config.runtime.default_image_id)

    def preflight_fork(self, parent: str, image: str | None = None) -> None:
        """Validate inherited/explicit child image artifacts before evidence."""

        parent_process = self._require_active_parent(parent, "fork")
        self._run_before_spawn_hooks(image or parent_process.image_id)

    def preflight_spawn_child(self, parent: str, image: str | None = None) -> None:
        """Validate a fresh child image artifact before operation evidence."""

        parent_process = self._require_active_parent(parent, "spawn child from")
        self._run_before_spawn_hooks(image or parent_process.image_id)

    def preflight_exec(self, image: str) -> None:
        """Validate the replacement image artifact before operation evidence."""

        self._run_before_spawn_hooks(image)

    def bind_object_task_terminal_notifier(self, notifier: Callable[[str], None]) -> None:
        self._object_task_terminal_notifier = notifier

    def spawn(
        self,
        image: str | None = None,
        goal: dict[str, Any] | str | ObjectHandle | None = None,
        capabilities: builtins.list[dict[str, Any]] | None = None,
        resource_budget: ResourceBudget | None = None,
        working_directory: str | None = None,
        llm_profile_id: str | None = None,
        authority_manifest: Any | None = None,
        _claim_owner_id: str | None = None,
        _execution_token_out: builtins.list[ProcessExecutionToken] | None = None,
    ) -> str:
        if (_claim_owner_id is None) != (_execution_token_out is None):
            raise ValidationError(
                "claimed process spawn requires both an owner id and token receiver"
            )
        selected_image = image or self.config.runtime.default_image_id
        selected_root_budget = (
            self._coerce_resource_budget(resource_budget)
            if resource_budget is not None
            else None
        )
        now = utc_now()
        pid = new_id("pid")
        selected_llm_profile = self._resolve_root_llm_profile(selected_image, llm_profile_id)
        cwd = self._normalize_working_directory(working_directory or self.config.process.default_working_directory)
        publication_id, control_scope = self._begin_controlled_launch(
            pid, "spawn", selected_image, None
        )
        try:
            process = AgentProcess(
                pid=pid,
                parent_pid=None,
                image_id=selected_image,
                status=ProcessStatus.CREATED,
                goal_oid=None,
                memory_view=None,
                capabilities=[],
                loaded_skills={},
                tool_table={},
                event_cursor=None,
                checkpoint_head=None,
                resource_budget=selected_root_budget or self._default_resource_budget(),
                resource_usage=ResourceUsage(),
                created_at=now,
                updated_at=now,
                working_directory=cwd,
                llm_profile_id=selected_llm_profile,
            )
            self.store.insert_process(process)
            self._publication_phase(publication_id, "process_inserted")
            goal_handle = self._initialize_launch_goal(
                publication_id,
                pid,
                goal,
                image_id=selected_image,
                record_initial_goal_recovery=True,
            )
            # A process starts with a mutable goal-rooted view; later results append to it.
            view = self.memory.create_view(pid, [goal_handle], mode=ViewMode.MUTABLE)
            process = self._get(pid)
            process.goal_oid = goal_handle.oid
            process.memory_view = view
            process.updated_at = utc_now()
            process = self.store.patch_process(
                pid,
                {
                    "goal_oid": process.goal_oid,
                    "memory_view": process.memory_view,
                    "updated_at": process.updated_at,
                },
                expected_revision=process.revision,
            )
            self._compile_root_launch_authority(
                publication_id,
                pid=pid,
                image_id=selected_image,
                goal_handle=goal_handle,
                capabilities=capabilities or [],
                resource_budget=selected_root_budget,
                authority_manifest=authority_manifest,
            )
            self._run_after_spawn_hooks(pid, selected_image, publication_id)
            self._publication_phase(publication_id, "image_configured")
            with self.store.transaction():
                process = self._get(pid)
                process = self.transitions.transition(
                    pid,
                    ProcessStatus.RUNNABLE,
                    expected_revision=process.revision,
                    expected_status=ProcessStatus.CREATED,
                )
                execution_token: ProcessExecutionToken | None = None
                if _claim_owner_id is not None:
                    execution_token = self.store.claim_execution(
                        pid,
                        owner_id=_claim_owner_id,
                    )
                    if execution_token is None:
                        raise ProcessError(
                            f"new process could not be atomically claimed: {pid}"
                        )
                    process = self._get(pid)
                event = self.events.emit(
                    EventType.PROCESS_CREATED,
                    source="runtime",
                    target=pid,
                    payload={
                        "pid": pid,
                        "image": selected_image,
                        "goal_oid": goal_handle.oid,
                        "working_directory": cwd,
                        "llm_profile_id": selected_llm_profile,
                    },
                )
                audit = self.audit.record(
                    actor="runtime",
                    action="process.spawn",
                    target=f"process:{pid}",
                    output_refs=[goal_handle.oid],
                    decision={"image": selected_image, "working_directory": cwd, "llm_profile_id": selected_llm_profile},
                )
                self._commit_launch_publication(
                    publication_id,
                    receipt={
                        "phase": "committed",
                        "event_id": event.event_id,
                        "audit_id": audit.record_id,
                        "revision": process.revision,
                    },
                )
            if execution_token is not None:
                assert _execution_token_out is not None
                _execution_token_out.append(execution_token)
            return pid
        except BaseException as exc:
            return self._finish_failed_launch(publication_id, pid, exc)
        finally:
            control_scope.close()

    def spawn_claimed(
        self,
        *,
        owner_id: str,
        image: str | None = None,
        goal: dict[str, Any] | str | ObjectHandle | None = None,
        capabilities: builtins.list[dict[str, Any]] | None = None,
        resource_budget: ResourceBudget | None = None,
        working_directory: str | None = None,
        llm_profile_id: str | None = None,
        authority_manifest: Any | None = None,
    ) -> tuple[str, ProcessExecutionToken]:
        """Publish a root process already owned by one execution controller.

        The RUNNABLE transition, execution claim, launch evidence, and
        publication commit share one Store transaction, so a scheduler can
        never observe and claim the workflow process first.
        """

        selected_owner = str(owner_id).strip()
        if not selected_owner:
            raise ValidationError("claimed process spawn owner_id must be non-empty")
        tokens: builtins.list[ProcessExecutionToken] = []
        pid = self.spawn(
            image=image,
            goal=goal,
            capabilities=capabilities,
            resource_budget=resource_budget,
            working_directory=working_directory,
            llm_profile_id=llm_profile_id,
            authority_manifest=authority_manifest,
            _claim_owner_id=selected_owner,
            _execution_token_out=tokens,
        )
        if len(tokens) != 1:
            raise ProcessError(f"claimed process spawn returned no execution token: {pid}")
        return pid, tokens[0]

    def fork(
        self,
        parent: str,
        goal: dict[str, Any] | str | ObjectHandle,
        memory_view: MemoryView | MemoryViewSpec | None = None,
        capabilities: builtins.list[dict[str, Any]] | None = None,
        inherit_capabilities: builtins.list[dict[str, Any]] | None = None,
        resource_budget: ResourceBudget | dict[str, Any] | None = None,
        image: str | None = None,
        mode: ForkMode | str = ForkMode.RESTRICTED,
        working_directory: str | None = None,
        llm_profile_id: str | None = None,
        authority_manifest: Any | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        _launch_authority_reservations: Iterable[str] = (),
        _validated_working_directory: str | None = None,
    ) -> str:
        parent_proc = self._require_active_parent(parent, "fork")
        launch_authority_reservations = tuple(_launch_authority_reservations)
        fork_mode = ForkMode(mode)
        selected_image = image or parent_proc.image_id
        self._require_child_budget(parent_proc)
        inherit_specs = inherit_capabilities or []
        self._validate_inherit_capability_specs(parent, inherit_specs)
        selected_budget = self._select_child_resource_budget(
            parent_proc,
            resource_budget,
            authority_manifest=authority_manifest,
        )
        cwd = self._select_child_working_directory(
            working_directory,
            inherited=parent_proc.working_directory,
            validated=_validated_working_directory,
        )
        selected_llm_profile = self._resolve_child_llm_profile(parent_proc, llm_profile_id)
        now = utc_now()
        child_pid = new_id("pid")
        publication_id, control_scope = self._begin_controlled_launch(
            child_pid, "fork", selected_image, parent
        )
        try:
            self._reserve_child_budget(parent, child_pid, selected_budget)
            self._publication_phase(publication_id, "budget_reserved")
            child = AgentProcess(
                pid=child_pid,
                parent_pid=parent,
                image_id=selected_image,
                status=ProcessStatus.CREATED,
                goal_oid=None,
                memory_view=None,
                capabilities=[],
                loaded_skills=dict(parent_proc.loaded_skills),
                tool_table={},
                event_cursor=None,
                checkpoint_head=None,
                resource_budget=selected_budget,
                resource_usage=ResourceUsage(),
                created_at=now,
                updated_at=now,
                working_directory=cwd,
                llm_profile_id=selected_llm_profile,
            )
            self.store.insert_process(child)
            self._publication_phase(publication_id, "process_inserted")
            goal_handle = self._initialize_launch_goal(
                publication_id,
                child_pid,
                goal,
                parent_pid=parent,
                source_oids=self.flow_source_oids(parent, source_oids),
                source_labels=source_labels,
                source_context=source_context,
            )
            requested_specs = [*(capabilities or []), *inherit_specs]
            manifest = None
            if self.authority_manifests is not None:
                manifest = self.authority_manifests.prepare_launch(
                    pid=child_pid,
                    image_id=child.image_id,
                    goal_ref=goal_handle.oid,
                    supplied=authority_manifest,
                    authorized_capabilities=requested_specs,
                    resource_budget=selected_budget,
                    parent_pid=parent,
                    issued_by=f"process.fork:{parent}",
                )
                self._assert_manifest_resource_budget(manifest, selected_budget)
                self._assert_goal_data_flow(child_pid, goal_handle)
            source_view = parent_proc.memory_view or self.memory.create_view(parent, [], mode=ViewMode.READ_ONLY)
            if isinstance(memory_view, MemoryView):
                source_view = memory_view
                spec = MemoryViewSpec(mode=self._fork_mode_to_view_mode(fork_mode))
            else:
                spec = memory_view or MemoryViewSpec(mode=self._fork_mode_to_view_mode(fork_mode))
            # Forking exposes only parent-selected roots and rights granted into the child view.
            child = self._publish_fork_launch_view(
                publication_id,
                parent_pid=parent,
                child_pid=child_pid,
                source_view=source_view,
                spec=spec,
                goal_handle=goal_handle,
            )
            self._publish_child_launch_authority(
                publication_id,
                parent_pid=parent,
                child_pid=child_pid,
                manifest=manifest,
                requested_capabilities=capabilities or [],
                inherit_specs=inherit_specs,
                transition_kind="process.fork",
            )
            self._run_after_spawn_hooks(child_pid, child.image_id, publication_id)
            self._publication_phase(publication_id, "image_configured")
            with self.store.transaction():
                self._require_parent_launch_fence(
                    parent_proc,
                    action="fork",
                )
                self._commit_launch_authority_reservations(
                    launch_authority_reservations,
                    parent_pid=parent,
                    operation="process.fork",
                )
                self._charge_child_creation(parent)
                child = self._get(child_pid)
                child = self.transitions.transition(
                    child_pid,
                    ProcessStatus.RUNNABLE,
                    expected_revision=child.revision,
                    expected_status=ProcessStatus.CREATED,
                )
                event = self.events.emit(
                    EventType.PROCESS_FORKED,
                    source=parent,
                    target=child_pid,
                    payload={
                        "parent": parent,
                        "child": child_pid,
                        "mode": fork_mode.value,
                        "working_directory": cwd,
                        "llm_profile_id": selected_llm_profile,
                    },
                )
                audit = self.audit.record(
                    actor=parent,
                    action="process.fork",
                    target=f"process:{child_pid}",
                    input_refs=[parent_proc.goal_oid] if parent_proc.goal_oid else [],
                    output_refs=[goal_handle.oid],
                    decision={
                        "mode": fork_mode.value,
                        "image": child.image_id,
                        "working_directory": child.working_directory,
                        "llm_profile_id": selected_llm_profile,
                    },
                )
                self._commit_launch_publication(
                    publication_id,
                    receipt={
                        "phase": "committed",
                        "event_id": event.event_id,
                        "audit_id": audit.record_id,
                        "revision": child.revision,
                    },
                )
            return child_pid
        except BaseException as exc:
            return self._finish_failed_launch(publication_id, child_pid, exc)
        finally:
            control_scope.close()

    def spawn_child(
        self,
        parent: str,
        goal: dict[str, Any] | str | ObjectHandle,
        capabilities: builtins.list[dict[str, Any]] | None = None,
        inherit_capabilities: builtins.list[dict[str, Any]] | None = None,
        resource_budget: ResourceBudget | dict[str, Any] | None = None,
        image: str | None = None,
        working_directory: str | None = None,
        initial_status: ProcessStatus | str = ProcessStatus.RUNNABLE,
        initial_wait_state: ProcessWaitState | None = None,
        llm_profile_id: str | None = None,
        authority_manifest: Any | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        _launch_authority_reservations: Iterable[str] = (),
        _validated_working_directory: str | None = None,
    ) -> str:
        parent_proc = self._require_active_parent(parent, "spawn child from")
        launch_authority_reservations = tuple(_launch_authority_reservations)
        selected_image = image or parent_proc.image_id
        self._require_child_budget(parent_proc)
        inherit_specs = inherit_capabilities or []
        self._validate_inherit_capability_specs(parent, inherit_specs)
        selected_budget = self._select_child_resource_budget(
            parent_proc,
            resource_budget,
            authority_manifest=authority_manifest,
        )
        selected_initial_status = ProcessStatus(initial_status)
        if selected_initial_status in self.TERMINAL_STATUSES:
            raise ProcessError(f"spawn_child initial_status cannot be terminal: {selected_initial_status.value}")
        cwd = self._select_child_working_directory(
            working_directory,
            inherited=parent_proc.working_directory,
            validated=_validated_working_directory,
        )
        selected_llm_profile = self._resolve_child_llm_profile(parent_proc, llm_profile_id)
        now = utc_now()
        child_pid = new_id("pid")
        publication_id, control_scope = self._begin_controlled_launch(
            child_pid, "spawn_child", selected_image, parent
        )
        try:
            self._reserve_child_budget(parent, child_pid, selected_budget)
            self._publication_phase(publication_id, "budget_reserved")
            child = AgentProcess(
                pid=child_pid,
                parent_pid=parent,
                image_id=selected_image,
                status=ProcessStatus.CREATED,
                goal_oid=None,
                memory_view=None,
                capabilities=[],
                loaded_skills={},
                tool_table={},
                event_cursor=None,
                checkpoint_head=None,
                resource_budget=selected_budget,
                resource_usage=ResourceUsage(),
                created_at=now,
                updated_at=now,
                working_directory=cwd,
                llm_profile_id=selected_llm_profile,
            )
            self.store.insert_process(child)
            self._publication_phase(publication_id, "process_inserted")
            goal_handle = self._initialize_launch_goal(
                publication_id,
                child_pid,
                goal,
                parent_pid=parent,
                source_oids=self.flow_source_oids(parent, source_oids),
                source_labels=source_labels,
                source_context=source_context,
            )
            # Unlike fork(), spawn_child() starts with a fresh child-goal-only view.
            child = self._get(child_pid)
            child.memory_view = self.memory.create_view(child_pid, [goal_handle], mode=ViewMode.MUTABLE)
            child.goal_oid = goal_handle.oid
            child.updated_at = utc_now()
            child = self.store.patch_process(
                child_pid,
                {
                    "goal_oid": child.goal_oid,
                    "memory_view": child.memory_view,
                    "updated_at": child.updated_at,
                },
                expected_revision=child.revision,
            )
            requested_specs = [*(capabilities or []), *inherit_specs]
            manifest = None
            if self.authority_manifests is not None:
                manifest = self.authority_manifests.prepare_launch(
                    pid=child_pid,
                    image_id=child.image_id,
                    goal_ref=goal_handle.oid,
                    supplied=authority_manifest,
                    authorized_capabilities=requested_specs,
                    resource_budget=selected_budget,
                    parent_pid=parent,
                    issued_by=f"process.spawn_child:{parent}",
                )
                self._assert_manifest_resource_budget(manifest, selected_budget)
                self._assert_goal_data_flow(child_pid, goal_handle)
            self._publish_child_launch_authority(
                publication_id,
                parent_pid=parent,
                child_pid=child_pid,
                manifest=manifest,
                requested_capabilities=capabilities or [],
                inherit_specs=inherit_specs,
                transition_kind="process.spawn_child",
            )
            self._run_after_spawn_hooks(child_pid, child.image_id, publication_id)
            self._publication_phase(publication_id, "image_configured")
            with self.store.transaction():
                self._require_parent_launch_fence(
                    parent_proc,
                    action="spawn child from",
                )
                self._commit_launch_authority_reservations(
                    launch_authority_reservations,
                    parent_pid=parent,
                    operation="process.spawn_child",
                )
                self._charge_child_creation(parent)
                child = self._get(child_pid)
                child = self.transitions.transition(
                    child_pid,
                    selected_initial_status,
                    expected_revision=child.revision,
                    expected_status=ProcessStatus.CREATED,
                    wait_state=initial_wait_state,
                )
                event = self.events.emit(
                    EventType.PROCESS_CREATED,
                    source=parent,
                    target=child_pid,
                    payload={
                        "parent": parent,
                        "child": child_pid,
                        "image": child.image_id,
                        "goal_oid": goal_handle.oid,
                        "working_directory": child.working_directory,
                        "status": child.status.value,
                        "llm_profile_id": selected_llm_profile,
                    },
                )
                audit = self.audit.record(
                    actor=parent,
                    action="process.spawn_child",
                    target=f"process:{child_pid}",
                    output_refs=[goal_handle.oid],
                    decision={
                        "image": child.image_id,
                        "working_directory": child.working_directory,
                        "status": child.status.value,
                        "llm_profile_id": selected_llm_profile,
                    },
                )
                self._commit_launch_publication(
                    publication_id,
                    receipt={
                        "phase": "committed",
                        "event_id": event.event_id,
                        "audit_id": audit.record_id,
                        "revision": child.revision,
                    },
                )
            return child_pid
        except BaseException as exc:
            return self._finish_failed_launch(publication_id, child_pid, exc)
        finally:
            control_scope.close()

    def apply_exec_state(
        self,
        pid: str,
        image: str,
        args: dict[str, Any] | None = None,
        goal: dict[str, Any] | str | ObjectHandle | None = None,
        preserve_memory: bool = True,
        preserve_capabilities: bool = False,
        llm_profile_id: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        _record_evidence: bool = True,
        _capability_rollback_token: str | None = None,
    ) -> None:
        """Apply the SQL/Object-Memory phase of an ImageBoot publication."""

        with self.store.transaction(include_object_payloads=True):
            self._exec_uncommitted(
                pid,
                image,
                args=args,
                goal=goal,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
                record_evidence=_record_evidence,
                capability_rollback_token=_capability_rollback_token,
            )

    def _exec_uncommitted(
        self,
        pid: str,
        image: str,
        args: dict[str, Any] | None = None,
        goal: dict[str, Any] | str | ObjectHandle | None = None,
        preserve_memory: bool = True,
        preserve_capabilities: bool = False,
        llm_profile_id: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        record_evidence: bool = True,
        capability_rollback_token: str | None = None,
    ) -> None:
        process = self._get(pid)
        if process.status in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot exec terminated process: {pid}")
        old_image = process.image_id
        goal_handle = (
            self._ensure_goal(
                pid,
                goal,
                source_oids=self.flow_source_oids(pid, source_oids),
                source_labels=source_labels,
                source_context=source_context,
            )
            if goal is not None
            else None
        )
        process = self._get(pid)
        if not preserve_capabilities:
            kept: builtins.list[str] = []
            process_namespace_resource = f"object_namespace:{self.memory.process_namespace(pid)}"
            for cap in self.capabilities.capabilities_for(pid):
                if cap.resource.startswith("object:") or cap.resource == process_namespace_resource:
                    kept.append(cap.cap_id)
                elif capability_rollback_token is not None:
                    self.capabilities.stage_exec_revocation(
                        cap.cap_id,
                        rollback_token=capability_rollback_token,
                    )
                else:
                    self.capabilities.revoke(
                        cap.cap_id,
                        revoked_by="process.exec",
                        reason="exec capability shrink",
                        require_authority=False,
                    )
            process = self._get(pid)
            process.capabilities = kept
        if goal_handle is not None:
            if preserve_memory:
                process = self._add_handle_to_process_view(process, goal_handle)
            else:
                process.memory_view = self.memory.create_view(pid, [goal_handle], mode=ViewMode.MUTABLE)
            process.goal_oid = goal_handle.oid
        elif not preserve_memory:
            process.memory_view = None
        process.image_id = image
        if llm_profile_id is not None:
            process.llm_profile_id = self._normalize_llm_profile_id(llm_profile_id)
        process.loaded_skills = {}
        process.updated_at = utc_now()
        process = self.store.patch_process(
            pid,
            {
                "image_id": process.image_id,
                "goal_oid": process.goal_oid,
                "memory_view": process.memory_view,
                "capabilities": process.capabilities,
                "loaded_skills": process.loaded_skills,
                "llm_profile_id": process.llm_profile_id,
                "updated_at": process.updated_at,
            },
            expected_revision=process.revision,
        )
        if record_evidence:
            self.record_exec_evidence(
                pid,
                old_image=old_image,
                args=args,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                new_goal_oid=goal_handle.oid if goal_handle is not None else None,
            )

    def record_exec_evidence(
        self,
        pid: str,
        *,
        old_image: str,
        args: dict[str, Any] | None,
        preserve_memory: bool,
        preserve_capabilities: bool,
        new_goal_oid: str | None,
    ) -> tuple[Any, Any]:
        """Publish exec event/audit after all image boot phases have succeeded."""

        process = self._get(pid)
        goal_oid = new_goal_oid or process.goal_oid
        event = self.events.emit(
            EventType.PROCESS_EXEC,
            source=pid,
            target=pid,
            payload={
                "old_image": old_image,
                "new_image": process.image_id,
                "preserve_memory": preserve_memory,
                "preserve_capabilities": preserve_capabilities,
                "goal_oid": goal_oid,
                "working_directory": process.working_directory,
                "llm_profile_id": process.llm_profile_id,
            },
        )
        audit = self.audit.record(
            actor=pid,
            action="process.exec",
            target=f"process:{pid}",
            output_refs=[new_goal_oid] if new_goal_oid is not None else [],
            decision={
                "old_image": old_image,
                "new_image": process.image_id,
                "args": args or {},
                "goal_oid": goal_oid,
                "preserve_memory": preserve_memory,
                "preserve_capabilities": preserve_capabilities,
                "working_directory": process.working_directory,
                "llm_profile_id": process.llm_profile_id,
            },
        )
        return event, audit

    def wait(self, pid: str, child: str, timeout: float | None = None) -> ProcessResult:
        with self.store.locked():
            parent = self._require_active_parent(pid, "wait for child from")
            child_proc = self._require_child(parent.pid, child)
            if child_proc.status not in self.TERMINAL_STATUSES:
                if timeout == 0:
                    raise TimeoutError(f"child still running: {child}")
                parent = self.transitions.transition(
                    pid,
                    ProcessStatus.WAITING_EVENT,
                    expected_revision=parent.revision,
                    expected_state_generation=parent.state_generation,
                    wait_state=ChildProcessWait(child_pid=child),
                )
                wait_token = self.transitions.wait_token(parent)
                child_proc = self._require_child(parent.pid, child)
                if child_proc.status not in self.TERMINAL_STATUSES:
                    raise ProcessWaitRequired(child_pid=child, message=f"{pid} is waiting for child process {child}")
                parent = self.transitions.wake(wait_token, control=False)
        result_handle = None
        # Keep the receiver-domain check, source ownership transfer, and
        # handle publication under one lock domain so a mutable result cannot
        # change labels between validation and delivery.
        with self.memory.ownership_locked(), self.store.transaction(
            include_object_payloads=True
        ):
            parent = self._require_active_parent(pid, "receive child result in")
            child_proc = self._require_child(parent.pid, child)
            outcome = child_proc.outcome
            if isinstance(outcome, (ExitedProcessOutcome, FailedProcessOutcome)) and outcome.result_oid:
                oid = outcome.result_oid
                self._assert_object_data_flow(pid, oid)
                self.memory.preserve_process_owned(child, {oid})
                result_handle = self.capabilities.handle_for_object(
                    pid,
                    oid,
                    {"read", "materialize", "link", "diff"},
                    issued_by=f"process.wait:{child}",
                )
                parent = self._add_handle_to_process_view(parent, result_handle)
            if (
                parent.status == ProcessStatus.WAITING_EVENT
                and isinstance(parent.wait_state, ChildProcessWait)
                and parent.wait_state.child_pid == child
            ):
                parent = self.transitions.wake(
                    self.transitions.wait_token(parent),
                    control=False,
                )
            self.audit.record(
                actor=pid,
                action="process.wait",
                target=f"process:{child}",
                output_refs=[result_handle.oid] if result_handle else [],
                decision={"child_status": child_proc.status.value},
            )
            return ProcessResult(
                pid=child,
                status=child_proc.status,
                result=result_handle,
                message=child_proc.status_message,
                wait_state=child_proc.wait_state,
                outcome=child_proc.outcome,
                state_generation=child_proc.state_generation,
            )

    def set_working_directory(self, pid: str, path: str) -> AgentProcess:
        return self._persist_working_directory(
            pid,
            self._normalize_working_directory(path),
        )

    def set_validated_working_directory(self, pid: str, path: str) -> AgentProcess:
        """Persist one provider-validated canonical workspace identity unchanged.

        This is the controlled component boundary used by ``ProcessLaunchService``
        after the filesystem provider has authorized and canonicalized ``path``.
        It still validates the identity's shape before changing process state.
        """

        return self._persist_working_directory(
            pid,
            self._validated_working_directory_identity(path),
        )

    def _persist_working_directory(self, pid: str, path: str) -> AgentProcess:
        process = self._get(pid)
        if process.status in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot change working directory for terminated process: {pid}")
        process.working_directory = path
        process.updated_at = utc_now()
        process = self.store.patch_process(
            pid,
            {
                "working_directory": process.working_directory,
                "updated_at": process.updated_at,
            },
            expected_revision=process.revision,
        )
        self.audit.record(
            actor=pid,
            action="process.chdir",
            target=f"process:{pid}",
            decision={"working_directory": process.working_directory},
        )
        return process

    def working_directory(self, pid: str) -> str:
        return self._get(pid).working_directory

    def list_children(self, pid: str, include_terminal: bool = True) -> builtins.list[AgentProcess]:
        self._get(pid)
        children = self.store.list_child_processes(pid)
        if not include_terminal:
            children = [process for process in children if process.status not in self.TERMINAL_STATUSES]
        self.audit.record(
            actor=pid,
            action="process.list_children",
            target=f"process:{pid}",
            decision={"count": len(children), "include_terminal": include_terminal},
        )
        return children

    def signal_child(
        self,
        pid: str,
        child: str,
        signal: ProcessSignal | str,
        reason: str | None = None,
        *,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> AgentProcess:
        child_proc = self._require_child(pid, child)
        sig = ProcessSignal(signal)
        if (
            child_proc.status in self.TERMINAL_STATUSES
            and sig in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}
        ):
            cleanup = self.terminal_cleanup_state(child)
            if cleanup["state"] != "completed":
                self.retry_terminal_cleanup(
                    child,
                    _actor=pid,
                    _legacy_audit_action="process.signal_finalize_failed",
                    _context={"signal": sig.value, "retry": True},
                )
                return self._get(child)
        self._require_signal_applicable(child_proc, sig)
        payload: dict[str, Any] = {}
        reason_handle: ObjectHandle | None = None
        takeover_scope = self._signal_takeover_scope(
            child_proc,
            sig,
            reason_text=reason,
            reason_action="process.signal_child.reason" if reason is not None else None,
            require_host_resume=False,
        )
        # The labeled reason carrier is part of the signal transition, not a
        # preparatory side effect.  Keep its Object payload, capability, child
        # MemoryView root, process state, event, and audit in one rollback scope.
        with self.memory.ownership_locked(), self.store.transaction(
            include_object_payloads=True
        ):
            with takeover_scope, trusted_process_control_mutation(
                child,
                allowed_statuses={child_proc.status},
                reason="process.signal_child changes an authorized child",
            ):
                if reason is not None:
                    reason_handle = self._create_flow_text_carrier(
                        recipient_pid=child,
                        text=reason,
                        payload_field="reason",
                        object_type=ObjectType.MESSAGE,
                        title="Process signal reason",
                        tags=["process_signal", "reason"],
                        created_from_action="process.signal_child.reason",
                        source_pid=pid,
                        source_oids=source_oids,
                        source_labels=source_labels,
                        source_context=source_context,
                    )
                    payload["reason_oid"] = reason_handle.oid
                updated = self._apply_signal(
                    child_proc,
                    sig,
                    payload=payload,
                    allow_host_resume=False,
                )
            self._record_signal_transition(
                updated,
                sig,
                payload=payload,
                actor=pid,
                action="process.signal_child",
            )
        if updated.status in self.TERMINAL_STATUSES:
            self._complete_terminal_signal(updated, actor=pid, signal=sig)
        return updated

    def merge_child_memory(
        self,
        pid: str,
        child: str,
        policy: MergePolicy | None = None,
    ) -> MergeResult:
        selected_policy = policy or MergePolicy()
        # Ownership is the outer lifecycle lock everywhere in Object Memory.
        # The SQL transaction then makes handle grants, parent-view roots,
        # ownership transfer/release, child-view consumption, and audits one
        # rollback unit.
        with self.memory.ownership_locked(), self.store.transaction(
            include_object_payloads=True
        ):
            self._require_active_parent(pid, "merge child memory for")
            child_proc = self._require_child(pid, child)
            if child_proc.status not in self.TERMINAL_STATUSES:
                raise ProcessError(f"cannot merge running child process: {child}")

            process_owned_oids = set(
                self.objects.list_object_oids_owned_by(
                    ObjectOwnerKind.PROCESS,
                    child,
                )
            )
            result_owned_oids = set(
                self.objects.list_object_oids_owned_by(
                    ObjectOwnerKind.PROCESS_RESULT,
                    child,
                )
            )
            child_roots = (
                list(child_proc.memory_view.roots)
                if child_proc.memory_view is not None
                else []
            )
            if not child_roots and not process_owned_oids and not result_owned_oids:
                return MergeResult(merged_oids=[], skipped_oids=[])

            child_view = child_proc.memory_view
            if child_view is None:
                child_view = MemoryView(
                    view_id=new_id("view"),
                    owner_pid=child,
                    roots=[],
                    filters=[],
                    rights_policy="merge_terminal_child",
                    created_from=None,
                    mode=ViewMode.READ_ONLY,
                )
            candidate_oids = (
                {handle.oid for handle in child_view.roots}
                if selected_policy.include_updated
                else set()
            )
            if selected_policy.include_child_created:
                candidate_oids.update(process_owned_oids)
            for oid in sorted(candidate_oids):
                if self.objects.get_object(oid) is not None:
                    self._assert_object_data_flow(pid, oid)
            result = self.memory.merge_view(pid, child_view, policy=selected_policy)
            parent = self._get(pid)
            for handle in result.merged_handles:
                self._add_handle_to_process_view(parent, handle)
            adopted = self.memory.adopt_process_owned(child, pid, result.merged_oids)
            released = self.memory.release_process_owned(child)
            retained_child_owned = sorted(
                {
                    *self.objects.list_object_oids_owned_by(
                        ObjectOwnerKind.PROCESS,
                        child,
                    ),
                    *self.objects.list_object_oids_owned_by(
                        ObjectOwnerKind.PROCESS_RESULT,
                        child,
                    ),
                }
            )
            # release_process_owned has no preserve set on merge and raises on
            # any failed deletion. Successful leftovers are therefore exactly
            # the ObjectTask-pinned tail that must remain child-owned.
            result.adopted_oids = sorted(adopted)
            result.released_oids = sorted(released)
            result.retained_child_owned_oids = retained_child_owned
            result.pinned_child_owned_oids = list(retained_child_owned)

            # Clearing the terminal child's roots is the durable idempotency
            # marker. A replay sees no roots or child-owned objects and returns
            # before issuing handles or audit rows.
            if child_proc.memory_view is not None and child_proc.memory_view.roots:
                consumed_view = deepcopy(child_proc.memory_view)
                consumed_view.roots = []
                with trusted_terminal_process_mutation(
                    child,
                    expected_revision=child_proc.revision,
                    expected_generation=child_proc.execution_generation,
                    allowed_statuses={child_proc.status},
                    execution_token=current_process_execution_token(),
                    reason="consume terminal child memory after merge",
                ):
                    self.store.patch_process_control(
                        child,
                        {
                            "memory_view": consumed_view,
                            "updated_at": utc_now(),
                        },
                        expected_revision=child_proc.revision,
                        allowed_statuses={child_proc.status},
                        reason="consume terminal child memory after merge",
                    )
            self.audit.record(
                actor=pid,
                action="process.merge_child_memory",
                target=f"process:{child}",
                output_refs=result.merged_oids,
                decision={
                    "merged": len(result.merged_oids),
                    "skipped": result.skipped_oids,
                    "adopted": adopted,
                    "released_child_owned": released,
                    "retained_child_owned": retained_child_owned,
                    "pinned_child_owned": retained_child_owned,
                    "child_view_consumed": bool(child_roots),
                },
            )
            return result

    def signal(
        self,
        target: str,
        signal: ProcessSignal | str,
        payload: dict[str, Any] | None = None,
        *,
        _require_host_resume: bool = False,
    ) -> None:
        proc = self._get(target)
        sig = ProcessSignal(signal)
        if (
            proc.status in self.TERMINAL_STATUSES
            and sig in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}
        ):
            cleanup = self.terminal_cleanup_state(target)
            if cleanup["state"] != "completed":
                self.retry_terminal_cleanup(
                    target,
                    _actor="runtime",
                    _legacy_audit_action="process.signal_finalize_failed",
                    _context={"signal": sig.value, "retry": True},
                )
                return
        selected_payload = dict(payload or {})
        reason = selected_payload.pop("reason", None)
        self._require_signal_applicable(proc, sig)
        reason_handle: ObjectHandle | None = None
        reason_text = str(reason) if reason is not None else None
        takeover_scope = self._signal_takeover_scope(
            proc,
            sig,
            reason_text=reason_text,
            reason_action="process.signal.reason" if reason is not None else None,
            require_host_resume=_require_host_resume,
        )
        with self.memory.ownership_locked(), self.store.transaction(
            include_object_payloads=True
        ):
            with takeover_scope, trusted_process_control_mutation(
                target,
                allowed_statuses={proc.status},
                reason="process.signal applies trusted Host control",
            ):
                if reason_text is not None:
                    reason_handle = self._create_flow_text_carrier(
                        recipient_pid=target,
                        text=reason_text,
                        payload_field="reason",
                        object_type=ObjectType.MESSAGE,
                        title="Process signal reason",
                        tags=["process_signal", "reason"],
                        created_from_action="process.signal.reason",
                        source_pid=target,
                    )
                    selected_payload["reason_oid"] = reason_handle.oid
                updated = self._apply_signal(
                    proc,
                    sig,
                    payload=selected_payload,
                    allow_host_resume=True,
                    require_host_resume=_require_host_resume,
                )
            self._record_signal_transition(
                updated,
                sig,
                payload=selected_payload,
                actor="runtime",
                action="process.signal",
            )
        if updated.status in self.TERMINAL_STATUSES:
            self._complete_terminal_signal(updated, actor="runtime", signal=sig)

    @staticmethod
    def _signal_takeover_scope(
        process: AgentProcess,
        signal: ProcessSignal,
        *,
        reason_text: str | None,
        reason_action: str | None,
        require_host_resume: bool,
    ) -> AbstractContextManager[Any]:
        if process.status != ProcessStatus.RUNNING or signal not in {
            ProcessSignal.PAUSE,
            ProcessSignal.CANCEL,
            ProcessSignal.TERMINATE,
        }:
            return contextlib.nullcontext()
        if (
            process.execution_owner_id is None
            and process.execution_lease_id is None
        ):
            return contextlib.nullcontext()
        if process.execution_owner_id is None or process.execution_lease_id is None:
            raise ProcessError(
                f"running process has an incomplete execution lease: {process.pid}"
            )
        intended_status = (
            ProcessStatus.PAUSED
            if signal == ProcessSignal.PAUSE
            else ProcessStatus.KILLED
        )
        wait_kind = None
        if signal == ProcessSignal.PAUSE:
            wait_kind = "host_resume" if require_host_resume else "paused"
        return trusted_process_execution_takeover(
            process.pid,
            source_revision=process.revision,
            source_state_generation=process.state_generation,
            source_execution_token=ProcessExecutionToken(
                pid=process.pid,
                generation=process.execution_generation,
                owner_id=process.execution_owner_id,
                lease_id=process.execution_lease_id,
            ),
            intended_status=intended_status,
            reason="trusted process signal takes over an execution lease",
            nonce=new_id("process_takeover"),
            reason_text=reason_text,
            reason_action=reason_action,
            wait_kind=wait_kind,
            outcome_code=(
                signal.value if intended_status == ProcessStatus.KILLED else None
            ),
        )

    def _require_signal_applicable(self, proc: AgentProcess, sig: ProcessSignal) -> None:
        if proc.status in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot signal terminal process: {proc.pid} status={proc.status.value}")
        if sig == ProcessSignal.INTERRUPT:
            raise ProcessError(
                "process interrupt signals are not state transitions; "
                "send a durable interrupt process message instead"
            )
        self.transitions.require_signal_preserves_condition_wait(proc, sig)

    def _create_flow_text_carrier(
        self,
        *,
        recipient_pid: str,
        text: str,
        payload_field: str,
        object_type: ObjectType,
        title: str,
        tags: list[str],
        created_from_action: str,
        source_pid: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        attach_to_view: bool = True,
    ) -> ObjectHandle:
        selected_sources = (
            self.flow_source_oids(source_pid, source_oids)
            if source_pid is not None
            else self._normalize_source_oids(source_oids)
        )
        selected_context = source_context
        if selected_context is None and self.data_flow is not None:
            selected_context = self.data_flow.current_context()
        metadata = self.flow_metadata(
            selected_sources,
            source_labels,
            selected_context,
            base=ObjectMetadata(title=title, tags=tags),
        )
        # A process deriving its own result and a Host management-plane
        # injection are not cross-process ingress.  Enforce the recipient
        # identity domain only for an actual process-to-process handoff.
        if (
            self.authority_manifests is not None
            and source_pid is not None
            and source_pid != recipient_pid
        ):
            self.authority_manifests.assert_data_flow_labels(
                recipient_pid,
                DataLabels.from_object_metadata(metadata),
            )
        handle = self.memory.create_object(
            pid=recipient_pid,
            object_type=object_type,
            payload={payload_field: text},
            metadata=metadata,
            immutable=True,
            provenance=Provenance(
                created_from_action=created_from_action,
                parent_oids=selected_sources,
            ),
        )
        if attach_to_view:
            self._add_handle_to_process_view(self._get(recipient_pid), handle)
        return handle

    def _apply_signal(
        self,
        proc: AgentProcess,
        sig: ProcessSignal,
        payload: dict[str, Any],
        *,
        allow_host_resume: bool,
        require_host_resume: bool = False,
    ) -> AgentProcess:
        # Persist the process state, reservation release, parent wakeup, event,
        # and audit as one lifecycle transition.  Object/Host cleanup and
        # terminal callbacks remain post-commit because their external effects
        # cannot be rolled back with the SQL transaction.
        with self.store.transaction():
            proc = self._get(proc.pid)
            self._require_signal_applicable(proc, sig)
            host_resume_required = isinstance(proc.wait_state, HostResumeProcessWait)
            if sig == ProcessSignal.RESUME and host_resume_required and not allow_host_resume:
                raise ProcessError(f"process requires explicit Host resume: {proc.pid}")
            wait_state: ProcessWaitState | None = None
            outcome = None
            if sig == ProcessSignal.PAUSE:
                selected_status = ProcessStatus.PAUSED
            elif sig == ProcessSignal.RESUME:
                if proc.status in {ProcessStatus.PAUSED, ProcessStatus.SUSPENDED}:
                    selected_status = ProcessStatus.RUNNABLE
                else:
                    selected_status = proc.status
            elif sig in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
                selected_status = ProcessStatus.KILLED
            else:  # pragma: no cover - interrupt is rejected by _require_signal_applicable
                raise ProcessError(f"unsupported process signal: {sig.value}")
            reason_oid = str(payload.get("reason_oid") or "").strip()
            if sig == ProcessSignal.PAUSE and require_host_resume:
                wait_state = HostResumeProcessWait(reason_oid=reason_oid)
            elif sig == ProcessSignal.PAUSE and host_resume_required:
                # Ordinary child signaling must not erase an existing Host-only
                # resume gate by pausing the process again with a new reason.
                wait_state = proc.wait_state
            elif sig == ProcessSignal.PAUSE:
                wait_state = PausedProcessWait(reason_oid=reason_oid or None)
            elif sig in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
                outcome = KilledProcessOutcome(
                    reason_oid=reason_oid or None,
                    code=sig.value,
                )
            proc = self.transitions.transition(
                proc.pid,
                selected_status,
                expected_revision=proc.revision,
                expected_status=proc.status,
                expected_state_generation=proc.state_generation,
                wait_state=wait_state,
                outcome=outcome,
                control=True,
                allowed_statuses={proc.status},
                reason="trusted process signal state transition",
            )
        return proc

    def _record_signal_transition(
        self,
        process: AgentProcess,
        signal: ProcessSignal,
        *,
        payload: dict[str, Any],
        actor: str,
        action: str,
    ) -> None:
        if process.status in self.TERMINAL_STATUSES:
            self._release_child_budget(process.pid)
        self.events.emit(
            EventType.PROCESS_SIGNAL,
            source=actor,
            target=process.pid,
            payload={"signal": signal.value, "payload": payload or {}},
        )
        self.audit.record(
            actor=actor,
            action=action,
            target=f"process:{process.pid}",
            decision={"signal": signal.value, "payload": payload or {}},
        )
        if process.status in self.TERMINAL_STATUSES:
            self._wake_parent_waiting_on_child(process)

    def _complete_terminal_signal(self, process: AgentProcess, *, actor: str, signal: ProcessSignal) -> None:
        self._complete_terminal_cleanup(
            process,
            actor=actor,
            audit_action="process.signal_finalize_failed",
            context={"signal": signal.value},
        )

    def _complete_terminal_cleanup(
        self,
        process: AgentProcess,
        *,
        actor: str,
        audit_action: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return self.retry_terminal_cleanup(
            process.pid,
            _actor=actor,
            _legacy_audit_action=audit_action,
            _context=context,
        )

    def terminal_cleanup_state(self, pid: str) -> dict[str, Any]:
        process = self._get(pid)
        if process.status not in self.TERMINAL_STATUSES:
            raise ProcessError(
                f"process is not terminal and has no cleanup outcome: {pid}"
            )
        cleanup = self.store.get_process_terminal_cleanup(pid)
        if cleanup is None:
            raise ProcessError(f"terminal cleanup intent is missing: {pid}")
        return cleanup

    def retry_terminal_cleanup(
        self,
        pid: str,
        *,
        _actor: str = "runtime.process_terminal",
        _legacy_audit_action: str | None = None,
        _context: dict[str, Any] | None = None,
        _recover_stale: bool = False,
    ) -> dict[str, Any]:
        """Finish the durable, phase-fenced cleanup for one terminal process."""

        process = self._get(pid)
        if process.status not in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot clean up non-terminal process: {pid}")
        with self._terminal_cleanup_lock(pid):
            cleanup = self.terminal_cleanup_state(pid)
            if cleanup["state"] == "completed":
                return cleanup

            expected_lease_id: str | None = None
            if cleanup["state"] == "running":
                lease_owner = cleanup.get("owner_id")
                if not _recover_stale and lease_owner != self.owner_instance_id:
                    raise ProcessError(
                        "process terminal cleanup is already running: "
                        f"{pid} owner={lease_owner}"
                    )
                expected_lease_id = str(cleanup["lease_id"])

            lease_id = new_id("process_cleanup_lease")
            claimed = self.store.claim_process_terminal_cleanup(
                pid,
                owner_id=self.owner_instance_id,
                lease_id=lease_id,
                expected_lease_id=expected_lease_id,
            )
            if claimed is None:
                latest = self.terminal_cleanup_state(pid)
                if latest["state"] == "completed":
                    return latest
                raise ProcessError(
                    "process terminal cleanup lease was not acquired: "
                    f"{pid} state={latest['state']}"
                )

            completed_phases = set(claimed["completed_phases"])
            phase_failures: list[tuple[str, BaseException]] = []
            for phase in self.TERMINAL_CLEANUP_PHASES:
                if phase in completed_phases:
                    continue
                try:
                    process = self._get(pid)
                    if phase == "terminal_notify":
                        self._notify_object_task_process_terminal(pid)
                    else:
                        self._finalize_terminal_process(
                            process,
                            preserve_oids=self._terminal_preserve_oids(process),
                        )
                    marked = self.store.complete_process_terminal_cleanup_phase(
                        pid,
                        lease_id=lease_id,
                        phase=phase,
                    )
                    if marked is None:
                        raise ProcessError(
                            "process terminal cleanup lost its phase lease: "
                            f"{pid} phase={phase}"
                        )
                    claimed = marked
                    completed_phases.add(phase)
                except BaseException as exc:
                    # Cleanup phases are independent and idempotent.  Keep
                    # attempting later phases so a broken notifier cannot
                    # strand memory/capability finalizers (or vice versa).
                    phase_failures.append((phase, exc))

            if phase_failures:
                self._raise_terminal_cleanup_failures(
                    pid,
                    lease_id=lease_id,
                    claimed=claimed,
                    phase_failures=phase_failures,
                    actor=_actor,
                    terminal_status=process.status.value,
                    legacy_audit_action=_legacy_audit_action,
                    context=_context,
                )

            finished = self.store.finish_process_terminal_cleanup(
                pid,
                lease_id=lease_id,
            )
            if finished is None:
                raise ProcessError(
                    f"process terminal cleanup could not commit completion: {pid}"
                )
            try:
                self.audit.record(
                    actor=_actor,
                    action="process.terminal_cleanup_completed",
                    target=f"process:{pid}",
                    decision={
                        "terminal_status": process.status.value,
                        "attempt": finished["attempt_count"],
                        "completed_phases": list(finished["completed_phases"]),
                    },
                )
            except BaseException:
                # The cleanup row is the authoritative observation.  A failed
                # diagnostic sink cannot reopen already-completed finalizers.
                pass
            return finished

    def _raise_terminal_cleanup_failures(
        self,
        pid: str,
        *,
        lease_id: str,
        claimed: dict[str, Any],
        phase_failures: list[tuple[str, BaseException]],
        actor: str,
        terminal_status: str,
        legacy_audit_action: str | None,
        context: dict[str, Any] | None,
    ) -> NoReturn:
        failed_phase, first_error = phase_failures[0]
        errors = [
            {"phase": phase, **internal_exception_observation(exc)}
            for phase, exc in phase_failures
        ]
        durable_error: dict[str, Any] = {
            **internal_exception_observation(first_error),
            "errors": errors,
        }
        failed = None
        persistence_error: dict[str, Any] | None = None
        try:
            failed = self.store.fail_process_terminal_cleanup(
                pid,
                lease_id=lease_id,
                phase=failed_phase,
                error=durable_error,
            )
        except BaseException as persistence_exc:
            persistence_error = internal_exception_observation(persistence_exc)
        observed = failed or claimed
        self._record_terminal_cleanup_failure(
            pid,
            actor=actor,
            attempt=int(observed["attempt_count"]),
            terminal_status=terminal_status,
            errors=errors,
            persistence_error=persistence_error,
            legacy_audit_action=legacy_audit_action,
            context=context,
        )
        # Do not chain ordinary provider errors: durable evidence retains only
        # their sanitized observation. Control-flow BaseExceptions are retained
        # as leaves after every independent phase and diagnostic has run.
        required = ProcessTerminalCleanupRequired(
            pid=pid,
            phase=failed_phase,
            attempt=int(observed["attempt_count"]),
        )
        interruptions = [
            exc for _phase, exc in phase_failures if not isinstance(exc, Exception)
        ]
        if interruptions:
            raise BaseExceptionGroup(
                "process terminal cleanup interrupted after all phases were attempted",
                [required, *interruptions],
            )
        raise required

    def recover_terminal_cleanups(self) -> dict[str, Any]:
        """Reclaim incomplete cleanup leases through bounded keyset pages."""

        self._require_recovery_lease()
        page_size = (
            self.config.runtime.process_terminal_cleanup_recovery_page_size
        )
        after: ProcessCursor | None = None
        pending_total = 0
        recovered_total = 0
        failure_total = 0
        recovered_sample: list[str] = []
        failure_sample: list[dict[str, Any]] = []
        interruptions: list[BaseException] = []
        while True:
            page = self.store.list_process_terminal_cleanups(
                after=after,
                limit=page_size,
            )
            if not page:
                break
            for cleanup in page:
                pending_total += 1
                pid = str(cleanup["pid"])
                try:
                    completed = self.retry_terminal_cleanup(
                        pid,
                        _actor="runtime.recovery",
                        _recover_stale=True,
                    )
                    if completed["state"] == "completed":
                        recovered_total += 1
                        if len(recovered_sample) < page_size:
                            recovered_sample.append(pid)
                except BaseException as exc:
                    failure_total += 1
                    if not isinstance(exc, Exception):
                        interruptions.append(exc)
                    if len(failure_sample) < page_size:
                        failure_sample.append(
                            {
                                "pid": pid,
                                "error": internal_exception_observation(exc),
                            }
                        )
            last = page[-1]
            next_cursor = ProcessCursor(
                created_at=str(last["created_at"]),
                pid=str(last["pid"]),
            )
            if after is not None and next_cursor <= after:
                raise ProcessError(
                    "process terminal cleanup recovery page did not advance"
                )
            after = next_cursor
            if len(page) < page_size:
                break
        if interruptions:
            if len(interruptions) == 1:
                raise interruptions[0]
            raise BaseExceptionGroup(
                "process terminal cleanup recovery interrupted after bounded recovery",
                interruptions,
            )
        return {
            "pending": pending_total,
            "recovered_count": recovered_total,
            "failure_count": failure_total,
            "recovered": recovered_sample,
            "failures": failure_sample,
            "diagnostics_truncated": (
                recovered_total > len(recovered_sample)
                or failure_total > len(failure_sample)
            ),
        }

    def _terminal_cleanup_lock(self, pid: str) -> threading.RLock:
        stripe = sum(pid.encode("utf-8")) % len(self._terminal_cleanup_locks)
        return self._terminal_cleanup_locks[stripe]

    @staticmethod
    def _terminal_preserve_oids(process: AgentProcess) -> set[str]:
        outcome = process.outcome
        if isinstance(outcome, (ExitedProcessOutcome, FailedProcessOutcome)):
            return {outcome.result_oid} if outcome.result_oid else set()
        if isinstance(outcome, KilledProcessOutcome):
            return {outcome.reason_oid} if outcome.reason_oid else set()
        return set()

    def _record_terminal_cleanup_failure(
        self,
        pid: str,
        *,
        actor: str,
        attempt: int,
        terminal_status: str,
        errors: list[dict[str, Any]],
        persistence_error: dict[str, Any] | None,
        legacy_audit_action: str | None,
        context: dict[str, Any] | None,
    ) -> None:
        first = errors[0]
        decision: dict[str, Any] = {
            "terminal_status": terminal_status,
            "phase": first["phase"],
            "attempt": attempt,
            "error": {
                "error_type": first["error_type"],
                "exception_text": dict(first["exception_text"]),
            },
            "errors": [dict(error) for error in errors],
        }
        if persistence_error is not None:
            decision["persistence_error"] = dict(persistence_error)
        try:
            self.audit.record(
                actor=actor,
                action="process.terminal_cleanup_failed",
                target=f"process:{pid}",
                decision=decision,
            )
        except BaseException:
            pass
        if legacy_audit_action is None:
            return
        try:
            self.audit.record(
                actor=actor,
                action=legacy_audit_action,
                target=f"process:{pid}",
                decision={
                    **dict(context or {}),
                    "errors": [dict(error) for error in errors],
                },
            )
        except BaseException:
            pass

    def pause(self, pid: str, reason: str) -> None:
        self.signal(pid, ProcessSignal.PAUSE, {"reason": reason})

    def pause_for_host_resume(self, pid: str, reason: str) -> None:
        self.signal(
            pid,
            ProcessSignal.PAUSE,
            {"reason": reason},
            _require_host_resume=True,
        )

    def resume(self, pid: str) -> None:
        self.signal(pid, ProcessSignal.RESUME, {})

    def cancel(self, pid: str, reason: str) -> None:
        self.signal(pid, ProcessSignal.CANCEL, {"reason": reason})

    def exit(
        self,
        pid: str,
        result: ObjectHandle | None = None,
        failed: bool = False,
        message: str | None = None,
        *,
        payload: dict[str, Any] | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> ObjectHandle | None:
        process = self._get(pid)
        if process.status in self.TERMINAL_STATUSES:
            self._release_rejected_exit_result(pid, result)
            cleanup = self.terminal_cleanup_state(pid)
            if cleanup["state"] != "completed":
                self.retry_terminal_cleanup(
                    pid,
                    _actor=pid,
                    _legacy_audit_action="process.exit_finalize_failed",
                    _context={"status": process.status.value, "retry": True},
                )
                return None
            raise ProcessError(f"cannot exit terminal process: {pid} status={process.status.value}")
        live_descendants = self._live_descendant_pids(pid)
        if live_descendants and failed:
            self._terminate_descendants_for_failed_exit(pid)
            live_descendants = self._live_descendant_pids(pid)
        if live_descendants:
            raise ProcessError(
                "cannot exit process while descendants are nonterminal; "
                "wait for or terminate them first: "
                + ", ".join(live_descendants)
            )
        message_present = message is not None
        # Terminal state, child-budget release, evidence, and parent wakeup are
        # one durable lifecycle transition, including any generated result.
        # Host/object finalizers remain post-commit because provider cleanup
        # cannot be rolled back safely.
        with self.memory.ownership_locked(), self.store.transaction(include_object_payloads=True):
            process = self._get(pid)
            if process.status in self.TERMINAL_STATUSES:
                self._release_rejected_exit_result(pid, result)
                raise ProcessError(f"cannot exit terminal process: {pid} status={process.status.value}")
            live_descendants = self._live_descendant_pids(pid)
            if live_descendants:
                raise ProcessError(
                    "cannot exit process while descendants are nonterminal; "
                    "wait for or terminate them first: "
                    + ", ".join(live_descendants)
                )
            if result is None and payload is not None:
                selected_sources = self.flow_source_oids(pid, source_oids)
                result = self.memory.create_object(
                    pid=pid,
                    object_type=ObjectType.SUMMARY,
                    payload=payload,
                    metadata=self.flow_metadata(
                        selected_sources,
                        source_labels,
                        source_context,
                        base=ObjectMetadata(title="Process final result", tags=["final"]),
                    ),
                    provenance=Provenance(
                        created_from_action="process.exit",
                        parent_oids=selected_sources,
                    ),
                )
            elif result is None and message is not None:
                result = self._create_flow_text_carrier(
                    recipient_pid=pid,
                    text=message,
                    payload_field="message",
                    object_type=ObjectType.SUMMARY,
                    title="Process final result",
                    tags=["final", "process_exit"],
                    created_from_action="process.exit.message",
                    source_pid=pid,
                    source_oids=source_oids,
                    source_labels=source_labels,
                    source_context=source_context,
                )
            # Result construction may update the process MemoryView. Reload the
            # row before applying the terminal transition so neither write wins
            # over the other.
            process = self._get(pid)
            selected_status = ProcessStatus.FAILED if failed else ProcessStatus.EXITED
            result_oid = result.oid if result is not None else None
            outcome = (
                FailedProcessOutcome(result_oid=result_oid)
                if failed
                else ExitedProcessOutcome(result_oid=result_oid)
            )
            process = self.transitions.transition(
                pid,
                selected_status,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                outcome=outcome,
            )
            self._release_child_budget(pid)
            self.events.emit(
                EventType.PROCESS_EXITED,
                source=pid,
                target=process.parent_pid,
                payload={"pid": pid, "status": process.status.value, "result_oid": result.oid if result else None},
            )
            self.audit.record(
                actor=pid,
                action="process.exit",
                target=f"process:{pid}",
                output_refs=[result.oid] if result else [],
                decision={"status": process.status.value, "message_present": message_present},
            )
            self._wake_parent_waiting_on_child(process)
        self._complete_terminal_cleanup(
            process,
            actor=pid,
            audit_action="process.exit_finalize_failed",
            context={"status": process.status.value, "result_oid": result.oid if result is not None else None},
        )
        return result

    def _live_descendant_pids(self, pid: str) -> builtins.list[str]:
        stack = list(self.store.list_child_processes(pid))
        live: builtins.list[str] = []
        while stack:
            child = stack.pop()
            stack.extend(self.store.list_child_processes(child.pid))
            if (
                child.status not in self.TERMINAL_STATUSES
                and not self._is_host_managed_runner(child.pid)
            ):
                live.append(child.pid)
        return sorted(live)

    def _is_host_managed_runner(self, pid: str) -> bool:
        checker = self._host_managed_runner_checker
        if checker is None:
            return False
        try:
            return bool(checker(pid))
        except Exception:
            # Runner classification is a safety exception. Failure to prove it
            # keeps the process under ordinary descendant lifecycle rules.
            return False

    def _terminate_descendants_for_failed_exit(self, pid: str) -> None:
        stack: builtins.list[tuple[AgentProcess, int]] = [
            (child, 1) for child in self.store.list_child_processes(pid)
        ]
        descendants: builtins.list[tuple[AgentProcess, int]] = []
        while stack:
            child, depth = stack.pop()
            descendants.append((child, depth))
            stack.extend(
                (grandchild, depth + 1)
                for grandchild in self.store.list_child_processes(child.pid)
            )
        for child, _depth in sorted(
            descendants,
            key=lambda item: (-item[1], item[0].pid),
        ):
            current = self.store.get_process(child.pid)
            if current is None or current.status in self.TERMINAL_STATUSES:
                continue
            if self._is_host_managed_runner(child.pid):
                continue
            try:
                self.signal(
                    child.pid,
                    ProcessSignal.TERMINATE,
                    {"reason": f"ancestor process failed: {pid}"},
                )
            except ProcessTerminalCleanupRequired:
                latest = self.store.get_process(child.pid)
                if latest is None or latest.status not in self.TERMINAL_STATUSES:
                    raise

    def finalize_killed_processes(self, pids: Iterable[str], *, reason: str) -> None:
        failures: list[tuple[str, str, BaseException]] = []
        interruptions: list[BaseException] = []
        for pid in pids:
            process = self.store.get_process(pid)
            if process is None or process.status != ProcessStatus.KILLED:
                continue
            try:
                self.retry_terminal_cleanup(
                    pid,
                    _actor="resource_manager",
                    _context={"reason": reason},
                )
            except BaseException as exc:
                failures.append((pid, "terminal_cleanup", exc))
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
            try:
                self._emit_killed_process_exit_event(
                    process,
                    reason=reason,
                )
            except BaseException as exc:
                failures.append((pid, "exit_event", exc))
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
        if failures:
            self._raise_killed_process_finalization_failure(
                failures,
                interruptions=interruptions,
            )

    def _emit_killed_process_exit_event(
        self,
        process: AgentProcess,
        *,
        reason: str,
    ) -> None:
        self.events.emit_once(
            self._terminal_event_id(process, EventType.PROCESS_EXITED),
            EventType.PROCESS_EXITED,
            source=process.pid,
            target=process.parent_pid,
            payload={
                "pid": process.pid,
                "status": process.status.value,
                "result_oid": None,
                "reason": reason,
            },
            causality={
                "kind": "process_terminal_state",
                "pid": process.pid,
                "state_generation": process.state_generation,
            },
        )

    @staticmethod
    def _terminal_event_id(process: AgentProcess, event_type: EventType) -> str:
        identity = (
            f"{event_type.value}\x00{process.pid}\x00{process.state_generation}"
        ).encode("utf-8")
        return f"evt_terminal_{hashlib.sha256(identity).hexdigest()}"

    @staticmethod
    def _raise_killed_process_finalization_failure(
        failures: list[tuple[str, str, BaseException]],
        *,
        interruptions: list[BaseException],
    ) -> NoReturn:
        aggregate = RuntimeError("killed process finalization failed")
        public_error = public_error_envelope(
            aggregate,
            code="killed_process_finalization_failed",
        )
        aggregate.args = (public_error["message"],)
        observations = tuple(
            {
                "pid": pid,
                "phase": phase,
                **internal_exception_observation(
                    error,
                    correlation_id=public_error["correlation_id"],
                ),
            }
            for pid, phase, error in failures
        )
        aggregate.finalization_failures = observations  # type: ignore[attr-defined]
        if interruptions:
            sanitized_interruptions = [
                _SanitizedFinalizationInterruption(
                    public_error_envelope(
                        interruption,
                        code="killed_process_finalization_interrupted",
                    )["message"]
                )
                for interruption in interruptions
            ]
            grouped = BaseExceptionGroup(
                "killed process finalization interrupted after full cleanup",
                [aggregate, *sanitized_interruptions],
            )
            grouped.finalization_failures = observations  # type: ignore[attr-defined]
            raise grouped
        failures.clear()
        raise aggregate

    def _finalize_terminal_process(self, process: AgentProcess, preserve_oids: set[str]) -> None:
        self._release_terminal_child_memory(process.pid, preserve_oids=preserve_oids)
        if process.parent_pid is None:
            # Root process-owned memory is volatile and is reclaimed immediately.
            # Child process memory is held until the parent can merge or discard
            # it, so merge_child_memory remains meaningful after child exit.
            self.memory.release_process_owned(process.pid, preserve_oids=preserve_oids)
            self._redact_terminal_root_spawn_initial_goal(process)

    def _release_terminal_child_memory(self, pid: str, preserve_oids: set[str]) -> None:
        stack = [
            child
            for child in self.store.list_child_processes(pid)
            if child.status in self.TERMINAL_STATUSES
        ]
        terminal_children: builtins.list[AgentProcess] = []
        while stack:
            child = stack.pop()
            if child.status in self.TERMINAL_STATUSES:
                terminal_children.append(child)
                stack.extend(
                    grandchild
                    for grandchild in self.store.list_child_processes(child.pid)
                    if grandchild.status in self.TERMINAL_STATUSES
                )
        for child in reversed(terminal_children):
            self.memory.release_process_owned(child.pid, preserve_oids=preserve_oids)

    def get(self, pid: str) -> AgentProcess:
        return self._get(pid)

    def list(
        self,
        *,
        limit: int | None = None,
        active_first: bool = False,
    ) -> builtins.list[AgentProcess]:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValidationError("process list limit must be a positive integer")
        if not isinstance(active_first, bool):
            raise ValidationError("process active_first must be boolean")
        return self.store.list_processes(limit=limit, active_first=active_first)

    def _get(self, pid: str) -> AgentProcess:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process

    def _require_active_parent(self, pid: str, action: str) -> AgentProcess:
        process = self._get(pid)
        if process.status in self.TERMINAL_STATUSES:
            raise ProcessError(f"cannot {action} terminated process: {pid}")
        return process

    def _require_parent_launch_fence(
        self,
        expected: AgentProcess,
        *,
        action: str,
    ) -> AgentProcess:
        current = self._require_active_parent(expected.pid, action)
        if (
            current.status != expected.status
            or current.state_generation != expected.state_generation
        ):
            raise ProcessError(
                "parent process changed during child publication: "
                f"{expected.pid} expected={expected.status.value}/"
                f"{expected.state_generation} found={current.status.value}/"
                f"{current.state_generation}"
            )
        return current

    def _commit_launch_authority_reservations(
        self,
        reservation_ids: tuple[str, ...],
        *,
        parent_pid: str,
        operation: str,
    ) -> None:
        if any(
            type(reservation_id) is not str or not reservation_id
            for reservation_id in reservation_ids
        ) or len(reservation_ids) != len(set(reservation_ids)):
            raise ValidationError("process launch authority reservations must be unique ids")
        for reservation_id in reservation_ids:
            committed = self.capabilities.commit_reserved_use(
                reservation_id,
                committed_by=f"{operation}:{parent_pid}",
                reason="one-time process launch authority committed",
            )
            if not committed:
                raise CapabilityDenied(
                    "process launch authority reservation could not be committed"
                )

    def _default_resource_budget(self) -> ResourceBudget:
        defaults = self.config.process
        return ResourceBudget(
            max_tool_calls=defaults.max_tool_calls,
            max_child_processes=defaults.max_child_processes,
            max_runtime_seconds=defaults.max_runtime_seconds,
            max_context_materialization_tokens=defaults.max_context_materialization_tokens,
            max_context_materialization_total_tokens=defaults.max_context_materialization_total_tokens,
            max_llm_calls=defaults.max_llm_calls,
            max_llm_total_tokens=defaults.max_llm_total_tokens,
            max_subprocess_wall_seconds=defaults.max_subprocess_wall_seconds,
            max_subprocess_cpu_seconds=defaults.max_subprocess_cpu_seconds,
            max_subprocess_memory_bytes=defaults.max_subprocess_memory_bytes,
            max_external_read_bytes=defaults.max_external_read_bytes,
            max_external_write_bytes=defaults.max_external_write_bytes,
            max_jsonrpc_bytes=defaults.max_jsonrpc_bytes,
            max_mcp_bytes=defaults.max_mcp_bytes,
            max_deno_syscalls=defaults.max_deno_syscalls,
        )

    def _normalize_working_directory(self, path: str | None) -> str:
        raw = (path or self.config.process.default_working_directory).replace("\\", "/").strip()
        if raw in {"", "."}:
            return "."
        if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
            raise ProcessError(f"working directory must be workspace-relative: {path}")
        parts: list[str] = []
        for part in raw.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    raise ProcessError(f"working directory escapes workspace root: {path}")
                parts.pop()
                continue
            parts.append(part)
        return "/".join(parts) if parts else "."

    def _select_child_working_directory(
        self,
        path: str | None,
        *,
        inherited: str,
        validated: str | None,
    ) -> str:
        if validated is not None:
            if path is not None:
                raise ValidationError(
                    "raw and validated working directories are mutually exclusive"
                )
            return self._validated_working_directory_identity(validated)
        if not path:
            return self._validated_working_directory_identity(inherited)
        return self._normalize_working_directory(path)

    @staticmethod
    def _validated_working_directory_identity(path: str) -> str:
        """Check canonical shape without rewriting the provider's path identity."""

        if not isinstance(path, str) or not path or "\x00" in path:
            raise ProcessError("validated working directory must be a canonical string")
        if path == ".":
            return path
        has_windows_drive = os.name == "nt" and bool(os.path.splitdrive(path)[0])
        if os.path.isabs(path) or has_windows_drive or any(
            part in {"", ".", ".."} for part in path.split("/")
        ):
            raise ProcessError("validated working directory is not canonical")
        return path

    def _resolve_root_llm_profile(self, image_id: str, explicit_profile_id: str | None) -> str:
        if self._llm_profile_resolver is not None:
            return self._llm_profile_resolver(image_id, explicit_profile_id)
        if explicit_profile_id is not None:
            return self._normalize_llm_profile_id(explicit_profile_id)
        return self.config.llm.default_profile_id

    def _resolve_child_llm_profile(self, parent: AgentProcess, explicit_profile_id: str | None) -> str:
        if explicit_profile_id is not None:
            return self._normalize_llm_profile_id(explicit_profile_id)
        return parent.llm_profile_id or self.config.llm.default_profile_id

    def _normalize_llm_profile_id(self, profile_id: str) -> str:
        selected = str(profile_id or "").strip()
        if not selected:
            raise ProcessError("LLM profile id must be a non-empty string")
        return selected

    def _child_resource_budget(self, parent: AgentProcess) -> ResourceBudget:
        budget = self.resources.remaining_budget(parent.pid) if self.resources is not None else parent.resource_budget
        divisor = self.config.process.fork_budget_divisor
        return ResourceBudget(
            max_tool_calls=self._attenuate_int(budget.max_tool_calls, divisor, self.config.process.fork_min_tool_calls),
            max_child_processes=self._attenuate_int(
                budget.max_child_processes,
                divisor,
                self.config.process.fork_min_child_processes,
            ),
            max_runtime_seconds=self._attenuate_float(budget.max_runtime_seconds, divisor),
            max_context_materialization_tokens=budget.max_context_materialization_tokens,
            max_context_materialization_total_tokens=self._attenuate_int(
                budget.max_context_materialization_total_tokens,
                divisor,
                0,
            ),
            max_llm_calls=self._attenuate_int(budget.max_llm_calls, divisor, 0),
            max_llm_total_tokens=self._attenuate_int(budget.max_llm_total_tokens, divisor, 0),
            max_subprocess_wall_seconds=self._attenuate_float(budget.max_subprocess_wall_seconds, divisor),
            max_subprocess_cpu_seconds=self._attenuate_float(budget.max_subprocess_cpu_seconds, divisor),
            max_subprocess_memory_bytes=self._attenuate_int(budget.max_subprocess_memory_bytes, divisor, 0),
            max_external_read_bytes=self._attenuate_int(budget.max_external_read_bytes, divisor, 0),
            max_external_write_bytes=self._attenuate_int(budget.max_external_write_bytes, divisor, 0),
            max_jsonrpc_bytes=self._attenuate_int(budget.max_jsonrpc_bytes, divisor, 0),
            max_mcp_bytes=self._attenuate_int(budget.max_mcp_bytes, divisor, 0),
            max_deno_syscalls=self._attenuate_int(budget.max_deno_syscalls, divisor, 0),
        )

    def _select_child_resource_budget(
        self,
        parent: AgentProcess,
        requested: ResourceBudget | dict[str, Any] | None,
        *,
        authority_manifest: Any | None = None,
    ) -> ResourceBudget:
        selected = (
            self._coerce_resource_budget(requested)
            if requested is not None
            else self._child_resource_budget(parent)
        )
        if self.authority_manifests is not None:
            selected = ResourceBudget(
                **self.authority_manifests.resolve_launch_resource_budget(
                    supplied=authority_manifest,
                    resource_budget=selected,
                    parent_pid=parent.pid,
                )
            )
        if self.resources is not None:
            self.resources.validate_child_budget(
                parent.pid,
                selected,
                reserved_usage=ResourceUsage(child_processes=1),
            )
        return selected

    @staticmethod
    def _assert_manifest_resource_budget(
        manifest: Any,
        selected: ResourceBudget,
    ) -> None:
        if ResourceBudget(**manifest.resource_budget) != selected:
            raise ProcessError("authority manifest resource budget changed during launch")

    def _reserve_child_budget(self, parent_pid: str, child_pid: str, budget: ResourceBudget) -> None:
        if self.resources is None:
            return
        self.resources.reserve_child_budget(parent_pid, child_pid, budget)

    def _release_child_budget(self, pid: str) -> None:
        if self.resources is None:
            return
        self.resources.release_process_reservations(pid)

    def release_child_budget(self, pid: str) -> None:
        """Release a child reservation after an external launch rollback."""

        self._release_child_budget(pid)

    def _charge_child_creation(self, parent_pid: str) -> None:
        if self.resources is None:
            return
        self.resources.charge(
            parent_pid,
            ResourceUsage(child_processes=1),
            source="process.child_create",
            context={"parent_pid": parent_pid},
            allow_overage=False,
            kill_on_exceed=False,
        )

    def _coerce_resource_budget(self, value: ResourceBudget | dict[str, Any]) -> ResourceBudget:
        if isinstance(value, ResourceBudget):
            raw = {
                name: getattr(value, name)
                for name in value.__dataclass_fields__
            }
            try:
                return ResourceBudget(**raw)
            except (TypeError, ValueError) as exc:
                raise ProcessError(str(exc)) from exc
        if not isinstance(value, dict):
            raise ProcessError("resource_budget must be a mapping")
        allowed = set(ResourceBudget.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ProcessError(f"unknown resource_budget fields: {unknown}")
        try:
            return ResourceBudget(**{key: item for key, item in value.items() if key in allowed})
        except (TypeError, ValueError) as exc:
            raise ProcessError(str(exc)) from exc

    def _attenuate_int(self, value: int | None, divisor: int, minimum: int) -> int | None:
        if value is None:
            return None
        return max(minimum, int(value) // divisor)

    def _attenuate_float(self, value: float | int | None, divisor: int) -> float | None:
        if value is None:
            return None
        return float(value) / divisor

    def flow_source_oids(self, pid: str, source_oids: Iterable[str] | None = None) -> list[str]:
        """Return trusted ambient Object sources for a process-originated value.

        Explicit sources come from runtime-owned ToolContext metadata.  The
        current goal and MemoryView roots are included conservatively so a
        process cannot wash a label merely by copying observed content into a
        raw child goal or process message.
        """

        process = self._get(pid)
        explicit = self._normalize_source_oids(source_oids)
        candidates = [*explicit]
        if process.goal_oid:
            candidates.append(process.goal_oid)
        if process.memory_view is not None:
            candidates.extend(handle.oid for handle in process.memory_view.roots)
        selected: list[str] = []
        seen: set[str] = set()
        for oid in candidates:
            if oid in seen:
                continue
            obj = self.objects.get_object(oid)
            if obj is None:
                if oid in explicit:
                    raise NotFound(f"data-flow source object not found: {oid}")
                continue
            seen.add(oid)
            selected.append(oid)
        return selected

    def flow_metadata(
        self,
        source_oids: Iterable[str],
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
        *,
        base: ObjectMetadata | None = None,
    ) -> ObjectMetadata:
        labels: list[DataLabels] = []
        for oid in self._normalize_source_oids(source_oids):
            obj = self.objects.get_object(oid)
            if obj is None:
                raise NotFound(f"data-flow source object not found: {oid}")
            labels.append(DataLabels.from_object_metadata(obj.metadata))
        if source_context is not None:
            if not isinstance(source_context, DataFlowContext):
                raise ProcessError("trusted source_context must use DataFlowContext")
            labels.append(source_context.labels)
        supplied = metadata_from_labels(source_labels)
        if supplied is not None:
            labels.append(DataLabels.from_object_metadata(supplied))
        aggregate = metadata_from_labels(DataLabels.aggregate(labels))
        return propagate_object_labels(
            base or ObjectMetadata(),
            [aggregate] if aggregate is not None else [],
        )

    def flow_context(
        self,
        pid: str,
        source_oids: Iterable[str],
        *,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> DataFlowContext:
        selected_oids = self._normalize_source_oids(source_oids)
        contexts: list[DataFlowContext] = []
        if selected_oids and self.data_flow is not None:
            contexts.append(
                self.data_flow.context_from_source_oids(
                    pid,
                    selected_oids,
                    include_current=False,
                )
            )
        elif selected_oids:
            metadata: list[ObjectMetadata] = []
            for oid in selected_oids:
                obj = self.objects.get_object(oid)
                if obj is None:
                    raise NotFound(f"data-flow source object not found: {oid}")
                metadata.append(obj.metadata)
            contexts.append(
                DataFlowContext(
                    labels=DataLabels.aggregate(
                        DataLabels.from_object_metadata(item) for item in metadata
                    )
                )
            )
        if source_context is not None:
            if not isinstance(source_context, DataFlowContext):
                raise ProcessError("trusted source_context must use DataFlowContext")
            contexts.append(source_context)
        aggregate = metadata_from_labels(source_labels)
        if aggregate is not None:
            contexts.append(
                DataFlowContext(labels=DataLabels.from_object_metadata(aggregate))
            )
        return DataFlowContext.aggregate(contexts)

    def observe_message_labels(self, pid: str, messages: Iterable[ProcessMessage]) -> list[str]:
        """Persist received-message labels as metadata-only process roots.

        Message text already remains in the mailbox.  The carrier contains
        only its id, keeping payload duplication out of Object Memory while
        making later goal/message derivations inherit the received labels.
        """

        selected_messages = list(messages)
        observed: list[str] = []
        refreshed_metadata: list[tuple[ProcessMessage, dict[str, Any]]] = []
        with self.memory.ownership_locked(), self.store.transaction(include_object_payloads=True):
            process = self._get(pid)
            if process.status in self.TERMINAL_STATUSES:
                raise ProcessError(
                    f"cannot observe messages for terminal process: {pid} "
                    f"status={process.status.value}"
                )
            for supplied_message in selected_messages:
                message = self.store.get_process_message(supplied_message.message_id)
                if message is None:
                    raise ProcessError(
                        f"cannot observe missing process message: {supplied_message.message_id}"
                    )
                if message.recipient_pid != pid:
                    raise ProcessError(
                        f"process message {message.message_id} belongs to "
                        f"{message.recipient_pid}, not {pid}"
                    )
                labels = metadata_from_labels(message.metadata)
                if labels is None:
                    refreshed_metadata.append((supplied_message, dict(message.metadata)))
                    continue
                existing_oid = str(message.metadata.get("label_carrier_oid") or "").strip()
                existing = self.objects.get_object(existing_oid) if existing_oid else None
                if existing is not None:
                    handle = self.memory.handle_for_oid(
                        pid,
                        existing.oid,
                        required_rights={"read"},
                        optional_rights={"materialize", "link", "diff"},
                        issued_by="process.message.observe",
                    )
                else:
                    source_oids = [
                        oid
                        for oid in self._normalize_source_oids(message.metadata.get("source_oids"))
                        if self.objects.get_object(oid) is not None
                    ]
                    metadata = self.flow_metadata(
                        source_oids,
                        labels,
                        base=ObjectMetadata(
                            title="Observed process message labels",
                            tags=["process_message", "label_carrier"],
                        ),
                    )
                    handle = self.memory.create_object(
                        pid=pid,
                        object_type=ObjectType.MESSAGE,
                        payload={"message_id": message.message_id},
                        metadata=metadata,
                        immutable=True,
                        provenance=Provenance(
                            source_refs=[f"process_message:{message.message_id}"],
                            created_from_action="process.message.observe",
                            parent_oids=source_oids,
                        ),
                    )
                    expected_metadata = dict(message.metadata)
                    message.metadata["label_carrier_oid"] = handle.oid
                    message.updated_at = utc_now()
                    if not self.store.update_process_message_metadata(
                        message.message_id,
                        recipient_pid=pid,
                        expected_metadata=expected_metadata,
                        metadata=message.metadata,
                        updated_at=message.updated_at,
                    ):
                        raise ProcessError(
                            f"process message metadata changed while observing: {message.message_id}"
                        )
                self._add_handle_to_process_view(process, handle)
                process = self._get(pid)
                observed.append(handle.oid)
                refreshed_metadata.append((supplied_message, dict(message.metadata)))
        # Keep caller-held message snapshots useful without ever copying their
        # stale status or ACK timestamps back into durable storage.
        for supplied_message, metadata in refreshed_metadata:
            supplied_message.metadata.clear()
            supplied_message.metadata.update(metadata)
        return observed

    def _ensure_goal(
        self,
        pid: str,
        goal: dict[str, Any] | str | ObjectHandle | None,
        *,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> ObjectHandle:
        if isinstance(goal, ObjectHandle):
            selected_sources = self._normalize_source_oids(source_oids)
            has_additional_flow = bool(selected_sources) or source_labels is not None or source_context is not None
            if not has_additional_flow:
                if (
                    self.authority_manifests is not None
                    and self.authority_manifests.get_for_process(pid) is not None
                ):
                    self._assert_goal_data_flow(pid, goal)
                return goal
            source_goal = self.objects.get_object(goal.oid)
            if source_goal is None:
                raise NotFound(f"process goal object not found: {goal.oid}")
            if goal.oid not in selected_sources:
                selected_sources.append(goal.oid)
            payload = deepcopy(source_goal.payload)
        else:
            default_goal = self.config.process.default_goal_text
            payload = {"text": goal or default_goal} if isinstance(goal, str) or goal is None else goal
            selected_sources = self._normalize_source_oids(source_oids)
        metadata = self.flow_metadata(
            selected_sources,
            source_labels,
            source_context,
            base=ObjectMetadata(title="Process goal", tags=["goal"]),
        )
        if (
            self.authority_manifests is not None
            and self.authority_manifests.get_for_process(pid) is not None
        ):
            self.authority_manifests.assert_data_flow_labels(
                pid,
                DataLabels.from_object_metadata(metadata),
            )
        return self.memory.create_object(
            pid=pid,
            object_type=ObjectType.GOAL,
            payload=payload,
            metadata=metadata,
            immutable=True,
            provenance=Provenance(
                created_from_action="process.goal",
                parent_oids=selected_sources,
            ),
        )

    def _assert_goal_data_flow(self, pid: str, goal: ObjectHandle) -> None:
        self._assert_object_data_flow(pid, goal.oid)

    def _assert_object_data_flow(self, pid: str, oid: str) -> None:
        if self.authority_manifests is None:
            return
        obj = self.objects.get_object(oid)
        if obj is not None:
            labels = DataLabels.from_object_metadata(obj.metadata)
        else:
            # Object payloads are runtime-local and may be released on reopen,
            # while process-result metadata remains durable. The receiver
            # domain check must therefore use that Host-written label record
            # before handing the caller a handle whose materialization will
            # still fail. Direct database tampering is outside this boundary.
            try:
                metadata = self.objects.get_persisted_object_metadata(oid)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"invalid persisted metadata for released object {oid}: {exc}"
                ) from exc
            if metadata is None:
                raise NotFound(f"data-flow source object not found: {oid}")
            labels = DataLabels.from_object_metadata(metadata)
        self.authority_manifests.assert_data_flow_labels(
            pid,
            labels,
        )

    def _normalize_source_oids(self, source_oids: Iterable[str] | None) -> list[str]:
        if source_oids is None:
            return []
        if isinstance(source_oids, (str, bytes)):
            raise ProcessError("data-flow source_oids must be a collection of Object ids")
        selected: list[str] = []
        seen: set[str] = set()
        for value in source_oids:
            oid = str(value or "").strip()
            if not oid:
                raise ProcessError("data-flow source_oids cannot contain empty Object ids")
            if oid not in seen:
                selected.append(oid)
                seen.add(oid)
        return selected

    def _initialize_launch_goal(
        self,
        publication_id: str,
        pid: str,
        goal: dict[str, Any] | str | ObjectHandle | None,
        *,
        image_id: str | None = None,
        record_initial_goal_recovery: bool = False,
        parent_pid: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> ObjectHandle:
        with self._launch_capability_receipts(publication_id, pid):
            self.memory.ensure_process_namespace(pid, parent_pid=parent_pid)
            self._publication_phase(publication_id, "namespace_created")
            handle = self._ensure_goal(
                pid,
                goal,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )
            receipt: dict[str, Any] = {"goal_oid": handle.oid}
            if record_initial_goal_recovery:
                if image_id is None:
                    raise ValidationError(
                        "root spawn initial goal recovery requires an image identity"
                    )
                goal_object = self.objects.get_object(handle.oid)
                if goal_object is None or goal_object.type is not ObjectType.GOAL:
                    raise ValidationError(
                        "root spawn initial goal payload is unavailable before publication"
                    )
                if goal_object.immutable:
                    identity = initial_goal_object_identity(
                        oid=goal_object.oid,
                        namespace=goal_object.namespace,
                        name=goal_object.name,
                        object_type=goal_object.type.value,
                        schema_version=goal_object.schema_version,
                        metadata=to_jsonable(goal_object.metadata),
                        provenance=to_jsonable(goal_object.provenance),
                        version=goal_object.version,
                        immutable=goal_object.immutable,
                        created_by=goal_object.created_by,
                        owner_kind=goal_object.owner_kind.value,
                        owner_id=str(goal_object.owner_id or ""),
                        created_at=goal_object.created_at,
                    )
                    try:
                        launch_process = self._get(pid)
                        receipt[INITIAL_GOAL_RECOVERY_KEY] = (
                            build_initial_goal_recovery_envelope(
                                pid=pid,
                                process_created_at=launch_process.created_at,
                                image_id=image_id,
                                goal_oid=handle.oid,
                                object_version=goal_object.version,
                                object_identity_sha256=(
                                    initial_goal_object_identity_sha256(identity)
                                ),
                                payload=goal_object.payload,
                                persist_full_io=self.config.llm.persist_full_io,
                                payload_hard_limit_bytes=(
                                    self.config.tools.memory_payload_hard_limit_bytes
                                ),
                            )
                        )
                    except ValueError as exc:
                        raise ValidationError(
                            f"invalid root spawn initial goal recovery payload: {exc}"
                        ) from exc
            self._publication_phase(publication_id, "goal_created", **receipt)
        return handle

    def _compile_root_launch_authority(
        self,
        publication_id: str,
        *,
        pid: str,
        image_id: str,
        goal_handle: ObjectHandle,
        capabilities: builtins.list[dict[str, Any]],
        resource_budget: ResourceBudget | None,
        authority_manifest: Any | None,
    ) -> None:
        with self._launch_capability_receipts(publication_id, pid):
            if self.authority_manifests is None:
                self._grant_specs(pid, capabilities, issued_by="process.spawn")
            else:
                manifest = self.authority_manifests.prepare_launch(
                    pid=pid,
                    image_id=image_id,
                    goal_ref=goal_handle.oid,
                    supplied=authority_manifest,
                    authorized_capabilities=capabilities,
                    resource_budget=resource_budget,
                    issued_by="process.spawn",
                )
                if manifest.resource_budget:
                    process = self._get(pid)
                    process.resource_budget = ResourceBudget(**manifest.resource_budget)
                    process.updated_at = utc_now()
                    self.store.patch_process(
                        pid,
                        {
                            "resource_budget": process.resource_budget,
                            "updated_at": process.updated_at,
                        },
                        expected_revision=process.revision,
                    )
                self._assert_goal_data_flow(pid, goal_handle)
                self.authority_manifests.compile_root_capabilities(manifest)
            self._publication_phase(publication_id, "authority_compiled")

    def _publish_fork_launch_view(
        self,
        publication_id: str,
        *,
        parent_pid: str,
        child_pid: str,
        source_view: MemoryView,
        spec: MemoryViewSpec,
        goal_handle: ObjectHandle,
    ) -> AgentProcess:
        with self._launch_capability_receipts(publication_id, child_pid):
            child_view = self.memory.fork_view(
                parent_pid,
                child_pid,
                source_view,
                spec,
            )
            for root in child_view.roots:
                self._assert_object_data_flow(child_pid, root.oid)
            child_view.roots.append(goal_handle)
            child = self._get(child_pid)
            child.goal_oid = goal_handle.oid
            child.memory_view = child_view
            child.updated_at = utc_now()
            return self.store.patch_process(
                child_pid,
                {
                    "goal_oid": child.goal_oid,
                    "memory_view": child.memory_view,
                    "updated_at": child.updated_at,
                },
                expected_revision=child.revision,
            )

    def _publish_child_launch_authority(
        self,
        publication_id: str,
        *,
        parent_pid: str,
        child_pid: str,
        manifest: Any | None,
        requested_capabilities: builtins.list[dict[str, Any]],
        inherit_specs: builtins.list[dict[str, Any]],
        transition_kind: str,
    ) -> None:
        with self._launch_capability_receipts(publication_id, child_pid):
            self._compile_child_authority(
                parent_pid=parent_pid,
                child_pid=child_pid,
                manifest=manifest,
                requested_capabilities=requested_capabilities,
                inherit_specs=inherit_specs,
                transition_kind=transition_kind,
            )
            self._publication_phase(publication_id, "authority_compiled")

    def _grant_specs(self, pid: str, specs: Iterable[dict[str, Any]], issued_by: str) -> None:
        for spec in specs:
            self.capabilities.grant(
                subject=pid,
                resource=spec["resource"],
                rights=spec.get("rights", [CapabilityRight.READ.value]),
                issued_by=issued_by,
                constraints=spec.get("constraints"),
                expires_at=spec.get("expires_at"),
                delegable=spec.get("delegable", False),
                revocable=spec.get("revocable", True),
            )

    def _compile_child_authority(
        self,
        *,
        parent_pid: str,
        child_pid: str,
        manifest: Any | None,
        requested_capabilities: list[dict[str, Any]],
        inherit_specs: list[dict[str, Any]],
        transition_kind: str,
    ) -> None:
        selected_specs = (
            self.authority_manifests.bounded_capability_specs(manifest)
            if manifest is not None
            else [*requested_capabilities, *inherit_specs]
        )
        if not selected_specs:
            return
        self.capabilities.derive_authority(
            source_subject=parent_pid,
            target_subject=child_pid,
            requested_specs=selected_specs,
            transition_kind=transition_kind,
            ceiling_specs=selected_specs if manifest is not None else None,
        )

    def _inherit_capability_specs(
        self,
        parent_pid: str,
        child_pid: str,
        specs: Iterable[dict[str, Any]],
        issued_by: str,
    ) -> None:
        for spec in specs:
            self.capabilities.inherit(
                parent=parent_pid,
                child=child_pid,
                resource=spec["resource"],
                rights=spec.get("rights", [CapabilityRight.READ.value]),
                issued_by=issued_by,
                constraints=spec.get("constraints") if isinstance(spec.get("constraints"), dict) else None,
            )

    def _validate_inherit_capability_specs(self, parent_pid: str, specs: Iterable[dict[str, Any]]) -> None:
        for spec in specs:
            try:
                self.capabilities.validate_delegation(parent_pid, spec)
            except CapabilityDenied as exc:
                resource = spec.get("resource")
                rights = spec.get("rights", [CapabilityRight.READ.value])
                raise CapabilityDenied(
                    f"{parent_pid} cannot inherit {rights} on {resource}: {exc}"
                ) from exc

    def _fork_mode_to_view_mode(self, mode: ForkMode) -> ViewMode:
        if mode == ForkMode.COPY:
            return ViewMode.COPY_ON_WRITE
        if mode == ForkMode.SPECULATIVE:
            return ViewMode.EPHEMERAL
        if mode == ForkMode.WORKER:
            return ViewMode.READ_ONLY
        return ViewMode.READ_ONLY

    def _run_after_spawn_hooks(
        self,
        pid: str,
        image_id: str,
        publication_id: str,
    ) -> None:
        for hook in self._after_spawn_hooks:
            hook(pid, image_id, publication_id)

    def _run_before_spawn_hooks(self, image_id: str) -> None:
        for hook in self._before_spawn_hooks:
            hook(image_id)

    def recover_root_spawn_initial_goal_payloads(self) -> tuple[str, ...]:
        """Rehydrate only exact active root goals before volatile payload sweep."""

        self._require_recovery_lease()
        recovered: list[str] = []
        after: ProcessCursor | None = None
        page_size = self.config.runtime.jit_rehydration_page_size
        while True:
            page = self.store.query_processes(after=after, limit=page_size)
            previous = after
            for process in page.records:
                cursor = ProcessCursor(process.created_at, process.pid)
                if previous is not None and cursor <= previous:
                    raise ValidationError(
                        "process repository returned an invalid initial-goal recovery page"
                    )
                previous = cursor
                recovered_oid = self._recover_root_spawn_initial_goal_payload(
                    process
                )
                if recovered_oid is not None:
                    recovered.append(recovered_oid)
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "process repository returned an invalid initial-goal recovery cursor"
                )
            after = page.next_cursor
        return tuple(recovered)

    def _recover_root_spawn_initial_goal_payload(
        self,
        process: AgentProcess,
    ) -> str | None:
        if process.parent_pid is not None or not process.goal_oid:
            return None
        publication = self.publications.get_committed_root_spawn_publication(
            process.pid
        )
        if publication is None:
            return None
        envelope = self._root_spawn_initial_goal_envelope(publication)
        if envelope is None:
            return None
        if process.status in self.TERMINAL_STATUSES:
            self._redact_root_spawn_initial_goal(publication, envelope=envelope)
            return None
        if envelope.recoverable and not self.config.llm.persist_full_io:
            self._redact_root_spawn_initial_goal(publication, envelope=envelope)
            return None
        if not self._root_spawn_goal_envelope_matches_process(envelope, process):
            return None
        if self.objects.get_object(process.goal_oid) is not None:
            return None
        restored = self.objects.rehydrate_root_spawn_goal_payload(
            process.goal_oid,
            envelope.payload,
            expected_version=envelope.object_version,
            expected_identity_sha256=envelope.object_identity_sha256,
            require_recovery_lease=self._require_recovery_lease,
        )
        return process.goal_oid if restored else None

    @staticmethod
    def _root_spawn_goal_envelope_matches_process(
        envelope: InitialGoalRecoveryEnvelope,
        process: AgentProcess,
    ) -> bool:
        return (
            envelope.recoverable
            and envelope.pid == process.pid
            and envelope.process_created_at == process.created_at
            and envelope.goal_oid == process.goal_oid
        )

    def _root_spawn_initial_goal_envelope(
        self,
        publication: dict[str, Any],
    ) -> InitialGoalRecoveryEnvelope | None:
        receipt = publication.get("receipt")
        phases = receipt.get("phases") if isinstance(receipt, dict) else None
        if not isinstance(phases, list):
            raise ValidationError("root spawn publication phases are invalid")
        matches = [
            phase
            for phase in phases
            if isinstance(phase, dict)
            and phase.get("phase") == "goal_created"
            and INITIAL_GOAL_RECOVERY_KEY in phase
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValidationError(
                "root spawn publication has multiple initial goal envelopes"
            )
        phase = matches[0]
        try:
            envelope = decode_initial_goal_recovery_envelope(
                phase[INITIAL_GOAL_RECOVERY_KEY],
                payload_hard_limit_bytes=(
                    self.config.tools.memory_payload_hard_limit_bytes
                ),
            )
        except ValueError as exc:
            raise ValidationError(
                f"invalid root spawn initial goal recovery envelope: {exc}"
            ) from exc
        if phase.get("goal_oid") != envelope.goal_oid:
            raise ValidationError("root spawn goal phase identity changed")
        plan = publication.get("plan")
        if (
            not isinstance(plan, dict)
            or plan.get("pid") != envelope.pid
            or plan.get("image_id") != envelope.image_id
            or plan.get("launch_kind") != "spawn"
            or plan.get("parent_pid") is not None
        ):
            raise ValidationError("root spawn initial goal publication anchor changed")
        return envelope

    def _redact_root_spawn_initial_goal(
        self,
        publication: dict[str, Any],
        *,
        envelope: InitialGoalRecoveryEnvelope | None = None,
        expected_states: Iterable[str] = ("committed",),
    ) -> None:
        selected = envelope or self._root_spawn_initial_goal_envelope(publication)
        if selected is None:
            return
        if not self.publications.redact_root_spawn_initial_goal(
            publication["publication_id"],
            pid=selected.pid,
            goal_oid=selected.goal_oid,
            expected_payload_sha256=selected.payload_sha256,
            expected_states=expected_states,
        ):
            raise ProcessError(
                "root spawn initial goal redaction lost its publication CAS: "
                f"{publication['publication_id']}"
            )

    def _redact_terminal_root_spawn_initial_goal(self, process: AgentProcess) -> None:
        if process.parent_pid is not None:
            return
        publication = self.publications.get_committed_root_spawn_publication(
            process.pid
        )
        if publication is None:
            return
        self._redact_root_spawn_initial_goal(publication)

    def _cleanup_failed_launch(self, pid: str) -> None:
        with contextlib.suppress(Exception):
            self.memory.release_process_owned(pid)
        namespace = self.memory.process_namespace(pid)
        namespace_resource = f"object_namespace:{namespace}"
        with contextlib.suppress(Exception):
            self.store.delete_process_scaffold(
                pid,
                namespace=namespace,
                namespace_resource=namespace_resource,
            )

    def recover_incomplete_publications(self) -> list[str]:
        """Compensate process launches interrupted before their atomic commit."""

        self._require_recovery_lease()

        recovered: list[str] = []
        page_size = self.config.runtime.publication_reconciliation_page_size
        for state in ("planning", "applying", "rollback_pending", "failed", "manual"):
            for operation_reconciled in (False, True):
                self._recover_launch_publication_state(
                    state,
                    operation_reconciled=operation_reconciled,
                    recovered=recovered,
                )

        self.reconcile_terminal_publications()
        self._fail_orphaned_created_processes()
        return recovered

    def _recover_launch_publication_state(
        self,
        state: str,
        *,
        operation_reconciled: bool,
        recovered: list[str],
    ) -> None:
        page_size = self.config.runtime.publication_reconciliation_page_size
        after: RuntimePublicationCursor | None = None
        while True:
            page = self.publications.query_runtime_publication_recovery(
                kind="process_launch",
                state=state,
                operation_reconciled=operation_reconciled,
                after=after,
                limit=page_size,
            )
            previous = after
            for publication in page.records:
                cursor = RuntimePublicationCursor(
                    publication["created_at"],
                    publication["publication_id"],
                )
                if (
                    publication["kind"] != "process_launch"
                    or publication["state"] != state
                    or publication["operation_reconciled"] is not operation_reconciled
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "runtime publication repository returned an invalid launch recovery page"
                    )
                recovered_id = self._recover_launch_publication(publication)
                if recovered_id is not None and len(recovered) < page_size:
                    recovered.append(recovered_id)
                previous = cursor
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "runtime publication repository returned an invalid launch recovery cursor"
                )
            after = page.next_cursor

    def _recover_launch_publication(
        self,
        publication: dict[str, Any],
    ) -> str | None:
        if publication["state"] == "manual":
            self._fail_closed_manual_launch_publication(publication)
        with self.store.transaction():
            claimed = self.publications.claim_runtime_publication_recovery(
                publication["publication_id"],
                claimant_instance_id=self.owner_instance_id,
                expected_owner_instance_id=publication["owner_instance_id"],
                expected_state=publication["state"],
                classification="compensate_process_launch",
                max_attempts=self.config.runtime.publication_recovery_max_attempts,
                allow_orphaned_claim_takeover=True,
            )
            if claimed is not None:
                claimed_state = str(claimed["state"])
                if claimed_state not in {"rollback_pending", "manual"}:
                    raise ProcessError(
                        "process launch recovery claim entered an invalid state: "
                        f"{claimed_state}"
                    )
                self._redact_root_spawn_initial_goal(
                    claimed,
                    expected_states=(claimed_state,),
                )
        if claimed is None:
            self._require_launch_publication_resolved(publication["publication_id"])
            return None
        if claimed["state"] == "manual":
            self._fail_closed_manual_launch_publication(claimed)
        recovery_lease_id = self._launch_recovery_lease_id(claimed)
        try:
            self._compensate_claimed_launch_publication(
                claimed,
                recovery_lease_id=recovery_lease_id,
            )
        except Exception as exc:
            self._record_launch_recovery_failure(
                claimed,
                recovery_lease_id=recovery_lease_id,
                error=exc,
            )
            raise ProcessError(
                f"cannot compensate process publication {claimed['publication_id']}"
            ) from exc
        return str(claimed["publication_id"])

    def _compensate_claimed_launch_publication(
        self,
        claimed: dict[str, Any],
        *,
        recovery_lease_id: str,
    ) -> None:
        publication_id = str(claimed["publication_id"])
        with self.store.transaction(include_object_payloads=True):
            current = self.publications.get_runtime_publication(publication_id)
            if current is None:
                raise ProcessError(
                    f"process publication disappeared: {publication_id}"
                )
            if self._launch_recovery_lease_id(current) != recovery_lease_id:
                raise ProcessError(
                    f"process publication recovery lease changed: {publication_id}"
                )
            self._cleanup_failed_launch_strict(current)
            self._redact_root_spawn_initial_goal(
                current,
                expected_states=("rollback_pending",),
            )
            if not self.publications.advance_runtime_publication(
                publication_id,
                state="rolled_back",
                phase="startup_compensated",
                receipt={"phase": "startup_compensated", "pid": current["pid"]},
                expected_states={"rollback_pending"},
                recovery_lease_id=recovery_lease_id,
            ):
                raise ProcessError(
                    "process publication recovery lease changed before terminal state: "
                    f"{publication_id}"
                )
            rolled_back = self.publications.get_runtime_publication(publication_id)
            if rolled_back is not None:
                self._reconcile_launch_publication_operation(
                    rolled_back,
                    OperationOutcome.FAILED,
                )

    def _record_launch_recovery_failure(
        self,
        claimed: dict[str, Any],
        *,
        recovery_lease_id: str,
        error: Exception,
    ) -> None:
        publication_id = str(claimed["publication_id"])
        with self.store.transaction():
            current = self.publications.get_runtime_publication(publication_id)
            if current is None:
                raise ProcessError(
                    f"process publication disappeared: {publication_id}"
                )
            self._redact_root_spawn_initial_goal(
                current,
                expected_states=("rollback_pending",),
            )
            if not self.publications.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="startup_compensation_failed",
                error={
                    "code": "publication_compensation_failed",
                    "error_type": type(error).__name__,
                },
                expected_states={"rollback_pending"},
                recovery_lease_id=recovery_lease_id,
            ):
                raise ProcessError(
                    "process publication recovery lease changed before failure: "
                    f"{publication_id}"
                ) from error
            failed = self.publications.get_runtime_publication(publication_id)
            if failed is not None:
                self._reconcile_launch_publication_operation(
                    failed,
                    OperationOutcome.UNKNOWN,
                )

    def _fail_closed_manual_launch_publication(
        self,
        publication: dict[str, Any],
    ) -> None:
        with self.store.transaction():
            self._redact_root_spawn_initial_goal(
                publication,
                expected_states=("manual",),
            )
            self._reconcile_launch_publication_operation(
                publication,
                OperationOutcome.UNKNOWN,
            )
        raise ProcessError(
            f"process publication requires manual recovery: {publication['publication_id']}"
        )

    @staticmethod
    def _launch_recovery_lease_id(publication: dict[str, Any]) -> str:
        recovery_lease_id = str(
            (publication["receipt"].get("recovery") or {}).get("lease_id") or ""
        )
        if not recovery_lease_id:
            raise ProcessError(
                "process publication recovery claim has no lease: "
                f"{publication['publication_id']}"
            )
        return recovery_lease_id

    def _require_launch_publication_resolved(self, publication_id: str) -> None:
        current = self.publications.get_runtime_publication(publication_id)
        if current is None or current["state"] in {"committed", "rolled_back"}:
            return
        raise ProcessError(
            f"cannot claim unresolved process publication: {publication_id}"
        )

    def _fail_orphaned_created_processes(self) -> None:
        after: ProcessCursor | None = None
        page_size = self.config.runtime.publication_reconciliation_page_size
        while True:
            page = self.store.query_orphaned_created_processes(
                after=after,
                limit=page_size,
            )
            previous = after
            for process in page.records:
                cursor = ProcessCursor(process.created_at, process.pid)
                if (
                    process.status != ProcessStatus.CREATED
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "process repository returned an invalid orphan recovery page"
                    )
                with self.store.transaction():
                    failed = self.transitions.transition(
                        process.pid,
                        ProcessStatus.FAILED,
                        expected_revision=process.revision,
                        expected_status=ProcessStatus.CREATED,
                        expected_state_generation=process.state_generation,
                        outcome=FailedProcessOutcome(code="orphaned_launch"),
                        status_message="orphaned_launch",
                    )
                    event = self.events.emit_once(
                        self._terminal_event_id(
                            failed,
                            EventType.PROCESS_EXITED,
                        ),
                        EventType.PROCESS_EXITED,
                        source=failed.pid,
                        target=failed.parent_pid,
                        payload={
                            "pid": failed.pid,
                            "status": failed.status.value,
                            "result_oid": None,
                            "reason": "orphaned_launch",
                        },
                        causality={
                            "kind": "process_terminal_state",
                            "pid": failed.pid,
                            "state_generation": failed.state_generation,
                        },
                    )
                    self.audit.record(
                        actor="runtime.recovery",
                        action="orphaned_launch",
                        target=f"process:{process.pid}",
                        decision={
                            "status": "failed",
                            "event_id": event.event_id,
                        },
                    )
                previous = cursor
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "process repository returned an invalid orphan recovery cursor"
                )
            after = page.next_cursor

    def _reconcile_launch_publication_operation(
        self,
        publication: dict[str, Any],
        outcome: OperationOutcome,
    ) -> None:
        operations = self.audit.operations
        operation_id = self._launch_publication_operation_id(
            publication,
            operations,
        )
        if operation_id is None or operations is None:
            if not self.publications.mark_runtime_publication_operation_reconciled(
                str(publication["publication_id"]),
                expected_kind="process_launch",
                expected_state=str(publication["state"]),
                expected_phase=str(publication["phase"]),
                expected_operation_id=None,
            ):
                raise ValidationError(
                    "process launch publication changed while marking unbound reconciliation: "
                    f"{publication['publication_id']}"
                )
            return
        publication_id = str(publication["publication_id"])
        operation = self._evidence.get_operation(operation_id)
        if operation is None:
            raise ValidationError(
                "runtime publication references a missing operation: "
                f"{publication_id} -> {operation_id}"
            )
        expected_name, expected_actor, expected_pid = (
            self._launch_publication_operation_contract(publication, operation)
        )
        if expected_name == "process.spawn" and operation.pid is None:
            operations.set_pid(expected_pid, operation_id=operation_id)
        operations.reconcile_runtime_publication(
            operation_id,
            outcome,
            publication_id=publication_id,
            publication_kind="process_launch",
            publication_state=str(publication["state"]),
            publication_phase=str(publication["phase"]),
            expected_kind="runtime",
            expected_name=expected_name,
            expected_actor=expected_actor,
            expected_pid=expected_pid,
        )

    def _launch_publication_operation_id(
        self,
        publication: dict[str, Any],
        operations: Any | None,
    ) -> str | None:
        plan = publication["plan"]
        operation_id = str(plan.get("operation_id") or "")
        if operations is None:
            if operation_id or plan.get("operation_binding_version") is not None:
                raise ValidationError(
                    "process launch publication cannot resolve its operation manager: "
                    f"{publication['publication_id']}"
                )
            return None
        if operation_id:
            return operation_id
        reverse_bindings = operations.runtime_publication_binding_operation_ids(
            str(publication["publication_id"])
        )
        if plan.get("operation_binding_version") is not None or reverse_bindings:
            raise ValidationError(
                "process launch publication lost its durable operation binding: "
                f"{publication['publication_id']} -> "
                f"{reverse_bindings or '<missing>'}"
            )
        return None

    @staticmethod
    def _launch_publication_operation_contract(
        publication: dict[str, Any],
        operation: Any,
    ) -> tuple[str, str, str]:
        publication_id = str(publication["publication_id"])
        publication_pid = str(publication["pid"])
        plan = publication["plan"]
        if (
            publication["kind"] != "process_launch"
            or str(plan.get("pid") or "") != publication_pid
        ):
            raise ValidationError(
                f"invalid process launch publication identity: {publication_id}"
            )
        launch_kind = str(plan.get("launch_kind") or "")
        if launch_kind == "spawn":
            if plan.get("parent_pid") is not None or operation.pid not in {
                None,
                publication_pid,
            }:
                raise ValidationError(
                    f"invalid root process spawn operation: {publication_id}"
                )
            return "process.spawn", "runtime", publication_pid
        parent_pid = str(plan.get("parent_pid") or "")
        if not parent_pid or launch_kind not in {"fork", "spawn_child"}:
            raise ValidationError(
                f"invalid child process launch operation: {publication_id}"
            )
        return f"process.{launch_kind}", parent_pid, parent_pid

    def reconcile_terminal_publications(self) -> list[str]:
        outcomes = {
            "committed": OperationOutcome.SUCCEEDED,
            "rolled_back": OperationOutcome.FAILED,
            "failed": OperationOutcome.UNKNOWN,
            "manual": OperationOutcome.UNKNOWN,
        }
        reconciled: list[str] = []
        page_size = self.config.runtime.publication_reconciliation_page_size
        for state, outcome in outcomes.items():
            after: RuntimePublicationCursor | None = None
            while True:
                page = self.publications.query_runtime_publication_operation_reconciliation(
                    kind="process_launch",
                    state=state,
                    after=after,
                    limit=page_size,
                )
                previous = after
                for publication in page.records:
                    cursor = RuntimePublicationCursor(
                        publication["created_at"],
                        publication["publication_id"],
                    )
                    if (
                        publication["kind"] != "process_launch"
                        or publication["state"] != state
                        or publication["operation_reconciled"]
                        or (previous is not None and cursor <= previous)
                    ):
                        raise ValidationError(
                            "runtime publication repository returned an invalid launch reconciliation page"
                        )
                    with self.store.transaction():
                        if state in {"rolled_back", "failed", "manual"}:
                            self._redact_root_spawn_initial_goal(
                                publication,
                                expected_states=(state,),
                            )
                        self._reconcile_launch_publication_operation(
                            publication,
                            outcome,
                        )
                    if len(reconciled) < page_size:
                        reconciled.append(str(publication["publication_id"]))
                    previous = cursor
                if page.next_cursor is None:
                    break
                if previous is None or page.next_cursor != previous:
                    raise ValidationError(
                        "runtime publication repository returned an invalid launch reconciliation cursor"
                    )
                after = page.next_cursor
        return reconciled

    def _begin_launch_publication(
        self,
        *,
        pid: str,
        launch_kind: str,
        image_id: str,
        parent_pid: str | None,
    ) -> str:
        publication_id = new_id("publication")
        operations = self.audit.operations
        operation_id = operations.current_id() if operations is not None else None
        if launch_kind == "spawn":
            expected_name = "process.spawn"
            expected_actor = "runtime"
            expected_pid = None
        elif launch_kind in {"fork", "spawn_child"} and parent_pid is not None:
            expected_name = f"process.{launch_kind}"
            expected_actor = parent_pid
            expected_pid = parent_pid
        else:
            raise ValidationError(
                f"invalid process launch publication contract: {launch_kind}"
            )
        with self.store.transaction():
            self.publications.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_launch",
                pid=pid,
                owner_instance_id=self.owner_instance_id,
                plan={
                    "launch_kind": launch_kind,
                    "pid": pid,
                    "parent_pid": parent_pid,
                    "image_id": image_id,
                    "artifact_owner": f"publication:{publication_id}",
                    "operation_id": operation_id,
                    "operation_binding_version": (
                        1 if operation_id is not None else None
                    ),
                },
            )
            if operations is not None and operation_id is not None:
                operations.bind_runtime_publication(
                    operation_id,
                    publication_id=publication_id,
                    publication_kind="process_launch",
                    expected_kind="runtime",
                    expected_name=expected_name,
                    expected_actor=expected_actor,
                    expected_pid=expected_pid,
                )
        return publication_id

    def _begin_controlled_launch(
        self,
        pid: str,
        launch_kind: str,
        image_id: str,
        parent_pid: str | None,
    ) -> tuple[str, contextlib.ExitStack]:
        publication_id = self._begin_launch_publication(
            pid=pid,
            launch_kind=launch_kind,
            image_id=image_id,
            parent_pid=parent_pid,
        )
        control_scope = contextlib.ExitStack()
        control_scope.enter_context(
            trusted_process_control_mutation(
                pid,
                allowed_statuses={ProcessStatus.CREATED},
                reason=f"process.{launch_kind} initializes a new process",
            )
        )
        return publication_id, control_scope

    def _commit_launch_publication(
        self,
        publication_id: str,
        *,
        receipt: dict[str, Any],
    ) -> None:
        """Commit launch evidence and its operation in the caller's transaction."""

        if not self.publications.advance_runtime_publication(
            publication_id,
            state="committed",
            phase="committed",
            receipt=receipt,
            expected_states={"applying"},
        ):
            raise ProcessError(f"cannot commit process publication: {publication_id}")
        publication = self.publications.get_runtime_publication(publication_id)
        if publication is None:
            raise ProcessError(f"process publication disappeared: {publication_id}")
        self._reconcile_launch_publication_operation(
            publication,
            OperationOutcome.SUCCEEDED,
        )

    def _publication_phase(
        self,
        publication_id: str,
        phase: str,
        **receipt: Any,
    ) -> None:
        if not self.publications.advance_runtime_publication(
            publication_id,
            state="applying",
            phase=phase,
            receipt={"phase": phase, **receipt},
            expected_states={"planning", "applying"},
        ):
            raise ProcessError(f"process publication is no longer applicable: {publication_id}")

    def _record_launch_capability_artifacts(
        self,
        publication_id: str,
        pid: str,
        *,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> None:
        for capability in self._authority.list_capabilities(pid):
            if capability.cap_id in exclude_ids:
                continue
            if not self.publications.record_runtime_publication_artifact(
                publication_id,
                {
                    "artifact_id": f"capability:{capability.cap_id}",
                    "kind": "capability",
                    "capability_id": capability.cap_id,
                    "resource": capability.resource,
                },
                expected_states={"planning", "applying"},
            ):
                raise ProcessError(
                    "process publication changed while recording capability: "
                    f"{publication_id}"
                )

    @contextlib.contextmanager
    def _launch_capability_receipts(
        self,
        publication_id: str,
        pid: str,
    ) -> Iterator[None]:
        """Commit every launch-created capability with its exact receipt."""

        with self.memory.ownership_locked(), self.store.transaction(
            include_object_payloads=True
        ):
            existing_ids = frozenset(
                capability.cap_id
                for capability in self._authority.list_capabilities(pid)
            )
            yield
            self._record_launch_capability_artifacts(
                publication_id,
                pid,
                exclude_ids=existing_ids,
            )

    def _rollback_launch_publication(
        self,
        publication_id: str,
        pid: str,
        exc: BaseException,
    ) -> bool:
        """Compensate one failed launch, returning true if it already committed.

        The initial rollback transition is a CAS.  A false result or storage
        exception cannot be treated as an ordinary launch failure: effects may
        already exist while another durable state owns their disposition.
        Re-read that state and either honor a resolved publication or fail
        mutation admission closed until startup recovery can compensate it.
        """

        try:
            with self.store.transaction():
                rollback_started = self.publications.advance_runtime_publication(
                    publication_id,
                    state="rollback_pending",
                    phase="compensating",
                    error={
                        "code": "process_launch_failed",
                        "error_type": type(exc).__name__,
                    },
                    expected_states={"planning", "applying"},
                )
                if rollback_started:
                    rollback_publication = (
                        self.publications.get_runtime_publication(publication_id)
                    )
                    if rollback_publication is None:
                        raise ProcessError(
                            f"process publication disappeared: {publication_id}"
                        )
                    self._redact_root_spawn_initial_goal(
                        rollback_publication,
                        expected_states=("rollback_pending",),
                    )
        except BaseException as transition_error:
            transition_failure = BaseExceptionGroup(
                "process launch and rollback transition failed",
                [exc, transition_error],
            )
            return self._resolve_failed_launch_rollback_transition(
                publication_id,
                transition_failure,
            )
        if not rollback_started:
            transition_failure = BaseExceptionGroup(
                "process launch rollback transition lost",
                [
                    exc,
                    ProcessError(
                        "process publication changed before compensation: "
                        f"{publication_id}"
                    ),
                ],
            )
            return self._resolve_failed_launch_rollback_transition(
                publication_id,
                transition_failure,
            )
        try:
            publication = self.publications.get_runtime_publication(publication_id)
            if publication is None:
                raise ProcessError(f"process publication disappeared: {publication_id}")
            self._cleanup_failed_launch_strict(publication)
        except BaseException as cleanup_exc:
            compensation_failure = BaseExceptionGroup(
                "process launch and compensation failed",
                [exc, cleanup_exc],
            )
            self._fence_failed_launch(publication_id, compensation_failure)
            try:
                with self._recovery_terminalization_scope(publication_id):
                    self._terminalize_launch_publication(
                        publication_id,
                        state="failed",
                        phase="compensation_failed",
                        outcome=OperationOutcome.UNKNOWN,
                        error={
                            "code": "publication_compensation_failed",
                            "error_type": type(cleanup_exc).__name__,
                        },
                    )
            except BaseException as terminal_error:
                self._raise_pending_launch_outcome(
                    publication_id,
                    BaseExceptionGroup(
                        "process launch compensation terminalization failed",
                        [compensation_failure, terminal_error],
                    ),
                    recovery_already_required=True,
                )
            raise compensation_failure from exc
        try:
            self._terminalize_launch_publication(
                publication_id,
                state="rolled_back",
                phase="compensated",
                outcome=OperationOutcome.FAILED,
                receipt={"phase": "compensated", "pid": pid},
            )
        except BaseException as terminal_error:
            terminal_failure = BaseExceptionGroup(
                "process launch compensation terminalization acknowledgement failed",
                [exc, terminal_error],
            )
            return self._resolve_failed_launch_rollback_transition(
                publication_id,
                terminal_failure,
            )
        return False

    def _finish_failed_launch(
        self,
        publication_id: str,
        pid: str,
        error: BaseException,
    ) -> str:
        """Return a concurrently committed launch or propagate its failure."""

        if self._rollback_launch_publication(publication_id, pid, error):
            if isinstance(error, Exception):
                return pid
        raise error

    def _resolve_failed_launch_rollback_transition(
        self,
        publication_id: str,
        cause: BaseException,
    ) -> bool:
        """Resolve a lost/failed rollback CAS from its durable publication."""

        resolved_state: str | None = None
        try:
            with self.store.transaction():
                publication = self.publications.get_runtime_publication(
                    publication_id
                )
                if publication is None:
                    raise ProcessError(
                        f"process publication disappeared: {publication_id}"
                    )
                state = str(publication["state"])
                if state == "committed":
                    self._reconcile_launch_publication_operation(
                        publication,
                        OperationOutcome.SUCCEEDED,
                    )
                    resolved_state = state
                elif state == "rolled_back":
                    self._reconcile_launch_publication_operation(
                        publication,
                        OperationOutcome.FAILED,
                    )
                    resolved_state = state
        except BaseException as resolution_error:
            # A failed read or terminal-operation reconciliation leaves the
            # durable outcome unknown to this Runtime.  Fence before reporting
            # the diagnostic failure so no later mutation can build on it.
            resolution_failure = BaseExceptionGroup(
                "process launch rollback resolution failed",
                [cause, resolution_error],
            )
            self._fence_failed_launch(publication_id, resolution_failure)
            if not isinstance(resolution_failure, Exception):
                raise resolution_failure from cause
            raise ProcessError(
                "cannot resolve process publication after rollback transition "
                f"failure: {publication_id}"
            ) from resolution_failure

        if resolved_state == "committed":
            if not isinstance(cause, Exception):
                raise cause
            return True
        if resolved_state == "rolled_back":
            raise cause

        # Planning/applying/rollback_pending as well as retryable failed/manual
        # states still own partial launch artifacts.  Keep the linked operation
        # pending when its exact current binding is intact, and always poison
        # mutation admission before performing diagnostic association reads.
        self._fence_failed_launch(publication_id, cause)
        self._raise_pending_launch_outcome(
            publication_id,
            cause,
            recovery_already_required=True,
        )
        raise AssertionError("pending launch publication signal did not raise")

    def _terminalize_launch_publication(
        self,
        publication_id: str,
        *,
        state: str,
        phase: str,
        outcome: OperationOutcome,
        receipt: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with self.store.transaction():
            current = self.publications.get_runtime_publication(publication_id)
            if current is None:
                raise ProcessError(
                    f"process publication disappeared: {publication_id}"
                )
            self._redact_root_spawn_initial_goal(
                current,
                expected_states=("rollback_pending",),
            )
            if not self.publications.advance_runtime_publication(
                publication_id,
                state=state,
                phase=phase,
                receipt=receipt,
                error=error,
                expected_states={"rollback_pending"},
            ):
                raise ProcessError(
                    "process publication changed during terminalization: "
                    f"{publication_id}"
                )
            publication = self.publications.get_runtime_publication(publication_id)
            if publication is None:
                raise ProcessError(
                    f"process publication disappeared: {publication_id}"
                )
            self._reconcile_launch_publication_operation(publication, outcome)

    def _mark_launch_recovery_required(self, publication_id: str) -> None:
        if self._recovery_required_callback is not None:
            self._recovery_required_callback(publication_id=publication_id)

    def _fence_failed_launch(
        self,
        publication_id: str,
        cause: BaseException,
    ) -> None:
        """Preserve both the launch failure and any fence interruption."""

        try:
            self._mark_launch_recovery_required(publication_id)
        except BaseException as fence_error:
            fence_failure = BaseExceptionGroup(
                "process launch recovery fence failed",
                [cause, fence_error],
            )
            fallback = self._recovery_required_fallback
            if fallback is None:
                raise fence_failure from cause
            try:
                fallback(publication_id=publication_id)
            except BaseException as fallback_error:
                raise BaseExceptionGroup(
                    "process launch recovery fence and fallback failed",
                    [fence_failure, fallback_error],
                ) from cause
            raise fence_failure from cause

    def _raise_pending_launch_outcome(
        self,
        publication_id: str,
        cause: BaseException,
        *,
        recovery_already_required: bool = False,
    ) -> None:
        try:
            publication = self.publications.get_runtime_publication(publication_id)
        except BaseException as diagnostic_error:
            diagnostic_failure = BaseExceptionGroup(
                "process launch publication diagnostic failed",
                [cause, diagnostic_error],
            )
            if not recovery_already_required:
                self._fence_failed_launch(publication_id, diagnostic_failure)
            self._raise_launch_failure_preserving_control_flow(
                f"cannot inspect pending process publication: {publication_id}",
                diagnostic_failure,
                cause=cause,
            )
        unresolved = publication is None or publication["state"] not in {
            "committed",
            "rolled_back",
        }
        if unresolved and not recovery_already_required:
            self._fence_failed_launch(publication_id, cause)
        if publication is not None and publication["state"] in {
            "planning",
            "applying",
            "rollback_pending",
        }:
            operation_id = str(publication["plan"].get("operation_id") or "")
            operations = self.audit.operations
            current_operation_id = (
                str(operations.current_id() or "")
                if operations is not None
                else ""
            )
            try:
                operation = (
                    self._evidence.get_operation(operation_id)
                    if (
                        operations is not None
                        and operation_id
                        and operation_id == current_operation_id
                    )
                    else None
                )
            except BaseException as diagnostic_error:
                diagnostic_failure = BaseExceptionGroup(
                    "process launch operation diagnostic failed",
                    [cause, diagnostic_error],
                )
                self._raise_launch_failure_preserving_control_flow(
                    "cannot inspect pending process publication operation: "
                    f"{publication_id}",
                    diagnostic_failure,
                    cause=cause,
                )
            if operation is not None and operation.state != OperationState.TERMINAL:
                pending = RuntimePublicationPending(
                    publication_id=publication_id,
                    operation_id=operation_id,
                    state=str(publication["state"]),
                    phase=str(publication["phase"]),
                )
                if not isinstance(cause, Exception):
                    raise BaseExceptionGroup(
                        "process launch interruption remains publication pending",
                        [cause, pending],
                    ) from cause
                raise pending from cause
        self._raise_launch_failure_preserving_control_flow(
            f"cannot terminalize process publication: {publication_id}",
            cause,
            cause=cause,
        )

    @staticmethod
    def _raise_launch_failure_preserving_control_flow(
        message: str,
        failure: BaseException,
        *,
        cause: BaseException,
    ) -> NoReturn:
        if not isinstance(failure, Exception):
            if cause is failure:
                raise failure
            raise failure from cause
        raise ProcessError(message) from failure

    def _cleanup_failed_launch_strict(self, publication: dict[str, Any]) -> None:
        pid = str(publication["pid"])
        if self._failed_launch_artifact_cleanup is not None:
            self._failed_launch_artifact_cleanup(publication)
        self.memory.release_process_owned(pid)
        namespace = self.memory.process_namespace(pid)
        namespace_resource = f"object_namespace:{namespace}"
        self.store.delete_process_scaffold(
            pid,
            namespace=namespace,
            namespace_resource=namespace_resource,
        )
        if self._failed_launch_artifact_cleanup is not None:
            self._failed_launch_artifact_cleanup(publication)

    def finalize_exec_capability_revocations(
        self,
        pid: str,
        rollback_token: str,
    ) -> None:
        self.capabilities.finalize_exec_revocations(
            pid,
            rollback_token=rollback_token,
        )

    def cleanup_failed_launch(self, pid: str) -> None:
        """Remove partial process state created by a failed external launch."""
        publications = [
            publication
            for publication in self.publications.list_runtime_publications(pid=pid)
            if publication["kind"] == "process_launch"
        ]
        if publications and self._failed_launch_artifact_cleanup is not None:
            self._failed_launch_artifact_cleanup(publications[-1])
        self._cleanup_failed_launch(pid)

    def _release_rejected_exit_result(self, pid: str, result: ObjectHandle | None) -> None:
        if result is None:
            return
        obj = self.objects.get_object(result.oid)
        if obj is None or obj.owner_kind != ObjectOwnerKind.PROCESS or obj.owner_id != pid:
            return
        self.memory.delete_object_trusted("process", result.oid, reason="terminal_exit_rejected")

    def _require_child(self, parent: str, child: str) -> AgentProcess:
        self._get(parent)
        child_proc = self._get(child)
        if child_proc.parent_pid != parent:
            raise ProcessError(f"{child} is not a child of {parent}")
        return child_proc

    def _require_child_budget(self, parent: AgentProcess) -> None:
        if self.resources is not None:
            return
        if parent.resource_budget.max_child_processes is None:
            return
        child_count = len(self.store.list_child_processes(parent.pid))
        if child_count >= parent.resource_budget.max_child_processes:
            raise ProcessError(
                f"process {parent.pid} exhausted child process budget: "
                f"{child_count}/{parent.resource_budget.max_child_processes}"
            )

    def _add_handle_to_process_view(self, process: AgentProcess, handle: ObjectHandle) -> AgentProcess:
        process = self._get(process.pid)
        if process.memory_view is None:
            process.memory_view = self.memory.create_view(process.pid, [handle], mode=ViewMode.READ_ONLY)
            process.updated_at = utc_now()
            return self.store.patch_process(
                process.pid,
                {"memory_view": process.memory_view, "updated_at": process.updated_at},
                expected_revision=process.revision,
            )
        return self.store.append_process_memory_roots(process.pid, [handle])

    def add_handle_to_process_view(
        self,
        pid: str,
        handle: ObjectHandle,
    ) -> AgentProcess:
        """Publish one handle through the Process repository's CAS path."""

        return self._add_handle_to_process_view(self._get(pid), handle)

    def _wake_parent_waiting_on_child(self, child: AgentProcess) -> None:
        if child.parent_pid is None:
            return
        parent = self.store.get_process(child.parent_pid)
        if parent is None:
            return
        if parent.status != ProcessStatus.WAITING_EVENT:
            return
        if not isinstance(parent.wait_state, ChildProcessWait):
            return
        if parent.wait_state.child_pid != child.pid:
            return
        self.transitions.wake(
            self.transitions.wait_token(parent),
            control=True,
            reason="terminal child wakes its waiting parent",
        )
        self.audit.record(
            actor="process",
            action="process.wait_wake",
            target=f"process:{parent.pid}",
            decision={"child": child.pid, "child_status": child.status.value},
        )

    def _notify_object_task_process_terminal(self, pid: str) -> None:
        if self._object_task_terminal_notifier is None:
            return
        self._object_task_terminal_notifier(pid)
