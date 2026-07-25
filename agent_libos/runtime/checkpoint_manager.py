from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, MutableMapping, NoReturn

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    CapabilityStatus,
    Checkpoint,
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptState,
    ChildProcessWait,
    DataFlowContext,
    DataLabels,
    EventType,
    ExitedProcessOutcome,
    FailedProcessOutcome,
    HumanRequestStatus,
    HumanProcessWait,
    HostResumeProcessWait,
    JITRehydrationArtifact,
    KilledProcessOutcome,
    MessageProcessWait,
    ObjectOwnerKind,
    ObjectTaskStatus,
    ObjectType,
    OperationOutcome,
    ProcessMessageStatus,
    ProcessStatus,
    ProcessWaitState,
    PausedProcessWait,
    ResourceBudget,
    ResourceUsage,
    ToolHandle,
    ToolProcessWait,
    legacy_status_message,
    process_outcome_from_json,
    process_outcome_to_mapping,
    process_state_to_mapping,
    process_wait_state_from_json,
    process_wait_state_to_mapping,
    remap_process_outcome,
    remap_process_wait_state,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ProcessError, ValidationError
from agent_libos.ports import (
    CheckpointMessagePort,
    CheckpointRestorePublicationWriterPort,
    RuntimePublicationOperationPort,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.checkpoint_reconciliation import CheckpointRestoreReconciler
from agent_libos.runtime.event_bus import EventBus
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.evidence import external_effect_summary, external_effect_to_json
from agent_libos.runtime.snapshots import (
    ProcessSnapshot,
    SnapshotCodec,
    SnapshotCoordinator,
    SnapshotIdentityMap,
    SnapshotRemapper,
    SnapshotRows,
)
from agent_libos.storage import StoreAssemblyReadiness, UnitOfWork
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, loads, to_jsonable


@dataclass
class _CheckpointForkPublication:
    actor: str
    checkpoint: Checkpoint
    snapshot: dict[str, Any]
    parent_pid: str | None
    require_capability: bool
    remapped: dict[str, Any]
    fork_rows: SnapshotRows
    object_payloads: Mapping[str, Any]
    root_pid: str
    restored_image_ids: list[str]
    publication_state: dict[str, Any]
    post_commit_failures: list[dict[str, str]]


class CheckpointManager:
    """Durable checkpoints for reconstructable AgentProcess runtime state.

    Checkpoints deliberately do not roll back audit records, LLM calls, events,
    or external provider side effects. Provider-decided external effect records
    are reported during diff/restore, while restore edits only the scoped
    process subtree state that can be reconstructed from the checkpoint payload.
    """

    PROCESS_RESOURCE_PREFIX = "checkpoint:process:"
    CHECKPOINT_RESOURCE_PREFIX = "checkpoint:"
    HISTORY_TABLES = {"audit_records", "events", "llm_calls", "checkpoints", "external_effects"}
    RESTORE_EXTERNAL_POLICY = "report_only"
    TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
    ACTIVE_OBJECT_TASK_STATUSES = {
        ObjectTaskStatus.QUEUED,
        ObjectTaskStatus.RUNNING,
        ObjectTaskStatus.WAITING_HUMAN,
        ObjectTaskStatus.WAITING_PROCESS,
        ObjectTaskStatus.WAITING_MESSAGE,
    }
    FORK_TRANSIENT_STATUSES = {
        ProcessStatus.RUNNING.value,
        ProcessStatus.WAITING_EVENT.value,
        ProcessStatus.WAITING_TOOL.value,
        ProcessStatus.WAITING_HUMAN.value,
    }
    FORK_COMMITTED_RECEIPT_ATTRIBUTE = "checkpoint_fork_receipt"

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        audit: AuditManager,
        events: EventBus,
        capabilities: CapabilityManager | None = None,
        scheduler: Any | None = None,
        registry_lifecycle_lock: Any | None = None,
        memory: Any | None = None,
        images: MutableMapping[str, AgentImage] | None = None,
        authority_manifests: Any | None = None,
        tools: Any | None = None,
        resources: Any | None = None,
        *,
        messages: CheckpointMessagePort,
        operations: RuntimePublicationOperationPort,
        owner_instance_id: str,
        checkpoint_publication_writer: CheckpointRestorePublicationWriterPort,
        recovery_required_callback: Any | None = None,
        require_recovery_lease: Callable[[], None],
        recovery_terminalization_scope: Callable[
            [str], AbstractContextManager[Any]
        ],
        transitions: ProcessTransitionService | None = None,
        config: AgentLibOSConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self._unit_of_work = unit_of_work
        self._processes = unit_of_work.processes
        self._objects = unit_of_work.objects
        self._authority = unit_of_work.authority
        self._publications = unit_of_work.publications
        self._evidence = unit_of_work.evidence
        self._extensions = unit_of_work.extensions
        self.audit = audit
        self.events = events
        self.capabilities = capabilities
        self._scheduler = scheduler
        self._registry_lifecycle_lock = registry_lifecycle_lock
        self._memory = memory
        self._images = images if images is not None else {}
        self._authority_manifests = authority_manifests
        self._tools = tools
        self._resources = resources
        self._messages = messages
        self._operations = operations
        self._owner_instance_id = str(owner_instance_id)
        self._recovery_required_callback = recovery_required_callback
        self._require_recovery_lease = require_recovery_lease
        self._process_transitions = transitions or ProcessTransitionService(
            self._processes
        )
        self._snapshot_rows = unit_of_work.snapshots
        self._modules: Any | None = None
        self._image_registry: Any | None = None
        self.snapshots = SnapshotCoordinator(unit_of_work)
        self._restore_single_flight_lock = threading.Lock()
        self._preflight_artifact: ContextVar[
            tuple[str, Checkpoint, dict[str, Any], ProcessSnapshot] | None
        ] = ContextVar(
            f"agent_libos_checkpoint_preflight_{id(self)}",
            default=None,
        )
        # Resolve phase methods through lambdas so runtime instrumentation and
        # deterministic fault injection still observe the live implementation.
        self._restore_reconciler = CheckpointRestoreReconciler(
            store=self._publications,
            writer=checkpoint_publication_writer,
            operations=self._operations,
            owner_instance_id=self._owner_instance_id,
            recovery_max_attempts=self.config.runtime.publication_recovery_max_attempts,
            reconciliation_page_size=(
                self.config.runtime.publication_reconciliation_page_size
            ),
            registry_scope=self._runtime_registry_lifecycle_quiescent,
            load_checkpoint=lambda checkpoint_id: self._read_checkpoint_artifact(
                checkpoint_id
            ),
            restore_object_payloads=lambda snapshot: self._snapshot_rows.reconcile_checkpoint_object_payloads(
                SnapshotCodec.decode_mapping(snapshot)
            ),
            restore_images=lambda snapshot: self._restore_images(snapshot),
            restore_jit_sources=lambda snapshot: self._restore_jit_sources(snapshot),
            prune_jit_tools=lambda tool_ids, scoped_pids: self._prune_stale_ephemeral_jit_tools(
                tool_ids,
                scoped_pids=scoped_pids,
            ),
            run_finalizer=self._run_durable_restore_finalizer,
            record_failure=lambda actor, checkpoint, phase, exc: self._restore_post_commit_failure(
                actor=actor,
                checkpoint=checkpoint,
                phase=phase,
                exc=exc,
            ),
            require_recovery_lease=require_recovery_lease,
            recovery_required=recovery_required_callback,
            recovery_terminalization_scope=recovery_terminalization_scope,
        )

    def bind_modules(self, modules: Any) -> None:
        self._modules = modules

    def bind_image_registry(self, image_registry: Any) -> None:
        self._image_registry = image_registry

    def process_resource(self, pid: str) -> str:
        return f"{self.PROCESS_RESOURCE_PREFIX}{pid}"

    def checkpoint_resource(self, checkpoint_id: str) -> str:
        return f"{self.CHECKPOINT_RESOURCE_PREFIX}{checkpoint_id}"

    def grant_process_defaults(
        self,
        pid: str,
        *,
        issued_by: str = "checkpoint.process",
        publication_id: str | None = None,
    ) -> Capability | None:
        if self.capabilities is None:
            return None
        with self._unit_of_work.transaction():
            capability = self.capabilities.grant(
                subject=pid,
                resource=self.process_resource(pid),
                rights=[CapabilityRight.READ, CapabilityRight.WRITE],
                issued_by=issued_by,
            )
            if publication_id is not None and not self._publications.record_runtime_publication_artifact(
                publication_id,
                {
                    "artifact_id": f"capability:{capability.cap_id}",
                    "kind": "capability",
                    "capability_id": capability.cap_id,
                    "resource": capability.resource,
                },
                expected_states={"planning", "applying"},
            ):
                raise ValidationError(
                    "runtime publication changed while recording checkpoint authority: "
                    f"{publication_id}"
                )
        return capability

    def create(
        self,
        pid: str,
        reason: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # Module discovery acquires the shared registry lifecycle lock. Take it
        # before any capability/store access so checkpoint creation follows the
        # same registry -> store order as module load/unload.
        with self._runtime_registry_lifecycle_quiescent():
            return self._create_registry_locked(
                pid,
                reason,
                actor=actor,
                require_capability=require_capability,
                metadata=metadata,
            )

    def _create_registry_locked(
        self,
        pid: str,
        reason: str,
        *,
        actor: str | None,
        require_capability: bool,
        metadata: dict[str, Any] | None,
    ) -> str:
        selected_actor = actor or pid
        if require_capability:
            self._require_process_right(selected_actor, pid, CapabilityRight.WRITE)
        checkpoint_id = new_id("ckpt")
        created_at = utc_now()
        # Snapshot discovery spans process, object, capability, message, image,
        # and in-memory object-payload state. Keep every read plus the durable
        # checkpoint/head write behind one store transaction so a concurrent
        # mutation cannot produce a row from one instant and a payload from
        # another.
        with self.snapshots.capture_scope():
            snapshot = self._build_snapshot(
                checkpoint_id=checkpoint_id,
                pid=pid,
                reason=reason,
                created_at=created_at,
                created_by=selected_actor,
            )
            snapshot_bytes = len(dumps(snapshot).encode("utf-8"))
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                pid=pid,
                reason=reason,
                created_at=created_at,
                created_by=selected_actor,
                snapshot_version=SnapshotCodec.schema_version,
                metadata={
                    **(metadata or {}),
                    "subtree_pids": snapshot["subtree_pids"],
                    "object_count": len(snapshot["object_payloads"]),
                    "module_count": len(snapshot.get("modules", [])),
                    "snapshot_bytes": snapshot_bytes,
                },
                effect_ledger_seq=self._evidence.current_effect_ledger_seq(),
            )
            if snapshot_bytes > self.config.checkpoint.snapshot_hard_limit_bytes:
                raise ValidationError(
                    "checkpoint snapshot exceeded "
                    f"snapshot_hard_limit_bytes={self.config.checkpoint.snapshot_hard_limit_bytes}"
                )
            self._snapshot_rows.insert_checkpoint(
                checkpoint,
                SnapshotCodec.decode_mapping(snapshot),
            )
            process = self._processes.get_process(pid)
            if process is not None:
                self._processes.patch_process(
                    pid,
                    {"checkpoint_head": checkpoint_id},
                    expected_revision=process.revision,
                )
            # The checkpoint row/head, its initial read authority, and its
            # creation evidence are one API outcome.  Keeping these writes in
            # the snapshot transaction prevents a caller-visible failure from
            # leaving behind a committed checkpoint with no returned id.
            if self.capabilities is not None:
                self.capabilities.grant(
                    subject=pid,
                    resource=self.checkpoint_resource(checkpoint_id),
                    rights=[CapabilityRight.READ],
                    issued_by="checkpoint.create",
                )
            self.events.emit(
                EventType.CHECKPOINT_CREATED,
                source=selected_actor,
                target=pid,
                payload={"checkpoint_id": checkpoint_id, "reason": reason, "subtree_pids": snapshot["subtree_pids"]},
            )
            self.audit.record(
                actor=selected_actor,
                action="checkpoint.create",
                target=self.checkpoint_resource(checkpoint_id),
                decision={
                    "reason": reason,
                    "pid": pid,
                    "subtree_pids": snapshot["subtree_pids"],
                    "snapshot_bytes": snapshot_bytes,
                },
            )
        return checkpoint_id

    def checkpoint(self, pid: str, reason: str) -> str:
        return self.create(pid, reason, actor=pid, require_capability=False)

    def list(
        self,
        pid: str | None = None,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        selected_limit = self._bounded_list_limit(limit)
        if require_capability and actor is not None:
            if pid is None:
                self._require_checkpoint_right(actor, "*", CapabilityRight.READ)
            else:
                self._require_process_right(actor, pid, CapabilityRight.READ)
        return [
            self._checkpoint_summary(item)
            for item in self._snapshot_rows.list_checkpoints(
                pid=pid,
                limit=selected_limit,
            )
        ]

    def _bounded_list_limit(self, limit: int | None) -> int:
        selected = self.config.checkpoint.list_limit if limit is None else limit
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValidationError("checkpoint list limit must be an integer")
        if selected < 1:
            raise ValidationError("checkpoint list limit must be >= 1")
        if selected > self.config.checkpoint.list_limit:
            raise ValidationError(
                f"checkpoint list limit exceeds configured maximum {self.config.checkpoint.list_limit}"
            )
        return selected

    def inspect(
        self,
        checkpoint_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability and actor is not None:
            with self._checkpoint_or_process_read_scope(actor, checkpoint):
                return self.inspect(checkpoint_id, actor=actor, require_capability=False)
        return {
            "checkpoint": self._checkpoint_summary(checkpoint),
            "snapshot_version": snapshot.get("version"),
            "subtree_pids": list(snapshot.get("subtree_pids", [])),
            "modules": list(snapshot.get("modules", [])),
            "counts": self._snapshot_counts(snapshot),
            "processes": [
                self._snapshot_process_summary(row)
                for row in snapshot["rows"].get("processes", [])
            ],
        }

    @staticmethod
    def _snapshot_process_summary(row: dict[str, Any]) -> dict[str, Any]:
        wait_state = process_wait_state_from_json(row["wait_state_json"])
        outcome = process_outcome_from_json(row["outcome_json"])
        return {
            "pid": row["pid"],
            "parent_pid": row.get("parent_pid"),
            "image_id": row["image_id"],
            "working_directory": row.get("working_directory", "."),
            "goal_oid": row.get("goal_oid"),
            **process_state_to_mapping(
                row["status"],
                wait_state,
                outcome,
                row["state_generation"],
            ),
        }

    def diff(
        self,
        checkpoint_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability and actor is not None:
            with self._checkpoint_or_process_read_scope(actor, checkpoint):
                return self.diff(checkpoint_id, actor=actor, require_capability=False)
        current = self._build_current_state_for_diff(snapshot)
        tables: dict[str, Any] = {}
        for table in [
            "processes",
            "objects",
            "capabilities",
            "process_resource_reservations",
            "process_messages",
            "llm_pending_actions",
            "tool_candidates",
            "skills",
        ]:
            before = self._index_rows(table, snapshot["rows"].get(table, []))
            after = self._index_rows(table, current.get(table, []))
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
            tables[table] = {
                "added": added[: self.config.checkpoint.diff_preview_items],
                "removed": removed[: self.config.checkpoint.diff_preview_items],
                "changed": changed[: self.config.checkpoint.diff_preview_items],
                "added_count": len(added),
                "removed_count": len(removed),
                "changed_count": len(changed),
            }
        external_effect_records = self._external_effect_records_since(
            checkpoint,
            snapshot=snapshot,
        )
        return {
            "checkpoint_id": checkpoint_id,
            "pid": checkpoint.pid,
            "tables": tables,
            "external_effects_since_checkpoint": [
                external_effect_to_json(record) for record in external_effect_records
            ],
            "external_effect_summary": external_effect_summary(external_effect_records),
            "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
        }

    def restore(
        self,
        actor: str,
        checkpoint_id: str,
        *,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        with self._restore_single_flight():
            return self._restore_once(
                actor,
                checkpoint_id,
                require_capability=require_capability,
            )

    def _restore_once(
        self,
        actor: str,
        checkpoint_id: str,
        *,
        require_capability: bool,
    ) -> dict[str, Any]:
        with self.snapshots.restore_runtime_scope(
            self._runtime_quiescent_for_restore,
        ):
            with self.snapshots.restore_registry_scope(
                self._runtime_registry_lifecycle_quiescent,
            ):
                checkpoint, snapshot, typed_snapshot = (
                    self._load_checkpoint_typed(checkpoint_id)
                )
                # Fail closed on module identity before opening any authority
                # or main-state publication transaction. The registry lock is
                # already held, so a matching module cannot be replaced before
                # the transactional recheck in publication preparation.
                self._require_snapshot_modules(snapshot)
                publication_id = self._restore_reconciler.new_publication_id()
                with self.snapshots.restore_atomic_scope(
                    self._runtime_object_ownership_quiescent,
                ):
                    restore_right_decisions = self._restore_right_decisions(
                        actor,
                        checkpoint_id,
                        snapshot,
                        require_capability=require_capability,
                    )
                    # AuthorityTransaction is deliberately the outer UoW. Its
                    # exit settles finite-use authority only after the snapshot
                    # rows and core evidence have been written; any prepare,
                    # publish, sink, link, or settlement failure rolls all of
                    # them back, including in-memory Object payload journals.
                    with self._restore_main_commit_scope(
                        restore_right_decisions,
                        actor=actor,
                        publication_id=publication_id,
                    ):
                        prepared, published = self.snapshots.atomic_publish(
                            typed_snapshot,
                            prepare=lambda typed: self._prepare_restore_publication(
                                checkpoint,
                                typed,
                                snapshot=snapshot,
                                actor=actor,
                                publication_id=publication_id,
                            ),
                            publish=lambda typed, plan: self._publish_restore_rows(
                                checkpoint,
                                typed,
                                plan,
                            ),
                        )
                        current_pids = prepared["current_pids"]
                        snapshot_pids = prepared["snapshot_pids"]
                        stale_tool_ids = prepared["stale_tool_ids"]
                        external_effects = prepared["external_effects"]
                        external_effect_summary = prepared["external_effect_summary"]
                        (
                            cancelled_human_requests,
                            superseded_messages,
                            superseded_object_tasks,
                            release_finalizer_objects,
                        ) = published
                        self._record_restore_commit_evidence(
                            actor=actor,
                            checkpoint=checkpoint,
                            snapshot_pids=snapshot_pids,
                            current_pids=current_pids,
                            cancelled_human_requests=cancelled_human_requests,
                            superseded_messages=superseded_messages,
                            superseded_object_tasks=superseded_object_tasks,
                            external_effects=external_effects,
                            external_effect_summary=external_effect_summary,
                        )
                        self._restore_reconciler.mark_main_state_committed(
                            publication_id
                        )
                # Registry reconciliation is post-commit but remains inside the
                # lifecycle lock, so module rollback cannot interleave with the
                # image and JIT cache/store updates.
                post_commit_failures = self._restore_reconciler.reconcile_online_registry(
                    publication_id
                )
            # Terminal reconciliation and external release hooks are deliberately
            # outside all registry,
            # ownership, and store locks because they may call host code.
            if not post_commit_failures:
                post_commit_failures.extend(
                    self._restore_reconciler.reconcile_online_finalizers(
                        publication_id
                    )
                )
            if not post_commit_failures:
                self._restore_reconciler.finish_online(publication_id)
            status = self._restore_status(post_commit_failures)
            return {
                "checkpoint_id": checkpoint_id,
                "publication_id": publication_id,
                "pid": checkpoint.pid,
                "status": status,
                "main_state_committed": True,
                "reconciliation_pending": bool(post_commit_failures),
                "post_commit_failures": post_commit_failures,
                "restored_pids": snapshot_pids,
                "previous_pids": current_pids,
                "cancelled_human_requests": cancelled_human_requests,
                "superseded_messages": superseded_messages,
                "superseded_object_tasks": superseded_object_tasks,
                "external_effects_since_checkpoint": external_effects,
                "external_effect_summary": external_effect_summary,
                "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
            }

    @contextmanager
    def _restore_single_flight(self):
        if not self._restore_single_flight_lock.acquire(blocking=False):
            raise ValidationError(
                "checkpoint restore or recovery is already in progress"
            )
        try:
            yield
        finally:
            self._restore_single_flight_lock.release()

    def _restore_right_decisions(
        self,
        actor: str,
        checkpoint_id: str,
        snapshot: dict[str, Any],
        *,
        require_capability: bool,
    ) -> list[CapabilityDecision]:
        decisions: list[CapabilityDecision] = []
        if not require_capability:
            return decisions
        checkpoint_decision = self._require_checkpoint_right(
            actor,
            checkpoint_id,
            CapabilityRight.ADMIN,
            consume=False,
        )
        if checkpoint_decision is not None:
            decisions.append(checkpoint_decision)
        decisions.extend(
            self._require_snapshot_image_restore_rights(
                actor,
                snapshot,
                overwrite_existing=True,
                consume=False,
            )
        )
        return decisions

    @contextmanager
    def _restore_main_commit_scope(
        self,
        decisions: Iterable[CapabilityDecision],
        *,
        actor: str,
        publication_id: str,
    ):
        try:
            with self._restore_authority_transaction(
                decisions,
                actor=actor,
            ):
                yield
        except BaseException as exc:
            self._restore_reconciler.handle_main_commit_scope_escape(
                publication_id,
                exc,
            )
            raise

    @contextmanager
    def _restore_authority_transaction(
        self,
        decisions: Iterable[CapabilityDecision],
        *,
        actor: str,
    ):
        if self.capabilities is None:
            with self._unit_of_work.transaction(include_object_payloads=True):
                yield
            return
        with self.capabilities.authority_transaction(
            decisions,
            actor=actor,
            operation="checkpoint restore",
        ):
            yield

    def _record_restore_commit_evidence(
        self,
        *,
        actor: str,
        checkpoint: Checkpoint,
        snapshot_pids: list[str],
        current_pids: list[str],
        cancelled_human_requests: list[Any],
        superseded_messages: list[Any],
        superseded_object_tasks: list[Any],
        external_effects: list[Any],
        external_effect_summary: dict[str, Any],
    ) -> None:
        """Write the core restore event and audit before authority settlement."""

        self.events.emit(
            EventType.ROLLBACK,
            source=actor,
            target=checkpoint.pid,
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "restored_pids": snapshot_pids,
                "superseded_object_tasks": superseded_object_tasks,
                "external_effects_since_checkpoint": len(external_effects),
                "external_effect_summary": external_effect_summary,
                "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
                "main_state_committed": True,
            },
        )
        self.audit.record(
            actor=actor,
            action="checkpoint.restore",
            target=self.checkpoint_resource(checkpoint.checkpoint_id),
            decision={
                "restored_for": checkpoint.pid,
                "restored_pids": snapshot_pids,
                "previous_pids": current_pids,
                "cancelled_human_requests": cancelled_human_requests,
                "superseded_messages": superseded_messages,
                "superseded_object_tasks": superseded_object_tasks,
                "external_effects_since_checkpoint": external_effects,
                "external_effect_summary": external_effect_summary,
                "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
                "main_state_committed": True,
            },
        )

    def _prepare_restore_publication(
        self,
        checkpoint: Checkpoint,
        typed: ProcessSnapshot,
        *,
        snapshot: dict[str, Any] | None = None,
        actor: str,
        publication_id: str,
    ) -> dict[str, Any]:
        snapshot = snapshot or self.snapshots.encode(typed)
        self._require_snapshot_modules(snapshot)
        self._validate_snapshot_flow_rows(snapshot)
        current_pids = self._subtree_pids(checkpoint.pid)
        snapshot_pids = list(typed.subtree_pids)
        self._reject_active_object_tasks_for_restore(snapshot, current_pids)
        self._validate_snapshot_restore_assets(snapshot)
        stale_tool_ids = self._stale_ephemeral_tool_ids_for_restore(
            snapshot,
            current_pids,
        )
        current_owned_jit_ids = (
            self._snapshot_rows.registered_jit_tool_ids_for_processes(
                current_pids
            )
        )
        snapshot_owned_jit_ids = self._registered_tool_ids_from_candidate_rows(
            snapshot.get("rows", {}).get("tool_candidates", [])
        )
        stale_tool_ids.update(current_owned_jit_ids - snapshot_owned_jit_ids)
        self._validate_process_local_jit_restore_scope(
            stale_tool_ids | set(current_owned_jit_ids),
            scoped_pids=set(current_pids),
            lookup_pid=checkpoint.pid,
        )
        snapshot_object_oids = self._snapshot_owned_object_oids(snapshot)
        current_object_oids = set(self._current_scoped_object_oids(current_pids))
        finalizer_work_items = self._prepare_durable_restore_finalizers(
            self._object_release_finalizer_objects(
                current_object_oids - snapshot_object_oids
            ),
            publication_id=publication_id,
        )
        effect_pids = self._external_effect_pids(
            checkpoint,
            snapshot=snapshot,
            current_pids=current_pids,
        )
        external_effect_records = self._external_effect_records_since(
            checkpoint,
            pids=effect_pids,
        )
        # The caller owns an outer transaction. The exact work plan and its
        # operation binding are therefore rolled back with any later restore
        # row, evidence, authority-settlement, or payload failure.
        self._restore_reconciler.begin(
            publication_id=publication_id,
            actor=actor,
            checkpoint=checkpoint,
            snapshot=snapshot,
            current_pids=current_pids,
            snapshot_pids=snapshot_pids,
            stale_tool_ids=stale_tool_ids,
            finalizer_work_items=finalizer_work_items,
        )
        return {
            "snapshot": snapshot,
            "current_pids": current_pids,
            "snapshot_pids": snapshot_pids,
            "stale_tool_ids": stale_tool_ids,
            "external_effects": [
                external_effect_to_json(record) for record in external_effect_records
            ],
            "external_effect_summary": external_effect_summary(external_effect_records),
        }

    def recover_incomplete_restore_publications(self) -> list[str]:
        """Resume restore work during startup before general JIT rehydration.

        This entry point is startup-only and rejects calls without the
        Runtime lifecycle's active recovery lease.
        """

        # Preserve the startup-only admission contract before consulting the
        # per-Runtime restore gate. This callback is read-only and must reject
        # an open Runtime without touching publication state.
        self._require_recovery_lease()
        with self._restore_single_flight():
            return self._restore_reconciler.recover_incomplete()

    def reconcile_terminal_restore_publications(self) -> list[str]:
        """Reconcile dirtied terminal restore-operation bindings."""

        with self._restore_single_flight():
            return self._restore_reconciler.reconcile_terminal_publications()

    def _begin_startup_payload_delivery(
        self,
    ) -> CheckpointPayloadDeliveryAttempt | None:
        with self._restore_single_flight():
            return self._restore_reconciler.begin_payload_delivery()

    def _prepare_startup_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        with self._restore_single_flight():
            self._restore_reconciler.prepare_payload_delivery(attempt)

    def _complete_startup_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        with self._restore_single_flight():
            self._restore_reconciler.complete_payload_delivery(attempt)

    def _ack_startup_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        with self._restore_single_flight():
            self._restore_reconciler.ack_payload_delivery(attempt)

    def _get_startup_payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None:
        with self._restore_single_flight():
            return self._restore_reconciler.payload_delivery_attempt_state(attempt)

    def _reopen_startup_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt | None,
    ) -> None:
        with self._restore_single_flight():
            self._restore_reconciler.reopen_payload_delivery(attempt)

    def _publish_restore_rows(
        self,
        checkpoint: Checkpoint,
        typed: ProcessSnapshot,
        plan: dict[str, Any],
    ) -> tuple[Any, Any, Any, Any]:
        snapshot = plan["snapshot"]
        if typed.header.checkpoint_id != checkpoint.checkpoint_id:
            raise ValidationError("checkpoint row and snapshot id do not match")
        return self._restore_scoped_rows(
            snapshot,
            plan["current_pids"],
            checkpoint,
        )

    def rollback(self, pid: str, checkpoint_id: str) -> dict[str, Any]:
        return self.restore(pid, checkpoint_id, require_capability=False)

    def preflight_checkpoint(self, checkpoint_id: str) -> None:
        """Reject incompatible immutable artifacts before operation evidence."""

        self._preflight_artifact.set(None)
        checkpoint, snapshot, typed = self._read_checkpoint_artifact(
            checkpoint_id
        )
        self._preflight_artifact.set(
            (checkpoint_id, checkpoint, snapshot, typed)
        )

    def load_checkpoint_artifact(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, dict[str, Any]]:
        """Return one strictly decoded checkpoint artifact for an internal service."""

        return self._load_checkpoint(checkpoint_id)

    def fork_from_checkpoint(
        self,
        actor: str,
        checkpoint_id: str,
        *,
        parent_pid: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        # Reject same-thread nesting before taking the registry lock. This
        # preserves the global registry -> store lock order while ensuring a
        # savepoint can never be mistaken for a durable fork commit.
        self._reject_nested_checkpoint_fork_transaction()
        # Fork preparation and publication both touch the global JIT/image
        # registries. Keep the lifecycle lock outside every store transaction.
        with self._runtime_registry_lifecycle_quiescent():
            checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
            return self._fork_from_checkpoint_registry_locked(
                actor,
                checkpoint,
                snapshot,
                parent_pid=parent_pid,
                require_capability=require_capability,
            )

    def _fork_from_checkpoint_registry_locked(
        self,
        actor: str,
        checkpoint: Checkpoint,
        snapshot: dict[str, Any],
        *,
        parent_pid: str | None,
        require_capability: bool,
    ) -> dict[str, Any]:
        publication = self._prepare_checkpoint_fork(
            actor,
            checkpoint,
            snapshot,
            parent_pid=parent_pid,
            require_capability=require_capability,
        )
        self._publish_checkpoint_fork(publication)
        self._emit_checkpoint_fork_event(publication)
        self._record_checkpoint_fork_audit(publication)
        return self._fork_publication_committed_receipt(publication)

    def _prepare_checkpoint_fork(
        self,
        actor: str,
        checkpoint: Checkpoint,
        snapshot: dict[str, Any],
        *,
        parent_pid: str | None,
        require_capability: bool,
    ) -> _CheckpointForkPublication:
        # Recheck after crossing the registry boundary so an instrumentation
        # wrapper cannot introduce an outer transaction between the first
        # probe and publication preparation.
        self._reject_nested_checkpoint_fork_transaction()
        # Preflight without consuming finite authority. The authoritative
        # checks and any finite-use claims happen again in the same transaction
        # that publishes the fork rows, closing revoke/terminal-state TOCTOU
        # windows without charging authority for failed preparation.
        self._validate_fork_parent(
            actor,
            parent_pid,
            require_capability=require_capability,
            consume=False,
        )
        if require_capability:
            self._require_checkpoint_right(
                actor,
                checkpoint.checkpoint_id,
                CapabilityRight.EXECUTE,
                consume=False,
            )
        self._require_snapshot_modules(snapshot)
        self._validate_snapshot_flow_rows(snapshot)
        if require_capability:
            self._require_snapshot_image_restore_rights(
                actor,
                snapshot,
                overwrite_existing=False,
                consume=False,
            )
        remapped = self._remap_snapshot(
            snapshot,
            parent_pid=parent_pid,
            root_pid=checkpoint.pid,
        )
        publication = _CheckpointForkPublication(
            actor=actor,
            checkpoint=checkpoint,
            snapshot=snapshot,
            parent_pid=parent_pid,
            require_capability=require_capability,
            remapped=remapped,
            fork_rows=SnapshotRows.from_mapping(deepcopy(remapped["rows"])),
            object_payloads=deepcopy(remapped["object_payloads"]),
            root_pid=remapped["pid_map"][checkpoint.pid],
            restored_image_ids=[],
            publication_state={"main_state_committed": False},
            post_commit_failures=[],
        )
        try:
            # Prepare in-memory implementations before the transaction that
            # publishes fork process rows. A scheduler can therefore never
            # claim a fork whose process-local JIT handles are not ready.
            self._restore_jit_sources(remapped)
        except BaseException as exc:
            self._cleanup_uncommitted_fork(
                remapped,
                publication.restored_image_ids,
                primary=exc,
            )
            raise
        return publication

    def _publish_checkpoint_fork(
        self,
        publication: _CheckpointForkPublication,
    ) -> None:
        # Keep the store lock across the commit boundary and any lost-ACK
        # diagnosis. Newly committed nonterminal rows remain in CREATED
        # quarantine until their volatile Object payloads are verified.
        with self._unit_of_work.locked():
            self._reject_nested_checkpoint_fork_transaction()
            commit_interruption = self._insert_checkpoint_fork_main_state(
                publication
            )
            published_rows, published_payloads = (
                self._checkpoint_fork_publication_artifacts(publication)
            )
            self._rehydrate_checkpoint_fork_payloads(
                publication,
                published_rows=published_rows,
                published_payloads=published_payloads,
                commit_interruption=commit_interruption,
            )
            self._publish_checkpoint_fork_process_rows(
                publication,
                published_rows=published_rows,
            )
            if commit_interruption is not None and not isinstance(
                commit_interruption,
                Exception,
            ):
                self._attach_fork_receipt(
                    commit_interruption,
                    self._fork_publication_committed_receipt(publication),
                )
                raise commit_interruption

    def _insert_checkpoint_fork_main_state(
        self,
        publication: _CheckpointForkPublication,
    ) -> BaseException | None:
        try:
            self._insert_fork_rows(
                publication.remapped,
                fork_rows=publication.fork_rows,
                object_payloads=publication.object_payloads,
                actor=publication.actor,
                checkpoint_id=publication.checkpoint.checkpoint_id,
                require_capability=publication.require_capability,
                fork_parent_pid=publication.parent_pid,
                fork_root_pid=publication.root_pid,
                image_snapshot=publication.snapshot,
                restored_image_ids=publication.restored_image_ids,
                publication_state=publication.publication_state,
            )
        except BaseException as exc:
            durable_commit: bool | None = (
                True
                if publication.publication_state["main_state_committed"]
                else None
            )
            if durable_commit is None:
                try:
                    durable_commit = self._fork_root_is_persisted(
                        publication.root_pid
                    )
                except BaseException as diagnostic_exc:
                    self._raise_ambiguous_checkpoint_fork_commit(
                        publication,
                        interruption=exc,
                        diagnostic_error=diagnostic_exc,
                    )
            if not durable_commit:
                self._cleanup_uncommitted_fork(
                    publication.remapped,
                    publication.restored_image_ids,
                    primary=exc,
                )
                raise
            publication.post_commit_failures.append(
                self._fork_failure(
                    phase="fork_commit_acknowledgement",
                    exc=exc,
                )
            )
            return exc
        return None

    def _raise_ambiguous_checkpoint_fork_commit(
        self,
        publication: _CheckpointForkPublication,
        *,
        interruption: BaseException,
        diagnostic_error: BaseException,
    ) -> NoReturn:
        failures = [
            self._fork_failure(
                phase="fork_commit_acknowledgement",
                exc=interruption,
            ),
            self._fork_failure(
                phase="fork_commit_confirmation",
                exc=diagnostic_error,
            ),
        ]
        secured = self._secure_checkpoint_fork_subtree(
            remapped=publication.remapped,
            failures=failures,
        )
        if secured == "absent":
            self._cleanup_uncommitted_fork(
                publication.remapped,
                publication.restored_image_ids,
                primary=interruption,
            )
            raise interruption from diagnostic_error
        receipt = (
            self._fork_recovery_required_receipt(
                checkpoint=publication.checkpoint,
                root_pid=publication.root_pid,
                remapped=publication.remapped,
                post_commit_failures=failures,
                subtree_quarantined=(secured == "terminalized"),
            )
            if secured == "terminalized"
            else self._fork_unknown_outcome_receipt(
                checkpoint=publication.checkpoint,
                root_pid=publication.root_pid,
                remapped=publication.remapped,
                interruption=interruption,
                diagnostic_error=diagnostic_error,
            )
        )
        if secured is None:
            self._record_and_fence_checkpoint_fork_recovery(
                actor=publication.actor,
                receipt=receipt,
            )
        else:
            self._record_checkpoint_fork_recovery(
                actor=publication.actor,
                receipt=receipt,
            )
        self._attach_fork_receipt(interruption, receipt)
        raise interruption from diagnostic_error

    def _checkpoint_fork_publication_artifacts(
        self,
        publication: _CheckpointForkPublication,
    ) -> tuple[SnapshotRows, Mapping[str, Any]]:
        published_rows = publication.publication_state.get("fork_rows")
        published_payloads = publication.publication_state.get("object_payloads")
        if isinstance(published_rows, SnapshotRows) and isinstance(
            published_payloads,
            Mapping,
        ):
            return published_rows, published_payloads
        artifact_exc = ValidationError(
            "checkpoint fork committed without a frozen publication artifact"
        )
        recovery_failures = [
            *publication.post_commit_failures,
            self._fork_failure(
                phase="fork_publication_artifact_recovery",
                exc=artifact_exc,
            ),
        ]
        receipt = self._checkpoint_fork_recovery_receipt(
            publication,
            recovery_failures=recovery_failures,
        )
        self._attach_fork_receipt(artifact_exc, receipt)
        raise artifact_exc

    def _rehydrate_checkpoint_fork_payloads(
        self,
        publication: _CheckpointForkPublication,
        *,
        published_rows: SnapshotRows,
        published_payloads: Mapping[str, Any],
        commit_interruption: BaseException | None,
    ) -> None:
        try:
            self._snapshot_rows.rehydrate_checkpoint_fork_object_payloads(
                published_rows,
                object_payloads=published_payloads,
            )
        except BaseException as rehydration_exc:
            recovery_failures = [
                *publication.post_commit_failures,
                self._fork_failure(
                    phase="fork_object_payload_rehydration",
                    exc=rehydration_exc,
                ),
            ]
            receipt = self._checkpoint_fork_recovery_receipt(
                publication,
                recovery_failures=recovery_failures,
            )
            propagated = (
                commit_interruption
                if commit_interruption is not None
                and not isinstance(commit_interruption, Exception)
                else rehydration_exc
            )
            self._attach_fork_receipt(propagated, receipt)
            raise propagated from (
                rehydration_exc
                if propagated is commit_interruption
                else commit_interruption
            )

    def _publish_checkpoint_fork_process_rows(
        self,
        publication: _CheckpointForkPublication,
        *,
        published_rows: SnapshotRows,
    ) -> None:
        try:
            self._snapshot_rows.publish_checkpoint_fork_process_rows(
                published_rows
            )
        except BaseException as publication_exc:
            publication_failure = self._fork_failure(
                phase="fork_process_state_publication",
                exc=publication_exc,
            )
            try:
                published = self._snapshot_rows.checkpoint_fork_process_rows_match(
                    published_rows
                )
            except BaseException as diagnostic_exc:
                recovery_failures = [
                    *publication.post_commit_failures,
                    publication_failure,
                    self._fork_failure(
                        phase="fork_process_state_publication_confirmation",
                        exc=diagnostic_exc,
                    ),
                ]
                receipt = self._checkpoint_fork_recovery_receipt(
                    publication,
                    recovery_failures=recovery_failures,
                )
                self._attach_fork_receipt(publication_exc, receipt)
                raise publication_exc from diagnostic_exc
            if not published:
                recovery_failures = [
                    *publication.post_commit_failures,
                    publication_failure,
                ]
                receipt = self._checkpoint_fork_recovery_receipt(
                    publication,
                    recovery_failures=recovery_failures,
                )
                self._attach_fork_receipt(publication_exc, receipt)
                raise
            publication.post_commit_failures.append(publication_failure)
            if not isinstance(publication_exc, Exception):
                self._attach_fork_receipt(
                    publication_exc,
                    self._fork_publication_committed_receipt(publication),
                )
                raise

    def _checkpoint_fork_recovery_receipt(
        self,
        publication: _CheckpointForkPublication,
        *,
        recovery_failures: list[dict[str, str]],
    ) -> dict[str, Any]:
        secured = self._secure_checkpoint_fork_subtree(
            remapped=publication.remapped,
            failures=recovery_failures,
        )
        receipt = self._fork_recovery_required_receipt(
            checkpoint=publication.checkpoint,
            root_pid=publication.root_pid,
            remapped=publication.remapped,
            post_commit_failures=recovery_failures,
            subtree_quarantined=(secured == "terminalized"),
        )
        if secured is None:
            self._record_and_fence_checkpoint_fork_recovery(
                actor=publication.actor,
                receipt=receipt,
            )
        else:
            self._record_checkpoint_fork_recovery(
                actor=publication.actor,
                receipt=receipt,
            )
        return receipt

    def _emit_checkpoint_fork_event(
        self,
        publication: _CheckpointForkPublication,
    ) -> None:
        try:
            self.events.emit(
                EventType.PROCESS_FORKED,
                source=publication.actor,
                target=publication.root_pid,
                payload={
                    "checkpoint_id": publication.checkpoint.checkpoint_id,
                    "source_pid": publication.checkpoint.pid,
                    "fork_root_pid": publication.root_pid,
                },
            )
        except Exception as exc:
            failure = self._fork_failure(phase="fork_event_emission", exc=exc)
            try:
                failure = self._fork_post_commit_failure(
                    actor=publication.actor,
                    checkpoint=publication.checkpoint,
                    fork_root_pid=publication.root_pid,
                    phase="fork_event_emission",
                    exc=exc,
                )
            except BaseException as record_exc:
                failure["failure_record_error_type"] = type(record_exc).__name__
                failure["failure_record_error"] = str(record_exc)
                publication.post_commit_failures.append(failure)
                self._attach_fork_receipt(
                    record_exc,
                    self._fork_publication_committed_receipt(publication),
                )
                raise
            publication.post_commit_failures.append(failure)
        except BaseException as exc:
            publication.post_commit_failures.append(
                self._fork_failure(phase="fork_event_emission", exc=exc)
            )
            self._attach_fork_receipt(
                exc,
                self._fork_publication_committed_receipt(publication),
            )
            raise

    def _record_checkpoint_fork_audit(
        self,
        publication: _CheckpointForkPublication,
    ) -> None:
        try:
            self.audit.record(
                actor=publication.actor,
                action="checkpoint.fork",
                target=self.checkpoint_resource(
                    publication.checkpoint.checkpoint_id
                ),
                decision={
                    "source_pid": publication.checkpoint.pid,
                    "fork_root_pid": publication.root_pid,
                    "pid_map": publication.remapped["pid_map"],
                    "main_state_committed": True,
                    "status": (
                        "forked_with_warnings"
                        if publication.post_commit_failures
                        else "forked"
                    ),
                    "post_commit_failures": list(
                        publication.post_commit_failures
                    ),
                },
            )
        except Exception as exc:
            publication.post_commit_failures.append(
                self._fork_post_commit_failure(
                    actor=publication.actor,
                    checkpoint=publication.checkpoint,
                    fork_root_pid=publication.root_pid,
                    phase="fork_audit_recording",
                    exc=exc,
                    record_failure=False,
                )
            )
        except BaseException as exc:
            publication.post_commit_failures.append(
                self._fork_failure(phase="fork_audit_recording", exc=exc)
            )
            self._attach_fork_receipt(
                exc,
                self._fork_publication_committed_receipt(publication),
            )
            raise

    def _fork_publication_committed_receipt(
        self,
        publication: _CheckpointForkPublication,
    ) -> dict[str, Any]:
        return self._fork_committed_receipt(
            checkpoint=publication.checkpoint,
            root_pid=publication.root_pid,
            remapped=publication.remapped,
            post_commit_failures=publication.post_commit_failures,
        )

    def _reject_nested_checkpoint_fork_transaction(self) -> None:
        if (
            self._unit_of_work.probe_runtime_assembly_readiness()
            is StoreAssemblyReadiness.ACTIVE_TRANSACTION
        ):
            raise ValidationError(
                "checkpoint fork cannot run inside an existing store transaction"
            )

    @staticmethod
    def _fork_failure(*, phase: str, exc: BaseException) -> dict[str, str]:
        return {
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    def _fork_committed_receipt(
        self,
        *,
        checkpoint: Checkpoint,
        root_pid: str,
        remapped: dict[str, Any],
        post_commit_failures: Iterable[Mapping[str, str]],
    ) -> dict[str, Any]:
        failures = [dict(item) for item in post_commit_failures]
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "source_pid": checkpoint.pid,
            "fork_root_pid": root_pid,
            "pid_map": dict(remapped["pid_map"]),
            "object_map": dict(remapped["object_map"]),
            "tool_map": dict(remapped["tool_map"]),
            "status": "forked_with_warnings" if failures else "forked",
            "main_state_committed": True,
            "reconciliation_pending": False,
            "post_commit_failures": failures,
        }

    def _fork_unknown_outcome_receipt(
        self,
        *,
        checkpoint: Checkpoint,
        root_pid: str,
        remapped: dict[str, Any],
        interruption: BaseException,
        diagnostic_error: BaseException,
    ) -> dict[str, Any]:
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "source_pid": checkpoint.pid,
            "fork_root_pid": root_pid,
            "pid_map": dict(remapped["pid_map"]),
            "object_map": dict(remapped["object_map"]),
            "tool_map": dict(remapped["tool_map"]),
            "status": "fork_outcome_unknown",
            "main_state_committed": None,
            "reconciliation_pending": True,
            "post_commit_failures": [],
            "outcome_diagnostic": {
                "phase": "fork_commit_confirmation",
                "interruption_error_type": type(interruption).__name__,
                "interruption": str(interruption),
                "diagnostic_error_type": type(diagnostic_error).__name__,
                "diagnostic_error": str(diagnostic_error),
                "prepared_runtime_assets_retained": True,
            },
        }

    def _fork_recovery_required_receipt(
        self,
        *,
        checkpoint: Checkpoint,
        root_pid: str,
        remapped: dict[str, Any],
        post_commit_failures: Iterable[Mapping[str, str]],
        subtree_quarantined: bool,
    ) -> dict[str, Any]:
        failures = [dict(item) for item in post_commit_failures]
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "source_pid": checkpoint.pid,
            "fork_root_pid": root_pid,
            "pid_map": dict(remapped["pid_map"]),
            "object_map": dict(remapped["object_map"]),
            "tool_map": dict(remapped["tool_map"]),
            "status": "fork_recovery_required",
            "main_state_committed": True,
            "reconciliation_pending": True,
            "post_commit_failures": failures,
            "outcome_diagnostic": {
                "phase": (
                    failures[-1]["phase"]
                    if failures
                    else "fork_recovery_required"
                ),
                "fork_subtree_quarantined": subtree_quarantined,
                "prepared_runtime_assets_retained": True,
            },
        }

    def _secure_checkpoint_fork_subtree(
        self,
        *,
        remapped: Mapping[str, Any],
        failures: list[dict[str, str]],
    ) -> str | None:
        """Confirm absence or durably terminalize a whole ambiguous subtree."""

        target_pids = sorted(
            {str(pid) for pid in remapped.get("pid_map", {}).values()}
        )
        try:
            with self._unit_of_work.transaction():
                processes = [
                    self._processes.get_process(pid) for pid in target_pids
                ]
                present = [process for process in processes if process is not None]
                if present and len(present) != len(target_pids):
                    raise ValidationError(
                        "checkpoint fork recovery found a partial durable subtree"
                    )
                for process in present:
                    if process.status in self.TERMINAL_STATUSES:
                        continue
                    self._process_transitions.transition(
                        process.pid,
                        ProcessStatus.FAILED,
                        expected_revision=process.revision,
                        expected_status=process.status,
                        expected_state_generation=process.state_generation,
                        wait_state=None,
                        outcome=FailedProcessOutcome(
                            code="checkpoint_fork_recovery_required"
                        ),
                        control=True,
                        allowed_statuses={process.status},
                        reason="checkpoint fork publication recovery required",
                    )
        except BaseException as exc:
            failures.append(
                self._fork_failure(
                    phase="fork_subtree_terminalization",
                    exc=exc,
                )
            )
        try:
            processes = [self._processes.get_process(pid) for pid in target_pids]
        except BaseException as exc:
            failures.append(
                self._fork_failure(
                    phase="fork_subtree_terminalization_confirmation",
                    exc=exc,
                )
            )
            return None
        if all(process is None for process in processes):
            return "absent"
        if all(
            process is not None and process.status in self.TERMINAL_STATUSES
            for process in processes
        ):
            return "terminalized"
        if processes:
            failures.append(
                {
                    "phase": "fork_subtree_terminalization_confirmation",
                    "error_type": "ValidationError",
                    "message": "checkpoint fork subtree remained schedulable",
                }
            )
        return None

    def _record_and_fence_checkpoint_fork_recovery(
        self,
        *,
        actor: str,
        receipt: MutableMapping[str, Any],
    ) -> None:
        """Persist a recovery signal, then close mutation admission."""

        self._record_checkpoint_fork_recovery(actor=actor, receipt=receipt)
        diagnostic = dict(receipt.get("outcome_diagnostic") or {})

        diagnostic["lifecycle_fence_requested"] = True
        receipt["outcome_diagnostic"] = diagnostic
        try:
            self._operations.finish(
                OperationOutcome.UNKNOWN,
                metadata={
                    "checkpoint_fork_receipt": deepcopy(dict(receipt)),
                    "checkpoint_fork_recovery_required": True,
                },
            )
            diagnostic["operation_recovery_signal_recorded"] = True
        except BaseException as exc:
            diagnostic["operation_recovery_signal_error_type"] = type(exc).__name__
            diagnostic["operation_recovery_signal_error"] = str(exc)
        callback = self._recovery_required_callback
        if callback is None:
            diagnostic["lifecycle_fence_error_type"] = "RuntimeError"
            diagnostic["lifecycle_fence_error"] = (
                "checkpoint fork recovery fence is unavailable"
            )
        else:
            try:
                callback(
                    publication_id=(
                        "checkpoint-fork:"
                        f"{receipt['checkpoint_id']}:{receipt['fork_root_pid']}"
                    )
                )
                diagnostic["lifecycle_fenced"] = True
            except BaseException as exc:
                diagnostic["lifecycle_fence_error_type"] = type(exc).__name__
                diagnostic["lifecycle_fence_error"] = str(exc)
        receipt["outcome_diagnostic"] = diagnostic

    def _record_checkpoint_fork_recovery(
        self,
        *,
        actor: str,
        receipt: MutableMapping[str, Any],
    ) -> None:
        """Record recovery only after the safety state commits independently."""

        diagnostic = dict(receipt.get("outcome_diagnostic") or {})
        try:
            record = self.audit.record(
                actor=actor,
                action="checkpoint.fork.recovery_required",
                target=self.checkpoint_resource(str(receipt["checkpoint_id"])),
                decision=deepcopy(dict(receipt)),
            )
            diagnostic["recovery_signal_record_id"] = record.record_id
        except BaseException as exc:
            diagnostic["recovery_signal_error_type"] = type(exc).__name__
            diagnostic["recovery_signal_error"] = str(exc)
        receipt["outcome_diagnostic"] = diagnostic

    def _attach_fork_receipt(
        self,
        exc: BaseException,
        receipt: Mapping[str, Any],
    ) -> None:
        stable_receipt = deepcopy(dict(receipt))
        setattr(exc, self.FORK_COMMITTED_RECEIPT_ATTRIBUTE, stable_receipt)
        if stable_receipt.get("reconciliation_pending") is True:
            diagnostic = dict(stable_receipt.get("outcome_diagnostic") or {})
            exc.add_note(
                "checkpoint fork requires recovery before any retry; inspect "
                f"{self.FORK_COMMITTED_RECEIPT_ATTRIBUTE} "
                f"(checkpoint_id={stable_receipt['checkpoint_id']}, "
                f"fork_root_pid={stable_receipt['fork_root_pid']}, "
                f"phase={diagnostic.get('phase')})"
            )
            return
        if stable_receipt.get("main_state_committed") is not True:
            diagnostic = dict(stable_receipt.get("outcome_diagnostic") or {})
            exc.add_note(
                "checkpoint fork durable outcome could not be confirmed; "
                "prepared runtime assets were retained; inspect "
                f"{self.FORK_COMMITTED_RECEIPT_ATTRIBUTE} before recovery "
                f"(checkpoint_id={stable_receipt['checkpoint_id']}, "
                f"fork_root_pid={stable_receipt['fork_root_pid']}, "
                f"diagnostic={diagnostic.get('diagnostic_error_type')})"
            )
            return
        exc.add_note(
            "checkpoint fork main state committed; inspect "
            f"{self.FORK_COMMITTED_RECEIPT_ATTRIBUTE} before any retry "
            f"(checkpoint_id={stable_receipt['checkpoint_id']}, "
            f"fork_root_pid={stable_receipt['fork_root_pid']})"
        )

    def replay_to_event(
        self,
        checkpoint_id: str,
        event_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability and actor is not None:
            with self._checkpoint_or_process_read_scope(actor, checkpoint):
                return self.replay_to_event(
                    checkpoint_id,
                    event_id,
                    actor=actor,
                    require_capability=False,
                )
        events = self._evidence.list_events()
        checkpoint_event_index = self._checkpoint_replay_start_index(
            events,
            checkpoint,
        )
        # Replay scope starts from the frozen subtree, then follows only
        # explicit parent/child creation events in the durable event stream.
        # It therefore remains stable if post-checkpoint descendants later
        # exit or are removed by restore, without treating detached checkpoint
        # forks merely initiated by the owner as descendants.
        scoped_pids = {str(pid) for pid in snapshot.get("subtree_pids", [])}
        selected = []
        reached = False
        for event in events[checkpoint_event_index:]:
            if event.source not in scoped_pids and event.target not in scoped_pids:
                continue
            selected.append(self._checkpoint_replay_event(event))
            self._extend_checkpoint_replay_scope(event, scoped_pids)
            if event.event_id == event_id:
                reached = True
                break
        if not reached:
            raise NotFound(f"event not found after checkpoint: {event_id}")
        self.audit.record(
            actor=actor or "runtime",
            action="checkpoint.replay_to_event",
            target=self.checkpoint_resource(checkpoint_id),
            decision={"event_id": event_id, "events": len(selected), "diagnostic_only": True},
        )
        return {
            "checkpoint_id": checkpoint_id,
            "event_id": event_id,
            "diagnostic_only": True,
            "events": selected,
        }

    @staticmethod
    def _checkpoint_replay_start_index(
        events: list[Any],
        checkpoint: Checkpoint,
    ) -> int:
        for index, event in enumerate(events):
            if (
                event.type == EventType.CHECKPOINT_CREATED
                and event.target == checkpoint.pid
                and event.payload.get("checkpoint_id") == checkpoint.checkpoint_id
            ):
                return index
        raise NotFound(
            "checkpoint creation event not found for replay: "
            f"{checkpoint.checkpoint_id}"
        )

    @staticmethod
    def _checkpoint_replay_event(event: Any) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "type": event.type.value,
            "source": event.source,
            "target": event.target,
            "created_at": event.created_at,
            "payload": event.payload,
        }

    @staticmethod
    def _extend_checkpoint_replay_scope(event: Any, scoped_pids: set[str]) -> None:
        if event.type not in {EventType.PROCESS_CREATED, EventType.PROCESS_FORKED}:
            return
        parent = event.payload.get("parent")
        child = event.payload.get("child")
        if (
            isinstance(parent, str)
            and isinstance(child, str)
            and parent in scoped_pids
            and event.source == parent
            and event.target == child
        ):
            scoped_pids.add(child)

    def _build_snapshot(
        self,
        *,
        checkpoint_id: str,
        pid: str,
        reason: str,
        created_at: str,
        created_by: str,
    ) -> dict[str, Any]:
        subtree_pids = self._subtree_pids(pid)
        if not subtree_pids:
            raise NotFound(f"process not found: {pid}")
        process_rows = [
            self._safe_point_process_row(row, checkpoint_id)
            for row in self._snapshot_rows.load_process_snapshot_rows(
                subtree_pids
            ).processes
        ]
        object_oids = self._scoped_object_oids(process_rows, subtree_pids)
        referenced_object_oids = self._referenced_object_oids(process_rows, object_oids)
        referenced_object_types = {
            oid: obj.type.value
            for oid in referenced_object_oids
            if (obj := self._objects.get_object(oid)) is not None
        }
        namespace_names = self._scoped_namespaces(object_oids, subtree_pids)
        captured_rows, _repository_payloads = self._snapshot_rows.capture_checkpoint_rows(
            process_rows,
            object_oids=object_oids,
            namespace_names=namespace_names,
        )
        # Keep the public checkpoint payload-limit/fault-injection seam while
        # the surrounding snapshot scope holds the shared repository lock.
        object_payloads = self._object_payload_snapshot(object_oids)
        # Checkpoint control authority governs the snapshot artifact itself. It
        # is neither reconstructable process authority nor safe to duplicate,
        # so exclude it from the artifact in addition to restore/fork filtering.
        capability_rows = [
            row
            for row in captured_rows.capabilities
            if not str(row.get("resource", "")).startswith(
                self.CHECKPOINT_RESOURCE_PREFIX
            )
        ]
        capability_ids_by_subject: dict[str, list[str]] = {selected_pid: [] for selected_pid in subtree_pids}
        for capability_row in capability_rows:
            capability_ids_by_subject.setdefault(str(capability_row["subject"]), []).append(
                str(capability_row["cap_id"])
            )
        # capabilities_json is a denormalized process index. Derive it from the
        # capability rows captured by this same transaction so a capability
        # insertion racing its process-index attachment cannot leave a torn
        # checkpoint.
        for process_row in process_rows:
            process_row["capabilities_json"] = dumps(
                sorted(capability_ids_by_subject.get(str(process_row["pid"]), []))
            )
        captured_rows = replace(
            captured_rows,
            processes=tuple(process_rows),
            capabilities=tuple(capability_rows),
        )
        snapshot = {
            "version": SnapshotCodec.schema_version,
            "checkpoint_id": checkpoint_id,
            "pid": pid,
            "reason": reason,
            "created_at": created_at,
            "created_by": created_by,
            "subtree_pids": subtree_pids,
            "object_oids": object_oids,
            "owned_object_oids": object_oids,
            "referenced_object_oids": referenced_object_oids,
            "referenced_object_types": referenced_object_types,
            "namespaces": namespace_names,
            "owned_namespaces": namespace_names,
            "rows": captured_rows.to_mapping(),
            "object_payloads": object_payloads,
            "images": self._image_snapshot(process_rows),
            "image_artifacts": self._image_artifact_snapshot(process_rows),
            "jit_sources": self._jit_source_snapshot(process_rows),
            "modules": self._module_snapshot(),
        }
        return self.snapshots.normalize(snapshot)

    def _module_snapshot(self) -> list[dict[str, Any]]:
        if self._modules is None:
            return []
        return self._modules.loaded_module_summaries()

    def _require_snapshot_modules(self, snapshot: dict[str, Any]) -> None:
        required_modules = ProcessSnapshot.decode_module_requirements(
            snapshot.get("modules")
        )
        if not required_modules:
            return
        if self._modules is None:
            raise ValidationError(
                "checkpoint requires startup modules but the runtime module registry is unavailable"
            )
        missing = []
        for module in required_modules:
            module_id = str(module["module_id"])
            source_sha256 = str(module["source_sha256"])
            if not self._modules.is_loaded(module_id, source_sha256):
                missing.append({"module_id": module_id, "source_sha256": source_sha256})
        if missing:
            raise ValidationError(f"checkpoint requires startup modules that are not loaded: {missing}")

    def require_snapshot_modules(self, snapshot: dict[str, Any]) -> None:
        """Validate the module contract carried by a snapshot artifact."""

        self._require_snapshot_modules(snapshot)

    def _runtime_quiescent_for_restore(self):
        quiescent_state = getattr(self._scheduler, "quiescent_state", None)
        if callable(quiescent_state):
            return quiescent_state(reason="checkpoint restore")
        return _NullContext()

    def _runtime_registry_lifecycle_quiescent(self):
        return (
            self._registry_lifecycle_lock
            if self._registry_lifecycle_lock is not None
            else _NullContext()
        )

    def _runtime_object_ownership_quiescent(self):
        ownership_locked = getattr(self._memory, "ownership_locked", None)
        if callable(ownership_locked):
            return ownership_locked()
        return _NullContext()

    def _reject_active_object_tasks_for_restore(self, snapshot: dict[str, Any], current_pids: list[str]) -> None:
        scoped_pids = set(current_pids) | {str(pid) for pid in snapshot.get("subtree_pids", [])}
        scoped_oids = set(self._current_scoped_object_oids(current_pids)) | self._snapshot_owned_object_oids(snapshot)
        if not scoped_pids and not scoped_oids:
            return
        blocked: list[str] = []
        for task in self._processes.list_object_tasks(include_terminal=False):
            if (
                str(task.creator_pid) in scoped_pids
                or (task.runner_pid is not None and str(task.runner_pid) in scoped_pids)
                or str(task.owner_oid) in scoped_oids
            ):
                blocked.append(str(task.task_id))
        if blocked:
            raise ValidationError(
                "checkpoint restore refused while scoped ObjectTasks are active: "
                + ", ".join(sorted(blocked))
            )

    def _restore_scoped_rows(
        self,
        snapshot: dict[str, Any],
        current_pids: list[str],
        checkpoint: Checkpoint,
    ) -> tuple[list[str], list[str], list[str], list[Any]]:
        typed_snapshot = SnapshotCodec.decode_mapping(snapshot)
        rows = typed_snapshot.rows
        snapshot_object_oids = self._snapshot_owned_object_oids(snapshot)
        current_object_oids = set(self._current_scoped_object_oids(current_pids))
        object_oids = snapshot_object_oids | current_object_oids
        snapshot_namespaces = self._snapshot_owned_namespaces(snapshot, snapshot_object_oids)
        namespace_names = snapshot_namespaces | set(self._current_scoped_namespaces(current_pids))
        release_finalizer_objects = self._object_release_finalizer_objects(current_object_oids - snapshot_object_oids)
        with self._unit_of_work.transaction(include_object_payloads=True):
            # Capability status/uses and current deny policy must be sampled in
            # the same locked transaction that inserts the restored rows. A
            # revoke that wins before this point is never overwritten; one that
            # waits for this transaction applies immediately after it.
            restored_capability_rows = self._filtered_restored_capability_rows(
                list(rows.capabilities)
            )
            restored_process_rows = self._prepare_restored_process_rows(
                None,
                list(rows.processes),
                restored_capability_rows,
            )
            # Pending human/message state belongs to the same reconstructable
            # restore boundary as process rows. If a later insert fails, these
            # status changes must roll back with the SQLite rows and in-memory
            # object payloads.
            cancelled_human_requests = self._cancel_pending_human_requests(
                None,
                current_pids,
                checkpoint,
            )
            superseded_messages = self._supersede_post_checkpoint_messages(
                None,
                current_pids,
                checkpoint,
            )
            superseded_object_tasks = self._supersede_post_checkpoint_object_tasks(
                None,
                current_pids,
                object_oids,
                checkpoint,
            )
            self._snapshot_rows.replace_checkpoint_scope(
                typed_snapshot,
                current_pids=current_pids,
                current_object_oids=current_object_oids,
                all_object_oids=object_oids,
                all_namespace_names=namespace_names,
                snapshot_object_oids=snapshot_object_oids,
                snapshot_namespaces=snapshot_namespaces,
                restored_process_rows=restored_process_rows,
                restored_capability_rows=restored_capability_rows,
                before_insert=self._insert_row,
            )
            self._reconcile_restored_wait_states(
                None,
                [str(row["pid"]) for row in restored_process_rows],
            )
            self._reconcile_restored_object_task_results(
                None,
                snapshot,
                checkpoint,
            )
        return cancelled_human_requests, superseded_messages, superseded_object_tasks, release_finalizer_objects

    def _prepare_restored_process_rows(
        self,
        cur: object | None,
        process_rows: list[dict[str, Any]],
        capability_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fence snapshot rows with monotonic process concurrency epochs."""
        del cur  # retained for private fault-injection compatibility
        rows = SnapshotRows(
            processes=tuple(dict(row) for row in process_rows),
            capabilities=tuple(dict(row) for row in capability_rows),
        )
        return list(
            self._snapshot_rows.prepare_checkpoint_restore_process_rows(
                rows,
                restored_capability_rows=capability_rows,
            )
        )

    def _validate_snapshot_flow_rows(self, snapshot: dict[str, Any]) -> None:
        rows = snapshot.get("rows", {})
        for pending in rows.get("llm_pending_actions", []):
            canonical = self._canonical_snapshot_data_flow_context(
                pending.get("data_flow_context_json")
            )
            if canonical is None:
                raise ValidationError(
                    "0.3 checkpoint pending action has no canonical data-flow context"
                )
        for message in rows.get("process_messages", []):
            canonical = self._canonical_snapshot_message_metadata(
                message.get("metadata_json")
            )
            if canonical is None:
                raise ValidationError(
                    "0.3 checkpoint process message has no canonical label metadata"
                )

    @staticmethod
    def _canonical_snapshot_data_flow_context(value: Any) -> dict[str, Any] | None:
        try:
            decoded = loads(value) if isinstance(value, str) else value
            if not isinstance(decoded, dict) or not decoded:
                return None
            labels = decoded.get("labels")
            if not isinstance(labels, dict) or set(labels) != {
                "sensitivity",
                "trust_level",
                "integrity",
                "origin",
                "tenant",
                "principal",
                "declassification_authority",
            }:
                return None
            return DataFlowContext.from_dict(decoded).to_dict()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _canonical_snapshot_message_metadata(value: Any) -> dict[str, Any] | None:
        try:
            decoded = loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError):
            return None
        if not isinstance(decoded, dict):
            return None
        labels = decoded.get("data_labels")
        if not isinstance(labels, dict) or set(labels) != {
            "sensitivity",
            "trust_level",
            "integrity",
            "origin",
            "tenant",
            "principal",
            "declassification_authority",
        }:
            return None
        if not isinstance(decoded.get("source_oids"), list):
            return None
        try:
            decoded["data_labels"] = DataLabels.from_dict(labels).to_dict()
        except (TypeError, ValueError):
            return None
        return decoded

    def _run_restore_registry_post_commit_phases(
        self,
        *,
        actor: str,
        checkpoint: Checkpoint,
        snapshot: dict[str, Any],
        stale_tool_ids: set[str],
        scoped_pids: set[str],
    ) -> list[dict[str, str]]:
        phases = [
            ("image_reconciliation", lambda: self._restore_images(snapshot)),
            ("jit_source_reconciliation", lambda: self._restore_jit_sources(snapshot)),
            (
                "jit_pruning",
                lambda: self._prune_stale_ephemeral_jit_tools(stale_tool_ids, scoped_pids=scoped_pids),
            ),
        ]
        failures: list[dict[str, str]] = []
        for phase, operation in phases:
            try:
                operation()
            except Exception as exc:
                failures.append(
                    self._restore_post_commit_failure(
                        actor=actor,
                        checkpoint=checkpoint,
                        phase=phase,
                        exc=exc,
                    )
                )
        return failures

    def _run_restore_object_release_finalizer_phase(
        self,
        *,
        actor: str,
        checkpoint: Checkpoint,
        release_finalizer_objects: list[Any],
    ) -> list[dict[str, str]]:
        try:
            self._run_object_release_finalizers_for_objects(
                release_finalizer_objects,
                actor="checkpoint.restore",
                reason="checkpoint_restore",
            )
        except Exception as exc:
            return [
                self._restore_post_commit_failure(
                    actor=actor,
                    checkpoint=checkpoint,
                    phase="object_release_finalizers",
                    exc=exc,
                )
            ]
        return []

    def _restore_status(self, failures: list[dict[str, str]]) -> str:
        return "restored_with_warnings" if failures else "restored"

    def _restore_post_commit_failure(
        self,
        *,
        actor: str,
        checkpoint: Checkpoint,
        phase: str,
        exc: Exception,
        record_failure: bool = True,
    ) -> dict[str, str]:
        failure = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if not record_failure:
            return failure
        try:
            self.audit.record(
                actor=actor,
                action="checkpoint.restore.post_commit_failure",
                target=self.checkpoint_resource(checkpoint.checkpoint_id),
                decision={
                    **failure,
                    "pid": checkpoint.pid,
                    "main_state_committed": True,
                },
            )
        except Exception as audit_exc:
            # The committed restore must remain observable to the caller even
            # when its append-only audit sink is unavailable. Preserve that
            # secondary failure in-band rather than hiding the committed state.
            failure["audit_error_type"] = type(audit_exc).__name__
            failure["audit_error"] = str(audit_exc)
        return failure

    def _fork_post_commit_failure(
        self,
        *,
        actor: str,
        checkpoint: Checkpoint,
        fork_root_pid: str,
        phase: str,
        exc: Exception,
        record_failure: bool = True,
    ) -> dict[str, str]:
        failure = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if not record_failure:
            return failure
        try:
            self.audit.record(
                actor=actor,
                action="checkpoint.fork.post_commit_failure",
                target=self.checkpoint_resource(checkpoint.checkpoint_id),
                decision={
                    **failure,
                    "source_pid": checkpoint.pid,
                    "fork_root_pid": fork_root_pid,
                    "main_state_committed": True,
                },
            )
        except Exception as audit_exc:
            failure["audit_error_type"] = type(audit_exc).__name__
            failure["audit_error"] = str(audit_exc)
        return failure

    def _remap_snapshot(self, snapshot: dict[str, Any], *, parent_pid: str | None, root_pid: str) -> dict[str, Any]:
        rows = deepcopy(snapshot["rows"])
        identities, non_clonable_object_oids, source_capability_rows = (
            self._prepare_fork_identities(snapshot, rows)
        )
        rows["processes"] = [
            self._remap_process_row(
                row,
                identities,
                parent_pid,
                root_pid,
                non_clonable_object_oids,
            )
            for row in rows.get("processes", [])
        ]
        rows["object_namespaces"] = [
            self._remap_namespace_row(row, identities)
            for row in rows.get("object_namespaces", [])
            if str(row.get("namespace")) in identities.namespaces
        ]
        rows["objects"] = [
            self._remap_object_row(row, identities)
            for row in rows.get("objects", [])
            if str(row.get("oid")) in identities.objects
        ]
        rows["object_links"] = [
            self._remap_link_row(row, identities)
            for row in rows.get("object_links", [])
            if row["src_oid"] in identities.objects
            and row["dst_oid"] in identities.objects
        ]
        rows["capabilities"] = [
            self._remap_capability_row(row, identities)
            for row in rows.get("capabilities", [])
            if row["subject"] in identities.pids
            and not str(row["resource"]).startswith(
                self.CHECKPOINT_RESOURCE_PREFIX
            )
        ]
        rows["process_resource_reservations"] = [
            self._remap_resource_reservation_row(row, identities)
            for row in rows.get("process_resource_reservations", [])
            if row["parent_pid"] in identities.pids
            and row["child_pid"] in identities.pids
        ]
        rows["process_messages"] = [
            self._remap_message_row(row, identities)
            for row in rows.get("process_messages", [])
            if row["recipient_pid"] in identities.pids
        ]
        rows["llm_pending_actions"] = []
        rows["tool_candidates"] = [
            self._remap_tool_candidate_row(row, identities)
            for row in rows.get("tool_candidates", [])
            if row["pid"] in identities.pids
        ]
        rows["tools"] = [self._remap_tool_row(row, identities) for row in rows.get("tools", [])]
        object_types = {
            str(row.get("oid")): str(row.get("type"))
            for row in snapshot.get("rows", {}).get("objects", [])
        }
        payloads = {
            identities.objects[oid]: self._remap_object_payload(
                payload,
                object_type=object_types.get(str(oid)),
                candidate_map=dict(identities.candidates),
            )
            for oid, payload in snapshot.get("object_payloads", {}).items()
            if oid in identities.objects
        }
        return {
            "rows": rows,
            "object_payloads": payloads,
            "pid_map": dict(identities.pids),
            "object_map": dict(identities.objects),
            "namespace_map": dict(identities.namespaces),
            "capability_map": dict(identities.capabilities),
            "tool_map": dict(identities.tools),
            "candidate_map": dict(identities.candidates),
            "jit_sources": {
                identities.tools.get(str(tool_id), str(tool_id)): source
                for tool_id, source in snapshot.get("jit_sources", {}).items()
            },
            "source_capability_rows": source_capability_rows,
            "source_capability_ids": {
                target_cap_id: source_cap_id
                for source_cap_id, target_cap_id in identities.capabilities.items()
            },
            "non_clonable_object_oids": non_clonable_object_oids,
        }

    def _prepare_fork_identities(
        self,
        snapshot: dict[str, Any],
        rows: dict[str, list[dict[str, Any]]],
    ) -> tuple[SnapshotIdentityMap, set[str], dict[str, dict[str, Any]]]:
        pid_map = {pid: new_id("pid") for pid in snapshot["subtree_pids"]}
        owned_object_oids = self._snapshot_owned_object_oids(snapshot)
        non_clonable_object_oids = self._non_clonable_object_oids(snapshot)
        object_map = {
            oid: new_id("obj")
            for oid in sorted(owned_object_oids - non_clonable_object_oids)
        }
        namespace_map = {
            namespace: self._remap_namespace(namespace, pid_map)
            for namespace in self._snapshot_owned_namespaces(
                snapshot,
                owned_object_oids,
            )
        }
        tool_map = {
            str(row["tool_id"]): new_id("tool")
            for row in rows.get("tools", [])
            if bool(row.get("ephemeral"))
        }
        candidate_map = {
            str(row["candidate_id"]): new_id("tcand")
            for row in rows.get("tool_candidates", [])
        }
        source_rows = [
            row
            for row in rows.get("capabilities", [])
            if not self._capability_references_any_object(
                row,
                non_clonable_object_oids,
            )
        ]
        rows["capabilities"] = self._fork_capability_rows(source_rows)
        source_capability_rows = {
            str(row["cap_id"]): dict(row)
            for row in rows.get("capabilities", [])
        }
        capability_map = {
            row["cap_id"]: new_id("cap")
            for row in rows.get("capabilities", [])
        }
        identities = SnapshotIdentityMap(
            pids=pid_map,
            objects=object_map,
            namespaces=namespace_map,
            capabilities=capability_map,
            tools=tool_map,
            candidates=candidate_map,
        )
        return identities, non_clonable_object_oids, source_capability_rows

    def _non_clonable_object_oids(self, snapshot: dict[str, Any]) -> set[str]:
        candidate_oids = self._snapshot_owned_object_oids(snapshot) | {
            str(oid) for oid in snapshot.get("referenced_object_oids", [])
        } | self._process_row_referenced_oids(snapshot.get("rows", {}).get("processes", []))
        snapshot_types = {
            str(row.get("oid")): str(row.get("type"))
            for row in snapshot.get("rows", {}).get("objects", [])
        }
        snapshot_types.update(
            {
                str(oid): str(object_type)
                for oid, object_type in snapshot.get("referenced_object_types", {}).items()
            }
        )
        non_clonable: set[str] = set()
        for oid in candidate_oids:
            object_type = snapshot_types.get(oid)
            if object_type is None:
                current = self._objects.get_object(oid)
                object_type = current.type.value if current is not None else None
            if object_type == ObjectType.EXTERNAL_REF.value:
                non_clonable.add(oid)
        return non_clonable

    def _capability_references_any_object(self, row: dict[str, Any], object_oids: set[str]) -> bool:
        resource = str(row.get("resource") or "")
        return resource.startswith("object:") and resource.split(":", 1)[1] in object_oids

    def _fork_capability_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("resource", "")).startswith(self.CHECKPOINT_RESOURCE_PREFIX):
                continue
            current = self._authority.get_capability(str(row.get("cap_id")))
            if current is None or not current.active:
                continue
            if self.capabilities.is_expired(current):
                continue
            # Forking is authority duplication, so a remaining-use counter
            # cannot safely be copied. Capability delegation/grant already
            # rejects finite parents for the same reason. Conservatively omit
            # every finite record rather than turning one remaining use into
            # one use in both the source and fork.
            item = dict(row)
            item["uses_remaining"] = current.uses_remaining
            item["status"] = current.status.value
            item["rights_json"] = dumps(sorted(current.rights))
            item["effect"] = current.effect.value
            allowed_rights = self.capabilities.transition_allowed_rights(
                current,
                transition_kind="checkpoint.fork",
                duplicates_authority=True,
            )
            if not allowed_rights:
                continue
            item["rights_json"] = dumps(allowed_rights)
            kept.append(item)
        return kept

    def _capability_is_expired(self, capability: Any) -> bool:
        if self.capabilities is None:
            return False
        return bool(self.capabilities.is_expired(capability))

    def _currently_allowed_fork_rights(self, capability: Any) -> list[str]:
        """Keep only rights that still survive current policy before checkpoint fork.

        A checkpoint may contain a broad allow capability. If a narrower deny or
        ask policy was added after the checkpoint, copying the broad allow into a
        new subject would bypass the current restriction because that new subject
        does not hold the post-checkpoint restrictive record. Capabilities have
        no exception syntax, so the conservative fork rule is to drop any right
        whose resource may overlap a currently active restrictive policy.
        """

        if self.capabilities is None:
            return sorted(capability.rights)
        return self.capabilities.transition_allowed_rights(
            capability,
            transition_kind="checkpoint.restore_or_fork",
            duplicates_authority=False,
        )

    def _filter_restored_capability_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if str(row.get("resource", "")).startswith(self.CHECKPOINT_RESOURCE_PREFIX):
            return None
        current = self._authority.get_capability(str(row.get("cap_id")))
        if current is not None:
            if not current.active or self._capability_is_expired(current):
                return None
            capability = current
        else:
            capability = self._capability_from_row(row)
            if self._capability_is_expired(capability):
                return None
        item = dict(row)
        item["uses_remaining"] = capability.uses_remaining
        item["status"] = CapabilityStatus.ACTIVE.value
        item["effect"] = capability.effect.value
        item["rights_json"] = dumps(sorted(capability.rights))
        if capability.effect == CapabilityEffect.ALLOW:
            allowed_rights = self._currently_allowed_fork_rights(capability)
            if not allowed_rights:
                return None
            item["rights_json"] = dumps(allowed_rights)
        return item

    def _filtered_restored_capability_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in rows:
            filtered = self._filter_restored_capability_row(row)
            if filtered is not None:
                kept.append(filtered)
        return kept

    def _capability_from_row(self, row: dict[str, Any]) -> Capability:
        return Capability(
            cap_id=str(row["cap_id"]),
            subject=str(row["subject"]),
            resource=str(row["resource"]),
            rights=set(loads(row.get("rights_json"), [])),
            constraints=loads(row.get("constraints_json"), {}),
            issued_by=str(row.get("issued_by") or "checkpoint.restore"),
            issued_at=str(row.get("issued_at") or utc_now()),
            expires_at=row.get("expires_at"),
            delegable=bool(row.get("delegable")),
            revocable=bool(row.get("revocable", True)),
            effect=CapabilityEffect(str(row.get("effect") or CapabilityEffect.ALLOW.value)),
            issuer_cap_id=row.get("issuer_cap_id"),
            parent_cap_id=row.get("parent_cap_id"),
            delegation_depth=int(row.get("delegation_depth") or 0),
            max_delegation_depth=(
                int(row["max_delegation_depth"])
                if row.get("max_delegation_depth") is not None
                else None
            ),
            uses_remaining=row.get("uses_remaining"),
            status=CapabilityStatus(str(row.get("status") or CapabilityStatus.ACTIVE.value)),
            metadata=loads(row.get("metadata_json"), {}),
        )

    def _insert_fork_rows(
        self,
        remapped: dict[str, Any],
        *,
        fork_rows: SnapshotRows,
        object_payloads: Mapping[str, Any],
        actor: str,
        checkpoint_id: str,
        require_capability: bool,
        fork_parent_pid: str | None = None,
        fork_root_pid: str | None = None,
        image_snapshot: dict[str, Any] | None = None,
        restored_image_ids: list[str] | None = None,
        publication_state: MutableMapping[str, Any] | None = None,
    ) -> None:
        with self._unit_of_work.transaction(include_object_payloads=True):
            # Start from the pre-frozen artifact so callbacks cannot mutate the
            # publication inputs. Authority revalidation may only remove or
            # narrow capabilities before the final canonical artifact is set.
            remapped["rows"] = fork_rows.to_mapping()
            # These checks must share the publication transaction. A revoke or
            # parent exit committed after preflight therefore wins before any
            # fork process/object/image row becomes visible.
            self._validate_fork_parent(
                actor,
                fork_parent_pid,
                require_capability=require_capability,
            )
            if require_capability:
                self._require_checkpoint_right(actor, checkpoint_id, CapabilityRight.EXECUTE)
                if image_snapshot is not None:
                    self._require_snapshot_image_restore_rights(
                        actor,
                        image_snapshot,
                        overwrite_existing=False,
                    )
            self._revalidate_remapped_fork_capabilities(remapped)
            published_rows = SnapshotRows.from_mapping(
                deepcopy(remapped["rows"])
            )
            published_payloads = deepcopy(dict(object_payloads))
            if publication_state is not None:
                publication_state["fork_rows"] = published_rows
                publication_state["object_payloads"] = published_payloads
            if image_snapshot is not None:
                # Image/artifact rows and process rows commit together. The
                # Runtime's in-memory image entries are explicitly discarded
                # by the caller if this transaction fails.
                existing_image_ids = set(self._images)
                try:
                    self._restore_images(image_snapshot, overwrite_existing=False)
                finally:
                    if restored_image_ids is not None:
                        restored_image_ids.extend(
                            image_id
                            for image_id in self._images
                            if image_id not in existing_image_ids and image_id not in restored_image_ids
                        )
            if fork_parent_pid is not None and fork_root_pid is not None:
                self._reserve_fork_parent_child_budget(fork_parent_pid, fork_root_pid, remapped)
            self._snapshot_rows.insert_checkpoint_fork_rows(
                published_rows,
                object_payloads=published_payloads,
                before_insert=self._insert_row,
            )
            self._bind_fork_authority_manifests(remapped, actor=actor)
            if fork_parent_pid is not None:
                # Storage helpers now honor the outer transaction, so the
                # parent charge, reservation, and fork rows become visible as
                # one unit or roll back as one unit.
                self._charge_fork_parent_child_create(fork_parent_pid)
        if publication_state is not None:
            # Fast-path acknowledgement for post-commit sinks/wrappers. If an
            # interruption lands before this assignment, the caller confirms
            # the unique root from durable storage before deciding cleanup.
            self._acknowledge_fork_main_state_commit(publication_state)

    @staticmethod
    def _acknowledge_fork_main_state_commit(
        publication_state: MutableMapping[str, Any],
    ) -> None:
        publication_state["main_state_committed"] = True

    def _fork_root_is_persisted(self, root_pid: str) -> bool:
        """Confirm the unique fork root after a lost in-memory commit ack."""

        return self._processes.get_process(root_pid) is not None

    def _bind_fork_authority_manifests(self, remapped: dict[str, Any], *, actor: str) -> None:
        manifests = self._authority_manifests
        if manifests is None:
            return
        rows = [dict(row) for row in remapped["rows"].get("processes", [])]
        rows_by_pid = {str(row["pid"]): row for row in rows}
        source_by_target = {
            str(target_pid): str(source_pid)
            for source_pid, target_pid in remapped.get("pid_map", {}).items()
        }
        capability_rows = [dict(row) for row in remapped["rows"].get("capabilities", [])]
        pending = dict(rows_by_pid)
        bound: dict[str, str] = {}
        while pending:
            progressed = False
            for target_pid, row in list(pending.items()):
                parent_pid = str(row.get("parent_pid") or "") or None
                if parent_pid in pending:
                    continue
                manifest = manifests.bind_checkpoint_fork(
                    source_pid=source_by_target[target_pid],
                    target_pid=target_pid,
                    image_id=str(row["image_id"]),
                    goal_ref=str(row["goal_oid"]) if row.get("goal_oid") is not None else None,
                    authorized_capabilities=self._fork_manifest_capability_specs(
                        capability_rows,
                        target_pid=target_pid,
                    ),
                    resource_budget=loads(row.get("resource_budget_json"), {}),
                    parent_manifest_id=bound.get(parent_pid) if parent_pid is not None else None,
                    issued_by=f"checkpoint.fork:{actor}",
                )
                bound[target_pid] = manifest.manifest_id
                pending.pop(target_pid)
                progressed = True
            if not progressed:
                raise ValidationError("checkpoint fork process hierarchy contains a cycle")

    @staticmethod
    def _fork_manifest_capability_specs(
        rows: list[dict[str, Any]],
        *,
        target_pid: str,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for row in rows:
            resource = str(row.get("resource") or "")
            if (
                str(row.get("subject")) != target_pid
                or str(row.get("status")) != CapabilityStatus.ACTIVE.value
                or str(row.get("effect")) != CapabilityEffect.ALLOW.value
                or resource.startswith(("object:", "object_namespace:", "checkpoint:"))
            ):
                continue
            spec: dict[str, Any] = {
                "resource": resource,
                "rights": loads(row.get("rights_json"), []),
                "constraints": loads(row.get("constraints_json"), {}),
                "delegable": bool(row.get("delegable")),
                "revocable": bool(row.get("revocable", True)),
            }
            for key in ("expires_at", "uses_remaining", "max_delegation_depth"):
                if row.get(key) is not None:
                    spec[key] = row[key]
            specs.append(spec)
        return specs

    def _discard_remapped_jit_sources(self, remapped: dict[str, Any]) -> None:
        tools = self._tools
        if tools is None:
            return
        for tool_id in remapped.get("tool_map", {}).values():
            tools.forget_loaded_jit(tool_id)

    def _discard_uncommitted_fork_images(self, image_ids: Iterable[str]) -> None:
        for image_id in image_ids:
            self._images.pop(str(image_id), None)

    def _cleanup_uncommitted_fork(
        self,
        remapped: dict[str, Any],
        image_ids: Iterable[str],
        *,
        primary: BaseException,
    ) -> None:
        cleanup_failures: list[BaseException] = []
        for cleanup in (
            lambda: self._discard_remapped_jit_sources(remapped),
            lambda: self._discard_uncommitted_fork_images(image_ids),
        ):
            try:
                cleanup()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures:
            raise BaseExceptionGroup(
                "checkpoint fork failed and uncommitted cleanup was interrupted",
                [primary, *cleanup_failures],
            ) from primary

    def _revalidate_remapped_fork_capabilities(self, remapped: dict[str, Any]) -> None:
        """Refresh source authority immediately before fork-row insertion.

        Snapshot remapping allocates new ids outside the transaction. Authority
        remains provisional until this transaction-local check, which prevents
        a revoke committed after preflight from being copied into the fork.
        """

        source_rows = list(remapped.get("source_capability_rows", {}).values())
        current_source_rows = {
            str(row["cap_id"]): row
            for row in self._fork_capability_rows(source_rows)
        }
        source_ids = remapped.get("source_capability_ids", {})
        kept_rows: list[dict[str, Any]] = []
        for remapped_row in remapped["rows"].get("capabilities", []):
            source_id = source_ids.get(str(remapped_row["cap_id"]))
            current = current_source_rows.get(str(source_id)) if source_id is not None else None
            if current is None:
                continue
            item = dict(remapped_row)
            for key in ("rights_json", "uses_remaining", "status", "effect"):
                item[key] = current[key]
            kept_rows.append(item)
        remapped["rows"]["capabilities"] = kept_rows
        kept_ids = {str(row["cap_id"]) for row in kept_rows}
        for process_row in remapped["rows"].get("processes", []):
            process_row["capabilities_json"] = dumps(
                [
                    cap_id
                    for cap_id in loads(process_row.get("capabilities_json"), [])
                    if str(cap_id) in kept_ids
                ]
            )
            view = loads(process_row.get("memory_view_json"), {}) if process_row.get("memory_view_json") else None
            if not view:
                continue
            view["roots"] = [
                root
                for root in view.get("roots", [])
                if root.get("capability_id") is None or str(root.get("capability_id")) in kept_ids
            ]
            process_row["memory_view_json"] = dumps(view)

    def _reserve_fork_parent_child_budget(self, parent_pid: str, fork_root_pid: str, remapped: dict[str, Any]) -> None:
        resources = self._resources
        if resources is None:
            return
        resources.reserve_child_budget(parent_pid, fork_root_pid, self._fork_root_resource_budget(fork_root_pid, remapped))

    def _charge_fork_parent_child_create(self, parent_pid: str | None) -> None:
        if parent_pid is None:
            return
        resources = self._resources
        if resources is None:
            return
        resources.charge(
            parent_pid,
            ResourceUsage(child_processes=1),
            source="process.child_create",
            context={"parent_pid": parent_pid},
            allow_overage=False,
            kill_on_exceed=False,
        )

    def _fork_root_resource_budget(self, fork_root_pid: str, remapped: dict[str, Any]) -> ResourceBudget:
        for row in remapped["rows"].get("processes", []):
            if row["pid"] == fork_root_pid:
                return ResourceBudget(**loads(row.get("resource_budget_json"), {}))
        raise NotFound(f"fork root process not found in remapped snapshot: {fork_root_pid}")

    def _load_checkpoint(self, checkpoint_id: str) -> tuple[Checkpoint, dict[str, Any]]:
        checkpoint, snapshot, _typed = self._load_checkpoint_typed(checkpoint_id)
        return checkpoint, snapshot

    def _load_checkpoint_typed(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, dict[str, Any], ProcessSnapshot]:
        cached = self._preflight_artifact.get()
        self._preflight_artifact.set(None)
        if cached is not None and cached[0] == checkpoint_id:
            return cached[1], cached[2], cached[3]
        return self._read_checkpoint_artifact(checkpoint_id)

    def _read_checkpoint_artifact(
        self,
        checkpoint_id: str,
    ) -> tuple[Checkpoint, dict[str, Any], ProcessSnapshot]:
        found = self._snapshot_rows.get_checkpoint_snapshot(checkpoint_id)
        if found is None:
            raise NotFound(f"checkpoint not found: {checkpoint_id}")
        checkpoint, stored_snapshot = found
        typed, validated = self.snapshots.canonicalize(
            stored_snapshot.to_mapping()
        )
        return checkpoint, validated, typed

    def _checkpoint_summary(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "pid": checkpoint.pid,
            "reason": checkpoint.reason,
            "created_at": checkpoint.created_at,
            "created_by": checkpoint.created_by,
            "snapshot_version": checkpoint.snapshot_version,
            "effect_ledger_seq": checkpoint.effect_ledger_seq,
            "metadata": checkpoint.metadata or {},
        }

    def _require_process_right(self, actor: str, pid: str, right: CapabilityRight) -> None:
        if self.capabilities is None:
            return
        self.capabilities.require(actor, self.process_resource(pid), right)

    def _require_checkpoint_right(
        self,
        actor: str,
        checkpoint_id: str,
        right: CapabilityRight,
        *,
        consume: bool = True,
    ) -> CapabilityDecision | None:
        if self.capabilities is None:
            return None
        resource = "checkpoint:*" if checkpoint_id == "*" else self.checkpoint_resource(checkpoint_id)
        return self.capabilities.require(actor, resource, right, consume=consume)

    @contextmanager
    def _checkpoint_or_process_read_scope(
        self,
        actor: str,
        checkpoint: Checkpoint,
        *,
        purpose: str = "checkpoint diagnostic read",
    ):
        if self.capabilities is None:
            yield
            return
        decision = self.capabilities.authorize(
            actor,
            self.checkpoint_resource(checkpoint.checkpoint_id),
            CapabilityRight.READ,
        )
        if not decision.allowed:
            decision = self.capabilities.authorize(
                actor,
                self.process_resource(checkpoint.pid),
                CapabilityRight.READ,
            )
        if not decision.allowed:
            raise CapabilityDenied(f"{actor} lacks read on checkpoint {checkpoint.checkpoint_id}")
        with self.capabilities.authority_transaction(
            [decision],
            actor=actor,
            operation=purpose,
        ):
            yield

    def _require_checkpoint_or_process_read(self, actor: str, checkpoint: Checkpoint) -> None:
        """Consume checkpoint/process read for non-diagnostic commit workflows."""
        with self._checkpoint_or_process_read_scope(
            actor,
            checkpoint,
            purpose="checkpoint image commit read",
        ):
            return

    def checkpoint_or_process_read_scope(
        self,
        actor: str,
        checkpoint: Checkpoint,
        *,
        purpose: str = "checkpoint diagnostic read",
    ) -> Any:
        """Return the public authority scope used by image/checkpoint readers."""

        return self._checkpoint_or_process_read_scope(
            actor,
            checkpoint,
            purpose=purpose,
        )

    def _validate_fork_parent(
        self,
        actor: str,
        parent_pid: str | None,
        *,
        require_capability: bool,
        consume: bool = True,
    ) -> None:
        if parent_pid is None:
            return
        parent = self._processes.get_process(parent_pid)
        if parent is None:
            raise NotFound(f"process not found: {parent_pid}")
        if parent.status in self.TERMINAL_STATUSES:
            raise ProcessError(
                f"cannot attach checkpoint fork to terminal process: "
                f"{parent_pid} status={parent.status.value}"
            )
        if not require_capability or actor == parent_pid:
            return
        if self.capabilities is None:
            raise CapabilityDenied("checkpoint fork parent attachment requires a capability manager")
        self.capabilities.require(
            actor,
            self.process_resource(parent_pid),
            CapabilityRight.ADMIN,
            consume=consume,
        )

    def _subtree_pids(self, root_pid: str) -> list[str]:
        processes = {
            process.pid: process for process in self._processes.list_processes()
        }
        if root_pid not in processes:
            return []
        selected: list[str] = []
        queue = [root_pid]
        while queue:
            pid = queue.pop(0)
            if pid in selected:
                continue
            selected.append(pid)
            children = sorted(item.pid for item in processes.values() if item.parent_pid == pid)
            queue.extend(children)
        return selected

    def _safe_point_process_row(self, row: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
        item = dict(row)
        if item.get("status") == ProcessStatus.RUNNING.value:
            item["status"] = ProcessStatus.RUNNABLE.value
        item["checkpoint_head"] = checkpoint_id
        return item

    def _scoped_object_oids(self, process_rows: list[dict[str, Any]], pids: list[str]) -> list[str]:
        """Return objects whose lifecycle belongs to the process subtree.

        Memory-view roots are references, not ownership. Including every root in
        the destructive restore scope lets a borrower roll back another
        process's object and revoke unrelated subjects' handles.
        """

        oids: set[str] = set()
        for pid in pids:
            for owner_kind in (ObjectOwnerKind.PROCESS, ObjectOwnerKind.PROCESS_RESULT):
                for obj in self._objects.list_objects_owned_by(owner_kind, pid):
                    oids.add(obj.oid)
        # A completed ObjectTask can still own a result while the creator's
        # process view holds it. Such results are lifecycle-local to the
        # subtree, unlike an arbitrary borrowed root.
        pid_set = set(pids)
        for oid in self._process_row_referenced_oids(process_rows):
            obj = self._objects.get_object(oid)
            if (
                obj is not None
                and obj.owner_kind == ObjectOwnerKind.OBJECT_TASK
                and str(obj.created_by) in pid_set
            ):
                oids.add(oid)
        return sorted(oid for oid in oids if self._objects.has_object_payload(oid))

    def _referenced_object_oids(
        self,
        process_rows: list[dict[str, Any]],
        owned_object_oids: Iterable[str],
    ) -> list[str]:
        owned = {str(oid) for oid in owned_object_oids}
        return sorted(
            oid
            for oid in self._process_row_referenced_oids(process_rows)
            if oid not in owned and self._objects.has_object_payload(oid)
        )

    def _process_row_referenced_oids(self, process_rows: list[dict[str, Any]]) -> set[str]:
        oids: set[str] = set()
        for row in process_rows:
            if row.get("goal_oid"):
                oids.add(str(row["goal_oid"]))
            wait_state = process_wait_state_from_json(row.get("wait_state_json"))
            if isinstance(wait_state, (PausedProcessWait, HostResumeProcessWait)):
                if wait_state.reason_oid is not None:
                    oids.add(wait_state.reason_oid)
            outcome = process_outcome_from_json(row.get("outcome_json"))
            if isinstance(outcome, (ExitedProcessOutcome, FailedProcessOutcome)):
                if outcome.result_oid is not None:
                    oids.add(outcome.result_oid)
            elif isinstance(outcome, KilledProcessOutcome):
                if outcome.reason_oid is not None:
                    oids.add(outcome.reason_oid)
            view = loads(row.get("memory_view_json"), {}) if row.get("memory_view_json") else {}
            for root in view.get("roots", []):
                if isinstance(root, dict) and root.get("oid"):
                    oids.add(str(root["oid"]))
        return oids

    def _snapshot_owned_object_oids(self, snapshot: dict[str, Any]) -> set[str]:
        explicit = snapshot.get("owned_object_oids")
        if explicit is not None:
            return {str(oid) for oid in explicit}
        # Legacy snapshots mixed borrowed roots into object_oids. Infer
        # ownership from the captured rows so loading an old checkpoint cannot
        # regain the destructive behavior.
        pids = {str(pid) for pid in snapshot.get("subtree_pids", [])}
        owned: set[str] = set()
        for row in snapshot.get("rows", {}).get("objects", []):
            owner_kind = str(row.get("owner_kind") or ObjectOwnerKind.PROCESS.value)
            owner_id = str(row.get("owner_id") or row.get("created_by") or "")
            if owner_kind in {ObjectOwnerKind.PROCESS.value, ObjectOwnerKind.PROCESS_RESULT.value}:
                if owner_id in pids:
                    owned.add(str(row["oid"]))
            elif owner_kind == ObjectOwnerKind.OBJECT_TASK.value and str(row.get("created_by") or "") in pids:
                owned.add(str(row["oid"]))
        return owned

    def _snapshot_owned_namespaces(
        self,
        snapshot: dict[str, Any],
        owned_object_oids: set[str],
    ) -> set[str]:
        del owned_object_oids  # object references do not imply namespace ownership
        explicit = snapshot.get("owned_namespaces")
        if explicit is not None:
            return {str(namespace) for namespace in explicit}
        pids = {str(pid) for pid in snapshot.get("subtree_pids", [])}
        namespaces = {
            f"{self.config.memory.process_namespace_prefix}:{pid}"
            for pid in pids
        }
        for row in snapshot.get("rows", {}).get("object_namespaces", []):
            if str(row.get("created_by") or "") in pids:
                namespaces.add(str(row["namespace"]))
        return namespaces

    def _current_scoped_object_oids(self, pids: list[str]) -> list[str]:
        process_rows = list(
            self._snapshot_rows.load_process_snapshot_rows(pids).processes
        )
        return self._scoped_object_oids(process_rows, pids)

    def _scoped_namespaces(self, object_oids: list[str], pids: list[str]) -> list[str]:
        del object_oids  # object references do not imply namespace ownership
        namespaces = {f"{self.config.memory.process_namespace_prefix}:{pid}" for pid in pids}
        for pid in pids:
            for namespace in self._objects.list_namespaces_created_by(pid):
                namespaces.add(namespace.namespace)
        for namespace in list(namespaces):
            found = self._objects.get_namespace(namespace)
            if found is not None:
                namespaces.add(found.namespace)
        return sorted(namespaces)

    def _current_scoped_namespaces(self, pids: list[str]) -> list[str]:
        return self._scoped_namespaces(self._current_scoped_object_oids(pids), pids)

    def _image_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, Any]:
        image_ids = {row["image_id"] for row in process_rows}
        return {
            image_id: to_jsonable(self._images[image_id])
            for image_id in sorted(image_ids)
            if image_id in self._images
        }

    def _image_artifact_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        for row in process_rows:
            image = self._images.get(row["image_id"])
            if image is None or image.boot.get("kind") not in {"checkpoint_commit", "image_package"}:
                continue
            artifact_id = str(image.boot.get("artifact_id") or "")
            if not artifact_id:
                continue
            found = self._extensions.get_image_artifact(artifact_id)
            if found is None:
                continue
            artifact, metadata = found
            artifacts[artifact_id] = {"artifact": artifact, **metadata}
        return artifacts

    def _snapshot_image_ids_to_restore(self, snapshot: dict[str, Any], *, overwrite_existing: bool) -> list[str]:
        image_ids: list[str] = []
        for image_id, data in snapshot.get("images", {}).items():
            if image_id in self._images:
                if not overwrite_existing:
                    continue
                if to_jsonable(self._images[image_id]) == data:
                    continue
            image_ids.append(str(image_id))
        return image_ids

    def _validate_snapshot_restore_assets(self, snapshot: dict[str, Any]) -> None:
        registry = self._image_registry
        image_artifacts = snapshot.get("image_artifacts", {})
        tool_rows = {str(row.get("tool_id")) for row in snapshot.get("rows", {}).get("tools", [])}
        for image_id, data in snapshot.get("images", {}).items():
            image = AgentImage(**data)
            if registry is not None:
                registry.validate_image(image, validate_tools=False)
            boot_kind = str(image.boot.get("kind", "fresh"))
            artifact_id = str(image.boot.get("artifact_id") or "")
            if boot_kind in {"checkpoint_commit", "image_package"}:
                if not artifact_id:
                    raise ValidationError(f"checkpoint image {image_id} {boot_kind} boot is missing artifact_id")
                if (
                    artifact_id not in image_artifacts
                    and self._extensions.get_image_artifact(artifact_id) is None
                ):
                    raise ValidationError(f"checkpoint image {image_id} requires missing image artifact {artifact_id}")
                artifact_entry = image_artifacts.get(artifact_id)
                if artifact_entry is not None:
                    artifact_kind = str(artifact_entry.get("kind", boot_kind))
                    if artifact_kind != boot_kind:
                        raise ValidationError(
                            f"checkpoint image artifact {artifact_id} kind mismatch: expected {boot_kind}, found {artifact_kind}"
                        )
                    if not isinstance(artifact_entry.get("artifact", {}), dict):
                        raise ValidationError(f"checkpoint image artifact {artifact_id} payload must be an object")
        for tool_id, source in snapshot.get("jit_sources", {}).items():
            if str(tool_id) not in tool_rows:
                raise ValidationError(f"checkpoint JIT source references missing tool row: {tool_id}")
            if not isinstance(source, str):
                raise ValidationError(f"checkpoint JIT source must be text: {tool_id}")

    def _stale_ephemeral_tool_ids_for_restore(self, snapshot: dict[str, Any], current_pids: list[str]) -> set[str]:
        current_rows = list(
            self._snapshot_rows.load_process_snapshot_rows(
                current_pids
            ).processes
        )
        current_tool_ids = self._tool_ids_from_process_rows(current_rows)
        snapshot_tool_ids = self._tool_ids_from_process_rows(snapshot.get("rows", {}).get("processes", []))
        return current_tool_ids - snapshot_tool_ids

    def _validate_process_local_jit_restore_scope(
        self,
        tool_ids: set[str],
        *,
        scoped_pids: set[str],
        lookup_pid: str,
    ) -> None:
        """Reject a restore whose stale process-local JIT escaped its owner scope."""

        if not tool_ids:
            return
        selected: dict[str, JITRehydrationArtifact] = {}
        ordered_ids = sorted(tool_ids)
        hard_limit = self.config.runtime.jit_rehydration_page_hard_limit
        for offset in range(0, len(ordered_ids), hard_limit):
            requested_ids = frozenset(
                ordered_ids[offset : offset + hard_limit]
            )
            artifacts = self._snapshot_rows.get_jit_rehydration_artifacts(
                pid=lookup_pid,
                tool_ids=requested_ids,
            )
            for artifact in artifacts:
                if (
                    not isinstance(artifact, JITRehydrationArtifact)
                    or artifact.tool_id not in requested_ids
                    or artifact.tool_id in selected
                ):
                    raise ValidationError(
                        "checkpoint restore JIT ownership lookup returned invalid metadata"
                    )
                selected[artifact.tool_id] = artifact
        loaded_jit_ids = (
            set(self._tools.loaded_jit_tool_ids())
            if self._tools is not None
            else set()
        )
        rejected = [
            tool_id
            for tool_id in sorted((loaded_jit_ids & tool_ids) - set(selected))
            if (
                (handle := self._tools.loaded_tool_handle(tool_id)) is None
                or handle.scope == "ephemeral_process"
            )
        ]
        for tool_id, artifact in selected.items():
            is_process_local_jit = (
                artifact.candidate_match_count > 0
                or (
                    artifact.scope == "ephemeral_process"
                    and tool_id in loaded_jit_ids
                )
            )
            if not is_process_local_jit:
                continue
            has_exact_scoped_owner = (
                artifact.candidate_match_count == 1
                and artifact.candidate_pid in scoped_pids
                and bool(artifact.source_code)
            )
            escaped_scope = self._tool_id_used_outside_scope(
                tool_id,
                scoped_pids,
            )
            if not has_exact_scoped_owner or escaped_scope:
                rejected.append(tool_id)
        if rejected:
            raise ValidationError(
                "checkpoint restore refused process-local JIT bindings outside "
                f"the durable owner scope: {', '.join(sorted(rejected))}"
            )

    def _tool_ids_from_process_rows(self, process_rows: list[dict[str, Any]]) -> set[str]:
        tool_ids: set[str] = set()
        for row in process_rows:
            for tool_id in loads(row.get("tool_table_json"), {}).values():
                tool_ids.add(str(tool_id))
        return tool_ids

    @staticmethod
    def _registered_tool_ids_from_candidate_rows(
        candidate_rows: Iterable[dict[str, Any]],
    ) -> frozenset[str]:
        return frozenset(
            str(row["registered_tool_id"])
            for row in candidate_rows
            if row.get("registered_tool_id")
        )

    def _prune_stale_ephemeral_jit_tools(self, tool_ids: set[str], *, scoped_pids: set[str]) -> None:
        if not tool_ids:
            return
        tools = self._tools
        tool_rows = {
            str(row.get("tool_id")): row for row in self._extensions.list_tools()
        }
        selected: list[str] = []
        for tool_id in sorted(tool_ids):
            row = tool_rows.get(tool_id)
            if row is None or not bool(row.get("ephemeral")):
                continue
            if self._tool_id_used_outside_scope(tool_id, scoped_pids):
                continue
            selected.append(tool_id)
        if not selected:
            return
        before_handles: dict[str, ToolHandle] = {}
        before_sources: dict[str, str] = {}
        loaded_jit_ids: set[str] = set()
        if tools is not None:
            before_handles, before_sources = tools.snapshot_loaded_tool_state(selected)
            loaded_jit_ids = set(tools.loaded_jit_tool_ids())
        try:
            with self._unit_of_work.transaction():
                for tool_id in selected:
                    if tools is not None and tool_id in loaded_jit_ids:
                        tools.forget_loaded_jit(tool_id)
                    self._extensions.delete_tool(tool_id)
        except BaseException:
            if tools is not None:
                tools.restore_loaded_jit_state(before_handles, before_sources)
            raise

    def _tool_id_used_outside_scope(self, tool_id: str, scoped_pids: set[str]) -> bool:
        return self._snapshot_rows.tool_id_used_outside_scope(
            tool_id,
            scoped_pids=scoped_pids,
        )

    def _require_snapshot_image_restore_rights(
        self,
        actor: str,
        snapshot: dict[str, Any],
        *,
        overwrite_existing: bool,
        consume: bool = True,
    ) -> list[CapabilityDecision]:
        if self.capabilities is None:
            return []
        registry = self._image_registry
        decisions: list[CapabilityDecision] = []
        for image_id in self._snapshot_image_ids_to_restore(snapshot, overwrite_existing=overwrite_existing):
            resource = registry.resource_for(image_id) if registry is not None else f"image:{image_id}"
            required_right = (
                CapabilityRight.ADMIN
                if overwrite_existing and image_id in self._images
                else CapabilityRight.WRITE
            )
            decisions.append(
                self.capabilities.require(actor, resource, required_right, consume=consume)
            )
        return decisions

    def _restore_images(self, snapshot: dict[str, Any], *, overwrite_existing: bool = True) -> None:
        images = snapshot.get("images", {})
        registry = self._image_registry
        atomic_registrations = getattr(registry, "atomic_image_registrations", None)
        atomic_scope = (
            atomic_registrations(images)
            if callable(atomic_registrations)
            else self._unit_of_work.transaction()
        )
        # A restore may reconcile several images and their artifacts. Treat the
        # cache and durable rows as one batch so any failed write restores the
        # complete pre-reconciliation cache, not a partially updated prefix.
        with atomic_scope:
            restored_artifact_ids: set[str] | None = set() if not overwrite_existing else None
            for image_id, data in images.items():
                if not overwrite_existing and image_id in self._images:
                    continue
                image = AgentImage(**data)
                self._images[image_id] = image
                self._extensions.upsert_image(
                    image,
                    registered_by="checkpoint.restore",
                    source=f"checkpoint:{snapshot.get('checkpoint_id')}",
                    created_at=utc_now(),
                )
                if restored_artifact_ids is not None:
                    artifact_id = str(image.boot.get("artifact_id") or "")
                    if artifact_id:
                        restored_artifact_ids.add(artifact_id)
            for artifact_id, data in snapshot.get("image_artifacts", {}).items():
                if restored_artifact_ids is not None and artifact_id not in restored_artifact_ids:
                    continue
                existing = self._extensions.get_image_artifact(artifact_id)
                if existing is not None:
                    artifact, metadata = existing
                    expected_artifact = data.get("artifact", {})
                    expected_metadata = data.get("metadata", {})
                    if (
                        artifact != expected_artifact
                        or str(metadata.get("kind") or "")
                        != str(data.get("kind", "checkpoint_commit"))
                        or str(metadata.get("sha256") or "")
                        != str(data.get("sha256", ""))
                        or metadata.get("metadata", {}) != expected_metadata
                    ):
                        raise ValidationError(
                            "checkpoint image artifact conflicts with durable row: "
                            f"{artifact_id}"
                        )
                    continue
                self._extensions.insert_image_artifact(
                    artifact_id=artifact_id,
                    kind=str(data.get("kind", "checkpoint_commit")),
                    artifact=data.get("artifact", {}),
                    sha256=str(data.get("sha256", "")),
                    created_by="checkpoint.restore",
                    created_at=utc_now(),
                    metadata=data.get("metadata", {}),
                )

    def _object_payload_snapshot(self, object_oids: list[str]) -> dict[str, Any]:
        payloads: dict[str, Any] = {}
        limit = self.config.checkpoint.payload_capture_limit_bytes
        for oid in object_oids:
            if not self._objects.has_object_payload(oid):
                continue
            payload = deepcopy(self._objects.object_payload(oid))
            payload_bytes = len(dumps(payload).encode("utf-8"))
            if payload_bytes > limit:
                raise ValidationError(
                    f"object payload {oid} exceeds checkpoint payload_capture_limit_bytes={limit}"
                )
            payloads[oid] = payload
        return payloads

    def _jit_source_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, str]:
        tools = self._tools
        if tools is None:
            return {}
        tool_ids: set[str] = set()
        for row in process_rows:
            for tool_id in loads(row.get("tool_table_json"), {}).values():
                tool_ids.add(str(tool_id))
        _, sources = tools.snapshot_loaded_tool_state(tool_ids)
        return sources

    def _restore_jit_sources(self, snapshot: dict[str, Any]) -> None:
        tools = self._tools
        if tools is None:
            return
        tool_rows = {row["tool_id"]: row for row in snapshot.get("rows", {}).get("tools", [])}
        handles: dict[str, ToolHandle] = {}
        sources: dict[str, str] = {}
        for tool_id, source in snapshot.get("jit_sources", {}).items():
            row = tool_rows.get(tool_id)
            if row is None:
                continue
            handles[tool_id] = ToolHandle(
                tool_id=tool_id,
                name=row["name"],
                capability_id=None,
                scope=row["scope"],
            )
            sources[tool_id] = source
        selected_ids = set(handles)
        before_handles, before_sources = tools.snapshot_loaded_tool_state(
            selected_ids
        )
        try:
            tools.restore_loaded_jit_state(handles, sources)
        except BaseException:
            # Registry restoration is additive, so remove only implementations
            # introduced by this attempt before reinstalling the exact prior
            # set. A retry after a phase-receipt crash is therefore harmless.
            for tool_id in sorted(selected_ids - set(before_handles)):
                tools.forget_loaded_jit(tool_id)
            tools.restore_loaded_jit_state(before_handles, before_sources)
            raise

    def _external_effect_records_since(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        pids: Iterable[str] | None = None,
    ) -> list[Any]:
        selected_pids = self._external_effect_pids(checkpoint, snapshot=snapshot, current_pids=pids)
        return self._evidence.list_external_effects_changed_after(
            checkpoint.effect_ledger_seq,
            pids=selected_pids,
        )

    def _external_effects_since(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        pids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            external_effect_to_json(record)
            for record in self._external_effect_records_since(
                checkpoint,
                snapshot=snapshot,
                pids=pids,
            )
        ]

    def _external_effect_summary_since(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        pids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return external_effect_summary(
            self._external_effect_records_since(
                checkpoint,
                snapshot=snapshot,
                pids=pids,
            )
        )

    def _external_effect_pids(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        current_pids: Iterable[str] | None = None,
    ) -> list[str]:
        selected: set[str] = set()
        if snapshot is not None:
            selected.update(str(pid) for pid in snapshot.get("subtree_pids", []))
        if current_pids is None:
            selected.update(self._subtree_pids(checkpoint.pid))
        else:
            selected.update(str(pid) for pid in current_pids)
        return sorted(selected)

    def _cancel_pending_human_requests(
        self,
        cur: object | None,
        pids: list[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        del cur
        return self._snapshot_rows.cancel_pending_human_requests_after_checkpoint(
            pids,
            checkpoint,
        )

    def _supersede_post_checkpoint_messages(
        self,
        cur: object | None,
        pids: list[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        del cur
        return self._snapshot_rows.supersede_messages_after_checkpoint(
            pids,
            checkpoint,
        )

    def _supersede_post_checkpoint_object_tasks(
        self,
        cur: object | None,
        pids: list[str],
        object_oids: set[str],
        checkpoint: Checkpoint,
    ) -> list[str]:
        del cur
        return self._snapshot_rows.supersede_object_tasks_after_checkpoint(
            pids,
            object_oids,
            checkpoint,
        )

    def _reconcile_restored_object_task_results(
        self,
        cur: object | None,
        snapshot: dict[str, Any],
        checkpoint: Checkpoint,
    ) -> list[str]:
        del cur
        return self._snapshot_rows.reconcile_restored_object_task_results(
            SnapshotCodec.decode_mapping(snapshot),
            checkpoint,
        )

    def _reconcile_restored_wait_states(
        self,
        cur: object | None,
        pids: list[str],
    ) -> None:
        del cur
        # The caller's transaction keeps process-row replacement and the
        # semantic reconciliation transition atomic. The transition service
        # owns validation, compatibility projection, CAS, and generation bumps.
        for pid in pids:
            process = self._processes.get_process(pid)
            if process is None:
                continue
            status, wait_state, message = self._resolved_restored_wait_state(
                process.pid,
                process.status,
                process.wait_state,
            )
            if status is None:
                continue
            self._process_transitions.transition(
                pid,
                status,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                wait_state=wait_state,
                outcome=None,
                status_message=message,
                control=True,
                allowed_statuses={process.status},
                reason="checkpoint restore reconciles a persisted wait",
            )

    def _resolved_restored_wait_state(
        self,
        pid: str,
        status: ProcessStatus,
        wait_state: ProcessWaitState | None,
    ) -> tuple[ProcessStatus | None, ProcessWaitState | None, str | None]:
        if status == ProcessStatus.WAITING_TOOL:
            return self._resolved_restored_tool_wait(pid, wait_state)
        if status == ProcessStatus.WAITING_HUMAN:
            return self._resolved_restored_human_wait(wait_state)
        if status == ProcessStatus.WAITING_EVENT:
            return self._resolved_restored_event_wait(pid, wait_state)
        return None, None, None

    def _resolved_restored_tool_wait(
        self,
        pid: str,
        wait_state: ProcessWaitState | None,
    ) -> tuple[ProcessStatus, PausedProcessWait, str]:
        if not isinstance(wait_state, ToolProcessWait):
            return (
                ProcessStatus.PAUSED,
                PausedProcessWait(),
                "restored tool wait state has no typed ObjectTask identity",
            )
        task = self._processes.get_object_task(wait_state.operation_id)
        if task is None:
            reason = (
                "restored tool wait ObjectTask is missing: "
                f"{wait_state.operation_id}"
            )
        elif task.status not in self.ACTIVE_OBJECT_TASK_STATUSES:
            reason = (
                "restored tool wait ObjectTask is not active: "
                f"{wait_state.operation_id}/{task.status.value}"
            )
        elif task.runner_pid is None or str(task.runner_pid) != pid:
            actual_runner = (
                str(task.runner_pid) if task.runner_pid is not None else "none"
            )
            reason = (
                "restored tool wait ObjectTask runner does not match: "
                f"{wait_state.operation_id}/expected={pid}/actual={actual_runner}"
            )
        else:
            # The restore preflight rejects every scoped nonterminal
            # ObjectTask. Reaching this branch means that invariant was
            # bypassed; keep the process inert instead of reviving work.
            reason = (
                "restored tool wait ObjectTask remained active after preflight: "
                f"{wait_state.operation_id}/{task.status.value}"
            )
        return ProcessStatus.PAUSED, PausedProcessWait(), reason

    def _resolved_restored_human_wait(
        self,
        wait_state: ProcessWaitState | None,
    ) -> tuple[ProcessStatus | None, ProcessWaitState | None, str | None]:
        if not isinstance(wait_state, HumanProcessWait):
            return (
                ProcessStatus.PAUSED,
                PausedProcessWait(),
                "restored human wait state has no typed request identity",
            )
        request_ids = list(wait_state.request_ids)
        requests = [
            self._processes.get_human_request(request_id)
            for request_id in request_ids
        ]
        missing = [
            request_id
            for request_id, request in zip(request_ids, requests)
            if request is None
        ]
        if missing:
            return (
                ProcessStatus.PAUSED,
                PausedProcessWait(),
                f"restored human requests are missing: {','.join(missing)}",
            )
        if any(
            request.status == HumanRequestStatus.PENDING
            for request in requests
            if request is not None
        ):
            return None, None, None
        blocking_rejections = [
            request
            for request in requests
            if request is not None
            and request.status != HumanRequestStatus.APPROVED
            and request.payload.get("type") != "permission_request"
        ]
        if blocking_rejections:
            outcomes = ",".join(
                f"{request.request_id}:{request.status.value}"
                for request in blocking_rejections
            )
            return (
                ProcessStatus.PAUSED,
                PausedProcessWait(),
                f"human requests resolved without approval: {outcomes}",
            )
        return ProcessStatus.RUNNABLE, None, None

    def _resolved_restored_event_wait(
        self,
        pid: str,
        wait_state: ProcessWaitState | None,
    ) -> tuple[ProcessStatus | None, ProcessWaitState | None, str | None]:
        if isinstance(wait_state, ChildProcessWait):
            child = self._processes.get_process(wait_state.child_pid)
            if child is not None and child.status in self.TERMINAL_STATUSES:
                return ProcessStatus.RUNNABLE, None, None
            return None, None, None
        if isinstance(wait_state, MessageProcessWait):
            if self._messages.has_matching_unread_wait(pid, wait_state):
                return ProcessStatus.RUNNABLE, None, None
            return None, None, None
        return (
            ProcessStatus.PAUSED,
            PausedProcessWait(),
            "restored event wait state has no typed child or message identity",
        )

    def _build_current_state_for_diff(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        pids = self._subtree_pids(snapshot["pid"])
        process_rows = list(
            self._snapshot_rows.load_process_snapshot_rows(pids).processes
        )
        object_oids = self._current_scoped_object_oids(pids)
        rows, _payloads = self._snapshot_rows.capture_checkpoint_rows(
            process_rows,
            object_oids=object_oids,
            namespace_names=(),
        )
        return {
            "processes": process_rows,
            "objects": list(rows.objects),
            "capabilities": list(rows.capabilities),
            "process_resource_reservations": list(
                rows.process_resource_reservations
            ),
            "process_messages": list(rows.process_messages),
            "llm_pending_actions": list(rows.llm_pending_actions),
            "tool_candidates": list(rows.tool_candidates),
            "skills": list(rows.skills),
        }

    def _index_rows(self, table: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        key_by_table = {
            "processes": "pid",
            "objects": "oid",
            "capabilities": "cap_id",
            "process_resource_reservations": ("parent_pid", "child_pid"),
            "process_messages": "message_id",
            "llm_pending_actions": "pid",
            "tool_candidates": "candidate_id",
            "skills": "skill_id",
        }
        key = key_by_table[table]
        if isinstance(key, tuple):
            return {":".join(str(row[item]) for item in key): row for row in rows}
        return {str(row[key]): row for row in rows}

    def _snapshot_counts(self, snapshot: dict[str, Any]) -> dict[str, int]:
        return {table: len(rows) for table, rows in snapshot.get("rows", {}).items()}

    def _run_object_release_finalizers(self, object_oids: set[str], *, actor: str, reason: str) -> None:
        self._run_object_release_finalizers_for_objects(
            self._object_release_finalizer_objects(object_oids),
            actor=actor,
            reason=reason,
        )

    def _object_release_finalizer_objects(self, object_oids: set[str]) -> list[Any]:
        if not object_oids:
            return []
        objects = []
        for oid in sorted(object_oids):
            obj = self._objects.get_object(oid)
            if obj is not None:
                objects.append(obj)
        return objects

    def _run_object_release_finalizers_for_objects(self, objects: list[Any], *, actor: str, reason: str) -> None:
        if not objects or self._memory is None:
            return
        run_finalizers = getattr(
            self._memory,
            "run_object_release_finalizers_trusted",
            None,
        )
        if not callable(run_finalizers):
            return
        for obj in objects:
            run_finalizers(obj, actor=actor, reason=reason)

    def _prepare_durable_restore_finalizers(
        self,
        objects: list[Any],
        *,
        publication_id: str,
    ) -> list[dict[str, Any]]:
        if not objects or self._memory is None:
            return []
        prepare = getattr(
            self._memory,
            "prepare_checkpoint_restore_finalizers",
            None,
        )
        if not callable(prepare):
            return []
        return prepare(
            objects,
            publication_id=publication_id,
            actor="checkpoint.restore",
            reason="checkpoint_restore",
            intent_limit_bytes=self.config.checkpoint.payload_capture_limit_bytes,
            total_limit_bytes=self.config.checkpoint.snapshot_hard_limit_bytes,
        )

    def _run_durable_restore_finalizer(self, work_item: Any) -> None:
        if self._memory is None:
            raise ValidationError(
                "checkpoint restore durable finalizer memory service is unavailable"
            )
        run = getattr(
            self._memory,
            "run_checkpoint_restore_finalizer",
            None,
        )
        if not callable(run):
            raise ValidationError(
                "checkpoint restore durable finalizer service is unavailable"
            )
        run(
            work_item,
            actor="checkpoint.restore",
            reason="checkpoint_restore",
        )

    def _insert_row(self, cur: object, table: str, row: dict[str, Any]) -> None:
        """Compatibility-only insert fault hook; persistence lives in the repository."""

        del cur, table, row

    def _remap_namespace(self, namespace: str, pid_map: dict[str, str]) -> str:
        prefix = f"{self.config.memory.process_namespace_prefix}:"
        if namespace.startswith(prefix):
            old_pid = namespace[len(prefix) :]
            if old_pid in pid_map:
                return f"{prefix}{pid_map[old_pid]}"
        if self._objects.namespace_exists(namespace):
            return f"checkpoint_fork/{new_id('ns')}/{namespace}"
        return namespace

    def _remap_process_state_fields(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
        non_clonable_object_oids: set[str],
    ) -> dict[str, Any]:
        """Remap nested typed process identities without consulting legacy text."""

        wait_state = process_wait_state_from_json(row.get("wait_state_json"))
        outcome = process_outcome_from_json(row.get("outcome_json"))
        removed_state_reference = False
        if (
            isinstance(wait_state, (PausedProcessWait, HostResumeProcessWait))
            and wait_state.reason_oid in non_clonable_object_oids
        ):
            wait_state = PausedProcessWait()
            removed_state_reference = True
        if isinstance(outcome, ExitedProcessOutcome):
            if outcome.result_oid in non_clonable_object_oids:
                outcome = ExitedProcessOutcome()
                removed_state_reference = True
        elif isinstance(outcome, FailedProcessOutcome):
            if outcome.result_oid in non_clonable_object_oids:
                outcome = FailedProcessOutcome(code=outcome.code)
                removed_state_reference = True
        elif isinstance(outcome, KilledProcessOutcome):
            if outcome.reason_oid in non_clonable_object_oids:
                outcome = KilledProcessOutcome(code=outcome.code)
                removed_state_reference = True
        wait_state = remap_process_wait_state(
            wait_state,
            pids=identities.pids,
            objects=identities.objects,
        )
        outcome = remap_process_outcome(outcome, objects=identities.objects)
        status = row.get("status")
        status_fallback = None if removed_state_reference else row.get("status_message")
        if status in self.FORK_TRANSIENT_STATUSES:
            status = ProcessStatus.RUNNABLE.value
            wait_state = None
            outcome = None
            status_fallback = None
        return {
            "status": status,
            "wait_state_json": dumps(process_wait_state_to_mapping(wait_state)),
            "outcome_json": dumps(process_outcome_to_mapping(outcome)),
            "status_message": legacy_status_message(
                wait_state,
                outcome,
                status_fallback,
            ),
        }

    @staticmethod
    def _remap_tool_table_json(raw_table: Any, tool_map: Mapping[str, str]) -> str:
        table = loads(raw_table, {})
        return dumps(
            {
                str(name): tool_map.get(str(tool_id), str(tool_id))
                for name, tool_id in table.items()
            }
        )

    def _remap_process_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
        parent_pid: str | None,
        root_pid: str,
        non_clonable_object_oids: set[str],
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        pid_map = identities.pids
        object_map = identities.objects
        capability_map = identities.capabilities
        tool_map = identities.tools
        old_pid = str(row["pid"])
        old_parent = row.get("parent_pid")
        remapped_parent = pid_map.get(old_parent) if old_parent else None
        item["parent_pid"] = remapped_parent if remapped_parent is not None else (parent_pid if old_pid == root_pid else None)
        if row.get("goal_oid") in non_clonable_object_oids:
            item["goal_oid"] = None
        item["checkpoint_head"] = None
        # A checkpoint fork is a new process identity. Never copy the source
        # process's optimistic-CAS or scheduler lease identity into the fork.
        item["revision"] = 0
        item["execution_generation"] = 0
        item["state_generation"] = 0
        item["execution_owner_id"] = None
        item["execution_lease_id"] = None
        item.update(
            self._remap_process_state_fields(
                row,
                identities,
                non_clonable_object_oids,
            )
        )
        item["capabilities_json"] = dumps([capability_map[cap] for cap in loads(item["capabilities_json"], []) if cap in capability_map])
        item["tool_table_json"] = self._remap_tool_table_json(
            item.get("tool_table_json"),
            tool_map,
        )
        item["model_tool_table_json"] = self._remap_tool_table_json(
            item.get("model_tool_table_json"),
            tool_map,
        )
        loaded_skills = loads(item.get("loaded_skills_json"), {})
        for loaded in loaded_skills.values():
            if not isinstance(loaded, dict):
                continue
            for field in ("tool_ids", "jit_tool_ids", "base_tool_ids", "base_model_tool_ids"):
                identifiers = loaded.get(field)
                if not isinstance(identifiers, dict):
                    continue
                loaded[field] = {
                    str(name): tool_map.get(str(tool_id), str(tool_id))
                    for name, tool_id in identifiers.items()
                }
        item["loaded_skills_json"] = dumps(loaded_skills)
        view = loads(item.get("memory_view_json"), {}) if item.get("memory_view_json") else None
        if view:
            view["view_id"] = new_id("view")
            view["owner_pid"] = item["pid"]
            view["created_from"] = None
            roots = []
            for root in view.get("roots", []):
                if root.get("oid") in non_clonable_object_oids:
                    continue
                capability_id = root.get("capability_id")
                if capability_id is not None and capability_id not in capability_map:
                    continue
                if root.get("oid") in object_map:
                    root["oid"] = object_map[root["oid"]]
                if capability_id in capability_map:
                    root["capability_id"] = capability_map[capability_id]
                roots.append(root)
            view["roots"] = roots
            item["memory_view_json"] = dumps(view)
        now = utc_now()
        item["created_at"] = now
        item["updated_at"] = now
        return item

    def _remap_namespace_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        item["updated_at"] = utc_now()
        return item

    def _remap_object_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        pid_map = identities.pids
        object_map = identities.objects
        original_created_by = str(row.get("created_by") or "")
        if item.get("owner_kind") in {ObjectOwnerKind.PROCESS.value, ObjectOwnerKind.PROCESS_RESULT.value}:
            if item.get("owner_id") in pid_map:
                item["owner_id"] = pid_map[item["owner_id"]]
        elif item.get("owner_kind") == ObjectOwnerKind.OBJECT_TASK.value:
            if original_created_by in pid_map:
                item["owner_kind"] = ObjectOwnerKind.PROCESS_RESULT.value
                item["owner_id"] = pid_map[original_created_by]
        elif item.get("owner_kind") is None:
            item["owner_kind"] = ObjectOwnerKind.PROCESS.value
            item["owner_id"] = pid_map.get(str(item.get("created_by")), item.get("created_by"))
        provenance = loads(item["provenance_json"], {})
        provenance["parent_oids"] = [object_map.get(oid, oid) for oid in provenance.get("parent_oids", [])]
        item["provenance_json"] = dumps(provenance)
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        return item

    def _remap_link_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        item["id"] = new_id("link")
        item["created_at"] = utc_now()
        return item

    def _remap_capability_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        object_map = identities.objects
        namespace_map = identities.namespaces
        resource = str(item["resource"])
        if resource.startswith("object:"):
            oid = resource.split(":", 1)[1]
            item["resource"] = f"object:{object_map.get(oid, oid)}"
        elif resource.startswith("object_namespace:"):
            namespace = resource.split(":", 1)[1]
            item["resource"] = f"object_namespace:{namespace_map.get(namespace, namespace)}"
        item["issued_by"] = f"checkpoint.fork:{item['issued_by']}"
        item["issued_at"] = utc_now()
        return item

    def _remap_resource_reservation_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        now = utc_now()
        item["created_at"] = now
        item["updated_at"] = now
        return item

    def _remap_message_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        object_map = identities.objects
        item["message_id"] = new_id("pmsg")
        item["payload_json"] = dumps({**loads(item["payload_json"], {}), "forked_from_message_id": row["message_id"]})
        raw_metadata = item.get("metadata_json")
        metadata = (
            loads(raw_metadata)
            if isinstance(raw_metadata, str)
            else (raw_metadata if raw_metadata is not None else {})
        )
        if not isinstance(metadata, dict):
            raise ValidationError("checkpoint process message metadata must be a JSON object")
        canonical_metadata = self._canonical_snapshot_message_metadata(metadata)
        if canonical_metadata is None:
            raise ValidationError(
                "0.3 checkpoint process message has no canonical label metadata"
            )
        metadata = canonical_metadata
        carrier_oid = str(metadata.get("label_carrier_oid") or "").strip()
        if carrier_oid:
            remapped_carrier_oid = object_map.get(carrier_oid)
            if remapped_carrier_oid is None:
                raise ValidationError(
                    "checkpoint process message label carrier is outside the fork Object scope"
                )
            metadata["label_carrier_oid"] = remapped_carrier_oid
        item["metadata_json"] = dumps(metadata)
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        item["acked_at"] = None
        return item

    def _remap_tool_candidate_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        item = SnapshotRemapper.remap_row(row, identities)
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        return item

    def _remap_tool_row(
        self,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        return SnapshotRemapper.remap_row(row, identities)

    def _remap_object_payload(
        self,
        payload: Any,
        *,
        object_type: str | None,
        candidate_map: dict[str, str],
    ) -> Any:
        item = deepcopy(payload)
        if object_type != ObjectType.TOOL_CANDIDATE.value or not isinstance(item, dict):
            return item
        candidate_id = item.get("candidate_id")
        if candidate_id is not None:
            item["candidate_id"] = candidate_map.get(str(candidate_id), str(candidate_id))
        return item


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
