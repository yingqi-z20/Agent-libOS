from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    Capability,
    CapabilityEffect,
    CapabilityRight,
    CapabilityStatus,
    Checkpoint,
    EventType,
    HumanRequestStatus,
    ObjectOwnerKind,
    ProcessMessageStatus,
    ProcessStatus,
    ToolHandle,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.external_effects import external_effect_summary, external_effect_to_json
from agent_libos.storage import SQLiteStore
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, loads, to_jsonable


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
    FORK_TRANSIENT_STATUSES = {
        ProcessStatus.RUNNING.value,
        ProcessStatus.WAITING_EVENT.value,
        ProcessStatus.WAITING_TOOL.value,
        ProcessStatus.WAITING_HUMAN.value,
    }

    def __init__(
        self,
        store: SQLiteStore,
        audit: AuditManager,
        events: EventBus,
        capabilities: CapabilityManager | None = None,
        *,
        config: AgentLibOSConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.audit = audit
        self.events = events
        self.capabilities = capabilities
        self.runtime: Any | None = None

    def bind_runtime(self, runtime: Any) -> None:
        self.runtime = runtime

    def process_resource(self, pid: str) -> str:
        return f"{self.PROCESS_RESOURCE_PREFIX}{pid}"

    def checkpoint_resource(self, checkpoint_id: str) -> str:
        return f"{self.CHECKPOINT_RESOURCE_PREFIX}{checkpoint_id}"

    def grant_process_defaults(self, pid: str, *, issued_by: str = "checkpoint.process") -> None:
        if self.capabilities is None:
            return
        self.capabilities.grant(
            subject=pid,
            resource=self.process_resource(pid),
            rights=[CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by=issued_by,
        )

    def create(
        self,
        pid: str,
        reason: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        selected_actor = actor or pid
        if require_capability:
            self._require_process_right(selected_actor, pid, CapabilityRight.WRITE)
        checkpoint_id = new_id("ckpt")
        created_at = utc_now()
        snapshot = self._build_snapshot(
            checkpoint_id=checkpoint_id,
            pid=pid,
            reason=reason,
            created_at=created_at,
            created_by=selected_actor,
        )
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            pid=pid,
            reason=reason,
            created_at=created_at,
            created_by=selected_actor,
            snapshot_version=self.config.checkpoint.snapshot_version,
            metadata={
                **(metadata or {}),
                "subtree_pids": snapshot["subtree_pids"],
                "object_count": len(snapshot["object_payloads"]),
                "module_count": len(snapshot.get("modules", [])),
                "snapshot_bytes": len(dumps(snapshot).encode("utf-8")),
            },
        )
        snapshot_bytes = len(dumps(snapshot).encode("utf-8"))
        if snapshot_bytes > self.config.checkpoint.snapshot_hard_limit_bytes:
            raise ValidationError(
                "checkpoint snapshot exceeded "
                f"snapshot_hard_limit_bytes={self.config.checkpoint.snapshot_hard_limit_bytes}"
            )
        self.store.insert_checkpoint(checkpoint, snapshot)
        process = self.store.get_process(pid)
        if process is not None:
            process.checkpoint_head = checkpoint_id
            process.updated_at = utc_now()
            self.store.update_process(process)
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
        if require_capability and actor is not None:
            if pid is None:
                self._require_checkpoint_right(actor, "*", CapabilityRight.READ)
            else:
                self._require_process_right(actor, pid, CapabilityRight.READ)
        selected_limit = self.config.checkpoint.list_limit if limit is None else limit
        return [self._checkpoint_summary(item) for item in self.store.list_checkpoints(pid=pid, limit=selected_limit)]

    def inspect(
        self,
        checkpoint_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability and actor is not None:
            self._require_checkpoint_or_process_read(actor, checkpoint)
        return {
            "checkpoint": self._checkpoint_summary(checkpoint),
            "snapshot_version": snapshot.get("version"),
            "subtree_pids": list(snapshot.get("subtree_pids", [])),
            "modules": list(snapshot.get("modules", [])),
            "counts": self._snapshot_counts(snapshot),
            "processes": [
                {
                    "pid": row["pid"],
                    "parent_pid": row.get("parent_pid"),
                    "image_id": row["image_id"],
                    "status": row["status"],
                    "working_directory": row.get("working_directory", "."),
                    "goal_oid": row.get("goal_oid"),
                }
                for row in snapshot["rows"].get("processes", [])
            ],
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
            self._require_checkpoint_or_process_read(actor, checkpoint)
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
        return {
            "checkpoint_id": checkpoint_id,
            "pid": checkpoint.pid,
            "tables": tables,
            "external_effects_since_checkpoint": self._external_effects_since(checkpoint, snapshot=snapshot),
            "external_effect_summary": self._external_effect_summary_since(checkpoint, snapshot=snapshot),
            "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
        }

    def restore(
        self,
        actor: str,
        checkpoint_id: str,
        *,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability:
            self._require_checkpoint_right(actor, checkpoint_id, CapabilityRight.ADMIN)
        self._require_snapshot_modules(snapshot)
        current_pids = self._subtree_pids(checkpoint.pid)
        snapshot_pids = list(snapshot.get("subtree_pids", []))
        external_effect_pids = self._external_effect_pids(checkpoint, snapshot=snapshot, current_pids=current_pids)
        external_effects = self._external_effects_since(checkpoint, pids=external_effect_pids)
        external_effect_summary = self._external_effect_summary_since(checkpoint, pids=external_effect_pids)
        cancelled_human_requests, superseded_messages = self._restore_scoped_rows(snapshot, current_pids, checkpoint)
        self._restore_images(snapshot)
        self._restore_jit_sources(snapshot)
        self.events.emit(
            EventType.ROLLBACK,
            source=actor,
            target=checkpoint.pid,
            payload={
                "checkpoint_id": checkpoint_id,
                "restored_pids": snapshot_pids,
                "external_effects_since_checkpoint": len(external_effects),
                "external_effect_summary": external_effect_summary,
                "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
            },
        )
        self.audit.record(
            actor=actor,
            action="checkpoint.restore",
            target=self.checkpoint_resource(checkpoint_id),
            decision={
                "restored_for": checkpoint.pid,
                "restored_pids": snapshot_pids,
                "previous_pids": current_pids,
                "cancelled_human_requests": cancelled_human_requests,
                "superseded_messages": superseded_messages,
                "external_effects_since_checkpoint": external_effects,
                "external_effect_summary": external_effect_summary,
                "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
            },
        )
        return {
            "checkpoint_id": checkpoint_id,
            "pid": checkpoint.pid,
            "status": "restored",
            "restored_pids": snapshot_pids,
            "previous_pids": current_pids,
            "cancelled_human_requests": cancelled_human_requests,
            "superseded_messages": superseded_messages,
            "external_effects_since_checkpoint": external_effects,
            "external_effect_summary": external_effect_summary,
            "restore_external_policy": self.RESTORE_EXTERNAL_POLICY,
        }

    def rollback(self, pid: str, checkpoint_id: str) -> dict[str, Any]:
        return self.restore(pid, checkpoint_id, require_capability=False)

    def fork_from_checkpoint(
        self,
        actor: str,
        checkpoint_id: str,
        *,
        parent_pid: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, snapshot = self._load_checkpoint(checkpoint_id)
        self._validate_fork_parent(actor, parent_pid, require_capability=require_capability)
        if require_capability:
            self._require_checkpoint_right(actor, checkpoint_id, CapabilityRight.EXECUTE)
        self._require_snapshot_modules(snapshot)
        if require_capability:
            self._require_snapshot_image_restore_rights(actor, snapshot, overwrite_existing=False)
        remapped = self._remap_snapshot(snapshot, parent_pid=parent_pid)
        self._insert_fork_rows(remapped)
        self._restore_images(snapshot, overwrite_existing=False)
        self._restore_jit_sources(snapshot)
        root_pid = remapped["pid_map"][checkpoint.pid]
        self.events.emit(
            EventType.PROCESS_FORKED,
            source=actor,
            target=root_pid,
            payload={"checkpoint_id": checkpoint_id, "source_pid": checkpoint.pid, "fork_root_pid": root_pid},
        )
        self.audit.record(
            actor=actor,
            action="checkpoint.fork",
            target=self.checkpoint_resource(checkpoint_id),
            decision={"source_pid": checkpoint.pid, "fork_root_pid": root_pid, "pid_map": remapped["pid_map"]},
        )
        return {
            "checkpoint_id": checkpoint_id,
            "source_pid": checkpoint.pid,
            "fork_root_pid": root_pid,
            "pid_map": remapped["pid_map"],
            "object_map": remapped["object_map"],
        }

    def replay_to_event(
        self,
        checkpoint_id: str,
        event_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
    ) -> dict[str, Any]:
        checkpoint, _snapshot = self._load_checkpoint(checkpoint_id)
        if require_capability and actor is not None:
            self._require_checkpoint_or_process_read(actor, checkpoint)
        events = self.store.list_events()
        scoped_pids = set(self._subtree_pids(checkpoint.pid))
        selected = []
        reached = False
        for event in events:
            if event.created_at < checkpoint.created_at:
                continue
            if event.source not in scoped_pids and event.target not in scoped_pids:
                continue
            selected.append(
                {
                    "event_id": event.event_id,
                    "type": event.type.value,
                    "source": event.source,
                    "target": event.target,
                    "created_at": event.created_at,
                    "payload": event.payload,
                }
            )
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
        process_rows = [self._safe_point_process_row(row, checkpoint_id) for row in self._rows_by_ids("processes", "pid", subtree_pids)]
        object_oids = self._scoped_object_oids(process_rows, subtree_pids)
        namespace_names = self._scoped_namespaces(object_oids, subtree_pids)
        capability_rows = self._capability_rows_for_subjects(subtree_pids)
        rows = {
            "processes": process_rows,
            "object_namespaces": self._rows_by_ids("object_namespaces", "namespace", namespace_names),
            "objects": self._rows_by_ids("objects", "oid", object_oids),
            "object_links": self._link_rows_for_objects(object_oids),
            "capabilities": capability_rows,
            "process_resource_reservations": self._resource_reservation_rows_for_processes(subtree_pids),
            "process_messages": self._message_rows_for_recipients(subtree_pids),
            "llm_pending_actions": self._rows_by_ids("llm_pending_actions", "pid", subtree_pids),
            "skills": self._skill_rows_for_processes(process_rows),
            "tools": self._tool_rows_for_processes(process_rows),
            "tool_candidates": self._rows_by_ids("tool_candidates", "pid", subtree_pids),
        }
        object_payloads = self._object_payload_snapshot(object_oids)
        return {
            "version": self.config.checkpoint.snapshot_version,
            "checkpoint_id": checkpoint_id,
            "pid": pid,
            "reason": reason,
            "created_at": created_at,
            "created_by": created_by,
            "subtree_pids": subtree_pids,
            "object_oids": object_oids,
            "namespaces": namespace_names,
            "rows": rows,
            "object_payloads": object_payloads,
            "images": self._image_snapshot(process_rows),
            "image_artifacts": self._image_artifact_snapshot(process_rows),
            "jit_sources": self._jit_source_snapshot(process_rows),
            "modules": self._module_snapshot(),
        }

    def _module_snapshot(self) -> list[dict[str, Any]]:
        if self.runtime is None or not hasattr(self.runtime, "modules"):
            return []
        return self.runtime.modules.loaded_module_summaries()

    def _require_snapshot_modules(self, snapshot: dict[str, Any]) -> None:
        if self.runtime is None or not hasattr(self.runtime, "modules"):
            return
        missing = []
        for module in snapshot.get("modules", []):
            module_id = str(module.get("module_id", ""))
            source_sha256 = str(module.get("source_sha256", ""))
            if not module_id:
                continue
            if not self.runtime.modules.is_loaded(module_id, source_sha256 or None):
                missing.append({"module_id": module_id, "source_sha256": source_sha256})
        if missing:
            raise ValidationError(f"checkpoint requires startup modules that are not loaded: {missing}")

    def _restore_scoped_rows(
        self,
        snapshot: dict[str, Any],
        current_pids: list[str],
        checkpoint: Checkpoint,
    ) -> tuple[list[str], list[str]]:
        rows = snapshot["rows"]
        object_oids = set(snapshot.get("object_oids", [])) | set(self._current_scoped_object_oids(current_pids))
        namespace_names = set(snapshot.get("namespaces", [])) | set(self._current_scoped_namespaces(current_pids))
        restored_capability_rows = self._filtered_restored_capability_rows(rows.get("capabilities", []))
        with self.store.transaction(include_object_payloads=True) as cur:
            # Pending human/message state belongs to the same reconstructable
            # restore boundary as process rows. If a later insert fails, these
            # status changes must roll back with the SQLite rows and in-memory
            # object payloads.
            cancelled_human_requests = self._cancel_pending_human_requests(cur, current_pids, checkpoint)
            superseded_messages = self._supersede_post_checkpoint_messages(cur, current_pids, checkpoint)
            self._delete_object_links(cur, object_oids)
            self._delete_rows_by_ids(cur, "objects", "oid", object_oids)
            self._delete_object_capabilities(cur, object_oids)
            for oid in object_oids:
                self.store.forget_object_payload(oid)
            self._delete_rows_by_ids(cur, "object_namespaces", "namespace", namespace_names)
            self._delete_non_checkpoint_capabilities(cur, current_pids)
            self._delete_resource_reservations(cur, current_pids)
            self._delete_rows_by_ids(cur, "llm_pending_actions", "pid", current_pids)
            self._delete_rows_by_ids(cur, "tool_candidates", "pid", current_pids)
            self._delete_rows_by_ids(cur, "processes", "pid", current_pids)
            for row in rows.get("object_namespaces", []):
                self._insert_row(cur, "object_namespaces", row)
            for row in rows.get("objects", []):
                item = dict(row)
                item["payload_json"] = dumps(self.store._memory_payload_marker(present=True))
                self._insert_row(cur, "objects", item)
                oid = str(item["oid"])
                if oid in snapshot["object_payloads"]:
                    self.store.set_object_payload(oid, deepcopy(snapshot["object_payloads"][oid]))
            for row in rows.get("object_links", []):
                self._insert_row(cur, "object_links", row)
            for row in restored_capability_rows:
                self._insert_row(cur, "capabilities", row)
            for row in rows.get("process_resource_reservations", []):
                self._insert_row(cur, "process_resource_reservations", row)
            for row in rows.get("tool_candidates", []):
                self._insert_row(cur, "tool_candidates", row)
            for row in rows.get("skills", []):
                self._upsert_row(cur, "skills", row, "skill_id")
            for row in rows.get("tools", []):
                exists = cur.execute("SELECT 1 FROM tools WHERE tool_id = ?", (row["tool_id"],)).fetchone()
                if exists is None:
                    self._insert_row(cur, "tools", row)
            for row in rows.get("process_messages", []):
                self._upsert_row(cur, "process_messages", row, "message_id")
            for row in rows.get("llm_pending_actions", []):
                self._upsert_row(cur, "llm_pending_actions", row, "pid")
            for row in rows.get("processes", []):
                self._insert_row(cur, "processes", row)
            self._reconcile_restored_wait_states(cur, [str(row["pid"]) for row in rows.get("processes", [])])
        return cancelled_human_requests, superseded_messages

    def _remap_snapshot(self, snapshot: dict[str, Any], *, parent_pid: str | None) -> dict[str, Any]:
        original_pids = list(snapshot["subtree_pids"])
        pid_map = {pid: new_id("pid") for pid in original_pids}
        object_map = {oid: new_id("obj") for oid in snapshot.get("object_oids", [])}
        namespace_map = {
            namespace: self._remap_namespace(namespace, pid_map)
            for namespace in snapshot.get("namespaces", [])
        }
        rows = deepcopy(snapshot["rows"])
        rows["capabilities"] = self._fork_capability_rows(rows.get("capabilities", []))
        capability_map = {row["cap_id"]: new_id("cap") for row in rows.get("capabilities", [])}
        rows["processes"] = [
            self._remap_process_row(row, pid_map, object_map, capability_map, parent_pid)
            for row in rows.get("processes", [])
        ]
        rows["object_namespaces"] = [
            self._remap_namespace_row(row, pid_map, namespace_map)
            for row in rows.get("object_namespaces", [])
        ]
        rows["objects"] = [
            self._remap_object_row(row, pid_map, object_map, namespace_map)
            for row in rows.get("objects", [])
        ]
        rows["object_links"] = [
            self._remap_link_row(row, object_map)
            for row in rows.get("object_links", [])
            if row["src_oid"] in object_map and row["dst_oid"] in object_map
        ]
        rows["capabilities"] = [
            self._remap_capability_row(row, pid_map, object_map, namespace_map, capability_map)
            for row in rows.get("capabilities", [])
            if row["subject"] in pid_map and not str(row["resource"]).startswith(self.CHECKPOINT_RESOURCE_PREFIX)
        ]
        rows["process_resource_reservations"] = [
            self._remap_resource_reservation_row(row, pid_map)
            for row in rows.get("process_resource_reservations", [])
            if row["parent_pid"] in pid_map and row["child_pid"] in pid_map
        ]
        rows["process_messages"] = [
            self._remap_message_row(row, pid_map)
            for row in rows.get("process_messages", [])
            if row["recipient_pid"] in pid_map
        ]
        rows["llm_pending_actions"] = []
        rows["tool_candidates"] = [
            self._remap_tool_candidate_row(row, pid_map)
            for row in rows.get("tool_candidates", [])
            if row["pid"] in pid_map
        ]
        payloads = {
            object_map[oid]: deepcopy(payload)
            for oid, payload in snapshot.get("object_payloads", {}).items()
            if oid in object_map
        }
        return {
            "rows": rows,
            "object_payloads": payloads,
            "pid_map": pid_map,
            "object_map": object_map,
            "namespace_map": namespace_map,
            "capability_map": capability_map,
        }

    def _fork_capability_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("resource", "")).startswith(self.CHECKPOINT_RESOURCE_PREFIX):
                continue
            current = self.store.get_capability(str(row.get("cap_id")))
            if current is None or not current.active:
                continue
            if self._capability_is_expired(current):
                continue
            item = dict(row)
            item["uses_remaining"] = current.uses_remaining
            item["status"] = current.status.value
            item["rights_json"] = dumps(sorted(current.rights))
            item["effect"] = current.effect.value
            if current.effect == CapabilityEffect.ALLOW:
                allowed_rights = self._currently_allowed_fork_rights(current)
                if not allowed_rights:
                    continue
                item["rights_json"] = dumps(allowed_rights)
            kept.append(item)
        return kept

    def _capability_is_expired(self, capability: Any) -> bool:
        if self.capabilities is None:
            return False
        return bool(self.capabilities._is_expired(capability))

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
        restrictive = self._current_restrictive_capabilities(capability.subject)
        allowed: list[str] = []
        for right in sorted(capability.rights):
            if any(
                right in cap.rights and self._resources_may_overlap(capability.resource, cap.resource)
                for cap in restrictive
            ):
                continue
            decision = self.capabilities.authorize(
                capability.subject,
                capability.resource,
                right,
                {"primitive": "checkpoint", "operation": "fork_capability_filter"},
            )
            if decision.allowed and decision.selected_capability_id == capability.cap_id:
                allowed.append(right)
        return allowed

    def _filter_restored_capability_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if str(row.get("resource", "")).startswith(self.CHECKPOINT_RESOURCE_PREFIX):
            return None
        current = self.store.get_capability(str(row.get("cap_id")))
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

    def _current_restrictive_capabilities(self, subject: str) -> list[Any]:
        if self.capabilities is None:
            return []
        result = []
        for cap in self.store.list_capabilities(subject=subject):
            if not cap.active or self._capability_is_expired(cap):
                continue
            if cap.effect in {CapabilityEffect.DENY, CapabilityEffect.ASK}:
                result.append(cap)
        return result

    def _resources_may_overlap(self, left: str, right: str) -> bool:
        if self.capabilities is None:
            return left == right
        try:
            left_pattern = self.capabilities.parse_resource_pattern(left)
            right_pattern = self.capabilities.parse_resource_pattern(right)
        except CapabilityDenied:
            return left == right
        if left_pattern.kind != right_pattern.kind:
            return False
        if self.capabilities._resource_matches(left, right) or self.capabilities._resource_matches(right, left):
            return True
        left_has_wildcard = left.endswith(":*") or left.endswith("/*")
        right_has_wildcard = right.endswith(":*") or right.endswith("/*")
        if left_has_wildcard or right_has_wildcard:
            return left_pattern.body.startswith(right_pattern.body) or right_pattern.body.startswith(left_pattern.body)
        return False

    def _insert_fork_rows(self, remapped: dict[str, Any]) -> None:
        rows = remapped["rows"]
        with self.store.transaction(include_object_payloads=True) as cur:
            for row in rows.get("object_namespaces", []):
                if cur.execute("SELECT 1 FROM object_namespaces WHERE namespace = ?", (row["namespace"],)).fetchone() is None:
                    self._insert_row(cur, "object_namespaces", row)
            for row in rows.get("objects", []):
                item = dict(row)
                item["payload_json"] = dumps(self.store._memory_payload_marker(present=True))
                self._insert_row(cur, "objects", item)
                self.store.set_object_payload(item["oid"], deepcopy(remapped["object_payloads"][item["oid"]]))
            for table in [
                "object_links",
                "capabilities",
                "process_resource_reservations",
                "process_messages",
                "llm_pending_actions",
                "tool_candidates",
            ]:
                for row in rows.get(table, []):
                    self._insert_row(cur, table, row)
            for row in rows.get("skills", []):
                self._upsert_row(cur, "skills", row, "skill_id")
            for row in rows.get("tools", []):
                exists = cur.execute("SELECT 1 FROM tools WHERE tool_id = ?", (row["tool_id"],)).fetchone()
                if exists is None:
                    self._insert_row(cur, "tools", row)
            for row in rows.get("processes", []):
                self._insert_row(cur, "processes", row)

    def _load_checkpoint(self, checkpoint_id: str) -> tuple[Checkpoint, dict[str, Any]]:
        found = self.store.get_checkpoint_snapshot(checkpoint_id)
        if found is None:
            raise NotFound(f"checkpoint not found: {checkpoint_id}")
        return found

    def _checkpoint_summary(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "pid": checkpoint.pid,
            "reason": checkpoint.reason,
            "created_at": checkpoint.created_at,
            "created_by": checkpoint.created_by,
            "snapshot_version": checkpoint.snapshot_version,
            "metadata": checkpoint.metadata or {},
        }

    def _require_process_right(self, actor: str, pid: str, right: CapabilityRight) -> None:
        if self.capabilities is None:
            return
        self.capabilities.require(actor, self.process_resource(pid), right)

    def _require_checkpoint_right(self, actor: str, checkpoint_id: str, right: CapabilityRight) -> None:
        if self.capabilities is None:
            return
        resource = "checkpoint:*" if checkpoint_id == "*" else self.checkpoint_resource(checkpoint_id)
        self.capabilities.require(actor, resource, right)

    def _require_checkpoint_or_process_read(self, actor: str, checkpoint: Checkpoint) -> None:
        if self.capabilities is None:
            return
        if self.capabilities.check(actor, self.checkpoint_resource(checkpoint.checkpoint_id), CapabilityRight.READ):
            return
        if self.capabilities.check(actor, self.process_resource(checkpoint.pid), CapabilityRight.READ):
            return
        raise CapabilityDenied(f"{actor} lacks read on checkpoint {checkpoint.checkpoint_id}")

    def _validate_fork_parent(self, actor: str, parent_pid: str | None, *, require_capability: bool) -> None:
        if parent_pid is None:
            return
        if self.store.get_process(parent_pid) is None:
            raise NotFound(f"process not found: {parent_pid}")
        if not require_capability or actor == parent_pid:
            return
        if self.capabilities is None:
            raise CapabilityDenied("checkpoint fork parent attachment requires a capability manager")
        self.capabilities.require(actor, self.process_resource(parent_pid), CapabilityRight.ADMIN)

    def _subtree_pids(self, root_pid: str) -> list[str]:
        processes = {process.pid: process for process in self.store.list_processes()}
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
        oids: set[str] = set()
        for row in process_rows:
            if row.get("goal_oid"):
                oids.add(str(row["goal_oid"]))
            view = loads(row.get("memory_view_json"), {}) if row.get("memory_view_json") else {}
            for root in view.get("roots", []):
                if isinstance(root, dict) and root.get("oid"):
                    oids.add(str(root["oid"]))
        for pid in pids:
            for owner_kind in (ObjectOwnerKind.PROCESS, ObjectOwnerKind.PROCESS_RESULT):
                for obj in self.store.list_objects_owned_by(owner_kind, pid):
                    oids.add(obj.oid)
        return sorted(oid for oid in oids if oid in self.store._object_payloads)

    def _current_scoped_object_oids(self, pids: list[str]) -> list[str]:
        process_rows = self._rows_by_ids("processes", "pid", pids)
        return self._scoped_object_oids(process_rows, pids)

    def _scoped_namespaces(self, object_oids: list[str], pids: list[str]) -> list[str]:
        namespaces = {
            row["namespace"]
            for row in self._rows_by_ids("objects", "oid", object_oids)
        }
        process_namespaces = {f"{self.config.memory.process_namespace_prefix}:{pid}" for pid in pids}
        namespaces |= process_namespaces
        for pid in pids:
            for namespace in self.store.list_namespaces_created_by(pid):
                namespaces.add(namespace.namespace)
        for namespace in list(namespaces):
            found = self.store.get_namespace(namespace)
            if found is not None:
                namespaces.add(found.namespace)
        return sorted(namespaces)

    def _current_scoped_namespaces(self, pids: list[str]) -> list[str]:
        return self._scoped_namespaces(self._current_scoped_object_oids(pids), pids)

    def _rows_by_ids(self, table: str, column: str, values: Iterable[str]) -> list[dict[str, Any]]:
        selected = list(dict.fromkeys(values))
        if not selected:
            return []
        placeholders = ", ".join("?" for _ in selected)
        return self.store.select_table_rows(table, f"{column} IN ({placeholders})", selected, order_by=column)

    def _link_rows_for_objects(self, object_oids: list[str]) -> list[dict[str, Any]]:
        if not object_oids:
            return []
        placeholders = ", ".join("?" for _ in object_oids)
        return self.store.select_table_rows(
            "object_links",
            f"src_oid IN ({placeholders}) OR dst_oid IN ({placeholders})",
            [*object_oids, *object_oids],
            order_by="id",
        )

    def _capability_rows_for_subjects(self, pids: list[str]) -> list[dict[str, Any]]:
        rows = self._rows_by_ids("capabilities", "subject", pids)
        return [row for row in rows if not str(row["resource"]).startswith(self.CHECKPOINT_RESOURCE_PREFIX)]

    def _message_rows_for_recipients(self, pids: list[str]) -> list[dict[str, Any]]:
        return self._rows_by_ids("process_messages", "recipient_pid", pids)

    def _resource_reservation_rows_for_processes(self, pids: list[str]) -> list[dict[str, Any]]:
        selected = set(pids)
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for pid in pids:
            for row in self.store.select_table_rows("process_resource_reservations", "parent_pid = ?", (pid,)):
                if row["child_pid"] in selected:
                    rows[(row["parent_pid"], row["child_pid"])] = row
            for row in self.store.select_table_rows("process_resource_reservations", "child_pid = ?", (pid,)):
                if row["parent_pid"] in selected:
                    rows[(row["parent_pid"], row["child_pid"])] = row
        return [rows[key] for key in sorted(rows)]

    def _tool_rows_for_processes(self, process_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_ids: set[str] = set()
        for row in process_rows:
            for tool_id in loads(row.get("tool_table_json"), {}).values():
                tool_ids.add(str(tool_id))
        return self._rows_by_ids("tools", "tool_id", sorted(tool_ids))

    def _skill_rows_for_processes(self, process_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        skill_ids = self._loaded_skill_ids(process_rows)
        return self._rows_by_ids("skills", "skill_id", sorted(skill_ids))

    def _loaded_skill_ids(self, process_rows: list[dict[str, Any]]) -> set[str]:
        skill_ids: set[str] = set()
        for row in process_rows:
            for skill_id in loads(row.get("loaded_skills_json"), {}).keys():
                skill_ids.add(str(skill_id))
        return skill_ids

    def _image_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {}
        image_ids = {row["image_id"] for row in process_rows}
        return {
            image_id: to_jsonable(runtime.images[image_id])
            for image_id in sorted(image_ids)
            if image_id in runtime.images
        }

    def _image_artifact_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, Any]:
        runtime = self.runtime
        if runtime is None:
            return {}
        artifacts: dict[str, Any] = {}
        for row in process_rows:
            image = runtime.images.get(row["image_id"])
            if image is None or image.boot.get("kind") != "checkpoint_commit":
                continue
            artifact_id = str(image.boot.get("artifact_id") or "")
            if not artifact_id:
                continue
            found = runtime.store.get_image_artifact(artifact_id)
            if found is None:
                continue
            artifact, metadata = found
            artifacts[artifact_id] = {"artifact": artifact, **metadata}
        return artifacts

    def _snapshot_image_ids_to_restore(self, snapshot: dict[str, Any], *, overwrite_existing: bool) -> list[str]:
        runtime = self.runtime
        if runtime is None:
            return []
        image_ids: list[str] = []
        for image_id in snapshot.get("images", {}):
            if not overwrite_existing and image_id in runtime.images:
                continue
            image_ids.append(str(image_id))
        return image_ids

    def _require_snapshot_image_restore_rights(
        self,
        actor: str,
        snapshot: dict[str, Any],
        *,
        overwrite_existing: bool,
    ) -> None:
        runtime = self.runtime
        if self.capabilities is None or runtime is None:
            return
        registry = getattr(runtime, "image_registry", None)
        for image_id in self._snapshot_image_ids_to_restore(snapshot, overwrite_existing=overwrite_existing):
            resource = registry.resource_for(image_id) if registry is not None else f"image:{image_id}"
            self.capabilities.require(actor, resource, CapabilityRight.WRITE)

    def _restore_images(self, snapshot: dict[str, Any], *, overwrite_existing: bool = True) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        restored_artifact_ids: set[str] | None = set() if not overwrite_existing else None
        for image_id, data in snapshot.get("images", {}).items():
            if not overwrite_existing and image_id in runtime.images:
                continue
            image = AgentImage(**data)
            runtime.images[image_id] = image
            runtime.store.upsert_image(
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
            if runtime.store.get_image_artifact(artifact_id) is not None:
                continue
            runtime.store.insert_image_artifact(
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
            if oid not in self.store._object_payloads:
                continue
            payload = deepcopy(self.store.object_payload(oid))
            payload_bytes = len(dumps(payload).encode("utf-8"))
            if payload_bytes > limit:
                raise ValidationError(
                    f"object payload {oid} exceeds checkpoint payload_capture_limit_bytes={limit}"
                )
            payloads[oid] = payload
        return payloads

    def _jit_source_snapshot(self, process_rows: list[dict[str, Any]]) -> dict[str, str]:
        runtime = self.runtime
        tools = getattr(runtime, "tools", None)
        if tools is None:
            return {}
        sources = getattr(tools, "_jit_sources", {})
        tool_ids: set[str] = set()
        for row in process_rows:
            for tool_id in loads(row.get("tool_table_json"), {}).values():
                tool_ids.add(str(tool_id))
        return {tool_id: sources[tool_id] for tool_id in sorted(tool_ids) if tool_id in sources}

    def _restore_jit_sources(self, snapshot: dict[str, Any]) -> None:
        runtime = self.runtime
        tools = getattr(runtime, "tools", None)
        if tools is None:
            return
        sources = getattr(tools, "_jit_sources", None)
        handles = getattr(tools, "_handles", None)
        names = getattr(tools, "_tool_ids_by_name", None)
        if sources is None or handles is None:
            return
        tool_rows = {row["tool_id"]: row for row in snapshot.get("rows", {}).get("tools", [])}
        for tool_id, source in snapshot.get("jit_sources", {}).items():
            sources[tool_id] = source
            row = tool_rows.get(tool_id)
            if row is None:
                continue
            handle = ToolHandle(
                tool_id=tool_id,
                name=row["name"],
                capability_id=None,
                scope=row["scope"],
            )
            handles[tool_id] = handle
            if names is not None and names.get(handle.name) == tool_id:
                names.pop(handle.name, None)

    def _external_effects_since(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        pids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_pids = self._external_effect_pids(checkpoint, snapshot=snapshot, current_pids=pids)
        records = self.store.list_external_effects(created_after=checkpoint.created_at, pids=selected_pids)
        return [external_effect_to_json(record) for record in records]

    def _external_effect_summary_since(
        self,
        checkpoint: Checkpoint,
        *,
        snapshot: dict[str, Any] | None = None,
        pids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected_pids = self._external_effect_pids(checkpoint, snapshot=snapshot, current_pids=pids)
        return external_effect_summary(self.store.list_external_effects(created_after=checkpoint.created_at, pids=selected_pids))

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

    def _cancel_pending_human_requests(self, cur: Any, pids: list[str], checkpoint: Checkpoint) -> list[str]:
        cancelled: list[str] = []
        for pid in pids:
            for request in self.store.list_human_requests(pid=pid, status=HumanRequestStatus.PENDING):
                if request.created_at <= checkpoint.created_at:
                    continue
                cur.execute(
                    """
                    UPDATE human_requests
                       SET status = ?, decision_json = ?, updated_at = ?
                     WHERE request_id = ?
                    """,
                    (
                        HumanRequestStatus.CANCELLED.value,
                        dumps({"cancelled_by": f"checkpoint:{checkpoint.checkpoint_id}"}),
                        utc_now(),
                        request.request_id,
                    ),
                )
                cancelled.append(request.request_id)
        return cancelled

    def _supersede_post_checkpoint_messages(self, cur: Any, pids: list[str], checkpoint: Checkpoint) -> list[str]:
        superseded: list[str] = []
        for pid in pids:
            for message in self.store.list_process_messages(pid, status=ProcessMessageStatus.UNREAD):
                if message.created_at <= checkpoint.created_at:
                    continue
                payload = {
                    **message.payload,
                    "superseded_by_restore": checkpoint.checkpoint_id,
                    "superseded_at": utc_now(),
                }
                cur.execute(
                    """
                    UPDATE process_messages
                       SET payload_json = ?, status = ?, updated_at = ?
                     WHERE message_id = ?
                    """,
                    (
                        dumps(payload),
                        ProcessMessageStatus.SUPERSEDED_BY_RESTORE.value,
                        utc_now(),
                        message.message_id,
                    ),
                )
                superseded.append(message.message_id)
        return superseded

    def _reconcile_restored_wait_states(self, cur: Any, pids: list[str]) -> None:
        for pid in pids:
            process = self.store.get_process(pid)
            if process is None:
                continue
            status, message = self._resolved_restored_wait_state(process.pid, process.status, process.status_message)
            if status is None:
                continue
            cur.execute(
                """
                UPDATE processes
                   SET status = ?, status_message = ?, updated_at = ?
                 WHERE pid = ?
                """,
                (status.value, message, utc_now(), pid),
            )

    def _resolved_restored_wait_state(
        self,
        pid: str,
        status: ProcessStatus,
        status_message: str | None,
    ) -> tuple[ProcessStatus | None, str | None]:
        if status == ProcessStatus.WAITING_HUMAN:
            request_id = self._human_request_id_from_wait(status_message)
            request = self.store.get_human_request(request_id) if request_id is not None else None
            if request is None or request.status == HumanRequestStatus.PENDING:
                return None, None
            if request.status == HumanRequestStatus.APPROVED or request.payload.get("type") == "permission_request":
                return ProcessStatus.RUNNABLE, None
            return ProcessStatus.PAUSED, f"human request {request_id} resolved as {request.status.value}"
        if status != ProcessStatus.WAITING_EVENT:
            return None, None
        child_pid = self._child_pid_from_wait(status_message)
        if child_pid is not None:
            child = self.store.get_process(child_pid)
            if child is not None and child.status in self.TERMINAL_STATUSES:
                return ProcessStatus.RUNNABLE, None
        runtime = getattr(self, "runtime", None)
        messages = getattr(runtime, "messages", None)
        if messages is not None:
            filters = messages._filters_from_wait_status(status_message)
            if filters is not None and self.store.list_process_messages(
                pid,
                status=ProcessMessageStatus.UNREAD,
                kind=filters.get("kind"),
                sender=filters.get("sender"),
                channel=filters.get("channel"),
                correlation_id=filters.get("correlation_id"),
                reply_to=filters.get("reply_to"),
                message_ids=filters.get("message_ids"),
            ):
                return ProcessStatus.RUNNABLE, None
        return None, None

    def _human_request_id_from_wait(self, status_message: str | None) -> str | None:
        prefix = "waiting for human request "
        if not status_message or not status_message.startswith(prefix):
            return None
        return status_message[len(prefix) :].strip() or None

    def _child_pid_from_wait(self, status_message: str | None) -> str | None:
        prefix = "waiting for "
        if not status_message or not status_message.startswith(prefix):
            return None
        child_pid = status_message[len(prefix) :].strip()
        return child_pid or None

    def _build_current_state_for_diff(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        pids = self._subtree_pids(snapshot["pid"])
        process_rows = self._rows_by_ids("processes", "pid", pids)
        object_oids = self._current_scoped_object_oids(pids)
        return {
            "processes": process_rows,
            "objects": self._rows_by_ids("objects", "oid", object_oids),
            "capabilities": self._capability_rows_for_subjects(pids),
            "process_resource_reservations": self._resource_reservation_rows_for_processes(pids),
            "process_messages": self._message_rows_for_recipients(pids),
            "llm_pending_actions": self._rows_by_ids("llm_pending_actions", "pid", pids),
            "tool_candidates": self._rows_by_ids("tool_candidates", "pid", pids),
            "skills": self._skill_rows_for_processes(process_rows),
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

    def _delete_object_links(self, cur: Any, object_oids: set[str]) -> None:
        if not object_oids:
            return
        placeholders = ", ".join("?" for _ in object_oids)
        cur.execute(
            f"DELETE FROM object_links WHERE src_oid IN ({placeholders}) OR dst_oid IN ({placeholders})",
            [*object_oids, *object_oids],
        )

    def _delete_object_capabilities(self, cur: Any, object_oids: set[str]) -> None:
        if not object_oids:
            return
        resources = [f"object:{oid}" for oid in sorted(object_oids)]
        placeholders = ", ".join("?" for _ in resources)
        cur.execute(f"DELETE FROM capabilities WHERE resource IN ({placeholders})", resources)

    def _delete_rows_by_ids(self, cur: Any, table: str, column: str, values: Iterable[str]) -> None:
        selected = list(dict.fromkeys(values))
        if not selected:
            return
        placeholders = ", ".join("?" for _ in selected)
        cur.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", selected)

    def _delete_non_checkpoint_capabilities(self, cur: Any, pids: list[str]) -> None:
        if not pids:
            return
        placeholders = ", ".join("?" for _ in pids)
        cur.execute(
            f"DELETE FROM capabilities WHERE subject IN ({placeholders}) AND resource NOT LIKE 'checkpoint:%'",
            pids,
        )

    def _delete_resource_reservations(self, cur: Any, pids: list[str]) -> None:
        if not pids:
            return
        placeholders = ", ".join("?" for _ in pids)
        cur.execute(
            f"""
            DELETE FROM process_resource_reservations
             WHERE parent_pid IN ({placeholders})
                OR child_pid IN ({placeholders})
            """,
            [*pids, *pids],
        )

    def _insert_row(self, cur: Any, table: str, row: dict[str, Any]) -> None:
        item = self._object_row_with_lifecycle_defaults(row) if table == "objects" else row
        columns = list(item)
        placeholders = ", ".join("?" for _ in columns)
        cur.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(item[column] for column in columns),
        )

    def _object_row_with_lifecycle_defaults(self, row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        created_by = str(item.get("created_by", "runtime"))
        if not item.get("owner_kind"):
            if created_by.startswith("process_result:"):
                item["owner_kind"] = ObjectOwnerKind.PROCESS_RESULT.value
            elif created_by.startswith("object_task:"):
                item["owner_kind"] = ObjectOwnerKind.OBJECT_TASK.value
            elif created_by == "runtime" or created_by.startswith("runtime."):
                item["owner_kind"] = ObjectOwnerKind.RUNTIME.value
            else:
                item["owner_kind"] = ObjectOwnerKind.PROCESS.value
        if not item.get("owner_id"):
            if created_by.startswith("process_result:"):
                item["owner_id"] = created_by.split(":", 1)[1]
            elif created_by.startswith("object_task:"):
                item["owner_id"] = created_by.split(":", 1)[1]
            else:
                item["owner_id"] = created_by
        item.setdefault("lifecycle_state", "live")
        item.setdefault("deleted_at", None)
        return item

    def _upsert_row(self, cur: Any, table: str, row: dict[str, Any], key: str) -> None:
        columns = list(row)
        assignments = ", ".join(f"{column} = excluded.{column}" for column in columns if column != key)
        placeholders = ", ".join("?" for _ in columns)
        cur.execute(
            f"""
            INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})
            ON CONFLICT({key}) DO UPDATE SET {assignments}
            """,
            tuple(row[column] for column in columns),
        )

    def _remap_namespace(self, namespace: str, pid_map: dict[str, str]) -> str:
        prefix = f"{self.config.memory.process_namespace_prefix}:"
        if namespace.startswith(prefix):
            old_pid = namespace[len(prefix) :]
            if old_pid in pid_map:
                return f"{prefix}{pid_map[old_pid]}"
        if self.store.namespace_exists(namespace):
            return f"checkpoint_fork/{new_id('ns')}/{namespace}"
        return namespace

    def _remap_process_row(
        self,
        row: dict[str, Any],
        pid_map: dict[str, str],
        object_map: dict[str, str],
        capability_map: dict[str, str],
        parent_pid: str | None,
    ) -> dict[str, Any]:
        item = dict(row)
        old_pid = item["pid"]
        item["pid"] = pid_map[old_pid]
        old_parent = item.get("parent_pid")
        item["parent_pid"] = pid_map.get(old_parent) if old_parent else parent_pid
        if item.get("goal_oid") in object_map:
            item["goal_oid"] = object_map[item["goal_oid"]]
        item["checkpoint_head"] = None
        if item.get("status") in self.FORK_TRANSIENT_STATUSES:
            item["status"] = ProcessStatus.RUNNABLE.value
            item["status_message"] = None
        item["capabilities_json"] = dumps([capability_map[cap] for cap in loads(item["capabilities_json"], []) if cap in capability_map])
        view = loads(item.get("memory_view_json"), {}) if item.get("memory_view_json") else None
        if view:
            view["view_id"] = new_id("view")
            view["owner_pid"] = item["pid"]
            view["created_from"] = None
            roots = []
            for root in view.get("roots", []):
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
        pid_map: dict[str, str],
        namespace_map: dict[str, str],
    ) -> dict[str, Any]:
        item = dict(row)
        item["namespace"] = namespace_map.get(item["namespace"], item["namespace"])
        if item.get("parent_namespace") in namespace_map:
            item["parent_namespace"] = namespace_map[item["parent_namespace"]]
        if item.get("created_by") in pid_map:
            item["created_by"] = pid_map[item["created_by"]]
        item["updated_at"] = utc_now()
        return item

    def _remap_object_row(
        self,
        row: dict[str, Any],
        pid_map: dict[str, str],
        object_map: dict[str, str],
        namespace_map: dict[str, str],
    ) -> dict[str, Any]:
        item = dict(row)
        item["oid"] = object_map[item["oid"]]
        item["namespace"] = namespace_map.get(item["namespace"], item["namespace"])
        if item.get("created_by") in pid_map:
            item["created_by"] = pid_map[item["created_by"]]
        if item.get("owner_kind") in {ObjectOwnerKind.PROCESS.value, ObjectOwnerKind.PROCESS_RESULT.value}:
            if item.get("owner_id") in pid_map:
                item["owner_id"] = pid_map[item["owner_id"]]
        elif item.get("owner_kind") is None:
            item["owner_kind"] = ObjectOwnerKind.PROCESS.value
            item["owner_id"] = pid_map.get(str(item.get("created_by")), item.get("created_by"))
        provenance = loads(item["provenance_json"], {})
        provenance["parent_oids"] = [object_map.get(oid, oid) for oid in provenance.get("parent_oids", [])]
        item["provenance_json"] = dumps(provenance)
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        return item

    def _remap_link_row(self, row: dict[str, Any], object_map: dict[str, str]) -> dict[str, Any]:
        item = dict(row)
        item["id"] = new_id("link")
        item["src_oid"] = object_map[item["src_oid"]]
        item["dst_oid"] = object_map[item["dst_oid"]]
        item["created_at"] = utc_now()
        return item

    def _remap_capability_row(
        self,
        row: dict[str, Any],
        pid_map: dict[str, str],
        object_map: dict[str, str],
        namespace_map: dict[str, str],
        capability_map: dict[str, str],
    ) -> dict[str, Any]:
        item = dict(row)
        item["cap_id"] = capability_map[item["cap_id"]]
        item["subject"] = pid_map[item["subject"]]
        if item.get("issuer_cap_id") in capability_map:
            item["issuer_cap_id"] = capability_map[item["issuer_cap_id"]]
        if item.get("parent_cap_id") in capability_map:
            item["parent_cap_id"] = capability_map[item["parent_cap_id"]]
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

    def _remap_resource_reservation_row(self, row: dict[str, Any], pid_map: dict[str, str]) -> dict[str, Any]:
        item = dict(row)
        item["parent_pid"] = pid_map[item["parent_pid"]]
        item["child_pid"] = pid_map[item["child_pid"]]
        now = utc_now()
        item["created_at"] = now
        item["updated_at"] = now
        return item

    def _remap_message_row(self, row: dict[str, Any], pid_map: dict[str, str]) -> dict[str, Any]:
        item = dict(row)
        item["message_id"] = new_id("pmsg")
        item["recipient_pid"] = pid_map[item["recipient_pid"]]
        if item["sender"] in pid_map:
            item["sender"] = pid_map[item["sender"]]
        item["payload_json"] = dumps({**loads(item["payload_json"], {}), "forked_from_message_id": row["message_id"]})
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        item["acked_at"] = None
        return item

    def _remap_tool_candidate_row(self, row: dict[str, Any], pid_map: dict[str, str]) -> dict[str, Any]:
        item = dict(row)
        item["candidate_id"] = new_id("tcand")
        item["pid"] = pid_map[item["pid"]]
        item["created_at"] = utc_now()
        item["updated_at"] = utc_now()
        return item
