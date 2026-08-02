from __future__ import annotations

from typing import TYPE_CHECKING

from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models.snapshot import ExecRollbackState, SNAPSHOT_SCHEMA_VERSION
from agent_libos.runtime.snapshots.codec import SnapshotCodec
from agent_libos.storage import SnapshotCheckpointRepositoryProtocol
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import loads

if TYPE_CHECKING:
    from agent_libos.tools.broker import ToolBroker


class ProcessExecStateService:
    """Capture and atomically restore the process-local state changed by exec."""

    def __init__(
        self,
        snapshots: SnapshotCheckpointRepositoryProtocol,
        memory: ObjectMemoryManager,
        tools: ToolBroker,
    ) -> None:
        self._snapshots = snapshots
        self._memory = memory
        self._tools = tools

    def capture(self, pid: str) -> ExecRollbackState:
        rows, object_payloads = self._snapshots.capture_process_exec_snapshot_rows(
            pid,
            process_namespace=self._memory.process_namespace(pid),
        )
        process_rows = list(rows.processes)
        object_rows = list(rows.objects)
        object_oids = [str(row["oid"]) for row in object_rows]
        namespace_rows = list(rows.object_namespaces)
        tool_ids = set(loads(process_rows[0].get("tool_table_json"), {}).values())
        tool_handles, jit_sources = self._tools.snapshot_loaded_tool_state(tool_ids)
        created_at = utc_now()
        snapshot = SnapshotCodec.decode_mapping(
            {
                "version": SNAPSHOT_SCHEMA_VERSION,
                "checkpoint_id": f"exec:{pid}:{created_at}",
                "pid": pid,
                "reason": "process.exec rollback",
                "created_at": created_at,
                "created_by": "runtime.process.exec",
                "subtree_pids": [pid],
                "object_oids": object_oids,
                "owned_object_oids": object_oids,
                "referenced_object_oids": [],
                "referenced_object_types": {},
                "namespaces": [str(row["namespace"]) for row in namespace_rows],
                "owned_namespaces": [
                    str(row["namespace"])
                    for row in namespace_rows
                ],
                "rows": rows.to_mapping(),
                "object_payloads": object_payloads,
                "images": {},
                "image_artifacts": {},
                "jit_sources": jit_sources,
                "modules": [],
            }
        )
        return ExecRollbackState(
            snapshot=snapshot,
            tool_ids=frozenset(str(tool_id) for tool_id in tool_ids),
            tool_handles=tool_handles,
        )

    def restore(
        self,
        state: ExecRollbackState,
        *,
        fence_execution: bool = True,
    ) -> None:
        snapshot = state.snapshot
        pid = snapshot.header.root_pid
        stale_tool_ids = self._snapshots.restore_process_exec_snapshot(
            snapshot,
            process_namespace=self._memory.process_namespace(pid),
            captured_tool_ids=state.tool_ids,
            capability_rollback_token=state.capability_rollback_token,
            fence_execution=fence_execution,
        )
        self._prune_stale_tools(stale_tool_ids, pid)
        self._tools.restore_loaded_jit_state(
            state.tool_handles,
            dict(snapshot.jit_sources),
        )

    def _prune_stale_tools(self, tool_ids: frozenset[str], pid: str) -> None:
        for tool_id in tool_ids:
            if self._snapshots.delete_tool_if_unreferenced(
                tool_id,
                excluding_pid=pid,
            ):
                self._tools.forget_loaded_jit(tool_id)


__all__ = ["ProcessExecStateService"]
