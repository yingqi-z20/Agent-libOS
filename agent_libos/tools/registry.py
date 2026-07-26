from __future__ import annotations

import hashlib
from copy import deepcopy
from collections.abc import Iterable
from typing import Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.models import ToolHandle
from agent_libos.models.exceptions import NotFound
from agent_libos.ports import AuditPort
from agent_libos.storage import UnitOfWork
from agent_libos.tools.base import BaseAgentTool
from agent_libos.utils.ids import new_id, utc_now


class ToolRegistry:
    """Own tool identity, implementation publication, and durable rows."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        audit: AuditPort,
        config: AgentLibOSConfig,
    ) -> None:
        self.unit_of_work = unit_of_work
        self.extensions = unit_of_work.extensions
        self.processes = unit_of_work.processes
        self.audit = audit
        self.config = config
        self._implementations: dict[str, BaseAgentTool] = {}
        self._tool_ids_by_name: dict[str, str] = {}
        self._handles: dict[str, ToolHandle] = {}
        self._jit_sources: dict[str, str] = {}

    def register(
        self,
        tool: BaseAgentTool,
        *,
        registered_by: str,
        scope: str,
        ephemeral: bool,
    ) -> ToolHandle:
        spec = tool.spec(config=self.config)
        if spec.name in self._tool_ids_by_name:
            raise ValueError(f"tool already registered: {spec.name}")
        tool_id = (
            new_id("tool")
            if ephemeral
            else stable_static_tool_id(
                spec.name,
                digest_chars=self.config.tools.static_tool_id_digest_chars,
            )
        )
        handle = ToolHandle(
            tool_id=tool_id,
            name=spec.name,
            capability_id=None,
            scope=scope,
        )
        existing = next(
            (
                row
                for row in self.extensions.list_tools()
                if row["tool_id"] == tool_id
            ),
            None,
        )
        if existing is not None and existing["name"] != spec.name:
            raise ValueError(f"tool id collision: {tool_id}")
        try:
            with self.unit_of_work.transaction():
                if existing is None:
                    self.extensions.insert_tool(
                        handle,
                        spec,
                        registered_by=registered_by,
                        created_at=utc_now(),
                        ephemeral=ephemeral,
                    )
                else:
                    self.extensions.update_tool(
                        handle,
                        spec,
                        registered_by=registered_by,
                        ephemeral=ephemeral,
                    )
                self._implementations[tool_id] = tool
                self._tool_ids_by_name[spec.name] = tool_id
                self._handles[tool_id] = handle
                self.audit.record(
                    actor=registered_by,
                    action="tool.register",
                    target=f"tool:{tool_id}",
                    decision={
                        "name": spec.name,
                        "version": spec.version,
                        "policy": spec.policy,
                        "tags": spec.tags,
                    },
                )
        except BaseException:
            self._implementations.pop(tool_id, None)
            if self._tool_ids_by_name.get(spec.name) == tool_id:
                self._tool_ids_by_name.pop(spec.name, None)
            self._handles.pop(tool_id, None)
            raise
        return handle

    def unregister(
        self,
        tool: ToolHandle | str,
        *,
        registered_by: str | None,
    ) -> bool:
        handle = self.handle_for_unregistration(tool)
        if handle is None:
            return False
        row = next(
            (
                item
                for item in self.extensions.list_tools()
                if item["tool_id"] == handle.tool_id
            ),
            None,
        )
        if (
            registered_by is not None
            and row is not None
            and row.get("registered_by") != registered_by
        ):
            return False
        implementation_present = handle.tool_id in self._implementations
        implementation = self._implementations.get(handle.tool_id)
        jit_source_present = handle.tool_id in self._jit_sources
        jit_source = self._jit_sources.get(handle.tool_id)
        loaded_handle_present = handle.tool_id in self._handles
        loaded_handle = self._handles.get(handle.tool_id)
        name_binding_present = handle.name in self._tool_ids_by_name
        name_binding = self._tool_ids_by_name.get(handle.name)
        try:
            with self.unit_of_work.transaction():
                self._implementations.pop(handle.tool_id, None)
                self._jit_sources.pop(handle.tool_id, None)
                self._handles.pop(handle.tool_id, None)
                if self._tool_ids_by_name.get(handle.name) == handle.tool_id:
                    self._tool_ids_by_name.pop(handle.name, None)
                self.extensions.delete_tool(
                    handle.tool_id,
                    registered_by=registered_by,
                )
                self.audit.record(
                    actor=registered_by or "tool_broker",
                    action="tool.unregister",
                    target=f"tool:{handle.tool_id}",
                    decision={"name": handle.name},
                )
        except BaseException:
            if implementation_present:
                assert implementation is not None
                self._implementations[handle.tool_id] = implementation
            else:
                self._implementations.pop(handle.tool_id, None)
            if jit_source_present:
                assert jit_source is not None
                self._jit_sources[handle.tool_id] = jit_source
            else:
                self._jit_sources.pop(handle.tool_id, None)
            if loaded_handle_present:
                assert loaded_handle is not None
                self._handles[handle.tool_id] = loaded_handle
            else:
                self._handles.pop(handle.tool_id, None)
            if name_binding_present:
                assert name_binding is not None
                self._tool_ids_by_name[handle.name] = name_binding
            else:
                self._tool_ids_by_name.pop(handle.name, None)
            raise
        return True

    def discard_loaded_registration(self, handle: ToolHandle) -> bool:
        """Fail closed by removing only the captured in-memory registration."""

        current = self._handles.get(handle.tool_id)
        if current is not None and current is not handle:
            return False
        changed = any(
            (
                handle.tool_id in self._implementations,
                handle.tool_id in self._jit_sources,
                handle.tool_id in self._handles,
                self._tool_ids_by_name.get(handle.name) == handle.tool_id,
            )
        )
        self._implementations.pop(handle.tool_id, None)
        self._jit_sources.pop(handle.tool_id, None)
        self._handles.pop(handle.tool_id, None)
        if self._tool_ids_by_name.get(handle.name) == handle.tool_id:
            self._tool_ids_by_name.pop(handle.name, None)
        return changed

    def resolve(self, tool: ToolHandle | str, *, pid: str | None = None) -> ToolHandle:
        if isinstance(tool, ToolHandle):
            handle = self._handles.get(tool.tool_id)
            if (
                handle is None
                or handle != tool
                or (
                    tool.tool_id not in self._implementations
                    and tool.tool_id not in self._jit_sources
                )
            ):
                raise NotFound(f"tool not found: {tool.tool_id}")
            return handle
        process_tool_id: str | None = None
        process_tool_ids: set[str] = set()
        if pid is not None:
            process = self.processes.get_process(pid)
            if process is not None:
                process_tool_ids = {str(value) for value in process.tool_table.values()}
                if tool in process.tool_table:
                    process_tool_id = str(process.tool_table[tool])
                    if process_tool_id in self._handles:
                        return self._handles[process_tool_id]
        if tool in self._handles:
            handle = self._handles[tool]
            if pid is None and handle.tool_id in self._jit_sources:
                raise NotFound(f"tool not found: {tool}")
            return handle
        if tool in self._tool_ids_by_name:
            return self._handles[self._tool_ids_by_name[tool]]
        return self._load_persisted_handle(
            str(tool),
            pid=pid,
            process_tool_id=process_tool_id,
            process_tool_ids=process_tool_ids,
        )

    def _load_persisted_handle(
        self,
        tool: str,
        *,
        pid: str | None,
        process_tool_id: str | None,
        process_tool_ids: set[str],
    ) -> ToolHandle:
        for row in self.extensions.list_tools():
            row_tool_id = str(row["tool_id"])
            direct_id = row_tool_id == tool
            process_name = process_tool_id is not None and row_tool_id == process_tool_id
            name_match = row["name"] == tool
            if bool(row["ephemeral"]):
                if pid is None or row_tool_id not in process_tool_ids:
                    continue
                if not (direct_id or process_name):
                    continue
            elif not (direct_id or name_match):
                continue
            if row_tool_id not in self._implementations and row_tool_id not in self._jit_sources:
                raise NotFound(f"tool implementation not loaded: {row_tool_id}")
            handle = ToolHandle(
                tool_id=row_tool_id,
                name=row["name"],
                capability_id=None,
                scope=row["scope"],
            )
            self._handles[handle.tool_id] = handle
            if not bool(row["ephemeral"]):
                self._tool_ids_by_name.setdefault(handle.name, handle.tool_id)
            return handle
        raise NotFound(f"tool not found: {tool}")

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            from agent_libos.models.exceptions import ValidationError

            raise ValidationError("tool list limit must be a positive integer")
        return self.extensions.list_tools(limit=limit)

    def name_collides_with_static_tool(self, name: str) -> bool:
        mapped = self._tool_ids_by_name.get(name)
        if mapped in self._implementations:
            return True
        return any(
            row["name"] == name and not bool(row["ephemeral"])
            for row in self.extensions.list_tools()
        )

    def process_has_tool(self, pid: str, handle: ToolHandle) -> bool:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process.tool_table.get(handle.name) == handle.tool_id

    def handle_for_unregistration(
        self,
        tool: ToolHandle | str,
    ) -> ToolHandle | None:
        if isinstance(tool, ToolHandle):
            loaded = self._handles.get(tool.tool_id)
            return loaded if loaded == tool else None
        if tool in self._handles:
            return self._handles[tool]
        tool_id = self._tool_ids_by_name.get(str(tool))
        return self._handles.get(tool_id) if tool_id is not None else None

    def implementation(self, tool_id: str) -> BaseAgentTool | None:
        """Return a loaded static implementation without exposing registry state."""

        return self._implementations.get(tool_id)

    def implementation_ids(self) -> frozenset[str]:
        return frozenset(self._implementations)

    def handle(self, tool_id: str) -> ToolHandle | None:
        return self._handles.get(tool_id)

    def loaded_handles(self) -> tuple[ToolHandle, ...]:
        return tuple(
            self._handles[tool_id]
            for tool_id in sorted(self._handles)
        )

    def jit_source(self, tool_id: str) -> str | None:
        return self._jit_sources.get(tool_id)

    def is_jit(self, tool_id: str) -> bool:
        return tool_id in self._jit_sources

    def jit_ids(self) -> frozenset[str]:
        return frozenset(self._jit_sources)

    def publish_jit(self, handle: ToolHandle, source: str) -> None:
        """Publish one already-persisted JIT implementation atomically in memory."""

        existing_handle = self._handles.get(handle.tool_id)
        existing_source = self._jit_sources.get(handle.tool_id)
        if existing_handle is not None and existing_handle != handle:
            raise ValueError(f"tool handle collision: {handle.tool_id}")
        if existing_source is not None and existing_source != source:
            raise ValueError(f"JIT source collision: {handle.tool_id}")
        self._handles[handle.tool_id] = handle
        self._jit_sources[handle.tool_id] = source

    def forget_jit(self, tool_id: str) -> None:
        """Remove only the in-memory JIT implementation for a durable tool id."""

        self._jit_sources.pop(tool_id, None)
        self._handles.pop(tool_id, None)

    def snapshot_loaded_state(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[dict[str, ToolHandle], dict[str, str]]:
        selected = {str(tool_id) for tool_id in tool_ids}
        handles = {
            tool_id: deepcopy(handle)
            for tool_id, handle in self._handles.items()
            if tool_id in selected
        }
        sources = {
            tool_id: deepcopy(source)
            for tool_id, source in self._jit_sources.items()
            if tool_id in selected
        }
        return handles, sources

    def restore_loaded_jit_state(
        self,
        handles: dict[str, ToolHandle],
        sources: dict[str, str],
    ) -> None:
        """Restore a trusted rollback snapshot through collision-checked APIs."""

        for tool_id, source in sources.items():
            handle = handles.get(tool_id)
            if handle is None:
                raise ValueError(f"missing JIT handle for rollback source: {tool_id}")
            self.publish_jit(deepcopy(handle), deepcopy(source))


def stable_static_tool_id(name: str, *, digest_chars: int) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:digest_chars]
    return f"tool_static_{digest}"


__all__ = ["ToolRegistry", "stable_static_tool_id"]
