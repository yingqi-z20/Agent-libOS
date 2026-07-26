from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    ForkMode,
    MemoryView,
    MemoryViewSpec,
    ObjectHandle,
    ObjectMetadata,
)
from agent_libos.models.exceptions import NotFound, ValidationError


class ProcessLaunchService:
    """Authority and path policy for spawn/fork/working-directory changes."""

    def __init__(
        self,
        *,
        process: Any,
        capabilities: Any,
        filesystem: Any,
        images: Mapping[str, AgentImage],
        image_resource: Callable[[str], str],
        config: AgentLibOSConfig,
    ) -> None:
        self._process = process
        self._capabilities = capabilities
        self._filesystem = filesystem
        self._images = images
        self._image_resource = image_resource
        self._config = config

    def require_image(self, image_id: str) -> AgentImage:
        image = self._images.get(image_id)
        if image is None:
            raise NotFound(f"agent image not found: {image_id}")
        return image

    def require_spawn_authority(
        self,
        pid: str,
        *,
        image_id: str | None = None,
        operation: str = "process.spawn",
        consume: bool = True,
    ) -> Any:
        return self._capabilities.require(
            pid,
            "process:spawn",
            CapabilityRight.WRITE,
            {
                "primitive": "runtime.process_launch",
                "operation": operation,
                "authority_operation": operation,
                **({"image_id": image_id} if image_id is not None else {}),
            },
            consume=consume,
        )

    def require_image_boot_authority(
        self,
        pid: str,
        image_id: str,
        *,
        consume: bool = True,
    ) -> Any:
        return self._capabilities.require(
            pid,
            self._image_resource(image_id),
            CapabilityRight.READ,
            consume=consume,
        )

    def resolve_llm_profile_id(
        self,
        image_id: str,
        explicit_profile_id: str | None,
    ) -> str:
        if explicit_profile_id is not None:
            selected = str(explicit_profile_id).strip()
            if not selected:
                raise ValidationError("LLM profile id must be a non-empty string")
            return selected
        image = self._images.get(image_id)
        if image is not None and image.llm_profile_id:
            return image.llm_profile_id
        return self._config.llm.default_profile_id

    def resolve_working_directory(self, pid: str, path: str) -> str:
        self._require_workspace_relative_working_directory(path)
        current_cwd = self._process.working_directory(pid)
        return self._filesystem.validate_directory(pid, path, cwd=current_cwd)

    def set_working_directory(self, pid: str, path: str) -> Any:
        relative = self.resolve_working_directory(pid, path)
        return self._process.set_validated_working_directory(pid, relative)

    def spawn_child(
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
        selected_image, selected_cwd, authority_reservations = self._child_preflight(
            parent,
            image=image,
            working_directory=working_directory,
            operation="process.spawn_child",
        )
        try:
            return self._process.spawn_child(
                parent=parent,
                goal=goal,
                image=selected_image,
                inherit_capabilities=inherit_capabilities,
                resource_budget=resource_budget,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
                _launch_authority_reservations=authority_reservations,
                _validated_working_directory=selected_cwd,
            )
        except BaseException:
            self._restore_launch_authority(parent, authority_reservations)
            raise

    def fork_child(
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
        selected_image, selected_cwd, authority_reservations = self._child_preflight(
            parent,
            image=image,
            working_directory=working_directory,
            operation="process.fork",
        )
        try:
            return self._process.fork(
                parent=parent,
                goal=goal,
                memory_view=memory_view,
                capabilities=capabilities,
                inherit_capabilities=inherit_capabilities,
                resource_budget=resource_budget,
                image=selected_image,
                mode=mode,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
                _launch_authority_reservations=authority_reservations,
                _validated_working_directory=selected_cwd,
            )
        except BaseException:
            self._restore_launch_authority(parent, authority_reservations)
            raise

    def _child_preflight(
        self,
        parent: str,
        *,
        image: str | None,
        working_directory: str | None,
        operation: str,
    ) -> tuple[str, str, tuple[str, ...]]:
        parent_process = self._process.get(parent)
        selected_image = image or parent_process.image_id
        decisions = [self.require_spawn_authority(
            parent,
            image_id=selected_image,
            operation=operation,
            consume=False,
        )]
        if selected_image != parent_process.image_id:
            decisions.append(
                self.require_image_boot_authority(
                    parent,
                    selected_image,
                    consume=False,
                )
            )
        reservations: list[str] = []
        try:
            for decision in decisions:
                reservation_id = self._capabilities.reserve_decision_use(
                    decision,
                    used_by=f"{operation}:{parent}",
                    reason="one-time process launch authority reserved",
                )
                if reservation_id is not None:
                    reservations.append(reservation_id)
            # Image existence and path validation are deliberately after
            # authorization but before finite-use settlement.
            self.require_image(selected_image)
            selected_cwd = (
                self.resolve_working_directory(parent, working_directory)
                if working_directory is not None
                else parent_process.working_directory
            )
            return selected_image, selected_cwd, tuple(reservations)
        except BaseException:
            self._restore_launch_authority(parent, reservations)
            raise

    @staticmethod
    def _require_workspace_relative_working_directory(path: str) -> None:
        has_windows_drive = os.name == "nt" and bool(os.path.splitdrive(path)[0])
        if os.path.isabs(path) or has_windows_drive:
            raise ValidationError("working directory must be workspace-relative")

    def _restore_launch_authority(
        self,
        parent: str,
        reservation_ids: Iterable[str],
    ) -> None:
        for reservation_id in reservation_ids:
            self._capabilities.restore_reserved_use(
                reservation_id,
                restored_by=f"process.launch:{parent}",
                reason="process launch did not commit",
            )


__all__ = ["ProcessLaunchService"]
