from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    ResourceUsage,
    ToolHandle,
    ToolSpec,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.ports import AuditPort
from agent_libos.runtime.image_artifact import ImageArtifactLoader
from agent_libos.storage.repositories import (
    ExtensionRepository,
    ProcessRepository,
    RuntimePublicationRepository,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)

if TYPE_CHECKING:
    from agent_libos.tools.broker import ToolBroker


class ImagePackageInstaller:
    """Materialize and compensate one validated image-package boot."""

    def __init__(
        self,
        *,
        loader: ImageArtifactLoader,
        processes: ProcessRepository,
        publications: RuntimePublicationRepository,
        extensions: ExtensionRepository,
        tools: ToolBroker,
        filesystem: Any,
        resources: Any,
        audit: AuditPort,
        workspace_root: str | Path,
        config: AgentLibOSConfig,
    ) -> None:
        self._loader = loader
        self._processes = processes
        self._publications = publications
        self._extensions = extensions
        self._tools = tools
        self._filesystem = filesystem
        self._resources = resources
        self._audit = audit
        self._workspace_root = Path(workspace_root).resolve()
        self._config = config

    def preflight(self, image: AgentImage) -> None:
        self._loader.load(image, expected_kind="image_package")

    def planned_workspace_root(
        self,
        pid: str,
        image: AgentImage,
        *,
        materialization_id: str,
    ) -> str:
        return (
            Path(self._config.image.materialized_workspace_root)
            / self._safe_segment(pid)
            / self._safe_segment(materialization_id)
            / self._safe_segment(str(image.boot.get("artifact_id") or "image"))
            / "workspace"
        ).as_posix()

    def install(
        self,
        pid: str,
        image: AgentImage,
        *,
        workspace_root: str | None = None,
        publication_id: str | None = None,
    ) -> None:
        artifact = self._loader.load(image, expected_kind="image_package")
        materialized_workspace_root: str | None = None
        working_directory: str | None = None
        registered_jit: list[str] = []
        try:
            workspace_paths = self._materialize_workspace(
                pid,
                image,
                artifact,
                workspace_root=workspace_root,
                publication_id=publication_id,
            )
            if workspace_paths is not None:
                materialized_workspace_root, working_directory = workspace_paths
            registered_jit = self._register_jit_tools(
                pid,
                image,
                artifact,
                publication_id=publication_id,
            )
            process = self._require_process(pid)
            if workspace_paths is not None:
                process.working_directory = working_directory
                process.updated_at = utc_now()
                self._processes.patch_process(
                    pid,
                    {
                        "working_directory": process.working_directory,
                        "updated_at": process.updated_at,
                    },
                    expected_revision=process.revision,
                )
                self._grant_workspace(
                    pid,
                    image,
                    artifact,
                    materialized_workspace_root,
                    publication_id=publication_id,
                )
        except Exception:
            self._remove_registered_jit_tool_names(pid, registered_jit)
            self._cleanup_materialized_workspace(
                materialized_workspace_root or workspace_root,
                actor=f"image:{image.image_id}",
                reason="image_package_boot_failed",
            )
            raise
        self._audit.record(
            actor=f"image:{image.image_id}",
            action="image.boot.package",
            target=f"process:{pid}",
            decision={
                "image_id": image.image_id,
                "artifact_id": image.boot.get("artifact_id"),
                "package_sha256": artifact.get("package_sha256"),
                "workspace_root": materialized_workspace_root,
                "working_directory": working_directory,
                "jit_tools": registered_jit,
                "publication_id": publication_id,
            },
        )

    def cleanup(
        self,
        pid: str,
        image: AgentImage | str,
        *,
        reason: str,
        workspace_root: str | None = None,
        strict: bool = False,
    ) -> None:
        image_id = image.image_id if isinstance(image, AgentImage) else str(image)
        process = self._processes.get_process(pid)
        tool_rows = {
            str(row.get("tool_id")): row
            for row in self._extensions.list_tools()
        }
        handles: dict[str, ToolHandle] = {}
        if process is not None:
            for name, tool_id in list(process.tool_table.items()):
                row = tool_rows.get(str(tool_id))
                if row is None or not bool(row.get("ephemeral")):
                    continue
                if str(row.get("registered_by")) != f"image.package:{image_id}":
                    continue
                handle = self._tools.loaded_tool_handle(tool_id)
                handles[name] = handle or ToolHandle(
                    tool_id=tool_id,
                    name=name,
                    capability_id=None,
                    scope=str(row.get("scope") or "ephemeral_process"),
                )
        self._remove_registered_jit_tools(pid, handles)
        self._cleanup_materialized_workspace(
            workspace_root
            or self._materialized_workspace_root_for_cwd(
                process.working_directory if process is not None else None
            ),
            actor=f"image:{image_id}",
            reason=reason,
            strict=strict,
        )

    def cleanup_publication_workspace(
        self,
        workspace_root: str,
        *,
        reason: str,
    ) -> None:
        """Delete only one exact publication-planned workspace tree."""

        self._cleanup_materialized_workspace(
            workspace_root,
            actor="runtime.publication",
            reason=reason,
            strict=True,
        )

    def publication_workspace_exists(self, workspace_root: str) -> bool:
        root = self._validated_materialized_root(workspace_root)
        return bool(root is not None and root.exists())

    def _materialize_workspace(
        self,
        pid: str,
        image: AgentImage,
        artifact: dict[str, Any],
        *,
        workspace_root: str | None = None,
        publication_id: str | None = None,
    ) -> tuple[str, str] | None:
        workspace = artifact.get("workspace") or {}
        source = workspace.get("source")
        if not source:
            return None
        root_relative = Path(
            workspace_root
            or self.planned_workspace_root(
                pid,
                image,
                materialization_id=new_id("boot"),
            )
        )
        root = self._validated_materialized_root(root_relative.as_posix())
        if root is None:
            raise ValidationError("invalid planned image workspace root")
        files = [
            record
            for record in artifact.get("files", [])
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
        self._record_publication_artifact(
            publication_id,
            {
                "artifact_id": f"workspace_intent:{root_relative.as_posix()}",
                "kind": "workspace",
                "path": root_relative.as_posix(),
                "status": "intent",
            },
        )
        self._resources.preflight(
            pid,
            usage,
            source="image.workspace.materialize",
            context=context,
        )
        try:
            cwd = self._write_workspace(root, workspace, files, str(source))
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            self._prune_empty_parents(root.parent)
            raise
        self._resources.charge(
            pid,
            usage,
            source="image.workspace.materialize",
            context=context,
        )
        return (
            root.relative_to(self._workspace_root).as_posix(),
            cwd.relative_to(self._workspace_root).as_posix(),
        )

    def _write_workspace(
        self,
        root: Path,
        workspace: dict[str, Any],
        files: list[dict[str, Any]],
        source: str,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for record in files:
            package_path = str(record["path"])
            relative = self._relative_artifact_path(package_path, source)
            target = (root / relative).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(
                    f"image workspace file escaped materialized root: {package_path}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._artifact_file_bytes(record))
        working_directory = str(workspace.get("working_directory") or ".")
        cwd = (root if working_directory == "." else root / working_directory).resolve()
        if root not in cwd.parents and cwd != root:
            raise RuntimeError(
                "image workspace working_directory escaped materialized root"
            )
        cwd.mkdir(parents=True, exist_ok=True)
        return cwd

    def _grant_workspace(
        self,
        pid: str,
        image: AgentImage,
        artifact: dict[str, Any],
        workspace_root: str,
        *,
        publication_id: str | None = None,
    ) -> None:
        workspace = artifact.get("workspace") or {}
        granted: list[dict[str, Any]] = []
        for grant in workspace.get("grants", []):
            relative = str(grant.get("path") or ".")
            target = (
                workspace_root
                if relative == "."
                else f"{workspace_root.rstrip('/')}/{relative}"
            )
            rights = [CapabilityRight(right) for right in grant.get("rights", [])]
            grant_method = (
                self._filesystem.grant_directory
                if grant["recursive"]
                else self._filesystem.grant_path
            )
            with self._processes.transaction():
                capability = grant_method(
                    pid,
                    target,
                    rights,
                    issued_by=f"image.package:{image.image_id}",
                    delegable=grant["delegable"],
                    metadata=(
                        {
                            "runtime_publication_id": publication_id,
                            "runtime_publication_kind": "image_workspace_grant",
                        }
                        if publication_id is not None
                        else None
                    ),
                )
                self._record_publication_artifact(
                    publication_id,
                    {
                        "artifact_id": f"capability:{capability.cap_id}",
                        "kind": "capability",
                        "capability_id": capability.cap_id,
                        "resource": capability.resource,
                    },
                )
            granted.append(
                {
                    "capability_id": capability.cap_id,
                    "resource": capability.resource,
                    "rights": sorted(capability.rights),
                }
            )
        if granted:
            self._audit.record(
                actor=f"image:{image.image_id}",
                action="image.workspace.grants",
                target=f"process:{pid}",
                decision={"grants": granted},
            )

    def _register_jit_tools(
        self,
        pid: str,
        image: AgentImage,
        artifact: dict[str, Any],
        *,
        publication_id: str | None = None,
    ) -> list[str]:
        process = self._require_process(pid)
        prepared: list[tuple[str, str]] = []
        candidate_ids: list[str] = []
        handles: dict[str, ToolHandle] = {}
        try:
            for item in artifact.get("jit_tools", []):
                candidate = self._prepare_jit_candidate(
                    pid,
                    image,
                    process,
                    item,
                    publication_id=publication_id,
                )
                if candidate is not None:
                    prepared.append(candidate)
                    candidate_ids.append(candidate[1])
            for name, candidate_id in prepared:
                handles[name] = self._tools.register(
                    pid,
                    candidate_id,
                    approver=(
                        f"publication:{publication_id}"
                        if publication_id is not None
                        else f"image.package:{image.image_id}"
                    ),
                    publication_id=publication_id,
                )
            registered = sorted(handles)
            if registered:
                self._audit.record(
                    actor=f"image:{image.image_id}",
                    action="image.package_jit.register",
                    target=f"process:{pid}",
                    decision={"tools": registered},
                )
            return registered
        except Exception:
            self._remove_registered_jit_tools(pid, handles)
            for candidate_id in reversed(candidate_ids):
                self._tools.discard_candidate(
                    pid,
                    candidate_id,
                    discarded_by=f"image.package:{image.image_id}",
                    reason="image_package_jit_boot_failed",
                )
            raise

    def _prepare_jit_candidate(
        self,
        pid: str,
        image: AgentImage,
        process: Any,
        item: dict[str, Any],
        *,
        publication_id: str | None = None,
    ) -> tuple[str, str] | None:
        name = str(item.get("name") or "")
        if not name:
            return None
        if name in process.tool_table:
            raise RuntimeError(
                f"image package JIT tool conflicts with visible tool: {name}"
            )
        if self._tools.name_collides_with_static_tool(name):
            raise RuntimeError(
                f"image package JIT tool conflicts with static tool: {name}"
            )
        spec = ToolSpec(
            name=name,
            description=str(item.get("description") or ""),
            input_schema=dict(item.get("input_schema") or {}),
            output_schema=dict(item.get("output_schema") or {}),
            policy=(
                {"sandbox_timeout_s": float(item["timeout_s"])}
                if item.get("timeout_s") is not None
                else {}
            ),
            tags=["image", "jit", "package"],
            metadata={
                "image_id": image.image_id,
                "artifact_id": image.boot.get("artifact_id"),
                "source_path": item.get("source_path"),
                **dict(item.get("metadata") or {}),
            },
        )
        candidate_id = self._tools.propose(
            pid,
            spec,
            source_code=str(item.get("source") or ""),
            tests=[dict(test) for test in item.get("tests", [])],
            publication_id=publication_id,
        )
        validation = self._tools.validate(candidate_id, pid=pid)
        if not validation.ok:
            raise ValidationError(
                f"image package JIT tool {name} failed validation: "
                f"{'; '.join(validation.errors)}"
            )
        return name, candidate_id

    def _remove_registered_jit_tools(
        self,
        pid: str,
        handles: dict[str, ToolHandle],
    ) -> None:
        if not handles:
            return
        process = self._processes.get_process(pid)
        if process is not None:
            self._processes.remove_process_tool_bindings(
                pid,
                {name: handle.tool_id for name, handle in handles.items()},
            )
        for handle in handles.values():
            candidate_rows = self._extensions.list_tool_candidate_rows_for_registration(
                pid,
                handle.tool_id,
            )
            self._tools.forget_loaded_jit(handle.tool_id)
            self._extensions.delete_tool(handle.tool_id)
            for row in candidate_rows:
                self._tools.discard_candidate(
                    pid,
                    str(row["candidate_id"]),
                    discarded_by="runtime",
                    reason="image_package_jit_unpublished",
                )

    def _remove_registered_jit_tool_names(
        self,
        pid: str,
        names: list[str],
    ) -> None:
        if not names:
            return
        process = self._processes.get_process(pid)
        if process is None:
            return
        handles: dict[str, ToolHandle] = {}
        for name in names:
            tool_id = process.tool_table.get(name)
            if tool_id is None or not self._tools.is_jit_tool_id(tool_id):
                continue
            handle = self._tools.loaded_tool_handle(tool_id)
            handles[name] = handle or ToolHandle(
                tool_id=tool_id,
                name=name,
                capability_id=None,
                scope="ephemeral_process",
            )
        self._remove_registered_jit_tools(pid, handles)

    def _cleanup_materialized_workspace(
        self,
        workspace_root: str | None,
        *,
        actor: str,
        reason: str,
        strict: bool = False,
    ) -> None:
        root = self._validated_materialized_root(workspace_root)
        if root is None:
            if strict and workspace_root:
                raise ValidationError("invalid materialized image workspace root")
            return
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            return
        except Exception as exc:
            envelope = public_error_envelope(
                exc,
                code="image_workspace_cleanup_failed",
            )
            self._audit.record(
                actor=actor,
                action="image.workspace.cleanup_failed",
                target=f"workspace:{workspace_root}",
                decision={
                    "reason": reason,
                    "error": envelope,
                    "internal_error": internal_exception_observation(
                        exc,
                        correlation_id=envelope["correlation_id"],
                    ),
                },
                correlation_id=envelope["correlation_id"],
            )
            if strict:
                raise
            return
        self._prune_empty_parents(root.parent)
        self._audit.record(
            actor=actor,
            action="image.workspace.cleanup",
            target=f"workspace:{workspace_root}",
            decision={"reason": reason},
        )

    def _validated_materialized_root(self, workspace_root: str | None) -> Path | None:
        if not workspace_root:
            return None
        relative = Path(workspace_root)
        materialized_root = Path(self._config.image.materialized_workspace_root)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        if relative.parts[: len(materialized_root.parts)] != materialized_root.parts:
            return None
        root = (self._workspace_root / relative).resolve()
        if self._workspace_root not in root.parents and root != self._workspace_root:
            return None
        return root

    def _materialized_workspace_root_for_cwd(self, cwd: str | None) -> str | None:
        if not cwd:
            return None
        relative = Path(cwd)
        materialized_root = Path(self._config.image.materialized_workspace_root)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        if relative.parts[: len(materialized_root.parts)] != materialized_root.parts:
            return None
        required_parts = len(materialized_root.parts) + 4
        if len(relative.parts) < required_parts:
            return None
        return Path(*relative.parts[:required_parts]).as_posix()

    def _prune_empty_parents(self, start: Path) -> None:
        boundary = (
            self._workspace_root / self._config.image.materialized_workspace_root
        ).resolve()
        current = start.resolve()
        while current != boundary and boundary in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _require_process(self, pid: str) -> Any:
        process = self._processes.get_process(pid)
        if process is None:
            from agent_libos.models.exceptions import NotFound

            raise NotFound(f"process not found: {pid}")
        return process

    def _record_publication_artifact(
        self,
        publication_id: str | None,
        artifact: dict[str, Any],
    ) -> None:
        if publication_id is None:
            return
        if not self._publications.record_runtime_publication_artifact(
            publication_id,
            artifact,
            expected_states={"planning", "applying"},
        ):
            raise ValidationError(
                f"runtime publication changed while recording artifact: {publication_id}"
            )

    @staticmethod
    def _artifact_file_bytes(record: dict[str, Any]) -> bytes:
        if record.get("kind") == "base64":
            return base64.b64decode(str(record.get("content_base64") or ""))
        return str(record.get("content") or "").encode("utf-8")

    @staticmethod
    def _artifact_path_under(path: str, root: str) -> bool:
        return path == root or path.startswith(f"{root.rstrip('/')}/")

    @staticmethod
    def _relative_artifact_path(path: str, root: str) -> Path:
        selected_root = root.rstrip("/")
        if path == selected_root:
            return Path()
        if not path.startswith(f"{selected_root}/"):
            raise RuntimeError(f"artifact path is outside root: {path}")
        return Path(path[len(selected_root) + 1 :])

    @staticmethod
    def _safe_segment(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.@+-]", "_", value)[:160] or "image"


__all__ = ["ImagePackageInstaller"]
