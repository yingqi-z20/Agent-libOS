from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.models import AuthorityRisk, Capability, CapabilityEffect, CapabilityRight, EventType, ResourceUsage
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.external_effects import (
    classify_external_effect,
    record_external_effect,
    require_external_effect_classifier,
)
from agent_libos.substrate import FilesystemProvider, LocalFilesystemProvider, ResolvedPath

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_RESOURCE_SEGMENT_SAFE = "-._~"


@dataclass(frozen=True)
class FileReadResult:
    path: str
    content: str
    bytes_read: int
    truncated: bool


@dataclass(frozen=True)
class FileBytesReadResult:
    path: str
    content: bytes
    bytes_read: int
    truncated: bool


@dataclass(frozen=True)
class FileWriteResult:
    path: str
    bytes_written: int
    created: bool


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    path: str
    kind: str
    size_bytes: int | None
    modified_at: str


@dataclass(frozen=True)
class DirectoryReadResult:
    path: str
    entries: list[DirectoryEntry]
    count: int
    truncated: bool


@dataclass(frozen=True)
class DirectoryWriteResult:
    path: str
    created: bool


@dataclass(frozen=True)
class DeleteResult:
    path: str
    kind: str
    deleted: bool
    recursive: bool = False


class FilesystemAdapter:
    """Workspace-contained filesystem primitive."""

    def __init__(
        self,
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus,
        root: str | os.PathLike[str] | None = None,
        namespace: str = _RUNTIME_DEFAULTS.workspace_namespace,
        human: Any | None = None,
        provider: FilesystemProvider | None = None,
        resources: Any | None = None,
        config: AgentLibOSConfig | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        if provider is None:
            if root is None:
                raise ValueError("FilesystemAdapter requires either root or provider")
            provider = LocalFilesystemProvider(root, namespace=namespace)
        self.provider = provider
        self.root = provider.root_display
        self.namespace = provider.namespace
        self.human = human
        self.resources = resources

    def read_text(
        self,
        pid: str,
        path: str | os.PathLike[str],
        encoding: str = _TOOL_DEFAULTS.default_text_encoding,
        max_bytes: int = _TOOL_DEFAULTS.filesystem_read_max_bytes,
        cwd: str | os.PathLike[str] | None = None,
    ) -> FileReadResult:
        max_bytes = self._bounded_positive_int(
            max_bytes,
            label="max_bytes",
            hard_limit=self.config.tools.filesystem_read_hard_limit_bytes,
        )
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.resource_for(relative)
        decision = self.capabilities.require(
            pid,
            resource,
            CapabilityRight.READ,
            self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.read_text",
                operation="read_text",
                right=CapabilityRight.READ.value,
                extra={"max_bytes": max_bytes, "encoding": encoding},
            ),
        )
        target_state = self.provider.state(target)
        if not target_state.exists:
            raise NotFound(f"file does not exist: {relative}")
        if target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        effect_context = {"path": relative, "resource": resource, "encoding": encoding, "max_bytes": max_bytes}
        read_limit = self._read_limit_for_state(target_state.size_bytes, max_bytes)
        self._preflight_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=read_limit),
            source="primitive.filesystem.read_text",
            context=effect_context,
        )
        require_external_effect_classifier(self.provider, "read_bytes")
        self._consume_one_time_decision(decision, used_by="filesystem")
        raw = self._provider_read_bytes(target, max_bytes=read_limit)
        truncated = self._is_truncated_read(target_state.size_bytes, len(raw), max_bytes)
        selected = raw[:max_bytes]
        content = self._decode_text_prefix(selected, encoding, truncated=truncated)
        event = self.events.emit(
            EventType.EXTERNAL_READ,
            source=pid,
            target=resource,
            payload={"adapter": "filesystem", "path": relative, "bytes_read": len(selected), "truncated": truncated},
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.read_text",
            target=resource,
            decision={"path": relative, "bytes_read": len(selected), "truncated": truncated},
        )
        self._record_external_effect(
            pid=pid,
            operation="read_bytes",
            target=resource,
            context=effect_context,
            result={"bytes_read": len(selected), "truncated": truncated},
            event=event,
            audit_record=audit_record,
        )
        self._charge_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=len(raw)),
            source="primitive.filesystem.read_text",
            context=effect_context,
        )
        return FileReadResult(path=relative, content=content, bytes_read=len(selected), truncated=truncated)

    def read_bytes(
        self,
        pid: str,
        path: str | os.PathLike[str],
        max_bytes: int = _TOOL_DEFAULTS.filesystem_read_max_bytes,
        cwd: str | os.PathLike[str] | None = None,
    ) -> FileBytesReadResult:
        max_bytes = self._bounded_positive_int(
            max_bytes,
            label="max_bytes",
            hard_limit=self.config.tools.filesystem_read_hard_limit_bytes,
        )
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.resource_for(relative)
        decision = self.capabilities.require(
            pid,
            resource,
            CapabilityRight.READ,
            self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.read_bytes",
                operation="read_bytes",
                right=CapabilityRight.READ.value,
                extra={"max_bytes": max_bytes},
            ),
        )
        target_state = self.provider.state(target)
        if not target_state.exists:
            raise NotFound(f"file does not exist: {relative}")
        if target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        effect_context = {"path": relative, "resource": resource, "max_bytes": max_bytes}
        read_limit = self._read_limit_for_state(target_state.size_bytes, max_bytes)
        self._preflight_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=read_limit),
            source="primitive.filesystem.read_bytes",
            context=effect_context,
        )
        require_external_effect_classifier(self.provider, "read_bytes")
        self._consume_one_time_decision(decision, used_by="filesystem")
        raw = self._provider_read_bytes(target, max_bytes=read_limit)
        truncated = self._is_truncated_read(target_state.size_bytes, len(raw), max_bytes)
        selected = raw[:max_bytes]
        event = self.events.emit(
            EventType.EXTERNAL_READ,
            source=pid,
            target=resource,
            payload={
                "adapter": "filesystem",
                "operation": "read_bytes",
                "path": relative,
                "bytes_read": len(selected),
                "truncated": truncated,
            },
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.read_bytes",
            target=resource,
            decision={"path": relative, "bytes_read": len(selected), "truncated": truncated},
        )
        self._record_external_effect(
            pid=pid,
            operation="read_bytes",
            target=resource,
            context=effect_context,
            result={"bytes_read": len(selected), "truncated": truncated},
            event=event,
            audit_record=audit_record,
        )
        self._charge_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=len(raw)),
            source="primitive.filesystem.read_bytes",
            context=effect_context,
        )
        return FileBytesReadResult(path=relative, content=selected, bytes_read=len(selected), truncated=truncated)

    def write_text(
        self,
        pid: str,
        path: str | os.PathLike[str],
        text: str,
        encoding: str = _TOOL_DEFAULTS.default_text_encoding,
        overwrite: bool = True,
        cwd: str | os.PathLike[str] | None = None,
    ) -> FileWriteResult:
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.resource_for(relative)
        self._reject_definite_permission_denial(
            pid,
            resource,
            CapabilityRight.WRITE,
            context=self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.write_text",
                operation="write_text",
                right=CapabilityRight.WRITE.value,
                extra={"encoding": encoding, "overwrite": overwrite},
            ),
        )
        target_state = self.provider.state(target)
        if target_state.exists and target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        if target_state.exists and not overwrite:
            raise FileExistsError(f"file already exists: {relative}")
        consume_capability_id = self._require_write(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            text=text,
            encoding=encoding,
            overwrite=overwrite,
        )
        bytes_to_write = len(text.encode(encoding))
        self._preflight_resource_usage(
            pid,
            ResourceUsage(external_write_bytes=bytes_to_write),
            source="primitive.filesystem.write_text",
            context={"path": relative, "resource": resource, "encoding": encoding, "overwrite": overwrite},
        )
        target_state = self.provider.state(target)
        created = not target_state.exists
        if target_state.exists and target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        if target_state.exists and not overwrite:
            raise FileExistsError(f"file already exists: {relative}")
        effect_context = {
            "path": relative,
            "resource": resource,
            "encoding": encoding,
            "overwrite": overwrite,
            "created": created,
        }
        require_external_effect_classifier(self.provider, "write_text")
        intent_record = self._record_mutation_intent(
            pid=pid,
            action="primitive.filesystem.write_text.intent",
            target=resource,
            decision={"path": relative, "bytes_to_write": bytes_to_write, "created": created},
        )
        reservation = self._reserve_mutation_capability(consume_capability_id, right="write")
        try:
            self.provider.write_text(target, text, encoding=encoding, newline="\n")
        except Exception as exc:
            self._restore_mutation_capability(reservation, right="write")
            self._record_mutation_failure(
                pid=pid,
                action="primitive.filesystem.write_text.failed",
                target=resource,
                intent_record=intent_record,
                decision={"path": relative, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        bytes_written = bytes_to_write
        event = self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=resource,
            payload={"adapter": "filesystem", "path": relative, "bytes_written": bytes_written, "created": created},
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.write_text",
            target=resource,
            decision={"path": relative, "bytes_written": bytes_written, "created": created},
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )
        self._record_external_effect(
            pid=pid,
            operation="write_text",
            target=resource,
            context=effect_context,
            result={"bytes_written": bytes_written, "created": created},
            event=event,
            audit_record=audit_record,
        )
        self._charge_resource_usage(
            pid,
            ResourceUsage(external_write_bytes=bytes_written),
            source="primitive.filesystem.write_text",
            context=effect_context,
        )
        return FileWriteResult(path=relative, bytes_written=bytes_written, created=created)

    def read_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        limit: int = _TOOL_DEFAULTS.directory_entry_limit,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DirectoryReadResult:
        limit = self._bounded_positive_int(
            limit,
            label="limit",
            hard_limit=self.config.tools.directory_entry_hard_limit,
        )
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.directory_resource_for(relative)
        decision = self.capabilities.require(
            pid,
            resource,
            CapabilityRight.READ,
            self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.read_directory",
                operation="read_directory",
                right=CapabilityRight.READ.value,
                extra={"limit": limit},
            ),
        )
        target_state = self.provider.state(target)
        if not target_state.exists:
            raise NotFound(f"directory does not exist: {relative}")
        if target_state.kind != "directory":
            raise CapabilityDenied(f"path is not a directory: {relative}")
        effect_context = {"path": relative, "resource": resource, "limit": limit}
        estimated_metadata_bytes = self._directory_metadata_preflight_bytes(limit)
        self._preflight_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=estimated_metadata_bytes),
            source="primitive.filesystem.read_directory",
            context={**effect_context, "estimated_metadata_bytes": estimated_metadata_bytes},
        )
        require_external_effect_classifier(self.provider, "list_directory")
        self._consume_one_time_decision(decision, used_by="filesystem")
        children = list(self.provider.list_directory(target, limit=limit + 1))
        selected = children[:limit]
        entries = [DirectoryEntry(**entry.__dict__) for entry in selected]
        truncated = len(children) > len(selected)
        metadata_bytes = self._directory_metadata_bytes(children)
        event = self.events.emit(
            EventType.EXTERNAL_READ,
            source=pid,
            target=resource,
            payload={
                "adapter": "filesystem",
                "operation": "read_directory",
                "path": relative,
                "count": len(entries),
                "truncated": truncated,
            },
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.read_directory",
            target=resource,
            decision={"path": relative, "count": len(entries), "truncated": truncated},
        )
        self._record_external_effect(
            pid=pid,
            operation="list_directory",
            target=resource,
            context=effect_context,
            result={"count": len(entries), "truncated": truncated},
            event=event,
            audit_record=audit_record,
        )
        self._charge_resource_usage(
            pid,
            ResourceUsage(external_read_bytes=metadata_bytes),
            source="primitive.filesystem.read_directory",
            context={**effect_context, "metadata_bytes": metadata_bytes, "listed_entries": len(children)},
        )
        return DirectoryReadResult(path=relative, entries=entries, count=len(entries), truncated=truncated)

    def write_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        parents: bool = True,
        exist_ok: bool = True,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DirectoryWriteResult:
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.directory_resource_for(relative)
        self._reject_definite_permission_denial(
            pid,
            resource,
            CapabilityRight.WRITE,
            context=self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.write_directory",
                operation="write_directory",
                right=CapabilityRight.WRITE.value,
                extra={"parents": parents, "exist_ok": exist_ok},
            ),
        )
        target_state = self.provider.state(target)
        if target_state.exists and target_state.kind != "directory":
            raise CapabilityDenied(f"path is not a directory: {relative}")
        if target_state.exists and not exist_ok:
            raise FileExistsError(f"directory already exists: {relative}")
        consume_capability_id = self._require_write_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="write_directory",
            primitive="runtime.filesystem.write_directory",
            question=f"Allow this process to create or update directory {relative}?",
            extra_context={"parents": parents, "exist_ok": exist_ok},
        )
        target_state = self.provider.state(target)
        created = not target_state.exists
        if target_state.exists and target_state.kind != "directory":
            raise CapabilityDenied(f"path is not a directory: {relative}")
        if target_state.exists and not exist_ok:
            raise FileExistsError(f"directory already exists: {relative}")
        effect_context = {
            "path": relative,
            "resource": resource,
            "parents": parents,
            "exist_ok": exist_ok,
            "created": created,
        }
        require_external_effect_classifier(self.provider, "make_directory")
        intent_record = self._record_mutation_intent(
            pid=pid,
            action="primitive.filesystem.write_directory.intent",
            target=resource,
            decision={"path": relative, "created": created, "parents": parents, "exist_ok": exist_ok},
        )
        reservation = self._reserve_mutation_capability(consume_capability_id, right="write")
        try:
            self.provider.make_directory(target, parents=parents, exist_ok=exist_ok)
        except Exception as exc:
            self._restore_mutation_capability(reservation, right="write")
            self._record_mutation_failure(
                pid=pid,
                action="primitive.filesystem.write_directory.failed",
                target=resource,
                intent_record=intent_record,
                decision={"path": relative, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        event = self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=resource,
            payload={"adapter": "filesystem", "operation": "write_directory", "path": relative, "created": created},
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.write_directory",
            target=resource,
            decision={"path": relative, "created": created, "parents": parents, "exist_ok": exist_ok},
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )
        self._record_external_effect(
            pid=pid,
            operation="make_directory",
            target=resource,
            context=effect_context,
            result={"created": created},
            event=event,
            audit_record=audit_record,
        )
        return DirectoryWriteResult(path=relative, created=created)

    def delete_file(
        self,
        pid: str,
        path: str | os.PathLike[str],
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DeleteResult:
        target, relative = self._resolve(path, cwd=cwd)
        resource = self.resource_for(relative)
        self._reject_definite_permission_denial(
            pid,
            resource,
            CapabilityRight.DELETE,
            context=self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.delete_file",
                operation="delete_file",
                right=CapabilityRight.DELETE.value,
                extra={"missing_ok": missing_ok},
            ),
        )
        target_state = self.provider.state(target)
        if not target_state.exists and not missing_ok:
            raise NotFound(f"file does not exist: {relative}")
        if target_state.exists and target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        consume_capability_id = self._require_delete(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="delete_file",
            recursive=False,
            missing_ok=missing_ok,
        )
        target_state = self.provider.state(target)
        if not target_state.exists:
            if not missing_ok:
                raise NotFound(f"file does not exist: {relative}")
            # A no-op delete still consumes the exact one-shot approval: if the
            # path appears later, this grant must not turn into a real delete.
            self._reserve_mutation_capability(consume_capability_id, right="delete")
            return DeleteResult(path=relative, kind="missing", deleted=False)
        if target_state.kind != "file":
            raise CapabilityDenied(f"path is not a file: {relative}")
        effect_context = {"path": relative, "resource": resource, "missing_ok": missing_ok}
        require_external_effect_classifier(self.provider, "delete_file")
        intent_record = self._record_mutation_intent(
            pid=pid,
            action="primitive.filesystem.delete_file.intent",
            target=resource,
            decision={"path": relative, "missing_ok": missing_ok},
        )
        reservation = self._reserve_mutation_capability(consume_capability_id, right="delete")
        try:
            self.provider.delete_file(target)
        except Exception as exc:
            self._restore_mutation_capability(reservation, right="delete")
            self._record_mutation_failure(
                pid=pid,
                action="primitive.filesystem.delete_file.failed",
                target=resource,
                intent_record=intent_record,
                decision={"path": relative, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        event = self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=resource,
            payload={"adapter": "filesystem", "operation": "delete_file", "path": relative},
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.delete_file",
            target=resource,
            decision={"path": relative, "deleted": True},
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )
        self._record_external_effect(
            pid=pid,
            operation="delete_file",
            target=resource,
            context=effect_context,
            result={"deleted": True},
            event=event,
            audit_record=audit_record,
        )
        return DeleteResult(path=relative, kind="file", deleted=True)

    def delete_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        recursive: bool = False,
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DeleteResult:
        target, relative = self._resolve(path, cwd=cwd)
        if target.is_root:
            raise CapabilityDenied("cannot delete filesystem adapter root")
        resource = self.directory_resource_for(relative)
        self._reject_definite_permission_denial(
            pid,
            resource,
            CapabilityRight.DELETE,
            context=self._authorization_context(
                pid=pid,
                resource=resource,
                relative=relative,
                primitive="runtime.filesystem.delete_directory",
                operation="delete_directory",
                right=CapabilityRight.DELETE.value,
                extra={"recursive": recursive, "missing_ok": missing_ok},
            ),
        )
        target_state = self.provider.state(target)
        if not target_state.exists and not missing_ok:
            raise NotFound(f"directory does not exist: {relative}")
        if target_state.exists and target_state.kind != "directory":
            raise CapabilityDenied(f"path is not a directory: {relative}")
        consume_capability_id = self._require_delete(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="delete_directory",
            recursive=recursive,
            missing_ok=missing_ok,
        )
        target_state = self.provider.state(target)
        if not target_state.exists:
            if not missing_ok:
                raise NotFound(f"directory does not exist: {relative}")
            # A no-op delete still consumes the exact one-shot approval: if the
            # path appears later, this grant must not turn into a real delete.
            self._reserve_mutation_capability(consume_capability_id, right="delete")
            return DeleteResult(path=relative, kind="missing", deleted=False, recursive=recursive)
        if target_state.kind != "directory":
            raise CapabilityDenied(f"path is not a directory: {relative}")
        effect_context = {
            "path": relative,
            "resource": resource,
            "recursive": recursive,
            "missing_ok": missing_ok,
        }
        require_external_effect_classifier(self.provider, "delete_directory")
        intent_record = self._record_mutation_intent(
            pid=pid,
            action="primitive.filesystem.delete_directory.intent",
            target=resource,
            decision={"path": relative, "recursive": recursive, "missing_ok": missing_ok},
        )
        reservation = self._reserve_mutation_capability(consume_capability_id, right="delete")
        try:
            self.provider.delete_directory(target, recursive=recursive)
        except Exception as exc:
            self._restore_mutation_capability(reservation, right="delete")
            self._record_mutation_failure(
                pid=pid,
                action="primitive.filesystem.delete_directory.failed",
                target=resource,
                intent_record=intent_record,
                decision={"path": relative, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        event = self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=resource,
            payload={
                "adapter": "filesystem",
                "operation": "delete_directory",
                "path": relative,
                "recursive": recursive,
            },
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.filesystem.delete_directory",
            target=resource,
            decision={"path": relative, "deleted": True, "recursive": recursive},
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )
        self._record_external_effect(
            pid=pid,
            operation="delete_directory",
            target=resource,
            context=effect_context,
            result={"deleted": True, "recursive": recursive},
            event=event,
            audit_record=audit_record,
        )
        return DeleteResult(path=relative, kind="directory", deleted=True, recursive=recursive)

    def grant_workspace(
        self,
        pid: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "filesystem",
        delegable: bool = True,
    ) -> Capability:
        return self.capabilities.grant(
            subject=pid,
            resource=self.workspace_resource(),
            rights=rights,
            issued_by=issued_by,
            delegable=delegable,
        )

    def grant_path(
        self,
        pid: str,
        path: str | os.PathLike[str],
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "filesystem",
        cwd: str | os.PathLike[str] | None = None,
        delegable: bool = True,
    ) -> Capability:
        return self.capabilities.grant(
            subject=pid,
            resource=self.resource_for_path(path, cwd=cwd),
            rights=rights,
            issued_by=issued_by,
            delegable=delegable,
        )

    def grant_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "filesystem",
        cwd: str | os.PathLike[str] | None = None,
        delegable: bool = True,
    ) -> Capability:
        return self.capabilities.grant(
            subject=pid,
            resource=self.directory_resource_for_path(path, cwd=cwd),
            rights=rights,
            issued_by=issued_by,
            delegable=delegable,
        )

    def grant_path_list(
        self,
        pid: str,
        *,
        read_files: Iterable[str | os.PathLike[str]] = (),
        write_files: Iterable[str | os.PathLike[str]] = (),
        delete_files: Iterable[str | os.PathLike[str]] = (),
        read_dirs: Iterable[str | os.PathLike[str]] = (),
        write_dirs: Iterable[str | os.PathLike[str]] = (),
        delete_dirs: Iterable[str | os.PathLike[str]] = (),
        issued_by: str = "filesystem",
        cwd: str | os.PathLike[str] | None = None,
        delegable: bool = True,
    ) -> list[Capability]:
        grants: list[Capability] = []
        for path in read_files:
            grants.append(self.grant_path(pid, path, [CapabilityRight.READ], issued_by=issued_by, cwd=cwd, delegable=delegable))
        for path in write_files:
            grants.append(self.grant_path(pid, path, [CapabilityRight.WRITE], issued_by=issued_by, cwd=cwd, delegable=delegable))
        for path in delete_files:
            grants.append(self.grant_path(pid, path, [CapabilityRight.DELETE], issued_by=issued_by, cwd=cwd, delegable=delegable))
        for path in read_dirs:
            grants.append(self.grant_directory(pid, path, [CapabilityRight.READ], issued_by=issued_by, cwd=cwd, delegable=delegable))
        for path in write_dirs:
            grants.append(self.grant_directory(pid, path, [CapabilityRight.WRITE], issued_by=issued_by, cwd=cwd, delegable=delegable))
        for path in delete_dirs:
            grants.append(self.grant_directory(pid, path, [CapabilityRight.DELETE], issued_by=issued_by, cwd=cwd, delegable=delegable))
        return grants

    def workspace_resource(self) -> str:
        return f"filesystem:{self.namespace}:*"

    def resource_for(self, path: str | os.PathLike[str]) -> str:
        relative = self._resource_path(path)
        if relative in {"", "."}:
            return f"filesystem:{self.namespace}:"
        return f"filesystem:{self.namespace}:{relative}"

    def resource_for_path(self, path: str | os.PathLike[str], cwd: str | os.PathLike[str] | None = None) -> str:
        _target, relative = self._resolve(path, cwd=cwd)
        return self.resource_for(relative)

    def directory_resource_for(self, path: str | os.PathLike[str]) -> str:
        relative = self._resource_path(path).rstrip("/")
        if relative in {"", "."}:
            return self.workspace_resource()
        return f"filesystem:{self.namespace}:{relative}/*"

    def directory_resource_for_path(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> str:
        _target, relative = self._resolve(path, cwd=cwd)
        return self.directory_resource_for(relative)

    def resolve_path(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> tuple[ResolvedPath, str]:
        return self._resolve(path, cwd=cwd)

    def _record_external_effect(
        self,
        *,
        pid: str,
        operation: str,
        target: str,
        context: dict[str, Any],
        result: Any,
        event: Any,
        audit_record: Any,
    ) -> None:
        classification = classify_external_effect(self.provider, operation, context, result)
        record_external_effect(
            self.audit.store,
            pid=pid,
            provider="filesystem",
            operation=operation,
            target=target,
            classification=classification,
            audit_record=audit_record,
            event=event,
            metadata={"context": context, "result": result},
        )

    def _preflight_resource_usage(
        self,
        pid: str,
        usage: ResourceUsage,
        *,
        source: str,
        context: dict[str, Any],
    ) -> None:
        if self.resources is None:
            return
        self.resources.preflight(pid, usage, source=source, context=context)

    def _charge_resource_usage(
        self,
        pid: str,
        usage: ResourceUsage,
        *,
        source: str,
        context: dict[str, Any],
    ) -> None:
        if self.resources is None:
            return
        self.resources.charge(
            pid,
            usage,
            source=source,
            context=context,
            allow_overage=True,
            kill_on_exceed=True,
        )

    def _read_limit_for_state(self, size_bytes: int | None, max_bytes: int) -> int:
        if size_bytes is None:
            # Unknown-size providers must read one extra byte to detect
            # truncation. Known-size providers can avoid that extra I/O.
            return max_bytes + 1
        return min(max(0, size_bytes), max_bytes)

    def _is_truncated_read(self, size_bytes: int | None, bytes_read: int, max_bytes: int) -> bool:
        if size_bytes is not None:
            return size_bytes > max_bytes
        return bytes_read > max_bytes

    def _provider_read_bytes(self, target: ResolvedPath, *, max_bytes: int) -> bytes:
        try:
            return self.provider.read_bytes(target, max_bytes=max_bytes)
        except TypeError as exc:
            raise ValidationError("filesystem provider must support max_bytes-limited reads") from exc

    def _directory_metadata_bytes(self, entries: Iterable[Any]) -> int:
        payload = [getattr(entry, "__dict__", {"entry": str(entry)}) for entry in entries]
        return len(json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8"))

    def _directory_metadata_preflight_bytes(self, limit: int) -> int:
        # Directory entry names and timestamps are only known after reading the
        # directory. Reserve a conservative per-entry envelope first so tight
        # information-flow budgets fail closed before metadata is observed.
        return max(1, (limit + 1) * 512)

    def _consume_one_time_decision(self, decision: Any, *, used_by: str) -> None:
        self.capabilities.claim_decision_use(
            decision,
            used_by=used_by,
            reason="one-time filesystem permission consumed",
        )

    def _reserve_mutation_capability(self, capability_id: str | None, *, right: str) -> str | None:
        if capability_id is None:
            return None
        # Filesystem mutations cross an external provider boundary, so one-shot
        # grants are reserved before the side effect and refunded only if the
        # provider raises. This closes the concurrent authorize-then-write race
        # without consuming approval for ordinary provider failures.
        self.capabilities.consume_use(
            capability_id,
            used_by="filesystem",
            reason=f"one-time filesystem {right} permission reserved",
        )
        return capability_id

    def _restore_mutation_capability(self, capability_id: str | None, *, right: str) -> None:
        if capability_id is None:
            return
        self.capabilities._restore_reserved_use(
            capability_id,
            restored_by="filesystem",
            reason=f"one-time filesystem {right} permission restored after provider failure",
        )

    def _record_mutation_intent(
        self,
        *,
        pid: str,
        action: str,
        target: str,
        decision: dict[str, Any],
    ) -> Any:
        return self.audit.record(
            actor=pid,
            action=action,
            target=target,
            decision=decision,
        )

    def _record_mutation_failure(
        self,
        *,
        pid: str,
        action: str,
        target: str,
        intent_record: Any,
        decision: dict[str, Any],
    ) -> None:
        self.audit.record(
            actor=pid,
            action=action,
            target=target,
            decision=decision,
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )

    def _resolve(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> tuple[ResolvedPath, str]:
        target = self.provider.resolve(self._path_with_cwd(path, cwd))
        return target, target.relative

    def _logical_path(self, path: str | os.PathLike[str]) -> str:
        return os.fspath(path)

    def _resource_path(self, path: str | os.PathLike[str]) -> str:
        logical = self._logical_path(path)
        if logical in {"", "."}:
            return logical
        return "/".join(quote(part, safe=_RESOURCE_SEGMENT_SAFE) for part in logical.split("/"))

    def _path_with_cwd(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None,
    ) -> str:
        raw = os.fspath(path)
        if os.path.isabs(raw) or cwd is None or os.fspath(cwd) in {"", "."}:
            return raw
        cwd_path = self._logical_path(cwd).strip("/")
        if cwd_path in {"", "."}:
            return raw
        return f"{cwd_path}/{raw}"

    def _require_write(
        self,
        pid: str,
        resource: str,
        target: ResolvedPath,
        relative: str,
        text: str,
        encoding: str,
        overwrite: bool,
    ) -> str | None:
        return self._require_write_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="write_text",
            primitive="runtime.filesystem.write_text",
            question=f"Allow this process to write {relative}?",
            extra_context={
                "encoding": encoding,
                "overwrite": overwrite,
                **self._content_context(text, encoding),
            },
        )

    def _reject_definite_permission_denial(
        self,
        pid: str,
        resource: str,
        right: CapabilityRight,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        # Do not stat the target before a definite deny/miss; existence and
        # kind are filesystem facts that require some matching policy first.
        policy = self.capabilities.permission_policy(pid, resource, right, context)
        if policy in {CapabilityManager.MISSING, CapabilityManager.ALWAYS_DENY}:
            self.capabilities.require(pid, resource, right, context)

    def _require_write_operation(
        self,
        pid: str,
        resource: str,
        target: ResolvedPath,
        relative: str,
        operation: str,
        primitive: str,
        question: str,
        extra_context: dict[str, Any] | None = None,
    ) -> str | None:
        operation_context = self._operation_context(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive=primitive,
            operation=operation,
            right=CapabilityRight.WRITE.value,
            extra=extra_context or {},
        )
        decision = self.capabilities.authorize(pid, resource, CapabilityRight.WRITE, operation_context)
        if decision.allowed:
            return decision.consume_capability_id
        if decision.policy == CapabilityManager.ALWAYS_DENY:
            raise CapabilityDenied(f"{pid} denied write on {resource}")
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for write on {resource}")
            # This primitive has the concrete path, overwrite state, byte count,
            # and preview needed for a safe per-use human decision.
            request_id = self.human.query(
                pid=pid,
                    human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": question,
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [CapabilityRight.WRITE.value],
                        "constraints": self._approval_constraints(operation_context, right=CapabilityRight.WRITE.value),
                    },
                    "context": {
                        **operation_context,
                    },
                },
                blocking=True,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to write {resource}",
            )
        raise CapabilityDenied(f"{pid} lacks write on {resource}")

    def _require_delete(
        self,
        pid: str,
        resource: str,
        target: ResolvedPath,
        relative: str,
        operation: str,
        recursive: bool,
        missing_ok: bool,
    ) -> str | None:
        operation_context = self._operation_context(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive=f"runtime.filesystem.{operation}",
            operation=operation,
            right=CapabilityRight.DELETE.value,
            extra={"recursive": recursive, "missing_ok": missing_ok},
        )
        decision = self.capabilities.authorize(pid, resource, CapabilityRight.DELETE, operation_context)
        if decision.allowed:
            return decision.consume_capability_id
        if decision.policy == CapabilityManager.ALWAYS_DENY:
            raise CapabilityDenied(f"{pid} denied delete on {resource}")
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for delete on {resource}")
            request_id = self.human.query(
                pid=pid,
                    human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": f"Allow this process to delete {relative}?",
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [CapabilityRight.DELETE.value],
                        "constraints": self._approval_constraints(operation_context, right=CapabilityRight.DELETE.value),
                    },
                    "context": operation_context,
                },
                blocking=True,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to delete {resource}",
            )
        raise CapabilityDenied(f"{pid} lacks delete on {resource}")

    def _operation_context(
        self,
        pid: str,
        resource: str,
        target: ResolvedPath,
        relative: str,
        primitive: str,
        operation: str,
        right: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        target_state = self._target_state(target)
        will_overwrite = bool(target_state["exists"] and target_state["kind"] == "file")
        profile = self.capabilities.profiles.filesystem(
            resource=resource,
            right=right,
            effect=CapabilityEffect.ASK,
            risk=self._risk_for_filesystem_right(right),
            path=relative,
        )
        return {
            "adapter": "filesystem",
            "primitive": primitive,
            "operation": operation,
            "authority_operation": f"filesystem.{right}",
            "pid": pid,
            "workspace_root": self.root,
            "path": relative,
            "absolute_path": target.display,
            "resource": resource,
            "right": right,
            "sandbox_profile": self._profile_json(profile),
            "grant_scope": "one_time",
            "will_create": not target_state["exists"],
            "will_overwrite": will_overwrite,
            "target": target_state,
            **extra,
        }

    def _authorization_context(
        self,
        *,
        pid: str,
        resource: str,
        relative: str,
        primitive: str,
        operation: str,
        right: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.capabilities.profiles.filesystem(
            resource=resource,
            right=right,
            effect=CapabilityEffect.ALLOW,
            risk=self._risk_for_filesystem_right(right),
            path=relative,
        )
        return {
            "adapter": "filesystem",
            "primitive": primitive,
            "operation": operation,
            "authority_operation": f"filesystem.{right}",
            "pid": pid,
            "workspace_root": self.root,
            "path": relative,
            "resource": resource,
            "right": right,
            "sandbox_profile": self._profile_json(profile),
            **(extra or {}),
        }

    def _approval_constraints(self, context: dict[str, Any], *, right: str) -> dict[str, Any]:
        condition_keys = [
            "path",
            "content_sha256",
            "overwrite",
            "parents",
            "exist_ok",
            "recursive",
            "missing_ok",
        ]
        conditions = {key: context[key] for key in condition_keys if key in context}
        return {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": f"filesystem.approval.{right}",
                    "operation": f"filesystem.{right}",
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": self._risk_for_filesystem_right(right).value,
                    "conditions": conditions,
                    "description": "one-shot human approval for exact filesystem operation",
                }
            ]
        }

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return {
            "operation": profile.operation,
            "resource": profile.resource,
            "effect": profile.effect.value,
            "risk": profile.risk.value,
            "rule_id": profile.rule_id,
            "restrictions": profile.restrictions,
        }

    def _risk_for_filesystem_right(self, right: str) -> AuthorityRisk:
        if right == CapabilityRight.DELETE.value:
            return AuthorityRisk.DESTRUCTIVE
        if right == CapabilityRight.WRITE.value:
            return AuthorityRisk.HIGH
        return AuthorityRisk.LOW

    def _content_context(self, text: str, encoding: str) -> dict[str, Any]:
        encoded = text.encode(encoding)
        preview, preview_truncated = self._preview_text(text)
        return {
            "content_bytes": len(encoded),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "content_preview": preview,
            "content_preview_chars": len(preview),
            "content_preview_truncated": preview_truncated,
        }

    def _preview_text(self, text: str, limit: int | None = None) -> tuple[str, bool]:
        selected_limit = self.config.tools.approval_preview_chars if limit is None else limit
        preview = text[:selected_limit]
        # repr() prevents newlines or prompt-like text from masquerading as
        # separate approval instructions in the human terminal prompt.
        return repr(preview), len(text) > selected_limit

    def _decode_text_prefix(self, data: bytes, encoding: str, *, truncated: bool) -> str:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            if truncated and exc.end == len(data):
                return data[: exc.start].decode(encoding)
            raise

    def _target_state(self, target: ResolvedPath) -> dict[str, Any]:
        state = self.provider.state(target)
        if not state.exists:
            return {"exists": False, "kind": "missing"}
        result = {
            "exists": True,
            "kind": state.kind,
            "size_bytes": state.size_bytes,
            "modified_at": state.modified_at,
        }
        return result

    def _bounded_positive_int(self, value: int, *, label: str, hard_limit: int) -> int:
        if isinstance(value, bool):
            raise ValidationError(f"{label} must be an integer")
        try:
            selected = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label} must be an integer") from exc
        if selected < 1:
            raise ValidationError(f"{label} must be >= 1")
        if selected > hard_limit:
            raise ValidationError(f"{label} exceeds hard limit {hard_limit}")
        return selected
