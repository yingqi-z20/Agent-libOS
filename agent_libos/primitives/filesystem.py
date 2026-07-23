from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from urllib.parse import quote

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.models import (
    AuthorityRisk,
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ResourceUsage,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.substrate import (
    FilesystemProvider,
    HierarchicalPathLock,
    LocalFilesystemProvider,
    ProviderEffectNotStarted,
    ResolvedPath,
)
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourceSettlement,
)

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_RESOURCE_SEGMENT_SAFE = "-._~"
_DIRECTORY_STATE_OBSERVATION_BYTES = 512

_FileLabelPathLocks = HierarchicalPathLock


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
        audit: AuditPort,
        events: EventPort,
        *,
        protected_operations: ProtectedOperationSDK,
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
        self.protected_operations = protected_operations
        if provider is None:
            if root is None:
                raise ValueError("FilesystemAdapter requires either root or provider")
            provider = LocalFilesystemProvider(root, namespace=namespace)
        self.provider = provider
        self.root = provider.root_display
        self.namespace = provider.namespace
        self.human = human
        self.resources = resources
        self._file_label_io_lock = _FileLabelPathLocks()

    @contextmanager
    def hold_file_label_io_paths(self, paths: Iterable[str]) -> Iterator[None]:
        """Serialize provider writes with file-label binding updates.

        Typed repository primitives use this narrow coordination surface so
        file bytes and their lineage binding cannot be observed half-updated.
        It intentionally exposes no underlying lock implementation.
        """

        paths_by_order_key: dict[tuple[str, ...], str] = {}
        for path in paths:
            display_path = str(path)
            order_key = self._file_label_io_lock.order_key(display_path)
            paths_by_order_key.setdefault(order_key, display_path)
        with ExitStack() as stack:
            for order_key in sorted(paths_by_order_key):
                stack.enter_context(
                    self._file_label_io_lock.hold(paths_by_order_key[order_key])
                )
            yield

    def validate_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> str:
        """Authorize and validate one directory-state observation.

        Working-directory selection needs host filesystem metadata, so it must
        cross the same capability, finite-use, resource, audit/event, and
        external-effect boundary as other filesystem reads.  The returned path
        is the normalized lexical workspace-relative path; provider state/sink
        checks resolve real paths only after authorization.
        """

        target, relative = self._resolve(path, cwd=cwd)
        resource = self.directory_resource_for(relative)
        decision, authority_context = self._require_read_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive="runtime.filesystem.validate_directory",
            operation="state",
            question=f"Allow this process to inspect directory {relative or '.'}?",
        )
        effect_context = {
            "path": relative,
            "resource": resource,
            "expected_kind": "directory",
        }
        usage = ResourceUsage(external_read_bytes=_DIRECTORY_STATE_OBSERVATION_BYTES)
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=authority_context,
            observation=effect_context,
            preflight_usage=usage,
            resource_source="primitive.filesystem.validate_directory",
            resource_context=effect_context,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid, resource, "primitive.filesystem.validate_directory.failed", effect_context, error, phase
            ),
        )
        with self._protected().start(
            "primitive.filesystem.validate_directory", invocation, provider=self.provider
        ) as protected:
            state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            error: Exception | None = None
            if not state.exists:
                outcome = "not_found"
                error = NotFound(f"working directory does not exist: {relative}")
            elif state.kind != "directory":
                outcome = "not_directory"
                error = NotFound(f"working directory is not a directory: {relative}")
            else:
                outcome = "validated"
            result_payload = {"outcome": outcome, "state_kind": state.kind}
            protected.complete(
                state,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_READ,
                    "primitive.filesystem.validate_directory",
                    {"adapter": "filesystem", "operation": "state", "path": relative, **result_payload},
                    {"path": relative, "state_kind": state.kind, **result_payload},
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=ResourceSettlement(
                    usage=usage,
                    source="primitive.filesystem.validate_directory",
                    context=effect_context,
                ),
            )
            if error is not None:
                raise error
            return relative or "."

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
        decision, authority_context = self._require_read_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive="runtime.filesystem.read_text",
            operation="read_text",
            question=f"Allow this process to read {relative}?",
            extra_context={"max_bytes": max_bytes, "encoding": encoding},
        )
        effect_context = {"path": relative, "resource": resource, "encoding": encoding, "max_bytes": max_bytes}
        with ExitStack() as stack:
            stack.enter_context(self._file_label_io_lock.hold(relative))
            read_context, read_binding_state = self._data_flow().file_snapshot(relative)
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target=resource,
                decisions=(decision,),
                canonical_args=authority_context,
                observation=effect_context,
                preflight_usage=ResourceUsage(external_read_bytes=max_bytes),
                resource_source="primitive.filesystem.read_text",
                resource_context=effect_context,
                data_flow_ingress_context=read_context,
                failure_evidence=lambda error, phase: self._protected_failure_evidence(
                    pid, resource, "primitive.filesystem.read_text.failed", effect_context, error, phase
                ),
            )
            protected = stack.enter_context(self._protected().start(
                "primitive.filesystem.read_text", invocation, provider=self.provider
            ))
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            if not target_state.exists:
                error = NotFound(f"file does not exist: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_text.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_text",
                )
                raise error
            if target_state.kind != "file":
                error = CapabilityDenied(f"path is not a file: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_text.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_text",
                )
                raise error
            read_limit = self._read_limit_for_state(target_state.size_bytes, max_bytes)
            raw = protected.call(
                ProviderPhase("read", information_flow=True),
                self._provider_read_bytes,
                target,
                max_bytes=read_limit,
            )
            current_binding_state = self._data_flow().file_state_version(relative)
            if current_binding_state != read_binding_state:
                self._data_flow().observe_ingress(
                    DataFlowContext.aggregate(
                        (read_context, self._data_flow().file_context(relative))
                    )
                )
                raise CapabilityDenied("filesystem label binding changed during read")
            truncated = self._is_truncated_read(target_state.size_bytes, len(raw), max_bytes)
            selected = raw[:max_bytes]
            content = self._decode_text_prefix(selected, encoding, truncated=truncated)
            result = FileReadResult(
                path=relative, content=content, bytes_read=len(selected), truncated=truncated
            )
            result_payload = {"bytes_read": len(selected), "truncated": truncated}
            completed = protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_READ,
                    "primitive.filesystem.read_text",
                    {"adapter": "filesystem", "path": relative, **result_payload},
                    {"path": relative, **result_payload},
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=ResourceSettlement(
                    usage=ResourceUsage(external_read_bytes=len(selected)),
                    source="primitive.filesystem.read_text",
                    context=effect_context,
                ),
            )
            return completed

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
        decision, authority_context = self._require_read_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive="runtime.filesystem.read_bytes",
            operation="read_bytes",
            question=f"Allow this process to read {relative}?",
            extra_context={"max_bytes": max_bytes},
        )
        effect_context = {"path": relative, "resource": resource, "max_bytes": max_bytes}
        with ExitStack() as stack:
            stack.enter_context(self._file_label_io_lock.hold(relative))
            read_context, read_binding_state = self._data_flow().file_snapshot(relative)
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target=resource,
                decisions=(decision,),
                canonical_args=authority_context,
                observation=effect_context,
                preflight_usage=ResourceUsage(external_read_bytes=max_bytes),
                resource_source="primitive.filesystem.read_bytes",
                resource_context=effect_context,
                data_flow_ingress_context=read_context,
                failure_evidence=lambda error, phase: self._protected_failure_evidence(
                    pid, resource, "primitive.filesystem.read_bytes.failed", effect_context, error, phase
                ),
            )
            protected = stack.enter_context(self._protected().start(
                "primitive.filesystem.read_bytes", invocation, provider=self.provider
            ))
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            if not target_state.exists:
                error = NotFound(f"file does not exist: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_bytes.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_bytes",
                )
                raise error
            if target_state.kind != "file":
                error = CapabilityDenied(f"path is not a file: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_bytes.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_bytes",
                )
                raise error
            raw = protected.call(
                ProviderPhase("read", information_flow=True),
                self._provider_read_bytes,
                target,
                max_bytes=self._read_limit_for_state(target_state.size_bytes, max_bytes),
            )
            current_binding_state = self._data_flow().file_state_version(relative)
            if current_binding_state != read_binding_state:
                self._data_flow().observe_ingress(
                    DataFlowContext.aggregate(
                        (read_context, self._data_flow().file_context(relative))
                    )
                )
                raise CapabilityDenied("filesystem label binding changed during read")
            truncated = self._is_truncated_read(target_state.size_bytes, len(raw), max_bytes)
            selected = raw[:max_bytes]
            result = FileBytesReadResult(
                path=relative, content=selected, bytes_read=len(selected), truncated=truncated
            )
            result_payload = {"bytes_read": len(selected), "truncated": truncated}
            completed = protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_READ,
                    "primitive.filesystem.read_bytes",
                    {"adapter": "filesystem", "operation": "read_bytes", "path": relative, **result_payload},
                    {"path": relative, **result_payload},
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=ResourceSettlement(
                    usage=ResourceUsage(external_read_bytes=len(selected)),
                    source="primitive.filesystem.read_bytes",
                    context=effect_context,
                ),
            )
            return completed

    def write_text(
        self,
        pid: str,
        path: str | os.PathLike[str],
        text: str,
        encoding: str = _TOOL_DEFAULTS.default_text_encoding,
        overwrite: bool = True,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
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
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        sink = DataSink(resource)
        data_flow_payload = {
            "path": relative,
            "text": text,
            "encoding": encoding,
            "overwrite": overwrite,
        }
        target_label_generation = self._data_flow().store.get_file_label_binding_generation(
            relative
        )
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=data_flow_payload,
            operation="filesystem.write_text",
            target_state_version=target_label_generation,
        )
        decision, authority_context = self._require_write(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            text=text,
            encoding=encoding,
            overwrite=overwrite,
            source_oids=source_oids,
        )
        bytes_to_write = len(text.encode(encoding))
        effect_context = {
            "path": relative,
            "resource": resource,
            "encoding": encoding,
            "overwrite": overwrite,
            "created": None,
        }
        intent: dict[str, Any] = {}
        mutation_attempted = False
        missing_parent_paths: list[str] = []

        def prepare() -> None:
            intent["record"] = self._record_mutation_intent(
                pid=pid,
                action="primitive.filesystem.write_text.intent",
                target=resource,
                decision={"path": relative, "bytes_to_write": bytes_to_write},
            )

        def write_provider() -> None:
            nonlocal mutation_attempted
            mutation_attempted = True
            self.provider.write_text(
                target,
                text,
                encoding=encoding,
                newline="\n",
                overwrite=overwrite,
            )

        def settle_ambiguous_write(error: BaseException, _phase: str) -> None:
            if not mutation_attempted or isinstance(error, ProviderEffectNotStarted):
                return
            # A generic provider or post-provider settlement failure cannot
            # prove whether bytes reached the workspace.  Persist the intended
            # labels conservatively so a later read cannot wash the source.
            self._bind_written_path_set(
                pid=pid,
                context=flow_context,
                parent_paths=missing_parent_paths,
                final_path=relative,
                final_content=text.encode(encoding),
            )

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=authority_context,
            observation=effect_context,
            preflight_usage=ResourceUsage(external_write_bytes=bytes_to_write),
            resource_source="primitive.filesystem.write_text",
            resource_context=effect_context,
            prepare=prepare,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid,
                resource,
                "primitive.filesystem.write_text.failed",
                effect_context,
                error,
                phase,
                intent.get("record"),
            ),
            failure_settlement=settle_ambiguous_write,
            data_sink=sink,
            data_flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            data_flow_operation="filesystem.write_text",
            data_flow_target_state_version=target_label_generation,
            data_flow_target_state_version_resolver=lambda: (
                self._data_flow().store.get_file_label_binding_generation(relative)
            ),
        )
        with (
            self._file_label_io_lock.hold(
                HierarchicalPathLock.creation_scope(relative)
            ),
            self._protected().start(
                "primitive.filesystem.write_text", invocation, provider=self.provider
            ) as protected,
        ):
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            created = not target_state.exists
            effect_context.update({"created": created, "state_observed": True})
            if created:
                missing_parent_paths.extend(
                    protected.call(
                        ProviderPhase("parent_state", information_flow=True),
                        self._missing_parent_paths,
                        relative,
                    )
                )
            if target_state.exists and target_state.kind != "file":
                error = CapabilityDenied(f"path is not a file: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.write_text.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                    resource_source="primitive.filesystem.write_text",
                )
                raise error
            if target_state.exists and not overwrite:
                error = FileExistsError(f"file already exists: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.write_text.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                    resource_source="primitive.filesystem.write_text",
                )
                raise error
            protected.call(
                ProviderPhase("write", state_mutation=True, information_flow=True),
                write_provider,
            )
            result = FileWriteResult(path=relative, bytes_written=bytes_to_write, created=created)
            result_payload = {"bytes_written": bytes_to_write, "created": created}
            completed = protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_WRITE,
                    "primitive.filesystem.write_text",
                    {"adapter": "filesystem", "path": relative, **result_payload},
                    {"path": relative, **result_payload},
                    result_payload,
                    intent.get("record"),
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                settle_success=lambda: self._bind_written_path_set(
                    pid=pid,
                    context=flow_context,
                    parent_paths=missing_parent_paths,
                    final_path=relative,
                    final_content=text.encode(encoding),
                ),
                resource=ResourceSettlement(
                    usage=ResourceUsage(external_write_bytes=bytes_to_write),
                    source="primitive.filesystem.write_text",
                    context=effect_context,
                ),
            )
            self._data_flow().observe_ingress(self._data_flow().file_context(relative))
            return completed

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
        decision, authority_context = self._require_read_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            primitive="runtime.filesystem.read_directory",
            operation="read_directory",
            question=f"Allow this process to list directory {relative or '.'}?",
            extra_context={"limit": limit},
        )
        effect_context = {"path": relative, "resource": resource, "limit": limit}
        estimated_metadata_bytes = self._directory_metadata_preflight_bytes(limit)
        with ExitStack() as stack:
            stack.enter_context(self._file_label_io_lock.hold(relative))
            label_snapshot, label_state_version = (
                self._data_flow().directory_label_snapshot(relative)
            )
            external_context = self._data_flow().external_file_context()
            directory_base_context = label_snapshot.get(relative, external_context)
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target=resource,
                decisions=(decision,),
                canonical_args=authority_context,
                observation=effect_context,
                preflight_usage=ResourceUsage(external_read_bytes=estimated_metadata_bytes),
                resource_source="primitive.filesystem.read_directory",
                resource_context={**effect_context, "estimated_metadata_bytes": estimated_metadata_bytes},
                data_flow_ingress_context=directory_base_context,
                failure_evidence=lambda error, phase: self._protected_failure_evidence(
                    pid, resource, "primitive.filesystem.read_directory.failed", effect_context, error, phase
                ),
            )
            protected = stack.enter_context(self._protected().start(
                "primitive.filesystem.read_directory", invocation, provider=self.provider
            ))
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            if not target_state.exists:
                error = NotFound(f"directory does not exist: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_directory.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_directory",
                )
                raise error
            if target_state.kind != "directory":
                error = CapabilityDenied(f"path is not a directory: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.read_directory.rejected",
                    context=effect_context,
                    error=error,
                    resource_source="primitive.filesystem.read_directory",
                )
                raise error
            children = protected.call(
                ProviderPhase("list", information_flow=True),
                lambda: list(self.provider.list_directory(target, limit=limit + 1)),
            )
            directory_context = DataFlowContext.aggregate(
                [
                    directory_base_context,
                    *(
                        label_snapshot.get(child.path, external_context)
                        for child in children
                    ),
                ]
            )
            if (
                self._data_flow().directory_label_state_version(relative)
                != label_state_version
            ):
                self._data_flow().observe_ingress(
                    DataFlowContext.aggregate(
                        (
                            directory_context,
                            self._data_flow().file_context(relative),
                        )
                    )
                )
                raise CapabilityDenied(
                    "filesystem directory-child label bindings changed during listing"
                )
            selected = children[:limit]
            entries = [DirectoryEntry(**entry.__dict__) for entry in selected]
            truncated = len(children) > len(selected)
            metadata_bytes = self._directory_metadata_bytes(children)
            result = DirectoryReadResult(
                path=relative, entries=entries, count=len(entries), truncated=truncated
            )
            result_payload = {"count": len(entries), "truncated": truncated}
            completed = protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_READ,
                    "primitive.filesystem.read_directory",
                    {"adapter": "filesystem", "operation": "read_directory", "path": relative, **result_payload},
                    {"path": relative, **result_payload},
                    result_payload,
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                resource=ResourceSettlement(
                    usage=ResourceUsage(external_read_bytes=metadata_bytes),
                    source="primitive.filesystem.read_directory",
                    context={**effect_context, "metadata_bytes": metadata_bytes, "listed_entries": len(children)},
                ),
            )
            self._data_flow().observe_ingress(directory_context)
            return completed

    def write_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        parents: bool = True,
        exist_ok: bool = True,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
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
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        sink = DataSink(self.resource_for(relative))
        data_flow_payload = {"path": relative, "parents": parents, "exist_ok": exist_ok}
        target_label_generation = self._data_flow().store.get_file_label_binding_generation(
            relative
        )
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=data_flow_payload,
            operation="filesystem.write_directory",
            target_state_version=target_label_generation,
        )
        decision, authority_context = self._require_write_operation(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="write_directory",
            primitive="runtime.filesystem.write_directory",
            question=f"Allow this process to create or update directory {relative}?",
            extra_context={"parents": parents, "exist_ok": exist_ok},
            source_oids=source_oids,
        )
        effect_context = {
            "path": relative,
            "resource": resource,
            "parents": parents,
            "exist_ok": exist_ok,
            "created": None,
        }
        intent: dict[str, Any] = {}
        mutation_attempted = False
        missing_parent_paths: list[str] = []

        def prepare() -> None:
            intent["record"] = self._record_mutation_intent(
                pid=pid,
                action="primitive.filesystem.write_directory.intent",
                target=resource,
                decision={"path": relative, "parents": parents, "exist_ok": exist_ok},
            )

        def make_directory_provider() -> None:
            nonlocal mutation_attempted
            mutation_attempted = True
            self.provider.make_directory(
                target,
                parents=parents,
                exist_ok=exist_ok,
            )

        def settle_ambiguous_directory(error: BaseException, _phase: str) -> None:
            if not mutation_attempted or isinstance(error, ProviderEffectNotStarted):
                return
            self._bind_written_path_set(
                pid=pid,
                context=flow_context,
                parent_paths=missing_parent_paths,
                final_path=relative,
                final_content=b"<agent-libos-directory>",
            )

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=authority_context,
            observation=effect_context,
            data_sink=sink,
            data_flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            data_flow_operation="filesystem.write_directory",
            data_flow_target_state_version=target_label_generation,
            data_flow_target_state_version_resolver=lambda: (
                self._data_flow().store.get_file_label_binding_generation(relative)
            ),
            prepare=prepare,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid,
                resource,
                "primitive.filesystem.write_directory.failed",
                effect_context,
                error,
                phase,
                intent.get("record"),
            ),
            failure_settlement=settle_ambiguous_directory,
        )
        with (
            self._file_label_io_lock.hold(
                HierarchicalPathLock.creation_scope(relative)
            ),
            self._protected().start(
                "primitive.filesystem.write_directory", invocation, provider=self.provider
            ) as protected,
        ):
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            created = not target_state.exists
            effect_context.update({"created": created, "state_observed": True})
            if created and parents:
                missing_parent_paths.extend(
                    protected.call(
                        ProviderPhase("parent_state", information_flow=True),
                        self._missing_parent_paths,
                        relative,
                    )
                )
            if target_state.exists and target_state.kind != "directory":
                error = CapabilityDenied(f"path is not a directory: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.write_directory.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                )
                raise error
            if target_state.exists and not exist_ok:
                error = FileExistsError(f"directory already exists: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.write_directory.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                )
                raise error
            protected.call(
                ProviderPhase("make_directory", state_mutation=True, information_flow=True),
                make_directory_provider,
            )
            result = DirectoryWriteResult(path=relative, created=created)
            result_payload = {"created": created}
            return protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid,
                    resource,
                    EventType.EXTERNAL_WRITE,
                    "primitive.filesystem.write_directory",
                    {"adapter": "filesystem", "operation": "write_directory", "path": relative, **result_payload},
                    {"path": relative, "parents": parents, "exist_ok": exist_ok, **result_payload},
                    result_payload,
                    intent.get("record"),
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                settle_success=lambda: self._bind_written_path_set(
                    pid=pid,
                    context=flow_context,
                    parent_paths=missing_parent_paths,
                    final_path=relative,
                    final_content=b"<agent-libos-directory>",
                ),
            )

    def delete_file(
        self,
        pid: str,
        path: str | os.PathLike[str],
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> DeleteResult:
        _target, relative = self._resolve(path, cwd=cwd)
        with self._file_label_io_lock.hold(relative):
            return self._delete_file_serialized(
                pid,
                path,
                missing_ok,
                cwd,
                source_oids=source_oids,
            )

    def _delete_file_serialized(
        self,
        pid: str,
        path: str | os.PathLike[str],
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
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
        (
            target_context,
            target_label_state_version,
            target_binding_snapshot,
        ) = self._data_flow().file_deletion_snapshot(relative)
        flow_context = DataFlowContext.aggregate(
            (
                self._data_flow().context_from_source_oids(pid, source_oids),
                target_context,
            )
        )
        data_flow_payload = {"path": relative, "missing_ok": missing_ok}
        self._data_flow().authorize_egress(
            pid=pid,
            sink=DataSink(resource),
            context=flow_context,
            payload=data_flow_payload,
            operation="filesystem.delete_file",
            target_state_version=target_label_state_version,
        )
        decision, authority_context = self._require_delete(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="delete_file",
            recursive=False,
            missing_ok=missing_ok,
            source_oids=source_oids,
        )
        effect_context = {"path": relative, "resource": resource, "missing_ok": missing_ok}
        intent: dict[str, Any] = {}

        def prepare() -> None:
            intent["record"] = self._record_mutation_intent(
                pid=pid,
                action="primitive.filesystem.delete_file.intent",
                target=resource,
                decision={"path": relative, "missing_ok": missing_ok},
            )

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=authority_context,
            observation=effect_context,
            data_sink=DataSink(resource),
            data_flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            data_flow_operation="filesystem.delete_file",
            data_flow_target_state_version=target_label_state_version,
            data_flow_target_state_version_resolver=lambda: (
                self._data_flow().file_state_version(relative)
            ),
            prepare=prepare,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid, resource, "primitive.filesystem.delete_file.failed", effect_context, error, phase, intent.get("record")
            ),
        )
        with self._protected().start(
            "primitive.filesystem.delete_file", invocation, provider=self.provider
        ) as protected:
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            effect_context["state_observed"] = True
            if not target_state.exists:
                if not missing_ok:
                    error = NotFound(f"file does not exist: {relative}")
                    self._complete_state_rejection(
                        protected,
                        pid=pid,
                        target=resource,
                        audit_action="primitive.filesystem.delete_file.rejected",
                        context=effect_context,
                        error=error,
                        intent_record=intent.get("record"),
                    )
                    raise error
                result = DeleteResult(path=relative, kind="missing", deleted=False)
                result_payload = {"path": relative, "deleted": False, "missing_ok": True}
                return protected.complete(
                    result,
                    self._protected_filesystem_evidence(
                        pid, resource, EventType.EXTERNAL_WRITE, "primitive.filesystem.delete_file",
                        {"adapter": "filesystem", "operation": "delete_file", **result_payload},
                        result_payload, result_payload, intent.get("record"),
                    ),
                    classification_override=self._state_only_classification(),
                )
            if target_state.kind != "file":
                error = CapabilityDenied(f"path is not a file: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.delete_file.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                )
                raise error
            protected.call(
                ProviderPhase("delete", state_mutation=True, information_flow=True),
                self.provider.delete_file,
                target,
            )
            result = DeleteResult(path=relative, kind="file", deleted=True)
            result_payload = {"path": relative, "deleted": True}
            return protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid, resource, EventType.EXTERNAL_WRITE, "primitive.filesystem.delete_file",
                    {"adapter": "filesystem", "operation": "delete_file", "path": relative},
                    result_payload, {"deleted": True}, intent.get("record"),
                ),
                classification_context=effect_context,
                classification_result={"deleted": True},
                settle_success=lambda: self._data_flow().tombstone_path_tree(
                    pid=pid,
                    expected_bindings=target_binding_snapshot,
                ),
            )

    def delete_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        recursive: bool = False,
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
    ) -> DeleteResult:
        _target, relative = self._resolve(path, cwd=cwd)
        with self._file_label_io_lock.hold(relative):
            return self._delete_directory_serialized(
                pid,
                path,
                recursive,
                missing_ok,
                cwd,
                source_oids=source_oids,
            )

    def _delete_directory_serialized(
        self,
        pid: str,
        path: str | os.PathLike[str],
        recursive: bool = False,
        missing_ok: bool = False,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: Iterable[str] | None = None,
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
        target_binding_snapshot: dict[str, tuple[str, int]] = {}
        if recursive:
            (
                target_context,
                target_label_generation,
                target_binding_snapshot,
            ) = self._data_flow().file_tree_deletion_snapshot(relative)
        else:
            (
                target_context,
                target_label_generation,
                target_binding_snapshot,
            ) = self._data_flow().file_deletion_snapshot(relative)
        flow_context = DataFlowContext.aggregate(
            (
                self._data_flow().context_from_source_oids(pid, source_oids),
                target_context,
            )
        )
        sink = DataSink(self.resource_for(relative))
        data_flow_payload = {
            "path": relative,
            "recursive": recursive,
            "missing_ok": missing_ok,
        }
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=data_flow_payload,
            operation="filesystem.delete_directory",
            target_state_version=target_label_generation,
        )
        decision, authority_context = self._require_delete(
            pid=pid,
            resource=resource,
            target=target,
            relative=relative,
            operation="delete_directory",
            recursive=recursive,
            missing_ok=missing_ok,
            source_oids=source_oids,
        )
        effect_context = {
            "path": relative,
            "resource": resource,
            "recursive": recursive,
            "missing_ok": missing_ok,
        }
        intent: dict[str, Any] = {}

        def prepare() -> None:
            intent["record"] = self._record_mutation_intent(
                pid=pid,
                action="primitive.filesystem.delete_directory.intent",
                target=resource,
                decision={"path": relative, "recursive": recursive, "missing_ok": missing_ok},
            )

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=authority_context,
            observation=effect_context,
            data_sink=sink,
            data_flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            data_flow_operation="filesystem.delete_directory",
            data_flow_target_state_version=target_label_generation,
            data_flow_target_state_version_resolver=lambda: (
                self._data_flow().file_tree_state_version(relative)
                if recursive
                else self._data_flow().file_state_version(relative)
            ),
            prepare=prepare,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid, resource, "primitive.filesystem.delete_directory.failed", effect_context, error, phase, intent.get("record")
            ),
        )
        with self._protected().start(
            "primitive.filesystem.delete_directory", invocation, provider=self.provider
        ) as protected:
            target_state = protected.call(
                ProviderPhase("state", information_flow=True), self.provider.state, target
            )
            effect_context["state_observed"] = True
            if not target_state.exists:
                if not missing_ok:
                    error = NotFound(f"directory does not exist: {relative}")
                    self._complete_state_rejection(
                        protected,
                        pid=pid,
                        target=resource,
                        audit_action="primitive.filesystem.delete_directory.rejected",
                        context=effect_context,
                        error=error,
                        intent_record=intent.get("record"),
                    )
                    raise error
                result = DeleteResult(
                    path=relative, kind="missing", deleted=False, recursive=recursive
                )
                result_payload = {
                    "path": relative,
                    "deleted": False,
                    "missing_ok": True,
                    "recursive": recursive,
                }
                return protected.complete(
                    result,
                    self._protected_filesystem_evidence(
                        pid, resource, EventType.EXTERNAL_WRITE, "primitive.filesystem.delete_directory",
                        {"adapter": "filesystem", "operation": "delete_directory", **result_payload},
                        result_payload, result_payload, intent.get("record"),
                    ),
                    classification_override=self._state_only_classification(),
                )
            if target_state.kind != "directory":
                error = CapabilityDenied(f"path is not a directory: {relative}")
                self._complete_state_rejection(
                    protected,
                    pid=pid,
                    target=resource,
                    audit_action="primitive.filesystem.delete_directory.rejected",
                    context=effect_context,
                    error=error,
                    intent_record=intent.get("record"),
                )
                raise error
            protected_descendant_names = (
                (".git",)
                if recursive and self.config.git.protect_git_metadata
                else ()
            )
            if protected_descendant_names:
                protected_metadata_scan = getattr(
                    self.provider,
                    "contains_descendant_name",
                    None,
                )
                protected_directory_delete = getattr(
                    self.provider,
                    "delete_directory_protected",
                    None,
                )
                if not callable(protected_metadata_scan) or not callable(
                    protected_directory_delete
                ):
                    error = ValidationError(
                        "filesystem provider must support protected recursive deletion"
                    )
                    self._complete_state_rejection(
                        protected,
                        pid=pid,
                        target=resource,
                        audit_action="primitive.filesystem.delete_directory.rejected",
                        context=effect_context,
                        error=error,
                        intent_record=intent.get("record"),
                    )
                    raise error
                contains_protected_metadata = protected.call(
                    ProviderPhase("protected_metadata_scan", information_flow=True),
                    protected_metadata_scan,
                    target,
                    names=protected_descendant_names,
                )
                if contains_protected_metadata:
                    error = CapabilityDenied(
                        "Git metadata is only accessible through the Runtime Git primitive"
                    )
                    self._complete_state_rejection(
                        protected,
                        pid=pid,
                        target=resource,
                        audit_action="primitive.filesystem.delete_directory.rejected",
                        context=effect_context,
                        error=error,
                        intent_record=intent.get("record"),
                    )
                    raise error
                protected.call(
                    ProviderPhase("delete", state_mutation=True, information_flow=True),
                    protected_directory_delete,
                    target,
                    recursive=True,
                    protected_descendant_names=protected_descendant_names,
                )
            else:
                protected.call(
                    ProviderPhase("delete", state_mutation=True, information_flow=True),
                    self.provider.delete_directory,
                    target,
                    recursive=recursive,
                )
            result = DeleteResult(
                path=relative, kind="directory", deleted=True, recursive=recursive
            )
            result_payload = {"deleted": True, "recursive": recursive}
            return protected.complete(
                result,
                self._protected_filesystem_evidence(
                    pid, resource, EventType.EXTERNAL_WRITE, "primitive.filesystem.delete_directory",
                    {"adapter": "filesystem", "operation": "delete_directory", "path": relative, "recursive": recursive},
                    {"path": relative, **result_payload}, result_payload, intent.get("record"),
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                settle_success=lambda: self._data_flow().tombstone_path_tree(
                    pid=pid,
                    expected_bindings=target_binding_snapshot,
                ),
            )

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
        metadata: dict[str, Any] | None = None,
    ) -> Capability:
        return self.capabilities.grant(
            subject=pid,
            resource=self.resource_for_path(path, cwd=cwd),
            rights=rights,
            issued_by=issued_by,
            delegable=delegable,
            metadata=metadata,
        )

    def grant_directory(
        self,
        pid: str,
        path: str | os.PathLike[str],
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "filesystem",
        cwd: str | os.PathLike[str] | None = None,
        delegable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Capability:
        return self.capabilities.grant(
            subject=pid,
            resource=self.directory_resource_for_path(path, cwd=cwd),
            rights=rights,
            issued_by=issued_by,
            delegable=delegable,
            metadata=metadata,
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

    def _protected(self) -> Any:
        return self.protected_operations

    def _data_flow(self) -> Any:
        manager = getattr(self, "data_flow", None) or getattr(
            self._protected(),
            "data_flow",
            None,
        )
        if manager is None:
            raise ValidationError("filesystem data-flow manager is not attached")
        return manager

    @staticmethod
    def _state_only_classification(
        outcome: str = "state_observed",
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={"outcome": outcome},
        )

    def _complete_state_rejection(
        self,
        protected: Any,
        *,
        pid: str,
        target: str,
        audit_action: str,
        context: dict[str, Any],
        error: BaseException,
        intent_record: Any | None = None,
        resource_source: str | None = None,
    ) -> None:
        outcome = "rejected_after_state_observation"
        result = {
            "outcome": outcome,
            "phase": "local_validation",
            "error_type": type(error).__name__,
        }
        resource = (
            ResourceSettlement(
                usage=ResourceUsage(),
                source=resource_source,
                context=context,
            )
            if resource_source is not None
            else None
        )
        protected.complete(
            None,
            self._protected_filesystem_evidence(
                pid,
                target,
                EventType.EXTERNAL_READ,
                audit_action,
                {"adapter": "filesystem", **result},
                {
                    "path": context.get("path"),
                    "effect_outcome": "failed",
                    **result,
                },
                result,
                intent_record,
            ),
            classification_override=self._state_only_classification(outcome),
            resource=resource,
        )

    def _protected_filesystem_evidence(
        self,
        pid: str,
        target: str,
        event_type: EventType,
        audit_action: str,
        event_payload: dict[str, Any],
        audit_decision: dict[str, Any],
        effect_metadata: dict[str, Any],
        intent_record: Any | None = None,
    ) -> ProtectedOperationEvidence:
        parent_id = getattr(intent_record, "record_id", None)
        return ProtectedOperationEvidence(
            event_type=event_type,
            event_source=pid,
            event_target=target,
            event_payload=event_payload,
            audit_action=audit_action,
            audit_actor=pid,
            audit_target=target,
            audit_decision=audit_decision,
            correlation_id=parent_id,
            parent_record_id=parent_id,
            effect_metadata=effect_metadata,
        )

    def _protected_failure_evidence(
        self,
        pid: str,
        target: str,
        audit_action: str,
        context: dict[str, Any],
        error: BaseException,
        phase: str,
        intent_record: Any | None = None,
    ) -> ProtectedOperationEvidence:
        is_mutation = any(
            marker in audit_action for marker in ("write", "delete", "make_directory")
        )
        return self._protected_filesystem_evidence(
            pid,
            target,
            EventType.EXTERNAL_WRITE if is_mutation else EventType.EXTERNAL_READ,
            audit_action,
            {
                "adapter": "filesystem",
                "outcome": "unknown",
                "phase": phase,
                "error_type": type(error).__name__,
            },
            {
                "path": context.get("path"),
                "effect_outcome": "unknown",
                "phase": phase,
                "error_type": type(error).__name__,
            },
            {"outcome": "unknown", "phase": phase, "error_type": type(error).__name__},
            intent_record,
        )

    def _read_limit_for_state(self, size_bytes: int | None, max_bytes: int) -> int:
        # The state snapshot is advisory: a file may grow between state() and
        # read_bytes(). If it already proves truncation, do not read beyond the
        # caller's information-flow budget. Otherwise request a sentinel byte
        # so growth cannot turn a partial read into a false complete result.
        if size_bytes is not None and size_bytes > max_bytes:
            return max_bytes
        return max_bytes + 1

    def _is_truncated_read(self, size_bytes: int | None, bytes_read: int, max_bytes: int) -> bool:
        return (size_bytes is not None and size_bytes > max_bytes) or bytes_read > max_bytes

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

    def _missing_parent_paths(self, normalized_path: str) -> list[str]:
        parts = [part for part in str(normalized_path).split("/") if part]
        parents = ["/".join(parts[:index]) for index in range(1, len(parts))]
        missing: list[str] = []
        for parent in parents:
            target = self.provider.resolve(parent)
            if not self.provider.state(target).exists:
                missing.append(target.relative)
        return missing

    def _bind_written_path_set(
        self,
        *,
        pid: str,
        context: DataFlowContext,
        parent_paths: Iterable[str],
        final_path: str,
        final_content: bytes,
    ) -> None:
        with self._data_flow().store.transaction():
            for parent in dict.fromkeys(parent_paths):
                self._data_flow().bind_written_file(
                    pid=pid,
                    normalized_path=parent,
                    content=b"<agent-libos-directory>",
                    context=context,
                )
            self._data_flow().bind_written_file(
                pid=pid,
                normalized_path=final_path,
                content=final_content,
                context=context,
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

    def _resolve(
        self,
        path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
    ) -> tuple[ResolvedPath, str]:
        target = self.provider.resolve(self._path_with_cwd(path, cwd))
        if self.config.git.protect_git_metadata and any(
            part.casefold() == ".git" for part in target.relative.split("/")
        ):
            raise CapabilityDenied(
                "Git metadata is only accessible through the Runtime Git primitive"
            )
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
        source_oids: Iterable[str] | None = None,
    ) -> tuple[CapabilityDecision, dict[str, Any]]:
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
            source_oids=source_oids,
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

    def _require_read_operation(
        self,
        *,
        pid: str,
        resource: str,
        target: ResolvedPath,
        relative: str,
        primitive: str,
        operation: str,
        question: str,
        extra_context: dict[str, Any] | None = None,
        source_oids: Iterable[str] | None = None,
    ) -> tuple[CapabilityDecision, dict[str, Any]]:
        authority_context = self._authorization_context(
            pid=pid,
            resource=resource,
            relative=relative,
            primitive=primitive,
            operation=operation,
            right=CapabilityRight.READ.value,
            extra=extra_context,
        )
        decision = self.capabilities.authorize(
            pid,
            resource,
            CapabilityRight.READ,
            authority_context,
        )
        if decision.allowed:
            return (
                self.capabilities.require(
                    pid,
                    resource,
                    CapabilityRight.READ,
                    authority_context,
                    consume=False,
                ),
                authority_context,
            )
        if decision.policy == CapabilityManager.ALWAYS_DENY:
            return (
                self.capabilities.require(
                    pid,
                    resource,
                    CapabilityRight.READ,
                    authority_context,
                    consume=False,
                ),
                authority_context,
            )
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            operation_context = self._operation_context(
                pid=pid,
                resource=resource,
                target=target,
                relative=relative,
                primitive=primitive,
                operation=operation,
                right=CapabilityRight.READ.value,
                extra=extra_context or {},
            )
            decision = self.capabilities.authorize(
                pid,
                resource,
                CapabilityRight.READ,
                operation_context,
                audit=True,
            )
            if decision.allowed:
                return decision, operation_context
            if decision.policy == CapabilityManager.ALWAYS_DENY:
                raise CapabilityDenied(decision.reason)
            if decision.policy != CapabilityManager.ASK_EACH_TIME:
                raise CapabilityDenied(decision.reason)
            if self.human is None:
                raise CapabilityDenied(
                    f"{pid} requires human approval for read on {resource}"
                )
            request_id = self.human.query(
                pid=pid,
                human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": question,
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [CapabilityRight.READ.value],
                        "constraints": self._approval_constraints(
                            operation_context,
                            right=CapabilityRight.READ.value,
                        ),
                    },
                    "context": operation_context,
                },
                blocking=True,
                source_oids=source_oids,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to read {resource}",
            )
        return (
            self.capabilities.require(
                pid,
                resource,
                CapabilityRight.READ,
                authority_context,
                consume=False,
            ),
            authority_context,
        )

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
        source_oids: Iterable[str] | None = None,
    ) -> tuple[CapabilityDecision, dict[str, Any]]:
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
            return decision, operation_context
        if decision.policy == CapabilityManager.ALWAYS_DENY:
            raise CapabilityDenied(f"{pid} denied write on {resource}")
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for write on {resource}")
            # This primitive has the concrete path, caller-declared overwrite
            # policy, byte count, and preview needed for a safe per-use human
            # decision. Target state is deliberately deferred until after the
            # one-time approval has been issued and reserved.
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
                source_oids=source_oids,
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
        source_oids: Iterable[str] | None = None,
    ) -> tuple[CapabilityDecision, dict[str, Any]]:
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
            return decision, operation_context
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
                source_oids=source_oids,
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
            "target_state_observation": "deferred_until_authorized",
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
