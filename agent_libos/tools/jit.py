from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.config import AgentLibOSConfig
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.models import (
    JIT_MULTIPLEXER_TOOL_NAME,
    JITRehydrationArtifact,
    JITRehydrationRecord,
    JITRehydrationSummary,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectMetadata,
    ObjectType,
    OPENAI_TOOL_NAME_MAX_CHARS,
    ResourceUsage,
    ProcessToolBindingCursor,
    ProcessToolBindingPage,
    ProcessToolBindingRecord,
    ToolCandidate,
    ToolCandidateStatus,
    ToolHandle,
    ToolSpec,
    ValidationResult,
    is_openai_tool_name,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.ports import AuditPort
from agent_libos.storage import UnitOfWork
from agent_libos.substrate import (
    CommandMetrics,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
)
from agent_libos.tools.observability import ensure_json_size, sanitize_for_observability
from agent_libos.tools.registry import ToolRegistry
from agent_libos.tools.sandbox import SandboxBackend
from agent_libos.utils.ids import new_id, utc_now


class JITToolService:
    """Candidate validation and atomic process-local JIT publication."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        memory: ObjectMemoryManager,
        audit: AuditPort,
        sandbox: SandboxBackend,
        registry: ToolRegistry,
        config: AgentLibOSConfig,
        declared_permissions: Iterable[str],
        resources: Any | None,
        images: dict[str, Any],
        require_recovery_lease: Callable[[], None] | None = None,
    ) -> None:
        self.unit_of_work = unit_of_work
        self.extensions = unit_of_work.extensions
        self.processes = unit_of_work.processes
        self.memory = memory
        self.audit = audit
        self.sandbox = sandbox
        self.registry = registry
        self.config = config
        self.declared_permissions = frozenset(declared_permissions)
        self.resources = resources
        self.images = images
        self._require_recovery_lease = (
            require_recovery_lease
            if require_recovery_lease is not None
            else self._recovery_lease_not_configured
        )

    def propose(
        self,
        pid: str,
        spec: ToolSpec | dict[str, Any],
        source_code: str,
        tests: list[dict[str, Any]] | None = None,
        requested_capabilities: list[dict[str, Any]] | None = None,
    ) -> str:
        candidate_id, _descriptor_oid = self.propose_with_descriptor(
            pid,
            spec,
            source_code,
            tests=tests,
            requested_capabilities=requested_capabilities,
        )
        return candidate_id

    def propose_with_descriptor(
        self,
        pid: str,
        spec: ToolSpec | dict[str, Any],
        source_code: str,
        tests: list[dict[str, Any]] | None = None,
        requested_capabilities: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        raw_spec = spec if isinstance(spec, ToolSpec) else ToolSpec(**spec)
        tool_spec = conservative_jit_tool_spec(
            raw_spec,
            declared_permissions=self.declared_permissions,
        )
        self._validate_tool_spec(tool_spec)
        if (
            self.exposure_for_process(pid) == JIT_TOOL_EXPOSURE_MULTIPLEXED
            and tool_spec.name == JIT_MULTIPLEXER_TOOL_NAME
        ):
            raise ValidationError(
                f"{JIT_MULTIPLEXER_TOOL_NAME} is reserved by multiplexed JIT tool exposure"
            )
        selected_tests = list(tests or [])
        self._validate_source_and_tests(source_code, selected_tests)
        now = utc_now()
        candidate = ToolCandidate(
            candidate_id=new_id("tcand"),
            pid=pid,
            spec=tool_spec,
            source_code=source_code,
            tests=selected_tests,
            requested_capabilities=list(requested_capabilities or []),
            status=ToolCandidateStatus.PROPOSED,
            validation=None,
            created_at=now,
            updated_at=now,
        )
        with self.memory.ownership_locked(), self.unit_of_work.transaction(
            include_object_payloads=True
        ):
            self.extensions.insert_tool_candidate(candidate)
            candidate_object = self.memory.create_object(
                pid=pid,
                object_type=ObjectType.TOOL_CANDIDATE,
                payload={
                    "candidate_id": candidate.candidate_id,
                    "language": self.sandbox.language,
                    "spec": {
                        "name": tool_spec.name,
                        "description": tool_spec.description,
                        "input_schema": tool_spec.input_schema,
                        "output_schema": tool_spec.output_schema,
                        "side_effects": tool_spec.side_effects,
                    },
                    "tests": candidate.tests,
                    "requested_capabilities": candidate.requested_capabilities,
                },
                metadata=ObjectMetadata(
                    title=f"Tool candidate: {tool_spec.name}",
                    tags=["tool", "candidate"],
                ),
                immutable=True,
            )
            self.audit.record(
                actor=pid,
                action="tool.propose",
                target=f"tool_candidate:{candidate.candidate_id}",
                output_refs=[candidate_object.oid],
                decision={"name": tool_spec.name},
            )
        return candidate.candidate_id, candidate_object.oid

    def validate(
        self,
        candidate_id: str,
        *,
        pid: str | None = None,
    ) -> ValidationResult:
        candidate = self.get_candidate(candidate_id)
        owner_pid = pid or candidate.pid
        self.require_candidate_owner(candidate, owner_pid)
        try:
            result = self._run_candidate_tests(candidate, owner_pid)
        except (SubprocessLimitExceeded, SubprocessTimeoutExpired) as exc:
            self._charge_subprocess_metrics(
                owner_pid,
                exc.metrics,
                source="tool.validate.deno",
                context={
                    "candidate_id": candidate_id,
                    "tool": candidate.spec.name,
                },
            )
            raise
        errors = list(result.errors)
        if candidate.requested_capabilities:
            errors.append("Deno/TypeScript JIT tools cannot request external capabilities")
        validation = ValidationResult(
            ok=not errors and result.ok,
            errors=errors,
            warnings=list(result.warnings),
            logs=result.logs,
            metadata=result.metadata,
        )
        candidate.validation = self._validation_observation(
            validation,
            self.sandbox.metadata_for_source(candidate.source_code),
        )
        candidate.status = (
            ToolCandidateStatus.VALIDATED
            if validation.ok
            else ToolCandidateStatus.REJECTED
        )
        candidate.updated_at = utc_now()
        with self.unit_of_work.transaction():
            current = self.get_candidate(candidate_id)
            self.require_candidate_owner(current, owner_pid)
            state_changed = current.status != ToolCandidateStatus.REGISTERED
            if state_changed:
                current.validation = candidate.validation
                current.status = candidate.status
                current.updated_at = candidate.updated_at
                self.extensions.update_tool_candidate(current)
            self.audit.record(
                actor="tool_broker",
                action="tool.validate",
                target=f"tool_candidate:{candidate_id}",
                decision={
                    **(candidate.validation or {}),
                    "candidate_state_changed": state_changed,
                },
            )
        return validation

    def register(
        self,
        pid: str,
        candidate_id: str,
        *,
        approver: str,
        scope: str,
        replace_tool_id: str | None,
    ) -> ToolHandle:
        candidate = self.get_candidate(candidate_id)
        self.require_candidate_owner(candidate, pid)
        if candidate.status == ToolCandidateStatus.REGISTERED:
            raise ValidationError(f"tool candidate is already registered: {candidate_id}")
        if self.registry.name_collides_with_static_tool(candidate.spec.name):
            raise ValidationError(f"tool name already exists: {candidate.spec.name}")
        if candidate.status != ToolCandidateStatus.VALIDATED:
            validation = self.validate(candidate_id, pid=pid)
            if not validation.ok:
                raise ValidationError("; ".join(validation.errors))
        return self._publish(
            pid,
            candidate_id,
            approver=approver,
            scope=scope,
            replace_tool_id=replace_tool_id,
        )

    def _publish(
        self,
        pid: str,
        candidate_id: str,
        *,
        approver: str,
        scope: str,
        replace_tool_id: str | None,
    ) -> ToolHandle:
        candidate = self.get_candidate(candidate_id)
        tool_id = new_id("tool")
        handle = ToolHandle(
            tool_id=tool_id,
            name=candidate.spec.name,
            capability_id=None,
            scope=scope,
        )
        try:
            with self.unit_of_work.transaction():
                candidate = self.get_candidate(candidate_id)
                self.require_candidate_owner(candidate, pid)
                self._require_publishable(candidate, candidate_id)
                process = self.processes.get_process(pid)
                if process is None:
                    raise NotFound(f"process not found: {pid}")
                self._validate_replacement(
                    process.tool_table.get(candidate.spec.name),
                    replace_tool_id,
                    candidate.spec.name,
                )
                self.extensions.insert_tool(
                    handle,
                    candidate.spec,
                    registered_by=approver,
                    created_at=utc_now(),
                    ephemeral=True,
                )
                candidate.status = ToolCandidateStatus.REGISTERED
                candidate.registered_tool_id = tool_id
                candidate.updated_at = utc_now()
                self.extensions.update_tool_candidate(candidate)
                process.tool_table[candidate.spec.name] = tool_id
                process.model_tool_table[candidate.spec.name] = tool_id
                process.updated_at = utc_now()
                self.processes.patch_process(
                    pid,
                    {
                        "tool_table": process.tool_table,
                        "model_tool_table": process.model_tool_table,
                        "updated_at": process.updated_at,
                    },
                    expected_revision=process.revision,
                )
                self.registry.publish_jit(handle, candidate.source_code)
                self.audit.record(
                    actor=approver,
                    action="tool.register",
                    target=f"tool:{tool_id}",
                    decision={
                        "candidate_id": candidate_id,
                        "scope": scope,
                        "replaced_tool_id": replace_tool_id,
                    },
                )
        except BaseException:
            self.registry.forget_jit(tool_id)
            raise
        return handle

    def install_committed(
        self,
        pid: str,
        *,
        name: str,
        scope: str,
        spec: ToolSpec,
        source_code: str,
        registered_by: str,
    ) -> tuple[ToolHandle, str]:
        """Install a JIT tool carried by a trusted checkpoint-commit artifact."""

        tool_id = new_id("tool")
        handle = ToolHandle(
            tool_id=tool_id,
            name=name,
            capability_id=None,
            scope=scope,
        )
        now = utc_now()
        candidate = ToolCandidate(
            candidate_id=new_id("tcand"),
            pid=pid,
            spec=spec,
            source_code=source_code,
            tests=[],
            requested_capabilities=[],
            status=ToolCandidateStatus.REGISTERED,
            validation={"ok": True, "source": "image.commit"},
            created_at=now,
            updated_at=now,
            registered_tool_id=tool_id,
        )
        try:
            with self.unit_of_work.transaction():
                self.extensions.insert_tool(
                    handle,
                    spec,
                    registered_by=registered_by,
                    created_at=now,
                    ephemeral=True,
                )
                self.extensions.insert_tool_candidate(candidate)
                self.registry.publish_jit(handle, source_code)
        except BaseException:
            self.registry.forget_jit(tool_id)
            raise
        return handle, candidate.candidate_id

    def rehydrate_registered(self) -> JITRehydrationSummary:
        """Restore JIT bindings with bounded historical scans and diagnostics."""

        self._require_recovery_lease()
        page_size = self.config.runtime.jit_rehydration_page_size
        restored_total = 0
        pruned_total = 0
        restored_sample: list[JITRehydrationRecord] = []
        pruned_sample: list[JITRehydrationRecord] = []
        cursor: ProcessToolBindingCursor | None = None
        while True:
            page = self.processes.query_process_tool_bindings(
                after=cursor,
                limit=page_size,
            )
            self._validate_process_page(page, after=cursor, limit=page_size)
            page_restored, page_pruned = self._rehydrate_binding_page(page.records)
            restored_total += len(page_restored)
            pruned_total += len(page_pruned)
            self._extend_sample(
                restored_sample,
                page_restored,
                limit=page_size,
            )
            self._extend_sample(
                pruned_sample,
                page_pruned,
                limit=page_size,
            )
            cursor = page.next_cursor
            if cursor is None:
                break

        summary = JITRehydrationSummary(
            restored_total=restored_total,
            pruned_stale_total=pruned_total,
            restored_sample=tuple(restored_sample),
            pruned_stale_sample=tuple(pruned_sample),
        )
        self._record_rehydration_summary(summary)
        return summary

    def preflight_process_bindings(
        self,
        pid: str,
        handles: Iterable[ToolHandle],
        *,
        assigned_by: str,
    ) -> None:
        """Reject process-local JIT aliases whose durable owner is not ``pid``.

        Loaded registry state is sufficient to select JIT implementations, but
        it is not an ownership authority.  The exact ephemeral tool and
        registered-candidate rows remain the source of truth for process-local
        binding decisions.
        """

        selected_handles = tuple(handles)
        if not selected_handles:
            return
        artifacts_by_id: dict[str, JITRehydrationArtifact] = {}
        hard_limit = self.config.runtime.jit_rehydration_page_hard_limit
        ordered_ids = sorted({handle.tool_id for handle in selected_handles})
        for offset in range(0, len(ordered_ids), hard_limit):
            requested_ids = frozenset(
                ordered_ids[offset : offset + hard_limit]
            )
            artifacts = self.extensions.get_jit_rehydration_artifacts(
                pid=pid,
                tool_ids=requested_ids,
            )
            artifacts_by_id.update(
                self._validated_artifact_map(
                    artifacts,
                    requested_ids=requested_ids,
                )
            )
        rejected: list[ToolHandle] = []
        for handle in selected_handles:
            tool_id = handle.tool_id
            artifact = artifacts_by_id.get(tool_id)
            loaded_jit = self.registry.is_jit(tool_id)
            loaded_static = self.registry.implementation(tool_id) is not None
            candidate_backed = bool(
                artifact is not None
                and artifact.candidate_match_count > 0
            )
            if not loaded_jit and not candidate_backed:
                if loaded_static or artifact is None or artifact.scope != "ephemeral_process":
                    continue
            if artifact is None and handle.scope != "ephemeral_process":
                continue
            if artifact is None or not self._artifact_matches_binding(
                artifact,
                pid=pid,
                name=handle.name,
            ):
                rejected.append(handle)
        if not rejected:
            return
        rejected_tools = [
            {"name": handle.name, "tool_id": handle.tool_id}
            for handle in sorted(rejected, key=lambda item: (item.name, item.tool_id))
        ]
        self.audit.record(
            actor=assigned_by,
            action="process.tools.configure_denied",
            target=f"process:{pid}",
            decision={
                "reason": "process_local_jit_owner_mismatch",
                "tools": rejected_tools,
            },
        )
        raise ValidationError(
            "process-local JIT tools may only be bound to their durable owner"
        )

    def _rehydrate_binding_page(
        self,
        records: tuple[ProcessToolBindingRecord, ...],
    ) -> tuple[list[JITRehydrationRecord], list[JITRehydrationRecord]]:
        requested_ids = frozenset(record.tool_id for record in records)
        if not requested_ids:
            return [], []
        artifacts = self.extensions.get_jit_rehydration_artifacts_for_tool_ids(
            requested_ids
        )
        artifacts_by_id = self._validated_artifact_map(
            artifacts,
            requested_ids=requested_ids,
        )
        if frozenset(artifacts_by_id) != requested_ids:
            raise ValidationError(
                "JIT binding recovery artifact page changed during recovery"
            )
        restored: list[JITRehydrationRecord] = []
        pruned: list[JITRehydrationRecord] = []
        stale_bindings: dict[str, dict[str, str]] = {}
        for binding_record in records:
            name = binding_record.tool_name
            tool_id = binding_record.tool_id
            artifact = artifacts_by_id[tool_id]
            record = JITRehydrationRecord(
                pid=binding_record.pid,
                tool_id=tool_id,
                name=name,
            )
            if not self._artifact_matches_binding(
                artifact,
                pid=binding_record.pid,
                name=name,
            ):
                stale_bindings.setdefault(binding_record.pid, {})[name] = tool_id
                pruned.append(record)
            elif not self._is_loaded(tool_id):
                self._publish_rehydrated_artifact(artifact)
                restored.append(record)
        for pid, bindings in stale_bindings.items():
            self.processes.remove_process_tool_bindings(pid, bindings)
        return restored, pruned

    def _is_loaded(self, tool_id: str) -> bool:
        return (
            self.registry.implementation(tool_id) is not None
            or self.registry.is_jit(tool_id)
        )

    def _publish_rehydrated_artifact(self, artifact: JITRehydrationArtifact) -> None:
        source = artifact.source_code
        if source is None:
            raise ValidationError("rehydratable JIT artifact is missing source code")
        self.registry.publish_jit(
            ToolHandle(
                tool_id=artifact.tool_id,
                name=artifact.name,
                capability_id=None,
                scope=artifact.scope,
            ),
            source,
        )

    @staticmethod
    def _artifact_matches_binding(
        artifact: JITRehydrationArtifact,
        *,
        pid: str,
        name: str,
    ) -> bool:
        return (
            artifact.name == name
            and artifact.candidate_match_count == 1
            and artifact.candidate_pid == pid
            and bool(artifact.source_code)
        )

    @staticmethod
    def _validated_artifact_map(
        artifacts: Iterable[JITRehydrationArtifact],
        *,
        requested_ids: frozenset[str],
    ) -> dict[str, JITRehydrationArtifact]:
        selected: dict[str, JITRehydrationArtifact] = {}
        for artifact in artifacts:
            if not isinstance(artifact, JITRehydrationArtifact):
                raise ValidationError("JIT rehydration returned an invalid artifact")
            if artifact.tool_id not in requested_ids or artifact.tool_id in selected:
                raise ValidationError(
                    "JIT rehydration returned a duplicate or unexpected artifact"
                )
            selected[artifact.tool_id] = artifact
        return selected

    @staticmethod
    def _validate_process_page(
        page: ProcessToolBindingPage,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> None:
        if not isinstance(page, ProcessToolBindingPage) or len(page.records) > limit:
            raise ValidationError("process tool binding recovery returned an invalid page")
        previous = after
        for process in page.records:
            if not isinstance(process, ProcessToolBindingRecord):
                raise ValidationError(
                    "process tool binding recovery returned an invalid record"
                )
            current = ProcessToolBindingCursor(
                process.pid,
                process.tool_name,
            )
            if previous is not None and current <= previous:
                raise ValidationError(
                    "process tool binding recovery page is not strictly ordered"
                )
            previous = current
        if page.next_cursor is not None:
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "process tool binding recovery returned an invalid next cursor"
                )

    @staticmethod
    def _extend_sample(
        selected: list[JITRehydrationRecord],
        records: Iterable[JITRehydrationRecord],
        *,
        limit: int,
    ) -> None:
        remaining = limit - len(selected)
        if remaining > 0:
            selected.extend(islice(records, remaining))

    def _record_rehydration_summary(self, summary: JITRehydrationSummary) -> None:
        if summary.restored_total == 0 and summary.pruned_stale_total == 0:
            return
        self.audit.record(
            actor="runtime",
            action="runtime.jit.rehydrate",
            target="tool:jits",
            decision={
                "restored_total": summary.restored_total,
                "pruned_stale_total": summary.pruned_stale_total,
                "restored": [item.to_mapping() for item in summary.restored_sample],
                "pruned_stale": [
                    item.to_mapping() for item in summary.pruned_stale_sample
                ],
                "restored_truncated": summary.restored_truncated,
                "pruned_stale_truncated": summary.pruned_stale_truncated,
            },
        )

    @staticmethod
    def _recovery_lease_not_configured() -> None:
        raise RuntimeError("JIT rehydration requires a configured recovery lease")

    def get_candidate(self, candidate_id: str) -> ToolCandidate:
        candidate = self.extensions.get_tool_candidate(candidate_id)
        if candidate is None:
            raise NotFound(f"tool candidate not found: {candidate_id}")
        return candidate

    @staticmethod
    def require_candidate_owner(candidate: ToolCandidate, pid: str) -> None:
        if candidate.pid != pid:
            raise ValidationError(
                f"tool candidate {candidate.candidate_id} belongs to "
                f"process {candidate.pid}, not {pid}"
            )

    @staticmethod
    def _require_publishable(candidate: ToolCandidate, candidate_id: str) -> None:
        if candidate.status == ToolCandidateStatus.REGISTERED:
            raise ValidationError(f"tool candidate is already registered: {candidate_id}")
        if candidate.status != ToolCandidateStatus.VALIDATED:
            raise ValidationError(f"tool candidate is not validated: {candidate_id}")

    @staticmethod
    def _validate_replacement(
        existing_tool_id: str | None,
        replace_tool_id: str | None,
        name: str,
    ) -> None:
        if existing_tool_id is not None and existing_tool_id != replace_tool_id:
            raise ValidationError(f"process already has a tool named: {name}")
        if existing_tool_id is None and replace_tool_id is not None:
            raise ValidationError(
                f"tool replacement target is stale for {name}: {replace_tool_id}"
            )

    def exposure_for_process(self, pid: str) -> str:
        process = self.processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        image = self.images.get(process.image_id)
        return str(getattr(image, "jit_tool_exposure", "") or "")

    def static_check_source(self, source: str) -> ValidationResult:
        return self.sandbox.static_check(source)

    def _supported_run_tests_kwargs(self) -> set[str]:
        signature = inspect.signature(self.sandbox.run_tests)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return {"timeout", "limits", "return_metrics"}
        return set(signature.parameters)

    def _require_validation_resource_controls(self) -> None:
        supported = self._supported_run_tests_kwargs()
        if "limits" not in supported:
            raise ValidationError(
                "sandbox backend must accept SubprocessLimits when validating with resource limits"
            )
        if "return_metrics" not in supported:
            raise ValidationError(
                "sandbox backend must return validation subprocess metrics"
            )

    def _run_candidate_tests(
        self,
        candidate: ToolCandidate,
        owner_pid: str,
    ) -> ValidationResult:
        batches = [[test] for test in candidate.tests] or [[]]
        errors: list[str] = []
        warnings: list[str] = []
        logs: list[str] = []
        metrics: list[CommandMetrics] = []
        ok = True
        for index, tests in enumerate(batches, start=1):
            limits = self._subprocess_limits(owner_pid)
            self._preflight_validation_budget(
                owner_pid,
                limits,
                candidate,
                index,
            )
            result = self._run_candidate_test_batch(candidate, tests, limits)
            ok = ok and result.ok
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            if result.logs:
                logs.append(result.logs)
            result_metrics = self._metrics_from_validation(
                result.metadata.get("metrics")
            )
            self._charge_subprocess_metrics(
                owner_pid,
                result_metrics,
                source="tool.validate.deno",
                context={
                    "candidate_id": candidate.candidate_id,
                    "tool": candidate.spec.name,
                    "test_index": index,
                },
            )
            if result_metrics is not None:
                metrics.append(result_metrics)
        return ValidationResult(
            ok=ok and not errors,
            errors=errors,
            warnings=warnings,
            logs=self._bounded_validation_logs(logs),
            metadata={"metrics": self._aggregate_command_metrics(metrics)},
        )

    def _run_candidate_test_batch(
        self,
        candidate: ToolCandidate,
        tests: list[dict[str, Any]],
        limits: SubprocessLimits | None,
    ) -> ValidationResult:
        kwargs: dict[str, Any] = {
            "timeout": self.config.tools.jit_validation_timeout_s,
            "limits": limits,
            "return_metrics": True,
        }
        if limits is not None:
            self._require_validation_resource_controls()
        supported = self._supported_run_tests_kwargs()
        selected = {key: value for key, value in kwargs.items() if key in supported}
        if limits is not None and "limits" not in selected:
            raise ValidationError(
                "sandbox backend must accept SubprocessLimits when validating with resource limits"
            )
        if limits is not None and "return_metrics" not in selected:
            raise ValidationError(
                "sandbox backend must return validation subprocess metrics"
            )
        return self.sandbox.run_tests(candidate.source_code, tests, **selected)

    def _preflight_validation_budget(
        self,
        owner_pid: str,
        limits: SubprocessLimits | None,
        candidate: ToolCandidate,
        test_index: int,
    ) -> None:
        if self.resources is None or limits is None:
            return
        usage = ResourceUsage()
        if limits.wall_seconds is not None and limits.wall_seconds <= 0:
            usage = ResourceUsage(subprocess_wall_seconds=1e-9)
        elif limits.cpu_seconds is not None and limits.cpu_seconds <= 0:
            usage = ResourceUsage(subprocess_cpu_seconds=1e-9)
        elif limits.memory_bytes is not None and limits.memory_bytes <= 0:
            usage = ResourceUsage(subprocess_peak_memory_bytes=1)
        else:
            return
        self.resources.preflight(
            owner_pid,
            usage,
            source="tool.validate.deno",
            context={
                "candidate_id": candidate.candidate_id,
                "tool": candidate.spec.name,
                "test_index": test_index,
            },
        )

    def _validate_source_and_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
    ) -> None:
        if len(source_code) > self.config.tools.jit_source_max_chars:
            raise ValidationError(
                f"JIT source exceeds {self.config.tools.jit_source_max_chars} chars"
            )
        if len(tests) > self.config.tools.jit_tests_max_count:
            raise ValidationError(
                f"JIT tests exceed {self.config.tools.jit_tests_max_count} cases"
            )
        for index, test in enumerate(tests, start=1):
            ensure_json_size(
                test,
                self.config.tools.jit_test_case_max_bytes,
                f"JIT test {index}",
            )

    def _validate_tool_spec(self, spec: ToolSpec) -> None:
        if not is_openai_tool_name(spec.name):
            raise ValidationError(
                "JIT tool name must match OpenAI tool name syntax "
                f"[A-Za-z0-9_-]{{1,{OPENAI_TOOL_NAME_MAX_CHARS}}}: {spec.name!r}"
            )
        if not isinstance(spec.description, str) or not spec.description.strip():
            raise ValidationError("JIT tool description must be a non-empty string")
        self._validate_json_schema(
            spec.input_schema or {"type": "object"},
            "input_schema",
        )
        self._validate_json_schema(
            spec.output_schema or {"type": "object"},
            "output_schema",
        )
        timeout = spec.policy.get("sandbox_timeout_s")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ValidationError("JIT tool sandbox_timeout_s must be a number")
            if isinstance(timeout, int):
                if timeout <= 0:
                    raise ValidationError(
                        "JIT tool sandbox_timeout_s must be finite and > 0"
                    )
                if timeout > self.config.tools.deno_timeout_hard_limit_s:
                    raise ValidationError(
                        "JIT tool sandbox_timeout_s exceeds tools.deno_timeout_hard_limit_s="
                        f"{self.config.tools.deno_timeout_hard_limit_s}"
                    )
            selected_timeout = float(timeout)
            if not math.isfinite(selected_timeout) or selected_timeout <= 0:
                raise ValidationError("JIT tool sandbox_timeout_s must be finite and > 0")
            if selected_timeout > self.config.tools.deno_timeout_hard_limit_s:
                raise ValidationError(
                    "JIT tool sandbox_timeout_s exceeds tools.deno_timeout_hard_limit_s="
                    f"{self.config.tools.deno_timeout_hard_limit_s}"
                )

    def _validate_json_schema(self, schema: dict[str, Any], field: str) -> None:
        if not isinstance(schema, dict):
            raise ValidationError(f"JIT tool {field} must be a JSON schema object")
        ensure_json_size(
            schema,
            self.config.tools.jit_test_case_max_bytes,
            f"JIT tool {field}",
        )
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError(
                f"JIT tool {field} is not a valid JSON schema: {exc.message}"
            ) from exc

    def _subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        if self.resources is None:
            return None
        wall = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        cpu = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_cpu_seconds",
            "subprocess_cpu_seconds",
        )
        memory = self.resources.peak_limit(pid, "max_subprocess_memory_bytes")
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(
            wall_seconds=wall,
            cpu_seconds=cpu,
            memory_bytes=memory,
        )

    def _charge_subprocess_metrics(
        self,
        pid: str,
        metrics: CommandMetrics | None,
        *,
        source: str,
        context: dict[str, Any],
    ) -> None:
        if self.resources is None or metrics is None:
            return
        self.resources.charge(
            pid,
            ResourceUsage(
                subprocess_wall_seconds=max(0.0, metrics.wall_seconds),
                subprocess_cpu_seconds=max(0.0, metrics.cpu_seconds),
                subprocess_peak_memory_bytes=max(0, metrics.peak_memory_bytes),
            ),
            source=source,
            context={**context, "metrics": self._metrics_json(metrics)},
            allow_overage=True,
            kill_on_exceed=True,
        )

    @staticmethod
    def _metrics_json(metrics: CommandMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    @staticmethod
    def _metrics_from_validation(value: Any) -> CommandMetrics | None:
        if not isinstance(value, dict):
            return None
        return CommandMetrics(
            wall_seconds=float(value.get("wall_seconds") or 0.0),
            cpu_seconds=float(value.get("cpu_seconds") or 0.0),
            peak_memory_bytes=int(value.get("peak_memory_bytes") or 0),
            killed=bool(value.get("killed", False)),
            limit_kind=(
                str(value["limit_kind"])
                if value.get("limit_kind") is not None
                else None
            ),
        )

    @staticmethod
    def _aggregate_command_metrics(
        metrics: list[CommandMetrics],
    ) -> dict[str, Any]:
        if not metrics:
            return {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_memory_bytes": 0,
                "killed": False,
                "limit_kind": None,
            }
        return {
            "wall_seconds": sum(item.wall_seconds for item in metrics),
            "cpu_seconds": sum(item.cpu_seconds for item in metrics),
            "peak_memory_bytes": max(item.peak_memory_bytes for item in metrics),
            "killed": any(item.killed for item in metrics),
            "limit_kind": next(
                (item.limit_kind for item in metrics if item.limit_kind),
                None,
            ),
        }

    def _bounded_validation_logs(self, logs: list[str]) -> str:
        text = "\n".join(logs)
        if len(text) <= self.config.tools.jit_validation_log_max_chars:
            return text
        digest = hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()
        return (
            text[: self.config.tools.jit_validation_log_max_chars]
            + f"\n[validation logs truncated chars={len(text)} sha256={digest}]"
        )

    def _validation_observation(
        self,
        validation: ValidationResult,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ok": validation.ok,
            "errors": [self._error_observation(error) for error in validation.errors],
            "warnings": [self._error_observation(item) for item in validation.warnings],
            "logs": self._error_observation(validation.logs),
            "metrics": validation.metadata.get("metrics"),
            **metadata,
        }

    def _error_observation(self, text: str) -> dict[str, Any]:
        return sanitize_for_observability(
            text,
            preview_chars=self.config.tools.tool_observability_preview_chars,
        )


def conservative_jit_tool_spec(
    spec: ToolSpec,
    *,
    declared_permissions: Iterable[str],
) -> ToolSpec:
    policy = dict(spec.policy)
    permissions = _string_set(policy.get("declared_permissions")) | set(
        declared_permissions
    )
    policy["side_effects"] = True
    policy["idempotent"] = False
    policy["declared_permissions"] = sorted(permissions)
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        input_schema=dict(spec.input_schema),
        output_schema=dict(spec.output_schema),
        policy=policy,
        tags=list(dict.fromkeys([*spec.tags, "jit", "side_effect"])),
        metadata=dict(spec.metadata),
        required_capabilities=[dict(item) for item in spec.required_capabilities],
        side_effects=sorted(set(spec.side_effects) | permissions),
    )


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {str(value)}


__all__ = ["JITToolService", "conservative_jit_tool_spec"]
