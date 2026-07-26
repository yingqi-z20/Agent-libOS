from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.llm.context_management import context_management_policy
from agent_libos.llm.event_projection import project_prompt_events
from agent_libos.memory.object_memory import ObjectVersionConflict
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.utils.ids import estimate_tokens, new_id, utc_now
from agent_libos.models import (
    AgentImage,
    AgentObject,
    AgentProcess,
    Capability,
    CapabilityRight,
    ContextMaterializationManifest,
    DataFlowContext,
    DataLabels,
    Event,
    EventType,
    MaterializedContext,
    MemoryView,
    ObjectHandle,
    ObjectMetadata,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    ResourceUsage,
    ViewMode,
    process_outcome_to_mapping,
    process_wait_state_to_mapping,
)
from agent_libos.memory.data_labels import (
    labels_for_explain,
    metadata_from_labels,
    propagate_object_labels,
)
from agent_libos.ports import OperationPort, ResourcePort
from agent_libos.storage import EvidenceRepository, ObjectRepository, ProcessRepository
from agent_libos.tools.observability import json_size_bytes

if TYPE_CHECKING:
    from agent_libos.memory.object_memory import ObjectMemoryManager

_LLM_CONTEXT_DEFAULTS = DEFAULT_CONFIG.llm_context
LLM_CONTEXT_ENRICHMENT_RESOURCE = "context:enrichment"
LLM_CONTEXT_MAINTENANCE_RESOURCE = "context:maintenance"
LLM_CONTEXT_OBJECT_POLICY = "llm_context_object"
_EVIDENCE_TIMESTAMP_ENTRY_KINDS = frozenset(
    {
        "process_started",
        "process_snapshot",
        "process_delta",
        "capabilities_snapshot",
        "capabilities_delta",
        "tool_table_snapshot",
        "tool_table_delta",
        "events_delta",
        "memory_delta",
        "context_omissions",
        "context_compacted",
    }
)


class LLMContextStoragePressure(Exception):
    """Ask the executor to compact before Object Memory reaches its hard limit."""

    def __init__(
        self,
        *,
        context_oid: str,
        context_version: int,
        context_generation: str,
        payload_bytes: int,
        persisted_payload_bytes: int,
        threshold_bytes: int,
        effective_threshold_bytes: int,
        hard_limit_bytes: int,
        entry_count: int,
        compaction_baseline_bytes: int | None = None,
        rearm_at_bytes: int | None = None,
    ) -> None:
        self.context_oid = context_oid
        self.context_version = context_version
        self.context_generation = context_generation
        self.payload_bytes = payload_bytes
        self.persisted_payload_bytes = persisted_payload_bytes
        self.threshold_bytes = threshold_bytes
        self.effective_threshold_bytes = effective_threshold_bytes
        self.hard_limit_bytes = hard_limit_bytes
        self.entry_count = entry_count
        self.compaction_baseline_bytes = compaction_baseline_bytes
        self.rearm_at_bytes = rearm_at_bytes
        super().__init__(
            "LLM context storage pressure requires compaction: "
            f"payload_bytes={payload_bytes}, threshold_bytes={threshold_bytes}, "
            f"hard_limit_bytes={hard_limit_bytes}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trigger": "storage_payload",
            "triggered": True,
            "context_oid": self.context_oid,
            "context_version": self.context_version,
            "context_generation": self.context_generation,
            "payload_bytes": self.payload_bytes,
            "persisted_payload_bytes": self.persisted_payload_bytes,
            "projected_payload_bytes": self.payload_bytes,
            "threshold_bytes": self.threshold_bytes,
            "effective_threshold_bytes": self.effective_threshold_bytes,
            "hard_limit_bytes": self.hard_limit_bytes,
            "entry_count": self.entry_count,
            "compaction_baseline_bytes": self.compaction_baseline_bytes,
            "rearm_at_bytes": self.rearm_at_bytes,
        }


class LLMContextMemory:
    """Materialize source context and optionally maintain an enriched context object."""

    def __init__(
        self,
        processes: ProcessRepository,
        objects: ObjectRepository,
        evidence: EvidenceRepository,
        memory: "ObjectMemoryManager",
        capabilities: CapabilityManager,
        operations: OperationPort,
        resources: ResourcePort | None,
        *,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self._processes = processes
        self._objects = objects
        self._evidence = evidence
        self._memory = memory
        self._capabilities = capabilities
        self._operations = operations
        self._resources = resources
        self._config = config or DEFAULT_CONFIG

    def object_name(self, pid: str) -> str:
        return context_object_name(pid, config=self._config)

    def prepare(
        self,
        pid: str,
        image: AgentImage,
        process: AgentProcess,
        source_context: MaterializedContext,
        events: list[Event],
        capabilities: list[Capability],
        tools: list[dict[str, Any]],
        *,
        label_events: list[Event] | None = None,
    ) -> MaterializedContext:
        trusted_label_events = events if label_events is None else label_events
        if not self._require_persistent_context_effect(pid):
            return self._record_source_only_context(pid, process, source_context)
        handle = self.ensure(pid, image, process, tools)
        obj = self._memory.get_object(pid, handle)
        payload = self._payload(obj)
        persisted_payload_bytes = json_size_bytes(payload)
        changed = self._append_deltas(
            payload=payload,
            process=process,
            image=image,
            source_context=source_context,
            events=events,
            capabilities=capabilities,
            tools=tools,
        )
        storage_metadata_changed = self._check_storage_pressure(
            pid,
            image,
            obj,
            payload,
            persisted_payload_bytes=persisted_payload_bytes,
        )
        changed = storage_metadata_changed or changed
        metadata = self._context_metadata(
            pid=pid,
            title=f"LLM context for {pid}",
            summary="Append-only process prompt context optimized for prompt caching.",
            tags=["llm_context", "prompt_cache"],
            token_estimate=estimate_tokens(payload),
            historical=obj.metadata,
            source_context=source_context,
            events=trusted_label_events,
        )
        metadata = self._persist_context_label_history(pid, metadata)
        label_history = labels_for_explain(metadata)
        if payload.get("label_history") != label_history:
            payload["label_history"] = label_history
            changed = True
        if changed:
            metadata.token_estimate = estimate_tokens(payload)
            self._memory.update_object(
                pid,
                handle,
                ObjectPatch(payload=payload, metadata=metadata),
                _trusted_label_propagation=True,
            )
            obj = self._memory.get_object(pid, handle)
        rendered = self.render(obj.payload)
        token_count = estimate_tokens(rendered)
        self._charge_rendered_context(pid, process, obj.oid, token_count)
        materialization_id = source_context.materialization_id or new_id("ctxmat")
        cache_strategy = obj.payload.get("cache_strategy", {}) if isinstance(obj.payload, dict) else {}
        transform = "compacted" if str(cache_strategy.get("mode") or "").startswith("compacted") else "verbatim"
        object_manifest = [
            *source_context.object_manifest,
            {
                "oid": obj.oid,
                "version": obj.version,
                "type": obj.type.value,
                "disposition": "included",
                "reason": "llm_context",
                "transform": transform,
                "tokens": token_count,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "labels": labels_for_explain(obj.metadata),
            },
        ]
        manifest = ContextMaterializationManifest(
            materialization_id=materialization_id,
            pid=pid,
            view_id=source_context.view_id or (process.memory_view.view_id if process.memory_view is not None else ""),
            policy=source_context.policy_used,
            budget_tokens=int(source_context.budget_tokens or process.resource_budget.max_context_materialization_tokens),
            rendered_tokens=token_count,
            rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            context_generation=self._processes.get_llm_context_generation(pid),
            context_oid=obj.oid,
            context_version=obj.version,
            objects=object_manifest,
            compaction={
                "mode": cache_strategy.get("mode"),
                "compacted_at": cache_strategy.get("compacted_at"),
                "transform": transform,
            },
            created_at=utc_now(),
        )
        self._evidence.insert_context_materialization_manifest(manifest)
        self._operations.link_evidence(
            "context_manifest",
            manifest.materialization_id,
            "context",
            metadata={"rendered_tokens": token_count, "object_count": len(object_manifest)},
        )
        self._operations.expect("context")
        return MaterializedContext(
            text=rendered,
            object_refs=[obj.oid, *source_context.object_refs],
            token_count=token_count,
            omitted_objects=source_context.omitted_objects,
            policy_used=LLM_CONTEXT_OBJECT_POLICY,
            materialization_id=materialization_id,
            view_id=source_context.view_id,
            budget_tokens=source_context.budget_tokens,
            object_manifest=object_manifest,
        )

    def _record_source_only_context(
        self,
        pid: str,
        process: AgentProcess,
        source_context: MaterializedContext,
    ) -> MaterializedContext:
        """Record evidence for the selected context without changing its content."""

        self._charge_rendered_context(
            pid,
            process,
            None,
            source_context.token_count,
            policy="source_only",
        )
        materialization_id = source_context.materialization_id or new_id("ctxmat")
        manifest = ContextMaterializationManifest(
            materialization_id=materialization_id,
            pid=pid,
            view_id=source_context.view_id
            or (process.memory_view.view_id if process.memory_view is not None else ""),
            policy=source_context.policy_used,
            budget_tokens=int(
                source_context.budget_tokens
                or process.resource_budget.max_context_materialization_tokens
            ),
            rendered_tokens=source_context.token_count,
            rendered_sha256=hashlib.sha256(
                source_context.text.encode("utf-8")
            ).hexdigest(),
            context_generation=self._processes.get_llm_context_generation(pid),
            context_oid=None,
            context_version=None,
            objects=list(source_context.object_manifest),
            compaction={
                "mode": "source_only",
                "compacted_at": None,
                "transform": "verbatim",
            },
            created_at=utc_now(),
        )
        self._evidence.insert_context_materialization_manifest(manifest)
        self._operations.link_evidence(
            "context_manifest",
            manifest.materialization_id,
            "context",
            metadata={
                "rendered_tokens": source_context.token_count,
                "object_count": len(source_context.object_manifest),
                "source_only": True,
            },
        )
        self._operations.expect("context")
        return replace(source_context, materialization_id=materialization_id)

    def _check_storage_pressure(
        self,
        pid: str,
        image: AgentImage,
        obj: AgentObject,
        payload: dict[str, Any],
        *,
        persisted_payload_bytes: int,
    ) -> bool:
        threshold = self._config.llm_context.storage_compaction_threshold_bytes
        payload_bytes = json_size_bytes(payload)
        if context_management_policy(image.planner).mode != "auto_compact":
            return False
        if not self._capabilities.check(
            pid,
            LLM_CONTEXT_MAINTENANCE_RESOURCE,
            CapabilityRight.EXECUTE,
        ):
            return False
        hard_limit = self._config.tools.memory_payload_hard_limit_bytes
        cache_strategy = self._storage_cache_strategy(payload)
        compacted_mode = str(cache_strategy.get("mode") or "").startswith("compacted")
        baseline = self._storage_watermark(
            cache_strategy,
            "storage_compaction_baseline_bytes",
        )
        rearm_at = self._storage_watermark(
            cache_strategy,
            "storage_compaction_rearm_at_bytes",
        )
        if self._storage_compaction_needs_rearm(
            cache_strategy,
            compacted_mode=compacted_mode,
            baseline=baseline,
            rearm_at=rearm_at,
        ):
            # Rearming changes the durable maintenance state. Consume finite
            # authority only at this committed effect boundary, not while
            # evaluating the feature predicate above.
            self._capabilities.require(
                pid,
                LLM_CONTEXT_MAINTENANCE_RESOURCE,
                CapabilityRight.EXECUTE,
            )
            self._arm_storage_compaction(
                payload,
                cache_strategy,
                payload_bytes=payload_bytes,
                threshold=threshold,
                hard_limit=hard_limit,
            )
            return True
        effective_threshold = (
            max(threshold, rearm_at)
            if compacted_mode and rearm_at is not None
            else threshold
        )
        if payload_bytes < effective_threshold:
            return False
        entries = payload.get("entries")
        raise LLMContextStoragePressure(
            context_oid=obj.oid,
            context_version=obj.version,
            context_generation=self._processes.get_llm_context_generation(pid),
            payload_bytes=payload_bytes,
            persisted_payload_bytes=persisted_payload_bytes,
            threshold_bytes=threshold,
            effective_threshold_bytes=effective_threshold,
            hard_limit_bytes=hard_limit,
            entry_count=len(entries) if isinstance(entries, list) else 0,
            compaction_baseline_bytes=baseline,
            rearm_at_bytes=rearm_at,
        )

    @staticmethod
    def _storage_cache_strategy(payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("cache_strategy")
        if raw is None:
            cache_strategy: dict[str, Any] = {}
            payload["cache_strategy"] = cache_strategy
            return cache_strategy
        if not isinstance(raw, dict):
            raise ValidationError("LLM context cache_strategy must be a JSON object")
        return raw

    @staticmethod
    def _storage_watermark(
        cache_strategy: dict[str, Any],
        field: str,
    ) -> int | None:
        value = cache_strategy.get(field)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _storage_compaction_needs_rearm(
        cache_strategy: dict[str, Any],
        *,
        compacted_mode: bool,
        baseline: int | None,
        rearm_at: int | None,
    ) -> bool:
        if not compacted_mode:
            return False
        return (
            cache_strategy.get("storage_compaction_rearm_pending") is True
            or baseline is None
            or rearm_at is None
        )

    @staticmethod
    def _arm_storage_compaction(
        payload: dict[str, Any],
        cache_strategy: dict[str, Any],
        *,
        payload_bytes: int,
        threshold: int,
        hard_limit: int,
    ) -> None:
        if payload_bytes >= hard_limit:
            raise ResourceLimitExceeded(
                "post-compaction LLM context deltas cannot fit within the "
                f"Object Memory hard limit: projected_bytes={payload_bytes}, "
                f"hard_limit_bytes={hard_limit}"
            )
        cache_strategy["storage_compaction_rearm_pending"] = False
        baseline = payload_bytes
        # The byte-width of the metadata values can change the payload size.
        # Iterate to a small fixed point so the durable baseline describes the
        # actual post-arm artifact rather than an approximation.
        for _ in range(4):
            cache_strategy["storage_compaction_baseline_bytes"] = baseline
            cache_strategy["storage_compaction_rearm_at_bytes"] = min(
                hard_limit - 1,
                baseline + threshold,
            )
            measured = json_size_bytes(payload)
            if measured == baseline:
                return
            baseline = measured

    def persistent_context_enabled(self, pid: str) -> bool:
        """Return whether the Host explicitly enabled persistent enrichment."""

        return (
            self._config.llm_context.policy == LLM_CONTEXT_OBJECT_POLICY
            or self._capabilities.check(
                pid,
                LLM_CONTEXT_ENRICHMENT_RESOURCE,
                CapabilityRight.EXECUTE,
            )
        )

    def _require_persistent_context_effect(self, pid: str) -> bool:
        """Authorize one persistent-context materialization effect.

        The read-only predicate intentionally uses ``check`` so callers can
        plan without spending a finite lease. The actual prepare path consumes
        the lease through ``require`` immediately before it starts mutating the
        persistent context object.
        """

        if self._config.llm_context.policy == LLM_CONTEXT_OBJECT_POLICY:
            return True
        if not self._capabilities.check(
            pid,
            LLM_CONTEXT_ENRICHMENT_RESOURCE,
            CapabilityRight.EXECUTE,
        ):
            return False
        try:
            self._capabilities.require(
                pid,
                LLM_CONTEXT_ENRICHMENT_RESOURCE,
                CapabilityRight.EXECUTE,
            )
        except CapabilityDenied:
            return False
        return True

    def _charge_rendered_context(
        self,
        pid: str,
        process: AgentProcess,
        context_oid: str | None,
        token_count: int,
        *,
        policy: str = LLM_CONTEXT_OBJECT_POLICY,
    ) -> None:
        resources = self._resources
        if resources is None:
            return
        window_limit = resources.context_materialization_window_limit(pid)
        if token_count > window_limit:
            raise ResourceLimitExceeded(
                "llm_context materialization tokens="
                f"{token_count} exceeds max_context_materialization_tokens={window_limit}"
            )
        resources.charge(
            pid,
            ResourceUsage(context_materialized_tokens=token_count),
            source="llm.context_memory",
            context={
                "view_id": process.memory_view.view_id if process.memory_view is not None else None,
                "object_oid": context_oid,
                "policy": policy,
            },
            allow_overage=False,
            kill_on_exceed=False,
        )

    def ensure(self, pid: str, image: AgentImage, process: AgentProcess, tools: list[dict[str, Any]]) -> ObjectHandle:
        name = self.object_name(pid)
        namespace = self._memory.resolve_namespace(pid)
        existing = self._objects.get_object_by_name(name, namespace=namespace)
        rights = {
            ObjectRight.READ.value,
            ObjectRight.WRITE.value,
            ObjectRight.MATERIALIZE.value,
            ObjectRight.LINK.value,
            ObjectRight.DIFF.value,
        }
        if existing is None:
            payload = self._initial_payload(pid, image, process, tools)
            metadata = ObjectMetadata(
                title=f"LLM context for {pid}",
                summary="Append-only process prompt context optimized for prompt caching.",
                tags=["llm_context", "prompt_cache"],
            )
            durable_metadata = self._durable_context_label_metadata(pid)
            if durable_metadata is not None:
                metadata = propagate_object_labels(metadata, [durable_metadata])
                payload["label_history"] = labels_for_explain(metadata)
                metadata.token_estimate = estimate_tokens(payload)
            handle = self._memory.create_object(
                pid=pid,
                object_type=ObjectType.PROCESS_STATE,
                payload=payload,
                metadata=metadata,
                immutable=False,
                name=name,
            )
        else:
            handle = self._capabilities.handle_for_object(
                pid,
                existing.oid,
                rights,
                issued_by="llm.context",
            )
        self._add_handle_to_view(pid, handle)
        return handle

    def view_without_context(self, pid: str, view: MemoryView) -> MemoryView:
        context_oid = self._context_oid(pid)
        if context_oid is None:
            roots = list(view.roots)
        else:
            roots = [handle for handle in view.roots if handle.oid != context_oid]
        return replace(view, roots=roots)

    def render(self, payload: dict[str, Any]) -> str:
        static_prefix = payload.get("static_prefix", {})
        lines = [
            "LLM context object:",
            "Cache strategy: append_only_stable_prefix",
            "The static prefix below should remain stable. New process facts are appended after it.",
            "",
            "Static prefix:",
            _stable_json(static_prefix),
            "",
            "Append-only entries:",
        ]
        for entry in payload.get("entries", []):
            lines.append("")
            lines.append("---")
            lines.append(_stable_json(_model_facing_entry(entry)))
        return "\n".join(lines).rstrip()

    def replace_with_compacted_summary(
        self,
        pid: str,
        *,
        context_oid: str,
        expected_version: int,
        summary: dict[str, Any],
        compaction_method: str,
        compaction_metadata: dict[str, Any] | None = None,
        preserve_recent_entries: int,
        source_tokens: int,
        target_tokens: int,
        compressor_pids: list[str],
        source_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace LLM context entries with a validated compact summary."""
        obj = self._objects.get_object(context_oid)
        if obj is None and source_payload is None:
            raise ValidationError(f"LLM context object not found: {context_oid}")
        if obj is not None and obj.version != expected_version:
            raise ValidationError(
                "LLM context changed during compaction: "
                f"expected version {expected_version}, found {obj.version}"
            )
        if obj is None:
            current = self._objects.get_object_by_name(
                self.object_name(pid),
                namespace=self._memory.resolve_namespace(pid),
            )
            if current is not None:
                raise ValidationError(
                    "LLM context changed during compaction: "
                    f"expected missing oid {context_oid}, found {current.oid} version {current.version}"
                )
            payload = self._payload_dict(source_payload)
        else:
            payload = self._payload(obj)
        compacted_payload, compacted_tokens, preserved_count = self._build_compacted_payload(
            context_oid=context_oid,
            expected_version=expected_version,
            payload=payload,
            summary=summary,
            compaction_method=compaction_method,
            compaction_metadata=compaction_metadata,
            preserve_recent_entries=preserve_recent_entries,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            compressor_pids=compressor_pids,
        )
        # Advance the durable provider-chain epoch before replacing volatile
        # context payload state. If the replacement is interrupted, the next
        # Responses request resets stateless rather than continuing against a
        # context generation whose exact contents are uncertain.
        self._processes.set_llm_context_generation(
            pid,
            str(compacted_payload["cache_strategy"]["compacted_at"]),
        )
        metadata = ObjectMetadata(
            title=f"LLM context for {pid}",
            summary="Compacted process prompt context optimized for bounded long-running sessions.",
            tags=[
                "llm_context",
                "prompt_cache",
                "compacted",
                f"compaction_method:{compaction_method}",
                f"compacted_at:{compacted_payload['cache_strategy']['compacted_at']}",
            ],
            token_estimate=estimate_tokens(compacted_payload),
        )
        historical_metadata = (
            obj.metadata
            if obj is not None
            else metadata_from_labels(compacted_payload.get("label_history"))
        )
        if historical_metadata is None:
            # A payload-only recovery cannot prove the original classification.
            # Preserve confidentiality by choosing the top/lowest labels instead
            # of silently recreating a normal, unknown-integrity context.
            historical_metadata = ObjectMetadata(
                sensitivity="secret",
                trust_level="untrusted",
                integrity="untrusted",
                origin="derived",
                tenant="mixed",
                principal="mixed",
            )
        historical_sources = [historical_metadata]
        durable_metadata = self._durable_context_label_metadata(pid)
        if durable_metadata is not None:
            historical_sources.append(durable_metadata)
        metadata = propagate_object_labels(metadata, historical_sources)
        metadata = self._persist_context_label_history(pid, metadata)
        compacted_payload["label_history"] = labels_for_explain(metadata)
        metadata.token_estimate = estimate_tokens(compacted_payload)
        if obj is None:
            handle = self._memory.create_object(
                pid=pid,
                object_type=ObjectType.PROCESS_STATE,
                payload=compacted_payload,
                metadata=metadata,
                immutable=False,
                name=self.object_name(pid),
            )
            updated_obj = self._memory.get_object(pid, handle)
        else:
            handle = self._memory.handle_for_oid(
                pid,
                context_oid,
                required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
                optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
                issued_by="llm.context.compact",
            )
            try:
                updated = self._memory.update_object(
                    pid,
                    handle,
                    ObjectPatch(payload=compacted_payload, metadata=metadata),
                    expected_version=expected_version,
                    _trusted_label_propagation=True,
                )
            except ObjectVersionConflict as exc:
                raise ValidationError(
                    "LLM context changed during compaction: "
                    f"expected version {exc.expected_version}, found {exc.actual_version}"
                ) from exc
            updated_obj = self._memory.get_object(pid, updated)
        view_handle = self._capabilities.handle_for_object(
            pid,
            updated_obj.oid,
            {
                ObjectRight.READ.value,
                ObjectRight.WRITE.value,
                ObjectRight.MATERIALIZE.value,
                ObjectRight.LINK.value,
                ObjectRight.DIFF.value,
            },
            issued_by="llm.context.compact",
        )
        self._add_handle_to_view(pid, view_handle)
        return {
            "context_oid": updated_obj.oid,
            "old_version": expected_version,
            "new_version": updated_obj.version,
            "source_tokens": source_tokens,
            "compacted_tokens": compacted_tokens,
            "preserved_recent_entries": preserved_count,
        }

    def _build_compacted_payload(
        self,
        *,
        context_oid: str,
        expected_version: int,
        payload: dict[str, Any],
        summary: dict[str, Any],
        compaction_method: str,
        compaction_metadata: dict[str, Any] | None,
        preserve_recent_entries: int,
        source_tokens: int,
        target_tokens: int,
        compressor_pids: list[str],
    ) -> tuple[dict[str, Any], int, int]:
        compact_summary = self._validate_compact_summary(summary)
        selected_method = self._validate_compaction_method(compaction_method)
        selected_metadata = self._validate_compaction_metadata(compaction_metadata)
        entries = list(payload.get("entries", []))
        preserved_count = max(0, min(int(preserve_recent_entries), len(entries)))
        preserved_entries = deepcopy(entries[-preserved_count:]) if preserved_count else []
        compacted_payload = deepcopy(payload)
        compact_entry = {
            "kind": "context_compacted",
            "at": utc_now(),
            "source_oid": context_oid,
            "source_version": expected_version,
            "source_entry_count": len(entries),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "compaction_method": selected_method,
            "compaction_metadata": selected_metadata,
            "compressor_pids": list(compressor_pids),
            "summary": compact_summary,
            "preserved_recent_entries": preserved_count,
        }
        compacted_payload["entries"] = [compact_entry, *preserved_entries]
        captured = dict(compacted_payload.get("captured") or {})
        captured["process_signature"] = None
        captured["capability_signature"] = None
        captured["tool_signature"] = None
        compacted_payload["captured"] = captured
        compacted_payload["cache_strategy"] = {
            **dict(compacted_payload.get("cache_strategy") or {}),
            "mode": "compacted_stable_prefix",
            "reason": (
                "Older append-only entries were summarized by context compaction; "
                "recent entries remain verbatim."
            ),
            "compacted_at": compact_entry["at"],
            "storage_compaction_rearm_pending": True,
            "storage_compaction_baseline_bytes": None,
            "storage_compaction_rearm_at_bytes": None,
        }
        rendered = self.render(compacted_payload)
        compacted_tokens = estimate_tokens(rendered)
        compact_entry["compacted_tokens"] = compacted_tokens
        return compacted_payload, compacted_tokens, preserved_count

    def _validate_compaction_method(self, method: str) -> str:
        if not isinstance(method, str) or not method.strip():
            raise ValidationError("context compaction method must be a non-empty string")
        selected = method.strip()
        if len(selected) > 128:
            raise ValidationError("context compaction method is too long")
        return selected

    def _validate_compaction_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise ValidationError("context compaction metadata must be a JSON object")
        return deepcopy(metadata)

    def _validate_compact_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(summary, dict):
            raise ValidationError("context compaction summary must be a JSON object")
        required = {
            "goal",
            "constraints",
            "user_preferences",
            "completed",
            "pending",
            "key_references",
            "recent_decisions",
            "risks",
            "uncertainties",
            "next_steps",
        }
        missing = sorted(required - set(summary))
        if missing:
            raise ValidationError(f"context compaction summary missing fields: {missing}")
        if not any(summary.get(key) for key in required):
            raise ValidationError("context compaction summary is empty")
        return deepcopy(summary)

    def _initial_payload(
        self,
        pid: str,
        image: AgentImage,
        process: AgentProcess,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "kind": "llm_context",
            "schema_version": self._config.llm_context.schema_version,
            "cache_strategy": {
                "mode": "append_only_stable_prefix",
                "reason": "Keep repeated instructions, tool names, and early process context at the front; append changes at the end.",
            },
            "static_prefix": {
                "pid": pid,
                "image_id": image.image_id,
                "image_name": image.name,
                "image_version": image.version,
                "safety_profile": image.safety_profile,
                "context_policy": image.context_policy,
                "parent_pid": process.parent_pid,
                "initial_working_directory": process.working_directory,
                "goal_oid": process.goal_oid,
                "tool_names": sorted(_tool_name(tool) for tool in tools if _tool_name(tool)),
            },
            "entries": [
                {
                    "kind": "process_started",
                    "at": process.created_at,
                    "pid": pid,
                    "parent_pid": process.parent_pid,
                    "goal_oid": process.goal_oid,
                    "working_directory": process.working_directory,
                }
            ],
            "captured": {
                "object_oids": [],
                "objects": {},
                "event_ids": [],
                "capability_signature": None,
                "tool_signature": _tool_signature(tools),
                "process_signature": None,
            },
        }

    def _append_deltas(
        self,
        payload: dict[str, Any],
        process: AgentProcess,
        image: AgentImage,
        source_context: MaterializedContext,
        events: list[Event],
        capabilities: list[Capability],
        tools: list[dict[str, Any]],
    ) -> bool:
        changed = False
        captured = payload.setdefault("captured", {})
        entries = payload.setdefault("entries", [])

        changed = self._append_process_delta(captured, entries, process, image) or changed
        changed = self._append_signature_change(
            captured,
            entries,
            captured_key="capability_signature",
            current_signature=_capability_signature(capabilities),
            signature_key="cap_id",
            delta_kind="capabilities_delta",
            snapshot_kind="capabilities_snapshot",
            snapshot_field="capabilities",
            removed_field="removed_capability_ids",
        ) or changed
        changed = self._append_tool_delta(captured, entries, tools) or changed
        changed = self._append_event_delta(captured, entries, process, events) or changed
        changed = self._append_memory_delta(captured, entries, source_context) or changed
        return changed

    @staticmethod
    def _append_process_delta(
        captured: dict[str, Any],
        entries: list[dict[str, Any]],
        process: AgentProcess,
        image: AgentImage,
    ) -> bool:
        process_signature = {
            "status": process.status.value,
            "status_message": process.status_message,
            "wait_state": process_wait_state_to_mapping(process.wait_state),
            "outcome": process_outcome_to_mapping(process.outcome),
            "checkpoint_head": process.checkpoint_head,
            "image_id": image.image_id,
            "working_directory": process.working_directory,
        }
        previous_signature = captured.get("process_signature")
        if previous_signature == process_signature:
            return False
        if isinstance(previous_signature, dict):
            changed_fields = {
                key: deepcopy(value)
                for key, value in process_signature.items()
                if previous_signature.get(key) != value
            }
            if not changed_fields:
                # Older payloads can carry fencing-only fields such as
                # ``state_generation``.  Normalize the durable signature
                # without manufacturing a model-visible process change.
                captured["process_signature"] = process_signature
                return True
            entries.append(
                {
                    "kind": "process_delta",
                    "at": utc_now(),
                    "changed": changed_fields,
                }
            )
        else:
            entries.append({"kind": "process_snapshot", "at": utc_now(), **process_signature})
        captured["process_signature"] = process_signature
        return True

    @staticmethod
    def _append_tool_delta(
        captured: dict[str, Any],
        entries: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> bool:
        """Record only tool identities; native tool schemas travel out of band."""

        current_signature = _tool_signature(tools)
        previous_signature = captured.get("tool_signature")
        if previous_signature == current_signature:
            return False
        if isinstance(previous_signature, list):
            upserted, removed = _signature_delta(
                previous_signature,
                current_signature,
                key="name",
            )
            entry = {
                "kind": "tool_table_delta",
                "at": utc_now(),
                "upserted": upserted,
                "removed_tool_names": removed,
                "tool_count": len(current_signature),
            }
        else:
            entry = {
                "kind": "tool_table_snapshot",
                "at": utc_now(),
                "tools": current_signature,
                "tool_count": len(current_signature),
            }
        entries.append(entry)
        captured["tool_signature"] = current_signature
        return True

    @staticmethod
    def _append_signature_change(
        captured: dict[str, Any],
        entries: list[dict[str, Any]],
        *,
        captured_key: str,
        current_signature: list[dict[str, Any]],
        signature_key: str,
        delta_kind: str,
        snapshot_kind: str,
        snapshot_field: str,
        removed_field: str,
    ) -> bool:
        previous_signature = captured.get(captured_key)
        if previous_signature == current_signature:
            return False
        if isinstance(previous_signature, list):
            upserted, removed = _signature_delta(
                previous_signature,
                current_signature,
                key=signature_key,
            )
            entry = {
                "kind": delta_kind,
                "at": utc_now(),
                "upserted": upserted,
                removed_field: removed,
            }
        else:
            entry = {
                "kind": snapshot_kind,
                "at": utc_now(),
                snapshot_field: current_signature,
            }
        entries.append(entry)
        captured[captured_key] = current_signature
        return True

    def _append_event_delta(
        self,
        captured: dict[str, Any],
        entries: list[dict[str, Any]],
        process: AgentProcess,
        events: list[Event],
    ) -> bool:
        captured_events = set(captured.get("event_ids", []))
        # ``events`` is already the store-bounded post-cursor window selected
        # with the active Runtime configuration.  Applying the import-time
        # default here used to discard leading rows from a larger configured
        # window even though the durable cursor advanced past those rows.
        new_events = [event for event in events if event.event_id not in captured_events]
        if not new_events:
            return False
        projection = project_prompt_events(
            new_events,
            context_object_name=self.object_name(process.pid),
            payload_max_chars=self._config.llm_context.prompt_event_payload_max_chars,
        )
        event_entry: dict[str, Any] = {
            "kind": "events_delta",
            "at": utc_now(),
            "events": projection.visible_records,
            "projection_summary": projection.summary,
        }
        if projection.omitted_counts:
            event_entry["omitted_evidence_event_counts"] = projection.omitted_counts
        if projection.resource_usage_delta:
            event_entry["resource_usage_delta"] = projection.resource_usage_delta
        entries.append(event_entry)
        captured["event_ids"] = sorted(captured_events | {event.event_id for event in new_events})
        return True

    def _append_memory_delta(
        self,
        captured: dict[str, Any],
        entries: list[dict[str, Any]],
        source_context: MaterializedContext,
    ) -> bool:
        changed = False
        captured_objects = _captured_object_signatures(captured)
        changed_oids: list[str] = []
        next_captured_objects = dict(captured_objects)
        for oid in source_context.object_refs:
            signature = self._object_signature(oid)
            if captured_objects.get(oid) != signature:
                changed_oids.append(oid)
            next_captured_objects[oid] = signature
        if changed_oids:
            entries.append(
                {
                    "kind": "memory_delta",
                    "at": utc_now(),
                    "policy": source_context.policy_used,
                    "token_estimate": source_context.token_count,
                    "omitted_objects": list(source_context.omitted_objects),
                    "objects": [self._object_entry(oid) for oid in changed_oids],
                }
            )
            captured["objects"] = dict(sorted(next_captured_objects.items()))
            captured["object_oids"] = sorted(next_captured_objects)
            changed = True

        if source_context.omitted_objects:
            omitted = sorted(set(source_context.omitted_objects))
            if captured.get("omitted_objects") != omitted:
                entries.append(
                    {"kind": "context_omissions", "at": utc_now(), "omitted_objects": omitted}
                )
                captured["omitted_objects"] = omitted
                changed = True
        return changed

    def _context_metadata(
        self,
        *,
        pid: str,
        title: str,
        summary: str,
        tags: list[str],
        token_estimate: int,
        historical: ObjectMetadata,
        source_context: MaterializedContext,
        events: list[Event],
    ) -> ObjectMetadata:
        sources = [
            historical,
            *self._source_context_metadata(source_context),
            *self._event_metadata(events),
        ]
        durable_metadata = self._durable_context_label_metadata(pid)
        if durable_metadata is not None:
            sources.append(durable_metadata)
        return propagate_object_labels(
            ObjectMetadata(
                title=title,
                summary=summary,
                tags=tags,
                token_estimate=token_estimate,
            ),
            sources,
        )

    def include_event_labels(
        self,
        context: DataFlowContext,
        events: list[Event],
    ) -> DataFlowContext:
        """Merge trusted labels for every event represented in the prompt batch."""

        event_metadata = self._event_metadata(events)
        if not event_metadata:
            return context
        return replace(
            context,
            labels=DataLabels.aggregate(
                (
                    context.labels,
                    *(
                        DataLabels.from_object_metadata(metadata)
                        for metadata in event_metadata
                    ),
                )
            ),
        )

    @staticmethod
    def _event_metadata(events: list[Event]) -> list[ObjectMetadata]:
        """Recover trusted labels for event payloads copied into the prompt."""

        sources: list[ObjectMetadata] = []
        canonical_fields = frozenset(DataLabels().to_dict())
        for event in events:
            if "data_labels" not in event.payload:
                continue
            raw_labels = event.payload["data_labels"]
            if (
                not isinstance(raw_labels, Mapping)
                or frozenset(raw_labels) != canonical_fields
            ):
                raise ValidationError(
                    f"event {event.event_id} contains malformed trusted data_labels"
                )
            try:
                labels = metadata_from_labels(DataLabels.from_dict(raw_labels))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"event {event.event_id} contains malformed trusted data_labels"
                ) from exc
            if labels is None:
                raise ValidationError(
                    f"event {event.event_id} contains malformed trusted data_labels"
                )
            sources.append(labels)
        return sources

    def _durable_context_label_metadata(self, pid: str) -> ObjectMetadata | None:
        labels = self._processes.get_llm_context_label_history(pid)
        return metadata_from_labels(labels)

    def _persist_context_label_history(
        self,
        pid: str,
        metadata: ObjectMetadata,
    ) -> ObjectMetadata:
        labels = self._processes.merge_llm_context_label_history(
            pid,
            DataLabels.from_object_metadata(metadata),
        )
        durable_metadata = metadata_from_labels(labels)
        if durable_metadata is None:
            raise ValidationError("persisted LLM context label history is missing")
        return propagate_object_labels(metadata, [durable_metadata])

    def _source_context_metadata(self, source_context: MaterializedContext) -> list[ObjectMetadata]:
        """Recover immutable label evidence for every included prompt source."""

        sources: list[ObjectMetadata] = []
        manifested_oids: set[str] = set()
        for item in source_context.object_manifest:
            if not isinstance(item, dict) or item.get("disposition") != "included":
                continue
            oid = str(item.get("oid") or "")
            labels = metadata_from_labels(item.get("labels"))
            if not oid or labels is None:
                raise ValidationError("included LLM context source is missing trusted label evidence")
            manifested_oids.add(oid)
            sources.append(labels)

        for oid in dict.fromkeys(source_context.object_refs):
            obj = self._objects.get_object(oid)
            if obj is not None:
                # Also merge the live label in case classification increased
                # after materialization but before provider dispatch.
                sources.append(obj.metadata)
            elif oid not in manifested_oids:
                raise ValidationError(
                    f"included LLM context source has no recoverable labels: {oid}"
                )
        return sources

    def _object_entry(self, oid: str) -> dict[str, Any]:
        obj = self._objects.get_object(oid)
        if obj is None:
            return {"oid": oid, "missing": True}
        return {
            "oid": obj.oid,
            "name": obj.name,
            "type": obj.type.value,
            "version": obj.version,
            "title": obj.metadata.title,
            "summary": obj.metadata.summary,
            "payload": self._memory.prompt_payload(obj),
            "transform": self._memory.prompt_transform(obj),
        }

    def _object_signature(self, oid: str) -> dict[str, Any]:
        obj = self._objects.get_object(oid)
        if obj is None:
            return {"missing": True}
        return {"version": obj.version, "updated_at": obj.updated_at}

    def _payload(self, obj: AgentObject) -> dict[str, Any]:
        return self._payload_dict(obj.payload, label=obj.name)

    def _payload_dict(self, payload: Any, *, label: str = "payload") -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("kind") != "llm_context":
            raise ValidationError(f"object is not an LLM context object: {label}")
        expected_schema = self._config.llm_context.schema_version
        found_schema = payload.get("schema_version")
        if type(found_schema) is not int or found_schema != expected_schema:
            raise ValidationError(
                "LLM context object schema_version mismatch: "
                f"expected {expected_schema}, found {found_schema!r}"
            )
        return payload

    def _context_oid(self, pid: str) -> str | None:
        obj = self._objects.get_object_by_name(
            self.object_name(pid),
            namespace=self._memory.resolve_namespace(pid),
        )
        return obj.oid if obj is not None else None

    def _add_handle_to_view(self, pid: str, handle: ObjectHandle) -> None:
        # Child resource usage is charged hierarchically and therefore mutates
        # this process row even while its own quantum is running.  Serialize the
        # read/modify/CAS sequence with those accounting writes; the repository
        # still applies the ambient execution-token fence at ``patch_process``.
        with self._processes.locked():
            process = self._require_process(pid)
            if process.memory_view is None:
                process.memory_view = self._memory.create_view(pid, [handle], mode=ViewMode.MUTABLE)
            elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
                process.memory_view.roots.insert(0, handle)
            else:
                process.memory_view.roots = [
                    handle if existing.oid == handle.oid and "write" not in existing.rights else existing
                    for existing in process.memory_view.roots
                ]
            process.updated_at = utc_now()
            self._processes.patch_process(
                pid,
                {"memory_view": process.memory_view, "updated_at": process.updated_at},
                expected_revision=process.revision,
            )

    def _require_process(self, pid: str) -> AgentProcess:
        process = self._processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process


def context_object_name(
    pid: str,
    *,
    config: AgentLibOSConfig | None = None,
) -> str:
    selected = config.llm_context if config is not None else _LLM_CONTEXT_DEFAULTS
    return f"{selected.object_name_prefix}:{pid}"


def _tool_name(tool: dict[str, Any]) -> str | None:
    spec_json = tool.get("spec_json")
    if isinstance(spec_json, str):
        try:
            spec = json.loads(spec_json)
            if isinstance(spec, dict):
                return str(spec.get("name") or tool.get("name") or "")
        except json.JSONDecodeError:
            pass
    return str(tool.get("name") or "") or None


def _tool_signature(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        spec_json = tool.get("spec_json")
        spec = {}
        if isinstance(spec_json, str):
            try:
                decoded = json.loads(spec_json)
                if isinstance(decoded, dict):
                    spec = decoded
            except json.JSONDecodeError:
                spec = {}
        name = str(spec.get("name") or tool.get("name") or "")
        if not name:
            continue
        input_schema = spec.get("input_schema", {})
        schema_json = json.dumps(
            input_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        result.append(
            {
                "name": name,
                "version": str(spec.get("version") or ""),
                "schema_sha256": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
            }
        )
    return sorted(result, key=lambda item: item["name"])


def _capability_signature(capabilities: list[Capability]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "cap_id": cap.cap_id,
                "resource": cap.resource,
                "rights": sorted(cap.rights),
                "effect": cap.effect.value,
                "status": cap.status.value,
                "policy": _capability_policy(cap),
                "uses_remaining": cap.uses_remaining,
                "delegable": cap.delegable,
                "delegation_depth": cap.delegation_depth,
                "issuer": cap.issued_by,
                "parent_cap_id": cap.parent_cap_id,
                "expires_at": cap.expires_at,
            }
            for cap in capabilities
            if cap.active
        ],
        key=lambda item: (item["resource"], ",".join(item["rights"]), item["cap_id"]),
    )


def _capability_policy(cap: Capability) -> str:
    if cap.effect.value == "allow":
        return "allow_once" if cap.uses_remaining is not None else "always_allow"
    if cap.effect.value == "deny":
        return "always_deny"
    if cap.effect.value == "ask":
        return "ask_each_time"
    return cap.effect.value


def _signature_delta(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    key: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    previous_by_key = {
        str(item[key]): item
        for item in previous
        if isinstance(item, dict) and item.get(key) is not None
    }
    current_by_key = {
        str(item[key]): item
        for item in current
        if isinstance(item, dict) and item.get(key) is not None
    }
    upserted = [
        deepcopy(item)
        for item in current
        if item.get(key) is not None
        and previous_by_key.get(str(item[key])) != item
    ]
    removed = sorted(set(previous_by_key) - set(current_by_key))
    return upserted, removed


def _captured_object_signatures(captured: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = captured.get("objects")
    if isinstance(objects, dict):
        return {str(oid): dict(signature) for oid, signature in objects.items() if isinstance(signature, dict)}
    return {str(oid): {"legacy_captured": True} for oid in captured.get("object_oids", [])}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _model_facing_entry(entry: Any) -> Any:
    """Drop evidence-only envelope fields from an append-only prompt entry."""

    if not isinstance(entry, dict):
        return entry
    projected = deepcopy(entry)
    if projected.get("kind") in _EVIDENCE_TIMESTAMP_ENTRY_KINDS:
        projected.pop("at", None)
    return projected
