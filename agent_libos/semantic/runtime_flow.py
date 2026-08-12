from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent_libos.models import DataLabels, DataSensitivity
from agent_libos.models.data_flow import DataFlowContext
from agent_libos.models.memory import AgentObject, MaterializedContext, ObjectType
from agent_libos.models.semantic import SemanticDataFinding, SemanticDataLocator
from agent_libos.primitives.filesystem import (
    FileBytesReadResult,
    FileReadResult,
    FileWriteResult,
)
from agent_libos.sdk import PostCommitResultObservation
from agent_libos.models.git import GitDiffResult, GitStatusResult
from agent_libos.semantic.flow import (
    FLOW_NO_TENANT_BUCKET_SHA256,
    FLOW_UNBUCKETED_IDENTITY_SHA256,
    FlowActivityKind,
    FlowCoverageStatus,
    FlowEdgeRelation,
    FlowEntityKind,
    FlowInputEdge,
    FlowLabelSource,
    FlowNodeRef,
    FlowNodeType,
    FlowOutputEdge,
    SemanticFlowService,
    provider_result_entity_id,
    root_goal_entity_id,
)
from agent_libos.semantic.projection import LocalDlpAccumulator
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable


_MAX_OBJECT_INPUTS = 64
_MAX_PENDING_SELECTIONS = 128
_MAX_MODEL_TOOL_CALLS = 256
_MAX_TRACKED_OBJECT_VERSIONS = 4_096
_MAX_TRACKED_MATERIALIZATIONS = 1_024
_MAX_TRACKED_PROCESSES = 1_024
_MAX_PENDING_SELECTION_KEYS = 1_024


def _digest(value: Any) -> str:
    encoded = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _worse(
    left: FlowCoverageStatus,
    right: FlowCoverageStatus,
) -> FlowCoverageStatus:
    order = {
        FlowCoverageStatus.COMPLETE: 0,
        FlowCoverageStatus.PARTIAL: 1,
        FlowCoverageStatus.UNKNOWN: 2,
        FlowCoverageStatus.STALE: 3,
        FlowCoverageStatus.CONFLICT: 4,
    }
    return left if order[left] >= order[right] else right


def _lru_get(mapping: OrderedDict[Any, Any], key: Any) -> Any | None:
    value = mapping.get(key)
    if value is not None:
        mapping.move_to_end(key)
    return value


def _lru_put(
    mapping: OrderedDict[Any, Any],
    key: Any,
    value: Any,
    *,
    limit: int,
) -> None:
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > limit:
        mapping.popitem(last=False)


@dataclass(frozen=True, slots=True)
class _CapturedObject:
    ref: FlowNodeRef
    labels: DataLabels
    tenant_bucket_sha256: str
    coverage: FlowCoverageStatus


@dataclass(frozen=True, slots=True)
class _CurrentFileRead:
    relative: str
    content: bytes
    first_state: Any
    second_state: Any
    before_label_state: str
    after_context: DataFlowContext
    after_label_state: str
    binding: Any | None


@dataclass(frozen=True, slots=True)
class _ModelInputBinding:
    inputs: tuple[FlowInputEdge, ...]
    coverage: FlowCoverageStatus
    materialized: _CapturedObject | None
    provider_observation: PostCommitResultObservation | None
    provider_input: FlowInputEdge | None


class SemanticRuntimeFlowObserver:
    """Host-only bridge from committed Runtime artifacts to FlowGraph.

    The observer may inspect transient Host values to calculate digests, but it
    passes only typed labels, stable identifiers, and SHA-256 values to the
    append-only Flow service.  Every unresolved or ambiguous source lowers
    coverage; this bridge never promotes evidence to complete by inference.
    """

    def __init__(
        self,
        flow: SemanticFlowService,
        *,
        objects: Any,
        enabled: Callable[[], bool],
        tenant_bucketer: Callable[[str], str] | None = None,
        capture_failure: Callable[..., None] | None = None,
        filesystem: Any | None = None,
        data_flow: Any | None = None,
    ) -> None:
        if not isinstance(flow, SemanticFlowService):
            raise TypeError("runtime flow observer requires SemanticFlowService")
        if not callable(enabled):
            raise TypeError("runtime flow observer enabled predicate must be callable")
        self._flow = flow
        self._objects = objects
        self._enabled = enabled
        self._tenant_bucketer = tenant_bucketer
        self._capture_failure = capture_failure
        self._filesystem = filesystem
        self._data_flow = data_flow
        self._lock = threading.RLock()
        self._object_entities: OrderedDict[
            tuple[str, str, int], _CapturedObject
        ] = OrderedDict()
        self._materializations: OrderedDict[
            tuple[str, str], _CapturedObject
        ] = OrderedDict()
        self._latest_model_output: OrderedDict[
            str, _CapturedObject
        ] = OrderedDict()
        self._pending_selections: OrderedDict[
            tuple[str, str], deque[_CapturedObject]
        ] = OrderedDict()

    def observe_memory(self, kind: str, **facts: Any) -> None:
        if not self._enabled():
            return
        source = {
            "object_version": "runtime_object_version",
            "materialization": "runtime_materialization",
        }.get(kind, "runtime_observation")
        self._isolated(
            source,
            lambda: self._observe_memory(kind, facts),
        )

    def observe_model_output(
        self,
        state: Any,
        completion: Any,
        record: Any,
    ) -> None:
        if not self._enabled():
            return
        self._isolated(
            "runtime_model_output",
            lambda: self._capture_model_output(state, completion, record),
        )

    def observe_root_goal_binding(
        self,
        *,
        pid: str,
        goal: AgentObject,
        root_state_sha256: str,
    ) -> None:
        if not self._enabled():
            return
        self._isolated(
            "runtime_root_goal_binding",
            lambda: self._capture_root_goal_binding(
                pid=pid,
                goal=goal,
                root_state_sha256=root_state_sha256,
            ),
        )

    def observe_provider_result(self, result: Any, observation: Any) -> None:
        if not self._enabled():
            return
        self._isolated(
            "runtime_file_git_version",
            lambda: self._capture_provider_version(result, observation),
        )

    def cache_status(self) -> dict[str, int]:
        """Return bounded payload-free observer cache health."""

        with self._lock:
            return {
                "object_versions": len(self._object_entities),
                "object_version_limit": _MAX_TRACKED_OBJECT_VERSIONS,
                "materializations": len(self._materializations),
                "materialization_limit": _MAX_TRACKED_MATERIALIZATIONS,
                "model_processes": len(self._latest_model_output),
                "model_process_limit": _MAX_TRACKED_PROCESSES,
                "selection_keys": len(self._pending_selections),
                "selection_key_limit": _MAX_PENDING_SELECTION_KEYS,
            }

    def _isolated(self, source: str, callback: Callable[[], None]) -> None:
        before = self._flow.capture_failure_count
        try:
            callback()
        except Exception as exc:
            if (
                self._flow.capture_failure_count == before
                and self._capture_failure is not None
            ):
                try:
                    self._capture_failure(
                        source=source,
                        error_type=type(exc).__name__,
                    )
                except Exception:
                    pass

    def _observe_memory(self, kind: str, facts: Mapping[str, Any]) -> None:
        if kind == "object_version":
            value = facts.get("value")
            pid = facts.get("pid")
            operation = facts.get("operation")
            if not isinstance(value, AgentObject) or not isinstance(pid, str):
                raise TypeError("object flow observation is malformed")
            if operation not in {"create", "update", "append"}:
                raise ValueError("object flow operation is invalid")
            self._capture_object(pid, value, operation=operation)
            return
        if kind == "materialization":
            value = facts.get("value")
            pid = facts.get("pid")
            if not isinstance(value, MaterializedContext) or not isinstance(pid, str):
                raise TypeError("materialization flow observation is malformed")
            self._capture_materialization(pid, value)
            return
        raise ValueError("runtime flow observation kind is invalid")

    def _capture_object(
        self,
        pid: str,
        obj: AgentObject,
        *,
        operation: str,
    ) -> _CapturedObject:
        with self._lock:
            labels = DataLabels.from_object_metadata(obj.metadata)
            bucket = self._bucket(labels)
            inputs, coverage = self._object_inputs(pid, obj, bucket=bucket)
            content_sha256, version_sha256, state_sha256, provenance_sha256 = (
                self._object_digests(obj)
            )
            bundle = self._flow.capture_object_version(
                operation=operation,
                pid=pid,
                action_id=f"memory.object.{operation}",
                content_sha256=content_sha256,
                version_sha256=version_sha256,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                labels=labels,
                tenant_bucket_sha256=self._tenant_bucket_arg(labels),
                inputs=inputs,
                coverage=coverage,
                created_at=utc_now(),
            )
            if bundle is None or len(bundle.entities) != 1:
                raise RuntimeError("object FlowGraph capture returned no entity")
            captured = _CapturedObject(
                FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
                labels,
                bundle.entities[0].tenant_bucket_sha256,
                FlowCoverageStatus(bundle.entities[0].coverage),
            )
            self._capture_object_derivation(
                pid,
                captured,
                inputs=inputs,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
            )
            if obj.type is ObjectType.TOOL_RESULT:
                captured = self._capture_tool_result(
                    pid,
                    obj,
                    captured,
                    content_sha256=content_sha256,
                    version_sha256=version_sha256,
                    state_sha256=state_sha256,
                    provenance_sha256=provenance_sha256,
                )
            _lru_put(
                self._object_entities,
                self._object_key(pid, obj),
                captured,
                limit=_MAX_TRACKED_OBJECT_VERSIONS,
            )
            return captured

    def _capture_root_goal_binding(
        self,
        *,
        pid: str,
        goal: AgentObject,
        root_state_sha256: str,
    ) -> None:
        if goal.type is not ObjectType.GOAL:
            raise TypeError("root goal FlowGraph binding requires a Goal Object")
        with self._lock:
            captured = _lru_get(
                self._object_entities,
                self._object_key(pid, goal),
            )
            if captured is None:
                raise RuntimeError(
                    "initial root goal Object version was not captured"
                )
            content, version, _state, provenance = self._object_digests(goal)
            object_record = self._flow.get_entity(captured.ref.node_id)
            if (
                object_record is None
                or object_record.kind != FlowEntityKind.OBJECT_VERSION.value
                or object_record.pid != pid
                or object_record.content_sha256 != content
                or object_record.version_sha256 != version
                or object_record.provenance_sha256 != provenance
                or object_record.tenant_bucket_sha256
                != captured.tenant_bucket_sha256
            ):
                raise RuntimeError("initial root goal Object binding changed")
            root_entity = root_goal_entity_id(
                pid=pid,
                content_sha256=content,
                state_sha256=root_state_sha256,
                goal_oid=goal.oid,
                goal_version=goal.version,
            )
            self._flow.bind_root_goal_object(
                pid=pid,
                root_entity_id=root_entity,
                object_entity_id=captured.ref.node_id,
                root_state_sha256=root_state_sha256,
                object_content_sha256=content,
                object_version_sha256=version,
                object_provenance_sha256=provenance,
                tenant_bucket_sha256=captured.tenant_bucket_sha256,
            )

    def _object_inputs(
        self,
        pid: str,
        obj: AgentObject,
        *,
        bucket: str,
    ) -> tuple[tuple[FlowInputEdge, ...], FlowCoverageStatus]:
        refs: list[FlowInputEdge] = []
        coverage = FlowCoverageStatus.COMPLETE
        if obj.version > 1:
            prior = _lru_get(
                self._object_entities,
                (pid, _text_digest(obj.oid), obj.version - 1)
            )
            if prior is None:
                coverage = FlowCoverageStatus.PARTIAL
            elif prior.tenant_bucket_sha256 != bucket:
                coverage = FlowCoverageStatus.CONFLICT
            else:
                refs.append(FlowInputEdge(prior.ref, FlowEdgeRelation.DIRECT))
        parents = tuple(dict.fromkeys(obj.provenance.parent_oids))
        if len(parents) > _MAX_OBJECT_INPUTS:
            parents = parents[:_MAX_OBJECT_INPUTS]
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        for oid in parents:
            parent = self._objects.get_object(oid)
            if parent is None:
                coverage = _worse(coverage, FlowCoverageStatus.UNKNOWN)
                continue
            source = self._known_or_read_object(pid, parent)
            if source.tenant_bucket_sha256 != bucket:
                coverage = FlowCoverageStatus.CONFLICT
                continue
            refs.append(FlowInputEdge(source.ref, FlowEdgeRelation.DIRECT))
            coverage = _worse(coverage, source.coverage)
        if obj.provenance.source_refs:
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        return tuple(dict.fromkeys(refs)), coverage

    def _known_or_read_object(self, pid: str, obj: AgentObject) -> _CapturedObject:
        known = _lru_get(self._object_entities, self._object_key(pid, obj))
        if known is not None:
            return known
        labels = DataLabels.from_object_metadata(obj.metadata)
        content, version, state, provenance = self._object_digests(obj)
        bundle = self._flow.capture_object_version(
            operation="read",
            pid=pid,
            action_id="memory.object.read",
            content_sha256=content,
            version_sha256=version,
            state_sha256=state,
            provenance_sha256=provenance,
            labels=labels,
            tenant_bucket_sha256=self._tenant_bucket_arg(labels),
            inputs=(),
            coverage=FlowCoverageStatus.PARTIAL,
            created_at=utc_now(),
        )
        if bundle is None or len(bundle.entities) != 1:
            raise RuntimeError("object read FlowGraph capture returned no entity")
        captured = _CapturedObject(
            FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
            labels,
            bundle.entities[0].tenant_bucket_sha256,
            FlowCoverageStatus(bundle.entities[0].coverage),
        )
        _lru_put(
            self._object_entities,
            self._object_key(pid, obj),
            captured,
            limit=_MAX_TRACKED_OBJECT_VERSIONS,
        )
        return captured

    @staticmethod
    def _object_digests(obj: AgentObject) -> tuple[str, str, str, str]:
        content = _digest(obj.payload)
        state = _digest(
            {
                "schema_version": 1,
                "object_type": obj.type.value,
                "object_schema_sha256": _text_digest(obj.schema_version),
                "version": obj.version,
                "immutable": obj.immutable,
                "lifecycle_state": obj.lifecycle_state.value,
                "labels": {
                    "sensitivity": obj.metadata.sensitivity,
                    "integrity": obj.metadata.integrity,
                    "trust_level": obj.metadata.trust_level,
                },
            }
        )
        version = _digest(
            {
                "schema_version": 1,
                "oid_sha256": _text_digest(obj.oid),
                "version": obj.version,
                "content_sha256": content,
                "state_sha256": state,
            }
        )
        provenance = _digest(
            {
                "schema_version": 1,
                "created_from_action_sha256": (
                    _text_digest(obj.provenance.created_from_action)
                    if obj.provenance.created_from_action
                    else None
                ),
                "parent_oid_sha256s": [
                    _text_digest(value) for value in obj.provenance.parent_oids
                ],
                "source_ref_sha256s": [
                    _text_digest(value) for value in obj.provenance.source_refs
                ],
                "source_operation_sha256s": [
                    _text_digest(value)
                    for value in obj.provenance.source_operation_ids
                ],
            }
        )
        return content, version, state, provenance

    def _capture_object_derivation(
        self,
        pid: str,
        output: _CapturedObject,
        *,
        inputs: tuple[FlowInputEdge, ...],
        state_sha256: str,
        provenance_sha256: str,
    ) -> None:
        if not inputs:
            return
        kind = (
            FlowActivityKind.AGGREGATION
            if len(inputs) > 1
            else FlowActivityKind.TRANSFORMATION
        )
        self._flow.capture_activity(
            activity_kind=kind,
            pid=pid,
            action_id=f"memory.{kind.value}",
            state_sha256=state_sha256,
            provenance_sha256=provenance_sha256,
            tenant_bucket_sha256=output.tenant_bucket_sha256,
            inputs=inputs,
            outputs=(FlowOutputEdge(output.ref, FlowEdgeRelation.DIRECT),),
            created_at=utc_now(),
        )

    def _capture_tool_result(
        self,
        pid: str,
        obj: AgentObject,
        object_version: _CapturedObject,
        *,
        content_sha256: str,
        version_sha256: str,
        state_sha256: str,
        provenance_sha256: str,
    ) -> _CapturedObject:
        source_action = obj.provenance.created_from_action or "tool.unknown"
        tool_schema_sha256 = _digest(
            {"tool_action_sha256": _text_digest(source_action)}
        )
        selection, _inferred = self._selection_for(
            pid,
            source_action,
            object_version,
        )
        inputs = (
            FlowInputEdge(object_version.ref, FlowEdgeRelation.DIRECT),
            FlowInputEdge(selection.ref, FlowEdgeRelation.CONTROL),
        )
        coverage = object_version.coverage
        # The Object Memory seam has only a derived action identifier, not the
        # frozen Tool registry spec/effect binding. A matching model selection
        # therefore remains advisory and can never make this capture complete.
        coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        bundle = self._flow.capture_tool_result(
            pid=pid,
            # Object provenance is caller-supplied. Persist only a fixed
            # ontology value until a Tool execution observer proves the exact
            # registered action and schema.
            action_id="tool.unknown",
            content_sha256=content_sha256,
            version_sha256=_digest(
                {
                    "object_version_sha256": version_sha256,
                    "tool_schema_sha256": tool_schema_sha256,
                }
            ),
            state_sha256=state_sha256,
            provenance_sha256=provenance_sha256,
            labels=object_version.labels,
            tenant_bucket_sha256=self._tenant_bucket_arg(object_version.labels),
            inputs=inputs,
            tool_schema_sha256=tool_schema_sha256,
            coverage=coverage,
            created_at=utc_now(),
        )
        if bundle is None or len(bundle.entities) != 1:
            raise RuntimeError("tool result FlowGraph capture returned no entity")
        tool_ref = FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY)
        self._flow.capture_activity(
            activity_kind=FlowActivityKind.CONDITIONAL,
            pid=pid,
            action_id="runtime.tool_dispatch_condition",
            state_sha256=state_sha256,
            provenance_sha256=provenance_sha256,
            tenant_bucket_sha256=bundle.entities[0].tenant_bucket_sha256,
            inputs=(FlowInputEdge(selection.ref, FlowEdgeRelation.CONTROL),),
            outputs=(FlowOutputEdge(tool_ref, FlowEdgeRelation.CONTROL),),
            tool_schema_sha256=tool_schema_sha256,
            created_at=utc_now(),
        )
        return _CapturedObject(
            tool_ref,
            object_version.labels,
            bundle.entities[0].tenant_bucket_sha256,
            FlowCoverageStatus(bundle.entities[0].coverage),
        )

    def _selection_for(
        self,
        pid: str,
        action: str,
        labels_from: _CapturedObject,
    ) -> tuple[_CapturedObject, bool]:
        key = (pid, _text_digest(action))
        queued = _lru_get(self._pending_selections, key)
        if queued:
            selected = queued.popleft()
            if not queued:
                self._pending_selections.pop(key, None)
            return selected, False
        model = _lru_get(self._latest_model_output, pid)
        inputs = (
            (FlowInputEdge(model.ref, FlowEdgeRelation.CONTROL),)
            if model is not None
            and model.tenant_bucket_sha256 == labels_from.tenant_bucket_sha256
            else ()
        )
        digest = _digest(
            {
                "schema_version": 1,
                "pid_sha256": _text_digest(pid),
                "tool_action_sha256": _text_digest(action),
            }
        )
        bundle = self._flow.capture_derived_entity(
            entity_kind=FlowEntityKind.MODEL_OUTPUT,
            activity_kind=FlowActivityKind.TOOL_SELECTION,
            pid=pid,
            action_id="runtime.tool_selection",
            content_sha256=digest,
            version_sha256=_digest({"selection_sha256": digest, "inferred": True}),
            state_sha256=_digest({"selection_sha256": digest}),
            provenance_sha256=digest,
            labels=labels_from.labels,
            tenant_bucket_sha256=self._tenant_bucket_arg(labels_from.labels),
            inputs=inputs,
            coverage=FlowCoverageStatus.PARTIAL,
            created_at=utc_now(),
        )
        if bundle is None or len(bundle.entities) != 1:
            raise RuntimeError("tool selection FlowGraph capture returned no entity")
        return (
            _CapturedObject(
                FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
                labels_from.labels,
                bundle.entities[0].tenant_bucket_sha256,
                FlowCoverageStatus(bundle.entities[0].coverage),
            ),
            True,
        )

    def _capture_materialization(
        self,
        pid: str,
        context: MaterializedContext,
    ) -> None:
        with self._lock:
            included, omitted, coverage = self._materialization_sources(pid, context)
            labels = DataLabels.aggregate(item.labels for item in included)
            bucket = self._bucket(labels)
            direct_inputs: list[FlowInputEdge] = []
            for item in included:
                if item.tenant_bucket_sha256 != bucket:
                    coverage = FlowCoverageStatus.CONFLICT
                    continue
                coverage = _worse(coverage, item.coverage)
                direct_inputs.append(FlowInputEdge(item.ref, FlowEdgeRelation.DIRECT))
            for item in omitted:
                coverage = _worse(coverage, item.coverage)
                if item.tenant_bucket_sha256 != bucket:
                    coverage = FlowCoverageStatus.CONFLICT
            content_sha256 = _text_digest(context.text)
            manifest_digest = _digest(self._payload_free_manifest(context.object_manifest))
            state_sha256 = _digest(
                {
                    "schema_version": 1,
                    "policy_sha256": _text_digest(context.policy_used),
                    "token_count": context.token_count,
                    "budget_tokens": context.budget_tokens,
                    "included_count": len(included),
                    "omitted_count": len(context.omitted_objects),
                }
            )
            version_sha256 = _digest(
                {
                    "materialization_id_sha256": (
                        _text_digest(context.materialization_id)
                        if context.materialization_id
                        else None
                    ),
                    "content_sha256": content_sha256,
                    "manifest_sha256": manifest_digest,
                    "state_sha256": state_sha256,
                }
            )
            provenance_sha256 = _digest(
                {
                    "view_id_sha256": (
                        _text_digest(context.view_id) if context.view_id else None
                    ),
                    "manifest_sha256": manifest_digest,
                }
            )
            bundle = self._flow.capture_materialization(
                pid=pid,
                action_id="memory.materialize_context",
                content_sha256=content_sha256,
                version_sha256=version_sha256,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                labels=labels,
                tenant_bucket_sha256=self._tenant_bucket_arg(labels),
                inputs=tuple(direct_inputs),
                coverage=coverage,
                created_at=utc_now(),
            )
            if bundle is None or len(bundle.entities) != 1:
                raise RuntimeError("materialization FlowGraph capture returned no entity")
            output = _CapturedObject(
                FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
                labels,
                bundle.entities[0].tenant_bucket_sha256,
                FlowCoverageStatus(bundle.entities[0].coverage),
            )
            if context.materialization_id:
                _lru_put(
                    self._materializations,
                    (pid, _text_digest(context.materialization_id)),
                    output,
                    limit=_MAX_TRACKED_MATERIALIZATIONS,
                )
            self._capture_materialization_activities(
                pid,
                output,
                direct_inputs=tuple(direct_inputs),
                omitted=omitted,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
            )

    def _materialization_sources(
        self,
        pid: str,
        context: MaterializedContext,
    ) -> tuple[list[_CapturedObject], list[_CapturedObject], FlowCoverageStatus]:
        included: list[_CapturedObject] = []
        omitted: list[_CapturedObject] = []
        coverage = FlowCoverageStatus.COMPLETE
        included_oids: set[str] = set()
        omitted_oids: set[str] = set()
        seen_oids: set[str] = set()
        manifests = tuple(context.object_manifest)
        if len(manifests) > _MAX_OBJECT_INPUTS:
            manifests = manifests[:_MAX_OBJECT_INPUTS]
            coverage = FlowCoverageStatus.PARTIAL
        for item in manifests:
            if not isinstance(item, Mapping):
                coverage = _worse(coverage, FlowCoverageStatus.UNKNOWN)
                continue
            oid = item.get("oid")
            version = item.get("version")
            disposition = item.get("disposition")
            if (
                not isinstance(oid, str)
                or type(version) is not int
                or version <= 0
                or disposition not in {"included", "omitted"}
            ):
                coverage = _worse(coverage, FlowCoverageStatus.UNKNOWN)
                continue
            if oid in seen_oids:
                coverage = _worse(coverage, FlowCoverageStatus.CONFLICT)
                continue
            seen_oids.add(oid)
            obj = self._objects.get_object(oid)
            if obj is None or obj.version != version:
                coverage = _worse(coverage, FlowCoverageStatus.STALE)
                continue
            source = self._known_or_read_object(pid, obj)
            if disposition == "included":
                included_oids.add(oid)
                included.append(source)
            else:
                omitted_oids.add(oid)
                omitted.append(source)
        context_refs = tuple(context.object_refs)
        omitted_refs = tuple(context.omitted_objects)
        if (
            any(not isinstance(value, str) for value in context_refs)
            or len(context_refs) != len(set(context_refs))
            or any(not isinstance(value, str) for value in omitted_refs)
            or len(omitted_refs) != len(set(omitted_refs))
        ):
            coverage = _worse(coverage, FlowCoverageStatus.CONFLICT)
        elif set(context_refs) != included_oids or set(omitted_refs) != omitted_oids:
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        return included, omitted, coverage

    @staticmethod
    def _payload_free_manifest(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for value in values[:_MAX_OBJECT_INPUTS]:
            if not isinstance(value, Mapping):
                selected.append({"invalid": True})
                continue
            oid = value.get("oid")
            selected.append(
                {
                    "oid_sha256": _text_digest(oid) if isinstance(oid, str) else None,
                    "version": value.get("version"),
                    "disposition": value.get("disposition"),
                    "reason": value.get("reason"),
                    "transform": value.get("transform"),
                    "tokens": value.get("tokens"),
                    "rendered_sha256": value.get("rendered_sha256"),
                }
            )
        return selected

    def _capture_materialization_activities(
        self,
        pid: str,
        output: _CapturedObject,
        *,
        direct_inputs: tuple[FlowInputEdge, ...],
        omitted: list[_CapturedObject],
        state_sha256: str,
        provenance_sha256: str,
    ) -> None:
        if direct_inputs:
            self._flow.capture_activity(
                activity_kind=FlowActivityKind.MEMORY_RETRIEVAL,
                pid=pid,
                action_id="memory.retrieve_context",
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                tenant_bucket_sha256=output.tenant_bucket_sha256,
                inputs=tuple(
                    FlowInputEdge(item.source, FlowEdgeRelation.INDIRECT)
                    for item in direct_inputs
                ),
                outputs=(FlowOutputEdge(output.ref, FlowEdgeRelation.DIRECT),),
                created_at=utc_now(),
            )
        if len(direct_inputs) > 1:
            self._flow.capture_activity(
                activity_kind=FlowActivityKind.AGGREGATION,
                pid=pid,
                action_id="memory.aggregate_context",
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                tenant_bucket_sha256=output.tenant_bucket_sha256,
                inputs=direct_inputs,
                outputs=(FlowOutputEdge(output.ref, FlowEdgeRelation.DIRECT),),
                created_at=utc_now(),
            )
        conditional = tuple(
            FlowInputEdge(item.ref, FlowEdgeRelation.CONTROL)
            for item in omitted
            if item.tenant_bucket_sha256 == output.tenant_bucket_sha256
        )
        if conditional:
            self._flow.capture_activity(
                activity_kind=FlowActivityKind.CONDITIONAL,
                pid=pid,
                action_id="memory.materialization_selection",
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                tenant_bucket_sha256=output.tenant_bucket_sha256,
                inputs=conditional,
                outputs=(FlowOutputEdge(output.ref, FlowEdgeRelation.CONTROL),),
                created_at=utc_now(),
            )

    def _capture_model_output(self, state: Any, completion: Any, record: Any) -> None:
        pid = getattr(state, "pid", None)
        flow_context = getattr(state, "flow_context", None)
        if not isinstance(pid, str) or not isinstance(flow_context, DataFlowContext):
            raise TypeError("LLM flow observation is malformed")
        with self._lock:
            labels = flow_context.labels
            bucket = self._bucket(labels)
            binding = self._model_input_binding(
                state,
                pid=pid,
                flow_context=flow_context,
                labels=labels,
                bucket=bucket,
            )
            coverage = binding.coverage
            content = str(getattr(completion, "content", "") or "")
            tool_calls = tuple(getattr(completion, "tool_calls", ()) or ())
            if len(tool_calls) > _MAX_MODEL_TOOL_CALLS:
                raise ValueError("LLM FlowGraph tool-call bound is exceeded")
            call_descriptors = self._tool_call_descriptors(tool_calls)
            tool_calls_sha256 = _digest(tool_calls)
            if len(tool_calls) > _MAX_PENDING_SELECTIONS:
                coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
            content_sha256 = _digest(
                {
                    "content_sha256": _text_digest(content),
                    "tool_calls_sha256": tool_calls_sha256,
                    "tool_call_count": len(tool_calls),
                    "selection_descriptor_count": len(call_descriptors),
                }
            )
            model_artifact_sha256, model_coverage = self._model_provenance(
                state,
                record,
            )
            coverage = _worse(coverage, model_coverage)
            state_sha256 = _digest(
                {
                    "call_id_sha256": _text_digest(str(getattr(state, "call_id", ""))),
                    "attempt": getattr(state, "attempt", None),
                    "source_refs_sha256": flow_context.source_refs_hash(),
                    "provider_result_sha256": (
                        binding.provider_observation.result_sha256
                        if binding.provider_observation is not None
                        else None
                    ),
                    "provider_effect_id_sha256": (
                        _text_digest(binding.provider_observation.effect_id)
                        if binding.provider_observation is not None
                        else None
                    ),
                    "materialization_id_sha256": (
                        _text_digest(flow_context.materialization_id)
                        if flow_context.materialization_id
                        else None
                    ),
                }
            )
            version_sha256 = _digest(
                {
                    "content_sha256": content_sha256,
                    "model_artifact_sha256": model_artifact_sha256,
                    "state_sha256": state_sha256,
                }
            )
            provenance_sha256 = _digest(
                {
                    "source_refs_sha256": flow_context.source_refs_hash(),
                    "materialization_sha256": (
                        binding.materialized.ref.node_id
                        if binding.materialized is not None
                        else None
                    ),
                    "provider_entity_sha256": (
                        binding.provider_input.source.node_id
                        if binding.provider_input is not None
                        else None
                    ),
                }
            )
            bundle = self._flow.capture_model_output(
                pid=pid,
                action_id="llm.complete",
                content_sha256=content_sha256,
                version_sha256=version_sha256,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                labels=labels,
                tenant_bucket_sha256=self._tenant_bucket_arg(labels),
                inputs=binding.inputs,
                effect_id=(
                    binding.provider_observation.effect_id
                    if binding.provider_observation is not None
                    else None
                ),
                provider_spec_sha256=(
                    binding.provider_observation.provider_spec_sha256
                    if binding.provider_observation is not None
                    else None
                ),
                model_artifact_sha256=model_artifact_sha256,
                coverage=coverage,
                created_at=utc_now(),
            )
            if bundle is None or len(bundle.entities) != 1:
                raise RuntimeError("model output FlowGraph capture returned no entity")
            model_output = _CapturedObject(
                FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
                labels,
                bundle.entities[0].tenant_bucket_sha256,
                FlowCoverageStatus(bundle.entities[0].coverage),
            )
            _lru_put(
                self._latest_model_output,
                pid,
                model_output,
                limit=_MAX_TRACKED_PROCESSES,
            )
            self._capture_tool_selections(
                pid,
                model_output,
                call_descriptors=call_descriptors,
                model_artifact_sha256=model_artifact_sha256,
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
            )

    def _model_input_binding(
        self,
        state: Any,
        *,
        pid: str,
        flow_context: DataFlowContext,
        labels: DataLabels,
        bucket: str,
    ) -> _ModelInputBinding:
        materialized = self._model_materialization(pid, flow_context)
        edges: list[FlowInputEdge] = []
        coverage = FlowCoverageStatus.COMPLETE
        if materialized is None:
            coverage = FlowCoverageStatus.PARTIAL
        elif materialized.tenant_bucket_sha256 != bucket:
            coverage = FlowCoverageStatus.CONFLICT
        else:
            edges.append(FlowInputEdge(materialized.ref, FlowEdgeRelation.DIRECT))
            coverage = _worse(coverage, materialized.coverage)

        raw_observation = getattr(state, "semantic_provider_observation", None)
        observation = (
            raw_observation
            if isinstance(raw_observation, PostCommitResultObservation)
            else None
        )
        provider_input = None
        if observation is None:
            coverage = _worse(
                coverage,
                FlowCoverageStatus.CONFLICT
                if raw_observation is not None
                else FlowCoverageStatus.PARTIAL,
            )
        elif (
            observation.contract_name != "primitive.llm.complete"
            or observation.pid != pid
        ):
            coverage = FlowCoverageStatus.CONFLICT
        else:
            selected = self._provider_input(observation, labels)
            if selected:
                provider_input = selected[0]
                edges.append(provider_input)
                provider_entity = self._flow.get_entity(provider_input.source.node_id)
                coverage = _worse(
                    coverage,
                    FlowCoverageStatus.UNKNOWN
                    if provider_entity is None
                    else FlowCoverageStatus(provider_entity.coverage),
                )
            else:
                coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        return _ModelInputBinding(
            inputs=tuple(dict.fromkeys(edges)),
            coverage=coverage,
            materialized=materialized,
            provider_observation=observation,
            provider_input=provider_input,
        )

    def _model_materialization(
        self,
        pid: str,
        flow_context: DataFlowContext,
    ) -> _CapturedObject | None:
        if not flow_context.materialization_id:
            return None
        return _lru_get(
            self._materializations,
            (pid, _text_digest(flow_context.materialization_id)),
        )

    @staticmethod
    def _model_provenance(
        state: Any,
        record: Any,
    ) -> tuple[str, FlowCoverageStatus]:
        resolved = getattr(state, "resolved", None)
        sink = getattr(state, "sink", None)
        profile = getattr(resolved, "profile", None)
        resolved_identity = getattr(resolved, "identity_sha256", None)
        sink_identity = getattr(sink, "identity_sha256", None)
        configured_model = getattr(profile, "model", None)
        observed_model = getattr(record, "model", None)
        requested_profile_id = getattr(state, "profile_id", None)
        resolved_profile_id = getattr(resolved, "profile_id", None)
        coverage = FlowCoverageStatus.COMPLETE
        required = (
            resolved_identity,
            sink_identity,
            configured_model,
            observed_model,
            requested_profile_id,
            resolved_profile_id,
        )
        if any(not isinstance(value, str) or not value for value in required):
            coverage = FlowCoverageStatus.PARTIAL
        if (
            isinstance(resolved_identity, str)
            and isinstance(sink_identity, str)
            and resolved_identity != sink_identity
        ):
            coverage = FlowCoverageStatus.CONFLICT
        if (
            isinstance(configured_model, str)
            and isinstance(observed_model, str)
            and configured_model != observed_model
        ):
            coverage = FlowCoverageStatus.CONFLICT
        if (
            isinstance(requested_profile_id, str)
            and isinstance(resolved_profile_id, str)
            and requested_profile_id != resolved_profile_id
        ):
            coverage = FlowCoverageStatus.CONFLICT
        return (
            _digest(
                {
                    "profile_id_sha256": (
                        _text_digest(requested_profile_id)
                        if isinstance(requested_profile_id, str)
                        else None
                    ),
                    "resolved_profile_id_sha256": (
                        _text_digest(resolved_profile_id)
                        if isinstance(resolved_profile_id, str)
                        else None
                    ),
                    "profile_identity_sha256": (
                        resolved_identity
                        if isinstance(resolved_identity, str)
                        else None
                    ),
                    "sink_identity_sha256": (
                        sink_identity if isinstance(sink_identity, str) else None
                    ),
                    "configured_model_sha256": (
                        _text_digest(configured_model)
                        if isinstance(configured_model, str)
                        else None
                    ),
                    "observed_model_sha256": (
                        _text_digest(observed_model)
                        if isinstance(observed_model, str)
                        else None
                    ),
                }
            ),
            coverage,
        )

    @staticmethod
    def _tool_call_descriptors(tool_calls: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
        selected: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls[:_MAX_PENDING_SELECTIONS]):
            name = _tool_call_name(call)
            selected.append(
                {
                    "index": index,
                    "call_sha256": _digest(call),
                    "tool_name_sha256": _text_digest(name) if name else None,
                }
            )
        return tuple(selected)

    def _capture_tool_selections(
        self,
        pid: str,
        model_output: _CapturedObject,
        *,
        call_descriptors: tuple[dict[str, Any], ...],
        model_artifact_sha256: str,
        state_sha256: str,
        provenance_sha256: str,
    ) -> None:
        for descriptor in call_descriptors:
            content_sha256 = descriptor["call_sha256"]
            bundle = self._flow.capture_derived_entity(
                entity_kind=FlowEntityKind.MODEL_OUTPUT,
                activity_kind=FlowActivityKind.TOOL_SELECTION,
                pid=pid,
                action_id="runtime.tool_selection",
                content_sha256=content_sha256,
                version_sha256=_digest(
                    {
                        "model_output": model_output.ref.node_id,
                        "call_sha256": content_sha256,
                    }
                ),
                state_sha256=state_sha256,
                provenance_sha256=provenance_sha256,
                labels=model_output.labels,
                tenant_bucket_sha256=self._tenant_bucket_arg(model_output.labels),
                inputs=(
                    FlowInputEdge(model_output.ref, FlowEdgeRelation.CONTROL),
                ),
                model_artifact_sha256=model_artifact_sha256,
                coverage=model_output.coverage,
                created_at=utc_now(),
            )
            if bundle is None or len(bundle.entities) != 1:
                raise RuntimeError("model tool selection capture returned no entity")
            tool_name_sha256 = descriptor.get("tool_name_sha256")
            if not isinstance(tool_name_sha256, str):
                continue
            selection = _CapturedObject(
                FlowNodeRef(bundle.entities[0].entity_id, FlowNodeType.ENTITY),
                model_output.labels,
                bundle.entities[0].tenant_bucket_sha256,
                FlowCoverageStatus(bundle.entities[0].coverage),
            )
            key = (pid, tool_name_sha256)
            queue = _lru_get(self._pending_selections, key)
            if queue is None:
                queue = deque()
                _lru_put(
                    self._pending_selections,
                    key,
                    queue,
                    limit=_MAX_PENDING_SELECTION_KEYS,
                )
            if len(queue) >= _MAX_PENDING_SELECTIONS:
                queue.popleft()
            queue.append(selection)

    def _capture_provider_version(self, result: Any, observation: Any) -> None:
        if self._capture_current_file_version(result, observation):
            return
        self._capture_git_version(result, observation)

    def _capture_current_file_version(self, result: Any, observation: Any) -> bool:
        if type(result) not in {FileReadResult, FileBytesReadResult, FileWriteResult}:
            return False
        if self._filesystem is None or self._data_flow is None:
            raise RuntimeError("filesystem FlowGraph resolver is unavailable")
        snapshot = self._read_current_file(result)
        if snapshot is None:
            raise RuntimeError("current file version could not be captured")
        content_sha256 = hashlib.sha256(snapshot.content).hexdigest()
        coverage = self._current_file_coverage(
            result,
            snapshot,
            content_sha256=content_sha256,
        )
        detector = LocalDlpAccumulator(input_sha256=content_sha256)
        detector.scan(snapshot.content)
        dlp_findings = detector.findings
        labels = self._labels_after_local_dlp(
            snapshot.after_context.labels,
            matched=bool(dlp_findings),
        )
        state_sha256 = _digest(
            {
                "resource_sha256": _text_digest(
                    self._filesystem.resource_for(snapshot.relative)
                ),
                "size_bytes": snapshot.second_state.size_bytes,
                "modified_at_sha256": _digest(
                    snapshot.second_state.modified_at
                ),
                "label_state_sha256": _digest(snapshot.after_label_state),
            }
        )
        version_sha256 = _digest(
            {
                "path_sha256": _text_digest(snapshot.relative),
                "content_sha256": content_sha256,
                "state_sha256": state_sha256,
            }
        )
        provenance_sha256 = _digest(
            {
                "effect_id_sha256": _text_digest(observation.effect_id),
                "source_refs_sha256": snapshot.after_context.source_refs_hash(),
            }
        )
        inputs = self._provider_input(observation, labels)
        if not inputs:
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        operation = "write" if type(result) is FileWriteResult else "read"
        bundle = self._flow.capture_file_version(
            operation=operation,
            pid=observation.pid,
            action_id=f"filesystem.{operation}",
            content_sha256=content_sha256,
            version_sha256=version_sha256,
            state_sha256=state_sha256,
            provenance_sha256=provenance_sha256,
            labels=labels,
            tenant_bucket_sha256=self._tenant_bucket_arg(labels),
            inputs=inputs,
            effect_id=observation.effect_id,
            provider_spec_sha256=observation.provider_spec_sha256,
            coverage=coverage,
            created_at=utc_now(),
        )
        if bundle is None or len(bundle.entities) != 1:
            raise RuntimeError("file version FlowGraph capture returned no entity")
        self._append_local_dlp_assertions(
            entity_id=bundle.entities[0].entity_id,
            findings=dlp_findings,
            labels=labels,
            coverage=coverage,
        )
        return True

    def _read_current_file(self, result: Any) -> _CurrentFileRead | None:
        target, relative = self._filesystem.resolve_path(result.path)
        maximum = int(self._filesystem.config.tools.filesystem_read_hard_limit_bytes)
        with self._filesystem.hold_file_label_io_paths((relative,)):
            _before_context, before_label_state = self._data_flow.file_snapshot(
                relative
            )
            first_state = self._filesystem.provider.state(target)
            if (
                not first_state.exists
                or first_state.kind != "file"
                or first_state.size_bytes is None
                or first_state.size_bytes < 0
                or first_state.size_bytes > maximum
            ):
                return None
            content = self._filesystem.provider.read_bytes(target, max_bytes=maximum)
            second_state = self._filesystem.provider.state(target)
            after_context, after_label_state = self._data_flow.file_snapshot(relative)
            binding = self._data_flow.current_file_label_binding(relative)
        return _CurrentFileRead(
            relative=relative,
            content=content,
            first_state=first_state,
            second_state=second_state,
            before_label_state=before_label_state,
            after_context=after_context,
            after_label_state=after_label_state,
            binding=binding,
        )

    @staticmethod
    def _current_file_coverage(
        result: Any,
        snapshot: _CurrentFileRead,
        *,
        content_sha256: str,
    ) -> FlowCoverageStatus:
        coverage = FlowCoverageStatus.COMPLETE
        if (
            snapshot.first_state != snapshot.second_state
            or snapshot.before_label_state != snapshot.after_label_state
        ):
            coverage = FlowCoverageStatus.STALE
        if snapshot.second_state.modified_at is None:
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        if type(result) is FileReadResult:
            if result.truncated or result.content_sha256 != content_sha256:
                coverage = _worse(coverage, FlowCoverageStatus.STALE)
        elif type(result) is FileBytesReadResult:
            observed = hashlib.sha256(result.content).hexdigest()
            if result.truncated or observed != content_sha256:
                coverage = _worse(coverage, FlowCoverageStatus.STALE)
        binding = snapshot.binding
        if (
            binding is None
            or binding.tombstoned
            or not binding.active
            or binding.content_sha256 is None
        ):
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        elif binding.content_sha256 != content_sha256:
            coverage = _worse(coverage, FlowCoverageStatus.STALE)
        return coverage

    @staticmethod
    def _labels_after_local_dlp(
        labels: DataLabels,
        *,
        matched: bool,
    ) -> DataLabels:
        if not matched:
            return labels
        # Tighten the baseline as well as asserting the finding, so an
        # assertion append failure can never expose a normal-labelled grant.
        return DataLabels(
            sensitivity=DataSensitivity.SECRET,
            integrity=labels.integrity,
            trust_level=labels.trust_level,
            origin=labels.origin,
            tenant=labels.tenant,
            principal=labels.principal,
            declassification_authority=labels.declassification_authority,
        )

    def _append_local_dlp_assertions(
        self,
        *,
        entity_id: str,
        findings: tuple[Any, ...],
        labels: DataLabels,
        coverage: FlowCoverageStatus,
    ) -> None:
        for finding in findings:
            self._flow.append_label_assertion(
                entity_id=entity_id,
                finding=SemanticDataFinding(
                    category=finding.category,
                    field=SemanticDataLocator.PROVIDER_RESULT,
                    span_start=None,
                    span_end=None,
                    sensitivity_floor=DataSensitivity.SECRET,
                    integrity_ceiling=labels.integrity,
                    trust_ceiling=labels.trust_level,
                    confidence_bps=10_000,
                    evidence_sha256=finding.evidence_sha256,
                ),
                source=FlowLabelSource.HOST,
                assessment_id=None,
                coverage=coverage,
                created_at=utc_now(),
            )

    def _capture_git_version(self, result: Any, observation: Any) -> bool:
        if type(result) not in {GitStatusResult, GitDiffResult}:
            return False
        labels = observation.data_labels or DataLabels(
            trust_level="untrusted",
            integrity="untrusted",
            origin="external:git",
        )
        action_id = "git.diff" if type(result) is GitDiffResult else "git.read"
        content_sha256 = result.sha256
        state_sha256 = _digest(result.state)
        version_sha256 = _digest(
            {
                "repository_id_sha256": _text_digest(result.repository_id),
                "worktree_id_sha256": _text_digest(result.worktree_id),
                "content_sha256": content_sha256,
                "state_sha256": state_sha256,
            }
        )
        provenance_sha256 = _digest(
            {
                "effect_id_sha256": _text_digest(observation.effect_id),
                "provider_spec_sha256": observation.provider_spec_sha256,
                "source_refs_sha256": observation.source_refs_sha256,
            }
        )
        coverage = (
            FlowCoverageStatus.COMPLETE
            if not result.truncated and observation.provider_spec_sha256 is not None
            else FlowCoverageStatus.PARTIAL
        )
        inputs = self._provider_input(observation, labels)
        if not inputs:
            coverage = _worse(coverage, FlowCoverageStatus.PARTIAL)
        self._flow.capture_git_snapshot(
            pid=observation.pid,
            action_id=action_id,
            content_sha256=content_sha256,
            version_sha256=version_sha256,
            state_sha256=state_sha256,
            provenance_sha256=provenance_sha256,
            labels=labels,
            tenant_bucket_sha256=self._tenant_bucket_arg(labels),
            inputs=inputs,
            effect_id=observation.effect_id,
            provider_spec_sha256=observation.provider_spec_sha256,
            coverage=coverage,
            created_at=utc_now(),
        )
        return True

    def _provider_input(
        self,
        observation: Any,
        labels: DataLabels,
    ) -> tuple[FlowInputEdge, ...]:
        if not isinstance(observation.result_sha256, str):
            return ()
        entity_id = provider_result_entity_id(
            pid=observation.pid,
            effect_id=observation.effect_id,
            result_sha256=observation.result_sha256,
            state_sha256=_digest(
                {
                    "provider": observation.provider,
                    "operation": observation.operation,
                    "target_sha256": _digest(observation.target),
                    "contract_name": observation.contract_name,
                    "data_flow_direction": observation.data_flow_direction,
                    "result_descriptor": dict(observation.result_descriptor),
                }
            ),
        )
        entity = self._flow.get_entity(entity_id)
        if entity is None or entity.tenant_bucket_sha256 != self._bucket(labels):
            return ()
        return (
            FlowInputEdge(
                FlowNodeRef(entity_id, FlowNodeType.ENTITY),
                FlowEdgeRelation.DIRECT,
            ),
        )

    def _bucket(self, labels: DataLabels) -> str:
        if labels.tenant is not None:
            if self._tenant_bucketer is None:
                return FLOW_UNBUCKETED_IDENTITY_SHA256
            selected = self._tenant_bucketer(labels.tenant)
            if not isinstance(selected, str) or len(selected) != 64:
                raise ValueError("runtime FlowGraph tenant bucketer returned invalid digest")
            return selected
        if labels.principal is not None:
            return FLOW_UNBUCKETED_IDENTITY_SHA256
        return FLOW_NO_TENANT_BUCKET_SHA256

    def _tenant_bucket_arg(self, labels: DataLabels) -> str | None:
        if labels.tenant is None:
            return None
        if self._tenant_bucketer is None:
            return None
        return self._bucket(labels)

    @staticmethod
    def _object_key(pid: str, obj: AgentObject) -> tuple[str, str, int]:
        return pid, _text_digest(obj.oid), obj.version


def _tool_call_name(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("name")
    if isinstance(direct, str) and direct:
        return f"tool.{direct}"
    function = value.get("function")
    if isinstance(function, Mapping):
        selected = function.get("name")
        if isinstance(selected, str) and selected:
            return f"tool.{selected}"
    return None


__all__ = ["SemanticRuntimeFlowObserver"]
