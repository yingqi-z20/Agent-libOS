from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from reprlib import Repr
from collections.abc import Callable
from types import TracebackType
from typing import Any, Iterable

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.utils.ids import estimate_tokens, new_id, utc_now
from agent_libos.models import (
    EventType,
    MaterializedContext,
    MemoryView,
    MemoryViewSpec,
    MergePolicy,
    MergeResult,
    ObjectNamespace,
    ObjectFilter,
    ObjectHandle,
    ObjectLifecycleState,
    ObjectLink,
    ObjectMetadata,
    ObjectPatch,
    ObjectQuery,
    ObjectRight,
    ObjectOwnerKind,
    ObjectType,
    Provenance,
    RelationType,
    ResourceUsage,
    UNSET,
    ViewMode,
    AgentObject,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.storage import SQLiteStore
from agent_libos.tools.observability import ensure_json_size


class ObjectLifetimeScope:
    """Runtime-internal RAII guard for multi-step Object Memory writes."""

    def __init__(
        self,
        manager: ObjectMemoryManager,
        *,
        actor: str,
        owner_kind: ObjectOwnerKind | str,
        owner_id: str,
        reason: str,
    ) -> None:
        self.manager = manager
        self.actor = actor
        self.owner_kind = ObjectOwnerKind(owner_kind)
        self.owner_id = owner_id
        self.reason = reason
        self._oids: set[str] = set()
        self._committed = False

    def __enter__(self) -> ObjectLifetimeScope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if not self._committed:
            for oid in sorted(self._oids):
                self.manager.delete_object_trusted(
                    self.actor,
                    oid,
                    reason=f"{self.reason}.scope_discard",
                )
        return False

    def create_object(self, pid: str, *args: Any, **kwargs: Any) -> ObjectHandle:
        kwargs.setdefault("owner_kind", self.owner_kind)
        kwargs.setdefault("owner_id", self.owner_id)
        handle = self.manager.create_object(pid, *args, **kwargs)
        self._oids.add(handle.oid)
        return handle

    def track(self, handle_or_oid: ObjectHandle | str) -> None:
        self._oids.add(handle_or_oid.oid if isinstance(handle_or_oid, ObjectHandle) else str(handle_or_oid))

    def preserve(self, handle_or_oid: ObjectHandle | str) -> None:
        self._oids.discard(handle_or_oid.oid if isinstance(handle_or_oid, ObjectHandle) else str(handle_or_oid))

    def transfer(
        self,
        handle_or_oid: ObjectHandle | str,
        *,
        owner_kind: ObjectOwnerKind | str,
        owner_id: str,
    ) -> None:
        oid = handle_or_oid.oid if isinstance(handle_or_oid, ObjectHandle) else str(handle_or_oid)
        transferred = self.manager.transfer_owner(
            self.owner_kind,
            self.owner_id,
            ObjectOwnerKind(owner_kind),
            owner_id,
            [oid],
            actor=self.actor,
            reason=f"{self.reason}.scope_transfer",
        )
        if oid in transferred:
            self._oids.discard(oid)

    def commit(self) -> None:
        self._committed = True
        self._oids.clear()


class ObjectMemoryManager:
    """Typed Object Memory with capability-checked handles and namespace-local names."""

    def __init__(
        self,
        store: SQLiteStore,
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.resources = resources
        self._object_pin_checker: Callable[[str], bool] | None = None
        self._object_change_notifier: Callable[[str, dict[str, Any], str], None] | None = None
        self._object_release_finalizers: list[Callable[[AgentObject, str, str], None]] = []

    def bind_object_pin_checker(self, checker: Callable[[str], bool] | None) -> None:
        self._object_pin_checker = checker

    def bind_object_change_notifier(self, notifier: Callable[[str, dict[str, Any], str], None] | None) -> None:
        self._object_change_notifier = notifier

    def bind_object_release_finalizer(self, finalizer: Callable[[AgentObject, str, str], None]) -> None:
        self._object_release_finalizers.append(finalizer)

    def create_object(
        self,
        pid: str,
        object_type: ObjectType | str,
        payload: Any,
        metadata: ObjectMetadata | None = None,
        immutable: bool = True,
        provenance: Provenance | None = None,
        name: str | None = None,
        namespace: str | None = None,
        owner_kind: ObjectOwnerKind | str = ObjectOwnerKind.PROCESS,
        owner_id: str | None = None,
    ) -> ObjectHandle:
        obj_type = ObjectType(object_type)
        self._validate_payload_size(payload, "object payload")
        with self.store._lock:
            now = utc_now()
            oid = new_id("obj")
            object_namespace = self.resolve_namespace(pid, namespace)
            object_name = self._normalize_name(name or self._default_name(obj_type, oid))
            namespace_decision = self._require_namespace_right(pid, object_namespace, "write")
            self._require_namespace_exists(object_namespace)
            # Names are stable namespace directory entries, not authority. Reads by
            # name still resolve to an oid and pass through object capability checks.
            self._require_unique_name(object_name, object_namespace)
            meta = self._metadata_for_payload(payload, metadata)
            obj = AgentObject(
                oid=oid,
                namespace=object_namespace,
                name=object_name,
                type=obj_type,
                schema_version=self.config.memory.object_schema_version,
                payload=payload,
                metadata=meta,
                provenance=provenance or Provenance(created_from_action="memory.create_object"),
                version=1,
                immutable=immutable,
                created_by=pid,
                created_at=now,
                updated_at=now,
                owner_kind=ObjectOwnerKind(owner_kind),
                owner_id=owner_id or pid,
                lifecycle_state=ObjectLifecycleState.LIVE,
                deleted_at=None,
            )
            rights = {
                ObjectRight.READ.value,
                ObjectRight.LINK.value,
                ObjectRight.DIFF.value,
                ObjectRight.MATERIALIZE.value,
                ObjectRight.DELETE.value,
                ObjectRight.GRANT.value,
            }
            if not immutable:
                rights.add(ObjectRight.WRITE.value)
            self._consume_one_time_decision(namespace_decision)
            self.store.insert_object(obj)
            handle = self.capabilities.handle_for_object(pid, obj.oid, rights, issued_by="memory")
        self.events.emit(
            EventType.OBJECT_CREATED,
            source=pid,
            target=pid,
            payload={
                "oid": obj.oid,
                "namespace": obj.namespace,
                "name": obj.name,
                "qualified_name": self.qualified_name(obj),
                "type": obj.type.value,
            },
        )
        self.audit.record(
            actor=pid,
            action="memory.create_object",
            target=f"object:{obj.oid}",
            output_refs=[obj.oid],
            capability_refs=[handle.capability_id],
            decision={"namespace": obj.namespace, "name": obj.name, "type": obj.type.value},
        )
        return handle

    def process_namespace(self, pid: str) -> str:
        return f"{self.config.memory.process_namespace_prefix}:{pid}"

    def resolve_namespace(self, pid: str, namespace: str | None = None) -> str:
        if namespace is None:
            return self.process_namespace(pid)
        return self._normalize_namespace(namespace)

    def ensure_process_namespace(self, pid: str, parent_pid: str | None = None) -> ObjectNamespace:
        namespace_name = self.process_namespace(pid)
        existing = self.store.get_namespace(namespace_name)
        if existing is None:
            now = utc_now()
            namespace = ObjectNamespace(
                namespace=namespace_name,
                parent_namespace=None,
                metadata={"kind": "process", "pid": pid, "parent_pid": parent_pid},
                created_by=pid,
                created_at=now,
                updated_at=now,
            )
            self.store.insert_namespace(namespace)
            existing = namespace
            self.audit.record(
                actor=pid,
                action="memory.ensure_process_namespace",
                target=self._namespace_resource(namespace_name),
                decision={"namespace": namespace_name, "parent_pid": parent_pid, "created": True},
            )
        self.capabilities.grant(
            subject=pid,
            resource=self._namespace_resource(namespace_name),
            rights=["read", "write", "admin"],
            issued_by="memory.process_namespace",
        )
        return existing

    def create_namespace(
        self,
        pid: str,
        namespace: str,
        parent_namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectNamespace:
        with self.store._lock:
            namespace_name = self._normalize_namespace(namespace)
            if self.store.namespace_exists(namespace_name):
                raise ValidationError(f"Object Memory namespace already exists: {namespace_name}")
            parent = self._normalize_namespace(parent_namespace) if parent_namespace else self._parent_namespace(namespace_name)
            parent_decision = None
            if parent is not None:
                parent_decision = self._require_namespace_right(pid, parent, "write")
                self._require_namespace_exists(parent)
            now = utc_now()
            ns = ObjectNamespace(
                namespace=namespace_name,
                parent_namespace=parent,
                metadata=dict(metadata or {}),
                created_by=pid,
                created_at=now,
                updated_at=now,
            )
            if parent_decision is not None:
                self._consume_one_time_decision(parent_decision)
            self.store.insert_namespace(ns)
            self.capabilities.grant(
                subject=pid,
                resource=self._namespace_resource(namespace_name),
                rights=["read", "write", "admin"],
                issued_by="memory.namespace",
            )
        self.audit.record(
            actor=pid,
            action="memory.create_namespace",
            target=self._namespace_resource(namespace_name),
            decision={"namespace": namespace_name, "parent_namespace": parent},
        )
        return ns

    def get_namespace(self, pid: str, namespace: str | None = None) -> ObjectNamespace:
        namespace_name = self.resolve_namespace(pid, namespace)
        namespace_decision = self._require_namespace_right(pid, namespace_name, "read")
        ns = self.store.get_namespace(namespace_name)
        if ns is None:
            raise NotFound(f"Object Memory namespace not found: {namespace_name}")
        self.audit.record(
            actor=pid,
            action="memory.get_namespace",
            target=self._namespace_resource(namespace_name),
            decision={"namespace": namespace_name},
        )
        self._consume_one_time_decision(namespace_decision)
        return ns

    def list_namespace(self, pid: str, namespace: str | None = None) -> dict[str, Any]:
        namespace_name = self.resolve_namespace(pid, namespace)
        namespace_decision = self._require_namespace_right(pid, namespace_name, "read")
        self._require_namespace_exists(namespace_name)
        objects = [
            obj
            for obj in self.store.list_objects(namespace=namespace_name)
            if self.capabilities.check(pid, f"object:{obj.oid}", ObjectRight.READ)
        ]
        child_namespaces = [
            ns
            for ns in self.store.list_namespaces(parent_namespace=namespace_name)
            if self._can_read_namespace(pid, ns.namespace)
        ]
        self.audit.record(
            actor=pid,
            action="memory.list_namespace",
            target=self._namespace_resource(namespace_name),
            output_refs=[obj.oid for obj in objects],
            decision={"namespace": namespace_name, "objects": len(objects), "namespaces": len(child_namespaces)},
        )
        self._consume_one_time_decision(namespace_decision)
        return {
            "namespace": namespace_name,
            "objects": objects,
            "namespaces": child_namespaces,
        }

    def get_object(self, pid: str, handle: ObjectHandle) -> AgentObject:
        self.capabilities.assert_handle(pid, handle, ObjectRight.READ)
        obj = self.store.get_object(handle.oid)
        if obj is None:
            raise NotFound(f"object not found: {handle.oid}")
        self.audit.record(
            actor=pid,
            action="memory.get_object",
            target=f"object:{handle.oid}",
            input_refs=[handle.oid],
            capability_refs=[handle.capability_id],
        )
        return obj

    def get_object_by_name(self, pid: str, name: str, namespace: str | None = None) -> AgentObject:
        object_namespace = self.resolve_namespace(pid, namespace)
        object_name = self._normalize_name(name)
        namespace_decision = self._require_namespace_right(pid, object_namespace, "read")
        self._require_namespace_exists(object_namespace)
        obj = self.store.get_object_by_name(object_name, namespace=object_namespace)
        if obj is None:
            raise NotFound(f"object not found: {self.qualified_name_parts(object_namespace, object_name)}")
        # Name lookup never bypasses the object capability model.
        decision = self.capabilities.require(pid, f"object:{obj.oid}", ObjectRight.READ)
        self.audit.record(
            actor=pid,
            action="memory.get_object_by_name",
            target=f"object:{self.qualified_name(obj)}",
            input_refs=[obj.oid],
            decision={"namespace": obj.namespace, "name": obj.name, "oid": obj.oid},
        )
        self._consume_one_time_decision(namespace_decision)
        self._consume_one_time_decision(decision)
        return obj

    def handle_for_name(
        self,
        pid: str,
        name: str,
        rights: set[str] | list[str] | tuple[str, ...] | None = None,
        issued_by: str = "memory.name",
        namespace: str | None = None,
    ) -> ObjectHandle:
        object_namespace = self.resolve_namespace(pid, namespace)
        object_name = self._normalize_name(name)
        namespace_decision = self._require_namespace_right(pid, object_namespace, "read")
        self._require_namespace_exists(object_namespace)
        obj = self.store.get_object_by_name(object_name, namespace=object_namespace)
        if obj is None:
            raise NotFound(f"object not found: {self.qualified_name_parts(object_namespace, object_name)}")
        requested = {str(right) for right in (rights or {ObjectRight.READ.value})}
        # A name can be resolved only into rights the process already has.
        decisions = []
        for right in requested:
            decisions.append(self.capabilities.require(pid, f"object:{obj.oid}", right))
        one_time = any(decision.consume_capability_id is not None for decision in decisions)
        handle = self.capabilities.handle_for_object(
            pid,
            obj.oid,
            requested,
            issued_by=issued_by,
            uses_remaining=1 if one_time else None,
        )
        self._consume_one_time_decisions(decisions)
        self.audit.record(
            actor=pid,
            action="memory.handle_for_name",
            target=f"object:{self.qualified_name(obj)}",
            output_refs=[obj.oid],
            capability_refs=[handle.capability_id],
            decision={"namespace": obj.namespace, "name": obj.name, "rights": sorted(requested)},
        )
        self._consume_one_time_decision(namespace_decision)
        return handle

    def handle_for_oid(
        self,
        pid: str,
        oid: str,
        *,
        required_rights: Iterable[str | ObjectRight] | None = None,
        optional_rights: Iterable[str | ObjectRight] | None = None,
        issued_by: str = "memory.oid",
    ) -> ObjectHandle:
        if self.store.get_object(oid) is None:
            raise NotFound(f"object not found: {oid}")
        required = {str(right) for right in (required_rights or {ObjectRight.READ.value})}
        optional = {str(right) for right in (optional_rights or set())} - required
        rights, decisions = self._authorized_object_rights(
            pid,
            oid,
            required_rights=required,
            optional_rights=optional,
        )
        handle = self.capabilities.handle_for_object(
            pid,
            oid,
            rights,
            issued_by=issued_by,
            uses_remaining=1 if self._has_one_time_decision(decisions) else None,
        )
        self._consume_one_time_decisions(decisions)
        self.audit.record(
            actor=pid,
            action="memory.handle_for_oid",
            target=f"object:{oid}",
            output_refs=[oid],
            capability_refs=[handle.capability_id],
            decision={"rights": sorted(rights)},
        )
        return handle

    def update_object(self, pid: str, handle: ObjectHandle, patch: ObjectPatch) -> ObjectHandle:
        with self.store._lock:
            write_decision = self.capabilities.authorize_handle(pid, handle, ObjectRight.WRITE)
            if not write_decision.allowed:
                raise CapabilityDenied(write_decision.reason)
            current = self.store.get_object(handle.oid)
            if current is None:
                raise NotFound(f"object not found: {handle.oid}")
            if current.immutable:
                raise CapabilityDenied(f"immutable object cannot be updated: {handle.oid}")
            next_namespace = current.namespace
            next_name = current.name
            namespace_decisions = []
            if patch.namespace is not None:
                next_namespace = self._normalize_namespace(patch.namespace)
                namespace_decisions.append(self._require_namespace_right(pid, next_namespace, "write"))
                self._require_namespace_exists(next_namespace)
            if patch.name is not None:
                next_name = self._normalize_name(patch.name)
            if next_namespace != current.namespace or next_name != current.name:
                namespace_decisions.append(self._require_namespace_right(pid, current.namespace, "write"))
                self._require_unique_name(next_name, next_namespace, except_oid=current.oid)
            payload_is_set = patch.payload is not UNSET
            if payload_is_set:
                self._validate_payload_size(patch.payload, "object payload")
            next_payload = current.payload if not payload_is_set else patch.payload
            if patch.metadata is None:
                next_metadata = (
                    current.metadata
                    if not payload_is_set
                    else self._metadata_for_payload(next_payload, current.metadata, force_token_estimate=True)
                )
            else:
                next_metadata = self._metadata_for_payload(next_payload, patch.metadata)
            changed_fields: list[str] = []
            if payload_is_set:
                changed_fields.append("payload")
            if patch.metadata is not None:
                changed_fields.append("metadata")
            if patch.provenance is not None:
                changed_fields.append("provenance")
            if next_namespace != current.namespace:
                changed_fields.append("namespace")
            if next_name != current.name:
                changed_fields.append("name")
            updated = replace(
                current,
                namespace=next_namespace,
                name=next_name,
                payload=next_payload,
                metadata=next_metadata,
                provenance=current.provenance if patch.provenance is None else patch.provenance,
                version=current.version + 1,
                updated_at=utc_now(),
            )
            self._consume_one_time_decisions([write_decision, *namespace_decisions])
            self.store.update_object(updated)
        event = self.events.emit(
            EventType.OBJECT_UPDATED,
            source=pid,
            target=pid,
            payload={
                "oid": updated.oid,
                "namespace": updated.namespace,
                "name": updated.name,
                "qualified_name": self.qualified_name(updated),
                "version": updated.version,
            },
        )
        self.audit.record(
            actor=pid,
            action="memory.update_object",
            target=f"object:{updated.oid}",
            input_refs=[updated.oid],
            output_refs=[updated.oid],
            capability_refs=[handle.capability_id],
            decision={"namespace": updated.namespace, "name": updated.name, "version": updated.version},
        )
        self._notify_object_changed(
            updated.oid,
            {
                "event": "updated",
                "event_id": event.event_id,
                "version": updated.version,
                "change": {"operation": "patch", "fields": sorted(changed_fields)},
            },
            pid,
        )
        return handle

    def append_object_by_name(
        self,
        pid: str,
        name: str,
        entry: Any,
        list_field: str = "entries",
        namespace: str | None = None,
        *,
        issued_by: str = "memory.append",
    ) -> tuple[AgentObject, str | None, int]:
        with self.store._lock:
            object_namespace = self.resolve_namespace(pid, namespace)
            object_name = self._normalize_name(name)
            namespace_decision = self._require_namespace_right(pid, object_namespace, "read")
            self._require_namespace_exists(object_namespace)
            obj = self.store.get_object_by_name(object_name, namespace=object_namespace)
            if obj is None:
                raise NotFound(f"object not found: {self.qualified_name_parts(object_namespace, object_name)}")
            rights, decisions = self._authorized_object_rights(
                pid,
                obj.oid,
                required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
                optional_rights=set(),
            )
            if obj.immutable:
                raise CapabilityDenied(f"immutable object cannot be updated: {obj.oid}")
            ensure_json_size(entry, self.config.tools.memory_append_entry_max_bytes, "memory append entry")
            payload = deepcopy(obj.payload)
            if isinstance(payload, dict):
                values = payload.setdefault(list_field, [])
                if not isinstance(values, list):
                    raise ValidationError("target object list_field is not a list")
                values.append(entry)
                length = len(values)
                output_list_field: str | None = list_field
            elif isinstance(payload, list):
                payload.append(entry)
                length = len(payload)
                output_list_field = None
            else:
                raise ValidationError("target object payload is not appendable")
            self._validate_payload_size(payload, "memory payload")
            updated = replace(
                obj,
                payload=payload,
                metadata=self._metadata_for_payload(payload, obj.metadata, force_token_estimate=True),
                version=obj.version + 1,
                updated_at=utc_now(),
            )
            self._consume_one_time_decisions([*decisions, namespace_decision])
            self.store.update_object(updated)
        event = self.events.emit(
            EventType.OBJECT_UPDATED,
            source=pid,
            target=pid,
            payload={
                "oid": updated.oid,
                "namespace": updated.namespace,
                "name": updated.name,
                "qualified_name": self.qualified_name(updated),
                "version": updated.version,
            },
        )
        self.audit.record(
            actor=pid,
            action="memory.append_object",
            target=f"object:{updated.oid}",
            input_refs=[updated.oid],
            output_refs=[updated.oid],
            capability_refs=[
                cap_id for cap_id in (decision.selected_capability_id for decision in decisions) if cap_id is not None
            ],
            decision={
                "namespace": updated.namespace,
                "name": updated.name,
                "version": updated.version,
                "rights": sorted(rights),
                "issued_by": issued_by,
            },
        )
        self._notify_object_changed(
            updated.oid,
            {
                "event": "updated",
                "event_id": event.event_id,
                "version": updated.version,
                "change": {
                    "operation": "append",
                    "list_field": output_list_field,
                    "length": length,
                },
            },
            pid,
        )
        return updated, output_list_field, length

    def link_objects(
        self,
        pid: str,
        src: ObjectHandle,
        relation: RelationType | str,
        dst: ObjectHandle,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.capabilities.assert_handle(pid, src, ObjectRight.LINK)
        self.capabilities.assert_handle(pid, dst, ObjectRight.READ)
        link = ObjectLink(
            link_id=new_id("lnk"),
            src=src.oid,
            relation=RelationType(relation),
            dst=dst.oid,
            metadata=metadata or {},
            created_by=pid,
            created_at=utc_now(),
        )
        self.store.insert_link(link)
        event = self.events.emit(
            EventType.OBJECT_LINKED,
            source=pid,
            target=pid,
            payload={"src": src.oid, "relation": link.relation.value, "dst": dst.oid},
        )
        self.audit.record(
            actor=pid,
            action="memory.link_objects",
            target=f"object:{src.oid}",
            input_refs=[src.oid, dst.oid],
            capability_refs=[src.capability_id, dst.capability_id],
            decision={"relation": link.relation.value},
        )
        updated_src = self.store.get_object(src.oid)
        self._notify_object_changed(
            src.oid,
            {
                "event": "linked",
                "event_id": event.event_id,
                "version": updated_src.version if updated_src is not None else None,
                "relation": link.relation.value,
                "dst_oid": dst.oid,
                "link_id": link.link_id,
                "change": {"operation": "link"},
            },
            pid,
        )

    def link_objects_trusted(
        self,
        actor: str,
        src_oid: str,
        relation: RelationType | str,
        dst_oid: str,
        metadata: dict[str, Any] | None = None,
        *,
        reason: str,
    ) -> None:
        if self.store.get_object(src_oid) is None:
            raise NotFound(f"object not found: {src_oid}")
        if self.store.get_object(dst_oid) is None:
            raise NotFound(f"object not found: {dst_oid}")
        link = ObjectLink(
            link_id=new_id("lnk"),
            src=src_oid,
            relation=RelationType(relation),
            dst=dst_oid,
            metadata=metadata or {},
            created_by=actor,
            created_at=utc_now(),
        )
        self.store.insert_link(link)
        event = self.events.emit(
            EventType.OBJECT_LINKED,
            source=actor,
            target=actor,
            payload={"src": src_oid, "relation": link.relation.value, "dst": dst_oid},
        )
        self.audit.record(
            actor=actor,
            action="memory.link_objects_trusted",
            target=f"object:{src_oid}",
            input_refs=[src_oid, dst_oid],
            decision={"relation": link.relation.value, "reason": reason},
        )
        updated_src = self.store.get_object(src_oid)
        self._notify_object_changed(
            src_oid,
            {
                "event": "linked",
                "event_id": event.event_id,
                "version": updated_src.version if updated_src is not None else None,
                "relation": link.relation.value,
                "dst_oid": dst_oid,
                "link_id": link.link_id,
                "change": {"operation": "link"},
            },
            actor,
        )

    def query_objects(self, pid: str, query: ObjectQuery) -> list[ObjectHandle]:
        results: list[ObjectHandle] = []
        namespace = self.resolve_namespace(pid, query.namespace)
        namespace_decision = self._require_namespace_right(pid, namespace, "read")
        self._require_namespace_exists(namespace)
        limit = self._validate_query_limit(query.limit)
        if query.name is None:
            candidates = self.store.list_objects(namespace=namespace)
        else:
            object_name = self._normalize_name(query.name)
            obj = self.store.get_object_by_name(object_name, namespace=namespace)
            candidates = [] if obj is None else [obj]
        query_text = query.text.lower() if query.text else None
        for obj in candidates:
            if query.type is not None and obj.type.value != str(query.type):
                continue
            if query.tags and not set(query.tags).issubset(set(obj.metadata.tags)):
                continue
            if query_text and query_text not in self._search_text(obj).lower():
                continue
            decisions: list[Any]
            rights: set[str]
            try:
                rights, decisions = self._authorized_object_rights(
                    pid,
                    obj.oid,
                    required_rights={ObjectRight.READ.value},
                    optional_rights=set(),
                )
            except CapabilityDenied:
                continue
            handle = self.capabilities.handle_for_object(
                pid,
                obj.oid,
                rights,
                issued_by="memory.query",
                uses_remaining=1 if self._has_one_time_decision(decisions) else None,
            )
            self._consume_one_time_decisions(decisions)
            results.append(handle)
            if len(results) >= limit:
                break
        self.audit.record(
            actor=pid,
            action="memory.query_objects",
            target=f"object_namespace:{namespace}",
            output_refs=[handle.oid for handle in results],
            decision={"count": len(results), "namespace": namespace},
        )
        self._consume_one_time_decision(namespace_decision)
        return results

    def create_view(
        self,
        pid: str,
        roots: list[ObjectHandle],
        mode: ViewMode | str = ViewMode.READ_ONLY,
        filters: list[ObjectFilter] | None = None,
    ) -> MemoryView:
        view_mode = ViewMode(mode)
        for handle in roots:
            self.capabilities.assert_handle(pid, handle, ObjectRight.READ)
        view = MemoryView(
            view_id=new_id("view"),
            owner_pid=pid,
            roots=roots,
            filters=filters or [],
            rights_policy="attenuate" if view_mode == ViewMode.READ_ONLY else "inherit",
            created_from=None,
            mode=view_mode,
        )
        self.audit.record(
            actor=pid,
            action="memory.create_view",
            target=f"view:{view.view_id}",
            input_refs=[handle.oid for handle in roots],
            capability_refs=[handle.capability_id for handle in roots],
            decision={"mode": view.mode.value},
        )
        return view

    def fork_view(
        self,
        parent_pid: str,
        child_pid: str,
        parent_view: MemoryView,
        spec: MemoryViewSpec | None = None,
    ) -> MemoryView:
        spec = spec or MemoryViewSpec()
        source_roots = spec.roots if spec.roots is not None else parent_view.roots
        if not spec.include_parent_roots and spec.roots is None:
            source_roots = []
        child_roots: list[ObjectHandle] = []
        for handle in source_roots:
            rights, decisions = self._fork_child_rights(parent_pid, handle, spec)
            child_roots.append(
                self.capabilities.handle_for_object(
                    child_pid,
                    handle.oid,
                    rights,
                    issued_by=f"process:{parent_pid}:fork",
                    uses_remaining=1 if self._has_one_time_decision(decisions) else None,
                )
            )
            self._consume_one_time_decisions(decisions)
        view = MemoryView(
            view_id=new_id("view"),
            owner_pid=child_pid,
            roots=child_roots,
            filters=list(parent_view.filters),
            rights_policy="fork_attenuated",
            created_from=parent_view.view_id,
            mode=spec.mode,
        )
        self.audit.record(
            actor=parent_pid,
            action="memory.fork_view",
            target=f"view:{view.view_id}",
            input_refs=[handle.oid for handle in source_roots],
            output_refs=[handle.oid for handle in child_roots],
            capability_refs=[handle.capability_id for handle in child_roots],
            decision={"child_pid": child_pid, "mode": view.mode.value},
        )
        return view

    def merge_view(
        self,
        parent_pid: str,
        child_view: MemoryView,
        policy: MergePolicy | None = None,
    ) -> MergeResult:
        policy = policy or MergePolicy()
        merged: list[str] = []
        merged_handles: list[ObjectHandle] = []
        skipped: list[str] = []
        candidate_oids = {handle.oid for handle in child_view.roots}
        child_handles = {handle.oid: handle for handle in child_view.roots}
        if policy.include_child_created:
            candidate_oids.update(
                obj.oid
                for obj in self.store.list_objects_owned_by(ObjectOwnerKind.PROCESS, child_view.owner_pid)
            )
        for oid in sorted(candidate_oids):
            obj = self.store.get_object(oid)
            if obj is None:
                skipped.append(oid)
                continue
            try:
                if oid in child_handles:
                    rights, decisions = self._authorized_handle_rights(
                        child_view.owner_pid,
                        child_handles[oid],
                        required_rights={ObjectRight.READ.value},
                        optional_rights={str(right) for right in policy.grant_rights} - {ObjectRight.READ.value},
                        require_all=False,
                    )
                else:
                    rights, decisions = self._authorized_object_rights(
                        child_view.owner_pid,
                        oid,
                        required_rights={ObjectRight.READ.value},
                        optional_rights={str(right) for right in policy.grant_rights} - {ObjectRight.READ.value},
                    )
            except CapabilityDenied:
                skipped.append(oid)
                continue
            handle = self.capabilities.handle_for_object(
                parent_pid,
                oid,
                rights,
                issued_by=f"memory.merge:{child_view.owner_pid}",
                uses_remaining=1 if self._has_one_time_decision(decisions) else None,
            )
            self._consume_one_time_decisions(decisions)
            merged_handles.append(handle)
            merged.append(oid)
        self.audit.record(
            actor=parent_pid,
            action="memory.merge_view",
            target=f"view:{child_view.view_id}",
            input_refs=sorted(candidate_oids),
            output_refs=merged,
            decision={"merged": len(merged), "skipped": skipped},
        )
        return MergeResult(merged_oids=merged, skipped_oids=skipped, merged_handles=merged_handles)

    def snapshot_view(self, pid: str, view: MemoryView) -> str:
        snapshot_id = new_id("snap")
        self.audit.record(
            actor=pid,
            action="memory.snapshot_view",
            target=f"snapshot:{snapshot_id}",
            input_refs=[handle.oid for handle in view.roots],
            decision={"view_id": view.view_id},
        )
        return snapshot_id

    def release_process_owned(self, pid: str, preserve_oids: set[str] | None = None) -> list[str]:
        preserve = set(preserve_oids or set())
        if preserve:
            self.retain_as_process_result(pid, preserve)
        released = self.release_owner(
            ObjectOwnerKind.PROCESS,
            pid,
            preserve_oids=preserve,
            actor="memory",
            reason="process_owned_release",
        )
        released.extend(
            self.release_owner(
                ObjectOwnerKind.PROCESS_RESULT,
                pid,
                preserve_oids=preserve,
                actor="memory",
                reason="process_result_release",
            )
        )
        return released

    def release_owner(
        self,
        owner_kind: ObjectOwnerKind | str,
        owner_id: str,
        *,
        preserve_oids: set[str] | None = None,
        actor: str = "memory",
        reason: str = "owner_release",
    ) -> list[str]:
        preserve = set(preserve_oids or set())
        released: list[str] = []
        pinned: list[str] = []
        preserved: list[str] = []
        selected_owner_kind = ObjectOwnerKind(owner_kind)
        for oid in self.store.list_object_oids_owned_by(selected_owner_kind, owner_id):
            if self._is_object_pinned(oid):
                pinned.append(oid)
                continue
            if oid in preserve:
                preserved.append(oid)
                continue
            if self.delete_object_trusted(actor, oid, reason=reason):
                released.append(oid)
        if released or preserve or pinned:
            self.audit.record(
                actor=actor,
                action="memory.release_owner",
                target=f"object_owner:{selected_owner_kind.value}:{owner_id}",
                input_refs=released,
                output_refs=sorted(preserved or preserve),
                decision={
                    "owner_kind": selected_owner_kind.value,
                    "owner_id": owner_id,
                    "released": released,
                    "preserved": sorted(preserved or preserve),
                    "pinned": pinned,
                    "reason": reason,
                },
            )
        return released

    def delete_object_trusted(self, actor: str, oid: str, *, reason: str) -> bool:
        obj = self.store.get_object(oid)
        if obj is None:
            return False
        # Release finalizers bind host resources to Object Memory lifetimes.
        # They run before capability revocation so a failed cleanup cannot leave
        # an unreachable host handle alive.
        self._run_object_release_finalizers(obj, actor, reason)
        revoked = self.capabilities.revoke_resource_trusted(
            f"object:{oid}",
            revoked_by=actor,
            reason=f"object released: {reason}",
        )
        self.store.delete_object(oid)
        self.audit.record(
            actor=actor,
            action="memory.delete_object",
            target=f"object:{oid}",
            input_refs=[oid],
            capability_refs=[cap.cap_id for cap in revoked],
            decision={
                "reason": reason,
                "owner_kind": obj.owner_kind.value,
                "owner_id": obj.owner_id,
                "revoked_capabilities": len(revoked),
            },
        )
        return True

    def _is_object_pinned(self, oid: str) -> bool:
        if self._object_pin_checker is None:
            return False
        return bool(self._object_pin_checker(oid))

    def _run_object_release_finalizers(self, obj: AgentObject, actor: str, reason: str) -> None:
        for finalizer in list(self._object_release_finalizers):
            try:
                finalizer(obj, actor, reason)
            except Exception as exc:
                self.audit.record(
                    actor=actor,
                    action="memory.object_release_finalizer_failed",
                    target=f"object:{obj.oid}",
                    input_refs=[obj.oid],
                    decision={"reason": reason, "error_type": type(exc).__name__, "error": str(exc)},
                )
                raise

    def _notify_object_changed(self, oid: str, change: dict[str, Any], actor_pid: str) -> None:
        if self._object_change_notifier is None:
            return
        try:
            self._object_change_notifier(oid, change, actor_pid)
        except Exception as exc:
            self.audit.record(
                actor="memory",
                action="memory.object_change_notify_failed",
                target=f"object:{oid}",
                input_refs=[oid],
                decision={"actor_pid": actor_pid, "change": change, "error": str(exc)},
            )

    def preserve_process_owned(self, pid: str, oids: Iterable[str]) -> list[str]:
        return self.retain_as_process_result(pid, oids)

    def retain_as_process_result(self, pid: str, oids: Iterable[str]) -> list[str]:
        return self.transfer_owner(
            ObjectOwnerKind.PROCESS,
            pid,
            ObjectOwnerKind.PROCESS_RESULT,
            pid,
            oids,
            actor="memory",
            reason="process_result",
        )

    def adopt_process_owned(self, from_pid: str, to_pid: str, oids: Iterable[str]) -> list[str]:
        selected_oids = sorted(set(oids))
        adopted = self.transfer_owner(
            ObjectOwnerKind.PROCESS,
            from_pid,
            ObjectOwnerKind.PROCESS,
            to_pid,
            selected_oids,
            actor="memory",
            reason="process_adopt",
        )
        adopted.extend(
            self.transfer_owner(
                ObjectOwnerKind.PROCESS_RESULT,
                from_pid,
                ObjectOwnerKind.PROCESS,
                to_pid,
                selected_oids,
                actor="memory",
                reason="process_result_adopt",
            )
        )
        return adopted

    def transfer_owner(
        self,
        from_owner_kind: ObjectOwnerKind | str,
        from_owner_id: str,
        to_owner_kind: ObjectOwnerKind | str,
        to_owner_id: str,
        oids: Iterable[str],
        *,
        actor: str = "memory",
        reason: str = "owner_transfer",
    ) -> list[str]:
        transferred: list[str] = []
        selected_from_kind = ObjectOwnerKind(from_owner_kind)
        selected_to_kind = ObjectOwnerKind(to_owner_kind)
        for oid in sorted(set(oids)):
            obj = self.store.get_object(oid)
            if obj is None or obj.owner_kind != selected_from_kind or obj.owner_id != from_owner_id:
                continue
            self.store.update_object(
                replace(
                    obj,
                    owner_kind=selected_to_kind,
                    owner_id=to_owner_id,
                    updated_at=utc_now(),
                )
            )
            transferred.append(oid)
        if transferred:
            self.audit.record(
                actor=actor,
                action="memory.transfer_owner",
                target=f"object_owner:{selected_from_kind.value}:{from_owner_id}",
                input_refs=transferred,
                output_refs=transferred,
                decision={
                    "from_owner_kind": selected_from_kind.value,
                    "from_owner_id": from_owner_id,
                    "to_owner_kind": selected_to_kind.value,
                    "to_owner_id": to_owner_id,
                    "transferred": transferred,
                    "reason": reason,
                },
            )
        return transferred

    def lifetime_scope(
        self,
        *,
        actor: str,
        owner_kind: ObjectOwnerKind | str,
        owner_id: str,
        reason: str,
    ) -> ObjectLifetimeScope:
        return ObjectLifetimeScope(
            self,
            actor=actor,
            owner_kind=owner_kind,
            owner_id=owner_id,
            reason=reason,
        )

    def materialize_context(
        self,
        pid: str,
        view: MemoryView,
        policy: str | None = None,
        budget_tokens: int | None = None,
        charge_resources: bool = True,
    ) -> MaterializedContext:
        selected_policy = policy or self.config.memory.context_policy
        selected_budget = budget_tokens if budget_tokens is not None else self.config.memory.materialize_budget_tokens
        resources = getattr(self, "resources", None)
        if resources is not None:
            selected_budget = min(selected_budget, resources.context_materialization_window_limit(pid))
            remaining = resources.remaining_cumulative(
                pid,
                "max_context_materialization_total_tokens",
                "context_materialized_tokens",
            )
            if remaining is not None:
                selected_budget = min(selected_budget, max(0, int(remaining)))
        objects: list[AgentObject] = []
        omitted: list[str] = []
        filtered: list[str] = []
        for handle in view.roots:
            try:
                self.capabilities.assert_handle(pid, handle, ObjectRight.MATERIALIZE)
                obj = self.store.get_object(handle.oid)
                if obj is None:
                    continue
                if not self._matches_view_filters(obj, view.filters):
                    omitted.append(obj.oid)
                    filtered.append(obj.oid)
                else:
                    objects.append(obj)
            except CapabilityDenied:
                omitted.append(handle.oid)

        objects = self._sort_for_policy(objects, selected_policy)
        chunks: list[str] = []
        refs: list[str] = []
        total = 0
        for obj in objects:
            tokens = obj.metadata.token_estimate or estimate_tokens(obj.payload)
            if total + tokens > selected_budget:
                omitted.append(obj.oid)
                continue
            chunks.append(self._render_object(obj))
            refs.append(obj.oid)
            total += tokens
        context = MaterializedContext(
            text="\n\n".join(chunks),
            object_refs=refs,
            token_count=total,
            omitted_objects=omitted,
            policy_used=selected_policy,
        )
        self.audit.record(
            actor=pid,
            action="memory.materialize_context",
            target=f"view:{view.view_id}",
            input_refs=[handle.oid for handle in view.roots],
            output_refs=refs,
            decision={
                "tokens": total,
                "omitted": omitted,
                "filtered": filtered,
                "policy": selected_policy,
                "charged": charge_resources,
            },
        )
        if resources is not None and charge_resources:
            resources.charge(
                pid,
                ResourceUsage(context_materialized_tokens=total),
                source="memory.materialize_context",
                context={"view_id": view.view_id, "policy": selected_policy},
                allow_overage=False,
                kill_on_exceed=False,
            )
        return context

    def _matches_view_filters(self, obj: AgentObject, filters: list[ObjectFilter]) -> bool:
        if not filters:
            return True
        return any(self._matches_filter(obj, item) for item in filters)

    def _matches_filter(self, obj: AgentObject, item: ObjectFilter) -> bool:
        if item.type is not None and obj.type != ObjectType(item.type):
            return False
        if item.tags and not set(item.tags).issubset(set(obj.metadata.tags)):
            return False
        if item.text and item.text.lower() not in self._search_text(obj).lower():
            return False
        return True

    def _search_text(self, obj: AgentObject) -> str:
        payload_preview = self._bounded_payload_repr(obj.payload)
        return " ".join(
            [
                obj.namespace,
                obj.name,
                obj.metadata.title or "",
                obj.metadata.summary or "",
                " ".join(obj.metadata.tags),
                payload_preview,
            ]
        )

    def _bounded_payload_repr(self, payload: Any) -> str:
        renderer = Repr()
        # Text search is a lightweight directory aid, not a full payload index.
        # Keep representation bounded so a query cannot render every large
        # Object Memory payload in the namespace.
        renderer.maxstring = self.config.tools.memory_payload_chars
        renderer.maxother = self.config.tools.memory_payload_chars
        renderer.maxlist = 50
        renderer.maxdict = 50
        render = renderer.repr(payload)
        return render[: self.config.tools.memory_payload_chars]

    def _sort_for_policy(self, objects: list[AgentObject], policy: str) -> list[AgentObject]:
        if policy == "recency_first":
            return sorted(objects, key=lambda obj: obj.updated_at, reverse=True)
        if policy == "evidence_first":
            return sorted(objects, key=lambda obj: obj.type != ObjectType.EVIDENCE)
        if policy == "plan_first":
            priority = {ObjectType.GOAL: 0, ObjectType.TASK: 1, ObjectType.PLAN: 2, ObjectType.STEP: 3}
            return sorted(objects, key=lambda obj: priority.get(obj.type, 10))
        if policy == "error_debug":
            priority = {ObjectType.ERROR_TRACE: 0, ObjectType.TEST_RESULT: 1, ObjectType.CODE_PATCH: 2}
            return sorted(objects, key=lambda obj: priority.get(obj.type, 10))
        return objects

    def _consume_one_time_decision(self, decision: Any) -> None:
        if decision.consume_capability_id is None:
            return
        self.capabilities.consume_use(
            decision.consume_capability_id,
            used_by="object_memory",
            reason="one-time object memory permission consumed",
        )

    def _consume_one_time_decisions(self, decisions: Iterable[Any]) -> None:
        consumed: set[str] = set()
        for decision in decisions:
            cap_id = decision.consume_capability_id
            if cap_id is None or cap_id in consumed:
                continue
            consumed.add(cap_id)
            self._consume_one_time_decision(decision)

    def _has_one_time_decision(self, decisions: Iterable[Any]) -> bool:
        return any(decision.consume_capability_id is not None for decision in decisions)

    def _authorized_object_rights(
        self,
        pid: str,
        oid: str,
        *,
        required_rights: Iterable[str | ObjectRight],
        optional_rights: Iterable[str | ObjectRight] = (),
    ) -> tuple[set[str], list[Any]]:
        rights: set[str] = set()
        decisions: list[Any] = []
        resource = f"object:{oid}"
        for right in sorted({str(item) for item in required_rights}):
            decision = self.capabilities.authorize(pid, resource, right)
            if not decision.allowed:
                raise CapabilityDenied(decision.reason)
            rights.add(right)
            decisions.append(decision)
        for right in sorted({str(item) for item in optional_rights} - rights):
            decision = self.capabilities.authorize(pid, resource, right)
            if decision.allowed:
                rights.add(right)
                decisions.append(decision)
        if not rights:
            raise CapabilityDenied(f"{pid} lacks object rights on {oid}")
        return rights, decisions

    def _authorized_handle_rights(
        self,
        pid: str,
        handle: ObjectHandle,
        *,
        required_rights: Iterable[str | ObjectRight],
        optional_rights: Iterable[str | ObjectRight] = (),
        require_all: bool = True,
    ) -> tuple[set[str], list[Any]]:
        rights: set[str] = set()
        decisions: list[Any] = []
        for right in sorted({str(item) for item in required_rights}):
            decision = self.capabilities.authorize_handle(pid, handle, right)
            if not decision.allowed:
                raise CapabilityDenied(decision.reason)
            rights.add(right)
            decisions.append(decision)
        for right in sorted({str(item) for item in optional_rights} - rights):
            if right not in handle.rights:
                if require_all:
                    raise CapabilityDenied(f"object handle lacks {right}: {handle.oid}")
                continue
            decision = self.capabilities.authorize_handle(pid, handle, right)
            if decision.allowed:
                rights.add(right)
                decisions.append(decision)
            elif require_all:
                raise CapabilityDenied(decision.reason)
        if not rights:
            raise CapabilityDenied(f"{pid} lacks object handle rights on {handle.oid}")
        return rights, decisions

    def _fork_child_rights(
        self,
        parent_pid: str,
        handle: ObjectHandle,
        spec: MemoryViewSpec,
    ) -> tuple[set[str], list[Any]]:
        if spec.rights is not None:
            requested = {str(right) for right in spec.rights}
            requested.add(ObjectRight.READ.value)
            missing = requested - {str(right) for right in handle.rights}
            if missing:
                raise CapabilityDenied(
                    f"forked MemoryView cannot grant rights absent from parent handle: {sorted(missing)}"
                )
            return self._authorized_handle_rights(
                parent_pid,
                handle,
                required_rights=requested,
                optional_rights=set(),
                require_all=True,
            )

        optional = {ObjectRight.MATERIALIZE.value, ObjectRight.DIFF.value}
        if spec.mode in {ViewMode.MUTABLE, ViewMode.COPY_ON_WRITE}:
            optional.add(ObjectRight.WRITE.value)
        # Forking is attenuation, not capability minting: optional rights are
        # inherited only when the parent handle itself and current policy allow
        # them. A read-only root therefore remains read-only in the child.
        return self._authorized_handle_rights(
            parent_pid,
            handle,
            required_rights={ObjectRight.READ.value},
            optional_rights=optional,
            require_all=False,
        )

    def _render_object(self, obj: AgentObject) -> str:
        title = f" title={obj.metadata.title!r}" if obj.metadata.title else ""
        summary = f"\nsummary: {obj.metadata.summary}" if obj.metadata.summary else ""
        return (
            f"[{obj.oid}] namespace={obj.namespace!r} name={obj.name!r} "
            f"qualified_name={self.qualified_name(obj)!r} type={obj.type.value} "
            f"version={obj.version}{title}{summary}\npayload: {obj.payload!r}"
        )

    def qualified_name(self, obj: AgentObject) -> str:
        return self.qualified_name_parts(obj.namespace, obj.name)

    def qualified_name_parts(self, namespace: str, name: str) -> str:
        return f"{namespace}/{name}"

    def _default_name(self, object_type: ObjectType, oid: str) -> str:
        return f"{object_type.value}:{oid}"

    def _normalize_namespace(self, namespace: str) -> str:
        normalized = namespace.strip().replace("\\", "/").strip("/")
        if not normalized:
            raise ValidationError("Object Memory namespace must be non-empty")
        segments = normalized.split("/")
        if any(not segment or segment in {".", ".."} or segment.strip() != segment for segment in segments):
            raise ValidationError(f"invalid Object Memory namespace: {namespace}")
        return normalized

    def _normalize_name(self, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValidationError("object name must be non-empty")
        # Names are a single directory entry inside a namespace. Keeping path
        # separators out makes qualified_name_parts(namespace, name) a stable
        # display and audit identifier instead of an ambiguous pseudo-path.
        if "/" in normalized or "\\" in normalized:
            raise ValidationError("object name must not contain namespace separators")
        if normalized in {".", ".."}:
            raise ValidationError(f"invalid object name: {name}")
        return normalized

    def _parent_namespace(self, namespace: str) -> str | None:
        if "/" not in namespace:
            return None
        return namespace.rsplit("/", 1)[0]

    def _require_namespace_exists(self, namespace: str) -> None:
        if not self.store.namespace_exists(namespace):
            raise NotFound(f"Object Memory namespace not found: {namespace}")

    def _namespace_resource(self, namespace: str) -> str:
        return f"object_namespace:{namespace}"

    def _require_namespace_right(
        self,
        pid: str,
        namespace: str,
        right: str,
    ) -> Any:
        # Namespace checks protect directory-style lookup and mutation. The
        # caller consumes returned one-shot decisions only after the operation
        # succeeds, so failed validation does not burn an approval.
        return self.capabilities.require(pid, self._namespace_resource(namespace), right)

    def _can_read_namespace(self, pid: str, namespace: str) -> bool:
        return self.capabilities.check(
            pid,
            self._namespace_resource(namespace),
            "read",
        )

    def _require_unique_name(self, name: str, namespace: str, except_oid: str | None = None) -> None:
        if self.store.object_name_exists(name, except_oid=except_oid, namespace=namespace):
            raise ValidationError(f"object name already exists in namespace {namespace}: {name}")

    def _validate_payload_size(self, payload: Any, label: str) -> None:
        ensure_json_size(payload, self.config.tools.memory_payload_hard_limit_bytes, label)

    def _validate_query_limit(self, limit: int) -> int:
        selected = int(limit)
        if selected < 1:
            raise ValidationError("Object Memory query limit must be >= 1")
        if selected > self.config.memory.query_limit:
            raise ValidationError(f"Object Memory query limit must be <= {self.config.memory.query_limit}")
        return selected

    def _metadata_for_payload(
        self,
        payload: Any,
        metadata: ObjectMetadata | None,
        *,
        force_token_estimate: bool = False,
    ) -> ObjectMetadata:
        meta = deepcopy(metadata) if metadata is not None else ObjectMetadata()
        if force_token_estimate or meta.token_estimate is None:
            meta.token_estimate = estimate_tokens(payload)
        return meta
