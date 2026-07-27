from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.process_state import (
    legacy_status_message,
    process_outcome_from_json,
    process_outcome_to_mapping,
    process_wait_state_from_json,
    process_wait_state_to_mapping,
    remap_process_outcome,
    remap_process_wait_state,
)
from agent_libos.runtime.snapshots.models import ProcessSnapshot, SnapshotHeader, SnapshotRows
from agent_libos.utils.serde import bounded_json_loads, dumps


@dataclass(frozen=True)
class SnapshotIdentityMap:
    pids: Mapping[str, str] = field(default_factory=dict)
    objects: Mapping[str, str] = field(default_factory=dict)
    namespaces: Mapping[str, str] = field(default_factory=dict)
    capabilities: Mapping[str, str] = field(default_factory=dict)
    tools: Mapping[str, str] = field(default_factory=dict)
    candidates: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pids", "objects", "namespaces", "capabilities", "tools", "candidates"):
            selected = dict(getattr(self, name))
            if len(selected.values()) != len(set(selected.values())):
                raise ValidationError(f"snapshot identity map {name} must be one-to-one")
            if any(not str(source) or not str(target) for source, target in selected.items()):
                raise ValidationError(f"snapshot identity map {name} contains an empty id")


class SnapshotRemapper:
    """Pure remapping helpers shared by fork, exec rollback, and image commit."""

    _NESTED_JSON_MAX_BYTES = 1_048_576

    _FIELD_MAPS = {
        "pid": "pids",
        "parent_pid": "pids",
        "subject": "pids",
        "creator_pid": "pids",
        "recipient_pid": "pids",
        "sender": "pids",
        "sender_pid": "pids",
        "owner_pid": "pids",
        "runner_pid": "pids",
        "child_pid": "pids",
        "created_by": "pids",
        "oid": "objects",
        "src_oid": "objects",
        "dst_oid": "objects",
        "result_oid": "objects",
        "goal_oid": "objects",
        "namespace": "namespaces",
        "parent_namespace": "namespaces",
        "cap_id": "capabilities",
        "parent_cap_id": "capabilities",
        "issuer_cap_id": "capabilities",
        "capability_id": "capabilities",
        "tool_id": "tools",
        "registered_tool_id": "tools",
        "candidate_id": "candidates",
    }
    _PRIMARY_ROW_IDENTITIES = {
        "pids": ("processes", "pid"),
        "objects": ("objects", "oid"),
        "namespaces": ("object_namespaces", "namespace"),
        "capabilities": ("capabilities", "cap_id"),
        "tools": ("tools", "tool_id"),
        "candidates": ("tool_candidates", "candidate_id"),
    }
    _RESOURCE_MAPS = (
        ("checkpoint:process:", "pids"),
        ("object_namespace:", "namespaces"),
        ("process:", "pids"),
        ("object:", "objects"),
        ("tool:", "tools"),
    )

    @classmethod
    def remap_row(cls, row: Mapping[str, Any], identities: SnapshotIdentityMap) -> dict[str, Any]:
        remapped = deepcopy(dict(row))
        for field_name, map_name in cls._FIELD_MAPS.items():
            value = remapped.get(field_name)
            selected_map = getattr(identities, map_name)
            if value is not None and str(value) in selected_map:
                remapped[field_name] = selected_map[str(value)]
        if "wait_state_json" in remapped and "outcome_json" in remapped:
            wait_state = remap_process_wait_state(
                process_wait_state_from_json(remapped["wait_state_json"]),
                pids=identities.pids,
                objects=identities.objects,
            )
            outcome = remap_process_outcome(
                process_outcome_from_json(remapped["outcome_json"]),
                objects=identities.objects,
            )
            remapped["wait_state_json"] = dumps(
                process_wait_state_to_mapping(wait_state)
            )
            remapped["outcome_json"] = dumps(process_outcome_to_mapping(outcome))
            remapped["status_message"] = legacy_status_message(
                wait_state,
                outcome,
                remapped.get("status_message"),
            )
        return remapped

    @classmethod
    def remap_rows(cls, rows: SnapshotRows, identities: SnapshotIdentityMap) -> SnapshotRows:
        return SnapshotRows(
            **{
                table: tuple(
                    cls._remap_nested_row(
                        table,
                        cls.remap_row(row, identities),
                        identities,
                    )
                    for row in getattr(rows, table)
                )
                for table in SnapshotRows.TABLES
            }
        )

    @classmethod
    def remap(cls, snapshot: ProcessSnapshot, identities: SnapshotIdentityMap) -> ProcessSnapshot:
        cls._validate_identity_collisions(snapshot, identities)
        root_pid = identities.pids.get(snapshot.header.root_pid, snapshot.header.root_pid)
        remapped = ProcessSnapshot(
            header=SnapshotHeader(
                schema_version=snapshot.header.schema_version,
                checkpoint_id=snapshot.header.checkpoint_id,
                root_pid=root_pid,
                reason=snapshot.header.reason,
                created_at=snapshot.header.created_at,
                created_by=snapshot.header.created_by,
            ),
            subtree_pids=tuple(identities.pids.get(pid, pid) for pid in snapshot.subtree_pids),
            object_oids=tuple(identities.objects.get(oid, oid) for oid in snapshot.object_oids),
            owned_object_oids=tuple(identities.objects.get(oid, oid) for oid in snapshot.owned_object_oids),
            referenced_object_oids=tuple(
                identities.objects.get(oid, oid) for oid in snapshot.referenced_object_oids
            ),
            referenced_object_types={
                identities.objects.get(oid, oid): object_type
                for oid, object_type in snapshot.referenced_object_types.items()
            },
            namespaces=tuple(identities.namespaces.get(name, name) for name in snapshot.namespaces),
            owned_namespaces=tuple(
                identities.namespaces.get(name, name) for name in snapshot.owned_namespaces
            ),
            rows=cls.remap_rows(snapshot.rows, identities),
            object_payloads={
                identities.objects.get(oid, oid): deepcopy(payload)
                for oid, payload in snapshot.object_payloads.items()
            },
            images=deepcopy(snapshot.images),
            image_artifacts=deepcopy(snapshot.image_artifacts),
            jit_sources={
                identities.tools.get(tool_id, tool_id): source
                for tool_id, source in snapshot.jit_sources.items()
            },
            modules=tuple(deepcopy(module) for module in snapshot.modules),
        )
        cls._validate_remapped_cardinality(snapshot, remapped)
        cls._validate_remapped_references(snapshot, remapped, identities)
        return remapped

    @classmethod
    def _remap_nested_row(
        cls,
        table: str,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> dict[str, Any]:
        if table == "processes":
            cls._remap_process_carriers(row, identities)
        elif table == "objects":
            cls._remap_object_carriers(row, identities)
        elif table == "capabilities":
            cls._remap_capability_resource(row, identities)
        elif table == "process_messages":
            cls._remap_message_carriers(row, identities)
        elif table == "tool_candidates":
            cls._remap_candidate_carriers(row, identities)
        return row

    @classmethod
    def _remap_process_carriers(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        cls._remap_process_capability_index(row, identities.capabilities)
        cls._remap_process_tool_tables(row, identities.tools)
        if identities.tools:
            cls._remap_loaded_skill_tools(row, identities.tools)
        cls._remap_process_memory_view(row, identities)

    @classmethod
    def _remap_process_capability_index(
        cls,
        row: dict[str, Any],
        capability_map: Mapping[str, str],
    ) -> None:
        if not capability_map or row.get("capabilities_json") is None:
            return
        capability_ids = cls._json_container(
            row["capabilities_json"],
            field_name="processes.capabilities_json",
            expected_type=list,
        )
        if any(not isinstance(cap_id, str) or not cap_id for cap_id in capability_ids):
            raise ValidationError(
                "snapshot processes.capabilities_json must contain capability ids"
            )
        remapped = [capability_map.get(cap_id, cap_id) for cap_id in capability_ids]
        if remapped != capability_ids:
            row["capabilities_json"] = dumps(remapped)

    @classmethod
    def _remap_process_tool_tables(
        cls,
        row: dict[str, Any],
        tool_map: Mapping[str, str],
    ) -> None:
        if not tool_map:
            return
        for field_name in ("tool_table_json", "model_tool_table_json"):
            if row.get(field_name) is None:
                continue
            table = cls._json_container(
                row[field_name],
                field_name=f"processes.{field_name}",
                expected_type=dict,
            )
            if any(
                not isinstance(name, str)
                or not isinstance(tool_id, str)
                or not tool_id
                for name, tool_id in table.items()
            ):
                raise ValidationError(
                    f"snapshot processes.{field_name} must be a string mapping"
                )
            remapped_table = {
                name: tool_map.get(tool_id, tool_id)
                for name, tool_id in table.items()
            }
            if remapped_table != table:
                row[field_name] = dumps(remapped_table)

    @classmethod
    def _remap_process_memory_view(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        if (
            identities.pids
            or identities.objects
            or identities.capabilities
        ) and row.get("memory_view_json") is None:
            return
        if not (identities.pids or identities.objects or identities.capabilities):
            return
        view = cls._json_container(
            row["memory_view_json"],
            field_name="processes.memory_view_json",
            expected_type=dict,
        )
        remapped_view = deepcopy(view)
        cls._remap_memory_view_owner(remapped_view, identities.pids)
        cls._remap_memory_view_roots(remapped_view, identities)
        if remapped_view != view:
            row["memory_view_json"] = dumps(remapped_view)

    @staticmethod
    def _remap_memory_view_owner(
        view: dict[str, Any],
        pid_map: Mapping[str, str],
    ) -> None:
        owner_pid = view.get("owner_pid")
        if isinstance(owner_pid, str):
            view["owner_pid"] = pid_map.get(owner_pid, owner_pid)

    @classmethod
    def _remap_memory_view_roots(
        cls,
        view: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        roots = view.get("roots", [])
        if not isinstance(roots, list) or any(
            not isinstance(root, dict) for root in roots
        ):
            raise ValidationError(
                "snapshot processes.memory_view_json roots must be a list of objects"
            )
        for root in roots:
            oid = root.get("oid")
            capability_id = root.get("capability_id")
            if isinstance(oid, str):
                root["oid"] = identities.objects.get(oid, oid)
            if isinstance(capability_id, str):
                root["capability_id"] = identities.capabilities.get(
                    capability_id,
                    capability_id,
                )

    @classmethod
    def _remap_loaded_skill_tools(
        cls,
        row: dict[str, Any],
        tool_map: Mapping[str, str],
    ) -> None:
        raw = row.get("loaded_skills_json")
        if raw is None:
            return
        loaded_skills = cls._json_container(
            raw,
            field_name="processes.loaded_skills_json",
            expected_type=dict,
        )
        remapped_skills = deepcopy(loaded_skills)
        for skill_id, loaded in remapped_skills.items():
            if not isinstance(skill_id, str) or not isinstance(loaded, dict):
                raise ValidationError(
                    "snapshot processes.loaded_skills_json must map skill ids to objects"
                )
            for field_name in (
                "tool_ids",
                "jit_tool_ids",
                "base_tool_ids",
                "base_model_tool_ids",
            ):
                identifiers = loaded.get(field_name)
                if identifiers is None:
                    continue
                if not isinstance(identifiers, dict) or any(
                    not isinstance(name, str)
                    or not isinstance(tool_id, str)
                    or not tool_id
                    for name, tool_id in identifiers.items()
                ):
                    raise ValidationError(
                        "snapshot processes.loaded_skills_json "
                        f"{field_name} must be a string mapping"
                    )
                loaded[field_name] = {
                    name: tool_map.get(tool_id, tool_id)
                    for name, tool_id in identifiers.items()
                }
        if remapped_skills != loaded_skills:
            row["loaded_skills_json"] = dumps(remapped_skills)

    @classmethod
    def _remap_object_carriers(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        if identities.pids and row.get("owner_kind") in {
            "process",
            "process_result",
        }:
            owner_id = row.get("owner_id")
            if isinstance(owner_id, str):
                row["owner_id"] = identities.pids.get(owner_id, owner_id)
        if identities.objects and row.get("provenance_json") is not None:
            provenance = cls._json_container(
                row["provenance_json"],
                field_name="objects.provenance_json",
                expected_type=dict,
            )
            parent_oids = provenance.get("parent_oids", [])
            if not isinstance(parent_oids, list) or any(
                not isinstance(oid, str) or not oid for oid in parent_oids
            ):
                raise ValidationError(
                    "snapshot objects.provenance_json parent_oids must be a list of ids"
                )
            remapped = deepcopy(provenance)
            remapped["parent_oids"] = [
                identities.objects.get(oid, oid) for oid in parent_oids
            ]
            if remapped != provenance:
                row["provenance_json"] = dumps(remapped)

    @classmethod
    def _remap_capability_resource(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        resource = row.get("resource")
        if not isinstance(resource, str):
            return
        row["resource"] = cls._expected_resource(resource, identities)

    @classmethod
    def _expected_resource(
        cls,
        resource: str,
        identities: SnapshotIdentityMap,
    ) -> str:
        for prefix, map_name in cls._RESOURCE_MAPS:
            if not resource.startswith(prefix):
                continue
            identity = resource[len(prefix) :]
            selected_map = getattr(identities, map_name)
            if identity in selected_map:
                return f"{prefix}{selected_map[identity]}"
            return resource
        return resource

    @classmethod
    def _remap_message_carriers(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        if not identities.objects or row.get("metadata_json") is None:
            return
        metadata = cls._json_container(
            row["metadata_json"],
            field_name="process_messages.metadata_json",
            expected_type=dict,
        )
        carrier_oid = metadata.get("label_carrier_oid")
        if not isinstance(carrier_oid, str) or carrier_oid not in identities.objects:
            return
        remapped = deepcopy(metadata)
        remapped["label_carrier_oid"] = identities.objects[carrier_oid]
        row["metadata_json"] = dumps(remapped)

    @classmethod
    def _remap_candidate_carriers(
        cls,
        row: dict[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        raw = row.get("requested_capabilities_json")
        if raw is None or not any(
            (
                identities.pids,
                identities.objects,
                identities.namespaces,
                identities.tools,
            )
        ):
            return
        requested = cls._json_container(
            raw,
            field_name="tool_candidates.requested_capabilities_json",
            expected_type=list,
        )
        if any(not isinstance(item, dict) for item in requested):
            raise ValidationError(
                "snapshot tool_candidates.requested_capabilities_json "
                "must be a list of objects"
            )
        remapped = deepcopy(requested)
        for item in remapped:
            cls._remap_capability_resource(item, identities)
        if remapped != requested:
            row["requested_capabilities_json"] = dumps(remapped)

    @staticmethod
    def _json_container(
        raw: Any,
        *,
        field_name: str,
        expected_type: type[list[Any]] | type[dict[str, Any]],
    ) -> list[Any] | dict[str, Any]:
        if not isinstance(raw, str):
            raise ValidationError(f"snapshot {field_name} must be canonical JSON text")
        try:
            decoded = bounded_json_loads(
                raw,
                max_bytes=SnapshotRemapper._NESTED_JSON_MAX_BYTES,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"snapshot {field_name} contains malformed JSON") from exc
        if type(decoded) is not expected_type:
            expected = "a list" if expected_type is list else "an object"
            raise ValidationError(f"snapshot {field_name} must decode to {expected}")
        return decoded

    @classmethod
    def _validate_identity_collisions(
        cls,
        snapshot: ProcessSnapshot,
        identities: SnapshotIdentityMap,
    ) -> None:
        sources = cls._snapshot_identity_sources(snapshot)
        for map_name, source_ids in sources.items():
            selected_map = getattr(identities, map_name)
            transformed = [selected_map.get(source, source) for source in source_ids]
            if len(transformed) != len(set(transformed)):
                raise ValidationError(
                    "snapshot identity map "
                    f"{map_name} collides with an unchanged snapshot identity"
                )

    @classmethod
    def _snapshot_identity_sources(
        cls,
        snapshot: ProcessSnapshot,
    ) -> dict[str, set[str]]:
        selected: dict[str, set[str]] = {
            "pids": set(snapshot.subtree_pids),
            "objects": {
                *snapshot.object_oids,
                *snapshot.owned_object_oids,
                *snapshot.referenced_object_oids,
                *snapshot.referenced_object_types,
                *snapshot.object_payloads,
            },
            "namespaces": {
                *snapshot.namespaces,
                *snapshot.owned_namespaces,
            },
            "capabilities": set(),
            "tools": set(snapshot.jit_sources),
            "candidates": set(),
        }
        for map_name, (table, field_name) in cls._PRIMARY_ROW_IDENTITIES.items():
            selected[map_name].update(
                str(value)
                for row in getattr(snapshot.rows, table)
                if (value := row.get(field_name)) is not None and str(value)
            )
        return selected

    @classmethod
    def _validate_remapped_cardinality(
        cls,
        source: ProcessSnapshot,
        remapped: ProcessSnapshot,
    ) -> None:
        for field_name in (
            "subtree_pids",
            "object_oids",
            "owned_object_oids",
            "referenced_object_oids",
            "namespaces",
            "owned_namespaces",
        ):
            before = getattr(source, field_name)
            after = getattr(remapped, field_name)
            if len(after) != len(before) or len(after) != len(set(after)):
                raise ValidationError(
                    f"snapshot remap lost or duplicated {field_name} identities"
                )
        for field_name in (
            "referenced_object_types",
            "object_payloads",
            "jit_sources",
        ):
            if len(getattr(source, field_name)) != len(getattr(remapped, field_name)):
                raise ValidationError(
                    f"snapshot remap lost {field_name} entries"
                )
        for map_name, (table, field_name) in cls._PRIMARY_ROW_IDENTITIES.items():
            del map_name
            source_rows = getattr(source.rows, table)
            remapped_rows = getattr(remapped.rows, table)
            identities = [row.get(field_name) for row in remapped_rows]
            if len(remapped_rows) != len(source_rows) or len(identities) != len(
                set(identities)
            ):
                raise ValidationError(
                    f"snapshot remap lost or duplicated rows.{table}.{field_name} identities"
                )

    @classmethod
    def _validate_remapped_references(
        cls,
        source: ProcessSnapshot,
        remapped: ProcessSnapshot,
        identities: SnapshotIdentityMap,
    ) -> None:
        cls._validate_top_level_references(source, remapped, identities)
        for table in SnapshotRows.TABLES:
            source_rows = getattr(source.rows, table)
            remapped_rows = getattr(remapped.rows, table)
            for index, (source_row, remapped_row) in enumerate(
                zip(source_rows, remapped_rows)
            ):
                cls._validate_direct_row_references(
                    table,
                    index,
                    source_row,
                    remapped_row,
                    identities,
                )
                if table == "processes":
                    cls._validate_process_references(
                        index,
                        source_row,
                        remapped_row,
                        identities,
                    )
                elif table == "objects":
                    cls._validate_object_references(
                        index,
                        source_row,
                        remapped_row,
                        identities,
                    )
                elif table == "capabilities":
                    resource = source_row.get("resource")
                    if isinstance(resource, str) and remapped_row.get(
                        "resource"
                    ) != cls._expected_resource(resource, identities):
                        cls._reference_error(table, index, "resource")
                elif table == "process_messages":
                    cls._validate_message_references(
                        index,
                        source_row,
                        remapped_row,
                        identities,
                    )
                elif table == "tool_candidates":
                    cls._validate_candidate_references(
                        index,
                        source_row,
                        remapped_row,
                        identities,
                    )

    @classmethod
    def _validate_top_level_references(
        cls,
        source: ProcessSnapshot,
        remapped: ProcessSnapshot,
        identities: SnapshotIdentityMap,
    ) -> None:
        expected_sequences = {
            "subtree_pids": tuple(
                identities.pids.get(value, value)
                for value in source.subtree_pids
            ),
            "object_oids": tuple(
                identities.objects.get(value, value)
                for value in source.object_oids
            ),
            "owned_object_oids": tuple(
                identities.objects.get(value, value)
                for value in source.owned_object_oids
            ),
            "referenced_object_oids": tuple(
                identities.objects.get(value, value)
                for value in source.referenced_object_oids
            ),
            "namespaces": tuple(
                identities.namespaces.get(value, value)
                for value in source.namespaces
            ),
            "owned_namespaces": tuple(
                identities.namespaces.get(value, value)
                for value in source.owned_namespaces
            ),
        }
        for field_name, expected in expected_sequences.items():
            if getattr(remapped, field_name) != expected:
                raise ValidationError(
                    f"snapshot remap left an inconsistent {field_name} reference"
                )
        expected_object_keys = {
            identities.objects.get(key, key)
            for key in source.object_payloads
        }
        if set(remapped.object_payloads) != expected_object_keys:
            raise ValidationError(
                "snapshot remap left inconsistent object_payloads references"
            )
        expected_type_keys = {
            identities.objects.get(key, key)
            for key in source.referenced_object_types
        }
        if set(remapped.referenced_object_types) != expected_type_keys:
            raise ValidationError(
                "snapshot remap left inconsistent referenced_object_types references"
            )
        expected_tool_keys = {
            identities.tools.get(key, key) for key in source.jit_sources
        }
        if set(remapped.jit_sources) != expected_tool_keys:
            raise ValidationError(
                "snapshot remap left inconsistent jit_sources references"
            )

    @classmethod
    def _validate_direct_row_references(
        cls,
        table: str,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        for field_name, map_name in cls._FIELD_MAPS.items():
            value = source_row.get(field_name)
            if value is None:
                continue
            selected_map = getattr(identities, map_name)
            expected = selected_map.get(str(value), value)
            if remapped_row.get(field_name) != expected:
                cls._reference_error(table, index, field_name)

    @classmethod
    def _validate_process_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        cls._validate_process_capability_index(
            index,
            source_row,
            remapped_row,
            identities.capabilities,
        )
        cls._validate_process_tool_tables(
            index,
            source_row,
            remapped_row,
            identities.tools,
        )
        cls._validate_loaded_skill_references(
            index,
            source_row,
            remapped_row,
            identities.tools,
        )
        cls._validate_memory_view_references(
            index,
            source_row,
            remapped_row,
            identities,
        )

    @classmethod
    def _validate_process_capability_index(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        capability_map: Mapping[str, str],
    ) -> None:
        if not capability_map or source_row.get("capabilities_json") is None:
            return
        source_ids = cls._json_container(
            source_row["capabilities_json"],
            field_name="processes.capabilities_json",
            expected_type=list,
        )
        remapped_ids = cls._json_container(
            remapped_row["capabilities_json"],
            field_name="processes.capabilities_json",
            expected_type=list,
        )
        expected = [capability_map.get(str(cap_id), cap_id) for cap_id in source_ids]
        if remapped_ids != expected:
            cls._reference_error("processes", index, "capabilities_json")

    @classmethod
    def _validate_process_tool_tables(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        tool_map: Mapping[str, str],
    ) -> None:
        if not tool_map:
            return
        for field_name in ("tool_table_json", "model_tool_table_json"):
            if source_row.get(field_name) is None:
                continue
            source_table = cls._json_container(
                source_row[field_name],
                field_name=f"processes.{field_name}",
                expected_type=dict,
            )
            remapped_table = cls._json_container(
                remapped_row[field_name],
                field_name=f"processes.{field_name}",
                expected_type=dict,
            )
            expected = {
                name: tool_map.get(str(tool_id), tool_id)
                for name, tool_id in source_table.items()
            }
            if remapped_table != expected:
                cls._reference_error("processes", index, field_name)

    @classmethod
    def _validate_loaded_skill_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        tool_map: Mapping[str, str],
    ) -> None:
        if not tool_map or source_row.get("loaded_skills_json") is None:
            return
        source_skills = cls._json_container(
            source_row["loaded_skills_json"],
            field_name="processes.loaded_skills_json",
            expected_type=dict,
        )
        actual_skills = cls._json_container(
            remapped_row["loaded_skills_json"],
            field_name="processes.loaded_skills_json",
            expected_type=dict,
        )
        expected_skills = deepcopy(source_skills)
        for loaded in expected_skills.values():
            cls._remap_loaded_skill_record(loaded, tool_map)
        if actual_skills != expected_skills:
            cls._reference_error("processes", index, "loaded_skills_json")

    @staticmethod
    def _remap_loaded_skill_record(
        loaded: Any,
        tool_map: Mapping[str, str],
    ) -> None:
        if not isinstance(loaded, dict):
            return
        for field_name in (
            "tool_ids",
            "jit_tool_ids",
            "base_tool_ids",
            "base_model_tool_ids",
        ):
            identifiers = loaded.get(field_name)
            if not isinstance(identifiers, dict):
                continue
            loaded[field_name] = {
                name: tool_map.get(str(tool_id), tool_id)
                for name, tool_id in identifiers.items()
            }

    @classmethod
    def _validate_memory_view_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        has_maps = identities.pids or identities.objects or identities.capabilities
        if not has_maps or source_row.get("memory_view_json") is None:
            return
        source_view = cls._json_container(
            source_row["memory_view_json"],
            field_name="processes.memory_view_json",
            expected_type=dict,
        )
        actual_view = cls._json_container(
            remapped_row["memory_view_json"],
            field_name="processes.memory_view_json",
            expected_type=dict,
        )
        expected_view = deepcopy(source_view)
        cls._remap_memory_view_owner(expected_view, identities.pids)
        cls._remap_memory_view_roots(expected_view, identities)
        if actual_view != expected_view:
            cls._reference_error("processes", index, "memory_view_json")

    @classmethod
    def _validate_object_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        expected_owner = source_row.get("owner_id")
        if identities.pids and source_row.get("owner_kind") in {
            "process",
            "process_result",
        } and isinstance(expected_owner, str):
            expected_owner = identities.pids.get(
                expected_owner,
                expected_owner,
            )
        if remapped_row.get("owner_id") != expected_owner:
            cls._reference_error("objects", index, "owner_id")
        if identities.objects and source_row.get("provenance_json") is not None:
            source_provenance = cls._json_container(
                source_row["provenance_json"],
                field_name="objects.provenance_json",
                expected_type=dict,
            )
            actual_provenance = cls._json_container(
                remapped_row["provenance_json"],
                field_name="objects.provenance_json",
                expected_type=dict,
            )
            expected_provenance = deepcopy(source_provenance)
            parent_oids = expected_provenance.get("parent_oids", [])
            if isinstance(parent_oids, list):
                expected_provenance["parent_oids"] = [
                    identities.objects.get(str(oid), oid)
                    for oid in parent_oids
                ]
            if actual_provenance != expected_provenance:
                cls._reference_error("objects", index, "provenance_json")

    @classmethod
    def _validate_message_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        if not identities.objects or source_row.get("metadata_json") is None:
            return
        source_metadata = cls._json_container(
            source_row["metadata_json"],
            field_name="process_messages.metadata_json",
            expected_type=dict,
        )
        actual_metadata = cls._json_container(
            remapped_row["metadata_json"],
            field_name="process_messages.metadata_json",
            expected_type=dict,
        )
        expected_metadata = deepcopy(source_metadata)
        carrier_oid = expected_metadata.get("label_carrier_oid")
        if isinstance(carrier_oid, str):
            expected_metadata["label_carrier_oid"] = identities.objects.get(
                carrier_oid,
                carrier_oid,
            )
        if actual_metadata != expected_metadata:
            cls._reference_error("process_messages", index, "metadata_json")

    @classmethod
    def _validate_candidate_references(
        cls,
        index: int,
        source_row: Mapping[str, Any],
        remapped_row: Mapping[str, Any],
        identities: SnapshotIdentityMap,
    ) -> None:
        if source_row.get("requested_capabilities_json") is None:
            return
        source_requested = cls._json_container(
            source_row["requested_capabilities_json"],
            field_name="tool_candidates.requested_capabilities_json",
            expected_type=list,
        )
        actual_requested = cls._json_container(
            remapped_row["requested_capabilities_json"],
            field_name="tool_candidates.requested_capabilities_json",
            expected_type=list,
        )
        expected_requested = deepcopy(source_requested)
        for item in expected_requested:
            if not isinstance(item, dict):
                continue
            resource = item.get("resource")
            if isinstance(resource, str):
                item["resource"] = cls._expected_resource(resource, identities)
        if actual_requested != expected_requested:
            cls._reference_error(
                "tool_candidates",
                index,
                "requested_capabilities_json",
            )

    @staticmethod
    def _reference_error(table: str, index: int, field_name: str) -> None:
        raise ValidationError(
            "snapshot remap left an inconsistent identity reference at "
            f"rows.{table}[{index}].{field_name}"
        )
