from __future__ import annotations

"""Canonical checkpoint snapshot value objects.

These types live in the domain model layer so storage contracts can exchange
typed snapshots without importing concrete runtime orchestration modules.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, ClassVar, Mapping

from agent_libos.models.capability import (
    CapabilityEffect,
    CapabilityRight,
    CapabilityStatus,
)
from agent_libos.models.checkpoint import CHECKPOINT_SNAPSHOT_VERSION
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.process import ProcessStatus
from agent_libos.models.process_state import (
    process_outcome_from_json,
    process_wait_state_from_json,
    validate_process_state_fields,
)
from agent_libos.utils.serde import loads


SNAPSHOT_SCHEMA_VERSION = CHECKPOINT_SNAPSHOT_VERSION
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"snapshot {field_name} must be a string")
    selected = value.strip()
    if not allow_empty and not selected:
        raise ValidationError(f"snapshot {field_name} must not be empty")
    return selected


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValidationError(f"snapshot {field_name} must be a list")
    selected = tuple(_string(item, f"{field_name}[]") for item in value)
    if len(selected) != len(set(selected)):
        raise ValidationError(f"snapshot {field_name} must not contain duplicates")
    return selected


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"snapshot {field_name} must be an object")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _module_requirements(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValidationError("snapshot modules must be a list of objects")
    modules: list[dict[str, Any]] = []
    seen_module_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"snapshot modules[{index}] must be an object"
            )
        module_id = item.get("module_id")
        if (
            not isinstance(module_id, str)
            or not module_id
            or module_id != module_id.strip()
        ):
            raise ValidationError(
                f"snapshot modules[{index}].module_id must be a non-empty canonical string"
            )
        source_sha256 = item.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or not _SHA256_PATTERN.fullmatch(source_sha256)
        ):
            raise ValidationError(
                "snapshot modules["
                f"{index}].source_sha256 must be a lowercase 64-character SHA-256 digest"
            )
        if module_id in seen_module_ids:
            raise ValidationError(
                f"snapshot modules contain duplicate module_id: {module_id}"
            )
        seen_module_ids.add(module_id)
        modules.append(
            {str(key): deepcopy(module_value) for key, module_value in item.items()}
        )
    return tuple(modules)


def _row_list(
    value: Any,
    field_name: str,
    expected_columns: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValidationError(f"snapshot rows.{field_name} must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValidationError(f"snapshot rows.{field_name}[{index}] must be an object")
        row = {str(key): deepcopy(row_value) for key, row_value in item.items()}
        columns = frozenset(row)
        if columns != expected_columns:
            missing = expected_columns - columns
            unknown = columns - expected_columns
            raise ValidationError(
                f"snapshot rows.{field_name}[{index}] is not canonical; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        rows.append(row)
    return tuple(rows)


def _validate_process_rows(rows: tuple[dict[str, Any], ...]) -> None:
    for index, row in enumerate(rows):
        try:
            status = ProcessStatus(row["status"])
            for field_name in ("wait_state_json", "outcome_json"):
                if not isinstance(row[field_name], str):
                    raise ValidationError(
                        f"snapshot process {field_name} must be canonical JSON text"
                    )
            wait_state = process_wait_state_from_json(row["wait_state_json"])
            outcome = process_outcome_from_json(row["outcome_json"])
            validate_process_state_fields(status.value, wait_state, outcome)
            generation = row["state_generation"]
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 0
            ):
                raise ValidationError(
                    "snapshot process state_generation must be a non-negative integer"
                )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ValidationError(
                f"invalid snapshot rows.processes[{index}]: {exc}"
            ) from exc


def _capability_field_error(
    index: int,
    field_name: str,
    message: str,
) -> ValidationError:
    return ValidationError(
        f"snapshot rows.capabilities[{index}].{field_name} {message}"
    )


def _capability_text(
    row: Mapping[str, Any],
    index: int,
    field_name: str,
    *,
    nullable: bool = False,
) -> str | None:
    value = row[field_name]
    if nullable and value is None:
        return None
    if type(value) is not str or not value.strip():
        suffix = (
            "must be null or a non-empty string"
            if nullable
            else "must be a non-empty string"
        )
        raise _capability_field_error(index, field_name, suffix)
    return value


def _capability_non_negative_integer(
    row: Mapping[str, Any],
    index: int,
    field_name: str,
    *,
    nullable: bool = False,
) -> int | None:
    value = row[field_name]
    if nullable and value is None:
        return None
    if type(value) is not int or value < 0:
        suffix = (
            "must be null or a non-negative integer"
            if nullable
            else "must be a non-negative integer"
        )
        raise _capability_field_error(index, field_name, suffix)
    return value


def _capability_json_container(
    row: Mapping[str, Any],
    index: int,
    field_name: str,
    container_type: type[list[Any]] | type[dict[str, Any]],
) -> list[Any] | dict[str, Any]:
    encoded = row[field_name]
    if type(encoded) is not str:
        raise _capability_field_error(
            index,
            field_name,
            "must be canonical JSON text",
        )
    try:
        decoded = loads(encoded)
    except (TypeError, ValueError) as exc:
        raise _capability_field_error(
            index,
            field_name,
            "must be valid JSON text",
        ) from exc
    if type(decoded) is not container_type:
        label = "a list" if container_type is list else "an object"
        raise _capability_field_error(
            index,
            field_name,
            f"must decode to {label}",
        )
    return decoded


def _validate_capability_columns(
    row: Any,
    index: int,
    expected_columns: frozenset[str],
) -> None:
    if not isinstance(row, Mapping):
        raise ValidationError(
            f"snapshot rows.capabilities[{index}] must be an object"
        )
    columns = frozenset(row)
    if columns != expected_columns:
        missing = expected_columns - columns
        unknown = columns - expected_columns
        raise ValidationError(
            f"snapshot rows.capabilities[{index}] is not canonical; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _validate_capability_text_fields(
    row: Mapping[str, Any],
    index: int,
) -> None:
    for field_name in (
        "cap_id",
        "subject",
        "resource",
        "issued_by",
        "issued_at",
    ):
        _capability_text(row, index, field_name)
    for field_name in ("expires_at", "issuer_cap_id", "parent_cap_id"):
        _capability_text(row, index, field_name, nullable=True)


def _validate_capability_boolean_fields(
    row: Mapping[str, Any],
    index: int,
) -> None:
    for field_name in ("delegable", "revocable"):
        if type(row[field_name]) is not bool:
            raise _capability_field_error(
                index,
                field_name,
                "must be a boolean",
            )


def _validate_capability_delegation_fields(
    row: Mapping[str, Any],
    index: int,
) -> None:
    delegation_depth = _capability_non_negative_integer(
        row,
        index,
        "delegation_depth",
    )
    max_delegation_depth = _capability_non_negative_integer(
        row,
        index,
        "max_delegation_depth",
        nullable=True,
    )
    _capability_non_negative_integer(
        row,
        index,
        "uses_remaining",
        nullable=True,
    )
    if (
        max_delegation_depth is not None
        and delegation_depth is not None
        and max_delegation_depth < delegation_depth
    ):
        raise _capability_field_error(
            index,
            "max_delegation_depth",
            "must not be less than delegation_depth",
        )


def _validate_capability_rights(
    row: Mapping[str, Any],
    index: int,
) -> None:
    rights = _capability_json_container(
        row,
        index,
        "rights_json",
        list,
    )
    if not rights:
        raise _capability_field_error(
            index,
            "rights_json",
            "must contain at least one right",
        )
    if any(type(right) is not str for right in rights):
        raise _capability_field_error(
            index,
            "rights_json",
            "must contain only strings",
        )
    try:
        canonical_rights = [CapabilityRight(right).value for right in rights]
    except ValueError as exc:
        raise _capability_field_error(
            index,
            "rights_json",
            "contains an unknown capability right",
        ) from exc
    if len(canonical_rights) != len(set(canonical_rights)):
        raise _capability_field_error(
            index,
            "rights_json",
            "must not contain duplicates",
        )


def _validate_capability_enum(
    row: Mapping[str, Any],
    index: int,
    field_name: str,
    enum_type: type[CapabilityEffect] | type[CapabilityStatus],
) -> None:
    value = row[field_name]
    if type(value) is not str:
        raise _capability_field_error(index, field_name, "must be a string")
    try:
        enum_type(value)
    except ValueError as exc:
        raise _capability_field_error(
            index,
            field_name,
            f"must be a recognized capability {field_name}",
        ) from exc


def _validate_capability_lease_state(
    row: Mapping[str, Any],
    index: int,
) -> None:
    uses_remaining = row["uses_remaining"]
    if (
        row["status"] == CapabilityStatus.ACTIVE.value
        and uses_remaining is not None
        and uses_remaining < 1
    ):
        # Finite-use settlement atomically changes an exhausted capability to
        # a non-active status. Zero is valid retained history, never live
        # authority, regardless of the capability effect.
        raise _capability_field_error(
            index,
            "uses_remaining",
            "active capability uses_remaining must be positive",
        )


def _validate_capability_rows(
    rows: tuple[dict[str, Any], ...],
    expected_columns: frozenset[str],
) -> None:
    for index, row in enumerate(rows):
        _validate_capability_columns(row, index, expected_columns)
        _validate_capability_text_fields(row, index)
        _validate_capability_boolean_fields(row, index)
        _validate_capability_delegation_fields(row, index)
        _validate_capability_rights(row, index)
        _capability_json_container(row, index, "constraints_json", dict)
        _capability_json_container(row, index, "metadata_json", dict)
        _validate_capability_enum(row, index, "effect", CapabilityEffect)
        _validate_capability_enum(row, index, "status", CapabilityStatus)
        _validate_capability_lease_state(row, index)


@dataclass(frozen=True)
class SnapshotHeader:
    schema_version: int
    checkpoint_id: str
    root_pid: str
    reason: str
    created_at: str
    created_by: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotHeader":
        version = value.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValidationError("snapshot version must be an integer")
        return cls(
            schema_version=version,
            checkpoint_id=_string(value.get("checkpoint_id"), "checkpoint_id"),
            root_pid=_string(value.get("pid"), "pid"),
            reason=_string(value.get("reason"), "reason", allow_empty=True),
            created_at=_string(value.get("created_at"), "created_at"),
            created_by=_string(value.get("created_by"), "created_by"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "pid": self.root_pid,
            "reason": self.reason,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


@dataclass(frozen=True)
class SnapshotRows:
    # ``process_terminal_cleanups`` is deliberately not checkpoint-scoped.
    # Its owner/lease and completed phases describe Host work in one Runtime,
    # not agent state that may be replayed under remapped process identities.
    # Checkpoint restore/fork repositories atomically create a fresh pending
    # cleanup intent whenever they publish a terminal target process.
    processes: tuple[dict[str, Any], ...] = ()
    object_namespaces: tuple[dict[str, Any], ...] = ()
    objects: tuple[dict[str, Any], ...] = ()
    object_links: tuple[dict[str, Any], ...] = ()
    capabilities: tuple[dict[str, Any], ...] = ()
    process_resource_reservations: tuple[dict[str, Any], ...] = ()
    process_messages: tuple[dict[str, Any], ...] = ()
    llm_pending_actions: tuple[dict[str, Any], ...] = ()
    skills: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tool_candidates: tuple[dict[str, Any], ...] = ()

    TABLES: ClassVar[tuple[str, ...]] = (
        "processes",
        "object_namespaces",
        "objects",
        "object_links",
        "capabilities",
        "process_resource_reservations",
        "process_messages",
        "llm_pending_actions",
        "skills",
        "tools",
        "tool_candidates",
    )
    ROW_COLUMNS: ClassVar[dict[str, frozenset[str]]] = {
        "processes": frozenset(
            {
                "pid", "parent_pid", "image_id", "status", "goal_oid",
                "memory_view_json", "capabilities_json", "loaded_skills_json",
                "tool_table_json", "model_tool_table_json", "event_cursor",
                "checkpoint_head", "status_message", "resource_budget_json",
                "resource_usage_json", "working_directory", "llm_profile_id",
                "wait_state_json", "outcome_json", "state_generation",
                "revision", "execution_generation", "execution_owner_id",
                "execution_lease_id",
                "created_at", "updated_at",
            }
        ),
        "object_namespaces": frozenset(
            {"namespace", "parent_namespace", "metadata_json", "created_by", "created_at", "updated_at"}
        ),
        "objects": frozenset(
            {
                "oid", "namespace", "name", "type", "schema_version", "payload_json",
                "metadata_json", "provenance_json", "version", "immutable", "created_by",
                "owner_kind", "owner_id", "lifecycle_state", "deleted_at", "created_at", "updated_at",
            }
        ),
        "object_links": frozenset(
            {"id", "src_oid", "relation", "dst_oid", "metadata_json", "created_by", "created_at"}
        ),
        "capabilities": frozenset(
            {
                "cap_id", "subject", "resource", "rights_json", "constraints_json", "issued_by",
                "issued_at", "expires_at", "delegable", "revocable", "effect", "issuer_cap_id",
                "parent_cap_id", "delegation_depth", "max_delegation_depth", "uses_remaining",
                "status", "metadata_json",
            }
        ),
        "process_resource_reservations": frozenset(
            {"parent_pid", "child_pid", "reservation_json", "created_at", "updated_at"}
        ),
        "process_messages": frozenset(
            {
                "message_id", "sender", "recipient_pid", "kind", "channel", "correlation_id",
                "reply_to", "subject", "body", "payload_json", "metadata_json", "status",
                "created_at", "updated_at", "acked_at",
            }
        ),
        "llm_pending_actions": frozenset(
            {
                "pid", "resume_token", "llm_operation_id", "tool_operation_id", "wait_type",
                "request_id", "child_pid", "response_id", "tool_call_id", "tool_name",
                "filters_json", "action_json", "data_flow_context_json", "content_preview",
                "tool_call_count", "status", "created_at", "updated_at",
            }
        ),
        "skills": frozenset(
            {
                "skill_id", "name", "version", "package_json", "source_type", "source",
                "package_sha256", "registered_by", "created_at", "updated_at",
            }
        ),
        "tools": frozenset(
            {"tool_id", "name", "spec_json", "scope", "registered_by", "created_at", "ephemeral"}
        ),
        "tool_candidates": frozenset(
            {
                "candidate_id", "pid", "spec_json", "source_code", "tests_json",
                "requested_capabilities_json", "status", "registered_tool_id",
                "validation_json", "created_at", "updated_at",
            }
        ),
    }

    def __post_init__(self) -> None:
        _validate_capability_rows(
            self.capabilities,
            self.ROW_COLUMNS["capabilities"],
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotRows":
        unknown = set(value) - set(cls.TABLES)
        if unknown:
            raise ValidationError(f"snapshot contains unsupported row tables: {sorted(unknown)}")
        missing = set(cls.TABLES) - set(value)
        if missing:
            raise ValidationError(
                f"snapshot rows are not canonical; missing tables: {sorted(missing)}"
            )
        selected = {
            name: _row_list(value[name], name, cls.ROW_COLUMNS[name])
            for name in cls.TABLES
        }
        _validate_process_rows(selected["processes"])
        return cls(**selected)

    @classmethod
    def from_trusted_durable_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SnapshotRows":
        """Decode rows read directly from the trusted durable repository.

        SQLite exposes BOOLEAN columns as integer ``0``/``1`` values. Only
        this repository-only ingress may canonicalize those two exact integer
        values; serialized snapshot documents must use ``from_mapping`` and
        carry real JSON booleans.
        """

        selected = deepcopy(dict(value))
        capability_rows = selected.get("capabilities")
        if isinstance(capability_rows, list):
            normalized_rows: list[Any] = []
            for row in capability_rows:
                if not isinstance(row, Mapping):
                    normalized_rows.append(row)
                    continue
                item = dict(row)
                for field_name in ("delegable", "revocable"):
                    field_value = item.get(field_name)
                    if type(field_value) is int and field_value in {0, 1}:
                        item[field_name] = bool(field_value)
                normalized_rows.append(item)
            selected["capabilities"] = normalized_rows
        return cls.from_mapping(selected)

    def to_mapping(
        self,
        *,
        copy_values: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            name: [
                deepcopy(row) if copy_values else dict(row)
                for row in getattr(self, name)
            ]
            for name in self.TABLES
        }


@dataclass(frozen=True)
class ProcessSnapshot:
    header: SnapshotHeader
    subtree_pids: tuple[str, ...]
    object_oids: tuple[str, ...]
    owned_object_oids: tuple[str, ...]
    referenced_object_oids: tuple[str, ...]
    referenced_object_types: dict[str, str]
    namespaces: tuple[str, ...]
    owned_namespaces: tuple[str, ...]
    rows: SnapshotRows
    object_payloads: dict[str, Any] = field(default_factory=dict)
    images: dict[str, Any] = field(default_factory=dict)
    image_artifacts: dict[str, Any] = field(default_factory=dict)
    jit_sources: dict[str, str] = field(default_factory=dict)
    modules: tuple[dict[str, Any], ...] = ()

    TOP_LEVEL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "version",
            "checkpoint_id",
            "pid",
            "reason",
            "created_at",
            "created_by",
            "subtree_pids",
            "object_oids",
            "owned_object_oids",
            "referenced_object_oids",
            "referenced_object_types",
            "namespaces",
            "owned_namespaces",
            "rows",
            "object_payloads",
            "images",
            "image_artifacts",
            "jit_sources",
            "modules",
        }
    )

    def __post_init__(self) -> None:
        if not self.subtree_pids:
            raise ValidationError("snapshot subtree_pids must not be empty")
        if self.header.root_pid not in self.subtree_pids:
            raise ValidationError("snapshot root pid must belong to subtree_pids")
        process_pids = tuple(row.get("pid") for row in self.rows.processes)
        if any(
            not isinstance(pid, str) or not pid.strip()
            for pid in process_pids
        ):
            raise ValidationError("snapshot process rows require non-empty pid values")
        if len(process_pids) != len(set(process_pids)):
            raise ValidationError("snapshot process rows must not contain duplicate pids")
        if set(process_pids) != set(self.subtree_pids):
            raise ValidationError(
                "snapshot process rows must exactly match subtree_pids"
            )

    @staticmethod
    def decode_module_requirements(value: Any) -> tuple[dict[str, Any], ...]:
        """Strictly decode module identities embedded in checkpoint artifacts."""

        return _module_requirements(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProcessSnapshot":
        unknown = set(value) - cls.TOP_LEVEL_KEYS
        if unknown:
            raise ValidationError(f"snapshot contains unsupported fields: {sorted(unknown)}")
        missing = cls.TOP_LEVEL_KEYS - set(value)
        if missing:
            raise ValidationError(
                f"snapshot is not canonical; missing fields: {sorted(missing)}"
            )
        rows_value = value.get("rows")
        if not isinstance(rows_value, Mapping):
            raise ValidationError("snapshot rows must be an object")
        referenced_types = _mapping(value.get("referenced_object_types", {}), "referenced_object_types")
        jit_sources = _mapping(value.get("jit_sources", {}), "jit_sources")
        return cls(
            header=SnapshotHeader.from_mapping(value),
            subtree_pids=_string_list(value.get("subtree_pids"), "subtree_pids"),
            object_oids=_string_list(value.get("object_oids", []), "object_oids"),
            owned_object_oids=_string_list(value.get("owned_object_oids", []), "owned_object_oids"),
            referenced_object_oids=_string_list(
                value.get("referenced_object_oids", []),
                "referenced_object_oids",
            ),
            referenced_object_types={str(key): _string(item, f"referenced_object_types.{key}") for key, item in referenced_types.items()},
            namespaces=_string_list(value.get("namespaces", []), "namespaces"),
            owned_namespaces=_string_list(value.get("owned_namespaces", []), "owned_namespaces"),
            rows=SnapshotRows.from_mapping(rows_value),
            object_payloads=_mapping(value.get("object_payloads", {}), "object_payloads"),
            images=_mapping(value.get("images", {}), "images"),
            image_artifacts=_mapping(value.get("image_artifacts", {}), "image_artifacts"),
            jit_sources={str(key): _string(item, f"jit_sources.{key}", allow_empty=True) for key, item in jit_sources.items()},
            modules=cls.decode_module_requirements(value.get("modules")),
        )

    def to_mapping(self, *, copy_values: bool = True) -> dict[str, Any]:
        copy_mapping = deepcopy if copy_values else dict
        return {
            **self.header.to_mapping(),
            "subtree_pids": list(self.subtree_pids),
            "object_oids": list(self.object_oids),
            "owned_object_oids": list(self.owned_object_oids),
            "referenced_object_oids": list(self.referenced_object_oids),
            "referenced_object_types": dict(self.referenced_object_types),
            "namespaces": list(self.namespaces),
            "owned_namespaces": list(self.owned_namespaces),
            "rows": self.rows.to_mapping(copy_values=copy_values),
            "object_payloads": copy_mapping(self.object_payloads),
            "images": copy_mapping(self.images),
            "image_artifacts": copy_mapping(self.image_artifacts),
            "jit_sources": dict(self.jit_sources),
            "modules": [
                deepcopy(module) if copy_values else dict(module)
                for module in self.modules
            ],
        }


@dataclass(frozen=True)
class ExecRollbackState:
    """Typed reconstructable snapshot plus process-local executable handles."""

    snapshot: ProcessSnapshot
    tool_ids: frozenset[str]
    tool_handles: dict[str, Any] = field(default_factory=dict)
    capability_rollback_token: str | None = None
