from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.evidence.payload_retention import (
    PayloadRetentionCursor,
    PayloadRetentionPage,
    PayloadRetentionTier,
    external_effect_payload_retention_tier,
    llm_call_payload_can_be_provider_chain_head,
    llm_call_payload_retention_tier,
    validate_external_effect_payload_retention_update,
    validate_llm_call_payload_retention_update,
)
from agent_libos.models.exceptions import (
    ProcessRevisionConflict,
    UnsupportedStoreVersion,
    ValidationError,
)
from agent_libos.process_execution import (
    current_post_exec_completion_mutation,
    current_process_control_mutation,
    current_process_execution_takeover_intent,
    current_process_execution_token,
    current_terminal_process_mutation,
    trusted_process_control_mutation,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.models import (
    AgentObject,
    AgentImage,
    AgentProcess,
    AgentRating,
    AuditRecord,
    Capability,
    CapabilityEffect,
    CapabilityStatus,
    CapabilityUseReservationRecoverySummary,
    Checkpoint,
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptPage,
    CheckpointPayloadDeliveryAttemptState,
    ContextMaterializationManifest,
    DataFlowDecision,
    DataFlowContext,
    DataFlowDirection,
    DataFlowOutcome,
    DataLabels,
    DataSourceRef,
    Event,
    EventPriority,
    EventType,
    ExternalEffectRecord,
    ExternalEffectCursor,
    ExternalEffectPage,
    ExternalEffectRecoveryQuery,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    FileLabelBinding,
    HumanRequest,
    HumanRequestStatus,
    HostResumeProcessWait,
    JsonRpcEndpointSpec,
    JsonRpcHeaderSpec,
    JsonRpcMethodSpec,
    JITRehydrationArtifact,
    JIT_TOOL_EXPOSURES,
    KilledProcessOutcome,
    LLMCallRecord,
    McpHeaderSpec,
    McpHttpTransportSpec,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolSpec,
    MemoryView,
    ObjectFilter,
    ObjectHandle,
    ObjectLifecycleState,
    ObjectLink,
    ObjectMetadata,
    ObjectNamespace,
    ObjectOwnerKind,
    ObjectPayloadRecoverySummary,
    PersistedObjectState,
    ObjectTask,
    ObjectTaskNotification,
    ObjectTaskNotificationStatus,
    ObjectTaskOwnerWatch,
    ObjectTaskRecoveryCursor,
    ObjectTaskRecoveryKind,
    ObjectTaskRecoveryPage,
    ObjectTaskStatus,
    ObjectType,
    OperationCursor,
    OperationEvidenceLink,
    OperationKind,
    OperationOutcome,
    OperationPage,
    OperationRecord,
    OperationState,
    ProcessOutcome,
    ProcessCursor,
    ProcessPage,
    ProcessRestoreEpoch,
    ProcessToolBindingCursor,
    ProcessToolBindingPage,
    ProcessToolBindingRecord,
    ProcessStatus,
    ProcessWaitState,
    StaleExecutionRecoverySummary,
    ProcessExecutionToken,
    PausedProcessWait,
    legacy_status_message,
    process_outcome_from_json,
    process_outcome_to_mapping,
    process_wait_state_from_json,
    process_wait_state_to_mapping,
    upcast_legacy_process_state,
    validate_process_state_fields,
    PROMPT_MODES,
    ProcessMessage,
    ProcessMessageKind,
    ProcessMessageStatus,
    Provenance,
    RelationType,
    ResourceBudget,
    ResourceReservation,
    ResourceUsage,
    ResourceUsageReservation,
    ResourceUsageReservationCursor,
    ResourceUsageReservationPage,
    ResourceUsageReservationStatus,
    RuntimePublicationCursor,
    RuntimePublicationKind,
    RuntimePublicationPage,
    PayloadDeliveryState,
    parse_runtime_publication_kind,
    parse_runtime_publication_state,
    validate_runtime_publication_record,
    SinkTrustLevel,
    SinkTrustRule,
    SinkTrustSpec,
    TaskAuthorityManifest,
    PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY,
    PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION,
    encode_permitted_effects_policy,
    upcast_permitted_effects_policy,
    ToolCandidate,
    ToolCandidateStatus,
    ToolHandle,
    ToolSpec,
    ViewMode,
)
from agent_libos.skills.schema import ActionSchema, JitToolSpec, SkillPackage, SkillResource
from agent_libos.storage.base import (
    StoreAssemblyReadiness,
    StoreAssemblyReservation,
    StoreCloseClaimOutcome,
    StoreCloseOutcome,
)
from agent_libos.storage.engine import SqlEngine, split_sql_script
from agent_libos.storage.gui_visibility import (
    is_gui_presentation_audit_fields,
    is_gui_presentation_event_fields,
)
from agent_libos.utils.serde import dumps, loads

_RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS = (
    '{"present": true, "storage": "runtime_memory"}',
    '{"storage": "runtime_memory", "present": true}',
    '{"present":true,"storage":"runtime_memory"}',
    '{"storage":"runtime_memory","present":true}',
)
_RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS_SQL = ", ".join(
    f"'{marker}'" for marker in _RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS
)


def _is_recovered_runtime_object_payload_marker(payload_json: Any) -> bool:
    try:
        marker = loads(payload_json, {})
    except (TypeError, ValueError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("storage") == "runtime_memory"
        and marker.get("present") is False
        and marker.get("recovered_after_reopen") is True
        and set(marker) == {
            "storage",
            "present",
            "recovered_after_reopen",
        }
    )


def _persisted_object_payload_is_present_without_cache(payload_json: Any) -> bool:
    """Classify durable payload storage without materializing it into Object cache."""

    try:
        value = loads(payload_json, {})
    except (TypeError, ValueError) as exc:
        raise ValueError("Object payload JSON is malformed") from exc
    if not isinstance(value, dict) or value.get("storage") != "runtime_memory":
        # Legacy durable payloads are present, but this projection must never
        # publish their value into the runtime-only payload cache.
        return True
    allowed_keys = {"storage", "present", "recovered_after_reopen"}
    if (
        not {"storage", "present"}.issubset(value)
        or not set(value).issubset(allowed_keys)
        or type(value.get("present")) is not bool
        or (
            "recovered_after_reopen" in value
            and value.get("recovered_after_reopen") is not True
        )
        or (
            value.get("present") is True
            and "recovered_after_reopen" in value
        )
    ):
        raise ValueError("Object runtime payload marker is malformed")
    return False


@dataclass(frozen=True, slots=True)
class _PayloadDeliveryTransitionTarget:
    receipt: dict[str, Any]
    state: str
    attempt_id: str | None
    started_at: str | None
    owner_instance_id: str
    operation_reconciled: bool
    control_attempt: CheckpointPayloadDeliveryAttempt | None


def _payload_delivery_source_matches(
    publication: Mapping[str, Any],
    *,
    expected_delivery_state: str | None,
    expected_attempt: CheckpointPayloadDeliveryAttempt | None,
) -> bool:
    expected_attempt_id = (
        expected_attempt.attempt_id if expected_attempt is not None else None
    )
    expected_started_at = (
        expected_attempt.started_at if expected_attempt is not None else None
    )
    return bool(
        publication["kind"] == "checkpoint_restore"
        and publication["state"] == "committed"
        and publication["phase"] == "reconciled"
        and publication["payload_delivery_state"] == expected_delivery_state
        and publication["payload_delivery_attempt_id"] == expected_attempt_id
        and publication["payload_delivery_started_at"] == expected_started_at
    )


def _initial_payload_delivery_recovery_matches(
    receipt: Mapping[str, Any],
    *,
    expected_delivery_state: str | None,
    recovery_lease_id: str | None,
) -> bool:
    if expected_delivery_state is not None:
        return True
    recovery = receipt.get("recovery")
    return bool(
        isinstance(recovery, dict)
        and recovery.get("disposition") == "terminal"
        and recovery_lease_id
        and recovery.get("lease_id") == recovery_lease_id
    )


def _payload_delivery_receipt_target(
    receipt: Mapping[str, Any],
    *,
    delivery_state: str,
    delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
) -> tuple[dict[str, Any], str | None, str | None]:
    selected = deepcopy(receipt)
    selected["payload_delivery"] = {"state": delivery_state}
    if delivery_attempt is None:
        selected.pop("payload_delivery_attempt", None)
        return selected, None, None
    selected["payload_delivery_attempt"] = {
        "attempt_id": delivery_attempt.attempt_id,
        "started_at": delivery_attempt.started_at,
    }
    return selected, delivery_attempt.attempt_id, delivery_attempt.started_at


def _payload_delivery_owner_target(
    publication: Mapping[str, Any],
    *,
    expected_delivery_state: str | None,
    delivery_state: str,
    expected_attempt: CheckpointPayloadDeliveryAttempt | None,
    delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
    owner_instance_id: str | None,
) -> str | None:
    publication_owner = str(publication["owner_instance_id"])
    selected_owner = owner_instance_id or publication_owner
    if delivery_state in {"confirmed", "completed"} and (
        delivery_attempt is None
        or delivery_attempt.owner_instance_id != selected_owner
    ):
        raise ValidationError(
            "checkpoint payload delivery attempt owner does not match publication owner"
        )
    if expected_delivery_state in {"confirmed", "completed"} and (
        expected_attempt is None
        or expected_attempt.owner_instance_id != publication_owner
    ):
        return None
    return selected_owner


def _nullable_exact_cas_predicate(
    column: str,
    expected: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Build an exact nullable predicate without an untyped SQL NULL bind."""

    if expected is None:
        return f"{column} IS NULL", ()
    return f"{column} = ?", (expected,)


def _validate_payload_delivery_transition_request(
    *,
    expected_delivery_state: str | None,
    delivery_state: str,
    expected_attempt: CheckpointPayloadDeliveryAttempt | None,
    delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
    owner_instance_id: str | None,
    recovery_lease_id: str | None,
) -> None:
    transitions = {
        (None, "pending"),
        ("pending", "pending"),
        ("pending", "confirmed"),
        ("confirmed", "pending"),
        ("confirmed", "completed"),
        ("completed", "pending"),
    }
    if (expected_delivery_state, delivery_state) not in transitions:
        raise ValidationError("invalid checkpoint restore payload delivery transition")
    if expected_delivery_state is not None and recovery_lease_id is not None:
        raise ValidationError(
            "payload delivery recovery lease is only valid for pending creation"
        )
    if owner_instance_id is not None and (
        not isinstance(owner_instance_id, str) or not owner_instance_id
    ):
        raise ValidationError("checkpoint payload delivery owner is invalid")
    if delivery_state == "confirmed" and delivery_attempt is None:
        raise ValidationError("confirmed payload delivery requires an attempt")
    if delivery_state == "completed" and (
        delivery_attempt is None or delivery_attempt != expected_attempt
    ):
        raise ValidationError("completed payload delivery must preserve its attempt")
    if (
        expected_delivery_state in {"confirmed", "completed"}
        and delivery_state == "pending"
        and (expected_attempt is None or delivery_attempt is not None)
    ):
        raise ValidationError(
            "compensated payload delivery must clear its assigned attempt"
        )
    if expected_delivery_state is None and delivery_attempt is not None:
        raise ValidationError("new pending payload delivery cannot have an attempt")


def _payload_delivery_transition_target(
    publication: Mapping[str, Any],
    *,
    expected_delivery_state: str | None,
    delivery_state: str,
    expected_attempt: CheckpointPayloadDeliveryAttempt | None,
    delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
    owner_instance_id: str | None,
    recovery_lease_id: str | None,
) -> _PayloadDeliveryTransitionTarget | None:
    if not _payload_delivery_source_matches(
        publication,
        expected_delivery_state=expected_delivery_state,
        expected_attempt=expected_attempt,
    ):
        return None
    if not _initial_payload_delivery_recovery_matches(
        publication["receipt"],
        expected_delivery_state=expected_delivery_state,
        recovery_lease_id=recovery_lease_id,
    ):
        return None
    receipt, attempt_id, started_at = _payload_delivery_receipt_target(
        publication["receipt"],
        delivery_state=delivery_state,
        delivery_attempt=delivery_attempt,
    )
    selected_owner = _payload_delivery_owner_target(
        publication,
        expected_delivery_state=expected_delivery_state,
        delivery_state=delivery_state,
        expected_attempt=expected_attempt,
        delivery_attempt=delivery_attempt,
        owner_instance_id=owner_instance_id,
    )
    if selected_owner is None:
        return None
    return _PayloadDeliveryTransitionTarget(
        receipt=receipt,
        state=delivery_state,
        attempt_id=attempt_id,
        started_at=started_at,
        owner_instance_id=selected_owner,
        # Delivery state and terminal operation repair are independent.  A
        # payload CAS must preserve, rather than synthesize, the operation
        # reconciliation marker.
        operation_reconciled=bool(publication["operation_reconciled"]),
        # The source token fences compensation; otherwise the newly assigned
        # target token fences preparation.  ACK closes this control row, so a
        # commit-unknown retry can never reopen an already delivered payload.
        control_attempt=expected_attempt or delivery_attempt,
    )

@contextmanager
def _persisted_model_decode(label: str):
    try:
        yield
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid persisted {label}: {exc}") from exc


def _persisted_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _operation_runtime_publication_id(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("runtime_publication_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValidationError("operation runtime publication id must be canonical text")
    return value


_DATA_LABEL_FIELDS = frozenset(
    {
        "sensitivity",
        "trust_level",
        "integrity",
        "origin",
        "tenant",
        "principal",
        "declassification_authority",
    }
)
_REQUIRED_DATA_LABEL_FIELDS = frozenset(
    {"sensitivity", "trust_level", "integrity"}
)

_STALE_OPERATION_RECOVERY_TEMP_TABLE = (
    "agent_libos_stale_operation_recovery_unknown"
)


def _canonical_pending_data_flow_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValidationError("pending LLM action requires a trusted data-flow context")
    labels = value.get("labels")
    if (
        not isinstance(labels, dict)
        or not _REQUIRED_DATA_LABEL_FIELDS.issubset(labels)
        or not set(labels).issubset(_DATA_LABEL_FIELDS)
    ):
        raise ValidationError(
            "pending LLM action data-flow context requires complete security labels"
        )
    try:
        return DataFlowContext.from_dict(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid pending LLM action data-flow context: {exc}") from exc


def _canonical_process_message_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("process message metadata must be an object")
    selected = dict(value)
    labels = selected.get("data_labels")
    if (
        not isinstance(labels, dict)
        or not _REQUIRED_DATA_LABEL_FIELDS.issubset(labels)
        or not set(labels).issubset(_DATA_LABEL_FIELDS)
    ):
        raise ValidationError("process message metadata requires complete security labels")
    if not isinstance(selected.get("source_oids"), list):
        raise ValidationError("process message metadata requires canonical source_oids")
    try:
        selected["data_labels"] = DataLabels.from_dict(labels).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid process message security labels: {exc}") from exc
    return selected


_MISSING_OBJECT_PAYLOAD = object()
_MISSING_PAYLOAD_BEFORE_IMAGE = object()
_LLM_CONTEXT_LABEL_SCHEMA_VERSION = 1
_TOOL_ID_LOOKUP_BATCH_SIZE = 500
STORE_SCHEMA_VERSION = 3
# Python cursor models compare strings by Unicode code point.  SQLite BINARY
# and PostgreSQL "C" are the backend collations that preserve that ordering for
# UTF-8 text.  Every durable text component used by a startup/recovery keyset
# is canonical schema, rather than inheriting a deployment locale.
_V3_KEYSET_TEXT_COLUMNS: dict[str, frozenset[str]] = {
    "capability_use_reservations": frozenset({"created_at", "reservation_id"}),
    "checkpoint_payload_delivery_attempts": frozenset(
        {"attempt_id", "started_at"}
    ),
    "events": frozenset({"created_at", "event_id"}),
    "external_effects": frozenset({"created_at", "effect_id"}),
    "llm_calls": frozenset({"call_id", "created_at"}),
    "objects": frozenset({"created_at", "oid"}),
    "object_tasks": frozenset({"created_at", "task_id"}),
    "operation_evidence": frozenset(
        {"created_at", "evidence_id", "link_id", "operation_id"}
    ),
    "operations": frozenset(
        {
            "operation_id",
            "parent_operation_id",
            "root_operation_id",
            "runtime_publication_id",
            "started_at",
        }
    ),
    "processes": frozenset({"created_at", "parent_pid", "pid"}),
    "process_tool_bindings": frozenset({"pid", "tool_name"}),
    "resource_usage_reservations": frozenset(
        {"created_at", "reservation_id"}
    ),
    "runtime_publications": frozenset(
        {
            "created_at",
            "payload_delivery_attempt_id",
            "payload_delivery_started_at",
            "pid",
            "publication_id",
        }
    ),
}
_PROCESS_REVISION_COUNTER_PREFIX = "process_revision:"
_PROCESS_EXECUTION_COUNTER_PREFIX = "process_execution_generation:"
_PROCESS_STATE_COUNTER_PREFIX = "process_state_generation:"
# Six bind values per PID are used by the bulk floor UPSERT. 150 keeps every
# statement below SQLite's historical 999-variable limit while also bounding
# PostgreSQL restore round trips.
_PROCESS_RESTORE_EPOCH_PID_BATCH_SIZE = 150
_PROCESS_SEMANTIC_FIELDS = frozenset(
    {"status", "wait_state", "outcome", "state_generation"}
)


def _validated_jit_rehydration_limit(
    config: AgentLibOSConfig,
    limit: int,
) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValidationError("JIT rehydration limit must be a positive integer")
    hard_limit = config.runtime.jit_rehydration_page_hard_limit
    if limit > hard_limit:
        raise ValidationError(
            "JIT rehydration limit exceeds configured hard cap: "
            f"{limit} > {hard_limit}"
        )
    return limit


_V3_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "runtime_schema": frozenset("singleton schema_version".split()),
    "agent_ratings": frozenset(
        "rating_id pid score comment rater source metadata_json created_at updated_at".split()
    ),
    "audit_records": frozenset(
        "record_id timestamp actor action target input_refs_json output_refs_json "
        "capability_refs_json decision_json correlation_id parent_record_id "
        "gui_snapshot_visible".split()
    ),
    "authority_manifests": frozenset(
        "manifest_id pid image_id goal_ref authorized_capabilities_json "
        "required_capabilities_json permitted_effects_json resource_budget_json "
        "approval_policy_json data_flow_policy_json expires_at issued_by "
        "parent_manifest_id manifest_hash metadata_json created_at".split()
    ),
    "capabilities": frozenset(
        "cap_id subject resource rights_json constraints_json issued_by issued_at "
        "expires_at delegable revocable effect issuer_cap_id parent_cap_id "
        "delegation_depth max_delegation_depth uses_remaining status metadata_json".split()
    ),
    "capability_use_reservations": frozenset(
        "reservation_id cap_id count status reserved_by reason created_at updated_at".split()
    ),
    "checkpoints": frozenset(
        "checkpoint_id pid reason snapshot_json created_at created_by snapshot_version "
        "metadata_json effect_ledger_seq".split()
    ),
    "checkpoint_payload_delivery_attempts": frozenset(
        "attempt_id owner_instance_id state started_at acked_at updated_at".split()
    ),
    "context_materialization_manifests": frozenset(
        "materialization_id pid view_id policy budget_tokens rendered_tokens "
        "rendered_sha256 context_generation context_oid context_version objects_json "
        "compaction_json created_at".split()
    ),
    "data_flow_decisions": frozenset(
        "decision_id pid sink direction outcome reason labels_json source_refs_json "
        "payload_hash trust_id trust_hash registry_generation release_capability_id "
        "created_at".split()
    ),
    "events": frozenset(
        "event_id type source target payload_json priority created_at correlation_id "
        "causality_json gui_snapshot_visible".split()
    ),
    "external_effects": frozenset(
        "effect_id record_id event_id pid provider operation target rollback_class "
        "rollback_status state_mutation information_flow provider_metadata_json "
        "created_at effect_state transaction_state canonical_args_hash idempotency_key "
        "provider_receipt_json updated_at payload_retention_schema_version "
        "payload_retention_tier payload_retention_sha256".split()
    ),
    "external_effect_transitions": frozenset(
        "seq effect_id effect_state transaction_state occurred_at".split()
    ),
    "file_label_bindings": frozenset(
        "binding_id normalized_path content_sha256 labels_json source_refs_json "
        "generation tombstoned active created_by created_at superseded_at".split()
    ),
    "human_requests": frozenset(
        "request_id pid human payload_json status decision_json blocking created_at "
        "updated_at".split()
    ),
    "image_artifacts": frozenset(
        "artifact_id kind artifact_json sha256 created_by created_at metadata_json".split()
    ),
    "images": frozenset(
        "image_id manifest_json registered_by source created_at updated_at".split()
    ),
    "jsonrpc_endpoints": frozenset(
        "endpoint_id spec_json registered_by created_at updated_at".split()
    ),
    "llm_calls": frozenset(
        "call_id pid image_id purpose status api model request_id response_id "
        "messages_json tools_json request_options_json response_content tool_calls_json "
        "reasoning_json usage_json raw_response_json observability_json error created_at "
        "completed_at payload_retention_tier".split()
    ),
    "llm_context_generations": frozenset(
        "pid generation labels_schema_version labels_json updated_at".split()
    ),
    "llm_pending_actions": frozenset(
        "pid resume_token llm_operation_id tool_operation_id wait_type request_id "
        "child_pid response_id tool_call_id tool_name filters_json action_json "
        "data_flow_context_json content_preview tool_call_count status created_at "
        "updated_at".split()
    ),
    "llm_tool_outputs": frozenset(
        "pid response_id call_id tool_name output_text created_at updated_at".split()
    ),
    "mcp_servers": frozenset(
        "server_id spec_json registered_by created_at updated_at".split()
    ),
    "object_links": frozenset(
        "id src_oid relation dst_oid metadata_json created_by created_at".split()
    ),
    "object_namespaces": frozenset(
        "namespace parent_namespace metadata_json created_by created_at updated_at".split()
    ),
    "object_tasks": frozenset(
        "task_id owner_oid creator_pid runner_pid tool tool_id status notification_status "
        "notification_recipient_pid notification_json "
        "owner_watch_json result_oid error wait_json created_at updated_at started_at "
        "completed_at".split()
    ),
    "objects": frozenset(
        "oid namespace name type schema_version payload_json metadata_json provenance_json "
        "version immutable created_by owner_kind owner_id lifecycle_state deleted_at "
        "created_at updated_at".split()
    ),
    "operation_evidence": frozenset(
        "link_id operation_id evidence_type evidence_id role created_at metadata_json".split()
    ),
    "operations": frozenset(
        "operation_id root_operation_id parent_operation_id kind name actor pid state "
        "outcome expected_roles_json metadata_json runtime_publication_id started_at "
        "updated_at completed_at".split()
    ),
    "process_messages": frozenset(
        "message_id sender recipient_pid kind channel correlation_id reply_to subject "
        "body payload_json metadata_json status created_at updated_at acked_at".split()
    ),
    "process_resource_reservations": frozenset(
        "parent_pid child_pid reservation_json created_at updated_at".split()
    ),
    "process_tool_bindings": frozenset(
        "pid binding_kind tool_name tool_id jit_rehydration_eligible".split()
    ),
    "resource_usage_reservations": frozenset(
        "reservation_id pid usage_json status reserved_by reason settled_usage_json "
        "created_at updated_at".split()
    ),
    "runtime_publications": frozenset(
        "publication_id kind pid owner_instance_id state phase plan_json receipt_json "
        "error_json operation_reconciled payload_delivery_state "
        "payload_delivery_attempt_id payload_delivery_started_at created_at updated_at".split()
    ),
    "processes": frozenset(
        "pid parent_pid image_id status goal_oid memory_view_json capabilities_json "
        "loaded_skills_json tool_table_json model_tool_table_json event_cursor "
        "checkpoint_head status_message wait_state_json outcome_json state_generation "
        "resource_budget_json resource_usage_json "
        "working_directory llm_profile_id revision execution_generation "
        "execution_owner_id execution_lease_id created_at updated_at".split()
    ),
    "runtime_modules": frozenset(
        "module_id name version entrypoint manifest_path manifest_sha256 source_path "
        "source_sha256 status loaded_at registered_json error metadata_json updated_at".split()
    ),
    "runtime_counters": frozenset("counter_name value".split()),
    "sink_trust_records": frozenset(
        "trust_id schema_version pattern trust_level max_sensitivity tenants_json "
        "principals_json identity_sha256 generation spec_hash active created_by created_at "
        "deactivated_at".split()
    ),
    "sink_trust_registry": frozenset(
        "registry_key generation updated_at".split()
    ),
    "skill_trust": frozenset(
        "trust_id source_type source package_sha256 trusted_by created_at metadata_json".split()
    ),
    "skills": frozenset(
        "skill_id name version package_json source_type source package_sha256 "
        "registered_by created_at updated_at".split()
    ),
    "tool_candidates": frozenset(
        "candidate_id pid spec_json source_code tests_json requested_capabilities_json "
        "status registered_tool_id validation_json created_at updated_at".split()
    ),
    "tools": frozenset(
        "tool_id name spec_json scope registered_by created_at ephemeral".split()
    ),
}


class _StoreRLock:
    """RLock with supported current-thread ownership tracking.

    CPython's private ``RLock._is_owned`` is not a portable runtime contract.
    The store needs an exact, public-to-this-module ownership signal so an
    event-loop thread can refuse to offload close while it still owns any store
    scope. Every acquisition, including internal repository scopes, passes
    through this wrapper.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()

    @property
    def owned_by_current_thread(self) -> bool:
        return int(getattr(self._local, "depth", 0)) > 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if timeout == -1:
            acquired = self._lock.acquire(blocking)
        else:
            acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._local.depth = int(getattr(self._local, "depth", 0)) + 1
        return acquired

    def release(self) -> None:
        self._lock.release()
        depth = int(getattr(self._local, "depth", 0)) - 1
        if depth > 0:
            self._local.depth = depth
        elif hasattr(self._local, "depth"):
            del self._local.depth

    def _is_owned(self) -> bool:
        """Compatibility for diagnostics which inspect standard RLock state."""

        return self.owned_by_current_thread

    def __enter__(self) -> _StoreRLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class SQLRuntimeStore:
    """Shared SQL repository used by runtime store backends.

    The store is intentionally thin: policy, permissions, and process semantics
    live in managers. This layer only owns durable shape and reconstruction.
    """

    SYSTEM_NAMESPACE = "system"
    KEYSET_TEXT_COLLATION: str | None = None
    ALLOWED_TABLES = frozenset(
        {
            "objects",
            "object_namespaces",
            "object_links",
            "processes",
            "authority_manifests",
            "process_resource_reservations",
            "process_tool_bindings",
            "resource_usage_reservations",
            "runtime_publications",
            "events",
            "capabilities",
            "capability_use_reservations",
            "sink_trust_registry",
            "sink_trust_records",
            "data_flow_decisions",
            "file_label_bindings",
            "audit_records",
            "operations",
            "operation_evidence",
            "context_materialization_manifests",
            "external_effects",
            "external_effect_transitions",
            "checkpoints",
            "checkpoint_payload_delivery_attempts",
            "human_requests",
            "llm_calls",
            "llm_context_generations",
            "llm_tool_outputs",
            "llm_pending_actions",
            "process_messages",
            "object_tasks",
            "agent_ratings",
            "skills",
            "skill_trust",
            "jsonrpc_endpoints",
            "mcp_servers",
            "images",
            "image_artifacts",
            "tools",
            "tool_candidates",
            "runtime_modules",
            "runtime_counters",
        }
    )

    def _init_store(
        self,
        path: str | Path,
        *,
        config: AgentLibOSConfig | None,
        conn: SqlEngine,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.path = str(path)
        self.conn: SqlEngine = conn
        self._lock = _StoreRLock()
        self.__checkpoint_restore_writer_token = object()
        # Object payloads are runtime memory, not durable database state. SQL
        # rows store only metadata plus a marker saying whether a payload was
        # present in this process.
        self._object_payloads: dict[str, Any] = {}
        self._transaction_depth = 0
        self._payload_transaction_frames: list[dict[str, Any]] = []
        self._stale_operation_recovery_index_depth = 0
        self._stale_operation_recovery_index_active = False
        self._poisoned_reason: str | None = None
        self._released_ownership_reason: str | None = None
        self._backend_ownership_release_observed = False
        self._admission_commit_guard: (
            Callable[[], AbstractContextManager[None]] | None
        ) = None
        self._admission_guard_close_claim: (
            Callable[[], AbstractContextManager[None]] | None
        ) = None
        self._runtime_assembly_reservation: StoreAssemblyReservation | None = None
        self._runtime_assembly_claimant_thread_id: int | None = None
        fresh_store = self._require_supported_store_version()
        with self.transaction(include_object_payloads=True):
            self.initialize()
            if fresh_store:
                # Existing v3 stores were checked by the version gate.  Only a
                # fresh store needs its canonical DDL checked after creation.
                self._require_v3_keyset_text_collations(conn)
                self._write_store_schema_version()

    def _issue_checkpoint_restore_writer_token(self) -> object:
        """Issue the internal checkpoint publication mutation capability."""

        return self.__checkpoint_restore_writer_token

    def _require_supported_store_version(self) -> bool:
        """Reject pre-0.3 stores before initialization can mutate them."""

        return self._require_supported_store_version_for(self.conn)

    @classmethod
    def _require_supported_store_version_for(cls, conn: SqlEngine) -> bool:
        marker_exists, marker_row = cls._probe_schema_marker(conn)
        if marker_exists:
            version = marker_row.get("schema_version") if marker_row is not None else None
            if version != STORE_SCHEMA_VERSION:
                raise UnsupportedStoreVersion(
                    f"unsupported Agent libOS store schema: {version!r}; "
                    f"expected {STORE_SCHEMA_VERSION}"
                )
            cls._require_v3_schema_shape(conn)
            return False
        if cls._probe_user_schema_objects(conn):
            raise UnsupportedStoreVersion(
                "unversioned Agent libOS store detected; 0.2 stores are archive-only "
                "and cannot be opened by 0.3"
            )
        return True

    @classmethod
    def _require_v3_schema_shape(cls, conn: SqlEngine) -> None:
        missing: dict[str, list[str]] = {}
        for table, required in _V3_REQUIRED_COLUMNS.items():
            columns = cls._probe_columns(conn, table)
            absent = sorted(required - columns)
            if absent:
                missing[table] = absent
        if cls._probe_table(conn, "storage_migrations"):
            missing["storage_migrations"] = ["obsolete table must be absent"]
        if missing:
            raise UnsupportedStoreVersion(
                "unsupported or incomplete Agent libOS 0.3 store schema: "
                f"{missing}"
            )
        cls._require_v3_keyset_text_collations(conn)

    @classmethod
    def _require_v3_keyset_text_collations(cls, conn: SqlEngine) -> None:
        expected = cls.KEYSET_TEXT_COLLATION
        if expected is None:
            raise NotImplementedError(
                f"{cls.__name__} must declare its canonical keyset text collation"
            )
        incompatible: dict[str, list[str]] = {}
        collations = cls._probe_text_column_collations(conn)
        for table, columns in _V3_KEYSET_TEXT_COLUMNS.items():
            for column in sorted(columns):
                actual = collations.get((table, column))
                if actual != expected:
                    incompatible.setdefault(table, []).append(
                        f"{column}={actual or 'missing'} (expected {expected})"
                    )
        if incompatible:
            raise UnsupportedStoreVersion(
                "unsupported Agent libOS 0.3 keyset collation schema: "
                f"{incompatible}"
            )

    @classmethod
    def _probe_text_column_collations(
        cls,
        conn: SqlEngine,
    ) -> Mapping[tuple[str, str], str]:
        raise NotImplementedError(
            f"{cls.__name__} must inspect durable text column collations in bulk"
        )

    @classmethod
    def _probe_columns(cls, conn: SqlEngine, table: str) -> set[str]:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})")
            return {str(row["name"]) for row in rows}
        except Exception:
            cls._rollback_probe(conn)
            return set()

    @classmethod
    def _probe_user_schema_objects(cls, conn: SqlEngine) -> set[str]:
        """Return backend-owned user schema objects before fresh initialization."""

        raise NotImplementedError(
            f"{cls.__name__} must enumerate user schema objects before initialization"
        )

    @classmethod
    def _probe_schema_marker(
        cls,
        conn: SqlEngine,
    ) -> tuple[bool, dict[str, Any] | None]:
        try:
            row = conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone()
        except Exception:
            cls._rollback_probe(conn)
            return False, None
        if row is None:
            return True, None
        if isinstance(row, dict):
            return True, dict(row)
        try:
            return True, {"schema_version": row["schema_version"]}
        except (KeyError, TypeError, IndexError):
            return True, {"schema_version": None}

    @classmethod
    def _probe_table(cls, conn: SqlEngine, table: str) -> bool:
        try:
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        except Exception:
            cls._rollback_probe(conn)
            return False
        return True

    @staticmethod
    def _rollback_probe(conn: SqlEngine) -> None:
        try:
            conn.rollback()
        except Exception:
            pass

    def _write_store_schema_version(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE runtime_schema (
              singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
              schema_version INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            "INSERT INTO runtime_schema (singleton, schema_version) VALUES (?, ?)",
            (1, STORE_SCHEMA_VERSION),
        )

    def _execute_script(self, script: str) -> None:
        for statement in split_sql_script(script):
            self.conn.execute(statement)

    @contextmanager
    def locked(self):
        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            yield

    def bind_admission_commit_guard(
        self,
        guard: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """Bind the runtime admission fence used by outer commit paths.

        RuntimeStore owns the durable commit point, so it is the only layer that
        can close the final-revalidation-to-commit race for every repository.
        Only one lifecycle may own the binding at a time: two lifecycle owners
        for one store would make the epoch being checked ambiguous.  A fully
        cleaned failed assembly may release its exact binding for an in-place
        retry; successful Runtime assembly never releases it.
        """

        if not callable(guard):
            raise TypeError("admission commit guard must be callable")
        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            if self._transaction_depth != 0:
                raise RuntimeError(
                    "cannot bind admission commit guard during a store transaction"
                )
            if self._admission_commit_guard is not None:
                raise RuntimeError("admission commit guard is already bound")
            self._admission_commit_guard = guard

    def unbind_admission_commit_guard(
        self,
        guard: Callable[[], AbstractContextManager[None]],
    ) -> bool:
        """Release ``guard`` only while it is still this store's exact owner.

        Failed Runtime assembly leaves caller-owned stores open so callers can
        inspect and retry them.  The compare-and-clear must share the store
        lock with transactions and binding: a stale cleanup attempt must never
        clear a guard installed by a later live Runtime.
        """

        if not callable(guard):
            raise TypeError("admission commit guard must be callable")
        with self._lock:
            if self._transaction_depth != 0:
                raise RuntimeError(
                    "cannot unbind admission commit guard during a store transaction"
                )
            if getattr(self, "_admission_guard_close_claim", None) is guard:
                raise RuntimeError(
                    "cannot unbind admission commit guard while its close is pending"
                )
            if self._admission_commit_guard is not guard:
                return False
            self._admission_commit_guard = None
            return True

    def replace_admission_commit_guard(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]] | None,
        replacement_guard: Callable[[], AbstractContextManager[None]],
    ) -> bool:
        """Atomically replace an exact guard with a failed-open close claim.

        The replacement shares the store lock with binding and transactions.
        Once installed, ordinary lifecycle cleanup can only identity-unbind its
        now-stale guard, while the unique replacement continues to block a
        successor until its cleanup handle closes the store through
        :meth:`release_admission_guard_and_close`.
        """

        if expected_guard is not None and not callable(expected_guard):
            raise TypeError("expected admission commit guard must be callable or None")
        if not callable(replacement_guard):
            raise TypeError("replacement admission commit guard must be callable")
        with self._lock:
            if self._ownership_control_outcome() is not StoreCloseClaimOutcome.READY:
                return False
            if getattr(self, "_admission_guard_close_claim", None) is not None:
                raise RuntimeError(
                    "cannot replace admission commit guard while close is pending"
                )
            if self._transaction_depth != 0:
                raise RuntimeError(
                    "cannot replace admission commit guard during a store transaction"
                )
            if self._admission_commit_guard is not expected_guard:
                return False
            self._admission_commit_guard = replacement_guard
            return True

    def try_replace_admission_commit_guard(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]] | None,
        replacement_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Nonblockingly reserve a failed-open guard replacement.

        This is the repair counterpart to guarded-close probe/claim. It lets an
        async cleanup handle recover a deliberately unbound, still-owned store
        without ever waiting on a lock held by its own event-loop thread.
        """

        if expected_guard is not None and not callable(expected_guard):
            raise TypeError("expected admission commit guard must be callable or None")
        if not callable(replacement_guard):
            raise TypeError("replacement admission commit guard must be callable")

        lock = self._lock
        if bool(getattr(lock, "owned_by_current_thread", False)):
            if self._transaction_depth != 0:
                return StoreCloseClaimOutcome.ACTIVE_TRANSACTION
            return StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED
        if not lock.acquire(blocking=False):
            return StoreCloseClaimOutcome.LOCK_BUSY
        try:
            ownership_outcome = self._ownership_control_outcome()
            if ownership_outcome is not StoreCloseClaimOutcome.READY:
                return ownership_outcome
            if self._transaction_depth != 0:
                return StoreCloseClaimOutcome.ACTIVE_TRANSACTION
            current_guard = self._admission_commit_guard
            close_claim = getattr(
                self,
                "_admission_guard_close_claim",
                None,
            )
            if current_guard is None and close_claim is replacement_guard:
                # A close failure followed by an interrupted guard restore
                # retains this exact claim as the sole ownership token.
                return StoreCloseClaimOutcome.READY
            if current_guard is replacement_guard:
                if close_claim is None or close_claim is replacement_guard:
                    return StoreCloseClaimOutcome.READY
                return StoreCloseClaimOutcome.GUARD_MISMATCH
            if close_claim is not None or current_guard is not expected_guard:
                return StoreCloseClaimOutcome.GUARD_MISMATCH
            self._admission_commit_guard = replacement_guard
            return StoreCloseClaimOutcome.READY
        finally:
            lock.release()

    def probe_admission_guard_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Inspect exact close readiness without blocking or changing state."""

        return self._admission_guard_close_readiness(
            expected_guard,
            install_claim=False,
        )

    def probe_runtime_assembly_readiness(self) -> StoreAssemblyReadiness:
        """Report whether off-thread assembly can enter the store immediately."""

        lock = self._lock
        if bool(getattr(lock, "owned_by_current_thread", False)):
            if self._transaction_depth != 0:
                return StoreAssemblyReadiness.ACTIVE_TRANSACTION
            return StoreAssemblyReadiness.CURRENT_THREAD_LOCKED
        if not lock.acquire(blocking=False):
            return StoreAssemblyReadiness.LOCK_BUSY
        try:
            if self._transaction_depth != 0:
                return StoreAssemblyReadiness.ACTIVE_TRANSACTION
            if self._runtime_assembly_reservation is not None:
                return StoreAssemblyReadiness.LOCK_BUSY
            return StoreAssemblyReadiness.READY
        finally:
            lock.release()

    def reserve_runtime_assembly(
        self,
        reservation: StoreAssemblyReservation,
    ) -> StoreAssemblyReadiness:
        """Atomically fence the readiness-to-worker handoff for assembly."""

        if not isinstance(reservation, StoreAssemblyReservation):
            raise TypeError("runtime assembly reservation has the wrong type")
        lock = self._lock
        if lock.owned_by_current_thread:
            if self._transaction_depth != 0:
                return StoreAssemblyReadiness.ACTIVE_TRANSACTION
            return StoreAssemblyReadiness.CURRENT_THREAD_LOCKED
        if not lock.acquire(blocking=False):
            return StoreAssemblyReadiness.LOCK_BUSY
        try:
            if self._transaction_depth != 0:
                return StoreAssemblyReadiness.ACTIVE_TRANSACTION
            if self._runtime_assembly_reservation is not None:
                return StoreAssemblyReadiness.LOCK_BUSY
            self._runtime_assembly_reservation = reservation
            self._runtime_assembly_claimant_thread_id = None
            return StoreAssemblyReadiness.READY
        finally:
            lock.release()

    @contextmanager
    def claim_runtime_assembly(
        self,
        reservation: StoreAssemblyReservation,
    ) -> Iterator[None]:
        """Activate one exact reservation for this startup worker thread."""

        if not isinstance(reservation, StoreAssemblyReservation):
            raise TypeError("runtime assembly reservation has the wrong type")
        claimant_thread_id = threading.get_ident()
        with self._lock:
            self._ensure_healthy()
            self._ensure_no_admission_guard_close_claim()
            if self._runtime_assembly_reservation is not reservation:
                raise RuntimeError(
                    "runtime assembly reservation is not owned by this worker"
                )
            if self._runtime_assembly_claimant_thread_id is not None:
                raise RuntimeError("runtime assembly reservation is already claimed")
            self._runtime_assembly_claimant_thread_id = claimant_thread_id
        try:
            yield
        finally:
            with self._lock:
                if (
                    self._runtime_assembly_reservation is reservation
                    and self._runtime_assembly_claimant_thread_id
                    == claimant_thread_id
                ):
                    self._runtime_assembly_claimant_thread_id = None
                    self._runtime_assembly_reservation = None

    def release_runtime_assembly_reservation(
        self,
        reservation: StoreAssemblyReservation,
    ) -> bool:
        """Compare-and-clear one exact reservation which was never claimed."""

        if not isinstance(reservation, StoreAssemblyReservation):
            raise TypeError("runtime assembly reservation has the wrong type")
        with self._lock:
            if self._runtime_assembly_reservation is not reservation:
                return False
            if self._runtime_assembly_claimant_thread_id is not None:
                raise RuntimeError(
                    "cannot release an actively claimed runtime assembly reservation"
                )
            self._runtime_assembly_reservation = None
            return True

    def claim_admission_guard_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseClaimOutcome:
        """Atomically reserve an exact close without blocking the caller.

        The caller is normally an event-loop thread which will offload the
        blocking backend close. Once claimed, new store lock and transaction
        scopes fail fast, closing the preflight-to-worker TOCTOU window.
        """

        return self._admission_guard_close_readiness(
            expected_guard,
            install_claim=True,
        )

    def _admission_guard_close_readiness(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
        *,
        install_claim: bool,
    ) -> StoreCloseClaimOutcome:
        if not callable(expected_guard):
            raise TypeError("expected admission commit guard must be callable")

        lock = self._lock
        owned_by_current_thread = bool(
            getattr(lock, "owned_by_current_thread", False)
        )
        if owned_by_current_thread:
            if self._transaction_depth != 0:
                return StoreCloseClaimOutcome.ACTIVE_TRANSACTION
            ownership_outcome = self._ownership_control_outcome()
            if ownership_outcome is not StoreCloseClaimOutcome.READY:
                return ownership_outcome
            return StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED

        if not lock.acquire(blocking=False):
            return StoreCloseClaimOutcome.LOCK_BUSY
        try:
            ownership_outcome = self._ownership_control_outcome()
            if ownership_outcome is not StoreCloseClaimOutcome.READY:
                return ownership_outcome
            if self._transaction_depth != 0:
                return StoreCloseClaimOutcome.ACTIVE_TRANSACTION
            close_claim = getattr(
                self,
                "_admission_guard_close_claim",
                None,
            )
            guard_matches = self._admission_commit_guard is expected_guard
            retained_claim_matches = (
                self._admission_commit_guard is None
                and close_claim is expected_guard
            )
            if not guard_matches and not retained_claim_matches:
                return StoreCloseClaimOutcome.GUARD_MISMATCH
            if close_claim is not None and close_claim is not expected_guard:
                return StoreCloseClaimOutcome.GUARD_MISMATCH
            if install_claim:
                self._admission_guard_close_claim = expected_guard
            return StoreCloseClaimOutcome.READY
        finally:
            lock.release()

    def release_admission_guard_and_close(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> StoreCloseOutcome:
        """Compare-and-clear one guard, then close under the same store lock.

        Recovery handoff must not leave a window in which a successor can bind
        a guard after the failed runtime releases its ownership but before the
        backend ownership is released. Backends expose their exact session or
        lease state after every close attempt. A failure before that release
        point restores the exact guard and remains retryable. A failure after
        it returns as a warning, never restores the stale guard, and permanently
        makes this store instance unusable. A secondary restore failure is
        reported alongside the primary close failure.
        """

        if not callable(expected_guard):
            raise TypeError("expected admission commit guard must be callable")
        with self._lock:
            if self._transaction_depth != 0:
                raise RuntimeError(
                    "cannot release admission commit guard during a store transaction"
                )
            close_claim = getattr(self, "_admission_guard_close_claim", None)
            if (
                self._ownership_control_outcome()
                is StoreCloseClaimOutcome.OWNERSHIP_RELEASED
            ):
                guard_matched = (
                    (
                        self._admission_commit_guard is expected_guard
                        and (close_claim is None or close_claim is expected_guard)
                    )
                    or (
                        self._admission_commit_guard is None
                        and close_claim is expected_guard
                    )
                )
                if guard_matched:
                    self._admission_commit_guard = None
                    self._admission_guard_close_claim = None
                return StoreCloseOutcome(
                    guard_matched=guard_matched,
                    ownership_released=True,
                )
            if close_claim is not None and close_claim is not expected_guard:
                return StoreCloseOutcome(
                    guard_matched=False,
                    ownership_released=False,
                )
            guard_matches = self._admission_commit_guard is expected_guard
            retained_claim_matches = (
                self._admission_commit_guard is None
                and close_claim is expected_guard
            )
            if not guard_matches and not retained_claim_matches:
                return StoreCloseOutcome(
                    guard_matched=False,
                    ownership_released=False,
                )

            self._admission_commit_guard = None
            close_was_claimed = close_claim is expected_guard
            if close_was_claimed:
                # This worker now owns both the exact guard transition and the
                # store lock. The lock replaces the pending gate while backend
                # close runs and permits trusted same-thread diagnostic reads
                # used to observe the close point.
                self._admission_guard_close_claim = None
            try:
                outcome = self._close_backend_for_handoff()
            except BaseException as close_error:
                restore_error: BaseException | None = None
                if self._admission_commit_guard is None:
                    try:
                        self._restore_admission_commit_guard_after_close_failure(
                            expected_guard
                        )
                        if self._admission_commit_guard is not expected_guard:
                            raise RuntimeError(
                                "admission commit guard restore did not retain its exact owner"
                            )
                    except BaseException as exc:
                        restore_error = exc
                if restore_error is not None:
                    # Ownership is retained without a restored commit fence.
                    # Preserve the exact prior owner as a claim-only retry
                    # token before releasing the store lock; successor binding
                    # and every ordinary store scope remain fail-closed.
                    self._admission_guard_close_claim = expected_guard
                    raise BaseExceptionGroup(
                        "runtime store close and admission guard restore failed",
                        [close_error, restore_error],
                    ) from None
                raise
            return outcome

    def _restore_admission_commit_guard_after_close_failure(
        self,
        expected_guard: Callable[[], AbstractContextManager[None]],
    ) -> None:
        """Restore the exact close owner even when the data plane is poisoned."""

        if self._admission_commit_guard is not None:
            raise RuntimeError(
                "admission commit guard restore found a different live owner"
            )
        self._admission_commit_guard = expected_guard

    def _close_backend_for_handoff(self) -> StoreCloseOutcome:
        """Close and classify the backend's irreversible ownership point."""

        close_error: BaseException | None = None
        try:
            self.close()
        except BaseException as exc:
            close_error = exc

        ownership_released, observation_errors = (
            self._observe_runtime_ownership_after_close()
        )
        if ownership_released:
            warnings = (
                (() if close_error is None else (close_error,))
                + observation_errors
            )
            reason = (
                "backend ownership was closed"
                if not warnings
                else "backend ownership was released while close reported diagnostics"
            )
            mark_error = self._record_runtime_ownership_released(reason)
            if mark_error is not None:
                warnings += (mark_error,)
            return StoreCloseOutcome(
                guard_matched=True,
                ownership_released=True,
                warnings=warnings,
            )

        failures: list[BaseException] = []
        if close_error is not None:
            failures.append(close_error)
        failures.extend(observation_errors)
        if close_error is None:
            failures.append(
                RuntimeError(
                    "runtime store close returned without releasing backend ownership"
                )
            )
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup(
            "runtime store close failed before ownership release was observable",
            failures,
        ) from None

    def _observe_runtime_ownership_after_close(
        self,
    ) -> tuple[bool, tuple[BaseException, ...]]:
        """Observe the close point without losing a backend's monotonic signal."""

        observation_errors: list[BaseException] = []
        try:
            observed = self._runtime_ownership_released()
        except BaseException as exc:
            observation_errors.append(exc)
            observed = False

        # SQLite and PostgreSQL set this marker at their concrete lease/session
        # release point. It is deliberately independent of the generic observer
        # so an interrupted diagnostic probe cannot resurrect a stale guard.
        if observed or getattr(
            self,
            "_backend_ownership_release_observed",
            False,
        ):
            return True, tuple(observation_errors)

        if observation_errors:
            # A transient observer interruption is diagnostic, not ownership
            # state. Retry once before classifying the close as retained.
            try:
                observed = self._runtime_ownership_released()
            except BaseException as exc:
                observation_errors.append(exc)
            if observed:
                return True, tuple(observation_errors)
        return False, tuple(observation_errors)

    def _record_runtime_ownership_released(
        self,
        reason: str,
    ) -> BaseException | None:
        """Persist terminal state while demoting post-release diagnostics."""

        try:
            self._mark_runtime_ownership_released(reason)
        except BaseException as exc:
            # Crossing the backend release point is irreversible. Preserve the
            # fact directly so an instrumentation/diagnostic failure cannot
            # make the outer handoff restore a stale guard.
            self._released_ownership_reason = reason
            return exc
        return None

    def _runtime_ownership_released(self) -> bool:
        """Return backend-observed lease/session ownership, without probing SQL."""

        raise NotImplementedError(
            f"{type(self).__name__} must report runtime ownership state"
        )

    def _mark_runtime_ownership_released(self, reason: str) -> None:
        if getattr(self, "_released_ownership_reason", None) is None:
            self._released_ownership_reason = reason

    def _ensure_no_admission_guard_close_claim(self) -> None:
        if getattr(self, "_admission_guard_close_claim", None) is not None:
            raise RuntimeError(
                "runtime store admission-guard close is pending; retry after close"
            )

    def _ensure_store_scope_admitted(self) -> None:
        """Fail fast while close or another thread's assembly owns admission."""

        self._ensure_no_admission_guard_close_claim()
        reservation = getattr(self, "_runtime_assembly_reservation", None)
        if reservation is None:
            return
        if self._runtime_assembly_claimant_thread_id == threading.get_ident():
            return
        raise RuntimeError(
            "runtime store assembly is reserved for its startup worker"
        )

    def _ownership_control_outcome(self) -> StoreCloseClaimOutcome:
        """Report whether exact ownership controls still have work to do.

        A rollback failure may close the database connection while a distinct
        SQLite file lease remains held. Data access must stay fail-closed, but
        rejecting guard repair would strand that lease forever. Ownership
        controls remain available while the backend proves this instance still
        owns a releasable lease/session. If ownership is already gone, callers
        receive a structured terminal result instead of an exception that could
        be mistaken for a retryable cleanup failure.
        """

        if getattr(self, "_released_ownership_reason", None) is not None:
            return StoreCloseClaimOutcome.OWNERSHIP_RELEASED
        if self._runtime_ownership_released():
            self._record_runtime_ownership_released(
                "backend ownership had already been released"
            )
            return StoreCloseClaimOutcome.OWNERSHIP_RELEASED
        return StoreCloseClaimOutcome.READY

    def _admission_commit_scope(self) -> AbstractContextManager[None]:
        guard = self._admission_commit_guard
        return nullcontext() if guard is None else guard()

    def close(self) -> None:
        self.conn.close()

    def _ensure_healthy(self) -> None:
        released_ownership_reason = getattr(
            self,
            "_released_ownership_reason",
            None,
        )
        if released_ownership_reason is not None:
            raise ValidationError(
                "runtime store is unusable after backend ownership release: "
                f"{released_ownership_reason}"
            )
        poisoned_reason = getattr(self, "_poisoned_reason", None)
        if poisoned_reason is not None:
            raise ValidationError(
                "runtime store is unusable after transaction rollback failure: "
                f"{poisoned_reason}"
            )

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason
        try:
            self.conn.close()
        except Exception:
            pass

    def _rollback_scope(
        self,
        *,
        depth: int,
        savepoint: str,
        payload_frame: dict[str, Any] | None,
        operation: str,
    ) -> None:
        rollback_error: BaseException | None = None
        payload_error: BaseException | None = None
        try:
            if depth == 0:
                self.conn.rollback()
            else:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except BaseException as exc:
            rollback_error = exc
        try:
            if payload_frame is not None:
                self._restore_payload_frame(payload_frame)
        except BaseException as exc:
            payload_error = exc

        if rollback_error is None and payload_error is None:
            return
        failures: list[str] = []
        if rollback_error is not None:
            failures.append(f"SQL rollback failed: {rollback_error}")
        if payload_error is not None:
            failures.append(f"payload rollback failed: {payload_error}")
        reason = f"{operation}; " + "; ".join(failures)
        self._poison(reason)
        raise ValidationError(
            f"runtime store is unusable after transaction rollback failure: {reason}"
        ) from (rollback_error or payload_error)

    def _journal_object_payload(self, oid: str) -> None:
        """Capture an OID's value at the start of the current transaction frame."""

        if not self._payload_transaction_frames:
            return
        frame = self._payload_transaction_frames[-1]
        if oid in frame:
            return
        if oid in self._object_payloads:
            frame[oid] = deepcopy(self._object_payloads[oid])
        else:
            frame[oid] = _MISSING_PAYLOAD_BEFORE_IMAGE

    def _restore_payload_frame(self, frame: dict[str, Any]) -> None:
        for oid, before_image in frame.items():
            if before_image is _MISSING_PAYLOAD_BEFORE_IMAGE:
                self._object_payloads.pop(oid, None)
            else:
                self._object_payloads[oid] = deepcopy(before_image)

    def _merge_payload_frame_into_parent(self, frame: dict[str, Any]) -> None:
        if len(self._payload_transaction_frames) < 2:
            return
        parent = self._payload_transaction_frames[-2]
        for oid, before_image in frame.items():
            parent.setdefault(oid, before_image)

    def _set_cached_object_payload(self, oid: str, payload: Any) -> None:
        self._journal_object_payload(oid)
        self._object_payloads[oid] = deepcopy(payload)

    def _forget_cached_object_payload(self, oid: str) -> None:
        self._journal_object_payload(oid)
        self._object_payloads.pop(oid, None)

    def initialize(self) -> None:
        with self._lock:
            self._execute_script(
                """
                CREATE TABLE IF NOT EXISTS objects (
                  oid TEXT COLLATE BINARY PRIMARY KEY,
                  namespace TEXT NOT NULL DEFAULT 'system',
                  name TEXT NOT NULL,
                  type TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  provenance_json TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  immutable INTEGER NOT NULL,
                  created_by TEXT NOT NULL,
                  owner_kind TEXT NOT NULL DEFAULT 'process',
                  owner_id TEXT,
                  lifecycle_state TEXT NOT NULL DEFAULT 'live',
                  deleted_at TEXT,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS object_namespaces (
                  namespace TEXT PRIMARY KEY,
                  parent_namespace TEXT,
                  metadata_json TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS object_links (
                  id TEXT PRIMARY KEY,
                  src_oid TEXT NOT NULL,
                  relation TEXT NOT NULL,
                  dst_oid TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processes (
                  pid TEXT COLLATE BINARY PRIMARY KEY,
                  parent_pid TEXT COLLATE BINARY,
                  image_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  goal_oid TEXT,
                  memory_view_json TEXT,
                  capabilities_json TEXT NOT NULL,
                  loaded_skills_json TEXT NOT NULL,
                  tool_table_json TEXT NOT NULL,
                  model_tool_table_json TEXT NOT NULL DEFAULT '{}',
                  event_cursor TEXT,
                  checkpoint_head TEXT,
                  status_message TEXT,
                  wait_state_json TEXT NOT NULL DEFAULT 'null',
                  outcome_json TEXT NOT NULL DEFAULT 'null',
                  state_generation BIGINT NOT NULL DEFAULT 0,
                  resource_budget_json TEXT NOT NULL,
                  resource_usage_json TEXT NOT NULL DEFAULT '{}',
                  working_directory TEXT NOT NULL DEFAULT '.',
                  llm_profile_id TEXT NOT NULL DEFAULT 'default',
                  revision BIGINT NOT NULL DEFAULT 0,
                  execution_generation BIGINT NOT NULL DEFAULT 0,
                  execution_owner_id TEXT,
                  execution_lease_id TEXT,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS authority_manifests (
                  manifest_id TEXT PRIMARY KEY,
                  pid TEXT NOT NULL UNIQUE,
                  image_id TEXT NOT NULL,
                  goal_ref TEXT,
                  authorized_capabilities_json TEXT NOT NULL,
                  required_capabilities_json TEXT NOT NULL,
                  permitted_effects_json TEXT NOT NULL,
                  resource_budget_json TEXT NOT NULL,
                  approval_policy_json TEXT NOT NULL,
                  data_flow_policy_json TEXT NOT NULL,
                  expires_at TEXT,
                  issued_by TEXT NOT NULL,
                  parent_manifest_id TEXT,
                  manifest_hash TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_authority_manifests_parent
                  ON authority_manifests(parent_manifest_id, created_at, manifest_id);

                CREATE TABLE IF NOT EXISTS process_resource_reservations (
                  parent_pid TEXT NOT NULL,
                  child_pid TEXT NOT NULL,
                  reservation_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(parent_pid, child_pid)
                );

                CREATE INDEX IF NOT EXISTS idx_resource_reservations_parent
                  ON process_resource_reservations(parent_pid, child_pid);

                CREATE INDEX IF NOT EXISTS idx_resource_reservations_child
                  ON process_resource_reservations(child_pid, parent_pid);

                CREATE TABLE IF NOT EXISTS resource_usage_reservations (
                  reservation_id TEXT COLLATE BINARY PRIMARY KEY,
                  pid TEXT NOT NULL,
                  usage_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  reserved_by TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  settled_usage_json TEXT,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_usage_reservations_pid_status ON resource_usage_reservations(pid, status, created_at COLLATE BINARY, reservation_id COLLATE BINARY);

                CREATE INDEX IF NOT EXISTS idx_usage_reservations_recovery ON resource_usage_reservations(status, created_at COLLATE BINARY, reservation_id COLLATE BINARY);
                CREATE TABLE IF NOT EXISTS events (
                  event_id TEXT COLLATE BINARY PRIMARY KEY,
                  type TEXT NOT NULL,
                  source TEXT NOT NULL,
                  target TEXT,
                  payload_json TEXT NOT NULL,
                  priority TEXT NOT NULL,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  correlation_id TEXT,
                  causality_json TEXT NOT NULL,
                  gui_snapshot_visible INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_created
                  ON events(created_at COLLATE BINARY, event_id COLLATE BINARY);

                CREATE INDEX IF NOT EXISTS idx_events_target_created
                  ON events(target, created_at COLLATE BINARY, event_id COLLATE BINARY);

                CREATE TABLE IF NOT EXISTS capabilities (
                  cap_id TEXT PRIMARY KEY,
                  subject TEXT NOT NULL,
                  resource TEXT NOT NULL,
                  rights_json TEXT NOT NULL,
                  constraints_json TEXT NOT NULL,
                  issued_by TEXT NOT NULL,
                  issued_at TEXT NOT NULL,
                  expires_at TEXT,
                  delegable INTEGER NOT NULL,
                  revocable INTEGER NOT NULL,
                  effect TEXT NOT NULL,
                  issuer_cap_id TEXT,
                  parent_cap_id TEXT,
                  delegation_depth INTEGER NOT NULL,
                  max_delegation_depth INTEGER,
                  uses_remaining INTEGER,
                  status TEXT NOT NULL,
                  metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS capability_use_reservations (
                  reservation_id TEXT COLLATE BINARY PRIMARY KEY,
                  cap_id TEXT NOT NULL,
                  count INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  reserved_by TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_records (
                  record_id TEXT PRIMARY KEY,
                  timestamp TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  action TEXT NOT NULL,
                  target TEXT,
                  input_refs_json TEXT NOT NULL,
                  output_refs_json TEXT NOT NULL,
                  capability_refs_json TEXT NOT NULL,
                  decision_json TEXT,
                  correlation_id TEXT,
                  parent_record_id TEXT,
                  gui_snapshot_visible INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_records_created
                  ON audit_records(timestamp, record_id);

                CREATE INDEX IF NOT EXISTS idx_audit_records_actor_created
                  ON audit_records(actor, timestamp, record_id);

                CREATE INDEX IF NOT EXISTS idx_audit_records_target_created
                  ON audit_records(target, timestamp, record_id);

                CREATE INDEX IF NOT EXISTS idx_audit_records_correlation_created
                  ON audit_records(correlation_id, timestamp, record_id);

                CREATE TABLE IF NOT EXISTS external_effects (
                  effect_id TEXT COLLATE BINARY PRIMARY KEY,
                  record_id TEXT,
                  event_id TEXT,
                  pid TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  operation TEXT NOT NULL,
                  target TEXT,
                  rollback_class TEXT NOT NULL,
                  rollback_status TEXT NOT NULL,
                  state_mutation INTEGER NOT NULL,
                  information_flow INTEGER NOT NULL,
                  provider_metadata_json TEXT NOT NULL,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  effect_state TEXT NOT NULL DEFAULT 'finalized',
                  transaction_state TEXT NOT NULL DEFAULT 'committed',
                  canonical_args_hash TEXT,
                  idempotency_key TEXT,
                  provider_receipt_json TEXT NOT NULL DEFAULT '{}',
                  updated_at TEXT,
                  payload_retention_schema_version INTEGER NOT NULL DEFAULT 1,
                  payload_retention_tier TEXT NOT NULL DEFAULT 'full',
                  payload_retention_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS runtime_counters (
                  counter_name TEXT PRIMARY KEY,
                  value BIGINT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_effect_transitions (
                  seq BIGINT PRIMARY KEY,
                  effect_id TEXT NOT NULL,
                  effect_state TEXT NOT NULL,
                  transaction_state TEXT NOT NULL,
                  occurred_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_effect_transitions_effect_seq
                  ON external_effect_transitions(effect_id, seq);

                CREATE TABLE IF NOT EXISTS checkpoints (
                  checkpoint_id TEXT PRIMARY KEY,
                  pid TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  snapshot_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  created_by TEXT,
                  snapshot_version INTEGER NOT NULL DEFAULT 4,
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  effect_ledger_seq BIGINT NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS human_requests (
                  request_id TEXT PRIMARY KEY,
                  pid TEXT NOT NULL,
                  human TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  decision_json TEXT,
                  blocking INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_human_requests_pid_created
                  ON human_requests(pid, created_at, request_id);

                CREATE INDEX IF NOT EXISTS idx_human_requests_human_status_created
                  ON human_requests(human, status, created_at, request_id);

                CREATE INDEX IF NOT EXISTS idx_human_requests_status_created
                  ON human_requests(status, created_at, request_id);
                CREATE TABLE IF NOT EXISTS llm_calls (
                  call_id TEXT COLLATE BINARY PRIMARY KEY,
                  pid TEXT,
                  image_id TEXT,
                  purpose TEXT NOT NULL,
                  status TEXT NOT NULL,
                  api TEXT,
                  model TEXT,
                  request_id TEXT,
                  response_id TEXT,
                  messages_json TEXT NOT NULL,
                  tools_json TEXT NOT NULL,
                  request_options_json TEXT NOT NULL,
                  response_content TEXT NOT NULL,
                  tool_calls_json TEXT NOT NULL,
                  reasoning_json TEXT,
                  usage_json TEXT NOT NULL,
                  raw_response_json TEXT,
                  observability_json TEXT NOT NULL DEFAULT '{}',
                  error TEXT,
                  created_at TEXT COLLATE BINARY NOT NULL,
                  completed_at TEXT,
                  payload_retention_tier TEXT NOT NULL DEFAULT 'full' CHECK (
                    payload_retention_tier IN ('full', 'summary', 'hash_only')
                  )
                );
                CREATE INDEX IF NOT EXISTS idx_llm_calls_pid_created
                  ON llm_calls(pid, created_at);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_request_id
                  ON llm_calls(request_id);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_response_id
                  ON llm_calls(response_id);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_retention_eligible
                  ON llm_calls(created_at COLLATE BINARY, call_id COLLATE BINARY,
                               status, completed_at, payload_retention_tier)
                  WHERE status IN ('ok', 'error') AND completed_at IS NOT NULL
                    AND payload_retention_tier IN ('full', 'summary');
                CREATE TABLE IF NOT EXISTS llm_pending_actions (
                  pid TEXT PRIMARY KEY,
                  resume_token TEXT,
                  llm_operation_id TEXT,
                  tool_operation_id TEXT,
                  wait_type TEXT NOT NULL,
                  request_id TEXT,
                  child_pid TEXT,
                  response_id TEXT,
                  tool_call_id TEXT,
                  tool_name TEXT,
                  filters_json TEXT NOT NULL,
                  action_json TEXT NOT NULL,
                  data_flow_context_json TEXT NOT NULL,
                  content_preview TEXT NOT NULL,
                  tool_call_count INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_tool_outputs (
                  pid TEXT NOT NULL,
                  response_id TEXT NOT NULL,
                  call_id TEXT NOT NULL,
                  tool_name TEXT,
                  output_text TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(pid, response_id, call_id)
                );

                CREATE INDEX IF NOT EXISTS idx_llm_tool_outputs_response
                  ON llm_tool_outputs(pid, response_id);

                CREATE TABLE IF NOT EXISTS llm_context_generations (
                  pid TEXT PRIMARY KEY,
                  generation TEXT NOT NULL,
                  labels_schema_version INTEGER NOT NULL DEFAULT 1,
                  labels_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS process_messages (
                  message_id TEXT PRIMARY KEY,
                  sender TEXT NOT NULL,
                  recipient_pid TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  channel TEXT NOT NULL DEFAULT 'default',
                  correlation_id TEXT,
                  reply_to TEXT,
                  subject TEXT NOT NULL,
                  body TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  acked_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_process_messages_recipient_status_kind
                  ON process_messages(recipient_pid, status, kind, channel, created_at);

                CREATE INDEX IF NOT EXISTS idx_process_messages_recipient_created
                  ON process_messages(recipient_pid, created_at, message_id);

                CREATE INDEX IF NOT EXISTS idx_process_messages_correlation
                  ON process_messages(recipient_pid, correlation_id, status, created_at);

                CREATE TABLE IF NOT EXISTS agent_ratings (
                  rating_id TEXT PRIMARY KEY,
                  pid TEXT NOT NULL,
                  score INTEGER NOT NULL,
                  comment TEXT NOT NULL,
                  rater TEXT NOT NULL,
                  source TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(pid, rater, source)
                );

                CREATE INDEX IF NOT EXISTS idx_agent_ratings_pid
                  ON agent_ratings(pid, updated_at);

                CREATE TABLE IF NOT EXISTS skills (
                  skill_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  version TEXT NOT NULL,
                  package_json TEXT NOT NULL,
                  source_type TEXT NOT NULL,
                  source TEXT,
                  package_sha256 TEXT NOT NULL,
                  registered_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_skills_name
                  ON skills(name);

                CREATE TABLE IF NOT EXISTS skill_trust (
                  trust_id TEXT PRIMARY KEY,
                  source_type TEXT NOT NULL,
                  source TEXT NOT NULL,
                  package_sha256 TEXT NOT NULL,
                  trusted_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  UNIQUE(source_type, source, package_sha256)
                );

                CREATE TABLE IF NOT EXISTS jsonrpc_endpoints (
                  endpoint_id TEXT PRIMARY KEY,
                  spec_json TEXT NOT NULL,
                  registered_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mcp_servers (
                  server_id TEXT PRIMARY KEY,
                  spec_json TEXT NOT NULL,
                  registered_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                  image_id TEXT PRIMARY KEY,
                  manifest_json TEXT NOT NULL,
                  registered_by TEXT NOT NULL,
                  source TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS image_artifacts (
                  artifact_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  artifact_json TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tools (
                  tool_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  spec_json TEXT NOT NULL,
                  scope TEXT NOT NULL,
                  registered_by TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  ephemeral INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_candidates (
                  candidate_id TEXT PRIMARY KEY,
                  pid TEXT NOT NULL,
                  spec_json TEXT NOT NULL,
                  source_code TEXT NOT NULL,
                  tests_json TEXT NOT NULL,
                  requested_capabilities_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  registered_tool_id TEXT,
                  validation_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                """
            )
            self._create_process_tool_binding_schema()
            self._create_object_task_schema()
            self._create_runtime_publication_schema()
            self._create_external_effect_indexes()
            self._create_runtime_publication_indexes()
            self._create_runtime_module_schema()
            self._finish_schema_initialization()

    def _create_process_tool_binding_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS process_tool_bindings (
              pid TEXT COLLATE BINARY NOT NULL,
              binding_kind TEXT NOT NULL,
              tool_name TEXT COLLATE BINARY NOT NULL,
              tool_id TEXT NOT NULL,
              jit_rehydration_eligible INTEGER NOT NULL DEFAULT 0 CHECK (
                jit_rehydration_eligible IN (0, 1)
                AND (jit_rehydration_eligible = 0 OR binding_kind = 'callable')
              ),
              PRIMARY KEY(pid, binding_kind, tool_name),
              FOREIGN KEY(pid) REFERENCES processes(pid) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_process_tool_bindings_tool_pid
              ON process_tool_bindings(tool_id, pid, binding_kind, tool_name);
            CREATE INDEX IF NOT EXISTS idx_process_tool_bindings_jit_eligible_recovery
              ON process_tool_bindings(
                pid COLLATE BINARY, tool_name COLLATE BINARY, tool_id
              )
              WHERE jit_rehydration_eligible = 1;
            """
        )

    def _create_object_task_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS object_tasks (
              task_id TEXT COLLATE BINARY PRIMARY KEY,
              owner_oid TEXT NOT NULL,
              creator_pid TEXT NOT NULL,
              runner_pid TEXT,
              tool TEXT NOT NULL,
              tool_id TEXT,
              status TEXT NOT NULL,
              notification_status TEXT NOT NULL DEFAULT 'none',
              notification_recipient_pid TEXT,
              notification_json TEXT NOT NULL,
              owner_watch_json TEXT NOT NULL,
              result_oid TEXT,
              error TEXT,
              wait_json TEXT NOT NULL,
              created_at TEXT COLLATE BINARY NOT NULL,
              updated_at TEXT NOT NULL,
              started_at TEXT,
              completed_at TEXT
            );
            """
        )

    def _finish_schema_initialization(self) -> None:
        self._initialize_v3_schema()
        self._create_llm_call_indexes()
        self._create_object_task_indexes()

    def _create_llm_call_indexes(self) -> None:
        self._execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_calls_provider_chain_head
              ON llm_calls(
                pid, purpose, created_at COLLATE BINARY, call_id COLLATE BINARY
              );
            """
        )

    def _create_object_task_indexes(self) -> None:
        self._execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_object_tasks_owner_status
              ON object_tasks(owner_oid, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_object_tasks_creator_status
              ON object_tasks(creator_pid, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_object_tasks_runner
              ON object_tasks(runner_pid);
            """
        )

    def _create_runtime_module_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS runtime_modules (
              module_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              entrypoint TEXT NOT NULL,
              manifest_path TEXT NOT NULL,
              manifest_sha256 TEXT NOT NULL,
              source_path TEXT NOT NULL,
              source_sha256 TEXT NOT NULL,
              status TEXT NOT NULL,
              loaded_at TEXT,
              registered_json TEXT NOT NULL,
              error TEXT,
              metadata_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )

    def _create_external_effect_indexes(self) -> None:
        self._execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_external_effects_created
              ON external_effects(
                created_at COLLATE BINARY, effect_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_external_effects_pid_created
              ON external_effects(
                pid, created_at COLLATE BINARY, effect_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_external_effects_recovery_state
              ON external_effects(
                effect_state, created_at COLLATE BINARY,
                effect_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_external_effects_recovery_transaction
              ON external_effects(
                effect_state, transaction_state,
                created_at COLLATE BINARY, effect_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_external_effects_transaction_state
              ON external_effects(transaction_state, effect_id);
            CREATE INDEX IF NOT EXISTS idx_external_effects_retention_eligible
              ON external_effects(
                created_at COLLATE BINARY, effect_id COLLATE BINARY,
                transaction_state,
                payload_retention_tier
              )
              WHERE effect_state = 'finalized'
                AND transaction_state IN ('committed', 'failed', 'compensated')
                AND payload_retention_tier IN ('full', 'summary');
            """
        )

    def _create_runtime_publication_indexes(self) -> None:
        self._execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_runtime_publications_state
              ON runtime_publications(
                state, created_at COLLATE BINARY,
                publication_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_pid
              ON runtime_publications(
                pid COLLATE BINARY, created_at COLLATE BINARY,
                publication_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_operation_reconciliation
              ON runtime_publications(
                state, kind, operation_reconciled,
                created_at COLLATE BINARY, publication_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_pid_kind
              ON runtime_publications(
                pid COLLATE BINARY, kind, publication_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_invalid_domain
              ON runtime_publications(publication_id)
              WHERE kind NOT IN (
                'process_launch', 'process_exec', 'checkpoint_restore'
              ) OR state NOT IN (
                'planning', 'applying', 'reconciliation_pending',
                'committed', 'rollback_pending', 'rolled_back', 'failed', 'manual'
              ) OR operation_reconciled NOT IN (0, 1);

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_payload_delivery_page
              ON runtime_publications(
                payload_delivery_state,
                created_at COLLATE BINARY, publication_id COLLATE BINARY
              )
              WHERE kind = 'checkpoint_restore'
                AND state = 'committed'
                AND phase = 'reconciled'
                AND payload_delivery_state IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_payload_delivery_attempt
              ON runtime_publications(
                payload_delivery_attempt_id, payload_delivery_state,
                created_at COLLATE BINARY, publication_id COLLATE BINARY
              )
              WHERE kind = 'checkpoint_restore'
                AND state = 'committed'
                AND phase = 'reconciled'
                AND payload_delivery_attempt_id IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_runtime_publications_payload_delivery_guard
              ON runtime_publications(
                payload_delivery_attempt_id, payload_delivery_state,
                owner_instance_id, operation_reconciled,
                created_at COLLATE BINARY, publication_id COLLATE BINARY
              )
              WHERE kind = 'checkpoint_restore'
                AND state = 'committed'
                AND phase = 'reconciled'
                AND payload_delivery_attempt_id IS NOT NULL;

            """
        )
        self._validate_runtime_publication_domain()

    def _create_runtime_publication_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS runtime_publications (
              publication_id TEXT COLLATE BINARY PRIMARY KEY,
              kind TEXT NOT NULL CHECK (
                kind IN ('process_launch', 'process_exec', 'checkpoint_restore')
              ),
              pid TEXT COLLATE BINARY NOT NULL,
              owner_instance_id TEXT NOT NULL,
              state TEXT NOT NULL CHECK (
                state IN (
                  'planning', 'applying', 'reconciliation_pending',
                  'committed', 'rollback_pending', 'rolled_back',
                  'failed', 'manual'
                )
              ),
              phase TEXT NOT NULL,
              plan_json TEXT NOT NULL,
              receipt_json TEXT NOT NULL,
              error_json TEXT,
              operation_reconciled INTEGER NOT NULL DEFAULT 0 CHECK (
                operation_reconciled IN (0, 1)
              ),
              payload_delivery_state TEXT CHECK (
                payload_delivery_state IN ('pending', 'confirmed', 'completed')
                OR payload_delivery_state IS NULL
              ),
              payload_delivery_attempt_id TEXT COLLATE BINARY,
              payload_delivery_started_at TEXT COLLATE BINARY,
              created_at TEXT COLLATE BINARY NOT NULL,
              updated_at TEXT NOT NULL,
              CHECK (
                (payload_delivery_attempt_id IS NULL)
                = (payload_delivery_started_at IS NULL)
              ),
              CHECK (
                (
                  payload_delivery_state IS NULL
                  AND payload_delivery_attempt_id IS NULL
                ) OR (
                  payload_delivery_state IS NOT NULL
                  AND
                  kind = 'checkpoint_restore'
                  AND state = 'committed'
                  AND phase = 'reconciled'
                  AND (
                    (
                      payload_delivery_state = 'pending'
                      AND payload_delivery_attempt_id IS NULL
                    ) OR (
                      payload_delivery_state = 'confirmed'
                      AND payload_delivery_attempt_id IS NOT NULL
                    ) OR (
                      payload_delivery_state = 'completed'
                      AND payload_delivery_attempt_id IS NOT NULL
                    )
                  )
                )
              )
            );

            CREATE TABLE IF NOT EXISTS checkpoint_payload_delivery_attempts (
              attempt_id TEXT COLLATE BINARY PRIMARY KEY,
              owner_instance_id TEXT NOT NULL,
              state TEXT NOT NULL CHECK (state IN ('preparing', 'acked', 'aborted')),
              started_at TEXT COLLATE BINARY NOT NULL,
              acked_at TEXT,
              updated_at TEXT NOT NULL,
              CHECK ((state = 'acked') = (acked_at IS NOT NULL))
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoint_payload_delivery_attempts_state
              ON checkpoint_payload_delivery_attempts(
                state, started_at COLLATE BINARY, attempt_id COLLATE BINARY
              );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_checkpoint_payload_delivery_attempts_preparing
              ON checkpoint_payload_delivery_attempts(state)
              WHERE state = 'preparing';
            """
        )

    def _validate_runtime_publication_domain(self) -> None:
        rows = self._query(
            "SELECT publication_id, kind, state, operation_reconciled "
            "FROM runtime_publications "
            "INDEXED BY idx_runtime_publications_invalid_domain "
            "WHERE kind NOT IN "
            "('process_launch', 'process_exec', 'checkpoint_restore') "
            "OR state NOT IN ('planning', 'applying', 'reconciliation_pending', "
            "'committed', 'rollback_pending', 'rolled_back', 'failed', 'manual') "
            "OR operation_reconciled NOT IN (0, 1) LIMIT 1"
        )
        if rows:
            row = rows[0]
            raise ValidationError(
                "invalid durable runtime publication domain: "
                f"{row['publication_id']!r} kind={row['kind']!r} "
                f"state={row['state']!r} "
                f"operation_reconciled={row['operation_reconciled']!r}"
            )

    def _initialize_v3_schema(self) -> None:
        """Finish the fresh 0.3 schema without migration or backfill paths."""

        self._create_v3_data_flow_schema()
        self._create_v3_operation_schema()
        self._create_v3_recovery_indexes()
        now = utc_now()
        self.conn.execute(
            "INSERT OR IGNORE INTO runtime_counters (counter_name, value) VALUES (?, ?)",
            ("external_effect_ledger", 0),
        )
        for counter_name in (
            "jsonrpc_registry_generation",
            "mcp_registry_generation",
        ):
            self.conn.execute(
                "INSERT OR IGNORE INTO runtime_counters (counter_name, value) VALUES (?, ?)",
                (counter_name, 0),
            )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO object_namespaces (
                namespace, parent_namespace, metadata_json, created_by,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.SYSTEM_NAMESPACE, None, dumps({"kind": "root"}), "runtime", now, now),
        )

    def _create_v3_recovery_indexes(self) -> None:
        """Create the canonical bounded-startup and reconciliation indexes."""

        self._execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_processes_status_created
              ON processes(
                status, created_at COLLATE BINARY, pid COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_processes_created
              ON processes(created_at COLLATE BINARY, pid COLLATE BINARY);
            CREATE INDEX IF NOT EXISTS idx_processes_execution_recovery
              ON processes(status, pid COLLATE BINARY, execution_owner_id);
            CREATE INDEX IF NOT EXISTS idx_processes_parent_created
              ON processes(
                parent_pid COLLATE BINARY, created_at COLLATE BINARY,
                pid COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_tool_candidates_jit_rehydration
              ON tool_candidates(registered_tool_id, status, pid, candidate_id);
            CREATE INDEX IF NOT EXISTS idx_tool_candidates_owner_registration
              ON tool_candidates(pid, registered_tool_id, candidate_id, status);

            CREATE INDEX IF NOT EXISTS idx_capabilities_subject_status
              ON capabilities(subject, status);
            CREATE INDEX IF NOT EXISTS idx_capabilities_subject_resource_status
              ON capabilities(subject, resource, status);
            CREATE INDEX IF NOT EXISTS idx_capabilities_parent
              ON capabilities(parent_cap_id);
            CREATE INDEX IF NOT EXISTS idx_capability_reservations_cap_status
              ON capability_use_reservations(cap_id, status);
            CREATE INDEX IF NOT EXISTS idx_capability_reservations_recovery
              ON capability_use_reservations(
                status, created_at COLLATE BINARY,
                reservation_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_events_gui_snapshot_visible_created
              ON events(
                gui_snapshot_visible, created_at COLLATE BINARY,
                event_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_audit_gui_snapshot_visible_created
              ON audit_records(gui_snapshot_visible, timestamp, record_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_external_effects_pid_idempotency
              ON external_effects(pid, idempotency_key)
              WHERE idempotency_key IS NOT NULL;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_objects_namespace_name_live
              ON objects(namespace, name)
              WHERE lifecycle_state = 'live';
            CREATE INDEX IF NOT EXISTS idx_objects_created_by
              ON objects(created_by, created_at, oid);
            CREATE INDEX IF NOT EXISTS idx_objects_owner_live
              ON objects(owner_kind, owner_id, lifecycle_state, created_at, oid);
            CREATE INDEX IF NOT EXISTS idx_objects_namespace_updated
              ON objects(namespace, updated_at, created_at, oid);
            CREATE INDEX IF NOT EXISTS idx_objects_namespace_type_updated
              ON objects(namespace, type, updated_at, created_at, oid);
            CREATE INDEX IF NOT EXISTS idx_objects_payload_recovery
              ON objects(
                created_at COLLATE BINARY, oid COLLATE BINARY
              )
              WHERE lifecycle_state = 'live'
                AND payload_json IN (
                  '{"present": true, "storage": "runtime_memory"}',
                  '{"storage": "runtime_memory", "present": true}',
                  '{"present":true,"storage":"runtime_memory"}',
                  '{"storage":"runtime_memory","present":true}'
                );
            CREATE INDEX IF NOT EXISTS idx_object_namespaces_created_by
              ON object_namespaces(created_by, namespace);
            CREATE INDEX IF NOT EXISTS idx_object_namespaces_parent
              ON object_namespaces(parent_namespace, namespace);

            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_status
              ON object_tasks(
                status, created_at COLLATE BINARY, task_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_result
              ON object_tasks(
                status, result_oid, created_at COLLATE BINARY,
                task_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_notification
              ON object_tasks(
                status, notification_status, notification_recipient_pid,
                created_at COLLATE BINARY, task_id COLLATE BINARY
              );
            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_active_eligible
              ON object_tasks(
                created_at COLLATE BINARY, task_id COLLATE BINARY
              )
              WHERE status IN (
                'queued', 'running', 'waiting_human',
                'waiting_process', 'waiting_message'
              );
            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_result_eligible
              ON object_tasks(
                created_at COLLATE BINARY, task_id COLLATE BINARY
              )
              WHERE status = 'succeeded' AND result_oid IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_object_tasks_recovery_notification_eligible
              ON object_tasks(
                created_at COLLATE BINARY, task_id COLLATE BINARY
              )
              WHERE status IN ('succeeded', 'failed', 'cancelled')
                AND notification_status IN ('none', 'failed')
                AND notification_recipient_pid IS NOT NULL;
            """
        )

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> Any:
        with self._join_or_begin_transaction() as cur:
            return cur.execute(sql, tuple(params))

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[Any]:
        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            return list(self.conn.execute(sql, tuple(params)))

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        """Run direct SQL mutations atomically.

        Object payloads live outside SQL. Every transaction frame lazily
        journals its first mutation of each OID so nested rollback restores the
        savepoint value and nested commit propagates the earliest before-image
        to its parent. ``include_object_payloads=True`` remains compatible and
        eagerly captures the currently cached payloads.
        """

        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            payload_frame = (
                {oid: deepcopy(payload) for oid, payload in self._object_payloads.items()}
                if include_object_payloads
                else {}
            )
            depth = self._transaction_depth
            savepoint = f"agent_libos_sp_{depth}"
            if depth == 0:
                self.conn.execute("BEGIN")
            else:
                self.conn.execute(f"SAVEPOINT {savepoint}")
            self._payload_transaction_frames.append(payload_frame)
            self._transaction_depth += 1
            try:
                yield self.conn.cursor()
            except BaseException:
                self._rollback_scope(
                    depth=depth,
                    savepoint=savepoint,
                    payload_frame=payload_frame,
                    operation="transaction body failed",
                )
                raise
            else:
                try:
                    if depth == 0:
                        with self._admission_commit_scope():
                            self.conn.commit()
                    else:
                        self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                        self._merge_payload_frame_into_parent(payload_frame)
                except BaseException:
                    self._rollback_scope(
                        depth=depth,
                        savepoint=savepoint,
                        payload_frame=payload_frame,
                        operation="transaction commit failed" if depth == 0 else "savepoint release failed",
                    )
                    raise
            finally:
                self._transaction_depth -= 1
                self._payload_transaction_frames.pop()

    @contextmanager
    def _join_or_begin_transaction(self):
        """Join a caller transaction or create one for a repository mutation.

        The store lock must cover the depth check: another thread may own an
        outer transaction, and observing its depth without the lock could make
        this thread incorrectly run an auto-committed statement.  This explicit
        outer scope is required even for one statement because the PostgreSQL
        adapter keeps connection-level autocommit enabled so read queries do not
        leave implicit transactions open.  When the caller already owns the
        transaction, do not add a savepoint or a new post-commit failure boundary
        around the repository write.
        """

        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            if self._transaction_depth:
                yield self.conn.cursor()
            else:
                with self.transaction() as cur:
                    yield cur

    def validate_table_identifier(self, table: str) -> str:
        if table not in self.ALLOWED_TABLES:
            raise ValidationError(f"unsupported runtime store table: {table}")
        return table

    def validate_column_identifier(self, table: str, column: str) -> str:
        with self._lock:
            self._ensure_healthy()
            self._ensure_store_scope_admitted()
            self.validate_table_identifier(table)
            if (
                not column
                or not column.replace("_", "").isalnum()
                or not column[0].isalpha()
            ):
                raise ValidationError(
                    f"unsupported runtime store column: {table}.{column}"
                )
            columns = {
                str(row["name"])
                for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                raise ValidationError(
                    f"unsupported runtime store column: {table}.{column}"
                )
            return column

    def insert_object(self, obj: AgentObject) -> None:
        with self.transaction(include_object_payloads=True) as cur:
            self._set_cached_object_payload(obj.oid, obj.payload)
            cur.execute(
                """
                INSERT INTO objects (
                    oid, namespace, name, type, schema_version, payload_json, metadata_json,
                    provenance_json, version, immutable, created_by, owner_kind, owner_id,
                    lifecycle_state, deleted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj.oid,
                    obj.namespace,
                    obj.name,
                    obj.type.value,
                    obj.schema_version,
                    dumps(self.payload_marker(present=True)),
                    dumps(obj.metadata),
                    dumps(obj.provenance),
                    obj.version,
                    int(obj.immutable),
                    obj.created_by,
                    obj.owner_kind.value,
                    obj.owner_id,
                    obj.lifecycle_state.value,
                    obj.deleted_at,
                    obj.created_at,
                    obj.updated_at,
                ),
            )

    def update_object(
        self,
        obj: AgentObject,
        *,
        expected_version: int | None = None,
        expected_owner_kind: ObjectOwnerKind | str | None = None,
        expected_owner_id: str | None = None,
    ) -> bool:
        with self.transaction(include_object_payloads=True) as cur:
            where = ["oid = ?", "lifecycle_state = ?"]
            where_params: list[Any] = [obj.oid, ObjectLifecycleState.LIVE.value]
            if expected_version is not None:
                where.append("version = ?")
                where_params.append(expected_version)
            if expected_owner_kind is not None:
                where.append("owner_kind = ?")
                where_params.append(str(expected_owner_kind))
            if expected_owner_id is not None:
                where.append("owner_id = ?")
                where_params.append(expected_owner_id)
            updated = cur.execute(
                """
                UPDATE objects
                   SET namespace = ?, name = ?, type = ?, schema_version = ?, payload_json = ?, metadata_json = ?,
                       provenance_json = ?, version = ?, immutable = ?, created_by = ?,
                       owner_kind = ?, owner_id = ?, lifecycle_state = ?, deleted_at = ?,
                       created_at = ?, updated_at = ?
                 WHERE """
                + " AND ".join(where),
                (
                    obj.namespace,
                    obj.name,
                    obj.type.value,
                    obj.schema_version,
                    dumps(self.payload_marker(present=True)),
                    dumps(obj.metadata),
                    dumps(obj.provenance),
                    obj.version,
                    int(obj.immutable),
                    obj.created_by,
                    obj.owner_kind.value,
                    obj.owner_id,
                    obj.lifecycle_state.value,
                    obj.deleted_at,
                    obj.created_at,
                    obj.updated_at,
                    *where_params,
                ),
            )
            if updated.rowcount != 1:
                return False
            self._set_cached_object_payload(obj.oid, obj.payload)
            return True

    def get_object(self, oid: str) -> AgentObject | None:
        rows = self._query(
            "SELECT * FROM objects WHERE oid = ? AND lifecycle_state = ?",
            (oid, ObjectLifecycleState.LIVE.value),
        )
        if not rows or not self.has_object_payload(oid, row=rows[0]):
            return None
        return self._row_to_object(rows[0])

    def get_object_by_name(self, name: str, namespace: str) -> AgentObject | None:
        rows = self._query(
            "SELECT * FROM objects WHERE namespace = ? AND name = ? AND lifecycle_state = ?",
            (namespace, name, ObjectLifecycleState.LIVE.value),
        )
        if not rows or not self.has_object_payload(str(rows[0]["oid"]), row=rows[0]):
            return None
        return self._row_to_object(rows[0])

    def get_object_ref_by_name(self, name: str, namespace: str) -> dict[str, Any] | None:
        rows = self._query(
            """
            SELECT oid, namespace, name
              FROM objects
             WHERE namespace = ? AND name = ? AND lifecycle_state = ?
            """,
            (namespace, name, ObjectLifecycleState.LIVE.value),
        )
        return self._row_to_dict(rows[0]) if rows else None

    def object_name_exists(self, name: str, namespace: str, except_oid: str | None = None) -> bool:
        rows = self._query(
            "SELECT oid FROM objects WHERE namespace = ? AND name = ? AND lifecycle_state = ?",
            (namespace, name, ObjectLifecycleState.LIVE.value),
        )
        return any(row["oid"] != except_oid for row in rows)

    def list_objects(self, namespace: str | None = None) -> list[AgentObject]:
        if namespace is None:
            rows = self._query(
                "SELECT * FROM objects WHERE lifecycle_state = ? ORDER BY updated_at DESC, created_at DESC, oid ASC",
                (ObjectLifecycleState.LIVE.value,),
            )
        else:
            rows = self._query(
                """
                SELECT * FROM objects
                 WHERE namespace = ? AND lifecycle_state = ?
                 ORDER BY updated_at DESC, created_at DESC, oid ASC
                """,
                (namespace, ObjectLifecycleState.LIVE.value),
            )
        return [
            self._row_to_object(row)
            for row in rows
            if self.has_object_payload(str(row["oid"]), row=row)
        ]

    def list_object_oids_created_by(self, created_by: str) -> list[str]:
        rows = self._query(
            "SELECT oid FROM objects WHERE created_by = ? AND lifecycle_state = ? ORDER BY created_at",
            (created_by, ObjectLifecycleState.LIVE.value),
        )
        return [str(row["oid"]) for row in rows]

    def list_objects_created_by(self, created_by: str) -> list[AgentObject]:
        rows = self._query(
            "SELECT * FROM objects WHERE created_by = ? AND lifecycle_state = ? ORDER BY created_at, oid",
            (created_by, ObjectLifecycleState.LIVE.value),
        )
        return [
            self._row_to_object(row)
            for row in rows
            if self.has_object_payload(str(row["oid"]), row=row)
        ]

    def list_object_oids_owned_by(self, owner_kind: str | ObjectOwnerKind, owner_id: str) -> list[str]:
        rows = self._query(
            """
            SELECT oid FROM objects
             WHERE owner_kind = ? AND owner_id = ? AND lifecycle_state = ?
             ORDER BY created_at, oid
            """,
            (str(owner_kind), owner_id, ObjectLifecycleState.LIVE.value),
        )
        return [str(row["oid"]) for row in rows]

    def list_objects_owned_by(self, owner_kind: str | ObjectOwnerKind, owner_id: str) -> list[AgentObject]:
        rows = self._query(
            """
            SELECT * FROM objects
             WHERE owner_kind = ? AND owner_id = ? AND lifecycle_state = ?
             ORDER BY created_at, oid
            """,
            (str(owner_kind), owner_id, ObjectLifecycleState.LIVE.value),
        )
        return [
            self._row_to_object(row)
            for row in rows
            if self.has_object_payload(str(row["oid"]), row=row)
        ]

    def delete_object(
        self,
        oid: str,
        *,
        expected_version: int | None = None,
        expected_owner_kind: ObjectOwnerKind | str | None = None,
        expected_owner_id: str | None = None,
    ) -> bool:
        now = utc_now()
        with self.transaction(include_object_payloads=True) as cur:
            where = ["oid = ?", "lifecycle_state = ?"]
            where_params: list[Any] = [oid, ObjectLifecycleState.LIVE.value]
            if expected_version is not None:
                where.append("version = ?")
                where_params.append(expected_version)
            if expected_owner_kind is not None:
                where.append("owner_kind = ?")
                where_params.append(str(expected_owner_kind))
            if expected_owner_id is not None:
                where.append("owner_id = ?")
                where_params.append(expected_owner_id)
            released = cur.execute(
                """
                UPDATE objects
                   SET payload_json = ?, lifecycle_state = ?, deleted_at = ?, updated_at = ?
                 WHERE """
                + " AND ".join(where),
                (
                    dumps(self.payload_marker(present=False)),
                    ObjectLifecycleState.RELEASED.value,
                    now,
                    now,
                    *where_params,
                ),
            )
            if released.rowcount != 1:
                return False
            self._forget_cached_object_payload(oid)
            cur.execute("DELETE FROM object_links WHERE src_oid = ? OR dst_oid = ?", (oid, oid))
            return True

    def insert_namespace(self, namespace: ObjectNamespace) -> None:
        self._execute(
            """
            INSERT INTO object_namespaces (
                namespace, parent_namespace, metadata_json, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                namespace.namespace,
                namespace.parent_namespace,
                dumps(namespace.metadata),
                namespace.created_by,
                namespace.created_at,
                namespace.updated_at,
            ),
        )

    def get_namespace(self, namespace: str) -> ObjectNamespace | None:
        rows = self._query("SELECT * FROM object_namespaces WHERE namespace = ?", (namespace,))
        return self._row_to_namespace(rows[0]) if rows else None

    def namespace_exists(self, namespace: str) -> bool:
        rows = self._query("SELECT 1 FROM object_namespaces WHERE namespace = ?", (namespace,))
        return bool(rows)

    def list_namespaces(self, parent_namespace: str | None = None) -> list[ObjectNamespace]:
        if parent_namespace is None:
            rows = self._query("SELECT * FROM object_namespaces ORDER BY namespace")
        else:
            rows = self._query(
                "SELECT * FROM object_namespaces WHERE parent_namespace = ? ORDER BY namespace",
                (parent_namespace,),
            )
        return [self._row_to_namespace(row) for row in rows]

    def list_namespaces_created_by(self, created_by: str) -> list[ObjectNamespace]:
        rows = self._query(
            "SELECT * FROM object_namespaces WHERE created_by = ? ORDER BY namespace",
            (created_by,),
        )
        return [self._row_to_namespace(row) for row in rows]

    def insert_link(self, link: ObjectLink) -> None:
        self._execute(
            "INSERT INTO object_links VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                link.link_id,
                link.src,
                link.relation.value,
                link.dst,
                dumps(link.metadata),
                link.created_by,
                link.created_at,
            ),
        )

    def list_links(self, src: str | None = None, dst: str | None = None) -> list[ObjectLink]:
        if src is not None:
            rows = self._query("SELECT * FROM object_links WHERE src_oid = ?", (src,))
        elif dst is not None:
            rows = self._query("SELECT * FROM object_links WHERE dst_oid = ?", (dst,))
        else:
            rows = self._query("SELECT * FROM object_links")
        return [self._row_to_link(row) for row in rows]

    def insert_process(self, process: AgentProcess) -> None:
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO processes (
                    pid, parent_pid, image_id, status, goal_oid, memory_view_json,
                    capabilities_json, loaded_skills_json, tool_table_json, model_tool_table_json, event_cursor,
                    checkpoint_head, status_message, wait_state_json, outcome_json,
                    state_generation, resource_budget_json, resource_usage_json,
                    working_directory, llm_profile_id, revision, execution_generation,
                    execution_owner_id, execution_lease_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._process_params(process),
            )
            self._replace_process_tool_bindings(
                cur,
                process.pid,
                process.tool_table,
                process.model_tool_table,
            )
            self.observe_process_concurrency(
                process.pid,
                revision=process.revision,
                execution_generation=process.execution_generation,
                state_generation=process.state_generation,
                cursor=cur,
            )

    @staticmethod
    def _replace_process_tool_bindings(
        cursor: Any,
        pid: str,
        tool_table: Mapping[str, str],
        model_tool_table: Mapping[str, str],
    ) -> None:
        """Replace the indexed reverse projection of one process's tool maps."""

        if not isinstance(pid, str) or not pid or "\x00" in pid:
            raise ValidationError("process tool bindings require a valid process ID")
        rows: list[tuple[str, str, str, str]] = []
        for binding_kind, bindings in (
            ("callable", tool_table),
            ("model", model_tool_table),
        ):
            if not isinstance(bindings, Mapping):
                raise ValidationError("process tool bindings require object mappings")
            for tool_name, tool_id in bindings.items():
                if (
                    not isinstance(tool_name, str)
                    or not tool_name
                    or "\x00" in tool_name
                    or not isinstance(tool_id, str)
                    or not tool_id
                    or "\x00" in tool_id
                ):
                    raise ValidationError(
                        "process tool bindings require non-empty string names and identities"
                    )
                rows.append((pid, binding_kind, tool_name, tool_id))
        cursor.execute("DELETE FROM process_tool_bindings WHERE pid = ?", (pid,))
        if rows:
            cursor.executemany(
                "INSERT INTO process_tool_bindings "
                "(pid, binding_kind, tool_name, tool_id, "
                "jit_rehydration_eligible) VALUES (?, ?, ?, ?, ?)",
                ((*row, 0) for row in rows),
            )
            SQLRuntimeStore._refresh_process_binding_jit_eligibility(
                cursor,
                pid=pid,
            )

    @staticmethod
    def _refresh_process_binding_jit_eligibility(
        cursor: Any,
        *,
        pid: str | None = None,
        tool_id: str | None = None,
    ) -> None:
        """Synchronize the indexed JIT projection with durable Tool metadata."""

        if (pid is None) == (tool_id is None):
            raise ValueError(
                "JIT binding eligibility refresh requires exactly one identity"
            )
        where_column = "pid" if pid is not None else "tool_id"
        where_value = pid if pid is not None else tool_id
        cursor.execute(
            "UPDATE process_tool_bindings "
            "SET jit_rehydration_eligible = CASE "
            "WHEN binding_kind = ? AND EXISTS ("
            "SELECT 1 FROM tools "
            "WHERE tools.tool_id = process_tool_bindings.tool_id "
            "AND tools.ephemeral = 1"
            ") THEN 1 ELSE 0 END "
            f"WHERE {where_column} = ?",
            ("callable", where_value),
        )

    def tool_id_referenced_outside_process(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool:
        if (
            not isinstance(tool_id, str)
            or not tool_id
            or "\x00" in tool_id
            or not isinstance(excluding_pid, str)
            or not excluding_pid
            or "\x00" in excluding_pid
        ):
            raise ValidationError(
                "process tool reverse lookup requires valid tool and process identities"
            )
        rows = self._query(
            "SELECT 1 FROM process_tool_bindings "
            "WHERE tool_id = ? AND pid <> ? LIMIT 1",
            (tool_id, excluding_pid),
        )
        return bool(rows)

    def tool_id_referenced_outside_scope(
        self,
        tool_id: str,
        *,
        scoped_pids: Iterable[str],
    ) -> bool:
        if not isinstance(tool_id, str) or not tool_id or "\x00" in tool_id:
            raise ValidationError(
                "process tool reverse lookup requires a valid tool identity"
            )
        if isinstance(scoped_pids, (str, bytes, bytearray)):
            raise ValidationError("process tool scope must be an iterable of identities")
        selected_pids: set[str] = set()
        for pid in scoped_pids:
            if not isinstance(pid, str) or not pid or "\x00" in pid:
                raise ValidationError(
                    "process tool scope requires valid process identities"
                )
            selected_pids.add(pid)
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT DISTINCT pid FROM process_tool_bindings WHERE tool_id = ?",
                (tool_id,),
            )
            while True:
                row = cursor.fetchone()
                if row is None:
                    return False
                pid = row["pid"]
                if not isinstance(pid, str) or not pid or "\x00" in pid:
                    raise ValidationError(
                        "process tool reverse lookup returned an invalid process identity"
                    )
                if pid not in selected_pids:
                    return True

    def delete_tool_if_unreferenced(
        self,
        tool_id: str,
        *,
        excluding_pid: str,
    ) -> bool:
        if (
            not isinstance(tool_id, str)
            or not tool_id
            or "\x00" in tool_id
            or not isinstance(excluding_pid, str)
            or not excluding_pid
            or "\x00" in excluding_pid
        ):
            raise ValidationError(
                "tool deletion requires valid tool and process identities"
            )
        with self.transaction() as cursor:
            referenced = cursor.execute(
                "SELECT 1 FROM process_tool_bindings "
                "WHERE tool_id = ? AND pid <> ? LIMIT 1",
                (tool_id, excluding_pid),
            ).fetchone()
            if referenced is not None:
                return False
            cursor.execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))
            self._refresh_process_binding_jit_eligibility(
                cursor,
                tool_id=tool_id,
            )
            return True

    @staticmethod
    def _process_concurrency_counter_names(pid: str) -> tuple[str, str, str]:
        return (
            f"{_PROCESS_REVISION_COUNTER_PREFIX}{pid}",
            f"{_PROCESS_EXECUTION_COUNTER_PREFIX}{pid}",
            f"{_PROCESS_STATE_COUNTER_PREFIX}{pid}",
        )

    @staticmethod
    def _raise_runtime_counter_floor(cur: Any, counter_name: str, floor: int) -> None:
        selected_floor = max(0, int(floor))
        cur.execute(
            """
            INSERT INTO runtime_counters (counter_name, value)
            VALUES (?, ?)
            ON CONFLICT(counter_name) DO NOTHING
            """,
            (counter_name, selected_floor),
        )
        cur.execute(
            """
            UPDATE runtime_counters
               SET value = CASE WHEN value < ? THEN ? ELSE value END
             WHERE counter_name = ?
            """,
            (selected_floor, selected_floor, counter_name),
        )

    def observe_process_concurrency(
        self,
        pid: str,
        *,
        revision: int,
        execution_generation: int,
        state_generation: int | None = None,
        cursor: Any | None = None,
    ) -> None:
        """Persist monotonic process concurrency high-water marks."""

        if cursor is None:
            with self.transaction() as cur:
                self.observe_process_concurrency(
                    pid,
                    revision=revision,
                    execution_generation=execution_generation,
                    state_generation=state_generation,
                    cursor=cur,
                )
            return
        revision_counter, execution_counter, state_counter = (
            self._process_concurrency_counter_names(pid)
        )
        self._raise_runtime_counter_floor(cursor, revision_counter, revision)
        self._raise_runtime_counter_floor(
            cursor,
            execution_counter,
            execution_generation,
        )
        if state_generation is not None:
            self._raise_runtime_counter_floor(
                cursor,
                state_counter,
                state_generation,
            )

    def reserve_process_restore_epoch(
        self,
        pid: str,
        *,
        revision_floor: int,
        execution_generation_floor: int,
        state_generation_floor: int,
        cursor: Any | None = None,
    ) -> tuple[int, int, int]:
        """Allocate concurrency values newer than every previously observed value."""

        floor = ProcessRestoreEpoch(
            pid=pid,
            revision=max(0, int(revision_floor)),
            execution_generation=max(0, int(execution_generation_floor)),
            state_generation=max(0, int(state_generation_floor)),
        )
        reserved = self.reserve_process_restore_epochs((floor,), cursor=cursor)[0]
        return (
            reserved.revision,
            reserved.execution_generation,
            reserved.state_generation,
        )

    def reserve_process_restore_epochs(
        self,
        floors: Iterable[ProcessRestoreEpoch],
        *,
        cursor: Any | None = None,
    ) -> tuple[ProcessRestoreEpoch, ...]:
        """Reserve a deterministic, atomic batch of monotonic process epochs.

        Each PID is floored and incremented exactly once. Sorting identities
        before taking counter-row locks prevents overlapping PostgreSQL restores
        from acquiring them in conflicting orders.
        """

        selected = tuple(floors)
        if any(not isinstance(item, ProcessRestoreEpoch) for item in selected):
            raise ValidationError(
                "process restore epoch batch requires typed floor values"
            )
        selected_pids = [str(item.pid) for item in selected]
        if len(selected_pids) != len(set(selected_pids)):
            raise ValidationError(
                "process restore epoch batch must not contain duplicate PIDs"
            )
        if not selected:
            return ()
        if cursor is None:
            with self.transaction() as cur:
                return self.reserve_process_restore_epochs(
                    selected,
                    cursor=cur,
                )

        ordered = tuple(sorted(selected, key=lambda item: str(item.pid)))
        reserved_by_pid: dict[str, ProcessRestoreEpoch] = {}
        for offset in range(
            0,
            len(ordered),
            _PROCESS_RESTORE_EPOCH_PID_BATCH_SIZE,
        ):
            batch = ordered[
                offset : offset + _PROCESS_RESTORE_EPOCH_PID_BATCH_SIZE
            ]
            counter_rows: list[tuple[str, int]] = []
            counters_by_pid: dict[str, tuple[str, str, str]] = {}
            for floor in batch:
                counters = self._process_concurrency_counter_names(str(floor.pid))
                counters_by_pid[str(floor.pid)] = counters
                counter_rows.extend(
                    (
                        (counters[0], floor.revision),
                        (counters[1], floor.execution_generation),
                        (counters[2], floor.state_generation),
                    )
                )

            value_placeholders = ", ".join("(?, ?)" for _ in counter_rows)
            cursor.execute(
                "INSERT INTO runtime_counters (counter_name, value) VALUES "
                f"{value_placeholders} "
                "ON CONFLICT(counter_name) DO UPDATE SET value = CASE "
                "WHEN runtime_counters.value < excluded.value "
                "THEN excluded.value ELSE runtime_counters.value END",
                tuple(
                    value
                    for counter_name, counter_value in counter_rows
                    for value in (counter_name, counter_value)
                ),
            )
            counter_names = tuple(counter_name for counter_name, _ in counter_rows)
            name_placeholders = ", ".join("?" for _ in counter_names)
            cursor.execute(
                "UPDATE runtime_counters SET value = value + 1 "
                f"WHERE counter_name IN ({name_placeholders})",
                counter_names,
            )
            rows = list(
                cursor.execute(
                    "SELECT counter_name, value FROM runtime_counters "
                    f"WHERE counter_name IN ({name_placeholders}) "
                    "ORDER BY counter_name",
                    counter_names,
                )
            )
            values = {
                str(row["counter_name"]): int(row["value"])
                for row in rows
            }
            if len(values) != len(counter_names) or set(values) != set(counter_names):
                raise ValidationError(
                    "process restore epoch reservation returned an incomplete "
                    "or unexpected counter set"
                )
            for floor in batch:
                pid = str(floor.pid)
                revision_counter, execution_counter, state_counter = (
                    counters_by_pid[pid]
                )
                reserved_by_pid[pid] = ProcessRestoreEpoch(
                    pid=pid,
                    revision=values[revision_counter],
                    execution_generation=values[execution_counter],
                    state_generation=values[state_counter],
                )

        if set(reserved_by_pid) != set(selected_pids):
            raise ValidationError(
                "process restore epoch reservation returned an incomplete or "
                "unexpected process set"
            )
        return tuple(reserved_by_pid[pid] for pid in sorted(reserved_by_pid))

    def update_process(
        self,
        process: AgentProcess,
    ) -> None:
        """Compatibility CAS for callers not yet expressed as a domain patch.

        The revision carried by ``process`` is the expected revision.  A stale
        object can no longer overwrite a concurrent mutation, and an ordinary
        update can never move a terminal process to another state.
        """

        self._update_process(process, allow_state_transition=False)

    def _update_process(
        self,
        process: AgentProcess,
        *,
        allow_state_transition: bool,
    ) -> None:
        expected_revision = int(process.revision)
        with self.transaction() as cur:
            rows = list(cur.execute("SELECT * FROM processes WHERE pid = ?", (process.pid,)))
            if not rows:
                raise ProcessRevisionConflict(f"process no longer exists: {process.pid}")
            current = rows[0]
            generation, owner_id, lease_id, worker_clause, worker_params = (
                self._process_update_fence(
                    current,
                    process,
                    expected_revision=expected_revision,
                    cursor=cur,
                )
            )
            persisted = self._row_to_process(current)
            self._validate_process_state_write(
                process,
                persisted=persisted,
                allow_state_transition=allow_state_transition,
            )

            updated = cur.execute(
                f"""
                UPDATE processes
                   SET parent_pid = ?, image_id = ?, status = ?, goal_oid = ?,
                       memory_view_json = ?, capabilities_json = ?, loaded_skills_json = ?,
                       tool_table_json = ?, model_tool_table_json = ?, event_cursor = ?, checkpoint_head = ?,
                       status_message = ?, wait_state_json = ?, outcome_json = ?, state_generation = ?,
                       resource_budget_json = ?, resource_usage_json = ?,
                       working_directory = ?, llm_profile_id = ?, created_at = ?, updated_at = ?,
                       revision = revision + 1, execution_generation = ?,
                       execution_owner_id = ?, execution_lease_id = ?
                 WHERE pid = ? AND revision = ?{worker_clause}
                """,
                (
                    process.parent_pid,
                    process.image_id,
                    process.status.value,
                    process.goal_oid,
                    dumps(process.memory_view) if process.memory_view else None,
                    dumps(process.capabilities),
                    dumps(process.loaded_skills),
                    dumps(process.tool_table),
                    dumps(process.model_tool_table),
                    process.event_cursor,
                    process.checkpoint_head,
                    process.status_message,
                    dumps(process_wait_state_to_mapping(process.wait_state)),
                    dumps(process_outcome_to_mapping(process.outcome)),
                    process.state_generation,
                    dumps(process.resource_budget),
                    dumps(process.resource_usage),
                    process.working_directory,
                    process.llm_profile_id,
                    process.created_at,
                    process.updated_at,
                    generation,
                    owner_id,
                    lease_id,
                    process.pid,
                    expected_revision,
                    *worker_params,
                ),
            )
            if updated.rowcount != 1:
                raise ProcessRevisionConflict(f"process revision conflict for {process.pid}")
            self._replace_process_tool_bindings(
                cur,
                process.pid,
                process.tool_table,
                process.model_tool_table,
            )
            self.observe_process_concurrency(
                process.pid,
                revision=expected_revision + 1,
                execution_generation=generation,
                state_generation=process.state_generation,
                cursor=cur,
            )
        process.revision = expected_revision + 1
        process.execution_generation = generation
        process.execution_owner_id = owner_id
        process.execution_lease_id = lease_id

    @staticmethod
    def _validate_process_state_write(
        process: AgentProcess,
        *,
        persisted: AgentProcess | None = None,
        allow_state_transition: bool,
    ) -> None:
        validate_process_state_fields(
            process.status.value,
            process.wait_state,
            process.outcome,
        )
        if (
            not isinstance(process.state_generation, int)
            or isinstance(process.state_generation, bool)
            or process.state_generation < 0
        ):
            raise ValidationError(
                "process state_generation must be a non-negative integer"
            )
        if persisted is None:
            return
        semantic_changed = (
            process.status != persisted.status
            or process.wait_state != persisted.wait_state
            or process.outcome != persisted.outcome
            or process.state_generation != persisted.state_generation
        )
        if semantic_changed and not allow_state_transition:
            raise ValidationError(
                "process semantic state must use ProcessTransitionService"
            )
        if allow_state_transition and (
            process.state_generation != persisted.state_generation + 1
        ):
            raise ProcessRevisionConflict(
                f"process state generation conflict for {process.pid}: "
                f"expected {persisted.state_generation + 1}, "
                f"found {process.state_generation}"
            )

    def _process_update_fence(
        self,
        current: Any,
        process: AgentProcess,
        *,
        expected_revision: int,
        cursor: Any,
    ) -> tuple[int, Any, Any, str, tuple[Any, ...]]:
        terminal = {
            ProcessStatus.EXITED.value,
            ProcessStatus.FAILED.value,
            ProcessStatus.KILLED.value,
        }
        current_revision = int(current["revision"])
        current_status = str(current["status"])
        if current_revision != expected_revision:
            raise ProcessRevisionConflict(
                f"process revision conflict for {process.pid}: "
                f"expected {expected_revision}, found {current_revision}"
            )
        terminal_mutation = current_terminal_process_mutation()
        post_exec_mutation = current_post_exec_completion_mutation()
        if current_status in terminal:
            self._assert_terminal_mutation_allowed(
                terminal_mutation,
                process=process,
                current=current,
                expected_revision=expected_revision,
            )
        if current_status in terminal and process.status.value != current_status:
            raise ProcessRevisionConflict(
                f"terminal process {process.pid} cannot transition from "
                f"{current_status} to {process.status.value}"
            )

        generation = int(current["execution_generation"])
        owner_id = current["execution_owner_id"]
        lease_id = current["execution_lease_id"]
        if process.status.value in terminal or (
            current_status == ProcessStatus.RUNNING.value
            and process.status.value != ProcessStatus.RUNNING.value
        ):
            generation += 1
            owner_id = None
            lease_id = None

        execution_token = current_process_execution_token()
        self._assert_execution_target_allowed(
            execution_token,
            target_pid=process.pid,
            current_status=current_status,
        )
        self._assert_process_exec_admission_update_owner(
            cursor,
            current=current,
            process=process,
            execution_token=execution_token,
        )
        post_exec_fence = post_exec_mutation is not None
        if post_exec_fence:
            self._assert_post_exec_completion_allowed(
                post_exec_mutation,
                process=process,
                current=current,
                expected_revision=expected_revision,
            )
        worker_fence = (
            execution_token is not None
            and execution_token.pid == process.pid
            and current_status not in terminal
            and not post_exec_fence
        )
        if worker_fence and (
            current_status != ProcessStatus.RUNNING.value
            or int(current["execution_generation"]) != execution_token.generation
            or current["execution_owner_id"] != execution_token.owner_id
            or current["execution_lease_id"] != execution_token.lease_id
        ):
            raise ProcessRevisionConflict(
                f"stale process execution token cannot mutate {process.pid}"
            )
        if post_exec_fence:
            mutation_clause = (
                " AND status = ? AND execution_generation = ? "
                "AND execution_owner_id IS NULL AND execution_lease_id IS NULL"
            )
            mutation_params = (
                ProcessStatus.RUNNABLE.value,
                post_exec_mutation.expected_generation,
            )
        elif worker_fence:
            mutation_clause = (
                " AND status = ? AND execution_generation = ? "
                "AND execution_owner_id = ? AND execution_lease_id = ?"
            )
            mutation_params: tuple[Any, ...] = (
                ProcessStatus.RUNNING.value,
                execution_token.generation,
                execution_token.owner_id,
                execution_token.lease_id,
            )
        elif current_status in terminal:
            mutation_clause = (
                " AND status = ? AND execution_generation = ? "
                "AND execution_owner_id IS NULL AND execution_lease_id IS NULL"
            )
            mutation_params = (current_status, int(current["execution_generation"]))
        else:
            mutation_clause = ""
            mutation_params = ()
        return generation, owner_id, lease_id, mutation_clause, mutation_params

    def _assert_process_exec_admission_update_owner(
        self,
        cur: Any,
        *,
        current: Any,
        process: AgentProcess,
        execution_token: ProcessExecutionToken | None,
    ) -> None:
        """Fence an admitted exec while preserving explicit emergency control."""

        if str(current["status"]) != ProcessStatus.RUNNING.value:
            return
        pid = process.pid
        generation = int(current["execution_generation"])
        owner_id = current["execution_owner_id"]
        lease_id = current["execution_lease_id"]
        if owner_id is None or lease_id is None:
            return
        current_token = ProcessExecutionToken(
            pid=pid,
            generation=generation,
            owner_id=str(owner_id),
            lease_id=str(lease_id),
        )
        takeover_intent = current_process_execution_takeover_intent()
        if takeover_intent is None and execution_token == current_token:
            return
        takeover_write = self._process_exec_takeover_write_allowed(
            current,
            process,
            current_token,
        )
        if takeover_intent is not None and not takeover_write:
            raise ProcessRevisionConflict(
                f"process execution takeover write is outside its exact intent: {pid}"
            )
        publications = list(
            cur.execute(
                """
                SELECT publication_id, plan_json
                  FROM runtime_publications
                 WHERE kind = 'process_exec' AND pid = ?
                   AND state IN ('planning', 'applying', 'rollback_pending',
                                 'failed', 'manual')
                 ORDER BY created_at, publication_id
                """,
                (pid,),
            )
        )
        for publication in publications:
            try:
                plan = loads(publication["plan_json"], {})
            except (TypeError, ValueError):
                raise ProcessRevisionConflict(
                    "active process exec publication plan is invalid: "
                    f"{publication['publication_id']}"
                ) from None
            if not isinstance(plan, Mapping):
                raise ProcessRevisionConflict(
                    "active process exec publication plan is invalid: "
                    f"{publication['publication_id']}"
                )
            publication_token = (
                plan.get("admission_execution_generation"),
                plan.get("admission_execution_owner_id"),
                plan.get("admission_execution_lease_id"),
            )
            if publication_token == (generation, owner_id, lease_id):
                if takeover_write:
                    return
                raise ProcessRevisionConflict(
                    "process exec admission rejects a non-owner process write: "
                    f"{pid} ({publication['publication_id']})"
                )

    def _process_exec_takeover_write_allowed(
        self,
        current: Any,
        process: AgentProcess,
        current_token: ProcessExecutionToken,
    ) -> bool:
        intent = current_process_execution_takeover_intent()
        if (
            intent is None
            or intent.target_pid != process.pid
            or intent.source_execution_token != current_token
            or intent.source_state_generation != int(current["state_generation"])
        ):
            return False
        if process.status == ProcessStatus.RUNNING:
            if intent.stage == 0:
                return self._record_takeover_reason_capability(
                    current,
                    process,
                    intent,
                )
            if intent.stage == 1:
                return self._record_takeover_reason_root(current, process, intent)
            return False
        return self._validate_takeover_state_transition(current, process, intent)

    def _record_takeover_reason_capability(
        self,
        current: Any,
        process: AgentProcess,
        intent: Any,
    ) -> bool:
        current_process = self._row_to_process(current)
        if (
            intent.reason_text is None
            or intent.reason_action is None
            or int(current["revision"]) != intent.source_revision
            or process.capabilities[:-1] != current_process.capabilities
            or len(process.capabilities) != len(current_process.capabilities) + 1
        ):
            return False
        expected = deepcopy(current_process)
        expected.capabilities = list(process.capabilities)
        expected.updated_at = process.updated_at
        if expected != process:
            return False
        capability = self.get_capability(process.capabilities[-1])
        oid = (
            capability.resource.removeprefix("object:")
            if capability is not None
            else ""
        )
        obj = self.get_object(oid) if oid else None
        valid = self._takeover_reason_artifacts_match(
            capability,
            obj,
            process=process,
            oid=oid,
            intent=intent,
        )
        if valid:
            assert capability is not None
            intent.reason_capability_id = capability.cap_id
            intent.reason_oid = oid
            intent.stage = 1
        return valid

    @staticmethod
    def _takeover_reason_artifacts_match(
        capability: Capability | None,
        obj: AgentObject | None,
        *,
        process: AgentProcess,
        oid: str,
        intent: Any,
    ) -> bool:
        return bool(
            capability is not None
            and capability.active
            and capability.subject == process.pid
            and capability.issued_by == "memory"
            and capability.metadata.get("object_handle") is True
            and capability.resource == f"object:{oid}"
            and obj is not None
            and obj.type == ObjectType.MESSAGE
            and obj.created_by == process.pid
            and obj.owner_kind == ObjectOwnerKind.PROCESS
            and obj.owner_id == process.pid
            and obj.payload == {"reason": intent.reason_text}
            and obj.provenance.created_from_action == intent.reason_action
        )

    def _record_takeover_reason_root(
        self,
        current: Any,
        process: AgentProcess,
        intent: Any,
    ) -> bool:
        if int(current["revision"]) != intent.source_revision + 1:
            return False
        current_process = self._row_to_process(current)
        expected = deepcopy(current_process)
        expected.memory_view = deepcopy(process.memory_view)
        expected.updated_at = process.updated_at
        if expected != process or process.memory_view is None:
            return False
        old_roots = (
            current_process.memory_view.roots
            if current_process.memory_view is not None
            else []
        )
        if (
            process.memory_view.roots[:-1] != old_roots
            or len(process.memory_view.roots) != len(old_roots) + 1
        ):
            return False
        root = process.memory_view.roots[-1]
        if current_process.memory_view is None:
            view_valid = bool(
                process.memory_view.view_id
                and process.memory_view.owner_pid == process.pid
                and process.memory_view.filters == []
                and process.memory_view.rights_policy == "attenuate"
                and process.memory_view.created_from is None
                and process.memory_view.mode == ViewMode.READ_ONLY
            )
        else:
            expected_view = deepcopy(current_process.memory_view)
            expected_view.roots.append(root)
            view_valid = process.memory_view == expected_view
        capability = self.get_capability(str(intent.reason_capability_id or ""))
        valid = bool(
            view_valid
            and capability is not None
            and root.oid == intent.reason_oid
            and root.capability_id == capability.cap_id
            and root.rights == capability.rights
            and root.expires_at == capability.expires_at
        )
        if valid:
            intent.stage = 2
        return valid

    def _validate_takeover_state_transition(
        self,
        current: Any,
        process: AgentProcess,
        intent: Any,
    ) -> bool:
        reason_expected = intent.reason_text is not None
        prep_count = 2 if reason_expected else 0
        if (
            int(current["revision"]) != intent.source_revision + prep_count
            or intent.stage != prep_count
            or process.status.value != intent.intended_status
            or process.state_generation != intent.source_state_generation + 1
        ):
            return False
        current_process = self._row_to_process(current)
        expected = deepcopy(current_process)
        expected.status = process.status
        expected.wait_state = process.wait_state
        expected.outcome = process.outcome
        expected.state_generation = process.state_generation
        expected.status_message = process.status_message
        expected.updated_at = process.updated_at
        if expected != process:
            return False
        reason_oid = intent.reason_oid if reason_expected else None
        if process.status == ProcessStatus.PAUSED:
            wait_state = process.wait_state
            valid = bool(
                process.outcome is None
                and isinstance(wait_state, (PausedProcessWait, HostResumeProcessWait))
                and wait_state.kind == intent.wait_kind
                and wait_state.reason_oid == reason_oid
            )
        else:
            outcome = process.outcome
            valid = bool(
                process.wait_state is None
                and isinstance(outcome, KilledProcessOutcome)
                and outcome.reason_oid == reason_oid
                and outcome.code == intent.outcome_code
            )
        if valid:
            intent.stage = 3
        return valid

    def _assert_post_exec_completion_allowed(
        self,
        mutation: Any,
        *,
        process: AgentProcess,
        current: Any,
        expected_revision: int,
    ) -> None:
        current_process = self._row_to_process(current)
        token = current_process_execution_token()
        receipt = self._post_exec_completion_receipt(mutation)
        exact_fence = (
            mutation.target_pid == process.pid
            and mutation.expected_revision == expected_revision
            and mutation.expected_generation == int(current["execution_generation"])
            and mutation.execution_token == token
            and token is not None
            and token.pid == process.pid
            and int(current["execution_generation"]) == token.generation + 1
            and str(current["status"]) == ProcessStatus.RUNNABLE.value
            and current["execution_owner_id"] is None
            and current["execution_lease_id"] is None
            and receipt.get("revision") == expected_revision
            and receipt.get("execution_generation") == mutation.expected_generation
            and receipt.get("prior_execution_generation") == token.generation
            and receipt.get("prior_execution_owner_id") == token.owner_id
            and receipt.get("prior_execution_lease_id") == token.lease_id
        )
        if not exact_fence:
            raise ProcessRevisionConflict(
                f"post-exec completion fence conflict for {process.pid}"
            )
        self._assert_post_exec_tool_result_append(current_process, process)

    def _post_exec_completion_receipt(self, mutation: Any) -> dict[str, Any]:
        publication = self.get_runtime_publication(mutation.publication_id)
        self._assert_post_exec_publication_identity(publication, mutation)
        assert publication is not None
        operation = self.get_operation(mutation.operation_id)
        self._assert_post_exec_operation_identity(operation, publication, mutation)
        phases = list((publication.get("receipt") or {}).get("phases") or [])
        committed = [
            dict(item)
            for item in phases
            if isinstance(item, dict) and item.get("phase") == "committed"
        ]
        if len(committed) != 1:
            raise ProcessRevisionConflict(
                f"post-exec completion receipt conflict for {mutation.target_pid}"
            )
        return committed[0]

    @staticmethod
    def _assert_post_exec_publication_identity(
        publication: dict[str, Any] | None,
        mutation: Any,
    ) -> None:
        if (
            publication is None
            or publication["kind"] != "process_exec"
            or publication["pid"] != mutation.target_pid
            or publication["state"] != "committed"
            or publication["phase"] != "committed"
            or publication["plan"].get("operation_id") != mutation.operation_id
        ):
            raise ProcessRevisionConflict(
                f"post-exec publication fence conflict for {mutation.target_pid}"
            )

    @staticmethod
    def _assert_post_exec_operation_identity(
        operation: Any,
        publication: dict[str, Any],
        mutation: Any,
    ) -> None:
        if (
            operation is None
            or operation.kind.value != "runtime"
            or operation.name != "process.exec"
            or operation.actor != mutation.target_pid
            or operation.pid != mutation.target_pid
            or operation.metadata.get("runtime_publication_id")
            != mutation.publication_id
            or operation.metadata.get("runtime_publication_kind") != "process_exec"
            or operation.metadata.get("runtime_publication_bound") is not True
            or operation.metadata.get("runtime_publication_binding_version") != 1
            or publication["plan"].get("operation_binding_version") != 1
        ):
            raise ProcessRevisionConflict(
                f"post-exec operation fence conflict for {mutation.target_pid}"
            )

    def _assert_post_exec_tool_result_append(
        self,
        current: AgentProcess,
        proposed: AgentProcess,
    ) -> None:
        immutable_fields = (
            "pid", "parent_pid", "image_id", "status", "goal_oid", "memory_view",
            "loaded_skills", "tool_table", "model_tool_table", "event_cursor",
            "checkpoint_head", "status_message", "resource_budget", "resource_usage",
            "wait_state", "outcome", "state_generation", "working_directory",
            "llm_profile_id", "created_at", "revision",
            "execution_generation", "execution_owner_id", "execution_lease_id",
        )
        if any(getattr(current, name) != getattr(proposed, name) for name in immutable_fields):
            raise ProcessRevisionConflict(
                f"post-exec completion may only append a ToolResult handle: {proposed.pid}"
            )
        if (
            proposed.capabilities[:-1] != current.capabilities
            or len(proposed.capabilities) != len(current.capabilities) + 1
        ):
            raise ProcessRevisionConflict(
                f"post-exec completion capability append conflict for {proposed.pid}"
            )
        capability = self.get_capability(proposed.capabilities[-1])
        oid = capability.resource.removeprefix("object:") if capability is not None else ""
        obj = self.get_object(oid) if oid else None
        if (
            capability is None
            or capability.subject != proposed.pid
            or not capability.active
            or not capability.metadata.get("object_handle")
            or capability.issued_by != "memory"
            or not capability.resource.startswith("object:")
            or obj is None
            or obj.type != ObjectType.TOOL_RESULT
            or obj.created_by != proposed.pid
            or obj.owner_kind != ObjectOwnerKind.PROCESS
            or obj.owner_id != proposed.pid
            or obj.provenance.created_from_action != "tool.exec_process"
        ):
            raise ProcessRevisionConflict(
                f"post-exec completion requires its exact ToolResult handle: {proposed.pid}"
            )

    @staticmethod
    def _assert_terminal_mutation_allowed(
        terminal_mutation: Any,
        *,
        process: AgentProcess,
        current: Any,
        expected_revision: int,
    ) -> None:
        current_status = str(current["status"])
        current_generation = int(current["execution_generation"])
        execution_token = current_process_execution_token()
        if terminal_mutation is None:
            raise ProcessRevisionConflict(
                f"terminal process {process.pid} is immutable: {current_status}"
            )
        if (
            terminal_mutation.target_pid != process.pid
            or terminal_mutation.expected_revision != expected_revision
            or terminal_mutation.expected_generation != current_generation
            or current_status not in terminal_mutation.allowed_statuses
            or terminal_mutation.execution_token != execution_token
            or current["execution_owner_id"] is not None
            or current["execution_lease_id"] is not None
        ):
            raise ProcessRevisionConflict(
                f"terminal process mutation fence conflict for {process.pid}"
            )
        if (
            execution_token is not None
            and execution_token.pid == process.pid
            and current_generation != execution_token.generation + 1
        ):
            raise ProcessRevisionConflict(
                f"stale process execution token cannot mutate terminal process {process.pid}"
            )

    @staticmethod
    def _assert_execution_target_allowed(
        execution_token: ProcessExecutionToken | None,
        *,
        target_pid: str,
        current_status: str,
    ) -> None:
        takeover = current_process_execution_takeover_intent()
        if takeover is not None and takeover.target_pid != target_pid:
            raise ProcessRevisionConflict(
                "process execution takeover cannot mutate another process: "
                f"{target_pid}"
            )
        if execution_token is None or execution_token.pid == target_pid:
            return
        control_mutation = current_process_control_mutation()
        if (
            control_mutation is not None
            and control_mutation.target_pid == target_pid
            and current_status in control_mutation.allowed_statuses
        ):
            return
        raise ProcessRevisionConflict(
            "process execution token for "
            f"{execution_token.pid} cannot mutate {target_pid}"
        )

    def patch_process_control(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        allowed_statuses: Iterable[ProcessStatus | str],
        reason: str,
    ) -> AgentProcess:
        """Apply an explicit Host/control-plane process mutation.

        This is the only compatibility path by which code executing under a
        worker token may intentionally mutate a different PID.  The ordinary
        patch API never silently downgrades a mismatched worker into a Host
        write.
        """

        selected_statuses = frozenset(ProcessStatus(status) for status in allowed_statuses)
        if not selected_statuses:
            raise ValidationError("process control allowed statuses must be non-empty")
        current = self.get_process(pid)
        if current is None:
            raise ProcessRevisionConflict(f"process no longer exists: {pid}")
        if current.revision != int(expected_revision):
            raise ProcessRevisionConflict(
                f"process revision conflict for {pid}: "
                f"expected {expected_revision}, found {current.revision}"
            )
        if current.status not in selected_statuses:
            allowed = ", ".join(sorted(status.value for status in selected_statuses))
            raise ProcessRevisionConflict(
                f"process control status conflict for {pid}: "
                f"expected one of {allowed}, found {current.status.value}"
            )
        with trusted_process_control_mutation(
            pid,
            allowed_statuses=selected_statuses,
            reason=reason,
        ):
            return self.patch_process(
                pid,
                patch,
                expected_revision=expected_revision,
                expected_status=current.status,
            )

    def patch_process(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
    ) -> AgentProcess:
        """Apply a field-level process CAS and return the committed row."""

        allowed = {
            "parent_pid",
            "image_id",
            "goal_oid",
            "memory_view",
            "capabilities",
            "loaded_skills",
            "tool_table",
            "model_tool_table",
            "event_cursor",
            "checkpoint_head",
            "status_message",
            "resource_budget",
            "resource_usage",
            "working_directory",
            "llm_profile_id",
            "updated_at",
        }
        unknown = set(patch) - allowed
        if unknown:
            semantic = unknown & _PROCESS_SEMANTIC_FIELDS
            if semantic:
                raise ValidationError(
                    "process semantic state must use ProcessTransitionService: "
                    f"{', '.join(sorted(semantic))}"
                )
            raise ValidationError(f"unsupported process patch fields: {', '.join(sorted(unknown))}")
        return self._patch_process(
            pid,
            patch,
            expected_revision=expected_revision,
            expected_status=expected_status,
            allow_state_transition=False,
        )

    def _patch_process(
        self,
        pid: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None,
        allow_state_transition: bool,
    ) -> AgentProcess:
        with self._lock:
            current = self.get_process(pid)
            if current is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            if current.revision != int(expected_revision):
                raise ProcessRevisionConflict(
                    f"process revision conflict for {pid}: expected {expected_revision}, found {current.revision}"
                )
            if expected_status is not None and current.status != ProcessStatus(expected_status):
                raise ProcessRevisionConflict(
                    f"process status conflict for {pid}: expected {ProcessStatus(expected_status).value}, "
                    f"found {current.status.value}"
                )
            for field_name, value in patch.items():
                if field_name == "status":
                    value = ProcessStatus(value)
                setattr(current, field_name, deepcopy(value))
            if "updated_at" not in patch:
                current.updated_at = utc_now()
            self._update_process(
                current,
                allow_state_transition=allow_state_transition,
            )
            return current

    def apply_process_state_transition(
        self,
        pid: str,
        status: ProcessStatus | str,
        *,
        expected_revision: int,
        expected_status: ProcessStatus | str | None = None,
        expected_state_generation: int | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
        status_message: str | None = None,
        control: bool = False,
        allowed_statuses: Iterable[ProcessStatus | str] | None = None,
        reason: str | None = None,
    ) -> AgentProcess:
        selected_status = ProcessStatus(status)
        validate_process_state_fields(
            selected_status.value,
            wait_state,
            outcome,
        )
        with self._lock:
            current = self.get_process(pid)
            if current is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            if (
                expected_state_generation is not None
                and current.state_generation != int(expected_state_generation)
            ):
                raise ProcessRevisionConflict(
                    f"process state generation conflict for {pid}: "
                    f"expected {expected_state_generation}, "
                    f"found {current.state_generation}"
                )
            patch = {
                "status": selected_status,
                "wait_state": wait_state,
                "outcome": outcome,
                "state_generation": current.state_generation + 1,
                "status_message": legacy_status_message(
                    wait_state,
                    outcome,
                    status_message,
                ),
            }
            if not control:
                return self._patch_process(
                    pid,
                    patch,
                    expected_revision=expected_revision,
                    expected_status=expected_status,
                    allow_state_transition=True,
                )
            selected_allowed = frozenset(
                ProcessStatus(item) for item in (allowed_statuses or ())
            )
            if not selected_allowed:
                raise ValidationError(
                    "control process transition requires allowed statuses"
                )
            if current.status not in selected_allowed:
                allowed = ", ".join(
                    sorted(item.value for item in selected_allowed)
                )
                raise ProcessRevisionConflict(
                    f"process control status conflict for {pid}: "
                    f"expected one of {allowed}, found {current.status.value}"
                )
            with trusted_process_control_mutation(
                pid,
                allowed_statuses=selected_allowed,
                reason=reason or "semantic process state transition",
            ):
                return self._patch_process(
                    pid,
                    patch,
                    expected_revision=expected_revision,
                    expected_status=current.status,
                    allow_state_transition=True,
                )

    def append_process_memory_roots(
        self,
        pid: str,
        roots: Iterable[ObjectHandle],
    ) -> AgentProcess:
        """Commutatively append roots while holding the store serialization lock."""

        additions = [deepcopy(root) for root in roots]
        with self._lock:
            process = self.get_process(pid)
            if process is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            if process.memory_view is None:
                raise ValidationError(f"process has no memory view: {pid}")
            known = {root.oid for root in process.memory_view.roots}
            process.memory_view.roots.extend(root for root in additions if root.oid not in known)
            process.updated_at = utc_now()
            self.update_process(process)
            return process

    def remove_process_memory_roots(
        self,
        pid: str,
        oids: Iterable[str],
    ) -> AgentProcess:
        """Commutatively remove selected roots while preserving concurrent additions."""

        removals = {str(oid) for oid in oids}
        with self._lock:
            process = self.get_process(pid)
            if process is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            if process.memory_view is None:
                return process
            retained = [root for root in process.memory_view.roots if root.oid not in removals]
            if len(retained) == len(process.memory_view.roots):
                return process
            process.memory_view.roots = retained
            process.updated_at = utc_now()
            self.update_process(process)
            return process

    def append_process_capability_ids(
        self,
        pid: str,
        capability_ids: Iterable[str],
    ) -> AgentProcess:
        """Commutatively attach capability identifiers to a process row."""

        additions = [str(cap_id) for cap_id in capability_ids]
        with self._lock:
            process = self.get_process(pid)
            if process is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            known = set(process.capabilities)
            selected = [cap_id for cap_id in additions if cap_id not in known]
            if not selected:
                return process
            process.capabilities.extend(selected)
            process.updated_at = utc_now()
            self.update_process(process)
            return process

    def patch_process_tool_tables(
        self,
        pid: str,
        *,
        tool_table: Mapping[str, str] | None = None,
        model_tool_table: Mapping[str, str] | None = None,
    ) -> AgentProcess:
        """Commutatively merge tool mappings under the store lock."""

        with self._lock:
            process = self.get_process(pid)
            if process is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            if tool_table:
                process.tool_table.update({str(key): str(value) for key, value in tool_table.items()})
            if model_tool_table:
                process.model_tool_table.update(
                    {str(key): str(value) for key, value in model_tool_table.items()}
                )
            process.updated_at = utc_now()
            self.update_process(process)
            return process

    def remove_process_tool_bindings(
        self,
        pid: str,
        bindings: Mapping[str, str],
    ) -> AgentProcess:
        """Remove only bindings that still point at the supplied tool IDs."""

        expected = {str(name): str(tool_id) for name, tool_id in bindings.items()}
        with self._lock:
            process = self.get_process(pid)
            if process is None:
                raise ProcessRevisionConflict(f"process no longer exists: {pid}")
            changed = False
            for name, tool_id in expected.items():
                if process.tool_table.get(name) == tool_id:
                    process.tool_table.pop(name, None)
                    changed = True
                if process.model_tool_table.get(name) == tool_id:
                    process.model_tool_table.pop(name, None)
                    changed = True
            if not changed:
                return process
            process.updated_at = utc_now()
            self.update_process(process)
            return process

    def replace_process_for_restore(self, process: AgentProcess) -> None:
        """Controlled snapshot-only replacement, deliberately outside normal CAS."""

        with self.transaction() as cur:
            existing = self.get_process(process.pid)
            (
                process.revision,
                process.execution_generation,
                process.state_generation,
            ) = self.reserve_process_restore_epoch(
                process.pid,
                revision_floor=max(
                    int(process.revision),
                    int(existing.revision) if existing is not None else 0,
                ),
                execution_generation_floor=max(
                    int(process.execution_generation),
                    int(existing.execution_generation) if existing is not None else 0,
                ),
                state_generation_floor=max(
                    int(process.state_generation),
                    int(existing.state_generation) if existing is not None else 0,
                ),
                cursor=cur,
            )
            process.execution_owner_id = None
            process.execution_lease_id = None
            if existing is not None:
                cur.execute(
                    """
                    DELETE FROM processes WHERE pid = ?
                    """,
                    (process.pid,),
                )
            self.insert_process(process)

    def restore_process_for_exec(
        self,
        before_row: Mapping[str, Any],
        *,
        expected_revision: int,
        publication_id: str | None = None,
        capability_ids: Iterable[str] | None = None,
        fence_execution: bool = True,
    ) -> bool:
        """Restore only the exact process-exec admission owned by a publication."""

        pid = str(before_row["pid"])
        restored, columns = self._prepare_process_exec_restore(
            before_row,
            capability_ids=capability_ids,
            fence_execution=fence_execution,
        )
        assignments = ", ".join(f"{column} = ?" for column in columns)
        execution_assignment = (
            "execution_generation = execution_generation + 1, "
            "execution_owner_id = NULL, execution_lease_id = NULL"
            if fence_execution
            else "execution_generation = ?, execution_owner_id = ?, execution_lease_id = ?"
        )
        execution_params: tuple[Any, ...] = ()
        if not fence_execution:
            execution_params = (
                int(restored.get("execution_generation") or 0),
                restored.get("execution_owner_id"),
                restored.get("execution_lease_id"),
            )
        with self.transaction() as cur:
            process_fence_clause, process_fence_params = (
                self._process_exec_restore_fence(
                    cur,
                    publication_id=str(publication_id or ""),
                    pid=pid,
                    before_row=before_row,
                    fence_execution=fence_execution,
                )
            )
            updated = cur.execute(
                f"""
                UPDATE processes
                   SET {assignments}, revision = revision + 1,
                       state_generation = state_generation + 1,
                       {execution_assignment}
                 WHERE pid = ? AND revision = ?
                   {process_fence_clause}
                """,
                (
                    *(restored[column] for column in columns),
                    *execution_params,
                    pid,
                    int(expected_revision),
                    *process_fence_params,
                ),
            )
            if updated.rowcount != 1:
                return False
            self._replace_process_tool_bindings(
                cur,
                pid,
                loads(restored["tool_table_json"], {}),
                loads(restored["model_tool_table_json"], {}),
            )
            self._observe_restored_process_concurrency(cur, pid)
            return True

    @staticmethod
    def _prepare_process_exec_restore(
        before_row: Mapping[str, Any],
        *,
        capability_ids: Iterable[str] | None,
        fence_execution: bool,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        restored = dict(before_row)
        if fence_execution and restored.get("status") == ProcessStatus.RUNNING.value:
            restored.update(
                status=ProcessStatus.RUNNABLE.value,
                status_message=None,
                wait_state_json="null",
                outcome_json="null",
            )
        if capability_ids is not None:
            restored["capabilities_json"] = dumps(list(dict.fromkeys(capability_ids)))
        return restored, (
            "image_id", "status", "goal_oid", "memory_view_json",
            "capabilities_json", "loaded_skills_json", "tool_table_json",
            "model_tool_table_json", "status_message", "wait_state_json",
            "outcome_json", "working_directory", "llm_profile_id", "updated_at",
        )

    def _process_exec_restore_fence(
        self,
        cur: Any,
        *,
        publication_id: str,
        pid: str,
        before_row: Mapping[str, Any],
        fence_execution: bool,
    ) -> tuple[str, tuple[Any, ...]]:
        if not fence_execution:
            return " AND status NOT IN (?, ?, ?)", (
                ProcessStatus.EXITED.value,
                ProcessStatus.FAILED.value,
                ProcessStatus.KILLED.value,
            )
        token, state_generation = self._process_exec_rollback_admission_fence(
            cur,
            publication_id=publication_id,
            pid=pid,
            before_row=before_row,
        )
        return (
            " AND status = ? AND state_generation = ? "
            "AND execution_generation = ? "
            "AND execution_owner_id = ? AND execution_lease_id = ?",
            (
                ProcessStatus.RUNNING.value,
                state_generation,
                token.generation,
                token.owner_id,
                token.lease_id,
            ),
        )

    def _observe_restored_process_concurrency(self, cur: Any, pid: str) -> None:
        rows = list(
            cur.execute(
                """
                SELECT revision, execution_generation, state_generation
                  FROM processes WHERE pid = ?
                """,
                (pid,),
            )
        )
        self.observe_process_concurrency(
            pid,
            revision=int(rows[0]["revision"]),
            execution_generation=int(rows[0]["execution_generation"]),
            state_generation=int(rows[0]["state_generation"]),
            cursor=cur,
        )

    @staticmethod
    def _process_exec_rollback_admission_fence(
        cur: Any,
        *,
        publication_id: str,
        pid: str,
        before_row: Mapping[str, Any],
    ) -> tuple[ProcessExecutionToken, int]:
        """Authenticate one rollback against its immutable admission plan."""

        plan = SQLRuntimeStore._process_exec_rollback_plan(
            cur,
            publication_id=publication_id,
            pid=pid,
        )
        SQLRuntimeStore._assert_process_exec_rollback_snapshot(
            plan,
            publication_id=publication_id,
            pid=pid,
            before_row=before_row,
        )
        return SQLRuntimeStore._process_exec_rollback_token(
            plan,
            publication_id=publication_id,
            pid=pid,
            before_row=before_row,
        )

    @staticmethod
    def _process_exec_rollback_plan(
        cur: Any,
        *,
        publication_id: str,
        pid: str,
    ) -> Mapping[str, Any]:
        if not publication_id:
            raise ProcessRevisionConflict(
                f"process exec rollback requires its durable publication: {pid}"
            )
        rows = list(
            cur.execute(
                """
                SELECT kind, pid, state, phase, plan_json
                  FROM runtime_publications
                 WHERE publication_id = ?
                """,
                (publication_id,),
            )
        )
        if len(rows) != 1:
            raise ProcessRevisionConflict(
                f"process exec rollback publication is missing: {publication_id}"
            )
        publication = rows[0]
        if (
            publication["kind"] != "process_exec"
            or publication["pid"] != pid
            or publication["state"] != "rollback_pending"
            or publication["phase"] not in {"compensating", "recovery_claimed"}
        ):
            raise ProcessRevisionConflict(
                "process exec rollback publication identity or phase conflict: "
                f"{publication_id}"
            )
        try:
            plan = loads(publication["plan_json"], {})
        except (TypeError, ValueError):
            raise ProcessRevisionConflict(
                f"process exec rollback publication plan is invalid: {publication_id}"
            ) from None
        if not isinstance(plan, Mapping):
            raise ProcessRevisionConflict(
                f"process exec rollback publication plan is invalid: {publication_id}"
            )
        return plan

    @staticmethod
    def _assert_process_exec_rollback_snapshot(
        plan: Mapping[str, Any],
        *,
        publication_id: str,
        pid: str,
        before_row: Mapping[str, Any],
    ) -> None:
        durable_snapshot = (
            plan.get("before_snapshot")
        )
        durable_rows = (
            durable_snapshot.get("rows")
            if isinstance(durable_snapshot, Mapping)
            else None
        )
        durable_processes = (
            durable_rows.get("processes")
            if isinstance(durable_rows, Mapping)
            else None
        )
        if (
            plan.get("pid") != pid
            or not isinstance(durable_processes, list)
            or len(durable_processes) != 1
            or not isinstance(durable_processes[0], Mapping)
            or dict(durable_processes[0]) != dict(before_row)
        ):
            raise ProcessRevisionConflict(
                f"process exec rollback snapshot is invalid: {publication_id}"
            )

    @staticmethod
    def _process_exec_rollback_token(
        plan: Mapping[str, Any],
        *,
        publication_id: str,
        pid: str,
        before_row: Mapping[str, Any],
    ) -> tuple[ProcessExecutionToken, int]:
        generation = plan.get("admission_execution_generation")
        owner_id = plan.get("admission_execution_owner_id")
        lease_id = plan.get("admission_execution_lease_id")
        before_state_generation = before_row.get("state_generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or not isinstance(owner_id, str)
            or not owner_id
            or not isinstance(lease_id, str)
            or not lease_id
            or isinstance(before_state_generation, bool)
            or not isinstance(before_state_generation, int)
            or before_state_generation < 0
        ):
            raise ProcessRevisionConflict(
                f"process exec rollback admission plan is invalid: {publication_id}"
            )
        token = ProcessExecutionToken(
            pid=pid,
            generation=generation,
            owner_id=owner_id,
            lease_id=lease_id,
        )
        ambient = current_process_execution_token()
        if ambient is not None and ambient != token:
            raise ProcessRevisionConflict(
                "process exec rollback ambient admission token conflict: "
                f"{publication_id}"
            )
        return token, before_state_generation + 1

    def commit_process_exec_epoch(
        self,
        pid: str,
        *,
        publication_id: str,
        expected_revision: int,
    ) -> AgentProcess:
        """Commit a successful exec admission epoch.

        The helper is designed to run inside the exec publication commit
        transaction under the exact admission execution token recorded by that
        publication.  Admission already rotated the execution generation and
        invalidated the previous image's worker token.  Commit clears the
        internal lease and hands the RUNNING process back to the runnable queue
        without consuming a second execution generation.
        """

        execution_token = current_process_execution_token()
        if execution_token is None:
            raise ProcessRevisionConflict(
                f"process exec epoch commit requires its exact admission token: {pid}"
            )
        now = utc_now()
        with self.transaction() as cur:
            self._assert_process_exec_commit_publication(
                cur,
                publication_id=publication_id,
                pid=pid,
                execution_token=execution_token,
            )
            (
                current_revision,
                worker_clause,
                worker_params,
            ) = self._process_exec_epoch_fence(
                cur,
                pid=pid,
                expected_revision=expected_revision,
                execution_token=execution_token,
            )
            updated = cur.execute(
                f"""
                UPDATE processes
                   SET status = ?, updated_at = ?, revision = revision + 1,
                       status_message = NULL, wait_state_json = 'null',
                       outcome_json = 'null', state_generation = state_generation + 1,
                       execution_owner_id = NULL, execution_lease_id = NULL
                 WHERE pid = ? AND revision = ?{worker_clause}
                """,
                (
                    ProcessStatus.RUNNABLE.value,
                    now,
                    pid,
                    current_revision,
                    *worker_params,
                ),
            )
            if updated.rowcount != 1:
                raise ProcessRevisionConflict(
                    f"process exec epoch conflict for {pid}"
                )
            counters = list(
                cur.execute(
                """
                SELECT revision, execution_generation, state_generation
                  FROM processes WHERE pid = ?
                """,
                    (pid,),
                )
            )
            self.observe_process_concurrency(
                pid,
                revision=int(counters[0]["revision"]),
                execution_generation=int(counters[0]["execution_generation"]),
                state_generation=int(counters[0]["state_generation"]),
                cursor=cur,
            )
        committed = self.get_process(pid)
        if committed is None:
            raise ProcessRevisionConflict(f"process no longer exists: {pid}")
        return committed

    @staticmethod
    def _assert_process_exec_commit_publication(
        cur: Any,
        *,
        publication_id: str,
        pid: str,
        execution_token: ProcessExecutionToken,
    ) -> None:
        rows = list(
            cur.execute(
                """
                SELECT kind, pid, state, phase, plan_json
                  FROM runtime_publications
                 WHERE publication_id = ?
                """,
                (publication_id,),
            )
        )
        if len(rows) != 1:
            raise ProcessRevisionConflict(
                f"process exec commit publication is missing: {publication_id}"
            )
        publication = rows[0]
        if (
            publication["kind"] != "process_exec"
            or publication["pid"] != pid
            or publication["state"] != "applying"
            or publication["phase"] != "skills_configured"
        ):
            raise ProcessRevisionConflict(
                "process exec commit publication identity or phase conflict: "
                f"{publication_id}"
            )
        try:
            plan = loads(publication["plan_json"], {})
        except (TypeError, ValueError):
            raise ProcessRevisionConflict(
                f"process exec commit publication plan is invalid: {publication_id}"
            ) from None
        if not isinstance(plan, Mapping):
            raise ProcessRevisionConflict(
                f"process exec commit publication plan is invalid: {publication_id}"
            )
        generation = plan.get("admission_execution_generation")
        owner_id = plan.get("admission_execution_owner_id")
        lease_id = plan.get("admission_execution_lease_id")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or not isinstance(owner_id, str)
            or not owner_id
            or not isinstance(lease_id, str)
            or not lease_id
        ):
            raise ProcessRevisionConflict(
                f"process exec commit publication plan is invalid: {publication_id}"
            )
        if (
            execution_token.pid != pid
            or generation != execution_token.generation
            or owner_id != execution_token.owner_id
            or lease_id != execution_token.lease_id
        ):
            raise ProcessRevisionConflict(
                "process exec commit publication admission token conflict: "
                f"{publication_id}"
            )

    @staticmethod
    def _process_exec_epoch_fence(
        cur: Any,
        *,
        pid: str,
        expected_revision: int,
        execution_token: ProcessExecutionToken,
    ) -> tuple[int, str, tuple[Any, ...]]:
        if execution_token.pid != pid:
            raise ProcessRevisionConflict(
                f"process execution token for {execution_token.pid} cannot exec {pid}"
            )
        rows = list(
            cur.execute(
                """
                SELECT status, revision, execution_generation,
                       execution_owner_id, execution_lease_id
                  FROM processes WHERE pid = ?
                """,
                (pid,),
            )
        )
        if not rows:
            raise ProcessRevisionConflict(f"process no longer exists: {pid}")
        row = rows[0]
        current_status = ProcessStatus(str(row["status"]))
        current_revision = int(row["revision"])
        if current_revision != int(expected_revision):
            raise ProcessRevisionConflict(
                f"process revision conflict for {pid}: "
                f"expected {expected_revision}, found {current_revision}"
            )
        if (
            current_status != ProcessStatus.RUNNING
            or int(row["execution_generation"]) != execution_token.generation
            or row["execution_owner_id"] != execution_token.owner_id
            or row["execution_lease_id"] != execution_token.lease_id
        ):
            raise ProcessRevisionConflict(
                f"stale process execution token cannot exec {pid}"
            )
        return (
            current_revision,
            " AND status = ? AND execution_generation = ? "
            "AND execution_owner_id = ? AND execution_lease_id = ?",
            (
                ProcessStatus.RUNNING.value,
                execution_token.generation,
                execution_token.owner_id,
                execution_token.lease_id,
            ),
        )

    def get_process(self, pid: str) -> AgentProcess | None:
        rows = self._query("SELECT * FROM processes WHERE pid = ?", (pid,))
        return self._row_to_process(rows[0]) if rows else None

    def insert_runtime_publication(
        self,
        *,
        publication_id: str,
        kind: RuntimePublicationKind | str,
        pid: str,
        owner_instance_id: str,
        plan: Mapping[str, Any],
        phase: str = "planned",
        _checkpoint_restore_writer_token: object | None = None,
    ) -> dict[str, Any]:
        try:
            selected_kind = parse_runtime_publication_kind(kind)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        for field_name, value in (
            ("publication_id", publication_id),
            ("pid", pid),
            ("owner_instance_id", owner_instance_id),
            ("phase", phase),
        ):
            if type(value) is not str or not value:
                raise ValidationError(
                    f"runtime publication {field_name} must be non-empty text"
                )
        if not isinstance(plan, Mapping):
            raise ValidationError("runtime publication plan must be a mapping")
        if (
            selected_kind is RuntimePublicationKind.CHECKPOINT_RESTORE
            and _checkpoint_restore_writer_token
            is not self.__checkpoint_restore_writer_token
        ):
            raise ValidationError(
                "checkpoint restore publications require the internal writer"
            )
        now = utc_now()
        self._execute(
            """
            INSERT INTO runtime_publications (
                publication_id, kind, pid, owner_instance_id, state, phase,
                plan_json, receipt_json, error_json, operation_reconciled,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                publication_id,
                selected_kind.value,
                pid,
                owner_instance_id,
                "planning",
                phase,
                dumps(dict(plan)),
                dumps({"phases": [], "artifacts": []}),
                None,
                0,
                now,
                now,
            ),
        )
        publication = self.get_runtime_publication(publication_id)
        assert publication is not None
        return publication

    def get_runtime_publication(self, publication_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM runtime_publications WHERE publication_id = ?",
            (publication_id,),
        )
        return self._runtime_publication_row(rows[0]) if rows else None

    def list_runtime_publications(
        self,
        *,
        states: Iterable[str] | None = None,
        pid: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        try:
            selected_states = [
                parse_runtime_publication_state(state) for state in states or []
            ]
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if selected_states:
            placeholders = ", ".join("?" for _ in selected_states)
            clauses.append(f"state IN ({placeholders})")
            params.extend(selected_states)
        if pid is not None:
            clauses.append("pid = ?")
            params.append(pid)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM runtime_publications{where} ORDER BY created_at, publication_id",
            params,
        )
        return [self._runtime_publication_row(row) for row in rows]

    def query_runtime_publication_operation_reconciliation(
        self,
        *,
        kind: str,
        state: str,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        """Return one indexed page of terminal publications needing repair."""

        try:
            selected_kind = parse_runtime_publication_kind(kind).value
            selected_state = parse_runtime_publication_state(state)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        terminal_states = {"committed", "rolled_back", "failed", "manual"}
        if selected_state not in terminal_states:
            raise ValidationError(
                f"runtime publication operation reconciliation requires a terminal state: {state}"
            )
        return self._query_runtime_publication_page(
            kind=selected_kind,
            state=selected_state,
            operation_reconciled=False,
            after=after,
            limit=limit,
            purpose="operation-reconciliation",
        )

    def query_runtime_publication_recovery(
        self,
        *,
        kind: str,
        state: str,
        operation_reconciled: bool,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        """Return one indexed page of an exact publication recovery state."""

        try:
            selected_kind = parse_runtime_publication_kind(kind).value
            selected_state = parse_runtime_publication_state(state)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        recoverable_states = {
            "planning",
            "applying",
            "reconciliation_pending",
            "rollback_pending",
            "failed",
            "manual",
        }
        if selected_state not in recoverable_states:
            raise ValidationError(
                f"runtime publication recovery requires an unresolved state: {state}"
            )
        return self._query_runtime_publication_page(
            kind=selected_kind,
            state=selected_state,
            operation_reconciled=operation_reconciled,
            after=after,
            limit=limit,
            purpose="recovery",
        )

    def _query_runtime_publication_page(
        self,
        *,
        kind: str,
        state: str,
        operation_reconciled: bool,
        after: RuntimePublicationCursor | None,
        limit: int,
        purpose: str,
    ) -> RuntimePublicationPage:
        selected_limit = self._runtime_publication_reconciliation_limit(limit)
        clauses = ["kind = ?", "state = ?"]
        params: list[Any] = [kind, state]
        if purpose == "operation-reconciliation":
            clauses.append("operation_reconciled = 0")
        else:
            clauses.append("operation_reconciled = ?")
            params.append(int(operation_reconciled))
        if after is not None:
            clauses.append("(created_at, publication_id) > (?, ?)")
            params.extend((after.created_at, after.publication_id))
        params.append(selected_limit + 1)
        rows = self._query(
            "SELECT * FROM runtime_publications "
            f"/* {purpose} */ "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at, publication_id LIMIT ?",
            params,
        )
        records = tuple(
            self._runtime_publication_row(row) for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = RuntimePublicationCursor(
                last["created_at"],
                last["publication_id"],
            )
        return RuntimePublicationPage(records=records, next_cursor=next_cursor)

    def mark_runtime_publication_operation_reconciled(
        self,
        publication_id: str,
        *,
        expected_kind: str,
        expected_state: str,
        expected_phase: str,
        expected_operation_id: str | None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """CAS-mark an exact terminal publication's operation repair complete."""

        terminal_states = {"committed", "rolled_back", "failed", "manual"}
        if expected_state not in terminal_states:
            raise ValidationError(
                "only terminal runtime publications can complete operation reconciliation"
            )
        with self._lock:
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return False
            if (
                publication["kind"] == "checkpoint_restore"
                and _checkpoint_restore_writer_token
                is not self.__checkpoint_restore_writer_token
            ):
                raise ValidationError(
                    "checkpoint restore operation reconciliation requires "
                    "the internal writer"
                )
            plan = publication["plan"]
            durable_operation_id = str(plan.get("operation_id") or "")
            selected_operation_id = str(expected_operation_id or "")
            bound_operation_ids = self.list_operation_ids_by_runtime_publication_id(
                publication_id
            )
            binding_matches = (
                durable_operation_id == selected_operation_id
                and (
                    bool(selected_operation_id)
                    == (plan.get("operation_binding_version") == 1)
                )
                and bound_operation_ids
                == ([selected_operation_id] if selected_operation_id else [])
            )
            identity_matches = (
                publication["kind"] == expected_kind
                and publication["state"] == expected_state
                and publication["phase"] == expected_phase
                and binding_matches
            )
            if not identity_matches:
                return False
            if publication["operation_reconciled"]:
                return True
            updated = self._execute(
                "UPDATE runtime_publications SET operation_reconciled = 1, updated_at = ? "
                "WHERE publication_id = ? AND kind = ? AND state = ? AND phase = ? "
                "AND operation_reconciled = 0",
                (
                    utc_now(),
                    publication_id,
                    expected_kind,
                    expected_state,
                    expected_phase,
                ),
            )
            return updated.rowcount == 1

    def query_checkpoint_payload_delivery_attempts(
        self,
        *,
        after: CheckpointPayloadDeliveryAttempt | None,
        limit: int,
    ) -> CheckpointPayloadDeliveryAttemptPage:
        """Return one indexed page of unacknowledged delivery attempts."""

        selected_limit = self._runtime_publication_reconciliation_limit(limit)
        params: list[Any] = []
        cursor_clause = ""
        if after is not None:
            cursor_clause = (
                "AND (started_at COLLATE BINARY, attempt_id COLLATE BINARY) "
                "> (?, ?) "
            )
            params.extend((after.started_at, after.attempt_id))
        params.append(selected_limit + 1)
        rows = self._query(
            "SELECT attempt_id, owner_instance_id, started_at "
            "FROM checkpoint_payload_delivery_attempts "
            "INDEXED BY idx_checkpoint_payload_delivery_attempts_state "
            "WHERE state = 'preparing' "
            f"{cursor_clause}"
            "ORDER BY started_at COLLATE BINARY, attempt_id COLLATE BINARY "
            "LIMIT ?",
            params,
        )
        records = tuple(
            CheckpointPayloadDeliveryAttempt(
                started_at=str(row["started_at"]),
                attempt_id=str(row["attempt_id"]),
                owner_instance_id=str(row["owner_instance_id"]),
            )
            for row in rows[:selected_limit]
        )
        next_cursor = records[-1] if len(rows) > selected_limit and records else None
        return CheckpointPayloadDeliveryAttemptPage(
            records=records,
            next_cursor=next_cursor,
        )

    def get_checkpoint_payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None:
        """Read and validate one exact attempt control row through its PK."""

        if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
            raise ValidationError("checkpoint payload delivery attempt is invalid")
        rows = self._query(
            "SELECT state, acked_at FROM checkpoint_payload_delivery_attempts "
            "WHERE attempt_id = ? AND owner_instance_id = ? AND started_at = ?",
            (
                attempt.attempt_id,
                attempt.owner_instance_id,
                attempt.started_at,
            ),
        )
        if not rows:
            return None
        row = rows[0]
        try:
            state = CheckpointPayloadDeliveryAttemptState(row["state"])
            acked_at = row["acked_at"]
            if state is CheckpointPayloadDeliveryAttemptState.ACKED:
                if not isinstance(acked_at, str) or not acked_at:
                    raise ValueError("acked attempt requires acked_at")
            elif acked_at is not None:
                raise ValueError("open or aborted attempt cannot retain acked_at")
            return state
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid checkpoint payload delivery attempt state: {exc}"
            ) from exc

    def query_checkpoint_restore_payload_deliveries(
        self,
        *,
        delivery_state: PayloadDeliveryState | str,
        attempt_id: str | None,
        after: RuntimePublicationCursor | None,
        limit: int,
    ) -> RuntimePublicationPage:
        """Return one exact, indexed delivery page for an owner or attempt."""

        if delivery_state not in {"pending", "confirmed", "completed"}:
            raise ValidationError("checkpoint payload delivery state is invalid")
        if attempt_id is None and delivery_state != "pending":
            raise ValidationError(
                "checkpoint payload delivery query requires an attempt"
            )
        selected_limit = self._runtime_publication_reconciliation_limit(limit)
        clauses = [
            "kind = 'checkpoint_restore'",
            "state = 'committed'",
            "phase = 'reconciled'",
            "payload_delivery_state = ?",
        ]
        params: list[Any] = [delivery_state]
        if attempt_id is None:
            index = "idx_runtime_publications_payload_delivery_page"
        else:
            if not isinstance(attempt_id, str) or not attempt_id:
                raise ValidationError("checkpoint payload delivery attempt is invalid")
            index = "idx_runtime_publications_payload_delivery_attempt"
            clauses.append("payload_delivery_attempt_id = ?")
            params.append(attempt_id)
        if after is not None:
            clauses.append(
                "(created_at COLLATE BINARY, publication_id COLLATE BINARY) > (?, ?)"
            )
            params.extend((after.created_at, after.publication_id))
        params.append(selected_limit + 1)
        rows = self._query(
            f"SELECT * FROM runtime_publications INDEXED BY {index} "
            "/* checkpoint-payload-delivery */ "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at COLLATE BINARY, publication_id COLLATE BINARY "
            "LIMIT ?",
            params,
        )
        records = tuple(
            self._runtime_publication_row(row) for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = RuntimePublicationCursor(
                last["created_at"],
                last["publication_id"],
            )
        return RuntimePublicationPage(records=records, next_cursor=next_cursor)

    def begin_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        if (
            _checkpoint_restore_writer_token
            is not self.__checkpoint_restore_writer_token
        ):
            raise ValidationError(
                "checkpoint payload delivery attempts require the internal writer"
            )
        if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
            raise ValidationError("checkpoint payload delivery attempt is invalid")
        with self.transaction() as cursor:
            inserted = cursor.execute(
                "INSERT INTO checkpoint_payload_delivery_attempts ("
                "attempt_id, owner_instance_id, state, started_at, acked_at, updated_at"
                ") VALUES (?, ?, 'preparing', ?, NULL, ?) ON CONFLICT DO NOTHING",
                (
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.started_at,
                    attempt.started_at,
                ),
            )
            if inserted.rowcount != 1:
                return False
            self._require_checkpoint_payload_delivery_attempt_readback(
                cursor,
                attempt,
                expected_state="preparing",
                expected_acked_at=None,
            )
            return True

    def ack_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        if (
            _checkpoint_restore_writer_token
            is not self.__checkpoint_restore_writer_token
        ):
            raise ValidationError(
                "checkpoint payload delivery attempts require the internal writer"
            )
        if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
            raise ValidationError("checkpoint payload delivery attempt is invalid")
        now = utc_now()
        with self.transaction() as cursor:
            updated = cursor.execute(
                "UPDATE checkpoint_payload_delivery_attempts "
                "SET state = 'acked', acked_at = ?, updated_at = ? "
                "WHERE attempt_id = ? AND owner_instance_id = ? "
                "AND state = 'preparing' AND started_at = ? "
                "AND EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? "
                "AND payload_delivery_state = 'completed' "
                "AND owner_instance_id = ? "
                "AND operation_reconciled = 1 LIMIT 1"
                ") AND NOT EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? "
                "AND payload_delivery_state = 'confirmed' LIMIT 1"
                ") AND NOT EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? "
                "AND payload_delivery_state = 'completed' "
                "AND owner_instance_id = ? "
                "AND operation_reconciled = 0 LIMIT 1"
                ") AND NOT EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? "
                "AND payload_delivery_state = 'completed' "
                "AND owner_instance_id < ? LIMIT 1"
                ") AND NOT EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? "
                "AND payload_delivery_state = 'completed' "
                "AND owner_instance_id > ? LIMIT 1"
                ")",
                (
                    now,
                    now,
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.started_at,
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.attempt_id,
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                ),
            )
            if updated.rowcount != 1:
                return False
            self._require_checkpoint_payload_delivery_attempt_readback(
                cursor,
                attempt,
                expected_state="acked",
                expected_acked_at=now,
            )
            return True

    def abort_checkpoint_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        if (
            _checkpoint_restore_writer_token
            is not self.__checkpoint_restore_writer_token
        ):
            raise ValidationError(
                "checkpoint payload delivery attempts require the internal writer"
            )
        if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
            raise ValidationError("checkpoint payload delivery attempt is invalid")
        with self.transaction() as cursor:
            updated = cursor.execute(
                "UPDATE checkpoint_payload_delivery_attempts "
                "SET state = 'aborted', updated_at = ? "
                "WHERE attempt_id = ? AND owner_instance_id = ? "
                "AND state = 'preparing' AND started_at = ? "
                "AND NOT EXISTS ("
                "SELECT 1 FROM runtime_publications "
                "INDEXED BY idx_runtime_publications_payload_delivery_guard "
                "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
                "AND phase = 'reconciled' "
                "AND payload_delivery_attempt_id IS NOT NULL "
                "AND payload_delivery_attempt_id = ? LIMIT 1"
                ")",
                (
                    utc_now(),
                    attempt.attempt_id,
                    attempt.owner_instance_id,
                    attempt.started_at,
                    attempt.attempt_id,
                ),
            )
            if updated.rowcount != 1:
                return False
            self._require_checkpoint_payload_delivery_attempt_readback(
                cursor,
                attempt,
                expected_state="aborted",
                expected_acked_at=None,
            )
            return True

    def _checkpoint_payload_delivery_attempt_row(
        self,
        cursor: Any,
        attempt_id: str,
    ) -> Any | None:
        rows = list(
            cursor.execute(
                "SELECT attempt_id, owner_instance_id, state, started_at, acked_at "
                "FROM checkpoint_payload_delivery_attempts WHERE attempt_id = ?",
                (attempt_id,),
            )
        )
        return rows[0] if rows else None

    def _require_checkpoint_payload_delivery_attempt_readback(
        self,
        cursor: Any,
        attempt: CheckpointPayloadDeliveryAttempt,
        *,
        expected_state: str,
        expected_acked_at: str | None,
    ) -> None:
        row = self._checkpoint_payload_delivery_attempt_row(
            cursor,
            attempt.attempt_id,
        )
        try:
            matches = bool(
                row is not None
                and row["attempt_id"] == attempt.attempt_id
                and row["owner_instance_id"] == attempt.owner_instance_id
                and row["state"] == expected_state
                and row["started_at"] == attempt.started_at
                and row["acked_at"] == expected_acked_at
            )
        except (KeyError, TypeError):
            matches = False
        if not matches:
            raise ValidationError(
                "checkpoint payload delivery attempt readback diverged"
            )

    def _apply_checkpoint_payload_delivery_transition(
        self,
        publication: Mapping[str, Any],
        target: _PayloadDeliveryTransitionTarget,
    ) -> bool:
        control_attempt = target.control_attempt
        requires_attempt = control_attempt is not None
        state_predicate, state_params = _nullable_exact_cas_predicate(
            "payload_delivery_state",
            publication["payload_delivery_state"],
        )
        attempt_predicate, attempt_params = _nullable_exact_cas_predicate(
            "payload_delivery_attempt_id",
            publication["payload_delivery_attempt_id"],
        )
        started_predicate, started_params = _nullable_exact_cas_predicate(
            "payload_delivery_started_at",
            publication["payload_delivery_started_at"],
        )
        updated = self._execute(
            "UPDATE runtime_publications SET receipt_json = ?, "
            "payload_delivery_state = ?, payload_delivery_attempt_id = ?, "
            "payload_delivery_started_at = ?, operation_reconciled = ?, "
            "owner_instance_id = ?, updated_at = ? "
            "WHERE publication_id = ? AND kind = ? AND state = ? "
            "AND phase = ? AND owner_instance_id = ? "
            "AND operation_reconciled = ? AND receipt_json = ? "
            f"AND {state_predicate} "
            f"AND {attempt_predicate} "
            f"AND {started_predicate} "
            "AND (? = 0 OR EXISTS ("
            "SELECT 1 FROM checkpoint_payload_delivery_attempts "
            "WHERE attempt_id = ? AND owner_instance_id = ? "
            "AND started_at = ? AND state = 'preparing'"
            "))",
            (
                dumps(target.receipt),
                target.state,
                target.attempt_id,
                target.started_at,
                int(target.operation_reconciled),
                target.owner_instance_id,
                utc_now(),
                publication["publication_id"],
                "checkpoint_restore",
                "committed",
                "reconciled",
                publication["owner_instance_id"],
                int(publication["operation_reconciled"]),
                dumps(publication["receipt"]),
                *state_params,
                *attempt_params,
                *started_params,
                int(requires_attempt),
                control_attempt.attempt_id if control_attempt is not None else None,
                (
                    control_attempt.owner_instance_id
                    if control_attempt is not None
                    else None
                ),
                control_attempt.started_at if control_attempt is not None else None,
            ),
        )
        return updated.rowcount == 1

    def _require_checkpoint_payload_delivery_readback(
        self,
        publication_id: str,
        target: _PayloadDeliveryTransitionTarget,
    ) -> None:
        readback = self.get_runtime_publication(publication_id)
        if readback is None or (
            readback["payload_delivery_state"] != target.state
            or readback["payload_delivery_attempt_id"] != target.attempt_id
            or readback["payload_delivery_started_at"] != target.started_at
            or readback["receipt"] != target.receipt
            or readback["operation_reconciled"]
            is not target.operation_reconciled
            or readback["owner_instance_id"] != target.owner_instance_id
        ):
            raise ValidationError(
                "checkpoint payload delivery transition readback diverged"
            )

    def transition_checkpoint_restore_payload_delivery(
        self,
        publication_id: str,
        *,
        expected_delivery_state: str | None,
        delivery_state: str,
        expected_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        delivery_attempt: CheckpointPayloadDeliveryAttempt | None = None,
        owner_instance_id: str | None = None,
        recovery_lease_id: str | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """CAS the startup delivery handshake for restored volatile payloads."""

        _validate_payload_delivery_transition_request(
            expected_delivery_state=expected_delivery_state,
            delivery_state=delivery_state,
            expected_attempt=expected_attempt,
            delivery_attempt=delivery_attempt,
            owner_instance_id=owner_instance_id,
            recovery_lease_id=recovery_lease_id,
        )
        with self._join_or_begin_transaction():
            if (
                _checkpoint_restore_writer_token
                is not self.__checkpoint_restore_writer_token
            ):
                raise ValidationError(
                    "checkpoint restore payload delivery requires the internal writer"
                )
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return False
            target = _payload_delivery_transition_target(
                publication,
                expected_delivery_state=expected_delivery_state,
                delivery_state=delivery_state,
                expected_attempt=expected_attempt,
                delivery_attempt=delivery_attempt,
                owner_instance_id=owner_instance_id,
                recovery_lease_id=recovery_lease_id,
            )
            if target is None:
                return False
            if not self._apply_checkpoint_payload_delivery_transition(
                publication,
                target,
            ):
                return False
            self._require_checkpoint_payload_delivery_readback(
                publication_id,
                target,
            )
            return True

    def runtime_publication_exists_for_pid(self, pid: str, *, kind: str) -> bool:
        if not isinstance(pid, str) or not pid:
            raise ValidationError("runtime publication PID must not be empty")
        try:
            selected_kind = parse_runtime_publication_kind(kind).value
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return bool(
            self._query(
                "SELECT 1 AS present FROM runtime_publications "
                "WHERE pid = ? AND kind = ? LIMIT 1",
                (pid, selected_kind),
            )
        )

    def claim_runtime_publication_recovery(
        self,
        publication_id: str,
        *,
        claimant_instance_id: str,
        expected_owner_instance_id: str,
        expected_state: str,
        classification: str,
        max_attempts: int | None = None,
        allow_orphaned_claim_takeover: bool = False,
        claimed_state: str = "rollback_pending",
        _checkpoint_restore_writer_token: object | None = None,
    ) -> dict[str, Any] | None:
        """CAS-claim one incomplete publication and persist its attempt class.

        Callers must use the owner/state read by their recovery scan. A stale
        scanner therefore cannot steal a claim after another runtime wins;
        after a real restart, the newly observed owner becomes the next fenced
        expectation and the durable attempt counter advances.
        """

        if expected_state not in {
            "planning",
            "applying",
            "reconciliation_pending",
            "rollback_pending",
            "failed",
        }:
            return None
        if claimed_state not in {"reconciliation_pending", "rollback_pending"}:
            return None
        with self._lock:
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return None
            if (
                publication["kind"] == "checkpoint_restore"
                and _checkpoint_restore_writer_token
                is not self.__checkpoint_restore_writer_token
            ):
                raise ValidationError(
                    "checkpoint restore recovery claims require the internal writer"
                )
            if publication["kind"] == "checkpoint_restore":
                self._validate_checkpoint_restore_recovery_claim(
                    publication,
                    expected_state=expected_state,
                    classification=classification,
                    claimed_state=claimed_state,
                )
            if (
                publication["owner_instance_id"] != str(expected_owner_instance_id)
                or publication["state"] != str(expected_state)
            ):
                return None
            existing_claim = self._existing_publication_recovery_claim_action(
                publication,
                claimed_state=claimed_state,
                claimant_instance_id=claimant_instance_id,
                allow_orphaned_claim_takeover=allow_orphaned_claim_takeover,
            )
            if existing_claim == "resume":
                return publication
            if existing_claim == "reject":
                return None
            receipt = deepcopy(publication["receipt"])
            phases = receipt.setdefault("phases", [])
            if not isinstance(phases, list):
                raise ValidationError("runtime publication receipt phases must be a list")
            attempt = 1 + sum(
                1
                for phase in phases
                if isinstance(phase, dict) and phase.get("phase") == "recovery_claimed"
            )
            lease_id = new_id("publication_recovery")
            recovery = {
                "attempt": attempt,
                "classification": str(classification),
                "claimant_instance_id": str(claimant_instance_id),
                "lease_id": lease_id,
                "disposition": "retryable",
            }
            phases.append(
                {
                    "phase": "recovery_claimed",
                    **recovery,
                }
            )
            receipt["recovery"] = recovery
            exhausted = max_attempts is not None and attempt > int(max_attempts)
            selected_state = "manual" if exhausted else claimed_state
            selected_phase = (
                "recovery_attempts_exhausted" if exhausted else "recovery_claimed"
            )
            if exhausted:
                recovery["disposition"] = "manual"
                phases[-1]["disposition"] = "manual"
            cur = self._execute(
                """
                UPDATE runtime_publications
                   SET owner_instance_id = ?, state = ?, phase = ?, receipt_json = ?,
                       error_json = NULL, operation_reconciled = 0, updated_at = ?
                 WHERE publication_id = ? AND owner_instance_id = ? AND state = ?
                """,
                (
                    str(claimant_instance_id),
                    selected_state,
                    selected_phase,
                    dumps(receipt),
                    utc_now(),
                    publication_id,
                    str(expected_owner_instance_id),
                    str(expected_state),
                ),
            )
            if cur.rowcount != 1:
                return None
            return self.get_runtime_publication(publication_id)

    @classmethod
    def _validate_checkpoint_restore_recovery_claim(
        cls,
        publication: Mapping[str, Any],
        *,
        expected_state: str,
        classification: str,
        claimed_state: str,
    ) -> None:
        cls._checkpoint_restore_transcript(publication)
        if (
            publication["state"] not in {"reconciliation_pending", "failed"}
            or expected_state != publication["state"]
            or classification != "reconcile_checkpoint_restore"
            or claimed_state != "reconciliation_pending"
        ):
            raise ValidationError(
                "invalid checkpoint restore publication recovery claim"
            )

    @staticmethod
    def _existing_publication_recovery_claim_action(
        publication: Mapping[str, Any],
        *,
        claimed_state: str,
        claimant_instance_id: str,
        allow_orphaned_claim_takeover: bool,
    ) -> str:
        recovery = publication["receipt"].get("recovery")
        is_retryable_claim = (
            publication["state"] == claimed_state
            and publication["phase"] == "recovery_claimed"
            and isinstance(recovery, dict)
            and recovery.get("disposition") == "retryable"
        )
        if not is_retryable_claim:
            return "new"
        if recovery.get("claimant_instance_id") == str(claimant_instance_id):
            # Same-runtime retries resume the exact lease without consuming a
            # new durable attempt.
            return "resume"
        # Startup callers hold the backend-wide runtime lease and may opt into
        # taking over a claim orphaned by another Runtime instance.
        return "takeover" if allow_orphaned_claim_takeover else "reject"

    def advance_runtime_publication(
        self,
        publication_id: str,
        *,
        state: str,
        phase: str,
        receipt: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        expected_states: Iterable[str] | None = None,
        expected_phase: str | None = None,
        recovery_lease_id: str | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """Advance a publication with an optional state CAS predicate.

        ``None`` disables the state predicate.  A supplied but empty iterable
        matches no state and therefore rejects the mutation.
        """

        allowed_states = {
            "planning",
            "applying",
            "reconciliation_pending",
            "committed",
            "rollback_pending",
            "rolled_back",
            "failed",
            "manual",
        }
        if state not in allowed_states:
            raise ValidationError(f"invalid runtime publication state: {state}")
        with self._lock:
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return False
            if publication["kind"] == "checkpoint_restore":
                if (
                    _checkpoint_restore_writer_token
                    is not self.__checkpoint_restore_writer_token
                ):
                    raise ValidationError(
                        "checkpoint restore publication transitions require "
                        "the internal writer"
                    )
            if expected_states is not None:
                expected = set(expected_states)
                if publication["state"] not in expected:
                    return False
            if publication["kind"] == "checkpoint_restore":
                self._validate_checkpoint_restore_publication_advance(
                    publication,
                    state=state,
                    phase=phase,
                    receipt=receipt,
                    error=error,
                    recovery_lease_id=recovery_lease_id,
                )
            if expected_phase is not None and publication["phase"] != expected_phase:
                return False
            merged_receipt = deepcopy(publication["receipt"])
            if not self._advance_publication_recovery_disposition(
                merged_receipt,
                state=state,
                recovery_lease_id=recovery_lease_id,
            ):
                return False
            if receipt:
                phase_receipts = merged_receipt.setdefault("phases", [])
                if not isinstance(phase_receipts, list):
                    raise ValidationError("runtime publication receipt phases must be a list")
                phase_receipts.append(deepcopy(dict(receipt)))
            selected_error = (
                deepcopy(publication["error"])
                if error is None
                else deepcopy(dict(error))
            )
            phase_clause = " AND phase = ?" if expected_phase is not None else ""
            params: list[Any] = [
                state,
                phase,
                dumps(merged_receipt),
                dumps(selected_error) if selected_error is not None else None,
                utc_now(),
                publication_id,
                publication["state"],
            ]
            if expected_phase is not None:
                params.append(expected_phase)
            cur = self._execute(
                f"""
                UPDATE runtime_publications
                   SET state = ?, phase = ?, receipt_json = ?, error_json = ?,
                       operation_reconciled = 0, updated_at = ?
                 WHERE publication_id = ? AND state = ?{phase_clause}
                """,
                params,
            )
            return cur.rowcount == 1

    @classmethod
    def _validate_checkpoint_restore_publication_advance(
        cls,
        publication: Mapping[str, Any],
        *,
        state: str,
        phase: str,
        receipt: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
        recovery_lease_id: str | None,
    ) -> None:
        """Validate the checkpoint transcript before its CAS write."""

        current_state = str(publication["state"])
        selected_receipt = dict(receipt or {})
        phase_order, completed_phases, planned_work_ids, completed_work_ids = (
            cls._checkpoint_restore_transcript(publication)
        )
        if selected_receipt == {"phase": "main_state_committed"}:
            valid = cls._is_checkpoint_restore_main_commit(
                publication,
                state=state,
                phase=phase,
                error=error,
                completed_phases=completed_phases,
                completed_work_ids=completed_work_ids,
            )
        elif selected_receipt.get("phase") == "checkpoint_restore_finalizer_completed":
            valid = cls._is_checkpoint_restore_finalizer_completion(
                selected_receipt,
                current_state=current_state,
                state=state,
                phase=phase,
                error=error,
                phase_order=phase_order,
                completed_phases=completed_phases,
                planned_work_ids=planned_work_ids,
                completed_work_ids=completed_work_ids,
            )
        elif selected_receipt.get("phase") == "checkpoint_restore_phase_completed":
            valid = cls._is_checkpoint_restore_phase_completion(
                selected_receipt,
                current_state=current_state,
                state=state,
                phase=phase,
                error=error,
                phase_order=phase_order,
                completed_phases=completed_phases,
                planned_work_ids=planned_work_ids,
                completed_work_ids=completed_work_ids,
            )
        elif selected_receipt == {"phase": "reconciled"}:
            valid = cls._is_checkpoint_restore_finish(
                current_state=current_state,
                state=state,
                phase=phase,
                error=error,
                phase_order=phase_order,
                completed_phases=completed_phases,
                planned_work_ids=planned_work_ids,
                completed_work_ids=completed_work_ids,
            )
        else:
            valid = cls._is_checkpoint_restore_failure(
                selected_receipt,
                publication=publication,
                current_state=current_state,
                state=state,
                phase=phase,
                error=error,
                recovery_lease_id=recovery_lease_id,
                phase_order=phase_order,
                completed_phases=completed_phases,
            )
        if not valid:
            raise ValidationError(
                "invalid checkpoint restore publication transition: "
                f"{current_state}/{publication['phase']} -> {state}/{phase}"
            )

    @classmethod
    def _checkpoint_restore_transcript(
        cls,
        publication: Mapping[str, Any],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        phases = publication["receipt"].get("phases")
        if not isinstance(phases, list):
            raise ValidationError("checkpoint restore receipt phases are invalid")
        plan_version = publication["plan"].get("plan_version")
        phase_order = {
            1: [
                "image_reconciliation",
                "jit_source_reconciliation",
                "jit_pruning",
                "object_release_finalizers",
            ],
            2: [
                "object_payload_reconciliation",
                "image_reconciliation",
                "jit_source_reconciliation",
                "jit_pruning",
                "object_release_finalizers",
            ],
        }.get(plan_version)
        if (
            type(plan_version) is not int
            or phase_order is None
            or publication["plan"].get("phase_order") != phase_order
        ):
            raise ValidationError(
                "checkpoint restore publication plan program is invalid"
            )
        completed_phases = cls._checkpoint_restore_phase_markers(phases)
        if completed_phases != phase_order[: len(completed_phases)]:
            raise ValidationError(
                "checkpoint restore phase transcript is out of order"
            )
        planned_work_ids = cls._checkpoint_restore_planned_work_ids(
            publication["plan"]
        )
        cls._validate_checkpoint_restore_causal_transcript(
            publication,
            phases,
            phase_order=phase_order,
            planned_work_ids=planned_work_ids,
        )
        completed_work_ids = cls._checkpoint_restore_work_markers(phases)
        if completed_work_ids != planned_work_ids[: len(completed_work_ids)]:
            raise ValidationError(
                "checkpoint restore finalizer transcript is out of order"
            )
        return phase_order, completed_phases, planned_work_ids, completed_work_ids

    @staticmethod
    def _validate_checkpoint_restore_causal_transcript(
        publication: Mapping[str, Any],
        phases: list[Any],
        *,
        phase_order: list[str],
        planned_work_ids: list[str],
    ) -> None:
        program: list[dict[str, Any]] = [{"phase": "main_state_committed"}]
        program.extend(
            {
                "phase": "checkpoint_restore_phase_completed",
                "name": name,
            }
            for name in phase_order[:-1]
        )
        program.extend(
            {
                "phase": "checkpoint_restore_finalizer_completed",
                "work_id": work_id,
            }
            for work_id in planned_work_ids
        )
        program.append(
            {
                "phase": "checkpoint_restore_phase_completed",
                "name": phase_order[-1],
            }
        )
        program.append({"phase": "reconciled"})
        causal = [
            item
            for item in phases
            if not isinstance(item, dict) or item.get("phase") != "recovery_claimed"
        ]
        if causal != program[: len(causal)]:
            raise ValidationError("checkpoint restore causal transcript is invalid")
        state = str(publication["state"])
        invalid_state_shape = (
            (state == "planning" and bool(causal))
            or (state != "planning" and not causal)
            or (state == "committed" and causal != program)
            or (state != "committed" and causal == program)
        )
        if invalid_state_shape:
            raise ValidationError(
                "checkpoint restore state does not match its causal transcript"
            )
        SQLRuntimeStore._validate_checkpoint_restore_recovery_markers(phases)

    @staticmethod
    def _validate_checkpoint_restore_recovery_markers(phases: list[Any]) -> None:
        expected_attempt = 1
        main_seen = False
        for item in phases:
            if not isinstance(item, dict):
                raise ValidationError("checkpoint restore receipt marker is invalid")
            if item == {"phase": "main_state_committed"}:
                main_seen = True
                continue
            if item.get("phase") != "recovery_claimed":
                continue
            valid = (
                main_seen
                and set(item)
                == {
                    "phase",
                    "attempt",
                    "classification",
                    "claimant_instance_id",
                    "lease_id",
                    "disposition",
                }
                and item.get("attempt") == expected_attempt
                and item.get("classification") == "reconcile_checkpoint_restore"
                and bool(str(item.get("claimant_instance_id") or ""))
                and bool(str(item.get("lease_id") or ""))
                and item.get("disposition") in {"retryable", "manual"}
            )
            if not valid:
                raise ValidationError(
                    "checkpoint restore recovery transcript is invalid"
                )
            expected_attempt += 1

    @staticmethod
    def _checkpoint_restore_phase_markers(
        phases: list[Any],
    ) -> list[str]:
        return [
            str(item.get("name") or "")
            for item in phases
            if isinstance(item, dict)
            and item.get("phase") == "checkpoint_restore_phase_completed"
        ]

    @staticmethod
    def _checkpoint_restore_planned_work_ids(plan: Mapping[str, Any]) -> list[str]:
        work_items = plan.get("finalizer_work_items")
        if not isinstance(work_items, list):
            raise ValidationError("checkpoint restore finalizer plan is invalid")
        planned_work_ids = [
            str(item.get("work_id") or "")
            for item in work_items
            if isinstance(item, dict)
        ]
        if len(planned_work_ids) != len(work_items) or any(
            not work_id for work_id in planned_work_ids
        ):
            raise ValidationError("checkpoint restore finalizer plan is invalid")
        return planned_work_ids

    @staticmethod
    def _checkpoint_restore_work_markers(phases: list[Any]) -> list[str]:
        return [
            str(item.get("work_id") or "")
            for item in phases
            if isinstance(item, dict)
            and item.get("phase") == "checkpoint_restore_finalizer_completed"
        ]

    @staticmethod
    def _is_checkpoint_restore_main_commit(
        publication: Mapping[str, Any],
        *,
        state: str,
        phase: str,
        error: Mapping[str, Any] | None,
        completed_phases: list[str],
        completed_work_ids: list[str],
    ) -> bool:
        return (
            publication["state"] == "planning"
            and publication["phase"] == "planned"
            and state == "reconciliation_pending"
            and phase == "main_state_committed"
            and error is None
            and not completed_phases
            and not completed_work_ids
        )

    @staticmethod
    def _is_checkpoint_restore_finalizer_completion(
        receipt: Mapping[str, Any],
        *,
        current_state: str,
        state: str,
        phase: str,
        error: Mapping[str, Any] | None,
        phase_order: list[str],
        completed_phases: list[str],
        planned_work_ids: list[str],
        completed_work_ids: list[str],
    ) -> bool:
        work_id = str(receipt.get("work_id") or "")
        return (
            set(receipt) == {"phase", "work_id"}
            and current_state == "reconciliation_pending"
            and state == "reconciliation_pending"
            and phase == "object_release_finalizers"
            and len(completed_phases) == len(phase_order) - 1
            and len(completed_work_ids) < len(planned_work_ids)
            and work_id == planned_work_ids[len(completed_work_ids)]
            and error is None
        )

    @staticmethod
    def _is_checkpoint_restore_phase_completion(
        receipt: Mapping[str, Any],
        *,
        current_state: str,
        state: str,
        phase: str,
        error: Mapping[str, Any] | None,
        phase_order: list[str],
        completed_phases: list[str],
        planned_work_ids: list[str],
        completed_work_ids: list[str],
    ) -> bool:
        name = str(receipt.get("name") or "")
        next_index = len(completed_phases)
        finalizer_ready = (
            name != "object_release_finalizers"
            or completed_work_ids == planned_work_ids
        )
        return (
            set(receipt) == {"phase", "name"}
            and current_state == "reconciliation_pending"
            and state == "reconciliation_pending"
            and next_index < len(phase_order)
            and name == phase_order[next_index]
            and phase == f"{name}_completed"
            and error is None
            and finalizer_ready
        )

    @staticmethod
    def _is_checkpoint_restore_finish(
        *,
        current_state: str,
        state: str,
        phase: str,
        error: Mapping[str, Any] | None,
        phase_order: list[str],
        completed_phases: list[str],
        planned_work_ids: list[str],
        completed_work_ids: list[str],
    ) -> bool:
        return (
            current_state == "reconciliation_pending"
            and state == "committed"
            and phase == "reconciled"
            and completed_phases == phase_order
            and completed_work_ids == planned_work_ids
            and error is None
        )

    @staticmethod
    def _is_checkpoint_restore_failure(
        receipt: Mapping[str, Any],
        *,
        publication: Mapping[str, Any],
        current_state: str,
        state: str,
        phase: str,
        error: Mapping[str, Any] | None,
        recovery_lease_id: str | None,
        phase_order: list[str],
        completed_phases: list[str],
    ) -> bool:
        if (
            not receipt
            and current_state == "reconciliation_pending"
            and isinstance(error, Mapping)
            and set(error) == {"code", "error_type"}
            and error.get("code") == "checkpoint_restore_reconciliation_failed"
            and bool(str(error.get("error_type") or ""))
        ):
            recovery = publication["receipt"].get("recovery")
            active_recovery = (
                isinstance(recovery, dict)
                and recovery.get("disposition") == "retryable"
            )
            next_phase = (
                phase_order[len(completed_phases)]
                if len(completed_phases) < len(phase_order)
                else "terminalization"
            )
            if active_recovery:
                exact_lease = str(recovery.get("lease_id") or "") == str(
                    recovery_lease_id or ""
                )
                expected_phase = "startup_reconciliation_failed"
            else:
                exact_lease = recovery_lease_id is None
                expected_phase = f"{next_phase}_failed"
            durable_finalizer_failure = (
                state == "manual"
                and next_phase == "object_release_finalizers"
                and phase == "durable_finalizer_handler_unavailable"
                and error.get("error_type")
                == "DurableObjectFinalizerUnavailable"
            )
            ordinary_failure = state == "failed" and phase == expected_phase
            return exact_lease and (ordinary_failure or durable_finalizer_failure)
        return False

    @staticmethod
    def _advance_publication_recovery_disposition(
        receipt: dict[str, Any],
        *,
        state: str,
        recovery_lease_id: str | None,
    ) -> bool:
        recovery = receipt.get("recovery")
        if not isinstance(recovery, dict):
            return True
        if (
            recovery.get("disposition") == "retryable"
            and str(recovery.get("lease_id") or "")
            != str(recovery_lease_id or "")
        ):
            return False
        disposition = {
            "committed": "terminal",
            "rolled_back": "terminal",
            "failed": "retryable",
            "manual": "manual",
        }.get(state)
        if disposition is not None:
            recovery["disposition"] = disposition
        return True

    def update_runtime_publication_plan(
        self,
        publication_id: str,
        update: Mapping[str, Any],
        *,
        expected_states: Iterable[str] | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """Durably add the exact launch boot inputs before their effects.

        Process-exec and checkpoint-restore plans are complete when inserted
        and therefore immutable.  A no-op write remains supported so callers
        can deliberately dirty and revalidate the operation marker.

        ``None`` disables the state predicate.  A supplied but empty iterable
        matches no state and therefore rejects the mutation.
        """

        with self._lock:
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return False
            if (
                publication["kind"] == "checkpoint_restore"
                and _checkpoint_restore_writer_token
                is not self.__checkpoint_restore_writer_token
            ):
                raise ValidationError(
                    "checkpoint restore plan updates require the internal writer"
                )
            if expected_states is not None:
                expected = set(expected_states)
                if publication["state"] not in expected:
                    return False
            plan = deepcopy(publication["plan"])
            if plan.get("operation_binding_version") == 1 and (
                (
                    "operation_id" in update
                    and str(update["operation_id"] or "")
                    != str(plan.get("operation_id") or "")
                )
                or (
                    "operation_binding_version" in update
                    and update["operation_binding_version"] != 1
                )
            ):
                return False
            selected_update = deepcopy(dict(update))
            updated_plan = deepcopy(plan)
            updated_plan.update(selected_update)
            if updated_plan != plan:
                if (
                    publication["kind"] != "process_launch"
                    or publication["state"] not in {"planning", "applying"}
                    or set(selected_update)
                    != {"boot_kind", "materialized_workspace_root"}
                ):
                    return False
            cur = self._execute(
                """
                UPDATE runtime_publications
                   SET plan_json = ?, operation_reconciled = 0, updated_at = ?
                 WHERE publication_id = ? AND state = ?
                """,
                (
                    dumps(updated_plan),
                    utc_now(),
                    publication_id,
                    publication["state"],
                ),
            )
            return cur.rowcount == 1

    def record_runtime_publication_artifact(
        self,
        publication_id: str,
        artifact: Mapping[str, Any],
        *,
        expected_states: Iterable[str] | None = None,
        _checkpoint_restore_writer_token: object | None = None,
    ) -> bool:
        """Append one exact, idempotent compensation receipt.

        ``None`` disables the state predicate.  A supplied but empty iterable
        matches no state and therefore rejects the mutation.
        """

        selected = deepcopy(dict(artifact))
        artifact_id = str(selected.get("artifact_id") or "").strip()
        if not artifact_id:
            raise ValidationError("runtime publication artifact requires artifact_id")
        selected["artifact_id"] = artifact_id
        with self._lock:
            publication = self.get_runtime_publication(publication_id)
            if publication is None:
                return False
            if (
                publication["kind"] == "checkpoint_restore"
                and _checkpoint_restore_writer_token
                is not self.__checkpoint_restore_writer_token
            ):
                raise ValidationError(
                    "checkpoint restore artifacts require the internal writer"
                )
            if expected_states is not None:
                expected = set(expected_states)
                if publication["state"] not in expected:
                    return False
            if publication["kind"] == "checkpoint_restore":
                self._validate_checkpoint_restore_plan_anchor(
                    publication,
                    selected,
                )
            receipt = deepcopy(publication["receipt"])
            artifacts = receipt.setdefault("artifacts", [])
            if not isinstance(artifacts, list):
                raise ValidationError("runtime publication receipt artifacts must be a list")
            for current in artifacts:
                if isinstance(current, dict) and current.get("artifact_id") == artifact_id:
                    if current != selected:
                        raise ValidationError(
                            f"runtime publication artifact receipt changed: {artifact_id}"
                        )
                    return True
            artifacts.append(selected)
            cur = self._execute(
                """
                UPDATE runtime_publications
                   SET receipt_json = ?, updated_at = ?
                 WHERE publication_id = ? AND state = ?
                """,
                (
                    dumps(receipt),
                    utc_now(),
                    publication_id,
                    publication["state"],
                ),
            )
            return cur.rowcount == 1

    @staticmethod
    def _validate_checkpoint_restore_plan_anchor(
        publication: Mapping[str, Any],
        artifact: Mapping[str, Any],
    ) -> None:
        publication_id = str(publication["publication_id"])
        plan_version = publication["plan"].get("plan_version")
        if type(plan_version) is not int or plan_version not in {1, 2}:
            raise ValidationError(
                "invalid checkpoint restore publication plan anchor version"
            )
        expected = {
            "artifact_id": (
                f"{publication_id}:checkpoint_restore_plan:v{plan_version}"
            ),
            "artifact_type": "checkpoint_restore_plan_anchor",
            "anchor_version": plan_version,
            "plan_sha256": hashlib.sha256(
                dumps(dict(publication["plan"])).encode("utf-8")
            ).hexdigest(),
        }
        artifacts = publication["receipt"].get("artifacts")
        valid_existing = artifacts in ([], [expected])
        if (
            publication["state"] != "planning"
            or publication["phase"] != "planned"
            or not valid_existing
            or dict(artifact) != expected
        ):
            raise ValidationError(
                "invalid checkpoint restore publication plan anchor"
            )

    @staticmethod
    def _runtime_publication_row(row: Any) -> dict[str, Any]:
        operation_reconciled = row["operation_reconciled"]
        if type(operation_reconciled) not in {int, bool} or operation_reconciled not in {
            0,
            1,
            False,
            True,
        }:
            raise ValidationError(
                "runtime publication operation_reconciled must be stored as 0 or 1"
            )
        publication_id = row["publication_id"]
        try:
            return validate_runtime_publication_record(
                {
                    "publication_id": publication_id,
                    "kind": row["kind"],
                    "pid": row["pid"],
                    "owner_instance_id": row["owner_instance_id"],
                    "state": row["state"],
                    "phase": row["phase"],
                    "plan": loads(row["plan_json"], {}),
                    "receipt": loads(row["receipt_json"], {}),
                    "error": (
                        loads(row["error_json"]) if row["error_json"] else None
                    ),
                    "operation_reconciled": bool(operation_reconciled),
                    "payload_delivery_state": row["payload_delivery_state"],
                    "payload_delivery_attempt_id": row[
                        "payload_delivery_attempt_id"
                    ],
                    "payload_delivery_started_at": row[
                        "payload_delivery_started_at"
                    ],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid persisted runtime publication {publication_id!r}: {exc}"
            ) from exc

    def list_processes(self, limit: int | None = None, *, active_first: bool = False) -> list[AgentProcess]:
        params: list[Any] = []
        sql = "SELECT * FROM processes"
        if active_first:
            sql += " ORDER BY CASE WHEN status IN (?, ?, ?) THEN 1 ELSE 0 END, updated_at DESC, pid DESC"
            params.extend(
                [
                    ProcessStatus.EXITED.value,
                    ProcessStatus.FAILED.value,
                    ProcessStatus.KILLED.value,
                ]
            )
        else:
            sql += " ORDER BY created_at, pid"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [self._row_to_process(row) for row in self._query(sql, params)]

    def query_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage:
        """Return one stable, hard-bounded process page for startup scans."""

        selected_limit = _validated_jit_rehydration_limit(self.config, limit)
        clauses: list[str] = []
        params: list[Any] = []
        if after is not None:
            if not isinstance(after, ProcessCursor):
                raise ValidationError("process page cursor has an invalid type")
            clauses.append("(created_at, pid) > (?, ?)")
            params.extend((after.created_at, after.pid))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(selected_limit + 1)
        rows = self._query(
            f"SELECT * FROM processes{where} "
            "ORDER BY created_at, pid LIMIT ?",
            params,
        )
        records = tuple(self._row_to_process(row) for row in rows[:selected_limit])
        next_cursor = None
        if len(rows) > selected_limit:
            last = records[-1]
            next_cursor = ProcessCursor(last.created_at, last.pid)
        return ProcessPage(records=records, next_cursor=next_cursor)

    def query_process_tool_bindings(
        self,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> ProcessToolBindingPage:
        """Read one global, hard-bounded page of durable JIT bindings."""

        selected_limit = _validated_jit_rehydration_limit(self.config, limit)
        clauses = ["binding.jit_rehydration_eligible = 1"]
        params: list[Any] = []
        if after is not None:
            if not isinstance(after, ProcessToolBindingCursor):
                raise ValidationError("process tool binding cursor has an invalid type")
            clauses.append("(binding.pid, binding.tool_name) > (?, ?)")
            params.extend((after.pid, after.tool_name))
        params.append(selected_limit + 1)
        rows = self._query(
            "SELECT binding.pid, binding.tool_name, binding.tool_id "
            "FROM process_tool_bindings AS binding "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY binding.pid, binding.tool_name LIMIT ?",
            params,
        )
        selected_rows = rows[:selected_limit]
        records = tuple(
            ProcessToolBindingRecord(
                pid=str(row["pid"]),
                tool_name=str(row["tool_name"]),
                tool_id=str(row["tool_id"]),
            )
            for row in selected_rows
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = ProcessToolBindingCursor(
                last.pid,
                last.tool_name,
            )
        return ProcessToolBindingPage(records=records, next_cursor=next_cursor)

    def get_processes_with_ancestors(self, pids: Iterable[str]) -> list[AgentProcess]:
        selected = sorted({str(pid) for pid in pids if str(pid)})
        if not selected:
            return []
        placeholders = ", ".join("?" for _ in selected)
        rows = self._query(
            f"""
            WITH RECURSIVE ancestors(pid) AS (
                SELECT pid FROM processes WHERE pid IN ({placeholders})
                UNION
                SELECT processes.parent_pid
                  FROM processes
                  JOIN ancestors ON processes.pid = ancestors.pid
                 WHERE processes.parent_pid IS NOT NULL
            )
            SELECT processes.*
              FROM processes
              JOIN ancestors ON ancestors.pid = processes.pid
             ORDER BY processes.created_at, processes.pid
            """,
            selected,
        )
        return [self._row_to_process(row) for row in rows]

    def list_processes_by_status(self, status: ProcessStatus | str) -> list[AgentProcess]:
        selected = ProcessStatus(status).value
        rows = self._query(
            "SELECT * FROM processes WHERE status = ? ORDER BY created_at, pid",
            (selected,),
        )
        return [self._row_to_process(row) for row in rows]

    def query_orphaned_created_processes(
        self,
        *,
        after: ProcessCursor | None,
        limit: int,
    ) -> ProcessPage:
        """Return CREATED processes with no durable launch publication."""

        selected_limit = self._runtime_publication_reconciliation_limit(limit)
        clauses = [
            "processes.status = ?",
            "NOT EXISTS ("
            "SELECT 1 FROM runtime_publications "
            "WHERE runtime_publications.pid = processes.pid "
            "AND runtime_publications.kind = ? LIMIT 1 OFFSET 0)",
        ]
        params: list[Any] = [
            ProcessStatus.CREATED.value,
            "process_launch",
        ]
        if after is not None:
            clauses.append("(processes.created_at, processes.pid) > (?, ?)")
            params.extend((after.created_at, after.pid))
        params.append(selected_limit + 1)
        rows = self._query(
            f"SELECT processes.* FROM processes WHERE {' AND '.join(clauses)} "
            "ORDER BY processes.created_at, processes.pid LIMIT ?",
            params,
        )
        records = tuple(
            self._row_to_process(row) for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = ProcessCursor(last.created_at, last.pid)
        return ProcessPage(records=records, next_cursor=next_cursor)

    def list_child_processes(self, parent_pid: str) -> list[AgentProcess]:
        rows = self._query(
            "SELECT * FROM processes WHERE parent_pid = ? ORDER BY created_at, pid",
            (parent_pid,),
        )
        return [self._row_to_process(row) for row in rows]

    def insert_authority_manifest(self, manifest: TaskAuthorityManifest) -> None:
        if (
            manifest.permitted_effects_policy_schema_version
            != PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
            or manifest.metadata.get(PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY)
            != PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
        ):
            raise ValidationError(
                "new authority manifests require current effect-policy provenance"
            )
        self._execute(
            """
            INSERT INTO authority_manifests (
                manifest_id, pid, image_id, goal_ref,
                authorized_capabilities_json, required_capabilities_json,
                permitted_effects_json, resource_budget_json,
                approval_policy_json, data_flow_policy_json, expires_at,
                issued_by, parent_manifest_id, manifest_hash, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.manifest_id,
                manifest.pid,
                manifest.image_id,
                manifest.goal_ref,
                dumps(manifest.authorized_capabilities),
                dumps(manifest.required_capabilities),
                dumps(encode_permitted_effects_policy(manifest.permitted_effects)),
                dumps(manifest.resource_budget),
                dumps(manifest.approval_policy),
                dumps(manifest.data_flow_policy),
                manifest.expires_at,
                manifest.issued_by,
                manifest.parent_manifest_id,
                manifest.manifest_hash,
                dumps(manifest.metadata),
                manifest.created_at,
            ),
        )

    def get_authority_manifest(self, manifest_id: str) -> TaskAuthorityManifest | None:
        rows = self._query("SELECT * FROM authority_manifests WHERE manifest_id = ?", (manifest_id,))
        return self._row_to_authority_manifest(rows[0]) if rows else None

    def get_authority_manifest_for_process(self, pid: str) -> TaskAuthorityManifest | None:
        rows = self._query("SELECT * FROM authority_manifests WHERE pid = ?", (pid,))
        return self._row_to_authority_manifest(rows[0]) if rows else None

    def list_authority_manifests(
        self,
        *,
        parent_manifest_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskAuthorityManifest]:
        params: list[Any] = []
        where = ""
        if parent_manifest_id is not None:
            where = " WHERE parent_manifest_id = ?"
            params.append(parent_manifest_id)
        sql = f"SELECT * FROM authority_manifests{where} ORDER BY created_at, manifest_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [self._row_to_authority_manifest(row) for row in self._query(sql, params)]

    def claim_runnable_process(self, pid: str) -> AgentProcess | None:
        """Compatibility facade for legacy callers without an owner token."""

        token = self.claim_execution(pid, owner_id="legacy.scheduler")
        if token is None:
            return None
        return self.get_process(pid)

    def claim_execution(self, pid: str, *, owner_id: str) -> ProcessExecutionToken | None:
        if not owner_id:
            raise ValidationError("execution owner id must be non-empty")
        now = utc_now()
        lease_id = new_id("execution_lease")
        with self.transaction() as cur:
            rows = list(
                cur.execute(
                    """
                    SELECT revision, execution_generation, state_generation
                      FROM processes
                     WHERE pid = ? AND status = ?
                    """,
                    (pid, ProcessStatus.RUNNABLE.value),
                )
            )
            if not rows:
                return None
            generation = int(rows[0]["execution_generation"]) + 1
            updated = cur.execute(
                """
                UPDATE processes
                   SET status = ?, updated_at = ?, revision = revision + 1,
                       status_message = NULL, wait_state_json = 'null',
                       outcome_json = 'null', state_generation = state_generation + 1,
                       execution_generation = ?, execution_owner_id = ?, execution_lease_id = ?
                 WHERE pid = ? AND status = ?
                   AND execution_generation = ?
                """,
                (
                    ProcessStatus.RUNNING.value,
                    now,
                    generation,
                    owner_id,
                    lease_id,
                    pid,
                    ProcessStatus.RUNNABLE.value,
                    generation - 1,
                ),
            )
            if updated.rowcount != 1:
                return None
            self.observe_process_concurrency(
                pid,
                revision=int(rows[0]["revision"]) + 1,
                execution_generation=generation,
                state_generation=(
                    int(rows[0]["state_generation"]) + 1
                    if "state_generation" in rows[0].keys()
                    else 1
                ),
                cursor=cur,
            )
            return ProcessExecutionToken(
                pid=pid,
                generation=generation,
                owner_id=owner_id,
                lease_id=lease_id,
            )

    def claim_host_process_exec(
        self,
        pid: str,
        *,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
        expected_execution_generation: int,
    ) -> ProcessExecutionToken | None:
        """Claim a RUNNABLE row for Host exec using its complete concurrency tuple.

        ImageBoot captures the rollback snapshot and invokes this CAS inside one
        transaction.  A scheduler claim or state transition that wins between
        preflight and this boundary therefore prevents publication admission;
        the losing exec never owns a snapshot that it may replay.
        """

        if not owner_id:
            raise ValidationError("process exec owner id must be non-empty")
        expected = (
            expected_revision,
            expected_state_generation,
            expected_execution_generation,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected
        ):
            raise ValidationError(
                "process exec concurrency values must be non-negative integers"
            )
        generation = expected_execution_generation + 1
        lease_id = new_id("execution_lease")
        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE processes
                   SET status = ?, updated_at = ?, revision = revision + 1,
                       status_message = NULL, wait_state_json = 'null',
                       outcome_json = 'null', state_generation = state_generation + 1,
                       execution_generation = ?, execution_owner_id = ?, execution_lease_id = ?
                 WHERE pid = ? AND status = ? AND revision = ?
                   AND state_generation = ? AND execution_generation = ?
                   AND execution_owner_id IS NULL AND execution_lease_id IS NULL
                   AND wait_state_json = 'null'
                """,
                (
                    ProcessStatus.RUNNING.value,
                    utc_now(),
                    generation,
                    owner_id,
                    lease_id,
                    pid,
                    ProcessStatus.RUNNABLE.value,
                    expected_revision,
                    expected_state_generation,
                    expected_execution_generation,
                ),
            )
            if updated.rowcount != 1:
                return None
            self.observe_process_concurrency(
                pid,
                revision=expected_revision + 1,
                execution_generation=generation,
                state_generation=expected_state_generation + 1,
                cursor=cur,
            )
        return ProcessExecutionToken(
            pid=pid,
            generation=generation,
            owner_id=owner_id,
            lease_id=lease_id,
        )

    def claim_worker_process_exec(
        self,
        pid: str,
        *,
        execution_token: ProcessExecutionToken,
        owner_id: str,
        expected_revision: int,
        expected_state_generation: int,
    ) -> ProcessExecutionToken | None:
        """Atomically rotate an exact worker lease into exec-owned admission.

        The old worker token is invalid as soon as this CAS commits.  Exec then
        runs under the returned internal token, so a concurrent quantum
        completion either wins before admission (and prevents publication) or
        loses without being revivable by snapshot compensation.
        """

        if not owner_id:
            raise ValidationError("process exec owner id must be non-empty")
        if execution_token.pid != pid:
            raise ValidationError(
                f"process execution token for {execution_token.pid} cannot exec {pid}"
            )
        expected = (
            expected_revision,
            expected_state_generation,
            execution_token.generation,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected
        ):
            raise ValidationError(
                "process exec concurrency values must be non-negative integers"
            )
        generation = execution_token.generation + 1
        lease_id = new_id("execution_lease")
        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE processes
                   SET updated_at = ?, revision = revision + 1,
                       status_message = NULL, wait_state_json = 'null',
                       outcome_json = 'null', state_generation = state_generation + 1,
                       execution_generation = ?, execution_owner_id = ?, execution_lease_id = ?
                 WHERE pid = ? AND status = ? AND revision = ?
                   AND state_generation = ? AND execution_generation = ?
                   AND execution_owner_id = ? AND execution_lease_id = ?
                   AND wait_state_json = 'null'
                """,
                (
                    utc_now(),
                    generation,
                    owner_id,
                    lease_id,
                    pid,
                    ProcessStatus.RUNNING.value,
                    expected_revision,
                    expected_state_generation,
                    execution_token.generation,
                    execution_token.owner_id,
                    execution_token.lease_id,
                ),
            )
            if updated.rowcount != 1:
                return None
            self.observe_process_concurrency(
                pid,
                revision=expected_revision + 1,
                execution_generation=generation,
                state_generation=expected_state_generation + 1,
                cursor=cur,
            )
        return ProcessExecutionToken(
            pid=pid,
            generation=generation,
            owner_id=owner_id,
            lease_id=lease_id,
        )

    def complete_execution(
        self,
        token: ProcessExecutionToken,
        *,
        status: ProcessStatus | str = ProcessStatus.RUNNABLE,
        status_message: str | None = None,
        wait_state: ProcessWaitState | None = None,
        outcome: ProcessOutcome | None = None,
    ) -> bool:
        """Complete only the exact claimed generation/owner/lease."""

        selected = ProcessStatus(status)
        validate_process_state_fields(selected.value, wait_state, outcome)
        selected_status_message = legacy_status_message(
            wait_state,
            outcome,
            status_message,
        )
        wait_state_json = dumps(process_wait_state_to_mapping(wait_state))
        outcome_json = dumps(process_outcome_to_mapping(outcome))
        generation_increment = 1 if selected in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        } else 0
        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE processes
                   SET status = ?, status_message = ?, wait_state_json = ?,
                       outcome_json = ?, state_generation = state_generation + 1,
                       updated_at = ?, revision = revision + 1,
                       execution_generation = execution_generation + ?,
                       execution_owner_id = NULL, execution_lease_id = NULL
                 WHERE pid = ? AND status = ? AND execution_generation = ?
                   AND execution_owner_id = ? AND execution_lease_id = ?
                """,
                (
                    selected.value,
                    selected_status_message,
                    wait_state_json,
                    outcome_json,
                    utc_now(),
                    generation_increment,
                    token.pid,
                    ProcessStatus.RUNNING.value,
                    token.generation,
                    token.owner_id,
                    token.lease_id,
                ),
            )
            if updated.rowcount != 1:
                return False
            rows = list(
                cur.execute(
                    """
                    SELECT revision, execution_generation, state_generation
                      FROM processes WHERE pid = ?
                    """,
                    (token.pid,),
                )
            )
            self.observe_process_concurrency(
                token.pid,
                revision=int(rows[0]["revision"]),
                execution_generation=int(rows[0]["execution_generation"]),
                state_generation=int(rows[0]["state_generation"]),
                cursor=cur,
            )
            return True

    def release_execution(self, token: ProcessExecutionToken) -> bool:
        """Fence and detach an exact execution lease without changing status."""

        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE processes
                   SET updated_at = ?, revision = revision + 1,
                       execution_generation = execution_generation + 1,
                       execution_owner_id = NULL, execution_lease_id = NULL
                 WHERE pid = ? AND execution_generation = ?
                   AND execution_owner_id = ? AND execution_lease_id = ?
                """,
                (
                    utc_now(),
                    token.pid,
                    token.generation,
                    token.owner_id,
                    token.lease_id,
                ),
            )
            if updated.rowcount != 1:
                return False
            rows = list(
                cur.execute(
                    "SELECT revision, execution_generation FROM processes WHERE pid = ?",
                    (token.pid,),
                )
            )
            self.observe_process_concurrency(
                token.pid,
                revision=int(rows[0]["revision"]),
                execution_generation=int(rows[0]["execution_generation"]),
                cursor=cur,
            )
            return True

    def recover_stale_executions(
        self,
        *,
        owner_id: str,
        require_recovery_lease: Callable[[], None],
        on_recovered: Callable[[str], None],
    ) -> StaleExecutionRecoverySummary:
        """Fail closed persisted RUNNING rows owned by an earlier Runtime."""

        require_recovery_lease()
        if not callable(on_recovered):
            raise TypeError("stale execution recovery callback must be callable")
        page_size = self.config.runtime.operation_recovery_page_size
        recovered_total = 0
        recovered_sample: list[str] = []
        after_pid: str | None = None
        while True:
            with self.transaction() as cur:
                params: list[Any] = [ProcessStatus.RUNNING.value, owner_id]
                after_clause = ""
                if after_pid is not None:
                    after_clause = " AND pid > ?"
                    params.append(after_pid)
                params.append(page_size)
                rows = list(
                    cur.execute(
                        f"""
                        SELECT pid FROM processes
                         WHERE status = ?
                           AND (execution_owner_id IS NULL OR execution_owner_id <> ?)
                           {after_clause}
                         ORDER BY pid
                         LIMIT ?
                        """,
                        params,
                    )
                )
                pids = [str(row["pid"]) for row in rows]
                if not pids:
                    break
                placeholders = ", ".join("?" for _ in pids)
                cur.execute(
                    f"""
                    UPDATE processes
                       SET status = ?, status_message = ?, updated_at = ?,
                           wait_state_json = ?, outcome_json = 'null',
                           state_generation = state_generation + 1,
                           revision = revision + 1,
                           execution_generation = execution_generation + 1,
                           execution_owner_id = NULL, execution_lease_id = NULL
                     WHERE pid IN ({placeholders}) AND status = ?
                    """,
                    (
                        ProcessStatus.PAUSED.value,
                        "stale_execution_recovery",
                        utc_now(),
                        dumps(
                            process_wait_state_to_mapping(
                                PausedProcessWait(reason_oid=None)
                            )
                        ),
                        *pids,
                        ProcessStatus.RUNNING.value,
                    ),
                )
                concurrency_rows = list(
                    cur.execute(
                        f"""
                        SELECT pid, revision, execution_generation, state_generation
                          FROM processes
                         WHERE pid IN ({placeholders})
                        """,
                        pids,
                    )
                )
                for row in concurrency_rows:
                    self.observe_process_concurrency(
                        str(row["pid"]),
                        revision=int(row["revision"]),
                        execution_generation=int(row["execution_generation"]),
                        state_generation=int(row["state_generation"]),
                        cursor=cur,
                    )
                for pid in pids:
                    on_recovered(pid)
                recovered_total += len(pids)
                if len(recovered_sample) < page_size:
                    recovered_sample.extend(
                        pids[: page_size - len(recovered_sample)]
                    )
                after_pid = pids[-1]
        return StaleExecutionRecoverySummary(
            total_count=recovered_total,
            sample_pids=tuple(recovered_sample),
        )

    def upsert_resource_reservation(self, reservation: ResourceReservation) -> None:
        self._execute(
            """
            INSERT INTO process_resource_reservations (
                parent_pid, child_pid, reservation_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(parent_pid, child_pid) DO UPDATE SET
                reservation_json = excluded.reservation_json,
                updated_at = excluded.updated_at
            """,
            (
                reservation.parent_pid,
                reservation.child_pid,
                dumps(reservation.reserved),
                reservation.created_at,
                reservation.updated_at,
            ),
        )

    def get_resource_reservation(self, parent_pid: str, child_pid: str) -> ResourceReservation | None:
        rows = self._query(
            "SELECT * FROM process_resource_reservations WHERE parent_pid = ? AND child_pid = ?",
            (parent_pid, child_pid),
        )
        return self._row_to_resource_reservation(rows[0]) if rows else None

    def insert_resource_usage_reservation(
        self,
        *,
        reservation_id: str,
        pid: str,
        usage: ResourceUsage,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> None:
        self._execute(
            """
            INSERT INTO resource_usage_reservations (
                reservation_id, pid, usage_json, status, reserved_by, reason,
                settled_usage_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reservation_id,
                pid,
                dumps(usage),
                "active",
                reserved_by,
                reason,
                None,
                created_at,
                created_at,
            ),
        )

    def get_resource_usage_reservation(self, reservation_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM resource_usage_reservations WHERE reservation_id = ?",
            (reservation_id,),
        )
        return self._resource_usage_reservation_row(rows[0]) if rows else None

    def list_resource_usage_reservations(
        self,
        *,
        pid: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if pid is not None:
            clauses.append("pid = ?")
            params.append(pid)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM resource_usage_reservations{where} "
            "ORDER BY created_at, reservation_id",
            params,
        )
        return [self._resource_usage_reservation_row(row) for row in rows]

    def query_resource_usage_reservation_recovery(
        self,
        *,
        after: ResourceUsageReservationCursor | None,
        limit: int,
    ) -> ResourceUsageReservationPage:
        """Return one stable, hard-bounded page of active reservations."""

        selected_limit = self._resource_usage_reservation_recovery_limit(limit)
        clauses = ["status = ?"]
        params: list[Any] = [ResourceUsageReservationStatus.ACTIVE.value]
        if after is not None:
            if not isinstance(after, ResourceUsageReservationCursor):
                raise ValidationError(
                    "resource usage reservation recovery cursor has an invalid type"
                )
            clauses.append("(created_at, reservation_id) > (?, ?)")
            params.extend((after.created_at, after.reservation_id))
        params.append(selected_limit + 1)
        rows = self._query(
            "SELECT * FROM resource_usage_reservations "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at, reservation_id LIMIT ?",
            params,
        )
        records = tuple(
            self._row_to_resource_usage_reservation(row)
            for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit:
            last = records[-1]
            next_cursor = ResourceUsageReservationCursor(
                last.created_at,
                last.reservation_id,
            )
        return ResourceUsageReservationPage(
            records=records,
            next_cursor=next_cursor,
        )

    def settle_resource_usage_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        settled_usage: ResourceUsage,
        updated_at: str,
    ) -> bool:
        if status not in {"settled", "released", "charged_maximum"}:
            raise ValidationError(f"invalid resource usage reservation status: {status}")
        cur = self._execute(
            """
            UPDATE resource_usage_reservations
               SET status = ?, settled_usage_json = ?, updated_at = ?
             WHERE reservation_id = ? AND status = ?
            """,
            (status, dumps(settled_usage), updated_at, reservation_id, "active"),
        )
        return cur.rowcount == 1

    @staticmethod
    def _resource_usage_reservation_row(row: Any) -> dict[str, Any]:
        return {
            "reservation_id": str(row["reservation_id"]),
            "pid": str(row["pid"]),
            "usage": ResourceUsage(**loads(row["usage_json"], {})),
            "status": str(row["status"]),
            "reserved_by": str(row["reserved_by"]),
            "reason": str(row["reason"]),
            "settled_usage": (
                ResourceUsage(**loads(row["settled_usage_json"], {}))
                if row["settled_usage_json"]
                else None
            ),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _row_to_resource_usage_reservation(
        row: Any,
    ) -> ResourceUsageReservation:
        with _persisted_model_decode(
            f"resource usage reservation {row['reservation_id']}"
        ):
            return ResourceUsageReservation(
                reservation_id=str(row["reservation_id"]),
                pid=str(row["pid"]),
                usage=ResourceUsage(**loads(row["usage_json"], {})),
                status=ResourceUsageReservationStatus(str(row["status"])),
                reserved_by=str(row["reserved_by"]),
                reason=str(row["reason"]),
                settled_usage=(
                    ResourceUsage(**loads(row["settled_usage_json"], {}))
                    if row["settled_usage_json"]
                    else None
                ),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

    def get_jit_rehydration_artifacts(
        self,
        *,
        pid: str,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        """Read only requested ephemeral tools and their exact owner candidates."""

        if not isinstance(pid, str) or not pid or "\x00" in pid:
            raise ValidationError("JIT rehydration requires a valid process identity")
        return self.get_jit_rehydration_artifacts_for_tool_ids(tool_ids)

    def get_jit_rehydration_artifacts_for_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        """Bulk-read one hard-bounded page of exact ephemeral Tool artifacts."""

        selected_ids = self._validated_jit_rehydration_tool_ids(tool_ids)
        if not selected_ids:
            return ()

        tool_rows = self._query_jit_rehydration_tool_rows(selected_ids)
        if not tool_rows:
            return ()

        ephemeral_ids = sorted(str(row["tool_id"]) for row in tool_rows)
        # Do not predicate candidates by the requesting pid: returning the
        # exact durable owner is what lets configuration and startup reject a
        # cross-process alias instead of mistaking it for missing metadata.
        candidate_counts, first_candidates = (
            self._query_jit_rehydration_candidate_rows(
                tool_ids=ephemeral_ids,
                expected_ids=frozenset(selected_ids),
            )
        )
        return tuple(
            self._jit_rehydration_artifact_from_row(
                row,
                candidate_counts=candidate_counts,
                first_candidates=first_candidates,
            )
            for row in tool_rows
        )

    def delete_jit_tool_rows(
        self,
        pid: str,
        tool_ids: Iterable[str],
    ) -> None:
        """Atomically delete one process's exact ephemeral JIT artifacts.

        The durable binding eligibility column is a derived projection. Clear
        bindings for the exact requested identities, then delete only matching
        ephemeral Tool rows and owner-candidate rows in bounded set-based
        statements. This avoids one transaction per tool identity and also
        repairs an impossible stale-positive projection for a missing/static
        Tool row.
        """

        if not isinstance(pid, str) or not pid or "\x00" in pid:
            raise ValidationError("JIT tool deletion requires a valid process identity")
        if isinstance(tool_ids, (str, bytes, bytearray)):
            raise ValidationError("JIT tool deletion IDs must be an iterable")
        selected_ids: set[str] = set()
        for tool_id in tool_ids:
            if not isinstance(tool_id, str) or not tool_id or "\x00" in tool_id:
                raise ValidationError(
                    "JIT tool deletion IDs must be non-empty text"
                )
            selected_ids.add(tool_id)
        if not selected_ids:
            return

        ordered_ids = tuple(sorted(selected_ids))
        with self.transaction() as cursor:
            for offset in range(0, len(ordered_ids), _TOOL_ID_LOOKUP_BATCH_SIZE):
                batch = ordered_ids[
                    offset : offset + _TOOL_ID_LOOKUP_BATCH_SIZE
                ]
                placeholders = ", ".join("?" for _ in batch)
                cursor.execute(
                    "UPDATE process_tool_bindings "
                    "SET jit_rehydration_eligible = 0 "
                    "WHERE jit_rehydration_eligible = 1 "
                    f"AND tool_id IN ({placeholders})",
                    batch,
                )
                cursor.execute(
                    "DELETE FROM tools WHERE ephemeral = 1 "
                    f"AND tool_id IN ({placeholders})",
                    batch,
                )
                cursor.execute(
                    "DELETE FROM tool_candidates WHERE pid = ? "
                    f"AND registered_tool_id IN ({placeholders})",
                    (pid, *batch),
                )

    def _validated_jit_rehydration_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> tuple[str, ...]:
        if isinstance(tool_ids, (str, bytes, bytearray)):
            raise ValidationError("JIT rehydration tool IDs must be an iterable")
        hard_limit = self.config.runtime.jit_rehydration_page_hard_limit
        selected_ids: set[str] = set()
        for item_count, tool_id in enumerate(tool_ids, start=1):
            if item_count > hard_limit:
                raise ValidationError(
                    "JIT rehydration tool ID batch exceeds configured hard cap: "
                    f"> {hard_limit}"
                )
            if not isinstance(tool_id, str) or not tool_id or "\x00" in tool_id:
                raise ValidationError(
                    "JIT rehydration tool IDs must be non-empty text"
                )
            selected_ids.add(tool_id)
        return tuple(sorted(selected_ids))

    def _query_jit_rehydration_tool_rows(
        self,
        tool_ids: tuple[str, ...],
    ) -> list[Any]:
        tool_rows: list[Any] = []
        for offset in range(0, len(tool_ids), _TOOL_ID_LOOKUP_BATCH_SIZE):
            batch = tool_ids[offset : offset + _TOOL_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            tool_rows.extend(
                self._query(
                    "SELECT tool_id, name, scope FROM tools "
                    f"WHERE ephemeral = 1 AND tool_id IN ({placeholders}) "
                    "ORDER BY tool_id",
                    batch,
                )
            )
        return tool_rows

    def _query_jit_rehydration_candidate_rows(
        self,
        *,
        tool_ids: list[str],
        expected_ids: frozenset[str],
    ) -> tuple[dict[str, int], dict[str, Any]]:
        hard_limit = self.config.runtime.jit_rehydration_page_hard_limit
        candidate_counts: dict[str, int] = {}
        first_candidates: dict[str, Any] = {}
        candidate_row_total = 0
        for offset in range(0, len(tool_ids), _TOOL_ID_LOOKUP_BATCH_SIZE):
            batch = tool_ids[offset : offset + _TOOL_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            remaining = hard_limit - candidate_row_total
            rows = self._query(
                "SELECT registered_tool_id, candidate_id, pid, source_code "
                "FROM tool_candidates "
                f"WHERE registered_tool_id IN ({placeholders}) "
                "AND status = ? "
                "ORDER BY registered_tool_id, pid, candidate_id LIMIT ?",
                (*batch, ToolCandidateStatus.REGISTERED.value, remaining + 1),
            )
            candidate_row_total += len(rows)
            if candidate_row_total > hard_limit:
                raise ValidationError(
                    "JIT rehydration candidate result exceeds configured hard cap: "
                    f"> {hard_limit}"
                )
            self._accumulate_jit_rehydration_candidates(
                rows,
                expected_ids=expected_ids,
                candidate_counts=candidate_counts,
                first_candidates=first_candidates,
            )
        return candidate_counts, first_candidates

    @staticmethod
    def _accumulate_jit_rehydration_candidates(
        rows: Iterable[Any],
        *,
        expected_ids: frozenset[str],
        candidate_counts: dict[str, int],
        first_candidates: dict[str, Any],
    ) -> None:
        for row in rows:
            tool_id = str(row["registered_tool_id"] or "")
            if tool_id not in expected_ids:
                raise ValidationError(
                    "JIT rehydration candidate lookup returned an unexpected tool"
                )
            candidate_counts[tool_id] = candidate_counts.get(tool_id, 0) + 1
            first_candidates.setdefault(tool_id, row)

    @staticmethod
    def _jit_rehydration_artifact_from_row(
        row: Any,
        *,
        candidate_counts: dict[str, int],
        first_candidates: dict[str, Any],
    ) -> JITRehydrationArtifact:
        tool_id = str(row["tool_id"])
        candidate_count = candidate_counts.get(tool_id, 0)
        candidate = first_candidates.get(tool_id) if candidate_count == 1 else None
        return JITRehydrationArtifact(
            tool_id=tool_id,
            name=str(row["name"]),
            scope=str(row["scope"]),
            candidate_match_count=candidate_count,
            candidate_id=(
                str(candidate["candidate_id"]) if candidate is not None else None
            ),
            candidate_pid=(str(candidate["pid"]) if candidate is not None else None),
            source_code=(
                str(candidate["source_code"]) if candidate is not None else None
            ),
        )

    def list_resource_reservations(
        self,
        *,
        parent_pid: str | None = None,
        parent_pids: Iterable[str] | None = None,
        child_pid: str | None = None,
    ) -> list[ResourceReservation]:
        if parent_pid is not None and parent_pids is not None:
            raise ValueError("resource reservation query cannot combine parent_pid and parent_pids")
        clauses: list[str] = []
        params: list[Any] = []
        if parent_pid is not None:
            clauses.append("parent_pid = ?")
            params.append(parent_pid)
        if parent_pids is not None:
            selected_parent_pids = sorted({str(pid) for pid in parent_pids if str(pid)})
            if not selected_parent_pids:
                return []
            clauses.append(f"parent_pid IN ({', '.join('?' for _ in selected_parent_pids)})")
            params.extend(selected_parent_pids)
        if child_pid is not None:
            clauses.append("child_pid = ?")
            params.append(child_pid)
        sql = "SELECT * FROM process_resource_reservations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY parent_pid, child_pid"
        return [self._row_to_resource_reservation(row) for row in self._query(sql, params)]

    def delete_resource_reservation(self, parent_pid: str, child_pid: str) -> None:
        self._execute(
            "DELETE FROM process_resource_reservations WHERE parent_pid = ? AND child_pid = ?",
            (parent_pid, child_pid),
        )

    def delete_resource_reservations_for_process(self, pid: str) -> None:
        self._execute(
            "DELETE FROM process_resource_reservations WHERE parent_pid = ? OR child_pid = ?",
            (pid, pid),
        )

    def select_table_rows(
        self,
        table: str,
        where_sql: str = "",
        params: Iterable[Any] = (),
        *,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        table = self.validate_table_identifier(table)
        sql = f"SELECT * FROM {table}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        return [self._row_to_dict(row) for row in self._query(sql, params)]

    def insert_table_row(self, table: str, row: dict[str, Any]) -> None:
        table = self.validate_table_identifier(table)
        if table == "process_tool_bindings":
            raise ValidationError(
                "process tool bindings are a derived typed projection"
            )
        columns = list(row)
        for column in columns:
            self.validate_column_identifier(table, column)
        placeholders = ", ".join("?" for _ in columns)
        col_sql = ", ".join(columns)
        params = tuple(row[column] for column in columns)
        if table not in {"processes", "tools"}:
            self._execute(
                f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
                params,
            )
            return
        with self.transaction() as cursor:
            cursor.execute(
                f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
                params,
            )
            if table == "processes":
                self._replace_process_tool_bindings(
                    cursor,
                    str(row["pid"]),
                    loads(row.get("tool_table_json"), {}),
                    loads(row.get("model_tool_table_json"), {}),
                )
            else:
                self._refresh_process_binding_jit_eligibility(
                    cursor,
                    tool_id=str(row["tool_id"]),
                )

    def delete_table_rows(self, table: str, where_sql: str, params: Iterable[Any] = ()) -> None:
        table = self.validate_table_identifier(table)
        if table == "process_tool_bindings":
            raise ValidationError(
                "process tool bindings are a derived typed projection"
            )
        if table != "tools":
            self._execute(f"DELETE FROM {table} WHERE {where_sql}", params)
            return
        selected_params = tuple(params)
        selected_tool_ids = "SELECT tool_id FROM tools"
        if where_sql:
            selected_tool_ids += f" WHERE {where_sql}"
        with self.transaction() as cursor:
            cursor.execute(
                "UPDATE process_tool_bindings "
                "SET jit_rehydration_eligible = 0 "
                "WHERE jit_rehydration_eligible = 1 "
                f"AND tool_id IN ({selected_tool_ids})",
                selected_params,
            )
            delete_where = f" WHERE {where_sql}" if where_sql else ""
            cursor.execute(
                f"DELETE FROM tools{delete_where}",
                selected_params,
            )

    def payload_marker(
        self,
        *,
        present: bool,
        recovered_after_reopen: bool = False,
    ) -> dict[str, Any]:
        if present and recovered_after_reopen:
            raise ValueError("a present Object payload cannot be marked as recovered")
        marker: dict[str, Any] = {
            "storage": "runtime_memory",
            "present": present,
        }
        if recovered_after_reopen:
            marker["recovered_after_reopen"] = True
        return marker

    def object_payload(self, oid: str) -> Any:
        self._ensure_healthy()
        if oid in self._object_payloads:
            return deepcopy(self._object_payloads[oid])
        rows = self._query("SELECT payload_json FROM objects WHERE oid = ?", (oid,))
        if not rows:
            raise KeyError(oid)
        payload = self._decode_stored_object_payload(rows[0]["payload_json"])
        if payload is _MISSING_OBJECT_PAYLOAD:
            raise KeyError(oid)
        self._set_cached_object_payload(oid, payload)
        return deepcopy(payload)

    def set_object_payload(self, oid: str, payload: Any) -> None:
        self._ensure_healthy()
        with self.transaction(include_object_payloads=True) as cur:
            self._set_cached_object_payload(oid, payload)
            cur.execute(
                "UPDATE objects SET payload_json = ? WHERE oid = ?",
                (dumps(self.payload_marker(present=True)), oid),
            )

    def forget_object_payload(self, oid: str) -> None:
        self._ensure_healthy()
        self._forget_cached_object_payload(oid)

    def has_object_payload(self, oid: str, *, row: Any | None = None) -> bool:
        self._ensure_healthy()
        if oid in self._object_payloads:
            return True
        selected_row = row
        if selected_row is None:
            rows = self._query("SELECT payload_json FROM objects WHERE oid = ?", (oid,))
            selected_row = rows[0] if rows else None
        if selected_row is None:
            return False
        payload = self._decode_stored_object_payload(selected_row["payload_json"])
        if payload is _MISSING_OBJECT_PAYLOAD:
            return False
        self._set_cached_object_payload(oid, payload)
        return True

    def get_persisted_object_state(
        self,
        oid: str,
    ) -> PersistedObjectState | None:
        """Read the exact payload-free Object state used for revalidation."""

        with self._lock:
            rows = self._query(
                "SELECT oid, lifecycle_state, version, payload_json "
                "FROM objects WHERE oid = ?",
                (oid,),
            )
            if not rows:
                return None
            row = rows[0]
            with _persisted_model_decode("Object state"):
                return PersistedObjectState(
                    oid=row["oid"],
                    lifecycle_state=ObjectLifecycleState(row["lifecycle_state"]),
                    version=row["version"],
                    payload_present=(
                        oid in self._object_payloads
                        or _persisted_object_payload_is_present_without_cache(
                            row["payload_json"]
                        )
                    ),
                    recovered_after_reopen=(
                        _is_recovered_runtime_object_payload_marker(
                            row["payload_json"]
                        )
                    ),
                )

    def is_recovered_object_payload(self, oid: str) -> bool:
        rows = self._query("SELECT payload_json FROM objects WHERE oid = ?", (oid,))
        if not rows:
            return False
        return _is_recovered_runtime_object_payload_marker(
            rows[0]["payload_json"]
        )

    def snapshot_object_payloads(self, oids: Iterable[str]) -> dict[str, Any]:
        payloads: dict[str, Any] = {}
        for oid in oids:
            if self.has_object_payload(oid):
                payloads[oid] = self.object_payload(oid)
        return payloads

    def insert_event(self, event: Event) -> None:
        payload_json = dumps(event.payload)
        self._execute(
            """
            INSERT INTO events (
                event_id, type, source, target, payload_json, priority,
                created_at, correlation_id, causality_json, gui_snapshot_visible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.type.value,
                event.source,
                event.target,
                payload_json,
                event.priority.value,
                event.created_at,
                event.correlation_id,
                dumps(event.causality),
                int(
                    not is_gui_presentation_event_fields(
                        event.type,
                        loads(payload_json, {}),
                    )
                ),
            ),
        )

    def list_events(
        self,
        target: str | None = None,
        limit: int | None = None,
        before_event_id: str | None = None,
        after_event_id: str | None = None,
        *,
        include_gui_presentation: bool = True,
    ) -> list[Event]:
        if before_event_id is not None and after_event_id is not None:
            raise ValueError("event query cannot use before_event_id and after_event_id together")
        clauses: list[str] = []
        params: list[Any] = []
        if not include_gui_presentation:
            clauses.append("gui_snapshot_visible = 1")
        if target is not None:
            clauses.append("(target IS NULL OR target = ?)")
            params.append(target)
        if before_event_id is not None:
            cursor_rows = self._query(
                "SELECT created_at, event_id FROM events WHERE event_id = ?",
                (before_event_id,),
            )
            if not cursor_rows:
                return []
            cursor = cursor_rows[0]
            clauses.append("(created_at, event_id) < (?, ?)")
            params.extend((cursor["created_at"], cursor["event_id"]))
        if after_event_id is not None:
            cursor_rows = self._query(
                "SELECT created_at, event_id FROM events WHERE event_id = ?",
                (after_event_id,),
            )
            if not cursor_rows:
                return []
            cursor = cursor_rows[0]
            clauses.append("(created_at, event_id) > (?, ?)")
            params.extend((cursor["created_at"], cursor["event_id"]))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        if limit is None:
            rows = self._query(
                f"SELECT * FROM events{where} ORDER BY created_at, event_id",
                params,
            )
        elif after_event_id is not None:
            params.append(max(0, int(limit)))
            rows = self._query(
                f"SELECT * FROM events{where} ORDER BY created_at, event_id LIMIT ?",
                params,
            )
        else:
            selected_limit = max(0, int(limit))
            params.append(selected_limit)
            rows = self._query(
                "SELECT * FROM ("
                f"SELECT * FROM events{where} "
                "ORDER BY created_at DESC, event_id DESC LIMIT ?"
                ") AS recent_events ORDER BY created_at, event_id",
                params,
            )
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str) -> Event | None:
        rows = self._query("SELECT * FROM events WHERE event_id = ?", (event_id,))
        return self._row_to_event(rows[0]) if rows else None

    def insert_capability(self, cap: Capability) -> None:
        self._execute(
            """
            INSERT INTO capabilities (
                cap_id, subject, resource, rights_json, constraints_json,
                issued_by, issued_at, expires_at, delegable, revocable, effect,
                issuer_cap_id, parent_cap_id, delegation_depth, max_delegation_depth,
                uses_remaining, status, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cap.cap_id,
                cap.subject,
                cap.resource,
                dumps(cap.rights),
                dumps(cap.constraints),
                cap.issued_by,
                cap.issued_at,
                cap.expires_at,
                int(cap.delegable),
                int(cap.revocable),
                cap.effect.value,
                cap.issuer_cap_id,
                cap.parent_cap_id,
                cap.delegation_depth,
                cap.max_delegation_depth,
                cap.uses_remaining,
                cap.status.value,
                dumps(cap.metadata),
            ),
        )

    def consume_capability_uses(self, cap_id: str, count: int = 1) -> Capability | None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._join_or_begin_transaction() as cur:
            updated = cur.execute(
                """
                UPDATE capabilities
                   SET uses_remaining = uses_remaining - ?,
                       status = CASE
                           WHEN uses_remaining - ? <= 0 THEN ?
                           ELSE status
                       END
                 WHERE cap_id = ?
                   AND status = ?
                   AND uses_remaining IS NOT NULL
                   AND uses_remaining >= ?
                """,
                (
                    count,
                    count,
                    CapabilityStatus.REVOKED.value,
                    cap_id,
                    CapabilityStatus.ACTIVE.value,
                    count,
                ),
            )
            if updated.rowcount != 1:
                return None
            rows = list(cur.execute("SELECT * FROM capabilities WHERE cap_id = ?", (cap_id,)))
            return self._row_to_capability(rows[0]) if rows else None

    def reserve_capability_uses(
        self,
        cap_id: str,
        reservation_id: str,
        *,
        count: int = 1,
        reserved_by: str,
        reason: str,
        created_at: str,
    ) -> Capability | None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE capabilities
                   SET uses_remaining = uses_remaining - ?,
                       status = CASE
                           WHEN uses_remaining - ? <= 0 THEN ?
                           ELSE status
                       END
                 WHERE cap_id = ?
                   AND status = ?
                   AND uses_remaining IS NOT NULL
                   AND uses_remaining >= ?
                """,
                (
                    count,
                    count,
                    CapabilityStatus.REVOKED.value,
                    cap_id,
                    CapabilityStatus.ACTIVE.value,
                    count,
                ),
            )
            if updated.rowcount != 1:
                return None
            cur.execute(
                """
                INSERT INTO capability_use_reservations (
                    reservation_id, cap_id, count, status, reserved_by, reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (reservation_id, cap_id, count, "reserved", reserved_by, reason, created_at, created_at),
            )
            rows = list(cur.execute("SELECT * FROM capabilities WHERE cap_id = ?", (cap_id,)))
            return self._row_to_capability(rows[0]) if rows else None

    def commit_capability_use_reservation(self, reservation_id: str, *, updated_at: str) -> bool:
        with self._join_or_begin_transaction() as cur:
            updated = cur.execute(
                """
                UPDATE capability_use_reservations
                   SET status = ?, updated_at = ?
                 WHERE reservation_id = ? AND status = ?
                """,
                ("committed", updated_at, reservation_id, "reserved"),
            )
            return updated.rowcount == 1

    def restore_capability_use_reservation(self, reservation_id: str, *, updated_at: str) -> Capability | None:
        with self.transaction() as cur:
            reservations = list(
                cur.execute(
                    """
                    SELECT * FROM capability_use_reservations
                     WHERE reservation_id = ? AND status = ?
                    """,
                    (reservation_id, "reserved"),
                )
            )
            if not reservations:
                return None
            reservation = reservations[0]
            cap_id = str(reservation["cap_id"])
            count = int(reservation["count"])
            restored = cur.execute(
                """
                UPDATE capabilities
                   SET uses_remaining = uses_remaining + ?,
                       status = CASE
                           WHEN status = ? THEN ?
                           ELSE status
                       END
                 WHERE cap_id = ?
                   AND uses_remaining IS NOT NULL
                   AND status IN (?, ?)
                """,
                (
                    count,
                    CapabilityStatus.REVOKED.value,
                    CapabilityStatus.ACTIVE.value,
                    cap_id,
                    CapabilityStatus.ACTIVE.value,
                    CapabilityStatus.REVOKED.value,
                ),
            )
            if restored.rowcount != 1:
                cur.execute(
                    """
                    UPDATE capability_use_reservations
                       SET status = ?, updated_at = ?
                     WHERE reservation_id = ? AND status = ?
                    """,
                    ("invalidated", updated_at, reservation_id, "reserved"),
                )
                return None
            cur.execute(
                """
                UPDATE capability_use_reservations
                   SET status = ?, updated_at = ?
                 WHERE reservation_id = ? AND status = ?
                """,
                ("restored", updated_at, reservation_id, "reserved"),
            )
            rows = list(cur.execute("SELECT * FROM capabilities WHERE cap_id = ?", (cap_id,)))
            return self._row_to_capability(rows[0]) if rows else None

    def get_capability_use_reservation(self, reservation_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM capability_use_reservations WHERE reservation_id = ?",
            (reservation_id,),
        )
        return dict(rows[0]) if rows else None

    def update_capability(self, cap: Capability) -> None:
        sql = """
            UPDATE capabilities
               SET subject = ?, resource = ?, rights_json = ?, constraints_json = ?,
                   issued_by = ?, issued_at = ?, expires_at = ?, delegable = ?,
                   revocable = ?, effect = ?, issuer_cap_id = ?, parent_cap_id = ?,
                   delegation_depth = ?, max_delegation_depth = ?, uses_remaining = ?,
                   status = ?, metadata_json = ?
             WHERE cap_id = ?
            """
        params = (
            cap.subject,
            cap.resource,
            dumps(cap.rights),
            dumps(cap.constraints),
            cap.issued_by,
            cap.issued_at,
            cap.expires_at,
            int(cap.delegable),
            int(cap.revocable),
            cap.effect.value,
            cap.issuer_cap_id,
            cap.parent_cap_id,
            cap.delegation_depth,
            cap.max_delegation_depth,
            cap.uses_remaining,
            cap.status.value,
            dumps(cap.metadata),
            cap.cap_id,
        )
        if cap.status == CapabilityStatus.ACTIVE:
            self._execute(sql, params)
            return
        with self.transaction() as cur:
            cur.execute(sql, params)
            cur.execute(
                """
                UPDATE capability_use_reservations
                   SET status = ?, updated_at = ?
                 WHERE cap_id = ? AND status = ?
                """,
                ("invalidated", utc_now(), cap.cap_id, "reserved"),
            )

    def transition_capability_status(
        self,
        cap_id: str,
        *,
        expected_status: CapabilityStatus,
        status: CapabilityStatus,
        metadata: dict[str, Any],
    ) -> Capability | None:
        """Atomically move one capability only from its expected lifecycle state."""

        selected_expected = CapabilityStatus(expected_status)
        selected_status = CapabilityStatus(status)
        with self.transaction() as cur:
            updated = cur.execute(
                """
                UPDATE capabilities
                   SET status = ?, metadata_json = ?
                 WHERE cap_id = ? AND status = ?
                """,
                (
                    selected_status.value,
                    dumps(metadata),
                    cap_id,
                    selected_expected.value,
                ),
            )
            if updated.rowcount != 1:
                return None
            if selected_status != CapabilityStatus.ACTIVE:
                cur.execute(
                    """
                    UPDATE capability_use_reservations
                       SET status = ?, updated_at = ?
                     WHERE cap_id = ? AND status = ?
                    """,
                    ("invalidated", utc_now(), cap_id, "reserved"),
                )
            rows = list(
                cur.execute(
                    "SELECT * FROM capabilities WHERE cap_id = ?",
                    (cap_id,),
                )
            )
            return self._row_to_capability(rows[0]) if rows else None

    def get_capability(self, cap_id: str) -> Capability | None:
        rows = self._query("SELECT * FROM capabilities WHERE cap_id = ?", (cap_id,))
        return self._row_to_capability(rows[0]) if rows else None

    def list_capabilities(self, subject: str | None = None) -> list[Capability]:
        if subject is None:
            rows = self._query("SELECT * FROM capabilities ORDER BY subject ASC, issued_at ASC, cap_id ASC")
        else:
            rows = self._query(
                "SELECT * FROM capabilities WHERE subject = ? ORDER BY issued_at ASC, cap_id ASC",
                (subject,),
            )
        return [self._row_to_capability(row) for row in rows]

    def register_sink_trust(self, spec: SinkTrustSpec, *, replace: bool = False) -> SinkTrustSpec:
        if not isinstance(spec, SinkTrustSpec):
            raise ValidationError("sink trust registration requires a validated SinkTrustSpec")
        if not spec.active:
            raise ValidationError("new sink trust record must be active")
        with self.transaction() as cur:
            current = self._sink_trust_generation_from_cursor(cur)
            if spec.generation != current + 1:
                raise ValidationError(
                    f"sink trust generation conflict: expected {current + 1}, got {spec.generation}"
                )
            cur.execute(
                "SELECT trust_id FROM sink_trust_records WHERE pattern = ? AND active = 1",
                (spec.pattern,),
            )
            existing = cur.fetchone()
            if existing is not None and not replace:
                raise ValidationError(f"active sink trust rule already exists: {spec.pattern}")
            if existing is not None:
                cur.execute(
                    """
                    UPDATE sink_trust_records
                       SET active = 0, deactivated_at = ?
                     WHERE pattern = ? AND active = 1
                    """,
                    (spec.created_at, spec.pattern),
                )
            cur.execute(
                """
                INSERT INTO sink_trust_records (
                    trust_id, schema_version, pattern, trust_level,
                    max_sensitivity, tenants_json, principals_json,
                    identity_sha256, generation, spec_hash, active,
                    created_by, created_at, deactivated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.trust_id,
                    spec.schema_version,
                    spec.pattern,
                    spec.trust_level.value,
                    spec.max_sensitivity.value,
                    dumps(spec.tenants),
                    dumps(spec.principals),
                    spec.identity_sha256,
                    spec.generation,
                    spec.spec_hash,
                    1,
                    spec.created_by,
                    spec.created_at,
                    None,
                ),
            )
            cur.execute(
                "UPDATE sink_trust_registry SET generation = ?, updated_at = ? WHERE registry_key = 'default'",
                (spec.generation, spec.created_at),
            )
            if cur.rowcount != 1:
                raise ValidationError("sink trust registry metadata is missing")
        return spec

    def unregister_sink_trust(
        self,
        pattern: str,
        *,
        generation: int,
        deactivated_at: str,
    ) -> bool:
        try:
            SinkTrustRule(pattern=pattern)
        except ValueError as exc:
            raise ValidationError(f"invalid sink trust pattern: {exc}") from exc
        if not isinstance(deactivated_at, str) or not deactivated_at.strip():
            raise ValidationError("sink trust deactivated_at must be a non-empty string")
        with self.transaction() as cur:
            current = self._sink_trust_generation_from_cursor(cur)
            if generation != current + 1:
                raise ValidationError(
                    f"sink trust generation conflict: expected {current + 1}, got {generation}"
                )
            cur.execute(
                "SELECT trust_id FROM sink_trust_records WHERE pattern = ? AND active = 1",
                (pattern,),
            )
            if cur.fetchone() is None:
                return False
            cur.execute(
                """
                UPDATE sink_trust_records
                   SET active = 0, deactivated_at = ?
                 WHERE pattern = ? AND active = 1
                """,
                (deactivated_at, pattern),
            )
            if cur.rowcount != 1:
                raise ValidationError(f"sink trust rule changed concurrently: {pattern}")
            cur.execute(
                "UPDATE sink_trust_registry SET generation = ?, updated_at = ? WHERE registry_key = 'default'",
                (generation, deactivated_at),
            )
            if cur.rowcount != 1:
                raise ValidationError("sink trust registry metadata is missing")
        return True

    def get_sink_trust(self, trust_id: str) -> SinkTrustSpec | None:
        rows = self._query("SELECT * FROM sink_trust_records WHERE trust_id = ?", (trust_id,))
        return self._row_to_sink_trust(rows[0]) if rows else None

    def inspect_sink_trust(self, pattern: str) -> SinkTrustSpec | None:
        rows = self._query(
            "SELECT * FROM sink_trust_records WHERE pattern = ? AND active = 1",
            (pattern,),
        )
        return self._row_to_sink_trust(rows[0]) if rows else None

    def list_sink_trust(
        self,
        *,
        active_only: bool = True,
        generation: int | None = None,
        limit: int | None = None,
    ) -> list[SinkTrustSpec]:
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active = 1")
        if generation is not None:
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
                raise ValidationError("sink trust generation filter must be a non-negative integer")
            clauses.append("generation = ?")
            params.append(generation)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM sink_trust_records{where} ORDER BY generation DESC, pattern, trust_id"
        if limit is not None:
            selected_limit = self._data_flow_list_limit(
                limit,
                default=self.config.data_flow.registry_list_limit,
                hard_limit=self.config.data_flow.registry_list_limit,
                label="sink trust",
            )
            sql += " LIMIT ?"
            params.append(selected_limit)
        # Enforcement resolution calls this method without a limit and must
        # see the complete Host registry. Explicit administrative list calls
        # may opt into the configured bounded window.
        rows = self._query(sql, params)
        return [self._row_to_sink_trust(row) for row in rows]

    def get_sink_trust_generation(self) -> int:
        rows = self._query(
            "SELECT generation FROM sink_trust_registry WHERE registry_key = 'default'"
        )
        if not rows:
            raise ValidationError("sink trust registry metadata is missing")
        generation = rows[0]["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValidationError("invalid persisted sink trust registry generation")
        return generation

    def insert_data_flow_decision(self, decision: DataFlowDecision) -> None:
        if not isinstance(decision, DataFlowDecision):
            raise ValidationError("data-flow decision insert requires a validated DataFlowDecision")
        self._execute(
            """
            INSERT INTO data_flow_decisions (
                decision_id, pid, sink, direction, outcome, reason,
                labels_json, source_refs_json, payload_hash, trust_id,
                trust_hash, registry_generation, release_capability_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.pid,
                decision.sink,
                decision.direction.value,
                decision.outcome.value,
                decision.reason,
                dumps(decision.labels.to_dict()),
                dumps([ref.to_dict() for ref in decision.source_refs]),
                decision.payload_hash,
                decision.trust_id,
                decision.trust_hash,
                decision.registry_generation,
                decision.release_capability_id,
                decision.created_at,
            ),
        )

    def get_data_flow_decision(self, decision_id: str) -> DataFlowDecision | None:
        rows = self._query("SELECT * FROM data_flow_decisions WHERE decision_id = ?", (decision_id,))
        return self._row_to_data_flow_decision(rows[0]) if rows else None

    def list_data_flow_decisions(
        self,
        *,
        pid: str | None = None,
        sink: str | None = None,
        outcome: str | None = None,
        limit: int | None = None,
    ) -> list[DataFlowDecision]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("pid", pid), ("sink", sink)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if outcome is not None:
            try:
                selected_outcome = DataFlowOutcome(outcome).value
            except ValueError as exc:
                raise ValidationError(f"invalid data-flow outcome filter: {outcome}") from exc
            clauses.append("outcome = ?")
            params.append(selected_outcome)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        selected_limit = self._data_flow_list_limit(
            limit,
            default=self.config.data_flow.decision_list_limit,
            hard_limit=self.config.data_flow.decision_list_limit,
            label="data-flow decision",
        )
        rows = self._query(
            f"SELECT * FROM data_flow_decisions{where} "
            "ORDER BY created_at DESC, decision_id DESC LIMIT ?",
            [*params, selected_limit],
        )
        return [self._row_to_data_flow_decision(row) for row in rows]

    def upsert_file_label_binding(self, binding: FileLabelBinding) -> FileLabelBinding:
        if not isinstance(binding, FileLabelBinding):
            raise ValidationError("file label upsert requires a validated FileLabelBinding")
        if binding.tombstoned or not binding.active:
            raise ValidationError("file label upsert requires an active, non-tombstoned binding")
        with self.transaction() as cur:
            current_generation = self._file_label_generation_from_cursor(cur, binding.normalized_path)
            if binding.generation != current_generation + 1:
                raise ValidationError(
                    "file label generation conflict for "
                    f"{binding.normalized_path}: expected {current_generation + 1}, got {binding.generation}"
                )
            cur.execute(
                """
                UPDATE file_label_bindings
                   SET active = 0, superseded_at = ?
                 WHERE normalized_path = ? AND active = 1
                """,
                (binding.created_at, binding.normalized_path),
            )
            self._insert_file_label_binding(cur, binding)
        return binding

    def get_file_label_binding(self, normalized_path: str) -> FileLabelBinding | None:
        rows = self._query(
            """
            SELECT * FROM file_label_bindings
             WHERE normalized_path = ? AND active = 1 AND tombstoned = 0
            """,
            (normalized_path,),
        )
        return self._row_to_file_label_binding(rows[0]) if rows else None

    def get_file_label_binding_by_id(
        self,
        binding_id: str,
    ) -> FileLabelBinding | None:
        rows = self._query(
            """
            SELECT * FROM file_label_bindings
             WHERE binding_id = ?
            """,
            (binding_id,),
        )
        return self._row_to_file_label_binding(rows[0]) if rows else None

    def get_file_label_binding_generation(self, normalized_path: str) -> int:
        rows = self._query(
            "SELECT MAX(generation) AS generation FROM file_label_bindings WHERE normalized_path = ?",
            (normalized_path,),
        )
        value = rows[0]["generation"] if rows else None
        return int(value) if value is not None else 0

    def list_file_label_bindings(
        self,
        *,
        normalized_path: str | None = None,
        include_history: bool = False,
        include_tombstones: bool = False,
        limit: int | None = None,
    ) -> list[FileLabelBinding]:
        clauses: list[str] = []
        params: list[Any] = []
        if normalized_path is not None:
            clauses.append("normalized_path = ?")
            params.append(normalized_path)
        if not include_history:
            clauses.append("active = 1")
        if not include_tombstones:
            clauses.append("tombstoned = 0")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        selected_limit = self._data_flow_list_limit(
            limit,
            default=self.config.data_flow.file_binding_list_limit,
            hard_limit=self.config.data_flow.file_binding_list_limit,
            label="file label binding",
        )
        rows = self._query(
            f"SELECT * FROM file_label_bindings{where} "
            "ORDER BY normalized_path, generation DESC, binding_id LIMIT ?",
            [*params, selected_limit],
        )
        return [self._row_to_file_label_binding(row) for row in rows]

    def list_file_label_bindings_for_tree(
        self,
        normalized_path: str,
    ) -> list[FileLabelBinding]:
        selected = str(normalized_path).rstrip("/")
        batch_size = self.config.data_flow.file_binding_list_limit
        found: list[FileLabelBinding] = []
        last_path: str | None = None
        while True:
            params: list[Any] = []
            clauses = [
                "active = 1",
                "tombstoned = 0",
                "normalized_path COLLATE BINARY != '.git'",
                "normalized_path COLLATE BINARY NOT LIKE '.git/%'",
            ]
            if selected not in {"", "."}:
                descendant_prefix = f"{selected}/"
                escaped_prefix = (
                    descendant_prefix.replace("!", "!!")
                    .replace("%", "!%")
                    .replace("_", "!_")
                )
                # The outer range drives an index seek; the escaped LIKE keeps
                # exact subtree semantics for adjacent names such as treehouse.
                upper_bound = f"{selected}\U0010ffff"
                clauses.extend(
                    (
                        "normalized_path COLLATE BINARY >= ?",
                        "normalized_path COLLATE BINARY < ?",
                        "(normalized_path COLLATE BINARY = ? "
                        "OR normalized_path COLLATE BINARY LIKE ? ESCAPE '!')",
                    )
                )
                params.extend(
                    (selected, upper_bound, selected, f"{escaped_prefix}%")
                )
            if last_path is not None:
                clauses.append("normalized_path COLLATE BINARY > ?")
                params.append(last_path)
            params.append(batch_size)
            rows = self._query(
                "SELECT * FROM file_label_bindings "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY normalized_path COLLATE BINARY, generation DESC, binding_id LIMIT ?",
                params,
            )
            found.extend(self._row_to_file_label_binding(row) for row in rows)
            if len(rows) < batch_size:
                break
            last_path = str(rows[-1]["normalized_path"])
        return found

    def tombstone_file_label_binding(
        self,
        normalized_path: str,
        *,
        binding_id: str,
        created_by: str,
        created_at: str,
        expected_binding_id: str | None = None,
        expected_generation: int | None = None,
    ) -> FileLabelBinding | None:
        if (expected_binding_id is None) != (expected_generation is None):
            raise ValidationError(
                "file label tombstone CAS requires both binding ID and generation"
            )
        with self.transaction() as cur:
            cur.execute(
                """
                SELECT * FROM file_label_bindings
                 WHERE normalized_path = ? AND active = 1 AND tombstoned = 0
                """,
                (normalized_path,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            previous = self._row_to_file_label_binding(row)
            if expected_binding_id is not None and (
                previous.binding_id != expected_binding_id
                or previous.generation != expected_generation
            ):
                return None
            tombstone = FileLabelBinding(
                binding_id=binding_id,
                normalized_path=normalized_path,
                content_sha256=None,
                labels=previous.labels,
                source_refs=previous.source_refs,
                generation=previous.generation + 1,
                tombstoned=True,
                active=True,
                created_by=created_by,
                created_at=created_at,
            )
            cur.execute(
                """
                UPDATE file_label_bindings
                   SET active = 0, superseded_at = ?
                 WHERE binding_id = ? AND active = 1
                """,
                (created_at, previous.binding_id),
            )
            if cur.rowcount != 1:
                if expected_binding_id is not None:
                    return None
                raise ValidationError(f"file label binding changed concurrently: {normalized_path}")
            self._insert_file_label_binding(cur, tombstone)
        return tombstone

    def insert_audit(self, record: AuditRecord) -> None:
        decision_json = (
            dumps(record.decision) if record.decision is not None else None
        )
        self._execute(
            """
            INSERT INTO audit_records (
                record_id, timestamp, actor, action, target, input_refs_json,
                output_refs_json, capability_refs_json, decision_json,
                correlation_id, parent_record_id, gui_snapshot_visible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.timestamp,
                record.actor,
                record.action,
                record.target,
                dumps(record.input_refs),
                dumps(record.output_refs),
                dumps(record.capability_refs),
                decision_json,
                record.correlation_id,
                record.parent_record_id,
                int(
                    not is_gui_presentation_audit_fields(
                        record.action,
                        record.target,
                        loads(decision_json) if decision_json is not None else None,
                    )
                ),
            ),
        )

    def list_audit(
        self,
        limit: int | None = None,
        *,
        actor: str | None = None,
        target: str | None = None,
        match_any: bool = False,
        include_gui_presentation: bool = True,
    ) -> list[AuditRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_gui_presentation:
            clauses.append("gui_snapshot_visible = 1")
        selectors: list[str] = []
        if actor is not None:
            selectors.append("actor = ?")
            params.append(actor)
        if target is not None:
            selectors.append("target = ?")
            params.append(target)
        if selectors:
            joiner = " OR " if match_any and len(selectors) > 1 else " AND "
            clauses.append(f"({joiner.join(selectors)})")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "ORDER BY timestamp, record_id"
        if limit is None:
            return [self._row_to_audit(row) for row in self._query(f"SELECT * FROM audit_records{where} {order}", params)]
        selected_limit = int(limit)
        if selected_limit <= 0:
            return []
        # Limited audit reads are used by the GUI and API list views. Select the
        # newest window first, then return it chronologically so append streams
        # do not lose recent records once the table is larger than the window.
        limited = (
            f"SELECT audit_records.* FROM audit_records{where} "
            "ORDER BY timestamp DESC, record_id DESC LIMIT ?"
        )
        rows = self._query(
            f"SELECT * FROM ({limited}) AS limited_audit ORDER BY timestamp, record_id",
            [*params, selected_limit],
        )
        return [self._row_to_audit(row) for row in rows]

    def get_audit(self, record_id: str) -> AuditRecord | None:
        rows = self._query("SELECT * FROM audit_records WHERE record_id = ?", (record_id,))
        return self._row_to_audit(rows[0]) if rows else None

    def insert_operation(self, record: OperationRecord) -> None:
        runtime_publication_id = _operation_runtime_publication_id(record.metadata)
        with self._join_or_begin_transaction():
            self._require_operation_publication_binding_available(
                runtime_publication_id,
                operation_id=record.operation_id,
            )
            self._execute(
                """
                INSERT INTO operations (
                    operation_id, root_operation_id, parent_operation_id, kind, name,
                    actor, pid, state, outcome, expected_roles_json, metadata_json,
                    runtime_publication_id, started_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.operation_id,
                    record.root_operation_id,
                    record.parent_operation_id,
                    record.kind.value,
                    record.name,
                    record.actor,
                    record.pid,
                    record.state.value,
                    record.outcome.value,
                    dumps(record.expected_roles),
                    dumps(record.metadata),
                    runtime_publication_id,
                    record.started_at,
                    record.updated_at,
                    record.completed_at,
                ),
            )
            self._invalidate_runtime_publication_operation_reconciliation(
                (runtime_publication_id,)
            )

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        rows = self._query("SELECT * FROM operations WHERE operation_id = ?", (operation_id,))
        return self._row_to_operation(rows[0]) if rows else None

    def list_operation_ids_by_runtime_publication_id(
        self,
        publication_id: str,
    ) -> list[str]:
        selected = str(publication_id)
        if not selected:
            return []
        rows = self._query(
            "SELECT operation_id FROM operations "
            "WHERE runtime_publication_id = ? ORDER BY operation_id",
            (selected,),
        )
        return [str(row["operation_id"]) for row in rows]

    def _require_operation_publication_binding_available(
        self,
        publication_id: str | None,
        *,
        operation_id: str,
    ) -> None:
        if publication_id is None:
            return
        bound_ids = self.list_operation_ids_by_runtime_publication_id(publication_id)
        if any(bound_id != operation_id for bound_id in bound_ids):
            raise ValidationError(
                "runtime publication is already bound to another operation: "
                f"{publication_id} -> {bound_ids}"
            )

    def _invalidate_runtime_publication_operation_reconciliation(
        self,
        publication_ids: Iterable[str | None],
    ) -> None:
        """Dirty exact publication bindings in the caller's SQL transaction."""

        selected_ids = sorted(
            {
                publication_id
                for publication_id in publication_ids
                if publication_id is not None
            }
        )
        if not selected_ids:
            return
        placeholders = ", ".join("?" for _ in selected_ids)
        self._execute(
            "UPDATE runtime_publications "
            "SET operation_reconciled = 0, updated_at = ? "
            f"WHERE publication_id IN ({placeholders})",
            (utc_now(), *selected_ids),
        )

    def list_operations(
        self,
        *,
        pid: str | None = None,
        root_operation_id: str | None = None,
        roots_only: bool = False,
        state: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if pid is not None:
            clauses.append("pid = ?")
            params.append(pid)
        if root_operation_id is not None:
            clauses.append("root_operation_id = ?")
            params.append(root_operation_id)
        if roots_only:
            clauses.append("parent_operation_id IS NULL")
        if state is not None:
            clauses.append("state = ?")
            params.append(str(state))
        if cursor is not None:
            cursor_row = self.get_operation(cursor)
            if cursor_row is None:
                return []
            clauses.append("(started_at, operation_id) < (?, ?)")
            params.extend((cursor_row.started_at, cursor_row.operation_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM operations{where} ORDER BY started_at DESC, operation_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [self._row_to_operation(row) for row in self._query(sql, params)]

    def scan_stale_running_operations(
        self,
        *,
        after: OperationCursor | None,
        limit: int,
    ) -> OperationPage:
        """Read one stable, hard-bounded startup-recovery page."""

        selected_limit = self._operation_recovery_limit(limit)
        clauses = ["state = ?"]
        params: list[Any] = [OperationState.RUNNING.value]
        if after is not None:
            if not isinstance(after, OperationCursor):
                raise ValidationError("operation recovery cursor has an invalid type")
            clauses.append("(started_at, operation_id) < (?, ?)")
            params.extend((after.started_at, after.operation_id))
        params.append(selected_limit + 1)
        rows = self._query(
            "SELECT * FROM operations "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC, operation_id DESC LIMIT ?",
            params,
        )
        records = tuple(self._row_to_operation(row) for row in rows[:selected_limit])
        next_cursor = None
        if len(rows) > selected_limit:
            last = records[-1]
            next_cursor = OperationCursor(last.started_at, last.operation_id)
        return OperationPage(records=records, next_cursor=next_cursor)

    def operation_ids_with_unknown_external_effects(
        self,
        operation_ids: Iterable[str],
    ) -> set[str]:
        """Read one bounded page from the active startup recovery index."""
        with self._lock:
            self._ensure_healthy()
            if not self._stale_operation_recovery_index_active:
                raise ValidationError(
                    "stale operation recovery index is not active"
                )
            hard_limit = self.config.runtime.operation_recovery_page_hard_limit
            selected: list[str] = []
            seen: set[str] = set()
            for index, operation_id in enumerate(operation_ids):
                if index >= hard_limit:
                    raise ValidationError(
                        "operation recovery id batch exceeds configured hard cap: "
                        f"> {hard_limit}"
                    )
                if not isinstance(operation_id, str) or not operation_id:
                    raise ValidationError(
                        "operation recovery ids must be non-empty text"
                    )
                if operation_id not in seen:
                    seen.add(operation_id)
                    selected.append(operation_id)
            selected_ids = tuple(selected)
            if not selected_ids:
                return set()
            placeholders = ", ".join("?" for _ in selected_ids)
            rows = self._query(
                "SELECT operation_id "
                f"FROM {_STALE_OPERATION_RECOVERY_TEMP_TABLE} "
                f"WHERE operation_id IN ({placeholders}) ORDER BY operation_id",
                selected_ids,
            )
        return {str(row["operation_id"]) for row in rows}

    @contextmanager
    def stale_operation_recovery_index(self) -> Iterator[None]:
        """Materialize uncertain running ancestors in a connection-local table.

        The store lock spans the snapshot, every bounded membership page, and
        cleanup.  Reentrant lifecycle writes can terminalize each page, while
        another thread using the same store cannot mutate evidence underneath
        the recovery classification.
        """

        table = _STALE_OPERATION_RECOVERY_TEMP_TABLE
        with self._lock:
            self._ensure_healthy()
            if self._stale_operation_recovery_index_depth > 0:
                self._stale_operation_recovery_index_depth += 1
                try:
                    yield
                finally:
                    self._stale_operation_recovery_index_depth -= 1
                return
            self._stale_operation_recovery_index_depth = 1
            try:
                self._execute(
                    f"CREATE TEMP TABLE IF NOT EXISTS {table} ("
                    "operation_id TEXT PRIMARY KEY)"
                )
                self._execute(f"DELETE FROM {table}")
                self._populate_stale_operation_recovery_index(table)
                self._stale_operation_recovery_index_active = True
            except BaseException as setup_error:
                self._stale_operation_recovery_index_depth = 0
                self._stale_operation_recovery_index_active = False
                try:
                    self._execute(f"DROP TABLE IF EXISTS {table}")
                except BaseException as cleanup_error:
                    self._poison(
                        "stale operation recovery setup and cleanup failed: "
                        f"{cleanup_error}"
                    )
                    setup_error.add_note(
                        "stale operation recovery cleanup also failed: "
                        f"{cleanup_error}"
                    )
                raise
            body_error: BaseException | None = None
            try:
                yield
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                self._stale_operation_recovery_index_depth = 0
                self._stale_operation_recovery_index_active = False
                try:
                    # DROP (rather than merely DELETE) also recovers from a
                    # partially populated or malformed temporary snapshot.
                    self._execute(f"DROP TABLE IF EXISTS {table}")
                except BaseException as cleanup_error:
                    self._poison(
                        "stale operation recovery cleanup failed: "
                        f"{cleanup_error}"
                    )
                    if body_error is not None:
                        body_error.add_note(
                            "stale operation recovery cleanup also failed: "
                            f"{cleanup_error}"
                        )
                    else:
                        raise ValidationError(
                            "stale operation recovery cleanup failed; store is unusable"
                        ) from cleanup_error

    def _populate_stale_operation_recovery_index(self, table: str) -> None:
        self._execute(
            f"""
            WITH RECURSIVE
            uncertain_effects(effect_id) AS MATERIALIZED (
                SELECT effect_id
                  FROM external_effects
                 WHERE effect_state = ?
                UNION
                SELECT effect_id
                  FROM external_effects
                 WHERE transaction_state = ?
            ),
            unknown_nodes(
                root_operation_id,
                operation_id,
                parent_operation_id
            ) AS MATERIALIZED (
                SELECT DISTINCT operation.root_operation_id,
                       operation.operation_id,
                       operation.parent_operation_id
                  FROM uncertain_effects
                  CROSS JOIN operation_evidence AS evidence
                  CROSS JOIN operations AS operation
                 WHERE evidence.evidence_type = 'external_effect'
                   AND evidence.evidence_id = uncertain_effects.effect_id
                   AND operation.operation_id = evidence.operation_id
            ),
            ancestors(
                root_operation_id,
                operation_id,
                parent_operation_id
            ) AS (
                SELECT root_operation_id,
                       operation_id,
                       parent_operation_id
                  FROM unknown_nodes
                UNION
                SELECT ancestors.root_operation_id,
                       parent.operation_id,
                       parent.parent_operation_id
                  FROM ancestors
                  JOIN operations AS parent
                    ON parent.operation_id = ancestors.parent_operation_id
                   AND parent.root_operation_id = ancestors.root_operation_id
            )
            INSERT INTO {table} (operation_id)
            SELECT DISTINCT operation.operation_id
              FROM ancestors
              JOIN operations AS operation
                ON operation.operation_id = ancestors.operation_id
             WHERE operation.state = ?
            """,
            ("pending", "unknown", OperationState.RUNNING.value),
        )

    def operation_has_unknown_external_effect(self, operation_id: str) -> bool:
        """Test one live operation subtree with indexed recursive EXISTS."""

        if not isinstance(operation_id, str) or not operation_id:
            raise ValidationError("operation recovery id must be non-empty text")
        rows = self._query(
            """
            WITH RECURSIVE
            selected(root_operation_id, operation_id) AS (
                SELECT root_operation_id, operation_id
                  FROM operations
                 WHERE operation_id = ?
            ),
            subtree(root_operation_id, operation_id) AS (
                SELECT root_operation_id, operation_id FROM selected
                UNION
                SELECT subtree.root_operation_id, child.operation_id
                  FROM subtree
                  JOIN operations AS child
                    ON child.parent_operation_id = subtree.operation_id
                   AND child.root_operation_id = subtree.root_operation_id
            )
            SELECT EXISTS (
                SELECT 1
                  FROM subtree
                 WHERE (
                    SELECT evidence.evidence_id
                      FROM operation_evidence AS evidence
                      JOIN external_effects AS effect
                        ON effect.effect_id = evidence.evidence_id
                     WHERE evidence.operation_id = subtree.operation_id
                       AND evidence.evidence_type = 'external_effect'
                       AND (
                            effect.effect_state = ?
                            OR effect.transaction_state = ?
                       )
                     LIMIT 1
                 ) IS NOT NULL
            ) AS has_unknown_external_effect
            """,
            (operation_id, "pending", "unknown"),
        )
        return bool(rows and rows[0]["has_unknown_external_effect"])

    def update_operation(
        self,
        record: OperationRecord,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        """Replace an operation with an optional state CAS predicate.

        ``None`` disables the state predicate.  A supplied but empty iterable
        matches no state and therefore rejects the mutation.
        """

        selected_states = (
            None
            if expected_states is None
            else sorted({str(value) for value in expected_states})
        )
        if selected_states is not None and not selected_states:
            return False
        runtime_publication_id = _operation_runtime_publication_id(record.metadata)
        params: list[Any] = [
            record.root_operation_id,
            record.parent_operation_id,
            record.kind.value,
            record.name,
            record.actor,
            record.pid,
            record.state.value,
            record.outcome.value,
            dumps(record.expected_roles),
            dumps(record.metadata),
            runtime_publication_id,
            record.started_at,
            record.updated_at,
            record.completed_at,
            record.operation_id,
        ]
        where = "operation_id = ?"
        if selected_states is not None:
            placeholders = ", ".join("?" for _ in selected_states)
            where += f" AND state IN ({placeholders})"
            params.extend(selected_states)
        with self._join_or_begin_transaction():
            existing = self._query(
                "SELECT runtime_publication_id FROM operations WHERE operation_id = ?",
                (record.operation_id,),
            )
            previous_publication_id = (
                str(existing[0]["runtime_publication_id"])
                if existing and existing[0]["runtime_publication_id"] is not None
                else None
            )
            self._require_operation_publication_binding_available(
                runtime_publication_id,
                operation_id=record.operation_id,
            )
            cursor = self._execute(
                f"""
                UPDATE operations
                   SET root_operation_id = ?, parent_operation_id = ?, kind = ?, name = ?,
                       actor = ?, pid = ?, state = ?, outcome = ?, expected_roles_json = ?,
                       metadata_json = ?, runtime_publication_id = ?, started_at = ?,
                       updated_at = ?, completed_at = ?
                 WHERE {where}
                """,
                params,
            )
            if cursor.rowcount != 1:
                return False

            # Operation reconciliation is an online cache of the exact durable
            # publication/operation binding.  Any successful mutation of a
            # bound operation makes that cache stale.  Invalidate both sides
            # when a repository caller moves a binding so startup can validate
            # the old publication and fill/validate the new one without ever
            # scanning settled operation history.  This runs in the caller's
            # transaction or a repository-owned transaction with the CAS.
            self._invalidate_runtime_publication_operation_reconciliation(
                (previous_publication_id, runtime_publication_id)
            )
            return True

    def insert_operation_evidence(self, link: OperationEvidenceLink) -> bool:
        cursor = self._execute(
            """
            INSERT INTO operation_evidence (
                link_id, operation_id, evidence_type, evidence_id, role, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(operation_id, evidence_type, evidence_id, role) DO NOTHING
            """,
            (
                link.link_id,
                link.operation_id,
                link.evidence_type,
                link.evidence_id,
                link.role,
                link.created_at,
                dumps(link.metadata),
            ),
        )
        return cursor.rowcount == 1

    def list_operation_evidence(
        self,
        *,
        operation_ids: Iterable[str] | None = None,
        evidence_types: Iterable[str] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[OperationEvidenceLink]:
        clauses: list[str] = []
        params: list[Any] = []
        selected_operations = sorted({str(value) for value in operation_ids or []})
        if operation_ids is not None:
            if not selected_operations:
                return []
            placeholders = ", ".join("?" for _ in selected_operations)
            clauses.append(f"operation_id IN ({placeholders})")
            params.extend(selected_operations)
        selected_types = sorted({str(value) for value in evidence_types or []})
        if evidence_types is not None:
            if not selected_types:
                return []
            placeholders = ", ".join("?" for _ in selected_types)
            clauses.append(f"evidence_type IN ({placeholders})")
            params.extend(selected_types)
        if evidence_id is not None:
            clauses.append("evidence_id = ?")
            params.append(evidence_id)
        if cursor is not None:
            cursor_rows = self._query("SELECT * FROM operation_evidence WHERE link_id = ?", (cursor,))
            if not cursor_rows:
                return []
            cursor_row = cursor_rows[0]
            clauses.append("(created_at, link_id) > (?, ?)")
            params.extend((cursor_row["created_at"], cursor_row["link_id"]))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM operation_evidence{where} ORDER BY created_at, link_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [self._row_to_operation_evidence(row) for row in self._query(sql, params)]

    def insert_context_materialization_manifest(self, manifest: ContextMaterializationManifest) -> None:
        self._execute(
            """
            INSERT INTO context_materialization_manifests (
                materialization_id, pid, view_id, policy, budget_tokens, rendered_tokens,
                rendered_sha256, context_generation, context_oid, context_version,
                objects_json, compaction_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.materialization_id,
                manifest.pid,
                manifest.view_id,
                manifest.policy,
                manifest.budget_tokens,
                manifest.rendered_tokens,
                manifest.rendered_sha256,
                manifest.context_generation,
                manifest.context_oid,
                manifest.context_version,
                dumps(manifest.objects),
                dumps(manifest.compaction),
                manifest.created_at,
            ),
        )

    def get_context_materialization_manifest(
        self,
        materialization_id: str,
    ) -> ContextMaterializationManifest | None:
        rows = self._query(
            "SELECT * FROM context_materialization_manifests WHERE materialization_id = ?",
            (materialization_id,),
        )
        return self._row_to_context_materialization_manifest(rows[0]) if rows else None

    def list_context_materialization_manifests(
        self,
        *,
        pid: str | None = None,
        limit: int | None = None,
    ) -> list[ContextMaterializationManifest]:
        params: list[Any] = []
        where = ""
        if pid is not None:
            where = " WHERE pid = ?"
            params.append(pid)
        sql = (
            "SELECT * FROM context_materialization_manifests"
            f"{where} ORDER BY created_at DESC, materialization_id DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [self._row_to_context_materialization_manifest(row) for row in self._query(sql, params)]

    def insert_external_effect(self, record: ExternalEffectRecord) -> None:
        if (
            external_effect_payload_retention_tier(record)
            is not PayloadRetentionTier.FULL
        ):
            raise ValidationError(
                "new external effect records must contain full provider payloads"
            )
        occurred_at = record.updated_at or record.created_at
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO external_effects (
                    effect_id, record_id, event_id, pid, provider, operation, target,
                    rollback_class, rollback_status, state_mutation, information_flow,
                    provider_metadata_json, created_at, effect_state, transaction_state,
                    canonical_args_hash, idempotency_key, provider_receipt_json, updated_at,
                    payload_retention_schema_version, payload_retention_tier,
                    payload_retention_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.effect_id,
                    record.record_id,
                    record.event_id,
                    record.pid,
                    record.provider,
                    record.operation,
                    record.target,
                    record.rollback_class.value,
                    record.rollback_status.value,
                    int(record.state_mutation),
                    int(record.information_flow),
                    dumps(record.provider_metadata),
                    record.created_at,
                    record.effect_state,
                    record.transaction_state,
                    record.canonical_args_hash,
                    record.idempotency_key,
                    dumps(record.provider_receipt),
                    occurred_at,
                    record.payload_retention_schema_version,
                    record.payload_retention_tier,
                    record.payload_retention_sha256,
                ),
            )
            self._append_external_effect_transition(
                cur,
                effect_id=record.effect_id,
                effect_state=record.effect_state,
                transaction_state=record.transaction_state,
                occurred_at=occurred_at,
            )

    def finalize_external_effect(self, intent_effect_id: str, record: ExternalEffectRecord) -> bool:
        if record.effect_id != intent_effect_id:
            raise ValidationError("external effect finalization record id must match intent id")
        if record.effect_state != "finalized":
            raise ValidationError("external effect finalization record must be finalized")
        if (
            external_effect_payload_retention_tier(record)
            is not PayloadRetentionTier.FULL
        ):
            raise ValidationError(
                "external effect finalization must contain full provider payloads"
            )
        occurred_at = record.updated_at or record.created_at
        with self.transaction() as cur:
            cursor = cur.execute(
                """
                UPDATE external_effects
               SET record_id = ?, event_id = ?, rollback_class = ?, rollback_status = ?,
                   state_mutation = ?, information_flow = ?, provider_metadata_json = ?, created_at = ?,
                   effect_state = ?, transaction_state = ?, canonical_args_hash = ?,
                   idempotency_key = ?, provider_receipt_json = ?, updated_at = ?,
                   payload_retention_schema_version = ?, payload_retention_tier = ?,
                   payload_retention_sha256 = ?
             WHERE effect_id = ?
               AND pid = ?
               AND provider = ?
               AND operation = ?
               AND ((target IS NULL AND CAST(? AS TEXT) IS NULL) OR target = ?)
               AND record_id IS NULL
               AND event_id IS NULL
               AND rollback_class = ?
               AND rollback_status = ?
               AND effect_state = 'pending'
                """,
                (
                record.record_id,
                record.event_id,
                record.rollback_class.value,
                record.rollback_status.value,
                int(record.state_mutation),
                int(record.information_flow),
                dumps(record.provider_metadata),
                record.created_at,
                record.effect_state,
                record.transaction_state,
                record.canonical_args_hash,
                record.idempotency_key,
                dumps(record.provider_receipt),
                record.updated_at or record.created_at,
                record.payload_retention_schema_version,
                record.payload_retention_tier,
                record.payload_retention_sha256,
                intent_effect_id,
                record.pid,
                record.provider,
                record.operation,
                record.target,
                record.target,
                ExternalEffectRollbackClass.UNKNOWN.value,
                ExternalEffectRollbackStatus.UNKNOWN.value,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._append_external_effect_transition(
                cur,
                effect_id=record.effect_id,
                effect_state=record.effect_state,
                transaction_state=record.transaction_state,
                occurred_at=occurred_at,
            )
            return True

    def transition_external_effect(
        self,
        effect_id: str,
        *,
        expected_states: Iterable[str],
        transaction_state: str,
        provider_metadata: dict[str, Any] | None = None,
        provider_receipt: dict[str, Any] | None = None,
        updated_at: str,
    ) -> bool:
        states = list(dict.fromkeys(str(state) for state in expected_states))
        if not states:
            raise ValidationError("external effect transition requires expected states")
        placeholders = ", ".join("?" for _ in states)
        with self.transaction() as cur:
            rows = list(cur.execute("SELECT * FROM external_effects WHERE effect_id = ?", (effect_id,)))
            if not rows:
                return False
            current = self._row_to_external_effect(rows[0])
            if current.transaction_state not in states:
                return False
            if (
                external_effect_payload_retention_tier(current)
                is not PayloadRetentionTier.FULL
            ):
                raise ValidationError(
                    "retained external effect payloads cannot be transitioned"
                )
            cursor = cur.execute(
                f"""
                UPDATE external_effects
                   SET transaction_state = ?, provider_metadata_json = ?,
                       provider_receipt_json = ?, updated_at = ?
                 WHERE effect_id = ? AND transaction_state IN ({placeholders})
                """,
                (
                    transaction_state,
                    dumps(provider_metadata if provider_metadata is not None else current.provider_metadata),
                    dumps(provider_receipt if provider_receipt is not None else current.provider_receipt),
                    updated_at,
                    effect_id,
                    *states,
                ),
            )
            if cursor.rowcount != 1:
                return False
            self._append_external_effect_transition(
                cur,
                effect_id=effect_id,
                effect_state=current.effect_state,
                transaction_state=transaction_state,
                occurred_at=updated_at,
            )
            return True

    def current_effect_ledger_seq(self) -> int:
        rows = self._query(
            "SELECT value FROM runtime_counters WHERE counter_name = ?",
            ("external_effect_ledger",),
        )
        if not rows:
            raise ValidationError("external effect ledger counter is missing")
        return int(rows[0]["value"])

    def list_external_effects_changed_after(
        self,
        effect_ledger_seq: int,
        *,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]:
        """Pin changed effect ids, then read one consistent current-row batch."""

        with self.transaction() as cur:
            id_rows = list(
                cur.execute(
                    """
                    SELECT DISTINCT effect_id
                      FROM external_effect_transitions
                     WHERE seq > ?
                     ORDER BY effect_id
                    """,
                    (int(effect_ledger_seq),),
                )
            )
            effect_ids = [str(row["effect_id"]) for row in id_rows]
            if not effect_ids:
                return []
            clauses = [f"effect_id IN ({', '.join('?' for _ in effect_ids)})"]
            params: list[Any] = list(effect_ids)
            if pids is not None:
                selected_pids = list(dict.fromkeys(str(pid) for pid in pids))
                if not selected_pids:
                    return []
                clauses.append(f"pid IN ({', '.join('?' for _ in selected_pids)})")
                params.extend(selected_pids)
            rows = list(
                cur.execute(
                    f"SELECT * FROM external_effects WHERE {' AND '.join(clauses)} "
                    "ORDER BY created_at, effect_id",
                    params,
                )
            )
            return [self._row_to_external_effect(row) for row in rows]

    def _append_external_effect_transition(
        self,
        cur: Any,
        *,
        effect_id: str,
        effect_state: str,
        transaction_state: str,
        occurred_at: str,
    ) -> int:
        updated = cur.execute(
            """
            UPDATE runtime_counters
               SET value = value + 1
             WHERE counter_name = ?
            """,
            ("external_effect_ledger",),
        )
        if updated.rowcount != 1:
            raise ValidationError("external effect ledger counter is missing")
        rows = list(
            cur.execute(
                "SELECT value FROM runtime_counters WHERE counter_name = ?",
                ("external_effect_ledger",),
            )
        )
        seq = int(rows[0]["value"])
        cur.execute(
            """
            INSERT INTO external_effect_transitions (
                seq, effect_id, effect_state, transaction_state, occurred_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (seq, effect_id, effect_state, transaction_state, occurred_at),
        )
        return seq

    def abandon_external_effect_intent(self, effect_id: str) -> bool:
        cursor = self._execute(
            """
            DELETE FROM external_effects
             WHERE effect_id = ?
               AND record_id IS NULL
               AND event_id IS NULL
               AND rollback_class = ?
               AND rollback_status = ?
               AND effect_state = 'pending'
            """,
            (
                effect_id,
                ExternalEffectRollbackClass.UNKNOWN.value,
                ExternalEffectRollbackStatus.UNKNOWN.value,
            ),
        )
        return cursor.rowcount == 1

    def list_external_effects(
        self,
        *,
        created_after: str | None = None,
        pid: str | None = None,
        pids: Iterable[str] | None = None,
    ) -> list[ExternalEffectRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if created_after is not None:
            clauses.append("created_at > ?")
            params.append(created_after)
        if pid is not None:
            clauses.append("pid = ?")
            params.append(pid)
        if pids is not None:
            selected_pids = list(dict.fromkeys(str(item) for item in pids))
            if not selected_pids:
                return []
            placeholders = ", ".join("?" for _ in selected_pids)
            clauses.append(f"pid IN ({placeholders})")
            params.extend(selected_pids)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM external_effects{where} ORDER BY created_at, effect_id",
            params,
        )
        return [self._row_to_external_effect(row) for row in rows]

    def query_external_effect_recovery(
        self,
        query: ExternalEffectRecoveryQuery,
    ) -> ExternalEffectPage:
        hard_limit = self.config.runtime.external_effect_recovery_page_hard_limit
        if query.limit > hard_limit:
            raise ValidationError(
                "external effect recovery page limit exceeds configured hard limit: "
                f"{query.limit} > {hard_limit}"
            )
        clauses = ["effect_state = ?"]
        params: list[Any] = [query.effect_state]
        if query.transaction_states:
            placeholders = ", ".join("?" for _ in query.transaction_states)
            clauses.append(f"transaction_state IN ({placeholders})")
            params.extend(query.transaction_states)
        if query.after is not None:
            clauses.append("(created_at, effect_id) > (?, ?)")
            params.extend(
                (
                    query.after.created_at,
                    query.after.effect_id,
                )
            )
        params.append(query.limit + 1)
        rows = self._query(
            f"""
            SELECT * FROM external_effects
             WHERE {' AND '.join(clauses)}
             ORDER BY created_at, effect_id
             LIMIT ?
            """,
            params,
        )
        selected_rows = rows[: query.limit]
        records = tuple(
            self._row_to_external_effect(row) for row in selected_rows
        )
        next_cursor = None
        if len(rows) > query.limit and records:
            last = records[-1]
            next_cursor = ExternalEffectCursor(last.created_at, last.effect_id)
        return ExternalEffectPage(records=records, next_cursor=next_cursor)

    def get_external_effect_by_idempotency(
        self,
        pid: str,
        idempotency_key: str,
    ) -> ExternalEffectRecord | None:
        rows = self._query(
            """
            SELECT * FROM external_effects
             WHERE pid = ? AND idempotency_key = ?
            """,
            (pid, idempotency_key),
        )
        if len(rows) > 1:
            raise ValidationError(
                "external effect idempotency index returned duplicate rows"
            )
        return self._row_to_external_effect(rows[0]) if rows else None

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None:
        rows = self._query("SELECT * FROM external_effects WHERE effect_id = ?", (effect_id,))
        return self._row_to_external_effect(rows[0]) if rows else None

    def scan_external_effect_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[ExternalEffectRecord]:
        """Read one bounded external-effect retention page by stable keyset."""

        selected_limit = self._payload_retention_limit(limit)
        if not isinstance(older_than, str) or not older_than:
            raise ValidationError("payload retention older_than must not be empty")
        clauses = [
            "effect_state = 'finalized'",
            "transaction_state IN ('committed', 'failed', 'compensated')",
            "payload_retention_tier IN ('full', 'summary')",
            "created_at <= ?",
        ]
        params: list[Any] = [older_than]
        if after is not None:
            clauses.append("(created_at, effect_id) > (?, ?)")
            params.extend((after.created_at, after.record_id))
        params.append(selected_limit + 1)
        rows = self._query(
            f"""
            SELECT source.*
              FROM external_effects AS source
              JOIN (
                    SELECT effect_id, created_at
                      FROM external_effects INDEXED BY idx_external_effects_retention_eligible
                     WHERE {' AND '.join(clauses)}
                     ORDER BY created_at, effect_id
                     LIMIT ?
                   ) AS candidates
                ON candidates.effect_id = source.effect_id
             ORDER BY candidates.created_at, candidates.effect_id
            """,
            params,
        )
        records = tuple(
            self._row_to_external_effect(row) for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = PayloadRetentionCursor(last.created_at, last.effect_id)
        return PayloadRetentionPage(records=records, next_cursor=next_cursor)

    def update_external_effect_payload_retention(
        self,
        record: ExternalEffectRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        """CAS-update only reducible payload columns of one terminal effect."""

        with self.transaction() as cur:
            rows = list(
                cur.execute(
                    "SELECT * FROM external_effects WHERE effect_id = ?",
                    (record.effect_id,),
                )
            )
            if len(rows) != 1:
                return False
            current = self._row_to_external_effect(rows[0])
            try:
                validate_external_effect_payload_retention_update(
                    current,
                    record,
                    expected_payload_sha256=expected_payload_sha256,
                    expected_tier=expected_tier,
                    expected_effect_state=expected_effect_state,
                    expected_transaction_state=expected_transaction_state,
                )
            except (TypeError, ValueError):
                return False
            updated = cur.execute(
                """
                UPDATE external_effects
                   SET provider_metadata_json = ?, provider_receipt_json = ?,
                       payload_retention_schema_version = ?,
                       payload_retention_tier = ?, payload_retention_sha256 = ?
                 WHERE effect_id = ?
                   AND effect_state = ?
                   AND transaction_state = ?
                   AND payload_retention_schema_version = ?
                   AND payload_retention_tier = ?
                   AND (
                        (payload_retention_sha256 IS NULL AND CAST(? AS TEXT) IS NULL)
                        OR payload_retention_sha256 = ?
                   )
                   AND provider_metadata_json = ?
                   AND provider_receipt_json = ?
                """,
                (
                    dumps(record.provider_metadata),
                    dumps(record.provider_receipt),
                    record.payload_retention_schema_version,
                    record.payload_retention_tier,
                    record.payload_retention_sha256,
                    record.effect_id,
                    expected_effect_state,
                    expected_transaction_state,
                    current.payload_retention_schema_version,
                    current.payload_retention_tier,
                    current.payload_retention_sha256,
                    current.payload_retention_sha256,
                    dumps(current.provider_metadata),
                    dumps(current.provider_receipt),
                ),
            )
            return updated.rowcount == 1

    def insert_human_request(self, request: HumanRequest) -> None:
        self._execute(
            "INSERT INTO human_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.request_id,
                request.pid,
                request.human,
                dumps(request.payload),
                request.status.value,
                dumps(request.decision) if request.decision is not None else None,
                int(request.blocking),
                request.created_at,
                request.updated_at,
            ),
        )

    def update_human_request(self, request: HumanRequest) -> None:
        self._execute(
            """
            UPDATE human_requests
               SET pid = ?, human = ?, payload_json = ?, status = ?, decision_json = ?,
                   blocking = ?, created_at = ?, updated_at = ?
             WHERE request_id = ?
            """,
            (
                request.pid,
                request.human,
                dumps(request.payload),
                request.status.value,
                dumps(request.decision) if request.decision is not None else None,
                int(request.blocking),
                request.created_at,
                request.updated_at,
                request.request_id,
            ),
        )

    def get_human_request(self, request_id: str) -> HumanRequest | None:
        rows = self._query("SELECT * FROM human_requests WHERE request_id = ?", (request_id,))
        return self._row_to_human_request(rows[0]) if rows else None

    def list_human_requests(
        self,
        pid: str | None = None,
        *,
        human: str | None = None,
        status: HumanRequestStatus | str | None = None,
        limit: int | None = None,
        newest: bool = False,
    ) -> list[HumanRequest]:
        clauses: list[str] = []
        params: list[Any] = []
        if pid is not None:
            clauses.append("pid = ?")
            params.append(pid)
        if human is not None:
            clauses.append("human = ?")
            params.append(human)
        if status is not None:
            selected_status = status.value if isinstance(status, HumanRequestStatus) else str(status)
            clauses.append("status = ?")
            params.append(selected_status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "DESC" if newest else "ASC"
        sql = f"SELECT * FROM human_requests{where} ORDER BY created_at {direction}, request_id {direction}"
        if limit is not None:
            selected_limit = max(0, int(limit))
            sql += " LIMIT ?"
            params.append(selected_limit)
        rows = self._query(sql, params)
        return [self._row_to_human_request(row) for row in rows]

    def insert_llm_call(self, record: LLMCallRecord) -> None:
        retention_tier = llm_call_payload_retention_tier(record)
        self._execute(
            """
            INSERT INTO llm_calls (
                call_id, pid, image_id, purpose, status, api, model, request_id, response_id,
                messages_json, tools_json, request_options_json, response_content, tool_calls_json,
                reasoning_json, usage_json, raw_response_json, observability_json, error, created_at,
                completed_at, payload_retention_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.call_id,
                record.pid,
                record.image_id,
                record.purpose,
                record.status,
                record.api,
                record.model,
                record.request_id,
                record.response_id,
                dumps(record.messages),
                dumps(record.tools),
                dumps(record.request_options),
                record.response_content,
                dumps(record.tool_calls),
                dumps(record.reasoning) if record.reasoning is not None else None,
                dumps(record.usage),
                dumps(record.raw_response) if record.raw_response is not None else None,
                dumps(record.observability),
                record.error,
                record.created_at,
                record.completed_at,
                retention_tier.value,
            ),
        )

    def list_llm_calls(self, pid: str | None = None, limit: int | None = None) -> list[LLMCallRecord]:
        selected_limit = self._llm_call_limit(limit)
        params: list[Any] = []
        sql = "SELECT * FROM (SELECT * FROM llm_calls"
        if pid is not None:
            sql += " WHERE pid = ?"
            params.append(pid)
        sql += " ORDER BY created_at DESC, call_id DESC LIMIT ?"
        params.append(selected_limit)
        sql += ") AS latest_llm_calls ORDER BY created_at ASC, call_id ASC"
        return [self._row_to_llm_call(row) for row in self._query(sql, params)]

    def get_llm_call(self, call_id: str) -> LLMCallRecord | None:
        rows = self._query("SELECT * FROM llm_calls WHERE call_id = ?", (call_id,))
        return self._row_to_llm_call(rows[0]) if rows else None

    def scan_llm_call_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[LLMCallRecord]:
        """Read one bounded LLM-call retention page by stable keyset."""

        selected_limit = self._payload_retention_limit(limit)
        if not isinstance(older_than, str) or not older_than:
            raise ValidationError("payload retention older_than must not be empty")
        clauses = [
            "status IN ('ok', 'error')",
            "completed_at IS NOT NULL",
            "payload_retention_tier IN ('full', 'summary')",
            "created_at <= ?",
        ]
        params: list[Any] = [older_than]
        if after is not None:
            clauses.append("(created_at, call_id) > (?, ?)")
            params.extend((after.created_at, after.record_id))
        params.append(selected_limit + 1)
        rows = self._query(
            f"""
            SELECT source.*,
                   CASE
                     WHEN source.pid IS NOT NULL
                      AND source.status = 'ok'
                      AND source.api = 'responses'
                      AND source.response_id IS NOT NULL
                      AND source.response_id <> ''
                      AND NOT EXISTS (
                       SELECT 1
                         FROM llm_calls AS newer
                              INDEXED BY idx_llm_calls_provider_chain_head
                        WHERE newer.pid = source.pid
                          AND newer.purpose = source.purpose
                          AND (
                            newer.created_at COLLATE BINARY,
                            newer.call_id COLLATE BINARY
                          ) > (
                            source.created_at COLLATE BINARY,
                            source.call_id COLLATE BINARY
                          )
                     ) THEN 1
                     ELSE 0
                   END AS payload_retention_is_latest_llm_call
              FROM llm_calls AS source
              JOIN (
                    SELECT call_id, created_at
                      FROM llm_calls INDEXED BY idx_llm_calls_retention_eligible
                     WHERE {' AND '.join(clauses)}
                     ORDER BY created_at, call_id
                     LIMIT ?
                   ) AS candidates
                ON candidates.call_id = source.call_id
             ORDER BY candidates.created_at, candidates.call_id
            """,
            params,
        )
        records = tuple(
            self._row_to_llm_call(row) for row in rows[:selected_limit]
        )
        latest_llm_call_ids = frozenset(
            str(row["call_id"])
            for row in rows[:selected_limit]
            if bool(row["payload_retention_is_latest_llm_call"])
        )
        next_cursor = None
        if len(rows) > selected_limit and records:
            last = records[-1]
            next_cursor = PayloadRetentionCursor(last.created_at, last.call_id)
        return PayloadRetentionPage(
            records=records,
            next_cursor=next_cursor,
            latest_llm_call_ids=latest_llm_call_ids,
        )

    def update_llm_call_payload_retention(
        self,
        record: LLMCallRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
    ) -> bool:
        """CAS-update only payload-bearing columns of one terminal LLM call."""

        with self.transaction() as cur:
            rows = list(
                cur.execute(
                    "SELECT * FROM llm_calls WHERE call_id = ?",
                    (record.call_id,),
                )
            )
            if len(rows) != 1:
                return False
            current = self._row_to_llm_call(rows[0])
            try:
                target_tier = validate_llm_call_payload_retention_update(
                    current,
                    record,
                    expected_payload_sha256=expected_payload_sha256,
                    expected_tier=expected_tier,
                    provider_chain_head=False,
                )
            except (TypeError, ValueError):
                return False
            updated = cur.execute(
                """
                UPDATE llm_calls
                   SET messages_json = ?, tools_json = ?, response_content = ?,
                       tool_calls_json = ?, reasoning_json = ?, raw_response_json = ?,
                       observability_json = ?, error = ?, payload_retention_tier = ?
                 WHERE call_id = ? AND status = ? AND completed_at = ?
                   AND payload_retention_tier = ?
                   AND messages_json = ? AND tools_json = ?
                   AND response_content = ? AND tool_calls_json = ?
                   AND (
                        (reasoning_json IS NULL AND CAST(? AS TEXT) IS NULL)
                        OR reasoning_json = ?
                   )
                   AND (
                        (raw_response_json IS NULL AND CAST(? AS TEXT) IS NULL)
                        OR raw_response_json = ?
                   )
                   AND observability_json = ?
                   AND (
                        (error IS NULL AND CAST(? AS TEXT) IS NULL)
                        OR error = ?
                   )
                   AND (
                        ? = 0 OR EXISTS (
                          SELECT 1
                            FROM llm_calls AS newer
                                 INDEXED BY idx_llm_calls_provider_chain_head
                           WHERE newer.pid = llm_calls.pid
                             AND newer.purpose = llm_calls.purpose
                             AND (
                               newer.created_at COLLATE BINARY,
                               newer.call_id COLLATE BINARY
                             ) > (
                               llm_calls.created_at COLLATE BINARY,
                               llm_calls.call_id COLLATE BINARY
                             )
                        )
                   )
                """,
                (
                    dumps(record.messages),
                    dumps(record.tools),
                    record.response_content,
                    dumps(record.tool_calls),
                    dumps(record.reasoning) if record.reasoning is not None else None,
                    dumps(record.raw_response) if record.raw_response is not None else None,
                    dumps(record.observability),
                    record.error,
                    target_tier.value,
                    record.call_id,
                    current.status,
                    current.completed_at,
                    expected_tier.value,
                    dumps(current.messages),
                    dumps(current.tools),
                    current.response_content,
                    dumps(current.tool_calls),
                    dumps(current.reasoning) if current.reasoning is not None else None,
                    dumps(current.reasoning) if current.reasoning is not None else None,
                    dumps(current.raw_response) if current.raw_response is not None else None,
                    dumps(current.raw_response) if current.raw_response is not None else None,
                    dumps(current.observability),
                    current.error,
                    current.error,
                    int(llm_call_payload_can_be_provider_chain_head(current)),
                ),
            )
            return updated.rowcount == 1

    def get_latest_llm_call(self, *, pid: str, purpose: str | None = None) -> LLMCallRecord | None:
        params: list[Any] = [pid]
        sql = "SELECT * FROM llm_calls WHERE pid = ?"
        if purpose is not None:
            sql += " AND purpose = ?"
            params.append(purpose)
        sql += " ORDER BY created_at DESC, call_id DESC LIMIT 1"
        rows = self._query(sql, params)
        return self._row_to_llm_call(rows[0]) if rows else None

    def upsert_llm_tool_output(
        self,
        *,
        pid: str,
        response_id: str,
        call_id: str,
        tool_name: str | None,
        output: str,
    ) -> None:
        now = utc_now()
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO llm_tool_outputs (
                    pid, response_id, call_id, tool_name, output_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pid, response_id, call_id) DO NOTHING
                """,
                (pid, response_id, call_id, tool_name, output, now, now),
            )
            if cursor.rowcount == 1:
                return
            cursor.execute(
                """
                SELECT tool_name, output_text
                  FROM llm_tool_outputs
                 WHERE pid = ? AND response_id = ? AND call_id = ?
                """,
                (pid, response_id, call_id),
            )
            existing = cursor.fetchone()
            if existing is None or existing["tool_name"] != tool_name or existing["output_text"] != output:
                raise ValidationError(
                    "conflicting durable LLM tool output for "
                    f"pid={pid} response_id={response_id} call_id={call_id}"
                )

    def list_llm_tool_outputs(self, *, pid: str, response_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            """
            SELECT pid, response_id, call_id, tool_name, output_text, created_at, updated_at
              FROM llm_tool_outputs
             WHERE pid = ? AND response_id = ?
             ORDER BY created_at, call_id
            """,
            (pid, response_id),
        )
        return [dict(row) for row in rows]

    def get_llm_context_generation(self, pid: str) -> str:
        rows = self._query(
            "SELECT generation FROM llm_context_generations WHERE pid = ?",
            (pid,),
        )
        return str(rows[0]["generation"]) if rows else "initial"

    def set_llm_context_generation(self, pid: str, generation: str) -> None:
        self._execute(
            """
            INSERT INTO llm_context_generations (
                pid, generation, labels_schema_version, labels_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pid) DO UPDATE SET
                generation = excluded.generation,
                updated_at = excluded.updated_at
            """,
            (
                pid,
                generation,
                _LLM_CONTEXT_LABEL_SCHEMA_VERSION,
                dumps(DataLabels()),
                utc_now(),
            ),
        )

    def get_llm_context_label_history(self, pid: str) -> DataLabels | None:
        rows = self._query(
            """
            SELECT labels_schema_version, labels_json
              FROM llm_context_generations
             WHERE pid = ?
            """,
            (pid,),
        )
        if not rows:
            return None
        return self._decode_llm_context_label_history(rows[0])

    def merge_llm_context_label_history(
        self,
        pid: str,
        labels: DataLabels,
    ) -> DataLabels:
        if not isinstance(labels, DataLabels):
            raise ValidationError("LLM context label history requires DataLabels")
        with self.transaction() as cur:
            row = cur.execute(
                """
                SELECT labels_schema_version, labels_json
                  FROM llm_context_generations
                 WHERE pid = ?
                """,
                (pid,),
            ).fetchone()
            current = self._decode_llm_context_label_history(row) if row is not None else None
            merged = DataLabels.aggregate(
                (labels,) if current is None else (current, labels)
            )
            cur.execute(
                """
                INSERT INTO llm_context_generations (
                    pid, generation, labels_schema_version, labels_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pid) DO UPDATE SET
                    labels_schema_version = excluded.labels_schema_version,
                    labels_json = excluded.labels_json,
                    updated_at = excluded.updated_at
                """,
                (
                    pid,
                    "initial",
                    _LLM_CONTEXT_LABEL_SCHEMA_VERSION,
                    dumps(merged),
                    utc_now(),
                ),
            )
        return merged

    def _decode_llm_context_label_history(self, row: Any) -> DataLabels:
        with _persisted_model_decode("LLM context label history"):
            version = row["labels_schema_version"]
            if version != _LLM_CONTEXT_LABEL_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported schema version "
                    f"{version!r}; expected {_LLM_CONTEXT_LABEL_SCHEMA_VERSION}"
                )
            value = loads(row["labels_json"])
            return DataLabels.from_dict(value)

    def upsert_llm_pending_action(self, pid: str, pending: dict[str, Any]) -> None:
        now = utc_now()
        created_at = str(pending.get("created_at") or now)
        resume_token = str(pending.get("resume_token") or new_id("llmwait"))
        data_flow_context = _canonical_pending_data_flow_context(
            pending.get("data_flow_context")
        )
        self._execute(
            """
            INSERT INTO llm_pending_actions (
                pid, resume_token, llm_operation_id, tool_operation_id,
                wait_type, request_id, child_pid, response_id, tool_call_id, tool_name,
                filters_json, action_json, data_flow_context_json,
                content_preview, tool_call_count, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pid) DO UPDATE SET
                resume_token = excluded.resume_token,
                llm_operation_id = excluded.llm_operation_id,
                tool_operation_id = excluded.tool_operation_id,
                wait_type = excluded.wait_type,
                request_id = excluded.request_id,
                child_pid = excluded.child_pid,
                response_id = excluded.response_id,
                tool_call_id = excluded.tool_call_id,
                tool_name = excluded.tool_name,
                filters_json = excluded.filters_json,
                action_json = excluded.action_json,
                data_flow_context_json = excluded.data_flow_context_json,
                content_preview = excluded.content_preview,
                tool_call_count = excluded.tool_call_count,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                pid,
                resume_token,
                pending.get("llm_operation_id"),
                pending.get("tool_operation_id"),
                str(pending["wait_type"]),
                pending.get("request_id"),
                pending.get("child_pid"),
                pending.get("response_id"),
                pending.get("tool_call_id"),
                pending.get("tool_name"),
                dumps(pending.get("filters") or {}),
                dumps(pending.get("action") or {}),
                dumps(data_flow_context),
                str(pending.get("content_preview") or ""),
                int(pending.get("tool_call_count") or 0),
                str(pending.get("status") or "pending"),
                created_at,
                now,
            ),
        )

    def get_llm_pending_action(self, pid: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM llm_pending_actions WHERE pid = ?", (pid,))
        return self._row_to_llm_pending_action(rows[0]) if rows else None

    def list_llm_pending_actions(self, *, status: str | None = "pending") -> list[dict[str, Any]]:
        if status is None:
            rows = self._query("SELECT * FROM llm_pending_actions ORDER BY updated_at, pid")
        else:
            rows = self._query("SELECT * FROM llm_pending_actions WHERE status = ? ORDER BY updated_at, pid", (status,))
        return [self._row_to_llm_pending_action(row) for row in rows]

    def claim_llm_pending_action(self, pid: str, *, resume_token: str) -> dict[str, Any] | None:
        with self._join_or_begin_transaction() as cur:
            updated = cur.execute(
                """
                UPDATE llm_pending_actions
                   SET status = ?, updated_at = ?
                 WHERE pid = ? AND status = ? AND resume_token = ?
                """,
                ("resuming", utc_now(), pid, "pending", resume_token),
            )
            if updated.rowcount != 1:
                return None
            row = cur.execute("SELECT * FROM llm_pending_actions WHERE pid = ?", (pid,)).fetchone()
            return self._row_to_llm_pending_action(row) if row is not None else None

    def complete_llm_pending_action(self, pid: str, *, resume_token: str) -> bool:
        cursor = self._execute(
            """
            UPDATE llm_pending_actions
               SET status = ?, updated_at = ?
             WHERE pid = ? AND status = ? AND resume_token = ?
            """,
            ("completed", utc_now(), pid, "resuming", resume_token),
        )
        return cursor.rowcount == 1

    def insert_process_message(self, message: ProcessMessage) -> None:
        metadata = _canonical_process_message_metadata(message.metadata)
        self._execute(
            """
            INSERT INTO process_messages (
                message_id, sender, recipient_pid, kind, channel, correlation_id, reply_to,
                subject, body, payload_json, metadata_json, status, created_at, updated_at, acked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.message_id,
                message.sender,
                message.recipient_pid,
                message.kind.value,
                message.channel,
                message.correlation_id,
                message.reply_to,
                message.subject,
                message.body,
                dumps(message.payload),
                dumps(metadata),
                message.status.value,
                message.created_at,
                message.updated_at,
                message.acked_at,
            ),
        )

    def update_process_message(self, message: ProcessMessage) -> None:
        metadata = _canonical_process_message_metadata(message.metadata)
        self._execute(
            """
            UPDATE process_messages
               SET sender = ?, recipient_pid = ?, kind = ?, subject = ?, body = ?,
                   channel = ?, correlation_id = ?, reply_to = ?, payload_json = ?,
                   metadata_json = ?, status = ?, created_at = ?, updated_at = ?, acked_at = ?
             WHERE message_id = ?
            """,
            (
                message.sender,
                message.recipient_pid,
                message.kind.value,
                message.subject,
                message.body,
                message.channel,
                message.correlation_id,
                message.reply_to,
                dumps(message.payload),
                dumps(metadata),
                message.status.value,
                message.created_at,
                message.updated_at,
                message.acked_at,
                message.message_id,
            ),
        )

    def update_process_message_metadata(
        self,
        message_id: str,
        *,
        recipient_pid: str,
        expected_metadata: dict[str, Any],
        metadata: dict[str, Any],
        updated_at: str,
    ) -> bool:
        selected_metadata = _canonical_process_message_metadata(metadata)
        cursor = self._execute(
            """
            UPDATE process_messages
               SET metadata_json = ?, updated_at = ?
             WHERE message_id = ? AND recipient_pid = ? AND metadata_json = ?
            """,
            (
                dumps(selected_metadata),
                updated_at,
                message_id,
                recipient_pid,
                dumps(expected_metadata),
            ),
        )
        return cursor.rowcount == 1

    def get_process_message(self, message_id: str) -> ProcessMessage | None:
        rows = self._query("SELECT * FROM process_messages WHERE message_id = ?", (message_id,))
        return self._row_to_process_message(rows[0]) if rows else None

    def list_process_messages(
        self,
        recipient_pid: str | None = None,
        *,
        status: ProcessMessageStatus | str | None = None,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ProcessMessage]:
        clauses: list[str] = []
        params: list[Any] = []
        if message_ids is not None and not message_ids:
            return []
        if recipient_pid is not None:
            clauses.append("recipient_pid = ?")
            params.append(recipient_pid)
        if status is not None:
            selected_status = ProcessMessageStatus(status)
            clauses.append("status = ?")
            params.append(selected_status.value)
        if kind is not None:
            selected_kind = ProcessMessageKind(kind)
            clauses.append("kind = ?")
            params.append(selected_kind.value)
        if sender is not None:
            clauses.append("sender = ?")
            params.append(sender)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)
        if correlation_id is not None:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        if reply_to is not None:
            clauses.append("reply_to = ?")
            params.append(reply_to)
        if message_ids is not None:
            placeholders = ", ".join("?" for _ in message_ids)
            clauses.append(f"message_id IN ({placeholders})")
            params.extend(message_ids)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self._query(f"SELECT * FROM process_messages{where} ORDER BY created_at, message_id{limit_sql}", params)
        return [self._row_to_process_message(row) for row in rows]

    def get_process_activity_summaries(
        self,
        pids: Iterable[str],
        *,
        recent_message_limit: int,
        recent_llm_call_limit: int,
    ) -> dict[str, dict[str, Any]]:
        """Return bounded GUI-oriented process activity without per-pid queries."""

        selected = sorted({str(pid) for pid in pids if str(pid)})
        if not selected:
            return {}
        selected_message_limit = max(0, int(recent_message_limit))
        selected_llm_limit = max(0, int(recent_llm_call_limit))
        placeholders = ", ".join("?" for _ in selected)
        stats_rows = self._query(
            f"""
            SELECT processes.pid,
                   (
                       SELECT COUNT(*)
                         FROM process_messages
                        WHERE process_messages.recipient_pid = processes.pid
                          AND process_messages.status = ?
                   ) AS unread_message_count,
                   (
                       SELECT COUNT(*)
                         FROM process_messages
                        WHERE process_messages.recipient_pid = processes.pid
                          AND process_messages.status = ?
                          AND process_messages.kind = ?
                   ) AS interrupt_count
              FROM processes
             WHERE processes.pid IN ({placeholders})
            """,
            [
                ProcessMessageStatus.UNREAD.value,
                ProcessMessageStatus.UNREAD.value,
                ProcessMessageKind.INTERRUPT.value,
                *selected,
            ],
        )
        summaries: dict[str, dict[str, Any]] = {
            str(row["pid"]): {
                "unread_message_count": int(row["unread_message_count"] or 0),
                "interrupt_count": int(row["interrupt_count"] or 0),
                "llm_call_count": 0,
                "token_total": 0,
                "messages": [],
            }
            for row in stats_rows
        }
        if selected_message_limit > 0:
            recent_rows = self._query(
                f"""
                SELECT *
                  FROM (
                        SELECT process_messages.*,
                               ROW_NUMBER() OVER (
                                   PARTITION BY recipient_pid
                                   ORDER BY created_at DESC, message_id DESC
                               ) AS snapshot_row_number
                          FROM process_messages
                         WHERE recipient_pid IN ({placeholders})
                       ) AS ranked_messages
                 WHERE snapshot_row_number <= ?
                 ORDER BY recipient_pid, created_at, message_id
                """,
                [*selected, selected_message_limit],
            )
            for row in recent_rows:
                pid = str(row["recipient_pid"])
                if pid in summaries:
                    summaries[pid]["messages"].append(self._row_to_process_message(row))
        if selected_llm_limit > 0:
            llm_rows = self._query(
                f"""
                SELECT pid, usage_json
                  FROM (
                        SELECT pid, usage_json,
                               ROW_NUMBER() OVER (
                                   PARTITION BY pid
                                   ORDER BY created_at DESC, call_id DESC
                               ) AS snapshot_row_number
                          FROM llm_calls
                         WHERE pid IN ({placeholders})
                       ) AS ranked_llm_calls
                 WHERE snapshot_row_number <= ?
                """,
                [*selected, selected_llm_limit],
            )
            for row in llm_rows:
                pid = str(row["pid"])
                if pid not in summaries:
                    continue
                usage = loads(row["usage_json"], {})
                summaries[pid]["llm_call_count"] += 1
                summaries[pid]["token_total"] += int(usage.get("total_tokens", 0) or 0)
        return summaries

    def insert_object_task(self, task: ObjectTask) -> None:
        self._execute(
            """
            INSERT INTO object_tasks (
                task_id, owner_oid, creator_pid, runner_pid, tool, tool_id, status,
                notification_status, notification_recipient_pid, notification_json,
                owner_watch_json, result_oid, error, wait_json, created_at,
                updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._object_task_params(task),
        )

    def update_object_task(self, task: ObjectTask) -> None:
        self._execute(
            """
            UPDATE object_tasks
                   SET owner_oid = ?, creator_pid = ?, runner_pid = ?, tool = ?, tool_id = ?,
                   status = ?, notification_status = ?, notification_recipient_pid = ?,
                   notification_json = ?, owner_watch_json = ?, result_oid = ?, error = ?, wait_json = ?,
                   created_at = ?, updated_at = ?, started_at = ?, completed_at = ?
             WHERE task_id = ?
            """,
            (
                task.owner_oid,
                task.creator_pid,
                task.runner_pid,
                task.tool,
                task.tool_id,
                task.status.value,
                task.notification.status.value,
                task.notification.recipient_pid,
                dumps(task.notification),
                dumps(task.owner_watch),
                task.result_oid,
                task.error,
                dumps(task.wait),
                task.created_at,
                task.updated_at,
                task.started_at,
                task.completed_at,
                task.task_id,
            ),
        )

    def get_object_task(self, task_id: str) -> ObjectTask | None:
        rows = self._query("SELECT * FROM object_tasks WHERE task_id = ?", (task_id,))
        return self._row_to_object_task(rows[0]) if rows else None

    def list_object_tasks(
        self,
        *,
        owner_oid: str | None = None,
        creator_pid: str | None = None,
        statuses: Iterable[str | ObjectTaskStatus] | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[ObjectTask]:
        clauses: list[str] = []
        params: list[Any] = []
        if owner_oid is not None:
            clauses.append("owner_oid = ?")
            params.append(owner_oid)
        if creator_pid is not None:
            clauses.append("creator_pid = ?")
            params.append(creator_pid)
        if statuses is not None:
            selected = [ObjectTaskStatus(status).value for status in statuses]
            if not selected:
                return []
            clauses.append(f"status IN ({', '.join('?' for _ in selected)})")
            params.extend(selected)
        elif not include_terminal:
            terminal = [
                ObjectTaskStatus.SUCCEEDED.value,
                ObjectTaskStatus.FAILED.value,
                ObjectTaskStatus.CANCELLED.value,
                ObjectTaskStatus.ABANDONED.value,
                ObjectTaskStatus.SUPERSEDED_BY_RESTORE.value,
                ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN.value,
            ]
            clauses.append(f"status NOT IN ({', '.join('?' for _ in terminal)})")
            params.extend(terminal)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self._query(f"SELECT * FROM object_tasks{where} ORDER BY updated_at DESC, created_at DESC, task_id ASC{limit_sql}", params)
        return [self._row_to_object_task(row) for row in rows]

    def query_object_task_recovery(
        self,
        *,
        kind: ObjectTaskRecoveryKind,
        after: ObjectTaskRecoveryCursor | None,
        limit: int,
    ) -> ObjectTaskRecoveryPage:
        """Return one hard-bounded keyset page for startup reconciliation."""

        selected_limit = self._object_task_recovery_limit(limit)
        if not isinstance(kind, ObjectTaskRecoveryKind):
            raise ValidationError("object task recovery kind has an invalid type")
        if after is not None and not isinstance(after, ObjectTaskRecoveryCursor):
            raise ValidationError("object task recovery cursor has an invalid type")
        clauses, params = self._object_task_recovery_filter(kind)
        recovery_index = self._object_task_recovery_index(kind)
        if after is not None:
            clauses.append("(created_at, task_id) > (?, ?)")
            params.extend((after.created_at, after.task_id))
        params.append(selected_limit + 1)
        rows = self._query(
            f"SELECT * FROM object_tasks INDEXED BY {recovery_index} "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at, task_id LIMIT ?",
            params,
        )
        records = tuple(
            self._row_to_object_task(row) for row in rows[:selected_limit]
        )
        next_cursor = None
        if len(rows) > selected_limit:
            last = records[-1]
            next_cursor = ObjectTaskRecoveryCursor(last.created_at, last.task_id)
        return ObjectTaskRecoveryPage(records=records, next_cursor=next_cursor)

    def abandon_object_task_after_reopen(
        self,
        task_id: str,
        *,
        expected_status: ObjectTaskStatus,
        reason: str,
        updated_at: str,
    ) -> ObjectTask | None:
        if not isinstance(expected_status, ObjectTaskStatus):
            raise ValidationError("object task recovery status has an invalid type")
        with self._join_or_begin_transaction() as cur:
            updated = cur.execute(
                """
                UPDATE object_tasks
                   SET status = ?, error = ?, updated_at = ?, completed_at = ?
                 WHERE task_id = ? AND status = ?
                """,
                (
                    ObjectTaskStatus.ABANDONED.value,
                    reason,
                    updated_at,
                    updated_at,
                    task_id,
                    expected_status.value,
                ),
            )
            if updated.rowcount != 1:
                return None
            row = cur.execute(
                "SELECT * FROM object_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_object_task(row) if row is not None else None

    def mark_object_task_result_unavailable_after_reopen(
        self,
        task_id: str,
        *,
        expected_result_oid: str,
        wait: Mapping[str, Any],
        error: str,
        updated_at: str,
    ) -> ObjectTask | None:
        with self._join_or_begin_transaction() as cur:
            updated = cur.execute(
                """
                UPDATE object_tasks
                   SET status = ?, result_oid = NULL, error = ?, wait_json = ?,
                       updated_at = ?
                 WHERE task_id = ? AND status = ? AND result_oid = ?
                """,
                (
                    ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN.value,
                    error,
                    dumps(wait),
                    updated_at,
                    task_id,
                    ObjectTaskStatus.SUCCEEDED.value,
                    expected_result_oid,
                ),
            )
            if updated.rowcount != 1:
                return None
            row = cur.execute(
                "SELECT * FROM object_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_object_task(row) if row is not None else None

    def _object_task_recovery_limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(
                "object task recovery limit must be a positive integer"
            )
        hard_limit = self.config.runtime.object_task_recovery_page_hard_limit
        if limit > hard_limit:
            raise ValidationError(
                "object task recovery limit exceeds configured hard cap: "
                f"{limit} > {hard_limit}"
            )
        return limit

    @staticmethod
    def _object_task_recovery_filter(
        kind: ObjectTaskRecoveryKind,
    ) -> tuple[list[str], list[Any]]:
        if kind == ObjectTaskRecoveryKind.ACTIVE:
            return [
                "status IN ("
                "'queued', 'running', 'waiting_human', "
                "'waiting_process', 'waiting_message'"
                ")"
            ], []
        if kind == ObjectTaskRecoveryKind.MISSING_RESULT:
            return ["status = 'succeeded'", "result_oid IS NOT NULL"], []
        return [
            "status IN ('succeeded', 'failed', 'cancelled')",
            "notification_status IN ('none', 'failed')",
            "notification_recipient_pid IS NOT NULL",
        ], []

    @staticmethod
    def _object_task_recovery_index(kind: ObjectTaskRecoveryKind) -> str:
        return {
            ObjectTaskRecoveryKind.ACTIVE: (
                "idx_object_tasks_recovery_active_eligible"
            ),
            ObjectTaskRecoveryKind.MISSING_RESULT: (
                "idx_object_tasks_recovery_result_eligible"
            ),
            ObjectTaskRecoveryKind.NOTIFICATION: (
                "idx_object_tasks_recovery_notification_eligible"
            ),
        }[kind]

    def upsert_agent_rating(self, rating: AgentRating) -> AgentRating:
        self._execute(
            """
            INSERT INTO agent_ratings (
                rating_id, pid, score, comment, rater, source,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pid, rater, source) DO UPDATE SET
                score = excluded.score,
                comment = excluded.comment,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                rating.rating_id,
                rating.pid,
                rating.score,
                rating.comment,
                rating.rater,
                rating.source,
                dumps(rating.metadata),
                rating.created_at,
                rating.updated_at,
            ),
        )
        saved = self.get_agent_rating(rating.pid, rating.rater, rating.source)
        if saved is None:
            raise ValidationError(f"failed to persist agent rating for process {rating.pid}")
        return saved

    def get_agent_rating(self, pid: str, rater: str, source: str = "gui") -> AgentRating | None:
        rows = self._query(
            "SELECT * FROM agent_ratings WHERE pid = ? AND rater = ? AND source = ?",
            (pid, rater, source),
        )
        return self._row_to_agent_rating(rows[0]) if rows else None

    def get_agent_ratings_for_processes(
        self,
        pids: Iterable[str],
        *,
        rater: str,
        source: str = "gui",
    ) -> dict[str, AgentRating]:
        selected = sorted({str(pid) for pid in pids if str(pid)})
        if not selected:
            return {}
        placeholders = ", ".join("?" for _ in selected)
        rows = self._query(
            f"""
            SELECT *
              FROM agent_ratings
             WHERE pid IN ({placeholders})
               AND rater = ?
               AND source = ?
             ORDER BY pid
            """,
            [*selected, rater, source],
        )
        return {str(row["pid"]): self._row_to_agent_rating(row) for row in rows}

    def list_agent_ratings(self, pid: str | None = None, limit: int | None = None) -> list[AgentRating]:
        params: list[Any] = []
        where = ""
        if pid is not None:
            where = " WHERE pid = ?"
            params.append(pid)
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self._query(
            f"SELECT * FROM agent_ratings{where} ORDER BY updated_at DESC, rating_id ASC{limit_sql}",
            params,
        )
        return [self._row_to_agent_rating(row) for row in rows]

    def mark_object_tasks_abandoned(self, reason: str) -> list[str]:
        active = self.list_object_tasks(include_terminal=False)
        if not active:
            return []
        now = utc_now()
        task_ids = [task.task_id for task in active]
        with self._join_or_begin_transaction() as cur:
            cur.executemany(
                """
                UPDATE object_tasks
                   SET status = ?, error = ?, updated_at = ?, completed_at = ?
                 WHERE task_id = ?
                """,
                [(ObjectTaskStatus.ABANDONED.value, reason, now, now, task_id) for task_id in task_ids],
            )
        return task_ids

    def insert_tool(self, handle: ToolHandle, spec: ToolSpec, registered_by: str, created_at: str, ephemeral: bool) -> None:
        with self._join_or_begin_transaction() as cursor:
            cursor.execute(
                "INSERT INTO tools VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    handle.tool_id,
                    handle.name,
                    dumps(spec),
                    handle.scope,
                    registered_by,
                    created_at,
                    int(ephemeral),
                ),
            )
            self._refresh_process_binding_jit_eligibility(
                cursor,
                tool_id=handle.tool_id,
            )

    def update_tool(self, handle: ToolHandle, spec: ToolSpec, registered_by: str, ephemeral: bool) -> None:
        with self._join_or_begin_transaction() as cursor:
            cursor.execute(
                """
                UPDATE tools
                   SET name = ?, spec_json = ?, scope = ?, registered_by = ?, ephemeral = ?
                 WHERE tool_id = ?
                """,
                (
                    handle.name,
                    dumps(spec),
                    handle.scope,
                    registered_by,
                    int(ephemeral),
                    handle.tool_id,
                ),
            )
            self._refresh_process_binding_jit_eligibility(
                cursor,
                tool_id=handle.tool_id,
            )

    def delete_tool(self, tool_id: str, *, registered_by: str | None = None) -> None:
        with self._join_or_begin_transaction() as cursor:
            if registered_by is None:
                cursor.execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))
            else:
                cursor.execute(
                    "DELETE FROM tools WHERE tool_id = ? AND registered_by = ?",
                    (tool_id, registered_by),
                )
            self._refresh_process_binding_jit_eligibility(
                cursor,
                tool_id=tool_id,
            )

    def get_tool_spec(self, tool_id: str) -> ToolSpec | None:
        rows = self._query("SELECT * FROM tools WHERE tool_id = ?", (tool_id,))
        if not rows:
            return None
        return self._dict_to_tool_spec(loads(rows[0]["spec_json"]))

    def get_existing_tool_ids(
        self,
        tool_ids: Iterable[str],
    ) -> frozenset[str]:
        """Return only requested durable tool identities through the PK index."""

        if isinstance(tool_ids, (str, bytes)):
            raise ValidationError("tool ID lookup requires an iterable of identities")
        selected_ids: set[str] = set()
        hard_limit = self.config.runtime.publication_artifact_lookup_hard_limit
        for item_count, tool_id in enumerate(tool_ids, start=1):
            if item_count > hard_limit:
                raise ValidationError(
                    "tool ID lookup exceeds configured hard cap: "
                    f"{item_count} > {hard_limit}"
                )
            if not isinstance(tool_id, str) or not tool_id or "\x00" in tool_id:
                raise ValidationError("tool ID lookup requires non-empty string identities")
            selected_ids.add(tool_id)
        if not selected_ids:
            return frozenset()

        ordered_ids = sorted(selected_ids)
        existing_ids: set[str] = set()
        for offset in range(0, len(ordered_ids), _TOOL_ID_LOOKUP_BATCH_SIZE):
            batch = ordered_ids[offset : offset + _TOOL_ID_LOOKUP_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._query(
                f"SELECT tool_id FROM tools WHERE tool_id IN ({placeholders})",
                batch,
            )
            for row in rows:
                stored_tool_id = row["tool_id"]
                if not isinstance(stored_tool_id, str) or stored_tool_id not in selected_ids:
                    raise ValidationError("tool ID lookup returned an invalid durable identity")
                existing_ids.add(stored_tool_id)
        return frozenset(existing_ids)

    def list_tools(self, limit: int | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        sql = "SELECT * FROM tools ORDER BY created_at, tool_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [dict(row) for row in self._query(sql, params)]

    def insert_tool_candidate(self, candidate: ToolCandidate) -> None:
        self._execute(
            """
            INSERT INTO tool_candidates (
                candidate_id, pid, spec_json, source_code, tests_json,
                requested_capabilities_json, status, registered_tool_id,
                validation_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.candidate_id,
                candidate.pid,
                dumps(candidate.spec),
                candidate.source_code,
                dumps(candidate.tests),
                dumps(candidate.requested_capabilities),
                candidate.status.value,
                candidate.registered_tool_id,
                dumps(candidate.validation) if candidate.validation is not None else None,
                candidate.created_at,
                candidate.updated_at,
            ),
        )

    def update_tool_candidate(self, candidate: ToolCandidate) -> None:
        self._execute(
            """
            UPDATE tool_candidates
               SET pid = ?, spec_json = ?, source_code = ?, tests_json = ?,
                   requested_capabilities_json = ?, status = ?, registered_tool_id = ?,
                   validation_json = ?, created_at = ?, updated_at = ?
             WHERE candidate_id = ?
            """,
            (
                candidate.pid,
                dumps(candidate.spec),
                candidate.source_code,
                dumps(candidate.tests),
                dumps(candidate.requested_capabilities),
                candidate.status.value,
                candidate.registered_tool_id,
                dumps(candidate.validation) if candidate.validation is not None else None,
                candidate.created_at,
                candidate.updated_at,
                candidate.candidate_id,
            ),
        )

    def get_tool_candidate(self, candidate_id: str) -> ToolCandidate | None:
        rows = self._query("SELECT * FROM tool_candidates WHERE candidate_id = ?", (candidate_id,))
        return self._row_to_tool_candidate(rows[0]) if rows else None

    def upsert_skill(
        self,
        skill: SkillPackage,
        *,
        source_type: str,
        source: str | None,
        package_sha256: str,
        registered_by: str,
        created_at: str,
    ) -> None:
        self._execute(
            """
            INSERT INTO skills (
                skill_id, name, version, package_json, source_type, source,
                package_sha256, registered_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                name = excluded.name,
                version = excluded.version,
                package_json = excluded.package_json,
                source_type = excluded.source_type,
                source = excluded.source,
                package_sha256 = excluded.package_sha256,
                registered_by = excluded.registered_by,
                updated_at = excluded.updated_at
            """,
            (
                skill.skill_id,
                skill.name,
                skill.version,
                dumps(skill),
                source_type,
                source,
                package_sha256,
                registered_by,
                created_at,
                created_at,
            ),
        )

    def get_skill(self, skill_id: str) -> tuple[SkillPackage, dict[str, Any]] | None:
        rows = self._query("SELECT * FROM skills WHERE skill_id = ?", (skill_id,))
        if not rows:
            return None
        row = rows[0]
        return self._dict_to_skill_package(loads(row["package_json"], {})), self._skill_row_metadata(row)

    def list_skills(self, text: str | None = None, limit: int | None = None) -> list[tuple[SkillPackage, dict[str, Any]]]:
        params: list[Any] = []
        sql = "SELECT * FROM skills"
        if text:
            needle = f"%{text.lower()}%"
            sql += " WHERE lower(skill_id) LIKE ? OR lower(name) LIKE ? OR lower(package_json) LIKE ?"
            params.extend([needle, needle, needle])
        sql += " ORDER BY name, skill_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            (self._dict_to_skill_package(loads(row["package_json"], {})), self._skill_row_metadata(row))
            for row in self._query(sql, params)
        ]

    def insert_skill_trust(
        self,
        *,
        trust_id: str,
        source_type: str,
        source: str,
        package_sha256: str,
        trusted_by: str,
        created_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT OR REPLACE INTO skill_trust (
                trust_id, source_type, source, package_sha256, trusted_by, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trust_id,
                source_type,
                source,
                package_sha256,
                trusted_by,
                created_at,
                dumps(metadata or {}),
            ),
        )

    def delete_skill_trust(self, *, source_type: str, source: str, package_sha256: str) -> None:
        self._execute(
            "DELETE FROM skill_trust WHERE source_type = ? AND source = ? AND package_sha256 = ?",
            (source_type, source, package_sha256),
        )

    def is_skill_trusted(self, *, source_type: str, source: str, package_sha256: str) -> bool:
        rows = self._query(
            "SELECT 1 FROM skill_trust WHERE source_type = ? AND source = ? AND package_sha256 = ?",
            (source_type, source, package_sha256),
        )
        return bool(rows)

    def list_skill_trust(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._query("SELECT * FROM skill_trust ORDER BY created_at, source")]

    def upsert_jsonrpc_endpoint(self, endpoint: JsonRpcEndpointSpec, *, registered_by: str, created_at: str) -> None:
        spec_json = self._canonical_provider_registry_spec_json(
            "jsonrpc",
            endpoint,
        )
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO jsonrpc_endpoints (
                    endpoint_id, spec_json, registered_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(endpoint_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    registered_by = excluded.registered_by,
                    updated_at = excluded.updated_at
                """,
                (
                    endpoint.endpoint_id,
                    spec_json,
                    registered_by,
                    created_at,
                    created_at,
                ),
            )
            self._advance_provider_registry_generation(
                cur,
                "jsonrpc_registry_generation",
            )

    def get_jsonrpc_endpoint(self, endpoint_id: str) -> tuple[JsonRpcEndpointSpec, dict[str, Any]] | None:
        rows = self._query("SELECT * FROM jsonrpc_endpoints WHERE endpoint_id = ?", (endpoint_id,))
        if not rows:
            return None
        row = rows[0]
        return self._dict_to_jsonrpc_endpoint(loads(row["spec_json"], {})), self._jsonrpc_endpoint_row_metadata(row)

    def list_jsonrpc_endpoints(self, text: str | None = None, limit: int | None = None) -> list[tuple[JsonRpcEndpointSpec, dict[str, Any]]]:
        params: list[Any] = []
        sql = "SELECT * FROM jsonrpc_endpoints"
        if text:
            needle = f"%{text.lower()}%"
            sql += " WHERE lower(endpoint_id) LIKE ? OR lower(spec_json) LIKE ?"
            params.extend([needle, needle])
        sql += " ORDER BY endpoint_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            (self._dict_to_jsonrpc_endpoint(loads(row["spec_json"], {})), self._jsonrpc_endpoint_row_metadata(row))
            for row in self._query(sql, params)
        ]

    def delete_jsonrpc_endpoint(self, endpoint_id: str) -> None:
        with self.transaction() as cur:
            deleted = cur.execute(
                "DELETE FROM jsonrpc_endpoints WHERE endpoint_id = ?",
                (endpoint_id,),
            )
            if deleted.rowcount == 1:
                self._advance_provider_registry_generation(
                    cur,
                    "jsonrpc_registry_generation",
                )

    def get_jsonrpc_registry_binding(self, endpoint_id: str) -> dict[str, Any]:
        return self._provider_registry_binding(
            table="jsonrpc_endpoints",
            id_column="endpoint_id",
            item_id=endpoint_id,
            counter_name="jsonrpc_registry_generation",
            registry="jsonrpc",
        )

    def upsert_mcp_server(self, server: McpServerSpec, *, registered_by: str, created_at: str) -> None:
        spec_json = self._canonical_provider_registry_spec_json(
            "mcp",
            server,
        )
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO mcp_servers (
                    server_id, spec_json, registered_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(server_id) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    registered_by = excluded.registered_by,
                    updated_at = excluded.updated_at
                """,
                (
                    server.server_id,
                    spec_json,
                    registered_by,
                    created_at,
                    created_at,
                ),
            )
            self._advance_provider_registry_generation(
                cur,
                "mcp_registry_generation",
            )

    def get_mcp_server(self, server_id: str) -> tuple[McpServerSpec, dict[str, Any]] | None:
        rows = self._query("SELECT * FROM mcp_servers WHERE server_id = ?", (server_id,))
        if not rows:
            return None
        row = rows[0]
        return self._dict_to_mcp_server(loads(row["spec_json"], {})), self._mcp_server_row_metadata(row)

    def list_mcp_servers(self, text: str | None = None, limit: int | None = None) -> list[tuple[McpServerSpec, dict[str, Any]]]:
        params: list[Any] = []
        sql = "SELECT * FROM mcp_servers"
        if text:
            needle = f"%{text.lower()}%"
            sql += " WHERE lower(server_id) LIKE ? OR lower(spec_json) LIKE ?"
            params.extend([needle, needle])
        sql += " ORDER BY server_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            (self._dict_to_mcp_server(loads(row["spec_json"], {})), self._mcp_server_row_metadata(row))
            for row in self._query(sql, params)
        ]

    def delete_mcp_server(self, server_id: str) -> None:
        with self.transaction() as cur:
            deleted = cur.execute(
                "DELETE FROM mcp_servers WHERE server_id = ?",
                (server_id,),
            )
            if deleted.rowcount == 1:
                self._advance_provider_registry_generation(
                    cur,
                    "mcp_registry_generation",
                )

    def get_mcp_registry_binding(self, server_id: str) -> dict[str, Any]:
        return self._provider_registry_binding(
            table="mcp_servers",
            id_column="server_id",
            item_id=server_id,
            counter_name="mcp_registry_generation",
            registry="mcp",
        )

    @staticmethod
    def _advance_provider_registry_generation(cur: Any, counter_name: str) -> int:
        updated = cur.execute(
            """
            UPDATE runtime_counters
               SET value = value + 1
             WHERE counter_name = ?
            """,
            (counter_name,),
        )
        if updated.rowcount != 1:
            raise ValidationError(
                f"provider registry generation counter is missing: {counter_name}"
            )
        row = cur.execute(
            "SELECT value FROM runtime_counters WHERE counter_name = ?",
            (counter_name,),
        ).fetchone()
        if row is None or isinstance(row["value"], bool) or int(row["value"]) < 0:
            raise ValidationError(
                f"provider registry generation counter is invalid: {counter_name}"
            )
        return int(row["value"])

    def _provider_registry_binding(
        self,
        *,
        table: str,
        id_column: str,
        item_id: str,
        counter_name: str,
        registry: str,
    ) -> dict[str, Any]:
        # Table and column names are fixed internal constants selected by the
        # two public wrappers above; caller-provided identifiers remain values.
        rows = self._query(
            f"""
            SELECT counter.value AS registry_generation,
                   item.spec_json AS spec_json
              FROM runtime_counters AS counter
              LEFT JOIN {table} AS item ON item.{id_column} = ?
             WHERE counter.counter_name = ?
            """,
            (item_id, counter_name),
        )
        if len(rows) != 1:
            raise ValidationError(
                f"provider registry generation counter is missing: {counter_name}"
            )
        generation = rows[0]["registry_generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValidationError(
                f"provider registry generation counter is invalid: {counter_name}"
            )
        spec_json = rows[0]["spec_json"]
        digest_input = (
            self._canonical_provider_registry_spec_json(registry, str(spec_json))
            if spec_json is not None
            else dumps(
                {
                    "registry": registry,
                    "item_id": item_id,
                    "state": "absent",
                }
            )
        )
        return {
            "registry_generation": generation,
            "registry_spec_sha256": hashlib.sha256(
                digest_input.encode("utf-8")
            ).hexdigest(),
        }

    def _canonical_provider_registry_spec_json(
        self,
        registry: str,
        value: JsonRpcEndpointSpec | McpServerSpec | str,
    ) -> str:
        """Return the one durable/hash representation for provider specs.

        Existing databases may contain semantically valid numeric spellings
        such as ``timeout_s: 1`` while typed reads intentionally expose a
        float.  Decode and re-encode through the typed storage model so new
        writes, legacy raw rows, live comparisons, and reopen all bind the
        same complete spec without weakening the registry generation fence.
        """

        payload = loads(value) if isinstance(value, str) else loads(dumps(value))
        if not isinstance(payload, dict):
            raise ValidationError(f"{registry} registry spec must encode an object")
        if registry == "jsonrpc":
            return dumps(self._dict_to_jsonrpc_endpoint(payload))
        if registry == "mcp":
            return dumps(self._dict_to_mcp_server(payload))
        raise ValidationError(f"unknown provider registry: {registry}")

    def upsert_image(
        self,
        image: AgentImage,
        *,
        registered_by: str,
        source: str | None,
        created_at: str,
    ) -> None:
        self._execute(
            """
            INSERT INTO images (
                image_id, manifest_json, registered_by, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                manifest_json = excluded.manifest_json,
                registered_by = excluded.registered_by,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                image.image_id,
                dumps(image),
                registered_by,
                source,
                created_at,
                created_at,
            ),
        )

    def get_image(self, image_id: str) -> tuple[AgentImage, dict[str, Any]] | None:
        rows = self._query("SELECT * FROM images WHERE image_id = ?", (image_id,))
        if not rows:
            return None
        row = rows[0]
        return self._dict_to_agent_image(loads(row["manifest_json"], {})), self._image_row_metadata(row)

    def list_images(self, limit: int | None = None) -> list[tuple[AgentImage, dict[str, Any]]]:
        params: list[Any] = []
        sql = "SELECT * FROM images ORDER BY image_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [
            (self._dict_to_agent_image(loads(row["manifest_json"], {})), self._image_row_metadata(row))
            for row in self._query(sql, params)
        ]

    def delete_image(self, image_id: str, *, registered_by: str | None = None) -> None:
        if registered_by is None:
            self._execute("DELETE FROM images WHERE image_id = ?", (image_id,))
            return
        self._execute("DELETE FROM images WHERE image_id = ? AND registered_by = ?", (image_id, registered_by))

    def insert_image_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        artifact: dict[str, Any],
        sha256: str,
        created_by: str,
        created_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO image_artifacts (
                artifact_id, kind, artifact_json, sha256, created_by, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                kind,
                dumps(artifact),
                sha256,
                created_by,
                created_at,
                dumps(metadata or {}),
            ),
        )

    def get_image_artifact(self, artifact_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        rows = self._query("SELECT * FROM image_artifacts WHERE artifact_id = ?", (artifact_id,))
        if not rows:
            return None
        row = rows[0]
        return loads(row["artifact_json"], {}), {
            "artifact_id": row["artifact_id"],
            "kind": row["kind"],
            "sha256": row["sha256"],
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "metadata": loads(row["metadata_json"], {}),
        }

    def list_image_artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": row["artifact_id"],
                "kind": row["kind"],
                "sha256": row["sha256"],
                "created_by": row["created_by"],
                "created_at": row["created_at"],
                "metadata": loads(row["metadata_json"], {}),
            }
            for row in self._query("SELECT * FROM image_artifacts ORDER BY created_at, artifact_id")
        ]

    def upsert_runtime_module(
        self,
        *,
        module_id: str,
        name: str,
        version: str,
        entrypoint: str,
        manifest_path: str,
        manifest_sha256: str,
        source_path: str,
        source_sha256: str,
        status: str,
        loaded_at: str | None,
        registered: dict[str, Any],
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        updated_at = utc_now()
        self._execute(
            """
            INSERT INTO runtime_modules (
                module_id, name, version, entrypoint, manifest_path,
                manifest_sha256, source_path, source_sha256, status, loaded_at,
                registered_json, error, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(module_id) DO UPDATE SET
                name = excluded.name,
                version = excluded.version,
                entrypoint = excluded.entrypoint,
                manifest_path = excluded.manifest_path,
                manifest_sha256 = excluded.manifest_sha256,
                source_path = excluded.source_path,
                source_sha256 = excluded.source_sha256,
                status = excluded.status,
                loaded_at = excluded.loaded_at,
                registered_json = excluded.registered_json,
                error = excluded.error,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                module_id,
                name,
                version,
                entrypoint,
                manifest_path,
                manifest_sha256,
                source_path,
                source_sha256,
                status,
                loaded_at,
                dumps(registered),
                error,
                dumps(metadata),
                updated_at,
            ),
        )

    def get_runtime_module(self, module_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM runtime_modules WHERE module_id = ?", (module_id,))
        return self._runtime_module_row(rows[0]) if rows else None

    def list_runtime_modules(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runtime_modules ORDER BY module_id"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._runtime_module_row(row) for row in self._query(sql, params)]

    def insert_checkpoint(self, checkpoint: Checkpoint, snapshot: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO checkpoints (
                checkpoint_id, pid, reason, snapshot_json, created_at,
                created_by, snapshot_version, metadata_json, effect_ledger_seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.checkpoint_id,
                checkpoint.pid,
                checkpoint.reason,
                dumps(snapshot),
                checkpoint.created_at,
                checkpoint.created_by,
                checkpoint.snapshot_version,
                dumps(checkpoint.metadata or {}),
                checkpoint.effect_ledger_seq,
            ),
        )

    def get_checkpoint_snapshot(self, checkpoint_id: str) -> tuple[Checkpoint, dict[str, Any]] | None:
        rows = self._query("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        if not rows:
            return None
        row = rows[0]
        checkpoint = self._row_to_checkpoint(row)
        return checkpoint, loads(row["snapshot_json"], {})

    def list_checkpoints(self, pid: str | None = None, limit: int | None = None) -> list[Checkpoint]:
        params: list[Any] = []
        sql = "SELECT * FROM checkpoints"
        if pid is not None:
            sql += " WHERE pid = ?"
            params.append(pid)
        sql += " ORDER BY created_at DESC, checkpoint_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_checkpoint(row) for row in self._query(sql, params)]

    def snapshot_tables(self) -> dict[str, list[dict[str, Any]]]:
        raise RuntimeError(
            "full-table SQLite snapshots are disabled; use CheckpointManager.create for scoped durable checkpoints"
        )

    def restore_tables(self, snapshot: dict[str, list[dict[str, Any]]]) -> None:
        raise RuntimeError(
            "full-table SQLite restore is disabled; use CheckpointManager.restore to preserve append-only history"
        )

    def abandon_stale_capability_use_reservations(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> CapabilityUseReservationRecoverySummary:
        """Fail closed reservations left after prepared-effect recovery."""

        require_recovery_lease()
        page_size = (
            self.config.runtime.capability_use_reservation_recovery_page_size
        )
        recovered_total = 0
        recovered_sample: list[str] = []
        after: tuple[str, str] | None = None
        while True:
            clauses = ["status = ?"]
            params: list[Any] = ["reserved"]
            if after is not None:
                clauses.append("(created_at, reservation_id) > (?, ?)")
                params.extend(after)
            params.append(page_size)
            with self.transaction() as cur:
                rows = list(
                    cur.execute(
                        "SELECT reservation_id, created_at "
                        "FROM capability_use_reservations "
                        f"WHERE {' AND '.join(clauses)} "
                        "ORDER BY created_at, reservation_id LIMIT ?",
                        params,
                    )
                )
                if not rows:
                    break
                reservation_ids = tuple(str(row["reservation_id"]) for row in rows)
                for reservation_id in reservation_ids:
                    updated = cur.execute(
                        "UPDATE capability_use_reservations "
                        "SET status = ?, updated_at = ? "
                        "WHERE reservation_id = ? AND status = ?",
                        ("abandoned", utc_now(), reservation_id, "reserved"),
                    )
                    if updated.rowcount == 1:
                        recovered_total += 1
                        if len(recovered_sample) < page_size:
                            recovered_sample.append(reservation_id)
                last = rows[-1]
                after = (str(last["created_at"]), str(last["reservation_id"]))
        return CapabilityUseReservationRecoverySummary(
            total_count=recovered_total,
            sample_reservation_ids=tuple(recovered_sample),
        )

    def _create_v3_data_flow_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS sink_trust_registry (
              registry_key TEXT PRIMARY KEY,
              generation INTEGER NOT NULL,
              updated_at TEXT NOT NULL
            );

            INSERT INTO sink_trust_registry (registry_key, generation, updated_at)
              VALUES ('default', 0, '')
              ON CONFLICT(registry_key) DO NOTHING;

            CREATE TABLE IF NOT EXISTS sink_trust_records (
              trust_id TEXT PRIMARY KEY,
              schema_version INTEGER NOT NULL,
              pattern TEXT NOT NULL,
              trust_level TEXT NOT NULL,
              max_sensitivity TEXT NOT NULL,
              tenants_json TEXT NOT NULL,
              principals_json TEXT NOT NULL,
              identity_sha256 TEXT,
              generation INTEGER NOT NULL,
              spec_hash TEXT NOT NULL,
              active INTEGER NOT NULL,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              deactivated_at TEXT,
              UNIQUE(pattern, generation)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_sink_trust_active_pattern
              ON sink_trust_records(pattern) WHERE active = 1;

            CREATE INDEX IF NOT EXISTS idx_sink_trust_generation
              ON sink_trust_records(generation, pattern, trust_id);

            CREATE TABLE IF NOT EXISTS data_flow_decisions (
              decision_id TEXT PRIMARY KEY,
              pid TEXT NOT NULL,
              sink TEXT NOT NULL,
              direction TEXT NOT NULL,
              outcome TEXT NOT NULL,
              reason TEXT NOT NULL,
              labels_json TEXT NOT NULL,
              source_refs_json TEXT NOT NULL,
              payload_hash TEXT NOT NULL,
              trust_id TEXT,
              trust_hash TEXT,
              registry_generation INTEGER NOT NULL,
              release_capability_id TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_data_flow_decisions_created
              ON data_flow_decisions(created_at, decision_id);

            CREATE INDEX IF NOT EXISTS idx_data_flow_decisions_pid_created
              ON data_flow_decisions(pid, created_at, decision_id);

            CREATE INDEX IF NOT EXISTS idx_data_flow_decisions_sink_created
              ON data_flow_decisions(sink, created_at, decision_id);

            CREATE TABLE IF NOT EXISTS file_label_bindings (
              binding_id TEXT PRIMARY KEY,
              normalized_path TEXT NOT NULL,
              content_sha256 TEXT,
              labels_json TEXT NOT NULL,
              source_refs_json TEXT NOT NULL,
              generation INTEGER NOT NULL,
              tombstoned INTEGER NOT NULL,
              active INTEGER NOT NULL,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              superseded_at TEXT,
              UNIQUE(normalized_path, generation)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_file_label_active_path
              ON file_label_bindings(normalized_path) WHERE active = 1;

            CREATE INDEX IF NOT EXISTS idx_file_label_path_generation
              ON file_label_bindings(normalized_path, generation, binding_id);

            CREATE INDEX IF NOT EXISTS idx_file_label_tree_scan
              ON file_label_bindings(
                active,
                tombstoned,
                normalized_path COLLATE BINARY,
                generation DESC,
                binding_id
              );
            """
        )

    def _create_v3_operation_schema(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS operations (
              operation_id TEXT COLLATE BINARY PRIMARY KEY,
              root_operation_id TEXT COLLATE BINARY NOT NULL,
              parent_operation_id TEXT COLLATE BINARY,
              kind TEXT NOT NULL,
              name TEXT NOT NULL,
              actor TEXT NOT NULL,
              pid TEXT,
              state TEXT NOT NULL,
              outcome TEXT NOT NULL,
              expected_roles_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              runtime_publication_id TEXT COLLATE BINARY,
              started_at TEXT COLLATE BINARY NOT NULL,
              updated_at TEXT NOT NULL,
              completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_operations_pid_started
              ON operations(
                pid, started_at COLLATE BINARY,
                operation_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_operations_root_started
              ON operations(
                root_operation_id COLLATE BINARY, started_at COLLATE BINARY,
                operation_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_operations_state_updated
              ON operations(state, updated_at, operation_id);

            CREATE INDEX IF NOT EXISTS idx_operations_state_started
              ON operations(
                state, started_at COLLATE BINARY DESC,
                operation_id COLLATE BINARY DESC
              );

            CREATE INDEX IF NOT EXISTS idx_operations_parent_root
              ON operations(
                parent_operation_id COLLATE BINARY,
                root_operation_id COLLATE BINARY,
                operation_id COLLATE BINARY
              );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_operations_runtime_publication
              ON operations(runtime_publication_id COLLATE BINARY)
              WHERE runtime_publication_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS operation_evidence (
              link_id TEXT COLLATE BINARY PRIMARY KEY,
              operation_id TEXT COLLATE BINARY NOT NULL,
              evidence_type TEXT NOT NULL,
              evidence_id TEXT COLLATE BINARY NOT NULL,
              role TEXT NOT NULL,
              created_at TEXT COLLATE BINARY NOT NULL,
              metadata_json TEXT NOT NULL,
              UNIQUE(operation_id, evidence_type, evidence_id, role)
            );

            CREATE INDEX IF NOT EXISTS idx_operation_evidence_operation_created
              ON operation_evidence(
                operation_id COLLATE BINARY, created_at COLLATE BINARY,
                link_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_operation_evidence_created
              ON operation_evidence(
                created_at COLLATE BINARY, link_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_operation_evidence_operation_type
              ON operation_evidence(
                operation_id COLLATE BINARY, evidence_type,
                evidence_id COLLATE BINARY
              );

            CREATE INDEX IF NOT EXISTS idx_operation_evidence_lookup
              ON operation_evidence(
                evidence_type, evidence_id COLLATE BINARY,
                operation_id COLLATE BINARY
              );

            CREATE TABLE IF NOT EXISTS context_materialization_manifests (
              materialization_id TEXT PRIMARY KEY,
              pid TEXT NOT NULL,
              view_id TEXT NOT NULL,
              policy TEXT NOT NULL,
              budget_tokens INTEGER NOT NULL,
              rendered_tokens INTEGER NOT NULL,
              rendered_sha256 TEXT NOT NULL,
              context_generation TEXT,
              context_oid TEXT,
              context_version INTEGER,
              objects_json TEXT NOT NULL,
              compaction_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_context_manifests_pid_created
              ON context_materialization_manifests(pid, created_at, materialization_id);
            """
        )

    def _decode_stored_object_payload(self, payload_json: str) -> Any:
        payload = loads(payload_json, {})
        if (
            isinstance(payload, dict)
            and {"storage", "present"}.issubset(payload)
            and set(payload).issubset(
                {"storage", "present", "recovered_after_reopen"}
            )
            and payload.get("storage") == "runtime_memory"
            and isinstance(payload.get("present"), bool)
        ):
            return _MISSING_OBJECT_PAYLOAD
        return payload

    def recover_missing_runtime_object_payloads(
        self,
        *,
        require_recovery_lease: Callable[[], None],
    ) -> ObjectPayloadRecoverySummary:
        """Release volatile payload rows in keyset-paged, constant-size writes."""

        require_recovery_lease()
        page_size = self.config.runtime.object_payload_recovery_page_size
        cursor: tuple[str, str] | None = None
        total_count = 0
        sample_oids: list[str] = []
        while True:
            rows = self._query_object_payload_recovery_page(
                after=cursor,
                limit=page_size,
            )
            if not rows:
                break
            released = self._release_object_payload_recovery_page(
                rows,
            )
            total_count += len(released)
            remaining = page_size - len(sample_oids)
            if remaining > 0:
                sample_oids.extend(released[:remaining])
            last = rows[-1]
            cursor = (str(last["created_at"]), str(last["oid"]))
        return ObjectPayloadRecoverySummary(
            total_count=total_count,
            sample_oids=tuple(sample_oids),
        )

    def _query_object_payload_recovery_page(
        self,
        *,
        after: tuple[str, str] | None,
        limit: int,
    ) -> list[Any]:
        clauses = [
            "lifecycle_state = ?",
            f"payload_json IN ({_RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS_SQL})",
        ]
        params: list[Any] = [ObjectLifecycleState.LIVE.value]
        if after is not None:
            clauses.append("(created_at, oid) > (?, ?)")
            params.extend(after)
        params.append(limit)
        return self._query(
            "SELECT oid, created_at FROM objects "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at, oid LIMIT ?",
            params,
        )

    def _release_object_payload_recovery_page(
        self,
        rows: Iterable[Any],
    ) -> list[str]:
        released: list[str] = []
        now = utc_now()
        missing_marker = dumps(
            self.payload_marker(present=False, recovered_after_reopen=True)
        )
        with self._join_or_begin_transaction() as cur:
            for row in rows:
                oid = str(row["oid"])
                if oid in self._object_payloads:
                    continue
                updated = cur.execute(
                    f"""
                    UPDATE objects
                       SET payload_json = ?, lifecycle_state = ?, deleted_at = ?,
                           updated_at = ?
                     WHERE oid = ? AND lifecycle_state = ?
                       AND payload_json IN (
                         {', '.join('?' for _ in _RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS)}
                       )
                    """,
                    (
                        missing_marker,
                        ObjectLifecycleState.RELEASED.value,
                        now,
                        now,
                        oid,
                        ObjectLifecycleState.LIVE.value,
                        *_RUNTIME_OBJECT_PRESENT_PAYLOAD_MARKERS,
                    ),
                )
                if updated.rowcount != 1:
                    continue
                cur.execute(
                    "DELETE FROM object_links WHERE src_oid = ? OR dst_oid = ?",
                    (oid, oid),
                )
                cur.execute(
                    """
                    UPDATE capabilities SET status = ?
                     WHERE resource = ? AND status = ?
                    """,
                    (
                        CapabilityStatus.REVOKED.value,
                        f"object:{oid}",
                        CapabilityStatus.ACTIVE.value,
                    ),
                )
                released.append(oid)
        return released

    def _process_params(self, process: AgentProcess) -> tuple[Any, ...]:
        self._validate_process_state_write(
            process,
            allow_state_transition=False,
        )
        return (
            process.pid,
            process.parent_pid,
            process.image_id,
            process.status.value,
            process.goal_oid,
            dumps(process.memory_view) if process.memory_view else None,
            dumps(process.capabilities),
            dumps(process.loaded_skills),
            dumps(process.tool_table),
            dumps(process.model_tool_table),
            process.event_cursor,
            process.checkpoint_head,
            process.status_message,
            dumps(process_wait_state_to_mapping(process.wait_state)),
            dumps(process_outcome_to_mapping(process.outcome)),
            process.state_generation,
            dumps(process.resource_budget),
            dumps(process.resource_usage),
            process.working_directory,
            process.llm_profile_id,
            process.revision,
            process.execution_generation,
            process.execution_owner_id,
            process.execution_lease_id,
            process.created_at,
            process.updated_at,
        )

    def _object_task_params(self, task: ObjectTask) -> tuple[Any, ...]:
        return (
            task.task_id,
            task.owner_oid,
            task.creator_pid,
            task.runner_pid,
            task.tool,
            task.tool_id,
            task.status.value,
            task.notification.status.value,
            task.notification.recipient_pid,
            dumps(task.notification),
            dumps(task.owner_watch),
            task.result_oid,
            task.error,
            dumps(task.wait),
            task.created_at,
            task.updated_at,
            task.started_at,
            task.completed_at,
        )

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    def _runtime_module_row(self, row: Any) -> dict[str, Any]:
        data = self._row_to_dict(row)
        data["registered"] = loads(data.pop("registered_json"), {})
        data["metadata"] = loads(data.pop("metadata_json"), {})
        return data

    def _image_row_metadata(self, row: Any) -> dict[str, Any]:
        return {
            "registered_by": row["registered_by"],
            "source": row["source"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _dict_to_agent_image(self, data: dict[str, Any]) -> AgentImage:
        with _persisted_model_decode(f"agent image {data.get('image_id', '<unknown>') if isinstance(data, dict) else '<unknown>'}"):
            item = dict(data)
            item.setdefault("boot", {"kind": "fresh"})
            image = AgentImage(**item)
            if image.prompt_mode not in PROMPT_MODES:
                raise ValidationError(f"invalid persisted agent image {image.image_id}: unknown prompt_mode {image.prompt_mode}")
            if image.jit_tool_exposure not in JIT_TOOL_EXPOSURES:
                raise ValidationError(
                    f"invalid persisted agent image {image.image_id}: "
                    f"unknown jit_tool_exposure {image.jit_tool_exposure}"
                )
            return image

    def _row_to_object(self, row: Any) -> AgentObject:
        with _persisted_model_decode(f"object {row['oid']}"):
            metadata = ObjectMetadata.from_persisted(
                loads(row["metadata_json"], {})
            )
            provenance = Provenance(**loads(row["provenance_json"], {}))
            return AgentObject(
                oid=row["oid"],
                namespace=row["namespace"],
                name=row["name"],
                type=ObjectType(row["type"]),
                schema_version=row["schema_version"],
                payload=self.object_payload(str(row["oid"])),
                metadata=metadata,
                provenance=provenance,
                version=row["version"],
                immutable=bool(row["immutable"]),
                created_by=row["created_by"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                owner_kind=ObjectOwnerKind(row["owner_kind"]),
                owner_id=row["owner_id"],
                lifecycle_state=ObjectLifecycleState(row["lifecycle_state"]),
                deleted_at=row["deleted_at"],
            )

    def _row_to_namespace(self, row: Any) -> ObjectNamespace:
        return ObjectNamespace(
            namespace=row["namespace"],
            parent_namespace=row["parent_namespace"],
            metadata=loads(row["metadata_json"], {}),
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_link(self, row: Any) -> ObjectLink:
        with _persisted_model_decode(f"object link {row['id']}"):
            return ObjectLink(
                link_id=row["id"],
                src=row["src_oid"],
                relation=RelationType(row["relation"]),
                dst=row["dst_oid"],
                metadata=loads(row["metadata_json"], {}),
                created_by=row["created_by"],
                created_at=row["created_at"],
            )

    def _row_to_process(self, row: Any) -> AgentProcess:
        with _persisted_model_decode(f"process {row['pid']}"):
            row_keys = set(row.keys())
            if "wait_state_json" in row_keys and "outcome_json" in row_keys:
                wait_state = process_wait_state_from_json(row["wait_state_json"])
                outcome = process_outcome_from_json(row["outcome_json"])
                status_message = row["status_message"]
            else:
                upcast = upcast_legacy_process_state(
                    str(row["status"]),
                    row["status_message"],
                )
                wait_state = upcast.wait_state
                outcome = upcast.outcome
                status_message = upcast.status_message
            try:
                validate_process_state_fields(
                    str(row["status"]),
                    wait_state,
                    outcome,
                )
            except ValidationError as exc:
                raise ValidationError(
                    f"invalid persisted process {row['pid']}: {exc}"
                ) from exc
            return AgentProcess(
                pid=row["pid"],
                parent_pid=row["parent_pid"],
                image_id=row["image_id"],
                status=ProcessStatus(row["status"]),
                goal_oid=row["goal_oid"],
                memory_view=self._dict_to_view(loads(row["memory_view_json"])) if row["memory_view_json"] else None,
                capabilities=loads(row["capabilities_json"], []),
                loaded_skills=loads(row["loaded_skills_json"], {}),
                tool_table=loads(row["tool_table_json"], {}),
                model_tool_table=loads(
                    row["model_tool_table_json"]
                    if "model_tool_table_json" in row.keys()
                    else row["tool_table_json"],
                    {},
                ),
                event_cursor=row["event_cursor"],
                checkpoint_head=row["checkpoint_head"],
                resource_budget=ResourceBudget(**loads(row["resource_budget_json"], {})),
                resource_usage=ResourceUsage(
                    **loads(row["resource_usage_json"] if "resource_usage_json" in row.keys() else None, {})
                ),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                working_directory=row["working_directory"] if "working_directory" in row.keys() else ".",
                status_message=status_message,
                wait_state=wait_state,
                outcome=outcome,
                state_generation=(
                    int(row["state_generation"])
                    if "state_generation" in row_keys
                    else 0
                ),
                llm_profile_id=(
                    row["llm_profile_id"]
                    if "llm_profile_id" in row.keys() and row["llm_profile_id"]
                    else self.config.llm.default_profile_id
                ),
                revision=int(row["revision"]),
                execution_generation=int(row["execution_generation"]),
                execution_owner_id=row["execution_owner_id"],
                execution_lease_id=row["execution_lease_id"],
            )

    def _row_to_authority_manifest(self, row: Any) -> TaskAuthorityManifest:
        with _persisted_model_decode(f"authority manifest {row['manifest_id']}"):
            permitted_effects_payload = loads(row["permitted_effects_json"])
            policy_schema_version = (
                1
                if isinstance(permitted_effects_payload, list)
                else PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
            )
            return TaskAuthorityManifest(
                manifest_id=str(row["manifest_id"]),
                pid=str(row["pid"]),
                image_id=str(row["image_id"]),
                goal_ref=row["goal_ref"],
                authorized_capabilities=loads(row["authorized_capabilities_json"], []),
                required_capabilities=loads(row["required_capabilities_json"], []),
                permitted_effects=upcast_permitted_effects_policy(
                    permitted_effects_payload
                ),
                permitted_effects_policy_schema_version=policy_schema_version,
                resource_budget=loads(row["resource_budget_json"], {}),
                approval_policy=loads(row["approval_policy_json"], {}),
                data_flow_policy=loads(row["data_flow_policy_json"], {}),
                expires_at=row["expires_at"],
                issued_by=str(row["issued_by"]),
                parent_manifest_id=row["parent_manifest_id"],
                manifest_hash=str(row["manifest_hash"]),
                metadata=loads(row["metadata_json"], {}),
                created_at=str(row["created_at"]),
            )

    def _row_to_resource_reservation(self, row: Any) -> ResourceReservation:
        return ResourceReservation(
            parent_pid=row["parent_pid"],
            child_pid=row["child_pid"],
            reserved={key: float(value) for key, value in loads(row["reservation_json"], {}).items()},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_event(self, row: Any) -> Event:
        with _persisted_model_decode(f"event {row['event_id']}"):
            return Event(
                event_id=row["event_id"],
                type=EventType(row["type"]),
                source=row["source"],
                target=row["target"],
                payload=loads(row["payload_json"], {}),
                priority=EventPriority(row["priority"]),
                created_at=row["created_at"],
                correlation_id=row["correlation_id"],
                causality=loads(row["causality_json"], {}),
            )

    def _row_to_capability(self, row: Any) -> Capability:
        with _persisted_model_decode(f"capability {row['cap_id']}"):
            keys = set(row.keys())
            return Capability(
                cap_id=row["cap_id"],
                subject=row["subject"],
                resource=row["resource"],
                rights=set(loads(row["rights_json"], [])),
                constraints=loads(row["constraints_json"], {}),
                issued_by=row["issued_by"],
                issued_at=row["issued_at"],
                expires_at=row["expires_at"],
                delegable=bool(row["delegable"]),
                revocable=bool(row["revocable"]),
                effect=CapabilityEffect(row["effect"]) if "effect" in keys else CapabilityEffect.ALLOW,
                issuer_cap_id=row["issuer_cap_id"] if "issuer_cap_id" in keys else None,
                parent_cap_id=row["parent_cap_id"] if "parent_cap_id" in keys else None,
                delegation_depth=int(row["delegation_depth"]) if "delegation_depth" in keys else 0,
                max_delegation_depth=(
                    int(row["max_delegation_depth"])
                    if "max_delegation_depth" in keys and row["max_delegation_depth"] is not None
                    else None
                ),
                uses_remaining=row["uses_remaining"] if "uses_remaining" in keys else None,
                status=(
                    CapabilityStatus(row["status"])
                    if "status" in keys
                    else (CapabilityStatus.REVOKED if bool(row["revoked"]) else CapabilityStatus.ACTIVE)
                ),
                metadata=loads(row["metadata_json"], {}) if "metadata_json" in keys else {},
            )

    def _row_to_audit(self, row: Any) -> AuditRecord:
        return AuditRecord(
            record_id=row["record_id"],
            timestamp=row["timestamp"],
            actor=row["actor"],
            action=row["action"],
            target=row["target"],
            input_refs=loads(row["input_refs_json"], []),
            output_refs=loads(row["output_refs_json"], []),
            capability_refs=loads(row["capability_refs_json"], []),
            decision=loads(row["decision_json"]) if row["decision_json"] else None,
            correlation_id=row["correlation_id"],
            parent_record_id=row["parent_record_id"],
        )

    def _row_to_operation(self, row: Any) -> OperationRecord:
        with _persisted_model_decode(f"operation {row['operation_id']}"):
            metadata = loads(row["metadata_json"], {})
            metadata_publication_id = _operation_runtime_publication_id(metadata)
            if row["runtime_publication_id"] != metadata_publication_id:
                raise ValidationError(
                    "operation runtime publication binding columns are inconsistent: "
                    f"{row['operation_id']}"
                )
            return OperationRecord(
                operation_id=row["operation_id"],
                root_operation_id=row["root_operation_id"],
                parent_operation_id=row["parent_operation_id"],
                kind=OperationKind(row["kind"]),
                name=row["name"],
                actor=row["actor"],
                pid=row["pid"],
                state=OperationState(row["state"]),
                outcome=OperationOutcome(row["outcome"]),
                expected_roles=loads(row["expected_roles_json"], []),
                metadata=metadata,
                started_at=row["started_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
            )

    def _row_to_operation_evidence(self, row: Any) -> OperationEvidenceLink:
        return OperationEvidenceLink(
            link_id=row["link_id"],
            operation_id=row["operation_id"],
            evidence_type=row["evidence_type"],
            evidence_id=row["evidence_id"],
            role=row["role"],
            created_at=row["created_at"],
            metadata=loads(row["metadata_json"], {}),
        )

    def _row_to_context_materialization_manifest(
        self,
        row: Any,
    ) -> ContextMaterializationManifest:
        return ContextMaterializationManifest(
            materialization_id=row["materialization_id"],
            pid=row["pid"],
            view_id=row["view_id"],
            policy=row["policy"],
            budget_tokens=int(row["budget_tokens"]),
            rendered_tokens=int(row["rendered_tokens"]),
            rendered_sha256=row["rendered_sha256"],
            context_generation=row["context_generation"],
            context_oid=row["context_oid"],
            context_version=(int(row["context_version"]) if row["context_version"] is not None else None),
            objects=loads(row["objects_json"], []),
            compaction=loads(row["compaction_json"], {}),
            created_at=row["created_at"],
        )

    def _row_to_external_effect(self, row: Any) -> ExternalEffectRecord:
        with _persisted_model_decode(f"external effect {row['effect_id']}"):
            return ExternalEffectRecord(
                effect_id=row["effect_id"],
                record_id=row["record_id"],
                event_id=row["event_id"],
                pid=row["pid"],
                provider=row["provider"],
                operation=row["operation"],
                target=row["target"],
                rollback_class=ExternalEffectRollbackClass(row["rollback_class"]),
                rollback_status=ExternalEffectRollbackStatus(row["rollback_status"]),
                state_mutation=bool(row["state_mutation"]),
                information_flow=bool(row["information_flow"]),
                provider_metadata=loads(row["provider_metadata_json"], {}),
                created_at=row["created_at"],
                effect_state=str(row["effect_state"]),
                transaction_state=(
                    str(row["transaction_state"])
                    if "transaction_state" in row.keys()
                    else ("prepared" if str(row["effect_state"]) == "pending" else "committed")
                ),
                canonical_args_hash=(row["canonical_args_hash"] if "canonical_args_hash" in row.keys() else None),
                idempotency_key=(row["idempotency_key"] if "idempotency_key" in row.keys() else None),
                provider_receipt=(
                    loads(row["provider_receipt_json"], {})
                    if "provider_receipt_json" in row.keys()
                    else {}
                ),
                updated_at=(row["updated_at"] if "updated_at" in row.keys() else row["created_at"]),
                payload_retention_schema_version=int(
                    row["payload_retention_schema_version"]
                ),
                payload_retention_tier=str(row["payload_retention_tier"]),
                payload_retention_sha256=row["payload_retention_sha256"],
            )

    def _row_to_checkpoint(self, row: Any) -> Checkpoint:
        keys = set(row.keys())
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            pid=row["pid"],
            reason=row["reason"],
            created_at=row["created_at"],
            created_by=row["created_by"] if "created_by" in keys else None,
            snapshot_version=int(row["snapshot_version"]) if "snapshot_version" in keys else 1,
            metadata=loads(row["metadata_json"], {}) if "metadata_json" in keys else {},
            effect_ledger_seq=int(row["effect_ledger_seq"]),
        )

    def _row_to_sink_trust(self, row: Any) -> SinkTrustSpec:
        with _persisted_model_decode(f"sink trust record {row['trust_id']}"):
            active_raw = row["active"]
            if active_raw not in {0, 1, False, True}:
                raise ValueError("active must be stored as 0 or 1")
            return SinkTrustSpec(
                trust_id=row["trust_id"],
                schema_version=int(row["schema_version"]),
                pattern=row["pattern"],
                trust_level=SinkTrustLevel(row["trust_level"]),
                max_sensitivity=row["max_sensitivity"],
                tenants=tuple(loads(row["tenants_json"], [])),
                principals=tuple(loads(row["principals_json"], [])),
                identity_sha256=row["identity_sha256"],
                generation=int(row["generation"]),
                spec_hash=row["spec_hash"],
                active=bool(active_raw),
                created_by=row["created_by"],
                created_at=row["created_at"],
                deactivated_at=row["deactivated_at"],
            )

    def _row_to_data_flow_decision(self, row: Any) -> DataFlowDecision:
        with _persisted_model_decode(f"data-flow decision {row['decision_id']}"):
            raw_refs = loads(row["source_refs_json"], [])
            if not isinstance(raw_refs, list):
                raise ValueError("source_refs_json must contain an array")
            return DataFlowDecision(
                decision_id=row["decision_id"],
                pid=row["pid"],
                sink=row["sink"],
                direction=DataFlowDirection(row["direction"]),
                outcome=DataFlowOutcome(row["outcome"]),
                reason=row["reason"],
                labels=DataLabels.from_dict(loads(row["labels_json"], {})),
                source_refs=tuple(DataSourceRef.from_dict(item) for item in raw_refs),
                payload_hash=row["payload_hash"],
                trust_id=row["trust_id"],
                trust_hash=row["trust_hash"],
                registry_generation=int(row["registry_generation"]),
                release_capability_id=row["release_capability_id"],
                created_at=row["created_at"],
            )

    def _row_to_file_label_binding(self, row: Any) -> FileLabelBinding:
        with _persisted_model_decode(f"file label binding {row['binding_id']}"):
            raw_refs = loads(row["source_refs_json"], [])
            if not isinstance(raw_refs, list):
                raise ValueError("source_refs_json must contain an array")
            tombstoned_raw = row["tombstoned"]
            active_raw = row["active"]
            if tombstoned_raw not in {0, 1, False, True} or active_raw not in {0, 1, False, True}:
                raise ValueError("tombstoned and active must be stored as 0 or 1")
            return FileLabelBinding(
                binding_id=row["binding_id"],
                normalized_path=row["normalized_path"],
                content_sha256=row["content_sha256"],
                labels=DataLabels.from_dict(loads(row["labels_json"], {})),
                source_refs=tuple(DataSourceRef.from_dict(item) for item in raw_refs),
                generation=int(row["generation"]),
                tombstoned=bool(tombstoned_raw),
                active=bool(active_raw),
                created_by=row["created_by"],
                created_at=row["created_at"],
                superseded_at=row["superseded_at"],
            )

    @staticmethod
    def _sink_trust_generation_from_cursor(cur: Any) -> int:
        cur.execute("SELECT generation FROM sink_trust_registry WHERE registry_key = 'default'")
        row = cur.fetchone()
        if row is None:
            raise ValidationError("sink trust registry metadata is missing")
        generation = row["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValidationError("invalid persisted sink trust registry generation")
        return generation

    @staticmethod
    def _file_label_generation_from_cursor(cur: Any, normalized_path: str) -> int:
        cur.execute(
            "SELECT MAX(generation) AS generation FROM file_label_bindings WHERE normalized_path = ?",
            (normalized_path,),
        )
        row = cur.fetchone()
        value = row["generation"] if row is not None else None
        return int(value) if value is not None else 0

    @staticmethod
    def _insert_file_label_binding(cur: Any, binding: FileLabelBinding) -> None:
        cur.execute(
            """
            INSERT INTO file_label_bindings (
                binding_id, normalized_path, content_sha256, labels_json,
                source_refs_json, generation, tombstoned, active,
                created_by, created_at, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                binding.binding_id,
                binding.normalized_path,
                binding.content_sha256,
                dumps(binding.labels.to_dict()),
                dumps([ref.to_dict() for ref in binding.source_refs]),
                binding.generation,
                int(binding.tombstoned),
                int(binding.active),
                binding.created_by,
                binding.created_at,
                binding.superseded_at,
            ),
        )

    @staticmethod
    def _data_flow_list_limit(
        limit: int | None,
        *,
        default: int,
        hard_limit: int,
        label: str,
    ) -> int:
        selected = default if limit is None else limit
        if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
            raise ValidationError(f"{label} limit must be a positive integer")
        if selected > hard_limit:
            raise ValidationError(f"{label} limit exceeds hard cap {hard_limit}")
        return selected

    def _row_to_human_request(self, row: Any) -> HumanRequest:
        with _persisted_model_decode(f"human request {row['request_id']}"):
            return HumanRequest(
                request_id=row["request_id"],
                pid=row["pid"],
                human=row["human"],
                payload=loads(row["payload_json"], {}),
                status=HumanRequestStatus(row["status"]),
                decision=loads(row["decision_json"]) if row["decision_json"] else None,
                blocking=bool(row["blocking"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def _row_to_llm_call(self, row: Any) -> LLMCallRecord:
        with _persisted_model_decode(f"LLM call {row['call_id']}"):
            record = LLMCallRecord(
                call_id=row["call_id"],
                pid=row["pid"],
                image_id=row["image_id"],
                purpose=row["purpose"],
                status=row["status"],
                api=row["api"],
                model=row["model"],
                request_id=row["request_id"],
                response_id=row["response_id"],
                messages=loads(row["messages_json"], []),
                tools=loads(row["tools_json"], []),
                request_options=loads(row["request_options_json"], {}),
                response_content=row["response_content"],
                tool_calls=loads(row["tool_calls_json"], []),
                reasoning=(
                    loads(row["reasoning_json"])
                    if row["reasoning_json"]
                    else None
                ),
                usage=loads(row["usage_json"], {}),
                raw_response=(
                    loads(row["raw_response_json"])
                    if row["raw_response_json"]
                    else None
                ),
                observability=loads(row["observability_json"], {}),
                error=row["error"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
            )
            persisted_tier = PayloadRetentionTier(
                str(row["payload_retention_tier"])
            )
            if llm_call_payload_retention_tier(record) is not persisted_tier:
                raise ValueError(
                    "payload retention tier disagrees with its durable marker"
                )
            return record

    def _row_to_llm_pending_action(self, row: Any) -> dict[str, Any]:
        with _persisted_model_decode(f"pending LLM action {row['pid']}"):
            return self._decode_llm_pending_action(row)

    def _decode_llm_pending_action(self, row: Any) -> dict[str, Any]:
        data_flow_context = _canonical_pending_data_flow_context(
            loads(row["data_flow_context_json"])
        )
        return {
            "pid": row["pid"],
            "resume_token": row["resume_token"],
            "llm_operation_id": row["llm_operation_id"],
            "tool_operation_id": row["tool_operation_id"],
            "wait_type": row["wait_type"],
            "request_id": row["request_id"],
            "child_pid": row["child_pid"],
            "response_id": row["response_id"],
            "tool_call_id": row["tool_call_id"],
            "tool_name": row["tool_name"],
            "filters": loads(row["filters_json"], {}),
            "action": loads(row["action_json"], {}),
            "data_flow_context": data_flow_context,
            "content_preview": row["content_preview"],
            "tool_call_count": row["tool_call_count"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_object_task(self, row: Any) -> ObjectTask:
        with _persisted_model_decode(f"object task {row['task_id']}"):
            raw_notification = loads(row["notification_json"], {})
            if isinstance(raw_notification.get("status"), str):
                raw_notification["status"] = ObjectTaskNotificationStatus(raw_notification["status"])
            notification = ObjectTaskNotification(**raw_notification)
            persisted_notification_status = ObjectTaskNotificationStatus(
                row["notification_status"]
            )
            if notification.status != persisted_notification_status:
                raise ValueError("normalized notification status does not match payload")
            if notification.recipient_pid != row["notification_recipient_pid"]:
                raise ValueError("normalized notification recipient does not match payload")
            raw_owner_watch = loads(row["owner_watch_json"], {})
            owner_watch = ObjectTaskOwnerWatch(**raw_owner_watch)
            return ObjectTask(
                task_id=row["task_id"],
                owner_oid=row["owner_oid"],
                creator_pid=row["creator_pid"],
                runner_pid=row["runner_pid"],
                tool=row["tool"],
                tool_id=row["tool_id"],
                status=ObjectTaskStatus(row["status"]),
                notification=notification,
                owner_watch=owner_watch,
                result_oid=row["result_oid"],
                error=row["error"],
                wait=loads(row["wait_json"], {}),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
            )

    def _row_to_agent_rating(self, row: Any) -> AgentRating:
        with _persisted_model_decode(f"agent rating {row['rating_id']}"):
            return AgentRating(
                rating_id=row["rating_id"],
                pid=row["pid"],
                score=int(row["score"]),
                comment=row["comment"],
                rater=row["rater"],
                source=row["source"],
                metadata=loads(row["metadata_json"], {}),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def _llm_call_limit(self, limit: int | None) -> int:
        selected = self.config.llm.call_record_list_limit if limit is None else int(limit)
        if selected <= 0:
            raise ValidationError("llm call limit must be positive")
        if selected > self.config.llm.call_record_hard_limit:
            raise ValidationError(f"llm call limit exceeds hard cap {self.config.llm.call_record_hard_limit}")
        return selected

    def _payload_retention_limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError("payload retention limit must be a positive integer")
        hard_limit = self.config.runtime.payload_retention_page_hard_limit
        if limit > hard_limit:
            raise ValidationError(
                "payload retention limit exceeds configured hard cap: "
                f"{limit} > {hard_limit}"
            )
        return limit

    def _runtime_publication_reconciliation_limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(
                "runtime publication reconciliation limit must be a positive integer"
            )
        hard_limit = self.config.runtime.publication_reconciliation_page_hard_limit
        if limit > hard_limit:
            raise ValidationError(
                "runtime publication reconciliation limit exceeds configured hard cap: "
                f"{limit} > {hard_limit}"
            )
        return limit

    def _operation_recovery_limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(
                "operation recovery limit must be a positive integer"
            )
        hard_limit = self.config.runtime.operation_recovery_page_hard_limit
        if limit > hard_limit:
            raise ValidationError(
                "operation recovery limit exceeds configured hard cap: "
                f"{limit} > {hard_limit}"
            )
        return limit

    def _resource_usage_reservation_recovery_limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(
                "resource usage reservation recovery limit must be a positive integer"
            )
        hard_limit = (
            self.config.runtime.resource_usage_reservation_recovery_page_hard_limit
        )
        if limit > hard_limit:
            raise ValidationError(
                "resource usage reservation recovery limit exceeds configured hard cap: "
                f"{limit} > {hard_limit}"
            )
        return limit

    def _row_to_process_message(self, row: Any) -> ProcessMessage:
        with _persisted_model_decode(f"process message {row['message_id']}"):
            raw_payload = loads(row["payload_json"], {})
            raw_metadata = (
                loads(row["metadata_json"])
                if "metadata_json" in row.keys() and row["metadata_json"] is not None
                else None
            )
            payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
            metadata = _canonical_process_message_metadata(
                raw_metadata if raw_metadata is not None else {}
            )
            return ProcessMessage(
                message_id=row["message_id"],
                sender=row["sender"],
                recipient_pid=row["recipient_pid"],
                kind=ProcessMessageKind(row["kind"]),
                channel=row["channel"] if "channel" in row.keys() else "default",
                correlation_id=row["correlation_id"] if "correlation_id" in row.keys() else None,
                reply_to=row["reply_to"] if "reply_to" in row.keys() else None,
                subject=row["subject"],
                body=row["body"],
                payload=payload,
                metadata=metadata,
                status=ProcessMessageStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                acked_at=row["acked_at"],
            )

    def _row_to_tool_candidate(self, row: Any) -> ToolCandidate:
        with _persisted_model_decode(f"tool candidate {row['candidate_id']}"):
            return ToolCandidate(
                candidate_id=row["candidate_id"],
                pid=row["pid"],
                spec=self._dict_to_tool_spec(loads(row["spec_json"], {})),
                source_code=row["source_code"],
                tests=loads(row["tests_json"], []),
                requested_capabilities=loads(row["requested_capabilities_json"], []),
                status=ToolCandidateStatus(row["status"]),
                validation=loads(row["validation_json"]) if row["validation_json"] else None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                registered_tool_id=row["registered_tool_id"] if "registered_tool_id" in row.keys() else None,
            )

    def _skill_row_metadata(self, row: Any) -> dict[str, Any]:
        return {
            "source_type": row["source_type"],
            "source": row["source"],
            "package_sha256": row["package_sha256"],
            "registered_by": row["registered_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _jsonrpc_endpoint_row_metadata(self, row: Any) -> dict[str, Any]:
        return {
            "registered_by": row["registered_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _mcp_server_row_metadata(self, row: Any) -> dict[str, Any]:
        return {
            "registered_by": row["registered_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _dict_to_jsonrpc_endpoint(self, data: dict[str, Any]) -> JsonRpcEndpointSpec:
        with _persisted_model_decode(
            f"JSON-RPC endpoint {data.get('endpoint_id', '<unknown>') if isinstance(data, dict) else '<unknown>'}"
        ):
            return JsonRpcEndpointSpec(
                schema_version=int(data.get("schema_version", 1)),
                endpoint_id=data["endpoint_id"],
                url=data["url"],
                headers={
                    str(name): JsonRpcHeaderSpec(
                        env=str(value["env"]),
                        prefix=str(value.get("prefix", "")),
                        suffix=str(value.get("suffix", "")),
                    )
                    for name, value in dict(data.get("headers") or {}).items()
                },
                methods=[
                    JsonRpcMethodSpec(
                        method_id=item["method_id"],
                        rpc_method=item["rpc_method"],
                        right=item["right"],
                        rollback_class=item["rollback_class"],
                        rollback_status=item.get("rollback_status"),
                        state_mutation=_persisted_bool(item["state_mutation"], "state_mutation"),
                        information_flow=_persisted_bool(item["information_flow"], "information_flow"),
                        params_schema=dict(item.get("params_schema") or {}),
                        metadata=dict(item.get("metadata") or {}),
                    )
                    for item in list(data.get("methods") or [])
                ],
                timeout_s=float(data["timeout_s"]),
                max_request_bytes=int(data["max_request_bytes"]),
                max_response_bytes=int(data["max_response_bytes"]),
                metadata=dict(data.get("metadata") or {}),
            )

    def _dict_to_mcp_server(self, data: dict[str, Any]) -> McpServerSpec:
        with _persisted_model_decode(
            f"MCP server {data.get('server_id', '<unknown>') if isinstance(data, dict) else '<unknown>'}"
        ):
            stdio_data = data.get("stdio")
            http_data = data.get("http")
            return McpServerSpec(
                schema_version=int(data.get("schema_version", 1)),
                server_id=data["server_id"],
                transport=data["transport"],
                stdio=(
                    McpStdioTransportSpec(
                        command=str(stdio_data["command"]),
                        args=[str(item) for item in list(stdio_data.get("args") or [])],
                        env={str(name): str(value) for name, value in dict(stdio_data.get("env") or {}).items()},
                        cwd=str(stdio_data["cwd"]) if stdio_data.get("cwd") is not None else None,
                    )
                    if isinstance(stdio_data, dict)
                    else None
                ),
                http=(
                    McpHttpTransportSpec(
                        url=str(http_data["url"]),
                        headers={
                            str(name): McpHeaderSpec(
                                env=str(value["env"]),
                                prefix=str(value.get("prefix", "")),
                                suffix=str(value.get("suffix", "")),
                            )
                            for name, value in dict(http_data.get("headers") or {}).items()
                        },
                    )
                    if isinstance(http_data, dict)
                    else None
                ),
                tools=[
                    McpToolSpec(
                        tool_id=item["tool_id"],
                        mcp_name=item["mcp_name"],
                        right=item["right"],
                        rollback_class=item["rollback_class"],
                        rollback_status=item.get("rollback_status"),
                        state_mutation=_persisted_bool(item["state_mutation"], "state_mutation"),
                        information_flow=_persisted_bool(item["information_flow"], "information_flow"),
                        input_schema=dict(item.get("input_schema") or {}),
                        metadata=dict(item.get("metadata") or {}),
                    )
                    for item in list(data.get("tools") or [])
                ],
                timeout_s=float(data["timeout_s"]),
                max_request_bytes=int(data["max_request_bytes"]),
                max_response_bytes=int(data["max_response_bytes"]),
                metadata=dict(data.get("metadata") or {}),
            )

    def _dict_to_skill_package(self, data: dict[str, Any]) -> SkillPackage:
        return SkillPackage(
            schema_version=int(data.get("schema_version", 1)),
            skill_id=data["skill_id"],
            name=data["name"],
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
            version=data.get("version", "v0"),
            license=data.get("license", ""),
            compatibility=data.get("compatibility", ""),
            metadata={str(key): str(value) for key, value in dict(data.get("metadata", {})).items()},
            allowed_tools=list(data.get("allowed_tools", [])),
            actions=[ActionSchema(**item) for item in data.get("actions", [])],
            jit_tools=[JitToolSpec(**item) for item in data.get("jit_tools", [])],
            required_capabilities=list(data.get("required_capabilities", [])),
            resources=[SkillResource(**item) for item in data.get("resources", [])],
            package_sha256=data.get("package_sha256", ""),
            diagnostics=list(data.get("diagnostics", [])),
        )

    def _dict_to_tool_spec(self, data: dict[str, Any]) -> ToolSpec:
        return ToolSpec(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            policy=data.get("policy", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            required_capabilities=data.get("required_capabilities", []),
            side_effects=data.get("side_effects", []),
        )

    def _dict_to_view(self, data: dict[str, Any]) -> MemoryView:
        return MemoryView(
            view_id=data["view_id"],
            owner_pid=data["owner_pid"],
            roots=[self._dict_to_handle(item) for item in data.get("roots", [])],
            filters=[self._dict_to_filter(item) for item in data.get("filters", [])],
            rights_policy=data.get("rights_policy", "inherit"),
            created_from=data.get("created_from"),
            mode=ViewMode(data.get("mode", ViewMode.READ_ONLY.value)),
        )

    def _dict_to_filter(self, data: dict[str, Any]) -> ObjectFilter:
        return ObjectFilter(
            type=ObjectType(data["type"]) if data.get("type") else None,
            tags=data.get("tags", []),
            text=data.get("text"),
        )

    def _dict_to_handle(self, data: dict[str, Any]) -> ObjectHandle:
        return ObjectHandle(
            oid=data["oid"],
            rights=set(data.get("rights", [])),
            capability_id=data["capability_id"],
            expires_at=data.get("expires_at"),
        )
