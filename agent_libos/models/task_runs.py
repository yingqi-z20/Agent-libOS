from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping

from agent_libos.models.base import StrEnum


TASK_RUN_SPEC_SCHEMA_VERSION = 1
TASK_RUN_DISPLAY_TITLE_MAX_CHARS = 256
TASK_RUN_ID_MAX_CHARS = 256
TASK_RUN_PUBLIC_METADATA_MAX_BYTES = 16_384
_TASK_RUN_SIGNED_BIGINT_MAX = 2**63 - 1


class TaskRunRetention(StrEnum):
    PURGE_ON_TERMINAL = "purge_on_terminal"
    PERMANENT = "permanent"


class TaskRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    WAITING_PROCESS = "waiting_process"
    WAITING_MESSAGE = "waiting_message"
    WAITING_TOOL = "waiting_tool"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    FINALIZING = "finalizing"
    NEEDS_ATTENTION = "needs_attention"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRunAction(StrEnum):
    """Host-computed mutations that may be advertised for a Run revision."""

    RUN = "run"
    WAIT = "wait"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    FOLLOW_UP = "follow_up"
    RECOVER = "recover"
    RERUN = "rerun"


TASK_RUN_TERMINAL_STATUSES = frozenset(
    {
        TaskRunStatus.SUCCEEDED,
        TaskRunStatus.FAILED,
        TaskRunStatus.CANCELLED,
    }
)

TASK_RUN_DISPATCHABLE_STATUSES = frozenset(
    {
        TaskRunStatus.QUEUED,
        TaskRunStatus.RUNNING,
        TaskRunStatus.WAITING_HUMAN,
        TaskRunStatus.WAITING_PROCESS,
        TaskRunStatus.WAITING_MESSAGE,
        TaskRunStatus.WAITING_TOOL,
    }
)


class TaskRunRequirementKind(StrEnum):
    INITIAL = "initial"
    FOLLOW_UP = "follow_up"


class TaskRunRequirementStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    WAIVED = "waived"


class TaskRunPayloadRetention(StrEnum):
    PLAINTEXT = "plaintext"
    HASH_ONLY = "hash_only"


class TaskRunLedgerKind(StrEnum):
    REQUIREMENT = "requirement"
    PROCESS = "process"
    LLM_TURN = "llm_turn"
    TOOL_CALL = "tool_call"
    HUMAN_WAIT = "human_wait"
    MESSAGE_WAIT = "message_wait"
    CHECKPOINT = "checkpoint"
    EFFECT = "effect"
    STATUS_TRANSITION = "status_transition"


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_identity(value: object, label: str) -> str:
    selected = _require_text(value, label)
    if len(selected) > TASK_RUN_ID_MAX_CHARS:
        raise ValueError(f"{label} exceeds {TASK_RUN_ID_MAX_CHARS} characters")
    return selected


def _require_display_title(value: object) -> str:
    selected = _require_text(value, "TaskRun display_title")
    if len(selected) > TASK_RUN_DISPLAY_TITLE_MAX_CHARS:
        raise ValueError(
            "TaskRun display_title exceeds "
            f"{TASK_RUN_DISPLAY_TITLE_MAX_CHARS} characters"
        )
    return selected


def _canonical_absolute_timestamp(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    _require_text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an absolute UTC offset")
    # RuntimeStore admission compares this value to canonical UTC wall-clock
    # timestamps.  Persisting arbitrary equivalent offsets as TEXT would make
    # those comparisons timezone-dependent, so the public model normalizes the
    # value before it participates in request hashing or storage.
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _walk_strict_json(value: Any) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        item_type = type(item)
        if item is None or item_type in {str, bool, int}:
            continue
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("TaskRun JSON must not contain non-finite numbers")
            continue
        if item_type not in {dict, list}:
            raise ValueError(
                "TaskRun JSON must contain only JSON values; "
                f"found {item_type.__name__}"
            )
        identity = id(item)
        if identity in seen:
            raise ValueError("TaskRun JSON must not contain container cycles")
        seen.add(identity)
        if item_type is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("TaskRun JSON object keys must be strings")
                pending.append(child)
        else:
            pending.extend(item)


def canonical_task_run_json(value: Any) -> str:
    """Return the integrity-bound canonical JSON representation used by v4."""

    _walk_strict_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("TaskRun value is not strict JSON") from exc


def task_run_payload_sha256(canonical_json: str) -> str:
    if not isinstance(canonical_json, str):
        raise ValueError("TaskRun canonical payload must be text")
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskRunSpecV1:
    goal: Any
    display_title: str
    image_id: str | None = None
    launch_options: dict[str, Any] = field(default_factory=dict)
    authority_manifest_id: str | None = None
    deadline_at: str | None = None
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL
    schema_version: int = TASK_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TaskRunSpecV1 schema_version must be exactly 1")
        _require_display_title(self.display_title)
        if self.image_id is not None:
            _require_identity(self.image_id, "TaskRun image_id")
        if self.authority_manifest_id is not None:
            _require_identity(
                self.authority_manifest_id,
                "TaskRun authority_manifest_id",
            )
        object.__setattr__(
            self,
            "deadline_at",
            _canonical_absolute_timestamp(self.deadline_at, "TaskRun deadline_at"),
        )
        if type(self.launch_options) is not dict:
            raise ValueError("TaskRun launch_options must be a JSON object")
        canonical_task_run_json(self.goal)
        canonical_task_run_json(self.launch_options)
        object.__setattr__(self, "goal", deepcopy(self.goal))
        object.__setattr__(self, "launch_options", deepcopy(self.launch_options))
        object.__setattr__(self, "retention", TaskRunRetention(self.retention))

    def to_mapping(self) -> dict[str, Any]:
        """Return the sole canonical public serialization shape."""

        return {
            "schema_version": self.schema_version,
            "goal": deepcopy(self.goal),
            "display_title": self.display_title,
            "image_id": self.image_id,
            "launch_options": deepcopy(self.launch_options),
            "authority_manifest_id": self.authority_manifest_id,
            "deadline_at": self.deadline_at,
            "retention": self.retention.value,
        }

    def canonical_json(self) -> str:
        return canonical_task_run_json(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskRunSpecV1:
        if not isinstance(value, Mapping):
            raise ValueError("TaskRun spec must be a mapping")
        raw = dict(value)
        # Legacy input aliases remain accepted; canonical output never emits them.
        if "goal" not in raw and "objective" in raw:
            raw["goal"] = raw.pop("objective")
        if "display_title" not in raw and "title" in raw:
            raw["display_title"] = raw.pop("title")
        allowed = {
            "schema_version",
            "goal",
            "display_title",
            "image_id",
            "launch_options",
            "authority_manifest_id",
            "deadline_at",
            "retention",
        }
        extras = sorted(set(raw) - allowed)
        if extras:
            raise ValueError(f"unknown TaskRun spec fields: {extras}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class TaskRunRecord:
    run_id: str
    status: TaskRunStatus
    display_title: str
    image_id: str
    launch_options: dict[str, Any] = field(default_factory=dict)
    authority_manifest_id: str | None = None
    deadline_at: str | None = None
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL
    revision: int = 0
    runtime_epoch: int = 0
    root_pid: str | None = None
    active_pid: str | None = None
    pause_generation: int = 0
    cancel_generation: int = 0
    binding_hash: str | None = None
    blockers: tuple[dict[str, Any], ...] = ()
    requirement_count: int = 0
    satisfied_requirement_count: int = 0
    step_count: int = 0
    completed_step_count: int = 0
    result_ref: str | None = None
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    finalized_at: str | None = None
    payloads_purged_at: str | None = None

    def __post_init__(self) -> None:
        _require_identity(self.run_id, "TaskRun run_id")
        _require_display_title(self.display_title)
        _require_identity(self.image_id, "TaskRun image_id")
        if type(self.launch_options) is not dict:
            raise ValueError("TaskRun launch_options must be a JSON object")
        canonical_task_run_json(self.launch_options)
        object.__setattr__(self, "launch_options", deepcopy(self.launch_options))
        object.__setattr__(self, "status", TaskRunStatus(self.status))
        object.__setattr__(self, "retention", TaskRunRetention(self.retention))
        for name in (
            "revision",
            "runtime_epoch",
            "pause_generation",
            "cancel_generation",
            "requirement_count",
            "satisfied_requirement_count",
            "step_count",
            "completed_step_count",
        ):
            _require_nonnegative_int(getattr(self, name), f"TaskRun {name}")
        if self.satisfied_requirement_count > self.requirement_count:
            raise ValueError("TaskRun satisfied requirements exceed total")
        if self.completed_step_count > self.step_count:
            raise ValueError("TaskRun completed steps exceed total")
        object.__setattr__(
            self,
            "deadline_at",
            _canonical_absolute_timestamp(self.deadline_at, "TaskRun deadline_at"),
        )
        for name in ("root_pid", "active_pid", "authority_manifest_id"):
            value = getattr(self, name)
            if value is not None:
                _require_text(value, f"TaskRun {name}")
        if self.binding_hash is not None:
            _require_text(self.binding_hash, "TaskRun binding_hash")
        if not isinstance(self.blockers, tuple):
            object.__setattr__(self, "blockers", tuple(self.blockers))
        blockers_json = canonical_task_run_json(list(self.blockers))
        if len(blockers_json.encode("utf-8")) > TASK_RUN_PUBLIC_METADATA_MAX_BYTES:
            raise ValueError("TaskRun blockers exceed the public metadata limit")

    @classmethod
    def from_spec(
        cls,
        run_id: str,
        spec: TaskRunSpecV1,
        *,
        status: TaskRunStatus = TaskRunStatus.QUEUED,
        image_id: str | None = None,
        **values: Any,
    ) -> TaskRunRecord:
        resolved_image_id = image_id or spec.image_id
        if resolved_image_id is None:
            raise ValueError("TaskRunRecord requires a resolved image_id")
        return cls(
            run_id=run_id,
            status=status,
            display_title=spec.display_title,
            image_id=resolved_image_id,
            launch_options=spec.launch_options,
            authority_manifest_id=spec.authority_manifest_id,
            deadline_at=spec.deadline_at,
            retention=spec.retention,
            **values,
        )


@dataclass(frozen=True, slots=True)
class TaskRunSummary:
    run_id: str
    revision: int
    status: TaskRunStatus
    display_title: str
    root_pid: str | None = None
    active_pid: str | None = None
    step_count: int = 0
    completed_step_count: int = 0
    requirement_count: int = 0
    satisfied_requirement_count: int = 0
    blockers: tuple[dict[str, Any], ...] = ()
    allowed_actions: tuple[TaskRunAction, ...] = ()
    result_ref: str | None = None
    payloads_purged: bool = False
    retention: TaskRunRetention = TaskRunRetention.PURGE_ON_TERMINAL
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    schema_version: int = TASK_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identity(self.run_id, "TaskRun summary run_id")
        _require_display_title(self.display_title)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TaskRunSummary schema_version must be exactly 1")
        object.__setattr__(self, "status", TaskRunStatus(self.status))
        object.__setattr__(self, "retention", TaskRunRetention(self.retention))
        if type(self.payloads_purged) is not bool:
            raise ValueError("TaskRun summary payloads_purged must be boolean")
        for name in (
            "revision",
            "step_count",
            "completed_step_count",
            "requirement_count",
            "satisfied_requirement_count",
        ):
            value = _require_nonnegative_int(
                getattr(self, name),
                f"TaskRun summary {name}",
            )
            if value > _TASK_RUN_SIGNED_BIGINT_MAX:
                raise ValueError(
                    f"TaskRun summary {name} exceeds signed BIGINT"
                )
        if not isinstance(self.blockers, tuple):
            object.__setattr__(self, "blockers", tuple(self.blockers))
        if not isinstance(self.allowed_actions, tuple):
            object.__setattr__(self, "allowed_actions", tuple(self.allowed_actions))
        selected_actions = tuple(
            TaskRunAction(action) for action in self.allowed_actions
        )
        if len(set(selected_actions)) != len(selected_actions):
            raise ValueError("TaskRun summary allowed_actions contain duplicates")
        object.__setattr__(self, "allowed_actions", selected_actions)
        blockers_json = canonical_task_run_json(list(self.blockers))
        if len(blockers_json.encode("utf-8")) > TASK_RUN_PUBLIC_METADATA_MAX_BYTES:
            raise ValueError("TaskRun summary blockers exceed the public metadata limit")
        if self.completed_step_count > self.step_count:
            raise ValueError("TaskRun completed steps cannot exceed total steps")
        if self.satisfied_requirement_count > self.requirement_count:
            raise ValueError(
                "TaskRun satisfied requirements cannot exceed total requirements"
            )


@dataclass(frozen=True, slots=True)
class TaskRunRequirement:
    requirement_id: str
    run_id: str
    ordinal: int
    kind: TaskRunRequirementKind
    status: TaskRunRequirementStatus
    payload_id: str
    requirement_sha256: str
    label: str
    created_by: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    waived_by: str | None = None
    waiver_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "run_id",
            "payload_id",
            "label",
            "created_by",
            "created_at",
            "updated_at",
        ):
            _require_text(getattr(self, name), f"TaskRun requirement {name}")
        _require_nonnegative_int(self.ordinal, "TaskRun requirement ordinal")
        object.__setattr__(self, "kind", TaskRunRequirementKind(self.kind))
        object.__setattr__(self, "status", TaskRunRequirementStatus(self.status))
        _validate_sha256(self.requirement_sha256, "TaskRun requirement hash")
        if (self.waived_by is None) != (self.waiver_reason is None):
            raise ValueError("TaskRun waiver actor and reason must be set together")
        if self.status is TaskRunRequirementStatus.WAIVED:
            _require_text(self.waived_by, "TaskRun waived_by")
            _require_text(self.waiver_reason, "TaskRun waiver_reason")
        elif self.waived_by is not None:
            raise ValueError("only a waived TaskRun requirement may retain waiver data")


@dataclass(frozen=True, slots=True)
class TaskRunPayload:
    payload_id: str
    run_id: str
    role: str
    label: str
    canonical_json: str | None
    sha256: str
    size_bytes: int
    retention_state: TaskRunPayloadRetention
    created_at: str
    updated_at: str
    purged_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("payload_id", "run_id", "role", "label", "created_at", "updated_at"):
            _require_text(getattr(self, name), f"TaskRun payload {name}")
        _validate_sha256(self.sha256, "TaskRun payload sha256")
        _require_nonnegative_int(self.size_bytes, "TaskRun payload size_bytes")
        object.__setattr__(
            self,
            "retention_state",
            TaskRunPayloadRetention(self.retention_state),
        )
        if self.retention_state is TaskRunPayloadRetention.PLAINTEXT:
            if self.canonical_json is None:
                raise ValueError("plaintext TaskRun payload requires canonical_json")
            encoded = self.canonical_json.encode("utf-8")
            if len(encoded) != self.size_bytes:
                raise ValueError("TaskRun payload byte size mismatch")
            if task_run_payload_sha256(self.canonical_json) != self.sha256:
                raise ValueError("TaskRun payload SHA-256 mismatch")
            if canonical_task_run_json(json.loads(self.canonical_json)) != self.canonical_json:
                raise ValueError("TaskRun payload is not canonical JSON")
            if self.purged_at is not None:
                raise ValueError("plaintext TaskRun payload cannot have purged_at")
        else:
            if self.canonical_json is not None or self.purged_at is None:
                raise ValueError("hash-only TaskRun payload must be purged")

    @classmethod
    def plaintext(
        cls,
        *,
        payload_id: str,
        run_id: str,
        role: str,
        label: str,
        value: Any,
        created_at: str,
        updated_at: str | None = None,
    ) -> TaskRunPayload:
        encoded = canonical_task_run_json(value)
        return cls(
            payload_id=payload_id,
            run_id=run_id,
            role=role,
            label=label,
            canonical_json=encoded,
            sha256=task_run_payload_sha256(encoded),
            size_bytes=len(encoded.encode("utf-8")),
            retention_state=TaskRunPayloadRetention.PLAINTEXT,
            created_at=created_at,
            updated_at=updated_at or created_at,
        )


@dataclass(frozen=True, slots=True)
class TaskRunResumePoint:
    run_id: str
    pid: str
    task_run_epoch: int
    process_revision: int
    context_generation: str
    safe_point_seq: int
    binding_hash: str
    image_binding_hash: str
    tool_binding_hash: str
    provider_binding_hash: str
    transcript_payload_id: str
    integrity_sha256: str
    created_at: str
    updated_at: str
    summary_payload_id: str | None = None
    pending_action_payload_id: str | None = None
    last_effect_seq: int = 0
    complete: bool = True

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "pid",
            "binding_hash",
            "image_binding_hash",
            "tool_binding_hash",
            "provider_binding_hash",
            "transcript_payload_id",
            "context_generation",
            "created_at",
            "updated_at",
        ):
            _require_text(getattr(self, name), f"TaskRun resume point {name}")
        if type(self.task_run_epoch) is not int or self.task_run_epoch <= 0:
            raise ValueError("TaskRun resume point task_run_epoch must be positive")
        for name in ("process_revision", "safe_point_seq", "last_effect_seq"):
            _require_nonnegative_int(getattr(self, name), f"TaskRun resume point {name}")
        _validate_sha256(self.integrity_sha256, "TaskRun resume point integrity")
        if type(self.complete) is not bool:
            raise ValueError("TaskRun resume point complete must be boolean")


@dataclass(frozen=True, slots=True)
class TaskRunCommand:
    command_id: str
    client_request_id: str | None
    run_id: str
    command_kind: str
    request_hash: str
    result: dict[str, Any]
    result_revision: int
    created_at: str

    def __post_init__(self) -> None:
        for name in (
            "command_id",
            "run_id",
            "command_kind",
            "created_at",
        ):
            _require_text(getattr(self, name), f"TaskRun command {name}")
        if self.client_request_id is not None:
            _require_text(
                self.client_request_id,
                "TaskRun command client_request_id",
            )
        _validate_sha256(self.request_hash, "TaskRun command request hash")
        _require_nonnegative_int(self.result_revision, "TaskRun command result revision")
        if self.result_revision > _TASK_RUN_SIGNED_BIGINT_MAX:
            raise ValueError(
                "TaskRun command result revision exceeds signed BIGINT"
            )
        if type(self.result) is not dict:
            raise ValueError("TaskRun command result must be an object")
        canonical_task_run_json(self.result)


@dataclass(frozen=True, slots=True)
class TaskRunLedgerItem:
    item_id: str
    run_id: str
    seq: int
    kind: TaskRunLedgerKind
    status: str
    label: str
    occurred_at: str
    requirement_id: str | None = None
    pid: str | None = None
    operation_id: str | None = None
    effect_id: str | None = None
    human_request_id: str | None = None
    llm_call_id: str | None = None
    checkpoint_id: str | None = None
    object_task_id: str | None = None
    payload_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TASK_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("item_id", "run_id", "status", "label", "occurred_at"):
            _require_text(getattr(self, name), f"TaskRun ledger {name}")
        _require_nonnegative_int(self.seq, "TaskRun ledger seq")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TaskRunLedgerItem schema_version must be exactly 1")
        object.__setattr__(self, "kind", TaskRunLedgerKind(self.kind))
        if type(self.metadata) is not dict:
            raise ValueError("TaskRun ledger metadata must be an object")
        metadata_json = canonical_task_run_json(self.metadata)
        if len(metadata_json.encode("utf-8")) > TASK_RUN_PUBLIC_METADATA_MAX_BYTES:
            raise ValueError("TaskRun ledger metadata exceed the public metadata limit")


@dataclass(frozen=True, slots=True)
class TaskRunLink:
    link_id: str
    run_id: str
    ledger_seq: int
    evidence_type: str
    evidence_id: str
    role: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "link_id",
            "run_id",
            "evidence_type",
            "evidence_id",
            "role",
            "created_at",
        ):
            _require_text(getattr(self, name), f"TaskRun link {name}")
        _require_nonnegative_int(self.ledger_seq, "TaskRun link ledger_seq")
        metadata_json = canonical_task_run_json(self.metadata)
        if len(metadata_json.encode("utf-8")) > TASK_RUN_PUBLIC_METADATA_MAX_BYTES:
            raise ValueError("TaskRun link metadata exceed the public metadata limit")


@dataclass(frozen=True, order=True, slots=True)
class TaskRunCursor:
    created_at: str
    run_id: str

    def __post_init__(self) -> None:
        _require_text(self.created_at, "TaskRun cursor created_at")
        _require_text(self.run_id, "TaskRun cursor run_id")


@dataclass(frozen=True, slots=True)
class TaskRunPage:
    records: tuple[TaskRunRecord, ...]
    next_cursor: TaskRunCursor | None = None
    schema_version: int = TASK_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise ValueError("TaskRun page records must be a tuple")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TaskRunPage schema_version must be exactly 1")
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty TaskRun page cannot have a cursor")


@dataclass(frozen=True, order=True, slots=True)
class TaskRunLedgerCursor:
    seq: int
    item_id: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seq, "TaskRun ledger cursor seq")
        _require_text(self.item_id, "TaskRun ledger cursor item_id")


@dataclass(frozen=True, slots=True)
class TaskRunLedgerPage:
    records: tuple[TaskRunLedgerItem, ...]
    next_cursor: TaskRunLedgerCursor | None = None
    schema_version: int = TASK_RUN_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise ValueError("TaskRun ledger page records must be a tuple")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TaskRunLedgerPage schema_version must be exactly 1")
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty TaskRun ledger page cannot have a cursor")


def _validate_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
