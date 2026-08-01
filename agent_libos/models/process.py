from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.base import CapabilityID, CheckpointID, EventID, OID, PID, StrEnum
from agent_libos.models.memory import MemoryView, ObjectHandle
from agent_libos.models.process_state import ProcessOutcome, ProcessWaitState


class ProcessStatus(StrEnum):
    CREATED = "created"
    RUNNABLE = "runnable"
    RUNNING = "running"
    WAITING_EVENT = "waiting_event"
    WAITING_TOOL = "waiting_tool"
    WAITING_HUMAN = "waiting_human"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    EXITED = "exited"
    FAILED = "failed"
    KILLED = "killed"


class ForkMode(StrEnum):
    COPY = "copy"
    RESTRICTED = "restricted"
    SPECULATIVE = "speculative"
    WORKER = "worker"


class ProcessSignal(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    INTERRUPT = "interrupt"
    TERMINATE = "terminate"


_CONTINUOUS_BUDGET_FIELDS = {
    "max_runtime_seconds",
    "max_subprocess_wall_seconds",
    "max_subprocess_cpu_seconds",
}

_CONTINUOUS_USAGE_FIELDS = {
    "runtime_seconds",
    "subprocess_wall_seconds",
    "subprocess_cpu_seconds",
}


@dataclass
class ResourceBudget:
    max_tool_calls: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_tool_calls)
    max_child_processes: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_child_processes)
    max_runtime_seconds: float | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_runtime_seconds)
    max_context_materialization_tokens: int = field(
        default_factory=lambda: DEFAULT_CONFIG.process.max_context_materialization_tokens
    )
    max_context_materialization_total_tokens: int | None = field(
        default_factory=lambda: DEFAULT_CONFIG.process.max_context_materialization_total_tokens
    )
    max_llm_calls: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_llm_calls)
    max_llm_total_tokens: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_llm_total_tokens)
    max_subprocess_wall_seconds: float | None = field(
        default_factory=lambda: DEFAULT_CONFIG.process.max_subprocess_wall_seconds
    )
    max_subprocess_cpu_seconds: float | None = field(
        default_factory=lambda: DEFAULT_CONFIG.process.max_subprocess_cpu_seconds
    )
    max_subprocess_memory_bytes: int | None = field(
        default_factory=lambda: DEFAULT_CONFIG.process.max_subprocess_memory_bytes
    )
    max_external_read_bytes: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_external_read_bytes)
    max_external_write_bytes: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_external_write_bytes)
    max_jsonrpc_bytes: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_jsonrpc_bytes)
    max_mcp_bytes: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_mcp_bytes)
    max_deno_syscalls: int | None = field(default_factory=lambda: DEFAULT_CONFIG.process.max_deno_syscalls)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            allow_none = item.name != "max_context_materialization_tokens"
            _validate_resource_number(
                item.name,
                value,
                allow_none=allow_none,
                require_integer=item.name not in _CONTINUOUS_BUDGET_FIELDS,
            )


@dataclass
class ResourceUsage:
    runtime_seconds: float = 0.0
    tool_calls: int = 0
    child_processes: int = 0
    context_materialized_tokens: int = 0
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    subprocess_wall_seconds: float = 0.0
    subprocess_cpu_seconds: float = 0.0
    subprocess_peak_memory_bytes: int = 0
    external_read_bytes: int = 0
    external_write_bytes: int = 0
    jsonrpc_request_bytes: int = 0
    jsonrpc_response_bytes: int = 0
    mcp_request_bytes: int = 0
    mcp_response_bytes: int = 0
    deno_syscalls: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for item in fields(self):
            _validate_resource_number(
                item.name,
                getattr(self, item.name),
                allow_none=False,
                require_integer=item.name not in _CONTINUOUS_USAGE_FIELDS,
            )


@dataclass
class ResourceReservation:
    parent_pid: PID
    child_pid: PID
    reserved: dict[str, float]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        for key, value in self.reserved.items():
            _validate_resource_number(key, value, allow_none=False, require_integer=False)


class ResourceUsageReservationStatus(StrEnum):
    ACTIVE = "active"
    SETTLED = "settled"
    RELEASED = "released"
    CHARGED_MAXIMUM = "charged_maximum"


@dataclass(frozen=True)
class ResourceUsageReservation:
    """Typed durable envelope for one provider-side resource effect."""

    reservation_id: str
    pid: PID
    usage: ResourceUsage
    status: ResourceUsageReservationStatus
    reserved_by: str
    reason: str
    settled_usage: ResourceUsage | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "reservation_id",
            "pid",
            "reserved_by",
            "reason",
            "created_at",
            "updated_at",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, order=True, slots=True)
class ResourceUsageReservationCursor:
    """Stable keyset cursor for bounded usage-reservation recovery."""

    created_at: str
    reservation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError(
                "resource usage reservation cursor created_at must not be empty"
            )
        if not isinstance(self.reservation_id, str) or not self.reservation_id:
            raise ValueError(
                "resource usage reservation cursor reservation_id must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ResourceUsageReservationPage:
    """One hard-bounded page of active usage reservations."""

    records: tuple[ResourceUsageReservation, ...]
    next_cursor: ResourceUsageReservationCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not isinstance(
            self.next_cursor,
            ResourceUsageReservationCursor,
        ):
            raise ValueError(
                "resource usage reservation page cursor has an invalid type"
            )
        if self.next_cursor is not None and not self.records:
            raise ValueError(
                "empty resource usage reservation page cannot have a cursor"
            )


@dataclass(frozen=True, slots=True)
class ResourceUsageReservationRecoverySummary:
    """Bounded diagnostics for a fully processed reservation backlog."""

    total_count: int
    sample_reservation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("resource usage recovery total_count must be non-negative")
        if len(self.sample_reservation_ids) > self.total_count:
            raise ValueError("resource usage recovery sample exceeds total_count")
        if any(
            not isinstance(reservation_id, str) or not reservation_id
            for reservation_id in self.sample_reservation_ids
        ):
            raise ValueError("resource usage recovery sample ids must be non-empty")

    @property
    def truncated(self) -> bool:
        return len(self.sample_reservation_ids) < self.total_count

    def __len__(self) -> int:
        return self.total_count


@dataclass(frozen=True, slots=True)
class StaleExecutionRecoverySummary:
    """Bounded diagnostics for stale process-execution recovery."""

    total_count: int
    sample_pids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("stale execution recovery total_count must be non-negative")
        if len(self.sample_pids) > self.total_count:
            raise ValueError("stale execution recovery sample exceeds total")
        if any(not isinstance(pid, str) or not pid for pid in self.sample_pids):
            raise ValueError("stale execution recovery sample PIDs must not be empty")

    @property
    def truncated(self) -> bool:
        return len(self.sample_pids) < self.total_count

    def __len__(self) -> int:
        return self.total_count


def _validate_resource_number(
    name: str,
    value: Any,
    *,
    allow_none: bool,
    require_integer: bool,
) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    if require_integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


PROMPT_MODE_IMAGE_ONLY = "image_only"
PROMPT_MODE_MINIMAL_RUNTIME = "minimal_runtime"
PROMPT_MODE_LIBOS_DEFAULT = "libos_default"
PROMPT_MODES = frozenset(
    {
        PROMPT_MODE_IMAGE_ONLY,
        PROMPT_MODE_MINIMAL_RUNTIME,
        PROMPT_MODE_LIBOS_DEFAULT,
    }
)

JIT_TOOL_EXPOSURE_DIRECT = "direct"
JIT_TOOL_EXPOSURE_MULTIPLEXED = "multiplexed"
JIT_TOOL_EXPOSURES = frozenset(
    {
        JIT_TOOL_EXPOSURE_DIRECT,
        JIT_TOOL_EXPOSURE_MULTIPLEXED,
    }
)


@dataclass(frozen=True)
class AgentImage:
    image_id: str
    name: str
    version: str = "v0"
    system_prompt: str = ""
    prompt_mode: str = PROMPT_MODE_IMAGE_ONLY
    jit_tool_exposure: str = JIT_TOOL_EXPOSURE_DIRECT
    planner: dict[str, Any] = field(default_factory=dict)
    action_schema: dict[str, Any] = field(default_factory=dict)
    default_skills: list[str] = field(default_factory=list)
    default_tools: list[str] = field(default_factory=list)
    context_policy: str = "plan_first"
    safety_profile: str = "default"
    llm_profile_id: str | None = None
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    required_modules: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None
    boot: dict[str, Any] = field(default_factory=lambda: {"kind": "fresh"})


@dataclass
class AgentProcess:
    pid: PID
    parent_pid: PID | None
    image_id: str
    status: ProcessStatus
    goal_oid: OID | None
    memory_view: MemoryView | None
    capabilities: list[CapabilityID]
    loaded_skills: dict[str, Any]
    tool_table: dict[str, str]
    event_cursor: EventID | None
    checkpoint_head: CheckpointID | None
    resource_budget: ResourceBudget
    resource_usage: ResourceUsage
    created_at: str
    updated_at: str
    working_directory: str = "."
    status_message: str | None = None
    wait_state: ProcessWaitState | None = None
    outcome: ProcessOutcome | None = None
    state_generation: int = 0
    llm_profile_id: str = field(default_factory=lambda: DEFAULT_CONFIG.llm.default_profile_id)
    model_tool_table: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    execution_generation: int = 0
    execution_owner_id: str | None = None
    execution_lease_id: str | None = None
    task_run_id: str | None = None
    task_run_epoch: int | None = None
    task_run_role: str | None = None


@dataclass(frozen=True, order=True, slots=True)
class ProcessCursor:
    """Stable keyset cursor for bounded process recovery scans."""

    created_at: str
    pid: str

    def __post_init__(self) -> None:
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("process cursor created_at must not be empty")
        if not isinstance(self.pid, str) or not self.pid:
            raise ValueError("process cursor pid must not be empty")


@dataclass(frozen=True, slots=True)
class ProcessPage:
    """One hard-bounded page of process rows."""

    records: tuple[AgentProcess, ...]
    next_cursor: ProcessCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty process page cannot have a cursor")


@dataclass(frozen=True, slots=True)
class ProcessToolBindingRecord:
    """One durable binding from the indexed JIT-recovery projection."""

    pid: str
    tool_name: str
    tool_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.pid, str) or not self.pid or "\x00" in self.pid:
            raise ValueError("process tool binding PID must not be empty")
        for field_name, value in (
            ("tool name", self.tool_name),
            ("tool identity", self.tool_id),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(
                    f"process tool binding {field_name} must not be empty"
                )


@dataclass(frozen=True, order=True, slots=True)
class ProcessToolBindingCursor:
    """Stable global keyset for the normalized JIT binding projection."""

    pid: str
    tool_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.pid, str) or not self.pid or "\x00" in self.pid:
            raise ValueError("process tool binding cursor PID must not be empty")
        if (
            not isinstance(self.tool_name, str)
            or not self.tool_name
            or "\x00" in self.tool_name
        ):
            raise ValueError(
                "process tool binding cursor tool name must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ProcessToolBindingPage:
    """One hard-bounded page of normalized durable JIT bindings."""

    records: tuple[ProcessToolBindingRecord, ...]
    next_cursor: ProcessToolBindingCursor | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not isinstance(
            self.next_cursor,
            ProcessToolBindingCursor,
        ):
            raise ValueError("process tool binding page cursor has an invalid type")
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty process tool binding page cannot have a cursor")
        keys = [
            ProcessToolBindingCursor(record.pid, record.tool_name)
            for record in self.records
        ]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "process tool binding recovery page must be strictly ordered"
            )
        if self.next_cursor is not None and self.next_cursor != keys[-1]:
            raise ValueError(
                "process tool binding recovery cursor must match the last record"
            )


@dataclass(frozen=True)
class ProcessExecutionToken:
    """Fences writes produced by one claimed scheduler quantum."""

    pid: PID
    generation: int
    owner_id: str
    lease_id: str
    task_run_epoch: int | None = None


@dataclass(frozen=True, order=True, slots=True)
class ProcessRestoreEpoch:
    """One process's durable restore high-water floor or reserved epoch."""

    pid: PID
    revision: int
    execution_generation: int
    state_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.pid, str) or not self.pid or "\x00" in self.pid:
            raise ValueError("process restore epoch PID must not be empty")
        for field_name in (
            "revision",
            "execution_generation",
            "state_generation",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"process restore epoch {field_name} must be a non-negative integer"
                )


@dataclass
class ProcessResult:
    pid: PID
    status: ProcessStatus
    result: ObjectHandle | None = None
    message: str | None = None
    wait_state: ProcessWaitState | None = None
    outcome: ProcessOutcome | None = None
    state_generation: int = 0
