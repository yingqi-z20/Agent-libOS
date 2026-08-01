from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from typing import Any, ClassVar, Mapping, TypeAlias

from agent_libos.models.exceptions import ValidationError


PROCESS_STATE_SCHEMA_VERSION = 1


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"process state {field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name)


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(
            f"process state {field_name} must be a non-negative integer"
        )
    return value


def _positive_integer(value: Any, field_name: str) -> int:
    selected = _nonnegative_integer(value, field_name)
    if selected == 0:
        raise ValidationError(f"process state {field_name} must be positive")
    return selected


def _sha256(value: Any, field_name: str) -> str:
    selected = _non_empty_string(value, field_name)
    if re.fullmatch(r"[0-9a-f]{64}", selected) is None:
        raise ValidationError(
            f"process state {field_name} must be a canonical SHA-256"
        )
    return selected


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field_name)


def _canonical_mapping(
    value: Mapping[str, Any],
    *,
    kind: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if any(type(key) is not str for key in value):
        raise ValidationError(
            f"process state {kind} contains a non-string JSON object key"
        )
    selected = {key: deepcopy(item) for key, item in value.items()}
    expected = {"schema_version", "kind", *fields}
    if set(selected) != expected:
        raise ValidationError(
            f"process state {kind} is not canonical; "
            f"missing={sorted(expected - set(selected))}, "
            f"unknown={sorted(set(selected) - expected)}"
        )
    schema_version = selected["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PROCESS_STATE_SCHEMA_VERSION
    ):
        raise ValidationError(
            "unsupported process state schema_version: "
            f"{schema_version!r}; expected {PROCESS_STATE_SCHEMA_VERSION}"
        )
    if selected["kind"] != kind:
        raise ValidationError(
            f"process state kind mismatch: {selected['kind']!r}; expected {kind!r}"
        )
    return selected


def _strict_json_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> Any:
    """Return an exact JSON tree without coercing caller-supplied values."""

    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValidationError(
                f"process state message filters contain a non-finite number at {path}"
            )
        return value
    if value_type is dict:
        identity = id(value)
        if identity in active_containers:
            raise ValidationError(
                f"process state message filters contain a cycle at {path}"
            )
        active_containers.add(identity)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValidationError(
                        "process state message filters contain a non-string "
                        f"JSON object key at {path}"
                    )
                normalized[key] = _strict_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                )
            return normalized
        finally:
            active_containers.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active_containers:
            raise ValidationError(
                f"process state message filters contain a cycle at {path}"
            )
        active_containers.add(identity)
        try:
            return [
                _strict_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise ValidationError(
        "process state message filters contain non-JSON value "
        f"{value_type.__name__} at {path}"
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in pairs:
        if key in selected:
            raise ValidationError(
                f"persisted process state contains duplicate JSON object key: {key!r}"
            )
        selected[key] = value
    return selected


def _reject_non_finite_json_constant(value: str) -> Any:
    raise ValidationError(
        f"persisted process state contains non-finite JSON number: {value}"
    )


def _load_strict_json(value: str, *, label: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_finite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"persisted process {label} is not valid JSON") from exc


@dataclass(frozen=True, slots=True)
class ChildProcessWait:
    kind: ClassVar[str] = "child"
    child_pid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_pid", _non_empty_string(self.child_pid, "child_pid"))


@dataclass(frozen=True, slots=True)
class MessageProcessWait:
    kind: ClassVar[str] = "message"
    filters: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.filters, dict):
            raise ValidationError("process state message filters must be an object")
        object.__setattr__(
            self,
            "filters",
            _strict_json_value(
                self.filters,
                path="$",
                active_containers=set(),
            ),
        )


@dataclass(frozen=True, slots=True)
class HumanProcessWait:
    kind: ClassVar[str] = "human"
    request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        request_ids = tuple(
            _non_empty_string(request_id, "request_ids[]")
            for request_id in self.request_ids
        )
        if not request_ids:
            raise ValidationError("process state human request_ids must not be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValidationError("process state human request_ids must not contain duplicates")
        object.__setattr__(self, "request_ids", request_ids)


@dataclass(frozen=True, slots=True)
class ToolProcessWait:
    kind: ClassVar[str] = "tool"
    operation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _non_empty_string(self.operation_id, "operation_id"))


@dataclass(frozen=True, slots=True)
class PausedProcessWait:
    kind: ClassVar[str] = "paused"
    reason_oid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_oid", _optional_string(self.reason_oid, "reason_oid"))


@dataclass(frozen=True, slots=True)
class HostResumeProcessWait:
    kind: ClassVar[str] = "host_resume"
    reason_oid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_oid", _non_empty_string(self.reason_oid, "reason_oid"))


@dataclass(frozen=True, slots=True)
class StaleExecutionProcessWait:
    """Store-only receipt for one fenced stale-execution takeover.

    Execution owner and lease identities are retained only as canonical hashes:
    the compatibility/API projection must never expose a former worker token.
    TaskRun safe-point and binding evidence deliberately remain in their
    authoritative tables instead of being copied into this receipt.
    """

    kind: ClassVar[str] = "stale_execution"
    pid: str
    recovered_by_owner_sha256: str
    prior_owner_sha256: str | None
    prior_lease_sha256: str | None
    prior_execution_generation: int
    recovered_execution_generation: int
    recovered_state_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pid", _non_empty_string(self.pid, "pid"))
        object.__setattr__(
            self,
            "recovered_by_owner_sha256",
            _sha256(
                self.recovered_by_owner_sha256,
                "recovered_by_owner_sha256",
            ),
        )
        object.__setattr__(
            self,
            "prior_owner_sha256",
            _optional_sha256(self.prior_owner_sha256, "prior_owner_sha256"),
        )
        object.__setattr__(
            self,
            "prior_lease_sha256",
            _optional_sha256(self.prior_lease_sha256, "prior_lease_sha256"),
        )
        object.__setattr__(
            self,
            "prior_execution_generation",
            _nonnegative_integer(
                self.prior_execution_generation,
                "prior_execution_generation",
            ),
        )
        object.__setattr__(
            self,
            "recovered_execution_generation",
            _positive_integer(
                self.recovered_execution_generation,
                "recovered_execution_generation",
            ),
        )
        object.__setattr__(
            self,
            "recovered_state_generation",
            _positive_integer(
                self.recovered_state_generation,
                "recovered_state_generation",
            ),
        )


ProcessWaitState: TypeAlias = (
    ChildProcessWait
    | MessageProcessWait
    | HumanProcessWait
    | ToolProcessWait
    | PausedProcessWait
    | HostResumeProcessWait
    | StaleExecutionProcessWait
)


@dataclass(frozen=True, slots=True)
class ExitedProcessOutcome:
    kind: ClassVar[str] = "exited"
    result_oid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_oid", _optional_string(self.result_oid, "result_oid"))


@dataclass(frozen=True, slots=True)
class FailedProcessOutcome:
    kind: ClassVar[str] = "failed"
    result_oid: str | None = None
    code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_oid", _optional_string(self.result_oid, "result_oid"))
        object.__setattr__(self, "code", _optional_string(self.code, "code"))


@dataclass(frozen=True, slots=True)
class KilledProcessOutcome:
    kind: ClassVar[str] = "killed"
    reason_oid: str | None = None
    code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_oid", _optional_string(self.reason_oid, "reason_oid"))
        object.__setattr__(self, "code", _optional_string(self.code, "code"))


ProcessOutcome: TypeAlias = ExitedProcessOutcome | FailedProcessOutcome | KilledProcessOutcome


def validate_process_state_fields(
    status: str,
    wait_state: ProcessWaitState | None,
    outcome: ProcessOutcome | None,
) -> None:
    """Validate the persisted status/wait/outcome product type."""

    selected_status = str(status)
    if selected_status == "waiting_event":
        if not isinstance(wait_state, (ChildProcessWait, MessageProcessWait)):
            raise ValidationError(
                "waiting_event requires a child or message process wait state"
            )
    elif selected_status == "waiting_human":
        if not isinstance(wait_state, HumanProcessWait):
            raise ValidationError("waiting_human requires a human process wait state")
    elif selected_status == "waiting_tool":
        if not isinstance(wait_state, ToolProcessWait):
            raise ValidationError("waiting_tool requires a tool process wait state")
    elif selected_status == "paused":
        if not isinstance(
            wait_state,
            (PausedProcessWait, HostResumeProcessWait, StaleExecutionProcessWait),
        ):
            raise ValidationError("paused requires a paused process wait state")
    elif wait_state is not None:
        raise ValidationError(f"{selected_status} processes cannot carry a wait state")

    if selected_status == "exited":
        if not isinstance(outcome, ExitedProcessOutcome):
            raise ValidationError("exited requires an exited process outcome")
    elif selected_status == "failed":
        if not isinstance(outcome, FailedProcessOutcome):
            raise ValidationError("failed requires a failed process outcome")
    elif selected_status == "killed":
        if not isinstance(outcome, KilledProcessOutcome):
            raise ValidationError("killed requires a killed process outcome")
    elif outcome is not None:
        raise ValidationError(
            f"{selected_status} processes cannot carry a terminal outcome"
        )


def process_wait_state_to_mapping(value: ProcessWaitState | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ChildProcessWait):
        fields = {"child_pid": value.child_pid}
    elif isinstance(value, MessageProcessWait):
        fields = {"filters": deepcopy(value.filters)}
    elif isinstance(value, HumanProcessWait):
        fields = {"request_ids": list(value.request_ids)}
    elif isinstance(value, ToolProcessWait):
        fields = {"operation_id": value.operation_id}
    elif isinstance(value, (PausedProcessWait, HostResumeProcessWait)):
        fields = {"reason_oid": value.reason_oid}
    elif isinstance(value, StaleExecutionProcessWait):
        fields = {
            "pid": value.pid,
            "recovered_by_owner_sha256": value.recovered_by_owner_sha256,
            "prior_owner_sha256": value.prior_owner_sha256,
            "prior_lease_sha256": value.prior_lease_sha256,
            "prior_execution_generation": value.prior_execution_generation,
            "recovered_execution_generation": value.recovered_execution_generation,
            "recovered_state_generation": value.recovered_state_generation,
        }
    else:  # pragma: no cover - a defensive boundary for dynamically supplied values
        raise ValidationError(f"unsupported process wait state: {type(value).__name__}")
    return {
        "schema_version": PROCESS_STATE_SCHEMA_VERSION,
        "kind": value.kind,
        **fields,
    }


def process_wait_state_from_mapping(value: Mapping[str, Any] | None) -> ProcessWaitState | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("process wait state must be an object or null")
    kind = value.get("kind")
    if kind == ChildProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"child_pid"}))
        return ChildProcessWait(child_pid=selected["child_pid"])
    if kind == MessageProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"filters"}))
        filters = selected["filters"]
        if not isinstance(filters, dict):
            raise ValidationError("process state message filters must be an object")
        return MessageProcessWait(filters=filters)
    if kind == HumanProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"request_ids"}))
        request_ids = selected["request_ids"]
        if not isinstance(request_ids, list):
            raise ValidationError("process state human request_ids must be a list")
        return HumanProcessWait(request_ids=tuple(request_ids))
    if kind == ToolProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"operation_id"}))
        return ToolProcessWait(operation_id=selected["operation_id"])
    if kind == PausedProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"reason_oid"}))
        return PausedProcessWait(reason_oid=selected["reason_oid"])
    if kind == HostResumeProcessWait.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"reason_oid"}))
        return HostResumeProcessWait(reason_oid=selected["reason_oid"])
    if kind == StaleExecutionProcessWait.kind:
        selected = _canonical_mapping(
            value,
            kind=kind,
            fields=frozenset(
                {
                    "pid",
                    "recovered_by_owner_sha256",
                    "prior_owner_sha256",
                    "prior_lease_sha256",
                    "prior_execution_generation",
                    "recovered_execution_generation",
                    "recovered_state_generation",
                }
            ),
        )
        return StaleExecutionProcessWait(
            pid=selected["pid"],
            recovered_by_owner_sha256=selected["recovered_by_owner_sha256"],
            prior_owner_sha256=selected["prior_owner_sha256"],
            prior_lease_sha256=selected["prior_lease_sha256"],
            prior_execution_generation=selected["prior_execution_generation"],
            recovered_execution_generation=selected[
                "recovered_execution_generation"
            ],
            recovered_state_generation=selected["recovered_state_generation"],
        )
    raise ValidationError(f"unsupported process wait state kind: {kind!r}")


def process_outcome_to_mapping(value: ProcessOutcome | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ExitedProcessOutcome):
        fields = {"result_oid": value.result_oid}
    elif isinstance(value, FailedProcessOutcome):
        fields = {"result_oid": value.result_oid, "code": value.code}
    elif isinstance(value, KilledProcessOutcome):
        fields = {"reason_oid": value.reason_oid, "code": value.code}
    else:  # pragma: no cover - a defensive boundary for dynamically supplied values
        raise ValidationError(f"unsupported process outcome: {type(value).__name__}")
    return {
        "schema_version": PROCESS_STATE_SCHEMA_VERSION,
        "kind": value.kind,
        **fields,
    }


def process_state_to_mapping(
    status: str,
    wait_state: ProcessWaitState | None,
    outcome: ProcessOutcome | None,
    state_generation: int,
) -> dict[str, Any]:
    """Return the canonical public projection of durable process state."""

    if not isinstance(status, str):
        raise ValidationError("process state status must be a string")
    if (
        not isinstance(state_generation, int)
        or isinstance(state_generation, bool)
        or state_generation < 0
    ):
        raise ValidationError(
            "process state state_generation must be a non-negative integer"
        )
    selected_status = str(status)
    validate_process_state_fields(selected_status, wait_state, outcome)
    return {
        "status": selected_status,
        "wait_state": process_wait_state_to_mapping(wait_state),
        "outcome": process_outcome_to_mapping(outcome),
        "state_generation": state_generation,
    }


def process_outcome_from_mapping(value: Mapping[str, Any] | None) -> ProcessOutcome | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("process outcome must be an object or null")
    kind = value.get("kind")
    if kind == ExitedProcessOutcome.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"result_oid"}))
        return ExitedProcessOutcome(result_oid=selected["result_oid"])
    if kind == FailedProcessOutcome.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"result_oid", "code"}))
        return FailedProcessOutcome(result_oid=selected["result_oid"], code=selected["code"])
    if kind == KilledProcessOutcome.kind:
        selected = _canonical_mapping(value, kind=kind, fields=frozenset({"reason_oid", "code"}))
        return KilledProcessOutcome(reason_oid=selected["reason_oid"], code=selected["code"])
    raise ValidationError(f"unsupported process outcome kind: {kind!r}")


def process_wait_state_from_json(value: str | None) -> ProcessWaitState | None:
    if value is None:
        return None
    decoded = _load_strict_json(value, label="wait state")
    return process_wait_state_from_mapping(decoded)


def process_outcome_from_json(value: str | None) -> ProcessOutcome | None:
    if value is None:
        return None
    decoded = _load_strict_json(value, label="outcome")
    return process_outcome_from_mapping(decoded)


def legacy_status_message(
    wait_state: ProcessWaitState | None,
    outcome: ProcessOutcome | None,
    fallback: str | None = None,
) -> str | None:
    """Project typed state into the 0.3 public compatibility field.

    Runtime logic must never parse this projection. It exists so older CLI/GUI
    clients continue to receive the field while the typed API is adopted.
    """

    if isinstance(wait_state, ChildProcessWait):
        return f"waiting for {wait_state.child_pid}"
    if isinstance(wait_state, MessageProcessWait):
        return f"waiting_message:{json.dumps(wait_state.filters, sort_keys=True)}"
    if isinstance(wait_state, HumanProcessWait):
        prefix = "waiting for human request " if len(wait_state.request_ids) == 1 else "waiting for human requests "
        return prefix + ",".join(wait_state.request_ids)
    if isinstance(wait_state, HostResumeProcessWait):
        return f"host_resume_required:{wait_state.reason_oid}"
    if isinstance(wait_state, StaleExecutionProcessWait):
        return "stale_execution_recovery"
    if isinstance(wait_state, PausedProcessWait) and wait_state.reason_oid is not None:
        return f"result_oid:{wait_state.reason_oid}"
    if isinstance(outcome, (ExitedProcessOutcome, FailedProcessOutcome)) and outcome.result_oid is not None:
        return f"result_oid:{outcome.result_oid}"
    if isinstance(outcome, KilledProcessOutcome) and outcome.reason_oid is not None:
        return f"result_oid:{outcome.reason_oid}"
    return fallback


@dataclass(frozen=True, slots=True)
class UpcastProcessState:
    wait_state: ProcessWaitState | None
    outcome: ProcessOutcome | None
    status_message: str | None


def _upcast_legacy_event_wait(message: str | None) -> UpcastProcessState:
    if message and message.startswith("waiting_message:"):
        encoded = message[len("waiting_message:") :]
        try:
            filters = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                "legacy process message wait contains invalid JSON"
            ) from exc
        if not isinstance(filters, dict):
            raise ValidationError(
                "legacy process message wait filters must be an object"
            )
        wait_state: ProcessWaitState = MessageProcessWait(filters=filters)
    elif message and message.startswith("waiting for "):
        wait_state = ChildProcessWait(child_pid=message[len("waiting for ") :])
    else:
        raise ValidationError(
            "legacy waiting_event process has no identifiable wait state"
        )
    return UpcastProcessState(
        wait_state,
        None,
        legacy_status_message(wait_state, None),
    )


def _upcast_legacy_human_wait(message: str | None) -> UpcastProcessState:
    request_ids: tuple[str, ...] = ()
    if message:
        for prefix in (
            "waiting for human requests ",
            "waiting for human request ",
        ):
            if message.startswith(prefix):
                request_ids = tuple(
                    item.strip()
                    for item in message[len(prefix) :].split(",")
                    if item.strip()
                )
                break
    if not request_ids:
        raise ValidationError(
            "legacy waiting_human process has no identifiable request"
        )
    wait_state = HumanProcessWait(request_ids=request_ids)
    return UpcastProcessState(
        wait_state,
        None,
        legacy_status_message(wait_state, None),
    )


def _upcast_legacy_pause(message: str | None) -> UpcastProcessState:
    if message and message.startswith("host_resume_required:"):
        wait_state: ProcessWaitState = HostResumeProcessWait(
            reason_oid=message[len("host_resume_required:") :]
        )
        return UpcastProcessState(
            wait_state,
            None,
            legacy_status_message(wait_state, None),
        )
    if message and message.startswith("result_oid:"):
        wait_state = PausedProcessWait(
            reason_oid=message[len("result_oid:") :]
        )
        return UpcastProcessState(
            wait_state,
            None,
            legacy_status_message(wait_state, None),
        )
    return UpcastProcessState(PausedProcessWait(), None, message)


def _upcast_legacy_outcome(status: str, message: str | None) -> UpcastProcessState:
    result_oid = None
    if message and message.startswith("result_oid:"):
        result_oid = _non_empty_string(
            message[len("result_oid:") :],
            "result_oid",
        )
    if status == "exited":
        outcome: ProcessOutcome = ExitedProcessOutcome(result_oid=result_oid)
    elif status == "failed":
        outcome = FailedProcessOutcome(result_oid=result_oid, code=None)
    else:
        outcome = KilledProcessOutcome(reason_oid=result_oid, code=None)
    return UpcastProcessState(
        None,
        outcome,
        legacy_status_message(None, outcome, message),
    )


def upcast_legacy_process_state(status: str, status_message: str | None) -> UpcastProcessState:
    """Decode the exact pre-typed status-message protocol once.

    Ambiguous waiting rows fail closed. Ordinary diagnostic messages remain in
    ``status_message`` and are not reinterpreted as machine state.
    """

    selected_status = str(status)
    message = status_message if status_message is None else str(status_message)
    if selected_status == "waiting_event":
        return _upcast_legacy_event_wait(message)
    if selected_status == "waiting_human":
        return _upcast_legacy_human_wait(message)
    if selected_status == "waiting_tool":
        raise ValidationError("legacy waiting_tool process has no typed operation identity")
    if selected_status == "paused":
        return _upcast_legacy_pause(message)
    if selected_status in {"exited", "failed", "killed"}:
        return _upcast_legacy_outcome(selected_status, message)
    return UpcastProcessState(None, None, message)


def remap_process_wait_state(
    value: ProcessWaitState | None,
    *,
    pids: Mapping[str, str],
    objects: Mapping[str, str],
) -> ProcessWaitState | None:
    if isinstance(value, ChildProcessWait):
        return ChildProcessWait(child_pid=pids.get(value.child_pid, value.child_pid))
    if isinstance(value, (PausedProcessWait, HostResumeProcessWait)):
        reason_oid = objects.get(value.reason_oid, value.reason_oid) if value.reason_oid is not None else None
        return type(value)(reason_oid=reason_oid)  # type: ignore[call-arg]
    if isinstance(value, StaleExecutionProcessWait):
        # A fork/restore cannot transfer an execution takeover receipt to a
        # different concurrency identity. Preserve the safe paused posture but
        # intentionally discard the non-transferable recovery provenance.
        return PausedProcessWait(reason_oid=None)
    return value


def remap_process_outcome(
    value: ProcessOutcome | None,
    *,
    objects: Mapping[str, str],
) -> ProcessOutcome | None:
    if isinstance(value, ExitedProcessOutcome):
        return ExitedProcessOutcome(
            result_oid=objects.get(value.result_oid, value.result_oid) if value.result_oid is not None else None
        )
    if isinstance(value, FailedProcessOutcome):
        return FailedProcessOutcome(
            result_oid=objects.get(value.result_oid, value.result_oid) if value.result_oid is not None else None,
            code=value.code,
        )
    if isinstance(value, KilledProcessOutcome):
        return KilledProcessOutcome(
            reason_oid=objects.get(value.reason_oid, value.reason_oid) if value.reason_oid is not None else None,
            code=value.code,
        )
    return None
