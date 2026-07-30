from __future__ import annotations

import ast
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import pytest

from agent_libos.models import (
    AgentProcess,
    ChildProcessWait,
    ExitedProcessOutcome,
    FailedProcessOutcome,
    HostResumeProcessWait,
    HumanProcessWait,
    KilledProcessOutcome,
    legacy_status_message,
    MessageProcessWait,
    PausedProcessWait,
    ProcessOutcome,
    ProcessStatus,
    ProcessWaitState,
    ResourceBudget,
    ResourceUsage,
    ToolProcessWait,
    process_outcome_from_mapping,
    process_outcome_to_mapping,
    process_wait_state_from_json,
    process_wait_state_from_mapping,
    process_wait_state_to_mapping,
    remap_process_outcome,
    upcast_legacy_process_state,
)
from agent_libos.models.exceptions import ProcessError, ProcessRevisionConflict, ValidationError
from agent_libos.process_transition import (
    ProcessStateToken,
    ProcessTransitionService,
    validate_process_state,
)
from agent_libos.ports.processes import ProcessTransitionRepositoryPort
from agent_libos.utils.ids import utc_now


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _process(pid: str = "pid_1") -> AgentProcess:
    now = utc_now()
    return AgentProcess(
        pid=pid,
        parent_pid=None,
        image_id="base-agent:v0",
        status=ProcessStatus.RUNNABLE,
        goal_oid=None,
        memory_view=None,
        capabilities=[],
        loaded_skills={},
        tool_table={},
        event_cursor=None,
        checkpoint_head=None,
        resource_budget=ResourceBudget(),
        resource_usage=ResourceUsage(),
        created_at=now,
        updated_at=now,
    )


class _ProcessRepository:
    def __init__(self, process: AgentProcess):
        self.process = deepcopy(process)

    def get_process(self, pid: str) -> AgentProcess | None:
        if pid != self.process.pid:
            return None
        return deepcopy(self.process)
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
        del reason
        if pid != self.process.pid:
            raise ProcessRevisionConflict(f"process no longer exists: {pid}")
        if self.process.revision != expected_revision:
            raise ProcessRevisionConflict("revision conflict")
        if expected_status is not None and self.process.status != ProcessStatus(expected_status):
            raise ProcessRevisionConflict("status conflict")
        if (
            expected_state_generation is not None
            and self.process.state_generation != expected_state_generation
        ):
            raise ProcessRevisionConflict("state generation conflict")
        if control:
            allowed = {
                ProcessStatus(item)
                for item in (allowed_statuses or ())
            }
            if self.process.status not in allowed:
                raise ProcessRevisionConflict("control status conflict")
        self.process.status = ProcessStatus(status)
        self.process.wait_state = deepcopy(wait_state)
        self.process.outcome = deepcopy(outcome)
        self.process.status_message = legacy_status_message(
            self.process.wait_state,
            self.process.outcome,
            status_message,
        )
        self.process.state_generation += 1
        self.process.revision += 1
        return deepcopy(self.process)


class _InterleavingProcessRepository(_ProcessRepository):
    def __init__(self, process: AgentProcess):
        super().__init__(process)
        self.interleave_after_next_read = False

    def get_process(self, pid: str) -> AgentProcess | None:
        observed = super().get_process(pid)
        if self.interleave_after_next_read and observed is not None:
            self.interleave_after_next_read = False
            replacement = MessageProcessWait(filters={"channel": "racing-owner"})
            self.process.status = ProcessStatus.WAITING_EVENT
            self.process.wait_state = replacement
            self.process.outcome = None
            self.process.status_message = legacy_status_message(
                replacement,
                None,
                None,
            )
            self.process.state_generation += 1
            self.process.revision += 1
        return observed


def test_process_transition_repository_port_covers_the_service_surface() -> None:
    service_tree = ast.parse(
        (_PROJECT_ROOT / "agent_libos" / "process_transition.py").read_text(
            encoding="utf-8"
        )
    )
    service_methods = {
        node.func.attr
        for node in ast.walk(service_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "store"
    }
    port_methods = {
        name
        for name, member in vars(ProcessTransitionRepositoryPort).items()
        if callable(member) and not name.startswith("_")
    }

    assert service_methods == {"apply_process_state_transition", "get_process"}
    assert service_methods <= port_methods


@pytest.mark.parametrize(
    "value",
    [
        ChildProcessWait(child_pid="pid_child"),
        MessageProcessWait(filters={"kind": "interrupt", "unread_only": True}),
        HumanProcessWait(request_ids=("hreq_1", "hreq_2")),
        ToolProcessWait(operation_id="op_1"),
        PausedProcessWait(reason_oid="obj_pause"),
        HostResumeProcessWait(reason_oid="obj_host"),
    ],
)
def test_wait_state_codec_is_strict_and_lossless(value: object) -> None:
    encoded = process_wait_state_to_mapping(value)  # type: ignore[arg-type]
    assert process_wait_state_from_mapping(encoded) == value
    assert encoded is not None
    encoded["unknown"] = True
    with pytest.raises(ValidationError, match="not canonical"):
        process_wait_state_from_mapping(encoded)


@pytest.mark.parametrize(
    "value",
    [
        ExitedProcessOutcome(result_oid="obj_result"),
        FailedProcessOutcome(result_oid="obj_error", code="tool_failed"),
        KilledProcessOutcome(reason_oid="obj_reason", code="cancel"),
    ],
)
def test_process_outcome_codec_is_strict_and_lossless(value: object) -> None:
    encoded = process_outcome_to_mapping(value)  # type: ignore[arg-type]
    assert process_outcome_from_mapping(encoded) == value
    assert encoded is not None
    encoded["schema_version"] = 999
    with pytest.raises(ValidationError, match="unsupported process state schema_version"):
        process_outcome_from_mapping(encoded)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_process_state_schema_version_rejects_non_integer_aliases(
    schema_version: object,
) -> None:
    encoded = process_wait_state_to_mapping(
        ChildProcessWait(child_pid="pid_child")
    )
    assert encoded is not None
    encoded["schema_version"] = schema_version
    with pytest.raises(
        ValidationError,
        match="unsupported process state schema_version",
    ):
        process_wait_state_from_mapping(encoded)


def test_message_wait_filters_require_a_lossless_strict_json_tree() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    invalid_filters: tuple[dict[object, object], ...] = (
        {1: "coerced key"},
        {"bytes": b"not JSON"},
        {"tuple": ("not", "a", "JSON", "array")},
        {"set": {"not JSON"}},
        {"number": float("nan")},
        {"number": float("inf")},
        cycle,
    )

    for filters in invalid_filters:
        with pytest.raises(ValidationError, match="message filters"):
            MessageProcessWait(filters=filters)  # type: ignore[arg-type]

    canonical = MessageProcessWait(
        filters={
            "channel": "control",
            "nested": {"values": [None, True, 1, 2.5, "text"]},
        }
    )
    assert process_wait_state_from_mapping(
        process_wait_state_to_mapping(canonical)
    ) == canonical


def test_persisted_process_state_rejects_duplicate_json_object_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate JSON object key"):
        process_wait_state_from_json(
            '{"schema_version":1,"kind":"message","filters":'
            '{"channel":"first","channel":"second"}}'
        )


def test_legacy_protocol_is_upcast_at_one_explicit_boundary() -> None:
    child = upcast_legacy_process_state("waiting_event", "waiting for pid_child")
    assert child.wait_state == ChildProcessWait(child_pid="pid_child")

    terminal = upcast_legacy_process_state("exited", "result_oid:obj_result")
    assert terminal.outcome == ExitedProcessOutcome(result_oid="obj_result")

    with pytest.raises(ValidationError, match="no identifiable wait state"):
        upcast_legacy_process_state("waiting_event", "ordinary diagnostic")
    with pytest.raises(ValidationError, match="operation identity"):
        upcast_legacy_process_state("waiting_tool", "legacy tool wait")


def test_cross_field_state_invariants_fail_closed() -> None:
    with pytest.raises(ValidationError, match="waiting_event requires"):
        validate_process_state(ProcessStatus.WAITING_EVENT, None, None)
    with pytest.raises(ValidationError, match="exited requires"):
        validate_process_state(ProcessStatus.EXITED, None, None)
    with pytest.raises(ValidationError, match="cannot carry a terminal outcome"):
        validate_process_state(
            ProcessStatus.RUNNABLE,
            None,
            ExitedProcessOutcome(result_oid="obj_result"),
        )


def test_state_generation_fences_a_repeated_wait_against_aba_wakeup() -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)

    first = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_EVENT,
        expected_revision=0,
        wait_state=ChildProcessWait(child_pid="pid_child"),
    )
    stale_token = transitions.wait_token(first)
    runnable = transitions.wake(stale_token, control=False)
    second = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_EVENT,
        expected_revision=runnable.revision,
        wait_state=ChildProcessWait(child_pid="pid_child"),
    )

    assert first.state_generation == 1
    assert runnable.state_generation == 2
    assert second.state_generation == 3
    with pytest.raises(ProcessRevisionConflict, match="stale process wait token"):
        transitions.wake(stale_token, control=False)


@pytest.mark.parametrize(
    "pause_wait",
    (
        PausedProcessWait(reason_oid="obj_pause"),
        HostResumeProcessWait(reason_oid="obj_host_resume"),
    ),
    ids=("paused", "host-resume"),
)
def test_pause_wait_cannot_be_turned_into_a_condition_wake_token(
    pause_wait: ProcessWaitState,
) -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    paused = transitions.transition(
        "pid_1",
        ProcessStatus.PAUSED,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=pause_wait,
    )

    with pytest.raises(ProcessRevisionConflict, match="no condition-owned wait"):
        transitions.wait_token(paused)
    forged = ProcessStateToken(
        pid=paused.pid,
        state_generation=paused.state_generation,
        wait_state=pause_wait,
    )
    with pytest.raises(ProcessRevisionConflict, match="not condition-owned"):
        transitions.wake(forged, control=False)

    assert repository.get_process("pid_1") == paused


@pytest.mark.parametrize(
    ("current_status", "current_wait", "requested_status", "requested_wait"),
    (
        (
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "control"}),
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_new",)),
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_current",)),
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "control"}),
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_current",)),
            ProcessStatus.WAITING_EVENT,
            ChildProcessWait(child_pid="pid_child"),
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_current",)),
            ProcessStatus.WAITING_TOOL,
            ToolProcessWait(operation_id="otask_new"),
        ),
        (
            ProcessStatus.WAITING_TOOL,
            ToolProcessWait(operation_id="otask_current"),
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_new",)),
        ),
        (
            ProcessStatus.PAUSED,
            PausedProcessWait(reason_oid="obj_pause"),
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "control"}),
        ),
        (
            ProcessStatus.PAUSED,
            HostResumeProcessWait(reason_oid="obj_host"),
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "control"}),
        ),
    ),
    ids=(
        "message-to-human",
        "human-to-message",
        "human-to-child",
        "human-to-tool",
        "tool-to-human",
        "pause-to-message",
        "host-resume-to-message",
    ),
)
def test_condition_wait_registration_cannot_replace_another_owner(
    current_status: ProcessStatus,
    current_wait: ProcessWaitState,
    requested_status: ProcessStatus,
    requested_wait: ProcessWaitState,
) -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    waiting = transitions.transition(
        "pid_1",
        current_status,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=current_wait,
    )

    with pytest.raises(ProcessError, match="owned by another condition"):
        transitions.transition(
            "pid_1",
            requested_status,
            expected_revision=waiting.revision,
            expected_status=waiting.status,
            expected_state_generation=waiting.state_generation,
            wait_state=requested_wait,
        )

    assert repository.get_process("pid_1") == waiting


def test_suspended_process_cannot_register_a_condition_wait() -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    suspended = transitions.transition(
        "pid_1",
        ProcessStatus.SUSPENDED,
        expected_revision=0,
        expected_state_generation=0,
    )

    with pytest.raises(ProcessError, match="suspended process"):
        transitions.transition(
            "pid_1",
            ProcessStatus.WAITING_HUMAN,
            expected_revision=suspended.revision,
            expected_status=suspended.status,
            expected_state_generation=suspended.state_generation,
            wait_state=HumanProcessWait(request_ids=("hreq_new",)),
        )

    assert repository.get_process("pid_1") == suspended


def test_condition_wait_owner_can_update_its_own_generation() -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    first = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_HUMAN,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=HumanProcessWait(request_ids=("hreq_first",)),
    )

    second = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_HUMAN,
        expected_revision=first.revision,
        expected_status=first.status,
        expected_state_generation=first.state_generation,
        wait_state=HumanProcessWait(
            request_ids=("hreq_first", "hreq_second"),
        ),
    )

    assert second.wait_state == HumanProcessWait(
        request_ids=("hreq_first", "hreq_second")
    )
    assert second.state_generation == first.state_generation + 1
    assert second.revision == first.revision + 1

    third = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_HUMAN,
        expected_revision=second.revision,
        expected_status=second.status,
        expected_state_generation=second.state_generation,
        wait_state=HumanProcessWait(request_ids=("hreq_second",)),
    )
    assert third.wait_state == HumanProcessWait(request_ids=("hreq_second",))
    assert third.state_generation == second.state_generation + 1


@pytest.mark.parametrize(
    ("status", "current_wait", "requested_wait", "error"),
    (
        (
            ProcessStatus.WAITING_EVENT,
            ChildProcessWait(child_pid="pid_first"),
            ChildProcessWait(child_pid="pid_second"),
            "cannot retarget",
        ),
        (
            ProcessStatus.WAITING_EVENT,
            MessageProcessWait(filters={"channel": "first"}),
            MessageProcessWait(filters={"channel": "second"}),
            "cannot retarget",
        ),
        (
            ProcessStatus.WAITING_TOOL,
            ToolProcessWait(operation_id="otask_first"),
            ToolProcessWait(operation_id="otask_second"),
            "cannot retarget",
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_first",)),
            HumanProcessWait(request_ids=("hreq_second",)),
            "mixed request set",
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_first", "hreq_keep")),
            HumanProcessWait(request_ids=("hreq_keep", "hreq_new")),
            "mixed request set",
        ),
    ),
    ids=("child", "message", "tool", "human-disjoint", "human-mixed"),
)
def test_condition_wait_owner_cannot_retarget_an_active_wait(
    status: ProcessStatus,
    current_wait: ProcessWaitState,
    requested_wait: ProcessWaitState,
    error: str,
) -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    waiting = transitions.transition(
        "pid_1",
        status,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=current_wait,
    )

    with pytest.raises(ProcessError, match=error):
        transitions.transition(
            "pid_1",
            status,
            expected_revision=waiting.revision,
            expected_status=waiting.status,
            expected_state_generation=waiting.state_generation,
            wait_state=requested_wait,
        )

    assert repository.get_process("pid_1") == waiting


def test_wait_registration_reports_a_stale_revision_before_owner_mismatch() -> None:
    repository = _ProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    human_wait = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_HUMAN,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=HumanProcessWait(request_ids=("hreq_stale",)),
    )
    current = repository.process
    replacement = MessageProcessWait(filters={"channel": "current"})
    current.status = ProcessStatus.WAITING_EVENT
    current.wait_state = replacement
    current.status_message = legacy_status_message(replacement, None, None)
    current.state_generation += 1
    current.revision += 1

    with pytest.raises(ProcessRevisionConflict, match="stale process wait registration"):
        transitions.transition(
            "pid_1",
            ProcessStatus.WAITING_HUMAN,
            expected_revision=human_wait.revision,
            expected_status=human_wait.status,
            expected_state_generation=human_wait.state_generation,
            wait_state=HumanProcessWait(
                request_ids=("hreq_stale", "hreq_new"),
            ),
        )

    assert repository.get_process("pid_1") == current


def test_wait_registration_cas_rejects_owner_change_after_guard_read() -> None:
    repository = _InterleavingProcessRepository(_process())
    transitions = ProcessTransitionService(repository)
    human_wait = transitions.transition(
        "pid_1",
        ProcessStatus.WAITING_HUMAN,
        expected_revision=0,
        expected_state_generation=0,
        wait_state=HumanProcessWait(request_ids=("hreq_first",)),
    )
    repository.interleave_after_next_read = True

    with pytest.raises(ProcessRevisionConflict, match="revision conflict"):
        transitions.transition(
            "pid_1",
            ProcessStatus.WAITING_HUMAN,
            expected_revision=human_wait.revision,
            expected_status=human_wait.status,
            expected_state_generation=human_wait.state_generation,
            wait_state=HumanProcessWait(
                request_ids=("hreq_first", "hreq_second"),
            ),
        )

    current = repository.get_process("pid_1")
    assert current is not None
    assert current.status == ProcessStatus.WAITING_EVENT
    assert current.wait_state == MessageProcessWait(
        filters={"channel": "racing-owner"}
    )


def test_forked_terminal_outcome_uses_the_cloned_result_object() -> None:
    outcome = ExitedProcessOutcome(result_oid="obj_source")
    remapped = remap_process_outcome(
        outcome,
        objects={"obj_source": "obj_fork"},
    )

    assert remapped == ExitedProcessOutcome(result_oid="obj_fork")


def test_runtime_does_not_parse_the_status_message_compatibility_projection() -> None:
    violations: list[str] = []
    for path in (_PROJECT_ROOT / "agent_libos").rglob("*.py"):
        if path.name == "process_state.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if (
                    isinstance(receiver, ast.Attribute)
                    and receiver.attr == "status_message"
                    and node.func.attr in {"startswith", "split"}
                ):
                    violations.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if any(
                    isinstance(operand, ast.Attribute)
                    and operand.attr == "status_message"
                    for operand in operands
                ):
                    violations.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}")
    assert violations == []


def test_runtime_process_status_writes_use_the_transition_service() -> None:
    violations: list[str] = []
    semantic_fields = {"status", "wait_state", "outcome", "state_generation"}
    for path in (_PROJECT_ROOT / "agent_libos").rglob("*.py"):
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
        # Typed execution-lease and snapshot-restore CAS primitives live in the
        # storage layer because their state update must share one SQL commit
        # point with the corresponding concurrency fence.  Normal runtime
        # orchestration remains constrained to ProcessTransitionService.
        if relative.startswith("agent_libos/storage/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            operation = node.func.attr
            if operation in {
                "transition_process",
                "transition_process_control",
                "update_process",
            }:
                violations.append(f"{relative}:{node.lineno}:{operation}")
                continue
            if operation == "apply_process_state_transition":
                if relative != "agent_libos/process_transition.py":
                    violations.append(f"{relative}:{node.lineno}:{operation}")
                continue
            if operation not in {"patch_process", "patch_process_control"}:
                continue
            mappings = [
                item
                for item in [*node.args, *(keyword.value for keyword in node.keywords)]
                if isinstance(item, ast.Dict)
            ]
            literal_keys = {
                key.value
                for mapping in mappings
                for key in mapping.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if literal_keys & semantic_fields:
                violations.append(f"{relative}:{node.lineno}:{operation}")
    assert violations == []


def test_process_exec_epoch_commit_exception_has_one_runtime_call_site() -> None:
    callers: list[str] = []
    for path in (_PROJECT_ROOT / "agent_libos").rglob("*.py"):
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "commit_process_exec_epoch"
            ):
                callers.append(f"{relative}:{node.lineno}")

    assert sorted(item.rsplit(":", 1)[0] for item in callers) == [
        "agent_libos/runtime/image_boot.py",
        "agent_libos/storage/repositories.py",
    ]
