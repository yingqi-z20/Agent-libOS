from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentProcess,
    ChildProcessWait,
    EventPriority,
    EventType,
    KilledProcessOutcome,
    ProcessExecutionToken,
    ProcessStatus,
    ResourceBudget,
    ResourceReservation,
    ResourceUsage,
    ResourceUsageReservation,
    ResourceUsageReservationCursor,
    ResourceUsageReservationRecoverySummary,
    ResourceUsageReservationStatus,
)
from agent_libos.models.exceptions import NotFound, ResourceLimitExceeded, ValidationError
from agent_libos.process_execution import (
    current_process_execution_token,
    trusted_process_execution_takeover,
    trusted_terminal_process_mutation,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.storage import RuntimeStore, UnitOfWork
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import to_jsonable
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)


_TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_USAGE_FIELD_NAMES = {field.name for field in fields(ResourceUsage)}

# Most counters are cumulative. Peak RSS is different: one large subprocess
# should not make future small subprocesses impossible, but it must still be
# compared against the configured per-process tree memory ceiling.
_PEAK_USAGE_FIELDS = {"subprocess_peak_memory_bytes"}

_BUDGET_USAGE_MAP: dict[str, tuple[str, ...]] = {
    "max_tool_calls": ("tool_calls",),
    "max_child_processes": ("child_processes",),
    "max_runtime_seconds": ("runtime_seconds",),
    "max_context_materialization_total_tokens": ("context_materialized_tokens",),
    "max_llm_calls": ("llm_calls",),
    "max_llm_total_tokens": ("llm_total_tokens",),
    "max_subprocess_wall_seconds": ("subprocess_wall_seconds",),
    "max_subprocess_cpu_seconds": ("subprocess_cpu_seconds",),
    "max_subprocess_memory_bytes": ("subprocess_peak_memory_bytes",),
    "max_external_read_bytes": ("external_read_bytes",),
    "max_external_write_bytes": ("external_write_bytes",),
    "max_jsonrpc_bytes": ("jsonrpc_request_bytes", "jsonrpc_response_bytes"),
    "max_mcp_bytes": ("mcp_request_bytes", "mcp_response_bytes"),
    "max_deno_syscalls": ("deno_syscalls",),
}

_NON_RESERVABLE_BUDGET_FIELDS = {"max_subprocess_memory_bytes", "max_child_processes"}
_CONTINUOUS_BUDGET_FIELDS = {
    "max_runtime_seconds",
    "max_subprocess_wall_seconds",
    "max_subprocess_cpu_seconds",
}

_ResourceNumber = int | float


@dataclass(frozen=True, slots=True)
class _ResourceLimitFinalization:
    """Terminal hooks that are safe only after their durable kill commits."""

    pid: str
    killed_pids: tuple[str, ...]
    reason: str


class ResourceManager:
    """Hierarchical process resource accounting.

    Capability grants decide whether a process may attempt an operation.
    Resource budgets decide whether the process tree still has enough quota to
    spend on that operation. Charging walks the pid -> parent chain so a parent
    can bound the total consumption of all descendants.
    """

    def __init__(
        self,
        unit_of_work: UnitOfWork | RuntimeStore,
        audit: AuditManager,
        events: EventBus,
        *,
        require_recovery_lease: Callable[[], None],
        transitions: ProcessTransitionService | None = None,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        # Keep direct-store construction working for embedders while all
        # runtime composition goes through one explicit UnitOfWork boundary.
        self.unit_of_work = (
            unit_of_work
            if isinstance(unit_of_work, UnitOfWork)
            else UnitOfWork(unit_of_work)
        )
        self.store = self.unit_of_work.processes
        self.resource_repository = self.unit_of_work.resources
        self.effects = self.unit_of_work.evidence
        self.audit = audit
        self.events = events
        self.config = config or DEFAULT_CONFIG
        self._require_recovery_lease = require_recovery_lease
        self._transitions = transitions or ProcessTransitionService(self.store)
        self._process_kill_finalizer: Callable[..., None] | None = None
        self._object_task_terminal_notifier: Callable[[str], None] | None = None

    def bind_process_kill_finalizer(self, finalizer: Callable[..., None]) -> None:
        self._process_kill_finalizer = finalizer

    def bind_object_task_terminal_notifier(self, notifier: Callable[[str], None]) -> None:
        self._object_task_terminal_notifier = notifier

    def preflight(
        self,
        pid: str,
        request: ResourceUsage | dict[str, Any],
        *,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        usage = self._coerce_usage(request)
        if self._is_zero(usage):
            return
        with self.unit_of_work.locked():
            self._preflight_locked(pid, usage, source=source, context=context)

    def reserve_usage(
        self,
        pid: str,
        usage: ResourceUsage | dict[str, Any],
        *,
        source: str,
        context: dict[str, Any] | None = None,
        reservation_id: str | None = None,
        reserved_by: str | None = None,
    ) -> str:
        """Atomically reserve a maximum usage envelope before provider dispatch."""

        selected = self._coerce_usage(usage)
        if self._is_zero(selected):
            raise ValidationError("resource usage reservation must be non-zero")
        selected_id = reservation_id or new_id("usage_reservation")
        now = utc_now()
        with self.unit_of_work.transaction():
            self._preflight_locked(pid, selected, source=source, context=context)
            self.resource_repository.insert_resource_usage_reservation(
                reservation_id=selected_id,
                pid=pid,
                usage=selected,
                reserved_by=reserved_by or source,
                reason=source,
                created_at=now,
            )
        return selected_id

    def settle_usage_reservation(
        self,
        reservation_id: str,
        *,
        actual_usage: ResourceUsage | dict[str, Any] | None = None,
        charge_maximum: bool = False,
        release: bool = False,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> ResourceUsage:
        """Settle exactly once; unknown provider outcomes charge the full envelope."""

        return self._settle_usage_reservation(
            reservation_id,
            actual_usage=actual_usage,
            charge_maximum=charge_maximum,
            release=release,
            source=source,
            context=context,
            recovery=False,
        )

    def _settle_usage_reservation(
        self,
        reservation_id: str,
        *,
        actual_usage: ResourceUsage | dict[str, Any] | None = None,
        charge_maximum: bool = False,
        release: bool = False,
        source: str,
        context: dict[str, Any] | None = None,
        recovery: bool,
    ) -> ResourceUsage:
        if charge_maximum and release:
            raise ValidationError("resource reservation cannot both release and charge maximum")
        post_commit_finalizations: list[_ResourceLimitFinalization] = []
        with self.unit_of_work.transaction():
            reservation = self.resource_repository.get_resource_usage_reservation(
                reservation_id
            )
            if reservation is None:
                raise ValidationError(f"resource usage reservation not found: {reservation_id}")
            if reservation.status.value != "active":
                return reservation.settled_usage or ResourceUsage()
            maximum = reservation.usage
            if release:
                selected = ResourceUsage()
                status = "released"
            elif charge_maximum:
                selected = maximum
                status = "charged_maximum"
            else:
                selected = self._coerce_usage(actual_usage or ResourceUsage())
                self._assert_usage_within_reservation(selected, maximum)
                status = "settled"
            if not self.resource_repository.settle_resource_usage_reservation(
                reservation_id,
                status=status,
                settled_usage=selected,
                updated_at=utc_now(),
            ):
                raise ValidationError(
                    f"resource usage reservation changed concurrently: {reservation_id}"
                )
            if not self._is_zero(selected):
                if recovery:
                    try:
                        self._charge(
                            reservation.pid,
                            selected,
                            source=source,
                            context={
                                **(context or {}),
                                "reservation_id": reservation_id,
                                "settlement": status,
                            },
                            allow_overage=True,
                            kill_on_exceed=True,
                            include_active_reservations=False,
                            deferred_resource_limit_finalizations=post_commit_finalizations,
                        )
                    except ResourceLimitExceeded:
                        # Recovery charges an ambiguous provider effect even if
                        # that pushes the durable usage over budget.  The
                        # charge and resource-limit termination are already in
                        # this outer transaction; continue draining the backlog.
                        pass
                else:
                    self._charge(
                        reservation.pid,
                        selected,
                        source=source,
                        context={
                            **(context or {}),
                            "reservation_id": reservation_id,
                            "settlement": status,
                        },
                        allow_overage=False,
                        kill_on_exceed=True,
                        include_active_reservations=True,
                        deferred_resource_limit_finalizations=post_commit_finalizations,
                    )
        self._run_resource_limit_finalizations(post_commit_finalizations)
        return selected

    def recover_usage_reservations(
        self,
    ) -> ResourceUsageReservationRecoverySummary:
        """Release certified pre-dispatch rows and charge ambiguous rows maximally."""

        self._require_recovery_lease()
        page_size = (
            self.config.runtime.resource_usage_reservation_recovery_page_size
        )
        sample: list[str] = []
        total_count = 0
        for reservation in self._iter_active_usage_reservations():
            effect = self.effects.get_external_effect(reservation.reserved_by)
            release = effect is None or effect.transaction_state == "prepared"
            self._settle_usage_reservation(
                reservation.reservation_id,
                release=release,
                charge_maximum=not release,
                source="resource.recovery",
                context={
                    "effect_id": reservation.reserved_by,
                    "outcome": (
                        "not_started" if release else "unknown_after_dispatch"
                    ),
                },
                recovery=True,
            )
            total_count += 1
            if len(sample) < page_size:
                sample.append(reservation.reservation_id)
        return ResourceUsageReservationRecoverySummary(
            total_count=total_count,
            sample_reservation_ids=tuple(sample),
        )

    def _iter_active_usage_reservations(
        self,
    ) -> Iterator[ResourceUsageReservation]:
        page_size = (
            self.config.runtime.resource_usage_reservation_recovery_page_size
        )
        after: ResourceUsageReservationCursor | None = None
        while True:
            page = self.resource_repository.query_resource_usage_reservation_recovery(
                after=after,
                limit=page_size,
            )
            if len(page.records) > page_size:
                raise ValidationError(
                    "resource usage reservation recovery repository exceeded the page limit"
                )
            previous = after
            for reservation in page.records:
                cursor = ResourceUsageReservationCursor(
                    reservation.created_at,
                    reservation.reservation_id,
                )
                if (
                    reservation.status is not ResourceUsageReservationStatus.ACTIVE
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "resource usage reservation recovery repository returned "
                        "an invalid page"
                    )
                previous = cursor
            if page.next_cursor is not None and (
                previous is None or page.next_cursor != previous
            ):
                raise ValidationError(
                    "resource usage reservation recovery repository returned "
                    "an invalid next cursor"
                )
            yield from page.records
            if page.next_cursor is None:
                break
            after = page.next_cursor

    def charge(
        self,
        pid: str,
        usage: ResourceUsage | dict[str, Any],
        *,
        source: str,
        context: dict[str, Any] | None = None,
        allow_overage: bool = False,
        kill_on_exceed: bool = True,
    ) -> None:
        self._charge(
            pid,
            usage,
            source=source,
            context=context,
            allow_overage=allow_overage,
            kill_on_exceed=kill_on_exceed,
            include_active_reservations=True,
        )

    def _charge(
        self,
        pid: str,
        usage: ResourceUsage | dict[str, Any],
        *,
        source: str,
        context: dict[str, Any] | None,
        allow_overage: bool,
        kill_on_exceed: bool,
        include_active_reservations: bool,
        deferred_resource_limit_finalizations: list[_ResourceLimitFinalization] | None = None,
    ) -> None:
        delta = self._coerce_usage(usage)
        if self._is_zero(delta):
            return
        exceeded_after_charge: tuple[AgentProcess, dict[str, Any]] | None = None
        # A hierarchical charge is one accounting mutation.  In particular,
        # never leave the child charged when an ancestor update, reservation
        # consume, event, or audit write fails part-way through the chain.
        with self.unit_of_work.transaction():
            chain = self._process_chain(pid)
            relevant_fields = self._nonzero_fields(delta)
            if not allow_overage:
                self._preflight_locked(pid, delta, source=source, context=context)
            for index, process in enumerate(chain):
                latest = self._get(process.pid)
                latest.resource_usage = self._merge_usage(latest.resource_usage, delta)
                latest.updated_at = utc_now()
                patch = {
                    "resource_usage": latest.resource_usage,
                    "updated_at": latest.updated_at,
                }
                terminal_scope = (
                    trusted_terminal_process_mutation(
                        latest.pid,
                        expected_revision=latest.revision,
                        expected_generation=latest.execution_generation,
                        allowed_statuses={latest.status},
                        execution_token=current_process_execution_token(),
                        reason="resource accounting appends usage to a terminal process",
                    )
                    if latest.status in _TERMINAL_STATUSES
                    else nullcontext()
                )
                with terminal_scope:
                    if index == 0:
                        latest = self.store.patch_process(
                            latest.pid,
                            patch,
                            expected_revision=latest.revision,
                        )
                    else:
                        latest = self.store.patch_process_control(
                            latest.pid,
                            patch,
                            expected_revision=latest.revision,
                            allowed_statuses={latest.status},
                            reason="hierarchical resource charge updates an ancestor",
                        )
                if index > 0:
                    self._consume_reservation_locked(latest.pid, chain[index - 1].pid, delta, relevant_fields)
                exceeded = self._first_exceeded_effective(
                    latest,
                    relevant_fields=relevant_fields,
                    include_active_reservations=include_active_reservations,
                )
                if exceeded is not None and exceeded_after_charge is None:
                    exceeded_after_charge = (latest, exceeded)
            self.events.emit(
                EventType.RESOURCE_CHARGED,
                source=source,
                target=pid,
                payload={"pid": pid, "usage": to_jsonable(delta), "context": context or {}},
            )
            self.audit.record(
                actor=pid,
                action="resource.charge",
                target=f"process:{pid}",
                decision={
                    "source": source,
                    "usage": to_jsonable(delta),
                    "charged_pids": [process.pid for process in chain],
                    "context": context or {},
                },
            )
        if exceeded_after_charge is None:
            return
        owner, exceeded = exceeded_after_charge
        message = self._limit_message(owner.pid, exceeded)
        # The durable kill may join a caller's outer transaction.  Its terminal
        # hooks acquire Human/Object Memory locks, so either run them after this
        # direct charge committed or return a receipt to the outer committer.
        if kill_on_exceed:
            finalization = self._persist_resource_limit_kill(
                owner.pid,
                reason=message,
                owner_pid=owner.pid,
                limit=exceeded,
            )
            if deferred_resource_limit_finalizations is None:
                self._finalize_resource_limit(finalization)
            else:
                deferred_resource_limit_finalizations.append(finalization)
        raise ResourceLimitExceeded(message)

    def kill_if_exceeded(
        self,
        pid: str,
        *,
        reason: str,
        owner_pid: str | None = None,
        limit: dict[str, Any] | None = None,
    ) -> None:
        finalization = self._persist_resource_limit_kill(
            pid,
            reason=reason,
            owner_pid=owner_pid,
            limit=limit,
        )
        self._finalize_resource_limit(finalization)

    def _persist_resource_limit_kill(
        self,
        pid: str,
        *,
        reason: str,
        owner_pid: str | None,
        limit: dict[str, Any] | None,
    ) -> _ResourceLimitFinalization:
        killed: list[str] = []
        # Persist the complete descendant state transition, reservation
        # release, and corresponding evidence atomically.  Cross-subsystem
        # terminal hooks run only after this transaction releases the store
        # lock, avoiding a store -> terminal-lock / terminal-lock -> store
        # inversion with HumanRequestManager.
        with self.unit_of_work.transaction():
            for process in self._descendant_tree(pid):
                if process.status in _TERMINAL_STATUSES:
                    continue
                previous_status = process.status
                takeover_scope = self._resource_limit_takeover_scope(process)
                with takeover_scope:
                    self._transitions.transition(
                        process.pid,
                        ProcessStatus.KILLED,
                        expected_revision=process.revision,
                        expected_state_generation=process.state_generation,
                        outcome=KilledProcessOutcome(code="resource_limit_exceeded"),
                        status_message=reason,
                        control=True,
                        allowed_statuses={previous_status},
                        reason="resource limit terminates an affected process tree",
                    )
                self.resource_repository.delete_resource_reservations_for_process(
                    process.pid
                )
                killed.append(process.pid)
            self.events.emit(
                EventType.RESOURCE_LIMIT_EXCEEDED,
                source="resource_manager",
                target=pid,
                priority=EventPriority.CRITICAL,
                payload={"pid": pid, "owner_pid": owner_pid or pid, "reason": reason, "killed_pids": killed, "limit": limit or {}},
            )
            self.audit.record(
                actor="resource_manager",
                action="resource.limit_exceeded",
                target=f"process:{pid}",
                decision={"reason": reason, "owner_pid": owner_pid or pid, "killed_pids": killed, "limit": limit or {}},
            )
        return _ResourceLimitFinalization(
            pid=pid,
            killed_pids=tuple(killed),
            reason=reason,
        )

    def _run_resource_limit_finalizations(
        self,
        finalizations: Iterable[_ResourceLimitFinalization],
    ) -> None:
        for finalization in finalizations:
            self._finalize_resource_limit(finalization)

    def _finalize_resource_limit(
        self,
        finalization: _ResourceLimitFinalization,
    ) -> None:
        pid = finalization.pid
        killed = list(finalization.killed_pids)
        reason = finalization.reason
        finalizer_errors: list[dict[str, Any]] = []
        interruptions: list[BaseException] = []
        for killed_pid in killed:
            try:
                self._wake_parent_waiting_on_child(killed_pid)
            except BaseException as exc:
                finalizer_errors.append(
                    self._finalizer_error_record("wake_parent", killed_pid, exc)
                )
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
            try:
                self._notify_object_task_process_terminal(killed_pid)
            except BaseException as exc:
                finalizer_errors.append(
                    self._finalizer_error_record("terminal_notify", killed_pid, exc)
                )
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
        if killed and self._process_kill_finalizer is not None:
            try:
                self._process_kill_finalizer(killed, reason=reason)
            except BaseException as exc:
                finalizer_errors.append(
                    self._finalizer_error_record("process_finalize", pid, exc)
                )
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
        if finalizer_errors:
            try:
                self.audit.record(
                    actor="resource_manager",
                    action="resource.limit_finalize_failed",
                    target=f"process:{pid}",
                    decision={"reason": reason, "errors": finalizer_errors},
                )
            except BaseException as exc:
                # The terminal state is already committed.  Do not replace the
                # caller's ResourceLimitExceeded with a secondary warning-sink
                # failure or skip the remaining cleanup callbacks.
                if not isinstance(exc, Exception):
                    interruptions.append(exc)
        if interruptions:
            raise BaseExceptionGroup(
                "resource limit finalization interrupted after full cleanup",
                interruptions,
            )

    @staticmethod
    def _finalizer_error_record(
        phase: str,
        pid: str,
        error: BaseException,
    ) -> dict[str, Any]:
        envelope = public_error_envelope(
            error,
            code="resource_finalize_failed",
        )
        observation = internal_exception_observation(
            error,
            correlation_id=envelope["correlation_id"],
        )
        return {
            "phase": phase,
            "pid": pid,
            "code": envelope["code"],
            "error_type": envelope["error_type"],
            "correlation_id": envelope["correlation_id"],
            "exception_text": observation["exception_text"],
        }

    @staticmethod
    def _resource_limit_takeover_scope(process: AgentProcess) -> Any:
        if process.status != ProcessStatus.RUNNING:
            return nullcontext()
        if (
            process.execution_owner_id is None
            and process.execution_lease_id is None
        ):
            return nullcontext()
        if process.execution_owner_id is None or process.execution_lease_id is None:
            raise ValidationError(
                f"running process has an incomplete execution lease: {process.pid}"
            )
        return trusted_process_execution_takeover(
            process.pid,
            source_revision=process.revision,
            source_state_generation=process.state_generation,
            source_execution_token=ProcessExecutionToken(
                pid=process.pid,
                generation=process.execution_generation,
                owner_id=process.execution_owner_id,
                lease_id=process.execution_lease_id,
            ),
            intended_status=ProcessStatus.KILLED,
            reason="resource limit takes over an execution lease",
            nonce=new_id("process_takeover"),
            outcome_code="resource_limit_exceeded",
        )

    def _wake_parent_waiting_on_child(self, child_pid: str) -> None:
        with self.unit_of_work.transaction():
            child = self.store.get_process(child_pid)
            if child is None or child.parent_pid is None:
                return
            parent = self.store.get_process(child.parent_pid)
            if parent is None:
                return
            if parent.status != ProcessStatus.WAITING_EVENT:
                return
            if not isinstance(parent.wait_state, ChildProcessWait):
                return
            if parent.wait_state.child_pid != child.pid:
                return
            token = self._transitions.wait_token(parent)
            self._transitions.wake(
                token,
                control=True,
                reason="resource termination wakes a waiting parent",
            )
            self.audit.record(
                actor="resource_manager",
                action="process.wait_wake",
                target=f"process:{parent.pid}",
                decision={"child": child.pid, "child_status": child.status.value},
            )

    def _notify_object_task_process_terminal(self, pid: str) -> None:
        if self._object_task_terminal_notifier is None:
            return
        self._object_task_terminal_notifier(pid)

    def has_limit(self, pid: str, budget_field: str) -> bool:
        if budget_field not in _BUDGET_USAGE_MAP:
            raise ValidationError(f"unknown resource budget field: {budget_field}")
        return any(getattr(process.resource_budget, budget_field) is not None for process in self._process_chain(pid))

    def remaining_cumulative(
        self,
        pid: str,
        budget_field: str,
        usage_field: str,
    ) -> _ResourceNumber | None:
        if budget_field not in _BUDGET_USAGE_MAP or usage_field not in _USAGE_FIELD_NAMES:
            raise ValidationError(f"unknown resource remaining query: {budget_field}/{usage_field}")
        with self.unit_of_work.locked():
            return self._remaining_budget_field_locked(pid, budget_field, (usage_field,))

    def peak_limit(self, pid: str, budget_field: str) -> int | None:
        if budget_field not in _BUDGET_USAGE_MAP:
            raise ValidationError(f"unknown resource budget field: {budget_field}")
        selected: int | None = None
        for process in self._process_chain(pid):
            limit = getattr(process.resource_budget, budget_field)
            if limit is None:
                continue
            value = int(limit)
            selected = value if selected is None else min(selected, value)
        return selected

    def validate_child_budget(
        self,
        parent_pid: str,
        child_budget: ResourceBudget,
        *,
        reserved_usage: ResourceUsage | None = None,
    ) -> None:
        reserve = reserved_usage or ResourceUsage()
        self._coerce_usage(reserve)
        with self.unit_of_work.locked():
            if not self._is_zero(reserve):
                self._preflight_locked(
                    parent_pid,
                    reserve,
                    source="resource.validate_child_budget",
                    context={"child_budget": to_jsonable(child_budget)},
                )
            parent_window = self.context_materialization_window_limit(parent_pid)
            if child_budget.max_context_materialization_tokens > parent_window:
                raise ResourceLimitExceeded(
                    "child budget max_context_materialization_tokens="
                    f"{child_budget.max_context_materialization_tokens} exceeds parent limit {parent_window}"
                )
            for budget_field, usage_fields in _BUDGET_USAGE_MAP.items():
                requested = getattr(child_budget, budget_field)
                if requested is None:
                    continue
                if budget_field in _NON_RESERVABLE_BUDGET_FIELDS:
                    limit = self.peak_limit(parent_pid, budget_field)
                    if limit is not None and requested > limit:
                        raise ResourceLimitExceeded(
                            f"child budget {budget_field}={requested} exceeds parent limit {limit}"
                        )
                    continue
                remaining = self._remaining_budget_field_locked(
                    parent_pid,
                    budget_field,
                    usage_fields,
                    reserved_usage=reserve,
                )
                if remaining is None:
                    continue
                if requested > remaining:
                    raise ResourceLimitExceeded(
                        f"child budget {budget_field}={requested} exceeds parent remaining {remaining:g}"
                    )

    def remaining_budget(self, pid: str) -> ResourceBudget:
        return self.remaining_budgets([pid])[pid]

    def remaining_budgets(self, pids: Iterable[str]) -> dict[str, ResourceBudget]:
        """Compute hierarchical remaining budgets with a constant query count."""

        selected = sorted({str(pid) for pid in pids if str(pid)})
        if not selected:
            return {}
        with self.unit_of_work.locked():
            process_by_pid = {
                process.pid: process
                for process in self.store.get_processes_with_ancestors(selected)
            }
            missing = [pid for pid in selected if pid not in process_by_pid]
            if missing:
                raise NotFound(f"process not found: {missing[0]}")
            reservations = self.resource_repository.list_resource_reservations(
                parent_pids=process_by_pid
            )
            reserved_by_parent: dict[str, dict[str, _ResourceNumber]] = {}
            for reservation in reservations:
                totals = reserved_by_parent.setdefault(reservation.parent_pid, {})
                for budget_field, value in reservation.reserved.items():
                    normalized = self._budget_number(budget_field, value)
                    totals[budget_field] = (
                        totals.get(budget_field, self._budget_zero(budget_field))
                        + normalized
                    )

            result: dict[str, ResourceBudget] = {}
            for pid in selected:
                chain: list[AgentProcess] = []
                seen: set[str] = set()
                current = process_by_pid[pid]
                while True:
                    if current.pid in seen:
                        raise ValidationError(f"process parent cycle detected: {current.pid}")
                    seen.add(current.pid)
                    chain.append(current)
                    if current.parent_pid is None:
                        break
                    parent = process_by_pid.get(current.parent_pid)
                    if parent is None:
                        raise NotFound(f"process not found: {current.parent_pid}")
                    current = parent

                values: dict[str, Any] = {
                    "max_context_materialization_tokens": min(
                        int(process.resource_budget.max_context_materialization_tokens)
                        for process in chain
                    ),
                }
                for budget_field, usage_fields in _BUDGET_USAGE_MAP.items():
                    if budget_field == "max_subprocess_memory_bytes":
                        limits = [
                            int(limit)
                            for process in chain
                            if (limit := getattr(process.resource_budget, budget_field)) is not None
                        ]
                        values[budget_field] = min(limits) if limits else None
                        continue
                    remaining: _ResourceNumber | None = None
                    for process in chain:
                        limit = getattr(process.resource_budget, budget_field)
                        if limit is None:
                            continue
                        used = self._usage_value(
                            process.resource_usage,
                            budget_field,
                            usage_fields,
                        )
                        if budget_field not in _NON_RESERVABLE_BUDGET_FIELDS:
                            used += reserved_by_parent.get(process.pid, {}).get(
                                budget_field,
                                self._budget_zero(budget_field),
                            )
                            used += self._reserved_usage_value_locked(process.pid, budget_field)
                        process_remaining = max(
                            self._budget_zero(budget_field),
                            self._budget_number(budget_field, limit) - used,
                        )
                        remaining = process_remaining if remaining is None else min(remaining, process_remaining)
                    values[budget_field] = remaining
                result[pid] = ResourceBudget(**values)
            return result

    def reserve_child_budget(self, parent_pid: str, child_pid: str, child_budget: ResourceBudget) -> None:
        with self.unit_of_work.locked():
            self.validate_child_budget(parent_pid, child_budget, reserved_usage=ResourceUsage(child_processes=1))
            reserved = self._reservation_from_budget(child_budget)
            if not reserved:
                return
            now = utc_now()
            self.resource_repository.upsert_resource_reservation(
                ResourceReservation(
                    parent_pid=parent_pid,
                    child_pid=child_pid,
                    reserved=reserved,
                    created_at=now,
                    updated_at=now,
                )
            )
            self.audit.record(
                actor=parent_pid,
                action="resource.reserve_child_budget",
                target=f"process:{child_pid}",
                decision={"reserved": reserved},
            )

    def release_process_reservations(self, pid: str) -> None:
        with self.unit_of_work.locked():
            reservations = self.resource_repository.list_resource_reservations(
                child_pid=pid
            )
            reservations.extend(
                self.resource_repository.list_resource_reservations(parent_pid=pid)
            )
            self.resource_repository.delete_resource_reservations_for_process(pid)
            if reservations:
                self.audit.record(
                    actor="resource_manager",
                    action="resource.release_process_reservations",
                    target=f"process:{pid}",
                    decision={
                        "reservations": [
                            {
                                "parent_pid": item.parent_pid,
                                "child_pid": item.child_pid,
                                "reserved": item.reserved,
                            }
                            for item in reservations
                        ]
                    },
                )

    def context_materialization_window_limit(self, pid: str) -> int:
        selected: int | None = None
        for process in self._process_chain(pid):
            value = int(process.resource_budget.max_context_materialization_tokens)
            selected = value if selected is None else min(selected, value)
        return selected if selected is not None else 0

    def _get(self, pid: str) -> AgentProcess:
        process = self.store.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        return process

    def _preflight_locked(
        self,
        pid: str,
        usage: ResourceUsage,
        *,
        source: str,
        context: dict[str, Any] | None,
    ) -> None:
        relevant_fields = self._nonzero_fields(usage)
        chain = self._process_chain(pid)
        for index, process in enumerate(chain):
            candidate = self._merge_usage(process.resource_usage, usage)
            consuming_child_pid = chain[index - 1].pid if index > 0 else None
            exceeded = self._first_exceeded_effective(
                process,
                usage=candidate,
                relevant_fields=relevant_fields,
                consuming_child_pid=consuming_child_pid,
                consuming_usage=usage,
            )
            if exceeded is None:
                continue
            message = self._limit_message(process.pid, exceeded)
            self.audit.record(
                actor=pid,
                action="resource.preflight_denied",
                target=f"process:{process.pid}",
                decision={
                    "source": source,
                    "context": context or {},
                    "requested_usage": to_jsonable(usage),
                    "limit": exceeded,
                    "message": message,
                },
            )
            raise ResourceLimitExceeded(message)

    def _remaining_budget_field_locked(
        self,
        pid: str,
        budget_field: str,
        usage_fields: tuple[str, ...],
        *,
        reserved_usage: ResourceUsage | None = None,
    ) -> _ResourceNumber | None:
        remaining: _ResourceNumber | None = None
        for process in self._process_chain(pid):
            limit = getattr(process.resource_budget, budget_field)
            if limit is None:
                continue
            usage = (
                self._merge_usage(process.resource_usage, reserved_usage)
                if reserved_usage is not None
                else process.resource_usage
            )
            value = self._usage_value(usage, budget_field, usage_fields)
            if budget_field not in _NON_RESERVABLE_BUDGET_FIELDS:
                value += self._reserved_budget_value_locked(process.pid, budget_field)
                value += self._reserved_usage_value_locked(process.pid, budget_field)
            process_remaining = max(
                self._budget_zero(budget_field),
                self._budget_number(budget_field, limit) - value,
            )
            remaining = process_remaining if remaining is None else min(remaining, process_remaining)
        return remaining

    def _process_chain(self, pid: str) -> list[AgentProcess]:
        chain: list[AgentProcess] = []
        current = self._get(pid)
        while True:
            chain.append(current)
            if current.parent_pid is None:
                return chain
            current = self._get(current.parent_pid)

    def _descendant_tree(self, pid: str) -> list[AgentProcess]:
        selected: list[AgentProcess] = []
        stack = [self._get(pid)]
        while stack:
            process = stack.pop()
            selected.append(process)
            stack.extend(self.store.list_child_processes(process.pid))
        return selected

    def _reservation_from_budget(
        self,
        budget: ResourceBudget,
    ) -> dict[str, _ResourceNumber]:
        reserved: dict[str, _ResourceNumber] = {}
        for budget_field in _BUDGET_USAGE_MAP:
            if budget_field in _NON_RESERVABLE_BUDGET_FIELDS:
                continue
            value = getattr(budget, budget_field)
            if value is None:
                continue
            reserved[budget_field] = self._budget_number(budget_field, value)
        return reserved

    def _consume_reservation_locked(
        self,
        parent_pid: str,
        child_pid: str,
        usage: ResourceUsage,
        relevant_fields: set[str],
    ) -> None:
        reservation = self.resource_repository.get_resource_reservation(
            parent_pid,
            child_pid,
        )
        if reservation is None:
            return
        changed = False
        remaining = dict(reservation.reserved)
        for budget_field, usage_fields in _BUDGET_USAGE_MAP.items():
            if budget_field in _NON_RESERVABLE_BUDGET_FIELDS:
                continue
            if not (set(usage_fields) & relevant_fields):
                continue
            current = self._budget_number(
                budget_field,
                remaining.get(budget_field, self._budget_zero(budget_field)),
            )
            if current <= 0:
                continue
            consumed = min(current, self._usage_value(usage, budget_field, usage_fields))
            if consumed <= 0:
                continue
            next_value = current - consumed
            if next_value <= 0:
                remaining.pop(budget_field, None)
            else:
                remaining[budget_field] = next_value
            changed = True
        if not changed:
            return
        if remaining:
            self.resource_repository.upsert_resource_reservation(
                ResourceReservation(
                    parent_pid=parent_pid,
                    child_pid=child_pid,
                    reserved=remaining,
                    created_at=reservation.created_at,
                    updated_at=utc_now(),
                )
            )
        else:
            self.resource_repository.delete_resource_reservation(parent_pid, child_pid)

    def _reserved_budget_value_locked(
        self,
        parent_pid: str,
        budget_field: str,
        *,
        consuming_child_pid: str | None = None,
        consuming_usage: ResourceUsage | None = None,
    ) -> _ResourceNumber:
        total = self._budget_zero(budget_field)
        usage_fields = _BUDGET_USAGE_MAP[budget_field]
        for reservation in self.resource_repository.list_resource_reservations(
            parent_pid=parent_pid
        ):
            value = self._budget_number(
                budget_field,
                reservation.reserved.get(
                    budget_field,
                    self._budget_zero(budget_field),
                ),
            )
            if value <= 0:
                continue
            if consuming_child_pid == reservation.child_pid and consuming_usage is not None:
                value = max(
                    self._budget_zero(budget_field),
                    value
                    - self._usage_value(
                        consuming_usage,
                        budget_field,
                        usage_fields,
                    ),
                )
            total += value
        return total

    def _reserved_usage_value_locked(
        self,
        owner_pid: str,
        budget_field: str,
    ) -> _ResourceNumber:
        usage_fields = _BUDGET_USAGE_MAP[budget_field]
        total = self._budget_zero(budget_field)
        for reservation in self._iter_active_usage_reservations():
            try:
                chain_pids = {
                    process.pid for process in self._process_chain(reservation.pid)
                }
            except NotFound:
                # A durable reservation whose process vanished is uncertain.
                # Keep it visible to the root budget instead of silently freeing it.
                chain_pids = {reservation.pid}
            if owner_pid not in chain_pids:
                continue
            total += self._usage_value(reservation.usage, budget_field, usage_fields)
        return total

    def _usage_value(
        self,
        usage: ResourceUsage,
        budget_field: str,
        usage_fields: tuple[str, ...],
    ) -> _ResourceNumber:
        if budget_field == "max_subprocess_memory_bytes":
            return getattr(usage, "subprocess_peak_memory_bytes")
        return sum(getattr(usage, usage_field) for usage_field in usage_fields)

    @staticmethod
    def _budget_zero(budget_field: str) -> _ResourceNumber:
        return 0.0 if budget_field in _CONTINUOUS_BUDGET_FIELDS else 0

    @staticmethod
    def _budget_number(
        budget_field: str,
        value: _ResourceNumber,
    ) -> _ResourceNumber:
        """Normalize by field domain without rounding discrete counters."""

        if budget_field in _CONTINUOUS_BUDGET_FIELDS:
            return float(value)
        if isinstance(value, bool):
            raise ValidationError(
                f"discrete resource value must be an integer: {budget_field}"
            )
        if isinstance(value, int):
            return value
        # Older stores wrote reservation JSON using floats.  Preserve safe,
        # exactly integral legacy values while refusing ambiguous fractions.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValidationError(
            f"discrete resource value must be an integer: {budget_field}"
        )

    def _coerce_usage(self, usage: ResourceUsage | dict[str, Any]) -> ResourceUsage:
        if isinstance(usage, ResourceUsage):
            self._validate_usage(usage)
            return usage
        unknown = sorted(set(usage) - _USAGE_FIELD_NAMES)
        if unknown:
            raise ValidationError("resource usage contains unknown fields")
        try:
            coerced = ResourceUsage(**dict(usage))
        except ValueError as exc:
            raise ValidationError("resource usage values are invalid") from exc
        self._validate_usage(coerced)
        return coerced

    def _merge_usage(self, current: ResourceUsage, delta: ResourceUsage) -> ResourceUsage:
        values: dict[str, Any] = {}
        for field in fields(ResourceUsage):
            name = field.name
            current_value = getattr(current, name)
            delta_value = getattr(delta, name)
            if name in _PEAK_USAGE_FIELDS:
                values[name] = max(current_value, delta_value)
            else:
                values[name] = current_value + delta_value
        return ResourceUsage(**values)

    def _first_exceeded(
        self,
        budget: ResourceBudget,
        usage: ResourceUsage,
        *,
        relevant_fields: set[str] | None = None,
    ) -> dict[str, Any] | None:
        for budget_field, usage_fields in _BUDGET_USAGE_MAP.items():
            if relevant_fields is not None and not (set(usage_fields) & relevant_fields):
                continue
            limit = getattr(budget, budget_field)
            if limit is None:
                continue
            if budget_field == "max_subprocess_memory_bytes":
                value = getattr(usage, "subprocess_peak_memory_bytes")
            else:
                value = sum(getattr(usage, usage_field) for usage_field in usage_fields)
            if value > limit:
                return {"budget": budget_field, "usage": list(usage_fields), "value": value, "limit": limit}
        return None

    def _first_exceeded_effective(
        self,
        process: AgentProcess,
        *,
        usage: ResourceUsage | None = None,
        relevant_fields: set[str] | None = None,
        consuming_child_pid: str | None = None,
        consuming_usage: ResourceUsage | None = None,
        include_active_reservations: bool = True,
    ) -> dict[str, Any] | None:
        selected_usage = usage or process.resource_usage
        for budget_field, usage_fields in _BUDGET_USAGE_MAP.items():
            if relevant_fields is not None and not (set(usage_fields) & relevant_fields):
                continue
            limit = getattr(process.resource_budget, budget_field)
            if limit is None:
                continue
            value = self._usage_value(selected_usage, budget_field, usage_fields)
            if budget_field not in _NON_RESERVABLE_BUDGET_FIELDS:
                value += self._reserved_budget_value_locked(
                    process.pid,
                    budget_field,
                    consuming_child_pid=consuming_child_pid,
                    consuming_usage=consuming_usage,
                )
                if include_active_reservations:
                    value += self._reserved_usage_value_locked(
                        process.pid,
                        budget_field,
                    )
            if value > limit:
                return {"budget": budget_field, "usage": list(usage_fields), "value": value, "limit": limit}
        return None

    def _is_zero(self, usage: ResourceUsage) -> bool:
        return all(getattr(usage, name) == 0 for name in _USAGE_FIELD_NAMES)

    def _nonzero_fields(self, usage: ResourceUsage) -> set[str]:
        return {name for name in _USAGE_FIELD_NAMES if getattr(usage, name) != 0}

    def _validate_usage(self, usage: ResourceUsage) -> None:
        try:
            usage.validate()
        except ValueError as exc:
            raise ValidationError("resource usage values are invalid") from exc

    def _assert_usage_within_reservation(
        self,
        actual: ResourceUsage,
        maximum: ResourceUsage,
    ) -> None:
        exceeded = [
            name
            for name in _USAGE_FIELD_NAMES
            if getattr(actual, name) > getattr(maximum, name)
        ]
        if exceeded:
            raise ResourceLimitExceeded(
                "resource settlement exceeded reserved maximum for fields: "
                + ", ".join(sorted(exceeded))
            )

    def _limit_message(self, pid: str, exceeded: dict[str, Any]) -> str:
        if exceeded["budget"] == "max_child_processes":
            return (
                f"process {pid} exhausted child process budget: "
                f"{exceeded['value']}/{exceeded['limit']}"
            )
        return (
            f"process {pid} exceeded {exceeded['budget']}: "
            f"{exceeded['value']} > {exceeded['limit']}"
        )
