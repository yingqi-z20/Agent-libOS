from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from agent_libos.models import (
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptPage,
    CheckpointPayloadDeliveryAttemptState,
    OperationKind,
    OperationOutcome,
    OperationState,
    PayloadDeliveryState,
    RuntimePublicationCursor,
    RuntimePublicationPage,
    RuntimePublicationRecord,
)
from agent_libos.models.exceptions import (
    DurableObjectFinalizerUnavailable,
    RuntimePublicationPending,
    ValidationError,
)
from agent_libos.ports.operations import RuntimePublicationOperationPort
from agent_libos.ports.publication import (
    CheckpointRestorePublicationReader,
    CheckpointRestorePublicationWriterPort,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps


CHECKPOINT_RESTORE_PUBLICATION_KIND = "checkpoint_restore"
CHECKPOINT_RESTORE_PLAN_VERSION = 2
CHECKPOINT_RESTORE_PLAN_ANCHOR_VERSION = 2
CHECKPOINT_RESTORE_V1_PHASES = (
    "image_reconciliation",
    "jit_source_reconciliation",
    "jit_pruning",
    "object_release_finalizers",
)
CHECKPOINT_RESTORE_PHASES = (
    "object_payload_reconciliation",
    "image_reconciliation",
    "jit_source_reconciliation",
    "jit_pruning",
    "object_release_finalizers",
)


@dataclass(frozen=True, slots=True)
class CheckpointRestorePlan:
    checkpoint_id: str
    pid: str
    actor: str
    operation_id: str
    snapshot_version: int
    snapshot_sha256: str
    current_pids: tuple[str, ...]
    snapshot_pids: tuple[str, ...]
    scoped_pids: tuple[str, ...]
    stale_tool_ids: tuple[str, ...]
    finalizer_work_items: tuple[dict[str, Any], ...]
    plan_version: int = CHECKPOINT_RESTORE_PLAN_VERSION
    phase_order: tuple[str, ...] = CHECKPOINT_RESTORE_PHASES

    def to_mapping(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "checkpoint_id": self.checkpoint_id,
            "pid": self.pid,
            "actor": self.actor,
            "operation_id": self.operation_id,
            "operation_binding_version": 1,
            "snapshot_version": self.snapshot_version,
            "snapshot_sha256": self.snapshot_sha256,
            "current_pids": list(self.current_pids),
            "snapshot_pids": list(self.snapshot_pids),
            "scoped_pids": list(self.scoped_pids),
            "stale_tool_ids": list(self.stale_tool_ids),
            "phase_order": list(self.phase_order),
            "finalizer_work_items": [dict(item) for item in self.finalizer_work_items],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointRestorePlan":
        expected_keys = {
            "plan_version",
            "checkpoint_id",
            "pid",
            "actor",
            "operation_id",
            "operation_binding_version",
            "snapshot_version",
            "snapshot_sha256",
            "current_pids",
            "snapshot_pids",
            "scoped_pids",
            "stale_tool_ids",
            "phase_order",
            "finalizer_work_items",
        }
        if set(value) != expected_keys:
            raise ValidationError("checkpoint restore publication plan shape is invalid")
        plan_version = value.get("plan_version")
        if type(plan_version) is not int:
            raise ValidationError("checkpoint restore publication plan version is invalid")
        expected_phase_order = {
            1: CHECKPOINT_RESTORE_V1_PHASES,
            CHECKPOINT_RESTORE_PLAN_VERSION: CHECKPOINT_RESTORE_PHASES,
        }.get(plan_version)
        if expected_phase_order is None:
            raise ValidationError("checkpoint restore publication plan version is invalid")
        operation_binding_version = value.get("operation_binding_version")
        if type(operation_binding_version) is not int or operation_binding_version != 1:
            raise ValidationError("checkpoint restore operation binding version is invalid")
        phase_order = tuple(value.get("phase_order") or ())
        if phase_order != expected_phase_order:
            raise ValidationError("checkpoint restore phase order is invalid")
        finalizer_items = value.get("finalizer_work_items")
        if not isinstance(finalizer_items, list) or any(
            not isinstance(item, dict) for item in finalizer_items
        ):
            raise ValidationError("checkpoint restore finalizer work must be a list of objects")
        plan = cls(
            checkpoint_id=cls._required_text(value, "checkpoint_id"),
            pid=cls._required_text(value, "pid"),
            actor=cls._required_text(value, "actor"),
            operation_id=cls._required_text(value, "operation_id"),
            snapshot_version=cls._required_int(value, "snapshot_version"),
            snapshot_sha256=cls._required_digest(value, "snapshot_sha256"),
            current_pids=cls._text_tuple(value, "current_pids"),
            snapshot_pids=cls._text_tuple(value, "snapshot_pids"),
            scoped_pids=cls._text_tuple(value, "scoped_pids"),
            stale_tool_ids=cls._text_tuple(value, "stale_tool_ids"),
            finalizer_work_items=cls._validated_finalizer_items(finalizer_items),
            plan_version=int(plan_version),
            phase_order=phase_order,
        )
        if set(plan.scoped_pids) != set(plan.current_pids) | set(plan.snapshot_pids):
            raise ValidationError("checkpoint restore publication scope is inconsistent")
        return plan

    @classmethod
    def _validated_finalizer_items(
        cls,
        finalizer_items: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        work_ids: list[str] = []
        selected_items: list[dict[str, Any]] = []
        for item in finalizer_items:
            if set(item) != {
                "work_id",
                "finalizer_id",
                "object_oid",
                "object_version",
                "intent",
                "intent_sha256",
            }:
                raise ValidationError(
                    "checkpoint restore finalizer work item shape is invalid"
                )
            work_id = cls._required_text(item, "work_id")
            finalizer_id = cls._required_text(item, "finalizer_id")
            if len(finalizer_id) > 256 or any(
                ord(char) < 33 or ord(char) == 127 for char in finalizer_id
            ):
                raise ValidationError(
                    "checkpoint restore finalizer handler id is invalid"
                )
            cls._required_text(item, "object_oid")
            cls._required_int(item, "object_version")
            intent = item.get("intent")
            if not isinstance(intent, dict):
                raise ValidationError(
                    "checkpoint restore finalizer intent is invalid"
                )
            expected_digest = cls._required_digest(item, "intent_sha256")
            if hashlib.sha256(dumps(intent).encode("utf-8")).hexdigest() != expected_digest:
                raise ValidationError(
                    "checkpoint restore finalizer intent digest is invalid"
                )
            work_ids.append(work_id)
            selected_items.append(dict(item))
        if len(work_ids) != len(set(work_ids)):
            raise ValidationError("checkpoint restore finalizer work ids are invalid")
        return tuple(selected_items)

    @staticmethod
    def _required_text(value: Mapping[str, Any], name: str) -> str:
        selected = value.get(name)
        if not isinstance(selected, str) or not selected:
            raise ValidationError(f"checkpoint restore publication {name} is invalid")
        return selected

    @staticmethod
    def _required_int(value: Mapping[str, Any], name: str) -> int:
        selected = value.get(name)
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
            raise ValidationError(f"checkpoint restore publication {name} is invalid")
        return selected

    @staticmethod
    def _required_digest(value: Mapping[str, Any], name: str) -> str:
        selected = CheckpointRestorePlan._required_text(value, name)
        if len(selected) != 64 or any(char not in "0123456789abcdef" for char in selected):
            raise ValidationError(f"checkpoint restore publication {name} is invalid")
        return selected

    @staticmethod
    def _text_tuple(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
        selected = value.get(name)
        if not isinstance(selected, list) or any(
            not isinstance(item, str) or not item for item in selected
        ):
            raise ValidationError(f"checkpoint restore publication {name} is invalid")
        values = tuple(selected)
        if len(values) != len(set(values)):
            raise ValidationError(f"checkpoint restore publication {name} contains duplicates")
        return values


class CheckpointRestoreReconciler:
    """Durably resume the forward-only work left by a committed restore."""

    def __init__(
        self,
        *,
        store: CheckpointRestorePublicationReader,
        writer: CheckpointRestorePublicationWriterPort,
        operations: RuntimePublicationOperationPort,
        owner_instance_id: str,
        recovery_max_attempts: int,
        reconciliation_page_size: int,
        registry_scope: Callable[[], AbstractContextManager[Any]],
        load_checkpoint: Callable[[str], tuple[Any, dict[str, Any], Any]],
        restore_object_payloads: Callable[[dict[str, Any]], None],
        restore_images: Callable[[dict[str, Any]], None],
        restore_jit_sources: Callable[[dict[str, Any]], None],
        prune_jit_tools: Callable[[set[str], set[str]], None],
        run_finalizer: Callable[[Mapping[str, Any]], None],
        record_failure: Callable[[str, Any, str, BaseException], dict[str, str]],
        require_recovery_lease: Callable[[], None],
        recovery_required: Callable[..., None] | None,
        recovery_terminalization_scope: Callable[
            [str], AbstractContextManager[Any]
        ],
    ) -> None:
        self._store = store
        self._writer = writer
        self._operations = operations
        self._owner_instance_id = str(owner_instance_id)
        self._recovery_max_attempts = int(recovery_max_attempts)
        self._reconciliation_page_size = int(reconciliation_page_size)
        self._registry_scope = registry_scope
        self._load_checkpoint = load_checkpoint
        self._restore_object_payloads = restore_object_payloads
        self._restore_images = restore_images
        self._restore_jit_sources = restore_jit_sources
        self._prune_jit_tools = prune_jit_tools
        self._run_finalizer = run_finalizer
        self._record_failure = record_failure
        self._require_recovery_lease = require_recovery_lease
        self._recovery_required = recovery_required
        self._recovery_terminalization_scope = recovery_terminalization_scope
        self._recovery_lock = threading.Lock()

    @staticmethod
    def new_publication_id() -> str:
        return new_id("publication")

    def begin(
        self,
        *,
        publication_id: str,
        actor: str,
        checkpoint: Any,
        snapshot: Mapping[str, Any],
        current_pids: Iterable[str],
        snapshot_pids: Iterable[str],
        stale_tool_ids: Iterable[str],
        finalizer_work_items: Iterable[Mapping[str, Any]],
    ) -> CheckpointRestorePlan:
        operation_id = str(self._operations.current_id() or "")
        if not operation_id:
            raise ValidationError("checkpoint restore publication requires an active operation")
        selected_current = tuple(dict.fromkeys(str(pid) for pid in current_pids))
        selected_snapshot = tuple(dict.fromkeys(str(pid) for pid in snapshot_pids))
        plan = CheckpointRestorePlan(
            checkpoint_id=str(checkpoint.checkpoint_id),
            pid=str(checkpoint.pid),
            actor=str(actor),
            operation_id=operation_id,
            snapshot_version=int(checkpoint.snapshot_version),
            snapshot_sha256=self.snapshot_sha256(snapshot),
            current_pids=selected_current,
            snapshot_pids=selected_snapshot,
            scoped_pids=tuple(dict.fromkeys((*selected_current, *selected_snapshot))),
            stale_tool_ids=tuple(sorted({str(tool_id) for tool_id in stale_tool_ids})),
            finalizer_work_items=tuple(dict(item) for item in finalizer_work_items),
        )
        with self._store.transaction():
            plan_mapping = plan.to_mapping()
            self._writer.insert_runtime_publication(
                publication_id=publication_id,
                kind=CHECKPOINT_RESTORE_PUBLICATION_KIND,
                pid=plan.pid,
                owner_instance_id=self._owner_instance_id,
                plan=plan_mapping,
            )
            if not self._writer.record_runtime_publication_artifact(
                publication_id,
                self._plan_anchor(publication_id, plan_mapping),
                expected_states={"planning"},
            ):
                raise ValidationError(
                    "checkpoint restore publication plan anchor was not persisted: "
                    f"{publication_id}"
                )
            self._operations.bind_runtime_publication(
                operation_id,
                publication_id=publication_id,
                publication_kind=CHECKPOINT_RESTORE_PUBLICATION_KIND,
                expected_kind="runtime",
                expected_name="checkpoint.restore",
                expected_actor=plan.actor,
                expected_pid=plan.actor,
            )
        return plan

    def mark_main_state_committed(self, publication_id: str) -> None:
        if not self._writer.advance_runtime_publication(
            publication_id,
            state="reconciliation_pending",
            phase="main_state_committed",
            receipt={"phase": "main_state_committed"},
            expected_states={"planning"},
            expected_phase="planned",
        ):
            raise ValidationError(
                f"checkpoint restore publication changed before main commit: {publication_id}"
            )

    def handle_main_commit_scope_escape(
        self,
        publication_id: str,
        primary: BaseException,
    ) -> None:
        """Classify an authority/UoW exit failure after the restore body ran."""

        try:
            current = self._store.get_runtime_publication(publication_id)
        except BaseException as confirmation_error:
            self._fence_preserving(publication_id, primary)
            self._raise_with_pending_publication(
                publication_id,
                "checkpoint restore main commit confirmation failed",
                primary,
                confirmation_error,
            )
        if current is None:
            return
        try:
            plan = self._validated_plan(current)
            self._require_main_commit_identity(current, plan)
            if current["state"] == "planning":
                return
            if current["state"] == "committed":
                if self._terminal_commit_confirmed(
                    publication_id,
                    publication=current,
                ):
                    return
                raise ValidationError(
                    "checkpoint restore terminal truth is incomplete after main commit"
                )
        except BaseException as confirmation_error:
            self._fence_preserving(publication_id, primary)
            self._raise_with_pending_publication(
                publication_id,
                "checkpoint restore durable main commit is invalid",
                primary,
                confirmation_error,
                fallback_publication=current,
            )
        self._fence_preserving(publication_id, primary)
        try:
            with self._recovery_terminalization_scope(publication_id):
                if current["state"] in {"failed", "manual"}:
                    with self._store.transaction():
                        self._reconcile_operation(current, OperationOutcome.UNKNOWN)
                    return
                if current["state"] != "reconciliation_pending":
                    raise ValidationError(
                        "checkpoint restore durable main commit state is invalid: "
                        f"{publication_id}/{current['state']}"
                    )
                checkpoint, _snapshot, _typed = self._load_checkpoint(
                    plan.checkpoint_id
                )
                phase = self._next_incomplete_phase(current, plan) or "terminalization"
                self._record_failure(plan.actor, checkpoint, phase, primary)
                self._mark_online_failure(
                    publication_id,
                    phase=phase,
                    error=primary,
                    manual=False,
                )
        except BaseException as handling_error:
            self._raise_with_pending_publication(
                publication_id,
                "checkpoint restore main commit handling failed",
                primary,
                handling_error,
                fallback_publication=current,
                fallback_plan=plan,
            )

    def reconcile_online_registry(self, publication_id: str) -> list[dict[str, str]]:
        return self._reconcile_online(
            publication_id,
            CHECKPOINT_RESTORE_PHASES[:-1],
        )

    def reconcile_online_finalizers(self, publication_id: str) -> list[dict[str, str]]:
        return self._reconcile_online(
            publication_id,
            CHECKPOINT_RESTORE_PHASES[-1:],
        )

    def finish_online(self, publication_id: str) -> None:
        try:
            self._finish(publication_id, recovery_lease_id=None)
            if not self._terminal_commit_confirmed(publication_id):
                raise ValidationError(
                    "checkpoint restore terminal commit was not durably confirmed"
                )
        except BaseException as exc:
            try:
                terminal_commit_confirmed = self._terminal_commit_confirmed(
                    publication_id
                )
            except BaseException as confirmation_error:
                self._fence_preserving(publication_id, exc)
                self._raise_with_pending_publication(
                    publication_id,
                    "checkpoint restore terminal commit confirmation failed",
                    exc,
                    confirmation_error,
                )
            if terminal_commit_confirmed:
                raise
            self._fence_preserving(publication_id, exc)
            if not isinstance(exc, Exception):
                raise
            raise self._pending_signal(publication_id) from exc

    def recover_incomplete(self) -> list[str]:
        self._require_recovery_lease()
        with self._recovery_lock:
            self._recover_preparing_payload_delivery_attempts()
            recovered: list[str] = []
            for state in (
                "planning",
                "applying",
                "reconciliation_pending",
                "failed",
                "manual",
            ):
                for operation_reconciled in (False, True):
                    self._recover_publication_state(
                        state,
                        operation_reconciled=operation_reconciled,
                        recovered=recovered,
                    )
            for publication_id in self._recover_pending_payload_deliveries():
                if (
                    publication_id not in recovered
                    and len(recovered) < self._reconciliation_page_size
                ):
                    recovered.append(publication_id)
            for publication_id in self.reconcile_terminal_publications(
                recover_payload_delivery=True,
            ):
                if (
                    publication_id not in recovered
                    and len(recovered) < self._reconciliation_page_size
                ):
                    recovered.append(publication_id)
            return recovered

    def reconcile_terminal_publications(
        self,
        *,
        recover_payload_delivery: bool = False,
    ) -> list[str]:
        """Converge operations for committed checkpoint restore receipts.

        Failed and manual restore publications remain forward-recovery inputs
        and are handled above before this pass.  A committed receipt, however,
        can have its reconciliation marker dirtied by a supported operation
        update. Repair those exact rows before generic stale-operation
        recovery can interrupt the authoritative restore operation.
        """

        reconciled: list[str] = []
        after: RuntimePublicationCursor | None = None
        while True:
            page = self._store.query_runtime_publication_operation_reconciliation(
                kind=CHECKPOINT_RESTORE_PUBLICATION_KIND,
                state="committed",
                after=after,
                limit=self._reconciliation_page_size,
            )
            previous = after
            for publication in page.records:
                cursor = RuntimePublicationCursor(
                    publication["created_at"],
                    publication["publication_id"],
                )
                if (
                    publication["kind"] != CHECKPOINT_RESTORE_PUBLICATION_KIND
                    or publication["state"] != "committed"
                    or publication["operation_reconciled"]
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "runtime publication repository returned an invalid "
                        "checkpoint operation reconciliation page"
                    )
                publication_id = self._reconcile_terminal_publication(
                    publication,
                    recover_payload_delivery=recover_payload_delivery,
                )
                if len(reconciled) < self._reconciliation_page_size:
                    reconciled.append(publication_id)
                previous = cursor
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "runtime publication repository returned an invalid "
                    "checkpoint operation reconciliation cursor"
                )
            after = page.next_cursor
        return reconciled

    def _reconcile_terminal_publication(
        self,
        publication: Mapping[str, Any],
        *,
        recover_payload_delivery: bool,
    ) -> str:
        publication_id = str(publication["publication_id"])
        with self._store.transaction():
            current, plan, _checkpoint, _snapshot = self._load_publication(
                publication_id
            )
            if (
                current["state"] != "committed"
                or current["phase"] != "reconciled"
                or current["phase"] != publication["phase"]
                or current["operation_reconciled"]
            ):
                raise ValidationError(
                    "checkpoint restore publication changed during operation "
                    f"reconciliation: {publication_id}"
                )
            self._require_completion_receipts(current, plan)
            delivery_state = self._payload_delivery_state(current)
            if delivery_state in {"pending", "confirmed"}:
                self._recover_terminal_payload_delivery(
                    current,
                    delivery_state=delivery_state,
                    recover_payload_delivery=recover_payload_delivery,
                )
            self._reconcile_operation(current, OperationOutcome.SUCCEEDED)
        return publication_id

    def _recover_terminal_payload_delivery(
        self,
        publication: Mapping[str, Any],
        *,
        delivery_state: str,
        recover_payload_delivery: bool,
    ) -> None:
        if not recover_payload_delivery:
            raise ValidationError(
                "checkpoint restore payload delivery requires startup recovery"
            )
        if (
            delivery_state != "pending"
            or publication["payload_delivery_attempt_id"] is not None
            or publication["payload_delivery_started_at"] is not None
            or publication["owner_instance_id"] != self._owner_instance_id
        ):
            raise ValidationError(
                "checkpoint restore payload delivery was not durably hydrated "
                "and claimed before terminal reconciliation"
            )

    @staticmethod
    def _payload_delivery_state(publication: Mapping[str, Any]) -> str | None:
        delivery = publication["receipt"].get("payload_delivery")
        if delivery is None:
            if publication.get("payload_delivery_state") is not None:
                raise ValidationError(
                    "checkpoint restore payload delivery projection is invalid"
                )
            return None
        if (
            not isinstance(delivery, dict)
            or set(delivery) != {"state"}
            or delivery.get("state") not in {"pending", "confirmed", "completed"}
        ):
            raise ValidationError(
                "checkpoint restore payload delivery receipt is invalid"
            )
        selected = str(delivery["state"])
        if publication.get("payload_delivery_state") != selected:
            raise ValidationError(
                "checkpoint restore payload delivery receipt and projection differ"
            )
        return selected

    def begin_payload_delivery(self) -> CheckpointPayloadDeliveryAttempt | None:
        """Persist one scalar attempt before paging any pending deliveries."""

        attempt = CheckpointPayloadDeliveryAttempt(
            started_at=utc_now(),
            attempt_id=new_id("checkpoint_payload_delivery"),
            owner_instance_id=self._owner_instance_id,
        )
        with self._store.transaction():
            page = self._store.query_checkpoint_restore_payload_deliveries(
                delivery_state="pending",
                attempt_id=None,
                after=None,
                limit=self._reconciliation_page_size,
            )
            self._validate_payload_delivery_page(
                page,
                delivery_state="pending",
                attempt=None,
                after=None,
            )
            if not page.records:
                return None
            if not self._writer.begin_checkpoint_payload_delivery_attempt(attempt):
                raise ValidationError(
                    "checkpoint payload delivery attempt could not be persisted"
                )
        return attempt

    def prepare_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        """Page pending deliveries into one durable startup attempt."""

        self._require_current_payload_delivery_attempt(attempt)
        self._transition_payload_delivery_pages(
            expected_delivery_state="pending",
            delivery_state="confirmed",
            expected_attempt=None,
            delivery_attempt=attempt,
            owner_instance_id=self._owner_instance_id,
            expected_owner_instance_id=self._owner_instance_id,
        )

    def complete_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        """Page every confirmed row to completed while retaining its attempt."""

        self._require_current_payload_delivery_attempt(attempt)
        self._transition_payload_delivery_pages(
            expected_delivery_state="confirmed",
            delivery_state="completed",
            expected_attempt=attempt,
            delivery_attempt=attempt,
            owner_instance_id=attempt.owner_instance_id,
            expected_owner_instance_id=attempt.owner_instance_id,
        )

    def ack_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        """CAS the attempt ack inside the caller's OPEN-commit transaction."""

        self._require_current_payload_delivery_attempt(attempt)
        if not self._writer.ack_checkpoint_payload_delivery_attempt(attempt):
            raise ValidationError(
                "checkpoint payload delivery attempt acknowledgement failed"
            )

    def payload_delivery_attempt_state(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> CheckpointPayloadDeliveryAttemptState | None:
        """Confirm the exact durable control state after an ambiguous commit."""

        self._require_current_payload_delivery_attempt(attempt)
        return self._store.get_checkpoint_payload_delivery_attempt_state(attempt)

    def reopen_payload_delivery(
        self,
        attempt: CheckpointPayloadDeliveryAttempt | None,
    ) -> None:
        """Reopen one attempt, or every durable preparing attempt after restart."""

        if attempt is None:
            self._recover_preparing_payload_delivery_attempts()
            return
        if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
            raise ValidationError("checkpoint payload delivery attempt is invalid")
        for delivery_state in ("confirmed", "completed"):
            self._transition_payload_delivery_pages(
                expected_delivery_state=delivery_state,
                delivery_state="pending",
                expected_attempt=attempt,
                delivery_attempt=None,
                owner_instance_id=attempt.owner_instance_id,
                expected_owner_instance_id=attempt.owner_instance_id,
            )
        with self._store.transaction():
            if not self._writer.abort_checkpoint_payload_delivery_attempt(attempt):
                raise ValidationError(
                    "checkpoint payload delivery attempt compensation failed"
                )

    def _recover_preparing_payload_delivery_attempts(self) -> None:
        after: CheckpointPayloadDeliveryAttempt | None = None
        while True:
            with self._store.transaction():
                page = self._store.query_checkpoint_payload_delivery_attempts(
                    after=after,
                    limit=self._reconciliation_page_size,
                )
                previous = self._validate_payload_delivery_attempt_page(page, after=after)
            for attempt in page.records:
                self.reopen_payload_delivery(attempt)
            if page.next_cursor is None:
                break
            if previous is None:
                raise ValidationError(
                    "checkpoint payload delivery attempt page lost its cursor"
                )
            after = page.next_cursor

    def _recover_pending_payload_deliveries(self) -> list[str]:
        """Hydrate every pending row while retaining only a bounded sample."""

        recovered: list[str] = []
        after: RuntimePublicationCursor | None = None
        while True:
            with self._store.transaction():
                page = self._store.query_checkpoint_restore_payload_deliveries(
                    delivery_state="pending",
                    attempt_id=None,
                    after=after,
                    limit=self._reconciliation_page_size,
                )
                previous = self._validate_payload_delivery_page(
                    page,
                    delivery_state="pending",
                    attempt=None,
                    after=after,
                )
                for publication in page.records:
                    publication_id = str(publication["publication_id"])
                    current, plan, _checkpoint, snapshot = self._load_publication(
                        publication_id
                    )
                    if current != publication:
                        raise ValidationError(
                            "checkpoint payload delivery changed before hydration: "
                            f"{publication_id}"
                        )
                    self._require_completion_receipts(current, plan)
                    requires_hydration = (
                        publication["owner_instance_id"]
                        != self._owner_instance_id
                    )
                    if not self._writer.transition_payload_delivery(
                        publication_id,
                        expected_delivery_state="pending",
                        delivery_state="pending",
                        expected_attempt=None,
                        delivery_attempt=None,
                        owner_instance_id=self._owner_instance_id,
                        recovery_lease_id=None,
                    ):
                        raise ValidationError(
                            "checkpoint payload delivery ownership claim failed: "
                            f"{publication_id}"
                        )
                    if requires_hydration:
                        self._restore_object_payloads(snapshot)
                    if len(recovered) < self._reconciliation_page_size:
                        recovered.append(publication_id)
            if page.next_cursor is None:
                break
            if previous is None:
                raise ValidationError(
                    "checkpoint payload delivery page lost its cursor"
                )
            after = page.next_cursor
        return recovered

    def _transition_payload_delivery_pages(
        self,
        *,
        expected_delivery_state: PayloadDeliveryState,
        delivery_state: PayloadDeliveryState,
        expected_attempt: CheckpointPayloadDeliveryAttempt | None,
        delivery_attempt: CheckpointPayloadDeliveryAttempt | None,
        owner_instance_id: str,
        expected_owner_instance_id: str | None,
    ) -> None:
        after: RuntimePublicationCursor | None = None
        while True:
            with self._store.transaction():
                page = self._store.query_checkpoint_restore_payload_deliveries(
                    delivery_state=expected_delivery_state,
                    attempt_id=(
                        expected_attempt.attempt_id
                        if expected_attempt is not None
                        else None
                    ),
                    after=after,
                    limit=self._reconciliation_page_size,
                )
                previous = self._validate_payload_delivery_page(
                    page,
                    delivery_state=expected_delivery_state,
                    attempt=expected_attempt,
                    after=after,
                    expected_owner_instance_id=expected_owner_instance_id,
                )
                for publication in page.records:
                    publication_id = str(publication["publication_id"])
                    current, plan, _checkpoint, _snapshot = self._load_publication(
                        publication_id
                    )
                    if current != publication:
                        raise ValidationError(
                            "checkpoint payload delivery changed before transition: "
                            f"{publication_id}"
                        )
                    self._require_completion_receipts(current, plan)
                    if not self._writer.transition_payload_delivery(
                        publication_id,
                        expected_delivery_state=expected_delivery_state,
                        delivery_state=delivery_state,
                        expected_attempt=expected_attempt,
                        delivery_attempt=delivery_attempt,
                        owner_instance_id=owner_instance_id,
                        recovery_lease_id=None,
                    ):
                        raise ValidationError(
                            "checkpoint payload delivery transition failed: "
                            f"{publication_id}/{expected_delivery_state}->{delivery_state}"
                        )
            if page.next_cursor is None:
                break
            if previous is None:
                raise ValidationError(
                    "checkpoint payload delivery page lost its cursor"
                )
            after = page.next_cursor

    def _validate_payload_delivery_attempt_page(
        self,
        page: CheckpointPayloadDeliveryAttemptPage,
        *,
        after: CheckpointPayloadDeliveryAttempt | None,
    ) -> CheckpointPayloadDeliveryAttempt | None:
        if len(page.records) > self._reconciliation_page_size:
            raise ValidationError(
                "checkpoint payload delivery attempt page exceeded its limit"
            )
        previous = after
        previous_key = self._payload_delivery_attempt_key(after)
        for attempt in page.records:
            if not isinstance(attempt, CheckpointPayloadDeliveryAttempt):
                raise ValidationError(
                    "checkpoint payload delivery attempt page is invalid"
                )
            key = self._payload_delivery_attempt_key(attempt)
            if previous_key is not None and key <= previous_key:
                raise ValidationError(
                    "checkpoint payload delivery attempt page is not monotonic"
                )
            previous = attempt
            previous_key = key
        if page.next_cursor is not None and (
            previous is None or page.next_cursor != previous
        ):
            raise ValidationError(
                "checkpoint payload delivery attempt cursor is invalid"
            )
        return previous

    def _validate_payload_delivery_page(
        self,
        page: RuntimePublicationPage,
        *,
        delivery_state: PayloadDeliveryState,
        attempt: CheckpointPayloadDeliveryAttempt | None,
        after: RuntimePublicationCursor | None,
        expected_owner_instance_id: str | None = None,
    ) -> RuntimePublicationCursor | None:
        if len(page.records) > self._reconciliation_page_size:
            raise ValidationError(
                "checkpoint payload delivery page exceeded its limit"
            )
        previous = after
        for publication in page.records:
            cursor = RuntimePublicationCursor(
                publication["created_at"],
                publication["publication_id"],
            )
            attempt_id = attempt.attempt_id if attempt is not None else None
            started_at = attempt.started_at if attempt is not None else None
            if (
                publication["kind"] != CHECKPOINT_RESTORE_PUBLICATION_KIND
                or publication["state"] != "committed"
                or publication["phase"] != "reconciled"
                or self._payload_delivery_state(publication) != delivery_state
                or publication["payload_delivery_attempt_id"] != attempt_id
                or publication["payload_delivery_started_at"] != started_at
                or (
                    expected_owner_instance_id is not None
                    and publication["owner_instance_id"]
                    != expected_owner_instance_id
                )
                or (previous is not None and cursor <= previous)
            ):
                raise ValidationError(
                    "checkpoint payload delivery repository returned an invalid page"
                )
            previous = cursor
        if page.next_cursor is not None and (
            previous is None or page.next_cursor != previous
        ):
            raise ValidationError(
                "checkpoint payload delivery repository returned an invalid cursor"
            )
        return previous

    def _require_current_payload_delivery_attempt(
        self,
        attempt: CheckpointPayloadDeliveryAttempt,
    ) -> None:
        if (
            not isinstance(attempt, CheckpointPayloadDeliveryAttempt)
            or attempt.owner_instance_id != self._owner_instance_id
        ):
            raise ValidationError(
                "checkpoint payload delivery attempt does not belong to this Runtime"
            )

    @staticmethod
    def _payload_delivery_attempt_key(
        attempt: CheckpointPayloadDeliveryAttempt | None,
    ) -> tuple[str, str] | None:
        if attempt is None:
            return None
        return attempt.started_at, attempt.attempt_id

    def _recover_publication_state(
        self,
        state: str,
        *,
        operation_reconciled: bool,
        recovered: list[str],
    ) -> None:
        after: RuntimePublicationCursor | None = None
        while True:
            page = self._store.query_runtime_publication_recovery(
                kind=CHECKPOINT_RESTORE_PUBLICATION_KIND,
                state=state,
                operation_reconciled=operation_reconciled,
                after=after,
                limit=self._reconciliation_page_size,
            )
            previous = after
            for publication in page.records:
                cursor = RuntimePublicationCursor(
                    publication["created_at"],
                    publication["publication_id"],
                )
                if (
                    publication["kind"] != CHECKPOINT_RESTORE_PUBLICATION_KIND
                    or publication["state"] != state
                    or publication["operation_reconciled"] is not operation_reconciled
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "runtime publication repository returned an invalid checkpoint recovery page"
                    )
                recovered_id = self._recover_publication(publication)
                if (
                    recovered_id is not None
                    and len(recovered) < self._reconciliation_page_size
                ):
                    recovered.append(recovered_id)
                previous = cursor
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "runtime publication repository returned an invalid checkpoint recovery cursor"
                )
            after = page.next_cursor

    def _recover_publication(self, publication: Mapping[str, Any]) -> str | None:
        # Validate the immutable recovery program and its checkpoint digest
        # before taking a lease or changing any durable state. A corrupt plan
        # or snapshot must not gain an attempt or reach a replay/finalizer
        # callback.
        publication_id = str(publication["publication_id"])
        current, _plan, _checkpoint, _snapshot = self._load_publication(
            publication_id
        )
        if current != publication:
            raise ValidationError(
                "checkpoint restore publication changed before recovery claim: "
                f"{publication_id}"
            )
        if publication["state"] in {"planning", "applying"}:
            raise ValidationError(
                "checkpoint restore publication exposes an impossible pre-commit state: "
                f"{publication['publication_id']}"
            )
        if publication["state"] == "manual":
            self._fail_closed_manual(publication)
        claimed = self._writer.claim_runtime_publication_recovery(
            publication["publication_id"],
            claimant_instance_id=self._owner_instance_id,
            expected_owner_instance_id=publication["owner_instance_id"],
            expected_state=publication["state"],
            classification="reconcile_checkpoint_restore",
            max_attempts=self._recovery_max_attempts,
            allow_orphaned_claim_takeover=True,
            claimed_state="reconciliation_pending",
        )
        if claimed is None:
            current = self._store.get_runtime_publication(publication["publication_id"])
            if current is None or current["state"] == "committed":
                return None
            raise ValidationError(
                "cannot claim unresolved checkpoint restore publication: "
                f"{publication['publication_id']}"
            )
        if claimed["state"] == "manual":
            self._fail_closed_manual(claimed)
        claimed_plan = self._validated_plan(claimed)
        recovery_lease_id = self._recovery_lease_id(claimed)
        try:
            with self._registry_scope():
                self._run_phases(
                    claimed["publication_id"],
                    claimed_plan.phase_order[:-1],
                    recovery_lease_id=recovery_lease_id,
                )
            self._run_phases(
                claimed["publication_id"],
                claimed_plan.phase_order[-1:],
                recovery_lease_id=recovery_lease_id,
            )
        except Exception as exc:
            self._record_recovery_failure(
                claimed["publication_id"],
                recovery_lease_id=recovery_lease_id,
                error=exc,
                manual=isinstance(exc, DurableObjectFinalizerUnavailable),
            )
            raise ValidationError(
                "cannot reconcile checkpoint restore publication: "
                f"{claimed['publication_id']}"
            ) from exc
        try:
            self._finish(
                claimed["publication_id"],
                recovery_lease_id=recovery_lease_id,
            )
        except BaseException as exc:
            try:
                terminal_commit_confirmed = self._terminal_commit_confirmed(
                    str(claimed["publication_id"])
                )
            except BaseException as confirmation_error:
                self._raise_with_pending_publication(
                    str(claimed["publication_id"]),
                    "checkpoint restore recovery terminal confirmation failed",
                    exc,
                    confirmation_error,
                    fallback_publication=claimed,
                    fallback_plan=claimed_plan,
                )
            if terminal_commit_confirmed:
                if not isinstance(exc, Exception):
                    raise
                return str(claimed["publication_id"])
            if not isinstance(exc, Exception):
                raise
            try:
                self._record_recovery_failure(
                    claimed["publication_id"],
                    recovery_lease_id=recovery_lease_id,
                    error=exc,
                    manual=isinstance(exc, DurableObjectFinalizerUnavailable),
                )
            except BaseException as handling_error:
                self._raise_with_pending_publication(
                    str(claimed["publication_id"]),
                    "checkpoint restore recovery failure handling failed",
                    exc,
                    handling_error,
                    fallback_publication=claimed,
                    fallback_plan=claimed_plan,
                )
            raise ValidationError(
                "cannot reconcile checkpoint restore publication: "
                f"{claimed['publication_id']}"
            ) from exc
        return str(claimed["publication_id"])

    def _reconcile_online(
        self,
        publication_id: str,
        phases: Iterable[str],
    ) -> list[dict[str, str]]:
        publication: RuntimePublicationRecord | None = None
        plan: CheckpointRestorePlan | None = None
        checkpoint: Any | None = None
        try:
            publication, plan, checkpoint, snapshot = self._load_publication(
                publication_id
            )
            self._run_loaded_phases(
                publication_id,
                phases,
                publication=publication,
                plan=plan,
                snapshot=snapshot,
                recovery_lease_id=None,
            )
        except BaseException as exc:
            # The main snapshot rows are already committed.  Fail mutation
            # admission closed before any fallible diagnostic read, audit
            # write, or publication terminalization so a secondary failure
            # cannot leave the restored Runtime open.
            self._fence_preserving(publication_id, exc)
            if isinstance(exc, RuntimePublicationPending):
                # A storage adapter that reports a successful finalizer receipt
                # write without making it visible leaves the provider effect in
                # an explicitly retryable, unknown state.  Do not convert that
                # ambiguity into an ordinary failed phase or let later phases
                # run against a receipt that never became durable.
                raise
            try:
                with self._recovery_terminalization_scope(publication_id):
                    current = self._require_publication(publication_id)
                    current_plan = self._validated_plan(current)
                    if plan is None:
                        plan = current_plan
                    elif current_plan != plan:
                        raise ValidationError(
                            "checkpoint restore publication plan changed during "
                            f"online reconciliation: {publication_id}"
                        )
                    if checkpoint is None:
                        checkpoint, _snapshot, _typed = self._load_checkpoint(
                            plan.checkpoint_id
                        )
                    phase = self._next_incomplete_phase(current, plan) or "terminalization"
                    failure = self._record_failure(
                        plan.actor,
                        checkpoint,
                        phase,
                        exc,
                    )
                    self._mark_online_failure(
                        publication_id,
                        phase=phase,
                        error=exc,
                        manual=isinstance(exc, DurableObjectFinalizerUnavailable),
                    )
            except BaseException as handling_exc:
                if not isinstance(exc, Exception):
                    self._raise_with_pending_publication(
                        publication_id,
                        "checkpoint restore post-commit handling failed",
                        exc,
                        handling_exc,
                        fallback_publication=publication,
                        fallback_plan=plan,
                    )
                raise self._pending_signal(
                    publication_id,
                    fallback_publication=publication,
                    fallback_plan=plan,
                ) from handling_exc
            if not isinstance(exc, Exception):
                raise
            return [failure]
        return []

    def _run_phases(
        self,
        publication_id: str,
        phases: Iterable[str],
        *,
        recovery_lease_id: str | None,
    ) -> None:
        publication, plan, _checkpoint, snapshot = self._load_publication(
            publication_id
        )
        self._run_loaded_phases(
            publication_id,
            phases,
            publication=publication,
            plan=plan,
            snapshot=snapshot,
            recovery_lease_id=recovery_lease_id,
        )

    def _run_loaded_phases(
        self,
        publication_id: str,
        phases: Iterable[str],
        *,
        publication: RuntimePublicationRecord,
        plan: CheckpointRestorePlan,
        snapshot: dict[str, Any],
        recovery_lease_id: str | None,
    ) -> None:
        selected_phases = tuple(phases)
        completed = self._completed_phases(publication, plan)
        if (
            recovery_lease_id is not None
            and selected_phases == plan.phase_order[:-1]
        ):
            # The phase receipt proves that the durable Object row marker was
            # reconciled, but the payload cache is intentionally process-local.
            # Rehydrate it from the hash-anchored checkpoint on every startup
            # attempt, including attempts where this phase already completed.
            self._restore_object_payloads(snapshot)
        for phase in selected_phases:
            if phase in completed:
                continue
            expected_completed = [*completed, phase]
            if phase == "object_payload_reconciliation":
                # The payload-aware main transaction already published the
                # row marker and cache online. Startup recovery performs the
                # only replay so this receipt cannot overwrite a legitimate
                # post-commit Object mutation.
                pass
            elif phase == "image_reconciliation":
                self._restore_images(snapshot)
            elif phase == "jit_source_reconciliation":
                self._restore_jit_sources(snapshot)
            elif phase == "jit_pruning":
                self._prune_jit_tools(set(plan.stale_tool_ids), set(plan.scoped_pids))
            elif phase == "object_release_finalizers":
                self._run_finalizer_items(
                    publication_id,
                    plan,
                    recovery_lease_id=recovery_lease_id,
                )
            else:  # pragma: no cover - guarded by the constant phase order
                raise ValidationError(f"unknown checkpoint restore phase: {phase}")
            self._complete_phase(
                publication_id,
                phase,
                recovery_lease_id=recovery_lease_id,
            )
            publication = self._require_publication(publication_id)
            persisted_completed = self._completed_phases(publication, plan)
            if persisted_completed != expected_completed:
                raise self._pending_signal(publication_id)
            completed = persisted_completed

    def _run_finalizer_items(
        self,
        publication_id: str,
        plan: CheckpointRestorePlan,
        *,
        recovery_lease_id: str | None,
    ) -> None:
        publication = self._require_publication(publication_id)
        completed_work = self._completed_finalizer_work(publication, plan)
        operation_scope = (
            self._operations.attach(plan.operation_id)
            if recovery_lease_id is not None
            else nullcontext()
        )
        with operation_scope:
            for work_item in plan.finalizer_work_items:
                work_id = str(work_item["work_id"])
                if work_id in completed_work:
                    continue
                self._run_finalizer(work_item)
                current = self._require_publication(publication_id)
                if not self._writer.advance_runtime_publication(
                    publication_id,
                    state="reconciliation_pending",
                    phase="object_release_finalizers",
                    receipt={
                        "phase": "checkpoint_restore_finalizer_completed",
                        "work_id": work_id,
                    },
                    expected_states={"reconciliation_pending"},
                    expected_phase=str(current["phase"]),
                    recovery_lease_id=recovery_lease_id,
                ):
                    raise ValidationError(
                        "checkpoint restore finalizer receipt lost its lease: "
                        f"{work_id}"
                    )
                current = self._require_publication(publication_id)
                persisted_work = self._completed_finalizer_work(current, plan)
                if work_id not in persisted_work:
                    raise self._pending_signal(publication_id)
                completed_work = persisted_work

    def _complete_phase(
        self,
        publication_id: str,
        phase: str,
        *,
        recovery_lease_id: str | None,
    ) -> None:
        current = self._require_publication(publication_id)
        if not self._writer.advance_runtime_publication(
            publication_id,
            state="reconciliation_pending",
            phase=f"{phase}_completed",
            receipt={
                "phase": "checkpoint_restore_phase_completed",
                "name": phase,
            },
            expected_states={"reconciliation_pending"},
            expected_phase=str(current["phase"]),
            recovery_lease_id=recovery_lease_id,
        ):
            raise ValidationError(
                f"checkpoint restore phase receipt lost its lease: {publication_id}/{phase}"
            )

    def _finish(self, publication_id: str, *, recovery_lease_id: str | None) -> None:
        with self._store.transaction():
            current = self._require_publication(publication_id)
            plan = self._validated_plan(current)
            self._require_completion_receipts(current, plan)
            if not self._writer.advance_runtime_publication(
                publication_id,
                state="committed",
                phase="reconciled",
                receipt={"phase": "reconciled"},
                expected_states={"reconciliation_pending"},
                expected_phase=str(current["phase"]),
                recovery_lease_id=recovery_lease_id,
            ):
                raise ValidationError(
                    f"checkpoint restore publication terminalization failed: {publication_id}"
                )
            self._reconcile_operation(
                self._require_publication(publication_id),
                OperationOutcome.SUCCEEDED,
            )
            if recovery_lease_id is not None and not self._writer.transition_payload_delivery(
                publication_id,
                expected_delivery_state=None,
                delivery_state="pending",
                recovery_lease_id=recovery_lease_id,
            ):
                raise ValidationError(
                    "checkpoint restore payload delivery intent was not persisted: "
                    f"{publication_id}"
                )

    def _mark_online_failure(
        self,
        publication_id: str,
        *,
        phase: str,
        error: BaseException,
        manual: bool,
    ) -> None:
        with self._store.transaction():
            current = self._require_publication(publication_id)
            if not self._writer.advance_runtime_publication(
                publication_id,
                state="manual" if manual else "failed",
                phase=(
                    "durable_finalizer_handler_unavailable"
                    if manual
                    else f"{phase}_failed"
                ),
                error={
                    "code": "checkpoint_restore_reconciliation_failed",
                    "error_type": type(error).__name__,
                },
                expected_states={"reconciliation_pending"},
                expected_phase=str(current["phase"]),
            ):
                raise ValidationError(
                    f"checkpoint restore failure terminalization failed: {publication_id}"
                )
            self._reconcile_operation(
                self._require_publication(publication_id),
                OperationOutcome.UNKNOWN,
            )

    def _record_recovery_failure(
        self,
        publication_id: str,
        *,
        recovery_lease_id: str,
        error: Exception,
        manual: bool,
    ) -> None:
        with self._store.transaction():
            current = self._require_publication(publication_id)
            if not self._writer.advance_runtime_publication(
                publication_id,
                state="manual" if manual else "failed",
                phase=(
                    "durable_finalizer_handler_unavailable"
                    if manual
                    else "startup_reconciliation_failed"
                ),
                error={
                    "code": "checkpoint_restore_reconciliation_failed",
                    "error_type": type(error).__name__,
                },
                expected_states={"reconciliation_pending"},
                expected_phase=str(current["phase"]),
                recovery_lease_id=recovery_lease_id,
            ):
                raise ValidationError(
                    f"checkpoint restore recovery lease changed: {publication_id}"
                ) from error
            self._reconcile_operation(
                self._require_publication(publication_id),
                OperationOutcome.UNKNOWN,
            )

    def _load_publication(
        self,
        publication_id: str,
    ) -> tuple[
        RuntimePublicationRecord,
        CheckpointRestorePlan,
        Any,
        dict[str, Any],
    ]:
        publication = self._require_publication(publication_id)
        if publication["kind"] != CHECKPOINT_RESTORE_PUBLICATION_KIND:
            raise ValidationError(
                f"unexpected checkpoint restore publication kind: {publication_id}"
            )
        plan = self._validated_plan(publication)
        if publication["pid"] != plan.pid:
            raise ValidationError(
                f"checkpoint restore publication pid mismatch: {publication_id}"
            )
        checkpoint, snapshot, _typed = self._load_checkpoint(plan.checkpoint_id)
        if (
            str(checkpoint.pid) != plan.pid
            or int(checkpoint.snapshot_version) != plan.snapshot_version
            or self.snapshot_sha256(snapshot) != plan.snapshot_sha256
        ):
            raise ValidationError(
                f"checkpoint restore publication snapshot changed: {publication_id}"
            )
        return publication, plan, checkpoint, snapshot

    def _require_publication(
        self,
        publication_id: str,
    ) -> RuntimePublicationRecord:
        publication = self._store.get_runtime_publication(publication_id)
        if publication is None:
            raise ValidationError(
                f"checkpoint restore publication disappeared: {publication_id}"
            )
        return publication

    @staticmethod
    def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
        return hashlib.sha256(dumps(dict(snapshot)).encode("utf-8")).hexdigest()

    @staticmethod
    def _completed_phases(
        publication: Mapping[str, Any],
        plan: CheckpointRestorePlan,
    ) -> list[str]:
        markers = [
            str(item.get("name") or "")
            for item in publication["receipt"].get("phases", [])
            if isinstance(item, dict)
            and item.get("phase") == "checkpoint_restore_phase_completed"
        ]
        if markers != list(plan.phase_order[: len(markers)]):
            raise ValidationError(
                "checkpoint restore phase receipts are duplicated or out of order"
            )
        return markers

    @staticmethod
    def _completed_finalizer_work(
        publication: Mapping[str, Any],
        plan: CheckpointRestorePlan,
    ) -> set[str]:
        expected = {str(item["work_id"]) for item in plan.finalizer_work_items}
        markers = [
            str(item.get("work_id") or "")
            for item in publication["receipt"].get("phases", [])
            if isinstance(item, dict)
            and item.get("phase") == "checkpoint_restore_finalizer_completed"
        ]
        expected_prefix = [
            str(item["work_id"])
            for item in plan.finalizer_work_items[: len(markers)]
        ]
        if (
            markers != expected_prefix
            or any(not marker or marker not in expected for marker in markers)
        ):
            raise ValidationError("checkpoint restore finalizer receipts are invalid")
        return set(markers)

    def _next_incomplete_phase(
        self,
        publication: Mapping[str, Any],
        plan: CheckpointRestorePlan,
    ) -> str | None:
        completed = self._completed_phases(publication, plan)
        return (
            plan.phase_order[len(completed)]
            if len(completed) < len(plan.phase_order)
            else None
        )

    @staticmethod
    def _require_main_commit_identity(
        publication: Mapping[str, Any],
        plan: CheckpointRestorePlan,
    ) -> None:
        if (
            publication["kind"] != CHECKPOINT_RESTORE_PUBLICATION_KIND
            or publication["pid"] != plan.pid
        ):
            raise ValidationError("checkpoint restore publication identity is invalid")
        phases = publication["receipt"].get("phases")
        if publication["state"] == "planning":
            if publication["phase"] != "planned" or phases != []:
                raise ValidationError(
                    "checkpoint restore planning transcript is invalid"
                )
            return
        if not isinstance(phases, list) or not phases:
            raise ValidationError("checkpoint restore main commit receipt is missing")
        if phases[0] != {"phase": "main_state_committed"}:
            raise ValidationError("checkpoint restore main commit receipt is invalid")

    def _require_completion_receipts(
        self,
        publication: Mapping[str, Any],
        plan: CheckpointRestorePlan,
    ) -> None:
        expected: list[dict[str, Any]] = [{"phase": "main_state_committed"}]
        expected.extend(
            {
                "phase": "checkpoint_restore_phase_completed",
                "name": phase,
            }
            for phase in plan.phase_order[:-1]
        )
        expected.extend(
            {
                "phase": "checkpoint_restore_finalizer_completed",
                "work_id": str(item["work_id"]),
            }
            for item in plan.finalizer_work_items
        )
        expected.append(
            {
                "phase": "checkpoint_restore_phase_completed",
                "name": plan.phase_order[-1],
            }
        )
        if publication["state"] == "committed":
            expected.append({"phase": "reconciled"})
        phases = publication["receipt"].get("phases")
        if not isinstance(phases, list):
            raise ValidationError("checkpoint restore receipt transcript is invalid")
        causal = [
            item
            for item in phases
            if not isinstance(item, dict) or item.get("phase") != "recovery_claimed"
        ]
        if causal != expected:
            raise ValidationError(
                "checkpoint restore completion transcript is incomplete or invalid: "
                f"{publication['publication_id']}"
            )

    def _terminal_commit_confirmed(
        self,
        publication_id: str,
        *,
        publication: Mapping[str, Any] | None = None,
    ) -> bool:
        current = (
            dict(publication)
            if publication is not None
            else self._require_publication(publication_id)
        )
        plan = self._validated_plan(current)
        self._require_main_commit_identity(current, plan)
        if current["state"] != "committed":
            return False
        self._payload_delivery_state(current)
        if current["phase"] != "reconciled" or (
            current["operation_reconciled"] is not True
        ):
            raise ValidationError(
                "checkpoint restore terminal publication is not fully reconciled"
            )
        self._require_completion_receipts(current, plan)
        operation_ids = self._operations.runtime_publication_binding_operation_ids(
            publication_id
        )
        operation = self._operations.get_operation(plan.operation_id)
        if operation_ids != [plan.operation_id] or operation is None:
            raise ValidationError(
                "checkpoint restore terminal operation binding is invalid"
            )
        metadata = operation.metadata
        exact_terminal = (
            operation.kind == OperationKind.RUNTIME
            and operation.name == "checkpoint.restore"
            and operation.actor == plan.actor
            and operation.pid == plan.actor
            and operation.state == OperationState.TERMINAL
            and operation.outcome == OperationOutcome.SUCCEEDED
            and metadata.get("runtime_publication_id") == publication_id
            and metadata.get("runtime_publication_kind")
            == CHECKPOINT_RESTORE_PUBLICATION_KIND
            and metadata.get("runtime_publication_bound") is True
            and metadata.get("runtime_publication_binding_version") == 1
            and metadata.get("runtime_publication_state") == "committed"
            and metadata.get("runtime_publication_phase") == "reconciled"
            and metadata.get("runtime_publication_reconciled") is True
        )
        if not exact_terminal:
            raise ValidationError(
                "checkpoint restore terminal operation truth is invalid"
            )
        return True

    def _raise_with_pending_publication(
        self,
        publication_id: str,
        message: str,
        *errors: BaseException,
        fallback_publication: Mapping[str, Any] | None = None,
        fallback_plan: CheckpointRestorePlan | None = None,
    ) -> None:
        leaves = list(errors)
        try:
            pending = self._pending_signal(
                publication_id,
                fallback_publication=fallback_publication,
                fallback_plan=fallback_plan,
            )
        except BaseException as pending_error:
            leaves.append(pending_error)
        else:
            if pending.state in {
                "planning",
                "applying",
                "reconciliation_pending",
                "rollback_pending",
            }:
                leaves.append(pending)
        raise BaseExceptionGroup(message, leaves) from None

    def _reconcile_operation(
        self,
        publication: Mapping[str, Any],
        outcome: OperationOutcome,
    ) -> None:
        plan = self._validated_plan(publication)
        self._operations.reconcile_runtime_publication(
            plan.operation_id,
            outcome,
            publication_id=str(publication["publication_id"]),
            publication_kind=CHECKPOINT_RESTORE_PUBLICATION_KIND,
            publication_state=str(publication["state"]),
            publication_phase=str(publication["phase"]),
            expected_kind="runtime",
            expected_name="checkpoint.restore",
            expected_actor=plan.actor,
            expected_pid=plan.actor,
            _publication_reconciled_marker=(
                self._writer.mark_runtime_publication_operation_reconciled
            ),
        )

    def _fence(self, publication_id: str) -> None:
        if self._recovery_required is not None:
            self._recovery_required(publication_id=publication_id)

    def _fence_preserving(
        self,
        publication_id: str,
        primary: BaseException,
    ) -> None:
        try:
            self._fence(publication_id)
        except BaseException as fence_error:
            try:
                with self._recovery_terminalization_scope(publication_id):
                    pass
            except BaseException as confirmation_error:
                raise BaseExceptionGroup(
                    "checkpoint restore recovery fence failed",
                    [primary, fence_error, confirmation_error],
                ) from None
            self._raise_with_pending_publication(
                publication_id,
                "checkpoint restore recovery fence reported failure after closing",
                primary,
                fence_error,
            )

    def _pending_signal(
        self,
        publication_id: str,
        *,
        fallback_publication: Mapping[str, Any] | None = None,
        fallback_plan: CheckpointRestorePlan | None = None,
    ) -> RuntimePublicationPending:
        try:
            publication = self._require_publication(publication_id)
            plan = self._validated_plan(publication)
        except Exception:
            if fallback_publication is None or fallback_plan is None:
                raise
            publication = fallback_publication
            plan = fallback_plan
        return RuntimePublicationPending(
            publication_id=publication_id,
            operation_id=plan.operation_id,
            state=str(publication["state"]),
            phase=str(publication["phase"]),
        )

    @staticmethod
    def _plan_anchor(
        publication_id: str,
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan_version = plan.get("plan_version")
        if plan_version not in {1, CHECKPOINT_RESTORE_PLAN_VERSION}:
            raise ValidationError("checkpoint restore plan anchor version is invalid")
        return {
            "artifact_id": (
                f"{publication_id}:checkpoint_restore_plan:"
                f"v{plan_version}"
            ),
            "artifact_type": "checkpoint_restore_plan_anchor",
            "anchor_version": plan_version,
            "plan_sha256": hashlib.sha256(
                dumps(dict(plan)).encode("utf-8")
            ).hexdigest(),
        }

    def _validated_plan(
        self,
        publication: Mapping[str, Any],
    ) -> CheckpointRestorePlan:
        plan = CheckpointRestorePlan.from_mapping(publication["plan"])
        expected_anchor = self._plan_anchor(
            str(publication["publication_id"]),
            plan.to_mapping(),
        )
        artifacts = publication["receipt"].get("artifacts")
        if artifacts != [expected_anchor]:
            raise ValidationError(
                "checkpoint restore publication plan anchor is invalid: "
                f"{publication['publication_id']}"
            )
        return plan

    @staticmethod
    def _recovery_lease_id(publication: Mapping[str, Any]) -> str:
        recovery = publication["receipt"].get("recovery") or {}
        lease_id = str(recovery.get("lease_id") or "")
        if not lease_id:
            raise ValidationError(
                "checkpoint restore recovery claim has no lease: "
                f"{publication['publication_id']}"
            )
        return lease_id

    def _fail_closed_manual(self, publication: Mapping[str, Any]) -> None:
        with self._store.transaction():
            self._reconcile_operation(publication, OperationOutcome.UNKNOWN)
        raise ValidationError(
            "checkpoint restore publication requires manual recovery: "
            f"{publication['publication_id']}"
        )


__all__ = [
    "CHECKPOINT_RESTORE_PHASES",
    "CHECKPOINT_RESTORE_PLAN_ANCHOR_VERSION",
    "CHECKPOINT_RESTORE_PLAN_VERSION",
    "CHECKPOINT_RESTORE_PUBLICATION_KIND",
    "CHECKPOINT_RESTORE_V1_PHASES",
    "CheckpointRestorePlan",
    "CheckpointRestoreReconciler",
]
