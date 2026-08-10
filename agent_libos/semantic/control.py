from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import (
    SemanticControlStateV1,
    SemanticPolicyEpochV1,
    SemanticRuntimeMode,
    SemanticTripCode,
)
from agent_libos.semantic.enforcement import SemanticAuthorityControlView
from agent_libos.storage.semantic_v6 import (
    SemanticControlStateRecord,
    SemanticHealthEventRecord,
    SemanticPolicyEpochRecord,
    control_state_storage_record,
)
from agent_libos.utils.ids import new_id, utc_now


_AUTHORITY_MODES = frozenset(
    {SemanticRuntimeMode.ENFORCE_DENY, SemanticRuntimeMode.CANARY_AUTO}
)
_INACTIVE_MODES = frozenset(
    {SemanticRuntimeMode.OFF, SemanticRuntimeMode.SHADOW}
)
_POLICY_SCAN_PAGE_SIZE = 500
_POLICY_SCAN_MAX_RECORDS = 10_000
_ROLLOUT_REQUIRED_SAFE_REVIEWS = 1_000
_ROLLOUT_MINIMUM_AGE = timedelta(days=7)
_UNSAFE_REVIEW_UNSETTLED_EVENT = "semantic_unsafe_review_control_unsettled"
_ROLLOUT_SCOPE_KEYS = frozenset(
    {
        "schema_version",
        "tenant_bucket_sha256s",
        "auto_approval_rules",
        "hard_deny_rules",
        "allow_parameters",
    }
)
_ROLLOUT_RULE_KEYS = frozenset(
    {
        "rule_id_sha256",
        "action_id",
        "rights",
        "resource_kind",
        "match_sha256",
        "covering_prefix_sha256s",
    }
)
_ROLLOUT_ALLOW_PARAMETER_KEYS = frozenset(
    {
        "catalog_version",
        "classifier_profile_id_sha256",
        "classifier_profile_sha256",
        "classifier_model_sha256",
        "minimum_confidence_bps",
        "required_calibration_bucket",
        "capability_ttl_s",
        "per_rule_per_minute_limit",
        "per_rule_per_day_limit",
        "max_inflight",
    }
)
_ROLLOUT_ACTION_RIGHTS = {
    "filesystem.read": frozenset({"read"}),
    "git.read": frozenset({"read"}),
    "git.diff": frozenset({"diff"}),
}
_ROLLOUT_DENY_RIGHTS = frozenset(
    {"read", "write", "execute", "link", "diff", "materialize", "delete"}
)
_ROLLOUT_ACTION_ID = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z"
)


class SemanticControlConflict(ValidationError):
    """Static semantic policy cannot safely reconcile with durable state."""


class _SemanticControlCASRetry(RuntimeError):
    pass


class SemanticRuntimeControl:
    """Host-owned durable semantic mode and policy-epoch coordinator.

    Construction is side-effect free.  The composition root must call
    :meth:`admit` once during startup, before exposing a Runtime.  Thereafter,
    every authority check uses :meth:`authority_view`, which re-reads durable
    state and refuses stale static policy.

    ``repository`` is intentionally structural.  A
    ``SemanticAssessmentRepository`` is the production implementation, while
    narrow deterministic stores can be used in tests.
    """

    def __init__(
        self,
        repository: Any,
        *,
        mode: SemanticRuntimeMode | str,
        policy_epoch: SemanticPolicyEpochV1 | None,
        now: Callable[[], str] = utc_now,
        max_cas_attempts: int = 8,
    ) -> None:
        if repository is None:
            raise TypeError("semantic runtime control requires a repository")
        for operation in (
            "transaction",
            "append_semantic_policy_epoch",
            "get_semantic_policy_epoch",
            "get_semantic_control_state",
            "compare_and_set_semantic_control_state",
            "semantic_unsafe_review_count",
            "query_semantic_health_events",
        ):
            if not callable(getattr(repository, operation, None)):
                raise TypeError(
                    f"semantic runtime control repository lacks {operation}"
                )
        try:
            selected_mode = SemanticRuntimeMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported semantic runtime mode") from exc
        if policy_epoch is not None and not isinstance(
            policy_epoch, SemanticPolicyEpochV1
        ):
            raise TypeError(
                "semantic runtime control policy_epoch must be SemanticPolicyEpochV1"
            )
        if selected_mode in _AUTHORITY_MODES and policy_epoch is None:
            raise ValueError(
                f"semantic mode {selected_mode.value} requires a static policy epoch"
            )
        if not callable(now):
            raise TypeError("semantic runtime control clock must be callable")
        if (
            type(max_cas_attempts) is not int
            or max_cas_attempts < 1
            or max_cas_attempts > 32
        ):
            raise ValueError("semantic control max_cas_attempts must be in [1, 32]")
        self._repository = repository
        self._mode = selected_mode
        self._policy_epoch = policy_epoch
        self._now = now
        self._max_cas_attempts = max_cas_attempts
        self._unsafe_review_latched = False

    @property
    def configured_mode(self) -> SemanticRuntimeMode:
        return self._mode

    @property
    def policy_epoch(self) -> SemanticPolicyEpochV1 | None:
        return self._policy_epoch

    def admit(self) -> SemanticControlStateV1:
        """Reconcile static configuration with durable control state.

        Policy evidence and its active pointer are committed atomically.  CAS
        contention is retried with a fresh durable read.  A semantic conflict
        is never repaired or widened implicitly.
        """

        transition_kind: str | None = None
        last_cas_conflict = False
        try:
            for _attempt in range(self._max_cas_attempts):
                try:
                    with self._repository.transaction():
                        # Inactive configuration never stages immutable policy
                        # rows.  Persisting an unvalidated future generation
                        # while authority is off/shadow could poison the unique
                        # generation slot and make the later valid rotation
                        # impossible.
                        if (
                            self._policy_epoch is not None
                            and self._mode not in _INACTIVE_MODES
                        ):
                            self._repository.append_semantic_policy_epoch(
                                self._policy_epoch
                            )
                        expected = self._repository.get_semantic_control_state()
                        target, transition_kind = self._startup_target(expected)
                        target, transition_kind = self._unsafe_review_startup_target(
                            expected,
                            target,
                            transition_kind,
                        )
                        if target is None:
                            if expected is None:
                                raise SemanticControlConflict(
                                    "semantic control disappeared during startup"
                                )
                            return _public_control_state(expected)
                        target_record = control_state_storage_record(target)
                        if not self._repository.compare_and_set_semantic_control_state(
                            expected,
                            target_record,
                        ):
                            raise _SemanticControlCASRetry
                    if transition_kind is not None:
                        self._record_transition_health(
                            transition_kind,
                            target,
                        )
                    return target
                except _SemanticControlCASRetry:
                    last_cas_conflict = True
                    continue
        except Exception as exc:
            if not isinstance(exc, _SemanticControlCASRetry):
                self._record_conflict_health(exc)
            raise
        self._record_conflict_health(
            SemanticControlConflict(
                "semantic control startup CAS did not converge"
            )
        )
        qualifier = " after contention" if last_cas_conflict else ""
        raise SemanticControlConflict(
            f"semantic control startup CAS did not converge{qualifier}"
        )

    def current(self) -> SemanticControlStateV1:
        """Return the current durable control state, never a cached snapshot."""

        record = self._repository.get_semantic_control_state()
        if record is None:
            raise SemanticControlConflict(
                "semantic control has not been admitted at startup"
            )
        return _public_control_state(record)

    def control_status(self) -> SemanticControlStateV1:
        """Compatibility alias for read-only API/status composition."""

        return self.current()

    def status(self) -> SemanticControlStateV1:
        """Compatibility alias for callers using a generic status port."""

        return self.current()

    def disable(
        self,
        mode: SemanticRuntimeMode | str = SemanticRuntimeMode.OFF,
    ) -> SemanticControlStateV1:
        """Immediately revoke machine authority through a durable CAS.

        Only the authority-narrowing ``off`` and ``shadow`` targets are
        accepted.  The active pointer and trip marker are cleared while the
        generation high-water mark is retained, so neither a grant nor its old
        epoch can be revived by an in-memory mode change or restart.
        """

        try:
            selected_mode = SemanticRuntimeMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported semantic disable mode") from exc
        if selected_mode not in _INACTIVE_MODES:
            raise ValueError("semantic disable mode must be off or shadow")
        for _attempt in range(self._max_cas_attempts):
            try:
                with self._repository.transaction():
                    expected = self._repository.get_semantic_control_state()
                    target, transition_kind = self._inactive_target(
                        expected,
                        mode=selected_mode,
                    )
                    if target is None:
                        if expected is None:
                            raise SemanticControlConflict(
                                "semantic control disappeared during disable"
                            )
                        current = _public_control_state(expected)
                    else:
                        if not self._repository.compare_and_set_semantic_control_state(
                            expected,
                            control_state_storage_record(target),
                        ):
                            raise _SemanticControlCASRetry
                        current = target
                # Prevent a later accidental admit() call on this same live
                # coordinator from re-enabling its original static mode.
                self._mode = selected_mode
                if target is not None and transition_kind is not None:
                    self._record_transition_health(transition_kind, target)
                return current
            except _SemanticControlCASRetry:
                continue
        self._record_health(
            event_kind="semantic_control_disable_conflict",
            severity="critical",
            epoch_id=None,
            tenant_bucket_sha256=None,
            evidence_sha256=_evidence_sha256(
                {
                    "event": "disable_conflict",
                    "target_mode": selected_mode.value,
                }
            ),
        )
        raise SemanticControlConflict("semantic disable CAS did not converge")

    def authority_view(self) -> SemanticAuthorityControlView:
        """Resolve an exact live control/epoch pair for authority validation."""

        if self._unsafe_review_latched:
            raise SemanticControlConflict(
                "semantic machine authority is locally fenced by unsafe review"
            )
        control = self.current()
        if control.mode not in _AUTHORITY_MODES:
            raise SemanticControlConflict("semantic machine authority is disabled")
        epoch = self._policy_epoch
        if epoch is None:
            raise SemanticControlConflict(
                "semantic active control has no static policy epoch"
            )
        policy_sha256 = epoch.canonical_sha256()
        if (
            control.active_epoch_id != epoch.epoch_id
            or control.active_policy_sha256 != policy_sha256
            or control.generation != epoch.generation
        ):
            raise SemanticControlConflict(
                "semantic static policy differs from durable active control"
            )
        stored = self._repository.get_semantic_policy_epoch(epoch.epoch_id)
        if (
            stored is None
            or stored.generation != epoch.generation
            or stored.catalog_version != epoch.catalog_version
            or stored.policy_sha256 != policy_sha256
        ):
            raise SemanticControlConflict(
                "semantic durable policy evidence is absent or inconsistent"
            )
        return SemanticAuthorityControlView(control=control, epoch=epoch)

    def latch_unsafe_review(self) -> None:
        """Immediately fence this process after a non-linearized unsafe import."""

        self._unsafe_review_latched = True

    def trip(
        self,
        reason: SemanticTripCode | str,
        *,
        evidence_sha256: str | None = None,
        tenant_bucket_sha256: str | None = None,
    ) -> SemanticControlStateV1:
        """Durably trip the current epoch without ever reviving or replacing it."""

        try:
            selected_reason = SemanticTripCode(reason)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported semantic safety trip reason") from exc
        if evidence_sha256 is not None:
            _require_sha256(evidence_sha256, "semantic trip evidence_sha256")
        if tenant_bucket_sha256 is not None:
            _require_sha256(
                tenant_bucket_sha256,
                "semantic trip tenant_bucket_sha256",
            )
        for _attempt in range(self._max_cas_attempts):
            try:
                with self._repository.transaction():
                    expected = self._repository.get_semantic_control_state()
                    if expected is None:
                        raise SemanticControlConflict(
                            "semantic control has not been admitted at startup"
                        )
                    current = _public_control_state(expected)
                    if (
                        current.mode not in _AUTHORITY_MODES
                        or current.active_epoch_id is None
                    ):
                        raise SemanticControlConflict(
                            "semantic safety trip requires an active policy epoch"
                        )
                    if current.tripped:
                        # The first durable trip remains canonical.  A later
                        # symptom may be recorded separately by its caller but
                        # cannot rewrite the reason or revive the epoch.
                        return current
                    target = SemanticControlStateV1(
                        revision=current.revision + 1,
                        generation=current.generation,
                        mode=current.mode,
                        active_epoch_id=current.active_epoch_id,
                        active_policy_sha256=current.active_policy_sha256,
                        tripped=True,
                        trip_code=selected_reason,
                        updated_at=self._now(),
                    )
                    if not self._repository.compare_and_set_semantic_control_state(
                        expected,
                        control_state_storage_record(target),
                    ):
                        raise _SemanticControlCASRetry
                self._record_health(
                    event_kind=f"semantic_safety_trip:{selected_reason.value}",
                    severity="critical",
                    epoch_id=target.active_epoch_id,
                    tenant_bucket_sha256=tenant_bucket_sha256,
                    evidence_sha256=evidence_sha256
                    or _evidence_sha256(
                        {
                            "event": "semantic_safety_trip",
                            "reason": selected_reason.value,
                            "epoch_id": target.active_epoch_id,
                            "policy_sha256": target.active_policy_sha256,
                            "generation": target.generation,
                            "revision": target.revision,
                        }
                    ),
                )
                return target
            except _SemanticControlCASRetry:
                continue
        raise SemanticControlConflict("semantic safety-trip CAS did not converge")

    def _startup_target(
        self,
        expected: SemanticControlStateRecord | None,
    ) -> tuple[SemanticControlStateV1 | None, str | None]:
        if self._mode in _INACTIVE_MODES:
            return self._inactive_target(expected)
        epoch = self._policy_epoch
        if epoch is None:  # Defensive; rejected during construction.
            raise SemanticControlConflict(
                "active semantic mode has no static policy epoch"
            )
        policy_sha256 = epoch.canonical_sha256()
        if expected is None:
            return self._first_active_target(epoch, policy_sha256)

        current = _public_control_state(expected)
        if (
            current.mode is self._mode
            and current.active_epoch_id == epoch.epoch_id
            and current.active_policy_sha256 == policy_sha256
            and current.generation == epoch.generation
        ):
            # In particular, preserve a tripped epoch.  Restarting with the
            # same static configuration can never clear its durable trip.
            self._verify_current_epoch_record(current)
            return None, None

        if (
            current.active_epoch_id == epoch.epoch_id
            or current.active_policy_sha256 == policy_sha256
            or current.generation == epoch.generation
        ):
            raise SemanticControlConflict(
                "semantic active mode or epoch identity conflicts with durable state"
            )
        if (
            current.generation == 0
            and current.active_epoch_id is None
            and current.active_policy_sha256 is None
        ):
            return self._first_active_target(
                epoch,
                policy_sha256,
                revision=current.revision + 1,
            )
        return self._rotation_target(current, epoch, policy_sha256)

    def _unsafe_review_startup_target(
        self,
        expected: SemanticControlStateRecord | None,
        target: SemanticControlStateV1 | None,
        transition_kind: str | None,
    ) -> tuple[SemanticControlStateV1 | None, str | None]:
        """Trip an active startup target when durable unsafe evidence exists.

        Direct labels are scoped to the epoch that issued the reviewed
        settlement.  A critical unsettled-health event additionally covers
        the degraded import case where an old-epoch label arrived while this
        epoch was active but the atomic control trip could not converge.
        """

        selected = (
            target
            if target is not None
            else (_public_control_state(expected) if expected is not None else None)
        )
        if (
            selected is None
            or selected.mode not in _AUTHORITY_MODES
            or selected.active_epoch_id is None
            or selected.tripped
        ):
            return target, transition_kind
        if not self._has_unsafe_review_startup_evidence(
            selected.active_epoch_id
        ):
            return target, transition_kind
        revision = (
            selected.revision
            if target is not None
            else selected.revision + 1
        )
        return (
            SemanticControlStateV1(
                revision=revision,
                generation=selected.generation,
                mode=selected.mode,
                active_epoch_id=selected.active_epoch_id,
                active_policy_sha256=selected.active_policy_sha256,
                tripped=True,
                trip_code=SemanticTripCode.UNSAFE_REVIEW,
                updated_at=self._now(),
            ),
            "startup_unsafe_review_trip",
        )

    def _has_unsafe_review_startup_evidence(self, epoch_id: str) -> bool:
        if self._unsafe_review_count(epoch_id=epoch_id) > 0:
            return True
        page = self._repository.query_semantic_health_events(
            limit=1,
            epoch_id=epoch_id,
            event_kind=_UNSAFE_REVIEW_UNSETTLED_EVENT,
        )
        records = getattr(page, "records", None)
        if not isinstance(records, tuple):
            raise SemanticControlConflict(
                "semantic unsafe-review health evidence is malformed"
            )
        return bool(records)

    def _unsafe_review_count(self, *, epoch_id: str | None = None) -> int:
        reader = getattr(
            self._repository,
            "semantic_unsafe_review_count",
            None,
        )
        if not callable(reader):
            raise SemanticControlConflict(
                "semantic unsafe-review evidence is unavailable"
            )
        try:
            count = reader(epoch_id=epoch_id)
        except Exception as exc:
            raise SemanticControlConflict(
                "semantic unsafe-review evidence cannot be read"
            ) from exc
        if type(count) is not int or count < 0:
            raise SemanticControlConflict(
                "semantic unsafe-review evidence count is malformed"
            )
        return count

    def _first_active_target(
        self,
        epoch: SemanticPolicyEpochV1,
        policy_sha256: str,
        *,
        revision: int = 0,
    ) -> tuple[SemanticControlStateV1, str]:
        if epoch.generation != 1 or epoch.expected_previous_sha256 is not None:
            raise SemanticControlConflict(
                "the first active semantic epoch must be generation 1 with no predecessor"
            )
        return (
            SemanticControlStateV1(
                revision=revision,
                generation=epoch.generation,
                mode=self._mode,
                active_epoch_id=epoch.epoch_id,
                active_policy_sha256=policy_sha256,
                tripped=False,
                trip_code=None,
                updated_at=self._now(),
            ),
            "activated",
        )

    def _rotation_target(
        self,
        current: SemanticControlStateV1,
        epoch: SemanticPolicyEpochV1,
        policy_sha256: str,
    ) -> tuple[SemanticControlStateV1, str]:
        if epoch.generation != current.generation + 1:
            raise SemanticControlConflict(
                "semantic policy rotation must advance generation exactly once"
            )
        previous_record = self._previous_policy_record(current)
        previous_sha256 = (
            previous_record.policy_sha256
            if previous_record is not None
            else None
        )
        if (
            previous_sha256 is None
            or epoch.expected_previous_sha256 != previous_sha256
        ):
            raise SemanticControlConflict(
                "semantic policy expected_previous_sha256 does not match durable history"
            )
        self._validate_rollout_expansion(
            epoch,
            previous_record=previous_record,
        )
        return (
            SemanticControlStateV1(
                revision=current.revision + 1,
                generation=epoch.generation,
                mode=self._mode,
                active_epoch_id=epoch.epoch_id,
                active_policy_sha256=policy_sha256,
                tripped=False,
                trip_code=None,
                updated_at=self._now(),
            ),
            "rotated",
        )

    def _validate_rollout_expansion(
        self,
        epoch: SemanticPolicyEpochV1,
        *,
        previous_record: Any,
    ) -> None:
        """Admit rollout expansion only from complete durable canary evidence.

        Policy digests and cardinalities cannot prove that a cohort or rule was
        narrowed: a same-sized tenant swap, weakened hard deny, relaxed budget,
        or classifier artifact replacement can all expand effective authority.
        The v6 policy row therefore retains a bounded, digest-only rollout
        scope.  The comparison below consumes that immutable scope and never
        trusts the candidate epoch's backdateable ``created_at`` value.

        A strict subset is always safe to activate.  Every action whose tenant
        or rule coverage grows must instead have spent seven days in the exact
        preceding durable epoch and have 1,000 distinct issuances whose first
        1,000 reviews are completely safe.  One unsafe review anywhere in that
        epoch/action blocks expansion.
        """

        candidate_record = self._repository.get_semantic_policy_epoch(
            epoch.epoch_id
        )
        if previous_record is None or candidate_record is None:
            raise SemanticControlConflict(
                "semantic rollout scope evidence is absent"
            )
        previous_scope = getattr(previous_record, "rollout_scope", None)
        candidate_scope = getattr(candidate_record, "rollout_scope", None)
        affected_actions = _expanded_rollout_actions(
            previous_scope,
            candidate_scope,
        )
        if not affected_actions:
            return

        # Expansion is a one-way safety gate.  An unsafe Host review in any
        # historical semantic settlement remains append-only evidence and
        # permanently prevents widening a later tenant/rule cohort.  A new
        # epoch may still narrow authority, but it cannot erase this proof by
        # moving the per-action review window forward.
        if self._unsafe_review_count() > 0:
            raise SemanticControlConflict(
                "semantic rollout has unsafe review evidence"
            )

        evidence_reader = getattr(
            self._repository,
            "semantic_rollout_review_evidence",
            None,
        )
        if not callable(evidence_reader):
            raise SemanticControlConflict(
                "semantic rollout review evidence is unavailable"
            )
        now = _timestamp_value(self._now(), "semantic rollout Host clock")
        for action_id in affected_actions:
            evidence = evidence_reader(
                epoch_id=previous_record.epoch_id,
                action_id=action_id,
                limit=_ROLLOUT_REQUIRED_SAFE_REVIEWS,
            )
            _validate_rollout_review_evidence(
                evidence,
                action_id=action_id,
                now=now,
            )

    def _inactive_target(
        self,
        expected: SemanticControlStateRecord | None,
        *,
        mode: SemanticRuntimeMode | None = None,
    ) -> tuple[SemanticControlStateV1 | None, str | None]:
        selected_mode = mode or self._mode
        if selected_mode not in _INACTIVE_MODES:
            raise SemanticControlConflict(
                "semantic inactive target must be off or shadow"
            )
        if expected is None:
            return (
                SemanticControlStateV1(
                    revision=0,
                    generation=0,
                    mode=selected_mode,
                    active_epoch_id=None,
                    active_policy_sha256=None,
                    tripped=False,
                    trip_code=None,
                    updated_at=self._now(),
                ),
                None,
            )
        current = _public_control_state(expected)
        if (
            current.mode is selected_mode
            and current.active_epoch_id is None
            and current.active_policy_sha256 is None
            and not current.tripped
        ):
            return None, None
        return (
            SemanticControlStateV1(
                revision=current.revision + 1,
                # Preserve the high-water mark: an off/shadow transition
                # invalidates old grants without allowing an old epoch to be
                # activated again later.
                generation=current.generation,
                mode=selected_mode,
                active_epoch_id=None,
                active_policy_sha256=None,
                tripped=False,
                trip_code=None,
                updated_at=self._now(),
            ),
            "authority_cleared",
        )

    def _previous_policy_record(
        self,
        current: SemanticControlStateV1,
    ) -> Any | None:
        if current.generation == 0:
            return None
        if current.active_policy_sha256 is not None:
            self._verify_current_epoch_record(current)
            assert current.active_epoch_id is not None
            return self._repository.get_semantic_policy_epoch(
                current.active_epoch_id
            )
        query = getattr(self._repository, "query_semantic_policy_epochs", None)
        if not callable(query):
            raise SemanticControlConflict(
                "semantic inactive control cannot verify prior policy history"
            )
        after = None
        scanned = 0
        while scanned < _POLICY_SCAN_MAX_RECORDS:
            page = query(limit=_POLICY_SCAN_PAGE_SIZE, after=after)
            records = tuple(page.records)
            scanned += len(records)
            for record in records:
                if record.generation == current.generation:
                    return record
            if page.next_cursor is None:
                break
            if not records:
                raise SemanticControlConflict(
                    "semantic policy history returned an invalid empty page"
                )
            after = page.next_cursor
        raise SemanticControlConflict(
            "semantic previous policy is absent or outside the bounded history scan"
        )

    def _verify_current_epoch_record(
        self,
        current: SemanticControlStateV1,
    ) -> None:
        if current.active_epoch_id is None or current.active_policy_sha256 is None:
            raise SemanticControlConflict(
                "semantic active epoch identity is incomplete"
            )
        record = self._repository.get_semantic_policy_epoch(
            current.active_epoch_id
        )
        if (
            record is None
            or record.generation != current.generation
            or record.policy_sha256 != current.active_policy_sha256
        ):
            raise SemanticControlConflict(
                "semantic active policy evidence is absent or inconsistent"
            )

    def _record_transition_health(
        self,
        transition_kind: str,
        target: SemanticControlStateV1,
    ) -> None:
        if transition_kind == "authority_cleared":
            self._record_health(
                event_kind="semantic_control_authority_cleared",
                severity="warning",
                epoch_id=None,
                tenant_bucket_sha256=None,
                evidence_sha256=_evidence_sha256(
                    {
                        "event": transition_kind,
                        "mode": target.mode.value,
                        "generation": target.generation,
                        "revision": target.revision,
                    }
                ),
            )
        elif transition_kind in {"activated", "rotated"}:
            self._record_health(
                event_kind=f"semantic_policy_{transition_kind}",
                severity="info",
                epoch_id=target.active_epoch_id,
                tenant_bucket_sha256=None,
                evidence_sha256=_evidence_sha256(
                    {
                        "event": transition_kind,
                        "mode": target.mode.value,
                        "epoch_id": target.active_epoch_id,
                        "policy_sha256": target.active_policy_sha256,
                        "generation": target.generation,
                        "revision": target.revision,
                    }
                ),
            )
        elif transition_kind == "startup_unsafe_review_trip":
            self._record_health(
                event_kind="semantic_safety_trip:unsafe_review",
                severity="critical",
                epoch_id=target.active_epoch_id,
                tenant_bucket_sha256=None,
                evidence_sha256=_evidence_sha256(
                    {
                        "event": transition_kind,
                        "epoch_id": target.active_epoch_id,
                        "policy_sha256": target.active_policy_sha256,
                        "generation": target.generation,
                        "revision": target.revision,
                    }
                ),
            )

    def _record_conflict_health(self, error: Exception) -> None:
        current_record = None
        try:
            current_record = self._repository.get_semantic_control_state()
        except Exception:
            pass
        current = (
            _public_control_state(current_record)
            if current_record is not None
            else None
        )
        self._record_health(
            event_kind="semantic_control_startup_conflict",
            severity="critical",
            epoch_id=(
                self._policy_epoch.epoch_id
                if self._policy_epoch is not None
                else None
            ),
            tenant_bucket_sha256=None,
            evidence_sha256=_evidence_sha256(
                {
                    "event": "startup_conflict",
                    "error_type": type(error).__name__,
                    "configured_mode": self._mode.value,
                    "configured_epoch_sha256": (
                        self._policy_epoch.canonical_sha256()
                        if self._policy_epoch is not None
                        else None
                    ),
                    "durable_revision": (
                        current.revision if current is not None else None
                    ),
                    "durable_generation": (
                        current.generation if current is not None else None
                    ),
                    "durable_mode": (
                        current.mode.value if current is not None else None
                    ),
                    "durable_policy_sha256": (
                        current.active_policy_sha256
                        if current is not None
                        else None
                    ),
                }
            ),
        )

    def _record_health(
        self,
        *,
        event_kind: str,
        severity: str,
        epoch_id: str | None,
        tenant_bucket_sha256: str | None,
        evidence_sha256: str,
    ) -> None:
        append = getattr(
            self._repository,
            "append_semantic_health_event",
            None,
        )
        if not callable(append):
            return
        try:
            append(
                SemanticHealthEventRecord(
                    event_id=new_id("semantic_health"),
                    event_kind=event_kind,
                    severity=severity,
                    epoch_id=epoch_id,
                    tenant_bucket_sha256=tenant_bucket_sha256,
                    evidence_sha256=evidence_sha256,
                    created_at=self._now(),
                )
            )
        except Exception:
            # Health evidence must never prevent a kill switch, safety trip,
            # or a fail-closed startup decision from taking effect.
            return


def _public_control_state(
    record: SemanticControlStateRecord | SemanticControlStateV1,
) -> SemanticControlStateV1:
    if isinstance(record, SemanticControlStateV1):
        return record
    if not isinstance(record, SemanticControlStateRecord):
        raise SemanticControlConflict(
            "semantic repository returned an invalid control state"
        )
    return SemanticControlStateV1.from_dict(record.to_dict())


def _evidence_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256 digest")


def _expanded_rollout_actions(
    previous_scope: object,
    candidate_scope: object,
) -> tuple[str, ...]:
    (
        previous_tenants,
        previous_rules,
        previous_denies,
        previous_parameters,
    ) = _rollout_scope_parts(
        previous_scope,
        "preceding semantic rollout scope",
    )
    (
        candidate_tenants,
        candidate_rules,
        candidate_denies,
        candidate_parameters,
    ) = _rollout_scope_parts(
        candidate_scope,
        "candidate semantic rollout scope",
    )
    previous_actions = {rule["action_id"] for rule in previous_rules}
    candidate_actions = {rule["action_id"] for rule in candidate_rules}
    affected: set[str] = set()
    if not candidate_tenants.issubset(previous_tenants):
        affected.update(candidate_actions)
    if _rollout_allow_parameters_expand(
        previous_parameters,
        candidate_parameters,
    ):
        affected.update(candidate_actions)
    for candidate in candidate_rules:
        if not any(
            _rollout_rule_covers(
                previous,
                candidate,
                require_rule_identity=True,
            )
            for previous in previous_rules
        ):
            affected.add(candidate["action_id"])
    for previous_deny in previous_denies:
        action_id = previous_deny["action_id"]
        if action_id in candidate_actions and not _rollout_deny_union_covers(
            previous_deny,
            candidate_denies,
        ):
            affected.add(action_id)
    missing_canary = affected - previous_actions
    if missing_canary:
        raise SemanticControlConflict(
            "semantic rollout expansion introduces an action without a preceding canary: "
            + ", ".join(sorted(missing_canary))
        )
    return tuple(sorted(affected))


def _rollout_scope_parts(
    value: object,
    label: str,
) -> tuple[
    set[str],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    if not isinstance(value, Mapping):
        raise SemanticControlConflict(f"{label} is absent or malformed")
    if set(value) != _ROLLOUT_SCOPE_KEYS or value.get("schema_version") != 2:
        raise SemanticControlConflict(f"{label} has an unsupported contract")
    raw_tenants = value.get("tenant_bucket_sha256s")
    raw_rules = value.get("auto_approval_rules")
    raw_denies = value.get("hard_deny_rules")
    if not isinstance(raw_tenants, (list, tuple)) or not isinstance(
        raw_rules,
        (list, tuple),
    ) or not isinstance(
        raw_denies,
        (list, tuple),
    ):
        raise SemanticControlConflict(f"{label} collections are malformed")
    tenants: set[str] = set()
    for tenant in raw_tenants:
        try:
            _require_sha256(tenant, f"{label} tenant")
        except ValueError as exc:
            raise SemanticControlConflict(str(exc)) from exc
        if tenant in tenants:
            raise SemanticControlConflict(f"{label} contains duplicate tenants")
        tenants.add(tenant)
    rules: list[dict[str, Any]] = []
    for raw_rule in raw_rules:
        rules.append(_rollout_rule(raw_rule, label, catalog_rule=True))
    denies: list[dict[str, Any]] = []
    for raw_deny in raw_denies:
        denies.append(_rollout_rule(raw_deny, label, catalog_rule=False))
    all_ids = [
        *(rule["rule_id_sha256"] for rule in rules),
        *(rule["rule_id_sha256"] for rule in denies),
    ]
    if len(all_ids) != len(set(all_ids)):
        raise SemanticControlConflict(f"{label} contains duplicate rule ids")
    parameters = _rollout_allow_parameters(value.get("allow_parameters"), label)
    return tenants, tuple(rules), tuple(denies), parameters


def _rollout_rule(
    value: object,
    label: str,
    *,
    catalog_rule: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROLLOUT_RULE_KEYS:
        raise SemanticControlConflict(f"{label} rule metadata is malformed")
    action_id = value.get("action_id")
    rights = value.get("rights")
    resource_kind = value.get("resource_kind")
    prefixes = value.get("covering_prefix_sha256s")
    if (
        not isinstance(action_id, str)
        or _ROLLOUT_ACTION_ID.fullmatch(action_id) is None
        or not isinstance(rights, (list, tuple))
        or not rights
        or any(not isinstance(right, str) or not right for right in rights)
        or len(set(rights)) != len(rights)
        or tuple(rights) != tuple(sorted(rights))
        or resource_kind not in {"exact", "prefix"}
        or not isinstance(prefixes, (list, tuple))
        or (catalog_rule and frozenset(rights) != _ROLLOUT_ACTION_RIGHTS.get(action_id))
        or (not catalog_rule and not set(rights).issubset(_ROLLOUT_DENY_RIGHTS))
    ):
        raise SemanticControlConflict(f"{label} rule metadata is malformed")
    for field in ("rule_id_sha256", "match_sha256"):
        try:
            _require_sha256(value.get(field), f"{label} {field}")
        except ValueError as exc:
            raise SemanticControlConflict(str(exc)) from exc
    selected_prefixes = _rollout_covering_prefixes(
        prefixes,
        label=label,
        resource_kind=resource_kind,
        match_sha256=value["match_sha256"],
    )
    return {
        "rule_id_sha256": value["rule_id_sha256"],
        "action_id": action_id,
        "rights": frozenset(rights),
        "resource_kind": resource_kind,
        "match_sha256": value["match_sha256"],
        "covering_prefix_sha256s": selected_prefixes,
    }


def _rollout_allow_parameters(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROLLOUT_ALLOW_PARAMETER_KEYS:
        raise SemanticControlConflict(f"{label} allow parameters are malformed")
    if type(value.get("catalog_version")) is not int or value.get("catalog_version") != 1:
        raise SemanticControlConflict(f"{label} catalog version is unsupported")
    identity = tuple(
        value.get(field)
        for field in (
            "classifier_profile_id_sha256",
            "classifier_profile_sha256",
            "classifier_model_sha256",
        )
    )
    for field, item in zip(
        (
            "classifier_profile_id_sha256",
            "classifier_profile_sha256",
            "classifier_model_sha256",
        ),
        identity,
        strict=True,
    ):
        if item is not None:
            try:
                _require_sha256(item, f"{label} {field}")
            except ValueError as exc:
                raise SemanticControlConflict(str(exc)) from exc
    if any(item is None for item in identity) and any(
        item is not None for item in identity
    ):
        raise SemanticControlConflict(f"{label} classifier identity is incomplete")
    if value.get("required_calibration_bucket") != "very_high":
        raise SemanticControlConflict(f"{label} calibration bucket is unsupported")
    bounds = {
        "minimum_confidence_bps": (9_900, 10_000),
        "capability_ttl_s": (1, 300),
        "per_rule_per_minute_limit": (1, 10),
        "per_rule_per_day_limit": (1, 100),
        "max_inflight": (1, 2),
    }
    selected = dict(value)
    for field, (minimum, maximum) in bounds.items():
        item = selected.get(field)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            raise SemanticControlConflict(f"{label} {field} is malformed")
    return selected


def _rollout_allow_parameters_expand(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    identity_fields = (
        "catalog_version",
        "classifier_profile_id_sha256",
        "classifier_profile_sha256",
        "classifier_model_sha256",
        "required_calibration_bucket",
    )
    if any(candidate[field] != previous[field] for field in identity_fields):
        return True
    if candidate["minimum_confidence_bps"] < previous["minimum_confidence_bps"]:
        return True
    for field in (
        "capability_ttl_s",
        "per_rule_per_minute_limit",
        "per_rule_per_day_limit",
        "max_inflight",
    ):
        if candidate[field] > previous[field]:
            return True
    return False


def _rollout_covering_prefixes(
    prefixes: list[Any] | tuple[Any, ...],
    *,
    label: str,
    resource_kind: object,
    match_sha256: object,
) -> frozenset[str]:
    selected_prefixes: set[str] = set()
    for prefix in prefixes:
        try:
            _require_sha256(prefix, f"{label} covering prefix")
        except ValueError as exc:
            raise SemanticControlConflict(str(exc)) from exc
        if prefix in selected_prefixes:
            raise SemanticControlConflict(
                f"{label} contains duplicate covering prefixes"
            )
        selected_prefixes.add(prefix)
    if resource_kind == "prefix" and match_sha256 not in selected_prefixes:
        raise SemanticControlConflict(
            f"{label} prefix rule does not cover its own match"
        )
    return frozenset(selected_prefixes)


def _rollout_rule_covers(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    require_rule_identity: bool = False,
) -> bool:
    if (
        previous["action_id"] != candidate["action_id"]
        or (
            require_rule_identity
            and previous["rule_id_sha256"] != candidate["rule_id_sha256"]
        )
        or not candidate["rights"].issubset(previous["rights"])
    ):
        return False
    return _rollout_resource_covers(previous, candidate)


def _rollout_resource_covers(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    if previous["resource_kind"] == "exact":
        return (
            candidate["resource_kind"] == "exact"
            and previous["match_sha256"] == candidate["match_sha256"]
        )
    return previous["match_sha256"] in candidate["covering_prefix_sha256s"]


def _rollout_deny_union_covers(
    previous: Mapping[str, Any],
    candidates: tuple[dict[str, Any], ...],
) -> bool:
    return all(
        any(
            candidate["action_id"] == previous["action_id"]
            and right in candidate["rights"]
            and _rollout_resource_covers(candidate, previous)
            for candidate in candidates
        )
        for right in previous["rights"]
    )


def _validate_rollout_review_evidence(
    value: object,
    *,
    action_id: str,
    now: datetime,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "activated_at",
        "issued_count",
        "required_count",
        "completely_safe_count",
        "unsafe_count",
    } or value.get("schema_version") != 1:
        raise SemanticControlConflict(
            f"semantic rollout evidence for {action_id} is malformed"
        )
    activated_at = value.get("activated_at")
    if activated_at is None:
        raise SemanticControlConflict(
            f"semantic rollout action {action_id} has no durable activation history"
        )
    activated = _timestamp_value(
        activated_at,
        f"semantic rollout activation for {action_id}",
    )
    if now < activated or now - activated < _ROLLOUT_MINIMUM_AGE:
        raise SemanticControlConflict(
            f"semantic rollout action {action_id} has not completed seven days"
        )
    counts: dict[str, int] = {}
    for field in (
        "issued_count",
        "required_count",
        "completely_safe_count",
        "unsafe_count",
    ):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SemanticControlConflict(
                f"semantic rollout evidence {field} for {action_id} is invalid"
            )
        counts[field] = item
    if counts["issued_count"] < _ROLLOUT_REQUIRED_SAFE_REVIEWS:
        raise SemanticControlConflict(
            f"semantic rollout action {action_id} has fewer than 1000 issuances"
        )
    if (
        counts["required_count"] != _ROLLOUT_REQUIRED_SAFE_REVIEWS
        or counts["completely_safe_count"]
        != _ROLLOUT_REQUIRED_SAFE_REVIEWS
    ):
        raise SemanticControlConflict(
            f"semantic rollout action {action_id} lacks 1000 complete safe reviews"
        )
    if counts["unsafe_count"] != 0:
        raise SemanticControlConflict(
            f"semantic rollout action {action_id} has unsafe review evidence"
        )


def _timestamp_value(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise SemanticControlConflict(f"{label} is not a bounded timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticControlConflict(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SemanticControlConflict(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "SemanticControlConflict",
    "SemanticRuntimeControl",
]
