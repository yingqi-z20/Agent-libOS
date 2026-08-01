"""Evidence-layer helpers for atomic protected external-effect settlement.

Provider-facing subsystems use :mod:`agent_libos.sdk`; this module is the
narrow ledger implementation behind that boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
import hashlib
from typing import Any, Callable, Mapping

from agent_libos.models import (
    AuditRecord,
    Event,
    ExternalEffectClassification,
    ExternalEffectCursor,
    ExternalEffectRecord,
    ExternalEffectRecoveryQuery,
    ExternalEffectRecoverySettlement,
    ExternalEffectRecoverySummary,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.ports import EffectAuthorityPort, OperationPort, ProtectedEffectPort
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.capability.effect_binding import (
    canonical_effect_hash,
    current_approval_effect_binding,
)
from agent_libos.utils.serde import dumps


def require_external_effect_classifier(provider: Any, operation: str) -> None:
    if not callable(getattr(provider, "classify_external_effect", None)):
        raise ValidationError(
            f"provider {provider.__class__.__name__} cannot classify external effect operation {operation!r}"
        )


def classify_external_effect(
    provider: Any,
    operation: str,
    context: dict[str, Any],
    result: Any,
) -> ExternalEffectClassification:
    require_external_effect_classifier(provider, operation)
    raw = provider.classify_external_effect(operation, context, result)
    if isinstance(raw, ExternalEffectClassification):
        return raw
    if isinstance(raw, dict):
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass(str(raw["rollback_class"])),
            rollback_status=ExternalEffectRollbackStatus(str(raw["rollback_status"])),
            state_mutation=bool(raw["state_mutation"]),
            information_flow=bool(raw["information_flow"]),
            metadata=dict(raw.get("metadata") or {}),
        )
    raise ValidationError("provider external effect classifier must return ExternalEffectClassification")


def record_external_effect(
    store: ProtectedEffectPort,
    *,
    pid: str,
    provider: str,
    operation: str,
    target: str | None,
    classification: ExternalEffectClassification,
    audit_record: AuditRecord | None,
    event: Event | None,
    metadata: dict[str, Any] | None = None,
    intent_effect_id: str | None = None,
    operations: OperationPort | None = None,
) -> ExternalEffectRecord:
    provider_metadata = {
        **dict(classification.metadata),
        **dict(metadata or {}),
        "effect_state": "finalized",
    }
    intent = store.get_external_effect(intent_effect_id) if intent_effect_id is not None else None
    # Rollback support and provider outcome are separate axes. A provider may
    # confirm that an effect committed while being unable to classify how it
    # could be rolled back. Only an explicitly unknown outcome makes the
    # transaction outcome unknown, while an explicit failed outcome records a
    # terminal dispatch failure rather than manufacturing a commit.
    provider_outcome = str(provider_metadata.get("outcome") or "")
    transaction_state = (
        "unknown"
        if provider_outcome.startswith("unknown")
        else "failed"
        if provider_outcome == "failed"
        else "committed"
    )
    receipt = provider_metadata.get("provider_receipt")
    record = ExternalEffectRecord(
        effect_id=intent_effect_id or new_id("eff"),
        record_id=audit_record.record_id if audit_record is not None else None,
        event_id=event.event_id if event is not None else None,
        pid=pid,
        provider=provider,
        operation=operation,
        target=target,
        rollback_class=classification.rollback_class,
        rollback_status=classification.rollback_status,
        state_mutation=classification.state_mutation,
        information_flow=classification.information_flow,
        provider_metadata=provider_metadata,
        created_at=utc_now(),
        effect_state="finalized",
        transaction_state=transaction_state,
        provider_receipt=dict(receipt) if isinstance(receipt, dict) else {},
        canonical_args_hash=intent.canonical_args_hash if intent is not None else None,
        idempotency_key=intent.idempotency_key if intent is not None else None,
        updated_at=utc_now(),
    )
    if intent_effect_id is None:
        store.insert_external_effect(record)
    elif not store.finalize_external_effect(intent_effect_id, record):
        raise ValidationError(
            "external effect intent was missing, already finalized, or did not match its provider boundary"
        )
    if operations is not None:
        operations.link_evidence(
            "external_effect",
            record.effect_id,
            "effect",
            metadata={"effect_state": "finalized", "provider": provider, "operation": operation},
        )
    return record


def prepare_external_effect_intent(
    store: ProtectedEffectPort,
    *,
    pid: str,
    provider: str,
    operation: str,
    target: str | None,
    state_mutation: bool,
    information_flow: bool,
    metadata: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    canonical_args: dict[str, Any] | None = None,
    operations: OperationPort | None = None,
    authority_policy: EffectAuthorityPort | None = None,
) -> ExternalEffectRecord:
    """Persist conservative evidence before a provider boundary.

    The returned transaction remains ``prepared``.  Callers must CAS it to
    ``dispatched`` immediately before invoking a provider.  Keeping these two
    steps separate lets the protected-operation SDK distinguish a local
    pre-provider abort from a process crash at the provider boundary.

    An explicit idempotency key is unique only within ``pid`` and remains
    claimed for every retained effect state.  When omitted, the generated key
    includes the current operation/effect identity and therefore prevents a
    duplicate lifecycle for that identity; it is not a cross-operation retry
    key.  Abandoning a certified-not-started intent deletes the row and releases
    its key because no provider phase began.
    """

    manifest = _effect_manifest(authority_policy, pid, provider, operation)
    context = dict((metadata or {}).get("context") or metadata or {})
    binding_context = dict(canonical_args) if canonical_args is not None else context
    operation_id = operations.current_id() if operations is not None else None
    effect_id, args_hash, selected_idempotency_key = _effect_identity(
        store=store,
        pid=pid,
        provider=provider,
        operation=operation,
        target=target,
        binding_context=binding_context,
        canonical_args_supplied=canonical_args is not None,
        operation_id=operation_id,
        idempotency_key=idempotency_key,
    )
    now = utc_now()
    record = ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider=provider,
        operation=operation,
        target=target,
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=state_mutation,
        information_flow=information_flow,
        provider_metadata={
            **_effect_metadata(metadata, information_flow, manifest),
            "effect_state": "pending",
            "outcome": "unknown_after_provider_boundary",
        },
        created_at=now,
        effect_state="pending",
        transaction_state="prepared",
        canonical_args_hash=args_hash,
        idempotency_key=selected_idempotency_key,
        updated_at=now,
    )
    _insert_effect_intent(store, record)
    if operations is not None:
        _link_pending_effect(operations, record)
    return record


def _effect_manifest(
    authority_policy: EffectAuthorityPort | None,
    pid: str,
    provider: str,
    operation: str,
) -> Any | None:
    if authority_policy is None:
        return None
    authority_policy.assert_effect(pid, f"{provider}.{operation}")
    return authority_policy.get_for_process(pid)


def _effect_identity(
    *,
    store: ProtectedEffectPort,
    pid: str,
    provider: str,
    operation: str,
    target: str | None,
    binding_context: dict[str, Any],
    canonical_args_supplied: bool,
    operation_id: str | None,
    idempotency_key: str | None,
) -> tuple[str, str, str]:
    approval = current_approval_effect_binding(store, operation_id)
    effect_id = approval["effect_id"] if approval is not None else new_id("effintent")
    computed_hash = canonical_effect_hash(binding_context)
    if (
        approval is not None
        and canonical_args_supplied
        and approval["canonical_args_hash"] != computed_hash
    ):
        raise ValidationError(
            "protected operation arguments do not match the approved external effect"
        )
    args_hash = approval["canonical_args_hash"] if approval is not None else computed_hash
    selected_key = idempotency_key or _default_idempotency_key(
        operation_id=operation_id,
        effect_id=effect_id,
        provider=provider,
        operation=operation,
        target=target,
        args_hash=args_hash,
    )
    existing = _effect_with_idempotency_key(store, pid, selected_key)
    if existing is not None:
        raise ValidationError(
            "duplicate external effect dispatch blocked by idempotency key: "
            f"{selected_key} existing_effect={existing.effect_id} "
            f"state={existing.transaction_state}"
        )
    return effect_id, args_hash, selected_key


def _default_idempotency_key(
    *,
    operation_id: str | None,
    effect_id: str,
    provider: str,
    operation: str,
    target: str | None,
    args_hash: str,
) -> str:
    return hashlib.sha256(
        dumps(
            {
                "operation_id": operation_id or effect_id,
                "provider": provider,
                "operation": operation,
                "target": target,
                "canonical_args_hash": args_hash,
            }
        ).encode("utf-8")
    ).hexdigest()


def _effect_with_idempotency_key(
    store: ProtectedEffectPort,
    pid: str,
    idempotency_key: str,
) -> ExternalEffectRecord | None:
    return store.get_external_effect_by_idempotency(pid, idempotency_key)


def iter_external_effect_recovery(
    store: ProtectedEffectPort,
    query: ExternalEffectRecoveryQuery,
) -> Iterator[ExternalEffectRecord]:
    """Yield one validated keyset page at a time without scanning history."""

    current = query
    while True:
        page = store.query_external_effect_recovery(current)
        if len(page.records) > current.limit:
            raise ValidationError(
                "external effect recovery repository exceeded the requested page limit"
            )
        previous = current.after
        for effect in page.records:
            cursor = ExternalEffectCursor(effect.created_at, effect.effect_id)
            if previous is not None and cursor <= previous:
                raise ValidationError(
                    "external effect recovery repository returned a non-monotonic page"
                )
            if effect.effect_state != current.effect_state:
                raise ValidationError(
                    "external effect recovery repository returned an ineligible effect state"
                )
            if (
                current.transaction_states
                and effect.transaction_state not in current.transaction_states
            ):
                raise ValidationError(
                    "external effect recovery repository returned an ineligible transaction state"
                )
            yield effect
            previous = cursor
        if page.next_cursor is None:
            return
        if previous is None or page.next_cursor != previous:
            raise ValidationError(
                "external effect recovery repository returned an invalid next cursor"
            )
        current = replace(current, after=page.next_cursor)


def _effect_metadata(
    metadata: dict[str, Any] | None,
    information_flow: bool,
    manifest: Any | None,
) -> dict[str, Any]:
    selected = dict(metadata or {})
    if not information_flow:
        return selected
    raw_labels = selected.get("data_labels")
    label_fields = {
        "sensitivity",
        "trust_level",
        "integrity",
        "origin",
        "tenant",
        "principal",
    }
    selected["information_flow_evidence"] = {
        "mode": "observe_only",
        "labels": {
            str(key): value
            for key, value in dict(raw_labels or {}).items()
            if str(key) in label_fields
        },
        "manifest_policy": (
            dict(manifest.data_flow_policy) if manifest is not None else {}
        ),
    }
    return selected


def _insert_effect_intent(
    store: ProtectedEffectPort,
    record: ExternalEffectRecord,
) -> None:
    try:
        store.insert_external_effect(record)
    except Exception as exc:
        raced = _effect_with_idempotency_key(
            store,
            record.pid,
            str(record.idempotency_key),
        )
        if raced is not None:
            raise ValidationError(
                "duplicate external effect dispatch blocked by concurrent idempotency claim: "
                f"{record.idempotency_key} existing_effect={raced.effect_id}"
            ) from exc
        raise


def _link_pending_effect(
    operations: OperationPort,
    record: ExternalEffectRecord,
) -> None:
    # Event/audit roles become required only after a provider returned or
    # became ambiguous. Pre-dispatch aborts must not report false gaps.
    operations.expect("effect")
    operations.link_evidence(
        "external_effect",
        record.effect_id,
        "effect",
        metadata={
            "effect_state": "pending",
            "provider": record.provider,
            "operation": record.operation,
        },
    )


def mark_external_effect_dispatched(store: ProtectedEffectPort, effect_id: str) -> ExternalEffectRecord:
    current = store.get_external_effect(effect_id)
    if current is None:
        raise ValidationError(f"external effect intent not found: {effect_id}")
    metadata = {**dict(current.provider_metadata), "transaction_state": "dispatched"}
    if not store.transition_external_effect(
        effect_id,
        expected_states=("prepared", "authorized", "approved"),
        transaction_state="dispatched",
        provider_metadata=metadata,
        updated_at=utc_now(),
    ):
        refreshed = store.get_external_effect(effect_id)
        if refreshed is None or refreshed.transaction_state != "dispatched":
            raise ValidationError(f"external effect intent cannot be dispatched: {effect_id}")
    return store.get_external_effect(effect_id) or current


def mark_external_effect_unknown(
    store: ProtectedEffectPort,
    effect_id: str,
    *,
    reason: str,
    provider_receipt: dict[str, Any] | None = None,
) -> ExternalEffectRecord:
    current = store.get_external_effect(effect_id)
    if current is None:
        raise ValidationError(f"external effect intent not found: {effect_id}")
    selected_receipt = (
        provider_receipt
        if provider_receipt is not None
        else dict(current.provider_receipt)
    )
    metadata = {
        **dict(current.provider_metadata),
        "outcome": "unknown",
        "reconciliation_reason": reason,
        "transaction_state": "unknown",
    }
    if (
        current.transaction_state == "unknown"
        and current.provider_metadata == metadata
        and current.provider_receipt == selected_receipt
    ):
        return current
    if not store.transition_external_effect(
        effect_id,
        expected_states=("prepared", "authorized", "approved", "dispatched", "unknown"),
        transaction_state="unknown",
        provider_metadata=metadata,
        provider_receipt=selected_receipt,
        updated_at=utc_now(),
    ):
        raise ValidationError(f"external effect intent cannot become unknown: {effect_id}")
    return store.get_external_effect(effect_id) or current


def reconcile_pending_external_effects(
    store: ProtectedEffectPort,
    substrate: Any,
    *,
    require_recovery_lease: Callable[[], None],
    page_size: int = 500,
    provider_overrides: Mapping[str, Any] | None = None,
) -> ExternalEffectRecoverySummary:
    """Reconcile without replay; unsupported providers remain explicitly unknown."""

    require_recovery_lease()

    reconciled_sample: list[str] = []
    reconciled_total = 0
    query = ExternalEffectRecoveryQuery(limit=page_size)
    for effect in iter_external_effect_recovery(store, query):
        protected = effect.provider_metadata.get("protected_operation")
        if effect.transaction_state == "prepared" and isinstance(protected, dict):
            # The SDK owns local prepare recovery, including capability and
            # domain-state restoration. Never ask a provider about a boundary
            # that was durably proven not to have been dispatched.
            continue
        _reconcile_pending_external_effect(
            store,
            substrate,
            effect,
            provider_overrides=provider_overrides,
        )
        reconciled_total += 1
        if len(reconciled_sample) < page_size:
            reconciled_sample.append(effect.effect_id)
    return ExternalEffectRecoverySummary(
        total_count=reconciled_total,
        sample_effect_ids=tuple(reconciled_sample),
    )


def _reconcile_pending_external_effect(
    store: ProtectedEffectPort,
    substrate: Any,
    effect: ExternalEffectRecord,
    *,
    provider_overrides: Mapping[str, Any] | None,
) -> None:
    provider = (
        provider_overrides[effect.provider]
        if provider_overrides is not None and effect.provider in provider_overrides
        else getattr(substrate, effect.provider, None)
    )
    reconcile = getattr(provider, "reconcile_external_effect", None)
    if not callable(reconcile):
        mark_external_effect_unknown(
            store,
            effect.effect_id,
            reason="provider_does_not_support_reconciliation",
        )
        return
    try:
        result = reconcile(effect)
    except Exception as exc:
        mark_external_effect_unknown(
            store,
            effect.effect_id,
            reason=f"provider_reconciliation_error:{type(exc).__name__}",
        )
        return
    if not isinstance(result, Mapping):
        mark_external_effect_unknown(
            store,
            effect.effect_id,
            reason="invalid_reconciliation_result",
        )
        return
    _persist_provider_reconciliation_result(store, effect, result)


def _persist_provider_reconciliation_result(
    store: ProtectedEffectPort,
    effect: ExternalEffectRecord,
    result: Mapping[str, Any],
) -> None:
    receipt = result.get("provider_receipt")
    state = _normalized_provider_reconciliation_state(
        result.get("state"),
        receipt,
    )
    if state not in {
        "committed",
        "failed",
        "compensated",
        "not_started",
        "unknown",
    }:
        state = "unknown"
    if effect.transaction_state == "prepared" and state not in {
        "not_started",
        "unknown",
    }:
        mark_external_effect_unknown(
            store,
            effect.effect_id,
            reason="prepared_effect_requires_certified_not_started",
            provider_receipt=(
                dict(receipt) if isinstance(receipt, Mapping) else None
            ),
        )
        return
    if state in {"committed", "failed", "compensated", "not_started"}:
        _persist_provider_terminal_reconciliation(store, effect, state, receipt)
        return
    metadata = {
        **dict(effect.provider_metadata),
        "reconciled": True,
        "transaction_state": state,
        "outcome": state,
    }
    selected_receipt = dict(receipt) if isinstance(receipt, Mapping) else {}
    if (
        effect.transaction_state == "unknown"
        and effect.provider_metadata == metadata
        and effect.provider_receipt == selected_receipt
    ):
        return
    if not store.transition_external_effect(
        effect.effect_id,
        expected_states=("prepared", "authorized", "approved", "dispatched", "unknown"),
        transaction_state=state,
        provider_metadata=metadata,
        provider_receipt=selected_receipt,
        updated_at=utc_now(),
    ):
        raise ValidationError(
            f"external effect reconciliation raced: {effect.effect_id}"
        )


def _persist_provider_terminal_reconciliation(
    store: ProtectedEffectPort,
    effect: ExternalEffectRecord,
    provider_state: str,
    receipt: Any,
) -> None:
    if not isinstance(receipt, Mapping) or not receipt:
        mark_external_effect_unknown(
            store,
            effect.effect_id,
            reason="provider_terminal_state_without_authoritative_receipt",
        )
        return
    _settle_external_effect_recovery(
        store,
        effect=effect,
        provider_state=provider_state,
        provider_receipt=dict(receipt),
        source="provider_reconciliation",
    )


def settle_external_effect_from_authoritative_receipt(
    store: ProtectedEffectPort,
    *,
    provider: Any,
    run_id: str,
    effect_id: str,
    expected_transaction_state: str,
    provider_receipt: Mapping[str, Any],
    runtime_epoch: int,
    require_recovery_lease: Callable[[], None],
) -> ExternalEffectRecoverySettlement:
    """Verify and atomically settle one Host-selected provider receipt.

    The caller-provided receipt is never itself a certificate.  A configured
    provider must implement ``verify_external_effect_receipt`` and return the
    authoritative state plus the normalized receipt that should be retained.
    The SQL boundary then repeats the exact effect/state/Run epoch checks.
    """

    require_recovery_lease()
    if not isinstance(effect_id, str) or not effect_id:
        raise ValidationError("authoritative receipt requires an effect id")
    if expected_transaction_state not in {
        "prepared",
        "authorized",
        "approved",
        "dispatched",
        "unknown",
    }:
        raise ValidationError(
            "authoritative receipt expected state is not recoverable"
        )
    if not isinstance(provider_receipt, Mapping) or not provider_receipt:
        raise ValidationError("authoritative provider receipt is required")
    effect = store.get_external_effect(effect_id)
    if effect is None:
        raise ValidationError(f"external effect intent not found: {effect_id}")
    if (
        effect.effect_state != "pending"
        or effect.transaction_state != expected_transaction_state
    ):
        raise ValidationError(
            "authoritative receipt no longer matches the expected effect state"
        )
    verify = getattr(provider, "verify_external_effect_receipt", None)
    if not callable(verify):
        raise ValidationError(
            "configured provider cannot verify authoritative effect receipts"
        )
    verified = verify(effect, dict(provider_receipt))
    provider_state, normalized_receipt = _verified_provider_receipt(verified)
    settled = _settle_external_effect_recovery(
        store,
        effect=effect,
        provider_state=provider_state,
        provider_receipt=normalized_receipt,
        source="host_verified_receipt",
        run_id=run_id,
        runtime_epoch=runtime_epoch,
    )
    return settled


def _verified_provider_receipt(value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValidationError(
            "provider receipt verifier must return an authoritative result object"
        )
    receipt = value.get("provider_receipt")
    if not isinstance(receipt, Mapping) or not receipt:
        raise ValidationError(
            "provider receipt verifier returned no authoritative receipt"
        )
    state = _normalized_provider_reconciliation_state(value.get("state"), receipt)
    if state not in {"committed", "failed", "compensated", "not_started"}:
        raise ValidationError(
            "provider receipt verifier returned an unsupported effect state"
        )
    return state, dict(receipt)


def _normalized_provider_reconciliation_state(
    value: Any,
    receipt: Any,
) -> str:
    state = str(value or "unknown")
    if state == "failed" and _provider_receipt_certifies_not_started(receipt):
        return "not_started"
    return state


def _provider_receipt_certifies_not_started(receipt: Any) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    return bool(
        (
            receipt.get("dispatch_status") == "not_started"
            and receipt.get("certified") is True
        )
        or receipt.get("certified_not_started") is True
    )


def _settle_external_effect_recovery(
    store: ProtectedEffectPort,
    *,
    effect: ExternalEffectRecord,
    provider_state: str,
    provider_receipt: dict[str, Any],
    source: str,
    run_id: str | None = None,
    runtime_epoch: int | None = None,
) -> ExternalEffectRecoverySettlement:
    receipt_sha256 = hashlib.sha256(
        dumps(provider_receipt).encode("utf-8")
    ).hexdigest()
    settled_transaction_state = (
        "failed" if provider_state == "not_started" else provider_state
    )
    metadata = {
        **dict(effect.provider_metadata),
        "reconciled": True,
        "reconciliation_source": source,
        "provider_reconciliation_state": provider_state,
        "transaction_state": settled_transaction_state,
        "outcome": provider_state,
        "provider_receipt_sha256": receipt_sha256,
    }
    if provider_state == "not_started":
        metadata.update(
            {
                "certified_not_started": True,
                "dispatch_status": "not_started",
            }
        )
    now = utc_now()
    audit = AuditRecord(
        record_id=new_id("audit"),
        timestamp=now,
        actor="runtime.recovery",
        action="external_effect.recovery_settled",
        target=f"external_effect:{effect.effect_id}",
        input_refs=[],
        output_refs=[],
        capability_refs=[],
        decision={
            "source": source,
            "provider": effect.provider,
            "operation": effect.operation,
            "provider_state": provider_state,
            "provider_receipt_sha256": receipt_sha256,
        },
        correlation_id=effect.effect_id,
    )
    settled = store.settle_external_effect_recovery(
        effect.effect_id,
        expected_transaction_state=effect.transaction_state,
        provider_state=provider_state,
        provider_metadata=metadata,
        provider_receipt=provider_receipt,
        audit_record=audit,
        updated_at=now,
        run_id=run_id,
        runtime_epoch=runtime_epoch,
    )
    if settled is None:
        raise ValidationError(
            f"external effect recovery raced or failed its fence: {effect.effect_id}"
        )
    return settled


def abandon_external_effect_intent(
    store: ProtectedEffectPort,
    intent_effect_id: str | None,
    *,
    operations: OperationPort | None = None,
) -> None:
    """Remove an intent only when no effectful provider phase can have begun.

    Valid callers are a pre-dispatch abort, startup recovery of a transaction
    still durably marked ``prepared``, or a phase-local not-started certificate
    when no earlier phase mutated state, observed information, or committed
    authority.
    """

    if intent_effect_id is not None:
        if not store.abandon_external_effect_intent(intent_effect_id):
            raise ValidationError("external effect intent was missing or already finalized")
        if operations is not None:
            operations.link_evidence(
                "external_effect",
                intent_effect_id,
                "result",
                metadata={"outcome": "not_started", "effect_state": "abandoned"},
            )


def external_effect_to_json(record: ExternalEffectRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["rollback_class"] = record.rollback_class.value
    payload["rollback_status"] = record.rollback_status.value
    return payload


def external_effect_summary(records: list[ExternalEffectRecord]) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_provider_operation: dict[str, int] = {}
    state_mutations = 0
    information_flows = 0
    by_state: dict[str, int] = {}
    for record in records:
        by_class[record.rollback_class.value] = by_class.get(record.rollback_class.value, 0) + 1
        key = f"{record.provider}.{record.operation}"
        by_provider_operation[key] = by_provider_operation.get(key, 0) + 1
        state_mutations += int(record.state_mutation)
        information_flows += int(record.information_flow)
        by_state[record.effect_state] = by_state.get(record.effect_state, 0) + 1
    return {
        "total": len(records),
        "by_rollback_class": by_class,
        "by_provider_operation": by_provider_operation,
        "state_mutations": state_mutations,
        "information_flows": information_flows,
        "by_state": by_state,
        "pending": by_state.get("pending", 0),
    }
