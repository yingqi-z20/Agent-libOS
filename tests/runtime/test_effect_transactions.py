from __future__ import annotations

from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.evidence.external_effects import (
    abandon_external_effect_intent,
    record_external_effect,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.external_effects import begin_external_effect_intent


def test_effect_transaction_records_hash_idempotency_dispatch_receipt_and_commit() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="effect transaction")
        with runtime.operations.scope(
            kind="primitive",
            name="test.effect",
            actor=pid,
            pid=pid,
        ):
            intent = begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="test",
                operation="write",
                target="record:1",
                state_mutation=True,
                information_flow=False,
                metadata={"context": {"record_id": "1", "value": "next"}},
                idempotency_key="effect-test-1",
            )
            assert intent.transaction_state == "dispatched"
            assert len(intent.canonical_args_hash or "") == 64
            committed = record_external_effect(
                runtime.uow.protected_effects,
                pid=pid,
                provider="test",
                operation="write",
                target="record:1",
                classification=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                    state_mutation=True,
                    information_flow=False,
                    metadata={},
                ),
                audit_record=None,
                event=None,
                metadata={"provider_receipt": {"revision": "2"}},
                intent_effect_id=intent.effect_id,
                operations=runtime.operations,
            )

        assert committed.transaction_state == "committed"
        assert committed.idempotency_key == "effect-test-1"
        assert committed.provider_receipt == {"revision": "2"}
        with pytest.raises(ValidationError, match="duplicate external effect dispatch"):
            begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="test",
                operation="write",
                target="record:1",
                state_mutation=True,
                information_flow=False,
                metadata={"context": {"record_id": "1", "value": "next"}},
                idempotency_key="effect-test-1",
            )
    finally:
        runtime.close()


def test_startup_reconciliation_queries_provider_without_replaying_effect(tmp_path: Path) -> None:
    database = tmp_path / "reconcile.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_substrate = LocalResourceProviderSubstrate(workspace)
    runtime = Runtime.open(database, substrate=first_substrate)
    try:
        pid = runtime.process.spawn(goal="pending effect")
        pending = begin_external_effect_intent(
            runtime,
            pid=pid,
            provider="jsonrpc",
            operation="call",
            target="connector:1",
            state_mutation=True,
            information_flow=True,
            metadata={"context": {"request_id": "stable-1"}},
            idempotency_key="stable-1",
        )
    finally:
        runtime.close()

    reconciler = _Reconciler()
    second_substrate = LocalResourceProviderSubstrate(workspace)
    second_substrate.jsonrpc = reconciler
    reopened = Runtime.open(database, substrate=second_substrate)
    try:
        recovered = reopened.store.get_external_effect(pending.effect_id)
        assert recovered is not None
        assert recovered.transaction_state == "committed"
        assert recovered.effect_state == "finalized"
        assert recovered.provider_receipt == {"provider_id": "receipt-1"}
        assert reconciler.reconcile_calls == [pending.effect_id]
        assert reconciler.provider_calls == 0
    finally:
        reopened.close()

    third_substrate = LocalResourceProviderSubstrate(workspace)
    third_substrate.jsonrpc = reconciler
    third = Runtime.open(database, substrate=third_substrate)
    try:
        assert reconciler.reconcile_calls == [pending.effect_id]
    finally:
        third.close()


def test_startup_reconciliation_failure_keeps_runtime_available_and_effect_unknown(tmp_path: Path) -> None:
    database = tmp_path / "reconcile-error.sqlite"
    workspace = tmp_path / "workspace-error"
    workspace.mkdir()
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(workspace))
    try:
        pid = runtime.process.spawn(goal="pending reconciliation error")
        pending = begin_external_effect_intent(
            runtime,
            pid=pid,
            provider="jsonrpc",
            operation="call",
            target="connector:error",
            state_mutation=True,
            information_flow=True,
            idempotency_key="reconcile-error",
        )
    finally:
        runtime.close()

    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.jsonrpc = _FailingReconciler()
    reopened = Runtime.open(database, substrate=substrate)
    try:
        recovered = reopened.store.get_external_effect(pending.effect_id)
        assert recovered is not None
        assert recovered.effect_state == "pending"
        assert recovered.transaction_state == "unknown"
        assert recovered.provider_metadata["reconciliation_reason"] == "provider_reconciliation_error:RuntimeError"
        transition_count = len(
            reopened.store._query(  # noqa: SLF001 - exact append-only evidence assertion
                "SELECT seq FROM external_effect_transitions WHERE effect_id = ?",
                (pending.effect_id,),
            )
        )
    finally:
        reopened.close()

    repeated_substrate = LocalResourceProviderSubstrate(workspace)
    repeated_substrate.jsonrpc = _FailingReconciler()
    repeated = Runtime.open(database, substrate=repeated_substrate)
    try:
        unchanged = repeated.store.get_external_effect(pending.effect_id)
        assert unchanged == recovered
        assert len(
            repeated.store._query(  # noqa: SLF001 - exact append-only evidence assertion
                "SELECT seq FROM external_effect_transitions WHERE effect_id = ?",
                (pending.effect_id,),
            )
        ) == transition_count
    finally:
        repeated.close()


def test_approval_binding_rejects_changed_arguments_and_target_version() -> None:
    runtime = Runtime.open("local")
    try:
        resource = "jsonrpc:demo:update"
        pid = runtime.process.spawn(
            goal="bound approval",
            authority_manifest={
                "authorized_capabilities": [
                    {"resource": "human:owner", "rights": [CapabilityRight.WRITE.value]},
                ]
            },
        )
        context = {
            "primitive": "runtime.jsonrpc.call",
            "operation": "update",
            "resource": resource,
            "record_id": "customer-1",
            "value": "approved",
            "target_state_version": "7",
        }
        request_id = runtime.human.query(
            pid=pid,
            human="owner",
            request={
                "type": "external_operation_approval",
                "question": "Approve update?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.WRITE.value],
                },
                "context": context,
            },
            blocking=True,
        )
        runtime.human.drain_terminal_queue(auto_approve=True)
        request = runtime.human.get(request_id)
        binding = request.payload["effect_binding"]

        matching = runtime.capability.authorize(pid, resource, CapabilityRight.WRITE, context)
        changed_args = runtime.capability.authorize(
            pid,
            resource,
            CapabilityRight.WRITE,
            {**context, "value": "changed"},
        )
        changed_version = runtime.capability.authorize(
            pid,
            resource,
            CapabilityRight.WRITE,
            {**context, "target_state_version": "8"},
        )

        assert matching.allowed
        assert not changed_args.allowed
        assert not changed_version.allowed
        with runtime.operations.scope(
            kind="primitive",
            name="test.approved_effect",
            actor=pid,
            pid=pid,
        ):
            reservation_id = runtime.capability.reserve_decision_use(
                matching,
                used_by="test.approved_effect",
                reason="reserve approved effect",
            )
            intent = begin_external_effect_intent(
                runtime,
                pid=pid,
                provider="jsonrpc",
                operation="call",
                target=resource,
                state_mutation=True,
                information_flow=True,
                metadata={"context": context},
            )
            assert intent.effect_id == binding["effect_id"]
            assert intent.canonical_args_hash == binding["canonical_args_hash"]
            with runtime.uow.transaction():
                abandon_external_effect_intent(
                    runtime.uow.protected_effects,
                    intent.effect_id,
                    operations=runtime.operations,
                )
                runtime.capability.commit_reserved_use(
                    reservation_id,
                    committed_by="test.approved_effect",
                    reason="cleanup approved effect",
                )
    finally:
        runtime.close()


class _Reconciler:
    def __init__(self) -> None:
        self.reconcile_calls: list[str] = []
        self.provider_calls = 0

    def reconcile_external_effect(self, effect):
        self.reconcile_calls.append(effect.effect_id)
        return {
            "state": "committed",
            "provider_receipt": {"provider_id": "receipt-1"},
        }

    def call(self, *_args, **_kwargs):
        self.provider_calls += 1
        raise AssertionError("reconciliation must not replay provider calls")


class _FailingReconciler:
    def reconcile_external_effect(self, _effect):
        raise RuntimeError("connector temporarily unavailable")
