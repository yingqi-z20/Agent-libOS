from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from agent_libos.models import OperationOutcome, OperationState
from agent_libos.models.exceptions import RuntimePublicationPending
from agent_libos.runtime.operation_manager import OperationManager
from agent_libos.storage.sqlite import SQLiteStore


def _pending_signal(
    operation_id: str,
    *,
    phase: str = "publishing",
) -> RuntimePublicationPending:
    return RuntimePublicationPending(
        publication_id="publication-test",
        operation_id=operation_id,
        state="applying",
        phase=phase,
    )


@pytest.mark.parametrize(
    "foreign_error_factory",
    [
        pytest.param(lambda: RuntimeError("mixed failure"), id="exception"),
        pytest.param(lambda: asyncio.CancelledError("mixed cancellation"), id="cancel"),
        pytest.param(lambda: KeyboardInterrupt("mixed interrupt"), id="keyboard-interrupt"),
    ],
)
def test_owned_pending_group_cannot_mask_an_unrelated_leaf(
    monkeypatch: pytest.MonkeyPatch,
    foreign_error_factory: Callable[[], BaseException],
) -> None:
    store = SQLiteStore(":memory:")
    operations = OperationManager(store)
    operation_id = ""
    monkeypatch.setattr(
        operations,
        "_owns_pending_runtime_publication",
        lambda _operation_id, _pending: True,
    )
    try:
        with pytest.raises(BaseExceptionGroup):
            with operations.scope(
                kind="runtime",
                name="test.grouped-pending-mixed-leaf",
                actor="test",
                pid=None,
            ) as operation:
                operation_id = operation.operation_id
                raise BaseExceptionGroup(
                    "pending signal mixed with an unrelated failure",
                    [
                        _pending_signal(operation.operation_id),
                        foreign_error_factory(),
                    ],
                )

        stored = store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.TERMINAL
        assert stored.outcome == OperationOutcome.FAILED
        assert stored.metadata["runtime_publication_mismatch"] is True
    finally:
        store.close()


def test_homogeneous_owned_pending_group_preserves_running_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    operations = OperationManager(store)
    operation_id = ""
    monkeypatch.setattr(
        operations,
        "_owns_pending_runtime_publication",
        lambda _operation_id, _pending: True,
    )
    try:
        with pytest.raises(BaseExceptionGroup):
            with operations.scope(
                kind="runtime",
                name="test.grouped-pending-homogeneous",
                actor="test",
                pid=None,
            ) as operation:
                operation_id = operation.operation_id
                raise BaseExceptionGroup(
                    "same owned pending signal repeated",
                    [
                        _pending_signal(operation.operation_id),
                        _pending_signal(operation.operation_id),
                    ],
                )

        stored = store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.RUNNING
        assert stored.outcome == OperationOutcome.PENDING
        operations.finish(
            OperationOutcome.INTERRUPTED,
            operation_id=operation_id,
        )
    finally:
        store.close()


def test_owned_pending_group_requires_one_identical_signal_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    operations = OperationManager(store)
    operation_id = ""
    monkeypatch.setattr(
        operations,
        "_owns_pending_runtime_publication",
        lambda _operation_id, _pending: True,
    )
    try:
        with pytest.raises(BaseExceptionGroup):
            with operations.scope(
                kind="runtime",
                name="test.grouped-pending-envelope-mismatch",
                actor="test",
                pid=None,
            ) as operation:
                operation_id = operation.operation_id
                raise BaseExceptionGroup(
                    "different pending signals cannot share one outcome",
                    [
                        _pending_signal(operation.operation_id),
                        _pending_signal(
                            operation.operation_id,
                            phase="different-phase",
                        ),
                    ],
                )

        stored = store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.TERMINAL
        assert stored.outcome == OperationOutcome.FAILED
        assert stored.metadata["runtime_publication_mismatch"] is True
    finally:
        store.close()
