from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent_libos.models import OperationOutcome, OperationState
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.operation_manager import OperationManager
from agent_libos.storage.sqlite import SQLiteStore


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


class _ListSubclass(list[object]):
    pass


class _DictSubclass(dict[str, object]):
    pass


def _start_operation(manager: OperationManager, **kwargs: Any):
    return manager.start(
        kind="runtime",
        name="test.validation",
        actor="test",
        pid=None,
        **kwargs,
    )


def _invalid_operation_ids() -> tuple[object, ...]:
    return (
        "",
        " ",
        "\top_1",
        0,
        1,
        False,
        True,
        _StringSubclass("op_1"),
    )


def _invalid_metadata_values() -> list[tuple[str, object]]:
    cycle: dict[str, object] = {}
    cycle["cycle"] = cycle
    shared: list[object] = []
    excessive_depth: object = None
    for _ in range(32):
        excessive_depth = [excessive_depth]
    return [
        ("false", False),
        ("zero", 0),
        ("empty-string", ""),
        ("empty-list", []),
        ("empty-tuple", ()),
        ("integer-key", {1: "value"}),
        ("string-subclass-key", {_StringSubclass("key"): "value"}),
        ("set-value", {"value": set()}),
        ("tuple-value", {"value": ()}),
        ("bytes-value", {"value": b"bytes"}),
        ("integer-subclass-value", {"value": _IntegerSubclass(1)}),
        ("string-subclass-value", {"value": _StringSubclass("value")}),
        ("list-subclass-value", {"value": _ListSubclass()}),
        ("dict-subclass-value", {"value": _DictSubclass()}),
        ("nan", {"value": float("nan")}),
        ("positive-infinity", {"value": float("inf")}),
        ("negative-infinity", {"value": float("-inf")}),
        ("invalid-utf8-key", {"\ud800": "value"}),
        ("invalid-utf8-value", {"value": "\ud800"}),
        ("cycle", cycle),
        ("alias", {"left": shared, "right": shared}),
        ("excessive-depth", {"value": excessive_depth}),
        ("excessive-nodes", {"value": [None] * 4_095}),
        ("excessive-bytes", {"value": "x" * 131_072}),
        ("excessive-integer", {"value": 10**5_000}),
        ("reserved-key", {"runtime_publication_id": "publication_1"}),
    ]


def test_explicit_invalid_optional_operation_ids_never_use_ambient_operation() -> None:
    store = SQLiteStore(":memory:")
    manager = OperationManager(store)
    operation = _start_operation(manager)
    baseline = manager.get_operation(operation.operation_id)
    assert baseline is not None

    try:
        with manager.attach(operation.operation_id):
            for invalid_id in _invalid_operation_ids():
                calls: tuple[Callable[[], object], ...] = (
                    lambda invalid_id=invalid_id: manager.expect(
                        "audit",
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                    lambda invalid_id=invalid_id: manager.merge_metadata(
                        {"changed": True},
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                    lambda invalid_id=invalid_id: manager.set_pid(
                        "changed",
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                    lambda invalid_id=invalid_id: manager.finish(
                        OperationOutcome.SUCCEEDED,
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                    lambda invalid_id=invalid_id: manager.wait(
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                    lambda invalid_id=invalid_id: manager.link_evidence(
                        "audit",
                        "audit_1",
                        "audit",
                        operation_id=invalid_id,  # type: ignore[arg-type]
                    ),
                )
                for call in calls:
                    with pytest.raises(
                        ValidationError,
                        match="operation id must be an exact non-empty string",
                    ):
                        call()
                with pytest.raises(
                    ValidationError,
                    match="operation id must be an exact non-empty string",
                ):
                    _start_operation(
                        manager,
                        parent_operation_id=invalid_id,  # type: ignore[arg-type]
                    )
                assert manager.current_id() == operation.operation_id

        assert manager.get_operation(operation.operation_id) == baseline
        assert store.list_operations() == [baseline]
        assert store.list_operation_evidence() == []
    finally:
        store.close()


def test_direct_operation_id_boundaries_reject_non_exact_ids_before_reads() -> None:
    store = SQLiteStore(":memory:")
    manager = OperationManager(store)
    try:
        for invalid_id in _invalid_operation_ids():
            with pytest.raises(ValidationError):
                manager.get_operation(invalid_id)  # type: ignore[arg-type]
            with pytest.raises(ValidationError):
                manager.resume(invalid_id)  # type: ignore[arg-type]
            with pytest.raises(ValidationError):
                with manager.attach(invalid_id):  # type: ignore[arg-type]
                    raise AssertionError("invalid operation must not attach")
            with pytest.raises(ValidationError):
                manager.bind_runtime_publication(
                    invalid_id,  # type: ignore[arg-type]
                    publication_id="publication_1",
                    publication_kind="test",
                    expected_kind="runtime",
                    expected_name="test.validation",
                    expected_actor="test",
                    expected_pid=None,
                )
            with pytest.raises(ValidationError):
                manager.reconcile_runtime_publication(
                    invalid_id,  # type: ignore[arg-type]
                    OperationOutcome.SUCCEEDED,
                    publication_id="publication_1",
                    publication_kind="test",
                    publication_state="committed",
                    publication_phase="committed",
                    expected_kind="runtime",
                    expected_name="test.validation",
                    expected_actor="test",
                    expected_pid=None,
                )
        assert store.list_operations() == []
        assert store.list_operation_evidence() == []
    finally:
        store.close()


def test_operation_metadata_rejects_non_exact_or_unbounded_json_before_insert() -> None:
    store = SQLiteStore(":memory:")
    manager = OperationManager(store)
    try:
        for case, invalid_metadata in _invalid_metadata_values():
            with pytest.raises(ValidationError):
                _start_operation(
                    manager,
                    metadata=invalid_metadata,  # type: ignore[arg-type]
                )
            assert store.list_operations() == [], case
            assert store.list_operation_evidence() == [], case
    finally:
        store.close()


def test_operation_metadata_mutators_reject_before_state_or_evidence_changes() -> None:
    store = SQLiteStore(":memory:")
    manager = OperationManager(store)
    operation = _start_operation(manager, metadata={"stable": True})
    baseline = manager.get_operation(operation.operation_id)
    assert baseline is not None
    shared: list[object] = []

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: manager.merge_metadata(  # type: ignore[arg-type]
            False,
            operation_id=operation.operation_id,
        ),
        lambda: manager.wait(
            operation_id=operation.operation_id,
            metadata=[],  # type: ignore[arg-type]
        ),
        lambda: manager.finish(
            OperationOutcome.SUCCEEDED,
            operation_id=operation.operation_id,
            metadata={"value": float("inf")},
        ),
        lambda: manager.link_evidence(
            "audit",
            "audit_1",
            "audit",
            operation_id=operation.operation_id,
            metadata={"left": shared, "right": shared},
        ),
        lambda: manager.merge_metadata(
            {"runtime_publication_id": "publication_1"},
            operation_id=operation.operation_id,
        ),
    )
    try:
        for invalid_call in invalid_calls:
            with pytest.raises(ValidationError):
                invalid_call()
            assert manager.get_operation(operation.operation_id) == baseline
            assert store.list_operation_evidence() == []
        latest = manager.get_operation(operation.operation_id)
        assert latest is not None
        assert latest.state is OperationState.RUNNING
        assert latest.outcome is OperationOutcome.PENDING
    finally:
        store.close()


def test_operation_metadata_is_detached_and_evidence_allows_non_binding_prefix() -> None:
    store = SQLiteStore(":memory:")
    manager = OperationManager(store)
    operation_metadata = {
        "nested": {"items": [1, 1.5, "value", True, None]},
    }
    evidence_metadata = {
        "runtime_publication_note": {"items": ["diagnostic"]},
    }
    try:
        operation = _start_operation(manager, metadata=operation_metadata)
        link = manager.link_evidence(
            "audit",
            "audit_1",
            "audit",
            operation_id=operation.operation_id,
            metadata=evidence_metadata,
        )
        assert link is not None
        operation_metadata["nested"]["items"].append("mutated")  # type: ignore[index,union-attr]
        evidence_metadata["runtime_publication_note"]["items"].append(  # type: ignore[index,union-attr]
            "mutated"
        )

        stored = manager.get_operation(operation.operation_id)
        assert stored is not None
        assert stored.metadata == {
            "nested": {"items": [1, 1.5, "value", True, None]},
        }
        assert link.metadata == {
            "runtime_publication_note": {"items": ["diagnostic"]},
        }
    finally:
        store.close()
