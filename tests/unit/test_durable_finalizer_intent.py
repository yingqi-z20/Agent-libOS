from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps, loads


FINALIZER_ID = "test.durable-finalizer:v1"


class _CustomValue:
    pass


class _DuplicateKeyMapping(Mapping[str, Any]):
    """Adversarial Mapping that exposes the same JSON key twice."""

    def __getitem__(self, key: str) -> Any:
        if key != "duplicate":
            raise KeyError(key)
        return 1

    def __iter__(self) -> Iterator[str]:
        return iter(("duplicate", "duplicate"))

    def __len__(self) -> int:
        return 2


def _manager(
    prepare: Any,
    finalize: Any | None = None,
) -> ObjectMemoryManager:
    manager = ObjectMemoryManager.__new__(ObjectMemoryManager)
    manager._object_release_finalizers = []
    manager._durable_object_release_finalizers = {}
    manager.config = DEFAULT_CONFIG
    manager.audit = SimpleNamespace(record=lambda **_kwargs: None)
    manager.bind_durable_object_release_finalizer(
        FINALIZER_ID,
        prepare,
        finalize or (lambda _intent, _actor, _reason, _work_id: None),
    )
    return manager


def _object(index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(oid=f"obj-{index}", version=1)


def _prepare(
    manager: ObjectMemoryManager,
    objects: Any,
    *,
    intent_limit_bytes: int = 16_384,
    total_limit_bytes: int = 1_048_576,
) -> list[dict[str, Any]]:
    return manager.prepare_checkpoint_restore_finalizers(
        objects,
        publication_id="publication-1",
        actor="test",
        reason="checkpoint_restore",
        intent_limit_bytes=intent_limit_bytes,
        total_limit_bytes=total_limit_bytes,
    )


@pytest.mark.parametrize(
    "invalid_intent",
    [
        pytest.param({1: "integer-key"}, id="non-string-key"),
        pytest.param({"1": "string-key", 1: "colliding-key"}, id="key-collision"),
        pytest.param(_DuplicateKeyMapping(), id="duplicate-mapping-key"),
        pytest.param({"value": b"bytes"}, id="bytes"),
        pytest.param({"value": _CustomValue()}, id="custom-object"),
        pytest.param({"value": {"set-value"}}, id="set"),
        pytest.param({"value": ("tuple-value",)}, id="tuple"),
        pytest.param({"value": float("nan")}, id="nan"),
        pytest.param({"value": float("inf")}, id="positive-infinity"),
        pytest.param({"value": float("-inf")}, id="negative-infinity"),
    ],
)
def test_prepare_rejects_values_that_cannot_round_trip_as_exact_json(
    invalid_intent: Mapping[Any, Any],
) -> None:
    manager = _manager(lambda _obj, _actor, _reason, _work_id: invalid_intent)

    with pytest.raises(ValidationError):
        _prepare(manager, [_object()])


def test_prepare_rejects_cyclic_json_containers() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    manager = _manager(lambda _obj, _actor, _reason, _work_id: cyclic)

    with pytest.raises(ValidationError, match="contains a cycle"):
        _prepare(manager, [_object()])


def test_prepare_rejects_intent_before_exceeding_depth_node_or_byte_bounds() -> None:
    deep: dict[str, Any] = {}
    cursor = deep
    for _ in range(ObjectMemoryManager._BOUNDED_JSON_MAX_DEPTH + 1):
        nested: dict[str, Any] = {}
        cursor["nested"] = nested
        cursor = nested
    cases = (
        (deep, 16_384, "maximum JSON depth"),
        ({"items": [None] * 16}, 8, "maximum JSON nodes"),
        ({"padding": "x" * 64}, 32, "exceeds 32 bytes"),
    )

    for intent, limit_bytes, expected_error in cases:
        manager = _manager(lambda _obj, _actor, _reason, _work_id, value=intent: value)
        with pytest.raises(ValidationError, match=expected_error):
            _prepare(
                manager,
                [_object()],
                intent_limit_bytes=limit_bytes,
            )


def test_online_release_applies_the_configured_durable_intent_byte_bound() -> None:
    finalized: list[dict[str, Any]] = []
    manager = _manager(
        lambda _obj, _actor, _reason, _work_id: {"padding": "x" * 64},
        lambda intent, _actor, _reason, _work_id: finalized.append(intent),
    )
    manager.config = SimpleNamespace(
        checkpoint=SimpleNamespace(payload_capture_limit_bytes=32)
    )

    with pytest.raises(ValidationError, match="exceeds 32 bytes"):
        manager.run_object_release_finalizers_trusted(
            _object(),  # type: ignore[arg-type]
            actor="test.host",
            reason="bounded online release",
        )

    assert finalized == []


def test_mapping_intent_round_trips_without_changing_authenticated_value() -> None:
    nested_values: list[Any] = [None, True, 7, 1.25, "resource"]
    raw_intent = MappingProxyType(
        {
            "provider": MappingProxyType({"resource_id": "remote-1"}),
            "values": nested_values,
        }
    )
    finalized: list[dict[str, Any]] = []

    def finalize(
        intent: Mapping[str, Any],
        _actor: str,
        _reason: str,
        _work_id: str,
    ) -> None:
        finalized.append(deepcopy(dict(intent)))
        # The callback must not be able to mutate the persisted work item.
        intent["provider"]["resource_id"] = "callback-mutated"  # type: ignore[index]

    manager = _manager(
        lambda _obj, _actor, _reason, _work_id: raw_intent,
        finalize,
    )
    work_item = _prepare(manager, [_object()])[0]
    expected_intent = {
        "provider": {"resource_id": "remote-1"},
        "values": [None, True, 7, 1.25, "resource"],
    }

    assert work_item["intent"] == expected_intent
    nested_values.append("prepare-result-mutated")
    assert work_item["intent"] == expected_intent

    persisted_work_item = loads(dumps(work_item))
    manager.run_checkpoint_restore_finalizer(
        persisted_work_item,
        actor="test",
        reason="checkpoint_restore",
    )

    assert finalized == [expected_intent]
    assert persisted_work_item["intent"] == expected_intent


def test_total_budget_stops_before_materializing_all_objects_and_work() -> None:
    prepared_oids: list[str] = []

    def prepare(obj: SimpleNamespace, _actor: str, _reason: str, _work_id: str) -> dict[str, Any]:
        prepared_oids.append(obj.oid)
        return {"resource_id": obj.oid, "padding": "x" * 64}

    manager = _manager(prepare)
    probe = _prepare(manager, [_object(0)])
    one_item_limit = len(manager._canonical_json_bytes(probe))
    prepared_oids.clear()
    yielded_oids: list[str] = []

    def objects() -> Iterator[SimpleNamespace]:
        for index in range(100):
            obj = _object(index)
            yielded_oids.append(obj.oid)
            yield obj

    with pytest.raises(
        ValidationError,
        match="checkpoint restore durable finalizer work exceeds",
    ):
        _prepare(
            manager,
            objects(),
            total_limit_bytes=one_item_limit,
        )

    assert yielded_oids == ["obj-0", "obj-1"]
    assert prepared_oids == ["obj-0", "obj-1"]
