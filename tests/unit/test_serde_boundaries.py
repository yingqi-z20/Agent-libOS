from __future__ import annotations

import pytest

from agent_libos.utils import serde
from agent_libos.utils.serde import bounded_json_loads, dumps


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_dumps_rejects_nested_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        dumps({"outer": [1.25, {"value": value}]})


def test_bounded_json_loads_rejects_excessive_nodes() -> None:
    payload = '{"values":[' + ",".join("0" for _ in range(100_001)) + "]}"

    with pytest.raises(ValueError, match="JSON document exceeds maximum nodes=100000"):
        bounded_json_loads(payload)


def test_bounded_json_and_dumps_preserve_nested_finite_values() -> None:
    payload = '{"outer":[1.25,{"minimum":-4,"enabled":true,"none":null}]}'

    decoded = bounded_json_loads(payload)

    assert decoded == {
        "outer": [1.25, {"minimum": -4, "enabled": True, "none": None}]
    }
    assert "NaN" not in dumps(decoded)
    assert "Infinity" not in dumps(decoded)


@pytest.mark.parametrize("operation", ["load", "dump"])
def test_shared_json_helpers_do_not_swallow_memory_error(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    if operation == "load":
        monkeypatch.setattr(serde.json, "loads", lambda *_args, **_kwargs: _raise_memory())
        invoke = lambda: bounded_json_loads("{}")
    else:
        monkeypatch.setattr(serde.json, "dumps", lambda *_args, **_kwargs: _raise_memory())
        invoke = lambda: dumps({})

    with pytest.raises(MemoryError, match="allocation failed"):
        invoke()


def _raise_memory() -> None:
    raise MemoryError("allocation failed")
