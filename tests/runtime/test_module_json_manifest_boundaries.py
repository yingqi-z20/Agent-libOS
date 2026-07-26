from __future__ import annotations

import json
import sys
from typing import Any

import pytest

import agent_libos.modules.loader as module_loader
from agent_libos.models.exceptions import ValidationError
from agent_libos.modules.loader import ModuleLoader


def _manifest(metadata: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "module_id": "json-boundary:v0",
        "name": "JSON boundary",
        "entrypoint": "./module.py:register_module",
        "provides": {},
        "sha256": "0" * 64,
        "metadata": metadata,
    }


def test_module_json_manifest_rejects_depth_before_recursive_decode_failure() -> None:
    nested: Any = 0
    for _ in range(500):
        nested = {"value": nested}

    with pytest.raises(
        ValidationError,
        match="invalid module manifest JSON: JSON nesting exceeds maximum depth=256",
    ):
        ModuleLoader().parse_manifest(json.dumps(_manifest(nested)))


def test_module_json_manifest_rejects_oversized_integer_with_interpreter_guard_disabled() -> None:
    prefix = json.dumps(_manifest({}))[:-2]
    payload = prefix + '"large":' + ("9" * 100_000) + "}}"
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(
            ValidationError,
            match="invalid module manifest JSON: JSON integer exceeds maximum digits=4300",
        ):
            ModuleLoader().parse_manifest(payload)
    finally:
        sys.set_int_max_str_digits(previous)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_module_json_manifest_rejects_nested_nonfinite_metadata(value: float) -> None:
    payload = json.dumps(_manifest({"outer": [1.25, {"value": value}]}))

    with pytest.raises(
        ValidationError,
        match="invalid module manifest JSON: JSON numbers must be finite",
    ):
        ModuleLoader().parse_manifest(payload)


def test_module_json_manifest_rejects_excessive_nodes() -> None:
    prefix = json.dumps(_manifest({}))[:-2]
    payload = prefix + '"values":[' + ",".join("0" for _ in range(100_001)) + "]}}"

    with pytest.raises(
        ValidationError,
        match="invalid module manifest JSON: JSON document exceeds maximum nodes=100000",
    ):
        ModuleLoader().parse_manifest(payload)


def test_module_json_manifest_preserves_duplicate_key_rejection() -> None:
    payload = json.dumps(_manifest({})).replace(
        '"name": "JSON boundary"',
        '"name": "first", "name": "second"',
    )

    with pytest.raises(ValidationError, match="duplicate module manifest JSON key: 'name'"):
        ModuleLoader().parse_manifest(payload)


def test_module_json_manifest_preserves_nested_finite_metadata() -> None:
    finite = {"outer": [1.25, {"minimum": -4, "enabled": True, "none": None}]}

    parsed = ModuleLoader().parse_manifest(json.dumps(_manifest(finite)))

    assert parsed.metadata == finite


@pytest.mark.parametrize("error", [MemoryError("allocation failed"), KeyboardInterrupt()])
def test_module_json_preflight_does_not_swallow_control_or_memory_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def fail(_text: str, **_options: object) -> None:
        raise error

    monkeypatch.setattr(module_loader, "bounded_json_loads", fail)

    with pytest.raises(type(error)):
        ModuleLoader().parse_manifest(json.dumps(_manifest({})))
