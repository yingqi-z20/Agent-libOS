from __future__ import annotations

import copy
from collections.abc import Callable
import sys
from typing import Any

import pytest
import yaml

from agent_libos import Runtime
from agent_libos.models.exceptions import ValidationError


def _mcp_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server_id": "numeric-boundary",
        "transport": "stdio",
        "stdio": {"command": sys.executable},
        "tools": [
            {
                "tool_id": "echo",
                "mcp_name": "demo.echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {"type": "object"},
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _jsonrpc_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "endpoint_id": "numeric-boundary",
        "url": "https://api.example.test/jsonrpc",
        "headers": {},
        "methods": [
            {
                "method_id": "echo",
                "rpc_method": "demo.echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "params_schema": {"type": "object"},
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _set_manifest_field(
    manifest: dict[str, Any],
    *,
    adapter: str,
    field: str,
    value: Any,
) -> None:
    if field == "metadata":
        manifest["metadata"] = value
    elif adapter == "mcp" and field == "child metadata":
        manifest["tools"][0]["metadata"] = value
    elif adapter == "jsonrpc" and field == "child metadata":
        manifest["methods"][0]["metadata"] = value
    elif adapter == "mcp" and field == "schema":
        manifest["tools"][0]["input_schema"] = value
    elif adapter == "jsonrpc" and field == "schema":
        manifest["methods"][0]["params_schema"] = value
    else:  # pragma: no cover - test matrix owns every supported pair
        raise AssertionError((adapter, field))


def _register_mapping(
    runtime: Runtime,
    *,
    adapter: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if adapter == "mcp":
        return runtime.mcp.register_server(
            manifest,
            actor="cli",
            require_capability=False,
        )
    return runtime.jsonrpc.register_endpoint(
        manifest,
        actor="cli",
        require_capability=False,
    )


def _register_yaml(
    runtime: Runtime,
    *,
    adapter: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    text = yaml.safe_dump(manifest, sort_keys=False)
    if adapter == "mcp":
        return runtime.mcp.register_server_from_yaml_text(
            text,
            actor="cli",
            require_capability=False,
        )
    return runtime.jsonrpc.register_endpoint_from_yaml_text(
        text,
        actor="cli",
        require_capability=False,
    )


@pytest.mark.parametrize(
    "adapter",
    ["mcp", "jsonrpc"],
    ids=["tool-protocol", "rpc-protocol"],
)
@pytest.mark.parametrize("field", ["metadata", "child metadata", "schema"])
@pytest.mark.parametrize("register", [_register_mapping, _register_yaml], ids=["mapping", "yaml"])
def test_registry_manifests_reject_nested_nonfinite_json_values_without_mutation(
    adapter: str,
    field: str,
    register: Callable[..., dict[str, Any]],
) -> None:
    manifest = _mcp_manifest() if adapter == "mcp" else _jsonrpc_manifest()
    _set_manifest_field(
        manifest,
        adapter=adapter,
        field=field,
        value={"outer": [1.25, {"nan": float("nan"), "inf": float("inf")}]},
    )
    expected = (
        f"MCP {'tool ' if field == 'child metadata' else ''}"
        f"{'input_schema' if field == 'schema' else 'metadata'} must be JSON-serializable"
        if adapter == "mcp"
        else f"JSON-RPC {'method ' if field != 'metadata' else ''}"
        f"{'params_schema' if field == 'schema' else 'metadata'} must be JSON-serializable"
    )

    runtime = Runtime.open(":memory:")
    try:
        with pytest.raises(ValidationError, match=f"^{expected}$"):
            register(runtime, adapter=adapter, manifest=manifest)
        if adapter == "mcp":
            assert runtime.store.list_mcp_servers() == []
        else:
            assert runtime.store.list_jsonrpc_endpoints() == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "adapter",
    ["mcp", "jsonrpc"],
    ids=["tool-protocol", "rpc-protocol"],
)
def test_registry_manifests_preserve_nested_finite_json_metadata(adapter: str) -> None:
    manifest = _mcp_manifest() if adapter == "mcp" else _jsonrpc_manifest()
    finite = {"outer": [1.25, {"minimum": -4, "enabled": True, "none": None}]}
    manifest["metadata"] = copy.deepcopy(finite)
    _set_manifest_field(
        manifest,
        adapter=adapter,
        field="child metadata",
        value=copy.deepcopy(finite),
    )

    runtime = Runtime.open(":memory:")
    try:
        registered = _register_yaml(runtime, adapter=adapter, manifest=manifest)
        assert registered["metadata"] == finite
        child = registered["tools"][0] if adapter == "mcp" else registered["methods"][0]
        assert child["metadata"] == finite
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "adapter",
    ["mcp", "jsonrpc"],
    ids=["tool-protocol", "rpc-protocol"],
)
@pytest.mark.parametrize("error", [MemoryError("allocation failed"), KeyboardInterrupt()])
def test_registry_json_validation_does_not_swallow_control_or_memory_failures(
    monkeypatch: pytest.MonkeyPatch,
    adapter: str,
    error: BaseException,
) -> None:
    runtime = Runtime.open(":memory:")
    primitive = runtime.mcp if adapter == "mcp" else runtime.jsonrpc
    module = (
        "agent_libos.primitives.mcp.json.dumps"
        if adapter == "mcp"
        else "agent_libos.primitives.jsonrpc.json.dumps"
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, fail)
            with pytest.raises(type(error)):
                primitive._validate_json_value({}, "metadata")
    finally:
        runtime.close()
