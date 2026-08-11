from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.api.cli import _run_jsonrpc_command, _run_mcp_command


class _RecordingJsonRpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, Any]] = []

    def call(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        params: Any,
    ) -> dict[str, Any]:
        self.calls.append((pid, endpoint_id, method_id, params))
        return {"params": params}


class _RecordingMcp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    def call_tool(
        self,
        pid: str,
        server_id: str,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((pid, server_id, tool_id, arguments))
        return {"arguments": arguments}


def _jsonrpc_call_args(params_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        actor_pid=None,
        jsonrpc_command="call",
        pid="pid-1",
        endpoint_id="endpoint-1",
        method_id="method-1",
        params_json=params_json,
    )


def _mcp_call_args(arguments_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        actor_pid=None,
        mcp_command="call",
        pid="pid-1",
        server_id="server-1",
        tool_id="tool-1",
        arguments_json=arguments_json,
    )


@pytest.mark.parametrize(
    "params_json",
    [
        "plain text",
        '{"unterminated":',
        '{"duplicate": 1, "duplicate": 2}',
        '{"number": NaN}',
    ],
)
def test_jsonrpc_params_json_rejects_non_strict_json_before_dispatch(
    params_json: str,
) -> None:
    jsonrpc = _RecordingJsonRpc()

    with pytest.raises(SystemExit, match="--params-json must be valid JSON"):
        _run_jsonrpc_command(
            SimpleNamespace(jsonrpc=jsonrpc),
            _jsonrpc_call_args(params_json),
        )

    assert jsonrpc.calls == []


def test_jsonrpc_params_json_preserves_valid_array_value() -> None:
    jsonrpc = _RecordingJsonRpc()
    params = [1, {"enabled": True}]

    result = _run_jsonrpc_command(
        SimpleNamespace(jsonrpc=jsonrpc),
        _jsonrpc_call_args('[1, {"enabled": true}]'),
    )

    assert result == {"params": params}
    assert jsonrpc.calls == [("pid-1", "endpoint-1", "method-1", params)]


@pytest.mark.parametrize("params_json", ["true", "7", '"scalar"'])
def test_jsonrpc_params_json_rejects_scalar_before_dispatch(
    params_json: str,
) -> None:
    jsonrpc = _RecordingJsonRpc()

    with pytest.raises(
        SystemExit,
        match="--params-json must be a JSON object, array, or null",
    ):
        _run_jsonrpc_command(
            SimpleNamespace(jsonrpc=jsonrpc),
            _jsonrpc_call_args(params_json),
        )

    assert jsonrpc.calls == []


@pytest.mark.parametrize(
    "arguments_json",
    [
        "plain text",
        '{"unterminated":',
        '{"duplicate": 1, "duplicate": 2}',
        '{"number": Infinity}',
    ],
)
def test_mcp_arguments_json_rejects_non_strict_json_before_dispatch(
    arguments_json: str,
) -> None:
    mcp = _RecordingMcp()

    with pytest.raises(SystemExit, match="--arguments-json must be valid JSON"):
        _run_mcp_command(
            SimpleNamespace(mcp=mcp),
            _mcp_call_args(arguments_json),
        )

    assert mcp.calls == []


@pytest.mark.parametrize("arguments_json", ["null", "[]", '"text"', "1", "true"])
def test_mcp_arguments_json_requires_an_object_before_dispatch(
    arguments_json: str,
) -> None:
    mcp = _RecordingMcp()

    with pytest.raises(SystemExit, match="--arguments-json must be a JSON object"):
        _run_mcp_command(
            SimpleNamespace(mcp=mcp),
            _mcp_call_args(arguments_json),
        )

    assert mcp.calls == []


def test_mcp_arguments_json_forwards_a_strict_object() -> None:
    mcp = _RecordingMcp()
    arguments = {"city": "Beijing", "units": ["metric"]}

    result = _run_mcp_command(
        SimpleNamespace(mcp=mcp),
        _mcp_call_args('{"city": "Beijing", "units": ["metric"]}'),
    )

    assert result == {"arguments": arguments}
    assert mcp.calls == [("pid-1", "server-1", "tool-1", arguments)]
