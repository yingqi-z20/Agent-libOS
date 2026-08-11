from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.api.cli import (
    _run_jsonrpc_command,
    _run_management_cli_command,
    _run_mcp_command,
)


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


def test_mcp_arguments_json_applies_hard_limit_before_dispatch() -> None:
    mcp = _RecordingMcp()
    runtime = SimpleNamespace(
        mcp=mcp,
        config=SimpleNamespace(
            mcp=SimpleNamespace(max_request_hard_limit_bytes=16),
        ),
    )

    with pytest.raises(SystemExit, match="max_bytes=16"):
        _run_mcp_command(
            runtime,
            _mcp_call_args('{"payload":"too large"}'),
        )

    assert mcp.calls == []


def test_mcp_list_preserves_bounded_window_truncation_signal() -> None:
    observed: list[dict[str, Any]] = []

    class RecordingMcp:
        @staticmethod
        def list_servers_window(**kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
            observed.append(kwargs)
            return ([{"server_id": "server-1"}], True)

    result = _run_mcp_command(
        SimpleNamespace(mcp=RecordingMcp()),
        SimpleNamespace(
            actor_pid=None,
            mcp_command="list",
            text="server",
            limit=1,
        ),
    )

    assert result == {
        "servers": [{"server_id": "server-1"}],
        "has_more": True,
    }
    assert observed == [
        {
            "actor": None,
            "require_capability": False,
            "text": "server",
            "limit": 1,
        }
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"123456789", "exceeds max_bytes=8"),
        (b"\xff", "must be valid UTF-8"),
    ],
)
def test_mcp_host_register_bounds_and_decodes_manifest_before_dispatch(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    manifest = tmp_path / "mcp.yaml"
    manifest.write_bytes(payload)
    calls: list[str] = []

    class RecordingMcp:
        @staticmethod
        def register_server_from_yaml_text(text: str, **_kwargs: Any) -> dict[str, Any]:
            calls.append(text)
            return {"server_id": "server-1"}

    runtime = SimpleNamespace(
        mcp=RecordingMcp(),
        config=SimpleNamespace(mcp=SimpleNamespace(manifest_max_bytes=8)),
    )
    args = SimpleNamespace(
        actor_pid=None,
        mcp_command="register",
        path=str(manifest),
        replace=False,
    )

    with pytest.raises(SystemExit, match=message):
        _run_mcp_command(runtime, args)

    assert calls == []


def test_mcp_host_register_accepts_manifest_at_exact_byte_limit(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "mcp.yaml"
    manifest.write_bytes("éééé".encode("utf-8"))
    calls: list[str] = []

    class RecordingMcp:
        @staticmethod
        def register_server_from_yaml_text(text: str, **_kwargs: Any) -> dict[str, Any]:
            calls.append(text)
            return {"server_id": "server-1"}

    result = _run_mcp_command(
        SimpleNamespace(
            mcp=RecordingMcp(),
            config=SimpleNamespace(mcp=SimpleNamespace(manifest_max_bytes=8)),
        ),
        SimpleNamespace(
            actor_pid=None,
            mcp_command="register",
            path=str(manifest),
            replace=False,
        ),
    )

    assert result == {"server_id": "server-1"}
    assert calls == ["éééé"]


def test_mcp_failed_call_prints_result_once_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str, str, dict[str, Any]]] = []
    failure = {
        "server_id": "server-1",
        "tool_id": "tool-1",
        "status": "mcp_error",
        "ok": False,
        "error": {"code": -32603, "message": "provider rejected call"},
    }

    class FailingMcp:
        @staticmethod
        def call_tool(
            pid: str,
            server_id: str,
            tool_id: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            calls.append((pid, server_id, tool_id, arguments))
            return failure

    args = _mcp_call_args("{}")
    args.command = "mcp"

    with pytest.raises(SystemExit) as exc_info:
        _run_management_cli_command(SimpleNamespace(mcp=FailingMcp()), args)

    assert exc_info.value.code == 1
    assert json.loads(capsys.readouterr().out) == failure
    assert calls == [("pid-1", "server-1", "tool-1", {})]
