from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpProviderCallResult,
    McpProviderTool,
    McpToolListResult,
)
from agent_libos.models.exceptions import ProviderHostError
from agent_libos.storage import open_store
from agent_libos.utils.serde import dumps, to_jsonable


@pytest.fixture(
    params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)],
    name="backend_runtime",
)
def _backend_runtime(request: pytest.FixtureRequest) -> Iterator[Runtime]:
    with _runtime_for_backend(str(request.param)) as selected:
        yield selected


@pytest.mark.parametrize("failure_stage", ["list", "call"])
def test_legacy_unknown_stage_settles_only_current_response_envelope(
    backend_runtime: Runtime,
    failure_stage: str,
) -> None:
    runtime = backend_runtime
    runtime.mcp.provider = _StageFailingMcpProvider(failure_stage)
    pid = runtime.process.spawn(image="base-agent:v0", goal=f"MCP {failure_stage} settlement")
    runtime.mcp.register_server_from_yaml_text(
        _stdio_manifest("stage-settlement"),
        actor="cli",
        require_capability=False,
    )
    runtime.capability.grant(
        pid,
        "mcp:stage-settlement:echo",
        [CapabilityRight.READ],
        issued_by="test",
    )
    _grant_stdio_spawn(runtime, pid)

    if failure_stage == "list":
        secret = runtime.mcp.provider.secret
        with pytest.raises(ProviderHostError) as raised:
            runtime.mcp.call_tool(
                pid,
                "stage-settlement",
                "echo",
                {"text": "hello"},
            )
        assert secret not in str(raised.value)
        assert raised.value.code == "mcp_provider_error"
        assert raised.value.error_type == "RuntimeError"
        assert raised.value.correlation_id
        effect = runtime.store.list_external_effects(pid=pid)[0]
        assert effect.provider_metadata["result"]["error"] == raised.value.to_dict()
        assert secret not in dumps(effect.provider_metadata)
        expected_response_bytes = runtime.config.mcp.max_response_bytes
    else:
        result = runtime.mcp.call_tool(
            pid,
            "stage-settlement",
            "echo",
            {"text": "hello"},
        )
        assert not result.ok
        expected_response_bytes = (
            runtime.mcp.provider.list_response_bytes
            + runtime.config.mcp.max_response_bytes
        )

    reservation = runtime.store.list_resource_usage_reservations(pid=pid)[0]
    assert reservation["status"] == "settled"
    assert reservation["settled_usage"].mcp_response_bytes == expected_response_bytes
    assert runtime.process.get(pid).resource_usage.mcp_response_bytes == expected_response_bytes


def test_static_provider_error_envelope_is_durable(
    backend_runtime: Runtime,
) -> None:
    secret = "MCP_RELEASE_BOUNDARY_PROVIDER_SECRET"
    runtime = backend_runtime
    runtime.mcp.provider = _StageFailingMcpProvider("list", secret=secret)
    pid = runtime.process.spawn(image="base-agent:v0", goal="MCP durable envelope")
    runtime.mcp.register_server_from_yaml_text(
        _stdio_manifest("durable-envelope"),
        actor="cli",
        require_capability=False,
    )
    runtime.capability.grant(
        pid,
        "mcp_server:durable-envelope",
        [CapabilityRight.READ, CapabilityRight.EXECUTE],
        issued_by="test",
    )
    _grant_stdio_spawn(runtime, pid)
    runtime.tools.configure_process_tools(pid, ["list_mcp_tools"], assigned_by="test")

    result = runtime.tools.call(
        pid,
        "list_mcp_tools",
        {"server_id": "durable-envelope", "refresh": True},
    )

    assert not result.ok
    assert result.result_handle is not None
    durable = runtime.store.get_object(result.result_handle.oid)
    serialized = dumps({"result": to_jsonable(result), "durable": to_jsonable(durable)})
    assert secret not in serialized
    public_error = durable.payload["failure"]["error"]
    assert result.payload["error"]["details"] == {
        key: public_error[key]
        for key in ("code", "error_type", "correlation_id")
    }
    assert public_error["code"] == "mcp_provider_error"
    assert public_error["error_type"] == "RuntimeError"
    assert public_error["correlation_id"]


@contextlib.contextmanager
def _runtime_for_backend(backend: str) -> Iterator[Runtime]:
    if backend == "sqlite":
        runtime = Runtime.open("local")
        try:
            yield runtime
        finally:
            runtime.close()
        return
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        runtime = Runtime(open_store(dsn, config=config), config=config)
        try:
            yield runtime
        finally:
            runtime.close()


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_mcp_boundary_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )


def _grant_stdio_spawn(runtime: Runtime, pid: str) -> None:
    args = ["-m", "demo_server"]
    runtime.capability.grant(pid, "process:spawn", [CapabilityRight.WRITE], issued_by="test")
    runtime.capability.grant(
        pid,
        runtime.mcp.stdio_resource_for_argv("python3", args),
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )


def _stdio_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: python3
  args: ["-m", "demo_server"]
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    input_schema:
      type: object
      properties:
        text:
          type: string
      additionalProperties: false
""".strip()


class _StageFailingMcpProvider:
    def __init__(self, failure_stage: str, *, secret: str = "provider-secret") -> None:
        self.failure_stage = failure_stage
        self.secret = secret
        self.list_response_bytes = 0

    def list_tools(self, server: object, **_kwargs: object) -> McpToolListResult:
        if self.failure_stage == "list":
            raise RuntimeError(self.secret)
        tools = [
            McpProviderTool(
                name="demo.echo",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ]
        self.list_response_bytes = len(
            dumps([to_jsonable(tool) for tool in tools]).encode("utf-8")
        )
        return McpToolListResult(
            server_id=str(getattr(server, "server_id")),
            tools=tools,
            response_bytes=self.list_response_bytes,
            duration_s=0.01,
        )

    def call_tool(
        self,
        _server: object,
        _tool: object,
        _arguments: dict[str, object],
        **_kwargs: object,
    ) -> McpProviderCallResult:
        raise RuntimeError(self.secret)

    def classify_external_effect(
        self,
        _operation: str,
        _context: dict[str, object],
        _result: object,
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
        )
