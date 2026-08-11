from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    McpProviderCallResult,
    McpProviderTool,
    McpToolListResult,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.primitives.mcp import McpPrimitive
from agent_libos.utils.serde import dumps, to_jsonable
from tests.support.mcp import MCP_TEST_STDIO_COMMAND


def test_provider_capability_names_use_their_dedicated_limit() -> None:
    primitive = object.__new__(McpPrimitive)
    primitive.config = replace(
        DEFAULT_CONFIG,
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            list_limit=100,
            provider_capability_limit=2,
        ),
    )

    assert primitive._validated_connection_names(
        ("tools", "resources"),
        field="capabilities",
    ) == ("tools", "resources")
    with pytest.raises(TypeError, match="capabilities is invalid"):
        primitive._validated_connection_names(
            ("tools", "resources", "prompts"),
            field="capabilities",
        )


def _provider_tool_list_bytes(tools: list[McpProviderTool]) -> int:
    return len(dumps([to_jsonable(tool) for tool in tools]).encode("utf-8"))


def _provider_call_bytes(content: Any, structured_content: Any) -> int:
    return len(
        dumps(
            {
                "content": content,
                "structured_content": structured_content,
            }
        ).encode("utf-8")
    )


class _SecurityBoundaryMcpProvider:
    supports_subprocess_limits = True

    def __init__(
        self,
        *,
        live_schema: dict[str, Any] | None = None,
        reflected_secret: bool = False,
    ) -> None:
        self.live_schema = live_schema or {}
        self.reflected_secret = reflected_secret
        self.list_calls = 0
        self.call_argument_objects: list[dict[str, Any]] = []

    @staticmethod
    def _secret(runtime_environment: Any) -> str:
        if runtime_environment is None:
            return ""
        return str(runtime_environment.get("MCP_TEST_TOKEN", ""))

    def list_tools(
        self,
        server: Any,
        *,
        runtime_environment: Any = None,
        **_kwargs: Any,
    ) -> McpToolListResult:
        self.list_calls += 1
        secret = self._secret(runtime_environment) if self.reflected_secret else ""
        schema = self.live_schema
        if secret:
            schema = {
                **schema,
                "description": f"provider schema reflected {secret}",
            }
        tools = [
            McpProviderTool(
                name="demo.echo",
                description=(
                    f"provider description reflected {secret}"
                    if secret
                    else "Echo"
                ),
                input_schema=schema,
                metadata=(
                    {f"provider-key-{secret}": secret}
                    if secret
                    else {}
                ),
            )
        ]
        return McpToolListResult(
            server_id=server.server_id,
            tools=tools,
            response_bytes=_provider_tool_list_bytes(tools),
            duration_s=0.01,
        )

    def call_tool(
        self,
        _server: Any,
        _tool: Any,
        arguments: dict[str, Any],
        *,
        runtime_environment: Any = None,
        **_kwargs: Any,
    ) -> McpProviderCallResult:
        self.call_argument_objects.append(arguments)
        secret = self._secret(runtime_environment) if self.reflected_secret else ""
        if secret:
            content: Any = [
                {
                    "type": "text",
                    "text": f"provider content reflected {secret}",
                }
            ]
            structured_content: Any = {
                f"provider-key-{secret}": {
                    "opaque": secret,
                    "composed": f"prefix/{secret}/suffix",
                }
            }
        else:
            content = [{"type": "text", "text": "ok"}]
            structured_content = {"echo": arguments}
        is_error = arguments.get("fail") is True
        return McpProviderCallResult(
            content=content,
            structured_content=structured_content,
            is_error=is_error,
            response_bytes=_provider_call_bytes(content, structured_content),
            duration_s=0.02,
            call_started=True,
            error_type=(f"RemoteFailure/{secret}" if is_error and secret else None),
            correlation_id=(f"remote-correlation/{secret}" if is_error and secret else None),
        )

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        _result: Any,
    ) -> ExternalEffectClassification:
        if operation == "list_tools":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"operation": operation},
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass(
                str(context["rollback_class"])
            ),
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=bool(context["state_mutation"]),
            information_flow=bool(context["information_flow"]),
        )


def _manifest(
    server_id: str,
    *,
    input_schema: dict[str, Any] | None = None,
    env_source: str | None = None,
) -> dict[str, Any]:
    environment = (
        {"MCP_TEST_TOKEN": env_source}
        if env_source is not None
        else {}
    )
    return {
        "schema_version": 1,
        "server_id": server_id,
        "transport": "stdio",
        "stdio": {
            "command": MCP_TEST_STDIO_COMMAND,
            "args": ["-m", "demo_server"],
            "env": environment,
        },
        "tools": [
            {
                "tool_id": "echo",
                "mcp_name": "demo.echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": input_schema or {},
            }
        ],
        "timeout_s": 5,
        "max_request_bytes": 1_048_576,
        "max_response_bytes": 1_048_576,
    }


def _grant_call_authority(
    runtime: Runtime,
    pid: str,
    server_id: str,
    *,
    env_source: str | None = None,
    list_tools: bool = False,
) -> None:
    runtime.capability.grant(
        pid,
        f"mcp:{server_id}:echo",
        [CapabilityRight.READ],
        issued_by="test",
    )
    if list_tools:
        runtime.capability.grant(
            pid,
            f"mcp_server:{server_id}",
            [CapabilityRight.READ, CapabilityRight.EXECUTE],
            issued_by="test",
        )
    runtime.capability.grant(
        pid,
        "process:spawn",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    environment = (
        {"MCP_TEST_TOKEN": env_source}
        if env_source is not None
        else {}
    )
    runtime.capability.grant(
        pid,
        runtime.mcp.stdio_resource_for_argv(
            MCP_TEST_STDIO_COMMAND,
            ["-m", "demo_server"],
            env=environment,
        ),
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )


def test_call_uses_one_detached_canonical_argument_object_across_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    provider = _SecurityBoundaryMcpProvider()
    runtime.mcp.provider = provider
    original_arguments = {"text": ["before-dispatch"]}
    boundary_ids: dict[str, int] = {}

    try:
        server_id = "canonical-arguments"
        runtime.mcp.register_server(
            _manifest(server_id),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="canonical MCP argument snapshot",
        )
        _grant_call_authority(runtime, pid, server_id)

        original_schema_validation = runtime.mcp._validate_arguments_against_schema

        def capture_schema(
            server: Any,
            tool: Any,
            arguments: dict[str, Any],
        ) -> None:
            boundary_ids["schema"] = id(arguments)
            original_schema_validation(server, tool, arguments)

        monkeypatch.setattr(
            runtime.mcp,
            "_validate_arguments_against_schema",
            capture_schema,
        )
        original_precheck = runtime.data_flow.precheck_egress_clearance
        original_authorize = runtime.data_flow.authorize_egress

        def capture_precheck(*args: Any, **kwargs: Any) -> Any:
            boundary_ids["precheck"] = id(kwargs["payload"])
            return original_precheck(*args, **kwargs)

        def capture_authorize(*args: Any, **kwargs: Any) -> Any:
            boundary_ids["authorize"] = id(kwargs["payload"])
            return original_authorize(*args, **kwargs)

        monkeypatch.setattr(
            runtime.data_flow,
            "precheck_egress_clearance",
            capture_precheck,
        )
        monkeypatch.setattr(
            runtime.data_flow,
            "authorize_egress",
            capture_authorize,
        )
        original_environment = runtime.mcp._require_runtime_environment

        def mutate_original_after_authorization(*args: Any, **kwargs: Any) -> Any:
            original_arguments["text"][0] = "mutated-after-authorization"
            original_arguments["late"] = True
            return original_environment(*args, **kwargs)

        monkeypatch.setattr(
            runtime.mcp,
            "_require_runtime_environment",
            mutate_original_after_authorization,
        )

        result = runtime.mcp.call_tool(
            pid,
            server_id,
            "echo",
            original_arguments,
        )

        assert result.ok
        assert len(provider.call_argument_objects) == 1
        provider_arguments = provider.call_argument_objects[0]
        assert provider_arguments == {"text": ["before-dispatch"]}
        assert provider_arguments is not original_arguments
        assert provider_arguments["text"] is not original_arguments["text"]
        boundary_ids["provider"] = id(provider_arguments)
        assert len(set(boundary_ids.values())) == 1

        expected_hash = hashlib.sha256(
            dumps(provider_arguments).encode("utf-8")
        ).hexdigest()
        call_audits = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "primitive.mcp.call"
        ]
        assert len(call_audits) == 1
        assert call_audits[0].decision["arguments_sha256"] == expected_hash
    finally:
        runtime.close()


def test_python_none_arguments_remain_compatible_as_empty_canonical_object() -> None:
    runtime = Runtime.open("local")
    provider = _SecurityBoundaryMcpProvider()
    runtime.mcp.provider = provider
    try:
        server_id = "python-none-arguments"
        runtime.mcp.register_server(
            _manifest(server_id),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="Python None MCP compatibility",
        )
        _grant_call_authority(runtime, pid, server_id)

        result = runtime.mcp.call_tool(pid, server_id, "echo", None)

        assert result.ok
        assert provider.call_argument_objects == [{}]
        assert type(provider.call_argument_objects[0]) is dict
    finally:
        runtime.close()


class _HostileArguments(dict[str, Any]):
    def items(self) -> Any:
        raise AssertionError("hostile dict subclass was iterated")


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({1: "integer-key"}, id="integer-key"),
        pytest.param({"tuple": ("not", "json")}, id="tuple-value"),
        pytest.param(_HostileArguments({"text": "subclass"}), id="dict-subclass"),
    ],
)
def test_noncanonical_arguments_fail_before_authority_effect_or_provider(
    arguments: Any,
) -> None:
    runtime = Runtime.open("local")
    provider = _SecurityBoundaryMcpProvider()
    runtime.mcp.provider = provider
    try:
        server_id = "reject-noncanonical"
        runtime.mcp.register_server(
            _manifest(server_id),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject noncanonical MCP arguments",
        )
        capability = runtime.capability.issue_trusted(
            subject=pid,
            resource=f"mcp:{server_id}:echo",
            rights=[CapabilityRight.READ],
            issued_by="test",
            uses_remaining=1,
        )
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            runtime.mcp.stdio_resource_for_argv(
                MCP_TEST_STDIO_COMMAND,
                ["-m", "demo_server"],
            ),
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        with pytest.raises(
            ValidationError,
            match="MCP tool arguments must be a strict JSON object",
        ):
            runtime.mcp.call_tool(pid, server_id, "echo", arguments)

        assert provider.list_calls == 0
        assert provider.call_argument_objects == []
        assert runtime.store.list_external_effects(pid=pid) == []
        assert runtime.store.get_capability(capability.cap_id).uses_remaining == 1
    finally:
        runtime.close()


def test_reflected_transport_credential_never_reaches_results_or_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "opaque-mcp-provider-credential-without-known-prefix-73159"
    env_name = "AGENT_LIBOS_MCP_SECURITY_BOUNDARY_TOKEN"
    monkeypatch.setenv(env_name, secret)
    database = tmp_path / "mcp-provider-reflection.sqlite"
    runtime = Runtime.open(database)
    provider = _SecurityBoundaryMcpProvider(reflected_secret=True)
    runtime.mcp.provider = provider
    observed: dict[str, Any] = {}
    try:
        server_id = "credential-reflection"
        runtime.mcp.register_server(
            _manifest(server_id, env_source=env_name),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="public-safe MCP provider results",
        )
        _grant_call_authority(
            runtime,
            pid,
            server_id,
            env_source=env_name,
            list_tools=True,
        )
        runtime.tools.configure_process_tools(
            pid,
            ["list_mcp_tools", "call_mcp_tool"],
            assigned_by="test",
        )

        list_result = runtime.tools.call(
            pid,
            "list_mcp_tools",
            {"server_id": server_id, "refresh": True},
        )
        success_result = runtime.tools.call(
            pid,
            "call_mcp_tool",
            {
                "server_id": server_id,
                "tool_id": "echo",
                "arguments": {},
            },
        )
        error_result = runtime.tools.call(
            pid,
            "call_mcp_tool",
            {
                "server_id": server_id,
                "tool_id": "echo",
                "arguments": {"fail": True},
            },
        )

        assert list_result.result_handle is not None
        assert success_result.result_handle is not None
        assert error_result.result_handle is not None
        durable_results = [
            runtime.store.get_object(result.result_handle.oid)
            for result in (list_result, success_result, error_result)
        ]
        observed = {
            "list_result": list_result,
            "success_result": success_result,
            "error_result": error_result,
            "durable_results": durable_results,
            "audit": runtime.store.list_audit(),
            "events": runtime.store.list_events(),
            "operations": runtime.store.list_operations(),
            "effects": runtime.store.list_external_effects(pid=pid),
            "servers": runtime.store.list_mcp_servers(),
        }
        serialized = dumps(to_jsonable(observed))
        assert secret not in serialized
        assert "[redacted]" in serialized
    finally:
        runtime.close()

    persisted = database.read_bytes()
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(f"{database.name}{suffix}")
        if sidecar.exists():
            persisted += sidecar.read_bytes()
    assert secret.encode("utf-8") not in persisted


def test_short_reflected_credential_redaction_preserves_wire_byte_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "Q7vZ2pLm"
    env_name = "AGENT_LIBOS_MCP_SECURITY_SHORT_TOKEN"
    monkeypatch.setenv(env_name, secret)
    runtime = Runtime.open("local")
    provider = _SecurityBoundaryMcpProvider(reflected_secret=True)
    runtime.mcp.provider = provider
    try:
        server_id = "short-reflected-secret"
        runtime.mcp.register_server(
            _manifest(server_id, env_source=env_name),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="preserve raw MCP wire accounting after redaction",
        )
        _grant_call_authority(
            runtime,
            pid,
            server_id,
            env_source=env_name,
        )

        result = runtime.mcp.call_tool(pid, server_id, "echo", {})

        serialized = dumps(to_jsonable(result))
        assert result.ok
        assert secret not in serialized
        assert "[redacted]" in serialized
    finally:
        runtime.close()


def _regex_runtime(
    *,
    pattern_max_bytes: int = 1_024,
    max_evaluations: int = 4_096,
    timeout_s: float = 0.05,
) -> Runtime:
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            schema_regex_pattern_max_bytes=pattern_max_bytes,
            schema_regex_max_evaluations=max_evaluations,
            schema_regex_match_timeout_s=timeout_s,
        )
    )
    return Runtime.open("local", config=config)


def test_schema_regex_timeout_fails_closed_before_provider_or_effect() -> None:
    runtime = _regex_runtime(timeout_s=0.001)
    schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "pattern": "^(a|aa)+$",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    provider = _SecurityBoundaryMcpProvider(live_schema=schema)
    runtime.mcp.provider = provider
    try:
        server_id = "bounded-regex-timeout"
        runtime.mcp.register_server(
            _manifest(server_id, input_schema=schema),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="bound MCP schema regex",
        )
        _grant_call_authority(runtime, pid, server_id)

        with pytest.raises(
            ValidationError,
            match="MCP schema regex validation timed out",
        ):
            runtime.mcp.call_tool(
                pid,
                server_id,
                "echo",
                {"text": ("a" * 100_000) + "!"},
            )

        assert provider.list_calls == 0
        assert provider.call_argument_objects == []
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_schema_pattern_length_and_evaluation_budget_fail_closed() -> None:
    overlong_runtime = _regex_runtime(pattern_max_bytes=8)
    try:
        with pytest.raises(
            ValidationError,
            match="MCP input_schema regex pattern exceeds maximum bytes=8",
        ):
            overlong_runtime.mcp.register_server(
                _manifest(
                    "overlong-regex",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "pattern": "123456789",
                            }
                        },
                    },
                ),
                actor="test",
                require_capability=False,
            )
        assert overlong_runtime.mcp.list_servers(require_capability=False) == []
    finally:
        overlong_runtime.close()

    invalid_runtime = _regex_runtime()
    try:
        with pytest.raises(
            ValidationError,
            match="MCP input_schema regex pattern is invalid",
        ):
            invalid_runtime.mcp.register_server(
                _manifest(
                    "invalid-regex",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "pattern": "(",
                            }
                        },
                    },
                ),
                actor="test",
                require_capability=False,
            )
        assert invalid_runtime.mcp.list_servers(require_capability=False) == []
    finally:
        invalid_runtime.close()

    budget_runtime = _regex_runtime(max_evaluations=1)
    schema = {
        "type": "object",
        "patternProperties": {
            "^first$": {"type": "string"},
            "^second$": {"type": "string"},
        },
        "additionalProperties": False,
    }
    provider = _SecurityBoundaryMcpProvider(live_schema=schema)
    budget_runtime.mcp.provider = provider
    try:
        server_id = "regex-evaluation-budget"
        budget_runtime.mcp.register_server(
            _manifest(server_id, input_schema=schema),
            actor="test",
            require_capability=False,
        )
        pid = budget_runtime.process.spawn(
            image="base-agent:v0",
            goal="bound MCP schema regex evaluations",
        )
        _grant_call_authority(budget_runtime, pid, server_id)

        with pytest.raises(
            ValidationError,
            match="MCP schema regex evaluation budget exhausted",
        ):
            budget_runtime.mcp.call_tool(
                pid,
                server_id,
                "echo",
                {"first": "value", "second": "value"},
            )

        assert provider.list_calls == 0
        assert provider.call_argument_objects == []
        assert budget_runtime.store.list_external_effects(pid=pid) == []
    finally:
        budget_runtime.close()


def test_bounded_schema_regex_preserves_legitimate_pattern_semantics() -> None:
    runtime = _regex_runtime()
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "pattern": "^[A-Z]{2}-[0-9]{3}$",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    }
    provider = _SecurityBoundaryMcpProvider(live_schema=schema)
    runtime.mcp.provider = provider
    try:
        server_id = "legitimate-regex"
        runtime.mcp.register_server(
            _manifest(server_id, input_schema=schema),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="preserve legitimate MCP regex",
        )
        _grant_call_authority(runtime, pid, server_id)

        result = runtime.mcp.call_tool(
            pid,
            server_id,
            "echo",
            {"code": "AB-123"},
        )

        assert result.ok
        assert provider.call_argument_objects == [{"code": "AB-123"}]
    finally:
        runtime.close()


def test_bounded_schema_regex_preserves_unevaluated_properties_semantics() -> None:
    runtime = _regex_runtime()
    schema = {
        "type": "object",
        "patternProperties": {
            "^item-[0-9]+$": {"type": "string"},
        },
        "unevaluatedProperties": False,
    }
    provider = _SecurityBoundaryMcpProvider(live_schema=schema)
    runtime.mcp.provider = provider
    try:
        server_id = "bounded-unevaluated-properties"
        runtime.mcp.register_server(
            _manifest(server_id, input_schema=schema),
            actor="test",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="preserve bounded JSON Schema 2020-12 semantics",
        )
        _grant_call_authority(runtime, pid, server_id)

        with pytest.raises(
            ValidationError,
            match="unevaluated properties are not allowed",
        ):
            runtime.mcp.call_tool(
                pid,
                server_id,
                "echo",
                {"unexpected": "value"},
            )
        assert provider.call_argument_objects == []

        result = runtime.mcp.call_tool(
            pid,
            server_id,
            "echo",
            {"item-42": "value"},
        )
        assert result.ok
        assert provider.call_argument_objects == [{"item-42": "value"}]
    finally:
        runtime.close()
