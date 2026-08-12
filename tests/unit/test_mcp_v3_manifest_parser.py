from __future__ import annotations

from copy import deepcopy

import pytest

from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    parse_mcp_v3_manifest_mapping,
    parse_mcp_v3_manifest_yaml_text,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import McpProtocolMode
from agent_libos.utils.yaml_loader import YAML_MAX_UTF8_BYTES


def _stdio_manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "server_id": "stdio-demo",
        "transport": "stdio",
        "protocol_mode": "2026-07-28",
        "timeout_s": 10,
        "max_request_bytes": 4096,
        "max_response_bytes": 8192,
        "stdio": {
            "command": "demo-server",
            "args": ("--deterministic",),
            "env": {"DEMO_TOKEN": "HOST_DEMO_TOKEN"},
            "cwd": None,
        },
        "tools": [
            {
                "tool_id": "echo",
                "mcp_name": "demo.echo",
                "right": "execute",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
                "metadata": {"owner": "example"},
            }
        ],
        "prompts": [
            {
                "prompt_id": "summarize",
                "mcp_name": "demo.summarize",
                "argument_names": ["text"],
            }
        ],
        "subscriptions": ["toolsListChanged", "promptsListChanged"],
    }


def _http_manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "server_id": "http-demo",
        "transport": "streamable_http",
        "protocol_mode": "2026-07-28",
        "timeout_s": 2.5,
        "max_request_bytes": 4096,
        "max_response_bytes": 8192,
        "http": {
            "url": "http://127.0.0.1:8765/mcp",
            "headers": {
                "Authorization": {
                    "env": "MCP_HTTP_TOKEN",
                    "prefix": "Bearer ",
                }
            },
        },
        "resources": [
            {
                "resource_id": "readme",
                "remote_uri": "demo://readme",
                "mime_types": ["text/plain"],
            }
        ],
        "resource_templates": [
            {
                "template_id": "document",
                "remote_uri_template": "demo://documents/{name}",
                "variables": ["name"],
            }
        ],
        "metadata": {"environment": "test"},
    }


def test_parse_valid_stdio_mapping_constructs_typed_manifest() -> None:
    manifest = parse_mcp_v3_manifest_mapping(_stdio_manifest())

    assert isinstance(manifest, McpServerManifestV3)
    assert manifest.protocol_mode is McpProtocolMode.REVISION_2026_07_28
    assert manifest.timeout_s == 10.0
    assert type(manifest.timeout_s) is float
    assert manifest.stdio is not None
    assert manifest.stdio.args == ["--deterministic"]
    assert manifest.http is None
    assert manifest.tools[0].state_mutation is False
    assert manifest.prompts[0].argument_names == ("text",)


def test_parse_valid_http_yaml_constructs_nested_specs() -> None:
    manifest = parse_mcp_v3_manifest_yaml_text(
        """
schema_version: 3
server_id: http-demo
transport: streamable_http
protocol_mode: '2026-07-28'
timeout_s: 2.5
max_request_bytes: 4096
max_response_bytes: 8192
http:
  url: http://127.0.0.1:8765/mcp
  headers:
    Authorization:
      env: MCP_HTTP_TOKEN
      prefix: 'Bearer '
resources:
  - resource_id: readme
    remote_uri: demo://readme
    mime_types: [text/plain]
resource_templates:
  - template_id: document
    remote_uri_template: 'demo://documents/{name}'
    variables: [name]
"""
    )

    assert manifest.http is not None
    assert manifest.http.headers["Authorization"].env == "MCP_HTTP_TOKEN"
    assert manifest.http.headers["Authorization"].prefix == "Bearer "
    assert manifest.resources[0].mime_types == ("text/plain",)
    assert manifest.resource_templates[0].variables == ("name",)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"future_field": True}),
        lambda value: value["stdio"].update({"shell": True}),
        lambda value: value["tools"][0].update({"description": "not declared"}),
    ],
)
def test_parse_rejects_unknown_fields_at_closed_object_layers(mutate: object) -> None:
    value = _stdio_manifest()
    mutate(value)  # type: ignore[operator]

    with pytest.raises(ValidationError, match="unknown fields"):
        parse_mcp_v3_manifest_mapping(value)


def test_parse_rejects_unknown_http_header_fields() -> None:
    value = _http_manifest()
    value["http"]["headers"]["Authorization"]["value"] = "secret"  # type: ignore[index]

    with pytest.raises(ValidationError, match="unknown fields"):
        parse_mcp_v3_manifest_mapping(value)


@pytest.mark.parametrize(
    ("path", "invalid", "message"),
    [
        (("schema_version",), True, "must be an integer"),
        (("timeout_s",), False, "must be a number"),
        (("tools",), {}, "must be a list or tuple"),
        (("tools", 0, "state_mutation"), 0, "must be a boolean"),
        (("stdio", "args"), "--unsafe", "must be a list or tuple"),
    ],
)
def test_parse_rejects_ambiguous_or_wrong_types(
    path: tuple[object, ...], invalid: object, message: str
) -> None:
    value: object = deepcopy(_stdio_manifest())
    selected = value
    for part in path[:-1]:
        selected = selected[part]  # type: ignore[index]
    selected[path[-1]] = invalid  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "resource",
    [
        {"resource_id": "app", "remote_uri": "ui://widget"},
        {
            "resource_id": "app",
            "remote_uri": "demo://widget",
            "metadata": {"io.modelcontextprotocol/ui": {"entry": "widget"}},
        },
        {
            "resource_id": "app",
            "remote_uri": "demo://widget",
            "metadata": {"UI/ResourceUri": "ui://legacy-flat"},
        },
    ],
)
def test_parse_rejects_mcp_apps(resource: dict[str, object]) -> None:
    value = _http_manifest()
    value["resources"] = [resource]

    with pytest.raises(ValidationError, match="Apps"):
        parse_mcp_v3_manifest_mapping(value)


def test_yaml_parser_applies_shared_byte_bound_before_manifest_decode() -> None:
    oversized = "x" * (YAML_MAX_UTF8_BYTES + 1)

    with pytest.raises(ValidationError, match="YAML_MAX_UTF8_BYTES"):
        parse_mcp_v3_manifest_yaml_text(oversized)
