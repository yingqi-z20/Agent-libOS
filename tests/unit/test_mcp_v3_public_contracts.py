from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.mcp import (
    MCP_TASKS_EXTENSION_ID,
    McpProtocolMode,
    McpResourceSpec,
    McpServerManifestV3,
    McpStdioTransportSpec,
    McpTasksExtensionSpec,
    canonical_mcp_v3_manifest_json,
    validate_mcp_v3_manifest,
)
from agent_libos.models.exceptions import ValidationError


_TASKS_DIGEST = "a" * 64


def _manifest(**overrides: object) -> McpServerManifestV3:
    selected: dict[str, object] = {
        "schema_version": 3,
        "server_id": "demo",
        "transport": "stdio",
        "stdio": McpStdioTransportSpec(command="demo-server"),
        "timeout_s": 10.0,
        "max_request_bytes": 1024,
        "max_response_bytes": 4096,
        "protocol_mode": McpProtocolMode.REVISION_2026_07_28,
        "resources": (McpResourceSpec("readme", "docs://readme"),),
    }
    selected.update(overrides)
    return McpServerManifestV3(**selected)  # type: ignore[arg-type]


def test_v3_can_register_resource_only_server_and_has_stable_identity() -> None:
    manifest = _manifest()
    validate_mcp_v3_manifest(manifest)
    encoded = canonical_mcp_v3_manifest_json(manifest)
    assert '"schema_version": 3' in encoded
    assert '"protocol_mode": "2026-07-28"' in encoded
    assert encoded == canonical_mcp_v3_manifest_json(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("protocol_mode", McpProtocolMode.AUTO),
        ("protocol_mode", McpProtocolMode.LEGACY),
    ],
)
def test_v3_cannot_downgrade(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        validate_mcp_v3_manifest(replace(_manifest(), **{field: value}))


def test_v3_rejects_empty_authority_manifest() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        validate_mcp_v3_manifest(_manifest(resources=()))


@pytest.mark.parametrize(
    "resource",
    [
        McpResourceSpec("app", "ui://widget"),
        McpResourceSpec("app", "opaque://widget", mime_types=("text/html;profile=mcp-app",)),
        McpResourceSpec(
            "app", "opaque://widget", metadata={"io.modelcontextprotocol/ui": {}}
        ),
    ],
)
def test_v3_rejects_mcp_apps(resource: McpResourceSpec) -> None:
    with pytest.raises(ValidationError, match="Apps"):
        validate_mcp_v3_manifest(_manifest(resources=(resource,)))


def test_v3_tasks_require_exact_extension_and_host_digest_pin() -> None:
    extension = McpTasksExtensionSpec(MCP_TASKS_EXTENSION_ID, _TASKS_DIGEST)
    manifest = _manifest(tasks_extension=extension, subscriptions=("taskIds",))
    validate_mcp_v3_manifest(manifest, tasks_extension_sha256=_TASKS_DIGEST)
    with pytest.raises(ValidationError, match="Host pin"):
        validate_mcp_v3_manifest(manifest, tasks_extension_sha256="b" * 64)


def test_oauth_reference_is_http_only() -> None:
    with pytest.raises(ValidationError, match="streamable_http"):
        validate_mcp_v3_manifest(_manifest(auth_profile_id="host-profile"))
