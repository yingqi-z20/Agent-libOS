from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from agent_libos.mcp.manifest import (
    MCP_TASKS_EXTENSION_ID,
    McpManifestV3HostPolicy,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_mapping,
    validate_mcp_v3_manifest,
)
from agent_libos.models.exceptions import ValidationError


_TASKS_DIGEST = "a" * 64


class _DictSubclass(dict[str, object]):
    pass


class _ListSubclass(list[object]):
    pass


def _tool(index: int = 0) -> dict[str, object]:
    suffix = "" if index == 0 else str(index)
    return {
        "tool_id": f"echo{suffix}",
        "mcp_name": f"demo.echo{suffix}",
        "right": "execute",
        "rollback_class": "no_rollback_required",
        "state_mutation": False,
        "information_flow": True,
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        },
    }


def _stdio_manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "server_id": "stdio-secure",
        "transport": "stdio",
        "protocol_mode": "2026-07-28",
        "timeout_s": 10,
        "max_request_bytes": 4096,
        "max_response_bytes": 8192,
        "stdio": {
            "command": "demo-server",
            "args": ["--deterministic"],
            "env": {"TOKEN": "AGENT_LIBOS_MCP_TOKEN"},
            "cwd": "examples/mcp",
        },
        "tools": [_tool()],
    }


def _http_manifest(url: str = "https://mcp.example.test/rpc") -> dict[str, object]:
    value = _stdio_manifest()
    value.update(
        {
            "server_id": "http-secure",
            "transport": "streamable_http",
            "http": {
                "url": url,
                "headers": {
                    "Authorization": {
                        "env": "AGENT_LIBOS_MCP_TOKEN",
                        "prefix": "Bearer ",
                    }
                },
            },
        }
    )
    value.pop("stdio")
    return value


@pytest.mark.parametrize(
    "url",
    [
        "ftp://mcp.example.test/rpc",
        "https://user:secret@mcp.example.test/rpc",
        "https://@mcp.example.test/rpc",
        "https://mcp.example.test/rpc#fragment",
        "https:///rpc",
        "https://mcp.example.test:99999/rpc",
        "https://mcp.example.test:0/rpc",
        "http://mcp.example.test/rpc",
        "https://metadata.google.internal/computeMetadata/v1",
        "https://metadata。google。internal/computeMetadata/v1",
        "https://metadata./latest",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1/rpc",
        "https://100.100.100.200/latest/meta-data",
        "https://168.63.129.16/metadata/instance",
        "https://2130706433/rpc",
        "https://0x7f000001/rpc",
    ],
)
def test_url_validation_rejects_unsafe_or_ambiguous_targets(url: str) -> None:
    with pytest.raises(ValidationError):
        parse_mcp_v3_manifest_mapping(_http_manifest(url))


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8765/mcp",
        "http://127.0.0.1:8765/mcp",
        "http://[::1]:8765/mcp",
        "https://mcp.example.test/rpc",
        "https://8.8.8.8/rpc",
    ],
)
def test_url_validation_accepts_only_explicit_local_http_or_safe_https(url: str) -> None:
    parse_mcp_v3_manifest_mapping(_http_manifest(url))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("command", "python -m demo", "single argv"),
        ("command", "~/bin/demo", "home-directory"),
        ("command", "demo;other", "single argv"),
        ("args", ["ok", "bad\x00arg"], "NUL"),
        ("cwd", "/tmp", "relative"),
        ("cwd", "../outside", "escapes"),
        ("cwd", "safe/../../outside", "escapes"),
        ("cwd", "C:\\outside", "relative"),
    ],
)
def test_stdio_is_direct_argv_and_cwd_cannot_escape(
    field: str, value: object, message: str
) -> None:
    manifest = _stdio_manifest()
    manifest["stdio"][field] = value  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(manifest)


@pytest.mark.parametrize(
    "env",
    [
        {"BAD-NAME": "AGENT_LIBOS_MCP_TOKEN"},
        {"TOKEN": "BAD-NAME"},
    ],
)
def test_stdio_environment_names_use_portable_grammar(
    env: dict[str, str],
) -> None:
    manifest = _stdio_manifest()
    manifest["stdio"]["env"] = env  # type: ignore[index]
    with pytest.raises(ValidationError, match="environment name"):
        parse_mcp_v3_manifest_mapping(manifest)


@pytest.mark.parametrize(
    ("name", "spec", "message"),
    [
        ("Bad Header", {"env": "AGENT_LIBOS_MCP_TOKEN"}, "header name"),
        ("x" * 129, {"env": "AGENT_LIBOS_MCP_TOKEN"}, "header name"),
        ("Content-Type", {"env": "AGENT_LIBOS_MCP_TOKEN"}, "forbidden"),
        ("Mcp-Param-Token", {"env": "AGENT_LIBOS_MCP_TOKEN"}, "forbidden"),
        ("X-Token", {"env": "BAD-NAME"}, "environment name"),
        (
            "X-Token",
            {"env": "AGENT_LIBOS_MCP_TOKEN", "prefix": "secret="},
            "prefix",
        ),
        (
            "X-Token",
            {"env": "AGENT_LIBOS_MCP_TOKEN", "suffix": " literal"},
            "suffix",
        ),
    ],
)
def test_modern_headers_are_env_backed_and_reserved_headers_are_closed(
    name: str, spec: dict[str, str], message: str
) -> None:
    manifest = _http_manifest()
    manifest["http"]["headers"] = {name: spec}  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tool_id", "x" * 97, "tool_id"),
        ("mcp_name", "x" * 257, "mcp_name"),
        ("mcp_name", "demo\necho", "mcp_name"),
        ("right", "admin", "right"),
        ("rollback_class", "none", "rollback"),
        ("rollback_status", "complete", "rollback"),
        ("information_flow", False, "information_flow"),
    ],
)
def test_tool_authority_fields_are_closed_and_bounded(
    field: str, value: object, message: str
) -> None:
    manifest = _stdio_manifest()
    manifest["tools"][0][field] = value  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(manifest)


def test_mutation_cannot_claim_no_rollback_required() -> None:
    manifest = _stdio_manifest()
    manifest["tools"][0]["state_mutation"] = True  # type: ignore[index]
    with pytest.raises(ValidationError, match="state_mutation"):
        parse_mcp_v3_manifest_mapping(manifest)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_DictSubclass({"nested": "value"}), "strict JSON"),
        (_ListSubclass(["value"]), "strict JSON"),
        (("tuple",), "strict JSON"),
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
    ],
)
def test_nested_metadata_requires_exact_json_container_types(
    value: object,
    message: str,
) -> None:
    mapping = _stdio_manifest()
    mapping["metadata"] = {"nested": value}
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(mapping)


def test_nested_tool_schema_rejects_custom_container_subclasses() -> None:
    mapping = _stdio_manifest()
    mapping["tools"][0]["input_schema"] = {  # type: ignore[index]
        "type": "object",
        "properties": _DictSubclass(
            {"text": _DictSubclass({"type": "string"})}
        ),
    }
    with pytest.raises(ValidationError, match="strict JSON"):
        parse_mcp_v3_manifest_mapping(mapping)


@pytest.mark.parametrize(
    "mime_type",
    [
        "text/html;profile=mcp-app",
        "TEXT/HTML;PROFILE=MCP-APP",
        "text/html; profile=mcp-app",
        'text/html ; profile = "mcp-app"',
        "text/html;charset=utf-8; profile = 'MCP-APP'",
    ],
)
def test_apps_mime_is_rejected_after_parameter_normalization(
    mime_type: str,
) -> None:
    mapping = _stdio_manifest()
    mapping["resources"] = [
        {
            "resource_id": "app-resource",
            "remote_uri": "demo://app",
            "mime_types": [mime_type],
        }
    ]
    with pytest.raises(ValidationError, match="Apps HTML"):
        parse_mcp_v3_manifest_mapping(mapping)


def test_non_app_html_profile_is_not_overblocked() -> None:
    mapping = _stdio_manifest()
    mapping["resources"] = [
        {
            "resource_id": "ordinary-html",
            "remote_uri": "demo://ordinary",
            "mime_types": ["text/html; profile=ordinary"],
        }
    ]
    parse_mcp_v3_manifest_mapping(mapping)


def test_host_policy_is_explicit_and_canonical_identity_is_policy_independent() -> None:
    mapping = _stdio_manifest()
    mapping["stdio"]["env"] = {"TOKEN": "CUSTOM_MCP_TOKEN"}  # type: ignore[index]

    structural = parse_mcp_v3_manifest_mapping(mapping)
    identity = canonical_mcp_v3_manifest_json(structural)
    with pytest.raises(ValidationError, match="allowlisted"):
        parse_mcp_v3_manifest_mapping(mapping, enforce_host_policy=True)

    custom = McpManifestV3HostPolicy(
        stdio_env_allowlist=("CUSTOM_MCP_*",),
    )
    admitted = parse_mcp_v3_manifest_mapping(
        mapping,
        host_policy=custom,
        enforce_host_policy=True,
    )
    assert canonical_mcp_v3_manifest_json(admitted) == identity
    with pytest.raises(ValidationError, match="requires enforce_host_policy"):
        validate_mcp_v3_manifest(structural, host_policy=custom)


def test_host_policy_attenuates_hard_caps_and_catalogs() -> None:
    mapping = _stdio_manifest()
    mapping["tools"] = [_tool(0), _tool(1)]
    policy = McpManifestV3HostPolicy(
        tool_catalog_limit=1,
        stdio_env_allowlist=("AGENT_LIBOS_MCP_*",),
    )
    with pytest.raises(ValidationError, match="tool catalog"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=policy,
            enforce_host_policy=True,
        )

    mapping["tools"] = [_tool()]
    policy = replace(
        policy,
        tool_catalog_limit=1,
        max_request_hard_limit_bytes=1024,
    )
    with pytest.raises(ValidationError, match="max_request_bytes"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=policy,
            enforce_host_policy=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timeout_s", 60.01, "timeout_s"),
        ("max_request_bytes", 1_048_577, "max_request_bytes"),
        ("max_response_bytes", 8_388_609, "max_response_bytes"),
        ("server_id", "s" * 97, "server_id"),
    ],
)
def test_release_hard_caps_apply_without_deployment_policy(
    field: str, value: object, message: str
) -> None:
    mapping = _stdio_manifest()
    mapping[field] = value
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(mapping)


@pytest.mark.parametrize(
    ("catalog", "items", "policy"),
    [
        (
            "resources",
            [
                {"resource_id": "r1", "remote_uri": "demo://r1"},
                {"resource_id": "r2", "remote_uri": "demo://r2"},
            ],
            McpManifestV3HostPolicy(resource_catalog_limit=1),
        ),
        (
            "resource_templates",
            [
                {
                    "template_id": "t1",
                    "remote_uri_template": "demo://t1/{id}",
                    "variables": ["id"],
                },
                {
                    "template_id": "t2",
                    "remote_uri_template": "demo://t2/{id}",
                    "variables": ["id"],
                },
            ],
            McpManifestV3HostPolicy(resource_template_limit=1),
        ),
        (
            "prompts",
            [
                {"prompt_id": "p1", "mcp_name": "demo.p1"},
                {"prompt_id": "p2", "mcp_name": "demo.p2"},
            ],
            McpManifestV3HostPolicy(prompt_catalog_limit=1),
        ),
    ],
)
def test_each_modern_surface_has_an_independent_host_catalog_limit(
    catalog: str,
    items: list[dict[str, object]],
    policy: McpManifestV3HostPolicy,
) -> None:
    mapping = _stdio_manifest()
    mapping[catalog] = items
    with pytest.raises(ValidationError, match="catalog exceeds"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=policy,
            enforce_host_policy=True,
        )


def test_host_policy_cannot_expand_release_caps_or_use_bare_wildcard() -> None:
    mapping = _stdio_manifest()
    with pytest.raises(ValidationError, match="release maximum"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(tool_catalog_limit=101),
            enforce_host_policy=True,
        )
    with pytest.raises(ValidationError, match="pattern is invalid"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(stdio_env_allowlist=("*",)),
            enforce_host_policy=True,
        )


def test_oauth_and_tasks_require_enablement_and_exact_host_pin() -> None:
    oauth = _http_manifest()
    oauth["http"]["headers"] = {}  # type: ignore[index]
    oauth["auth_profile_id"] = "oauth-profile"
    with pytest.raises(ValidationError, match="OAuth is disabled"):
        parse_mcp_v3_manifest_mapping(oauth, enforce_host_policy=True)
    parse_mcp_v3_manifest_mapping(
        oauth,
        host_policy=McpManifestV3HostPolicy(oauth_enabled=True),
        enforce_host_policy=True,
    )

    tasks = _stdio_manifest()
    tasks["subscriptions"] = ["taskIds"]
    tasks["tasks_extension"] = {
        "extension_id": MCP_TASKS_EXTENSION_ID,
        "spec_sha256": _TASKS_DIGEST,
    }
    with pytest.raises(ValidationError, match="disabled"):
        parse_mcp_v3_manifest_mapping(tasks, enforce_host_policy=True)
    policy = McpManifestV3HostPolicy(
        tasks_extension_enabled=True,
        tasks_extension_spec_sha256=_TASKS_DIGEST,
    )
    parse_mcp_v3_manifest_mapping(
        tasks,
        host_policy=policy,
        enforce_host_policy=True,
    )
    with pytest.raises(ValidationError, match="policy pin"):
        parse_mcp_v3_manifest_mapping(
            tasks,
            host_policy=replace(
                policy,
                tasks_extension_spec_sha256="b" * 64,
            ),
            enforce_host_policy=True,
        )


def test_oauth_profile_rejects_ambiguous_static_authorization_for_mapping_and_typed_manifest() -> None:
    mapping = _http_manifest()
    mapping["auth_profile_id"] = "oauth-profile"
    with pytest.raises(ValidationError, match="cannot be combined"):
        parse_mcp_v3_manifest_mapping(mapping)

    mapping.pop("auth_profile_id")
    typed = parse_mcp_v3_manifest_mapping(mapping)
    with pytest.raises(ValidationError, match="cannot be combined"):
        validate_mcp_v3_manifest(
            replace(typed, auth_profile_id="oauth-profile")
        )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "string"}, "root type"),
        ({"type": "object", "$ref": "https://example.test/schema"}, "external"),
        ({"type": "object", "$ref": "#/missing"}, "unresolved"),
        ({"type": "object", "$ref": "#"}, "cyclic"),
        ({"type": "object", "$dynamicRef": "#node"}, "dynamic"),
        ({"type": "object", "properties": {"x": {"type": "invalid"}}}, "valid JSON Schema"),
        ({"type": "object", "patternProperties": {"[": {"type": "string"}}}, "invalid regex"),
        ({"type": "object", "properties": {"x": {"type": "string", "pattern": "x" * 1025}}}, "UTF-8 bytes"),
    ],
)
def test_json_schema_is_valid_local_and_bounded(
    schema: dict[str, object], message: str
) -> None:
    mapping = _stdio_manifest()
    mapping["tools"][0]["input_schema"] = schema  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        parse_mcp_v3_manifest_mapping(mapping)


def test_json_schema_accepts_bounded_acyclic_local_references() -> None:
    mapping = _stdio_manifest()
    mapping["tools"][0]["input_schema"] = {  # type: ignore[index]
        "type": "object",
        "$defs": {"text": {"type": "string", "pattern": "^[a-z]+$"}},
        "properties": {"text": {"$ref": "#/$defs/text"}},
    }
    parse_mcp_v3_manifest_mapping(mapping)


def test_schema_policy_bounds_composition_regex_budget_and_deadline() -> None:
    mapping = _stdio_manifest()
    mapping["tools"][0]["input_schema"] = {  # type: ignore[index]
        "type": "object",
        "allOf": [{"type": "object"}, {"type": "object"}],
    }
    with pytest.raises(ValidationError, match="combinator expansion"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(
                schema_max_composition_expansions=1
            ),
            enforce_host_policy=True,
        )

    mapping["tools"][0]["input_schema"] = {  # type: ignore[index]
        "type": "object",
        "properties": {
            "a": {"type": "string", "pattern": "a"},
            "b": {"type": "string", "pattern": "b"},
        },
    }
    with pytest.raises(ValidationError, match="regex evaluation budget"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(
                schema_regex_max_evaluations=1
            ),
            enforce_host_policy=True,
        )

    with pytest.raises(ValidationError, match="timed out"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(
                schema_regex_match_timeout_s=1e-12
            ),
            enforce_host_policy=True,
        )


def test_nested_schema_depth_and_node_caps_fail_closed() -> None:
    mapping = _stdio_manifest()
    nested: dict[str, object] = {"type": "object"}
    selected = nested
    for index in range(4):
        child: dict[str, object] = {"type": "object"}
        selected["properties"] = {f"p{index}": child}
        selected = child
    mapping["tools"][0]["input_schema"] = deepcopy(nested)  # type: ignore[index]
    with pytest.raises(ValidationError, match="schema depth"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(schema_max_depth=3),
            enforce_host_policy=True,
        )
    with pytest.raises(ValidationError, match="schema nodes"):
        parse_mcp_v3_manifest_mapping(
            mapping,
            host_policy=McpManifestV3HostPolicy(schema_max_nodes=5),
            enforce_host_policy=True,
        )
