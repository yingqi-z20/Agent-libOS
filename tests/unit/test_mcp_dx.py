from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.mcp.dx import (
    MCP_DX_CANDIDATE_KIND,
    McpDxConfirmation,
    McpDxConfirmationRequired,
    McpDxManagerAdapter,
    McpProbeReport,
    approve_scaffold_candidate,
    doctor_manifest_text,
    export_registry_bundle,
    import_one_from_bundle,
    plan_import_bundle,
    probe_manifest,
    scaffold_manifest_candidate,
    validate_manifest_text,
)
from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_yaml_text,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import canonical_mcp_server_spec_json
from agent_libos.utils.serde import dumps


def _manifest(server_id: str = "dx-demo") -> str:
    return f"""
schema_version: 2
protocol_mode: "2026-07-28"
server_id: {server_id}
transport: stdio
stdio:
  command: python3
  args: ["-m", "demo_mcp"]
  env:
    DEMO_TOKEN: AGENT_LIBOS_MCP_DX_TEST_TOKEN
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
        text: {{type: string}}
      required: [text]
      additionalProperties: false
""".strip()


def _base_manifest(server_id: str = "scaffold-demo") -> dict[str, Any]:
    return {
        "schema_version": 3,
        "protocol_mode": "2026-07-28",
        "server_id": server_id,
        "transport": "stdio",
        "stdio": {"command": "python3", "args": ["-m", "demo_mcp"]},
        "tools": [
            {
                "tool_id": "onboarding_placeholder",
                "mcp_name": "onboarding.placeholder",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "rollback_status": "not_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {},
            }
        ],
        "timeout_s": 10,
        "max_request_bytes": 65536,
        "max_response_bytes": 1048576,
    }


def _confirmation() -> McpDxConfirmation:
    return McpDxConfirmation(
        confirmed=True,
        actor="test-reviewer",
        reason="review deterministic MCP candidate",
    )


def _manifest_digest(adapter: McpDxManagerAdapter, manifest: dict[str, Any]) -> str:
    selected = adapter.coerce_mapping(manifest)
    canonical = (
        canonical_mcp_v3_manifest_json(selected)
        if isinstance(selected, McpServerManifestV3)
        else canonical_mcp_server_spec_json(selected)
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def test_validate_manifest_is_offline_and_does_not_mutate_runtime() -> None:
    runtime = Runtime.open("local")
    try:
        before_events = runtime.events.list()
        before_audit = runtime.audit.trace()

        report = validate_manifest_text(McpDxManagerAdapter(runtime.mcp), _manifest())

        assert report.server_id == "dx-demo"
        assert report.schema_version == 2
        assert report.protocol_mode == "2026-07-28"
        assert report.stdio_authority_resource is not None
        assert runtime.mcp.list_servers(require_capability=False) == []
        assert runtime.events.list() == before_events
        assert runtime.audit.trace() == before_audit
    finally:
        runtime.close()


def test_doctor_reports_environment_presence_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "MCP_DX_SECRET_MUST_NOT_APPEAR"
    monkeypatch.setenv("AGENT_LIBOS_MCP_DX_TEST_TOKEN", secret)
    runtime = Runtime.open("local")
    try:
        report = doctor_manifest_text(
            McpDxManagerAdapter(runtime.mcp),
            _manifest(),
        )
        encoded = json.dumps(report.to_jsonable(), sort_keys=True)

        assert "AGENT_LIBOS_MCP_DX_TEST_TOKEN" in encoded
        assert secret not in encoded
        assert report.environment == (
            {
                "purpose": "stdio.env.DEMO_TOKEN",
                "name": "AGENT_LIBOS_MCP_DX_TEST_TOKEN",
                "present": True,
            },
        )
        assert any(item.code == "provider.dispatch" and item.status == "deferred" for item in report.checks)
    finally:
        runtime.close()


def test_scaffold_requires_confirmation_and_complete_full_catalog() -> None:
    runtime = Runtime.open("local")
    adapter = McpDxManagerAdapter(runtime.mcp)
    tools = [
        {
            "name": "demo.write/item",
            "description": "writes one item",
            "input_schema": {"type": "object"},
        }
    ]
    try:
        with pytest.raises(McpDxConfirmationRequired):
            scaffold_manifest_candidate(
                adapter,
                _base_manifest(),
                tools,
                probe_manifest_sha256=_manifest_digest(adapter, _base_manifest()),
                confirmation=McpDxConfirmation(False, "reviewer", "not yet"),
            )
        with pytest.raises(ValidationError, match="complete full-catalog"):
            scaffold_manifest_candidate(
                adapter,
                _base_manifest(),
                tools,
                probe_manifest_sha256=_manifest_digest(adapter, _base_manifest()),
                confirmation=_confirmation(),
                catalog_scope="registered_allowlist",
            )
    finally:
        runtime.close()


def test_scaffold_is_deterministic_conservative_and_not_directly_registerable() -> None:
    runtime = Runtime.open("local")
    adapter = McpDxManagerAdapter(runtime.mcp)
    tools = [
        {
            "name": "demo.write/item",
            "description": "writes one item",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
        {
            "name": "demo.write_item",
            "description": None,
            "input_schema": {"type": "object"},
        },
    ]
    try:
        first = scaffold_manifest_candidate(
            adapter,
            _base_manifest(),
            tools,
            probe_manifest_sha256=_manifest_digest(adapter, _base_manifest()),
            confirmation=_confirmation(),
        )
        second = scaffold_manifest_candidate(
            adapter,
            _base_manifest(),
            list(reversed(tools)),
            probe_manifest_sha256=_manifest_digest(adapter, _base_manifest()),
            confirmation=_confirmation(),
        )

        assert first == second
        assert first["kind"] == MCP_DX_CANDIDATE_KIND
        generated = first["manifest"]["tools"]
        assert len({item["tool_id"] for item in generated}) == 2
        for item in generated:
            assert item["right"] == "execute"
            assert item["rollback_class"] == "unknown"
            assert item["rollback_status"] == "unknown"
            assert item["state_mutation"] is True
            assert item["information_flow"] is True

        with pytest.raises(ValidationError):
            runtime.mcp.register_server(first, actor="test", require_capability=False)

        approved = approve_scaffold_candidate(
            adapter,
            first,
            confirmation=_confirmation(),
        )
        assert approved["metadata"]["agent_libos_dx_review"]["status"] == "reviewed"
        runtime.mcp.register_server(approved, actor="test", require_capability=False)
        assert runtime.mcp.inspect_server("scaffold-demo", require_capability=False)["server_id"] == "scaffold-demo"
    finally:
        runtime.close()


def test_scaffold_preserves_all_four_catalogs_and_exact_probe_binding() -> None:
    runtime = Runtime.open("local")
    adapter = McpDxManagerAdapter(runtime.mcp)
    base = _base_manifest("full-catalog")
    digest = _manifest_digest(adapter, base)
    try:
        with pytest.raises(ValidationError, match="exact probe manifest"):
            scaffold_manifest_candidate(
                adapter,
                base,
                [],
                probe_manifest_sha256="0" * 64,
                confirmation=_confirmation(),
            )

        candidate = scaffold_manifest_candidate(
            adapter,
            base,
            [
                {
                    "name": "demo.echo",
                    "description": "echo",
                    "input_schema": {"type": "object"},
                }
            ],
            live_resources=[
                {
                    "resource_id": "demo://status",
                    "name": "demo-status",
                    "title": None,
                    "description": "status",
                    "mime_type": "application/json",
                    "size": None,
                    "annotations": None,
                    "metadata": {},
                }
            ],
            live_resource_templates=[
                {
                    "template_id": "demo://greeting/{name}",
                    "name": "demo-greeting",
                    "title": None,
                    "description": "greeting",
                    "mime_type": "text/plain",
                    "annotations": None,
                    "metadata": {},
                }
            ],
            live_prompts=[
                {
                    "prompt_id": "demo.review",
                    "name": "demo.review",
                    "title": None,
                    "description": "review",
                    "arguments": [{"name": "subject", "required": True}],
                    "metadata": {},
                }
            ],
            probe_manifest_sha256=digest,
            confirmation=_confirmation(),
        )

        manifest = candidate["manifest"]
        assert manifest["tools"][0]["right"] == "execute"
        assert manifest["resources"][0]["remote_uri"] == "demo://status"
        assert manifest["resources"][0]["model_visible"] is False
        assert manifest["resource_templates"][0]["variables"] == ["name"]
        assert manifest["prompts"][0]["mcp_name"] == "demo.review"
        assert manifest["prompts"][0]["argument_names"] == ["subject"]
        assert candidate["source_manifest_sha256"] == digest
    finally:
        runtime.close()


def test_export_plan_and_single_import_are_digest_bound() -> None:
    source = Runtime.open("local")
    target = Runtime.open("local")
    try:
        source.mcp.register_server_from_yaml_text(
            _manifest("exported"),
            actor="test",
            require_capability=False,
        )
        bundle = export_registry_bundle(
            McpDxManagerAdapter(source.mcp),
            server_ids=["exported"],
        )
        encoded = json.dumps(bundle, sort_keys=True)
        assert "AGENT_LIBOS_MCP_DX_TEST_TOKEN" in encoded
        assert "MCP_DX_SECRET_MUST_NOT_APPEAR" not in encoded

        target_adapter = McpDxManagerAdapter(target.mcp)
        plan = plan_import_bundle(target_adapter, bundle)
        assert [(item.server_id, item.action) for item in plan.actions] == [
            ("exported", "create")
        ]
        assert plan.atomic_apply_supported

        result = import_one_from_bundle(
            target_adapter,
            bundle,
            server_id="exported",
            confirmation=_confirmation(),
        )
        assert result["action"] == "create"
        assert plan_import_bundle(target_adapter, bundle).actions[0].action == "unchanged"

        tampered = json.loads(json.dumps(bundle))
        tampered["servers"][0]["timeout_s"] = 11
        with pytest.raises(ValidationError, match="digest"):
            plan_import_bundle(target_adapter, tampered)
    finally:
        target.close()
        source.close()


def test_probe_requires_governed_adapter_and_manifest_digest_binding() -> None:
    runtime = Runtime.open("local")
    adapter = McpDxManagerAdapter(runtime.mcp)

    class FakeGovernedProbe:
        def probe(self, spec: Any, *, confirmation: McpDxConfirmation) -> McpProbeReport:
            confirmation.require("fake governed probe")
            digest = hashlib.sha256(
                canonical_mcp_server_spec_json(spec).encode("utf-8")
            ).hexdigest()
            return McpProbeReport(
                manifest_sha256=digest,
                catalog_scope="full_catalog",
                complete=True,
                tools=(
                    {
                        "name": "demo.echo",
                        "description": "echo",
                        "input_schema": {"type": "object"},
                        "metadata": {},
                    },
                ),
                diagnostics={"provider_started": True, "stage": "tools/list"},
            )

    try:
        report = probe_manifest(
            adapter,
            _manifest(),
            probe_adapter=FakeGovernedProbe(),
            confirmation=_confirmation(),
        )
        assert report.complete
        assert report.catalog_scope == "full_catalog"
        assert report.tools[0]["name"] == "demo.echo"
    finally:
        runtime.close()


def test_v3_validate_doctor_and_cas_import_are_offline_until_apply() -> None:
    runtime = Runtime.open("local")
    adapter = McpDxManagerAdapter(runtime.mcp)
    manifest_text = Path("examples/mcp/stdio-v3.yaml").read_text(encoding="utf-8")
    try:
        report = validate_manifest_text(adapter, manifest_text)

        assert report.schema_version == 3
        assert report.protocol_mode == "2026-07-28"
        assert report.tool_count == 1
        assert report.resource_count == 1
        assert report.resource_template_count == 1
        assert report.prompt_count == 1
        assert report.stdio_authority_resource is not None

        doctor = doctor_manifest_text(adapter, manifest_text, environment={})
        assert doctor.ready_for_registration is True
        assert any(
            item.code == "registry.v3_cas_bridge" and item.status == "ok"
            for item in doctor.checks
        )

        manifest = json.loads(
            canonical_mcp_v3_manifest_json(
                parse_mcp_v3_manifest_yaml_text(manifest_text)
            )
        )
        unsigned = {
            "kind": "agent-libos.mcp-registry-export",
            "schema_version": 1,
            "secrets_included": False,
            "servers": [manifest],
        }
        bundle = {
            **unsigned,
            "bundle_sha256": hashlib.sha256(
                dumps(unsigned).encode("utf-8")
            ).hexdigest(),
        }
        plan = plan_import_bundle(adapter, bundle)
        assert plan.actions[0].action == "create"

        result = import_one_from_bundle(
            adapter,
            bundle,
            server_id="demo-stdio-v3",
            confirmation=_confirmation(),
        )
        assert result["action"] == "create"
        assert [
            item["server_id"]
            for item in runtime.mcp.list_servers(require_capability=False)
        ] == ["demo-stdio-v3"]
    finally:
        runtime.close()
