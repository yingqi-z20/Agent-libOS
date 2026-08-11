from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_yaml_text,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.mcp import canonical_mcp_server_spec_json


_EXAMPLE = Path("examples/mcp/stdio-v3.yaml")


def _manifest(server_id: str = "registry-v3") -> McpServerManifestV3:
    selected = parse_mcp_v3_manifest_yaml_text(_EXAMPLE.read_text(encoding="utf-8"))
    return replace(selected, server_id=server_id)


def _legacy_manifest(server_id: str = "registry-v1") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server_id": server_id,
        "transport": "stdio",
        "stdio": {"command": "python3", "args": ["-m", "demo_server"]},
        "tools": [
            {
                "tool_id": "echo",
                "mcp_name": "demo.echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "input_schema": {},
            }
        ],
    }


def test_v3_registry_round_trips_with_legacy_rows_and_reopen(tmp_path: Path) -> None:
    target = tmp_path / "mcp-v3.sqlite"
    runtime = Runtime.open(target)
    try:
        legacy = runtime.mcp._coerce_server(_legacy_manifest())
        legacy_canonical = canonical_mcp_server_spec_json(legacy)
        runtime.mcp.register_server(
            legacy,
            actor="test",
            require_capability=False,
        )
        result = runtime.mcp.register_server_from_yaml_text(
            _EXAMPLE.read_text(encoding="utf-8"),
            actor="test",
            require_capability=False,
        )

        assert result["schema_version"] == 3
        assert result["protocol_mode"] == "2026-07-28"
        assert {
            "resources",
            "resource_templates",
            "prompts",
            "auth_profile_id",
            "subscriptions",
            "tasks_extension",
        }.isdisjoint(result)
        assert "demo://status" not in json.dumps(result)
        listed = runtime.mcp.list_servers(require_capability=False)
        assert [item["server_id"] for item in listed] == [
            "demo-stdio-v3",
            "registry-v1",
        ]
        assert "demo://status" not in json.dumps(listed)
        host_projection = runtime.mcp.inspect_server(
            "demo-stdio-v3",
            require_capability=False,
            include_sensitive_fields=True,
        )
        assert host_projection["resources"][0]["resource_id"] == "status"
        assert host_projection["resources"][0]["remote_uri"] == "demo://status"
        assert host_projection["resource_templates"][0]["template_id"] == "greeting"
        assert host_projection["prompts"][0]["prompt_id"] == "review"
        assert canonical_mcp_server_spec_json(
            runtime.mcp._load_server("registry-v1")[0]
        ) == legacy_canonical
    finally:
        runtime.close()

    reopened = Runtime.open(target)
    try:
        selected, _metadata = reopened.mcp._load_server("demo-stdio-v3")
        assert isinstance(selected, McpServerManifestV3)
        assert canonical_mcp_v3_manifest_json(selected) == (
            canonical_mcp_v3_manifest_json(
                parse_mcp_v3_manifest_yaml_text(
                    _EXAMPLE.read_text(encoding="utf-8")
                )
            )
        )
    finally:
        reopened.close()


def test_v3_import_requires_exact_registry_digest_without_failed_evidence() -> None:
    runtime = Runtime.open(":memory:")
    try:
        original = _manifest()
        runtime.mcp.import_v3_manifest(
            original,
            actor="test",
            replace=False,
            require_capability=False,
            expected_current_sha256=None,
        )
        current_sha256 = runtime.mcp._server_spec_sha256(original)
        replacement = replace(original, metadata={"revision": 2})
        before_events = runtime.events.list()
        before_audit = runtime.audit.trace()

        with pytest.raises(ValidationError, match="changed after import planning"):
            runtime.mcp.import_v3_manifest(
                replacement,
                actor="test",
                replace=True,
                require_capability=False,
                expected_current_sha256="0" * 64,
            )

        assert runtime.events.list() == before_events
        assert runtime.audit.trace() == before_audit
        assert runtime.mcp._server_spec_sha256(
            runtime.mcp._load_server(original.server_id)[0]
        ) == current_sha256

        runtime.mcp.import_v3_manifest(
            replacement,
            actor="test",
            replace=True,
            require_capability=False,
            expected_current_sha256=current_sha256,
        )
        assert runtime.mcp.inspect_server(
            original.server_id,
            require_capability=False,
        )["metadata"] == {"revision": 2}
    finally:
        runtime.close()


def test_v3_import_cas_serializes_same_runtime_race() -> None:
    runtime = Runtime.open(":memory:")
    try:
        original = _manifest("cas-race")
        runtime.mcp.import_v3_manifest(
            original,
            actor="test",
            replace=False,
            require_capability=False,
            expected_current_sha256=None,
        )
        expected = runtime.mcp._server_spec_sha256(original)
        barrier = threading.Barrier(3)
        outcomes: list[str] = []
        outcome_lock = threading.Lock()

        def replace_once(revision: int) -> None:
            candidate = replace(original, metadata={"revision": revision})
            barrier.wait()
            try:
                runtime.mcp.import_v3_manifest(
                    candidate,
                    actor=f"test-{revision}",
                    replace=True,
                    require_capability=False,
                    expected_current_sha256=expected,
                )
            except ValidationError:
                outcome = "stale"
            else:
                outcome = "applied"
            with outcome_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=replace_once, args=(revision,), daemon=True)
            for revision in (1, 2)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert sorted(outcomes) == ["applied", "stale"]
        assert runtime.mcp.inspect_server(
            original.server_id,
            require_capability=False,
        )["metadata"]["revision"] in {1, 2}
    finally:
        runtime.close()


def test_v3_replace_and_unregister_trigger_modern_invalidation() -> None:
    runtime = Runtime.open(":memory:")
    server_ids: list[str] = []
    supervisor = runtime._mcp_connection_supervisor
    original_invalidate = supervisor.invalidate_server_nowait

    def observed_invalidate(server_id: str, **keywords: object) -> None:
        server_ids.append(server_id)
        original_invalidate(server_id, **keywords)

    supervisor.invalidate_server_nowait = observed_invalidate
    try:
        original = _manifest("invalidate-v3")
        runtime.mcp.register_server(
            original,
            actor="test",
            require_capability=False,
        )
        runtime.mcp.register_server(
            replace(original, metadata={"revision": 2}),
            actor="test",
            replace=True,
            require_capability=False,
        )
        runtime.mcp.unregister_server(
            original.server_id,
            actor="test",
            require_capability=False,
        )

        assert server_ids == [original.server_id, original.server_id]
    finally:
        runtime.close()


def test_v3_host_policy_rejection_has_no_registry_or_evidence_side_effect() -> None:
    manifest = json.loads(canonical_mcp_v3_manifest_json(_manifest("bad-env")))
    manifest["stdio"]["env"] = {"TOKEN": "UNSCOPED_HOST_SECRET"}
    runtime = Runtime.open(":memory:")
    try:
        before_events = runtime.events.list()
        before_audit = runtime.audit.trace()

        with pytest.raises(ValidationError, match="not allowlisted"):
            runtime.mcp.register_server(
                manifest,
                actor="test",
                require_capability=False,
            )

        assert runtime.mcp.list_servers(require_capability=False) == []
        assert runtime.events.list() == before_events
        assert runtime.audit.trace() == before_audit
    finally:
        runtime.close()
