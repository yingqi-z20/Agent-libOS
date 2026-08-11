#!/usr/bin/env python3
"""Probe unregistered v3 stdio/HTTP candidates and scaffold four allowlists."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from agent_libos import Runtime
from agent_libos.mcp.dx import (
    CandidateMcpProbeAdapter,
    McpDxConfirmation,
    McpDxManagerAdapter,
    probe_manifest,
    scaffold_manifest_candidate,
)
from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_yaml_text,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "mcp"


def main() -> int:
    port = _free_loopback_port()
    http_process = _start_http_server(port)
    runtime = Runtime.open(":memory:")
    try:
        _wait_for_loopback(port, http_process)
        stdio = _load_manifest(EXAMPLE_ROOT / "stdio-v3.yaml")
        http = _load_manifest(EXAMPLE_ROOT / "http-v3.yaml")
        assert stdio.stdio is not None and http.http is not None
        stdio = replace(
            stdio,
            stdio=replace(
                stdio.stdio,
                command=sys.executable,
                args=[str(EXAMPLE_ROOT / "stdio_server.py")],
            ),
        )
        http = replace(
            http,
            http=replace(http.http, url=f"http://127.0.0.1:{port}/mcp"),
        )
        confirmation = McpDxConfirmation(
            confirmed=True,
            actor="examples.mcp.onboarding-reviewer",
            reason="deterministic full-catalog onboarding review",
        )
        adapter = McpDxManagerAdapter(runtime.mcp)
        before = runtime.mcp.list_servers(require_capability=False)
        summaries: dict[str, object] = {}
        for transport, manifest in (("stdio", stdio), ("streamable_http", http)):
            text = canonical_mcp_v3_manifest_json(manifest)
            report = probe_manifest(
                adapter,
                text,
                probe_adapter=CandidateMcpProbeAdapter(adapter),
                confirmation=confirmation,
            )
            candidate = scaffold_manifest_candidate(
                adapter,
                json.loads(text),
                report.tools,
                live_resources=report.resources,
                live_resource_templates=report.resource_templates,
                live_prompts=report.prompts,
                probe_manifest_sha256=report.manifest_sha256,
                confirmation=confirmation,
                catalog_scope=report.catalog_scope,
                complete=report.complete,
            )
            selected = candidate["manifest"]
            summaries[transport] = {
                "manifest_sha256": report.manifest_sha256,
                "catalog_scope": report.catalog_scope,
                "complete": report.complete,
                "counts": {
                    "tools": len(selected["tools"]),
                    "resources": len(selected["resources"]),
                    "resource_templates": len(selected["resource_templates"]),
                    "prompts": len(selected["prompts"]),
                },
                "candidate_manifest_sha256": candidate["manifest_sha256"],
                "review_required": candidate["review"]["required"],
            }
        after = runtime.mcp.list_servers(require_capability=False)
        effects = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "probe_candidate"
        ]
        audits = [
            row
            for row in runtime.audit.trace()
            if row.action == "primitive.mcp.probe_candidate"
        ]
        if before or after:
            raise RuntimeError("candidate probe or scaffold mutated the MCP registry")
        if len(effects) != 2 or len(audits) != 2:
            raise RuntimeError("candidate probes did not emit complete protected evidence")
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_servers_before": len(before),
                    "registry_servers_after": len(after),
                    "protected_effects": [
                        {
                            "effect_state": effect.effect_state,
                            "transaction_state": effect.transaction_state,
                            "operation": effect.operation,
                        }
                        for effect in effects
                    ],
                    "audit_reviews": [
                        {
                            "actor": row.actor,
                            "reviewer": row.decision.get("reviewer"),
                            "reason": row.decision.get("reason"),
                        }
                        for row in audits
                    ],
                    "transports": summaries,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.close()
        _stop_process(http_process)


def _load_manifest(path: Path) -> McpServerManifestV3:
    return parse_mcp_v3_manifest_yaml_text(path.read_text(encoding="utf-8"))


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
        selected.bind(("127.0.0.1", 0))
        return int(selected.getsockname()[1])


def _start_http_server(port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(EXAMPLE_ROOT / "http_server.py"), "--port", str(port)],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _wait_for_loopback(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("loopback MCP demo exited before accepting connections")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as selected:
            selected.settimeout(0.1)
            if selected.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise RuntimeError("loopback MCP demo did not start within 15 seconds")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
