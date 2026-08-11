from __future__ import annotations

import hashlib
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import agent_libos.primitives.mcp as mcp_primitive_module

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
from agent_libos.models.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "mcp"


def _manifest() -> McpServerManifestV3:
    selected = parse_mcp_v3_manifest_yaml_text(
        (EXAMPLES / "stdio-v3.yaml").read_text(encoding="utf-8")
    )
    assert selected.stdio is not None
    return replace(
        selected,
        stdio=replace(
            selected.stdio,
            command=sys.executable,
            args=[str(EXAMPLES / "stdio_server.py")],
        ),
    )


def _digest(manifest: McpServerManifestV3) -> str:
    return hashlib.sha256(
        canonical_mcp_v3_manifest_json(manifest).encode("utf-8")
    ).hexdigest()


def test_candidate_probe_is_pending_first_single_session_and_authority_neutral() -> None:
    runtime = Runtime.open(":memory:")
    manifest = _manifest()
    manifest_text = canonical_mcp_v3_manifest_json(manifest)
    manifest_sha256 = _digest(manifest)
    confirmation = McpDxConfirmation(
        confirmed=True,
        actor="candidate-reviewer",
        reason="review all four unregistered catalogs",
    )
    provider = runtime.mcp.provider
    original_session = provider.modern_session
    observed: dict[str, Any] = {"sessions": 0, "snapshot_sha256": None}

    @asynccontextmanager
    async def guarded_session(*args: Any, **kwargs: Any):
        observed["sessions"] += 1
        pending = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp"
            and effect.operation == "probe_candidate"
            and effect.effect_state == "pending"
        ]
        assert len(pending) == 1
        snapshot = kwargs.get("executable_snapshot")
        assert snapshot is not None
        observed["snapshot_sha256"] = snapshot.content_sha256
        async with original_session(*args, **kwargs) as session:
            yield session

    provider.modern_session = guarded_session
    try:
        servers_before = runtime.mcp.list_servers(require_capability=False)
        capabilities_before = runtime.store.list_capabilities()
        adapter = McpDxManagerAdapter(runtime.mcp)
        report = probe_manifest(
            adapter,
            manifest_text,
            probe_adapter=CandidateMcpProbeAdapter(adapter),
            confirmation=confirmation,
        )
        candidate = scaffold_manifest_candidate(
            adapter,
            json.loads(manifest_text),
            report.tools,
            live_resources=report.resources,
            live_resource_templates=report.resource_templates,
            live_prompts=report.prompts,
            probe_manifest_sha256=report.manifest_sha256,
            confirmation=confirmation,
            catalog_scope=report.catalog_scope,
            complete=report.complete,
        )

        assert observed["sessions"] == 1
        assert isinstance(observed["snapshot_sha256"], str)
        assert report.manifest_sha256 == manifest_sha256
        assert tuple(
            map(len, (report.tools, report.resources, report.resource_templates, report.prompts))
        ) == (1, 1, 1, 1)
        assert candidate["source_manifest_sha256"] == manifest_sha256
        assert runtime.mcp.list_servers(require_capability=False) == servers_before == []
        assert runtime.store.list_capabilities() == capabilities_before == []

        effects = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "probe_candidate"
        ]
        assert len(effects) == 1
        assert effects[0].effect_state == "finalized"
        assert effects[0].transaction_state == "committed"
        assert effects[0].target == f"mcp_candidate:{manifest_sha256}"
        audits = [
            row
            for row in runtime.audit.trace()
            if row.action == "primitive.mcp.probe_candidate"
        ]
        assert len(audits) == 1
        assert audits[0].actor == "mcp-dx-probe"
        assert audits[0].decision["reviewer"] == confirmation.actor
        assert audits[0].decision["reason"] == confirmation.reason
        events = [
            row
            for row in runtime.events.list()
            if row.source == "mcp-dx-probe" and row.target == effects[0].target
        ]
        assert len(events) == 1

        # The first actual registry mutation must still advance generation from
        # its untouched initial value to exactly one.
        runtime.mcp.register_server(
            manifest,
            actor="candidate-test",
            require_capability=False,
        )
        binding = runtime.uow.extensions.get_mcp_registry_binding(manifest.server_id)
        assert binding["registry_generation"] == 1
    finally:
        provider.modern_session = original_session
        runtime.close()


def test_candidate_probe_stdio_fingerprint_deadline_precedes_pending_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(":memory:")
    manifest = replace(_manifest(), timeout_s=0.01)
    observed_deadlines: list[float | None] = []

    def expired_fingerprint(
        _path: Any,
        *,
        deadline: float | None = None,
    ) -> str:
        observed_deadlines.append(deadline)
        raise TimeoutError("candidate fingerprint deadline")

    monkeypatch.setattr(
        mcp_primitive_module,
        "executable_content_sha256",
        expired_fingerprint,
    )
    try:
        with pytest.raises(TimeoutError, match="candidate fingerprint"):
            runtime.mcp.probe_candidate_manifest(
                manifest,
                expected_manifest_sha256=_digest(manifest),
                confirmed=True,
                reviewer="deadline-reviewer",
                reason="verify candidate fingerprint deadline",
            )
        assert len(observed_deadlines) == 1
        assert isinstance(observed_deadlines[0], float)
        assert not [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "probe_candidate"
        ]
    finally:
        runtime.close()


def test_registered_v3_stdio_revalidation_reuses_deadline_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(":memory:")
    manifest = replace(_manifest(), timeout_s=1.0)
    runtime.mcp.register_server(
        manifest,
        actor="deadline-test",
        require_capability=False,
    )
    actual_fingerprint = mcp_primitive_module.executable_content_sha256
    observed_deadlines: list[float | None] = []

    def fingerprint(
        path: Any,
        *,
        deadline: float | None = None,
    ) -> str:
        observed_deadlines.append(deadline)
        if len(observed_deadlines) == 2:
            raise TimeoutError("revalidation fingerprint deadline")
        return actual_fingerprint(path, deadline=deadline)

    def unexpected_provider(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("provider ran after stdio fingerprint timeout")

    monkeypatch.setattr(
        mcp_primitive_module,
        "executable_content_sha256",
        fingerprint,
    )
    monkeypatch.setattr(
        runtime.mcp._modern_client,
        "list_resources",
        unexpected_provider,
    )
    before = tuple(runtime.store.list_external_effects())
    try:
        with pytest.raises(TimeoutError, match="revalidation fingerprint"):
            runtime.mcp.list_resources(manifest.server_id, actor="gui")
        assert len(observed_deadlines) == 2
        assert observed_deadlines[0] == observed_deadlines[1]
        assert isinstance(observed_deadlines[0], float)
        assert tuple(runtime.store.list_external_effects()) == before
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("confirmed", "reviewer", "reason", "digest"),
    [
        (False, "reviewer", "reason", None),
        (True, "", "reason", None),
        (True, "reviewer", "", None),
        (True, "reviewer", "reason", "0" * 64),
    ],
)
def test_candidate_probe_rejects_review_or_digest_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    confirmed: bool,
    reviewer: str,
    reason: str,
    digest: str | None,
) -> None:
    runtime = Runtime.open(":memory:")
    manifest = _manifest()
    touched = {"dns": 0, "session": 0, "stdio": 0}

    def unexpected(name: str):
        def selected(*_args: Any, **_kwargs: Any) -> Any:
            touched[name] += 1
            raise AssertionError(f"unexpected candidate transport touch: {name}")

        return selected

    monkeypatch.setattr(runtime.mcp, "_validate_runtime_resolution", unexpected("dns"))
    monkeypatch.setattr(runtime.mcp.provider, "modern_session", unexpected("session"))
    monkeypatch.setattr(
        runtime.mcp.provider,
        "resolve_stdio_executable",
        unexpected("stdio"),
    )
    try:
        with pytest.raises(ValidationError):
            runtime.mcp.probe_candidate_manifest(
                manifest,
                expected_manifest_sha256=digest or _digest(manifest),
                confirmed=confirmed,
                reviewer=reviewer,
                reason=reason,
            )
        assert touched == {"dns": 0, "session": 0, "stdio": 0}
        assert runtime.mcp.list_servers(require_capability=False) == []
        assert runtime.store.list_capabilities() == []
        assert not [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.provider == "mcp" and effect.operation == "probe_candidate"
        ]
        assert not [
            row
            for row in runtime.audit.trace()
            if row.action == "primitive.mcp.probe_candidate"
        ]
    finally:
        runtime.close()
