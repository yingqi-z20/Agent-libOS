"""Safe Host-side MCP manifest and registry developer experience helpers.

These helpers are deliberately not model tools.  They validate and prepare
Host-owned configuration, but only the MCP primitive may register a server or
cross a provider boundary.  The one probe entry point therefore requires an
explicit adapter whose contract says that it uses the governed Runtime path.

The adapter around ``Runtime.mcp`` isolates the legacy v1/v2 private
compatibility calls in this module. Manifest v3 uses the public strict parser
and exact-CAS import bridge; a non-Runtime adapter may validate it offline but
cannot report registration readiness or apply it without that bridge.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_libos.mcp.manifest import (
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
    parse_mcp_v3_manifest_mapping,
)
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.models.mcp import (
    McpProviderTool,
    McpServerSpec,
    canonical_mcp_server_spec_json,
    mcp_server_spec_to_jsonable,
)
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable
from agent_libos.utils.yaml_loader import YAML_MAX_UTF8_BYTES, load_yaml_mapping


MCP_DX_CANDIDATE_KIND = "agent-libos.mcp-manifest-candidate"
MCP_DX_BUNDLE_KIND = "agent-libos.mcp-registry-export"
MCP_DX_FORMAT_VERSION = 1
MCP_DX_IMPORT_MAX_BYTES = 8 * 1_048_576
_ID_MAX_CHARS_FALLBACK = 96
_MAX_PUBLIC_DIAGNOSTIC_CHARS = 256
_SAFE_ID_CHARACTER = re.compile(r"[^A-Za-z0-9_.@+-]+")

McpDxManifest = McpServerSpec | McpServerManifestV3


class McpDxConfirmationRequired(ValidationError):
    """Raised when an external probe or candidate transition lacks approval."""


@dataclass(frozen=True)
class McpDxConfirmation:
    """Explicit Host confirmation for a DX operation.

    ``actor`` and ``reason`` are evidence carried by the caller/adapter.  They
    are not capabilities and never bypass Runtime authority checks.
    """

    confirmed: bool
    actor: str
    reason: str

    def require(self, operation: str) -> None:
        if self.confirmed is not True:
            raise McpDxConfirmationRequired(
                f"MCP DX {operation} requires explicit Host confirmation"
            )
        if type(self.actor) is not str or not self.actor.strip():
            raise ValidationError("MCP DX confirmation actor must be non-empty")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValidationError("MCP DX confirmation reason must be non-empty")


@dataclass(frozen=True)
class McpManifestValidationReport:
    server_id: str
    schema_version: int
    protocol_mode: str
    transport: str
    tool_count: int
    resource_count: int
    resource_template_count: int
    prompt_count: int
    manifest_sha256: str
    stdio_authority_resource: str | None

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class McpDoctorCheck:
    code: str
    status: str
    detail: str


@dataclass(frozen=True)
class McpDoctorReport:
    validation: McpManifestValidationReport
    ready_for_registration: bool
    ready_for_live_probe: bool
    checks: tuple[McpDoctorCheck, ...]
    environment: tuple[dict[str, Any], ...]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class McpProbeReport:
    """Bounded output from a governed Host probe adapter."""

    manifest_sha256: str
    catalog_scope: str
    complete: bool
    tools: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    resources: tuple[dict[str, Any], ...] = ()
    resource_templates: tuple[dict[str, Any], ...] = ()
    prompts: tuple[dict[str, Any], ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class McpImportAction:
    server_id: str
    action: str
    current_sha256: str | None
    proposed_sha256: str


@dataclass(frozen=True)
class McpImportPlan:
    bundle_sha256: str
    actions: tuple[McpImportAction, ...]
    requires_confirmation: bool
    atomic_apply_supported: bool

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class McpDxProbeAdapter(Protocol):
    """Adapter for a *governed* provider probe.

    Implementations must enter the Runtime MCP protected-operation path,
    preserve its authority/data-flow/effect/resource checks, and must not call
    an SDK provider directly.  ``full_catalog`` is the only scope accepted by
    scaffold generation.
    """

    def probe(
        self,
        spec: McpDxManifest,
        *,
        confirmation: McpDxConfirmation,
    ) -> McpProbeReport: ...


class McpDxManagerAdapter:
    """Narrow compatibility adapter around the current Runtime MCP manager."""

    def __init__(self, manager: Any) -> None:
        if manager is None:
            raise TypeError("MCP DX manager is required")
        self.manager = manager

    @property
    def manifest_max_bytes(self) -> int:
        return int(self.manager.config.mcp.manifest_max_bytes)

    @property
    def server_page_limit(self) -> int:
        return int(self.manager.config.mcp.server_page_limit)

    @property
    def tool_catalog_limit(self) -> int:
        return int(self.manager.config.mcp.tool_catalog_limit)

    @property
    def resource_catalog_limit(self) -> int:
        return int(self.manager.config.mcp.resource_catalog_limit)

    @property
    def resource_template_limit(self) -> int:
        return int(self.manager.config.mcp.resource_template_limit)

    @property
    def prompt_catalog_limit(self) -> int:
        return int(self.manager.config.mcp.prompt_catalog_limit)

    @property
    def tool_id_max_chars(self) -> int:
        return int(
            getattr(
                self.manager.config.mcp,
                "tool_id_max_chars",
                _ID_MAX_CHARS_FALLBACK,
            )
        )

    @property
    def supports_v3_import(self) -> bool:
        return callable(getattr(self.manager, "import_v3_manifest", None))

    def coerce_mapping(self, value: Mapping[str, Any]) -> McpDxManifest:
        validate = getattr(self.manager, "validate_server_manifest", None)
        if not callable(validate):
            if value.get("schema_version") == 3:
                # Structural-only fallback for a non-Runtime adapter.  Such an
                # adapter cannot report registration readiness or apply.
                return parse_mcp_v3_manifest_mapping(value)
            raise ValidationError(
                "MCP manager does not expose the manifest-validation adapter"
            )
        return validate(dict(value))

    def load_server(self, server_id: str) -> McpDxManifest:
        load = getattr(self.manager, "get_server_manifest", None)
        if not callable(load):
            raise ValidationError("MCP manager does not expose the registry adapter")
        return load(server_id)

    def list_server_ids(self) -> tuple[str, ...]:
        rows, has_more = self.manager.list_servers_window(
            actor=None,
            require_capability=False,
            limit=self.server_page_limit,
        )
        if has_more:
            raise ValidationError(
                "MCP registry export is incomplete at the configured list limit; "
                "supply explicit server ids"
            )
        ids: list[str] = []
        for row in rows:
            server_id = row.get("server_id") if isinstance(row, dict) else None
            if type(server_id) is not str or not server_id:
                raise ValidationError("MCP registry returned an invalid server id")
            ids.append(server_id)
        return tuple(ids)

    def stdio_resource(self, spec: McpDxManifest) -> str | None:
        select = getattr(self.manager, "stdio_resource_for_server", None)
        if not callable(select):
            raise ValidationError("MCP manager does not expose stdio authority derivation")
        return select(spec)

    def register_import(
        self,
        spec: McpDxManifest,
        *,
        actor: str,
        replace: bool,
        require_capability: bool,
        expected_current_sha256: str | None,
    ) -> dict[str, Any]:
        """Register one import under a registry digest CAS."""

        apply = getattr(self.manager, "import_server_manifest", None)
        if not callable(apply):
            raise ValidationError(
                "MCP apply requires the public Host CAS bridge "
                "manager.import_server_manifest(..., expected_current_sha256=...); "
                "import planning remains available without it"
            )
        if isinstance(spec, McpServerManifestV3) and not self.supports_v3_import:
            raise ValidationError(
                "MCP Manifest v3 apply requires the exact Host CAS bridge; "
                "import planning remains available without it"
            )
        return apply(
            spec,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            expected_current_sha256=expected_current_sha256,
            source=f"{MCP_DX_BUNDLE_KIND}/v{MCP_DX_FORMAT_VERSION}",
        )


class RegisteredMcpProbeAdapter:
    """Governed probe for an already registered allowlist.

    This adapter is useful for health/schema-drift diagnostics.  Its catalog
    scope is deliberately ``registered_allowlist`` and therefore cannot feed a
    scaffold: the primitive intentionally hides undeclared live Tools.
    """

    def __init__(
        self,
        adapter: McpDxManagerAdapter,
        server_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = False,
    ) -> None:
        self.adapter = adapter
        self.server_id = server_id
        self.actor = actor
        self.require_capability = require_capability

    def probe(
        self,
        spec: McpDxManifest,
        *,
        confirmation: McpDxConfirmation,
    ) -> McpProbeReport:
        confirmation.require("registered-server probe")
        current = self.adapter.load_server(self.server_id)
        if _spec_sha256(current) != _spec_sha256(spec):
            raise ValidationError(
                "MCP registered probe manifest does not match the current registry"
            )
        result = self.adapter.manager.list_tools(
            self.server_id,
            actor=self.actor,
            require_capability=self.require_capability,
            refresh=True,
        )
        if not isinstance(result, dict) or result.get("refreshed") is not True:
            raise ValidationError("MCP registered probe did not perform a live refresh")
        selected: list[dict[str, Any]] = []
        complete = True
        raw_tools = result.get("tools")
        if not isinstance(raw_tools, list):
            raise ValidationError("MCP registered probe returned an invalid Tool list")
        for item in raw_tools:
            if not isinstance(item, dict):
                raise ValidationError("MCP registered probe Tool entry is invalid")
            live = item.get("live")
            if not isinstance(live, dict):
                complete = False
                continue
            selected.append(
                {
                    "name": live.get("name"),
                    "description": live.get("description"),
                    "input_schema": live.get("input_schema", {}),
                    "metadata": {},
                }
            )
        tools = _normalize_probe_tools(
            selected,
            limit=self.adapter.tool_catalog_limit,
        )
        diagnostics = _bounded_diagnostics(
            {
                "stage": "tools/list",
                "provider_started": True,
                "response_bytes": result.get("response_bytes", 0),
                "schema_version": result.get("schema_version", spec.schema_version),
                "protocol_mode": result.get("protocol_mode", "legacy"),
            }
        )
        return McpProbeReport(
            manifest_sha256=_spec_sha256(spec),
            catalog_scope="registered_allowlist",
            complete=complete,
            tools=tuple(tools),
            diagnostics=diagnostics,
        )


class CandidateMcpProbeAdapter:
    """Host-only full-catalog probe for an unregistered v3 candidate.

    The manager owns the protected operation, transport snapshot, deadline,
    SSRF/stdio validation, and evidence.  This adapter only converts its typed
    detached catalog into the bounded DX report; it never opens an SDK session
    or mutates the registry.
    """

    def __init__(self, adapter: McpDxManagerAdapter) -> None:
        self.adapter = adapter

    def probe(
        self,
        spec: McpDxManifest,
        *,
        confirmation: McpDxConfirmation,
    ) -> McpProbeReport:
        confirmation.require("unregistered candidate probe")
        if not isinstance(spec, McpServerManifestV3):
            raise ValidationError("MCP unregistered candidate probe requires Manifest v3")
        probe = getattr(self.adapter.manager, "probe_candidate_manifest", None)
        if not callable(probe):
            raise ValidationError(
                "MCP manager does not expose the governed candidate probe boundary"
            )
        manifest_sha256 = _spec_sha256(spec)
        raw = to_jsonable(
            probe(
                spec,
                expected_manifest_sha256=manifest_sha256,
                confirmed=confirmation.confirmed,
                reviewer=confirmation.actor,
                reason=confirmation.reason,
            )
        )
        if type(raw) is not dict:
            raise ValidationError("MCP candidate probe returned an invalid catalog")
        pages = {
            "tool_pages": raw.get("tool_pages"),
            "resource_pages": raw.get("resource_pages"),
            "resource_template_pages": raw.get("resource_template_pages"),
            "prompt_pages": raw.get("prompt_pages"),
        }
        if any(type(value) is not int or value <= 0 for value in pages.values()):
            raise ValidationError("MCP candidate probe page evidence is invalid")
        return McpProbeReport(
            manifest_sha256=manifest_sha256,
            catalog_scope="full_catalog",
            complete=True,
            tools=tuple(
                _normalize_probe_tools(
                    _required_catalog_array(raw, "tools"),
                    limit=self.adapter.tool_catalog_limit,
                )
            ),
            resources=tuple(
                _normalize_probe_resources(
                    _required_catalog_array(raw, "resources"),
                    limit=self.adapter.resource_catalog_limit,
                )
            ),
            resource_templates=tuple(
                _normalize_probe_resource_templates(
                    _required_catalog_array(raw, "resource_templates"),
                    limit=self.adapter.resource_template_limit,
                )
            ),
            prompts=tuple(
                _normalize_probe_prompts(
                    _required_catalog_array(raw, "prompts"),
                    limit=self.adapter.prompt_catalog_limit,
                )
            ),
            diagnostics=_bounded_diagnostics(
                {
                    "stage": "full_catalog",
                    "provider_started": True,
                    "schema_version": 3,
                    "protocol_mode": "2026-07-28",
                    **pages,
                }
            ),
        )


def validate_manifest_text(
    adapter: McpDxManagerAdapter,
    text: str,
) -> McpManifestValidationReport:
    spec = _parse_manifest_text(adapter, text)
    return _validation_report(adapter, spec)


def doctor_manifest_text(
    adapter: McpDxManagerAdapter,
    text: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> McpDoctorReport:
    """Run offline checks only; this never starts or contacts a server."""

    spec = _parse_manifest_text(adapter, text)
    validation = _validation_report(adapter, spec)
    selected_environment = os.environ if environment is None else environment
    checks: list[McpDoctorCheck] = []
    dependency_ready = True
    for import_name, distribution in (
        ("mcp", "mcp"),
        ("anyio", "anyio"),
        ("httpx2", "httpx2"),
        ("httpcore2", "httpcore2"),
    ):
        installed = importlib.util.find_spec(import_name) is not None
        dependency_ready = dependency_ready and installed
        version = _distribution_version(distribution) if installed else None
        checks.append(
            McpDoctorCheck(
                code=f"dependency.{import_name}",
                status="ok" if installed else "missing",
                detail=(
                    f"installed version {version}"
                    if version is not None
                    else (
                        "installed"
                        if installed
                        else "install the optional MCP dependency"
                    )
                ),
            )
        )

    environment_rows: list[dict[str, Any]] = []
    environment_ready = True
    for purpose, name in _manifest_environment_references(spec):
        present = name in selected_environment
        environment_ready = environment_ready and present
        environment_rows.append(
            {"purpose": purpose, "name": name, "present": present}
        )
    checks.append(
        McpDoctorCheck(
            code="environment.references",
            status="ok" if environment_ready else "missing",
            detail=(
                "all manifest-referenced Host variables are present"
                if environment_ready
                else "one or more manifest-referenced Host variables are absent"
            ),
        )
    )
    registry_ready = not isinstance(spec, McpServerManifestV3) or adapter.supports_v3_import
    if isinstance(spec, McpServerManifestV3):
        checks.append(
            McpDoctorCheck(
                code="registry.v3_cas_bridge",
                status="ok" if registry_ready else "missing",
                detail=(
                    "Host v3 registry CAS bridge is available"
                    if registry_ready
                    else (
                        "Host must provide manager.import_v3_manifest with an "
                        "expected_current_sha256 CAS before apply"
                    )
                ),
            )
        )
    checks.append(
        McpDoctorCheck(
            code="provider.dispatch",
            status="deferred",
            detail="offline doctor never resolves DNS, starts stdio, or opens an MCP session",
        )
    )
    return McpDoctorReport(
        validation=validation,
        ready_for_registration=registry_ready,
        ready_for_live_probe=dependency_ready and environment_ready and registry_ready,
        checks=tuple(checks),
        environment=tuple(environment_rows),
    )


def probe_manifest(
    adapter: McpDxManagerAdapter,
    text: str,
    *,
    probe_adapter: McpDxProbeAdapter,
    confirmation: McpDxConfirmation,
) -> McpProbeReport:
    """Probe through an explicitly governed adapter; never call providers here."""

    confirmation.require("probe")
    if not isinstance(probe_adapter, McpDxProbeAdapter):
        raise ValidationError("MCP probe adapter does not satisfy the governed contract")
    spec = _parse_manifest_text(adapter, text)
    expected_sha256 = _spec_sha256(spec)
    report = probe_adapter.probe(spec, confirmation=confirmation)
    if not isinstance(report, McpProbeReport):
        raise ValidationError("MCP probe adapter returned an invalid report")
    if report.manifest_sha256 != expected_sha256:
        raise ValidationError("MCP probe report is not bound to the validated manifest")
    if report.catalog_scope not in {"full_catalog", "registered_allowlist"}:
        raise ValidationError("MCP probe report catalog scope is invalid")
    if type(report.complete) is not bool:
        raise ValidationError("MCP probe report completeness is invalid")
    return McpProbeReport(
        manifest_sha256=report.manifest_sha256,
        catalog_scope=report.catalog_scope,
        complete=report.complete,
        tools=tuple(
            _normalize_probe_tools(report.tools, limit=adapter.tool_catalog_limit)
        ),
        resources=tuple(
            _normalize_probe_resources(
                report.resources,
                limit=adapter.resource_catalog_limit,
            )
        ),
        resource_templates=tuple(
            _normalize_probe_resource_templates(
                report.resource_templates,
                limit=adapter.resource_template_limit,
            )
        ),
        prompts=tuple(
            _normalize_probe_prompts(
                report.prompts,
                limit=adapter.prompt_catalog_limit,
            )
        ),
        diagnostics=_bounded_diagnostics(report.diagnostics),
    )


def scaffold_manifest_candidate(
    adapter: McpDxManagerAdapter,
    base_manifest: Mapping[str, Any],
    live_tools: Sequence[McpProviderTool | Mapping[str, Any]],
    *,
    live_resources: Sequence[Mapping[str, Any]] = (),
    live_resource_templates: Sequence[Mapping[str, Any]] = (),
    live_prompts: Sequence[Mapping[str, Any]] = (),
    probe_manifest_sha256: str,
    confirmation: McpDxConfirmation,
    catalog_scope: str = "full_catalog",
    complete: bool = True,
) -> dict[str, Any]:
    """Create a deterministic, deliberately non-registerable review bundle.

    Every generated Tool starts from the most conservative supported contract:
    execute authority, unknown rollback, and possible state mutation and
    information flow.  A reviewer must explicitly approve or edit those fields
    before extracting the nested manifest.
    """

    confirmation.require("scaffold")
    if catalog_scope != "full_catalog" or complete is not True:
        raise ValidationError(
            "MCP scaffold requires one complete full-catalog probe; a registered "
            "allowlist refresh is insufficient"
        )
    manifest = _copy_json_mapping(base_manifest, context="MCP scaffold base manifest")
    source_spec = adapter.coerce_mapping(manifest)
    source_manifest_sha256 = _spec_sha256(source_spec)
    if (
        type(probe_manifest_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", probe_manifest_sha256) is None
        or probe_manifest_sha256 != source_manifest_sha256
    ):
        raise ValidationError(
            "MCP scaffold base manifest does not match the exact probe manifest"
        )
    catalogs = _normalize_scaffold_catalogs(
        adapter,
        tools=live_tools,
        resources=live_resources,
        resource_templates=live_resource_templates,
        prompts=live_prompts,
    )
    if not any(catalogs.values()):
        raise ValidationError("MCP scaffold requires at least one live catalog item")
    _replace_manifest_catalogs(adapter, manifest, catalogs)
    # Prove that the nested manifest is currently valid before handing it to a
    # human.  The wrapper itself is not a runtime manifest and cannot be passed
    # accidentally to `mcp register`.
    spec = adapter.coerce_mapping(manifest)
    canonical_manifest = _exportable_spec(spec)
    catalog_sha256 = hashlib.sha256(dumps(catalogs).encode("utf-8")).hexdigest()
    return {
        "kind": MCP_DX_CANDIDATE_KIND,
        "schema_version": MCP_DX_FORMAT_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "manifest_sha256": _spec_sha256(spec),
        "catalog_sha256": catalog_sha256,
        "catalog_scope": "full_catalog",
        "catalog_complete": True,
        "review": {
            "required": True,
            "status": "pending",
            "generated_policy": "conservative",
            "fields": [
                "tools.*.right",
                "tools.*.rollback_class",
                "tools.*.rollback_status",
                "tools.*.state_mutation",
                "tools.*.information_flow",
                "tools.*.input_schema",
                "resources.*.right",
                "resources.*.information_flow",
                "resources.*.model_visible",
                "resources.*.mime_types",
                "resource_templates.*.right",
                "resource_templates.*.information_flow",
                "resource_templates.*.model_visible",
                "resource_templates.*.variables",
                "prompts.*.argument_names",
            ],
        },
        "manifest": canonical_manifest,
    }


def _normalize_scaffold_catalogs(
    adapter: McpDxManagerAdapter,
    *,
    tools: Sequence[McpProviderTool | Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
    resource_templates: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "tools": _normalize_probe_tools(tools, limit=adapter.tool_catalog_limit),
        "resources": _normalize_probe_resources(
            resources,
            limit=adapter.resource_catalog_limit,
        ),
        "resource_templates": _normalize_probe_resource_templates(
            resource_templates,
            limit=adapter.resource_template_limit,
        ),
        "prompts": _normalize_probe_prompts(
            prompts,
            limit=adapter.prompt_catalog_limit,
        ),
    }


def _replace_manifest_catalogs(
    adapter: McpDxManagerAdapter,
    manifest: dict[str, Any],
    catalogs: Mapping[str, list[dict[str, Any]]],
) -> None:
    for name in ("tools", "resources", "resource_templates", "prompts"):
        manifest.pop(name, None)
    manifest["tools"] = _scaffold_tools(adapter, catalogs["tools"])
    if manifest.get("schema_version") == 3:
        manifest["resources"] = _scaffold_resources(adapter, catalogs["resources"])
        manifest["resource_templates"] = _scaffold_resource_templates(
            adapter,
            catalogs["resource_templates"],
        )
        manifest["prompts"] = _scaffold_prompts(adapter, catalogs["prompts"])
    elif any(
        catalogs[name] for name in ("resources", "resource_templates", "prompts")
    ):
        raise ValidationError(
            "MCP Resources, Resource Templates, and Prompts require a Manifest v3 base"
        )


def _scaffold_tools(
    adapter: McpDxManagerAdapter,
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for tool in tools:
        tool_id = _logical_tool_id(
            tool["name"],
            selected_ids,
            max_chars=adapter.tool_id_max_chars,
        )
        selected_ids.add(tool_id)
        generated.append(
            {
                "tool_id": tool_id,
                "mcp_name": tool["name"],
                "right": "execute",
                "rollback_class": "unknown",
                "rollback_status": "unknown",
                "state_mutation": True,
                "information_flow": True,
                "input_schema": tool["input_schema"],
                "metadata": _scaffold_metadata(tool),
            }
        )
    return generated


def _scaffold_resources(
    adapter: McpDxManagerAdapter,
    resources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for resource in resources:
        resource_id = _logical_tool_id(
            resource["name"], selected_ids, max_chars=adapter.tool_id_max_chars
        )
        selected_ids.add(resource_id)
        generated.append(
            {
                "resource_id": resource_id,
                "remote_uri": resource["resource_id"],
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": _scaffold_mime_types(resource),
                "metadata": _scaffold_metadata(resource),
            }
        )
    return generated


def _scaffold_resource_templates(
    adapter: McpDxManagerAdapter,
    templates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for template in templates:
        template_id = _logical_tool_id(
            template["name"], selected_ids, max_chars=adapter.tool_id_max_chars
        )
        selected_ids.add(template_id)
        generated.append(
            {
                "template_id": template_id,
                "remote_uri_template": template["template_id"],
                "variables": _template_variables(template["template_id"]),
                "right": "read",
                "information_flow": True,
                "model_visible": False,
                "mime_types": _scaffold_mime_types(template),
                "metadata": _scaffold_metadata(template),
            }
        )
    return generated


def _scaffold_prompts(
    adapter: McpDxManagerAdapter,
    prompts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for prompt in prompts:
        prompt_id = _logical_tool_id(
            prompt["name"], selected_ids, max_chars=adapter.tool_id_max_chars
        )
        selected_ids.add(prompt_id)
        generated.append(
            {
                "prompt_id": prompt_id,
                "mcp_name": prompt["prompt_id"],
                "argument_names": [item["name"] for item in prompt["arguments"]],
                "metadata": _scaffold_metadata(prompt),
            }
        )
    return generated


def _scaffold_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "agent_libos_scaffold": {
            "candidate": True,
            "human_review_required": True,
        }
    }
    for source, target in (
        ("title", "live_title"),
        ("description", "live_description"),
    ):
        if item.get(source) is not None:
            metadata[target] = item[source]
    return metadata


def _scaffold_mime_types(item: Mapping[str, Any]) -> list[str]:
    mime_type = item.get("mime_type")
    return [mime_type] if type(mime_type) is str and mime_type else []


def _template_variables(template: str) -> list[str]:
    return sorted(set(re.findall(r"\{([A-Za-z0-9][A-Za-z0-9_.@+-]*)\}", template)))


def approve_scaffold_candidate(
    adapter: McpDxManagerAdapter,
    candidate: Mapping[str, Any],
    *,
    confirmation: McpDxConfirmation,
) -> dict[str, Any]:
    """Validate and extract a human-reviewed manifest from a candidate bundle."""

    confirmation.require("candidate approval")
    selected = _copy_json_mapping(candidate, context="MCP scaffold candidate")
    _require_exact_keys(
        selected,
        {
            "kind",
            "schema_version",
            "source_manifest_sha256",
            "manifest_sha256",
            "catalog_sha256",
            "catalog_scope",
            "catalog_complete",
            "review",
            "manifest",
        },
        context="MCP scaffold candidate",
    )
    if selected["kind"] != MCP_DX_CANDIDATE_KIND:
        raise ValidationError("MCP scaffold candidate kind is invalid")
    if selected["schema_version"] != MCP_DX_FORMAT_VERSION:
        raise ValidationError("MCP scaffold candidate schema version is unsupported")
    if selected["catalog_scope"] != "full_catalog" or selected["catalog_complete"] is not True:
        raise ValidationError("MCP scaffold candidate does not contain a complete catalog")
    for digest_name in (
        "source_manifest_sha256",
        "manifest_sha256",
        "catalog_sha256",
    ):
        if (
            type(selected.get(digest_name)) is not str
            or re.fullmatch(r"[0-9a-f]{64}", selected[digest_name]) is None
        ):
            raise ValidationError(f"MCP scaffold candidate {digest_name} is invalid")
    manifest = selected.get("manifest")
    if not isinstance(manifest, dict):
        raise ValidationError("MCP scaffold candidate manifest must be an object")
    # A reviewer may edit policy fields, which intentionally changes the
    # manifest digest.  The catalog digest and wrapper shape still identify the
    # provenance, while final validation pins the approved manifest itself.
    spec = adapter.coerce_mapping(manifest)
    approved = _exportable_spec(spec)
    metadata = approved.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        approved["metadata"] = metadata
    metadata["agent_libos_dx_review"] = {
        "status": "reviewed",
        "reviewer": confirmation.actor.strip(),
        "source_manifest_sha256": selected["source_manifest_sha256"],
        "catalog_sha256": selected["catalog_sha256"],
    }
    # Validate again because the review evidence is part of the final Sink and
    # registry identity.
    return _exportable_spec(adapter.coerce_mapping(approved))


def export_registry_bundle(
    adapter: McpDxManagerAdapter,
    *,
    server_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export canonical manifests with environment references, never values."""

    selected_ids = tuple(server_ids) if server_ids is not None else adapter.list_server_ids()
    if len(set(selected_ids)) != len(selected_ids):
        raise ValidationError("MCP export server ids must be unique")
    servers = [
        _exportable_spec(adapter.load_server(server_id))
        for server_id in sorted(selected_ids)
    ]
    unsigned = {
        "kind": MCP_DX_BUNDLE_KIND,
        "schema_version": MCP_DX_FORMAT_VERSION,
        "secrets_included": False,
        "servers": servers,
    }
    return {
        **unsigned,
        "bundle_sha256": hashlib.sha256(dumps(unsigned).encode("utf-8")).hexdigest(),
    }


def plan_import_bundle(
    adapter: McpDxManagerAdapter,
    bundle: str | bytes | Mapping[str, Any],
) -> McpImportPlan:
    selected = _parse_bundle(bundle)
    actions: list[McpImportAction] = []
    for raw in selected["servers"]:
        spec = adapter.coerce_mapping(raw)
        proposed_sha256 = _spec_sha256(spec)
        try:
            current = adapter.load_server(spec.server_id)
        except NotFound:
            current = None
        current_sha256 = _spec_sha256(current) if current is not None else None
        action = (
            "create"
            if current is None
            else "unchanged"
            if current_sha256 == proposed_sha256
            else "replace"
        )
        actions.append(
            McpImportAction(
                server_id=spec.server_id,
                action=action,
                current_sha256=current_sha256,
                proposed_sha256=proposed_sha256,
            )
        )
    actions.sort(key=lambda item: item.server_id)
    return McpImportPlan(
        bundle_sha256=selected["bundle_sha256"],
        actions=tuple(actions),
        requires_confirmation=any(item.action != "unchanged" for item in actions),
        # The current Runtime Store has an atomic single-registry mutation but
        # no batch import transaction.  Applying exactly one selected entry is
        # safe; multi-entry bundles remain plan/export artifacts.
        atomic_apply_supported=sum(item.action != "unchanged" for item in actions) <= 1,
    )


def import_one_from_bundle(
    adapter: McpDxManagerAdapter,
    bundle: str | bytes | Mapping[str, Any],
    *,
    server_id: str,
    confirmation: McpDxConfirmation,
    actor: str = "mcp-dx",
    require_capability: bool = False,
) -> dict[str, Any]:
    """Apply one CAS-bound manifest import; never perform a partial batch."""

    confirmation.require("registry import")
    selected = _parse_bundle(bundle)
    raw = next(
        (
            item
            for item in selected["servers"]
            if isinstance(item, dict) and item.get("server_id") == server_id
        ),
        None,
    )
    if raw is None:
        raise NotFound(f"MCP import bundle has no server: {server_id}")
    spec = adapter.coerce_mapping(raw)
    plan = plan_import_bundle(adapter, selected)
    action = next(item for item in plan.actions if item.server_id == server_id)
    if action.action == "unchanged":
        return {
            "server_id": server_id,
            "action": "unchanged",
            "manifest_sha256": action.proposed_sha256,
        }
    result = adapter.register_import(
        spec,
        actor=actor,
        replace=action.action == "replace",
        require_capability=require_capability,
        expected_current_sha256=action.current_sha256,
    )
    return {
        "server_id": server_id,
        "action": action.action,
        "manifest_sha256": action.proposed_sha256,
        "result": result,
    }


def _parse_manifest_text(
    adapter: McpDxManagerAdapter,
    text: str,
) -> McpDxManifest:
    if type(text) is not str:
        raise ValidationError("MCP manifest must be text")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValidationError("MCP manifest must be valid UTF-8 text") from exc
    if size > adapter.manifest_max_bytes:
        raise ValidationError(
            f"MCP manifest exceeds manifest_max_bytes={adapter.manifest_max_bytes}"
        )
    data = load_yaml_mapping(text)
    if set(data) == {"mcp_server"} and isinstance(data["mcp_server"], dict):
        data = data["mcp_server"]
    if set(data) == {"server"} and isinstance(data["server"], dict):
        data = data["server"]
    return adapter.coerce_mapping(data)


def _validation_report(
    adapter: McpDxManagerAdapter,
    spec: McpDxManifest,
) -> McpManifestValidationReport:
    mode = getattr(spec, "protocol_mode", None)
    protocol_mode = (
        str(getattr(mode, "value", mode))
        if mode is not None
        else "legacy"
    )
    return McpManifestValidationReport(
        server_id=spec.server_id,
        schema_version=spec.schema_version,
        protocol_mode=protocol_mode,
        transport=spec.transport,
        tool_count=len(getattr(spec, "tools", ())),
        resource_count=len(getattr(spec, "resources", ())),
        resource_template_count=len(getattr(spec, "resource_templates", ())),
        prompt_count=len(getattr(spec, "prompts", ())),
        manifest_sha256=_spec_sha256(spec),
        stdio_authority_resource=adapter.stdio_resource(spec),
    )


def _manifest_environment_references(
    spec: McpDxManifest,
) -> tuple[tuple[str, str], ...]:
    selected: set[tuple[str, str]] = set()
    if spec.stdio is not None:
        for child_name, host_name in spec.stdio.env.items():
            selected.add((f"stdio.env.{child_name}", host_name))
    if spec.http is not None:
        for header_name, header in spec.http.headers.items():
            selected.add((f"http.headers.{header_name}", header.env))
    return tuple(sorted(selected))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalize_probe_tools(
    tools: Sequence[McpProviderTool | Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if isinstance(tools, (str, bytes, bytearray)) or not isinstance(tools, Sequence):
        raise ValidationError("MCP probe tools must be an array")
    if len(tools) > limit:
        raise ValidationError(f"MCP probe catalog exceeds configured tool limit={limit}")
    selected: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in tools:
        raw = to_jsonable(item)
        if not isinstance(raw, dict):
            raise ValidationError("MCP probe tool entries must be objects")
        name = raw.get("name")
        if type(name) is not str or not name:
            raise ValidationError("MCP probe tool name must be non-empty")
        if name in names:
            raise ValidationError("MCP probe tool names must be unique")
        names.add(name)
        description = raw.get("description")
        if description is not None and type(description) is not str:
            raise ValidationError("MCP probe tool description must be text or null")
        input_schema = raw.get("input_schema", raw.get("inputSchema", {}))
        metadata = raw.get("metadata", {})
        if not isinstance(input_schema, dict):
            raise ValidationError("MCP probe tool input schema must be an object")
        if not isinstance(metadata, dict):
            raise ValidationError("MCP probe tool metadata must be an object")
        optional: dict[str, Any] = {}
        for key in ("title", "output_schema", "annotations", "execution"):
            value = raw.get(key)
            if value is None and key == "output_schema":
                value = raw.get("outputSchema")
            if value is None:
                continue
            if key == "title":
                if type(value) is not str:
                    raise ValidationError("MCP probe tool title must be text or null")
            elif not isinstance(value, dict):
                raise ValidationError(f"MCP probe tool {key} must be an object")
            optional[key] = value
        bounded = bounded_json_loads(
            dumps(
                {
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                    "metadata": metadata,
                    **optional,
                }
            ),
            max_bytes=YAML_MAX_UTF8_BYTES,
        )
        selected.append(bounded)
    selected.sort(key=lambda item: item["name"])
    return selected


def _required_catalog_array(value: Mapping[str, Any], name: str) -> list[Any]:
    selected = value.get(name)
    if type(selected) is not list:
        raise ValidationError(f"MCP candidate probe {name} must be an array")
    return selected


def _normalize_probe_resources(
    resources: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _normalize_probe_read_catalog(
        resources,
        limit=limit,
        selector="resource_id",
        label="Resource",
        include_size=True,
    )


def _normalize_probe_resource_templates(
    templates: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _normalize_probe_read_catalog(
        templates,
        limit=limit,
        selector="template_id",
        label="Resource Template",
        include_size=False,
    )


def _normalize_probe_read_catalog(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    selector: str,
    label: str,
    include_size: bool,
) -> list[dict[str, Any]]:
    selected_items = _catalog_sequence(items, limit=limit, label=label)
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for item in selected_items:
        raw = _catalog_mapping(item, label=label)
        identity = raw.get(selector)
        name = raw.get("name")
        if type(identity) is not str or not identity:
            raise ValidationError(f"MCP probe {label} selector must be non-empty")
        if identity in identities:
            raise ValidationError(f"MCP probe {label} selectors must be unique")
        identities.add(identity)
        if type(name) is not str or not name:
            raise ValidationError(f"MCP probe {label} name must be non-empty")
        result: dict[str, Any] = {selector: identity, "name": name}
        for key in ("title", "description", "mime_type"):
            value = raw.get(key)
            if value is not None and type(value) is not str:
                raise ValidationError(f"MCP probe {label} {key} must be text or null")
            result[key] = value
        if include_size:
            size = raw.get("size")
            if size is not None and (type(size) is not int or size < 0):
                raise ValidationError("MCP probe Resource size is invalid")
            result["size"] = size
        for key in ("annotations", "metadata"):
            value = raw.get(key, {} if key == "metadata" else None)
            if value is None and key == "annotations":
                result[key] = None
            elif isinstance(value, dict):
                result[key] = value
            else:
                raise ValidationError(f"MCP probe {label} {key} must be an object")
        selected.append(_bounded_catalog_item(result, label=label))
    selected.sort(key=lambda item: item[selector])
    return selected


def _normalize_probe_prompts(
    prompts: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected_items = _catalog_sequence(prompts, limit=limit, label="Prompt")
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for item in selected_items:
        raw = _catalog_mapping(item, label="Prompt")
        prompt_id = raw.get("prompt_id")
        name = raw.get("name")
        if type(prompt_id) is not str or not prompt_id or prompt_id in identities:
            raise ValidationError("MCP probe Prompt ids must be non-empty and unique")
        identities.add(prompt_id)
        if type(name) is not str or not name:
            raise ValidationError("MCP probe Prompt name must be non-empty")
        arguments = raw.get("arguments", [])
        if type(arguments) is not list:
            raise ValidationError("MCP probe Prompt arguments must be an array")
        normalized_arguments: list[dict[str, Any]] = []
        argument_names: set[str] = set()
        for argument in arguments:
            selected_argument = _catalog_mapping(argument, label="Prompt argument")
            argument_name = selected_argument.get("name")
            if (
                type(argument_name) is not str
                or not argument_name
                or argument_name in argument_names
            ):
                raise ValidationError("MCP probe Prompt argument names are invalid")
            argument_names.add(argument_name)
            required = selected_argument.get("required", False)
            if type(required) is not bool:
                raise ValidationError("MCP probe Prompt argument required is invalid")
            normalized_arguments.append({"name": argument_name, "required": required})
        result: dict[str, Any] = {
            "prompt_id": prompt_id,
            "name": name,
            "arguments": normalized_arguments,
        }
        for key in ("title", "description"):
            value = raw.get(key)
            if value is not None and type(value) is not str:
                raise ValidationError(f"MCP probe Prompt {key} must be text or null")
            result[key] = value
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValidationError("MCP probe Prompt metadata must be an object")
        result["metadata"] = metadata
        selected.append(_bounded_catalog_item(result, label="Prompt"))
    selected.sort(key=lambda item: item["prompt_id"])
    return selected


def _catalog_sequence(items: Any, *, limit: int, label: str) -> Sequence[Any]:
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
        raise ValidationError(f"MCP probe {label} catalog must be an array")
    if len(items) > limit:
        raise ValidationError(f"MCP probe {label} catalog exceeds limit={limit}")
    return items


def _catalog_mapping(value: Any, *, label: str) -> dict[str, Any]:
    raw = to_jsonable(value)
    if type(raw) is not dict:
        raise ValidationError(f"MCP probe {label} entry must be an object")
    return raw


def _bounded_catalog_item(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        selected = bounded_json_loads(dumps(dict(value)), max_bytes=YAML_MAX_UTF8_BYTES)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(f"MCP probe {label} entry must contain finite JSON") from exc
    if type(selected) is not dict:
        raise ValidationError(f"MCP probe {label} entry must be an object")
    return selected


def _logical_tool_id(name: str, selected: set[str], *, max_chars: int) -> str:
    base = _SAFE_ID_CHARACTER.sub("_", name).strip("_")
    if not base or not base[0].isalnum():
        base = f"tool_{base}" if base else "tool"
    base = base[:max_chars]
    if base not in selected:
        return base
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    prefix = base[: max(1, max_chars - len(digest) - 1)]
    candidate = f"{prefix}_{digest}"
    counter = 2
    while candidate in selected:
        suffix = f"_{counter}"
        candidate = f"{prefix[: max(1, max_chars - len(digest) - len(suffix) - 1)]}_{digest}{suffix}"
        counter += 1
    return candidate


def _bounded_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("MCP probe diagnostics must be an object")
    selected: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValidationError("MCP probe diagnostic names must be non-empty strings")
        if item is None or type(item) in {bool, int, float}:
            selected[key] = item
        elif type(item) is str:
            selected[key] = item[:_MAX_PUBLIC_DIAGNOSTIC_CHARS]
        else:
            raise ValidationError("MCP probe diagnostics must contain scalar values")
    return bounded_json_loads(
        dumps(selected),
        max_bytes=YAML_MAX_UTF8_BYTES,
    )


def _exportable_spec(spec: McpDxManifest) -> dict[str, Any]:
    if isinstance(spec, McpServerManifestV3):
        selected = bounded_json_loads(
            canonical_mcp_v3_manifest_json(spec),
            max_bytes=MCP_DX_IMPORT_MAX_BYTES,
        )
        if not isinstance(selected, dict):  # pragma: no cover - canonical invariant
            raise TypeError("MCP Manifest v3 must encode as an object")
        return selected
    selected = dict(mcp_server_spec_to_jsonable(spec))
    if selected.get("stdio") is None:
        selected.pop("stdio", None)
    if selected.get("http") is None:
        selected.pop("http", None)
    return selected


def _spec_sha256(spec: McpDxManifest) -> str:
    canonical = (
        canonical_mcp_v3_manifest_json(spec)
        if isinstance(spec, McpServerManifestV3)
        else canonical_mcp_server_spec_json(spec)
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _copy_json_mapping(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{context} must be an object")
    try:
        selected = bounded_json_loads(
            dumps(dict(value)),
            max_bytes=MCP_DX_IMPORT_MAX_BYTES,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError(f"{context} must contain finite JSON") from exc
    if not isinstance(selected, dict):
        raise ValidationError(f"{context} must be an object")
    return selected


def _parse_bundle(bundle: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(bundle, Mapping):
        selected = _copy_json_mapping(bundle, context="MCP import bundle")
    elif isinstance(bundle, (str, bytes)):
        raw_bytes = bundle.encode("utf-8") if isinstance(bundle, str) else bundle
        if len(raw_bytes) > MCP_DX_IMPORT_MAX_BYTES:
            raise ValidationError(
                f"MCP import bundle exceeds max bytes={MCP_DX_IMPORT_MAX_BYTES}"
            )
        try:
            selected = bounded_json_loads(
                raw_bytes,
                max_bytes=MCP_DX_IMPORT_MAX_BYTES,
            )
        except ValueError as exc:
            raise ValidationError("MCP import bundle must be strict JSON") from exc
        if not isinstance(selected, dict):
            raise ValidationError("MCP import bundle must be an object")
    else:
        raise ValidationError("MCP import bundle must be JSON text or an object")
    _require_exact_keys(
        selected,
        {
            "kind",
            "schema_version",
            "secrets_included",
            "servers",
            "bundle_sha256",
        },
        context="MCP import bundle",
    )
    if selected["kind"] != MCP_DX_BUNDLE_KIND:
        raise ValidationError("MCP import bundle kind is invalid")
    if selected["schema_version"] != MCP_DX_FORMAT_VERSION:
        raise ValidationError("MCP import bundle schema version is unsupported")
    if selected["secrets_included"] is not False:
        raise ValidationError("MCP import bundle must never contain resolved secrets")
    servers = selected["servers"]
    if not isinstance(servers, list) or any(not isinstance(item, dict) for item in servers):
        raise ValidationError("MCP import bundle servers must be an array of objects")
    ids = [item.get("server_id") for item in servers]
    if any(type(item) is not str or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValidationError("MCP import bundle server ids must be unique non-empty strings")
    unsigned = {
        "kind": selected["kind"],
        "schema_version": selected["schema_version"],
        "secrets_included": selected["secrets_included"],
        "servers": servers,
    }
    expected = hashlib.sha256(dumps(unsigned).encode("utf-8")).hexdigest()
    if selected["bundle_sha256"] != expected:
        raise ValidationError("MCP import bundle digest does not match its contents")
    return selected


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise ValidationError(f"{context} fields are invalid")
