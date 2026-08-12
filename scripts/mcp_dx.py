#!/usr/bin/env python3
"""Host-only MCP validation, diagnostics, scaffold, probe, export and import.

This standalone entry point keeps the new DX workflow usable while the stable
``agent-libos mcp`` command remains backward compatible.  It never turns
transport configuration into a process/model input.  Live probe and registry
import require explicit command-line confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_libos import Runtime
from agent_libos.config import load_config_file
from agent_libos.mcp.dx import (
    CandidateMcpProbeAdapter,
    MCP_DX_IMPORT_MAX_BYTES,
    McpDxConfirmation,
    McpDxManagerAdapter,
    approve_scaffold_candidate,
    doctor_manifest_text,
    export_registry_bundle,
    import_one_from_bundle,
    plan_import_bundle,
    probe_manifest,
    scaffold_manifest_candidate,
    validate_manifest_text,
)
from agent_libos.models.exceptions import LibOSError, ValidationError
from agent_libos.utils.serde import bounded_json_loads
from agent_libos.utils.yaml_loader import YAML_MAX_UTF8_BYTES, load_yaml_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Host-only MCP manifest and registry developer-experience tools."
    )
    parser.add_argument("--config", help="Optional Agent libOS YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate",
        help="Validate one manifest offline without registry/provider effects.",
    )
    validate.add_argument("manifest")

    doctor = sub.add_parser(
        "doctor",
        help="Check a manifest, optional dependencies and environment references offline.",
    )
    doctor.add_argument("manifest")

    scaffold = sub.add_parser(
        "scaffold",
        help="Build a conservative non-registerable candidate from a complete Tool catalog.",
    )
    scaffold.add_argument("base_manifest")
    scaffold.add_argument("catalog_json")
    _add_confirmation(scaffold, flag="--confirm-scaffold")

    approve = sub.add_parser(
        "approve",
        help="Extract and validate a manually reviewed scaffold candidate manifest.",
    )
    approve.add_argument("candidate_json")
    _add_confirmation(approve, flag="--confirm-review")

    probe = sub.add_parser(
        "probe",
        help="Probe an unregistered v3 manifest through the governed Host boundary.",
    )
    probe.add_argument("--db", required=True)
    probe.add_argument("manifest")
    _add_confirmation(probe, flag="--confirm-probe")

    export = sub.add_parser(
        "export",
        help="Export canonical manifests; environment references are kept, values never are.",
    )
    export.add_argument("--db", required=True)
    export.add_argument("--server", action="append", dest="servers")

    plan = sub.add_parser(
        "import-plan",
        help="Validate an export and show create/replace/unchanged actions without mutation.",
    )
    plan.add_argument("--db", required=True)
    plan.add_argument("bundle_json")

    apply = sub.add_parser(
        "import-one",
        help="CAS-import exactly one selected server from a validated bundle.",
    )
    apply.add_argument("--db", required=True)
    apply.add_argument("bundle_json")
    apply.add_argument("server_id")
    apply.add_argument("--actor", default="mcp-dx")
    _add_confirmation(apply, flag="--confirm-import")
    return parser


def _add_confirmation(parser: argparse.ArgumentParser, *, flag: str) -> None:
    parser.add_argument(
        flag,
        action="store_true",
        dest="confirmed",
        help="Explicitly confirm this Host operation after reviewing its scope.",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    config = load_config_file(args.config) if args.config else None
    runtime: Runtime | None = None
    try:
        if args.command in {"validate", "doctor", "scaffold", "approve"}:
            # Offline commands must not depend on, migrate, or leave evidence in
            # the developer's default registry. An ephemeral Runtime supplies
            # the active Host policy without contacting an MCP provider.
            runtime = Runtime.open(":memory:", **({"config": config} if config else {}))
        else:
            runtime = Runtime.open(args.db, **({"config": config} if config else {}))
        adapter = McpDxManagerAdapter(runtime.mcp)
        result = _dispatch(adapter, args)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except LibOSError as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    finally:
        if runtime is not None:
            runtime.close()


def _dispatch(adapter: McpDxManagerAdapter, args: argparse.Namespace) -> Any:
    if args.command == "validate":
        return validate_manifest_text(
            adapter,
            _read_text(args.manifest, max_bytes=adapter.manifest_max_bytes),
        ).to_jsonable()
    if args.command == "doctor":
        return doctor_manifest_text(
            adapter,
            _read_text(args.manifest, max_bytes=adapter.manifest_max_bytes),
        ).to_jsonable()
    if args.command == "scaffold":
        base = load_yaml_mapping(
            _read_text(args.base_manifest, max_bytes=adapter.manifest_max_bytes)
        )
        catalog = _read_json(args.catalog_json, max_bytes=MCP_DX_IMPORT_MAX_BYTES)
        fields = _catalog_fields(catalog)
        return scaffold_manifest_candidate(
            adapter,
            base,
            fields["tools"],
            live_resources=fields["resources"],
            live_resource_templates=fields["resource_templates"],
            live_prompts=fields["prompts"],
            probe_manifest_sha256=fields["manifest_sha256"],
            confirmation=_confirmation(args),
            catalog_scope=fields["catalog_scope"],
            complete=fields["complete"],
        )
    if args.command == "approve":
        candidate = _read_json(args.candidate_json, max_bytes=MCP_DX_IMPORT_MAX_BYTES)
        if not isinstance(candidate, dict):
            raise ValidationError("MCP scaffold candidate must be an object")
        return approve_scaffold_candidate(
            adapter,
            candidate,
            confirmation=_confirmation(args),
        )
    if args.command == "probe":
        return probe_manifest(
            adapter,
            _read_text(args.manifest, max_bytes=adapter.manifest_max_bytes),
            probe_adapter=CandidateMcpProbeAdapter(adapter),
            confirmation=_confirmation(args),
        ).to_jsonable()
    if args.command == "export":
        return export_registry_bundle(adapter, server_ids=args.servers)
    if args.command == "import-plan":
        return plan_import_bundle(
            adapter,
            _read_bytes(args.bundle_json, max_bytes=MCP_DX_IMPORT_MAX_BYTES),
        ).to_jsonable()
    if args.command == "import-one":
        return import_one_from_bundle(
            adapter,
            _read_bytes(args.bundle_json, max_bytes=MCP_DX_IMPORT_MAX_BYTES),
            server_id=args.server_id,
            confirmation=_confirmation(args),
            actor=args.actor,
            require_capability=False,
        )
    raise AssertionError(f"unsupported MCP DX command: {args.command}")


def _confirmation(args: argparse.Namespace) -> McpDxConfirmation:
    return McpDxConfirmation(
        confirmed=args.confirmed is True,
        actor=args.reviewer,
        reason=args.reason,
    )


def _catalog_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        # A bare list has no machine-verifiable completeness receipt.  Requiring
        # the explicit report shape prevents a truncated first page from being
        # mistaken for a complete authority candidate.
        raise ValidationError(
            "MCP scaffold catalog must be a probe report with scope and completeness"
        )
    if not isinstance(value, dict):
        raise ValidationError("MCP scaffold catalog must be an object")
    selected: dict[str, Any] = {}
    for name in ("tools", "resources", "resource_templates", "prompts"):
        items = value.get(name)
        if not isinstance(items, list):
            raise ValidationError(f"MCP scaffold catalog {name} must be an array")
        selected[name] = items
    digest = value.get("manifest_sha256")
    scope = value.get("catalog_scope")
    if type(digest) is not str or type(scope) is not str:
        raise ValidationError("MCP scaffold catalog binding is invalid")
    return {
        **selected,
        "manifest_sha256": digest,
        "catalog_scope": scope,
        "complete": value.get("complete") is True,
    }


def _read_json(path: str, *, max_bytes: int) -> Any:
    try:
        return bounded_json_loads(_read_bytes(path, max_bytes=max_bytes), max_bytes=max_bytes)
    except ValueError as exc:
        raise ValidationError("MCP DX input must be strict JSON") from exc


def _read_text(path: str, *, max_bytes: int = YAML_MAX_UTF8_BYTES) -> str:
    raw = _read_bytes(path, max_bytes=max_bytes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("MCP DX input must be valid UTF-8 text") from exc


def _read_bytes(path: str, *, max_bytes: int) -> bytes:
    selected = Path(path).expanduser().resolve()
    try:
        with selected.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ValidationError("MCP DX input file cannot be read") from exc
    if len(raw) > max_bytes:
        raise ValidationError(f"MCP DX input exceeds max bytes={max_bytes}")
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
