from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.runtime_safety.loader import load_tasks
from agent_libos.utils.yaml_loader import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "invariants.yaml"
DOCUMENTATION = ROOT / "docs" / "invariants.md"
VALID_LANES = {"unit", "runtime", "security", "self-evolution", "providers", "benchmark"}
MANIFEST_SCHEMA_VERSION = 2
# The ordinary lane selector excludes the whole MCP product marker to avoid
# running those nodes twice.  Invariant evidence is a different question:
# deterministic MCP regressions that do not require a frozen SDK transport are
# valid non-optional witnesses and must remain eligible here.
DEFAULT_DETERMINISTIC_MARKER_EXPRESSION = (
    "not postgres and not real_llm and not mcp_transport"
)
PLATFORM_MARKERS = {
    "darwin": "platform_darwin",
    "linux": "platform_linux",
    "windows": "platform_windows",
}
INVARIANT_EXECUTION_RECEIPT_ENV = "AGENT_LIBOS_INVARIANT_EXECUTION_RECEIPT"
EXECUTION_RECEIPT_SCHEMA_VERSION = 1
_DOCUMENTED_INVARIANT_PATTERN = re.compile(r"^- `([^`]+)`:", re.MULTILINE)
_EXECUTED_NODEIDS: set[str] = set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the runtime invariant test manifest.")
    parser.add_argument("--manifest", default=str(MANIFEST), help="path to tests/invariants.yaml")
    parser.add_argument(
        "--documentation",
        default=str(DOCUMENTATION),
        help="path to docs/invariants.md",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(Path(args.manifest))
    collected = _collect_pytest_nodeids()
    deterministic_collected = _collect_pytest_nodeids(
        DEFAULT_DETERMINISTIC_MARKER_EXPRESSION
    )
    platform_collected = {
        platform: _collect_pytest_nodeids(
            f"{DEFAULT_DETERMINISTIC_MARKER_EXPRESSION} and {marker}"
        )
        for platform, marker in PLATFORM_MARKERS.items()
    }
    errors: list[str] = []
    invariant_ids, declared_attack_classes = _check_invariants(
        manifest,
        collected,
        deterministic_collected,
        errors,
        platform_collected=platform_collected,
    )
    _check_benchmark_attack_classes(manifest, invariant_ids, declared_attack_classes, errors)
    _check_documented_invariants(manifest, Path(args.documentation), errors)

    if errors:
        for error in errors:
            print(f"invariant check failed: {error}", file=sys.stderr)
        return 1
    print(
        f"validated {len(invariant_ids)} invariant declarations against "
        f"{len(collected)} collected pytest nodes; non-skipped execution "
        "evidence is enforced by scripts/test_matrix.py"
    )
    return 0


def _load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = load_yaml_mapping(text)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    return data


def _collect_pytest_nodeids(
    marker_expression: str | None = None,
    *,
    test_paths: tuple[str, ...] = ("tests",),
) -> set[str]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if marker_expression:
        command.extend(["-m", marker_expression])
    command.extend(test_paths)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith("<")
    }


def _check_invariant_execution(
    manifest: dict[str, Any],
    executed_nodeids: set[str],
    errors: list[str],
    *,
    lane: str | None = None,
    platform: str | None = None,
    selected_test_paths: tuple[str, ...] | None = None,
    selected_nodeids: set[str] | None = None,
) -> None:
    """Require an actual passing call receipt for every selected invariant."""

    invariants = manifest.get("invariants")
    if not isinstance(invariants, list):
        errors.append("manifest requires a non-empty invariants list")
        return
    normalized_executed = {
        node_id.replace("\\", "/") for node_id in executed_nodeids
    }
    normalized_selected_paths = (
        None
        if selected_test_paths is None
        else {path.replace("\\", "/") for path in selected_test_paths}
    )
    normalized_selected_nodeids = (
        None
        if selected_nodeids is None
        else {node_id.replace("\\", "/") for node_id in selected_nodeids}
    )

    def node_path_is_selected(node_id: str) -> bool:
        if normalized_selected_paths is None:
            return True
        node_path = node_id.split("::", 1)[0]
        return any(
            node_path == selected_path
            or node_path.startswith(selected_path.rstrip("/") + "/")
            for selected_path in normalized_selected_paths
        )

    selected_platform = platform or _current_platform_key()
    for invariant in invariants:
        if not isinstance(invariant, dict):
            continue
        invariant_id = invariant.get("id")
        invariant_lane = invariant.get("lane")
        if not isinstance(invariant_id, str) or not invariant_id:
            continue
        if lane is not None and invariant_lane != lane:
            continue
        node_ids = invariant.get("node_ids")
        if not isinstance(node_ids, list):
            continue
        normalized_nodes = {
            node_id.replace("\\", "/")
            for node_id in node_ids
            if isinstance(node_id, str) and node_id
        }
        required = invariant.get("required_platform_nodes", {})
        applicable_nodes = normalized_nodes
        if isinstance(required, dict) and required:
            platform_scoped_nodes = {
                node_id.replace("\\", "/")
                for platform_nodes in required.values()
                if isinstance(platform_nodes, list)
                for node_id in platform_nodes
                if isinstance(node_id, str) and node_id
            }
            generic_nodes = normalized_nodes - platform_scoped_nodes
            selected_platform_nodes = required.get(selected_platform, [])
            applicable_nodes = generic_nodes | {
                node_id.replace("\\", "/")
                for node_id in selected_platform_nodes
                if isinstance(node_id, str) and node_id
            }
            # A fully platform-scoped invariant does not apply on a Host for
            # which it declares no regression nodes. This keeps, for example,
            # Darwin/Linux filesystem identity evidence from making a native
            # Windows lane fail solely because every declared node correctly
            # skipped there.
            if not applicable_nodes:
                continue
        if normalized_selected_paths is not None:
            applicable_nodes = {
                node_id
                for node_id in applicable_nodes
                if node_path_is_selected(node_id)
            }
            if not applicable_nodes:
                continue
        if normalized_selected_nodeids is not None:
            applicable_nodes &= normalized_selected_nodeids
            if not applicable_nodes:
                continue
        if applicable_nodes.isdisjoint(normalized_executed):
            selected_lane = lane or "all deterministic"
            errors.append(
                f"{invariant_id}: no declared regression node completed "
                f"without skip in the {selected_lane} lane"
            )
        if not isinstance(required, dict) or selected_platform is None:
            continue
        platform_nodes = required.get(selected_platform, [])
        if not isinstance(platform_nodes, list):
            continue
        for node_id in platform_nodes:
            if not isinstance(node_id, str) or not node_id:
                continue
            normalized = node_id.replace("\\", "/")
            if (
                normalized_selected_paths is not None
                and not node_path_is_selected(normalized)
            ):
                continue
            if (
                normalized_selected_nodeids is not None
                and normalized not in normalized_selected_nodeids
            ):
                continue
            if normalized not in normalized_executed:
                errors.append(
                    f"{invariant_id}: required {selected_platform} pytest node "
                    f"did not complete without skip: {node_id}"
                )


def _current_platform_key() -> str | None:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return None


def load_execution_receipt(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read invariant execution receipt: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "passed_node_ids",
    }:
        raise ValueError("invalid invariant execution receipt shape")
    if payload.get("schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported invariant execution receipt schema_version")
    node_ids = payload.get("passed_node_ids")
    if not isinstance(node_ids, list) or any(
        not isinstance(node_id, str) or not node_id for node_id in node_ids
    ):
        raise ValueError("invariant execution receipt node ids must be strings")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("invariant execution receipt contains duplicate node ids")
    return {node_id.replace("\\", "/") for node_id in node_ids}


def pytest_sessionstart(session: Any) -> None:
    if os.getenv(INVARIANT_EXECUTION_RECEIPT_ENV):
        _EXECUTED_NODEIDS.clear()


def pytest_runtest_logreport(report: Any) -> None:
    if not os.getenv(INVARIANT_EXECUTION_RECEIPT_ENV):
        return
    if (
        getattr(report, "when", None) == "call"
        and bool(getattr(report, "passed", False))
        and not hasattr(report, "wasxfail")
    ):
        _EXECUTED_NODEIDS.add(str(report.nodeid).replace("\\", "/"))


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del exitstatus
    selected = os.getenv(INVARIANT_EXECUTION_RECEIPT_ENV)
    if not selected or hasattr(session.config, "workerinput"):
        return
    path = Path(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
                "passed_node_ids": sorted(_EXECUTED_NODEIDS),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _check_invariants(
    manifest: dict[str, Any],
    collected: set[str],
    deterministic_collected: set[str],
    errors: list[str],
    *,
    platform_collected: dict[str, set[str]] | None = None,
) -> tuple[set[str], dict[str, str]]:
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            "manifest schema_version must be exact integer "
            f"{MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
        )
    invariants = manifest.get("invariants")
    if not isinstance(invariants, list) or not invariants:
        errors.append("manifest requires a non-empty invariants list")
        return set(), {}

    ids: set[str] = set()
    attack_class_owners: dict[str, str] = {}
    for index, invariant in enumerate(invariants):
        if not isinstance(invariant, dict):
            errors.append(f"invariants[{index}] must be an object")
            continue
        invariant_id = _string_field(invariant, "id", errors, f"invariants[{index}]")
        if not invariant_id:
            continue
        if invariant_id in ids:
            errors.append(f"duplicate invariant id: {invariant_id}")
        ids.add(invariant_id)
        _string_field(invariant, "title", errors, invariant_id)
        lane = _string_field(invariant, "lane", errors, invariant_id)
        if lane and lane not in VALID_LANES:
            errors.append(f"{invariant_id}: lane must be one of {sorted(VALID_LANES)}, got {lane!r}")
        node_ids = invariant.get("node_ids")
        if not isinstance(node_ids, list) or not node_ids:
            errors.append(f"{invariant_id}: node_ids must be a non-empty list")
            continue
        normalized_node_ids: list[str] = []
        for node_id in node_ids:
            if not isinstance(node_id, str) or not node_id:
                errors.append(f"{invariant_id}: node_ids entries must be non-empty strings")
                continue
            normalized = node_id.replace("\\", "/")
            normalized_node_ids.append(normalized)
            if normalized not in collected:
                errors.append(f"{invariant_id}: pytest node not collected: {node_id}")
        if normalized_node_ids and not any(node_id in deterministic_collected for node_id in normalized_node_ids):
            errors.append(f"{invariant_id}: requires at least one deterministic regression node")
        _check_required_platform_nodes(
            invariant,
            invariant_id,
            set(normalized_node_ids),
            platform_collected or {},
            errors,
        )
        attack_classes = invariant.get("benchmark_attack_classes", [])
        if not isinstance(attack_classes, list):
            errors.append(f"{invariant_id}: benchmark_attack_classes must be a list")
            continue
        for attack_class in attack_classes:
            if not isinstance(attack_class, str) or not attack_class.strip():
                errors.append(f"{invariant_id}: benchmark_attack_classes entries must be non-empty strings")
                continue
            attack_class = attack_class.strip()
            previous_owner = attack_class_owners.get(attack_class)
            if previous_owner and previous_owner != invariant_id:
                errors.append(
                    f"{attack_class!r} is declared by both {previous_owner!r} and {invariant_id!r}"
                )
            attack_class_owners[attack_class] = invariant_id
    return ids, attack_class_owners


def _check_required_platform_nodes(
    invariant: dict[str, Any],
    invariant_id: str,
    invariant_node_ids: set[str],
    platform_collected: dict[str, set[str]],
    errors: list[str],
) -> None:
    required = invariant.get("required_platform_nodes", {})
    if not isinstance(required, dict):
        errors.append(f"{invariant_id}: required_platform_nodes must be an object")
        return
    for platform, node_ids in required.items():
        if not isinstance(platform, str) or platform not in PLATFORM_MARKERS:
            errors.append(
                f"{invariant_id}: required platform must be one of "
                f"{sorted(PLATFORM_MARKERS)}, got {platform!r}"
            )
            continue
        if not isinstance(node_ids, list) or not node_ids:
            errors.append(
                f"{invariant_id}: required_platform_nodes[{platform!r}] "
                "must be a non-empty list"
            )
            continue
        normalized_platform_nodes: list[str] = []
        for node_id in node_ids:
            if not isinstance(node_id, str) or not node_id:
                errors.append(
                    f"{invariant_id}: required_platform_nodes[{platform!r}] "
                    "entries must be non-empty strings"
                )
                continue
            normalized = node_id.replace("\\", "/")
            normalized_platform_nodes.append(normalized)
            if normalized not in invariant_node_ids:
                errors.append(
                    f"{invariant_id}: required {platform} pytest node is not "
                    f"declared in node_ids: {node_id}"
                )
            if normalized not in platform_collected.get(platform, set()):
                errors.append(
                    f"{invariant_id}: required {platform} pytest node is not "
                    f"deterministically collected with {PLATFORM_MARKERS[platform]}: {node_id}"
                )
        duplicate_nodes = sorted(
            node_id
            for node_id, count in Counter(normalized_platform_nodes).items()
            if count > 1
        )
        if duplicate_nodes:
            errors.append(
                f"{invariant_id}: required_platform_nodes[{platform!r}] contains "
                "duplicate nodes: " + ", ".join(duplicate_nodes)
            )


def _check_benchmark_attack_classes(
    manifest: dict[str, Any],
    invariant_ids: set[str],
    declared_attack_classes: dict[str, str],
    errors: list[str],
) -> None:
    mapping = manifest.get("benchmark_attack_classes")
    if not isinstance(mapping, dict) or not mapping:
        errors.append("manifest requires benchmark_attack_classes mapping")
        return
    for attack_class, invariant_id in mapping.items():
        if not isinstance(attack_class, str) or not attack_class.strip():
            errors.append("benchmark_attack_classes keys must be non-empty strings")
            continue
        if not isinstance(invariant_id, str) or not invariant_id.strip():
            errors.append(f"benchmark attack class {attack_class!r} must map to a non-empty invariant id")
            continue
        if invariant_id not in invariant_ids:
            errors.append(f"benchmark attack class {attack_class!r} maps to unknown invariant {invariant_id!r}")
        declared_owner = declared_attack_classes.get(attack_class)
        if declared_owner is None:
            errors.append(f"benchmark attack class {attack_class!r} is missing from invariant declarations")
        elif declared_owner != invariant_id:
            errors.append(
                f"benchmark attack class {attack_class!r} maps to {invariant_id!r} "
                f"but is declared on {declared_owner!r}"
            )
    for attack_class, invariant_id in declared_attack_classes.items():
        if attack_class not in mapping:
            errors.append(
                f"benchmark attack class {attack_class!r} is declared on {invariant_id!r} "
                "but missing from top-level mapping"
            )
    for task in load_tasks(ROOT / "benchmarks" / "runtime_safety"):
        if task.attack_class not in mapping:
            source = task.source_path.relative_to(ROOT) if task.source_path else task.id
            errors.append(f"{source}: attack_class {task.attack_class!r} is not mapped to an invariant")


def _check_documented_invariants(
    manifest: dict[str, Any],
    documentation: Path,
    errors: list[str],
) -> None:
    invariants = manifest.get("invariants")
    if not isinstance(invariants, list):
        return
    manifest_ids = [
        invariant.get("id")
        for invariant in invariants
        if isinstance(invariant, dict) and isinstance(invariant.get("id"), str)
    ]
    try:
        text = documentation.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read invariant documentation {documentation}: {exc}")
        return
    documented_ids = _DOCUMENTED_INVARIANT_PATTERN.findall(text)
    duplicate_ids = sorted(
        invariant_id
        for invariant_id, count in Counter(documented_ids).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(
            "docs/invariants.md contains duplicate invariant ids: "
            + ", ".join(duplicate_ids)
        )
    documented_set = set(documented_ids)
    manifest_set = set(manifest_ids)
    missing = [invariant_id for invariant_id in manifest_ids if invariant_id not in documented_set]
    stale = [invariant_id for invariant_id in documented_ids if invariant_id not in manifest_set]
    if missing:
        errors.append(
            "docs/invariants.md is missing manifest invariant ids: "
            + ", ".join(missing)
        )
    if stale:
        errors.append(
            "docs/invariants.md contains unknown invariant ids: "
            + ", ".join(stale)
        )


def _string_field(mapping: dict[str, Any], key: str, errors: list[str], prefix: str) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: {key} must be a non-empty string")
        return None
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
