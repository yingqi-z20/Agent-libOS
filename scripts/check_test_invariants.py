from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.runtime_safety.loader import load_tasks
from agent_libos.utils.yaml_loader import load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "invariants.yaml"
VALID_LANES = {"unit", "runtime", "security", "self-evolution", "providers", "benchmark"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the runtime invariant test manifest.")
    parser.add_argument("--manifest", default=str(MANIFEST), help="path to tests/invariants.yaml")
    args = parser.parse_args(argv)

    manifest = _load_manifest(Path(args.manifest))
    collected = _collect_pytest_nodeids()
    deterministic_collected = _collect_pytest_nodeids("not real_llm")
    errors: list[str] = []
    invariant_ids, declared_attack_classes = _check_invariants(
        manifest,
        collected,
        deterministic_collected,
        errors,
    )
    _check_benchmark_attack_classes(manifest, invariant_ids, declared_attack_classes, errors)

    if errors:
        for error in errors:
            print(f"invariant check failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {len(invariant_ids)} invariants against {len(collected)} collected pytest nodes")
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


def _collect_pytest_nodeids(marker_expression: str | None = None) -> set[str]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if marker_expression:
        command.extend(["-m", marker_expression])
    command.append("tests")
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


def _check_invariants(
    manifest: dict[str, Any],
    collected: set[str],
    deterministic_collected: set[str],
    errors: list[str],
) -> tuple[set[str], dict[str, str]]:
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


def _string_field(mapping: dict[str, Any], key: str, errors: list[str], prefix: str) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: {key} must be a non-empty string")
        return None
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
