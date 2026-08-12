from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_libos.llm.prompt_cache_gate import PromptCachePricing
from agent_libos.utils.serde import bounded_json_loads
from benchmarks.prompt_cache_release import (
    ProviderPromptCacheArmInput,
    build_prompt_cache_arm_report,
)


_MAX_MANIFEST_BYTES = 1_048_576
_MAX_PROVIDER_REPORT_BYTES = 16_777_216


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one redacted multi-provider prompt-cache release arm."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _read_object(
        args.manifest,
        max_bytes=_MAX_MANIFEST_BYTES,
        label="manifest",
    )
    security_invariants_passed = manifest.get("security_invariants_passed")
    if type(security_invariants_passed) is not bool:
        raise ValueError(
            "manifest.security_invariants_passed must be an explicit boolean"
        )
    providers = manifest.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError("manifest.providers must be a non-empty list")
    inputs = [_provider_input(row, args.manifest.parent) for row in providers]
    report = build_prompt_cache_arm_report(
        inputs,
        security_invariants_passed=security_invariants_passed,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _provider_input(value: Any, base: Path) -> ProviderPromptCacheArmInput:
    if not isinstance(value, dict):
        raise ValueError("each provider manifest entry must be an object")
    provider_id = value.get("provider_id")
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("provider_id must be a non-empty string")
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    report_path = value.get("report")
    if not isinstance(report_path, str) or not report_path:
        raise ValueError("provider report must be a non-empty path")
    selected_path = Path(report_path)
    if not selected_path.is_absolute():
        selected_path = base / selected_path
    if "pricing" in value:
        pricing_value = value["pricing"]
        if not isinstance(pricing_value, dict):
            raise ValueError("provider pricing must be an object when present")
        try:
            pricing = PromptCachePricing(**pricing_value)
        except TypeError as exc:
            raise ValueError("provider pricing has invalid fields") from exc
    else:
        pricing = None
    return ProviderPromptCacheArmInput(
        provider_id=provider_id,
        model_id=model_id,
        repetitions=value.get("repetitions"),
        report=_read_object(
            selected_path,
            max_bytes=_MAX_PROVIDER_REPORT_BYTES,
            label="provider report",
        ),
        pricing=pricing,
    )


def _read_object(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    with path.open("rb") as handle:
        encoded = handle.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds max_bytes={max_bytes}: {path}")
    value = bounded_json_loads(encoded, max_bytes=max_bytes)
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON value must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
