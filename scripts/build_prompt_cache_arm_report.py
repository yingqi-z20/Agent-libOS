from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_libos.llm.prompt_cache_gate import PromptCachePricing
from benchmarks.prompt_cache_release import (
    ProviderPromptCacheArmInput,
    build_prompt_cache_arm_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one redacted multi-provider prompt-cache release arm."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _read_object(args.manifest)
    providers = manifest.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError("manifest.providers must be a non-empty list")
    inputs = [_provider_input(row, args.manifest.parent) for row in providers]
    report = build_prompt_cache_arm_report(
        inputs,
        security_invariants_passed=(
            manifest.get("security_invariants_passed") is True
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _provider_input(value: Any, base: Path) -> ProviderPromptCacheArmInput:
    if not isinstance(value, dict):
        raise ValueError("each provider manifest entry must be an object")
    report_path = value.get("report")
    if not isinstance(report_path, str) or not report_path:
        raise ValueError("provider report must be a non-empty path")
    selected_path = Path(report_path)
    if not selected_path.is_absolute():
        selected_path = base / selected_path
    pricing_value = value.get("pricing")
    pricing = (
        PromptCachePricing(**pricing_value)
        if isinstance(pricing_value, dict)
        else None
    )
    return ProviderPromptCacheArmInput(
        provider_id=str(value.get("provider_id") or ""),
        model_id=str(value.get("model_id") or ""),
        repetitions=value.get("repetitions"),
        report=_read_object(selected_path),
        pricing=pricing,
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON value must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
