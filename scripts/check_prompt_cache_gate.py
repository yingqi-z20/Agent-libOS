from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_libos.llm.prompt_cache_gate import evaluate_prompt_cache_release_gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate paired legacy_v1/cache_optimized_v2 cache evidence."
    )
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Check token/cache/success gates without requiring release-evidence fields.",
    )
    args = parser.parse_args()
    legacy = _read_object(args.legacy)
    candidate = _read_object(args.candidate)
    result = evaluate_prompt_cache_release_gate(
        legacy,
        candidate,
        strict_release_evidence=not args.canary,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
