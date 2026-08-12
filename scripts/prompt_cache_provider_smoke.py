from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_libos.llm.client import LLMClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a credential-safe prompt-cache provider compatibility smoke."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api", choices=("chat", "responses"), required=True)
    parser.add_argument(
        "--mode",
        choices=("implicit", "explicit"),
        default="explicit",
    )
    parser.add_argument("--ttl", choices=("30m",), default="30m")
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to observe reuse")

    base: LLMClient | None = None
    client: LLMClient | None = None
    try:
        base = LLMClient.from_env(
            args.env_file,
            allow_custom_base_url=True,
        )
        client = LLMClient(
            base_url=base.base_url,
            model=base.model,
            api_key=base.api_key,
            timeout=base.timeout,
            max_retries=base.max_retries,
            api_mode=args.api,
            store=base.store,
            prompt_cache_key="provider-smoke-private-domain",
            prompt_cache_mode=args.mode,
            prompt_cache_ttl=args.ttl,
            parallel_tool_calls=False,
            fallback_json_actions=False,
            inherit_ambient_openai_sdk_config=False,
            allow_custom_base_url=True,
            defaults=base.defaults,
        )
        outcomes = [_one_call(client) for _index in range(args.repetitions)]
        print(
            json.dumps(
                {
                    "ok": True,
                    "api": args.api,
                    "mode": args.mode,
                    "repetitions": args.repetitions,
                    "outcomes": outcomes,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "api": args.api,
                    "mode": args.mode,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        if client is not None:
            client.close()
        if base is not None:
            base.close()


def _one_call(client: LLMClient) -> dict[str, Any]:
    stable = ("Stable provider compatibility context. " * 240).strip()
    completion = client.complete_with_metadata(
        [
            {"role": "system", "content": "Return only OK."},
            {
                "role": "user",
                "content": stable,
                "_agent_libos_cache_stable": True,
            },
            {"role": "user", "content": "Return only OK."},
        ],
        max_tokens=8,
        json_mode=False,
    )
    return {
        "api": completion.api,
        "usage": {
            key: value
            for key, value in completion.usage.items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        },
        "provider_request_options": completion.provider_request_options,
        "compatibility_removed_options": completion.compatibility_removed_options,
    }


if __name__ == "__main__":
    raise SystemExit(main())
