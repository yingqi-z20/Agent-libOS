from __future__ import annotations

from typing import Any


def has_committed_filesystem_write(
    runtime: Any,
    pid: str,
    results: list[Any],
    expected_path: str,
) -> bool:
    """Bind a script's success to its exact Tool receipt and durable effect."""

    receipt = any(
        isinstance(item, dict)
        and item.get("ok") is True
        and isinstance(item.get("action"), dict)
        and item["action"].get("action") == "write_text_file"
        and item["action"].get("path") == expected_path
        and isinstance(item.get("result"), dict)
        and item["result"].get("path") == expected_path
        and type(item["result"].get("bytes_written")) is int
        and item["result"]["bytes_written"] >= 0
        for item in results
    )
    if not receipt:
        return False
    resource = runtime.filesystem.resource_for(expected_path)
    return any(
        effect.operation == "filesystem.write_text"
        and effect.target == resource
        and effect.effect_state == "finalized"
        and effect.transaction_state == "committed"
        and effect.state_mutation
        for effect in runtime.store.list_external_effects(pid=pid)
    )


__all__ = ["has_committed_filesystem_write"]
