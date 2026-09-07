from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_libos.utils.ids import estimate_tokens


def plan_compaction_chunks(
    entries: Sequence[dict[str, Any]],
    *,
    target_tokens: int,
    max_chunks: int,
) -> list[int]:
    """Return exclusive entry boundaries for a lossless, ordered chunk plan.

    Use as few stages as the token target permits. If indivisible entries or
    the stage limit make that impossible, find the smallest feasible capacity
    instead of placing an arbitrarily large remainder in the last stage.
    The target covers source entries, not the compressor's complete request.
    """

    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("compaction target_tokens must be a positive integer")
    if type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("compaction max_chunks must be a positive integer")
    if not entries:
        return []
    # Include the JSON list separator. Entries stay intact, including tool
    # receipts, multilingual text, and the previous cumulative summary.
    weights = [estimate_tokens(entry) + 1 for entry in entries]
    lower = max(target_tokens, max(weights))
    upper = max(lower, sum(weights))
    while lower < upper:
        capacity = (lower + upper) // 2
        if len(_chunk_boundaries(weights, capacity)) <= max_chunks:
            upper = capacity
        else:
            lower = capacity + 1
    return _chunk_boundaries(weights, lower)


def _chunk_boundaries(weights: Sequence[int], capacity: int) -> list[int]:
    boundaries: list[int] = []
    used = 0
    for index, weight in enumerate(weights):
        if used and used + weight > capacity:
            boundaries.append(index)
            used = 0
        used += weight
    boundaries.append(len(weights))
    return boundaries
