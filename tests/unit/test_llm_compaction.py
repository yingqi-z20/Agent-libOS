from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.compaction import plan_compaction_chunks
from agent_libos.tools.base import ToolExecutionError
from agent_libos.tools.builtin.context import (
    CompactProcessContextArgs,
    _entry_chunks,
    _job_entry_chunks,
    _new_job_payload,
)
from agent_libos.utils.ids import estimate_tokens


def test_small_history_uses_one_compressor_stage() -> None:
    entries = [{"kind": "fact", "index": index} for index in range(40)]
    assert plan_compaction_chunks(entries, target_tokens=16_384, max_chunks=8) == [40]


def test_skewed_history_is_balanced_by_tokens_without_losing_order() -> None:
    entries = [
        {"kind": "tool_result", "output": "详细输出" * 500},
        {"kind": "tool_result", "output": "another output " * 300},
        *[{"kind": "fact", "index": index} for index in range(40)],
    ]
    boundaries = plan_compaction_chunks(entries, target_tokens=1_000, max_chunks=3)
    chunks = [entries[start:end] for start, end in zip([0, *boundaries], boundaries)]
    assert [entry for chunk in chunks for entry in chunk] == entries
    assert len(chunks) <= 3
    legacy = _entry_chunks({"entries": entries}, 3)
    assert max(map(estimate_tokens, chunks)) < max(map(estimate_tokens, legacy))


def test_stage_cap_relaxes_token_target_without_truncating_large_entries() -> None:
    entries = [{"output": "x" * 12_000}, {"constraint": "keep this exact wording"}]
    assert plan_compaction_chunks(entries, target_tokens=256, max_chunks=1) == [2]
    assert plan_compaction_chunks([], target_tokens=256, max_chunks=8) == []


def test_stage_plan_is_frozen_and_legacy_jobs_keep_original_boundaries() -> None:
    source = {"kind": "llm_context", "entries": [{"fact": i} for i in range(20)]}
    job = _new_job_payload(
        "process", CompactProcessContextArgs(max_chunks=8), "context", 1,
        source, estimate_tokens(source), chunk_target_tokens=16_384,
    )
    source["entries"].clear()
    assert job["chunk_end_indices"] == [20]
    assert len(_job_entry_chunks(deepcopy(job))) == 1
    legacy = deepcopy(job)
    del legacy["chunk_end_indices"]
    assert _job_entry_chunks(legacy) == _entry_chunks(legacy["source_payload"], 8)
    assert len(_job_entry_chunks(legacy)) > 1


@pytest.mark.parametrize("boundaries", [None, [], [0, 3], [2, 2, 3], [4], [True, 3], [1.0, 3], [2]])
def test_invalid_stage_plan_cannot_omit_or_reprocess_source_entries(boundaries) -> None:
    with pytest.raises(ToolExecutionError, match="Invalid context compaction chunk plan"):
        _job_entry_chunks({
            "source_payload": {"entries": [{"fact": i} for i in range(3)]},
            "max_chunks": 3,
            "chunk_end_indices": boundaries,
        })


@pytest.mark.parametrize("value", [0, -1, True, 256.5])
def test_chunk_token_target_requires_a_positive_integer(value) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, llm_context=replace(
            DEFAULT_CONFIG.llm_context, compaction_chunk_target_tokens=value,
        ))
