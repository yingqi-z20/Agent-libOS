from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_libos.llm.usage import LLM_USAGE_COUNTER_MAX
from agent_libos.models import LLMCallRecord
from agent_libos.storage import SQLiteStore
from experiments.inspect_long_horizon_run import inspect_database


def test_inspector_keeps_committed_wal_calls_across_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runtime ? snapshot.sqlite"
    store = SQLiteStore(path)
    store.insert_llm_call(LLMCallRecord(
        call_id="before-wal", pid="pid", image_id=None,
        purpose="action_selection", status="ok", messages=[], tools=[],
        tool_calls=[], created_at="2026-09-08T00:00:00+00:00",
    ))
    store.close()
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("UPDATE llm_calls SET call_id = 'committed-in-wal'")
        writer.commit()
        assert path.with_name(path.name + "-wal").stat().st_size > 0
        copy = shutil.copy2

        def checkpoint_between_file_copies(source, destination, *args, **kwargs):
            result = copy(source, destination, *args, **kwargs)
            if Path(source) == path:
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return result

        # Exercise the race that a filesystem-copy implementation cannot
        # coordinate with; SQLite's backup API reads the committed WAL itself.
        monkeypatch.setattr(shutil, "copy2", checkpoint_between_file_copies)
        summary = inspect_database(path, pid="pid")
        assert [call["call_id"] for call in summary["calls"]] == ["committed-in-wal"]
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert inspect_database(path)["calls"] == summary["calls"]
        assert inspect_database(path, pid="other")["calls"] == []


def test_inspector_reports_exclusively_locked_runtime(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "locked.sqlite")
    try:
        with pytest.raises(sqlite3.OperationalError, match="close the owning Runtime"):
            inspect_database(tmp_path / "locked.sqlite")
    finally:
        store.close()


def test_inspector_does_not_modify_the_source_schema(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite"
    store = SQLiteStore(path)
    store.close()
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    assert inspect_database(path)["calls"] == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == before


@pytest.mark.parametrize(
    ("api", "usage", "expected"),
    [
        ("chat", {"input_tokens": 17, "output_tokens": 9}, (17, 9, 0, 0)),
        ("responses", {"prompt_tokens": 17, "completion_tokens": 9}, (17, 9, 0, 0)),
        ("chat", {"prompt_tokens": True, "completion_tokens": -1,
                  "input_tokens": 17, "output_tokens": 9}, (17, 9, 0, 0)),
        ("responses", {"input_tokens": "17", "output_tokens": LLM_USAGE_COUNTER_MAX + 1},
         (0, 0, 0, 0)),
        ("responses", {"input_tokens": 17, "output_tokens": 9,
                       "input_tokens_details": {"cached_tokens": 0},
                       "cache_read_tokens": 12,
                       "output_tokens_details": {"reasoning_tokens": 4},
                       "reasoning_tokens": 6}, (17, 9, 0, 4)),
        ("responses", {"input_tokens": 17, "output_tokens": 9,
                       "input_tokens_details": {"cached_tokens": None},
                       "cache_read_tokens": 12,
                       "output_tokens_details": {"reasoning_tokens": 10}},
         (17, 9, 0, 0)),
    ],
    ids=["chat-fallback", "responses-fallback", "invalid-primary-fallback",
         "invalid-counters", "formal-zero-and-reasoning", "invalid-details"],
)
def test_inspector_uses_shared_provider_usage_normalization(
    tmp_path: Path,
    api: str,
    usage: dict[str, Any],
    expected: tuple[int, int, int, int],
) -> None:
    path = tmp_path / "runtime.sqlite"
    store = SQLiteStore(path)
    try:
        store.insert_llm_call(LLMCallRecord(
            call_id="usage", pid="pid", image_id=None,
            purpose="action_selection", status="ok", api=api,
            messages=[], tools=[], tool_calls=[], usage=usage,
            created_at="2026-09-08T00:00:00+00:00",
        ))
    finally:
        store.close()

    summary = inspect_database(path)
    call = summary["calls"][0]
    keys = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")
    assert tuple(call[key] for key in keys) == expected
    assert tuple(summary["totals"][key] for key in keys) == expected
