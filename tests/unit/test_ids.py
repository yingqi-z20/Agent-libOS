from __future__ import annotations

from datetime import datetime, timezone

import pytest

import agent_libos.utils.ids as ids


def test_utc_now_is_strictly_increasing_when_wall_clock_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return fixed if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(ids, "datetime", FrozenDateTime)
    monkeypatch.setattr(ids, "_last_utc_now", None)

    first = ids.utc_now()
    second = ids.utc_now()

    assert first == fixed.isoformat()
    assert second == fixed.replace(microsecond=1).isoformat()
