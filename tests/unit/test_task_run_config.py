from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG


def test_durable_task_payloads_are_explicitly_disabled_by_default() -> None:
    task_runs = DEFAULT_CONFIG.task_runs
    assert task_runs.plaintext_payloads_enabled is False
    assert task_runs.payload_max_bytes == 1_048_576
    assert task_runs.recovery_page_size == 500
    assert task_runs.recovery_page_hard_limit == 5_000
    assert 500 <= task_runs.list_hard_limit <= 5_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_max_bytes", 0),
        ("recovery_page_size", 0),
        ("recovery_page_size", 5_001),
        ("recovery_page_hard_limit", 0),
        ("list_hard_limit", 0),
    ],
)
def test_durable_task_runtime_bounds_fail_closed(field: str, value: int) -> None:
    with pytest.raises((PydanticValidationError, ValueError)):
        replace(DEFAULT_CONFIG.task_runs, **{field: value})
