from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import zipfile

import pytest

from scripts import async_clock_interleave_smoke as clock_smoke
from scripts.benchmark_core_latency import _compare
from scripts.check_release_artifacts import (
    _looks_like_secret_file,
    _validate_wheel,
)
from scripts.object_memory_file_copy_smoke import _contains_text
from scripts.workflow_evidence import has_committed_filesystem_write
from tests.unit.test_release_contract import _write_test_wheel


def test_clock_smoke_rejects_zero_iterations_before_opening_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    async def unexpected_open(_target: str):
        nonlocal opened
        opened = True
        raise AssertionError("runtime must not open for invalid controls")

    monkeypatch.setattr(clock_smoke, "aopen_runtime", unexpected_open)

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        asyncio.run(clock_smoke.run_interleaved_clock_demo(iterations=0))

    assert opened is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0, True, "1"])
def test_latency_comparison_rejects_invalid_baseline_numbers(
    tmp_path: Path,
    value: object,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "operations": {
                    "open_close": {
                        "median_ms": value,
                        "p95_ms": 1.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline 'open_close' median_ms"):
        _compare(
            {"open_close": {"median_ms": 1.0, "p95_ms": 1.0}},
            baseline,
            max_median_ratio=1.1,
            max_p95_ratio=1.2,
        )


def test_secret_name_detection_covers_env_variants_and_wheel_payloads(
    tmp_path: Path,
) -> None:
    assert _looks_like_secret_file(PurePosixPath("agent_libos/.env.production"))
    assert _looks_like_secret_file(PurePosixPath("agent_libos/.ENV.Local"))

    wheel = _write_test_wheel(tmp_path / "agent_libos-0.3.4-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("agent_libos/.env.production", b"TOKEN=not-for-release\n")

    with pytest.raises(ValueError, match="wheel contains a secret-like file"):
        _validate_wheel(wheel, "0.3.4")


def test_object_copy_visibility_check_uses_structured_text_not_json_encoding() -> None:
    source = 'line one\n"quoted" \\ slash\n'
    visible = {"nested": [{"content": source}]}

    assert _contains_text(visible, source) is True
    assert _contains_text({"nested": [{"digest": "abc"}]}, source) is False


def test_legacy_interactive_scripts_are_import_safe_and_chat_goal_is_text() -> None:
    from scripts import testChat, testCoding

    assert isinstance(testChat.CHAT_GOAL, str)
    assert "ask_human" in testChat.CHAT_GOAL
    assert callable(testChat.run_chat)
    assert callable(testCoding.run_coding)
    assert not hasattr(testChat, "runtime")
    assert not hasattr(testCoding, "runtime")


def test_write_evidence_requires_matching_receipt_and_committed_exact_effect() -> None:
    expected_path = "agent_outputs/result.txt"
    resource = f"filesystem:workspace:{expected_path}"
    effect = SimpleNamespace(
        operation="filesystem.write_text",
        target=resource,
        effect_state="finalized",
        transaction_state="committed",
        state_mutation=True,
    )
    runtime = SimpleNamespace(
        filesystem=SimpleNamespace(resource_for=lambda path: f"filesystem:workspace:{path}"),
        store=SimpleNamespace(list_external_effects=lambda **_kwargs: [effect]),
    )
    receipt = {
        "ok": True,
        "action": {"action": "write_text_file", "path": expected_path},
        "result": {"path": expected_path, "bytes_written": 7},
    }

    assert has_committed_filesystem_write(
        runtime,
        "pid-1",
        [receipt],
        expected_path,
    )
    assert not has_committed_filesystem_write(
        runtime,
        "pid-1",
        [{**receipt, "action": {"action": "write_text_file", "path": "other.txt"}}],
        expected_path,
    )
    effect.transaction_state = "unknown"
    assert not has_committed_filesystem_write(
        runtime,
        "pid-1",
        [receipt],
        expected_path,
    )
