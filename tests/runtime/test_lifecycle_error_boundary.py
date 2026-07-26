from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest

from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.storage.sqlite import SQLiteStore


class _BrokenTextError(RuntimeError):
    def __str__(self) -> str:
        raise KeyboardInterrupt("exception text must not execute across the boundary")


@pytest.mark.parametrize(
    ("error_factory", "sensitive_text", "observed_text"),
    [
        pytest.param(
            lambda: RuntimeError("SENSITIVE_SHUTDOWN_SENTINEL"),
            "SENSITIVE_SHUTDOWN_SENTINEL",
            "SENSITIVE_SHUTDOWN_SENTINEL",
            id="runtime-error",
        ),
        pytest.param(
            lambda: _BrokenTextError("BROKEN_TEXT_SECRET"),
            "BROKEN_TEXT_SECRET",
            "exception text is unavailable",
            id="broken-exception-text",
        ),
    ],
)
def test_shutdown_failure_returns_text_free_correlated_error_observation(
    error_factory: Callable[[], Exception],
    sensitive_text: str,
    observed_text: str,
) -> None:
    store = SQLiteStore(":memory:")

    class _Audit:
        def record(self, **_kwargs: object) -> None:
            return None

    class _Events:
        def emit(self, *_args: object, **_kwargs: object) -> None:
            return None

    lifecycle = RuntimeLifecycle(
        store=store,
        audit=_Audit(),
        events=_Events(),
        substrate=None,
    )
    lifecycle.begin_recovery()
    lifecycle.begin_starting()
    lifecycle.mark_open()
    selected_error = error_factory()
    calls = 0

    def finalizer() -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise selected_error
        return True

    lifecycle.bind_finalizer(finalizer)
    try:
        first = lifecycle.shutdown(actor="test", reason="error-boundary")
        assert first["ok"] is False
        assert sensitive_text not in repr(first)
        error = first["errors"][0]
        assert error["component"].startswith("finalizer_")
        assert error["code"] == "internal_error"
        assert error["error_type"] == type(selected_error).__name__
        assert error["correlation_id"] in error["error"]
        encoded = observed_text.encode("utf-8")
        assert error["error_text_bytes"] == str(len(encoded))
        assert error["error_text_sha256"] == hashlib.sha256(encoded).hexdigest()
    finally:
        if not lifecycle.closed:
            assert lifecycle.shutdown(actor="test", reason="error-boundary-retry")[
                "ok"
            ] is True
        store.close()
