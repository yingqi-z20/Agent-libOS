from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Callable

import pytest

from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.storage.sqlite import SQLiteStore


class _StringSubclass(str):
    pass


class _Audit:
    def record(self, **_kwargs: object) -> None:
        return None


class _Events:
    def emit(self, *_args: object, **_kwargs: object) -> None:
        return None


def _new_lifecycle(*, opened: bool = False) -> tuple[SQLiteStore, RuntimeLifecycle]:
    store = SQLiteStore(":memory:")
    lifecycle = RuntimeLifecycle(
        store=store,
        audit=_Audit(),
        events=_Events(),
        substrate=None,
    )
    if opened:
        lifecycle.begin_recovery()
        lifecycle.begin_starting()
        lifecycle.mark_open()
    return store, lifecycle


def _invalid_boolean_values() -> tuple[object, ...]:
    return (None, 0, 1, 0.0, 1.0, "", "false", [], {}, object())


def _invalid_publication_ids() -> tuple[object, ...]:
    return (
        None,
        0,
        1,
        False,
        True,
        "",
        " ",
        "\tpublication_1",
        _StringSubclass("publication_1"),
        object(),
    )


def _assert_text_free_error(
    error: dict[str, str],
    *,
    error_type: str,
    hidden_text: str,
    observed_text: str | None = None,
) -> None:
    assert error["error_type"] == error_type
    assert error["code"] == "internal_error"
    assert error["correlation_id"] in error["error"]
    assert hidden_text not in repr(error)
    encoded = (hidden_text if observed_text is None else observed_text).encode(
        "utf-8"
    )
    assert error["error_text_bytes"] == str(len(encoded))
    assert error["error_text_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_lifecycle_boolean_arguments_require_exact_bool_before_mutation() -> None:
    store, lifecycle = _new_lifecycle(opened=True)
    finalizer = lambda: True
    try:
        for invalid_boolean in _invalid_boolean_values():
            before_finalizers = lifecycle.finalizers_snapshot()
            with pytest.raises(TypeError, match="recovery_safe.*exact boolean"):
                lifecycle.bind_finalizer(
                    finalizer,
                    recovery_safe=invalid_boolean,  # type: ignore[arg-type]
                )
            assert lifecycle.finalizers_snapshot() == before_finalizers
            assert lifecycle.state == "open"

            with pytest.raises(TypeError, match="read_only.*exact boolean"):
                with lifecycle.admit(
                    read_only=invalid_boolean,  # type: ignore[arg-type]
                ):
                    raise AssertionError("invalid admission must not be yielded")
            assert lifecycle._active_leases == 0
            assert lifecycle.state == "open"

        with lifecycle.admit():
            assert lifecycle._active_leases == 1
            with pytest.raises(TypeError, match="read_only.*exact boolean"):
                with lifecycle.admit(read_only=1):  # type: ignore[arg-type]
                    raise AssertionError("invalid nested admission must not be yielded")
            assert lifecycle._active_leases == 1
        assert lifecycle._active_leases == 0
    finally:
        if not lifecycle.closed:
            assert lifecycle.shutdown(actor="test", reason="validation-cleanup")[
                "ok"
            ] is True
        store.close()


def test_lifecycle_publication_ids_fail_closed_before_fence_or_context_changes() -> None:
    store, lifecycle = _new_lifecycle(opened=True)
    capability = lifecycle._issue_recovery_terminalization_capability()
    try:
        with lifecycle.admit():
            baseline_epoch = lifecycle._recovery_fence_epoch
            for invalid_publication_id in _invalid_publication_ids():
                with pytest.raises(TypeError, match="publication id.*exact"):
                    lifecycle.mark_recovery_required(
                        publication_id=invalid_publication_id,  # type: ignore[arg-type]
                    )
                assert lifecycle.state == "open"
                assert lifecycle.shutdown_reason is None
                assert lifecycle._recovery_fence_epoch == baseline_epoch

                with pytest.raises(TypeError, match="publication id.*exact"):
                    with lifecycle.recovery_terminalization_scope(
                        invalid_publication_id,  # type: ignore[arg-type]
                        capability=capability,
                    ):
                        raise AssertionError("invalid scope must not be yielded")
                with pytest.raises(TypeError, match="publication id.*exact"):
                    with lifecycle.recovery_terminalization_scope_if_fenced(
                        invalid_publication_id,  # type: ignore[arg-type]
                        capability=capability,
                    ):
                        raise AssertionError("invalid scope must not be yielded")
                assert lifecycle._recovery_terminalization_publication.get() is None
                assert lifecycle._recovery_fence_epoch == baseline_epoch
    finally:
        if not lifecycle.closed:
            assert lifecycle.shutdown(actor="test", reason="validation-cleanup")[
                "ok"
            ] is True
        store.close()


def test_sync_failed_assembly_cleanup_is_single_flight_and_shares_safe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, lifecycle = _new_lifecycle()
    callback_entered = threading.Event()
    release_callback = threading.Event()
    follower_joined = threading.Event()
    callback_calls = 0
    results: list[list[dict[str, str]]] = []
    raised: list[BaseException] = []

    def finalizer() -> None:
        nonlocal callback_calls
        callback_calls += 1
        callback_entered.set()
        assert release_callback.wait(timeout=2)
        raise RuntimeError("SENSITIVE_SINGLE_FLIGHT_FAILURE")

    lifecycle.bind_finalizer(finalizer)
    original_start = lifecycle._start_failed_assembly_cleanup_attempt

    def observed_start(*, caller_task: asyncio.Task[object] | None):
        attempt, is_leader = original_start(caller_task=caller_task)
        if not is_leader:
            follower_joined.set()
        return attempt, is_leader

    monkeypatch.setattr(
        lifecycle,
        "_start_failed_assembly_cleanup_attempt",
        observed_start,
    )

    def cleanup() -> None:
        try:
            results.append(lifecycle.cleanup_failed_assembly())
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            raised.append(exc)

    leader = threading.Thread(target=cleanup, daemon=True)
    follower = threading.Thread(target=cleanup, daemon=True)
    try:
        leader.start()
        assert callback_entered.wait(timeout=2)
        follower.start()
        assert follower_joined.wait(timeout=2)
        release_callback.set()
        leader.join(timeout=2)
        follower.join(timeout=2)

        assert not leader.is_alive()
        assert not follower.is_alive()
        assert raised == []
        assert callback_calls == 1
        assert len(results) == 2
        assert results[0] == results[1]
        assert results[0] is not results[1]
        assert results[0][0] is not results[1][0]
        _assert_text_free_error(
            results[0][0],
            error_type="RuntimeError",
            hidden_text="SENSITIVE_SINGLE_FLIGHT_FAILURE",
        )
    finally:
        release_callback.set()
        leader.join(timeout=2)
        follower.join(timeout=2)
        store.close()


def test_async_failed_assembly_interrupt_is_leader_local_and_follower_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, lifecycle = _new_lifecycle()
    callback_calls = 0
    follower_joined = asyncio.Event()
    callback_entered = asyncio.Event()
    release_callback = asyncio.Event()

    async def finalizer() -> None:
        nonlocal callback_calls
        callback_calls += 1
        callback_entered.set()
        await release_callback.wait()
        raise asyncio.CancelledError("SENSITIVE_ASYNC_SINGLE_FLIGHT_INTERRUPT")

    lifecycle.bind_finalizer(finalizer)
    original_start = lifecycle._start_failed_assembly_cleanup_attempt

    def observed_start(*, caller_task: asyncio.Task[object] | None):
        attempt, is_leader = original_start(caller_task=caller_task)
        if not is_leader:
            follower_joined.set()
        return attempt, is_leader

    monkeypatch.setattr(
        lifecycle,
        "_start_failed_assembly_cleanup_attempt",
        observed_start,
    )

    async def exercise() -> None:
        leader = asyncio.create_task(lifecycle.acleanup_failed_assembly())
        await asyncio.wait_for(callback_entered.wait(), timeout=2)
        follower = asyncio.create_task(lifecycle.acleanup_failed_assembly())
        await asyncio.wait_for(follower_joined.wait(), timeout=2)
        release_callback.set()

        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.wait_for(leader, timeout=2)
        assert caught.value.subgroup(asyncio.CancelledError) is not None
        follower_result = await asyncio.wait_for(follower, timeout=2)
        assert len(follower_result) == 1
        _assert_text_free_error(
            follower_result[0],
            error_type="CancelledError",
            hidden_text="SENSITIVE_ASYNC_SINGLE_FLIGHT_INTERRUPT",
            observed_text="",
        )

    try:
        asyncio.run(exercise())
        assert callback_calls == 1
    finally:
        store.close()


def test_failed_assembly_cleanup_reentrancy_rejects_without_deadlock() -> None:
    store, lifecycle = _new_lifecycle()
    reentrant_errors: list[RuntimeError] = []

    def finalizer() -> None:
        try:
            lifecycle.cleanup_failed_assembly()
        except RuntimeError as exc:
            reentrant_errors.append(exc)

    lifecycle.bind_finalizer(finalizer)
    try:
        assert lifecycle.cleanup_failed_assembly() == []
        assert len(reentrant_errors) == 1
        assert "reentrant failed assembly cleanup" in str(reentrant_errors[0])
    finally:
        store.close()
