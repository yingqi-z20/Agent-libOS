from __future__ import annotations

import asyncio

import pytest

from agent_libos import Runtime
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.scheduler import AsyncProcessScheduler


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "1"])
def test_runtime_run_until_idle_rejects_invalid_quantum_budget_before_effects(
    invalid: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="strict scheduler quantum budget")
        before = runtime.process.get(pid)
        audit_ids = {record.record_id for record in runtime.audit.trace()}

        with pytest.raises(ValidationError, match="max_quanta"):
            runtime.run_until_idle(max_quanta=invalid)  # type: ignore[arg-type]

        after = runtime.process.get(pid)
        assert after.status is before.status
        assert after.revision == before.revision
        assert {record.record_id for record in runtime.audit.trace()} == audit_ids
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_scheduler_rejects_invalid_cancel_flag_before_quantum(
    invalid: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="strict scheduler cancellation flag")
        called: list[str] = []
        before = runtime.process.get(pid)

        with pytest.raises(ValidationError, match="cancel_inflight"):
            runtime.scheduler.run_until_idle(
                lambda selected_pid: called.append(selected_pid),
                max_quanta=1,
                cancel_inflight_on_budget_exhaustion=invalid,  # type: ignore[arg-type]
            )

        after = runtime.process.get(pid)
        assert called == []
        assert after.status is before.status
        assert after.revision == before.revision
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid", [0, "false", None])
def test_runtime_rejects_invalid_human_queue_flag_before_effects(
    invalid: object,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="strict runtime human queue flag")
        before = runtime.process.get(pid)

        with pytest.raises(ValidationError, match="process_human_queue"):
            runtime.run_until_idle(
                max_quanta=1,
                process_human_queue=invalid,  # type: ignore[arg-type]
            )

        after = runtime.process.get(pid)
        assert after.status is before.status
        assert after.revision == before.revision
    finally:
        runtime.close()


def test_async_runtime_rejects_invalid_controls_before_blocking_dispatch() -> None:
    runtime = Runtime.open("local")
    calls: list[str] = []

    async def scenario() -> None:
        original_run = runtime.blocking_work.run

        async def tracked_run(*args: object, **kwargs: object) -> object:
            calls.append("blocking")
            return await original_run(*args, **kwargs)

        runtime.blocking_work.run = tracked_run  # type: ignore[method-assign]
        with pytest.raises(ValidationError, match="max_quanta"):
            await runtime.arun_until_idle(max_quanta=True)

    try:
        asyncio.run(scenario())
        assert calls == []
    finally:
        runtime.close()


def test_async_process_run_rejects_invalid_budget_before_scheduler_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="strict async process quantum budget")
        calls: list[str] = []
        original_run = runtime.scheduler.arun_pid_until_idle

        async def tracked_run(*args: object, **kwargs: object) -> object:
            calls.append("scheduler")
            return await original_run(*args, **kwargs)

        monkeypatch.setattr(runtime.scheduler, "arun_pid_until_idle", tracked_run)

        async def scenario() -> None:
            with pytest.raises(ValidationError, match="max_quanta"):
                await runtime.arun_process_until_idle(pid, max_quanta=True)

        asyncio.run(scenario())
        assert calls == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("max_workers", True),
        ("max_workers", 1.5),
        ("poll_interval_s", False),
        ("poll_interval_s", float("inf")),
        ("drain_window_s", -1.0),
        ("shutdown_join_timeout_s", "1"),
        ("owner_id", ""),
    ],
)
def test_scheduler_constructor_rejects_invalid_scalars_before_worker_creation(
    field: str,
    invalid: object,
) -> None:
    kwargs = {field: invalid}
    with pytest.raises(ValidationError, match=field):
        AsyncProcessScheduler(object(), object(), **kwargs)  # type: ignore[arg-type]
