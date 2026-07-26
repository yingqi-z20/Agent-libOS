from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Protocol, TypeVar


T = TypeVar("T")


def _settled_worker_error(future: Future[Any]) -> BaseException | None:
    """Return a completed worker's exception without confusing task cancellation."""

    if not future.done():
        return None
    try:
        future.result()
    except BaseException as exc:
        return exc
    return None


class BlockingWorkPort(Protocol):
    """Runtime-owned boundary for cancellable, drainable blocking work."""

    async def run(
        self,
        function: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...


async def run_blocking_once(
    function: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run standalone blocking work without an untracked default executor.

    Composed Runtime services receive the shared ``BlockingWorkSupervisor``.
    A few reusable components also support standalone construction in tests and
    embedding hosts; their fallback must still wait for the underlying thread
    after cancellation rather than returning while it can mutate shared state.
    This one-call executor is therefore owned and drained by the await itself.
    """

    context = contextvars.copy_context()
    call = partial(function, *args, **kwargs)
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-libos-standalone-blocking",
    )
    try:
        # Cleanup ownership begins as soon as the executor exists.  Both
        # ``submit`` and ``wrap_future`` are fallible, and a successful submit
        # may already have started user work before wrapping fails.
        future = executor.submit(context.run, call)
        wrapped = asyncio.wrap_future(future)
        cancelled = False
        caller_task = asyncio.current_task()
        initial_cancellations = caller_task.cancelling() if caller_task is not None else 0
        while True:
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError as exc:
                worker_error = _settled_worker_error(future)
                if worker_error is not None:
                    selected_error: BaseException = (
                        exc
                        if isinstance(worker_error, FutureCancelledError)
                        else worker_error
                    )
                    caller_cancelled_now = bool(
                        caller_task is not None
                        and caller_task.cancelling() > initial_cancellations
                    )
                    if cancelled or caller_cancelled_now:
                        raise BaseExceptionGroup(
                            "blocking work was cancelled while its worker failed",
                            [asyncio.CancelledError(), selected_error],
                        ) from None
                    raise selected_error
                cancelled = True
                continue
            except BaseException as exc:
                if cancelled:
                    raise BaseExceptionGroup(
                        "blocking work was cancelled while its worker failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            if cancelled:
                raise asyncio.CancelledError()
            return result
    finally:
        # The loop above exits only after ``future`` is done, so this cannot
        # strand provider work.  During setup failure it may wait for a
        # successfully submitted worker, which is the required owned-resource
        # lifecycle rather than returning while work remains live.
        executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["BlockingWorkPort", "run_blocking_once"]
