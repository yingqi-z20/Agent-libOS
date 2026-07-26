from __future__ import annotations

import asyncio

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import ProcessStatus


_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime


async def run_coding(goal: str) -> str:
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")
    runtime = await Runtime.aopen(_RUNTIME_DEFAULTS.local_store_target)
    try:
        pid = runtime.process.spawn(
            image=_RUNTIME_DEFAULTS.coding_image_id,
            goal=goal.strip(),
        )
        await runtime.arun_until_idle()
        process = runtime.process.get(pid)
        if process.status != ProcessStatus.EXITED:
            raise RuntimeError(
                f"coding process did not exit; status={process.status.value}"
            )
        return pid
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def main() -> None:
    asyncio.run(run_coding(input("Write your goal here:")))


if __name__ == "__main__":
    main()
