from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import CapabilityRight, ProcessStatus

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.llm_context_probe import last_tool_result, static_prefix
    from scripts.runtime_assembly import aopen_runtime
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from llm_context_probe import last_tool_result, static_prefix
    from runtime_assembly import aopen_runtime

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts


@dataclass
class ProcessPlan:
    label: str
    actions: list[dict[str, Any]]


async def run_interleaved_clock_demo(
    *,
    db: str = _RUNTIME_DEFAULTS.local_store_target,
    iterations: int = _SCRIPT_DEFAULTS.clock_demo_iterations,
    interval_s: float = _SCRIPT_DEFAULTS.clock_demo_interval_s,
    offset_s: float | None = None,
    timezone: str = _SCRIPT_DEFAULTS.clock_demo_timezone,
    echo: bool = True,
) -> dict[str, Any]:
    runtime = await aopen_runtime(db)
    outputs: list[dict[str, Any]] = []
    output_lock = threading.Lock()
    client = InterleavingClockClient()
    runtime.llm.client = client

    def output_sink(message: str) -> None:
        label_match = re.match(r"^\[(?P<label>[^\]]+)\]", message)
        entry = {
            "monotonic": time.monotonic(),
            "label": label_match.group("label") if label_match else None,
            "message": message,
        }
        with output_lock:
            outputs.append(entry)
        if echo:
            print(message, flush=True)

    runtime.substrate.human.output_sink = output_sink
    try:
        offset = interval_s / 2 if offset_s is None else offset_s
        pid_a = runtime.process.spawn(
            image=_RUNTIME_DEFAULTS.default_image_id,
            goal=f"Process A: output the current time {iterations} times, sleeping between outputs.",
            authority_manifest={
                "authorized_capabilities": [
                    {"resource": _RUNTIME_DEFAULTS.default_human_resource, "rights": ["write"]},
                ],
                "permitted_effects": ["llm.*", "clock.*", "human.*"],
                "metadata": {"provided_by": "async_clock_interleave_smoke"},
            },
        )
        pid_b = runtime.process.spawn(
            image=_RUNTIME_DEFAULTS.default_image_id,
            goal=f"Process B: sleep {offset:.3f}s first, then output the current time {iterations} times.",
            authority_manifest={
                "authorized_capabilities": [
                    {"resource": _RUNTIME_DEFAULTS.default_human_resource, "rights": ["write"]},
                ],
                "permitted_effects": ["llm.*", "clock.*", "human.*"],
                "metadata": {"provided_by": "async_clock_interleave_smoke"},
            },
        )
        for pid in (pid_a, pid_b):
            runtime.capability.grant(pid, "clock:*", [CapabilityRight.READ], issued_by="script")
            runtime.skills.activate_skill(
                pid,
                "agent-libos-runtime-session",
                actor=pid,
            )
        client.configure(
            pid_a,
            label="A",
            iterations=iterations,
            interval_s=interval_s,
            initial_delay_s=0.0,
            timezone=timezone,
        )
        client.configure(
            pid_b,
            label="B",
            iterations=iterations,
            interval_s=interval_s,
            initial_delay_s=offset,
            timezone=timezone,
        )

        max_quanta = 2 * (iterations * 3 + 2)
        results = await runtime.arun_until_idle(max_quanta=max_quanta)
        statuses = {pid_a: runtime.process.get(pid_a).status, pid_b: runtime.process.get(pid_b).status}
        expected_labels = [label for _ in range(iterations) for label in ("A", "B")]
        actual_labels = [entry["label"] for entry in outputs if entry["label"] in {"A", "B"}]
        report = {
            "pids": {"A": pid_a, "B": pid_b},
            "iterations": iterations,
            "interval_s": interval_s,
            "offset_s": offset,
            "timezone": timezone,
            "outputs": outputs,
            "expected_order": expected_labels,
            "actual_order": actual_labels,
            "interleaved": actual_labels == expected_labels,
            "process_statuses": {pid: status.value for pid, status in statuses.items()},
            "scheduler_results": len(results),
            "model_calls": client.calls,
        }
        if any(status != ProcessStatus.EXITED for status in statuses.values()):
            raise RuntimeError(f"processes did not exit cleanly: {report['process_statuses']}")
        if not report["interleaved"]:
            raise RuntimeError(f"unexpected output order: {actual_labels}, expected {expected_labels}")
        return report
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


class InterleavingClockClient:
    def __init__(self) -> None:
        self._plans: dict[str, ProcessPlan] = {}
        self._lock = threading.Lock()
        self.calls = 0

    def configure(
        self,
        pid: str,
        *,
        label: str,
        iterations: int,
        interval_s: float,
        initial_delay_s: float,
        timezone: str,
    ) -> None:
        actions: list[dict[str, Any]] = []
        if initial_delay_s > 0:
            actions.append({"action": "sleep", "seconds": initial_delay_s})
        # Each loop needs three quanta: read clock, output the observed time,
        # then sleep. The async scheduler should interleave the two pid tasks.
        for iteration in range(1, iterations + 1):
            actions.append({"action": "get_current_time", "timezone": timezone})
            actions.append({"action": "human_output", "label": label, "iteration": iteration, "from_last_time": True})
            if iteration < iterations:
                actions.append({"action": "sleep", "seconds": interval_s})
        actions.append({"action": "process_exit", "payload": {"label": label, "iterations": iterations}})
        self._plans[pid] = ProcessPlan(label=label, actions=actions)

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        pid = self._pid_from_messages(messages)
        with self._lock:
            self.calls += 1
            plan = self._plans.get(pid)
            if plan is None:
                raise AssertionError(f"no action plan registered for pid {pid}")
            if not plan.actions:
                raise AssertionError(f"no planned action remains for pid {pid}")
            action = dict(plan.actions.pop(0))
        if action.pop("from_last_time", False):
            iso8601 = self._last_tool_time(messages)
            label = action.pop("label")
            iteration = action.pop("iteration")
            action["message"] = f"[{label}] iteration={iteration} time={iso8601}"
            action["channel"] = _RUNTIME_DEFAULTS.terminal_channel
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(
            content="",
            tool_calls=[{"id": f"clock_{self.calls}", "name": name, "arguments": json.dumps(args)}],
        )

    def _pid_from_messages(self, messages: list[dict[str, str]]) -> str:
        pid = static_prefix(messages).get("pid")
        if not isinstance(pid, str) or not pid:
            for message in messages:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                match = re.search(
                    r"(?m)^(?:Process|Process facts):\n- pid: (?P<pid>\S+)$",
                    content,
                )
                if match is not None:
                    pid = match.group("pid")
                    break
        if not isinstance(pid, str) or not pid:
            raise AssertionError("prompt did not include process pid")
        return pid

    def _last_tool_time(self, messages: list[dict[str, str]]) -> str:
        result = last_tool_result(messages, "get_current_time")
        if result is None or not isinstance(result.get("iso8601"), str):
            raise AssertionError("prompt did not include a get_current_time tool result")
        return result["iso8601"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two async-scheduled processes that alternate current-time output.")
    parser.add_argument(
        "--db",
        default=_RUNTIME_DEFAULTS.local_store_target,
        help=f"Runtime SQLite database path, or '{_RUNTIME_DEFAULTS.local_store_target}' for in-memory.",
    )
    parser.add_argument("--iterations", type=int, default=_SCRIPT_DEFAULTS.clock_demo_iterations)
    parser.add_argument("--interval", type=float, default=_SCRIPT_DEFAULTS.clock_demo_interval_s)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--timezone", default=_SCRIPT_DEFAULTS.clock_demo_timezone)
    parser.add_argument("--quiet", action="store_true", help="Only print the final JSON report.")
    args = parser.parse_args()
    report = asyncio.run(
        run_interleaved_clock_demo(
            db=args.db,
            iterations=args.iterations,
            interval_s=args.interval,
            offset_s=args.offset,
            timezone=args.timezone,
            echo=not args.quiet,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
