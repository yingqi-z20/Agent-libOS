from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

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
    completion_payload: dict[str, Any]


def _find_completion_review(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            return _find_completion_review(json.loads(value))
        except json.JSONDecodeError:
            for line in value.splitlines():
                candidate = line.strip()
                if not candidate.startswith("{"):
                    continue
                try:
                    found = _find_completion_review(json.loads(candidate))
                except json.JSONDecodeError:
                    continue
                if found is not None:
                    return found
            return None
    if isinstance(value, list):
        for item in value:
            found = _find_completion_review(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    review = value.get("completion_review")
    if isinstance(review, dict) and isinstance(review.get("review_token"), str):
        return review
    for item in value.values():
        found = _find_completion_review(item)
        if found is not None:
            return found
    return None


def _completion_claim(
    review: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    available = [
        str(tool)
        for tool in review.get("available_evidence_tools", [])
        if isinstance(tool, str) and tool != "process_exit"
    ]
    if not available:
        raise AssertionError("completion review exposed no successful evidence tool")
    requirements = review.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise AssertionError("completion review exposed no ordered requirements")
    checks = []
    for requirement in requirements:
        eligible = (
            requirement.get("eligible_evidence_tools")
            if isinstance(requirement, dict)
            else None
        )
        candidates = [tool for tool in available if not eligible or tool in eligible]
        checks.append(
            {
                "status": "completed",
                "evidence_tool_calls": candidates[:1],
                "evidence_summary": "Verified by the cited successful runtime tool call.",
            }
        )
    return {
        "payload": dict(payload),
        "review_token": review["review_token"],
        "completion_evidence": {
            "acceptance_checks": checks,
            "final_verification": available[:1],
        },
    }


async def run_interleaved_clock_demo(
    *,
    db: str = _RUNTIME_DEFAULTS.local_store_target,
    iterations: int = _SCRIPT_DEFAULTS.clock_demo_iterations,
    interval_s: float = _SCRIPT_DEFAULTS.clock_demo_interval_s,
    offset_s: float | None = None,
    timezone: str = _SCRIPT_DEFAULTS.clock_demo_timezone,
    echo: bool = True,
) -> dict[str, Any]:
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(interval_s, (int, float)) or isinstance(interval_s, bool):
        raise ValueError("interval_s must be a finite positive number")
    interval_s = float(interval_s)
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be a finite positive number")
    if offset_s is not None:
        if not isinstance(offset_s, (int, float)) or isinstance(offset_s, bool):
            raise ValueError("offset_s must be a finite non-negative number")
        offset_s = float(offset_s)
        if not math.isfinite(offset_s) or offset_s < 0:
            raise ValueError("offset_s must be a finite non-negative number")
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("timezone must be a non-empty string")
    timezone = timezone.strip()
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
            runtime.skills.activate_skill(
                pid,
                "agent-libos-human-collaboration",
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
        results = await runtime.arun_until_idle(
            max_quanta=max_quanta,
            pids=(pid_a, pid_b),
        )
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
        completion_payload = {"label": label, "iterations": iterations}
        actions.append({"action": "process_exit", "payload": completion_payload})
        self._plans[pid] = ProcessPlan(
            label=label,
            actions=actions,
            completion_payload=completion_payload,
        )

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        pid = self._pid_from_messages(messages)
        with self._lock:
            self.calls += 1
            plan = self._plans.get(pid)
            if plan is None:
                raise AssertionError(f"no action plan registered for pid {pid}")
            review = _find_completion_review(messages)
            if review is not None:
                action = {
                    "action": "process_exit",
                    **_completion_claim(review, plan.completion_payload),
                }
            else:
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
        if isinstance(pid, str) and pid:
            return pid
        # cache_optimized_v2 intentionally does not disclose the caller pid.
        # This deterministic fixture selects its Host-owned plan from the
        # semantic goal label instead of relying on private process identity.
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        matches = [
            candidate_pid
            for candidate_pid, plan in self._plans.items()
            if f"Process {plan.label}:" in rendered
        ]
        if len(matches) != 1:
            raise AssertionError("prompt did not identify one semantic clock plan")
        return matches[0]

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
