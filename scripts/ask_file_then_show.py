from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import ProcessStatus
from agent_libos.utils.serde import to_jsonable

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.llm_context_probe import last_tool_result, recent_events
    from scripts.runtime_assembly import aopen_runtime
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from llm_context_probe import last_tool_result, recent_events
    from runtime_assembly import aopen_runtime

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the human which workspace file to view, then show that file's content through HumanObject output."
    )
    parser.add_argument(
        "--db",
        default=_RUNTIME_DEFAULTS.local_store_target,
        help=f"Runtime SQLite database path, or '{_RUNTIME_DEFAULTS.local_store_target}' for in-memory.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=_SCRIPT_DEFAULTS.ask_file_max_bytes,
        help="Maximum bytes to read from the selected file.",
    )
    parser.add_argument(
        "--max-quanta",
        type=int,
        default=_SCRIPT_DEFAULTS.ask_file_max_quanta,
        help="Maximum Agent execution quanta to run.",
    )
    parser.add_argument(
        "--auto-answer",
        default=None,
        help="Non-interactive answer to the file-name question, for example README.md.",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run_file_viewer(
            db=args.db,
            max_bytes=args.max_bytes,
            max_quanta=args.max_quanta,
            auto_answer=args.auto_answer,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


async def run_file_viewer(
    *,
    db: str = _RUNTIME_DEFAULTS.local_store_target,
    max_bytes: int = _SCRIPT_DEFAULTS.ask_file_max_bytes,
    max_quanta: int = _SCRIPT_DEFAULTS.ask_file_max_quanta,
    auto_answer: str | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    runtime = await aopen_runtime(db)
    outputs: list[str] = []
    client = AskFileViewerClient(max_bytes=max_bytes)
    runtime.llm.client = client

    def output_sink(message: str) -> None:
        outputs.append(message)
        if echo:
            print(message, flush=True)

    runtime.substrate.human.output_sink = output_sink
    try:
        pid = runtime.process.spawn(
            image=_RUNTIME_DEFAULTS.coding_image_id,
            goal=(
                "Ask the human which workspace file they want to view. Read that file and show its content "
                "to the human. If reading fails, show the failure reason to the human. Then exit."
            ),
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": _RUNTIME_DEFAULTS.default_human_resource,
                        "rights": ["read", "write"],
                    },
                    {
                        "resource": runtime.filesystem.workspace_resource(),
                        "rights": ["read"],
                    },
                ],
                "permitted_effects": ["llm.*", "human.*", "filesystem.*"],
                "metadata": {"provided_by": "ask_file_then_show"},
            },
        )
        results = await runtime.arun_until_idle(
            max_quanta=max_quanta,
            pids=(pid,),
            human_auto_answer=auto_answer,
        )
        process = runtime.process.get(pid)
        report = {
            "pid": pid,
            "selected_path": client.selected_path,
            "displayed": client.displayed,
            "error": client.error,
            "process_status": process.status.value,
            "actions": [_action_name(result) for result in results],
            "outputs": outputs,
            "model_calls": client.calls,
            "results": to_jsonable(results),
        }
        if process.status != ProcessStatus.EXITED:
            raise RuntimeError(f"process did not exit after {max_quanta} quanta; status={process.status.value}")
        return report
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


class AskFileViewerClient:
    def __init__(self, *, max_bytes: int):
        self.max_bytes = max_bytes
        self.calls = 0
        self.step = 0
        self.selected_path: str | None = None
        self.displayed = False
        self.error: str | None = None

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        if self.step == 0:
            self.step = 1
            return self._completion(
                "discover_skills",
                {"text": "human collaboration", "limit": 5},
            )
        if self.step == 1:
            package_sha256 = self._discovered_skill_hash(
                messages,
                "agent-libos-human-collaboration",
            )
            self.step = 2
            return self._completion(
                "activate_skill",
                {
                    "skill_id": "agent-libos-human-collaboration",
                    "expected_package_sha256": package_sha256,
                },
            )
        if self.step == 2:
            # Drive the process through real Skill, Human, and filesystem
            # lifecycles while keeping the script deterministic and token-free.
            self.step = 3
            return self._completion(
                "ask_human",
                {
                    "question": "Which workspace file do you want to view?",
                    "context": {"path_rule": "Use a path under the runtime workspace root."},
                },
            )
        if self.step == 3:
            answer = self._last_tool_result(messages, "ask_human").get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise AssertionError("ask_human result did not include a non-empty answer")
            self.selected_path = answer.strip()
            self.step = 4
            return self._completion(
                "discover_skills",
                {"text": "workspace read text file", "limit": 5},
            )
        if self.step == 4:
            package_sha256 = self._discovered_skill_hash(
                messages,
                "agent-libos-workspace-navigation",
            )
            self.step = 5
            return self._completion(
                "activate_skill",
                {
                    "skill_id": "agent-libos-workspace-navigation",
                    "expected_package_sha256": package_sha256,
                },
            )
        if self.step == 5:
            self.step = 6
            return self._completion(
                "read_text_file",
                {"path": self.selected_path, "max_bytes": self.max_bytes},
            )
        if self.step == 6:
            read_result = self._last_tool_result(messages, "read_text_file", required=False)
            if read_result is None:
                self.error = self._last_tool_error(messages) or "read_text_file failed without a visible error"
                message = f"Could not read {self.selected_path!r}: {self.error}"
            else:
                content = str(read_result.get("content", ""))
                truncated = bool(read_result.get("truncated", False))
                suffix = "\n\n[content truncated]" if truncated else ""
                message = f"----- {self.selected_path} -----\n{content}{suffix}"
                self.displayed = True
            self.step = 7
            return self._completion("human_output", {"message": message})
        if self.step == 7:
            self.step = 8
            return self._completion(
                "process_exit",
                {
                    "payload": {
                        "selected_path": self.selected_path,
                        "displayed": self.displayed,
                        "error": self.error,
                    }
                },
            )
        if self.step == 8:
            review_result = self._last_tool_result(messages, "process_exit")
            review = review_result.get("completion_review")
            if not isinstance(review, dict):
                raise AssertionError("process_exit did not return a completion review")
            goal = review.get("goal")
            if not isinstance(goal, dict) or not isinstance(goal.get("oid"), str):
                raise AssertionError("completion review did not include the goal oid")
            review_token = review.get("review_token")
            message_ids = review.get("acknowledged_human_message_ids")
            if not isinstance(review_token, str) or not isinstance(message_ids, list):
                raise AssertionError("completion review identity is incomplete")
            evidence_tools = ["ask_human", "human_output"]
            if self.displayed:
                evidence_tools.insert(1, "read_text_file")
            self.step = 9
            return self._completion(
                "process_exit",
                {
                    "review_token": review_token,
                    "completion_evidence": {
                        "goal_oid": goal["oid"],
                        "reviewed_message_ids": message_ids,
                        "acceptance_checks": [
                            {
                                "requirement": "Ask which workspace file to view.",
                                "source_refs": [goal["oid"]],
                                "status": "completed",
                                "evidence_tool_calls": ["ask_human"],
                                "evidence_summary": "The human selected a workspace path.",
                            },
                            {
                                "requirement": "Read the file or report the read failure.",
                                "source_refs": [goal["oid"]],
                                "status": "completed",
                                "evidence_tool_calls": evidence_tools,
                                "evidence_summary": "The selected path was handled and its outcome was reported.",
                            },
                            {
                                "requirement": "Show the outcome to the human.",
                                "source_refs": [goal["oid"]],
                                "status": "completed",
                                "evidence_tool_calls": ["human_output"],
                                "evidence_summary": "The final file content or failure was delivered.",
                            },
                        ],
                        "final_verification": evidence_tools,
                    },
                    "payload": {
                        "selected_path": self.selected_path,
                        "displayed": self.displayed,
                        "error": self.error,
                    },
                },
            )
        raise AssertionError("file viewer action plan is already complete")

    def _discovered_skill_hash(
        self,
        messages: list[dict[str, str]],
        skill_id: str,
    ) -> str:
        discovered = self._last_tool_result(messages, "discover_skills")
        skills = discovered.get("skills")
        if isinstance(skills, list):
            for item in skills:
                if not isinstance(item, dict) or item.get("skill_id") != skill_id:
                    continue
                package_sha256 = item.get("package_sha256")
                if (
                    isinstance(package_sha256, str)
                    and len(package_sha256) == 64
                    and all(character in "0123456789abcdef" for character in package_sha256)
                ):
                    return package_sha256
                raise AssertionError(
                    f"discover_skills returned invalid package hash for {skill_id}"
                )
        raise AssertionError(
            f"discover_skills did not return expected Skill {skill_id}"
        )

    def _completion(self, name: str, args: dict[str, Any]) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[{"id": f"file_viewer_{self.calls}", "name": name, "arguments": json.dumps(args)}],
        )

    def _last_tool_result(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        result = last_tool_result(messages, tool_name)
        if result is not None:
            return result
        if required:
            raise AssertionError(f"no visible result for {tool_name}")
        return None

    def _last_tool_error(self, messages: list[dict[str, str]]) -> str | None:
        for event in reversed(recent_events(messages)):
            if event.get("type") != "tool_failed":
                continue
            payload = event.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                return payload["error"]
        return None


def _action_name(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    action = result.get("action")
    if isinstance(action, dict):
        return action.get("action")
    return None


if __name__ == "__main__":
    main()
