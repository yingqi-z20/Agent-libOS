from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import CapabilityRight, ProcessStatus

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.runtime_assembly import aopen_runtime
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from runtime_assembly import aopen_runtime

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


DEFAULT_SOURCE_TEXT = "Object Memory copy smoke source.\nCONTENT_STAYS_OUT_OF_PROCESS_CONTEXT\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a workspace text file through named Object Memory without materializing its content to the process."
    )
    parser.add_argument(
        "--db",
        default=_RUNTIME_DEFAULTS.local_store_target,
        help=f"Runtime SQLite database path, or '{_RUNTIME_DEFAULTS.local_store_target}' for in-memory.",
    )
    parser.add_argument("--source", default="agent_outputs/object_memory_copy_source.txt")
    parser.add_argument("--target", default="agent_outputs/object_memory_copy_target.txt")
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--encoding", default=_TOOL_DEFAULTS.default_text_encoding)
    parser.add_argument("--max-quanta", type=int, default=_SCRIPT_DEFAULTS.object_copy_max_quanta)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    asyncio.run(amain(args))


async def amain(args: argparse.Namespace) -> None:
    runtime = await aopen_runtime(args.db)
    try:
        source = _workspace_relative(args.source, runtime.workspace_root)
        target = _workspace_relative(args.target, runtime.workspace_root)
        source_path = runtime.workspace_root / source
        target_path = runtime.workspace_root / target
        if not source_path.exists():
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(DEFAULT_SOURCE_TEXT, encoding=args.encoding)
        if not source_path.is_file():
            raise SystemExit(f"source is not a file: {source}")
        source_text = source_path.read_text(encoding=args.encoding)
        object_name = args.object_name or f"file.copy.{uuid4().hex}"

        client = GuardedActionClient(
            actions=[
                {
                    "action": "create_object_from_file",
                    "name": object_name,
                    "path": source,
                    "encoding": args.encoding,
                },
                {
                    "action": "write_object_to_file",
                    "name": object_name,
                    "path": target,
                    "encoding": args.encoding,
                    "overwrite": True,
                },
                {
                    "action": "process_exit",
                    "payload": {"copied": True, "object_name": object_name, "source": source, "target": target},
                },
            ],
            forbidden_text=source_text,
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(
            image="review-agent:v0",
            goal=(
                f"Copy {source!r} to {target!r} by creating named Object {object_name!r}, "
                "then writing that Object to the target. Do not call read_text_file."
            ),
        )
        runtime.skills.activate_skill(
            pid,
            "agent-libos-object-file-transfer",
            actor=pid,
        )
        runtime.filesystem.grant_path(pid, source, [CapabilityRight.READ], issued_by="object_copy_smoke")
        runtime.filesystem.grant_path(pid, target, [CapabilityRight.WRITE], issued_by="object_copy_smoke")

        results = await runtime.arun_until_idle(max_quanta=args.max_quanta)

        if runtime.process.get(pid).status != ProcessStatus.EXITED:
            raise SystemExit(f"process did not exit after {args.max_quanta} quanta")
        if not target_path.exists():
            raise SystemExit(f"target was not written: {target}")
        target_text = target_path.read_text(encoding=args.encoding)
        if target_text != source_text:
            raise SystemExit("target content does not match source content")

        action_names = [result["action"]["action"] for result in results if isinstance(result, dict) and "action" in result]
        if action_names != ["create_object_from_file", "write_object_to_file", "process_exit"]:
            raise SystemExit(f"unexpected action sequence: {action_names}")
        serialized_results = json.dumps([result.get("result") for result in results if isinstance(result, dict)], ensure_ascii=False)
        content_hidden = source_text not in serialized_results
        if not content_hidden:
            raise SystemExit("source content appeared in process-visible tool results")

        report = {
            "pid": pid,
            "object_name": object_name,
            "source": source,
            "target": target,
            "bytes_copied": len(target_text.encode(args.encoding)),
            "actions": action_names,
            "model_calls": client.calls,
            "content_materialized_to_process": False,
            "target_matches_source": True,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.trace:
            print(f"source_text_chars={len(source_text)}", file=sys.stderr)
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def _workspace_relative(raw_path: str, workspace: Path) -> str:
    workspace = workspace.resolve()
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise SystemExit(f"path must stay under workspace root: {workspace}") from exc


class GuardedActionClient:
    def __init__(self, actions: list[dict[str, object]], forbidden_text: str):
        self.actions = list(actions)
        self.forbidden_text = forbidden_text
        self.calls = 0

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        serialized_messages = json.dumps(messages, ensure_ascii=False)
        # This assertion is the point of the smoke test: copying through named
        # Object Memory must not materialize file bytes into the prompt.
        if self.forbidden_text and self.forbidden_text in serialized_messages:
            raise AssertionError("source file content was materialized into the process prompt")
        if not self.actions:
            raise AssertionError("no planned action remains")
        action = self.actions.pop(0)
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(
            content="",
            tool_calls=[{"id": f"planned_{self.calls}", "name": name, "arguments": json.dumps(args)}],
        )


if __name__ == "__main__":
    main()
