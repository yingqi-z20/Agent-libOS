from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight, ProcessStatus
from agent_libos.utils.serde import to_jsonable

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.runtime_assembly import aopen_runtime
    from scripts.workflow_evidence import has_committed_filesystem_write
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from runtime_assembly import aopen_runtime
    from workflow_evidence import has_committed_filesystem_write

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real LLM smoke test for write_text_file.")
    parser.add_argument("--path", default=f"agent_outputs/llm_goal_{uuid4().hex[:8]}.txt")
    parser.add_argument("--content", default="Agent libOS LLM write-file smoke test passed.\n")
    parser.add_argument("--max-quanta", type=int, default=_SCRIPT_DEFAULTS.llm_write_smoke_max_quanta)
    args = parser.parse_args()
    asyncio.run(amain(args))


async def amain(args: argparse.Namespace) -> None:
    runtime = await aopen_runtime(_RUNTIME_DEFAULTS.local_store_target)
    try:
        target_relative = _workspace_relative_target(args.path, runtime.workspace_root)
        goal = (
            f"Use the write_text_file tool to create workspace file {target_relative!r} "
            f"with exactly this content: {args.content!r}. After the file is written, exit."
        )
        pid = runtime.process.spawn(image=_RUNTIME_DEFAULTS.coding_image_id, goal=goal)
        runtime.filesystem.grant_path(
            pid,
            target_relative,
            [CapabilityRight.WRITE],
            issued_by="smoke-test",
        )
        results = await runtime.arun_until_idle(max_quanta=args.max_quanta)
        target = runtime.workspace_root / target_relative
        file_exists = target.exists()
        actual_content = target.read_text(encoding=_TOOL_DEFAULTS.default_text_encoding) if file_exists else None
        process = runtime.process.get(pid)
        summary = {
            "pid": pid,
            "target": str(target),
            "file_exists": file_exists,
            "content_matches": actual_content == args.content,
            "process_status": process.status.value,
            "actions": [_action_name(result) for result in results],
            "write_receipt_bound": has_committed_filesystem_write(
                runtime,
                pid,
                results,
                target_relative,
            ),
            "results": to_jsonable(results),
            "audit_records": len(runtime.audit.trace()),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if (
            not file_exists
            or actual_content != args.content
            or not summary["write_receipt_bound"]
        ):
            raise SystemExit(2)
        if process.status != ProcessStatus.EXITED:
            raise SystemExit(3)
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def _action_name(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    action = result.get("action")
    if isinstance(action, dict):
        return action.get("action")
    return None


def _workspace_relative_target(raw_path: str, workspace: Path) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SystemExit("target path must be a non-empty string")
    workspace = workspace.resolve()
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit(f"target path must stay under workspace root: {workspace}") from exc
    if relative == Path("."):
        raise SystemExit("target path must name a file below the workspace root")
    return relative.as_posix()


if __name__ == "__main__":
    main()
