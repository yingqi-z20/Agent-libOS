from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import HumanRequestStatus, ProcessStatus, ResourceBudget

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.runtime_assembly import aopen_runtime
    from scripts.workflow_evidence import has_committed_filesystem_write
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from runtime_assembly import aopen_runtime
    from workflow_evidence import has_committed_filesystem_write


_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts
MAX_READ_BYTES = _SCRIPT_DEFAULTS.document_summary_max_read_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn an Agent process that reads a workspace document, writes a "
            "one-sentence summary to a file after the Agent requests write "
            "permission, then tells the human the output filename or failure reason."
        )
    )
    parser.add_argument(
        "document",
        nargs="?",
        default="README.md",
        help="Document path under the current workspace. Absolute paths must stay inside the workspace.",
    )
    parser.add_argument(
        "--db",
        default=_RUNTIME_DEFAULTS.local_store_target,
        help=f"Runtime SQLite database path, or '{_RUNTIME_DEFAULTS.local_store_target}' for in-memory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Summary output path under the current workspace. Defaults to agent_outputs/document_summary_<id>.txt.",
    )
    parser.add_argument("--language", default="Chinese", help="Language for the one-sentence summary.")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=_SCRIPT_DEFAULTS.document_summary_max_bytes,
        help="Maximum document bytes the Agent should read.",
    )
    parser.add_argument(
        "--max-quanta",
        type=int,
        default=_SCRIPT_DEFAULTS.document_summary_max_quanta,
        help="Maximum Agent execution quanta to run.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve per-use prompts. If --permission-policy is omitted, also choose always_allow.",
    )
    parser.add_argument(
        "--permission-policy",
        choices=[
            CapabilityManager.ALWAYS_ALLOW,
            CapabilityManager.ALWAYS_DENY,
            CapabilityManager.ASK_EACH_TIME,
        ],
        default=None,
        help="Automatically answer the Agent's permission-policy request.",
    )
    args = parser.parse_args()

    if args.max_bytes < 1 or args.max_bytes > MAX_READ_BYTES:
        parser.error(f"--max-bytes must be between 1 and {MAX_READ_BYTES}")
    asyncio.run(amain(args))


async def amain(args: argparse.Namespace) -> None:
    runtime = await aopen_runtime(args.db)
    try:
        document_path = _workspace_relative_document(args.document, runtime.workspace_root)
        output_path = _workspace_relative_output(args.output, runtime.workspace_root)
        if output_path == document_path:
            raise SystemExit("Output path must differ from the source document")
        # The process begins with read authority from the coding image but no
        # write authority; it must request a policy before write_text_file works.
        pid = runtime.process.spawn(
            image=_RUNTIME_DEFAULTS.coding_image_id,
            goal=_build_goal(
                document_path=document_path,
                output_path=output_path,
                language=args.language,
                max_bytes=args.max_bytes,
            ),
            resource_budget=ResourceBudget(max_context_materialization_tokens=_context_budget(args.max_bytes)),
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": runtime.filesystem.resource_for(document_path),
                        "rights": ["read"],
                    },
                    {
                        "resource": _RUNTIME_DEFAULTS.default_human_resource,
                        "rights": ["write"],
                    },
                ],
                "approval_policy": {
                    "requestable_capabilities": [
                        {
                            "resource": runtime.filesystem.resource_for(output_path),
                            "rights": ["write"],
                        }
                    ]
                },
                "permitted_effects": ["llm.*", "filesystem.*", "human.*"],
                "metadata": {"provided_by": "llm_summarize_document"},
            },
        )
        permission_policy = args.permission_policy
        if permission_policy is None and args.auto_approve:
            permission_policy = CapabilityManager.ALWAYS_ALLOW
        results = await runtime.arun_until_idle(
            max_quanta=args.max_quanta,
            human_auto_approve=True if args.auto_approve else None,
            human_auto_policy=permission_policy,
        )
        process = runtime.process.get(pid)
        output_exists = (runtime.workspace_root / output_path).exists()
        write_receipt_bound = has_committed_filesystem_write(
            runtime,
            pid,
            results,
            output_path,
        )

        if process.status == ProcessStatus.FAILED:
            raise SystemExit(f"Agent process failed: {process.status_message}")
        if process.status != ProcessStatus.EXITED:
            raise SystemExit(
                f"Agent process did not exit after {args.max_quanta} quanta; status={process.status.value}"
            )
        permission_rejected = _had_permission_rejection(runtime, pid)
        if (not output_exists or not write_receipt_bound) and not permission_rejected:
            raise SystemExit(f"Agent exited without writing summary file: {output_path}")
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def _workspace_relative_document(raw_path: str, workspace: Path) -> str:
    workspace = workspace.resolve()
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit(f"Document must be under workspace root: {workspace}") from exc
    if not resolved.exists():
        raise SystemExit(f"Document does not exist: {relative.as_posix()}")
    if not resolved.is_file():
        raise SystemExit(f"Document path is not a file: {relative.as_posix()}")
    return relative.as_posix()


def _workspace_relative_output(raw_path: str | None, workspace: Path) -> str:
    workspace = workspace.resolve()
    default_path = f"agent_outputs/document_summary_{uuid4().hex[:8]}.txt"
    path = Path(raw_path or default_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit(f"Output path must be under workspace root: {workspace}") from exc
    return relative.as_posix()


def _build_goal(*, document_path: str, output_path: str, language: str, max_bytes: int) -> str:
    output_resource = f"filesystem:{_RUNTIME_DEFAULTS.workspace_namespace}:{output_path}"
    return "\n".join(
        [
            "Read a workspace document, write a summary file, and tell the human the output filename.",
            "",
            "Required tool sequence:",
            f"1. First call read_text_file with path={document_path!r} and max_bytes={max_bytes}.",
            f"2. Then call request_permission for resource={output_resource!r}, rights=['write'], and reason='write the summary file'.",
            f"3. After the permission response is visible, write one {language} sentence to {output_path!r} using write_text_file.",
            f"4. After the write_text_file result is visible and successful, call human_output with exactly this filename: {output_path!r}.",
            "5. If permission is denied or the write cannot be completed, call human_output with one concise sentence explaining why no summary file was written.",
            "6. Finally call process_exit with a compact final payload containing output_path and either written=true or written=false with reason.",
            "",
            "Do not summarize before reading the file. Do not call read_text_file again if a result for this path is already visible. If the read result is truncated, summarize only the visible content. Do not choose a different output path.",
        ]
    )


def _context_budget(max_bytes: int) -> int:
    return min(
        _SCRIPT_DEFAULTS.document_context_max_tokens,
        max(
            _SCRIPT_DEFAULTS.document_context_min_tokens,
            max_bytes + _SCRIPT_DEFAULTS.document_context_slack_tokens,
        ),
    )


def _had_permission_rejection(runtime: Runtime, pid: str) -> bool:
    for request in runtime.human.list(pid):
        if request.status != HumanRequestStatus.REJECTED:
            continue
        if isinstance(request.payload.get("requested_permission"), dict):
            return True
        if isinstance(request.payload.get("requested_once_capability"), dict):
            return True
    return False


if __name__ == "__main__":
    main()
