from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_libos import Runtime  # noqa: E402
from agent_libos.capability.manager import CapabilityManager  # noqa: E402
from agent_libos.config import DEFAULT_CONFIG  # noqa: E402
from agent_libos.llm.client import load_dotenv  # noqa: E402
from agent_libos.models import Capability, CapabilityRight, ProcessStatus  # noqa: E402
from agent_libos.utils.serde import to_jsonable  # noqa: E402
from agent_libos.substrate import LocalResourceProviderSubstrate  # noqa: E402

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.runtime_assembly import aopen_runtime  # noqa: E402
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from runtime_assembly import aopen_runtime  # noqa: E402


_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_LAUNCHER_DEFAULTS = DEFAULT_CONFIG.launcher
_SHELL_DEFAULTS = DEFAULT_CONFIG.shell
PERMISSION_PRESETS = _LAUNCHER_DEFAULTS.permission_presets
SHELL_POLICY_CHOICES = (
    "none",
    _SHELL_DEFAULTS.always_deny_level,
    _SHELL_DEFAULTS.allowlist_auto_else_ask_level,
    _SHELL_DEFAULTS.blocklist_ask_else_auto_level,
    _SHELL_DEFAULTS.always_allow_level,
)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(amain(args))


async def amain(args: argparse.Namespace) -> None:
    _load_env(args)
    workspace = _resolve_workspace(args.workspace)
    runtime = await _aopen_runtime(args, workspace)
    try:
        goal = _load_goal(args, workspace)
        pid = runtime.process.spawn(image=_RUNTIME_DEFAULTS.coding_image_id, goal=goal)
        grants = configure_coding_agent_permissions(runtime, pid, args)
        results: list[Any] = []
        if not args.no_run:
            results = await runtime.arun_until_idle(
                max_quanta=args.max_quanta,
                pids=(pid,),
                human_auto_policy=args.human_auto_policy,
                human_auto_approve=_optional_bool(args.human_auto_approve),
                human_auto_answer=args.human_auto_answer,
            )
        process = runtime.process.get(pid)
        audit_counts = _audit_counts_for_process(runtime.audit.trace(), pid)
        summary = {
            "workspace": str(workspace),
            "database": "local" if args.ephemeral_db else str(_resolve_db_path(args, workspace)),
            "pid": pid,
            "image": _RUNTIME_DEFAULTS.coding_image_id,
            "permission_preset": args.permission_preset,
            "pregranted_capabilities": [_capability_summary(cap) for cap in grants],
            "ran": not args.no_run,
            "process_status": process.status.value,
            "results": to_jsonable(results),
            **audit_counts,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        if args.strict and process.status in {ProcessStatus.FAILED, ProcessStatus.KILLED}:
            raise SystemExit(2)
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Launch {_RUNTIME_DEFAULTS.coding_image_id} against any workspace with preconfigured filesystem permissions."
    )
    parser.add_argument("--goal", help="Goal for the coding agent.")
    parser.add_argument("--goal-file", help="Read the goal from a UTF-8 text file.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace root exposed to the agent. Defaults to the current directory.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "SQLite runtime DB path outside the exposed workspace. Relative "
            "paths resolve from the Host working directory. The default is a "
            "workspace-keyed file in a Host state directory."
        ),
    )
    parser.add_argument(
        "--env-file",
        help="LLM .env file to load before mounting the workspace. Defaults to this Agent libOS checkout's .env.",
    )
    parser.add_argument("--ephemeral-db", action="store_true", help="Use an in-memory runtime DB.")
    parser.add_argument(
        "--permission-preset",
        choices=PERMISSION_PRESETS,
        default=_LAUNCHER_DEFAULTS.default_permission_preset,
        help="read-only grants read only; edit grants read+write workspace; full grants read+write+delete workspace.",
    )
    parser.add_argument("--read-file", action="append", default=[], help="Extra workspace-relative file read grant.")
    parser.add_argument("--write-file", action="append", default=[], help="Extra workspace-relative file write grant.")
    parser.add_argument("--delete-file", action="append", default=[], help="Extra workspace-relative file delete grant.")
    parser.add_argument("--read-dir", action="append", default=[], help="Extra workspace-relative directory read grant.")
    parser.add_argument("--write-dir", action="append", default=[], help="Extra workspace-relative directory write grant.")
    parser.add_argument("--delete-dir", action="append", default=[], help="Extra workspace-relative directory delete grant.")
    parser.add_argument(
        "--shell-policy",
        choices=SHELL_POLICY_CHOICES,
        default=_SHELL_DEFAULTS.default_policy_level,
        help=(
            "Shell execution policy. Default auto-allows configured whitelist commands and asks for the rest. "
            "always_allow is high risk."
        ),
    )
    parser.add_argument(
        "--max-quanta",
        type=int,
        default=_RUNTIME_DEFAULTS.launcher_max_quanta,
        help="Maximum LLM/tool execution quanta.",
    )
    parser.add_argument("--no-run", action="store_true", help="Spawn and pregrant only; do not run the scheduler.")
    parser.add_argument(
        "--human-auto-policy",
        choices=[CapabilityManager.ALWAYS_ALLOW, CapabilityManager.ALWAYS_DENY, CapabilityManager.ASK_EACH_TIME],
        help="Automatically answer request_permission prompts while running.",
    )
    parser.add_argument(
        "--human-auto-approve",
        choices=["yes", "no"],
        help="Automatically approve or reject boolean/per-use human approval prompts while running.",
    )
    parser.add_argument("--human-auto-answer", help="Automatically answer ask_human questions while running.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if the process fails or is killed.")
    return parser


def configure_coding_agent_permissions(
    runtime: Runtime,
    pid: str,
    args: argparse.Namespace,
) -> list[Capability]:
    grants: list[Capability] = []
    grants.append(runtime.filesystem.grant_workspace(pid, [CapabilityRight.READ], issued_by="coding-agent-launcher"))
    if args.permission_preset in {_LAUNCHER_DEFAULTS.edit_preset, _LAUNCHER_DEFAULTS.full_preset}:
        grants.append(runtime.filesystem.grant_workspace(pid, [CapabilityRight.WRITE], issued_by="coding-agent-launcher"))
    if args.permission_preset == _LAUNCHER_DEFAULTS.full_preset:
        grants.append(runtime.filesystem.grant_workspace(pid, [CapabilityRight.DELETE], issued_by="coding-agent-launcher"))
    if args.shell_policy != "none":
        grants.append(runtime.shell.grant_policy(pid, args.shell_policy, issued_by="coding-agent-launcher"))

    grants.extend(
        runtime.filesystem.grant_path_list(
            pid,
            read_files=args.read_file,
            write_files=args.write_file,
            delete_files=args.delete_file,
            read_dirs=args.read_dir,
            write_dirs=args.write_dir,
            delete_dirs=args.delete_dir,
            issued_by="coding-agent-launcher",
        )
    )
    return grants


def _load_env(args: argparse.Namespace) -> None:
    if args.env_file:
        env_path = Path(args.env_file).expanduser().resolve()
    else:
        # Keep the default anchored to the logical checkout root. On macOS,
        # resolving temp/project paths can rewrite /var to /private/var, which
        # is harmless for I/O but makes launcher provenance and tests brittle.
        env_path = PROJECT_ROOT / ".env"
    if args.env_file and not env_path.exists():
        raise SystemExit(f"env file does not exist: {env_path}")
    if env_path.exists():
        load_dotenv(env_path)


async def _aopen_runtime(args: argparse.Namespace, workspace: Path) -> Runtime:
    substrate = LocalResourceProviderSubstrate(workspace)
    target = (
        "local"
        if args.ephemeral_db
        else _resolve_db_path(args, workspace)
    )
    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)
    return await aopen_runtime(target, substrate=substrate)


def _resolve_workspace(value: str) -> Path:
    workspace = Path(value).expanduser().resolve()
    if not workspace.exists():
        raise SystemExit(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise SystemExit(f"workspace is not a directory: {workspace}")
    return workspace


def _resolve_db_path(args: argparse.Namespace, workspace: Path) -> Path:
    if args.db is None:
        selected = _default_db_path(workspace)
    else:
        db_path = Path(args.db).expanduser()
        selected = db_path.resolve()
    if _path_is_within(selected, workspace):
        raise SystemExit(
            "Runtime database must stay outside the model-visible workspace; "
            "use --ephemeral-db or an external --db path"
        )
    return selected


def _default_db_path(workspace: Path) -> Path:
    workspace = workspace.resolve()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:24]
    candidates = (
        Path(tempfile.gettempdir()).resolve()
        / "agent-libos"
        / "coding-runtime"
        / f"{digest}.sqlite",
        Path.home().resolve()
        / ".agent-libos"
        / "coding-runtime"
        / f"{digest}.sqlite",
    )
    for candidate in candidates:
        if not _path_is_within(candidate, workspace):
            return candidate
    raise SystemExit(
        "cannot select a Runtime database outside this workspace; "
        "use --ephemeral-db"
    )


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _load_goal(args: argparse.Namespace, workspace: Path | None = None) -> str:
    if bool(args.goal) == bool(args.goal_file):
        raise SystemExit("provide exactly one of --goal or --goal-file")
    if args.goal_file:
        path = Path(args.goal_file).expanduser()
        if not path.is_absolute() and workspace is not None:
            path = workspace / path
        return path.read_text(encoding="utf-8").strip()
    return str(args.goal).strip()


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "yes"


def _audit_counts_for_process(audit_records: list[Any], pid: str) -> dict[str, int]:
    process_records = [record for record in audit_records if record.actor == pid]
    return {
        "audit_records": len(process_records),
        "audit_records_total": len(audit_records),
        "llm_repair_attempts": sum(1 for record in process_records if record.action == "llm.action_repair_requested"),
    }


def _capability_summary(capability: Capability) -> dict[str, Any]:
    return {
        "cap_id": capability.cap_id,
        "resource": capability.resource,
        "rights": sorted(capability.rights),
        "policy": capability.constraints.get(CapabilityManager.POLICY_KEY, CapabilityManager.ALWAYS_ALLOW),
        "shell_policy": capability.constraints.get(_SHELL_DEFAULTS.policy_capability_key),
    }


if __name__ == "__main__":
    main()
